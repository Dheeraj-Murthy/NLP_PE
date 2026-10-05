"""
Conversation context for chat, modelled on how a long chat works in the
Claude app: the model sees the whole conversation as long as it fits the
model's context window; when it doesn't, the oldest turns are condensed into
a running summary (made by the model answering) while recent turns stay word
for word. Documents an answer cited stay attached to the conversation and are
shown to the model again on later turns. History is stored model-neutral, so
switching models mid-chat needs nothing from the user — a smaller window just
means more of the conversation is summarised.

Retrieval only sees one query, so a follow-up such as "what are its
exceptions?" would search for "its exceptions". When a message looks like a
follow-up, the previous question's retrieval query is prepended, so a chain
of follow-ups stays on the topic it started with. Detection is a cheap
heuristic (referring words, a connective opener, or a very short message);
the query actually used is returned in debug info.
"""

import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Context windows
# --------------------------------------------------------------------------

# The local Qwen2.5-7B-Instruct-1M accepts up to 1M tokens, but GPU memory is
# the real limit on the server, so its window is a setting.
QWEN_CONTEXT_TOKENS = int(os.getenv("QWEN_CONTEXT_TOKENS", "32768"))

# (model id prefix, context window in tokens); first match wins.
MODEL_CONTEXT_TOKENS: List[Tuple[str, int]] = [
    ("claude-", 200_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-4o", 128_000),
    ("gpt-", 128_000),
    ("o1", 200_000),
    ("o3", 200_000),
    ("gemini-", 1_000_000),
]
UNKNOWN_MODEL_CONTEXT_TOKENS = 32_768


def context_window(model_id: Optional[str]) -> int:
    name = (model_id or "").lower()
    if not name or "qwen" in name:
        return QWEN_CONTEXT_TOKENS
    for prefix, tokens in MODEL_CONTEXT_TOKENS:
        if name.startswith(prefix):
            return tokens
    return UNKNOWN_MODEL_CONTEXT_TOKENS


def history_budget(window: int, fixed_tokens: int, max_new_tokens: int) -> int:
    """Tokens left for conversation history once the system prompt, the
    retrieved documents, the question and the answer are accounted for,
    with a safety margin (token counts for API models are estimates)."""
    margin = max(512, window // 20)
    return max(0, window - fixed_tokens - max_new_tokens - margin)

# Words that point back at something said earlier.
REFERRING_WORDS = frozenset(
    "it its itself this that these those they them their theirs he him his she her "
    "above aforesaid same such former latter said there".split()
)
FOLLOW_UP_OPENERS = (
    "and ", "also ", "but ", "so ", "then ", "what about", "how about",
    "what if", "why ", "why?", "how come", "in that case", "tell me more", "explain more",
)
SHORT_MESSAGE_WORDS = 4
# Cap on the carried-over topic, so long chains don't grow the query forever.
CARRIED_QUERY_WORDS = 40


def is_follow_up(message: str) -> bool:
    text = message.strip().lower()
    words = re.findall(r"[a-z']+", text)
    if not words:
        return False
    if len(words) <= SHORT_MESSAGE_WORDS:
        return True
    if text.startswith(FOLLOW_UP_OPENERS):
        return True
    return any(w in REFERRING_WORDS for w in words)


def retrieval_query(message: str, prior: List[Dict[str, Any]]) -> str:
    """The query to retrieve with: the message itself, or for a follow-up,
    the previous question's retrieval query followed by the message."""
    if not is_follow_up(message):
        return message
    previous: Optional[str] = None
    for msg in reversed(prior):
        if msg["role"] == "user":
            previous = msg.get("retrieval_query") or msg["content"]
            break
    if not previous:
        return message
    carried = " ".join(previous.split()[-CARRIED_QUERY_WORDS:])
    return f"{carried} {message}"


def _message_tokens(msg: Dict[str, Any], count_tokens: Callable[[str], int]) -> int:
    return count_tokens(msg.get("prompt_text") or msg["content"]) + 4  # role / separator overhead


def split_for_summary(
    messages: List[Dict[str, Any]],
    count_tokens: Callable[[str], int],
    budget_tokens: int,
    keep_share: float = 0.6,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split not-yet-summarised messages into (to_summarise, keep_verbatim):
    the newest turns that fit keep_share of the budget stay word for word,
    starting on a user turn; everything older gets summarised."""
    keep_budget = int(budget_tokens * keep_share)
    used = 0
    cut = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        cost = _message_tokens(messages[i], count_tokens)
        if used + cost > keep_budget:
            break
        used += cost
        cut = i
    while cut < len(messages) and messages[cut]["role"] != "user":
        cut += 1
    return messages[:cut], messages[cut:]


def fits(messages: List[Dict[str, Any]], count_tokens: Callable[[str], int], budget_tokens: int,
         summary: Optional[str] = None) -> bool:
    used = sum(_message_tokens(m, count_tokens) for m in messages)
    if summary:
        used += count_tokens(summary) + 20
    return used <= budget_tokens


SUMMARY_SYSTEM_PROMPT = (
    "You keep the memory of a long legal research conversation between a user and an "
    "Indian legal research assistant, so it can continue after older messages are removed. "
    "Write a compact summary in bullet points. Keep: what the user is trying to find out "
    "and any facts they gave; every conclusion the assistant reached; every case, "
    "statute section and constitutional article mentioned, each with what it was cited "
    "for; and questions still open. Leave out pleasantries and repetition. Never invent "
    "anything that is not in the conversation."
)


def summary_request(previous_summary: Optional[str], messages: List[Dict[str, Any]]) -> str:
    """The text asking the model to fold `messages` into the running summary."""
    parts = []
    if previous_summary:
        parts.append(f"Summary so far:\n{previous_summary}\n")
    parts.append("Conversation to add to the summary:")
    for msg in messages:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        parts.append(f"{speaker}: {msg.get('prompt_text') or msg['content']}")
    parts.append(
        "\nWrite the updated summary of the whole conversation so far as bullet points."
    )
    return "\n".join(parts)


def batches_for_summary(
    messages: List[Dict[str, Any]], count_tokens: Callable[[str], int], max_tokens: int
) -> List[List[Dict[str, Any]]]:
    """Consecutive groups of messages small enough to summarise in one model
    call (a single oversized message gets a group of its own)."""
    batches: List[List[Dict[str, Any]]] = [[]]
    used = 0
    for msg in messages:
        cost = _message_tokens(msg, count_tokens)
        if batches[-1] and used + cost > max_tokens:
            batches.append([])
            used = 0
        batches[-1].append(msg)
        used += cost
    return [b for b in batches if b]


def summary_system_note(summary: str) -> str:
    """Appended to the system prompt when older turns have been summarised."""
    return (
        "\n\nThe start of this conversation has been summarised to fit your context. "
        "Use it as background for the user's follow-up questions:\n" + summary
    )


# --------------------------------------------------------------------------
# Documents cited earlier in the conversation
# --------------------------------------------------------------------------

CITATION_MARK = re.compile(r"\[(\d+)\]")
DOCUMENT_FIELDS = (
    "doc_type", "chunk_id", "judgment_id", "text", "case", "court", "year", "para",
    "section", "statute_name", "section_number",
)


def cited_documents(answer: str, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The retrieved chunks an answer actually cited ([n] marks), trimmed to
    the fields needed to show them to the model again."""
    out, seen = [], set()
    for mark in CITATION_MARK.findall(answer or ""):
        idx = int(mark) - 1
        if 0 <= idx < len(retrieved_chunks) and idx not in seen:
            seen.add(idx)
            chunk = retrieved_chunks[idx]
            out.append({k: chunk[k] for k in DOCUMENT_FIELDS if k in chunk})
    return out


def document_key(chunk: Dict[str, Any]) -> Tuple[str, Any]:
    return (chunk.get("doc_type", "judgment"), chunk.get("chunk_id"))


def carried_documents(
    prior: List[Dict[str, Any]],
    exclude: List[Dict[str, Any]],
    recent_answers: int = 3,
    max_documents: int = 6,
) -> List[Dict[str, Any]]:
    """Documents cited in the last few answers that this turn's retrieval
    didn't already return, newest answer first."""
    taken = {document_key(c) for c in exclude}
    out: List[Dict[str, Any]] = []
    answers = [m for m in prior if m["role"] == "assistant"][-recent_answers:]
    for msg in reversed(answers):
        for doc in (msg.get("details") or {}).get("cited_documents", []):
            key = document_key(doc)
            if key in taken:
                continue
            taken.add(key)
            out.append({**doc, "similarity": doc.get("similarity", 0.0), "carried": True})
            if len(out) >= max_documents:
                return out
    return out


def prompt_history(
    prior: List[Dict[str, Any]],
    count_tokens: Callable[[str], int],
    budget_tokens: int,
) -> List[Dict[str, str]]:
    """As much of the conversation as fits the token budget, newest turns
    first, returned oldest first as {"role", "content"}. Assistant turns use
    their plain answer (without the appended source list)."""
    selected: List[Dict[str, str]] = []
    used = 0
    for msg in reversed(prior):
        content = msg.get("prompt_text") or msg["content"]
        cost = count_tokens(content) + 4  # role / separator overhead
        if used + cost > budget_tokens:
            break
        selected.append({"role": msg["role"], "content": content})
        used += cost
    selected.reverse()
    # Start on a user turn so the model never sees an answer without its question.
    while selected and selected[0]["role"] != "user":
        selected.pop(0)
    return selected


def display_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Messages as the API returns them: role, content, timestamp, and for
    answers the details needed to show them again (citations, sources,
    confidence, model)."""
    out = []
    for msg in messages:
        created = msg.get("created_at")
        item = {
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": created.isoformat() if hasattr(created, "isoformat") else created,
        }
        if msg["role"] == "assistant":
            details = msg.get("details") or {}
            item["citations"] = details.get("citations", [])
            item["confidence"] = details.get("confidence", 0.0)
            item.update(
                {k: v for k, v in details.items() if k not in item and k != "cited_documents"}
            )
        out.append(item)
    return out


# --------------------------------------------------------------------------
# Export / import
# --------------------------------------------------------------------------

EXPORT_FORMAT = "legal-rag-chat"
EXPORT_VERSION = 1
MAX_IMPORT_MESSAGES = 5000


def export_conversation(
    session_id: str, messages: List[Dict[str, Any]], summary: Dict[str, Any]
) -> Dict[str, Any]:
    """Everything needed to continue the conversation elsewhere: messages
    (with cited documents), the running summary and how many messages it
    covers."""
    covered = sum(1 for m in messages if m.get("message_id", 0) <= (summary.get("upto") or 0))
    return {
        "format": EXPORT_FORMAT,
        "version": EXPORT_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "summary": summary.get("summary"),
        "summary_covers_messages": covered if summary.get("summary") else 0,
        "messages": [
            {
                "role": m["role"],
                "content": m["content"],
                "prompt_text": m.get("prompt_text"),
                "retrieval_query": m.get("retrieval_query"),
                "details": m.get("details") or {},
                "created_at": m["created_at"].isoformat()
                if hasattr(m.get("created_at"), "isoformat") else m.get("created_at"),
            }
            for m in messages
        ],
    }


def export_markdown(messages: List[Dict[str, Any]]) -> str:
    """A readable transcript with each answer's citations."""
    lines = ["# Legal research conversation", ""]
    for m in messages:
        if m["role"] == "user":
            lines += [f"**You:** {m['content']}", ""]
            continue
        details = m.get("details") or {}
        model = details.get("model_id")
        lines += [f"**Assistant{f' ({model})' if model else ''}:**", "", m["content"], ""]
        citations = details.get("citations") or []
        if citations and "Sources:" not in m["content"]:
            lines += ["Citations:"] + [f"- {c}" for c in citations] + [""]
    return "\n".join(lines)


def parse_import(data: Any) -> Tuple[List[Dict[str, Any]], Optional[str], int]:
    """Validate an exported conversation; returns (messages, summary,
    number of leading messages the summary covers). Raises ValueError."""
    if not isinstance(data, dict) or data.get("format") != EXPORT_FORMAT:
        raise ValueError("Not a conversation exported from this app")
    if data.get("version") != EXPORT_VERSION:
        raise ValueError(f"Unsupported export version: {data.get('version')}")
    raw = data.get("messages")
    if not isinstance(raw, list) or not raw:
        raise ValueError("The export has no messages")
    if len(raw) > MAX_IMPORT_MESSAGES:
        raise ValueError(f"Too many messages (limit {MAX_IMPORT_MESSAGES})")
    messages = []
    for m in raw:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant") \
                or not isinstance(m.get("content"), str):
            raise ValueError("Each message needs a role (user/assistant) and text content")
        details = m.get("details") if isinstance(m.get("details"), dict) else {}
        messages.append(
            {
                "role": m["role"],
                "content": m["content"],
                "prompt_text": m.get("prompt_text") if isinstance(m.get("prompt_text"), str) else None,
                "retrieval_query": m.get("retrieval_query") if isinstance(m.get("retrieval_query"), str) else None,
                "details": details,
            }
        )
    summary = data.get("summary") if isinstance(data.get("summary"), str) else None
    covers = data.get("summary_covers_messages") or 0
    if not isinstance(covers, int) or not 0 <= covers <= len(messages):
        raise ValueError("Invalid summary_covers_messages")
    return messages, summary, covers if summary else 0
