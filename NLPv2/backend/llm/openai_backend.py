import concurrent.futures
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI

from llm.base import GenerationResult, LLMBackend

DEFAULT_OPENAI_MODEL = "gpt-4o"


class OpenAIBackend(LLMBackend):
    requires_external_ok = True

    def __init__(
        self,
        model_id: str = DEFAULT_OPENAI_MODEL,
        api_key: Optional[str] = None,
        max_new_tokens: int = 1024,
    ):
        load_dotenv()
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=key)
        self._model_id = model_id
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        context_block: str,
        user_query: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(
            {"role": m["role"], "content": m["content"]} for m in (conversation_history or [])
        )
        messages.append(
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion:\n{user_query}"}
        )

        resp = self.client.chat.completions.create(
            model=self._model_id,
            max_tokens=max_new_tokens or self.max_new_tokens,
            messages=messages,
        )

        choice = resp.choices[0]

        return GenerationResult(
            text=(choice.message.content or "").strip(),
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            model_id=self._model_id,
            model_version=resp.model,
            raw={"finish_reason": choice.finish_reason},
        )

    def count_tokens(self, text: str) -> int:
        # No local tokenizer bundled for OpenAI models; word-count heuristic
        # is good enough for context-budget estimation (not billing-accurate).
        return int(len(text.split()) * 1.3)

    @property
    def model_id(self) -> str:
        return self._model_id


PROBE_TIMEOUT_S = 10  # per-call cap, mirrors gemini_backend.PROBE_TIMEOUT_MS

# rag_pipeline._resolve_backend only routes ids starting with "gpt-"/"o1"/"o3"
# to this backend — anything else listed here would be a dead end if picked.
ROUTABLE_PREFIXES = ("gpt-", "o1", "o3")

# Catalog entries that match the routable prefixes by name but aren't plain
# chat-completions models (embeddings, audio, image, moderation, realtime
# variants) — excluded the same way gemini_backend filters out its
# non-chat-capable "gemini-"-prefixed entries.
UNSUITABLE_NAME_MARKERS = (
    "embedding", "audio", "realtime", "transcribe", "tts", "image", "moderation",
)


def list_available_models(api_key: str) -> List[str]:
    """Models this key can actually use for a plain chat-completions call —
    mirrors gemini_backend.list_available_models (see that docstring for why
    a live probe, not just the catalog listing, is needed: the catalog
    doesn't reflect per-key access or per-model quota). Raises
    openai.APIStatusError (e.g. invalid_api_key) if the key itself is bad."""
    client = OpenAI(api_key=api_key, timeout=PROBE_TIMEOUT_S)
    candidates = sorted(
        {
            m.id
            for m in client.models.list()
            if m.id.startswith(ROUTABLE_PREFIXES)
            and not any(marker in m.id for marker in UNSUITABLE_NAME_MARKERS)
        }
    )

    def _is_usable(name: str) -> bool:
        try:
            client.chat.completions.create(
                model=name, max_tokens=1, messages=[{"role": "user", "content": "hi"}]
            )
            return True
        except APIStatusError as e:
            if e.status_code in (400, 403, 404):
                return False  # key/model can't serve this call shape at all
            # Anything else (e.g. a transient 429/5xx) means it's reachable —
            # don't punish that.
            return True
        except Exception:
            # Timeout or other transient failure — inconclusive, not a
            # confirmed denial, so don't drop a model over it.
            return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        usable = list(pool.map(_is_usable, candidates))

    return [name for name, ok in zip(candidates, usable) if ok]
