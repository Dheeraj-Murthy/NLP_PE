"""
Smoke Test for chat conversation context (no database or model needed)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_context import (
    carried_documents,
    cited_documents,
    context_window,
    export_conversation,
    is_follow_up,
    parse_import,
    prompt_history,
    retrieval_query,
    split_for_summary,
)
from chat_store import TITLE_MAX_CHARS, is_valid_session_id, title_from


def test_follow_up_detection():
    print("--- Testing follow-up detection ---")
    follow_ups = [
        "What are its exceptions?",
        "And who decides that?",
        "what about bail?",
        "Why?",
        "Explain the above in simple terms",
    ]
    fresh = [
        "What is the punishment for theft under the Bharatiya Nyaya Sanhita?",
        "Explain the doctrine of basic structure in Indian constitutional law",
    ]
    for m in follow_ups:
        assert is_follow_up(m), m
    for m in fresh:
        assert not is_follow_up(m), m
    print("✓ follow-up detection passed")


def test_retrieval_query():
    print("--- Testing retrieval query for follow-ups ---")
    prior = [
        {"role": "user", "content": "What does Article 21 protect?", "retrieval_query": "What does Article 21 protect?"},
        {"role": "assistant", "content": "Answer: life and liberty"},
    ]
    assert retrieval_query("What are its exceptions?", prior) == "What does Article 21 protect? What are its exceptions?"
    fresh = "What is the punishment for theft under the Bharatiya Nyaya Sanhita?"
    assert retrieval_query(fresh, prior) == fresh
    assert retrieval_query("And that?", []) == "And that?"
    print("✓ retrieval query passed")


def test_prompt_history():
    print("--- Testing prompt history budget ---")
    prior = []
    for i in range(6):
        prior.append({"role": "user", "content": f"question {i} " + "word " * 10})
        prior.append({"role": "assistant", "content": f"Answer: {i}\n\nSources:\n- x", "prompt_text": f"answer {i} " + "word " * 10})
    count = lambda text: len(text.split())
    full = prompt_history(prior, count, 10_000)
    assert len(full) == 12 and all("Sources:" not in m["content"] for m in full)
    small = prompt_history(prior, count, 50)
    assert 0 < len(small) < 12 and small[0]["role"] == "user"
    assert small[-1]["content"].startswith("answer 5")
    assert prompt_history(prior, count, 1) == []
    print("✓ prompt history passed")


def test_windows_and_summary_split():
    print("--- Testing context windows and summary split ---")
    assert context_window("claude-sonnet-5") == 200_000
    assert context_window("gemini-2.5-pro") == 1_000_000
    assert context_window("Qwen/Qwen2.5-7B-Instruct-1M") == context_window(None)
    msgs = []
    for i in range(10):
        msgs.append({"role": "user", "content": "q " * 50, "message_id": 2 * i + 1})
        msgs.append({"role": "assistant", "content": "a " * 50, "message_id": 2 * i + 2})
    count = lambda t: len(t.split())
    old, keep = split_for_summary(msgs, count, 500)
    assert old and keep and old + keep == msgs and keep[0]["role"] == "user"
    print("✓ windows and summary split passed")


def test_carried_documents_and_export():
    print("--- Testing carried documents and export/import ---")
    chunks = [{"doc_type": "judgment", "chunk_id": i, "judgment_id": 10 + i, "text": f"t{i}", "case": f"C{i}"} for i in range(3)]
    docs = cited_documents("held [2] and [3] and again [2]", chunks)
    assert [d["chunk_id"] for d in docs] == [1, 2]
    prior = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "details": {"cited_documents": docs}}]
    carried = carried_documents(prior, exclude=[chunks[1]])
    assert [d["chunk_id"] for d in carried] == [2] and carried[0]["carried"]
    from datetime import datetime
    messages = [dict(m, message_id=i + 1, created_at=datetime.now()) for i, m in enumerate(prior)]
    exported = export_conversation("abc", messages, {"summary": "- s", "upto": 1})
    back, summary, covers = parse_import(exported)
    assert len(back) == 2 and summary == "- s" and covers == 1
    for bad in [None, {"format": "x"}, dict(exported, messages=[{"role": "system", "content": "x"}])]:
        try:
            parse_import(bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    print("✓ carried documents and export/import passed")


def test_session_ids():
    assert is_valid_session_id("default")
    assert is_valid_session_id("0a193011d45e43d4a8b2faa6d9d3e72d")
    for bad in ["", "new", "x; drop table", "0A193011D45E43D4A8B2FAA6D9D3E72D", None]:
        assert not is_valid_session_id(bad), bad
    print("✓ session id validation passed")


def test_titles():
    assert title_from("  What is\n bail?  ") == "What is bail?"
    long = title_from("word " * 100)
    assert len(long) <= TITLE_MAX_CHARS and long.endswith("…") and not long[:-1].endswith(" ")
    assert title_from("") == "New chat"
    print("✓ chat titles passed")


if __name__ == "__main__":
    test_follow_up_detection()
    test_retrieval_query()
    test_prompt_history()
    test_windows_and_summary_split()
    test_carried_documents_and_export()
    test_session_ids()
    test_titles()
    print("\nAll chat context smoke tests passed.")
