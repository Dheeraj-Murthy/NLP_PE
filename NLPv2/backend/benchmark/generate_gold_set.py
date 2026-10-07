#!/usr/bin/env python3
"""Generate a synthetic gold query set for retrieval-quality benchmarking.

For each sampled judgment_chunks row, asks an LLM to write one
natural-language question that chunk specifically answers. The chunk's own
id becomes the ground-truth target that retrieval_eval.py checks for.

Uses the local Qwen model by default — same as the rest of this repo,
nothing leaves the machine. Pass --model claude-*/gpt-*/gemini-* plus
--external-ok to use an external API instead (same opt-in gate
rag_pipeline.py's _resolve_backend uses for those backends).

Two quality guards, since a gold set this benchmark trusts blindly is worse
than no benchmark at all:
  - Garbled-content filter: skips chunks that are mostly non-ASCII, a sign
    of pdftotext mis-extraction (mojibake), not real judgment text.
  - Question/content overlap check: rejects a generated question that
    shares almost no vocabulary with its source chunk (the LLM drifting
    into a vague or only tangentially related question) and tries a
    different chunk from the same section instead.
  - content_hash + corpus snapshot counts are recorded so retrieval_eval.py
    can detect a gold set that's gone stale — e.g. after a schema reset and
    full re-ingest reassigns every chunk_id from scratch, this file's
    source_chunk_id references would otherwise silently point at different
    (or missing) content and corrupt every metric without any error.

Usage:
    cd NLPv2/backend
    python benchmark/generate_gold_set.py --n 80 --seed 42
    python benchmark/generate_gold_set.py --n 80 --model claude-sonnet-5 --external-ok
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg2

# Add backend/ to the path so the dotted `llm.*` subpackage imports below
# resolve whether this script is run directly or as a module — same shim
# as backend/tests/smoke_test_citation_graph.py.
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

from llm.base import LLMBackend, PrivacyGateError  # noqa: E402

SECTIONS = ["facts", "issues", "arguments", "ratio", "judgment"]
MAX_CHUNKS_PER_JUDGMENT = 2
MIN_CHUNK_CHARS = 200  # skip near-empty chunks that can't support a real question
MAX_NON_ASCII_RATIO = 0.2  # skip chunks with garbled/mis-encoded PDF extraction
MIN_QUESTION_OVERLAP = 0.25  # min fraction of non-trivial question words found in the chunk
MAX_ATTEMPTS_PER_SECTION_MULTIPLIER = 4  # how many extra candidates to try per section before giving up

# Near-duplicate detection: a gold query's chunk_id is only ONE of
# possibly several chunks that equally answer it — e.g. formulaic
# disposition language ("the appeal is dismissed with costs...") repeats
# near-verbatim across thousands of unrelated judgments. Scoring only the
# literal source chunk as "correct" then counts a retrieval that surfaces
# an equally-valid near-duplicate as a miss, which isn't a retrieval
# defect — it's a single-label gold set being unfair to a multi-answer
# question. This threshold/limit pair controls how we detect that and
# build the wider accepted-answer set, via a cosine self-join on the
# chunk's own existing embedding (nothing re-encoded).
NEAR_DUP_SIMILARITY_THRESHOLD = 0.92
NEAR_DUP_FETCH_LIMIT = 50  # comfortably above stage-1 candidate_k=30 — no point tracking duplicates retrieval could never surface anyway

GOLD_GEN_SYSTEM_PROMPT = (
    "You are generating evaluation data for a legal search system. Given an "
    "excerpt from the '{section}' portion of an Indian court judgment, write "
    "ONE natural-language question a lawyer might ask that this excerpt "
    "specifically and directly answers, using the same names, terms, and "
    "specifics that appear in the excerpt. Return only the question, nothing else."
)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "does", "did", "do", "what", "which", "who", "whom", "how",
    "why", "when", "where", "under", "according", "that", "this", "these",
    "those", "it", "its", "be", "been", "being", "can", "could", "would",
    "should", "will", "shall", "with", "by", "as", "at", "from", "not", "any",
}

DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gold_set.json")


def _default_db_connection_string() -> str:
    """Build the default DSN from env vars — same convention as
    retrieval/retriever.py's private helper (each module builds its own
    connection; there's no shared db.py in this repo)."""
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5433")
    dbname = os.environ.get("DB_NAME", "legal_rag")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "postgres")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def _content_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _is_garbled(content: str) -> bool:
    """Flags chunks with a high proportion of non-ASCII characters — this
    corpus is all-English Supreme Court judgments, so a chunk heavy in
    non-Latin codepoints is a pdftotext font-encoding mis-extraction
    (mojibake), not real content, and would waste a gold query on
    something not even a human can read.

    Threshold is ord(ch) > 127 (true non-ASCII), not some higher cutoff —
    Bengali/Devanagari/Tamil etc. all live at U+0900-U+0FFF (2304-4095),
    well under a cutoff like 0x2000 (8192) that only catches CJK and
    similarly high-codepoint scripts. A higher cutoff would silently let
    exactly the Indic-script mojibake this corpus actually produces
    through the filter."""
    if not content:
        return True
    non_ascii = sum(1 for ch in content if ord(ch) > 127)
    return (non_ascii / len(content)) > MAX_NON_ASCII_RATIO


def _question_overlap(question: str, content: str) -> float:
    """Fraction of the question's non-trivial words that actually appear in
    the source chunk. Mirrors post_processor.py's lexical-overlap heuristic
    (answer_words & context_words), applied here to catch a generated
    question that's too vague or has drifted from what the excerpt says."""
    q_words = {w for w in re.findall(r"[a-z]+", question.lower()) if w not in _STOPWORDS and len(w) > 2}
    if not q_words:
        return 0.0
    c_words = {w for w in re.findall(r"[a-z]+", content.lower())}
    return len(q_words & c_words) / len(q_words)


def _fetch_all_chunks(cur) -> List[Dict[str, Any]]:
    cur.execute("SELECT chunk_id, judgment_id, section, content FROM judgment_chunks WHERE length(content) > %s", (MIN_CHUNK_CHARS,))
    return [
        {"chunk_id": r[0], "judgment_id": r[1], "section": r[2], "content": r[3]}
        for r in cur.fetchall()
        if not _is_garbled(r[3])
    ]


def _near_duplicate_chunk_ids(
    cur, chunk_id: int, threshold: float = NEAR_DUP_SIMILARITY_THRESHOLD, limit: int = NEAR_DUP_FETCH_LIMIT
) -> List[int]:
    """Other chunks whose embedding is near-identical to chunk_id's own
    embedding (self-join on judgment_embeddings, reusing the same
    `embedding <=> embedding` cosine-distance pattern retriever.py uses
    against a query embedding — here both sides are corpus chunks). These
    are content-equivalent answers to whatever question chunk_id's content
    generated, not just topically related chunks, so the threshold is set
    high (0.92) on purpose. Capped at `limit` since nothing beyond the
    stage-1 candidate pool size could ever be retrieved anyway."""
    cur.execute(
        """
        SELECT je2.chunk_id
        FROM judgment_embeddings je1
        JOIN judgment_embeddings je2 ON je2.chunk_id != je1.chunk_id
        WHERE je1.chunk_id = %s
          AND 1 - (je2.embedding <=> je1.embedding) >= %s
        ORDER BY je2.embedding <=> je1.embedding
        LIMIT %s
        """,
        (chunk_id, threshold, limit),
    )
    return [r[0] for r in cur.fetchall()]


def _corpus_counts(cur) -> Dict[str, int]:
    cur.execute("SELECT count(*) FROM judgments")
    judgment_count = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM judgment_chunks")
    chunk_count = cur.fetchone()[0]
    return {"judgment_count": judgment_count, "chunk_count": chunk_count}


def _candidate_pool(cur, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    """All eligible chunks, bucketed and shuffled by section. Each bucket is
    a queue of candidates to draw from — more than the final quota, so
    build_gold_set can skip a low-quality question and move to the next
    candidate without re-querying the database."""
    rng = random.Random(seed)
    all_chunks = _fetch_all_chunks(cur)
    by_section: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in all_chunks:
        by_section[chunk["section"]].append(chunk)
    for bucket in by_section.values():
        rng.shuffle(bucket)
    return by_section


def _resolve_backend(model: Optional[str], external_ok: bool, api_key: Optional[str]) -> LLMBackend:
    """Same prefix-based dispatch as rag_pipeline.LegalRAGPipeline._resolve_backend,
    standalone (no pipeline/DB-connected retriever needed to generate questions).
    None/"qwen" -> local Qwen (default, nothing leaves the machine); claude-*/gpt-*/
    o1*/o3*/gemini-* -> external API, gated behind --external-ok."""
    if not model or model == "qwen":
        from llm.qwen_backend import QwenBackend

        return QwenBackend()

    if model.startswith("claude-"):
        from llm.anthropic_backend import AnthropicBackend

        backend_cls = AnthropicBackend
    elif model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        from llm.openai_backend import OpenAIBackend

        backend_cls = OpenAIBackend
    elif model.startswith("gemini-"):
        from llm.gemini_backend import GeminiBackend

        backend_cls = GeminiBackend
    else:
        raise ValueError(f"Unknown model: {model}")

    if backend_cls.requires_external_ok and not external_ok:
        raise PrivacyGateError(
            f"Model '{model}' is an external API backend. Pass --external-ok to confirm "
            f"sending judgment_chunks content to this provider."
        )
    return backend_cls(model_id=model, api_key=api_key) if api_key else backend_cls(model_id=model)


def generate_question_for_chunk(backend: LLMBackend, chunk: Dict[str, Any]) -> str:
    gen = backend.generate(
        context_block=chunk["content"],
        user_query="Write the question now.",
        system_prompt=GOLD_GEN_SYSTEM_PROMPT.format(section=chunk["section"]),
    )
    return gen.text.strip()


def build_gold_record(
    chunk: Dict[str, Any], question: str, model_id: str, near_dup_ids: List[int]
) -> Dict[str, Any]:
    return {
        "query": question,
        "source_chunk_id": chunk["chunk_id"],
        "source_judgment_id": chunk["judgment_id"],
        "section": chunk["section"],
        "content_hash": _content_hash(chunk["content"]),
        "generator_model": model_id,
        # Equivalence class retrieval_eval.py scores against: the literal
        # source chunk plus any near-duplicate found above. A match on ANY
        # of these counts as a hit — see metrics.rank_of.
        "acceptable_chunk_ids": [chunk["chunk_id"]] + near_dup_ids,
        # How many OTHER chunks are near-identical to this one. High
        # crowd_size (common for formulaic `judgment`-section disposition
        # text) flags a query whose source chunk was never uniquely
        # identifiable in the first place — a gold-set artifact, not a
        # retrieval weakness. retrieval_eval.py reports recall split by
        # this so the two don't get conflated.
        "crowd_size": len(near_dup_ids),
    }


def build_gold_set(
    n: int,
    seed: int,
    model: Optional[str],
    external_ok: bool,
    api_key: Optional[str],
    near_dup_threshold: float = NEAR_DUP_SIMILARITY_THRESHOLD,
    near_dup_limit: int = NEAR_DUP_FETCH_LIMIT,
) -> Dict[str, Any]:
    conn = psycopg2.connect(_default_db_connection_string())
    try:
        cur = conn.cursor()
        try:
            by_section = _candidate_pool(cur, seed)
            corpus_counts = _corpus_counts(cur)
        finally:
            cur.close()
    finally:
        conn.close()

    backend = _resolve_backend(model, external_ok, api_key)
    per_section = max(n // len(SECTIONS), 1)
    max_attempts_per_section = per_section * MAX_ATTEMPTS_PER_SECTION_MULTIPLIER
    judgment_counts: Dict[int, int] = defaultdict(int)

    records: List[Dict[str, Any]] = []
    skipped_overlap = 0
    crowded_count = 0

    # Separate connection for near-duplicate lookups, opened only once an
    # accepted question needs one — kept open across the loop rather than
    # reopened per record, closed in the finally below.
    dup_conn = psycopg2.connect(_default_db_connection_string())
    try:
        dup_cur = dup_conn.cursor()
        try:
            for section in SECTIONS:
                accepted = 0
                attempts = 0
                for chunk in by_section.get(section, []):
                    if accepted >= per_section or len(records) >= n or attempts >= max_attempts_per_section:
                        break
                    if judgment_counts[chunk["judgment_id"]] >= MAX_CHUNKS_PER_JUDGMENT:
                        continue
                    attempts += 1

                    question = generate_question_for_chunk(backend, chunk)
                    overlap = _question_overlap(question, chunk["content"])
                    if overlap < MIN_QUESTION_OVERLAP:
                        skipped_overlap += 1
                        print(
                            f"  [skip] chunk {chunk['chunk_id']} ({section}): overlap={overlap:.2f} "
                            f"< {MIN_QUESTION_OVERLAP} — {question}"
                        )
                        continue

                    near_dup_ids = _near_duplicate_chunk_ids(
                        dup_cur, chunk["chunk_id"], threshold=near_dup_threshold, limit=near_dup_limit
                    )
                    if near_dup_ids:
                        crowded_count += 1

                    judgment_counts[chunk["judgment_id"]] += 1
                    accepted += 1
                    records.append(build_gold_record(chunk, question, backend.model_id, near_dup_ids))
                    print(
                        f"[{len(records)}/{n}] chunk {chunk['chunk_id']} ({section}, overlap={overlap:.2f}, "
                        f"crowd_size={len(near_dup_ids)}): {question}"
                    )
        finally:
            dup_cur.close()
    finally:
        dup_conn.close()

    random.Random(seed).shuffle(records)
    print(
        f"\n{len(records)} accepted, {skipped_overlap} skipped for low question/content overlap, "
        f"{crowded_count} with 1+ near-duplicate chunk (crowd_size > 0)"
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_model": backend.model_id,
        "corpus_judgment_count": corpus_counts["judgment_count"],
        "corpus_chunk_count": corpus_counts["chunk_count"],
        "near_dup_threshold": near_dup_threshold,
        "near_dup_limit": near_dup_limit,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic gold query set from real judgment_chunks rows."
    )
    parser.add_argument(
        "--n", type=int, default=80, help="Number of gold queries to generate (default: 80)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for chunk sampling (default: 42). "
        "Note: this only makes chunk SAMPLING reproducible, not the LLM-generated question "
        "wording, which can vary across runs even for the same chunk."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model id used to write questions: omit for local Qwen (default), or "
        "claude-*/gpt-*/gemini-* for an external backend",
    )
    parser.add_argument(
        "--external-ok", action="store_true",
        help="Confirm sending judgment_chunks content to an external model API "
        "(required with --model claude-*/gpt-*/gemini-*)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for an external model, overriding ANTHROPIC_API_KEY/OPENAI_API_KEY/"
        "GEMINI_API_KEY from .env",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT, help="Output JSON path"
    )
    parser.add_argument(
        "--near-dup-threshold", type=float, default=NEAR_DUP_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity above which another chunk counts as a near-duplicate "
        f"answer to the same gold question (default: {NEAR_DUP_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--near-dup-limit", type=int, default=NEAR_DUP_FETCH_LIMIT,
        help=f"Max near-duplicate chunks to record per gold query (default: {NEAR_DUP_FETCH_LIMIT})",
    )
    args = parser.parse_args()

    gold_set = build_gold_set(
        args.n, args.seed, args.model, args.external_ok, args.api_key,
        near_dup_threshold=args.near_dup_threshold, near_dup_limit=args.near_dup_limit,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(gold_set, f, indent=2)

    print(f"\nWrote {len(gold_set['records'])} gold records to {args.output}")


if __name__ == "__main__":
    main()
