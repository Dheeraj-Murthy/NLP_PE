#!/usr/bin/env python3
"""Generate a synthetic gold query set for retrieval-quality benchmarking.

For each sampled judgment_chunks row, asks an LLM to write one
natural-language question that chunk specifically answers. The chunk's own
id becomes the ground-truth target that retrieval_eval.py checks for.

This sends real judgment_chunks content to Anthropic's API, so --external-ok
is required, the same opt-in gate rag_pipeline.py uses for external models.

Usage:
    cd NLPv2/backend
    python benchmark/generate_gold_set.py --n 80 --seed 42 --external-ok
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import psycopg2

# Add backend/ to the path so the dotted `llm.*` subpackage import below
# resolves whether this script is run directly or as a module — same shim
# as backend/tests/smoke_test_citation_graph.py.
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

from llm.anthropic_backend import AnthropicBackend  # noqa: E402

SECTIONS = ["facts", "issues", "arguments", "ratio", "judgment"]
MAX_CHUNKS_PER_JUDGMENT = 2
MIN_CHUNK_CHARS = 200  # skip near-empty chunks that can't support a real question

GOLD_GEN_SYSTEM_PROMPT = (
    "You are generating evaluation data for a legal search system. Given an "
    "excerpt from the '{section}' portion of an Indian court judgment, write "
    "ONE natural-language question a lawyer might ask that this excerpt "
    "specifically and directly answers. Return only the question, nothing else."
)

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


def _fetch_all_chunks(cur) -> List[Dict[str, Any]]:
    cur.execute(
        "SELECT chunk_id, judgment_id, section, content "
        "FROM judgment_chunks WHERE length(content) > %s",
        (MIN_CHUNK_CHARS,),
    )
    return [
        {"chunk_id": r[0], "judgment_id": r[1], "section": r[2], "content": r[3]}
        for r in cur.fetchall()
    ]


def sample_chunks(cur, n: int, seed: int) -> List[Dict[str, Any]]:
    """Stratified sample across `section` values, capped at
    MAX_CHUNKS_PER_JUDGMENT per judgment_id so a handful of long judgments
    can't dominate the gold set."""
    rng = random.Random(seed)
    all_chunks = _fetch_all_chunks(cur)

    by_section: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in all_chunks:
        by_section[chunk["section"]].append(chunk)
    for bucket in by_section.values():
        rng.shuffle(bucket)

    per_section = max(n // len(SECTIONS), 1)
    judgment_counts: Dict[int, int] = defaultdict(int)
    sampled: List[Dict[str, Any]] = []

    for section in SECTIONS:
        taken = 0
        for chunk in by_section.get(section, []):
            if taken >= per_section or len(sampled) >= n:
                break
            if judgment_counts[chunk["judgment_id"]] >= MAX_CHUNKS_PER_JUDGMENT:
                continue
            sampled.append(chunk)
            judgment_counts[chunk["judgment_id"]] += 1
            taken += 1

    rng.shuffle(sampled)
    return sampled[:n]


def generate_question_for_chunk(backend: AnthropicBackend, chunk: Dict[str, Any]) -> str:
    gen = backend.generate(
        context_block=chunk["content"],
        user_query="Write the question now.",
        system_prompt=GOLD_GEN_SYSTEM_PROMPT.format(section=chunk["section"]),
    )
    return gen.text.strip()


def build_gold_record(chunk: Dict[str, Any], question: str, model_id: str) -> Dict[str, Any]:
    return {
        "query": question,
        "source_chunk_id": chunk["chunk_id"],
        "source_judgment_id": chunk["judgment_id"],
        "section": chunk["section"],
        "generator_model": model_id,
    }


def build_gold_set(n: int, seed: int, model: str, api_key: Optional[str]) -> List[Dict[str, Any]]:
    conn = psycopg2.connect(_default_db_connection_string())
    try:
        cur = conn.cursor()
        try:
            chunks = sample_chunks(cur, n, seed)
        finally:
            cur.close()
    finally:
        conn.close()

    backend = AnthropicBackend(model_id=model, api_key=api_key)
    records = []
    for i, chunk in enumerate(chunks, 1):
        question = generate_question_for_chunk(backend, chunk)
        records.append(build_gold_record(chunk, question, backend.model_id))
        print(f"[{i}/{len(chunks)}] chunk {chunk['chunk_id']} ({chunk['section']}): {question}")
    return records


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
        "--model", type=str, default="claude-sonnet-5",
        help="Anthropic model id used to write questions (default: claude-sonnet-5)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="Overrides ANTHROPIC_API_KEY from .env"
    )
    parser.add_argument(
        "--external-ok", action="store_true", required=True,
        help="Confirm sending judgment_chunks content to Anthropic's API (required)",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT, help="Output JSON path"
    )
    args = parser.parse_args()

    records = build_gold_set(args.n, args.seed, args.model, args.api_key)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nWrote {len(records)} gold records to {args.output}")


if __name__ == "__main__":
    main()
