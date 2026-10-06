#!/usr/bin/env python3
"""Interface-only hook for a future end-to-end answer-quality benchmark.

Not implemented yet. This is scaffolding so retrieval_eval.py's gold set
(query + known source_chunk_id/source_judgment_id) can be reused directly
once answer-quality eval is built: run LegalRAGPipeline.query() per gold
query, then judge the answer against the known source chunk for
groundedness, citation correctness, and relevance.

Usage (once implemented):
    cd NLPv2/backend
    python benchmark/answer_eval_stub.py --gold benchmark/data/gold_set.json \
        --model claude-sonnet-5 --pipeline-model claude-sonnet-5 --external-ok
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Add backend/ to the path so the dotted `llm.*` subpackage import and the
# flat `rag_pipeline` import below resolve whether this script is run
# directly or as a module — same shim as backend/tests/smoke_test_citation_graph.py.
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

from llm.anthropic_backend import AnthropicBackend  # noqa: E402,F401
from rag_pipeline import LegalRAGPipeline  # noqa: E402,F401


@dataclass
class JudgeVerdict:
    groundedness: float  # 1-5: is the answer supported by retrieved context
    citation_correct: bool  # does result["citations"] include source_judgment_id
    relevance: float  # 1-5: does the answer address the gold query
    raw_judge_text: str


def judge_answer(
    gold: Dict[str, Any], pipeline_result: Dict[str, Any], judge_backend: AnthropicBackend
) -> JudgeVerdict:
    """Prompt judge_backend with the gold query, the pipeline's answer and
    citations, and the known source chunk text as ground truth, asking for
    a groundedness/citation-correctness/relevance rubric score. Reuses
    AnthropicBackend.generate(...) exactly as rag_pipeline._resolve_backend()
    does for claude-* models."""
    raise NotImplementedError("answer-quality judging is not implemented yet")


def run_answer_eval(
    gold_path: str, pipeline_kwargs: Dict[str, Any], judge_model: str, external_ok: bool
) -> Dict[str, Any]:
    """For each gold record: LegalRAGPipeline(**pipeline_kwargs).query(
    gold['query'], external_ok=external_ok) -> judge_answer(...) -> aggregate
    mean groundedness/citation_correct/relevance -> tracking.log_benchmark_run(
    'answer_eval', ...). Reuses the same gold file retrieval_eval.py uses;
    source_chunk_id/source_judgment_id double as ground truth for
    citation-correctness scoring."""
    raise NotImplementedError("answer-quality eval is not implemented yet")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="(Not yet implemented) End-to-end answer-quality benchmark."
    )
    parser.add_argument("--gold", type=str, required=True, help="Path to gold_set.json")
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-5", help="Judge model id"
    )
    parser.add_argument(
        "--pipeline-model", type=str, default=None,
        help="Model the pipeline answers with (default: local Qwen)",
    )
    parser.add_argument(
        "--external-ok", action="store_true",
        help="Confirm sending case text to external model APIs",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    pipeline_kwargs: Dict[str, Any] = {}
    if args.pipeline_model:
        pipeline_kwargs["model_name"] = args.pipeline_model

    run_answer_eval(
        gold_path=args.gold,
        pipeline_kwargs=pipeline_kwargs,
        judge_model=args.model,
        external_ok=args.external_ok,
    )


if __name__ == "__main__":
    main()
