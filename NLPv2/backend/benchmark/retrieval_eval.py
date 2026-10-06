#!/usr/bin/env python3
"""Retrieval-quality benchmark.

For each gold query, runs the SAME two-stage retrieval the real pipeline
runs for judgments — LegalRetriever.retrieve_judgment_candidates() then
CrossEncoderReranker.rerank() — and checks whether the chunk the query was
generated from (source_chunk_id) comes back, and at what rank, at each
stage. Judgment chunks only; statutes have a separate table/id space and
are out of scope for this benchmark.

Defaults mirror rag_pipeline.py's ACTUAL hardcoded retrieval config
(stage1_k=30, stage1_threshold=0.2, stage2_k=top_k=8) — not main.py's CLI
--threshold default of 0.3, which is accepted but never passed into
retrieval (rag_pipeline.py stores it only for metrics/debug output; the
real stage-1 call always uses the hardcoded 0.2). Using 0.3 here would
silently benchmark a different configuration than what the live pipeline
actually runs.

No LLM is loaded — LegalRetriever only loads the BGE encoder and
CrossEncoderReranker only loads the MiniLM cross-encoder, so this is
inherently CPU-safe.

Usage:
    cd NLPv2/backend
    python benchmark/retrieval_eval.py --gold benchmark/data/gold_set.json
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

# Add backend/ to the path so the dotted `retrieval.*` subpackage import and
# the flat `tracking` import below resolve whether this script is run
# directly or as a module — same shim as backend/tests/smoke_test_citation_graph.py.
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _backend_dir)

from retrieval.retriever import LegalRetriever  # noqa: E402
from retrieval.reranker import CrossEncoderReranker  # noqa: E402
import tracking  # noqa: E402

from metrics import aggregate, rank_of  # noqa: E402

DEFAULT_K_VALUES = (5, 8, 10, 30)


def load_gold_set(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def run_two_stage_retrieval(
    retriever: LegalRetriever,
    reranker: CrossEncoderReranker,
    query: str,
    candidate_k: int,
    threshold: float,
    top_k: int,
    graph_boost: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Same two calls rag_pipeline.query() makes for judgments: hybrid
    recall, then cross-encoder rerank."""
    stage1 = retriever.retrieve_judgment_candidates(
        query=query,
        candidate_k=candidate_k,
        similarity_threshold=threshold,
        graph_boost=graph_boost,
    )
    stage2 = reranker.rerank(query, list(stage1), top_n=top_k)
    return stage1, stage2


def evaluate_query(
    gold: Dict[str, Any], stage1_ids: List[Any], stage2_ids: List[Any], k_values: Sequence[int]
) -> Dict[str, Any]:
    target = gold["source_chunk_id"]
    result: Dict[str, Any] = {
        "query": gold["query"],
        "source_chunk_id": target,
        "source_judgment_id": gold["source_judgment_id"],
        "section": gold["section"],
        "stage1_rank": rank_of(target, stage1_ids),
        "stage2_rank": rank_of(target, stage2_ids),
    }
    for k in k_values:
        result[f"stage1_recall@{k}"] = int(
            result["stage1_rank"] is not None and result["stage1_rank"] <= k
        )
        result[f"stage2_recall@{k}"] = int(
            result["stage2_rank"] is not None and result["stage2_rank"] <= k
        )
    return result


def run_eval(
    gold_path: str,
    candidate_k: int = 30,
    threshold: float = 0.2,
    top_k: int = 8,
    graph_boost: float = 0.0,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> Dict[str, Any]:
    gold_set = load_gold_set(gold_path)
    k_values = sorted(set(k_values))

    # Load real models ONCE — BGE encoder + MiniLM cross-encoder are
    # expensive to construct; never re-instantiate per query.
    retriever = LegalRetriever()
    reranker = CrossEncoderReranker()

    per_query = []
    stage1_pairs: List[Tuple[Any, List[Any]]] = []
    stage2_pairs: List[Tuple[Any, List[Any]]] = []

    start = time.time()
    for gold in gold_set:
        stage1, stage2 = run_two_stage_retrieval(
            retriever, reranker, gold["query"], candidate_k, threshold, top_k, graph_boost
        )
        stage1_ids = [c["chunk_id"] for c in stage1]
        stage2_ids = [c["chunk_id"] for c in stage2]
        target = gold["source_chunk_id"]

        stage1_pairs.append((target, stage1_ids))
        stage2_pairs.append((target, stage2_ids))
        per_query.append(evaluate_query(gold, stage1_ids, stage2_ids, k_values))
    elapsed = time.time() - start

    return {
        "config": {
            "candidate_k": candidate_k,
            "threshold": threshold,
            "top_k": top_k,
            "graph_boost": graph_boost,
            "k_values": k_values,
            "n_queries": len(gold_set),
            "elapsed_s": round(elapsed, 2),
        },
        "per_query": per_query,
        "aggregate_stage1": aggregate(stage1_pairs, k_values),
        "aggregate_stage2": aggregate(stage2_pairs, k_values),
    }


def _flatten_for_mlflow(
    aggregate_stage1: Dict[str, float], aggregate_stage2: Dict[str, float]
) -> Dict[str, float]:
    # mlflow metric names don't allow "@" — rewrite recall@8 -> recall_at_8
    # for the logged metrics; the JSON report keeps the readable "@" form.
    flat: Dict[str, float] = {}
    for k, v in aggregate_stage1.items():
        flat[f"stage1_{k}".replace("@", "_at_")] = v
    for k, v in aggregate_stage2.items():
        flat[f"stage2_{k}".replace("@", "_at_")] = v
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against a synthetic gold query set."
    )
    parser.add_argument("--gold", type=str, required=True, help="Path to gold_set.json")
    parser.add_argument(
        "--candidate-k", type=int, default=30,
        help="Stage-1 candidate_k (default: 30, matches rag_pipeline.py's stage1_k)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Stage-1 similarity threshold (default: 0.2 — rag_pipeline.py's actual "
        "hardcoded stage1_threshold, not main.py's cosmetic --threshold default of 0.3)",
    )
    parser.add_argument(
        "--top-k", type=int, default=8, help="Stage-2 rerank top_k (default: 8)"
    )
    parser.add_argument(
        "--graph-boost", type=float, default=0.0,
        help="Citation-graph PageRank boost (default: 0.0 = off)",
    )
    parser.add_argument(
        "--k-values", type=str, default="5,8,10,30",
        help="Comma-separated k values for recall@k (default: 5,8,10,30)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON report path (default: benchmark/reports/retrieval_eval_<timestamp>.json)",
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    report = run_eval(args.gold, args.candidate_k, args.threshold, args.top_k, args.graph_boost, k_values)

    output = args.output
    if output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        output = os.path.join(reports_dir, f"retrieval_eval_{ts}.json")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nConfig: {report['config']}\n")
    print("Stage 1 (pre-rerank, hybrid recall):")
    for k, v in report["aggregate_stage1"].items():
        print(f"  {k}: {v:.3f}")
    print("\nStage 2 (post-rerank, final top_k):")
    for k, v in report["aggregate_stage2"].items():
        print(f"  {k}: {v:.3f}")
    print(f"\nWrote report to {output}")

    tracking.log_benchmark_run(
        script="retrieval_eval",
        params={
            "candidate_k": args.candidate_k,
            "threshold": args.threshold,
            "top_k": args.top_k,
            "graph_boost": args.graph_boost,
            "n_queries": report["config"]["n_queries"],
        },
        metrics=_flatten_for_mlflow(report["aggregate_stage1"], report["aggregate_stage2"]),
    )


if __name__ == "__main__":
    main()
