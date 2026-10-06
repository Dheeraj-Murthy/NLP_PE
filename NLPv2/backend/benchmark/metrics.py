"""Pure IR metric functions for the retrieval benchmark.

No pytest suite exists in this repo (see AGENTS.md) — verify these by
running retrieval_eval.py and reading its printed output, the same
convention as ingestion/verify_search.py.

Every gold query has exactly one known-relevant chunk (the chunk it was
synthetically generated from), so these are the single-relevant-document
simplifications of recall@k/MRR/nDCG@k, not the general multi-relevant
versions.
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def rank_of(relevant_id: Any, ranked_ids: Sequence[Any]) -> Optional[int]:
    """1-based rank of relevant_id in ranked_ids, or None if absent."""
    for i, rid in enumerate(ranked_ids):
        if rid == relevant_id:
            return i + 1
    return None


def recall_at_k(relevant_id: Any, ranked_ids: Sequence[Any], k: int) -> int:
    """1 if the relevant doc is within the top k, else 0."""
    rank = rank_of(relevant_id, ranked_ids)
    return 1 if rank is not None and rank <= k else 0


def mrr(relevant_id: Any, ranked_ids: Sequence[Any]) -> float:
    rank = rank_of(relevant_id, ranked_ids)
    return 1.0 / rank if rank is not None else 0.0


def ndcg_at_k(relevant_id: Any, ranked_ids: Sequence[Any], k: int) -> float:
    """Single-relevant-doc nDCG@k. Ideal DCG is 1 (relevant doc at rank 1),
    so this reduces to 1/log2(rank+1) when the doc is within the top k,
    else 0."""
    rank = rank_of(relevant_id, ranked_ids)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def aggregate(
    targets_and_ranked: List[Tuple[Any, Sequence[Any]]], k_values: Sequence[int]
) -> Dict[str, float]:
    """Mean recall@k / nDCG@k per k, and mean MRR, across all queries."""
    n = len(targets_and_ranked)
    if n == 0:
        return {}
    out: Dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = sum(recall_at_k(t, r, k) for t, r in targets_and_ranked) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(t, r, k) for t, r in targets_and_ranked) / n
    out["mrr"] = sum(mrr(t, r) for t, r in targets_and_ranked) / n
    return out
