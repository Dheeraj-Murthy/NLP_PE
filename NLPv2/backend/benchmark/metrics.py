"""Pure IR metric functions for the retrieval benchmark.

No pytest suite exists in this repo (see AGENTS.md) — verify these by
running retrieval_eval.py and reading its printed output, the same
convention as ingestion/verify_search.py.

Each gold query has one PRIMARY known-relevant chunk (the chunk it was
synthetically generated from), but `rank_of` also accepts a set/list of
IDs — the primary chunk plus any near-duplicate chunks found at gold-set
generation time (see generate_gold_set.py's near-duplicate detection).
That makes this a single-equivalence-class simplification of
recall@k/MRR/nDCG@k: any one match in the accepted set counts as the hit,
scored at its best (lowest) rank. A plain scalar id still works exactly
as before — it's just treated as a one-element class.
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

RelevantIds = Union[Any, Sequence[Any], Set[Any]]


def _as_id_set(relevant_ids: RelevantIds) -> Set[Any]:
    if isinstance(relevant_ids, (set, frozenset, list, tuple)):
        return set(relevant_ids)
    return {relevant_ids}


def rank_of(relevant_ids: RelevantIds, ranked_ids: Sequence[Any]) -> Optional[int]:
    """1-based rank of the first id in ranked_ids that belongs to
    relevant_ids (a single id, or a set/list of ids treated as one
    equivalence class), or None if none of them appear."""
    targets = _as_id_set(relevant_ids)
    for i, rid in enumerate(ranked_ids):
        if rid in targets:
            return i + 1
    return None


def recall_at_k(relevant_ids: RelevantIds, ranked_ids: Sequence[Any], k: int) -> int:
    """1 if any id in the relevant equivalence class is within the top k, else 0."""
    rank = rank_of(relevant_ids, ranked_ids)
    return 1 if rank is not None and rank <= k else 0


def mrr(relevant_ids: RelevantIds, ranked_ids: Sequence[Any]) -> float:
    rank = rank_of(relevant_ids, ranked_ids)
    return 1.0 / rank if rank is not None else 0.0


def ndcg_at_k(relevant_ids: RelevantIds, ranked_ids: Sequence[Any], k: int) -> float:
    """Single-relevant-equivalence-class nDCG@k. Ideal DCG is 1 (a relevant
    id at rank 1), so this reduces to 1/log2(rank+1) when the best-ranked
    match is within the top k, else 0."""
    rank = rank_of(relevant_ids, ranked_ids)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def aggregate(
    targets_and_ranked: List[Tuple[RelevantIds, Sequence[Any]]], k_values: Sequence[int]
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
