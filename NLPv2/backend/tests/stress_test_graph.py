"""
Backend-only stress test for the citation graph endpoints (issue #3).

Calls the API directly, with no Streamlit or browser involved, so a slow
citation graph page can be pinned on the backend (query + JSON) or on the
frontend (Graphviz / SVG drawing). For the most-cited cases it times:

  - GET /graph/judgment/{id}            ego graph, every depth x max_nodes
  - GET /graph/judgment/{id}/mind-map   several cases-per-branch settings
  - GET /graph/judgment/{id}/cited-by   the network table, with and without a search

and prints latency (median / max over --repeat runs), payload size and
node / edge counts. --concurrency fires the same requests in parallel to see
how the API holds up under several users.

Usage (API running):
    python backend/tests/stress_test_graph.py
    python backend/tests/stress_test_graph.py --cases 10 --repeat 5 --concurrency 4
    python backend/tests/stress_test_graph.py --ids 123 456 --api http://host:8000
"""

import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import requests

DEPTHS = [1, 2, 3]
MAX_NODES = [100, 250, 500]
PER_BRANCH = [12, 30, 50]


def timed_get(session: requests.Session, url: str, params: Dict[str, Any], timeout: float) -> Tuple[float, int, Any]:
    """Seconds taken, response size in bytes and the "data" payload."""
    start = time.perf_counter()
    resp = session.get(url, params=params, timeout=timeout)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    return elapsed, len(resp.content), resp.json().get("data")


def run_case(api: str, judgment_id: int, repeat: int, concurrency: int, timeout: float) -> List[Dict[str, Any]]:
    requests_to_make = [
        (f"ego depth={d} max_nodes={n}", f"/graph/judgment/{judgment_id}", {"depth": d, "max_nodes": n})
        for d in DEPTHS
        for n in MAX_NODES
    ]
    requests_to_make += [
        (f"mind-map per_branch={p}", f"/graph/judgment/{judgment_id}/mind-map", {"per_branch": p})
        for p in PER_BRANCH
    ]
    requests_to_make += [
        ("table cited-by", f"/graph/judgment/{judgment_id}/cited-by", {"limit": 50}),
        ("table cited-by q=v.", f"/graph/judgment/{judgment_id}/cited-by", {"limit": 50, "q": "v."}),
    ]

    rows = []
    session = requests.Session()
    for name, path, params in requests_to_make:
        url = f"{api}{path}"
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(
                    pool.map(lambda _: timed_get(session, url, params, timeout), range(repeat * concurrency))
                )
        except requests.RequestException as e:
            rows.append({"judgment_id": judgment_id, "request": name, "error": str(e)[:80]})
            continue
        times = [r[0] for r in results]
        data = results[0][2] or {}
        rows.append(
            {
                "judgment_id": judgment_id,
                "request": name,
                "median_ms": statistics.median(times) * 1000,
                "max_ms": max(times) * 1000,
                "kb": results[0][1] / 1024,
                "nodes": len(data.get("nodes", [])) if "nodes" in data else "",
                "edges": len(data.get("edges", [])) if "edges" in data else "",
                "total": data.get("total_nodes", data.get("total", "")),
            }
        )
    return rows


def print_rows(rows: List[Dict[str, Any]]) -> None:
    print(f"{'case':>8}  {'request':<28} {'median ms':>10} {'max ms':>9} {'KB':>8} {'nodes':>6} {'edges':>6} {'total':>7}")
    for r in rows:
        if "error" in r:
            print(f"{r['judgment_id']:>8}  {r['request']:<28} ERROR {r['error']}")
            continue
        print(
            f"{r['judgment_id']:>8}  {r['request']:<28} {r['median_ms']:>10.0f} {r['max_ms']:>9.0f} "
            f"{r['kb']:>8.1f} {r['nodes']!s:>6} {r['edges']!s:>6} {r['total']!s:>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/"))
    parser.add_argument("--ids", type=int, nargs="*", help="Judgment IDs to test (default: the top landmark cases)")
    parser.add_argument("--cases", type=int, default=5, help="How many landmark cases to test when --ids is not given")
    parser.add_argument("--repeat", type=int, default=3, help="Runs per request (per worker)")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel requests per measurement")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    ids = args.ids
    if not ids:
        resp = requests.get(f"{args.api}/graph/landmark-cases", params={"limit": args.cases}, timeout=args.timeout)
        resp.raise_for_status()
        ids = [c["judgment_id"] for c in resp.json()["data"]]
        print(f"Testing the {len(ids)} most-cited cases: {ids}")

    print(f"API {args.api} · repeat {args.repeat} · concurrency {args.concurrency}\n")
    for judgment_id in ids:
        print_rows(run_case(args.api, judgment_id, args.repeat, args.concurrency, args.timeout))
        print()


if __name__ == "__main__":
    main()
