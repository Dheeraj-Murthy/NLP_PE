"""
Compute Citation Graph Stats for Legal RAG

Computes PageRank and in/out degree over citation_edges and stores them in
judgment_graph_stats, so the API reads scores instead of running PageRank
while serving. Run after build_citation_edges.py.

    python compute_graph_stats.py

Only edge IDs are loaded (no citation text), and the table is replaced in one
transaction.
"""

import json
import os
import sys
import time
from pathlib import Path

import networkx as nx
import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from graph.schema import ensure_schema
import tracking

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

METRICS_PATH = Path(__file__).resolve().parent / "outputs" / "graph_stats_metrics.json"


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "legal_rag"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )


def compute_graph_stats() -> dict:
    started = time.time()
    conn = get_db_connection()
    ensure_schema(conn)

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM judgments;")
            nodes = [r[0] for r in cur.fetchall()]
            # Same edge set the API walks: resolved, de-duplicated, no self-citations.
            cur.execute(
                "SELECT DISTINCT source_judgment_id, target_judgment_id FROM citation_edges "
                "WHERE target_judgment_id IS NOT NULL AND source_judgment_id <> target_judgment_id;"
            )
            edges = cur.fetchall()

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        print(f"Graph: {graph.number_of_nodes()} judgments, {graph.number_of_edges()} edges.")

        pagerank = nx.pagerank(graph) if graph.number_of_edges() else {}
        rows = [
            (n, graph.in_degree(n), graph.out_degree(n), float(pagerank.get(n, 0.0)))
            for n in graph.nodes
        ]

        with conn.cursor() as cur:
            cur.execute("DELETE FROM judgment_graph_stats;")
            execute_values(
                cur,
                "INSERT INTO judgment_graph_stats (judgment_id, in_degree, out_degree, pagerank) VALUES %s",
                rows,
                page_size=5000,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    metrics = {
        "judgments": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "judgments_cited": sum(1 for r in rows if r[1] > 0),
        "max_in_degree": max((r[1] for r in rows), default=0),
        "duration_seconds": round(time.time() - started, 1),
    }
    print(f"Stored stats for {len(rows)} judgments in {metrics['duration_seconds']}s.")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    tracking.log_ingestion_run("compute_graph_stats.py", {}, metrics)
    return metrics


if __name__ == "__main__":
    compute_graph_stats()
