"""
Build Citation Edges Script for Legal RAG

Scans judgments in PostgreSQL, extracts citations from judgment texts,
resolves them to target judgments, and populates the `citation_edges` table.

    python build_citation_edges.py            # judgments with no edges yet
    python build_citation_edges.py --rebuild  # replace every edge

Works on judgment text already in the database; no re-ingestion needed.
Each judgment's own reporter citations are read from the CITATION: header
inside its stored text, so citations like "[1950] S.C.R. 940" resolve too.

Without --rebuild, judgments that already have edges are left alone. That is
right after adding new judgments, but citations from older judgments to the
new ones stay unlinked until the next --rebuild. --rebuild runs in a single
transaction: the API keeps serving the old edges until it commits.

Run compute_graph_stats.py afterwards to refresh PageRank.

Also writes outputs/edge_review_sample.csv: random edges per resolution
method, each citation next to the case it was linked to, with an empty
`correct` column to fill in by hand. Accuracy on the real corpus can only be
measured that way.
"""

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from citation_extractor import CitationExtractor  # also puts backend/ on sys.path
from db_schema import ensure_schema
import tracking

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

METRICS_PATH = Path(__file__).resolve().parent / "outputs" / "build_edges_metrics.json"
REVIEW_PATH = Path(__file__).resolve().parent / "outputs" / "edge_review_sample.csv"
REVIEW_PER_METHOD = 50
# Enough of the judgment text to cover the JUDIS header block.
HEADER_CHARS = 6000
BATCH_SIZE = 500


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5433"),
        dbname=os.getenv("DB_NAME", "legal_rag"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )


def _flush(cur, batch):
    if batch:
        execute_values(
            cur,
            """
            INSERT INTO citation_edges
                (source_judgment_id, target_judgment_id, cited_text, relationship_type,
                 resolution_method, resolution_score)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            batch,
        )
        batch.clear()


def write_review_sample(conn, per_method: int = REVIEW_PER_METHOD) -> int:
    """Random edges per resolution method, for checking by hand."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT method, source_judgment_id, cited_text, target_judgment_id, target_name, score
            FROM (
                SELECT e.resolution_method AS method, e.source_judgment_id,
                       regexp_replace(e.cited_text, '\\s+', ' ', 'g') AS cited_text,
                       e.target_judgment_id,
                       t.petitioner || ' v. ' || t.respondent AS target_name,
                       e.resolution_score AS score,
                       row_number() OVER (PARTITION BY e.resolution_method ORDER BY random()) AS rn
                FROM citation_edges e
                LEFT JOIN judgments t ON t.id = e.target_judgment_id
                WHERE e.resolution_method IN ('reporter', 'exact_name', 'fuzzy', 'ambiguous')
            ) s
            WHERE rn <= %s
            ORDER BY method, source_judgment_id
            """,
            (per_method,),
        )
        rows = cur.fetchall()
    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "source_judgment_id", "cited_text", "target_judgment_id",
                         "target_name", "score", "correct"])
        for row in rows:
            writer.writerow(list(row) + [""])
    return len(rows)


def populate_citation_edges(rebuild: bool = False) -> dict:
    started = time.time()
    conn = get_db_connection()
    ensure_schema(conn)
    extractor = CitationExtractor()

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        print("Fetching judgments metadata...")
        cur.execute(
            "SELECT id, petitioner, respondent, court, date_of_judgment, citations, "
            "left(judgment_text, %s) AS header_text FROM judgments;",
            (HEADER_CHARS,),
        )
        judgments_meta = [dict(r) for r in cur.fetchall()]

    if not judgments_meta:
        print("No judgments found in database. Ingest judgments first.")
        conn.close()
        return {}
    print(f"Loaded metadata for {len(judgments_meta)} judgments.")

    extractor.build_lookup_index(judgments_meta)
    resolver = extractor.resolver
    alias_rows = resolver.alias_rows()
    print(
        f"Built resolver: {len(resolver.case_names)} case names, "
        f"{len(resolver.reporters)} reporter citations."
    )
    del judgments_meta
    if resolver._automaton is None:
        print("WARNING: pyahocorasick is not installed; every case-name citation is "
              "compared against every case name, which takes hours on the full corpus. "
              "pip install pyahocorasick")

    counts: Counter = Counter()
    judgments_scanned = 0
    judgments_with_citations = 0

    try:
        with conn.cursor() as cur:
            # Aliases are derived data: always replaced in full.
            cur.execute("DELETE FROM judgment_aliases;")
            execute_values(
                cur,
                "INSERT INTO judgment_aliases (alias_norm, judgment_id, kind) VALUES %s "
                "ON CONFLICT DO NOTHING",
                alias_rows,
                page_size=5000,
            )

            if rebuild:
                cur.execute("DELETE FROM citation_edges;")
                print("Rebuild: cleared existing citation edges.")
                where = "judgment_text IS NOT NULL"
            else:
                where = (
                    "judgment_text IS NOT NULL AND NOT EXISTS "
                    "(SELECT 1 FROM citation_edges e WHERE e.source_judgment_id = j.id)"
                )

            with conn.cursor(name="judgment_texts") as text_cur:
                text_cur.itersize = 200
                text_cur.execute(f"SELECT j.id, j.judgment_text FROM judgments j WHERE {where};")

                insert_batch = []
                for source_id, text in text_cur:
                    judgments_scanned += 1
                    citations = extractor.extract_citations(text or "")
                    if citations:
                        judgments_with_citations += 1

                    for cit in citations:
                        res = resolver.resolve(cit["cited_text"], source_id=source_id)
                        counts[res.method] += 1
                        if res.method == "self":
                            continue  # a judgment's own title or header citation
                        insert_batch.append(
                            (
                                source_id,
                                res.judgment_id,
                                cit["cited_text"],
                                cit["relationship_type"],
                                res.method,
                                res.score,
                            )
                        )
                        if len(insert_batch) >= BATCH_SIZE:
                            _flush(cur, insert_batch)

                    if judgments_scanned % 2000 == 0:
                        print(f"  Progress: {judgments_scanned} judgments scanned, "
                              f"{sum(counts.values()) - counts['self']} edges so far.")

                _flush(cur, insert_batch)

        conn.commit()
        sampled = write_review_sample(conn)
        print(f"Wrote {sampled} edges to {REVIEW_PATH} for review by hand.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    edges = sum(counts.values()) - counts["self"]
    resolved = counts["reporter"] + counts["exact_name"] + counts["fuzzy"]
    metrics = {
        "judgments_scanned": judgments_scanned,
        "judgments_with_citations": judgments_with_citations,
        "edges_inserted": edges,
        "edges_resolved": resolved,
        "resolution_rate": round(resolved / edges, 4) if edges else 0.0,
        "self_citations_skipped": counts["self"],
        **{f"method_{m}": counts[m] for m in ("reporter", "exact_name", "fuzzy", "ambiguous", "unresolved")},
        "aliases": len(alias_rows),
        "duration_seconds": round(time.time() - started, 1),
    }
    print(
        f"Scanned {judgments_scanned} judgments ({judgments_with_citations} with citations); "
        f"inserted {edges} citation edges, {resolved} resolved "
        f"(reporter {counts['reporter']}, exact name {counts['exact_name']}, fuzzy {counts['fuzzy']}); "
        f"{counts['ambiguous']} ambiguous, {counts['unresolved']} unresolved."
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    tracking.log_ingestion_run("build_citation_edges.py", {"rebuild": rebuild}, metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate every citation edge (one transaction).",
    )
    args = parser.parse_args()
    populate_citation_edges(rebuild=args.rebuild)
