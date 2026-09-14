"""
Build Citation Edges Script for Legal RAG

Scans judgments in PostgreSQL, extracts citations from judgment texts,
matches them to target judgments, and populates the `citation_edges` table.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from citation_extractor import CitationExtractor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "legal_rag"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )


def populate_citation_edges():
    conn = get_db_connection()
    extractor = CitationExtractor()

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Fetch all judgments metadata
        print("Fetching judgments metadata...")
        cur.execute("SELECT id, petitioner, respondent, court, date_of_judgment, citations FROM judgments;")
        judgments_meta_raw = cur.fetchall()

        if not judgments_meta_raw:
            print("No judgments found in database. Ingest judgments first.")
            conn.close()
            return

        judgments_meta = [dict(r) for r in judgments_meta_raw]
        print(f"Loaded metadata for {len(judgments_meta)} judgments.")

        # Build O(1) lookup index on petitioner / case names
        extractor.build_lookup_index(judgments_meta)
        print("Built case-name lookup index.")

        # Ensure citation_edges table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS citation_edges (
                edge_id SERIAL PRIMARY KEY,
                source_judgment_id INTEGER NOT NULL REFERENCES judgments(id) ON DELETE CASCADE,
                target_judgment_id INTEGER REFERENCES judgments(id) ON DELETE CASCADE,
                cited_text TEXT NOT NULL,
                relationship_type VARCHAR(50) DEFAULT 'cited',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Dedupe guard for re-runs
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_citation_edges_unique
            ON citation_edges(source_judgment_id, COALESCE(target_judgment_id, -1), cited_text);
        """)
        conn.commit()

        # Fetch judgment texts to extract citations
        cur.execute("SELECT id, judgment_text FROM judgments WHERE judgment_text IS NOT NULL;")
        judgments_data = cur.fetchall()

        total_edges_inserted = 0
        total_judgments_with_citations = 0

        batch_size = 500
        insert_batch = []

        for idx, row in enumerate(judgments_data, start=1):
            source_id = row["id"]
            text = row["judgment_text"] or ""
            if not text:
                continue

            extracted_citations = extractor.extract_citations(text)
            if extracted_citations:
                total_judgments_with_citations += 1

            for cit in extracted_citations:
                cited_text = cit["cited_text"]
                rel_type = cit["relationship_type"]

                # Try to resolve target judgment ID via O(1) index
                target_id = extractor.match_target_judgment(cited_text)

                insert_batch.append(
                    (source_id, target_id, cited_text, rel_type)
                )
                total_edges_inserted += 1

                if len(insert_batch) >= batch_size:
                    cur.executemany(
                        """
                        INSERT INTO citation_edges
                        (source_judgment_id, target_judgment_id, cited_text, relationship_type)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING;
                        """,
                        insert_batch,
                    )
                    insert_batch.clear()

            if idx % 2000 == 0 or idx == len(judgments_data):
                print(f"  Progress: {idx}/{len(judgments_data)} judgments scanned, "
                      f"{total_edges_inserted} edges inserted so far.")

        if insert_batch:
            cur.executemany(
                """
                INSERT INTO citation_edges
                (source_judgment_id, target_judgment_id, cited_text, relationship_type)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
                """,
                insert_batch,
            )

        conn.commit()
        print(f"Successfully processed {len(judgments_data)} judgments "
              f"({total_judgments_with_citations} with citations) and inserted "
              f"{total_edges_inserted} citation edges.")

    conn.close()


if __name__ == "__main__":
    populate_citation_edges()
