"""
Build Citation Edges Script for Legal RAG

Scans judgments in PostgreSQL, extracts citations from judgment texts,
matches them to target judgments, and populates the `citation_edges` table.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from citation_extractor import CitationExtractor


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "legal_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
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
        conn.commit()

        # Fetch judgment texts to extract citations
        cur.execute("SELECT id, judgment_text FROM judgments WHERE judgment_text IS NOT NULL;")
        judgments_data = cur.fetchall()

        total_edges_inserted = 0

        for row in judgments_data:
            source_id = row["id"]
            text = row["judgment_text"] or ""
            if not text:
                continue

            extracted_citations = extractor.extract_citations(text)
            if not extracted_citations:
                continue

            for cit in extracted_citations:
                cited_text = cit["cited_text"]
                rel_type = cit["relationship_type"]

                # Try to resolve target judgment ID
                target_id = extractor.match_target_judgment(cited_text, judgments_meta)

                cur.execute(
                    """
                    INSERT INTO citation_edges (source_judgment_id, target_judgment_id, cited_text, relationship_type)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (source_id, target_id, cited_text, rel_type),
                )
                total_edges_inserted += 1

        conn.commit()
        print(f"Successfully processed {len(judgments_data)} judgments and inserted {total_edges_inserted} citation edges.")

    conn.close()


if __name__ == "__main__":
    populate_citation_edges()
