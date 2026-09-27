"""Remove duplicate judgments caused by repeated ingestion runs over the
same PDFs. Dedup key is the exact judgment_text content (md5 hash), not
petitioner/respondent/date — those can collapse many genuinely different
judgments together when metadata extraction fails (e.g. every 'Unknown'
row), so deduping on them would delete real, distinct judgments.

judgment_chunks and judgment_embeddings both cascade on judgment delete
(ON DELETE CASCADE in the schema), so deleting a duplicate judgments row
cleans up its chunks/embeddings automatically.

Usage: python dedupe_judgments.py [--execute]   (dry-run by default)
"""

import argparse
import os

import psycopg2


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate judgments")
    parser.add_argument(
        "--execute", action="store_true", help="Actually delete (default: dry-run/report only)"
    )
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5433"),
        dbname=os.environ.get("DB_NAME", "legal_rag"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
    )

    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT count(*) FROM (
                SELECT md5(judgment_text) AS h, count(*) AS n
                FROM judgments
                GROUP BY h
                HAVING count(*) > 1
            ) dupes
            """
        )
        (dup_groups,) = cur.fetchone()

        cur.execute(
            """
            SELECT sum(n - 1) FROM (
                SELECT md5(judgment_text) AS h, count(*) AS n
                FROM judgments
                GROUP BY h
                HAVING count(*) > 1
            ) dupes
            """
        )
        (rows_to_delete,) = cur.fetchone()
        rows_to_delete = rows_to_delete or 0

        cur.execute("SELECT count(*) FROM judgments")
        (total,) = cur.fetchone()

        print(f"Total judgments: {total}")
        print(f"Duplicate groups (identical judgment_text, >1 copy): {dup_groups}")
        print(f"Rows that would be deleted (keeping one copy per group): {rows_to_delete}")
        print(f"Remaining after dedup: {total - rows_to_delete}")

        if not args.execute:
            print("\nDry run only — rerun with --execute to actually delete.")
            return

        cur.execute(
            """
            DELETE FROM judgments
            WHERE id NOT IN (
                SELECT min(id)
                FROM judgments
                GROUP BY md5(judgment_text)
            )
            """
        )
        deleted = cur.rowcount
        conn.commit()
        print(f"\nDeleted {deleted} duplicate judgment(s) (chunks/embeddings cascaded).")

        cur.execute("SELECT count(*) FROM judgments")
        (remaining,) = cur.fetchone()
        print(f"Judgments remaining: {remaining}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
