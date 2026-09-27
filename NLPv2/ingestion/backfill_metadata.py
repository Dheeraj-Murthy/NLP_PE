"""Re-parse petitioner/respondent/bench/date for already-ingested judgments
whose metadata came out as 'Unknown' (or missing bench/date) under the old
parser — without re-running OCR, chunking, or embedding. The clean judgment
text is already stored in the `judgments` table, so this just re-runs the
(now-improved) parse_judgment_metadata against it and updates the row.

Usage: python backfill_metadata.py [--dry-run]
"""

import argparse
import os

import psycopg2
from tqdm import tqdm

from ingest import parse_judgment_metadata


def main():
    parser = argparse.ArgumentParser(description="Backfill judgment metadata")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change, write nothing"
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
            SELECT id, judgment_text, petitioner, respondent, bench, date_of_judgment
            FROM judgments
            WHERE petitioner = 'Unknown' OR respondent = 'Unknown'
            """
        )
        rows = cur.fetchall()
        cur.close()

        print(f"Found {len(rows)} judgment(s) with unresolved metadata.")

        fixed = 0
        still_unknown = 0

        for judgment_id, judgment_text, old_petitioner, old_respondent, old_bench, old_date in tqdm(
            rows, desc="Re-parsing", unit="doc"
        ):
            metadata = parse_judgment_metadata(judgment_text or "", "")

            new_petitioner = metadata["petitioner"]
            new_respondent = metadata["respondent"]
            new_bench = metadata["bench"] or old_bench
            new_date = metadata["date_of_judgment"] or old_date

            changed = (
                new_petitioner != old_petitioner
                or new_respondent != old_respondent
                or new_bench != old_bench
                or new_date != old_date
            )

            if not changed:
                if new_petitioner == "Unknown" and new_respondent == "Unknown":
                    still_unknown += 1
                continue

            fixed += 1
            if new_petitioner == "Unknown" and new_respondent == "Unknown":
                still_unknown += 1

            if not args.dry_run:
                update_cur = conn.cursor()
                update_cur.execute(
                    """
                    UPDATE judgments
                    SET petitioner = %s, respondent = %s, bench = %s, date_of_judgment = %s
                    WHERE id = %s
                    """,
                    (new_petitioner, new_respondent, new_bench, new_date, judgment_id),
                )
                update_cur.close()
                conn.commit()

        print(
            f"\n{'Would update' if args.dry_run else 'Updated'} {fixed}/{len(rows)} judgment(s). "
            f"{still_unknown} remain fully unresolved (genuinely no cause title found)."
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
