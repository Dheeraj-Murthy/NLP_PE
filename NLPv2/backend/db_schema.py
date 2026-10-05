"""Applies the database schema: schema/judgments.sql and schema/statutes.sql.

The files only run when they differ from what the database last applied
(schema_version), so normal startups take no table locks.
"""

import hashlib
from pathlib import Path

SCHEMA_DIR = Path(__file__).with_name("schema")
SCHEMA_FILES = [SCHEMA_DIR / "judgments.sql", SCHEMA_DIR / "statutes.sql"]


def schema_fingerprint() -> str:
    """sha256 of the schema files, concatenated; init_db.sh computes the same."""
    return hashlib.sha256(b"".join(p.read_bytes() for p in SCHEMA_FILES)).hexdigest()


def ensure_schema(conn) -> None:
    """Create any missing tables, columns and indexes, then commit."""
    fingerprint = schema_fingerprint()
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('schema_version') IS NOT NULL")
        if cur.fetchone()[0]:
            cur.execute("SELECT sha256 FROM schema_version")
            row = cur.fetchone()
            if row and row[0] == fingerprint:
                conn.commit()
                return
        for path in SCHEMA_FILES:
            cur.execute(path.read_text())
        cur.execute(
            "INSERT INTO schema_version (sha256) VALUES (%s) "
            "ON CONFLICT (id) DO UPDATE SET sha256 = EXCLUDED.sha256, applied_at = now()",
            (fingerprint,),
        )
    conn.commit()
