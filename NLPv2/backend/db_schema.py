"""Applies the database schema: schema/judgments.sql and schema/statutes.sql."""

from pathlib import Path

SCHEMA_DIR = Path(__file__).with_name("schema")
SCHEMA_FILES = [SCHEMA_DIR / "judgments.sql", SCHEMA_DIR / "statutes.sql"]


def ensure_schema(conn) -> None:
    """Create any missing tables, columns and indexes, then commit."""
    with conn.cursor() as cur:
        for path in SCHEMA_FILES:
            cur.execute(path.read_text())
    conn.commit()
