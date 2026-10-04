"""Applies graph/schema.sql, plus the optional pg_trgm search index."""

from pathlib import Path

SCHEMA_SQL = Path(__file__).with_name("schema.sql")

# pg_trgm makes /graph/search fast (LIKE and similarity on aliases). It is a
# trusted extension, but if the role still can't create it, search falls back
# to a slower scan rather than failing.
TRIGRAM_SQL = [
    "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
    "CREATE INDEX IF NOT EXISTS idx_judgment_aliases_trgm "
    "ON judgment_aliases USING GIN (alias_norm gin_trgm_ops);",
]


def ensure_schema(conn) -> bool:
    """Create the graph tables if missing and commit. Returns whether the
    trigram index is available."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text())
    conn.commit()

    try:
        with conn.cursor() as cur:
            for stmt in TRIGRAM_SQL:
                cur.execute(stmt)
        conn.commit()
        return True
    except Exception as err:
        conn.rollback()
        print(f"Warning: pg_trgm unavailable, search will use a slower scan: {err}")
        return False
