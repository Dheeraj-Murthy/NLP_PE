#!/bin/bash
# Database initialization script for Legal RAG.
# Creates the database if needed and applies the schema in backend/schema/
# (judgments.sql, statutes.sql). Safe to re-run: it only adds what is missing.
#
#   bash init_db.sh            create / bring up to date
#   bash init_db.sh --reset    DROP every table and recreate empty (asks first)

set -e

echo "=== Legal RAG Database Setup ==="

# Get database connection details from env or use defaults
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5433}"
DB_NAME="${DB_NAME:-legal_rag}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"
export PGPASSWORD="$DB_PASSWORD"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_DIR="$SCRIPT_DIR/../../backend/schema"

echo "Database: $DB_NAME on $DB_HOST:$DB_PORT"

# Check if psql is available
if ! command -v psql &>/dev/null; then
	echo "Error: psql not found. Install PostgreSQL client."
	exit 1
fi

PSQL=(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER")

# Create database if it doesn't exist
echo "Creating database if not exists..."
"${PSQL[@]}" -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || true

if [ "$1" = "--reset" ]; then
	read -r -p "This deletes ALL data in $DB_NAME. Type the database name to confirm: " answer
	if [ "$answer" != "$DB_NAME" ]; then
		echo "Aborted."
		exit 1
	fi
	"${PSQL[@]}" -d "$DB_NAME" -v ON_ERROR_STOP=1 \
		-c "DROP SCHEMA public CASCADE;" -c "CREATE SCHEMA public;"
fi

echo "Applying schema..."
"${PSQL[@]}" -d "$DB_NAME" -v ON_ERROR_STOP=1 \
	-f "$SCHEMA_DIR/judgments.sql" -f "$SCHEMA_DIR/statutes.sql"
# Same fingerprint as backend/db_schema.py, so apps don't re-apply it.
FINGERPRINT="$(cat "$SCHEMA_DIR/judgments.sql" "$SCHEMA_DIR/statutes.sql" | sha256sum | cut -d' ' -f1)"
"${PSQL[@]}" -d "$DB_NAME" -v ON_ERROR_STOP=1 -q -c \
	"INSERT INTO schema_version (sha256) VALUES ('$FINGERPRINT') ON CONFLICT (id) DO UPDATE SET sha256 = EXCLUDED.sha256, applied_at = now();"
"${PSQL[@]}" -d "$DB_NAME" -c '\dt'

echo "=== Database setup complete ==="
echo "Next: Run ingest.py to populate with PDFs"
echo "  python ingestion/ingest.py --input /path/to/pdfs"
