#!/bin/bash
# PDF ingestion script for Legal RAG
# Usage: ./ingest_pdfs.sh /path/to/pdfs [num_workers]

set -e

INPUT_DIR="${1:-./pdfs}"
WORKERS="${2:-8}"

if [ ! -d "$INPUT_DIR" ]; then
	echo "Error: Directory $INPUT_DIR does not exist"
	exit 1
fi

PDF_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*.pdf" | wc -l)
if [ "$PDF_COUNT" -eq 0 ]; then
	echo "Error: No PDF files found in $INPUT_DIR"
	exit 1
fi

echo "Found $PDF_COUNT PDF files to ingest"
echo "Using $WORKERS workers"

cd "$(dirname "$0")/../.."

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-legal_rag}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"

export DB_CONNECTION="host=$DB_HOST port=$DB_PORT dbname=$DB_NAME user=$DB_USER password=$DB_PASSWORD"

if [ -f "database/ingest_fast.py" ]; then
	echo "Using parallel ingestion..."
	python -c "
import sys
sys.path.insert(0, 'database')
import os
os.environ['DB_CONNECTION'] = '$DB_CONNECTION'
exec(open('database/ingest_fast.py').read())
" --input "$INPUT_DIR" --workers "$WORKERS"
else
	echo "Using standard ingestion..."
	python -c "
import sys
sys.path.insert(0, 'database')
import os  
os.environ['DB_CONNECTION'] = '$DB_CONNECTION'
exec(open('database/ingest.py').read())
" --input "$INPUT_DIR"
fi

echo "Ingestion complete"
