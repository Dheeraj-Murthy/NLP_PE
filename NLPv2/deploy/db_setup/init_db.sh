#!/bin/bash
# Database initialization script for Legal RAG
# Run this once to set up PostgreSQL with pgvector

set -e

echo "=== Legal RAG Database Setup ==="

# Get database connection details from env or use defaults
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-legal_rag}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"

echo "Database: $DB_NAME on $DB_HOST:$DB_PORT"

# Check if psql is available
if ! command -v psql &>/dev/null; then
	echo "Error: psql not found. Install PostgreSQL client."
	exit 1
fi

# Create database if it doesn't exist
echo "Creating database if not exists..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || true

# Connect to database and run schema
echo "Creating schema and tables..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<'EOF'
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main judgments table
CREATE TABLE IF NOT EXISTS judgments (
    id SERIAL PRIMARY KEY,
    petitioner TEXT,
    respondent TEXT,
    court TEXT,
    date_of_judgment DATE,
    bench TEXT[],
    citations JSONB,
    judgment_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Judgment chunks table
CREATE TABLE IF NOT EXISTS judgment_chunks (
    chunk_id SERIAL PRIMARY KEY,
    judgment_id INTEGER REFERENCES judgments(id) ON DELETE CASCADE,
    section TEXT CHECK (section IN ('facts', 'issues', 'arguments', 'ratio', 'judgment')),
    content TEXT,
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings table with 768-dimensional vectors for BGE-base-en-v1.5
CREATE TABLE IF NOT EXISTS judgment_embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES judgment_chunks(chunk_id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create HNSW index for efficient similarity search
CREATE INDEX IF NOT EXISTS judgment_embedding_hnsw_idx ON judgment_embeddings 
USING hnsw (embedding vector_cosine_ops);

-- Create additional indexes
CREATE INDEX IF NOT EXISTS judgment_chunks_judgment_id_idx ON judgment_chunks(judgment_id);
CREATE INDEX IF NOT EXISTS judgment_chunks_section_idx ON judgment_chunks(section);

-- GIN index for BM25-style full-text search (hybrid retrieval, alongside pgvector)
CREATE INDEX IF NOT EXISTS judgment_chunks_content_tsv_idx ON judgment_chunks USING GIN (content_tsv);

-- Show created tables
\dt
SELECT 'Database initialized successfully!' as status;
EOF

echo "=== Database setup complete ==="
echo "Next: Run ingest.py to populate with PDFs"
echo "  python ingestion/ingest.py --input /path/to/pdfs"
