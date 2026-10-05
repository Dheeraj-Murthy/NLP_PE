-- Legal RAG schema, part 2: statutes (Constitution of India, Bharatiya
-- Nyaya Sanhita). Independent of the judgment tables.
--
-- Additive and idempotent: safe on an empty database, on one that already
-- has data, and to run more than once. Columns added after a table first
-- shipped are created with the table AND added by ALTER ... IF NOT EXISTS,
-- so older databases are brought up to date without losing anything.
--
-- Applied together with the other file in this folder by
-- deploy/db_setup/init_db.sh and backend/db_schema.py; no code path keeps
-- its own copy of any table.

CREATE EXTENSION IF NOT EXISTS vector;


-- ===========================================================================
-- Statutes: Constitution of India + Bharatiya Nyaya Sanhita
-- ===========================================================================

CREATE TABLE IF NOT EXISTS statutes (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,          -- 'Constitution of India, 1950'
    short_title TEXT,             -- 'Constitution' / 'BNS'
    year INT,
    -- Original PDF, relative to the repository root (see backend/documents.py).
    source_file TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE statutes ADD COLUMN IF NOT EXISTS source_file TEXT;

-- One chunk per article (Constitution) or section (BNS)
CREATE TABLE IF NOT EXISTS statute_sections (
    section_id SERIAL PRIMARY KEY,
    statute_id INT NOT NULL REFERENCES statutes(id) ON DELETE CASCADE,
    section_number TEXT NOT NULL,         -- '21', '124A', '302'
    heading TEXT,                         -- 'Right to life...'
    content TEXT NOT NULL,
    content_tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(heading || ' ' || content, ''))
    ) STORED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 768-dim embeddings (BGE-base-en-v1.5)
CREATE TABLE IF NOT EXISTS statute_embeddings (
    section_id INT PRIMARY KEY REFERENCES statute_sections(section_id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS statute_embedding_hnsw_idx
    ON statute_embeddings USING hnsw (embedding vector_cosine_ops);
-- BM25 full-text search
CREATE INDEX IF NOT EXISTS statute_sections_tsv_idx
    ON statute_sections USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS statute_sections_statute_idx
    ON statute_sections(statute_id);
-- Prevent duplicates on re-run (idempotent ingest)
CREATE UNIQUE INDEX IF NOT EXISTS idx_statute_sections_unique
    ON statute_sections(statute_id, section_number);
