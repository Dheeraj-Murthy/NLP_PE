-- Statute schema: Constitution of India + Bharatiya Nyaya Sanhita
-- Run AFTER clean_schema.sql (does not touch judgment tables)

-- Master statute record
CREATE TABLE IF NOT EXISTS statutes (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,          -- 'Constitution of India, 1950'
    short_title TEXT,             -- 'Constitution' / 'BNS'
    year INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

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

-- HNSW index for statute similarity search
CREATE INDEX IF NOT EXISTS statute_embedding_hnsw_idx
    ON statute_embeddings USING hnsw (embedding vector_cosine_ops);

-- BM25 full-text search
CREATE INDEX IF NOT EXISTS statute_sections_tsv_idx
    ON statute_sections USING GIN (content_tsv);

-- Lookup by statute + section number
CREATE INDEX IF NOT EXISTS statute_sections_statute_idx
    ON statute_sections(statute_id);

-- Prevent duplicates on re-run (idempotent ingest)
CREATE UNIQUE INDEX IF NOT EXISTS idx_statute_sections_unique
    ON statute_sections(statute_id, section_number);
