-- Clean and recreate Legal RAG schema

DROP SCHEMA public CASCADE;
CREATE SCHEMA public;

-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main judgments table
CREATE TABLE judgments (
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
CREATE TABLE judgment_chunks (
    chunk_id SERIAL PRIMARY KEY,
    judgment_id INTEGER REFERENCES judgments(id) ON DELETE CASCADE,
    section TEXT CHECK (section IN ('facts', 'issues', 'arguments', 'ratio', 'judgment')),
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings table with 768-dimensional vectors for BGE-base-en-v1.5
CREATE TABLE judgment_embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES judgment_chunks(chunk_id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create HNSW index for efficient similarity search
CREATE INDEX judgment_embedding_hnsw_idx ON judgment_embeddings 
USING hnsw (embedding vector_cosine_ops);

-- Create additional indexes for better performance
CREATE INDEX judgment_chunks_judgment_id_idx ON judgment_chunks(judgment_id);
CREATE INDEX judgment_chunks_section_idx ON judgment_chunks(section);