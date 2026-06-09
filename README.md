# Legal RAG — NLP System for Indian Case Law

A production-ready Retrieval-Augmented Generation system for searching,
querying, and analyzing Indian court judgments using local LLM inference and
vector search.

---

## Background

This project began as a Project Elective under Prof. Tulika at IIIT Bangalore,
focused on building NLP systems for the Indian legal scene. Over the previous
semester, we developed a 2-stage RAG pipeline accumulating data from over 50,000
Indian Supreme Court judgments spanning 75+ years, using a Qwen Instruct model
to create a chatbot that understands Indian law and answers queries relevant to
the Indian Penal Code and jurisdiction.

As part of our research and market analysis, we conducted surveys and interviews
with students, professors, and working alumni from reputed National Law
Universities including NLSIU Bangalore and NUSRL Ranchi. The survey instrument
and full outcomes are available at
[`AI Legal Assistant – User Assessment Survey(1-217).xlsx`](<./AI%20Legal%20Assistant%20%E2%80%93%20User%20Assessment%20Survey(1-217).xlsx>).

---

## Problem Statement & Vision

India's legal ecosystem produces massive volumes of judgments, appeals, and
legal commentary. However:

- Legal texts are long, complex, and difficult to search manually
- Lawyers and students spend hours reading case law without tooling support
- Existing legal research tools are expensive and not optimized for Indian legal
  language

**Vision**: Build a production-ready NLP system tailored to Indian legal
language that enables fast semantic search, grounded question-answering,
summarization, petition generation, and insights over court judgments.

**End users**: Lawyers, law students, legal researchers, LegalTech startups, and
citizens checking case context.

---

## Objectives & Scope

### Core Objectives

- Build a searchable case-law knowledge base with vector embeddings
- Deliver grounded Q/A over legal queries with citations
- Provide high-quality, citation-anchored summaries of judgments
- Facilitate petition generation from case context and legal principles
- Create a secure, scalable web API and UI

### Out of Scope

- Predict case outcomes
- Replace legal professionals

---

## Implementation: 2-Stage Retrieval Pipeline

```
User Query
  → BGE Embedding (768d)
  → Stage 1: pgvector Cosine Search (candidate_k=30, threshold=0.2)
  → Stage 2: Cross-Encoder Reranking (top_k=8)
  → Prompt Builder (Qwen chat template)
  → Qwen2.5-7B-Instruct-1M (GPU, greedy decoding)
  → Post-Processor (citation extraction, quality checks)
  → Answer + Citations
```

1. **Stage 1 — High-Recall Retrieval**: Query is embedded with
   `BAAI/bge-base-en-v1.5`. PostgreSQL/pgvector performs cosine similarity
   search over 768-dimensional judgment chunk embeddings with HNSW index.
   Returns 30 candidates at threshold 0.2.

2. **Stage 2 — Reranking**: Cross-encoder
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each candidate pair (query,
   chunk). Top 8 are selected by relevance.

3. **Generation**: Selected chunks are formatted into a Qwen-instruct prompt
   with strict system instruction: _"Answer ONLY using the provided context."_
   The LLM runs with `temperature=0.2, do_sample=False` for deterministic,
   grounded output.

4. **Post-Processing**: Citations are extracted from the response text,
   confidence scores computed from embedding similarity and lexical overlap, and
   quality metrics reported.

---

## Datasets

1. **LLM Fine Tuning Dataset of Indian Legal Texts** — Curated question-answer
   pairs from IPC, CRPC, and the Indian Constitution. Used for instruction
   tuning evaluation.

2. **Indian Supreme Court Judgments (Kaggle)** — CSV index of judgments
   extracted from the Supreme Court API, including PDFs. Metadata contains
   court, case number, date, and language. Used for ingestion and retrieval.

3. **Legal Dataset: SC Judgments India (1950–2024)** — 26,000+ PDFs covering
   ~98% of Supreme Court judgments available on Indian Kanoon. Primary corpus
   for the vector knowledge base. Judgments span 75+ years of Indian legal
   history.

---

## Tech Stack

| Layer          | Technology                                          |
| -------------- | --------------------------------------------------- |
| Backend        | Python 3.11, FastAPI, Uvicorn                       |
| LLM            | Qwen/Qwen2.5-7B-Instruct-1M (7B, ~16GB VRAM)        |
| Embeddings     | BAAI/bge-base-en-v1.5 (768d, sentence-transformers) |
| Reranker       | cross-encoder/ms-marco-MiniLM-L-6-v2                |
| Vector DB      | PostgreSQL 16 + pgvector (HNSW, cosine_ops)         |
| Ingestion      | pdftotext (poppler-utils), custom metadata parser   |
| OCR            | Tesseract + pytesseract + pdf2image (optional)      |
| Frontend       | Next.js (chatbot-ui) + bare HTML/JS fallback        |
| Deployment     | Docker Compose (3 services)                         |
| Infrastructure | GPU host (NVIDIA, ~16GB VRAM)                       |

---

## System Architecture

```
┌──────────┐    ┌──────────────────────────────────────┐
│  User     │    │           FastAPI Server             │
│  (CLI/UI) │───▶│  ┌─────────┐  ┌──────────────────┐  │
└──────────┘    │  │Retriever│  │ Reranker         │  │
                │  │(pgvector│  │(Cross-Encoder)   │  │
                │  │ cosine) │  │                  │  │
                │  └────┬────┘  └────────┬─────────┘  │
                │       │                │             │
                │  ┌────▼────────────────▼─────────┐  │
                │  │      Prompt Builder            │  │
                │  └────────────────┬───────────────┘  │
                │                   │                  │
                │  ┌────────────────▼───────────────┐  │
                │  │   Qwen2.5-7B-Instruct (GPU)    │  │
                │  └────────────────┬───────────────┘  │
                │                   │                  │
                │  ┌────────────────▼───────────────┐  │
                │  │    Post-Processor               │  │
                │  │  (citations, confidence, QA )   │  │
                │  └────────────────┬───────────────┘  │
                └───────────────────┼──────────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │    PostgreSQL + pgvector     │
                     │  ┌────────────────────────┐ │
                     │  │ judgments              │ │
                     │  │ judgment_chunks        │ │
                     │  │ judgment_embeddings    │ │
                     │  │   (vector(768), HNSW)  │ │
                     │  └────────────────────────┘ │
                     └─────────────────────────────┘
```

---

## Database Schema

```sql
judgments (id, petitioner, respondent, court, date_of_judgment DATE,
           bench TEXT[], citations JSONB, judgment_text TEXT)

judgment_chunks (chunk_id PK, judgment_id FK,
                 section CHECK(facts|issues|arguments|ratio|judgment),
                 content TEXT)

judgment_embeddings (chunk_id PK FK, embedding vector(768))

-- HNSW index on cosine similarity
CREATE INDEX judgment_embedding_hnsw_idx
  ON judgment_embeddings USING hnsw (embedding vector_cosine_ops);
```

---

## Software Requirements

### Functional

- Answer legal questions with grounded case citations
  - Search judgments by semantic similarity
  - Refuse or warn when answer is unsupported by retrieved context
  - Always include "not legal advice" disclaimer
- Show authoritative supporting cases clearly
  - Indicate court + year for every citation
  - Provide multiple relevant precedents when available
- Display quoted passages, not just references
  - Highlight relevant sections in responses
  - Link answer text to cited paragraph IDs
- Preserve traceability to original documents
  - Maintain chunk-to-judgment mapping
  - Citations include case name, court, year, and paragraph
- Handle conversation and ambiguity
  - Support follow-up questions via chat mode
  - Ask for clarification when retrieval returns no results

### Non-Functional

- Fast and reliable performance
  - Retrieval in ~200ms, generation in ~2–4s
  - Consistent as corpus grows (HNSW index, bounded candidate retrieval)
- High trust and accuracy
  - Strict context-only prompting: LLM must not fabricate
  - Confidence scoring with similarity + lexical overlap
- Security and privacy
  - No user data logged to external services
  - Self-hosted GPU inference, no API calls to third parties
- Maintainable and observable
  - Modular pipeline: retriever, reranker, LLM, post-processor
  - Metrics and debug info per query
- Clear and safe UX
  - Consistent response format across CLI, API, and UI
  - "Not legal advice" disclaimer on every response

---

## Deployment

### Docker (Recommended)

```bash
docker-compose -f deploy/docker/docker-compose.yml up -d
```

Three services: | Service | Port | Description |
|---------|------|-------------| | postgres | 5432 | PostgreSQL 16 + pgvector |
| api | 8000 | FastAPI backend | | frontend | 3000 | Next.js chat interface |

### Manual Setup

```bash
# 1. Database
bash deploy/db_setup/init_db.sh

# 2. Ingest PDFs
python database/ingest.py --input /path/to/pdfs

# 3. Start API
cd rag_model && python api.py          # FastAPI at :8000

# 4. Query via CLI
cd rag_model && python main.py --query "principles of natural justice?"
```

### API Endpoints

| Endpoint          | Method | Description                       |
| ----------------- | ------ | --------------------------------- |
| `/query`          | POST   | Single grounded query             |
| `/chat`           | POST   | Multi-turn chat with memory       |
| `/chat/clear`     | POST   | Clear conversation history        |
| `/chat/history`   | GET    | Retrieve chat history             |
| `/document`       | POST   | OCR + query on uploaded PDF/image |
| `/retrieval-test` | POST   | Debug: test retrieval without LLM |
| `/status`         | GET    | System configuration and health   |

---

## Usage Examples

```bash
# CLI — single query
python main.py --query "What are the principles of natural justice?"

# CLI — interactive
python main.py --interactive

# CLI — chat with memory
python main.py --chat

# CLI — retrieval only (no GPU needed)
python main.py --retrieval-test

# API
python api.py   # → http://localhost:8000/docs
```

**Python API**:

```python
from rag_pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()
result = pipeline.query("What is due process in administrative law?")
print(result["answer"])       # Grounded answer text
print(result["citations"])    # List of case citations
print(result["confidence"])   # Confidence score [0, 1]
```

---

## Project Status

- ✅ Core RAG pipeline (retriever → reranker → LLM → post-processor)
- ✅ PostgreSQL/pgvector integration with HNSW index
- ✅ FastAPI server with 7 endpoints
- ✅ CLI with single, interactive, chat, and test modes
- ✅ PDF ingestion pipeline with metadata parsing
- ✅ Docker Compose deployment (postgres + api + frontend)
- ✅ OCR support for scanned documents
- ✅ Multiple frontends (Next.js chatbot + bare HTML/JS)
- 🔄 Web scraper for Karnataka High Court judgments
- 🔄 LegalParam model integration
- 📝 No automated tests or CI/CD yet
- 📝 No production monitoring/logging infrastructure

---

## Next Steps

- **Petition Generation**: Build templates and LLM-guided drafting workflows for
  legal petitions grounded in retrieved case law.
- **Fine-tuned Legal Models**: Instruction-tune a smaller LLM on Indian legal
  question-answer pairs for reduced cost and improved accuracy.
- **Hybrid Retrieval**: Combine dense vector search with sparse keyword
  retrieval (BM25) for improved recall.
- **Multilingual Support**: Extend to Indian regional language judgments.
- **High Court Integration**: Expand beyond Supreme Court to Karnataka, Delhi,
  Bombay, and other High Courts.
- **Citation Graph Search**: Enable traversal of citation networks between
  judgments.
- **Automated Evaluation**: Build a benchmark suite with curated legal queries
  and ground-truth citations.
- **Production Hardening**: CI/CD, monitoring, logging, and automated testing.

---

## Disclaimer

This system is a research and productivity tool. It does **not** provide legal
advice. All responses should be verified against original case law by qualified
legal professionals.
