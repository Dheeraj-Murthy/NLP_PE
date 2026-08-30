# Legal RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for legal document analysis using local GPU inference with Qwen2.5-14B-Instruct and PostgreSQL pgvector database.

---

## Features

- **2-Stage Retrieval**: Vector search + cross-encoder reranking for precision
- **Multiple Query Modes**: Single query, interactive, chat with memory
- **Document OCR**: Upload PDFs/images for analysis
- **API Server**: FastAPI endpoints for chatbot integration
- **Citation-First**: Grounded answers with legal citations

---

## Architecture

```
User Query
 → Query Embedding (BGE-base)
 → Stage-1 Vector Search (pgvector + HNSW)
 → Stage-2 Cross-Encoder Reranking
 → Prompt Builder
 → Qwen2.5-14B-Instruct (GPU)
 → Post-Processing
 → Answer + Citations
```

---

## File Structure

```
backend/
├── retriever.py            # Stage-1 vector retrieval
├── reranker.py             # Stage-2 cross-encoder reranking
├── prompt_builder.py       # RAG prompt construction
├── llm_inference.py        # Qwen2.5 inference engine
├── post_processor.py       # Response cleaning & citations
├── document_processor.py   # PDF/image OCR
├── rag_pipeline.py        # Main orchestrator
├── api.py                  # FastAPI server
├── main.py                 # CLI interface
├── demo.py                 # Quick demo without LLM
├── requirements.txt        # Dependencies
├── .gitignore
├── README.md
└── AGENTS.md               # Agent guidelines
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt

# For OCR support (optional)
pip install pytesseract pdf2image Pillow
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
```

### Prerequisites

- PostgreSQL with pgvector extension
- GPU with ~48GB VRAM (for 14B model)
- Python 3.8+
- Database `legal_rag` with embedded judgments

---

## Usage

### CLI

```bash
# Single query
python main.py --query "What are the principles of natural justice?"

# Interactive mode
python main.py --interactive

# Chat mode (multi-turn with memory)
python main.py --chat
python main.py --chat --clear-history  # Fresh start

# Document OCR
python main.py --document case.pdf --query "Summarize this case"
python main.py --document scan.jpg --no-retrieval  # OCR only

# Test queries
python main.py --test
python main.py --retrieval-test

# Custom config
python main.py --query "legal question" --top-k 10 --threshold 0.25 --debug
```

### API Server

```bash
python api.py
# Runs at http://localhost:8000
# Docs at http://localhost:8000/docs
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Single query |
| `/chat` | POST | Multi-turn chat |
| `/chat/clear` | POST | Clear history |
| `/chat/history` | GET | Get history |
| `/document` | POST | OCR + query |
| `/retrieval-test` | POST | Test retrieval |
| `/status` | GET | System config |

### Python Usage

```python
from rag_pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()

# Single query
result = pipeline.query("What are the principles of natural justice?")

# Chat
result = pipeline.chat("Tell me more about that")

# Document
result = pipeline.query_with_document("case.pdf", "Summarize this")

# Access response
print(result["answer"])
print(result["confidence"])
print(result["citations"])
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 8 | Final chunks after reranking |
| `similarity_threshold` | 0.3 | Minimum similarity |
| `candidate_k` | 30 | Stage-1 pool size |
| `max_context_length` | 4000 | Prompt character limit |
| `max_new_tokens` | 512 | LLM output limit |
| `model_name` | Qwen/Qwen2.5-7B-Instruct-1M | LLM model |

---

## Database Schema

```sql
judgments (id, petitioner, respondent, court, date_of_judgment, bench, citations, judgment_text)
judgment_chunks (chunk_id, judgment_id, section, content)
judgment_embeddings (chunk_id, embedding vector(768))
```

Sections: `facts | issues | arguments | ratio | judgment`

---

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
accelerate>=0.25.0
psycopg2-binary>=2.9.0
numpy>=1.24.0
tqdm>=4.65.0
pytesseract>=0.3.10
pdf2image>=1.16.3
Pillow>=9.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
```

---

## Design Principles

- **Hallucination Prevention**: Strict RAG, context-only answering
- **Citation-First**: Grounded answers with legal citations
- **Modular**: Each component independently testable
- **Production-Ready**: Error handling, metrics, logging
