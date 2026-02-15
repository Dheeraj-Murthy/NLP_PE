# AGENTS.md - Legal RAG System

This file provides context for agentic coding assistants operating in this repository.

## Project Overview

A production-ready Retrieval-Augmented Generation (RAG) system for legal document analysis using local GPU inference with Qwen2.5-14B-Instruct and PostgreSQL pgvector database. Uses 2-stage retrieval (vector search + cross-encoder reranking).

## Repository Structure

```
rag_model/
├── retriever.py          # Stage-1 vector retrieval (pgvector + BGE embedding)
├── reranker.py           # Stage-2 cross-encoder reranking
├── prompt_builder.py     # RAG prompt construction
├── llm_inference.py      # Qwen2.5 inference engine
├── post_processor.py     # Response cleaning & citations
├── document_processor.py  # OCR for PDF/image documents
├── rag_pipeline.py       # Main orchestrator (2-stage pipeline)
├── main.py               # CLI interface
├── demo.py               # Quick demo without LLM
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

## Build, Run & Test Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running the System
```bash
# Full RAG query
python main.py --query "What are the regulations for educational institutions in Karnataka?"

# Interactive mode
python main.py --interactive

# Built-in test queries
python main.py --test

# Retrieval only (no LLM)
python main.py --retrieval-test

# With debug info
python main.py --query "educational fees" --debug

# Custom top-k and threshold
python main.py --query "natural justice" --top-k 10 --threshold 0.25

# Document OCR mode (alternate path)
python main.py --document case.pdf --query "What is this case about?"
python main.py --document scan.jpg --no-retrieval  # OCR only, no DB retrieval

# Chat mode (multi-turn with memory)
python main.py --chat
python main.py --chat --clear-history  # Start with fresh history
```

### Individual Module Testing
```bash
python retriever.py         # Test retrieval layer
python prompt_builder.py   # Test prompt building
python post_processor.py   # Test post-processing
python demo.py             # Quick retrieval demo
```

### Running Single Test (Manual)
Since there's no formal test framework, test individual modules by running them directly:
```bash
python -c "from rag_pipeline import LegalRAGPipeline; p = LegalRAGPipeline(load_llm=False); print(p.test_retrieval_only('test query'))"
```

## Code Style Guidelines

### Imports
- Standard library imports first, then third-party, then local
- Group by: `os/sys` → `typing` → `third-party` → `local modules`
- Use explicit relative imports: `from retriever import LegalRetriever`

### Formatting
- **Indentation**: 4 spaces (PEP 8 standard)
- **Line length**: Max 100 characters (soft guideline)
- **Blank lines**: 2 between top-level definitions, 1 between methods
- **Trailing whitespace**: Avoid

### Naming Conventions
- **Classes**: PascalCase (e.g., `LegalRAGPipeline`, `CrossEncoderReranker`)
- **Functions/variables**: snake_case (e.g., `retrieve_relevant_chunks`, `query_embedding`)
- **Constants**: UPPER_SNAKE_CASE
- **Private methods**: Leading underscore (e.g., `_load_embedding_model`)

### Type Hints
- Use `typing` module for type annotations
- Common patterns:
  ```python
  from typing import List, Dict, Any, Optional, Tuple
  
  def retrieve_relevant_chunks(
      self, 
      query: str, 
      top_k: int = 8,
      similarity_threshold: float = 0.3
  ) -> List[Dict[str, Any]]:
  ```

### Dataclasses
Use `@dataclass` for simple data containers:
```python
from dataclasses import dataclass

@dataclass
class RAGMetrics:
    retrieval_time: float
    generation_time: float
    total_time: float
    chunks_retrieved: int
```

### Error Handling
- Use try/except with specific exception types
- Always close resources (use `finally` blocks for database connections)
- Raise descriptive errors with context:
  ```python
  raise RuntimeError(f"Failed to load embedding model: {e}")
  ```

### Documentation
- Use docstrings for public classes and functions
- Comment complex logic inline
- Use stage markers in pipeline code: `# ✅ STAGE 1: high-recall retrieval`

### String Formatting
- Prefer f-strings: `f"Query: {query}"`
- Use formatted strings for alignment: `f"{result['confidence']:.2f}"`

### Database Connections
- Always close cursors and connections in `finally` blocks
- Use context managers where possible

### Configuration
- Class `__init__` should accept configurable parameters with sensible defaults
- Default values should match production-ready settings:
  ```python
  def __init__(
      self,
      db_connection_string: str = "dbname=legal_rag",
      model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M",
      top_k: int = 8,
      similarity_threshold: float = 0.3
  ):
  ```

## Key Design Patterns

### 2-Stage Retrieval Pipeline
1. **Stage 1**: Vector search (high recall, candidate_k=30, threshold=0.2)
2. **Stage 2**: Cross-encoder reranking (precision, final_k=8)

### Modular Architecture
- Each component (retriever, reranker, prompt_builder, llm, post_processor) is independently testable
- Components communicate via typed dictionaries
- Easy model swapping (change model_name in constructor)

### Return Values
Pipeline methods return typed dictionaries:
```python
{
    "answer": str,
    "answer_found": bool,
    "confidence": float,
    "citations": List[str],
    "sources": List[Dict],
    "metrics": Dict[str, float]
}
```

## Database Schema

```sql
judgments (id, petitioner, respondent, court, date_of_judgment, bench, citations, judgment_text)
judgment_chunks (chunk_id, judgment_id, section, content)
judgment_embeddings (chunk_id, embedding vector(768))
```

Sections: `facts | issues | arguments | ratio | judgment`

## Environment Requirements

- PostgreSQL with pgvector extension
- Database: `legal_rag`
- GPU with ~48GB VRAM (for 14B model)
- Python 3.8+

## Dependencies

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
accelerate>=0.25.0
psycopg2-binary>=2.9.0
numpy>=1.24.0
tqdm>=4.65.0

# OCR dependencies (optional)
pytesseract>=0.3.10
pdf2image>=1.16.3
Pillow>=9.0.0

# API dependencies
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
```

## API Server

Start the API server:
```bash
pip install fastapi uvicorn python-multipart
python api.py
```

Server runs at `http://localhost:8000`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Service health status |
| GET | `/status` | System configuration |
| POST | `/query` | Single query RAG |
| POST | `/chat` | Multi-turn chat |
| POST | `/chat/clear` | Clear chat history |
| GET | `/chat/history` | Get chat history |
| POST | `/document` | OCR + query document |
| POST | `/retrieval-test` | Test retrieval only |

### API Example (curl)

```bash
# Query
curl -X POST "http://localhost:8000/query" \
  -F "query=What are the principles of natural justice?"

# Chat
curl -X POST "http://localhost:8000/chat" \
  -F "message=What are the principles of natural justice?"

# Document
curl -X POST "http://localhost:8000/document" \
  -F "file=@case.pdf" \
  -F "query=What is this case about?"
```

Swagger docs: `http://localhost:8000/docs`
