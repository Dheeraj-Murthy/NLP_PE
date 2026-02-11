# Local Legal RAG System

🚀 **Upgrade:** This system now uses a **2-stage retrieval pipeline (vector search + cross-encoder reranking)** for higher legal relevance and better grounding.

A production-ready Retrieval-Augmented Generation (RAG) system for legal document analysis using local GPU inference with Qwen2.5-14B-Instruct and PostgreSQL pgvector database.

---

## 🏗️ Architecture

```
User Query
 → Query Embedding (BGE-base)
 → Stage-1 Vector Search (pgvector + HNSW)
 → Candidate Chunks (broad recall)
 → Stage-2 Cross-Encoder Reranking
 → Top-K Chunks (precision filtered)
 → Prompt Builder
 → Qwen2.5-14B-Instruct (GPU)
 → Post-Processing
 → Answer + Citations
```

---

## 🔎 Two-Stage Retrieval (Implemented)

The system uses a **two-stage retrieval pipeline** to improve chunk relevance and reduce noisy context.

### Stage-1 — Vector Candidate Retrieval
- Embedding model: **BAAI/bge-base-en-v1.5 (768-dim)**
- Similarity: cosine distance via pgvector
- Index: HNSW
- Fetches a wider candidate pool (default ≈ 30)
- Optimized for **recall + speed**

### Stage-2 — Cross-Encoder Reranking
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores each *(query, chunk)* pair jointly
- Uses full transformer attention across query + chunk
- Produces semantic relevance score
- Reorders candidates
- Keeps best top-K (default ≈ 8)

**Effect:** Better ordering, stronger legal grounding, cleaner citations.

---

## 📁 File Structure

```
rag_model/
├── retriever.py          # Stage-1 vector retrieval layer
├── reranker.py           # Stage-2 cross-encoder reranking
├── prompt_builder.py     # RAG prompt construction  
├── llm_inference.py      # Qwen2.5-14B inference engine
├── post_processor.py     # Response cleaning & citations
├── rag_pipeline.py       # Main orchestrator (2-stage pipeline)
├── main.py               # CLI interface
├── demo.py               # Quick demo without LLM
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🚀 Quick Start

### Prerequisites

- PostgreSQL with pgvector extension
- GPU with ~48GB VRAM (for 14B model)
- Python 3.8+
- Legal documents embedded in `legal_rag` database

---

### Installation

```bash
pip install -r requirements.txt
```

Database expected:

```
legal_rag
 ├── judgments
 ├── judgment_chunks
 └── judgment_embeddings (vector(768))
```

---

## ▶️ Usage

### Quick Demo (Retrieval Only)

```bash
python demo.py
```

---

### Full RAG System

```bash
python main.py --query "What are the regulations for educational institutions in Karnataka?"
python main.py --interactive
python main.py --test
python main.py --retrieval-test
python main.py --query "educational fees" --debug
```

---

### Advanced Configuration

```bash
python main.py --query "natural justice" --top-k 10 --threshold 0.25
```

---

## 🔧 Configuration

### Retrieval Layer

**Embedding Model:** BAAI/bge-base-en-v1.5 (768-dim)  
**Vector Metric:** Cosine similarity (pgvector)  
**Stage-1 candidate_k:** 30 (configurable)  
**Stage-2 final_k:** 8 (configurable)  
**Reranker Model:** cross-encoder/ms-marco-MiniLM-L-6-v2  

---

### LLM Inference

**Model:** Qwen/Qwen2.5-14B-Instruct  
**Precision:** FP16  
**Max New Tokens:** 512  
**Temperature:** 0.2  
**Sampling:** Disabled  

---

### Prompt Builder

**Max Context Length:** 4000 characters  
**Policy:** Strict RAG — *answer only from retrieved context*

---

## 📊 Database Schema

```sql
judgments (
    id, petitioner, respondent, court, date_of_judgment, 
    bench, citations, judgment_text
)

judgment_chunks (
    chunk_id, judgment_id, section, content
)

judgment_embeddings (
    chunk_id, embedding vector(768)
)
```

Sections:
```
facts | issues | arguments | ratio | judgment
```

---

## 🎯 Design Principles

### Hallucination Prevention
- Strict RAG prompt constraints
- Context-only answering
- Deterministic decoding
- Fallback responses when evidence missing

---

### Citation-First Design
- Every chunk carries legal metadata
- Automatic citation extraction
- Deduplicated source lists

---

### Modular Architecture
- Each layer independently testable
- Easy model swapping
- Retrieval and generation decoupled

---

### Performance Optimized
- HNSW vector index
- Batch embedding
- GPU inference pipeline
- 2-stage precision filtering

---

## 📈 Metrics Tracked

- Retrieval time
- Reranking time
- Generation time
- Confidence score
- Chunk similarity
- Citation coverage

---

## 🔍 Sample Usage

```python
from rag_pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()

result = pipeline.query("What are the principles of natural justice?")

print(result["answer"])
print(result["confidence"])
print(result["sources"])
```

---

## 🛠️ Customization

### Swap Models

```python
pipeline = LegalRAGPipeline(model_name="Qwen/Qwen2.5-7B-Instruct")
```

---

### Tune Retrieval

```python
pipeline = LegalRAGPipeline(top_k=12, similarity_threshold=0.25)
```

---

## 📋 Known Limitations

1. 14B model requires high VRAM
2. Context window limited by prompt size
3. Single-turn queries only
4. English embedding optimized

---

## 🚧 Future Extensions

- Hybrid retrieval (BM25 + vector + reranker)
- Multi-hop retrieval
- Domain fine-tuning
- Embedding caching
- API server layer
- Evaluation benchmark suite

---

## 🔧 Debugging

```bash
python main.py --query "test" --debug
python main.py --retrieval-test --query "educational fees"
python retriever.py
python prompt_builder.py
python post_processor.py
```

---

## 📦 Requirements

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
accelerate>=0.25.0
psycopg2-binary>=2.9.0
numpy>=1.24.0
tqdm>=4.65.0
```

---

## ✅ Design Goals Met

✅ 2-stage retrieval implemented  
✅ Cross-encoder reranking integrated  
✅ Reduced retrieval noise  
✅ Citation-grounded answers  
✅ GPU-optimized inference  
✅ pgvector integration  
✅ Modular + production-ready  

---

The system now runs a **precision-focused 2-stage legal RAG pipeline** suitable for high-quality legal QA and scalable deployment.
