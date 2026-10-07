# Legal RAG — NLP System for Indian Case Law

A RAG system for searching and asking questions over Indian Supreme Court
judgments, using local LLM inference and vector search — no external API
calls, everything runs on your own GPU.

Started as a Project Elective under Prof. Tulika at IIIT Bangalore. The
corpus covers 50,000+ Supreme Court judgments spanning 75+ years. We also
ran a survey with students, professors, and alumni from NLSIU Bangalore and
NUSRL Ranchi to sanity-check the idea — results are in
[`AI Legal Assistant – User Assessment Survey(1-217).xlsx`](<./AI%20Legal%20Assistant%20%E2%80%93%20User%20Assessment%20Survey(1-217).xlsx>).

## How it works

```
query → BGE embedding + BM25
      → hybrid retrieval (pgvector cosine + full-text search, fused via RRF)
      → cross-encoder reranking (top 8)
      → Qwen2.5-7B-Instruct, grounded prompt with bracketed citations [1] [2]
      → answer + citations + confidence score
```

Two-stage retrieval: a wide, high-recall first pass (dense + keyword search
fused with Reciprocal Rank Fusion) followed by cross-encoder reranking to
pick the best 8 chunks. The LLM is instructed to only answer from those
chunks and cite them — if nothing relevant comes back, it says so instead of
making something up.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI |
| LLM | Qwen2.5-7B-Instruct-1M (~16GB VRAM) |
| Embeddings | BAAI/bge-base-en-v1.5 (768d) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Database | PostgreSQL + pgvector (HNSW) |
| Frontend | Streamlit |
| Deployment | Docker Compose |

## Running it

```bash
docker compose -f deploy/docker/docker-compose.yml up -d
```

Spins up Postgres (`5433`), the API (`8000`), and the Streamlit frontend
(`3000`). Or set it up manually — from `NLPv2/`, with a venv active:

```bash
source venv/bin/activate                              # or create one: python3 -m venv venv

bash deploy/db_setup/init_db.sh                       # database + schema
python ingestion/ingest.py --input /path/to/pdfs      # ingest judgments

cd backend
python api.py                                          # API at :8000, docs at /docs
```

From another terminal (same venv), start the frontend:

```bash
cd NLPv2/frontend/streamlit
pip install -r requirements.txt
RAG_API_URL=http://localhost:8000 streamlit run Home.py --server.port 8501 --server.address 0.0.0.0
```

Or, on a server where the venv already exists, `run.sh` does both of the
above as backgrounded processes in one command:

```bash
./run.sh start      # git pull, install deps if needed, start API + frontend
./run.sh stop        # stop both
./run.sh restart      # stop then start
./run.sh status       # check what's running, with PIDs
```

(`./run.sh` with no argument defaults to `start`.) Logs land in `logs/`,
PID files in `.run/`.

Try a query from the CLI instead:

```bash
cd backend && python main.py --query "What are the principles of natural justice?"
```

Or from Python:

```python
from rag_pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()
result = pipeline.query("What is due process in administrative law?")
print(result["answer"], result["citations"], result["confidence"])
```

## What's there / what's next

Working: hybrid retrieval, reranking, grounded generation, citation graph
(precedent networks + landmark cases), OCR on uploaded documents, chat with
memory, Streamlit frontend, Docker deployment.

In progress: Karnataka High Court scraper, LegalParam model evaluation.

On the roadmap: petition generation, fine-tuning a smaller legal model,
regional-language support, more High Courts, an automated eval benchmark,
and general production hardening (there's no CI/test suite yet).

## Disclaimer

Research tool, not legal advice. Verify every citation against the original
judgment before relying on it.

## Contributors

- M S Dheeraj Murthy
- Mathew Joseph
- Ayush Tiwari