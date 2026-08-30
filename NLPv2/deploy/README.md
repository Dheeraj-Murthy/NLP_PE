# Deploy Quickstart

## 1. Database Setup

```bash
# Initialize PostgreSQL with pgvector
./deploy/db_setup/init_db.sh

# Environment variables (or set in .env)
export DB_HOST=your_postgres_host
export DB_PORT=5432
export DB_NAME=legal_rag
export DB_USER=postgres
export DB_PASSWORD=your_password
```

## 2. Ingest PDFs

```bash
./deploy/db_setup/ingest_pdfs.sh /path/to/judgment_pdfs [workers]

# Example:
./deploy/db_setup/ingest_pdfs.sh ./my_pdfs 8
```

## 3. Start Backend API

```bash
cd deploy/run_model
cp .env.example .env
# Edit .env with your DB credentials

./start_api.sh 8000
# API runs at http://localhost:8000/docs
```

## 4. Start Frontend

```bash
cd deploy/run_model
# Edit .env with your API URL

./start_frontend.sh 3000
# Frontend runs at http://localhost:3000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DB_HOST | localhost | PostgreSQL host |
| DB_PORT | 5432 | PostgreSQL port |
| DB_NAME | legal_rag | Database name |
| DB_USER | postgres | DB username |
| DB_PASSWORD | postgres | DB password |
| API_PORT | 8000 | API server port |
| API_URL | http://localhost:8000 | Backend URL for frontend |

## RunPod Notes

- Ensure PostgreSQL has pgvector extension
- GPU required for Qwen2.5 model (7B needs ~14GB VRAM)
- Or use `--retrieval-test` for vector search only (no GPU)

---

## Docker Deployment

```bash
cd deploy/docker

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| postgres | 5432 | PostgreSQL with pgvector |
| api | 8000 | FastAPI backend |
| frontend | 3000 | Next.js UI |

### With GPU (for LLM)

Build api image with GPU support, then run:
```bash
docker-compose up -d postgres
docker run --gpus all -d \
  --network deploy_docker_default \
  -e DB_HOST=postgres \
  -e DB_NAME=legal_rag \
  -v $(pwd)/../..:/app \
  your-gpu-api-image
```