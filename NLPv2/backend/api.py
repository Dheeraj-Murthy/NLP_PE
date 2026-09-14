from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import tempfile
import os

from rag_pipeline import LegalRAGPipeline
from citation_graph import CitationGraphManager

app = FastAPI(
    title="Legal RAG API",
    description="Retrieval-Augmented Generation for legal documents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[LegalRAGPipeline] = None
graph_manager: Optional[CitationGraphManager] = None


@app.on_event("startup")
async def startup_event():
    global pipeline, graph_manager
    pipeline = LegalRAGPipeline(load_llm=True)
    graph_manager = CitationGraphManager()
    graph_manager.load_graph_from_db()


@app.get("/")
async def root():
    return {"status": "ok", "message": "Legal RAG API is running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline.llm is not None if pipeline else False,
    }


@app.post("/query")
async def query_legal(
    query: str = Form(...),
    top_k: int = Form(8),
    threshold: float = Form(0.3),
    include_debug: bool = Form(False),
):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        result = pipeline.query(user_query=query, include_debug_info=include_debug)

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_legal(message: str = Form(...), include_debug: bool = Form(False)):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        result = pipeline.chat(user_message=message, include_debug_info=include_debug)

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/clear")
async def clear_chat():
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    pipeline.clear_chat_history()
    return {"success": True, "message": "Chat history cleared"}


@app.get("/chat/history")
async def get_chat_history():
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    history = pipeline.get_chat_history()
    return {"success": True, "data": history}


@app.post("/document")
async def query_document(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    include_retrieval: bool = Form(True),
    include_debug: bool = Form(False),
):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.query_with_document(
            document_path=tmp_path,
            user_query=query,
            include_retrieval=include_retrieval,
            include_debug_info=include_debug,
        )

        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/retrieval-test")
async def test_retrieval(query: str = Form(...)):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        result = pipeline.test_retrieval_only(query)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    return {"success": True, "data": pipeline.get_system_status()}


@app.get("/graph/judgment/{judgment_id}")
async def get_citation_graph(judgment_id: int, depth: int = 2):
    if not graph_manager:
        raise HTTPException(status_code=500, detail="Graph manager not initialized")

    try:
        subgraph = graph_manager.get_subgraph(judgment_id, depth=depth)
        return {"success": True, "data": subgraph}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/landmark-cases")
async def get_landmark_cases(limit: int = 10):
    if not graph_manager:
        raise HTTPException(status_code=500, detail="Graph manager not initialized")

    try:
        landmarks = graph_manager.get_landmark_cases(limit=limit)
        return {"success": True, "data": landmarks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/path")
async def get_precedent_path(source_id: int, target_id: int):
    if not graph_manager:
        raise HTTPException(status_code=500, detail="Graph manager not initialized")

    try:
        path = graph_manager.get_shortest_path(source_id=source_id, target_id=target_id)
        if path is None:
            return {"success": False, "message": "No citation path found between cases"}
        return {"success": True, "data": {"path": path}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
