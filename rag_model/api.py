from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import tempfile
import os

from rag_pipeline import LegalRAGPipeline

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


@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = LegalRAGPipeline(load_llm=True)


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
