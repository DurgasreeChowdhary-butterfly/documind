# main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.config import settings
from app.loader import (
    extract_text_from_pdf, chunk_text,
    load_pdf_registry, register_pdf, unregister_pdf
)
from app.vector_store import (
    add_documents_to_store,
    delete_documents_by_source,
    get_store_stats
)
from app.rag_chain import ask_question


# ── Startup ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs("static", exist_ok=True)
    print(f"🚀 DocuMind AI started")
    print(f"   Vector Store : {settings.VECTOR_STORE.upper()}")
    print(f"   LLM Model    : {settings.GEMINI_MODEL}")
    yield
    print("👋 DocuMind AI shutting down")


app = FastAPI(
    title="DocuMind AI",
    description="Production-grade Multi-PDF RAG Chatbot",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Serve UI ──────────────────────────────────────────────────────
@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


# ── Health Check ──────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    stats = get_store_stats()
    registry = load_pdf_registry()
    return {
        "status": "healthy",
        "vector_store": stats["store"],
        "total_vectors": stats["total_vectors"],
        "total_pdfs": len(registry),
        "model": settings.GEMINI_MODEL
    }


# ── Upload PDF ────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Validate type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted.")

    # Validate size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"File too large. Max: {settings.MAX_FILE_SIZE_MB}MB")

    # Check for duplicate
    registry = load_pdf_registry()
    if file.filename in registry:
        raise HTTPException(409, f"'{file.filename}' is already indexed. Delete it first to re-upload.")

    # Save to disk
    pdf_path = os.path.join(settings.DATA_DIR, file.filename)
    with open(pdf_path, "wb") as f:
        f.write(content)

    try:
        # Extract + chunk
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            os.remove(pdf_path)
            raise HTTPException(422, "Could not extract text from PDF. Is it scanned/image-only?")

        chunks = chunk_text(raw_text, source_name=file.filename)

        # Store in vector DB
        add_documents_to_store(chunks)

        # Register
        register_pdf(file.filename, len(chunks), size_mb * 1024)

        return {
            "message": f"'{file.filename}' indexed successfully!",
            "filename": file.filename,
            "chunks": len(chunks),
            "size_kb": round(size_mb * 1024, 2),
            "vector_store": settings.VECTOR_STORE
        }

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise HTTPException(500, f"Indexing failed: {str(e)}")


# ── List PDFs ─────────────────────────────────────────────────────
@app.get("/api/pdfs")
async def list_pdfs():
    registry = load_pdf_registry()
    return {
        "pdfs": list(registry.values()),
        "total": len(registry)
    }


# ── Delete PDF ────────────────────────────────────────────────────
@app.delete("/api/pdfs/{filename}")
async def delete_pdf(filename: str):
    registry = load_pdf_registry()
    if filename not in registry:
        raise HTTPException(404, f"'{filename}' not found in index.")

    try:
        # Remove from vector store
        delete_documents_by_source(filename)

        # Remove file from disk
        pdf_path = os.path.join(settings.DATA_DIR, filename)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        # Remove from registry
        unregister_pdf(filename)

        return {"message": f"'{filename}' deleted successfully."}

    except Exception as e:
        raise HTTPException(500, f"Deletion failed: {str(e)}")


# ── Chat ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    registry = load_pdf_registry()
    if not registry:
        raise HTTPException(400, "No PDFs indexed. Upload a document first.")

    try:
        result = ask_question(request.question, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(500, f"Chat error: {str(e)}")

# Add this import at the top of main.py
from app.graph import run_graph

# Add this new endpoint — keep /api/chat as-is
class AgentChatRequest(BaseModel):
    question: str
    chat_history: list = []   # frontend sends history each time

@app.post("/api/agent-chat")
async def agent_chat(request: AgentChatRequest):
    """
    Multi-agent chat endpoint.
    Uses LangGraph — replaces single-chain /api/chat.
    """
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    registry = load_pdf_registry()
    if not registry:
        raise HTTPException(400, "No PDFs indexed. Upload a document first.")

    try:
        result = run_graph(
            question=request.question,
            chat_history=request.chat_history
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Agent error: {str(e)}")