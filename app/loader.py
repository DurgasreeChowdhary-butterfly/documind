# app/loader.py
import fitz  # PyMuPDF
import os
import pickle
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file page by page."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def chunk_text(text: str, source_name: str) -> list[dict]:
    """
    Split text into overlapping chunks.
    Each chunk is a dict with text + metadata.
    Metadata is used by LangChain and Pinecone for filtering.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    texts = splitter.split_text(text)

    chunks = [
        {
            "text": t,
            "source": source_name,
            "chunk_index": i
        }
        for i, t in enumerate(texts)
    ]
    return chunks


def load_pdf_registry() -> dict:
    """Load the registry of all indexed PDFs."""
    if os.path.exists(settings.PDF_REGISTRY_PATH):
        with open(settings.PDF_REGISTRY_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_pdf_registry(registry: dict):
    """Save the registry of all indexed PDFs."""
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    with open(settings.PDF_REGISTRY_PATH, "wb") as f:
        pickle.dump(registry, f)


def register_pdf(filename: str, num_chunks: int, file_size_kb: float):
    """Add a PDF entry to the registry."""
    registry = load_pdf_registry()
    registry[filename] = {
        "filename": filename,
        "num_chunks": num_chunks,
        "file_size_kb": round(file_size_kb, 2),
        "uploaded_at": datetime.now().isoformat(),
        "status": "indexed"
    }
    save_pdf_registry(registry)


def unregister_pdf(filename: str):
    """Remove a PDF from the registry."""
    registry = load_pdf_registry()
    if filename in registry:
        del registry[filename]
        save_pdf_registry(registry)
