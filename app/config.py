# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # ── LLM ────────────────────────────────────────────────────────
    GEMINI_API_KEY: str     = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str       = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # ── VECTOR STORE ───────────────────────────────────────────────
    PINECONE_API_KEY: str   = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME:str = os.getenv("PINECONE_INDEX_NAME", "documind")

    # Auto-select: use Pinecone if key exists, else FAISS
    @property
    def VECTOR_STORE(self) -> str:
        if self.PINECONE_API_KEY:
            return "pinecone"
        return "faiss"

    # ── CHUNKING ───────────────────────────────────────────────────
    CHUNK_SIZE: int         = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int      = int(os.getenv("CHUNK_OVERLAP", 100))
    TOP_K_RESULTS: int      = int(os.getenv("TOP_K_RESULTS", 4))
    MAX_FILE_SIZE_MB: int   = int(os.getenv("MAX_FILE_SIZE_MB", 50))

    # ── PATHS ──────────────────────────────────────────────────────
    DATA_DIR: str           = "data"
    FAISS_INDEX_PATH: str   = "data/faiss_index.bin"
    CHUNKS_PATH: str        = "data/chunks.pkl"
    PDF_REGISTRY_PATH: str  = "data/pdf_registry.pkl"

settings = Settings()
