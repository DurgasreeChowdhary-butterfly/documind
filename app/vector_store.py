# app/vector_store.py
"""
Vector store abstraction.
Automatically uses Pinecone if API key is set, else FAISS.
Both are wrapped in LangChain for a unified interface.
"""
import os
import pickle
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

# ── Embedding model (shared across both stores) ───────────────────
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=settings.GEMINI_API_KEY
)


def _get_pinecone_store(index_name: str):
    """Initialize Pinecone vector store via LangChain."""
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    # Create index if it doesn't exist
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=768,          # Google embedding-001 output size
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Pinecone index '{index_name}' created")

    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings_model
    )


def _get_faiss_store():
    """Load existing FAISS store from disk, or return None if empty."""
    if os.path.exists(settings.FAISS_INDEX_PATH):
        store = FAISS.load_local(
            settings.DATA_DIR,
            embeddings_model,
            index_name="faiss_index",
            allow_dangerous_deserialization=True
        )
        return store
    return None


def add_documents_to_store(chunks: list[dict]) -> int:
    """
    Add chunks to the active vector store.
    Works with both Pinecone and FAISS.
    Returns number of vectors added.
    """
    from langchain_core.documents import Document

    # Convert our chunk dicts to LangChain Documents
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"]
            }
        )
        for chunk in chunks
    ]

    if settings.VECTOR_STORE == "pinecone":
        store = _get_pinecone_store(settings.PINECONE_INDEX_NAME)
        store.add_documents(documents)
        print(f"✅ Added {len(documents)} chunks to Pinecone")

    else:  # FAISS
        existing = _get_faiss_store()
        if existing:
            existing.add_documents(documents)
            existing.save_local(settings.DATA_DIR, index_name="faiss_index")
        else:
            new_store = FAISS.from_documents(documents, embeddings_model)
            new_store.save_local(settings.DATA_DIR, index_name="faiss_index")
        print(f"✅ Added {len(documents)} chunks to FAISS")

    return len(documents)


def delete_documents_by_source(source_name: str):
    """
    Remove all chunks belonging to a specific PDF.
    Pinecone supports metadata filtering.
    FAISS: rebuild index without that source.
    """
    if settings.VECTOR_STORE == "pinecone":
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        # Delete by metadata filter
        index.delete(filter={"source": {"$eq": source_name}})
        print(f"✅ Deleted '{source_name}' from Pinecone")

    else:  # FAISS — rebuild without deleted source
        store = _get_faiss_store()
        if store is None:
            return
        # Get all docs, filter out the deleted source
        all_docs = list(store.docstore._dict.values())
        remaining = [d for d in all_docs if d.metadata.get("source") != source_name]
        if remaining:
            new_store = FAISS.from_documents(remaining, embeddings_model)
            new_store.save_local(settings.DATA_DIR, index_name="faiss_index")
        else:
            # No docs left — remove index files
            for f in ["faiss_index.faiss", "faiss_index.pkl"]:
                path = os.path.join(settings.DATA_DIR, f)
                if os.path.exists(path):
                    os.remove(path)
        print(f"✅ Deleted '{source_name}' from FAISS")


def get_retriever(top_k: int = None):
    """
    Returns a LangChain retriever for use in RetrievalQA chain.
    This is the key LangChain integration.
    """
    k = top_k or settings.TOP_K_RESULTS

    if settings.VECTOR_STORE == "pinecone":
        store = _get_pinecone_store(settings.PINECONE_INDEX_NAME)
    else:
        store = _get_faiss_store()
        if store is None:
            return None

    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def get_store_stats() -> dict:
    """Return basic stats about the vector store."""
    if settings.VECTOR_STORE == "pinecone":
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(settings.PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            return {
                "store": "pinecone",
                "total_vectors": stats.total_vector_count,
                "dimension": 768
            }
        except Exception:
            return {"store": "pinecone", "total_vectors": 0}
    else:
        store = _get_faiss_store()
        total = store.index.ntotal if store else 0
        return {
            "store": "faiss",
            "total_vectors": total,
            "dimension": 768
        }
