# app/agents/retriever.py
"""
Retriever Agent — searches the vector store for relevant chunks.
Uses your existing vector_store.py — no duplication.
"""
from app.agents.state import DocuMindState
from app.vector_store import get_retriever

def retriever_agent(state: DocuMindState) -> DocuMindState:
    """
    Takes question from state.
    Searches FAISS/Pinecone.
    Writes top chunks + sources back to state.
    """
    print(f"\n🔍 RETRIEVER: Searching for '{state['question']}'")

    retriever = get_retriever(top_k=4)

    if retriever is None:
        print("   → No index found")
        return {
            "retrieved_chunks": [],
            "sources": []
        }

    # LangChain retriever returns Document objects
    docs = retriever.invoke(state["question"])

    chunks = [doc.page_content for doc in docs]
    sources = list(set(
        doc.metadata.get("source", "unknown")
        for doc in docs
    ))

    print(f"   → Found {len(chunks)} chunks from: {sources}")
    return {
        "retrieved_chunks": chunks,
        "sources": sources
    }