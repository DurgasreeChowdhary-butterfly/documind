# app/rag_chain.py
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings
from app.vector_store import get_retriever


def get_llm():
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.2,
        convert_system_message_to_human=True
    )


RAG_PROMPT_TEMPLATE = """You are DocuMind AI — a precise, intelligent document assistant.
Your job is to answer questions strictly from the provided document context.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer ONLY using information from the context above
- If the answer is not found in the context, respond exactly:
  "I couldn't find this information in the uploaded documents."
- Be concise, clear, and professional
- If quoting directly, mention the source document
- Never hallucinate or use outside knowledge

ANSWER:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def ask_question(question: str, top_k: int = None) -> dict:
    retriever = get_retriever(top_k=top_k)
    if retriever is None:
        return {
            "answer": "No documents indexed yet. Please upload a PDF first.",
            "sources": [],
            "model": settings.GEMINI_MODEL,
            "chunks_used": 0
        }

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = RAG_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = list(set(doc.metadata.get("source", "unknown") for doc in docs))

    return {
        "answer": answer,
        "sources": sources,
        "model": settings.GEMINI_MODEL,
        "chunks_used": len(docs)
    }
