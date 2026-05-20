# app/agents/answer.py
"""
Answer Agent — generates the final answer.
Handles 3 cases:
- retrieval: answer from chunks
- memory:    answer from chat history
- chitchat:  friendly response
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from app.config import settings
from app.agents.state import DocuMindState

# ── Prompt for document-based answers ────────────────────────────
RETRIEVAL_PROMPT = PromptTemplate.from_template("""
You are DocuMind AI — a precise document assistant.

CONVERSATION HISTORY:
{chat_history}

RELEVANT DOCUMENT CHUNKS:
{context}

CURRENT QUESTION:
{question}

INSTRUCTIONS:
- Answer using ONLY the document chunks above
- If answer not found: say "I couldn't find this in the documents."
- Keep conversation history in mind for context
- Be concise and professional

ANSWER:
""")

# ── Prompt for memory-based answers ──────────────────────────────
MEMORY_PROMPT = PromptTemplate.from_template("""
You are DocuMind AI. The user is asking about our conversation history.

FULL CONVERSATION HISTORY:
{chat_history}

QUESTION ABOUT HISTORY:
{question}

Answer based on the conversation history above.
ANSWER:
""")

# ── Prompt for chitchat ───────────────────────────────────────────
CHITCHAT_PROMPT = PromptTemplate.from_template("""
You are DocuMind AI — a friendly, professional document assistant.
You help users understand their uploaded PDF documents.

User says: {question}

Respond briefly and helpfully. If they ask what you can do,
explain you can answer questions about uploaded PDF documents.
""")


def answer_agent(state: DocuMindState) -> DocuMindState:
    """
    Picks the right prompt based on state["route"].
    Generates answer with Gemini.
    """
    print(f"\n💬 ANSWER AGENT: Generating answer (route={state['route']})")

    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.2
    )

    route = state.get("route", "retrieval")

    # Format chat history
    history_text = ""
    if state.get("chat_history"):
        pairs = state["chat_history"]
        history_text = "\n".join([
            f"{'User' if i % 2 == 0 else 'AI'}: {msg}"
            for i, msg in enumerate(pairs[-8:])  # last 4 Q&A pairs
        ])

    # ── Pick prompt based on route ────────────────────────────────
    if route == "memory":
        chain = MEMORY_PROMPT | llm
        result = chain.invoke({
            "question": state["question"],
            "chat_history": history_text or "No conversation yet."
        })

    elif route == "chitchat":
        chain = CHITCHAT_PROMPT | llm
        result = chain.invoke({
            "question": state["question"]
        })

    else:  # retrieval
        context = "\n\n---\n\n".join(state.get("retrieved_chunks", []))
        if not context:
            context = "No document chunks retrieved."

        chain = RETRIEVAL_PROMPT | llm
        result = chain.invoke({
            "question": state["question"],
            "context": context,
            "chat_history": history_text or "No previous conversation."
        })

    answer = result.content.strip()
    print(f"   → Answer generated ({len(answer)} chars)")
    return {"answer": answer}