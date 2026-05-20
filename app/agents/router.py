# app/agents/router.py
"""
Router Agent — reads the question and decides:
- "retrieval"  → question needs searching the PDFs
- "memory"     → question refers to conversation history
- "chitchat"   → greeting or off-topic question
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from app.config import settings
from app.agents.state import DocuMindState

# Lightweight prompt — just needs to classify, not answer
ROUTER_PROMPT = PromptTemplate.from_template("""
You are a routing assistant. Classify the user's question into ONE of these categories:

1. "retrieval"  — needs searching uploaded documents
   Examples: "What is the leave policy?", "Summarize the PDF", "Find clauses about payment"

2. "memory"     — refers to previous conversation  
   Examples: "What did I ask before?", "Repeat your last answer", "What were we discussing?"

3. "chitchat"   — greeting, thanks, or off-topic
   Examples: "Hello", "Thanks!", "Who are you?", "What can you do?"

Conversation history so far:
{chat_history}

Current question: {question}

Respond with ONLY one word: retrieval, memory, or chitchat
""")

def router_agent(state: DocuMindState) -> DocuMindState:
    """
    Reads the question → decides which agent handles it.
    Writes decision to state["route"].
    """
    print(f"\n🔀 ROUTER: Processing '{state['question']}'")

    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0   # deterministic — we want consistent routing
    )

    # Format chat history for context
    history_text = ""
    if state.get("chat_history"):
        last_3 = state["chat_history"][-6:]  # last 3 Q&A pairs
        history_text = "\n".join([
            f"{'User' if i % 2 == 0 else 'AI'}: {msg}"
            for i, msg in enumerate(last_3)
        ])

    chain = ROUTER_PROMPT | llm
    result = chain.invoke({
        "question": state["question"],
        "chat_history": history_text or "No history yet."
    })

    route = result.content.strip().lower()

    # Safety net — if Gemini returns unexpected value, default to retrieval
    if route not in ["retrieval", "memory", "chitchat"]:
        route = "retrieval"

    print(f"   → Route decided: {route.upper()}")
    return {"route": route}


def decide_next_agent(state: DocuMindState) -> str:
    """
    This function is called by LangGraph's conditional edge.
    Returns the NAME of the next node to visit.
    """
    return state["route"]