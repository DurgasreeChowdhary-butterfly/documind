# app/graph.py
"""
The LangGraph graph — connects all agents.

Flow:
router → [retriever OR memory OR chitchat] → answer → memory_save → END
          ↑ conditional edge                  ↑ fixed edges
"""
from langgraph.graph import StateGraph, END
from app.agents.state import DocuMindState
from app.agents.router import router_agent, decide_next_agent
from app.agents.retriever import retriever_agent
from app.agents.answer import answer_agent
from app.agents.memory import memory_agent


def build_graph():
    """
    Builds and compiles the LangGraph multi-agent graph.
    Call once at startup, reuse for every question.
    """

    # ── 1. Initialize graph with our State ───────────────────────
    graph = StateGraph(DocuMindState)

    # ── 2. Add all agent nodes ────────────────────────────────────
    graph.add_node("router",        router_agent)
    graph.add_node("retriever",     retriever_agent)
    graph.add_node("generator",     answer_agent)
    graph.add_node("memory_save",   memory_agent)

    # ── 3. Set entry point ────────────────────────────────────────
    graph.set_entry_point("router")

    # ── 4. Conditional edges from router ─────────────────────────
    graph.add_conditional_edges(
        "router",
        decide_next_agent,
        {
            "retrieval": "retriever",
            "memory":    "generator",
            "chitchat":  "generator"
        }
    )

    # ── 5. Fixed edges ────────────────────────────────────────────
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "memory_save")

    # After memory_save → END
    graph.add_edge("memory_save", END)

    # ── 6. Compile ────────────────────────────────────────────────
    return graph.compile()


# Build once at module load — reused for all requests
documind_graph = build_graph()


def run_graph(question: str, chat_history: list = None) -> dict:
    """
    Main entry point — called by FastAPI.
    Runs the full multi-agent graph for one question.
    """
    initial_state = {
        "question": question,
        "retrieved_chunks": [],
        "answer": "",
        "route": "",
        "chat_history": chat_history or [],
        "sources": []
    }

    # Run the full graph
    final_state = documind_graph.invoke(initial_state)

    return {
        "answer":       final_state["answer"],
        "route":        final_state["route"],
        "sources":      final_state.get("sources", []),
        "chat_history": final_state["chat_history"]
    }