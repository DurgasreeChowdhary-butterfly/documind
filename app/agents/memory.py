# app/agents/memory.py
"""
Memory Agent — saves Q&A to chat history.
Always runs LAST — after answer is generated.
Simple but powerful: enables multi-turn conversation.
"""
from app.agents.state import DocuMindState

def memory_agent(state: DocuMindState) -> DocuMindState:
    """
    Appends current Q&A to chat_history.
    Because chat_history uses operator.add in State,
    this APPENDS — never overwrites previous history.
    """
    print(f"\n🧠 MEMORY: Saving to history")

    new_entries = [
        state["question"],      # user message
        state["answer"]         # AI response
    ]

    print(f"   → History now has "
          f"{len(state.get('chat_history', [])) + 2} entries")

    # Return new entries — LangGraph auto-appends via operator.add
    return {"chat_history": new_entries}