# app/agents/state.py
from typing import TypedDict, Annotated
import operator

class DocuMindState(TypedDict):
    """
    Shared state — every agent reads and writes here.
    Think of it as the whiteboard all agents share.
    
    Annotated[list, operator.add] means:
    when multiple agents write to chat_history,
    LangGraph APPENDS instead of overwriting.
    """
    question: str                              # user's current question
    retrieved_chunks: list[str]                # chunks found by retriever
    answer: str                                # final answer
    route: str                                 # router's decision
    chat_history: Annotated[list, operator.add]# full conversation memory
    sources: list[str]                         # which PDFs were used