"""LangGraph state."""

from __future__ import annotations

from typing import Any, TypedDict

from app.rag.store import Hit
from app.schemas import Lead, Message


class GraphState(TypedDict, total=False):
    session_id: str
    message: str
    history: list[Message]
    lead: Lead
    stage: str
    asked_slots: list[str]
    user_turns: int
    transcript: str

    intent: str
    hits: list[Hit]
    answer: str
    lead_ask: str
    crm_status: str
    lead_id: str | None
    debug: dict[str, Any]
