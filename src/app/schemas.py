"""Pydantic schemas of the public API."""

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant", "system"]
Stage = Literal[
    "greeting", "discovery", "qualifying", "collecting", "submitted", "declined"
]


class Message(BaseModel):
    role: Role
    content: str


class Source(BaseModel):
    id: str
    title: str
    source: str
    score: float
    snippet: str


class Lead(BaseModel):
    """Slots the bot fills in over the course of the conversation."""

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    preferred_time: str | None = None
    capital_range: str | None = None
    interest: str | None = None
    tier_interest: str | None = None
    notes: str | None = None
    refusals: int = 0

    @property
    def has_channel(self) -> bool:
        return bool(self.email or self.phone)

    @property
    def crm_ready(self) -> bool:
        return bool(self.name) and self.has_channel

    def missing_slots(self) -> list[str]:
        order = ("name", "email", "phone", "preferred_time")
        return [slot for slot in order if not getattr(self, slot)]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source] = []
    lead: Lead = Lead()
    stage: Stage = "discovery"
    crm_status: Literal["pending", "submitted", "failed"] = "pending"
    lead_id: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    messages: list[Message]
    lead: Lead
    stage: Stage
    crm_status: str


class HealthResponse(BaseModel):
    status: str
    model: str
    embeddings: bool
    llm_configured: bool
    kb_chunks: int


class SearchHit(BaseModel):
    id: str
    title: str
    source: str
    score: float
    text: str
