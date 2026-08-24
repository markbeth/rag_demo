"""In-memory session storage: history, lead slots, funnel stage."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from app.schemas import Lead, Message, Stage


@dataclass
class Session:
    id: str
    messages: list[Message] = field(default_factory=list)
    lead: Lead = field(default_factory=Lead)
    stage: Stage = "greeting"
    crm_status: str = "pending"
    lead_id: str | None = None
    asked_slots: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)
    # Serialises turns of the same conversation. Without it two concurrent messages
    # both read crm_status == "pending" and the lead is submitted twice.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))  # type: ignore[arg-type]
        self.updated_at = time.time()

    def user_turns(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")

    def transcript(self) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)


SWEEP_INTERVAL_S = 60.0


class SessionStore:
    def __init__(self, ttl_s: int = 6 * 3600, history_max: int = 20) -> None:
        self._sessions: dict[str, Session] = {}
        self._ttl = ttl_s
        self._history_max = history_max
        self._last_sweep = time.time()

    def get_or_create(self, session_id: str | None) -> Session:
        self._evict_expired()
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        new_id = session_id or f"s_{uuid.uuid4().hex[:12]}"
        session = Session(id=new_id)
        self._sessions[new_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def drop(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def trim(self, session: Session) -> None:
        if len(session.messages) > self._history_max:
            session.messages = session.messages[-self._history_max :]

    def _evict_expired(self) -> None:
        """Swept at most once a minute: a full scan on every request buys nothing."""
        now = time.time()
        if now - self._last_sweep < SWEEP_INTERVAL_S:
            return
        self._last_sweep = now
        cutoff = now - self._ttl
        for sid in [
            s for s, sess in self._sessions.items() if sess.updated_at < cutoff
        ]:
            del self._sessions[sid]
