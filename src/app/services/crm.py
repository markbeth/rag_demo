"""Mock CRM. The interface is deliberately narrow so HubSpot/amoCRM is a one-class swap."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from app.config import RUNTIME_DIR
from app.schemas import Lead


class CrmSink(Protocol):
    async def submit(self, session_id: str, lead: Lead, transcript: str) -> str: ...
    async def list_leads(self, limit: int = 50) -> list[dict]: ...


class JsonlCrm:
    """Appends leads to an append-only jsonl file. The lock guards concurrent sessions."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (RUNTIME_DIR / "leads.jsonl")
        self._lock = asyncio.Lock()

    async def submit(self, session_id: str, lead: Lead, transcript: str) -> str:
        lead_id = f"lead_{uuid.uuid4().hex[:10]}"
        record = {
            "lead_id": lead_id,
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "source": "web-chat",
            **lead.model_dump(exclude={"refusals"}),
            "transcript": transcript[-4000:],
        }
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(_append_line, self.path, line)
        return lead_id

    async def list_leads(self, limit: int = 50) -> list[dict]:
        """Newest first. The file is streamed and only `limit` records are held."""
        if not self.path.exists():
            return []
        records = await asyncio.to_thread(_read_tail, self.path, limit)
        return records[::-1]


def _read_tail(path: Path, limit: int) -> list[dict]:
    tail: deque[dict] = deque(maxlen=limit)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                tail.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(tail)


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
