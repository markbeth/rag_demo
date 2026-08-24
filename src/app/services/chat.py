"""Application layer: ties sessions, the graph and streaming together for the API."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from app.graph.nodes import ANSWER_MAX_TOKENS, Nodes, compose_answer, should_submit
from app.graph.state import GraphState
from app.llm.client import LLMError
from app.rag.store import Hit
from app.schemas import ChatResponse, Lead, Source
from app.services.sessions import Session, SessionStore

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, graph, nodes: Nodes, sessions: SessionStore) -> None:
        self._graph = graph
        self._nodes = nodes
        self._sessions = sessions

    # --- buffered response -----------------------------------------------

    async def respond(self, message: str, session_id: str | None) -> ChatResponse:
        session = self._sessions.get_or_create(session_id)
        async with session.lock:
            state = self._initial_state(session, message)
            result: GraphState = await self._graph.ainvoke(state)
            return self._commit(session, message, result)

    # --- streaming -------------------------------------------------------

    async def stream(self, message: str, session_id: str | None) -> AsyncIterator[str]:
        """SSE events: meta -> sources -> token* -> done.

        The node order is unrolled by hand here because LangGraph emits a node at a
        time, while the client needs generation tokens as they arrive. Keep this in
        step with the graph in app/graph/graph.py.
        """
        session = self._sessions.get_or_create(session_id)
        # The lock is held across the yields, so a second message in the same chat
        # waits for this turn instead of racing it.
        async with session.lock:
            state = self._initial_state(session, message)
            yield _sse("meta", {"session_id": session.id})

            try:
                hits = await self._before_answer(state)
                yield _sse(
                    "sources", {"sources": [s.model_dump() for s in _to_sources(hits)]}
                )

                tokens: list[str] = []
                async for token in self._nodes.llm.chat_stream(
                    self._nodes.answer_messages(state), max_tokens=ANSWER_MAX_TOKENS
                ):
                    tokens.append(token)
                    yield _sse("token", {"text": token})
                state["answer"] = "".join(tokens).strip()

                await self._after_answer(state)
                if ask := state.get("lead_ask"):
                    yield _sse("token", {"text": f"\n\n{ask}"})
            except LLMError as exc:
                logger.error("Streaming aborted: %s", exc)
                yield _sse(
                    "error",
                    {"message": "LLM provider is unavailable, please try again."},
                )
                return

            response = self._commit(session, message, state)
            yield _sse("done", response.model_dump(exclude={"sources"}))

    async def _before_answer(self, state: GraphState) -> list[Hit]:
        """route -> (extract_lead || retrieve), the two being independent."""
        state.update(await self._nodes.route(state))
        extracted, retrieved = await asyncio.gather(
            self._nodes.extract_lead(state), self._nodes.retrieve(state)
        )
        state.update(extracted)
        state.update(retrieved)
        return state.get("hits", [])

    async def _after_answer(self, state: GraphState) -> None:
        """lead_strategy -> [crm_submit]."""
        state.update(await self._nodes.lead_strategy(state))
        if should_submit(state) == "crm_submit":
            state.update(await self._nodes.crm_submit(state))

    # --- shared ----------------------------------------------------------

    def _initial_state(self, session: Session, message: str) -> GraphState:
        return {
            "session_id": session.id,
            "message": message,
            "history": list(session.messages),
            "lead": session.lead,
            "stage": session.stage,
            "asked_slots": list(session.asked_slots),
            "user_turns": session.user_turns() + 1,
            "transcript": session.transcript(),
            "crm_status": session.crm_status,
        }

    def _commit(
        self, session: Session, message: str, result: GraphState
    ) -> ChatResponse:
        """Folds the graph result into the session and renders the API response."""
        answer = compose_answer(result.get("answer", ""), result.get("lead_ask", ""))
        session.add("user", message)
        session.add("assistant", answer)
        lead = result.get("lead")
        if isinstance(lead, Lead):
            session.lead = lead
        session.stage = result.get("stage", session.stage)  # type: ignore[assignment]
        session.asked_slots = result.get("asked_slots", session.asked_slots)
        session.crm_status = result.get("crm_status", session.crm_status)
        session.lead_id = result.get("lead_id") or session.lead_id
        self._sessions.trim(session)
        return ChatResponse(
            session_id=session.id,
            answer=answer,
            sources=_to_sources(result.get("hits", [])),
            lead=session.lead,
            stage=session.stage,  # type: ignore[arg-type]
            crm_status=session.crm_status,  # type: ignore[arg-type]
            lead_id=session.lead_id,
        )


def _to_sources(hits: list[Hit]) -> list[Source]:
    return [
        Source(
            id=hit.chunk.id,
            title=hit.chunk.title,
            source=hit.chunk.source,
            score=round(hit.score, 4),
            snippet=hit.chunk.text[:280],
        )
        for hit in hits
    ]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
