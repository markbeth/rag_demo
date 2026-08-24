"""Health, sessions, leads and retriever debugging."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.deps import Deps
from app.schemas import HealthResponse, SearchHit, SessionResponse

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health", response_model=HealthResponse)
async def health(deps: Deps) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=deps.settings.llm_model,
        embeddings=deps.store.has_vectors,
        llm_configured=deps.settings.llm_configured,
        kb_chunks=deps.store.size,
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, deps: Deps) -> SessionResponse:
    session = deps.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found or expired")
    return SessionResponse(
        session_id=session.id,
        messages=session.messages,
        lead=session.lead,
        stage=session.stage,
        crm_status=session.crm_status,
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def drop_session(session_id: str, deps: Deps) -> None:
    if not deps.sessions.drop(session_id):
        raise HTTPException(404, "Session not found")


@router.get("/leads")
async def leads(deps: Deps, limit: Annotated[int, Query(ge=1, le=500)] = 50):
    return {"items": await deps.crm.list_leads(limit)}


@router.get("/kb/search", response_model=list[SearchHit])
async def kb_search(
    deps: Deps,
    q: Annotated[str, Query(min_length=2)],
    k: Annotated[int, Query(ge=1, le=20)] = 4,
    include_internal: bool = False,
) -> list[SearchHit]:
    hits = await deps.store.search(
        q, llm=deps.llm, k=k, include_internal=include_internal
    )
    return [
        SearchHit(
            id=hit.chunk.id,
            title=hit.chunk.title,
            source=hit.chunk.source,
            score=round(hit.score, 4),
            text=hit.chunk.text,
        )
        for hit in hits
    ]
