"""Chat: buffered response and SSE streaming."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.deps import Container, Deps
from app.llm.client import LLMError
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def with_llm(deps: Deps) -> Container:
    """Both chat routes are useless without a provider key, so the guard lives here."""
    if not deps.settings.llm_configured:
        raise HTTPException(503, "OPENAI_API_KEY is not set: add it to src/.env")
    return deps


LlmDeps = Annotated[Container, Depends(with_llm)]


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, deps: LlmDeps) -> ChatResponse:
    try:
        return await deps.chat.respond(payload.message, payload.session_id)
    except LLMError as exc:
        raise HTTPException(502, f"LLM provider unavailable: {exc}") from exc


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, deps: LlmDeps):
    return StreamingResponse(
        deps.chat.stream(payload.message, payload.session_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
