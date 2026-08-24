"""Chat: buffered response and SSE streaming."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.deps import Container, container
from app.llm.client import LLMError
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, deps: Container = Depends(container)) -> ChatResponse:
    if not deps.settings.llm_configured:
        raise HTTPException(503, "OPENAI_API_KEY is not set: add it to backend/.env")
    try:
        return await deps.chat.respond(payload.message, payload.session_id)
    except LLMError as exc:
        raise HTTPException(502, f"LLM provider unavailable: {exc}") from exc


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, deps: Container = Depends(container)):
    if not deps.settings.llm_configured:
        raise HTTPException(503, "OPENAI_API_KEY is not set: add it to backend/.env")
    return StreamingResponse(
        deps.chat.stream(payload.message, payload.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
