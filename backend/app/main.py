"""FastAPI entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_chat, routes_meta
from app.config import get_settings
from app.deps import build_container

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.container = await build_container()
    try:
        yield
    finally:
        await app.state.container.aclose()


settings = get_settings()
app = FastAPI(
    title="Family Office RAG Chat",
    version="0.1.0",
    description="RAG chatbot for family office services with lead capture into a CRM",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routes_chat.router)
app.include_router(routes_meta.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "familyoffice-rag", "docs": "/docs", "health": "/api/health"}
