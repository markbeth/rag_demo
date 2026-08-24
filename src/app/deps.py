"""Application wiring (plain manual DI through app.state)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.graph.graph import build_graph
from app.graph.nodes import Nodes
from app.llm.client import LLMClient
from app.rag.loader import load_chunks, load_playbook
from app.rag.store import HybridStore
from app.services.chat import ChatService
from app.services.crm import JsonlCrm
from app.services.sessions import SessionStore

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    llm: LLMClient
    store: HybridStore
    sessions: SessionStore
    crm: JsonlCrm
    chat: ChatService

    async def aclose(self) -> None:
        await self.llm.aclose()


async def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()
    llm = LLMClient(settings)
    chunks = load_chunks()
    store = HybridStore(chunks)
    playbook = load_playbook()

    if settings.use_embeddings and settings.llm_configured:
        await store.build_vectors(llm)
    else:
        logger.info("Embeddings disabled or no API key: falling back to lexical search")

    crm = JsonlCrm()
    sessions = SessionStore(
        ttl_s=settings.session_ttl_s, history_max=settings.history_max_messages
    )
    nodes = Nodes(llm, store, playbook, crm, top_k=settings.retriever_top_k)
    chat = ChatService(build_graph(nodes), nodes, sessions)
    logger.info("Knowledge base loaded: %s chunks", store.size)
    return Container(settings, llm, store, sessions, crm, chat)


def container(request: Request) -> Container:
    return request.app.state.container


Deps = Annotated[Container, Depends(container)]
