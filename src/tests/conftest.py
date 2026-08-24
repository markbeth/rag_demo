"""Shared fixtures. The LLM provider is faked, so no test touches the network."""

import pytest

from app.config import Settings
from app.graph.graph import build_graph
from app.graph.nodes import Nodes
from app.rag.loader import load_chunks, load_playbook
from app.rag.store import HybridStore
from app.services.chat import ChatService
from app.services.sessions import SessionStore
from tests.fakes import FakeCrm, FakeLLM


@pytest.fixture
def settings() -> Settings:
    return Settings(OPENAI_API_KEY="test", USE_EMBEDDINGS=False)


@pytest.fixture
def store() -> HybridStore:
    return HybridStore(load_chunks())


@pytest.fixture
def playbook() -> dict:
    return load_playbook()


@pytest.fixture
def make_chat(store, playbook):
    """Builds a ChatService wired to the fakes, returning it with the CRM double."""

    def _make(llm: FakeLLM, crm: FakeCrm | None = None) -> tuple[ChatService, FakeCrm]:
        crm = crm or FakeCrm()
        nodes = Nodes(llm, store, playbook, crm, top_k=3)
        return ChatService(build_graph(nodes), nodes, SessionStore()), crm

    return _make
