import pytest

from app.config import Settings
from app.graph.graph import build_graph
from app.graph.nodes import Nodes
from app.rag.loader import load_chunks, load_playbook
from app.rag.store import HybridStore
from app.services.chat import ChatService
from app.services.sessions import SessionStore


class FakeLLM:
    """Подменяет провайдера: отдаёт заранее заданные ответы по типу промпта."""

    def __init__(self, extract: dict | None = None, answer: str = "Ответ по базе знаний.") -> None:
        self.extract_result = extract or {}
        self.answer = answer
        self.calls: list[list[dict]] = []

    async def chat(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        self.calls.append(list(messages))
        content = messages[-1]["content"]
        if "Нужно добавить к ответу ОДНУ короткую фразу" in content:
            return "Оставьте email — пришлю сравнение тарифов."
        if "Заявка передана партнёру" in content:
            return "Спасибо, передал заявку партнёру."
        return self.answer

    async def chat_json(self, messages, *, temperature=0.0):
        self.calls.append(list(messages))
        return dict(self.extract_result)

    async def chat_stream(self, messages, *, temperature=None, max_tokens=None):
        for token in self.answer.split(" "):
            yield token + " "

    async def embed(self, texts):
        raise AssertionError("В тестах эмбеддинги не используются")

    async def aclose(self):
        return None


class FakeCrm:
    def __init__(self) -> None:
        self.submitted: list[tuple[str, object]] = []

    async def submit(self, session_id, lead, transcript):
        self.submitted.append((session_id, lead))
        return f"lead_test_{len(self.submitted)}"

    async def list_leads(self, limit=50):
        return []


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
    def _make(llm: FakeLLM, crm: FakeCrm | None = None):
        crm = crm or FakeCrm()
        nodes = Nodes(llm, store, playbook, crm, top_k=3)
        service = ChatService(build_graph(nodes), nodes, SessionStore())
        return service, crm

    return _make
