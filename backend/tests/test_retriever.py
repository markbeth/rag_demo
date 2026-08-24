import pytest

from app.rag.store import HybridStore, tokenize


async def top_id(store: HybridStore, query: str) -> str:
    hits = await store.search(query, llm=None, k=3)
    assert hits, f"nothing retrieved for query: {query}"
    return hits[0].chunk.id


def test_chunks_loaded(store: HybridStore):
    ids = {chunk.id for chunk in store.chunks}
    assert {"tier-essential", "tier-private", "tier-bespoke", "company"} <= ids
    assert store.size > 15


def test_tokenize_drops_stopwords():
    assert tokenize("Сколько стоит и как в этом") == ["скольк", "стоит", "этом"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("сколько стоит Private Office в месяц", "tier-private"),
        ("Bespoke Dynasty преемственность", "tier-bespoke"),
        ("минимальный размер капитала", "faq-min-ticket"),
        ("есть ли скидки при оплате за год", "discounts"),
    ],
)
async def test_lexical_retrieval_finds_right_chunk(store, query, expected):
    assert await top_id(store, query) == expected


async def test_internal_playbook_hidden_by_default(store):
    query = "как просить контакты у клиента"
    hits = await store.search(query, llm=None, k=6)
    assert all(not hit.chunk.metadata.get("internal") for hit in hits)

    internal = await store.search(query, llm=None, k=6, include_internal=True)
    assert any(hit.chunk.metadata.get("internal") for hit in internal)
