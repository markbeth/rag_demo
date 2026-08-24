import pytest

from app.rag.store import HybridStore, stem, tokenize


async def top_id(store: HybridStore, query: str) -> str:
    hits = await store.search(query, llm=None, k=3)
    assert hits, f"nothing retrieved for query: {query}"
    return hits[0].chunk.id


def test_chunks_loaded(store: HybridStore):
    ids = {chunk.id for chunk in store.chunks}
    assert {"tier-essential", "tier-private", "tier-bespoke", "company"} <= ids
    assert store.size > 15


def test_tokenize_drops_stopwords():
    assert tokenize("How much does the Private Office cost") == [
        "privat",
        "offic",
        "cost",
    ]


@pytest.mark.parametrize(
    ("word", "expected"),
    [("discounts", "discount"), ("prices", "pric"), ("taxes", "tax"), ("fees", "fee")],
)
def test_stemmer_collapses_inflections(word, expected):
    assert stem(word) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("how much does Private Office cost per month", "tier-private"),
        ("Bespoke Dynasty succession next generation", "tier-bespoke"),
        ("minimum amount of capital", "faq-min-ticket"),
        ("are there discounts for paying a year upfront", "discounts"),
        ("do you charge a performance fee", "faq-fees"),
    ],
)
async def test_lexical_retrieval_finds_right_chunk(store, query, expected):
    assert await top_id(store, query) == expected


async def test_internal_playbook_hidden_by_default(store):
    query = "how to ask the client for contact details"
    hits = await store.search(query, llm=None, k=6)
    assert all(not hit.chunk.metadata.get("internal") for hit in hits)

    internal = await store.search(query, llm=None, k=6, include_internal=True)
    assert any(hit.chunk.metadata.get("internal") for hit in internal)
