"""Hybrid in-memory index: embeddings (cosine) plus lexical BM25.

Why hybrid: lexical search wins on numbers and tier names ("Bespoke", "setup
fee"), embeddings win on paraphrased questions. It also keeps the demo working
when the provider exposes no /embeddings endpoint.
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.config import RUNTIME_DIR
from app.llm.client import LLMClient, LLMError
from app.rag.loader import Chunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
# fmt: off
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
    "have", "has", "had", "it", "its", "this", "that", "these", "those", "i", "we", "you",
    "they", "he", "she", "my", "our", "your", "their", "me", "us", "them", "what", "which",
    "who", "how", "when", "where", "why", "can", "could", "would", "should", "will", "there",
    "about", "into", "than", "then", "so", "not", "no", "any", "all", "also", "much", "many",
}
# fmt: on

_BM25_K1 = 1.5
_BM25_B = 0.75
_COSINE_INLINE_LIMIT = 200
_VECTOR_WEIGHT = 0.65
_LEXICAL_WEIGHT = 0.35


# A light stemmer instead of nltk/Porter: without it BM25 misses obvious matches
# ("discounts" would not find "discount", "pricing" would not find "price").
_MIN_STEM = 4


def stem(token: str) -> str:
    if len(token) <= 3 or token.isdigit():  # "fee", "tax", "eur" stay as they are
        return token
    if token.endswith("ies"):
        token = token[:-3] + "y"
    elif token.endswith(("sses", "xes", "zes", "ches", "shes")):  # "taxes" -> "tax"
        token = token[:-2]
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    for suffix in ("ing", "edly", "ed", "ly", "ment", "tion", "ness"):
        if len(token) - len(suffix) >= _MIN_STEM and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    # "including" and "included" must collapse onto the same stem as "include".
    if len(token) > _MIN_STEM and token.endswith("e"):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    raw = (m.group().lower() for m in _TOKEN_RE.finditer(text))
    return [stem(t) for t in raw if t not in _STOPWORDS]


@dataclass(slots=True)
class Hit:
    chunk: Chunk
    score: float


class HybridStore:
    """An index over tens to hundreds of chunks, which is enough for this demo."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._vectors: list[list[float]] | None = None
        self._norms: list[float] = []
        # Inverted index: term -> [(chunk index, term frequency)]. Scoring then touches
        # only the chunks that contain a query term instead of scanning every chunk.
        self._postings: dict[str, list[tuple[int, int]]] = {}
        # BM25 length normalisation per chunk, precomputed once.
        self._length_norms: list[float] = []
        self._build_lexical_index()

    def _build_lexical_index(self) -> None:
        # The title counts twice: it describes the chunk's topic better than the body.
        docs = [
            Counter(tokenize(f"{c.title} {c.title}\n{c.text}")) for c in self.chunks
        ]
        lengths = [max(sum(doc.values()), 1) for doc in docs]
        avg_len = sum(lengths) / max(len(lengths), 1)
        self._length_norms = [
            _BM25_K1 * (1 - _BM25_B + _BM25_B * ln / avg_len) for ln in lengths
        ]
        for index, doc in enumerate(docs):
            for term, tf in doc.items():
                self._postings.setdefault(term, []).append((index, tf))

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def has_vectors(self) -> bool:
        return self._vectors is not None

    # --- index building -------------------------------------------------

    async def build_vectors(
        self, llm: LLMClient, *, cache_dir: Path | None = None
    ) -> None:
        """Embeds all chunks, cached on disk keyed by a hash of the content."""
        cache_path = (cache_dir or RUNTIME_DIR) / "embeddings_cache.json"
        fingerprint = self._fingerprint()
        cached = await asyncio.to_thread(_read_cache, cache_path)
        if cached and cached.get("fingerprint") == fingerprint:
            self._set_vectors(cached["vectors"])
            logger.info("Embeddings restored from cache: %s chunks", len(self._vectors))
            return
        try:
            texts = [f"{c.title}\n{c.text}" for c in self.chunks]
            self._set_vectors(await llm.embed(texts))
        except LLMError as exc:
            logger.warning(
                "Embeddings unavailable, falling back to lexical search: %s", exc
            )
            self._vectors = None
            return
        payload = json.dumps({"fingerprint": fingerprint, "vectors": self._vectors})
        await asyncio.to_thread(_write_cache, cache_path, payload)
        logger.info("Embeddings computed and cached: %s chunks", len(self._vectors))

    def _set_vectors(self, vectors: list[list[float]]) -> None:
        """Chunk norms are cached here so cosine does not recompute them per query."""
        self._vectors = vectors
        self._norms = [math.sqrt(sum(x * x for x in vector)) for vector in vectors]

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        for chunk in self.chunks:
            digest.update(chunk.id.encode())
            digest.update(chunk.text.encode())
        return digest.hexdigest()

    # --- search ---------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        llm: LLMClient | None = None,
        k: int = 4,
        include_internal: bool = False,
    ) -> list[Hit]:
        lexical = _normalize(self._lexical_scores(query))
        vector = _normalize(await self._vector_scores(query, llm))
        vec_weight, lex_weight = (
            (_VECTOR_WEIGHT, _LEXICAL_WEIGHT) if any(vector) else (0.0, 1.0)
        )

        scored = (
            Hit(chunk=chunk, score=vec_weight * vector[i] + lex_weight * lexical[i])
            for i, chunk in enumerate(self.chunks)
            if include_internal or not chunk.metadata.get("internal")
        )
        top = heapq.nlargest(k, scored, key=lambda hit: hit.score)
        return [hit for hit in top if hit.score > 0.0]

    async def _vector_scores(self, query: str, llm: LLMClient | None) -> list[float]:
        if not self._vectors or llm is None:
            return [0.0] * self.size
        try:
            query_vector = (await llm.embed([query]))[0]
        except LLMError as exc:
            logger.warning("Query embedding failed, lexical scores only: %s", exc)
            return [0.0] * self.size
        # Pure-Python cosine is ~1 ms for 30 chunks but ~40 ms for 1000, and that time is
        # spent with the event loop blocked, so past the threshold it moves to a thread.
        if self.size > _COSINE_INLINE_LIMIT:
            return await asyncio.to_thread(self._cosine_scores, query_vector)
        return self._cosine_scores(query_vector)

    def _cosine_scores(self, query_vector: list[float]) -> list[float]:
        query_norm = math.sqrt(sum(x * x for x in query_vector))
        if not query_norm or not self._vectors:
            return [0.0] * self.size
        return [
            (
                sum(x * y for x, y in zip(query_vector, vector, strict=False))
                / (query_norm * norm)
                if norm
                else 0.0
            )
            for vector, norm in zip(self._vectors, self._norms, strict=False)
        ]

    def _lexical_scores(self, query: str) -> list[float]:
        """BM25 with no external dependencies, scored through the inverted index."""
        scores = [0.0] * self.size
        n = max(self.size, 1)
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, tf in postings:
                scores[index] += (
                    idf * (tf * (_BM25_K1 + 1)) / (tf + self._length_norms[index])
                )
        return scores


def _read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _normalize(values: list[float]) -> list[float]:
    top = max(values, default=0.0)
    return [v / top for v in values] if top > 0 else list(values)
