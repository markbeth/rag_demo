"""Hybrid in-memory index: embeddings (cosine) plus lexical BM25.

Why hybrid: lexical search wins on numbers and tier names ("Bespoke", "setup
fee"), embeddings win on paraphrased questions. It also keeps the demo working
when the provider exposes no /embeddings endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.config import RUNTIME_DIR
from app.llm.client import LLMClient, LLMError
from app.rag.loader import Chunk, load_chunks

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
    "have", "has", "had", "it", "its", "this", "that", "these", "those", "i", "we", "you",
    "they", "he", "she", "my", "our", "your", "their", "me", "us", "them", "what", "which",
    "who", "how", "when", "where", "why", "can", "could", "would", "should", "will", "there",
    "about", "into", "than", "then", "so", "not", "no", "any", "all", "also", "much", "many",
}

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
        # The title counts twice: it describes the chunk's topic better than the body.
        self._docs: list[Counter[str]] = [
            Counter(tokenize(f"{c.title} {c.title}\n{c.text}")) for c in chunks
        ]
        self._lengths = [max(sum(d.values()), 1) for d in self._docs]
        self._avg_len = sum(self._lengths) / max(len(self._lengths), 1)
        self._df: Counter[str] = Counter()
        for doc in self._docs:
            self._df.update(doc.keys())

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def has_vectors(self) -> bool:
        return self._vectors is not None

    # --- index building -------------------------------------------------

    async def build_vectors(self, llm: LLMClient, *, cache_dir: Path | None = None) -> None:
        """Embeds all chunks, cached on disk keyed by a hash of the content."""
        cache_path = (cache_dir or RUNTIME_DIR) / "embeddings_cache.json"
        fingerprint = self._fingerprint()
        cached = _read_cache(cache_path)
        if cached and cached.get("fingerprint") == fingerprint:
            self._vectors = cached["vectors"]
            logger.info("Embeddings restored from cache: %s chunks", len(self._vectors))
            return
        try:
            texts = [f"{c.title}\n{c.text}" for c in self.chunks]
            self._vectors = await llm.embed(texts)
        except LLMError as exc:
            logger.warning("Embeddings unavailable, falling back to lexical search: %s", exc)
            self._vectors = None
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"fingerprint": fingerprint, "vectors": self._vectors}), encoding="utf-8"
        )
        logger.info("Embeddings computed and cached: %s chunks", len(self._vectors))

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
        lexical = self._lexical_scores(query)
        vector: list[float] = [0.0] * self.size
        if self._vectors and llm is not None:
            try:
                query_vector = (await llm.embed([query]))[0]
                vector = [_cosine(query_vector, v) for v in self._vectors]
            except LLMError as exc:
                logger.warning("Query embedding failed, lexical scores only: %s", exc)

        lex_norm = _normalize(lexical)
        vec_norm = _normalize(vector)
        weights = (
            (_VECTOR_WEIGHT, _LEXICAL_WEIGHT) if any(vector) else (0.0, 1.0)
        )
        hits = [
            Hit(chunk=chunk, score=weights[0] * vec_norm[i] + weights[1] * lex_norm[i])
            for i, chunk in enumerate(self.chunks)
            if include_internal or not chunk.metadata.get("internal")
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return [hit for hit in hits[:k] if hit.score > 0.0]

    def _lexical_scores(self, query: str) -> list[float]:
        """BM25 with no external dependencies (k1=1.5, b=0.75)."""
        terms = tokenize(query)
        if not terms:
            return [0.0] * self.size
        k1, b, n = 1.5, 0.75, max(self.size, 1)
        scores = [0.0] * self.size
        for term in set(terms):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self._docs):
                tf = doc.get(term, 0)
                if not tf:
                    continue
                denom = tf + k1 * (1 - b + b * self._lengths[i] / self._avg_len)
                scores[i] += idf * (tf * (k1 + 1)) / denom
        return scores


def _read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _normalize(values: list[float]) -> list[float]:
    top = max(values, default=0.0)
    return [v / top for v in values] if top > 0 else list(values)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def build_store() -> HybridStore:
    return HybridStore(load_chunks())
