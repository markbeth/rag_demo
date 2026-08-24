# Backend — Family Office RAG Chat


Swagger: http://localhost:8000/docs

Without an API key the service still starts, but `/api/chat` returns 503. The retriever
and `/api/kb/search` work with no key at all (lexical search); `USE_EMBEDDINGS=false`
disables every call to `/embeddings`.

## Tests

```bash
uv run pytest -q       # 24 tests, the LLM is faked, no network needed
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | model, chunk count, whether embeddings are live |
| POST | `/api/chat` | buffered answer: `{message, session_id?}` |
| POST | `/api/chat/stream` | SSE: `meta` → `sources` → `token*` → `done` |
| GET | `/api/sessions/{id}` | conversation history and the collected lead |
| DELETE | `/api/sessions/{id}` | reset the conversation |
| GET | `/api/leads` | leads from the mock CRM (`src/data/leads.jsonl`) |
| GET | `/api/kb/search?q=&k=` | retriever debugging, with scores |

```bash
curl -X POST localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"message":"How much does Private Office cost?"}'
```

## How it works

`route → (extract_lead ∥ retrieve) → generate → lead_strategy → [crm_submit]` — the graph
lives in [app/graph/graph.py](app/graph/graph.py).

- **route** — intent (pricing / onboarding / contact / question) and funnel stage, no LLM call.
- **extract_lead** — contact details from the message: regexes (email, phone, amounts) plus
  the LLM in JSON mode. Regex results win, and junk from the model such as `wrong@@` is dropped.
- **retrieve** — hybrid search: cosine over embeddings (0.65) plus BM25 over an inverted index
  with a light English stemmer (0.35). Short follow-ups ("and the second one?") are prefixed with
  the client's last question.
- **generate** — answers strictly from the retrieved context; the prompt forbids inventing prices.
- **lead_strategy** — one slot at a time (name → email → phone → time), never before the client's
  second message (except for pricing questions), at most two asks per slot, and it stops asking
  after two refusals.
- **crm_submit** — as soon as a name and any contact channel are known, the lead goes to the CRM.

All storage is in-memory: sessions in a dict with a TTL, the index in RAM, embeddings cached in
`src/data/embeddings_cache.json` (keyed by a hash of the knowledge base).

Cost per request: one embedding call for the query, one LLM call for extraction, one for the
answer, and one for the contact ask when a slot is due, so three to four provider calls.
Complexity is linear in the number of chunks; see the note below.

| Path | Complexity |
|---|---|
| BM25 scoring | `O(matching postings)` via the inverted index, not `O(terms x chunks)` |
| Vector scoring | `O(N x d)` exact cosine, with chunk norms precomputed at build time |
| Top-k selection | `O(N log k)` (`heapq.nlargest`) |
| Session lookup | `O(1)`; expired sessions are swept at most once a minute |
| `GET /api/leads` | streams the jsonl file, holding at most `limit` records in memory |

## Concurrency notes

- **One turn at a time per chat.** Each session carries an `asyncio.Lock` held for the whole
  turn, streaming included. Without it two concurrent messages in one session both read
  `crm_status == "pending"` and the lead is submitted twice; `test_concurrent_turns_submit_the_lead_once`
  covers exactly that. Different sessions never block each other.
- **Nothing blocking runs on the event loop.** Disk access goes through `asyncio.to_thread`
  (embedding cache, CRM jsonl). Cosine scoring is pure Python: ~1 ms for 30 chunks, so it runs
  inline, and above 200 chunks it moves to a thread rather than stalling the loop.
- **Provider calls per turn:** extraction and retrieval run concurrently via `asyncio.gather`,
  then the answer, then the contact ask if a slot is due. The shared `httpx.AsyncClient` allows
  20 connections, so heavier load queues instead of failing.
- **Run a single worker.** Sessions live in process memory, so with `--workers 2` a client only
  keeps its history if the load balancer pins it to the same process. Redis is the fix, and it is
  out of scope for the demo. Appends to `leads.jsonl` are single short `O_APPEND` writes, so
  lines from several processes cannot interleave, but ordering across processes is not guaranteed.
- **Client disconnects** cancel the streaming generator: the lock is released and the partial
  turn is simply not committed to the session.

## Layout

```
app/
  config.py          settings from .env
  schemas.py         API contracts (Lead, ChatResponse, ...)
  deps.py            dependency container wiring
  llm/client.py      async httpx: chat, chat_stream, chat_json, embed, retries
  rag/loader.py      JSON -> meaning-sized chunks
  rag/store.py       hybrid index (BM25 + cosine)
  graph/             prompts, state, nodes, graph
  services/          chat (incl. SSE), sessions, crm
  api/               routes_chat, routes_meta
  data/              mock data: services, faq, playbook
```

Packages use implicit namespace packages (PEP 420), so there are no `__init__.py` files;
`pythonpath = ["."]` in `pyproject.toml` keeps pytest imports working.
