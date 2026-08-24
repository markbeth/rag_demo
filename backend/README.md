# Backend — Family Office RAG Chat


Swagger: http://localhost:8000/docs

Without an API key the service still starts, but `/api/chat` returns 503. The retriever
and `/api/kb/search` work with no key at all (lexical search); `USE_EMBEDDINGS=false`
disables every call to `/embeddings`.

## Tests

```bash
uv run pytest -q       # 19 tests, the LLM is faked, no network needed
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | model, chunk count, whether embeddings are live |
| POST | `/api/chat` | buffered answer: `{message, session_id?}` |
| POST | `/api/chat/stream` | SSE: `meta` → `sources` → `token*` → `done` |
| GET | `/api/sessions/{id}` | conversation history and the collected lead |
| DELETE | `/api/sessions/{id}` | reset the conversation |
| GET | `/api/leads` | leads from the mock CRM (`backend/data/leads.jsonl`) |
| GET | `/api/kb/search?q=&k=` | retriever debugging, with scores |

```bash
curl -X POST localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"message":"Сколько стоит Private Office?"}'
```

## How it works

`route → (extract_lead ∥ retrieve) → generate → lead_strategy → [crm_submit]` — the graph
lives in [app/graph/graph.py](app/graph/graph.py).

- **route** — intent (pricing / onboarding / contact / question) and funnel stage, no LLM call.
- **extract_lead** — contact details from the message: regexes (email, phone, amounts) plus
  the LLM in JSON mode. Regex results win, and junk from the model such as `wrong@@` is dropped.
- **retrieve** — hybrid search: cosine over embeddings (0.65) plus BM25 with a Russian stemmer
  (0.35). Short follow-ups ("and the second one?") are prefixed with the client's last question.
- **generate** — answers strictly from the retrieved context; the prompt forbids inventing prices.
- **lead_strategy** — one slot at a time (name → email → phone → time), never before the client's
  second message (except for pricing questions), at most two asks per slot, and it stops asking
  after two refusals.
- **crm_submit** — as soon as a name and any contact channel are known, the lead goes to the CRM.

All storage is in-memory: sessions in a dict with a TTL, the index in RAM, embeddings cached in
`backend/data/embeddings_cache.json` (keyed by a hash of the knowledge base).

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
