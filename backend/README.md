# Backend — Family Office RAG Chat

FastAPI + LangGraph + асинхронный httpx-клиент к любому OpenAI-совместимому провайдеру.

## Запуск

```bash
cd backend
cp .env.example .env          # вписать OPENAI_API_KEY и OPENAI_BASE_URL
uv sync --extra dev
uv run uvicorn app.main:app --reload --port 8000
```

Swagger: http://localhost:8000/docs

Без ключа сервис поднимается, но `/api/chat` вернёт 503. Ретривер и `/api/kb/search`
работают и без ключа (лексический поиск), `USE_EMBEDDINGS=false` отключает вызовы `/embeddings`.

## Тесты

```bash
uv run pytest -q       # 19 тестов, LLM подменён фейком, сеть не нужна
```

## Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/health` | модель, число чанков, есть ли эмбеддинги |
| POST | `/api/chat` | ответ целиком: `{message, session_id?}` |
| POST | `/api/chat/stream` | SSE: `meta` → `sources` → `token*` → `done` |
| GET | `/api/sessions/{id}` | история диалога и собранный лид |
| DELETE | `/api/sessions/{id}` | сброс диалога |
| GET | `/api/leads` | заявки из мок-CRM (`backend/data/leads.jsonl`) |
| GET | `/api/kb/search?q=&k=` | отладка ретривера со скорами |

```bash
curl -X POST localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"message":"Сколько стоит Private Office?"}'
```

## Как это работает

`route → (extract_lead ∥ retrieve) → generate → lead_strategy → [crm_submit]` — граф в
[app/graph/graph.py](app/graph/graph.py).

- **route** — интент (цена / онбординг / контакт / вопрос) и стадия воронки, без вызова LLM.
- **extract_lead** — контакты из сообщения: регексы (email, телефон, суммы) + LLM в JSON-режиме;
  результат регекса приоритетнее, LLM-мусор вроде `wrong@@` отбрасывается.
- **retrieve** — гибридный поиск: cosine по эмбеддингам (0.65) + BM25 с русским стеммером (0.35).
  Короткие реплики («а второй?») дополняются предыдущим вопросом клиента.
- **generate** — ответ строго по контексту; цены выдумывать запрещено промптом.
- **lead_strategy** — один слот за раз (имя → email → телефон → время), не раньше второго
  сообщения (кроме вопроса про цену), максимум два запроса на слот, стоп после двух отказов.
- **crm_submit** — как только есть имя и любой канал связи, заявка уходит в CRM.

Хранилища in-memory: сессии в dict с TTL, индекс в памяти, эмбеддинги кэшируются в
`backend/data/embeddings_cache.json` (ключ — хэш содержимого базы знаний).

## Структура

```
app/
  config.py          настройки из .env
  schemas.py         контракты API (Lead, ChatResponse, ...)
  deps.py            сборка контейнера зависимостей
  llm/client.py      async httpx: chat, chat_stream, chat_json, embed, retry
  rag/loader.py      JSON → смысловые чанки
  rag/store.py       гибридный индекс (BM25 + cosine)
  graph/             prompts, state, nodes, graph
  services/          chat (в т.ч. SSE), sessions, crm
  api/               routes_chat, routes_meta
  data/              мок-данные: services, faq, playbook
```
