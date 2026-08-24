# Family Office RAG Chat

A monorepo demo of a RAG chatbot: it advises prospects on family office services and
pricing, and collects their contact details into a CRM along the way.

```
backlog.md    requirements and the backend/frontend task list
backend/      FastAPI + LangGraph + httpx (done)
frontend/     React + Vite (next up)
```

Stack: Python 3.12, FastAPI, LangGraph, any OpenAI-compatible provider, React + Vite.

Backend quickstart: [backend/README.md](backend/README.md).

> Note on languages: code, comments and docs are English. The mock knowledge base and
> the bot's client-facing copy are Russian, because the demo plays a Russian-speaking
> family office. The prompts instruct the model to reply in the client's language.
