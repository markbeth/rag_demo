"""Test doubles for the LLM provider and the CRM sink."""

from __future__ import annotations

LEAD_ASK_MARKER = "Append ONE short line to the answer"
HANDOFF_MARKER = "The lead has been handed to a partner"
LEAD_ASK_REPLY = "Leave an email and I will send the tier comparison."


class FakeLLM:
    """Stands in for the provider: canned replies keyed by the kind of prompt."""

    def __init__(
        self, extract: dict | None = None, answer: str = "Answer from the KB."
    ) -> None:
        self.extract_result = extract or {}
        self.answer = answer
        self.calls: list[list[dict]] = []

    async def chat(
        self, messages, *, temperature=None, max_tokens=None, json_mode=False
    ):
        self.calls.append(list(messages))
        content = messages[-1]["content"]
        if LEAD_ASK_MARKER in content:
            return LEAD_ASK_REPLY
        if HANDOFF_MARKER in content:
            return "Thank you, a partner will be in touch."
        return self.answer

    async def chat_json(self, messages, *, temperature=0.0):
        self.calls.append(list(messages))
        return dict(self.extract_result)

    async def chat_stream(self, messages, *, temperature=None, max_tokens=None):
        for token in self.answer.split(" "):
            yield token + " "

    async def embed(self, texts):
        raise AssertionError("The tests never use embeddings")

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
