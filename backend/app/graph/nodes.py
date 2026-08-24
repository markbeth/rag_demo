"""Узлы графа: маршрутизация, извлечение лида, поиск, генерация, стратегия сбора, CRM."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.graph import prompts
from app.graph.state import GraphState
from app.llm.client import LLMClient, LLMError
from app.rag.store import HybridStore
from app.schemas import Lead, Message
from app.services.crm import CrmSink

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}")
PHONE_RE = re.compile(r"(?:\+|00)?[\d][\d\s\-().]{7,17}\d")
MONEY_RE = re.compile(r"\d[\d\s.,]*\s*(?:млн|млрд|mln|m\b|kk|миллион|million)", re.IGNORECASE)

PRICING_WORDS = (
    "цена", "цены", "стоимость", "стоит", "сколько", "тариф", "прайс", "скидк", "дорого",
    "бюджет", "оплат", "fee", "price", "cost",
)
START_WORDS = ("как начать", "как стартовать", "с чего начать", "заявк", "договор", "созвон", "звонок")
REFUSAL_WORDS = ("не хочу", "не буду", "не готов", "позже", "не дам", "без контакт", "не оставлю")

SLOT_LABELS = {
    "name": "имя клиента",
    "email": "email клиента",
    "phone": "телефон клиента",
    "preferred_time": "удобное время звонка",
}
FALLBACK_ANSWER = (
    "Извините, сейчас не могу обратиться к базе знаний. "
    "Могу передать вопрос партнёру — он ответит лично."
)


class Nodes:
    """Держит зависимости, чтобы узлы оставались чистыми функциями состояния."""

    def __init__(
        self,
        llm: LLMClient,
        store: HybridStore,
        playbook: dict[str, Any],
        crm: CrmSink,
        *,
        top_k: int = 4,
        company: str = "Meridian Family Office",
    ) -> None:
        self.llm = llm
        self.store = store
        self.playbook = playbook
        self.crm = crm
        self.top_k = top_k
        self.company = company

    # --- 1. маршрутизация ------------------------------------------------

    async def route(self, state: GraphState) -> GraphState:
        text = state["message"].lower()
        lead: Lead = state["lead"]
        if any(word in text for word in PRICING_WORDS):
            intent = "pricing"
        elif any(word in text for word in START_WORDS):
            intent = "onboarding"
        elif EMAIL_RE.search(text) or PHONE_RE.search(text):
            intent = "contact"
        elif len(text.split()) <= 3 and not text.endswith("?"):
            intent = "smalltalk"
        else:
            intent = "question"

        stage = state.get("stage", "greeting")
        if lead.crm_ready and state.get("crm_status") == "submitted":
            stage = "submitted"
        elif lead.refusals >= 2:
            stage = "declined"
        elif lead.name or lead.has_channel:
            stage = "collecting"
        elif intent in {"pricing", "onboarding"} or MONEY_RE.search(text):
            stage = "qualifying"
        elif state.get("user_turns", 0) >= 1:
            stage = "discovery"
        return {"intent": intent, "stage": stage}

    # --- 2. извлечение данных клиента ------------------------------------

    async def extract_lead(self, state: GraphState) -> GraphState:
        message = state["message"]
        lead = state["lead"].model_copy()

        # Регексы — надёжный слой: их результат приоритетнее LLM.
        if email := EMAIL_RE.search(message):
            lead.email = email.group()
        if phone := _clean_phone(message):
            lead.phone = phone
        if money := MONEY_RE.search(message):
            lead.capital_range = lead.capital_range or money.group().strip()

        extracted = await self._llm_extract(message, lead)
        for field in ("name", "preferred_time", "capital_range", "interest", "tier_interest"):
            value = extracted.get(field)
            if isinstance(value, str) and value.strip() and not getattr(lead, field):
                setattr(lead, field, value.strip()[:200])
        if not lead.email and _valid_email(extracted.get("email")):
            lead.email = extracted["email"].strip()
        if not lead.phone and (phone := _clean_phone(str(extracted.get("phone") or ""))):
            lead.phone = phone

        refused = bool(extracted.get("refused_contact")) or any(
            word in message.lower() for word in REFUSAL_WORDS
        )
        if refused and not lead.has_channel:
            lead.refusals += 1
        return {"lead": lead}

    async def _llm_extract(self, message: str, lead: Lead) -> dict[str, Any]:
        if len(message) > 2000:
            message = message[:2000]
        prompt = prompts.EXTRACT_LEAD.format(known=_known(lead), message=message)
        try:
            return await self.llm.chat_json([{"role": "user", "content": prompt}])
        except LLMError as exc:
            logger.warning("Извлечение лида не удалось: %s", exc)
            return {}

    # --- 3. поиск --------------------------------------------------------

    async def retrieve(self, state: GraphState) -> GraphState:
        query = state["message"]
        history: list[Message] = state.get("history", [])
        # Короткие реплики («а второй?») без контекста не находят ничего —
        # дописываем предыдущий вопрос клиента.
        if len(query.split()) <= 4:
            previous = next((m.content for m in reversed(history) if m.role == "user"), "")
            query = f"{previous} {query}".strip()
        hits = await self.store.search(query, llm=self.llm, k=self.top_k)
        return {"hits": hits}

    # --- 4. генерация ответа ---------------------------------------------

    async def generate(self, state: GraphState) -> GraphState:
        messages = self.answer_messages(state)
        try:
            answer = await self.llm.chat(messages, max_tokens=600)
        except LLMError as exc:
            logger.error("Генерация ответа не удалась: %s", exc)
            return {"answer": FALLBACK_ANSWER}
        return {"answer": answer.strip()}

    def answer_messages(self, state: GraphState) -> list[dict[str, str]]:
        context = _format_context(state.get("hits", []))
        system = prompts.SYSTEM_ANSWER.format(
            company=self.company, context=context, known=_known(state["lead"])
        )
        messages = [{"role": "system", "content": system}]
        for message in state.get("history", [])[-8:]:
            messages.append({"role": message.role, "content": message.content})
        messages.append({"role": "user", "content": state["message"]})
        return messages

    # --- 5. стратегия сбора контактов ------------------------------------

    async def lead_strategy(self, state: GraphState) -> GraphState:
        lead: Lead = state["lead"]
        slot = self._next_slot(state, lead)
        if slot is None:
            return {"lead_ask": ""}
        hints = self.playbook.get("slot_questions", {}).get(slot, [])
        hint = hints[min(state.get("user_turns", 1) - 1, len(hints) - 1)] if hints else ""
        prompt = prompts.LEAD_ASK.format(
            slot_label=SLOT_LABELS.get(slot, slot),
            hint=hint,
            stage=state.get("stage", "discovery"),
            turns=state.get("user_turns", 1),
            known=_known(lead),
            answer=state.get("answer", ""),
        )
        try:
            ask = (await self.llm.chat([{"role": "user", "content": prompt}], temperature=0.5,
                                       max_tokens=140)).strip()
        except LLMError as exc:
            logger.warning("Не удалось сгенерировать запрос слота, беру шаблон: %s", exc)
            ask = hint
        asked = [*state.get("asked_slots", []), slot]
        return {"lead_ask": ask, "asked_slots": asked}

    def _next_slot(self, state: GraphState, lead: Lead) -> str | None:
        """Один слот за раз, только когда это уместно (см. playbook)."""
        if lead.refusals >= 2 or state.get("stage") == "submitted":
            return None
        turns = state.get("user_turns", 1)
        triggered = state.get("intent") in {"pricing", "onboarding", "contact"} or bool(
            lead.capital_range
        )
        if turns < 2 and not triggered:  # value-first: в первом ответе не просим ничего
            return None
        asked = state.get("asked_slots", [])
        for slot in lead.missing_slots():
            if asked.count(slot) < 2:  # один слот просим максимум дважды
                return slot
        return None

    # --- 6. заявка в CRM --------------------------------------------------

    async def crm_submit(self, state: GraphState) -> GraphState:
        lead: Lead = state["lead"]
        transcript = f"{state.get('transcript', '')}\nuser: {state['message']}"
        try:
            lead_id = await self.crm.submit(state["session_id"], lead, transcript)
        except OSError as exc:
            logger.error("Не удалось записать лид в CRM: %s", exc)
            return {"crm_status": "failed"}

        contacts = ", ".join(filter(None, [lead.name, lead.email, lead.phone]))
        try:
            note = await self.llm.chat(
                [{"role": "user", "content": prompts.CONFIRM_HANDOFF.format(contacts=contacts)}],
                temperature=0.4,
                max_tokens=120,
            )
        except LLMError:
            channel = lead.email or lead.phone
            note = self.playbook["handoff_message"].format(
                name=lead.name or "", channel=channel or "указанному контакту"
            )
        return {
            "crm_status": "submitted",
            "lead_id": lead_id,
            "stage": "submitted",
            "lead_ask": note.strip(),
        }


# --- вспомогательное -----------------------------------------------------


def should_submit(state: GraphState) -> str:
    lead: Lead = state["lead"]
    if lead.crm_ready and state.get("crm_status") != "submitted":
        return "crm_submit"
    return "end"


def compose_answer(answer: str, lead_ask: str) -> str:
    if not lead_ask:
        return answer
    if lead_ask.strip().lower() in answer.lower():
        return answer
    return f"{answer.rstrip()}\n\n{lead_ask.strip()}"


def _format_context(hits: list) -> str:
    if not hits:
        return "(-- в базе знаний нет релевантных данных --)"
    return "\n\n".join(
        f"[{i + 1}] {hit.chunk.title} (источник: {hit.chunk.source})\n{hit.chunk.text}"
        for i, hit in enumerate(hits)
    )


def _known(lead: Lead) -> str:
    known = {
        key: value
        for key, value in lead.model_dump(exclude={"refusals", "notes"}).items()
        if value
    }
    return str(known) if known else "ничего не известно"


def _valid_email(value: object) -> bool:
    return isinstance(value, str) and bool(EMAIL_RE.fullmatch(value.strip()))


def _clean_phone(text: str) -> str | None:
    """Отсекает номера-обманки: годы, суммы, «5 млн»."""
    for match in PHONE_RE.finditer(text):
        raw = match.group()
        digits = re.sub(r"\D", "", raw)
        if not 9 <= len(digits) <= 15:
            continue
        tail = text[match.end() : match.end() + 12].lower()
        if any(word in tail for word in ("млн", "млрд", "€", "eur", "%")):
            continue
        return ("+" if raw.strip().startswith("+") else "") + digits
    return None
