import pytest

from app.graph.nodes import _clean_phone
from app.schemas import Lead
from tests.fakes import FakeLLM


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("мой номер +7 916 123-45-67", "+79161234567"),
        ("звоните 89161234567", "89161234567"),
        ("капитал примерно 25 млн евро", None),  # an amount, not a phone number
        ("основаны в 2011 году", None),  # a year, not a phone number
    ],
)
def test_phone_cleanup(text, expected):
    assert _clean_phone(text) == expected


def test_lead_slots():
    lead = Lead()
    assert lead.missing_slots() == ["name", "email", "phone", "preferred_time"]
    assert not lead.crm_ready

    lead.name, lead.email = "Иван", "ivan@example.com"
    assert lead.crm_ready
    assert lead.missing_slots() == ["phone", "preferred_time"]


async def test_regex_wins_over_llm(make_chat):
    """Regex results outrank the model, and malformed model values are dropped."""
    llm = FakeLLM(extract={"email": "wrong@@", "phone": "12", "name": "Иван"})
    service, _ = make_chat(llm)

    response = await service.respond("Пишите на ivan@example.com, я Иван", None)
    assert response.lead.email == "ivan@example.com"
    assert response.lead.name == "Иван"
    assert response.lead.phone is None
