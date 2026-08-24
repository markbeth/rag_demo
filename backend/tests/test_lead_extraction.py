import pytest

from app.graph.nodes import _clean_phone
from app.schemas import Lead
from tests.fakes import FakeLLM


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("my number is +41 79 123 45 67", "+41791234567"),
        ("call me on 07700 900123", "07700900123"),
        ("our capital is around 25 million EUR", None),  # an amount, not a phone number
        ("we were founded in 2011", None),  # a year, not a phone number
    ],
)
def test_phone_cleanup(text, expected):
    assert _clean_phone(text) == expected


def test_lead_slots():
    lead = Lead()
    assert lead.missing_slots() == ["name", "email", "phone", "preferred_time"]
    assert not lead.crm_ready

    lead.name, lead.email = "James", "james@example.com"
    assert lead.crm_ready
    assert lead.missing_slots() == ["phone", "preferred_time"]


async def test_regex_wins_over_llm(make_chat):
    """Regex results outrank the model, and malformed model values are dropped."""
    llm = FakeLLM(extract={"email": "wrong@@", "phone": "12", "name": "James"})
    service, _ = make_chat(llm)

    response = await service.respond("I am James, write to james@example.com", None)
    assert response.lead.email == "james@example.com"
    assert response.lead.name == "James"
    assert response.lead.phone is None
