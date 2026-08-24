from tests.fakes import LEAD_ASK_REPLY, FakeCrm, FakeLLM


async def test_no_contact_request_in_first_answer(make_chat):
    """Value first: the opening answer must not ask for anything."""
    llm = FakeLLM(answer="We are an independent family office.")
    service, _ = make_chat(llm)

    response = await service.respond("Tell me who you are", None)
    assert response.answer == "We are an independent family office."
    assert response.stage in {"discovery", "greeting"}


async def test_pricing_question_triggers_contact_request(make_chat):
    """A pricing question is a trigger signal, so the ask comes right away."""
    llm = FakeLLM(answer="Private Office is 28 000 EUR per month.")
    service, _ = make_chat(llm)

    response = await service.respond("How much does Private Office cost?", None)
    assert LEAD_ASK_REPLY in response.answer
    assert response.stage == "qualifying"
    assert response.sources


async def test_lead_lands_in_crm_when_name_and_channel_known(make_chat):
    llm = FakeLLM(extract={"name": "Maria"}, answer="Yes, of course.")
    service, crm = make_chat(llm, FakeCrm())

    first = await service.respond("What is the price of Essential Wealth?", None)
    assert not crm.submitted, "no channel yet, nothing to submit"

    second = await service.respond("Maria, maria@example.com", first.session_id)
    assert crm.submitted, "the lead should reach the CRM"
    assert second.crm_status == "submitted"
    assert second.lead_id
    assert second.lead.email == "maria@example.com"
    assert second.stage == "submitted"


async def test_two_refusals_stop_asking(make_chat):
    llm = FakeLLM(extract={"refused_contact": True}, answer="Understood.")
    service, _ = make_chat(llm)

    session = (await service.respond("How much does it cost?", None)).session_id
    await service.respond("I don't want to leave my contact details", session)
    third = await service.respond("I do not want to share anything", session)

    assert LEAD_ASK_REPLY not in third.answer
    assert third.stage == "declined"


async def test_history_and_session_reuse(make_chat):
    llm = FakeLLM(answer="Here you go.")
    service, _ = make_chat(llm)

    first = await service.respond("What is included in Private Office?", None)
    second = await service.respond("and the second tier?", first.session_id)
    assert first.session_id == second.session_id


async def test_stream_emits_sse_events(make_chat):
    llm = FakeLLM(answer="It costs 12 000 EUR.")
    service, _ = make_chat(llm)

    joined = "".join([chunk async for chunk in service.stream("Price of Essential?", None)])
    assert "event: meta" in joined
    assert "event: sources" in joined
    assert "event: token" in joined
    assert "event: done" in joined
