from tests.fakes import LEAD_ASK_REPLY, FakeCrm, FakeLLM


async def test_no_contact_request_in_first_answer(make_chat):
    """Value first: the opening answer must not ask for anything."""
    llm = FakeLLM(answer="Мы независимый family office.")
    service, _ = make_chat(llm)

    response = await service.respond("Расскажите, кто вы такие", None)
    assert response.answer == "Мы независимый family office."
    assert response.stage in {"discovery", "greeting"}


async def test_pricing_question_triggers_contact_request(make_chat):
    """A pricing question is a trigger signal, so the ask comes right away."""
    llm = FakeLLM(answer="Private Office — 28 000 € в месяц.")
    service, _ = make_chat(llm)

    response = await service.respond("Сколько стоит Private Office?", None)
    assert LEAD_ASK_REPLY in response.answer
    assert response.stage == "qualifying"
    assert response.sources


async def test_lead_lands_in_crm_when_name_and_channel_known(make_chat):
    llm = FakeLLM(extract={"name": "Мария"}, answer="Да, конечно.")
    service, crm = make_chat(llm, FakeCrm())

    first = await service.respond("Сколько стоит Essential Wealth?", None)
    assert not crm.submitted, "no channel yet, nothing to submit"

    second = await service.respond("Мария, maria@example.com", first.session_id)
    assert crm.submitted, "the lead should reach the CRM"
    assert second.crm_status == "submitted"
    assert second.lead_id
    assert second.lead.email == "maria@example.com"
    assert second.stage == "submitted"


async def test_two_refusals_stop_asking(make_chat):
    llm = FakeLLM(extract={"refused_contact": True}, answer="Хорошо.")
    service, _ = make_chat(llm)

    session = (await service.respond("Сколько стоит?", None)).session_id
    await service.respond("Не хочу оставлять контакты", session)
    third = await service.respond("Не буду оставлять данные", session)

    assert LEAD_ASK_REPLY not in third.answer
    assert third.stage == "declined"


async def test_history_and_session_reuse(make_chat):
    llm = FakeLLM(answer="Ответ.")
    service, _ = make_chat(llm)

    first = await service.respond("Что входит в Private Office?", None)
    second = await service.respond("а во второй тариф?", first.session_id)
    assert first.session_id == second.session_id


async def test_stream_emits_sse_events(make_chat):
    llm = FakeLLM(answer="Стоимость 12 000 евро.")
    service, _ = make_chat(llm)

    joined = "".join([chunk async for chunk in service.stream("Сколько стоит Essential?", None)])
    assert "event: meta" in joined
    assert "event: sources" in joined
    assert "event: token" in joined
    assert "event: done" in joined
