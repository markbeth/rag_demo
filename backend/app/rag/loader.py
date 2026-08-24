"""Turns the mock JSON files into knowledge base chunks.

Chunking follows meaning, not character counts: one chunk per tier, add-on, FAQ
entry or playbook principle. For a structured source that beats a text splitter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import DATA_DIR


@dataclass(slots=True)
class Chunk:
    id: str
    title: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_price(value: Any) -> str:
    return f"{value:,} €".replace(",", " ") if isinstance(value, int | float) else str(value)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def load_chunks(data_dir: Path | None = None) -> list[Chunk]:
    data_dir = data_dir or DATA_DIR
    chunks: list[Chunk] = []
    services = _read(data_dir / "services.json")
    company = services["company"]

    chunks.append(
        Chunk(
            id="company",
            title=f"О компании {company['name']}",
            source="services.json",
            metadata={"category": "company"},
            text=(
                f"{company['name']} — независимый family office, основан в {company['founded']} году.\n"
                f"Активы под управлением: {company['aum_eur']} €. Клиентов: {company['clients']} семей.\n"
                f"Юрисдикции: {', '.join(company['jurisdictions'])}.\n"
                f"Модель оплаты: {company['billing']}"
            ),
        )
    )

    for tier in services["tiers"]:
        excludes = (
            f"\nНе входит:\n{_bullets(tier['excludes'])}" if tier.get("excludes") else ""
        )
        chunks.append(
            Chunk(
                id=f"tier-{tier['id']}",
                title=f"Тариф {tier['name']}",
                source="services.json",
                metadata={"category": "pricing", "tier": tier["id"], "tier_name": tier["name"]},
                text=(
                    f"Тариф «{tier['name']}» — {tier['tagline']}.\n"
                    f"Для кого: {tier['target_client']}.\n"
                    f"Стоимость: {_fmt_price(tier['price_monthly_eur'])} в месяц. "
                    f"Setup fee: {_fmt_price(tier['setup_fee_eur'])}. "
                    f"Минимальный срок договора: {tier['min_contract_months']} месяцев. "
                    f"{tier['price_note']}.\n"
                    f"Онбординг: {tier['onboarding_weeks']} недель. Команда: {tier['team']}.\n"
                    f"Входит в тариф:\n{_bullets(tier['includes'])}{excludes}"
                ),
            )
        )

    addons = services.get("addons", [])
    if addons:
        chunks.append(
            Chunk(
                id="addons",
                title="Дополнительные услуги (add-ons)",
                source="services.json",
                metadata={"category": "pricing"},
                text="Дополнительные услуги сверх тарифа:\n"
                + "\n".join(
                    f"- {a['name']}: {_fmt_price(a['price_eur'])}, срок: {a['duration']}"
                    for a in addons
                ),
            )
        )

    discounts = services.get("discounts", [])
    if discounts:
        chunks.append(
            Chunk(
                id="discounts",
                title="Скидки и условия оплаты",
                source="services.json",
                metadata={"category": "pricing"},
                text="Действующие скидки и специальные условия:\n" + _bullets(discounts),
            )
        )

    for item in _read(data_dir / "faq.json")["items"]:
        chunks.append(
            Chunk(
                id=item["id"],
                title=item["question"],
                source="faq.json",
                metadata={"category": item["category"]},
                text=f"Вопрос: {item['question']}\nОтвет: {item['answer']}",
            )
        )

    playbook = _read(data_dir / "playbook.json")
    for principle in playbook["principles"]:
        chunks.append(
            Chunk(
                id=f"playbook-{principle['id']}",
                title=f"Принцип работы с клиентом: {principle['title']}",
                source="playbook.json",
                metadata={"category": "playbook", "internal": True},
                text=f"{principle['title']}. {principle['detail']}",
            )
        )
    for i, objection in enumerate(playbook.get("objection_handling", [])):
        chunks.append(
            Chunk(
                id=f"objection-{i}",
                title=f"Возражение: {objection['objection']}",
                source="playbook.json",
                metadata={"category": "playbook", "internal": True},
                text=f"Возражение клиента: «{objection['objection']}».\n"
                f"Как отвечать: {objection['response']}",
            )
        )

    return chunks


def load_playbook(data_dir: Path | None = None) -> dict[str, Any]:
    return _read((data_dir or DATA_DIR) / "playbook.json")
