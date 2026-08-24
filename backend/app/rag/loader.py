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
    return f"{value:,} EUR".replace(",", " ") if isinstance(value, int | float) else str(value)


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
            title=f"About {company['name']}",
            source="services.json",
            metadata={"category": "company"},
            text=(
                f"{company['name']} is an independent family office founded in {company['founded']}.\n"
                f"Assets under advisory: {company['aum_eur']} EUR. Clients: {company['clients']} families.\n"
                f"Jurisdictions: {', '.join(company['jurisdictions'])}.\n"
                f"Fee model: {company['billing']}"
            ),
        )
    )

    for tier in services["tiers"]:
        excludes = (
            f"\nNot included:\n{_bullets(tier['excludes'])}" if tier.get("excludes") else ""
        )
        chunks.append(
            Chunk(
                id=f"tier-{tier['id']}",
                title=f"{tier['name']} tier pricing",
                source="services.json",
                metadata={"category": "pricing", "tier": tier["id"], "tier_name": tier["name"]},
                text=(
                    f"The {tier['name']} tier: {tier['tagline']}.\n"
                    f"Who it is for: {tier['target_client']}.\n"
                    f"Price: {_fmt_price(tier['price_monthly_eur'])} per month. "
                    f"Setup fee: {_fmt_price(tier['setup_fee_eur'])}. "
                    f"Minimum contract term: {tier['min_contract_months']} months. "
                    f"{tier['price_note']}.\n"
                    f"Onboarding: {tier['onboarding_weeks']} weeks. Team: {tier['team']}.\n"
                    f"Included in the tier:\n{_bullets(tier['includes'])}{excludes}"
                ),
            )
        )

    addons = services.get("addons", [])
    if addons:
        chunks.append(
            Chunk(
                id="addons",
                title="Additional services (add-ons)",
                source="services.json",
                metadata={"category": "pricing"},
                text="Additional services on top of the tier:\n"
                + "\n".join(
                    f"- {a['name']}: {_fmt_price(a['price_eur'])}, duration: {a['duration']}"
                    for a in addons
                ),
            )
        )

    discounts = services.get("discounts", [])
    if discounts:
        chunks.append(
            Chunk(
                id="discounts",
                title="Discounts and payment terms",
                source="services.json",
                metadata={"category": "pricing"},
                text="Current discounts and special terms:\n" + _bullets(discounts),
            )
        )

    for item in _read(data_dir / "faq.json")["items"]:
        chunks.append(
            Chunk(
                id=item["id"],
                title=item["question"],
                source="faq.json",
                metadata={"category": item["category"]},
                text=f"Question: {item['question']}\nAnswer: {item['answer']}",
            )
        )

    playbook = _read(data_dir / "playbook.json")
    for principle in playbook["principles"]:
        chunks.append(
            Chunk(
                id=f"playbook-{principle['id']}",
                title=f"Client-handling principle: {principle['title']}",
                source="playbook.json",
                metadata={"category": "playbook", "internal": True},
                text=f"{principle['title']}. {principle['detail']}",
            )
        )
    for i, objection in enumerate(playbook.get("objection_handling", [])):
        chunks.append(
            Chunk(
                id=f"objection-{i}",
                title=f"Objection: {objection['objection']}",
                source="playbook.json",
                metadata={"category": "playbook", "internal": True},
                text=f"Client objection: \"{objection['objection']}\".\n"
                f"How to respond: {objection['response']}",
            )
        )

    return chunks


def load_playbook(data_dir: Path | None = None) -> dict[str, Any]:
    return _read((data_dir or DATA_DIR) / "playbook.json")
