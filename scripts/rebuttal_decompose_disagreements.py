"""Decompose the nine disputed retrieval paragraphs into atomic claims."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import (
    OpenRouterConfig,
    OpenRouterTextClient,
)
from LeakageEvaluationFramework.leakage_eval.schemas import ClaimRecord
from LeakageEvaluationFramework.leakage_eval.taxonomy.date_verifier import LlmDateVerifier


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"
MODEL = "google/gemini-3.1-flash-lite"

DECOMPOSE_SYSTEM = """Decompose a retrieved forecasting passage into exhaustive atomic factual claims.
Each claim must contain exactly one independently verifiable factual assertion.
Preserve names, dates, quantities, temporal qualifiers, negation, and source attribution.
Include explicit absence/non-occurrence assertions as claims.
Do not include opinions, forecasts, confidence statements, or reasoning.
Do not add or correct information. Return JSON only."""


async def decompose_one(
    client: OpenRouterTextClient,
    source: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[str]]:
    prompt = (
        f"CUTOFF CONTEXT (do not use it to alter the text): {source['cutoff_date']}\n"
        f"PASSAGE:\n{source['retrieved_text']}\n\n"
        'Return {"claims":[{"id":1,"text":"..."}, ...]}.'
    )
    async with semaphore:
        raw = await client.call_text(prompt, system=DECOMPOSE_SYSTEM)
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError(f"No JSON decomposition for {source['sample_id']}")
    payload = json.loads(match.group())
    claims = [
        str(item["text"]).strip()
        for item in payload.get("claims", [])
        if isinstance(item, dict) and str(item.get("text", "")).strip()
    ]
    if not claims:
        raise ValueError(f"Empty decomposition for {source['sample_id']}")
    return source["sample_id"], claims


async def decompose_all(sources: list[dict]) -> dict[str, list[str]]:
    client = OpenRouterTextClient(
        OpenRouterConfig(model=MODEL, temperature=0.0, max_tokens=4096)
    )
    semaphore = asyncio.Semaphore(5)
    pairs = await asyncio.gather(
        *(decompose_one(client, source, semaphore) for source in sources)
    )
    return dict(pairs)


async def verify_parent(
    client: OpenRouterTextClient,
    source: dict,
    claims: list[str],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    records = [
        ClaimRecord(
            instance_id=source["sample_id"],
            claim_id=index,
            claim_text=claim,
            declared_source_date=None,
            cutoff_date=source["cutoff_date"],
            task="binary",
            target_description=source["question"],
            model="retrieval_audit",
            method="atomic_verifier",
        )
        for index, claim in enumerate(claims, 1)
    ]
    async with semaphore:
        decisions = await LlmDateVerifier(client, batch_size=20).verify(records)
    return [
        {
            "parent_id": source["sample_id"],
            "claim_id": decision.claim_id,
            "claim_text": decision.claim_text,
            "cutoff_date": source["cutoff_date"],
            "source_url": source["source_url"],
            "automatic_category": decision.category,
            "automatic_effective_date": decision.effective_source_date,
            "automatic_is_leaked": decision.is_leaked,
            "automatic_confidence": decision.confidence,
            "automatic_reason": decision.reason,
        }
        for decision in decisions
    ]


async def verify_all(
    source_by_id: dict[str, dict],
    decompositions: dict[str, list[str]],
) -> list[dict]:
    client = OpenRouterTextClient(
        OpenRouterConfig(model=MODEL, temperature=0.0, max_tokens=4096)
    )
    semaphore = asyncio.Semaphore(4)
    groups = await asyncio.gather(
        *(
            verify_parent(client, source_by_id[parent_id], claims, semaphore)
            for parent_id, claims in decompositions.items()
        )
    )
    return [row for group in groups for row in group]


def main() -> None:
    disagreement_report = json.loads(
        (AUDIT / "verifier_vs_professor.json").read_text(encoding="utf-8")
    )
    disputed_ids = {row["sample_id"] for row in disagreement_report["disagreements"]}
    all_sources = [
        json.loads(line)
        for line in (AUDIT / "conservative_auto_results.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    sources = [row for row in all_sources if row["sample_id"] in disputed_ids]
    source_by_id = {row["sample_id"]: row for row in sources}
    decompositions = asyncio.run(decompose_all(sources))
    verified = asyncio.run(verify_all(source_by_id, decompositions))

    with (AUDIT / "atomic_disagreement_claims.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in verified:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Blinded human sheet: automatic labels are intentionally excluded.
    lines = [
        "# Atomic Claims from the Nine Disputed Retrieval Paragraphs",
        "",
        "Label each atomic claim:",
        "",
        "- `A`: publicly knowable before cutoff;",
        "- `B`: knowable only after cutoff;",
        "- `C`: insufficient historical evidence;",
        "- `F`: factually false/unsupported as stated.",
        "",
        "Respond as `R###.N | A/B/C/F | H/M/L | optional note`.",
        "",
    ]
    for parent_id in sorted(decompositions):
        source = source_by_id[parent_id]
        lines.extend(
            [
                f"## {parent_id}",
                "",
                f"- Cutoff: {source['cutoff_date']}",
                f"- Source: {source['source_url']}",
                "",
            ]
        )
        for index, claim in enumerate(decompositions[parent_id], 1):
            lines.append(f"- **{parent_id}.{index}:** {claim}")
        lines.append("")
    (AUDIT / "atomic_disagreement_human_sheet.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    private = {
        "parents": len(decompositions),
        "atomic_claims": len(verified),
        "automatic_leaked": sum(row["automatic_is_leaked"] for row in verified),
        "automatic_safe": sum(not row["automatic_is_leaked"] for row in verified),
    }
    (AUDIT / "atomic_disagreement_summary.json").write_text(
        json.dumps(private, indent=2), encoding="utf-8"
    )
    print(json.dumps(private, indent=2))


if __name__ == "__main__":
    main()
