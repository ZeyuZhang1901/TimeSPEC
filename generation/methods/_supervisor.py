"""Shared C0-C8 supervisor logic for B2 (LLM-only) and B4 (LLM + search)."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import TYPE_CHECKING, Any

from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import (
    OpenRouterConfig,
    OpenRouterTextClient,
)
from LeakageEvaluationFramework.leakage_eval.schemas import ClaimRecord
from LeakageEvaluationFramework.leakage_eval.taxonomy.date_verifier import LlmDateVerifier
from LeakageEvaluationFramework.leakage_eval.taxonomy.search_escalation import should_escalate

if TYPE_CHECKING:
    from LeakageEvaluationFramework.leakage_eval.schemas import ClaimDecision

logger = logging.getLogger(__name__)

_SEARCH_REVERIFY_SYSTEM = """You are a temporal provenance auditor. You have been given a claim and search results that were retrieved to verify the public availability date of that claim.

Based on the search results, determine the earliest reliable date on which the information in the claim would first have been publicly knowable.

Return only valid JSON: {"effective_source_date": "YYYY-MM-DD or null", "is_leaked": true/false, "confidence": "high|medium|low", "reason": "brief explanation"}"""


def run_llm_supervisor(
    instance_id: str,
    evidence: list[dict[str, Any]],
    temporal_cutoff: str,
    answer_type: str,
    question: str,
    model: str,
    method: str,
    or_client: OpenRouterTextClient,
) -> list["ClaimDecision"]:
    """Run LLM-only C0-C8 taxonomy verification (B2).

    Args:
        instance_id: Identifier for the parent instance.
        evidence: List of evidence dicts with keys 'id', 'fact', 'source_date'.
        temporal_cutoff: YYYY-MM-DD cutoff date for the instance.
        answer_type: "binary", "multiple_choice", or "numeric".
        question: Question text used as target_description.
        model: Name of the generator model.
        method: Method name ("b2" or "b4").
        or_client: Configured OpenRouter client.

    Returns:
        List of ClaimDecision objects, one per evidence item.
    """

    import asyncio
    records = _build_claim_records(
        instance_id, evidence, temporal_cutoff, answer_type, question, model, method
    )
    if not records:
        return []
    verifier = LlmDateVerifier(or_client)
    return asyncio.run(verifier.verify(records))


def run_search_supervisor(
    instance_id: str,
    evidence: list[dict[str, Any]],
    temporal_cutoff: str,
    answer_type: str,
    question: str,
    model: str,
    method: str,
    or_client: OpenRouterTextClient,
    search_client: Any,
    judge_model: str,
) -> list["ClaimDecision"]:
    """Run C0-C8 taxonomy verification with search escalation (B4).

    First runs LlmDateVerifier, then escalates uncertain/borderline claims
    to Perplexity search and re-verifies those using the search snippets.

    Args:
        search_client: PerplexitySearchClient instance.
        judge_model: OpenRouter model for re-verification after search.
        (other args same as run_llm_supervisor)

    Returns:
        List of ClaimDecision objects with search-updated decisions where applicable.
    """

    decisions = run_llm_supervisor(
        instance_id, evidence, temporal_cutoff, answer_type,
        question, model, method, or_client
    )
    escalation_needed = [
        d for d in decisions if should_escalate(d, temporal_cutoff)
    ]
    if not escalation_needed:
        return decisions

    logger.info(
        "Search escalation for %s: %d / %d claims",
        instance_id, len(escalation_needed), len(decisions),
    )
    updated = _search_and_reverify(
        escalation_needed, temporal_cutoff, or_client, search_client, judge_model
    )
    updated_by_id = {d.claim_id: d for d in updated}
    return [updated_by_id.get(d.claim_id, d) for d in decisions]


def leaked_claim_ids(decisions: list["ClaimDecision"]) -> list[int]:
    """Return claim IDs from decisions where is_leaked is True."""

    return [d.claim_id for d in decisions if d.is_leaked]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_claim_records(
    instance_id: str,
    evidence: list[dict[str, Any]],
    temporal_cutoff: str,
    answer_type: str,
    question: str,
    model: str,
    method: str,
) -> list[ClaimRecord]:
    records: list[ClaimRecord] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        fact = item.get("fact", "")
        if not isinstance(fact, str) or not fact.strip():
            continue
        records.append(ClaimRecord(
            instance_id=instance_id,
            claim_id=int(item.get("id", len(records) + 1)),
            claim_text=fact.strip(),
            declared_source_date=item.get("source_date") or None,
            cutoff_date=temporal_cutoff,
            task=answer_type,  # type: ignore[arg-type]
            target_description=question,
            model=model,
            method=method,
        ))
    return records


def _search_and_reverify(
    decisions: list["ClaimDecision"],
    temporal_cutoff: str,
    or_client: OpenRouterTextClient,
    search_client: Any,
    judge_model: str,
) -> list["ClaimDecision"]:
    import asyncio
    import json

    updated: list["ClaimDecision"] = []
    for decision in decisions:
        try:
            results = search_client.search(
                [decision.claim_text], cutoff_date=None, max_results=3
            )
            if not results:
                updated.append(decision)
                continue
            snippets = "\n".join(
                f"[{i+1}] {r.get('title', '')} ({r.get('date', 'unknown date')}): {r.get('snippet', '')}"
                for i, r in enumerate(results[:3])
            )
            prompt = (
                f"Claim: {decision.claim_text}\n"
                f"Declared source date: {decision.declared_source_date or 'unknown'}\n"
                f"Cutoff date: {temporal_cutoff}\n\n"
                f"Search results:\n{snippets}\n\n"
                "Based on these search results, what is the earliest reliable public availability date for this claim? "
                "Return JSON only."
            )
            raw = asyncio.run(or_client.call_text(prompt, system=_SEARCH_REVERIFY_SYSTEM))
            reverified = _parse_reverify_response(decision, raw, temporal_cutoff)
            updated.append(reverified)
        except Exception as exc:
            logger.warning(
                "Search escalation failed for claim %d: %s", decision.claim_id, exc
            )
            updated.append(decision)
    return updated


def _parse_reverify_response(
    original: "ClaimDecision",
    text: str,
    temporal_cutoff: str,
) -> "ClaimDecision":
    from LeakageEvaluationFramework.leakage_eval.schemas import ClaimDecision as CD

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return original
    try:
        import json
        data = json.loads(match.group())
    except Exception:
        return original

    eff_date = data.get("effective_source_date")
    if not eff_date or not isinstance(eff_date, str):
        return original

    try:
        eff = date.fromisoformat(eff_date)
        cutoff = date.fromisoformat(temporal_cutoff)
        is_leaked = eff > cutoff
    except ValueError:
        return original

    return CD(
        instance_id=original.instance_id,
        claim_id=original.claim_id,
        claim_text=original.claim_text,
        category=original.category,
        declared_source_date=original.declared_source_date,
        effective_source_date=eff_date,
        is_leaked=is_leaked,
        verification_method="search_escalation",
        date_precision="search_supported",
        confidence=data.get("confidence", "medium"),  # type: ignore[arg-type]
        reason=data.get("reason", "Date inferred from search results."),
        declared_date_plausible=original.declared_date_plausible,
        date_corrected=True,
    )
