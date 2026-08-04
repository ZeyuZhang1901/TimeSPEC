"""Date plausibility and binary leakage helpers."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from LeakageEvaluationFramework.leakage_eval.audit.logger import AuditLogger, stable_text_hash
from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import OpenRouterTextClient
from LeakageEvaluationFramework.leakage_eval.schemas import ClaimDecision, ClaimRecord

from .classifier import heuristic_category
from .prompts import TAXONOMY_DATE_SYSTEM_PROMPT, build_taxonomy_date_prompt

_CATEGORIES = {"C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"}


class LlmDateVerifier:
    """LLM-backed C0-C8 taxonomy and effective-date verifier."""

    def __init__(self, client: OpenRouterTextClient, *, batch_size: int = 100, audit_logger: AuditLogger | None = None) -> None:
        self.client = client
        self.batch_size = batch_size
        self.audit_logger = audit_logger

    async def verify(self, records: list[ClaimRecord]) -> list[ClaimDecision]:
        """Verify claim categories and binary leakage labels."""

        decisions: list[ClaimDecision] = []
        for start in range(0, len(records), self.batch_size):
            chunk = records[start : start + self.batch_size]
            decisions.extend(await self._verify_chunk(chunk))
        return decisions

    async def _verify_chunk(self, records: list[ClaimRecord]) -> list[ClaimDecision]:
        if not records:
            return []
        first = records[0]
        items = [
            {
                "id": record.claim_id,
                "claim": record.claim_text,
                "declared_source_date": record.declared_source_date,
            }
            for record in records
        ]
        prompt = build_taxonomy_date_prompt(
            json.dumps(items, ensure_ascii=False, separators=(",", ":")),
            cutoff_date=first.cutoff_date,
            task=first.task,
            target_description=first.target_description,
        )
        text = await self.client.call_text(prompt, system=TAXONOMY_DATE_SYSTEM_PROMPT, stage="supervisor_taxonomy")
        by_id = _parse_llm_decisions(text)
        if self.audit_logger is not None:
            self.audit_logger.log_taxonomy_batch(
                {
                    "instance_id": first.instance_id,
                    "task": first.task,
                    "judge_model": self.client.config.model,
                    "claim_ids": [record.claim_id for record in records],
                    "prompt_hash": stable_text_hash(TAXONOMY_DATE_SYSTEM_PROMPT + "\n" + prompt),
                    "raw_response": text,
                    "parsed_results": by_id,
                    "usage": _usage_dict(self.client.last_usage),
                }
            )
        return [_decision_from_llm(record, by_id.get(record.claim_id)) for record in records]


def local_claim_decision(record: ClaimRecord) -> ClaimDecision:
    """Build a deterministic local claim decision.

    This fallback trusts parseable declared dates for time-verifiable claims.
    Production runs should replace or augment it with the LLM date verifier.
    """

    category = heuristic_category(record)
    declared = _parse_date(record.declared_source_date)
    cutoff = date.fromisoformat(record.cutoff_date)

    if category in {"C7", "C8"}:
        return _decision(record, category, record.declared_source_date, True, "category_rule", "category_rule", "high")

    if category in {"C0", "C1"} and declared is None:
        return _decision(record, category, None, False, "category_rule", "category_rule", "medium")

    if declared is None:
        return _decision(
            record,
            category,
            None,
            True,
            "missing_date_conservative",
            "missing",
            "low",
            reason="Missing source_date for time-verifiable claim; conservative leakage label.",
        )

    return _decision(
        record,
        category,
        declared.isoformat(),
        declared > cutoff,
        "declared_date_local",
        "declared",
        "medium",
    )


def _decision(
    record: ClaimRecord,
    category: str,
    effective_date: str | None,
    is_leaked: bool,
    method: str,
    precision: str,
    confidence: str,
    *,
    reason: str | None = None,
    declared_date_plausible: bool | None = None,
    date_corrected: bool = False,
    postcheck_flags: list[str] | None = None,
    needs_manual_review: bool = False,
) -> ClaimDecision:
    return ClaimDecision(
        instance_id=record.instance_id,
        claim_id=record.claim_id,
        claim_text=record.claim_text,
        category=category,  # type: ignore[arg-type]
        declared_source_date=record.declared_source_date,
        effective_source_date=effective_date,
        is_leaked=is_leaked,
        verification_method=method,
        date_precision=precision,
        confidence=confidence,  # type: ignore[arg-type]
        reason=reason or f"Local deterministic decision using category {category}.",
        declared_date_plausible=declared_date_plausible,
        date_corrected=date_corrected,
        postcheck_flags=postcheck_flags or [],
        needs_manual_review=needs_manual_review,
    )


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_llm_decisions(text: str) -> dict[int, dict[str, Any]]:
    payload = _extract_json_object(text)
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Taxonomy/date verifier output must contain a results list")
    out: dict[int, dict[str, Any]] = {}
    for item in raw_results:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            claim_id = int(item["id"])
        except (TypeError, ValueError):
            continue
        out[claim_id] = item
    return out


def _extract_json_object(text: str) -> dict[str, Any]:
    """Tolerant JSON extractor for the taxonomy/date verifier output.

    The judge model occasionally returns nearly-valid JSON whose ``reason``
    fields contain unescaped quotes, trailing commas, or other minor breakage.
    We try strict parsing first, then ``json_repair`` on the bracketed slice,
    and only raise if both fail. This keeps a whole instance from erroring
    out because one of its taxonomy claim entries had a quote in the reason.
    """
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        raise ValueError("No JSON object found in taxonomy/date verifier output")
    candidate = match.group(0)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
        except ImportError as exc:
            raise ValueError(f"Taxonomy/date verifier output is malformed JSON: {exc}") from None
        repaired = repair_json(candidate)
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Taxonomy/date verifier output remains malformed after json_repair: {exc}"
            ) from None
    if not isinstance(payload, dict):
        raise ValueError("Taxonomy/date verifier output JSON must be an object")
    return payload


def _decision_from_llm(record: ClaimRecord, item: dict[str, Any] | None) -> ClaimDecision:
    if item is None:
        local = local_claim_decision(record)
        return _replace_method(
            local,
            "llm_missing_local_fallback",
            f"LLM verifier omitted claim {record.claim_id}; used local conservative fallback.",
        )

    category = str(item.get("category", "")).upper()
    if category not in _CATEGORIES:
        category = heuristic_category(record)

    declared_plausible = item.get("declared_date_plausible")
    declared_plausible = declared_plausible if isinstance(declared_plausible, bool) else None
    effective_date = item.get("effective_source_date") or item.get("corrected_date") or record.declared_source_date
    effective_date = str(effective_date) if effective_date not in (None, "") else None
    parsed_effective = _parse_date(effective_date)
    parsed_declared = _parse_date(record.declared_source_date)
    cutoff = date.fromisoformat(record.cutoff_date)
    flags: list[str] = []

    if category in {"C7", "C8"}:
        is_leaked = True
        if item.get("is_leaked") is False:
            flags.append("overrode_llm_c7_c8_must_leak")
    elif parsed_effective is None and category not in {"C0", "C1"}:
        is_leaked = True
        flags.append("missing_effective_date_conservative_leak")
    else:
        is_leaked = bool(parsed_effective and parsed_effective > cutoff)
    if isinstance(item.get("is_leaked"), bool) and bool(item["is_leaked"]) != is_leaked:
        flags.append("llm_label_overridden_by_effective_date")

    date_corrected = bool(parsed_declared and parsed_effective and parsed_declared != parsed_effective)
    if date_corrected:
        flags.append("declared_date_corrected")
    if parsed_effective is None:
        flags.append("missing_effective_date")
    if category in {"C0", "C1"} and parsed_effective and parsed_effective > cutoff:
        flags.append("stable_or_background_post_cutoff_date")
    if declared_plausible is False and not date_corrected:
        flags.append("declared_implausible_without_correction")

    confidence = str(item.get("confidence", "medium")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    precision = str(item.get("date_precision") or ("missing" if effective_date is None else "llm_effective"))
    if precision in {"conservative_latest", "missing"} and confidence == "high":
        confidence = "medium"
        flags.append("confidence_downgraded_for_uncertain_date")
    if date_corrected and confidence == "high":
        confidence = "medium"
        flags.append("confidence_downgraded_for_corrected_date")
    reason = str(item.get("reason") or "LLM taxonomy/date verifier decision.")
    needs_review = bool(flags) or confidence == "low"

    return _decision(
        record,
        category,
        effective_date,
        is_leaked,
        "llm_taxonomy_date",
        precision,
        confidence,
        reason=reason,
        declared_date_plausible=declared_plausible,
        date_corrected=date_corrected,
        postcheck_flags=flags,
        needs_manual_review=needs_review,
    )


def _replace_method(decision: ClaimDecision, method: str, reason: str) -> ClaimDecision:
    return ClaimDecision(
        instance_id=decision.instance_id,
        claim_id=decision.claim_id,
        claim_text=decision.claim_text,
        category=decision.category,
        declared_source_date=decision.declared_source_date,
        effective_source_date=decision.effective_source_date,
        is_leaked=decision.is_leaked,
        verification_method=method,
        date_precision=decision.date_precision,
        confidence=decision.confidence,
        reason=reason,
        declared_date_plausible=decision.declared_date_plausible,
        date_corrected=decision.date_corrected,
        postcheck_flags=decision.postcheck_flags,
        needs_manual_review=True,
    )


def _usage_dict(usage: Any | None) -> dict[str, int] | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": int(usage.prompt_tokens),
        "completion_tokens": int(usage.completion_tokens),
        "total_tokens": int(usage.total_tokens),
    }

