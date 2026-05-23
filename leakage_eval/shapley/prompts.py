"""Prompt builders for closed-world Shapley coalition evaluation."""

from __future__ import annotations

import json

from LeakageEvaluationFramework.leakage_eval.schemas import EvalInput, EvidenceItem

from LeakageEvaluationFramework.leakage_eval.io.metadata import prompt_safe_metadata


def build_single_coalition_prompt(instance: EvalInput, evidence_subset: list[EvidenceItem]) -> str:
    """Build a compact closed-world coalition prompt (single string).

    For backwards compatibility — returns system + user concatenated.
    Prefer build_coalition_messages() for API calls with prompt caching.
    """
    system, user = build_coalition_messages(instance, evidence_subset)
    if system:
        return f"{system}\n\n{user}"
    return user


def build_coalition_messages(
    instance: EvalInput,
    evidence_subset: list[EvidenceItem],
) -> tuple[str, str]:
    """Build (system, user) messages for a coalition evaluation.

    The system message contains the task instruction and question (stable
    across all coalitions for the same instance — cacheable).  The user
    message contains only the variable evidence subset.
    """

    if instance.task == "legal":
        return _build_legal_coalition_prompt(instance, evidence_subset)
    if instance.task == "stock":
        return _build_stock_coalition_prompt(instance, evidence_subset)
    if instance.task == "binary":
        return _build_binary_coalition_prompt(instance, evidence_subset)
    if instance.task == "multiple_choice":
        return _build_mc_coalition_prompt(instance, evidence_subset)
    if instance.task == "numeric":
        return _build_numeric_coalition_prompt(instance, evidence_subset)

    output = _output_instruction(instance.task)
    system = "You are an estimator. Output ONLY valid JSON, no other text."
    facts = "\n".join(f"{item.id}. {item.fact}" for item in evidence_subset)
    user = f"""Q: {instance.question}
Facts: {facts}
{output}"""
    return system, user


def _build_binary_coalition_prompt(instance: EvalInput, evidence_subset: list[EvidenceItem]) -> tuple[str, str]:
    system = (
        "You are a probability estimator. Output ONLY valid JSON (no markdown, no prose). "
        "The JSON object must contain the single key probability_yes — nothing else."
    )
    evidence_json = json.dumps(
        [{"id": item.id, "fact": item.fact} for item in evidence_subset],
        ensure_ascii=False, separators=(",", ":"),
    )
    user = f"""Estimate P(YES) for: {instance.question}
Evidence: {evidence_json}
Reply with exactly: {{"probability_yes": <float 0.0-1.0>}}"""
    return system, user


def _build_mc_coalition_prompt(instance: EvalInput, evidence_subset: list[EvidenceItem]) -> tuple[str, str]:
    choices_raw = _metadata_object(instance, "choices")
    if isinstance(choices_raw, list):
        choices = [str(c) for c in choices_raw]
    else:
        choices = []
    choices_str = "; ".join(choices) if choices else "(see question)"
    probs_schema = ", ".join(f'"{c}": <float>' for c in choices) if choices else '"<choice>": <float>'
    system = (
        "You are a probability estimator. Output ONLY valid JSON (no markdown, no prose). "
        "The JSON object must contain only the key probabilities — nothing else."
    )
    evidence_json = json.dumps(
        [{"id": item.id, "fact": item.fact} for item in evidence_subset],
        ensure_ascii=False, separators=(",", ":"),
    )
    user = f"""Predict probabilities over choices (must sum to 1.0).
Q: {instance.question}
Choices: {choices_str}
Evidence: {evidence_json}
Reply with exactly: {{"probabilities": {{{probs_schema}}}}}"""
    return system, user


def _build_numeric_coalition_prompt(instance: EvalInput, evidence_subset: list[EvidenceItem]) -> tuple[str, str]:
    unit = _metadata_value(instance, "unit") or ""
    unit_hint = f" in {unit}" if unit else ""
    system = (
        "You are a numeric estimator. Output ONLY valid JSON (no markdown, no prose). "
        "The JSON object must contain only the key prediction — nothing else."
    )
    evidence_json = json.dumps(
        [{"id": item.id, "fact": item.fact} for item in evidence_subset],
        ensure_ascii=False, separators=(",", ":"),
    )
    user = f"""Give a numeric estimate.
Q: {instance.question}
Evidence: {evidence_json}
Reply with exactly: {{"prediction": <float{unit_hint}>}}"""
    return system, user


def _build_legal_coalition_prompt(instance: EvalInput, evidence_subset: list[EvidenceItem]) -> tuple[str, str]:
    case_name = _metadata_value(instance, "case_name") or instance.question
    petitioner = _metadata_value(instance, "petitioner") or "the petitioner"
    respondent = _metadata_value(instance, "respondent") or "the respondent"
    system = (
        "You are a probability estimator. Output ONLY valid JSON (no markdown, no prose). "
        "The JSON object must contain only the key probability_petitioner — nothing else."
    )
    evidence_json = json.dumps(
        [{"id": item.id, "fact": item.fact} for item in evidence_subset],
        ensure_ascii=False, separators=(",", ":"),
    )
    user = f"""Predict P(petitioner wins).
Case: {case_name} | Petitioner: {petitioner} | Respondent: {respondent}
Evidence: {evidence_json}
Reply with exactly: {{"probability_petitioner": <float 0.0-1.0>}}"""
    return system, user


def _metadata_value(instance: EvalInput, key: str) -> str | None:
    for source in (instance.raw.get("dataset_metadata"), instance.raw.get("completion_row")):
        if isinstance(source, dict):
            prompt_safe = source.get("prompt_safe")
            if isinstance(prompt_safe, dict):
                value = prompt_safe.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
            raw = source.get("raw")
            if isinstance(raw, dict):
                raw_value = prompt_safe_metadata(raw).get(key)
                if isinstance(raw_value, str) and raw_value.strip():
                    return raw_value
    return None


def _metadata_object(instance: EvalInput, key: str) -> object | None:
    for source in (instance.raw.get("dataset_metadata"), instance.raw.get("completion_row")):
        if isinstance(source, dict):
            prompt_safe = source.get("prompt_safe")
            if isinstance(prompt_safe, dict) and key in prompt_safe:
                return prompt_safe[key]
            if key in source:
                return source[key]
    return None


def _build_stock_coalition_prompt(instance: EvalInput, evidence_subset: list[EvidenceItem]) -> tuple[str, str]:
    companies = _metadata_object(instance, "companies")
    company_names = _metadata_object(instance, "company_names")
    companies = companies if isinstance(companies, list) else []
    company_names = company_names if isinstance(company_names, dict) else {}
    universe = [
        {"ticker": str(ticker), "name": str(company_names.get(str(ticker), ticker))}
        for ticker in companies
    ]
    system = (
        "You are a stock analyst. Output ONLY valid JSON (no markdown, no prose). "
        "The JSON object must contain only the key ranking — nothing else."
    )
    evidence_json = json.dumps(
        [{"id": item.id, "fact": item.fact} for item in evidence_subset],
        ensure_ascii=False, separators=(",", ":"),
    )
    user = f"""Rank tickers by expected return.
Allowed: {json.dumps(universe, ensure_ascii=False, separators=(",",":"))}
Evidence: {evidence_json}
Reply with exactly: {{"ranking": ["TICKER1", "TICKER2", ...]}}"""
    return system, user


def _output_instruction(task: str) -> str:
    if task == "legal":
        return 'Output schema: {"p": number_between_0_and_1}'
    if task == "salary":
        return 'Output schema: {"salary": positive_number}'
    if task == "stock":
        return 'Output schema: {"ranking": ["TICKER1", "TICKER2", "..."]}'
    raise ValueError(f"Unsupported task: {task}")

