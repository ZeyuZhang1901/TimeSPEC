"""Task-specific parsers for coalition evaluator outputs."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_prediction(task: str, text: str) -> dict[str, Any]:
    """Parse a compact task prediction from model output."""

    # Strip <think> tags from reasoning models
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    if task == "binary":
        payload = _extract_json_object(text, allow_missing=True) or {}
        prob = payload.get("probability_yes", payload.get("p_yes", payload.get("p")))
        if prob is None:
            prob = _extract_float_by_key(text, ("probability_yes", "p_yes", "p"))
        if prob is None:
            raise ValueError("Binary prediction missing probability_yes")
        p = float(prob)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Binary probability out of range: {p}")
        pred = str(payload.get("prediction", "yes" if p >= 0.5 else "no")).lower()
        return {"probability_yes": p, "prediction": pred}

    if task == "multiple_choice":
        payload = _extract_json_object(text, allow_missing=True) or {}
        probs = payload.get("probabilities", {})
        if not isinstance(probs, dict) or not probs:
            raise ValueError("Multiple-choice prediction missing probabilities dict")
        total = sum(float(v) for v in probs.values())
        if total > 0:
            probs = {k: float(v) / total for k, v in probs.items()}
        pred = str(payload.get("prediction", max(probs, key=lambda k: probs[k])))
        return {"probabilities": probs, "prediction": pred}

    if task == "numeric":
        payload = _extract_json_object(text, allow_missing=True) or {}
        value = payload.get("prediction")
        if value is None:
            value = _extract_float_by_key(text, ("prediction", "value", "estimate"))
        if value is None:
            raise ValueError("Numeric prediction missing prediction value")
        return {"prediction": float(value)}

    if task == "legal":
        payload = _extract_json_object(text, allow_missing=True) or {}
        if "p" in payload and "probability_petitioner" not in payload:
            payload["probability_petitioner"] = payload["p"]
        if "probability_petitioner" in payload:
            prob = float(payload["probability_petitioner"])
        else:
            prob = _extract_legal_probability(text)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"Legal probability out of range: {prob}")
        return {"probability_petitioner": prob}

    payload = _extract_json_object(text, allow_missing=True) or {}
    if task == "salary":
        value = payload.get("predicted_salary", payload.get("salary", payload.get("aav")))
        salary: float | None = None
        if value is not None:
            try:
                salary = float(value) if not isinstance(value, str) else _money_to_float(value)
            except (ValueError, TypeError):
                salary = None
        if salary is None:
            salary = _extract_money_by_key(text, ("predicted_salary", "salary", "aav"))
        if salary is None:
            raise ValueError("Salary prediction missing predicted_salary")
        if salary < 0:
            raise ValueError(f"Salary prediction must be non-negative: {salary}")
        return {"predicted_salary": float(salary)}
    if task == "stock":
        ranking = payload.get("ranking") if isinstance(payload, dict) else None
        if not isinstance(ranking, list) or len(ranking) < 2:
            ranking = _extract_ranking(text)
        if not isinstance(ranking, list) or len(ranking) < 2:
            raise ValueError("Stock prediction must contain ranking list")
        cleaned = [str(item).strip() for item in ranking if str(item).strip() and "TICKER" not in str(item).upper() and str(item).strip() != "..."]
        if len(cleaned) < 2:
            raise ValueError("Stock prediction must contain at least two concrete ranking entries")
        return {"ranking": cleaned}
    raise ValueError(f"Unsupported task: {task}")


def _extract_json_object(text: str, *, allow_missing: bool = False) -> dict[str, Any] | None:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        if allow_missing:
            return None
        raise ValueError("No JSON object found in model output")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        if allow_missing:
            return None
        raise
    if not isinstance(payload, dict):
        if allow_missing:
            return None
        raise ValueError("Model output JSON must be an object")
    return payload


def _extract_float_by_key(text: str, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        pattern = rf'"{re.escape(key)}"\s*:\s*([0-9]*\.?[0-9]+)'
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


_MULTIPLIERS = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}


def _money_to_float(raw: str) -> float | None:
    """Parse a currency-style string like ``"$12,500,000"`` or ``"7.5M"`` to float."""
    match = re.search(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([kKmMbB])?\b", raw)
    if not match:
        return None
    num = match.group(1).replace(",", "")
    mult = _MULTIPLIERS.get((match.group(2) or "").lower(), 1.0)
    try:
        return float(num) * mult
    except ValueError:
        return None


def _extract_money_by_key(text: str, keys: tuple[str, ...]) -> float | None:
    """Extract a currency-style numeric value for any of the given JSON keys.

    Accepts values with $, commas, K/M/B suffixes (e.g. ``"$12,500,000"``,
    ``"12.5M"``), which break the strict JSON parser.
    """
    for key in keys:
        pattern = rf'"{re.escape(key)}"\s*:\s*"?\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*([kKmMbB])?\b'
        match = re.search(pattern, text)
        if match:
            raw_num = match.group(1).replace(",", "")
            mult = _MULTIPLIERS.get((match.group(2) or "").lower(), 1.0)
            try:
                return float(raw_num) * mult
            except ValueError:
                continue
    return None


def _extract_ranking(text: str) -> list[str] | None:
    """Last-resort extractor for a JSON-style ticker list (e.g. ``["AAPL", "MSFT"]``)."""
    match = re.search(r'"ranking"\s*:\s*\[([^\]]+)\]', text)
    if not match:
        return None
    items = re.findall(r'"([A-Za-z][A-Za-z0-9._\-]*)"', match.group(1))
    return items if len(items) >= 2 else None


def _extract_legal_probability(text: str) -> float:
    patterns = [
        r'"probability_petitioner"\s*:\s*([01](?:\.\d+)?)',
        r'"p"\s*:\s*([01](?:\.\d+)?)',
        r"probability(?:\s+that)?(?:\s+the)?\s+petitioner(?:\s+wins)?\D+([01](?:\.\d+)?)",
        r"p_petitioner\D+([01](?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    raise ValueError("No legal probability found in model output")

