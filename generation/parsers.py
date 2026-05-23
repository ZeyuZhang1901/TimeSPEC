"""Answer-type-dispatched completion parser for generation methods."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_completion(answer_type: str, text: str) -> dict[str, Any] | None:
    """Parse raw LLM text into a structured completion dict.

    Args:
        answer_type: One of "binary", "multiple_choice", "numeric".
        text: Raw text from the LLM.

    Returns:
        Parsed dict on success, None on any parse or validation failure.
    """

    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    payload = _extract_json_object(text)
    if payload is None:
        return None

    evidence = _validate_evidence(payload.get("evidence"))
    reasoning = payload.get("reasoning", "")
    if not isinstance(reasoning, str):
        return None

    if answer_type == "binary":
        return _parse_binary(payload, evidence, reasoning)
    if answer_type == "multiple_choice":
        return _parse_multiple_choice(payload, evidence, reasoning)
    if answer_type == "numeric":
        return _parse_numeric(payload, evidence, reasoning)
    if answer_type == "legal":
        return _parse_legal(payload, evidence, reasoning)
    if answer_type == "salary":
        return _parse_salary(payload, evidence, reasoning)
    if answer_type == "stock":
        return _parse_stock(payload, evidence, reasoning)

    raise ValueError(f"Unknown answer_type: {answer_type!r}")


def extract_evidence_items(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the evidence list from a parsed completion, normalising item fields."""

    items: list[dict[str, Any]] = []
    for item in parsed.get("evidence", []):
        if not isinstance(item, dict):
            continue
        fact = item.get("fact", "")
        if not isinstance(fact, str) or not fact.strip():
            continue
        items.append({
            "id": int(item["id"]) if "id" in item else len(items) + 1,
            "fact": fact.strip(),
            "source_date": str(item["source_date"]) if item.get("source_date") else None,
        })
    return items


def _parse_binary(payload: dict, evidence: list | None, reasoning: str) -> dict | None:
    if evidence is None:
        return None
    try:
        prob = float(payload["probability_yes"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0.0 <= prob <= 1.0):
        return None
    prediction = str(payload.get("prediction", "")).strip().lower()
    if prediction not in {"yes", "no"}:
        prediction = "yes" if prob >= 0.5 else "no"
    return {
        "evidence": evidence,
        "reasoning": reasoning,
        "probability_yes": prob,
        "prediction": prediction,
    }


def _parse_multiple_choice(payload: dict, evidence: list | None, reasoning: str) -> dict | None:
    if evidence is None:
        return None
    probabilities = payload.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        return None
    try:
        probs = {str(k): float(v) for k, v in probabilities.items()}
    except (TypeError, ValueError):
        return None
    total = sum(probs.values())
    if total <= 0:
        return None
    # Normalize so probabilities sum to 1.0
    probs = {k: v / total for k, v in probs.items()}

    prediction = str(payload.get("prediction", "")).strip()
    if prediction not in probs:
        # Fall back to argmax
        prediction = max(probs, key=lambda k: probs[k])
    return {
        "evidence": evidence,
        "reasoning": reasoning,
        "probabilities": probs,
        "prediction": prediction,
    }


def _parse_numeric(payload: dict, evidence: list | None, reasoning: str) -> dict | None:
    if evidence is None:
        return None
    raw_pred = payload.get("prediction")
    try:
        prediction = float(raw_pred)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    unit = payload.get("unit")
    result: dict[str, Any] = {
        "evidence": evidence,
        "reasoning": reasoning,
        "prediction": prediction,
    }
    if unit is not None:
        result["unit"] = str(unit)
    return result


def _parse_legal(payload: dict, evidence: list | None, reasoning: str) -> dict | None:
    if evidence is None:
        return None
    raw = payload.get("probability_petitioner", payload.get("p"))
    try:
        prob = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not 0.0 <= prob <= 1.0:
        return None
    return {
        "evidence": evidence,
        "reasoning": reasoning,
        "probability_petitioner": prob,
    }


def _parse_salary(payload: dict, evidence: list | None, reasoning: str) -> dict | None:
    if evidence is None:
        return None
    raw = payload.get("predicted_salary", payload.get("salary"))
    if raw is None:
        return None
    try:
        # Tolerate "$12,500,000" string outputs.
        value = float(str(raw).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return {
        "evidence": evidence,
        "reasoning": reasoning,
        "predicted_salary": value,
    }


def _parse_stock(payload: dict, evidence: list | None, reasoning: str) -> dict | None:
    if evidence is None:
        return None
    raw = payload.get("ranking")
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    ranking = [str(item).strip().upper() for item in raw if str(item).strip()]
    if len(ranking) < 2:
        return None
    return {
        "evidence": evidence,
        "reasoning": reasoning,
        "ranking": ranking,
    }


def _validate_evidence(raw: Any) -> list[dict[str, Any]] | None:
    """Return a validated evidence list or None if it is absent or empty."""

    if not isinstance(raw, list) or not raw:
        return None
    valid: list[dict[str, Any]] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        fact = item.get("fact", "")
        if not isinstance(fact, str) or not fact.strip():
            continue
        valid.append({
            "id": int(item["id"]) if "id" in item else i,
            "fact": fact.strip(),
            "source_date": str(item["source_date"]) if item.get("source_date") else None,
        })
    return valid if valid else None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first well-formed JSON object from text."""

    # Prefer a fenced code block first.
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find all { positions and try to match a complete JSON object.
    for match in re.finditer(r"\{", text):
        candidate = text[match.start():]
        # Truncate at the last } to avoid runaway grabs.
        end = candidate.rfind("}")
        if end == -1:
            continue
        try:
            return json.loads(candidate[: end + 1])
        except json.JSONDecodeError:
            continue

    return None


# ===========================================================================
# Lenient parser for offline recovery of malformed JSON completions.
# Adds three repair layers on top of the strict parser:
#   (1) json_repair package handles unescaped inner quotes / missing commas,
#   (2) targeted regex extraction recovers a missing answer field even when
#       JSON repair truncated everything after the malformation,
#   (3) prose patterns like ``Prediction: 5000000`` rescue cases where the
#       model abandoned JSON format altogether.
# Only used for retroactive re-parsing of saved raw_text — never in live
# generation, where strict parsing is the right signal of output quality.
# ===========================================================================


def parse_completion_lenient(answer_type: str, text: str) -> dict[str, Any] | None:
    """Strict parser first; on miss or missing answer field, apply repairs."""

    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    payload = _extract_json_object(cleaned)

    if payload is None or not _has_answer_field(payload, answer_type):
        repaired = _try_json_repair(cleaned)
        if repaired is not None:
            payload = _merge_payloads(payload, repaired)

    if payload is None:
        payload = {}

    if not _has_answer_field(payload, answer_type):
        regex_val = _regex_extract_answer_field(cleaned, answer_type)
        if regex_val is not None:
            field, value = regex_val
            payload[field] = value

    if not payload:
        return None

    _canonicalize_answer_field(payload, answer_type)

    evidence = _validate_evidence(payload.get("evidence"))
    if evidence is None:
        evidence = _regex_extract_evidence(text)
    # In lenient mode we accept an empty evidence list if a valid prediction
    # was extracted by regex — better an evidence-empty completion than no
    # completion at all (downstream OLR/DCLR will see 0 claims for it).
    if evidence is None and _has_answer_field(payload, answer_type):
        evidence = []
    reasoning_raw = payload.get("reasoning", "")
    reasoning = reasoning_raw if isinstance(reasoning_raw, str) else str(reasoning_raw)

    if answer_type == "binary":
        return _parse_binary(payload, evidence, reasoning)
    if answer_type == "multiple_choice":
        return _parse_multiple_choice(payload, evidence, reasoning)
    if answer_type == "numeric":
        return _parse_numeric(payload, evidence, reasoning)
    if answer_type == "legal":
        return _parse_legal(payload, evidence, reasoning)
    if answer_type == "salary":
        return _parse_salary(payload, evidence, reasoning)
    if answer_type == "stock":
        return _parse_stock(payload, evidence, reasoning)
    return None


_FIELD_ALIASES = {
    "binary": ["probability_yes", "probability", "p"],
    "multiple_choice": ["probabilities"],
    "numeric": ["prediction"],
    "legal": ["probability_petitioner", "p", "probability"],
    "salary": ["predicted_salary", "salary", "aav"],
    "stock": ["ranking", "_ranking_", "rank", "ranked_tickers"],
}


def _has_answer_field(payload: dict | None, answer_type: str) -> bool:
    """Return True only if the answer field is present AND has the right type.

    A JSON-repair pass can leave the answer field stuffed with a malformed
    value (e.g. ranking as a comma-separated string, or a None probability).
    Returning False here lets the regex fallback override such cases.
    """
    if not isinstance(payload, dict):
        return False
    for k in _FIELD_ALIASES.get(answer_type, []):
        if k not in payload:
            continue
        v = payload[k]
        if v is None:
            continue
        if answer_type == "stock":
            if isinstance(v, list) and len(v) >= 2:
                return True
        elif answer_type == "multiple_choice":
            if isinstance(v, dict) and v:
                return True
        else:
            if isinstance(v, (int, float)):
                return True
            if isinstance(v, str):
                try:
                    float(v.replace("$", "").replace(",", "").strip())
                    return True
                except ValueError:
                    pass
    return False


def _canonicalize_answer_field(payload: dict, answer_type: str) -> None:
    """If the payload has an aliased answer field but not the canonical one,
    copy it under the canonical name in place.

    Defends against models that emit ``_ranking_`` instead of ``ranking``
    when sampling penalties discourage repeating prompt tokens.
    """
    canon = _FIELD_ALIASES.get(answer_type, [None])[0]
    if canon is None or canon in payload:
        return
    for alias in _FIELD_ALIASES.get(answer_type, [])[1:]:
        if alias in payload:
            payload[canon] = payload[alias]
            return


def _try_json_repair(text: str) -> dict[str, Any] | None:
    try:
        from json_repair import repair_json  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        obj = repair_json(text, return_objects=True)
    except Exception:
        return None
    if isinstance(obj, list) and obj:
        obj = obj[0]
    return obj if isinstance(obj, dict) else None


def _merge_payloads(
    primary: dict[str, Any] | None,
    secondary: dict[str, Any],
) -> dict[str, Any]:
    """Prefer keys from primary (strict); fill missing keys from secondary (repaired)."""

    if not primary:
        return dict(secondary)
    merged = dict(secondary)
    merged.update(primary)
    return merged


def _regex_extract_answer_field(text: str, answer_type: str) -> tuple[str, Any] | None:
    """Last-resort regex extraction for the answer field directly from raw text."""

    if answer_type == "legal":
        m = re.search(
            r'(?:probability_petitioner|"p")\s*[:=]\s*([0-9]*\.?[0-9]+)',
            text,
        )
        if m:
            try:
                v = float(m.group(1))
                if 0.0 <= v <= 1.0:
                    return "probability_petitioner", v
            except ValueError:
                pass
        return None

    if answer_type == "salary":
        patterns = [
            r'(?:"?predicted_salary"?|"?salary"?|"?aav"?)\s*[:=]\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)',
            r'(?:Prediction|AAV|Predicted\s+Salary)\s*[:=\-]\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)',
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                try:
                    v = float(m.group(1).replace(",", ""))
                    if v > 0:
                        return "predicted_salary", v
                except ValueError:
                    continue
        return None

    if answer_type == "stock":
        m = re.search(r'"?ranking"?\s*[:=]\s*(\[[^\]]+\])', text)
        if m:
            try:
                lst = json.loads(m.group(1).replace("'", '"'))
                if isinstance(lst, list) and len(lst) >= 2:
                    cleaned = [str(t).strip().upper() for t in lst if str(t).strip()]
                    if len(cleaned) >= 2:
                        return "ranking", cleaned
            except json.JSONDecodeError:
                pass
        # Prose ranking: "INCY>RVTY>HSIC>CI>JNJ" or "INCY, RVTY, HSIC, CI, JNJ"
        for pat in (
            r'\b([A-Z][A-Z0-9.]{0,5}(?:\s*[>→]\s*[A-Z][A-Z0-9.]{0,5}){2,})',
            r'(?:ranking|ordered)[^:]*?:\s*([A-Z][A-Z0-9.]{0,5}(?:\s*,\s*[A-Z][A-Z0-9.]{0,5}){2,})',
        ):
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                raw_seq = m.group(1)
                tokens = re.split(r"\s*[>→,]\s*", raw_seq)
                cleaned = [t.strip().rstrip(".,;:").upper() for t in tokens if t.strip()]
                cleaned = [t for t in cleaned if t]
                if len(cleaned) >= 3:
                    return "ranking", cleaned
        return None

    if answer_type == "binary":
        m = re.search(r'(?:probability_yes|"p")\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
        if m:
            try:
                v = float(m.group(1))
                if 0.0 <= v <= 1.0:
                    return "probability_yes", v
            except ValueError:
                pass
        return None

    return None


def _regex_extract_evidence(text: str) -> list[dict[str, Any]] | None:
    """Last-resort evidence extraction by matching {id, fact, source_date} triples.

    Used when JSON parsing + json_repair both fail to extract the evidence array.
    Returns at most 20 items to bound runtime; None if nothing matches.
    """

    pattern = re.compile(
        r'"id"\s*:\s*(\d+)\s*,\s*'
        r'"fact"\s*:\s*"((?:[^"\\]|\\.){10,2000}?)"\s*,\s*'
        r'"source_date"\s*:\s*"(\d{4}-\d{2}-\d{2})"',
        re.DOTALL,
    )
    items: list[dict[str, Any]] = []
    for m in pattern.finditer(text):
        try:
            items.append({
                "id": int(m.group(1)),
                "fact": m.group(2).strip(),
                "source_date": m.group(3),
            })
        except ValueError:
            continue
        if len(items) >= 20:
            break
    return items if items else None
