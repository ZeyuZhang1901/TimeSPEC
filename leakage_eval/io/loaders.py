"""Load method outputs and dataset metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from LeakageEvaluationFramework.leakage_eval.schemas import EvalInput, EvidenceItem

from .metadata import validate_metadata_for_completions


def load_json(path: Path) -> Any:
    """Load JSON from a file."""

    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_completion_rows(path: Path) -> list[dict[str, Any]]:
    """Load completion rows from a JSON file or JSONL file."""

    if path.suffix == ".jsonl":
        from .jsonl import read_jsonl

        return read_jsonl(path)
    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("completions"), list):
        rows = payload["completions"]
    else:
        raise ValueError(f"Unsupported completion file shape: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Completion rows must be JSON objects: {path}")
    return rows


def load_dataset_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Load dataset metadata keyed by instance_id.

    Accepted shapes:
    - JSONL file (one object per line)
    - JSON list of objects
    - JSON object with an "instances" list
    - JSON mapping from instance_id to metadata object
    """

    # JSONL files: one JSON object per line
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        payload = load_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("instances"), list):
            rows = payload["instances"]
        elif isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            return {str(k): v for k, v in payload.items() if isinstance(v, dict)}
        else:
            raise ValueError(f"Unsupported dataset metadata shape: {path}")

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Dataset metadata row must be an object in {path}")
        # Accept both "instance_id" (old eval format) and "id" (EMNLP dataset schema)
        row_id = row.get("instance_id") or row.get("id")
        if not row_id:
            raise ValueError(f"Dataset metadata row missing instance_id/id in {path}")
        out[str(row_id)] = row
    return out


def build_eval_inputs(
    completion_rows: list[dict[str, Any]],
    dataset_metadata: dict[str, dict[str, Any]],
    *,
    model: str,
    method: str,
) -> list[EvalInput]:
    """Build canonical evaluation inputs from method outputs and metadata.

    The method output file intentionally contains only generated outputs.
    Question text, cutoff dates, and ground truth must come from dataset
    metadata keyed by the same ``instance_id``.
    """

    validation_errors = validate_metadata_for_completions(completion_rows, dataset_metadata)
    if validation_errors:
        preview = "; ".join(validation_errors[:5])
        suffix = f" ... and {len(validation_errors) - 5} more" if len(validation_errors) > 5 else ""
        raise ValueError(f"Invalid dataset metadata: {preview}{suffix}")

    inputs: list[EvalInput] = []
    skipped: list[str] = []
    for row in completion_rows:
        instance_id = _require_str(row, "instance_id")
        if instance_id not in dataset_metadata:
            raise KeyError(f"Missing dataset metadata for instance_id={instance_id}")

        # Skip completions that failed to parse during generation (no usable evidence)
        if not row.get("parse_success", True) or not row.get("evidence"):
            import logging
            logging.getLogger(__name__).warning(
                "Skipping instance %s: parse_success=%s evidence_count=%d",
                instance_id, row.get("parse_success"), len(row.get("evidence") or []),
            )
            skipped.append(instance_id)
            continue

        meta = dataset_metadata[instance_id]
        task = str(row.get("task") or row.get("answer_type") or meta.get("task") or meta.get("answer_type") or "")
        _NEW_ANSWER_TYPES = {"binary", "multiple_choice", "numeric"}
        _OLD_TASKS = {"legal", "salary", "stock"}
        if task not in _OLD_TASKS | _NEW_ANSWER_TYPES:
            raise ValueError(f"Unsupported or missing task/answer_type for {instance_id}: {task}")

        payload = _parse_completion_payload_for_row(row)
        inputs.append(
            EvalInput(
                instance_id=instance_id,
                task=task,  # type: ignore[arg-type]
                question=_question_from_metadata(meta),
                cutoff_date=_cutoff_date_from_metadata(meta),
                ground_truth=_ground_truth_from_metadata(meta),
                evidence=_evidence_from_payload(payload, instance_id=instance_id),
                prediction=_prediction_from_payload(task, payload),
                reasoning=_optional_payload_str(payload, "reasoning"),
                model=str(row.get("model") or meta.get("model") or model),
                method=str(row.get("method") or meta.get("method") or method),
                target_description=str(meta.get("target_description") or ""),
                raw={"completion_row": row, "dataset_metadata": meta},
            )
        )
    return inputs


def _parse_completion_payload_for_row(row: dict[str, Any]) -> dict[str, Any]:
    """Detect new-format completions (generation pipeline) vs legacy completions.

    New format: completion fields (evidence, prediction, reasoning) are at the top
    level of the row dict — no nested ``completion`` wrapper.
    Legacy format: row has a ``completion`` field (string or dict).
    """
    if "completion" not in row and "evidence" in row:
        # New generation pipeline format — row IS the payload
        return dict(row)
    return parse_completion_payload(row.get("completion"))


def _cutoff_date_from_metadata(meta: dict[str, Any]) -> str:
    """Read cutoff date from metadata, accepting both old and new field names."""
    for key in ("cutoff_date", "temporal_cutoff"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("Dataset metadata must contain cutoff_date or temporal_cutoff")


def parse_completion_payload(completion: Any) -> dict[str, Any]:
    """Parse one structured method completion payload."""

    if isinstance(completion, dict):
        return completion
    if not isinstance(completion, str):
        raise ValueError(f"Completion must be a JSON string or object, got {type(completion).__name__}")

    text = re.sub(r"<think>[\s\S]*?</think>", "", completion, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("Completion does not contain a JSON object") from None
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise ValueError("Completion JSON must be an object")
    return payload


def _evidence_from_payload(payload: dict[str, Any], *, instance_id: str) -> list[EvidenceItem]:
    raw_evidence = payload.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise ValueError(f"Completion for {instance_id} must contain a non-empty evidence list")

    evidence: list[EvidenceItem] = []
    for idx, item in enumerate(raw_evidence, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Evidence item {idx} for {instance_id} must be an object")
        fact = item.get("fact")
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError(f"Evidence item {idx} for {instance_id} is missing fact text")
        raw_id = item.get("id", idx)
        try:
            claim_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Evidence item {idx} for {instance_id} has non-integer id: {raw_id}") from exc
        source_date = item.get("source_date")
        evidence.append(
            EvidenceItem(
                id=claim_id,
                fact=fact,
                source_date=str(source_date) if source_date not in (None, "") else None,
            )
        )
    return evidence


def _prediction_from_payload(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    # --- new answer-type-based format ---
    if task == "binary":
        prob = payload.get("probability_yes", payload.get("probability", payload.get("p")))
        pred = payload.get("prediction", "")
        if prob is None:
            raise ValueError("Binary completion is missing probability_yes or probability")
        probability = float(prob)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Binary probability out of range: {probability}")
        return {"probability_yes": probability, "prediction": str(pred)}
    if task == "multiple_choice":
        probs = payload.get("probabilities")
        pred  = payload.get("prediction", "")
        if not isinstance(probs, dict) or not probs:
            raise ValueError("Multiple-choice completion is missing probabilities dict")
        return {"probabilities": {str(k): float(v) for k, v in probs.items()}, "prediction": str(pred)}
    if task == "numeric":
        value = payload.get("prediction")
        if value is None:
            raise ValueError("Numeric completion is missing prediction value")
        return {"prediction": float(value)}
    # --- legacy task format ---
    if task == "legal":
        prob = payload.get("probability_petitioner", payload.get("p"))
        if prob is None:
            raise ValueError("Legal completion is missing probability_petitioner")
        probability = float(prob)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Legal probability out of range: {probability}")
        return {"probability_petitioner": probability}
    if task == "salary":
        salary = payload.get("predicted_salary", payload.get("salary"))
        if salary is None:
            raise ValueError("Salary completion is missing predicted_salary")
        value = float(str(salary).replace("$", "").replace(",", ""))
        if value < 0:
            raise ValueError(f"Salary prediction must be non-negative: {value}")
        return {"predicted_salary": value}
    if task == "stock":
        ranking = payload.get("ranking")
        if not isinstance(ranking, list) or len(ranking) < 2:
            raise ValueError("Stock completion must contain a ranking list with at least two items")
        return {"ranking": [str(item) for item in ranking]}
    raise ValueError(f"Unsupported task: {task}")


def _question_from_metadata(meta: dict[str, Any]) -> str:
    for key in ("question", "question_text", "prompt", "event_description"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raw = meta.get("raw")
    if isinstance(raw, dict):
        for key in ("question", "question_text", "prompt", "case_name"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
    raise ValueError("Dataset metadata must contain question, question_text, prompt, or raw.case_name")


def _ground_truth_from_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    ground_truth = meta.get("ground_truth")
    if isinstance(ground_truth, dict):
        if "true_ranking" in ground_truth and "ranking" not in ground_truth:
            return {**ground_truth, "ranking": ground_truth["true_ranking"]}
        return ground_truth
    # New benchmark dataset schema
    if "answer" in meta:
        return {"answer": meta["answer"], "answer_type": meta.get("answer_type", "")}
    # Legacy fields
    if "outcome" in meta:
        return {"outcome": meta["outcome"]}
    if "winner" in meta:
        return {"winner": meta["winner"]}
    if "aav" in meta:
        return {"aav": meta["aav"]}
    if "ranking" in meta:
        return {"ranking": meta["ranking"]}
    if "true_ranking" in meta:
        return {"ranking": meta["true_ranking"]}
    raise ValueError("Dataset metadata is missing ground_truth")


def _require_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string field: {key}")
    return value


def _require_payload_str(payload: dict[str, Any], key: str, instance_id: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Completion for {instance_id} is missing {key}")
    return value


def _optional_payload_str(payload: dict[str, Any], key: str) -> str:
    """Lenient read for optional fields that are not consumed by metrics.

    Rescued completions may have empty reasoning when the original generation
    was truncated mid-JSON; only evidence and prediction are required by the
    downstream pipeline, so missing reasoning is downgraded to "".
    """
    value = payload.get(key)
    return value if isinstance(value, str) else ""

