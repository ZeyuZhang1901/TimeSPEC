"""Dataset metadata export and validation for leakage evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


_UNSAFE_PROMPT_KEYS = {
    "outcome",
    "winner",
    "label",
    "conclusion",
    "decision_date",
    "vote_split",
    "majority_author",
    "aav",
    "actual_salary",
    "salary",
    "total_value",
    "contract_value",
    "ranking",
    "true_ranking",
    "returns",
    "future_returns",
}


def export_quality_gated_eval_metadata(data_root: Path, tasks: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Export metadata for QualityGatedExperiment evaluation datasets."""

    tasks = tasks or ["legal", "salary", "stock"]
    metadata: dict[str, dict[str, Any]] = {}
    if "legal" in tasks:
        metadata.update(_export_legal(data_root / "evaluation" / "legal_cases.json"))
    if "salary" in tasks:
        metadata.update(_export_salary(data_root / "evaluation" / "nba_salaries.csv"))
    if "stock" in tasks:
        metadata.update(_export_stock(data_root / "evaluation" / "stock_rankings.json"))
    return metadata


def validate_metadata_for_completions(completion_rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]) -> list[str]:
    """Return validation errors for completion rows and metadata."""

    errors: list[str] = []
    for row in completion_rows:
        instance_id = str(row.get("instance_id", ""))
        if not instance_id:
            errors.append("Completion row missing instance_id")
            continue
        meta = metadata.get(instance_id)
        if meta is None:
            errors.append(f"Missing metadata for {instance_id}")
            continue
        errors.extend(_validate_one(
            instance_id,
            str(row.get("task") or row.get("answer_type") or meta.get("task") or meta.get("answer_type") or ""),
            meta,
        ))
    return errors


def prompt_safe_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Return only metadata fields allowed in prediction prompts."""

    safe_keys = {
        "instance_id",
        "task",
        "question",
        "question_text",
        "cutoff_date",
        "target_description",
        "case_name",
        "court",
        "term",
        "petitioner",
        "respondent",
        "facts",
        "player",
        "sport",
        "year",
        "team",
        "from_team",
        "to_team",
        "status",
        "companies",
        "company_names",
        "sector",
        "window_id",
        "past_period",
    }
    return {key: value for key, value in meta.items() if key in safe_keys and key.lower() not in _UNSAFE_PROMPT_KEYS}


def write_metadata(path: Path, metadata: dict[str, dict[str, Any]]) -> None:
    """Write metadata JSON mapping keyed by instance_id."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)


def _export_legal(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_json(path)
    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        instance_id = str(row.get("case_id") or f"legal_eval_{idx}")
        case_name = str(row.get("case_name") or "")
        out[instance_id] = {
            "instance_id": instance_id,
            "task": "legal",
            "question": f"Predict the probability that the petitioner wins in {case_name}.",
            "cutoff_date": str(row.get("cutoff_date") or ""),
            "ground_truth": {"outcome": row.get("outcome", "")},
            "target_description": f"Whether the petitioner wins in {case_name}.",
            "case_name": case_name,
            "court": row.get("court", ""),
            "term": row.get("term", ""),
            "petitioner": row.get("petitioner", ""),
            "respondent": row.get("respondent", ""),
            "facts": row.get("facts", ""),
            "raw": row,
            "prompt_safe": prompt_safe_metadata(
                {
                    "instance_id": instance_id,
                    "task": "legal",
                    "question": f"Predict the probability that the petitioner wins in {case_name}.",
                    "cutoff_date": row.get("cutoff_date", ""),
                    "target_description": f"Whether the petitioner wins in {case_name}.",
                    "case_name": case_name,
                    "court": row.get("court", ""),
                    "term": row.get("term", ""),
                    "petitioner": row.get("petitioner", ""),
                    "respondent": row.get("respondent", ""),
                    "facts": row.get("facts", ""),
                }
            ),
        }
    return out


def _export_salary(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_csv(path)
    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        instance_id = f"salary_eval_{idx}"
        player = str(row.get("Player", "")).strip()
        year = str(row.get("Year", "")).strip()
        out[instance_id] = {
            "instance_id": instance_id,
            "task": "salary",
            "question": f"Predict {player}'s Average Annual Value for the {year} contract.",
            "cutoff_date": str(row.get("Signing_Date") or ""),
            "ground_truth": {"aav": _parse_money(row.get("AAV", "0"))},
            "target_description": f"{player}'s contract AAV.",
            "player": player,
            "sport": row.get("Sport", "NBA"),
            "year": year,
            "team": row.get("Team", row.get("To_Team", "")),
            "status": row.get("Status", ""),
            "raw": row,
            "prompt_safe": prompt_safe_metadata(
                {
                    "instance_id": instance_id,
                    "task": "salary",
                    "question": f"Predict {player}'s Average Annual Value for the {year} contract.",
                    "cutoff_date": str(row.get("Signing_Date") or ""),
                    "target_description": f"{player}'s contract AAV.",
                    "player": player,
                    "sport": row.get("Sport", "NBA"),
                    "year": year,
                    "team": row.get("Team", row.get("To_Team", "")),
                    "status": row.get("Status", ""),
                }
            ),
        }
    return out


def _export_stock(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    questions = payload.get("questions", payload) if isinstance(payload, dict) else payload
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    windows = metadata.get("windows", {}) if isinstance(metadata, dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(questions):
        qid = row.get("question_id", idx)
        instance_id = f"stock_eval_{qid}"
        period = row.get("past_period", {})
        window_id = str(row.get("window_id", ""))
        cutoff = row.get("model_knowledge_cutoff") or period.get("cutoff") or windows.get(window_id, {}).get("cutoff", "")
        ranking = period.get("ranking", row.get("true_ranking", []))
        out[instance_id] = {
            "instance_id": instance_id,
            "task": "stock",
            "question": "Rank the listed companies by expected stock return.",
            "cutoff_date": str(cutoff),
            "ground_truth": {"ranking": ranking},
            "target_description": "Future stock return ranking.",
            "companies": row.get("companies", []),
            "company_names": row.get("company_names", {}),
            "sector": row.get("sector", ""),
            "window_id": window_id,
            "past_period": _safe_past_period(period),
            "raw": row,
            "prompt_safe": prompt_safe_metadata(
                {
                    "instance_id": instance_id,
                    "task": "stock",
                    "question": "Rank the listed companies by expected stock return.",
                    "cutoff_date": str(cutoff),
                    "target_description": "Future stock return ranking.",
                    "companies": row.get("companies", []),
                    "company_names": row.get("company_names", {}),
                    "sector": row.get("sector", ""),
                    "window_id": window_id,
                    "past_period": _safe_past_period(period),
                }
            ),
        }
    return out


def _validate_one(instance_id: str, task: str, meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _NEW_ANSWER_TYPES = {"binary", "multiple_choice", "numeric"}
    _OLD_TASKS = {"legal", "salary", "stock"}
    _ALL_TASKS = _NEW_ANSWER_TYPES | _OLD_TASKS

    # For new benchmark schema, required keys differ from old QGE schema
    if task in _NEW_ANSWER_TYPES:
        # Validate against new dataset schema
        if not str(meta.get("question", "")).strip():
            errors.append(f"{instance_id}: missing question")
        if not (meta.get("temporal_cutoff") or meta.get("cutoff_date")):
            errors.append(f"{instance_id}: missing cutoff_date/temporal_cutoff")
        if not str(meta.get("answer", "")).strip():
            errors.append(f"{instance_id}: missing answer")
        return errors

    # Legacy QGE schema validation
    for key in ("task", "question", "cutoff_date", "ground_truth"):
        if key not in meta or meta[key] in ("", None):
            errors.append(f"{instance_id}: missing {key}")
    if task == "legal":
        for key in ("case_name", "petitioner", "respondent"):
            if not str(meta.get(key, "")).strip():
                errors.append(f"{instance_id}: missing legal field {key}")
        if not str(meta.get("ground_truth", {}).get("outcome", "")).strip():
            errors.append(f"{instance_id}: missing legal ground_truth.outcome")
    elif task == "salary":
        if float(meta.get("ground_truth", {}).get("aav", 0.0)) <= 0:
            errors.append(f"{instance_id}: salary ground_truth.aav must be positive")
    elif task == "stock":
        ranking = meta.get("ground_truth", {}).get("ranking", [])
        if not isinstance(ranking, list) or len(ranking) < 2:
            errors.append(f"{instance_id}: stock ground_truth.ranking must contain at least two items")
    elif task not in _ALL_TASKS:
        errors.append(f"{instance_id}: unsupported task {task}")
    unsafe = set(meta.get("prompt_safe", {})) & _UNSAFE_PROMPT_KEYS
    if unsafe:
        errors.append(f"{instance_id}: prompt_safe contains unsafe keys {sorted(unsafe)}")
    return errors


def _safe_past_period(period: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in period.items() if key not in {"ranking", "returns"}}


def _parse_money(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).replace("$", "").replace(",", "").strip()
    return float(text) if text else 0.0


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
