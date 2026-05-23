"""Configuration models for leakage evaluation jobs."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _pick_dataclass_fields(dc: type, raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in dataclasses.fields(dc)}
    return {k: v for k, v in raw.items() if k in allowed}


@dataclass(frozen=True)
class LeakageConfig:
    """Settings for taxonomy and leakage detection."""

    judge_model: str = "google/gemini-3.1-flash-lite"
    taxonomy_batch_size: int = 100
    taxonomy_concurrency: int = 20
    search_concurrency: int = 10
    search_only_on_escalation: bool = True


@dataclass(frozen=True)
class ShapleyConfig:
    """Settings for Shapley computation."""

    skip_if_no_leakage: bool = True
    use_paper_leverage_shap: bool = True
    evaluator_backend: str = "openrouter"
    evaluator_model_policy: str = "same_as_generator"
    evaluator_model_override: str = ""
    temperature: float = 0.0
    coalition_concurrency: int = 8
    coalition_budget_c: float = 15.0
    coalition_budget_cap: int = 1200
    exact_threshold_n: int = 8
    high_rigor_c: float = 25.0
    high_rigor_subset_size: int = 50
    random_seed: int = 42


@dataclass(frozen=True)
class EvalJobConfig:
    """Top-level evaluation job configuration."""

    input_path: Path
    dataset_path: Path
    output_dir: Path
    leakage: LeakageConfig
    shapley: ShapleyConfig


def load_config(path: Path, *, input_path: Path | None = None, dataset_path: Path | None = None, output_dir: Path | None = None) -> EvalJobConfig:
    """Load an evaluation job config from YAML.

    Args:
        path: YAML config path.
        input_path: Optional CLI override for method outputs.
        dataset_path: Optional CLI override for dataset metadata.
        output_dir: Optional CLI override for result directory.

    Returns:
        Parsed evaluation job config.
    """

    raw = _read_yaml(path)
    leakage = LeakageConfig(**_pick_dataclass_fields(LeakageConfig, raw.get("leakage", {})))
    shapley = ShapleyConfig(**_pick_dataclass_fields(ShapleyConfig, raw.get("shapley", {})))

    return EvalJobConfig(
        input_path=input_path or Path(raw["input_path"]),
        dataset_path=dataset_path or Path(raw["dataset_path"]),
        output_dir=output_dir or Path(raw["output_dir"]),
        leakage=leakage,
        shapley=shapley,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return payload

