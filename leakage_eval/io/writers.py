"""Result writers for leakage evaluation jobs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .jsonl import append_jsonl


class ResultWriter:
    """Incremental writer for one evaluation job."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.audit_dir = output_dir / "audit"
        self.logs_dir = output_dir / "logs"
        for path in (self.output_dir, self.audit_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def write_config(self, config: Any) -> None:
        """Write the resolved job config."""

        self.write_json("config.json", config)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        """Write a run manifest."""

        self.write_json("manifest.json", manifest)

    def write_summary(self, summary: dict[str, Any]) -> None:
        """Write aggregate summary."""

        self.write_json("summary.json", summary)

    def write_json(self, name: str, payload: Any) -> None:
        """Write one JSON file relative to the job directory."""

        path = self.output_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2, sort_keys=True)

    def append_instance(self, row: Any) -> None:
        append_jsonl(self.output_dir / "instance_results.jsonl", row)

    def append_claim(self, row: Any) -> None:
        append_jsonl(self.output_dir / "claim_results.jsonl", row)

    def append_shapley(self, row: Any) -> None:
        append_jsonl(self.output_dir / "shapley_results.jsonl", row)

    def append_skipped_shapley(self, row: Any) -> None:
        append_jsonl(self.output_dir / "skipped_shapley.jsonl", row)

    def append_error(self, row: Any) -> None:
        append_jsonl(self.output_dir / "errors.jsonl", row)

    def append_low_faithfulness(self, row: Any) -> None:
        append_jsonl(self.audit_dir / "low_faithfulness_instances.jsonl", row)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value

