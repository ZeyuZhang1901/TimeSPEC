"""Persistent state for batch evaluation jobs.

Saves batch IDs and per-request manifests to disk so a multi-hour batch run
can survive process restarts. Each (model, baseline) job lives in its own
output directory, with the following state files:

    state/
        taxonomy_batch.json    {batch_id, submitted_at, request_count}
        taxonomy_requests.jsonl  {custom_id, instance_id, claim_ids: [...]}
        shapley_batches.json   [{batch_id, submitted_at, request_count}, ...]
        shapley_requests.jsonl   {custom_id, instance_id, mask: [...]}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via tmp + os.replace.

    Guarantees the file is either fully written or unchanged — never partial.
    A crash between the tmp write and the rename leaves the previous version
    intact (or no file at all on first write).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


@dataclass
class BatchHandle:
    """Reference to a submitted batch."""

    batch_id: str
    submitted_at: str
    request_count: int


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


class JobState:
    """Per-job (one model, one baseline) state on disk."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.state_dir = self.output_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # ------------- taxonomy -------------

    def save_taxonomy_batch(self, batch_id: str, request_count: int) -> None:
        h = BatchHandle(batch_id=batch_id, submitted_at=_now(), request_count=request_count)
        _atomic_write_text(self.state_dir / "taxonomy_batch.json", json.dumps(asdict(h), indent=2))

    def load_taxonomy_batch(self) -> BatchHandle | None:
        path = self.state_dir / "taxonomy_batch.json"
        if not path.exists():
            return None
        return BatchHandle(**json.loads(path.read_text()))

    def save_taxonomy_requests(self, records: list[dict]) -> None:
        with (self.state_dir / "taxonomy_requests.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def load_taxonomy_requests(self) -> list[dict]:
        path = self.state_dir / "taxonomy_requests.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # ------------- shapley -------------

    def save_shapley_batches(self, batches: list[BatchHandle]) -> None:
        _atomic_write_text(
            self.state_dir / "shapley_batches.json",
            json.dumps([asdict(b) for b in batches], indent=2),
        )

    def load_shapley_batches(self) -> list[BatchHandle]:
        path = self.state_dir / "shapley_batches.json"
        if not path.exists():
            return []
        return [BatchHandle(**d) for d in json.loads(path.read_text())]

    def save_shapley_requests(self, records: list[dict]) -> None:
        with (self.state_dir / "shapley_requests.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def load_shapley_requests(self) -> list[dict]:
        path = self.state_dir / "shapley_requests.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # ------------- phase markers -------------

    def mark_taxonomy_done(self) -> None:
        (self.state_dir / "taxonomy_done.flag").write_text(_now())

    def is_taxonomy_done(self) -> bool:
        return (self.state_dir / "taxonomy_done.flag").exists()

    def mark_shapley_done(self) -> None:
        (self.state_dir / "shapley_done.flag").write_text(_now())

    def is_shapley_done(self) -> bool:
        return (self.state_dir / "shapley_done.flag").exists()
