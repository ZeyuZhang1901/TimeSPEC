"""Audit logging helpers for leakage evaluation runs."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from LeakageEvaluationFramework.leakage_eval.io.jsonl import append_jsonl


class AuditLogger:
    """Append-only audit logger under a run's audit directory."""

    def __init__(self, audit_dir: Path) -> None:
        self.audit_dir = audit_dir
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_taxonomy_batch(self, row: dict[str, Any]) -> None:
        """Log one LLM taxonomy/date verifier batch."""

        with self._lock:
            append_jsonl(self.audit_dir / "taxonomy_batches.jsonl", row)

    def log_coalition(self, row: dict[str, Any]) -> None:
        """Log one evaluated Shapley coalition."""

        with self._lock:
            append_jsonl(self.audit_dir / "coalitions.jsonl", row)


def stable_text_hash(text: str) -> str:
    """Return a stable SHA256 hash for prompt/audit text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact_json_hash(value: Any) -> str:
    """Return a stable hash for a JSON-serializable value."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return stable_text_hash(payload)
