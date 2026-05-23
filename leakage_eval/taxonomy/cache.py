"""Persistent JSONL caches for taxonomy and leakage detection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from LeakageEvaluationFramework.leakage_eval.io.jsonl import append_jsonl, read_jsonl


class JsonlCache:
    """Small append-only JSONL cache keyed by stable hashes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._items = {row["key"]: row["value"] for row in read_jsonl(path) if "key" in row and "value" in row}

    def get(self, key_parts: list[Any]) -> Any | None:
        """Return cached value for key parts."""

        return self._items.get(stable_key(key_parts))

    def put(self, key_parts: list[Any], value: Any) -> None:
        """Store a value if it is not already cached."""

        key = stable_key(key_parts)
        if key in self._items:
            return
        self._items[key] = value
        append_jsonl(self.path, {"key": key, "value": value})


def stable_key(parts: list[Any]) -> str:
    """Build a stable SHA256 cache key."""

    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

