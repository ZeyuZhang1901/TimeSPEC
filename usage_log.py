"""Opt-in, thread-safe per-call usage logging for cost accounting (camera-ready E6).

Enabled by setting the environment variable ``CR_USAGE_LOG`` to a JSONL path.
When unset, ``record`` is a no-op, so instrumented code paths behave exactly
as before. Each record captures the component (which client), the pipeline
stage, the model, token counts, provider-reported cost when available, and
wall-clock latency.
"""

from __future__ import annotations

import json
import os
import threading
import time

_LOCK = threading.Lock()
_PATH: str | None = None
_RESOLVED = False


def _path() -> str | None:
    global _PATH, _RESOLVED
    if not _RESOLVED:
        _PATH = os.getenv("CR_USAGE_LOG") or None
        _RESOLVED = True
    return _PATH


def record(
    *,
    component: str,
    stage: str,
    model: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost: float | None = None,
    latency_s: float | None = None,
    extra: dict | None = None,
) -> None:
    path = _path()
    if path is None:
        return
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "component": component,
        "stage": stage,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        "latency_s": round(latency_s, 4) if latency_s is not None else None,
    }
    if extra:
        row.update(extra)
    line = json.dumps(row, ensure_ascii=False)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def record_response(
    *,
    component: str,
    stage: str,
    model: str,
    response: object,
    latency_s: float,
) -> None:
    """Extract usage from an OpenAI-SDK response object and record it."""
    usage = getattr(response, "usage", None)
    cost = None
    prompt_tokens = completion_tokens = total_tokens = None
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        cost = getattr(usage, "cost", None)
        if cost is None:
            model_extra = getattr(usage, "model_extra", None) or {}
            cost = model_extra.get("cost")
    record(
        component=component,
        stage=stage,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        latency_s=latency_s,
    )
