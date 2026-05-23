"""Deduplicate batch_eval JSONL outputs and recompute summary.

The batch pipeline appended one row per (worker run × instance), so each
killed-and-restarted worker rewrote the same instances. Because the underlying
API results are deterministic, duplicate rows are identical — we can safely
keep the last occurrence per primary key and rebuild summary.json.

Usage:
    uv run python LeakageEvaluationFramework/scripts/dedupe_batch_results.py \\
        --root LeakageEvaluationFramework/runtime/claude_sonnet4/eval_results_self_batch
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Callable, Iterable

from LeakageEvaluationFramework.leakage_eval.metrics.aggregation import aggregate_instance_rows


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _dedupe_last(rows: list[dict], key_fn: Callable[[dict], object]) -> list[dict]:
    seen: dict[object, dict] = {}
    for r in rows:
        seen[key_fn(r)] = r
    return list(seen.values())


def dedupe_one_baseline(out_dir: Path) -> dict:
    backup_dir = out_dir / "dedupe_backup"
    backup_dir.mkdir(exist_ok=True)

    files_pk = [
        ("instance_results.jsonl", lambda r: r["instance_id"]),
        ("shapley_results.jsonl",  lambda r: r["instance_id"]),
        ("claim_results.jsonl",    lambda r: (r["instance_id"], r["claim_id"])),
    ]
    stats = {"baseline": out_dir.name}
    for fname, key_fn in files_pk:
        src = out_dir / fname
        if not src.exists():
            stats[fname] = "missing"
            continue
        if not (backup_dir / fname).exists():
            shutil.copy2(src, backup_dir / fname)
        rows = _read_jsonl(src)
        deduped = _dedupe_last(rows, key_fn)
        _write_jsonl(src, deduped)
        stats[fname] = {"before": len(rows), "after": len(deduped), "dropped": len(rows) - len(deduped)}

    instance_rows = _read_jsonl(out_dir / "instance_results.jsonl")
    summary = aggregate_instance_rows(instance_rows)
    summary_path = out_dir / "summary.json"
    if not (backup_dir / "summary.json").exists() and summary_path.exists():
        shutil.copy2(summary_path, backup_dir / "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    stats["summary_n_instances"] = summary["n_instances"]
    stats["summary_olr_mean"] = summary["olr"]["mean"]
    stats["summary_dclr_mean"] = summary["shapley_dclr"]["mean"]
    stats["summary_performance_mean"] = summary["performance"]["mean"]
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="Parent directory containing per-baseline subdirs (b1, b2, b3, b4).")
    p.add_argument("--baselines", nargs="+", default=["b1", "b2", "b3", "b4"])
    args = p.parse_args()

    for bm in args.baselines:
        out_dir = args.root / bm
        if not out_dir.is_dir():
            print(f"{bm}: skipped (no dir)")
            continue
        stats = dedupe_one_baseline(out_dir)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
