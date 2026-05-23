"""Re-parse every parse_success=False completion in-place using the lenient parser.

Costs zero API calls — operates exclusively on the saved raw_text. For each
rescued row we:
  - update prediction fields (ranking / probability_petitioner / predicted_salary /
    probability / probabilities / prediction / evidence),
  - flip parse_success to True,
  - leave raw_text untouched.

A .bak copy of every completions.json is written before any mutation, so the
operation is fully reversible.

Usage:
    uv run python -m LeakageEvaluationFramework.generation.scripts.relenient_reparse \
        --root runtime/task_sensitivity_full/qwen35_35b
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from LeakageEvaluationFramework.generation.parsers import parse_completion_lenient


_METHODS = ("b1", "b2", "b3", "b4")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Model output directory (e.g. runtime/task_sensitivity_full/qwen35_35b)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report rescue counts without writing.")
    args = parser.parse_args(argv)

    if not args.root.exists():
        raise SystemExit(f"root not found: {args.root}")

    summary: dict[str, dict[str, int]] = {}
    for method in _METHODS:
        path = args.root / method / "completions.json"
        if not path.exists():
            print(f"  [skip] {method}: no completions.json")
            continue

        rows = json.loads(path.read_text(encoding="utf-8"))
        rescued, still_failed, examined = _reparse_rows(rows)
        summary[method] = {
            "total": len(rows),
            "examined": examined,
            "rescued": rescued,
            "still_failed": still_failed,
        }

        if not args.dry_run and rescued:
            backup = path.with_suffix(".json.bak")
            if not backup.exists():
                shutil.copy(path, backup)
            path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print()
    print(f"{'method':<8}{'total':<8}{'examined':<10}{'rescued':<10}{'still_failed':<14}{'new_rate':<10}")
    print("-" * 60)
    for m in _METHODS:
        if m not in summary:
            continue
        s = summary[m]
        new_pass = s["total"] - s["still_failed"]
        print(f"{m:<8}{s['total']:<8}{s['examined']:<10}{s['rescued']:<10}{s['still_failed']:<14}{new_pass/s['total']*100:5.1f}%")

    if args.dry_run:
        print("\n[dry-run] no files written")


def _reparse_rows(rows: list[dict]) -> tuple[int, int, int]:
    rescued = 0
    examined = 0
    still_failed = 0
    for row in rows:
        if row.get("parse_success"):
            continue
        examined += 1
        answer_type = row.get("answer_type")
        raw = row.get("raw_text", "") or ""
        parsed = parse_completion_lenient(answer_type, raw)
        if parsed is None:
            still_failed += 1
            continue
        _apply_parsed(row, answer_type, parsed)
        rescued += 1
    return rescued, still_failed, examined


def _apply_parsed(row: dict, answer_type: str, parsed: dict) -> None:
    """Splice rescued fields into the original row in place."""

    row["evidence"] = parsed.get("evidence", row.get("evidence", []))
    row["reasoning"] = parsed.get("reasoning", row.get("reasoning", ""))
    row["parse_success"] = True

    if answer_type == "binary":
        row["probability"] = parsed.get("probability_yes")
        row["prediction"] = parsed.get("prediction", "")
    elif answer_type == "multiple_choice":
        row["probabilities"] = parsed.get("probabilities")
        row["prediction"] = parsed.get("prediction", "")
    elif answer_type == "numeric":
        row["prediction"] = parsed.get("prediction", "")
    elif answer_type == "legal":
        row["probability_petitioner"] = parsed.get("probability_petitioner")
    elif answer_type == "salary":
        row["predicted_salary"] = parsed.get("predicted_salary")
    elif answer_type == "stock":
        row["ranking"] = parsed.get("ranking")


if __name__ == "__main__":
    main()
