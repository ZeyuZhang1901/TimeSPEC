"""E6: aggregate the per-call usage log into a per-stage cost/latency table.

Produces the camera-ready cost table: per pipeline stage — calls, prompt and
completion tokens, provider-reported dollars, and latency statistics — plus
per-method and grand totals, cross-checkable against the /auth/key snapshots
in camera_ready/usage_snapshots.jsonl.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def cost_value(cost) -> float:
    if isinstance(cost, (int, float)):
        return float(cost)
    if isinstance(cost, dict):
        return float(cost.get("total_cost") or cost.get("cost") or 0.0)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usage-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-questions", type=int, default=None,
                        help="For per-question normalization")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.usage_log).open(encoding="utf-8") if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["component"], row["stage"], row["model"])].append(row)

    table = []
    for (component, stage, model), grp in sorted(groups.items()):
        latencies = [r["latency_s"] for r in grp if r.get("latency_s") is not None]
        entry = {
            "component": component,
            "stage": stage,
            "model": model,
            "calls": len(grp),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in grp),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in grp),
            "cost_usd": round(sum(cost_value(r.get("cost")) for r in grp), 4),
            "latency_mean_s": round(statistics.mean(latencies), 2) if latencies else None,
            "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
            "latency_p95_s": round(sorted(latencies)[int(0.95 * len(latencies))], 2) if len(latencies) >= 20 else None,
            "latency_total_s": round(sum(latencies), 1) if latencies else None,
        }
        if args.n_questions:
            entry["cost_per_question_usd"] = round(entry["cost_usd"] / args.n_questions, 5)
        table.append(entry)

    total_cost = round(sum(e["cost_usd"] for e in table), 4)
    report = {
        "n_calls_total": len(rows),
        "total_cost_usd": total_cost,
        "n_questions": args.n_questions,
        "cost_per_question_usd": round(total_cost / args.n_questions, 5) if args.n_questions else None,
        "stages": table,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    header = f"{'component':<22} {'stage':<24} {'calls':>6} {'p_tok':>9} {'c_tok':>9} {'cost$':>8} {'lat_mean':>9}"
    print(header)
    for entry in table:
        print(f"{entry['component']:<22} {entry['stage']:<24} {entry['calls']:>6} "
              f"{entry['prompt_tokens']:>9} {entry['completion_tokens']:>9} "
              f"{entry['cost_usd']:>8.3f} {str(entry['latency_mean_s']):>9}")
    print(f"TOTAL: ${total_cost}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
