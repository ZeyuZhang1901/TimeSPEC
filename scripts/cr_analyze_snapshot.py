"""E3: compare Snapshot-RAG vs. live Temporal RAG vs. TimeSPEC with bootstrap CIs.

All three arms are evaluated on the same matched question set. Reports per-arm
means with 95% percentile bootstrap CIs and paired bootstrap CIs for the
DCLR/OLR deltas between arms.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_METRICS = ("performance", "olr", "shapley_dclr", "full_reconstruction_gap")


def load_instances(path: Path, ids: set[str]) -> dict[str, dict]:
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row["instance_id"] in ids:
                out[row["instance_id"]] = row
    return out


def bootstrap_mean_ci(values: list[float], rng: random.Random, n_boot: int = 10000) -> dict:
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    return {
        "mean": round(sum(values) / n, 4),
        "ci95": [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4)],
        "n": n,
    }


def paired_bootstrap_delta(a: list[float], b: list[float], rng: random.Random, n_boot: int = 10000) -> dict:
    n = len(a)
    deltas = sorted(
        sum(a[i] - b[i] for i in (rng.randrange(n) for _ in range(n))) / n
        for _ in range(n_boot)
    )
    point = sum(x - y for x, y in zip(a, b)) / n
    return {
        "delta_mean": round(point, 4),
        "ci95": [round(deltas[int(0.025 * n_boot)], 4), round(deltas[int(0.975 * n_boot)], 4)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Matched-question jsonl (snapshot run dataset)")
    parser.add_argument("--arm", action="append", required=True,
                        help="name=path/to/instance_results.jsonl (repeatable)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ids = {json.loads(line)["id"] for line in Path(args.dataset).open(encoding="utf-8") if line.strip()}
    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = load_instances(Path(path), ids)

    # Restrict to questions evaluated in every arm (parse failures can differ).
    common = set(ids)
    for name, rows in arms.items():
        common &= set(rows)
    print(f"matched questions: {len(ids)}, common across arms: {len(common)}")
    order = sorted(common)

    rng = random.Random(args.seed)
    report = {"n_matched": len(ids), "n_common": len(common), "methods": {}, "paired_deltas": {}}
    for name, rows in arms.items():
        method_summary = {}
        for metric in _METRICS:
            values = [float(rows[iid][metric]) for iid in order if rows[iid].get(metric) is not None]
            if values:
                method_summary[metric] = bootstrap_mean_ci(values, rng)
        method_summary["high_medium_faithfulness_rate"] = round(
            sum(rows[iid]["faithfulness_status"] != "low_faithfulness" for iid in order) / len(order), 4
        )
        report["methods"][name] = method_summary

    names = list(arms)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            a_name, b_name = names[i], names[j]
            for metric in ("shapley_dclr", "olr", "performance"):
                a_vals = [float(arms[a_name][iid][metric]) for iid in order]
                b_vals = [float(arms[b_name][iid][metric]) for iid in order]
                report["paired_deltas"][f"{a_name}_minus_{b_name}:{metric}"] = paired_bootstrap_delta(
                    a_vals, b_vals, rng
                )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "paired_deltas"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
