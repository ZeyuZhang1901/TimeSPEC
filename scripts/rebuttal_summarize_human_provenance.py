"""Summarize the stratified human provenance audit with weighted estimates."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "rebuttal_validation" / "provenance_audit"
INPUT = AUDIT / "human_verification.csv"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def main() -> None:
    with INPUT.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    strata = defaultdict(list)
    for row in rows:
        strata[row["auto_label"]].append(row)

    audited_counts = Counter()
    for row in rows:
        if row["human_corrected_label"]:
            audited_counts[(row["auto_label"], row["human_corrected_label"])] += 1

    projected = Counter()
    intervals: dict[str, tuple[float, float]] = {}
    for auto_label, stratum_rows in strata.items():
        audited = [row for row in stratum_rows if row["human_corrected_label"]]
        if not audited:
            continue
        human_counts = Counter(row["human_corrected_label"] for row in audited)
        for human_label in ("A", "B", "C"):
            share = human_counts[human_label] / len(audited)
            projected[human_label] += share * len(stratum_rows)
            lo, hi = wilson(human_counts[human_label], len(audited))
            intervals[f"{auto_label}_to_{human_label}"] = (lo, hi)

    estimate = {label: projected[label] / len(rows) for label in ("A", "B", "C")}
    # The only sampled (rather than fully audited) stratum is auto-C: 20 of 74.
    c_rows = strata["C"]
    c_audited = [row for row in c_rows if row["human_corrected_label"]]
    c_counts = Counter(row["human_corrected_label"] for row in c_audited)
    overall_intervals = {}
    fully_audited = {
        label: len(strata[label])
        for label in ("A", "B")
    }
    for human_label in ("A", "B", "C"):
        lo, hi = wilson(c_counts[human_label], len(c_audited))
        fixed = (
            sum(
                1
                for auto_label in ("A", "B")
                for row in strata[auto_label]
                if row["human_corrected_label"] == human_label
            )
            / len(rows)
        )
        weight = len(c_rows) / len(rows)
        overall_intervals[human_label] = (fixed + weight * lo, fixed + weight * hi)

    report = {
        "n_total": len(rows),
        "n_human_labeled": sum(bool(row["human_corrected_label"]) for row in rows),
        "stratum_sizes": {label: len(strata[label]) for label in ("A", "B", "C")},
        "audited_auto_c": len(c_audited),
        "audited_auto_c_human_counts": dict(c_counts),
        "weighted_estimate": estimate,
        "weighted_95_percent_intervals": {
            label: {"low": values[0], "high": values[1]}
            for label, values in overall_intervals.items()
        },
        "scope": (
            "Single-annotator stratified audit: all automatic A/B items plus a "
            "random 20-of-74 sample of automatic C items. No inter-annotator kappa."
        ),
    }
    (AUDIT / "human_provenance_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Human Provenance Audit Summary",
        "",
        f"- Total retrievals: {report['n_total']}",
        f"- Human-labeled: {report['n_human_labeled']}",
        f"- Auto-C stratum audited: {len(c_audited)}/{len(c_rows)}",
        "",
        "## Weighted Estimate",
        "",
    ]
    names = {"A": "Verified pre-cutoff", "B": "Post-cutoff leakage", "C": "Unverifiable"}
    for label in ("A", "B", "C"):
        interval = overall_intervals[label]
        lines.append(
            f"- {names[label]}: {estimate[label]:.1%} "
            f"(approx. 95% interval {interval[0]:.1%}–{interval[1]:.1%})"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            report["scope"],
            "Human A labels may use contemporaneous public evidence beyond Wayback; "
            "they do not imply that the original cited URL itself had an archived pre-cutoff copy.",
        ]
    )
    (AUDIT / "human_provenance_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
