"""E1: inter-annotator agreement between the two independent C0-C8 annotators.

Joins annotator 1 (rebuttal_validation/human_taxonomy_120/human_labels.csv) and
annotator 2 (camera_ready/e1_human/annotator2_labels.csv) via their private ID
mappings, and scores category / tier agreement (accuracy + Cohen's kappa), each
annotator against the gold construction labels, annotator 2's leakage labels
against gold, and per-class confusion. Also re-runs annotator 2's internal
consistency checks. Follows the same conventions as
rebuttal_score_human_taxonomy_120.py (tiers: C0/C1 safe, C7/C8 leaked, rest
verify; balanced_90 = subset == "taxonomy").
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A1_ROOT = ROOT / "rebuttal_validation" / "human_taxonomy_120"
A2_ROOT = ROOT / "camera_ready" / "e1_human"
CHALLENGE = ROOT / "rebuttal_validation" / "data" / "taxonomy_date_challenge.jsonl"
CATEGORIES = [f"C{i}" for i in range(9)]


def tier(category: str) -> str:
    if category in {"C0", "C1"}:
        return "safe"
    if category in {"C7", "C8"}:
        return "leaked"
    return "verify"


def cohen_kappa(a: list[str], b: list[str], labels: list[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def agreement_block(a: list[str], b: list[str], labels: list[str]) -> dict:
    return {
        "n": len(a),
        "accuracy": round(sum(x == y for x, y in zip(a, b)) / len(a), 4),
        "cohen_kappa": round(cohen_kappa(a, b, labels), 4),
    }


def main() -> None:
    gold = {json.loads(l)["id"]: json.loads(l) for l in CHALLENGE.open(encoding="utf-8") if l.strip()}

    a1_rows = list(csv.DictReader((A1_ROOT / "human_labels.csv").open(encoding="utf-8")))
    a1_map = {r["human_id"]: r["original_id"] for r in json.loads((A1_ROOT / "private_id_mapping.json").read_text())}
    a1 = {a1_map[r["human_id"]]: r for r in a1_rows}

    a2_rows = list(csv.DictReader((A2_ROOT / "annotator2_labels.csv").open(encoding="utf-8")))
    a2_mapping = json.loads((A2_ROOT / "annotator2_private_id_mapping.json").read_text())
    a2_map = {r["annotator2_id"]: r["original_id"] for r in a2_mapping}
    subset_of = {r["original_id"]: r["subset"] for r in a2_mapping}
    a2 = {a2_map[r["annotator2_id"]]: r for r in a2_rows}

    ids = sorted(set(a1) & set(a2) & set(gold))
    assert len(ids) == 120, f"expected 120 joined claims, got {len(ids)}"

    # --- internal consistency of annotator 2 ---
    consistency_errors = []
    for oid in ids:
        row = a2[oid]
        cat = row["human_category"].strip()
        leaked = row["human_is_leaked"].strip().lower()
        eff = row["human_effective_date"].strip().lower()
        cutoff = row["cutoff_date"].strip()
        if cat not in CATEGORIES:
            consistency_errors.append((oid, f"bad category {cat!r}"))
            continue
        if cat in {"C7", "C8"} and leaked != "yes":
            consistency_errors.append((oid, "C7/C8 must be leaked"))
        if cat in {"C0", "C1"} and (eff != "timeless" or leaked != "no"):
            consistency_errors.append((oid, f"C0/C1 must be timeless/no (got {eff}/{leaked})"))
        if eff not in {"timeless", "unresolved"} and len(eff) == 10:
            expect = "yes" if eff > cutoff else "no"
            if cat not in {"C7", "C8"} and leaked != expect:
                consistency_errors.append((oid, f"leakage {leaked} inconsistent with {eff} vs cutoff {cutoff}"))

    def block(id_subset: list[str]) -> dict:
        a1_cat = [a1[i]["human_category"].strip() for i in id_subset]
        a2_cat = [a2[i]["human_category"].strip() for i in id_subset]
        g_cat = [gold[i]["gold_category"] for i in id_subset]
        a1_tier = [tier(c) for c in a1_cat]
        a2_tier = [tier(c) for c in a2_cat]
        g_tier = [tier(c) for c in g_cat]
        a2_leak = [a2[i]["human_is_leaked"].strip().lower() for i in id_subset]
        g_leak = ["yes" if gold[i]["gold_is_leaked"] else "no" for i in id_subset]

        confusion = defaultdict(int)
        for x, y in zip(a1_cat, a2_cat):
            if x != y:
                confusion[f"a1={x}|a2={y}"] += 1
        disagreements = [
            {
                "original_id": i,
                "a1": a1[i]["human_category"].strip(),
                "a2": a2[i]["human_category"].strip(),
                "gold": gold[i]["gold_category"],
                "claim": gold[i]["fact"][:100],
            }
            for i in id_subset
            if a1[i]["human_category"].strip() != a2[i]["human_category"].strip()
        ]
        return {
            "a1_vs_a2_category": agreement_block(a1_cat, a2_cat, CATEGORIES),
            "a1_vs_a2_tier": agreement_block(a1_tier, a2_tier, ["safe", "verify", "leaked"]),
            "a1_vs_gold_category": agreement_block(a1_cat, g_cat, CATEGORIES),
            "a2_vs_gold_category": agreement_block(a2_cat, g_cat, CATEGORIES),
            "a1_vs_gold_tier": agreement_block(a1_tier, g_tier, ["safe", "verify", "leaked"]),
            "a2_vs_gold_tier": agreement_block(a2_tier, g_tier, ["safe", "verify", "leaked"]),
            "a2_vs_gold_leakage": agreement_block(a2_leak, g_leak, ["yes", "no"]),
            "a1_a2_confusion_offdiagonal": dict(sorted(confusion.items(), key=lambda kv: -kv[1])),
            "disagreements": disagreements,
        }

    balanced = [i for i in ids if subset_of[i] == "taxonomy"]
    report = {
        "n_claims": len(ids),
        "annotator2_consistency_errors": [f"{o}: {m}" for o, m in consistency_errors],
        "all_120": block(ids),
        "balanced_90": block(balanced),
        "a2_confidence_distribution": dict(Counter(a2[i]["confidence"].strip().lower() for i in ids)),
        "a2_category_distribution": dict(Counter(a2[i]["human_category"].strip() for i in ids)),
        "a1_category_distribution": dict(Counter(a1[i]["human_category"].strip() for i in ids)),
    }

    out = A2_ROOT / "iaa_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "consistency_errors": len(consistency_errors),
        "all120_a1_vs_a2_kappa": report["all_120"]["a1_vs_a2_category"]["cohen_kappa"],
        "all120_a1_vs_a2_accuracy": report["all_120"]["a1_vs_a2_category"]["accuracy"],
        "all120_tier_kappa": report["all_120"]["a1_vs_a2_tier"]["cohen_kappa"],
        "all120_a2_vs_gold_leakage_kappa": report["all_120"]["a2_vs_gold_leakage"]["cohen_kappa"],
        "balanced90_a1_vs_a2_kappa": report["balanced_90"]["a1_vs_a2_category"]["cohen_kappa"],
        "n_disagreements_all120": len(report["all_120"]["disagreements"]),
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
