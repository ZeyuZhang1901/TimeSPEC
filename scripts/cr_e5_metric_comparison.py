"""E5 offline: compare Shapley-DCLR against simpler alternatives (OLR, Top-K, LOO).

Reads cached evaluation outputs (instance/claim/shapley results plus coalition
audit logs) and computes, per method and task:
  - OLR (unweighted leakage rate) vs. Shapley-DCLR: correlation + disagreements
  - Top-K leakage (K = 1, 3, 5) from Shapley rankings
  - LOO-weighted leakage where cached coalitions cover v(N \\ {i}); reports
    coverage so the top-up script knows what remains to evaluate.

No API calls. Everything derives from files on disk.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy import stats


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def analyze_method(method_dir: Path, supplement_path: Path | None = None) -> dict:
    instances = {r["instance_id"]: r for r in load_jsonl(method_dir / "instance_results.jsonl")}
    claims = defaultdict(dict)
    for row in load_jsonl(method_dir / "claim_results.jsonl"):
        claims[row["instance_id"]][row["claim_id"]] = bool(row["is_leaked"])
    shapley = {r["instance_id"]: r for r in load_jsonl(method_dir / "shapley_results.jsonl")}

    coalition_path = method_dir / "audit" / "coalitions.jsonl"
    coalitions = defaultdict(list)
    for row in load_jsonl(coalition_path):
        coalitions[row["instance_id"]].append(row)
    if supplement_path is not None:
        for row in load_jsonl(supplement_path):
            if row.get("value") is not None:
                coalitions[row["instance_id"]].append(row)

    per_instance = []
    loo_cov_num = loo_cov_den = 0
    for iid, inst in instances.items():
        srow = shapley.get(iid)
        cmap = claims.get(iid, {})
        if not cmap:
            continue
        if (not srow or not srow.get("claim_shapley")) and not any(cmap.values()):
            # Zero-leakage instance where Shapley was skipped: by the paper's
            # Appendix D convention all leakage metrics are exactly 0.
            per_instance.append({
                "instance_id": iid,
                "task": inst.get("task"),
                "n_claims": len(cmap),
                "olr": 0.0,
                "dclr": 0.0,
                "top1": 0.0, "top3": 0.0, "top5": 0.0,
                "loo_leakage": 0.0,
                "loo_covered": len(cmap),
                "shapley_skipped_no_leakage": True,
            })
            continue
        if not srow:
            continue
        claim_shapley = srow.get("claim_shapley") or []
        ranked = sorted(claim_shapley, key=lambda c: c["abs_shapley_value"], reverse=True)
        leak_ranked = [cmap.get(c["claim_id"], False) for c in ranked]
        n = len(leak_ranked)
        if n == 0:
            continue
        topk = {f"top{k}": (sum(leak_ranked[:k]) / min(k, n)) for k in (1, 3, 5)}

        # LOO from cached coalitions: value at masks with exactly one claim missing.
        # Mask positions index the evidence list, whose order matches claim_shapley
        # (claim_results may have fewer rows when taxonomy dropped malformed claims).
        v_full = srow.get("v_full")
        position_to_claim = [c["claim_id"] for c in claim_shapley]
        loo_weights: dict[int, float] = {}
        for crow in coalitions.get(iid, []):
            mask = crow.get("mask") or []
            if (
                v_full is not None
                and mask
                and len(mask) == len(position_to_claim)
                and sum(mask) == len(mask) - 1
                and crow.get("value") is not None
            ):
                missing_pos = mask.index(0)
                loo_weights[position_to_claim[missing_pos]] = abs(v_full - crow["value"])
        loo_cov_num += len(loo_weights)
        loo_cov_den += len(position_to_claim)
        loo_leakage = None
        if len(loo_weights) == n and sum(loo_weights.values()) > 0:
            total = sum(loo_weights.values())
            loo_leakage = sum(w for cid, w in loo_weights.items() if cmap.get(cid, False)) / total

        per_instance.append({
            "instance_id": iid,
            "task": inst.get("task"),
            "n_claims": n,
            "olr": inst.get("olr"),
            "dclr": inst.get("shapley_dclr"),
            **topk,
            "loo_leakage": loo_leakage,
            "loo_covered": len(loo_weights),
        })

    olr = [r["olr"] for r in per_instance if r["olr"] is not None and r["dclr"] is not None]
    dclr = [r["dclr"] for r in per_instance if r["olr"] is not None and r["dclr"] is not None]
    corr = {}
    if len(olr) >= 3:
        pear = stats.pearsonr(olr, dclr)
        spear = stats.spearmanr(olr, dclr)
        corr = {"pearson_r": round(pear.statistic, 4), "spearman_rho": round(spear.statistic, 4)}

    # Disagreements: similar OLR (within 0.02) but DCLR differing by > 0.15
    pairs = []
    valid = [r for r in per_instance if r["olr"] is not None and r["dclr"] is not None and r["olr"] > 0]
    valid.sort(key=lambda r: r["olr"])
    for i in range(len(valid) - 1):
        for j in range(i + 1, min(i + 6, len(valid))):
            a, b = valid[i], valid[j]
            if abs(a["olr"] - b["olr"]) <= 0.02 and abs(a["dclr"] - b["dclr"]) > 0.15:
                pairs.append({
                    "instance_a": a["instance_id"], "instance_b": b["instance_id"],
                    "olr_a": round(a["olr"], 3), "olr_b": round(b["olr"], 3),
                    "dclr_a": round(a["dclr"], 3), "dclr_b": round(b["dclr"], 3),
                })

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    by_task = defaultdict(list)
    for r in per_instance:
        by_task[r["task"]].append(r)
    task_summary = {
        task: {
            "n": len(rows),
            "olr": mean([r["olr"] for r in rows]),
            "dclr": mean([r["dclr"] for r in rows]),
            "top1": mean([r["top1"] for r in rows]),
            "top3": mean([r["top3"] for r in rows]),
            "top5": mean([r["top5"] for r in rows]),
            "loo": mean([r["loo_leakage"] for r in rows]),
            "loo_complete_instances": sum(1 for r in rows if r["loo_leakage"] is not None),
        }
        for task, rows in sorted(by_task.items())
    }

    return {
        "n_instances": len(per_instance),
        "olr_dclr_correlation": corr,
        "identical_olr_divergent_dclr_pairs": pairs[:20],
        "n_divergent_pairs": len(pairs),
        "loo_cache_coverage": round(loo_cov_num / loo_cov_den, 4) if loo_cov_den else None,
        "task_summary": task_summary,
        "per_instance": per_instance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, help="Dir containing b1..b4 (or arm) subdirs")
    parser.add_argument("--methods", default="b1,b2,b3,b4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--supplement-dir", default=None,
                        help="Dir with loo_topup_<method>.jsonl files from cr_loo_topup.py")
    args = parser.parse_args()

    root = Path(args.results_root)
    report = {}
    for method in [m.strip() for m in args.methods.split(",")]:
        method_dir = root / method
        if not (method_dir / "instance_results.jsonl").exists():
            print(f"skip {method}: no instance_results")
            continue
        supplement = None
        if args.supplement_dir:
            candidate = Path(args.supplement_dir) / f"loo_topup_{method}.jsonl"
            supplement = candidate if candidate.exists() else None
        report[method] = analyze_method(method_dir, supplement)
        summary = {k: v for k, v in report[method].items() if k != "per_instance"}
        print(f"== {method}: n={summary['n_instances']} corr={summary['olr_dclr_correlation']} "
              f"loo_cov={summary['loo_cache_coverage']} divergent_pairs={summary['n_divergent_pairs']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
