#!/usr/bin/env python3
"""Five-instance Shapley pilot with ``exact_threshold_n=8`` vs full enumeration.

Default cohort: one instance with **fewer than eight IDs** in ``leaked_claim_ids``
(and ``n \\le 8``, so the **exact** Shapley branch runs), plus four instances with
**more than eight claims** so Leverage SHAP runs (unless capped). Rows must have
non-empty ``leaked_claim_ids`` unless ``--no-require-leakage``.

There are not four B4 rows that simultaneously have both nine-plus leaked IDs in
metadata and nine-plus claims (heavy leakage ids typically attach to small ``n``);
use ``--instance-ids`` / ``--completions`` for alternate cohorts.

Reports:

- ``m_pipeline``: coalition evaluations in production configuration
- ``t_pipeline_s``: wall time for those evaluations only
- ``rel_l2_sq``: ``||phi_pipe - phi_exact||_2^2 / max(||phi_exact||_2^2, eps)``
- ``max_abs``: ``max_j |phi_pipe_j - phi_exact_j|``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from LeakageEvaluationFramework.leakage_eval.config import ShapleyConfig
from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import (
    OpenRouterCoalitionPredictor,
)
from LeakageEvaluationFramework.leakage_eval.io.loaders import (
    build_eval_inputs,
    load_completion_rows,
    load_dataset_metadata,
)
from LeakageEvaluationFramework.leakage_eval.shapley.coalition_sampler import compute_budget, exact_coalitions
from LeakageEvaluationFramework.leakage_eval.shapley.exact import exact_shapley
from LeakageEvaluationFramework.leakage_eval.shapley.paper_leverage_shap import (
    paper_leverage_masks,
    solve_paper_leverage_regression,
)
from LeakageEvaluationFramework.scripts.compare_paper_vs_exact_shapley import (
    _fill_cache,
    _mask_key,
)

DEFAULT_INSTANCE_IDS = (
    "fb_2025-11-09_0x1ca2465222c60f",
    "autocast_G447",
    "autocast_M2936",
    "autocast_M4565",
    "autocast_G985",
)


def _corr_abs(phi_exact: np.ndarray, phi_pipe: np.ndarray) -> float:
    a = np.abs(phi_exact)
    b = np.abs(phi_pipe)
    if a.size < 2 or np.std(a) < 1e-14 or np.std(b) < 1e-14:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _pipeline_masks_and_exact_flag(
    n: int, cfg: ShapleyConfig
) -> tuple[list[tuple[int, ...]], np.ndarray | None, bool]:
    budget = compute_budget(
        n,
        c=cfg.coalition_budget_c,
        cap=cfg.coalition_budget_cap,
        exact_threshold_n=cfg.exact_threshold_n,
    )
    if budget >= 2**n:
        masks = exact_coalitions(n)
        return masks, None, True
    masks, probs = paper_leverage_masks(n, budget, seed=cfg.random_seed, pairing_trick=True)
    return masks, probs, False


async def _run_one(
    instance_id: str,
    *,
    rows_by_id: dict[str, dict],
    meta: dict,
    model: str,
    method: str,
    cfg: ShapleyConfig,
    predictor: OpenRouterCoalitionPredictor,
    concurrency: int,
) -> dict[str, object]:
    row = rows_by_id[instance_id]
    instance = build_eval_inputs([row], meta, model=model, method=method)[0]
    n = len(instance.evidence)
    masks_all = exact_coalitions(n)
    full_mask = tuple([1] * n)

    pipe_masks, samp_probs, is_exact = _pipeline_masks_and_exact_flag(n, cfg)
    cache: dict[str, float] = {}

    t0 = time.perf_counter()
    cache = await _fill_cache(instance, pipe_masks, predictor, cache, concurrency)
    t_pipeline = time.perf_counter() - t0

    missing_gt = [m for m in masks_all if _mask_key(m) not in cache]
    if missing_gt:
        t1 = time.perf_counter()
        cache = await _fill_cache(instance, missing_gt, predictor, cache, concurrency)
        t_gt_extra = time.perf_counter() - t1
    else:
        t_gt_extra = 0.0

    values_all = [cache[_mask_key(m)] for m in masks_all]
    phi_exact = exact_shapley(masks_all, values_all)
    v_full = float(cache[_mask_key(full_mask)])

    if is_exact:
        phi_pipe = phi_exact
        rel_l2_sq = 0.0
        max_abs = 0.0
        rho = 1.0
    else:
        assert samp_probs is not None
        vals_pipe = [cache[_mask_key(m)] for m in pipe_masks]
        phi_pipe = solve_paper_leverage_regression(
            pipe_masks, vals_pipe, samp_probs, v0=0.0, v1=v_full
        )
        diff = phi_exact - phi_pipe
        rel_l2_sq = float(np.dot(diff, diff) / max(np.dot(phi_exact, phi_exact), 1e-12))
        max_abs = float(np.max(np.abs(diff)))
        rho = _corr_abs(phi_exact, phi_pipe)

    leak_ids = row.get("leaked_claim_ids") or []
    leak_n = len(leak_ids) if isinstance(leak_ids, list) else -1

    return {
        "instance_id": instance_id,
        "n_claims": n,
        "len_leaked_claim_ids": leak_n,
        "branch": "exact" if is_exact else "paper_leverage_shap",
        "m_pipeline": len(pipe_masks),
        "m_full": len(masks_all),
        "t_pipeline_s": round(t_pipeline, 3),
        "t_gt_extra_s": round(t_gt_extra, 3),
        "rel_l2_sq": rel_l2_sq,
        "max_abs_phi": max_abs,
        "corr_abs_phi": rho,
        "v_full": v_full,
        "sum_phi_exact": float(phi_exact.sum()),
        "sum_phi_pipe": float(phi_pipe.sum()),
    }


async def _async_main(
    *,
    completions_path: Path,
    dataset_path: Path,
    instance_ids: tuple[str, ...],
    model: str,
    method: str,
    exact_threshold_n: int,
    seed: int,
    concurrency: int,
    max_tokens: int,
    require_leakage: bool,
) -> list[dict[str, object]]:
    completions = load_completion_rows(completions_path)
    meta = load_dataset_metadata(dataset_path)
    rows_by_id = {str(r["instance_id"]): r for r in completions}

    cfg = ShapleyConfig(exact_threshold_n=exact_threshold_n, random_seed=seed)
    predictor = OpenRouterCoalitionPredictor(
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    out: list[dict[str, object]] = []
    for instance_id in instance_ids:
        logging.info("Pilot instance %s", instance_id)
        row = rows_by_id.get(instance_id)
        if row is None:
            raise KeyError(f"instance_id not in completions: {instance_id}")
        leak_ids = row.get("leaked_claim_ids") or []
        if require_leakage and not leak_ids:
            logging.warning("Skipping %s: empty leaked_claim_ids", instance_id)
            continue
        out.append(
            await _run_one(
                instance_id,
                rows_by_id=rows_by_id,
                meta=meta,
                model=model,
                method=method,
                cfg=cfg,
                predictor=predictor,
                concurrency=concurrency,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions", type=Path, default=Path("runtime/benchmark/qwen35_35b/b4/completions.json"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset/benchmark/final/binary.jsonl"))
    parser.add_argument(
        "--instance-ids",
        default=",".join(DEFAULT_INSTANCE_IDS),
        help="Comma-separated instance ids (default: five-instance cohort).",
    )
    parser.add_argument("--model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--method", default="b4")
    parser.add_argument("--exact-threshold-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--require-leakage", action="store_true", default=True)
    parser.add_argument("--no-require-leakage", action="store_false", dest="require_leakage")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    instance_ids = tuple(i.strip() for i in args.instance_ids.split(",") if i.strip())

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = asyncio.run(
        _async_main(
            completions_path=args.completions,
            dataset_path=args.dataset,
            instance_ids=instance_ids,
            model=args.model,
            method=args.method,
            exact_threshold_n=args.exact_threshold_n,
            seed=args.seed,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            require_leakage=args.require_leakage,
        )
    )

    hdr = (
        f"{'instance_id':<26} {'n':>2} {'leaks':>5} {'branch':<20} "
        f"{'m_pipe':>6} {'m_full':>6} {'t_pipe':>8} {'rel_L2sq':>10} {'max|dphi|':>10} {'corr':>6}"
    )
    print("\n=== Pilot: pipeline Shapley vs full enumeration (cached v(S)) ===")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{str(r['instance_id']):<26} {int(r['n_claims']):>2} {int(r['len_leaked_claim_ids']):>5} "
            f"{str(r['branch']):<20} {int(r['m_pipeline']):>6} {int(r['m_full']):>6} "
            f"{float(r['t_pipeline_s']):>8.3f} {float(r['rel_l2_sq']):>10.6f} "
            f"{float(r['max_abs_phi']):>10.6f} {float(r['corr_abs_phi']):>6.3f}"
        )

    total_pipe_t = sum(float(r["t_pipeline_s"]) for r in rows)
    print("-" * len(hdr))
    print(f"Sum t_pipeline_s (production coalition phase): {total_pipe_t:.3f}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2))
        logging.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()