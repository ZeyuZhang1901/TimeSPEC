#!/usr/bin/env python3
"""Compare exact Shapley vs paper-faithful Leverage SHAP (Musco & Witter ref. code).

Uses instance ``autocast_G2061`` (n=13) by default — same setting as prior diagnostics.

Without ``--cache``, performs 8192 coalition LLM calls once, saves ``--cache-path``,
then runs paper Leverage SHAP on a subset (budget ``--budget``, default 500) using
cached values only (no extra LLM).

**Budget vs paper:** Musco & Witter (arXiv:2410.01917) use ``m = O(n\\log n + ...)``
samples; production uses ``m \\approx \\lfloor c\\,n\\ln n\\rfloor`` (see
``compute_budget``). For ``n=8`` that is ``249`` rows; note ``n \\le exact_threshold_n``
(8) uses **exact** ``2^n`` in the live evaluator, not Leverage SHAP.

Example (``n=8``, high leakage in B4, paper-style ``m=249``)::

    uv run python LeakageEvaluationFramework/scripts/compare_paper_vs_exact_shapley.py \\
        --instance-id fb_2025-11-09_0x1ca2465222c60f \\
        --completions runtime/benchmark/qwen35_35b/b4/completions.json \\
        --cache-path runtime/benchmark/shapley_cache_fb_n8_highleak.json \\
        --budget 249 --concurrency 16

Example (``n\\approx 7``, **eight** leaked claim IDs in generation metadata)::

    uv run python ... --instance-id autocast_M9744 \\
        --completions runtime/benchmark/qwen35_35b/b4/completions.json \\
        --budget 180
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
from LeakageEvaluationFramework.leakage_eval.metrics.dclr import shapley_dclr
from LeakageEvaluationFramework.leakage_eval.shapley.coalition_sampler import (
    compute_budget,
    exact_coalitions,
)
from LeakageEvaluationFramework.leakage_eval.shapley.exact import exact_shapley
from LeakageEvaluationFramework.leakage_eval.shapley.paper_leverage_shap import (
    paper_leverage_masks,
    solve_paper_leverage_regression,
)
from LeakageEvaluationFramework.leakage_eval.shapley.value_function import evaluate_coalition


def _mask_key(m: tuple[int, ...]) -> str:
    return "".join(str(b) for b in m)


def _load_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {k: float(v) for k, v in data.items()}


def _save_cache(path: Path, cache: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=0))


async def _fill_cache(
    instance,
    masks: list[tuple[int, ...]],
    predictor,
    cache: dict[str, float],
    concurrency: int,
) -> dict[str, float]:
    missing = [m for m in masks if _mask_key(m) not in cache]
    if not missing:
        return cache

    sem = asyncio.Semaphore(concurrency)

    async def one(mask: tuple[int, ...]) -> tuple[str, float]:
        async with sem:
            v, _ = await evaluate_coalition(instance, mask, predictor, None)
            return _mask_key(mask), float(v)

    pairs = await asyncio.gather(*[one(m) for m in missing])
    for k, v in pairs:
        cache[k] = v
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", default="autocast_G2061")
    parser.add_argument("--completions", type=Path, default=Path("runtime/benchmark/qwen35_35b/b1/completions.json"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset/benchmark/final/binary.jsonl"))
    parser.add_argument("--cache-path", type=Path, default=Path("runtime/benchmark/shapley_cache_autocast_G2061.json"))
    parser.add_argument("--budget", type=int, default=500, help="Paper Leverage SHAP coalition budget (even).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--skip-llm", action="store_true", help="Only compare solvers using existing cache (must exist).")
    parser.add_argument(
        "--synthetic-additive",
        action="store_true",
        help="No LLM: v(S)=sum of random weights (sanity check for paper sampler + regression).",
    )
    parser.add_argument("--synthetic-n", type=int, default=13)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.synthetic_additive:
        rng = np.random.default_rng(args.seed)
        n = args.synthetic_n
        weights = rng.uniform(0.1, 1.0, size=n)
        masks_all = exact_coalitions(n)
        cache = {_mask_key(m): float(sum(weights[i] for i in range(n) if m[i])) for m in masks_all}
        values_all = [cache[_mask_key(m)] for m in masks_all]
        phi_exact = exact_shapley(masks_all, values_all)
        full_mask = tuple([1] * n)
        v_full = cache[_mask_key(full_mask)]
        masks_paper, probs = paper_leverage_masks(n, args.budget, seed=args.seed)
        values_paper = [cache[_mask_key(m)] for m in masks_paper]
        phi_paper = solve_paper_leverage_regression(
            masks_paper, values_paper, probs, v0=0.0, v1=float(v_full)
        )
        diff = phi_exact - phi_paper
        rel_l2_sq = float(np.dot(diff, diff) / max(np.dot(phi_exact, phi_exact), 1e-12))
        print("\n=== Synthetic additive game (no LLM) ===")
        print(f"n={n}, budget={args.budget}, relative L2 squared error={rel_l2_sq:.6e}")
        print(f"max_abs_err={np.max(np.abs(diff)):.6e}")
        print(f"weights recovered max_err vs exact={np.max(np.abs(phi_exact - weights)):.6e}")
        return

    completions = load_completion_rows(args.completions)
    meta = load_dataset_metadata(args.dataset)
    rows = [r for r in completions if r["instance_id"] == args.instance_id]
    if not rows:
        raise SystemExit(f"instance_id {args.instance_id} not found in {args.completions}")

    instance = build_eval_inputs(rows, meta, model="qwen/qwen3.5-35b-a3b", method="b1")[0]
    n = len(instance.evidence)
    logging.info("Instance %s: n_claims=%d", args.instance_id, n)

    masks_all = exact_coalitions(n)
    full_mask = tuple([1] * n)
    cfg = ShapleyConfig()

    budget_exact = compute_budget(n, c=cfg.coalition_budget_c, cap=cfg.coalition_budget_cap, exact_threshold_n=8)
    logging.info("Production approximate budget (exact_threshold_n=8): %d", budget_exact)

    cache = _load_cache(args.cache_path)
    predictor = OpenRouterCoalitionPredictor(
        model="qwen/qwen3.5-35b-a3b",
        temperature=0.0,
        max_tokens=128,
    )

    if args.skip_llm:
        if len(cache) < len(masks_all):
            raise SystemExit(f"Cache incomplete: {len(cache)} / {len(masks_all)} masks in {args.cache_path}")
    else:
        t0 = time.time()
        cache = asyncio.run(
            _fill_cache(instance, masks_all, predictor, cache, args.concurrency)
        )
        _save_cache(args.cache_path, cache)
        logging.info("Filled cache: %d masks in %.1fs → %s", len(cache), time.time() - t0, args.cache_path)

    values_all = [cache[_mask_key(m)] for m in masks_all]
    phi_exact = exact_shapley(masks_all, values_all)
    v_full = cache[_mask_key(full_mask)]

    masks_paper, probs = paper_leverage_masks(n, args.budget, seed=args.seed)
    values_paper = [cache[_mask_key(m)] for m in masks_paper]
    phi_paper = solve_paper_leverage_regression(
        masks_paper, values_paper, probs, v0=0.0, v1=float(v_full)
    )

    diff = phi_exact - phi_paper
    rel_l2_sq = float(np.dot(diff, diff) / max(np.dot(phi_exact, phi_exact), 1e-12))
    max_abs = float(np.max(np.abs(diff)))

    labels_true = [True] + [False] * (n - 1)
    d_exact = shapley_dclr(phi_exact.tolist(), labels_true)["shapley_dclr"]
    d_paper = shapley_dclr(phi_paper.tolist(), labels_true)["shapley_dclr"]

    print("\n=== Paper Leverage SHAP vs exact (same cached v(S)) ===")
    print(f"n={n}, paper_budget={args.budget}, paper_rows={len(masks_paper)}, exact_rows={len(masks_all)}")
    print(f"relative ||phi||_2^2 error (paper vs exact): {rel_l2_sq:.6f}")
    print(f"max_j |phi_j_exact - phi_j_paper|:          {max_abs:.6f}")
    print(f"DCLR (oracle leak only claim 1 — diagnostic): exact={d_exact:.6f}, paper={d_paper:.6f}")
    print(f"sum(phi_exact)={phi_exact.sum():.6f}, sum(phi_paper)={phi_paper.sum():.6f}, v_full={v_full:.6f}")

    rho = np.corrcoef(np.abs(phi_exact), np.abs(phi_paper))[0, 1]
    print(f"corr(|phi_exact|, |phi_paper|): {rho:.4f}")


if __name__ == "__main__":
    main()
