"""Coalition sampling for exact Shapley and Leverage SHAP."""

from __future__ import annotations

import itertools
import math

import numpy as np


def compute_budget(n_claims: int, *, c: float, cap: int, exact_threshold_n: int) -> int:
    """Compute total coalition budget (rows), including empty and full coalitions.

    When ``n_claims > exact_threshold_n`` (default ``8``) and ``2 ** n_claims`` exceeds this budget,
    the runner uses **Leverage SHAP** (Musco & Witter, ICLR 2025). Their Theorem 1.1
    gives sample complexity ``m = O(n log(n/delta) + n/(epsilon delta))``; we implement
    the dominant **``O(n log n)``** scaling as ``floor(c * n * ln(n))`` (clipped),
    matching the reference implementation's budget knob ``num_samples``.
    """

    if n_claims < 0:
        raise ValueError("n_claims must be non-negative")
    if n_claims <= exact_threshold_n:
        return 2 ** n_claims
    budget = int(math.floor(c * n_claims * math.log(n_claims)))
    return min(cap, max(2 * n_claims + 2, budget))


def exact_coalitions(n_claims: int) -> list[tuple[int, ...]]:
    """Enumerate all coalition masks."""

    if n_claims < 0:
        raise ValueError("n_claims must be non-negative")
    return [tuple(int(bit) for bit in bits) for bits in itertools.product([0, 1], repeat=n_claims)]


def leverage_shap_coalitions(n_claims: int, budget: int, *, seed: int) -> list[tuple[int, ...]]:
    """Sample paired Leverage SHAP coalition masks.

    This engineering sampler follows the plan: sample coalition size uniformly,
    sample a subset uniformly within that size, and include its complement.
    """

    if n_claims <= 0:
        return [tuple()]
    max_coalitions = 2 ** n_claims
    if budget >= max_coalitions:
        return exact_coalitions(n_claims)

    rng = np.random.default_rng(seed)
    masks: dict[tuple[int, ...], None] = {
        tuple([0] * n_claims): None,
        tuple([1] * n_claims): None,
    }

    while len(masks) < budget:
        size = int(rng.integers(1, n_claims))
        chosen = set(rng.choice(n_claims, size=size, replace=False).tolist())
        mask = tuple(1 if i in chosen else 0 for i in range(n_claims))
        comp = tuple(1 - bit for bit in mask)
        masks.setdefault(mask, None)
        if len(masks) < budget:
            masks.setdefault(comp, None)
        if len(masks) == max_coalitions:
            break

    return list(masks.keys())


def coalition_indices(mask: tuple[int, ...]) -> list[int]:
    """Return zero-based indices included in a coalition mask."""

    return [idx for idx, bit in enumerate(mask) if bit]

