"""Paper-faithful Leverage SHAP (Musco & Witter, ICLR 2025).

Implementation follows the reference code accompanying the paper:
https://github.com/rtealwitter/leverageshap

CoalitionSampler + weighted regression match ``leverageshap/estimators/sampling.py``
and ``leverageshap/estimators/leverage_shap.py`` (uniform per-size weights:
``sampling_weights = ones(n-1)``, paired complements, without-replacement sampling).

See also: https://arxiv.org/abs/2410.01917
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


def comb(n: int, k: int) -> float:
    return float(math.comb(n, k))


def comb_vec(n: int, sizes: np.ndarray) -> np.ndarray:
    return np.array([comb(n, int(s)) for s in sizes], dtype=float)


class CoalitionSampler:
    """Sample coalitions without replacement (reference ``CoalitionSampler``)."""

    def __init__(
        self,
        n_players: int,
        sampling_weights: np.ndarray,
        *,
        pairing_trick: bool = True,
        random_state: int | None = None,
        sample_with_replacement: bool = False,
    ) -> None:
        self._n_players = n_players

        sw = np.asarray(sampling_weights, dtype=float)
        if len(sw) == n_players + 1:
            sw = sw[1:-1]
        elif len(sw) == n_players:
            sw = sw[1:]
        elif len(sw) != n_players - 1:
            raise ValueError(
                f"sampling_weights must have length n_players-1={n_players - 1}, got {len(sw)}"
            )

        self._distribution = sw / np.min(sw)
        self._distribution = np.concatenate(([0.0], self._distribution, [0.0]))
        self._pairing_trick = pairing_trick
        self._rng = np.random.default_rng(seed=random_state)
        self._sample_with_replacement = sample_with_replacement
        self._constant = 0.0

    def _sampling_probs(self, sizes: np.ndarray) -> np.ndarray:
        cn = comb_vec(self._n_players, sizes)
        return np.minimum(self._constant * self._distribution[sizes] / cn, 1.0)

    def _get_sampling_probs(self, budget: int) -> None:
        n = self._n_players
        sizes = np.arange(1, n)
        comb_counts = comb_vec(n, sizes)
        weights = self._distribution[sizes].astype(float)
        target_total = float(np.clip(budget, 0.0, float(np.sum(comb_counts))))
        if target_total == 0.0:
            self._constant = 0.0
            return

        saturation_thresholds = comb_counts / weights
        order = np.argsort(saturation_thresholds)
        comb_counts_sorted = comb_counts[order]
        weights_sorted = weights[order]
        thresholds_sorted = saturation_thresholds[order]

        saturated_prefix = np.concatenate(([0.0], np.cumsum(comb_counts_sorted[:-1])))
        weights_prefix = np.concatenate(([0.0], np.cumsum(weights_sorted[:-1])))
        remaining_weight = np.sum(weights_sorted) - weights_prefix
        expected_at_threshold = saturated_prefix + thresholds_sorted * remaining_weight

        segment_idx = int(np.searchsorted(expected_at_threshold, target_total, side="left"))

        if segment_idx >= len(thresholds_sorted):
            scale = float(thresholds_sorted[-1])
        else:
            denom = remaining_weight[segment_idx]
            scale = (
                float(thresholds_sorted[segment_idx])
                if denom == 0
                else min(
                    (target_total - saturated_prefix[segment_idx]) / denom,
                    thresholds_sorted[segment_idx],
                )
            )
        self._constant = float(scale)

    def _symmetric_round_even(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.size
        tgt = int(np.round(x.sum() / 2) * 2)
        out = np.floor(x).astype(int)
        rem = tgt - out.sum()
        frac = x - np.floor(x)
        pairs = [(i, n - 1 - i, frac[i] + frac[n - 1 - i]) for i in range(n // 2)]
        pairs.sort(key=lambda t: t[2], reverse=True)
        for i, j, _ in pairs:
            if rem < 2:
                break
            out[i] += 1
            out[j] += 1
            rem -= 2
        if n % 2 == 1 and rem == 1:
            out[n // 2] += 1
        return out

    def _index_th_combination(self, pool: Sequence[T], size: int, index: int) -> Tuple[T, ...]:
        n = len(pool)
        k = size
        if not (0 <= k <= n):
            raise ValueError("size must be between 0 and len(pool)")
        total = math.comb(n, k)
        if not (0 <= index < total):
            raise IndexError(f"index must be in [0, {total - 1}] for C({n},{k})")
        combo: list[T] = []
        idx = index
        for i in range(n):
            if k == 0:
                break
            if n - i == k:
                combo.extend(pool[i : i + k])
                break
            c = math.comb(n - i - 1, k - 1)
            if idx < c:
                combo.append(pool[i])
                k -= 1
            else:
                idx -= c
        return tuple(combo)

    def _combination_generator(self, n: int, s: int, num_samples: int):
        num_combos = math.comb(n, s)
        try:
            assert not self._sample_with_replacement
            indices = self._rng.choice(num_combos, num_samples, replace=False)
            for i in indices:
                yield self._index_th_combination(range(n), s, int(i))
        except (OverflowError, AssertionError):
            for _ in range(num_samples):
                yield tuple(sorted(self._rng.choice(n, size=s, replace=False).tolist()))

    def _add_one_sample(self, indices: Sequence[int]) -> None:
        self._sampled_coalitions_matrix[self._coalition_idx, list(indices)] = True
        self._coalition_idx += 1

    def sample(self, budget: int) -> None:
        assert budget >= 2
        budget = min(budget, 2**self._n_players)
        budget += budget % 2

        self._get_sampling_probs(budget - 2)
        sizes = np.arange(1, self._n_players)
        cn_players = comb_vec(self._n_players, sizes)
        samples_per_size = self._symmetric_round_even(self._sampling_probs(sizes) * cn_players)
        sampling_probs = samples_per_size / cn_players

        self._sampled_coalitions_matrix = np.zeros((budget, self._n_players), dtype=bool)
        self._coalition_idx = 0

        self._add_one_sample([])
        self._add_one_sample(list(range(self._n_players)))

        for idx, size in enumerate(sizes):
            sz = int(size)
            if idx >= self._n_players // 2 and self._pairing_trick:
                break
            if self._pairing_trick and sz == self._n_players // 2 and self._n_players % 2 == 0:
                half = samples_per_size[idx] // 2
                combo_gen = self._combination_generator(self._n_players - 1, sz - 1, half)
                for indices in combo_gen:
                    ii = list(indices) + [self._n_players - 1]
                    self._add_one_sample(ii)
                    self._add_one_sample(list(set(range(self._n_players - 1)) - set(indices)))
            else:
                for indices in self._combination_generator(self._n_players, sz, samples_per_size[idx]):
                    self._add_one_sample(list(indices))
                    if self._pairing_trick:
                        self._add_one_sample(list(set(range(self._n_players)) - set(indices)))

        coalition_sizes = np.sum(self._sampled_coalitions_matrix, axis=1)
        self._sampled_coalitions_probability = np.ones(self._sampled_coalitions_matrix.shape[0])
        filt = (coalition_sizes > 0) & (coalition_sizes < self._n_players)
        self._sampled_coalitions_probability[filt] = sampling_probs[coalition_sizes[filt] - 1]


def paper_leverage_masks(
    n_claims: int,
    budget: int,
    *,
    seed: int = 42,
    pairing_trick: bool = True,
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Return coalition masks and per-row sampling probabilities (reference sampler).

    Uses uniform leverage weights over coalition sizes (paper reference code default).
    """

    if n_claims <= 0:
        return [tuple()], np.array([1.0])
    sampler = CoalitionSampler(
        n_claims,
        np.ones(n_claims - 1),
        pairing_trick=pairing_trick,
        random_state=seed,
        sample_with_replacement=False,
    )
    sampler.sample(budget)
    mat = sampler._sampled_coalitions_matrix
    probs = sampler._sampled_coalitions_probability
    masks = [tuple(int(x) for x in row) for row in mat]
    return masks, probs


def solve_paper_leverage_regression(
    masks: list[tuple[int, ...]],
    values: list[float],
    sampling_probs: np.ndarray,
    *,
    v0: float,
    v1: float,
) -> np.ndarray:
    """Solve Leverage SHAP regression (reference ``LeverageSHAP.shap_values``)."""

    if len(masks) != len(values):
        raise ValueError("masks and values length mismatch")
    n = len(masks[0])
    z = np.array(masks, dtype=float)
    y = np.array(values, dtype=float)
    coalition_sizes = np.sum(z, axis=1)

    filt = (coalition_sizes > 0) & (coalition_sizes < n)
    z = z[filt]
    y = y[filt]
    probs = np.asarray(sampling_probs, dtype=float)[filt]
    coalition_sizes = coalition_sizes[filt]

    if z.shape[0] == 0:
        return np.full(n, (v1 - v0) / n, dtype=float)

    y_adj = y - (v1 - v0) * coalition_sizes / n
    regression_weights = 1.0 / (
        np.array([comb(n, int(s)) for s in coalition_sizes])
        * coalition_sizes
        * (n - coalition_sizes)
    )
    kernel_weights = regression_weights / probs

    p_mat = np.eye(n) - (1.0 / n) * np.ones((n, n))
    w_diag = np.diag(kernel_weights)
    ata = p_mat @ z.T @ w_diag @ z @ p_mat
    atb = p_mat @ z.T @ w_diag @ y_adj

    sqrt_alpha = 1e-3
    if np.linalg.cond(ata) > 1.0 / np.finfo(ata.dtype).eps and z.shape[0] <= 3 * n:
        ata = ata + (sqrt_alpha**2) * np.eye(n)

    phi_perp, _, _, _ = np.linalg.lstsq(ata, atb, rcond=None)
    return phi_perp + (v1 - v0) / n
