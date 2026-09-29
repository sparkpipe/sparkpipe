#!/usr/bin/env python3
"""Statistics for quantization A/B comparisons (lanes/quant-ab-design.md §3).

Everything here is deterministic: the document bootstrap draws its resample
indices from a SplitMix64 stream keyed by (seed, replicate, draw), so the same
seed gives the same intervals on every host and numpy version.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
SPLITMIX_M1 = np.uint64(0xBF58476D1CE4E5B9)
SPLITMIX_M2 = np.uint64(0x94D049BB133111EB)
NORMAL = NormalDist()


def splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = values.astype(np.uint64) + SPLITMIX_GAMMA
        z = (z ^ (z >> np.uint64(30))) * SPLITMIX_M1
        z = (z ^ (z >> np.uint64(27))) * SPLITMIX_M2
        return z ^ (z >> np.uint64(31))


def bootstrap_indices(seed: int, replicates: int, units: int) -> np.ndarray:
    if units <= 0 or replicates <= 0:
        raise ValueError("bootstrap needs at least one unit and one replicate")
    counter = np.arange(replicates * units, dtype=np.uint64)
    with np.errstate(over="ignore"):
        keyed = counter + np.uint64(seed & 0xFFFFFFFFFFFFFFFF) * np.uint64(0x2545F4914F6CDD1D)
    draws = splitmix64(keyed)
    return (draws % np.uint64(units)).astype(np.int64).reshape(replicates, units)


def resample_statistic(statistic, columns, indices: np.ndarray) -> np.ndarray:
    return np.asarray(statistic(*[column[indices] for column in columns]), dtype=np.float64)


def jackknife(statistic, columns) -> np.ndarray:
    n = len(columns[0])
    keep = np.ones(n, dtype=bool)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        keep[i] = False
        out[i] = float(statistic(*[column[keep][None, :] for column in columns])[0])
        keep[i] = True
    return out


def percentile_bound(replicates: np.ndarray, level: float, side: str) -> float:
    q = level if side == "upper" else 1.0 - level
    return float(np.quantile(replicates, q, method="linear"))


def bca_bound(estimate: float, replicates: np.ndarray, jack: np.ndarray, level: float, side: str) -> float:
    below = float(np.mean(replicates < estimate)) + 0.5 * float(np.mean(replicates == estimate))
    below = min(max(below, 1.0 / (len(replicates) + 1)), 1.0 - 1.0 / (len(replicates) + 1))
    z0 = NORMAL.inv_cdf(below)
    centered = jack.mean() - jack
    denominator = 6.0 * float(np.sum(centered ** 2)) ** 1.5
    acceleration = float(np.sum(centered ** 3)) / denominator if denominator > 0 else 0.0
    alpha = level if side == "upper" else 1.0 - level
    z = NORMAL.inv_cdf(alpha)
    adjusted = NORMAL.cdf(z0 + (z0 + z) / (1.0 - acceleration * (z0 + z)))
    return float(np.quantile(replicates, adjusted, method="linear"))


def bounds(statistic, columns, seed: int, replicates: int, level: float) -> dict:
    columns = [np.asarray(column, dtype=np.float64) for column in columns]
    estimate = float(statistic(*[column[None, :] for column in columns])[0])
    indices = bootstrap_indices(seed, replicates, len(columns[0]))
    draws = resample_statistic(statistic, columns, indices)
    jack = jackknife(statistic, columns)
    out = {"estimate": estimate, "level": level, "replicates": replicates, "seed": seed}
    for side in ("lower", "upper"):
        p = percentile_bound(draws, level, side)
        b = bca_bound(estimate, draws, jack, level, side)
        out[f"{side}_percentile"] = p
        out[f"{side}_bca"] = b
        out[side] = min(p, b) if side == "lower" else max(p, b)
    out["draws"] = draws
    return out


def mean_statistic(values):
    return values.mean(axis=1)


def ratio_statistic(numerator, denominator):
    return numerator.mean(axis=1) / denominator.mean(axis=1)


def difference_statistic(left, right):
    return left.mean(axis=1) - right.mean(axis=1)


def interaction_statistic(a, b, c, d):
    return a.mean(axis=1) - b.mean(axis=1) - c.mean(axis=1) + d.mean(axis=1)


def bootstrap_p_outside(draws: np.ndarray, margin: float, direction: str) -> float:
    if direction == "upper":
        outside = int(np.sum(draws >= margin))
    else:
        outside = int(np.sum(draws <= margin))
    return (outside + 1) / (len(draws) + 1)


def holm_levels(p_values: list, alpha: float) -> list:
    m = len(p_values)
    order = sorted(range(m), key=lambda index: (p_values[index], index))
    levels = [0.0] * m
    for rank, index in enumerate(order):
        levels[index] = alpha / (m - rank)
    return order, levels


def mcnemar_exact(pass_to_fail: int, fail_to_pass: int) -> float:
    n = pass_to_fail + fail_to_pass
    if n == 0:
        return 1.0
    k = min(pass_to_fail, fail_to_pass)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def wilson(successes: int, total: int, z: float) -> tuple:
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return ((centre - spread) / denominator, (centre + spread) / denominator)


def newcombe_paired(both: int, first_only: int, second_only: int, neither: int, level: float = 0.95) -> tuple:
    n = both + first_only + second_only + neither
    if n == 0:
        raise ValueError("newcombe needs at least one pair")
    z = NORMAL.inv_cdf(1 - (1 - level) / 2)
    p1 = (both + first_only) / n
    p2 = (both + second_only) / n
    l1, u1 = wilson(both + first_only, n, z)
    l2, u2 = wilson(both + second_only, n, z)
    difference = p1 - p2
    a, b, c, d = both, first_only, second_only, neither
    product = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if product == 0 else (a * d - b * c) / math.sqrt(product)
    lower = difference - math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    upper = difference + math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return (difference, lower, upper)


def kaplan_meier_median(events: list) -> dict:
    ordered = sorted(events, key=lambda item: (item[0], not item[1]))
    at_risk = len(ordered)
    survival = 1.0
    median = None
    index = 0
    while index < len(ordered):
        time = ordered[index][0]
        deaths = 0
        leaving = 0
        while index < len(ordered) and ordered[index][0] == time:
            deaths += 1 if ordered[index][1] else 0
            leaving += 1
            index += 1
        if deaths:
            survival *= 1 - deaths / at_risk
            if median is None and survival <= 0.5:
                median = time
        at_risk -= leaving
    never = sum(1 for _, event in events if not event)
    return {"median": median, "never_fraction": never / len(events) if events else 0.0, "cases": len(events)}


def log_softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    top = logits.max(axis=-1, keepdims=True)
    return logits - (top + np.log(np.exp(logits - top).sum(axis=-1, keepdims=True)))


def exact_kl_rows(reference_logits: np.ndarray, arm_logits: np.ndarray) -> np.ndarray:
    lp = log_softmax_rows(reference_logits)
    lq = log_softmax_rows(arm_logits)
    return np.sum(np.exp(lp) * (lp - lq), axis=-1)


def bucket_kl_rows(reference_probe_logits, reference_lse, arm_probe_logits, arm_lse, tolerance: float = 1e-6) -> dict:
    lp = np.asarray(reference_probe_logits, dtype=np.float64) - np.asarray(reference_lse, dtype=np.float64)[:, None]
    lq = np.asarray(arm_probe_logits, dtype=np.float64) - np.asarray(arm_lse, dtype=np.float64)[:, None]
    head = np.sum(np.exp(lp) * (lp - lq), axis=1)
    p_log_mass = np.logaddexp.reduce(lp, axis=1)
    q_log_mass = np.logaddexp.reduce(lq, axis=1)
    if np.any(p_log_mass > tolerance) or np.any(q_log_mass > tolerance):
        raise ValueError("probe probabilities sum above 1 beyond tolerance: the dump or its LSE is inconsistent")
    p_zero = p_log_mass >= 0
    q_zero = q_log_mass >= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        p_tail = np.where(p_zero, 0.0, -np.expm1(np.minimum(p_log_mass, 0.0)))
        q_tail = np.where(q_zero, 0.0, -np.expm1(np.minimum(q_log_mass, 0.0)))
        tail = np.where(p_tail > 0, p_tail * (np.log(p_tail) - np.log(q_tail)), 0.0)
    infinite = int(np.sum((p_tail > 0) & (q_tail == 0)))
    return {"kl": head + tail, "p_tail": p_tail, "q_tail": q_tail,
            "tail_zero_rows": int(np.sum(p_zero)), "infinite_rows": infinite}
