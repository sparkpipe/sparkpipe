#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import ab_dump
import ab_stats


def probe_rows(reference, arm, k):
    ids, _ = ab_dump.top_k(reference.astype(np.float32), k)
    ref32 = reference.astype(np.float32)
    arm32 = arm.astype(np.float32)
    lse = lambda x: np.log(np.exp(x.astype(np.float64) - x.max(1, keepdims=True)).sum(1)) + x.max(1).astype(np.float64)
    return (np.take_along_axis(ref32, ids.astype(np.int64), 1), lse(ref32),
            np.take_along_axis(arm32, ids.astype(np.int64), 1), lse(arm32))


class KlTest(unittest.TestCase):
    def test_identical_is_exactly_zero(self):
        rng = np.random.default_rng(1)
        logits = rng.normal(0, 3, (200, 1000))
        a, la, b, lb = probe_rows(logits, logits.copy(), 64)
        kl = ab_stats.bucket_kl_rows(a, la, b, lb)["kl"]
        self.assertTrue(np.all(kl == 0.0))
        self.assertEqual(float(np.sum(kl)), 0.0)
        exact = ab_stats.exact_kl_rows(logits, logits.copy())
        self.assertTrue(np.all(exact == 0.0))

    def test_bucket_kl_is_a_lower_bound_of_exact_kl(self):
        rng = np.random.default_rng(2)
        for scale, noise, k in ((1.0, 0.1, 64), (3.0, 0.5, 8), (6.0, 1.0, 64), (0.3, 0.05, 1)):
            reference = rng.normal(0, scale, (300, 800))
            arm = reference + rng.normal(0, noise, reference.shape)
            a, la, b, lb = probe_rows(reference, arm, k)
            bucket = ab_stats.bucket_kl_rows(a, la, b, lb)["kl"]
            exact = ab_stats.exact_kl_rows(reference.astype(np.float32), arm.astype(np.float32))
            self.assertTrue(np.all(bucket <= exact + 1e-9 * np.maximum(1, exact)), (scale, noise, k))
            self.assertTrue(np.all(bucket >= -1e-12))
            self.assertGreater(float(bucket.mean()), 0.0)

    def test_bound_tightens_when_the_tail_is_small(self):
        rng = np.random.default_rng(3)
        reference = rng.normal(0, 1, (100, 500))
        reference[:, :4] += 16
        arm = reference + rng.normal(0, 0.2, reference.shape)
        a, la, b, lb = probe_rows(reference, arm, 64)
        result = ab_stats.bucket_kl_rows(a, la, b, lb)
        exact = ab_stats.exact_kl_rows(reference.astype(np.float32), arm.astype(np.float32))
        self.assertLess(float(np.quantile(result["p_tail"], 0.99)), 1e-3)
        self.assertGreater(float(result["kl"].mean() / exact.mean()), 0.9)

    def test_inconsistent_probe_mass_is_refused(self):
        a = np.array([[0.0, 0.0]])
        with self.assertRaises(ValueError):
            ab_stats.bucket_kl_rows(a, np.array([0.0]), a, np.array([0.0]))


class TestsTest(unittest.TestCase):
    def test_mcnemar_known_answers(self):
        self.assertEqual(ab_stats.mcnemar_exact(6, 0), 0.03125)
        self.assertEqual(ab_stats.mcnemar_exact(0, 6), 0.03125)
        self.assertEqual(ab_stats.mcnemar_exact(5, 0), 0.0625)
        self.assertEqual(ab_stats.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(ab_stats.mcnemar_exact(7, 1), 0.0703125)

    def test_newcombe_interval(self):
        difference, lower, upper = ab_stats.newcombe_paired(40, 6, 0, 46)
        self.assertAlmostEqual(difference, 6 / 92)
        self.assertTrue(lower < difference < upper)
        self.assertGreater(lower, 0.0)
        difference, lower, upper = ab_stats.newcombe_paired(14, 0, 0, 3)
        self.assertEqual(difference, 0.0)
        self.assertAlmostEqual(lower, -upper)
        self.assertTrue(-1 <= lower <= 0 <= upper <= 1)

    def test_holm_levels(self):
        order, levels = ab_stats.holm_levels([0.04, 0.001, 0.02], 0.05)
        self.assertEqual(order, [1, 2, 0])
        self.assertEqual(levels, [0.05, 0.05 / 3, 0.025])

    def test_kaplan_meier(self):
        km = ab_stats.kaplan_meier_median([(3, True), (9, False), (9, False), (9, False)])
        self.assertEqual(km["median"], None)
        self.assertEqual(km["never_fraction"], 0.75)
        km = ab_stats.kaplan_meier_median([(3, True), (5, True), (7, True), (9, False)])
        self.assertEqual(km["median"], 5)


class BootstrapTest(unittest.TestCase):
    def test_reproducible_by_seed(self):
        first = ab_stats.bootstrap_indices(20260929, 1000, 290)
        second = ab_stats.bootstrap_indices(20260929, 1000, 290)
        other = ab_stats.bootstrap_indices(20260930, 1000, 290)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, other))
        self.assertEqual(first.shape, (1000, 290))
        self.assertTrue(first.min() >= 0 and first.max() < 290)
        self.assertEqual(int(first[0, 0]), int(ab_stats.bootstrap_indices(20260929, 1, 290)[0, 0]))
        values = np.random.default_rng(5).gamma(2.0, 0.001, 290)
        a = ab_stats.bounds(ab_stats.mean_statistic, [values], 20260929, 2000, 0.975)
        b = ab_stats.bounds(ab_stats.mean_statistic, [values], 20260929, 2000, 0.975)
        for key in ("lower", "upper", "lower_bca", "upper_percentile"):
            self.assertEqual(a[key], b[key])

    def test_indices_are_uniform(self):
        draws = ab_stats.bootstrap_indices(7, 2000, 10).ravel()
        counts = np.bincount(draws, minlength=10)
        self.assertLess(np.abs(counts / draws.size - 0.1).max(), 0.01)

    def test_interval_covers_the_mean(self):
        rng = np.random.default_rng(11)
        covered = 0
        for trial in range(60):
            values = rng.normal(1.0, 0.5, 120)
            result = ab_stats.bounds(ab_stats.mean_statistic, [values], trial, 1000, 0.975)
            covered += result["lower"] <= 1.0 <= result["upper"]
            self.assertLessEqual(result["lower"], min(result["lower_percentile"], result["lower_bca"]))
        self.assertGreaterEqual(covered, 52)

    def test_ratio_and_paired_statistics(self):
        rng = np.random.default_rng(12)
        anchor = rng.gamma(4.0, 0.001, 200)
        arm = anchor * 1.5
        result = ab_stats.bounds(ab_stats.ratio_statistic, [arm, anchor], 1, 1000, 0.95)
        self.assertAlmostEqual(result["estimate"], 1.5)
        self.assertAlmostEqual(result["lower"], 1.5)
        self.assertAlmostEqual(result["upper"], 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=1)
