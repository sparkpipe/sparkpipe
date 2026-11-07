#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import hy4_reference
import ling3_reference
import reference_device_error as device


def bf16(values):
    return ling3_reference.bf16(values)


def chained(left, right):
    products = left[:, None, :] * right.T[None, :, :]
    total = np.zeros(products.shape[:2], dtype=np.float32)
    for first in range(0, products.shape[2], device.SUM_STEP):
        step = products[:, :, first:first + device.SUM_STEP]
        while step.shape[2] > 1:
            step = (step[:, :, 0::2] + step[:, :, 1::2]).astype(np.float32)
        total = (total + step[:, :, 0]).astype(np.float32)
    return total


def operands(rows, depth, outputs, seed):
    rng = np.random.default_rng(seed)
    left = bf16(rng.standard_normal((rows, depth)).astype(np.float32))
    right = bf16((0.02 * rng.standard_normal((depth, outputs))).astype(np.float32))
    return left, right


class AccumulationTest(unittest.TestCase):
    def tearDown(self):
        device.select("a")
        ling3_reference.FP32 = False
        ling3_reference.REVERSED = False

    def product(self, label, left, right):
        ling3_reference.FP32 = True
        ling3_reference.REVERSED = label != "a"
        device.select(label)
        return ling3_reference.matmul(left, right)

    def test_a_and_b_keep_their_products(self):
        left, right = operands(48, 2500, 40, 1)
        self.assertTrue(np.array_equal(self.product("a", left, right), left @ right))
        expected = None
        for first in reversed(range(0, 2500, ling3_reference.CHUNK)):
            part = left[:, first:first + ling3_reference.CHUNK] @ right[first:first + ling3_reference.CHUNK]
            expected = part if expected is None else expected + part
        self.assertTrue(np.array_equal(self.product("b", left, right), expected))

    def test_h_is_deterministic_and_moves_products(self):
        left, right = operands(48, 2500, 40, 2)
        first = self.product("h", left, right)
        self.assertTrue(np.array_equal(first, self.product("h", left, right)))
        self.assertTrue(np.array_equal(first, self.product("hp", left, right)))
        self.assertGreater(np.mean(first != self.product("b", left, right)), 0.9)

    def test_h_matches_a_four_product_fp32_chain(self):
        left, right = operands(64, 2048, 96, 3)
        exact = left.astype(np.float64) @ right.astype(np.float64)
        chain = chained(left, right).astype(np.float64) - exact
        modelled = self.product("h", left, right).astype(np.float64) - self.product("b", left, right).astype(np.float64)
        self.assertAlmostEqual(np.mean(np.abs(modelled)) / np.mean(np.abs(chain)), 1.0, delta=0.2)
        self.assertLess(abs(np.mean(modelled)), 0.05 * np.sqrt(np.mean(modelled ** 2)))
        self.assertAlmostEqual(np.sqrt(np.mean(modelled ** 2) / np.mean(chain ** 2)), 1.0, delta=0.25)

    def test_error_grows_with_the_running_sum(self):
        left, right = operands(64, 4096, 64, 4)
        shifted = left.copy()
        shifted[:, :8] = 64.0
        heavy = right.copy()
        heavy[:8] = 0.5
        exact = shifted.astype(np.float64) @ heavy.astype(np.float64)
        early = np.abs(self.product("h", shifted, heavy).astype(np.float64) - exact)
        late_left, late_right = shifted[:, ::-1].copy(), heavy[::-1].copy()
        late = np.abs(self.product("h", late_left, late_right).astype(np.float64) - exact)
        self.assertGreater(np.mean(early), 2.0 * np.mean(late))

    def test_infinite_products_stay_infinite(self):
        device.select("h")
        left = np.ones((2, 4), dtype=np.float32)
        right = np.ones((4, 2), dtype=np.float32)
        total = np.array([[np.inf, 4.0], [-np.inf, np.nan]], dtype=np.float32)
        result = device.accumulated(left, right, total)
        self.assertEqual(result[0, 0], np.inf)
        self.assertEqual(result[1, 0], -np.inf)
        self.assertTrue(np.isnan(result[1, 1]))

    def test_parts_must_match_the_chunking(self):
        device.select("h")
        left, right = operands(4, 2500, 4, 5)
        with self.assertRaises(ValueError):
            device.accumulated(left, right, left @ right, [left @ right], 1024)

    def test_hy4_b_splits_its_products(self):
        self.assertEqual(hy4_reference.CHUNK, 1024)
        left, right = operands(16, 2500, 8, 6)
        hy4_reference.REVERSED = True
        try:
            device.select("b")
            expected = None
            for first in reversed(range(0, 2500, 1024)):
                part = left[:, first:first + 1024] @ right[first:first + 1024]
                expected = part if expected is None else expected + part
            self.assertTrue(np.array_equal(hy4_reference.product(left, right), expected))
        finally:
            hy4_reference.REVERSED = False


if __name__ == "__main__":
    unittest.main()
