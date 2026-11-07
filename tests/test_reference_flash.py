#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import ling3_reference
import reference_flash as flash


def bf16(values):
    return ling3_reference.bf16(values)


def product(left, right):
    return (left @ right).astype(np.float32)


def total(weights):
    return weights.sum(axis=-1, keepdims=True, dtype=np.float32)


def emulate(scores, values, backward):
    out = np.zeros((scores.shape[0], values.shape[1]), dtype=np.float32)
    for row in range(scores.shape[0]):
        visible = np.flatnonzero(np.isfinite(scores[row]))
        tiles = sorted({int(key) // flash.TILE for key in visible}, reverse=backward)
        running, acc, denominator = np.float32(-np.inf), np.zeros(values.shape[1], dtype=np.float32), np.float32(0.0)
        for tile in tiles:
            keys = np.arange(tile * flash.TILE, min((tile + 1) * flash.TILE, scores.shape[1]))
            block = scores[row, keys]
            peak = np.float32(max(running, block.max()))
            factor = np.float32(np.exp(running - peak)) if np.isfinite(running) else np.float32(0.0)
            weights = np.exp(block - peak).astype(np.float32)
            acc = (acc * factor + bf16(weights)[None, :] @ values[keys]).astype(np.float32)[0]
            denominator = np.float32(denominator * factor + weights.sum(dtype=np.float32))
            running = peak
        out[row] = acc * (np.float32(1.0) / denominator) if denominator != 0.0 else acc
    return out


def causal_scores(rows, keys, seed, spread=4.0):
    rng = np.random.default_rng(seed)
    scores = (spread * rng.standard_normal((rows, keys))).astype(np.float32)
    first = keys - rows
    visible = np.arange(keys)[None, :] <= (first + np.arange(rows))[:, None]
    return np.where(visible, scores, -np.inf).astype(np.float32)


class FlashTest(unittest.TestCase):
    def test_one_tile_is_the_single_maximum_form(self):
        scores = causal_scores(40, 60, 1)
        values = bf16(np.random.default_rng(2).standard_normal((60, 16)).astype(np.float32))
        acc, denominator, running = flash.attend(scores, values, product, total, np.exp, bf16)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True)).astype(np.float32)
        self.assertTrue(np.array_equal(acc, product(bf16(weights), values)))
        self.assertTrue(np.array_equal(denominator, total(weights)))
        self.assertTrue(np.array_equal(running, scores.max(axis=-1, keepdims=True)))

    def test_tiles_walk_back_from_the_diagonal(self):
        scores = causal_scores(70, 300, 3)
        scores[:, :8] += np.float32(6.0)
        values = bf16(np.random.default_rng(4).standard_normal((300, 32)).astype(np.float32))
        acc, denominator, _ = flash.attend(scores, values, product, total, np.exp, bf16)
        out = flash.normalize(acc, denominator)
        backward, forward = emulate(scores, values, True), emulate(scores, values, False)
        self.assertLess(np.max(np.abs(out - backward)), 1e-6 * np.max(np.abs(backward)))
        self.assertGreater(np.mean(bf16(out) != bf16(forward)), 0.0)
        single = np.exp(scores - scores.max(axis=-1, keepdims=True)).astype(np.float32)
        self.assertGreater(np.mean(bf16(out) != bf16(product(bf16(single), values) / total(single))), 0.0)

    def test_tiles_align_from_key_zero(self):
        scores = causal_scores(1, 130, 5)
        values = bf16(np.random.default_rng(6).standard_normal((130, 8)).astype(np.float32))
        calls = []
        flash.attend(scores, values, lambda weights, part: calls.append(part.shape[0]) or product(weights, part), total, np.exp, bf16)
        self.assertEqual(calls, [2, 64, 64])

    def test_rows_that_see_nothing_stay_zero(self):
        scores = np.full((3, 100), -np.inf, dtype=np.float32)
        scores[1, 70] = np.float32(1.5)
        values = np.ones((100, 4), dtype=np.float32)
        with np.errstate(invalid="raise"):
            acc, denominator, _ = flash.attend(scores, values, product, total, np.exp, bf16)
            out = flash.normalize(acc, denominator)
        self.assertTrue(np.all(out[0] == 0.0) and np.all(out[2] == 0.0))
        self.assertTrue(np.allclose(out[1], 1.0))


if __name__ == "__main__":
    unittest.main()
