"""Prefill attention in FlashAttention-2's plain forward, the publisher form.

FA2's forward (csrc/flash_attn/src/flash_fwd_launch_template.h and
flash_fwd_kernel.h) cuts a row's keys into 64-key tiles aligned from key 0:
kBlockN is 64 for causal prefill at head size 128 on every GPU class (sm80
and sm90 launch <128, 128, 64>, sm86 and sm89 <128, 64, 64>). It walks the
tiles from the one holding the row's diagonal back to the first key, so the
running maximum starts at the diagonal tile. Each tile's weights exp(s - m),
against the maximum so far, are rounded to bf16 for the value product, the
row sum adds the FP32 weights, and when a tile raises the maximum the
accumulated values and sum are first scaled by exp(m_old - m_new). The
output is the accumulated values times the reciprocal of the sum (a row
whose sum is 0 keeps 0). FA2 runs this one kernel whenever the launch fills
the GPU (flash_api.cpp's num_splits_heuristic: batch x heads x ceil(rows /
64) at least 0.8 x 2 x the SM count), as every gate prefill chunk does;
smaller launches split the keys by the GPU's SM count, which has no single
form. A tile that a row cannot see at all leaves it unchanged, so walking
every tile from the last one back gives each row FA2's order.

attend() takes masked scores (-inf where a key is not visible), the values
and the reference's own product, sum, exp and bf16, so variants a, b and h
each keep their order of sums and device error; normalize() applies FA2's
reciprocal.
"""

import numpy as np

TILE = 64


def attend(scores, values, product, reduce_sum, exp, bf16, tile=TILE):
    rows = scores.shape[0]
    running = np.full((rows, 1), -np.inf, dtype=np.float32)
    total = np.zeros((rows, 1), dtype=np.float32)
    acc = np.zeros((rows, values.shape[-1]), dtype=np.float32)
    for first in reversed(range(0, scores.shape[1], tile)):
        block = scores[:, first:first + tile]
        peak = np.maximum(running, block.max(axis=-1, keepdims=True))
        live = np.isfinite(peak)
        shift = np.where(live, peak, np.float32(0.0)).astype(np.float32)
        factor = np.where(live, exp(running - shift), np.float32(0.0)).astype(np.float32)
        weights = np.where(live, exp(block - shift), np.float32(0.0)).astype(np.float32)
        acc = (acc * factor + product(bf16(weights), values[first:first + tile])).astype(np.float32)
        total = (total * factor + reduce_sum(weights)).astype(np.float32)
        running = np.where(live, peak, running).astype(np.float32)
    return acc, total, running


def normalize(acc, total):
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = np.where((total == 0.0) | np.isnan(total), np.float32(1.0), np.float32(1.0) / total).astype(np.float32)
    return (acc * inverse).astype(np.float32)
