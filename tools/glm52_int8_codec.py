from __future__ import annotations

import numpy as np

BLOCK = 128
LEVELS = 127


def bf16_to_f32(values):
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values):
    source = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = source + np.uint32(0x7FFF) + ((source >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def encode(values, block=BLOCK):
    source = np.ascontiguousarray(values).reshape(-1).astype(np.uint16, copy=False)
    if source.size % block:
        raise ValueError("int8 codec element count is not a multiple of the block size")
    exponents = (source >> np.uint16(7)) & np.uint16(0xFF)
    if int(exponents.max(initial=0)) == 0xFF:
        raise ValueError("int8 codec source contains inf or nan")
    blocks = bf16_to_f32(source).reshape(-1, block)
    absolute_maximum = np.abs(blocks).max(axis=1)
    # The stored scale is itself bf16 so that decode is exact in the kernel and
    # re-encoding is idempotent: the block maximum decodes to code LEVELS, whose
    # product with the scale reproduces the stored scale exactly.
    scales = f32_to_bf16(absolute_maximum.astype(np.float32) / np.float32(LEVELS))
    divisor = bf16_to_f32(scales).reshape(-1, 1)
    divisor = np.where(divisor == 0.0, np.float32(1.0), divisor)
    codes = np.clip(np.rint(blocks / divisor), -LEVELS, LEVELS).astype(np.int8)
    zero_blocks = int((absolute_maximum == 0.0).sum())
    stats = {
        "element_count": int(source.size),
        "block_count": int(scales.size),
        "block": int(block),
        "zero_block_count": zero_blocks,
        "clipped_count": int((np.abs(codes) == LEVELS).sum()),
        "bits_per_weight": 8.0 + 16.0 / block,
    }
    return codes, scales, stats


def decode(codes, scales, block=BLOCK):
    packed = np.asarray(codes, dtype=np.int8).reshape(-1, block).astype(np.float32)
    factor = bf16_to_f32(np.asarray(scales, dtype=np.uint16)).reshape(-1, 1)
    return f32_to_bf16(packed * factor).reshape(-1)


def verify(values, codes, scales, block=BLOCK):
    source = np.ascontiguousarray(values).reshape(-1).astype(np.uint16, copy=False)
    decoded = decode(codes, scales, block)
    if decoded.shape != source.shape:
        raise ValueError("int8 codec decode produced the wrong element count")
    recodes, rescales, _ = encode(decoded, block)
    if not np.array_equal(rescales, scales):
        raise ValueError("int8 codec decode and re-encode changed the block scales")
    if not np.array_equal(recodes, codes):
        raise ValueError("int8 codec decode and re-encode is not idempotent")
    original = bf16_to_f32(source).astype(np.float64)
    reconstructed = bf16_to_f32(decoded).astype(np.float64)
    residual = np.linalg.norm(original - reconstructed)
    reference = np.linalg.norm(original)
    if reference > 0.0 and residual / reference > 0.02:
        raise ValueError("int8 codec relative error exceeded the 2 percent guard")
