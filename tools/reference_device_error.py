"""exp and tanh for the FP32 references, exact or with a GPU's measured error.

The references compute every transcendental in exact FP32. A GPU does not:
the M5 Ultra's exp under -fno-fast-math is off from double precision by a
relative error of mean 1.64e-7 and standard deviation 4.35e-7 over [-30, 10]
(about 18 times FP32 rounding), its tanh by an absolute error of mean 6.1e-9
and standard deviation 3.67e-8 over [-6, 6], and its rsqrt rounds like FP32.
Reference variant h (and hp, h with pinned routes) selects that error;
an exp that overflows stays infinite, as on the GPU, rather than turning
into inf - inf. As in
a GPU's approximation the error is a fixed function of the input: a 32-bit
hash of the input's FP32 bits (the same one tools/qwen38_27b_reference.c
uses) gives a uniform draw with that mean and standard deviation, so a
against h measures FP32 reduction-order noise plus the transcendental noise
any GPU adds, which is the floor the Metal gate compares against. Variants
a and b leave both functions exact.

h also carries a GPU matrix unit's accumulation error. The references
multiply in FP32 through BLAS, whose blocked sums are about as accurate as
FP32 allows; the M5 Ultra's MPP matmul keeps one FP32 accumulator for a
whole inner dimension and adds four products to it at a time, and its error
is 3.7 times BLAS's, unbiased, with twice the bf16 rounding flips. A matrix
product a reference takes through accumulated() gains, in h, a normal draw
whose variance is 0.11 u^2 (u = 2^-24) times the expected sum, over the
accumulator's additions, of the running sum's square plus the four-product
tree's own squares: with the inner dimension cut into 1,024-wide chunks of
partial sum Y and squared-product sum V, and S the sum of the chunks before,
each chunk of n additions contributes n (S^2 + S Y + Y^2 / 3 + V / 6) + 2 V.
0.11 matches the M5 Ultra's bf16 rounding flips and its mean error in bf16
steps on random K = 4,096 products and on four of Solar Open 2's projections
(flips within 15%, mean error within 10%). The draw is a Box-Muller normal
from two hashes of the product's FP32 bits, so h stays deterministic. A
product chained over its whole inner dimension is the longest chain a GPU
needs; split-K and tensor-parallel shards add less, so h's allowance holds
for any of them. References computed before this model carry only the
transcendental error: their floors are tighter, and a gate passed against
them stands.
"""

import numpy as np

EXP_ERROR = (1.637e-7, 4.354e-7)
TANH_ERROR = (6.1e-9, 3.67e-8)
SUM_ERROR = 0.11
SUM_STEP = 4
SUM_WIDTH = 1024
ACTIVE = False


def select(label):
    global ACTIVE
    ACTIVE = label in ("h", "hp")


def centered(values, salt):
    x = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32) ^ np.uint32((salt * 0x9E3779B9) & 0xFFFFFFFF)
    x = x ^ (x >> np.uint32(16))
    x = x * np.uint32(0x7FEB352D)
    x = x ^ (x >> np.uint32(15))
    x = x * np.uint32(0x846CA68B)
    x = x ^ (x >> np.uint32(16))
    return ((x >> np.uint32(8)).astype(np.float32) + np.float32(0.5)) * np.float32(2.0 / 16777216.0) - np.float32(1.0)


def draw(values, salt, error):
    return np.float32(error[0]) + np.float32(error[1] * np.sqrt(3.0)) * centered(values, salt)


def exp(values):
    exact = np.exp(values)
    if not ACTIVE:
        return exact
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(exact), exact + exact * draw(values, 1, EXP_ERROR), exact)


def tanh(values):
    exact = np.tanh(values)
    if not ACTIVE:
        return exact
    return exact + draw(values, 2, TANH_ERROR)


def normal(values, salt):
    first = (centered(values, salt).astype(np.float64) + 1.0) / 2.0
    second = centered(values, salt + 1).astype(np.float64)
    return np.sqrt(-2.0 * np.log(first)) * np.cos(np.pi * second)


def accumulated(left, right, total, parts=None, width=SUM_WIDTH):
    if not ACTIVE:
        return total
    depth = left.shape[-1]
    if parts is None:
        parts = [total] if depth <= width else [left[..., first:first + width] @ right[first:first + width] for first in range(0, depth, width)]
    elif len(parts) != -(-depth // width):
        raise ValueError(f"{len(parts)} partial products for an inner dimension of {depth} in chunks of {width}")
    running = np.zeros(np.shape(total), dtype=np.float64)
    expected = np.zeros(np.shape(total), dtype=np.float64)
    for index, part in enumerate(parts):
        first = index * width
        chunk_left = np.asarray(left[..., first:first + width], dtype=np.float32)
        chunk_right = np.asarray(right[first:first + width], dtype=np.float32)
        squares = ((chunk_left * chunk_left) @ (chunk_right * chunk_right)).astype(np.float64)
        part = np.asarray(part, dtype=np.float64)
        expected += chunk_left.shape[-1] / SUM_STEP * (running * running + running * part + part * part / 3.0 + squares / 6.0) + 2.0 * squares
        running += part
    total = np.asarray(total, dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        noise = (np.sqrt(SUM_ERROR * expected) * 2.0 ** -24 * normal(total, 3)).astype(np.float32)
        return np.where(np.isfinite(total), total + noise, total)
