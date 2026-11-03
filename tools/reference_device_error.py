"""exp and tanh for the FP32 references, exact or with a GPU's measured error.

The references compute every transcendental in exact FP32. A GPU does not:
the M5 Ultra's exp under -fno-fast-math is off from double precision by a
relative error of mean 1.64e-7 and standard deviation 4.35e-7 over [-30, 10]
(about 18 times FP32 rounding), its tanh by an absolute error of mean 6.1e-9
and standard deviation 3.67e-8 over [-6, 6], and its rsqrt rounds like FP32.
Reference variant h (and hp, h with pinned routes) selects that error. As in
a GPU's approximation the error is a fixed function of the input: a 32-bit
hash of the input's FP32 bits (the same one tools/qwen38_27b_reference.c
uses) gives a uniform draw with that mean and standard deviation, so a
against h measures FP32 reduction-order noise plus the transcendental noise
any GPU adds, which is the floor the Metal gate compares against. Variants
a and b leave both functions exact.
"""

import numpy as np

EXP_ERROR = (1.637e-7, 4.354e-7)
TANH_ERROR = (6.1e-9, 3.67e-8)
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
    return exact + exact * draw(values, 1, EXP_ERROR)


def tanh(values):
    exact = np.tanh(values)
    if not ACTIVE:
        return exact
    return exact + draw(values, 2, TANH_ERROR)
