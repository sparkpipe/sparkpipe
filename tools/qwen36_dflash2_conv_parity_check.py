#!/usr/bin/env python3
"""Compare the DFlash2 conv kernel output against the _grouped_conv oracle.

Reads /tmp/dflash2_conv_parity.bin dumped by qwen36_dflash2_conv_parity.cu and
asserts the kernel (BF16 delta, both sides) matches the vLLM _grouped_conv port.
The kernel truncates its output to BF16, so the oracle output is truncated to
BF16 before the bit-exact compare; the f32 max-abs-diff is reported for context.
"""
import numpy as np

B, H, SIDES, TAPS, NUM_GROUPS, GROUP_SIZE = 8, 5120, 2, 2, 320, 16


def _grouped_conv(hidden, delta, base, block_size, num_groups, group_size, taps):
    T = hidden.shape[0]
    blocks = hidden.reshape(T, num_groups, group_size)
    coefficients = base.reshape(1, taps, num_groups, group_size) + delta[:, :, :, None]
    output = coefficients[:, 0] * blocks
    position = np.arange(T, dtype=np.int64)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = np.concatenate(
            [np.zeros((tap, num_groups, group_size), dtype=hidden.dtype), blocks[:-tap]], axis=0
        )
        output += coefficients[:, tap] * shifted * (position >= tap).reshape(-1, 1, 1)
    return output.reshape(T, -1)


def bf16(f):
    u = f.astype(np.float32).view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    u = u + np.uint32(0x7FFF) + lsb
    return ((u >> np.uint32(16)).astype(np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


raw = np.fromfile("/tmp/dflash2_conv_parity.bin", dtype=np.float32)
off = 0
x = raw[off:off + B * H].reshape(B, H); off += B * H
delta_all = raw[off:off + B * SIDES * TAPS * NUM_GROUPS].reshape(B, SIDES, TAPS, NUM_GROUPS); off += B * SIDES * TAPS * NUM_GROUPS
base = raw[off:off + SIDES * TAPS * H].reshape(SIDES, TAPS, H); off += SIDES * TAPS * H
out0 = raw[off:off + B * H].reshape(B, H); off += B * H
out1 = raw[off:off + B * H].reshape(B, H); off += B * H
assert off == raw.size, (off, raw.size)

worst = 0.0
for side, out in ((0, out0), (1, out1)):
    ref = _grouped_conv(x, delta_all[:, side], base[side], B, NUM_GROUPS, GROUP_SIZE, TAPS)
    ref_bf16 = bf16(ref)
    # The kernel's __float2bfloat16 and numpy's round-to-nearest-even can land on
    # opposite zero signs for tiny values; normalize -0.0 -> +0.0 before the
    # bit-exact compare (a sign-bit-only diff is not a numeric error).
    nz = np.count_nonzero((out == 0.0) & (ref_bf16 == 0.0))
    out_n = np.where(out == 0.0, 0.0, out)
    ref_n = np.where(ref_bf16 == 0.0, 0.0, ref_bf16)
    np.testing.assert_array_equal(out_n.view(np.uint32), ref_n.view(np.uint32),
                                  err_msg=f"side {side} BF16 mismatch")
    d = np.abs(out - ref).max()
    worst = max(worst, float(d))
    print(f"side {side}: bit-exact BF16 PASS, f32 max|diff|={d:.3e}")

print(f"CONV_PARITY PASS (both sides bit-exact BF16; f32 worst={worst:.3e})")
