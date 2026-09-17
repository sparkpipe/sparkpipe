#!/usr/bin/env python3
"""Bisect the DFlash2 device forward against the numpy reference.

The stage dumps for ctx/block0/taps/c0/hidden/logits are one-shot (first
draft frame); the in-loop l0_* dumps are unguarded (last frame). Mask rows
(1..7) are frame-invariant, so every stage compares rows 1..7 only."""
import struct
import sys

import numpy as np

sys.path.insert(0, "/home/spark2/sparkpipe/tools")
import qwen38_27b_dspark_reference as R  # noqa: E402

CONV_ROWS = 2 * R.CONV_KERNEL_SIZE * (R.HIDDEN // R.CONV_GROUP_SIZE)  # 1280
MASK = slice(1, R.BLOCK)


def ld(name, shape):
    a = np.fromfile(f"/tmp/dflash2_stage_{name}.bin", dtype=np.uint16)
    return R.bf16_to_f32(a).reshape(shape)


def rel(a, b):
    a, b = a[MASK], b[MASK]
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


taps = R.bf16_to_f32(np.fromfile("/tmp/dflash2_taps.bin", dtype=np.uint16)).reshape(R.TAPS, R.HIDDEN)
c0 = struct.unpack("<I", open("/tmp/dflash2_c0.bin", "rb").read(4))[0]
print("c0", c0, "(all comparisons on frame-invariant mask rows 1..7)")
drafter = R.load_drafter()
lm_head, embed = R.load_target_shared()

ctx = R.rms_norm(drafter["fc.weight"] @ taps.reshape(-1), drafter["hidden_norm.weight"])
print(f"{'ctx':<13}", rel(ld("ctx", (R.HIDDEN,))[None, :], ctx[None, :]))

block = np.empty((R.BLOCK, R.HIDDEN), dtype=np.float32)
block[0] = embed[c0]
block[1:] = embed[R.MASK_TOKEN_ID]
print(f"{'block0':<13}", rel(ld("block0", (R.BLOCK, R.HIDDEN)), block))

lw = {k.split("layers.0.")[1]: drafter[k] for k in drafter if "layers.0." in k}
h = R.bf16(R.rms_norm(block, lw["input_layernorm.weight"]))
print(f"{'l0_norm':<13}", rel(ld("l0_norm", (R.BLOCK, R.HIDDEN)), h))

h2, coeff = R.conv_prepare(h, lw, "attention_conv")
h2 = R.bf16(h2)
print(f"{'l0_convprep':<13}", rel(ld("l0_convh", (R.BLOCK, R.HIDDEN)), h2))
dev_conva = ld("l0_conva", (R.BLOCK, CONV_ROWS))
ref_delta = R.bf16_to_f32(R.f32_to_bf16((h @ lw["attention_conv.kernel_projection.weight"].T).reshape(R.BLOCK, 2, R.CONV_KERNEL_SIZE, -1)))
print(f"{'l0_conva':<13}", rel(dev_conva.reshape(R.BLOCK, 2, R.CONV_KERNEL_SIZE, -1), ref_delta))

positions_q = np.arange(R.BASE_POS, R.BASE_POS + R.BLOCK, dtype=np.float32)
pos_ctx = R.BASE_POS - 1.0
q = R.bf16(h2 @ lw["self_attn.q_proj.weight"].T).reshape(R.BLOCK, R.N_Q_HEADS, R.HEAD_DIM)
q = R.bf16(R.rms_norm(q, lw["self_attn.q_norm.weight"]))
q = R.apply_rope(q, positions_q[:, None])
k_ctx = R.bf16(ctx[None, :] @ lw["self_attn.k_proj.weight"].T)
k_blk = R.bf16(h2 @ lw["self_attn.k_proj.weight"].T)
v_ctx = R.bf16(ctx[None, :] @ lw["self_attn.v_proj.weight"].T)
v_blk = R.bf16(h2 @ lw["self_attn.v_proj.weight"].T)
k = np.concatenate([k_ctx, k_blk], axis=0).reshape(R.BLOCK + 1, R.N_KV_HEADS, R.HEAD_DIM)
v = np.concatenate([v_ctx, v_blk], axis=0).reshape(R.BLOCK + 1, R.N_KV_HEADS, R.HEAD_DIM)
k = R.bf16(R.rms_norm(k, lw["self_attn.k_norm.weight"]))
k = R.apply_rope(k, np.concatenate([np.array([pos_ctx]), positions_q])[:, None])
scale = R.HEAD_DIM ** -0.5
attn = np.zeros((R.BLOCK, R.N_Q_HEADS, R.HEAD_DIM), dtype=np.float32)
for qi in range(R.BLOCK):
    for qh in range(R.N_Q_HEADS):
        kvh = qh // (R.N_Q_HEADS // R.N_KV_HEADS)
        s = (q[qi, qh] @ k[:, kvh].T) * scale
        s = s - s.max()
        p = np.exp(s)
        p = p / p.sum()
        attn[qi, qh] = p @ v[:, kvh]
attn = R.bf16(attn.reshape(R.BLOCK, R.N_Q_HEADS * R.HEAD_DIM))
dev_q = ld("l0_q", (R.BLOCK, R.N_Q_HEADS * R.HEAD_DIM))
ref_q_raw = R.bf16(h2 @ lw["self_attn.q_proj.weight"].T).reshape(R.BLOCK, -1)
print(f"{'l0_q_raw':<13}", rel(dev_q, ref_q_raw))
dev_k = ld("l0_k", (R.BLOCK + 1, R.N_KV_HEADS * R.HEAD_DIM))
ref_k_raw = np.concatenate([k_ctx, R.bf16(h2 @ lw["self_attn.k_proj.weight"].T)], axis=0).reshape(R.BLOCK + 1, -1)
print(f"{'l0_k_raw(rows1+)':<13}", float(np.linalg.norm(dev_k[MASK] - ref_k_raw[MASK]) / (np.linalg.norm(ref_k_raw[MASK]) + 1e-9)))
print(f"{'l0_attn':<13}", rel(ld("l0_attn", (R.BLOCK, R.N_Q_HEADS * R.HEAD_DIM)), attn))

o = R.bf16(attn.reshape(R.BLOCK, -1) @ lw["self_attn.o_proj.weight"].T)
fin = R.bf16(R.conv_finish(o, coeff, lw, "attention_conv"))
print(f"{'l0_attfinish':<13}", rel(ld("l0_attfinish", (R.BLOCK, R.HIDDEN)), fin))
x1 = R.bf16(block + fin)
print(f"{'l0_x1':<13}", rel(ld("l0_x1", (R.BLOCK, R.HIDDEN)), x1))

h2 = R.bf16(R.rms_norm(x1, lw["post_attention_layernorm.weight"]))
print(f"{'l0_norm2':<13}", rel(ld("l0_norm2", (R.BLOCK, R.HIDDEN)), h2))
h3, coeff2 = R.conv_prepare(h2, lw, "mlp_conv")
h3 = R.bf16(h3)
print(f"{'l0_mlpprep':<13}", rel(ld("l0_mlpprep", (R.BLOCK, R.HIDDEN)), h3))
dev_convm = ld("l0_convm", (R.BLOCK, CONV_ROWS))
ref_dm = R.bf16_to_f32(R.f32_to_bf16((h2 @ lw["mlp_conv.kernel_projection.weight"].T).reshape(R.BLOCK, 2, R.CONV_KERNEL_SIZE, -1)))
print(f"{'l0_convm':<13}", rel(dev_convm.reshape(R.BLOCK, 2, R.CONV_KERNEL_SIZE, -1), ref_dm))
gate = R.bf16(h3 @ lw["mlp.gate_proj.weight"].T)
up = R.bf16(h3 @ lw["mlp.up_proj.weight"].T)
ff = R.bf16(R.silu(gate) * up)
print(f"{'l0_ff(swiglu)':<13}", rel(ld("l0_ff", (R.BLOCK, R.FFN)), ff))
down = R.bf16(ff @ lw["mlp.down_proj.weight"].T)
fin2 = R.bf16(R.conv_finish(down, coeff2, lw, "mlp_conv"))
print(f"{'l0_mlpfinish':<13}", rel(ld("l0_mlpfinish", (R.BLOCK, R.HIDDEN)), fin2))
