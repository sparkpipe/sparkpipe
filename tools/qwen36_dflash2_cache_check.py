#!/usr/bin/env python3
"""Stage-bisect the cache-based DFlash2 forward against the device dumps:
window taps -> context normed -> per-layer ctx/block K/V (post prep) ->
attention out -> drafts. Uses the /tmp/l0_* and /tmp/ctxwin_* dumps."""
import sys

import numpy as np

sys.path.insert(0, "/home/spark2/sparkpipe/tools")
import qwen36_dspark_reference as R  # noqa: E402

WINDOW = 129
BASE = 129
ANCHOR = 270
N_Q, N_KV, HD, RD = 32, 8, 128, 64


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


taps_win = R.bf16_to_f32(np.fromfile("/tmp/ctxwin_taps.bin", dtype=np.uint16)).reshape(WINDOW, 5, 5120)
sample_rows = [int(l) for l in open("/tmp/l0_sample_rows.txt")]
drafter = R.load_drafter()
lm_head, embed = R.load_target_shared()
lw = {k.split("layers.0.")[1]: drafter[k] for k in drafter if "layers.0." in k}

# 1) context window: fc + hidden_norm
ctx = taps_win.reshape(WINDOW, -1) @ drafter["fc.weight"].T
ctx = R.bf16_to_f32(R.f32_to_bf16(ctx))
ctx_n = R.bf16(R.rms_norm(ctx, drafter["hidden_norm.weight"]))

# 2) layer-0 context k/v (post prep = k-norm + rope at window positions)
k_ctx = R.bf16(ctx_n @ lw["self_attn.k_proj.weight"].T).reshape(WINDOW, N_KV, HD)
k_ctx = R.bf16(R.rms_norm(k_ctx, lw["self_attn.k_norm.weight"]))
k_ctx = R.apply_rope(k_ctx, np.arange(BASE - WINDOW, BASE)[:, None])
v_ctx = R.bf16(ctx_n @ lw["self_attn.v_proj.weight"].T).reshape(WINDOW, N_KV, HD)

# 3) block: conv prepare then q/k/v
block = np.empty((8, 5120), dtype=np.float32)
block[0] = embed[ANCHOR]
block[1:] = embed[R.MASK_TOKEN_ID]
h = R.bf16(R.rms_norm(block, lw["input_layernorm.weight"]))
h2, _ = R.conv_prepare(h, lw, "attention_conv")
h2 = R.bf16(h2)
q = R.bf16(h2 @ lw["self_attn.q_proj.weight"].T).reshape(8, N_Q, HD)
q = R.bf16(R.rms_norm(q, lw["self_attn.q_norm.weight"]))
q = R.apply_rope(q, np.arange(BASE, BASE + 8)[:, None])
k_blk = R.bf16(h2 @ lw["self_attn.k_proj.weight"].T).reshape(8, N_KV, HD)
k_blk = R.bf16(R.rms_norm(k_blk, lw["self_attn.k_norm.weight"]))
k_blk = R.apply_rope(k_blk, np.arange(BASE, BASE + 8)[:, None])
v_blk = R.bf16(h2 @ lw["self_attn.v_proj.weight"].T).reshape(8, N_KV, HD)

K = np.concatenate([k_ctx.reshape(WINDOW, -1), k_blk.reshape(8, -1)])
V = np.concatenate([v_ctx.reshape(WINDOW, -1), v_blk.reshape(8, -1)])

# 4) compare against device dumps
kv_k_dev = R.bf16_to_f32(np.fromfile("/tmp/l0_kv_k.bin", dtype=np.uint16)).reshape(-1, 1024)
kv_v_dev = R.bf16_to_f32(np.fromfile("/tmp/l0_kv_v.bin", dtype=np.uint16)).reshape(-1, 1024)
q_dev = R.bf16_to_f32(np.fromfile("/tmp/l0_q.bin", dtype=np.uint16)).reshape(-1, 4096)
for si, r in enumerate(sample_rows):
    print(f"row {r}: k_dev_vs_ref rel={rel(kv_k_dev[si], K[r]):.5f}  v rel={rel(kv_v_dev[si], V[r]):.5f}")
print("q row0 rel:", rel(q_dev[0], q[0].reshape(-1)))
print("q row1 rel:", rel(q_dev[1], q[1].reshape(-1)))

# 5) reference attention output row 0
Kf = K.reshape(-1, N_KV, HD)
Vf = V.reshape(-1, N_KV, HD)
attn = np.zeros((8, N_Q, HD), dtype=np.float32)
scale = HD ** -0.5
for qi in range(8):
    for qh in range(N_Q):
        kvh = qh // (N_Q // N_KV)
        sc = (q[qi, qh] @ Kf[:, kvh].T) * scale
        sc = sc - sc.max()
        p = np.exp(sc)
        p /= p.sum()
        attn[qi, qh] = p @ Vf[:, kvh]
attn_dev = R.bf16_to_f32(np.fromfile("/tmp/l0_attn.bin", dtype=np.uint16)).reshape(8, 4096)
print("attn row0 rel:", rel(attn_dev[0], R.bf16(attn[0].reshape(-1))))
