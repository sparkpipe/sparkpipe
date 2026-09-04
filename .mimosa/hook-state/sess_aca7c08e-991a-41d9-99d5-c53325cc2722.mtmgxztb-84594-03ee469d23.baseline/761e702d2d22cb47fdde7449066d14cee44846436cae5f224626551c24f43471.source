#!/usr/bin/env python3
"""Context-convention sweep on the GSM round-4 dump: which window/anchor
convention makes the oracle's draft[1] hit emitted[0]=14235?

Dumps: /tmp/ctxwin_taps.bin = taps for positions 84..115 (32 positions),
base=100, window=99 (the device used 0..98 + block at 99..106).
Round 4 truth: emitted[0]=220? no - round 4 is the 4th diag line:
drafts=[5632,5632,25,5632] emitted=[220,220,220,220]: draft[1] must be 220.
Also round 2: drafts[1] must be 14235, round 3: 21902."""
import sys

import numpy as np

sys.path.insert(0, "/home/spark2/sparkpipe/tools")
import qwen38_27b_dspark_reference as R  # noqa: E402

NB_BASE, NB = 84, 32
BASE = 100
N_Q, N_KV, HD = 32, 8, 128
ANCHOR = 5632  # round-4 C0 (the anchor the device used: last input row)
TARGET_DRAFT1 = 220  # emitted[0] for round 4

DRAFTER = R.load_drafter()
LM, EMB = R.load_target_shared()
TAPS_NB = R.bf16_to_f32(np.fromfile("/tmp/ctxwin_taps.bin", dtype=np.uint16)).reshape(NB, 5, 5120)


def run(win_lo, win_hi, anchor, block_base, taps=TAPS_NB, tbase=NB_BASE):
    """win_lo..win_hi-1: context rows (absolute positions); block at
    block_base..block_base+7 (row0=anchor, rows1-7=mask)."""
    lo = max(win_lo, tbase)
    hi = min(win_hi, tbase + NB)
    W = hi - lo
    if W <= 0:
        return None
    ctx = R.bf16_to_f32(R.f32_to_bf16(taps[lo - tbase:hi - tbase].reshape(W, -1) @ DRAFTER["fc.weight"].T))
    ctx_n = R.bf16(R.rms_norm(ctx, DRAFTER["hidden_norm.weight"]))
    block = np.empty((8, 5120), dtype=np.float32)
    block[0] = EMB[anchor]
    block[1:] = EMB[R.MASK_TOKEN_ID]
    x = block
    for L in range(R.N_LAYERS):
        lw = {k.split(f"layers.{L}.")[1]: DRAFTER[k] for k in DRAFTER if f"layers.{L}." in k}
        h = R.bf16(R.rms_norm(x, lw["input_layernorm.weight"]))
        h2, coeff = R.conv_prepare(h, lw, "attention_conv")
        h2 = R.bf16(h2)
        q = R.bf16(h2 @ lw["self_attn.q_proj.weight"].T).reshape(8, N_Q, HD)
        q = R.bf16(R.rms_norm(q, lw["self_attn.q_norm.weight"]))
        q = R.apply_rope(q, np.arange(block_base, block_base + 8)[:, None])
        if W:
            k_ctx = R.bf16(ctx_n @ lw["self_attn.k_proj.weight"].T).reshape(W, N_KV, HD)
            k_ctx = R.bf16(R.rms_norm(k_ctx, lw["self_attn.k_norm.weight"]))
            k_ctx = R.apply_rope(k_ctx, np.arange(lo, hi)[:, None])
            v_ctx = R.bf16(ctx_n @ lw["self_attn.v_proj.weight"].T).reshape(W, N_KV, HD)
        else:
            k_ctx = np.zeros((0, N_KV, HD), dtype=np.float32)
            v_ctx = np.zeros((0, N_KV, HD), dtype=np.float32)
        k_blk = R.bf16(h2 @ lw["self_attn.k_proj.weight"].T).reshape(8, N_KV, HD)
        k_blk = R.bf16(R.rms_norm(k_blk, lw["self_attn.k_norm.weight"]))
        k_blk = R.apply_rope(k_blk, np.arange(block_base, block_base + 8)[:, None])
        v_blk = R.bf16(h2 @ lw["self_attn.v_proj.weight"].T).reshape(8, N_KV, HD)
        K = np.concatenate([k_ctx, k_blk])
        V = np.concatenate([v_ctx, v_blk])
        attn = np.zeros((8, N_Q, HD), dtype=np.float32)
        for qi in range(8):
            for qh in range(N_Q):
                kvh = qh // (N_Q // N_KV)
                sc = (q[qi, qh] @ K[:, kvh].T) * (HD ** -0.5)
                sc -= sc.max()
                p = np.exp(sc)
                p /= p.sum()
                attn[qi, qh] = p @ V[:, kvh]
        o = R.bf16(attn.reshape(8, -1) @ lw["self_attn.o_proj.weight"].T)
        x = R.bf16(x + R.bf16(R.conv_finish(o, coeff, lw, "attention_conv")))
        h = R.bf16(R.rms_norm(x, lw["post_attention_layernorm.weight"]))
        h2, coeff = R.conv_prepare(h, lw, "mlp_conv")
        h2 = R.bf16(h2)
        gate = R.bf16(h2 @ lw["mlp.gate_proj.weight"].T)
        up = R.bf16(h2 @ lw["mlp.up_proj.weight"].T)
        ff = R.bf16(R.silu(gate) * up)
        down = R.bf16(ff @ lw["mlp.down_proj.weight"].T)
        x = R.bf16(x + R.bf16(R.conv_finish(down, coeff, lw, "mlp_conv")))
    hidden = R.rms_norm(x, DRAFTER["norm.weight"])
    ml = R.bf16_to_f32(R.f32_to_bf16((hidden[1:] @ LM.T).astype(np.float32)))
    top_ids = np.argsort(-ml, axis=-1, kind="stable")[:, :R.SELECTOR_TOP_K]
    unary = np.take_along_axis(ml, top_ids, axis=-1)
    hp = R.bf16_to_f32(R.f32_to_bf16(hidden[1:] @ DRAFTER["candidate_selector.hidden_projection.weight"].T))
    sc = R._score_edges(DRAFTER["candidate_selector.predecessor_codebook"],
                        DRAFTER["candidate_selector.successor_codebook"],
                        top_ids[None], unary[None], hp[None],
                        np.array([anchor]), R.SELECTOR_TOP_K)[0]
    drafts = R._greedy_walk(sc, top_ids)
    return drafts, top_ids[:, 0].tolist()


print("target: draft[1] ==", TARGET_DRAFT1, " top1_slot1 should contain it", flush=True)
for name, lo, hi, anch, bb in [
    ("device-current   ctx 0..98  blk@99 anchor=5632", 0, 99, ANCHOR, 99),
    ("incl verify rows ctx 0..107 blk@99 anchor=5632", 0, 108, ANCHOR, 99),
    ("incl base-1      ctx 0..99  blk@100 anchor=5632", 0, 100, ANCHOR, 100),
    ("anchor=emission? ctx 0..99  blk@100 anchor=C0x", 0, 100, 220, 100),
]:
    r = run(lo, hi, anch, bb)
    if r is None:
        print(name, "-> out of dump range")
        continue
    drafts, top1s = r
    hit = "HIT" if drafts[0] == TARGET_DRAFT1 else "miss"
    intop = TARGET_DRAFT1 in top1s[:1]
    print(f"{name}: draft1={drafts[0]} ({hit})  top1s={top1s[:3]}", flush=True)
