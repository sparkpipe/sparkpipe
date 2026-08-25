#!/usr/bin/env python3
"""Deep-draft parity: run the numpy reference on one live drafter dump and
score it against the device's own drafts and the round's truth.

Inputs (produced by SPARK_QWEN38_27B_DFLASH2_CTX_DUMP=N on spark2):
  /tmp/ctxwin_taps.bin   [nb, 5, H] bf16 taps for positions 0..nb-1
  /tmp/ctxwin.meta       base=... window=... nb_base=... nb=...
  /tmp/ctxwin_anchor     the anchor token id (the frame EMISSION)
  /tmp/ctxwin_device_drafts  the device walk's 7 draft ids, same run
  argv[1] (optional): the round's 8 emitted ids (truth), comma-separated,
  from the widened qwen38_27b_spec_diag line of the round that consumed these
  drafts (its drafts[] tail must equal the device dump).

Convention under test (the device's): context rows = window positions
[base-window, base-2], block rows at [base-1, base+6], anchor embed at
base-1. Truth[j] = emitted[j]: draft j predicts the token at base+j.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/spark2/sparkpipe/tools")
import qwen38_27b_dspark_reference as R  # noqa: E402

N_Q, N_KV, HD = R.N_Q_HEADS, R.N_KV_HEADS, R.HEAD_DIM


def load_meta():
    meta = {}
    for field in open("/tmp/ctxwin.meta").read().split():
        k, _, v = field.partition("=")
        meta[k] = int(v)
    return meta


def run_reference(taps, tbase, lo, hi, anchor, block_base):
    W = hi - lo
    drafter = R.load_drafter()
    lm, emb = R.load_target_shared()
    ctx = R.bf16_to_f32(R.f32_to_bf16(taps[lo - tbase:hi - tbase].reshape(W, -1) @ drafter["fc.weight"].T))
    ctx_n = R.bf16(R.rms_norm(ctx, drafter["hidden_norm.weight"]))
    block = np.empty((R.BLOCK, R.HIDDEN), dtype=np.float32)
    block[0] = emb[anchor]
    block[1:] = emb[R.MASK_TOKEN_ID]
    x = block
    for layer in range(R.N_LAYERS):
        lw = {k.split(f"layers.{layer}.")[1]: drafter[k] for k in drafter if f"layers.{layer}." in k}
        h = R.bf16(R.rms_norm(x, lw["input_layernorm.weight"]))
        h2, coeff = R.conv_prepare(h, lw, "attention_conv")
        h2 = R.bf16(h2)
        q = R.bf16(h2 @ lw["self_attn.q_proj.weight"].T).reshape(R.BLOCK, N_Q, HD)
        q = R.bf16(R.rms_norm(q, lw["self_attn.q_norm.weight"]))
        q = R.apply_rope_neox(q, np.arange(block_base, block_base + R.BLOCK)[:, None])
        k_ctx = R.bf16(ctx_n @ lw["self_attn.k_proj.weight"].T).reshape(W, N_KV, HD)
        k_ctx = R.bf16(R.rms_norm(k_ctx, lw["self_attn.k_norm.weight"]))
        k_ctx = R.apply_rope_neox(k_ctx, np.arange(lo, hi)[:, None])
        v_ctx = R.bf16(ctx_n @ lw["self_attn.v_proj.weight"].T).reshape(W, N_KV, HD)
        k_blk = R.bf16(h2 @ lw["self_attn.k_proj.weight"].T).reshape(R.BLOCK, N_KV, HD)
        k_blk = R.bf16(R.rms_norm(k_blk, lw["self_attn.k_norm.weight"]))
        k_blk = R.apply_rope_neox(k_blk, np.arange(block_base, block_base + R.BLOCK)[:, None])
        v_blk = R.bf16(h2 @ lw["self_attn.v_proj.weight"].T).reshape(R.BLOCK, N_KV, HD)
        K = np.concatenate([k_ctx, k_blk])
        V = np.concatenate([v_ctx, v_blk])
        attn = np.zeros((R.BLOCK, N_Q, HD), dtype=np.float32)
        for qi in range(R.BLOCK):
            for qh in range(N_Q):
                kvh = qh // (N_Q // N_KV)
                sc = (q[qi, qh] @ K[:, kvh].T) * (HD ** -0.5)
                sc -= sc.max()
                p = np.exp(sc)
                p /= p.sum()
                attn[qi, qh] = p @ V[:, kvh]
        o = R.bf16(attn.reshape(R.BLOCK, -1) @ lw["self_attn.o_proj.weight"].T)
        x = R.bf16(x + R.bf16(R.conv_finish(o, coeff, lw, "attention_conv")))
        h = R.bf16(R.rms_norm(x, lw["post_attention_layernorm.weight"]))
        h2, coeff = R.conv_prepare(h, lw, "mlp_conv")
        h2 = R.bf16(h2)
        gate = R.bf16(h2 @ lw["mlp.gate_proj.weight"].T)
        up = R.bf16(h2 @ lw["mlp.up_proj.weight"].T)
        ff = R.bf16(R.silu(gate) * up)
        down = R.bf16(ff @ lw["mlp.down_proj.weight"].T)
        x = R.bf16(x + R.bf16(R.conv_finish(down, coeff, lw, "mlp_conv")))
    hidden = R.rms_norm(x, drafter["norm.weight"])
    ml = R.bf16_to_f32(R.f32_to_bf16((hidden[1:] @ lm.T).astype(np.float32)))
    top_ids = np.argsort(-ml, axis=-1, kind="stable")[:, :R.SELECTOR_TOP_K]
    unary = np.take_along_axis(ml, top_ids, axis=-1)
    hp = R.bf16_to_f32(R.f32_to_bf16(hidden[1:] @ drafter["candidate_selector.hidden_projection.weight"].T))
    sc = R._score_edges(
        drafter["candidate_selector.predecessor_codebook"],
        drafter["candidate_selector.successor_codebook"],
        top_ids[None], unary[None], hp[None],
        np.array([anchor]), R.SELECTOR_TOP_K,
    )[0]
    return R._greedy_walk(sc, top_ids), top_ids[:, 0].tolist()


def main():
    meta = load_meta()
    base, window, nb_base, nb = meta["base"], meta["window"], meta["nb_base"], meta["nb"]
    argv = sys.argv[1:]
    anchor = int(argv[0]) if argv else int(open("/tmp/ctxwin_anchor").read())
    device = ([int(t) for t in argv[1].split(",")] if len(argv) > 1
              else [int(t) for t in open("/tmp/ctxwin_device_drafts").read().split()])
    truth = [int(t) for t in argv[2].split(",")] if len(argv) > 2 else None
    taps = R.bf16_to_f32(np.fromfile("/tmp/ctxwin_taps.bin", dtype=np.uint16)).reshape(nb, R.TAPS, R.HIDDEN)
    print(f"base={base} window={window} anchor={anchor} nb_base={nb_base} nb={nb}")
    print(f"device drafts: {device}")
    if truth is not None:
        dev_hits = "".join("1" if d == t else "." for d, t in zip(device, truth))
        print(f"device hits:   {dev_hits}  ({sum(d == t for d, t in zip(device, truth))}/{len(device)})")
    # the device convention: window rows ending at base-2, block at base-1
    # the device convention since the sweep fix: context [base-window, base)
    # INCLUDING the walked row's tap g_P, block at base (the deferred pair)
    lo, hi, bb = base - window, base, base
    if lo < nb_base or bb + R.BLOCK > nb_base + nb:
        print(f"dump range {nb_base}..{nb_base + nb - 1} does not cover ctx {lo}..{hi - 1} + block {bb}..{bb + 7}")
        return 1
    ref, top1 = run_reference(taps, nb_base, lo, hi, anchor, bb)
    print(f"ref drafts:    {ref}")
    print(f"ref top1s:     {top1}")
    agree = "".join("1" if a == b else "." for a, b in zip(ref, device))
    print(f"dev==ref:      {agree}")
    if truth is not None:
        ref_hits = "".join("1" if d == t else "." for d, t in zip(ref, truth))
        t1_hits = "".join("1" if d == t else "." for d, t in zip(top1, truth))
        print(f"ref hits:      {ref_hits}  ({sum(d == t for d, t in zip(ref, truth))}/{len(ref)})")
        print(f"top1 hits:     {t1_hits}  (greedy-argmax baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
