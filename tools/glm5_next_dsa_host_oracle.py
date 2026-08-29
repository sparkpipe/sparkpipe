#!/usr/bin/env python3
# glm5_next DSA-site host oracle — glm5-dsa lane.
#
# Recomputes the FIRST DSA layer's attention site (layer 3: mHC wrap +
# nope-absorbed MLA + latent cache) on the host from the CHECKPOINT's
# semantics, and compares every stage against the G5N-VEC dumps the
# module's diag build printed for the same request. Shares no math with
# the module: weights come from the checkpoint safetensors (fp8 block
# dequant for q_a/q_b/kv_a/o_proj, bf16 for kv_b/hc_*), the formulas from
# the reference (absorbed nope-only scoring, qk_head_dim 256, scale
# 256**-0.5, kv latent 512, no rope; mHC pre=sigmoid(w*s0+b0)+eps,
# post=2*sigmoid(...), comb=softmax+eps then 20-iter sinkhorn, collapse
# = sum_s pre_s * stream_s) — not from layer.cuh / attn.cuh.
#
# Stage ladder, per pass (waves are single-row: pass p IS position p):
#   hc_streams (INPUT, dumped) -> mixes -> pre/post/comb (sinkhorn)
#   -> hc_collapsed -> attn_normed -> q_a -> q_compressed (normed)
#   -> q_b -> q | kv_a -> kv_slot (also the oracle's cache row p)
#   -> absorbed query_latent -> attention (softmax over cache 0..p)
#   -> attn_latent -> v proj -> attn_value -> o_proj rank partial.
# Every stage consumes the DUMPED upstream buffer, so any divergence
# convicts ONE kernel, not an accumulation.
#
# usage: glm5_next_dsa_host_oracle.py <residentd.log> <prompt_ids.json> [layer]
# Emit the dumps with: SPARK_GLM5_NEXT_PROBE_VEC=1 SPARK_GLM5_NEXT_PROBE_VEC_DSA=1
# SPARK_GLM5_NEXT_PROBE_VEC_LAYER=<layer> (default 3) [SPARK_GLM5_NEXT_PROBE_VEC_PASSES=n]
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm5_next_kda_host_oracle import (  # noqa: E402
    CKPT, PREFIX, TP, RANK, HIDDEN, RMS_EPS,
    Safetensors, Report,
    bf16_to_f32, f32_to_bf16_u16, bf16_round_f32, parse_log, rmsnorm,
)

LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else 3
HEADS = 64
LATENT = 512
NOPE = 256
VDIM = 256
Q_A = 1536
QK_SCALE = NOPE ** -0.5  # the reference's qk_head_dim ** -0.5
HC = 4
MIX_ROWS = (2 + HC) * HC
FLAT = HC * HIDDEN
SINKHORN = 20
HC_EPS = 1e-6

RANK_HEADS = HEADS // TP  # 4

# fp8 e4m3 (OCP, bias 7) decode LUT
_FP8_LUT = np.zeros(256, dtype=np.float32)
for _i in range(256):
    _s = -1.0 if _i & 0x80 else 1.0
    _e = (_i >> 3) & 0xF
    _m = _i & 0x7
    if _e == 0:
        _v = _m / 8.0 * 2.0 ** -6
    elif _e == 15 and _m == 7:
        _v = np.nan
    else:
        _v = (1.0 + _m / 8.0) * 2.0 ** (_e - 7)
    _FP8_LUT[_i] = _s * _v


def fp8_block_to_bf16(payload_u8, scale_inv, out_dim, in_dim):
    """checkpoint fp8_block weight -> the bf16 the pack loads (expanded
    across the [128,128] scale blocks at load)."""
    w = _FP8_LUT[payload_u8.reshape(out_dim, in_dim)].astype(np.float32)
    s = np.repeat(np.repeat(scale_inv, 128, axis=0), 128, axis=1)
    s = s[:out_dim, :in_dim]
    return f32_to_bf16_u16(w * s)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class DsaSafetensors(Safetensors):
    """the DSA layers carry fp8 tensors (dtype 'F8_E4M3'); the payload is
    read as raw u8 and decoded through the LUT above."""

    @staticmethod
    def _np(dt):
        if dt == "F8_E4M3":
            return np.uint8
        return Safetensors._np(dt)


def sinkhorn_comb(mixes_f32, scale3, base24):
    """the checkpoint's mHC comb: softmax(+eps) per row, then 20 sinkhorn
    iterations - column norm EVERY iteration, row norm skipped on the
    first (elided by the softmax)."""
    comb = (mixes_f32[2 * HC:].reshape(HC, HC) * scale3[2]
            + base24[2 * HC:].reshape(HC, HC)).astype(np.float32)
    comb = np.exp(comb - comb.max())
    comb = comb / comb.sum(axis=1, keepdims=True) + HC_EPS
    for it in range(SINKHORN):
        if it != 0:
            comb = comb / (comb.sum(axis=1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(axis=0, keepdims=True) + HC_EPS)
    return comb


def main():
    log = sys.argv[1]
    prompt_ids = json.load(open(sys.argv[2]))
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["prompt_token_ids"]
    prompt_len = len(prompt_ids)
    st = DsaSafetensors(CKPT)
    P = PREFIX + f"{LAYER}.self_attn."

    # --- checkpoint weights, rank-0 shards
    def w_fp8(name):
        return (st.raw(P + name + ".weight"),
                st.raw(P + name + ".weight_scale_inv").astype(np.float32))

    qa_p, qa_s = w_fp8("q_a_proj")
    qb_p, qb_s = w_fp8("q_b_proj")
    kva_p, kva_s = w_fp8("kv_a_proj_with_mqa")
    op_p, op_s = w_fp8("o_proj")
    qa_w = bf16_to_f32(fp8_block_to_bf16(qa_p, qa_s, Q_A, HIDDEN)).astype(np.float32)
    qb_w = bf16_to_f32(fp8_block_to_bf16(qb_p, qb_s, HEADS * NOPE, Q_A)).astype(np.float32)
    kva_w = bf16_to_f32(fp8_block_to_bf16(kva_p, kva_s, LATENT, HIDDEN)).astype(np.float32)
    op_w = bf16_to_f32(fp8_block_to_bf16(op_p, op_s, HIDDEN, HEADS * VDIM)).astype(np.float32)
    kvb = st.raw(P + "kv_b_proj.weight")  # bf16 [64*512, 512]
    q_a_norm_w = bf16_to_f32(st.raw(P + "q_a_layernorm.weight")).astype(np.float32)
    kv_a_norm_w = bf16_to_f32(st.raw(P + "kv_a_layernorm.weight")).astype(np.float32)
    attn_norm_w = bf16_to_f32(st.raw(PREFIX + f"{LAYER}.input_layernorm.weight")).astype(np.float32)
    hc_fn = bf16_to_f32(st.raw(PREFIX + f"{LAYER}.hc_attn_fn")).astype(np.float32)  # [24, 16384]
    hc_base = st.raw(PREFIX + f"{LAYER}.hc_attn_base").astype(np.float32)  # [24]
    hc_scale = st.raw(PREFIX + f"{LAYER}.hc_attn_scale").astype(np.float32)  # [3]
    print(f"loaded checkpoint tensors for layer {LAYER} rank {RANK}/{TP} "
          f"(kv_b bf16; q_a/q_b/kv_a/o_proj fp8->bf16)")

    passes, _head_rows = parse_log(log)
    got = sorted(p for p in passes if passes[p].get("attn_normed"))
    if not got:
        sys.exit("no G5N-VEC attn_normed dumps found in the log "
                 "(arm SPARK_GLM5_NEXT_PROBE_VEC=1 SPARK_GLM5_NEXT_PROBE_VEC_DSA=1)")
    print(f"log has passes {got[0]}..{got[-1]}, prompt_len={prompt_len}")

    rep = Report()
    cache = {}  # position -> bf16 kv_slot (u16[512]), the oracle's latent cache

    for p in got:
        d = passes[p]
        hc_streams = d["hc_streams"][1]
        hc_collapsed = d["hc_collapsed"][1]
        hc_mixes = d["hc_mixes"][1]
        hc_pre = d["hc_pre"][1]
        hc_post = d["hc_post"][1]
        hc_comb = d["hc_comb"][1]
        normed = d["attn_normed"][1]
        q_compressed = d["q_compressed"][1]
        q_dump = d["q"][1]
        kv_slot = d["kv_slot"][1]
        query_latent = d["query_latent"][1]
        attn_latent = d["attn_latent"][1]
        attn_value = d["attn_value"][1]
        out_partial = d["attn_out_partial"][1]

        # ---- 1) mHC mix projection: unweighted-RMSNorm'd flat streams x fn
        flat = bf16_to_f32(hc_streams).astype(np.float32)
        inv = 1.0 / np.sqrt((flat * flat).sum() / FLAT + RMS_EPS)
        mixes = (hc_fn @ flat) * inv
        rep.add("hc_mixes", p, mixes.view(np.uint32), hc_mixes, "f32")

        # ---- 2) pre / post / comb (sinkhorn), from the DUMPED mixes
        mx = hc_mixes.view(np.float32).astype(np.float32)
        pre = sigmoid(mx[0:HC] * hc_scale[0] + hc_base[0:HC]) + HC_EPS
        post = 2.0 * sigmoid(mx[HC:2 * HC] * hc_scale[1] + hc_base[HC:2 * HC])
        comb = sinkhorn_comb(mx, hc_scale, hc_base)
        rep.add("hc_pre", p, pre.view(np.uint32), hc_pre, "f32")
        rep.add("hc_post", p, post.view(np.uint32), hc_post, "f32")
        rep.add("hc_comb", p, comb.reshape(-1).view(np.uint32), hc_comb, "f32")

        # ---- 3) collapse: sum_s pre_s * stream_s (from DUMPED pre)
        pre_d = hc_pre.view(np.float32).astype(np.float32)
        streams = flat.reshape(HC, HIDDEN)
        collapsed = (pre_d.reshape(HC, 1) * streams).sum(axis=0)
        rep.add("hc_collapsed", p, f32_to_bf16_u16(collapsed), hc_collapsed, "bf16")

        # ---- 4) attn input norm (from DUMPED collapsed)
        cx = bf16_to_f32(hc_collapsed).astype(np.float32)
        rep.add("attn_normed", p,
                f32_to_bf16_u16(rmsnorm(cx, attn_norm_w, RMS_EPS)), normed, "bf16")

        # ---- 5) q_a: GEMM then plain RMSNorm (from DUMPED normed)
        x = bf16_to_f32(normed).astype(np.float32)
        qc_raw = bf16_round_f32(x @ qa_w.T)
        rep.add("q_compressed", p,
                f32_to_bf16_u16(rmsnorm(qc_raw, q_a_norm_w, RMS_EPS)), q_compressed, "bf16")

        # ---- 6) q_b (from DUMPED q_compressed): rank rows = heads 0..3
        qc = bf16_to_f32(q_compressed).astype(np.float32)
        rep.add("q", p, f32_to_bf16_u16(bf16_round_f32(qc @ qb_w[:RANK_HEADS * NOPE].T)),
                q_dump, "bf16")

        # ---- 7) kv_a + norm (from DUMPED normed); ALSO the oracle's cache row
        kvs_raw = bf16_round_f32(x @ kva_w.T)
        rep.add("kv_slot", p,
                f32_to_bf16_u16(rmsnorm(kvs_raw, kv_a_norm_w, RMS_EPS)), kv_slot, "bf16")
        cache[p] = kv_slot

        # ---- 8) absorbed scoring projection (from DUMPED q):
        # query_latent[h] = q[h] @ W_k[h] with W_k[h] = kv_b rows
        # [h*512, h*512+256) ([NOPE, LATENT], bf16) - the pack's transposed
        # key half read the other way.
        qh = bf16_to_f32(q_dump).astype(np.float32).reshape(RANK_HEADS, NOPE)
        ql = np.empty((RANK_HEADS, LATENT), dtype=np.float32)
        for h in range(RANK_HEADS):
            wk = bf16_to_f32(kvb[(RANK * RANK_HEADS + h) * (NOPE + VDIM):
                                 (RANK * RANK_HEADS + h) * (NOPE + VDIM) + NOPE]).astype(np.float32)
            ql[h] = bf16_round_f32(qh[h] @ wk)
        rep.add("query_latent", p, f32_to_bf16_u16(ql.reshape(-1)), query_latent, "bf16")

        # ---- 9) attention: softmax over cache positions 0..p of the DUMPED
        # kv_slot rows (the module stored exactly these), scaled dot per head.
        qlh = bf16_to_f32(query_latent).astype(np.float32).reshape(RANK_HEADS, LATENT)
        positions = sorted(cache)
        slots = np.stack([bf16_to_f32(cache[pp]).astype(np.float32) for pp in positions])
        al = np.empty((RANK_HEADS, LATENT), dtype=np.float32)
        for h in range(RANK_HEADS):
            scores = (slots @ qlh[h]) * np.float32(QK_SCALE)
            e = np.exp(scores - scores.max())
            e /= e.sum()
            al[h] = e @ slots
        rep.add("attn_latent", p, f32_to_bf16_u16(al.reshape(-1)), attn_latent, "bf16")

        # ---- 10) value projection (from DUMPED attn_latent)
        alh = bf16_to_f32(attn_latent).astype(np.float32).reshape(RANK_HEADS, LATENT)
        av = np.empty((RANK_HEADS, VDIM), dtype=np.float32)
        for h in range(RANK_HEADS):
            wv = bf16_to_f32(kvb[(RANK * RANK_HEADS + h) * (NOPE + VDIM) + NOPE:
                                 (RANK * RANK_HEADS + h) * (NOPE + VDIM) + NOPE + VDIM]).astype(np.float32)
            av[h] = bf16_round_f32(alh[h] @ wv.T)
        rep.add("attn_value", p, f32_to_bf16_u16(av.reshape(-1)), attn_value, "bf16")

        # ---- 11) o_proj rank partial (from DUMPED attn_value): columns
        # 0..RANK_HEADS*VDIM of the [4096, 64*256] weight.
        avf = bf16_to_f32(attn_value).astype(np.float32)
        rep.add("attn_out_partial", p,
                f32_to_bf16_u16(bf16_round_f32(avf @ op_w[:, :RANK_HEADS * VDIM].T)),
                out_partial, "bf16")

    bad = rep.show()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
