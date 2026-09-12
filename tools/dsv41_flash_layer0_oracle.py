#!/usr/bin/env python3
"""dsv41-flash decode host oracle (M7 host-oracle-first piece ladder).

Recomputes the layer-0 block (SWA-only) and the kv_source CSA2 layer blocks
(per layer_plan: 2/8/14 pool m=2 softmax-gated 2-token pairs with carried
state, 20 projects m=1 norm(wkv(x)) per token; each owns its indexer
32h x 128d top-512) from a TP8 rank stagepack and EMITS per-piece
expectations; tests/test_dsv41_flash_layer0_anchor.c recomputes the same
pieces from the same pack bytes in C and prints the per-piece deltas. No
module/driver code is imported: the formulas come from the pinned reference
sources (cache/dsv41_warm/ref/{model,kernel}.py at revision dba1be0a, sha256
recorded in model_contracts/dsv41_flash_authoritative.json):

  linear    = act_quant(x, 32, ue8m0) then fp8-block GEMM: per 32-wide
              k-block fp32 accumulation scaled by sa[i,kb]*sb[j,kb], bf16 out
  act_quant = s = 2^ceil(log2(amax/448)) per 32 block (amax floored 1e-4),
              q = fp8e4m3(clamp(x/s, +/-448)); the inplace variant the
              window-KV cache uses stores bf16(q*s)
  fp4 rt    = indexer q/k: e8m0 scale 2^ceil(log2(amax/6)) per 32, dequant
              written back bf16; compressed KV: e4m3 scale fp8e4m3(amax/6)
              per 16 (amax floored 6*2^-9), dequant written back bf16
  mHC       = pre=sigmoid(m*s0+b0)+eps; post=2*sigmoid(m*s1+b1);
              comb=softmax(m*s2+b2, rows)+eps, col-norm, then
              (iters-1) x (row-norm, col-norm), all +eps
  gate      = sqrt(softplus(x.W)) fp32 (bf16 weight, fp32 math); top-6 of
              score+bias; weights/(sum + 1e-20) * 1.5
  rope      = layer 0: pure theta 10000 on the 64-dim tail. CSA2 layers 2/8:
              ONE yarn table (base 160000, factor 16, original 65536,
              beta 32/1) shared by q, window kv, compressed kv, indexer q
              AND indexer k; group j takes position j*ratio (freq row
              pos+1-ratio during decode); conjugate on the attention output
  compressor= kv=wkv(x) score=wgate(x) in fp32 (bf16 weights promoted);
              decode slot=pos%2; on odd pos latent = norm(bf16(sum(kv_state
              * softmax(score_state)))) published pre-rope
  indexer   = k=fp4rt(k_norm(wk(latent))) roped at pos-1, cached; q=fp4rt(
              wq_b_idx(qr)); weights=weights_proj(x)*(128^-0.5*32^-0.5);
              score = relu(q.k)*weights summed over LOCAL heads then
              all_reduce (this oracle emits the RANK-0 PARTIAL, no
              cross-rank summands exist in one pack); top-512 sorted,
              mapped +window, -1 unreachable
  attention = softmax over the window ring plus compressed positions with
              the per-head sink exp(sink - max) in the denominator

Modes:
  synth --out DIR              constrained-random TP8 stagepack carrying
                               layers 0/2/8 + expectations (CI; no weights)
  real --pack FILE --out DIR   expectations from a real rank pack; fixture
                               token ids must be < VOCAB // TP (rank 0 owns
                               only its vocab slice)
  --verify-checkpoint DIR      re-derive sample planes from the warm
                               checkpoint shards and assert byte equality

Layer 0 is fed the tiled embedding (its true input). Layers 2/8 are ALSO fed
the tiled embedding: their true input depends on the engram-gated layer 1,
which is fail-closed, so the 2/8 ladders are per-piece instruments on a
synthetic-but-identical input stream, not an end-to-end chain. o_a is
consumed in the reference-faithful TP8 layout (rank rows-slice x full 4096
cols); real packs staged before the o_a ruling fail closed with a REPACK
message.

Outputs: DIR/dsv41_expectations.bin + DIR/dsv41_manifest.json
"""
import argparse
import hashlib
import json
import math
import mmap
import os
import struct

import numpy as np

HIDDEN = 5120
VOCAB = 129280
HEADS = 64
HEAD_DIM = 512
ROPE = 64
Q_LORA = 1280
KV_LATENT = 512
O_LORA = 1024
O_GROUPS = 8
SINKS = 64
N_EXPERTS = 384
TOPK = 6
MOE_INTER = 2304
HC = 4
HC_ROWS = 24
HC_FLAT = HC * HIDDEN
SINKHORN_ITERS = 20
HC_EPS = 1e-6
NORM_EPS = 1e-20
ROPE_THETA = 10000.0
WINDOW = 128
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
TP = 8
RANK = 0
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
COMP_THETA = 160000.0
YARN_FACTOR = 16.0
YARN_ORIGINAL = 65536
YARN_BETA_FAST = 32
YARN_BETA_SLOW = 1
IDX_HEADS = 32
IDX_HEADS_LOCAL = IDX_HEADS // TP
IDX_DIM = 128
IDX_TOPK = 512
IDX_SCALE = (IDX_DIM ** -0.5) * (IDX_HEADS ** -0.5)

MAGIC = 0x31413444
FORMAT_VERSION = 1
HEADER_BYTES = 257
ENTRY_BYTES = 64
ALIGN = 256
GLOBAL_LAYER = 0xFFFFFFFF
PT_BF16, PT_F32, PT_PACKED = 1, 2, 4
COD_BF16, COD_NONE, COD_FP8, COD_MXFP4 = 1, 0, 5, 7
SE_NONE, SE_E8M0 = 0, 3

(K_EMBED, K_FNORM, K_HEAD, K_ATTN_NORM, K_FFN_NORM, K_QA, K_QB, K_KVA, K_QNORM,
 K_KVNORM, K_SINK, K_OA, K_OB) = range(13)
(K_IQB, K_IWK, K_IWP, K_IKN, K_CWKV, K_CWGATE, K_CNORM) = range(13, 20)
(K_HCAF, K_HCAB, K_HCAS, K_HCFF, K_HCFB, K_HCFS, K_ROUTER, K_RBIAS,
 K_W1, K_W2, K_W3, K_SW1, K_SW2, K_SW3) = (20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34)

LOCAL_HEADS = HEADS // TP
LOCAL_QB_ROWS = HEADS * HEAD_DIM // TP
LOCAL_OA_ROWS = O_GROUPS * O_LORA // TP
OA_FULL_COLS = HEADS * HEAD_DIM // O_GROUPS
LOCAL_OB_COLS = O_GROUPS * O_LORA // TP
LOCAL_EXPERTS = N_EXPERTS // TP
LOCAL_SINKS = SINKS // TP
LOCAL_EMBED = VOCAB // TP
W1_SHAPE = (MOE_INTER, HIDDEN // 2)
W2_SHAPE = (HIDDEN, MOE_INTER // 2)
W3_SHAPE = (MOE_INTER, HIDDEN // 2)
IDX_QB_SHAPE = (IDX_HEADS * IDX_DIM // TP, Q_LORA)
IDX_WP_SHAPE = (IDX_HEADS // TP, HIDDEN)
COMP_W_SHAPE = (KV_LATENT, HIDDEN)
LAYER_PLAN = {0: 0, 1: 0, 2: 2, 8: 2, 14: 2, 20: 1, 38: 0, 39: 0}
SYNTH_LAYERS = {0: 0, 1: 0, 2: 2, 8: 2}
TAG_POSITIONS = {0: 3, 1: 6, 2: 6, 8: 6, 14: 6, 20: 6, 38: 1, 39: 1}
ENGRAM_LAYERS = (1, 14)
ENGRAM_HEADS = 8
ENGRAM_ORDERS = 3
ENGRAM_COLS = ENGRAM_ORDERS * ENGRAM_HEADS
ENGRAM_HEAD_DIM = 256
ENGRAM_SCALE_COLS = 8
ENGRAM_VOCAB = 99092
ENGRAM_PAD_ID = 2
ENGRAM_MULT_SEED = 10007
ENGRAM_PRIME_START = 16000000 - 1
ENGRAM_SYNTH_PRIME_START = 4008
ENGRAM_WKV_ROWS = 25600
ENGRAM_WKV_COLS = ENGRAM_COLS * ENGRAM_HEAD_DIM
ENGRAM_GK_ROWS = 4
ENGRAM_NORM_EPS = 1e-20
ENGRAM_CLAMP = 1e-6
ENGRAM_PART_RANKS = 16

FP8_LUT = np.zeros(256, dtype=np.float32)
for _i in range(256):
    _s = -1.0 if _i & 0x80 else 1.0
    _e = (_i >> 3) & 0xF
    _m = _i & 0x7
    if _e == 0:
        _v = _m * 0.125 * 2.0 ** -6
    elif _e == 15 and _m == 7:
        _v = np.nan
    else:
        _v = (1.0 + _m * 0.125) * 2.0 ** (_e - 7)
    FP8_LUT[_i] = _s * _v

FP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                   dtype=np.float32)
FP4_TIES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
FP4_TIE_MAG = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], dtype=np.float32)


def bf16_u16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def e8m0_f32(u8):
    return np.ldexp(np.ones(u8.shape, dtype=np.float32), u8.astype(np.int32) - 127)


def rne_shift(v, s):
    q = v >> s
    rem = v & ((np.int32(1) << s) - 1)
    half = np.int32(1) << (s - 1)
    up = (rem > half) | ((rem == half) & ((q & 1) == 1))
    return np.where(s == 0, v, np.where(up, q + 1, q))


def fp8_code(x):
    x = np.clip(np.asarray(x, dtype=np.float32), -448.0, 448.0)
    u = x.view(np.uint32)
    sign = ((u >> 24) & 0x80).astype(np.uint8)
    bits = u & 0x7FFFFFFF
    exp = (bits >> 23).astype(np.int32)
    man = (bits & 0x7FFFFF).astype(np.int32)
    m3 = (man >> 20).astype(np.int32)
    rem = man & 0xFFFFF
    up = (rem > 0x80000) | ((rem == 0x80000) & ((m3 & 1) == 1))
    m3 = np.where(up, m3 + 1, m3)
    exp = np.where(m3 > 7, exp + 1, exp)
    m3 = np.where(m3 > 7, 0, m3)
    code = np.zeros(x.shape, dtype=np.int32)
    normal = (bits >= 0x3D000000) & (bits < 0x7F800000)
    code = np.where(normal, ((exp - 120) << 3) | m3, code)
    sub = (bits > 0) & (bits < 0x3D000000)
    sub_m = rne_shift(0x800000 | man, np.clip(141 - exp, 0, 31))
    code = np.where(sub, sub_m, code)
    return (sign | code.astype(np.uint8)).astype(np.uint8)


def act_quant(x_rows, group=32):
    rows = np.asarray(x_rows, dtype=np.float32).reshape(-1, x_rows.shape[-1])
    n = rows.shape[-1]
    blocks = rows.reshape(rows.shape[0], n // group, group)
    amax = np.maximum(np.abs(blocks).max(axis=2), 1e-4)
    t = (amax * (1.0 / 448.0)).astype(np.float32).view(np.uint32)
    log2_ceil = ((t >> 23).astype(np.int32) - 127 +
                 ((t & 0x7FFFFF) != 0).astype(np.int32))
    scale = np.ldexp(np.ones_like(amax), log2_ceil).astype(np.float32)
    q = np.clip(blocks / scale[:, :, None], -448.0, 448.0)
    vals = FP8_LUT[fp8_code(q)].reshape(rows.shape)
    return vals, scale


def act_dequant(x_bf16_flat, group=32):
    vals, scale = act_quant(bf16_f32(x_bf16_flat).reshape(1, -1), group)
    deq = (vals.reshape(-1, group) * scale.reshape(-1, 1)).astype(np.float32)
    return bf16_u16(deq.reshape(-1))


def fp4_nearest_mag(v):
    a = np.abs(np.asarray(v, dtype=np.float32))
    idx = np.searchsorted(FP4_TIES, a, side="right")
    mag = FP4_LUT[idx]
    tie = np.searchsorted(FP4_TIES, a, side="left")
    exact = (tie < FP4_TIES.size) & (FP4_TIES[tie.clip(0, FP4_TIES.size - 1)] == a)
    mag = np.where(exact, FP4_TIE_MAG[tie.clip(0, FP4_TIES.size - 1)], mag)
    return np.sign(v) * mag


def fp4_rt_e8m0(x_f32, group=32):
    rows = np.asarray(x_f32, dtype=np.float32).reshape(-1, x_f32.shape[-1])
    n = rows.shape[-1]
    blocks = rows.reshape(rows.shape[0], n // group, group)
    amax = np.maximum(np.abs(blocks).max(axis=2), 6.0 * 2.0 ** -126)
    t = (amax * (1.0 / 6.0)).astype(np.float32).view(np.uint32)
    log2_ceil = ((t >> 23).astype(np.int32) - 127 +
                 ((t & 0x7FFFFF) != 0).astype(np.int32))
    scale = np.ldexp(np.ones_like(amax), log2_ceil).astype(np.float32)
    q = np.clip(blocks / scale[:, :, None], -6.0, 6.0)
    mag = fp4_nearest_mag(q)
    deq = (mag * scale[:, :, None]).reshape(rows.shape)
    return bf16_u16(deq.reshape(-1))


def fp4_rt_e4m3(x_f32, group=16):
    rows = np.asarray(x_f32, dtype=np.float32).reshape(-1, x_f32.shape[-1])
    n = rows.shape[-1]
    blocks = rows.reshape(rows.shape[0], n // group, group)
    amax = np.maximum(np.abs(blocks).max(axis=2), 6.0 * 2.0 ** -9)
    scale = FP8_LUT[fp8_code(amax * (1.0 / 6.0))]
    q = np.clip(blocks / scale[:, :, None], -6.0, 6.0)
    mag = fp4_nearest_mag(q)
    deq = (mag * scale[:, :, None]).reshape(rows.shape)
    return bf16_u16(deq.reshape(-1))


def fp8_gemm_vector(x_bf16, w_u8, w_scale_u8, out_rows):
    x = bf16_f32(x_bf16).reshape(-1)
    k = x.shape[0]
    aq, sa = act_quant(x.reshape(1, k))
    aq = aq.reshape(k)
    sa = sa.reshape(k // 32)
    w = FP8_LUT[w_u8.reshape(out_rows, k)]
    ws = e8m0_f32(w_scale_u8.reshape(out_rows // 32, k // 32))
    acc = np.zeros(out_rows, dtype=np.float32)
    for kb in range(k // 32):
        part = (aq[None, kb * 32:(kb + 1) * 32] * w[:, kb * 32:(kb + 1) * 32]).sum(axis=1, dtype=np.float32)
        acc += part * sa[kb] * np.repeat(ws[:, kb], 32)
    return bf16_u16(acc)


def fp4_gemm_vector(x_bf16, w_u8, w_scale_u8, out_rows, k):
    w_u8 = w_u8.reshape(out_rows, k // 2)
    x = bf16_f32(x_bf16).reshape(-1)
    aq, sa = act_quant(x.reshape(1, k))
    aq = aq.reshape(k)
    sa = sa.reshape(k // 32)
    nib = np.empty((out_rows, k), dtype=np.uint8)
    nib[:, 0::2] = w_u8 & 0xF
    nib[:, 1::2] = w_u8 >> 4
    w = FP4_LUT[nib]
    ws = e8m0_f32(w_scale_u8.reshape(out_rows, k // 32))
    acc = np.zeros(out_rows, dtype=np.float32)
    for kb in range(k // 32):
        part = (aq[None, kb * 32:(kb + 1) * 32] * w[:, kb * 32:(kb + 1) * 32]).sum(axis=1, dtype=np.float32)
        acc += part * sa[kb] * ws[:, kb]
    return bf16_u16(acc)


def bf16_gemm_vector(x_bf16, w_bf16, out_rows):
    x = bf16_f32(x_bf16).reshape(-1).astype(np.float32)
    w = bf16_f32(w_bf16.reshape(out_rows, x.shape[0])).astype(np.float32)
    return bf16_u16(w @ x)


def rmsnorm(x_bf16, weight_bf16):
    x = bf16_f32(np.asarray(x_bf16).reshape(-1)).astype(np.float32)
    w = bf16_f32(np.asarray(weight_bf16).reshape(-1)).astype(np.float32)
    rstd = 1.0 / math.sqrt(float((x * x).mean(dtype=np.float32)) + NORM_EPS)
    return bf16_u16(w * x * rstd)


def _yarn_band(base):
    def corrected(rotations):
        return ROPE * math.log(YARN_ORIGINAL / (rotations * 2.0 * math.pi)) / \
            (2.0 * math.log(base))
    low = max(math.floor(corrected(YARN_BETA_FAST)), 0)
    high = min(math.ceil(corrected(YARN_BETA_SLOW)), ROPE - 1)
    ramp = (np.arange(ROPE // 2, dtype=np.float32) - np.float32(low)) / \
        np.float32(max(high - low, 1e-3))
    ramp = np.clip(ramp, 0.0, 1.0).astype(np.float32)
    return 1.0 - ramp


def rope_freqs(last_pos):
    inv = 1.0 / (np.float32(ROPE_THETA) ** (np.arange(0, ROPE, 2, dtype=np.float32) /
                                            np.float32(ROPE)))
    return np.outer(np.arange(last_pos + 1, dtype=np.float32), inv).astype(np.float32)


def rope_freqs_yarn(last_pos, base):
    inv = 1.0 / (np.float32(base) ** (np.arange(0, ROPE, 2, dtype=np.float32) /
                                      np.float32(ROPE)))
    smooth = _yarn_band(base)
    inv = (inv / np.float32(YARN_FACTOR) * (1.0 - smooth) + inv * smooth).astype(np.float32)
    return np.outer(np.arange(last_pos + 1, dtype=np.float32), inv).astype(np.float32)


def apply_rope_tail(vec_bf16, freq_row, inverse=False):
    v = bf16_f32(vec_bf16).copy()
    tail = v[-ROPE:].reshape(-1, 2)
    angle = -freq_row if inverse else freq_row
    c, s = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    re = tail[:, 0] * c - tail[:, 1] * s
    im = tail[:, 0] * s + tail[:, 1] * c
    tail[:, 0], tail[:, 1] = re, im
    return bf16_u16(v)


def apply_rope_rows(rows_bf16, freq_row, inverse=False):
    out = np.empty_like(rows_bf16)
    for r in range(rows_bf16.shape[0]):
        out[r] = apply_rope_tail(rows_bf16[r], freq_row, inverse)
    return out


def hc_mixes(stream_bf16, fn_f32, scale3, base24):
    x = bf16_f32(stream_bf16.reshape(-1)).astype(np.float32)
    rstd = 1.0 / math.sqrt(float((x * x).mean(dtype=np.float32)) + NORM_EPS)
    mixes = (fn_f32 @ x * rstd).astype(np.float32)
    pre = 1.0 / (1.0 + np.exp(-(mixes[:HC] * scale3[0] + base24[:HC]))) + HC_EPS
    post = 2.0 / (1.0 + np.exp(-(mixes[HC:2 * HC] * scale3[1] + base24[HC:2 * HC])))
    comb = (mixes[2 * HC:].reshape(HC, HC) * scale3[2] + base24[2 * HC:].reshape(HC, HC)).astype(np.float32)
    comb = np.exp(comb - comb.max(axis=1, keepdims=True))
    comb = comb / comb.sum(axis=1, keepdims=True) + HC_EPS
    comb = comb / (comb.sum(axis=0, keepdims=True) + HC_EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(axis=1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(axis=0, keepdims=True) + HC_EPS)
    return mixes, pre.astype(np.float32), post.astype(np.float32), comb.astype(np.float32)


def hc_pre(stream_flat_bf16, pre):
    x = bf16_f32(stream_flat_bf16.reshape(HC, HIDDEN)).astype(np.float32)
    return bf16_u16((pre[:, None] * x).sum(axis=0))


def hc_post(x_bf16, residual_bf16, post, comb):
    xv = bf16_f32(x_bf16.reshape(-1)).astype(np.float32)
    r = bf16_f32(residual_bf16.reshape(HC, HIDDEN)).astype(np.float32)
    y = post[:, None] * xv[None, :] + np.einsum("sr,sd->rd", comb, r)
    return bf16_u16(y.reshape(-1))


def stable_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def softplus_sqrt(scores):
    return np.sqrt(np.log1p(np.exp(np.clip(scores, -30.0, 30.0))).astype(np.float32))


def gate_topk(x_bf16, router_bf16, bias_f32):
    x = bf16_f32(x_bf16.reshape(-1)).astype(np.float32)
    w = bf16_f32(router_bf16.reshape(N_EXPERTS, HIDDEN)).astype(np.float32)
    scores = (w @ x).astype(np.float32)
    ss = softplus_sqrt(scores)
    order = np.argsort(-(ss + bias_f32), kind="stable")[:TOPK]
    weights = ss[order].copy()
    weights = weights / (weights.sum(dtype=np.float32) + 1e-20) * ROUTE_SCALE
    return scores, order.astype(np.int32), weights.astype(np.float32)


def expert_mlp(x_bf16, w1_u8, w1s, w3_u8, w3s, w2_u8, w2s, weight):
    gate = bf16_f32(fp4_gemm_vector(x_bf16, w1_u8, w1s, MOE_INTER, HIDDEN)).astype(np.float32)
    up = bf16_f32(fp4_gemm_vector(x_bf16, w3_u8, w3s, MOE_INTER, HIDDEN)).astype(np.float32)
    gate = np.minimum(gate, SWIGLU_LIMIT)
    up = np.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    act = bf16_u16((gate * stable_sigmoid(gate) * up * weight).astype(np.float32))
    return fp4_gemm_vector(act, w2_u8, w2s, HIDDEN, MOE_INTER)


def shared_mlp(x_bf16, sw1, sw3, sw2):
    gate = bf16_f32(fp8_gemm_vector(x_bf16, sw1[0], sw1[1], MOE_INTER)).astype(np.float32)
    up = bf16_f32(fp8_gemm_vector(x_bf16, sw3[0], sw3[1], MOE_INTER)).astype(np.float32)
    gate = np.minimum(gate, SWIGLU_LIMIT)
    up = np.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    act = bf16_u16((gate * stable_sigmoid(gate) * up).astype(np.float32))
    return fp8_gemm_vector(act, sw2[0], sw2[1], HIDDEN)


def window_topk_idxs(start_pos):
    if start_pos == 0:
        return np.array([0], dtype=np.int32)
    oldest = start_pos % WINDOW + 1
    idxs = np.concatenate([np.arange(oldest, WINDOW), np.arange(oldest)])
    return np.where(idxs > start_pos, -1, idxs).astype(np.int32)


def sparse_attn(q_bf16, cache, idxs, sinks, scale):
    o = np.zeros((LOCAL_HEADS, HEAD_DIM), dtype=np.uint16)
    valid = [int(i) for i in idxs if i >= 0]
    for h in range(LOCAL_HEADS):
        qh = bf16_f32(q_bf16[h]).astype(np.float32)
        scores = np.empty(len(valid), dtype=np.float32)
        for j, idx in enumerate(valid):
            scores[j] = float(np.sum(qh * bf16_f32(cache[idx]).astype(np.float32),
                                     dtype=np.float32)) * scale
        top = scores.max() if len(valid) else -1e30
        ex = np.exp(scores - top).astype(np.float32) if len(valid) else \
            np.zeros(0, dtype=np.float32)
        sink_term = math.exp(float(sinks[h]) - float(top)) if top > -1e29 \
            else math.inf
        denom = (ex.sum(dtype=np.float32) + sink_term).astype(np.float32)
        acc = np.zeros(HEAD_DIM, dtype=np.float32)
        for j, idx in enumerate(valid):
            acc += ex[j] * bf16_f32(cache[idx]).astype(np.float32)
        o[h] = bf16_u16(acc / denom)
    return o


class Compressor:
    def __init__(self, ratio, wkv_bf16, wgate_bf16, norm_bf16):
        self.ratio = ratio
        self.wkv = bf16_f32(wkv_bf16.reshape(COMP_W_SHAPE)).astype(np.float32)
        self.wgate = bf16_f32(wgate_bf16.reshape(COMP_W_SHAPE)).astype(np.float32) \
            if wgate_bf16 is not None else None
        self.norm = norm_bf16
        self.kv_state = np.zeros((ratio, KV_LATENT), dtype=np.float32)
        self.score_state = np.zeros((ratio, KV_LATENT), dtype=np.float32) \
            if ratio > 1 else None

    def step(self, x_bf16, pos, pieces, tag):
        x = bf16_f32(x_bf16.reshape(-1)).astype(np.float32)
        kv = (self.wkv @ x).astype(np.float32)
        if self.ratio == 1:
            if tag:
                pieces.add(f"{tag}.c_kv_proj", kv, 2e-3, 1e-5)
            return rmsnorm(bf16_u16(kv), self.norm)
        score = (self.wgate @ x).astype(np.float32)
        slot = pos % self.ratio
        self.kv_state[slot] = kv
        self.score_state[slot] = score
        if tag:
            pieces.add(f"{tag}.c_kv_proj", kv, 2e-3, 1e-5)
            pieces.add(f"{tag}.c_gate_score", score, 2e-3, 1e-5)
            pieces.add(f"{tag}.c_state_kv", self.kv_state.reshape(-1), 2e-3, 1e-5)
            pieces.add(f"{tag}.c_state_score", self.score_state.reshape(-1), 2e-3, 1e-5)
        if (pos + 1) % self.ratio != 0:
            return None
        mx = self.score_state.max(axis=0)
        ex = np.exp(self.score_state - mx[None, :]).astype(np.float32)
        p = (ex / ex.sum(axis=0, keepdims=True)).astype(np.float32)
        pooled = (self.kv_state * p).sum(axis=0).astype(np.float32)
        latent = rmsnorm(bf16_u16(pooled), self.norm)
        if tag:
            pieces.add(f"{tag}.c_latent", bf16_f32(latent), 2e-3, 1e-5)
        return latent


class Indexer:
    def __init__(self, ratio, iqb, iqbs, iwk, iwp, ikn):
        self.ratio = ratio
        self.iqb = (iqb, iqbs)
        self.wk = iwk
        self.iwp = iwp
        self.ikn = ikn
        self.k_cache = {}

    def publish_k(self, latent_bf16, pos, freqs, pieces, tag):
        k = rmsnorm(bf16_gemm_vector(latent_bf16, self.wk, IDX_DIM), self.ikn)
        k = apply_rope_tail(k, freqs[pos + 1 - self.ratio])
        k = fp4_rt_e8m0(bf16_f32(k).reshape(1, -1)).reshape(IDX_DIM)
        self.k_cache[pos // self.ratio] = k
        if tag:
            pieces.add(f"{tag}.idx_k", bf16_f32(k), 0.25, 1e-6)

    def score(self, x_bf16, qr, pos, freqs, pieces, tag):
        q = fp8_gemm_vector(qr, self.iqb[0], self.iqb[1],
                            IDX_HEADS_LOCAL * IDX_DIM).reshape(IDX_HEADS_LOCAL, IDX_DIM)
        q = apply_rope_rows(q, freqs[pos])
        q = fp4_rt_e8m0(bf16_f32(q))
        w = bf16_gemm_vector(x_bf16, self.iwp, IDX_HEADS_LOCAL)
        w = bf16_u16(bf16_f32(w).astype(np.float32) * np.float32(IDX_SCALE))
        n = (pos + 1) // self.ratio
        keys = np.stack([self.k_cache[j] for j in range(n)]) if n else \
            np.zeros((0, IDX_DIM), dtype=np.uint16)
        qf = bf16_f32(q.reshape(-1)).astype(np.float32).reshape(IDX_HEADS_LOCAL, IDX_DIM)
        kf = bf16_f32(keys.reshape(-1)).astype(np.float32).reshape(n, IDX_DIM)
        dots = bf16_u16(np.einsum("hd,td->th", qf, kf).reshape(-1)) \
            if n else np.zeros(0, dtype=np.uint16)
        dots = bf16_f32(dots).reshape(n, IDX_HEADS_LOCAL) if n else \
            np.zeros((0, IDX_HEADS_LOCAL), dtype=np.float32)
        dots = np.maximum(dots, 0.0)
        wf = bf16_f32(w).astype(np.float32)
        part = bf16_u16((dots * wf[None, :]).reshape(-1))
        part = bf16_f32(part).reshape(n, IDX_HEADS_LOCAL) if n else \
            np.zeros((0, IDX_HEADS_LOCAL), dtype=np.float32)
        score = bf16_u16(part.sum(axis=1, dtype=np.float32)) if n else \
            np.zeros(0, dtype=np.uint16)
        sf = bf16_f32(score)
        top = min(IDX_TOPK, n)
        order = np.argsort(-sf, kind="stable")[:top]
        mapped = np.sort(order).astype(np.int32) + WINDOW
        if tag:
            pieces.add(f"{tag}.idx_q", bf16_f32(q.reshape(-1)), 0.25, 1e-6)
            pieces.add(f"{tag}.idx_w", bf16_f32(w), 2e-3, 1e-6)
            if n:
                pieces.add(f"{tag}.idx_score", sf, 2e-2, 1e-3)
                pieces.add(f"{tag}.idx_topk", mapped.astype(np.float32), 0.0, 0.0)
        return mapped


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def next_prime(start, seen):
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def engram_primes(synth):
    start = ENGRAM_SYNTH_PRIME_START if synth else ENGRAM_PRIME_START
    primes, seen = [], set()
    for _ in ENGRAM_LAYERS:
        per_order = []
        for _ in range(ENGRAM_ORDERS):
            sizes = []
            for _ in range(ENGRAM_HEADS):
                start = next_prime(start, seen)
                seen.add(start)
                sizes.append(start)
            per_order.append(sizes)
        primes.append(per_order)
    return np.array(primes, dtype=np.int64)


def engram_multipliers():
    bound = max(1, ((2 ** 63 - 1) // ENGRAM_VOCAB) // 2)
    rows = []
    for layer_id in ENGRAM_LAYERS:
        generator = np.random.default_rng(ENGRAM_MULT_SEED * layer_id)
        rows.append(generator.integers(0, bound, size=4, dtype=np.int64) * 2 + 1)
    return np.array(rows, dtype=np.int64)


def engram_offsets(primes_per_layer):
    flat = primes_per_layer.reshape(-1)
    return np.concatenate([[0], np.cumsum(flat)[:-1]]).astype(np.int64)


class EngramTable:
    def __init__(self, weights, scales):
        self.weights = weights
        self.scales = scales

    def row(self, row_id):
        return self.weights[row_id], self.scales[row_id]


class CheckpointEngramTable:
    def __init__(self, ckpt_dir, layer, entries):
        self.dir = ckpt_dir
        self.weight_name = f"layers.{layer}.engram.embed.weight"
        self.scale_name = f"layers.{layer}.engram.embed.scale"
        self.entries = entries
        self.views = {n: self._open(n) for n in (self.weight_name, self.scale_name)}

    def _open(self, name):
        with open(os.path.join(self.dir, "model.safetensors.index.json")) as f:
            shard = json.load(f)["weight_map"][name]
        path = os.path.join(self.dir, shard)
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        begin, _ = header[name]["data_offsets"]
        handle = open(path, "rb")
        base = 8 + n + begin
        handle.seek(base)
        return handle, base

    def row(self, row_id):
        if row_id >= self.entries:
            raise SystemExit(f"engram id {row_id} outside table entries {self.entries}")
        weight_handle, weight_base = self.views[self.weight_name]
        scale_handle, scale_base = self.views[self.scale_name]
        weight_handle.seek(weight_base + row_id * ENGRAM_HEAD_DIM)
        scale_handle.seek(scale_base + row_id * ENGRAM_SCALE_COLS)
        w = np.frombuffer(weight_handle.read(ENGRAM_HEAD_DIM), dtype=np.uint8)
        s = np.frombuffer(scale_handle.read(ENGRAM_SCALE_COLS), dtype=np.uint8)
        if len(w) != ENGRAM_HEAD_DIM or len(s) != ENGRAM_SCALE_COLS:
            raise SystemExit("short engram row read")
        return w, s


class Engram:
    def __init__(self, layer_id, hidx, multipliers, primes, weights, table):
        self.layer_id = layer_id
        self.hidx = hidx
        self.mult = multipliers[hidx]
        self.primes = primes[hidx]
        self.offsets = engram_offsets(primes[hidx])
        self.entries = int(primes[hidx].sum())
        self.part = (self.entries + ENGRAM_PART_RANKS - 1) // ENGRAM_PART_RANKS
        self.wkv_w, self.wkv_s = weights[0], weights[1]
        self.qw, self.kw = weights[2], weights[3]
        self.table = table
        self.used_ids = {}

    def ids_at(self, pos, compressed):
        tokens = [ENGRAM_PAD_ID if pos < shift
                  else compressed[(pos - shift) % len(compressed)]
                  for shift in range(4)]
        rolling = np.int64(tokens[0]) * self.mult[0]
        ids = np.zeros(ENGRAM_COLS, dtype=np.int64)
        for i in range(1, 4):
            rolling = rolling ^ (np.int64(tokens[i]) * self.mult[i])
            base = (i - 1) * ENGRAM_HEADS
            ids[base:base + ENGRAM_HEADS] = \
                rolling % self.primes[i - 1] + self.offsets[base:base + ENGRAM_HEADS]
        return ids

    def forward(self, stream_flat, pos, compressed, pieces, tag):
        ids = self.ids_at(pos, compressed)
        self.used_ids[pos] = ids
        rows = np.zeros((ENGRAM_COLS, ENGRAM_HEAD_DIM), dtype=np.uint8)
        scales = np.zeros((ENGRAM_COLS, 8), dtype=np.uint8)
        for j in range(ENGRAM_COLS):
            rows[j], scales[j] = self.table.row(int(ids[j]))
        values = bf16_u16((FP8_LUT[rows].reshape(ENGRAM_COLS, 8, 32)
                           * e8m0_f32(scales)[:, :, None]).reshape(-1))
        kv = fp8_gemm_vector(values, self.wkv_w, self.wkv_s, ENGRAM_WKV_ROWS)
        key = bf16_f32(kv[:ENGRAM_GK_ROWS * HIDDEN]).reshape(ENGRAM_GK_ROWS, HIDDEN)
        value = bf16_f32(kv[ENGRAM_GK_ROWS * HIDDEN:])
        w = bf16_f32(self.qw) * bf16_f32(self.kw)
        h = bf16_f32(stream_flat).reshape(ENGRAM_GK_ROWS, HIDDEN).astype(np.float32)
        rstd = (1.0 / np.sqrt(np.square(h).mean(-1) + ENGRAM_NORM_EPS)
                * (1.0 / np.sqrt(np.square(key).mean(-1) + ENGRAM_NORM_EPS)))
        dot = (h * w * key).sum(-1) * rstd * HIDDEN ** -0.5
        magnitude = np.clip(np.abs(dot), ENGRAM_CLAMP, None)
        gate = stable_sigmoid(np.copysign(np.sqrt(magnitude), dot))
        out = (h + gate[:, None] * value[None, :]).astype(np.float32)
        if tag:
            owner = ids // self.part
            local = ids - owner * self.part
            access = np.concatenate([owner.astype(np.float32),
                                     (local >> 24).astype(np.float32),
                                     (local & 0xFFFFFF).astype(np.float32)])
            pieces.add(f"{tag}.eg_access", access, 0.0, 0.0)
            pieces.add(f"{tag}.eg_rows", bf16_f32(values), 3e-2, 1e-2)
            pieces.add(f"{tag}.eg_value", value, 2e-2, 1e-5)
            pieces.add(f"{tag}.eg_gate", gate, 2e-3, 1e-6)
            pieces.add(f"{tag}.eg_out", out.reshape(-1), 2e-2, 1e-5)
        return bf16_u16(out.reshape(-1))

    def fixture_records(self):
        for pos in sorted(self.used_ids):
            for row_id in self.used_ids[pos]:
                yield int(row_id)


def write_engram_fixture(path, engrams):
    seen = {e.layer_id: set() for e in engrams}
    for e in engrams:
        for row_id in e.fixture_records():
            seen[e.layer_id].add(row_id)
    spec = bytearray()
    spec += struct.pack("<4sII", b"EGWF", 1, len(engrams))
    for e in engrams:
        spec += struct.pack("<IIQQ", e.layer_id, 0, e.entries, e.part)
        spec += e.mult.astype("<i8").tobytes()
        spec += e.primes.reshape(-1).astype("<i8").tobytes()
        spec += e.offsets.astype("<i8").tobytes()
    records = bytearray()
    count = 0
    for e in engrams:
        for row_id in sorted(seen[e.layer_id]):
            w, s = e.table.row(row_id)
            records += struct.pack("<IIQQ", 0, e.layer_id, 0, row_id)
            records += bytes(w) + bytes(s)
            count += 1
        records += struct.pack("<IIQQ", 1, e.layer_id, 0, 0)
        records += e.wkv_w.tobytes() + e.wkv_s.tobytes()
        count += 1
        for kind, plane in ((2, e.qw), (3, e.kw)):
            records += struct.pack("<IIQQ", kind, e.layer_id, 0, 0)
            records += plane.tobytes()
            count += 1
    with open(path, "wb") as f:
        f.write(bytes(spec))
        f.write(struct.pack("<I", count))
        f.write(bytes(records))
    return count


def synth_engram_tensors(rng, primes):
    def fp8w(shape):
        codes = rng.integers(8, 40, size=shape).astype(np.uint8)
        scales = rng.integers(118, 123, size=(shape[0] // 32, shape[1] // 32)).astype(np.uint8)
        return codes, scales

    tables = {}
    for hidx, layer_id in enumerate(ENGRAM_LAYERS):
        if layer_id not in SYNTH_LAYERS:
            continue
        entries = int(primes[hidx].sum())
        codes = rng.integers(0, 256, size=(entries, ENGRAM_HEAD_DIM)).astype(np.uint8)
        codes[codes == 127] = 126
        codes[codes == 255] = 254
        scales = rng.integers(118, 123, size=(entries, 8)).astype(np.uint8)
        wkv_codes, wkv_scales = fp8w((ENGRAM_WKV_ROWS, ENGRAM_WKV_COLS))
        tables[layer_id] = {
            "table": EngramTable(codes, scales),
            "wkv": (wkv_codes.reshape(-1), wkv_scales.reshape(-1)),
            "qw": bf16_u16((rng.standard_normal((ENGRAM_GK_ROWS, HIDDEN)) * 0.1)
                           .astype(np.float32)),
            "kw": bf16_u16((rng.standard_normal((ENGRAM_GK_ROWS, HIDDEN)) * 0.1)
                           .astype(np.float32)),
        }
    return tables


class PackWriter:
    def __init__(self, path):
        self.path = path
        self.entries = []
        self.blobs = bytearray()

    def _append(self, blob):
        while len(self.blobs) % ALIGN:
            self.blobs.append(0)
        off = len(self.blobs)
        self.blobs += blob
        return off, len(blob)

    def add(self, kind, layer, payload_type, codec, scale_enc, groups, rows, cols,
            payload, scale):
        p_off, _ = self._append(payload)
        s_off, s_bytes = self._append(scale) if scale else (0, 0)
        self.entries.append([kind, layer, payload_type, codec, scale_enc, groups,
                             rows, cols, p_off, len(payload), s_off, s_bytes])

    def write(self, rank):
        count = len(self.entries)
        directory_offset = (HEADER_BYTES + ALIGN - 1) & ~(ALIGN - 1)
        payload_base = (directory_offset + count * ENTRY_BYTES + ALIGN - 1) & ~(ALIGN - 1)
        for entry in self.entries:
            entry[8] += payload_base
            if entry[10]:
                entry[10] += payload_base
        blob = bytearray(self.blobs)
        file_bytes = payload_base + len(blob)
        header = bytearray(HEADER_BYTES)
        struct.pack_into("<20I2Q", header, 0,
                         MAGIC, FORMAT_VERSION, HEADER_BYTES, ENTRY_BYTES, 1, 0,
                         count, 1, 0, LAYER, 40, 40, HIDDEN, VOCAB, N_EXPERTS,
                         COD_FP8, COD_MXFP4, TP, rank, 0,
                         directory_offset, file_bytes)
        header[96:96 + len(REVISION)] = REVISION.encode()
        directory = b"".join(struct.pack("<8I4Q", *e) for e in self.entries)
        with open(self.path, "wb") as f:
            f.write(header)
            f.write(b"\x00" * (directory_offset - HEADER_BYTES))
            f.write(directory)
            f.write(b"\x00" * (payload_base - directory_offset - len(directory)))
            f.write(bytes(blob))
        return file_bytes


LAYER = 0


def add_dense(writer, kind, layer, payload, codec, scale=b"", groups=1):
    rows, cols = payload.shape
    if codec == COD_BF16:
        writer.add(kind, layer, PT_BF16, COD_BF16, SE_NONE, groups, rows, cols,
                   payload.tobytes(), b"")
    elif codec == COD_NONE:
        writer.add(kind, layer, PT_F32, COD_NONE, SE_NONE, groups, rows, cols,
                   payload.astype(np.float32).tobytes(), b"")
    else:
        writer.add(kind, layer, PT_PACKED, COD_FP8, SE_E8M0, groups, rows, cols,
                   payload.tobytes(), scale.tobytes())


def synth_tensors(rng):
    def bf16n(shape, scale=0.5):
        return bf16_u16((rng.standard_normal(shape) * scale).astype(np.float32))

    def f32n(shape, scale=0.5):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def fp8w(shape):
        codes = rng.integers(8, 40, size=shape).astype(np.uint8)
        scales = rng.integers(118, 123, size=(shape[0] // 32, shape[1] // 32)).astype(np.uint8)
        return codes, scales

    def fp4w(shape, scale_rows):
        bytes_ = rng.integers(0, 256, size=shape).astype(np.uint8)
        scales = rng.integers(116, 124, size=(scale_rows, shape[1] * 2 // 32)).astype(np.uint8)
        return bytes_, scales

    t = {"globals": {}, "layers": {}}
    t["globals"][K_EMBED] = bf16n((LOCAL_EMBED, HIDDEN), 0.1)
    t["globals"][K_FNORM] = bf16n((1, HIDDEN), 0.1)
    t["globals"][K_HEAD] = bf16n((LOCAL_EMBED, HIDDEN))
    for layer, ratio in SYNTH_LAYERS.items():
        tl = {}
        tl[K_ATTN_NORM] = bf16n((1, HIDDEN), 0.1)
        tl[K_FFN_NORM] = bf16n((1, HIDDEN), 0.1)
        tl[K_QA] = fp8w((Q_LORA, HIDDEN))
        tl[K_QB] = fp8w((LOCAL_QB_ROWS, Q_LORA))
        tl[K_KVA] = fp8w((KV_LATENT, HIDDEN))
        tl[K_QNORM] = bf16n((1, Q_LORA), 0.1)
        tl[K_KVNORM] = bf16n((1, KV_LATENT), 0.1)
        tl[K_SINK] = f32n((1, LOCAL_SINKS))
        tl[K_OA] = fp8w((LOCAL_OA_ROWS, OA_FULL_COLS))
        tl[K_OB] = fp8w((HIDDEN, LOCAL_OB_COLS))
        tl[K_HCAF] = f32n((HC_ROWS, HC_FLAT), 0.02)
        tl[K_HCAB] = f32n(HC_ROWS, 0.3)
        tl[K_HCAS] = np.abs(f32n(3, 0.5)) + 0.25
        tl[K_HCFF] = f32n((HC_ROWS, HC_FLAT), 0.02)
        tl[K_HCFB] = f32n(HC_ROWS, 0.3)
        tl[K_HCFS] = np.abs(f32n(3, 0.5)) + 0.25
        tl[K_ROUTER] = bf16n((N_EXPERTS, HIDDEN), 0.02)
        tl[K_RBIAS] = f32n(N_EXPERTS, 0.05)
        for kind, shape, srows in ((K_W1, W1_SHAPE, MOE_INTER), (K_W2, W2_SHAPE, HIDDEN),
                                   (K_W3, W3_SHAPE, MOE_INTER)):
            planes = [fp4w(shape, srows) for _ in range(LOCAL_EXPERTS)]
            tl[kind] = (np.concatenate([p[0].reshape(-1) for p in planes]),
                        np.concatenate([p[1].reshape(-1) for p in planes]))
        tl[K_SW1] = fp8w((MOE_INTER, HIDDEN))
        tl[K_SW2] = fp8w((HIDDEN, MOE_INTER))
        tl[K_SW3] = fp8w((MOE_INTER, HIDDEN))
        if ratio > 0:
            tl[K_IQB] = fp8w(IDX_QB_SHAPE)
            tl[K_IWK] = bf16n((IDX_DIM, KV_LATENT))
            tl[K_IWP] = bf16n(IDX_WP_SHAPE)
            tl[K_IKN] = bf16n((1, IDX_DIM), 0.1)
            tl[K_CWKV] = bf16n(COMP_W_SHAPE, 0.02)
            if ratio > 1:
                tl[K_CWGATE] = bf16n(COMP_W_SHAPE, 0.02)
            tl[K_CNORM] = bf16n((1, KV_LATENT), 0.1)
        t["layers"][layer] = tl
    t["engram_meta"] = {"primes": engram_primes(synth=True),
                        "multipliers": engram_multipliers()}
    t["engram"] = synth_engram_tensors(rng, t["engram_meta"]["primes"])
    return t


def synth_write_pack(path, t):
    writer = PackWriter(path)
    tg = t["globals"]
    for kind, payload, codec in ((K_EMBED, tg[K_EMBED], COD_BF16),
                                 (K_FNORM, tg[K_FNORM], COD_BF16),
                                 (K_HEAD, tg[K_HEAD], COD_BF16)):
        rows, cols = payload.shape
        writer.add(kind, GLOBAL_LAYER, PT_BF16, COD_BF16, SE_NONE, 1, rows, cols,
                   payload.tobytes(), b"")
    for layer, ratio in SYNTH_LAYERS.items():
        tl = t["layers"][layer]
        for kind, payload, codec in ((K_ATTN_NORM, tl[K_ATTN_NORM], COD_BF16),
                                     (K_FFN_NORM, tl[K_FFN_NORM], COD_BF16),
                                     (K_QNORM, tl[K_QNORM], COD_BF16),
                                     (K_KVNORM, tl[K_KVNORM], COD_BF16),
                                     (K_ROUTER, tl[K_ROUTER], COD_BF16)):
            rows, cols = payload.shape
            writer.add(kind, layer, PT_BF16, COD_BF16, SE_NONE, 1, rows, cols,
                       payload.tobytes(), b"")
        for kind, payload in ((K_SINK, tl[K_SINK]), (K_HCAF, tl[K_HCAF]),
                              (K_HCAB, tl[K_HCAB]), (K_HCAS, tl[K_HCAS]),
                              (K_HCFF, tl[K_HCFF]), (K_HCFB, tl[K_HCFB]),
                              (K_HCFS, tl[K_HCFS]), (K_RBIAS, tl[K_RBIAS])):
            flat = payload.reshape(1, -1)
            rows, cols = flat.shape
            writer.add(kind, layer, PT_F32, COD_NONE, SE_NONE, 1, rows, cols,
                       flat.astype(np.float32).tobytes(), b"")
        for kind, payload in ((K_QA, tl[K_QA]), (K_QB, tl[K_QB]), (K_KVA, tl[K_KVA]),
                              (K_OA, tl[K_OA]), (K_OB, tl[K_OB]), (K_SW1, tl[K_SW1]),
                              (K_SW2, tl[K_SW2]), (K_SW3, tl[K_SW3])):
            code, scales = payload
            rows, cols = code.shape
            writer.add(kind, layer, PT_PACKED, COD_FP8, SE_E8M0, 1, rows, cols,
                       code.tobytes(), scales.tobytes())
        for kind in (K_W1, K_W2, K_W3):
            code, scales = tl[kind]
            rows = MOE_INTER if kind != K_W2 else HIDDEN
            cols = code.size // LOCAL_EXPERTS // rows
            writer.add(kind, layer, PT_PACKED, COD_MXFP4, SE_E8M0, LOCAL_EXPERTS, rows,
                       cols, code.tobytes(), scales.tobytes())
        if ratio > 0:
            code, scales = tl[K_IQB]
            rows, cols = code.shape
            writer.add(K_IQB, layer, PT_PACKED, COD_FP8, SE_E8M0, 1, rows, cols,
                       code.tobytes(), scales.tobytes())
            for kind in (K_IWK, K_IWP, K_IKN, K_CWKV, K_CNORM):
                rows, cols = tl[kind].shape
                writer.add(kind, layer, PT_BF16, COD_BF16, SE_NONE, 1, rows, cols,
                           tl[kind].tobytes(), b"")
            if ratio > 1:
                rows, cols = tl[K_CWGATE].shape
                writer.add(K_CWGATE, layer, PT_BF16, COD_BF16, SE_NONE, 1, rows, cols,
                           tl[K_CWGATE].tobytes(), b"")
    return writer.write(RANK)


class PackReader:
    def __init__(self, path):
        self.file = open(path, "rb")
        header = self.file.read(HEADER_BYTES)
        fields = struct.unpack("<20I2Q65s32s32s32s", header)
        if fields[0] != MAGIC or fields[1] != FORMAT_VERSION:
            raise SystemExit(f"bad stagepack magic/version in {path}")
        self.count = fields[6]
        self.tp = fields[17]
        self.rank = fields[18]
        self.entries = {}
        self.file.seek(fields[20])
        raw = self.file.read(self.count * ENTRY_BYTES)
        for i in range(self.count):
            e = struct.unpack_from("<8I4Q", raw, i * ENTRY_BYTES)
            layer = GLOBAL_LAYER if e[0] <= K_HEAD else e[1]
            self.entries[(e[0], layer)] = e
        self.m = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def view(self, kind, layer=LAYER, expert=None):
        e = self.entries[(kind, GLOBAL_LAYER if kind <= K_HEAD else layer)]
        groups, p_off, p_bytes, s_off, s_bytes = e[5], e[8], e[9], e[10], e[11]
        if expert is None:
            payload = np.frombuffer(self.m, dtype=np.uint8, count=p_bytes, offset=p_off)
            scale = np.frombuffer(self.m, dtype=np.uint8, count=s_bytes, offset=s_off) \
                if s_bytes else np.zeros(0, dtype=np.uint8)
            return payload, scale
        per = p_bytes // groups
        sper = s_bytes // groups
        payload = np.frombuffer(self.m, dtype=np.uint8, count=per, offset=p_off + expert * per)
        scale = np.frombuffer(self.m, dtype=np.uint8, count=sper, offset=s_off + expert * sper)
        return payload, scale

    def bf16(self, kind, layer=LAYER):
        return np.frombuffer(self.view(kind, layer)[0], dtype=np.uint16)

    def f32(self, kind, layer=LAYER):
        return np.frombuffer(self.view(kind, layer)[0], dtype=np.float32)

    def fp8(self, kind, layer=LAYER, scale_rows=None, scale_cols=None):
        p, s = self.view(kind, layer)
        if scale_rows is not None:
            s = s.reshape(scale_rows, scale_cols)
        return p, s

    def experts(self, kind, layer=LAYER):
        def one(g):
            return self.view(kind, layer, g)
        return one


class Expectations:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.bin = bytearray()
        self.pieces = []

    def add(self, name, array, tol_rel, tol_abs):
        a = np.ascontiguousarray(array, dtype=np.float32)
        offset = len(self.bin)
        self.bin += a.tobytes()
        self.pieces.append({"name": name, "dtype": "f32", "shape": list(a.shape),
                            "offset": offset, "count": int(a.size),
                            "tol_rel": tol_rel, "tol_abs": tol_abs,
                            "tol_scale": float(np.abs(a).max()) if a.size else 0.0})

    def write(self, meta):
        os.makedirs(self.out_dir, exist_ok=True)
        bin_path = os.path.join(self.out_dir, "dsv41_expectations.bin")
        with open(bin_path, "wb") as f:
            f.write(bytes(self.bin))
        meta["pieces"] = self.pieces
        meta["expectations_sha256"] = hashlib.sha256(bytes(self.bin)).hexdigest()
        with open(os.path.join(self.out_dir, "dsv41_manifest.json"), "w") as f:
            json.dump(meta, f, indent=1)
        table_path = os.path.join(self.out_dir, "dsv41_piece_table.bin")
        tokens = meta["tokens"] + [0] * (16 - len(meta["tokens"]))
        header = struct.pack("<4s7I16iffI", b"L0AT", 1, meta["total_positions"],
                             meta["window"], len(meta["tokens"]), 0, 0, 0,
                             *tokens, meta["rope_theta"], meta["norm_eps"],
                             len(self.pieces))
        with open(table_path, "wb") as f:
            f.write(header)
            for piece in self.pieces:
                name = piece["name"].encode()[:63]
                f.write(struct.pack("<64sIIfff", name, piece["offset"],
                                    piece["count"], piece["tol_rel"],
                                    piece["tol_abs"], piece["tol_scale"]))
        print(f"expectations: {bin_path} ({len(self.pieces)} pieces, "
              f"{os.path.getsize(bin_path)} bytes) source={meta['source']}")


class _Drift:
    """Scales tol_rel for pieces emitted at drift-exposed positions."""

    def __init__(self, pieces, factor):
        self.pieces = pieces
        self.factor = factor

    def add(self, name, array, tol_rel, tol_abs):
        self.pieces.add(name, array, tol_rel * self.factor, tol_abs * self.factor)


class Block:
    def __init__(self, t, layer, ratio, freqs, engram=None, tol_amp=None):
        self.t = t
        self.layer = layer
        self.ratio = ratio
        self.freqs = freqs
        self.engram = engram
        self.is_csa2 = ratio > 0
        self.tol_amp = tol_amp if tol_amp is not None else (5.0 if self.is_csa2 else 1.0)
        if self.is_csa2:
            self.compressor = Compressor(ratio, t[K_CWKV],
                                         t[K_CWGATE] if ratio > 1 else None,
                                         t[K_CNORM])
            self.indexer = Indexer(ratio, t[K_IQB][0], t[K_IQB][1], t[K_IWK],
                                   t[K_IWP], t[K_IKN])

    def attention(self, x_collapsed, cache, start_pos, pieces, raw, tag):
        t = self.t
        freqs = self.freqs
        qr = rmsnorm(fp8_gemm_vector(x_collapsed, t[K_QA][0], t[K_QA][1], Q_LORA),
                     t[K_QNORM])
        q = np.zeros((LOCAL_HEADS, HEAD_DIM), dtype=np.uint16)
        for h in range(LOCAL_HEADS):
            row0 = h * HEAD_DIM
            qh = fp8_gemm_vector(qr, t[K_QB][0][row0:row0 + HEAD_DIM],
                                 t[K_QB][1][(row0 // 32):((row0 + HEAD_DIM) // 32)],
                                 HEAD_DIM)
            q[h] = apply_rope_tail(qh, freqs[start_pos])
        kv_normed = rmsnorm(fp8_gemm_vector(x_collapsed, t[K_KVA][0], t[K_KVA][1],
                                            KV_LATENT), t[K_KVNORM])
        kv_row = act_dequant(apply_rope_tail(kv_normed, freqs[start_pos]))
        if start_pos == 0:
            cache.append(kv_row)
        else:
            slot = start_pos % WINDOW
            if slot >= len(cache):
                cache.extend([np.zeros(KV_LATENT, dtype=np.uint16)] * (slot + 1 - len(cache)))
            cache[slot] = kv_row
        if self.is_csa2:
            latent = self.compressor.step(x_collapsed, start_pos, pieces, tag)
            if latent is not None:
                self.indexer.publish_k(latent, start_pos, freqs, pieces, tag)
                roped = apply_rope_tail(latent,
                                        freqs[start_pos + 1 - self.ratio])
                comp_row = fp4_rt_e4m3(bf16_f32(roped).reshape(1, -1)).reshape(KV_LATENT)
                self.comp_cache[start_pos // self.ratio] = comp_row
                if tag:
                    pieces.add(f"{tag}.c_kv_row", bf16_f32(comp_row), 0.25, 1e-6)
            idxs_c = self.indexer.score(x_collapsed, qr, start_pos, freqs, pieces, tag)
        if tag:
            a = self.tol_amp
            pieces.add(f"{tag}.q_lora", bf16_f32(qr), 2e-3 * a, 1e-5)
            pieces.add(f"{tag}.q", bf16_f32(q), 2e-3 * a, 1e-5)
            raw.add(f"{tag}.kv_row", bf16_f32(kv_row), 3e-2 * a, 1e-2 * a)
        idxs_w = window_topk_idxs(start_pos)
        if self.is_csa2:
            n = (start_pos + 1) // self.ratio
            pad = [np.zeros(KV_LATENT, dtype=np.uint16)] * (WINDOW - len(cache))
            comp_rows = [self.comp_cache[j] for j in range(n)]
            full_cache = cache[:WINDOW] + pad + comp_rows
            idxs = np.concatenate([idxs_w, idxs_c]).astype(np.int32)
        else:
            full_cache = cache
            idxs = idxs_w
        sinks = self.t[K_SINK].reshape(LOCAL_SINKS).astype(np.float32)
        o = sparse_attn(q, full_cache, idxs, sinks, HEAD_DIM ** -0.5)
        for h in range(LOCAL_HEADS):
            o[h] = apply_rope_tail(o[h], freqs[start_pos], inverse=True)
        if tag:
            raw.add(f"{tag}.attn_out_rope_inv", bf16_f32(o), 3e-2 * self.tol_amp,
                    1e-4)
        oa_vals = FP8_LUT[t[K_OA][0].reshape(LOCAL_OA_ROWS, OA_FULL_COLS)]
        oa_scales = e8m0_f32(t[K_OA][1].reshape(LOCAL_OA_ROWS // 32, OA_FULL_COLS // 32))
        oa = bf16_f32(bf16_u16(oa_vals * oa_scales.repeat(32, 0).repeat(32, 1))).astype(np.float32)
        partial = (oa @ bf16_f32(o.reshape(-1)).astype(np.float32)).astype(np.float32)
        attn_out = fp8_gemm_vector(bf16_u16(partial), t[K_OB][0], t[K_OB][1], HIDDEN)
        if tag:
            raw.add(f"{tag}.wo_b_out", bf16_f32(attn_out), 2e-2 * self.tol_amp,
                    1e-5)
        return attn_out

    def moe(self, x_collapsed, pieces, tag, full):
        t = self.t
        scores, indices, weights = gate_topk(x_collapsed, t[K_ROUTER],
                                             t[K_RBIAS].reshape(N_EXPERTS))
        routed = np.zeros(HIDDEN, dtype=np.float32)
        for slot in range(TOPK):
            expert = int(indices[slot])
            local = expert - RANK * LOCAL_EXPERTS
            if 0 <= local < LOCAL_EXPERTS:
                w1p, w1s = t[K_W1](local)
                w3p, w3s = t[K_W3](local)
                w2p, w2s = t[K_W2](local)
                contrib = expert_mlp(x_collapsed, w1p, w1s, w3p, w3s, w2p, w2s,
                                     float(weights[slot]))
                routed += bf16_f32(contrib).astype(np.float32)
        shared = shared_mlp(x_collapsed, t[K_SW1], t[K_SW3], t[K_SW2])
        moe_out = bf16_u16((routed + bf16_f32(shared).astype(np.float32)).astype(np.float32))
        if tag:
            pieces.add(f"{tag}.shared_out", bf16_f32(shared), 2e-2, 1e-5)
            if full:
                pieces.add(f"{tag}.router_scores", scores, 2e-3, 1e-5)
                pieces.add(f"{tag}.router_indices", indices.astype(np.float32), 0.0, 0.0)
                pieces.add(f"{tag}.router_weights", weights, 2e-3, 1e-6)
                pieces.add(f"{tag}.routed_sum", routed, 2e-2, 1e-5)
                pieces.add(f"{tag}.moe_out", bf16_f32(moe_out), 2e-2, 1e-5)
        return moe_out

    def forward(self, stream_in, start_pos, cache, comp_cache, pre_mix_in, attn, ffn,
                raw, tag):
        want = bool(tag)
        t = self.t
        self.comp_cache = comp_cache
        residual = stream_in.reshape(HC * HIDDEN)
        fn_a = t[K_HCAF].astype(np.float32)
        mixes_a, pre_a, post_a, comb_a = hc_mixes(residual, fn_a, t[K_HCAS], t[K_HCAB])
        collapsed_a = hc_pre(residual, pre_mix_in)
        normed_a = rmsnorm(collapsed_a, t[K_ATTN_NORM])
        attn_out = self.attention(normed_a, cache, start_pos, attn, raw, tag)
        stream_a = hc_post(attn_out, residual, post_a, comb_a)
        fn_f = t[K_HCFF].astype(np.float32)
        mixes_f, pre_f, post_f, comb_f = hc_mixes(stream_a, fn_f, t[K_HCFS], t[K_HCFB])
        collapsed_f = hc_pre(stream_a, pre_a)
        normed_f = rmsnorm(collapsed_f, t[K_FFN_NORM])
        moe_out = self.moe(normed_f, ffn, tag,
                           self.layer == LAYER or start_pos == 0)
        stream_f = hc_post(moe_out, stream_a, post_f, comb_f)
        if want:
            a = self.tol_amp
            attn.add(f"{tag}.mixes_attn", mixes_a, 2e-3 * a, 1e-6)
            attn.add(f"{tag}.pre_attn", pre_a, 2e-4 * a, 1e-7)
            attn.add(f"{tag}.post_attn", post_a, 2e-4 * a, 1e-7)
            attn.add(f"{tag}.comb_attn", comb_a, 2e-4 * a, 1e-7)
            attn.add(f"{tag}.collapsed_attn", bf16_f32(collapsed_a), 2e-3 * a, 1e-5)
            attn.add(f"{tag}.normed_attn", bf16_f32(normed_a), 2e-3 * a, 1e-5)
            ffn.add(f"{tag}.stream_after_attn", bf16_f32(stream_a), 2e-2 * a, 1e-5)
            ffn.add(f"{tag}.mixes_ffn", mixes_f, 2e-3 * a, 1e-6)
            ffn.add(f"{tag}.pre_ffn", pre_f, 2e-4 * a, 1e-7)
            ffn.add(f"{tag}.post_ffn", post_f, 2e-4 * a, 1e-7)
            ffn.add(f"{tag}.comb_ffn", comb_f, 2e-4 * a, 1e-7)
            ffn.add(f"{tag}.collapsed_ffn", bf16_f32(collapsed_f), 2e-3 * a, 1e-5)
            ffn.add(f"{tag}.normed_ffn", bf16_f32(normed_f), 2e-3 * a, 1e-5)
        return stream_f, pre_f


TOKENS = [11, 900, 5, 12999, 42, 7, 12345, 8000, 3133, 64, 1, 0, 999, 2048, 777]
TOTAL_POSITIONS = 131


def expert_views(t):
    views = dict(t)
    for kind in (K_W1, K_W2, K_W3):
        if callable(t[kind]):
            continue
        code, scales = t[kind]
        per, sper = code.size // LOCAL_EXPERTS, scales.size // LOCAL_EXPERTS
        views[kind] = lambda g, c=code, s=scales, p=per, q=sper: \
            (c[g * p:(g + 1) * p], s[g * q:(g + 1) * q])
    return views


def run_layer(block, layer, embed, pieces, prefix, engram=None):
    cache = []
    comp_cache = {}
    pre_mix = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for pos in range(TOTAL_POSITIONS + 1):
        token = TOKENS[pos % len(TOKENS)]
        if token >= LOCAL_EMBED:
            raise SystemExit(f"fixture token {token} outside rank-0 vocab slice")
        stream_in = np.tile(embed[token], HC)
        if layer == LAYER:
            tag = f"pos{pos}" if pos in (0, 1, 2) else None
        else:
            tag = f"{prefix}.p{pos}" if pos < TAG_POSITIONS[layer] else None
        if engram is not None:
            stream_in = engram.forward(stream_in, pos, TOKENS, pieces, tag)
        attn = _Drift(pieces, 1.0 if pos == 0 else 25.0)
        stream, pre_mix = block.forward(
            stream_in, pos, cache, comp_cache, pre_mix, attn, _Drift(pieces, 25.0),
            pieces, tag)
        if layer != LAYER:
            continue
        pieces.add(f"stream_pos{pos}", bf16_f32(stream), 8e-2, 1e-4)


def build_engrams(t, multipliers, primes):
    engrams = {}
    for hidx, layer_id in enumerate(ENGRAM_LAYERS):
        source = t["engram"].get(layer_id)
        if source is None:
            continue
        weights = (source["wkv"][0], source["wkv"][1], source["qw"], source["kw"])
        engrams[layer_id] = Engram(layer_id, hidx, multipliers, primes, weights,
                                   source["table"])
    return engrams


def run(t, out_dir, source, pack_path, ckpt_report, plan):
    pieces = Expectations(out_dir)
    freqs_pure = rope_freqs(TOTAL_POSITIONS)
    freqs_yarn = rope_freqs_yarn(TOTAL_POSITIONS, COMP_THETA)
    embed = t["globals"][K_EMBED].reshape(LOCAL_EMBED, HIDDEN)
    engrams = build_engrams(t, t["engram_meta"]["multipliers"],
                            t["engram_meta"]["primes"])
    for layer, ratio in plan.items():
        engram = engrams.get(layer)
        block = Block(expert_views(t["layers"][layer]), layer, ratio,
                      freqs_pure if ratio == 0 else freqs_yarn, engram,
                      5.0 if engram is not None else None)
        run_layer(block, layer, embed, pieces, f"l{layer}", engram)
    fixture_count = write_engram_fixture(os.path.join(out_dir, "dsv41_engram.bin"),
                                         [engrams[l] for l in sorted(engrams)]) \
        if engrams else 0
    meta = {
        "lane": "dsv5-flash", "model_revision": REVISION, "source": source,
        "pack": os.path.abspath(pack_path), "layers": sorted(plan),
        "layer_plan": {str(l): r for l, r in plan.items()},
        "tp": TP, "rank": RANK,
        "tokens": TOKENS, "total_positions": TOTAL_POSITIONS, "window": WINDOW,
        "rope_theta": ROPE_THETA,
        "csa2_rope": {"base": COMP_THETA, "factor": YARN_FACTOR,
                      "original_seq_len": YARN_ORIGINAL,
                      "beta_fast": YARN_BETA_FAST, "beta_slow": YARN_BETA_SLOW,
                      "note": "one yarn table for q/window-kv/compressed-kv/indexer-q/indexer-k on CSA2 layers"},
        "compress_ratio": "per layer_plan (kv_source layers only: 2/8/14 m=2, 20 m=1; all other CSA2 layers are Reuse and read the shared compressed cache)",
        "indexer": {"heads": IDX_HEADS, "head_dim": IDX_DIM, "topk": IDX_TOPK,
                    "score_piece": "rank-0 partial (local-head sum, pre-all-reduce)"},
        "input_note": "layer 0 consumes the tiled embedding; layer 1 consumes the tiled embedding through the engram row-shard gate (its true input; fixture compressed stream = tokens, identity token map); layers 2+ consume the same synthetic tiled stream as per-piece instruments because the full chain depends on unrolled layers not in the ladder plan",
        "engram": {
            "layers": list(ENGRAM_LAYERS),
            "heads": ENGRAM_HEADS, "orders": ENGRAM_ORDERS,
            "head_dim": ENGRAM_HEAD_DIM, "compressed_vocab": ENGRAM_VOCAB,
            "pad_id": ENGRAM_PAD_ID, "mult_seed": ENGRAM_MULT_SEED,
            "prime_start": ENGRAM_PRIME_START,
            "synth_prime_start": ENGRAM_SYNTH_PRIME_START,
            "partition": "owner=id//part, part=ceil(entries/16) at TP8xPP2=16 ranks",
            "fixture_pieces": "eg_access=owner|local>>24|local&0xffffff exact, eg_rows dequant, eg_value, eg_gate, eg_out",
            "fixture_records": fixture_count,
            "fixture_token_map": "identity on the fixture (production map is the 99092-entry normalization-collapsed lookup; ships with full-chain decode)",
        },
        "stream_note": "the bf16 hc-feedback chain is chaotic under ~1-ulp implementation-order differences (numpy BLAS/pairwise vs C sequential reductions) and amplifies at layer-dependent rates on the synthetic fixture (measured crossings: l2/l8 pos 46, l14 pos 40, l20 pos 4, l38 pos 7), so the end-to-end stream piece is emitted for layer 0 only and every other layer is verified by its per-operator tagged pieces",
        "norm_eps": NORM_EPS,
        "swiglu_limit": SWIGLU_LIMIT, "route_scale": ROUTE_SCALE,
        "hc": {"mult": HC, "sinkhorn_iters": SINKHORN_ITERS, "eps": HC_EPS},
        "checkpoint_report": ckpt_report,
        "reference_sha256": {
            "inference/model.py": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
            "inference/kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
            "inference/engram.py": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
            "config.json": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
            "DeepSeek_V41_Tech_Report.pdf": "ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d",
        },
    }
    pieces.write(meta)


def load_checkpoint_raw(ckpt_dir, name):
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    shard = wm[name]
    path = os.path.join(ckpt_dir, shard)
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header[name]
    begin, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + n + begin)
        return f.read(end - begin)


def load_checkpoint_tensor(ckpt_dir, name):
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    shard = wm[name]
    path = os.path.join(ckpt_dir, shard)
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header[name]
    begin, end = meta["data_offsets"]
    count = 1
    for d in meta["shape"]:
        count *= d
    with open(path, "rb") as f:
        f.seek(8 + n + begin)
        raw = f.read(end - begin)
    if meta["dtype"] == "BF16":
        return bf16_f32(np.frombuffer(raw, dtype=np.uint16, count=count))
    if meta["dtype"] == "F32":
        return np.frombuffer(raw, dtype=np.float32, count=count)
    if meta["dtype"] in ("F8_E4M3", "I8", "F8_E8M0"):
        return np.frombuffer(raw, dtype=np.uint8, count=count)
    raise SystemExit(f"unsupported checkpoint dtype {meta['dtype']} for {name}")


def verify_checkpoint(reader, ckpt_dir):
    report = []
    for kind, name, mode in ((K_QA, "layers.0.attn.wq_a.weight", None),
                             (K_QB, "layers.0.attn.wq_b.weight", "rows"),
                             (K_ROUTER, "layers.0.ffn.gate.weight", None),
                             (K_SW1, "layers.0.ffn.shared_experts.w1.weight", None)):
        ck = load_checkpoint_raw(ckpt_dir, name)
        pk = reader.view(kind)[0].tobytes()
        if mode == "rows":
            r = len(ck) // TP
            ck = ck[RANK * r:(RANK + 1) * r]
        if ck != pk:
            raise SystemExit(f"checkpoint cross-check FAILED for {name}")
        report.append({"tensor": name, "bytes": len(pk), "equal": True})
    name = "layers.0.ffn.experts.0.w1.weight"
    ck = load_checkpoint_raw(ckpt_dir, name)
    pk = reader.view(K_W1, expert=0)[0].tobytes()
    if ck != pk:
        raise SystemExit(f"checkpoint cross-check FAILED for {name}")
    report.append({"tensor": name, "bytes": len(pk), "equal": True})
    print(f"checkpoint cross-check: {len(report)} tensors byte-equal")
    return report


def real_tensors(reader, plan, warm_dir=None):
    t = {"globals": {}, "layers": {}}
    t["globals"][K_EMBED] = reader.bf16(K_EMBED).reshape(LOCAL_EMBED, HIDDEN)
    t["globals"][K_HEAD] = reader.bf16(K_HEAD).reshape(LOCAL_EMBED, HIDDEN)
    for layer, ratio in plan.items():
        tl = {}
        tl[K_ATTN_NORM] = reader.bf16(K_ATTN_NORM, layer)
        tl[K_FFN_NORM] = reader.bf16(K_FFN_NORM, layer)
        tl[K_QA] = reader.fp8(K_QA, layer, Q_LORA // 32, HIDDEN // 32)
        qb_p, qb_s = reader.fp8(K_QB, layer, LOCAL_QB_ROWS // 32, Q_LORA // 32)
        tl[K_QB] = (qb_p.reshape(LOCAL_QB_ROWS, Q_LORA), qb_s)
        tl[K_KVA] = reader.fp8(K_KVA, layer, KV_LATENT // 32, HIDDEN // 32)
        tl[K_QNORM] = reader.bf16(K_QNORM, layer)
        tl[K_KVNORM] = reader.bf16(K_KVNORM, layer)
        tl[K_SINK] = reader.f32(K_SINK, layer).reshape(LOCAL_SINKS)
        oa_p, oa_s = reader.view(K_OA, layer)
        if len(oa_p) != LOCAL_OA_ROWS * OA_FULL_COLS:
            raise SystemExit(f"pack o_a layer {layer} is {len(oa_p)} bytes; the "
                             f"reference-faithful TP8 layout is "
                             f"[{LOCAL_OA_ROWS}, {OA_FULL_COLS}] fp8 - REPACK required")
        tl[K_OA] = (oa_p, oa_s.reshape(LOCAL_OA_ROWS // 32, OA_FULL_COLS // 32))
        tl[K_OB] = reader.fp8(K_OB, layer, HIDDEN // 32, LOCAL_OB_COLS // 32)
        tl[K_HCAF] = reader.f32(K_HCAF, layer).reshape(HC_ROWS, HC_FLAT)
        tl[K_HCAB] = reader.f32(K_HCAB, layer).reshape(HC_ROWS)
        tl[K_HCAS] = reader.f32(K_HCAS, layer).reshape(3)
        tl[K_HCFF] = reader.f32(K_HCFF, layer).reshape(HC_ROWS, HC_FLAT)
        tl[K_HCFB] = reader.f32(K_HCFB, layer).reshape(HC_ROWS)
        tl[K_HCFS] = reader.f32(K_HCFS, layer).reshape(3)
        tl[K_ROUTER] = reader.bf16(K_ROUTER, layer)
        tl[K_RBIAS] = reader.f32(K_RBIAS, layer).reshape(N_EXPERTS)
        tl[K_SW1] = reader.fp8(K_SW1, layer, MOE_INTER // 32, HIDDEN // 32)
        tl[K_SW2] = reader.fp8(K_SW2, layer, HIDDEN // 32, MOE_INTER // 32)
        tl[K_SW3] = reader.fp8(K_SW3, layer, MOE_INTER // 32, HIDDEN // 32)
        tl[K_W1] = reader.experts(K_W1, layer)
        tl[K_W2] = reader.experts(K_W2, layer)
        tl[K_W3] = reader.experts(K_W3, layer)
        if ratio > 0:
            tl[K_IQB] = reader.fp8(K_IQB, layer, IDX_QB_SHAPE[0] // 32, Q_LORA // 32)
            tl[K_IWK] = reader.bf16(K_IWK, layer)
            tl[K_IWP] = reader.bf16(K_IWP, layer)
            tl[K_IKN] = reader.bf16(K_IKN, layer)
            tl[K_CWKV] = reader.bf16(K_CWKV, layer)
            if ratio > 1:
                tl[K_CWGATE] = reader.bf16(K_CWGATE, layer)
            tl[K_CNORM] = reader.bf16(K_CNORM, layer)
        t["layers"][layer] = tl
    engram_primes_real = engram_primes(synth=False)
    t["engram_meta"] = {"primes": engram_primes_real,
                        "multipliers": engram_multipliers()}
    t["engram"] = {}
    if warm_dir is None and any(l in plan for l in ENGRAM_LAYERS):
        raise SystemExit("engram ladder layers require --verify-checkpoint WARM_DIR "
                         "for the row tables and wkv/qk weights")
    for hidx, layer_id in enumerate(ENGRAM_LAYERS):
        if layer_id not in plan:
            continue
        entries = int(engram_primes_real[hidx].sum())
        wkv = np.frombuffer(load_checkpoint_raw(
            warm_dir, f"layers.{layer_id}.engram.wkv.weight"), dtype=np.uint8)
        wkv_s = np.frombuffer(load_checkpoint_raw(
            warm_dir, f"layers.{layer_id}.engram.wkv.scale"), dtype=np.uint8)
        if wkv.size != ENGRAM_WKV_ROWS * ENGRAM_WKV_COLS or wkv_s.size != 800 * 192:
            raise SystemExit(f"engram wkv plane bytes wrong for layer {layer_id}")
        t["engram"][layer_id] = {
            "table": CheckpointEngramTable(warm_dir, layer_id, entries),
            "wkv": (wkv, wkv_s),
            "qw": np.frombuffer(load_checkpoint_raw(
                warm_dir, f"layers.{layer_id}.engram.q_weight"), dtype=np.uint16),
            "kw": np.frombuffer(load_checkpoint_raw(
                warm_dir, f"layers.{layer_id}.engram.k_weight"), dtype=np.uint16),
        }
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-checkpoint")
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()
    if args.pack:
        reader = PackReader(args.pack)
        report = verify_checkpoint(reader, args.verify_checkpoint) \
            if args.verify_checkpoint else []
        run(real_tensors(reader, LAYER_PLAN, args.verify_checkpoint), args.out,
            "real", args.pack, report, LAYER_PLAN)
    else:
        rng = np.random.default_rng(args.seed)
        t = synth_tensors(rng)
        os.makedirs(args.out, exist_ok=True)
        pack_path = os.path.join(args.out, "dsv41_flash_synth_tp8.spstage")
        synth_write_pack(pack_path, t)
        run(t, args.out, "synth", pack_path, [], SYNTH_LAYERS)


if __name__ == "__main__":
    main()
