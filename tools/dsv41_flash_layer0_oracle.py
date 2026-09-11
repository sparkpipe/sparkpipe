#!/usr/bin/env python3
"""dsv41-flash layer-0 decode host oracle (M7 host-oracle-first).

Recomputes the layer-0 block (SWA-only: attention + single-pass mHC mult 4 +
router top-6 of 384 + routed/shared experts) from a TP8 rank stagepack and
EMITS per-piece expectations; tests/test_dsv41_flash_layer0_oracle.c
recomputes the same pieces from the same pack bytes in C and prints the
per-piece deltas. No module/driver code is imported: the formulas come from
the pinned reference sources (cache/dsv41_warm/ref/{model,kernel}.py at
revision dba1be0a, sha256 recorded in
model_contracts/dsv41_flash_authoritative.json):

  linear    = act_quant(x, 32, ue8m0) then fp8-block GEMM: per 32-wide
              k-block fp32 accumulation scaled by sa[i,kb]*sb[j,kb], bf16 out
  act_quant = s = 2^ceil(log2(amax/448)) per 32 block (amax floored 1e-4),
              q = fp8e4m3(clamp(x/s, +/-448)); the inplace variant the
              window-KV cache uses stores bf16(q*s)
  mHC       = pre=sigmoid(m*s0+b0)+eps; post=2*sigmoid(m*s1+b1);
              comb=softmax(m*s2+b2, rows)+eps, col-norm, then
              (iters-1) x (row-norm, col-norm), all +eps
  gate      = sqrt(softplus(x.W)) fp32 (bf16 weight, fp32 math); top-6 of
              score+bias; weights/(sum + 1e-20) * 1.5
  rope      = adjacent-pair complex rotation on the 64-dim tail, pure
              theta 10000 on SWA-only layer 0 (no YaRN), conjugate on the
              attention output tail
  attention = softmax over the window ring (fp8-round-tripped bf16 rows)
              with the per-head sink exp(sink - max) in the denominator

Modes:
  synth --out DIR              constrained-random layer-0-only TP8 stagepack
                               + expectations (CI; no weights needed)
  real --pack FILE --out DIR   expectations from a real rank pack; fixture
                               token ids must be < VOCAB // TP (rank 0 owns
                               only its vocab slice)
  --verify-checkpoint DIR      re-derive sample planes from the warm
                               checkpoint shards and assert byte equality

Outputs: DIR/layer0_expectations.bin + DIR/layer0_manifest.json
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
LAYER = 0
TP = 8
RANK = 0
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"

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
(K_HCAF, K_HCAB, K_HCAS, K_HCFF, K_HCFB, K_HCFS, K_ROUTER, K_RBIAS,
 K_W1, K_W2, K_W3, K_SW1, K_SW2, K_SW3) = (20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34)

LOCAL_HEADS = HEADS // TP
LOCAL_QB_ROWS = HEADS * HEAD_DIM // TP
LOCAL_OA_ROWS = O_GROUPS * O_LORA // TP
LOCAL_OA_COLS = HEADS * 64 // TP
LOCAL_OB_COLS = O_GROUPS * O_LORA // TP
LOCAL_EXPERTS = N_EXPERTS // TP
LOCAL_SINKS = SINKS // TP
LOCAL_EMBED = VOCAB // TP
W1_SHAPE = (MOE_INTER, HIDDEN // 2)
W2_SHAPE = (HIDDEN, MOE_INTER // 2)
W3_SHAPE = (MOE_INTER, HIDDEN // 2)

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
    normal = (bits >= 0x3C000000) & (bits < 0x7F800000)
    code = np.where(normal, ((exp - 120) << 3) | m3, code)
    sub = (bits > 0) & (bits < 0x3C000000)
    sub_m = rne_shift(0x800000 | man, np.clip(268 - exp, 0, 31))
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


def rmsnorm(x_bf16, weight_bf16):
    x = bf16_f32(np.asarray(x_bf16).reshape(-1)).astype(np.float32)
    w = bf16_f32(np.asarray(weight_bf16).reshape(-1)).astype(np.float32)
    rstd = 1.0 / math.sqrt(float((x * x).mean(dtype=np.float32)) + NORM_EPS)
    return bf16_u16(w * x * rstd)


def rope_freqs(last_pos):
    inv = 1.0 / (ROPE_THETA ** (np.arange(0, ROPE, 2, dtype=np.float32) / ROPE))
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
        top = scores.max()
        ex = np.exp(scores - top).astype(np.float32)
        denom = (ex.sum(dtype=np.float32) + math.exp(float(sinks[h]) - float(top))).astype(np.float32)
        acc = np.zeros(HEAD_DIM, dtype=np.float32)
        for j, idx in enumerate(valid):
            acc += ex[j] * bf16_f32(cache[idx]).astype(np.float32)
        o[h] = bf16_u16(acc / denom)
    return o


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


def add_dense(writer, kind, payload, codec, scale=b"", groups=1):
    rows, cols = payload.shape
    if codec == COD_BF16:
        writer.add(kind, LAYER, PT_BF16, COD_BF16, SE_NONE, groups, rows, cols,
                   payload.tobytes(), b"")
    elif codec == COD_NONE:
        writer.add(kind, LAYER, PT_F32, COD_NONE, SE_NONE, groups, rows, cols,
                   payload.astype(np.float32).tobytes(), b"")
    else:
        writer.add(kind, LAYER, PT_PACKED, COD_FP8, SE_E8M0, groups, rows, cols,
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

    t = {}
    t[K_EMBED] = bf16n((LOCAL_EMBED, HIDDEN), 0.1)
    t[K_FNORM] = bf16n((1, HIDDEN), 0.1)
    t[K_HEAD] = bf16n((LOCAL_EMBED, HIDDEN))
    t[K_ATTN_NORM] = bf16n((1, HIDDEN), 0.1)
    t[K_FFN_NORM] = bf16n((1, HIDDEN), 0.1)
    t[K_QA] = fp8w((Q_LORA, HIDDEN))
    t[K_QB] = fp8w((LOCAL_QB_ROWS, Q_LORA))
    t[K_KVA] = fp8w((KV_LATENT, HIDDEN))
    t[K_QNORM] = bf16n((1, Q_LORA), 0.1)
    t[K_KVNORM] = bf16n((1, KV_LATENT), 0.1)
    t[K_SINK] = f32n((1, LOCAL_SINKS))
    t[K_OA] = fp8w((LOCAL_OA_ROWS, LOCAL_OA_COLS))
    t[K_OB] = fp8w((HIDDEN, LOCAL_OB_COLS))
    t[K_HCAF] = (f32n((HC_ROWS, HC_FLAT), 0.02)).reshape(HC_ROWS, HC_FLAT)
    t[K_HCAB] = f32n(HC_ROWS, 0.3)
    t[K_HCAS] = np.abs(f32n(3, 0.5)) + 0.25
    t[K_HCFF] = f32n((HC_ROWS, HC_FLAT), 0.02)
    t[K_HCFB] = f32n(HC_ROWS, 0.3)
    t[K_HCFS] = np.abs(f32n(3, 0.5)) + 0.25
    t[K_ROUTER] = bf16n((N_EXPERTS, HIDDEN), 0.02)
    t[K_RBIAS] = f32n(N_EXPERTS, 0.05)
    for kind, shape, srows in ((K_W1, W1_SHAPE, MOE_INTER), (K_W2, W2_SHAPE, HIDDEN),
                               (K_W3, W3_SHAPE, MOE_INTER)):
        planes = [fp4w(shape, srows) for _ in range(LOCAL_EXPERTS)]
        t[kind] = (np.concatenate([p[0].reshape(-1) for p in planes]),
                   np.concatenate([p[1].reshape(-1) for p in planes]))
    t[K_SW1] = fp8w((MOE_INTER, HIDDEN))
    t[K_SW2] = fp8w((HIDDEN, MOE_INTER))
    t[K_SW3] = fp8w((MOE_INTER, HIDDEN))
    return t


def synth_write_pack(path, t):
    writer = PackWriter(path)
    for kind, payload, codec in ((K_EMBED, t[K_EMBED], COD_BF16),
                                 (K_FNORM, t[K_FNORM], COD_BF16),
                                 (K_HEAD, t[K_HEAD], COD_BF16)):
        rows, cols = payload.shape
        writer.add(kind, GLOBAL_LAYER, PT_BF16, COD_BF16, SE_NONE, 1, rows, cols,
                   payload.tobytes(), b"")
    for kind, payload, codec in ((K_ATTN_NORM, t[K_ATTN_NORM], COD_BF16),
                                 (K_FFN_NORM, t[K_FFN_NORM], COD_BF16),
                                 (K_QNORM, t[K_QNORM], COD_BF16),
                                 (K_KVNORM, t[K_KVNORM], COD_BF16),
                                 (K_ROUTER, t[K_ROUTER], COD_BF16)):
        rows, cols = payload.shape
        writer.add(kind, LAYER, PT_BF16, COD_BF16, SE_NONE, 1, rows, cols,
                   payload.tobytes(), b"")
    for kind, payload in ((K_SINK, t[K_SINK]), (K_HCAF, t[K_HCAF]), (K_HCAB, t[K_HCAB]),
                          (K_HCAS, t[K_HCAS]), (K_HCFF, t[K_HCFF]), (K_HCFB, t[K_HCFB]),
                          (K_HCFS, t[K_HCFS]), (K_RBIAS, t[K_RBIAS])):
        flat = payload.reshape(1, -1)
        rows, cols = flat.shape
        writer.add(kind, LAYER, PT_F32, COD_NONE, SE_NONE, 1, rows, cols,
                   flat.astype(np.float32).tobytes(), b"")
    for kind, payload, scale in ((K_QA, t[K_QA], None), (K_QB, t[K_QB], None),
                                 (K_KVA, t[K_KVA], None), (K_OA, t[K_OA], None),
                                 (K_OB, t[K_OB], None), (K_SW1, t[K_SW1], None),
                                 (K_SW2, t[K_SW2], None), (K_SW3, t[K_SW3], None)):
        code, scales = payload
        rows, cols = code.shape
        writer.add(kind, LAYER, PT_PACKED, COD_FP8, SE_E8M0, 1, rows, cols,
                   code.tobytes(), scales.tobytes())
    for kind in (K_W1, K_W2, K_W3):
        code, scales = t[kind]
        rows = MOE_INTER if kind != K_W2 else HIDDEN
        cols = code.size // LOCAL_EXPERTS // rows
        writer.add(kind, LAYER, PT_PACKED, COD_MXFP4, SE_E8M0, LOCAL_EXPERTS, rows,
                   cols, code.tobytes(), scales.tobytes())
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

    def view(self, kind, expert=None):
        e = self.entries[(kind, GLOBAL_LAYER if kind <= K_HEAD else LAYER)]
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

    def bf16(self, kind):
        return np.frombuffer(self.view(kind)[0], dtype=np.uint16)

    def f32(self, kind):
        return np.frombuffer(self.view(kind)[0], dtype=np.float32)

    def fp8(self, kind, scale_rows=None, scale_cols=None):
        p, s = self.view(kind)
        if scale_rows is not None:
            s = s.reshape(scale_rows, scale_cols)
        return p, s

    def experts(self, kind):
        def one(g):
            p, s = self.view(kind, g)
            return p, s
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
        bin_path = os.path.join(self.out_dir, "layer0_expectations.bin")
        with open(bin_path, "wb") as f:
            f.write(bytes(self.bin))
        meta["pieces"] = self.pieces
        meta["expectations_sha256"] = hashlib.sha256(bytes(self.bin)).hexdigest()
        with open(os.path.join(self.out_dir, "layer0_manifest.json"), "w") as f:
            json.dump(meta, f, indent=1)
        table_path = os.path.join(self.out_dir, "layer0_piece_table.bin")
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
    def __init__(self, t):
        self.t = t

    def attention(self, x_collapsed, freqs, cache, start_pos, pieces, tag):
        t = self.t
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
        idxs = window_topk_idxs(start_pos)
        sinks = self.t[K_SINK].reshape(LOCAL_SINKS).astype(np.float32)
        o = sparse_attn(q, cache, idxs, sinks, HEAD_DIM ** -0.5)
        for h in range(LOCAL_HEADS):
            o[h] = apply_rope_tail(o[h], freqs[start_pos], inverse=True)
        flat = o.reshape(-1)
        o_slice = flat[RANK * LOCAL_OA_COLS:(RANK + 1) * LOCAL_OA_COLS]
        oa_vals = FP8_LUT[t[K_OA][0].reshape(LOCAL_OA_ROWS, LOCAL_OA_COLS)]
        oa_scales = e8m0_f32(t[K_OA][1].reshape(LOCAL_OA_ROWS // 32, LOCAL_OA_COLS // 32))
        oa = bf16_f32(bf16_u16(oa_vals * oa_scales.repeat(32, 0).repeat(32, 1))).astype(np.float32)
        partial = (oa @ bf16_f32(o_slice).astype(np.float32)).astype(np.float32)
        attn_out = fp8_gemm_vector(bf16_u16(partial), t[K_OB][0], t[K_OB][1], HIDDEN)
        if tag:
            pieces.add(f"{tag}.q_lora", bf16_f32(qr), 2e-3, 1e-5)
            pieces.add(f"{tag}.q", bf16_f32(q), 2e-3, 1e-5)
            pieces.add(f"{tag}.kv_row", bf16_f32(kv_row), 2e-3, 1e-5)
            pieces.add(f"{tag}.attn_out_rope_inv", bf16_f32(o), 2e-3, 1e-5)
            pieces.add(f"{tag}.wo_b_out", bf16_f32(attn_out), 2e-2, 1e-5)
        return attn_out

    def moe(self, x_collapsed, pieces, tag):
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
            pieces.add(f"{tag}.router_scores", scores, 2e-3, 1e-5)
            pieces.add(f"{tag}.router_indices", indices.astype(np.float32), 0.0, 0.0)
            pieces.add(f"{tag}.router_weights", weights, 2e-3, 1e-6)
            pieces.add(f"{tag}.routed_sum", routed, 2e-2, 1e-5)
            pieces.add(f"{tag}.shared_out", bf16_f32(shared), 2e-2, 1e-5)
            pieces.add(f"{tag}.moe_out", bf16_f32(moe_out), 2e-2, 1e-5)
        return moe_out

    def forward(self, stream_in, start_pos, freqs, cache, pre_mix_in, pieces, tag):
        want = bool(tag)
        t = self.t
        residual = stream_in.reshape(HC * HIDDEN)
        fn_a = t[K_HCAF].astype(np.float32)
        mixes_a, pre_a, post_a, comb_a = hc_mixes(residual, fn_a, t[K_HCAS], t[K_HCAB])
        collapsed_a = hc_pre(residual, pre_mix_in)
        normed_a = rmsnorm(collapsed_a, t[K_ATTN_NORM])
        attn_out = self.attention(normed_a, freqs, cache, start_pos, pieces, tag)
        stream_a = hc_post(attn_out, residual, post_a, comb_a)
        fn_f = t[K_HCFF].astype(np.float32)
        mixes_f, pre_f, post_f, comb_f = hc_mixes(stream_a, fn_f, t[K_HCFS], t[K_HCFB])
        collapsed_f = hc_pre(stream_a, pre_a)
        normed_f = rmsnorm(collapsed_f, t[K_FFN_NORM])
        moe_out = self.moe(normed_f, pieces, tag)
        stream_f = hc_post(moe_out, stream_a, post_f, comb_f)
        if want:
            pieces.add(f"{tag}.mixes_attn", mixes_a, 2e-3, 1e-6)
            pieces.add(f"{tag}.pre_attn", pre_a, 2e-4, 1e-7)
            pieces.add(f"{tag}.post_attn", post_a, 2e-4, 1e-7)
            pieces.add(f"{tag}.comb_attn", comb_a, 2e-4, 1e-7)
            pieces.add(f"{tag}.collapsed_attn", bf16_f32(collapsed_a), 2e-3, 1e-5)
            pieces.add(f"{tag}.normed_attn", bf16_f32(normed_a), 2e-3, 1e-5)
            pieces.add(f"{tag}.stream_after_attn", bf16_f32(stream_a), 2e-2, 1e-5)
            pieces.add(f"{tag}.mixes_ffn", mixes_f, 2e-3, 1e-6)
            pieces.add(f"{tag}.pre_ffn", pre_f, 2e-4, 1e-7)
            pieces.add(f"{tag}.post_ffn", post_f, 2e-4, 1e-7)
            pieces.add(f"{tag}.comb_ffn", comb_f, 2e-4, 1e-7)
            pieces.add(f"{tag}.collapsed_ffn", bf16_f32(collapsed_f), 2e-3, 1e-5)
            pieces.add(f"{tag}.normed_ffn", bf16_f32(normed_f), 2e-3, 1e-5)
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


def run(t, out_dir, source, pack_path, ckpt_report):
    block = Block(expert_views(t))
    pieces = Expectations(out_dir)
    freqs = rope_freqs(TOTAL_POSITIONS)
    cache = []
    pre_mix = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for pos in range(TOTAL_POSITIONS + 1):
        token = TOKENS[pos % len(TOKENS)]
        if token >= LOCAL_EMBED:
            raise SystemExit(f"fixture token {token} outside rank-0 vocab slice")
        row = token * HIDDEN
        embed = t[K_EMBED].reshape(LOCAL_EMBED, HIDDEN)[token]
        stream_in = np.tile(embed, HC)
        tag = f"pos{pos}" if pos in (0, 1, 2) else None
        drift = 1.0 if pos == 0 else 5.0
        stream, pre_mix = block.forward(
            stream_in, pos, freqs, cache, pre_mix, _Drift(pieces, drift), tag)
        pieces.add(f"stream_pos{pos}", bf16_f32(stream), 8e-2, 1e-4)
    meta = {
        "lane": "dsv5-flash", "model_revision": REVISION, "source": source,
        "pack": os.path.abspath(pack_path), "layer": LAYER, "tp": TP, "rank": RANK,
        "tokens": TOKENS, "total_positions": TOTAL_POSITIONS, "window": WINDOW,
        "rope_theta": ROPE_THETA, "norm_eps": NORM_EPS,
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


def real_tensors(reader):
    t = {}
    t[K_EMBED] = reader.bf16(K_EMBED).reshape(LOCAL_EMBED, HIDDEN)
    t[K_FNORM] = reader.bf16(K_FNORM)
    t[K_HEAD] = reader.bf16(K_HEAD).reshape(LOCAL_EMBED, HIDDEN)
    t[K_ATTN_NORM] = reader.bf16(K_ATTN_NORM)
    t[K_FFN_NORM] = reader.bf16(K_FFN_NORM)
    t[K_QA] = reader.fp8(K_QA, Q_LORA // 32, HIDDEN // 32)
    qb_p, qb_s = reader.fp8(K_QB, LOCAL_QB_ROWS // 32, Q_LORA // 32)
    t[K_QB] = (qb_p.reshape(LOCAL_QB_ROWS, Q_LORA), qb_s)
    t[K_KVA] = reader.fp8(K_KVA, KV_LATENT // 32, HIDDEN // 32)
    t[K_QNORM] = reader.bf16(K_QNORM)
    t[K_KVNORM] = reader.bf16(K_KVNORM)
    t[K_SINK] = reader.f32(K_SINK).reshape(LOCAL_SINKS)
    t[K_OA] = reader.fp8(K_OA, LOCAL_OA_ROWS // 32, LOCAL_OA_COLS // 32)
    t[K_OB] = reader.fp8(K_OB, HIDDEN // 32, LOCAL_OB_COLS // 32)
    t[K_HCAF] = reader.f32(K_HCAF).reshape(HC_ROWS, HC_FLAT)
    t[K_HCAB] = reader.f32(K_HCAB).reshape(HC_ROWS)
    t[K_HCAS] = reader.f32(K_HCAS).reshape(3)
    t[K_HCFF] = reader.f32(K_HCFF).reshape(HC_ROWS, HC_FLAT)
    t[K_HCFB] = reader.f32(K_HCFB).reshape(HC_ROWS)
    t[K_HCFS] = reader.f32(K_HCFS).reshape(3)
    t[K_ROUTER] = reader.bf16(K_ROUTER)
    t[K_RBIAS] = reader.f32(K_RBIAS).reshape(N_EXPERTS)
    t[K_SW1] = reader.fp8(K_SW1, MOE_INTER // 32, HIDDEN // 32)
    t[K_SW2] = reader.fp8(K_SW2, HIDDEN // 32, MOE_INTER // 32)
    t[K_SW3] = reader.fp8(K_SW3, MOE_INTER // 32, HIDDEN // 32)
    t[K_W1] = reader.experts(K_W1)
    t[K_W2] = reader.experts(K_W2)
    t[K_W3] = reader.experts(K_W3)
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
        run(real_tensors(reader), args.out, "real", args.pack, report)
    else:
        rng = np.random.default_rng(args.seed)
        t = synth_tensors(rng)
        os.makedirs(args.out, exist_ok=True)
        pack_path = os.path.join(args.out, "dsv41_flash_layer0_synth_tp8.spstage")
        synth_write_pack(pack_path, t)
        run(t, args.out, "synth", pack_path, [])


if __name__ == "__main__":
    main()
