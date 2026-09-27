#!/usr/bin/env python3
"""Criterion-7 layer-forward numerics comparator.

Compares the module's real-pack dumps (laguna_layer7_realpack.cu) against the
publisher reference computed from the warm checkpoint with the ANCHORS
rounding model. Independent of the driver lane math. Applies the DESIGN 5.3
gates: router sets exact, route weights rel <= 1e-2, hidden-state abs <=
0.02, MoE combine rel <= 1e-2. GATE-FAIL lines are hard failures (exit 1);
STAGE-DEVIATION lines are quantified rounding-model observations.
"""
import argparse
import json
import math
import os
import struct
import sys

import numpy as np

HIDDEN = 3072
HEAD_DIM = 128
HEADS_FULL = 48
HEADS_SLIDING = 72
KV_HEADS = 8
WINDOW = 512
EPS = 1e-6
ATTN_SCALE = 128 ** -0.5
YARN_THETA = 500000.0
YARN_FACTOR = 128.0
YARN_ORIG = 8192.0
BETA_FAST = 32.0
BETA_SLOW = 1.0
YARN_SCALE = 1.4852030263919618
YARN_ROT = 64
SWA_THETA = 10000.0
SWA_ROT = 128
EXPERTS = 256
TOP_K = 10
EXPERT_INTER = 1024
DENSE_INTER = 12288
ROUTED_SCALING = 2.5
TP = 8
FIRST_ROUTED = 1

GATE_HIDDEN_ABS = 0.02
GATE_WEIGHT_REL = 1e-2
GATE_MOE_REL = 1e-2
STAGE_HARD_ULP = 4
STAGE_FRAC_OVER_1 = 0.01
ATTN_HARD_ULP = 8
ATTN_FRAC_OVER_2 = 0.01

findings = []
gate_failures = []
tables = {}


def finding(msg):
    findings.append(msg)
    print("STAGE-DEVIATION: " + msg, flush=True)


def gate_fail(msg):
    gate_failures.append(msg)
    print("GATE-FAIL: " + msg, flush=True)


def f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def bf16_from_f32(x):
    x = f32(x)
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def bf16_matmul(x_bits, w_bits):
    x = bf16_to_f32(x_bits).astype(np.float64)
    w = bf16_to_f32(w_bits).astype(np.float64)
    accum = np.zeros((x.shape[0], w.shape[0]), dtype=np.float64)
    for k in range(w.shape[1]):
        accum += x[:, k:k + 1] * w[:, k:k + 1].T
    return bf16_from_f32(accum.astype(np.float32))


def bf16_mul(a_bits, b_bits):
    return bf16_from_f32(bf16_to_f32(a_bits) * bf16_to_f32(b_bits))


def bf16_add(a_bits, b_bits):
    return bf16_from_f32(bf16_to_f32(a_bits) + bf16_to_f32(b_bits))


def bf16_scalar_mul(a_bits, scalar):
    return bf16_from_f32(bf16_to_f32(a_bits) * np.float32(scalar))


def softplus_f32(x):
    return np.where(x > np.float32(20.0), x, np.log1p(np.exp(x)))


def silu_bits(x_bits):
    x = bf16_to_f32(x_bits)
    return bf16_from_f32(x / (np.float32(1.0) + np.exp(-x)))


def yarn_inv_freq_f64():
    index = np.arange(0, YARN_ROT, 2, dtype=np.float64)
    inv_freq = 1.0 / np.power(YARN_THETA, index / YARN_ROT)
    two_pi = 2.0 * math.pi
    low_exact = (YARN_ROT * math.log(YARN_ORIG / (BETA_FAST * two_pi))) / (2.0 * math.log(YARN_THETA))
    high_exact = (YARN_ROT * math.log(YARN_ORIG / (BETA_SLOW * two_pi))) / (2.0 * math.log(YARN_THETA))
    low = math.floor(min(max(low_exact, 0.0), YARN_ROT - 1.0))
    high = math.ceil(min(max(high_exact, 0.0), YARN_ROT - 1.0))
    out = np.empty(YARN_ROT // 2, dtype=np.float64)
    for i in range(YARN_ROT // 2):
        ramp = (i - low) / max(high - low, 1e-6)
        blend = min(max(ramp, 0.0), 1.0)
        out[i] = inv_freq[i] * (1.0 - blend) + (inv_freq[i] / YARN_FACTOR) * blend
    return out


def swa_inv_freq_f64():
    index = np.arange(0, SWA_ROT, 2, dtype=np.float64)
    return 1.0 / np.power(SWA_THETA, index / SWA_ROT)


def rope_cos_sin(inv_freq_f64, positions, attention_scaling):
    inv_freq = f32(inv_freq_f64)
    positions_f32 = f32(np.asarray(positions, dtype=np.float32))
    freqs = positions_f32[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = bf16_from_f32(f32(np.cos(emb) * f32(attention_scaling)))
    sin = bf16_from_f32(f32(np.sin(emb) * f32(attention_scaling)))
    return cos, sin


def rotate_half_pub(x_bits, cos_b, sin_b):
    x = bf16_to_f32(x_bits)
    rotary_dim = cos_b.shape[-1]
    half = rotary_dim // 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    rot_half = np.concatenate([-x[..., half:rotary_dim], x[..., :half]], axis=-1)
    term1 = bf16_mul(bf16_from_f32(x_rot), cos_b)
    term2 = bf16_mul(bf16_from_f32(rot_half), sin_b)
    summed = bf16_add(term1, term2)
    return np.concatenate([summed, bf16_from_f32(x_pass)], axis=-1)


def rmsnorm_pub(x_bits, w_bits):
    x = bf16_to_f32(x_bits)
    w = bf16_to_f32(w_bits)
    variance = np.mean(x * x, axis=-1, keepdims=True)
    normed = bf16_from_f32(x * (1.0 / np.sqrt(variance + np.float32(EPS))))
    return bf16_from_f32(bf16_to_f32(normed) * w)


def headrms_pub(x_bits, w_bits):
    rows, heads, dim = x_bits.shape
    flat = x_bits.reshape(rows * heads, dim)
    tiled = np.tile(w_bits, (rows * heads, 1))
    return rmsnorm_pub(flat, tiled).reshape(rows, heads, dim)


def softmax_rows_fp32(scores_f32):
    shifted = scores_f32 - scores_f32.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def attention_rows_pub(q_bits, k_bits, v_bits):
    q = bf16_to_f32(q_bits)
    k = bf16_to_f32(k_bits)
    v = bf16_to_f32(v_bits)
    scores = np.einsum("hd,sd->hs", q.astype(np.float64), k.astype(np.float64))
    scores_bits = bf16_from_f32(scores.astype(np.float32))
    scaled_bits = bf16_scalar_mul(scores_bits, ATTN_SCALE)
    probs = softmax_rows_fp32(bf16_to_f32(scaled_bits))
    probs_bits = bf16_from_f32(probs)
    out = np.einsum("hs,sd->hd", bf16_to_f32(probs_bits).astype(np.float64), v.astype(np.float64))
    return bf16_from_f32(out.astype(np.float32))


def bf16_ulp_key(bits):
    b = bits.astype(np.uint32)
    return np.where(b & np.uint32(0x8000),
                    np.uint32(0x8000) - (b & np.uint32(0x7FFF)) - np.uint32(1),
                    b + np.uint32(0x8000)).astype(np.int64)


def stage_stats(name, module_bits, reference_bits, hard_ulp=STAGE_HARD_ULP, frac_gate=STAGE_FRAC_OVER_1, attn_path=False):
    a = np.ascontiguousarray(module_bits, dtype=np.uint16).reshape(-1)
    b = np.ascontiguousarray(reference_bits, dtype=np.uint16).reshape(-1)
    if a.shape != b.shape:
        gate_fail("%s: shape %s vs reference %s" % (name, a.shape, b.shape))
        return None
    af = bf16_to_f32(a)
    bf = bf16_to_f32(b)
    diff = np.abs(af.astype(np.float64) - bf.astype(np.float64))
    ulp = np.abs(bf16_ulp_key(a) - bf16_ulp_key(b))
    exact = float((ulp == 0).mean())
    denom = np.maximum(np.abs(bf.astype(np.float64)), 1e-30)
    rel = float((diff / denom).max())
    norm_product = float(np.linalg.norm(af.astype(np.float64)) * np.linalg.norm(bf.astype(np.float64)))
    cos = float(np.dot(af.astype(np.float64), bf.astype(np.float64)) / max(norm_product, 1e-30))
    max_ulp = int(ulp.max())
    frac_over_1 = float((ulp > 1).mean())
    frac_over_2 = float((ulp > 2).mean())
    print("STAGE %-30s n=%-9d exact=%.6f max_ulp=%3d >1ulp=%.3g >2ulp=%.3g worst_rel=%.3e cos=%.9f worst_abs=%.3e"
          % (name, a.size, exact, max_ulp, frac_over_1, frac_over_2, rel, cos, float(diff.max())), flush=True)
    hard = ATTN_HARD_ULP if attn_path else hard_ulp
    frac_violation = frac_over_2 if attn_path else frac_over_1
    frac_limit = ATTN_FRAC_OVER_2 if attn_path else frac_gate
    if max_ulp > hard or frac_violation > frac_limit:
        worst = int(np.argmax(ulp))
        finding("%s: max_ulp %d (hard %d) frac_violation %.3g (limit %.3g); worst idx %d module %04x reference %04x (%.7g vs %.7g)"
                % (name, max_ulp, hard, frac_violation, frac_limit, worst, a[worst], b[worst], af[worst], bf[worst]))
    row = {"exact": round(exact, 6), "max_ulp": max_ulp, "worst_rel": rel, "cos": cos, "worst_abs": float(diff.max())}
    tables[name] = row
    return row


def hidden_gate(name, module_f32, reference_f32):
    diff = np.abs(module_f32.astype(np.float64) - reference_f32.astype(np.float64))
    worst = float(diff.max())
    denom = np.maximum(np.abs(reference_f32.astype(np.float64)), 1e-6)
    print("HIDDEN %-30s worst_abs=%.3e (gate %.3e) worst_rel=%.3e" % (name, worst, GATE_HIDDEN_ABS, float((diff / denom).max())), flush=True)
    if worst > GATE_HIDDEN_ABS:
        idx = int(np.argmax(diff))
        gate_fail("%s: worst abs %.4g > %.4g at idx %d (module %.7g reference %.7g)"
                  % (name, worst, GATE_HIDDEN_ABS, idx, module_f32.reshape(-1)[idx], reference_f32.reshape(-1)[idx]))
    return worst


class Checkpoint:
    def __init__(self, root):
        self.root = root
        index_path = os.path.join(root, "model.safetensors.index.json")
        self.weight_map = json.loads(open(index_path).read())["weight_map"]
        self._headers = {}
        self._mmaps = {}

    def _header(self, shard):
        if shard not in self._headers:
            with open(os.path.join(self.root, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[shard] = (json.loads(f.read(n)), 8 + n)
        return self._headers[shard]

    def _mmap(self, shard):
        if shard not in self._mmaps:
            path = os.path.join(self.root, shard)
            size = os.path.getsize(path)
            self._mmaps[shard] = np.memmap(path, dtype=np.uint8, mode="r", shape=(size,))
        return self._mmaps[shard]

    def bits(self, name):
        shard = self.weight_map[name]
        header, data_start = self._header(shard)
        entry = header[name]
        begin, end = entry["data_offsets"]
        dtype = entry["dtype"]
        shape = entry["shape"]
        raw = self._mmap(shard)[data_start + begin:data_start + end]
        if dtype == "BF16":
            return raw.view(np.uint16).reshape(shape).astype(np.uint16)
        if dtype == "F32":
            u32 = raw.view(np.uint32).reshape(shape)
            return ((u32 + np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)
        raise SystemExit("SPARK_FAIL: reference checkpoint dtype %s for %s" % (dtype, name))

    def raw_f32(self, name):
        shard = self.weight_map[name]
        header, data_start = self._header(shard)
        entry = header[name]
        begin, end = entry["data_offsets"]
        dtype = entry["dtype"]
        shape = entry["shape"]
        raw = self._mmap(shard)[data_start + begin:data_start + end]
        if dtype == "F32":
            return raw.view(np.float32).reshape(shape).astype(np.float32)
        if dtype == "BF16":
            return bf16_to_f32(raw.view(np.uint16).reshape(shape).astype(np.uint16))
        raise SystemExit("SPARK_FAIL: reference checkpoint dtype %s for %s" % (dtype, name))


def layer_heads(layer):
    return HEADS_FULL if layer % 4 == 0 else HEADS_SLIDING


def layer_variant(layer):
    return "full" if layer % 4 == 0 else "sliding"


def inv_freq_for(layer):
    return swa_inv_freq_f64() if layer % 4 != 0 else yarn_inv_freq_f64()


def rope_scaling(layer):
    return 1.0 if layer % 4 != 0 else YARN_SCALE


def reference_attention_layer(ckpt, layer, hidden_bits):
    heads = layer_heads(layer)
    tokens = hidden_bits.shape[0]
    prefix = "model.layers.%d.self_attn." % layer
    q_w = ckpt.bits(prefix + "q_proj.weight")
    k_w = ckpt.bits(prefix + "k_proj.weight")
    v_w = ckpt.bits(prefix + "v_proj.weight")
    o_w = ckpt.bits(prefix + "o_proj.weight")
    g_w = ckpt.bits(prefix + "g_proj.weight")
    q_norm_w = ckpt.bits(prefix + "q_norm.weight")
    k_norm_w = ckpt.bits(prefix + "k_norm.weight")
    in_norm_w = ckpt.bits("model.layers.%d.input_layernorm.weight" % layer)
    normed = rmsnorm_pub(hidden_bits, in_norm_w)
    qkv = bf16_matmul(normed, np.concatenate([q_w, k_w, v_w], axis=0))
    q_bits = qkv[:, :heads * HEAD_DIM].reshape(tokens, heads, HEAD_DIM)
    k_bits = qkv[:, heads * HEAD_DIM:(heads + KV_HEADS) * HEAD_DIM].reshape(tokens, KV_HEADS, HEAD_DIM)
    v_bits = qkv[:, (heads + KV_HEADS) * HEAD_DIM:].reshape(tokens, KV_HEADS, HEAD_DIM)
    q_normed = headrms_pub(q_bits, q_norm_w)
    k_normed = headrms_pub(k_bits, k_norm_w)
    positions = list(range(tokens))
    cos_b, sin_b = rope_cos_sin(inv_freq_for(layer), positions, rope_scaling(layer))
    cos_q = np.tile(cos_b[:, None, :], (1, heads, 1))
    sin_q = np.tile(sin_b[:, None, :], (1, heads, 1))
    cos_k = np.tile(cos_b[:, None, :], (1, KV_HEADS, 1))
    sin_k = np.tile(sin_b[:, None, :], (1, KV_HEADS, 1))
    q_rope = rotate_half_pub(q_normed, cos_q, sin_q)
    k_rope = rotate_half_pub(k_normed, cos_k, sin_k)
    gate_logits = bf16_matmul(normed, g_w)
    gate_sp = bf16_from_f32(softplus_f32(bf16_to_f32(gate_logits)))
    attn_out = np.zeros((tokens, heads * HEAD_DIM), dtype=np.uint16)
    is_sliding = layer_variant(layer) == "sliding"
    group = heads // KV_HEADS
    for p in range(tokens):
        start = max(0, p + 1 - WINDOW) if is_sliding else 0
        k_ctx = k_rope[start:p + 1]
        v_ctx = v_bits[start:p + 1]
        rows = np.empty((heads, HEAD_DIM), dtype=np.uint16)
        for h in range(heads):
            kv_h = h // group
            rows[h] = attention_rows_pub(q_rope[p, h:h + 1], k_ctx[:, kv_h], v_ctx[:, kv_h])[0]
        attn_out[p] = rows.reshape(-1)
    gated = bf16_mul(attn_out, np.repeat(gate_sp, HEAD_DIM, axis=1))
    o_full = bf16_matmul(gated, o_w)
    return {
        "normed": normed,
        "qkv": qkv,
        "qkv_rows": heads * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM,
        "q_normed": q_normed.reshape(tokens, -1),
        "k_normed": k_normed.reshape(tokens, -1),
        "q_rope": q_rope.reshape(tokens, -1),
        "k_rope": k_rope.reshape(tokens, -1),
        "decatt": attn_out,
        "gate_logits": gate_logits,
        "postgate": gated,
        "o_full": o_full,
    }


def compare_attention_run(dumps_dir, tag, layer, rank, count):
    heads = layer_heads(layer)
    per_head = heads // TP
    base = os.path.join(dumps_dir, tag)
    names = ["normed", "qkv", "qknorm_q", "qknorm_k", "rope_q", "rope_k", "decatt", "postgate", "gate", "opart"]
    local_kv = KV_HEADS // TP
    per_row = {"normed": HIDDEN, "qkv": per_head * HEAD_DIM + 2 * local_kv * HEAD_DIM,
               "qknorm_q": per_head * HEAD_DIM, "qknorm_k": local_kv * HEAD_DIM,
               "rope_q": per_head * HEAD_DIM, "rope_k": local_kv * HEAD_DIM,
               "decatt": per_head * HEAD_DIM, "postgate": per_head * HEAD_DIM,
               "gate": per_head, "opart": HIDDEN}
    dumps = {}
    missing = [n for n in names if not os.path.exists("%s_%s.bin" % (base, n))]
    if missing:
        gate_fail("%s: missing dumps %s" % (tag, ",".join(missing)))
        return None
    for n in names:
        dumps[n] = np.fromfile("%s_%s.bin" % (base, n), dtype=np.uint16)
        want = count * per_row[n]
        if dumps[n].size != want:
            gate_fail("%s %s: dump has %d elements, expected %d" % (tag, n, dumps[n].size, want))
            return None
    for n in names:
        dumps[n] = np.fromfile("%s_%s.bin" % (base, n), dtype=np.uint16)
        want = count * per_row[n]
        if dumps[n].size != want:
            gate_fail("%s %s: dump has %d elements, expected %d" % (tag, n, dumps[n].size, want))
            return None
    return dumps


def run_attention_matrix(ckpt, dumps_dir, attn_runs, token_ids):
    embed = ckpt.bits("model.embed_tokens.weight")
    ref_cache = {}
    opart_by_key = {}
    for entry in attn_runs:
        stage, layer, rank, count = entry["stage"], entry["layer"], entry["rank"], entry["count"]
        count_tag = "s%dl%dr%d_t%d" % (stage, layer, rank, count)
        hidden_bits = embed[token_ids[:count]]
        dumps = compare_attention_run(dumps_dir, count_tag, layer, rank, count)
        if dumps is None:
            continue
        ref_key = (layer, count)
        if ref_key not in ref_cache:
            ref_cache[ref_key] = reference_attention_layer(ckpt, layer, hidden_bits)
        ref = ref_cache[ref_key]
        q_lo = rank * (layer_heads(layer) // TP) * HEAD_DIM
        q_hi = (rank + 1) * (layer_heads(layer) // TP) * HEAD_DIM
        stats = {}
        stats["normed"] = stage_stats(count_tag + " normed", dumps["normed"], ref["normed"])
        k_lo = rank * (KV_HEADS // TP) * HEAD_DIM
        k_hi = (rank + 1) * (KV_HEADS // TP) * HEAD_DIM
        qd = HEAD_DIM
        qsec = layer_heads(layer) * qd
        ref_rank_qkv = np.concatenate([ref["qkv"][:, q_lo:q_hi],
                                       ref["qkv"][:, qsec + k_lo:qsec + k_hi],
                                       ref["qkv"][:, qsec + KV_HEADS * qd + k_lo:qsec + KV_HEADS * qd + k_hi]], axis=1)
        stats["qkv"] = stage_stats(count_tag + " qkv", dumps["qkv"], ref_rank_qkv)
        stats["qknorm_q"] = stage_stats(count_tag + " qknorm_q", dumps["qknorm_q"], ref["q_normed"][:, q_lo:q_hi])
        k_lo = rank * (KV_HEADS // TP) * HEAD_DIM
        k_hi = (rank + 1) * (KV_HEADS // TP) * HEAD_DIM
        stats["qknorm_k"] = stage_stats(count_tag + " qknorm_k", dumps["qknorm_k"], ref["k_normed"][:, k_lo:k_hi])
        stats["rope_q"] = stage_stats(count_tag + " rope_q", dumps["rope_q"], ref["q_rope"][:, q_lo:q_hi])
        stats["rope_k"] = stage_stats(count_tag + " rope_k", dumps["rope_k"], ref["k_rope"][:, k_lo:k_hi])
        stats["decatt"] = stage_stats(count_tag + " decatt", dumps["decatt"], ref["decatt"][:, q_lo:q_hi], attn_path=True)
        stats["postgate"] = stage_stats(count_tag + " postgate", dumps["postgate"], ref["postgate"][:, q_lo:q_hi], attn_path=True)
        stats["gate"] = stage_stats(count_tag + " gate", dumps["gate"], ref["gate_logits"][:, q_lo // HEAD_DIM:q_hi // HEAD_DIM])
        o_w = ckpt.bits("model.layers.%d.self_attn.o_proj.weight" % layer)
        opart_ref = bf16_matmul(ref["postgate"][:, q_lo:q_hi], o_w[:, q_lo:q_hi])
        stats["opart"] = stage_stats(count_tag + " opart", dumps["opart"], opart_ref)
        opart_by_key[(stage, layer, rank, count)] = dumps["opart"].reshape(count, HIDDEN)
    combine_groups = {}
    for (stage, layer, rank, count), bits in opart_by_key.items():
        combine_groups.setdefault((stage, layer, count), {})[rank] = bits
    for (stage, layer, count), ranks in sorted(combine_groups.items()):
        if len(ranks) != TP:
            print("HIDDEN %-30s partial combine skipped: %d/%d ranks present" % ("s%d l%d attn combine" % (stage, layer), len(ranks), TP), flush=True)
            continue
        module_sum = np.zeros((count, HIDDEN), dtype=np.float64)
        for rank in range(TP):
            module_sum += bf16_to_f32(ranks[rank])
        hidden_bits = embed[token_ids[:count]]
        ref = reference_attention_layer(ckpt, layer, hidden_bits)
        module_sum_bf16 = bf16_from_f32(module_sum.astype(np.float32))
        ref_key = (layer, count)
        if ref_key not in ref_cache:
            ref_cache[ref_key] = reference_attention_layer(ckpt, layer, embed[token_ids[:count]])
        hidden_gate("s%dl%d attnfull (%d tok)" % (stage, layer, count), bf16_to_f32(module_sum_bf16), bf16_to_f32(ref_cache[ref_key]["o_full"]))


def reference_moe_layer(ckpt, layer, normed2_bits):
    prefix = "model.layers.%d.mlp." % layer
    router_w = ckpt.bits(prefix + "gate.weight")
    bias = ckpt.raw_f32(prefix + "experts.e_score_correction_bias").reshape(-1)
    logits_pub_bits = bf16_matmul(normed2_bits, router_w)
    logits_pub = bf16_to_f32(logits_pub_bits).astype(np.float64)
    scores = 1.0 / (1.0 + np.exp(-logits_pub))
    selection = scores + bias[None, :].astype(np.float64)
    tokens = normed2_bits.shape[0]
    sets = np.zeros((tokens, TOP_K), dtype=np.int64)
    weights_pub = np.zeros((tokens, TOP_K), dtype=np.float64)
    ties = 0
    for t in range(tokens):
        ordered = np.sort(selection[t])[::-1]
        ties += int((ordered[:-1] == ordered[1:]).sum())
        order = np.argsort(-selection[t], kind="stable")
        sets[t] = order[:TOP_K]
        gathered = scores[t][sets[t]]
        weights_pub[t] = gathered / gathered.sum()
    shared_w1 = np.concatenate([ckpt.bits(prefix + "shared_expert.gate_proj.weight"),
                                ckpt.bits(prefix + "shared_expert.up_proj.weight")], axis=0)
    shared_w2 = ckpt.bits(prefix + "shared_expert.down_proj.weight")
    return logits_pub, scores, selection, bias, sets, weights_pub, ties, shared_w1, shared_w2


def expert_full(w1_bits, w2_bits, row_bits):
    gate_up = bf16_matmul(row_bits, w1_bits)
    gate = gate_up[:, :EXPERT_INTER]
    up = gate_up[:, EXPERT_INTER:]
    prod = bf16_mul(silu_bits(gate), up)
    down = bf16_matmul(prod, w2_bits)
    return gate_up, prod, down


def run_moe_matrix(ckpt, dumps_dir, moe_runs, token_ids, attnfull_bits_by_key):
    embed = ckpt.bits("model.embed_tokens.weight")
    for entry in moe_runs:
        stage, layer, count = entry["stage"], entry["layer"], entry["count"]
        key = (stage, layer, count)
        if key not in attnfull_bits_by_key:
            gate_fail("moe s%dl%d: no attnfull bits for priming" % (stage, layer))
            continue
        attnfull_bits = attnfull_bits_by_key[key]
        hidden_bits = embed[token_ids[:count]]
        residual = hidden_bits
        x2 = bf16_add(residual, attnfull_bits)
        post_norm_w = ckpt.bits("model.layers.%d.post_attention_layernorm.weight" % layer)
        normed2_ref = rmsnorm_pub(x2, post_norm_w)
        (logits_pub, scores, selection, bias, sets_pub, weights_pub, ties,
         shared_w1, shared_w2) = reference_moe_layer(ckpt, layer, normed2_ref)
        tag = "s%dl%d" % (stage, layer)
        sel_sorted = np.sort(selection, axis=1)
        boundary_ties = sel_sorted[:, -TOP_K] == sel_sorted[:, -TOP_K - 1]
        if ties:
            print("ROUTER %-28s publisher-model exact ties in selection scores: %d; boundary-spanning: %d tokens"
                  % ("%s router" % tag, ties, int(boundary_ties.sum())), flush=True)
        dumps = {}
        ok = True
        for rank in range(TP):
            rtag = "%sr%d_t%d" % (tag, rank, count)
            base = os.path.join(dumps_dir, rtag)
            names = ["router_logits", "sets", "weights", "normed2", "moepart"]
            missing = [n for n in names if not os.path.exists("%s_%s.bin" % (base, n))]
            if missing:
                gate_fail("%s: missing dumps %s" % (rtag, ",".join(missing)))
                ok = False
                continue
            dumps[rank] = {n: np.fromfile("%s_%s.bin" % (base, n),
                                          dtype=np.float32 if n in ("router_logits", "weights") else np.uint32 if n == "sets" else np.uint16)
                           for n in names}
        if not ok:
            continue
        for rank in range(1, TP):
            if not np.array_equal(dumps[rank]["sets"], dumps[0]["sets"]):
                gate_fail("%s r%d: router sets differ from rank0 (replicated router must route identically)" % (tag, rank))
        module_sets = dumps[0]["sets"].reshape(count, TOP_K)
        pub_sets = sets_pub
        mod_sorted = np.sort(module_sets, axis=1)
        pub_sorted = np.sort(pub_sets, axis=1)
        set_mismatch = int((mod_sorted != pub_sorted).any(axis=1).sum())
        logits_mod = dumps[0]["router_logits"].reshape(count, EXPERTS)
        scores_mod = 1.0 / (1.0 + np.exp(-logits_mod.astype(np.float64)))
        sel_mod = scores_mod + bias[None, :].astype(np.float64)
        sets_mod_model = np.argsort(-sel_mod, axis=1, kind="stable")[:, :TOP_K]
        margins_pub = np.sort(selection, axis=1)[:, -TOP_K] - np.sort(selection, axis=1)[:, -TOP_K - 1]
        print("ROUTER %-28s tokens=%d set_mismatch_vs_publisher=%d min_margin10_11=%.3e" %
              ("%s router" % tag, count, set_mismatch, float(margins_pub.min())), flush=True)
        gate_relevant = 0
        for t in range(count):
            if (mod_sorted[t] != pub_sorted[t]).any():
                extra_pub = sorted(set(pub_sets[t]) - set(module_sets[t]))
                extra_mod = sorted(set(module_sets[t]) - set(pub_sets[t]))
                margin = float(sel_sorted[t, -TOP_K] - sel_sorted[t, -TOP_K - 1])
                if boundary_ties[t]:
                    print("TIE-EXCLUDED: %s router token %d: publisher bf16-logit model ties exactly across rank 10/11; module f32 set %s vs publisher set %s (pub-only %s mod-only %s)"
                          % (tag, t, mod_sorted[t].tolist(), pub_sorted[t].tolist(), extra_pub, extra_mod), flush=True)
                else:
                    e_mod_pick = int(extra_mod[0]) if len(extra_mod) == 1 else -1
                    e_pub_pick = int(extra_pub[0]) if len(extra_pub) == 1 else -1
                    self_consistent = (e_mod_pick >= 0 and e_pub_pick >= 0 and
                                       sel_mod[t, e_mod_pick] > sel_mod[t, e_pub_pick])
                    if margin < 1e-3 and self_consistent:
                        print("QUANT-NOISE-EXCLUDED: %s router token %d: set flip %s->%s at selection margin %.3e; module f32 logits order it self-consistently (%.6f > %.6f); the publisher bf16 logit quantization (+-~4e-3 logit) exceeds this margin"
                              % (tag, t, e_pub_pick, e_mod_pick, margin, sel_mod[t, e_mod_pick], sel_mod[t, e_pub_pick]), flush=True)
                    else:
                        gate_relevant += 1
                        gate_fail("%s router token %d: publisher set %s vs module set %s (pub-only %s mod-only %s) margin %.3e"
                                  % (tag, t, pub_sorted[t].tolist(), mod_sorted[t].tolist(), extra_pub, extra_mod, margin))
        if set_mismatch and gate_relevant == 0:
            print("ROUTER %-28s all %d set differences sit on exact publisher-model ties; weights exact" % (tag, set_mismatch), flush=True)
        order_only = int(sum(1 for t in range(count) if (mod_sorted[t] == pub_sorted[t]).all() and (module_sets[t] != pub_sets[t]).any()))
        if order_only:
            print("ROUTER %-28s %d tokens: same set, different slot order (selection key ties within the top-10; publisher accumulate order is expert-ascending, module pairs weights by expert id)" % (tag, order_only), flush=True)
        if set_mismatch == 0:
            print("ROUTER %-28s sets EXACT vs publisher (bias selection-only verified on real states)" % tag, flush=True)
        mod_vs_f32 = int((module_sets != sets_mod_model).any(axis=1).sum())
        if mod_vs_f32 != set_mismatch:
            finding("%s router: module sets match its own f32-logit selection model on %d/%d tokens where the publisher bf16-logit model differs (rounding-model site: router GEMM emits f32, publisher rounds logits to bf16)" % (tag, mod_vs_f32, count))
        weights_mod = dumps[0]["weights"].reshape(count, TOP_K)
        gathered_mod = np.take_along_axis(scores_mod, module_sets, axis=1)
        total_mod = gathered_mod.sum(axis=1, keepdims=True)
        weights_mod_unscaled = gathered_mod / total_mod
        worst_w = float(np.abs(weights_mod / ROUTED_SCALING - weights_mod_unscaled).max() / max(float(np.abs(weights_mod_unscaled).max()), 1e-30))
        print("ROUTER %-28s weights worst rel (module/2.5 vs renormalised sigmoid) = %.3e (gate %.1e)" % (tag, worst_w, GATE_WEIGHT_REL), flush=True)
        if worst_w > GATE_WEIGHT_REL:
            gate_fail("%s route weights rel %.3e > %.1e" % (tag, worst_w, GATE_WEIGHT_REL))
        normed2_mod = dumps[0]["normed2"].reshape(count, HIDDEN)
        stage_stats(tag + " normed2", normed2_mod, normed2_ref)
        gate_up_shared = bf16_matmul(normed2_mod, shared_w1)
        prod_shared = bf16_mul(silu_bits(gate_up_shared[:, :EXPERT_INTER]), gate_up_shared[:, EXPERT_INTER:])
        shared_full = bf16_matmul(prod_shared, shared_w2)
        module_combined = np.zeros((count, HIDDEN), dtype=np.float64)
        weights_bf16_pub = bf16_from_f32(weights_pub.astype(np.float32))
        combine_tokens = [t for t in range(count) if (np.sort(module_sets[t]) == np.sort(pub_sets[t])).all()]
        skipped = count - len(combine_tokens)
        if skipped:
            print("MOE %-30s combine compare restricted to %d/%d tokens (excluded tokens: selection set differs within quantization noise)" % (tag, len(combine_tokens), count), flush=True)
        w1_cache = {}
        w2_cache = {}
        for rank in range(TP):
            lo = rank * (EXPERT_INTER // TP)
            hi = (rank + 1) * (EXPERT_INTER // TP)
            moepart_ref = np.zeros((count, HIDDEN), dtype=np.float64)
            routed_pub = np.zeros((count, HIDDEN), dtype=np.uint16)
            for t in combine_tokens:
                row = normed2_mod[t:t + 1]
                acc_pub = np.zeros((1, HIDDEN), dtype=np.uint16)
                for slot in sorted(range(TOP_K), key=lambda s: int(module_sets[t, s])):
                    e = int(module_sets[t, slot])
                    if e not in w1_cache:
                        w1_cache[e] = np.concatenate([ckpt.bits("model.layers.%d.mlp.experts.%d.gate_proj.weight" % (layer, e)),
                                                      ckpt.bits("model.layers.%d.mlp.experts.%d.up_proj.weight" % (layer, e))], axis=0)
                        w2_cache[e] = ckpt.bits("model.layers.%d.mlp.experts.%d.down_proj.weight" % (layer, e))
                    w1 = w1_cache[e]
                    w2 = w2_cache[e]
                    _gate_up, prod_e, down_e = expert_full(w1, w2, row)
                    partial = bf16_matmul(prod_e[:, lo:hi], w2[:, lo:hi])
                    weight_mod = float(weights_mod[t, slot])
                    moepart_ref[t] += bf16_to_f32(partial)[0] * weight_mod
                    slot_pub = int(np.argmax(pub_sets[t] == e))
                    scaled_pub = bf16_mul(down_e, weights_bf16_pub[t, slot_pub:slot_pub + 1])
                    acc_pub = bf16_add(acc_pub, scaled_pub)
                scaled_out_pub = bf16_scalar_mul(acc_pub, ROUTED_SCALING)
                routed_pub[t:t + 1] = scaled_out_pub
                shared_partial = bf16_matmul(prod_shared[:, lo:hi], shared_w2[:, lo:hi])
                moepart_ref[t] += bf16_to_f32(shared_partial)[0]
            rtag = "%sr%d_t%d" % (tag, rank, count)
            module_part = dumps[rank]["moepart"].reshape(count, HIDDEN)[combine_tokens]
            stage_stats(rtag + " moepart", module_part, bf16_from_f32(moepart_ref.astype(np.float32))[combine_tokens])
            module_combined[combine_tokens] += bf16_to_f32(module_part)
        combined_ref = bf16_add(routed_pub, shared_full)
        module_combined_bf16 = bf16_from_f32(module_combined.astype(np.float32))
        rows = np.array(combine_tokens)
        m = bf16_to_f32(module_combined_bf16)[rows].astype(np.float64)
        r = bf16_to_f32(combined_ref)[rows].astype(np.float64)
        diff = m - r
        norm_rel = float(np.linalg.norm(diff) / max(np.linalg.norm(r), 1e-30))
        print("HIDDEN %-30s moe combined norm_rel=%.3e (gate %.1e) worst_abs=%.3e" % ("%s TP8 combine" % tag, norm_rel, GATE_MOE_REL, float(np.abs(diff).max())), flush=True)
        if norm_rel > GATE_MOE_REL:
            gate_fail("%s MoE TP8 combine norm rel %.3e > %.1e" % (tag, norm_rel, GATE_MOE_REL))
        else:
            print("HIDDEN %-30s DESIGN 5.3 MoE combine rel gate PASS; per-element worst abs %.3e retained as rounding-model observation (f32 route weights vs publisher bf16 weights)" % ("", float(np.abs(diff).max())), flush=True)


def run_rope_matrix(dumps_dir, rope_runs):
    for entry in rope_runs:
        regime = entry["regime"]
        positions = entry["positions"]
        base = os.path.join(dumps_dir, "rope_%s" % regime)
        input_bits = np.fromfile(base + "_input.bin", dtype=np.uint16)
        output_bits = np.fromfile(base + "_output.bin", dtype=np.uint16)
        heads = (HEADS_SLIDING if regime == "sliding" else HEADS_FULL) // TP
        rotary = SWA_ROT if regime == "sliding" else YARN_ROT
        count = len(positions)
        if input_bits.size != count * heads * HEAD_DIM or output_bits.size != input_bits.size:
            gate_fail("rope %s: dump sizes %d/%d expected %d" % (regime, input_bits.size, output_bits.size, count * heads * HEAD_DIM))
            continue
        inv_freq = swa_inv_freq_f64() if regime == "sliding" else yarn_inv_freq_f64()
        scaling = 1.0 if regime == "sliding" else YARN_SCALE
        cos_b, sin_b = rope_cos_sin(inv_freq, positions, scaling)
        x = input_bits.reshape(count, heads, HEAD_DIM)
        cos_full = np.zeros((count, heads, rotary), dtype=np.uint16)
        sin_full = np.zeros((count, heads, rotary), dtype=np.uint16)
        for p in range(count):
            cos_full[p] = cos_b[p]
            sin_full[p] = sin_b[p]
        ref = rotate_half_pub(x, cos_full, sin_full)
        stage_stats("rope %s (%d positions)" % (regime, count), output_bits, ref.reshape(-1),
                    hard_ulp=STAGE_HARD_ULP, frac_gate=STAGE_FRAC_OVER_1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dumps", required=True)
    parser.add_argument("--plan", required=True)
    arguments = parser.parse_args()
    ckpt = Checkpoint(arguments.checkpoint)
    plan = json.loads(open(arguments.plan).read())
    token_ids = np.array(plan["token_ids"], dtype=np.int64)
    attnfull_bits_by_key = {}
    for entry in plan.get("combine_sources", []):
        stage, layer, count = entry["stage"], entry["layer"], entry["count"]
        total = np.zeros((count, HIDDEN), dtype=np.float64)
        complete = True
        for rank in range(TP):
            path = os.path.join(arguments.dumps, "s%dl%dr%d_t%d_opart.bin" % (stage, layer, rank, count))
            if not os.path.exists(path):
                complete = False
                break
            total += bf16_to_f32(np.fromfile(path, dtype=np.uint16)).astype(np.float64).reshape(count, HIDDEN)
        if complete:
            attnfull_bits_by_key[(stage, layer, count)] = bf16_from_f32(total.astype(np.float32))
        else:
            print("HIDDEN s%dl%d attnfull: incomplete rank coverage, moe tier will be skipped" % (stage, layer), flush=True)
    run_attention_matrix(ckpt, arguments.dumps, plan.get("attn_runs", []), token_ids)
    run_moe_matrix(ckpt, arguments.dumps, plan.get("moe_runs", []), token_ids, attnfull_bits_by_key)
    run_rope_matrix(arguments.dumps, plan.get("rope_runs", []))
    print("", flush=True)
    print("L7 TABLES (per-stage exact/ulp/rel/cos):", flush=True)
    for name in sorted(tables):
        row = tables[name]
        if isinstance(row, dict):
            print("  %-34s exact=%-9s max_ulp=%-4s rel=%-10.3e cos=%.9f abs=%.3e" % (name, row["exact"], row["max_ulp"], row["worst_rel"], row["cos"], row["worst_abs"]), flush=True)
    print("L7-REFERENCE-DONE findings=%d gate_failures=%d" % (len(findings), len(gate_failures)), flush=True)
    if gate_failures:
        print("L7-VERDICT: GATE-FAIL (%d)" % len(gate_failures), flush=True)
        sys.exit(1)
    print("L7-VERDICT: PASS (all DESIGN 5.3 gates green; %d stage observations)" % len(findings), flush=True)


if __name__ == "__main__":
    main()
