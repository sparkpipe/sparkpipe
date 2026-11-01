#!/usr/bin/env python3
"""Teacher-forced numpy reference for DeepSeek V4.1 Flash (the FP8/FP4 release) for the Metal gate.

  gate-prompts  the Metal gate prompts (tools/gemma4_reference.py's set, each
             cut to 496 tokens: the stage refuses positions past 511 and the
             gate decodes 8 greedy tokens after the prompt) in the DeepSeek
             chat turn (thinking off) -> --out prompts.json
  teacher    FP32 logits of every prompt position but the last:
             --prompts prompts.json -> --out DIR/<name>.ref_<v>.logits for
             --variants a,b, through tools/ling3_reference.py's teacher: a
             multiplies in numpy's order; b splits every matmul's inner
             dimension into 1,024-wide chunks summed last chunk first,
             reverses the other reductions (norms, mixes, softmax, Sinkhorn
             and engram sums, the route weight sum, the stream sums) and
             combines the routed experts in reverse pick order, so a against
             b is the FP32 reduction-order floor

Every prompt is its own sequence at positions 0..n-1. A position at or past
index_topk (512) is refused: below it the sparse indexer keeps every
compressed key, so attention is dense. The rounding points follow the Metal
and CUDA stages (model-families/dsv41_flash): bf16() rounds to nearest even,
everything else is FP32.
  qdq8       blocks of 32: scale = the power of two at or above max(amax,
             1e-4) * (1/448), q = E4M3 round to nearest even of x / scale
             clamped to +-448, bf16(q scale)
  qdq4       blocks of 16: scale = E4M3(max(amax, 6 * 2^-9) / 6), q = E2M1
             round to nearest even of x / scale clamped to +-6, bf16(q scale)
  norms      bf16(x rsqrt(mean(x^2) + eps) w), BF16 gains widened
  weights    FP8 e4m3 x 2^(e - 127) of its 32 x 32 block, MXFP4 experts e2m1
             (low nibble the even column) x 2^(e - 127) of its 32 columns,
             BF16 widened; a projection is bf16(x W^T)
  hyper-connections  the embedding row in 4 BF16 streams; per site r =
             rsqrt(mean(x^2) + eps) over all 4H values, 24 mixes m = (fn . x)
             r, pre = sigmoid(m s0 + b) + 1e-6, post = 2 sigmoid(m s1 + b),
             the 4x4 combination softmax per row + 1e-6, then 20 Sinkhorn
             passes (a column normalisation, from the second pass a row one
             before it, each + 1e-6); x_t <- bf16(post_t out + sum_s C_st
             x_s). The collapse bf16(sum_s pre_s x_s) is lagged: the
             attention branch reads it with the previous layer's FFN pre
             (first [1, 0, 0, 0]), the FFN branch with the attention site's
             pre, the head with the last FFN pre
  engram     layers 1 and 14, before the attention mixes: the sequence's
             last 4 mapped tokens (the pad from the first shift before
             position 0 or a blocked token on) hashed with the multipliers,
             primes and offsets of tools/dsv41_flash_engram_tables.py, the 24
             rows bf16(e4m3 x 2^(e - 127)), qdq8, wkv into 4 keys and a
             value; per stream gate = sigmoid(signed sqrt of max(|d|, 1e-6))
             with d = (x w . k) rsqrt(mean x^2) rsqrt(mean k^2) / sqrt(H), w
             = q_weight k_weight, x <- bf16(x + gate v)
  attention  64 heads of 512 over one shared 512-wide key that is also the
             value: q = wq_b(qdq8(norm(wq_a(xq)))), the window key
             qdq8(rope(norm(wkv(xq)))), rope on the last 64 dims in
             interleaved pairs, each output rounded (ratio-0 layers theta
             10000, the others SparkRopeYarnRampTable's YaRN table); the
             compressing layers (2, 8, 14 at ratio 2, 20 at ratio 1) pool the
             normed input's FP32 wkv by the softmax of wgate per channel over
             each pair, bf16, norm, rope at the pair's first position, qdq4;
             a row at p attends its window keys max(0, p - 127)..p and its
             owner layer's compressed keys 0..(p + 1) / ratio - 1,
             softmax(q.k / sqrt(512)) with the head's sink in the
             denominator, bf16, inverse rope, qdq8, wo_a per output group,
             qdq8, wo_b
  MoE        router z = normed . gate (FP32), s = sqrt(softplus(z)), the top
             6 of s + bias (ties to the lower index), weights s / (tree sum +
             1e-20) x 1.5; each expert from xq: bf16(silu(min(g, 10))
             clamp(u, +-10) w), qdq8, w2; the shared FP8 expert likewise
             without w; bf16(sum of the expert outputs in pick order +
             shared)
"""

import argparse
import json
import math
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ling3_reference as base  # noqa: E402
from dsv41_flash_engram_tables import tables as engram_tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GATE_TOKENS = 496
BLOCK = 32
FP4_BLOCK = 16
EXPERT_WINDOW = 8
ENGRAM_THREADS = 16
BF16 = ("BF16",)
F32 = ("F32",)
FP8 = ("F8_E4M3",)
SCALE = ("F8_E8M0", "U8")
PACKED = ("I8", "U8")
DTYPES = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "U8": np.uint8, "I8": np.uint8}
ONE = np.float32(1.0)
FP8_MAX = np.float32(448.0)
FP8_INVERSE = ONE / FP8_MAX
FP8_FLOOR = np.float32(1e-4)
FP4_MAX = np.float32(6.0)
FP4_FLOOR = FP4_MAX * np.float32(2.0 ** -9)
POST_GAIN = np.float32(2.0)
GATE_FLOOR = np.float32(1e-6)
SOFTPLUS_LINEAR = np.float32(20.0)
ROUTE_EPSILON = np.float32(1e-20)


def e4m3_table():
    codes = np.arange(256, dtype=np.uint32)
    exponent, mantissa = (codes >> np.uint32(3)) & np.uint32(15), codes & np.uint32(7)
    normal = ((exponent + np.uint32(120)) << np.uint32(23)) | (mantissa << np.uint32(20))
    subnormal = (mantissa.astype(np.float32) * np.float32(2.0 ** -9)).view(np.uint32)
    bits = (np.where(exponent == 0, subnormal, normal) | ((codes & np.uint32(0x80)) << np.uint32(24))).astype(np.uint32)
    table = bits.view(np.float32).copy()
    table[(codes & np.uint32(0x7F)) == np.uint32(0x7F)] = np.nan
    return table


def ue8m0_table():
    codes = np.arange(256, dtype=np.uint32)
    table = np.where(codes == 0, np.uint32(0x00400000), codes << np.uint32(23)).astype(np.uint32).view(np.float32).copy()
    table[255] = np.nan
    return table


E4M3 = e4m3_table()
UE8M0 = ue8m0_table()
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
E2M1_PAIRS = np.stack([E2M1[np.arange(256) & 15], E2M1[np.arange(256) >> 4]], axis=-1)


def widen(u16):
    return (np.asarray(u16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def rsqrt(values):
    return (1.0 / np.sqrt(np.asarray(values, dtype=np.float64))).astype(np.float32)


def pow2_ceil(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    exponent = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32) - 127 + ((bits & np.uint32(0x7FFFFF)) != 0)
    return ((exponent + 127).astype(np.uint32) << np.uint32(23)).view(np.float32)


def round_minifloat(values, mantissa_bits, minimum_exponent, maximum):
    values = np.ascontiguousarray(values, dtype=np.float32)
    exponent = np.maximum(((values.view(np.uint32) >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32) - 127, minimum_exponent)
    step = ((exponent - mantissa_bits + 127).astype(np.uint32) << np.uint32(23)).view(np.float32)
    inverse = ((mantissa_bits - exponent + 127).astype(np.uint32) << np.uint32(23)).view(np.float32)
    return np.clip(np.rint(values * inverse) * step, -maximum, maximum)


def round_e4m3(values):
    return round_minifloat(values, 3, -6, FP8_MAX)


def round_e2m1(values):
    return round_minifloat(values, 1, 0, FP4_MAX)


def blocks_of(values, size):
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.shape[-1] % size:
        raise ValueError(f"a {values.shape[-1]}-wide row does not split into {size}-wide quantisation blocks")
    return values.reshape(-1, size)


def qdq8(values):
    blocks = blocks_of(values, BLOCK)
    scale = pow2_ceil(np.maximum(np.abs(blocks).max(axis=1, keepdims=True), FP8_FLOOR) * FP8_INVERSE)
    return base.bf16(round_e4m3(np.clip(blocks / scale, -FP8_MAX, FP8_MAX)) * scale).reshape(np.shape(values))


def qdq4(values):
    blocks = blocks_of(values, FP4_BLOCK)
    scale = round_e4m3(np.maximum(np.abs(blocks).max(axis=1, keepdims=True), FP4_FLOOR) / FP4_MAX)
    return base.bf16(round_e2m1(np.clip(blocks / scale, -FP4_MAX, FP4_MAX)) * scale).reshape(np.shape(values))


def tree_sum(values):
    values = values[:, ::-1] if base.REVERSED else values
    width = 1
    while width < values.shape[1]:
        width *= 2
    tree = np.zeros((values.shape[0], width), dtype=np.float32)
    tree[:, :values.shape[1]] = values
    while width > 1:
        width //= 2
        tree = tree[:, :width] + tree[:, width:2 * width]
    return tree


def powf(value, exponent):
    return np.float32(math.pow(float(np.float32(value)), float(np.float32(exponent))))


def plain_table(rope, theta):
    return np.array([ONE / powf(theta, np.float32(2 * pair) / np.float32(rope)) for pair in range(rope // 2)], dtype=np.float32)


def yarn_table(rope, theta, factor, beta_fast, beta_slow, original):
    theta, factor = float(np.float32(theta)), np.float32(factor)
    low = float(math.floor(rope * math.log(float(original) / (float(np.float32(beta_fast)) * 2.0 * math.pi)) / (2.0 * math.log(theta))))
    high = float(math.ceil(rope * math.log(float(original) / (float(np.float32(beta_slow)) * 2.0 * math.pi)) / (2.0 * math.log(theta))))
    low, high = max(low, 0.0), min(high, float(rope - 1))
    if low == high:
        high += 0.001
    span = np.float32(high) - np.float32(low)
    table = []
    for index in range(rope // 2):
        frequency = powf(theta, np.float32(2 * index) / np.float32(rope))
        extrapolation, interpolation = ONE / frequency, ONE / (factor * frequency)
        ramp = min(max((np.float32(index) - np.float32(low)) / span, np.float32(0.0)), ONE)
        keep = ONE - ramp
        table.append(interpolation * (ONE - keep) + extrapolation * keep)
    return np.array(table, dtype=np.float32)


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.weight_map = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

    def meta(self, name):
        shard = self.weight_map.get(name)
        if shard is None:
            raise ValueError(f"checkpoint has no tensor {name}")
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        base_offset, header = self.headers[shard]
        if name not in header:
            raise ValueError(f"{shard} has no tensor {name}")
        return shard, base_offset, header[name]

    def check(self, name, dtypes, shape):
        shard, base_offset, meta = self.meta(name)
        if meta["dtype"] not in dtypes or tuple(meta["shape"]) != tuple(shape):
            raise ValueError(f"{name} is {meta['dtype']} {meta['shape']}, the reference expects {'/'.join(dtypes)} {list(shape)}")
        begin, end = meta["data_offsets"]
        kind = np.dtype(DTYPES[meta["dtype"]])
        if end - begin != int(np.prod(shape, dtype=np.int64)) * kind.itemsize:
            raise ValueError(f"{name} holds {end - begin} bytes, not {list(shape)} of {meta['dtype']}")
        return self.directory / shard, base_offset + begin, kind

    def raw(self, name, dtypes, shape):
        path, offset, kind = self.check(name, dtypes, shape)
        return np.memmap(path, dtype=kind, mode="r", offset=offset, shape=tuple(shape))


class Reference:
    def __init__(self, directory, token_map):
        self.checkpoint = Checkpoint(directory)
        config = json.loads((Path(directory) / "config.json").read_text())
        text = config.get("text_config", config)
        self.hidden = int(text["hidden_size"])
        self.layers = int(text["num_hidden_layers"])
        self.vocab = int(text["vocab_size"])
        self.epsilon = np.float32(text["rms_norm_eps"])
        self.heads = int(text["num_attention_heads"])
        self.head_dim = int(text["head_dim"])
        self.rope_dim = int(text["qk_rope_head_dim"])
        self.query_lora = int(text["q_lora_rank"])
        self.output_lora = int(text["o_lora_rank"])
        self.groups = int(text["o_groups"])
        self.window = int(text["sliding_window"])
        self.limit = np.float32(text["swiglu_limit"])
        self.experts = int(text["n_routed_experts"])
        self.top_k = int(text["num_experts_per_tok"])
        self.moe_inter = int(text["moe_intermediate_size"])
        self.route_scale = np.float32(text["routed_scaling_factor"])
        self.streams = int(text["hc_mult"])
        self.mixes = (2 + self.streams) * self.streams
        self.sinkhorn = int(text["hc_sinkhorn_iters"])
        self.hc_epsilon = np.float32(text["hc_eps"])
        self.dense_keys = int(text["index_topk"])
        self.ratios = [int(ratio) for ratio in text["compress_ratios"][:self.layers]]
        self.sources = sorted(int(layer) for layer in text["kv_source_layer_ids"])
        if (len(self.ratios) != self.layers or any(ratio not in (0, 1, 2) for ratio in self.ratios)
                or any(layer >= self.layers or self.ratios[layer] == 0 for layer in self.sources)):
            raise ValueError("the reference implements compress ratios 0, 1 and 2 on every layer, with compressing source layers")
        self.owners = {}
        for layer, ratio in enumerate(self.ratios):
            owners = [source for source in self.sources if source <= layer]
            if ratio and (not owners or self.ratios[owners[-1]] != ratio):
                raise ValueError(f"layer {layer} (ratio {ratio}) has no compressing source layer of its ratio at or before it")
            if ratio:
                self.owners[layer] = owners[-1]
        if self.heads * self.head_dim % self.groups or self.rope_dim % 2 or self.rope_dim > self.head_dim or self.top_k > self.experts:
            raise ValueError("attention groups, rope or top-k geometry the reference does not implement")
        self.group_input = self.heads * self.head_dim // self.groups
        scaling = text["rope_scaling"]
        self.plain_rope = plain_table(self.rope_dim, text["rope_theta"])
        self.yarn_rope = yarn_table(self.rope_dim, text["compress_rope_theta"], scaling["factor"], scaling["beta_fast"], scaling["beta_slow"],
                                    scaling["original_max_position_embeddings"])
        self.token_map = np.load(token_map).astype(np.int64)
        derived = engram_tables(text, self.token_map)
        self.engram_layers = [int(layer) for layer in derived["layers"]]
        self.engram_rows = [int(rows) for rows in derived["rows"]]
        self.orders, self.engram_heads, self.pad = int(derived["orders"]), int(derived["heads"]), int(derived["pad"])
        self.multipliers = np.array(derived["multipliers"], dtype=np.int64)
        self.primes = np.array(derived["primes"], dtype=np.int64)
        self.offsets = np.array(derived["offsets"], dtype=np.int64)
        self.engram_dim = int(text["engram_head_dim"])
        self.engram_embed = self.orders * self.engram_heads * self.engram_dim
        if self.pad < 0 or any(layer >= self.layers for layer in self.engram_layers):
            raise ValueError("the engram pad maps to a blocked token or an engram layer lies past the last layer")
        self.head = None
        self.verify()

    def fp8_entries(self, name, rows, columns):
        return [(name + ".weight", FP8, (rows, columns)), (name + ".scale", SCALE, (rows // BLOCK, columns // BLOCK))]

    def fp4_entries(self, name, rows, columns):
        return [(name + ".weight", PACKED, (rows, columns // 2)), (name + ".scale", SCALE, (rows, columns // BLOCK))]

    def expert_shapes(self):
        return (("w1", self.moe_inter, self.hidden), ("w3", self.moe_inter, self.hidden), ("w2", self.hidden, self.moe_inter))

    def attention_shapes(self):
        return (("wq_a", self.query_lora, self.hidden), ("wq_b", self.heads * self.head_dim, self.query_lora), ("wkv", self.head_dim, self.hidden),
                ("wo_a", self.groups * self.output_lora, self.group_input), ("wo_b", self.hidden, self.groups * self.output_lora))

    def layer_entries(self, layer):
        p, a, f = f"layers.{layer}.", f"layers.{layer}.attn.", f"layers.{layer}.ffn."
        entries = [(p + "attn_norm.weight", BF16, (self.hidden,)), (p + "ffn_norm.weight", BF16, (self.hidden,))]
        for site in ("attn", "ffn"):
            entries += [(f"{p}hc_{site}_fn", F32, (self.mixes, self.streams * self.hidden)), (f"{p}hc_{site}_base", F32, (self.mixes,)),
                        (f"{p}hc_{site}_scale", F32, (3,))]
        for name, rows, columns in self.attention_shapes():
            entries += self.fp8_entries(a + name, rows, columns)
        entries += [(a + "q_norm.weight", BF16, (self.query_lora,)), (a + "kv_norm.weight", BF16, (self.head_dim,)), (a + "attn_sink", F32, (self.heads,))]
        if layer in self.sources:
            entries.append((a + "compressor.wkv.weight", BF16, (self.head_dim, self.hidden)))
            if self.ratios[layer] == 2:
                entries.append((a + "compressor.wgate.weight", BF16, (self.head_dim, self.hidden)))
            entries.append((a + "compressor.norm.weight", BF16, (self.head_dim,)))
        entries += [(f + "gate.weight", BF16, (self.experts, self.hidden)), (f + "gate.bias", F32, (self.experts,))]
        for expert in range(self.experts):
            for name, rows, columns in self.expert_shapes():
                entries += self.fp4_entries(f"{f}experts.{expert}.{name}", rows, columns)
        for name, rows, columns in self.expert_shapes():
            entries += self.fp8_entries(f"{f}shared_experts.{name}", rows, columns)
        if layer in self.engram_layers:
            e, rows = p + "engram.", self.engram_rows[self.engram_layers.index(layer)]
            entries += [(e + "embed.weight", FP8, (rows, self.engram_dim)), (e + "embed.scale", SCALE, (rows, self.engram_dim // BLOCK))]
            entries += self.fp8_entries(e + "wkv", (self.streams + 1) * self.hidden, self.engram_embed)
            entries += [(e + "q_weight", BF16, (self.streams, self.hidden)), (e + "k_weight", BF16, (self.streams, self.hidden))]
        return entries

    def verify(self):
        began = time.time()
        entries = [("embed.weight", BF16, (self.vocab, self.hidden)), ("head.weight", BF16, (self.vocab, self.hidden)), ("norm.weight", BF16, (self.hidden,))]
        for layer in range(self.layers):
            entries += self.layer_entries(layer)
        for name, dtypes, shape in entries:
            self.checkpoint.check(name, dtypes, shape)
        print(f"{len(entries)} checkpoint tensors match the reference's dtypes and shapes ({time.time() - began:.1f}s)", file=sys.stderr, flush=True)

    def widened(self, name, *shape):
        return widen(self.checkpoint.raw(name, BF16, shape))

    def f32(self, name, *shape):
        return np.array(self.checkpoint.raw(name, F32, shape), dtype=np.float32)

    def fp8(self, name, rows, columns):
        codes, scale = (np.asarray(self.checkpoint.raw(*entry)) for entry in self.fp8_entries(name, rows, columns))
        weight = E4M3[codes].reshape(rows // BLOCK, BLOCK, columns // BLOCK, BLOCK)
        weight *= UE8M0[scale][:, None, :, None]
        return weight.reshape(rows, columns)

    def fp4(self, name, rows, columns):
        packed, scale = (np.asarray(self.checkpoint.raw(*entry)) for entry in self.fp4_entries(name, rows, columns))
        weight = E2M1_PAIRS[packed].reshape(rows, columns // BLOCK, BLOCK)
        weight *= UE8M0[scale][:, :, None]
        return weight.reshape(rows, columns)

    def project(self, values, weight):
        return base.bf16(base.matmul(values, weight.T))

    def norm(self, values, weight):
        inverse = rsqrt(base.reduce_sum(values * values) / np.float32(values.shape[-1]) + self.epsilon)
        return base.bf16(values * inverse * weight)

    def rope(self, values, positions, table, inverse=False):
        angles = np.asarray(positions).astype(np.float32)[:, None] * table[None, :]
        angles = -angles if inverse else angles
        shape = (values.shape[0],) + (1,) * (values.ndim - 2) + (table.shape[0],)
        cos = np.cos(angles.astype(np.float64)).astype(np.float32).reshape(shape)
        sin = np.sin(angles.astype(np.float64)).astype(np.float32).reshape(shape)
        real, imag = values[..., 0::2], values[..., 1::2]
        rotated = np.empty(values.shape, dtype=np.float32)
        rotated[..., 0::2] = base.bf16(real * cos - imag * sin)
        rotated[..., 1::2] = base.bf16(real * sin + imag * cos)
        return rotated

    def hc_site(self, streams, prefix, site):
        rows, n = streams.shape[0], self.streams
        fn = self.f32(f"{prefix}hc_{site}_fn", self.mixes, n * self.hidden)
        bias = self.f32(f"{prefix}hc_{site}_base", self.mixes)
        scale = self.f32(f"{prefix}hc_{site}_scale", 3)
        flat = streams.reshape(rows, -1)
        mixes = base.matmul(flat, fn.T) * rsqrt(base.reduce_sum(flat * flat) / np.float32(flat.shape[1]) + self.epsilon)
        pre = base.sigmoid(mixes[:, :n] * scale[0] + bias[:n]) + self.hc_epsilon
        post = POST_GAIN * base.sigmoid(mixes[:, n:2 * n] * scale[1] + bias[n:2 * n])
        logits = mixes[:, 2 * n:].reshape(rows, n, n) * scale[2] + bias[2 * n:].reshape(n, n)
        comb = np.exp(logits - logits.max(axis=-1, keepdims=True))
        comb = comb / base.reduce_sum(comb, axis=-1) + self.hc_epsilon
        for iteration in range(self.sinkhorn):
            if iteration != 0:
                comb = comb / (base.reduce_sum(comb, axis=-1) + self.hc_epsilon)
            comb = comb / (base.reduce_sum(comb, axis=-2) + self.hc_epsilon)
        return pre, post, comb

    def collapse(self, streams, weights):
        return base.bf16(base.reduce_sum(weights[:, :, None] * streams, axis=1, keepdims=False))

    def hc_post(self, streams, output, post, comb):
        mixed = np.zeros_like(streams)
        for source in (range(self.streams - 1, -1, -1) if base.REVERSED else range(self.streams)):
            mixed += comb[:, source, :, None] * streams[:, source, None, :]
        return base.bf16(post[:, :, None] * output[:, None, :] + mixed)

    def engram_ids(self, segments):
        ids = [[] for _ in self.engram_layers]
        for segment in segments:
            mapped = self.token_map[np.asarray(segment, dtype=np.int64)]
            count = mapped.shape[0]
            index = np.arange(count)
            tokens = np.empty((count, self.orders + 1), dtype=np.int64)
            blocked = np.zeros(count, dtype=bool)
            for shift in range(self.orders + 1):
                source = index - shift
                value = np.where(source >= 0, mapped[np.maximum(source, 0)], -1)
                blocked |= (source < 0) | (value == -1)
                tokens[:, shift] = np.where(blocked, self.pad, value)
            for which in range(len(self.engram_layers)):
                products = tokens * self.multipliers[which][None, :]
                rolling = products[:, 0].copy()
                columns = np.empty((count, self.orders * self.engram_heads), dtype=np.int64)
                for order in range(1, self.orders + 1):
                    rolling ^= products[:, order]
                    span = slice((order - 1) * self.engram_heads, order * self.engram_heads)
                    columns[:, span] = rolling[:, None] % self.primes[which][None, span] + self.offsets[which][None, span]
                ids[which].append(columns)
        merged = [np.concatenate(parts) for parts in ids]
        for which, part in enumerate(merged):
            if part.min() < 0 or part.max() >= self.engram_rows[which]:
                raise ValueError(f"an engram hash of layer {self.engram_layers[which]} left its {self.engram_rows[which]}-row table")
        return merged

    def engram(self, layer, streams, ids):
        e = f"layers.{layer}.engram."
        rows, n, width = streams.shape
        table_rows = self.engram_rows[self.engram_layers.index(layer)]
        payload = self.checkpoint.raw(e + "embed.weight", FP8, (table_rows, self.engram_dim))
        exponents = self.checkpoint.raw(e + "embed.scale", SCALE, (table_rows, self.engram_dim // BLOCK))
        unique, inverse = np.unique(ids.reshape(-1), return_inverse=True)
        chunks = np.array_split(unique, min(ENGRAM_THREADS, unique.shape[0]))
        with ThreadPoolExecutor(ENGRAM_THREADS) as pool:
            parts = list(pool.map(lambda chunk: (np.asarray(payload[chunk]), np.asarray(exponents[chunk])), chunks))
        codes, scales = np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])
        gathered = (E4M3[codes].reshape(-1, self.engram_dim // BLOCK, BLOCK) * UE8M0[scales][:, :, None]).reshape(-1, self.engram_dim)
        embedded = qdq8(base.bf16(gathered)[inverse.reshape(-1)].reshape(rows, self.engram_embed))
        kv = self.project(embedded, self.fp8(e + "wkv", (n + 1) * width, self.engram_embed))
        keys, value = kv[:, :n * width].reshape(rows, n, width), kv[:, n * width:]
        weight = self.widened(e + "q_weight", n, width) * self.widened(e + "k_weight", n, width)
        dot = base.reduce_sum(streams * weight[None] * keys, keepdims=False)
        spread = rsqrt(base.reduce_sum(streams * streams, keepdims=False) / np.float32(width) + self.epsilon) * \
            rsqrt(base.reduce_sum(keys * keys, keepdims=False) / np.float32(width) + self.epsilon)
        scaled = dot * spread * (ONE / np.sqrt(np.float32(width)))
        gate = base.sigmoid(np.copysign(np.sqrt(np.maximum(np.abs(scaled), GATE_FLOOR)), scaled))
        return base.bf16(streams + gate[:, :, None] * value[:, None, :])

    def compress(self, layer, normed, bounds):
        a = f"layers.{layer}.attn.compressor."
        ratio, dim, rope = self.ratios[layer], self.head_dim, self.rope_dim
        values = base.matmul(normed, self.widened(a + "wkv.weight", dim, self.hidden).T)
        scores = base.matmul(normed, self.widened(a + "wgate.weight", dim, self.hidden).T) if ratio == 2 else None
        weight = self.widened(a + "norm.weight", dim)
        keys = []
        for first, last in zip(bounds[:-1], bounds[1:]):
            if ratio == 2:
                ends = np.arange(first + 1, last, 2)
                left, right = scores[ends - 1], scores[ends]
                top = np.maximum(left, right)
                early, late = np.exp(left - top), np.exp(right - top)
                pooled = base.bf16((values[ends - 1] * early + values[ends] * late) / (early + late))
                starts = ends - 1 - first
            else:
                pooled = base.bf16(values[first:last])
                starts = np.arange(last - first)
            latent = self.norm(pooled, weight)
            latent[:, dim - rope:] = self.rope(latent[:, dim - rope:], starts, self.yarn_rope)
            keys.append(qdq4(latent))
        return keys

    def attend(self, query, key, extra, ratio, sinks, scale):
        count = query.shape[0]
        index = np.arange(count)
        allowed = (index[None, :] <= index[:, None]) & (index[None, :] > index[:, None] - self.window)
        keys = key
        if extra is not None:
            allowed = np.concatenate([allowed, np.arange(extra.shape[0])[None, :] < ((index + 1) // ratio)[:, None]], axis=1)
            keys = np.concatenate([key, extra], axis=0)
        scores = base.matmul(query.transpose(1, 0, 2), keys.T) * scale
        scores = np.where(allowed[None], scores, np.float32(-np.inf))
        top = np.maximum(scores.max(axis=-1, keepdims=True), sinks[:, None, None])
        weights = np.exp(scores - top)
        total = base.reduce_sum(weights) + np.exp(sinks[:, None, None] - top)
        return (base.matmul(weights, keys) / total).transpose(1, 0, 2)

    def attention(self, layer, normed, positions, bounds, compressed):
        a = f"layers.{layer}.attn."
        rows, heads, dim, rope = normed.shape[0], self.heads, self.head_dim, self.rope_dim
        ratio = self.ratios[layer]
        table = self.yarn_rope if ratio else self.plain_rope
        quantized = qdq8(normed)
        latent = self.project(quantized, self.fp8(a + "wq_a", self.query_lora, self.hidden))
        query_input = qdq8(self.norm(latent, self.widened(a + "q_norm.weight", self.query_lora)))
        query = self.project(query_input, self.fp8(a + "wq_b", heads * dim, self.query_lora)).reshape(rows, heads, dim)
        query[..., dim - rope:] = self.rope(query[..., dim - rope:], positions, table)
        key = self.norm(self.project(quantized, self.fp8(a + "wkv", dim, self.hidden)), self.widened(a + "kv_norm.weight", dim))
        key[:, dim - rope:] = self.rope(key[:, dim - rope:], positions, table)
        key = qdq8(key)
        if layer in self.sources:
            compressed[layer] = self.compress(layer, normed, bounds)
        extra = compressed[self.owners[layer]] if ratio else None
        sinks = self.f32(a + "attn_sink", heads)
        scale = ONE / np.sqrt(np.float32(dim))
        output = np.empty((rows, heads, dim), dtype=np.float32)
        for segment, (first, last) in enumerate(zip(bounds[:-1], bounds[1:])):
            output[first:last] = self.attend(query[first:last], key[first:last], None if extra is None else extra[segment], ratio, sinks, scale)
        output = base.bf16(output)
        output[..., dim - rope:] = self.rope(output[..., dim - rope:], positions, table, inverse=True)
        output = qdq8(output.reshape(rows, heads * dim))
        wo_a = self.fp8(a + "wo_a", self.groups * self.output_lora, self.group_input)
        grouped = np.empty((rows, self.groups * self.output_lora), dtype=np.float32)
        for group in range(self.groups):
            outputs = slice(group * self.output_lora, (group + 1) * self.output_lora)
            grouped[:, outputs] = self.project(output[:, group * self.group_input:(group + 1) * self.group_input], wo_a[outputs])
        del wo_a
        return self.project(qdq8(grouped), self.fp8(a + "wo_b", self.hidden, self.groups * self.output_lora))

    def expert_weights(self, prefix):
        return tuple(self.fp4(prefix + name, rows, columns) for name, rows, columns in self.expert_shapes())

    def shared_weights(self, prefix):
        return tuple(self.fp8(prefix + name, rows, columns) for name, rows, columns in self.expert_shapes())

    def glu(self, values, gate_weight, up_weight, down_weight, weight=None):
        gate = np.minimum(self.project(values, gate_weight), self.limit)
        up = np.clip(self.project(values, up_weight), -self.limit, self.limit)
        activated = base.silu(gate) * up
        if weight is not None:
            activated = activated * weight
        return self.project(qdq8(base.bf16(activated)), down_weight)

    def route(self, f, normed):
        logits = base.matmul(normed, self.widened(f + "gate.weight", self.experts, self.hidden).T)
        with np.errstate(over="ignore"):
            softplus = np.where(logits > SOFTPLUS_LINEAR, logits, np.log1p(np.exp(logits)))
        scores = np.sqrt(np.maximum(softplus, np.float32(0.0)))
        selection = scores + self.f32(f + "gate.bias", self.experts)[None, :]
        chosen = np.argsort(-selection, axis=-1, kind="stable")[:, :self.top_k]
        picked = np.take_along_axis(scores, chosen, axis=-1)
        return chosen, (picked / (tree_sum(picked) + ROUTE_EPSILON) * self.route_scale).astype(np.float32)

    def moe(self, layer, normed):
        f = f"layers.{layer}.ffn."
        rows = normed.shape[0]
        quantized = qdq8(normed)
        chosen, weights = self.route(f, normed)
        outputs = np.zeros((rows, self.top_k, self.hidden), dtype=np.float32)
        experts = [int(expert) for expert in np.unique(chosen)]
        with ThreadPoolExecutor(EXPERT_WINDOW) as pool:
            pending = {expert: pool.submit(self.expert_weights, f"{f}experts.{expert}.") for expert in experts[:EXPERT_WINDOW]}
            for index, expert in enumerate(experts):
                if index + EXPERT_WINDOW < len(experts):
                    following = experts[index + EXPERT_WINDOW]
                    pending[following] = pool.submit(self.expert_weights, f"{f}experts.{following}.")
                members, slots = np.nonzero(chosen == expert)
                outputs[members, slots] = self.glu(quantized[members], *pending.pop(expert).result(), weights[members, slots][:, None])
        shared = self.glu(quantized, *self.shared_weights(f + "shared_experts."))
        combined = np.zeros((rows, self.hidden), dtype=np.float32)
        for slot in (range(self.top_k - 1, -1, -1) if base.REVERSED else range(self.top_k)):
            combined += outputs[:, slot]
        return base.bf16(combined + shared), chosen

    def forward(self, segments, dump=None):
        tokens = np.concatenate([np.asarray(s, dtype=np.int64) for s in segments])
        positions = np.concatenate([np.arange(len(s), dtype=np.int64) for s in segments])
        bounds = np.cumsum([0] + [len(s) for s in segments])
        if positions.max() >= self.dense_keys:
            raise ValueError(f"a prompt reaches position {positions.max()}: from position {self.dense_keys} on the indexer drops compressed keys "
                             "and this reference is dense")
        if tokens.min() < 0 or tokens.max() >= self.vocab:
            raise ValueError(f"a token lies outside the {self.vocab}-token vocabulary")
        hidden = widen(self.checkpoint.raw("embed.weight", BF16, (self.vocab, self.hidden))[tokens])
        streams = np.repeat(hidden[:, None, :], self.streams, axis=1)
        if dump is not None:
            dump["embed"] = hidden.copy()
        pending = np.zeros((tokens.shape[0], self.streams), dtype=np.float32)
        pending[:, 0] = ONE
        ids = self.engram_ids(segments)
        compressed, routes = {}, {}
        for layer in range(self.layers):
            began = time.time()
            p = f"layers.{layer}."
            if layer in self.engram_layers:
                streams = self.engram(layer, streams, ids[self.engram_layers.index(layer)])
            attention_pre, post, comb = self.hc_site(streams, p, "attn")
            normed = self.norm(self.collapse(streams, pending), self.widened(p + "attn_norm.weight", self.hidden))
            streams = self.hc_post(streams, self.attention(layer, normed, positions, bounds, compressed), post, comb)
            middle = time.time()
            pending, post, comb = self.hc_site(streams, p, "ffn")
            normed = self.norm(self.collapse(streams, attention_pre), self.widened(p + "ffn_norm.weight", self.hidden))
            moe, routes[layer] = self.moe(layer, normed)
            streams = self.hc_post(streams, moe, post, comb)
            if dump is not None:
                dump[f"layer{layer:02d}"] = streams.copy()
            print(f"layer {layer} ratio {self.ratios[layer]}{' engram' if layer in self.engram_layers else ''} done in {time.time() - began:.1f}s "
                  f"(attention {middle - began:.1f}s, moe {time.time() - middle:.1f}s)", file=sys.stderr, flush=True)
        return self.norm(self.collapse(streams, pending), self.widened("norm.weight", self.hidden)), routes

    def logits(self, normed):
        if self.head is None:
            self.head = widen(self.checkpoint.raw("head.weight", BF16, (self.vocab, self.hidden)))
        return base.matmul(normed, self.head.T)


def gate_prompts(args):
    import gemma4_reference
    gemma4_reference.GATE_PROMPTS = tuple((name, source, instruction, GATE_TOKENS) for name, source, instruction, _ in gemma4_reference.GATE_PROMPTS)
    gemma4_reference.gate_prompts(args, ("prefix", "user"), ("turn_suffix", "generation"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("gate-prompts", "teacher"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--variants", default="a,b")
    parser.add_argument("--template", default=str(ROOT / "model-families/dsv41_flash/chat_template.json"))
    parser.add_argument("--token-map", default=str(ROOT / "qualification/t1_reference/dsv41/engram_token_map.npy"),
                        help="the engram vocabulary -> compressed id map (tools/dsv41_flash_engram_tables.py's)")
    parser.add_argument("--dump", help="teacher: write the embedding and every layer's streams for all rows to PATH.a.npz / PATH.b.npz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "gate-prompts":
        gate_prompts(args)
    elif not args.prompts:
        parser.error("teacher needs --prompts")
    else:
        base.teacher(args, Reference(args.checkpoint, args.token_map))
    return 0


if __name__ == "__main__":
    sys.exit(main())
