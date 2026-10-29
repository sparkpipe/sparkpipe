#!/usr/bin/env python3
"""Teacher-forced numpy reference for Qwen 3.8 Flash (qwen4_exp, NVIDIA NVFP4 checkpoint).

  teacher   FP32 logits for every position but the last of each prompt
            (--prompts, e.g. qualification/metal/qwen38_27b/gate_prompts.json:
            Flash shares the 27B tokenizer and chat template) to
            <out>/<name>.ref_<variant>.logits for each of --variants: a sums in
            numpy's order, b splits every matmul's inner dimension into
            1,024-wide chunks summed last chunk first and reverses every other
            reduction, the hyper-connection stream sum and the expert
            combination, so a against b is the noise of FP32 reduction order
            (the Metal gate's floor)

No publisher modeling code exists for qwen4_exp, so the semantics are those of
SparkPipe's CUDA stage (model-families/qwen38/source/qwen38_hybrid_stage.cuh),
read from the checkpoint directly. R() is a round to bf16.
- projections: R(x @ W); NVFP4 experts dequantise exactly to e2m1 x ue4m3 x
  weight_scale_2 (the decode arithmetic; CUDA's prefill GEMM rounds that
  operand to bf16)
- hyper-connections, 4 streams: group norm R(x * rsqrt(mean(x^2) + eps) *
  (1 + w)) per stream; low = R(R(low / 4) * sigmoid(R(low / 4))) of R(n @
  down); mixed = R(sum over streams of R(R(sigmoid(R(n @ up))) * n) / 4);
  inject = R(2 R(sigmoid(R(R(n . w_s) / 4)))); residual = R(residual +
  R(out * inject))
- n-gram embedding at layer 1: CUDA's hash (int64 wrap, C remainder), rows
  R(e4m3 x scale); key R(e @ key) group-normed, query the group-normed
  residual, score R(R(sum R(k q)) / sqrt(2560)), gate R(sigmoid(sign x
  R(sqrt(max(|score|, 1e-6))))), gated R(gate x R(e @ value)); a 4-tap conv
  with dilation 3 (oldest tap first) of the group-normed gated rows, R'd and
  swished; residual = R(residual + R(gated + conv))
- GDN as tools/qwen38_27b_reference.c, with the sigmoid output gate
- attention as tools/qwen38_27b_reference.c (2 KV heads); the indexer keeps
  every block below 2,052 positions, so it is not modelled and longer prompts
  are refused
- MoE: router logits R(x @ router), FP32 softmax, top-10 (ties to the lower
  expert), weights p / (sum + 1e-20); each route R(R(silu(gate) x up x
  weight) @ down); the routes in pick order plus the shared expert R(R(R(silu
  (gate) x up) @ down) x sigmoid(R(x . shared_gate))), summed in FP32, R'd
- the head reads the final hyper-connection mix; there is no final norm
"""

import argparse
import json
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PREFIX = "model.language_model."
EPSILON = np.float32(1e-6)
NGRAM_SHARD_ROWS = 2500012
DENSE_INDEX_POSITIONS = 2051
DTYPES = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8, "I64": np.int64}

REVERSED = False
CHUNK = 1024


def bf16(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def sigmoid(values):
    return (1.0 / (1.0 + np.exp(-values.astype(np.float64)))).astype(np.float32)


def silu(values):
    wide = values.astype(np.float64)
    return (wide / (1.0 + np.exp(-wide))).astype(np.float32)


def reduce_sum(values, axis=-1, keepdims=False):
    return np.flip(values, axis=axis).sum(axis=axis, keepdims=keepdims) if REVERSED else values.sum(axis=axis, keepdims=keepdims)


def ordered(count):
    return range(count - 1, -1, -1) if REVERSED else range(count)


def matmul(left, right, chunk=CHUNK):
    if not REVERSED or left.shape[-1] <= chunk:
        return left @ right
    total = None
    for first in reversed(range(0, left.shape[-1], chunk)):
        part = left[..., first:first + chunk] @ right[first:first + chunk]
        total = part if total is None else total + part
    return total


def e4m3_table():
    table = np.empty(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exponent, mantissa = (code >> 3) & 15, code & 7
        if exponent == 15 and mantissa == 7:
            table[code] = np.nan
        elif exponent == 0:
            table[code] = sign * (mantissa / 8.0) * 2.0 ** -6
        else:
            table[code] = sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return table


E4M3 = e4m3_table()
UE4M3 = np.where(np.arange(256) & 0x7F == 0x7F, np.float32(448.0), E4M3[np.arange(256) & 0x7F]).astype(np.float32)
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        index = json.loads((self.directory / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.headers = {}

    def meta(self, name):
        shard = self.weight_map[name]
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        base, header = self.headers[shard]
        return shard, base, header[name]

    def raw(self, name, dtype):
        shard, base, meta = self.meta(name)
        if meta["dtype"] != dtype:
            raise ValueError(f"{name} is {meta['dtype']}, expected {dtype}")
        begin, end = meta["data_offsets"]
        kind = DTYPES[dtype]
        return np.memmap(self.directory / shard, dtype=kind, mode="r", offset=base + begin,
                         shape=((end - begin) // np.dtype(kind).itemsize,)).reshape(meta["shape"])

    def bf16(self, name):
        return (np.asarray(self.raw(name, "BF16")).astype(np.uint32) << 16).view(np.float32)

    def gain(self, name):
        return self.bf16(name) + np.float32(1.0)

    def scalar(self, name):
        value = self.bf16(name) if self.meta(name)[2]["dtype"] == "BF16" else np.asarray(self.raw(name, "F32"), dtype=np.float32)
        return np.float32(value.reshape(-1)[0])

    def nvfp4(self, name):
        packed = np.asarray(self.raw(name + ".weight", "U8"))
        plane = np.asarray(self.raw(name + ".weight_scale", "F8_E4M3"))
        weight = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
        weight[:, 0::2] = E2M1[packed & 15]
        weight[:, 1::2] = E2M1[packed >> 4]
        return weight * np.repeat(UE4M3[plane] * self.scalar(name + ".weight_scale_2"), 16, axis=1)


class Reference:
    def __init__(self, directory, threads):
        self.checkpoint = Checkpoint(directory)
        text = json.loads((Path(directory) / "config.json").read_text())["text_config"]
        self.threads = threads
        self.hidden = text["hidden_size"]
        self.layers = text["num_hidden_layers"]
        self.streams = text["hc_count"]
        self.key_heads, self.value_heads = text["linear_num_key_heads"], text["linear_num_value_heads"]
        self.key_dim, self.value_dim = text["linear_key_head_dim"], text["linear_value_head_dim"]
        self.heads, self.kv_heads, self.head_dim = text["num_attention_heads"], text["num_key_value_heads"], text["head_dim"]
        rope = text.get("rope_parameters") or {}
        self.theta = rope.get("rope_theta", text.get("rope_theta"))
        self.rope_dim = int(self.head_dim * rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0)))
        self.period = text["full_attention_interval"]
        self.top_k = text["num_experts_per_tok"]
        self.ngram = text["ngram_size"]
        self.ngram_heads = text["heads_per_ngram"]
        self.ple_taps = text["ple_conv_kernel_size"]
        self.eos = text["eos_token_id"]
        self.ple_layer = text["ple_layer_ids"][0] - 1
        if text.get("output_gate_type") != "sigmoid" or not text.get("norm_topk_prob") or len(text["ple_layer_ids"]) != 1 or \
                self.value_heads % self.key_heads != 0 or self.rope_dim % 2 != 0:
            raise ValueError("config is outside what this reference implements")

    def full(self, layer):
        return layer % self.period == self.period - 1

    def linear(self, x, weight):
        return bf16(matmul(x, weight.T))

    def group_norm(self, values, weight):
        rows = values.shape[0]
        grouped = values.reshape(rows, self.streams, self.hidden)
        mean = reduce_sum(grouped * grouped, keepdims=True) / np.float32(self.hidden)
        scale = np.float32(1.0) / np.sqrt(mean + EPSILON)
        return bf16((grouped * scale).reshape(rows, -1) * weight)

    def mixer(self, prefix, residual, inject):
        c = self.checkpoint
        normed = self.group_norm(residual, c.gain(prefix + "hc_norm.weight"))
        low = bf16(self.linear(normed, c.bf16(prefix + "input_mix_weight_down.weight")) / np.float32(self.streams))
        low = bf16(low * sigmoid(low))
        up = bf16(sigmoid(self.linear(low, c.bf16(prefix + "input_mix_weight_up.weight")))).reshape(-1, self.streams, self.hidden)
        grouped = normed.reshape(-1, self.streams, self.hidden)
        total = np.zeros((residual.shape[0], self.hidden), dtype=np.float32)
        for stream in ordered(self.streams):
            total += bf16(up[:, stream] * grouped[:, stream])
        mixed = bf16(total / np.float32(self.streams))
        if not inject:
            return mixed, None
        logit = bf16(matmul(normed, c.bf16(prefix + "block_inject_weight.weight").T))
        return mixed, bf16(np.float32(2.0) * bf16(sigmoid(bf16(logit / np.float32(self.streams)))))

    def inject(self, residual, output, weights):
        added = bf16(output[:, None, :] * weights[:, :, None]).reshape(residual.shape)
        return bf16(residual + added)

    def ngram_ids(self, tokens, prefix):
        c = self.checkpoint
        multipliers = np.asarray(c.raw(prefix + "layer_multipliers", "I64")).astype(np.int64)
        sizes = np.asarray(c.raw(prefix + "ngram_heads_vocab_sizes", "I64")).astype(np.int64)
        offsets = np.asarray(c.raw(prefix + "ngram_heads_offsets", "I64")).astype(np.int64)
        orders = self.ngram - 1
        count = len(tokens)
        tokens = np.asarray(tokens, dtype=np.int64)
        shifted = np.empty((orders + 1, count), dtype=np.int64)
        blocked = np.zeros(count, dtype=bool)
        positions = np.arange(count)
        shifted[0] = tokens
        for shift in range(1, orders + 1):
            source = np.where(positions >= shift, tokens[np.maximum(positions - shift, 0)], self.eos)
            blocked |= (positions < shift) | (source == self.eos)
            shifted[shift] = np.where(blocked, self.eos, source)
        ids = np.empty((count, orders * self.ngram_heads), dtype=np.int64)
        with np.errstate(over="ignore"):
            rolling = shifted[0] * multipliers[0]
            for order in range(1, orders + 1):
                rolling = rolling ^ (shifted[order] * multipliers[order])
                for head in range(self.ngram_heads):
                    column = (order - 1) * self.ngram_heads + head
                    ids[:, column] = np.fmod(rolling, sizes[column]) + offsets[column]
        return ids

    def ple(self, residual, sequences, spans):
        c = self.checkpoint
        p = f"{PREFIX}layers.{self.ple_layer}.ple."
        table = p + "ple_embedding.ngram_embedding."
        scale = c.scalar(table + "weight_scale")
        ids = np.concatenate([self.ngram_ids(tokens, p + "ple_embedding.") for tokens in sequences])
        if ids.min() < 0 or ids.max() >= len([n for n in c.weight_map if n.startswith(table + "shard_")]) * NGRAM_SHARD_ROWS:
            raise ValueError("an n-gram id falls outside the table")
        rows, columns = ids.shape
        parts = []
        for column in range(columns):
            shard, row = np.divmod(ids[:, column], NGRAM_SHARD_ROWS)
            values = np.empty((rows, c.raw(table + "shard_0.weight", "F8_E4M3").shape[1]), dtype=np.float32)
            for index in np.unique(shard):
                chosen = shard == index
                values[chosen] = E4M3[np.asarray(c.raw(f"{table}shard_{index}.weight", "F8_E4M3")[row[chosen]])]
            parts.append(bf16(values * scale))
        embed = np.concatenate(parts, axis=1)
        key = self.group_norm(self.linear(embed, c.bf16(p + "key_proj.weight")), c.gain(p + "norm_key.weight"))
        value = self.linear(embed, c.bf16(p + "value_proj.weight"))
        query = self.group_norm(residual, c.gain(p + "norm_query.weight"))
        products = bf16(key * query).reshape(rows, self.streams, self.hidden)
        score = bf16(bf16(reduce_sum(products)) / np.float32(np.sqrt(np.float32(self.hidden))))
        magnitude = bf16(np.sqrt(np.maximum(np.abs(score), np.float32(1e-6))))
        gate = bf16(sigmoid(np.sign(score) * magnitude))
        gated = bf16(gate[:, :, None] * value[:, None, :]).reshape(rows, -1)
        normed = self.group_norm(gated, c.gain(p + "norm_conv.weight"))
        weight = c.bf16(p + "conv1d.weight").reshape(normed.shape[1], self.ple_taps)
        dilation = self.ngram
        conv = np.empty_like(normed)
        for first, count in spans:
            window = np.concatenate([np.zeros(((self.ple_taps - 1) * dilation, normed.shape[1]), dtype=np.float32), normed[first:first + count]])
            total = np.zeros((count, normed.shape[1]), dtype=np.float32)
            for tap in ordered(self.ple_taps):
                total += window[tap * dilation:tap * dilation + count] * weight[:, tap]
            total = bf16(total)
            conv[first:first + count] = bf16(total * sigmoid(total))
        return bf16(residual + bf16(gated + conv))

    def gdn(self, layer, x, spans):
        c = self.checkpoint
        p = f"{PREFIX}layers.{layer}.linear_attn."
        qk, v_width = self.key_heads * self.key_dim, self.value_heads * self.value_dim
        mix = self.linear(x, c.bf16(p + "in_proj_qkv.weight"))
        z = self.linear(x, c.bf16(p + "in_proj_z.weight"))
        b = self.linear(x, c.bf16(p + "in_proj_b.weight"))
        a = self.linear(x, c.bf16(p + "in_proj_a.weight"))
        weight = c.bf16(p + "conv1d.weight").reshape(mix.shape[1], -1)
        taps = weight.shape[1]
        a_log = c.bf16(p + "A_log").astype(np.float64)
        dt_bias = c.bf16(p + "dt_bias")
        norm = c.bf16(p + "norm.weight")
        group = self.value_heads // self.key_heads
        core = np.empty((x.shape[0], self.value_heads, self.value_dim), dtype=np.float32)
        for first, count in spans:
            window = np.concatenate([np.zeros((taps - 1, mix.shape[1]), dtype=np.float32), mix[first:first + count]])
            total = np.zeros((count, mix.shape[1]), dtype=np.float32)
            for tap in ordered(taps):
                total += window[tap:tap + count] * weight[:, tap]
            mixed = bf16(silu(bf16(total)))
            q = mixed[:, :qk].reshape(count, self.key_heads, self.key_dim)
            k = mixed[:, qk:2 * qk].reshape(count, self.key_heads, self.key_dim)
            value = mixed[:, 2 * qk:].reshape(count, self.value_heads, self.value_dim)
            q_scale = (1.0 / np.sqrt((reduce_sum(q * q, keepdims=True) + EPSILON).astype(np.float64)) / np.sqrt(self.key_dim)).astype(np.float32)
            k_scale = (1.0 / np.sqrt((reduce_sum(k * k, keepdims=True) + EPSILON).astype(np.float64))).astype(np.float32)
            q = np.repeat(q * q_scale, group, axis=1)
            k = np.repeat(k * k_scale, group, axis=1)
            shifted = (a[first:first + count] + dt_bias).astype(np.float64)
            softplus = np.where(shifted > 20.0, shifted, np.log1p(np.exp(np.minimum(shifted, 20.0))))
            decay = np.exp(-np.exp(a_log) * softplus).astype(np.float32)
            beta = sigmoid(b[first:first + count])
            state = np.zeros((self.value_heads, self.key_dim, self.value_dim), dtype=np.float32)
            for t in range(count):
                state *= decay[t][:, None, None]
                memory = reduce_sum(state * k[t][:, :, None], axis=1)
                state += k[t][:, :, None] * (beta[t][:, None] * (value[t] - memory))[:, None, :]
                core[first + t] = bf16(reduce_sum(state * q[t][:, :, None], axis=1))
        mean = reduce_sum(core * core, keepdims=True) / np.float32(self.value_dim)
        scale = (1.0 / np.sqrt((mean + EPSILON).astype(np.float64))).astype(np.float32)
        normed = bf16(bf16(core * scale) * norm)
        gated = bf16(normed * sigmoid(z.reshape(core.shape)))
        return self.linear(gated.reshape(x.shape[0], -1), c.bf16(p + "out_proj.weight"))

    def head_norm_rope(self, values, weight, positions):
        mean = reduce_sum(values * values, keepdims=True) / np.float32(self.head_dim)
        scale = (1.0 / np.sqrt((mean + EPSILON).astype(np.float64))).astype(np.float32)
        normed = bf16(values * scale * weight)
        half = self.rope_dim // 2
        pairs = np.arange(half)
        frequency = (1.0 / np.power(self.theta, 2.0 * pairs / self.rope_dim)).astype(np.float32)
        angle = (positions.astype(np.float32)[:, None] * frequency[None, :]).astype(np.float64)
        cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
        low, high = normed[..., :half].astype(np.float64), normed[..., half:self.rope_dim].astype(np.float64)
        rotated = np.concatenate([low * cos - high * sin, high * cos + low * sin], axis=-1).astype(np.float32)
        return bf16(np.concatenate([rotated, normed[..., self.rope_dim:]], axis=-1))

    def attention(self, layer, x, spans):
        c = self.checkpoint
        p = f"{PREFIX}layers.{layer}.self_attn."
        rows = x.shape[0]
        gq = self.linear(x, c.bf16(p + "q_proj.weight")).reshape(rows, self.heads, 2 * self.head_dim)
        k = self.linear(x, c.bf16(p + "k_proj.weight")).reshape(rows, self.kv_heads, self.head_dim)
        v = self.linear(x, c.bf16(p + "v_proj.weight")).reshape(rows, self.kv_heads, self.head_dim)
        q_norm, k_norm = c.gain(p + "q_norm.weight"), c.gain(p + "k_norm.weight")
        group = self.heads // self.kv_heads
        gated = np.empty((rows, self.heads, self.head_dim), dtype=np.float32)
        for first, count in spans:
            positions = np.arange(count)
            query = self.head_norm_rope(gq[first:first + count, :, :self.head_dim], q_norm, positions)
            key = self.head_norm_rope(k[first:first + count], k_norm, positions)
            visible = positions[None, :] <= positions[:, None]
            for head in range(self.heads):
                scores = (query[:, head] @ key[:, head // group].T) * np.float32(1.0 / np.sqrt(self.head_dim))
                scores = np.where(visible, scores, -np.inf)
                weights = np.exp((scores - scores.max(axis=-1, keepdims=True)).astype(np.float64)).astype(np.float32)
                weights = weights / reduce_sum(weights, keepdims=True)
                out = bf16(matmul(weights, v[first:first + count, head // group]))
                gated[first:first + count, head] = bf16(out * sigmoid(gq[first:first + count, head, self.head_dim:]))
        return self.linear(gated.reshape(rows, -1), c.bf16(p + "o_proj.weight"))

    def moe(self, layer, x):
        c = self.checkpoint
        m = f"{PREFIX}layers.{layer}.mlp."
        logits = self.linear(x, c.bf16(m + "gate.weight")).astype(np.float64)
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True)).astype(np.float32)
        probs = probs / reduce_sum(probs, keepdims=True)
        chosen = np.argsort(-probs, axis=-1, kind="stable")[:, :self.top_k]
        weights = np.take_along_axis(probs, chosen, axis=-1)
        weights = weights / (reduce_sum(weights, keepdims=True) + np.float32(1e-20))
        routed = np.zeros((x.shape[0], self.top_k, self.hidden), dtype=np.float32)

        def run(expert):
            tokens, slots = np.nonzero(chosen == expert)
            e = f"{m}experts.{expert}."
            rows = x[tokens]
            gate = self.linear(rows, c.nvfp4(e + "gate_proj"))
            up = self.linear(rows, c.nvfp4(e + "up_proj"))
            act = bf16(silu(gate) * up * weights[tokens, slots][:, None])
            return tokens, slots, self.linear(act, c.nvfp4(e + "down_proj"))

        with ThreadPoolExecutor(self.threads) as pool:
            for tokens, slots, out in pool.map(run, np.unique(chosen)):
                routed[tokens, slots] = out
        gate = self.linear(x, c.bf16(m + "shared_expert.gate_proj.weight"))
        up = self.linear(x, c.bf16(m + "shared_expert.up_proj.weight"))
        shared = self.linear(bf16(silu(gate) * up), c.bf16(m + "shared_expert.down_proj.weight"))
        shared = bf16(shared * sigmoid(self.linear(x, c.bf16(m + "shared_expert_gate.weight"))))
        total = np.zeros((x.shape[0], self.hidden), dtype=np.float32)
        for slot in ordered(self.top_k):
            total += routed[:, slot]
        return bf16(total + shared)

    def forward(self, sequences):
        spans, first = [], 0
        for tokens in sequences:
            if len(tokens) > DENSE_INDEX_POSITIONS:
                raise ValueError(f"a {len(tokens)}-token prompt needs the indexer, which this reference does not model")
            spans.append((first, len(tokens)))
            first += len(tokens)
        embed = self.checkpoint.raw(PREFIX + "embed_tokens.weight", "BF16")
        flat = np.concatenate([np.asarray(tokens, dtype=np.int64) for tokens in sequences])
        hidden = (np.asarray(embed[flat]).astype(np.uint32) << 16).view(np.float32)
        residual = np.tile(hidden, (1, self.streams))
        for layer in range(self.layers):
            started = time.time()
            p = f"{PREFIX}layers.{layer}."
            if layer == self.ple_layer:
                residual = self.ple(residual, sequences, spans)
            mixed, weights = self.mixer(p + "attn_hyper_connection.", residual, True)
            out = self.attention(layer, mixed, spans) if self.full(layer) else self.gdn(layer, mixed, spans)
            residual = self.inject(residual, out, weights)
            mixed, weights = self.mixer(p + "mlp_hyper_connection.", residual, True)
            residual = self.inject(residual, self.moe(layer, mixed), weights)
            print(f"layer {layer} {'attention' if self.full(layer) else 'gdn'} done in {time.time() - started:.1f}s", file=sys.stderr, flush=True)
        final, _ = self.mixer(PREFIX + "hyper_connection_mixer.", residual, False)
        return final, spans

    def logits(self, hidden, chunk=16384):
        head = self.checkpoint.raw("lm_head.weight", "BF16")
        parts = []
        for first in range(0, head.shape[0], chunk):
            parts.append(matmul(hidden, (np.asarray(head[first:first + chunk]).astype(np.uint32) << 16).view(np.float32).T))
        return np.concatenate(parts, axis=-1)


def teacher(args):
    global REVERSED
    variants = args.variants.split(",")
    if not variants or any(label not in ("a", "b") for label in variants) or len(set(variants)) != len(variants):
        raise SystemExit("--variants takes distinct a and b")
    reference = Reference(args.checkpoint, args.threads)
    prompts = json.loads(Path(args.prompts).read_text())["prompts"]
    if args.only:
        prompts = [prompt for prompt in prompts if prompt["name"] in args.only.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for label in variants:
        REVERSED = label == "b"
        hidden, spans = reference.forward([prompt["prompt_token_ids"] for prompt in prompts])
        for prompt, (first, count) in zip(prompts, spans):
            with open(out / f"{prompt['name']}.ref_{label}.logits", "wb") as handle:
                for row in range(0, count - 1, 128):
                    last = min(row + 128, count - 1)
                    handle.write(np.ascontiguousarray(reference.logits(hidden[first + row:first + last]), dtype=np.float32).tobytes())
            print(f"{prompt['name']}: {count - 1} positions, variant {label}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("teacher",))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--variants", default="a,b")
    parser.add_argument("--only", help="comma-separated prompt names (default every prompt)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", required=True)
    teacher(parser.parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
