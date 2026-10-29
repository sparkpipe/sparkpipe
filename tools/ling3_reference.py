#!/usr/bin/env python3
"""Teacher-forced numpy reference for Ling 3.0 flash and flash-Fin (BailingMoeV3).

  tokenize   render the publisher chat_template.jinja from the checkpoint
             (thinking off, one user turn, generation prompt) and encode it
             with the checkpoint tokenizer: --prompts in.json (list of
             {name, text}) -> --out prompts.json ({"prompts": [{name,
             prompt_token_ids}]})
  score      one forward over prompt + served tokens for every prompt:
             --capture capture.json ({name: {tokens, logprobs}}, optionally
             with the entry's own prompt_token_ids) and --prompts prompts.json -> --out report.json with, per served
             token, the served top logprobs, the reference top-5, the
             reference logprob of the served token and the top-1/top-2
             logit margin; per prompt the top-1 agreement, the near-tie
             disagreements and the served-token logprob deltas
  gate-prompts  the Metal gate prompts (tools/gemma4_reference.py's set) in
             Ling's chat turn (thinking off) -> --out prompts.json
  teacher    FP32 logits of every prompt position but the last for the
             Metal gate: --prompts prompts.json -> --out
             DIR/<name>.ref_<v>.logits for --variants a,b. Both variants
             multiply in FP32 rather than the score mode's f64: a in numpy's
             order; b splits every matmul's inner dimension into 1,024-wide
             chunks summed last chunk first, reverses the other reductions
             (norms, the recurrent dot products, softmax sums) and combines
             the routed experts in descending order, so a against b is the
             noise of FP32 reduction order (the gate's floor)

Semantics follow the publisher modeling_bailing_moe_v3.py and the fla
kernels it calls (fla 0.5.2):
  layers (i + 1) % 6 == 0 are gated MLA, the rest KDA; layers 0-1 dense.
  KDA in its recurrent form token by token: q/k/v projections, depthwise
  causal conv (4 taps) + SiLU, q and k L2-normalized with eps 1e-6 inside
  the sqrt, q scaled by 1/sqrt(128), log decay lower_bound *
  sigmoid(exp(A_log) * (f + dt_bias)), beta = sigmoid(b), S <- exp(g) S,
  v' = beta (v - S^T k), S <- S + k v'^T, o = S^T q, then
  RMSNorm(o) * w * sigmoid(g_proj) in f32 and o_proj.
  MLA: q_proj (no q LoRA), kv_a_proj -> RMSNorm(latent) -> kv_b_proj
  (k_nope | v), interleaved rope on the 64 rope dims with bf16 cos/sin,
  causal softmax(QK^T / sqrt(192)) in f32 with bf16 probabilities, head-wise
  sigmoid gate, dense.
  MoE: f32 router logits, sigmoid scores, expert_bias for selection only,
  top-2 group sums, 4 of 8 groups, top-8, weights renormalized and scaled
  by 2.5; expert outputs combined in f32; shared expert added in bf16.
  SwiGLU limits: the HF modeling applies none; the publisher serving code
  (vLLM / sglang bailing_moe_v3 SwigluStepAndMul) applies min(silu(g), L) *
  clamp(u, -L, L) from expert_swiglu_limit_list (routed) and
  share_expert_swiglu_limit_list (shared), 0 meaning none. --swiglu-limits
  serving (default) applies them, modeling does not.
Rounding follows the bf16 forward: projections, norms, activations and
residual sums round to bf16; recurrent state, gates, softmax and router in
f32. Two deliberate f32 choices: attention scores (the HF eager path rounds
QK^T to bf16, the serving flash kernels do not) and the final logits.

In the score mode every matrix product accumulates in f64 before it is
rounded (t1_reference_common.matmul), so a prompt's numbers do not depend on
the other prompts in the run or on how many served tokens follow.

The checkpoint is memmapped; dense tensors and each routed expert are
converted one at a time. All prompts run together layer by layer so a
layer's experts are read once.
"""

import argparse
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from t1_reference_common import matmul as matmul_f64

FP32 = False
REVERSED = False
CHUNK = 1024


def matmul(left, right):
    if not FP32:
        return matmul_f64(left, right)
    if not REVERSED or left.shape[-1] <= CHUNK:
        return left @ right
    total = None
    for first in reversed(range(0, left.shape[-1], CHUNK)):
        part = left[..., first:first + CHUNK] @ right[first:first + CHUNK]
        total = part if total is None else total + part
    return total


def reduce_sum(values, axis=-1, keepdims=True):
    return np.flip(values, axis=axis).sum(axis=axis, keepdims=keepdims) if REVERSED else values.sum(axis=axis, keepdims=keepdims)


def dot_last(state, vector):
    if REVERSED:
        return np.einsum("hkv,hk->hv", state[:, ::-1], vector[:, ::-1])
    return np.einsum("hkv,hk->hv", state, vector)



def bf16(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def silu(values):
    return values / (np.float32(1.0) + np.exp(-values))


def sigmoid(values):
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-values))


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.weight_map = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

    def locate(self, name, shape, dtype):
        shard = self.weight_map.get(name)
        if shard is None:
            raise ValueError(f"checkpoint has no tensor {name}")
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        base, header = self.headers[shard]
        meta = header[name]
        if meta["dtype"] != dtype or tuple(meta["shape"]) != tuple(shape):
            raise ValueError(f"{name} is {meta['dtype']} {meta['shape']}, the reference expects {dtype} {list(shape)}")
        begin, end = meta["data_offsets"]
        return self.directory / shard, base + begin, end - begin

    def raw(self, name, shape, dtype):
        path, offset, length = self.locate(name, shape, dtype)
        kind = np.uint16 if dtype == "BF16" else np.float32
        return np.memmap(path, dtype=kind, mode="r", offset=offset, shape=(length // np.dtype(kind).itemsize,)).reshape(shape)

    def read_bf16(self, name, *shape):
        path, offset, length = self.locate(name, shape, "BF16")
        data = bytearray(length)
        view = memoryview(data)
        with open(path, "rb", buffering=0) as handle:
            done = 0
            while done < length:
                count = os.preadv(handle.fileno(), [view[done:]], offset + done)
                if count <= 0:
                    raise ValueError(f"short read of {name}")
                done += count
        return (np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)

    def bf16(self, name, *shape):
        return (np.asarray(self.raw(name, shape, "BF16")).astype(np.uint32) << 16).view(np.float32)

    def f32(self, name, *shape):
        return np.array(self.raw(name, shape, "F32"), dtype=np.float32)


class Reference:
    embed_name = "model.word_embeddings.weight"
    expert_bias_name = "gate.expert_bias"
    attention_kind = "mla"

    def __init__(self, directory, limits):
        self.checkpoint = Checkpoint(directory)
        config = json.loads((Path(directory) / "config.json").read_text())
        self.config = config
        expect = {"q_lora_rank": None, "no_kda_lora": True, "rope_interleave": True, "rope_scaling": None,
                  "gated_attention_proj_granularity_type": "head_wise", "num_shared_experts": 1, "score_function": "sigmoid",
                  "norm_topk_prob": True, "kda_safe_gate": True, "use_qkv_bias": False, "use_bias": False, "hidden_act": "silu",
                  "moe_router_enable_expert_bias": True}
        for key, value in expect.items():
            if config.get(key) != value:
                raise ValueError(f"config {key} is {config.get(key)!r}, the reference implements {value!r}")
        self.hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"]
        self.epsilon = np.float32(config["rms_norm_eps"])
        self.nope = config["qk_nope_head_dim"]
        self.rope_dim = config["qk_rope_head_dim"]
        self.value_dim = config["v_head_dim"]
        self.latent = config["kv_lora_rank"]
        self.qk_dim = self.nope + self.rope_dim
        self.conv = config["short_conv_kernel_size"]
        self.lower_bound = np.float32(config["kda_lower_bound"])
        self.experts = config["num_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.groups = config["n_group"]
        self.topk_groups = config["topk_group"]
        self.route_scale = np.float32(config["routed_scaling_factor"])
        self.moe_inter = config["moe_intermediate_size"]
        self.shared_inter = config["moe_shared_expert_intermediate_size"] * config["num_shared_experts"]
        self.layers = config["num_hidden_layers"]
        self.period = config["layer_group_size"]
        self.vocab = config["vocab_size"]
        layers = self.layers
        if limits == "serving":
            self.routed_limits = [float(v) for v in config["expert_swiglu_limit_list"]]
            self.shared_limits = [float(v) for v in config["share_expert_swiglu_limit_list"]]
        else:
            self.routed_limits = [0.0] * layers
            self.shared_limits = [0.0] * layers
        if len(self.routed_limits) != layers or len(self.shared_limits) != layers:
            raise ValueError("swiglu limit lists do not cover every layer")
        self.inv_freq = (1.0 / config["rope_theta"] ** (np.arange(0, self.rope_dim, 2, dtype=np.int64).astype(np.float32) / self.rope_dim)).astype(np.float32)

    def is_attention(self, layer):
        return (layer + 1) % self.period == 0 or layer >= self.layers // self.period * self.period

    def attention(self, layer, normed, positions):
        return self.mla(layer, normed, positions)

    def norm(self, values, name, dim):
        weight = self.checkpoint.bf16(name, dim)
        normed = bf16(values * np.power(reduce_sum(values * values) / np.float32(dim) + self.epsilon, np.float32(-0.5)))
        return bf16(normed * weight)

    def linear(self, values, name, out_dim, in_dim):
        return bf16(matmul(values, self.checkpoint.bf16(name, out_dim, in_dim).T))

    def output(self, values, name, out_dim, in_dim):
        return self.linear(values, name, out_dim, in_dim)

    def conv_silu(self, values, weight):
        padded = np.concatenate([np.zeros((self.conv - 1, values.shape[1]), dtype=np.float32), values], axis=0)
        total = np.zeros_like(values)
        for tap in range(self.conv):
            total += padded[tap:tap + values.shape[0]] * weight[:, tap][None, :]
        return bf16(silu(total))

    def kda(self, layer, normed):
        a = f"model.layers.{layer}.attention."
        rows, width, heads, dim = normed.shape[0], self.heads * self.head_dim, self.heads, self.head_dim
        streams = []
        for key in ("q", "k", "v"):
            projected = self.linear(normed, a + f"{key}_proj.weight", width, self.hidden)
            streams.append(self.conv_silu(projected, self.checkpoint.bf16(a + f"{key}_conv1d.weight", width, 1, self.conv)[:, 0, :]))
        q, k, v = (s.reshape(rows, heads, dim) for s in streams)
        f = self.linear(normed, a + "f_proj.weight", width, self.hidden).reshape(rows, heads, dim)
        a_log = self.checkpoint.f32(a + "A_log", heads)
        dt_bias = self.checkpoint.f32(a + "dt_bias", width).reshape(heads, dim)
        log_decay = self.lower_bound * sigmoid(np.exp(a_log)[None, :, None] * (f + dt_bias[None]))
        beta = sigmoid(self.linear(normed, a + "b_proj.weight", heads, self.hidden))
        q = q / np.sqrt(reduce_sum(q * q) + np.float32(1e-6)) * np.float32(dim ** -0.5)
        k = k / np.sqrt(reduce_sum(k * k) + np.float32(1e-6))
        state = np.zeros((heads, dim, dim), dtype=np.float32)
        output = np.empty((rows, heads, dim), dtype=np.float32)
        decay = np.exp(log_decay)
        for t in range(rows):
            state *= decay[t][:, :, None]
            delta = (v[t] - dot_last(state, k[t])) * beta[t][:, None]
            state += k[t][:, :, None] * delta[:, None, :]
            output[t] = dot_last(state, q[t])
        output = bf16(output)
        gate = self.linear(normed, a + "g_proj.weight", width, self.hidden).reshape(rows, heads, dim)
        weight = self.checkpoint.bf16(a + "o_norm.weight", dim)
        mixed = output * np.power(reduce_sum(output * output) / np.float32(dim) + self.epsilon, np.float32(-0.5)) * weight * sigmoid(gate)
        return self.output(bf16(mixed).reshape(rows, width), a + "o_proj.weight", self.hidden, width)

    def rope(self, values, positions):
        angles = positions[:, None].astype(np.float32) * self.inv_freq[None, :]
        cos = bf16(np.cos(np.concatenate([angles, angles], axis=-1)))[:, None, :]
        sin = bf16(np.sin(np.concatenate([angles, angles], axis=-1)))[:, None, :]
        shape = values.shape
        values = values.reshape(shape[0], shape[1], shape[2] // 2, 2).transpose(0, 1, 3, 2).reshape(shape)
        half = shape[-1] // 2
        rotated = np.concatenate([-values[..., half:], values[..., :half]], axis=-1)
        return bf16(bf16(values * cos) + bf16(rotated * sin))

    def mla(self, layer, normed, positions):
        a = f"model.layers.{layer}.attention."
        rows, heads = normed.shape[0], self.heads
        q = self.linear(normed, a + "q_proj.weight", heads * self.qk_dim, self.hidden).reshape(rows, heads, self.qk_dim)
        compressed = self.linear(normed, a + "kv_a_proj_with_mqa.weight", self.latent + self.rope_dim, self.hidden)
        latent = self.norm(compressed[:, :self.latent], a + "kv_a_layernorm.weight", self.latent)
        kv = self.linear(latent, a + "kv_b_proj.weight", heads * (self.nope + self.value_dim), self.latent).reshape(rows, heads, self.nope + self.value_dim)
        query = np.concatenate([q[..., :self.nope], self.rope(q[..., self.nope:], positions)], axis=-1)
        key_rope = self.rope(compressed[:, None, self.latent:], positions)
        key = np.concatenate([kv[..., :self.nope], np.broadcast_to(key_rope, (rows, heads, self.rope_dim))], axis=-1)
        value = kv[..., self.nope:]
        causal = positions[None, :] <= positions[:, None]
        scale = np.float32(self.qk_dim ** -0.5)
        output = np.empty((rows, heads, self.value_dim), dtype=np.float32)
        for head in range(heads):
            scores = matmul(query[:, head, :], key[:, head, :].T) * scale
            scores = np.where(causal, scores, -np.inf)
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights = bf16(weights / reduce_sum(weights))
            output[:, head, :] = bf16(matmul(weights, value[:, head, :]))
        gate = bf16(sigmoid(self.linear(normed, a + "g_proj.weight", heads, self.hidden)))
        output = bf16(output * gate[:, :, None])
        return self.output(output.reshape(rows, heads * self.value_dim), a + "dense.weight", self.hidden, heads * self.value_dim)

    def swiglu_weights(self, values, weights, limit):
        gate_weight, up_weight, down_weight = weights
        gate = bf16(matmul(values, gate_weight.T))
        up = bf16(matmul(values, up_weight.T))
        activated = bf16(silu(gate))
        if limit > 0.0:
            activated = np.minimum(activated, np.float32(limit))
            up = np.clip(up, np.float32(-limit), np.float32(limit))
        return bf16(matmul(bf16(activated * up), down_weight.T))

    def mlp_weights(self, prefix, inter):
        return (self.checkpoint.read_bf16(prefix + "gate_proj.weight", inter, self.hidden),
                self.checkpoint.read_bf16(prefix + "up_proj.weight", inter, self.hidden),
                self.checkpoint.read_bf16(prefix + "down_proj.weight", self.hidden, inter))

    def swiglu(self, values, prefix, inter, limit):
        return self.swiglu_weights(values, self.mlp_weights(prefix, inter), limit)

    def route(self, layer, normed):
        m = f"model.layers.{layer}.mlp."
        logits = matmul(normed, self.checkpoint.bf16(m + "gate.weight", self.experts, self.hidden).T)
        scores = sigmoid(logits)
        selection = scores + self.checkpoint.f32(m + self.expert_bias_name, self.experts)[None, :]
        rows = normed.shape[0]
        grouped = selection.reshape(rows, self.groups, -1)
        group_score = np.sort(grouped, axis=-1)[..., -2:].sum(axis=-1)
        group_order = np.argsort(-group_score, axis=-1, kind="stable")[:, :self.topk_groups]
        mask = np.zeros((rows, self.groups), dtype=bool)
        np.put_along_axis(mask, group_order, True, axis=-1)
        masked = np.where(np.repeat(mask, self.experts // self.groups, axis=-1), selection, -np.inf)
        chosen = np.argsort(-masked, axis=-1, kind="stable")[:, :self.top_k]
        picked = np.take_along_axis(scores, chosen, axis=-1)
        weights = picked / (picked.sum(axis=-1, keepdims=True) + np.float32(1e-20)) * self.route_scale
        return chosen, weights.astype(np.float32)

    def moe(self, layer, normed):
        m = f"model.layers.{layer}.mlp."
        chosen, weights = self.route(layer, normed)
        combined = np.zeros((normed.shape[0], self.hidden), dtype=np.float32)
        limit = self.routed_limits[layer]
        experts = [int(expert) for expert in (np.unique(chosen)[::-1] if REVERSED else np.unique(chosen))]
        window = 16
        with ThreadPoolExecutor(8) as pool:
            pending = {expert: pool.submit(self.mlp_weights, m + f"experts.{expert}.", self.moe_inter) for expert in experts[:window]}
            for index, expert in enumerate(experts):
                if index + window < len(experts):
                    following = experts[index + window]
                    pending[following] = pool.submit(self.mlp_weights, m + f"experts.{following}.", self.moe_inter)
                rows, slots = np.nonzero(chosen == expert)
                out = self.swiglu_weights(normed[rows], pending.pop(expert).result(), limit)
                combined[rows] += out * weights[rows, slots][:, None]
        shared = self.swiglu(normed, m + "shared_experts.", self.shared_inter, self.shared_limits[layer])
        return bf16(bf16(combined) + shared), chosen

    def forward(self, segments, dump=None):
        tokens = np.concatenate([np.asarray(s, dtype=np.int64) for s in segments])
        positions = np.concatenate([np.arange(len(s), dtype=np.int64) for s in segments])
        bounds = np.cumsum([0] + [len(s) for s in segments])
        embed = self.checkpoint.raw(self.embed_name, (self.vocab, self.hidden), "BF16")
        residual = (np.asarray(embed[tokens]).astype(np.uint32) << 16).view(np.float32)
        del embed
        if dump is not None:
            dump["embed"] = residual.copy()
        routes = {}
        for layer in range(self.layers):
            began = time.time()
            p = f"model.layers.{layer}."
            normed = self.norm(residual, p + "input_layernorm.weight", self.hidden)
            attention = np.empty_like(residual)
            for first, last in zip(bounds[:-1], bounds[1:]):
                part = normed[first:last]
                attention[first:last] = self.attention(layer, part, positions[first:last]) if self.is_attention(layer) else self.kda(layer, part)
            residual = bf16(residual + attention)
            if dump is not None:
                dump[f"attn{layer:02d}"] = residual.copy()
            normed = self.norm(residual, p + "post_attention_layernorm.weight", self.hidden)
            if layer < self.config["first_k_dense_replace"]:
                mlp = self.swiglu(normed, p + "mlp.", self.config["intermediate_size"], 0.0)
            else:
                mlp, routes[layer] = self.moe(layer, normed)
            residual = bf16(residual + mlp)
            if dump is not None:
                dump[f"layer{layer:02d}"] = residual.copy()
            print(f"layer {layer} {self.attention_kind if self.is_attention(layer) else 'kda'} done in {time.time() - began:.1f}s", file=sys.stderr, flush=True)
        return self.norm(residual, "model.norm.weight", self.hidden), routes

    def logits(self, normed, chunk=16384):
        head = self.checkpoint.raw("lm_head.weight", (self.vocab, self.hidden), "BF16")
        parts = []
        for first in range(0, self.vocab, chunk):
            rows = (np.asarray(head[first:first + chunk]).astype(np.uint32) << 16).view(np.float32)
            parts.append(matmul(normed, rows.T))
        return np.concatenate(parts, axis=-1)


def tokenize(args):
    import jinja2
    from tokenizers import Tokenizer
    checkpoint = Path(args.checkpoint)
    template = jinja2.Environment(trim_blocks=False, lstrip_blocks=False).from_string((checkpoint / "chat_template.jinja").read_text())
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    rendered = []
    for prompt in json.loads(Path(args.prompts).read_text()):
        text = template.render(messages=[{"role": "user", "content": prompt["text"]}], add_generation_prompt=True, enable_thinking=False, tools=None)
        rendered.append({"name": prompt["name"], "text": text, "prompt_token_ids": tokenizer.encode(text, add_special_tokens=False).ids})
    Path(args.out).write_text(json.dumps({"prompts": rendered}, indent=1))


def score(args, reference):
    prompts = {p["name"]: p["prompt_token_ids"] for p in json.loads(Path(args.prompts).read_text())["prompts"]}
    capture = json.loads(Path(args.capture).read_text())
    names = [name for name in capture if capture[name]["tokens"]]
    for name in names:
        if "prompt_token_ids" in capture[name]:
            if name in prompts and prompts[name] != capture[name]["prompt_token_ids"]:
                raise ValueError(f"{name}: the capture's prompt differs from --prompts")
            prompts[name] = capture[name]["prompt_token_ids"]
    segments = [prompts[name] + capture[name]["tokens"] for name in names]
    dump = {} if args.dump else None
    normed, routes = reference.forward(segments, dump)
    if dump is not None:
        np.savez(args.dump, names=np.array(names), bounds=np.cumsum([0] + [len(s) for s in segments]), **dump)
    bounds = np.cumsum([0] + [len(s) for s in segments])
    report = {"checkpoint": str(args.checkpoint), "swiglu_limits": getattr(args, "swiglu_limits", None), "prompts": {}}
    for index, name in enumerate(names):
        prompt, served = prompts[name], capture[name]
        first = bounds[index] + len(prompt) - 1
        logits = reference.logits(normed[first:first + len(served["tokens"])])
        logprobs = logits - logits.max(axis=-1, keepdims=True)
        logprobs -= np.log(np.exp(logprobs).sum(axis=-1, keepdims=True))
        rows = []
        for step, token in enumerate(served["tokens"]):
            order = np.argsort(-logits[step], kind="stable")[:5]
            served_top = served["logprobs"][step] if served.get("logprobs") else None
            served_logprob = served_top[0][1] if served_top and served_top[0][0] == token else None
            rows.append({"served": token, "served_top": served_top, "served_logprob": served_logprob,
                         "reference_logprob": float(logprobs[step, token]),
                         "top": [[int(t), float(logprobs[step, t])] for t in order],
                         "margin": float(logits[step, order[0]] - logits[step, order[1]])})
        disagree = [i for i, row in enumerate(rows) if row["top"][0][0] != row["served"]]
        ties = [i for i in disagree if rows[i]["margin"] < args.tie_margin]
        deltas = [abs(row["served_logprob"] - row["reference_logprob"]) for row in rows if row["served_logprob"] is not None]
        summary = {"prompt_tokens": len(prompt), "cached_tokens": served.get("cached_tokens"), "tokens": len(rows), "top1_agree": len(rows) - len(disagree), "disagree": disagree, "near_tie_disagree": ties,
                   "non_tie_disagree": [i for i in disagree if i not in ties],
                   "first_divergence": disagree[0] if disagree else None,
                   "served_logprob_delta_median": float(np.median(deltas)) if deltas else None,
                   "served_logprob_delta_max": float(np.max(deltas)) if deltas else None, "rows": rows}
        report["prompts"][name] = summary
        print(f"{name}: {summary['top1_agree']}/{len(rows)} top-1 agree, disagreements {disagree} (near ties {ties}), "
              f"served logprob delta median {summary['served_logprob_delta_median']} max {summary['served_logprob_delta_max']}", flush=True)
    Path(args.out).write_text(json.dumps(report))


def teacher(args):
    global FP32, REVERSED
    variants = args.variants.split(",")
    if not variants or any(label not in ("a", "b") for label in variants) or len(set(variants)) != len(variants):
        raise SystemExit("--variants takes distinct a and b")
    reference = Reference(args.checkpoint, args.swiglu_limits)
    prompts = json.loads(Path(args.prompts).read_text())["prompts"]
    bounds = np.cumsum([0] + [len(prompt["prompt_token_ids"]) for prompt in prompts])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    FP32 = True
    for label in variants:
        REVERSED = label == "b"
        normed, _ = reference.forward([prompt["prompt_token_ids"] for prompt in prompts])
        for index, prompt in enumerate(prompts):
            first, count = bounds[index], len(prompt["prompt_token_ids"])
            with open(out / f"{prompt['name']}.ref_{label}.logits", "wb") as handle:
                for row in range(0, count - 1, 128):
                    handle.write(np.ascontiguousarray(reference.logits(normed[first + row:first + min(row + 128, count - 1)]), dtype=np.float32).tobytes())
            print(f"{prompt['name']}: {count - 1} positions, variant {label}", flush=True)


def gate_prompts(args):
    import gemma4_reference
    gemma4_reference.gate_prompts(args, ("prefix", "plain_prefix", "user"), ("turn_suffix", "generation"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("tokenize", "score", "gate-prompts", "teacher"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--variants", default="a,b")
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/ling/chat_template.json"))
    parser.add_argument("--capture")
    parser.add_argument("--swiglu-limits", choices=("serving", "modeling"), default="serving")
    parser.add_argument("--tie-margin", type=float, default=0.1)
    parser.add_argument("--dump", help="write the embedding and every layer's post-attention and post-MLP residual for all rows to this .npz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "gate-prompts":
        gate_prompts(args)
    elif not args.prompts:
        parser.error(f"{args.mode} needs --prompts")
    elif args.mode == "tokenize":
        tokenize(args)
    elif args.mode == "teacher":
        teacher(args)
    else:
        if not args.capture:
            parser.error("score needs --capture")
        score(args, Reference(args.checkpoint, args.swiglu_limits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
