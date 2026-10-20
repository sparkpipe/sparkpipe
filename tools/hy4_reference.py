#!/usr/bin/env python3
"""Teacher-forced numpy reference for Hy4 preview (tencent/Hy4-preview-FP8).

  tokenize   render chat prompts with the checkpoint's own chat_template.jinja
             (reasoning_effort no_think, generation prompt) and tokenizer.json:
             --prompts in.json (list of {name, text}) -> --out prompts.json
             ({"prompts": [{name, prompt_token_ids}]})
  score      one layer-major forward over prompt + served tokens of every
             prompt at once: --capture capture.json ({name: {tokens,
             logprobs}}) and --prompts prompts.json -> --out report.json with,
             for every served token, the reference top-5, the reference
             logprob of the served token, the served logprob and the
             reference top-1/top-2 logit margin

The checkpoint ships no modeling code. The semantics follow config.json and
the hy-v4 graph llama.cpp merged at 49c0dc82 (MIT, see NOTICE), which mirrors the vLLM
hunyuan_v4 deployment. MXFP8 weights are E4M3 values times 2^(e8m0 - 127) per
32 input columns. Activations stay bf16 between operations, as in a bf16
deployment; accumulations are f32.

  residual   4 hyper-connection streams of the embedding, bf16
  hc pre     mixes = fn @ rms(flat streams); pre = sigmoid(mix*scale0 + base)
             + hc_eps; post = 2 sigmoid(mix*scale1 + base) + hc_eps; the
             branch input is sum pre_s * stream_s
  hc post    stream_s += post_s * branch output (f32, then bf16)
  attention  MLA (q lora 2048, kv lora 512, nope 192, rope 64, value 256),
             interleaved rope pairs (2i, 2i+1), theta 1e7, scale 1/sqrt(256),
             learned per-head sink logit in the softmax denominator,
             elementwise sigmoid gate (linear_gate of the normed input) on the
             64 x 256 head outputs before o_proj
  indexer    layers with indexer_types "full": q = wq_b(normed q lora), k =
             LayerNorm(wk(x)) (eps 1e-6, as vLLM hy_v4), rope on the last 64 of 128;
             score(t, s) = sum_h w_h relu(q_h . k_s) / sqrt(32 * 128) with w =
             weights_proj(x); the top 2048 positions s <= t are the attended
             set; "shared" layers reuse the last full layer's set
  moe        sigmoid router, e_score_correction_bias for selection only, top 8,
             weights normalised and scaled by 2.827; routed experts clamp the
             gate at +10 and the up projection to [-10, 10]; the shared expert
             and the dense layer 0 are not clamped
  head       hc collapse (pre weights only), RMSNorm, f32 lm_head
"""

import argparse
import json
import os
import resource
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

GROUP = 32
LOOKAHEAD = 8
CHUNK = 32 << 20
KINDS = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8}


def bf16(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def sigmoid(values):
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def silu(values):
    return (values * sigmoid(values)).astype(np.float32)


def rms_norm(values, weight, epsilon):
    inverse = np.power((values * values).mean(axis=-1, keepdims=True) + np.float32(epsilon), np.float32(-0.5))
    return bf16(weight * bf16(values * inverse))


E4M3 = np.array([(-1.0 if code & 0x80 else 1.0) * (
    (2.0 ** (((code >> 3) & 15) - 7)) * (1.0 + (code & 7) / 8.0) if (code >> 3) & 15 else (2.0 ** -6) * ((code & 7) / 8.0))
    for code in range(256)], dtype=np.float32)
E4M3[0x7F] = np.nan
E4M3[0xFF] = np.nan


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.weight_map = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}
        self.descriptors = {}
        self.guard = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.pending = {}

    def meta(self, name):
        shard = self.weight_map[name]
        with self.guard:
            if shard not in self.headers:
                with open(self.directory / shard, "rb") as handle:
                    length = struct.unpack("<Q", handle.read(8))[0]
                    self.headers[shard] = (8 + length, json.loads(handle.read(length)))
                self.descriptors[shard] = os.open(self.directory / shard, os.O_RDONLY)
        base, header = self.headers[shard]
        return self.descriptors[shard], base, header[name]

    def load(self, name, first, count):
        descriptor, base, meta = self.meta(name)
        begin, end = meta["data_offsets"]
        shape = list(meta["shape"])
        if first is not None:
            stride = (end - begin) // shape[0]
            begin, end, shape[0] = begin + first * stride, begin + (first + count) * stride, count
        buffer = np.empty(end - begin, dtype=np.uint8)
        view, done = memoryview(buffer), 0
        while done < buffer.size:
            got = os.preadv(descriptor, [view[done:done + CHUNK]], base + begin + done)
            if got <= 0:
                raise OSError(f"short read of {name} at {done}")
            done += got
        os.posix_fadvise(descriptor, base + begin, end - begin, os.POSIX_FADV_DONTNEED)
        return buffer.view(KINDS[meta["dtype"]]).reshape(shape), meta["dtype"]

    def prefetch(self, name, first=None, count=None):
        key = (name, first, count)
        if key not in self.pending:
            self.pending[key] = self.pool.submit(self.load, name, first, count)

    def prefetch_layer(self, layer):
        prefix = f"model.layers.{layer}."
        for name in self.weight_map:
            if name.startswith(prefix) and ".mlp.experts." not in name:
                self.prefetch(name)

    def read(self, name, dtype, shape=None, first=None, count=None):
        future = self.pending.pop((name, first, count), None)
        array, actual = future.result() if future is not None else self.load(name, first, count)
        if actual != dtype or (shape is not None and tuple(array.shape) != tuple(shape)):
            raise ValueError(f"{name}: {actual} {list(array.shape)}, expected {dtype} {shape}")
        return array

    def discard(self, prefix):
        unread = sorted(key[0] for key in self.pending if key[0].startswith(prefix))
        for key in [key for key in self.pending if key[0].startswith(prefix)]:
            self.pending.pop(key).result()
        return unread

    def gather(self, name, rows):
        descriptor, base, meta = self.meta(name)
        begin, end = meta["data_offsets"]
        table = np.memmap(self.directory / self.weight_map[name], dtype=KINDS[meta["dtype"]], mode="r", offset=base + begin,
                          shape=tuple(meta["shape"]))
        return np.asarray(table[rows])

    def bf16(self, name, shape=None, first=None, count=None):
        return (self.read(name, "BF16", shape, first, count).astype(np.uint32) << 16).view(np.float32)

    def f32(self, name, shape=None):
        return self.read(name, "F32", shape)

    def fp8(self, weight, scale):
        rows, columns = weight.shape
        values = np.take(E4M3, weight).reshape(rows, columns // GROUP, GROUP)
        values *= np.exp2(scale.astype(np.float32) - 127.0)[:, :, None]
        if not np.isfinite(values).all():
            raise ValueError("an E4M3 weight decodes to NaN")
        return values.reshape(rows, columns)

    def mxfp8(self, name, shape):
        return self.fp8(self.read(name + ".weight", "F8_E4M3", shape), self.read(name + ".weight_scale", "U8", (shape[0], shape[1] // GROUP)))


class Reference:
    def __init__(self, directory, lora_epsilon):
        self.checkpoint = Checkpoint(directory)
        config = json.loads((Path(directory) / "config.json").read_text())
        self.config = config
        self.hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.nope = config["qk_nope_head_dim"]
        self.rope_dim = config["qk_rope_head_dim"]
        self.value = config["v_head_dim"]
        self.latent = config["kv_lora_rank"]
        self.q_lora = config["q_lora_rank"]
        self.epsilon = config["rms_norm_eps"]
        self.lora_epsilon = lora_epsilon
        self.hc = config["hc_mult"]
        self.hc_epsilon = config["hc_eps"]
        self.hc_magnitude = config["hc_magnitude"]
        self.index_heads = config["index_n_heads"]
        self.index_dim = config["index_head_dim"]
        self.index_topk = config["index_topk"]
        self.experts = config["n_routed_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.moe_inter = config["moe_intermediate_size"]
        self.route_scale = np.float32(config["routed_scaling_factor"])
        self.swiglu_limit = np.float32(config["swiglu_limit"])
        self.scale = np.float32(1.0 / np.sqrt(self.nope + self.rope_dim))
        theta = float(config["rope_parameters"]["rope_theta"])
        self.inv_freq = 1.0 / theta ** (np.arange(0, self.rope_dim, 2, dtype=np.float64) / self.rope_dim)
        if config.get("rope_parameters", {}).get("rope_type", "default") != "default" or config["n_group"] != 1 or \
                config["gating_type"] != "elementwise" or not config["learnable_sink"] or not config["norm_topk_prob"] or \
                config["n_shared_experts"] != 1 or not config["enable_lm_head_fp32"]:
            raise ValueError("config differs from the Hy4 preview semantics this reference implements")
        self.stats = {}

    def note(self, key, value):
        self.stats.setdefault(key, []).append(float(value))

    def rope(self, values, positions):
        angles = positions[:, None].astype(np.float64) * self.inv_freq[None, :]
        shape = [values.shape[0]] + [1] * (values.ndim - 2) + [self.rope_dim // 2]
        cos = np.cos(angles).astype(np.float32).reshape(shape)
        sin = np.sin(angles).astype(np.float32).reshape(shape)
        even, odd = values[..., 0::2], values[..., 1::2]
        out = np.empty_like(values)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return bf16(out)

    def hc_weights(self, streams, fn, base, scale, post):
        rows = streams.shape[0]
        flat = streams.reshape(rows, -1)
        inverse = np.power((flat * flat).mean(axis=-1, keepdims=True) + np.float32(self.epsilon), np.float32(-0.5))
        mixes = (flat * inverse) @ fn.T
        pre = sigmoid(mixes[:, :self.hc] * scale[0] + base[:self.hc]) + np.float32(self.hc_epsilon)
        mixed = bf16(np.einsum("rs,rsd->rd", pre, streams))
        if not post:
            return mixed, None
        gate = np.float32(self.hc_magnitude) * sigmoid(mixes[:, self.hc:] * scale[1] + base[self.hc:]) + np.float32(self.hc_epsilon)
        return mixed, gate

    def indexer_select(self, layer, normed, q_lora, positions, spans):
        p = f"model.layers.{layer}.self_attn.indexer."
        c = self.checkpoint
        rows = normed.shape[0]
        query = bf16(q_lora @ c.mxfp8(p + "wq_b", (self.index_heads * self.index_dim, self.q_lora)).T).reshape(rows, self.index_heads, self.index_dim)
        key = bf16(normed @ c.mxfp8(p + "wk", (self.index_dim, self.hidden)).T)
        mean = key.mean(axis=-1, keepdims=True)
        centred = key - mean
        inverse = np.power((centred * centred).mean(axis=-1, keepdims=True) + np.float32(1e-6), np.float32(-0.5))
        key = bf16(centred * inverse * c.bf16(p + "k_norm.weight") + c.bf16(p + "k_norm.bias"))
        nope = self.index_dim - self.rope_dim
        query[..., nope:] = self.rope(query[..., nope:], positions)
        key[:, nope:] = self.rope(key[:, nope:], positions)
        weights = (normed @ c.bf16(p + "weights_proj.weight").T).astype(np.float32) * np.float32(1.0 / np.sqrt(self.index_heads * self.index_dim))
        selections = []
        for first, count in spans:
            if count <= self.index_topk:
                selections.append(None)
                continue
            q, k, w = query[first:first + count], key[first:first + count], weights[first:first + count]
            mask = np.zeros((count, count), dtype=bool)
            for begin in range(0, count, 128):
                end = min(count, begin + 128)
                score = np.einsum("thd,sd->ths", q[begin:end], k, optimize=True)
                np.maximum(score, 0.0, out=score)
                score = np.einsum("th,ths->ts", w[begin:end], score, optimize=True)
                for t in range(begin, end):
                    if t + 1 <= self.index_topk:
                        mask[t, :t + 1] = True
                    else:
                        chosen = np.argpartition(-score[t - begin, :t + 1], self.index_topk - 1)[:self.index_topk]
                        mask[t, chosen] = True
            selections.append(mask)
        return selections

    def attention(self, layer, normed, positions, spans, selections):
        p = f"model.layers.{layer}.self_attn."
        c = self.checkpoint
        rows = normed.shape[0]
        q_lora = bf16(normed @ c.mxfp8(p + "q_a_proj", (self.q_lora, self.hidden)).T)
        self.note("q_lora_mean_square_min", (q_lora * q_lora).mean(axis=-1).min())
        q_lora = rms_norm(q_lora, c.bf16(p + "q_a_layernorm.weight"), self.lora_epsilon)
        if self.config["indexer_types"][layer] == "full":
            selections[:] = self.indexer_select(layer, normed, q_lora, positions, spans)
        query = bf16(q_lora @ c.mxfp8(p + "q_b_proj", (self.heads * (self.nope + self.rope_dim), self.q_lora)).T)
        query = query.reshape(rows, self.heads, self.nope + self.rope_dim)
        query[..., self.nope:] = self.rope(query[..., self.nope:], positions)
        kv = bf16(normed @ c.mxfp8(p + "kv_a_proj_with_mqa", (self.latent + self.rope_dim, self.hidden)).T)
        latent = kv[:, :self.latent]
        self.note("kv_latent_mean_square_min", (latent * latent).mean(axis=-1).min())
        latent = rms_norm(latent, c.bf16(p + "kv_a_layernorm.weight"), self.lora_epsilon)
        key_rope = self.rope(kv[:, self.latent:], positions)
        kv_b = bf16(latent @ c.mxfp8(p + "kv_b_proj", (self.heads * (self.nope + self.value), self.latent)).T)
        kv_b = kv_b.reshape(rows, self.heads, self.nope + self.value)
        key_nope, value = kv_b[..., :self.nope], kv_b[..., self.nope:]
        sinks = c.f32(p + "learnable_sink_param", (self.heads,))
        output = np.empty((rows, self.heads, self.value), dtype=np.float32)
        for (first, count), selected in zip(spans, selections):
            window = slice(first, first + count)
            allowed = np.tril(np.ones((count, count), dtype=bool))
            if selected is not None:
                allowed &= selected
            rope_scores = np.einsum("thd,sd->hts", query[window, :, self.nope:], key_rope[window], optimize=True)
            for head in range(self.heads):
                scores = (query[window, head, :self.nope] @ key_nope[window, head].T + rope_scores[head]) * self.scale
                scores = np.where(allowed, scores, -np.inf)
                peak = np.maximum(scores.max(axis=-1, keepdims=True), sinks[head])
                weights = np.exp(scores - peak)
                denominator = weights.sum(axis=-1, keepdims=True) + np.exp(sinks[head] - peak)
                output[window, head] = bf16((weights @ value[window, head]) / denominator)
        gate = bf16(sigmoid(bf16(normed @ c.bf16(p + "linear_gate.weight", (self.heads * self.value, self.hidden)).T)))
        gated = bf16(output.reshape(rows, -1) * gate)
        return bf16(gated @ c.mxfp8(p + "o_proj", (self.hidden, self.heads * self.value)).T)

    def dense_mlp(self, layer, normed):
        p = f"model.layers.{layer}.mlp."
        c = self.checkpoint
        inter = self.config["intermediate_size"]
        gate = bf16(normed @ c.mxfp8(p + "gate_proj", (inter, self.hidden)).T)
        up = bf16(normed @ c.mxfp8(p + "up_proj", (inter, self.hidden)).T)
        return bf16(bf16(bf16(silu(gate)) * up) @ c.mxfp8(p + "down_proj", (self.hidden, inter)).T)

    def moe(self, layer, normed):
        p = f"model.layers.{layer}.mlp."
        c = self.checkpoint
        rows = normed.shape[0]
        logits = normed @ c.bf16(p + "gate.weight", (self.experts, self.hidden)).T
        scores = sigmoid(logits)
        selection = scores + c.f32(p + "gate.e_score_correction_bias", (self.experts,))
        chosen = np.argsort(-selection, axis=-1, kind="stable")[:, :self.top_k]
        weights = np.take_along_axis(scores, chosen, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + np.float32(1e-20)) * self.route_scale
        stacked = [(p + "experts.gate_up_proj", "F8_E4M3", (1, 2 * self.moe_inter, self.hidden)),
                   (p + "experts.gate_up_proj_scale", "U8", (1, 2 * self.moe_inter, self.hidden // GROUP)),
                   (p + "experts.down_proj", "F8_E4M3", (1, self.hidden, self.moe_inter)),
                   (p + "experts.down_proj_scale", "U8", (1, self.hidden, self.moe_inter // GROUP))]
        routed = np.zeros((rows, self.hidden), dtype=np.float32)
        limit = self.swiglu_limit
        for expert in range(self.experts):
            if expert + LOOKAHEAD < self.experts:
                for name, _, _ in stacked:
                    c.prefetch(name, expert + LOOKAHEAD, 1)
            if expert == self.experts // 2 and layer + 1 < self.config["num_hidden_layers"]:
                c.prefetch_layer(layer + 1)
            w_gate_up, s_gate_up, w_down, s_down = [c.read(name, dtype, shape, expert, 1)[0] for name, dtype, shape in stacked]
            token, slot = np.nonzero(chosen == expert)
            if token.size == 0:
                continue
            w = c.fp8(w_gate_up, s_gate_up)
            x = normed[token]
            g = np.minimum(bf16(x @ w[:self.moe_inter].T), limit)
            u = np.clip(bf16(x @ w[self.moe_inter:].T), -limit, limit)
            del w
            y = bf16(bf16(bf16(silu(g)) * u) @ c.fp8(w_down, s_down).T)
            routed[token] += y * weights[token, slot][:, None]
        shared = p + "shared_experts."
        g = bf16(normed @ c.mxfp8(shared + "gate_proj", (self.moe_inter, self.hidden)).T)
        u = bf16(normed @ c.mxfp8(shared + "up_proj", (self.moe_inter, self.hidden)).T)
        s = bf16(bf16(bf16(silu(g)) * u) @ c.mxfp8(shared + "down_proj", (self.hidden, self.moe_inter)).T)
        return bf16(bf16(routed) + s)

    def forward(self, sequences, last_layer=None, dump_rows=None):
        spans, first = [], 0
        for tokens in sequences:
            spans.append((first, len(tokens)))
            first += len(tokens)
        tokens = np.concatenate([np.asarray(t, dtype=np.int64) for t in sequences])
        positions = np.concatenate([np.arange(len(t), dtype=np.int64) for t in sequences])
        hidden = (self.checkpoint.gather("model.embed_tokens.weight", tokens).astype(np.uint32) << 16).view(np.float32)
        streams = np.repeat(hidden[:, None, :], self.hc, axis=1)
        selections = [None] * len(spans)
        layers = self.config["num_hidden_layers"] if last_layer is None else last_layer + 1
        dumps = []
        self.checkpoint.prefetch_layer(0)
        for layer in range(layers):
            started = time.time()
            p = f"model.layers.{layer}."
            c = self.checkpoint
            if self.config["mlp_layer_types"][layer] != "dense":
                for name in ("gate_up_proj", "gate_up_proj_scale", "down_proj", "down_proj_scale"):
                    for expert in range(LOOKAHEAD):
                        c.prefetch(p + "mlp.experts." + name, expert, 1)
            elif layer + 1 < layers:
                c.prefetch_layer(layer + 1)
            mixed, post = self.hc_weights(streams, c.f32(p + "hc_attn_layer.hc_pre.hc_fn"), c.f32(p + "hc_attn_layer.hc_pre.hc_base"),
                                          c.f32(p + "hc_attn_layer.hc_pre.hc_scale"), True)
            branch = self.attention(layer, rms_norm(mixed, c.bf16(p + "input_layernorm.weight"), self.epsilon), positions, spans, selections)
            streams = bf16(streams + post[:, :, None] * branch[:, None, :])
            mixed, post = self.hc_weights(streams, c.f32(p + "hc_mlp_layer.hc_pre.hc_fn"), c.f32(p + "hc_mlp_layer.hc_pre.hc_base"),
                                          c.f32(p + "hc_mlp_layer.hc_pre.hc_scale"), True)
            normed = rms_norm(mixed, c.bf16(p + "post_attention_layernorm.weight"), self.epsilon)
            branch = self.dense_mlp(layer, normed) if self.config["mlp_layer_types"][layer] == "dense" else self.moe(layer, normed)
            streams = bf16(streams + post[:, :, None] * branch[:, None, :])
            if dump_rows is not None:
                dumps.append(streams[dump_rows].copy())
            unread = c.discard(p)
            if unread:
                raise ValueError(f"layer {layer} tensors the reference did not use: {unread}")
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
            print(f"layer {layer} done {time.time() - started:.1f}s max_rss {rss:.2f} GiB", file=sys.stderr, flush=True)
        c = self.checkpoint
        collapsed, _ = self.hc_weights(streams, c.f32("model.hc_head.hc_head_fn"), c.f32("model.hc_head.hc_head_base"),
                                       c.f32("model.hc_head.hc_head_scale"), False)
        self.dumps = np.stack(dumps) if dumps else None
        return spans, streams, rms_norm(collapsed, c.bf16("model.norm.weight"), self.epsilon)

    def logits(self, normed, chunk=8192):
        vocab = self.config["vocab_size"]
        parts = []
        for first in range(0, vocab, chunk):
            rows = self.checkpoint.bf16("lm_head.weight", None, first, min(chunk, vocab - first))
            parts.append(normed @ rows.T)
        return np.concatenate(parts, axis=-1)


def tokenize(args):
    import jinja2
    from tokenizers import Tokenizer
    checkpoint = Path(args.checkpoint)
    environment = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message):
        raise ValueError(message)

    environment.globals["raise_exception"] = raise_exception
    template = environment.from_string((checkpoint / "chat_template.jinja").read_text())
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    rendered = []
    for prompt in json.loads(Path(args.prompts).read_text()):
        text = template.render(messages=[{"role": "user", "content": prompt["text"]}], add_generation_prompt=True, reasoning_effort="no_think")
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        rendered.append({"name": prompt["name"], "prompt_token_ids": ids})
        print(f"{prompt['name']}: {len(ids)} prompt tokens", flush=True)
    Path(args.out).write_text(json.dumps({"prompts": rendered}, indent=1))


def served_logprob(entry, token):
    if entry is None:
        return None
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        for key in (str(token), token):
            if key in entry:
                return float(entry[key])
        if "logprob" in entry:
            return float(entry["logprob"])
    if isinstance(entry, list):
        for pair in entry:
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and int(pair[0]) == int(token):
                return float(pair[1])
    return None


def score(args):
    reference = Reference(args.checkpoint, args.lora_epsilon)
    prompts = {p["name"]: p["prompt_token_ids"] for p in json.loads(Path(args.prompts).read_text())["prompts"]}
    capture = json.loads(Path(args.capture).read_text())
    names = [name for name in capture if name in prompts]
    if len(names) != len(capture):
        raise ValueError(f"capture names {sorted(set(capture) - set(prompts))} have no prompt")
    sequences = [prompts[name] + capture[name]["tokens"] for name in names]
    dump_rows, dump_labels, first = [], [], 0
    for name, tokens in zip(names, sequences):
        for row in range(len(prompts[name]) - 1, min(len(tokens), len(prompts[name]) + 3)):
            dump_rows.append(first + row)
            dump_labels.append(f"{name}:{row}")
        first += len(tokens)
    spans, _, normed = reference.forward(sequences, args.last_layer, np.array(dump_rows))
    if args.dump:
        np.savez(args.dump, streams=reference.dumps.view(np.uint32) >> 16, labels=np.array(dump_labels), normed=normed[dump_rows])
    report = {"lora_epsilon": args.lora_epsilon, "stats": {k: min(v) for k, v in reference.stats.items()}, "prompts": {}}
    starts, all_rows = [], []
    for name, (first, count) in zip(names, spans):
        starts.append(len(all_rows))
        all_rows.extend(range(first + len(prompts[name]) - 1, first + count - 1))
    all_logits = reference.logits(normed[all_rows])
    for name, (first, count), start in zip(names, spans, starts):
        prompt, served = prompts[name], capture[name]
        logits = all_logits[start:start + count - len(prompt)].astype(np.float64)
        logprobs = logits - logits.max(axis=-1, keepdims=True)
        logprobs -= np.log(np.exp(logprobs).sum(axis=-1, keepdims=True))
        rows = []
        for index, token in enumerate(served["tokens"]):
            order = np.argsort(-logits[index], kind="stable")[:5]
            entry = served["logprobs"][index] if served.get("logprobs") else None
            rows.append({"served": int(token),
                         "served_logprob": served_logprob(entry, token),
                         "reference_logprob": float(logprobs[index, token]),
                         "top": [[int(t), float(logprobs[index, t])] for t in order],
                         "margin": float(logits[index, order[0]] - logits[index, order[1]])})
        agree = sum(1 for row in rows if row["top"][0][0] == row["served"])
        ties = sum(1 for row in rows if row["top"][0][0] != row["served"] and row["margin"] < 0.1)
        hard = [i for i, row in enumerate(rows) if row["top"][0][0] != row["served"] and row["margin"] >= 0.1]
        deltas = [abs(row["served_logprob"] - row["reference_logprob"]) for row in rows if row["served_logprob"] is not None]
        report["prompts"][name] = {"prompt_tokens": len(prompt), "tokens": len(rows), "top1_agree": agree, "near_tie_disagree": ties,
                                   "hard_disagree": hard, "logprob_delta_median": float(np.median(deltas)) if deltas else None,
                                   "logprob_delta_max": float(np.max(deltas)) if deltas else None, "rows": rows}
        print(f"{name}: prompt {len(prompt)} served {len(rows)} top-1 agree {agree} near-tie {ties} hard {hard} "
              f"delta median {report['prompts'][name]['logprob_delta_median']} max {report['prompts'][name]['logprob_delta_max']}", flush=True)
    report["max_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
    Path(args.out).write_text(json.dumps(report))


def summary(args):
    report = json.loads(Path(args.report).read_text())
    print("| prompt | prompt tokens | served | top-1 agree | near-tie disagree (<0.1 nat) | hard disagree | logprob delta median | max |")
    print("|---|---|---|---|---|---|---|---|")
    for name, row in report["prompts"].items():
        median, peak = row["logprob_delta_median"], row["logprob_delta_max"]
        print(f"| {name} | {row['prompt_tokens']} | {row['tokens']} | {row['top1_agree']}/{row['tokens']} | {row['near_tie_disagree']} | "
              f"{row['hard_disagree'] or '-'} | {median if median is None else round(median, 4)} | {peak if peak is None else round(peak, 4)} |")
    print(f"stats {report.get('stats')} max_rss_gib {report.get('max_rss_gib')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("summary")
    m.add_argument("--report", required=True)
    t = sub.add_parser("tokenize")
    t.add_argument("--checkpoint", required=True)
    t.add_argument("--prompts", required=True)
    t.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--prompts", required=True)
    s.add_argument("--capture", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--lora-epsilon", type=float, default=1e-5)
    s.add_argument("--dump", help="npz of the per-layer streams (bf16 bits) of each prompt's last prompt row and first served rows")
    s.add_argument("--last-layer", type=int, help="stop after this layer (debug: the logits are then not the model's)")
    args = parser.parse_args()
    {"tokenize": tokenize, "score": score, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
