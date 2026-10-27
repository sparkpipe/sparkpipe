#!/usr/bin/env python3
"""Teacher-forced numpy reference for Gemma 4 31B, Gemma 4 26B A4B and Muse Glimmer 30B text models.

  tokenize   render chat prompts with a chat_template.json (prefix,
             plain_prefix when present, user, turn_suffix, generation) and
             the checkpoint tokenizer:
             --prompts in.json (list of {name, text}) -> --out prompts.json
             (list of {name, prompt_token_ids})
  score      one forward over prompt + served tokens for every prompt:
             --capture capture.json ({name: {tokens, logprobs}} from the
             served model) and --prompts prompts.json -> --out report.json
             with, for every served token, the served top logprobs, the
             reference top-5 (token, logprob), the reference logprob of the
             served token and the top-1/top-2 logit margin
  teacher    teacher-forced FP32 logits for every position but the last of
             each prompt (--prompts prompts.json) to
             <out>/<name>.ref_<variant>.logits for each of --variants
             (comma-separated, one pass): a sums every reduction in numpy's
             order, b splits every matmul's inner dimension into 1,024-wide
             chunks summed last chunk first and reverses every other
             reduction, so a against b is the noise of FP32 reduction order
             under the same bf16 rounding; bp (MoE models, with a) is b with
             every layer's experts taken from a's choice instead of its own
             top-k (the weights are still b's probabilities of those
             experts, renormalised), so a against bp is that noise without
             the discrete expert choice; c and d are two more draws of the
             same noise, like b with 512-wide chunks and the attention
             scores' inner dimension in four parts (c) or with 2,048-wide
             chunks and the scores unsplit (d). MoE models also write
             <out>/<name>.ref_<variant>.routes for a and b, the chosen
             experts as little-endian uint16 [layer][position][rank];
             --hidden also writes <out>/<name>.ref_<variant>.hidden, every
             layer's output residual as BF16 [layer][position][hidden]
  gate-prompts  the Metal gate prompts (the same instructions and repository
             files as the Qwen3.8-27B gate) in this model's chat framing,
             each file cut to the target token count -> --out prompts.json
  spec-prompts  the Metal speculation prompts (the same as the Qwen3.8-27B
             gate's: a whole-file rewrite of tools/token_prompts.c and a prose
             request) in this model's chat framing -> --out prompts.json
  summary    --report report.json -> a per-prompt markdown table: top-1
             agreement, mismatches whose reference top-1/top-2 margin is at
             least --tie (default 0.1 nat; smaller margins are near ties),
             first divergence, median and max |served - reference| logprob
             of the served token; exits 1 if any mismatch is not a near tie

The model kind comes from config.json text_config: model_type
muse_glimmer_text selects Muse Glimmer, enable_moe_block the Gemma 4 MoE
block, otherwise the Gemma 4 dense text model. Weights are read from the BF16
checkpoint one layer (one expert) at a time and all prompts go through each
layer together. Values are rounded to bf16 where the publisher's eager
PyTorch code rounds: norms in f32 then cast, projections in bf16, elementwise
bf16 products, rotary cos/sin in bf16. The attention softmax and P.V run in
f32 (SDPA/flash), logits are computed in f32.

Gemma 4 (transformers modeling_gemma4.py): embeddings scaled by
bf16(sqrt(hidden)); RMSNorm multiplies by w; q/k norms weighted, v norm
weightless, attention scale 1; sliding layers see the last sliding_window
positions with default rope; full layers use proportional rope
(partial_rotary_factor of the pairs, pairs (i, i + head_dim/2)) and, with
attention_k_eq_v, V is the raw K projection; MLP gelu_tanh; layer_scalar
after each layer; logits soft-capped. MoE layers add the router (weightless
norm of the residual times scale times hidden^-0.5, softmax, top-k,
renormalised, times per_expert_scale), experts on pre_feedforward_layernorm_2
of the residual, accumulated in bf16 in ascending expert order, and the
post_feedforward_layernorm_1/_2 branch norms.

Muse Glimmer (modeling_muse_glimmer.py): the embedding is a weightless RMS
norm of the table row (ids image_token_id and video_token_id map to 0); the
four in-layer norms are centred (1 + w), post norms use post_norm_eps; one
weightless qk norm, q times qk_scale_factor after the cast, no v norm,
scale head_dim^-0.5; rope only where layer_rope_theta is non-zero; sigmoid
output gate on the attention input; SwiGLU MLP; final norm multiplies by w;
untied lm_head; logits = cap * tanh(z * output_multiplier / cap).
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

PREFIX = "model.language_model."


def bf16(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


REVERSED = False
CHUNK = 1024
QK_SPLIT = 2
FLOOR = {"a": (False, 1024, 2), "b": (True, 1024, 2), "c": (True, 512, 4), "d": (True, 2048, 1)}


def reduce_sum(values, axis=-1, keepdims=False):
    return np.flip(values, axis=axis).sum(axis=axis, keepdims=keepdims) if REVERSED else values.sum(axis=axis, keepdims=keepdims)


def matmul(left, right, chunk=None):
    chunk = CHUNK if chunk is None else chunk
    if not REVERSED or left.shape[-1] <= chunk:
        return left @ right
    total = None
    for first in reversed(range(0, left.shape[-1], chunk)):
        part = left[..., first:first + chunk] @ right[first:first + chunk]
        total = part if total is None else total + part
    return total


def rms(values, weight=None, epsilon=1e-6):
    mean = reduce_sum(values * values, keepdims=True) / np.float32(values.shape[-1])
    normed = values * np.power(mean + np.float32(epsilon), np.float32(-0.5))
    return normed * weight if weight is not None else normed


def gelu_tanh(values):
    cube = values * values * values
    return 0.5 * values * (1.0 + np.tanh(np.float32(0.7978845608028654) * (values + np.float32(0.044715) * cube)))


def sigmoid(values):
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-values))


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.weight_map = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}
        self.cache = {}

    def raw(self, name):
        shard = self.weight_map[name]
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        base, header = self.headers[shard]
        meta = header[name]
        if meta["dtype"] != "BF16":
            raise ValueError(f"{name} is {meta['dtype']}, the reference reads BF16 only")
        begin, end = meta["data_offsets"]
        return np.memmap(self.directory / shard, dtype=np.uint16, mode="r", offset=base + begin,
                         shape=((end - begin) // 2,)).reshape(meta["shape"])

    def f32(self, name, index=None):
        key = (name, index)
        if key not in self.cache:
            self.cache[key] = widen(self.raw(name) if index is None else self.raw(name)[index])
        return self.cache[key]


def widen(raw):
    return (np.asarray(raw).astype(np.uint32) << 16).view(np.float32)


class Reference:
    def __init__(self, directory):
        self.checkpoint = Checkpoint(directory)
        top = json.loads((Path(directory) / "config.json").read_text())
        config = top["text_config"]
        self.config = config
        self.muse = config.get("model_type") == "muse_glimmer_text"
        self.moe = bool(config.get("enable_moe_block"))
        self.hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.window = config["sliding_window"]
        self.epsilon = config["rms_norm_eps"]
        self.softcap = config["final_logit_softcapping"]
        self.inv_freq = {}
        self.sequence = 0
        self.order_of = []
        self.pin_source = []
        self.head = None
        self.routes = {}
        self.layer_outputs = None
        if self.muse:
            self.post_epsilon = config["post_norm_eps"]
            self.qk_scale = np.float32(config["qk_scale_factor"])
            self.output_multiplier = np.float32(config["output_multiplier"])
            self.multimodal_ids = {top[key] for key in ("image_token_id", "video_token_id") if key in top}
            theta = config["rope_parameters"]["rope_theta"]
            if config["rope_parameters"].get("rope_type", "default") != "default" or \
                    any(value not in (0, theta) for value in config["layer_rope_theta"]):
                raise ValueError("Muse reference expects default rope and layer_rope_theta entries of 0 or rope_theta")
            self.rope_layers = [bool(value) for value in config["layer_rope_theta"]]
            dim = config["head_dim"]
            self.inv_freq["sliding"] = (1.0 / theta ** (np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / dim)).astype(np.float32)
        else:
            self.embed_scale = bf16(np.float32(self.hidden ** 0.5))
            rope = config["rope_parameters"]
            sliding_dim, full_dim = config["head_dim"], config["global_head_dim"]
            self.inv_freq["sliding"] = (1.0 / rope["sliding_attention"]["rope_theta"] **
                                        (np.arange(0, sliding_dim, 2, dtype=np.float64) / sliding_dim)).astype(np.float32)
            pairs = int(rope["full_attention"]["partial_rotary_factor"] * full_dim // 2)
            rotated = (1.0 / rope["full_attention"]["rope_theta"] **
                       (np.arange(0, 2 * pairs, 2, dtype=np.float64) / full_dim)).astype(np.float32)
            self.inv_freq["full"] = np.concatenate([rotated, np.zeros(full_dim // 2 - pairs, dtype=np.float32)])

    def linear(self, values, name):
        return bf16(matmul(values, self.checkpoint.f32(name).T))

    def norm(self, values, name, epsilon=None):
        epsilon = self.epsilon if epsilon is None else epsilon
        weight = self.checkpoint.f32(name)
        return bf16(rms(values, np.float32(1.0) + weight if self.muse else weight, epsilon))

    def rope(self, values, kind, positions):
        angles = positions[:, None].astype(np.float32) * self.inv_freq[kind][None, :]
        cos = bf16(np.cos(np.concatenate([angles, angles], axis=-1)))[:, None, :]
        sin = bf16(np.sin(np.concatenate([angles, angles], axis=-1)))[:, None, :]
        half = values.shape[-1] // 2
        rotated = np.concatenate([-values[..., half:], values[..., :half]], axis=-1)
        return bf16(bf16(values * cos) + bf16(rotated * sin))

    def attend(self, query, key, value, positions, window, scale):
        rows, heads, _ = query.shape
        group = heads // key.shape[1]
        causal = positions[None, :] <= positions[:, None]
        if window:
            causal &= positions[None, :] > positions[:, None] - window
        output = np.empty((rows, heads, value.shape[-1]), dtype=np.float32)
        for head in range(heads):
            scores = matmul(query[:, head, :], key[:, head // group, :].T, query.shape[-1] // QK_SPLIT) * np.float32(scale)
            scores = np.where(causal, scores, -np.inf)
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights /= reduce_sum(weights, keepdims=True)
            output[:, head, :] = bf16(matmul(weights, value[:, head // group, :]))
        return output.reshape(rows, -1)

    def attention(self, layer, normed, positions):
        p = f"{PREFIX}layers.{layer}.self_attn."
        full = self.config["layer_types"][layer] == "full_attention"
        rows = normed.shape[0]
        if self.muse:
            dim, kv_heads = self.config["head_dim"], self.config["num_key_value_heads"]
            query = self.linear(normed, p + "q_proj.weight").reshape(rows, self.heads, dim)
            key = self.linear(normed, p + "k_proj.weight").reshape(rows, kv_heads, dim)
            value = self.linear(normed, p + "v_proj.weight").reshape(rows, kv_heads, dim)
            query = bf16(bf16(rms(query, None, self.epsilon)) * self.qk_scale)
            key = bf16(rms(key, None, self.epsilon))
            if self.rope_layers[layer]:
                query, key = self.rope(query, "sliding", positions), self.rope(key, "sliding", positions)
            output = self.attend(query, key, value, positions, 0 if full else self.window, dim ** -0.5)
            output = bf16(output * bf16(sigmoid(self.linear(normed, p + "gate_proj.weight"))))
            return self.linear(output, p + "o_proj.weight")
        dim = self.config["global_head_dim"] if full else self.config["head_dim"]
        kv_heads = self.config["num_global_key_value_heads"] if full else self.config["num_key_value_heads"]
        query = self.linear(normed, p + "q_proj.weight").reshape(rows, self.heads, dim)
        key_raw = self.linear(normed, p + "k_proj.weight").reshape(rows, kv_heads, dim)
        value_raw = key_raw if full and self.config["attention_k_eq_v"] else \
            self.linear(normed, p + "v_proj.weight").reshape(rows, kv_heads, dim)
        kind = "full" if full else "sliding"
        query = self.rope(bf16(rms(query, self.checkpoint.f32(p + "q_norm.weight"), self.epsilon)), kind, positions)
        key = self.rope(bf16(rms(key_raw, self.checkpoint.f32(p + "k_norm.weight"), self.epsilon)), kind, positions)
        value = bf16(rms(value_raw, None, self.epsilon))
        return self.linear(self.attend(query, key, value, positions, 0 if full else self.window, 1.0), p + "o_proj.weight")

    def mlp(self, layer, normed):
        p = f"{PREFIX}layers.{layer}.mlp."
        gate = self.linear(normed, p + "gate_proj.weight")
        up = self.linear(normed, p + "up_proj.weight")
        activated = bf16(gate * sigmoid(gate)) if self.muse else bf16(gelu_tanh(gate))
        return self.linear(bf16(activated * up), p + "down_proj.weight")

    def experts(self, layer, residual):
        p = f"{PREFIX}layers.{layer}."
        top_k, inter = self.config["top_k_experts"], self.config["moe_intermediate_size"]
        routed = bf16(rms(residual, None, self.epsilon))
        routed = bf16(bf16(routed * self.checkpoint.f32(p + "router.scale")) * np.float32(self.hidden ** -0.5))
        logits = self.linear(routed, p + "router.proj.weight")
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= reduce_sum(probabilities, keepdims=True)
        chosen = np.argsort(-probabilities, axis=-1, kind="stable")[:, :top_k]
        if self.pin_source and self.pin_source[self.sequence] is not None:
            chosen = self.routes[self.pin_source[self.sequence]][layer].astype(np.int64)
        self.routes.setdefault(self.sequence, []).append(chosen.astype(np.uint16))
        weights = np.take_along_axis(probabilities, chosen, axis=-1)
        weights = (weights / reduce_sum(weights, keepdims=True)) * self.checkpoint.f32(p + "router.per_expert_scale")[chosen]
        source = self.norm(residual, p + "pre_feedforward_layernorm_2.weight")
        output = np.zeros_like(residual)
        for expert in np.unique(chosen):
            rows, slots = np.nonzero(chosen == expert)
            projected = bf16(matmul(source[rows], self.checkpoint.f32(p + "experts.gate_up_proj", expert).T))
            activated = bf16(bf16(gelu_tanh(projected[:, :inter])) * projected[:, inter:])
            contribution = bf16(bf16(matmul(activated, self.checkpoint.f32(p + "experts.down_proj", expert).T)) * weights[rows, slots][:, None])
            output[rows] = bf16(output[rows] + contribution)
        return output

    def embed(self, tokens):
        table = self.checkpoint.raw(PREFIX + "embed_tokens.weight")
        if self.muse:
            ids = np.asarray([0 if token in self.multimodal_ids else token for token in tokens])
            return bf16(rms(widen(table[ids]), None, self.epsilon))
        return bf16(widen(table[np.asarray(tokens)]) * self.embed_scale)

    def layer(self, layer, residual, positions):
        p = f"{PREFIX}layers.{layer}."
        post = self.post_epsilon if self.muse else self.epsilon
        attended = self.attention(layer, self.norm(residual, p + "input_layernorm.weight"), positions)
        residual = bf16(residual + self.norm(attended, p + "post_attention_layernorm.weight", post))
        dense = self.mlp(layer, self.norm(residual, p + "pre_feedforward_layernorm.weight"))
        if self.moe:
            dense = bf16(self.norm(dense, p + "post_feedforward_layernorm_1.weight") +
                         self.norm(self.experts(layer, residual), p + "post_feedforward_layernorm_2.weight"))
        residual = bf16(residual + self.norm(dense, p + "post_feedforward_layernorm.weight", post))
        if not self.muse:
            residual = bf16(residual * self.checkpoint.f32(p + "layer_scalar")[0])
        return residual

    def forward(self, sequences):
        global REVERSED, CHUNK, QK_SPLIT
        residuals = [self.embed(tokens) for tokens in sequences]
        positions = [np.arange(len(tokens), dtype=np.int64) for tokens in sequences]
        for layer in range(self.config["num_hidden_layers"]):
            self.checkpoint.cache.clear()
            for index, position in enumerate(positions):
                self.sequence = index
                if self.order_of:
                    REVERSED, CHUNK, QK_SPLIT = self.order_of[index]
                residuals[index] = self.layer(layer, residuals[index], position)
                if self.layer_outputs is not None:
                    self.layer_outputs.setdefault(index, []).append((np.ascontiguousarray(residuals[index], dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16))
            print(f"layer {layer} done", file=sys.stderr, flush=True)
        self.checkpoint.cache.clear()
        final = widen(self.checkpoint.raw(PREFIX + "norm.weight"))
        normed = []
        for index, residual in enumerate(residuals):
            if self.order_of:
                REVERSED, CHUNK, QK_SPLIT = self.order_of[index]
            normed.append(bf16(rms(residual, final, self.epsilon)))
        return normed

    def logits(self, normed, chunk=16384):
        if self.head is None:
            self.head = widen(self.checkpoint.raw("lm_head.weight" if self.muse else PREFIX + "embed_tokens.weight"))
        head = self.head
        logits = np.concatenate([matmul(normed, head[first:first + chunk].T) for first in range(0, head.shape[0], chunk)], axis=-1)
        if self.muse:
            logits = logits * self.output_multiplier
        return np.float32(self.softcap) * np.tanh(logits / np.float32(self.softcap))


def tokenize(args):
    from tokenizers import Tokenizer
    template = json.loads(Path(args.template).read_text())
    tokenizer = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))
    rendered = []
    for prompt in json.loads(Path(args.prompts).read_text()):
        text = template["prefix"] + (template.get("plain_prefix") or "") + template["user"] + prompt["text"] + template["turn_suffix"] + template["generation"]
        rendered.append({"name": prompt["name"], "prompt_token_ids": tokenizer.encode(text, add_special_tokens=False).ids})
    Path(args.out).write_text(json.dumps({"prompts": rendered}, indent=1))


def score(args):
    reference = Reference(args.checkpoint)
    prompts = {p["name"]: p["prompt_token_ids"] for p in json.loads(Path(args.prompts).read_text())["prompts"]}
    capture = json.loads(Path(args.capture).read_text())
    names = list(capture)
    normed = reference.forward([prompts[name] + capture[name]["tokens"] for name in names])
    report = {}
    for name, hidden in zip(names, normed):
        served, prompt = capture[name], prompts[name]
        logits = reference.logits(hidden[len(prompt) - 1:len(prompt) - 1 + len(served["tokens"])])
        logprobs = logits - logits.max(axis=-1, keepdims=True)
        logprobs -= np.log(np.exp(logprobs).sum(axis=-1, keepdims=True))
        rows = []
        for index, token in enumerate(served["tokens"]):
            order = np.argsort(-logits[index], kind="stable")[:5]
            rows.append({"served": token,
                         "served_top": served["logprobs"][index] if served.get("logprobs") else None,
                         "reference_logprob": float(logprobs[index, token]),
                         "top": [[int(t), float(logprobs[index, t])] for t in order],
                         "margin": float(logits[index, order[0]] - logits[index, order[1]])})
        agree = sum(1 for row in rows if row["top"][0][0] == row["served"])
        first = next((i for i, row in enumerate(rows) if row["top"][0][0] != row["served"]), None)
        report[name] = {"tokens": len(rows), "top1_agree": agree, "first_divergence": first,
                        "first_divergence_margin": rows[first]["margin"] if first is not None else None, "rows": rows}
        print(f"{name}: {agree}/{len(rows)} top-1 agree, first divergence {first}"
              + (f" (reference margin {rows[first]['margin']:.4f})" if first is not None else ""), flush=True)
    Path(args.out).write_text(json.dumps(report))


GATE_PROMPTS = (
    ("design", "docs/INFERENCE_OS_DESIGN.md", "Summarize the design described below in five short bullet points.\n\n", 1100),
    ("code", "inference/kernels/linear_attn.cuh", "Review this CUDA header. Explain what it does and name one risk in it.\n\n", 1800),
    ("korean", "site/index-ko.html", "다음 글을 세 문장으로 요약해 주세요.\n\n", 700),
    ("rules", "sparkpipe_invariants.md", "Read these engineering rules, then say which two matter most for a new driver and why.\n\n", 2040),
)


SPEC_PROMPTS = (
    ("rewrite", "tools/token_prompts.c",
     "Rewrite this C file in full. Rename SparkTokenPrompt to SparkPromptTokens everywhere and change nothing else. Output only the file.\n\n"),
    ("prose", None,
     "Write a short essay, about four hundred words, on why lighthouses were built where they were and what their keepers' days were like."),
)


def spec_prompts(args):
    import hashlib
    from tokenizers import Tokenizer
    root = Path(__file__).resolve().parents[1]
    template = json.loads(Path(args.template).read_text())
    tokenizer = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    head, tail = encode(template["prefix"] + template["user"]), encode(template["turn_suffix"] + template["generation"])
    prompts = []
    for name, source, instruction in SPEC_PROMPTS:
        path = root / source if source else None
        tokens = head + encode(instruction + (path.read_text(encoding="utf-8") if path else "")) + tail
        prompts.append({"name": name, "source": source, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
                        "instruction": instruction, "tokens": len(tokens), "prompt_token_ids": tokens})
        print(f"{name}: {len(tokens)} tokens", flush=True)
    Path(args.out).write_text(json.dumps({"tokenizer_sha256": hashlib.sha256((Path(args.checkpoint) / "tokenizer.json").read_bytes()).hexdigest(),
                                          "prompts": prompts}, ensure_ascii=False) + "\n", encoding="utf-8")


def gate_prompts(args):
    import hashlib
    import html
    import re
    from tokenizers import Tokenizer
    root = Path(__file__).resolve().parents[1]
    template = json.loads(Path(args.template).read_text())
    tokenizer = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    head, tail = encode(template["prefix"] + template["user"]), encode(template["turn_suffix"] + template["generation"])
    prompts = []
    for name, source, instruction, target in GATE_PROMPTS:
        path = root / source
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".html":
            text = re.sub(r"(?s)<(script|style)[^>]*>.*?</\1>", " ", text)
            text = html.unescape(re.sub(r"<[^>]+>", " ", text))
            text = re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t\r\f\v]+", " ", text)).strip()
        body = encode(instruction + text)
        keep = target - len(head) - len(tail)
        if len(body) < keep:
            raise ValueError(f"{source} yields {len(body)} tokens, fewer than the {keep} the prompt needs")
        tokens = head + body[:keep] + tail
        prompts.append({"name": name, "source": source, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "instruction": instruction, "tokens": len(tokens), "prompt_token_ids": tokens})
        print(f"{name}: {len(tokens)} tokens from {source}", flush=True)
    Path(args.out).write_text(json.dumps({"tokenizer_sha256": hashlib.sha256((Path(args.checkpoint) / "tokenizer.json").read_bytes()).hexdigest(),
                                          "prompts": prompts}, ensure_ascii=False) + "\n", encoding="utf-8")


def teacher(args):
    global REVERSED, CHUNK, QK_SPLIT
    variants = args.variants.split(",")
    if not variants or any(label not in ("a", "b", "bp", "c", "d") for label in variants) or len(set(variants)) != len(variants):
        raise ValueError("--variants takes distinct a, b, bp, c and d")
    reference = Reference(args.checkpoint)
    if "bp" in variants and ("a" not in variants or not reference.moe):
        raise ValueError("bp pins b to a's experts: it needs a in the same pass and an MoE model")
    prompts = json.loads(Path(args.prompts).read_text())["prompts"]
    runs = [(prompt, label) for label in variants for prompt in prompts]
    first_a = variants.index("a") * len(prompts) if "a" in variants else None
    reference.order_of = [FLOOR["b" if label == "bp" else label] for _, label in runs]
    reference.pin_source = [first_a + index % len(prompts) if label == "bp" else None for index, (_, label) in enumerate(runs)]
    if "bp" in variants and variants.index("bp") < variants.index("a"):
        raise ValueError("list a before bp so a's experts are chosen first in every layer")
    if args.hidden:
        reference.layer_outputs = {}
    normed = reference.forward([prompt["prompt_token_ids"] for prompt, _ in runs])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for index, ((prompt, label), hidden) in enumerate(zip(runs, normed)):
        if args.hidden:
            np.ascontiguousarray(np.stack(reference.layer_outputs[index]), dtype="<u2").tofile(out / f"{prompt['name']}.ref_{label}.hidden")
        REVERSED, CHUNK, QK_SPLIT = FLOOR["b" if label == "bp" else label]
        if reference.moe and label != "bp":
            np.ascontiguousarray(np.stack(reference.routes[index]), dtype="<u2").tofile(out / f"{prompt['name']}.ref_{label}.routes")
        with open(out / f"{prompt['name']}.ref_{label}.logits", "wb") as handle:
            for first in range(0, len(prompt["prompt_token_ids"]) - 1, 128):
                last = min(first + 128, len(prompt["prompt_token_ids"]) - 1)
                handle.write(np.ascontiguousarray(reference.logits(hidden[first:last]), dtype=np.float32).tobytes())
        print(f"{prompt['name']}: {len(prompt['prompt_token_ids']) - 1} positions, variant {label}", flush=True)


def summary(args):
    report = json.loads(Path(args.report).read_text())
    print("| prompt | tokens | top-1 agree | mismatches at margin >= %.2f | first divergence (margin) | served-token logprob delta median / max |" % args.tie)
    print("|---|---|---|---|---|---|")
    failing = 0
    for name, entry in report.items():
        deltas, hard = [], 0
        for row in entry["rows"]:
            if row["top"][0][0] != row["served"] and row["margin"] >= args.tie:
                hard += 1
            served = next((lp for token, lp in row["served_top"] or [] if token == row["served"] and lp is not None), None)
            if served is not None:
                deltas.append(abs(served - row["reference_logprob"]))
        failing += hard
        first = entry["first_divergence"]
        first_text = "none" if first is None else f"{first} ({entry['first_divergence_margin']:.3f})"
        delta_text = f"{float(np.median(deltas)):.4f} / {max(deltas):.4f}" if deltas else "no served logprobs"
        print(f"| {name} | {entry['tokens']} | {entry['top1_agree']}/{entry['tokens']} | {hard} | {first_text} | {delta_text} |")
    return 1 if failing else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tokenize")
    t.add_argument("--checkpoint", required=True)
    t.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/gemma4/chat_template.json"))
    t.add_argument("--prompts", required=True)
    t.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--prompts", required=True)
    s.add_argument("--capture", required=True)
    s.add_argument("--out", required=True)
    g = sub.add_parser("gate-prompts")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/gemma4/chat_template.json"))
    g.add_argument("--out", required=True)
    sp = sub.add_parser("spec-prompts")
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/gemma4/chat_template.json"))
    sp.add_argument("--out", required=True)
    r = sub.add_parser("teacher")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--prompts", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--variants", default="a")
    r.add_argument("--hidden", action="store_true")
    m = sub.add_parser("summary")
    m.add_argument("--report", required=True)
    m.add_argument("--tie", type=float, default=0.1)
    args = parser.parse_args()
    if args.command == "summary":
        return summary(args)
    {"tokenize": tokenize, "score": score, "gate-prompts": gate_prompts, "spec-prompts": spec_prompts, "teacher": teacher}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
