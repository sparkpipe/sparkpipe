#!/usr/bin/env python3
"""Teacher-forced numpy reference for Laguna S 2.1, from the BF16 checkpoint and the publisher's modeling_laguna.py.

  tokenize   render chat prompts with model-families/laguna/chat_template.json
             and the checkpoint tokenizer: --prompts in.json (list of
             {name, text}) -> --out prompts.json ({"prompts": [{name,
             prompt_token_ids}]}). The rendering is checked against the
             publisher's chat_template.jinja (jinja2, enable_thinking false,
             add_generation_prompt true); a difference fails.
  score      one forward over prompt + served tokens for every prompt:
             --capture capture.json ({name: {tokens, logprobs}} from the
             served model) and --prompts prompts.json -> --out report.json
             with, for every served token, the served logprob, the
             reference top-5 (token, logprob), the reference logprob of the
             served token and the top-1/top-2 logit margin.
  yarn       print the full-attention YaRN inverse frequencies computed the
             way transformers' _compute_yarn_parameters does (truncate) and
             the SparkRopePlanBuildYarnInvFrequency formula, and their
             largest relative difference.

Semantics follow modeling_laguna.py (checkpoint revision 0f573140, sha256
765fd328...): pre-norm blocks; LagunaRMSNorm is w * bf16(x * rsqrt(mean
x^2 + eps)); q_norm/k_norm per head before rope; sliding layers (72 heads,
window 512) use default rope theta 1e4 on all 128 dims, full layers (48
heads) YaRN on the first 64 dims with cos/sin scaled by attention_factor and
cast to bf16; scale 1/sqrt(128); per-head gate softplus(g_proj(x)) cast to
bf16 multiplies each head before o_proj; layer 0 is a dense SwiGLU, layers
1..47 a sigmoid router (bf16 logits widened to f32, top-10 on score + bias,
weights the unbiased scores normalized and cast to bf16), routed experts
summed in bf16 in ascending expert order, times 2.5, plus the shared expert.
Projections round to bf16; the attention softmax and the final logits are
f32. Every prompt goes through one layer at a time, so each layer's weights
are read once per run.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

PREFIX = "model."


def bf16(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def rms(values, epsilon):
    return values * np.power((values * values).mean(axis=-1, keepdims=True) + np.float32(epsilon), np.float32(-0.5))


def silu(values):
    return values / (np.float32(1.0) + np.exp(-values))


def softplus(values):
    return np.where(values > 20.0, values, np.log1p(np.exp(np.minimum(values, np.float32(20.0))))).astype(np.float32)


def yarn_publisher(rope, head_dim):
    base = np.float32(rope["rope_theta"])
    dim = int(head_dim * rope.get("partial_rotary_factor", 1.0))
    factor = rope["factor"]
    original = rope["original_max_position_embeddings"]

    def correction(rotations):
        return dim * np.log(original / (rotations * 2 * np.pi)) / (2 * np.log(rope["rope_theta"]))

    low = max(int(np.floor(correction(rope.get("beta_fast") or 32))), 0)
    high = min(int(np.ceil(correction(rope.get("beta_slow") or 1))), dim - 1)
    if low == high:
        high += 0.001
    pos_freqs = base ** (np.arange(0, dim, 2).astype(np.float32) / np.float32(dim))
    extrapolation = np.float32(1.0) / pos_freqs
    interpolation = np.float32(1.0) / (np.float32(factor) * pos_freqs)
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - np.float32(low)) / np.float32(high - low), 0, 1).astype(np.float32)
    extrapolation_factor = np.float32(1.0) - ramp
    return (interpolation * (1 - extrapolation_factor) + extrapolation * extrapolation_factor).astype(np.float32), float(rope["attention_factor"])


def yarn_spark_plan(rope, head_dim):
    dim = int(head_dim * rope["partial_rotary_factor"])
    theta, factor, original = np.float32(rope["rope_theta"]), np.float32(rope["factor"]), np.float32(rope["original_max_position_embeddings"])
    two_pi = np.float32(6.283185307179586)
    low_exact = np.float32(dim) * np.log(original / (np.float32(rope["beta_fast"]) * two_pi)) / (np.float32(2.0) * np.log(theta))
    high_exact = np.float32(dim) * np.log(original / (np.float32(rope["beta_slow"]) * two_pi)) / (np.float32(2.0) * np.log(theta))
    low = np.floor(max(min(low_exact, np.float32(dim - 1)), np.float32(0.0)))
    high = np.ceil(max(min(high_exact, np.float32(dim - 1)), np.float32(0.0)))
    index = np.arange(dim // 2, dtype=np.float32)
    base = np.power(theta, np.float32(-2.0) * index / np.float32(dim))
    blend = np.clip((index - low) / max(high - low, np.float32(1e-6)), 0, 1)
    return (base * (1 - blend) + base / factor * blend).astype(np.float32)


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.weight_map = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

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

    def f32(self, name):
        return (np.asarray(self.raw(name)).astype(np.uint32) << 16).view(np.float32)


class Reference:
    def __init__(self, directory):
        self.checkpoint = Checkpoint(directory)
        config = json.loads((Path(directory) / "config.json").read_text())
        self.config = config
        self.epsilon = config["rms_norm_eps"]
        self.head_dim = config["head_dim"]
        self.kv_heads = config["num_key_value_heads"]
        self.window = config["sliding_window"]
        self.top_k = config["num_experts_per_tok"]
        self.routed_scale = np.float32(config["moe_routed_scaling_factor"])
        self.scale = np.float32(self.head_dim ** -0.5)
        rope = config["rope_parameters"]
        sliding = rope["sliding_attention"]
        sliding_dim = int(self.head_dim * sliding.get("partial_rotary_factor", 1.0))
        self.inv_freq = {"sliding_attention": (1.0 / (sliding["rope_theta"] ** (
            np.arange(0, sliding_dim, 2, dtype=np.int64).astype(np.float32) / sliding_dim))).astype(np.float32)}
        self.rope_scale = {"sliding_attention": np.float32(1.0)}
        full, factor = yarn_publisher(rope["full_attention"], self.head_dim)
        self.inv_freq["full_attention"] = full
        self.rope_scale["full_attention"] = np.float32(factor)
        self.cache = {}

    def weight(self, name):
        if name not in self.cache:
            self.cache[name] = self.checkpoint.f32(name)
        return self.cache[name]

    def norm(self, values, name):
        return bf16(self.weight(name) * bf16(rms(values, self.epsilon)))

    def linear(self, values, name):
        return bf16(values @ self.weight(name).T)

    def rope(self, values, kind, positions):
        inv_freq = self.inv_freq[kind]
        dim = 2 * inv_freq.shape[0]
        angles = positions[:, None].astype(np.float32) * inv_freq[None, :]
        emb = np.concatenate([angles, angles], axis=-1)
        cos = bf16(np.cos(emb) * self.rope_scale[kind])[:, None, :]
        sin = bf16(np.sin(emb) * self.rope_scale[kind])[:, None, :]
        rotary, passthrough = values[..., :dim], values[..., dim:]
        half = dim // 2
        rotated = np.concatenate([-rotary[..., half:], rotary[..., :half]], axis=-1)
        return np.concatenate([bf16(bf16(rotary * cos) + bf16(rotated * sin)), passthrough], axis=-1)

    def attention(self, layer, normed, positions):
        p = f"{PREFIX}layers.{layer}.self_attn."
        kind = self.config["layer_types"][layer]
        heads = self.config["num_attention_heads_per_layer"][layer]
        rows, dim = normed.shape[0], self.head_dim
        query = self.linear(normed, p + "q_proj.weight").reshape(rows, heads, dim)
        key = self.linear(normed, p + "k_proj.weight").reshape(rows, self.kv_heads, dim)
        value = self.linear(normed, p + "v_proj.weight").reshape(rows, self.kv_heads, dim)
        query = self.rope(self.norm(query, p + "q_norm.weight"), kind, positions)
        key = self.rope(self.norm(key, p + "k_norm.weight"), kind, positions)
        group = heads // self.kv_heads
        visible = positions[None, :] <= positions[:, None]
        if kind == "sliding_attention":
            visible &= positions[None, :] > positions[:, None] - self.window
        output = np.empty((rows, heads, dim), dtype=np.float32)
        for head in range(heads):
            scores = (query[:, head, :] @ key[:, head // group, :].T) * self.scale
            scores = np.where(visible, scores, -np.inf)
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            output[:, head, :] = bf16(weights @ value[:, head // group, :])
        gate = bf16(softplus(self.linear(normed, p + "g_proj.weight")))
        output = bf16(output * gate[:, :, None])
        return self.linear(output.reshape(rows, heads * dim), p + "o_proj.weight")

    def mlp(self, normed, prefix):
        gate = self.linear(normed, prefix + "gate_proj.weight")
        up = self.linear(normed, prefix + "up_proj.weight")
        return self.linear(bf16(bf16(silu(gate)) * up), prefix + "down_proj.weight")

    def moe(self, layer, normed):
        p = f"{PREFIX}layers.{layer}.mlp."
        shared = self.mlp(normed, p + "shared_expert.")
        logits = bf16(normed @ self.weight(p + "gate.weight").T)
        scores = (np.float32(1.0) / (np.float32(1.0) + np.exp(-logits))).astype(np.float32)
        selection = scores + self.weight(p + "experts.e_score_correction_bias")[None, :]
        chosen = np.argsort(-selection, axis=-1, kind="stable")[:, :self.top_k]
        weights = np.take_along_axis(scores, chosen, axis=-1)
        weights = bf16(weights / weights.sum(axis=-1, keepdims=True))
        routed = np.zeros_like(normed)
        for expert in np.unique(chosen):
            rows, slots = np.nonzero(chosen == expert)
            e = f"{p}experts.{int(expert)}."
            x = normed[rows]
            gate = bf16(x @ self.checkpoint.f32(e + "gate_proj.weight").T)
            up = bf16(x @ self.checkpoint.f32(e + "up_proj.weight").T)
            hidden = bf16(bf16(silu(gate)) * up)
            out = bf16(hidden @ self.checkpoint.f32(e + "down_proj.weight").T)
            routed[rows] = bf16(routed[rows] + bf16(out * weights[rows, slots][:, None]))
        return bf16(bf16(routed * self.routed_scale) + shared), chosen

    def forward(self, sequences):
        embed = self.checkpoint.raw(PREFIX + "embed_tokens.weight")
        residual = [(np.asarray(embed[np.asarray(tokens)]).astype(np.uint32) << 16).view(np.float32) for tokens in sequences]
        positions = [np.arange(len(tokens), dtype=np.int64) for tokens in sequences]
        bounds = np.cumsum([0] + [len(tokens) for tokens in sequences])
        routes = []
        for layer in range(self.config["num_hidden_layers"]):
            p = f"{PREFIX}layers.{layer}."
            self.cache = {}
            for index in range(len(sequences)):
                attended = self.attention(layer, self.norm(residual[index], p + "input_layernorm.weight"), positions[index])
                residual[index] = bf16(residual[index] + attended)
            normed = np.concatenate([self.norm(values, p + "post_attention_layernorm.weight") for values in residual])
            if layer in self.config["mlp_only_layers"]:
                out = self.mlp(normed, p + "mlp.")
            else:
                out, chosen = self.moe(layer, normed)
                routes.append(chosen)
            for index in range(len(sequences)):
                residual[index] = bf16(residual[index] + out[bounds[index]:bounds[index + 1]])
            print(f"layer {layer} done", file=sys.stderr, flush=True)
        self.cache = {}
        return [self.norm(values, PREFIX + "norm.weight") for values in residual]

    def logits(self, normed, chunk=16384):
        head = self.checkpoint.raw("lm_head.weight")
        parts = []
        for first in range(0, head.shape[0], chunk):
            rows = (np.asarray(head[first:first + chunk]).astype(np.uint32) << 16).view(np.float32)
            parts.append(normed @ rows.T)
        return np.concatenate(parts, axis=-1)


def render(template, text):
    return (template["prefix"] + template.get("plain_prefix", "") + template["user"] + text + template.get("user_suffix", "") +
            template["turn_suffix"] + template["generation"])


class GenerationTag:
    @staticmethod
    def extension():
        from jinja2 import nodes
        from jinja2.ext import Extension

        class Generation(Extension):
            tags = {"generation"}

            def parse(self, parser):
                lineno = next(parser.stream).lineno
                body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
                return nodes.Scope(body, lineno=lineno)

        return Generation


def publisher_render(checkpoint, text):
    import jinja2
    environment = jinja2.Environment(extensions=[GenerationTag.extension()], keep_trailing_newline=True)
    environment.filters["tojson"] = lambda value, ensure_ascii=True, **_: json.dumps(value, ensure_ascii=ensure_ascii)
    source = (Path(checkpoint) / "chat_template.jinja").read_text()
    return environment.from_string(source).render(messages=[{"role": "user", "content": text}], add_generation_prompt=True,
                                                  enable_thinking=False, tools=None)


def tokenize(args):
    from tokenizers import Tokenizer
    template = json.loads(Path(args.template).read_text())
    tokenizer = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))
    rendered = []
    for prompt in json.loads(Path(args.prompts).read_text()):
        text = render(template, prompt["text"])
        publisher = publisher_render(args.checkpoint, prompt["text"])
        if text != publisher:
            raise SystemExit(f"{prompt['name']}: chat_template.json renders {text!r}, the publisher jinja renders {publisher!r}")
        rendered.append({"name": prompt["name"], "prompt_token_ids": tokenizer.encode(text, add_special_tokens=False).ids})
    Path(args.out).write_text(json.dumps({"prompts": rendered}, indent=1))
    print(f"{len(rendered)} prompts rendered and matched against the publisher jinja", flush=True)


def served_logprob(entry, token):
    if not entry:
        return None
    for candidate, logprob in entry:
        if candidate == token:
            return logprob
    return None


def score(args):
    reference = Reference(args.checkpoint)
    prompts = {p["name"]: p["prompt_token_ids"] for p in json.loads(Path(args.prompts).read_text())["prompts"]}
    capture = json.loads(Path(args.capture).read_text())
    names = [name for name in capture if name in prompts]
    sequences = [prompts[name] + capture[name]["tokens"] for name in names]
    finals = reference.forward(sequences)
    report = {}
    for name, sequence, final in zip(names, sequences, finals):
        prompt, served = prompts[name], capture[name]
        logits = reference.logits(final[len(prompt) - 1:len(sequence) - 1])
        logprobs = logits - logits.max(axis=-1, keepdims=True)
        logprobs -= np.log(np.exp(logprobs).sum(axis=-1, keepdims=True))
        rows = []
        for index, token in enumerate(served["tokens"]):
            order = np.argsort(-logits[index], kind="stable")[:5]
            entry = served["logprobs"][index] if served.get("logprobs") else None
            rows.append({"served": token, "served_logprob": served_logprob(entry, token), "served_top": entry,
                         "reference_logprob": float(logprobs[index, token]),
                         "top": [[int(t), float(logprobs[index, t])] for t in order],
                         "margin": float(logits[index, order[0]] - logits[index, order[1]])})
        agree = sum(1 for row in rows if row["top"][0][0] == row["served"])
        disagree = [i for i, row in enumerate(rows) if row["top"][0][0] != row["served"]]
        near_tie = [i for i in disagree if rows[i]["margin"] < 0.1]
        deltas = [abs(row["served_logprob"] - row["reference_logprob"]) for row in rows if row["served_logprob"] is not None]
        summary = {"tokens": len(rows), "top1_agree": agree, "disagree": disagree, "disagree_near_tie": near_tie,
                   "disagree_clear": [i for i in disagree if i not in near_tie],
                   "logprob_delta_median": float(np.median(deltas)) if deltas else None,
                   "logprob_delta_max": float(np.max(deltas)) if deltas else None}
        report[name] = {**summary, "rows": rows}
        print(f"{name}: {agree}/{len(rows)} top-1 agree; disagreements {disagree} (near ties {near_tie}); "
              f"|served - reference| logprob median {summary['logprob_delta_median']} max {summary['logprob_delta_max']}", flush=True)
    Path(args.out).write_text(json.dumps(report))


def yarn(args):
    rope = json.loads((Path(args.checkpoint) / "config.json").read_text())["rope_parameters"]["full_attention"]
    publisher, factor = yarn_publisher(rope, 128)
    plan = yarn_spark_plan(rope, 128)
    relative = np.abs(plan - publisher) / publisher
    print(json.dumps({"attention_factor": factor, "publisher": publisher.tolist(), "spark_plan": plan.tolist(),
                      "max_relative_difference": float(relative.max())}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("tokenize", "score", "yarn"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--capture")
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/laguna/chat_template.json"))
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.mode == "yarn":
        yarn(args)
    elif args.mode == "tokenize":
        if not args.prompts or not args.out:
            parser.error("tokenize needs --prompts and --out")
        tokenize(args)
    else:
        if not args.capture or not args.prompts or not args.out:
            parser.error("score needs --capture, --prompts and --out")
        score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
