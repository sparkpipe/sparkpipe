#!/usr/bin/env python3
"""Teacher-forced numpy reference for the Gemma 4 31B text model.

  tokenize   render chat prompts with model-families/gemma4/chat_template.json
             and the checkpoint tokenizer: --prompts in.json (list of
             {name, text}) -> --out prompts.json (list of {name,
             prompt_token_ids})
  score      one forward over prompt + served tokens for every prompt:
             --capture capture.json ({name: {tokens, logprobs}} from the
             served model, e.g. lp_capture.py) and --prompts prompts.json
             -> --out report.json with, for every served token, the
             served top logprobs, the reference top-5 (token, logprob), the
             reference logprob of the served token and the top-1/top-2
             logit margin

The forward reads the BF16 checkpoint one layer at a time and rounds to
bf16 where transformers' Gemma4 text model does: norms in f32 then cast,
projections and the MLP product in bf16, rotary cos/sin in bf16, the
attention softmax in f32. Sliding layers see the last 1024 positions;
full layers use proportional rope (64 rotated pairs, base 1e6, pairs
(i, i + 256)) and k_eq_v (V is the scale-free RMS of the raw K
projection). Embeddings are scaled by bf16(sqrt(hidden)); logits are
soft-capped at 30.
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


def rms(values, weight=None, epsilon=1e-6):
    normed = values * np.power((values * values).mean(axis=-1, keepdims=True) + np.float32(epsilon), np.float32(-0.5))
    return normed * weight if weight is not None else normed


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
        config = json.loads((Path(directory) / "config.json").read_text())["text_config"]
        self.config = config
        self.hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.window = config["sliding_window"]
        self.epsilon = config["rms_norm_eps"]
        self.softcap = config["final_logit_softcapping"]
        self.embed_scale = bf16(np.float32(self.hidden ** 0.5))
        rope = config["rope_parameters"]
        sliding_dim, full_dim = config["head_dim"], config["global_head_dim"]
        self.inv_freq = {
            "sliding": (1.0 / rope["sliding_attention"]["rope_theta"] **
                        (np.arange(0, sliding_dim, 2, dtype=np.int64).astype(np.float32) / sliding_dim)).astype(np.float32),
        }
        pairs = int(rope["full_attention"]["partial_rotary_factor"] * full_dim // 2)
        rotated = (1.0 / rope["full_attention"]["rope_theta"] **
                   (np.arange(0, 2 * pairs, 2, dtype=np.int64).astype(np.float32) / full_dim)).astype(np.float32)
        self.inv_freq["full"] = np.concatenate([rotated, np.zeros(full_dim // 2 - pairs, dtype=np.float32)])

    def linear(self, values, name):
        return bf16(values @ self.checkpoint.f32(name).T)

    def rope(self, values, kind, positions):
        angles = positions[:, None].astype(np.float32) * self.inv_freq[kind][None, :]
        cos = bf16(np.cos(np.concatenate([angles, angles], axis=-1)))[:, None, :]
        sin = bf16(np.sin(np.concatenate([angles, angles], axis=-1)))[:, None, :]
        half = values.shape[-1] // 2
        rotated = np.concatenate([-values[..., half:], values[..., :half]], axis=-1)
        return bf16(bf16(values * cos) + bf16(rotated * sin))

    def attention(self, layer, normed, positions):
        p = f"{PREFIX}layers.{layer}.self_attn."
        full = self.config["layer_types"][layer] == "full_attention"
        dim = self.config["global_head_dim"] if full else self.config["head_dim"]
        kv_heads = self.config["num_global_key_value_heads"] if full else self.config["num_key_value_heads"]
        rows = normed.shape[0]
        query = self.linear(normed, p + "q_proj.weight").reshape(rows, self.heads, dim)
        key_raw = self.linear(normed, p + "k_proj.weight").reshape(rows, kv_heads, dim)
        value_raw = key_raw if full and self.config["attention_k_eq_v"] else \
            self.linear(normed, p + "v_proj.weight").reshape(rows, kv_heads, dim)
        query = self.rope(bf16(rms(query, self.checkpoint.f32(p + "q_norm.weight"), self.epsilon)), "full" if full else "sliding", positions)
        key = self.rope(bf16(rms(key_raw, self.checkpoint.f32(p + "k_norm.weight"), self.epsilon)), "full" if full else "sliding", positions)
        value = bf16(rms(value_raw, None, self.epsilon))
        group = self.heads // kv_heads
        causal = positions[None, :] <= positions[:, None]
        if not full:
            causal &= positions[None, :] > positions[:, None] - self.window
        output = np.empty((rows, self.heads, dim), dtype=np.float32)
        for head in range(self.heads):
            scores = query[:, head, :] @ key[:, head // group, :].T
            scores = np.where(causal, scores, -np.inf)
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            output[:, head, :] = bf16(weights @ value[:, head // group, :])
        return self.linear(output.reshape(rows, -1), p + "o_proj.weight")

    def mlp(self, layer, normed):
        p = f"{PREFIX}layers.{layer}.mlp."
        gate = self.linear(normed, p + "gate_proj.weight")
        up = self.linear(normed, p + "up_proj.weight")
        activated = bf16(0.5 * gate * (1.0 + np.tanh(np.float32(0.7978845608028654) * (gate + np.float32(0.044715) * gate * gate * gate))))
        return self.linear(bf16(activated * up), p + "down_proj.weight")

    def forward(self, tokens):
        positions = np.arange(len(tokens), dtype=np.int64)
        embed = self.checkpoint.raw(PREFIX + "embed_tokens.weight")
        residual = bf16((np.asarray(embed[np.asarray(tokens)]).astype(np.uint32) << 16).view(np.float32) * self.embed_scale)
        for layer in range(self.config["num_hidden_layers"]):
            p = f"{PREFIX}layers.{layer}."
            norm = lambda name, values: bf16(rms(values, self.checkpoint.f32(p + name + ".weight"), self.epsilon))
            residual = bf16(residual + norm("post_attention_layernorm", self.attention(layer, norm("input_layernorm", residual), positions)))
            residual = bf16(residual + norm("post_feedforward_layernorm", self.mlp(layer, norm("pre_feedforward_layernorm", residual))))
            residual = bf16(residual * self.checkpoint.f32(p + "layer_scalar")[0])
            print(f"layer {layer} done", file=sys.stderr, flush=True)
        return bf16(rms(residual, self.checkpoint.f32(PREFIX + "norm.weight"), self.epsilon))

    def logits(self, normed, chunk=16384):
        embed = self.checkpoint.raw(PREFIX + "embed_tokens.weight")
        parts = []
        for first in range(0, embed.shape[0], chunk):
            rows = (np.asarray(embed[first:first + chunk]).astype(np.uint32) << 16).view(np.float32)
            parts.append(normed @ rows.T)
        logits = np.concatenate(parts, axis=-1)
        return np.float32(self.softcap) * np.tanh(logits / np.float32(self.softcap))


def tokenize(args):
    from tokenizers import Tokenizer
    template = json.loads(Path(args.template).read_text())
    tokenizer = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))
    rendered = []
    for prompt in json.loads(Path(args.prompts).read_text()):
        text = template["prefix"] + template["user"] + prompt["text"] + template["turn_suffix"] + template["generation"]
        rendered.append({"name": prompt["name"], "prompt_token_ids": tokenizer.encode(text, add_special_tokens=False).ids})
    Path(args.out).write_text(json.dumps({"prompts": rendered}, indent=1))


def score(args):
    reference = Reference(args.checkpoint)
    prompts = {p["name"]: p["prompt_token_ids"] for p in json.loads(Path(args.prompts).read_text())["prompts"]}
    capture = json.loads(Path(args.capture).read_text())
    report = {}
    for name, served in capture.items():
        prompt = prompts[name]
        tokens = prompt + served["tokens"]
        normed = reference.forward(tokens)[len(prompt) - 1:len(tokens) - 1]
        logits = reference.logits(normed)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("tokenize", "score"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--capture")
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/gemma4/chat_template.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "tokenize":
        tokenize(args)
    else:
        if not args.capture:
            parser.error("score needs --capture")
        score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
