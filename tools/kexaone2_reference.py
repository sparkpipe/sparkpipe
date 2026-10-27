#!/usr/bin/env python3
"""Teacher-forced numpy reference for K-EXAONE 2.0, from the BF16 checkpoint and the publisher's modeling_exaone_moe.py.

  tokenize   render chat prompts with model-families/kexaone2/chat_template.json
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

Semantics follow model_contracts/references/modeling_exaone_moe.py
(nuxlear/transformers@969f8bdf): pre-norm blocks; RMSNorm is w * bf16(x *
rsqrt(mean x^2 + eps)) rounded to bf16; q_norm/k_norm per head; sliding
layers (window 4096 at layer 1, 128 elsewhere) apply default rope theta 1e6
on all 128 dims with cos/sin cast to bf16, global layers (0 and 4k+1 from 5)
apply none; 64 query and 8 KV heads, scale 1/sqrt(128); layers 0 and 1 are
dense SwiGLU, layers 2..77 a router with f32 logits, sigmoid scores, top-8 on
score + bias, weights the unbiased scores normalized (+1e-20) times 2.5 in
f32, each routed expert's bf16 output times its f32 weight cast to bf16 and
added in ascending expert order, plus the shared expert. Layers 62..77 clamp
the gate above at 7 before SiLU and the up projection to +-7 on every MLP.
Projections round to bf16; the attention softmax runs in f32 and its
probabilities round to bf16 before they weight the values, as the
publisher's eager attention (softmax in f32, then .to(bf16)) and flash
attention do; the final logits are f32. Every matrix product accumulates in
f64 before it is rounded, so a prompt's numbers do not depend on the other
prompts in the run or on how many served tokens follow (BLAS picks its
summation order by matrix shape, and f32 order noise rounded to bf16 over 78
layers exceeded the near-tie threshold). Every prompt goes through one layer
at a time, so each layer's weights are read once per run.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

from t1_reference_common import matmul

PREFIX = "model."


def bf16(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def rms(values, epsilon):
    return values * np.power((values * values).mean(axis=-1, keepdims=True) + np.float32(epsilon), np.float32(-0.5))


def silu(values):
    return values / (np.float32(1.0) + np.exp(-values))


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.weight_map = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

    def meta(self, name):
        shard = self.weight_map[name]
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        base, header = self.headers[shard]
        return shard, base, header[name]

    def raw(self, name):
        shard, base, meta = self.meta(name)
        if meta["dtype"] != "BF16":
            raise ValueError(f"{name} is {meta['dtype']}, expected BF16")
        begin, end = meta["data_offsets"]
        return np.memmap(self.directory / shard, dtype=np.uint16, mode="r", offset=base + begin,
                         shape=((end - begin) // 2,)).reshape(meta["shape"])

    def f32(self, name):
        shard, base, meta = self.meta(name)
        if meta["dtype"] == "F32":
            begin, end = meta["data_offsets"]
            return np.array(np.memmap(self.directory / shard, dtype=np.float32, mode="r", offset=base + begin,
                                      shape=((end - begin) // 4,)).reshape(meta["shape"]))
        return (np.asarray(self.raw(name)).astype(np.uint32) << 16).view(np.float32)


class Reference:
    def __init__(self, directory):
        self.checkpoint = Checkpoint(directory)
        config = json.loads((Path(directory) / "config.json").read_text())
        self.config = config
        self.epsilon = config["rms_norm_eps"]
        self.head_dim = config["head_dim"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.top_k = config["num_experts_per_tok"]
        self.routed_scale = np.float32(config["routed_scaling_factor"])
        self.scale = np.float32(self.head_dim ** -0.5)
        theta = config["rope_parameters"]["rope_theta"]
        self.inv_freq = (1.0 / (theta ** (np.arange(0, self.head_dim, 2, dtype=np.int64).astype(np.float32) / self.head_dim))).astype(np.float32)
        self.cache = {}

    def weight(self, name):
        if name not in self.cache:
            self.cache[name] = self.checkpoint.f32(name)
        return self.cache[name]

    def norm(self, values, name):
        return bf16(self.weight(name) * bf16(rms(values, self.epsilon)))

    def linear(self, values, name):
        return bf16(matmul(values, self.weight(name).T))

    def rope(self, values, positions):
        angles = positions[:, None].astype(np.float32) * self.inv_freq[None, :]
        emb = np.concatenate([angles, angles], axis=-1)
        cos = bf16(np.cos(emb))[:, None, :]
        sin = bf16(np.sin(emb))[:, None, :]
        half = self.head_dim // 2
        rotated = np.concatenate([-values[..., half:], values[..., :half]], axis=-1)
        return bf16(bf16(values * cos) + bf16(rotated * sin))

    def attention(self, layer, normed, positions):
        p = f"{PREFIX}layers.{layer}.self_attn."
        window = self.config["sliding_windows"][layer]
        rows, dim, heads = normed.shape[0], self.head_dim, self.heads
        query = self.norm(self.linear(normed, p + "q_proj.weight").reshape(rows, heads, dim), p + "q_norm.weight")
        key = self.norm(self.linear(normed, p + "k_proj.weight").reshape(rows, self.kv_heads, dim), p + "k_norm.weight")
        value = self.linear(normed, p + "v_proj.weight").reshape(rows, self.kv_heads, dim)
        visible = positions[None, :] <= positions[:, None]
        if window:
            query, key = self.rope(query, positions), self.rope(key, positions)
            visible &= positions[None, :] > positions[:, None] - window
        group = heads // self.kv_heads
        output = np.empty((rows, heads, dim), dtype=np.float32)
        for head in range(heads):
            scores = matmul(query[:, head, :], key[:, head // group, :].T) * self.scale
            scores = np.where(visible, scores, -np.inf)
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights = bf16(weights / weights.sum(axis=-1, keepdims=True))
            output[:, head, :] = bf16(matmul(weights, value[:, head // group, :]))
        return self.linear(output.reshape(rows, heads * dim), p + "o_proj.weight")

    def swiglu(self, gate, up, limit):
        if limit:
            gate = np.minimum(gate, np.float32(limit))
            up = np.clip(up, np.float32(-limit), np.float32(limit))
        return bf16(bf16(silu(gate)) * up)

    def mlp(self, normed, prefix, limit):
        gate = self.linear(normed, prefix + "gate_proj.weight")
        up = self.linear(normed, prefix + "up_proj.weight")
        return self.linear(self.swiglu(gate, up, limit), prefix + "down_proj.weight")

    def moe(self, layer, normed, limit):
        p = f"{PREFIX}layers.{layer}.mlp."
        shared = self.mlp(normed, p + "shared_experts.", limit)
        logits = matmul(normed, self.weight(p + "gate.weight").T)
        scores = (np.float32(1.0) / (np.float32(1.0) + np.exp(-logits))).astype(np.float32)
        selection = scores + self.weight(p + "e_score_correction_bias")[None, :]
        chosen = np.argsort(-selection, axis=-1, kind="stable")[:, :self.top_k]
        weights = np.take_along_axis(scores, chosen, axis=-1)
        weights = (weights / (weights.sum(axis=-1, keepdims=True) + np.float32(1e-20)) * self.routed_scale).astype(np.float32)
        routed = np.zeros_like(normed)
        for expert in np.unique(chosen):
            rows, slots = np.nonzero(chosen == expert)
            e = f"{p}experts.{int(expert)}."
            x = normed[rows]
            gate = bf16(matmul(x, self.checkpoint.f32(e + "gate_proj.weight").T))
            up = bf16(matmul(x, self.checkpoint.f32(e + "up_proj.weight").T))
            out = bf16(matmul(self.swiglu(gate, up, limit), self.checkpoint.f32(e + "down_proj.weight").T))
            routed[rows] = bf16(routed[rows] + bf16(out * weights[rows, slots][:, None]))
        return bf16(routed + shared), chosen

    def forward(self, sequences):
        embed = self.checkpoint.raw(PREFIX + "embed_tokens.weight")
        residual = [(np.asarray(embed[np.asarray(tokens)]).astype(np.uint32) << 16).view(np.float32) for tokens in sequences]
        positions = [np.arange(len(tokens), dtype=np.int64) for tokens in sequences]
        bounds = np.cumsum([0] + [len(tokens) for tokens in sequences])
        for layer in range(self.config["num_hidden_layers"]):
            p = f"{PREFIX}layers.{layer}."
            limit = self.config["swiglu_limits"][layer]
            self.cache = {}
            for index in range(len(sequences)):
                attended = self.attention(layer, self.norm(residual[index], p + "input_layernorm.weight"), positions[index])
                residual[index] = bf16(residual[index] + attended)
            normed = np.concatenate([self.norm(values, p + "post_attention_layernorm.weight") for values in residual])
            if self.config["mlp_layer_types"][layer] == "dense":
                out = self.mlp(normed, p + "mlp.", limit)
            else:
                out, _ = self.moe(layer, normed, limit)
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
            parts.append(matmul(normed, rows.T))
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
    environment = jinja2.Environment(extensions=[GenerationTag.extension(), "jinja2.ext.loopcontrols"], trim_blocks=True, lstrip_blocks=True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("tokenize", "score"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--capture")
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/kexaone2/chat_template.json"))
    parser.add_argument("--out")
    args = parser.parse_args()
    return tokenize(args) if args.mode == "tokenize" else score(args)


if __name__ == "__main__":
    sys.exit(main())
