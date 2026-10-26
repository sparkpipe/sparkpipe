#!/usr/bin/env python3
import argparse
import json
import math
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


def sigmoid(values):
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-values))


def fp8_table():
    codes = np.arange(256, dtype=np.uint32)
    sign, exponent, mantissa = (codes >> 7) & 1, (codes >> 3) & 0xF, codes & 0x7
    value = np.where(exponent == 0, mantissa.astype(np.float32) * np.float32(2.0 ** -9),
                     (np.float32(1.0) + mantissa.astype(np.float32) / np.float32(8.0)) *
                     np.power(np.float32(2.0), (exponent.astype(np.int32) - 7).astype(np.float32))).astype(np.float32)
    value[(exponent == 15) & (mantissa == 7)] = np.float32("nan")
    return np.where(sign == 1, -value, value).astype(np.float32)


FP8 = fp8_table()


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

    def view(self, name, dtype, width):
        shard, base, meta = self.meta(name)
        begin, end = meta["data_offsets"]
        return np.memmap(self.directory / shard, dtype=dtype, mode="r", offset=base + begin,
                         shape=((end - begin) // width,)).reshape(meta["shape"]), meta["dtype"]

    def raw(self, name):
        values, kind = self.view(name, np.uint16, 2)
        if kind != "BF16":
            raise ValueError(f"{name} is {kind}, expected BF16")
        return values

    def f32(self, name):
        _, _, meta = self.meta(name)
        if meta["dtype"] == "F32":
            return np.array(self.view(name, np.float32, 4)[0])
        if meta["dtype"] == "BF16":
            return (np.asarray(self.raw(name)).astype(np.uint32) << 16).view(np.float32)
        if meta["dtype"] != "F8_E4M3":
            raise ValueError(f"{name}: unexpected {meta['dtype']}")
        codes = np.asarray(self.view(name, np.uint8, 1)[0])
        scale = np.array(self.view(name + "_scale_inv", np.float32, 4)[0])
        rows, cols = codes.shape
        expanded = np.repeat(np.repeat(scale, 128, axis=0)[:rows], 128, axis=1)[:, :cols]
        return FP8[codes] * expanded


def yarn_inv_freq(config):
    rope = config["rope_parameters"]
    dim, base, factor = config["qk_rope_head_dim"], rope["rope_theta"], rope["factor"]
    original = rope["original_max_position_embeddings"]

    def correction(rotations):
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction(rope["beta_fast"])), 0)
    high = min(math.ceil(correction(rope["beta_slow"])), dim - 1)
    if low == high:
        high += 0.001
    frequencies = np.power(np.float32(base), np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / np.float32(dim))
    extrapolation = (np.float32(1.0) / frequencies).astype(np.float32)
    interpolation = (np.float32(1.0) / (np.float32(factor) * frequencies)).astype(np.float32)
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - np.float32(low)) / np.float32(high - low), 0, 1).astype(np.float32)
    keep = (np.float32(1.0) - ramp).astype(np.float32)
    return (interpolation * (np.float32(1.0) - keep) + extrapolation * keep).astype(np.float32)


def mscale(factor, scale):
    return 1.0 if factor <= 1 else 0.1 * scale * math.log(factor) + 1.0


class Reference:
    def __init__(self, directory):
        self.checkpoint = Checkpoint(directory)
        config = json.loads((Path(directory) / "config.json").read_text())
        self.config = config
        self.epsilon = config["rms_norm_eps"]
        self.heads = config["num_attention_heads"]
        self.nope, self.rope_dim, self.value = config["qk_nope_head_dim"], config["qk_rope_head_dim"], config["v_head_dim"]
        self.latent = config["kv_lora_rank"]
        self.index_heads, self.index_dim, self.index_topk = config["index_n_heads"], config["index_head_dim"], config["index_topk"]
        self.top_k, self.groups, self.top_groups = config["num_experts_per_tok"], config["n_group"], config["topk_group"]
        self.experts = config["n_routed_experts"]
        self.routed_scale = np.float32(config["routed_scaling_factor"])
        rope = config["rope_parameters"]
        self.scale = np.float32((self.nope + self.rope_dim) ** -0.5 * mscale(rope["factor"], rope["mscale_all_dim"]) ** 2)
        self.inv_freq = yarn_inv_freq(config)
        self.cache = {}

    def weight(self, name):
        if name not in self.cache:
            self.cache[name] = self.checkpoint.f32(name)
        return self.cache[name]

    def norm(self, values, name, epsilon=None):
        return bf16(self.weight(name) * bf16(rms(values, self.epsilon if epsilon is None else epsilon)))

    def gated_norm(self, values, prefix):
        normed = self.norm(values, prefix + "norm.weight")
        hidden = bf16(silu(bf16(normed @ self.weight(prefix + "W_down.weight").T)))
        gate = bf16(hidden @ self.weight(prefix + "W_up.weight").T)
        return bf16(normed * sigmoid(gate))

    def linear(self, values, name):
        return bf16(values @ self.weight(name).T)

    def angles(self, positions):
        freqs = positions[:, None].astype(np.float32) * self.inv_freq[None, :]
        return bf16(np.cos(freqs)), bf16(np.sin(freqs))

    def rope_interleaved(self, values, cos, sin):
        first, second = values[..., 0::2], values[..., 1::2]
        return np.concatenate([bf16(bf16(first * cos) - bf16(second * sin)), bf16(bf16(second * cos) + bf16(first * sin))], axis=-1)

    def rope_half(self, values, cos, sin):
        cos2, sin2 = np.concatenate([cos, cos], axis=-1), np.concatenate([sin, sin], axis=-1)
        half = values.shape[-1] // 2
        rotated = np.concatenate([-values[..., half:], values[..., :half]], axis=-1)
        return bf16(bf16(values * cos2) + bf16(rotated * sin2))

    def indexer(self, p, normed, q_resid, positions, cos, sin):
        rows = normed.shape[0]
        query = self.linear(q_resid, p + "indexer.wq_b.weight").reshape(rows, self.index_heads, self.index_dim)
        key = self.linear(normed, p + "indexer.wk.weight")
        mean = key.mean(axis=-1, keepdims=True)
        variance = ((key - mean) ** 2).mean(axis=-1, keepdims=True)
        key = bf16((key - mean) / np.sqrt(variance + np.float32(1e-6)) * self.weight(p + "indexer.k_norm.weight") +
                   self.weight(p + "indexer.k_norm.bias"))
        r = self.rope_dim
        query = np.concatenate([self.rope_half(query[..., :r], cos[:, None, :], sin[:, None, :]), query[..., r:]], axis=-1)
        key = np.concatenate([self.rope_half(key[..., :r], cos, sin), key[..., r:]], axis=-1)
        weights = bf16(normed @ self.weight(p + "indexer.weights_proj.weight").T) * np.float32(self.index_heads ** -0.5)
        selected = []
        for row in range(rows):
            visible = positions <= positions[row]
            scores = np.maximum((query[row] @ key.T) * np.float32(self.index_dim ** -0.5), 0)
            total = weights[row] @ scores
            total = np.where(visible, total, -np.inf)
            keep = min(self.index_topk, rows)
            selected.append(np.argsort(-total, kind="stable")[:keep])
        return selected

    def attention(self, layer, normed, positions):
        p = f"{PREFIX}layers.{layer}.self_attn."
        rows, heads = normed.shape[0], self.heads
        compressed = self.linear(normed, p + "q_a_proj.weight")
        q_resid = self.norm(compressed, p + "q_a_layernorm.weight", 1e-6)
        fused = self.linear(np.concatenate([q_resid, compressed], axis=-1), p + "q_b_proj.weight").reshape(rows, heads, -1)
        query, gate = fused[..., :self.nope + self.rope_dim], fused[..., self.nope + self.rope_dim:]
        latent = self.linear(normed, p + "kv_a_proj_with_mqa.weight")
        k_pass = self.norm(latent[:, :self.latent], p + "kv_a_layernorm.weight", 1e-6)
        cos, sin = self.angles(positions)
        q_rot = self.rope_interleaved(query[..., self.nope:], cos[:, None, :], sin[:, None, :])
        k_rot = self.rope_interleaved(latent[:, self.latent:], cos, sin)
        expanded = self.linear(k_pass, p + "kv_b_proj.weight").reshape(rows, heads, self.nope + self.value)
        k_nope, value = expanded[..., :self.nope], expanded[..., self.nope:]
        visible = positions[None, :] <= positions[:, None]
        if rows > self.index_topk:
            mask = np.zeros((rows, rows), dtype=bool)
            for row, keep in enumerate(self.indexer(p, normed, q_resid, positions, cos, sin)):
                mask[row, keep] = True
            visible &= mask
        output = np.empty((rows, heads, self.value), dtype=np.float32)
        for head in range(heads):
            scores = (query[:, head, :self.nope] @ k_nope[:, head, :].T + q_rot[:, head, :] @ k_rot.T) * self.scale
            scores = np.where(visible, scores, -np.inf)
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            output[:, head, :] = bf16(bf16(weights) @ value[:, head, :])
        output = bf16(output * sigmoid(gate))
        return self.linear(output.reshape(rows, heads * self.value), p + "o_proj.weight")

    def mlp(self, normed, prefix):
        gate = self.linear(normed, prefix + "gate_proj.weight")
        up = self.linear(normed, prefix + "up_proj.weight")
        return self.linear(bf16(bf16(silu(gate)) * up), prefix + "down_proj.weight")

    def route(self, normed, p):
        logits = (normed @ self.weight(p + "gate.weight").T).astype(np.float32)
        scores = sigmoid(logits).astype(np.float32)
        choice = scores + self.weight(p + "gate.e_score_correction_bias")[None, :]
        rows, per_group = choice.shape[0], self.experts // self.groups
        grouped = np.sort(choice.reshape(rows, self.groups, per_group), axis=-1)
        group_scores = grouped[..., -1] + grouped[..., -2]
        top_groups = np.argsort(-group_scores, axis=-1, kind="stable")[:, :self.top_groups]
        allowed = np.zeros((rows, self.groups), dtype=bool)
        np.put_along_axis(allowed, top_groups, True, axis=-1)
        choice = np.where(np.repeat(allowed, per_group, axis=-1), choice, np.float32(0.0))
        chosen = np.argsort(-choice, axis=-1, kind="stable")[:, :self.top_k]
        weights = np.take_along_axis(scores, chosen, axis=-1)
        weights = (weights / (weights.sum(axis=-1, keepdims=True) + np.float32(1e-20)) * self.routed_scale).astype(np.float32)
        return chosen, weights

    def moe(self, layer, normed):
        p = f"{PREFIX}layers.{layer}.mlp."
        shared = self.mlp(normed, p + "shared_experts.")
        chosen, weights = self.route(normed, p)
        routed = np.zeros_like(normed)
        for expert in np.unique(chosen):
            rows, slots = np.nonzero(chosen == expert)
            e = f"{p}experts.{int(expert)}."
            x = normed[rows]
            gate = bf16(x @ self.checkpoint.f32(e + "gate_proj.weight").T)
            up = bf16(x @ self.checkpoint.f32(e + "up_proj.weight").T)
            out = bf16(bf16(bf16(silu(gate)) * up) @ self.checkpoint.f32(e + "down_proj.weight").T)
            routed[rows] = bf16(routed[rows] + bf16(out * weights[rows, slots][:, None]))
        return bf16(routed + shared)

    def forward(self, sequences):
        embed = self.checkpoint.raw(PREFIX + "embed_tokens.weight")
        residual = [(np.asarray(embed[np.asarray(tokens)]).astype(np.uint32) << 16).view(np.float32) for tokens in sequences]
        positions = [np.arange(len(tokens), dtype=np.int64) for tokens in sequences]
        bounds = np.cumsum([0] + [len(tokens) for tokens in sequences])
        first_dense = self.config.get("first_k_dense_replace", 1)
        for layer in range(self.config["num_hidden_layers"]):
            p = f"{PREFIX}layers.{layer}."
            self.cache = {}
            for index in range(len(sequences)):
                attended = self.attention(layer, self.gated_norm(residual[index], p + "input_layernorm."), positions[index])
                residual[index] = bf16(residual[index] + attended)
            if layer < first_dense:
                normed = np.concatenate([self.norm(values, p + "post_attention_layernorm.weight") for values in residual])
                out = self.mlp(normed, p + "mlp.")
            else:
                normed = np.concatenate([self.gated_norm(values, p + "post_attention_layernorm.") for values in residual])
                out = self.moe(layer, normed)
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


def publisher_render(checkpoint, text):
    import jinja2
    environment = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"], trim_blocks=True, lstrip_blocks=True)
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
    print(f"{len(rendered)} prompts rendered and matched against the publisher jinja; lengths "
          f"{[len(p['prompt_token_ids']) for p in rendered]}", flush=True)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("tokenize", "score"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--capture")
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/axk2/chat_template.json"))
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.mode == "tokenize":
        tokenize(args)
    else:
        score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
