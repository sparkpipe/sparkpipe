#!/usr/bin/env python3
"""Teacher-forced numpy reference for Motif 3, from the BF16 checkpoint and the publisher's modeling_motif.py.

  tokenize   render chat prompts with the checkpoint's chat_template.jinja
             (jinja2, add_generation_prompt true) and tokenizer: --prompts
             in.json (list of {name, text}) -> --out prompts.json
             ({"prompts": [{name, prompt_token_ids}]}).
  score      one forward over prompt + served tokens for every prompt:
             --capture capture.json ({name: {tokens, logprobs}}) and --prompts
             prompts.json -> --out report.json with, for every served token,
             the reference top-5, the reference logprob of the served token
             and the top-1/top-2 margin.
  gate-prompts  the Metal gate prompts (tools/gemma4_reference.py's set) in
             Motif's chat turn (thinking off) -> --out prompts.json
  teacher    FP32 logits of every prompt position: --prompts prompts.json ->
             --out DIR/<name>.ref_<v>.logits for --variants a,h. Variant b adds
             each row's routed experts in descending instead of ascending
             expert order; h is b with a GPU's exp error
             (tools/reference_device_error.py), the second reference whose
             distance from a is the gate's floor.

Semantics follow the publisher's modeling_motif.py at revision 2ed2ed5c:
- Four hyper-connection streams start as copies of the embedding. Each site
  takes the RMS norm of the 4x4096 row (eps 1e-6, rounded to bf16, times its
  bf16 weight), bf16 projections read as f32, pre = sigmoid(clamp(a p + b,
  +-10)), post = sigmoid(clamp(a p + b, +-10)) (coefficient 1), residual =
  20 Sinkhorn passes (rows then columns, sums floored at 1e-8) over
  exp(clamp(a p + b, +-20)). The block input is bf16(sum_s pre_s x_s); the
  block output is bf16(residual @ x + post x out) in f32.
- GDLA attention: q_latent = q_norm(x wq_a) in f32, q = bf16(q_latent wq_b),
  gate = bf16(bf16(q_latent) wq_b_gate); latent | k_pe = bf16(x wkv_a);
  k_nope | v = bf16(kv_norm(latent) wkv_b) per KV head; rope (rotate-half on
  the 64 rope dims, cos/sin rounded to bf16) with YaRN on full layers
  (layer % 4 == 0) and plain theta 1e4 on window layers (129 keys). The
  softmax runs in f32 on f32 scores (scale 192^-0.5, times mscale^2 on full
  layers) and its probabilities round to bf16. Heads come in 16 groups of
  five: out = sig - bf16(sigmoid(lambda)) x noise, times bf16(sigmoid(gate)),
  then wo.
- MLPs: gate and up clamped at +-1e6; PolyNorm poly = sigmoid(w0) n(g^3) +
  sigmoid(w1) n(g^2) + sigmoid(w2) n(g) + b in f32 with n(x) = x /
  sqrt(mean x^2 + 1e-6). Dense layers and the shared expert: bf16(poly x u)
  then x 0.5 in bf16; routed experts: b clamped at +-0.5, poly x u clamped at
  +-1e6, x 0.5 in f32, then bf16. The router takes f32 sigmoid scores, picks
  the top 8 of scores + expert_bias, and weights the unbiased scores
  normalized (+1e-20) times 2. Routed outputs (bf16) times their weight add
  in f32 in ascending expert order, plus the shared expert, then round once.
- The end takes the mean of the four streams, model.norm and lm_head (f32).

Every matrix product accumulates in f64 before it is rounded, except in the
teacher's FP32 variants.
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_device_error as device  # noqa: E402
from t1_reference_common import matmul  # noqa: E402

PREFIX = "model."
FP32 = False
REVERSED = False


def bf16(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def product(left, right):
    if FP32:
        return np.asarray(left, dtype=np.float32) @ np.asarray(right, dtype=np.float32)
    return matmul(left, right)


def sigmoid(values):
    return (np.float32(1.0) / (np.float32(1.0) + device.exp(-values))).astype(np.float32)


def rms_scale(values, epsilon):
    return np.power((values * values).mean(axis=-1, keepdims=True) + np.float32(epsilon), np.float32(-0.5))


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
            raise ValueError(f"{name} is {meta['dtype']}, expected BF16")
        begin, end = meta["data_offsets"]
        return np.memmap(self.directory / shard, dtype=np.uint16, mode="r", offset=base + begin, shape=((end - begin) // 2,)).reshape(meta["shape"])

    def f32(self, name):
        return (np.asarray(self.raw(name)).astype(np.uint32) << 16).view(np.float32)


def yarn_inv_freq(dim, theta, original, factor, beta_fast, beta_slow, end):
    freqs = (1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))).astype(np.float32)
    if end <= original:
        return freqs

    def correction(rotations):
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(theta))
    low, high = max(math.floor(correction(beta_fast)), 0), min(math.ceil(correction(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    smooth = 1 - ramp
    return (freqs / factor * (1 - smooth) + freqs * smooth).astype(np.float32)


class Reference:
    def __init__(self, directory):
        self.checkpoint = Checkpoint(directory)
        config = json.loads((Path(directory) / "config.json").read_text())
        self.config = config
        self.epsilon = config["rms_norm_eps"]
        self.heads, self.kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        self.groups = config["num_noise_heads"]
        self.group = self.heads // self.groups
        self.head_dim, self.rope_dim, self.value_dim = config["head_dim"], config["qk_rope_head_dim"], config["v_head_dim"]
        self.nope = self.head_dim - self.rope_dim
        self.latent = config["kv_lora_rank"]
        self.top_k = config["experts_top_k"]
        self.route_scale = np.float32(config["route_scale"])
        self.period = config["sliding_window_period"]
        self.window = config["sliding_window"] + 1
        self.clamp = np.float32(config["hidden_clamp"])
        self.out_scale = np.float32(config["polynorm_output_scale"])
        self.bias_clamp = np.float32(config["polynorm_bias_clamp"])
        self.streams = config["mhc_expansion_rate"]
        self.sinkhorn = config["mhc_sinkhorn_iters"]
        self.post_coefficient = np.float32(1.0 + float(config.get("mhc_h_post_alpha_end", 0.0)))
        rope = config["rope_scaling"]
        self.full_freq = yarn_inv_freq(self.rope_dim, rope["rope_theta"], rope["original_max_position_embeddings"], rope["factor"],
                                       rope["beta_fast"], rope["beta_slow"], config["max_position_embeddings"])
        theta = config["swa_rope_theta"]
        self.window_freq = (1.0 / (theta ** (np.arange(0, self.rope_dim, 2, dtype=np.int64).astype(np.float32) / self.rope_dim))).astype(np.float32)
        mscale = 0.1 * config["mscale"] * math.log(config["rope_factor"]) + 1.0
        self.full_scale = np.float32(self.head_dim ** -0.5 * mscale * mscale)
        self.window_scale = np.float32(self.head_dim ** -0.5)
        self.cache = {}

    def weight(self, name):
        if name not in self.cache:
            self.cache[name] = self.checkpoint.f32(name)
        return self.cache[name]

    def norm(self, values, name, epsilon=None):
        return bf16(self.weight(name) * bf16(values * rms_scale(values, self.epsilon if epsilon is None else epsilon)))

    def linear(self, values, name):
        return bf16(product(values, self.weight(name).T))

    def hyper(self, site, x):
        rows = x.shape[0]
        flat = x.reshape(rows, -1)
        normed = self.norm(flat, site + "rms_norm.weight", 1e-6)
        s = self.streams
        pre = self.linear(normed, site + "proj_pre.weight")
        post = self.linear(normed, site + "proj_post.weight")
        res = self.linear(normed, site + "proj_res.weight").reshape(rows, s, s)
        pre = sigmoid(np.clip(self.weight(site + "alpha_pre") * pre + self.weight(site + "bias_pre"), -10, 10))
        post = self.post_coefficient * sigmoid(np.clip(self.weight(site + "alpha_post") * post + self.weight(site + "bias_post"), -10, 10))
        m = device.exp(np.clip(self.weight(site + "alpha_res") * res + self.weight(site + "bias_res"), -20, 20)).astype(np.float32)
        for _ in range(self.sinkhorn):
            m = m / np.maximum(m.sum(axis=-1, keepdims=True), np.float32(1e-8))
            m = m / np.maximum(m.sum(axis=-2, keepdims=True), np.float32(1e-8))
        return pre.astype(np.float32), post.astype(np.float32), m.astype(np.float32)

    @staticmethod
    def combine(x, residual, post, out):
        return bf16(np.einsum("rij,rjd->rid", residual, x) + post[:, :, None] * out[:, None, :])

    def rope(self, values, positions, freq):
        angles = positions[:, None].astype(np.float32) * freq[None, :]
        emb = np.concatenate([angles, angles], axis=-1)
        cos, sin = bf16(np.cos(emb)), bf16(np.sin(emb))
        shape = (values.shape[0],) + (1,) * (values.ndim - 2) + (values.shape[-1],)
        cos, sin = cos.reshape(shape), sin.reshape(shape)
        half = values.shape[-1] // 2
        rotated = np.concatenate([-values[..., half:], values[..., :half]], axis=-1)
        return bf16(values * cos + rotated * sin)

    def attention(self, layer, x, positions):
        p = f"{PREFIX}layers.{layer}.self_attn."
        rows, window = x.shape[0], layer % self.period != 0
        q_latent = product(x, self.weight(p + "wq_a.weight").T).astype(np.float32)
        q_latent = self.weight(p + "q_norm.weight") * (q_latent * rms_scale(q_latent, self.epsilon))
        q = bf16(product(q_latent, self.weight(p + "wq_b.weight").T)).reshape(rows, self.heads, self.head_dim)
        gate = self.linear(bf16(q_latent), p + "wq_b_gate.weight").reshape(rows, -1, self.value_dim)
        kv = self.linear(x, p + "wkv_a.weight")
        latent, k_pe = kv[:, :self.latent], kv[:, self.latent:]
        freq = self.window_freq if window else self.full_freq
        q_pe = self.rope(q[:, :, self.nope:], positions, freq)
        k_pe = self.rope(k_pe, positions, freq)
        kv_b = self.linear(self.norm(latent, p + "kv_norm.weight"), p + "wkv_b.weight").reshape(rows, self.kv_heads, self.nope + self.value_dim)
        k_nope, value = kv_b[:, :, :self.nope], kv_b[:, :, self.nope:]
        lam = self.linear(x, p + "lambda_proj.weight")
        visible = positions[None, :] <= positions[:, None]
        if window:
            visible &= positions[None, :] > positions[:, None] - self.window
        scale = self.window_scale if window else self.full_scale
        per_kv = self.heads // self.kv_heads
        heads = np.empty((rows, self.heads, self.value_dim), dtype=np.float32)
        for head in range(self.heads):
            kv_head = head // per_kv
            scores = (product(q[:, head, :self.nope], k_nope[:, kv_head, :].T) + product(q_pe[:, head, :], k_pe.T)) * scale
            scores = np.where(visible, scores, -np.inf)
            weights = device.exp(scores - scores.max(axis=-1, keepdims=True))
            weights = bf16(weights / weights.sum(axis=-1, keepdims=True))
            heads[:, head, :] = bf16(product(weights, value[:, kv_head, :]))
        grouped = heads.reshape(rows, self.groups, self.group, self.value_dim)
        signal = grouped[:, :, :self.group - 1, :].reshape(rows, -1, self.value_dim)
        noise = np.repeat(grouped[:, :, self.group - 1, :], self.group - 1, axis=1)
        out = bf16(signal - bf16(bf16(sigmoid(lam))[:, :, None] * noise))
        out = bf16(out * bf16(sigmoid(gate)))
        return self.linear(out.reshape(rows, -1), p + "wo.weight")

    def poly(self, gate, up, weight, bias, routed):
        gate = np.clip(gate, -self.clamp, self.clamp)
        up = np.clip(up, -self.clamp, self.clamp)
        w = sigmoid(weight)
        b = np.clip(bias, -self.bias_clamp, self.bias_clamp) if routed else bias

        def n(values):
            return values / np.sqrt((values * values).mean(axis=-1, keepdims=True) + np.float32(1e-6))
        poly = (w[..., 0:1] * n(gate ** 3) + w[..., 1:2] * n(gate ** 2) + w[..., 2:3] * n(gate) + b).astype(np.float32)
        if routed:
            return bf16(np.clip(poly * up, -self.clamp, self.clamp) * self.out_scale)
        return bf16(bf16(poly * up) * self.out_scale)

    def mlp(self, x, prefix):
        gate = self.linear(x, prefix + "gate_proj.weight")
        up = self.linear(x, prefix + "up_proj.weight")
        hidden = self.poly(gate, up, self.weight(prefix + "act_fn.weight"), self.weight(prefix + "act_fn.bias"), False)
        return self.linear(hidden, prefix + "down_proj.weight")

    def moe(self, layer, x):
        p = f"{PREFIX}layers.{layer}.moe."
        logits = product(x, self.weight(p + "router.gate.weight").T).astype(np.float32)
        scores = sigmoid(logits)
        chosen = np.argsort(-(scores + self.weight(p + "expert_bias")[None, :]), axis=-1, kind="stable")[:, :self.top_k]
        weights = np.take_along_axis(scores, chosen, axis=-1)
        weights = (weights / (weights.sum(axis=-1, keepdims=True) + np.float32(1e-20)) * self.route_scale).astype(np.float32)
        gate_up = self.checkpoint.raw(p + "experts.gate_up_proj")
        down = self.checkpoint.raw(p + "experts.down_proj")
        act_w = self.weight(p + "experts.act_fn.weight")
        act_b = self.weight(p + "experts.act_fn.bias")
        inter = down.shape[2]
        routed = np.zeros_like(x)
        experts = np.unique(chosen)
        for expert in (experts[::-1] if REVERSED else experts):
            rows, slots = np.nonzero(chosen == expert)
            w13 = (np.asarray(gate_up[int(expert)]).astype(np.uint32) << 16).view(np.float32)
            w2 = (np.asarray(down[int(expert)]).astype(np.uint32) << 16).view(np.float32)
            fused = bf16(product(x[rows], w13.T))
            hidden = self.poly(fused[:, :inter], fused[:, inter:], act_w[int(expert)][None, :], act_b[int(expert)][None, :], True)
            out = bf16(product(hidden, w2.T))
            routed[rows] = routed[rows] + out * weights[rows, slots][:, None]
        return bf16(routed + self.mlp(x, p + "shared_experts."))

    def forward(self, sequences):
        embed = self.checkpoint.raw(PREFIX + "embed_tokens.weight")
        streams = []
        for tokens in sequences:
            rows = (np.asarray(embed[np.asarray(tokens)]).astype(np.uint32) << 16).view(np.float32)
            streams.append(np.repeat(rows[:, None, :], self.streams, axis=1))
        positions = [np.arange(len(tokens), dtype=np.int64) for tokens in sequences]
        bounds = np.cumsum([0] + [len(tokens) for tokens in sequences])
        first_routed = self.config["n_dense_first_layers"]
        for layer in range(self.config["num_hidden_layers"]):
            p = f"{PREFIX}layers.{layer}."
            self.cache = {}
            for index in range(len(sequences)):
                x = streams[index]
                pre, post, res = self.hyper(p + "mhc_attn.", x)
                reduced = bf16((x * pre[:, :, None]).sum(axis=1))
                out = self.attention(layer, self.norm(reduced, p + "input_layernorm.weight"), positions[index])
                streams[index] = self.combine(x, res, post, out)
            sites = [self.hyper(p + "mhc_ffn.", h) for h in streams]
            reduced = np.concatenate([bf16((h * site[0][:, :, None]).sum(axis=1)) for h, site in zip(streams, sites)])
            normed = self.norm(reduced, p + "post_attention_layernorm.weight")
            out = self.mlp(normed, p + "mlp.") if layer < first_routed else self.moe(layer, normed)
            for index, (h, site) in enumerate(zip(streams, sites)):
                streams[index] = self.combine(h, site[2], site[1], out[bounds[index]:bounds[index + 1]])
            print(f"layer {layer} done", file=sys.stderr, flush=True)
        self.cache = {}
        return [self.norm(bf16(h.mean(axis=1)), PREFIX + "norm.weight") for h in streams]

    def logits(self, normed, chunk=16384):
        head = self.checkpoint.raw("lm_head.weight")
        parts = []
        for first in range(0, head.shape[0], chunk):
            rows = (np.asarray(head[first:first + chunk]).astype(np.uint32) << 16).view(np.float32)
            parts.append(product(normed, rows.T))
        return np.concatenate(parts, axis=-1)


def tokenize(args):
    import jinja2
    from tokenizers import Tokenizer
    checkpoint = Path(args.checkpoint)
    template = jinja2.Environment(trim_blocks=False, lstrip_blocks=False).from_string((checkpoint / "chat_template.jinja").read_text())
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    rendered = []
    for prompt in json.loads(Path(args.prompts).read_text()):
        text = template.render(messages=[{"role": "user", "content": prompt["text"]}], add_generation_prompt=True)
        rendered.append({"name": prompt["name"], "prompt_token_ids": tokenizer.encode(text, add_special_tokens=False).ids})
    Path(args.out).write_text(json.dumps({"prompts": rendered}, indent=1))
    print(f"{len(rendered)} prompts rendered", flush=True)


def served_logprob(entry, token):
    for candidate, logprob in entry or []:
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
            rows.append({"served": token, "served_logprob": served_logprob(entry, token), "reference_logprob": float(logprobs[index, token]),
                         "top": [[int(t), float(logprobs[index, t])] for t in order],
                         "margin": float(logits[index, order[0]] - logits[index, order[1]])})
        agree = sum(1 for row in rows if row["top"][0][0] == row["served"])
        disagree = [i for i, row in enumerate(rows) if row["top"][0][0] != row["served"]]
        report[name] = {"tokens": len(rows), "top1_agree": agree, "disagree": disagree,
                        "disagree_near_tie": [i for i in disagree if rows[i]["margin"] < 0.1], "rows": rows}
        print(f"{name}: {agree}/{len(rows)} top-1 agree; disagreements {disagree}", flush=True)
    Path(args.out).write_text(json.dumps(report))


def teacher(args):
    global FP32, REVERSED
    variants = args.variants.split(",")
    if not variants or any(label not in ("a", "b", "h") for label in variants) or len(set(variants)) != len(variants):
        raise SystemExit("--variants takes distinct a, b and h")
    reference = Reference(args.checkpoint)
    prompts = json.loads(Path(args.prompts).read_text())["prompts"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    FP32 = True
    for label in variants:
        REVERSED = label != "a"
        device.select(label)
        finals = reference.forward([prompt["prompt_token_ids"] for prompt in prompts])
        for prompt, final in zip(prompts, finals):
            count = len(prompt["prompt_token_ids"])
            with open(out / f"{prompt['name']}.ref_{label}.logits", "wb") as handle:
                for row in range(0, count - 1, 128):
                    handle.write(np.ascontiguousarray(reference.logits(final[row:min(row + 128, count - 1)]), dtype=np.float32).tobytes())
            print(f"{prompt['name']}: {count - 1} positions, variant {label}", flush=True)


def gate_prompts(args):
    import gemma4_reference
    gemma4_reference.gate_prompts(args, ("prefix", "user"), ("user_suffix", "generation"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("tokenize", "score", "teacher", "gate-prompts"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--variants", default="a,h")
    parser.add_argument("--capture")
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/motif/chat_template.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    {"tokenize": tokenize, "score": score, "teacher": teacher, "gate-prompts": gate_prompts}[args.mode](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
