#!/usr/bin/env python3
"""Teacher-forced numpy reference for the MiMo V2.6 Flash / Pro text tower.

  tokenize  render --prompts (list of {name, text[, answer]}) with the
            publisher's chat_template.jinja (enable_thinking false), check the
            text equals model-families/mimo26/chat_template.json's rendering,
            encode with the checkpoint tokenizer -> --out prompts.json
            ({"prompts": [{name, prompt_token_ids}]}); --answers-out writes a
            capture-shaped {name: {tokens}} of the encoded answers, for a
            self-check of the reference alone
  score     one teacher-forced forward over prompt + served tokens for all
            prompts together: --capture ({name: {tokens, logprobs}} from the
            served model) and --prompts -> --out report.json with, per served
            token, the reference top-5, the reference logprob of the served
            token, the served logprob and the reference top-1/top-2 margin
  teacher   teacher-forced FP32 logits for every position but the last of
            each prompt (--prompts prompts.json) to
            <out>/<name>.ref_<variant>.logits for each of --variants: a sums
            in numpy's order, b splits every matmul's inner dimension into
            1,024-wide chunks summed last chunk first, reverses every other
            reduction and combines the experts in descending order, so a
            against b is the noise of FP32 reduction order, and h is b with a GPU's exp and tanh error (tools/reference_device_error.py):
            a against h is the Metal gate's floor
  gate-prompts, spec-prompts   the Metal gate and speculation prompts
            (tools/gemma4_reference.py's sources and cuts) in MiMo's chat
            framing -> --out prompts.json

Built from the checkpoint and the publisher's semantics only (config.json,
modeling_mimo_v2.py in the checkpoint directory), never from SparkPipe code:
- weights are dequantised exactly: fp8 e4m3 x weight_scale_inv per 128x128
  block; MXFP4 e2m1 (element 2i in the low nibble) x 2^(e8m0 - 127) per 32
  inputs; everything else is BF16 as stored
- the fused qkv_proj is stored as T = index metadata tp_size segments
  [q_s | k_s | v_s], its scale grid blocked per segment with each segment
  padded to whole 128-row blocks; the reference projects every segment and
  concatenates the q, k and v parts in segment order
- RMSNorm bf16(w * bf16(x * rsqrt(mean(x^2) + eps))); projections, the value
  scale, the rope products and the SwiGLU products round to bf16 as the
  modeling code does; rope on the first int(192 * 0.334) = 64 dims of each
  head (rotate_half on those 64), cos/sin in bf16, theta 1e7 full / 1e4 SWA
- attention scores, the softmax and the probabilities stay f32 (the eager
  path would round scores and probabilities to bf16; flash kernels do not);
  SWA keys j with i - 128 < j <= i; SWA layers append the per-head sink
  logit to the softmax and drop it from the output
- router: f32 logits, sigmoid, top-8 by sigmoid + e_score_correction_bias,
  weights = chosen sigmoid / (sum + 1e-20); expert outputs (bf16) combine
  in f32; logits f32 from the bf16 final norm and the untied bf16 lm_head
"""

import argparse
import json
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import reference_device_error as device

BLOCK = 128
GROUP = 32
DTYPES = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8}


def bf16(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


REVERSED = False
CHUNK = 1024


def reduce_sum(values, axis=-1, keepdims=False):
    return np.flip(values, axis=axis).sum(axis=axis, keepdims=keepdims) if REVERSED else values.sum(axis=axis, keepdims=keepdims)


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
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
E2M1_PAIRS = np.stack([E2M1[np.arange(256) & 15], E2M1[np.arange(256) >> 4]], axis=1)


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        index = json.loads((self.directory / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.metadata = index.get("metadata") or {}
        self.headers = {}

    def raw(self, name, dtype):
        shard = self.weight_map[name]
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        base, header = self.headers[shard]
        meta = header[name]
        if meta["dtype"] != dtype:
            raise ValueError(f"{name} is {meta['dtype']}, expected {dtype}")
        begin, end = meta["data_offsets"]
        kind = DTYPES[dtype]
        return np.memmap(self.directory / shard, dtype=kind, mode="r", offset=base + begin,
                         shape=((end - begin) // np.dtype(kind).itemsize,)).reshape(meta["shape"])

    def bf16(self, name):
        return (np.asarray(self.raw(name, "BF16")).astype(np.uint32) << 16).view(np.float32)

    def f32(self, name):
        return np.asarray(self.raw(name, "F32"))

    def fp8(self, name):
        weight = self.raw(name + ".weight", "F8_E4M3")
        scale = self.f32(name + ".weight_scale_inv")
        rows, cols = weight.shape
        if scale.shape != (-(-rows // BLOCK), cols // BLOCK):
            raise ValueError(f"{name}: scale {scale.shape} is not the 128x128 grid of {weight.shape}")
        grid = np.repeat(np.repeat(scale, BLOCK, axis=0)[:rows], BLOCK, axis=1)
        return E4M3[np.asarray(weight)] * grid

    def mxfp4(self, name):
        weight = np.asarray(self.raw(name + ".weight", "U8"))
        scale = np.asarray(self.raw(name + ".weight_scale", "U8"))
        rows, half = weight.shape
        if scale.shape != (rows, 2 * half // GROUP) or np.any(scale == 255):
            raise ValueError(f"{name}: e8m0 scale {scale.shape} does not fit {weight.shape} or holds NaN")
        values = E2M1_PAIRS[weight].reshape(rows, 2 * half // GROUP, GROUP)
        return (values * np.exp2(scale.astype(np.float32) - 127.0)[:, :, None]).reshape(rows, 2 * half)


def rms(values, weight, epsilon):
    mean = reduce_sum(values * values, keepdims=True) / np.float32(values.shape[-1])
    normed = bf16(values * np.power(mean + np.float32(epsilon), np.float32(-0.5)))
    return bf16(weight * normed)


class Reference:
    def __init__(self, directory, threads):
        self.checkpoint = Checkpoint(directory)
        config = json.loads((Path(directory) / "config.json").read_text())
        self.config = config
        self.threads = threads
        self.hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"]
        self.value_dim = config["v_head_dim"]
        self.rope_dim = int(self.head_dim * config["partial_rotary_factor"])
        self.window = config["sliding_window"]
        self.epsilon = config["layernorm_epsilon"]
        self.scaling = np.float32(self.head_dim ** -0.5)
        self.top_k = config["num_experts_per_tok"]
        self.interleave = int(self.checkpoint.metadata["tp_size"])
        if config["swa_head_dim"] != self.head_dim or config["swa_v_head_dim"] != self.value_dim or \
                config["swa_num_attention_heads"] != self.heads or config.get("routed_scaling_factor") is not None or \
                config["n_group"] != 1 or config["topk_group"] != 1 or config["scoring_func"] != "sigmoid":
            raise ValueError("config is outside what this reference implements")
        self.inv_freq = {}
        for kind, theta in (("full", config["rope_theta"]), ("swa", config["swa_rope_theta"])):
            exponents = np.arange(0, self.rope_dim, 2, dtype=np.int64).astype(np.float32) / np.float32(self.rope_dim)
            self.inv_freq[kind] = (np.float32(1.0) / np.power(np.float32(theta), exponents)).astype(np.float32)

    def swa(self, layer):
        return self.config["hybrid_layer_pattern"][layer] == 1

    def rope(self, values, kind, positions):
        angles = positions[:, None].astype(np.float32) * self.inv_freq[kind][None, :]
        emb = np.concatenate([angles, angles], axis=-1)
        cos, sin = bf16(np.cos(emb))[:, None, :], bf16(np.sin(emb))[:, None, :]
        part, rest = values[..., :self.rope_dim], values[..., self.rope_dim:]
        half = self.rope_dim // 2
        rotated = np.concatenate([-part[..., half:], part[..., :half]], axis=-1)
        return np.concatenate([bf16(bf16(part * cos) + bf16(rotated * sin)), rest], axis=-1)

    def qkv(self, layer, normed):
        a = f"model.layers.{layer}.self_attn."
        kv = self.config["swa_num_key_value_heads"] if self.swa(layer) else self.config["num_key_value_heads"]
        t = self.interleave
        q_rows, k_rows, v_rows = self.heads // t * self.head_dim, kv // t * self.head_dim, kv // t * self.value_dim
        segment = q_rows + k_rows + v_rows
        blocks = -(-segment // BLOCK)
        weight = self.checkpoint.raw(a + "qkv_proj.weight", "F8_E4M3")
        scale = self.checkpoint.f32(a + "qkv_proj.weight_scale_inv")
        if weight.shape != (t * segment, self.hidden) or scale.shape != (t * blocks, self.hidden // BLOCK):
            raise ValueError(f"layer {layer}: qkv {weight.shape} / {scale.shape} is not {t} segments of {segment} rows")
        parts = ([], [], [])
        for s in range(t):
            grid = np.repeat(np.repeat(scale[s * blocks:(s + 1) * blocks], BLOCK, axis=0)[:segment], BLOCK, axis=1)
            projected = bf16(matmul(normed, (E4M3[np.asarray(weight[s * segment:(s + 1) * segment])] * grid).T))
            parts[0].append(projected[:, :q_rows])
            parts[1].append(projected[:, q_rows:q_rows + k_rows])
            parts[2].append(projected[:, q_rows + k_rows:])
        rows = normed.shape[0]
        return (np.concatenate(parts[0], axis=1).reshape(rows, self.heads, self.head_dim),
                np.concatenate(parts[1], axis=1).reshape(rows, kv, self.head_dim),
                np.concatenate(parts[2], axis=1).reshape(rows, kv, self.value_dim), kv)

    def attention(self, layer, normed, spans):
        a = f"model.layers.{layer}.self_attn."
        swa = self.swa(layer)
        query, key, value, kv = self.qkv(layer, normed)
        value = bf16(value * np.float32(self.config["attention_value_scale"]))
        sinks = self.checkpoint.bf16(a + "attention_sink_bias") if swa else None
        group = self.heads // kv
        output = np.empty((normed.shape[0], self.heads, self.value_dim), dtype=np.float32)
        for first, count in spans:
            positions = np.arange(count, dtype=np.int64)
            q = self.rope(query[first:first + count], "swa" if swa else "full", positions)
            k = self.rope(key[first:first + count], "swa" if swa else "full", positions)
            v = value[first:first + count]
            visible = positions[None, :] <= positions[:, None]
            if swa:
                visible &= positions[None, :] > positions[:, None] - self.window
            for head in range(self.heads):
                scores = matmul(q[:, head, :], k[:, head // group, :].T) * self.scaling
                scores = np.where(visible, scores, -np.inf)
                top = scores.max(axis=-1, keepdims=True)
                if sinks is not None:
                    top = np.maximum(top, sinks[head])
                weights = device.exp(scores - top)
                total = reduce_sum(weights, keepdims=True)
                if sinks is not None:
                    total = total + device.exp(sinks[head] - top)
                output[first:first + count, head, :] = bf16(matmul(weights / total, v[:, head // group, :]))
        return bf16(matmul(output.reshape(normed.shape[0], -1), self.checkpoint.bf16(a + "o_proj.weight").T))

    @staticmethod
    def swiglu(gate, up):
        return bf16(bf16(gate / (np.float32(1.0) + device.exp(-gate))) * up)

    def dense(self, layer, normed):
        m = f"model.layers.{layer}.mlp."
        gate = bf16(matmul(normed, self.checkpoint.fp8(m + "gate_proj").T))
        up = bf16(matmul(normed, self.checkpoint.fp8(m + "up_proj").T))
        return bf16(matmul(self.swiglu(gate, up), self.checkpoint.fp8(m + "down_proj").T))

    def moe(self, layer, normed):
        m = f"model.layers.{layer}.mlp."
        logits = matmul(normed, self.checkpoint.bf16(m + "gate.weight").T)
        scores = np.float32(1.0) / (np.float32(1.0) + device.exp(-logits))
        choice = scores + self.checkpoint.f32(m + "gate.e_score_correction_bias")[None, :]
        chosen = np.argsort(-choice, axis=-1, kind="stable")[:, :self.top_k]
        weights = np.take_along_axis(scores, chosen, axis=-1)
        weights = weights / (reduce_sum(weights, keepdims=True) + np.float32(1e-20))
        output = np.zeros(normed.shape, dtype=np.float32)
        experts = np.unique(chosen)[::-1] if REVERSED else np.unique(chosen)

        def run(expert):
            tokens, slots = np.nonzero(chosen == expert)
            e = f"{m}experts.{expert}."
            x = normed[tokens]
            gate = bf16(matmul(x, self.checkpoint.mxfp4(e + "gate_proj").T))
            up = bf16(matmul(x, self.checkpoint.mxfp4(e + "up_proj").T))
            y = bf16(matmul(self.swiglu(gate, up), self.checkpoint.mxfp4(e + "down_proj").T))
            return tokens, y * weights[tokens, slots][:, None]

        with ThreadPoolExecutor(self.threads) as pool:
            for tokens, contribution in pool.map(run, experts):
                np.add.at(output, tokens, contribution)
        return bf16(output)

    def forward(self, sequences):
        spans, first = [], 0
        for tokens in sequences:
            spans.append((first, len(tokens)))
            first += len(tokens)
        embed = self.checkpoint.raw("model.embed_tokens.weight", "BF16")
        flat = np.concatenate([np.asarray(tokens, dtype=np.int64) for tokens in sequences])
        residual = (np.asarray(embed[flat]).astype(np.uint32) << 16).view(np.float32)
        for layer in range(self.config["num_hidden_layers"]):
            started = time.time()
            p = f"model.layers.{layer}."
            residual = bf16(residual + self.attention(layer, rms(residual, self.checkpoint.bf16(p + "input_layernorm.weight"), self.epsilon), spans))
            normed = rms(residual, self.checkpoint.bf16(p + "post_attention_layernorm.weight"), self.epsilon)
            if self.config["moe_layer_freq"][layer]:
                mlp = self.moe(layer, normed)
            else:
                mlp = self.dense(layer, normed)
            residual = bf16(residual + mlp)
            print(f"layer {layer} {'swa' if self.swa(layer) else 'full'} done in {time.time() - started:.1f}s", file=sys.stderr, flush=True)
        return rms(residual, self.checkpoint.bf16("model.norm.weight"), self.epsilon), spans

    def logits(self, normed, chunk=16384):
        head = self.checkpoint.raw("lm_head.weight", "BF16")
        parts = []
        for first in range(0, head.shape[0], chunk):
            parts.append(matmul(normed, (np.asarray(head[first:first + chunk]).astype(np.uint32) << 16).view(np.float32).T))
        return np.concatenate(parts, axis=-1)


def render_json(template, text):
    return template["prefix"] + template["user"] + text + template["turn_suffix"] + template["generation"]


def tokenize(args):
    import jinja2
    from tokenizers import Tokenizer
    template = json.loads(Path(args.template).read_text())
    environment = jinja2.Environment()
    environment.filters["tojson"] = lambda value, ensure_ascii=False: json.dumps(value, ensure_ascii=ensure_ascii)
    publisher = environment.from_string((Path(args.checkpoint) / "chat_template.jinja").read_text())
    tokenizer = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))
    rendered, answers = [], {}
    for prompt in json.loads(Path(args.prompts).read_text()):
        text = publisher.render(messages=[{"role": "user", "content": prompt["text"]}], add_generation_prompt=True, enable_thinking=False)
        if text != render_json(template, prompt["text"]):
            raise SystemExit(f"{prompt['name']}: the publisher template renders {text!r}, the SparkPipe template {render_json(template, prompt['text'])!r}")
        rendered.append({"name": prompt["name"], "prompt_token_ids": tokenizer.encode(text, add_special_tokens=False).ids})
        if "answer" in prompt:
            answers[prompt["name"]] = {"tokens": tokenizer.encode(prompt["answer"], add_special_tokens=False).ids, "logprobs": None}
    Path(args.out).write_text(json.dumps({"prompts": rendered}, indent=1))
    if args.answers_out:
        Path(args.answers_out).write_text(json.dumps(answers))


def served_logprob(entry):
    if entry is None:
        return None
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict) and "logprob" in entry:
        return float(entry["logprob"])
    return None


def score(args):
    reference = Reference(args.checkpoint, args.threads)
    prompts = {p["name"]: p["prompt_token_ids"] for p in json.loads(Path(args.prompts).read_text())["prompts"]}
    capture = json.loads(Path(args.capture).read_text())
    names = [name for name in capture if name in prompts]
    missing = [name for name in capture if name not in prompts]
    if missing:
        raise SystemExit(f"capture prompts {missing} are not in {args.prompts}")
    sequences = [prompts[name] + capture[name]["tokens"] for name in names]
    normed, spans = reference.forward(sequences)
    report = {"tie_margin": args.tie_margin, "prompts": {}}
    for name, (first, count) in zip(names, spans):
        prompt, served = prompts[name], capture[name]
        rows_normed = normed[first + len(prompt) - 1:first + count - 1]
        logits = reference.logits(rows_normed)
        logprobs = logits - logits.max(axis=-1, keepdims=True)
        logprobs -= np.log(np.exp(logprobs).sum(axis=-1, keepdims=True))
        rows, deltas, mismatches = [], [], []
        served_lp = served.get("logprobs") or [None] * len(served["tokens"])
        for index, token in enumerate(served["tokens"]):
            order = np.argsort(-logits[index], kind="stable")[:5]
            margin = float(logits[index, order[0]] - logits[index, order[1]])
            lp = served_logprob(served_lp[index])
            row = {"served": token, "served_logprob": lp, "reference_logprob": float(logprobs[index, token]),
                   "top": [[int(t), float(logprobs[index, t])] for t in order], "margin": margin}
            if lp is not None:
                deltas.append(abs(lp - row["reference_logprob"]))
            if int(order[0]) != token:
                mismatches.append({"index": index, "margin": margin, "reference_top1": int(order[0]), "served": token,
                                   "near_tie": margin < args.tie_margin})
            rows.append(row)
        summary = {"tokens": len(rows), "top1_agree": len(rows) - len(mismatches),
                   "mismatches_outside_ties": sum(1 for item in mismatches if not item["near_tie"]),
                   "mismatches": mismatches,
                   "logprob_delta_median": float(np.median(deltas)) if deltas else None,
                   "logprob_delta_max": float(np.max(deltas)) if deltas else None, "rows": rows}
        report["prompts"][name] = summary
        print(f"{name}: {summary['top1_agree']}/{summary['tokens']} top-1 agree, {summary['mismatches_outside_ties']} mismatches outside "
              f"ties (margin < {args.tie_margin}), served logprob delta median {summary['logprob_delta_median']} max {summary['logprob_delta_max']}",
              flush=True)
    Path(args.out).write_text(json.dumps(report))


def teacher(args):
    global REVERSED
    variants = args.variants.split(",")
    if not variants or any(label not in ("a", "b", "h") for label in variants) or len(set(variants)) != len(variants):
        raise SystemExit("--variants takes distinct a, b and h")
    reference = Reference(args.checkpoint, args.threads)
    prompts = json.loads(Path(args.prompts).read_text())["prompts"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for label in variants:
        REVERSED = label != "a"
        device.select(label)
        normed, spans = reference.forward([prompt["prompt_token_ids"] for prompt in prompts])
        for prompt, (first, count) in zip(prompts, spans):
            with open(out / f"{prompt['name']}.ref_{label}.logits", "wb") as handle:
                for row in range(0, count - 1, 128):
                    last = min(row + 128, count - 1)
                    handle.write(np.ascontiguousarray(reference.logits(normed[first + row:first + last]), dtype=np.float32).tobytes())
            print(f"{prompt['name']}: {count - 1} positions, variant {label}", flush=True)


def prompt_set(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gemma4_reference
    (gemma4_reference.gate_prompts if args.mode == "gate-prompts" else gemma4_reference.spec_prompts)(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("tokenize", "score", "teacher", "gate-prompts", "spec-prompts"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--variants", default="a,h")
    parser.add_argument("--capture")
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/mimo26/chat_template.json"))
    parser.add_argument("--answers-out")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--tie-margin", type=float, default=0.1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode in ("gate-prompts", "spec-prompts"):
        prompt_set(args)
    elif not args.prompts:
        parser.error(f"{args.mode} needs --prompts")
    elif args.mode == "tokenize":
        tokenize(args)
    elif args.mode == "teacher":
        teacher(args)
    else:
        if not args.capture:
            parser.error("score needs --capture")
        score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
