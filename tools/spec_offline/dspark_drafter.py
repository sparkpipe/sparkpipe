#!/usr/bin/env python3
"""Eager reference of DeepSeek-V4.1-Flash-1017's built-in DSpark drafter (checkpoint stages mtp.0-2).

Follows the publisher's inference/model.py and kernel.py operation for operation:
- fp8 and fp4 weights are dequantized to bf16, which is exact for their power-of-two block scales;
- activations get the same 32-wide fp8 quantize-dequantize with power-of-two scales before every fp8
  or fp4 matrix;
- matrices multiply in bf16 with fp32 accumulation.
The block-wise accumulation order of the publisher's kernels is not reproduced, so results match the
reference to rounding, not bit for bit. Only acceptance depends on that.

Round protocol (positions are absolute):
- ingest(context, positions, taps): taps rows are the mean of the four hyper-connection streams
  entering target layers 37, 38 and 39 (15360 bf16 per position). Each row becomes one 512-wide
  window key per stage, written to ring slot position % 128.
- draft(context, token, position, sampling): `token` is the pending token at `position`, whose taps do
  not exist yet; the ring must hold positions position-128 .. position-1. Returns the five drafted
  tokens for position+1 .. position+5 and their confidences (sigmoid of the confidence logit).
"""
import json
import math
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_drafter import coupled_choice  # noqa: E402

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
STORED = {"F8_E4M3": torch.uint8, "F8_E8M0": torch.uint8, "I8": torch.uint8, "BF16": torch.bfloat16, "F32": torch.float32}
FP8_MAX = 448.0
QUANT_BLOCK = 32
STAGES = 3


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.index = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

    def header(self, shard):
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        return self.headers[shard]

    def entry(self, name):
        shard = self.index[name]
        base, header = self.header(shard)
        return shard, base, header[name]

    def load(self, name, device):
        shard, base, info = self.entry(name)
        begin, end = info["data_offsets"]
        with open(self.directory / shard, "rb") as handle:
            handle.seek(base + begin)
            raw = bytearray(handle.read(end - begin))
        tensor = torch.frombuffer(raw, dtype=STORED[info["dtype"]]).reshape(info["shape"])
        return info["dtype"], tensor.to(device)


def e8m0(scale):
    return torch.exp2(scale.to(torch.float32) - 127.0)


def dequant_fp8(weight, scale):
    rows, columns = weight.shape
    values = weight.view(torch.float8_e4m3fn).to(torch.float32)
    factor = e8m0(scale).repeat_interleave(QUANT_BLOCK, 0)[:rows].repeat_interleave(QUANT_BLOCK, 1)[:, :columns]
    return (values * factor).to(torch.bfloat16)


def dequant_fp4(weight, scale, table):
    rows = weight.shape[0]
    low = (weight & 0x0F).long()
    high = (weight >> 4).long()
    values = torch.stack((table[low], table[high]), -1).reshape(rows, -1)
    return (values * e8m0(scale).repeat_interleave(QUANT_BLOCK, 1)).to(torch.bfloat16)


def pow2_ceil(values):
    bits = values.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return torch.exp2(exponent.to(torch.float32))


def act_qdq(x):
    shape = x.shape
    groups = x.to(torch.float32).reshape(*shape[:-1], shape[-1] // QUANT_BLOCK, QUANT_BLOCK)
    amax = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = pow2_ceil(amax * (1.0 / FP8_MAX))
    quantized = (groups / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    return (quantized * scale).reshape(shape).to(torch.bfloat16)


def qlinear(x, weight):
    return torch.matmul(act_qdq(x), weight.t())


def rms_norm(x, weight, eps):
    values = x.to(torch.float32)
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (weight.to(torch.float32) * values).to(x.dtype)


class Rope:
    def __init__(self, dimension, theta, device):
        self.inverse = 1.0 / (theta ** (torch.arange(0, dimension, 2, dtype=torch.float32, device=device) / dimension))

    def apply(self, x, positions, inverse=False):
        angle = torch.outer(positions.to(torch.float32), self.inverse)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        if inverse:
            sine = -sine
        shape = [positions.numel()] + [1] * (x.dim() - 2) + [self.inverse.numel()]
        cosine, sine = cosine.view(shape), sine.view(shape)
        pairs = x.to(torch.float32).unflatten(-1, (-1, 2))
        real, imaginary = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((real * cosine - imaginary * sine, real * sine + imaginary * cosine), -1).flatten(-2)
        return rotated.to(x.dtype)


class Context:
    def __init__(self, window, head_dim, device):
        self.keys = torch.zeros(STAGES, window, head_dim, dtype=torch.bfloat16, device=device)
        self.positions = torch.full((window,), -1, dtype=torch.int64, device=device)
        self.window = window

    def covered(self, position):
        first = max(position - self.window, 0)
        wanted = torch.arange(first, position, device=self.positions.device)
        return bool(torch.equal(self.positions[wanted % self.window], wanted))


class DsparkDrafter:
    def __init__(self, checkpoint_directory, device="cuda"):
        self.device = torch.device(device)
        directory = Path(checkpoint_directory)
        config = json.loads((directory / "inference" / "config.json").read_text())
        self.config = config
        self.dim = config["dim"]
        self.heads = config["n_heads"]
        self.head_dim = config["head_dim"]
        self.rope_dim = config["rope_head_dim"]
        self.window = config["window_size"]
        self.block = config["dspark_block_size"]
        self.noise = config["dspark_noise_token_id"]
        self.targets = tuple(config["dspark_target_layer_ids"])
        self.groups = config["o_groups"]
        self.o_rank = config["o_lora_rank"]
        self.eps = config["norm_eps"]
        self.hc = config["hc_mult"]
        self.sinkhorn = config["hc_sinkhorn_iters"]
        self.hc_eps = config["hc_eps"]
        self.topk = config["dspark_n_activated_experts"]
        self.experts = config["dspark_n_routed_experts"]
        self.route_scale = config["route_scale"]
        self.swiglu_limit = config["swiglu_limit"]
        self.softmax_scale = self.head_dim ** -0.5
        assert config["n_mtp_layers"] == STAGES and len(self.targets) == 3
        assert all(config["compress_ratios"][config["n_layers"] + stage] == 0 for stage in range(STAGES))
        self.table = torch.tensor(E2M1, dtype=torch.float32, device=self.device)
        self.rope = Rope(self.rope_dim, float(config["rope_theta"]), self.device)
        self.checkpoint = Checkpoint(directory)
        self.stages = [self.load_stage(stage) for stage in range(STAGES)]
        self.embed = self.plain("embed.weight")
        self.head = self.plain("head.weight").to(torch.float32)
        last = f"mtp.{STAGES - 1}."
        self.final_norm = self.plain(last + "norm.weight")
        self.markov_embed = self.plain(last + "markov_head.embed.weight")
        self.markov_head = self.plain(last + "markov_head.head.weight").to(torch.float32)
        self.confidence = self.plain(last + "confidence_head.proj.weight").to(torch.float32)
        self.main_proj = self.fp8("mtp.0.main_proj")
        self.main_norm = self.plain("mtp.0.main_norm.weight")
        self.noise_embed = self.embed[self.noise]
        self.layer_ids = self.targets
        self.tap_names = tuple(f"o{layer - 1}" for layer in self.targets)
        self.max_drafts = self.block

    def plain(self, name):
        dtype, tensor = self.checkpoint.load(name, self.device)
        assert dtype in ("BF16", "F32"), (name, dtype)
        return tensor

    def fp8(self, prefix):
        dtype, weight = self.checkpoint.load(prefix + ".weight", self.device)
        scale_dtype, scale = self.checkpoint.load(prefix + ".scale", self.device)
        assert dtype == "F8_E4M3" and scale_dtype == "F8_E8M0", (prefix, dtype, scale_dtype)
        return dequant_fp8(weight, scale)

    def load_stage(self, stage):
        p = f"mtp.{stage}."
        s = {
            "sink": self.plain(p + "attn.attn_sink").to(torch.float32),
            "q_norm": self.plain(p + "attn.q_norm.weight"),
            "kv_norm": self.plain(p + "attn.kv_norm.weight"),
            "wq_a": self.fp8(p + "attn.wq_a"),
            "wq_b": self.fp8(p + "attn.wq_b"),
            "wkv": self.fp8(p + "attn.wkv"),
            "wo_a": self.fp8(p + "attn.wo_a").view(self.groups, self.o_rank, -1),
            "wo_b": self.fp8(p + "attn.wo_b"),
            "attn_norm": self.plain(p + "attn_norm.weight"),
            "ffn_norm": self.plain(p + "ffn_norm.weight"),
            "gate": self.plain(p + "ffn.gate.weight").to(torch.float32),
            "gate_bias": self.plain(p + "ffn.gate.bias").to(torch.float32),
            "shared": [self.fp8(p + f"ffn.shared_experts.{w}") for w in ("w1", "w2", "w3")],
        }
        for kind in ("attn", "ffn"):
            s[f"hc_{kind}_fn"] = self.plain(p + f"hc_{kind}_fn").to(torch.float32)
            s[f"hc_{kind}_base"] = self.plain(p + f"hc_{kind}_base").to(torch.float32)
            s[f"hc_{kind}_scale"] = self.plain(p + f"hc_{kind}_scale").to(torch.float32)
        packed = []
        for expert in range(self.experts):
            weights = []
            for w in ("w1", "w2", "w3"):
                dtype, weight = self.checkpoint.load(p + f"ffn.experts.{expert}.{w}.weight", self.device)
                scale_dtype, scale = self.checkpoint.load(p + f"ffn.experts.{expert}.{w}.scale", self.device)
                assert dtype == "I8" and scale_dtype == "F8_E8M0", (p, expert, w, dtype, scale_dtype)
                weights.append((weight, scale))
            packed.append(weights)
        s["experts"] = packed
        return s

    def context(self):
        return Context(self.window, self.head_dim, self.device)

    @torch.inference_mode()
    def ingest(self, context, positions, taps):
        positions = torch.as_tensor(positions, dtype=torch.int64, device=self.device)
        taps = taps.to(device=self.device, dtype=torch.bfloat16)
        main = rms_norm(qlinear(taps, self.main_proj), self.main_norm, self.eps)
        for stage, s in enumerate(self.stages):
            keys = rms_norm(qlinear(main, s["wkv"]), s["kv_norm"], self.eps)
            keys = torch.cat((keys[:, :-self.rope_dim], self.rope.apply(keys[:, -self.rope_dim:], positions)), -1)
            context.keys[stage, positions % self.window] = act_qdq(keys)
        context.positions[positions % self.window] = positions

    def hc_mixes(self, h, fn, scale, base):
        flat = h.flatten(1).to(torch.float32)
        mixes = F.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        n = self.hc
        pre = torch.sigmoid(mixes[:, :n] * scale[0] + base[:n]) + self.hc_eps
        post = 2.0 * torch.sigmoid(mixes[:, n:2 * n] * scale[1] + base[n:2 * n])
        comb = (mixes[:, 2 * n:] * scale[2] + base[2 * n:]).view(-1, n, n)
        comb = torch.softmax(comb, -1) + self.hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        return pre, post, comb

    @staticmethod
    def hc_pre(h, mix):
        return torch.sum(mix.unsqueeze(-1) * h.to(torch.float32), 1).to(h.dtype)

    @staticmethod
    def hc_post(y, residual, post, comb):
        mixed = post.unsqueeze(-1) * y.to(torch.float32).unsqueeze(1)
        mixed = mixed + torch.sum(comb.unsqueeze(-1) * residual.to(torch.float32).unsqueeze(2), 1)
        return mixed.to(y.dtype)

    def attention(self, s, x, positions, ring, valid):
        rows = x.shape[0]
        qr = rms_norm(qlinear(x, s["wq_a"]), s["q_norm"], self.eps)
        q = qlinear(qr, s["wq_b"]).view(rows, self.heads, self.head_dim)
        q = torch.cat((q[..., :-self.rope_dim], self.rope.apply(q[..., -self.rope_dim:], positions)), -1)
        keys = rms_norm(qlinear(x, s["wkv"]), s["kv_norm"], self.eps)
        keys = torch.cat((keys[:, :-self.rope_dim], self.rope.apply(keys[:, -self.rope_dim:], positions)), -1)
        keys = torch.cat((ring[valid], act_qdq(keys)), 0)
        scores = torch.einsum("rhd,kd->rhk", q.to(torch.float32), keys.to(torch.float32)) * self.softmax_scale
        top = torch.maximum(scores.amax(-1, keepdim=True), torch.full_like(scores[..., :1], -1e30))
        weights = torch.exp(scores - top)
        total = weights.sum(-1, keepdim=True) + torch.exp(s["sink"].view(1, -1, 1) - top)
        output = torch.einsum("rhk,kd->rhd", weights.to(torch.bfloat16).to(torch.float32), keys.to(torch.float32)) / total
        output = output.to(torch.bfloat16)
        output = torch.cat((output[..., :-self.rope_dim], self.rope.apply(output[..., -self.rope_dim:], positions, True)), -1)
        grouped = act_qdq(output.view(rows, self.groups, -1))
        projected = torch.einsum("rgd,gkd->rgk", grouped.to(torch.float32), s["wo_a"].to(torch.float32)).to(torch.bfloat16)
        return qlinear(projected.flatten(1), s["wo_b"])

    def expert(self, x, weights, routing):
        gate = qlinear(x, weights[0]).to(torch.float32)
        up = qlinear(x, weights[2]).to(torch.float32)
        if self.swiglu_limit > 0:
            up = torch.clamp(up, -self.swiglu_limit, self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        hidden = F.silu(gate) * up
        if routing is not None:
            hidden = routing * hidden
        return qlinear(hidden.to(x.dtype), weights[1])

    def moe(self, s, x):
        scores = F.softplus(F.linear(x.to(torch.float32), s["gate"])).sqrt()
        chosen = (scores + s["gate_bias"]).topk(self.topk, -1)[1]
        routing = scores.gather(1, chosen)
        routing = routing / (routing.sum(-1, keepdim=True) + 1e-20) * self.route_scale
        y = torch.zeros_like(x, dtype=torch.float32)
        for expert in sorted(set(chosen.flatten().tolist())):
            rows, slot = torch.where(chosen == expert)
            packed = s["experts"][expert]
            weights = [dequant_fp4(w, scale, self.table) for w, scale in packed]
            y[rows] += self.expert(x[rows], weights, routing[rows, slot, None]).to(torch.float32)
        y += self.expert(x, s["shared"], None).to(torch.float32)
        return y.to(x.dtype)

    @torch.inference_mode()
    def draft(self, context, token, position, sampling=None):
        rows = self.block
        positions = torch.arange(position, position + rows, device=self.device)
        ids = torch.full((rows,), self.noise, dtype=torch.int64, device=self.device)
        ids[0] = int(token)
        h = self.embed[ids].unsqueeze(1).repeat(1, self.hc, 1)
        pre = torch.zeros(rows, self.hc, dtype=torch.float32, device=self.device)
        pre[:, 0] = 1.0
        valid = context.positions >= 0
        valid &= context.positions < position
        valid &= context.positions >= position - self.window
        for stage, s in enumerate(self.stages):
            residual = h
            attn_pre, attn_post, attn_comb = self.hc_mixes(h, s["hc_attn_fn"], s["hc_attn_scale"], s["hc_attn_base"])
            x = rms_norm(self.hc_pre(h, pre), s["attn_norm"], self.eps)
            x = self.attention(s, x, positions, context.keys[stage], valid)
            h = self.hc_post(x, residual, attn_post, attn_comb)
            residual = h
            ffn_pre, ffn_post, ffn_comb = self.hc_mixes(h, s["hc_ffn_fn"], s["hc_ffn_scale"], s["hc_ffn_base"])
            x = rms_norm(self.hc_pre(h, attn_pre), s["ffn_norm"], self.eps)
            x = self.moe(s, x)
            h = self.hc_post(x, residual, ffn_post, ffn_comb)
            pre = ffn_pre
        hidden = self.hc_pre(h, pre)
        logits = F.linear(rms_norm(hidden, self.final_norm, self.eps).to(torch.float32), self.head)
        tokens = [int(token)]
        embeddings = []
        for slot in range(rows):
            previous = self.markov_embed[tokens[-1]]
            embeddings.append(previous)
            row = logits[slot] + F.linear(previous.to(torch.float32), self.markov_head)
            if sampling is None:
                tokens.append(int(torch.argmax(row)))
            else:
                tokens.append(int(coupled_choice(row, sampling, position + slot)))
        features = torch.cat((hidden, torch.stack(embeddings)), -1).to(torch.float32)
        confidence = torch.sigmoid(F.linear(features, self.confidence).squeeze(-1))
        return tokens[1:], confidence.tolist()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    drafter = DsparkDrafter(arguments.checkpoint, arguments.device)
    context = drafter.context()
    generator = torch.Generator(device="cpu").manual_seed(0)
    taps = (torch.randn(200, 3 * drafter.dim, generator=generator) * 0.5).to(torch.bfloat16)
    drafter.ingest(context, list(range(200)), taps)
    single = drafter.context()
    for position in range(200):
        drafter.ingest(single, [position], taps[position:position + 1])
    difference = (context.keys.to(torch.float32) - single.keys.to(torch.float32)).abs().max().item()
    scale = single.keys.to(torch.float32).abs().max().item()
    print("DSPARK-INGEST batched vs single-row keys: max abs difference %.3g of max %.3g, positions equal %s, covered(200) %s"
          % (difference, scale, bool(torch.equal(context.positions, single.positions)), context.covered(200)))
    tokens, confidence = drafter.draft(context, 100, 200)
    print("DSPARK-DRAFT tokens", tokens, "confidence", [round(c, 4) for c in confidence])
    print("DSPARK-MEMORY allocated GiB %.2f" % (torch.cuda.memory_allocated(drafter.device) / 2**30))


if __name__ == "__main__":
    main()
