#!/usr/bin/env python3
"""Check one Qwen3.8 layer of a SPARK_QWEN38_*_LAYER_DUMP against numpy.

The dump (model-families/qwen38/source/qwen38_hybrid_stage.cuh) holds, per
step, the embedding rows and, after every layer, the residual with the
attention added and the FFN output not yet added. For a prefill that starts
at position 0 this script takes layer L's input from the dump (layer L-1's
residual plus FFN output), recomputes layer L in float32 numpy from the
checkpoint (GDN or gated attention, then the dense SwiGLU FFN or the MoE
with the routed NVFP4 experts and the gated shared expert) and reports the
relative error of the attention residual and of the FFN output separately.

  qwen38_layer_check.py --checkpoint DIR --dump FILE --layer L [--step N]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.map = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.config = json.loads((self.directory / "config.json").read_text())
        self.config = self.config.get("text_config", self.config)
        self.headers = {}
        self.prefix = "model.language_model." if any(k.startswith("model.language_model.") for k in self.map) else "model."

    def raw(self, name):
        shard = self.map[name]
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        base, header = self.headers[shard]
        meta = header[name]
        dtype = {"BF16": np.uint16, "F32": np.float32, "U8": np.uint8, "F8_E4M3": np.uint8}[meta["dtype"]]
        begin, end = meta["data_offsets"]
        return meta["dtype"], np.memmap(self.directory / shard, dtype=dtype, mode="r", offset=base + begin,
                                        shape=tuple(meta["shape"]))

    def f32(self, name):
        dtype, data = self.raw(name)
        if dtype == "BF16":
            return (np.asarray(data).astype(np.uint32) << 16).view(np.float32)
        if dtype == "F32":
            return np.asarray(data, dtype=np.float32)
        raise SystemExit(f"{name} is {dtype}, not a float tensor")

    def nvfp4(self, name):
        e2m1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
        _, packed = self.raw(name)
        _, plane = self.raw(name[:-len(".weight")] + ".weight_scale")
        global_scale = float(self.f32(name[:-len(".weight")] + ".weight_scale_2").reshape(-1)[0])
        packed = np.asarray(packed)
        weight = np.empty((packed.shape[0], packed.shape[1] * 2), np.float32)
        weight[:, 0::2] = e2m1[packed & 15]
        weight[:, 1::2] = e2m1[packed >> 4]
        return weight * np.repeat(e4m3(np.asarray(plane)), 16, axis=1) * global_scale


def e4m3(values):
    values = values.astype(np.int32)
    sign = np.where(values & 0x80, -1.0, 1.0)
    exponent = (values >> 3) & 15
    mantissa = values & 7
    return (sign * np.where(exponent == 0, mantissa / 8.0 * 2.0 ** -6, (1 + mantissa / 8.0) * 2.0 ** (exponent - 7))).astype(np.float32)


def bf16_round(values):
    bits = values.astype(np.float32).view(np.uint32)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.view(np.float32)


def rms(values, weight, eps, offset):
    scale = 1.0 / np.sqrt((values * values).mean(-1, keepdims=True) + eps)
    return values * scale * ((1.0 + weight) if offset else weight)


def silu(values):
    return values / (1.0 + np.exp(-values))


def softplus(values):
    return np.where(values > 20.0, values, np.log1p(np.exp(np.minimum(values, 20.0))))


def gdn(ckpt, layer, x):
    c = ckpt.config
    p = f"{ckpt.prefix}layers.{layer}.linear_attn."
    kh, vh, kd, vd = c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"]
    rows = x.shape[0]
    qkv = x @ ckpt.f32(p + "in_proj_qkv.weight").T
    z = x @ ckpt.f32(p + "in_proj_z.weight").T
    b = x @ ckpt.f32(p + "in_proj_b.weight").T
    a = x @ ckpt.f32(p + "in_proj_a.weight").T
    conv = ckpt.f32(p + "conv1d.weight").reshape(qkv.shape[1], -1)
    width = conv.shape[1]
    padded = np.concatenate([np.zeros((width - 1, qkv.shape[1]), np.float32), qkv])
    mixed = silu(np.stack([(padded[t:t + width] * conv.T).sum(0) for t in range(rows)]))
    q = mixed[:, :kh * kd].reshape(rows, kh, kd)
    k = mixed[:, kh * kd:2 * kh * kd].reshape(rows, kh, kd)
    v = mixed[:, 2 * kh * kd:].reshape(rows, vh, vd)
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / np.sqrt(kd)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    q = np.repeat(q, vh // kh, axis=1)
    k = np.repeat(k, vh // kh, axis=1)
    beta = 1.0 / (1.0 + np.exp(-b))
    g = -np.exp(ckpt.f32(p + "A_log")) * softplus(a + ckpt.f32(p + "dt_bias"))
    state = np.zeros((vh, kd, vd), np.float32)
    out = np.zeros((rows, vh, vd), np.float32)
    for t in range(rows):
        state *= np.exp(g[t])[:, None, None]
        memory = np.einsum("hk,hkv->hv", k[t], state)
        delta = (v[t] - memory) * beta[t][:, None]
        state += np.einsum("hk,hv->hkv", k[t], delta)
        out[t] = np.einsum("hk,hkv->hv", q[t], state)
    normed = rms(out, ckpt.f32(p + "norm.weight"), c["rms_norm_eps"], False) * silu(z.reshape(rows, vh, vd))
    return normed.reshape(rows, -1) @ ckpt.f32(p + "out_proj.weight").T


def attention(ckpt, layer, x, positions):
    c = ckpt.config
    p = f"{ckpt.prefix}layers.{layer}.self_attn."
    heads, kv_heads, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    rope = c.get("rope_parameters") or {}
    theta = rope.get("rope_theta", c.get("rope_theta"))
    rot = int(dim * rope.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0)))
    rows = x.shape[0]
    qg = (x @ ckpt.f32(p + "q_proj.weight").T).reshape(rows, heads, 2 * dim)
    q, gate = qg[..., :dim], qg[..., dim:]
    k = (x @ ckpt.f32(p + "k_proj.weight").T).reshape(rows, kv_heads, dim)
    v = (x @ ckpt.f32(p + "v_proj.weight").T).reshape(rows, kv_heads, dim)
    q = rms(q, ckpt.f32(p + "q_norm.weight"), c["rms_norm_eps"], True)
    k = rms(k, ckpt.f32(p + "k_norm.weight"), c["rms_norm_eps"], True)
    inv = 1.0 / theta ** (np.arange(0, rot, 2, dtype=np.float32) / rot)
    angle = positions[:, None].astype(np.float32) * inv[None, :]
    cos, sin = np.concatenate([np.cos(angle)] * 2, -1)[:, None, :], np.concatenate([np.sin(angle)] * 2, -1)[:, None, :]

    def rotate(t):
        head, tail = t[..., :rot], t[..., rot:]
        half = rot // 2
        turned = np.concatenate([-head[..., half:], head[..., :half]], -1)
        return np.concatenate([head * cos + turned * sin, tail], -1)

    q, k = rotate(q), rotate(k)
    k = np.repeat(k, heads // kv_heads, axis=1)
    v = np.repeat(v, heads // kv_heads, axis=1)
    scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(dim)
    scores = np.where(np.tril(np.ones((rows, rows), bool))[None], scores, -np.inf)
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    scores /= scores.sum(-1, keepdims=True)
    out = np.einsum("hqk,khd->qhd", scores, v) * (1.0 / (1.0 + np.exp(-gate)))
    return out.reshape(rows, -1) @ ckpt.f32(p + "o_proj.weight").T


def ffn(ckpt, layer, x):
    c = ckpt.config
    p = f"{ckpt.prefix}layers.{layer}.mlp."
    if "num_experts" not in c or not c["num_experts"]:
        gate = x @ ckpt.f32(p + "gate_proj.weight").T
        up = x @ ckpt.f32(p + "up_proj.weight").T
        return (silu(gate) * up) @ ckpt.f32(p + "down_proj.weight").T
    logits = bf16_round(x @ ckpt.f32(p + "gate.weight").T)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    top = np.argsort(-probs, axis=-1)[:, :c["num_experts_per_tok"]]
    weights = np.take_along_axis(probs, top, -1)
    weights /= weights.sum(-1, keepdims=True)
    out = np.zeros_like(x)
    for expert in np.unique(top):
        rows_for = np.where((top == expert).any(-1))[0]
        w = weights[rows_for, np.argmax(top[rows_for] == expert, axis=-1)]
        e = f"{p}experts.{expert}."
        gate = x[rows_for] @ ckpt.nvfp4(e + "gate_proj.weight").T
        up = x[rows_for] @ ckpt.nvfp4(e + "up_proj.weight").T
        out[rows_for] += ((silu(gate) * up) @ ckpt.nvfp4(e + "down_proj.weight").T) * w[:, None]
    shared = (silu(x @ ckpt.f32(p + "shared_expert.gate_proj.weight").T) * (x @ ckpt.f32(p + "shared_expert.up_proj.weight").T)) \
        @ ckpt.f32(p + "shared_expert.down_proj.weight").T
    gate = 1.0 / (1.0 + np.exp(-(x @ ckpt.f32(p + "shared_expert_gate.weight").T)))
    return out + shared * gate


def read_dump(path, hidden):
    data = open(path, "rb").read()
    steps, offset = [], 0
    while offset < len(data):
        magic, layer, rows = struct.unpack_from("<3I", data, offset)
        if magic != 0x444C5751:
            raise SystemExit(f"bad dump record at {offset}")
        offset += 12
        positions = np.frombuffer(data, "<u4", rows, offset).copy()
        offset += 4 * rows
        values = (np.frombuffer(data, "<u2", 2 * rows * hidden, offset).astype(np.uint32) << 16).view(np.float32).reshape(2, rows, hidden)
        offset += 4 * rows * hidden
        if layer == 0xFFFFFFFF:
            steps.append({"positions": positions, "embed": values[0], "layers": {}})
        else:
            steps[-1]["layers"][layer] = (values[0].copy(), values[1].copy())
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    ckpt = Checkpoint(args.checkpoint)
    c = ckpt.config
    step = read_dump(args.dump, c["hidden_size"])[args.step]
    order = np.argsort(step["positions"])
    positions = step["positions"][order]
    if positions[0] != 0 or np.any(np.diff(positions) != 1):
        raise SystemExit("the step must be one prefill from position 0")
    layer = args.layer
    x = step["embed"] if layer == 0 else sum(step["layers"][layer - 1])
    x = x[order]
    period = c.get("full_attention_interval", 4)
    norm = f"{ckpt.prefix}layers.{layer}."
    eps = c["rms_norm_eps"]
    h = rms(x, ckpt.f32(norm + "input_layernorm.weight"), eps, True)
    mixer = attention(ckpt, layer, h, positions) if layer % period == period - 1 else gdn(ckpt, layer, h)
    residual = x + mixer
    ours_residual, ours_ffn = (values[order] for values in step["layers"][layer])
    ref_ffn = ffn(ckpt, layer, rms(ours_residual, ckpt.f32(norm + "post_attention_layernorm.weight"), eps, True))

    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

    kind = "attention" if layer % period == period - 1 else "gdn"
    print(json.dumps({"layer": layer, "kind": kind, "rows": int(len(positions)),
                      "mixer_residual_rel": round(rel(ours_residual, residual), 5),
                      "mixer_delta_rel": round(rel(ours_residual - x, mixer), 5),
                      "ffn_rel": round(rel(ours_ffn, ref_ffn), 5),
                      "last_row_ffn_rel": round(rel(ours_ffn[-1], ref_ffn[-1]), 5)}))


if __name__ == "__main__":
    sys.exit(main())
