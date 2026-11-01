#!/usr/bin/env python3
"""Teacher-forced numpy reference for GLM 5.3 (Full) and A.X K2 (the FP8 releases) for the Metal gate.

  gate-prompts  the Metal gate prompts (tools/gemma4_reference.py's set) in
             the model's chat turn (model-families/glm52 or axk2
             chat_template.json, thinking off) -> --out prompts.json
  teacher    FP32 logits of every prompt position but the last:
             --prompts prompts.json -> --out DIR/<name>.ref_<v>.logits for
             --variants a,b, through tools/ling3_reference.py's teacher (b
             sums every product in 1,024-wide chunks last first, reverses the
             other reductions and combines the experts in descending order)

The rounding points follow the CUDA stage (model-families/glm52/source/
glm52_stage.cuh): FP8 projections are e4m3 x the FP32 block scale taken in
FP32 (the skinny kernels' arithmetic), kv_b is bf16(e4m3 x scale) as the CUDA
pack decodes it, and
  norms      the residual sum v = hidden + residual in FP32, residual =
             bf16(v), normed = bf16(v r w) with the BF16 gain; the q_a and
             kv_a norms bf16(x r w)
  MLA        q_a -> norm -> q_b (192 nope + 64 rope per head), kv_a -> the
             512 latent norm plus the 64-wide rope key; interleaved rope at
             theta 8e6 on the pairs (2i, 2i+1), each rotated value rounded
             once; the query absorbed per head through kv_b's key rows
             (rounded); softmax((q_l . k_l + q_r . k_r) / 16) in FP32 over
             every earlier position (below 2,048 keys the indexer keeps them
             all); the latent output rounded; kv_b's value rows; o_proj
  MLP        bf16(silu(g) u) rounded once; layers 0-2 dense, the rest the
             shared expert plus top 8 of 256 by sigmoid(z) +
             e_score_correction_bias, weights renormalised and scaled by
             2.5, each expert's output rounded and combined in pick order in
             FP32, then bf16(bf16(routed) + shared)
A.X K2 (glm52_stage.cuh's switches) adds: the rank-16 norm gate after the
input norm and the routed layers' post norm, bf16(x sigmoid(bf16(up .
bf16(silu(bf16(down . x)))))) with the sigmoid unrounded; q_b reading the
normed and the raw query side by side; a 128-wide gate per head after the
rope part of the query, bf16(v sigmoid(g)) on the head's values; YaRN rope
(theta 1e6, factor 2, beta 32 / 1, 131,072 original positions, the table of
SparkRopeYarnRampTable); scale 0.0825202391; layer 0 dense; routing over 8
groups of which the 4 with the best two-expert sums stay open.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ling3_reference as base  # noqa: E402
from glm52_resident_stagepack import Fp8SourceReader, fp8_e4m3_lut  # noqa: E402

MODELS = {
    "glm53full": {"theta": 8000000.0, "scale": 0.0625, "query_gate": 0, "gate_rank": 0, "groups": 1, "topk_groups": 1, "yarn": None,
                  "template": "model-families/glm52/chat_template.json", "head_keys": ("prefix", "user"), "tail_keys": ("turn_suffix", "generation")},
    "axk2": {"theta": 1000000.0, "scale": 0.0825202391, "query_gate": 128, "gate_rank": 16, "groups": 8, "topk_groups": 4, "yarn": (2.0, 32.0, 1.0, 131072),
             "template": "model-families/axk2/chat_template.json", "head_keys": ("prefix", "plain_prefix", "user"),
             "tail_keys": ("user_suffix", "turn_suffix", "generation")},
}


def yarn_table(rope, theta, factor, beta_fast, beta_slow, original):
    low = np.floor(rope * np.log(original / (beta_fast * 2.0 * np.pi)) / (2.0 * np.log(theta)))
    high = np.ceil(rope * np.log(original / (beta_slow * 2.0 * np.pi)) / (2.0 * np.log(theta)))
    low, high = max(low, 0.0), min(high, float(rope - 1))
    if low == high:
        high += 0.001
    table = np.empty(rope // 2, dtype=np.float32)
    for index in range(rope // 2):
        frequency = np.power(np.float32(theta), np.float32(2 * index) / np.float32(rope), dtype=np.float32)
        extrapolation, interpolation = np.float32(1.0) / frequency, np.float32(1.0) / (np.float32(factor) * frequency)
        ramp = min(max((np.float32(index) - np.float32(low)) / (np.float32(high) - np.float32(low)), np.float32(0.0)), np.float32(1.0))
        keep = np.float32(1.0) - np.float32(ramp)
        table[index] = interpolation * (np.float32(1.0) - keep) + extrapolation * keep
    return table


def widen(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


class Reference:
    def __init__(self, directory, model):
        self.model = MODELS[model]
        self.source = Fp8SourceReader(Path(directory), cache_byte_cap=2 * 1024 ** 3)
        config = self.source.config.get("text_config", self.source.config)
        self.hidden = config["hidden_size"]
        self.layers = config["num_hidden_layers"]
        self.vocab = config["vocab_size"]
        self.epsilon = np.float32(config["rms_norm_eps"])
        self.heads = config["num_attention_heads"]
        self.latent = config["kv_lora_rank"]
        self.nope = config["qk_nope_head_dim"]
        self.rope_dim = config["qk_rope_head_dim"]
        self.value_dim = config["v_head_dim"]
        self.experts = config["n_routed_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.route_scale = np.float32(config["routed_scaling_factor"])
        self.dense_layers = config["first_k_dense_replace"]
        self.index_topk = config.get("index_topk", 2048)
        if config.get("n_group", 1) != self.model["groups"] or config.get("topk_group", 1) != self.model["topk_groups"] or \
                not config.get("norm_topk_prob", True) or config.get("n_shared_experts") != 1 or not config.get("rope_interleave", True):
            raise ValueError("the checkpoint's routing groups, top-k normalisation, shared expert or rope layout differ from the model this reference implements")
        self.lut = fp8_e4m3_lut()
        if self.model["yarn"] is not None:
            self.inv_freq = yarn_table(self.rope_dim, self.model["theta"], *self.model["yarn"])
        else:
            pairs = np.arange(self.rope_dim // 2, dtype=np.float32)
            self.inv_freq = np.power(np.float32(self.model["theta"]), -np.float32(2.0) * pairs / np.float32(self.rope_dim)).astype(np.float32)
        self.scale = np.float32(self.model["scale"])
        self.query_gate = self.model["query_gate"]

    def weight(self, name):
        dtype, shape, _ = self.source.meta(name)
        if dtype == "BF16":
            return widen(self.source.raw(name).view(np.uint16).reshape(shape))
        if dtype != "F8_E4M3":
            raise ValueError(f"{name}: unexpected dtype {dtype}")
        codes = np.asarray(self.source.raw(name)).reshape(shape)
        scale_shape = self.source.meta(name + "_scale_inv")[1]
        scale = np.asarray(self.source.raw(name + "_scale_inv")).view(np.float32).reshape(scale_shape)
        expanded = np.repeat(np.repeat(scale, 128, axis=0)[:shape[0]], 128, axis=1)[:, :shape[1]]
        return self.lut[codes] * expanded

    def gain(self, name):
        return widen(self.source.spine_bf16(name)).reshape(-1)

    def linear(self, values, weight):
        return base.bf16(base.matmul(values, weight.T))

    def scale_of(self, values):
        return np.power(base.reduce_sum(values * values) / np.float32(values.shape[-1]) + self.epsilon, np.float32(-0.5))

    def norm_once(self, values, gain):
        return base.bf16(values * self.scale_of(values) * gain)

    def residual_norm(self, residual, addend, gain):
        total = residual + addend
        return base.bf16(total), base.bf16(total * self.scale_of(total) * gain)

    def norm_gate(self, normed, prefix):
        if self.model["gate_rank"] == 0:
            return normed
        bottleneck = base.bf16(base.silu(self.linear(normed, self.weight(prefix + ".W_down.weight"))))
        return base.bf16(normed * base.sigmoid(self.linear(bottleneck, self.weight(prefix + ".W_up.weight"))))

    def gain_of(self, prefix, gated):
        return self.gain(prefix + (".norm.weight" if gated else ".weight"))

    def rope(self, values, positions):
        angles = positions[:, None].astype(np.float32) * self.inv_freq[None, :]
        cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
        low, high = values[..., 0::2], values[..., 1::2]
        rotated = np.empty_like(values)
        rotated[..., 0::2] = base.bf16(low * cos - high * sin)
        rotated[..., 1::2] = base.bf16(low * sin + high * cos)
        return rotated

    def attention_weights(self, a):
        return {"q_a": self.weight(a + "q_a_proj.weight"), "q_norm": self.gain(a + "q_a_layernorm.weight"), "q_b": self.weight(a + "q_b_proj.weight"),
                "kv_a": self.weight(a + "kv_a_proj_with_mqa.weight"), "kv_norm": self.gain(a + "kv_a_layernorm.weight"),
                "kv_b": widen(self.source.spine_bf16(a + "kv_b_proj.weight")).reshape(self.heads, self.nope + self.value_dim, self.latent),
                "o": self.weight(a + "o_proj.weight")}

    def mla(self, normed, positions, weights):
        rows, heads = normed.shape[0], self.heads
        raw = self.linear(normed, weights["q_a"])
        compressed = self.norm_once(raw, weights["q_norm"])
        if self.query_gate:
            compressed = np.concatenate([compressed, raw], axis=-1)
        q = self.linear(compressed, weights["q_b"]).reshape(rows, heads, self.nope + self.rope_dim + self.query_gate)
        kv = self.linear(normed, weights["kv_a"])
        latent = self.norm_once(kv[:, :self.latent], weights["kv_norm"])
        key_rope = self.rope(kv[:, None, self.latent:], positions)[:, 0, :]
        query_rope = self.rope(q[..., self.nope:self.nope + self.rope_dim], positions)
        causal = positions[None, :] <= positions[:, None]
        output = np.empty((rows, heads, self.value_dim), dtype=np.float32)
        for head in range(heads):
            key_up, value_up = weights["kv_b"][head, :self.nope], weights["kv_b"][head, self.nope:]
            query = base.bf16(base.matmul(q[:, head, :self.nope], key_up))
            scores = (base.matmul(query, latent.T) + base.matmul(query_rope[:, head, :], key_rope.T)) * self.scale
            scores = np.where(causal, scores, -np.inf)
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities = probabilities / base.reduce_sum(probabilities)
            attended = base.bf16(base.matmul(probabilities, latent))
            output[:, head, :] = base.bf16(base.matmul(attended, value_up.T))
            if self.query_gate:
                output[:, head, :] = base.bf16(output[:, head, :] * base.sigmoid(q[:, head, self.nope + self.rope_dim:]))
        return self.linear(output.reshape(rows, heads * self.value_dim), weights["o"])

    def glu(self, values, gate_weight, up_weight, down_weight):
        gate = self.linear(values, gate_weight)
        up = self.linear(values, up_weight)
        return self.linear(base.bf16(base.silu(gate) * up), down_weight)

    def mlp_weights(self, prefix):
        return tuple(self.weight(prefix + f"{key}_proj.weight") for key in ("gate", "up", "down"))

    def route(self, m, normed):
        logits = base.matmul(normed, self.weight(m + "gate.weight").T)
        scores = base.sigmoid(logits)
        bias = np.asarray(self.source.raw(m + "gate.e_score_correction_bias")).view(np.float32)
        selection = scores + bias[None, :]
        groups = self.model["groups"]
        if groups > 1:
            grouped = selection.reshape(selection.shape[0], groups, -1)
            group_score = np.sort(grouped, axis=-1)[..., -2:].sum(axis=-1)
            kept = np.argsort(-group_score, axis=-1, kind="stable")[:, :self.model["topk_groups"]]
            mask = np.zeros(group_score.shape, dtype=bool)
            np.put_along_axis(mask, kept, True, axis=-1)
            selection = np.where(np.repeat(mask, self.experts // groups, axis=-1), selection, -np.inf)
        chosen = np.argsort(-selection, axis=-1, kind="stable")[:, :self.top_k]
        picked = np.take_along_axis(scores, chosen, axis=-1)
        return chosen, (picked / (base.reduce_sum(picked) + np.float32(1e-20)) * self.route_scale).astype(np.float32)

    def moe(self, layer, normed):
        m = f"model.layers.{layer}.mlp."
        chosen, weights = self.route(m, normed)
        outputs = np.zeros((normed.shape[0], self.top_k, self.hidden), dtype=np.float32)
        experts = [int(expert) for expert in (np.unique(chosen)[::-1] if base.REVERSED else np.unique(chosen))]
        window = 6
        with ThreadPoolExecutor(6) as pool:
            pending = {expert: pool.submit(self.mlp_weights, m + f"experts.{expert}.") for expert in experts[:window]}
            for index, expert in enumerate(experts):
                if index + window < len(experts):
                    following = experts[index + window]
                    pending[following] = pool.submit(self.mlp_weights, m + f"experts.{following}.")
                rows, slots = np.nonzero(chosen == expert)
                outputs[rows, slots] = self.glu(normed[rows], *pending.pop(expert).result())
        combined = np.zeros((normed.shape[0], self.hidden), dtype=np.float32)
        for slot in (range(self.top_k - 1, -1, -1) if base.REVERSED else range(self.top_k)):
            combined += outputs[:, slot] * weights[:, slot][:, None]
        shared = self.glu(normed, *self.mlp_weights(m + "shared_experts."))
        return base.bf16(base.bf16(combined) + shared), chosen

    def forward(self, segments, dump=None):
        tokens = np.concatenate([np.asarray(s, dtype=np.int64) for s in segments])
        positions = np.concatenate([np.arange(len(s), dtype=np.int64) for s in segments])
        bounds = np.cumsum([0] + [len(s) for s in segments])
        if positions.max() >= self.index_topk:
            raise ValueError(f"a prompt reaches position {positions.max()}: beyond {self.index_topk} keys the indexer drops some and this reference is dense")
        embed = self.source.raw("model.embed_tokens.weight").view(np.uint16).reshape(self.vocab, self.hidden)
        residual = np.zeros((len(tokens), self.hidden), dtype=np.float32)
        branch = widen(embed[tokens])
        if dump is not None:
            dump["embed"] = branch.copy()
        routes = {}
        for layer in range(self.layers):
            began = time.time()
            p = f"model.layers.{layer}."
            gated = self.model["gate_rank"] != 0
            routed = layer >= self.dense_layers
            residual, normed = self.residual_norm(residual, branch, self.gain_of(p + "input_layernorm", gated))
            normed = self.norm_gate(normed, p + "input_layernorm")
            weights = self.attention_weights(p + "self_attn.")
            attention = np.empty_like(normed)
            for first, last in zip(bounds[:-1], bounds[1:]):
                attention[first:last] = self.mla(normed[first:last], positions[first:last], weights)
            del weights
            residual, normed = self.residual_norm(residual, attention, self.gain_of(p + "post_attention_layernorm", gated and routed))
            if gated and routed:
                normed = self.norm_gate(normed, p + "post_attention_layernorm")
            if not routed:
                branch = self.glu(normed, *self.mlp_weights(p + "mlp."))
            else:
                branch, routes[layer] = self.moe(layer, normed)
            if dump is not None:
                dump[f"layer{layer:02d}"] = base.bf16(residual + branch)
            print(f"layer {layer} done in {time.time() - began:.1f}s", file=sys.stderr, flush=True)
        _, normed = self.residual_norm(residual, branch, self.gain("model.norm.weight"))
        return normed, routes

    def logits(self, normed, chunk=16384):
        head = self.source.raw("lm_head.weight").view(np.uint16).reshape(self.vocab, self.hidden)
        parts = []
        for first in range(0, self.vocab, chunk):
            parts.append(base.matmul(normed, widen(head[first:first + chunk]).T))
        return np.concatenate(parts, axis=-1)


def gate_prompts(args):
    import gemma4_reference
    model = MODELS[args.model]
    if args.template is None:
        args.template = str(Path(__file__).resolve().parents[1] / model["template"])
    gemma4_reference.gate_prompts(args, model["head_keys"], model["tail_keys"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("gate-prompts", "teacher"))
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--variants", default="a,b")
    parser.add_argument("--template")
    parser.add_argument("--dump", help="teacher: write the embedding and every layer's residual for all rows to PATH.a.npz / PATH.b.npz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "gate-prompts":
        gate_prompts(args)
    elif not args.prompts:
        parser.error("teacher needs --prompts")
    else:
        base.teacher(args, Reference(args.checkpoint, args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
