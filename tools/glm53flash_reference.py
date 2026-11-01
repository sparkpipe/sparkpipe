#!/usr/bin/env python3
"""Teacher-forced numpy reference for GLM 5.3 Flash (the FP8 release) for the Metal gate.

  gate-prompts  the Metal gate prompts (tools/gemma4_reference.py's set) in
             GLM's chat turn (thinking off) -> --out prompts.json
  teacher    FP32 logits of every prompt position but the last:
             --prompts prompts.json -> --out DIR/<name>.ref_<v>.logits for
             --variants a,b, through tools/ling3_reference.py's teacher: a
             multiplies in numpy's order; b splits every matmul's inner
             dimension into 1,024-wide chunks summed last chunk first,
             reverses the other reductions (norms, the recurrent dot
             products, softmax sums) and combines the routed experts in
             descending order, so a against b is the FP32 reduction-order
             floor

The rounding points follow the CUDA stage (model-families/glm5_next/source):
the BF16 projections are the CUDA pack's bf16(e4m3 x block scale) through
tools/glm5_next_resident_stagepack.py's reader, the routed experts are
e4m3 x FP32 block scale in FP32, and
  hyper-connections  four BF16 streams; per row r = rsqrt(mean(x^2) + eps)
             over all 4H values, 24 mixes m = (fn . x) r; pre = sigmoid(m
             s0 + b) + 1e-6, post = 2 sigmoid(m s1 + b), the 4x4
             combination softmax per row + 1e-6 then 20 Sinkhorn passes
             (from the second a row normalisation, then a column one, each
             + 1e-6); the branch input bf16(sum pre_s x_s); the update
             x_t <- bf16(bf16(bf16(post_t) out) + bf16(sum_s bf16(C_st) x_s));
             the head reads bf16(sum_s x_s / 4)
  norms      bf16(bf16(x r) w) with BF16 gains
  KDA        q/k/v projections, causal depthwise conv (4 taps) + SiLU
             rounded once, q and k L2-normalised (eps 1e-6 inside the sqrt),
             q scaled by 1/sqrt(128), decay exp(-5 sigmoid(exp(A_log) (f +
             dt_bias))) from the rank-128 f_a / f_b projections, beta =
             bf16(sigmoid(b)), the delta rule in FP32, the output rounded,
             then bf16(o r w sigmoid(g)) with the FP32 o_norm gain and the
             rank-128 g_a / g_b gate, and o_proj
  MLA        q_a -> norm -> q_b (256 per head, no rope), kv_a -> norm (the
             512-wide latent is key and value), the query absorbed per
             head through kv_b's key rows (rounded), softmax(q.k / 16) in
             FP32 over every earlier position (the indexer keeps them all
             below 2,048 keys), the latent output rounded, kv_b's value
             rows per head, o_proj
  MLP        bf16(silu(min(g, 10)) clamp(u, +-10)) rounded once; layers 0-2
             dense, the rest the shared expert plus top 8 of 288 experts
             by sigmoid(z) + e_score_correction_bias, weights renormalised
             and scaled by 2.5, each expert's output rounded and combined
             in pick order in FP32, then bf16(bf16(routed) + shared)
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ling3_reference as base  # noqa: E402
from glm5_next_resident_stagepack import SourceReader, fp8_e4m3_lut  # noqa: E402

PREFIX = "model.language_model."
HC_EPSILON = np.float32(1e-6)
L2_EPSILON = np.float32(1e-6)
LIMIT = np.float32(10.0)


def widen(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


class Reference:
    def __init__(self, directory):
        self.source = SourceReader(Path(directory), cache_byte_cap=2 * 1024 ** 3)
        config = self.source.config.get("text_config", self.source.config)
        self.config = config
        self.hidden = config["hidden_size"]
        self.layers = config["num_hidden_layers"]
        self.vocab = config["vocab_size"]
        self.epsilon = np.float32(config["rms_norm_eps"])
        self.streams = config["hc_mult"]
        self.sinkhorn = config["hc_sinkhorn_iters"]
        self.heads = config["num_attention_heads"]
        self.latent = config["kv_lora_rank"]
        self.nope = config["qk_nope_head_dim"]
        self.value_dim = config["v_head_dim"]
        self.query_lora = config["q_lora_rank"]
        linear = config["linear_attn_config"]
        self.kda_heads = linear["num_heads"]
        self.kda_dim = linear["head_dim"]
        self.conv = linear["short_conv_kernel_size"]
        self.lower_bound = np.float32(linear.get("gate_lower_bound", -5.0))
        self.experts = config["n_routed_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.route_scale = np.float32(config["routed_scaling_factor"])
        self.moe_inter = config["moe_intermediate_size"]
        self.dense_layers = config["first_k_dense_replace"]
        self.index_topk = config.get("index_topk", 2048)
        if config.get("qk_rope_head_dim", 0) != 0 or not config.get("norm_topk_prob", True) or config.get("n_shared_experts") != 1:
            raise ValueError("the reference implements GLM 5.3 Flash: no rope, renormalised top-k, one shared expert")
        self.lut = fp8_e4m3_lut()

    def spine(self, name):
        return widen(self.source.spine_bf16(name))

    def spine_vector(self, name):
        return self.spine(name).reshape(-1)

    def f32(self, name):
        return self.source.spine_f32(name)

    def expert(self, name):
        dtype, shape, _ = self.source.meta(name)
        if dtype != "F8_E4M3":
            raise ValueError(f"{name}: expected F8_E4M3, got {dtype}")
        codes = np.asarray(self.source.raw(name)).reshape(shape)
        scale = np.asarray(self.source.raw(name + "_scale_inv")).view(np.float32).reshape((shape[0] + 127) // 128, (shape[1] + 127) // 128)
        expanded = np.repeat(np.repeat(scale, 128, axis=0)[:shape[0]], 128, axis=1)[:, :shape[1]]
        return self.lut[codes] * expanded

    def is_kda(self, layer):
        return layer % 4 != 3

    def norm(self, values, weight):
        scale = np.power(base.reduce_sum(values * values) / np.float32(values.shape[-1]) + self.epsilon, np.float32(-0.5))
        return base.bf16(base.bf16(values * scale) * weight)

    def linear(self, values, weight):
        return base.bf16(base.matmul(values, weight.T))

    def hc_site(self, streams, prefix, site):
        rows = streams.shape[0]
        flat = streams.reshape(rows, -1)
        inverse = np.power(base.reduce_sum(flat * flat) / np.float32(flat.shape[1]) + self.epsilon, np.float32(-0.5))
        mixes = base.matmul(flat, self.f32(f"{prefix}hc_{site}_fn").T) * inverse
        bias = self.f32(f"{prefix}hc_{site}_base").reshape(-1)
        scale = self.f32(f"{prefix}hc_{site}_scale").reshape(-1)
        n = self.streams
        pre = base.sigmoid(mixes[:, :n] * scale[0] + bias[:n]) + HC_EPSILON
        post = np.float32(2.0) * base.sigmoid(mixes[:, n:2 * n] * scale[1] + bias[n:2 * n])
        logits = mixes[:, 2 * n:].reshape(rows, n, n) * scale[2] + bias[2 * n:].reshape(n, n)
        comb = np.exp(logits - logits.max(axis=-1, keepdims=True))
        comb = comb / base.reduce_sum(comb, axis=-1) + HC_EPSILON
        for iteration in range(self.sinkhorn):
            if iteration != 0:
                comb = comb / (base.reduce_sum(comb, axis=-1) + HC_EPSILON)
            comb = comb / (base.reduce_sum(comb, axis=-2) + HC_EPSILON)
        terms = pre[:, :, None] * streams
        collapsed = base.bf16(base.reduce_sum(terms, axis=1, keepdims=False))
        return collapsed, post, comb

    def hc_post(self, streams, output, post, comb):
        contribution = base.bf16(base.bf16(post)[:, :, None] * output[:, None, :])
        weights = base.bf16(comb)
        mixed = np.zeros_like(streams)
        for source in (range(self.streams - 1, -1, -1) if base.REVERSED else range(self.streams)):
            mixed += weights[:, source, :, None] * streams[:, source, None, :]
        return base.bf16(contribution + base.bf16(mixed))

    def conv_silu(self, values, weight):
        padded = np.concatenate([np.zeros((self.conv - 1, values.shape[1]), dtype=np.float32), values], axis=0)
        total = np.zeros_like(values)
        for tap in range(self.conv):
            total += padded[tap:tap + values.shape[0]] * weight[:, tap][None, :]
        return base.bf16(base.silu(total))

    def kda(self, a, normed, weights):
        rows, heads, dim = normed.shape[0], self.kda_heads, self.kda_dim
        q, k, v = (self.conv_silu(self.linear(normed, weights[key]), weights[key + "_conv"]).reshape(rows, heads, dim) for key in ("q", "k", "v"))
        decay_logit = self.linear(self.linear(normed, weights["f_a"]), weights["f_b"]).reshape(rows, heads, dim)
        decay = np.exp(self.lower_bound * base.sigmoid(np.exp(weights["a_log"])[None, :, None] * (decay_logit + weights["dt_bias"][None])))
        beta = base.bf16(base.sigmoid(self.linear(normed, weights["b"])))
        q = q / np.sqrt(base.reduce_sum(q * q) + L2_EPSILON) * np.float32(dim ** -0.5)
        k = k / np.sqrt(base.reduce_sum(k * k) + L2_EPSILON)
        state = np.zeros((heads, dim, dim), dtype=np.float32)
        output = np.empty((rows, heads, dim), dtype=np.float32)
        for t in range(rows):
            state *= decay[t][:, :, None]
            delta = (v[t] - base.dot_last(state, k[t])) * beta[t][:, None]
            state += k[t][:, :, None] * delta[:, None, :]
            output[t] = base.dot_last(state, q[t])
        output = base.bf16(output)
        gate = self.linear(self.linear(normed, weights["g_a"]), weights["g_b"]).reshape(rows, heads, dim)
        scale = np.power(base.reduce_sum(output * output) / np.float32(dim) + self.epsilon, np.float32(-0.5))
        mixed = base.bf16(output * scale * weights["o_norm"] * base.sigmoid(gate))
        return self.linear(mixed.reshape(rows, heads * dim), weights["o"])

    def kda_weights(self, a):
        names = {"q": "q_proj.weight", "k": "k_proj.weight", "v": "v_proj.weight", "b": "b_proj.weight", "f_a": "f_a_proj.weight",
                 "f_b": "f_b_proj.weight", "g_a": "g_a_proj.weight", "g_b": "g_b_proj.weight", "o": "o_proj.weight", "q_conv": "q_conv1d.weight",
                 "k_conv": "k_conv1d.weight", "v_conv": "v_conv1d.weight"}
        weights = {key: self.spine(a + name) for key, name in names.items()}
        weights["a_log"] = self.f32(a + "A_log").reshape(-1)
        weights["dt_bias"] = self.f32(a + "dt_bias").reshape(self.kda_heads, self.kda_dim)
        weights["o_norm"] = self.f32(a + "o_norm.weight").reshape(-1)
        return weights

    def mla(self, normed, positions, weights):
        rows, heads = normed.shape[0], self.heads
        compressed = self.norm(self.linear(normed, weights["q_a"]), weights["q_norm"])
        q = self.linear(compressed, weights["q_b"]).reshape(rows, heads, self.nope)
        latent = self.norm(self.linear(normed, weights["kv_a"]), weights["kv_norm"])
        causal = positions[None, :] <= positions[:, None]
        output = np.empty((rows, heads, self.value_dim), dtype=np.float32)
        for head in range(heads):
            key_up, value_up = weights["kv_b"][head, :self.nope], weights["kv_b"][head, self.nope:]
            query = base.bf16(base.matmul(q[:, head, :], key_up))
            scores = base.matmul(query, latent.T) * np.float32(1.0 / 16.0)
            scores = np.where(causal, scores, -np.inf)
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities = probabilities / base.reduce_sum(probabilities)
            attended = base.bf16(base.matmul(probabilities, latent))
            output[:, head, :] = base.bf16(base.matmul(attended, value_up.T))
        return self.linear(output.reshape(rows, heads * self.value_dim), weights["o"])

    def mla_weights(self, a):
        weights = {"q_a": self.spine(a + "q_a_proj.weight"), "q_norm": self.spine_vector(a + "q_a_layernorm.weight"),
                   "q_b": self.spine(a + "q_b_proj.weight"), "kv_a": self.spine(a + "kv_a_proj_with_mqa.weight"),
                   "kv_norm": self.spine_vector(a + "kv_a_layernorm.weight"), "o": self.spine(a + "o_proj.weight")}
        weights["kv_b"] = self.spine(a + "kv_b_proj.weight").reshape(self.heads, self.nope + self.value_dim, self.latent)
        return weights

    def glu(self, values, gate_weight, up_weight, down_weight):
        gate = np.minimum(self.linear(values, gate_weight), LIMIT)
        up = np.clip(self.linear(values, up_weight), -LIMIT, LIMIT)
        return self.linear(base.bf16(base.silu(gate) * up), down_weight)

    def mlp(self, values, prefix):
        return self.glu(values, self.spine(prefix + "gate_proj.weight"), self.spine(prefix + "up_proj.weight"), self.spine(prefix + "down_proj.weight"))

    def route(self, m, normed):
        logits = base.matmul(normed, self.spine(m + "gate.weight").T)
        scores = base.sigmoid(logits)
        selection = scores + self.f32(m + "gate.e_score_correction_bias").reshape(-1)[None, :]
        chosen = np.argsort(-selection, axis=-1, kind="stable")[:, :self.top_k]
        picked = np.take_along_axis(scores, chosen, axis=-1)
        weights = picked / (base.reduce_sum(picked) + np.float32(1e-20)) * self.route_scale
        return chosen, weights.astype(np.float32)

    def expert_weights(self, prefix):
        return tuple(self.expert(prefix + f"{key}_proj.weight") for key in ("gate", "up", "down"))

    def moe(self, layer, normed):
        m = f"{PREFIX}layers.{layer}.mlp."
        chosen, weights = self.route(m, normed)
        outputs = np.zeros((normed.shape[0], self.top_k, self.hidden), dtype=np.float32)
        experts = [int(expert) for expert in (np.unique(chosen)[::-1] if base.REVERSED else np.unique(chosen))]
        window = 8
        with ThreadPoolExecutor(8) as pool:
            pending = {expert: pool.submit(self.expert_weights, m + f"experts.{expert}.") for expert in experts[:window]}
            for index, expert in enumerate(experts):
                if index + window < len(experts):
                    following = experts[index + window]
                    pending[following] = pool.submit(self.expert_weights, m + f"experts.{following}.")
                rows, slots = np.nonzero(chosen == expert)
                outputs[rows, slots] = self.glu(normed[rows], *pending.pop(expert).result())
        order = range(self.top_k - 1, -1, -1) if base.REVERSED else range(self.top_k)
        combined = np.zeros((normed.shape[0], self.hidden), dtype=np.float32)
        for slot in order:
            combined += outputs[:, slot] * weights[:, slot][:, None]
        shared = self.mlp(normed, m + "shared_experts.")
        return base.bf16(base.bf16(combined) + shared), chosen

    def forward(self, segments, dump=None):
        tokens = np.concatenate([np.asarray(s, dtype=np.int64) for s in segments])
        positions = np.concatenate([np.arange(len(s), dtype=np.int64) for s in segments])
        bounds = np.cumsum([0] + [len(s) for s in segments])
        if positions.max() >= self.index_topk:
            raise ValueError(f"a prompt reaches position {positions.max()}: beyond {self.index_topk} keys the indexer drops some and this reference is dense")
        embed = self.source.raw(PREFIX + "embed_tokens.weight").view(np.uint16).reshape(self.vocab, self.hidden)
        hidden = widen(embed[tokens])
        streams = np.repeat(hidden[:, None, :], self.streams, axis=1)
        if dump is not None:
            dump["embed"] = hidden.copy()
        routes = {}
        for layer in range(self.layers):
            began = time.time()
            p = f"{PREFIX}layers.{layer}."
            a = p + "self_attn."
            collapsed, post, comb = self.hc_site(streams, p, "attn")
            normed = self.norm(collapsed, self.spine_vector(p + "input_layernorm.weight"))
            weights = self.kda_weights(a) if self.is_kda(layer) else self.mla_weights(a)
            attention = np.empty_like(normed)
            for first, last in zip(bounds[:-1], bounds[1:]):
                part = normed[first:last]
                attention[first:last] = self.kda(a, part, weights) if self.is_kda(layer) else self.mla(part, positions[first:last], weights)
            del weights
            streams = self.hc_post(streams, attention, post, comb)
            collapsed, post, comb = self.hc_site(streams, p, "ffn")
            normed = self.norm(collapsed, self.spine_vector(p + "post_attention_layernorm.weight"))
            if layer < self.dense_layers:
                mlp = self.mlp(normed, p + "mlp.")
            else:
                mlp, routes[layer] = self.moe(layer, normed)
            streams = self.hc_post(streams, mlp, post, comb)
            if dump is not None:
                dump[f"layer{layer:02d}"] = streams.copy()
            print(f"layer {layer} {'kda' if self.is_kda(layer) else 'mla'} done in {time.time() - began:.1f}s", file=sys.stderr, flush=True)
        mean = base.bf16(base.reduce_sum(streams, axis=1, keepdims=False) / np.float32(self.streams))
        return self.norm(mean, self.spine_vector(PREFIX + "norm.weight")), routes

    def logits(self, normed, chunk=16384):
        head = self.source.raw("lm_head.weight").view(np.uint16).reshape(self.vocab, self.hidden)
        parts = []
        for first in range(0, self.vocab, chunk):
            parts.append(base.matmul(normed, widen(head[first:first + chunk]).T))
        return np.concatenate(parts, axis=-1)


def gate_prompts(args):
    import gemma4_reference
    gemma4_reference.gate_prompts(args, ("prefix", "user"), ("turn_suffix", "generation"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("gate-prompts", "teacher"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--variants", default="a,b")
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/glm5_next/chat_template.json"))
    parser.add_argument("--dump", help="teacher: write the embedding and every layer's streams for all rows to PATH.a.npz / PATH.b.npz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "gate-prompts":
        gate_prompts(args)
    elif not args.prompts:
        parser.error("teacher needs --prompts")
    else:
        base.teacher(args, Reference(args.checkpoint))
    return 0


if __name__ == "__main__":
    sys.exit(main())
