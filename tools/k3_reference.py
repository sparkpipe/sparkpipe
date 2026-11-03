#!/usr/bin/env python3
"""Teacher-forced numpy reference for Kimi K3 for the Metal gate.

  gate-prompts  the Metal gate prompts (tools/gemma4_reference.py's set) in
             K3's chat turn (model-families/k3/chat_template.json), encoded
             with the tiktoken vocabulary the hub frontend serves
             (deployment/hub/chat_frontend.json) -> --out prompts.json
  teacher    FP32 logits of every prompt position but the last:
             --prompts prompts.json -> --out DIR/<name>.ref_<v>.logits for
             --variants a,b,h, through tools/ling3_reference.py's teacher (b
             sums every product in 1,024-wide chunks last first, reverses the
             other reductions and combines the experts in descending order;
             h is b with a GPU's exp and tanh error (tools/reference_device_error.py),
             and the three variants' distances from one another are the
             gate's floors)

The rounding points follow the CUDA stage (inference/llms/kimi_k3/layer.cuh):
  AttnRes    per source r = rsqrt(mean(v^2) + 1e-5), score = sum v r w with
             the folded bf16(gamma proj) weight, softmax over the sources,
             bf16(sum p v); the bank keeps the running partial at every
             twelfth layer, and the partial takes each branch as bf16(p + b)
             (set at a block boundary)
  norms      bf16(bf16(x r) w), the publisher's KimiRMSNorm (weight *
             x.to(bf16)), eps 1e-5 (1e-6 for q_a and the kv latent)
  KDA        q/k/v projections, the causal conv with FP32 weights and SiLU
             on the unrounded sum, q and k L2-normalised per head (eps 1e-5)
             and rounded, then normalised again inside the rule (eps 1e-6, q
             by 1/sqrt(128)); decay exp(-5 sigmoid(exp(A_log) (f + dt_bias)))
             from the rank-128 f_a / f_b; beta sigmoid(b) unrounded; the
             output rounded, normed with the FP32 o_norm and rounded, then
             bf16(o sigmoid(g)) with the full-rank gate, and o_proj
  MLA        q_a -> norm -> q_b (128 nope + 64 unrotated per head); kv_a ->
             the 512 latent norm and the 64 raw values; the query absorbed
             per head through kv_b's key rows (rounded); softmax((q_l . k_l +
             q_r . k_r) x 0.0722) in FP32 over every earlier position; the
             latent output rounded; kv_b's value rows; bf16(v sigmoid(g));
             o_proj
  MLP        SiTU: bf16(4 tanh(g/4) sigmoid(g) x 25 tanh(u/25)); layer 0
             dense; the rest the two fused shared experts plus top 16 of 896
             by sigmoid(z) + bias, weights renormalised, the experts in the
             3,584-wide latent (routed_down, MXFP4 experts e2m1 x 2^(e - 127)
             exactly, combined in pick order in FP32 and rounded,
             routed_norm, routed_up), branch bf16(routed + shared)
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import reference_device_error as device  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ling3_reference as base  # noqa: E402
from k3_pack import SafetensorDir, gamma_fold_bf16, quant_pair  # noqa: E402

S = "language_model.model."
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def widen(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def situ(gate, up):
    activated = np.float32(4.0) * device.tanh(gate / np.float32(4.0)) * base.sigmoid(gate)
    return base.bf16(activated * (np.float32(25.0) * device.tanh(up / np.float32(25.0))))


class Reference:
    def __init__(self, directory):
        self.st = SafetensorDir(directory)
        self.names = self.st.names()
        config = json.loads((Path(directory) / "config.json").read_text())
        config = config.get("text_config", config)
        self.hidden = config["hidden_size"]
        self.layers = config["num_hidden_layers"]
        self.vocab = config["vocab_size"]
        self.heads, self.key = 96, 128
        self.nope, self.rope, self.value, self.latent_rank, self.q_lora = 128, 64, 128, 512, 1536
        self.experts, self.top_k, self.inter, self.latent = 896, 16, 3072, 3584
        self.block = 12

    def bf16(self, name, shape):
        return widen(np.frombuffer(self.st.bf16(name), dtype=np.uint16).reshape(shape))

    def f32(self, name, shape):
        return np.frombuffer(self.st.f32(name), dtype=np.float32).reshape(shape)

    def gain(self, name, count):
        return self.bf16(name, (count,))

    def fold(self, gamma, proj):
        return widen(np.frombuffer(gamma_fold_bf16(self.st.bf16(proj, (1, self.hidden)), self.st.bf16(gamma, (self.hidden,)), self.hidden), dtype=np.uint16))

    def mxfp4(self, base_name, rows, columns):
        values, scales = quant_pair(self.st, base_name)
        packed = np.frombuffer(self.st.u8(values, (rows, columns // 2)), dtype=np.uint8).reshape(rows, columns // 2)
        exponents = np.frombuffer(self.st.u8(scales, (rows, columns // 32)), dtype=np.uint8).reshape(rows, columns // 32)
        weights = np.empty((rows, columns), dtype=np.float32)
        weights[:, 0::2] = E2M1[packed & 0x0F]
        weights[:, 1::2] = E2M1[packed >> 4]
        return weights * np.repeat(np.ldexp(np.float32(1.0), exponents.astype(np.int32) - 127).astype(np.float32), 32, axis=1)

    def scale_of(self, values, epsilon):
        return np.power(base.reduce_sum(values * values) / np.float32(values.shape[-1]) + np.float32(epsilon), np.float32(-0.5))

    def norm(self, values, gain, epsilon=1e-5):
        return base.bf16(base.bf16(values * self.scale_of(values, epsilon)) * gain)

    def linear(self, values, weight):
        return base.bf16(base.matmul(values, weight.T))

    def attnres(self, sources, weight):
        scores = np.stack([base.reduce_sum(source * self.scale_of(source, 1e-5) * weight, keepdims=False) for source in sources], axis=-1)
        probabilities = device.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities = probabilities / base.reduce_sum(probabilities)
        order = range(len(sources) - 1, -1, -1) if base.REVERSED else range(len(sources))
        total = np.zeros_like(sources[0])
        for index in order:
            total += probabilities[:, index:index + 1] * sources[index]
        return base.bf16(total)

    def conv_silu(self, values, weight):
        padded = np.concatenate([np.zeros((3, values.shape[1]), dtype=np.float32), values], axis=0)
        total = np.zeros_like(values)
        for tap in range(4):
            total += padded[tap:tap + values.shape[0]] * weight[:, tap][None, :]
        return base.bf16(base.silu(total))

    def kda(self, a, normed):
        rows, heads, dim, width = normed.shape[0], self.heads, self.key, self.heads * self.key
        streams = []
        for key in "qkv":
            projected = self.linear(normed, self.bf16(a + f"{key}_proj.weight", (width, self.hidden)))
            streams.append(self.conv_silu(projected, self.f32(a + f"{key}_conv1d.weight", (width, 1, 4))[:, 0, :]).reshape(rows, heads, dim))
        q, k, v = streams
        q = base.bf16(q / np.sqrt(base.reduce_sum(q * q) + np.float32(1e-5)))
        k = base.bf16(k / np.sqrt(base.reduce_sum(k * k) + np.float32(1e-5)))
        q = q / np.sqrt(base.reduce_sum(q * q) + np.float32(1e-6)) * np.float32(dim ** -0.5)
        k = k / np.sqrt(base.reduce_sum(k * k) + np.float32(1e-6))
        decay_logit = self.linear(self.linear(normed, self.bf16(a + "f_a_proj.weight", (dim, self.hidden))),
                                  self.bf16(a + "f_b_proj.weight", (width, dim))).reshape(rows, heads, dim)
        a_log = self.f32(a + "A_log", (128,))[:heads]
        dt_bias = self.f32(a + "dt_bias", (width,)).reshape(heads, dim)
        decay = device.exp(np.float32(-5.0) * base.sigmoid(device.exp(a_log)[None, :, None] * (decay_logit + dt_bias[None])))
        beta = base.sigmoid(self.linear(normed, self.bf16(a + "b_proj.weight", (heads, self.hidden))))
        state = np.zeros((heads, dim, dim), dtype=np.float32)
        output = np.empty((rows, heads, dim), dtype=np.float32)
        for t in range(rows):
            state *= decay[t][:, :, None]
            delta = (v[t] - base.dot_last(state, k[t])) * beta[t][:, None]
            state += k[t][:, :, None] * delta[:, None, :]
            output[t] = base.dot_last(state, q[t])
        output = base.bf16(output)
        normed_out = base.bf16(output * self.scale_of(output, 1e-5) * self.f32(a + "o_norm.weight", (dim,)))
        gate = self.linear(normed, self.bf16(a + "g_proj.weight", (width, self.hidden))).reshape(rows, heads, dim)
        mixed = base.bf16(normed_out * base.sigmoid(gate))
        return self.linear(mixed.reshape(rows, width), self.bf16(a + "o_proj.weight", (self.hidden, width)))

    def mla(self, a, normed, positions):
        rows, heads = normed.shape[0], self.heads
        compressed = self.norm(self.linear(normed, self.bf16(a + "q_a_proj.weight", (self.q_lora, self.hidden))),
                               self.gain(a + "q_a_layernorm.weight", self.q_lora), 1e-6)
        q = self.linear(compressed, self.bf16(a + "q_b_proj.weight", (heads * (self.nope + self.rope), self.q_lora))).reshape(rows, heads, -1)
        kv = self.linear(normed, self.bf16(a + "kv_a_proj_with_mqa.weight", (self.latent_rank + self.rope, self.hidden)))
        latent = self.norm(kv[:, :self.latent_rank], self.gain(a + "kv_a_layernorm.weight", self.latent_rank), 1e-6)
        tail = kv[:, self.latent_rank:]
        kv_b = self.bf16(a + "kv_b_proj.weight", (heads * (self.nope + self.value), self.latent_rank)).reshape(heads, self.nope + self.value, -1)
        gate = self.linear(normed, self.bf16(a + "g_proj.weight", (heads * self.value, self.hidden))).reshape(rows, heads, self.value)
        causal = positions[None, :] <= positions[:, None]
        output = np.empty((rows, heads, self.value), dtype=np.float32)
        for head in range(heads):
            query = base.bf16(base.matmul(q[:, head, :self.nope], kv_b[head, :self.nope]))
            scores = (base.matmul(query, latent.T) + base.matmul(q[:, head, self.nope:], tail.T)) * np.float32(0.07216878365)
            scores = np.where(causal, scores, -np.inf)
            probabilities = device.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities = probabilities / base.reduce_sum(probabilities)
            attended = base.bf16(base.matmul(probabilities, latent))
            value = base.bf16(base.matmul(attended, kv_b[head, self.nope:].T))
            output[:, head, :] = base.bf16(value * base.sigmoid(gate[:, head, :]))
        return self.linear(output.reshape(rows, heads * self.value), self.bf16(a + "o_proj.weight", (self.hidden, heads * self.value)))

    def gated(self, values, prefix, inter):
        gate = self.linear(values, self.bf16(prefix + "gate_proj.weight", (inter, self.hidden)))
        up = self.linear(values, self.bf16(prefix + "up_proj.weight", (inter, self.hidden)))
        return self.linear(situ(gate, up), self.bf16(prefix + "down_proj.weight", (self.hidden, inter)))

    def expert(self, prefix):
        return (self.mxfp4(prefix + "w1", self.inter, self.latent), self.mxfp4(prefix + "w3", self.inter, self.latent),
                self.mxfp4(prefix + "w2", self.latent, self.inter))

    def moe(self, m, normed):
        logits = base.matmul(normed, self.bf16(m + "gate.weight", (self.experts, self.hidden)).T)
        scores = base.sigmoid(logits)
        dtype, _, raw = self.st.tensor(m + "gate.e_score_correction_bias")
        bias = np.frombuffer(raw, dtype=np.float32) if dtype == "F32" else widen(np.frombuffer(raw, dtype=np.uint16))
        chosen = np.argsort(-(scores + bias[None, :]), axis=-1, kind="stable")[:, :self.top_k]
        picked = np.take_along_axis(scores, chosen, axis=-1)
        weights = picked / (base.reduce_sum(picked) + np.float32(1e-20))
        latent = self.linear(normed, self.bf16(m + "routed_expert_down_proj.weight", (self.latent, self.hidden)))
        outputs = np.zeros((normed.shape[0], self.top_k, self.latent), dtype=np.float32)
        experts = [int(e) for e in (np.unique(chosen)[::-1] if base.REVERSED else np.unique(chosen))]
        window = 6
        with ThreadPoolExecutor(6) as pool:
            pending = {e: pool.submit(self.expert, m + f"experts.{e}.") for e in experts[:window]}
            for index, expert in enumerate(experts):
                if index + window < len(experts):
                    following = experts[index + window]
                    pending[following] = pool.submit(self.expert, m + f"experts.{following}.")
                rows, slots = np.nonzero(chosen == expert)
                w1, w3, w2 = pending.pop(expert).result()
                x = latent[rows]
                outputs[rows, slots] = self.linear(situ(self.linear(x, w1), self.linear(x, w3)), w2)
        combined = np.zeros((normed.shape[0], self.latent), dtype=np.float32)
        for slot in (range(self.top_k - 1, -1, -1) if base.REVERSED else range(self.top_k)):
            combined += outputs[:, slot] * weights[:, slot][:, None]
        routed = self.linear(self.norm(base.bf16(combined), self.gain(m + "routed_expert_norm.weight", self.latent)),
                             self.bf16(m + "routed_expert_up_proj.weight", (self.hidden, self.latent)))
        shared = self.gated(normed, m + "shared_experts.", 6144)
        return base.bf16(routed + shared), chosen

    def forward(self, segments, dump=None):
        tokens = np.concatenate([np.asarray(s, dtype=np.int64) for s in segments])
        positions = np.concatenate([np.arange(len(s), dtype=np.int64) for s in segments])
        bounds = np.cumsum([0] + [len(s) for s in segments])
        embed = np.frombuffer(self.st.bf16(S + "embed_tokens.weight"), dtype=np.uint16).reshape(self.vocab, self.hidden)
        hidden = widen(embed[tokens])
        bank, partial, routes = [], None, {}
        for layer in range(self.layers):
            began = time.time()
            p = f"{S}layers.{layer}."
            if layer > 0:
                hidden = self.attnres(bank + [partial], self.fold(p + "self_attention_res_norm.weight", p + "self_attention_res_proj.weight"))
            if layer % self.block == 0:
                if layer == 0:
                    partial = hidden.copy()
                bank.append(partial.copy())
            normed = self.norm(hidden, self.gain(p + "input_layernorm.weight", self.hidden))
            branch = np.empty_like(normed)
            kda = layer % 4 != 3 and layer != self.layers - 1
            for first, last in zip(bounds[:-1], bounds[1:]):
                part = normed[first:last]
                branch[first:last] = self.kda(p + "self_attn.", part) if kda else self.mla(p + "self_attn.", part, positions[first:last])
            partial = branch if layer % self.block == 0 else base.bf16(partial + branch)
            hidden = self.attnres(bank + [partial], self.fold(p + "mlp_res_norm.weight", p + "mlp_res_proj.weight"))
            normed = self.norm(hidden, self.gain(p + "post_attention_layernorm.weight", self.hidden))
            m = p + "block_sparse_moe."
            if m + "gate.weight" in self.names:
                branch, routes[layer] = self.moe(m, normed)
            else:
                branch = self.gated(normed, p + "mlp.", 33792)
            partial = base.bf16(partial + branch)
            if dump is not None:
                dump[f"layer{layer:02d}"] = partial.copy()
            print(f"layer {layer} {'kda' if kda else 'mla'} done in {time.time() - began:.1f}s", file=sys.stderr, flush=True)
        hidden = self.attnres(bank + [partial], self.fold(S + "output_attn_res_norm.weight", S + "output_attn_res_proj.weight"))
        return self.norm(hidden, self.gain(S + "norm.weight", self.hidden)), routes

    def logits(self, normed, chunk=16384):
        head = np.frombuffer(self.st.bf16("language_model.lm_head.weight"), dtype=np.uint16).reshape(self.vocab, self.hidden)
        return np.concatenate([base.matmul(normed, widen(head[first:first + chunk]).T) for first in range(0, self.vocab, chunk)], axis=-1)


def gate_prompts(args):
    import gemma4_reference
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools" / "serving"))
    from chat_frontend import TiktokenVocabulary
    models = json.loads((root / "deployment/hub/chat_frontend.json").read_text())["models"]
    spec = next(model for model in models if model.get("id") == "kimi-k3")["tokenizer"]
    vocabulary = TiktokenVocabulary(Path(args.checkpoint), spec)
    gemma4_reference.gate_prompts(args, ("prefix", "user"), ("turn_suffix", "generation"), encode=lambda text: vocabulary.encode(text, True),
                                  tokenizer_file=spec["file"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("gate-prompts", "teacher"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--variants", default="a,b,h")
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/k3/chat_template.json"))
    parser.add_argument("--dump", help="teacher: write every layer's partial for all rows to PATH.a.npz / PATH.b.npz")
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
