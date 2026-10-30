#!/usr/bin/env python3
"""Teacher-forced numpy reference for Solar Open 2 (SolarOpen2ForCausalLM).

  tokenize   render the publisher chat_template.jinja from the checkpoint
             (reasoning_effort low, so the reply opens with an empty think
             block; one user turn, generation prompt) and encode it with the
             checkpoint tokenizer: --prompts in.json (list of {name, text})
             -> --out prompts.json ({"prompts": [{name, prompt_token_ids}]})
  score      as tools/ling3_reference.py score: one forward over prompt +
             served tokens per prompt against --capture capture.json
  gate-prompts, teacher  as tools/ling3_reference.py: the Metal gate prompts
             in Solar's chat turn, and FP32 references a and b

Semantics follow the publisher modular_solar_open2.py and the fla kernels its
fast path calls (model_contracts/references/fla_kda_gate.py):
  layers in gqa_layers (0, 4, ..., 44) are GQA, the rest KDA; every layer MoE.
  KDA token by token: q/k/v projections, each through its depthwise causal
  conv (4 taps), the conv output rounded to bf16 and SiLU applied in bf16 as
  F.conv1d then F.silu do; q and k L2-normalized with eps 1e-6 inside the
  sqrt, q scaled by 1/sqrt(128); decay g = f_b(f_a(x)) through
  fused_kda_gate with lower_bound -5: log decay -5 * sigmoid(exp(A_log) *
  (g + dt_bias)) in f32 (the torch fallback's -exp(A_log) * softplus clamp
  is a different function; the fla form is the one the fast path runs);
  beta = 2 * sigmoid(b) in bf16; S <- exp(g) S, v' = beta (v - S^T k),
  S <- S + k v'^T, o = S^T q; o rounded to bf16, then RMSNorm(o) * w *
  sigmoid(g_b(g_a(x))) in f32 and o_proj.
  GQA: 64 query heads over 8 KV heads, no rope, no qk norm, causal
  softmax(QK^T / sqrt(128)) in f32 with bf16 probabilities, output times
  bf16 sigmoid(g_proj(x)), o_proj.
  MoE: f32 router logits, sigmoid scores, e_score_correction_bias for
  selection only, one group, top-8, weights renormalized (+1e-20), scale 1;
  expert outputs combined in f32; the shared expert added in bf16.
Rounding otherwise follows tools/ling3_reference.py.

In the score mode every matrix product accumulates in f64 before it is
rounded (t1_reference_common.matmul), so a prompt's numbers do not depend on
the other prompts in the run or on how many served tokens follow; the teacher
mode multiplies and reduces as tools/ling3_reference.py teacher does.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ling3_reference  # noqa: E402
from ling3_reference import bf16, dot_last, matmul, reduce_sum, sigmoid, silu  # noqa: E402


class Reference(ling3_reference.Reference):
    embed_name = "model.embed_tokens.weight"
    expert_bias_name = "gate.e_score_correction_bias"
    attention_kind = "gqa"

    def __init__(self, directory, conv_rounding="publisher"):
        self.conv_rounding = conv_rounding
        self.checkpoint = ling3_reference.Checkpoint(directory)
        config = json.loads((Path(directory) / "config.json").read_text())
        self.config = config
        linear = config["linear_attn_config"]
        expect = {"model_type": "solar_open2", "use_rope": False, "use_gqa_gate": True, "kda_use_full_proj": False,
                  "kda_allow_neg_eigval": True, "first_k_dense_replace": 0, "n_shared_experts": 1, "norm_topk_prob": True,
                  "routed_scaling_factor": 1.0, "tie_word_embeddings": False}
        for key, value in expect.items():
            if config.get(key) != value:
                raise ValueError(f"config {key} is {config.get(key)!r}, the reference implements {value!r}")
        if config.get("kda_gate_lower_bound", -5.0) != -5.0 or config.get("use_qk_norm") or config.get("attention_bias") or \
                config.get("n_group", 1) != 1 or linear.get("num_kv_heads") not in (None, linear["num_heads"]):
            raise ValueError("the reference implements a -5 gate bound, no qk norm or attention bias, one routing group and full KDA heads")
        self.hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.kda_heads = linear["num_heads"]
        self.kda_dim = linear["head_dim"]
        self.conv = linear["short_conv_kernel_size"]
        self.epsilon = np.float32(config["rms_norm_eps"])
        self.lower_bound = np.float32(config.get("kda_gate_lower_bound", -5.0))
        self.experts = config["n_routed_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.groups = 1
        self.topk_groups = 1
        self.route_scale = np.float32(config["routed_scaling_factor"])
        self.moe_inter = config["moe_intermediate_size"]
        self.shared_inter = config["moe_intermediate_size"] * config["n_shared_experts"]
        self.layers = config["num_hidden_layers"]
        self.vocab = config["vocab_size"]
        self.gqa = set(config["gqa_layers"])
        self.routed_limits = [0.0] * self.layers
        self.shared_limits = [0.0] * self.layers

    def is_attention(self, layer):
        return layer in self.gqa

    def conv_silu(self, values, weight):
        padded = np.concatenate([np.zeros((self.conv - 1, values.shape[1]), dtype=np.float32), values], axis=0)
        total = np.zeros_like(values)
        for tap in range(self.conv):
            total += padded[tap:tap + values.shape[0]] * weight[:, tap][None, :]
        return bf16(silu(bf16(total) if self.conv_rounding == "publisher" else total))

    def low_rank(self, normed, prefix, width):
        return self.linear(self.linear(normed, prefix + "_a_proj.weight", self.kda_dim, self.hidden), prefix + "_b_proj.weight", width, self.kda_dim)

    def kda(self, layer, normed):
        a = f"model.layers.{layer}.self_attn."
        rows, heads, dim = normed.shape[0], self.kda_heads, self.kda_dim
        width = heads * dim
        streams = []
        for key in ("q", "k", "v"):
            projected = self.linear(normed, a + f"{key}_proj.weight", width, self.hidden)
            streams.append(self.conv_silu(projected, self.checkpoint.bf16(a + f"{key}_conv1d.weight", width, 1, self.conv)[:, 0, :]))
        q, k, v = (stream.reshape(rows, heads, dim) for stream in streams)
        f = self.low_rank(normed, a + "f", width).reshape(rows, heads, dim)
        a_log = self.checkpoint.bf16(a + "A_log", 1, 1, heads, 1).reshape(heads)
        dt_bias = self.checkpoint.bf16(a + "dt_bias", width).reshape(heads, dim)
        log_decay = self.lower_bound * sigmoid(np.exp(a_log)[None, :, None] * (f + dt_bias[None]))
        beta = bf16(sigmoid(self.linear(normed, a + "b_proj.weight", heads, self.hidden))) * np.float32(2.0)
        q = q / np.sqrt(reduce_sum(q * q) + np.float32(1e-6)) * np.float32(dim ** -0.5)
        k = k / np.sqrt(reduce_sum(k * k) + np.float32(1e-6))
        state = np.zeros((heads, dim, dim), dtype=np.float32)
        output = np.empty((rows, heads, dim), dtype=np.float32)
        decay = np.exp(log_decay)
        for t in range(rows):
            state *= decay[t][:, :, None]
            delta = (v[t] - dot_last(state, k[t])) * beta[t][:, None]
            state += k[t][:, :, None] * delta[:, None, :]
            output[t] = dot_last(state, q[t])
        output = bf16(output)
        gate = self.low_rank(normed, a + "g", width).reshape(rows, heads, dim)
        weight = self.checkpoint.bf16(a + "o_norm.weight", dim)
        mixed = output * np.power(reduce_sum(output * output) / np.float32(dim) + self.epsilon, np.float32(-0.5)) * weight * sigmoid(gate)
        return self.output(bf16(mixed).reshape(rows, width), a + "o_proj.weight", self.hidden, width)

    def attention(self, layer, normed, positions):
        a = f"model.layers.{layer}.self_attn."
        rows, heads, kv_heads, dim = normed.shape[0], self.heads, self.kv_heads, self.head_dim
        q = self.linear(normed, a + "q_proj.weight", heads * dim, self.hidden).reshape(rows, heads, dim)
        k = self.linear(normed, a + "k_proj.weight", kv_heads * dim, self.hidden).reshape(rows, kv_heads, dim)
        v = self.linear(normed, a + "v_proj.weight", kv_heads * dim, self.hidden).reshape(rows, kv_heads, dim)
        causal = positions[None, :] <= positions[:, None]
        scale = np.float32(dim ** -0.5)
        group = heads // kv_heads
        output = np.empty((rows, heads, dim), dtype=np.float32)
        for head in range(heads):
            scores = matmul(q[:, head, :], k[:, head // group, :].T) * scale
            scores = np.where(causal, scores, -np.inf)
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights = bf16(weights / reduce_sum(weights))
            output[:, head, :] = bf16(matmul(weights, v[:, head // group, :]))
        gate = bf16(sigmoid(self.linear(normed, a + "g_proj.weight", heads * dim, self.hidden)))
        output = bf16(output.reshape(rows, heads * dim) * gate)
        return self.output(output, a + "o_proj.weight", self.hidden, heads * dim)


def tokenize(args):
    import jinja2
    from tokenizers import Tokenizer
    checkpoint = Path(args.checkpoint)
    template = jinja2.Environment(trim_blocks=False, lstrip_blocks=False).from_string((checkpoint / "chat_template.jinja").read_text())
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    rendered = []
    for prompt in json.loads(Path(args.prompts).read_text()):
        text = template.render(messages=[{"role": "user", "content": prompt["text"]}], add_generation_prompt=True, reasoning_effort="low", tools=None)
        rendered.append({"name": prompt["name"], "text": text, "prompt_token_ids": tokenizer.encode(text, add_special_tokens=False).ids})
    Path(args.out).write_text(json.dumps({"prompts": rendered}, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("tokenize", "score", "gate-prompts", "teacher"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--variants", default="a,b")
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/solar_open2/chat_template.json"))
    parser.add_argument("--capture")
    parser.add_argument("--tie-margin", type=float, default=0.1)
    parser.add_argument("--conv-rounding", choices=("publisher", "fused"), default="publisher",
                        help="publisher rounds the short convolution to bf16 before SiLU as F.conv1d then F.silu do; fused rounds once after SiLU")
    parser.add_argument("--dump", help="write the embedding and every layer's post-attention and post-MLP residual for all rows to this .npz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "gate-prompts":
        ling3_reference.gate_prompts(args, ("prefix", "plain_prefix", "user"), ("user_suffix", "turn_suffix", "generation"))
    elif not args.prompts:
        parser.error(f"{args.mode} needs --prompts")
    elif args.mode == "tokenize":
        tokenize(args)
    elif args.mode == "teacher":
        ling3_reference.teacher(args, Reference(args.checkpoint, args.conv_rounding))
    else:
        if not args.capture:
            parser.error("score needs --capture")
        ling3_reference.score(args, Reference(args.checkpoint, args.conv_rounding))
    return 0


if __name__ == "__main__":
    sys.exit(main())
