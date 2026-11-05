#!/usr/bin/env python3
"""Teacher-forced numpy reference for Qwen 3.8 Max (NVFP4 routed experts, BF16 spine).

  gate-prompts  the Metal gate prompts (tools/gemma4_reference.py's set) in
             the Qwen 3.8 chat turn with an empty think block
             (model-families/qwen38/chat_template.json), encoded with the
             checkpoint's tokenizer.json -> --out prompts.json
  teacher    FP32 logits for every position but the last of each prompt
             (--prompts, qualification/metal/qwen38_max/gate_prompts.json)
             to <out>/<name>.ref_<variant>.logits for each of --variants: a
             sums in numpy's order, b splits every matmul's inner dimension
             into 1,024-wide chunks summed last chunk first and reverses
             every other reduction and the expert combination, so a against
             b is the noise of FP32 reduction order, and h is b with a GPU's
             exp and tanh error (tools/reference_device_error.py): the three
             variants' distances from one another are the Metal gate's floors

The semantics are SparkPipe's CUDA stage
(model-families/qwen38/source/qwen38_hybrid_stage.cuh) on Max's geometry. The
gated delta rule, the attention and the experts are
tools/qwen38_flash_reference.py's, which reads them for Flash; R() is a round
to bf16.
- the residual stream as tools/qwen38_27b_reference.c: h = R(h + out) after
  each branch, every norm R(h * rsqrt(mean(h^2) + 1e-6) * (1 + w)) of the
  rounded residual, and the final norm before the head
- GDN: 16 key and 128 value heads of 128, each value head reading key head
  v / 8; the gated norm R(R(R(o r) w) silu(z)), the SiLU output gate where
  Flash has the sigmoid
- attention: 64 query and 4 KV heads of 256, rotary on the first 64
  dimensions with theta 1e7, q and k norms, the sigmoid output gate
- MoE: router logits R(x @ router), FP32 softmax, top 10 of 512, weights
  p / (sum + 1e-20); each route R(R(silu(gate) x up x weight) @ down) with
  the NVFP4 experts dequantised exactly to e2m1 x ue4m3 x weight_scale_2 (the
  checkpoint's input_scale, an activation scale for W4A4, is not applied: CUDA
  and Metal multiply BF16 activations); the routes in pick order plus the
  shared expert R(R(R(silu(gate) x up) @ down) x sigmoid(R(x . shared_gate))),
  summed in FP32 and R'd
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen38_flash_reference as flash


class Reference(flash.Reference):
    def __init__(self, directory, threads):
        config = json.loads((Path(directory) / "config.json").read_text())
        rope = config["rope_parameters"]
        self.checkpoint = flash.Checkpoint(directory)
        self.threads = threads
        self.prefix = "model."
        self.output_gate = flash.silu
        self.hidden = config["hidden_size"]
        self.layers = config["num_hidden_layers"]
        self.key_heads, self.value_heads = config["linear_num_key_heads"], config["linear_num_value_heads"]
        self.key_dim, self.value_dim = config["linear_key_head_dim"], config["linear_value_head_dim"]
        self.heads, self.kv_heads, self.head_dim = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
        self.theta = rope["rope_theta"]
        self.rope_dim = int(self.head_dim * rope["partial_rotary_factor"])
        self.period = config["full_attention_interval"]
        self.top_k = config["num_experts_per_tok"]
        if config.get("output_gate_type") != "swish" or not config.get("attn_output_gate") or config.get("rms_norm_eps") != 1e-6 or \
                self.value_heads % self.key_heads != 0 or self.rope_dim % 2 != 0 or config.get("tie_word_embeddings"):
            raise ValueError("config is outside what this reference implements")

    def norm(self, values, name):
        mean = flash.reduce_sum(values * values, keepdims=True) / np.float32(self.hidden)
        scale = (1.0 / np.sqrt((mean + flash.EPSILON).astype(np.float64))).astype(np.float32)
        return flash.bf16(values * scale * self.checkpoint.gain(name))

    def forward(self, sequences):
        spans, first = [], 0
        for tokens in sequences:
            spans.append((first, len(tokens)))
            first += len(tokens)
        embed = self.checkpoint.raw(self.prefix + "embed_tokens.weight", "BF16")
        flat = np.concatenate([np.asarray(tokens, dtype=np.int64) for tokens in sequences])
        hidden = (np.asarray(embed[flat]).astype(np.uint32) << 16).view(np.float32)
        for layer in range(self.layers):
            started = time.time()
            p = f"{self.prefix}layers.{layer}."
            normed = self.norm(hidden, p + "input_layernorm.weight")
            out = self.attention(layer, normed, spans) if self.full(layer) else self.gdn(layer, normed, spans)
            hidden = flash.bf16(hidden + out)
            hidden = flash.bf16(hidden + self.moe(layer, self.norm(hidden, p + "post_attention_layernorm.weight")))
            print(f"layer {layer} {'attention' if self.full(layer) else 'gdn'} done in {time.time() - started:.1f}s", file=sys.stderr, flush=True)
        return self.norm(hidden, self.prefix + "norm.weight"), spans


def gate_prompts(args):
    import gemma4_reference
    gemma4_reference.gate_prompts(args, ("prefix", "user"), ("turn_suffix", "generation"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("gate-prompts", "teacher"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--variants", default="a,b,h")
    parser.add_argument("--only", help="comma-separated prompt names (default every prompt)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--template", default=str(Path(__file__).resolve().parents[1] / "model-families/qwen38/chat_template.json"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "gate-prompts":
        gate_prompts(args)
    elif not args.prompts:
        parser.error("teacher needs --prompts")
    else:
        flash.teacher(args, Reference(args.checkpoint, args.threads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
