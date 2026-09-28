#!/usr/bin/env python3
"""Greedy DeepSeek V4.1-Flash decode, token by token, through the SparkPipe dsv41 CUDA kernels.

Usage: dsv41_kernel_decode.py --checkpoint DIR --library libdsv41_kernels.so --reference reference_tokens.json --output OUT.json
"""
import argparse
import collections
import ctypes
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv41_official_harness as harness  # noqa: E402

P, U32, U64, F32 = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_float
SIGNATURES = {
    "dsv41_fp8_linear": (P, P, P, P, P, U32, U32, U32),
    "dsv41_mxfp4_linear": (P, P, P, P, P, U32, U32),
    "dsv41_bf16_linear": (P, P, P, P, U32, U32),
    "dsv41_grouped_fp8_linear": (P, P, P, P, U32, U32, U32),
    "dsv41_rmsnorm": (P, P, P, U32, U32, F32),
    "dsv41_rope": (P, P, U32, U32, U32, U32),
    "dsv41_fp8_qdq": (P, U64),
    "dsv41_kv_fp4_qdq": (P, U64, U32, U32),
    "dsv41_fp4_pow2_qdq": (P, U32, U32),
    "dsv41_sink_attention": (P, P, P, U32, P, P, U32, U32, F32),
    "dsv41_gate": (P, P, P, P, P, U32, U32, U32, F32),
    "dsv41_swiglu": (P, P, P, U32, F32, F32),
    "dsv41_hc_mixes": (P, P, P, P, P, P, P, P, U32, U32, U32, F32, F32),
    "dsv41_hc_pre": (P, P, P, U32, U32),
    "dsv41_hc_post": (P, P, P, P, P, U32, U32),
    "dsv41_compress_pool": (P, P, P, U32, U32),
    "dsv41_engram_gate": (P, P, P, P, U32, U32, U32, F32),
    "dsv41_index_score": (P, P, P, P, U32, U32, U32),
    "dsv41_candidate_mask": (P, P, U64, P, U64, U32, U32, U32),
}


class Kernels:
    def __init__(self, path):
        self.library = ctypes.CDLL(path)
        for name, types in SIGNATURES.items():
            getattr(self.library, name).argtypes = types
            getattr(self.library, name).restype = ctypes.c_int

    def __getattr__(self, name):
        function = getattr(self.library, "dsv41_" + name)

        def call(*arguments):
            converted = [a.data_ptr() if isinstance(a, torch.Tensor) else a for a in arguments]
            status = function(*converted)
            if status != 0:
                raise RuntimeError(f"dsv41_{name} returned cuda status {status}")
        return call


class Engine:
    def __init__(self, checkpoint_dir, library, device, expert_cache):
        self.device = device
        self.k = Kernels(library)
        self.ckpt = harness.Checkpoint(checkpoint_dir)
        inference_dir = os.path.join(checkpoint_dir, "inference")
        sys.path.insert(0, inference_dir)
        harness.install_kernel_module()
        import engram as publisher_engram
        import model as publisher
        from tokenizers import Tokenizer
        config = json.load(open(os.path.join(inference_dir, "config.json")))
        self.c = config
        fields = {k: v for k, v in config.items() if k in publisher.ModelArgs.__dataclass_fields__}
        fields.update(max_batch_size=1, max_seq_len=512, vision_n_layers=0, dspark_block_size=0, n_mtp_layers=0)
        args = publisher.ModelArgs(**{k: tuple(v) if isinstance(v, list) else v for k, v in fields.items()})

        class TokenizerView:
            def __init__(self, path):
                self.backend_tokenizer = Tokenizer.from_file(path)

            def __len__(self):
                return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

        layout = publisher_engram.EngramLayout.from_args(args)
        self.engram_hash = publisher_engram.NgramHashState(args, layout, TokenizerView(os.path.join(checkpoint_dir, "tokenizer.json")))
        self.engram_layers = list(layout.layer_ids)
        self.layers = config["n_layers"]
        self.ratios = config["compress_ratios"][:self.layers]
        self.kv_sources = set(config["kv_source_layers"])
        self.index_sources = set(config["index_source_layers"])
        self.candidate_source = config["candidate_source_layer"]
        self.dim, self.hc, self.heads, self.head_dim = config["dim"], config["hc_mult"], config["n_heads"], config["head_dim"]
        self.rope_dim, self.window = config["rope_head_dim"], config["window_size"]
        self.eps, self.hc_eps = float(config["norm_eps"]), float(config["hc_eps"])
        self.full_rope = torch.view_as_real(publisher.precompute_freqs_cis(
            self.rope_dim, 512, 0, config["rope_theta"], config["rope_factor"], config["beta_fast"], config["beta_slow"])).float().reshape(512, -1).to(device)
        self.compress_rope = torch.view_as_real(publisher.precompute_freqs_cis(
            self.rope_dim, 512, config["original_seq_len"], config["compress_rope_theta"], config["rope_factor"],
            config["beta_fast"], config["beta_slow"])).float().reshape(512, -1).to(device)
        self.w = {}
        started = time.time()
        for name in self.ckpt.st.map:
            if ".experts." in name and ".shared_experts." not in name:
                continue
            if name.startswith(("mtp.", "vision.", "aligner.", "image_")) or ".engram.embed." in name or name.endswith("bias_vl"):
                continue
            self.w[name] = self.ckpt.tensor(name, device)
        print(json.dumps({"loaded_tensors": len(self.w), "seconds": round(time.time() - started, 1)}), flush=True)
        self.experts = collections.OrderedDict()
        self.expert_cache = expert_cache
        self.scratch = torch.empty(65536, dtype=torch.float32, device=device)
        self.reset()

    def reset(self):
        self.ring = [torch.zeros(self.window, self.head_dim, dtype=torch.bfloat16, device=self.device) for _ in range(self.layers)]
        self.kv_state = {}
        self.score_state = {}
        self.compressed = collections.defaultdict(list)
        self.index_keys = collections.defaultdict(list)
        self.compress_owner = None
        self.index_owner = None
        self.topk_idxs = None
        self.candidates = None
        self.engram_hash.cache.zero_()

    def bf16(self, count):
        return torch.empty(count, dtype=torch.bfloat16, device=self.device)

    def fp8(self, prefix, x, rows_out):
        out = self.bf16(rows_out)
        self.k.fp8_linear(self.w[prefix + ".weight"], self.w[prefix + ".scale"], x, self.scratch, out, 1, x.numel(), rows_out)
        return out

    def norm(self, x, weight):
        out = self.bf16(x.numel())
        self.k.rmsnorm(x, self.w[weight], out, 1, x.numel(), self.eps)
        return out

    def expert(self, layer, index):
        key = (layer, index)
        if key in self.experts:
            self.experts.move_to_end(key)
            return self.experts[key]
        parts = {}
        for part in ("w1", "w2", "w3"):
            base = f"layers.{layer}.ffn.experts.{index}.{part}"
            parts[part] = (self.ckpt.tensor(base + ".weight", self.device), self.ckpt.tensor(base + ".scale", self.device))
        self.experts[key] = parts
        if len(self.experts) > self.expert_cache:
            self.experts.popitem(last=False)
        return parts

    def moe(self, layer, x):
        p = f"layers.{layer}.ffn."
        indices = torch.empty(6, dtype=torch.int32, device=self.device)
        weights = torch.empty(6, dtype=torch.float32, device=self.device)
        self.k.gate(x, self.w[p + "gate.weight"], self.w[p + "gate.bias"], indices, weights, self.c["n_routed_experts"],
                    self.dim, self.c["n_activated_experts"], float(self.c["route_scale"]))
        chosen = sorted(zip(indices.tolist(), weights.tolist()))
        inter = self.c["moe_inter_dim"]
        limit = float(self.c["swiglu_limit"])
        total = torch.zeros(self.dim, dtype=torch.float32, device=self.device)
        gate, up, hidden, down = self.bf16(inter), self.bf16(inter), self.bf16(inter), self.bf16(self.dim)
        for index, weight in chosen:
            parts = self.expert(layer, index)
            self.k.mxfp4_linear(parts["w1"][0], parts["w1"][1], x, self.scratch, gate, self.dim, inter)
            self.k.mxfp4_linear(parts["w3"][0], parts["w3"][1], x, self.scratch, up, self.dim, inter)
            self.k.swiglu(gate, up, hidden, inter, limit, weight)
            self.k.mxfp4_linear(parts["w2"][0], parts["w2"][1], hidden, self.scratch, down, inter, self.dim)
            total += down.float()
        shared = p + "shared_experts."
        self.k.fp8_linear(self.w[shared + "w1.weight"], self.w[shared + "w1.scale"], x, self.scratch, gate, 1, self.dim, inter)
        self.k.fp8_linear(self.w[shared + "w3.weight"], self.w[shared + "w3.scale"], x, self.scratch, up, 1, self.dim, inter)
        self.k.swiglu(gate, up, hidden, inter, limit, 1.0)
        self.k.fp8_linear(self.w[shared + "w2.weight"], self.w[shared + "w2.scale"], hidden, self.scratch, down, 1, inter, self.dim)
        total += down.float()
        return total.to(torch.bfloat16)

    def attention(self, layer, x, position):
        a = f"layers.{layer}.attn."
        ratio = self.ratios[layer]
        table = self.compress_rope if ratio else self.full_rope
        cos_sin = table[position]
        qr = self.norm(self.fp8(a + "wq_a", x, self.c["q_lora_rank"]), a + "q_norm.weight")
        q = self.fp8(a + "wq_b", qr, self.heads * self.head_dim)
        self.k.rope(q, cos_sin, self.heads, self.head_dim, self.rope_dim, 0)
        kv = self.norm(self.fp8(a + "wkv", x, self.head_dim), a + "kv_norm.weight")
        self.k.rope(kv, cos_sin, 1, self.head_dim, self.rope_dim, 0)
        self.k.fp8_qdq(kv, self.head_dim)
        self.ring[layer][position % self.window] = kv
        oldest = position % self.window + 1
        slots = [(oldest + c) % self.window if (oldest + c) < self.window else oldest + c - self.window for c in range(self.window)]
        slots = [s if s <= position else -1 for s in slots]
        keys = self.ring[layer]
        indices = torch.tensor(slots, dtype=torch.int32, device=self.device)
        if ratio:
            compress_len = (position + 1) // ratio
            latent = None
            if layer in self.kv_sources:
                cp = a + "compressor."
                if ratio == 1:
                    raw = self.bf16(self.head_dim)
                    self.k.bf16_linear(self.w[cp + "wkv.weight"], x, 0, raw, self.dim, self.head_dim)
                    latent = self.norm(raw, cp + "norm.weight")
                else:
                    if layer not in self.kv_state:
                        self.kv_state[layer] = torch.zeros(ratio, self.head_dim, dtype=torch.float32, device=self.device)
                        self.score_state[layer] = torch.zeros(ratio, self.head_dim, dtype=torch.float32, device=self.device)
                    slot = position % ratio
                    self.k.bf16_linear(self.w[cp + "wkv.weight"], x, self.kv_state[layer][slot], 0, self.dim, self.head_dim)
                    self.k.bf16_linear(self.w[cp + "wgate.weight"], x, self.score_state[layer][slot], 0, self.dim, self.head_dim)
                    if (position + 1) % ratio == 0:
                        pooled = self.bf16(self.head_dim)
                        self.k.compress_pool(self.kv_state[layer], self.score_state[layer], pooled, ratio, self.head_dim)
                        latent = self.norm(pooled, cp + "norm.weight")
                self.compress_owner = layer
            if layer in self.index_sources:
                ip = a + "indexer."
                index_dim, index_heads = self.c["index_head_dim"], self.c["index_n_heads"]
                if layer in self.kv_sources and latent is not None:
                    key = self.bf16(index_dim)
                    self.k.bf16_linear(self.w[ip + "wk.weight"], latent, 0, key, self.head_dim, index_dim)
                    key = self.norm(key, ip + "k_norm.weight")
                    self.k.rope(key, table[position + 1 - ratio], 1, index_dim, self.rope_dim, 0)
                    self.k.fp4_pow2_qdq(key, index_dim // 32, 32)
                    self.index_keys[layer].append(key)
                    self.index_owner = layer
                if compress_len == 0:
                    idxs = torch.empty(0, dtype=torch.int32, device=self.device)
                else:
                    iq = self.fp8(ip + "wq_b", qr, index_heads * index_dim)
                    self.k.rope(iq, cos_sin, index_heads, index_dim, self.rope_dim, 0)
                    self.k.fp4_pow2_qdq(iq, index_heads * index_dim // 32, 32)
                    head_weights = self.bf16(index_heads)
                    self.k.bf16_linear(self.w[ip + "weights_proj.weight"], x, 0, head_weights, self.dim, index_heads)
                    head_weights = (head_weights * (index_dim ** -0.5 * index_heads ** -0.5)).to(torch.bfloat16)
                    owned = torch.stack(self.index_keys[self.index_owner][:compress_len])
                    scores = torch.empty(compress_len, dtype=torch.float32, device=self.device)
                    self.k.index_score(iq, owned, head_weights, scores, compress_len, index_heads, index_dim)
                    if layer == self.candidate_source:
                        masked = scores.clone()
                        widths = torch.tensor([compress_len], dtype=torch.int32, device=self.device)
                        block = self.c["candidate_block_size"]
                        blocks = torch.empty((compress_len + block - 1) // block, dtype=torch.float32, device=self.device)
                        self.k.candidate_mask(masked, widths, compress_len, blocks, blocks.numel(), 1, block, self.c["candidate_topk_blocks"])
                        self.candidates = masked > float("-inf")
                    elif 0 <= self.candidate_source < layer:
                        scores = torch.where(self.candidates[:compress_len], scores, torch.full_like(scores, float("-inf")))
                    topk = min(self.c["index_topk"], compress_len)
                    chosen = torch.topk(scores, topk).indices.sort().values
                    idxs = torch.where(chosen < compress_len, chosen + self.window, torch.full_like(chosen, -1)).int()
                self.topk_idxs = idxs
            idxs = self.topk_idxs
            if latent is not None:
                self.k.rope(latent, table[position + 1 - ratio], 1, self.head_dim, self.rope_dim, 0)
                self.k.kv_fp4_qdq(latent, self.head_dim, 1, self.head_dim)
                self.compressed[layer].append(latent)
            rows = self.compressed[self.compress_owner][:compress_len]
            if rows:
                keys = torch.cat([keys, torch.stack(rows)])
                indices = torch.cat([indices, idxs])
        attended = self.bf16(self.heads * self.head_dim)
        self.k.sink_attention(q, keys.contiguous(), indices.contiguous(), indices.numel(), self.w[a + "attn_sink"], attended,
                              self.heads, self.head_dim, self.head_dim ** -0.5)
        self.k.rope(attended, cos_sin, self.heads, self.head_dim, self.rope_dim, 1)
        groups, o_lora = self.c["o_groups"], self.c["o_lora_rank"]
        grouped = self.bf16(groups * o_lora)
        self.k.grouped_fp8_linear(self.w[a + "wo_a.weight"], self.w[a + "wo_a.scale"], attended, grouped,
                                  self.heads * self.head_dim // groups, groups * o_lora, o_lora)
        return self.fp8(a + "wo_b", grouped, self.dim)

    def mixes(self, layer, kind, streams):
        p = f"layers.{layer}.hc_{kind}_"
        mix = torch.empty((2 + self.hc) * self.hc, dtype=torch.float32, device=self.device)
        pre = torch.empty(self.hc, dtype=torch.float32, device=self.device)
        post = torch.empty(self.hc, dtype=torch.float32, device=self.device)
        comb = torch.empty(self.hc * self.hc, dtype=torch.float32, device=self.device)
        self.k.hc_mixes(streams, self.w[p + "fn"], self.w[p + "scale"], self.w[p + "base"], mix, pre, post, comb,
                        self.hc, self.dim, self.c["hc_sinkhorn_iters"], self.eps, self.hc_eps)
        return pre, post, comb

    def collapse(self, streams, pre):
        out = self.bf16(self.dim)
        self.k.hc_pre(streams, pre, out, self.hc, self.dim)
        return out

    def expand(self, sublayer, residual, post, comb):
        out = self.bf16(self.hc * self.dim)
        self.k.hc_post(sublayer, residual, post, comb, out, self.hc, self.dim)
        return out

    def engram(self, layer, streams, hash_row):
        p = f"layers.{layer}.engram."
        rows = []
        for row in hash_row.tolist():
            weight = self.ckpt.rows(p + "embed.weight", int(row), 1, self.device).view(torch.float8_e4m3fn).float()
            scale = self.ckpt.rows(p + "embed.scale", int(row), 1, self.device).view(torch.uint8).view(torch.float8_e8m0fnu).float()
            rows.append((weight.view(-1, 32) * scale.view(-1, 1)).reshape(-1))
        embedded = torch.cat(rows).to(torch.bfloat16)
        kv = self.fp8(p + "wkv", embedded, (self.hc + 1) * self.dim)
        self.k.engram_gate(streams, kv, self.w[p + "q_weight"], self.w[p + "k_weight"], 1, self.hc, self.dim, self.eps)
        return streams

    def step(self, token, position):
        hashes = self.engram_hash(torch.tensor([[token]]), position)
        streams = self.w["embed.weight"][token].repeat(self.hc).contiguous()
        pre_mix = torch.zeros(self.hc, dtype=torch.float32, device=self.device)
        pre_mix[0] = 1.0
        for layer in range(self.layers):
            if layer in self.engram_layers:
                streams = self.engram(layer, streams, hashes[0, 0, self.engram_layers.index(layer)])
            attn_pre, attn_post, attn_comb = self.mixes(layer, "attn", streams)
            x = self.norm(self.collapse(streams, pre_mix), f"layers.{layer}.attn_norm.weight")
            middle = self.expand(self.attention(layer, x, position), streams, attn_post, attn_comb)
            ffn_pre, ffn_post, ffn_comb = self.mixes(layer, "ffn", middle)
            x = self.norm(self.collapse(middle, attn_pre), f"layers.{layer}.ffn_norm.weight")
            streams = self.expand(self.moe(layer, x), middle, ffn_post, ffn_comb)
            pre_mix = ffn_pre
        x = self.norm(self.collapse(streams, pre_mix), "norm.weight")
        logits = torch.empty(self.c["vocab_size"], dtype=torch.float32, device=self.device)
        self.k.bf16_linear(self.w["head.weight"], x, logits, 0, self.dim, self.c["vocab_size"])
        top = torch.topk(logits, 2)
        return int(top.indices[0]), float(top.values[0] - top.values[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    parser.add_argument("--expert-cache", type=int, default=200)
    arguments = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(arguments.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    engine = Engine(arguments.checkpoint, arguments.library, arguments.device, arguments.expert_cache)
    reference = json.load(open(arguments.reference))
    results = []
    with torch.inference_mode():
        for prompt in reference["prompts"]:
            engine.reset()
            ids = list(prompt["prompt_token_ids"])
            want = prompt["generated_token_ids"]
            generated, margins = [], []
            started = time.time()
            for position, token in enumerate(ids):
                predicted, margin = engine.step(token, position)
            for step in range(len(want)):
                generated.append(predicted)
                margins.append(round(margin, 4))
                print(json.dumps({"prompt": prompt["name"], "step": step, "token": predicted, "reference": want[step],
                                  "margin": round(margin, 4), "seconds": round(time.time() - started, 1)}), flush=True)
                if step + 1 < len(want):
                    predicted, margin = engine.step(predicted, len(ids) + step)
            diverge = next((i for i, (a, b) in enumerate(zip(generated, want)) if a != b), None)
            results.append({"name": prompt["name"], "generated_token_ids": generated, "reference_token_ids": want,
                            "first_divergence": diverge, "top1_margins": margins})
    json.dump({"generator": "tools/dsv41_kernel_decode.py", "prompts": results}, open(arguments.output, "w"), indent=1)
    exact = sum(1 for r in results if r["first_divergence"] is None)
    print(json.dumps({"token_exact": f"{exact}/{len(results)}"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
