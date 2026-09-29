import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F


def load_bf16(path, device):
    with open(path, "rb") as fh:
        size = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(size))
    header.pop("__metadata__", None)
    base = 8 + size
    tensors = {}
    for name, entry in header.items():
        if entry["dtype"] != "BF16":
            raise ValueError(f"{name}: {entry['dtype']} is not BF16")
        start, end = entry["data_offsets"]
        raw = np.memmap(path, dtype=np.uint16, mode="r", offset=base + start, shape=tuple(entry["shape"]))
        tensors[name] = torch.from_numpy(np.array(raw)).view(torch.bfloat16).to(device)
    return tensors


def rms(x, weight, eps):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


class BlockDrafter:
    def __init__(self, directory, device, vocab, context):
        self.config = json.load(open(os.path.join(directory, "config.json")))
        c = self.config
        self.w = load_bf16(os.path.join(directory, "model.safetensors"), device)
        self.hidden = int(c["hidden_size"])
        self.heads = int(c["num_attention_heads"])
        self.kv_heads = int(c["num_key_value_heads"])
        self.head_dim = int(c["head_dim"])
        self.layers = int(c["num_hidden_layers"])
        self.eps = float(c["rms_norm_eps"])
        self.block = int((c.get("dflash_config") or {}).get("block_size") or c.get("block_size"))
        generator = torch.Generator(device=device).manual_seed(1)
        self.lm_head = (torch.randn((vocab, self.hidden), generator=generator, device=device) * 0.02).to(torch.bfloat16)
        self.embed = (torch.randn((vocab, self.hidden), generator=generator, device=device) * 0.02).to(torch.bfloat16)
        self.k_cache = torch.randn((self.layers, self.kv_heads, context, self.head_dim), device=device).to(torch.bfloat16)
        self.v_cache = torch.randn_like(self.k_cache)
        self.context = context
        taps = self.w["fc.weight"].shape[1]
        self.taps_in = torch.randn((1, taps), device=device).to(torch.bfloat16)
        self.tokens = torch.zeros(self.block, dtype=torch.int64, device=device)

    def device_bytes(self):
        total = sum(t.numel() * t.element_size() for t in self.w.values())
        return total + self.lm_head.numel() * 2 + self.k_cache.numel() * 4

    def round(self, committed):
        w = self.w
        ctx = rms(self.taps_in @ w["fc.weight"].T, w["hidden_norm.weight"], self.eps)
        h = self.embed[self.tokens]
        for layer in range(self.layers):
            p = f"layers.{layer}."
            x = rms(h, w[p + "input_layernorm.weight"], self.eps)
            q = (x @ w[p + "self_attn.q_proj.weight"].T).view(self.block, self.heads, self.head_dim)
            k_new = torch.cat([ctx, x]) @ w[p + "self_attn.k_proj.weight"].T
            v_new = torch.cat([ctx, x]) @ w[p + "self_attn.v_proj.weight"].T
            q = rms(q, w[p + "self_attn.q_norm.weight"], self.eps).transpose(0, 1)
            k_new = rms(k_new.view(-1, self.kv_heads, self.head_dim), w[p + "self_attn.k_norm.weight"], self.eps)
            k = torch.cat([self.k_cache[layer][:, :committed], k_new.transpose(0, 1)], dim=1)
            v = torch.cat([self.v_cache[layer][:, :committed], v_new.view(-1, self.kv_heads, self.head_dim).transpose(0, 1)], dim=1)
            attn = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), enable_gqa=True)
            h = h + attn[0].transpose(0, 1).reshape(self.block, -1) @ w[p + "self_attn.o_proj.weight"].T
            x = rms(h, w[p + "post_attention_layernorm.weight"], self.eps)
            h = h + (F.silu(x @ w[p + "mlp.gate_proj.weight"].T) * (x @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T
        logits = rms(h[1:], w["norm.weight"], self.eps) @ self.lm_head.T
        return torch.argmax(logits, dim=-1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="latency of a DFlash-style block drafter on this GPU (synthetic taps and head)")
    parser.add_argument("--drafter", required=True)
    parser.add_argument("--vocab", type=int, required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--read-peak-gbs", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    drafter = BlockDrafter(args.drafter, "cuda", args.vocab, args.context)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            drafter.round(args.context - 64)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        drafts = drafter.round(args.context - 64)
    times = []
    for _ in range(args.repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    first, second = drafts.clone(), None
    graph.replay()
    second = drafts.clone()
    weight_bytes = sum(t.numel() * 2 for n, t in drafter.w.items()) + drafter.lm_head.numel() * 2
    kv_bytes = drafter.k_cache[0, :, :args.context].numel() * 2 * 2 * drafter.layers
    p50 = float(np.percentile(times, 50))
    report = {"drafter": args.drafter, "block": drafter.block, "round_p50_ms": p50,
              "round_p99_ms": float(np.percentile(times, 99)), "draft_tokens_per_round": drafter.block - 1,
              "weight_bytes": weight_bytes, "kv_bytes": kv_bytes, "device_bytes": drafter.device_bytes(),
              "memory_pct": 100.0 * (weight_bytes + kv_bytes) / (p50 / 1000.0) / (args.read_peak_gbs * 1e9),
              "replay_identical": bool(torch.equal(first, second)), "max_memory_allocated": torch.cuda.max_memory_allocated()}
    json.dump(report, open(args.output, "w"), indent=1)
    print(f"DRAFTD-BLOCK block={drafter.block} round_p50={p50:.3f}ms p99={report['round_p99_ms']:.3f}ms "
          f"bytes={weight_bytes + kv_bytes} memory={report['memory_pct']:.1f}% device_bytes={report['device_bytes']} "
          f"replay_identical={report['replay_identical']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
