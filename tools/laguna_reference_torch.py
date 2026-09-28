import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


class Checkpoint:
    def __init__(self, root):
        self.root = root
        index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        headers = {}
        for shard in sorted(set(index.values())):
            path = os.path.join(root, shard)
            with open(path, "rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                headers[shard] = (path, 8 + size, json.loads(handle.read(size)))
        self.where = {}
        for name, shard in index.items():
            path, base, header = headers[shard]
            entry = header[name]
            self.where[name] = (path, base + entry["data_offsets"][0], base + entry["data_offsets"][1], entry["dtype"], entry["shape"])

    def has(self, name):
        return name in self.where

    def host(self, name):
        path, start, end, dtype, shape = self.where[name]
        if dtype != "BF16":
            raise ValueError(f"{name} is {dtype}; the reference reads BF16 checkpoints only")
        with open(path, "rb") as handle:
            handle.seek(start)
            raw = handle.read(end - start)
        if len(raw) != end - start:
            raise IOError(f"short read of {name}")
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape)

    def rows(self, name, row_ids):
        path, start, end, dtype, shape = self.where[name]
        if dtype != "BF16" or len(shape) != 2:
            raise ValueError(f"{name} must be a BF16 matrix")
        stride = shape[1] * 2
        out = []
        with open(path, "rb") as handle:
            for row in row_ids:
                if row < 0 or row >= shape[0]:
                    raise ValueError(f"row {row} outside {name}")
                handle.seek(start + row * stride)
                out.append(torch.frombuffer(bytearray(handle.read(stride)), dtype=torch.bfloat16))
        return torch.stack(out)


def rms_norm(x, weight, eps):
    dtype = x.dtype
    y = x.to(torch.float32)
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return weight * y.to(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x, cos, sin):
    width = cos.shape[-1]
    rotated = x[..., :width] * cos + rotate_half(x[..., :width]) * sin
    return torch.cat([rotated, x[..., width:]], dim=-1)


def attention_allowed(query_positions, key_positions, window):
    allowed = key_positions[None, :] <= query_positions[:, None]
    if window is not None:
        allowed = allowed & (key_positions[None, :] > query_positions[:, None] - window)
    return allowed


def default_inv_freq(parameters, head_dim):
    dim = int(head_dim * parameters.get("partial_rotary_factor", 1.0))
    base = float(parameters["rope_theta"])
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(torch.float32) / dim)), 1.0


def yarn_inv_freq(parameters, head_dim):
    dim = int(head_dim * parameters.get("partial_rotary_factor", 1.0))
    base = float(parameters["rope_theta"])
    factor = float(parameters["factor"])
    original = float(parameters["original_max_position_embeddings"])
    beta_fast = float(parameters["beta_fast"])
    beta_slow = float(parameters["beta_slow"])
    attention_factor = float(parameters["attention_factor"])
    if parameters.get("truncate", True) is not True:
        raise ValueError("yarn truncate=false is not implemented by this reference")

    def correction_dim(rotations):
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    positions = base ** (torch.arange(0, dim, 2).to(dtype=torch.float32) / dim)
    extrapolation = 1.0 / positions
    interpolation = 1.0 / (factor * positions)
    keep = 1 - ramp
    return interpolation * (1 - keep) + extrapolation * keep, attention_factor


class Rope:
    def __init__(self, parameters, head_dim, device):
        kind = parameters.get("rope_type", "default")
        if kind == "default":
            inv_freq, scaling = default_inv_freq(parameters, head_dim)
        elif kind == "yarn":
            inv_freq, scaling = yarn_inv_freq(parameters, head_dim)
        else:
            raise ValueError(f"rope type {kind} is not implemented by this reference")
        self.inv_freq = inv_freq.to(device)
        self.scaling = scaling

    def __call__(self, positions, dtype):
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.scaling).to(dtype), (emb.sin() * self.scaling).to(dtype)


class Laguna:
    def __init__(self, root, device, expert_cache):
        self.config = json.load(open(os.path.join(root, "config.json")))
        c = self.config
        if c.get("model_type") != "laguna":
            raise ValueError("checkpoint is not a laguna model")
        if c.get("gating") != "per-head":
            raise ValueError("only per-head attention gating is implemented")
        if c.get("swa_attention_sink_enabled", False):
            raise ValueError("attention sinks are not implemented")
        if c.get("moe_apply_router_weight_on_input", False):
            raise ValueError("router weight on input is not implemented")
        if float(c.get("moe_router_logit_softcapping", 0.0) or 0.0) != 0.0:
            raise ValueError("router logit softcapping is not implemented")
        if c.get("decoder_sparse_step", 1) != 1:
            raise ValueError("decoder_sparse_step other than 1 is not implemented")
        if c.get("hidden_act", "silu") != "silu":
            raise ValueError("only silu is implemented")
        self.device = device
        self.ckpt = Checkpoint(root)
        self.eps = float(c["rms_norm_eps"])
        self.head_dim = int(c["head_dim"])
        self.kv_heads = int(c["num_key_value_heads"])
        self.layers = int(c["num_hidden_layers"])
        self.heads = list(c["num_attention_heads_per_layer"])
        self.types = list(c["layer_types"])
        self.window = int(c["sliding_window"])
        self.top_k = int(c["num_experts_per_tok"])
        self.norm_topk = bool(c["norm_topk_prob"])
        self.scaling = float(c.get("moe_routed_scaling_factor", 1.0))
        self.mlp_only = set(int(i) for i in c["mlp_only_layers"])
        rope = c["rope_parameters"]
        swa = c.get("swa_rope_parameters") or rope.get("sliding_attention")
        if swa is None:
            raise ValueError("sliding rope parameters missing")
        self.rope = {"full_attention": Rope(dict(rope["full_attention"]), self.head_dim, device), "sliding_attention": Rope(dict(swa), self.head_dim, device)}
        self.resident = {}
        self.expert_cache = collections.OrderedDict()
        self.expert_cache_limit = expert_cache
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        self.bytes_read = 0

    def weight(self, name):
        if name not in self.resident:
            tensor = self.ckpt.host(name)
            self.bytes_read += tensor.numel() * 2
            self.resident[name] = tensor.to(self.device)
        return self.resident[name]

    def correction_bias(self, layer):
        for name in (f"model.layers.{layer}.mlp.gate.e_score_correction_bias", f"model.layers.{layer}.mlp.experts.e_score_correction_bias"):
            if self.ckpt.has(name):
                return self.weight(name)
        raise ValueError(f"layer {layer} has no router correction bias")

    def expert(self, layer, index):
        key = (layer, index)
        if key in self.expert_cache:
            self.expert_cache.move_to_end(key)
            return self.expert_cache[key]
        raise KeyError(key)

    def fetch_experts(self, layer, indices):
        missing = [int(i) for i in indices if (layer, int(i)) not in self.expert_cache]
        prefix = f"model.layers.{layer}.mlp.experts."

        def load(index):
            gate = self.ckpt.host(f"{prefix}{index}.gate_proj.weight")
            up = self.ckpt.host(f"{prefix}{index}.up_proj.weight")
            down = self.ckpt.host(f"{prefix}{index}.down_proj.weight")
            return index, torch.cat([gate, up], dim=0), down

        for index, gate_up, down in self.pool.map(load, missing):
            self.bytes_read += (gate_up.numel() + down.numel()) * 2
            self.expert_cache[(layer, index)] = (gate_up.to(self.device), down.to(self.device))
            while len(self.expert_cache) > self.expert_cache_limit:
                self.expert_cache.popitem(last=False)

    def mlp(self, prefix, x):
        gate = F.linear(x, self.weight(prefix + "gate_proj.weight"))
        up = F.linear(x, self.weight(prefix + "up_proj.weight"))
        return F.linear(F.silu(gate) * up, self.weight(prefix + "down_proj.weight"))

    def moe(self, layer, x, trace):
        prefix = f"model.layers.{layer}.mlp."
        shared = self.mlp(prefix + "shared_expert.", x)
        logits = F.linear(x, self.weight(prefix + "gate.weight")).float()
        scores = torch.sigmoid(logits)
        selection = scores + self.correction_bias(layer).to(scores.dtype)
        _, selected = torch.topk(selection, self.top_k, dim=-1)
        weights = scores.gather(-1, selected)
        if self.norm_topk:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(x.dtype)
        trace.append({"layer": layer, "experts": selected[-1].tolist(), "weights": weights[-1].float().tolist()})
        hit = sorted(set(selected.flatten().tolist()))
        self.fetch_experts(layer, hit)
        out = torch.zeros_like(x)
        for index in hit:
            gate_up, down = self.expert(layer, index)
            rank, token = torch.where(selected.t() == index)
            current = x[token]
            gate, up = F.linear(current, gate_up).chunk(2, dim=-1)
            current = F.linear(F.silu(gate) * up, down)
            current = current * weights[token, rank, None]
            out.index_add_(0, token, current.to(out.dtype))
        if self.scaling != 1.0:
            out = out * self.scaling
        return out + shared

    def attention(self, layer, x, positions, cache):
        prefix = f"model.layers.{layer}.self_attn."
        heads = self.heads[layer]
        rows = x.shape[0]
        q = F.linear(x, self.weight(prefix + "q_proj.weight")).view(rows, heads, self.head_dim).transpose(0, 1)
        k = F.linear(x, self.weight(prefix + "k_proj.weight")).view(rows, self.kv_heads, self.head_dim).transpose(0, 1)
        v = F.linear(x, self.weight(prefix + "v_proj.weight")).view(rows, self.kv_heads, self.head_dim).transpose(0, 1)
        q = rms_norm(q, self.weight(prefix + "q_norm.weight"), self.eps)
        k = rms_norm(k, self.weight(prefix + "k_norm.weight"), self.eps)
        cos, sin = self.rope[self.types[layer]](positions, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache[layer] is not None:
            k = torch.cat([cache[layer][0], k], dim=1)
            v = torch.cat([cache[layer][1], v], dim=1)
        cache[layer] = (k, v)
        total = k.shape[1]
        window = self.window if self.types[layer] == "sliding_attention" else None
        allowed = attention_allowed(positions, torch.arange(total, device=x.device), window)
        mask = torch.zeros(allowed.shape, dtype=x.dtype, device=x.device).masked_fill(~allowed, torch.finfo(x.dtype).min)
        group = heads // self.kv_heads
        k = k.repeat_interleave(group, dim=0)
        v = v.repeat_interleave(group, dim=0)
        out = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=mask[None, None], scale=self.head_dim ** -0.5)[0]
        out = out.transpose(0, 1).reshape(rows, heads * self.head_dim)
        gate = F.softplus(F.linear(x, self.weight(prefix + "g_proj.weight")).float()).to(out.dtype)
        out = (out.view(rows, heads, self.head_dim) * gate.unsqueeze(-1)).view(rows, heads * self.head_dim)
        return F.linear(out, self.weight(prefix + "o_proj.weight"))

    def forward(self, token_ids, positions, cache, trace, dumps):
        h = self.ckpt.rows("model.embed_tokens.weight", token_ids).to(self.device)
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}."
            residual = h
            x = rms_norm(h, self.weight(prefix + "input_layernorm.weight"), self.eps)
            h = residual + self.attention(layer, x, positions, cache)
            residual = h
            x = rms_norm(h, self.weight(prefix + "post_attention_layernorm.weight"), self.eps)
            if layer in self.mlp_only:
                h = residual + self.mlp(prefix + "mlp.", x)
            else:
                h = residual + self.moe(layer, x, trace)
            dumps.append(h[-1].float().cpu().numpy())
        h = rms_norm(h[-1:], self.weight("model.norm.weight"), self.eps)
        logits = F.linear(h, self.weight("lm_head.weight"))[0]
        exact = F.linear(h.float(), self.weight("lm_head.weight").float())[0]
        return logits.to(torch.float32), exact

    def generate(self, prompt, count, stop):
        cache = [None] * self.layers
        tokens = list(prompt)
        steps = []
        inputs = list(prompt)
        start = 0
        for step in range(count):
            positions = torch.arange(start, start + len(inputs), device=self.device)
            trace = []
            dumps = []
            began = time.time()
            logits, exact = self.forward(inputs, positions, cache, trace, dumps)
            token = int(torch.argmax(logits).item())
            top = torch.topk(exact, 2)
            steps.append({"position": start + len(inputs) - 1, "token": token, "bf16_logit": float(logits[token].item()), "fp32_top2_tokens": top.indices.tolist(), "fp32_top2_logits": top.values.tolist(), "seconds": round(time.time() - began, 2), "routes": trace, "layer_last_row": np.stack(dumps)})
            start += len(inputs)
            tokens.append(token)
            inputs = [token]
            print(json.dumps({"step": step, "token": token, "fp32_top2": top.indices.tolist(), "seconds": steps[-1]["seconds"], "read_gib": round(self.bytes_read / 2 ** 30, 1)}), flush=True)
            if token in stop:
                break
        return tokens[len(prompt):], steps


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--expert-cache", type=int, required=True)
    arguments = parser.parse_args()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    prompts = json.load(open(arguments.prompts))
    model = Laguna(arguments.checkpoint, torch.device(arguments.device), arguments.expert_cache)
    eos = model.config["eos_token_id"]
    stop = set(eos if isinstance(eos, list) else [eos])
    os.makedirs(arguments.output, exist_ok=True)
    results = []
    with torch.inference_mode():
        for spec in prompts["prompts"]:
            generated, steps = model.generate([int(t) for t in spec["prompt_token_ids"]], int(spec["new_tokens"]), stop)
            arrays = {f"step{i:02d}_layer_last_row": s.pop("layer_last_row") for i, s in enumerate(steps)}
            np.savez_compressed(os.path.join(arguments.output, spec["name"] + ".hidden.npz"), **arrays)
            results.append({"name": spec["name"], "prompt_token_ids": spec["prompt_token_ids"], "generated_token_ids": generated, "steps": steps})
    document = {"generator": "tools/laguna_reference_torch.py", "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if arguments.device.startswith("cuda") else "cpu", "checkpoint": os.path.abspath(arguments.checkpoint), "config_sha256": sha256_file(os.path.join(arguments.checkpoint, "config.json")), "index_sha256": sha256_file(os.path.join(arguments.checkpoint, "model.safetensors.index.json")), "prompts_sha256": sha256_file(arguments.prompts), "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": results}
    with open(os.path.join(arguments.output, "reference.json"), "w") as handle:
        json.dump(document, handle, indent=1)
    print(json.dumps({"output": arguments.output, "generated": {r["name"]: r["generated_token_ids"] for r in results}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
