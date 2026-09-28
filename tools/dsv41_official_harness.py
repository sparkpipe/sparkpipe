#!/usr/bin/env python3
"""Greedy DeepSeek V4.1-Flash decode through the publisher inference/model.py with torch kernels.

Usage: dsv41_official_harness.py --checkpoint DIR --prompts JSON --new-tokens N --output OUT.json
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import Safetensors  # noqa: E402

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def pow2_ceil(t):
    mantissa, exponent = torch.frexp(t.float())
    return torch.ldexp(torch.ones_like(t, dtype=torch.float32),
                       (exponent - (mantissa == 0.5).int()).int())


def round_e2m1(x):
    grid = E2M1_GRID.to(x.device)
    magnitude = x.abs().clamp(max=6.0)
    hi = torch.bucketize(magnitude, grid).clamp(max=7)
    lo = (hi - 1).clamp(min=0)
    d_lo = magnitude - grid[lo]
    d_hi = grid[hi] - magnitude
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    return torch.where(pick_hi, grid[hi], grid[lo]) * torch.sign(x)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    shape = x.shape
    n = shape[-1]
    assert n % block_size == 0
    blocks = x.float().reshape(-1, n // block_size, block_size)
    amax = blocks.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = pow2_ceil(amax / 448.0) if scale_fmt is not None else amax / 448.0
    q = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    if inplace:
        x.copy_((q * scale).reshape(shape).to(x.dtype))
        return x
    return q.reshape(shape), scale.reshape(*shape[:-1], n // block_size)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert inplace
    shape = x.shape
    n = shape[-1]
    blocks = x.float().reshape(-1, n // block_size, block_size)
    amax = blocks.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        scale = (amax.clamp(min=6.0 * 2.0 ** -9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        scale = pow2_ceil(amax.clamp(min=6.0 * 2.0 ** -126) / 6.0)
    q = round_e2m1((blocks / scale).clamp(-6.0, 6.0))
    x.copy_((q * scale).reshape(shape).to(x.dtype))
    return x


def _expand_activation(a, a_s, block):
    k = a.size(-1)
    return (a.reshape(-1, k // block, block) * a_s.reshape(-1, k // block, 1)).reshape(-1, k)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    x = _expand_activation(a, a_s, block_size)
    n, k = b.shape
    scale = b_s.float().repeat_interleave(block_size, 0)[:n].repeat_interleave(block_size, 1)[:, :k]
    w = b.float() * scale
    out = x @ w.t()
    return out.reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    x = _expand_activation(a, a_s, act_block_size)
    packed = b.view(torch.uint8)
    n = packed.size(0)
    values = E2M1_VALUES.to(packed.device)
    w = torch.stack([values[(packed & 0xF).long()], values[(packed >> 4).long()]], dim=-1).reshape(n, -1)
    w = w * b_s.float().repeat_interleave(32, 1)
    out = x @ w.t()
    return out.reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    batch = torch.arange(b, device=q.device).view(b, 1, 1)
    keys = kv[batch, idx.clamp(min=0)].float()
    keys = torch.where(valid.unsqueeze(-1), keys, torch.zeros_like(keys))
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), keys) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    top = scores.amax(-1, keepdim=True).clamp(min=-1e30)
    p = torch.exp(scores - top)
    denominator = p.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - top.squeeze(-1))
    o = torch.einsum("bmht,bmtd->bmhd", p.to(torch.bfloat16).float(), keys) / denominator.unsqueeze(-1)
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    m = hc_mult
    pre = torch.sigmoid(mixes[..., :m] * hc_scale[0] + hc_base[:m]) + eps
    post = 2 * torch.sigmoid(mixes[..., m:2 * m] * hc_scale[1] + hc_base[m:2 * m])
    comb = (mixes[..., 2 * m:] * hc_scale[2] + hc_base[2 * m:]).unflatten(-1, (m, m))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def install_kernel_module():
    module = types.ModuleType("kernel")
    for function in (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn):
        setattr(module, function.__name__, function)
    sys.modules["kernel"] = module


NUMPY_TO_TORCH = {
    "BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu, "U8": torch.uint8, "I8": torch.int8, "F16": torch.float16,
    "I64": torch.int64,
}


class Checkpoint:
    def __init__(self, root):
        self.st = Safetensors(root)

    def dtype(self, name):
        return self.st.entry(name)["dtype"]

    def tensor(self, name, device):
        array = self.st.pread(name)
        stored = self.dtype(name)
        raw = torch.from_numpy(np.ascontiguousarray(array))
        if stored in ("BF16", "F8_E4M3", "F8_E8M0"):
            raw = raw.view(NUMPY_TO_TORCH[stored])
        return raw.to(device)

    def rows(self, name, first, count, device):
        array = self.st.raw_rows(name, first, count)
        stored = self.dtype(name)
        raw = torch.from_numpy(np.ascontiguousarray(array))
        if stored in ("BF16", "F8_E4M3", "F8_E8M0"):
            raw = raw.view(NUMPY_TO_TORCH[stored])
        return raw.to(device)

    def weight_for(self, name, parameter, device):
        stored = self.dtype(name)
        target = parameter.dtype
        value = self.tensor(name, device)
        if stored in ("U8", "I8", "F8_E4M3") and target in (torch.float8_e4m3fn,):
            return value.view(torch.float8_e4m3fn)
        if stored in ("U8", "I8") and target == torch.float4_e2m1fn_x2:
            return value.view(torch.uint8).view(torch.float4_e2m1fn_x2)
        if stored in ("U8", "I8", "F8_E4M3") and target in (torch.bfloat16, torch.float32):
            scale = self.tensor(name[:-len(".weight")] + ".scale", device).view(torch.float8_e8m0fnu).float()
            w = value.view(torch.float8_e4m3fn).float()
            n, k = w.shape
            block_n = n // scale.size(0) if n % scale.size(0) == 0 else 32
            block_k = k // scale.size(1)
            full = scale.repeat_interleave(block_n, 0)[:n].repeat_interleave(block_k, 1)[:, :k]
            return (w * full).to(target)
        if stored in ("F8_E8M0", "U8") and target == torch.float8_e8m0fnu:
            return value.view(torch.uint8).view(torch.float8_e8m0fnu)
        return value.to(target)


def build_model(inference_dir, checkpoint_dir, device, max_seq_len):
    sys.path.insert(0, inference_dir)
    install_kernel_module()
    import model as publisher
    import engram as publisher_engram
    from tokenizers import Tokenizer

    config = json.load(open(os.path.join(inference_dir, "config.json")))
    config = {k: v for k, v in config.items() if k in publisher.ModelArgs.__dataclass_fields__}
    config.update(max_batch_size=1, max_seq_len=max_seq_len, temperature=0.0,
                  vision_n_layers=0, dspark_block_size=0, n_mtp_layers=0)
    for key, value in list(config.items()):
        if isinstance(value, list):
            config[key] = tuple(value)
    args = publisher.ModelArgs(**config)

    class TokenizerView:
        def __init__(self, path):
            self.backend_tokenizer = Tokenizer.from_file(path)

        def __len__(self):
            return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

    tokenizer = TokenizerView(os.path.join(checkpoint_dir, "tokenizer.json"))
    checkpoint = Checkpoint(checkpoint_dir)

    real_expert_init = publisher.Expert.__init__
    real_embed_init = publisher.ParallelEngramEmbedding.__init__

    def expert_init(self, dim, inter_dim, dtype=None, swiglu_limit=0.0):
        if dtype == torch.float4_e2m1fn_x2:
            with torch.device("meta"):
                real_expert_init(self, dim, inter_dim, dtype=dtype, swiglu_limit=swiglu_limit)
        else:
            real_expert_init(self, dim, inter_dim, dtype=dtype, swiglu_limit=swiglu_limit)

    def engram_embed_init(self, num_embeddings, dim):
        with torch.device("meta"):
            real_embed_init(self, num_embeddings, dim)

    publisher.Expert.__init__ = expert_init
    publisher.ParallelEngramEmbedding.__init__ = engram_embed_init
    torch.set_default_dtype(torch.bfloat16)
    transformer = publisher.Transformer(args, tokenizer)
    for module in transformer.modules():
        for buffer_name, buffer in list(module._buffers.items()):
            if buffer is not None and not buffer.is_meta:
                module._buffers[buffer_name] = buffer.to(device)
    publisher.Expert.__init__ = real_expert_init
    publisher.ParallelEngramEmbedding.__init__ = real_embed_init

    loaded = 0
    for name, parameter in transformer.named_parameters():
        if parameter.is_meta:
            continue
        value = checkpoint.weight_for(name, parameter, device)
        if tuple(value.shape) != tuple(parameter.shape):
            raise SystemExit(f"{name}: checkpoint {tuple(value.shape)} vs model {tuple(parameter.shape)}")
        with torch.no_grad():
            parameter.data = value
        loaded += 1
    for module_name, module in transformer.named_modules():
        if isinstance(module, publisher.Linear) and module.scale is not None and not module.weight.is_meta:
            module.weight.scale = module.scale
    for module_name, module in transformer.named_modules():
        if isinstance(module, publisher.MoE):
            install_lazy_experts(publisher, module, module_name, checkpoint, device)
        if isinstance(module, publisher.ParallelEngramEmbedding):
            install_lazy_engram(module, module_name, checkpoint, device)
    return transformer, loaded


def install_lazy_experts(publisher, moe, prefix, checkpoint, device):
    def load(expert_index):
        base = f"{prefix}.experts.{expert_index}."
        expert = publisher.Expert.__new__(publisher.Expert)
        torch.nn.Module.__init__(expert)
        expert.swiglu_limit = moe.shared_experts.swiglu_limit
        for part in ("w1", "w2", "w3"):
            payload = checkpoint.tensor(base + part + ".weight", device).view(torch.uint8)
            scale = checkpoint.tensor(base + part + ".scale", device).view(torch.uint8).view(torch.float8_e8m0fnu)
            linear = publisher.Linear.__new__(publisher.Linear)
            torch.nn.Module.__init__(linear)
            linear.weight = torch.nn.Parameter(payload.view(torch.float4_e2m1fn_x2), requires_grad=False)
            linear.scale = torch.nn.Parameter(scale, requires_grad=False)
            linear.weight.scale = linear.scale
            linear.bias = None
            setattr(expert, part, linear)
        return expert

    def forward(x, image_mask=None):
        shape = x.size()
        x = x.view(-1, moe.dim)
        weights, indices = moe.gate(x, None)
        y = torch.zeros_like(x, dtype=torch.float32)
        for expert_index in sorted(set(indices.flatten().tolist())):
            expert = load(expert_index)
            idx, top = torch.where(indices == expert_index)
            y[idx] += expert(x[idx], weights[idx, top, None])
        y += moe.shared_experts(x)
        return y.type_as(x).view(shape)

    moe.forward = forward


def install_lazy_engram(embed, prefix, checkpoint, device):
    def forward(indices):
        flat = indices.flatten().tolist()
        weight_rows = []
        scale_rows = []
        for row in flat:
            weight_rows.append(checkpoint.rows(prefix + ".weight", int(row), 1, device))
            scale_rows.append(checkpoint.rows(prefix + ".scale", int(row), 1, device))
        values = torch.cat(weight_rows).view(torch.float8_e4m3fn).float()
        scales = torch.cat(scale_rows).view(torch.uint8).view(torch.float8_e8m0fnu).float()
        values = values.unflatten(-1, (-1, embed.block_size)) * scales.unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16).view(*indices.shape, embed.dim)

    embed.forward = forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inference-dir")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--new-tokens", type=int, default=12)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--capture-dir")
    parser.add_argument("--token-by-token", action="store_true")
    arguments = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if arguments.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(arguments.memory_fraction)
    inference_dir = arguments.inference_dir or os.path.join(arguments.checkpoint, "inference")
    started = time.time()
    transformer, loaded = build_model(inference_dir, arguments.checkpoint, arguments.device, arguments.max_seq_len)
    torch.set_default_device(arguments.device)
    print(json.dumps({"loaded_parameters": loaded, "seconds": round(time.time() - started, 1)}), flush=True)
    prompts = json.load(open(arguments.prompts))["prompts"]
    results = []
    captured = {}
    if arguments.capture_dir:
        os.makedirs(arguments.capture_dir, exist_ok=True)
        calls = {"count": 0}

        def save(key, tensor):
            captured[f"call{calls['count']:03d}_{key}"] = tensor.detach().to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)

        def save_float(key, tensor):
            captured[f"call{calls['count']:03d}_{key}"] = tensor.detach().float().cpu().numpy()

        def count_call(module, inputs, output):
            calls["count"] += 1

        for layer_index, layer in enumerate(transformer.layers):
            layer.register_forward_hook(lambda module, inputs, output, i=layer_index: save(f"layer{i:02d}_streams", output[0]))
            layer.register_forward_hook(lambda module, inputs, output, i=layer_index: save_float(f"layer{i:02d}_pre_mix_out", output[1]))
            layer.register_forward_pre_hook(lambda module, inputs, i=layer_index: save(f"layer{i:02d}_input", inputs[0]))
            layer.attn.register_forward_hook(lambda module, inputs, output, i=layer_index: save(f"layer{i:02d}_attn_out", output))
            layer.attn.register_forward_pre_hook(lambda module, inputs, i=layer_index: save(f"layer{i:02d}_attn_in", inputs[0]))
            layer.ffn.register_forward_hook(lambda module, inputs, output, i=layer_index: save(f"layer{i:02d}_ffn_out", output))
            layer.ffn.register_forward_pre_hook(lambda module, inputs, i=layer_index: save(f"layer{i:02d}_ffn_in", inputs[0]))
        transformer.head.register_forward_hook(count_call)
    with torch.inference_mode():
        for prompt in prompts:
            ids = list(prompt["prompt_token_ids"])
            tokens = torch.tensor([ids], device=arguments.device)
            if arguments.token_by_token:
                for position in range(len(ids) - 1):
                    transformer.forward(tokens[:, position:position + 1], position)
                tokens = tokens[:, len(ids) - 1:]
            generated = []
            margins = []
            start = len(ids) - 1 if arguments.token_by_token else 0
            step_started = time.time()
            for step in range(arguments.new_tokens):
                _, logits, _ = transformer.forward(tokens, start)
                token = int(logits[0].float().argmax())
                top2 = torch.topk(logits[0].float(), 2)
                generated.append(token)
                margins.append(round(float(top2.values[0] - top2.values[1]), 4))
                print(json.dumps({"prompt": prompt["name"], "step": step, "token": token,
                                  "margin": round(float(top2.values[0] - top2.values[1]), 4),
                                  "seconds": round(time.time() - step_started, 1)}), flush=True)
                start += tokens.size(1)
                tokens = torch.tensor([[token]], device=arguments.device)
                if token == 1:
                    break
            results.append({"name": prompt["name"], "prompt_token_ids": ids, "generated_token_ids": generated,
                            "top1_margins": margins})
            if arguments.capture_dir:
                np.savez(os.path.join(arguments.capture_dir, prompt["name"] + ".npz"), **captured)
                captured.clear()
    json.dump({"generator": "tools/dsv41_official_harness.py", "prompts": results}, open(arguments.output, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
