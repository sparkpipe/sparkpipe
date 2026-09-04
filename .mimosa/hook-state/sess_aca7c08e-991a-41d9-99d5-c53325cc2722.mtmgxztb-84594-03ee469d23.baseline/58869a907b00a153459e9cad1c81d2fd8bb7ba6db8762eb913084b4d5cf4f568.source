#!/usr/bin/env python3
"""Torch cross-check of the DFlash2 drafter on live captured inputs.

Runs the vLLM qwen3_dflash2.py math standalone (torch, bf16 on GPU) on the
ctxrun dumps captured from the SparkPipe device: same taps, same anchor, the
device's own context geometry. If torch drafts hit truth far better than the
device/numpy drafts on identical inputs, the shared port has a defect; if they
agree, the conditioning (inputs), not the forward, caps acceptance.

Bundle: /tmp/torchbundle/{picks.json, run<n>_taps.bin}
"""
import json
import math
import sys

import torch
from safetensors import safe_open

DRAFTER = "/home/spark0/sparkdata/qwen38-dflash2-drafter"
TARGET = None  # lm_head/embed come from the drafter pack if present

H, LAYERS, NQ, NKV, HD, ROPE = 5120, 5, 32, 8, 128, 128
FFN, VOCAB, BLOCK, TAPS = 17408, 248320, 8, 5
RANK, TOPK, CGS, TAPS2 = 256, 16, 16, 2
EPS, THETA = 1e-6, 1e7
MASK_ID = 248070
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(path, prefix_filter=None):
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if prefix_filter and not any(k.startswith(p) for p in prefix_filter):
                continue
            out[k] = f.get_tensor(k).to(DEV, torch.bfloat16)
    return out


def rope_cos_sin(pos):
    inv = 1.0 / (THETA ** (torch.arange(0, ROPE, 2, device=DEV, dtype=torch.float32) / ROPE))
    p = pos.float()[:, None] * inv[None, :]
    return p.cos(), p.sin()


def apply_rope(x, pos, style="neox"):
    half = ROPE // 2
    inv = 1.0 / (THETA ** (torch.arange(0, ROPE, 2, device=DEV, dtype=torch.float32) / ROPE))
    ang = pos.float()[:, None] * inv[None, :]
    cs = ang.cos()[:, None, :].to(x.dtype)
    sn = ang.sin()[:, None, :].to(x.dtype)
    out = x.clone()
    if style == "neox":
        re, im = x[..., :half].clone(), x[..., half:ROPE].clone()
        out[..., :half] = re * cs - im * sn
        out[..., half:ROPE] = re * sn + im * cs
    else:
        re, im = x[..., 0:ROPE:2].clone(), x[..., 1:ROPE:2].clone()
        out[..., 0:ROPE:2] = re * cs - im * sn
        out[..., 1:ROPE:2] = re * sn + im * cs
    return out


ROPE_STYLE = "neox"


def rms(x, w):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS).to(x.dtype) * w


def grouped_conv(h, delta, base, block):
    # h [T,H] bf16, delta [T, taps, groups], base [taps, H]
    T = h.shape[0]
    groups = H // CGS
    blocks = h.view(T, groups, CGS)
    coeff = base.view(1, TAPS2, groups, CGS) + delta[:, :, :, None]
    out = coeff[:, 0] * blocks
    pos = torch.arange(T, device=DEV) & (block - 1)
    for tap in range(1, TAPS2):
        shifted = torch.cat([torch.zeros(tap, groups, CGS, device=DEV, dtype=h.dtype), blocks[:-tap]], 0)
        out = out + coeff[:, tap] * shifted * (pos >= tap).view(-1, 1, 1)
    return out.view(T, H)


def layer_forward(x, ctx_kv, lw, q_pos, kv_pos):
    h = rms(x, lw["input_layernorm.weight"])
    coeff_all = (h @ lw["attention_conv.kernel_projection.weight"].T).view(BLOCK, 2, TAPS2, -1)
    h2 = grouped_conv(h, coeff_all[:, 0], lw["attention_conv.base_kernel"][0], BLOCK)
    q = (h2 @ lw["self_attn.q_proj.weight"].T).view(BLOCK, NQ, HD)
    q = rms(q, lw["self_attn.q_norm.weight"])
    q = apply_rope(q, q_pos, ROPE_STYLE)
    k_ctx = (ctx_kv @ lw["self_attn.k_proj.weight"].T).view(-1, NKV, HD)
    v_ctx = (ctx_kv @ lw["self_attn.v_proj.weight"].T).view(-1, NKV, HD)
    k_blk = (h2 @ lw["self_attn.k_proj.weight"].T).view(BLOCK, NKV, HD)
    v_blk = (h2 @ lw["self_attn.v_proj.weight"].T).view(BLOCK, NKV, HD)
    k = torch.cat([k_ctx, k_blk])
    v = torch.cat([v_ctx, v_blk])
    k = rms(k, lw["self_attn.k_norm.weight"])
    k = apply_rope(k, kv_pos, ROPE_STYLE)
    out = torch.empty(BLOCK, NQ, HD, device=DEV, dtype=torch.float32)
    for qh in range(NQ):
        kvh = qh // (NQ // NKV)
        s = (q[:, qh].float() @ k[:, kvh].float().T) * (HD ** -0.5)
        p = torch.softmax(s, -1)
        out[:, qh] = p @ v[:, kvh].float()
    att = out.to(torch.bfloat16).view(BLOCK, NQ * HD)
    o = att @ lw["self_attn.o_proj.weight"].T
    fin = grouped_conv(o, coeff_all[:, 1], lw["attention_conv.base_kernel"][1], BLOCK)
    x = x + fin
    h = rms(x, lw["post_attention_layernorm.weight"])
    coeff2 = (h @ lw["mlp_conv.kernel_projection.weight"].T).view(BLOCK, 2, TAPS2, -1)
    h2 = grouped_conv(h, coeff2[:, 0], lw["mlp_conv.base_kernel"][0], BLOCK)
    g = torch.nn.functional.silu(h2 @ lw["mlp.gate_proj.weight"].T)
    u = h2 @ lw["mlp.up_proj.weight"].T
    d = (g * u) @ lw["mlp.down_proj.weight"].T
    x = x + grouped_conv(d, coeff2[:, 1], lw["mlp_conv.base_kernel"][1], BLOCK)
    return x


def main():
    global ROPE_STYLE
    if len(sys.argv) > 1:
        ROPE_STYLE = sys.argv[1]
    print("rope style:", ROPE_STYLE)
    w = load(f"{DRAFTER}/model.safetensors")
    emb_w = load("/tmp/torchbundle/model-00003-of-00018.safetensors", ["model.language_model.embed_tokens.weight"])["model.language_model.embed_tokens.weight"]
    lm_w = load("/tmp/torchbundle/model-00018-of-00018.safetensors", ["lm_head.weight"])["lm_head.weight"]
    torch.manual_seed(0)
    bundle = json.load(open("/tmp/torchbundle/picks.json"))
    for run_s in sorted(bundle, key=int):
        item = bundle[run_s]
        meta = item["meta"]
        base, window, lo = int(meta["base"]), int(meta["window"]), int(meta["lo"])
        anchor = int(meta["anchor"])
        truth = item["round"]["emitted"]
        device = item["device_drafts"]
        import numpy as np
        taps32 = np.fromfile(f"/tmp/torchbundle/run{run_s}_taps.bin", dtype=np.uint16).astype(np.uint32) << 16
        taps = torch.from_numpy(taps32.view(np.float32).copy()).to(DEV).to(torch.bfloat16).view(-1, TAPS, H).reshape(-1, TAPS * H)
        wlo = base - 1 - window
        ctx = taps[wlo - lo:base - 1 - lo].view(window, -1).float() @ w["fc.weight"].float().T
        ctx = rms(ctx.to(torch.bfloat16), w["hidden_norm.weight"])
        embed_ids = torch.tensor([anchor] + [MASK_ID] * (BLOCK - 1), device=DEV)
        x = emb_w[embed_ids]
        q_pos = torch.arange(base - 1, base - 1 + BLOCK, device=DEV)
        kv_pos = torch.arange(wlo, wlo + window + BLOCK, device=DEV)
        for layer in range(LAYERS):
            lw = {k.split(f"layers.{layer}.")[1]: v for k, v in w.items() if f"layers.{layer}." in k}
            x = layer_forward(x, ctx, lw, q_pos, kv_pos)
        hidden = rms(x, w["norm.weight"])
        logits = (hidden[1:].float() @ lm_w.float().T)
        topv, topi = torch.topk(logits, TOPK, dim=-1)
        hp = hidden[1:] @ w["candidate_selector.hidden_projection.weight"].T
        preds = w["candidate_selector.predecessor_codebook"]
        succs = w["candidate_selector.successor_codebook"]
        prev_id = anchor
        ref = []
        for slot in range(BLOCK - 1):
            scores = topv[slot].float() + (preds[prev_id].float() * hp[slot].float()) @ succs[topi[slot]].float().T
            best = int(torch.argmax(scores))
            ref.append(int(topi[slot][best]))
            prev_id = ref[-1]
        top1 = topi[:, 0].tolist()
        dev_hits = "".join("1" if d == t else "." for d, t in zip(device, truth))
        ref_hits = "".join("1" if d == t else "." for d, t in zip(ref, truth))
        t1_hits = "".join("1" if d == t else "." for d, t in zip(top1, truth))
        print(f"run {run_s}: acc={item['round']['acc']} dev={dev_hits} torch={ref_hits} top1={t1_hits}")
        print(f"        device={device}")
        print(f"        torch ={ref}  top1={top1}")


if __name__ == "__main__":
    main()
