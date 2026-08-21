#!/usr/bin/env python3
"""Convention sweep on the O128 captures: find the block/context/rope geometry
the trained drafter actually wants, scored by per-position draft hits against
the round truth (both emission alignments).

Combos: ctx_end in {base-1 (device), base (g_P included)} x block_start in
{base-1 (device), base (pair layout)} x rope in {interleaved, neox-128}.
"""
import json
import sys

import torch
from safetensors import safe_open

DRAFTER = "/home/spark0/sparkdata/qwen38-dflash2-drafter"
H, LAYERS, NQ, NKV, HD, ROPE = 5120, 5, 32, 8, 128, 128
FFN, VOCAB, BLOCK, TAPS = 17408, 248320, 8, 5
RANK, TOPK, CGS, TAPS2 = 256, 16, 16, 2
EPS, THETA = 1e-6, 1e7
MASK_ID = 248070
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BUNDLE = "/tmp/o128bundle"


def load(path, names=None):
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if names and not any(n in k for n in names):
                continue
            out[k] = f.get_tensor(k).to(DEV, torch.bfloat16)
    return out


def apply_rope(x, pos, style):
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


def rms(x, w):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS).to(x.dtype) * w


def grouped_conv(h, delta, base, block):
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


def layer_forward(x, ctx_kv, lw, q_pos, kv_pos, rope):
    h = rms(x, lw["input_layernorm.weight"])
    coeff_all = (h @ lw["attention_conv.kernel_projection.weight"].T).view(BLOCK, 2, TAPS2, -1)
    h2 = grouped_conv(h, coeff_all[:, 0], lw["attention_conv.base_kernel"][0], BLOCK)
    q = (h2 @ lw["self_attn.q_proj.weight"].T).view(BLOCK, NQ, HD)
    q = rms(q, lw["self_attn.q_norm.weight"])
    q = apply_rope(q, q_pos, rope)
    k_ctx = (ctx_kv @ lw["self_attn.k_proj.weight"].T).view(-1, NKV, HD)
    v_ctx = (ctx_kv @ lw["self_attn.v_proj.weight"].T).view(-1, NKV, HD)
    k_blk = (h2 @ lw["self_attn.k_proj.weight"].T).view(BLOCK, NKV, HD)
    v_blk = (h2 @ lw["self_attn.v_proj.weight"].T).view(BLOCK, NKV, HD)
    k = rms(torch.cat([k_ctx, k_blk]), lw["self_attn.k_norm.weight"])
    k = apply_rope(k, kv_pos, rope)
    v = torch.cat([v_ctx, v_blk])
    out = torch.empty(BLOCK, NQ, HD, device=DEV, dtype=torch.float32)
    for qh in range(NQ):
        kvh = qh // (NQ // NKV)
        s = (q[:, qh].float() @ k[:, kvh].float().T) * (HD ** -0.5)
        out[:, qh] = torch.softmax(s, -1) @ v[:, kvh].float()
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


def forward_round(w, emb_w, lm_w, taps, taps_start, ctx_hi, block_start, anchor, rope):
    W = ctx_hi - taps_start
    ctx = taps[:W].view(W, -1).float() @ w["fc.weight"].float().T
    ctx = rms(ctx.to(torch.bfloat16), w["hidden_norm.weight"])
    ids = torch.tensor([anchor] + [MASK_ID] * (BLOCK - 1), device=DEV)
    x = emb_w[ids]
    q_pos = torch.arange(block_start, block_start + BLOCK, device=DEV)
    kv_pos = torch.arange(taps_start, taps_start + W + BLOCK, device=DEV)
    for layer in range(LAYERS):
        lw = {k.split(f"layers.{layer}.")[1]: v for k, v in w.items() if f"layers.{layer}." in k}
        x = layer_forward(x, ctx, lw, q_pos, kv_pos, rope)
    hidden = rms(x, w["norm.weight"])
    logits = hidden[1:].float() @ lm_w.float().T
    topv, topi = torch.topk(logits, TOPK, dim=-1)
    hp = hidden[1:] @ w["candidate_selector.hidden_projection.weight"].T
    preds = w["candidate_selector.predecessor_codebook"]
    succs = w["candidate_selector.successor_codebook"]
    prev_id = anchor
    drafts = []
    for slot in range(BLOCK - 1):
        scores = topv[slot].float() + (preds[prev_id].float() * hp[slot].float()) @ succs[topi[slot]].float().T
        best = int(torch.argmax(scores))
        drafts.append(int(topi[slot][best]))
        prev_id = drafts[-1]
    return drafts


def main():
    import numpy as np
    w = load(f"{DRAFTER}/model.safetensors")
    emb_w = load(f"{BUNDLE}/../torchbundle/model-00003-of-00018.safetensors", ["embed_tokens.weight"])["model.language_model.embed_tokens.weight"]
    lm_w = load(f"{BUNDLE}/../torchbundle/model-00018-of-00018.safetensors", ["lm_head.weight"])["lm_head.weight"]
    bundle = json.load(open(f"{BUNDLE}/picks.json"))
    combos = []
    for rope in ("inter", "neox"):
        for ctx_hi_off in (0, 1):        # ctx end = base-1+off
            for blk_off in (0, 1):       # block start = base-1+off
                combos.append((rope, ctx_hi_off, blk_off))
    stats = {c: {"next": [0] * 7, "own": [0] * 7} for c in combos}
    n = 0
    for run_s in sorted(bundle, key=int):
        item = bundle[run_s]
        meta, truth = item["meta"], item["round"]["emitted"]
        base, window, lo, anchor = int(meta["base"]), int(meta["window"]), int(meta["lo"]), int(meta["anchor"])
        raw = np.fromfile(f"{BUNDLE}/run{run_s}_taps.bin", dtype=np.uint16).astype(np.uint32) << 16
        taps_all = torch.from_numpy(raw.view(np.float32).copy()).to(DEV).to(torch.bfloat16).view(-1, TAPS * H)
        wlo = base - 1 - window
        taps = taps_all[wlo - lo:]  # index 0 = wlo (absolute)
        n += 1
        for rope, chi, blk in combos:
            drafts = forward_round(w, emb_w, lm_w, taps, wlo, base - 1 + chi, base - 1 + blk, anchor, rope)
            for pos in range(7):
                stats[(rope, chi, blk)]["next"][pos] += drafts[pos] == truth[pos]
                stats[(rope, chi, blk)]["own"][pos] += drafts[pos] == (truth[pos - 1] if pos else item["round"]["c0"])
    print(f"rounds: {n}")
    for c in combos:
        for align in ("next", "own"):
            tot = sum(stats[c][align])
            p0 = stats[c][align][0]
            print(f"rope={c[0]:5s} ctx_end=base-1+{c[1]} blk=base-1+{c[2]} align={align:4s} "
                  f"total={tot:3d} p0={p0:3d} profile={' '.join(str(h) for h in stats[c][align])}")


if __name__ == "__main__":
    main()
