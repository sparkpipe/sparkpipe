#!/usr/bin/env python3
"""Run OUR drafter forward on vLLM's dumped inputs and compare drafts.

Dumps: /tmp/vllmin_<n>.pt from the patched DFlashProposer.set_inputs_first_pass
on spark3 - target_hidden_states (the COMBINED per-row hiddens, i.e. our fc
output), target_positions, next_token_ids (the bonus), target_token_ids (the
verify's walked rows), num_rejected.

Round N's proposed drafts = round N+1's walked target_token_ids (rows after
the bonus). We rebuild our block forward (deferred pair: ctx rows at their
positions, anchor embed one past the last ctx row) with OUR weights and walk
OUR selector, then compare token-by-token.
"""
import sys

import torch
from safetensors import safe_open

DRAFTER = "/home/spark3/sparkdata/qwen38-dflash2-drafter"
H, LAYERS, NQ, NKV, HD, ROPE = 5120, 5, 32, 8, 128, 64
BLOCK, TAPS, FFN = 8, 5, 17408
TOPK, CGS, TAPS2 = 16, 16, 2
EPS, THETA = 1e-6, 1e7
MASK_ID = 248070
DEV = "cpu"


def load(path, names=None):
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if names and not any(n in k for n in names):
                continue
            out[k] = f.get_tensor(k).to(DEV, torch.bfloat16)
    return out


def rms(x, w):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS).to(x.dtype) * w


def rope(x, pos):
    inv = 1.0 / (THETA ** (torch.arange(0, ROPE, 2, device=DEV, dtype=torch.float32) / ROPE))
    ang = pos.float()[:, None] * inv[None, :]
    cs = ang.cos()[:, None, :].to(x.dtype)
    sn = ang.sin()[:, None, :].to(x.dtype)
    out = x.clone()
    re, im = x[..., 0:ROPE:2].clone(), x[..., 1:ROPE:2].clone()
    out[..., 0:ROPE:2] = re * cs - im * sn
    out[..., 1:ROPE:2] = re * sn + im * cs
    return out


def gconv(h, delta, base):
    T = h.shape[0]
    groups = H // CGS
    blocks = h.view(T, groups, CGS)
    coeff = base.view(1, TAPS2, groups, CGS) + delta[:, :, :, None]
    o = coeff[:, 0] * blocks
    pos = torch.arange(T, device=DEV) & (BLOCK - 1)
    for tap in range(1, TAPS2):
        shifted = torch.cat([torch.zeros(tap, groups, CGS, device=DEV, dtype=h.dtype), blocks[:-tap]], 0)
        o = o + coeff[:, tap] * shifted * (pos >= tap).view(-1, 1, 1)
    return o.view(T, H)


def layer(x, ctx, lw, q_pos, kv_pos):
    h = rms(x, lw["input_layernorm.weight"])
    ca = (h @ lw["attention_conv.kernel_projection.weight"].T).view(BLOCK, 2, TAPS2, -1)
    h2 = gconv(h, ca[:, 0], lw["attention_conv.base_kernel"][0])
    q = rms((h2 @ lw["self_attn.q_proj.weight"].T).view(BLOCK, NQ, HD), lw["self_attn.q_norm.weight"])
    q = rope(q, q_pos)
    k = rms(torch.cat([(ctx @ lw["self_attn.k_proj.weight"].T).view(-1, NKV, HD),
                       (h2 @ lw["self_attn.k_proj.weight"].T).view(BLOCK, NKV, HD)]), lw["self_attn.k_norm.weight"])
    k = rope(k, kv_pos)
    v = torch.cat([(ctx @ lw["self_attn.v_proj.weight"].T).view(-1, NKV, HD),
                   (h2 @ lw["self_attn.v_proj.weight"].T).view(BLOCK, NKV, HD)])
    out = torch.empty(BLOCK, NQ, HD, device=DEV, dtype=torch.float32)
    for qh in range(NQ):
        kvh = qh // (NQ // NKV)
        s = (q[:, qh].float() @ k[:, kvh].float().T) * (HD ** -0.5)
        out[:, qh] = torch.softmax(s, -1) @ v[:, kvh].float()
    att = out.to(torch.bfloat16).view(BLOCK, NQ * HD)
    o = att @ lw["self_attn.o_proj.weight"].T
    x = x + gconv(o, ca[:, 1], lw["attention_conv.base_kernel"][1])
    h = rms(x, lw["post_attention_layernorm.weight"])
    cm = (h @ lw["mlp_conv.kernel_projection.weight"].T).view(BLOCK, 2, TAPS2, -1)
    h2 = gconv(h, cm[:, 0], lw["mlp_conv.base_kernel"][0])
    g = torch.nn.functional.silu(h2 @ lw["mlp.gate_proj.weight"].T)
    u = h2 @ lw["mlp.up_proj.weight"].T
    x = x + gconv((g * u) @ lw["mlp.down_proj.weight"].T, cm[:, 1], lw["mlp_conv.base_kernel"][1])
    return x


def main():
    import glob
    import re
    w = load(f"{DRAFTER}/model.safetensors")
    emb_w = load("/home/spark3/sparkdata/Qwen3.8-27B-local/model-00003-of-00018.safetensors", ["embed_tokens.weight"])["model.language_model.embed_tokens.weight"]
    lm_w = load("/home/spark3/sparkdata/Qwen3.8-27B-local/model-00018-of-00018.safetensors", ["lm_head.weight"])["lm_head.weight"]
    # local copies may be whole-dir; fall back to names present
    if emb_w is None:
        raise SystemExit("embed shard missing")
    dumps = {}
    for f in glob.glob("/tmp/vllmin_*.pt"):
        n = int(re.search(r"vllmin_(\d+)", f).group(1))
        dumps[n] = torch.load(f, weights_only=False, map_location="cpu")
    agree = [0] * 7
    total = 0
    for n in sorted(dumps):
        if n + 1 not in dumps:
            continue
        cur, nxt = dumps[n], dumps[n + 1]
        hidden = cur["target_hidden_states"].to(DEV, torch.bfloat16)   # combined rows
        pos = cur["target_positions"].to(DEV).view(-1)
        bonus = int(cur["next_token_ids"][0])
        walked = [int(t) for t in nxt["target_token_ids"].view(-1).tolist()]                       # their proposed chain
        if len(walked) < 8 or not isinstance(walked[0], int):
            walked = [int(t) for t in walked[:8]]
        their_drafts = walked[1:8]
        # our forward: ctx = their combined rows (rms with our hidden_norm),
        # anchor = bonus roped one past the last ctx row
        nr = 0
        if cur.get("num_rejected") is not None:
            nr = int(cur["num_rejected"].view(-1)[0])
        vis = max(1, hidden.shape[0] - nr)
        import os as _os2
        _wcut = int(_os2.environ.get("CTX_TRUNC", "0"))
        if _wcut > 0 and vis > _wcut:
            vis = _wcut
        ctx = rms(hidden[:vis], w["hidden_norm.weight"])
        W = ctx.shape[0]
        pos = pos[:W]
        ids = torch.tensor([bonus] + [MASK_ID] * (BLOCK - 1), device=DEV)
        x = emb_w[ids]
        pmax = int(pos.max())
        q_pos = torch.arange(pmax + 1, pmax + 1 + BLOCK, device=DEV)
        kv_pos = torch.cat([pos, q_pos])
        for layer_i in range(LAYERS):
            lw = {k.split(f"layers.{layer_i}.")[1]: v for k, v in w.items() if f"layers.{layer_i}." in k}
            x = layer(x, ctx, lw, q_pos, kv_pos)
        hid = rms(x, w["norm.weight"])
        logits = hid[1:].float() @ lm_w.float().T
        topv, topi = torch.topk(logits, TOPK, dim=-1)
        hp = hid[1:] @ w["candidate_selector.hidden_projection.weight"].T
        preds = w["candidate_selector.predecessor_codebook"]
        succs = w["candidate_selector.successor_codebook"]
        prev_id = bonus
        ours = []
        for slot in range(BLOCK - 1):
            scores = topv[slot].float() + (preds[prev_id].float() * hp[slot].float()) @ succs[topi[slot]].float().T
            best = int(torch.argmax(scores))
            ours.append(int(topi[slot][best]))
            prev_id = ours[-1]
        total += 1
        for j in range(7):
            agree[j] += ours[j] == their_drafts[j]
        if n <= 6:
            print(f"round {n}: bonus={bonus} pos={pmax}")
            print(f"  theirs={their_drafts}")
            print(f"  ours  ={ours}")
    print(f"rounds compared: {total}")
    print("agreement per position:", " ".join(f"{a}/{total}" for a in agree))


if __name__ == "__main__":
    main()
