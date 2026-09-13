#!/usr/bin/env python3
"""Stage-by-stage diff of OUR DFlash2 forward vs the LIVE reference's dumped
tensors on spark3.

Inputs (produced by the instrumented reference, --enforce-eager):
  /tmp/vllmin_<n>.pt  — set_inputs_first_pass: the COMBINED (fc'd) context
                        rows, target_positions, bonus, num_rejected
  /tmp/vllmctx_<n>.pt — precompute_and_store_context_kv: context_states,
                        positions, slots, all_k [5, ctx, nkv, hd], all_v
  /tmp/vllmsel_<n>.pt — _generate_draft: query_input_ids, query_positions,
                        anchor, per-layer hiddens, embeds, mask_hidden,
                        candidate_ids, unary_logits, scores_lattice,
                        draft_tokens, temperature

Stages compared (first divergence localizes the bug):
  0  query ids + positions
  1  query embeddings (embed * input_embedding_scale)
  2  layer-0 context K/V (hidden_norm -> k_proj -> k_norm -> neox rope)
  3  per-layer block hiddens (residual stream after each of the 5 layers)
  4  final normed mask-row hidden
  5  top-16 ids + unary
  6  hidden projection + edge lattice
  7  walk (edge-only and unary+edge variants) vs draft_tokens
"""
import os
import sys

import torch
from safetensors import safe_open

DRAFTER = "/home/spark3/sparkdata/qwen38-dflash2-drafter"
TARGET = "/home/spark3/sparkdata/Qwen3.8-27B-local"
H, LAYERS, NQ, NKV, HD = 5120, 5, 32, 8, 128
BLOCK, TAPS2, TOPK, CGS = 8, 2, 16, 16
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


def cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def rms(x, w):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS).to(x.dtype) * w


def rope_neox(x, pos):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, device=DEV, dtype=torch.float32) / HD))
    ang = pos.float()[:, None] * inv[None, :]
    cs = ang.cos()[:, None, :].to(x.dtype)
    sn = ang.sin()[:, None, :].to(x.dtype)
    out = x.clone()
    half = HD // 2
    re, im = x[..., :half].clone(), x[..., half:HD].clone()
    out[..., :half] = re * cs - im * sn
    out[..., half:HD] = re * sn + im * cs
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


def layer_fwd(x, ctx_normed, lw, q_pos, kv_pos):
    """One conv-wrapped decoder layer. Context K/V from the (already
    hidden_norm'd) fc rows; block K/V from the conv-prepared h. Non-causal."""
    h = rms(x, lw["input_layernorm.weight"])
    ca = (h @ lw["attention_conv.kernel_projection.weight"].T).view(BLOCK, 2, TAPS2, -1)
    h2 = gconv(h, ca[:, 0], lw["attention_conv.base_kernel"][0])
    q = rms((h2 @ lw["self_attn.q_proj.weight"].T).view(BLOCK, NQ, HD), lw["self_attn.q_norm.weight"])
    q = rope_neox(q, q_pos)
    k_all = torch.cat([(ctx_normed @ lw["self_attn.k_proj.weight"].T).view(-1, NKV, HD),
                       (h2 @ lw["self_attn.k_proj.weight"].T).view(BLOCK, NKV, HD)])
    v_all = torch.cat([(ctx_normed @ lw["self_attn.v_proj.weight"].T).view(-1, NKV, HD),
                       (h2 @ lw["self_attn.v_proj.weight"].T).view(BLOCK, NKV, HD)])
    k = rope_neox(rms(k_all, lw["self_attn.k_norm.weight"]), kv_pos)
    v = v_all
    out = torch.empty(BLOCK, NQ, HD, device=DEV, dtype=torch.float32)
    for qh in range(NQ):
        kvh = qh // (NQ // NKV)
        s = (q[:, qh].float() @ k[:, kvh].float().T) * (HD ** -0.5)
        out[:, qh] = torch.softmax(s, -1) @ v[:, kvh].float()
    att = out.to(x.dtype).view(BLOCK, NQ * HD)
    o = att @ lw["self_attn.o_proj.weight"].T
    x = x + gconv(o, ca[:, 1], lw["attention_conv.base_kernel"][1])
    h = rms(x, lw["post_attention_layernorm.weight"])
    cm = (h @ lw["mlp_conv.kernel_projection.weight"].T).view(BLOCK, 2, TAPS2, -1)
    h2 = gconv(h, cm[:, 0], lw["mlp_conv.base_kernel"][0])
    g = torch.nn.functional.silu(h2 @ lw["mlp.gate_proj.weight"].T)
    u = h2 @ lw["mlp.up_proj.weight"].T
    x = x + gconv((g * u) @ lw["mlp.down_proj.weight"].T, cm[:, 1], lw["mlp_conv.base_kernel"][1])
    return x, k, v


def main():
    w = load(f"{DRAFTER}/model.safetensors")
    emb_w = load(f"{TARGET}/model-00003-of-00018.safetensors", ["embed_tokens.weight"])["model.language_model.embed_tokens.weight"]
    lm_w = load(f"{TARGET}/model-00018-of-00018.safetensors", ["lm_head.weight"])["lm_head.weight"]
    hp_w = w["candidate_selector.hidden_projection.weight"]
    preds = w["candidate_selector.predecessor_codebook"]
    succs = w["candidate_selector.successor_codebook"]
    max_rounds = int(os.environ.get("ROUNDS", "8"))
    verbose = os.environ.get("VERBOSE", "1") == "1"
    stage_worst = {}
    for n in range(max_rounds):
        sel_p, ctx_p, inp_p = f"/tmp/vllmsel_{n}.pt", f"/tmp/vllmctx_{n}.pt", f"/tmp/vllmin_{n}.pt"
        if not (os.path.exists(sel_p) and os.path.exists(ctx_p) and os.path.exists(inp_p)):
            break
        sel = torch.load(sel_p, weights_only=False, map_location=DEV)
        ctxd = torch.load(ctx_p, weights_only=False, map_location=DEV)
        inp = torch.load(inp_p, weights_only=False, map_location=DEV)
        rep = {}

        # stage 0: query ids + positions
        qids = sel["query_input_ids"].view(-1)
        qpos = sel["query_positions"].view(-1)
        bonus = int(sel["anchor_token_ids"].view(-1)[0])
        ours_ids = torch.tensor([bonus] + [MASK_ID] * (BLOCK - 1), device=DEV)
        rep["s0_ids"] = 1.0 if torch.equal(ours_ids, qids[:BLOCK].cpu()) else 0.0
        rep["s0_pos"] = 1.0 if torch.equal(qpos[:BLOCK].cpu(), torch.arange(int(qpos[0]), int(qpos[0]) + BLOCK)) else 0.0

        # stage 1: embeddings
        ours_emb = emb_w[ours_ids].to(torch.bfloat16)
        rep["s1_embed"] = cos(ours_emb, sel["embeds"][:BLOCK])

        # context rows: vllmin's combined rows minus the rejected tail
        nr = int(inp["num_rejected"].view(-1)[0]) if inp.get("num_rejected") is not None else 0
        comb = inp["target_hidden_states"].to(DEV, torch.bfloat16)
        vis = max(1, comb.shape[0] - nr)
        ctx_normed = rms(comb[:vis], w["hidden_norm.weight"])
        # cross-check: ctxd context_states == comb (alignment sanity)
        rep["s2_ctxstates"] = cos(ctxd["context_states"], comb[: ctxd["context_states"].shape[0]])
        cpos = ctxd["context_positions"].view(-1)
        # note: the dumped ctx rows may exceed the visible (committed) prefix;
        # compare only the first `vis` rows
        ncmp = min(vis, ctxd["context_states"].shape[0])

        # stage 2: layer-0 context K/V
        lw0 = {k.split("layers.0.")[1]: v for k, v in w.items() if "layers.0." in k}
        our_k = rope_neox(rms((ctx_normed[:ncmp] @ lw0["self_attn.k_proj.weight"].T).view(ncmp, NKV, HD),
                              lw0["self_attn.k_norm.weight"]), cpos[:ncmp])
        rep["s2_ctxK"] = cos(our_k, ctxd["all_k"][0][:ncmp])

        # stage 3: block forward with ctx+block-only KV
        x = ours_emb.clone()
        base = int(qpos[0])
        q_pos = torch.arange(base, base + BLOCK, device=DEV)
        kv_pos = torch.cat([cpos[:ncmp], q_pos])
        layer_cos = []
        for li in range(LAYERS):
            lw = {k.split(f"layers.{li}.")[1]: v for k, v in w.items() if f"layers.{li}." in k}
            x, _, _ = layer_fwd(x, ctx_normed[:ncmp], lw, q_pos, kv_pos)
            layer_cos.append(cos(x, sel["layer_hiddens"][li][:BLOCK]))
        rep["s3_layers"] = min(layer_cos)
        if verbose:
            print(f"r{n}: layers {[f'{c:.4f}' for c in layer_cos]}")

        # stage 4: final normed mask hidden
        hid = rms(x, w["norm.weight"])
        rep["s4_final"] = cos(hid[1:], sel["mask_hidden"].view(BLOCK - 1, H))

        # stage 5: top-16 ids + unary
        logits = hid[1:].float() @ lm_w.float().T
        topv, topi = torch.topk(logits, TOPK, dim=-1)
        cand = sel["candidate_ids"].view(BLOCK - 1, TOPK)
        un = sel["unary_logits"].view(BLOCK - 1, TOPK)
        id_match = sum(torch.equal(topi[s], cand[s].cpu()) for s in range(BLOCK - 1))
        rep["s5_top16_ids"] = id_match / (BLOCK - 1)
        rep["s5_unary"] = cos(topv, un)

        # stage 6: hidden projection + edge lattice
        hp = hid[1:] @ hp_w.T
        anchor = bonus
        prev_ids = torch.cat([torch.tensor([[anchor]] * (BLOCK - 1)), cand[:, :-1]], 1)  # [7,K]
        lat = torch.empty(BLOCK - 1, TOPK, TOPK)
        for s in range(BLOCK - 1):
            lat[s] = (preds[prev_ids[s]].float() * hp[s].float()[None]) @ succs[cand[s]].float().T
        rep["s6_edges"] = cos(lat, sel["scores_lattice"].view(BLOCK - 1, TOPK, TOPK) - un[:, :, None])

        # stage 7: walks vs their draft tokens
        theirs = sel["draft_tokens"].view(-1).tolist()
        temp = sel["temperature"].view(-1).tolist()
        for mode in ("edge", "unary"):
            prev_tok, prev_idx, ours = anchor, 0, []
            for s in range(BLOCK - 1):
                e = lat[s]  # [p, c] edges only
                if mode == "unary":
                    sc = topv[s].float()[None].expand(TOPK, TOPK) * 0  # placeholder
                    sc = topv[s].float()[None, :] + e
                    row = sc[prev_idx]
                else:
                    row = e[prev_idx]
                idx = int(torch.argmax(row))
                ours.append(int(cand[s][idx]))
                prev_idx = idx
                prev_tok = ours[-1]
            rep[f"s7_walk_{mode}"] = sum(o == t for o, t in zip(ours, theirs)) / (BLOCK - 1)
            if verbose and n < 4:
                print(f"r{n} walk_{mode}: ours={ours} theirs={theirs} temp={temp}")

        for k, v in rep.items():
            stage_worst.setdefault(k, 1.0)
            stage_worst[k] = min(stage_worst[k], v)
        print(f"r{n}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rep.items()))
    print("\n=== worst per stage over all rounds ===")
    for k, v in sorted(stage_worst.items()):
        flag = "  <-- DIVERGES" if v < 0.999 else ""
        print(f"{k}: {v:.6f}{flag}")


if __name__ == "__main__":
    main()
