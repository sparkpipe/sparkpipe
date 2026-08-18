#!/usr/bin/env python3
"""Forward-only numpy reference of the DSpark drafter (Qwen3.8-27B-DSpark-vLLM).

Pins the drafter math against the HF dflash.py forward: projector fc + hidden_norm
context, block = [embed(C0), embed(mask_token_id) x 6], 5 dual-source decoder layers,
final norm -> shared lm_head -> sequential Markov + BF16-truncated argmax.

Runs with numpy only (no torch). Weights are read directly from the safetensors:
the drafter's own model.safetensors (62 tensors) plus the target's shared lm_head
and embed_tokens rows (from the sharded 27B checkpoint). Inputs are synthesized
(5 random taps + a fixed committed token), so this tests the FORWARD, not the target.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

DRAFTER = Path("/home/spark3/extnvme/models/hf/Doopeworld/Qwen3.8-27B-DSpark-vLLM")
TARGET = Path("/home/spark3/extnvme/models/hf/Qwen/Qwen3.8-27B")

HIDDEN = 5120
N_LAYERS = 5
N_Q_HEADS = 40
N_KV_HEADS = 8
HEAD_DIM = 128
ROPE_DIM = 64
FFN = 10240
VOCAB = 248320
BLOCK = 7
TAPS = 5
TAP_LAYERS = (4, 16, 28, 40, 52)
MARKOV_RANK = 256
MASK_TOKEN_ID = 248077
EPS = 1e-6
ROPE_THETA = 1e7
BASE_POS = 64  # arbitrary: the first draft position


# ---- bf16 <-> fp32 ----
def bf16_to_f32(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << np.uint32(16)).view(np.float32)


def bf16(x):
    return bf16_to_f32(f32_to_bf16(x.astype(np.float32)))


def f32_to_bf16(f: np.ndarray) -> np.ndarray:
    u = f.view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    u = u + np.uint32(0x7FFF) + lsb
    return (u >> np.uint32(16)).astype(np.uint16)


# ---- safetensors reader ----
def read_safetensors_tensor(path: Path, name: str) -> np.ndarray:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n))
        if name not in meta:
            raise KeyError(f"{name} not in {path}")
        start, end = meta[name]["data_offsets"]
        dtype = meta[name]["dtype"]
        shape = meta[name]["shape"]
        fh.seek(8 + n + start)
        raw = fh.read(end - start)
    assert dtype == "BF16", (name, dtype)
    return np.frombuffer(raw, dtype=np.uint16).reshape(shape)


def load_drafter() -> dict:
    """Return {kind: weight} for every drafter tensor, as fp32 numpy arrays."""
    st = DRAFTER / "model.safetensors"
    names = {
        "fc.weight", "norm.weight", "hidden_norm.weight",
        "markov_head.markov_w1.weight", "markov_head.markov_w2.weight",
    }
    for L in range(N_LAYERS):
        for n in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                  "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                  "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                  "input_layernorm.weight", "post_attention_layernorm.weight",
                  "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
            names.add(f"layers.{L}.{n}")
    w = {}
    for n in sorted(names):
        w[n] = bf16_to_f32(read_safetensors_tensor(st, n)).copy()
    return w


def load_target_shared() -> tuple[np.ndarray, np.ndarray]:
    """Return (lm_head, embed_tokens) fp32; embed_tokens is full, lm_head full."""
    idx = json.loads((TARGET / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    lm_file = wmap["lm_head.weight"]
    emb_file = wmap["model.language_model.embed_tokens.weight"]
    lm = bf16_to_f32(read_safetensors_tensor(TARGET / lm_file, "lm_head.weight")).copy()
    emb = bf16_to_f32(read_safetensors_tensor(TARGET / emb_file, "model.language_model.embed_tokens.weight")).copy()
    return lm, emb


def rms_norm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    var = (x * x).mean(axis=-1, keepdims=True)
    return (x * (var + EPS) ** -0.5) * weight


def rope_freq() -> np.ndarray:
    pairs = np.arange(ROPE_DIM // 2, dtype=np.float64)
    return ROPE_THETA ** (-2.0 * pairs / ROPE_DIM)


def apply_rope(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """x: [..., head_dim]; rope over the first ROPE_DIM dims. pos: [...] broadcast."""
    freq = rope_freq().astype(np.float64)  # [32]
    ang = np.asarray(pos, dtype=np.float64)[..., None] * freq[None, :]  # [..., 32]
    c = np.cos(ang).astype(np.float32)
    s = np.sin(ang).astype(np.float32)
    out = x.copy()
    # COPY (not view): the in-place stores below would otherwise alias xr back
    # into the just-written even slots, feeding ROTATED (not original) even
    # values into the odd computation. The CUDA kernel reads re/im before
    # overwriting, so this must too.
    xr = out[..., 0:ROPE_DIM:2].copy()
    xi = out[..., 1:ROPE_DIM:2].copy()
    out[..., 0:ROPE_DIM:2] = xr * c - xi * s
    out[..., 1:ROPE_DIM:2] = xr * s + xi * c
    return out


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def forward_layer(x_block, ctx, lw, positions_q, pos_ctx):
    """One decoder layer. x_block: [7, H]; ctx: [H] (context feature)."""
    h = bf16(rms_norm(x_block, lw["input_layernorm.weight"]))  # [7, H]
    # Q
    q = bf16(h @ lw["self_attn.q_proj.weight"].T)  # [7, 5120]
    q = q.reshape(BLOCK, N_Q_HEADS, HEAD_DIM)
    q = bf16(rms_norm(q, lw["self_attn.q_norm.weight"]))  # per-head norm [7,40,128]
    q = apply_rope(q, positions_q[:, None])  # [7,40,128]
    # K/V: context (1) + block (7)
    k_ctx = bf16(ctx[None, :] @ lw["self_attn.k_proj.weight"].T)  # [1, 1024]
    k_noise = bf16(h @ lw["self_attn.k_proj.weight"].T)  # [7, 1024]
    v_ctx = bf16(ctx[None, :] @ lw["self_attn.v_proj.weight"].T)  # [1, 1024]
    v_noise = bf16(h @ lw["self_attn.v_proj.weight"].T)  # [7, 1024]
    k = np.concatenate([k_ctx, k_noise], axis=0).reshape(BLOCK + 1, N_KV_HEADS, HEAD_DIM)
    v = np.concatenate([v_ctx, v_noise], axis=0).reshape(BLOCK + 1, N_KV_HEADS, HEAD_DIM)
    k = bf16(rms_norm(k, lw["self_attn.k_norm.weight"]))  # per-head norm [8,8,128]
    # rope: context at pos_ctx, block at positions_q
    k_pos = np.concatenate([np.array([pos_ctx]), positions_q])  # [8]
    k = apply_rope(k, k_pos[:, None])  # [8,8,128]
    # GQA attention (non-causal): Q [7,40,128], K/V [8,8,128]
    scale = HEAD_DIM ** -0.5
    attn_out = np.zeros((BLOCK, N_Q_HEADS, HEAD_DIM), dtype=np.float32)
    for qi in range(BLOCK):
        for qh in range(N_Q_HEADS):
            kvh = qh // (N_Q_HEADS // N_KV_HEADS)  # 5 Q heads per KV head
            scores = (q[qi, qh] @ k[:, kvh].T) * scale  # [8]
            scores = scores - scores.max()
            p = np.exp(scores)
            p = p / p.sum()
            attn_out[qi, qh] = p @ v[:, kvh]
    attn_out = bf16(attn_out.reshape(BLOCK, N_Q_HEADS * HEAD_DIM))
    attn_out = bf16(attn_out @ lw["self_attn.o_proj.weight"].T)  # [7, H]
    x = bf16(x_block + attn_out)
    h = bf16(rms_norm(x, lw["post_attention_layernorm.weight"]))
    gate = bf16(h @ lw["mlp.gate_proj.weight"].T)
    up = bf16(h @ lw["mlp.up_proj.weight"].T)
    ff = bf16(silu(gate) * up)
    ff = bf16(ff @ lw["mlp.down_proj.weight"].T)
    return bf16(x + ff)


def markov_bias(w1, w2, prev: int) -> np.ndarray:
    return (w2 @ w1[prev]).astype(np.float32)  # [V]


def main() -> int:
    rng = np.random.default_rng(42)
    drafter = load_drafter()
    lm_head, embed_tokens = load_target_shared()

    # Synthesized inputs
    taps = rng.standard_normal((TAPS, HIDDEN)).astype(np.float32) * 0.1  # 5 taps
    c0 = 12345  # fixed committed token id
    positions_q = np.arange(BASE_POS, BASE_POS + BLOCK, dtype=np.float32)  # [7]
    pos_ctx = BASE_POS - 1.0  # committed position

    # 1) context = hidden_norm(fc(cat(5 taps)))
    ctx = drafter["fc.weight"] @ taps.reshape(-1)  # [H]
    ctx = rms_norm(ctx, drafter["hidden_norm.weight"])

    # 2) block = [embed(C0), embed(mask) x 6]
    block = np.empty((BLOCK, HIDDEN), dtype=np.float32)
    block[0] = embed_tokens[c0]
    block[1:] = embed_tokens[MASK_TOKEN_ID]

    # 3) 5 layers
    x = block
    for L in range(N_LAYERS):
        lw = {k.split(f"layers.{L}.")[1]: drafter[k] for k in drafter if f"layers.{L}." in k}
        x = forward_layer(x, ctx, lw, positions_q, pos_ctx)

    # 4) final norm -> lm_head -> base logits (BF16)
    x = rms_norm(x, drafter["norm.weight"])  # [7, H]
    base_logits = (x @ lm_head.T).astype(np.float32)  # [7, V]
    base_logits = bf16_to_f32(f32_to_bf16(base_logits))  # -> BF16

    # 5) sequential Markov + argmax
    w1 = drafter["markov_head.markov_w1.weight"]  # [V, R]
    w2 = drafter["markov_head.markov_w2.weight"]  # [V, R]
    prev = c0
    drafts = []
    for i in range(BLOCK):
        bias = markov_bias(w1, w2, prev)  # [V] fp32
        bias = bf16_to_f32(f32_to_bf16(bias))  # bias -> BF16
        logits = base_logits[i] + bias
        logits = bf16_to_f32(f32_to_bf16(logits))  # add -> BF16
        prev = int(np.argmax(logits))
        drafts.append(prev)

    print("draft_tokens =", drafts)
    # emit a few diagnostic values for cross-checking
    print("base_logits[0][:4] =", base_logits[0][:4].tolist())
    print("block[0][:4] =", block[0][:4].tolist())
    print("ctx[:4] =", ctx[:4].tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
