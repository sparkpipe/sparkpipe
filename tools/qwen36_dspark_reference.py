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
import os
import struct
import sys
from pathlib import Path

import numpy as np

DRAFTER = Path(os.environ.get("SPARK_QWEN36_DFLASH2_DRAFTER", "/home/spark3/sparkdata/qwen38-dflash2-drafter"))
TARGET = Path(os.environ.get("SPARK_QWEN36_DFLASH2_TARGET", "/home/spark3/extnvme/models/hf/Qwen/Qwen3.8-27B"))

HIDDEN = 5120
N_LAYERS = 5
N_Q_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
ROPE_DIM = 64
FFN = 17408
VOCAB = 248320
BLOCK = 8
TAPS = 5
TAP_LAYERS = (5, 19, 33, 47, 61)
SELECTOR_RANK = 256
SELECTOR_TOP_K = 16
CONV_KERNEL_SIZE = 2
CONV_GROUP_SIZE = 16
SLIDING_WINDOW = 2048
MASK_TOKEN_ID = 248070
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
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
        "candidate_selector.hidden_projection.weight",
    }
    for L in range(N_LAYERS):
        for n in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                  "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                  "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                  "input_layernorm.weight", "post_attention_layernorm.weight",
                  "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                  "attention_conv.base_kernel", "attention_conv.kernel_projection.weight",
                  "mlp_conv.base_kernel", "mlp_conv.kernel_projection.weight"):
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
    """x: [..., head_dim]; interleaved rope over the first ROPE_DIM dims
    (pairs (2i, 2i+1), theta^(-2i/64)) - the empirically winning drafter
    convention (the neox-128 variant wins p0 but zeroes deep drafts)."""
    freq = rope_freq().astype(np.float64)  # [32]
    ang = np.asarray(pos, dtype=np.float64)[..., None] * freq[None, :]
    c = np.cos(ang).astype(np.float32)
    s = np.sin(ang).astype(np.float32)
    out = x.copy()
    xr = out[..., 0:ROPE_DIM:2].copy()
    xi = out[..., 1:ROPE_DIM:2].copy()
    out[..., 0:ROPE_DIM:2] = xr * c - xi * s
    out[..., 1:ROPE_DIM:2] = xr * s + xi * c
    return out


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def conv_prepare(h, lw, module):
    """DFlashGroupedConv.prepare: one projection -> both sides. Returns (h2, coeff_finish)."""
    kp = lw[f"{module}.kernel_projection.weight"]  # [1280, H]
    base = lw[f"{module}.base_kernel"]  # [2, 2, H] (sides, taps, channels)
    T = h.shape[0]
    num_groups = HIDDEN // CONV_GROUP_SIZE
    delta_all = (h @ kp.T).reshape(T, 2, CONV_KERNEL_SIZE, num_groups)  # [T, 2, 2, 320]
    h2 = _grouped_conv(h, delta_all[:, 0], base[0], BLOCK, num_groups, CONV_GROUP_SIZE, CONV_KERNEL_SIZE)
    return bf16(h2), delta_all[:, 1]


def conv_finish(h, coeff, lw, module):
    """DFlashGroupedConv.finish: apply the side-1 conv with the stored coefficients."""
    base = lw[f"{module}.base_kernel"]
    num_groups = HIDDEN // CONV_GROUP_SIZE
    h2 = _grouped_conv(h, coeff, base[1], BLOCK, num_groups, CONV_GROUP_SIZE, CONV_KERNEL_SIZE)
    return bf16(h2)


def forward_layer(x_block, ctx, lw, positions_q, pos_ctx):
    """One DFlash2 decoder layer: conv-wrapped attention + MLP, sliding non-causal."""
    h = bf16(rms_norm(x_block, lw["input_layernorm.weight"]))
    h, coeff = conv_prepare(h, lw, "attention_conv")
    q = bf16(h @ lw["self_attn.q_proj.weight"].T)  # [B, 4096]
    q = q.reshape(BLOCK, N_Q_HEADS, HEAD_DIM)
    q = bf16(rms_norm(q, lw["self_attn.q_norm.weight"]))
    q = apply_rope(q, positions_q[:, None])
    k_ctx = bf16(ctx[None, :] @ lw["self_attn.k_proj.weight"].T)
    k_noise = bf16(h @ lw["self_attn.k_proj.weight"].T)
    v_ctx = bf16(ctx[None, :] @ lw["self_attn.v_proj.weight"].T)
    v_noise = bf16(h @ lw["self_attn.v_proj.weight"].T)
    k = np.concatenate([k_ctx, k_noise], axis=0).reshape(BLOCK + 1, N_KV_HEADS, HEAD_DIM)
    v = np.concatenate([v_ctx, v_noise], axis=0).reshape(BLOCK + 1, N_KV_HEADS, HEAD_DIM)
    k = bf16(rms_norm(k, lw["self_attn.k_norm.weight"]))
    k_pos = np.concatenate([np.array([pos_ctx]), positions_q])
    k = apply_rope(k, k_pos[:, None])
    scale = HEAD_DIM ** -0.5
    attn_out = np.zeros((BLOCK, N_Q_HEADS, HEAD_DIM), dtype=np.float32)
    for qi in range(BLOCK):
        for qh in range(N_Q_HEADS):
            kvh = qh // (N_Q_HEADS // N_KV_HEADS)
            scores = (q[qi, qh] @ k[:, kvh].T) * scale
            scores = scores - scores.max()
            p = np.exp(scores)
            p = p / p.sum()
            attn_out[qi, qh] = p @ v[:, kvh]
    attn_out = bf16(attn_out.reshape(BLOCK, N_Q_HEADS * HEAD_DIM))
    attn_out = bf16(attn_out @ lw["self_attn.o_proj.weight"].T)
    h = conv_finish(attn_out, coeff, lw, "attention_conv")
    x = bf16(x_block + h)
    h = bf16(rms_norm(x, lw["post_attention_layernorm.weight"]))
    h, coeff = conv_prepare(h, lw, "mlp_conv")
    gate = bf16(h @ lw["mlp.gate_proj.weight"].T)
    up = bf16(h @ lw["mlp.up_proj.weight"].T)
    ff = bf16(silu(gate) * up)
    ff = bf16(ff @ lw["mlp.down_proj.weight"].T)
    h = conv_finish(ff, coeff, lw, "mlp_conv")
    return bf16(x + h)


# ---- DFlash2: EXACT port of the vLLM PR #52816 oracles ----
# test_grouped_conv_matches_reference + test_selector_edges_match_sequential_reference
# are the only executable ground truth (no upstream reference impl exists). These
# numpy ports must match the sequential references before W2/W3 kernels are written.


def _grouped_conv(hidden, delta, base, block_size, num_groups, group_size, taps):
    """Exact port of vLLM PR #52816 _grouped_conv.

    hidden: [T, num_groups*group_size]; delta: [T, taps, num_groups];
    base: [taps, num_groups*group_size]. Grouped depthwise conv with hard zeroing
    across the block boundary (pos = i % block_size >= tap).
    """
    T = hidden.shape[0]
    blocks = hidden.reshape(T, num_groups, group_size)
    coefficients = base.reshape(1, taps, num_groups, group_size) + delta[:, :, :, None]
    output = coefficients[:, 0] * blocks
    position = np.arange(T, dtype=np.int64)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = np.concatenate(
            [np.zeros((tap, num_groups, group_size), dtype=hidden.dtype), blocks[:-tap]],
            axis=0,
        )
        output += coefficients[:, tap] * shifted * (position >= tap).reshape(-1, 1, 1)
    return output.reshape(T, -1)


def _score_edges(predecessor_table, successor_table, candidate_ids, unary_logits, hidden, anchor_token_ids, top_k):
    """Exact port of vLLM PR #52816 _score_edges.

    predecessor/successor_table: [vocab, rank]; candidate_ids: [B, steps, K] int;
    unary_logits: [B, steps, K]; hidden: [B, steps, rank] (already projected);
    anchor_token_ids: [B] int. Returns [B, steps, K, K].
    """
    successors = successor_table[candidate_ids]
    anchor_p = np.broadcast_to(
        anchor_token_ids[:, None, None], (anchor_token_ids.shape[0], 1, top_k)
    )
    predecessor_ids = np.concatenate([anchor_p, candidate_ids[:, :-1]], axis=1)
    predecessors = predecessor_table[predecessor_ids]
    gate = predecessors * hidden[:, :, None, :]
    edges = np.einsum("blpr,blcr->blpc", gate, successors)
    return unary_logits[:, :, None, :] + edges


def _greedy_walk(scores, candidate_ids):
    """Greedy path walk: previous=0 (anchor), argmax-first-max per slot.

    scores: [steps, K, K]; candidate_ids: [steps, K]. Returns [steps] draft ids.
    """
    steps = scores.shape[0]
    drafts = []
    previous = 0
    for step in range(steps):
        row = scores[step, previous]
        nxt = int(np.min(np.where(row == row.max())[0]))
        drafts.append(int(candidate_ids[step, nxt]))
        previous = nxt
    return drafts


def test_grouped_conv_matches_reference(block_size):
    rng = np.random.default_rng(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = rng.standard_normal((batch * block_size, num_groups * group_size)).astype(np.float32)
    delta = rng.standard_normal((batch * block_size, taps, num_groups)).astype(np.float32)
    base = rng.standard_normal((taps, num_groups * group_size)).astype(np.float32)
    actual = _grouped_conv(hidden, delta, base, block_size, num_groups, group_size, taps)
    hidden_blocks = hidden.reshape(batch, block_size, num_groups, group_size)
    base_b = base.reshape(taps, num_groups, group_size)
    delta_b = delta.reshape(batch, block_size, taps, num_groups)
    expected = np.zeros((batch, block_size, num_groups, group_size), dtype=np.float32)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base_b[tap] + delta_b[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]
    np.testing.assert_allclose(actual, expected.reshape(batch * block_size, num_groups * group_size), rtol=1e-5, atol=1e-5)


def test_selector_edges_match_sequential_reference():
    rng = np.random.default_rng(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = rng.standard_normal((vocab, rank)).astype(np.float32)
    successors = rng.standard_normal((vocab, rank)).astype(np.float32)
    candidate_ids = rng.integers(0, vocab, (batch, steps, top_k))
    unary = rng.standard_normal((batch, steps, top_k)).astype(np.float32)
    hidden = rng.standard_normal((batch, steps, rank)).astype(np.float32)
    anchors = rng.integers(0, vocab, (batch,))
    actual = _score_edges(predecessors, successors, candidate_ids, unary, hidden, anchors, top_k)
    expected = np.empty_like(actual)
    for step in range(steps):
        pred = (
            np.broadcast_to(anchors[:, None], (batch, top_k))
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + np.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


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

    # 2) block = [embed(C0), embed(mask) x 7]
    block = np.empty((BLOCK, HIDDEN), dtype=np.float32)
    block[0] = embed_tokens[c0]
    block[1:] = embed_tokens[MASK_TOKEN_ID]

    # 3) 5 layers (conv-wrapped)
    x = block
    for L in range(N_LAYERS):
        lw = {k.split(f"layers.{L}.")[1]: drafter[k] for k in drafter if f"layers.{L}." in k}
        x = forward_layer(x, ctx, lw, positions_q, pos_ctx)

    # 4) final norm -> hidden [8, H]
    hidden = rms_norm(x, drafter["norm.weight"])

    # 5) top-16 over the 7 mask-position lm_head logits (the C0 slot is the anchor)
    mask_logits = (hidden[1:] @ lm_head.T).astype(np.float32)  # [7, V]
    mask_logits = bf16_to_f32(f32_to_bf16(mask_logits))
    top_ids = np.argsort(-mask_logits, axis=-1, kind="stable")[:, :SELECTOR_TOP_K]  # [7, K]; stable = deterministic id-asc tie-break, matching the kernel
    unary = np.take_along_axis(mask_logits, top_ids, axis=-1)  # [7, K]

    # 6) selector: hidden_projection(hidden[1:]) -> score edges -> greedy walk
    hproj = hidden[1:] @ drafter["candidate_selector.hidden_projection.weight"].T  # [7, R]
    hproj = bf16_to_f32(f32_to_bf16(hproj))
    scores = _score_edges(
        drafter["candidate_selector.predecessor_codebook"],
        drafter["candidate_selector.successor_codebook"],
        top_ids[None],
        unary[None],
        hproj[None],
        np.array([c0]),
        SELECTOR_TOP_K,
    )[0]  # [7, K, K]
    drafts = _greedy_walk(scores, top_ids)

    print("draft_tokens =", drafts)
    print("ctx[:4] =", ctx[:4].tolist())
    print("hidden[0][:4] =", hidden[0][:4].tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
