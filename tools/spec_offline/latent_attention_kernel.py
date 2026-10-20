"""Fused split attention over latent key rows on a CUDA device: bf16 tensor-core scores and values with f32 accumulation"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
TARGET_PROGRAMS = 1024
ALIGNMENT = 32
CONFIG = {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2}


@triton.jit
def latent_partial_kernel(query, anchor, table, partial, stats, query_count, scale,
                          LATENT: tl.constexpr, ROPE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    rows = anchor + tl.multiple_of(tl.load(table + split * 3), 16)
    begin = tl.load(table + split * 3 + 1)
    end = tl.load(table + split * 3 + 2)
    width = LATENT + ROPE
    queries = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    present = queries < query_count
    latent_columns = tl.arange(0, LATENT)
    rope_columns = tl.arange(0, ROPE)
    query_latent = tl.load(query + queries[:, None] * width + latent_columns[None, :], mask=present[:, None], other=0.0)
    query_rope = tl.load(query + queries[:, None] * width + LATENT + rope_columns[None, :], mask=present[:, None], other=0.0)
    peak = tl.full([BLOCK_M], float("-inf"), tl.float32)
    total = tl.zeros([BLOCK_M], tl.float32)
    accumulated = tl.zeros([BLOCK_M, LATENT], tl.float32)
    for start in range(begin, end, BLOCK_N):
        keys = start + tl.arange(0, BLOCK_N)
        valid = keys < end
        offsets = keys.to(tl.int64)[:, None] * width
        key_latent = tl.load(rows + offsets + latent_columns[None, :], mask=valid[:, None], other=0.0)
        key_rope = tl.load(rows + offsets + LATENT + rope_columns[None, :], mask=valid[:, None], other=0.0)
        scores = tl.dot(query_latent, tl.trans(key_latent)) + tl.dot(query_rope, tl.trans(key_rope))
        scores = tl.where(valid[None, :], scores * scale, float("-inf"))
        new_peak = tl.maximum(peak, tl.max(scores, 1))
        carry = tl.exp2(peak - new_peak)
        weights = tl.exp2(scores - new_peak[:, None])
        total = total * carry + tl.sum(weights, 1)
        accumulated = accumulated * carry[:, None] + tl.dot(weights.to(tl.bfloat16), key_latent)
        peak = new_peak
    slot = split.to(tl.int64) * query_count + queries
    tl.store(partial + slot[:, None] * LATENT + latent_columns[None, :], accumulated, mask=present[:, None])
    tl.store(stats + slot * 2, peak, mask=present)
    tl.store(stats + slot * 2 + 1, total, mask=present)


def split_rows_for(total_rows, tiles, block_n):
    return max(block_n, triton.cdiv(triton.cdiv(total_rows * tiles, TARGET_PROGRAMS), block_n) * block_n)


def fused_latent_attention(absorbed, query_rope, rows, kv_rank, scale, config=None):
    config = config or CONFIG
    block_m, block_n = config["BLOCK_M"], config["BLOCK_N"]
    parts = [p if p.stride(1) == 1 and p.stride(0) == p.shape[1] else p.contiguous()
             for p in (rows if isinstance(rows, (list, tuple)) else [rows]) if p.shape[0] != 0]
    blocks, heads = absorbed.shape[0], absorbed.shape[1]
    count = blocks * heads
    query = torch.cat((absorbed.float(), query_rope.float()), -1).reshape(count, -1).to(torch.bfloat16).contiguous()
    rope = query.shape[1] - kv_rank
    tiles = triton.cdiv(count, block_m)
    split_rows = split_rows_for(sum(p.shape[0] for p in parts), tiles, block_n)
    parts = [p if p.data_ptr() % ALIGNMENT == 0 else p.clone() for p in parts]
    anchor = parts[0]
    table = [((p.data_ptr() - anchor.data_ptr()) // p.element_size(), begin, min(begin + split_rows, p.shape[0]))
             for p in parts for begin in range(0, p.shape[0], split_rows)]
    table = torch.tensor(table, dtype=torch.int64).to(query.device, non_blocking=True)
    partial = torch.empty((table.shape[0], count, kv_rank), dtype=torch.float32, device=query.device)
    stats = torch.empty((table.shape[0], count, 2), dtype=torch.float32, device=query.device)
    latent_partial_kernel[(tiles, table.shape[0])](query, anchor, table, partial, stats, count, scale * LOG2E,
                                                   LATENT=kv_rank, ROPE=rope, BLOCK_M=block_m, BLOCK_N=block_n,
                                                   num_warps=config["num_warps"], num_stages=config["num_stages"])
    peak = stats[..., 0]
    weight = torch.exp2(peak - peak.amax(0))
    total = (weight * stats[..., 1]).sum(0)
    return (torch.einsum("sm,smr->mr", weight, partial) / total[:, None]).view(blocks, heads, kv_rank)
