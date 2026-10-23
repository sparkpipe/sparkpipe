#!/usr/bin/env python3
"""DSV4.1 1017's DSpark drafter with device-only rounds replayed as CUDA graphs, for the hub draft server.

Same weights, round protocol and contexts as dspark_drafter.DsparkDrafter. The math differs from the eager
reference only at rounding level:
- routed experts multiply straight from MXFP4 in a Triton GEMV with fp32 accumulation;
- the head and the Markov head multiply in bf16 with fp32 output;
- ring keys outside the window are masked instead of dropped;
- the grouped output projection is a bf16 batched matmul.

draft() replays one captured graph per sampling mode; ingest() replays one graph per row count up to
INGEST_GRAPH_ROWS, in chunks of that many rows; warm() captures them all. `python3 dspark_fast.py --checkpoint DIR` compares the fast
rounds with the eager ones on random taps and prints the round times.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_drafter import coupled_choice  # noqa: E402
from dspark_drafter import STAGES, DsparkDrafter, act_qdq, qlinear, rms_norm  # noqa: E402

INGEST_GRAPH_ROWS = 16
GEMV_BLOCK_N = 16
GEMV_GROUPS = 8
SPLIT_WIDTH = 512


@triton.jit
def _e2m1(code):
    c = code.to(tl.int32)
    exponent = (c >> 1) & 3
    mantissa = c & 1
    bits = tl.where(exponent != 0, ((exponent + 126) << 23) | (mantissa << 22), (mantissa * 126) << 23)
    return (bits | ((c & 8) << 28)).to(tl.float32, bitcast=True)


@triton.jit
def _mxfp4_gemv(x_ptr, x_rows_ptr, ids_ptr, w_ptr, s_ptr, out_ptr, N, K, stride_we, stride_wn, stride_se, stride_sn,
                BLOCK_N: tl.constexpr, GROUPS: tl.constexpr):
    pair = tl.program_id(0)
    block = tl.program_id(1)
    expert = tl.load(ids_ptr + pair).to(tl.int64)
    row = tl.load(x_rows_ptr + pair).to(tl.int64)
    n = block * BLOCK_N + tl.arange(0, BLOCK_N)
    inside = n < N
    g = tl.arange(0, GROUPS)
    j = tl.arange(0, 16)
    w_base = w_ptr + expert * stride_we + n[:, None, None].to(tl.int64) * stride_wn
    s_base = s_ptr + expert * stride_se + n[:, None].to(tl.int64) * stride_sn
    total = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start in range(0, K // 32, GROUPS):
        packed = tl.load(w_base + (start + g)[None, :, None] * 16 + j[None, None, :], mask=inside[:, None, None], other=0)
        scale = (tl.load(s_base + (start + g)[None, :], mask=inside[:, None], other=127).to(tl.int32) << 23).to(tl.float32, bitcast=True)
        xs = tl.load(x_ptr + row * K + (start + g)[:, None] * 32 + tl.arange(0, 32)[None, :]).to(tl.float32)
        even, odd = tl.split(tl.reshape(xs, [GROUPS, 16, 2]))
        part = tl.sum(_e2m1(packed & 15) * even[None, :, :] + _e2m1(packed >> 4) * odd[None, :, :], axis=2)
        total += tl.sum(part * scale, axis=1)
    tl.store(out_ptr + pair * N + n, total, mask=inside)


def mxfp4_gemv(x, x_rows, ids, weight, scale):
    pairs, n, k = ids.shape[0], weight.shape[1], x.shape[1]
    out = torch.empty(pairs, n, dtype=torch.float32, device=x.device)
    assert (k // 32) % GEMV_GROUPS == 0 and weight.shape[2] * 2 == k and scale.shape[2] * 32 == k
    _mxfp4_gemv[(pairs, triton.cdiv(n, GEMV_BLOCK_N))](x, x_rows, ids, weight, scale, out, n, k, weight.stride(0), weight.stride(1),
                                                       scale.stride(0), scale.stride(1), BLOCK_N=GEMV_BLOCK_N, GROUPS=GEMV_GROUPS, num_warps=4)
    return out


def split_rows(weight, width):
    return weight.view(weight.shape[0], -1, width).transpose(0, 1).contiguous()


def matvec_split(x, split):
    return torch.bmm(x.view(x.shape[0], split.shape[0], -1).transpose(0, 1), split.transpose(1, 2)).sum(0)


def mm32(x, weight):
    return torch.mm(x, weight.t(), out_dtype=torch.float32)


class DsparkFast(DsparkDrafter):
    def __init__(self, checkpoint_directory, device="cuda", keep_reference=False, compiled=True):
        super().__init__(checkpoint_directory, device)
        self.compiled = compiled
        for s in self.stages:
            for index, name in enumerate(("w1", "w2", "w3")):
                s[name] = torch.stack([packed[index][0] for packed in s["experts"]])
                s["s" + name[1]] = torch.stack([packed[index][1] for packed in s["experts"]])
            s["experts"] = [[(s[name][expert], s["s" + name[1]][expert]) for name in ("w1", "w2", "w3")] for expert in range(self.experts)]
            s["wo_a_t"] = s["wo_a"].transpose(1, 2).contiguous()
            for name in ("hc_attn_fn", "hc_ffn_fn", "gate"):
                s[name + "_split"] = split_rows(s[name], SPLIT_WIDTH)
        self.head16 = self.head.to(torch.bfloat16)
        self.markov16 = self.markov_head.to(torch.bfloat16)
        if not keep_reference:
            self.head = self.markov_head = None
        torch.cuda.empty_cache()
        rows, pairs = self.block, self.block * self.topk
        self.pair_rows = torch.arange(rows, device=self.device, dtype=torch.int32).repeat_interleave(self.topk)
        self.pair_index = torch.arange(pairs, device=self.device, dtype=torch.int32)
        self.offsets = torch.arange(rows, device=self.device, dtype=torch.int64)
        self.noise_ids = torch.full((rows,), self.noise, dtype=torch.int64, device=self.device)
        self.block_valid = torch.ones(rows, dtype=torch.bool, device=self.device)
        self.static_keys = torch.zeros(STAGES, self.window, self.head_dim, dtype=torch.bfloat16, device=self.device)
        self.static_positions = torch.full((self.window,), -1, dtype=torch.int64, device=self.device)
        self.static_token = torch.zeros((), dtype=torch.int64, device=self.device)
        self.static_position = torch.zeros((), dtype=torch.int64, device=self.device)
        self.static_rule = (torch.zeros((), dtype=torch.int64, device=self.device), torch.ones((), device=self.device),
                            torch.zeros((), device=self.device), torch.zeros((), dtype=torch.int64, device=self.device))
        self.out_tokens = torch.zeros(rows, dtype=torch.int64, device=self.device)
        self.out_confidence = torch.zeros(rows, dtype=torch.float32, device=self.device)
        self.ingest_taps = torch.zeros(INGEST_GRAPH_ROWS, len(self.tap_names) * self.dim, dtype=torch.bfloat16, device=self.device)
        self.ingest_positions = torch.zeros(INGEST_GRAPH_ROWS, dtype=torch.int64, device=self.device)
        self.ingest_keys = torch.zeros(STAGES, INGEST_GRAPH_ROWS, self.head_dim, dtype=torch.bfloat16, device=self.device)
        self.draft_graphs = {}
        self.ingest_graphs = {}
        torch.cuda.synchronize(self.device)

    def hc_mixes_fast(self, h, s, kind):
        flat = h.flatten(1).to(torch.float32)
        mixes = matvec_split(flat, s[f"hc_{kind}_fn_split"]) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        scale, base, n = s[f"hc_{kind}_scale"], s[f"hc_{kind}_base"], self.hc
        pre = torch.sigmoid(mixes[:, :n] * scale[0] + base[:n]) + self.hc_eps
        post = 2.0 * torch.sigmoid(mixes[:, n:2 * n] * scale[1] + base[n:2 * n])
        comb = (mixes[:, 2 * n:] * scale[2] + base[2 * n:]).view(-1, n, n)
        comb = torch.softmax(comb, -1) + self.hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        return pre, post, comb

    def moe_fast(self, s, x):
        rows = x.shape[0]
        pairs = rows * self.topk
        scores = F.softplus(matvec_split(x.to(torch.float32), s["gate_split"])).sqrt()
        chosen = (scores + s["gate_bias"]).topk(self.topk, -1)[1]
        routing = scores.gather(1, chosen)
        routing = routing / (routing.sum(-1, keepdim=True) + 1e-20) * self.route_scale
        ids = chosen.flatten().to(torch.int32)
        xq = act_qdq(x)
        gate = mxfp4_gemv(xq, self.pair_rows[:pairs], ids, s["w1"], s["s1"]).to(x.dtype).to(torch.float32)
        up = mxfp4_gemv(xq, self.pair_rows[:pairs], ids, s["w3"], s["s3"]).to(x.dtype).to(torch.float32)
        if self.swiglu_limit > 0:
            up = torch.clamp(up, -self.swiglu_limit, self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        hidden = act_qdq((routing.reshape(pairs, 1) * (F.silu(gate) * up)).to(x.dtype))
        down = mxfp4_gemv(hidden, self.pair_index[:pairs], ids, s["w2"], s["s2"]).to(x.dtype).to(torch.float32)
        y = down.view(rows, self.topk, self.dim).sum(1)
        y += self.expert(x, s["shared"], None).to(torch.float32)
        return y.to(x.dtype)

    def attention_fast(self, s, x, positions, ring, valid):
        rows = x.shape[0]
        qr = rms_norm(qlinear(x, s["wq_a"]), s["q_norm"], self.eps)
        q = qlinear(qr, s["wq_b"]).view(rows, self.heads, self.head_dim)
        q = torch.cat((q[..., :-self.rope_dim], self.rope.apply(q[..., -self.rope_dim:], positions)), -1)
        keys = rms_norm(qlinear(x, s["wkv"]), s["kv_norm"], self.eps)
        keys = torch.cat((keys[:, :-self.rope_dim], self.rope.apply(keys[:, -self.rope_dim:], positions)), -1)
        keys = torch.cat((ring, act_qdq(keys)), 0)
        mask = torch.cat((valid, self.block_valid[:rows]))
        scores = torch.einsum("rhd,kd->rhk", q.to(torch.float32), keys.to(torch.float32)) * self.softmax_scale
        scores = scores.masked_fill(~mask, float("-inf"))
        top = torch.maximum(scores.amax(-1, keepdim=True), torch.full_like(scores[..., :1], -1e30))
        weights = torch.exp(scores - top)
        total = weights.sum(-1, keepdim=True) + torch.exp(s["sink"].view(1, -1, 1) - top)
        output = torch.einsum("rhk,kd->rhd", weights.to(torch.bfloat16).to(torch.float32), keys.to(torch.float32)) / total
        output = output.to(torch.bfloat16)
        output = torch.cat((output[..., :-self.rope_dim], self.rope.apply(output[..., -self.rope_dim:], positions, True)), -1)
        grouped = act_qdq(output.view(rows, self.groups, -1))
        projected = torch.bmm(grouped.transpose(0, 1), s["wo_a_t"]).transpose(0, 1)
        return qlinear(projected.flatten(1), s["wo_b"])

    def forward_fast(self):
        rows = self.block
        position = self.static_position
        positions = position + self.offsets
        ids = torch.cat((self.static_token.view(1), self.noise_ids[1:]))
        h = self.embed[ids].unsqueeze(1).repeat(1, self.hc, 1)
        pre = torch.zeros(rows, self.hc, dtype=torch.float32, device=self.device)
        pre[:, 0] = 1.0
        known = self.static_positions
        valid = (known >= 0) & (known < position) & (known >= position - self.window)
        for stage, s in enumerate(self.stages):
            residual = h
            attn_pre, attn_post, attn_comb = self.hc_mixes_fast(h, s, "attn")
            x = rms_norm(self.hc_pre(h, pre), s["attn_norm"], self.eps)
            x = self.attention_fast(s, x, positions, self.static_keys[stage], valid)
            h = self.hc_post(x, residual, attn_post, attn_comb)
            residual = h
            ffn_pre, ffn_post, ffn_comb = self.hc_mixes_fast(h, s, "ffn")
            x = rms_norm(self.hc_pre(h, attn_pre), s["ffn_norm"], self.eps)
            x = self.moe_fast(s, x)
            h = self.hc_post(x, residual, ffn_post, ffn_comb)
            pre = ffn_pre
        hidden = self.hc_pre(h, pre)
        return hidden, mm32(rms_norm(hidden, self.final_norm, self.eps), self.head16)

    def forward_reference(self, context, token, position):
        rows = self.block
        positions = torch.arange(position, position + rows, device=self.device)
        ids = torch.full((rows,), self.noise, dtype=torch.int64, device=self.device)
        ids[0] = int(token)
        h = self.embed[ids].unsqueeze(1).repeat(1, self.hc, 1)
        pre = torch.zeros(rows, self.hc, dtype=torch.float32, device=self.device)
        pre[:, 0] = 1.0
        valid = (context.positions >= 0) & (context.positions < position) & (context.positions >= position - self.window)
        for stage, s in enumerate(self.stages):
            residual = h
            attn_pre, attn_post, attn_comb = self.hc_mixes(h, s["hc_attn_fn"], s["hc_attn_scale"], s["hc_attn_base"])
            x = rms_norm(self.hc_pre(h, pre), s["attn_norm"], self.eps)
            x = self.attention(s, x, positions, context.keys[stage], valid)
            h = self.hc_post(x, residual, attn_post, attn_comb)
            residual = h
            ffn_pre, ffn_post, ffn_comb = self.hc_mixes(h, s["hc_ffn_fn"], s["hc_ffn_scale"], s["hc_ffn_base"])
            x = rms_norm(self.hc_pre(h, attn_pre), s["ffn_norm"], self.eps)
            x = self.moe(s, x)
            h = self.hc_post(x, residual, ffn_post, ffn_comb)
            pre = ffn_pre
        hidden = self.hc_pre(h, pre)
        return hidden, F.linear(rms_norm(hidden, self.final_norm, self.eps).to(torch.float32), self.head)

    def draft_body(self, sampled):
        rows = self.block
        position = self.static_position
        hidden, logits = self.forward_fast()
        previous = self.static_token
        embeddings = []
        for slot in range(rows):
            embedding = torch.index_select(self.markov_embed, 0, previous.view(1)).view(-1)
            embeddings.append(embedding)
            row = logits[slot] + mm32(embedding.view(1, -1), self.markov16).view(-1)
            previous = coupled_choice(row, self.static_rule, position + slot) if sampled else torch.argmax(row)
            self.out_tokens[slot] = previous
        features = torch.cat((hidden, torch.stack(embeddings)), -1).to(torch.float32)
        self.out_confidence.copy_(torch.sigmoid(F.linear(features, self.confidence).squeeze(-1)))

    def ingest_body(self, rows):
        taps = self.ingest_taps[:rows]
        positions = self.ingest_positions[:rows]
        main = rms_norm(qlinear(taps, self.main_proj), self.main_norm, self.eps)
        for stage, s in enumerate(self.stages):
            keys = rms_norm(qlinear(main, s["wkv"]), s["kv_norm"], self.eps)
            keys = torch.cat((keys[:, :-self.rope_dim], self.rope.apply(keys[:, -self.rope_dim:], positions)), -1)
            self.ingest_keys[stage, :rows] = act_qdq(keys)

    def capture(self, body):
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(2):
                body()
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        return graph

    @torch.inference_mode()
    def ingest(self, context, positions, taps):
        positions = list(positions)
        for first in range(0, len(positions), INGEST_GRAPH_ROWS):
            self.ingest_rows(context, positions[first:first + INGEST_GRAPH_ROWS], taps[first:first + INGEST_GRAPH_ROWS])

    def ingest_rows(self, context, positions, taps):
        rows = len(positions)
        if rows not in self.ingest_graphs:
            self.ingest_graphs[rows] = self.capture(lambda: self.ingest_body(rows))
        self.ingest_positions[:rows].copy_(torch.as_tensor(positions, dtype=torch.int64), non_blocking=True)
        self.ingest_taps[:rows].copy_(taps.to(dtype=torch.bfloat16), non_blocking=True)
        self.ingest_graphs[rows].replay()
        slots = self.ingest_positions[:rows] % self.window
        context.keys[:, slots] = self.ingest_keys[:, :rows]
        context.positions[slots] = self.ingest_positions[:rows]

    @torch.inference_mode()
    def warm(self):
        context = self.context()
        width = len(self.tap_names) * self.dim
        for rows in range(1, INGEST_GRAPH_ROWS + 1):
            self.ingest(context, list(range(rows)), torch.zeros(rows, width, dtype=torch.bfloat16, device=self.device))
        rule = (torch.tensor(1, device=self.device), torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device),
                torch.tensor(0, device=self.device))
        self.draft(context, 0, self.window, None)
        self.draft(context, 0, self.window, rule)
        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def draft(self, context, token, position, sampling=None):
        sampled = sampling is not None
        if sampled:
            for target, value in zip(self.static_rule, sampling):
                target.copy_(torch.as_tensor(value, dtype=target.dtype))
        self.static_token.fill_(int(token))
        self.static_position.fill_(int(position))
        self.static_keys.copy_(context.keys)
        self.static_positions.copy_(context.positions)
        if sampled not in self.draft_graphs:
            body = torch.compile(self.draft_body, dynamic=False) if self.compiled else self.draft_body
            self.draft_graphs[sampled] = self.capture(lambda: body(sampled))
        self.draft_graphs[sampled].replay()
        return self.out_tokens.tolist(), self.out_confidence.tolist()


def compare(checkpoint, rounds, compiled):
    fast = DsparkFast(checkpoint, keep_reference=True, compiled=compiled)
    eager_context, fast_context = fast.context(), fast.context()
    generator = torch.Generator(device="cpu").manual_seed(7)
    width = len(fast.tap_names) * fast.dim
    start = 20000
    taps = (torch.randn(fast.window + rounds, width, generator=generator) * 0.5).to(torch.bfloat16).to(fast.device)
    DsparkDrafter.ingest(fast, eager_context, list(range(start, start + fast.window)), taps[:fast.window])
    fast.ingest(fast_context, list(range(start, start + fast.window)), taps[:fast.window])
    print("FAST ingest keys max_abs_diff %.4g" % (eager_context.keys.float() - fast_context.keys.float()).abs().max().item())
    with torch.inference_mode():
        fast.static_token.fill_(1234)
        fast.static_position.fill_(start + fast.window)
        fast.static_keys.copy_(fast_context.keys)
        fast.static_positions.copy_(fast_context.positions)
        reference = fast.forward_reference(eager_context, 1234, start + fast.window)
        bodies = [("eager", fast.forward_fast)] + ([("compiled", torch.compile(fast.forward_fast, dynamic=False))] if compiled else [])
        for name, body in bodies:
            hidden, logits = body()
            print("FAST forward %s hidden rel %.3g logits rel %.3g top1 equal %d of %d" % (
                name, ((hidden.float() - reference[0].float()).norm() / reference[0].float().norm()).item(),
                ((logits - reference[1]).norm() / reference[1].norm()).item(),
                int((logits.argmax(-1) == reference[1].argmax(-1)).sum()), fast.block))
    same = slots = 0
    fast_ms, eager_ms, ingest_ms = [], [], []
    for k in range(rounds):
        position = start + fast.window + k
        token = int(1000 + 37 * k)
        torch.cuda.synchronize()
        t = time.perf_counter()
        eager_tokens, eager_confidence = DsparkDrafter.draft(fast, eager_context, token, position, None)
        torch.cuda.synchronize()
        eager_ms.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        fast_tokens, fast_confidence = fast.draft(fast_context, token, position, None)
        fast_ms.append((time.perf_counter() - t) * 1e3)
        for a, b in zip(eager_tokens, fast_tokens):
            same += int(a == b)
            if a != b:
                break
        slots += len(eager_tokens)
        row = taps[fast.window + k: fast.window + k + 1]
        DsparkDrafter.ingest(fast, eager_context, [position], row)
        torch.cuda.synchronize()
        t = time.perf_counter()
        fast.ingest(fast_context, [position], row)
        torch.cuda.synchronize()
        ingest_ms.append((time.perf_counter() - t) * 1e3)
    rule = (torch.tensor(1000, device=fast.device), torch.tensor(1.0, device=fast.device), torch.tensor(0.95, device=fast.device),
            torch.tensor(0, device=fast.device))
    sampled_same = 0
    for k in range(min(rounds, 20)):
        position = start + fast.window + rounds
        a, _ = DsparkDrafter.draft(fast, eager_context, 2000 + k, position, rule)
        b, _ = fast.draft(fast_context, 2000 + k, position, rule)
        sampled_same += int(a == b)
    def median(values):
        values = sorted(values[3:])
        return values[len(values) // 2]
    print("FAST greedy prefix agreement %d of %d slots; sampled rounds identical %d of %d" % (same, slots, sampled_same, min(rounds, 20)))
    print("FAST draft_ms median %.2f (eager %.2f) ingest1_ms median %.2f mem_gib %.1f" % (
        median(fast_ms), median(eager_ms), median(ingest_ms), torch.cuda.max_memory_allocated() / 2 ** 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--eager-body", action="store_true")
    args = parser.parse_args()
    compare(args.checkpoint, args.rounds, not args.eager_body)


if __name__ == "__main__":
    main()
