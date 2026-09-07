# TP16 allreduce: honest numbers and the usefulness verdict

Date: 2026-09-07. Measured on all 16 sparks (GB10, dual 100GbE rails, hidden_transport tree
backend) with tools/mb_doorbell.cu + tools/mb_run.sh on lane/allreduce-takeover. Sync mode =
true serialized per-op submit->done latency (the serving access pattern); each config 200 timed
ops after 68 warmup, bf16 sum verified at every rank (16/16 OK).

## The engine floor, honestly (p50 per-op latency)

| payload | rows | p50 | p95 | per token-row |
|---|---|---|---|---|
| 8 KiB | 1 | 147 us | 149 us | 147 us |
| 32 KiB | 4 | 148 us | 220 us | 37 us |
| 64 KiB | 8 | 219 us | 220 us | 27 us |
| 128 KiB | 16 | 290 us | 292 us | 18 us |

Claims retired by this measurement:
- "B1 66.4 us/op" (PR #807 body) was the ASYNC pipelined-throughput number, not per-op latency.
  The serving pattern is dependent/serialized per layer, so 147 us is the number that matters.
- "4xTP4->TP4 at 60 us" was a 4-rank CPU-mock minimum with 7x spread on identical configs
  (65-540 us). No such hierarchical benchmark ever ran at 16 nodes. Excluded.
- NCCL at 8 KiB/16-wide: 106 us (NCCL_16WIDE_RECEIPTS) — the tree engine's 147 us is WORSE
  than NCCL's bench, while its 66 us async claim was the comparison that made it look better.

## Structural findings

1. ~140 us of the 147 us is fixed CPU-side protocol overhead (4 dependent RDMA rounds +
   progress-thread poll + event query + pack copy + fold launch). The wire part is ~1 us.
   Payload grows 16x, latency grows 2x: bytes are free, rounds are the cost.
2. Async mode at credits=8 wedges permanently at ordinal ~120-128 on every rank (BUSY forever) —
   the "mid-run stall" noted in August's fixed_ring work is still live in the tree engine.
   credits=64 is rejected (engine cap is 8).
3. In serving (PR #811's instrumentation), per-op was 1-17 ms (avg ~5 ms) — a ~34x gap vs the
   147 us engine floor. That gap is CPU contention: the collective progress thread starves
   next to model work. The tree was never re-measured in serving after the print-stripping.

## The TP16 verdict

A glm5.3 decode step issues 91 collectives (90 x 8 KiB bf16 + 1 x 8 B maxloc; 92nd 32 KiB hc
when MTP is active). Collective floor per token:

| batch | collective/token | note |
|---|---|---|
| B1 | 91 x 147 us = 13.4 ms | TP16 NOT useful: the floor alone eats most of a 25 ms step |
| B4 | 91 x 148 / 4 = 3.4 ms | borderline |
| B8 | 91 x 219 / 8 = 2.5 ms | useful: ~10% of a healthy step |
| B16 | 91 x 290 / 16 = 1.65 ms | clearly useful |

TP16 at B1 cannot beat TP4xPP4 with this engine — and TP4xPP4's own glm5.3 number has never
been measured either (packs are placed; the "35-45 tok/s" figure is a projection). At B8+ the
amortization makes TP16 viable. At B1 the only paths are a one-round direct-all-to-all
(#760's d2a, projected 15-30 us, carved out unmeasured in #807) or compute/collective overlap.

## Optimization order (from these numbers)

1. Resurrect + measure d2a for the B1-B4 latency regime (one round of 15 parallel RDMA writes
   + local fold; the only path under ~50 us at 8 KiB).
2. Kill the ~140 us fixed overhead in the tree engine itself (fewer rounds, or fold the
   progress stages) — benefits every batch size.
3. Fix the async wedge at ~128 ops (credits=8) — it makes the pipelined mode unusable.
4. Fix the serving-side progress-thread starvation (147 us engine -> ~5 ms serving): CPU
   pinning/isolation for the collective progress thread.
5. Measure glm5.3 TP4xPP4 (packs placed, never benchmarked) for the direct topology verdict.
