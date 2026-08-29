# Perf program — kimi's three-deep-dive consolidation (2026-08-30)

THE ROSETTA STONE: dsv4 TP4 beats our own vLLM reference (40.46 vs
37.79) because it alone has the vLLM-shaped loop — device-resident
chaining (8 tok/submission, on-device token feedback) + async
callback-driven completion. Every other family: chain depth 1,
synchronous submit ending cudaStreamSynchronize (qwen38_27b_module.c
:2129) — full round trip per token, CPU-ahead depth zero. MEASURED:
27B B1 322ms wall vs 225ms GPU = ~30% serialized host bubble. The
firmware philosophy is ORTHOGONAL to the loop shape — same
discipline, vLLM-shaped loop, wins.

RANKED (kimi's order, each >10%):
P1 CHAIN+ASYNC GENERALIZATION: port dsv4's RESIDENT_DECODE_CHAIN +
async completion to the other adapters. Why the community cells look
terrible: vLLM+DSpark 90-123 vs our no-spec 33-40 = orchestration,
not kernels. FLEET-GATED (needs live modules; after closeout).
P2 PREDECLARED COLLECTIVE PROGRAM: 130-158 collectives/token, ~48us
host rendezvous each vs NCCL's 5-15us; 15-20% of TP step time. The
doc'd fix (device-side program replacing host submit/callback per
collective) is right — and graphs must WRAP device programs, not the
host protocol (the dsv4 full-graph regression wrapped 130 host
rendezvous).
P3 BATCHED-PATH KERNELS FROM B=2: the "r-law" is a DISPATCH issue —
FP8 rows<=4 hit the scalar GEMV whose grid re-streams weights per row
(spark_lm_kernels.cuh:5112-5125); B1==B2==8.31 because each row
re-reads 29.9GB. Batched GEMMs amortize one weight stream from B=2;
our curve climbs only at B8 and saturates ~41. HOST-ORACLE-FIRST
(independent of the fleet).
P4 SPLIT-K ATTENTION (long context): shared decode attn does a full
block reduction per position, 2-byte scalar loads, reads KV TWICE
(attn.cuh:198-214) = the 32K penalty.

QUICK WINS (now): capacity tax — GDN 150MB/lane oversized configs
measured 1.6-1.8x at mid-B; size max_active_sequences to served B,
spec defaults (1024/4096) are footguns (README rule added). Kernel
gaps bound to cells: routed-expert GEMV ~135GB/s (50% of peak, 21% of
dsv4 kernel time), glm52 head GEMV uncoalesced, gemm.cuh BF16-mma-
only caps B>=64 where FP8 paths pull away.
OPEN HOST NUMBER: dsv4 32K row-serial prefill (141min vs lean 17) —
TTTT-only, largest host-side item.
