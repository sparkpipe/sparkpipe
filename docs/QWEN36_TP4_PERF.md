# Qwen3.8-27B TP4 (qwen36) phase 2 performance plan

Measured state on the spark0-3 TP4 band (4 x GB10, hidden-transport verbs
backend), all numbers token-exact against the 64-token reference:

| batch | tok/s | step | notes |
|---|---|---|---|
| B1 | 13.4 | 74.8 ms | MTP speculation D=2 (exact; plain baseline 12.1) |
| B8 | 79.9 | 100.1 ms | small-batch tiled GEMM + fused FFN gate+up+swiglu |
| B64 | 158.2 | 404.5 ms | direct-all-to-all 655 KB deltas |

## Single-node bests to beat

Repo-internal (one GB10 spark, Qwen3.8-2.4T-A95B MoE, measured - docs/QWEN38_MAX_PERF.md):
- ~6 tok/s single-stream B1; ~10.5 tok/s saturated at B256 (MoE tile
  CTA-latency bound); ~100 tok/s at B256 projected after their two fixes.

External (Qwen3-32B dense, one H20 96GB, B1, 2048-token generation, official
Qwen3 speed benchmark):
- BF16: 20.7 tok/s (SGLang) / 26.2 (HF); FP8: 46.2; AWQ-INT4: 47.7.

Reading: our aggregate per node already exceeds the H20 B1 BF16 number by
~7.6x and the repo's measured single-GB10 saturation by ~15x. Single-stream
B1 latency is the gap: the GB10 memory system streams ~226 GB/s, so the BF16
weight floor is ~13.5 GB/rank/step = ~60 ms, versus ~4 TB/s HBM on the H20
(~13 ms floor for the same weights). The external FP8/INT4 rows show the
lever: weight quantization is the single largest single-node multiplier.

## Phase 2 targets (in expected value order)

1. Sharded-delta B64 collective: each rank's delta is 1280 of 5120 columns,
   but the reduce ships the full 655,360 bytes. A shard-scatter/gather op
   ships 163,840 -> the B64 step drops from ~404 ms toward ~150 ms
   (~400+ tok/s).
2. Weight quantization for B1/B8 (FP8 or 4-bit codec, dsv4 pattern): the
   ~60 ms B1 weight floor falls 2-4x (external FP8 = 2.2x, INT4 = 2.3x),
   taking B1 past the H20 BF16 single-node record on our own hardware.
3. B1 head shadow path: rows=1 currently reads the full BF16 lm_head
   (636 MB = ~2.8 ms); the screened MXFP4 shadow + rescore path is ~1.5 ms.
4. B8 GDN branch: ~470 us/layer sits above the ~215 us GEMM floor (conv,
   decay-beta, gdn-step, gated-norm small-kernel overheads).
5. Collective per-op latency (~27 us x 129/step): revisit multi-outstanding
   stream-ordered submissions once the hidden-transport backend's
   multi-outstanding failure is root-caused.

Verification gate stays: module GPU validator (TP4 standalone) plus
Qwen38-TP4-E2E-PASS (per-rank agreement + 64-token reference) on every
landed change.
