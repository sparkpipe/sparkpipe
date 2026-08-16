# Qwen3.8-27B TP4 (qwen36) phase 2 performance plan

Measured on the spark0-3 TP4 band (4 x GB10, hidden-transport verbs backend),
all numbers token-exact against the 64-token reference.

**Per-device accounting note:** TP4 shards the model COLUMNS, not the batch —
every spark processes every token through its own weight shard. So the band's
tokens/s IS the per-spark tokens/s; the table below is per-GB10-device
throughput, not a 4-device aggregate.

| batch | per-spark tok/s | step | notes |
|---|---|---|---|
| B1 | 12.1 (plain) / 13.4 (spec D=2) | 83 / 74.8 ms | MTP speculation, exact |
| B8 | 79.9 | 100.1 ms | small-batch tiled GEMM + fused FFN gate+up+swiglu |
| B64 | 158.2 | 404.5 ms | direct-all-to-all 655 KB deltas |

## Per-device comparison at the same precision (BF16)

Repo-internal, one GB10 spark (Qwen3.8-2.4T-A95B MoE, measured in
docs/QWEN38_MAX_PERF.md): ~6 tok/s B1 single-stream, ~10.5 tok/s saturated
at B256. Per device at BF16 we are ~2x ahead of that record (different,
larger model - weight bytes per token differ).

External, one H20 96GB, Qwen3-32B dense, B1, official Qwen3 speed benchmark:
BF16 20.7 (SGLang) / 26.2 (HF); FP8 46.2; AWQ-INT4 47.7 tok/s.
**Per device at B1 BF16 we are BEHIND the H20 (12.1-13.4 vs 20.7-26.2).**

Why: the GB10 memory system streams ~226 GB/s, so the per-spark BF16 weight
floor is 13.5 GB / 226 GB/s = ~60 ms = ~16.7 tok/s ceiling; we sit at ~80% of
that ceiling. The H20 streams 54 GB at ~4 TB/s = ~13.5 ms floor, so its 20.7
tok/s is only ~28% of its own ceiling. Per-device B1 at BF16 we cannot exceed
H20-class hardware: our memory floor is the binding constraint, and we are
already near it.

Per-device exceedance is only reachable by shrinking the bytes:
- FP8 weights: our floor falls to ~30 ms (~33 tok/s ceiling) - parity class
  with the H20 FP8 number, not above it.
- 4-bit weights: floor ~15 ms (~66 tok/s ceiling) - can exceed the H20
  AWQ-INT4 47.7 tok/s per device if per-step overheads stay in the 5-10 ms
  range.
- Batch-scaled per device (B64 = 158.2 tok/s) exceeds the repo's measured
  single-GB10 saturation by ~15x; no equivalent published H20 B64 baseline
  for the same model class was found, so no external claim is made there.

## Phase 2 targets (in expected value order)

1. Weight quantization for B1/B8 (FP8 then 4-bit codec, dsv4 pattern): the
   only lever that moves per-device B1 substantially; 4-bit is the path to
   exceeding the H20 INT4 per-device record.
2. Sharded-delta B64 collective: each rank's delta is 1280 of 5120 columns
   but the reduce ships the full 655,360 bytes; a shard-scatter/gather op
   ships 163,840 -> the B64 step drops from ~404 ms toward ~150 ms.
3. B1 head shadow path: rows=1 reads the full BF16 lm_head (636 MB =
   ~2.8 ms); the screened MXFP4 shadow + rescore path is ~1.5 ms.
4. B8 GDN branch: ~470 us/layer above the ~215 us GEMM floor (conv,
   decay-beta, gdn-step, gated-norm small-kernel overheads).
5. Collective per-op latency (~27 us x 129/step): revisit multi-outstanding
   stream-ordered submissions once the hidden-transport backend's
   multi-outstanding failure is root-caused.

Verification gate stays: module GPU validator (TP4 standalone) plus
Qwen38-TP4-E2E-PASS (per-rank agreement + 64-token reference) on every
landed change.
