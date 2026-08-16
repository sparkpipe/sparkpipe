# Qwen3.8-27B TP4 (qwen36) phase 2 performance plan

Measured on the spark0-3 TP4 band (4 x GB10, hidden-transport verbs backend),
all numbers token-exact against the 64-token reference.

**Per-device accounting note:** TP4 shards the model COLUMNS, not the batch —
every spark processes every token through its own weight shard. The band's
tokens/s IS the per-spark tokens/s; there is no batch multiplier to claim.
The comparison below is B1 only, per spark, BF16.

## B1 per-spark comparison (BF16, batch 1)

| device | model | tok/s | notes |
|---|---|---|---|
| GB10 (ours) | Qwen3.8-27B dense | 12.1 plain / 13.4 spec D=2 | 83 / 74.8 ms step |
| H20 96GB | Qwen3-32B dense | 20.7 (SGLang) / 26.2 (HF) | official Qwen3 speed benchmark |
| GB10 (repo record) | Qwen3.8-2.4T-A95B MoE | ~6 | same device, different model |

- The repo-internal GB10 row is the same device with a ~3.4x heavier weight
  stream per token; it is a datapoint, NOT an engineering comparison, and no
  credit is claimed from it (a 4-spark band running that model would just be
  four of those devices, each still near its own memory floor).
- Against the H20 at equal precision and batch we are BEHIND (12-13 vs
  20.7-26.2 tok/s per device). That is hardware, not tuning: the GB10 memory
  system streams ~226 GB/s, so the per-spark BF16 weight floor is
  13.5 GB / 226 GB/s = ~60 ms = ~16.7 tok/s ceiling, and we sit at ~80% of it.
  The H20 sits at 20.7 against a ~74 tok/s ceiling of its own (~28%).
- No published B1 BF16 record for a 27-32B dense model on GB10/DGX-Spark
  class hardware was found, so there is no same-device-class external number
  to beat; the H20 row is the reference point for now.

## Where a legitimate per-spark B1 exceedance exists

Shrinking the bytes is the only lever that moves per-device B1, in order:
- 4-bit weights: floor ~15 ms (~66 tok/s ceiling) — exceeds the H20
  AWQ-INT4 B1 record (47.7 tok/s) per device if per-step overheads stay in
  the 5-10 ms range. This is phase-2 target #1.
- FP8 weights: floor ~30 ms (~33 tok/s ceiling) — parity class with the H20
  FP8 number (46.2), not above it.
At BF16, per-device B1 is memory-floor bound and will not exceed H20-class
hardware on this silicon. Any claim otherwise is the more-sparks fallacy.

## Phase 2 targets (expected value order)

1. Weight quantization for B1/B8 (FP8 then 4-bit codec, dsv4 pattern).
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

Batch-scaled numbers (B8 79.9, B64 158.2 tok/s per spark) are recorded for
serving economics but excluded from the B1 comparison above, per review.

Verification gate stays: module GPU validator (TP4 standalone) plus
Qwen38-TP4-E2E-PASS (per-rank agreement + 64-token reference) on every
landed change.
