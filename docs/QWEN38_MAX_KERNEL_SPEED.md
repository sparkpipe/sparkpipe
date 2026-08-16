# Qwen 3.8 Max kernel speed work - measured experiments (no quantization)

Goal: close the gap between the measured ~39 tok/s (TP4xPP4 replicated) and
what the hardware can do, without touching the vendor FP8/FP16 quality
stance. Every change below was A/B measured on spark4 with a new parametrized
probe (tools/qwen38_timing_probe.c: PACK BATCH STEPS, plus the two
measurement-only module flags SPARK_QWEN38_STAGE_DEBUG_SKIP_GDN/_SKIP_MOE
that bisect one layer into its halves).

Baseline (one GDN layer, l1 pack, B=16 / B=256):
  full 25.6 / 259.5 ms; GDN-only 11.2 / 77.8 ms; MoE-only 14.9 / 180.1 ms.

## What WORKED

1. DENSE M-GROUP TILE (kept). The dense tile kernel re-read each weight
   strip once per m-tile: at B=256 (16 m-blocks) a dense layer's 872 MB
   spine was streamed ~5 GB. New SparkLmExpertTileMloopKernel
   (SPARK_LM_MLOOP_GROUP=8) stages each k-stage's weight strip once for
   eight m-tiles (2x amplification at B=256 instead of 16x); qwen38's
   dense BF16 linears switch to it at B>=32. Measured:
     GDN-only B=256: 77.8 -> 66.6 ms (-14%); B=128: 42.0 -> 40.1;
     B=32: 20.1 -> 17.1 (-15%); B=16 unchanged (11.2-12.7, run noise).
     Full layer B=256: 259.5 -> 247.2 ms (-5%); B=16 neutral.

## What DID NOT WORK (measured, reverted - saved for the record)

2. ROW-FUSED GDN STEP (reverted): one CTA per row walking 128 heads
   serially. GDN-only went 77.8 -> 85.7 ms at B=256 and 11.2 -> 17.7 at
   B=16: serializing heads inside a CTA starves the SMs, even though it
   cuts the CTA count 128x.
3. TRANSPOSED STATE LAYOUT (reverted): making each thread's state walk
   contiguous made each WARP stride 512 B - coalescing is across threads,
   not within them. 8x sector amplification: 64.8 -> 86.7 ms at B=256.
4. MULTI-HEAD GDN STEP (reverted): four heads per 512-thread CTA with
   global-streamed state. 66.6 -> 71.9 ms at B=256: the per-(row,head)
   CTA is already latency-matched to this shape; every structural variant
   measured regressed.

## Where the remaining time goes (B=256, one GDN layer, after the win)

  MoE tile path ~187 ms (grouped expert kernel at ~233 GB/s effective -
  the documented CTA-serialization regression; B=16 runs the same bytes
  at ~804 GB/s), GDN step + per-head CTAs ~50+ ms, dense now ~6 ms.

## Next levers, in order (all quantization-free)

1. cp.async double-buffered grouped-expert tile kernel: the measured
   233 -> 804 GB/s spread between B=256 and B=16 CTA shapes is pure
   staging serialization; pipelining the decode+MMA loop is the fix and
   the single biggest remaining win (~187 -> ~55 ms at B=256).
2. TP16 activation (collective wiring + head-sliced projections): cuts
   every per-step byte count 16x and, by the same CTA-count mechanism,
   lands every batch in the high-efficiency regime.
3. ncu-guided GDN step pass: three structural variants regressed, so the
   next attempt should follow a profile, not a model.
4. CUDA-graph capture of the layer sequence (launch overhead, B<=16).

Net measured improvement this round: ~5% end-to-end at B=256 (dense
m-loop), with the honest finding that the MoE tile kernel is the wall
and its fix is item 1.
