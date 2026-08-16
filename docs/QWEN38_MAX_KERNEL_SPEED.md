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

## The big win: grouped-expert m-loop (landed, measured)

Root cause found: the grouped expert grid launches m_blocks x n_tiles x
512 experts CTAs where m_blocks = ceil(rows/16). At B=256 a group averages
~5 rows, so ~122K of the 131K CTAs are empty m-tiles, each costing ~1.25 us
of launch/retire - the measured 233 GB/s collapse (the B-scaling fit:
B=64/128/256 dead-CTA deltas 33K/68K at 1.24/1.28 us each).

SparkLmExpertTileAllMloopKernel: grid (1, n_tiles, experts), each CTA walks
its expert's group in chunks of 8 m-tiles and shares each k-stage's staged
weight strip across the chunk. Measured (MoE-only / full layer):
  B=256: 187.0 -> 57.4 ms MoE-only (-69%); full 247.2 -> 122.4 ms (-50%)
  B=128: 99.6 -> 35.0 (-65%); full ~140 -> ~66 ms (-53%)
  B=64:  58.8 -> ~25 (-57%)
  B=32:  25.7 -> 15.5 (-40%)
  B=16:  15.1 -> 14.1 (neutral-to-better)
The replicated B=256 expert stream now runs at ~745 GB/s effective - the
800+ GB/s regime the TP16 estimates assumed, so those estimates stand.

## Next levers, in order (all quantization-free)

1. (DONE above) grouped-expert m-loop. Remaining expert-path items:
   cp.async staging to hide the producer load latency fully (the next
   increment on top of the m-loop).
2. TP16 activation (collective wiring + head-sliced projections): cuts
   every per-step byte count 16x and, by the same CTA-count mechanism,
   lands every batch in the high-efficiency regime.
3. ncu-guided GDN step pass: three structural variants regressed, so the
   next attempt should follow a profile, not a model.
4. CUDA-graph capture of the layer sequence (launch overhead, B<=16).

Net measured improvement this round: ~5% end-to-end at B=256 (dense
m-loop), with the honest finding that the MoE tile kernel is the wall
and its fix is item 1.
