# DSV4 Flash TP16 B1 decode — analytic prediction (pre-cell1)

Status: ANALYTICAL ESTIMATE ONLY (no TP16 dsv4 measurement exists yet).
Written 09-02 on spark3 while the fleet runs the glm53full wave, so that
cell1's measurement lands on a hypothesis instead of naked curiosity.
Baseline tool: tools/dsv4_tp4_decode_roofline.py (TP4, B1: 33.161 ms/step
= 24.3 bw + 8.6 collectives + 0.26 launches -> 30.16 tok/s raw; measured
cell reality 40.2-40.5 with graphs + exactness gate).

## What changes at TP16 (16 nodes, NCCL, one rank per node)

1. BANDWIDTH TERM SHRINKS ~4x. Per-rank sharded weight bytes drop from
   ~1/4 to ~1/16 of every sharded tensor; per-token read traffic scales
   the same way for the sharded fraction. TP4 traffic 4.312 GB/step
   decomposes roughly as: sharded weights read + replicated (norms, draft
   stack, embeddings half) + KV. At TP16 the sharded fraction (~most of
   the 4.3 GB) divides by 4 again; replicated parts don't. Rough split at
   TP4: ~3.6 GB sharded + ~0.7 GB replicated/KV -> TP16 per-rank traffic
   ~ 3.6/4 + 0.7 = ~1.6 GB/step.
   At the same 177 GB/s effective: ~9.0 ms (vs 24.3).

2. COLLECTIVE TERM GROWS, SHAPE UNKNOWN. TP4 had 4 phases/layer x43
   layers = 172 phases x 50 us = 8.6 ms. TP16 keeps the same phase COUNT
   (same collective sites) but each phase crosses 16 nodes over the
   100G switched fabric: ring/all-reduce latency grows ~log or ~ring-hop
   dependent; the glm5.3-flash serving datapoint (NCCL CPU-path, 16
   nodes) measured 12.7 tok/s at 95 collectives/token -> ~0.83 ms/step
   of collective-heavy step time there, i.e. NCCL 16-node phases are
   TENS of us each, not 50 us absurdly low — but dsv4's 4 phases/layer
   x 43 = 172 phases at even 30-80 us each = 5.2-13.8 ms. UNCERTAINTY
   BAND: 5-14 ms. This is the number cell1 measures.

3. LAUNCH TERM: graphs off in cell1 (config graphs=0). Expect the graph
   launch overhead to be REPLACED by eager launch overhead (larger).
   Cell2 (graphs 130) then recovers the TP4-era win.

## Prediction envelope for cell1 (graphs=0)

    step ~ 9.0 (bw) + 5..14 (collectives) + 1..3 (eager launches)
         = 15..26 ms  ->  ~38..65 tok/s RAW ceiling

    honest expectation: BELOW the TP4 cell's 40.x if 16-rank NCCL
    collectives land >10 ms/step; ABOVE it if the bandwidth saving
    dominates (collectives ~5-6 ms). Either result is informative:
      - if collectives dominate -> hill climb = collective count per
        token (fuse phases, one-shot allreduce) BEFORE graphs.
      - if bandwidth dominates -> graphs (cell2) + weightd (cell3) first.

## What cell1 must show regardless

Exact token vector 211462f2 first. Timing only after EXACT x3.
