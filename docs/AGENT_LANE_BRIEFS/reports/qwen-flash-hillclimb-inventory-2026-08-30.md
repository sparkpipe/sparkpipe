# qwen-flash: B1 hill-climb inventory (code-read, pre-baseline) — 2026-08-30

Module: modules/qwen4_flash_resident_decode_stage (2932-line cuda entry, 48
layers = 36 GDN + 12 full attn, MoE 512-expert top-10 + shared 640, TP4xPP4).
No GPU was used: this is the code-read inventory to rank once the wave gives
real baselines. Every candidate below is measured-only-after exactness.

## Where a B1 token's time must go (analytic, to be confirmed by profile)

1. MoE expert weights: 10 experts x (640x2560 gate + 640x2560 up + 2560x640
   down) x 2B = ~98 MB/layer, x48 layers = ~4.7 GB/token of bf16 weight
   reads. Bandwidth-bound BY DESIGN at B1; the hill here is overlap, not
   fewer bytes: cross-layer expert prefetch pipelined against GDN/attn
   compute (driver-level lever).
2. GDN stack: 36 layers each run ConvUpdate, DecayBeta, GdnStep as separate
   kernels per step + state pool traffic. Fusion/launch-overhead candidate;
   state reread count is the metric.
3. Head: SCREENED argmax is ALREADY the default on the owns_final_head rank
   (stage-3): 4-bit shadow (~255 MB) + certified error bounds + exact rescore
   of bounded candidates, vs the 1.02 GB bf16 shard. Verify the init log line
   "head_shadow_quantize" appears at the wave; plain HeadArgmax is only the
   null-shadow fallback.
4. PP chain: B1 latency serializes 4 stages + 3 boundary hops of the 4H
   stream vector (10240 bf16 = 20 KB — bytes are nothing, LATENCY is the
   cost). Candidates: async boundary send overlapped with next-layer compute;
   audit sync points per boundary.
5. TP4 collectives: TpCombineAdd (10240-wide) per boundary + U64Max head
   resolve per step — small; audit frequency only.

## Known unexercised code (correctness gate BEFORE any timing)

- bf16 grouped-expert TILE path (rows>=16) compiled but never run (S6 honest
  negative; tiers ran <=8 rows). First B>=16 cell must pass bit-exact vs the
  scalar path before its timing counts.

## Method note

Baselines first (wave cells), then per-candidate kill-switch cells with
exactness gates; each candidate gets one queued cell, one receipt. No
speculative rewrites of the qwen38_max-derived kernels (lane rule: STOP and
file an INTEGRATION REQUEST if geometry fights the kernel).

## Addendum (same day, boundary-path code read)

- Boundary transport is CLEAN: EmitHiddenOutput sends a DEVICE pointer +
  cuda_stream in the packet (FLAG_DEVICE_POINTER, no host staging);
  ConsumeHiddenInput is a single cudaMemcpyAsync D2D on the receive stream.
  No per-boundary host sync exists in runtime/pipeline_runtime.c (0
  synchronize calls in 718 lines) or the adapter glue.
- The measurable serialization is the FRAME PROTOCOL: every frame on every
  rank ends with cudaStreamSynchronize + frame_error D2H copyback + host
  check (module .c ~line 2769). At B1 that is 4 host round-trips inside
  every token's critical path. Candidate: double-buffer the frame-error
  record (check frame N's error while frame N+1 is enqueued). RED-stop
  semantics preserved: an error still fails loud one frame later; exactness
  gate = full stream-hash equality on the B1 smoke before/after.
