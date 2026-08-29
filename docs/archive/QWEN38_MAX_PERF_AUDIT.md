# Qwen 3.8 Max second audit - performance bugs, all-reduce focus

Second pass over the inference path with the all-reduce/collective design
and the per-frame/per-step overheads under scrutiny. Four performance bugs
found and fixed; the collective design was checked and holds.

## All-reduce audit (design and implementation)

The module does not run collectives yet (TP is replicated; head-parallel is
gated), so the audit covers the design in docs/QWEN38_MAX_PERF.md against
SparkTpDeviceCollective:

- Algorithm selection (SparkTpDeviceCollectiveSelectAlgorithm) is payload
  correct for the qwen38 shapes: B=1 residual all-reduce = 16 KB -> direct
  all-to-all (max 81920); B>=40 -> counter-rotating split ring (min
  655360); between -> recursive doubling. Sane.
- The T3 claim - ONE fused residual all-reduce per layer - holds under the
  head-parallel plan: attention o_proj and GDN out_proj are row-parallel
  (rank-partial 8192-wide deltas), the residual add folds both, one
  16 KB-per-row reduce completes hidden before the next norm. The router
  gate is replicated on a complete hidden (T1), so it needs no reduce;
  GDN state is head-sharded with no collective. Verified sound.
- The device collective runs a dedicated progress thread with
  stream-ordered completion support (no busy-poll on the compute path),
  and the host collective carries a BF16 wire kind (no F32 staging
  doubling). No defect found.
- Remaining T4 items stay open and are correctness-gated: the adapter's
  post_receive/send cudaStreamSynchronize calls are the only cross-stream
  ordering between the shim copies and the module's slot stream today;
  replacing them with recorded-event waits (async completions) is the
  fix, not dropping them.

## Performance bugs found and fixed

1. KV tier restores were SERIAL store round trips - one submit+wait per
   block, so a full-context lane restore was 4096 store latencies. Now
   up to SPARK_QWEN38_MODULE_KV_STAGING_RECORDS blocks ride one GET
   batch with one wait, then stream-ordered H2D copies into the window.

2. The KV tier rewrote and re-uploaded the WHOLE block table every frame
   (lane_count x lane_stride x 4 = 8 MB at 512 x 4096) - a per-frame
   memcpy + H2D regardless of batch size. Now only the frame's lanes
   upload (16 KB per lane slice); counts stay a 2 KB full upload.

3. The serving adapter made the same 8 MB per-submission H2D; now
   per-lane slices on the execution stream with one host sync.

4. Head emission ran the EXACT full-vocab matvec: 248320 x 8192 x 2 =
   4.07 GB of head-weight reads PER ROW (1.04 TB per step at B=256 -
   the last stage's dominant cost by two orders of magnitude). Wired the
   screened exact path: a one-time 4-bit E2M1 shadow with E8M0 scales
   and certified per-neuron error norms (quantized once at init, 1.02 GB
   device), a coarse screen, and an exact rescore of the bounded
   candidate set - identical argmax, ~4x fewer bytes per row for B>=2.

## Still open (documented, not bugs)

- CUDA-graph capture for the GDN/attention launch storm (~0.3 ms/token
  launch overhead at B=1; ~330 blocks per token in the prepare path).
- Event-based async completions replacing the adapter's cross-stream
  syncs (T4).
- Vocab-parallel head across the TP group: the screened path still reads
  1.02 GB of shadow per row; sharding the vocabulary 4 ways is the next
  head-scale step (dsv4 main carries the pattern).
- Activating head-parallel attention + the residual all-reduce
  (collective wiring) as planned in docs/QWEN38_MAX_PHASE2.md.
