# HILLCLIMB LOG - dsv4_rocm_gfx950_mi350p kernel completeness

Permanent AMD MI350P kernel-completeness agent. Each iteration: audit every
.hip against the CUDA reference
(modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu),
rank remaining gaps by inference-correctness impact, implement the top item,
verify on the authoring workstation (no gfx950 hardware in this box - see
EVIDENCE_HOST_EXECUTOR_PROBE.md), self-audit, and re-rank. Newest iteration
first.

---

## Iteration 1 - 2026-08-25 - L1 hcEnter half lands (top-ranked gap)

### Audit scope

Every .hip/.hiph TU in the repo was read end to end:

- `source/spark_dsv4_rocm_islands.hip` (this archive; E0/L5/L4/L3 entries,
  launch plumbing, validation) and `spark_dsv4_rocm_islands.h`.
- The per-island tree at
  `modules/dsv4_resident_decode_stage/source/rocm/` (E0/F1/L1/L2/L3/L4/L5 +
  common .hiph) - the "landed L1/L2/F1" the coordinator's state summary
  refers to lives THERE, while fused KvEmission and L4 route realization
  live only HERE. Tree reconciliation stays owned by those workstreams;
  this iteration grows THIS archive (the target module).
- CUDA reference kernels + launch orchestration (3472 lines) and the
  module-side call sequences (SparkDsv4ModuleHcEnter at module.c :3226),
  plus contract sources (hwiface_v1.md section 3.1 island contents rows,
  plan_amd_gfx950_mi350p.md sections 5.1-5.3).

### Gap list after audit, ranked by inference-correctness impact

1. **L1 hcEnter absent from this archive** (CHOSEN). Contract L1 row =
   "hcPost fold, hcEnter, RMSNorm, attention input projections (+ TP shard
   pack/unpack)". The hcPost fold consumes post/comb produced by nothing in
   either tree's archive side; without hcEnter's mix/Sinkhorn/pre-reduce the
   normalized hidden input for every layer never forms. Layer cannot run.
   (In the rocm/ tree the L1 norm stage also norms stream 0 only instead of
   folding pre[]-weighted streams - flagged for that workstream.)
2. **L2 KvPost clamp subset** (rocm/ tree only): clamps to +/-448 instead of
   the bit-exact per-group QuantSimGroup sim; quantized KV payload bytes
   diverge from CUDA. This archive already owns an exact port
   (SparkDsv4RocmQuantSimGroup) - port-over candidate for a later iteration
   or the rocm/ workstream.
3. **F1 markov bias + DSpark drafter heads**: draft-path only (MTP chain),
   verified against spark_dsv4_dspark_kernels.cuh :339-397. Lower mainline
   impact; S8 chain scope.
4. **L3 fused KvEmission missing from rocm/ tree** (present here since step
   4): reconciliation item.
5. **TP>1 / S8 surfaces**: ProjectionUnpack, AccumAdd/TP4-tree, standalone
   rope-inverse/Hadamard/IndexerPost, indexer score/top-k, sparse attention,
   BuildAttentionIndices, ValidateTid2Eid, HeadCertifiedFp8 variants.

### Implemented

L1 hcEnter as one entry, three kernels, appended to spark_dsv4_rocm_islands
.hip following its host-proof discipline (declared in the entry header):

- SparkDsv4RocmHcMixSplitKKernel - port of HcMixSplitKKernel: grid
  rows x splits (256-element splits, one thread per staged element), block
  sum-of-squares + per-mix-row dot partials into
  partials[(row*split+split)*(mix_rows+1)+{0,1+mix}].
- SparkDsv4RocmHcMixFinalizeSinkhornKernel - port of HcMixSplitKFinalize +
  HcParallelSinkhorn: one wavefront folds partials ascending-split per
  index, applies rsqrt(mean-square+rms_epsilon), publishes mixes, then runs
  the Sinkhorn exactly step-for-step (sigmoid pre +eps, doubled sigmoid
  post, comb row-softmax with +eps after the division so iteration zero's
  first row normalization IS the softmax, then N alternating row/column
  normalizations with +eps inside every divisor).
- SparkDsv4RocmHcPreReduceKernel - port of HcPreReduceKernel: verbatim
  residual byte copy + pre[]-weighted stream fold, one RNE bf16 store.
- Entry validation mirrors house rules: NaN geometry refuses, empty frame
  OK, flat%256!=0 INVALID, grid-capacity overflows EXHAUSTED, memory-handle
  check first, fork/join-free stream ordering, no syncs on success.

Fixed trees documented once at the section banner (B3/C3 discipline).
Deliberate deltas recorded honestly: wave64 lane subsets vs CUDA wave32;
expf where CUDA uses fast-math __expf (archive-wide spelling rule).

### Verification (ran-where: authoring workstation, no ROCm toolchain)

- tools/island_compile_check.sh vendor: full-TU host proof OK, header C11
  proof OK, l3_c2_host_probe still green, primitives regression green.
- NEW tools/l1_hcenter_host_probe.cpp, wired into the gate: executes
  SparkDsv4RocmHcPreReduceKernel under the executor shims bit-exactly vs an
  independent fp32 reference (residual verbatim + reduced RNE); replays the
  FIXED split-K trees deterministically (wave64 butterfly emulation) against
  naive fp64 sums; drives the finalize kernel shape off those partials and
  demands bit agreement with an independently organized finalize/Sinkhorn
  transcription; asserts post-Sinkhorn column-sum convergence.

### Self-audit findings and fixes

- Caught before landing: first draft passed a uint64_t host variable where
  the kernel takes uint32_t combined_blocks (ABI mismatch) - materialized a
  u32 after the capacity guard.
- Caught before landing: finalize-launch draft referenced a nonexistent
  helper with comb_f32 unmarshaled - rewritten to the plain 15-slot
  argument array.
- Probe bug #1: compared an unfilled partials buffer (the split-K kernel is
  cross-lane-coupled and correctly executor-excluded); tier rewritten to
  fp64-cross-checked emulation + independent-transcription comparison.
- Probe bug #2: emulator indexed the local 256-element staging buffer with
  the global split base (out-of-bounds reads for split>=1) - exactly the
  transcription class the probe exists to catch, found in the probe itself;
  fixed and documented in-place.
- Kernel hardening: explicit __syncthreads() between staged writes and the
  mix-dot reads so visibility never depends on barriers inside
  SparkDsv4RocmBlockReduceSum.
- Perf notes (S8 will retile, correctness shape is frozen): finalize is one
  wavefront per row (fine - it is 25 loads + 16-cell Sinkhorn per row);
  MixSplitK reads each fn element once per mix row per split from L2 -
  acceptable bring-up shape; PreReduce tile policy mirrors CUDA's
  minimum-blocks heuristic.

### Gap state after iteration 1

1. ~~L1 hcEnter~~ LANDED (this archive). Remaining L1 pieces: RMS-norm-after-
   enter wiring, projections + pack (live in rocm/ tree; port/reconcile),
   ProjectionUnpack (TP>1).
2. L2 KvPost exact QuantSimGroup swap-in (rocm/ tree; this archive owns the
   exact primitive).
3. F1 markov bias + drafter heads (small, self-contained; next candidate if
   reconciliation stalls).
4. rocm/-tree L3 KvEmission reconciliation.
5. TP>1 / S8 surfaces unchanged.
