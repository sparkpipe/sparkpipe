# CONSULT: multi-row DSA attention waves — nondeterministic KV-cache writes (2026-09-01)

Asking for help debugging a GPU nondeterminism. Everything below is
measured on hardware (spark0, GB10, sm_121a) unless marked "by source
read". Repro commands at the bottom.

## What we are building

glm5_next (GLM 5.3 Flash; 45 layers: 34 KDA linear-attention + 11 DSA
MLA) previously executed every prefill token as its own 1-row wave (a
correctness clamp); prefill ran at decode rate (~13.6 tok/s). We lifted
it: consecutive wave rows of one sequence now form a single sequential
"run" through the KDA recurrence kernels (they were already run-aware:
`LmDeltaRuleKernel` / `LmCausalConvKernel` take `sequence_row_begin`
and their contract is "a run of T is bit-identical to T one-row
waves"). The DSA layers batch per-row (attention with per-row causal
bounds via `row_position`). With numerics ignored, serving measures
**1024 tokens in 5.3 s (~195 tok/s), 4096 in 13.4 s (~306 tok/s)** —
the ~20x we wanted.

## What is PROVEN CORRECT (all on-hardware)

1. **KDA path bit-exact at multi-row**: validator tier3 = one 8-row run
   wave vs 8 sequential 1-row waves through the FULL layer-0 chain
   (HC site → KDA → attention-post → dense MLP → mlp-post):
   **bit-exact**. This clears: the run plumbing (run derivation, run
   arrays, `sequence_row_begin` wiring), slot-keyed state/windows/reset,
   conv-window carry, boundary/hidden staging, HC site at rows=8, the
   metadata staging path, the dense-MLP GEMM ladder at rows=8.
2. **run-of-1 through the new path bit-exact** (tier4a PASS) — the new
   wave/run code is not perturbing the 1-row case.
3. `query_latent` (attention query input) row 0: **matches** the 1-row
   reference at rows=8.
4. `kv_slot` (the pre-store KV projection output, the store kernel's
   source) row 0: **matches** the 1-row reference AND is **stable
   across two identical back-to-back executions** (0/512 elements
   differ) at rows=2.

## The bug (measured)

Validator tier4 = the same run-vs-sequential equivalence at a DSA layer
(layer 3; attention half: HC site + norms + projections + KV store +
attention + value/o_proj projections), any **rows ≥ 2**:

- The **stored KV cache at position 0** differs from the sequential
  reference in a **band that moves between executions** (e.g. bytes
  [384..767] one run, [0..894] the next, [256..1023] the next —
  KV_SLOT_BYTES region of the latent is 1024 bytes = 512 bf16
  elements).
- Sharper: **the run wave is nondeterministic against itself.** Two
  identical, fully-synchronized executions of the same wave write
  different bytes at position 0's cache slot:
  `pass1 vs pass2 cache byte 0: 52 vs b6 (kv_slot diffs: 0/512)`.
  Source stable, stored bytes different.
- rows=1: clean. rows=2: already broken. rows=8: broken.
- Serving shows the downstream effect: an 8-row prefill wave produces
  different tokens than the 1-row reference (this is why the fleet is
  rolled back to the 1-row driver).

## The write path under suspicion (kernel sequence, one CUDA stream)

`SparkGlm5NextRunLayerAttention` DSA branch
(`modules/glm5_next_resident_decode_stage/source/cuda/layer.cuh`,
~line 830-1090), in stream order, all launches on the same stream:

1. `Glm5NextHcSite` (rows) — HC stream mix → `hc_collapsed_bf16`
2. `LmFusedResidualRmsNormKernel` grid=rows — `hc_collapsed` → `normed`
3. q_a GEMM (`Glm5NextLaunchBf16Linear`, tiled) — `normed` → `q_compressed`
4. `LmFusedResidualRmsNormKernel` — q_a_norm, in place
5. `Glm5NextLayerIndexer` — index projections + `Glm5NextIndexPackKernel`
   (grid=rows) writing the SEPARATE index cache; pool scoring/expansion
   bypassed at small context (context ≤ TOPK=2048)
6. q_b GEMM — `q_compressed` → `q_bf16`
7. **kv_a GEMM — `normed` → `kv_slot` (output width = LATENT_ROW = 512,
   NoPE so no rope section)**
8. `LmFusedResidualRmsNormKernel` — kv_a_norm, in place on `kv_slot`
   (this is kv_slot's final form; the store's source)
9. q latent projection (`LmPerHeadProjectKernel`, grid dim3(rows,heads))
10. **`LmKvStoreKernel` grid=rows** (`inference/kernels/kv.cuh` ~368):
    a PURE COPY — `slot = LmKvSlotRequired(view, seq[row], pos[row])`;
    `slot[i] = kv_slot[row*512 + i]` for i<512. Slot lookup is
    read-only page-table arithmetic (table is host-staged, constant,
    identity in the fixture). One writer per (seq,pos). No growth
    mutation in-kernel.
11. `LmLatentAttentionDecodeKernel` grid dim3(rows, heads)
    (`inference/kernels/attn.cuh` ~147) — reads cache slots; per-row
    causal skip `if (row_position != 0 && position > row_position[row])
    continue;` before any slot read; online-softmax accumulation.
    At these sizes the split launcher picks partitions=1 (single-pass),
    same kernel for both 1-row and multi-row.
12. value projection (per-head) → `attention_value`
13. o_proj GEMM → `attention_out`

Metadata staging (`StageWaveMetadata`) before all of this:
H2D copies of token/slot/position arrays + the run arrays, then
`SparkGlm5NextWaveMetadataKernel` grid=rows which does
`context_lengths[resident_slots[row]] = positions[row] + 1u` — **note:
at rows≥2 of one slot this is a concurrent multi-writer race on
context_lengths[0]** (values 1..rows; winner nondeterministic). We have
NOT yet found a path from context_lengths to the STORED bytes (the
attention's walk bound reads it — that would explain attention output
nondeterminism, but our measured divergence is in the stored cache
bytes, which the store writes without reading context_lengths). The
per-pass capture of positions/context_lengths was instrumented but its
run was cut short — values pending.

## The contradiction we cannot resolve

A pure-copy store kernel, reading a source that is stable across two
synchronized passes, with a single writer per target, deterministic
addressing, on a single stream — yet the stored bytes at position 0
differ between the two passes, only when rows ≥ 2. Either:

- (a) something else writes `cache[(seq,0)]` that we have not found,
- (b) the store's source (`kv_slot`) differs AT STORE TIME but not at
  capture time (i.e., something rewrites kv_slot between the store and
  our capture — a late/raced GEMM or norm tile write), or
- (c) the store kernel's reads/writes are correct but its view
  (`LmKvView`, page_table pointer, strides) differs between passes.

For (b): the passes are fully separated by cudaStreamSynchronize; the
only kernels between the store and our capture are the attention +
projections (12, 13), which write `attention_latent`/`attention_value`/
`attention_out` — separate buffers... unless one of their grid
row-dimensions writes out of its buffer at rows≥2.

## What we already ruled out (with evidence)

- Fixture device-buffer overflow from padded GEMM tiles (tile_m=16
  writing 16 rows into 8-row buffers): buffers bumped to 16 rows —
  still fails identically. (Serving slot buffers are
  execution_row_capacity-sized (1024) so the serving divergence is not
  this overflow either.)
- Multi-stream/event overlap in the GEMM launcher: no stream/event API
  use anywhere in `runtime/gemm.cuh` — single stream.
- Page-table growth races: lookup is read-only; fixture table constant.
- The attention kernel's causal masking: source-read correct
  (skip-before-read, honest online softmax); row 0's walk is position 0
  only in both phases.
- kv_slot row 0 instability: measured stable (0/512 across passes).
- The KDA recurrence kernels: proven bit-exact (tier3) — and layer 3 is
  DSA anyway.
- The indexer at these sizes: pool scoring bypassed (context ≤ TOPK);
  index stores go to the separate index cache pool.

## Not yet checked (honest list)

- Whether `kv_slot` rows 1..7 are stable across passes (only row 0
  measured). If row 1's kv_slot varies, row 1's store varies — but that
  lands in position 1, not 0... unless a store target mapping we
  mis-read.
- Per-pass values of the device `positions` array and
  `context_lengths` (instrumentation written, run pending).
- Whether the value/o_proj per-head projection kernels (grid
  dim3(rows, heads)) have any cross-row accumulation or OOB write at
  rows≥2 (they write AFTER the store, so they could only matter for
  hypothesis (b) if they write past their buffers into the cache pool —
  fixture allocation adjacency is nondeterministic, which would match
  the moving band).
- The `LmGemmTilePrefixKernel` staging (runs when group_count>1;
  group_count=1 here, so skipped) — not the culprit in this config.
- The exact allocation adjacency of fixture->kv_cache vs the buffers
  written by kernels 9/12/13 (cudaMalloc order is fixed in the fixture
  source; adjacency is what it is — worth mapping).

## The question

Given the contradiction above: **what mechanism makes identical,
synchronized, single-stream executions of a multi-row wave store
different bytes at a fixed cache position, while the store's source is
stable?** Please rank the candidate sites in our kernel sequence and
name the next single instrument that splits the remaining hypothesis
space most sharply. We can run arbitrary validator-tier experiments on
the GPU within minutes.

## Repro

- Branch: `lane/glm5next-mtp-accept` (latest tip; all tier code is in
  `modules/glm5_next_resident_decode_stage/validation/
  spark_glm5_next_resident_decode_stage_cuda_validation.cu`, functions
  `SparkGlm5NextValBuildRunWave` / `SparkGlm5NextValRunTierRun`).
- The failing gate: "tier4b dsa run-of-2 (attention)" —
  run_equivalence_kvcache / RUN WAVE NONDETERMINISTIC.
- Run (on a spark node with the repo + the synthesized pack at
  /tmp/g5n_synth_mtp_tp16.g5nsp):
  see the publish command in
  /tmp/sparkqueue-g5n-tier34*.sh on spark0 (module_publish with the
  validator script; the tier prints land in its stderr).
- Hardware: GB10 (sm_121a), CUDA 13.0, one GPU, validator runs TP1
  single-process with real pack-derived weights at full model geometry.
