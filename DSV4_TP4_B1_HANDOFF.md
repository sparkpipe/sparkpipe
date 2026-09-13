# DeepSeek V4 Flash TP4 B1 handoff

Date: 2026-08-15 UTC

Repository: `https://github.com/sparkpipe/sparkpipe.git`

Branch: `codex/dsv4-tp4-b1-lean-handoff`

Base commit: `da7f91090c0d40729352b4e4180ad231971c90a2`

## Executive state

The selected source is a native DSV4 TP4 implementation. It does not route
DeepSeek through the GLM driver, does not use speculation, and keeps the target
head, activation spine, KV state, and exact final head rescore in BF16. The
checkpoint-declared packed weight formats remain unchanged. FP8 is used only
as a certified output-head screening shadow; every surviving vocabulary row is
rescored from the untouched BF16 head before token selection.

The current branch is the lean winner after removing two production changes
that were faster only in isolation but regressed the real ring:
projection-precollective overlap and cooperative Hc finalize/pre-reduce. The
production Query path is the canonical separate RMS then RoPE sequence. The
production KV and indexer post paths remain fused because their exact
end-to-end attribution was positive.

| Gate | Handoff result |
| --- | --- |
| Exact checkpoint metadata | `deepseek-ai/DeepSeek-V4-Flash-0731`, revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Native driver | Yes; adapter `spark.dsv4.flash-0731.serving-adapter.tp4.v1` |
| B1 definition | One request, one active sequence, one input row |
| Prefill | Resident cached prompt KV |
| Output | 128 tokens, 127 timed decode intervals |
| Speculation | Off |
| Exact O24 token hash | `18cd1dfc7c7fce04adf200916184cee952a185367412a1780c0d28486c42cbd8` |
| Exact O128 token hash | `211462f2525f73b76137ee1ce9bd4e015ad8a3fd825a7c45d38fff0488598083` |
| Integrated lean mean | 40.4553 decode tok/s |
| Best contemporaneous lean control mean | 40.4753 decode tok/s |
| 50 tok/s target | Not achieved |
| Merged-main zero-drift qualification | Not yet performed |

The performance numbers are branch evidence from `spark4` through `spark7`,
not a merged-main production claim. The deployed bits remain under
`/tmp/dsv4-integrated-lean-3d962820-runtime` on all four hosts and the driver
hash is still identical on all four ranks, but no `sparkpipe_model_batch`
process was active at the final handoff probe.

## Repository identity and selected source

The handoff worktree is `/private/tmp/dsv4-combined-plus-linear`. Its origin is
the official SparkPipe repository, not the deprecated ExperienceNow fork:

```text
origin  https://github.com/sparkpipe/sparkpipe.git
```

The machine-readable inventory is
[`qualification/dsv4/performance/tp4_b1_20260815/handoff_manifest.json`](qualification/dsv4/performance/tp4_b1_20260815/handoff_manifest.json).
The critical selected-source hashes are:

| File | SHA-256 |
| --- | --- |
| `inference/kernels/dtype.cuh` | `d8967399f717bb13562e9835c3fff8542e7b88a5b764f09dede6dfbf2ee31195` |
| `model-families/common/include/sparkpipe/spark_lm_kernels.cuh` | `a7261f234ac6515fa4eda8daf6d543977511bdc6e73e6ec579d69a70c8bd3b0f` |
| `include/sparkpipe/spark_head_screen.h` | `8d1d6caa837313307392ea61611318a85a2e008a274cb48f5195d9b6b4bd852c` |
| DSV4 CUDA source | `81dc95083e353f6a0edf991d31c5a6f36731aafd0a36069104e2957e1cd52311` |
| DSV4 host module | `4b6d7636869dd9820877513eebfba80c351ab85b7b0a4d7a920dbda4a92e421e` |
| GA CUDA validator | `f9c6d83b6bad0def16897eac8775b49372238a8c265ff504c58dfc68a7ef2cdd` |

The exact driver built from this selected source is
`3d962820608fbad251aa50b7650dba2ab4b1d19ec378251c0e0ee36922e7fce4`.
That hash was re-read from `lib/model_driver.so` on all four Spark ranks at
handoff. `SOURCE_COMMIT` inside the temporary runtime contains the base commit,
not a clean commit containing the dirty source; therefore the driver hash and
the source manifest are the correct branch identities until this branch is
merged and rebuilt from `main`.

## What is retained in production source

### Common model-kernel layer

- `LmE4m3PairToFloat2` converts two E4M3 values per operation for the retained
  vectorized FP8 paths.
- The common LinearPair launcher selects a measured work shape from runtime
  `{rows, input dimension, output dimensions, weight format}`. It is not a
  separate driver per B size. B1 can use the wide 16-warp geometry when the
  production work threshold warrants it; larger row counts continue through
  the same entry point.
- The B1 dense W13 GEMV uses a measured 1024-thread CTA shape, while the
  existing row-generic path remains the qualified route for other row counts.
- The certified FP8 head screen has full rank-local vocabulary capacity. It
  computes outward-rounded error bounds, retains every candidate that can beat
  the proven lower bound, and performs exact BF16 rescoring. There is no
  overflow fallback and no approximate token-selection result.
- All caller-owned scratch sizes for the certified screen live in
  `include/sparkpipe/spark_head_screen.h`; allocation occurs during module
  setup, not in the token hot path.

### Native DSV4 layer

- The compressor consumes BF16 KV and score projections directly, widens
  inside the compression kernel, and adds APE in the same pass. The obsolete
  temporary FP32 KV and score arrays and separate widening/add launches are
  gone.
- CSA work is limited to active slots. Its exact interleaved E2E result is
  +0.9395%, 3/3 positive, so the gain is retained even though an old arbitrary
  one-percent threshold would have rejected it.
- The B1 route path fuses gate scoring, exact selection, and route-table
  construction in one cooperative kernel. Row counts above one use the same
  public entry point and its batched implementation; there is no compile-time
  feature fork in the DSV4 driver.
- Hc entry preserves raw BF16 residual bits while performing the retained
  pre-reduce work; the separate residual `cudaMemcpyAsync` is absent.
- KV post processing retains one exact kernel for RMS, BF16 store/reload
  boundary, RoPE, and quantization simulation.
- Indexer post processing retains one row-generic B1-B1024 kernel for RoPE,
  BF16 store/reload boundary, Hadamard, and checkpoint-declared FP4
  quantization simulation.
- The Query post-processing production path intentionally remains separate
  Query-head RMS then RoPE. A fused comparison kernel remains in validation
  code, but it is not called by the production module.
- The output head allocates dedicated certified scratch and full candidate
  storage once per resident slot, then invokes the certified BF16-rescore path
  for B1.

### Validation and fail-closed contracts

- The GA CUDA validator now checks gate routing, post-processing fusions,
  multiple deterministic seeds, both base and compressed frequency tables,
  and positions through 262144. It compares fused and canonical arithmetic
  byte-for-byte.
- Source-contract tests require the selected production calls and reject
  reintroduction of sequential KV/indexer launch chains, compile-time feature
  forks in those launchers, split B1 route launches, head-screen overflow
  fallbacks, and the removed Hc residual copy.
- The row-generic indexer contract explicitly checks B1 through B1024 launch
  geometry. This does not by itself prove saturated B1024 scheduling or
  throughput; it proves that the selected kernel is not a B1-only driver fork.

## Precision and correctness contract

The accepted performance envelope is quality neutral:

1. The checkpoint-declared model and packed weight codecs are unchanged.
2. Hidden-state/spine arithmetic, KV storage, the original head weights, and
   final candidate rescore remain BF16.
3. The FP8 head copy is screening metadata only. Its error bound is rounded
   outward and cannot remove a vocabulary row that could win the exact BF16
   argmax.
4. Every accepted O128 run emitted the same 128-token vector.
5. A candidate does not pass on a microbenchmark alone. It must first pass the
   exact O24 gate and then alternating full-ring O128 runs.
6. Any repeatable positive exact E2E delta is a gain; there is no arbitrary
   minimum percentage. Mixed or negative paired results do not enter the lean
   production source.

The live receipt reports these runtime codecs:

| Field | Value |
| --- | ---: |
| `linear_weight_codec` | 5 |
| `expert_weight_codec` | 7 |
| `kv_cache_codec` | 1 |

These are checkpoint/runtime contract identifiers, not authorization to
quantize the BF16 spine or skip exact target verification.

## End-to-end performance ledger

All rows below are one real request, cached prompt KV, 128 generated tokens,
no speculation, TP4 on `spark4` through `spark7`, and 127 timed intervals.
Different experiments have different immediately preceding controls; do not
subtract arbitrary rows from one another.

| Experiment | Control tok/s | Candidate tok/s | Delta | Positive pairs | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| CSA active-slot | 38.0435 | 38.4009 | +0.9395% | 3/3 | Retain |
| Pre-post stack vs resident chain | 38.0019 | 40.2541 | +5.9266% | 3/3 | Retain |
| KV plus indexer post increment | 40.1901 | 40.4007 | +0.5241% | 3/3 | Retain |
| Query fusion alone | 40.3437 | 40.0519 | -0.7232% | 0/3 | Reject |
| All-post vs lean | 40.3869 | 40.4596 | +0.1800% | 4/8 | Inconclusive; lean wins on code size |
| Projection-precollective overlap | 40.4164 | 40.1835 | -0.5763% | 0/3 | Reject and remove |
| Integrated lean revalidation | 40.1503 | 40.4553 | +0.7597% | 3/3 | Positive observation, no new code attribution |
| Cooperative Hc finalize/pre-reduce | 40.4753 | 40.3402 | -0.3336% | 0/3 | Reject and remove |

The +5.9266% pre-post result and +0.5241% KV/indexer increment compose to a
normalized +6.4817%. That is a derived composition, not a directly measured
integrated-total result. CSA is already inside the pre-post stack and must not
be added a second time.

The integrated revalidation compared semantically identical lean production
module source, so its positive 0.7597% observation is real but is not assigned
to a nonexistent code change. The strongest contemporaneous estimate for the
selected lean source is therefore approximately 40.4-40.5 tok/s.

## Evidence retained in the repository

The complete performance ledger is [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md).
The most important aggregate receipts are:

- [CSA active-slot adjudication](qualification/dsv4/performance/tp4_b1_20260814/csa_active_slot_adjudication_da7f/summary.json)
- [pre-post stack attribution](qualification/dsv4/performance/tp4_b1_20260815/pre_post_stack_vs_resident_chain_a2bfd99f/summary.json)
- [KV/indexer incremental attribution](qualification/dsv4/performance/tp4_b1_20260815/kv_indexer_runtime_inverse_07c63b06/summary.json)
- [Query-fusion rejection](qualification/dsv4/performance/tp4_b1_20260815/query_runtime_inverse_e9116c07/summary.json)
- [lean/all-post selection](qualification/dsv4/performance/tp4_b1_20260815/query_selection_kv_indexer_vs_all_post/summary.json)
- [projection-precollective rejection](qualification/dsv4/performance/tp4_b1_20260814/projection_precollective_overlap_da7f/full_ring_rejection_5e0a3b14/summary.json)
- [integrated lean live revalidation](qualification/dsv4/performance/tp4_b1_20260815/integrated_lean_3d962820/summary.json)
- [cooperative Hc rejection](qualification/dsv4/performance/tp4_b1_20260815/hc_cooperative_full_ring_rejection_a7ebd51c/summary.json)

Every summary points to its raw client event streams. The exact deployed
`model_resident.json` and `dsv4_flash_tp4_stage.json` used by the integrated
lean run are preserved beside its receipt. The performance directories also
retain isolated source/PoC evidence for rejected ideas so they can be studied
without accidentally re-entering production.

## Rejected and quarantined work

### Rejected from the selected source

- **Query RMS plus RoPE fusion:** exact but -0.7232%, 0/3 positive. Its
  production call was removed.
- **All-post stack:** +0.1800% aggregate but only 4/8 positive. Canonical Query
  is selected under Solutions/(code size squared).
- **Projection-precollective overlap:** isolated PoC was positive, but the real
  ring regressed 0.5763%, 0/3. Production overlap code was removed.
- **Cooperative Hc finalize/pre-reduce:** isolated actual-shape PoC was 93/93
  exact and +13.1800%, but full-ring E2E regressed 0.3336%, 0/3. The rejected
  integration patch is preserved only under qualification evidence.
- **Persistent expert schedule:** routed-only timing looked positive, but the
  production overlap boundary regressed 0.8311%. Not integrated.
- **Persistent dense/projection bundle:** exact, but regressed in two
  independent interleaved validations. Not integrated.
- **Persistent weight read-ahead queue:** exact and removed 128 launches, but
  stole CTA capacity and regressed about 33%. Not integrated.

### Rolling collective Program quarantine

The latest rolling-Program transport experiment is preserved at
[`qualification/dsv4/performance/tp4_b1_20260815/rolling_program_quarantine_4678dfcf`](qualification/dsv4/performance/tp4_b1_20260815/rolling_program_quarantine_4678dfcf).
It is intentionally not in the selected source.

The patch gives each operation a dedicated stream, retains recursive-step
reservations, and keeps allocation/recapture/patching/legacy submission out of
the hot start path. Its focused local tests pass. It is blocked because:

1. Its final transport source hash `4678dfcf...` has no Spark hardware run.
2. Included live receipts identify the preceding `d297fc...` revision.
3. The receipt labeled B1024 moved the same 8 MiB byte count but executed
   descriptor `{1,4194304}`, not exact `{1024,4096}`.
4. Exact B1024 preparation is unit-tested, but start/progress/drain/rearm at
   the exact runtime descriptor is unproved.

Apply that patch only after current-source recursive hardware validation,
exact-shape B1024 validation, refreshed provenance hashes, and an exact DSV4
O24/O128 E2E test.

## Live artifact state at handoff

The integrated runtime directory still exists on `spark4`, `spark5`, `spark6`,
and `spark7`:

```text
/tmp/dsv4-integrated-lean-3d962820-runtime
```

All four `lib/model_driver.so` files hash to
`3d962820608fbad251aa50b7650dba2ab4b1d19ec378251c0e0ee36922e7fce4`.
The runtime configuration uses one active sequence, one resident sequence,
one input row, one in-flight submission, and 32 logical/physical KV pages.
The exact copied configuration is in the integrated receipt directory.

No `sparkpipe_model_batch` process was active on those four hosts at the final
probe. Restart from the preserved runtime only for branch reproduction. Do not
call it production deployment; the required final path is branch commit, PR,
merge to `main`, pull clean `main` on every target, rebuild, restart, and rerun
the exact acceptance workload.

## Reproduction workflow

### 1. Verify source identity

```bash
git status --short --branch
git remote -v
git rev-parse HEAD
sha256sum \
  modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu \
  modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c \
  modules/dsv4_resident_decode_stage/validation/spark_dsv4_resident_decode_stage_cuda_validation.cu
```

The three DSV4 hashes must match the table above before comparing against the
integrated receipts.

### 2. Run the fast source contracts

```bash
python3 tests/test_dsv4_hc_residual_fusion_source.py
python3 tests/test_dsv4_driver_source_contracts.py
python3 tests/test_dsv4_native_compute_source.py
python3 tests/test_dsv4_indexer_post_fusion_source.py
python3 tests/test_grouped_moe_source_contracts.py
git diff --check
```

These tests are fast and should run before a Spark build. They do not replace
the GPU validator or E2E token gate.

### 3. Build a B1 module archive and run the GA validator

The last successful rank-local validation build used the module's normal
single-source variant target, trimmed to bucket 1 for iteration:

```bash
make -C modules/dsv4_resident_decode_stage publish_variants \
  MODULE_BATCH_VARIANT_BUCKETS=1 \
  STAGE_PACK_PATH=/home/spark1/sparkdata/dsv4_flash.fp8.pp13.b16/packs/dsv4_flash_stage.spstage \
  STAGE_COUNT=13 \
  STAGE_INDEX=1 \
  STAGE_FIRST_LAYER=3 \
  STAGE_LAYER_COUNT=3 \
  MAX_ACTIVE_SEQUENCES=1 \
  PIPELINE_SLOT_COUNT=1 \
  CUDA_ARCH=sm_121a
```

Those stage variables identify the validation pack used in that build; they
are not the TP4 deployment topology. For another rank, use that rank's exact
pack metadata. Do not change source to create a different B-size driver. The
Makefile's canonical variant ladder is `1 2 4 ... 1024` from one template.

Compile the model driver with the normal model compiler and the resulting
module library. The successful B1 form was:

```bash
build/sparkpipe_model_compile \
  --model examples/model_descriptions/dsv4_resident_decode_stage_firmware_b1.json \
  --library build/module_library \
  --output /tmp/dsv4-selected-model-driver.so \
  --include include \
  --cc-arg -L/usr/local/cuda/lib64 \
  --cc-arg -lcuda \
  --cc-arg -lcudart \
  --cc-arg -lstdc++ \
  --cc-arg -lm \
  --cc-arg -ldl \
  --cc-arg -pthread
```

Record the archive, driver, source, validator, configuration, and stage-pack
hashes before deployment.

### 4. Generate and deploy TP4 configuration

Start from
[`examples/deployments/dsv4_flash_tp4_b1_host_rdma.spec.json`](examples/deployments/dsv4_flash_tp4_b1_host_rdma.spec.json)
and the rank-local stage template. Generate rather than hand-edit the resident
deployment:

```bash
python3 tools/generate_model_resident_deployment.py \
  --specification /tmp/dsv4-tp4-selected.spec.json \
  --output /tmp/dsv4-tp4-selected-model-resident.json
```

The measured run used ranks `spark4` through `spark7`, control port 18480,
collective ports 62620-62623, direct all-to-all through 80 KiB, recursive
doubling in the middle region, and split ring from 640 KiB. The copied exact
configuration is evidence, not a universal topology. Use the fleet topology
source to emit the sole live 100 Gb/s switched interface and the direct-pair
rail for each host; do not add a hidden fallback or duplicate hard-coded
topology in DSV4 device code.

### 5. Exact gate and timing

The retained client command shape is recorded in every raw JSON receipt. For
the selected temporary runtime it was:

```bash
ssh -o BatchMode=yes spark4 \
  /tmp/dsv4-integrated-lean-3d962820-runtime/bin/sparkpipe_model_batch \
  --deployment /tmp/dsv4-integrated-lean-3d962820-runtime/config/model_resident.json \
  --runtime-root /tmp/dsv4-integrated-lean-3d962820-runtime \
  --batch /tmp/dsv4-caller-poll-o128-batch.json
```

Acceptance order:

1. Run the GPU GA validator against the exact archive.
2. Run O24 once and require the canonical O24 hash.
3. Warm control and candidate runtimes.
4. Run at least three alternating O128 control/candidate pairs.
5. Require the canonical O128 hash for every run.
6. Compute decode rate from the 127 intervals after the first emitted token.
7. Retain any repeatable exact positive delta. Remove negative or mixed
   incremental changes unless more pairs establish a stable winner.

Do not use TTFT or prefill time as B1 cached-decode throughput. Do not use a
kernel-only PoC as the final decision.

## Immediate next work

1. **Land this handoff branch through the normal PR workflow.** After merge,
   pull clean `main` on the TP4 ranks, rebuild the exact driver, deploy from the
   clean checkout, and repeat O24 plus alternating O128. That closes the
   zero-drift qualification gap.
2. **Close the rolling Program proof without integrating it early.** Run the
   final `4678dfcf...` source on hardware, execute exact `{1024,4096}`, and
   refresh receipt provenance. Only then transplant it onto the lean source
   and repeat the model gate.
3. **Continue the non-collective hill climb from the measured bottleneck.** The
   strongest local next experiment is 16-byte vectorized GEMV weight loading
   at actual DSV4 shapes, with an isolated exact PoC first and full-ring E2E
   attribution second.
4. **Keep B-size selection dynamic.** The driver must expose one efficient
   B1-B1024 family selected by runtime rows and measured hardware profile. The
   scheduler owns microbatch formation; callers must not choose driver forks.
5. **Add DSpark only after the target path is stable.** Quantizing a draft
   model is acceptable because every proposal is verified by the exact target,
   but target/spine/KV quality must remain unchanged and accepted-token rate
   must be measured end to end.
6. **Then expand capacity.** Validate TP4xPP4 so B1 latency remains near TP4
   while larger B workloads gain roughly four times resident capacity. Do not
   claim that result from collective-only or byte-equivalent tests.

## Claims that are not yet justified

- 50 tok/s B1 without speculation.
- Merged-main production readiness.
- Exact B1024 runtime execution or saturated B1024 throughput.
- TP4xPP4 capacity scaling.
- DSpark speedup or acceptance rate.
- API-level 92-question accuracy for this exact local branch.
- A zero-drift release installed from a clean `main` checkout.

The handoff is therefore a correctness-preserving, measured 40.4-40.5 tok/s
TP4 B1 branch with all selected source and evidence preserved, plus a clearly
quarantined next-generation collective patch. It is a real optimization
milestone, not the final SOTA target.
