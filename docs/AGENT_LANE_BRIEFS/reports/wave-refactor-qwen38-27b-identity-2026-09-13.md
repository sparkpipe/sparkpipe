# wave refactor-qwen38-27b identity receipt + drift taxonomy, 2026-09-13

Lane: `lane/refactor-qwen38-27b`. Base: `lane/mod-infra-qwen` (5096079) merged with
main 3fc9ce8 (#997, M-0 mesh-kernel transport ABI). Scope: adopt the common GDN
stage kernel suite (M-1) and the M-0 mesh combines onto qwen38-27b, extract
`llm_defines.h`, classify the 10.7% third-generation drift. Code-only wave: no
inference run, no weightd contact, no GPU on the authoring host.

## 1. What was adopted (deletes what it replaces)

### M-1 common_gdn_stage_kernels into the 27b arm

`modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_cuda.cu`
now unity-includes `common/common_gdn_stage_kernels.cu` and delegates five launchers:

- `LaunchEmbeddingGather`, `LaunchResidualAdd`, `LaunchSwiGlu` — the 27b kernel
  bodies were byte-identical after family-token rename to the common bodies
  (verified by normalized diff before deletion); the common launchers launch the
  same kernels with the same grid/block arithmetic (SwiGlu adds a benign
  zero-block guard; SPARK_LM_CTA_THREADS == 256 == the deleted literal).
- `LaunchFusedResidualRmsNorm`, `LaunchRmsNorm` — bodies replaced by delegation
  to `LmGdnStage*`; the common bodies launch the same shared `SparkLm*` kernels
  with identical configuration.
- `SparkQwen38_27bConfigureCudaKernels` now calls `LmGdnStageConfigureKernels()`
  first, then re-applies the 27b-specific func attributes for the family-local
  TP-shard kernels.

Identity method: per-kernel line-sequence diff of the 27b kernel section against
the common module bodies after the rename map
`SparkQwen38_27b→LmGdnStage`, `SPARK_QWEN38_27B_CUDA_*→SPARK_GDN_STAGE_CUDA_*`,
`SPARK_QWEN38_27B_MODEL_*→SPARK_LLM_*`. Result: 3 kernels IDENT, 11 diverged by
the TP-shard mechanism (section 2), 1 perf-rot, rest family-local.

### M-0 mesh combines (per the mgr2 addendum)

`spark_qwen38_27b_tp.c` registers `SparkTpMeshRegisterCommonCombines` instead of
four private combine wrappers. Deleted from the 27b arm:

- `SparkQwen38_27bAccumAddKernel` + `LaunchAccumAdd` (live ring-combine path).
- `SparkQwen38_27bAccumAddRelayKernel` + `LaunchAccumAddRelay` — DEAD CODE: no
  in-tree transport ever reads `combine_relay_bf16_function`
  (ring/transport/tp_device_collective.c uses only combine_bf16/combine_u64_max;
  ring/transport/hidden_transport.c has zero combine references).
- `SparkQwen38_27bAccumAddTp4Kernel` + `LaunchAccumAddTp4` — same dead config
  field (`combine_tp4_bf16_function`).
- `SparkQwen38_27bAccumU64MaxKernel` + `LaunchAccumU64Max` (live u64-max path).

Deletion receipt = the numerics fix. The deleted combine class accumulates in
bf16 per step: the ring path applies degree-1 sequential bf16 read-modify-write
roundings per allreduce (for TP4: three roundings per element per round), and the
deleted Tp4 kernel added three peers into a bf16 destination stepwise. The M-0
fused path (`SparkGlm5NextSumRanksF32Kernel`, staged all-rank sources) accumulates
in FP32 and rounds to bf16 once per element per round. Same class of bug coredev
found in glm5_next's 30-window sweep; fleet receipts for glm52/dsv4/ling/laguna
are on the M-0 lane. With the fused callback registered, the #997 transport
dispatches fused-first, so the 27b TP4 path stops taking per-peer bf16 steps
entirely. Behavior deltas on the error path only: the common wrappers return
SPARK_STATUS_IO_ERROR where the private ones returned SPARK_STATUS_INTERNAL_ERROR.

### llm_defines.h parameterization

New `model-families/qwen38_27b/include/sparkpipe/llm_defines.h` (116 lines, 80
keys) is now the single source for every value the common modules and the family
shims consume. `spark_qwen38_27b_model.h` and the firmware header became pure
alias shims; all 38 model literals, 11 ABI literals, 8 weight-format literals,
8 frame-flag literals and the caps/block-token literals are aliases now. Family
extras stay as family literals (PREFIX_GDN_SLOT_COUNT, VERIFY_CHECKPOINT_SLOT_BASE,
DSPARK_* geometry, the four 27b-only frame-context flags). Coverage check: all 27
SPARK_LLM_* keys consumed by common_gdn_stage_kernels.cu are defined.

## 2. Drift taxonomy (the primary deliverable)

Per-kernel classification of 27b's 25 `__global__` kernels against the common
suite (19 kernels, max-seeded):

| Class | Count | Members | Disposition |
|---|---|---|---|
| identical-after-rename (clean) | 3 shared + 5 launchers | EmbeddingGather, ResidualAdd, SwiGlu (+ RmsNorm pair) | adopted common, copies deleted |
| (a) drift-bug / rot | 2 | ChunkSolve (missed the trunk's solve_row shared-memory staging — same math, slower memory path); AccumAdd/Relay/Tp4/U64Max (bf16-per-step accumulate class; Relay+Tp4 additionally unreachable dead code) | common body wins / M-0 deletes; perf port of solve_row proposed, not taken (pipeline coherence) |
| (b) real family divergence | 11 | ConvUpdate, DecayBeta, GdnStep, GatedNorm, AttnPrepare, AttnDecode, ChunkPrepare, ChunkTransform, ChunkStep, ChunkConv (+ ChunkQkDecay identical but pipeline-coupled) | kept family-local; see proposal below |
| (c) novel math, common-module candidate | 8 | SmallBatchLean1/Tiled/FfnGateUp/TiledRans (small-batch GEMM ladder), RansBuildTable/RansStageChunk/RansDecodeTileHalf (rANS entropy decode), HeadMaxLocPack/Unpack (speculative score pack) | proposed per the operator's rule; survey section 5 already flags rANS |

Worst drift found: NOT the kernel suite — the weight-format enum. 27b and max
assign different numeric codes to the same format names in their
`WEIGHT_FORMAT_*` sets (max: FP8_E4M3_F32B128 = 4; 27b: BF16_RANS = 4 and
FP8_E4M3_F32B128 = 5). The `SPARK_LLM_WEIGHT_FORMAT_*` key is therefore NOT a
cross-family contract: each unity build is only self-consistent. Any tool or
module that compares format codes across families silently misroutes (this is
the G4 wire-code-registry mechanism in docs/COMMON_MODULE_ARCHITECTURE.md
section 7; 27b's evidence makes it load-bearing, not optional).

The 10.7% module.c similarity explained: 22 of max's module functions exist
under the same names in 27b, but the entire KV residency cluster
(KvPrepareFrame/KvEvictSlot/KvMarkWritten/KvWaitBatch — the byte-identical
max/flash twin the common module adopts) has NO 27b counterpart: the third
generation replaced the driver-owned slot pool with caller-provided block tables
(`ModuleStagePosition` maps positions through `host_physical_block_indices`
directly). `common_kv_frame.h` therefore has nothing to absorb on 27b — zero
adoption, and that is correct, not an omission. The same generation split
explains the serving adapter (2,322 LOC speculative frame server: prefix
publish/borrow/cover, dflash2 fold — no common skeleton exists for the
speculative generation yet; survey conversion order item 7 puts 27b reabsorption
behind that common).

Class (b) proposal — the architecture missed ONE mechanism for third-generation
drivers: a runtime TP-shard geometry table. 27b's `SparkQwen38_27bTpDim` device
table (11 dims: local counts + rank bases for GDN/attention channels and heads,
uploaded by `SparkQwen38_27bTpSetGeometry` at TP init, defaulting to full-model
dims at degree 1) is what every one of the 11 class-(b) kernels consumes instead
of compile-time `SPARK_LLM_*` values. The mechanism subsumes the common suite's
current tp_degree/tp_rank launcher args (max's attention-KV shard arithmetic) and
the degree-1 default is exactly the common body. Proposed family-table extension:
promote the shard table into `common_gdn_stage_kernels.cu` as
`LmGdnStageSetShardGeometry(...)` + `SPARK_GDN_STAGE_TPD_*` indices, seeded from
the 27b copy; max adopts with the default table (byte-identical PTX at degree 1
remains to be proven on a spark node). Class-(b) kernel adoption for 27b is then
mechanical, and ChunkQkDecay + ChunkSolve come along for free with the chunk
pipeline.

Also flagged for the key-space doc: the gemma4 wave's sliding-window keys
(SPARK_LLM_SLIDING_WINDOW_TOKENS, SPARK_LLM_KV_ELEMENT_BYTES, per-class rope
base) do NOT apply to qwen38-27b — its full-attention layers are plain MHA-rope
with partial rotary first dims, KV elements are bf16 (SPARK_LLM_BF16_ELEMENT_BYTES
covers the element size), and there is one rope class. No family extras needed.

## 3. LOC ledger (capability alignment: DONE = replaced deleted)

Driver-source deletions: 227 lines (cuda.cu 138, tp.c 33, model.h 54
literal-block, firmware.h 42 literal-block). Additions are the single-source
llm_defines.h (115) — the same values that were restated — plus 215 test lines
and a 7-line Makefile rule. Net driver-source shrink: 112 lines while gaining
the common combine path, the fused-FP32 numerics fix, and single-source
parameterization. The serving adapter, module.c, validation harness and
dspark/native_ws are intentionally untouched (class (b)/(c) + speculative-frame
dependency).

## 4. Compile/test evidence (authoring host, no CUDA toolchain)

- `make build/test_qwen38_27b_llm_contract` — compiles the contract test AND the
  negative TU (-Wall -Wextra -Werror, C11) against the new llm_defines.h + both
  shims; `./build/test_qwen38_27b_llm_contract` prints
  `test_qwen38_27b_llm_contract PASS` (exit 0). Vectors: geometry derivations,
  48/16 layer partition, TP-shard divisibility for degrees 1/2/4 (the tp.c
  contract), shim alias identity, weight-format registry distinctness, KV block
  mapping. Negative control: `-DSPARK_LLM_KV_BLOCK_TOKENS=65u` flips through the
  real shim chain (firmware alias changes value, mapping algebra diverges) —
  proves single-source consumption, the K3A flipped-byte pattern.
- `cc -fsyntax-only -Wall -Wextra` on `spark_qwen38_27b_tp.c` with the cuda stub
  includes: clean (the M-0 registration path is host C).
- `make build/test_qwen38_27b_work_control` → `test_qwen38_27b_work_control PASS`.
- `make build/test_qwen38_27b_serving_adapter` builds the adapter dylib + driver
  fixture dylib against the shim'd firmware header and `test_qwen38_27b_serving_adapter`
  passes (exit 0; ERRLINE negative-path logs expected).
- NOT verifiable on this host: nvcc compile of the 27b arm (unity include of
  common_gdn_stage_kernels.cu + family-local kernels) and the sm_121a gate. The
  covering gates are the module Makefile build and
  `tests/test_qwen38_math_kernels`-class nvcc compile on a spark node. Every
  SPARK_LLM_ key the common .cu consumes is defined (mechanical check), and the
  common module itself is unchanged from the mod-infra-qwen receipt.
