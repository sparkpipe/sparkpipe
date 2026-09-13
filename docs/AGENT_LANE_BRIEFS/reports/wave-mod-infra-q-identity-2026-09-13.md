# wave mod-infra-qwen identity receipt, 2026-09-13

Baseline: origin main 5026f62 (files at HEAD).

## Cluster A: qwen cuda kernel suite -> common/common_gdn_stage_kernels.cu

Deleted source: modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_cuda.cu
(1668 lines, 19 __global__ kernels + 4 __device__ helpers + 27 launchers + configure).

Adoption: common/common_gdn_stage_kernels.cu holds the kernel section; the family
cuda.cu holds only the arm (view translator + 27 delegating launchers + configure).

Receipt method: the deleted kernel section was transformed with the token table
below and diffed against the common module's kernel section:

    diff lines: 0 (byte-identical after rename)

Token table (old -> neutral):
    SparkQwen38<Kernel>            -> LmGdnStage<Kernel>   (24 identifiers)
    SPARK_QWEN38_CUDA_*            -> SPARK_GDN_STAGE_CUDA_* (file-local shorthands)
    SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA -> SPARK_LLM_ATTN_HEADS_PER_CTA
    SPARK_QWEN38_ROUTER_SORT_CAPACITY    -> SPARK_LLM_ROUTER_SORT_CAPACITY
    SPARK_QWEN38_MAX_MODEL_*       -> SPARK_LLM_* (geometry keys, same values)
    SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS -> SPARK_LLM_KV_BLOCK_TOKENS

Kernel bodies, grid/block arithmetic, tiles, shared-memory budgets and the four
static_asserts are unchanged; llm_defines.h carries the identical numeric values
the deleted copy hardcoded, so the PTX/SASS-relevant parameter set is unchanged.
Renamed symbols change only the kernel symbol names. The sm_121a compile+run gate
remains tests/test_qwen38_math_kernels (nvcc-gated, spark node) plus
tests/test_llm_module_contract for the parameter algebra on host.

Normalized old section sha256: 3effa3e49faefd11c1388b4aa79ab672732d0b5c0ab5d127b0bdb6e46004620a
Normalized new section sha256: 3effa3e49faefd11c1388b4aa79ab672732d0b5c0ab5d127b0bdb6e46004620a

## Cluster C: SparkModuleKvPrepareFrame et al -> common/common_kv_frame.h

Deleted source: modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_module.c
functions KvWaitBatch (25 lines), KvEvictSlot (39), KvPrepareFrame (249),
KvMarkWritten (12). The 249-line KvPrepareFrame was measured byte-identical
modulo family token against the qwen4_flash copy (SEAM-1 cluster C).

Receipt method: the four deleted bodies were transformed with the kv token table
(work-control calls -> LmKvFrameOps members, kv_* state fields -> LmKvFrameState
fields, family constants -> SPARK_LLM_* keys) and diffed against the static
inline bodies now in common/common_kv_frame.h:

    diff lines: 9 total, of which the only semantic line is:

    old: if ( context->decode_batch == 0 || context->decode_batch->row_sequence_ids == 0 || table == 0 || ... )
    new: if ( row_sequence_ids == 0 || table == 0 || ... )

    The context/decode_batch null tests moved into the family delegate
    SparkQwen38MaxModuleKvPrepareFrame, which runs before the view is built and
    fails with the same SPARK_STATUS_INVALID_ARGUMENT. No behavior change.
    (Remaining diff lines: the old PrepareFrame check line above, and a missing
    trailing newline at EOF.)

Normalized old bodies sha256: b15a7ed02a5ad1f436a0928503223354833610a7da13e0d4750fda0d4d678892
Normalized new bodies sha256: c5fb13cb374a0bbc947f378752390667a9b1c411c781bcfbca8c522c2511816d

## Parameterization receipt (llm_defines.h)

model-families/qwen38_max/include/sparkpipe/llm_defines.h now holds every value
the two modules consume. The old spark_qwen38_max_model.h (81 lines of values)
and firmware.h defines became pure aliases of the SPARK_LLM_* keys; no numeric
literal changed value. Verify:

    grep -c 'define SPARK_LLM_' model-families/qwen38_max/include/sparkpipe/llm_defines.h
    git diff HEAD -- model-families/qwen38_max/include/sparkpipe/spark_qwen38_max_model.h | grep '^+' | grep -c '[0-9]u'
