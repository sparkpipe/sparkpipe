#ifndef SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include <stdint.h>

#include "sparkpipe/spark_module_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 13u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION 6144u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT 64u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION 512u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION 64u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION 2048u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION 16384u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION 576u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION 28672u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION 192u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION 256u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION 256u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT 256u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K 8u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION 2048u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION 12288u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK 128u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS 576u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_LATENT_PROJECTION_DIMENSION \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT * \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT * \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_ROPE_PROJECTION_DIMENSION 4096u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS 64u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT 2048u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT 256u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT 2u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE 32u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_NVFP4_GROUP_SIZE 16u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT 16u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_COUNT 18u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_PLAN_ABI_VERSION 2u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_LOGITS_PLAN_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_PLAN_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_PLAN_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_EVENT_COUNTER_COUNT 5u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 64u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_SLOT_STATE_ABI_VERSION 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_SCALAR_INDEX 0u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID UINT32_MAX
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_CANCELLED_TOKEN_ID UINT32_MAX

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_PREBOUND_PROJECTIONS 0x00000001u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_TILED_ONLINE_ATTENTION 0x00000002u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_GRAPH_REPLAY 0x00000004u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_PRESELECTED_SPARSE_INDICES 0x00000008u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MLP 0x00000010u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_RESTRICTED_LOGITS 0x00000020u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION 0x00000040u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH 0x00000080u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_VALIDATED_LATENCY 0x00000100u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MTP_DRAFT 0x00000200u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FULL_STAGE_PLAN 0x00000400u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_TENSOR_CORE_ALIGNMENT 0x00000800u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_NVFP4_ROUTE_SLOT_CACHE 0x00001000u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER 0x00002000u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_COMPONENT_FAST_PATH \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_PREBOUND_PROJECTIONS | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_TILED_ONLINE_ATTENTION | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_GRAPH_REPLAY | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_PRESELECTED_SPARSE_INDICES | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MLP | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_RESTRICTED_LOGITS | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_VALIDATED_LATENCY | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MTP_DRAFT | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_TENSOR_CORE_ALIGNMENT | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FUSED_STAGE_FAST_PATH \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FULL_STAGE_PLAN | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_VALIDATED_LATENCY | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_TENSOR_CORE_ALIGNMENT)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_SOTA_FAST_PATH \
    SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FUSED_STAGE_FAST_PATH
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_KNOWN_FLAGS \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_COMPONENT_FAST_PATH | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FULL_STAGE_PLAN | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_NVFP4_ROUTE_SLOT_CACHE | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER)

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_STREAM_ORDERED 0x00000001u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_CUDA_GRAPH_REPLAY 0x00000002u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_TENSOR_CORE_PROJECTIONS 0x00000004u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_TILED_ONLINE_ATTENTION 0x00000008u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_FUSED_ROPE_KV_WRITE 0x00000010u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_GROUPED_MOE 0x00000020u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_FAST_RESTRICTED_LOGITS 0x00000040u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_FAST_MTP_DRAFT 0x00000080u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_ZERO_HOST_STAGING 0x00000100u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_ZERO_DEVICE_MEMCPY 0x00000200u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_SOTA_CAPABILITIES \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_STREAM_ORDERED | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_CUDA_GRAPH_REPLAY | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_TENSOR_CORE_PROJECTIONS | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_TILED_ONLINE_ATTENTION | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_FUSED_ROPE_KV_WRITE | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_GROUPED_MOE | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_FAST_RESTRICTED_LOGITS | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_FAST_MTP_DRAFT | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_ZERO_HOST_STAGING | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_FULL_STAGE_CAPABILITY_ZERO_DEVICE_MEMCPY)

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_PLAN_KIND_EXTERNAL 0u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_PLAN_KIND_PERSISTENT_NVFP4_TOPK 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_STREAM_ORDERED 0x00000001u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_GROUPED_BY_EXPERT 0x00000002u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_PERSISTENT_WORKERS 0x00000004u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_NVFP4_TOPK 0x00000008u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_ROUTE_SLOT_CACHE 0x00000010u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_FUSED_SILU_QUANT 0x00000020u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_DIRECT_WEIGHTED_COMBINE 0x00000040u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_ZERO_HOST_STAGING 0x00000080u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_ZERO_DEVICE_MEMCPY 0x00000100u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_SOTA_CAPABILITIES \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_STREAM_ORDERED | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_GROUPED_BY_EXPERT | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_PERSISTENT_WORKERS | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_NVFP4_TOPK | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_ROUTE_SLOT_CACHE | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_FUSED_SILU_QUANT | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_DIRECT_WEIGHTED_COMBINE | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_ZERO_HOST_STAGING | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_CAPABILITY_ZERO_DEVICE_MEMCPY)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_WEIGHT_FORMAT_BF16 1u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_WEIGHT_FORMAT_NVFP4_E2M1_E4M3 2u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_GROUPED_MOE_MAX_ROUTE_TILE_COUNT 8u

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_MODULE_ID \
    "spark.glm52.resident_decode_stage.bf16.h6144.h64.d512.r64.k2048.b64.rv256.mtp2.v1"
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_TARGET \
    "cuda.sm121.glm52.resident_decode_stage.bf16"

typedef enum SparkGlm52ResidentDecodeStagePhase
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_SUBMITTED = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ATTENTION_NORM = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ATTENTION_PROJECTION = 2,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_DSA_SELECTION = 3,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_ROPE_KV_WRITE = 4,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MLA_ATTENTION = 5,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_OUTPUT_PROJECTION = 6,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_POST_ATTENTION_NORM = 7,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_LOCAL_MOE = 8,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_RESTRICTED_LOGITS = 9,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MTP_DRAFT = 10,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_MTP_VERIFY = 11,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_COMPLETION_READY = 12
} SparkGlm52ResidentDecodeStagePhase;

typedef enum SparkGlm52ResidentDecodeStageMtpCounter
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ACCEPTED = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_REJECTED = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED = 2,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ROLLBACK = 3,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_CANCELLED = 4
} SparkGlm52ResidentDecodeStageMtpCounter;

typedef enum SparkGlm52ResidentDecodeStageSparseIndexMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_PRESELECTED = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_COPY_CONTEXT_PREFIX = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DEBUG_SERIAL_TOPK = 2
} SparkGlm52ResidentDecodeStageSparseIndexMode;

typedef enum SparkGlm52ResidentDecodeStageLaunchCheckMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_NONE = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_PEEK = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_SYNC_ON_ERROR = 2
} SparkGlm52ResidentDecodeStageLaunchCheckMode;

typedef enum SparkGlm52ResidentDecodeStagePhaseClockMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DISABLED = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DEVICE_CLOCK64 = 1
} SparkGlm52ResidentDecodeStagePhaseClockMode;

typedef enum SparkGlm52ResidentDecodeStageProjectionMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_LOWERED_BF16 = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_BF16 = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3 = 2
} SparkGlm52ResidentDecodeStageProjectionMode;

typedef enum SparkGlm52ResidentDecodeStageLayerProgressionMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_ONLY = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_PRESELECTED_BF16_LOCAL_MOE = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP = 2,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTER_BF16_TOPK_ONLY = 3,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOP1 = 4,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK = 5
} SparkGlm52ResidentDecodeStageLayerProgressionMode;


typedef enum SparkGlm52ResidentDecodeStageProjectionBackendMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_SCALAR_REFERENCE = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT = 1
} SparkGlm52ResidentDecodeStageProjectionBackendMode;

typedef enum SparkGlm52ResidentDecodeStageMlpExecutionMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_SCALAR_REFERENCE = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_TENSOR_CORE = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_DRIVER_GROUPED_MOE = 2
} SparkGlm52ResidentDecodeStageMlpExecutionMode;

typedef enum SparkGlm52ResidentDecodeStageAttentionExecutionMode
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_SINGLE_BLOCK_REFERENCE = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_TILED_ONLINE_SOFTMAX = 1
} SparkGlm52ResidentDecodeStageAttentionExecutionMode;

typedef enum SparkGlm52ResidentDecodeStageLinearPlanIndex
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_QUERY_LATENT = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_QUERY_ROPE = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_KEY_ROPE = 2,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_KV_LATENT = 3,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_A = 4,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_QUERY_B = 5,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_A = 6,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RAW_KV_B = 7,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ATTENTION_OUTPUT = 8,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_GATE = 9,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_UP = 10,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DENSE_DOWN = 11,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_MOE_GATE = 12,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_MOE_UP = 13,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_MOE_DOWN = 14,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RESTRICTED_LOGITS = 15,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS = 16,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_RESERVED_1 = 17
} SparkGlm52ResidentDecodeStageLinearPlanIndex;

typedef enum SparkGlm52ResidentDecodeStageLinearPlanKind
{
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_UNUSED = 0,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR = 1,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_FP8_E4M3_ROW_MAJOR = 2,
    SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_DRIVER_CUSTOM = 3
} SparkGlm52ResidentDecodeStageLinearPlanKind;

typedef struct SparkGlm52ResidentDecodeStageLinearPlan
{
    uint32_t abi_version;
    uint32_t plan_kind;
    uint32_t input_dimension;
    uint32_t output_dimension;
    uint32_t maximum_active_sequence_count;
    uint32_t output_is_f32;
    void *cublaslt_handle;
    void *matmul_descriptor;
    void *input_layout;
    void *weight_layout;
    void *output_layout;
    const void *algorithm;
    void *workspace;
    uint64_t workspace_bytes;
    float alpha;
    float beta;
    void *custom_launch_function;
    void *custom_state;
} SparkGlm52ResidentDecodeStageLinearPlan;

typedef struct SparkGlm52ResidentDecodeStageGroupedMoePlan
{
    uint32_t abi_version;
    uint32_t plan_kind;
    uint32_t capability_flags;
    uint32_t maximum_active_sequence_count;
    uint32_t maximum_route_count;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t intermediate_dimension;
    uint32_t weight_format;
    uint32_t route_tile_count;
    uint32_t persistent_worker_block_count;
    uint32_t maximum_work_item_count;
    uint32_t reserved;
    void *launch_function;
    void *opaque_state;
    void *route_workspace;
    uint64_t route_workspace_bytes;
    uint32_t *expert_route_counts;
    uint32_t *expert_route_offsets;
    uint32_t *expert_route_write_cursors;
    uint32_t *route_indices_by_expert;
    uint32_t *work_item_count;
    uint32_t *work_item_cursor;
    uint32_t *work_item_overflow;
    uint64_t *work_items;
    void *persistent_kernel_state;
    uint64_t persistent_kernel_state_bytes;
    uint64_t launch_count;
    uint64_t validated_maximum_latency_ns;
} SparkGlm52ResidentDecodeStageGroupedMoePlan;

typedef struct SparkGlm52ResidentDecodeStageRestrictedLogitsPlan
{
    uint32_t abi_version;
    uint32_t restricted_vocab_count;
    uint32_t hidden_dimension;
    uint32_t output_is_f32;
    void *launch_function;
    void *opaque_state;
    void *workspace;
    uint64_t workspace_bytes;
    uint64_t validated_maximum_latency_ns;
} SparkGlm52ResidentDecodeStageRestrictedLogitsPlan;


typedef struct SparkGlm52ResidentDecodeStageMtpDraftPlan
{
    uint32_t abi_version;
    uint32_t restricted_vocab_count;
    uint32_t hidden_dimension;
    uint32_t draft_token_count;
    uint32_t weight_format;
    uint32_t reserved;
    void *launch_function;
    void *opaque_state;
    void *workspace;
    uint64_t workspace_bytes;
    uint64_t validated_maximum_latency_ns;
} SparkGlm52ResidentDecodeStageMtpDraftPlan;

typedef struct SparkGlm52ResidentDecodeStageFullStagePlan
{
    uint32_t abi_version;
    uint32_t maximum_active_sequence_count;
    uint32_t capability_flags;
    uint32_t reserved;
    void *launch_function;
    void *opaque_state;
    void *workspace;
    uint64_t workspace_bytes;
    uint64_t validated_maximum_latency_ns;
} SparkGlm52ResidentDecodeStageFullStagePlan;


typedef struct SparkGlm52ResidentDecodeStageCudaPipelineSlotState
{
    uint32_t abi_version;
    uint32_t graph_active_sequence_count;
    void *cuda_graph_exec;
    uint64_t graph_capture_count;
    uint64_t graph_replay_count;
    uint64_t launch_chain_count;
    uint64_t launch_error_count;
} SparkGlm52ResidentDecodeStageCudaPipelineSlotState;

typedef struct SparkGlm52ResidentDecodeStagePipelineSlot
{
    void *cuda_stream;
    const void *input_hidden_bf16;
    void *normalized_hidden_bf16;
    void *query_latent_bf16;
    void *query_rope_input_bf16;
    void *key_rope_input_bf16;
    void *current_kv_latent_bf16;
    void *raw_query_a_bf16;
    void *raw_query_a_normalized_bf16;
    void *raw_query_b_bf16;
    void *raw_kv_a_bf16;
    void *raw_kv_a_normalized_bf16;
    void *raw_kv_b_bf16;
    const uint32_t *positions;
    const uint32_t *slot_mapping;
    const uint32_t *block_table;
    const uint32_t *context_lengths;
    const uint32_t *first_block_token_offsets;
    const float *dsa_token_scores;
    uint32_t *sparse_token_indices;
    void *rotated_query_rope_bf16;
    void *attention_output_latent_bf16;
    void *attention_projected_hidden_bf16;
    void *post_attention_hidden_bf16;
    void *post_attention_normalized_hidden_bf16;
    uint32_t *moe_topk_expert_ids;
    float *moe_topk_weights;
    float *moe_router_logits;
    int32_t *moe_bound_expert_slots;
    void *moe_gate_bf16;
    void *moe_up_bf16;
    void *moe_intermediate_bf16;
    void *moe_route_output_bf16;
    void *layer_output_hidden_bf16;
    const void *mtp_draft_hidden_bf16;
    float *restricted_logits;
    float *mtp_draft_logits;
    uint32_t *restricted_selected_token_ids;
    float *restricted_selected_token_scores;
    uint32_t *mtp_draft_token_ids;
    const uint32_t *mtp_target_token_ids;
    uint32_t *mtp_accept_mask;
    uint32_t *mtp_committed_token_ids;
    uint32_t *mtp_event_counters;
    uint64_t *phase_clock_cycles;
} SparkGlm52ResidentDecodeStagePipelineSlot;

typedef struct SparkGlm52ResidentDecodeStageNodeContext
{
    uint32_t abi_version;
    uint32_t pipeline_slot_count;
    uint32_t max_active_sequence_count;
    uint32_t cache_token_capacity;
    uint32_t kv_block_count;
    uint32_t max_blocks_per_sequence;
    uint32_t position_count;
    uint32_t dsa_candidate_count;
    float qk_scale;
    float rms_norm_epsilon;
    const float *cos_table;
    const float *sin_table;
    void *mla_cache_bf16;
    void *key_nope_cache_bf16;
    void *value_cache_bf16;
    const void *attention_norm_weight_bf16;
    const void *query_latent_weight_bf16;
    const void *query_rope_weight_bf16;
    const void *key_rope_weight_bf16;
    const void *kv_latent_weight_bf16;
    const void *raw_query_a_weight_bf16;
    const void *raw_query_a_norm_weight_bf16;
    const void *raw_query_b_weight_bf16;
    const void *raw_kv_a_weight_bf16;
    const void *raw_kv_a_norm_weight_bf16;
    const void *raw_kv_b_weight_bf16;
    const uint8_t *raw_query_a_weight_fp8_e4m3;
    const float *raw_query_a_weight_scale_inv_f32;
    const uint8_t *raw_query_b_weight_fp8_e4m3;
    const float *raw_query_b_weight_scale_inv_f32;
    const uint8_t *raw_kv_a_weight_fp8_e4m3;
    const float *raw_kv_a_weight_scale_inv_f32;
    const uint8_t *raw_kv_b_weight_fp8_e4m3;
    const float *raw_kv_b_weight_scale_inv_f32;
    const void *attention_output_weight_bf16;
    const uint8_t *attention_output_weight_fp8_e4m3;
    const float *attention_output_weight_scale_inv_f32;
    const void *post_attention_norm_weight_bf16;
    const void *moe_gate_weight_bf16;
    const void *moe_up_weight_bf16;
    const void *moe_down_weight_bf16;
    const void *dense_gate_weight_bf16;
    const void *dense_up_weight_bf16;
    const void *dense_down_weight_bf16;
    const void *final_norm_weight_bf16;
    const void *restricted_lm_head_weight_bf16;
    const uint8_t *mtp_mxfp4_weight_payload_u8;
    const uint8_t *mtp_mxfp4_scale_e8m0_u8;
    const uint32_t *restricted_token_ids;
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slots;
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_pipeline_slot_states;
    uint32_t projection_mode;
    uint32_t layer_progression_mode;
    uint32_t moe_expert_count;
    uint32_t moe_first_bound_expert_id;
    uint32_t moe_bound_expert_count;
    uint32_t moe_top_k;
    uint32_t moe_intermediate_dimension;
    uint32_t dense_intermediate_dimension;
    uint32_t sparse_index_mode;
    uint32_t launch_check_mode;
    uint32_t phase_clock_mode;
    uint32_t enable_cuda_graph_replay;
    uint32_t projection_backend_mode;
    uint32_t mlp_execution_mode;
    uint32_t attention_execution_mode;
    uint32_t reserved_execution_flags;
    const SparkGlm52ResidentDecodeStageLinearPlan *linear_plans;
    uint32_t linear_plan_count;
    const SparkGlm52ResidentDecodeStageGroupedMoePlan *grouped_moe_plan;
    const SparkGlm52ResidentDecodeStageRestrictedLogitsPlan *restricted_logits_plan;
    const SparkGlm52ResidentDecodeStageMtpDraftPlan *mtp_draft_plan;
    const SparkGlm52ResidentDecodeStageFullStagePlan *full_stage_plan;
    uint64_t validated_stage_latency_ns;
    uint64_t estimated_service_time_ns;
    const void *moe_router_weight_bf16;
    const float *moe_router_score_bias_f32;
    float moe_routed_scaling_factor;
    uint32_t moe_norm_topk_prob;
    const uint8_t *moe_nvfp4_gate_weight_u8;
    const uint8_t *moe_nvfp4_up_weight_u8;
    const uint8_t *moe_nvfp4_down_weight_u8;
    const uint8_t *moe_nvfp4_gate_weight_scale_e4m3;
    const uint8_t *moe_nvfp4_up_weight_scale_e4m3;
    const uint8_t *moe_nvfp4_down_weight_scale_e4m3;
    float moe_nvfp4_gate_input_scale;
    float moe_nvfp4_gate_weight_scale_2;
    float moe_nvfp4_up_input_scale;
    float moe_nvfp4_up_weight_scale_2;
    float moe_nvfp4_down_input_scale;
    float moe_nvfp4_down_weight_scale_2;
    uint32_t moe_nvfp4_selected_expert_id;
    uint32_t moe_nvfp4_bound_expert_count;
    const uint32_t *moe_nvfp4_bound_expert_ids;
    const int32_t *moe_nvfp4_expert_id_to_bound_slot;
    const float *moe_nvfp4_gate_input_scale_f32;
    const float *moe_nvfp4_gate_weight_scale_2_f32;
    const float *moe_nvfp4_up_input_scale_f32;
    const float *moe_nvfp4_up_weight_scale_2_f32;
    const float *moe_nvfp4_down_input_scale_f32;
    const float *moe_nvfp4_down_weight_scale_2_f32;

} SparkGlm52ResidentDecodeStageNodeContext;

SparkStatus SparkGlm52ResidentDecodeStageInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state);

SparkStatus SparkGlm52ResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame);

SparkStatus SparkGlm52ResidentDecodeStageAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);

SparkStatus SparkGlm52ResidentDecodeStageSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot);

SparkStatus SparkGlm52ResidentDecodeStageLaunchPersistentGroupedNvfp4Moe(
    const SparkGlm52ResidentDecodeStageGroupedMoePlan *grouped_moe_plan,
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    uint32_t active_sequence_count,
    void *cuda_stream);

void SparkGlm52ResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif

#endif
