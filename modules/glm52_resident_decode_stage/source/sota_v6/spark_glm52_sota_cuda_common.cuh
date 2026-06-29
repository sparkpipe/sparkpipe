#ifndef SPARK_GLM52_SOTA_CUDA_COMMON_CUH
#define SPARK_GLM52_SOTA_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

#define SPARK_GLM52_SOTA_HIDDEN_DIMENSION 6144u
#define SPARK_GLM52_SOTA_DENSE_INTERMEDIATE_DIMENSION 12288u
#define SPARK_GLM52_SOTA_MOE_INTERMEDIATE_DIMENSION 2048u
#define SPARK_GLM52_SOTA_EXPERT_COUNT 256u
#define SPARK_GLM52_SOTA_TOP_K 8u
#define SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE 16u
#define SPARK_GLM52_SOTA_NVFP4_SCALE_GROUPS_HIDDEN 384u
#define SPARK_GLM52_SOTA_NVFP4_SCALE_GROUPS_MOE 128u
#define SPARK_GLM52_SOTA_ROPE_DIMENSION 64u
#define SPARK_GLM52_SOTA_LATENT_DIMENSION 512u
#define SPARK_GLM52_SOTA_VALUE_HEAD_DIMENSION 256u
#define SPARK_GLM52_SOTA_HEAD_COUNT 64u
#define SPARK_GLM52_SOTA_SELECTED_TOKEN_COUNT 2048u
#define SPARK_GLM52_SOTA_WARP_LANES 32u
#define SPARK_GLM52_SOTA_WMMA_M 16u
#define SPARK_GLM52_SOTA_WMMA_N 16u
#define SPARK_GLM52_SOTA_WMMA_K 16u
#define SPARK_GLM52_SOTA_INVALID_U32 0xffffffffu
#define SPARK_GLM52_SOTA_MAX_GRAPH_NODES 128u
#define SPARK_GLM52_SOTA_ROUTER_EXPERTS_PER_CTA 8u
#define SPARK_GLM52_SOTA_QUANT_GROUPS_PER_CTA 4u
#define SPARK_GLM52_SOTA_DEFAULT_WORKER_BLOCKS 256u

struct SparkGlm52SotaNvfp4MatrixView
{
    const uint8_t *payload_u8;
    const uint8_t *scale_e4m3_u8;
    uint32_t row_count;
    uint32_t column_count;
    uint32_t packed_row_stride_bytes;
    uint32_t scale_row_stride_bytes;
};

struct SparkGlm52SotaMutableNvfp4MatrixView
{
    uint8_t *payload_u8;
    uint8_t *scale_e4m3_u8;
    uint32_t row_count;
    uint32_t column_count;
    uint32_t packed_row_stride_bytes;
    uint32_t scale_row_stride_bytes;
};

struct SparkGlm52SotaGroupedRouteWorkspace
{
    uint32_t maximum_token_count;
    uint32_t maximum_route_count;
    uint32_t bound_expert_count;
    uint32_t top_k;
    const uint32_t *bound_expert_ids;
    const int32_t *expert_id_to_bound_slot;
    uint32_t *topk_expert_ids;
    float *topk_weights;
    int32_t *topk_bound_slots;
    uint32_t *expert_route_counts;
    uint32_t *expert_route_offsets;
    uint32_t *expert_route_write_cursors;
    uint32_t *route_indices_by_expert;
    uint32_t *work_item_count;
    uint32_t *work_item_cursor;
    uint64_t *work_items;
    uint32_t maximum_work_item_count;
    uint32_t persistent_worker_block_count;
    uint32_t *overflow_flag;
};

typedef cudaError_t (*SparkGlm52SotaLinearLaunchFn)(void *plan_context, cudaStream_t stream);
typedef cudaError_t (*SparkGlm52SotaGroupedMoeLaunchFn)(void *plan_context, cudaStream_t stream);
typedef cudaError_t (*SparkGlm52SotaAttentionLaunchFn)(void *plan_context, cudaStream_t stream);
typedef cudaError_t (*SparkGlm52SotaMtpLaunchFn)(void *plan_context, cudaStream_t stream);

struct SparkGlm52SotaLinearPlan
{
    void *plan_context;
    SparkGlm52SotaLinearLaunchFn launch;
    const void *input;
    const void *weight;
    const void *bias;
    void *output;
    void *workspace;
    uint64_t workspace_bytes;
    uint32_t input_columns;
    uint32_t output_columns;
    uint32_t active_rows;
    uint32_t input_stride_elements;
    uint32_t output_stride_elements;
    uint32_t weight_row_stride_elements;
    uint32_t required_alignment_bytes;
    uint32_t plan_flags;
};

struct SparkGlm52SotaRouterArguments
{
    const __nv_bfloat16 *hidden_bf16;
    const __nv_bfloat16 *router_weight_bf16;
    const float *router_bias_f32;
    float *router_logits_f32;
    uint32_t *topk_expert_ids;
    float *topk_weights;
    uint32_t active_token_count;
    uint32_t normalize_topk_prob;
    float routed_scaling_factor;
    SparkGlm52SotaLinearPlan router_projection_plan;
    uint32_t require_plan;
};

struct SparkGlm52SotaMoeArguments
{
    uint32_t active_token_count;
    uint32_t active_route_count;
    uint32_t bound_expert_count;
    const __nv_bfloat16 *hidden_bf16;
    SparkGlm52SotaMutableNvfp4MatrixView hidden_nvfp4_by_token;
    SparkGlm52SotaGroupedRouteWorkspace route_workspace;
    SparkGlm52SotaNvfp4MatrixView gate_weight_nvfp4;
    SparkGlm52SotaNvfp4MatrixView up_weight_nvfp4;
    SparkGlm52SotaNvfp4MatrixView down_weight_nvfp4;
    __nv_bfloat16 *gate_bf16_by_route;
    __nv_bfloat16 *up_bf16_by_route;
    SparkGlm52SotaMutableNvfp4MatrixView intermediate_nvfp4_by_route;
    float *combined_output_f32;
    __nv_bfloat16 *combined_output_bf16;
    const float *gate_weight_scale_2_f32;
    const float *up_weight_scale_2_f32;
    const float *down_weight_scale_2_f32;
    const float *gate_input_scale_f32;
    const float *up_input_scale_f32;
    const float *down_input_scale_f32;
    void *grouped_moe_plan_context;
    SparkGlm52SotaGroupedMoeLaunchFn grouped_moe_launch;
    uint32_t require_grouped_plan;
    uint32_t fuse_gate_up_silu_requant;
    uint32_t route_tile_size;
};

struct SparkGlm52SotaSparseMlaArguments
{
    const __nv_bfloat16 *query_latent_bf16;
    const __nv_bfloat16 *query_rope_bf16;
    const __nv_bfloat16 *latent_cache_bf16;
    const __nv_bfloat16 *key_rope_cache_bf16;
    const __nv_bfloat16 *value_cache_bf16;
    const uint32_t *sparse_token_indices;
    __nv_bfloat16 *attention_output_bf16;
    uint32_t active_token_count;
    uint32_t cache_token_capacity;
    uint32_t selected_token_count;
    uint32_t cache_token_stride_elements;
    float qk_scale;
    void *attention_plan_context;
    SparkGlm52SotaAttentionLaunchFn attention_launch;
    uint32_t require_attention_plan;
};

struct SparkGlm52SotaNormArguments
{
    const __nv_bfloat16 *input_bf16;
    const __nv_bfloat16 *residual_bf16;
    const __nv_bfloat16 *weight_bf16;
    __nv_bfloat16 *output_bf16;
    __nv_bfloat16 *residual_output_bf16;
    uint32_t active_token_count;
    uint32_t hidden_dimension;
    float epsilon;
};

struct SparkGlm52SotaRopeKvArguments
{
    const __nv_bfloat16 *query_rope_in_bf16;
    const __nv_bfloat16 *key_rope_in_bf16;
    const __nv_bfloat16 *kv_latent_in_bf16;
    const __nv_bfloat16 *value_in_bf16;
    const float *cos_f32;
    const float *sin_f32;
    const uint32_t *position_indices;
    const uint32_t *cache_token_indices;
    __nv_bfloat16 *query_rope_out_bf16;
    __nv_bfloat16 *latent_cache_bf16;
    __nv_bfloat16 *key_rope_cache_bf16;
    __nv_bfloat16 *value_cache_bf16;
    SparkGlm52SotaMutableNvfp4MatrixView latent_cache_nvfp4;
    SparkGlm52SotaMutableNvfp4MatrixView value_cache_nvfp4;
    uint32_t active_token_count;
    uint32_t cache_token_capacity;
    uint32_t cache_token_stride_elements;
    uint32_t write_nvfp4;
};

struct SparkGlm52SotaDenseMlpArguments
{
    const __nv_bfloat16 *gate_bf16;
    const __nv_bfloat16 *up_bf16;
    __nv_bfloat16 *activation_bf16;
    uint32_t active_token_count;
    uint32_t intermediate_dimension;
};

struct SparkGlm52SotaRestrictedLogitsArguments
{
    const __nv_bfloat16 *hidden_bf16;
    const __nv_bfloat16 *lm_head_bf16;
    const uint32_t *token_ids;
    float *logits_f32;
    uint32_t *selected_token_ids;
    uint32_t *selected_token_offsets;
    uint32_t active_token_count;
    uint32_t restricted_token_count;
    uint32_t hidden_dimension;
    uint32_t vocab_stride_elements;
    SparkGlm52SotaLinearPlan restricted_projection_plan;
    uint32_t require_plan;
};

struct SparkGlm52SotaMxfp4MatrixView
{
    const uint8_t *payload_u8;
    const uint8_t *scale_e8m0_u8;
    uint32_t row_count;
    uint32_t column_count;
    uint32_t packed_row_stride_bytes;
    uint32_t scale_row_stride_bytes;
};

struct SparkGlm52SotaMtpArguments
{
    const __nv_bfloat16 *hidden_bf16;
    SparkGlm52SotaMxfp4MatrixView draft_weight_mxfp4;
    const uint32_t *candidate_token_ids;
    const uint32_t *verify_token_ids;
    float *draft_logits_f32;
    uint32_t *draft_token_ids;
    uint32_t *accepted_count;
    uint32_t *rejected_count;
    uint32_t *rollback_count;
    uint32_t active_token_count;
    uint32_t candidate_token_count;
    uint32_t hidden_dimension;
    void *mtp_plan_context;
    SparkGlm52SotaMtpLaunchFn mtp_launch;
    uint32_t require_plan;
};

struct SparkGlm52SotaDecodeGraphPlan
{
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaEvent_t completion_event;
    uint32_t graph_ready;
    uint32_t capture_requested;
    uint32_t replay_count;
    uint32_t capture_count;
};

static __device__ __forceinline__ float SparkGlm52SotaBf16ToFloat(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

static __device__ __forceinline__ __nv_bfloat16 SparkGlm52SotaFloatToBf16(float value)
{
    return __float2bfloat16_rn(value);
}

static __host__ __device__ __forceinline__ uint32_t SparkGlm52SotaCeilDivU32(uint32_t numerator, uint32_t denominator)
{
    return denominator == 0u ? 0u : (numerator + denominator - 1u) / denominator;
}

static __device__ __forceinline__ float SparkGlm52SotaWarpReduceSum(float value)
{
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);
    return value;
}

static __device__ __forceinline__ float SparkGlm52SotaWarpReduceMax(float value)
{
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 16));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 8));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 4));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 2));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 1));
    return value;
}

static __device__ __forceinline__ float SparkGlm52SotaBlockReduceSum(float value)
{
    __shared__ float warp_sums[32];
    __shared__ float block_sum;
    uint32_t lane_index = threadIdx.x & 31u;
    uint32_t warp_index = threadIdx.x >> 5u;
    uint32_t warp_count = (blockDim.x + 31u) >> 5u;

    value = SparkGlm52SotaWarpReduceSum(value);
    if (lane_index == 0u)
    {
        warp_sums[warp_index] = value;
    }
    __syncthreads();

    value = lane_index < warp_count ? warp_sums[lane_index] : 0.0f;
    if (warp_index == 0u)
    {
        value = SparkGlm52SotaWarpReduceSum(value);
        if (lane_index == 0u)
        {
            block_sum = value;
        }
    }
    __syncthreads();
    return block_sum;
}

static __device__ __forceinline__ float SparkGlm52SotaBlockReduceMax(float value)
{
    __shared__ float warp_maxes[32];
    __shared__ float block_max;
    uint32_t lane_index = threadIdx.x & 31u;
    uint32_t warp_index = threadIdx.x >> 5u;
    uint32_t warp_count = (blockDim.x + 31u) >> 5u;

    value = SparkGlm52SotaWarpReduceMax(value);
    if (lane_index == 0u)
    {
        warp_maxes[warp_index] = value;
    }
    __syncthreads();

    value = lane_index < warp_count ? warp_maxes[lane_index] : -CUDART_INF_F;
    if (warp_index == 0u)
    {
        value = SparkGlm52SotaWarpReduceMax(value);
        if (lane_index == 0u)
        {
            block_max = value;
        }
    }
    __syncthreads();
    return block_max;
}

static __device__ __forceinline__ float SparkGlm52SotaDecodeE2m1(uint8_t nibble)
{
    float value;

    switch (nibble & 7u)
    {
        case 1u: value = 0.5f; break;
        case 2u: value = 1.0f; break;
        case 3u: value = 1.5f; break;
        case 4u: value = 2.0f; break;
        case 5u: value = 3.0f; break;
        case 6u: value = 4.0f; break;
        case 7u: value = 6.0f; break;
        default: value = 0.0f; break;
    }
    if ((nibble & 8u) != 0u)
    {
        value = -value;
    }
    return value;
}

static __device__ __forceinline__ uint8_t SparkGlm52SotaEncodeE2m1(float value)
{
    float absolute_value = fabsf(value);
    uint8_t sign = value < 0.0f ? 8u : 0u;
    uint8_t code;

    if (absolute_value < 0.25f)
    {
        code = 0u;
    }
    else if (absolute_value < 0.75f)
    {
        code = 1u;
    }
    else if (absolute_value < 1.25f)
    {
        code = 2u;
    }
    else if (absolute_value < 1.75f)
    {
        code = 3u;
    }
    else if (absolute_value < 2.5f)
    {
        code = 4u;
    }
    else if (absolute_value < 3.5f)
    {
        code = 5u;
    }
    else if (absolute_value < 5.0f)
    {
        code = 6u;
    }
    else
    {
        code = 7u;
    }
    return sign | code;
}

static __device__ __forceinline__ float SparkGlm52SotaDecodeE4m3(uint8_t code)
{
    uint32_t exponent = (code >> 3u) & 0x0fu;
    uint32_t mantissa = code & 0x07u;
    float base;

    if (code == 0u)
    {
        return 0.0f;
    }
    base = 1.0f + ((float)mantissa * 0.125f);
    return ldexpf(base, (int32_t)exponent - 7);
}

static __device__ __forceinline__ uint8_t SparkGlm52SotaEncodePositiveE4m3(float value)
{
    int exponent;
    float normalized;
    uint32_t biased_exponent;
    uint32_t mantissa;

    value = fmaxf(value, 1.0e-8f);
    normalized = frexpf(value, &exponent) * 2.0f;
    exponent -= 1;
    if (exponent + 7 < 0)
    {
        biased_exponent = 0u;
    }
    else if (exponent + 7 > 15)
    {
        biased_exponent = 15u;
    }
    else
    {
        biased_exponent = (uint32_t)(exponent + 7);
    }
    mantissa = (uint32_t)lrintf(fminf(fmaxf((normalized - 1.0f) * 8.0f, 0.0f), 7.0f));
    return (uint8_t)((biased_exponent << 3u) | mantissa);
}

static __device__ __forceinline__ float SparkGlm52SotaDecodeE8m0(uint8_t code)
{
    return ldexpf(1.0f, (int32_t)code - 127);
}

static __device__ __forceinline__ float SparkGlm52SotaLoadNvfp4(
    const uint8_t *__restrict__ payload_u8,
    const uint8_t *__restrict__ scale_u8,
    uint32_t packed_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    uint32_t row_index,
    uint32_t column_index)
{
    uint64_t packed_index = ((uint64_t)row_index * packed_row_stride_bytes) + (column_index >> 1u);
    uint64_t scale_index = ((uint64_t)row_index * scale_row_stride_bytes) + (column_index / SPARK_GLM52_SOTA_NVFP4_GROUP_SIZE);
    uint8_t packed_value = payload_u8[packed_index];
    uint8_t nibble = (column_index & 1u) == 0u ? (packed_value & 0x0fu) : (packed_value >> 4u);
    float scale = SparkGlm52SotaDecodeE4m3(scale_u8[scale_index]);

    return SparkGlm52SotaDecodeE2m1(nibble) * scale;
}

static __device__ __forceinline__ float SparkGlm52SotaLoadMxfp4(
    const uint8_t *__restrict__ payload_u8,
    const uint8_t *__restrict__ scale_u8,
    uint32_t packed_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    uint32_t row_index,
    uint32_t column_index)
{
    uint64_t packed_index = ((uint64_t)row_index * packed_row_stride_bytes) + (column_index >> 1u);
    uint64_t scale_index = ((uint64_t)row_index * scale_row_stride_bytes) + (column_index >> 5u);
    uint8_t packed_value = payload_u8[packed_index];
    uint8_t nibble = (column_index & 1u) == 0u ? (packed_value & 0x0fu) : (packed_value >> 4u);
    float scale = SparkGlm52SotaDecodeE8m0(scale_u8[scale_index]);

    return SparkGlm52SotaDecodeE2m1(nibble) * scale;
}

static __device__ __forceinline__ void SparkGlm52SotaStoreNvfp4Pair(
    uint8_t *__restrict__ payload_u8,
    uint64_t packed_index,
    uint8_t first_code,
    uint8_t second_code)
{
    payload_u8[packed_index] = (uint8_t)((first_code & 0x0fu) | ((second_code & 0x0fu) << 4u));
}

static __device__ __forceinline__ uint64_t SparkGlm52SotaPackWorkItem(
    uint32_t expert_slot,
    uint32_t route_tile,
    uint32_t output_tile)
{
    return ((uint64_t)output_tile << 32u) | ((uint64_t)route_tile << 16u) | (uint64_t)expert_slot;
}

static __device__ __forceinline__ uint32_t SparkGlm52SotaWorkItemExpert(uint64_t item)
{
    return (uint32_t)(item & 0xffffu);
}

static __device__ __forceinline__ uint32_t SparkGlm52SotaWorkItemRouteTile(uint64_t item)
{
    return (uint32_t)((item >> 16u) & 0xffffu);
}

static __device__ __forceinline__ uint32_t SparkGlm52SotaWorkItemOutputTile(uint64_t item)
{
    return (uint32_t)(item >> 32u);
}

static __device__ __forceinline__ float SparkGlm52SotaSilu(float value)
{
    return value / (1.0f + __expf(-value));
}

static __device__ __forceinline__ int32_t SparkGlm52SotaResolveBoundExpertSlot(
    SparkGlm52SotaGroupedRouteWorkspace route_workspace,
    uint32_t expert_id)
{
    if (route_workspace.expert_id_to_bound_slot != 0)
    {
        return route_workspace.expert_id_to_bound_slot[expert_id];
    }
    for (uint32_t scan_index = 0u; scan_index < route_workspace.bound_expert_count; ++scan_index)
    {
        if (route_workspace.bound_expert_ids[scan_index] == expert_id)
        {
            return (int32_t)scan_index;
        }
    }
    return -1;
}



#define SPARK_GLM52_SOTA_BLACKWELL_MOE_TILE_M 128u
#define SPARK_GLM52_SOTA_BLACKWELL_MOE_TILE_N 128u
#define SPARK_GLM52_SOTA_BLACKWELL_MOE_TILE_K 128u
#define SPARK_GLM52_SOTA_BLACKWELL_GLM52_EXACT_SHAPE 1u
#define SPARK_GLM52_SOTA_BLACKWELL_REQUIRE_CUTLASS_FP4 1u
#define SPARK_GLM52_SOTA_BLACKWELL_PLAN_ABI_VERSION 3u

struct SparkGlm52SotaBlackwellRoutePackPlan
{
    SparkGlm52SotaGroupedRouteWorkspace route_workspace;
    SparkGlm52SotaNvfp4MatrixView token_hidden_nvfp4;
    SparkGlm52SotaMutableNvfp4MatrixView grouped_hidden_nvfp4;
    SparkGlm52SotaMutableNvfp4MatrixView grouped_intermediate_nvfp4;
    uint32_t *grouped_row_by_route_index;
    uint32_t maximum_grouped_route_count;
    uint32_t active_token_count;
    uint32_t active_route_count;
    uint32_t require_exact_glm52_shape;
};

struct SparkGlm52SotaBlackwellGroupedNvfp4GemmPlan
{
    uint32_t abi_version;
    uint32_t group_count;
    uint32_t maximum_group_count;
    uint32_t input_columns;
    uint32_t output_columns;
    uint32_t output_is_nvfp4;
    uint32_t require_sm121;
    uint32_t require_exact_glm52_shape;
    uint32_t max_active_routes;
    const uint32_t *group_m;
    const uint32_t *group_n;
    const uint32_t *group_k;
    const void **a_payload_ptrs;
    const void **a_scale_ptrs;
    const void **b_payload_ptrs;
    const void **b_scale_ptrs;
    const void **c_ptrs;
    void **d_ptrs;
    void **d_scale_ptrs;
    void *problem_shape_device;
    void *stride_a_device;
    void *stride_b_device;
    void *stride_c_device;
    void *stride_d_device;
    void *layout_sfa_device;
    void *layout_sfb_device;
    void *layout_sfd_device;
    void *alpha_device;
    void *beta_device;
    void *norm_constant_device;
    void *cutlass_workspace;
    uint64_t cutlass_workspace_bytes;
    uint32_t max_sm_count;
    uint32_t use_pdl;
    uint32_t raster_order_n;
};

struct SparkGlm52SotaBlackwellGroupedMoePlan
{
    uint32_t abi_version;
    SparkGlm52SotaMoeArguments arguments;
    SparkGlm52SotaBlackwellRoutePackPlan route_pack_plan;
    SparkGlm52SotaBlackwellGroupedNvfp4GemmPlan gate_gemm_plan;
    SparkGlm52SotaBlackwellGroupedNvfp4GemmPlan up_gemm_plan;
    SparkGlm52SotaBlackwellGroupedNvfp4GemmPlan down_gemm_plan;
    __nv_bfloat16 *grouped_gate_bf16;
    __nv_bfloat16 *grouped_up_bf16;
    __nv_bfloat16 *grouped_down_bf16;
    uint32_t require_cutlass_blackwell_fp4;
    uint32_t require_no_local_wmma;
    uint32_t require_token_quant_once;
    uint32_t require_route_major_a;
    uint32_t fuse_weighted_combine_bf16;
};

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t SparkGlm52SotaTokenHiddenBf16ToNvfp4OnceSm121(
    const __nv_bfloat16 *hidden_bf16,
    SparkGlm52SotaMutableNvfp4MatrixView hidden_nvfp4_by_token,
    uint32_t active_token_count,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaRouterBf16ToTop8Sm121(
    const SparkGlm52SotaRouterArguments *arguments,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaBuildGroupedRoutesSm121(
    SparkGlm52SotaGroupedRouteWorkspace route_workspace,
    uint32_t active_token_count,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaBuildGroupedRoutesTensorCoreSm121(
    SparkGlm52SotaGroupedRouteWorkspace route_workspace,
    uint32_t active_token_count,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaGroupedMoeSm121(
    const SparkGlm52SotaMoeArguments *arguments,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaGroupedNvfp4MoeTensorCoreSm121(
    const SparkGlm52SotaMoeArguments *arguments,
    cudaStream_t stream);

cudaError_t SparkGlm52SotaTiledSparseMlaOnlineSm121(
    const SparkGlm52SotaSparseMlaArguments *arguments,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
