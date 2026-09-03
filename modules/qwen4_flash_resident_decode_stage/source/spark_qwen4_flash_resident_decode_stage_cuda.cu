#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "sparkpipe/spark_qwen4_flash_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_lm_kernels.cuh"
#include "inference/kernels/frame_error.cuh"
#include "inference/kernels/route.cuh"
#include "runtime/launch.h"


#define SPARK_QWEN4_FLASH_CUDA_DK SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_KEY_DIMENSION
#define SPARK_QWEN4_FLASH_CUDA_DV SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_VALUE_DIMENSION
#define SPARK_QWEN4_FLASH_CUDA_GVA_GROUP (SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT)
#define SPARK_QWEN4_FLASH_CUDA_ATTN_GROUP (SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT / SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT)
#define SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA 2u
#define SPARK_QWEN4_FLASH_CUDA_ATTN_VALUE_PAIRS_PER_LANE \
    (SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION / (2u * SPARK_LM_WARP_LANES))
#define SPARK_QWEN4_FLASH_CUDA_GDN_STATE_ELEMENTS \
    (SPARK_QWEN4_FLASH_CUDA_DK * SPARK_QWEN4_FLASH_CUDA_DV)
#define SPARK_QWEN4_FLASH_CUDA_GDN_DECODE_SHARED_BYTES \
    (SPARK_QWEN4_FLASH_CUDA_GDN_STATE_ELEMENTS * sizeof(float))

static __device__ __forceinline__ float SparkQwen4FlashRopeFrequency(uint32_t pair)
{
	return(exp2f(-((float)(2u * pair) / (float)SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION) * log2f((float)SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_THETA)));
}

static __global__ void SparkQwen4FlashConvUpdateKernel(const void *qkv_bf16, const void *conv_weight_bf16, void *conv_out_bf16, void *conv_tail_bf16, const uint32_t *row_lane_indices, const uint32_t *state_cold_by_row, uint32_t row_count, uint32_t gdn_layer_ordinal, uint64_t tail_lane_stride, uint64_t tail_layer_stride, uint32_t local_conv_channels)
{
	uint32_t row = blockIdx.y,channel = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t tail_base;
	float window[4],accumulator;
	uint32_t tap;
	if ( row >= row_count || channel >= local_conv_channels )
		return;
	tail_base = ((uint64_t)row_lane_indices[row] * tail_lane_stride) + ((uint64_t)gdn_layer_ordinal * tail_layer_stride) + ((uint64_t)channel * 3u);
	if ( state_cold_by_row[row] != 0u )
	{
		window[0] = 0.0f;
		window[1] = 0.0f;
		window[2] = 0.0f;
	}
	else
	{
		window[0] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 0u);
		window[1] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 1u);
		window[2] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 2u);
	}
	window[3] = SparkLmBf16ToFloat(qkv_bf16,((uint64_t)row * local_conv_channels) + channel);
	accumulator = 0.0f;
	for (tap = 0; tap < SPARK_QWEN4_FLASH_MODEL_GDN_CONV_KERNEL; tap++)
		accumulator += (window[tap] * SparkLmBf16ToFloat(conv_weight_bf16,((uint64_t)channel * SPARK_QWEN4_FLASH_MODEL_GDN_CONV_KERNEL) + tap));
	SparkLmFloatToBf16(conv_out_bf16,((uint64_t)row * local_conv_channels) + channel,SparkLmSwish(accumulator));
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 0u,window[1]);
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 1u,window[2]);
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 2u,window[3]);
}

static __global__ void SparkQwen4FlashDecayBetaKernel(const void *decay_pre_bf16, const void *beta_pre_bf16, const float *a_log_f32, const float *dt_bias_f32, float *log_decay_f32, float *beta_f32, uint32_t row_count, uint32_t local_value_heads)
{
	uint32_t row = blockIdx.x,head = threadIdx.x;
	uint64_t index;
	if ( row >= row_count || head >= local_value_heads )
		return;
	index = ((uint64_t)row * local_value_heads) + head;
	log_decay_f32[index] = -expf(a_log_f32[head]) * SparkLmSoftplus(SparkLmBf16ToFloat(decay_pre_bf16,index) + dt_bias_f32[head]);
	beta_f32[index] = SparkLmSigmoid(SparkLmBf16ToFloat(beta_pre_bf16,index));
}

static __global__ void SparkQwen4FlashGdnStepKernel(const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *state_f32, void *core_out_bf16, const uint32_t *row_lane_indices, const uint32_t *state_cold_by_row, uint32_t row_count, uint32_t gdn_layer_ordinal, uint64_t state_lane_stride, uint64_t state_layer_stride, uint32_t tp_degree)
{
    const uint32_t local_value_heads =
        SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree;
    const uint32_t local_qk =
        (SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT / tp_degree) *
        SPARK_QWEN4_FLASH_CUDA_DK;
    const uint32_t local_conv = 2u * local_qk +
        (local_value_heads * SPARK_QWEN4_FLASH_CUDA_DV);
    extern __shared__ float state_shared[];
    __shared__ float qn[SPARK_QWEN4_FLASH_CUDA_DK];
    __shared__ float kn[SPARK_QWEN4_FLASH_CUDA_DK];
    __shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
    uint32_t row;
    uint32_t head;
    uint32_t column;
    uint32_t key_head;
    uint32_t element;
    uint32_t state_index;
    uint64_t conv_row;
    uint64_t state_base;
    float value;
    float q_norm;
    float k_norm;
    float decay;
    float beta;
    float kv_memory;
    float delta;
    float output;

    row = blockIdx.y;
    head = blockIdx.x;
    column = threadIdx.x;
    key_head = head / SPARK_QWEN4_FLASH_CUDA_GVA_GROUP;
    if (row >= row_count)
    {
        return;
    }

    conv_row = (uint64_t)row * local_conv;
    value = SparkLmBf16ToFloat(
        conv_out_bf16,
        conv_row + ((uint64_t)key_head * SPARK_QWEN4_FLASH_CUDA_DK) + column);
    q_norm = SparkLmBlockReduceSum(value * value, reduce_scratch);
    qn[column] = value * rsqrtf(q_norm + 1.0e-6f) *
        rsqrtf((float)SPARK_QWEN4_FLASH_CUDA_DK);

    value = SparkLmBf16ToFloat(
        conv_out_bf16,
        conv_row + local_qk +
            ((uint64_t)key_head * SPARK_QWEN4_FLASH_CUDA_DK) + column);
    k_norm = SparkLmBlockReduceSum(value * value, reduce_scratch);
    kn[column] = value * rsqrtf(k_norm + 1.0e-6f);
    __syncthreads();

    state_base =
        ((uint64_t)row_lane_indices[row] * state_lane_stride) +
        ((uint64_t)gdn_layer_ordinal * state_layer_stride) +
        ((uint64_t)head * SPARK_QWEN4_FLASH_CUDA_GDN_STATE_ELEMENTS);
    decay = state_cold_by_row[row] != 0u
        ? 0.0f
        : expf(log_decay_f32[
            ((uint64_t)row * local_value_heads) + head]);
    beta = beta_f32[
        ((uint64_t)row * local_value_heads) + head];

    kv_memory = 0.0f;
    for (element = 0u; element < SPARK_QWEN4_FLASH_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN4_FLASH_CUDA_DV) + column;
        value = state_cold_by_row[row] != 0u
            ? 0.0f
            : state_f32[state_base + state_index] * decay;
        state_shared[state_index] = value;
        kv_memory = fmaf(value, kn[element], kv_memory);
    }

    delta = (SparkLmBf16ToFloat(
        conv_out_bf16,
        conv_row + (2u * local_qk) +
            ((uint64_t)head * SPARK_QWEN4_FLASH_CUDA_DV) + column) - kv_memory) * beta;
    output = 0.0f;
    for (element = 0u; element < SPARK_QWEN4_FLASH_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN4_FLASH_CUDA_DV) + column;
        value = fmaf(kn[element], delta, state_shared[state_index]);
        state_shared[state_index] = value;
        output = fmaf(value, qn[element], output);
    }
    SparkLmFloatToBf16(
        core_out_bf16,
        ((uint64_t)row * (local_value_heads * SPARK_QWEN4_FLASH_CUDA_DV)) +
            ((uint64_t)head * SPARK_QWEN4_FLASH_CUDA_DV) + column,
        output);

    for (element = 0u; element < SPARK_QWEN4_FLASH_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN4_FLASH_CUDA_DV) + column;
        state_f32[state_base + state_index] = state_shared[state_index];
    }
}

static __global__ void SparkQwen4FlashGatedNormKernel(const void *core_bf16, const void *z_bf16, const void *norm_weight_bf16, void *output_bf16, uint32_t row_count, float epsilon, uint32_t local_value_heads, uint32_t local_value_dimension)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.y,head = blockIdx.x,column = threadIdx.x;
	uint64_t index = ((uint64_t)row * local_value_dimension) + ((uint64_t)head * SPARK_QWEN4_FLASH_CUDA_DV) + column;
	float value,variance;
	if ( row >= row_count || head >= local_value_heads )
		return;
	value = SparkLmBf16ToFloat(core_bf16,index);
	variance = SparkLmBlockReduceSum(value * value,reduce_scratch) / (float)SPARK_QWEN4_FLASH_CUDA_DV;
	value = value * rsqrtf(variance + epsilon) * SparkLmBf16ToFloat(norm_weight_bf16,column) * SparkLmSwish(SparkLmBf16ToFloat(z_bf16,index));
	SparkLmFloatToBf16(output_bf16,index,value);
}

static __global__ void SparkQwen4FlashAttnPrepareKernel(
    void *q_fused_bf16,
    const void *k_bf16,
    const void *v_bf16,
    const void *q_norm_weight_bf16,
    const void *k_norm_weight_bf16,
    void *kv_cache_bf16,
    const uint32_t *slot_mapping,
    const uint64_t *row_positions,
    uint32_t row_count,
    uint32_t attn_layer_ordinal,
    uint64_t cache_layer_stride,
    uint64_t cache_block_stride,
    float epsilon,
    uint32_t tp_degree,
    uint32_t tp_rank)
{
    const uint32_t local_heads =
        SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT / tp_degree;
    const uint32_t local_kv_heads =
        (SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT % tp_degree) != 0u
            ? SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT
            : SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT / tp_degree;
    const uint32_t local_token_elements =
        2u * local_kv_heads * SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION;
    __shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
    __shared__ float query_shared[SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION];
    __shared__ float key_shared[SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION];
    __shared__ float rope_cosine[
        SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u];
    __shared__ float rope_sine[
        SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u];
    uint32_t row;
    uint32_t head;
    uint32_t column;
    uint32_t kv_head;
    uint32_t pair;
    uint32_t slot;
    uint32_t block;
    uint32_t offset;
    uint64_t query_base;
    uint64_t key_base;
    uint64_t cache_base;
    float value;
    float sum_squares;
    float inverse_rms;

    row = blockIdx.y;
    head = (tp_rank * local_heads) + blockIdx.x;
    column = threadIdx.x;
    kv_head = head / SPARK_QWEN4_FLASH_CUDA_ATTN_GROUP;
    if (row >= row_count ||
        blockIdx.x >= local_heads ||
        column >= SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION)
    {
        return;
    }

    query_base =
        ((uint64_t)row * 2u * (uint64_t)local_heads *
            SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) +
        ((uint64_t)blockIdx.x * 2u * SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION);
    value = SparkLmBf16ToFloat(q_fused_bf16, query_base + column);
    sum_squares = SparkLmBlockReduceSum(
        value * value,
        reduce_scratch);
    inverse_rms = rsqrtf(
        sum_squares / (float)SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION +
        epsilon);
    query_shared[column] =
        value * inverse_rms *
        SparkLmBf16ToFloat(q_norm_weight_bf16, column);
    if (column < SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u)
    {
        float angle;

        pair = column;
        angle =
            (float)row_positions[row] * SparkQwen4FlashRopeFrequency(pair);
        sincosf(angle, &rope_sine[pair], &rope_cosine[pair]);
    }
    __syncthreads();

    if (column < SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u)
    {
        float real;
        float imaginary;

        pair = column;
        real = query_shared[pair];
        imaginary = query_shared[
            pair + SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u];
        query_shared[pair] =
            real * rope_cosine[pair] - imaginary * rope_sine[pair];
        query_shared[
            pair + SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u] =
            imaginary * rope_cosine[pair] + real * rope_sine[pair];
    }
    __syncthreads();
    SparkLmFloatToBf16(
        q_fused_bf16,
        query_base + column,
        query_shared[column]);

    if ((blockIdx.x % SPARK_QWEN4_FLASH_CUDA_ATTN_GROUP) != 0u)
    {
        return;
    }

    key_base =
        ((uint64_t)row * (uint64_t)local_kv_heads *
            SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) +
        ((uint64_t)(kv_head - ((SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT % tp_degree) != 0u ? 0u : tp_rank * local_kv_heads)) *
            SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION);
    value = SparkLmBf16ToFloat(k_bf16, key_base + column);
    sum_squares = SparkLmBlockReduceSum(
        value * value,
        reduce_scratch);
    inverse_rms = rsqrtf(
        sum_squares / (float)SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION +
        epsilon);
    key_shared[column] =
        value * inverse_rms *
        SparkLmBf16ToFloat(k_norm_weight_bf16, column);
    __syncthreads();

    if (column < SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u)
    {
        float real;
        float imaginary;

        pair = column;
        real = key_shared[pair];
        imaginary = key_shared[
            pair + SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u];
        key_shared[pair] =
            real * rope_cosine[pair] - imaginary * rope_sine[pair];
        key_shared[
            pair + SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u] =
            imaginary * rope_cosine[pair] + real * rope_sine[pair];
    }
    __syncthreads();

    slot = slot_mapping[row];
    block = slot / SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    offset = slot % SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    cache_base =
        ((uint64_t)block * cache_block_stride) +
        ((uint64_t)attn_layer_ordinal * cache_layer_stride) +
        ((uint64_t)offset * local_token_elements) +
        ((uint64_t)(kv_head - ((SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT % tp_degree) != 0u ? 0u : tp_rank * local_kv_heads)) *
            SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION);
    SparkLmFloatToBf16(
        kv_cache_bf16,
        cache_base + column,
        key_shared[column]);
    SparkLmFloatToBf16(
        kv_cache_bf16,
        cache_base + ((uint64_t)local_kv_heads *
            SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) + column,
        SparkLmBf16ToFloat(v_bf16, key_base + column));
}

static __device__ __forceinline__ uint64_t SparkQwen4FlashAttnTokenBase(const uint32_t *block_indices, uint64_t lane_base, uint32_t token, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t local_kv_head, uint32_t local_token_elements)
{
	uint32_t block = __ldg(block_indices + lane_base + (token / SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS));
	return(((uint64_t)block * cache_block_stride) + ((uint64_t)attn_layer_ordinal * cache_layer_stride) + ((uint64_t)(token % SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) * local_token_elements) + ((uint64_t)local_kv_head * SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION));
}





static __global__ void SparkQwen4FlashAttnDecodeKernel(const void *q_fused_bf16, const void *kv_cache_bf16, const uint32_t *block_indices, const uint32_t *block_counts, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t lane_stride, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t tp_degree, uint32_t tp_rank, const uint8_t *token_mask, uint32_t mask_stride)
{
    const uint32_t local_heads =
        SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT / tp_degree;
    const uint32_t local_kv_heads =
        (SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT % tp_degree) != 0u
            ? SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT
            : SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT / tp_degree;
    const uint32_t local_token_elements =
        2u * local_kv_heads * SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION;
    const uint32_t heads_per_cta = SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA;
    __shared__ float q_shared[
        SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA *
        SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION];
    __shared__ float merge_max[
        SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA * SPARK_LM_CTA_WARPS];
    __shared__ float merge_den[
        SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA * SPARK_LM_CTA_WARPS];
    __shared__ float merge_scale[
        SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA * SPARK_LM_CTA_WARPS];
    __shared__ float inverse_denominator[
        SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA];
    __shared__ float merge_acc[
        SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA * SPARK_LM_CTA_WARPS *
        SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION];
    uint32_t row;
    uint32_t head_base;
    uint32_t kv_head;
    uint32_t warp;
    uint32_t lane;
    uint32_t local_head;
    uint32_t head;
    uint32_t context;
    uint32_t lane_index;
    uint32_t available_block_count;
    uint32_t required_block_count;
    uint32_t token;
    uint32_t pair;
    uint32_t element;
    uint32_t partial;
    uint64_t lane_base;
    uint64_t token_base;
    uint64_t q_base;
    uint64_t out_base;
    float running_max[SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA];
    float running_den[SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA];
    float local_logit[SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA];
    float rescale[SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA];
    float weight[SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA];
    float2 accumulator[
        SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA]
        [SPARK_QWEN4_FLASH_CUDA_ATTN_VALUE_PAIRS_PER_LANE];
    float2 key_pair;
    float2 value_pair;
    float head_max;
    float denominator;
    float merged;
    float gate;

    static_assert(
        SPARK_QWEN4_FLASH_CUDA_ATTN_GROUP %
                SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA ==
            0u,
        "grouped attention CTAs must not cross KV-head ownership");
    row = blockIdx.y;
    head_base = (tp_rank * local_heads) + (blockIdx.x * heads_per_cta);
    kv_head = head_base / SPARK_QWEN4_FLASH_CUDA_ATTN_GROUP;
    warp = threadIdx.x / SPARK_LM_WARP_LANES;
    lane = threadIdx.x % SPARK_LM_WARP_LANES;
    if (row >= row_count ||
        ((blockIdx.x + 1u) * heads_per_cta) > local_heads)
    {
        return;
    }
    lane_index = row_lane_indices[row];
    context = context_lengths[row];
    available_block_count = __ldg(block_counts + lane_index);
    required_block_count =
        (context + SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) /
        SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    if (context == 0u ||
        available_block_count > lane_stride ||
        required_block_count > available_block_count)
    {
        float invalid_output;

        invalid_output = __int_as_float(0x7fc00000);
        for (element = threadIdx.x;
             element < heads_per_cta * SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION;
             element += blockDim.x)
        {
            local_head = element / SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION;
            partial = element % SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION;
            out_base =
                ((uint64_t)row * (uint64_t)local_heads *
                    SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) +
                ((uint64_t)((blockIdx.x * heads_per_cta) + local_head) *
                    SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION);
            SparkLmFloatToBf16(
                head_out_bf16,
                out_base + partial,
                invalid_output);
        }
        return;
    }

    for (local_head = 0u; local_head < heads_per_cta; ++local_head)
    {
        head = head_base + local_head;
        q_base =
            ((uint64_t)row * 2u * (uint64_t)local_heads *
                SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) +
            ((uint64_t)((blockIdx.x * heads_per_cta) + local_head) *
                2u * SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION);
        for (element = threadIdx.x;
             element < SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION;
             element += blockDim.x)
        {
            q_shared[
                (local_head * SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) +
                element] = SparkLmBf16ToFloat(
                    q_fused_bf16,
                    q_base + element);
        }
        running_max[local_head] = -3.0e38f;
        running_den[local_head] = 0.0f;
        #pragma unroll
        for (pair = 0u;
             pair < SPARK_QWEN4_FLASH_CUDA_ATTN_VALUE_PAIRS_PER_LANE;
             ++pair)
        {
            accumulator[local_head][pair] = make_float2(0.0f, 0.0f);
        }
    }
    for (element = threadIdx.x;
         element < SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA *
             SPARK_LM_CTA_WARPS *
             SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION;
         element += blockDim.x)
    {
        merge_acc[element] = 0.0f;
    }
    __syncthreads();

    lane_base = (uint64_t)lane_index * lane_stride;
    for (token = warp; token < context; token += SPARK_LM_CTA_WARPS)
    {
        if (token_mask != 0 && token_mask[((uint64_t)row * mask_stride) + token] == 0u)
            continue;
        token_base = SparkQwen4FlashAttnTokenBase(
            block_indices,
            lane_base,
            token,
            attn_layer_ordinal,
            cache_layer_stride,
            cache_block_stride,
            kv_head - ((SPARK_QWEN4_FLASH_MODEL_ATTN_KV_HEAD_COUNT % tp_degree) != 0u ? 0u : tp_rank * local_kv_heads),
            local_token_elements);
        #pragma unroll
        for (local_head = 0u; local_head < heads_per_cta; ++local_head)
        {
            local_logit[local_head] = 0.0f;
        }
        #pragma unroll
        for (pair = 0u;
             pair < SPARK_QWEN4_FLASH_CUDA_ATTN_VALUE_PAIRS_PER_LANE;
             ++pair)
        {
            key_pair = SparkLmLoadBf16Pair(
                kv_cache_bf16,
                (token_base >> 1u) +
                    (pair * SPARK_LM_WARP_LANES) + lane);
            element = ((pair * SPARK_LM_WARP_LANES) + lane) << 1u;
            #pragma unroll
            for (local_head = 0u;
                 local_head < heads_per_cta;
                 ++local_head)
            {
                local_logit[local_head] = fmaf(
                    q_shared[
                        (local_head *
                            SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) +
                        element],
                    key_pair.x,
                    local_logit[local_head]);
                local_logit[local_head] = fmaf(
                    q_shared[
                        (local_head *
                            SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) +
                        element + 1u],
                    key_pair.y,
                    local_logit[local_head]);
            }
        }
        #pragma unroll
        for (local_head = 0u; local_head < heads_per_cta; ++local_head)
        {
            local_logit[local_head] = __shfl_sync(
                0xffffffffu,
                SparkLmWarpReduceSum(local_logit[local_head]),
                0) * (1.0f / 16.0f);
            rescale[local_head] = 0.0f;
            weight[local_head] = 0.0f;
            if (lane == 0u)
            {
                rescale[local_head] =
                    local_logit[local_head] > running_max[local_head]
                    ? __expf(
                        running_max[local_head] - local_logit[local_head])
                    : 1.0f;
                weight[local_head] =
                    local_logit[local_head] > running_max[local_head]
                    ? 1.0f
                    : __expf(
                        local_logit[local_head] - running_max[local_head]);
                running_max[local_head] =
                    local_logit[local_head] > running_max[local_head]
                    ? local_logit[local_head]
                    : running_max[local_head];
                running_den[local_head] = fmaf(
                    running_den[local_head],
                    rescale[local_head],
                    weight[local_head]);
            }
            rescale[local_head] = __shfl_sync(
                0xffffffffu,
                rescale[local_head],
                0);
            weight[local_head] = __shfl_sync(
                0xffffffffu,
                weight[local_head],
                0);
        }
        #pragma unroll
        for (pair = 0u;
             pair < SPARK_QWEN4_FLASH_CUDA_ATTN_VALUE_PAIRS_PER_LANE;
             ++pair)
        {
            value_pair = SparkLmLoadBf16Pair(
                kv_cache_bf16,
                ((token_base + ((uint64_t)local_kv_heads *
                    SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION)) >> 1u) +
                    (pair * SPARK_LM_WARP_LANES) + lane);
            #pragma unroll
            for (local_head = 0u;
                 local_head < heads_per_cta;
                 ++local_head)
            {
                accumulator[local_head][pair].x = fmaf(
                    accumulator[local_head][pair].x,
                    rescale[local_head],
                    weight[local_head] * value_pair.x);
                accumulator[local_head][pair].y = fmaf(
                    accumulator[local_head][pair].y,
                    rescale[local_head],
                    weight[local_head] * value_pair.y);
            }
        }
    }

    #pragma unroll
    for (local_head = 0u; local_head < heads_per_cta; ++local_head)
    {
        if (lane == 0u)
        {
            merge_max[(local_head * SPARK_LM_CTA_WARPS) + warp] =
                running_max[local_head];
            merge_den[(local_head * SPARK_LM_CTA_WARPS) + warp] =
                running_den[local_head];
        }
        #pragma unroll
        for (pair = 0u;
             pair < SPARK_QWEN4_FLASH_CUDA_ATTN_VALUE_PAIRS_PER_LANE;
             ++pair)
        {
            element = ((pair * SPARK_LM_WARP_LANES) + lane) << 1u;
            merge_acc[
                (((local_head * SPARK_LM_CTA_WARPS) + warp) *
                    SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) +
                element] = accumulator[local_head][pair].x;
            merge_acc[
                (((local_head * SPARK_LM_CTA_WARPS) + warp) *
                    SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) +
                element + 1u] = accumulator[local_head][pair].y;
        }
    }
    __syncthreads();

    if (threadIdx.x < heads_per_cta)
    {
        local_head = threadIdx.x;
        head_max = merge_max[local_head * SPARK_LM_CTA_WARPS];
        for (partial = 1u; partial < SPARK_LM_CTA_WARPS; ++partial)
        {
            head_max = fmaxf(
                head_max,
                merge_max[(local_head * SPARK_LM_CTA_WARPS) + partial]);
        }
        denominator = 0.0f;
        for (partial = 0u; partial < SPARK_LM_CTA_WARPS; ++partial)
        {
            float partial_scale;

            partial_scale = __expf(
                merge_max[(local_head * SPARK_LM_CTA_WARPS) + partial] -
                head_max);
            merge_scale[(local_head * SPARK_LM_CTA_WARPS) + partial] =
                partial_scale;
            denominator = fmaf(
                merge_den[(local_head * SPARK_LM_CTA_WARPS) + partial],
                partial_scale,
                denominator);
        }
        inverse_denominator[local_head] = denominator > 0.0f
            ? 1.0f / denominator
            : 0.0f;
    }
    __syncthreads();

    for (local_head = 0u; local_head < heads_per_cta; ++local_head)
    {
        head = head_base + local_head;
        q_base =
            ((uint64_t)row * 2u * (uint64_t)local_heads *
                SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) +
            ((uint64_t)((blockIdx.x * heads_per_cta) + local_head) *
                2u * SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION);
        out_base =
            ((uint64_t)row * (uint64_t)local_heads *
                SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) +
            ((uint64_t)((blockIdx.x * heads_per_cta) + local_head) *
                SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION);
        for (element = threadIdx.x;
             element < SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION;
             element += blockDim.x)
        {
            merged = 0.0f;
            for (partial = 0u; partial < SPARK_LM_CTA_WARPS; ++partial)
            {
                merged = fmaf(
                    merge_acc[
                        (((local_head * SPARK_LM_CTA_WARPS) + partial) *
                            SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION) +
                        element],
                    merge_scale[
                        (local_head * SPARK_LM_CTA_WARPS) + partial],
                    merged);
            }
            gate = SparkLmSigmoid(SparkLmBf16ToFloat(
                q_fused_bf16,
                q_base + SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION + element));
            SparkLmFloatToBf16(
                head_out_bf16,
                out_base + element,
                merged * inverse_denominator[local_head] * gate);
        }
    }
}

static __global__ void SparkQwen4FlashEmbeddingGatherKernel(const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t row = (uint32_t)(index / SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION),element = (uint32_t)(index % SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION);
	if ( row >= row_count )
		return;
	SparkLmFloatToBf16(hidden_bf16,index,SparkLmBf16ToFloat(embedding_bf16,((uint64_t)token_ids[row] * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION) + element));
}

#define SPARK_QWEN4_FLASH_CUDA_CHUNK SPARK_QWEN4_FLASH_MODEL_GDN_CHUNK_TOKENS
#define SPARK_QWEN4_FLASH_CUDA_GDN_QK_SHARED_BYTES \
    (2u * SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_DK * sizeof(float))
#define SPARK_QWEN4_FLASH_CUDA_GDN_CHUNK_SHARED_BYTES \
    ((SPARK_QWEN4_FLASH_CUDA_GDN_STATE_ELEMENTS + \
      (SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_DV) + \
      (2u * SPARK_QWEN4_FLASH_CUDA_CHUNK)) * sizeof(float))
static_assert(
    SPARK_QWEN4_FLASH_CUDA_GDN_CHUNK_SHARED_BYTES == 98816u,
    "Qwen GDN chunk shared layout must fit the SM 12.x 99-KB block limit");

typedef struct SparkQwen4FlashChunkWorkspaceView
{
	float *qn;
	float *kn;
	float *cum_g;
	float *decay;
	float *attn;
	float *w;
	float *kg;
} SparkQwen4FlashChunkWorkspaceView;

static __device__ __forceinline__ uint64_t SparkQwen4FlashChunkHeadOffset(uint32_t head, uint32_t per_head_elements)
{
	return((uint64_t)head * per_head_elements);
}

static __global__ void SparkQwen4FlashChunkPrepareKernel(const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, SparkQwen4FlashChunkWorkspaceView views, uint32_t token_count, uint32_t tp_degree)
{
	const uint32_t local_value_heads = SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree;
	const uint32_t local_qk = (SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT / tp_degree) * SPARK_QWEN4_FLASH_CUDA_DK;
	const uint32_t local_conv = 2u * local_qk + (local_value_heads * SPARK_QWEN4_FLASH_CUDA_DV);
	uint32_t head = blockIdx.x,row = threadIdx.x,key_head = head / SPARK_QWEN4_FLASH_CUDA_GVA_GROUP,element,column;
	uint64_t conv_row,qk_base = SparkQwen4FlashChunkHeadOffset(head,SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_DK);
	uint64_t mat_base = SparkQwen4FlashChunkHeadOffset(head,SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_CHUNK);
	float total,value,product;
	if ( row >= token_count )
		return;
	conv_row = (uint64_t)row * local_conv;
	total = 0.0f;
	for (element = 0; element < SPARK_QWEN4_FLASH_CUDA_DK; element++)
	{
		value = SparkLmBf16ToFloat(conv_out_bf16,conv_row + ((uint64_t)key_head * SPARK_QWEN4_FLASH_CUDA_DK) + element);
		total += (value * value);
	}
	total = rsqrtf(total + 1e-6f) * rsqrtf((float)SPARK_QWEN4_FLASH_CUDA_DK);
	for (element = 0; element < SPARK_QWEN4_FLASH_CUDA_DK; element++)
		views.qn[qk_base + ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_DK) + element] = SparkLmBf16ToFloat(conv_out_bf16,conv_row + ((uint64_t)key_head * SPARK_QWEN4_FLASH_CUDA_DK) + element) * total;
	total = 0.0f;
	for (element = 0; element < SPARK_QWEN4_FLASH_CUDA_DK; element++)
	{
		value = SparkLmBf16ToFloat(conv_out_bf16,conv_row + local_qk + ((uint64_t)key_head * SPARK_QWEN4_FLASH_CUDA_DK) + element);
		total += (value * value);
	}
	total = rsqrtf(total + 1e-6f);
	for (element = 0; element < SPARK_QWEN4_FLASH_CUDA_DK; element++)
		views.kn[qk_base + ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_DK) + element] = SparkLmBf16ToFloat(conv_out_bf16,conv_row + local_qk + ((uint64_t)key_head * SPARK_QWEN4_FLASH_CUDA_DK) + element) * total;
	if ( row == 0u )
	{
		total = 0.0f;
		for (element = 0; element < token_count; element++)
		{
			total += log_decay_f32[((uint64_t)element * local_value_heads) + head];
			views.cum_g[SparkQwen4FlashChunkHeadOffset(head,SPARK_QWEN4_FLASH_CUDA_CHUNK) + element] = total;
		}
	}
	__syncthreads();
	for (column = 0; column < token_count; column++)
	{
		value = column <= row ? __expf(views.cum_g[SparkQwen4FlashChunkHeadOffset(head,SPARK_QWEN4_FLASH_CUDA_CHUNK) + row] - views.cum_g[SparkQwen4FlashChunkHeadOffset(head,SPARK_QWEN4_FLASH_CUDA_CHUNK) + column]) : 0.0f;
		views.decay[mat_base + ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_CHUNK) + column] = value;
		product = 0.0f;
		for (element = 0; element < SPARK_QWEN4_FLASH_CUDA_DK && column < row; element++)
			product += (views.kn[qk_base + ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_DK) + element] * beta_f32[((uint64_t)row * local_value_heads) + head] * views.kn[qk_base + ((uint64_t)column * SPARK_QWEN4_FLASH_CUDA_DK) + element]);
		views.attn[mat_base + ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_CHUNK) + column] = column < row ? -(product * value) : 0.0f;
	}
}

static __global__ void SparkQwen4FlashChunkSolveKernel(SparkQwen4FlashChunkWorkspaceView views, uint32_t token_count)
{
	__shared__ float solve_row[SPARK_QWEN4_FLASH_CUDA_CHUNK];
	uint32_t head = blockIdx.x,column = threadIdx.x,row,element;
	uint64_t mat_base = SparkQwen4FlashChunkHeadOffset(head,SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_CHUNK);
	float accumulator;
	for (row = 1; row < token_count; row++)
	{
		if ( column < row )
			solve_row[column] = views.attn[mat_base + ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_CHUNK) + column];
		__syncthreads();
		if ( column < row )
		{
			accumulator = solve_row[column];
			for (element = 0; element < row; element++)
				accumulator += (solve_row[element] * views.attn[mat_base + ((uint64_t)element * SPARK_QWEN4_FLASH_CUDA_CHUNK) + column]);
			views.attn[mat_base + ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_CHUNK) + column] = accumulator;
		}
		__syncthreads();
	}
	if ( column < token_count )
		views.attn[mat_base + ((uint64_t)column * SPARK_QWEN4_FLASH_CUDA_CHUNK) + column] += 1.0f;
}

static __global__ void SparkQwen4FlashChunkTransformKernel(const void *conv_out_bf16, const float *beta_f32, SparkQwen4FlashChunkWorkspaceView views, uint32_t token_count, uint32_t tp_degree)
{
	const uint32_t local_value_heads = SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree;
	const uint32_t local_qk = (SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT / tp_degree) * SPARK_QWEN4_FLASH_CUDA_DK;
	const uint32_t local_conv = 2u * local_qk + (local_value_heads * SPARK_QWEN4_FLASH_CUDA_DV);
	__shared__ float exp_cum_g[SPARK_QWEN4_FLASH_CUDA_CHUNK];
	uint32_t head = blockIdx.x,row = blockIdx.y,column = threadIdx.x,element;
	uint64_t mat_base = SparkQwen4FlashChunkHeadOffset(head,SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_CHUNK);
	uint64_t vec_base = SparkQwen4FlashChunkHeadOffset(head,SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_DK);
	float accumulator,transform;
	if ( row >= token_count )
		return;
	if ( threadIdx.x < token_count )
		exp_cum_g[threadIdx.x] = __expf(
			views.cum_g[
				SparkQwen4FlashChunkHeadOffset(
					head,
					SPARK_QWEN4_FLASH_CUDA_CHUNK) +
				threadIdx.x]);
	__syncthreads();
	accumulator = 0.0f;
	for (element = 0; element < token_count; element++)
	{
		transform = views.attn[mat_base + ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_CHUNK) + element] * beta_f32[((uint64_t)element * local_value_heads) + head];
		accumulator += (transform * SparkLmBf16ToFloat(conv_out_bf16,((uint64_t)element * local_conv) + (2u * local_qk) + ((uint64_t)head * SPARK_QWEN4_FLASH_CUDA_DV) + column));
	}
	views.w[vec_base + ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_DV) + column] = accumulator;
	accumulator = 0.0f;
	for (element = 0; element < token_count; element++)
	{
		transform = views.attn[mat_base + ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_CHUNK) + element] * beta_f32[((uint64_t)element * local_value_heads) + head] * exp_cum_g[element];
		accumulator += (transform * views.kn[vec_base + ((uint64_t)element * SPARK_QWEN4_FLASH_CUDA_DK) + column]);
	}
	views.kg[vec_base + ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_DK) + column] = accumulator;
}


static __global__ void SparkQwen4FlashChunkQkDecayKernel(SparkQwen4FlashChunkWorkspaceView views, uint32_t token_count)
{
    extern __shared__ float qk_shared[];
    float *qn_shared;
    float *kn_shared;
    uint32_t head;
    uint32_t vector_element;
    uint32_t matrix_element;
    uint32_t row;
    uint32_t column;
    uint32_t element;
    uint64_t vector_base;
    uint64_t matrix_base;
    float dot;

    head = blockIdx.x;
    qn_shared = qk_shared;
    kn_shared = qn_shared +
        (SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_DK);
    vector_base = SparkQwen4FlashChunkHeadOffset(
        head,
        SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_DK);
    matrix_base = SparkQwen4FlashChunkHeadOffset(
        head,
        SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_CHUNK);

    for (vector_element = threadIdx.x;
         vector_element < token_count * SPARK_QWEN4_FLASH_CUDA_DK;
         vector_element += blockDim.x)
    {
        qn_shared[vector_element] = views.qn[vector_base + vector_element];
        kn_shared[vector_element] = views.kn[vector_base + vector_element];
    }
    __syncthreads();

    for (matrix_element = threadIdx.x;
         matrix_element < token_count * token_count;
         matrix_element += blockDim.x)
    {
        row = matrix_element / token_count;
        column = matrix_element % token_count;
        if (column <= row)
        {
            dot = 0.0f;
            for (element = 0u; element < SPARK_QWEN4_FLASH_CUDA_DK; ++element)
            {
                dot = fmaf(
                    qn_shared[(row * SPARK_QWEN4_FLASH_CUDA_DK) + element],
                    kn_shared[(column * SPARK_QWEN4_FLASH_CUDA_DK) + element],
                    dot);
            }
            views.decay[
                matrix_base +
                ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_CHUNK) + column] *= dot;
        }
        else
        {
            views.decay[
                matrix_base +
                ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_CHUNK) + column] = 0.0f;
        }
    }
}

static __global__ void SparkQwen4FlashChunkStepKernel(const float *log_decay_f32, SparkQwen4FlashChunkWorkspaceView views, float *state_f32, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal, uint64_t state_lane_stride, uint64_t state_layer_stride, uint32_t tp_degree)
{
    extern __shared__ float chunk_shared[];
    float *state_shared;
    float *v_new_shared;
    float *exp_cum_g_shared;
    float *carry_decay_shared;
    uint32_t head;
    uint32_t column;
    uint32_t row;
    uint32_t element;
    uint32_t state_index;
    uint64_t vector_base;
    uint64_t g_base;
    uint64_t matrix_base;
    uint64_t state_base;
    float accumulator;
    float g_last;
    float carry;

    head = blockIdx.x;
    column = threadIdx.x;
    state_shared = chunk_shared;
    v_new_shared = state_shared + SPARK_QWEN4_FLASH_CUDA_GDN_STATE_ELEMENTS;
    exp_cum_g_shared =
        v_new_shared + (SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_DV);
    carry_decay_shared =
        exp_cum_g_shared + SPARK_QWEN4_FLASH_CUDA_CHUNK;
    vector_base = SparkQwen4FlashChunkHeadOffset(
        head,
        SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_DK);
    g_base = SparkQwen4FlashChunkHeadOffset(
        head,
        SPARK_QWEN4_FLASH_CUDA_CHUNK);
    matrix_base = SparkQwen4FlashChunkHeadOffset(
        head,
        SPARK_QWEN4_FLASH_CUDA_CHUNK * SPARK_QWEN4_FLASH_CUDA_CHUNK);
    state_base =
        ((uint64_t)lane_index * state_lane_stride) +
        ((uint64_t)gdn_layer_ordinal * state_layer_stride) +
        ((uint64_t)head * SPARK_QWEN4_FLASH_CUDA_GDN_STATE_ELEMENTS);
    g_last = views.cum_g[g_base + token_count - 1u];
    (void)log_decay_f32;

    if (column < token_count)
    {
        exp_cum_g_shared[column] = __expf(
            views.cum_g[g_base + column]);
        carry_decay_shared[column] = __expf(
            g_last - views.cum_g[g_base + column]);
    }
    __syncthreads();

    for (element = 0u; element < SPARK_QWEN4_FLASH_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN4_FLASH_CUDA_DV) + column;
        state_shared[state_index] = state_f32[state_base + state_index];
    }

    for (row = 0u; row < token_count; ++row)
    {
        accumulator = 0.0f;
        for (element = 0u; element < SPARK_QWEN4_FLASH_CUDA_DK; ++element)
        {
            accumulator = fmaf(
                views.kg[
                    vector_base +
                    ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_DK) + element],
                state_shared[(element * SPARK_QWEN4_FLASH_CUDA_DV) + column],
                accumulator);
        }
        v_new_shared[(row * SPARK_QWEN4_FLASH_CUDA_DV) + column] =
            views.w[
                vector_base +
                ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_DV) + column] - accumulator;
    }

    for (row = 0u; row < token_count; ++row)
    {
        accumulator = 0.0f;
        for (element = 0u; element < SPARK_QWEN4_FLASH_CUDA_DK; ++element)
        {
            accumulator = fmaf(
                views.qn[
                    vector_base +
                    ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_DK) + element],
                state_shared[(element * SPARK_QWEN4_FLASH_CUDA_DV) + column],
                accumulator);
        }
        accumulator *= exp_cum_g_shared[row];
        for (element = 0u; element <= row; ++element)
        {
            accumulator = fmaf(
                views.decay[
                    matrix_base +
                    ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_CHUNK) + element],
                v_new_shared[(element * SPARK_QWEN4_FLASH_CUDA_DV) + column],
                accumulator);
        }
        SparkLmFloatToBf16(
            core_out_bf16,
            ((uint64_t)row * (SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree) * SPARK_QWEN4_FLASH_CUDA_DV) +
                ((uint64_t)head * SPARK_QWEN4_FLASH_CUDA_DV) + column,
            accumulator);
    }

    for (element = 0u; element < SPARK_QWEN4_FLASH_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN4_FLASH_CUDA_DV) + column;
        carry = state_shared[state_index] *
            exp_cum_g_shared[token_count - 1u];
        for (row = 0u; row < token_count; ++row)
        {
            carry = fmaf(
                views.kn[
                    vector_base +
                    ((uint64_t)row * SPARK_QWEN4_FLASH_CUDA_DK) + element] *
                    carry_decay_shared[row],
                v_new_shared[(row * SPARK_QWEN4_FLASH_CUDA_DV) + column],
                carry);
        }
        state_shared[state_index] = carry;
        state_f32[state_base + state_index] = state_shared[state_index];
    }
}

static __global__ void SparkQwen4FlashChunkConvKernel(const void *qkv_bf16, const void *conv_weight_bf16, void *conv_out_bf16, void *conv_tail_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal, uint64_t tail_lane_stride, uint64_t tail_layer_stride, uint32_t tp_degree)
{
	const uint32_t local_conv_channels = 2u * ((SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT / tp_degree) * SPARK_QWEN4_FLASH_CUDA_DK) + ((SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree) * SPARK_QWEN4_FLASH_CUDA_DV);
	uint32_t channel = (blockIdx.x * blockDim.x) + threadIdx.x,token,tap;
	uint64_t tail_base,element;
	float window[4],weight[4],accumulator;
	if ( channel >= local_conv_channels )
		return;
	tail_base = ((uint64_t)lane_index * tail_lane_stride) + ((uint64_t)gdn_layer_ordinal * tail_layer_stride) + ((uint64_t)channel * SPARK_QWEN4_FLASH_MODEL_GDN_CONV_TAIL_COLUMNS);
	window[0] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 0u);
	window[1] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 1u);
	window[2] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 2u);
	for (tap = 0; tap < SPARK_QWEN4_FLASH_MODEL_GDN_CONV_KERNEL; tap++)
		weight[tap] = SparkLmBf16ToFloat(conv_weight_bf16,((uint64_t)channel * SPARK_QWEN4_FLASH_MODEL_GDN_CONV_KERNEL) + tap);
	for (token = 0; token < token_count; token++)
	{
		element = ((uint64_t)token * local_conv_channels) + channel;
		window[3] = SparkLmBf16ToFloat(qkv_bf16,element);
		accumulator = 0.0f;
		for (tap = 0; tap < SPARK_QWEN4_FLASH_MODEL_GDN_CONV_KERNEL; tap++)
			accumulator += (window[tap] * weight[tap]);
		SparkLmFloatToBf16(conv_out_bf16,element,SparkLmSwish(accumulator));
		window[0] = window[1];
		window[1] = window[2];
		window[2] = window[3];
	}
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 0u,window[0]);
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 1u,window[1]);
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 2u,window[2]);
}

static __global__ void SparkQwen4FlashResidualAddKernel(void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t pair = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x,pair_count = ((uint64_t)row_count * dimension) >> 1u;
	float2 hidden_pair,delta_pair;
	if ( pair >= pair_count )
		return;
	hidden_pair = SparkLmLoadBf16Pair(hidden_bf16,pair);
	delta_pair = SparkLmLoadBf16Pair(delta_bf16,pair);
	SparkLmStoreBf16Pair(hidden_bf16,pair,hidden_pair.x + delta_pair.x,hidden_pair.y + delta_pair.y);
	if ( pair == 0u && (((uint64_t)row_count * dimension) & 1u) != 0u )
		SparkLmFloatToBf16(hidden_bf16,((uint64_t)row_count * dimension) - 1u,SparkLmBf16ToFloat(hidden_bf16,((uint64_t)row_count * dimension) - 1u) + SparkLmBf16ToFloat(delta_bf16,((uint64_t)row_count * dimension) - 1u));
}

extern "C" cudaError_t SparkQwen4FlashLaunchFusedResidualRmsNorm(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
    size_t shared_memory_bytes = (size_t)dimension * sizeof(float);

    SparkLmFusedResidualRmsNormKernel<<<row_count, SPARK_LM_CTA_THREADS, shared_memory_bytes, stream>>>(hidden_bf16, delta_bf16, gain_bf16, output_bf16, row_count, dimension, epsilon);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkQwen4FlashLaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
    size_t shared_memory_bytes = (size_t)dimension * sizeof(float);

    SparkLmRmsNormKernel<<<row_count, SPARK_LM_CTA_THREADS, shared_memory_bytes, stream>>>(input_bf16, gain_bf16, output_bf16, row_count, dimension, epsilon);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkQwen4FlashLaunchLinear(cudaStream_t stream, const SparkQwen4FlashLinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count)
{
	if ( view->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 && row_count >= 2u * SPARK_LM_TILE && view->input_dimension > SPARK_LM_TILE_K && view->output_dimension != 0u )
		return(SparkLmHostLaunchBatchedLinearMloop(stream,view->weight_payload,input_bf16,output_bf16,row_count,view->input_dimension,view->output_dimension));
	return(SparkLmHostLaunchBatchedLinear<32u>(stream,view->weight_format,view->weight_payload,view->weight_scale_e8m0,input_bf16,output_bf16,row_count,view->input_dimension,view->output_dimension));
}

extern "C" cudaError_t SparkQwen4FlashLaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen4FlashGdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen4FlashGdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal, uint32_t tp_degree)
{
	uint32_t local_conv_channels = 2u * ((SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT / tp_degree) * SPARK_QWEN4_FLASH_CUDA_DK) + ((SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree) * SPARK_QWEN4_FLASH_CUDA_DV);
	dim3 grid((local_conv_channels + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS,row_count,1u);
	SparkQwen4FlashConvUpdateKernel<<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(qkv_bf16,weights->conv_weight_bf16,conv_out_bf16,pool->conv_tail_bf16,row_lane_indices,pool->state_cold_by_row,row_count,gdn_layer_ordinal,pool->conv_tail_lane_stride_elements,pool->conv_tail_layer_stride_elements,local_conv_channels);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen4FlashLaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen4FlashGdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count, uint32_t tp_degree)
{
	uint32_t local_value_heads = SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree;
	SparkQwen4FlashDecayBetaKernel<<<row_count,local_value_heads,0,stream>>>(decay_pre_bf16,beta_pre_bf16,weights->a_log_f32,weights->dt_bias_f32,log_decay_f32,beta_f32,row_count,local_value_heads);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen4FlashLaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen4FlashGdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal, uint32_t tp_degree)
{
    dim3 grid(
        SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree,
        row_count,
        1u);
    SparkQwen4FlashGdnStepKernel<<<
        grid,
        SPARK_QWEN4_FLASH_CUDA_DV,
        SPARK_QWEN4_FLASH_CUDA_GDN_DECODE_SHARED_BYTES,
        stream>>>(
            conv_out_bf16,
            log_decay_f32,
            beta_f32,
            pool->state_f32,
            core_out_bf16,
            row_lane_indices,
            pool->state_cold_by_row,
            row_count,
            gdn_layer_ordinal,
            pool->state_lane_stride_elements,
            pool->state_layer_stride_elements,
            tp_degree);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkQwen4FlashLaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen4FlashGdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon, uint32_t tp_degree)
{
	uint32_t local_value_heads = SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree;
	uint32_t local_value_dimension = local_value_heads * SPARK_QWEN4_FLASH_CUDA_DV;
	dim3 grid(local_value_heads,row_count,1u);
	SparkQwen4FlashGatedNormKernel<<<grid,SPARK_QWEN4_FLASH_CUDA_DV,0,stream>>>(core_bf16,z_bf16,weights->gdn_norm_weight_bf16,output_bf16,row_count,epsilon,local_value_heads,local_value_dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen4FlashLaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen4FlashAttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon, uint32_t tp_degree, uint32_t tp_rank)
{
	if ( tp_degree == 0u || tp_degree > SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT || tp_rank >= tp_degree || (SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT % tp_degree) != 0u || 0u )
		return(cudaErrorInvalidValue);
	dim3 grid(SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT / tp_degree,row_count,1u);
	SparkQwen4FlashAttnPrepareKernel<<<grid,SPARK_QWEN4_FLASH_MODEL_ATTN_HEAD_DIMENSION,0,stream>>>(q_fused_bf16,k_bf16,v_bf16,weights->query_norm_weight_bf16,weights->key_norm_weight_bf16,kv_cache_bf16,slot_mapping,row_positions,row_count,attn_layer_ordinal,cache_layer_stride,cache_block_stride,epsilon,tp_degree,tp_rank);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen4FlashLaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen4FlashKvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t tp_degree, uint32_t tp_rank, const uint8_t *token_mask, uint32_t mask_stride)
{
    if ( tp_degree == 0u || tp_degree > SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT || tp_rank >= tp_degree || (SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT % tp_degree) != 0u || 0u )
        return(cudaErrorInvalidValue);
    dim3 grid(
        (SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT / tp_degree) /
            SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA,
        row_count,
        1u);
    static_assert(
        SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT %
            SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA == 0u,
        "query-head count must divide the grouped attention CTA width");
    static_assert(
        SPARK_QWEN4_FLASH_CUDA_ATTN_GROUP %
            SPARK_QWEN4_FLASH_CUDA_ATTN_HEADS_PER_CTA == 0u,
        "a grouped attention CTA may not cross KV-head ownership");
    SparkQwen4FlashAttnDecodeKernel<<<grid, SPARK_LM_CTA_THREADS, 0u, stream>>>(
        q_fused_bf16,
        kv_cache_bf16,
        table->physical_block_indices,
        table->lane_physical_block_counts,
        row_lane_indices,
        context_lengths,
        head_out_bf16,
        row_count,
        table->lane_stride,
        attn_layer_ordinal,
        cache_layer_stride,
        cache_block_stride,
        tp_degree,
        tp_rank,
        token_mask,
        mask_stride);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkQwen4FlashLaunchChunkConv(cudaStream_t stream, const void *qkv_bf16, const SparkQwen4FlashGdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen4FlashGdnStatePool *pool, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal, uint32_t tp_degree)
{
	uint32_t local_conv_channels = 2u * ((SPARK_QWEN4_FLASH_MODEL_GDN_KEY_HEAD_COUNT / tp_degree) * SPARK_QWEN4_FLASH_CUDA_DK) + ((SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree) * SPARK_QWEN4_FLASH_CUDA_DV);
	if ( token_count == 0u )
		return(cudaErrorInvalidValue);
	SparkQwen4FlashChunkConvKernel<<<(local_conv_channels + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS,SPARK_LM_CTA_THREADS,0,stream>>>(qkv_bf16,weights->conv_weight_bf16,conv_out_bf16,pool->conv_tail_bf16,lane_index,token_count,gdn_layer_ordinal,pool->conv_tail_lane_stride_elements,pool->conv_tail_layer_stride_elements,tp_degree);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen4FlashLaunchGdnChunk(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *workspace_qn, float *workspace_kn, float *workspace_cum_g, float *workspace_decay, float *workspace_attn, float *workspace_w, float *workspace_kg, const SparkQwen4FlashGdnStatePool *pool, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal, uint32_t tp_degree)
{
    SparkQwen4FlashChunkWorkspaceView views;
    cudaError_t status;
    const uint32_t local_value_heads =
        SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT / tp_degree;
    dim3 transform_grid(
        local_value_heads,
        token_count,
        1u);

    views.qn = workspace_qn;
    views.kn = workspace_kn;
    views.cum_g = workspace_cum_g;
    views.decay = workspace_decay;
    views.attn = workspace_attn;
    views.w = workspace_w;
    views.kg = workspace_kg;
    if (token_count == 0u || token_count > SPARK_QWEN4_FLASH_CUDA_CHUNK)
    {
        return cudaErrorInvalidValue;
    }

    SparkQwen4FlashChunkPrepareKernel<<<
        local_value_heads,
        SPARK_QWEN4_FLASH_CUDA_CHUNK,
        0u,
        stream>>>(conv_out_bf16, log_decay_f32, beta_f32, views, token_count, tp_degree);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    SparkQwen4FlashChunkSolveKernel<<<
        local_value_heads,
        SPARK_QWEN4_FLASH_CUDA_CHUNK,
        0u,
        stream>>>(views, token_count);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    SparkQwen4FlashChunkTransformKernel<<<
        transform_grid,
        SPARK_QWEN4_FLASH_CUDA_DV,
        0u,
        stream>>>(conv_out_bf16, beta_f32, views, token_count, tp_degree);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    SparkQwen4FlashChunkQkDecayKernel<<<
        local_value_heads,
        SPARK_LM_CTA_THREADS,
        SPARK_QWEN4_FLASH_CUDA_GDN_QK_SHARED_BYTES,
        stream>>>(views, token_count);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    SparkQwen4FlashChunkStepKernel<<<
        local_value_heads,
        SPARK_QWEN4_FLASH_CUDA_DV,
        SPARK_QWEN4_FLASH_CUDA_GDN_CHUNK_SHARED_BYTES,
        stream>>>(
            log_decay_f32,
            views,
            pool->state_f32,
            core_out_bf16,
            lane_index,
            token_count,
            gdn_layer_ordinal,
            pool->state_lane_stride_elements,
            pool->state_layer_stride_elements,
            tp_degree);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkQwen4FlashLaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count)
{
	uint64_t elements = (uint64_t)row_count * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
	SparkQwen4FlashEmbeddingGatherKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(token_ids,embedding_bf16,hidden_bf16,row_count);
	return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashTpCombineAddKernel(void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width)
{
	uint32_t row = blockIdx.x;
	uint64_t pair_base = ((uint64_t)row * width) >> 1u;
	uint64_t pair_count = width >> 1u;
	uint64_t pair;
	float2 dst,src;
	if ( row >= row_count )
		return;
	for (pair = threadIdx.x; pair < pair_count; pair += blockDim.x)
	{
		dst = SparkLmLoadBf16Pair(destination_bf16,pair_base + pair);
		src = SparkLmLoadBf16Pair(source_bf16,pair_base + pair);
		SparkLmStoreBf16Pair(destination_bf16,pair_base + pair,dst.x + src.x,dst.y + src.y);
	}
}

extern "C" cudaError_t SparkQwen4FlashLaunchTpCombineAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkQwen4FlashTpCombineAddKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen4FlashLaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t pairs = ((uint64_t)row_count * dimension + 1u) >> 1u;
	SparkQwen4FlashResidualAddKernel<<<(uint32_t)((pairs + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,delta_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen4FlashLaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchHeadShadowQuantize<SPARK_LM_HEAD_SHADOW_GROUP>(stream,head_bf16,shadow_payload,shadow_scale,error_norm,candidate_count,hidden_dimension));
}

extern "C" cudaError_t SparkQwen4FlashLaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count)
{
	return(SparkLmHostLaunchHeadScreenedArgmax(stream,hidden_bf16,head_weight_bf16,shadow_payload,shadow_scale,error_norm,logits_bf16,candidate_ids,candidate_counts,output_token_ids,row_count,candidate_count,SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION));
}

extern "C" cudaError_t SparkQwen4FlashLaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count)
{
	SparkLmHeadArgmaxKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,head_weight_bf16,token_ids,output_token_ids,row_count,SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION,candidate_count);
	return(cudaGetLastError());
}


static __device__ __forceinline__ uint32_t SparkQwen4FlashHeadOrderKey(float score)
{
	uint32_t bits = __float_as_uint(score);
	return((bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u);
}

static __global__ void SparkQwen4FlashHeadMaxLocPackKernel(const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count)
{
	uint32_t row = blockIdx.x;
	if ( row >= row_count )
		return;
	keys_u64[row] = ((uint64_t)SparkQwen4FlashHeadOrderKey(scores_f32[row]) << 32u) | (uint64_t)token_ids_u32[row];
}

extern "C" cudaError_t SparkQwen4FlashLaunchHeadScreenedArgmaxScore(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *scratch_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, float *output_scores, uint32_t candidate_offset, uint32_t row_count, uint32_t candidate_count)
{
	return(SparkLmHostLaunchHeadScreenedArgmaxWithScore(stream,hidden_bf16,head_weight_bf16,shadow_payload,shadow_scale,error_norm,scratch_bf16,candidate_ids,candidate_counts,output_token_ids,output_scores,candidate_offset,row_count,candidate_count,SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION));
}

extern "C" cudaError_t SparkQwen4FlashLaunchHeadMaxLocPack(cudaStream_t stream, const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count)
{
	SparkQwen4FlashHeadMaxLocPackKernel<<<row_count,1u,0,stream>>>(scores_f32,token_ids_u32,keys_u64,row_count);
	return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashHeadMaxLocUnpackKernel(const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count)
{
	uint32_t row = blockIdx.x;
	if ( row >= row_count )
		return;
	token_ids_u32[row] = (uint32_t)keys_u64[row];
}

extern "C" cudaError_t SparkQwen4FlashLaunchHeadMaxLocUnpack(cudaStream_t stream, const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count)
{
	SparkQwen4FlashHeadMaxLocUnpackKernel<<<row_count,1u,0,stream>>>(keys_u64,token_ids_u32,row_count);
	return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashTpCombineU64MaxKernel(uint64_t *destination, const uint64_t *source, uint32_t element_count)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t value;
	if ( index >= (uint64_t)element_count )
		return;
	value = source[index];
	if ( value > destination[index] )
		destination[index] = value;
}

extern "C" cudaError_t SparkQwen4FlashLaunchTpCombineU64Max(cudaStream_t stream, uint64_t *destination, const uint64_t *source, uint32_t element_count)
{
	uint32_t blocks = (element_count + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS;
	if ( destination == 0 || source == 0 || element_count == 0u )
		return(cudaErrorInvalidValue);
	SparkQwen4FlashTpCombineU64MaxKernel<<<blocks == 0u ? 1u : blocks,SPARK_LM_CTA_THREADS,0,stream>>>(destination,source,element_count);
	return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashEmbeddingGatherShardedKernel(const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t vocab_base, uint32_t vocab_rows)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t row = (uint32_t)(index / SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION),element = (uint32_t)(index % SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION);
	uint32_t token;
	float value;
	if ( row >= row_count )
		return;
	token = token_ids[row];
	value = token >= vocab_base && token < vocab_base + vocab_rows
		? SparkLmBf16ToFloat(embedding_bf16,((uint64_t)(token - vocab_base) * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION) + element)
		: 0.0f;
	SparkLmFloatToBf16(hidden_bf16,index,value);
}

extern "C" cudaError_t SparkQwen4FlashLaunchEmbeddingGatherSharded(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t vocab_base, uint32_t vocab_rows)
{
	uint64_t elements = (uint64_t)row_count * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
	if ( embedding_bf16 == 0 || vocab_rows == 0u )
		return(cudaErrorInvalidValue);
	SparkQwen4FlashEmbeddingGatherShardedKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(token_ids,embedding_bf16,hidden_bf16,row_count,vocab_base,vocab_rows);
	return(cudaGetLastError());
}


static __global__ void SparkQwen4FlashHcStreamReplicateKernel(const void *input_bf16, void *streams_bf16, uint32_t row_count)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t element_count = (uint64_t)row_count * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
	if ( index >= element_count )
		return;
	uint32_t stream = (uint32_t)(index / SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION) / SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT;
	uint64_t source = ((uint64_t)(index / SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH)) * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION + (index % SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION);
	(void)stream;
	SparkLmFloatToBf16(streams_bf16,index,SparkLmBf16ToFloat(input_bf16,source));
}

extern "C" cudaError_t SparkQwen4FlashLaunchHcStreamReplicate(cudaStream_t stream, const void *input_bf16, void *streams_bf16, uint32_t row_count)
{
	uint64_t elements = (uint64_t)row_count * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
	SparkQwen4FlashHcStreamReplicateKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(input_bf16,streams_bf16,row_count);
	return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashHcGroupNormKernel(const void *streams_bf16, const void *weight_bf16, void *normed_bf16, uint32_t row_count, float epsilon)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint64_t row = blockIdx.y;
	uint32_t stream = blockIdx.x;
	uint64_t base = (row * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH) + ((uint64_t)stream * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION);
	float sum_squares = 0.0f;
	float value;
	if ( row >= row_count )
		return;
	for (uint32_t element = threadIdx.x; element < SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION; element += blockDim.x)
	{
		value = SparkLmBf16ToFloat(streams_bf16,base + element);
		sum_squares += value * value;
	}
	sum_squares = SparkLmBlockReduceSum(sum_squares,reduce_scratch);
	float inverse_rms = rsqrtf(sum_squares / (float)SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION + epsilon);
	for (uint32_t element = threadIdx.x; element < SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION; element += blockDim.x)
	{
		value = SparkLmBf16ToFloat(streams_bf16,base + element) * inverse_rms *
			SparkLmBf16ToFloat(weight_bf16,(uint64_t)stream * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION + element);
		SparkLmFloatToBf16(normed_bf16,base + element,value);
	}
}

extern "C" cudaError_t SparkQwen4FlashLaunchHcGroupNorm(cudaStream_t stream, const void *streams_bf16, const void *weight_bf16, void *normed_bf16, uint32_t row_count, float epsilon)
{
	dim3 grid(SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT,row_count,1u);
	SparkQwen4FlashHcGroupNormKernel<<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(streams_bf16,weight_bf16,normed_bf16,row_count,epsilon);
	return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashHcSiluQuarterKernel(void *lowrank_bf16, uint64_t element_count)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	if ( index >= element_count )
		return;
	SparkLmFloatToBf16(lowrank_bf16,index,SparkLmSwish(SparkLmBf16ToFloat(lowrank_bf16,index) / (float)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT));
}

extern "C" cudaError_t SparkQwen4FlashLaunchHcSiluQuarter(cudaStream_t stream, void *lowrank_bf16, uint32_t row_count)
{
	uint64_t elements = (uint64_t)row_count * SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION;
	uint32_t blocks = (uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS);
	SparkQwen4FlashHcSiluQuarterKernel<<<blocks == 0u ? 1u : blocks,SPARK_LM_CTA_THREADS,0,stream>>>(lowrank_bf16,elements);
	return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashHcMixKernel(const void *up_bf16, const void *normed_bf16, void *mixed_bf16, uint32_t row_count)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t element = (uint32_t)(index % SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION);
	uint64_t row = index / SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
	float sum;
	if ( row >= row_count )
		return;
	sum = 0.0f;
	for (uint32_t stream = 0u; stream < SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT; stream++)
	{
		uint64_t at = (row * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH) + ((uint64_t)stream * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION) + element;
		sum += SparkLmSigmoid(SparkLmBf16ToFloat(up_bf16,at)) * SparkLmBf16ToFloat(normed_bf16,at);
	}
	SparkLmFloatToBf16(mixed_bf16,index,sum / (float)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT);
}

extern "C" cudaError_t SparkQwen4FlashLaunchHcMix(cudaStream_t stream, const void *up_bf16, const void *normed_bf16, void *mixed_bf16, uint32_t row_count)
{
	uint64_t elements = (uint64_t)row_count * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
	SparkQwen4FlashHcMixKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(up_bf16,normed_bf16,mixed_bf16,row_count);
	return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashHcInjectKernel(void *streams_bf16, const void *inject_pre_bf16, const void *sublayer_out_bf16, uint32_t row_count)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t element = (uint32_t)(index % SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION);
	uint64_t row = index / SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
	uint32_t stream = (uint32_t)((index / SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION) % SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT);
	float scale;
	if ( row >= row_count )
		return;
	scale = 2.0f * SparkLmSigmoid(SparkLmBf16ToFloat(inject_pre_bf16,(row * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT) + stream) / (float)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT);
	SparkLmFloatToBf16(streams_bf16,index,SparkLmBf16ToFloat(streams_bf16,index) +
		scale * SparkLmBf16ToFloat(sublayer_out_bf16,(row * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION) + element));
}

extern "C" cudaError_t SparkQwen4FlashLaunchHcInject(cudaStream_t stream, void *streams_bf16, const void *inject_pre_bf16, const void *sublayer_out_bf16, uint32_t row_count)
{
	uint64_t elements = (uint64_t)row_count * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
	SparkQwen4FlashHcInjectKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(streams_bf16,inject_pre_bf16,sublayer_out_bf16,row_count);
	return(cudaGetLastError());
}


static __global__ void SparkQwen4FlashIndexerPrepareKernel(
    const void *qk_bf16,
    const void *q_norm_bf16,
    const void *k_norm_bf16,
    void *query_bf16,
    void *raw_key_cache,
    void *pooled_key_cache,
    const uint32_t *slot_mapping,
    const uint32_t *block_indices,
    const uint64_t *row_positions,
    uint32_t row_count,
    uint32_t lane_stride,
    uint64_t cache_block_stride)
{
    __shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
    __shared__ float head_shared[SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
    __shared__ float rope_cosine[SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u];
    __shared__ float rope_sine[SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u];
    uint32_t row = blockIdx.y;
    uint32_t column = threadIdx.x;
    uint32_t slot,block,offset;
    uint64_t key_base;
    float value,sum_squares,inverse_rms;
    if ( row >= row_count || column >= SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION )
        return;
    {
        uint32_t head = blockIdx.x;
        uint64_t qk_base = ((uint64_t)row * ((SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT + SPARK_QWEN4_FLASH_MODEL_INDEXER_KV_HEAD_COUNT) * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION)) + ((uint64_t)head * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION);
        value = SparkLmBf16ToFloat(qk_bf16,qk_base + column);
        sum_squares = SparkLmBlockReduceSum(value * value,reduce_scratch);
        inverse_rms = rsqrtf(sum_squares / (float)SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION + 1e-6f);
        head_shared[column] = value * inverse_rms * SparkLmBf16ToFloat(q_norm_bf16,column);
        __syncthreads();
        if ( column < SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u )
        {
            float angle = (float)row_positions[row] * SparkQwen4FlashRopeFrequency(column);
            sincosf(angle,&rope_sine[column],&rope_cosine[column]);
        }
        __syncthreads();
        if ( column < SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u )
        {
            float real = head_shared[column];
            float imaginary = head_shared[column + SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u];
            head_shared[column] = real * rope_cosine[column] - imaginary * rope_sine[column];
            head_shared[column + SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u] = imaginary * rope_cosine[column] + real * rope_sine[column];
        }
        __syncthreads();
        SparkLmFloatToBf16(query_bf16,((uint64_t)row * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION) + ((uint64_t)head * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION) + column,head_shared[column]);
        __syncthreads();
    }
    if ( blockIdx.x != 0u )
        return;
    {
        uint64_t k_base = ((uint64_t)row * ((SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT + SPARK_QWEN4_FLASH_MODEL_INDEXER_KV_HEAD_COUNT) * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION)) + ((uint64_t)SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION);
        value = SparkLmBf16ToFloat(qk_bf16,k_base + column);
        sum_squares = SparkLmBlockReduceSum(value * value,reduce_scratch);
        inverse_rms = rsqrtf(sum_squares / (float)SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION + 1e-6f);
        head_shared[column] = value * inverse_rms * SparkLmBf16ToFloat(k_norm_bf16,column);
        __syncthreads();
        slot = slot_mapping[row];
        block = slot / SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
        offset = slot % SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
        key_base = ((uint64_t)block * cache_block_stride) + ((uint64_t)offset * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION);
        SparkLmFloatToBf16(raw_key_cache,key_base + column,head_shared[column]);
        if ( (offset % SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO) == SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO - 1u )
        {
            uint32_t first = offset - (SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO - 1u);
            float sum = 0.0f;
            for (uint32_t tap = 0u; tap < SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO; tap++)
                sum += SparkLmBf16ToFloat(raw_key_cache,((uint64_t)block * cache_block_stride) + ((uint64_t)(first + tap) * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION) + column);
            __shared__ float pooled[SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
            pooled[column] = __bfloat162float(__float2bfloat16(sum / (float)SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO));
            __syncthreads();
            float psum = pooled[column] * pooled[column];
            psum = SparkLmBlockReduceSum(psum,reduce_scratch);
            float prms = rsqrtf(psum / (float)SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION + 1e-6f);
            pooled[column] = pooled[column] * prms * SparkLmBf16ToFloat(k_norm_bf16,column);
            __syncthreads();
            if ( column < SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u )
            {
                float angle = (float)(row_positions[row] - (SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO - 1u)) * SparkQwen4FlashRopeFrequency(column);
                sincosf(angle,&rope_sine[column],&rope_cosine[column]);
            }
            __syncthreads();
            if ( column < SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u )
            {
                float real = pooled[column];
                float imaginary = pooled[column + SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u];
                pooled[column] = real * rope_cosine[column] - imaginary * rope_sine[column];
                pooled[column + SPARK_QWEN4_FLASH_MODEL_ATTN_ROPE_DIMENSION / 2u] = imaginary * rope_cosine[column] + real * rope_sine[column];
            }
            __syncthreads();
            (void)block_indices; (void)lane_stride;
            SparkLmFloatToBf16(pooled_key_cache,((uint64_t)(slot / SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO) * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION) + column,pooled[column]);
        }
    }
}

extern "C" cudaError_t SparkQwen4FlashLaunchIndexerPrepare(
    cudaStream_t stream,
    const void *qk_bf16,
    const SparkQwen4FlashIndexerWeights *weights,
    void *query_bf16,
    void *raw_key_cache,
    void *pooled_key_cache,
    const uint32_t *slot_mapping,
    const uint32_t *block_indices,
    const uint64_t *row_positions,
    uint32_t row_count,
    uint32_t lane_stride,
    uint64_t cache_block_stride)
{
    dim3 grid(SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT,row_count,1u);
    if ( qk_bf16 == 0 || weights == 0 || query_bf16 == 0 || raw_key_cache == 0 || pooled_key_cache == 0 || slot_mapping == 0 || row_positions == 0 || row_count == 0u )
        return(cudaErrorInvalidValue);
    SparkQwen4FlashIndexerPrepareKernel<<<grid,SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION,0,stream>>>(qk_bf16,weights->q_norm_weight_bf16,weights->k_norm_weight_bf16,query_bf16,raw_key_cache,pooled_key_cache,slot_mapping,block_indices,row_positions,row_count,lane_stride,cache_block_stride);
    return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashIndexerSelectKernel(
    const void *query_bf16,
    const void *pooled_key_cache,
    const uint32_t *block_indices,
    const uint32_t *block_counts,
    const uint32_t *row_lane_indices,
    const uint32_t *context_lengths,
    uint8_t *token_mask,
    uint32_t *score_keys_u32,
    uint32_t row_count,
    uint32_t lane_stride,
    uint32_t mask_stride,
    uint32_t score_stride)
{
    __shared__ float q_shared[SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION];
    __shared__ float count_scratch_f32[SPARK_LM_CTA_WARPS];
    __shared__ uint32_t threshold_key;
    __shared__ uint32_t strict_count;
    __shared__ uint32_t equal_taken;
    uint32_t row = blockIdx.x;
    uint32_t lane_index,context,available,required,visible;
    uint32_t complete_blocks,tail_start,block_topk;
    uint32_t *row_keys;
    if ( row >= row_count )
        return;
    lane_index = row_lane_indices[row];
    context = context_lengths[row];
    available = __ldg(block_counts + lane_index);
    required = (context + SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    (void)required;
    visible = context;
    if ( available > lane_stride )
        visible = 0u;
    else if ( (context + SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS > available )
        visible = available * SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    for (uint32_t element = threadIdx.x; element < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION; element += blockDim.x)
        q_shared[element] = SparkLmBf16ToFloat(query_bf16,((uint64_t)row * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION) + element);
    __syncthreads();
    for (uint32_t token = threadIdx.x; token < visible; token += blockDim.x)
        token_mask[(uint64_t)row * mask_stride + token] = 0u;
    __syncthreads();
    if ( visible == 0u )
        return;
    complete_blocks = visible / SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO;
    tail_start = complete_blocks * SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO;
    block_topk = SPARK_QWEN4_FLASH_MODEL_INDEXER_BUDGET / SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO;
    for (uint32_t token = tail_start + threadIdx.x; token < visible; token += blockDim.x)
        token_mask[(uint64_t)row * mask_stride + token] = 1u;
    if ( complete_blocks == 0u || complete_blocks <= block_topk )
    {
        for (uint32_t token = threadIdx.x; token < tail_start; token += blockDim.x)
            token_mask[(uint64_t)row * mask_stride + token] = 1u;
        return;
    }
    row_keys = score_keys_u32 + (uint64_t)row * score_stride;
    for (uint32_t b = threadIdx.x; b < complete_blocks; b += blockDim.x)
    {
        float total = 0.0f;
        for (uint32_t head = 0u; head < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT; head++)
        {
            float dot = 0.0f;
            const float *q_head = q_shared + (head * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION);
            for (uint32_t element = 0u; element < SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION; element++)
                dot = fmaf(q_head[element],SparkLmBf16ToFloat(pooled_key_cache,((uint64_t)b * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION) + element),dot);
            total += fmaxf(dot,0.0f);
        }
        total /= sqrtf((float)SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION);
        uint32_t bits = __float_as_uint(total);
        bits = (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
        row_keys[b] = bits;
    }
    __syncthreads();
    if ( threadIdx.x == 0u )
    {
        threshold_key = 0u;
        strict_count = 0u;
        equal_taken = 0u;
    }
    __syncthreads();
    {
        uint32_t low = 0u,high = 0xffffffffu;
        for (uint32_t round = 0u; round < 32u; round++)
        {
            uint32_t mid,count = 0u;
            if ( threadIdx.x == 0u )
                threshold_key = low + ((high - low) >> 1u);
            __syncthreads();
            mid = threshold_key;
            for (uint32_t b = threadIdx.x; b < complete_blocks; b += blockDim.x)
                if ( row_keys[b] >= mid )
                    count++;
            count = (uint32_t)(SparkLmBlockReduceSum((float)count,count_scratch_f32) + 0.5f);
            if ( threadIdx.x == 0u )
            {
                if ( count >= block_topk )
                    low = mid;
                else
                    high = mid - 1u;
                threshold_key = low;
            }
            __syncthreads();
        }
    }
    const uint32_t threshold = threshold_key;
    if ( threadIdx.x == 0u )
    {
        uint32_t strict = 0u;
        for (uint32_t b = 0u; b < complete_blocks; b++)
            if ( row_keys[b] > threshold )
                strict++;
        strict_count = strict;
        equal_taken = block_topk - strict;
    }
    __syncthreads();
    if ( threadIdx.x == 0u )
    {
        uint32_t equal_seen = 0u;
        for (uint32_t b = 0u; b < complete_blocks; b++)
        {
            if ( row_keys[b] > threshold )
                continue;
            if ( row_keys[b] == threshold )
            {
                if ( equal_seen < equal_taken )
                    equal_seen++;
                else
                    row_keys[b] = 0u;
            }
            else
                row_keys[b] = 0u;
        }
    }
    __syncthreads();
    for (uint32_t b = threadIdx.x; b < complete_blocks; b += blockDim.x)
    {
        if ( row_keys[b] == 0u )
            continue;
        uint32_t base = b * SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO;
        for (uint32_t tap = 0u; tap < SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO; tap++)
            token_mask[(uint64_t)row * mask_stride + base + tap] = 1u;
    }
}


static __global__ void SparkQwen4FlashPleHashGatherKernel(
    const uint32_t *history_u32,
    uint32_t token_count,
    const int64_t *multipliers,
    const int64_t *head_vocab_sizes,
    const int64_t *head_offsets,
    const void *ngram_table_bf16,
    void *embedding_bf16,
    uint32_t row_count,
    uint32_t vocab_base,
    uint32_t vocab_rows)
{
    uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
    uint64_t cells = (uint64_t)row_count * token_count * SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_HEAD_COUNT;
    if ( index >= cells )
        return;
    uint32_t head = (uint32_t)(index % SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_HEAD_COUNT);
    uint64_t row_token = index / SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_HEAD_COUNT;
    uint32_t row = (uint32_t)(row_token / token_count);
    uint32_t token = (uint32_t)(row_token % token_count);
    const uint32_t *window = history_u32 + ((uint64_t)row * (2u + token_count));
    uint32_t position = 2u + token;
    uint32_t ngram = (head / SPARK_QWEN4_FLASH_MODEL_PLE_HEADS_PER_NGRAM) + 2u;
    uint32_t shifted[3];
    for (uint32_t s = 0u; s < ngram; s++)
    {
        int32_t source = (int32_t)position - (int32_t)s;
        uint32_t valid = source >= 0 ? 1u : 0u;
        for (int32_t p = source >= 0 ? source : 0; p < (int32_t)position && valid != 0u; p++)
            if ( window[p] == SPARK_QWEN4_FLASH_MODEL_EOS_TOKEN_ID )
                valid = 0u;
        shifted[s] = valid != 0u ? window[source] : SPARK_QWEN4_FLASH_MODEL_EOS_TOKEN_ID;
    }
    uint64_t mixed = (uint64_t)((int64_t)shifted[0] * multipliers[0]);
    for (uint32_t s = 1u; s < ngram; s++)
        mixed ^= (uint64_t)((int64_t)shifted[s] * multipliers[s]);
    int64_t head_vocab = head_vocab_sizes[head];
    int64_t head_offset = head_offsets[head];
    int64_t m = (int64_t)mixed % head_vocab;
    if ( m < 0 )
        m += head_vocab;
    uint64_t global_id = (uint64_t)(head_offset + m);
    uint64_t out_base = (row_token * SPARK_QWEN4_FLASH_MODEL_PLE_EMBED_DIMENSION) + ((uint64_t)head * SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_HEAD_DIMENSION);
    if ( global_id >= (uint64_t)vocab_base && global_id < (uint64_t)vocab_base + vocab_rows )
    {
        const uint8_t *src = (const uint8_t *)ngram_table_bf16 + ((global_id - vocab_base) * SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_HEAD_DIMENSION * 2u);
        uint8_t *dst = (uint8_t *)embedding_bf16 + (out_base * 2u);
        for (uint32_t d = 0u; d < SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_HEAD_DIMENSION * 2u; d++)
            dst[d] = src[d];
    }
    else
    {
        uint8_t *dst = (uint8_t *)embedding_bf16 + (out_base * 2u);
        for (uint32_t d = 0u; d < SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_HEAD_DIMENSION * 2u; d++)
            dst[d] = 0u;
    }
}

extern "C" cudaError_t SparkQwen4FlashLaunchPleHashGather(
    cudaStream_t stream,
    const uint32_t *history_u32,
    uint32_t token_count,
    const SparkQwen4FlashPleWeights *ple,
    void *embedding_bf16,
    uint32_t row_count,
    uint32_t vocab_base,
    uint32_t vocab_rows)
{
    uint64_t cells = (uint64_t)row_count * token_count * SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_HEAD_COUNT;
    uint32_t blocks = (uint32_t)((cells + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS);
    if ( history_u32 == 0 || ple == 0 || ple->layer_multipliers == 0 || ple->head_vocab_sizes == 0 || ple->head_offsets == 0 || ple->ngram_embedding_bf16 == 0 || embedding_bf16 == 0 || row_count == 0u || token_count == 0u )
        return(cudaErrorInvalidValue);
    SparkQwen4FlashPleHashGatherKernel<<<blocks == 0u ? 1u : blocks,SPARK_LM_CTA_THREADS,0,stream>>>(
        history_u32,token_count,ple->layer_multipliers,ple->head_vocab_sizes,ple->head_offsets,
        ple->ngram_embedding_bf16,embedding_bf16,row_count,vocab_base,vocab_rows);
    return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashPleGateKernel(
    const void *key_normed_bf16,
    const void *query_normed_bf16,
    const void *value_bf16,
    void *gated_value_bf16)
{
    __shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
    uint64_t row = blockIdx.y;
    uint32_t stream = blockIdx.x;
    uint64_t stream_base = (row * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH) + ((uint64_t)stream * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION);
    float dot = 0.0f;
    if ( row >= (uint64_t)gridDim.y )
        return;
    for (uint32_t element = threadIdx.x; element < SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION; element += blockDim.x)
        dot = fmaf(SparkLmBf16ToFloat(key_normed_bf16,stream_base + element),SparkLmBf16ToFloat(query_normed_bf16,stream_base + element),dot);
    dot = SparkLmBlockReduceSum(dot,reduce_scratch) / sqrtf((float)SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION);
    float magnitude = sqrtf(fmaxf(fabsf(dot),1e-6f));
    float gate = copysignf(magnitude,dot);
    gate = SparkLmSigmoid(gate);
    for (uint32_t element = threadIdx.x; element < SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION; element += blockDim.x)
        SparkLmFloatToBf16(gated_value_bf16,stream_base + element,gate * SparkLmBf16ToFloat(value_bf16,(row * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION) + element));
}

extern "C" cudaError_t SparkQwen4FlashLaunchPleGate(
    cudaStream_t stream,
    const void *key_normed_bf16,
    const void *query_normed_bf16,
    const void *value_bf16,
    void *gated_value_bf16,
    uint32_t row_count)
{
    dim3 grid(SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT,row_count,1u);
    SparkQwen4FlashPleGateKernel<<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(key_normed_bf16,query_normed_bf16,value_bf16,gated_value_bf16);
    return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashPleConvUpdateKernel(
    const void *input_bf16,
    const void *conv_weight_bf16,
    void *output_bf16,
    void *tail_bf16,
    const uint32_t *row_lane_indices,
    const uint32_t *state_cold_by_row,
    uint32_t row_count,
    uint64_t tail_lane_stride)
{
    uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
    uint64_t channels = (uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
    if ( index >= channels )
        return;
    for (uint32_t row = 0u; row < row_count; row++)
    {
        uint64_t tail_base = ((uint64_t)row_lane_indices[row] * tail_lane_stride) + (index * SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS);
        uint32_t cold = state_cold_by_row[row] != 0u;
        float window[SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL];
        float weight[SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL];
        float accumulator = 0.0f;
        for (uint32_t tap = 0u; tap < SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL; tap++)
            weight[tap] = SparkLmBf16ToFloat(conv_weight_bf16,(index * SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL) + tap);
        window[0] = SparkLmBf16ToFloat(input_bf16,((uint64_t)row * channels) + index);
        for (uint32_t tap = 1u; tap < SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL; tap++)
            window[tap] = cold != 0u ? 0.0f : SparkLmBf16ToFloat(tail_bf16,tail_base + ((tap * SPARK_QWEN4_FLASH_MODEL_PLE_CONV_DILATION) - 1u));
        for (uint32_t tap = 0u; tap < SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL; tap++)
            accumulator += window[tap] * weight[tap];
        SparkLmFloatToBf16(output_bf16,((uint64_t)row * channels) + index,SparkLmSwish(accumulator));
        for (uint32_t tap = SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS; tap > 1u; tap--)
            SparkLmFloatToBf16(tail_bf16,tail_base + (tap - 1u),SparkLmBf16ToFloat(tail_bf16,tail_base + (tap - 2u)));
        SparkLmFloatToBf16(tail_bf16,tail_base + 0u,window[0]);
    }
}

extern "C" cudaError_t SparkQwen4FlashLaunchPleConvUpdate(
    cudaStream_t stream,
    const void *input_bf16,
    const SparkQwen4FlashPleWeights *ple,
    void *output_bf16,
    void *tail_bf16,
    const uint32_t *row_lane_indices,
    const uint32_t *state_cold_by_row,
    uint32_t row_count,
    uint64_t tail_lane_stride)
{
    uint64_t channels = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
    uint32_t blocks = (uint32_t)((channels + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS);
    SparkQwen4FlashPleConvUpdateKernel<<<blocks,SPARK_LM_CTA_THREADS,0,stream>>>(input_bf16,ple->conv_weight_bf16,output_bf16,tail_bf16,row_lane_indices,state_cold_by_row,row_count,tail_lane_stride);
    return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashPleConvChunkKernel(
    const void *input_bf16,
    const void *conv_weight_bf16,
    void *output_bf16,
    void *tail_bf16,
    uint32_t token_count,
    uint64_t tail_lane_stride)
{
    uint64_t channel = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
    uint64_t channels = (uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
    if ( channel >= channels )
        return;
    uint64_t tail_base = channel * SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS;
    float window[SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS + 1u];
    float weight[SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL];
    for (uint32_t tap = 0u; tap < SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS; tap++)
        window[tap] = SparkLmBf16ToFloat(tail_bf16,tail_base + (SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS - 1u - tap));
    window[SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS] = 0.0f;
    for (uint32_t tap = 0u; tap < SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL; tap++)
        weight[tap] = SparkLmBf16ToFloat(conv_weight_bf16,(channel * SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL) + tap);
    for (uint32_t token = 0u; token < token_count; token++)
    {
        float accumulator = 0.0f;
        uint64_t at = ((uint64_t)token * channels) + channel;
        for (uint32_t tap = 0u; tap < SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS; tap++)
            window[tap] = window[tap + 1u];
        window[SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS] = SparkLmBf16ToFloat(input_bf16,at);
        for (uint32_t tap = 0u; tap < SPARK_QWEN4_FLASH_MODEL_PLE_CONV_KERNEL; tap++)
            accumulator += window[SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS - (tap * SPARK_QWEN4_FLASH_MODEL_PLE_CONV_DILATION)] * weight[tap];
        SparkLmFloatToBf16(output_bf16,at,SparkLmSwish(accumulator));
    }
    for (uint32_t tap = 0u; tap < SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS; tap++)
        SparkLmFloatToBf16(tail_bf16,tail_base + tap,window[SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS - 1u - tap]);
    (void)tail_lane_stride;
}

extern "C" cudaError_t SparkQwen4FlashLaunchPleConvChunk(
    cudaStream_t stream,
    const void *input_bf16,
    const SparkQwen4FlashPleWeights *ple,
    void *output_bf16,
    void *tail_bf16,
    uint32_t token_count)
{
    uint64_t channels = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
    uint32_t blocks = (uint32_t)((channels + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS);
    if ( token_count == 0u )
        return(cudaErrorInvalidValue);
    SparkQwen4FlashPleConvChunkKernel<<<blocks,SPARK_LM_CTA_THREADS,0,stream>>>(input_bf16,ple->conv_weight_bf16,output_bf16,tail_bf16,token_count,0);
    return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen4FlashLaunchIndexerSelect(
    cudaStream_t stream,
    const void *query_bf16,
    const void *pooled_key_cache,
    const SparkQwen4FlashKvBlockTableView *table,
    const uint32_t *row_lane_indices,
    const uint32_t *context_lengths,
    uint8_t *token_mask,
    uint32_t *score_keys_u32,
    uint32_t row_count,
    uint32_t mask_stride,
    uint32_t score_stride)
{
    if ( query_bf16 == 0 || pooled_key_cache == 0 || table == 0 || row_lane_indices == 0 || context_lengths == 0 || token_mask == 0 || score_keys_u32 == 0 || row_count == 0u || mask_stride == 0u || score_stride == 0u )
        return(cudaErrorInvalidValue);
    SparkQwen4FlashIndexerSelectKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(
        query_bf16,pooled_key_cache,table->physical_block_indices,table->lane_physical_block_counts,
        row_lane_indices,context_lengths,token_mask,score_keys_u32,row_count,table->lane_stride,mask_stride,score_stride);
    return(cudaGetLastError());
}
#define SPARK_QWEN38_ROUTER_SORT_CAPACITY 512u

static __device__ __forceinline__ float SparkQwen4FlashWarpReduceMax(float value)
{
	#pragma unroll
	for (uint32_t offset = SPARK_LM_WARP_LANES >> 1u; offset != 0u; offset >>= 1u)
		value = fmaxf(value,__shfl_down_sync(0xffffffffu,value,offset));
	return(value);
}

static __global__ void SparkQwen4FlashGateSelectKernel(
    const float *scores_f32,
    const float *bias_f32,
    uint32_t row_count,
    uint32_t expert_count,
    uint32_t topk,
    float route_scale,
    uint32_t *indices_u32,
    float *weights_f32)
{
    __shared__ uint64_t ordered_keys[SPARK_QWEN38_ROUTER_SORT_CAPACITY];
    const float *row_scores;
    uint64_t selected_key;
    uint32_t row;
    uint32_t expert;
    uint32_t rank;
    uint32_t selected_expert;
    float selected_score;
    float selected_total;

    static_assert(
        SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT <=
            SPARK_QWEN38_ROUTER_SORT_CAPACITY,
        "qwen38 expert count exceeds router sort capacity");
    static_assert(
        SPARK_LM_MOE_MAX_TOPK <= SPARK_LM_WARP_LANES,
        "qwen38 router normalization requires one warp");
    row = blockIdx.x;
    if ( row >= row_count || expert_count == 0u ||
        expert_count > SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT ||
        topk == 0u || topk > SPARK_LM_MOE_MAX_TOPK || topk > expert_count )
        return;
    row_scores = scores_f32 + ((uint64_t)row * expert_count);
    rank = threadIdx.x;
    for (expert = threadIdx.x; expert < SPARK_QWEN38_ROUTER_SORT_CAPACITY;
        expert += blockDim.x)
    {
        float choice_score;
        choice_score = expert < expert_count
            ? row_scores[expert] + (bias_f32 != 0 ? bias_f32[expert] : 0.0f)
            : NAN;
        if ( expert < expert_count && isnan(choice_score) )
            choice_score = -INFINITY;
        ordered_keys[expert] = expert < expert_count
            ? SparkLmOrderedTopKKey(choice_score, expert)
            : 0u;
    }
    __syncthreads();
    SparkLmBitonicSortKeysAscending<SPARK_QWEN38_ROUTER_SORT_CAPACITY>(ordered_keys);
    selected_key = rank < topk
        ? ordered_keys[SPARK_QWEN38_ROUTER_SORT_CAPACITY - 1u - rank]
        : 0u;
    selected_expert = selected_key != 0u
        ? 0xffffffffu - (uint32_t)selected_key
        : UINT32_MAX;
    selected_score = rank < topk && selected_expert < expert_count &&
        !isnan(row_scores[selected_expert])
        ? row_scores[selected_expert]
        : -INFINITY;
    if ( threadIdx.x < SPARK_LM_WARP_LANES )
    {
        float max_score = SparkQwen4FlashWarpReduceMax(selected_score);
        float exp_score;
        max_score = __shfl_sync(0xffffffffu, max_score, 0u);
        exp_score = __expf(selected_score - max_score);
        selected_total = __shfl_sync(
            0xffffffffu,
            SparkLmWarpReduceSum(exp_score),
            0u);
        if ( rank < topk )
        {
            indices_u32[((uint64_t)row * topk) + rank] = selected_expert;
            weights_f32[((uint64_t)row * topk) + rank] =
                selected_total > 0.0f
                ? route_scale * exp_score / selected_total
                : 0.0f;
        }
    }
}

static __global__ void SparkQwen4FlashSwiGluKernel(const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t pair = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x,pair_count = ((uint64_t)row_count * dimension) >> 1u;
	float2 gate_pair,up_pair;
	if ( pair >= pair_count )
		return;
	gate_pair = SparkLmLoadBf16Pair(gate_bf16,pair);
	up_pair = SparkLmLoadBf16Pair(up_bf16,pair);
	SparkLmStoreBf16Pair(up_bf16,pair,SparkLmSwish(gate_pair.x) * up_pair.x,SparkLmSwish(gate_pair.y) * up_pair.y);
	if ( pair == 0u && (((uint64_t)row_count * dimension) & 1u) != 0u )
		SparkLmFloatToBf16(up_bf16,((uint64_t)row_count * dimension) - 1u,SparkLmSwish(SparkLmBf16ToFloat(gate_bf16,((uint64_t)row_count * dimension) - 1u)) * SparkLmBf16ToFloat(up_bf16,((uint64_t)row_count * dimension) - 1u));
}

static __global__ void SparkQwen4FlashSharedGateKernel(void *accum_bf16, const void *gate_weight_bf16, const void *gate_input_bf16, uint32_t row_count, uint32_t dimension)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x;
	uint64_t row_base = (uint64_t)row * dimension;
	uint64_t index;
	float logit = 0.0f,gate;
	if ( row >= row_count )
		return;
	for (index = threadIdx.x; index < dimension; index += blockDim.x)
		logit = fmaf(SparkLmBf16ToFloat(gate_input_bf16,row_base + index),SparkLmBf16ToFloat(gate_weight_bf16,index),logit);
	logit = SparkLmBlockReduceSum(logit,reduce_scratch);
	gate = SparkLmSigmoid(logit);
	for (index = threadIdx.x; index < dimension; index += blockDim.x)
		SparkLmFloatToBf16(accum_bf16,row_base + index,gate * SparkLmBf16ToFloat(accum_bf16,row_base + index));
}

extern "C" cudaError_t SparkQwen4FlashLaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32)
{
	SparkQwen4FlashGateSelectKernel<<<row_count, SPARK_LM_CTA_THREADS, 0, stream>>>(scores_f32,bias_f32,row_count,expert_count,topk,route_scale,indices_u32,weights_f32);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen4FlashLaunchMoeRoute(cudaStream_t stream, const uint32_t *route_expert, uint32_t rows, uint32_t expert_width, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2)
{
	int32_t launch_status = LmRouteBuild<SPARK_LM_CTA_THREADS,SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT>(route_expert,rows,rows * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN,SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN,group_row_offset,route_packed_row,route_source_token,expert_width,SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION,SPARK_LM_TILE_N,SPARK_LM_TILE_N,group_tile_prefix_w1,group_tile_prefix_w2,stream);
	return(launch_status == LM_LAUNCH_OK ? cudaSuccess : cudaErrorLaunchFailure);
}

extern "C" cudaError_t SparkQwen4FlashLaunchFusedExpertW13Act(cudaStream_t stream, const SparkQwen4FlashLinearView *w1, const SparkQwen4FlashLinearView *w3, const void *input_bf16, const uint32_t *route_source_token, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *activated_bf16, uint32_t rows, uint32_t expert_width, float limit, uint32_t multiprocessor_count)
{
	cudaError_t status;
	uint64_t required_rows = (uint64_t)SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT * expert_width;
	uint32_t lm_format;
	if ( w1 == 0 || w3 == 0 || input_bf16 == 0 || route_source_token == 0 || group_row_offset == 0 || group_tile_prefix == 0 || activated_bf16 == 0 || (w1->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && w1->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_NVFP4_PACKED) || w1->weight_format != w3->weight_format || w1->weight_payload == 0 || w3->weight_payload == 0 || w1->weight_scale_e8m0 == 0 || w3->weight_scale_e8m0 == 0 || w1->output_dimension != required_rows || w3->output_dimension != required_rows || w1->input_dimension != SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION || w3->input_dimension != SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION )
		return(cudaErrorInvalidValue);
	lm_format = w1->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_NVFP4_PACKED ? SPARK_LM_WEIGHT_FORMAT_NVFP4_E2M1 : SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1;
	status = SparkLmHostLaunchSm121FusedExpertW13(stream,w1->weight_payload,w1->weight_scale_e8m0,w3->weight_payload,w3->weight_scale_e8m0,input_bf16,route_source_token,group_row_offset,group_tile_prefix,activated_bf16,rows,SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN,SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT,SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION,expert_width,limit,lm_format,multiprocessor_count);
	return(status);
}

extern "C" cudaError_t SparkQwen4FlashLaunchExpertDown(cudaStream_t stream, const SparkQwen4FlashLinearView *stacked, const void *input_bf16, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t expert_width, uint32_t hidden_dimension, uint32_t multiprocessor_count)
{
	uint64_t required_rows = (uint64_t)SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT * hidden_dimension;
	uint32_t lm_format;
	if ( stacked == 0 || input_bf16 == 0 || group_row_offset == 0 || group_tile_prefix == 0 || output_bf16 == 0 || (stacked->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && stacked->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_NVFP4_PACKED) || stacked->weight_payload == 0 || stacked->weight_scale_e8m0 == 0 || stacked->output_dimension != required_rows || stacked->input_dimension != expert_width )
		return(cudaErrorInvalidValue);
	lm_format = stacked->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_NVFP4_PACKED ? SPARK_LM_WEIGHT_FORMAT_NVFP4_E2M1 : SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1;
	return(SparkLmHostLaunchSm121ExpertW2(stream,stacked->weight_payload,stacked->weight_scale_e8m0,input_bf16,group_row_offset,group_tile_prefix,output_bf16,rows,SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN,SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT,expert_width,hidden_dimension,lm_format,multiprocessor_count));
}

extern "C" cudaError_t SparkQwen4FlashLaunchMoePairReduce(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchMoePairReduce(stream,slot_out_bf16,inverse_map,pair_weights_f32,accum_bf16,row_count,SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN,hidden_dimension));
}

extern "C" cudaError_t SparkQwen4FlashLaunchMoePairReduceOverwrite(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *output_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchMoePairReduceOverwrite(stream,slot_out_bf16,inverse_map,pair_weights_f32,output_bf16,row_count,SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN,hidden_dimension));
}

extern "C" cudaError_t SparkQwen4FlashLaunchSwiGlu(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension)
{
	uint32_t threads = 256u;
	uint64_t pairs = ((uint64_t)row_count * dimension) >> 1u;
	uint32_t blocks = (uint32_t)((pairs + threads - 1u) / threads);
	SparkQwen4FlashSwiGluKernel<<<(blocks == 0u ? 1u : blocks), threads, 0, stream>>>(gate_bf16,up_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen4FlashLaunchSharedGate(cudaStream_t stream, void *accum_bf16, const void *gate_weight_bf16, const void *gate_input_bf16, uint32_t row_count, uint32_t dimension)
{
	SparkQwen4FlashSharedGateKernel<<<row_count, SPARK_LM_CTA_THREADS, 0, stream>>>(accum_bf16,gate_weight_bf16,gate_input_bf16,row_count,dimension);
	return(cudaGetLastError());
}

static __global__ void SparkQwen4FlashGateScoresKernel(const void *weight_bf16, const void *input_bf16, float *scores_f32, uint32_t row_count, uint32_t input_dimension, uint32_t expert_count)
{
	extern __shared__ float gate_shared[];
	uint32_t row = blockIdx.x,warp_count = blockDim.x / SPARK_LM_WARP_LANES;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t expert = blockIdx.y * warp_count + warp,element;
	float accumulator;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < input_dimension; element += blockDim.x)
		gate_shared[element] = SparkLmBf16ToFloat(input_bf16,((uint64_t)row * input_dimension) + element);
	__syncthreads();
	if ( expert >= expert_count )
		return;
	accumulator = SparkLmDotRowBf16(gate_shared,weight_bf16,expert,input_dimension,lane);
	accumulator = SparkLmWarpReduceSum(accumulator);
	if ( lane == 0u )
		scores_f32[((uint64_t)row * expert_count) + expert] = accumulator;
}

extern "C" cudaError_t SparkQwen4FlashLaunchGateScores(cudaStream_t stream, const SparkQwen4FlashLinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count)
{
	uint32_t warp_count = SPARK_LM_CTA_THREADS / SPARK_LM_WARP_LANES;
	uint32_t expert_blocks = (gate->output_dimension + warp_count - 1u) / warp_count;
	if ( gate == 0 || input_bf16 == 0 || scores_f32 == 0 || gate->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 || gate->input_dimension == 0u )
		return(cudaErrorInvalidValue);
	SparkQwen4FlashGateScoresKernel<<<dim3(row_count,expert_blocks),SPARK_LM_CTA_THREADS,gate->input_dimension * sizeof(float),stream>>>(gate->weight_payload,input_bf16,scores_f32,row_count,gate->input_dimension,gate->output_dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen4FlashConfigureCudaKernels(void)
{
    cudaError_t status;

    status = cudaFuncSetAttribute(
        SparkQwen4FlashGdnStepKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)SPARK_QWEN4_FLASH_CUDA_GDN_DECODE_SHARED_BYTES);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "qwen38 configure gdn_step failed %d shared=%d\n", (int)status, (int)SPARK_QWEN4_FLASH_CUDA_GDN_DECODE_SHARED_BYTES);
        return status;
    }
    status = cudaFuncSetAttribute(
        SparkQwen4FlashChunkQkDecayKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)SPARK_QWEN4_FLASH_CUDA_GDN_QK_SHARED_BYTES);
    if (status != cudaSuccess)
    {
        return status;
    }
    status = cudaFuncSetAttribute(
        SparkQwen4FlashChunkStepKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)SPARK_QWEN4_FLASH_CUDA_GDN_CHUNK_SHARED_BYTES);
    if (status != cudaSuccess)
    {
        return status;
    }
    uint32_t widest = SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_DIMENSION;
    if ( SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH > widest )
        widest = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
    return cudaFuncSetAttribute(
        (const void *)SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,SPARK_LM_CTA_WARPS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)(widest * sizeof(float)));
}

static __global__ void SparkQwen4FlashGroupedScalarE8m0Kernel(
	const void *payload_base,
	const uint8_t *scale_base,
	uint64_t payload_group_stride_bytes,
	uint64_t scale_group_stride_bytes,
	const void *input_bf16,
	const uint32_t *source_row_map,
	uint32_t source_row_count,
	const uint32_t *group_row_offset,
	const uint32_t *group_tile_prefix,
	void *output_bf16,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	const void *frame_error_void)
{
	extern __shared__ float shared_input[];
	const uint32_t subtile_count = (SPARK_LM_TILE_N + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS;
	uint32_t task,group,in_group,neuron_tiles,row_tile,neuron_tile,row_base,row_limit,local_row,row,source_row,element,subtile,neuron,warp,lane;
	const void *group_payload;
	const uint8_t *group_scale;
	float accumulator;
	LmFrameError *frame_error;
	frame_error = (LmFrameError *)frame_error_void;
	for (task = blockIdx.x; task < group_tile_prefix[group_count]; task += gridDim.x)
	{
		group = SparkLmGroupedScalarGroupOfTile(group_tile_prefix,group_count,task);
		in_group = task - group_tile_prefix[group];
		neuron_tiles = (output_dimension + SPARK_LM_TILE_N - 1u) / SPARK_LM_TILE_N;
		row_tile = in_group / neuron_tiles;
		neuron_tile = in_group % neuron_tiles;
		row_base = group_row_offset[group] + row_tile * SPARK_LM_TILE;
		row_limit = group_row_offset[group + 1u];
		group_payload = (const uint8_t *)payload_base + ((uint64_t)group * payload_group_stride_bytes);
		group_scale = scale_base + ((uint64_t)group * scale_group_stride_bytes);
		for (local_row = 0u; local_row < SPARK_LM_TILE && row_base + local_row < row_limit; local_row++)
		{
			row = row_base + local_row;
			source_row = source_row_map != 0 ? source_row_map[row] : row;
			if ( source_row >= source_row_count )
			{
				LmFrameErrorReport(frame_error,
					(uint32_t)LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE,
					0u,row,source_row,row_base,source_row_count);
				source_row = 0u;
			}
			for (element = threadIdx.x; element < input_dimension; element += blockDim.x)
				shared_input[element] = SparkLmBf16ToFloat(input_bf16,((uint64_t)source_row * input_dimension) + element);
			__syncthreads();
			warp = threadIdx.x / SPARK_LM_WARP_LANES;
			lane = threadIdx.x % SPARK_LM_WARP_LANES;
			for (subtile = 0u; subtile < subtile_count; subtile++)
			{
				neuron = (neuron_tile * SPARK_LM_TILE_N) + (subtile * SPARK_LM_CTA_WARPS) + warp;
				if ( neuron < output_dimension )
				{
					accumulator = SparkLmDotRowFp8E8m0(shared_input,group_payload,group_scale,neuron,input_dimension,lane);
					accumulator = SparkLmWarpReduceSum(accumulator);
					if ( lane == 0u )
						SparkLmFloatToBf16(output_bf16,((uint64_t)row * output_dimension) + neuron,accumulator);
				}
			}
			__syncthreads();
		}
	}
}

static cudaError_t SparkQwen4FlashLaunchGroupedScalarE8m0(
	cudaStream_t stream,
	const void *payload_base,
	const uint8_t *scale_base,
	uint64_t payload_group_stride_bytes,
	uint64_t scale_group_stride_bytes,
	const void *input_bf16,
	const uint32_t *source_row_map,
	uint32_t source_row_count,
	const uint32_t *group_row_offset,
	const uint32_t *group_tile_prefix,
	void *output_bf16,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	uint32_t multiprocessor_count,
	const void *frame_error)
{
	uint32_t block_count;
	size_t shared_bytes;
	if ( payload_base == 0 || scale_base == 0 || input_bf16 == 0 || group_row_offset == 0 ||
		group_tile_prefix == 0 || output_bf16 == 0 || source_row_count == 0u || group_count == 0u ||
		input_dimension == 0u || output_dimension == 0u || multiprocessor_count == 0u ||
		payload_group_stride_bytes == 0u || scale_group_stride_bytes == 0u ||
		(source_row_map == 0 && source_row_count == 0u) || (input_dimension % 128u) != 0u ||
		frame_error == 0 )
		return(cudaErrorInvalidValue);
	block_count = multiprocessor_count * 2u;
	if ( block_count == 0u )
		block_count = 1u;
	shared_bytes = (size_t)input_dimension * sizeof(float);
	SparkQwen4FlashGroupedScalarE8m0Kernel<<<block_count,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(
		payload_base,scale_base,payload_group_stride_bytes,scale_group_stride_bytes,
		input_bf16,source_row_map,source_row_count,group_row_offset,group_tile_prefix,
		output_bf16,group_count,input_dimension,output_dimension,frame_error);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen4FlashLaunchGroupedExpertLinear(
	cudaStream_t stream,
	const SparkQwen4FlashLinearView *view,
	const void *input_bf16,
	const uint32_t *source_row_map,
	const uint32_t *group_row_offset,
	const uint32_t *group_tile_prefix,
	void *output_bf16,
	uint32_t source_row_count,
	uint32_t multiprocessor_count,
	uint32_t tp_degree,
	uint32_t tp_rank,
	uint32_t route_group_base,
	const void *frame_error)
{
	uint64_t rows_per_expert;
	uint64_t payload_stride,scale_stride;
	uint32_t experts_per_rank = SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT / tp_degree;
	uint32_t lm_format;
	const uint8_t *payload;
	const uint8_t *scale;
	const uint32_t *offsets;
	const uint32_t *prefix;
	if ( view == 0 || input_bf16 == 0 ||
		group_row_offset == 0 || group_tile_prefix == 0 || output_bf16 == 0 ||
		(view->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 &&
			view->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 &&
			view->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16) ||
		view->weight_payload == 0 ||
		(view->weight_scale_e8m0 == 0 && view->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16) ||
		(source_row_map == 0 && source_row_count == 0u) ||
		tp_degree == 0u || tp_rank >= tp_degree ||
		(SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT % tp_degree) != 0u )
		return(cudaErrorInvalidValue);
	rows_per_expert = (uint64_t)view->output_dimension / experts_per_rank;
	if ( rows_per_expert * experts_per_rank != view->output_dimension )
		return(cudaErrorInvalidValue);
	payload_stride = rows_per_expert * view->input_dimension;
	if ( view->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
	{
		if ( (view->input_dimension % 128u) != 0u )
			return(cudaErrorInvalidValue);
		scale_stride = rows_per_expert * ((uint64_t)view->input_dimension / 128u);
		lm_format = SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128;
	}
	else if ( view->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
	{
		payload_stride *= 2u;
		scale_stride = 0u;
		lm_format = SPARK_LM_WEIGHT_FORMAT_BF16;
	}
	else
	{
		scale_stride = (rows_per_expert / 128u) * ((uint64_t)view->input_dimension / 128u) * 4u;
		lm_format = SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128;
	}
	payload = (const uint8_t *)view->weight_payload + ((uint64_t)tp_rank * experts_per_rank * payload_stride);
	scale = (const uint8_t *)view->weight_scale_e8m0 + ((uint64_t)tp_rank * experts_per_rank * scale_stride);
	offsets = group_row_offset + route_group_base;
	prefix = group_tile_prefix + route_group_base;
	if ( lm_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
	{
		return(SparkQwen4FlashLaunchGroupedScalarE8m0(stream,
			payload,scale,payload_stride,scale_stride,
			input_bf16,source_row_map,source_row_count,offsets,prefix,
			output_bf16,experts_per_rank,
			view->input_dimension,(uint32_t)rows_per_expert,multiprocessor_count,
			frame_error));
	}
	return(SparkLmHostLaunchGroupedScalarLinear<32u>(stream,
		lm_format,
		payload,scale,
		payload_stride,scale_stride,
		input_bf16,source_row_map,source_row_count,offsets,prefix,
		output_bf16,experts_per_rank,
		view->input_dimension,rows_per_expert,multiprocessor_count));
}

extern "C" cudaError_t SparkQwen4FlashLaunchGroupedExpertTileLinear(
	cudaStream_t stream,
	const SparkQwen4FlashLinearView *view,
	const void *input_bf16,
	const uint32_t *source_row_map,
	const uint32_t *group_row_offset,
	void *output_bf16,
	uint32_t source_row_count,
	uint32_t tp_degree,
	uint32_t tp_rank,
	uint32_t route_group_base)
{
	uint64_t rows_per_expert;
	uint64_t payload_stride;
	uint32_t m_blocks,n_tiles,experts_per_rank;
	const uint8_t *payload;
	const uint8_t *scale;
	const uint32_t *offsets;
	if ( view == 0 || input_bf16 == 0 || group_row_offset == 0 || output_bf16 == 0 ||
		(view->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 &&
			view->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 &&
			view->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16) ||
		view->input_dimension % 64u != 0u ||
		view->weight_payload == 0 ||
		(view->weight_scale_e8m0 == 0 && view->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16) ||
		source_row_count == 0u ||
		tp_degree == 0u || tp_rank >= tp_degree ||
		(SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT % tp_degree) != 0u )
		return(cudaErrorInvalidValue);
	experts_per_rank = SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT / tp_degree;
	rows_per_expert = (uint64_t)view->output_dimension / experts_per_rank;
	if ( rows_per_expert * experts_per_rank != view->output_dimension || rows_per_expert % SPARK_LM_TILE_N != 0u )
		return(cudaErrorInvalidValue);
	payload_stride = rows_per_expert * view->input_dimension;
	if ( view->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
		payload_stride *= 2u;
	m_blocks = (source_row_count + SPARK_LM_TILE - 1u) / SPARK_LM_TILE;
	n_tiles = (uint32_t)(rows_per_expert / SPARK_LM_TILE_N);
	(void)m_blocks;
	(void)n_tiles;
	payload = (const uint8_t *)view->weight_payload + ((uint64_t)tp_rank * experts_per_rank * payload_stride);
	offsets = group_row_offset + route_group_base;
	if ( view->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
	{
		uint32_t budget = experts_per_rank < 64u ? experts_per_rank : 64u;
		dim3 grid(1u,(uint32_t)(rows_per_expert / SPARK_LM_TILE_N),budget);
		if ( (view->input_dimension % 128u) != 0u )
			return(cudaErrorInvalidValue);
		scale = (const uint8_t *)view->weight_scale_e8m0 + ((uint64_t)tp_rank * experts_per_rank * (rows_per_expert * ((uint64_t)view->input_dimension / 128u)));
		SparkLmExpertTileAllMloopKernel<128u><<<grid,SPARK_LM_CTA_THREADS,0u,stream>>>(
			SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128,
			payload,scale,
			payload_stride,rows_per_expert * ((uint64_t)view->input_dimension / 128u),
			input_bf16,source_row_map,offsets,output_bf16,
			view->input_dimension,(uint32_t)rows_per_expert,
			experts_per_rank);
		return(cudaGetLastError());
	}
	if ( view->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
	{
		return(SparkLmHostLaunchGroupedExpertTileMloop(
			stream,
			SPARK_LM_WEIGHT_FORMAT_BF16,
			payload,(const uint8_t *)0,
			payload_stride,0u,
			input_bf16,source_row_map,offsets,output_bf16,
			view->input_dimension,(uint32_t)rows_per_expert,
			experts_per_rank));
	}
	scale = (const uint8_t *)view->weight_scale_e8m0 + ((uint64_t)tp_rank * experts_per_rank * ((rows_per_expert / 128u) * ((uint64_t)view->input_dimension / 128u) * 4u));
	return(SparkLmHostLaunchGroupedExpertTileMloop(
		stream,
		SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128,
		payload,scale,
		payload_stride,(rows_per_expert / 128u) * ((uint64_t)view->input_dimension / 128u) * 4u,
		input_bf16,source_row_map,offsets,output_bf16,
		view->input_dimension,(uint32_t)rows_per_expert,
		experts_per_rank));
}
