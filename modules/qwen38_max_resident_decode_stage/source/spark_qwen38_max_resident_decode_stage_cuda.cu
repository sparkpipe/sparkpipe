#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_lm_kernels.cuh"
#include "inference/kernels/route.cuh"
#include "runtime/launch.h"

/*
 * Qwen 3.6 27B device code. Two production hot paths share these kernels: a
 * decode microbatch of up to 512 rows, one next token per distinct lane; and
 * a prefill frame of up to 512 consecutive positions of ONE lane, whose
 * projections, norms and attention batch over all positions (each kernel is
 * already per-row correct) while the GDN core runs the chunked formulation
 * below, proven against the CPU chunk oracle and BITWISE carry-equal to the
 * recurrence.
 *
 * Shared machinery (RmsNorm, dual-format Linear, fused argmax head, reduces)
 * comes from spark_lm_kernels.cuh; this file holds only what is Qwen:
 * the depthwise conv state update, the recurrent gated delta step operating
 * in-place on the resident state pool, the fp32 gated head norm, and paged
 * GQA attention with per-head query|gate fusion, q/k head norms and partial
 * RoPE. All forms are the PINNED modeling_qwen3_5 forms, oracle-matched.
 *
 * Grid conventions: one block per (row, head) for head-shaped work, one
 * block per row for row-shaped work; row order everywhere follows the frame
 * decode batch view. Launchers are extern "C" and stream-ordered; nothing
 * here synchronizes.
 */

#define SPARK_QWEN38_CUDA_DK SPARK_QWEN38_MAX_MODEL_GDN_HEAD_KEY_DIMENSION
#define SPARK_QWEN38_CUDA_DV SPARK_QWEN38_MAX_MODEL_GDN_HEAD_VALUE_DIMENSION
#define SPARK_QWEN38_CUDA_GVA_GROUP (SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT / SPARK_QWEN38_MAX_MODEL_GDN_KEY_HEAD_COUNT)
#define SPARK_QWEN38_CUDA_ATTN_GROUP (SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT / SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT)
#define SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA 2u
#define SPARK_QWEN38_CUDA_ATTN_VALUE_PAIRS_PER_LANE \
    (SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION / (2u * SPARK_LM_WARP_LANES))
#define SPARK_QWEN38_CUDA_GDN_STATE_ELEMENTS \
    (SPARK_QWEN38_CUDA_DK * SPARK_QWEN38_CUDA_DV)
#define SPARK_QWEN38_CUDA_GDN_DECODE_SHARED_BYTES \
    (SPARK_QWEN38_CUDA_GDN_STATE_ELEMENTS * sizeof(float))

static __device__ __forceinline__ float SparkQwen38RopeFrequency(uint32_t pair)
{
	return(exp2f(-((float)(2u * pair) / (float)SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION) * log2f((float)SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_THETA)));
}

// Depthwise causal conv update for one decode token per row: one thread per
// (row, channel), window = carried tail (3) plus the fresh projection, dot
// with the 4-tap weight, silu, then rotate the tail in place. Cold rows read
// a zero tail. Matches causal_conv1d_update with bias absent.
static __global__ void SparkQwen38ConvUpdateKernel(const void *qkv_bf16, const void *conv_weight_bf16, void *conv_out_bf16, void *conv_tail_bf16, const uint32_t *row_lane_indices, const uint32_t *state_cold_by_row, uint32_t row_count, uint32_t gdn_layer_ordinal, uint64_t tail_lane_stride, uint64_t tail_layer_stride)
{
	uint32_t row = blockIdx.y,channel = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t tail_base;
	float window[4],accumulator;
	uint32_t tap;
	if ( row >= row_count || channel >= SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS )
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
	window[3] = SparkLmBf16ToFloat(qkv_bf16,((uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS) + channel);
	accumulator = 0.0f;
	for (tap = 0; tap < SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL; tap++)
		accumulator += (window[tap] * SparkLmBf16ToFloat(conv_weight_bf16,((uint64_t)channel * SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL) + tap));
	SparkLmFloatToBf16(conv_out_bf16,((uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS) + channel,SparkLmSwish(accumulator));
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 0u,window[1]);
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 1u,window[2]);
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 2u,window[3]);
}

// Per-head log decay and beta from the two 48-row projections plus the fp32
// decay parameters: g = -exp(a_log) * softplus(a + dt_bias), beta =
// sigmoid(b). One thread per (row, value head).
static __global__ void SparkQwen38DecayBetaKernel(const void *decay_pre_bf16, const void *beta_pre_bf16, const float *a_log_f32, const float *dt_bias_f32, float *log_decay_f32, float *beta_f32, uint32_t row_count)
{
	uint32_t row = blockIdx.x,head = threadIdx.x;
	uint64_t index;
	if ( row >= row_count || head >= SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT )
		return;
	index = ((uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT) + head;
	log_decay_f32[index] = -expf(a_log_f32[head]) * SparkLmSoftplus(SparkLmBf16ToFloat(decay_pre_bf16,index) + dt_bias_f32[head]);
	beta_f32[index] = SparkLmSigmoid(SparkLmBf16ToFloat(beta_pre_bf16,index));
}

/*
 * Recurrent gated delta step, one block per (row, value head), 128 threads,
 * thread j owning state column j so every state access is coalesced. The
 * conv output is channel order query(2048) | key(2048) | value(6144); the
 * key head for value head h is h / 3 (GVA). q and k are L2-normalized per
 * head with eps 1e-6 and q is scaled 1/sqrt(dk), matching the oracle
 * recurrence bitwise in structure: decay, predict, delta, rank-one update,
 * read-out. State is fp32 in the resident pool and updated in place; cold
 * rows start from zero without a separate memset pass.
 */
static __global__ void SparkQwen38GdnStepKernel(const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *state_f32, void *core_out_bf16, const uint32_t *row_lane_indices, const uint32_t *state_cold_by_row, uint32_t row_count, uint32_t gdn_layer_ordinal, uint64_t state_lane_stride, uint64_t state_layer_stride)
{
    extern __shared__ float state_shared[];
    __shared__ float qn[SPARK_QWEN38_CUDA_DK];
    __shared__ float kn[SPARK_QWEN38_CUDA_DK];
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
    key_head = head / SPARK_QWEN38_CUDA_GVA_GROUP;
    if (row >= row_count)
    {
        return;
    }

    conv_row = (uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS;
    value = SparkLmBf16ToFloat(
        conv_out_bf16,
        conv_row + ((uint64_t)key_head * SPARK_QWEN38_CUDA_DK) + column);
    q_norm = SparkLmBlockReduceSum(value * value, reduce_scratch);
    qn[column] = value * rsqrtf(q_norm + 1.0e-6f) *
        rsqrtf((float)SPARK_QWEN38_CUDA_DK);

    value = SparkLmBf16ToFloat(
        conv_out_bf16,
        conv_row + SPARK_QWEN38_MAX_MODEL_GDN_QK_DIMENSION +
            ((uint64_t)key_head * SPARK_QWEN38_CUDA_DK) + column);
    k_norm = SparkLmBlockReduceSum(value * value, reduce_scratch);
    kn[column] = value * rsqrtf(k_norm + 1.0e-6f);
    __syncthreads();

    state_base =
        ((uint64_t)row_lane_indices[row] * state_lane_stride) +
        ((uint64_t)gdn_layer_ordinal * state_layer_stride) +
        ((uint64_t)head * SPARK_QWEN38_CUDA_GDN_STATE_ELEMENTS);
    decay = state_cold_by_row[row] != 0u
        ? 0.0f
        : expf(log_decay_f32[
            ((uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT) + head]);
    beta = beta_f32[
        ((uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT) + head];

    kv_memory = 0.0f;
    for (element = 0u; element < SPARK_QWEN38_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN38_CUDA_DV) + column;
        value = state_cold_by_row[row] != 0u
            ? 0.0f
            : state_f32[state_base + state_index] * decay;
        state_shared[state_index] = value;
        kv_memory = fmaf(value, kn[element], kv_memory);
    }

    delta = (SparkLmBf16ToFloat(
        conv_out_bf16,
        conv_row + (2u * SPARK_QWEN38_MAX_MODEL_GDN_QK_DIMENSION) +
            ((uint64_t)head * SPARK_QWEN38_CUDA_DV) + column) - kv_memory) * beta;
    output = 0.0f;
    for (element = 0u; element < SPARK_QWEN38_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN38_CUDA_DV) + column;
        value = fmaf(kn[element], delta, state_shared[state_index]);
        state_shared[state_index] = value;
        output = fmaf(value, qn[element], output);
    }
    SparkLmFloatToBf16(
        core_out_bf16,
        ((uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION) +
            ((uint64_t)head * SPARK_QWEN38_CUDA_DV) + column,
        output);

    for (element = 0u; element < SPARK_QWEN38_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN38_CUDA_DV) + column;
        state_f32[state_base + state_index] = state_shared[state_index];
    }
}

// Gated head norm: fp32 RMSNorm over one value head, times weight, times
// silu(z). One block per (row, head), 128 threads. Norm before gate.
static __global__ void SparkQwen38GatedNormKernel(const void *core_bf16, const void *z_bf16, const void *norm_weight_bf16, void *output_bf16, uint32_t row_count, float epsilon)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.y,head = blockIdx.x,column = threadIdx.x;
	uint64_t index = ((uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION) + ((uint64_t)head * SPARK_QWEN38_CUDA_DV) + column;
	float value,variance;
	if ( row >= row_count )
		return;
	value = SparkLmBf16ToFloat(core_bf16,index);
	variance = SparkLmBlockReduceSum(value * value,reduce_scratch) / (float)SPARK_QWEN38_CUDA_DV;
	value = value * rsqrtf(variance + epsilon) * SparkLmBf16ToFloat(norm_weight_bf16,column) * SparkLmSwish(SparkLmBf16ToFloat(z_bf16,index));
	SparkLmFloatToBf16(output_bf16,index,value);
}

/*
 * Attention pre-pass for one decode token per row: per-head RMSNorm on the
 * query half of the fused query|gate projection and on the key projection,
 * partial RoPE on the first 64 dims of both, then the K and V rows land in
 * the paged cache at the row's slot. Fused layout: head h occupies columns
 * [h*512, h*512+256) query and [h*512+256, h*512+512) gate; the gate half is
 * left untouched here for the decode kernel to consume. One block per
 * (row, query head); key/value heads are written by the blocks whose query
 * head is the group leader so each cache row is written exactly once.
 */
static __global__ void SparkQwen38AttnPrepareKernel(
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
    /* Head-parallel geometry: rank r owns query heads
     * [r*local_heads, (r+1)*local_heads) and the matching local KV head
     * slice, so each rank's paged cache holds exactly 1/tp_degree of the
     * KV heads. tp_degree 1 reproduces the replicated layout exactly. */
    const uint32_t local_heads =
        SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT / tp_degree;
    const uint32_t local_kv_heads =
        SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT / tp_degree;
    const uint32_t local_token_elements =
        2u * local_kv_heads * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
    __shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
    __shared__ float query_shared[SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION];
    __shared__ float key_shared[SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION];
    __shared__ float rope_cosine[
        SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION / 2u];
    __shared__ float rope_sine[
        SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION / 2u];
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
    kv_head = head / SPARK_QWEN38_CUDA_ATTN_GROUP;
    if (row >= row_count ||
        blockIdx.x >= local_heads ||
        column >= SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION)
    {
        return;
    }

    query_base =
        ((uint64_t)row * 2u * (uint64_t)local_heads *
            SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) +
        ((uint64_t)blockIdx.x * 2u * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION);
    value = SparkLmBf16ToFloat(q_fused_bf16, query_base + column);
    sum_squares = SparkLmBlockReduceSum(
        value * value,
        reduce_scratch);
    inverse_rms = rsqrtf(
        sum_squares / (float)SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION +
        epsilon);
    query_shared[column] =
        value * inverse_rms *
        SparkLmBf16ToFloat(q_norm_weight_bf16, column);
    if (column < SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION / 2u)
    {
        float angle;

        pair = column;
        angle =
            (float)row_positions[row] * SparkQwen38RopeFrequency(pair);
        sincosf(angle, &rope_sine[pair], &rope_cosine[pair]);
    }
    __syncthreads();

    if (column < SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION / 2u)
    {
        float real;
        float imaginary;

        pair = column;
        real = query_shared[pair];
        imaginary = query_shared[
            pair + SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION / 2u];
        query_shared[pair] =
            real * rope_cosine[pair] - imaginary * rope_sine[pair];
        query_shared[
            pair + SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION / 2u] =
            imaginary * rope_cosine[pair] + real * rope_sine[pair];
    }
    __syncthreads();
    SparkLmFloatToBf16(
        q_fused_bf16,
        query_base + column,
        query_shared[column]);

    if ((blockIdx.x % SPARK_QWEN38_CUDA_ATTN_GROUP) != 0u)
    {
        return;
    }

    key_base =
        ((uint64_t)row * (uint64_t)local_kv_heads *
            SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) +
        ((uint64_t)(kv_head - (tp_rank * local_kv_heads)) *
            SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION);
    value = SparkLmBf16ToFloat(k_bf16, key_base + column);
    sum_squares = SparkLmBlockReduceSum(
        value * value,
        reduce_scratch);
    inverse_rms = rsqrtf(
        sum_squares / (float)SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION +
        epsilon);
    key_shared[column] =
        value * inverse_rms *
        SparkLmBf16ToFloat(k_norm_weight_bf16, column);
    __syncthreads();

    if (column < SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION / 2u)
    {
        float real;
        float imaginary;

        pair = column;
        real = key_shared[pair];
        imaginary = key_shared[
            pair + SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION / 2u];
        key_shared[pair] =
            real * rope_cosine[pair] - imaginary * rope_sine[pair];
        key_shared[
            pair + SPARK_QWEN38_MAX_MODEL_ATTN_ROPE_DIMENSION / 2u] =
            imaginary * rope_cosine[pair] + real * rope_sine[pair];
    }
    __syncthreads();

    slot = slot_mapping[row];
    block = slot / SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    offset = slot % SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    cache_base =
        ((uint64_t)block * cache_block_stride) +
        ((uint64_t)attn_layer_ordinal * cache_layer_stride) +
        ((uint64_t)offset * local_token_elements) +
        ((uint64_t)(kv_head - (tp_rank * local_kv_heads)) *
            SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION);
    SparkLmFloatToBf16(
        kv_cache_bf16,
        cache_base + column,
        key_shared[column]);
    SparkLmFloatToBf16(
        kv_cache_bf16,
        cache_base + ((uint64_t)local_kv_heads *
            SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) + column,
        SparkLmBf16ToFloat(v_bf16, key_base + column));
}

/*
 * Paged GQA decode with the fused per-head sigmoid gate, flash style:
 * eight warps stripe the context, each warp resolving its token's block
 * through the lane's table and computing the logit cooperatively - lanes
 * pair-load K so every fetch is a full transaction - into a per-warp
 * online softmax with the value half of the cache row accumulated in
 * registers. One pass, no per-token barriers, one staged merge at the
 * end where the gate multiplies the normalized output.
 */
static __device__ __forceinline__ uint64_t SparkQwen38AttnTokenBase(const uint32_t *block_indices, uint64_t lane_base, uint32_t token, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t local_kv_head, uint32_t local_token_elements)
{
	uint32_t block = __ldg(block_indices + lane_base + (token / SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS));
	return(((uint64_t)block * cache_block_stride) + ((uint64_t)attn_layer_ordinal * cache_layer_stride) + ((uint64_t)(token % SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) * local_token_elements) + ((uint64_t)local_kv_head * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION));
}

// Cross-warp merge with the fused sigmoid gate applied at the store.


// One token's logit: lanes pair-load the cached key against the shared
// query, warp-reduce, fixed 1/sqrt(head_dim) = 1/16 scale.


static __global__ void SparkQwen38AttnDecodeKernel(const void *q_fused_bf16, const void *kv_cache_bf16, const uint32_t *block_indices, const uint32_t *block_counts, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t lane_stride, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t tp_degree, uint32_t tp_rank)
{
    /* Head-parallel geometry, mirroring the prepare kernel: local query
     * heads per rank and the rank's local KV head slice; tp_degree 1 is
     * the replicated layout. */
    const uint32_t local_heads =
        SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT / tp_degree;
    const uint32_t local_kv_heads =
        SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT / tp_degree;
    const uint32_t local_token_elements =
        2u * local_kv_heads * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
    const uint32_t heads_per_cta = SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA;
    __shared__ float q_shared[
        SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA *
        SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION];
    __shared__ float merge_max[
        SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA * SPARK_LM_CTA_WARPS];
    __shared__ float merge_den[
        SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA * SPARK_LM_CTA_WARPS];
    __shared__ float merge_scale[
        SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA * SPARK_LM_CTA_WARPS];
    __shared__ float inverse_denominator[
        SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA];
    __shared__ float merge_acc[
        SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA * SPARK_LM_CTA_WARPS *
        SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION];
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
    float running_max[SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA];
    float running_den[SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA];
    float local_logit[SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA];
    float rescale[SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA];
    float weight[SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA];
    float2 accumulator[
        SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA]
        [SPARK_QWEN38_CUDA_ATTN_VALUE_PAIRS_PER_LANE];
    float2 key_pair;
    float2 value_pair;
    float head_max;
    float denominator;
    float merged;
    float gate;

    static_assert(
        SPARK_QWEN38_CUDA_ATTN_GROUP %
                SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA ==
            0u,
        "grouped attention CTAs must not cross KV-head ownership");
    row = blockIdx.y;
    head_base = (tp_rank * local_heads) + (blockIdx.x * heads_per_cta);
    kv_head = head_base / SPARK_QWEN38_CUDA_ATTN_GROUP;
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
        (context + SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) /
        SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    if (context == 0u ||
        available_block_count > lane_stride ||
        required_block_count > available_block_count)
    {
        float invalid_output;

        invalid_output = __int_as_float(0x7fc00000);
        for (element = threadIdx.x;
             element < heads_per_cta * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
             element += blockDim.x)
        {
            local_head = element / SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
            partial = element % SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
            out_base =
                ((uint64_t)row * (uint64_t)local_heads *
                    SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) +
                ((uint64_t)((blockIdx.x * heads_per_cta) + local_head) *
                    SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION);
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
                SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) +
            ((uint64_t)((blockIdx.x * heads_per_cta) + local_head) *
                2u * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION);
        for (element = threadIdx.x;
             element < SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
             element += blockDim.x)
        {
            q_shared[
                (local_head * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) +
                element] = SparkLmBf16ToFloat(
                    q_fused_bf16,
                    q_base + element);
        }
        running_max[local_head] = -3.0e38f;
        running_den[local_head] = 0.0f;
        #pragma unroll
        for (pair = 0u;
             pair < SPARK_QWEN38_CUDA_ATTN_VALUE_PAIRS_PER_LANE;
             ++pair)
        {
            accumulator[local_head][pair] = make_float2(0.0f, 0.0f);
        }
    }
    for (element = threadIdx.x;
         element < SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA *
             SPARK_LM_CTA_WARPS *
             SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
         element += blockDim.x)
    {
        merge_acc[element] = 0.0f;
    }
    __syncthreads();

    lane_base = (uint64_t)lane_index * lane_stride;
    for (token = warp; token < context; token += SPARK_LM_CTA_WARPS)
    {
        token_base = SparkQwen38AttnTokenBase(
            block_indices,
            lane_base,
            token,
            attn_layer_ordinal,
            cache_layer_stride,
            cache_block_stride,
            kv_head - (tp_rank * local_kv_heads),
            local_token_elements);
        #pragma unroll
        for (local_head = 0u; local_head < heads_per_cta; ++local_head)
        {
            local_logit[local_head] = 0.0f;
        }
        #pragma unroll
        for (pair = 0u;
             pair < SPARK_QWEN38_CUDA_ATTN_VALUE_PAIRS_PER_LANE;
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
                            SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) +
                        element],
                    key_pair.x,
                    local_logit[local_head]);
                local_logit[local_head] = fmaf(
                    q_shared[
                        (local_head *
                            SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) +
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
             pair < SPARK_QWEN38_CUDA_ATTN_VALUE_PAIRS_PER_LANE;
             ++pair)
        {
            value_pair = SparkLmLoadBf16Pair(
                kv_cache_bf16,
                ((token_base + ((uint64_t)local_kv_heads *
                    SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION)) >> 1u) +
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
             pair < SPARK_QWEN38_CUDA_ATTN_VALUE_PAIRS_PER_LANE;
             ++pair)
        {
            element = ((pair * SPARK_LM_WARP_LANES) + lane) << 1u;
            merge_acc[
                (((local_head * SPARK_LM_CTA_WARPS) + warp) *
                    SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) +
                element] = accumulator[local_head][pair].x;
            merge_acc[
                (((local_head * SPARK_LM_CTA_WARPS) + warp) *
                    SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) +
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
                SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) +
            ((uint64_t)((blockIdx.x * heads_per_cta) + local_head) *
                2u * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION);
        out_base =
            ((uint64_t)row * (uint64_t)local_heads *
                SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) +
            ((uint64_t)((blockIdx.x * heads_per_cta) + local_head) *
                SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION);
        for (element = threadIdx.x;
             element < SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
             element += blockDim.x)
        {
            merged = 0.0f;
            for (partial = 0u; partial < SPARK_LM_CTA_WARPS; ++partial)
            {
                merged = fmaf(
                    merge_acc[
                        (((local_head * SPARK_LM_CTA_WARPS) + partial) *
                            SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION) +
                        element],
                    merge_scale[
                        (local_head * SPARK_LM_CTA_WARPS) + partial],
                    merged);
            }
            gate = SparkLmSigmoid(SparkLmBf16ToFloat(
                q_fused_bf16,
                q_base + SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION + element));
            SparkLmFloatToBf16(
                head_out_bf16,
                out_base + element,
                merged * inverse_denominator[local_head] * gate);
        }
    }
}

// Embedding gather: one thread per (row, element); token ids are validated
// against the vocabulary on the host before upload, so the kernel trusts.
static __global__ void SparkQwen38EmbeddingGatherKernel(const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t row = (uint32_t)(index / SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION),element = (uint32_t)(index % SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION);
	if ( row >= row_count )
		return;
	SparkLmFloatToBf16(hidden_bf16,index,SparkLmBf16ToFloat(embedding_bf16,((uint64_t)token_ids[row] * SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION) + element));
}

/*
 * Chunked GDN prefill, mirroring the PROVEN CPU chunk stages one to one
 * (validation reference agrees with the recurrence at 2e-8 from a warmed
 * state). One launch sequence processes ONE 64-token chunk for one lane
 * across all value heads in parallel; the module loops chunks on the
 * stream, which serializes the state dependency for free. Workspace lives
 * in slot global memory (per head: qn/kn 64x128, decay/attn 64x64, w/kg
 * 64x128) because the set exceeds shared memory. This is the simple correct
 * formulation; the wmma tiling of the three inner products is the later
 * throughput commit, the same discipline as the decode attention.
 */
#define SPARK_QWEN38_CUDA_CHUNK SPARK_QWEN38_MAX_MODEL_GDN_CHUNK_TOKENS
#define SPARK_QWEN38_CUDA_GDN_QK_SHARED_BYTES \
    (2u * SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_DK * sizeof(float))
#define SPARK_QWEN38_CUDA_GDN_CHUNK_SHARED_BYTES \
    ((SPARK_QWEN38_CUDA_GDN_STATE_ELEMENTS + \
      (SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_DV) + \
      (2u * SPARK_QWEN38_CUDA_CHUNK)) * sizeof(float))
static_assert(
    SPARK_QWEN38_CUDA_GDN_CHUNK_SHARED_BYTES == 98816u,
    "Qwen GDN chunk shared layout must fit the SM 12.x 99-KB block limit");

typedef struct SparkQwen38ChunkWorkspaceView
{
	float *qn;
	float *kn;
	float *cum_g;
	float *decay;
	float *attn;
	float *w;
	float *kg;
} SparkQwen38ChunkWorkspaceView;

static __device__ __forceinline__ uint64_t SparkQwen38ChunkHeadOffset(uint32_t head, uint32_t per_head_elements)
{
	return((uint64_t)head * per_head_elements);
}

// Stage 1: per-head L2 norms with the 1/sqrt(dk) query scale, the intra-
// chunk decay cumsum, the decay mask and the strictly-lower beta-scaled
// -k_beta k^T attention seed. Block per head, thread per token row.
static __global__ void SparkQwen38ChunkPrepareKernel(const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, SparkQwen38ChunkWorkspaceView views, uint32_t token_count)
{
	uint32_t head = blockIdx.x,row = threadIdx.x,key_head = head / SPARK_QWEN38_CUDA_GVA_GROUP,element,column;
	uint64_t conv_row,qk_base = SparkQwen38ChunkHeadOffset(head,SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_DK);
	uint64_t mat_base = SparkQwen38ChunkHeadOffset(head,SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_CHUNK);
	float total,value,product;
	if ( row >= token_count )
		return;
	conv_row = (uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS;
	total = 0.0f;
	for (element = 0; element < SPARK_QWEN38_CUDA_DK; element++)
	{
		value = SparkLmBf16ToFloat(conv_out_bf16,conv_row + ((uint64_t)key_head * SPARK_QWEN38_CUDA_DK) + element);
		total += (value * value);
	}
	total = rsqrtf(total + 1e-6f) * rsqrtf((float)SPARK_QWEN38_CUDA_DK);
	for (element = 0; element < SPARK_QWEN38_CUDA_DK; element++)
		views.qn[qk_base + ((uint64_t)row * SPARK_QWEN38_CUDA_DK) + element] = SparkLmBf16ToFloat(conv_out_bf16,conv_row + ((uint64_t)key_head * SPARK_QWEN38_CUDA_DK) + element) * total;
	total = 0.0f;
	for (element = 0; element < SPARK_QWEN38_CUDA_DK; element++)
	{
		value = SparkLmBf16ToFloat(conv_out_bf16,conv_row + SPARK_QWEN38_MAX_MODEL_GDN_QK_DIMENSION + ((uint64_t)key_head * SPARK_QWEN38_CUDA_DK) + element);
		total += (value * value);
	}
	total = rsqrtf(total + 1e-6f);
	for (element = 0; element < SPARK_QWEN38_CUDA_DK; element++)
		views.kn[qk_base + ((uint64_t)row * SPARK_QWEN38_CUDA_DK) + element] = SparkLmBf16ToFloat(conv_out_bf16,conv_row + SPARK_QWEN38_MAX_MODEL_GDN_QK_DIMENSION + ((uint64_t)key_head * SPARK_QWEN38_CUDA_DK) + element) * total;
	if ( row == 0u )
	{
		total = 0.0f;
		for (element = 0; element < token_count; element++)
		{
			total += log_decay_f32[((uint64_t)element * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT) + head];
			views.cum_g[SparkQwen38ChunkHeadOffset(head,SPARK_QWEN38_CUDA_CHUNK) + element] = total;
		}
	}
	__syncthreads();
	for (column = 0; column < token_count; column++)
	{
		value = column <= row ? __expf(views.cum_g[SparkQwen38ChunkHeadOffset(head,SPARK_QWEN38_CUDA_CHUNK) + row] - views.cum_g[SparkQwen38ChunkHeadOffset(head,SPARK_QWEN38_CUDA_CHUNK) + column]) : 0.0f;
		views.decay[mat_base + ((uint64_t)row * SPARK_QWEN38_CUDA_CHUNK) + column] = value;
		product = 0.0f;
		for (element = 0; element < SPARK_QWEN38_CUDA_DK && column < row; element++)
			product += (views.kn[qk_base + ((uint64_t)row * SPARK_QWEN38_CUDA_DK) + element] * beta_f32[((uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT) + head] * views.kn[qk_base + ((uint64_t)column * SPARK_QWEN38_CUDA_DK) + element]);
		views.attn[mat_base + ((uint64_t)row * SPARK_QWEN38_CUDA_CHUNK) + column] = column < row ? -(product * value) : 0.0f;
	}
}

// Stage 2: the forward-substitution UT transform T = (I - A)^-1, in place.
// The row recurrence is sequential; columns of a row are parallel. Block
// per head, thread per column.
static __global__ void SparkQwen38ChunkSolveKernel(SparkQwen38ChunkWorkspaceView views, uint32_t token_count)
{
	/* The oracle snapshots the ORIGINAL row (T[i,:i] = A[i,:i] + A[i,:i] x
	 * T[:i,:i]) BEFORE the row's columns update. Reading live A[row,e]
	 * entries races the other columns' writes and folds extra powers of A
	 * into the transform, so each row is staged first, barrier, then
	 * applied - exactly the reference's clone-then-sum. */
	__shared__ float solve_row[SPARK_QWEN38_CUDA_CHUNK];
	uint32_t head = blockIdx.x,column = threadIdx.x,row,element;
	uint64_t mat_base = SparkQwen38ChunkHeadOffset(head,SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_CHUNK);
	float accumulator;
	for (row = 1; row < token_count; row++)
	{
		if ( column < row )
			solve_row[column] = views.attn[mat_base + ((uint64_t)row * SPARK_QWEN38_CUDA_CHUNK) + column];
		__syncthreads();
		if ( column < row )
		{
			accumulator = solve_row[column];
			for (element = 0; element < row; element++)
				accumulator += (solve_row[element] * views.attn[mat_base + ((uint64_t)element * SPARK_QWEN38_CUDA_CHUNK) + column]);
			views.attn[mat_base + ((uint64_t)row * SPARK_QWEN38_CUDA_CHUNK) + column] = accumulator;
		}
		__syncthreads();
	}
	if ( column < token_count )
		views.attn[mat_base + ((uint64_t)column * SPARK_QWEN38_CUDA_CHUNK) + column] += 1.0f;
}

// Stage 3: w = T (v o beta) and kg = T (k o beta o e^G). Block per (head,
// token row), thread per output column striped over dv then dk.
static __global__ void SparkQwen38ChunkTransformKernel(const void *conv_out_bf16, const float *beta_f32, SparkQwen38ChunkWorkspaceView views, uint32_t token_count)
{
	__shared__ float exp_cum_g[SPARK_QWEN38_CUDA_CHUNK];
	uint32_t head = blockIdx.x,row = blockIdx.y,column = threadIdx.x,element;
	uint64_t mat_base = SparkQwen38ChunkHeadOffset(head,SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_CHUNK);
	uint64_t vec_base = SparkQwen38ChunkHeadOffset(head,SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_DK);
	float accumulator,transform;
	if ( row >= token_count )
		return;
	if ( threadIdx.x < token_count )
		exp_cum_g[threadIdx.x] = __expf(
			views.cum_g[
				SparkQwen38ChunkHeadOffset(
					head,
					SPARK_QWEN38_CUDA_CHUNK) +
				threadIdx.x]);
	__syncthreads();
	accumulator = 0.0f;
	for (element = 0; element < token_count; element++)
	{
		transform = views.attn[mat_base + ((uint64_t)row * SPARK_QWEN38_CUDA_CHUNK) + element] * beta_f32[((uint64_t)element * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT) + head];
		accumulator += (transform * SparkLmBf16ToFloat(conv_out_bf16,((uint64_t)element * SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS) + (2u * SPARK_QWEN38_MAX_MODEL_GDN_QK_DIMENSION) + ((uint64_t)head * SPARK_QWEN38_CUDA_DV) + column));
	}
	views.w[vec_base + ((uint64_t)row * SPARK_QWEN38_CUDA_DV) + column] = accumulator;
	accumulator = 0.0f;
	for (element = 0; element < token_count; element++)
	{
		transform = views.attn[mat_base + ((uint64_t)row * SPARK_QWEN38_CUDA_CHUNK) + element] * beta_f32[((uint64_t)element * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT) + head] * exp_cum_g[element];
		accumulator += (transform * views.kn[vec_base + ((uint64_t)element * SPARK_QWEN38_CUDA_DK) + column]);
	}
	views.kg[vec_base + ((uint64_t)row * SPARK_QWEN38_CUDA_DK) + column] = accumulator;
}


static __global__ void SparkQwen38ChunkQkDecayKernel(SparkQwen38ChunkWorkspaceView views, uint32_t token_count)
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
        (SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_DK);
    vector_base = SparkQwen38ChunkHeadOffset(
        head,
        SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_DK);
    matrix_base = SparkQwen38ChunkHeadOffset(
        head,
        SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_CHUNK);

    for (vector_element = threadIdx.x;
         vector_element < token_count * SPARK_QWEN38_CUDA_DK;
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
            for (element = 0u; element < SPARK_QWEN38_CUDA_DK; ++element)
            {
                dot = fmaf(
                    qn_shared[(row * SPARK_QWEN38_CUDA_DK) + element],
                    kn_shared[(column * SPARK_QWEN38_CUDA_DK) + element],
                    dot);
            }
            views.decay[
                matrix_base +
                ((uint64_t)row * SPARK_QWEN38_CUDA_CHUNK) + column] *= dot;
        }
        else
        {
            views.decay[
                matrix_base +
                ((uint64_t)row * SPARK_QWEN38_CUDA_CHUNK) + column] = 0.0f;
        }
    }
}

// Stage 4: v_new = w - kg S, out = (q o e^G) S + (q k^T o D) v_new, and the
// carried state S <- S e^G_last + (k o e^(G_last - G))^T v_new. Block per
// (head, state row is the thread's dk stripe? No): thread per dv column,
// mirroring the decode step's coalesced state-column ownership.
static __global__ void SparkQwen38ChunkStepKernel(const float *log_decay_f32, SparkQwen38ChunkWorkspaceView views, float *state_f32, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal, uint64_t state_lane_stride, uint64_t state_layer_stride)
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
    v_new_shared = state_shared + SPARK_QWEN38_CUDA_GDN_STATE_ELEMENTS;
    exp_cum_g_shared =
        v_new_shared + (SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_DV);
    carry_decay_shared =
        exp_cum_g_shared + SPARK_QWEN38_CUDA_CHUNK;
    vector_base = SparkQwen38ChunkHeadOffset(
        head,
        SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_DK);
    g_base = SparkQwen38ChunkHeadOffset(
        head,
        SPARK_QWEN38_CUDA_CHUNK);
    matrix_base = SparkQwen38ChunkHeadOffset(
        head,
        SPARK_QWEN38_CUDA_CHUNK * SPARK_QWEN38_CUDA_CHUNK);
    state_base =
        ((uint64_t)lane_index * state_lane_stride) +
        ((uint64_t)gdn_layer_ordinal * state_layer_stride) +
        ((uint64_t)head * SPARK_QWEN38_CUDA_GDN_STATE_ELEMENTS);
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

    for (element = 0u; element < SPARK_QWEN38_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN38_CUDA_DV) + column;
        state_shared[state_index] = state_f32[state_base + state_index];
    }

    for (row = 0u; row < token_count; ++row)
    {
        accumulator = 0.0f;
        for (element = 0u; element < SPARK_QWEN38_CUDA_DK; ++element)
        {
            accumulator = fmaf(
                views.kg[
                    vector_base +
                    ((uint64_t)row * SPARK_QWEN38_CUDA_DK) + element],
                state_shared[(element * SPARK_QWEN38_CUDA_DV) + column],
                accumulator);
        }
        v_new_shared[(row * SPARK_QWEN38_CUDA_DV) + column] =
            views.w[
                vector_base +
                ((uint64_t)row * SPARK_QWEN38_CUDA_DV) + column] - accumulator;
    }

    for (row = 0u; row < token_count; ++row)
    {
        accumulator = 0.0f;
        for (element = 0u; element < SPARK_QWEN38_CUDA_DK; ++element)
        {
            accumulator = fmaf(
                views.qn[
                    vector_base +
                    ((uint64_t)row * SPARK_QWEN38_CUDA_DK) + element],
                state_shared[(element * SPARK_QWEN38_CUDA_DV) + column],
                accumulator);
        }
        accumulator *= exp_cum_g_shared[row];
        for (element = 0u; element <= row; ++element)
        {
            accumulator = fmaf(
                views.decay[
                    matrix_base +
                    ((uint64_t)row * SPARK_QWEN38_CUDA_CHUNK) + element],
                v_new_shared[(element * SPARK_QWEN38_CUDA_DV) + column],
                accumulator);
        }
        SparkLmFloatToBf16(
            core_out_bf16,
            ((uint64_t)row * SPARK_QWEN38_MAX_MODEL_GDN_VALUE_DIMENSION) +
                ((uint64_t)head * SPARK_QWEN38_CUDA_DV) + column,
            accumulator);
    }

    for (element = 0u; element < SPARK_QWEN38_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN38_CUDA_DV) + column;
        carry = state_shared[state_index] *
            exp_cum_g_shared[token_count - 1u];
        for (row = 0u; row < token_count; ++row)
        {
            carry = fmaf(
                views.kn[
                    vector_base +
                    ((uint64_t)row * SPARK_QWEN38_CUDA_DK) + element] *
                    carry_decay_shared[row],
                v_new_shared[(row * SPARK_QWEN38_CUDA_DV) + column],
                carry);
        }
        state_shared[state_index] = carry;
        state_f32[state_base + state_index] = state_shared[state_index];
    }
}

/*
 * Chunked depthwise causal conv for one lane's whole prefill frame: one
 * thread per channel slides a 4-tap register window over token_count
 * consecutive positions, seeded by the carried tail, silu on each output.
 * The register triple left after the walk is exactly the oracle
 * RefConvChannel tail for every token_count including short frames, so the
 * write-back needs no blending cases. Frames chain: the tail written here
 * seeds the next frame's first window.
 */
static __global__ void SparkQwen38ChunkConvKernel(const void *qkv_bf16, const void *conv_weight_bf16, void *conv_out_bf16, void *conv_tail_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal, uint64_t tail_lane_stride, uint64_t tail_layer_stride)
{
	uint32_t channel = (blockIdx.x * blockDim.x) + threadIdx.x,token,tap;
	uint64_t tail_base,element;
	float window[4],weight[4],accumulator;
	if ( channel >= SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS )
		return;
	tail_base = ((uint64_t)lane_index * tail_lane_stride) + ((uint64_t)gdn_layer_ordinal * tail_layer_stride) + ((uint64_t)channel * SPARK_QWEN38_MAX_MODEL_GDN_CONV_TAIL_COLUMNS);
	window[0] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 0u);
	window[1] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 1u);
	window[2] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 2u);
	for (tap = 0; tap < SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL; tap++)
		weight[tap] = SparkLmBf16ToFloat(conv_weight_bf16,((uint64_t)channel * SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL) + tap);
	for (token = 0; token < token_count; token++)
	{
		element = ((uint64_t)token * SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS) + channel;
		window[3] = SparkLmBf16ToFloat(qkv_bf16,element);
		accumulator = 0.0f;
		for (tap = 0; tap < SPARK_QWEN38_MAX_MODEL_GDN_CONV_KERNEL; tap++)
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

// Residual add and SwiGLU combine, both row-shaped elementwise.
static __global__ void SparkQwen38ResidualAddKernel(void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
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

extern "C" cudaError_t SparkQwen38LaunchFusedResidualRmsNorm(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
    size_t shared_memory_bytes = (size_t)dimension * sizeof(float);

    SparkLmFusedResidualRmsNormKernel<<<row_count, SPARK_LM_CTA_THREADS, shared_memory_bytes, stream>>>(hidden_bf16, delta_bf16, gain_bf16, output_bf16, row_count, dimension, epsilon);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkQwen38LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
    size_t shared_memory_bytes = (size_t)dimension * sizeof(float);

    SparkLmRmsNormKernel<<<row_count, SPARK_LM_CTA_THREADS, shared_memory_bytes, stream>>>(input_bf16, gain_bf16, output_bf16, row_count, dimension, epsilon);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkQwen38LaunchLinear(cudaStream_t stream, const SparkQwen38MaxLinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count)
{
	/* The dense BF16 spine at B>=32 uses the M-group tile: the plain grid
	 * re-reads each weight strip once per m-tile (16x at B=256); the
	 * m-loop stages it once per k-stage for eight m-tiles. */
	if ( view->weight_format == SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 && row_count >= 2u * SPARK_LM_TILE && view->input_dimension > SPARK_LM_TILE_K && view->output_dimension != 0u )
		return(SparkLmHostLaunchBatchedLinearMloop(stream,view->weight_payload,input_bf16,output_bf16,row_count,view->input_dimension,view->output_dimension));
	return(SparkLmHostLaunchBatchedLinear<32u>(stream,view->weight_format,view->weight_payload,view->weight_scale_e8m0,input_bf16,output_bf16,row_count,view->input_dimension,view->output_dimension));
}

extern "C" cudaError_t SparkQwen38LaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen38GdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen38GdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal)
{
	dim3 grid((SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS,row_count,1u);
	SparkQwen38ConvUpdateKernel<<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(qkv_bf16,weights->conv_weight_bf16,conv_out_bf16,pool->conv_tail_bf16,row_lane_indices,pool->state_cold_by_row,row_count,gdn_layer_ordinal,pool->conv_tail_lane_stride_elements,pool->conv_tail_layer_stride_elements);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen38LaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen38GdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count)
{
	SparkQwen38DecayBetaKernel<<<row_count,SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT,0,stream>>>(decay_pre_bf16,beta_pre_bf16,weights->a_log_f32,weights->dt_bias_f32,log_decay_f32,beta_f32,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen38LaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen38GdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal)
{
    dim3 grid(
        SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT,
        row_count,
        1u);
    SparkQwen38GdnStepKernel<<<
        grid,
        SPARK_QWEN38_CUDA_DV,
        SPARK_QWEN38_CUDA_GDN_DECODE_SHARED_BYTES,
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
            pool->state_layer_stride_elements);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkQwen38LaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen38GdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon)
{
	dim3 grid(SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT,row_count,1u);
	SparkQwen38GatedNormKernel<<<grid,SPARK_QWEN38_CUDA_DV,0,stream>>>(core_bf16,z_bf16,weights->gdn_norm_weight_bf16,output_bf16,row_count,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen38LaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen38AttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon, uint32_t tp_degree, uint32_t tp_rank)
{
	if ( tp_degree == 0u || tp_degree > SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT || tp_rank >= tp_degree || (SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT % tp_degree) != 0u || (SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT % tp_degree) != 0u || (SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT / tp_degree) == 0u )
		return(cudaErrorInvalidValue);
	dim3 grid(SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT / tp_degree,row_count,1u);
	SparkQwen38AttnPrepareKernel<<<grid,SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION,0,stream>>>(q_fused_bf16,k_bf16,v_bf16,weights->query_norm_weight_bf16,weights->key_norm_weight_bf16,kv_cache_bf16,slot_mapping,row_positions,row_count,attn_layer_ordinal,cache_layer_stride,cache_block_stride,epsilon,tp_degree,tp_rank);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen38LaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen38KvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t tp_degree, uint32_t tp_rank)
{
    if ( tp_degree == 0u || tp_degree > SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT || tp_rank >= tp_degree || (SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT % tp_degree) != 0u || (SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT % tp_degree) != 0u || (SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT / tp_degree) == 0u )
        return(cudaErrorInvalidValue);
    dim3 grid(
        (SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT / tp_degree) /
            SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA,
        row_count,
        1u);
    static_assert(
        SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT %
            SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA == 0u,
        "query-head count must divide the grouped attention CTA width");
    static_assert(
        SPARK_QWEN38_CUDA_ATTN_GROUP %
            SPARK_QWEN38_CUDA_ATTN_HEADS_PER_CTA == 0u,
        "a grouped attention CTA may not cross KV-head ownership");
    SparkQwen38AttnDecodeKernel<<<grid, SPARK_LM_CTA_THREADS, 0u, stream>>>(
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
        tp_rank);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkQwen38LaunchChunkConv(cudaStream_t stream, const void *qkv_bf16, const SparkQwen38GdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen38GdnStatePool *pool, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal)
{
	if ( token_count == 0u )
		return(cudaErrorInvalidValue);
	SparkQwen38ChunkConvKernel<<<(SPARK_QWEN38_MAX_MODEL_GDN_CONV_CHANNELS + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS,SPARK_LM_CTA_THREADS,0,stream>>>(qkv_bf16,weights->conv_weight_bf16,conv_out_bf16,pool->conv_tail_bf16,lane_index,token_count,gdn_layer_ordinal,pool->conv_tail_lane_stride_elements,pool->conv_tail_layer_stride_elements);
	return(cudaGetLastError());
}

/*
 * One chunk of one lane's prefill through the GDN core: conv_out and the
 * decay/beta arrays hold token_count (at most 64) consecutive positions.
 * The module loops chunks on the stream; the state dependency serializes
 * for free. Workspace pointers are slot-owned device buffers sized per the
 * view layout (per head: qn/kn/w/kg 64 x 128, decay/attn 64 x 64, cum_g 64).
 */
extern "C" cudaError_t SparkQwen38LaunchGdnChunk(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *workspace_qn, float *workspace_kn, float *workspace_cum_g, float *workspace_decay, float *workspace_attn, float *workspace_w, float *workspace_kg, const SparkQwen38GdnStatePool *pool, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal)
{
    SparkQwen38ChunkWorkspaceView views;
    cudaError_t status;
    dim3 transform_grid(
        SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT,
        token_count,
        1u);

    views.qn = workspace_qn;
    views.kn = workspace_kn;
    views.cum_g = workspace_cum_g;
    views.decay = workspace_decay;
    views.attn = workspace_attn;
    views.w = workspace_w;
    views.kg = workspace_kg;
    if (token_count == 0u || token_count > SPARK_QWEN38_CUDA_CHUNK)
    {
        return cudaErrorInvalidValue;
    }

    SparkQwen38ChunkPrepareKernel<<<
        SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT,
        SPARK_QWEN38_CUDA_CHUNK,
        0u,
        stream>>>(conv_out_bf16, log_decay_f32, beta_f32, views, token_count);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    SparkQwen38ChunkSolveKernel<<<
        SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT,
        SPARK_QWEN38_CUDA_CHUNK,
        0u,
        stream>>>(views, token_count);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    SparkQwen38ChunkTransformKernel<<<
        transform_grid,
        SPARK_QWEN38_CUDA_DV,
        0u,
        stream>>>(conv_out_bf16, beta_f32, views, token_count);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    SparkQwen38ChunkQkDecayKernel<<<
        SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT,
        SPARK_LM_CTA_THREADS,
        SPARK_QWEN38_CUDA_GDN_QK_SHARED_BYTES,
        stream>>>(views, token_count);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    SparkQwen38ChunkStepKernel<<<
        SPARK_QWEN38_MAX_MODEL_GDN_VALUE_HEAD_COUNT,
        SPARK_QWEN38_CUDA_DV,
        SPARK_QWEN38_CUDA_GDN_CHUNK_SHARED_BYTES,
        stream>>>(
            log_decay_f32,
            views,
            pool->state_f32,
            core_out_bf16,
            lane_index,
            token_count,
            gdn_layer_ordinal,
            pool->state_lane_stride_elements,
            pool->state_layer_stride_elements);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkQwen38LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count)
{
	uint64_t elements = (uint64_t)row_count * SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION;
	SparkQwen38EmbeddingGatherKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(token_ids,embedding_bf16,hidden_bf16,row_count);
	return(cudaGetLastError());
}

/* TP all-reduce combine: destination += source, one block per row, bf16
 * pairs added in f32 - the SparkTpDeviceCollective combine callback for
 * the hidden_transport backend. */
static __global__ void SparkQwen38TpCombineAddKernel(void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width)
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

extern "C" cudaError_t SparkQwen38LaunchTpCombineAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkQwen38TpCombineAddKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen38LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t pairs = ((uint64_t)row_count * dimension + 1u) >> 1u;
	SparkQwen38ResidualAddKernel<<<(uint32_t)((pairs + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,delta_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen38LaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchHeadShadowQuantize<SPARK_LM_HEAD_SHADOW_GROUP>(stream,head_bf16,shadow_payload,shadow_scale,error_norm,candidate_count,hidden_dimension));
}

// Screened exact head, the mimo25 pattern: coarse fp4 tile, certified
// screen, exact rescore, device-side overflow fallback; the token
// equals the reference argmax always.
extern "C" cudaError_t SparkQwen38LaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count)
{
	return(SparkLmHostLaunchHeadScreenedArgmax(stream,hidden_bf16,head_weight_bf16,shadow_payload,shadow_scale,error_norm,logits_bf16,candidate_ids,candidate_counts,output_token_ids,row_count,candidate_count,SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION));
}

extern "C" cudaError_t SparkQwen38LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count)
{
	SparkLmHeadArgmaxKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,head_weight_bf16,token_ids,output_token_ids,row_count,SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION,candidate_count);
	return(cudaGetLastError());
}
/*
 * Routed MoE execution, Qwen 3.8: BF16 router gate -> per-row top-k
 * selection with softmax-renormalized weights -> common grouped-MoE kernels
 * over FP8 experts -> weighted pair reduce (overwrite semantics: the
 * mixture starts fresh) into the delta buffer; BF16 shared expert with the
 * learned scalar sigmoid gate (Linear(hidden,1) on the MoE input). All
 * launchers are stream-ordered.
 */
#define SPARK_QWEN38_ROUTER_SORT_CAPACITY 512u

static __device__ __forceinline__ float SparkQwen38WarpReduceMax(float value)
{
	#pragma unroll
	for (uint32_t offset = SPARK_LM_WARP_LANES >> 1u; offset != 0u; offset >>= 1u)
		value = fmaxf(value,__shfl_down_sync(0xffffffffu,value,offset));
	return(value);
}

static __global__ void SparkQwen38GateSelectKernel(
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
        SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT <=
            SPARK_QWEN38_ROUTER_SORT_CAPACITY,
        "qwen38 expert count exceeds router sort capacity");
    static_assert(
        SPARK_LM_MOE_MAX_TOPK <= SPARK_LM_WARP_LANES,
        "qwen38 router normalization requires one warp");
    row = blockIdx.x;
    if ( row >= row_count || expert_count == 0u ||
        expert_count > SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT ||
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
        /* A NaN score must rank LAST, never first: the ordered-key map
         * sends NaN to key 0, and key 0 decodes as UINT32_MAX here, which
         * the route build would histogram out of bounds. -inf keeps the
         * expert selectable-safe (it sorts below every finite logit). */
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
        /* Softmax over the chosen top-k, the Qwen3_5MoeTopKRouter form:
         * routing_weights = softmax(logits) then topk then renormalize
         * over the selected experts. Raw logits divided by their sum (the
         * old code) made every negative logit a negative mixture weight.
         * Non-selected lanes hold -INFINITY, so the warp max and sum see
         * exactly the top-k and their exp() contributes zero. */
        float max_score = SparkQwen38WarpReduceMax(selected_score);
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
            /* A zero or non-finite total (all-NaN logits) yields zero
             * weights instead of NaN, which would poison the pair reduce
             * and the next layer's gate scores. */
            weights_f32[((uint64_t)row * topk) + rank] =
                selected_total > 0.0f
                ? route_scale * exp_score / selected_total
                : 0.0f;
        }
    }
}

static __global__ void SparkQwen38SwiGluKernel(const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension)
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

/* Shared-expert gate: the checkpoint's shared_expert_gate is a
 * Linear(hidden, 1) (Qwen3_5MoeSparseMoeBlock: sigmoid(gate(normed_input))
 * times the shared output). One scalar per row computed from the MoE input,
 * NOT sigmoid(weight[d]) per channel. */
static __global__ void SparkQwen38SharedGateKernel(void *accum_bf16, const void *gate_weight_bf16, const void *gate_input_bf16, uint32_t row_count, uint32_t dimension)
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

extern "C" cudaError_t SparkQwen38LaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32)
{
	SparkQwen38GateSelectKernel<<<row_count, SPARK_LM_CTA_THREADS, 0, stream>>>(scores_f32,bias_f32,row_count,expert_count,topk,route_scale,indices_u32,weights_f32);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen38LaunchMoeRoute(cudaStream_t stream, const uint32_t *route_expert, uint32_t rows, uint32_t expert_width, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2)
{
	int32_t launch_status = LmRouteBuild<SPARK_LM_CTA_THREADS,SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT>(route_expert,rows,rows * SPARK_QWEN38_MAX_MODEL_EXPERTS_PER_TOKEN,SPARK_QWEN38_MAX_MODEL_EXPERTS_PER_TOKEN,group_row_offset,route_packed_row,route_source_token,expert_width,SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION,SPARK_LM_TILE_N,SPARK_LM_TILE_N,group_tile_prefix_w1,group_tile_prefix_w2,stream);
	return(launch_status == LM_LAUNCH_OK ? cudaSuccess : cudaErrorLaunchFailure);
}

extern "C" cudaError_t SparkQwen38LaunchFusedExpertW13Act(cudaStream_t stream, const SparkQwen38MaxLinearView *w1, const SparkQwen38MaxLinearView *w3, const void *input_bf16, const uint32_t *route_source_token, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *activated_bf16, uint32_t rows, uint32_t expert_width, float limit, uint32_t multiprocessor_count)
{
	cudaError_t status;
	uint64_t required_rows = (uint64_t)SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT * expert_width;
	if ( w1 == 0 || w3 == 0 || input_bf16 == 0 || route_source_token == 0 || group_row_offset == 0 || group_tile_prefix == 0 || activated_bf16 == 0 || w1->weight_format != SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 || w3->weight_format != SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 || w1->weight_payload == 0 || w3->weight_payload == 0 || w1->weight_scale_e8m0 == 0 || w3->weight_scale_e8m0 == 0 || w1->output_dimension != required_rows || w3->output_dimension != required_rows || w1->input_dimension != SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION || w3->input_dimension != SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION )
		return(cudaErrorInvalidValue);
	status = SparkLmHostLaunchSm121FusedExpertW13(stream,w1->weight_payload,w1->weight_scale_e8m0,w3->weight_payload,w3->weight_scale_e8m0,input_bf16,route_source_token,group_row_offset,group_tile_prefix,activated_bf16,rows,SPARK_QWEN38_MAX_MODEL_EXPERTS_PER_TOKEN,SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT,SPARK_QWEN38_MAX_MODEL_HIDDEN_DIMENSION,expert_width,limit,multiprocessor_count);
	return(status);
}

extern "C" cudaError_t SparkQwen38LaunchExpertDown(cudaStream_t stream, const SparkQwen38MaxLinearView *stacked, const void *input_bf16, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t expert_width, uint32_t hidden_dimension, uint32_t multiprocessor_count)
{
	uint64_t required_rows = (uint64_t)SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT * hidden_dimension;
	if ( stacked == 0 || input_bf16 == 0 || group_row_offset == 0 || group_tile_prefix == 0 || output_bf16 == 0 || stacked->weight_format != SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 || stacked->weight_payload == 0 || stacked->weight_scale_e8m0 == 0 || stacked->output_dimension != required_rows || stacked->input_dimension != expert_width )
		return(cudaErrorInvalidValue);
	return(SparkLmHostLaunchSm121ExpertW2(stream,stacked->weight_payload,stacked->weight_scale_e8m0,input_bf16,group_row_offset,group_tile_prefix,output_bf16,rows,SPARK_QWEN38_MAX_MODEL_EXPERTS_PER_TOKEN,SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT,expert_width,hidden_dimension,multiprocessor_count));
}

extern "C" cudaError_t SparkQwen38LaunchMoePairReduce(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchMoePairReduce(stream,slot_out_bf16,inverse_map,pair_weights_f32,accum_bf16,row_count,SPARK_QWEN38_MAX_MODEL_EXPERTS_PER_TOKEN,hidden_dimension));
}

/* The routed-MoE mixture must START from zero: the attention/GDN delta was
 * already folded into hidden by the fused residual norm, and accumulating
 * on top of it would apply that delta twice per layer. */
extern "C" cudaError_t SparkQwen38LaunchMoePairReduceOverwrite(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *output_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchMoePairReduceOverwrite(stream,slot_out_bf16,inverse_map,pair_weights_f32,output_bf16,row_count,SPARK_QWEN38_MAX_MODEL_EXPERTS_PER_TOKEN,hidden_dimension));
}

extern "C" cudaError_t SparkQwen38LaunchSwiGlu(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension)
{
	uint32_t threads = 256u;
	uint64_t pairs = ((uint64_t)row_count * dimension) >> 1u;
	uint32_t blocks = (uint32_t)((pairs + threads - 1u) / threads);
	SparkQwen38SwiGluKernel<<<(blocks == 0u ? 1u : blocks), threads, 0, stream>>>(gate_bf16,up_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen38LaunchSharedGate(cudaStream_t stream, void *accum_bf16, const void *gate_weight_bf16, const void *gate_input_bf16, uint32_t row_count, uint32_t dimension)
{
	SparkQwen38SharedGateKernel<<<row_count, SPARK_LM_CTA_THREADS, 0, stream>>>(accum_bf16,gate_weight_bf16,gate_input_bf16,row_count,dimension);
	return(cudaGetLastError());
}

/* Qwen 3.8 router gate: plain bf16 dot to f32 scores (no activation here;
 * the select kernel softmax-normalizes the chosen top-k). */
static __global__ void SparkQwen38GateScoresKernel(const void *weight_bf16, const void *input_bf16, float *scores_f32, uint32_t row_count, uint32_t input_dimension, uint32_t expert_count)
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

extern "C" cudaError_t SparkQwen38LaunchGateScores(cudaStream_t stream, const SparkQwen38MaxLinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count)
{
	uint32_t warp_count = SPARK_LM_CTA_THREADS / SPARK_LM_WARP_LANES;
	uint32_t expert_blocks = (gate->output_dimension + warp_count - 1u) / warp_count;
	if ( gate == 0 || input_bf16 == 0 || scores_f32 == 0 || gate->weight_format != SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 || gate->input_dimension == 0u )
		return(cudaErrorInvalidValue);
	SparkQwen38GateScoresKernel<<<dim3(row_count,expert_blocks),SPARK_LM_CTA_THREADS,gate->input_dimension * sizeof(float),stream>>>(gate->weight_payload,input_bf16,scores_f32,row_count,gate->input_dimension,gate->output_dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen38ConfigureCudaKernels(void)
{
    cudaError_t status;

    status = cudaFuncSetAttribute(
        SparkQwen38GdnStepKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)SPARK_QWEN38_CUDA_GDN_DECODE_SHARED_BYTES);
    if (status != cudaSuccess)
    {
        fprintf(stderr, "qwen38 configure gdn_step failed %d shared=%d\n", (int)status, (int)SPARK_QWEN38_CUDA_GDN_DECODE_SHARED_BYTES);
        return status;
    }
    status = cudaFuncSetAttribute(
        SparkQwen38ChunkQkDecayKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)SPARK_QWEN38_CUDA_GDN_QK_SHARED_BYTES);
    if (status != cudaSuccess)
    {
        return status;
    }
    status = cudaFuncSetAttribute(
        SparkQwen38ChunkStepKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)SPARK_QWEN38_CUDA_GDN_CHUNK_SHARED_BYTES);
    if (status != cudaSuccess)
    {
        return status;
    }
    /* The scalar Linear path stages the input row in dynamic shared memory;
     * the widest qwen38 input is the 16384-wide attention/GDN output
     * projection (64KB of floats), past the 48KB static ceiling. */
    return cudaFuncSetAttribute(
        (const void *)SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,SPARK_LM_CTA_WARPS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)(SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_DIMENSION * sizeof(float)));
}

/*
 * FP8 grouped expert linear via the common grouped-scalar path (FP8_E4M3
 * F32B128). The pack stores experts expert-major, so the per-group strides
 * are constant; source_row_map is the route's packed->source row table.
 */
extern "C" cudaError_t SparkQwen38LaunchGroupedExpertLinear(
	cudaStream_t stream,
	const SparkQwen38MaxLinearView *view,
	const void *input_bf16,
	const uint32_t *source_row_map,
	const uint32_t *group_row_offset,
	const uint32_t *group_tile_prefix,
	void *output_bf16,
	uint32_t source_row_count,
	uint32_t multiprocessor_count,
	uint32_t tp_degree,
	uint32_t tp_rank)
{
	uint64_t rows_per_expert;
	uint64_t payload_stride,scale_stride;
	uint32_t experts_per_rank = SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT / tp_degree;
	const uint8_t *payload;
	const uint8_t *scale;
	const uint32_t *offsets;
	const uint32_t *prefix;
	if ( view == 0 || input_bf16 == 0 ||
		group_row_offset == 0 || group_tile_prefix == 0 || output_bf16 == 0 ||
		view->weight_format != SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 ||
		view->output_dimension % SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT != 0u ||
		view->weight_payload == 0 || view->weight_scale_e8m0 == 0 ||
		(source_row_map == 0 && source_row_count == 0u) ||
		tp_degree == 0u || tp_rank >= tp_degree ||
		(SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT % tp_degree) != 0u )
		return(cudaErrorInvalidValue);
	rows_per_expert = (uint64_t)view->output_dimension / SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT;
	payload_stride = rows_per_expert * view->input_dimension;
	scale_stride = (rows_per_expert / 128u) * ((uint64_t)view->input_dimension / 128u) * 4u;
	/* Expert shard: the pack stores experts expert-major, so the rank's
	 * slice is one contiguous pointer shift over payload, scales, offsets
	 * and tile prefixes; all row/output indexing stays global. */
	payload = (const uint8_t *)view->weight_payload + ((uint64_t)tp_rank * experts_per_rank * payload_stride);
	scale = (const uint8_t *)view->weight_scale_e8m0 + ((uint64_t)tp_rank * experts_per_rank * scale_stride);
	offsets = group_row_offset + ((uint64_t)tp_rank * experts_per_rank);
	prefix = group_tile_prefix + ((uint64_t)tp_rank * experts_per_rank);
	return(SparkLmHostLaunchGroupedScalarLinear<32u>(stream,
		SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128,
		payload,scale,
		payload_stride,scale_stride,
		input_bf16,source_row_map,source_row_count,offsets,prefix,
		output_bf16,experts_per_rank,
		view->input_dimension,rows_per_expert,multiprocessor_count));
}

/*
 * Grouped FP8 expert linear on the TENSOR-CORE tile path:
 * SparkLmExpertTileAllKernel spans the 512 experts on gridDim.z, decodes
 * FP8_E4M3 block-128 weights to BF16 fragments (the vendor scale layout,
 * verbatim) and consumes them through wmma mma_sync. This is the same
 * kernel the dense batched path uses; only the grouped host launcher was
 * missing. Used at rows >= 16, where every M-tile is full - below that the
 * M=16 padding re-reads each expert's weight tile sixteen times and the
 * scalar grouped path wins.
 *
 * source_row_map is the route's packed->source token table for w1/w3
 * (indirect A reads), or 0 for w2 whose input is already expert-major
 * packed (identity). group_row_offset is the per-expert packed range.
 */
extern "C" cudaError_t SparkQwen38LaunchGroupedExpertTileLinear(
	cudaStream_t stream,
	const SparkQwen38MaxLinearView *view,
	const void *input_bf16,
	const uint32_t *source_row_map,
	const uint32_t *group_row_offset,
	void *output_bf16,
	uint32_t source_row_count,
	uint32_t tp_degree,
	uint32_t tp_rank)
{
	uint64_t rows_per_expert;
	uint64_t payload_stride,scale_stride;
	uint32_t m_blocks,n_tiles,experts_per_rank;
	const uint8_t *payload;
	const uint8_t *scale;
	const uint32_t *offsets;
	if ( view == 0 || input_bf16 == 0 || group_row_offset == 0 || output_bf16 == 0 ||
		view->weight_format != SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 ||
		view->output_dimension % SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT != 0u ||
		view->input_dimension % 64u != 0u ||
		view->weight_payload == 0 || view->weight_scale_e8m0 == 0 ||
		source_row_count == 0u ||
		tp_degree == 0u || tp_rank >= tp_degree ||
		(SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT % tp_degree) != 0u )
		return(cudaErrorInvalidValue);
	rows_per_expert = (uint64_t)view->output_dimension / SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT;
	if ( rows_per_expert % SPARK_LM_TILE_N != 0u )
		return(cudaErrorInvalidValue);
	payload_stride = rows_per_expert * view->input_dimension;
	scale_stride = (rows_per_expert / 128u) * ((uint64_t)view->input_dimension / 128u) * 4u;
	m_blocks = (source_row_count + SPARK_LM_TILE - 1u) / SPARK_LM_TILE;
	n_tiles = (uint32_t)(rows_per_expert / SPARK_LM_TILE_N);
	(void)m_blocks;
	(void)n_tiles;
	experts_per_rank = SPARK_QWEN38_MAX_MODEL_ROUTED_EXPERT_COUNT / tp_degree;
	payload = (const uint8_t *)view->weight_payload + ((uint64_t)tp_rank * experts_per_rank * payload_stride);
	scale = (const uint8_t *)view->weight_scale_e8m0 + ((uint64_t)tp_rank * experts_per_rank * scale_stride);
	offsets = group_row_offset + ((uint64_t)tp_rank * experts_per_rank);
	/* The expert-stride m-loop at every tile shape: one launch of
	 * 64 x n_tiles CTAs, each walking several experts - empty experts
	 * cost a few cycles instead of a launch, and each k-strip is staged
	 * once per 8-m-tile chunk (the measured B=256 collapse fix). The
	 * expert shard is the pointer shift above; indexing stays global. */
	return(SparkLmHostLaunchGroupedExpertTileMloop(
		stream,
		SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128,
		payload,scale,
		payload_stride,scale_stride,
		input_bf16,source_row_map,offsets,output_bf16,
		view->input_dimension,(uint32_t)rows_per_expert,
		experts_per_rank));
}
