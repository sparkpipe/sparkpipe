#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "sparkpipe/spark_qwen36_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_lm_kernels.cuh"
#include "spark_qwen36_dspark_cuda.cuh"

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

#define SPARK_QWEN36_CUDA_DK SPARK_QWEN36_MODEL_GDN_HEAD_KEY_DIMENSION
#define SPARK_QWEN36_CUDA_DV SPARK_QWEN36_MODEL_GDN_HEAD_VALUE_DIMENSION
#define SPARK_QWEN36_CUDA_GVA_GROUP (SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT / SPARK_QWEN36_MODEL_GDN_KEY_HEAD_COUNT)
#define SPARK_QWEN36_CUDA_ATTN_GROUP (SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT / SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT)
#define SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA 2u
#define SPARK_QWEN36_CUDA_ATTN_VALUE_PAIRS_PER_LANE \
    (SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION / (2u * SPARK_LM_WARP_LANES))
#define SPARK_QWEN36_CUDA_GDN_STATE_ELEMENTS \
    (SPARK_QWEN36_CUDA_DK * SPARK_QWEN36_CUDA_DV)
#define SPARK_QWEN36_CUDA_GDN_DECODE_SHARED_BYTES \
    (SPARK_QWEN36_CUDA_GDN_STATE_ELEMENTS * sizeof(float))

/* Tensor-parallel per-rank geometry. Degree 1 uses the full-model values;
 * a TP pack writes its rank's shard dims once at initialize. Kernels read
 * the DEVICE table and launchers read the HOST mirror, so one binary serves
 * every TP degree without per-kernel signature churn. */
enum SparkQwen36TpDim
{
	SPARK_QWEN36_TPD_GDN_QK_CHANNELS = 0,
	SPARK_QWEN36_TPD_GDN_VALUE_CHANNELS,
	SPARK_QWEN36_TPD_GDN_CONV_CHANNELS,
	SPARK_QWEN36_TPD_GDN_KEY_HEADS,
	SPARK_QWEN36_TPD_GDN_VALUE_HEADS,
	SPARK_QWEN36_TPD_ATTN_QUERY_HEADS,
	SPARK_QWEN36_TPD_ATTN_KV_HEADS,
	SPARK_QWEN36_TPD_GDN_QK_CHANNEL_BASE,
	SPARK_QWEN36_TPD_GDN_VALUE_CHANNEL_BASE,
	SPARK_QWEN36_TPD_GDN_KEY_HEAD_BASE,
	SPARK_QWEN36_TPD_GDN_VALUE_HEAD_BASE,
	SPARK_QWEN36_TPD_COUNT
};

__device__ uint32_t spark_qwen36_tp_dim[SPARK_QWEN36_TPD_COUNT] =
{
	SPARK_QWEN36_MODEL_GDN_QK_DIMENSION,
	SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION,
	SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS,
	SPARK_QWEN36_MODEL_GDN_KEY_HEAD_COUNT,
	SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT,
	SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT,
	SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT,
	0u, 0u, 0u, 0u
};

static uint32_t spark_qwen36_tp_host_dim[SPARK_QWEN36_TPD_COUNT] =
{
	SPARK_QWEN36_MODEL_GDN_QK_DIMENSION,
	SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION,
	SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS,
	SPARK_QWEN36_MODEL_GDN_KEY_HEAD_COUNT,
	SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT,
	SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT,
	SPARK_QWEN36_MODEL_ATTN_KV_HEAD_COUNT,
	0u, 0u, 0u, 0u
};

static __host__ __device__ __forceinline__ uint32_t SparkQwen36TpDim(uint32_t index)
{
#ifdef __CUDA_ARCH__
	return(spark_qwen36_tp_dim[index]);
#else
	return(spark_qwen36_tp_host_dim[index]);
#endif
}

/* The base offsets are rank-derived in the packer/module; they index the
 * REPLICATED tensors (conv weights, beta/decay, A_log, dt_bias) from the
 * rank's shard coordinates. */

/* Map a rank-local conv channel (stitched q|k|v shard layout) to its row
 * in the REPLICATED full-width conv weight. Degree 1 is the identity. */
static __host__ __device__ __forceinline__ uint32_t SparkQwen36TpConvChannel(uint32_t channel)
{
	uint32_t qk = SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_QK_CHANNELS);
	if ( channel < qk )
		return(SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_QK_CHANNEL_BASE) + channel);
	if ( channel < 2u * qk )
		return(SPARK_QWEN36_MODEL_GDN_QK_DIMENSION +
			SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_QK_CHANNEL_BASE) +
			(channel - qk));
	return((2u * SPARK_QWEN36_MODEL_GDN_QK_DIMENSION) +
		SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_CHANNEL_BASE) +
		(channel - (2u * qk)));
}
extern "C" cudaError_t SparkQwen36TpSetGeometry(
	uint32_t gdn_qk_channels,uint32_t gdn_value_channels,
	uint32_t gdn_conv_channels,uint32_t gdn_key_heads,
	uint32_t gdn_value_heads,uint32_t attn_query_heads,
	uint32_t attn_kv_heads,
	uint32_t gdn_qk_channel_base,uint32_t gdn_value_channel_base,
	uint32_t gdn_key_head_base,uint32_t gdn_value_head_base)
{
	uint32_t values[SPARK_QWEN36_TPD_COUNT] =
	{
		gdn_qk_channels, gdn_value_channels, gdn_conv_channels,
		gdn_key_heads, gdn_value_heads, attn_query_heads, attn_kv_heads,
		gdn_qk_channel_base, gdn_value_channel_base,
		gdn_key_head_base, gdn_value_head_base
	};
	uint32_t index;
	cudaError_t error;
	for (index = 0u; index < SPARK_QWEN36_TPD_COUNT; index++)
		spark_qwen36_tp_host_dim[index] = values[index];
	error = cudaMemcpyToSymbol(spark_qwen36_tp_dim,values,sizeof(values),0u,
		cudaMemcpyHostToDevice);
	return(error);
}

static __device__ __forceinline__ float SparkQwen36RopeFrequency(uint32_t pair)
{
	return(exp2f(-((float)(2u * pair) / (float)SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION) * log2f((float)SPARK_QWEN36_MODEL_ATTN_ROPE_THETA)));
}

// Depthwise causal conv update for one decode token per row: one thread per
// (row, channel), window = carried tail (3) plus the fresh projection, dot
// with the 4-tap weight, silu, then rotate the tail in place. Cold rows read
// a zero tail. Matches causal_conv1d_update with bias absent.
static __global__ void SparkQwen36ConvUpdateKernel(const void *qkv_bf16, const void *conv_weight_bf16, void *conv_out_bf16, void *conv_tail_bf16, const uint32_t *row_lane_indices, const uint32_t *state_cold_by_row, uint32_t row_count, uint32_t gdn_layer_ordinal, uint64_t tail_lane_stride, uint64_t tail_layer_stride)
{
	uint32_t row,channel = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t tail_base;
	float window[4],accumulator;
	uint32_t tap;
	if ( channel >= SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_CONV_CHANNELS) )
		return;
	/* Serialize the rows: the conv tail is the sliding recurrence; parallel
	 * row blocks raced its read-modify-write (same class as the GDN step
	 * race; the DSV4 session's fix). */
	for (row = 0u; row < row_count; row++)
	{
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
	window[3] = SparkLmBf16ToFloat(qkv_bf16,((uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_CONV_CHANNELS)) + channel);
	accumulator = 0.0f;
	for (tap = 0; tap < SPARK_QWEN36_MODEL_GDN_CONV_KERNEL; tap++)
		accumulator += (window[tap] * SparkLmBf16ToFloat(conv_weight_bf16,((uint64_t)SparkQwen36TpConvChannel(channel) * SPARK_QWEN36_MODEL_GDN_CONV_KERNEL) + tap));
	SparkLmFloatToBf16(conv_out_bf16,((uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_CONV_CHANNELS)) + channel,SparkLmSwish(accumulator));
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 0u,window[1]);
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 1u,window[2]);
	SparkLmFloatToBf16(conv_tail_bf16,tail_base + 2u,window[3]);
	__syncthreads();
	}
}

// Per-head log decay and beta from the two 48-row projections plus the fp32
// decay parameters: g = -exp(a_log) * softplus(a + dt_bias), beta =
// sigmoid(b). One thread per (row, value head).
static __global__ void SparkQwen36DecayBetaKernel(const void *decay_pre_bf16, const void *beta_pre_bf16, const float *a_log_f32, const float *dt_bias_f32, float *log_decay_f32, float *beta_f32, uint32_t row_count)
{
	uint32_t row = blockIdx.x,head = threadIdx.x;
	uint64_t local_index,replicated_index,full_head;
	if ( row >= row_count || head >= SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS) )
		return;
	/* The decay/beta pre-activations and the fp32 head parameters are
	 * REPLICATED (every rank holds all 48 heads) while the log-decay and
	 * beta outputs are sharded to this rank's value-head window. */
	full_head = (uint64_t)head + SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEAD_BASE);
	local_index = ((uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS)) + head;
	replicated_index = ((uint64_t)row * SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT) + full_head;
	log_decay_f32[local_index] = -expf(a_log_f32[full_head]) * SparkLmSoftplus(SparkLmBf16ToFloat(decay_pre_bf16,replicated_index) + dt_bias_f32[full_head]);
	beta_f32[local_index] = SparkLmSigmoid(SparkLmBf16ToFloat(beta_pre_bf16,replicated_index));
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
static __global__ void SparkQwen36GdnStepKernel(const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *state_f32, void *core_out_bf16, const uint32_t *row_lane_indices, const uint32_t *state_cold_by_row, uint32_t row_count, uint32_t gdn_layer_ordinal, uint64_t state_lane_stride, uint64_t state_layer_stride)
{
    extern __shared__ float state_shared[];
    __shared__ float qn[SPARK_QWEN36_CUDA_DK];
    __shared__ float kn[SPARK_QWEN36_CUDA_DK];
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

    head = blockIdx.x;
    column = threadIdx.x;
    key_head = head / SPARK_QWEN36_CUDA_GVA_GROUP;
    /* Rows serialized inside each head block: the state_f32 read-modify-write
     * is the recurrence and parallel row blocks race it (last writer wins per
     * element - one row's accumulation lost per multi-row frame; the DSV4
     * session's silent-divergence fix). k-row frame == k sequential frames. */
    for (row = 0u; row < row_count; row++)
    {

    conv_row = (uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_CONV_CHANNELS);
    value = SparkLmBf16ToFloat(
        conv_out_bf16,
        conv_row + ((uint64_t)key_head * SPARK_QWEN36_CUDA_DK) + column);
    q_norm = SparkLmBlockReduceSum(value * value, reduce_scratch);
    qn[column] = value * rsqrtf(q_norm + 1.0e-6f) *
        rsqrtf((float)SPARK_QWEN36_CUDA_DK);

    value = SparkLmBf16ToFloat(
        conv_out_bf16,
        conv_row + SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_QK_CHANNELS) +
            ((uint64_t)key_head * SPARK_QWEN36_CUDA_DK) + column);
    k_norm = SparkLmBlockReduceSum(value * value, reduce_scratch);
    kn[column] = value * rsqrtf(k_norm + 1.0e-6f);
    __syncthreads();

    state_base =
        ((uint64_t)row_lane_indices[row] * state_lane_stride) +
        ((uint64_t)gdn_layer_ordinal * state_layer_stride) +
        ((uint64_t)head * SPARK_QWEN36_CUDA_GDN_STATE_ELEMENTS);
    decay = state_cold_by_row[row] != 0u
        ? 0.0f
        : expf(log_decay_f32[
            ((uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS)) + head]);
    beta = beta_f32[
        ((uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS)) + head];

    kv_memory = 0.0f;
    for (element = 0u; element < SPARK_QWEN36_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN36_CUDA_DV) + column;
        value = state_cold_by_row[row] != 0u
            ? 0.0f
            : state_f32[state_base + state_index] * decay;
        state_shared[state_index] = value;
        kv_memory = fmaf(value, kn[element], kv_memory);
    }

    delta = (SparkLmBf16ToFloat(
        conv_out_bf16,
        conv_row + (2u * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_QK_CHANNELS)) +
            ((uint64_t)head * SPARK_QWEN36_CUDA_DV) + column) - kv_memory) * beta;
    output = 0.0f;
    for (element = 0u; element < SPARK_QWEN36_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN36_CUDA_DV) + column;
        value = fmaf(kn[element], delta, state_shared[state_index]);
        state_shared[state_index] = value;
        output = fmaf(value, qn[element], output);
    }
    SparkLmFloatToBf16(
        core_out_bf16,
        ((uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_CHANNELS)) +
            ((uint64_t)head * SPARK_QWEN36_CUDA_DV) + column,
        output);

    for (element = 0u; element < SPARK_QWEN36_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN36_CUDA_DV) + column;
        state_f32[state_base + state_index] = state_shared[state_index];
    }
    __syncthreads();
    }
}

// Gated head norm: fp32 RMSNorm over one value head, times weight, times
// silu(z). One block per (row, head), 128 threads. Norm before gate. The
// core and output are this rank's value-channel shard, but the gate
// projection z is REPLICATED (full 6144 channels on every rank), so it is
// read at the rank's full-model head offset.
static __global__ void SparkQwen36GatedNormKernel(const void *core_bf16, const void *z_bf16, const void *norm_weight_bf16, void *output_bf16, uint32_t row_count, float epsilon)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.y,head = blockIdx.x,column = threadIdx.x;
	uint64_t index = ((uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_CHANNELS)) + ((uint64_t)head * SPARK_QWEN36_CUDA_DV) + column;
	uint64_t z_index = ((uint64_t)row * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION) + ((uint64_t)(head + SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEAD_BASE)) * SPARK_QWEN36_CUDA_DV) + column;
	float value,variance;
	if ( row >= row_count )
		return;
	value = SparkLmBf16ToFloat(core_bf16,index);
	variance = SparkLmBlockReduceSum(value * value,reduce_scratch) / (float)SPARK_QWEN36_CUDA_DV;
	value = value * rsqrtf(variance + epsilon) * SparkLmBf16ToFloat(norm_weight_bf16,column) * SparkLmSwish(SparkLmBf16ToFloat(z_bf16,z_index));
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
static __global__ void SparkQwen36AttnPrepareKernel(
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
    float epsilon)
{
    __shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
    __shared__ float query_shared[SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION];
    __shared__ float key_shared[SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION];
    __shared__ float rope_cosine[
        SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u];
    __shared__ float rope_sine[
        SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u];
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
    head = blockIdx.x;
    column = threadIdx.x;
    kv_head = head / SPARK_QWEN36_CUDA_ATTN_GROUP;
    if (row >= row_count ||
        head >= SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_QUERY_HEADS) ||
        column >= SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION)
    {
        return;
    }

    query_base =
        ((uint64_t)row * 2u * SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_QUERY_HEADS) * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
        ((uint64_t)head * 2u * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION);
    value = SparkLmBf16ToFloat(q_fused_bf16, query_base + column);
    sum_squares = SparkLmBlockReduceSum(
        value * value,
        reduce_scratch);
    inverse_rms = rsqrtf(
        sum_squares / (float)SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION +
        epsilon);
    query_shared[column] =
        value * inverse_rms *
        SparkLmBf16ToFloat(q_norm_weight_bf16, column);
    if (column < SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u)
    {
        float angle;

        pair = column;
        angle =
            (float)row_positions[row] * SparkQwen36RopeFrequency(pair);
        sincosf(angle, &rope_sine[pair], &rope_cosine[pair]);
    }
    __syncthreads();

    if (column < SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u)
    {
        float real;
        float imaginary;

        pair = column;
        real = query_shared[pair];
        imaginary = query_shared[
            pair + SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u];
        query_shared[pair] =
            real * rope_cosine[pair] - imaginary * rope_sine[pair];
        query_shared[
            pair + SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u] =
            imaginary * rope_cosine[pair] + real * rope_sine[pair];
    }
    __syncthreads();
    SparkLmFloatToBf16(
        q_fused_bf16,
        query_base + column,
        query_shared[column]);

    if ((head % SPARK_QWEN36_CUDA_ATTN_GROUP) != 0u)
    {
        return;
    }

    key_base =
        ((uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_KV_HEADS) * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
        ((uint64_t)kv_head * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION);
    value = SparkLmBf16ToFloat(k_bf16, key_base + column);
    sum_squares = SparkLmBlockReduceSum(
        value * value,
        reduce_scratch);
    inverse_rms = rsqrtf(
        sum_squares / (float)SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION +
        epsilon);
    key_shared[column] =
        value * inverse_rms *
        SparkLmBf16ToFloat(k_norm_weight_bf16, column);
    __syncthreads();

    if (column < SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u)
    {
        float real;
        float imaginary;

        pair = column;
        real = key_shared[pair];
        imaginary = key_shared[
            pair + SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u];
        key_shared[pair] =
            real * rope_cosine[pair] - imaginary * rope_sine[pair];
        key_shared[
            pair + SPARK_QWEN36_MODEL_ATTN_ROPE_DIMENSION / 2u] =
            imaginary * rope_cosine[pair] + real * rope_sine[pair];
    }
    __syncthreads();

    slot = slot_mapping[row];
    block = slot / SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    offset = slot % SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    cache_base =
        ((uint64_t)block * cache_block_stride) +
        ((uint64_t)attn_layer_ordinal * cache_layer_stride) +
        ((uint64_t)offset * 2u * SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_KV_HEADS) * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
        ((uint64_t)kv_head * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION);
    SparkLmFloatToBf16(
        kv_cache_bf16,
        cache_base + column,
        key_shared[column]);
    SparkLmFloatToBf16(
        kv_cache_bf16,
        cache_base + (SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_KV_HEADS) * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) + column,
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
static __device__ __forceinline__ uint64_t SparkQwen36AttnTokenBase(const uint32_t *block_indices, uint64_t lane_base, uint32_t token, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t kv_head)
{
	uint32_t block = __ldg(block_indices + lane_base + (token / SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS));
	return(((uint64_t)block * cache_block_stride) + ((uint64_t)attn_layer_ordinal * cache_layer_stride) + ((uint64_t)(token % SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) * 2u * SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_KV_HEADS) * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) + ((uint64_t)kv_head * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION));
}

// Cross-warp merge with the fused sigmoid gate applied at the store.


// One token's logit: lanes pair-load the cached key against the shared
// query, warp-reduce, fixed 1/sqrt(128) scale.


static __global__ void SparkQwen36AttnDecodeKernel(const void *q_fused_bf16, const void *kv_cache_bf16, const uint32_t *block_indices, const uint32_t *block_counts, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t lane_stride, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride)
{
    const uint32_t heads_per_cta = SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA;
    __shared__ float q_shared[
        SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA *
        SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION];
    __shared__ float merge_max[
        SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA * SPARK_LM_CTA_WARPS];
    __shared__ float merge_den[
        SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA * SPARK_LM_CTA_WARPS];
    __shared__ float merge_scale[
        SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA * SPARK_LM_CTA_WARPS];
    __shared__ float inverse_denominator[
        SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA];
    __shared__ float merge_acc[
        SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA * SPARK_LM_CTA_WARPS *
        SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION];
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
    float running_max[SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA];
    float running_den[SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA];
    float local_logit[SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA];
    float rescale[SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA];
    float weight[SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA];
    float2 accumulator[
        SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA]
        [SPARK_QWEN36_CUDA_ATTN_VALUE_PAIRS_PER_LANE];
    float2 key_pair;
    float2 value_pair;
    float head_max;
    float denominator;
    float merged;
    float gate;

    static_assert(
        SPARK_QWEN36_CUDA_ATTN_GROUP %
                SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA ==
            0u,
        "grouped attention CTAs must not cross KV-head ownership");
    row = blockIdx.y;
    head_base = blockIdx.x * heads_per_cta;
    kv_head = head_base / SPARK_QWEN36_CUDA_ATTN_GROUP;
    warp = threadIdx.x / SPARK_LM_WARP_LANES;
    lane = threadIdx.x % SPARK_LM_WARP_LANES;
    if (row >= row_count ||
        head_base + heads_per_cta >
            SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_QUERY_HEADS))
    {
        return;
    }
    lane_index = row_lane_indices[row];
    context = context_lengths[row];
    available_block_count = __ldg(block_counts + lane_index);
    required_block_count =
        (context + SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) /
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    if (context == 0u ||
        available_block_count > lane_stride ||
        required_block_count > available_block_count)
    {
        float invalid_output;

        invalid_output = __int_as_float(0x7fc00000);
        for (element = threadIdx.x;
             element < heads_per_cta * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION;
             element += blockDim.x)
        {
            local_head = element / SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION;
            partial = element % SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION;
            out_base =
                ((uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_QUERY_HEADS) * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
                ((uint64_t)(head_base + local_head) *
                    SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION);
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
            ((uint64_t)row * 2u * SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_QUERY_HEADS) * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
            ((uint64_t)head * 2u * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION);
        for (element = threadIdx.x;
             element < SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION;
             element += blockDim.x)
        {
            q_shared[
                (local_head * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
                element] = SparkLmBf16ToFloat(
                    q_fused_bf16,
                    q_base + element);
        }
        running_max[local_head] = -3.0e38f;
        running_den[local_head] = 0.0f;
        #pragma unroll
        for (pair = 0u;
             pair < SPARK_QWEN36_CUDA_ATTN_VALUE_PAIRS_PER_LANE;
             ++pair)
        {
            accumulator[local_head][pair] = make_float2(0.0f, 0.0f);
        }
    }
    for (element = threadIdx.x;
         element < SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA *
             SPARK_LM_CTA_WARPS *
             SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION;
         element += blockDim.x)
    {
        merge_acc[element] = 0.0f;
    }
    __syncthreads();

    lane_base = (uint64_t)lane_index * lane_stride;
    for (token = warp; token < context; token += SPARK_LM_CTA_WARPS)
    {
        token_base = SparkQwen36AttnTokenBase(
            block_indices,
            lane_base,
            token,
            attn_layer_ordinal,
            cache_layer_stride,
            cache_block_stride,
            kv_head);
        #pragma unroll
        for (local_head = 0u; local_head < heads_per_cta; ++local_head)
        {
            local_logit[local_head] = 0.0f;
        }
        #pragma unroll
        for (pair = 0u;
             pair < SPARK_QWEN36_CUDA_ATTN_VALUE_PAIRS_PER_LANE;
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
                            SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
                        element],
                    key_pair.x,
                    local_logit[local_head]);
                local_logit[local_head] = fmaf(
                    q_shared[
                        (local_head *
                            SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
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
             pair < SPARK_QWEN36_CUDA_ATTN_VALUE_PAIRS_PER_LANE;
             ++pair)
        {
            value_pair = SparkLmLoadBf16Pair(
                kv_cache_bf16,
                ((token_base + (SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_KV_HEADS) * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION)) >> 1u) +
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
             pair < SPARK_QWEN36_CUDA_ATTN_VALUE_PAIRS_PER_LANE;
             ++pair)
        {
            element = ((pair * SPARK_LM_WARP_LANES) + lane) << 1u;
            merge_acc[
                (((local_head * SPARK_LM_CTA_WARPS) + warp) *
                    SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
                element] = accumulator[local_head][pair].x;
            merge_acc[
                (((local_head * SPARK_LM_CTA_WARPS) + warp) *
                    SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
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
            ((uint64_t)row * 2u * SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_QUERY_HEADS) * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
            ((uint64_t)head * 2u * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION);
        out_base =
            ((uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_QUERY_HEADS) * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
            ((uint64_t)head * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION);
        for (element = threadIdx.x;
             element < SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION;
             element += blockDim.x)
        {
            merged = 0.0f;
            for (partial = 0u; partial < SPARK_LM_CTA_WARPS; ++partial)
            {
                merged = fmaf(
                    merge_acc[
                        (((local_head * SPARK_LM_CTA_WARPS) + partial) *
                            SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION) +
                        element],
                    merge_scale[
                        (local_head * SPARK_LM_CTA_WARPS) + partial],
                    merged);
            }
            gate = SparkLmSigmoid(SparkLmBf16ToFloat(
                q_fused_bf16,
                q_base + SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION + element));
            SparkLmFloatToBf16(
                head_out_bf16,
                out_base + element,
                merged * inverse_denominator[local_head] * gate);
        }
    }
}

// Embedding gather: one thread per (row, element); token ids are validated
// against the vocabulary on the host before upload, so the kernel trusts.
static __global__ void SparkQwen36EmbeddingGatherKernel(const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t row = (uint32_t)(index / SPARK_QWEN36_MODEL_HIDDEN_DIMENSION),element = (uint32_t)(index % SPARK_QWEN36_MODEL_HIDDEN_DIMENSION);
	if ( row >= row_count )
		return;
	SparkLmFloatToBf16(hidden_bf16,index,SparkLmBf16ToFloat(embedding_bf16,((uint64_t)token_ids[row] * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION) + element));
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
#define SPARK_QWEN36_CUDA_CHUNK SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS
#define SPARK_QWEN36_CUDA_GDN_QK_SHARED_BYTES \
    (2u * SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_DK * sizeof(float))
#define SPARK_QWEN36_CUDA_GDN_CHUNK_SHARED_BYTES \
    ((SPARK_QWEN36_CUDA_GDN_STATE_ELEMENTS + \
      (SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_DV) + \
      (2u * SPARK_QWEN36_CUDA_CHUNK)) * sizeof(float))
static_assert(
    SPARK_QWEN36_CUDA_GDN_CHUNK_SHARED_BYTES == 98816u,
    "Qwen GDN chunk shared layout must fit the SM 12.x 99-KB block limit");

typedef struct SparkQwen36ChunkWorkspaceView
{
	float *qn;
	float *kn;
	float *cum_g;
	float *decay;
	float *attn;
	float *w;
	float *kg;
} SparkQwen36ChunkWorkspaceView;

static __device__ __forceinline__ uint64_t SparkQwen36ChunkHeadOffset(uint32_t head, uint32_t per_head_elements)
{
	return((uint64_t)head * per_head_elements);
}

// Stage 1: per-head L2 norms with the 1/sqrt(dk) query scale, the intra-
// chunk decay cumsum, the decay mask and the strictly-lower beta-scaled
// -k_beta k^T attention seed. Block per head, thread per token row.
static __global__ void SparkQwen36ChunkPrepareKernel(const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, SparkQwen36ChunkWorkspaceView views, uint32_t token_count)
{
	uint32_t head = blockIdx.x,row = threadIdx.x,key_head = head / SPARK_QWEN36_CUDA_GVA_GROUP,element,column;
	uint64_t conv_row,qk_base = SparkQwen36ChunkHeadOffset(head,SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_DK);
	uint64_t mat_base = SparkQwen36ChunkHeadOffset(head,SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_CHUNK);
	float total,value,product;
	if ( row >= token_count )
		return;
	conv_row = (uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_CONV_CHANNELS);
	total = 0.0f;
	for (element = 0; element < SPARK_QWEN36_CUDA_DK; element++)
	{
		value = SparkLmBf16ToFloat(conv_out_bf16,conv_row + ((uint64_t)key_head * SPARK_QWEN36_CUDA_DK) + element);
		total += (value * value);
	}
	total = rsqrtf(total + 1e-6f) * rsqrtf((float)SPARK_QWEN36_CUDA_DK);
	for (element = 0; element < SPARK_QWEN36_CUDA_DK; element++)
		views.qn[qk_base + ((uint64_t)row * SPARK_QWEN36_CUDA_DK) + element] = SparkLmBf16ToFloat(conv_out_bf16,conv_row + ((uint64_t)key_head * SPARK_QWEN36_CUDA_DK) + element) * total;
	total = 0.0f;
	for (element = 0; element < SPARK_QWEN36_CUDA_DK; element++)
	{
		value = SparkLmBf16ToFloat(conv_out_bf16,conv_row + SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_QK_CHANNELS) + ((uint64_t)key_head * SPARK_QWEN36_CUDA_DK) + element);
		total += (value * value);
	}
	total = rsqrtf(total + 1e-6f);
	for (element = 0; element < SPARK_QWEN36_CUDA_DK; element++)
		views.kn[qk_base + ((uint64_t)row * SPARK_QWEN36_CUDA_DK) + element] = SparkLmBf16ToFloat(conv_out_bf16,conv_row + SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_QK_CHANNELS) + ((uint64_t)key_head * SPARK_QWEN36_CUDA_DK) + element) * total;
	if ( row == 0u )
	{
		total = 0.0f;
		for (element = 0; element < token_count; element++)
		{
			total += log_decay_f32[((uint64_t)element * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS)) + head];
			views.cum_g[SparkQwen36ChunkHeadOffset(head,SPARK_QWEN36_CUDA_CHUNK) + element] = total;
		}
	}
	__syncthreads();
	for (column = 0; column < token_count; column++)
	{
		value = column <= row ? __expf(views.cum_g[SparkQwen36ChunkHeadOffset(head,SPARK_QWEN36_CUDA_CHUNK) + row] - views.cum_g[SparkQwen36ChunkHeadOffset(head,SPARK_QWEN36_CUDA_CHUNK) + column]) : 0.0f;
		views.decay[mat_base + ((uint64_t)row * SPARK_QWEN36_CUDA_CHUNK) + column] = value;
		product = 0.0f;
		for (element = 0; element < SPARK_QWEN36_CUDA_DK && column < row; element++)
			product += (views.kn[qk_base + ((uint64_t)row * SPARK_QWEN36_CUDA_DK) + element] * beta_f32[((uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS)) + head] * views.kn[qk_base + ((uint64_t)column * SPARK_QWEN36_CUDA_DK) + element]);
		views.attn[mat_base + ((uint64_t)row * SPARK_QWEN36_CUDA_CHUNK) + column] = column < row ? -(product * value) : 0.0f;
	}
}

// Stage 2: the forward-substitution UT transform T = (I - A)^-1, in place.
// The row recurrence is sequential; columns of a row are parallel. Block
// per head, thread per column.
static __global__ void SparkQwen36ChunkSolveKernel(SparkQwen36ChunkWorkspaceView views, uint32_t token_count)
{
	uint32_t head = blockIdx.x,column = threadIdx.x,row,element;
	uint64_t mat_base = SparkQwen36ChunkHeadOffset(head,SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_CHUNK);
	float accumulator;
	for (row = 1; row < token_count; row++)
	{
		if ( column < row )
		{
			accumulator = views.attn[mat_base + ((uint64_t)row * SPARK_QWEN36_CUDA_CHUNK) + column];
			for (element = 0; element < row; element++)
				accumulator += (views.attn[mat_base + ((uint64_t)row * SPARK_QWEN36_CUDA_CHUNK) + element] * views.attn[mat_base + ((uint64_t)element * SPARK_QWEN36_CUDA_CHUNK) + column]);
			views.attn[mat_base + ((uint64_t)row * SPARK_QWEN36_CUDA_CHUNK) + column] = accumulator;
		}
		__syncthreads();
	}
	if ( column < token_count )
		views.attn[mat_base + ((uint64_t)column * SPARK_QWEN36_CUDA_CHUNK) + column] += 1.0f;
}

// Stage 3: w = T (v o beta) and kg = T (k o beta o e^G). Block per (head,
// token row), thread per output column striped over dv then dk.
static __global__ void SparkQwen36ChunkTransformKernel(const void *conv_out_bf16, const float *beta_f32, SparkQwen36ChunkWorkspaceView views, uint32_t token_count)
{
	__shared__ float exp_cum_g[SPARK_QWEN36_CUDA_CHUNK];
	uint32_t head = blockIdx.x,row = blockIdx.y,column = threadIdx.x,element;
	uint64_t mat_base = SparkQwen36ChunkHeadOffset(head,SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_CHUNK);
	uint64_t vec_base = SparkQwen36ChunkHeadOffset(head,SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_DK);
	float accumulator,transform;
	if ( row >= token_count )
		return;
	if ( threadIdx.x < token_count )
		exp_cum_g[threadIdx.x] = __expf(
			views.cum_g[
				SparkQwen36ChunkHeadOffset(
					head,
					SPARK_QWEN36_CUDA_CHUNK) +
				threadIdx.x]);
	__syncthreads();
	accumulator = 0.0f;
	for (element = 0; element < token_count; element++)
	{
		transform = views.attn[mat_base + ((uint64_t)row * SPARK_QWEN36_CUDA_CHUNK) + element] * beta_f32[((uint64_t)element * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS)) + head];
		accumulator += (transform * SparkLmBf16ToFloat(conv_out_bf16,((uint64_t)element * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_CONV_CHANNELS)) + (2u * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_QK_CHANNELS)) + ((uint64_t)head * SPARK_QWEN36_CUDA_DV) + column));
	}
	views.w[vec_base + ((uint64_t)row * SPARK_QWEN36_CUDA_DV) + column] = accumulator;
	accumulator = 0.0f;
	for (element = 0; element < token_count; element++)
	{
		transform = views.attn[mat_base + ((uint64_t)row * SPARK_QWEN36_CUDA_CHUNK) + element] * beta_f32[((uint64_t)element * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS)) + head] * exp_cum_g[element];
		accumulator += (transform * views.kn[vec_base + ((uint64_t)element * SPARK_QWEN36_CUDA_DK) + column]);
	}
	views.kg[vec_base + ((uint64_t)row * SPARK_QWEN36_CUDA_DK) + column] = accumulator;
}


static __global__ void SparkQwen36ChunkQkDecayKernel(SparkQwen36ChunkWorkspaceView views, uint32_t token_count)
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
        (SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_DK);
    vector_base = SparkQwen36ChunkHeadOffset(
        head,
        SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_DK);
    matrix_base = SparkQwen36ChunkHeadOffset(
        head,
        SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_CHUNK);

    for (vector_element = threadIdx.x;
         vector_element < token_count * SPARK_QWEN36_CUDA_DK;
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
            for (element = 0u; element < SPARK_QWEN36_CUDA_DK; ++element)
            {
                dot = fmaf(
                    qn_shared[(row * SPARK_QWEN36_CUDA_DK) + element],
                    kn_shared[(column * SPARK_QWEN36_CUDA_DK) + element],
                    dot);
            }
            views.decay[
                matrix_base +
                ((uint64_t)row * SPARK_QWEN36_CUDA_CHUNK) + column] *= dot;
        }
        else
        {
            views.decay[
                matrix_base +
                ((uint64_t)row * SPARK_QWEN36_CUDA_CHUNK) + column] = 0.0f;
        }
    }
}

// Stage 4: v_new = w - kg S, out = (q o e^G) S + (q k^T o D) v_new, and the
// carried state S <- S e^G_last + (k o e^(G_last - G))^T v_new. Block per
// (head, state row is the thread's dk stripe? No): thread per dv column,
// mirroring the decode step's coalesced state-column ownership.
static __global__ void SparkQwen36ChunkStepKernel(const float *log_decay_f32, SparkQwen36ChunkWorkspaceView views, float *state_f32, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal, uint64_t state_lane_stride, uint64_t state_layer_stride)
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
    v_new_shared = state_shared + SPARK_QWEN36_CUDA_GDN_STATE_ELEMENTS;
    exp_cum_g_shared =
        v_new_shared + (SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_DV);
    carry_decay_shared =
        exp_cum_g_shared + SPARK_QWEN36_CUDA_CHUNK;
    vector_base = SparkQwen36ChunkHeadOffset(
        head,
        SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_DK);
    g_base = SparkQwen36ChunkHeadOffset(
        head,
        SPARK_QWEN36_CUDA_CHUNK);
    matrix_base = SparkQwen36ChunkHeadOffset(
        head,
        SPARK_QWEN36_CUDA_CHUNK * SPARK_QWEN36_CUDA_CHUNK);
    state_base =
        ((uint64_t)lane_index * state_lane_stride) +
        ((uint64_t)gdn_layer_ordinal * state_layer_stride) +
        ((uint64_t)head * SPARK_QWEN36_CUDA_GDN_STATE_ELEMENTS);
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

    for (element = 0u; element < SPARK_QWEN36_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN36_CUDA_DV) + column;
        state_shared[state_index] = state_f32[state_base + state_index];
    }

    for (row = 0u; row < token_count; ++row)
    {
        accumulator = 0.0f;
        for (element = 0u; element < SPARK_QWEN36_CUDA_DK; ++element)
        {
            accumulator = fmaf(
                views.kg[
                    vector_base +
                    ((uint64_t)row * SPARK_QWEN36_CUDA_DK) + element],
                state_shared[(element * SPARK_QWEN36_CUDA_DV) + column],
                accumulator);
        }
        v_new_shared[(row * SPARK_QWEN36_CUDA_DV) + column] =
            views.w[
                vector_base +
                ((uint64_t)row * SPARK_QWEN36_CUDA_DV) + column] - accumulator;
    }

    for (row = 0u; row < token_count; ++row)
    {
        accumulator = 0.0f;
        for (element = 0u; element < SPARK_QWEN36_CUDA_DK; ++element)
        {
            accumulator = fmaf(
                views.qn[
                    vector_base +
                    ((uint64_t)row * SPARK_QWEN36_CUDA_DK) + element],
                state_shared[(element * SPARK_QWEN36_CUDA_DV) + column],
                accumulator);
        }
        accumulator *= exp_cum_g_shared[row];
        for (element = 0u; element <= row; ++element)
        {
            accumulator = fmaf(
                views.decay[
                    matrix_base +
                    ((uint64_t)row * SPARK_QWEN36_CUDA_CHUNK) + element],
                v_new_shared[(element * SPARK_QWEN36_CUDA_DV) + column],
                accumulator);
        }
        SparkLmFloatToBf16(
            core_out_bf16,
            ((uint64_t)row * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_CHANNELS)) +
                ((uint64_t)head * SPARK_QWEN36_CUDA_DV) + column,
            accumulator);
    }

    for (element = 0u; element < SPARK_QWEN36_CUDA_DK; ++element)
    {
        state_index = (element * SPARK_QWEN36_CUDA_DV) + column;
        carry = state_shared[state_index] *
            exp_cum_g_shared[token_count - 1u];
        for (row = 0u; row < token_count; ++row)
        {
            carry = fmaf(
                views.kn[
                    vector_base +
                    ((uint64_t)row * SPARK_QWEN36_CUDA_DK) + element] *
                    carry_decay_shared[row],
                v_new_shared[(row * SPARK_QWEN36_CUDA_DV) + column],
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
static __global__ void SparkQwen36ChunkConvKernel(const void *qkv_bf16, const void *conv_weight_bf16, void *conv_out_bf16, void *conv_tail_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal, uint64_t tail_lane_stride, uint64_t tail_layer_stride)
{
	uint32_t channel = (blockIdx.x * blockDim.x) + threadIdx.x,token,tap;
	uint64_t tail_base,element;
	float window[4],weight[4],accumulator;
	if ( channel >= SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_CONV_CHANNELS) )
		return;
	tail_base = ((uint64_t)lane_index * tail_lane_stride) + ((uint64_t)gdn_layer_ordinal * tail_layer_stride) + ((uint64_t)channel * SPARK_QWEN36_MODEL_GDN_CONV_TAIL_COLUMNS);
	window[0] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 0u);
	window[1] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 1u);
	window[2] = SparkLmBf16ToFloat(conv_tail_bf16,tail_base + 2u);
	for (tap = 0; tap < SPARK_QWEN36_MODEL_GDN_CONV_KERNEL; tap++)
		weight[tap] = SparkLmBf16ToFloat(conv_weight_bf16,((uint64_t)SparkQwen36TpConvChannel(channel) * SPARK_QWEN36_MODEL_GDN_CONV_KERNEL) + tap);
	for (token = 0; token < token_count; token++)
	{
		element = ((uint64_t)token * SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_CONV_CHANNELS)) + channel;
		window[3] = SparkLmBf16ToFloat(qkv_bf16,element);
		accumulator = 0.0f;
		for (tap = 0; tap < SPARK_QWEN36_MODEL_GDN_CONV_KERNEL; tap++)
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
static __global__ void SparkQwen36ResidualAddKernel(void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
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

static __global__ void SparkQwen36SwiGluKernel(const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension)
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

extern "C" cudaError_t SparkQwen36ConfigureCudaKernels(void)
{
    cudaError_t status;

    status = cudaFuncSetAttribute(
        SparkQwen36GdnStepKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)SPARK_QWEN36_CUDA_GDN_DECODE_SHARED_BYTES);
    if (status != cudaSuccess)
    {
        return status;
    }
    status = cudaFuncSetAttribute(
        SparkQwen36ChunkQkDecayKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)SPARK_QWEN36_CUDA_GDN_QK_SHARED_BYTES);
    if (status != cudaSuccess)
    {
        return status;
    }
    status = cudaFuncSetAttribute(
        SparkQwen36ChunkStepKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)SPARK_QWEN36_CUDA_GDN_CHUNK_SHARED_BYTES);
    if (status != cudaSuccess)
    {
        return status;
    }
    /* The scalar Linear path stages the input row in dynamic shared memory,
     * input_dimension floats deep; the FFN down projection reads the
     * 17408-wide intermediate, which is past the 48KB static ceiling and
     * must be opted in like the GDN kernels above. The DSpark projector fc
     * consumes 5 taps x hidden (25600), wider still, so opt in to the max
     * of both. */
    return cudaFuncSetAttribute(
        (const void *)SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,SPARK_LM_CTA_WARPS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)(SPARK_QWEN36_MODEL_FFN_INTERMEDIATE_DIMENSION * sizeof(float)));
}

extern "C" cudaError_t SparkQwen36LaunchFusedResidualRmsNorm(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
    size_t shared_memory_bytes = (size_t)dimension * sizeof(float);

    SparkLmFusedResidualRmsNormKernel<<<row_count, SPARK_LM_CTA_THREADS, shared_memory_bytes, stream>>>(hidden_bf16, delta_bf16, gain_bf16, output_bf16, row_count, dimension, epsilon);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkQwen36LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
    size_t shared_memory_bytes = (size_t)dimension * sizeof(float);

    SparkLmRmsNormKernel<<<row_count, SPARK_LM_CTA_THREADS, shared_memory_bytes, stream>>>(input_bf16, gain_bf16, output_bf16, row_count, dimension, epsilon);
    return cudaGetLastError();
}
/*
 * Small-batch dense linear for the decode microbatch (B1..B8): weights
 * stream from HBM exactly once per projection instead of once per row (the
 * library scalar path re-reads the whole strip for every row).
 *
 * The arithmetic is BIT-IDENTICAL to the library dense scalar path
 * (SparkLmDotRowBf16 + SparkLmWarpReduceSum): each lane accumulates the
 * k-pairs at warp stride 32 in ascending order, the warp sums the lane
 * partials with the same shfl-down tree, and lane 0 rounds to bf16 with
 * __float2bfloat16. The 128-wide k-chunk only tiles the memory traffic;
 * because a chunk covers exactly 64 pairs the per-lane pair sequence
 * (l, l+32, l+64, ...) continues across chunk boundaries in the same order
 * as the library's single full-length pass, so every fp32 add happens in
 * the identical order and the outputs match the library bit for bit.
 *
 * Row-specialized so every warp does useful work:
 *  - rows == 1: lean kernel, no shared staging or barriers, 16 warps x 4
 *    neurons per 64-neuron tile, weights read straight from L2 once.
 *  - rows <= 2/4/8: tiled kernel, 64x128 weight chunk staged to shared and
 *    shared by all rows; warp w owns row (w & (ROWS-1)) and
 *    64 / (32 / ROWS) neurons. Row counts 3 and 5..7 ride the next
 *    power-of-two template with a store guard.
 */
#define SPARK_QWEN36_SMALL_BATCH_MAX_ROWS 8u
#define SPARK_QWEN36_SMALL_BATCH_TILE_N 64u
#define SPARK_QWEN36_SMALL_BATCH_K_CHUNK 128u

/* ------------------------------------------------------------------
 * rANS lossless weight-stream decode for the decode microbatch kernels.
 *
 * Compressed tensor payload (little endian, stored in 64x128 tile order):
 *   u32 ndirect
 *   u32 entries[3 * (ndirect + 1)]   (sym u16 | f u16 | C u32; the entry
 *                                     ndirect is the escape, sym = 0xFFFF)
 *   u32 id_bits
 *   u16 id_table[1 << id_bits]       (escape rid -> symbol value)
 *   u32 chunk_count                  (== (rows/64) * (cols/128))
 *   u32 chunk_offsets[chunk_count]   (relative to the payload base)
 *   chunks: u32 states[128], u16 lens[128], then byte-interleaved data
 *           (byte j of substream i at 768 + j*128 + i), 4B padded.
 *
 * Codec: canonical rANS per tensor, M = 4096 (l = 12), 8-bit renorm with
 * the 2^23 bound; 128 substreams x 64 values per 8192-value tile; the
 * escape is followed by 2 bytes of raw symbol id in the lane stream.
 * The arithmetic reproduces the original weight bits exactly, so the dots
 * remain bit-identical to the library scalar path.
 */
#define SPARK_QWEN36_RANS_SLOTS 4096u
#define SPARK_QWEN36_RANS_L 12u
#define SPARK_QWEN36_RANS_BOUND (1u << 23)
#define SPARK_QWEN36_RANS_SUBSTREAMS 128u
#define SPARK_QWEN36_RANS_SUB_LEN 64u
#define SPARK_QWEN36_RANS_HEADER_BYTES 768u
#define SPARK_QWEN36_RANS_STAGE_BYTES 13312u

static __device__ void SparkQwen36RansBuildTable(uint32_t *s_sf, uint16_t *s_c, const uint32_t *entries3, uint32_t ndirect)
{
	for ( uint32_t i = threadIdx.x; i <= ndirect; i += (uint32_t)blockDim.x )
	{
		const uint16_t sym = (uint16_t)entries3[3u * i];
		const uint16_t fs = (uint16_t)entries3[3u * i + 1u];
		const uint32_t cs = entries3[3u * i + 2u];
		const uint32_t v = (uint32_t)sym | ((uint32_t)fs << 16);
		for ( uint32_t s = cs; s < cs + fs; s++ )
		{
			s_sf[s] = v;
			s_c[s] = (uint16_t)cs;
		}
	}
}

/* Decode this warp's quarter (32 substreams x 64 values) of one staged chunk. */
static __device__ void SparkQwen36RansDecodeTileHalf(const uint32_t *s_sf, const uint16_t *s_c, const uint8_t *stage, const uint16_t *id_table, uint32_t half, __nv_bfloat16 *tile)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t glane = half * 32u + lane;
	uint32_t x = ((const uint32_t *)stage)[glane];
	const uint8_t *sd = stage + SPARK_QWEN36_RANS_HEADER_BYTES + half * 32u;
	uint32_t j = 0u;
	#pragma unroll 4
	for ( uint32_t it = 0u; it < SPARK_QWEN36_RANS_SUB_LEN; it++ )
	{
		const uint32_t slot = x & (SPARK_QWEN36_RANS_SLOTS - 1u);
		const uint32_t sf = s_sf[slot];
		const uint16_t sym = (uint16_t)(sf & 0xFFFFu);
		const uint16_t fs = (uint16_t)(sf >> 16);
		const uint16_t cs = s_c[slot];
		uint32_t nx = (uint32_t)fs * (x >> SPARK_QWEN36_RANS_L) + slot - cs;
		while ( nx < SPARK_QWEN36_RANS_BOUND ) { nx = (nx << 8) | sd[j * SPARK_QWEN36_RANS_SUBSTREAMS + lane]; j++; }
		uint16_t value;
		if ( sym == 0xFFFFu )
		{
			const uint32_t rid = (uint32_t)sd[j * SPARK_QWEN36_RANS_SUBSTREAMS + lane] | ((uint32_t)sd[(j + 1u) * SPARK_QWEN36_RANS_SUBSTREAMS + lane] << 8);
			j += 2u;
			value = id_table[rid];
		}
		else
			value = sym;
		((uint16_t *)tile)[it * SPARK_QWEN36_RANS_SUBSTREAMS + glane] = value;
		x = nx;
	}
}

/* Cooperative copy of one chunk into the shared staging (per warp). */
static __device__ void SparkQwen36RansStageChunk(uint8_t *stage, const uint8_t *chunk, uint32_t chunk_bytes)
{
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t nwords = (chunk_bytes + 3u) >> 2u;
	const uint32_t *srcw = (const uint32_t *)chunk;
	uint32_t *dstw = (uint32_t *)stage;
	for ( uint32_t w = lane; w < nwords; w += 32u )
		dstw[w] = srcw[w];
	__syncwarp();
}

/*
 * rANS-compressed variant of the tiled small-batch kernel: the weight
 * tensor is stored as compressed 64x128 tiles (128 substreams x 64 values
 * per tile); warps 28..31 decode the next tile while warps 0..27 run the
 * same bit-exact dots from the double-buffered shared tiles. The input
 * staging and the reduction tree are unchanged, so the outputs match the
 * library scalar path bit for bit.
 */
template <uint32_t ROWS>
static __global__ void __launch_bounds__(1024u, 1u) SparkQwen36SmallBatchTiledRansKernel(const void *weight_rans, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension, uint64_t payload_bytes)
{
	extern __shared__ uint8_t rans_shared[];
	uint32_t *s_sf = (uint32_t *)rans_shared;                    /* 4096 x 4B */
	uint16_t *s_c = (uint16_t *)(rans_shared + 16384u);          /* 4096 x 2B */
	uint8_t *stage = rans_shared + 24576u;                       /* 1 x 23KB */
	__nv_bfloat16 *weight_tile = (__nv_bfloat16 *)(rans_shared + 44576u); /* 3 x 16KB */
	__nv_bfloat16 *input_tile = weight_tile + 3u * SPARK_QWEN36_SMALL_BATCH_TILE_N * SPARK_QWEN36_SMALL_BATCH_K_CHUNK;
	uint32_t *tile_ready = (uint32_t *)(input_tile + SPARK_QWEN36_SMALL_BATCH_MAX_ROWS * SPARK_QWEN36_SMALL_BATCH_K_CHUNK);
	uint32_t *tile_done = tile_ready + 4u;
	const uint32_t GROUP_SHIFT = (ROWS == 2u) ? 1u : ((ROWS == 4u) ? 2u : 3u);
	const uint32_t GROUPS = 32u >> GROUP_SHIFT;
	const uint32_t PER_GROUP = SPARK_QWEN36_SMALL_BATCH_TILE_N / GROUPS;
	const uint32_t neuron_base = blockIdx.x * SPARK_QWEN36_SMALL_BATCH_TILE_N;
	const uint32_t thread = threadIdx.x;
	const uint32_t warp = thread >> 5u;
	const uint32_t lane = thread & 31u;
	uint32_t k_base, p;
	float value;
	float acc[2u * 16u];
	/* parse the payload header */
	const uint8_t *payload = (const uint8_t *)weight_rans;
	const uint32_t ndirect = *(const uint32_t *)payload;
	const uint32_t *entries3 = (const uint32_t *)(payload + 4u);
	const uint32_t id_bits = *(const uint32_t *)(payload + 4u + 12u * (uint64_t)(ndirect + 1u));
	const uint16_t *id_table = (const uint16_t *)(payload + 4u + 12u * (uint64_t)(ndirect + 1u) + 4u);
	const uint32_t chunk_count = *(const uint32_t *)(payload + 4u + 12u * (uint64_t)(ndirect + 1u) + 4u + ((uint64_t)1u << id_bits) * 2u);
	const uint32_t *offsets = (const uint32_t *)(payload + 4u + 12u * (uint64_t)(ndirect + 1u) + 4u + ((uint64_t)1u << id_bits) * 2u + 4u);
	/* the offsets are payload-absolute; the chunks live at payload + offset */
	const uint32_t tiles_per_row = input_dimension >> 7u;
	#pragma unroll
	for ( p = 0u; p < 32u; p++ )
		acc[p] = 0.0f;
	if ( thread < 8u )
		tile_ready[thread] = 0u;
	SparkQwen36RansBuildTable(s_sf, s_c, entries3, ndirect);
	__syncthreads();
	if ( warp >= 28u )
	{
		/* the free-running decode: the tiles 0..tiles_per_row-1, two ahead;
		 * each warp decodes one quarter (32 substreams) of the tile and
		 * bumps the per-buffer ready counter under a block fence. */
		const uint32_t base_ti = blockIdx.x * tiles_per_row;
		for ( uint32_t t = 0u; base_ti + t < chunk_count && t < tiles_per_row; t++ )
		{
			const uint32_t ti = base_ti + t;
			const uint32_t off = offsets[ti];
			const uint32_t end = (ti + 1u < chunk_count) ? offsets[ti + 1u] : (uint32_t)payload_bytes;
			while ( *(volatile uint32_t *)&tile_done[t % 3u] < (t / 3u) ) {}
			SparkQwen36RansStageChunk(stage, payload + off, end - off);
			SparkQwen36RansDecodeTileHalf(s_sf, s_c, stage, id_table, warp - 28u, weight_tile + (t % 3u) * SPARK_QWEN36_SMALL_BATCH_TILE_N * SPARK_QWEN36_SMALL_BATCH_K_CHUNK);
			__threadfence_block();
			atomicAdd(&tile_ready[t % 3u], 1u);
			asm volatile("bar.sync 3, 128;");
		}
	}
	else
	{
		for ( k_base = 0u; k_base < input_dimension; k_base += SPARK_QWEN36_SMALL_BATCH_K_CHUNK )
		{
			const uint32_t ki = k_base >> 7u;
			/* 8x128 input tile: threads 0..127; pad rows past row_count. */
			if ( thread < 128u && (thread >> 4u) < row_count )
				((uint4 *)input_tile)[thread] = __ldg(((const uint4 *)input_bf16) + ((((uint64_t)(thread >> 4u) * input_dimension) + k_base) >> 3u) + (thread & 15u));
			else if ( thread < 128u )
				((uint4 *)input_tile)[thread] = make_uint4(0u, 0u, 0u, 0u);
			asm volatile("bar.sync 1, 896;");
			/* wait for the tile ki (the four decode quarters) */
			if ( warp == 0u )
			{
				const uint32_t need = 4u * ((ki / 3u) + 1u);
				while ( *(volatile uint32_t *)&tile_ready[ki % 3u] < need ) {}
			}
			asm volatile("bar.sync 1, 896;");
			const __nv_bfloat16 *wt = weight_tile + (ki % 3u) * SPARK_QWEN36_SMALL_BATCH_TILE_N * SPARK_QWEN36_SMALL_BATCH_K_CHUNK;
			#pragma unroll
			for ( p = 0u; p < PER_GROUP; p++ )
			{
				const uint32_t neuron = ((warp >> GROUP_SHIFT) * PER_GROUP) + p;
				const uint32_t row = warp & (ROWS - 1u);
				acc[p] = fmaf(__bfloat162float(wt[(neuron << 7u) + (lane << 1u)]), __bfloat162float(input_tile[(row << 7u) + (lane << 1u)]), acc[p]);
				acc[p] = fmaf(__bfloat162float(wt[(neuron << 7u) + (lane << 1u) + 1u]), __bfloat162float(input_tile[(row << 7u) + (lane << 1u) + 1u]), acc[p]);
				acc[p] = fmaf(__bfloat162float(wt[(neuron << 7u) + 64u + (lane << 1u)]), __bfloat162float(input_tile[(row << 7u) + 64u + (lane << 1u)]), acc[p]);
				acc[p] = fmaf(__bfloat162float(wt[(neuron << 7u) + 64u + (lane << 1u) + 1u]), __bfloat162float(input_tile[(row << 7u) + 64u + (lane << 1u) + 1u]), acc[p]);
			}
			/* the units 28..31: the warps 0..3 also cover them */
			if ( warp < 4u )
			{
				const uint32_t row2 = (28u + warp) & (ROWS - 1u);
				const uint32_t group2 = (28u + warp) >> GROUP_SHIFT;
				#pragma unroll
				for ( p = 0u; p < PER_GROUP; p++ )
				{
					const uint32_t neuron = group2 * PER_GROUP + p;
					acc[16u + p] = fmaf(__bfloat162float(wt[(neuron << 7u) + (lane << 1u)]), __bfloat162float(input_tile[(row2 << 7u) + (lane << 1u)]), acc[16u + p]);
					acc[16u + p] = fmaf(__bfloat162float(wt[(neuron << 7u) + (lane << 1u) + 1u]), __bfloat162float(input_tile[(row2 << 7u) + (lane << 1u) + 1u]), acc[16u + p]);
					acc[16u + p] = fmaf(__bfloat162float(wt[(neuron << 7u) + 64u + (lane << 1u)]), __bfloat162float(input_tile[(row2 << 7u) + 64u + (lane << 1u)]), acc[16u + p]);
					acc[16u + p] = fmaf(__bfloat162float(wt[(neuron << 7u) + 64u + (lane << 1u) + 1u]), __bfloat162float(input_tile[(row2 << 7u) + 64u + (lane << 1u) + 1u]), acc[16u + p]);
				}
			}
			asm volatile("bar.sync 2, 896;");
			if ( warp == 0u )
			{
				__threadfence_block();
				atomicAdd(&tile_done[ki % 3u], 1u);
			}
		}
	}
	/* reduction + store (the warps 0..27) */
	if ( warp < 28u )
	{
		#pragma unroll
		for ( p = 0u; p < PER_GROUP; p++ )
		{
			const uint32_t neuron = ((warp >> GROUP_SHIFT) * PER_GROUP) + p;
			const uint32_t row = warp & (ROWS - 1u);
			value = acc[p];
			value += __shfl_down_sync(0xffffffffu, value, 16u);
			value += __shfl_down_sync(0xffffffffu, value, 8u);
			value += __shfl_down_sync(0xffffffffu, value, 4u);
			value += __shfl_down_sync(0xffffffffu, value, 2u);
			value += __shfl_down_sync(0xffffffffu, value, 1u);
			if ( lane == 0u && row < row_count && neuron_base + neuron < output_dimension )
				SparkLmFloatToBf16(output_bf16, ((uint64_t)row * output_dimension) + neuron_base + neuron, value);
		}
		if ( warp < 4u )
		{
			const uint32_t row2 = (28u + warp) & (ROWS - 1u);
			const uint32_t group2 = (28u + warp) >> GROUP_SHIFT;
			#pragma unroll
			for ( p = 0u; p < PER_GROUP; p++ )
			{
				const uint32_t neuron = group2 * PER_GROUP + p;
				value = acc[16u + p];
				value += __shfl_down_sync(0xffffffffu, value, 16u);
				value += __shfl_down_sync(0xffffffffu, value, 8u);
				value += __shfl_down_sync(0xffffffffu, value, 4u);
				value += __shfl_down_sync(0xffffffffu, value, 2u);
				value += __shfl_down_sync(0xffffffffu, value, 1u);
				if ( lane == 0u && row2 < row_count && neuron_base + neuron < output_dimension )
					SparkLmFloatToBf16(output_bf16, ((uint64_t)row2 * output_dimension) + neuron_base + neuron, value);
			}
		}
	}
}


extern "C" cudaError_t SparkQwen36LaunchSmallBatchLinearRans(cudaStream_t stream, const void *weight_rans, uint64_t payload_bytes, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	uint32_t blocks = (output_dimension + SPARK_QWEN36_SMALL_BATCH_TILE_N - 1u) / SPARK_QWEN36_SMALL_BATCH_TILE_N;
	size_t shared_bytes = 44576u + (size_t)(3u * SPARK_QWEN36_SMALL_BATCH_TILE_N + SPARK_QWEN36_SMALL_BATCH_MAX_ROWS) * SPARK_QWEN36_SMALL_BATCH_K_CHUNK * sizeof(__nv_bfloat16) + 64u;
	cudaError_t ea = cudaFuncSetAttribute((const void *)SparkQwen36SmallBatchTiledRansKernel<8u>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);
	if (ea != cudaSuccess) return ea;
	ea = cudaFuncSetAttribute((const void *)SparkQwen36SmallBatchTiledRansKernel<4u>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);
	if (ea != cudaSuccess) return ea;
	ea = cudaFuncSetAttribute((const void *)SparkQwen36SmallBatchTiledRansKernel<2u>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);
	if (ea != cudaSuccess) return ea;
	if ( row_count <= 2u )
		SparkQwen36SmallBatchTiledRansKernel<2u><<<blocks, 1024u, shared_bytes, stream>>>(weight_rans, input_bf16, output_bf16, row_count, input_dimension, output_dimension, payload_bytes);
	else if ( row_count <= 4u )
		SparkQwen36SmallBatchTiledRansKernel<4u><<<blocks, 1024u, shared_bytes, stream>>>(weight_rans, input_bf16, output_bf16, row_count, input_dimension, output_dimension, payload_bytes);
	else
		SparkQwen36SmallBatchTiledRansKernel<8u><<<blocks, 1024u, shared_bytes, stream>>>(weight_rans, input_bf16, output_bf16, row_count, input_dimension, output_dimension, payload_bytes);
	{
		cudaError_t e = cudaGetLastError();
		if (e != cudaSuccess) fprintf(stderr, "rans_launch rows=%u blocks=%u shared=%zu err=%s\n", row_count, blocks, shared_bytes, cudaGetErrorString(e));
		return e;
	}
	return cudaSuccess;
}


static __global__ void SparkQwen36SmallBatchLean1Kernel(const void *weight_bf16, const void *input_bf16, void *output_bf16, uint32_t input_dimension, uint32_t output_dimension)
{
	const uint32_t neuron_base = blockIdx.x * SPARK_QWEN36_SMALL_BATCH_TILE_N;
	const uint32_t warp = threadIdx.x >> 5u;
	const uint32_t lane = threadIdx.x & 31u;
	uint32_t k_base, p;
	float value, x0, x1;
	float acc[4];
	#pragma unroll
	for (p = 0u; p < 4u; p++)
		acc[p] = 0.0f;
	for (k_base = 0u; k_base < input_dimension; k_base += SPARK_QWEN36_SMALL_BATCH_K_CHUNK)
	{
		const uint32_t pair0 = k_base >> 1u;
		/* input pairs at chunk-local pair indices l and l+32 */
		const uint32_t in_a = __ldg(((const uint32_t *)input_bf16) + pair0 + lane);
		const uint32_t in_b = __ldg(((const uint32_t *)input_bf16) + pair0 + 32u + lane);
		x0 = __bfloat162float(*(const __nv_bfloat16 *)&in_a);
		x1 = __bfloat162float(*((const __nv_bfloat16 *)&in_a + 1));
		#pragma unroll
		for (p = 0u; p < 4u; p++)
		{
			const uint64_t row = ((uint64_t)(neuron_base + (warp << 2u) + p) * input_dimension);
			const uint32_t wa = __ldg(((const uint32_t *)weight_bf16) + ((row + k_base) >> 1u) + lane);
			const uint32_t wb = __ldg(((const uint32_t *)weight_bf16) + ((row + k_base) >> 1u) + 32u + lane);
			acc[p] = fmaf(x0, __bfloat162float(*(const __nv_bfloat16 *)&wa), acc[p]);
			acc[p] = fmaf(x1, __bfloat162float(*((const __nv_bfloat16 *)&wa + 1)), acc[p]);
			acc[p] = fmaf(__bfloat162float(*(const __nv_bfloat16 *)&in_b), __bfloat162float(*(const __nv_bfloat16 *)&wb), acc[p]);
			acc[p] = fmaf(__bfloat162float(*((const __nv_bfloat16 *)&in_b + 1)), __bfloat162float(*((const __nv_bfloat16 *)&wb + 1)), acc[p]);
		}
	}
	#pragma unroll
	for (p = 0u; p < 4u; p++)
	{
		const uint32_t neuron = neuron_base + (warp << 2u) + p;
		value = acc[p];
		value += __shfl_down_sync(0xffffffffu, value, 16u);
		value += __shfl_down_sync(0xffffffffu, value, 8u);
		value += __shfl_down_sync(0xffffffffu, value, 4u);
		value += __shfl_down_sync(0xffffffffu, value, 2u);
		value += __shfl_down_sync(0xffffffffu, value, 1u);
		if ( lane == 0u && neuron < output_dimension )
			SparkLmFloatToBf16(output_bf16, neuron, value);
	}
}

template <uint32_t ROWS>
static __global__ void SparkQwen36SmallBatchTiledKernel(const void *weight_bf16, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	extern __shared__ __nv_bfloat16 tile[];
	__nv_bfloat16 *weight_tile = tile;
	__nv_bfloat16 *input_tile = tile + (SPARK_QWEN36_SMALL_BATCH_TILE_N * SPARK_QWEN36_SMALL_BATCH_K_CHUNK);
	const uint32_t GROUP_SHIFT = (ROWS == 2u) ? 1u : ((ROWS == 4u) ? 2u : 3u);
	const uint32_t GROUPS = 32u >> GROUP_SHIFT;
	const uint32_t PER_GROUP = SPARK_QWEN36_SMALL_BATCH_TILE_N / GROUPS;
	const uint32_t neuron_base = blockIdx.x * SPARK_QWEN36_SMALL_BATCH_TILE_N;
	const uint32_t thread = threadIdx.x;
	const uint32_t warp = thread >> 5u;
	const uint32_t lane = thread & 31u;
	const uint32_t row = warp & (ROWS - 1u);
	uint32_t k_base, neuron, p;
	float value;
	float acc[PER_GROUP];
	#pragma unroll
	for (p = 0u; p < PER_GROUP; p++)
		acc[p] = 0.0f;
	for (k_base = 0u; k_base < input_dimension; k_base += SPARK_QWEN36_SMALL_BATCH_K_CHUNK)
	{
		/* 64x128 weight tile: one uint4 (8 bf16) per thread. */
		((uint4 *)weight_tile)[thread] = __ldg(((const uint4 *)weight_bf16) + ((((uint64_t)(neuron_base + (thread >> 4u)) * input_dimension) + k_base) >> 3u) + (thread & 15u));
		/* 8x128 input tile: threads 0..127; pad rows past row_count with zero. */
		if ( thread < 128u && (thread >> 4u) < row_count )
			((uint4 *)input_tile)[thread] = __ldg(((const uint4 *)input_bf16) + ((((uint64_t)(thread >> 4u) * input_dimension) + k_base) >> 3u) + (thread & 15u));
		else if ( thread < 128u )
			((uint4 *)input_tile)[thread] = make_uint4(0u, 0u, 0u, 0u);
		__syncthreads();
		/*
		 * Lane l covers chunk-local pairs l and l+32 (k = 2l, 2l+1 and
		 * 2l+64, 2l+65), matching SparkLmDotRowBf16's stride-32 pair loop.
		 */
		#pragma unroll
		for (p = 0u; p < PER_GROUP; p++)
		{
			neuron = ((warp >> GROUP_SHIFT) * PER_GROUP) + p;
			acc[p] = fmaf(__bfloat162float(weight_tile[(neuron << 7u) + (lane << 1u)]), __bfloat162float(input_tile[(row << 7u) + (lane << 1u)]), acc[p]);
			acc[p] = fmaf(__bfloat162float(weight_tile[(neuron << 7u) + (lane << 1u) + 1u]), __bfloat162float(input_tile[(row << 7u) + (lane << 1u) + 1u]), acc[p]);
			acc[p] = fmaf(__bfloat162float(weight_tile[(neuron << 7u) + 64u + (lane << 1u)]), __bfloat162float(input_tile[(row << 7u) + 64u + (lane << 1u)]), acc[p]);
			acc[p] = fmaf(__bfloat162float(weight_tile[(neuron << 7u) + 64u + (lane << 1u) + 1u]), __bfloat162float(input_tile[(row << 7u) + 64u + (lane << 1u) + 1u]), acc[p]);
		}
		__syncthreads();
	}
	/*
	 * Same reduction tree as SparkLmWarpReduceSum, then the library's
	 * round-to-nearest bf16 store.
	 */
	#pragma unroll
	for (p = 0u; p < PER_GROUP; p++)
	{
		neuron = ((warp >> GROUP_SHIFT) * PER_GROUP) + p;
		value = acc[p];
		value += __shfl_down_sync(0xffffffffu, value, 16u);
		value += __shfl_down_sync(0xffffffffu, value, 8u);
		value += __shfl_down_sync(0xffffffffu, value, 4u);
		value += __shfl_down_sync(0xffffffffu, value, 2u);
		value += __shfl_down_sync(0xffffffffu, value, 1u);
		if ( lane == 0u && row < row_count && neuron_base + neuron < output_dimension )
			SparkLmFloatToBf16(output_bf16, ((uint64_t)row * output_dimension) + neuron_base + neuron, value);
	}
}

/*
 * Fused FFN gate+up+swiglu for the decode microbatch (rows 5..8): one block
 * of 32 warps per 64-neuron tile computes BOTH projections from one staged
 * input and applies the swiglu in-register, so each weight matrix streams
 * once, the input stages once, and the separate swiglu kernel (with its
 * gate/up memory round trip) disappears. The dots use the same lane-strided
 * pair order and shfl-down tree as SparkLmDotRowBf16/SparkLmWarpReduceSum,
 * and the activation is SparkLmSwish(gate)*up rounded to bf16 with
 * __float2bfloat16 - bit-identical to the separate kernels.
 */
#define SPARK_QWEN36_SMALL_BATCH_FFN_TILE_N 64u

static __global__ void SparkQwen36SmallBatchFfnGateUpKernel(const void *gate_weight_bf16, const void *up_weight_bf16, const void *input_bf16, void *gated_up_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	extern __shared__ __nv_bfloat16 tile[];
	__nv_bfloat16 *gate_tile = tile;
	__nv_bfloat16 *up_tile = tile + (SPARK_QWEN36_SMALL_BATCH_FFN_TILE_N * SPARK_QWEN36_SMALL_BATCH_K_CHUNK);
	__nv_bfloat16 *input_tile = up_tile + (SPARK_QWEN36_SMALL_BATCH_FFN_TILE_N * SPARK_QWEN36_SMALL_BATCH_K_CHUNK);
	const uint32_t neuron_base = blockIdx.x * SPARK_QWEN36_SMALL_BATCH_FFN_TILE_N;
	const uint32_t thread = threadIdx.x;
	const uint32_t warp = thread >> 5u;
	const uint32_t lane = thread & 31u;
	const uint32_t row = warp & 7u;
	uint32_t k_base, neuron, p;
	float value_g, value_u;
	float acc_g[16];
	float acc_u[16];
	#pragma unroll
	for (p = 0u; p < 16u; p++)
	{
		acc_g[p] = 0.0f;
		acc_u[p] = 0.0f;
	}
	for (k_base = 0u; k_base < input_dimension; k_base += SPARK_QWEN36_SMALL_BATCH_K_CHUNK)
	{
		/* two 64x128 weight tiles (1024 uint4s each) + 8x128 input tile */
		((uint4 *)gate_tile)[thread] = __ldg(((const uint4 *)gate_weight_bf16) + ((((uint64_t)(neuron_base + (thread >> 4u)) * input_dimension) + k_base) >> 3u) + (thread & 15u));
		((uint4 *)up_tile)[thread] = __ldg(((const uint4 *)up_weight_bf16) + ((((uint64_t)(neuron_base + (thread >> 4u)) * input_dimension) + k_base) >> 3u) + (thread & 15u));
		if ( thread < 128u && (thread >> 4u) < row_count )
			((uint4 *)input_tile)[thread] = __ldg(((const uint4 *)input_bf16) + ((((uint64_t)(thread >> 4u) * input_dimension) + k_base) >> 3u) + (thread & 15u));
		else if ( thread < 128u )
			((uint4 *)input_tile)[thread] = make_uint4(0u, 0u, 0u, 0u);
		__syncthreads();
		#pragma unroll
		for (p = 0u; p < 16u; p++)
		{
			neuron = ((warp >> 3u) << 4u) + p;
			acc_g[p] = fmaf(__bfloat162float(gate_tile[(neuron << 7u) + (lane << 1u)]), __bfloat162float(input_tile[(row << 7u) + (lane << 1u)]), acc_g[p]);
			acc_g[p] = fmaf(__bfloat162float(gate_tile[(neuron << 7u) + (lane << 1u) + 1u]), __bfloat162float(input_tile[(row << 7u) + (lane << 1u) + 1u]), acc_g[p]);
			acc_g[p] = fmaf(__bfloat162float(gate_tile[(neuron << 7u) + 64u + (lane << 1u)]), __bfloat162float(input_tile[(row << 7u) + 64u + (lane << 1u)]), acc_g[p]);
			acc_g[p] = fmaf(__bfloat162float(gate_tile[(neuron << 7u) + 64u + (lane << 1u) + 1u]), __bfloat162float(input_tile[(row << 7u) + 64u + (lane << 1u) + 1u]), acc_g[p]);
			acc_u[p] = fmaf(__bfloat162float(up_tile[(neuron << 7u) + (lane << 1u)]), __bfloat162float(input_tile[(row << 7u) + (lane << 1u)]), acc_u[p]);
			acc_u[p] = fmaf(__bfloat162float(up_tile[(neuron << 7u) + (lane << 1u) + 1u]), __bfloat162float(input_tile[(row << 7u) + (lane << 1u) + 1u]), acc_u[p]);
			acc_u[p] = fmaf(__bfloat162float(up_tile[(neuron << 7u) + 64u + (lane << 1u)]), __bfloat162float(input_tile[(row << 7u) + 64u + (lane << 1u)]), acc_u[p]);
			acc_u[p] = fmaf(__bfloat162float(up_tile[(neuron << 7u) + 64u + (lane << 1u) + 1u]), __bfloat162float(input_tile[(row << 7u) + 64u + (lane << 1u) + 1u]), acc_u[p]);
		}
		__syncthreads();
	}
	#pragma unroll
	for (p = 0u; p < 16u; p++)
	{
		neuron = ((warp >> 3u) << 4u) + p;
		value_g = acc_g[p];
		value_g += __shfl_down_sync(0xffffffffu, value_g, 16u);
		value_g += __shfl_down_sync(0xffffffffu, value_g, 8u);
		value_g += __shfl_down_sync(0xffffffffu, value_g, 4u);
		value_g += __shfl_down_sync(0xffffffffu, value_g, 2u);
		value_g += __shfl_down_sync(0xffffffffu, value_g, 1u);
		value_u = acc_u[p];
		value_u += __shfl_down_sync(0xffffffffu, value_u, 16u);
		value_u += __shfl_down_sync(0xffffffffu, value_u, 8u);
		value_u += __shfl_down_sync(0xffffffffu, value_u, 4u);
		value_u += __shfl_down_sync(0xffffffffu, value_u, 2u);
		value_u += __shfl_down_sync(0xffffffffu, value_u, 1u);
		if ( lane == 0u && row < row_count && neuron_base + neuron < output_dimension )
		{
			/* The separate path rounds the gate and up dots to bf16 before the
			 * swiglu reads them, so round here too for bit-exactness. */
			float gate_rounded = __bfloat162float(__float2bfloat16(value_g));
			float up_rounded = __bfloat162float(__float2bfloat16(value_u));
			((__nv_bfloat16 *)gated_up_bf16)[((uint64_t)row * output_dimension) + neuron_base + neuron] = __float2bfloat16(SparkLmSwish(gate_rounded) * up_rounded);
		}
	}
}

extern "C" cudaError_t SparkQwen36LaunchFfnGateUp(cudaStream_t stream, const void *gate_weight_bf16, const void *up_weight_bf16, const void *input_bf16, void *gated_up_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	uint32_t blocks = (output_dimension + SPARK_QWEN36_SMALL_BATCH_FFN_TILE_N - 1u) / SPARK_QWEN36_SMALL_BATCH_FFN_TILE_N;
	size_t shared_bytes = (size_t)(2u * SPARK_QWEN36_SMALL_BATCH_FFN_TILE_N + SPARK_QWEN36_SMALL_BATCH_MAX_ROWS) * SPARK_QWEN36_SMALL_BATCH_K_CHUNK * sizeof(__nv_bfloat16);
	SparkQwen36SmallBatchFfnGateUpKernel<<<blocks, 1024u, shared_bytes, stream>>>(gate_weight_bf16, up_weight_bf16, input_bf16, gated_up_bf16, row_count, input_dimension, output_dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchSmallBatchLinear(cudaStream_t stream, const void *weight_bf16, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	uint32_t blocks = (output_dimension + SPARK_QWEN36_SMALL_BATCH_TILE_N - 1u) / SPARK_QWEN36_SMALL_BATCH_TILE_N;
	size_t shared_bytes = (size_t)(SPARK_QWEN36_SMALL_BATCH_TILE_N + SPARK_QWEN36_SMALL_BATCH_MAX_ROWS) * SPARK_QWEN36_SMALL_BATCH_K_CHUNK * sizeof(__nv_bfloat16);
	if ( row_count == 1u )
		SparkQwen36SmallBatchLean1Kernel<<<blocks, 512u, 0u, stream>>>(weight_bf16, input_bf16, output_bf16, input_dimension, output_dimension);
	else if ( row_count <= 2u )
		SparkQwen36SmallBatchTiledKernel<2u><<<blocks, 1024u, shared_bytes, stream>>>(weight_bf16, input_bf16, output_bf16, row_count, input_dimension, output_dimension);
	else if ( row_count <= 4u )
		SparkQwen36SmallBatchTiledKernel<4u><<<blocks, 1024u, shared_bytes, stream>>>(weight_bf16, input_bf16, output_bf16, row_count, input_dimension, output_dimension);
	else
		SparkQwen36SmallBatchTiledKernel<8u><<<blocks, 1024u, shared_bytes, stream>>>(weight_bf16, input_bf16, output_bf16, row_count, input_dimension, output_dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchLinear(cudaStream_t stream, const SparkQwen36LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count)
{
	const char *gate = getenv("SPARK_QWEN36_SMALL_BATCH_GEMM");
	if ( view->weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16_RANS &&
		row_count <= SPARK_QWEN36_SMALL_BATCH_MAX_ROWS &&
		(view->input_dimension % SPARK_QWEN36_SMALL_BATCH_K_CHUNK) == 0u &&
		(view->output_dimension % SPARK_QWEN36_SMALL_BATCH_TILE_N) == 0u )
		return(SparkQwen36LaunchSmallBatchLinearRans(stream,view->weight_payload,view->weight_payload_bytes,input_bf16,output_bf16,row_count,view->input_dimension,view->output_dimension));
	/* Rows 1..4 stay on the library scalar path: its per-row weight re-reads
	 * hit L2 at those sizes and its 4x larger thread grid hides HBM latency
	 * better than the tiled kernel. Rows 5..8 re-read the full strip enough
	 * to exceed L2, where the once-per-projection shared tile wins. */
	if ( (gate == 0 || strcmp(gate, "0") != 0) &&
		view->weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 &&
		row_count >= 5u && row_count <= SPARK_QWEN36_SMALL_BATCH_MAX_ROWS &&
		(view->input_dimension % SPARK_QWEN36_SMALL_BATCH_K_CHUNK) == 0u &&
		(view->output_dimension % SPARK_QWEN36_SMALL_BATCH_TILE_N) == 0u )
		return(SparkQwen36LaunchSmallBatchLinear(stream,view->weight_payload,input_bf16,output_bf16,row_count,view->input_dimension,view->output_dimension));
	/* Wide-input B1 (DSpark projector fc 25600) exceeds scalar shared-memory budget on
	 * GB10 (101376 opt-in cap); the lean B1 kernel tiles K and needs no dynamic shared. */
	if ( (gate == 0 || strcmp(gate,"0") != 0) &&
		view->weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 &&
		row_count == 1u && view->input_dimension > 24576u &&
		(view->input_dimension % SPARK_QWEN36_SMALL_BATCH_K_CHUNK) == 0u &&
		(view->output_dimension % SPARK_QWEN36_SMALL_BATCH_TILE_N) == 0u )
		return(SparkQwen36LaunchSmallBatchLinear(stream,view->weight_payload,input_bf16,output_bf16,row_count,view->input_dimension,view->output_dimension));
	return(SparkLmHostLaunchBatchedLinear<32u>(stream,view->weight_format,view->weight_payload,view->weight_scale_e8m0,input_bf16,output_bf16,row_count,view->input_dimension,view->output_dimension));
}

extern "C" cudaError_t SparkQwen36LaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen36GdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen36GdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal)
{
	dim3 grid((SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_CONV_CHANNELS) + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS,1u,1u);
	SparkQwen36ConvUpdateKernel<<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(qkv_bf16,weights->conv_weight_bf16,conv_out_bf16,pool->conv_tail_bf16,row_lane_indices,pool->state_cold_by_row,row_count,gdn_layer_ordinal,pool->conv_tail_lane_stride_elements,pool->conv_tail_layer_stride_elements);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen36GdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count)
{
	SparkQwen36DecayBetaKernel<<<row_count,SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS),0,stream>>>(decay_pre_bf16,beta_pre_bf16,weights->a_log_f32,weights->dt_bias_f32,log_decay_f32,beta_f32,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen36GdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal)
{
    dim3 grid(
        SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS),
        1u,
        1u);
    SparkQwen36GdnStepKernel<<<
        grid,
        SPARK_QWEN36_CUDA_DV,
        SPARK_QWEN36_CUDA_GDN_DECODE_SHARED_BYTES,
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

extern "C" cudaError_t SparkQwen36LaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen36GdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon)
{
	dim3 grid(SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS),row_count,1u);
	SparkQwen36GatedNormKernel<<<grid,SPARK_QWEN36_CUDA_DV,0,stream>>>(core_bf16,z_bf16,weights->gdn_norm_weight_bf16,output_bf16,row_count,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen36AttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon)
{
	dim3 grid(SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_QUERY_HEADS),row_count,1u);
	SparkQwen36AttnPrepareKernel<<<grid,SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION,0,stream>>>(q_fused_bf16,k_bf16,v_bf16,weights->query_norm_weight_bf16,weights->key_norm_weight_bf16,kv_cache_bf16,slot_mapping,row_positions,row_count,attn_layer_ordinal,cache_layer_stride,cache_block_stride,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen36KvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride)
{
    dim3 grid(
        SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_QUERY_HEADS) /
            SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA,
        row_count,
        1u);
    static_assert(
        SPARK_QWEN36_MODEL_ATTN_QUERY_HEAD_COUNT %
            SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA == 0u,
        "query-head count must divide the grouped attention CTA width");
    static_assert(
        SPARK_QWEN36_CUDA_ATTN_GROUP %
            SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA == 0u,
        "a grouped attention CTA may not cross KV-head ownership");
    if ( (SparkQwen36TpDim(SPARK_QWEN36_TPD_ATTN_QUERY_HEADS) %
            SPARK_QWEN36_CUDA_ATTN_HEADS_PER_CTA) != 0u )
        return(cudaErrorInvalidValue);
    SparkQwen36AttnDecodeKernel<<<grid, SPARK_LM_CTA_THREADS, 0u, stream>>>(
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
        cache_block_stride);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkQwen36LaunchChunkConv(cudaStream_t stream, const void *qkv_bf16, const SparkQwen36GdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen36GdnStatePool *pool, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal)
{
	if ( token_count == 0u )
		return(cudaErrorInvalidValue);
	SparkQwen36ChunkConvKernel<<<(SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_CONV_CHANNELS) + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS,SPARK_LM_CTA_THREADS,0,stream>>>(qkv_bf16,weights->conv_weight_bf16,conv_out_bf16,pool->conv_tail_bf16,lane_index,token_count,gdn_layer_ordinal,pool->conv_tail_lane_stride_elements,pool->conv_tail_layer_stride_elements);
	return(cudaGetLastError());
}

/*
 * One chunk of one lane's prefill through the GDN core: conv_out and the
 * decay/beta arrays hold token_count (at most 64) consecutive positions.
 * The module loops chunks on the stream; the state dependency serializes
 * for free. Workspace pointers are slot-owned device buffers sized per the
 * view layout (per head: qn/kn/w/kg 64 x 128, decay/attn 64 x 64, cum_g 64).
 */
extern "C" cudaError_t SparkQwen36LaunchGdnChunk(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *workspace_qn, float *workspace_kn, float *workspace_cum_g, float *workspace_decay, float *workspace_attn, float *workspace_w, float *workspace_kg, const SparkQwen36GdnStatePool *pool, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal)
{
    SparkQwen36ChunkWorkspaceView views;
    cudaError_t status;
    dim3 transform_grid(
        SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS),
        token_count,
        1u);

    views.qn = workspace_qn;
    views.kn = workspace_kn;
    views.cum_g = workspace_cum_g;
    views.decay = workspace_decay;
    views.attn = workspace_attn;
    views.w = workspace_w;
    views.kg = workspace_kg;
    if (token_count == 0u || token_count > SPARK_QWEN36_CUDA_CHUNK)
    {
        return cudaErrorInvalidValue;
    }

    SparkQwen36ChunkPrepareKernel<<<
        SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS),
        SPARK_QWEN36_CUDA_CHUNK,
        0u,
        stream>>>(conv_out_bf16, log_decay_f32, beta_f32, views, token_count);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    SparkQwen36ChunkSolveKernel<<<
        SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS),
        SPARK_QWEN36_CUDA_CHUNK,
        0u,
        stream>>>(views, token_count);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    SparkQwen36ChunkTransformKernel<<<
        transform_grid,
        SPARK_QWEN36_CUDA_DV,
        0u,
        stream>>>(conv_out_bf16, beta_f32, views, token_count);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    SparkQwen36ChunkQkDecayKernel<<<
        SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS),
        SPARK_LM_CTA_THREADS,
        SPARK_QWEN36_CUDA_GDN_QK_SHARED_BYTES,
        stream>>>(views, token_count);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        return status;
    }
    SparkQwen36ChunkStepKernel<<<
        SparkQwen36TpDim(SPARK_QWEN36_TPD_GDN_VALUE_HEADS),
        SPARK_QWEN36_CUDA_DV,
        SPARK_QWEN36_CUDA_GDN_CHUNK_SHARED_BYTES,
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

extern "C" cudaError_t SparkQwen36LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count)
{
	uint64_t elements = (uint64_t)row_count * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
	SparkQwen36EmbeddingGatherKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(token_ids,embedding_bf16,hidden_bf16,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t pairs = ((uint64_t)row_count * dimension + 1u) >> 1u;
	SparkQwen36ResidualAddKernel<<<(uint32_t)((pairs + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,delta_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchSwiGlu(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t pairs = ((uint64_t)row_count * dimension + 1u) >> 1u;
	SparkQwen36SwiGluKernel<<<(uint32_t)((pairs + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(gate_bf16,up_bf16,row_count,dimension);
	return(cudaGetLastError());
}

static_assert(SPARK_QWEN36_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP == SPARK_LM_HEAD_SCREEN_CAP,"screen cap must match the shared kernels");

extern "C" cudaError_t SparkQwen36LaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchHeadShadowQuantize<SPARK_LM_HEAD_SHADOW_GROUP>(stream,head_bf16,shadow_payload,shadow_scale,error_norm,candidate_count,hidden_dimension));
}

// Screened exact head, the mimo25 pattern: coarse fp4 tile, certified
// screen, exact rescore, device-side overflow fallback; the token
// equals the reference argmax always.
extern "C" cudaError_t SparkQwen36LaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count)
{
	return(SparkLmHostLaunchHeadScreenedArgmax(stream,hidden_bf16,head_weight_bf16,shadow_payload,shadow_scale,error_norm,logits_bf16,candidate_ids,candidate_counts,output_token_ids,row_count,candidate_count,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION));
}

extern "C" cudaError_t SparkQwen36LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count)
{
	SparkLmHeadArgmaxKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,head_weight_bf16,token_ids,output_token_ids,row_count,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,candidate_count);
	return(cudaGetLastError());
}

/*
 * Tensor-parallel head shard reduce: each rank's screened argmax returns
 * its LOCAL winner with the exact f32 score; the rank packs (score, global
 * token) into a monotone u64 key and the collective maxloc picks the global
 * winner. The classic float total-order trick makes the unsigned compare
 * rank negative logits correctly; the low word tie-breaks on the smaller
 * global token, matching the reference argmax's tie rule.
 */
static __device__ __forceinline__ uint32_t SparkQwen36HeadOrderKey(float score)
{
	uint32_t bits = __float_as_uint(score);
	return((bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u);
}

static __global__ void SparkQwen36HeadMaxLocPackKernel(const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count)
{
	uint32_t row = blockIdx.x;
	if ( row >= row_count )
		return;
	keys_u64[row] = ((uint64_t)SparkQwen36HeadOrderKey(scores_f32[row]) << 32u) | (uint64_t)token_ids_u32[row];
}

extern "C" cudaError_t SparkQwen36LaunchHeadScreenedArgmaxScore(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *scratch_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, float *output_scores, uint32_t candidate_offset, uint32_t row_count, uint32_t candidate_count)
{
	return(SparkLmHostLaunchHeadScreenedArgmaxWithScore(stream,hidden_bf16,head_weight_bf16,shadow_payload,shadow_scale,error_norm,scratch_bf16,candidate_ids,candidate_counts,output_token_ids,output_scores,candidate_offset,row_count,candidate_count,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION));
}

extern "C" cudaError_t SparkQwen36LaunchHeadMaxLocPack(cudaStream_t stream, const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count)
{
	SparkQwen36HeadMaxLocPackKernel<<<row_count,1u,0,stream>>>(scores_f32,token_ids_u32,keys_u64,row_count);
	return(cudaGetLastError());
}

static __global__ void SparkQwen36HeadMaxLocUnpackKernel(const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count)
{
	uint32_t row = blockIdx.x;
	if ( row >= row_count )
		return;
	token_ids_u32[row] = (uint32_t)keys_u64[row];
}

extern "C" cudaError_t SparkQwen36LaunchHeadMaxLocUnpack(cudaStream_t stream, const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count)
{
	SparkQwen36HeadMaxLocUnpackKernel<<<row_count,1u,0,stream>>>(keys_u64,token_ids_u32,row_count);
	return(cudaGetLastError());
}

/*
 * Transport-collective combine kernels: fold the staged reduction into the
 * consumer buffer in place, stream-ordered between the producing kernels and
 * the consuming layer. BF16 adds are elementwise; the relay variant also
 * copies the source into the next route's send buffer; the TP4 tree variant
 * adds every peer rank's contribution except the destination's own; the u64
 * variant is an elementwise max.
 */
static __global__ void SparkQwen36AccumAddKernel(void *destination, const void *source, uint32_t element_count)
{
	uint64_t pair = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t pair_count = (uint64_t)element_count >> 1u;
	float2 dst,src;
	if ( pair >= pair_count )
		return;
	dst = SparkLmLoadBf16Pair(destination,pair);
	src = SparkLmLoadBf16Pair(source,pair);
	SparkLmStoreBf16Pair(destination,pair,dst.x + src.x,dst.y + src.y);
	if ( pair == 0u && ((uint64_t)element_count & 1u) != 0u )
		SparkLmFloatToBf16(destination,element_count - 1u,SparkLmBf16ToFloat(destination,element_count - 1u) + SparkLmBf16ToFloat(source,element_count - 1u));
}

static __global__ void SparkQwen36AccumAddRelayKernel(void *destination, const void *source, void *relay, uint32_t element_count)
{
	uint64_t pair = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t pair_count = (uint64_t)element_count >> 1u;
	float2 dst,src;
	if ( pair >= pair_count )
		return;
	dst = SparkLmLoadBf16Pair(destination,pair);
	src = SparkLmLoadBf16Pair(source,pair);
	SparkLmStoreBf16Pair(destination,pair,dst.x + src.x,dst.y + src.y);
	SparkLmStoreBf16Pair(relay,pair,src.x,src.y);
}

static __global__ void SparkQwen36AccumAddTp4Kernel(void *destination, const void *const *rank_devices, uint32_t tp_rank, uint32_t element_count)
{
	uint64_t pair = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t pair_count = (uint64_t)element_count >> 1u;
	float2 dst,peer;
	uint32_t rank;
	if ( pair >= pair_count )
		return;
	dst = SparkLmLoadBf16Pair(destination,pair);
	for (rank = 0u; rank < 4u; rank++)
	{
		if ( rank == tp_rank )
			continue;
		peer = SparkLmLoadBf16Pair(rank_devices[rank],pair);
		dst.x += peer.x;
		dst.y += peer.y;
	}
	SparkLmStoreBf16Pair(destination,pair,dst.x,dst.y);
}

static __global__ void SparkQwen36AccumU64MaxKernel(uint64_t *destination, const uint64_t *source, uint32_t element_count)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint64_t value;
	if ( index >= (uint64_t)element_count )
		return;
	value = source[index];
	if ( value > destination[index] )
		destination[index] = value;
}

extern "C" cudaError_t SparkQwen36LaunchAccumAdd(cudaStream_t stream, void *destination, const void *source, uint32_t active_sequence_count, uint32_t hidden_dimension)
{
	uint64_t elements = (uint64_t)active_sequence_count * hidden_dimension;
	SparkQwen36AccumAddKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(destination,source,(uint32_t)elements);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchAccumAddRelay(cudaStream_t stream, void *destination, const void *source, void *relay, uint32_t active_sequence_count, uint32_t hidden_dimension)
{
	uint64_t elements = (uint64_t)active_sequence_count * hidden_dimension;
	SparkQwen36AccumAddRelayKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(destination,source,relay,(uint32_t)elements);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchAccumAddTp4(cudaStream_t stream, void *destination, const void *const rank_devices[4], uint32_t tp_rank, uint32_t active_sequence_count, uint32_t hidden_dimension)
{
	uint64_t elements = (uint64_t)active_sequence_count * hidden_dimension;
	SparkQwen36AccumAddTp4Kernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(destination,rank_devices,tp_rank,(uint32_t)elements);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchAccumU64Max(cudaStream_t stream, uint64_t *destination, const uint64_t *source, uint32_t element_count)
{
	SparkQwen36AccumU64MaxKernel<<<(element_count + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS,SPARK_LM_CTA_THREADS,0,stream>>>(destination,source,element_count);
	return(cudaGetLastError());
}


extern "C" cudaError_t SparkQwen36LaunchDsparkTapStore(cudaStream_t stream, const void *hidden_bf16, const uint64_t *row_positions, void *taps_bf16, uint32_t rows, uint32_t tap_index, uint32_t hidden_dim, uint32_t tap_layers)
{
	SparkQwen36DsparkTapStoreKernel<<<rows, 256u, 0u, stream>>>((const __nv_bfloat16 *)hidden_bf16, row_positions, (__nv_bfloat16 *)taps_bf16, rows, tap_index, hidden_dim, tap_layers);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchDsparkKPrep(cudaStream_t stream, void *k_bf16, const void *k_norm_bf16, const uint64_t *positions, uint32_t rows)
{
	dim3 grid(rows, 8u);
	SparkQwen36DsparkKPrepKernel<<<grid, SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM, 0u, stream>>>((__nv_bfloat16 *)k_bf16, (const __nv_bfloat16 *)k_norm_bf16, positions, rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchDsparkQPrep(cudaStream_t stream, void *q_bf16, const void *q_norm_bf16, const uint64_t *positions, uint32_t rows)
{
	dim3 grid(rows, 32u);
	SparkQwen36DsparkQPrepKernel<<<grid, SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM, 0u, stream>>>((__nv_bfloat16 *)q_bf16, (const __nv_bfloat16 *)q_norm_bf16, positions, rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchDsparkCacheAttn(cudaStream_t stream, const void *q_bf16, const void *k_bf16, const void *v_bf16, void *attn_out_bf16, uint32_t block_rows, uint32_t nkv)
{
	dim3 grid(block_rows, 32u);
	SparkQwen36DsparkCacheAttnKernel<<<grid, SPARK_QWEN36_DSPARK_ATTN_HEAD_DIM, 2056u * sizeof(float), stream>>>((const __nv_bfloat16 *)q_bf16, (const __nv_bfloat16 *)k_bf16, (const __nv_bfloat16 *)v_bf16, (__nv_bfloat16 *)attn_out_bf16, nkv);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchDsparkConv(cudaStream_t stream, const void *x_bf16, const void *delta_bf16, const void *base_bf16, void *out_bf16, uint32_t block_size, uint32_t num_groups, uint32_t group_size, uint32_t side)
{
	dim3 grid(block_size, num_groups);
	SparkQwen36DsparkConvKernel<<<grid, group_size, 0u, stream>>>((const __nv_bfloat16 *)x_bf16, (const __nv_bfloat16 *)delta_bf16, (const __nv_bfloat16 *)base_bf16, (__nv_bfloat16 *)out_bf16, block_size, num_groups, group_size, side);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkQwen36LaunchDsparkMarkov(cudaStream_t stream, const void *markov_w1_bf16, const void *markov_w2_bf16, const uint32_t *prev_token_ids, uint32_t draft_count, uint32_t rank, void *bias_out, uint32_t vocab)
{
	dim3 grid(draft_count, (vocab + 255u) / 256u);
	SparkQwen36DsparkMarkovKernel<<<grid, 256u, 0u, stream>>>((const __nv_bfloat16 *)markov_w1_bf16, (const __nv_bfloat16 *)markov_w2_bf16, prev_token_ids, draft_count, rank, (float *)bias_out, vocab);
	return(cudaGetLastError());
}
