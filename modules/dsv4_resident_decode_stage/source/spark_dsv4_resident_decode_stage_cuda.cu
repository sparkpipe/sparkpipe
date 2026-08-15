#include "sparkpipe/spark_lm_kernels.cuh"
#include "sparkpipe/spark_row_compaction.cuh"
#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"
#include "spark_dsv4_pool_layout.h"
#include "spark_dsv4_hc_splitk.h"
#include "spark_dsv4_sparse_attention_split.h"
#include "spark_dsv4_stagepack_format.h"
#include "inference/kernels/route.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "runtime/gemm.cuh"

#include <cooperative_groups.h>
#include <math.h>
#include <stdio.h>

/* Bitonic sort needs a power of two; pad to the next one (Pro has 384
 * experts). The router pads non-existent experts with zero keys, which
 * sort below every real key, so the top-k tail still selects real experts. */
#if SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT <= 256u
#define SPARK_DSV4_ROUTER_SORT_CAPACITY 256u
#elif SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT <= 512u
#define SPARK_DSV4_ROUTER_SORT_CAPACITY 512u
#else
#define SPARK_DSV4_ROUTER_SORT_CAPACITY 1024u
#endif
#define SPARK_DSV4_EXPERT_STAGES 4u
#define SPARK_DSV4_EXPERT_WARPS 8u
#define SPARK_DSV4_HC_ELEMENT_TILE 256u
#define SPARK_DSV4_HC_MINIMUM_BLOCKS 16u

using SparkDsv4ExpertWeightFormat =
	typename LmWeightCodec<SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC>::Format;

static_assert(SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC != SPARK_WEIGHT_CODEC_NONE,
	"DSV4 requires an explicit routed-expert codec");
static_assert(SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC != SPARK_WEIGHT_CODEC_BF16,
	"DSV4 routed experts require a compressed package codec");
static_assert(SPARK_DSV4_MODEL_NON_EXPERT_ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_NONE,
	"DSV4 Flash requires BF16 non-expert activations");
static_assert(SPARK_DSV4_MODEL_EXPERT_ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0,
	"DSV4 Flash requires its declared expert activation codec");
static_assert(SPARK_DSV4_MODEL_OUTPUT_COMPOSITION_ACTIVATION_CODEC <= SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0,
	"DSV4 output composition requires a declared activation codec");
static_assert(SPARK_DSV4_WEIGHT_READ_AHEAD_THREAD_COUNT == SPARK_LM_CTA_THREADS,
	"DSV4 read-ahead scratch geometry must match its CUDA launch width");
static_assert(SPARK_DSV4_MODEL_HIDDEN_DIMENSION % SparkDsv4ExpertWeightFormat::kScaleGroup == 0u,
	"DSV4 hidden width must contain complete expert codec scale groups");
static_assert(SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION % SparkDsv4ExpertWeightFormat::kScaleGroup == 0u,
	"DSV4 expert width must contain complete expert codec scale groups");

static __device__ __forceinline__ uint32_t SparkDsv4OrderedHeadScore(float score)
{
	uint32_t bits;
	if ( isnan(score) )
		return(0u);
	if ( score == 0.0f )
		score = 0.0f;
	bits = __float_as_uint(score);
	return(bits ^ ((bits & UINT32_C(0x80000000)) != 0u ?
		UINT32_MAX : UINT32_C(0x80000000)));
}

static __global__ void SparkDsv4HeadMaxlocPackKernel(
	const float *scores,
	const uint32_t *token_ids,
	uint64_t *maxloc,
	uint32_t row_count)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		maxloc[row] = ((uint64_t)SparkDsv4OrderedHeadScore(scores[row]) << 32u) |
			(UINT32_MAX - token_ids[row]);
}

static __global__ void SparkDsv4HeadMaxlocUnpackKernel(
	const uint64_t *maxloc,
	uint32_t *token_ids,
	uint32_t row_count)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		token_ids[row] = UINT32_MAX - (uint32_t)maxloc[row];
}

static __global__ void SparkDsv4ResidentTokenFeedbackKernel(
	const uint32_t *output_token_ids,
	uint32_t *resident_token_ids,
	uint32_t *input_token_ids,
	uint64_t *row_positions,
	uint64_t *row_emit_positions,
	uint64_t *row_emit_positions_hca,
	uint32_t row_count,
	uint32_t tokens_per_sequence,
	uint32_t step_index,
	uint32_t advance)
{
	uint32_t row,token;
	uint64_t position;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row >= row_count )
		return;
	token = output_token_ids[row];
	resident_token_ids[row * tokens_per_sequence + step_index] = token;
	if ( advance == 0u )
		return;
	position = row_positions[row] + 1u;
	input_token_ids[row] = token;
	row_positions[row] = position;
	row_emit_positions[row] = position + 1u >=
		SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO ? position + 1u -
		SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO : 0u;
	row_emit_positions_hca[row] = position + 1u >=
		SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO ? position + 1u -
		SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO : 0u;
}

static __global__ void SparkDsv4AccumU64MaxKernel(
	uint64_t *destination,
	const uint64_t *source,
	uint32_t element_count)
{
	uint32_t element;
	element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < element_count && source[element] > destination[element] )
		destination[element] = source[element];
}

/*
 * Production DSV4 compute is an SM121-only, fail-closed route.  This runtime
 * architecture check is not a hardware-qualification receipt.  Cache the
 * check per executor thread (a resident module binds a thread to one device)
 * so the hot layer loop does not issue a device-properties query.  A caller
 * that moves the same resident executor thread to another device violates the
 * module binding contract; the SM121 device code is independently guarded by
 * LM_SM121_NATIVE_COMPUTE_PTX and traps on any non-SM121 image.
 */
static cudaError_t SparkDsv4RequireNativeSm121(void)
{
	static thread_local int32_t checked = 0;
	static thread_local cudaError_t cached = cudaErrorInvalidValue;
	cudaDeviceProp properties;
	int device;
	if ( checked != 0 )
		return(cached);
	checked = 1;
	cached = cudaGetDevice(&device);
	if ( cached == cudaSuccess )
		cached = cudaGetDeviceProperties(&properties,device);
	if ( cached != cudaSuccess || properties.major != 12 ||
		properties.minor != 1 )
		cached = cudaErrorInvalidValue;
	return(cached);
}

static cudaError_t SparkDsv4RequireNativeDecodeShape(uint32_t rows)
{
	if ( SparkLmSm121NativeDecodeShape(rows) == 0u )
		return(cudaErrorInvalidValue);
	return(SparkDsv4RequireNativeSm121());
}

/*
 * DeepSeek V4 device kernels. The variant model header arrives via the
 * build's -include ahead of everything here; nothing below names a
 * variant. Shared machinery (linear over bf16/fp8/mxfp4, rms norm, head
 * argmax, embedding gather, reductions) comes from spark_lm_kernels.cuh;
 * this file holds only what DeepSeek V4 adds: adjacent-pair rope and its
 * inverse, the unweighted query-head rms, the fp8/fp4 quantize-dequantize
 * cache sims with power-of-two scales, Hadamard rotation, sink-in-
 * denominator sparse attention over gathered cache slots, the gated
 * softmax compressor in both prefill and decode-state forms, indexer
 * scoring and iterative top-k, the two router gates, the swiglu clamp,
 * and the full mHC split with inference Sinkhorn. Every kernel body stays
 * within fifty lines; the sparse-attention two-pass and the compressor
 * pooling decompose through device helpers.
 */

// e2m1 snap with RN-even at every midpoint: the mantissa-zero neighbour
// wins ties, matching the reference cast exactly.
static __device__ __forceinline__ float SparkDsv4EncodeE2m1(float value)
{
	const float points[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
	const float ties[7] = {0.0f,1.0f,1.0f,2.0f,2.0f,4.0f,4.0f};
	float magnitude = fabsf(value),sign = value < 0.0f ? -1.0f : 1.0f,midpoint;
	uint32_t index;
	if ( magnitude >= 6.0f )
		return(sign * 6.0f);
	for (index = 0; index < 7u; index++)
	{
		midpoint = (points[index] + points[index + 1u]) * 0.5f;
		if ( magnitude < midpoint )
			return(sign * points[index]);
		if ( magnitude == midpoint )
			return(sign * ties[index]);
	}
	return(sign * 6.0f);
}

static __device__ __forceinline__ float SparkDsv4Pow2CeilScale(float amax, float format_max)
{
	return(exp2f(ceilf(log2f(amax / format_max))));
}

/*
 * In-place block quantize-dequantize sim over the trailing width of each
 * row: per block amax (floored 1e-4), power-of-two scale, snap, rescale -
 * fp8 with 448 or fp4 with 6 by format_max. One warp per (row, block).
 */
static __device__ __forceinline__ void SparkDsv4QuantSimGroup(
	void *data_bf16,uint32_t row,uint32_t group,uint32_t lane,
	uint32_t row_stride,uint32_t width,uint32_t block,float format_max,
	uint32_t fp4)
{
	uint32_t base = group * block,limit = base + block < width ? base + block : width,element;
	uint64_t offset = (uint64_t)row * row_stride;
	float value,amax = 1e-4f,scale;
	if ( base >= width )
		return;
	for (element = base + lane; element < limit; element += SPARK_LM_WARP_LANES)
	{
		value = fabsf(SparkLmBf16ToFloat(data_bf16,offset + element));
		if ( value > amax )
			amax = value;
	}
	for (element = SPARK_LM_WARP_LANES / 2u; element != 0u; element >>= 1u)
	{
		value = __shfl_down_sync(0xffffffffu,amax,element);
		if ( value > amax )
			amax = value;
	}
	amax = __shfl_sync(0xffffffffu,amax,0);
	scale = SparkDsv4Pow2CeilScale(amax,format_max);
	for (element = base + lane; element < limit; element += SPARK_LM_WARP_LANES)
	{
		value = SparkLmBf16ToFloat(data_bf16,offset + element) / scale;
		if ( value > format_max )
			value = format_max;
		if ( value < -format_max )
			value = -format_max;
		value = (fp4 != 0u ? SparkDsv4EncodeE2m1(value) : LmE4m3ToFloat(LmFloatToE4m3(value))) * scale;
		SparkLmFloatToBf16(data_bf16,offset + element,value);
	}
}

static __global__ void SparkDsv4QuantSimKernel(void *data_bf16, uint32_t row_count, uint32_t row_stride, uint32_t width, uint32_t block, float format_max, uint32_t fp4)
{
	uint32_t row = blockIdx.x,group = blockIdx.y,lane = threadIdx.x;
	if ( row >= row_count )
		return;
	SparkDsv4QuantSimGroup(data_bf16,row,group,lane,row_stride,width,block,
		format_max,fp4);
}

// Adjacent-pair rotation on the LAST rope_dim entries of every head; the
// inverse conjugates - the attention output's de-rotation. One block per
// (row, head), threads over pairs; freqs are the layer's YaRN table.
static __device__ __forceinline__ void SparkDsv4RopePair(
	void *data_bf16,const float *freqs_f32,const uint64_t *row_positions,
	uint32_t row,uint32_t head,uint32_t head_count,uint32_t head_dim,
	uint32_t rope_dim,uint32_t pair,uint32_t inverse)
{
	uint64_t base;
	float angle,cosine,sine,real,imaginary;
	if ( pair >= rope_dim / 2u )
		return;
	base = (((uint64_t)row * head_count) + head) * head_dim + (head_dim - rope_dim) + 2u * pair;
	angle = (float)row_positions[row] * freqs_f32[pair];
	cosine = cosf(angle);
	sine = inverse != 0u ? -sinf(angle) : sinf(angle);
	real = SparkLmBf16ToFloat(data_bf16,base);
	imaginary = SparkLmBf16ToFloat(data_bf16,base + 1u);
	SparkLmFloatToBf16(data_bf16,base,real * cosine - imaginary * sine);
	SparkLmFloatToBf16(data_bf16,base + 1u,real * sine + imaginary * cosine);
}

static __device__ __forceinline__ void SparkDsv4RopeRow(
	void *data_bf16,const float *freqs_f32,const uint64_t *row_positions,
	uint32_t row,uint32_t head,uint32_t head_count,uint32_t head_dim,
	uint32_t rope_dim,uint32_t inverse)
{
	SparkDsv4RopePair(data_bf16,freqs_f32,row_positions,row,head,head_count,
		head_dim,rope_dim,threadIdx.x,inverse);
}

static __global__ void SparkDsv4RopeKernel(void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse)
{
	uint32_t row = blockIdx.x,head = blockIdx.y;
	if ( row >= row_count || head >= head_count )
		return;
	SparkDsv4RopeRow(data_bf16,freqs_f32,row_positions,row,head,head_count,
		head_dim,rope_dim,inverse);
}

// The unweighted per-head query rms the reference applies before rope.
static __device__ __forceinline__ void SparkDsv4QueryHeadRmsRow(
	void *data_bf16,uint32_t row,uint32_t head,uint32_t head_count,
	uint32_t head_dim,float epsilon,float *reduce_scratch)
{
	uint32_t element;
	uint64_t base = (((uint64_t)row * head_count) + head) * head_dim;
	float value,total = 0.0f,inverse;
	for (element = threadIdx.x; element < head_dim; element += blockDim.x)
	{
		value = SparkLmBf16ToFloat(data_bf16,base + element);
		total += value * value;
	}
	total = SparkLmBlockReduceSum(total,reduce_scratch);
	inverse = rsqrtf(total / (float)head_dim + epsilon);
	for (element = threadIdx.x; element < head_dim; element += blockDim.x)
		SparkLmFloatToBf16(data_bf16,base + element,SparkLmBf16ToFloat(data_bf16,base + element) * inverse);
}

static __global__ void SparkDsv4QueryHeadRmsKernel(void *data_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim, float epsilon)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x,head = blockIdx.y;
	if ( row >= row_count || head >= head_count )
		return;
	SparkDsv4QueryHeadRmsRow(data_bf16,row,head,head_count,head_dim,epsilon,
		reduce_scratch);
}

static __global__ void SparkDsv4QueryHeadRmsRopeKernel(
	void *data_bf16,const float *freqs_f32,const uint64_t *row_positions,
	uint32_t row_count,uint32_t head_count,uint32_t head_dim,
	uint32_t rope_dim,float epsilon,uint32_t inverse)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x,head = blockIdx.y;
	if ( row >= row_count || head >= head_count )
		return;
	SparkDsv4QueryHeadRmsRow(data_bf16,row,head,head_count,head_dim,epsilon,
		reduce_scratch);
	__syncthreads();
	SparkDsv4RopeRow(data_bf16,freqs_f32,row_positions,row,head,head_count,
		head_dim,rope_dim,inverse);
}

static __global__ void SparkDsv4KvPostKernel(
	void *data_bf16,const void *gain_bf16,const float *freqs_f32,
	const uint64_t *row_positions,uint32_t row_count,uint32_t head_dim,
	uint32_t rope_dim,uint32_t quant_width,uint32_t quant_block,float epsilon,
	uint32_t inverse)
{
	extern __shared__ float staged_input[];
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x,warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	if ( row >= row_count )
		return;
	SparkLmRmsNormRow(data_bf16,gain_bf16,data_bf16,row,head_dim,epsilon,
		staged_input,reduce_scratch);
	__syncthreads();
	if ( warp * quant_block < quant_width )
		SparkDsv4QuantSimGroup(data_bf16,row,warp,lane,head_dim,quant_width,
			quant_block,448.0f,0u);
	else if ( warp == SPARK_LM_CTA_WARPS - 1u )
		SparkDsv4RopePair(data_bf16,freqs_f32,row_positions,row,0u,1u,
			head_dim,rope_dim,lane,inverse);
}

// In-place Hadamard rotation scaled n^-0.5 on power-of-two vectors; one
// block per vector, the whole vector staged in shared memory.
static __device__ __forceinline__ void SparkDsv4HadamardRow(
	void *data_bf16,uint32_t vector,uint32_t width,float *hadamard_shared)
{
	uint32_t element,half,partner;
	uint64_t base = (uint64_t)vector * width;
	float scale = rsqrtf((float)width),a,b;
	for (element = threadIdx.x; element < width; element += blockDim.x)
		hadamard_shared[element] = SparkLmBf16ToFloat(data_bf16,base + element);
	__syncthreads();
	for (half = 1; half < width; half <<= 1u)
	{
		for (element = threadIdx.x; element < width / 2u; element += blockDim.x)
		{
			partner = ((element / half) * half * 2u) + (element % half);
			a = hadamard_shared[partner];
			b = hadamard_shared[partner + half];
			hadamard_shared[partner] = a + b;
			hadamard_shared[partner + half] = a - b;
		}
		__syncthreads();
	}
	for (element = threadIdx.x; element < width; element += blockDim.x)
		SparkLmFloatToBf16(data_bf16,base + element,hadamard_shared[element] * scale);
}

static __global__ void SparkDsv4HadamardKernel(void *data_bf16, uint32_t vector_count, uint32_t width)
{
	extern __shared__ float hadamard_shared[];
	uint32_t vector = blockIdx.x;
	if ( vector >= vector_count )
		return;
	SparkDsv4HadamardRow(data_bf16,vector,width,hadamard_shared);
}

static __global__ void SparkDsv4IndexerPostKernel(
	void *data_bf16,const float *freqs_f32,const uint64_t *row_positions,
	uint32_t row_count,uint32_t head_count,uint32_t head_dim,
	uint32_t rope_dim,uint32_t quant_block,uint32_t inverse)
{
	extern __shared__ float hadamard_shared[];
	uint32_t row = blockIdx.x,head = blockIdx.y;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t vector;
	void *head_data;
	if ( row >= row_count || head >= head_count )
		return;
	SparkDsv4RopeRow(data_bf16,freqs_f32,row_positions,row,head,head_count,
		head_dim,rope_dim,inverse);
	__syncthreads();
	vector = row * head_count + head;
	SparkDsv4HadamardRow(data_bf16,vector,head_dim,hadamard_shared);
	__syncthreads();
	head_data = (uint8_t *)data_bf16 +
		(uint64_t)vector * head_dim * sizeof(uint16_t);
	if ( warp * quant_block < head_dim )
		SparkDsv4QuantSimGroup(head_data,0u,warp,lane,head_dim,head_dim,
			quant_block,SPARK_DSV4_MODEL_FP4_MAX,1u);
}

static __device__ __forceinline__ uint32_t SparkDsv4SparseAttnActiveSplitCount(
	uint32_t valid_topk,uint32_t split_count)
{
	uint32_t active;
	active = (valid_topk + SPARK_LM_CTA_WARPS - 1u) /
		SPARK_LM_CTA_WARPS;
	return(min(max(active,1u),split_count));
}

static __global__ void SparkDsv4SparseAttnKernel(
    const void *q_bf16,
    const void *kv_cache_bf16,
    uint64_t lane_stride_elements,
    const uint32_t *row_lane_indices,
    const uint32_t *row_page_table_indices,
    const uint32_t *physical_page_table,
    uint32_t page_table_stride,
    uint32_t compressed_entries_per_page,
    const int32_t *topk_idxs,
    const uint32_t *valid_topk_counts,
    uint32_t topk,
    const float *sink_f32,
    float scale,
    void *out_bf16,
    float *partials_f32,
    uint32_t row_count,
    uint32_t head_count,
    uint32_t head_dim,
    uint32_t split_count)
{
    static const uint32_t heads_per_cta = SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA;
    static const uint32_t maximum_pairs_per_lane =
        SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION /
        (2u * SPARK_LM_WARP_LANES);
    extern __shared__ unsigned char grouped_attention_shared[];
    __shared__ float merge_max[
        heads_per_cta * SPARK_LM_CTA_WARPS];
    __shared__ float merge_den[
        heads_per_cta * SPARK_LM_CTA_WARPS];
    __shared__ float merge_scale[
        heads_per_cta * SPARK_LM_CTA_WARPS];
    __shared__ float inverse_denominator[heads_per_cta];
    float *query_shared;
    float *merge_accumulator;
    float running_max[heads_per_cta];
    float running_denominator[heads_per_cta];
    float2 accumulator[heads_per_cta][maximum_pairs_per_lane];
    uint32_t row;
    uint32_t first_head;
    uint32_t active_head_count;
    uint32_t warp_index;
    uint32_t lane_index;
    uint32_t local_head;
    uint32_t pair_index;
    uint32_t element_index;
    uint32_t selected_slot;
    uint32_t pairs_per_lane;
    uint32_t page_ordinal;
    uint32_t split_index;
    uint32_t active_split_count;
    uint32_t valid_topk;

    row = blockIdx.x;
    first_head = blockIdx.y * heads_per_cta;
    split_index = blockIdx.z;
    if (row >= row_count || first_head >= head_count ||
        head_dim == 0u || head_dim > SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION ||
        (head_dim & 1u) != 0u || split_index >= split_count)
    {
        return;
    }
    valid_topk = min(__ldg(valid_topk_counts + row),topk);
    active_split_count = SparkDsv4SparseAttnActiveSplitCount(valid_topk,
        split_count);
    if ( split_index >= active_split_count )
        return;
    active_head_count = head_count - first_head;
    if (active_head_count > heads_per_cta)
    {
        active_head_count = heads_per_cta;
    }
    warp_index = threadIdx.x / SPARK_LM_WARP_LANES;
    lane_index = threadIdx.x % SPARK_LM_WARP_LANES;
    pairs_per_lane =
        ((head_dim >> 1u) + SPARK_LM_WARP_LANES - 1u) /
        SPARK_LM_WARP_LANES;
    query_shared = reinterpret_cast<float *>(grouped_attention_shared);
    merge_accumulator =
        query_shared + (heads_per_cta * head_dim);

    element_index = threadIdx.x;
    while (element_index < active_head_count * head_dim)
    {
        local_head = element_index / head_dim;
        query_shared[element_index] = SparkLmBf16ToFloat(
            q_bf16,
            (((uint64_t)row * head_count) + first_head + local_head) *
                head_dim +
                (element_index - (local_head * head_dim)));
        element_index += blockDim.x;
    }
    element_index = threadIdx.x;
    while (element_index <
        heads_per_cta * SPARK_LM_CTA_WARPS * head_dim)
    {
        merge_accumulator[element_index] = 0.0f;
        element_index += blockDim.x;
    }
    for (local_head = 0u; local_head < heads_per_cta; ++local_head)
    {
        running_max[local_head] = -3.0e38f;
        running_denominator[local_head] = 0.0f;
        for (pair_index = 0u;
             pair_index < maximum_pairs_per_lane;
             ++pair_index)
        {
            accumulator[local_head][pair_index] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    page_ordinal = row_page_table_indices[row];
    for (selected_slot = split_index * SPARK_LM_CTA_WARPS + warp_index;
         selected_slot < valid_topk;
         selected_slot += split_count * SPARK_LM_CTA_WARPS)
    {
        int32_t cache_index;

        cache_index = __ldg(
            topk_idxs + ((uint64_t)row * topk) + selected_slot);
        if (cache_index >= 0)
        {
            float local_logit[heads_per_cta];
            float logit[heads_per_cta];
            float rescale[heads_per_cta];
            float weight[heads_per_cta];
            float2 selected_values[maximum_pairs_per_lane];
            uint32_t local_cache_index;
            uint32_t physical_page;
            uint64_t cache_vector_base;

            local_cache_index = (uint32_t)cache_index;
            physical_page = row_lane_indices[row];
            if (local_cache_index >= SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS)
            {
                uint32_t compressed_index;
                uint32_t source_page;

                if (compressed_entries_per_page == 0u)
                {
                    continue;
                }
                compressed_index = local_cache_index -
                    SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS;
                source_page = compressed_index / compressed_entries_per_page;
                local_cache_index = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS +
                    (compressed_index % compressed_entries_per_page);
                physical_page = physical_page_table[
                    ((uint64_t)page_ordinal * page_table_stride) + source_page];
            }
            cache_vector_base = ((uint64_t)physical_page *
                lane_stride_elements) + ((uint64_t)local_cache_index * head_dim);
            for (local_head = 0u;
                 local_head < heads_per_cta;
                 ++local_head)
            {
                local_logit[local_head] = 0.0f;
            }
            for (pair_index = 0u;
                 pair_index < pairs_per_lane;
                 ++pair_index)
            {
                uint32_t value_pair_index;

                value_pair_index =
                    (pair_index * SPARK_LM_WARP_LANES) + lane_index;
                selected_values[pair_index] = make_float2(0.0f,0.0f);
                if (value_pair_index < (head_dim >> 1u))
                {
                    uint32_t query_element;

                    selected_values[pair_index] = SparkLmLoadBf16Pair(
                        kv_cache_bf16,
                        (cache_vector_base >> 1u) + value_pair_index);
                    query_element = value_pair_index << 1u;
                    for (local_head = 0u;
                         local_head < active_head_count;
                         ++local_head)
                    {
                        local_logit[local_head] = fmaf(
                            query_shared[
                                (local_head * head_dim) + query_element],
                            selected_values[pair_index].x,
                            local_logit[local_head]);
                        local_logit[local_head] = fmaf(
                            query_shared[
                                (local_head * head_dim) + query_element + 1u],
                            selected_values[pair_index].y,
                            local_logit[local_head]);
                    }
                }
            }
            for (local_head = 0u;
                 local_head < active_head_count;
                 ++local_head)
            {
                logit[local_head] = __shfl_sync(
                    0xffffffffu,
                    SparkLmWarpReduceSum(local_logit[local_head]),
                    0) * scale;
                rescale[local_head] = 0.0f;
                weight[local_head] = 0.0f;
                if (lane_index == 0u)
                {
                    rescale[local_head] =
                        logit[local_head] > running_max[local_head]
                        ? __expf(running_max[local_head] - logit[local_head])
                        : 1.0f;
                    weight[local_head] =
                        logit[local_head] > running_max[local_head]
                        ? 1.0f
                        : __expf(logit[local_head] - running_max[local_head]);
                    running_max[local_head] =
                        logit[local_head] > running_max[local_head]
                        ? logit[local_head]
                        : running_max[local_head];
                    running_denominator[local_head] = fmaf(
                        running_denominator[local_head],
                        rescale[local_head],
                        weight[local_head]);
                }
                rescale[local_head] = __shfl_sync(
                    0xffffffffu,rescale[local_head],0);
                weight[local_head] = __shfl_sync(
                    0xffffffffu,weight[local_head],0);
            }
            for (pair_index = 0u;
                 pair_index < pairs_per_lane;
                 ++pair_index)
            {
                uint32_t value_pair_index;

                value_pair_index =
                    (pair_index * SPARK_LM_WARP_LANES) + lane_index;
                if (value_pair_index < (head_dim >> 1u))
                {
                    for (local_head = 0u;
                         local_head < active_head_count;
                         ++local_head)
                    {
                        accumulator[local_head][pair_index].x = fmaf(
                            accumulator[local_head][pair_index].x,
                            rescale[local_head],
                            weight[local_head] *
                                selected_values[pair_index].x);
                        accumulator[local_head][pair_index].y = fmaf(
                            accumulator[local_head][pair_index].y,
                            rescale[local_head],
                            weight[local_head] *
                                selected_values[pair_index].y);
                    }
                }
            }
        }
    }

    if (lane_index == 0u)
    {
        for (local_head = 0u;
             local_head < active_head_count;
             ++local_head)
        {
            merge_max[(local_head * SPARK_LM_CTA_WARPS) + warp_index] =
                running_max[local_head];
            merge_den[(local_head * SPARK_LM_CTA_WARPS) + warp_index] =
                running_denominator[local_head];
        }
    }
    for (local_head = 0u;
         local_head < active_head_count;
         ++local_head)
    {
        for (pair_index = 0u;
             pair_index < pairs_per_lane;
             ++pair_index)
        {
            uint32_t value_pair_index;

            value_pair_index =
                (pair_index * SPARK_LM_WARP_LANES) + lane_index;
            if (value_pair_index < (head_dim >> 1u))
            {
                uint32_t output_element;
                uint64_t merge_base;

                output_element = value_pair_index << 1u;
                merge_base =
                    (((uint64_t)local_head * SPARK_LM_CTA_WARPS) +
                     warp_index) *
                    head_dim;
                merge_accumulator[merge_base + output_element] =
                    accumulator[local_head][pair_index].x;
                merge_accumulator[merge_base + output_element + 1u] =
                    accumulator[local_head][pair_index].y;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < active_head_count)
    {
        float block_max;
        float block_denominator;
        uint32_t partial_index;
        uint32_t actual_head;

        local_head = threadIdx.x;
        actual_head = first_head + local_head;
        block_max = merge_max[local_head * SPARK_LM_CTA_WARPS];
        for (partial_index = 1u;
             partial_index < SPARK_LM_CTA_WARPS;
             ++partial_index)
        {
            float partial_max;

            partial_max = merge_max[
                (local_head * SPARK_LM_CTA_WARPS) + partial_index];
            block_max = partial_max > block_max ? partial_max : block_max;
        }
        block_denominator = split_count == 1u && sink_f32 != 0
            ? __expf(sink_f32[actual_head] - block_max)
            : 0.0f;
        for (partial_index = 0u;
             partial_index < SPARK_LM_CTA_WARPS;
             ++partial_index)
        {
            uint32_t partial_offset;

            partial_offset =
                (local_head * SPARK_LM_CTA_WARPS) + partial_index;
            merge_scale[partial_offset] =
                __expf(merge_max[partial_offset] - block_max);
            block_denominator +=
                merge_den[partial_offset] * merge_scale[partial_offset];
        }
        inverse_denominator[local_head] = block_denominator > 0.0f
            ? 1.0f / block_denominator
            : 0.0f;
        if (split_count > 1u)
        {
            uint64_t partial_block =
                (((uint64_t)row * gridDim.y + blockIdx.y) * split_count +
                 split_index) * SPARK_DSV4_SPARSE_ATTN_PARTIAL_SCALARS;
            partials_f32[partial_block + local_head] = block_max;
            partials_f32[partial_block + heads_per_cta + local_head] =
                block_denominator;
        }
    }
    __syncthreads();

    element_index = threadIdx.x;
    while (element_index < active_head_count * head_dim)
    {
        float merged_value;
        uint32_t output_element;
        uint32_t actual_head;
        uint32_t partial_index;

        local_head = element_index / head_dim;
        output_element = element_index - (local_head * head_dim);
        actual_head = first_head + local_head;
        merged_value = 0.0f;
        for (partial_index = 0u;
             partial_index < SPARK_LM_CTA_WARPS;
             ++partial_index)
        {
            uint64_t merge_base;
            uint32_t partial_offset;

            merge_base =
                (((uint64_t)local_head * SPARK_LM_CTA_WARPS) +
                 partial_index) *
                head_dim;
            partial_offset =
                (local_head * SPARK_LM_CTA_WARPS) + partial_index;
            merged_value = fmaf(
                merge_accumulator[merge_base + output_element],
                merge_scale[partial_offset],
                merged_value);
        }
        if (split_count > 1u)
        {
            uint64_t partial_block =
                (((uint64_t)row * gridDim.y + blockIdx.y) * split_count +
                 split_index) * SPARK_DSV4_SPARSE_ATTN_PARTIAL_SCALARS;
            partials_f32[partial_block + 2u * heads_per_cta +
                (uint64_t)local_head * head_dim + output_element] = merged_value;
        }
        else
            SparkLmFloatToBf16(
                out_bf16,
                (((uint64_t)row * head_count) + actual_head) * head_dim +
                    output_element,
                merged_value * inverse_denominator[local_head]);
        element_index += blockDim.x;
    }
}

static __global__ void SparkDsv4SparseAttnMergeKernel(const float *partials_f32, const uint32_t *valid_topk_counts, const float *sink_f32, void *out_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t split_count)
{
    __shared__ float maxima[SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA];
    __shared__ float inverse[SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA];
    __shared__ float scales[SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA * SPARK_DSV4_SPARSE_ATTN_MAX_SPLITS];
    uint32_t row = blockIdx.x,first_head = blockIdx.y * SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA;
    uint32_t active_heads,active_split_count,entry = threadIdx.x,local_head,split,element;
    uint64_t block;
    float maximum,total,value;
    if ( row >= row_count || first_head >= head_count )
        return;
    active_split_count = SparkDsv4SparseAttnActiveSplitCount(
        __ldg(valid_topk_counts + row),split_count);
    active_heads = head_count - first_head;
    if ( active_heads > SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA )
        active_heads = SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA;
    if ( entry < active_heads )
    {
        maximum = -3.0e38f;
        for (split=0u; split<active_split_count; split++)
        {
            block = (((uint64_t)row * gridDim.y + blockIdx.y) * split_count + split) * SPARK_DSV4_SPARSE_ATTN_PARTIAL_SCALARS;
            maximum = fmaxf(maximum,partials_f32[block + entry]);
        }
        maxima[entry] = maximum;
    }
    __syncthreads();
    if ( entry < active_heads * active_split_count )
    {
        local_head = entry / active_split_count;
        split = entry % active_split_count;
        block = (((uint64_t)row * gridDim.y + blockIdx.y) * split_count + split) * SPARK_DSV4_SPARSE_ATTN_PARTIAL_SCALARS;
        scales[entry] = __expf(partials_f32[block + local_head] - maxima[local_head]);
    }
    __syncthreads();
    if ( entry < active_heads )
    {
        total = sink_f32 != 0 ? __expf(sink_f32[first_head + entry] - maxima[entry]) : 0.0f;
        for (split=0u; split<active_split_count; split++)
        {
            block = (((uint64_t)row * gridDim.y + blockIdx.y) * split_count + split) * SPARK_DSV4_SPARSE_ATTN_PARTIAL_SCALARS;
            total += partials_f32[block + SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA + entry] * scales[entry * active_split_count + split];
        }
        inverse[entry] = total > 0.0f ? 1.0f / total : 0.0f;
    }
    __syncthreads();
    for (element=entry; element<active_heads * head_dim; element+=blockDim.x)
    {
        local_head = element / head_dim;
        value = 0.0f;
        for (split=0u; split<active_split_count; split++)
        {
            block = (((uint64_t)row * gridDim.y + blockIdx.y) * split_count + split) * SPARK_DSV4_SPARSE_ATTN_PARTIAL_SCALARS;
            value = fmaf(partials_f32[block + 2u * SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA + (uint64_t)local_head * head_dim + element % head_dim],scales[local_head * active_split_count + split],value);
        }
        SparkLmFloatToBf16(out_bf16,((uint64_t)row * head_count + first_head + local_head) * head_dim + element % head_dim,value * inverse[local_head]);
    }
}

// Softmax pooling of one output channel over the pool slots: -inf scores
// drop out; shared by decode and prefill compressor forms.
static __device__ __forceinline__ float SparkDsv4PoolChannel(const float *kv, const float *score, uint32_t slots, uint32_t stride, uint32_t channel)
{
	uint32_t slot;
	float maximum = -3.0e38f,total = 0.0f,value = 0.0f,weight;
	for (slot = 0; slot < slots; slot++)
		if ( score[slot * stride + channel] > maximum )
			maximum = score[slot * stride + channel];
	for (slot = 0; slot < slots; slot++)
	{
		if ( score[slot * stride + channel] <= -3.0e38f )
			continue;
		weight = __expf(score[slot * stride + channel] - maximum);
		total += weight;
		value += weight * kv[slot * stride + channel];
	}
	return(value / total);
}

// The overlap gather pool: 2*ratio slots where slot i < ratio reads the
// previous group's FIRST channel half and slot i >= ratio the current
// group's SECOND half - the concatenation the reference builds before its
// single softmax pool.
static __device__ __forceinline__ float SparkDsv4PoolOverlapChannel(const float *kv_state, const float *score_state, uint32_t ratio, uint32_t channels, uint32_t width, uint32_t channel)
{
	uint32_t slot,source;
	float maximum = -3.0e38f,total = 0.0f,value = 0.0f,weight,score;
	for (slot = 0; slot < 2u * ratio; slot++)
	{
		source = slot < ratio ? slot * channels + channel : slot * channels + width + channel;
		if ( score_state[source] > maximum )
			maximum = score_state[source];
	}
	for (slot = 0; slot < 2u * ratio; slot++)
	{
		source = slot < ratio ? slot * channels + channel : slot * channels + width + channel;
		score = score_state[source];
		if ( score <= -3.0e38f )
			continue;
		weight = __expf(score - maximum);
		total += weight;
		value += weight * kv_state[source];
	}
	return(value / total);
}

/*
 * One CTA owns each live lane and advances its rows in packet order. This
 * keeps different lanes parallel without racing a lane's recurrent state.
 * State layout per lane: [coff*ratio slots][coff*d ch] f32.
 */
static __global__ void SparkDsv4CompressStepKernel(const void *kv_bf16,
	const void *score_bf16,const float *ape_f32,float *kv_state_f32,
	float *score_state_f32,uint64_t state_lane_stride,
	const uint32_t *row_lane_indices,const uint64_t *row_positions,
	uint32_t row_count,uint32_t ratio,uint32_t overlapped,uint32_t width,
	void *emit_bf16,uint32_t *emitted)
{
	uint32_t row = blockIdx.x,coff = overlapped != 0u ? 2u : 1u,channels = coff * width,channel;
	uint32_t slot,boundary,lane,previous;
	uint64_t ape_base,source,state_base;
	float *kv_state,*score_state;
	float pooled;
	if ( row >= row_count )
		return;
	lane = row_lane_indices[row];
	for (previous=0u; previous<row; previous++)
		if ( row_lane_indices[previous] == lane )
			return;
	state_base = (uint64_t)lane * state_lane_stride;
	kv_state = kv_state_f32 + state_base;
	score_state = score_state_f32 + state_base;
	for (; row<row_count; row++)
	{
		if ( row_lane_indices[row] != lane )
			continue;
		slot = (uint32_t)(row_positions[row] % ratio);
		boundary = (row_positions[row] + 1u) % ratio == 0u ? 1u : 0u;
		ape_base = (row_positions[row] % ratio) * channels;
		for (channel=threadIdx.x; channel<channels; channel+=blockDim.x)
		{
			source = (uint64_t)row * channels + channel;
			kv_state[((overlapped != 0u ? ratio : 0u) + slot) * channels +
				channel] = SparkLmBf16ToFloat(kv_bf16,source);
			score_state[((overlapped != 0u ? ratio : 0u) + slot) * channels +
				channel] = SparkLmBf16ToFloat(score_bf16,source) +
				__ldg(ape_f32 + ape_base + channel);
		}
		__syncthreads();
		if ( threadIdx.x == 0u )
			emitted[row] = boundary;
		for (channel = threadIdx.x; channel < width; channel += blockDim.x)
		{
			pooled = 0.0f;
			if ( boundary != 0u )
				pooled = overlapped != 0u ? SparkDsv4PoolOverlapChannel(kv_state,score_state,ratio,channels,width,channel) : SparkDsv4PoolChannel(kv_state,score_state,ratio,channels,channel);
			SparkLmFloatToBf16(emit_bf16,(uint64_t)row * width + channel,pooled);
		}
		__syncthreads();
		if ( overlapped != 0u && boundary != 0u )
			for (channel=threadIdx.x; channel<ratio * channels; channel+=blockDim.x)
			{
				kv_state[channel] = kv_state[ratio * channels + channel];
				score_state[channel] = score_state[ratio * channels + channel];
			}
		__syncthreads();
	}
}

static __device__ __forceinline__ void SparkDsv4CacheScatterRow(
	const uint16_t *source_bf16,
	uint16_t *cache_bf16,
	uint64_t cache_lane_stride,
	const uint32_t *row_lane_indices,
	const uint64_t *row_positions,
	uint32_t row,
	uint32_t width,
	uint64_t base_slot,
	uint32_t ratio,
	uint32_t ring_slots)
{
	uint32_t element;
	uint64_t destination,position,slot,source;
	position = row_positions[row];
	slot = ratio != 0u ? base_slot + ((position % ring_slots) / ratio) : position % ring_slots;
	destination = (uint64_t)row_lane_indices[row] * cache_lane_stride + slot * width;
	source = (uint64_t)row * width;
	for (element=threadIdx.x; element<width; element+=blockDim.x)
		cache_bf16[destination + element] = source_bf16[source + element];
}

static __global__ void SparkDsv4CacheScatterKernel(
	const uint16_t *source_bf16,
	const uint32_t *emitted,
	uint16_t *cache_bf16,
	uint64_t cache_lane_stride,
	const uint32_t *row_lane_indices,
	const uint64_t *row_positions,
	uint32_t row_count,
	uint32_t width,
	uint64_t base_slot,
	uint32_t ratio,
	uint32_t ring_slots)
{
	uint32_t row = blockIdx.x;
	if ( row >= row_count || (ratio != 0u && emitted[row] == 0u) )
		return;
	SparkDsv4CacheScatterRow(source_bf16,cache_bf16,cache_lane_stride,
		row_lane_indices,row_positions,row,width,base_slot,ratio,ring_slots);
}

/*
 * One boundary-predicated CTA replaces the five post-compressor launches.
 * Every emitting row retains each BF16 materialization boundary and the
 * reference operation order. A non-emitting CTA returns before touching the
 * staging row or cache, so graph replay never needs a host-side predicate.
 */
static __global__ void SparkDsv4KvEmissionKernel(
	void *emit_bf16,const uint32_t *emitted,const void *norm_weight_bf16,
	const float *freqs_f32,const uint64_t *row_emit_positions,
	uint16_t *cache_bf16,uint64_t cache_lane_stride,
	const uint32_t *row_lane_indices,const uint64_t *row_positions,
	uint32_t row_count,uint32_t width,uint64_t base_slot,uint32_t ratio,
	uint32_t ring_slots,uint32_t rotate)
{
	extern __shared__ float emission_shared[];
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x,warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES,group,group_count;
	uint32_t quant_block,quant_width;
	float format_max;
	if ( row >= row_count || (ratio != 0u && emitted[row] == 0u) )
		return;
	SparkLmRmsNormRow(emit_bf16,norm_weight_bf16,emit_bf16,row,width,
		SPARK_DSV4_MODEL_RMS_NORM_EPSILON,emission_shared,reduce_scratch);
	__syncthreads();
	SparkDsv4RopeRow(emit_bf16,freqs_f32,row_emit_positions,row,0u,1u,
		width,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	__syncthreads();
	if ( rotate != 0u )
		SparkDsv4HadamardRow(emit_bf16,row,width,emission_shared);
	__syncthreads();
	quant_width = rotate != 0u ? width :
		width - SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION;
	quant_block = rotate != 0u ? SPARK_DSV4_MODEL_FP4_QUANT_BLOCK :
		SPARK_DSV4_MODEL_KV_QUANT_BLOCK;
	format_max = rotate != 0u ? SPARK_DSV4_MODEL_FP4_MAX :
		SPARK_DSV4_MODEL_FP8_MAX;
	group_count = (quant_width + quant_block - 1u) / quant_block;
	for (group=warp; group<group_count; group+=SPARK_LM_CTA_WARPS)
		SparkDsv4QuantSimGroup(emit_bf16,row,group,lane,width,quant_width,
			quant_block,format_max,rotate);
	__syncthreads();
	SparkDsv4CacheScatterRow((const uint16_t *)emit_bf16,cache_bf16,
		cache_lane_stride,row_lane_indices,row_positions,row,width,base_slot,
		ratio,ring_slots);
}

static __global__ void SparkDsv4InitializePagesKernel(
	uint32_t *page_pool,
	uint64_t page_stride_words,
	const uint32_t *page_indices,
	const uint32_t *parent_page_indices,
	uint32_t page_count,
	const SparkDsv4PagedScoreSpan *score_spans,
	uint32_t score_span_count)
{
	float *score_base;
	uint32_t page,parent,span_index;
	uint64_t destination,source,word;
	if ( blockIdx.x >= page_count )
		return;
	page = page_indices[blockIdx.x];
	if ( page == UINT32_MAX )
		return;
	parent = parent_page_indices[blockIdx.x];
	destination = (uint64_t)page * page_stride_words;
	source = (uint64_t)parent * page_stride_words;
	for (word=threadIdx.x; word<page_stride_words; word+=blockDim.x)
		page_pool[destination + word] = parent == UINT32_MAX ? 0u : page_pool[source + word];
	__syncthreads();
	if ( parent != UINT32_MAX )
		return;
	score_base = (float *)(page_pool + destination);
	for (span_index=0u; span_index<score_span_count; span_index++)
		for (word=threadIdx.x; word<score_spans[span_index].element_count;
			word+=blockDim.x)
			score_base[score_spans[span_index].offset_words + word] = -INFINITY;
}

static __global__ void SparkDsv4UpdatePageTableKernel(
	uint32_t *page_table,
	const uint32_t *update_indices,
	const uint32_t *update_values,
	uint32_t update_count)
{
	uint32_t update;
	update = blockIdx.x * blockDim.x + threadIdx.x;
	if ( update < update_count )
		page_table[update_indices[update]] = update_values[update];
}

static __global__ void SparkDsv4BuildAttentionIndicesKernel(
    const uint64_t *row_positions,
    int32_t *indices,
    uint32_t *slot_counts,
    uint32_t *attention_slot_counts,
    uint32_t row_count,
	uint32_t column_count,
	uint32_t index_slot_capacity,
	uint32_t layer_kind)
{
	uint32_t attention_slots,column,compressed,row,valid_index_slots,window;
	uint64_t position;
	row = blockIdx.x;
	if ( row >= row_count )
		return;
	position = row_positions[row];
	window = position + 1u < SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS ? (uint32_t)(position + 1u) : SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS;
	compressed = layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA ? (uint32_t)((position + 1u) / SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO) : 0u;
	valid_index_slots = layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? (uint32_t)((position + 1u) / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO) : 0u;
	valid_index_slots = min(valid_index_slots,index_slot_capacity);
	attention_slots = window;
	if ( layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA )
		attention_slots += compressed;
	else if ( layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		attention_slots = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS +
			min(valid_index_slots,SPARK_DSV4_MODEL_INDEX_TOP_K);
	attention_slots = min(attention_slots,column_count);
	if ( threadIdx.x == 0u )
	{
		slot_counts[row] = valid_index_slots;
		attention_slot_counts[row] = attention_slots;
	}
	for (column=threadIdx.x; column<column_count; column+=blockDim.x)
	{
		if ( column < window )
			indices[(uint64_t)row * column_count + column] = (int32_t)SparkDsv4AttentionWindowSlot(position,column,SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS);
		else if ( layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA && column < window + compressed )
			indices[(uint64_t)row * column_count + column] = (int32_t)(SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS + column - window);
		else
			indices[(uint64_t)row * column_count + column] = -1;
	}
}

// The gate scores: linear against the router weight in fp32 with
// sqrtsoftplus applied.  B1 used to put all 256 experts behind one warp
// loop per row.  Keep the shared activation broadcast, but make the expert
// tile the grid-y axis so every warp owns one expert and the launch exposes
// the full SM parallelism without changing the arithmetic.
static __device__ __forceinline__ void SparkDsv4GateScoreExpert(
	const float *gate_shared,const void *weight_bf16,float *scores_f32,
	uint32_t input_dimension,uint32_t expert_count,uint32_t expert,
	uint32_t lane)
{
	float accumulator;
	if ( expert >= expert_count )
		return;
	accumulator = SparkLmDotRowBf16(gate_shared,weight_bf16,expert,
		input_dimension,lane);
	accumulator = SparkLmWarpReduceSum(accumulator);
	if ( lane == 0u )
		scores_f32[expert] = sqrtf(SparkLmSoftplus(accumulator));
}

static __global__ void SparkDsv4GateScoresKernel(const void *weight_bf16, const void *input_bf16, float *scores_f32, uint32_t row_count, uint32_t input_dimension, uint32_t expert_count)
{
	extern __shared__ float gate_shared[];
	uint32_t row = blockIdx.x,warp_count = blockDim.x / SPARK_LM_WARP_LANES;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t expert = blockIdx.y * warp_count + warp,element;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < input_dimension; element += blockDim.x)
		gate_shared[element] = SparkLmBf16ToFloat(input_bf16,((uint64_t)row * input_dimension) + element);
	__syncthreads();
	SparkDsv4GateScoreExpert(gate_shared,weight_bf16,
		scores_f32 + (uint64_t)row * expert_count,input_dimension,expert_count,
		expert,lane);
}

/*
 * noaux_tc selection and the hash path share one CTA per row.  The table
 * path copies the pinned expert ids directly; the score path performs exact
 * block-parallel top-k over scores plus bias with lower-index tie breaking.
 * Weights gather original scores, sum-normalize, and apply the route scale.
 */
static __global__ void SparkDsv4GateSelectKernel(
    const float *scores_f32,
    const float *bias_f32,
    const uint32_t *tid2eid_u32,
    const uint32_t *token_ids,
    uint32_t row_count,
    uint32_t expert_count,
    uint32_t topk,
    float route_scale,
    uint32_t *indices_u32,
    float *weights_f32)
{
    __shared__ uint64_t ordered_keys[SPARK_DSV4_ROUTER_SORT_CAPACITY];
    const float *row_scores;
    uint64_t selected_key;
    uint32_t row;
    uint32_t expert;
    uint32_t rank;
    uint32_t selected_expert;
    float selected_score;
    float selected_total;

    static_assert(
        SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT <=
            SPARK_DSV4_ROUTER_SORT_CAPACITY,
        "DSV4 expert count exceeds router sort capacity");
    static_assert(
        SPARK_LM_MOE_MAX_TOPK <= SPARK_LM_WARP_LANES,
        "DSV4 router normalization requires one warp");
    row = blockIdx.x;
    if (row >= row_count ||
        expert_count == 0u ||
        expert_count > SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT ||
        topk == 0u || topk > SPARK_LM_MOE_MAX_TOPK ||
        topk > expert_count)
    {
        return;
    }
    row_scores = scores_f32 + ((uint64_t)row * expert_count);
    rank = threadIdx.x;
    if (tid2eid_u32 != 0)
    {
        selected_expert = rank < topk
            ? tid2eid_u32[
                ((uint64_t)token_ids[row] * topk) + rank]
            : UINT32_MAX;
    }
    else
    {
        for (expert = threadIdx.x;
             expert < SPARK_DSV4_ROUTER_SORT_CAPACITY;
             expert += blockDim.x)
        {
            float choice_score;

            choice_score = expert < expert_count
                ? row_scores[expert] +
                    (bias_f32 != 0 ? bias_f32[expert] : 0.0f)
                : NAN;
            ordered_keys[expert] = expert < expert_count
                ? SparkLmOrderedTopKKey(choice_score, expert)
                : 0u;
        }
        __syncthreads();
        SparkLmBitonicSortKeysAscending<SPARK_DSV4_ROUTER_SORT_CAPACITY>(
            ordered_keys);
        selected_key = rank < topk
            ? ordered_keys[SPARK_DSV4_ROUTER_SORT_CAPACITY - 1u - rank]
            : 0u;
        selected_expert = selected_key != 0u
            ? 0xffffffffu - (uint32_t)selected_key
            : UINT32_MAX;
    }
    selected_score = rank < topk && selected_expert < expert_count
        ? row_scores[selected_expert]
        : 0.0f;
    if (threadIdx.x < SPARK_LM_WARP_LANES)
    {
        selected_total = SparkLmWarpReduceSum(selected_score);
        selected_total = __shfl_sync(0xffffffffu, selected_total, 0u);
        if (rank < topk)
        {
            indices_u32[((uint64_t)row * topk) + rank] = selected_expert;
            weights_f32[((uint64_t)row * topk) + rank] =
                selected_total > 0.0f
                ? selected_score / selected_total * route_scale
                : 0.0f;
        }
    }
}

static __device__ void SparkDsv4GateSelectShared(
	const float *scores_f32,const float *bias_f32,
	const uint32_t *tid2eid_u32,const uint32_t *token_ids,uint32_t topk,
	float route_scale,uint32_t *indices_u32,float *weights_f32,
	uint64_t *ordered_keys,uint32_t *selected_u32)
{
	uint64_t selected_key;
	uint32_t expert,rank = threadIdx.x,selected_expert;
	float selected_score,selected_total;
	if ( tid2eid_u32 != 0 )
		selected_expert = rank < topk ?
			tid2eid_u32[(uint64_t)token_ids[0] * topk + rank] : UINT32_MAX;
	else
	{
		for (expert=threadIdx.x; expert<SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT;
			expert+=blockDim.x)
			ordered_keys[expert] = SparkLmOrderedTopKKey(scores_f32[expert] +
				(bias_f32 != 0 ? bias_f32[expert] : 0.0f),expert);
		__syncthreads();
		SparkLmBitonicSortKeysAscending<SPARK_DSV4_ROUTER_SORT_CAPACITY>(
			ordered_keys);
		selected_key = rank < topk ? ordered_keys[
			SPARK_DSV4_ROUTER_SORT_CAPACITY - 1u - rank] : 0u;
		selected_expert = selected_key != 0u ?
			UINT32_MAX - (uint32_t)selected_key : UINT32_MAX;
	}
	selected_score = rank < topk &&
		selected_expert < SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT ?
		scores_f32[selected_expert] : 0.0f;
	if ( threadIdx.x < SPARK_LM_WARP_LANES )
	{
		selected_total = SparkLmWarpReduceSum(selected_score);
		selected_total = __shfl_sync(0xffffffffu,selected_total,0u);
		if ( rank < topk )
		{
			indices_u32[rank] = selected_expert;
			selected_u32[rank] = selected_expert;
			weights_f32[rank] = selected_total > 0.0f ?
				selected_score / selected_total * route_scale : 0.0f;
		}
	}
}

static __device__ void SparkDsv4GateRouteBuildShared(
	const uint32_t *route_expert,uint32_t topk,uint32_t *group_row_offset,
	uint32_t *route_packed_row,uint32_t *route_source_token,uint32_t tile_m,
	uint32_t neuron_tiles_up,uint32_t *tile_prefix_up,
	uint32_t neuron_tiles_down,uint32_t *tile_prefix_down,uint32_t *count)
{
	uint32_t index,expert,packed;
	for (index=threadIdx.x; index<SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT;
		index+=blockDim.x)
		count[index] = 0u;
	__syncthreads();
	for (index=threadIdx.x; index<topk; index+=blockDim.x)
		atomicAdd(&count[route_expert[index]],1u);
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		uint32_t total = 0u,held,up = 0u,down = 0u,row_tiles;
		for (index=0u; index<SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT; index++)
		{
			held = count[index];
			group_row_offset[index] = total;
			count[index] = total;
			total += held;
			row_tiles = (held + tile_m - 1u) / tile_m;
			tile_prefix_up[index] = up;
			tile_prefix_down[index] = down;
			up += row_tiles * neuron_tiles_up;
			down += row_tiles * neuron_tiles_down;
		}
		group_row_offset[SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT] = total;
		tile_prefix_up[SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT] = up;
		tile_prefix_down[SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT] = down;
	}
	__syncthreads();
	for (index=threadIdx.x; index<topk; index+=blockDim.x)
	{
		expert = route_expert[index];
		packed = atomicAdd(&count[expert],1u);
		route_packed_row[index] = packed;
		route_source_token[packed] = 0u;
	}
}

static __global__ void SparkDsv4GateRouteCooperativeKernel(
	const void *weight_bf16,const void *input_bf16,float *scores_f32,
	const float *bias_f32,const uint32_t *tid2eid_u32,
	const uint32_t *token_ids,uint32_t topk,float route_scale,
	uint32_t *indices_u32,float *weights_f32,uint32_t *group_row_offset,
	uint32_t *route_packed_row,uint32_t *route_source_token,uint32_t tile_m,
	uint32_t neuron_tiles_up,uint32_t *tile_prefix_up,
	uint32_t neuron_tiles_down,uint32_t *tile_prefix_down)
{
	extern __shared__ float gate_shared[];
	__shared__ uint32_t selected_u32[SPARK_LM_MOE_MAX_TOPK];
	cooperative_groups::grid_group grid = cooperative_groups::this_grid();
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t expert = blockIdx.x * SPARK_LM_CTA_WARPS + warp,element;
	for (element=threadIdx.x; element<SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
		element+=blockDim.x)
		gate_shared[element] = SparkLmBf16ToFloat(input_bf16,element);
	__syncthreads();
	SparkDsv4GateScoreExpert(gate_shared,weight_bf16,scores_f32,
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION,
		SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT,expert,lane);
	grid.sync();
	if ( blockIdx.x != 0u )
		return;
	SparkDsv4GateSelectShared(scores_f32,bias_f32,tid2eid_u32,token_ids,
		topk,route_scale,indices_u32,weights_f32,(uint64_t *)gate_shared,
		selected_u32);
	__syncthreads();
	SparkDsv4GateRouteBuildShared(selected_u32,topk,group_row_offset,
		route_packed_row,route_source_token,tile_m,neuron_tiles_up,
		tile_prefix_up,neuron_tiles_down,tile_prefix_down,
		(uint32_t *)gate_shared);
}

// The swiglu clamp on gathered gate/up rows, routing weight folded in:
// up two-sided, gate max-only, silu(gate)*up in fp32.
static __global__ void SparkDsv4SwigluClampKernel(const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width, float limit, const float *row_weights_f32, const uint32_t *weight_map)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = (uint64_t)row * width;
	float weight;
	float2 gate_pair,up_pair;
	if ( row >= row_count )
		return;
	weight = row_weights_f32 != 0 ? row_weights_f32[weight_map != 0 ? weight_map[row] : row] : 1.0f;
	for (element = threadIdx.x; element < (width >> 1u); element += blockDim.x)
	{
		gate_pair = SparkLmLoadBf16Pair(gate_bf16,(offset >> 1u) + element);
		up_pair = SparkLmLoadBf16Pair(up_bf16,(offset >> 1u) + element);
		if ( limit > 0.0f )
		{
			up_pair.x = up_pair.x > limit ? limit : (up_pair.x < -limit ? -limit : up_pair.x);
			up_pair.y = up_pair.y > limit ? limit : (up_pair.y < -limit ? -limit : up_pair.y);
			gate_pair.x = gate_pair.x > limit ? limit : gate_pair.x;
			gate_pair.y = gate_pair.y > limit ? limit : gate_pair.y;
		}
		SparkLmStoreBf16Pair(up_bf16,(offset >> 1u) + element,SparkLmSwish(gate_pair.x) * up_pair.x * weight,SparkLmSwish(gate_pair.y) * up_pair.y * weight);
	}
}

static __global__ void SparkDsv4AccumAddKernel(void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float2 destination_pair,source_pair;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (width >> 1u); element += blockDim.x)
	{
		destination_pair = SparkLmLoadBf16Pair(destination_bf16,offset + element);
		source_pair = SparkLmLoadBf16Pair(source_bf16,offset + element);
		SparkLmStoreBf16Pair(destination_bf16,offset + element,destination_pair.x + source_pair.x,destination_pair.y + source_pair.y);
	}
}

static __global__ void SparkDsv4AccumAddRelayKernel(void *destination_bf16,
	const void *source_bf16,void *relay_bf16,uint32_t row_count,
	uint32_t width)
{
	__nv_bfloat162 packed;
	uint32_t raw,row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float2 destination_pair,source_pair;
	if ( row >= row_count )
		return;
	for (element=threadIdx.x; element<(width >> 1u); element+=blockDim.x)
	{
		destination_pair = SparkLmLoadBf16Pair(destination_bf16,offset + element);
		source_pair = SparkLmLoadBf16Pair(source_bf16,offset + element);
		packed = __floats2bfloat162_rn(destination_pair.x + source_pair.x,
			destination_pair.y + source_pair.y);
		raw = *(const uint32_t *)&packed;
		((uint32_t *)destination_bf16)[offset + element] = raw;
		((uint32_t *)relay_bf16)[offset + element] = raw;
	}
}

static __global__ void SparkDsv4AccumAddTp4TreeKernel(
	void *destination_bf16,
	const void *rank0_bf16,
	const void *rank1_bf16,
	const void *rank2_bf16,
	const void *rank3_bf16,
	uint32_t tp_rank,
	uint32_t row_count,
	uint32_t width)
{
	__nv_bfloat162 pair01_bf16,pair23_bf16;
	float2 pair01,pair23,rank0,rank1,rank2,rank3;
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	if ( row >= row_count )
		return;
	for (element=threadIdx.x; element<(width >> 1u); element+=blockDim.x)
	{
		rank0 = SparkLmLoadBf16Pair(rank0_bf16,offset + element);
		rank1 = SparkLmLoadBf16Pair(rank1_bf16,offset + element);
		rank2 = SparkLmLoadBf16Pair(rank2_bf16,offset + element);
		rank3 = SparkLmLoadBf16Pair(rank3_bf16,offset + element);
		pair01_bf16 = __floats2bfloat162_rn(
			rank0.x + rank1.x,rank0.y + rank1.y);
		pair23_bf16 = __floats2bfloat162_rn(
			rank2.x + rank3.x,rank2.y + rank3.y);
		pair01 = __bfloat1622float2(pair01_bf16);
		pair23 = __bfloat1622float2(pair23_bf16);
		if ( tp_rank < 2u )
			SparkLmStoreBf16Pair(destination_bf16,offset + element,
				pair01.x + pair23.x,pair01.y + pair23.y);
		else
			SparkLmStoreBf16Pair(destination_bf16,offset + element,
				pair23.x + pair01.x,pair23.y + pair01.y);
	}
}

// The indexer score: relu(q_h . kv) per head times the projected head
// weight, summed over heads - one warp per slot, lanes over dims.
static __global__ void SparkDsv4IndexerScoreKernel(const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_page_table_indices, const uint32_t *physical_page_table, uint32_t page_table_stride, uint32_t entries_per_page, const uint32_t *slot_counts, const float *head_weights_f32, float *scores_f32, uint32_t row_count, uint32_t max_slots, uint32_t head_count, uint32_t head_dim)
{
    static const uint32_t maximum_pairs_per_lane =
        SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION /
        (2u * SPARK_LM_WARP_LANES);
    extern __shared__ float q_shared[];
    uint32_t row;
    uint32_t warp;
    uint32_t lane;
    uint32_t slot;
    uint32_t head;
    uint32_t pair_index;
    uint32_t value_pair_index;
    uint32_t bounded_slot_count;
    uint32_t physical_page;
    uint32_t query_element;
	uint32_t cta_first_slot;
    uint64_t q_base;
    uint64_t kv_base;
    float total;
    float accumulator;
    float2 q_pair;
    float2 key_pair[maximum_pairs_per_lane];

    row = blockIdx.x;
    if (row >= row_count)
    {
        return;
    }
	bounded_slot_count = slot_counts[row] < max_slots
		? slot_counts[row]
		: max_slots;
	cta_first_slot = blockIdx.y * SPARK_LM_CTA_WARPS;
	if (cta_first_slot >= bounded_slot_count)
	{
		return;
	}
	warp = threadIdx.x / SPARK_LM_WARP_LANES;
	lane = threadIdx.x % SPARK_LM_WARP_LANES;
	slot = cta_first_slot + warp;
    q_base = (uint64_t)row * head_count * head_dim;
    for (pair_index = threadIdx.x;
         pair_index < ((head_count * head_dim) >> 1u);
         pair_index += blockDim.x)
    {
        q_pair = SparkLmLoadBf16Pair(
            q_bf16,
            (q_base >> 1u) + pair_index);
        q_shared[pair_index << 1u] = q_pair.x;
        q_shared[(pair_index << 1u) + 1u] = q_pair.y;
    }
    __syncthreads();

    if (slot >= bounded_slot_count)
    {
        return;
    }

    physical_page = physical_page_table[
        ((uint64_t)row_page_table_indices[row] * page_table_stride) +
        (slot / entries_per_page)];
    kv_base = ((uint64_t)physical_page * lane_stride_elements) +
        ((uint64_t)(slot % entries_per_page) * head_dim);
    #pragma unroll
    for (pair_index = 0u;
         pair_index < maximum_pairs_per_lane;
         ++pair_index)
    {
        value_pair_index =
            (pair_index * SPARK_LM_WARP_LANES) + lane;
        key_pair[pair_index] = value_pair_index < (head_dim >> 1u)
            ? SparkLmLoadBf16Pair(
                kv_cache_bf16,
                (kv_base >> 1u) + value_pair_index)
            : make_float2(0.0f, 0.0f);
    }

    total = 0.0f;
    for (head = 0u; head < head_count; ++head)
    {
        accumulator = 0.0f;
        #pragma unroll
        for (pair_index = 0u;
             pair_index < maximum_pairs_per_lane;
             ++pair_index)
        {
            value_pair_index =
                (pair_index * SPARK_LM_WARP_LANES) + lane;
            if (value_pair_index < (head_dim >> 1u))
            {
                query_element = value_pair_index << 1u;
                accumulator = fmaf(
                    q_shared[((uint64_t)head * head_dim) + query_element],
                    key_pair[pair_index].x,
                    accumulator);
                accumulator = fmaf(
                    q_shared[
                        ((uint64_t)head * head_dim) + query_element + 1u],
                    key_pair[pair_index].y,
                    accumulator);
            }
        }
        accumulator = __shfl_sync(
            0xffffffffu,
            SparkLmWarpReduceSum(accumulator),
            0);
        if (lane == 0u && accumulator > 0.0f)
        {
            total = fmaf(
                accumulator,
                __ldg(
                    head_weights_f32 +
                    ((uint64_t)row * head_count) + head),
                total);
        }
    }
    if (lane == 0u)
    {
        scores_f32[((uint64_t)row * max_slots) + slot] = total;
    }
}

/*
 * Exact byte-radix top-k with canonical lower-slot tie breaking.
 */
static __device__ __forceinline__ uint64_t SparkDsv4OrderedTopKKey(
    float score,
    uint32_t slot)
{
    if (score <= -3.0e38f)
    {
        return 0u;
    }
    return SparkLmOrderedTopKKey(score, slot);
}

static __global__ void SparkDsv4TopKKernel(const float *scores_f32, const uint32_t *slot_counts, uint32_t max_slots, uint32_t topk, int32_t offset, int32_t *indices_out, uint64_t out_row_stride, uint32_t row_count)
{
    __shared__ uint32_t histogram[256];
    __shared__ uint64_t selected_keys[SPARK_DSV4_MODEL_INDEX_TOP_K];
    __shared__ uint64_t prefix;
    __shared__ uint64_t prefix_mask;
    __shared__ uint64_t threshold;
    __shared__ uint32_t valid_count;
    __shared__ uint32_t selected_cursor;
    __shared__ uint32_t remaining_rank;
	__shared__ uint32_t direct_sort_width;
    uint32_t row = blockIdx.x;
    uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
    uint32_t slot;
    uint32_t pass;
    uint32_t selected_count;
    uint32_t sort_width;
    uint32_t bounded_slot_count;
    const float *scores;

    if (row >= row_count || topk == 0u || topk > SPARK_DSV4_MODEL_INDEX_TOP_K)
    {
        return;
    }
    scores = scores_f32 + ((uint64_t)row * max_slots);
    bounded_slot_count = slot_counts[row] < max_slots
        ? slot_counts[row]
        : max_slots;
	if (bounded_slot_count <= topk)
	{
		if (threadIdx.x == 0u)
		{
			valid_count = 0u;
			sort_width = 1u;
			while (sort_width < bounded_slot_count)
			{
				sort_width <<= 1u;
			}
			direct_sort_width = sort_width;
		}
		__syncthreads();
		sort_width = direct_sort_width;
		for (slot = threadIdx.x; slot < sort_width; slot += blockDim.x)
		{
			uint64_t key = slot < bounded_slot_count
				? SparkDsv4OrderedTopKKey(scores[slot], slot)
				: 0u;
			selected_keys[slot] = key;
			if (key != 0u)
			{
				atomicAdd(&valid_count, 1u);
			}
		}
		__syncthreads();
		selected_count = valid_count;
		for (uint32_t size = 2u; size <= sort_width; size <<= 1u)
		{
			for (uint32_t stride = size >> 1u; stride != 0u; stride >>= 1u)
			{
				for (slot = threadIdx.x; slot < sort_width; slot += blockDim.x)
				{
					uint32_t partner = slot ^ stride;
					if (partner > slot)
					{
						uint64_t left = selected_keys[slot];
						uint64_t right = selected_keys[partner];
						uint32_t ascending = (slot & size) != 0u ? 1u : 0u;
						uint32_t swap = ascending != 0u ? left > right : left < right;
						if (swap != 0u)
						{
							selected_keys[slot] = right;
							selected_keys[partner] = left;
						}
					}
				}
				__syncthreads();
			}
		}
		for (slot = threadIdx.x; slot < topk; slot += blockDim.x)
		{
			int32_t selected_index = -1;
			if (slot < selected_count)
			{
				uint32_t original_slot = 0xffffffffu -
					(uint32_t)selected_keys[slot];
				selected_index = (int32_t)original_slot + offset;
			}
			indices_out[((uint64_t)row * out_row_stride) + slot] =
				selected_index;
		}
		return;
	}
    if (threadIdx.x == 0u)
    {
        valid_count = 0u;
        prefix = 0u;
        prefix_mask = 0u;
    }
    __syncthreads();

    for (slot = threadIdx.x; slot < bounded_slot_count; slot += blockDim.x)
    {
        if (SparkDsv4OrderedTopKKey(scores[slot], slot) != 0u)
        {
            atomicAdd(&valid_count, 1u);
        }
    }
    __syncthreads();
    selected_count = valid_count < topk ? valid_count : topk;
    if (selected_count == 0u)
    {
        for (slot = threadIdx.x; slot < topk; slot += blockDim.x)
        {
            indices_out[((uint64_t)row * out_row_stride) + slot] = -1;
        }
        return;
    }
    if (threadIdx.x == 0u)
    {
        remaining_rank = selected_count - 1u;
    }
    __syncthreads();

    for (pass = 0u; pass < 8u; pass++)
    {
        uint32_t shift = 56u - (pass * 8u);
        for (slot = threadIdx.x; slot < 256u; slot += blockDim.x)
        {
            histogram[slot] = 0u;
        }
        __syncthreads();

        for (slot = threadIdx.x; slot < bounded_slot_count; slot += blockDim.x)
        {
            uint64_t key = SparkDsv4OrderedTopKKey(scores[slot], slot);
            uint32_t matches_prefix =
                key != 0u && (key & prefix_mask) == prefix ? 1u : 0u;
            uint32_t digit = matches_prefix != 0u
                ? (uint32_t)((key >> shift) & 0xffu)
                : 0x100u + lane;
            uint32_t active_mask = __activemask();
            uint32_t peers = __match_any_sync(active_mask, digit);
            uint32_t leader = (uint32_t)(__ffs((int)peers) - 1);
            if (matches_prefix != 0u && lane == leader)
            {
                atomicAdd(&histogram[digit], (uint32_t)__popc(peers));
            }
        }
        __syncthreads();

        if (threadIdx.x == 0u)
        {
            int32_t digit;
            for (digit = 255; digit >= 0; --digit)
            {
                if (remaining_rank < histogram[digit])
                {
                    prefix |= (uint64_t)(uint32_t)digit << shift;
                    break;
                }
                remaining_rank -= histogram[digit];
            }
            prefix_mask |= (uint64_t)0xffu << shift;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u)
    {
        threshold = prefix;
        selected_cursor = 0u;
    }
    for (slot = threadIdx.x; slot < SPARK_DSV4_MODEL_INDEX_TOP_K; slot += blockDim.x)
    {
        selected_keys[slot] = 0u;
    }
    __syncthreads();

    for (slot = threadIdx.x; slot < bounded_slot_count; slot += blockDim.x)
    {
        uint64_t key = SparkDsv4OrderedTopKKey(scores[slot], slot);
        if (key != 0u && key >= threshold)
        {
            uint32_t destination = atomicAdd(&selected_cursor, 1u);
            if (destination < selected_count)
            {
                selected_keys[destination] = key;
            }
        }
    }
    __syncthreads();

    sort_width = 1u;
    while (sort_width < topk)
    {
        sort_width <<= 1u;
    }
    for (uint32_t size = 2u; size <= sort_width; size <<= 1u)
    {
        for (uint32_t stride = size >> 1u; stride != 0u; stride >>= 1u)
        {
            for (slot = threadIdx.x; slot < sort_width; slot += blockDim.x)
            {
                uint32_t partner = slot ^ stride;
                if (partner > slot)
                {
                    uint64_t left = selected_keys[slot];
                    uint64_t right = selected_keys[partner];
                    uint32_t ascending = (slot & size) != 0u ? 1u : 0u;
                    uint32_t swap = ascending != 0u ? left > right : left < right;
                    if (swap != 0u)
                    {
                        selected_keys[slot] = right;
                        selected_keys[partner] = left;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (slot = threadIdx.x; slot < topk; slot += blockDim.x)
    {
        int32_t selected_index = -1;
        if (slot < selected_count)
        {
            uint32_t original_slot = 0xffffffffu - (uint32_t)selected_keys[slot];
            selected_index = (int32_t)original_slot + offset;
        }
        indices_out[((uint64_t)row * out_row_stride) + slot] = selected_index;
    }
}

/*
 * mHC mixes for one row: 24 (or hc for the head) fp32 dot products of the
 * fn rows against the flattened streams, scaled by the rsqrt of the
 * flattened mean square - the norm applied to the mix, exactly the
 * reference order. One block per row, one warp per mix row.
 */
/* Pro tiles the HcMix staging: 4096 floats = 16 KB shared, safely under the
 * GB10 99 KB dynamic-shared limit for any hidden size. Flash builds ignore it. */
#define SPARK_DSV4_HC_MIX_TILE 4096u
#define SPARK_LM_HC_MIX_ROWS_PER_WARP 3u

static __global__ void SparkDsv4HcMixKernel(const void *streams_bf16, const float *fn_f32, float *mixes_f32, uint32_t row_count, uint32_t flat_dimension, uint32_t mix_rows, float epsilon)
{
	extern __shared__ float hc_shared[];
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x,warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES,mix,element;
	float value,total = 0.0f,inverse,accumulator;
	if ( row >= row_count )
		return;
#if defined(SPARK_DSV4_PRO_BUILD)
	/* Pro's flat dimension (4 streams x 7168 = 28672 floats = 112 KB) exceeds
	 * the GB10 dynamic-shared limit (101376 B), so tile the staging. Flash
	 * builds keep the original single-pass path. */
	float mix_accum[SPARK_LM_HC_MIX_ROWS_PER_WARP];
	uint32_t tile,tile_end,tile_elements,mix_index,mix_row;
	total = 0.0f;
	mix_index = 0u;
	for (mix = warp; mix < mix_rows; mix += SPARK_LM_CTA_WARPS)
	{
		mix_accum[mix_index] = 0.0f;
		mix_index++;
	}
	for (tile = 0u; tile < flat_dimension; tile += SPARK_DSV4_HC_MIX_TILE)
	{
		tile_end = tile + SPARK_DSV4_HC_MIX_TILE < flat_dimension ?
			tile + SPARK_DSV4_HC_MIX_TILE : flat_dimension;
		tile_elements = tile_end - tile;
		for (element = threadIdx.x; element < tile_elements; element += blockDim.x)
		{
			value = SparkLmBf16ToFloat(streams_bf16,
				((uint64_t)row * flat_dimension) + tile + element);
			hc_shared[element] = value;
			total += value * value;
		}
		__syncthreads();
		mix_index = 0u;
		for (mix = warp; mix < mix_rows; mix += SPARK_LM_CTA_WARPS)
		{
			mix_row = mix;
			accumulator = 0.0f;
			for (element = lane; element < tile_elements; element += SPARK_LM_WARP_LANES)
				accumulator += hc_shared[element] *
					fn_f32[((uint64_t)mix_row * flat_dimension) + tile + element];
			accumulator = SparkLmWarpReduceSum(accumulator);
			if ( lane == 0u )
				mix_accum[mix_index] += accumulator;
			mix_index++;
		}
		__syncthreads();
	}
	total = SparkLmBlockReduceSum(total,reduce_scratch);
	inverse = rsqrtf(total / (float)flat_dimension + epsilon);
	mix_index = 0u;
	for (mix = warp; mix < mix_rows; mix += SPARK_LM_CTA_WARPS)
	{
		if ( lane == 0u )
			mixes_f32[((uint64_t)row * mix_rows) + mix] = mix_accum[mix_index] * inverse;
		mix_index++;
	}
#else
	for (element = threadIdx.x; element < flat_dimension; element += blockDim.x)
	{
		value = SparkLmBf16ToFloat(streams_bf16,((uint64_t)row * flat_dimension) + element);
		hc_shared[element] = value;
		total += value * value;
	}
	total = SparkLmBlockReduceSum(total,reduce_scratch);
	inverse = rsqrtf(total / (float)flat_dimension + epsilon);
	for (mix = warp; mix < mix_rows; mix += SPARK_LM_CTA_WARPS)
	{
		accumulator = 0.0f;
		for (element = lane; element < flat_dimension; element += SPARK_LM_WARP_LANES)
			accumulator += hc_shared[element] * fn_f32[((uint64_t)mix * flat_dimension) + element];
		accumulator = SparkLmWarpReduceSum(accumulator);
		if ( lane == 0u )
			mixes_f32[((uint64_t)row * mix_rows) + mix] = accumulator * inverse;
	}
#endif
}

static __global__ void SparkDsv4HcMixSplitKKernel(const void *streams_bf16, const float *fn_f32, float *partials_f32, uint32_t row_count, uint32_t flat_dimension)
{
	__shared__ float values[SPARK_DSV4_HC_SPLIT_K_ELEMENTS];
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x / SPARK_DSV4_HC_SPLIT_K_COUNT;
	uint32_t split = blockIdx.x % SPARK_DSV4_HC_SPLIT_K_COUNT;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t base = split * SPARK_DSV4_HC_SPLIT_K_ELEMENTS,mix,element;
	uint64_t partial_base;
	float value,total,accumulator;
	if ( row >= row_count )
		return;
	value = SparkLmBf16ToFloat(streams_bf16,(uint64_t)row * flat_dimension + base + threadIdx.x);
	values[threadIdx.x] = value;
	total = SparkLmBlockReduceSum(value * value,reduce_scratch);
	partial_base = ((uint64_t)row * SPARK_DSV4_HC_SPLIT_K_COUNT + split) * SPARK_DSV4_HC_SPLIT_K_PARTIALS;
	if ( threadIdx.x == 0u )
		partials_f32[partial_base] = total;
	for (mix=warp; mix<SPARK_DSV4_MODEL_HC_MIX_ROWS; mix+=SPARK_LM_CTA_WARPS)
	{
		accumulator = 0.0f;
		for (element=lane; element<SPARK_DSV4_HC_SPLIT_K_ELEMENTS; element+=SPARK_LM_WARP_LANES)
			accumulator += values[element] * __ldg(fn_f32 + (uint64_t)mix * flat_dimension + base + element);
		accumulator = SparkLmWarpReduceSum(accumulator);
		if ( lane == 0u )
			partials_f32[partial_base + 1u + mix] = accumulator;
	}
}

static __device__ __forceinline__ void SparkDsv4HcParallelSinkhorn(const float *mixes, const float *scale3_f32, const float *base_f32, uint32_t iterations, float epsilon, float *pre_f32, float *post_f32, float *comb_f32, uint32_t row, float *comb, float *sums)
{
	uint32_t lane = threadIdx.x,i,j,iteration;
	float maximum,total;
	if ( lane < SPARK_DSV4_MODEL_HC_STREAM_COUNT )
	{
		pre_f32[(uint64_t)row * SPARK_DSV4_MODEL_HC_STREAM_COUNT + lane] = SparkLmSigmoid(mixes[lane] * scale3_f32[0] + base_f32[lane]) + epsilon;
		post_f32[(uint64_t)row * SPARK_DSV4_MODEL_HC_STREAM_COUNT + lane] = 2.0f * SparkLmSigmoid(mixes[SPARK_DSV4_MODEL_HC_STREAM_COUNT + lane] * scale3_f32[1] + base_f32[SPARK_DSV4_MODEL_HC_STREAM_COUNT + lane]);
		maximum = -3.0e38f;
		for (j=0u; j<SPARK_DSV4_MODEL_HC_STREAM_COUNT; j++)
		{
			comb[lane * SPARK_DSV4_MODEL_HC_STREAM_COUNT + j] = mixes[2u * SPARK_DSV4_MODEL_HC_STREAM_COUNT + lane * SPARK_DSV4_MODEL_HC_STREAM_COUNT + j] * scale3_f32[2] + base_f32[2u * SPARK_DSV4_MODEL_HC_STREAM_COUNT + lane * SPARK_DSV4_MODEL_HC_STREAM_COUNT + j];
			maximum = fmaxf(maximum,comb[lane * SPARK_DSV4_MODEL_HC_STREAM_COUNT + j]);
		}
		total = 0.0f;
		for (j=0u; j<SPARK_DSV4_MODEL_HC_STREAM_COUNT; j++)
			total += (comb[lane * SPARK_DSV4_MODEL_HC_STREAM_COUNT + j] = __expf(comb[lane * SPARK_DSV4_MODEL_HC_STREAM_COUNT + j] - maximum));
		for (j=0u; j<SPARK_DSV4_MODEL_HC_STREAM_COUNT; j++)
			comb[lane * SPARK_DSV4_MODEL_HC_STREAM_COUNT + j] = comb[lane * SPARK_DSV4_MODEL_HC_STREAM_COUNT + j] / total + epsilon;
	}
	__syncthreads();
	for (iteration=0u; iteration<iterations; iteration++)
	{
		if ( iteration != 0u && lane < SPARK_DSV4_MODEL_HC_STREAM_COUNT )
		{
			total = 0.0f;
			for (j=0u; j<SPARK_DSV4_MODEL_HC_STREAM_COUNT; j++)
				total += comb[lane * SPARK_DSV4_MODEL_HC_STREAM_COUNT + j];
			sums[lane] = total;
		}
		__syncthreads();
		if ( iteration != 0u && lane < SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT )
			comb[lane] /= sums[lane / SPARK_DSV4_MODEL_HC_STREAM_COUNT] + epsilon;
		__syncthreads();
		if ( lane < SPARK_DSV4_MODEL_HC_STREAM_COUNT )
		{
			total = 0.0f;
			for (i=0u; i<SPARK_DSV4_MODEL_HC_STREAM_COUNT; i++)
				total += comb[i * SPARK_DSV4_MODEL_HC_STREAM_COUNT + lane];
			sums[lane] = total;
		}
		__syncthreads();
		if ( lane < SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT )
			comb[lane] /= sums[lane % SPARK_DSV4_MODEL_HC_STREAM_COUNT] + epsilon;
		__syncthreads();
	}
	if ( lane < SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT )
		comb_f32[(uint64_t)row * SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT + lane] = comb[lane];
}

static __global__ void SparkDsv4HcMixSplitKFinalizeKernel(const float *partials_f32, const float *scale3_f32, const float *base_f32, uint32_t row_count, uint32_t iterations, float rms_epsilon, float hc_epsilon, float *mixes_f32, float *pre_f32, float *post_f32, float *comb_f32)
{
	__shared__ float mixes[SPARK_DSV4_MODEL_HC_MIX_ROWS];
	__shared__ float comb[SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT];
	__shared__ float sums[SPARK_DSV4_MODEL_HC_STREAM_COUNT];
	uint32_t row = blockIdx.x,lane = threadIdx.x,split;
	uint64_t partial_base;
	float total = 0.0f,inverse;
	if ( row >= row_count )
		return;
	if ( lane < SPARK_DSV4_HC_SPLIT_K_PARTIALS )
		for (split=0u; split<SPARK_DSV4_HC_SPLIT_K_COUNT; split++)
		{
			partial_base = ((uint64_t)row * SPARK_DSV4_HC_SPLIT_K_COUNT + split) * SPARK_DSV4_HC_SPLIT_K_PARTIALS;
			total += partials_f32[partial_base + lane];
		}
	if ( lane == 0u )
		sums[0] = rsqrtf(total / (float)SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS + rms_epsilon);
	__syncthreads();
	if ( lane > 0u && lane < SPARK_DSV4_HC_SPLIT_K_PARTIALS )
	{
		inverse = sums[0];
		mixes[lane - 1u] = total * inverse;
		mixes_f32[(uint64_t)row * SPARK_DSV4_MODEL_HC_MIX_ROWS + lane - 1u] = total * inverse;
	}
	__syncthreads();
	SparkDsv4HcParallelSinkhorn(mixes,scale3_f32,base_f32,iterations,hc_epsilon,pre_f32,post_f32,comb_f32,row,comb,sums);
}

/*
 * The split with inference Sinkhorn, one thread per row: sigmoid pre
 * (+eps), doubled sigmoid post, comb row-softmax +eps then the iteration
 * count of alternating row and column normalizations with +eps inside
 * every division - the first row pass is the softmax itself, matching the
 * reference kernel step for step at hc = 4.
 */
static __global__ void SparkDsv4HcSplitSinkhornKernel(const float *mixes_f32, const float *scale3_f32, const float *base_f32, uint32_t row_count, uint32_t hc, uint32_t iterations, float epsilon, float *pre_f32, float *post_f32, float *comb_f32)
{
	uint32_t row = blockIdx.x * blockDim.x + threadIdx.x,i,j,iteration,mix_rows = (2u + hc) * hc;
	const float *mixes;
	float comb[16],maximum,total;
	if ( row >= row_count || hc > 4u )
		return;
	mixes = mixes_f32 + ((uint64_t)row * mix_rows);
	for (i = 0; i < hc; i++)
	{
		pre_f32[((uint64_t)row * hc) + i] = SparkLmSigmoid(mixes[i] * scale3_f32[0] + base_f32[i]) + epsilon;
		post_f32[((uint64_t)row * hc) + i] = 2.0f * SparkLmSigmoid(mixes[hc + i] * scale3_f32[1] + base_f32[hc + i]);
	}
	for (i = 0; i < hc; i++)
	{
		maximum = -3.0e38f;
		for (j = 0; j < hc; j++)
		{
			comb[i * hc + j] = mixes[2u * hc + i * hc + j] * scale3_f32[2] + base_f32[2u * hc + i * hc + j];
			maximum = comb[i * hc + j] > maximum ? comb[i * hc + j] : maximum;
		}
		total = 0.0f;
		for (j = 0; j < hc; j++)
			total += (comb[i * hc + j] = __expf(comb[i * hc + j] - maximum));
		for (j = 0; j < hc; j++)
			comb[i * hc + j] = comb[i * hc + j] / total + epsilon;
	}
	for (iteration = 0; iteration < iterations; iteration++)
	{
		if ( iteration != 0u )
			for (i = 0; i < hc; i++)
			{
				total = 0.0f;
				for (j = 0; j < hc; j++)
					total += comb[i * hc + j];
				for (j = 0; j < hc; j++)
					comb[i * hc + j] /= total + epsilon;
			}
		for (j = 0; j < hc; j++)
		{
			total = 0.0f;
			for (i = 0; i < hc; i++)
				total += comb[i * hc + j];
			for (i = 0; i < hc; i++)
				comb[i * hc + j] /= total + epsilon;
		}
	}
	for (i = 0; i < hc * hc; i++)
		comb_f32[((uint64_t)row * hc * hc) + i] = comb[i];
}

// Stream reduction by pre, expansion by post + transposed comb, and the
// sigmoid head reduction - three small element kernels.
static __global__ void SparkDsv4HcPreReduceKernel(
	const void *streams_bf16,const float *pre_f32,void *reduced_bf16,
	void *residual_bf16,uint32_t row_count,uint32_t hc,uint32_t dimension,
	uint32_t tiles_per_row)
{
	uint32_t row = blockIdx.y;
	uint32_t tile = blockIdx.x,element,stream;
	uint64_t index;
	__nv_bfloat16 raw;
	float value;
	if ( row >= row_count )
		return;
	for (element=tile * blockDim.x + threadIdx.x; element<dimension;
		element+=tiles_per_row * blockDim.x)
	{
		value = 0.0f;
		for (stream = 0; stream < hc; stream++)
		{
			index = (((uint64_t)row * hc) + stream) * dimension + element;
			raw = ((const __nv_bfloat16 *)streams_bf16)[index];
			((__nv_bfloat16 *)residual_bf16)[index] = raw;
			value += pre_f32[((uint64_t)row * hc) + stream] *
				__bfloat162float(raw);
		}
		SparkLmFloatToBf16(reduced_bf16,((uint64_t)row * dimension) + element,value);
	}
}

static __global__ void SparkDsv4HcPostKernel(const void *out_bf16, const void *residual_bf16, const float *post_f32, const float *comb_f32, void *streams_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension, uint32_t tiles_per_row)
{
	__shared__ float post[SPARK_DSV4_MODEL_HC_STREAM_COUNT];
	__shared__ float comb[SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT];
	uint32_t row = blockIdx.y;
	uint32_t tile = blockIdx.x,element,stream,source;
	float residual[SPARK_DSV4_MODEL_HC_STREAM_COUNT],out,value;
	if ( row >= row_count || hc != SPARK_DSV4_MODEL_HC_STREAM_COUNT )
		return;
	if ( threadIdx.x < SPARK_DSV4_MODEL_HC_STREAM_COUNT )
		post[threadIdx.x] = post_f32[(uint64_t)row * SPARK_DSV4_MODEL_HC_STREAM_COUNT + threadIdx.x];
	if ( threadIdx.x < SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT )
		comb[threadIdx.x] = comb_f32[(uint64_t)row * SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT + threadIdx.x];
	__syncthreads();
	for (element=tile * blockDim.x + threadIdx.x; element<dimension;
		element+=tiles_per_row * blockDim.x)
	{
		out = SparkLmBf16ToFloat(out_bf16,(uint64_t)row * dimension + element);
		#pragma unroll
		for (source=0u; source<SPARK_DSV4_MODEL_HC_STREAM_COUNT; source++)
			residual[source] = SparkLmBf16ToFloat(residual_bf16,((uint64_t)row * SPARK_DSV4_MODEL_HC_STREAM_COUNT + source) * dimension + element);
		#pragma unroll
		for (stream=0u; stream<SPARK_DSV4_MODEL_HC_STREAM_COUNT; stream++)
		{
			value = __fmul_rn(post[stream],out);
			#pragma unroll
			for (source=0u; source<SPARK_DSV4_MODEL_HC_STREAM_COUNT; source++)
				value = __fmaf_rn(comb[source * SPARK_DSV4_MODEL_HC_STREAM_COUNT + stream],residual[source],value);
			SparkLmFloatToBf16(streams_bf16,((uint64_t)row * SPARK_DSV4_MODEL_HC_STREAM_COUNT + stream) * dimension + element,value);
		}
	}
}

static __global__ void SparkDsv4HcHeadReduceKernel(const void *streams_bf16, const float *mixes_f32, float scale, const float *base_f32, float epsilon, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension)
{
	uint32_t row = blockIdx.x,element,stream;
	float value,pre;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < dimension; element += blockDim.x)
	{
		value = 0.0f;
		for (stream = 0; stream < hc; stream++)
		{
			pre = SparkLmSigmoid(mixes_f32[((uint64_t)row * hc) + stream] * scale + base_f32[stream]) + epsilon;
			value += pre * SparkLmBf16ToFloat(streams_bf16,(((uint64_t)row * hc) + stream) * dimension + element);
		}
		SparkLmFloatToBf16(reduced_bf16,((uint64_t)row * dimension) + element,value);
	}
}

// bf16 rows widened to f32 (times a scalar) for the compressor's fp32
// pooling and the indexer's pre-scaled head weights.
static __global__ void SparkDsv4WidenKernel(const void *input_bf16, float *output_f32, uint32_t row_count, uint32_t width, float scale)
{
	uint32_t row = blockIdx.x,element;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < width; element += blockDim.x)
		output_f32[((uint64_t)row * width) + element] = SparkLmBf16ToFloat(input_bf16,((uint64_t)row * width) + element) * scale;
}

extern "C" cudaError_t SparkDsv4LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	SparkLmRmsNormKernel<<<row_count,SPARK_LM_CTA_THREADS,dimension * (uint32_t)sizeof(float),stream>>>(input_bf16,gain_bf16,output_bf16,row_count,dimension,epsilon);
	return(cudaGetLastError());
}

template <uint32_t ACTIVATION_CODEC>
static cudaError_t SparkDsv4LaunchLinearCodec(cudaStream_t stream,
	const SparkDsv4LinearView *view,const void *input_bf16,
	void *output_bf16,uint32_t row_count)
{
	cudaError_t status;
	if ( view == 0 || input_bf16 == 0 || output_bf16 == 0 ||
		view->payload == 0 || view->rows == 0u || view->columns == 0u )
		return(cudaErrorInvalidValue);
	status = SparkDsv4RequireNativeDecodeShape(row_count);
	if ( status != cudaSuccess )
		return(status);
	if ( view->weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 )
		return(SparkLmHostLaunchSm121DecodeLinear<128u,ACTIVATION_CODEC>(stream,view->weight_format,view->payload,view->scale_data,input_bf16,output_bf16,row_count,view->columns,view->rows));
	if ( view->weight_format == SPARK_LM_WEIGHT_FORMAT_BF16 )
		return(SparkLmHostLaunchSm121DecodeLinear<32u,SPARK_ACTIVATION_CODEC_NONE>(stream,view->weight_format,view->payload,view->scale_data,input_bf16,output_bf16,row_count,view->columns,view->rows));
	return(cudaErrorInvalidValue);
}

extern "C" cudaError_t SparkDsv4LaunchLinear(cudaStream_t stream,
	const SparkDsv4LinearView *view,const void *input_bf16,
	void *output_bf16,uint32_t row_count)
{
	return(SparkDsv4LaunchLinearCodec<
		SPARK_DSV4_MODEL_NON_EXPERT_ACTIVATION_CODEC>(stream,view,input_bf16,
		output_bf16,row_count));
}

extern "C" cudaError_t SparkDsv4LaunchExpertLinear(cudaStream_t stream,
	const SparkDsv4LinearView *view,const void *input_bf16,
	void *output_bf16,uint32_t row_count)
{
	if ( view == 0 || view->weight_format != SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 )
		return(cudaErrorInvalidValue);
	return(SparkDsv4LaunchLinearCodec<SPARK_DSV4_MODEL_EXPERT_ACTIVATION_CODEC>(
		stream,view,input_bf16,output_bf16,row_count));
}

extern "C" cudaError_t SparkDsv4LaunchFp8LinearPair(
	cudaStream_t stream,const SparkDsv4LinearView *first,
	const SparkDsv4LinearView *second,const void *input_bf16,
	void *first_output_bf16,void *second_output_bf16,uint32_t row_count)
{
	cudaError_t status;
	if ( first == 0 || second == 0 || input_bf16 == 0 ||
		first_output_bf16 == 0 || second_output_bf16 == 0 ||
		first->weight_format != SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 ||
		second->weight_format != SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 ||
		first->payload == 0 || first->scale_data == 0 ||
		second->payload == 0 || second->scale_data == 0 ||
		first->rows == 0u || second->rows == 0u || first->columns == 0u ||
		first->columns != second->columns )
		return(cudaErrorInvalidValue);
	status = SparkDsv4RequireNativeDecodeShape(row_count);
	if ( status != cudaSuccess )
		return(status);
	return(SparkLmHostLaunchSm121DecodeLinearPair<128u,
		SPARK_DSV4_MODEL_NON_EXPERT_ACTIVATION_CODEC>(stream,first->payload,
		(const uint8_t *)first->scale_data,first->rows,second->payload,
		(const uint8_t *)second->scale_data,second->rows,input_bf16,
		first_output_bf16,second_output_bf16,row_count,first->columns));
}

extern "C" cudaError_t SparkDsv4LaunchBf16LinearPair(
	cudaStream_t stream,const SparkDsv4LinearView *first,
	const SparkDsv4LinearView *second,const void *input_bf16,
	void *first_output_bf16,void *second_output_bf16,uint32_t row_count)
{
	cudaError_t status;
	if ( first == 0 || second == 0 || input_bf16 == 0 ||
		first_output_bf16 == 0 || second_output_bf16 == 0 ||
		first->weight_format != SPARK_LM_WEIGHT_FORMAT_BF16 ||
		second->weight_format != SPARK_LM_WEIGHT_FORMAT_BF16 ||
		first->payload == 0 || second->payload == 0 || first->rows == 0u ||
		first->columns == 0u || first->rows != second->rows ||
		first->columns != second->columns )
		return(cudaErrorInvalidValue);
	status = SparkDsv4RequireNativeDecodeShape(row_count);
	if ( status != cudaSuccess )
		return(status);
	return(SparkLmHostLaunchBf16LinearPair(stream,first->payload,
		second->payload,input_bf16,first_output_bf16,second_output_bf16,
		row_count,first->columns,first->rows));
}

static __global__ void SparkDsv4ProjectionPackKernel(
	const uint16_t *wq,const uint16_t *wkv,const uint16_t *compress_kv,
	const uint16_t *compress_score,const uint16_t *index_kv,
	const uint16_t *index_score,uint16_t *packed,uint32_t row_count,
	uint32_t tp_rank,uint32_t tp_degree,uint32_t compress_channels,
	uint32_t index_channels)
{
	uint32_t row = blockIdx.x,column = blockIdx.y * blockDim.x + threadIdx.x;
	uint32_t wq_width = SPARK_DSV4_MODEL_QUERY_LORA_RANK / tp_degree;
	uint32_t wkv_width = SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION / tp_degree;
	uint32_t compress_width = compress_channels / tp_degree;
	uint32_t index_width = index_channels / tp_degree;
	uint32_t base,local;
	uint16_t value = 0u;
	if ( row >= row_count || column >= SPARK_DSV4_MODEL_HIDDEN_DIMENSION )
		return;
	base = tp_rank * wq_width;
	if ( column >= base && column < base + wq_width )
		value = wq[(uint64_t)row * wq_width + column - base];
	base = SPARK_DSV4_MODEL_QUERY_LORA_RANK + tp_rank * wkv_width;
	if ( column >= base && column < base + wkv_width )
		value = wkv[(uint64_t)row * wkv_width + column - base];
	base = SPARK_DSV4_MODEL_QUERY_LORA_RANK + SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
	local = column >= base ? column - base : UINT32_MAX;
	if ( local < compress_channels * 2u && local % compress_channels >= tp_rank * compress_width && local % compress_channels < (tp_rank + 1u) * compress_width )
		value = local < compress_channels ? compress_kv[(uint64_t)row * compress_width + local - tp_rank * compress_width] : compress_score[(uint64_t)row * compress_width + local - compress_channels - tp_rank * compress_width];
	base += compress_channels * 2u;
	local = column >= base ? column - base : UINT32_MAX;
	if ( local < index_channels * 2u && local % index_channels >= tp_rank * index_width && local % index_channels < (tp_rank + 1u) * index_width )
		value = local < index_channels ? index_kv[(uint64_t)row * index_width + local - tp_rank * index_width] : index_score[(uint64_t)row * index_width + local - index_channels - tp_rank * index_width];
	packed[(uint64_t)row * SPARK_DSV4_MODEL_HIDDEN_DIMENSION + column] = value;
}

static __global__ void SparkDsv4ProjectionUnpackKernel(
	const uint16_t *packed,uint16_t *wq,uint16_t *wkv,
	uint16_t *compress_kv,uint16_t *compress_score,uint16_t *index_kv,
	uint16_t *index_score,uint32_t row_count,uint32_t compress_channels,
	uint32_t index_channels)
{
	uint32_t row = blockIdx.x,column = blockIdx.y * blockDim.x + threadIdx.x;
	uint32_t base;
	uint16_t value;
	if ( row >= row_count || column >= SPARK_DSV4_MODEL_HIDDEN_DIMENSION )
		return;
	value = packed[(uint64_t)row * SPARK_DSV4_MODEL_HIDDEN_DIMENSION + column];
	if ( column < SPARK_DSV4_MODEL_QUERY_LORA_RANK )
		wq[(uint64_t)row * SPARK_DSV4_MODEL_QUERY_LORA_RANK + column] = value;
	base = SPARK_DSV4_MODEL_QUERY_LORA_RANK;
	if ( column >= base && column < base + SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION )
		wkv[(uint64_t)row * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION + column - base] = value;
	base += SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
	if ( column >= base && column < base + compress_channels )
		compress_kv[(uint64_t)row * compress_channels + column - base] = value;
	base += compress_channels;
	if ( column >= base && column < base + compress_channels )
		compress_score[(uint64_t)row * compress_channels + column - base] = value;
	base += compress_channels;
	if ( column >= base && column < base + index_channels )
		index_kv[(uint64_t)row * index_channels + column - base] = value;
	base += index_channels;
	if ( column >= base && column < base + index_channels )
		index_score[(uint64_t)row * index_channels + column - base] = value;
}

static cudaError_t SparkDsv4ProjectionLayoutValidate(uint32_t tp_rank,
	uint32_t tp_degree,uint32_t compress_channels,uint32_t index_channels)
{
	uint64_t width = SPARK_DSV4_MODEL_QUERY_LORA_RANK +
		SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION +
		2u * (uint64_t)(compress_channels + index_channels);
	if ( tp_degree <= 1u || tp_rank >= tp_degree ||
		SPARK_DSV4_MODEL_QUERY_LORA_RANK % tp_degree != 0u ||
		SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION % tp_degree != 0u ||
		compress_channels % tp_degree != 0u ||
		index_channels % tp_degree != 0u ||
		width > SPARK_DSV4_MODEL_HIDDEN_DIMENSION )
		return(cudaErrorInvalidValue);
	return(cudaSuccess);
}

extern "C" cudaError_t SparkDsv4LaunchPackProjectionShards(cudaStream_t stream,
	const void *wq,const void *wkv,const void *compress_kv,
	const void *compress_score,const void *index_kv,const void *index_score,
	void *packed,uint32_t rows,uint32_t tp_rank,uint32_t tp_degree,
	uint32_t compress_channels,uint32_t index_channels)
{
	cudaError_t error = SparkDsv4ProjectionLayoutValidate(tp_rank,tp_degree,
		compress_channels,index_channels);
	if ( error != cudaSuccess || stream == 0 || wq == 0 || wkv == 0 ||
		packed == 0 || rows == 0u || (compress_channels != 0u &&
		(compress_kv == 0 || compress_score == 0)) || (index_channels != 0u &&
		(index_kv == 0 || index_score == 0)) )
		return(cudaErrorInvalidValue);
	dim3 grid(rows,(SPARK_DSV4_MODEL_HIDDEN_DIMENSION + 255u) / 256u);
	SparkDsv4ProjectionPackKernel<<<grid,256u,0u,stream>>>(
		(const uint16_t *)wq,(const uint16_t *)wkv,
		(const uint16_t *)compress_kv,(const uint16_t *)compress_score,
		(const uint16_t *)index_kv,(const uint16_t *)index_score,
		(uint16_t *)packed,rows,tp_rank,tp_degree,compress_channels,index_channels);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchUnpackProjectionShards(cudaStream_t stream,
	const void *packed,void *wq,void *wkv,void *compress_kv,
	void *compress_score,void *index_kv,void *index_score,uint32_t rows,
	uint32_t compress_channels,uint32_t index_channels)
{
	uint64_t width = SPARK_DSV4_MODEL_QUERY_LORA_RANK +
		SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION +
		2u * (uint64_t)(compress_channels + index_channels);
	if ( stream == 0 || packed == 0 || wq == 0 || wkv == 0 || rows == 0u ||
		width > SPARK_DSV4_MODEL_HIDDEN_DIMENSION || (compress_channels != 0u &&
		(compress_kv == 0 || compress_score == 0)) || (index_channels != 0u &&
		(index_kv == 0 || index_score == 0)) )
		return(cudaErrorInvalidValue);
	dim3 grid(rows,(SPARK_DSV4_MODEL_HIDDEN_DIMENSION + 255u) / 256u);
	SparkDsv4ProjectionUnpackKernel<<<grid,256u,0u,stream>>>(
		(const uint16_t *)packed,(uint16_t *)wq,(uint16_t *)wkv,
		(uint16_t *)compress_kv,(uint16_t *)compress_score,
		(uint16_t *)index_kv,(uint16_t *)index_score,rows,compress_channels,
		index_channels);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchStridedLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *payload, const uint8_t *scale, uint64_t weight_payload_group_stride_bytes, uint64_t weight_scale_group_stride_bytes, const void *input_bf16, uint64_t input_row_stride, uint32_t input_offset, uint32_t input_group_stride, void *output_bf16, uint64_t output_row_stride, uint32_t output_offset, uint32_t output_group_stride, uint32_t group_count, uint32_t row_count)
{
	cudaError_t status;
	if ( view == 0 || view->weight_format != SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 ||
		payload == 0 || scale == 0 || input_bf16 == 0 || output_bf16 == 0 ||
		group_count == 0u || view->columns % 128u != 0u ||
		view->rows % SPARK_LM_SM121_NATIVE_TILE_N != 0u )
		return(cudaErrorInvalidValue);
	status = SparkDsv4RequireNativeDecodeShape(row_count);
	if ( status != cudaSuccess )
		return(status);
	return(SparkLmHostLaunchSm121StridedDecodeLinear<128u,SPARK_DSV4_MODEL_OUTPUT_COMPOSITION_ACTIVATION_CODEC>(stream,view->weight_format,payload,scale,weight_payload_group_stride_bytes,weight_scale_group_stride_bytes,input_bf16,input_row_stride,input_offset,input_group_stride,output_bf16,output_row_stride,output_offset,output_group_stride,group_count,row_count,view->columns,view->rows));
}

extern "C" cudaError_t SparkDsv4LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	SparkLmEmbeddingGatherKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(token_ids,embedding_bf16,hidden_bf16,row_count,hidden_dimension);
	return(cudaGetLastError());
}

static_assert(SPARK_DSV4_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP == SPARK_LM_HEAD_SCREEN_CAP,"screen cap must match the shared kernels");

extern "C" cudaError_t SparkDsv4LaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchHeadShadowQuantize<SPARK_LM_HEAD_SHADOW_GROUP>(stream,head_bf16,shadow_payload,shadow_scale,error_norm,candidate_count,hidden_dimension));
}

extern "C" cudaError_t SparkDsv4LaunchHeadCertifiedFp8Quantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, float *shadow_scale_f32, float *cert_norm_f32, uint32_t candidate_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchHeadCertifiedFp8Quantize(stream,head_bf16,
		shadow_payload,shadow_scale_f32,cert_norm_f32,candidate_count,
		hidden_dimension));
}

// Screened exact head, the mimo25 pattern; replaces the block-per-row
// full scan outright - dsv4 never carried the intermediate tiled form.
extern "C" cudaError_t SparkDsv4LaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension)
{
	cudaError_t status = SparkDsv4RequireNativeDecodeShape(row_count);
	if ( status != cudaSuccess )
		return(status);
	return(SparkLmHostLaunchHeadScreenedArgmax(stream,hidden_bf16,head_weight_bf16,shadow_payload,shadow_scale,error_norm,logits_bf16,candidate_ids,candidate_counts,output_token_ids,row_count,candidate_count,hidden_dimension));
}

extern "C" cudaError_t SparkDsv4LaunchHeadScreenedArgmaxSharded(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, float *output_scores, uint32_t candidate_offset, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension)
{
	cudaError_t status = SparkDsv4RequireNativeDecodeShape(row_count);
	if ( status != cudaSuccess )
		return(status);
	return(SparkLmHostLaunchHeadScreenedArgmaxWithScore(stream,hidden_bf16,
		head_weight_bf16,shadow_payload,shadow_scale,error_norm,logits_bf16,
		candidate_ids,candidate_counts,output_token_ids,output_scores,
		candidate_offset,row_count,candidate_count,hidden_dimension));
}

extern "C" cudaError_t SparkDsv4LaunchHeadCertifiedFp8B1Sharded(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const float *shadow_scale_f32, const float *cert_norm_f32, void *scratch, uint32_t *candidate_ids, uint32_t *screened_count, uint32_t *output_token_id, float *output_score, uint32_t candidate_offset, uint32_t row_count, uint32_t vocabulary_count, uint32_t hidden_dimension)
{
	cudaError_t status = SparkDsv4RequireNativeDecodeShape(row_count);
	if ( status != cudaSuccess )
		return(status);
	return(SparkLmHostLaunchHeadCertifiedFp8B1WithScore(stream,hidden_bf16,
		head_weight_bf16,shadow_payload,shadow_scale_f32,cert_norm_f32,scratch,
		candidate_ids,screened_count,output_token_id,output_score,
		candidate_offset,row_count,vocabulary_count,hidden_dimension));
}

extern "C" cudaError_t SparkDsv4LaunchHeadMaxlocPack(cudaStream_t stream, const float *scores, const uint32_t *token_ids, uint64_t *maxloc, uint32_t row_count)
{
	if ( scores == 0 || token_ids == 0 || maxloc == 0 || row_count == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv4HeadMaxlocPackKernel<<<(row_count + 255u) / 256u,256u,0u,
		stream>>>(scores,token_ids,maxloc,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHeadMaxlocUnpack(cudaStream_t stream, const uint64_t *maxloc, uint32_t *token_ids, uint32_t row_count)
{
	if ( maxloc == 0 || token_ids == 0 || row_count == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv4HeadMaxlocUnpackKernel<<<(row_count + 255u) / 256u,256u,0u,
		stream>>>(maxloc,token_ids,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchResidentTokenFeedback(
	cudaStream_t stream,const uint32_t *output_token_ids,
	uint32_t *resident_token_ids,uint32_t *input_token_ids,
	uint64_t *row_positions,uint64_t *row_emit_positions,
	uint64_t *row_emit_positions_hca,uint32_t row_count,
	uint32_t tokens_per_sequence,uint32_t step_index,uint32_t advance)
{
	if ( stream == 0 || output_token_ids == 0 || resident_token_ids == 0 ||
		input_token_ids == 0 || row_positions == 0 || row_emit_positions == 0 ||
		row_emit_positions_hca == 0 || row_count == 0u ||
		tokens_per_sequence == 0u || step_index >= tokens_per_sequence ||
		advance > 1u )
		return(cudaErrorInvalidValue);
	SparkDsv4ResidentTokenFeedbackKernel<<<(row_count + 255u) / 256u,256u,0u,
		stream>>>(output_token_ids,resident_token_ids,input_token_ids,
		row_positions,row_emit_positions,row_emit_positions_hca,row_count,
		tokens_per_sequence,step_index,advance);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchAccumU64Max(cudaStream_t stream, uint64_t *destination, const uint64_t *source, uint32_t element_count)
{
	if ( stream == 0 || destination == 0 || source == 0 || element_count == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv4AccumU64MaxKernel<<<(element_count + 255u) / 256u,256u,0u,
		stream>>>(destination,source,element_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension)
{
	SparkLmHeadArgmaxKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,head_weight_bf16,token_ids,output_token_ids,row_count,hidden_dimension,candidate_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchGatherHeadRows(cudaStream_t stream, const void *source_bf16, const uint32_t *source_row_indices, void *destination_bf16, uint32_t row_count, uint32_t row_width)
{
	return(SparkLaunchGatherBf16Rows(stream,source_bf16,source_row_indices,destination_bf16,row_count,row_width));
}

extern "C" cudaError_t SparkDsv4LaunchScatterHeadTokens(cudaStream_t stream, const uint32_t *source, const uint32_t *destination_lane_indices, uint32_t *destination, uint32_t row_count)
{
	return(SparkLaunchScatterU32Rows(stream,source,destination_lane_indices,destination,row_count));
}

extern "C" cudaError_t SparkDsv4LaunchQuantSim(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t row_stride, uint32_t width, uint32_t block, uint32_t fp4)
{
	dim3 grid(row_count,(width + block - 1u) / block);
	SparkDsv4QuantSimKernel<<<grid,SPARK_LM_WARP_LANES,0,stream>>>(data_bf16,row_count,row_stride,width,block,fp4 != 0u ? 6.0f : 448.0f,fp4);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchRope(cudaStream_t stream, void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse)
{
	dim3 grid(row_count,head_count);
	SparkDsv4RopeKernel<<<grid,rope_dim / 2u,0,stream>>>(data_bf16,freqs_f32,row_positions,row_count,head_count,head_dim,rope_dim,inverse);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchQueryHeadRms(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim, float epsilon)
{
	dim3 grid(row_count,head_count);
	SparkDsv4QueryHeadRmsKernel<<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(data_bf16,row_count,head_count,head_dim,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchQueryHeadRmsRope(
	cudaStream_t stream,void *data_bf16,const float *freqs_f32,
	const uint64_t *row_positions,uint32_t row_count,uint32_t head_count,
	uint32_t head_dim,uint32_t rope_dim,float epsilon)
{
	dim3 grid(row_count,head_count);
	uint32_t inverse = 0u;
	if ( data_bf16 == 0 || freqs_f32 == 0 || row_positions == 0 ||
		row_count == 0u ||
		row_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT ||
		head_count == 0u || head_count > SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT ||
		(SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT % head_count) != 0u ||
		head_dim != SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION ||
		rope_dim != SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION ||
		epsilon != SPARK_DSV4_MODEL_RMS_NORM_EPSILON )
		return(cudaErrorInvalidValue);
	SparkDsv4QueryHeadRmsRopeKernel<<<grid,SPARK_LM_CTA_THREADS,0u,stream>>>(
		data_bf16,freqs_f32,row_positions,row_count,head_count,head_dim,
		rope_dim,epsilon,inverse);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchKvPost(
	cudaStream_t stream,void *data_bf16,const void *gain_bf16,
	const float *freqs_f32,const uint64_t *row_positions,uint32_t row_count,
	uint32_t head_dim,uint32_t rope_dim,uint32_t quant_width,
	uint32_t quant_block,float epsilon)
{
	uint32_t inverse = 0u;
	if ( data_bf16 == 0 || gain_bf16 == 0 || freqs_f32 == 0 ||
		row_positions == 0 || row_count == 0u ||
		row_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT ||
		head_dim != SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION ||
		rope_dim != SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION ||
		quant_width != head_dim - rope_dim ||
		quant_block != SPARK_DSV4_MODEL_KV_QUANT_BLOCK ||
		(quant_width + quant_block - 1u) / quant_block + 1u !=
			SPARK_LM_CTA_WARPS ||
		epsilon != SPARK_DSV4_MODEL_RMS_NORM_EPSILON )
		return(cudaErrorInvalidValue);
	SparkDsv4KvPostKernel<<<row_count,SPARK_LM_CTA_THREADS,
		head_dim * (uint32_t)sizeof(float),stream>>>(data_bf16,gain_bf16,
		freqs_f32,row_positions,row_count,head_dim,rope_dim,quant_width,
		quant_block,epsilon,inverse);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchIndexerPost(
	cudaStream_t stream,void *data_bf16,const float *freqs_f32,
	const uint64_t *row_positions,uint32_t row_count,uint32_t head_count,
	uint32_t head_dim,uint32_t rope_dim,uint32_t quant_block)
{
	dim3 grid(row_count,head_count);
	uint32_t inverse = 0u;
	if ( data_bf16 == 0 || freqs_f32 == 0 || row_positions == 0 ||
		row_count == 0u ||
		row_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT ||
		head_count != SPARK_DSV4_MODEL_INDEX_HEAD_COUNT ||
		head_dim != SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION ||
		rope_dim != SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION ||
		quant_block != SPARK_DSV4_MODEL_FP4_QUANT_BLOCK ||
		head_dim % quant_block != 0u ||
		head_dim / quant_block > SPARK_LM_CTA_WARPS )
		return(cudaErrorInvalidValue);
	SparkDsv4IndexerPostKernel<<<grid,SPARK_LM_CTA_THREADS,
		head_dim * (uint32_t)sizeof(float),stream>>>(data_bf16,freqs_f32,
		row_positions,row_count,head_count,head_dim,rope_dim,quant_block,
		inverse);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHadamard(cudaStream_t stream, void *data_bf16, uint32_t vector_count, uint32_t width)
{
    if (data_bf16 == 0 || vector_count == 0u || width == 0u ||
        (width & (width - 1u)) != 0u ||
        width > SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION)
    {
        return cudaErrorInvalidValue;
    }
    SparkDsv4HadamardKernel<<<
        vector_count,
        SPARK_LM_CTA_THREADS,
        width * (uint32_t)sizeof(float),
        stream>>>(data_bf16, vector_count, width);
    return cudaGetLastError();
}

static size_t SparkDsv4SparseAttnSharedBytes(uint32_t head_dimension)
{
    static const uint32_t heads_per_cta = SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA;

    return
        (size_t)heads_per_cta * head_dimension * sizeof(float) +
        (size_t)heads_per_cta * SPARK_LM_CTA_WARPS *
            head_dimension * sizeof(float);
}

static uint32_t SparkDsv4SparseAttnSplitCount(uint32_t row_count, uint32_t head_count, uint32_t topk, uint32_t compressed_entries_per_page, uint32_t multiprocessor_count)
{
    uint32_t groups = (head_count + SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA - 1u) / SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA;
    uint32_t blocks = row_count * groups,effective_topk,splits;
    if ( blocks == 0u || multiprocessor_count == 0u )
        return(0u);
    effective_topk = compressed_entries_per_page == 0u ? min(topk,SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS) : topk;
    splits = multiprocessor_count / blocks;
    splits = min(splits,SPARK_DSV4_SPARSE_ATTN_MAX_SPLITS);
    splits = min(splits,effective_topk / SPARK_DSV4_SPARSE_ATTN_MIN_SLOTS_PER_SPLIT);
    return(max(splits,1u));
}

extern "C" cudaError_t SparkDsv4ConfigureCudaKernels(uint32_t *multiprocessor_count)
{
    cudaError_t error;
	int32_t device,value;

	if ( multiprocessor_count == 0 )
		return(cudaErrorInvalidValue);
	error = cudaGetDevice(&device);
	if ( error == cudaSuccess )
		error = cudaDeviceGetAttribute(&value,cudaDevAttrMultiProcessorCount,device);
	if ( error == cudaSuccess && value <= 0 )
		error = cudaErrorInvalidDevice;
	if ( error == cudaSuccess )
		*multiprocessor_count = (uint32_t)value;
	if ( error != cudaSuccess )
		return(error);
    error = cudaFuncSetAttribute(
        SparkDsv4SparseAttnKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)SparkDsv4SparseAttnSharedBytes(
            SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION));
    if (error == cudaSuccess)
    {
        uint32_t hc_mix_shared_elements;
#if defined(SPARK_DSV4_PRO_BUILD)
        hc_mix_shared_elements = SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS <
            SPARK_DSV4_HC_MIX_TILE ? SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS :
            SPARK_DSV4_HC_MIX_TILE;
#else
        hc_mix_shared_elements = SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
#endif
        error = cudaFuncSetAttribute(
            SparkDsv4HcMixKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)(hc_mix_shared_elements * sizeof(float)));
    }
#if defined(SPARK_DSV4_PRO_BUILD)
    /* Pro's output composition (16 groups x 1024 = 16384) stages 64 KB of
     * activations in the decode LinearKernel - over the 48 KB default.
     * Opt the exact instantiation in; Flash's 8192-wide input never needs
     * this and keeps its original attribute set. */
    if ( error == cudaSuccess )
        error = cudaFuncSetAttribute(
            SparkLmLinearKernel<128u,
                SPARK_DSV4_MODEL_NON_EXPERT_ACTIVATION_CODEC,
                SPARK_LM_CTA_WARPS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)((uint64_t)SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT *
                SPARK_DSV4_MODEL_OUTPUT_LORA_RANK * sizeof(float)));
#endif
    return error;
}

extern "C" cudaError_t SparkDsv4LaunchWeightReadAhead(
	cudaStream_t stream,const void *payload,uint64_t bytes,
	const void *auxiliary_payload,uint64_t auxiliary_bytes,uint32_t *sink_u32,
	uint32_t block_capacity)
{
	cudaError_t error = SparkDsv4RequireNativeSm121();
	if ( error != cudaSuccess )
		return(error);
	return(SparkLmHostLaunchWeightReadAhead(stream,payload,bytes,
		auxiliary_payload,auxiliary_bytes,sink_u32,block_capacity));
}

extern "C" cudaError_t SparkDsv4LaunchSparseAttn(
    cudaStream_t stream,
    const void *q_bf16,
    const void *kv_cache_bf16,
    uint64_t lane_stride_elements,
    const uint32_t *row_lane_indices,
    const uint32_t *row_page_table_indices,
    const uint32_t *physical_page_table,
    uint32_t page_table_stride,
    uint32_t compressed_entries_per_page,
    const int32_t *topk_idxs,
    const uint32_t *valid_topk_counts,
    uint32_t topk,
    const float *sink_f32,
    float scale,
    void *out_bf16,
    float *partials_f32,
    uint32_t partial_capacity,
    uint32_t multiprocessor_count,
    uint32_t row_count,
    uint32_t head_count,
    uint32_t head_dim)
{
    static const uint32_t heads_per_cta = SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA;
    dim3 grid;
    cudaError_t error;
    size_t shared_bytes;
    uint32_t split_count,partial_count;

    if (q_bf16 == 0 || kv_cache_bf16 == 0 ||
        row_lane_indices == 0 || row_page_table_indices == 0 ||
        physical_page_table == 0 || page_table_stride == 0u ||
        topk_idxs == 0 || valid_topk_counts == 0 || out_bf16 == 0 ||
        partials_f32 == 0 ||
        topk == 0u || row_count == 0u || head_count == 0u ||
        head_dim == 0u || head_dim > SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION ||
        (head_dim & 1u) != 0u || partial_capacity == 0u ||
        multiprocessor_count == 0u)
    {
        return cudaErrorInvalidValue;
    }
    split_count = SparkDsv4SparseAttnSplitCount(row_count,head_count,topk,
        compressed_entries_per_page,multiprocessor_count);
    partial_count = row_count *
        ((head_count + heads_per_cta - 1u) / heads_per_cta) * split_count;
    if ( split_count == 0u || partial_count > partial_capacity )
        return(cudaErrorInvalidValue);
    grid = dim3(row_count,(head_count + heads_per_cta - 1u) / heads_per_cta,
        split_count);
    shared_bytes = SparkDsv4SparseAttnSharedBytes(head_dim);
    SparkDsv4SparseAttnKernel<<<
        grid,
        SPARK_LM_CTA_THREADS,
        shared_bytes,
        stream>>>(
        q_bf16,
        kv_cache_bf16,
        lane_stride_elements,
        row_lane_indices,
        row_page_table_indices,
        physical_page_table,
        page_table_stride,
        compressed_entries_per_page,
        topk_idxs,
        valid_topk_counts,
        topk,
        sink_f32,
        scale,
        out_bf16,
        partials_f32,
        row_count,
        head_count,
        head_dim,
        split_count);
    error = cudaGetLastError();
    if ( error != cudaSuccess || split_count == 1u )
        return(error);
    SparkDsv4SparseAttnMergeKernel<<<dim3(row_count,grid.y),SPARK_LM_CTA_THREADS,0,
        stream>>>(partials_f32,valid_topk_counts,sink_f32,out_bf16,row_count,
        head_count,head_dim,split_count);
    return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchWiden(cudaStream_t stream, const void *input_bf16, float *output_f32, uint32_t row_count, uint32_t width, float scale)
{
	SparkDsv4WidenKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(input_bf16,output_f32,row_count,width,scale);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchCompressStep(cudaStream_t stream,
	const void *kv_bf16,const void *score_bf16,const float *ape_f32,
	float *kv_state_f32,float *score_state_f32,uint64_t state_lane_stride,
	const uint32_t *row_lane_indices,const uint64_t *row_positions,
	uint32_t row_count,uint32_t ratio,uint32_t overlapped,uint32_t width,
	void *emit_bf16,uint32_t *emitted)
{
	if ( kv_bf16 == 0 || score_bf16 == 0 || ape_f32 == 0 ||
		kv_state_f32 == 0 || score_state_f32 == 0 ||
		row_lane_indices == 0 || row_positions == 0 || row_count == 0u ||
		ratio == 0u || width == 0u || emit_bf16 == 0 || emitted == 0 ||
		overlapped > 1u )
		return(cudaErrorInvalidValue);
	SparkDsv4CompressStepKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(
		kv_bf16,score_bf16,ape_f32,kv_state_f32,score_state_f32,
		state_lane_stride,row_lane_indices,row_positions,row_count,ratio,
		overlapped,width,emit_bf16,emitted);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchKvEmission(
	cudaStream_t stream,void *emit_bf16,const uint32_t *emitted,
	const void *norm_weight_bf16,const float *freqs_f32,
	const uint64_t *row_emit_positions,void *cache_bf16,
	uint64_t cache_lane_stride,const uint32_t *row_lane_indices,
	const uint64_t *row_positions,uint32_t row_count,uint32_t width,
	uint64_t base_slot,uint32_t ratio,uint32_t ring_slots,uint32_t rotate)
{
	if ( emit_bf16 == 0 || norm_weight_bf16 == 0 || freqs_f32 == 0 ||
		row_emit_positions == 0 || cache_bf16 == 0 || row_lane_indices == 0 ||
		row_positions == 0 || row_count == 0u || cache_lane_stride == 0u ||
		width == 0u || rotate > 1u ||
		width < SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION || ring_slots == 0u ||
		(rotate == 0u && width == SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION) ||
		(ratio != 0u && (emitted == 0 || ring_slots % ratio != 0u)) ||
		(rotate != 0u && ((width & (width - 1u)) != 0u ||
		width > SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION)) )
		return(cudaErrorInvalidValue);
	SparkDsv4KvEmissionKernel<<<row_count,SPARK_LM_CTA_THREADS,
		width * (uint32_t)sizeof(float),stream>>>(emit_bf16,emitted,
		norm_weight_bf16,freqs_f32,row_emit_positions,(uint16_t *)cache_bf16,
		cache_lane_stride,row_lane_indices,row_positions,row_count,width,
		base_slot,ratio,ring_slots,rotate);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchCacheScatter(cudaStream_t stream, const void *source_bf16, const uint32_t *emitted, void *cache_bf16, uint64_t cache_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t row_count, uint32_t width, uint64_t base_slot, uint32_t ratio, uint32_t ring_slots)
{
	if ( source_bf16 == 0 || cache_bf16 == 0 || row_lane_indices == 0 || row_positions == 0 || row_count == 0u || width == 0u || ring_slots == 0u || (ratio != 0u && (emitted == 0 || ring_slots % ratio != 0u)) )
		return(cudaErrorInvalidValue);
	SparkDsv4CacheScatterKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>((const uint16_t *)source_bf16,emitted,(uint16_t *)cache_bf16,cache_lane_stride,row_lane_indices,row_positions,row_count,width,base_slot,ratio,ring_slots);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchInitializePages(cudaStream_t stream, void *page_pool, uint64_t page_stride_bytes, const uint32_t *page_indices, const uint32_t *parent_page_indices, uint32_t page_count, const SparkDsv4PagedScoreSpan *score_spans, uint32_t score_span_count)
{
	if ( page_pool == 0 || page_indices == 0 || parent_page_indices == 0 || page_count == 0u || page_stride_bytes == 0u || page_stride_bytes % sizeof(uint32_t) != 0u || (score_span_count != 0u && score_spans == 0) )
		return(cudaErrorInvalidValue);
	SparkDsv4InitializePagesKernel<<<page_count,SPARK_LM_CTA_THREADS,0,stream>>>((uint32_t *)page_pool,page_stride_bytes / sizeof(uint32_t),page_indices,parent_page_indices,page_count,score_spans,score_span_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchUpdatePageTable(cudaStream_t stream, uint32_t *page_table, const uint32_t *update_indices, const uint32_t *update_values, uint32_t update_count)
{
	uint32_t blocks;
	if ( page_table == 0 || update_indices == 0 || update_values == 0 || update_count == 0u )
		return(cudaErrorInvalidValue);
	blocks = (update_count + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS;
	SparkDsv4UpdatePageTableKernel<<<blocks,SPARK_LM_CTA_THREADS,0,stream>>>(page_table,update_indices,update_values,update_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchBuildAttentionIndices(cudaStream_t stream, const uint64_t *row_positions, int32_t *indices, uint32_t *slot_counts, uint32_t *attention_slot_counts, uint32_t row_count, uint32_t column_count, uint32_t index_slot_capacity, uint32_t layer_kind)
{
	if ( row_positions == 0 || indices == 0 || slot_counts == 0 || attention_slot_counts == 0 || row_count == 0u || column_count == 0u || layer_kind > SPARK_DSV4_MODEL_LAYER_KIND_HCA )
		return(cudaErrorInvalidValue);
	SparkDsv4BuildAttentionIndicesKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(row_positions,indices,slot_counts,attention_slot_counts,row_count,column_count,index_slot_capacity,layer_kind);
	return(cudaGetLastError());
}

static cudaError_t SparkDsv4LaunchGateScoresSequence(cudaStream_t stream,
	const SparkDsv4LinearView *gate,const void *input_bf16,float *scores_f32,
	uint32_t row_count)
{
	dim3 grid(row_count,(gate->rows + (SPARK_LM_CTA_THREADS / SPARK_LM_WARP_LANES) - 1u) / (SPARK_LM_CTA_THREADS / SPARK_LM_WARP_LANES));
	SparkDsv4GateScoresKernel<<<grid,SPARK_LM_CTA_THREADS,gate->columns * (uint32_t)sizeof(float),stream>>>(gate->payload,input_bf16,scores_f32,row_count,gate->columns,gate->rows);
	return(cudaGetLastError());
}

static cudaError_t SparkDsv4LaunchGateSelectSequence(cudaStream_t stream,
	const float *scores_f32,const float *bias_f32,
	const uint32_t *tid2eid_u32,const uint32_t *token_ids,
	uint32_t row_count,uint32_t expert_count,uint32_t topk,float route_scale,
	uint32_t *indices_u32,float *weights_f32)
{
    if (scores_f32 == 0 || indices_u32 == 0 ||
        weights_f32 == 0 || row_count == 0u || expert_count == 0u ||
        expert_count > SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT ||
        topk == 0u || topk > SPARK_LM_MOE_MAX_TOPK ||
        topk > expert_count ||
        (tid2eid_u32 != 0 && token_ids == 0))
    {
        return cudaErrorInvalidValue;
    }
    SparkDsv4GateSelectKernel<<<
        row_count,
        SPARK_LM_CTA_THREADS,
        0u,
        stream>>>(
        scores_f32,
        bias_f32,
        tid2eid_u32,
        token_ids,
        row_count,
        expert_count,
        topk,
        route_scale,
        indices_u32,
        weights_f32);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkDsv4LaunchSwigluClamp(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width, float limit, const float *row_weights_f32, const uint32_t *weight_map)
{
	SparkDsv4SwigluClampKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(gate_bf16,up_bf16,row_count,width,limit,row_weights_f32,weight_map);
	return(cudaGetLastError());
}

// Init-time range scan of a hash routing table: any entry at or past
// the expert count trips the flag. Runs once per hash layer at
// initialize with a blocking readback - the load path is allowed to
// synchronize.
static __global__ void SparkDsv4ValidateTid2EidKernel(const uint32_t *tid2eid, uint64_t entry_count, uint32_t expert_count, uint32_t *violation_flag)
{
	uint64_t entry = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x,stride = (uint64_t)gridDim.x * blockDim.x;
	for (; entry < entry_count; entry += stride)
		if ( __ldg(tid2eid + entry) >= expert_count )
			atomicOr(violation_flag,1u);
}

extern "C" cudaError_t SparkDsv4LaunchValidateTid2Eid(cudaStream_t stream, const uint32_t *tid2eid, uint64_t entry_count, uint32_t *violation_flag)
{
	SparkDsv4ValidateTid2EidKernel<<<256u,SPARK_LM_CTA_THREADS,0,stream>>>(tid2eid,entry_count,SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT,violation_flag);
	return(cudaGetLastError());
}

static cudaError_t SparkDsv4GemmStatus(const char *site, int32_t status)
{
	if ( status == LM_LAUNCH_OK )
		return(cudaSuccess);
	fprintf(stderr,"dsv4_stage gemm_error site=%s status=%d\n",site,status);
	return(cudaErrorInvalidValue);
}

static cudaError_t SparkDsv4LaunchMoeRouteSequence(cudaStream_t stream,
	const uint32_t *route_expert,uint32_t rows,uint32_t expert_width,
	uint32_t *group_row_offset,uint32_t *route_packed_row,
	uint32_t *route_source_token,uint32_t *group_tile_prefix_w1,
	uint32_t *group_tile_prefix_w2)
{
	return(SparkDsv4GemmStatus("route",LmRouteBuild<SPARK_LM_CTA_THREADS,SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT>(route_expert,rows,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,group_row_offset,route_packed_row,route_source_token,expert_width,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,SparkLmSm121ExpertW13TileN(rows),SparkLmSm121ExpertW2TileN(rows),group_tile_prefix_w1,group_tile_prefix_w2,stream)));
}

static cudaError_t SparkDsv4LaunchGateRouteCooperative(
	cudaStream_t stream,const SparkDsv4LinearView *gate,
	const void *input_bf16,float *scores_f32,const float *bias_f32,
	const uint32_t *tid2eid_u32,const uint32_t *token_ids,uint32_t topk,
	float route_scale,uint32_t *indices_u32,float *weights_f32,
	uint32_t expert_width,uint32_t *group_row_offset,
	uint32_t *route_packed_row,uint32_t *route_source_token,
	uint32_t *group_tile_prefix_w1,uint32_t *group_tile_prefix_w2)
{
	const void *weight_bf16 = gate->payload;
	uint32_t blocks = (gate->rows + SPARK_LM_CTA_WARPS - 1u) /
		SPARK_LM_CTA_WARPS;
	uint32_t tile_m = LmLaunchGroupedTileM(1u,topk,gate->rows);
	uint32_t neuron_tiles_up = (expert_width +
		SparkLmSm121ExpertW13TileN(1u) - 1u) /
		SparkLmSm121ExpertW13TileN(1u);
	uint32_t neuron_tiles_down = (SPARK_DSV4_MODEL_HIDDEN_DIMENSION +
		SparkLmSm121ExpertW2TileN(1u) - 1u) /
		SparkLmSm121ExpertW2TileN(1u);
	void *arguments[] = {&weight_bf16,&input_bf16,&scores_f32,&bias_f32,
		&tid2eid_u32,&token_ids,&topk,&route_scale,&indices_u32,&weights_f32,
		&group_row_offset,&route_packed_row,&route_source_token,&tile_m,
		&neuron_tiles_up,&group_tile_prefix_w1,&neuron_tiles_down,
		&group_tile_prefix_w2};
	return(cudaLaunchCooperativeKernel(
		(void *)SparkDsv4GateRouteCooperativeKernel,dim3(blocks),
		dim3(SPARK_LM_CTA_THREADS),arguments,
		gate->columns * sizeof(float),stream));
}

static cudaError_t SparkDsv4LaunchGateRouteBatched(
	cudaStream_t stream,const SparkDsv4LinearView *gate,
	const void *input_bf16,float *scores_f32,const float *bias_f32,
	const uint32_t *tid2eid_u32,const uint32_t *token_ids,uint32_t row_count,
	uint32_t expert_count,uint32_t topk,float route_scale,
	uint32_t *indices_u32,float *weights_f32,uint32_t expert_width,
	uint32_t *group_row_offset,uint32_t *route_packed_row,
	uint32_t *route_source_token,uint32_t *group_tile_prefix_w1,
	uint32_t *group_tile_prefix_w2)
{
	cudaError_t error = SparkDsv4LaunchGateScoresSequence(stream,gate,
		input_bf16,scores_f32,row_count);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchGateSelectSequence(stream,scores_f32,bias_f32,
			tid2eid_u32,token_ids,row_count,expert_count,topk,route_scale,
			indices_u32,weights_f32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchMoeRouteSequence(stream,indices_u32,row_count,
			expert_width,group_row_offset,route_packed_row,route_source_token,
			group_tile_prefix_w1,group_tile_prefix_w2);
	return(error);
}

extern "C" cudaError_t SparkDsv4LaunchGateRoute(
	cudaStream_t stream,const SparkDsv4LinearView *gate,
	const void *input_bf16,float *scores_f32,const float *bias_f32,
	const uint32_t *tid2eid_u32,const uint32_t *token_ids,uint32_t row_count,
	uint32_t expert_count,uint32_t topk,float route_scale,
	uint32_t *indices_u32,float *weights_f32,uint32_t expert_width,
	uint32_t *group_row_offset,uint32_t *route_packed_row,
	uint32_t *route_source_token,uint32_t *group_tile_prefix_w1,
	uint32_t *group_tile_prefix_w2)
{
	if ( gate == 0 || gate->payload == 0 || input_bf16 == 0 ||
		scores_f32 == 0 || token_ids == 0 || indices_u32 == 0 ||
		weights_f32 == 0 || group_row_offset == 0 || route_packed_row == 0 ||
		route_source_token == 0 || group_tile_prefix_w1 == 0 ||
		group_tile_prefix_w2 == 0 || row_count == 0u ||
		row_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT ||
		gate->weight_format != SPARK_LM_WEIGHT_FORMAT_BF16 ||
		gate->rows != SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT ||
		gate->columns != SPARK_DSV4_MODEL_HIDDEN_DIMENSION ||
		expert_count != SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT ||
		topk != SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN || expert_width == 0u ||
		route_scale != SPARK_DSV4_MODEL_ROUTED_SCALING_FACTOR ||
		((bias_f32 == 0) == (tid2eid_u32 == 0)) )
		return(cudaErrorInvalidValue);
	if ( row_count == 1u )
		return(SparkDsv4LaunchGateRouteCooperative(stream,gate,input_bf16,
			scores_f32,bias_f32,tid2eid_u32,token_ids,topk,route_scale,
			indices_u32,weights_f32,expert_width,group_row_offset,
			route_packed_row,route_source_token,group_tile_prefix_w1,
			group_tile_prefix_w2));
	return(SparkDsv4LaunchGateRouteBatched(stream,gate,input_bf16,scores_f32,
		bias_f32,tid2eid_u32,token_ids,row_count,expert_count,topk,route_scale,
		indices_u32,weights_f32,expert_width,group_row_offset,route_packed_row,
		route_source_token,group_tile_prefix_w1,group_tile_prefix_w2));
}

extern "C" cudaError_t SparkDsv4LaunchExpertUp(cudaStream_t stream, const SparkDsv4LinearView *stacked, const void *input_bf16, const uint32_t *route_source_token, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t expert_width, uint32_t multiprocessor_count)
{
	/* W1 and W3 must be issued together by SparkDsv4LaunchFusedExpertW13Act. */
	(void)stream;
	(void)stacked;
	(void)input_bf16;
	(void)route_source_token;
	(void)group_row_offset;
	(void)group_tile_prefix;
	(void)output_bf16;
	(void)rows;
	(void)expert_width;
	(void)multiprocessor_count;
	return(cudaErrorInvalidValue);
}

extern "C" cudaError_t SparkDsv4LaunchFusedExpertW13Act(
	cudaStream_t stream,
	const SparkDsv4LinearView *w1,
	const SparkDsv4LinearView *w3,
	const void *input_bf16,
	const uint32_t *route_source_token,
	const uint32_t *group_row_offset,
	uint32_t *group_tile_prefix,
	void *activated_bf16,
	uint32_t rows,
	uint32_t expert_width,
	float limit,
	uint32_t multiprocessor_count)
{
	cudaError_t status;
	uint64_t required_rows =
		(uint64_t)SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT * expert_width;
	if ( w1 == 0 || w3 == 0 || input_bf16 == 0 ||
		route_source_token == 0 || group_row_offset == 0 ||
		group_tile_prefix == 0 || activated_bf16 == 0 ||
		w1->weight_format != SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1 ||
		w3->weight_format != SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1 ||
		w1->payload == 0 || w3->payload == 0 || w1->scale_data == 0 ||
		w3->scale_data == 0 || w1->rows != required_rows ||
		w3->rows != required_rows ||
		w1->columns != SPARK_DSV4_MODEL_HIDDEN_DIMENSION ||
		w3->columns != SPARK_DSV4_MODEL_HIDDEN_DIMENSION )
		return(cudaErrorInvalidValue);
	status = SparkDsv4RequireNativeDecodeShape(rows);
	if ( status != cudaSuccess )
		return(status);
	return(SparkLmHostLaunchSm121FusedExpertW13(stream,w1->payload,
		(const uint8_t *)w1->scale_data,w3->payload,
		(const uint8_t *)w3->scale_data,input_bf16,route_source_token,
		group_row_offset,group_tile_prefix,activated_bf16,rows,
		SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,
		SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT,
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION,expert_width,limit,
		multiprocessor_count));
}

extern "C" cudaError_t SparkDsv4LaunchFusedSharedW13Act(
	cudaStream_t stream,
	const SparkDsv4LinearView *w1,
	const SparkDsv4LinearView *w3,
	const void *input_bf16,
	void *activated_bf16,
	uint32_t rows,
	uint32_t expert_width,
	float limit)
{
	cudaError_t status;
	if ( w1 == 0 || w3 == 0 || input_bf16 == 0 || activated_bf16 == 0 ||
		w1->weight_format != SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 ||
		w3->weight_format != SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 ||
		w1->payload == 0 || w3->payload == 0 || w1->scale_data == 0 ||
		w3->scale_data == 0 || w1->rows != expert_width ||
		w3->rows != expert_width ||
		w1->columns != SPARK_DSV4_MODEL_HIDDEN_DIMENSION ||
		w3->columns != SPARK_DSV4_MODEL_HIDDEN_DIMENSION )
		return(cudaErrorInvalidValue);
	status = SparkDsv4RequireNativeDecodeShape(rows);
	if ( status != cudaSuccess )
		return(status);
	return(SparkLmHostLaunchSm121FusedDenseW13<
		SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(stream,w1->payload,
		(const uint8_t *)w1->scale_data,w3->payload,
		(const uint8_t *)w3->scale_data,input_bf16,activated_bf16,rows,
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION,expert_width,limit));
}

extern "C" cudaError_t SparkDsv4LaunchExpertDown(cudaStream_t stream, const SparkDsv4LinearView *stacked, const void *input_bf16, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t expert_width, uint32_t hidden_dimension, uint32_t multiprocessor_count)
{
	cudaError_t status;
	uint64_t required_rows =
		(uint64_t)SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT * hidden_dimension;
	if ( stacked == 0 || input_bf16 == 0 || group_row_offset == 0 ||
		group_tile_prefix == 0 || output_bf16 == 0 ||
		stacked->weight_format != SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1 ||
		stacked->payload == 0 || stacked->scale_data == 0 ||
		stacked->rows != required_rows || stacked->columns != expert_width )
		return(cudaErrorInvalidValue);
	status = SparkDsv4RequireNativeDecodeShape(rows);
	if ( status != cudaSuccess )
		return(status);
	return(SparkLmHostLaunchSm121ExpertW2(stream,stacked->payload,
		(const uint8_t *)stacked->scale_data,input_bf16,group_row_offset,
		group_tile_prefix,output_bf16,rows,
		SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,
		SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT,expert_width,hidden_dimension,
		multiprocessor_count));
}

extern "C" cudaError_t SparkDsv4LaunchMoePairReduce(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchMoePairReduce(stream,slot_out_bf16,inverse_map,pair_weights_f32,accum_bf16,row_count,SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,hidden_dimension));
}

extern "C" cudaError_t SparkDsv4LaunchMoePairReduceStrided(
	cudaStream_t stream,
	const void *slot_out_bf16,
	const uint32_t *inverse_map,
	const float *pair_weights_f32,
	void *accum_bf16,
	uint64_t accum_row_stride,
	uint32_t accum_offset,
	uint32_t row_count,
	uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchMoePairReduceStrided(stream,slot_out_bf16,
		inverse_map,pair_weights_f32,accum_bf16,accum_row_stride,
		accum_offset,row_count,SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,
		hidden_dimension));
}

extern "C" cudaError_t SparkDsv4LaunchAccumAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width)
{
	SparkDsv4AccumAddKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchAccumAddRelay(cudaStream_t stream,
	void *destination_bf16,const void *source_bf16,void *relay_bf16,
	uint32_t row_count,uint32_t width)
{
	if ( stream == 0 || destination_bf16 == 0 || source_bf16 == 0 ||
		relay_bf16 == 0 || row_count == 0u || width == 0u ||
		(width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkDsv4AccumAddRelayKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(
		destination_bf16,source_bf16,relay_bf16,row_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchAccumAddTp4Tree(
	cudaStream_t stream,
	void *destination_bf16,
	const void *const rank_devices[4],
	uint32_t tp_rank,
	uint32_t row_count,
	uint32_t width)
{
	if ( stream == 0 || destination_bf16 == 0 || rank_devices == 0 ||
		rank_devices[0] == 0 || rank_devices[1] == 0 ||
		rank_devices[2] == 0 || rank_devices[3] == 0 || tp_rank >= 4u ||
		row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkDsv4AccumAddTp4TreeKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(
		destination_bf16,rank_devices[0],rank_devices[1],rank_devices[2],
		rank_devices[3],tp_rank,row_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchIndexerScore(cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_page_table_indices, const uint32_t *physical_page_table, uint32_t page_table_stride, uint32_t entries_per_page, const uint32_t *slot_counts, const float *head_weights_f32, float *scores_f32, uint32_t row_count, uint32_t max_slots, uint32_t head_count, uint32_t head_dim)
{
    dim3 grid(
        row_count,
        (max_slots + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
    size_t shared_memory_bytes;

    if (q_bf16 == 0 || kv_cache_bf16 == 0 ||
        row_page_table_indices == 0 || physical_page_table == 0 ||
        page_table_stride == 0u || entries_per_page == 0u ||
        slot_counts == 0 ||
        head_weights_f32 == 0 || scores_f32 == 0 || row_count == 0u ||
        max_slots == 0u || head_count == 0u || head_dim == 0u ||
        head_dim > SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION ||
        (head_dim & 1u) != 0u)
    {
        return cudaErrorInvalidValue;
    }
    shared_memory_bytes =
        (size_t)head_count * head_dim * sizeof(float);
    SparkDsv4IndexerScoreKernel<<<
        grid,
        SPARK_LM_CTA_THREADS,
        shared_memory_bytes,
        stream>>>(
            q_bf16,
            kv_cache_bf16,
            lane_stride_elements,
            row_page_table_indices,
            physical_page_table,
            page_table_stride,
            entries_per_page,
            slot_counts,
            head_weights_f32,
            scores_f32,
            row_count,
            max_slots,
            head_count,
            head_dim);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkDsv4LaunchTopK(cudaStream_t stream, const float *scores_f32, const uint32_t *slot_counts, uint32_t max_slots, uint32_t topk, int32_t offset, int32_t *indices_out, uint64_t out_row_stride, uint32_t row_count)
{
    if (scores_f32 == 0 || slot_counts == 0 || indices_out == 0 ||
        max_slots == 0u || topk == 0u || topk > SPARK_DSV4_MODEL_INDEX_TOP_K ||
        row_count == 0u)
    {
        return cudaErrorInvalidValue;
    }
    SparkDsv4TopKKernel<<<row_count, SPARK_LM_CTA_THREADS, 0u, stream>>>(
        scores_f32,
        slot_counts,
        max_slots,
        topk,
        offset,
        indices_out,
        out_row_stride,
        row_count);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkDsv4LaunchHcMix(cudaStream_t stream, const void *streams_bf16, const float *fn_f32, float *mixes_f32, uint32_t row_count, uint32_t flat_dimension, uint32_t mix_rows, float epsilon)
{
#if defined(SPARK_DSV4_PRO_BUILD)
	uint32_t shared_elements = flat_dimension < SPARK_DSV4_HC_MIX_TILE ?
		flat_dimension : SPARK_DSV4_HC_MIX_TILE;
	SparkDsv4HcMixKernel<<<row_count,SPARK_LM_CTA_THREADS,shared_elements * (uint32_t)sizeof(float),stream>>>(streams_bf16,fn_f32,mixes_f32,row_count,flat_dimension,mix_rows,epsilon);
#else
	SparkDsv4HcMixKernel<<<row_count,SPARK_LM_CTA_THREADS,flat_dimension * (uint32_t)sizeof(float),stream>>>(streams_bf16,fn_f32,mixes_f32,row_count,flat_dimension,mix_rows,epsilon);
#endif
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHcSplitSinkhorn(cudaStream_t stream, const float *mixes_f32, const float *scale3_f32, const float *base_f32, uint32_t row_count, uint32_t hc, uint32_t iterations, float epsilon, float *pre_f32, float *post_f32, float *comb_f32)
{
	SparkDsv4HcSplitSinkhornKernel<<<(row_count + 63u) / 64u,64u,0,stream>>>(mixes_f32,scale3_f32,base_f32,row_count,hc,iterations,epsilon,pre_f32,post_f32,comb_f32);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHcMixSplitKSinkhorn(cudaStream_t stream, const void *streams_bf16, const float *fn_f32, const float *scale3_f32, const float *base_f32, float *partials_f32, float *mixes_f32, uint32_t row_count, uint32_t flat_dimension, uint32_t mix_rows, uint32_t hc, uint32_t iterations, float rms_epsilon, float hc_epsilon, float *pre_f32, float *post_f32, float *comb_f32)
{
	cudaError_t error;
	if ( streams_bf16 == 0 || fn_f32 == 0 || scale3_f32 == 0 || base_f32 == 0 || partials_f32 == 0 || mixes_f32 == 0 || pre_f32 == 0 || post_f32 == 0 || comb_f32 == 0 || row_count == 0u || flat_dimension != SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS || mix_rows != SPARK_DSV4_MODEL_HC_MIX_ROWS || hc != SPARK_DSV4_MODEL_HC_STREAM_COUNT || iterations == 0u || rms_epsilon <= 0.0f || hc_epsilon <= 0.0f )
		return(cudaErrorInvalidValue);
	SparkDsv4HcMixSplitKKernel<<<row_count * SPARK_DSV4_HC_SPLIT_K_COUNT,SPARK_LM_CTA_THREADS,0,stream>>>(streams_bf16,fn_f32,partials_f32,row_count,flat_dimension);
	error = cudaGetLastError();
	if ( error == cudaSuccess )
	{
		SparkDsv4HcMixSplitKFinalizeKernel<<<row_count,64u,0,stream>>>(partials_f32,scale3_f32,base_f32,row_count,iterations,rms_epsilon,hc_epsilon,mixes_f32,pre_f32,post_f32,comb_f32);
		error = cudaGetLastError();
	}
	return(error);
}

extern "C" cudaError_t SparkDsv4LaunchHcPreReduce(
	cudaStream_t stream,const void *streams_bf16,const float *pre_f32,
	void *reduced_bf16,void *residual_bf16,uint32_t row_count,uint32_t hc,
	uint32_t dimension)
{
	dim3 grid;
	uint32_t tiles_per_row,dimension_tiles;
	if ( streams_bf16 == 0 || pre_f32 == 0 || reduced_bf16 == 0 ||
		residual_bf16 == 0 || row_count == 0u || hc == 0u || dimension == 0u )
		return(cudaErrorInvalidValue);
	tiles_per_row = (SPARK_DSV4_HC_MINIMUM_BLOCKS + row_count - 1u) / row_count;
	dimension_tiles = (dimension + SPARK_DSV4_HC_ELEMENT_TILE - 1u) / SPARK_DSV4_HC_ELEMENT_TILE;
	tiles_per_row = tiles_per_row < dimension_tiles ? tiles_per_row : dimension_tiles;
	grid = dim3(tiles_per_row,row_count);
	SparkDsv4HcPreReduceKernel<<<grid,SPARK_DSV4_HC_ELEMENT_TILE,0,stream>>>(
		streams_bf16,pre_f32,reduced_bf16,residual_bf16,row_count,hc,
		dimension,tiles_per_row);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHcPost(cudaStream_t stream, const void *out_bf16, const void *residual_bf16, const float *post_f32, const float *comb_f32, void *streams_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension)
{
	dim3 grid;
	uint32_t tiles_per_row,dimension_tiles;
	if ( row_count == 0u || dimension == 0u )
		return(cudaErrorInvalidValue);
	tiles_per_row = (SPARK_DSV4_HC_MINIMUM_BLOCKS + row_count - 1u) / row_count;
	dimension_tiles = (dimension + SPARK_DSV4_HC_ELEMENT_TILE - 1u) / SPARK_DSV4_HC_ELEMENT_TILE;
	tiles_per_row = tiles_per_row < dimension_tiles ? tiles_per_row : dimension_tiles;
	grid = dim3(tiles_per_row,row_count);
	SparkDsv4HcPostKernel<<<grid,SPARK_DSV4_HC_ELEMENT_TILE,0,stream>>>(out_bf16,residual_bf16,post_f32,comb_f32,streams_bf16,row_count,hc,dimension,tiles_per_row);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkDsv4LaunchHcHeadReduce(cudaStream_t stream, const void *streams_bf16, const float *mixes_f32, float scale, const float *base_f32, float epsilon, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension)
{
	SparkDsv4HcHeadReduceKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(streams_bf16,mixes_f32,scale,base_f32,epsilon,reduced_bf16,row_count,hc,dimension);
	return(cudaGetLastError());
}
