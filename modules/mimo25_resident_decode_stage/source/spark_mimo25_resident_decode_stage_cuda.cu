#include "sparkpipe/spark_lm_kernels.cuh"
#include "sparkpipe/spark_mimo25_resident_decode_stage_firmware.h"

#include <math.h>

#define SPARK_MIMO25_ROUTER_SORT_CAPACITY 512u

/*
 * MiMo-V2.5 device kernels. The variant model header arrives via the
 * build's -include; nothing below names a variant. Shared machinery
 * (linear over bf16/f32-block fp8, rms norm, head argmax, embedding
 * gather) comes from spark_lm_kernels.cuh; this file holds what MiMo
 * adds: half-split rotate_half rope with a row stride and offset so it
 * runs in place on the fused qkv slices, in-place slice scaling for the
 * pre-cache value fold, the two-pass GQA decode attention that streams
 * the lane's k/v history (full range or the 128-ring with the sink in
 * the denominator), the f32 sigmoid gate, the biased ties-lower top-k
 * select with sum + 1e-20 weight normalization, the plain silu-mul, and
 * the weighted accumulate that lands each expert's OUTPUT times its
 * routing weight. Every kernel stays within fifty lines; the attention
 * decomposes through dot and probability helpers.
 */

// Half-split pairing on the FIRST rope_dim dims of each head: element i
// pairs with i + rope_dim/2. Row stride and offset address a head slice
// inside a wider fused row; the inverse conjugates.
static __global__ void SparkMimo25RopeKernel(void *data_bf16, uint64_t row_stride, uint32_t row_offset, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,pair = threadIdx.x,half = rope_dim / 2u;
	uint64_t base;
	float angle,cosine,sine,a,b;
	if ( row >= row_count || head >= head_count || pair >= half )
		return;
	base = ((uint64_t)row * row_stride) + row_offset + ((uint64_t)head * head_dim);
	angle = (float)row_positions[row] * __ldg(freqs_f32 + pair);
	__sincosf(angle,&sine,&cosine);
	sine = inverse != 0u ? -sine : sine;
	a = SparkLmBf16ToFloat(data_bf16,base + pair);
	b = SparkLmBf16ToFloat(data_bf16,base + pair + half);
	SparkLmFloatToBf16(data_bf16,base + pair,a * cosine - b * sine);
	SparkLmFloatToBf16(data_bf16,base + pair + half,b * cosine + a * sine);
}

// In-place scale of a column slice of every row - the value fold before
// the cache write.
static __global__ void SparkMimo25ScaleSliceKernel(void *data_bf16, uint64_t row_stride, uint32_t column_offset, uint32_t width, float scale, uint32_t row_count)
{
	uint64_t base = ((uint64_t)blockIdx.x * row_stride) + column_offset;
	uint32_t row = blockIdx.x,element,pair_count = ((base & 1u) == 0u) ? (width >> 1u) : 0u;
	float2 pair_value;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < pair_count; element += blockDim.x)
	{
		pair_value = SparkLmLoadBf16Pair(data_bf16,(base >> 1u) + element);
		SparkLmStoreBf16Pair(data_bf16,(base >> 1u) + element,pair_value.x * scale,pair_value.y * scale);
	}
	for (element = (pair_count << 1u) + threadIdx.x; element < width; element += blockDim.x)
		SparkLmFloatToBf16(data_bf16,base + element,SparkLmBf16ToFloat(data_bf16,base + element) * scale);
}

// Block max into a shared scalar: warp shuffles, per-warp scratch, a
// thread-zero scan, one barrier each side.
// The gate: f32 router weights against bf16 activations, sigmoid scores
// out - one warp per expert, activations staged shared.
static __global__ void SparkMimo25GateScoresKernel(const float *weight_f32, const void *input_bf16, float *scores_f32, uint32_t row_count, uint32_t input_dimension, uint32_t expert_count)
{
	extern __shared__ float gate_shared[];
	uint32_t row = blockIdx.x,warp_count = blockDim.x / SPARK_LM_WARP_LANES;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES,expert,element;
	float accumulator;
	float2 stage_pair,weight_pair;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (input_dimension >> 1u); element += blockDim.x)
	{
		stage_pair = SparkLmLoadBf16Pair(input_bf16,(((uint64_t)row * input_dimension) >> 1u) + element);
		gate_shared[element << 1u] = stage_pair.x;
		gate_shared[(element << 1u) + 1u] = stage_pair.y;
	}
	__syncthreads();
	for (expert = warp; expert < expert_count; expert += warp_count)
	{
		accumulator = 0.0f;
		for (element = lane; element < (input_dimension >> 1u); element += SPARK_LM_WARP_LANES)
		{
			weight_pair = __ldg(((const float2 *)weight_f32) + (((uint64_t)expert * input_dimension) >> 1u) + element);
			accumulator = fmaf(gate_shared[element << 1u],weight_pair.x,accumulator);
			accumulator = fmaf(gate_shared[(element << 1u) + 1u],weight_pair.y,accumulator);
		}
		accumulator = SparkLmWarpReduceSum(accumulator);
		if ( lane == 0u )
			scores_f32[((uint64_t)row * expert_count) + expert] = SparkLmSigmoid(accumulator);
	}
}

/*
 * noaux_tc with degenerate groups: plain top-k on scores + bias, ties to
 * the lower index by strict-greater scanning, weights from the ORIGINAL
 * sigmoid scores normalized by sum + epsilon and scaled - the routing
 * weight is applied downstream on the expert OUTPUT.
 */
static __global__ void SparkMimo25GateSelectKernel(
    const float *scores_f32,
    const float *bias_f32,
    uint32_t row_count,
    uint32_t expert_count,
    uint32_t topk,
    float epsilon,
    float route_scale,
    uint32_t *indices_u32,
    float *weights_f32)
{
    __shared__ uint64_t ordered_keys[SPARK_MIMO25_ROUTER_SORT_CAPACITY];
    uint32_t row;
    uint32_t expert;
    uint32_t rank;
    uint64_t selected_key;
    uint32_t selected_expert;
    float selected_score;
    float selected_total;
    const float *row_scores;

    static_assert(
        SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT <=
            SPARK_MIMO25_ROUTER_SORT_CAPACITY,
        "MiMo expert count exceeds router sort capacity");
    row = blockIdx.x;
    if (row >= row_count ||
        expert_count == 0u ||
        expert_count > SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT ||
        topk == 0u || topk > SPARK_LM_MOE_MAX_TOPK ||
        topk > expert_count)
    {
        return;
    }
    row_scores = scores_f32 + ((uint64_t)row * expert_count);
    for (expert = threadIdx.x;
         expert < SPARK_MIMO25_ROUTER_SORT_CAPACITY;
         expert += blockDim.x)
    {
        ordered_keys[expert] = expert < expert_count
            ? SparkLmOrderedTopKKey(
                row_scores[expert] + bias_f32[expert],
                expert)
            : 0u;
    }
    __syncthreads();
    SparkLmBitonicSortKeysAscending<SPARK_MIMO25_ROUTER_SORT_CAPACITY>(
        ordered_keys);

    rank = threadIdx.x;
    selected_key = rank < topk
        ? ordered_keys[SPARK_MIMO25_ROUTER_SORT_CAPACITY - 1u - rank]
        : 0u;
    selected_expert = selected_key != 0u
        ? 0xffffffffu - (uint32_t)selected_key
        : 0xffffffffu;
    selected_score = selected_expert < expert_count
        ? row_scores[selected_expert]
        : 0.0f;
    selected_total = threadIdx.x < SPARK_LM_WARP_LANES
        ? SparkLmWarpReduceSum(selected_score)
        : 0.0f;
    selected_total = __shfl_sync(0xffffffffu, selected_total, 0u);

    if (rank < topk && selected_expert < expert_count)
    {
        indices_u32[((uint64_t)row * topk) + rank] = selected_expert;
        weights_f32[((uint64_t)row * topk) + rank] =
            selected_score / (selected_total + epsilon) * route_scale;
    }
}

// Plain silu(gate) * up, no clamp anywhere in this model.
static __global__ void SparkMimo25SiluMulKernel(const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float2 gate_pair,up_pair;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (width >> 1u); element += blockDim.x)
	{
		gate_pair = SparkLmLoadBf16Pair(gate_bf16,offset + element);
		up_pair = SparkLmLoadBf16Pair(up_bf16,offset + element);
		SparkLmStoreBf16Pair(up_bf16,offset + element,SparkLmSwish(gate_pair.x) * up_pair.x,SparkLmSwish(gate_pair.y) * up_pair.y);
	}
	for (element = ((width >> 1u) << 1u) + threadIdx.x; element < width; element += blockDim.x)
		SparkLmFloatToBf16(up_bf16,((uint64_t)row * width) + element,SparkLmSwish(SparkLmBf16ToFloat(gate_bf16,((uint64_t)row * width) + element)) * SparkLmBf16ToFloat(up_bf16,((uint64_t)row * width) + element));
}

// destination += source * weight, the weight read from a device f32 - the
// routed expert's OUTPUT scaled at accumulation, per the reference.
static __global__ void SparkMimo25AccumScaledAddKernel(void *destination_bf16, const void *source_bf16, const float *weight_f32, uint32_t row_count, uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float weight = weight_f32 != 0 ? weight_f32[0] : 1.0f;
	float2 destination_pair,source_pair;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (width >> 1u); element += blockDim.x)
	{
		destination_pair = SparkLmLoadBf16Pair(destination_bf16,offset + element);
		source_pair = SparkLmLoadBf16Pair(source_bf16,offset + element);
		SparkLmStoreBf16Pair(destination_bf16,offset + element,fmaf(source_pair.x,weight,destination_pair.x),fmaf(source_pair.y,weight,destination_pair.y));
	}
	for (element = ((width >> 1u) << 1u) + threadIdx.x; element < width; element += blockDim.x)
		SparkLmFloatToBf16(destination_bf16,((uint64_t)row * width) + element,SparkLmBf16ToFloat(destination_bf16,((uint64_t)row * width) + element) + SparkLmBf16ToFloat(source_bf16,((uint64_t)row * width) + element) * weight);
}

static __global__ void SparkMimo25ResidualAddKernel(void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float2 hidden_pair,delta_pair;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (width >> 1u); element += blockDim.x)
	{
		hidden_pair = SparkLmLoadBf16Pair(hidden_bf16,offset + element);
		delta_pair = SparkLmLoadBf16Pair(delta_bf16,offset + element);
		SparkLmStoreBf16Pair(hidden_bf16,offset + element,hidden_pair.x + delta_pair.x,hidden_pair.y + delta_pair.y);
	}
	for (element = ((width >> 1u) << 1u) + threadIdx.x; element < width; element += blockDim.x)
		SparkLmFloatToBf16(hidden_bf16,((uint64_t)row * width) + element,SparkLmBf16ToFloat(hidden_bf16,((uint64_t)row * width) + element) + SparkLmBf16ToFloat(delta_bf16,((uint64_t)row * width) + element));
}

extern "C" cudaError_t SparkMimo25LaunchFusedResidualRmsNorm(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
    size_t shared_memory_bytes = (size_t)dimension * sizeof(float);

    SparkLmFusedResidualRmsNormKernel<<<row_count, SPARK_LM_CTA_THREADS, shared_memory_bytes, stream>>>(hidden_bf16, delta_bf16, gain_bf16, output_bf16, row_count, dimension, epsilon);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkMimo25LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
    size_t shared_memory_bytes = (size_t)dimension * sizeof(float);

    SparkLmRmsNormKernel<<<row_count, SPARK_LM_CTA_THREADS, shared_memory_bytes, stream>>>(input_bf16, gain_bf16, output_bf16, row_count, dimension, epsilon);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkMimo25LaunchLinear(cudaStream_t stream, const SparkMimo25LinearView *view, const void *payload, const void *scale, const void *input_bf16, void *output_bf16, uint32_t row_count)
{
	return(SparkLmHostLaunchBatchedLinear<128u>(stream,view->weight_format,payload != 0 ? payload : view->payload,scale != 0 ? scale : view->scale,input_bf16,output_bf16,row_count,view->columns,view->rows));
}

extern "C" cudaError_t SparkMimo25LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	SparkLmEmbeddingGatherKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(token_ids,embedding_bf16,hidden_bf16,row_count,hidden_dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension)
{
	SparkLmHeadArgmaxKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,head_weight_bf16,0,output_token_ids,row_count,hidden_dimension,candidate_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchRope(cudaStream_t stream, void *data_bf16, uint64_t row_stride, uint32_t row_offset, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse)
{
	dim3 grid(row_count,head_count);
	SparkMimo25RopeKernel<<<grid,rope_dim / 2u,0,stream>>>(data_bf16,row_stride,row_offset,freqs_f32,row_positions,row_count,head_count,head_dim,rope_dim,inverse);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchScaleSlice(cudaStream_t stream, void *data_bf16, uint64_t row_stride, uint32_t column_offset, uint32_t width, float scale, uint32_t row_count)
{
	SparkMimo25ScaleSliceKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(data_bf16,row_stride,column_offset,width,scale,row_count);
	return(cudaGetLastError());
}

// One block per row scatters the row's k and v qkv slices into the
// lane's cache at position % window (dense when window is zero): raw
// 16-byte moves, replacing the per-row pair of async device copies the
// module used to submit - two launches per layer instead of two per ROW
// per layer.
static __global__ void SparkMimo25CacheScatterKernel(const void *qkv_bf16, uint64_t qkv_row_stride, uint32_t k_offset, uint32_t k_width, uint32_t v_offset, uint32_t v_width, void *k_cache_bf16, void *v_cache_bf16, uint64_t k_lane_stride, uint64_t v_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t window_slots, uint32_t row_count)
{
	uint32_t row = blockIdx.x,element;
	uint64_t position,slot,k_base,v_base,k_source,v_source;
	if ( row >= row_count )
		return;
	position = row_positions[row];
	slot = window_slots != 0u ? position % window_slots : position;
	k_base = (((uint64_t)row_lane_indices[row] * k_lane_stride) + (slot * k_width)) >> 3u;
	v_base = (((uint64_t)row_lane_indices[row] * v_lane_stride) + (slot * v_width)) >> 3u;
	k_source = (((uint64_t)row * qkv_row_stride) + k_offset) >> 3u;
	v_source = (((uint64_t)row * qkv_row_stride) + v_offset) >> 3u;
	for (element = threadIdx.x; element < (k_width >> 3u); element += blockDim.x)
		((uint4 *)k_cache_bf16)[k_base + element] = __ldg(((const uint4 *)qkv_bf16) + k_source + element);
	for (element = threadIdx.x; element < (v_width >> 3u); element += blockDim.x)
		((uint4 *)v_cache_bf16)[v_base + element] = __ldg(((const uint4 *)qkv_bf16) + v_source + element);
}

extern "C" cudaError_t SparkMimo25LaunchMoeGroup(cudaStream_t stream, const uint32_t *pair_expert_ids, uint32_t pair_count, uint32_t *expert_offsets, uint32_t *grouped_rows, uint32_t *grouped_weight_slots, uint32_t *inverse_map)
{
	static_assert(SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT <= SPARK_LM_MOE_MAX_EXPERTS,"expert table exceeds group kernel shared capacity");
	static_assert(SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN <= SPARK_LM_MOE_MAX_TOPK,"topk exceeds reduce register cache");
	return(SparkLmHostLaunchMoeGroup(stream,pair_expert_ids,pair_count,SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT,SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN,expert_offsets,grouped_rows,grouped_weight_slots,inverse_map));
}

// One launch covers every routed expert for one projection shape: the
// grid's z axis is the expert table, offsets live on device, empty
// tiles exit immediately. rows_per_expert and columns describe ONE
// expert's matrix; strides derive from the fp8 block-scale layout.
extern "C" cudaError_t SparkMimo25LaunchExpertTileAll(cudaStream_t stream, const SparkMimo25LinearView *stacked, const void *unused_payload, const void *unused_scale, const void *input_bf16, const uint32_t *grouped_rows, const uint32_t *expert_offsets, void *output_bf16, uint32_t max_group_slots, uint64_t rows_per_expert, uint64_t columns)
{
	uint64_t payload_stride = rows_per_expert * columns;
	uint64_t scale_stride = (rows_per_expert / SPARK_MIMO25_MODEL_FP8_SCALE_BLOCK) * (columns / SPARK_MIMO25_MODEL_FP8_SCALE_BLOCK) * sizeof(float);
	dim3 grid((max_group_slots + SPARK_LM_TILE - 1u) / SPARK_LM_TILE,((uint32_t)rows_per_expert + SPARK_LM_TILE_N - 1u) / SPARK_LM_TILE_N,SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT);
	if ( stacked == 0 ||
		(stacked->weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 &&
			(stacked->scale == 0 || (columns % 128u) != 0u ||
				(rows_per_expert % 128u) != 0u)) )
		return(cudaErrorInvalidValue);
	(void)unused_payload;
	(void)unused_scale;
	SparkLmExpertTileAllKernel<128u><<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(stacked->weight_format,stacked->payload,stacked->scale,payload_stride,scale_stride,input_bf16,grouped_rows,expert_offsets,output_bf16,(uint32_t)columns,(uint32_t)rows_per_expert);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchMoePairReduce(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint32_t row_count)
{
	return(SparkLmHostLaunchMoePairReduceOverwrite(stream,slot_out_bf16,inverse_map,pair_weights_f32,accum_bf16,row_count,SPARK_MIMO25_MODEL_EXPERTS_PER_TOKEN,SPARK_MIMO25_MODEL_HIDDEN_DIMENSION));
}

extern "C" cudaError_t SparkMimo25LaunchCacheScatter(cudaStream_t stream, const void *qkv_bf16, uint64_t qkv_row_stride, uint32_t k_offset, uint32_t k_width, uint32_t v_offset, uint32_t v_width, void *k_cache_bf16, void *v_cache_bf16, uint64_t k_lane_stride, uint64_t v_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t window_slots, uint32_t row_count)
{
	SparkMimo25CacheScatterKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(qkv_bf16,qkv_row_stride,k_offset,k_width,v_offset,v_width,k_cache_bf16,v_cache_bf16,k_lane_stride,v_lane_stride,row_lane_indices,row_positions,window_slots,row_count);
	return(cudaGetLastError());
}

// Two-phase full-vocabulary head: the tensor-core tile computes the
// logits with the head weights read once per SIXTEEN rows, then the
// argmax reduce walks the logits - sixteenfold less weight traffic than
// the one-block-per-row argmax at a full row tile.
static_assert(SPARK_MIMO25_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP == SPARK_LM_HEAD_SCREEN_CAP,"screen cap must match the shared kernels");

extern "C" cudaError_t SparkMimo25LaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchHeadShadowQuantize<SPARK_LM_HEAD_SHADOW_GROUP>(stream,head_bf16,shadow_payload,shadow_scale,error_norm,candidate_count,hidden_dimension));
}

// Screened exact head: coarse fp4 tile, certified screen, exact rescore
// of the survivors, device-side overflow fallback. The emitted token is

extern "C" cudaError_t SparkMimo25LaunchAttnDecode(cudaStream_t stream, const void *q_bf16, uint64_t q_row_stride, const void *k_cache_bf16, const void *v_cache_bf16, uint64_t k_lane_stride, uint64_t v_lane_stride, uint64_t k_slot_stride, uint64_t v_slot_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, const float *sink_f32, float scale, void *out_bf16, uint32_t row_count, uint32_t head_count, uint32_t group_size, uint32_t head_dim, uint32_t value_dim, uint32_t window_slots)
{
    return SparkLmHostLaunchAdaptiveAttnDecode(
        stream,
        q_bf16,
        q_row_stride,
        k_cache_bf16,
        v_cache_bf16,
        k_lane_stride,
        v_lane_stride,
        k_slot_stride,
        v_slot_stride,
        row_lane_indices,
        row_positions,
        sink_f32,
        scale,
        out_bf16,
        row_count,
        head_count,
        group_size,
        head_dim,
        value_dim,
        window_slots);
}

extern "C" cudaError_t SparkMimo25LaunchGateScores(cudaStream_t stream, const SparkMimo25LinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count)
{
	SparkMimo25GateScoresKernel<<<row_count,SPARK_LM_CTA_THREADS,gate->columns * (uint32_t)sizeof(float),stream>>>((const float *)gate->payload,input_bf16,scores_f32,row_count,gate->columns,gate->rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, uint32_t row_count, uint32_t expert_count, uint32_t topk, float epsilon, float route_scale, uint32_t *indices_u32, float *weights_f32)
{
    if (scores_f32 == 0 || bias_f32 == 0 ||
        indices_u32 == 0 || weights_f32 == 0 || row_count == 0u ||
        expert_count == 0u ||
        expert_count > SPARK_MIMO25_MODEL_ROUTED_EXPERT_COUNT ||
        topk == 0u || topk > SPARK_LM_MOE_MAX_TOPK ||
        topk > expert_count)
    {
        return cudaErrorInvalidValue;
    }
    SparkMimo25GateSelectKernel<<<
        row_count,
        SPARK_LM_CTA_THREADS,
        0u,
        stream>>>(
        scores_f32,
        bias_f32,
        row_count,
        expert_count,
        topk,
        epsilon,
        route_scale,
        indices_u32,
        weights_f32);
    return cudaGetLastError();
}

extern "C" cudaError_t SparkMimo25LaunchSiluMul(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width)
{
	SparkMimo25SiluMulKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(gate_bf16,up_bf16,row_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchGatherLinear(cudaStream_t stream, const SparkMimo25LinearView *view, const void *payload, const void *scale, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count)
{
	dim3 grid(slot_count,(view->rows + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
	uint32_t shared_bytes = view->columns * (uint32_t)sizeof(float);
	SparkLmGatherLinearKernel<128u><<<grid,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(view->weight_format,payload != 0 ? payload : view->payload,scale != 0 ? scale : view->scale,input_bf16,input_row_map,output_bf16,slot_count,view->columns,view->rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchExpertTile(cudaStream_t stream, const SparkMimo25LinearView *view, const void *payload, const void *scale, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count)
{
	dim3 grid((slot_count + SPARK_LM_TILE - 1u) / SPARK_LM_TILE,(view->rows + SPARK_LM_TILE_N - 1u) / SPARK_LM_TILE_N);
	const void *effective_scale = scale != 0 ? scale : view->scale;
	if ( view->weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 &&
		(effective_scale == 0 || (view->columns % 128u) != 0u ||
			(view->rows % 128u) != 0u) )
		return(cudaErrorInvalidValue);
	SparkLmExpertTileKernel<128u><<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(view->weight_format,payload != 0 ? payload : view->payload,effective_scale,input_bf16,input_row_map,output_bf16,slot_count,view->columns,view->rows);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchScatterScaledAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, const uint32_t *row_map, const float *weights_f32, const uint32_t *weight_map, uint32_t slot_count, uint32_t width)
{
	SparkLmScatterScaledAddKernel<<<slot_count,SPARK_LM_CTA_THREADS,0,stream>>>(destination_bf16,source_bf16,row_map,weights_f32,weight_map,slot_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchAccumScaledAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, const float *weight_f32, uint32_t row_count, uint32_t width)
{
	SparkMimo25AccumScaledAddKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(destination_bf16,source_bf16,weight_f32,row_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMimo25LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t width)
{
	SparkMimo25ResidualAddKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,delta_bf16,row_count,width);
	return(cudaGetLastError());
}
