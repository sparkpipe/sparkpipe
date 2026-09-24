#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "sparkpipe/spark_gemma4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_hybrid_state.h"
#include "sparkpipe/spark_rope_plan.h"
#include "sparkpipe/spark_lm_kernels.cuh"
#include "sparkpipe/spark_tp_mesh_kernels.cuh"
#include "inference/kernels/frame_error.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/project.cuh"
#include "runtime/launch.h"
#define SPARK_FAMILY_CAMEL Gemma4
#define SPARK_FAMILY_UPPER GEMMA4
#define SPARK_FAMILY_LOWER gemma4

#include "sparkpipe/family/spark_family.h"

#define SPARK_GEMMA4_CUDA_THREADS 256u
#define SPARK_GEMMA4_CUDA_PAGE_SLOTS SPARK_GEMMA4_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS

typedef LmKvGeometry<SPARK_HYBRID_KV_HEAD_SLOT_BYTES(SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES),SPARK_GEMMA4_CUDA_PAGE_SLOTS,true> SparkGemma4SlidingGeometry1;
typedef LmKvGeometry<SPARK_HYBRID_KV_SLOT_BYTES(2u,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES),SPARK_GEMMA4_CUDA_PAGE_SLOTS,true> SparkGemma4SlidingGeometry2;
typedef LmKvGeometry<SPARK_HYBRID_KV_HEAD_SLOT_BYTES(SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES),SPARK_GEMMA4_CUDA_PAGE_SLOTS,true> SparkGemma4FullGeometry1;

static_assert(SparkGemma4SlidingGeometry1::kSlotBytes ==
		SPARK_HYBRID_KV_HEAD_SLOT_BYTES(SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,
		    SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES),
	"one sliding kv head stores a bf16 k and v row");
static_assert(SparkGemma4SlidingGeometry2::kSlotBytes ==
		SPARK_HYBRID_KV_SLOT_BYTES(2u,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,
		    SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES),
	"two sliding kv heads store bf16 k and v rows");
static_assert(SparkGemma4FullGeometry1::kSlotBytes ==
		SPARK_HYBRID_KV_HEAD_SLOT_BYTES(SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION,
		    SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES),
	"one full-attention kv head stores a bf16 shared k/v row pair");

static int32_t SparkGemma4BuildKvView(LmKvView *view, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, LmKvAccessError *access_error)
{
	if ( pool == 0 || page_table == 0 )
		return(-1);
	return(LmKvViewInitialize(view,(uint8_t *)pool,page_table,page_table_stride,sequence_count,pool_page_count,access_error));
}

static __global__ void SparkGemma4EmbeddingGatherShardedScaledKernel(const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t vocab_base, uint32_t vocab_rows)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t row = (uint32_t)(index / SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION),element = (uint32_t)(index % SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION);
	uint32_t token;
	float value;
	if ( row >= row_count )
		return;
	token = token_ids[row];
	value = token >= vocab_base && token < vocab_base + vocab_rows
		? SparkLmBf16ToFloat(embedding_bf16,((uint64_t)(token - vocab_base) * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION) + element)
		: 0.0f;
	SparkLmFloatToBf16(hidden_bf16,index,value * SPARK_GEMMA4_MODEL_EMBED_SCALE);
}

extern "C" cudaError_t SparkGemma4LaunchEmbeddingGatherShardedScaled(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t vocab_base, uint32_t vocab_rows)
{
	uint64_t elements = (uint64_t)row_count * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
	if ( embedding_bf16 == 0 || vocab_rows == 0u )
		return(cudaErrorInvalidValue);
	SparkGemma4EmbeddingGatherShardedScaledKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(token_ids,embedding_bf16,hidden_bf16,row_count,vocab_base,vocab_rows);
	return(cudaGetLastError());
}

static __global__ void SparkGemma4ResidualAddKernel(void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t row = (uint32_t)(index / dimension),element = (uint32_t)(index % dimension);
	float value;
	if ( row >= row_count )
		return;
	value = SparkLmBf16ToFloat(hidden_bf16,index) + SparkLmBf16ToFloat(delta_bf16,index);
	SparkLmFloatToBf16(hidden_bf16,index,value);
}

static __global__ void SparkGemma4BranchAddKernel(void *sum_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t row = (uint32_t)(index / dimension),element = (uint32_t)(index % dimension);
	float value;
	if ( row >= row_count )
		return;
	value = SparkLmBf16ToFloat(sum_bf16,index) + SparkLmBf16ToFloat(delta_bf16,index);
	SparkLmFloatToBf16(sum_bf16,index,value);
}

extern "C" cudaError_t SparkGemma4LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	size_t shared_memory_bytes = (size_t)dimension * sizeof(float);
	SparkLmRmsNormKernel<<<row_count,SPARK_LM_CTA_THREADS,shared_memory_bytes,stream>>>(input_bf16,gain_bf16,output_bf16,row_count,dimension,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchHeadRmsNorm(cudaStream_t stream, const void *input_bf16, const void *weight_or_null_bf16, void *output_bf16, uint32_t row_count, uint32_t heads, uint32_t head_dimension, float epsilon)
{
	LmHeadRmsNormKernel<SPARK_GEMMA4_CUDA_THREADS><<<dim3(heads,row_count),SPARK_GEMMA4_CUDA_THREADS,0,stream>>>((const uint16_t *)input_bf16,(const uint16_t *)weight_or_null_bf16,(uint16_t *)output_bf16,row_count,heads,head_dimension,epsilon,1.0f);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchLinear(cudaStream_t stream, const SparkGemma4LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count)
{
	(void)row_count;
	if ( view == 0 || view->weight_payload == 0 || view->input_dimension == 0u || view->output_dimension == 0u )
		return(cudaErrorInvalidValue);
	return(SparkLmHostLaunchBatchedLinear<32u>(stream,view->weight_format,view->weight_payload,view->weight_scale_e8m0,input_bf16,output_bf16,row_count,view->input_dimension,view->output_dimension));
}

extern "C" cudaError_t SparkGemma4LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t elements = (uint64_t)row_count * dimension;
	if ( hidden_bf16 == 0 || delta_bf16 == 0 || row_count == 0u || dimension == 0u )
		return(cudaErrorInvalidValue);
	SparkGemma4ResidualAddKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,delta_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchBranchAdd(cudaStream_t stream, void *sum_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t elements = (uint64_t)row_count * dimension;
	if ( sum_bf16 == 0 || delta_bf16 == 0 || row_count == 0u || dimension == 0u )
		return(cudaErrorInvalidValue);
	SparkGemma4BranchAddKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(sum_bf16,delta_bf16,row_count,dimension);
	return(cudaGetLastError());
}

static __device__ __forceinline__ float SparkGemma4GeluTanh(float value)
{
	float cube = value * value * value;
	return(0.5f * value * (1.0f + tanhf(0.7978845608028654f * (value + 0.044715f * cube))));
}

static __global__ void SparkGemma4GatedGeluKernel(void *gate_up_bf16, uint32_t row_count, uint32_t intermediate)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t row = (uint32_t)(index / intermediate),element = (uint32_t)(index % intermediate);
	uint64_t row_base;
	float gate,up;
	if ( row >= row_count )
		return;
	row_base = (uint64_t)row * intermediate * 2u;
	gate = SparkLmBf16ToFloat(gate_up_bf16,row_base + element);
	up = SparkLmBf16ToFloat(gate_up_bf16,row_base + intermediate + element);
	SparkLmFloatToBf16(gate_up_bf16,row_base + element,SparkGemma4GeluTanh(gate) * up);
}

extern "C" cudaError_t SparkGemma4LaunchGatedGelu(cudaStream_t stream, void *gate_up_bf16, uint32_t row_count, uint32_t intermediate)
{
	uint64_t elements = (uint64_t)row_count * intermediate;
	if ( gate_up_bf16 == 0 || row_count == 0u || intermediate == 0u )
		return(cudaErrorInvalidValue);
	SparkGemma4GatedGeluKernel<<<(uint32_t)((elements + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(gate_up_bf16,row_count,intermediate);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchRope(cudaStream_t stream, void *q_bf16, const uint32_t *positions, uint32_t row_count, uint32_t heads, const SparkRopeDomain *domain)
{
	if ( domain == 0 || domain->head_dimension < domain->rope_dimension )
		return(cudaErrorInvalidValue);
	LmRopePerHeadKernel<SPARK_GEMMA4_CUDA_THREADS><<<dim3(row_count,heads),SPARK_GEMMA4_CUDA_THREADS,0,stream>>>((uint16_t *)q_bf16,positions,heads,domain->head_dimension,domain->rope_dimension,domain->theta,domain->inv_freq_table,domain->attention_scale,domain->rope_offset);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchSlidingWindowPositions(cudaStream_t stream, const uint32_t *sequence_of_row, const uint32_t *context_lengths, const uint32_t *row_positions, uint32_t row_count, uint32_t *positions_out)
{
	LmBuildSlidingWindowPositionsKernel<SPARK_GEMMA4_CUDA_THREADS><<<row_count,SPARK_GEMMA4_CUDA_THREADS,0,stream>>>(sequence_of_row,context_lengths,row_positions,row_count,SPARK_GEMMA4_MODEL_SLIDING_WINDOW_TOKENS,positions_out);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchKvStoreSliding(cudaStream_t stream, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, void *access_error, const void *key_bf16, const void *value_bf16, const uint32_t *sequence_of_row, const uint32_t *positions, uint32_t row_count, uint32_t kv_heads)
{
	LmKvView view;
	if ( SparkGemma4BuildKvView(&view,pool,page_table,page_table_stride,sequence_count,pool_page_count,(LmKvAccessError *)access_error) != 0 )
		return(cudaErrorInvalidValue);
	if ( kv_heads == 1u )
		LmGqaKvStoreKernel<SparkGemma4SlidingGeometry1,SPARK_GEMMA4_CUDA_THREADS,1u,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION><<<row_count,SPARK_GEMMA4_CUDA_THREADS,0,stream>>>(view,(const uint16_t *)key_bf16,(const uint16_t *)value_bf16,sequence_of_row,positions,row_count);
	else if ( kv_heads == 2u )
		LmGqaKvStoreKernel<SparkGemma4SlidingGeometry2,SPARK_GEMMA4_CUDA_THREADS,2u,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION><<<row_count,SPARK_GEMMA4_CUDA_THREADS,0,stream>>>(view,(const uint16_t *)key_bf16,(const uint16_t *)value_bf16,sequence_of_row,positions,row_count);
	else
		return(cudaErrorInvalidValue);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchKvStoreFull(cudaStream_t stream, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, void *access_error, const void *key_bf16, const void *value_bf16, const uint32_t *sequence_of_row, const uint32_t *positions, uint32_t row_count)
{
	LmKvView view;
	if ( SparkGemma4BuildKvView(&view,pool,page_table,page_table_stride,sequence_count,pool_page_count,(LmKvAccessError *)access_error) != 0 )
		return(cudaErrorInvalidValue);
	LmGqaKvStoreKernel<SparkGemma4FullGeometry1,SPARK_GEMMA4_CUDA_THREADS,1u,SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION><<<row_count,SPARK_GEMMA4_CUDA_THREADS,0,stream>>>(view,(const uint16_t *)key_bf16,(const uint16_t *)value_bf16,sequence_of_row,positions,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchAttentionDecodeSliding(cudaStream_t stream, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, void *access_error, const void *query_bf16, const uint32_t *sequence_of_row, const uint32_t *context_lengths, const uint32_t *window_positions, uint32_t query_heads, void *output_bf16, uint32_t row_count, uint32_t kv_heads)
{
	LmKvView view;
	if ( SparkGemma4BuildKvView(&view,pool,page_table,page_table_stride,sequence_count,pool_page_count,(LmKvAccessError *)access_error) != 0 )
		return(cudaErrorInvalidValue);
	if ( kv_heads == 1u )
		LmGqaAttentionDecodeKernel<SparkGemma4SlidingGeometry1,SPARK_GEMMA4_CUDA_THREADS,1u,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION><<<dim3(row_count,query_heads),SPARK_GEMMA4_CUDA_THREADS,0,stream>>>((const uint16_t *)query_bf16,view,sequence_of_row,context_lengths,window_positions,SPARK_GEMMA4_MODEL_SLIDING_WINDOW_TOKENS,query_heads,SPARK_GEMMA4_MODEL_QK_SCALE,(uint16_t *)output_bf16,0);
	else if ( kv_heads == 2u )
		LmGqaAttentionDecodeKernel<SparkGemma4SlidingGeometry2,SPARK_GEMMA4_CUDA_THREADS,2u,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION><<<dim3(row_count,query_heads),SPARK_GEMMA4_CUDA_THREADS,0,stream>>>((const uint16_t *)query_bf16,view,sequence_of_row,context_lengths,window_positions,SPARK_GEMMA4_MODEL_SLIDING_WINDOW_TOKENS,query_heads,SPARK_GEMMA4_MODEL_QK_SCALE,(uint16_t *)output_bf16,0);
	else
		return(cudaErrorInvalidValue);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchAttentionDecodeFull(cudaStream_t stream, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, void *access_error, const void *query_bf16, const uint32_t *sequence_of_row, const uint32_t *context_lengths, uint32_t query_heads, void *output_bf16, uint32_t row_count)
{
	LmKvView view;
	if ( SparkGemma4BuildKvView(&view,pool,page_table,page_table_stride,sequence_count,pool_page_count,(LmKvAccessError *)access_error) != 0 )
		return(cudaErrorInvalidValue);
	LmGqaAttentionDecodeKernel<SparkGemma4FullGeometry1,SPARK_GEMMA4_CUDA_THREADS,1u,SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION,SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION><<<dim3(row_count,query_heads),SPARK_GEMMA4_CUDA_THREADS,0,stream>>>((const uint16_t *)query_bf16,view,sequence_of_row,context_lengths,0,0u,query_heads,SPARK_GEMMA4_MODEL_QK_SCALE,(uint16_t *)output_bf16,0);
	return(cudaGetLastError());
}

static __global__ void SparkGemma4LayerScaleKernel(void *hidden_bf16, const void *scalar_bf16, uint32_t dimension)
{
	float scale = SparkLmBf16ToFloat(scalar_bf16,0u);
	uint32_t index = threadIdx.x;
	while ( index < dimension )
	{
		SparkLmFloatToBf16(hidden_bf16,index,SparkLmBf16ToFloat(hidden_bf16,index) * scale);
		index += blockDim.x;
	}
}

extern "C" cudaError_t SparkGemma4LaunchLayerScale(cudaStream_t stream, void *hidden_bf16, const void *scalar_bf16, uint32_t row_count, uint32_t dimension)
{
	if ( hidden_bf16 == 0 || scalar_bf16 == 0 || row_count == 0u || dimension == 0u )
		return(cudaErrorInvalidValue);
	SparkGemma4LayerScaleKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,scalar_bf16,dimension);
	return(cudaGetLastError());
}

extern "C" uint32_t SparkGemma4HeadDirectArgmaxScratchElements(uint32_t rows)
{
	return(rows * SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT);
}

extern "C" cudaError_t SparkGemma4LaunchHeadDirectArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, void *scratch, uint32_t *candidate_counts, uint32_t *output_token_ids, float *output_scores, uint32_t candidate_offset, uint32_t row_count, uint32_t candidate_count)
{
	return(SparkLmHostLaunchHeadDirectArgmaxWithScore(stream,hidden_bf16,head_weight_bf16,
		scratch,candidate_counts,output_token_ids,output_scores,candidate_offset,
		row_count,candidate_count,SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION));
}

#if SPARK_GEMMA4_MODEL_MOE_BLOCK
extern "C" cudaError_t SparkGemma4LaunchRouterSoftmax(cudaStream_t stream, float *scores_f32, uint32_t row_count)
{
	LmHeadSoftmaxKernel<SPARK_GEMMA4_CUDA_THREADS><<<row_count,SPARK_GEMMA4_CUDA_THREADS,0,stream>>>(scores_f32,row_count,SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT,1.0f);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchRouterTopk(cudaStream_t stream, const float *scores_f32, uint32_t *indices_u32, float *weights_f32, uint32_t row_count)
{
	LmTopkSmallKernel<SPARK_GEMMA4_CUDA_THREADS,SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN,true,1u,1u,LM_TOPK_SCORE_IDENTITY><<<row_count,SPARK_GEMMA4_CUDA_THREADS,2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),stream>>>(scores_f32,SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT,indices_u32,weights_f32,0,0,1.0f);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchMoeRoute(cudaStream_t stream, const uint32_t *route_expert, uint32_t rows, uint32_t expert_width, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2)
{
	int32_t launch_status = LmRouteBuild<SPARK_LM_CTA_THREADS,SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT>(route_expert,rows,rows * SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN,SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN,group_row_offset,route_packed_row,route_source_token,expert_width,SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION,SPARK_LM_TILE_N,SPARK_LM_TILE_N,group_tile_prefix_w1,group_tile_prefix_w2,stream);
	return(launch_status == LM_LAUNCH_OK ? cudaSuccess : cudaErrorLaunchFailure);
}

#include "sparkpipe/family/cuda/spark_cuda_gate_scores.cuh"

extern "C" cudaError_t SparkGemma4LaunchGateScores(cudaStream_t stream, const SparkGemma4LinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count)
{
	uint32_t warp_count = SPARK_LM_CTA_THREADS / SPARK_LM_WARP_LANES;
	uint32_t expert_blocks = (gate->output_dimension + warp_count - 1u) / warp_count;
	if ( gate == 0 || input_bf16 == 0 || scores_f32 == 0 || gate->weight_format != SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 || gate->input_dimension == 0u )
		return(cudaErrorInvalidValue);
	SparkGemma4GateScoresKernel<<<dim3(row_count,expert_blocks),SPARK_LM_CTA_THREADS,gate->input_dimension * sizeof(float),stream>>>(gate->weight_payload,input_bf16,scores_f32,row_count,gate->input_dimension,gate->output_dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchGroupedExpertLinear(cudaStream_t stream, const SparkGemma4LinearView *view, const void *input_bf16, const uint32_t *source_row_map, const uint32_t *group_row_offset, const uint32_t *group_tile_prefix, void *output_bf16, uint32_t source_row_count, uint32_t multiprocessor_count, uint32_t tp_degree, uint32_t tp_rank, uint32_t route_group_base, const void *frame_error)
{
	uint64_t rows_per_expert;
	uint64_t payload_stride;
	uint32_t experts_per_rank = SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT / tp_degree;
	const uint8_t *payload;
	const uint32_t *offsets;
	const uint32_t *prefix;
	(void)multiprocessor_count;
	(void)frame_error;
	if ( view == 0 || input_bf16 == 0 || group_row_offset == 0 || group_tile_prefix == 0 || output_bf16 == 0 || view->weight_format != SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 || view->weight_payload == 0 || (source_row_map == 0 && source_row_count == 0u) || tp_degree == 0u || tp_rank >= tp_degree || (SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT % tp_degree) != 0u )
		return(cudaErrorInvalidValue);
	rows_per_expert = (uint64_t)view->output_dimension / experts_per_rank;
	if ( rows_per_expert * experts_per_rank != view->output_dimension )
		return(cudaErrorInvalidValue);
	payload_stride = rows_per_expert * view->input_dimension * 2u;
	payload = (const uint8_t *)view->weight_payload + ((uint64_t)tp_rank * experts_per_rank * payload_stride);
	offsets = group_row_offset + route_group_base;
	prefix = group_tile_prefix + route_group_base;
	return(SparkLmHostLaunchGroupedScalarLinear<32u>(stream,
		SPARK_LM_WEIGHT_FORMAT_BF16,
		payload,(const uint8_t *)0,
		payload_stride,0u,
		input_bf16,source_row_map,source_row_count,offsets,prefix,
		output_bf16,experts_per_rank,
		view->input_dimension,(uint32_t)rows_per_expert,multiprocessor_count));
}

extern "C" cudaError_t SparkGemma4LaunchMoePairReduceOverwrite(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *output_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchMoePairReduceOverwrite(stream,slot_out_bf16,inverse_map,pair_weights_f32,output_bf16,row_count,SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN,hidden_dimension));
}
#endif

extern "C" cudaError_t SparkGemma4ConfigureCudaKernels(void)
{
	uint32_t widest = SPARK_GEMMA4_MODEL_FULL_QUERY_DIMENSION;
	if ( SPARK_GEMMA4_MODEL_DENSE_INTERMEDIATE_DIMENSION > widest )
		widest = SPARK_GEMMA4_MODEL_DENSE_INTERMEDIATE_DIMENSION;
	return(cudaFuncSetAttribute(
		(const void *)SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,SPARK_LM_CTA_WARPS>,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		(int)(widest * sizeof(float))));
}

#include "sparkpipe/family/cuda/spark_cuda_head_maxloc_keys.cuh"
