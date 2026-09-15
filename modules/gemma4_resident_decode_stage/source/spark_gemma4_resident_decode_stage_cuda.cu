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

extern "C" cudaError_t SparkGemma4LaunchFusedResidualRmsNorm(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	size_t shared_memory_bytes = (size_t)dimension * sizeof(float);
	SparkLmFusedResidualRmsNormKernel<<<row_count,SPARK_LM_CTA_THREADS,shared_memory_bytes,stream>>>(hidden_bf16,delta_bf16,gain_bf16,output_bf16,row_count,dimension,epsilon);
	return(cudaGetLastError());
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

static __device__ __forceinline__ uint32_t SparkGemma4HeadOrderKey(float score)
{
	uint32_t bits = __float_as_uint(score);
	return((bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u);
}

static __global__ void SparkGemma4HeadMaxLocPackKernel(const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count)
{
	uint32_t row = blockIdx.x;
	uint32_t token;
	if ( row >= row_count )
		return;
	token = token_ids_u32[row];
	keys_u64[row] = ((uint64_t)SparkGemma4HeadOrderKey(scores_f32[row]) << 32u) | (uint64_t)(SPARK_GEMMA4_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID - token);
}

static __global__ void SparkGemma4HeadMaxLocUnpackKernel(const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count)
{
	uint32_t row = blockIdx.x;
	if ( row >= row_count )
		return;
	token_ids_u32[row] = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID - (uint32_t)keys_u64[row];
}

extern "C" cudaError_t SparkGemma4LaunchHeadMaxLocPack(cudaStream_t stream, const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count)
{
	SparkGemma4HeadMaxLocPackKernel<<<row_count,1u,0,stream>>>(scores_f32,token_ids_u32,keys_u64,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkGemma4LaunchHeadMaxLocUnpack(cudaStream_t stream, const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count)
{
	SparkGemma4HeadMaxLocUnpackKernel<<<row_count,1u,0,stream>>>(keys_u64,token_ids_u32,row_count);
	return(cudaGetLastError());
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

static __global__ void SparkGemma4GateScoresKernel(const void *weight_bf16, const void *input_bf16, float *scores_f32, uint32_t row_count, uint32_t input_dimension, uint32_t expert_count)
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

static __device__ __forceinline__ float2 SparkGemma4CombineLoadBf16Pair(const void *base,uint64_t element)
{
	uint32_t packed = ((const uint32_t *)base)[element];
	float2 pair;
	pair.x = __int_as_float((int32_t)((packed & UINT32_C(0x0000ffff)) << 16u));
	pair.y = __int_as_float((int32_t)(packed & UINT32_C(0xffff0000)));
	return(pair);
}

static __device__ __forceinline__ void SparkGemma4CombineStoreBf16Pair(void *base,uint64_t element,float x,float y)
{
	uint32_t packed = (__float_as_uint(y) & UINT32_C(0xffff0000)) |
		(__float_as_uint(x) >> 16u);
	((uint32_t *)base)[element] = packed;
}

struct SparkGemma4CombineRankSources
{
	const void *pointer[16u];
};

static __global__ void SparkGemma4CombineSumRanksF32Kernel(
    void *destination_bf16,
    SparkGemma4CombineRankSources sources,
    uint32_t source_count,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 acc,v;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		uint32_t source;
		acc.x = 0.0f;
		acc.y = 0.0f;
		for ( source = 0u; source < source_count; source++ )
		{
			v = SparkGemma4CombineLoadBf16Pair(sources.pointer[source],pair);
			acc.x += v.x;
			acc.y += v.y;
		}
		SparkGemma4CombineStoreBf16Pair(destination_bf16,pair,acc.x,acc.y);
	}
}

static __global__ void SparkGemma4CombineAccumAddKernel(
	void *destination_bf16,
	const void *source_bf16,
	uint32_t row_count,
	uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float2 destination_pair,source_pair;
	if ( row >= row_count )
		return;
	for (element=threadIdx.x; element<(width >> 1u); element+=blockDim.x)
	{
		destination_pair = SparkGemma4CombineLoadBf16Pair(destination_bf16,offset + element);
		source_pair = SparkGemma4CombineLoadBf16Pair(source_bf16,offset + element);
		SparkGemma4CombineStoreBf16Pair(destination_bf16,offset + element,destination_pair.x + source_pair.x,destination_pair.y + source_pair.y);
	}
}

static __global__ void SparkGemma4CombineSeedF32Kernel(
    float *destination_f32,
    const void *source_a_bf16,
    const void *source_b_bf16,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 a,b;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		a = SparkGemma4CombineLoadBf16Pair(source_a_bf16,pair);
		b = SparkGemma4CombineLoadBf16Pair(source_b_bf16,pair);
		destination_f32[2u * pair] = a.x + b.x;
		destination_f32[2u * pair + 1u] = a.y + b.y;
	}
}

static __global__ void SparkGemma4CombineAddF32Kernel(
    float *destination_f32,
    const void *source_bf16,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 b;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		b = SparkGemma4CombineLoadBf16Pair(source_bf16,pair);
		destination_f32[2u * pair] += b.x;
		destination_f32[2u * pair + 1u] += b.y;
	}
}

static __global__ void SparkGemma4CombineRoundF32Kernel(
    void *destination_bf16,
    const float *source_f32,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 v;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		v.x = source_f32[2u * pair];
		v.y = source_f32[2u * pair + 1u];
		SparkGemma4CombineStoreBf16Pair(destination_bf16,pair,v.x,v.y);
	}
}

static __global__ void SparkGemma4CombineAccumU64MaxKernel(
	uint64_t *destination,
	const uint64_t *source,
	uint32_t element_count)
{
	uint32_t element;
	element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < element_count && source[element] > destination[element] )
		destination[element] = source[element];
}

extern "C" cudaError_t SparkGlm5NextLaunchSumRanksF32(cudaStream_t stream,
    void *destination,const void *const *sources,uint32_t source_count,
    uint32_t element_count)
{
	SparkGemma4CombineRankSources by_value;
	uint32_t index;
	if ( destination == 0 || sources == 0 || source_count == 0u ||
	     source_count > 16u || element_count == 0u )
		return(cudaErrorInvalidValue);
	for ( index = 0u; index < source_count; index++ )
		by_value.pointer[index] = sources[index];
	{
		dim3 grid;
		uint32_t pairs = (element_count + 1u) / 2u;
		uint32_t rows = (pairs + 255u) / 256u;
		grid = dim3(rows < 1u ? 1u : rows,1u,1u);
		SparkGemma4CombineSumRanksF32Kernel<<<grid,256u,0u,stream>>>(
		    destination,by_value,source_count,pairs);
	}
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkGlm5NextLaunchSeedF32(cudaStream_t stream,
    float *destination,const void *a,const void *b,uint32_t element_count)
{
	SparkGemma4CombineSeedF32Kernel<<<1,256u,0u,stream>>>(
	    destination,a,b,(element_count + 1u) / 2u);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkGlm5NextLaunchAddF32(cudaStream_t stream,
    float *destination,const void *b,uint32_t element_count)
{
	SparkGemma4CombineAddF32Kernel<<<1,256u,0u,stream>>>(
	    destination,b,(element_count + 1u) / 2u);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkGlm5NextLaunchRoundF32(cudaStream_t stream,
    void *destination,const float *source,uint32_t element_count)
{
	SparkGemma4CombineRoundF32Kernel<<<1,256u,0u,stream>>>(
	    destination,source,(element_count + 1u) / 2u);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkGlm5NextLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkGemma4CombineAccumAddKernel<<<row_count,256u,0u,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm5NextLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	if ( destination == 0 || source == 0 || element_count == 0u )
		return(cudaErrorInvalidValue);
	SparkGemma4CombineAccumU64MaxKernel<<<(element_count + 255u) / 256u,256u,0u,stream>>>(destination,source,element_count);
	return(cudaPeekAtLastError());
}
