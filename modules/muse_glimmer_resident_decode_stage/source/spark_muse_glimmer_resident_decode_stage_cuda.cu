#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_muse_glimmer_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_lm_kernels.cuh"
#include "runtime/gemm.cuh"
#include "runtime/launch.h"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/formats/fp8.cuh"


#define SPARK_MUSE_GLIMMER_CUDA_THREADS 256u
#define SPARK_MUSE_GLIMMER_CUDA_HEAD_NORM_THREADS 512u
#define SPARK_MUSE_GLIMMER_CUDA_TILE_N 128u
#define SPARK_MUSE_GLIMMER_CUDA_STAGES 2u
#define SPARK_MUSE_GLIMMER_CUDA_WARPS 8u

using MuseGlimmerKv = LmKvHeads<16u, 1u, SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION, 64u>;
using MuseGlimmerKv2 = LmKvHeads<16u, 2u, SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION, 64u>;

static_assert(MuseGlimmerKv::kSlotBytes == SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_HEAD_COUNT(16u) * (SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION + SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION) * SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES,
	"the muse kv slot is one rank-local [K|V] head pair in bf16");
static_assert(MuseGlimmerKv::kSlotBytes == 512u,
	"the staged kv slot is 512 bytes");
static_assert(SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_QUERY_HEAD_COUNT(16u) == 2u,
	"each tp16 rank owns exactly two query heads");
static_assert(SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_HEAD_COUNT(16u) == 1u,
	"each tp16 rank owns exactly one replicated kv head");
static_assert(SPARK_MUSE_GLIMMER_MODEL_QGKV_LOCAL_ROWS(16u) == 768u,
	"the fused qgkv projection carries q256|gate256|k128|v128 rank-local rows");
static_assert(SPARK_MUSE_GLIMMER_MODEL_MLP_LOCAL_INTERMEDIATE(16u) == 1248u,
	"the mlp intermediate shards to 1248 columns per rank");
static_assert(SPARK_MUSE_GLIMMER_MODEL_LAYER_IS_FULL_ATTENTION(3u) && !SPARK_MUSE_GLIMMER_MODEL_LAYER_IS_FULL_ATTENTION(2u),
	"period 4 with full attention in phase 3");


static __global__ void SparkMuseGlimmerEmbeddingGatherKernel(const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t vocab_per_rank, uint32_t rank_offset)
{
	uint64_t index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t row = (uint32_t)(index / SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION),element = (uint32_t)(index % SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION);
	uint32_t token;
	uint64_t source;
	if ( row >= row_count )
		return;
	token = token_ids[row];
	source = ((uint64_t)(token - rank_offset) * SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION) + element;
	SparkLmFloatToBf16(hidden_bf16,index,token >= rank_offset && token < rank_offset + vocab_per_rank
		? SparkLmBf16ToFloat(embedding_bf16,source)
		: 0.0f);
}

static __global__ void SparkMuseGlimmerResidualAddKernel(void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
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

static __global__ void SparkMuseGlimmerTpCombineAddKernel(void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width)
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

extern "C" cudaError_t SparkMuseGlimmerLaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t tp_degree, uint32_t tp_rank)
{
	uint32_t vocab_per_rank = SPARK_MUSE_GLIMMER_MODEL_OUTPUT_VOCAB_COUNT / tp_degree;
	SparkMuseGlimmerEmbeddingGatherKernel<<<(uint32_t)(((uint64_t)row_count * SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(token_ids,embedding_bf16,hidden_bf16,row_count,vocab_per_rank,tp_rank * vocab_per_rank);
	return(cudaGetLastError());
}

static __device__ __forceinline__ uint32_t SparkMuseGlimmerOrderedHeadScore(float score)
{
	uint32_t bits;
	if ( isnan(score) )
		return(0u);
	bits = __float_as_uint(score);
	return(bits ^ ((bits & UINT32_C(0x80000000)) != 0u ? UINT32_MAX : UINT32_C(0x80000000)));
}

static __global__ void SparkMuseGlimmerHeadArgmaxPackKernel(const float *scores_f32, uint32_t *local_token_ids, uint64_t *maxloc, uint32_t row_count, uint32_t candidate_count, uint32_t rank_offset)
{
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES,candidate,winner;
	uint64_t row_offset;
	float accumulator,warp_best_score;
	uint32_t warp_best_candidate;
	if ( row >= row_count )
		return;
	row_offset = (uint64_t)row * candidate_count;
	warp_best_score = -3.0e38f;
	warp_best_candidate = 0u;
	for (candidate = warp; candidate < candidate_count; candidate += SPARK_LM_CTA_WARPS)
	{
		accumulator = scores_f32[row_offset + candidate];
		if ( accumulator > warp_best_score || (accumulator == warp_best_score && candidate < warp_best_candidate) )
		{
			warp_best_score = accumulator;
			warp_best_candidate = candidate;
		}
	}
	if ( lane == 0u )
	{
		best_score[warp] = warp_best_score;
		best_candidate[warp] = warp_best_candidate;
	}
	__syncthreads();
	if ( threadIdx.x != 0u )
		return;
	winner = 0u;
	for (candidate = 1u; candidate < SPARK_LM_CTA_WARPS; candidate++)
		if ( best_score[candidate] > best_score[winner] || (best_score[candidate] == best_score[winner] && best_candidate[candidate] < best_candidate[winner]) )
			winner = candidate;
	local_token_ids[row] = best_candidate[winner] + rank_offset;
	maxloc[row] = ((uint64_t)SparkMuseGlimmerOrderedHeadScore(best_score[winner]) << 32u) |
		(UINT32_MAX - (best_candidate[winner] + rank_offset));
}

static __global__ void SparkMuseGlimmerHeadMaxlocUnpackKernel(const uint64_t *maxloc, uint32_t *token_ids, uint32_t row_count)
{
	uint32_t row = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( row < row_count )
		token_ids[row] = UINT32_MAX - (uint32_t)maxloc[row];
}

extern "C" cudaError_t SparkMuseGlimmerLaunchHeadArgmaxPack(cudaStream_t stream, const float *scores_f32, uint32_t *local_token_ids, uint64_t *maxloc, uint32_t row_count, uint32_t candidate_count, uint32_t tp_degree, uint32_t tp_rank)
{
	SparkMuseGlimmerHeadArgmaxPackKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(scores_f32,local_token_ids,maxloc,row_count,candidate_count,(tp_rank * (SPARK_MUSE_GLIMMER_MODEL_OUTPUT_VOCAB_COUNT / tp_degree)));
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchHeadMaxlocUnpack(cudaStream_t stream, const uint64_t *maxloc, uint32_t *token_ids, uint32_t row_count)
{
	SparkMuseGlimmerHeadMaxlocUnpackKernel<<<(row_count + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS,SPARK_LM_CTA_THREADS,0,stream>>>(maxloc,token_ids,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchHeadRmsNorm(cudaStream_t stream, const void *input_bf16, const void *weight_bf16, void *output_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dimension, float epsilon, float head_multiply)
{
	dim3 grid(head_count,row_count);
	LmHeadRmsNormKernel<SPARK_MUSE_GLIMMER_CUDA_HEAD_NORM_THREADS><<<grid,SPARK_MUSE_GLIMMER_CUDA_HEAD_NORM_THREADS,0,stream>>>((const uint16_t *)input_bf16,(const uint16_t *)weight_bf16,(uint16_t *)output_bf16,row_count,head_count,head_dimension,epsilon,head_multiply);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchCenteredRmsNorm(cudaStream_t stream, const void *input_bf16, const void *weight_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	LmCenteredRmsNormKernel<SPARK_MUSE_GLIMMER_CUDA_THREADS><<<row_count,SPARK_MUSE_GLIMMER_CUDA_THREADS,(dimension + 8u) * sizeof(float),stream>>>((const uint16_t *)input_bf16,(const uint16_t *)weight_bf16,(uint16_t *)output_bf16,dimension,dimension,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *weight_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	LmBf16RmsNormKernel<SPARK_MUSE_GLIMMER_CUDA_THREADS><<<row_count,SPARK_MUSE_GLIMMER_CUDA_THREADS,(dimension + 8u) * sizeof(float),stream>>>((const uint16_t *)input_bf16,(const uint16_t *)weight_bf16,(uint16_t *)output_bf16,dimension,dimension,epsilon);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchCopyRows(cudaStream_t stream, const void *source_bf16, void *destination_bf16, uint32_t row_count, uint32_t dimension)
{
	LmCopyRowsKernel<SPARK_MUSE_GLIMMER_CUDA_THREADS><<<dim3((dimension + SPARK_MUSE_GLIMMER_CUDA_THREADS - 1u) / SPARK_MUSE_GLIMMER_CUDA_THREADS,row_count),SPARK_MUSE_GLIMMER_CUDA_THREADS,0,stream>>>((const uint16_t *)source_bf16,(uint16_t *)destination_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchAddRows(cudaStream_t stream, const void *a_bf16, const void *b_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension)
{
	LmAddRowsKernel<SPARK_MUSE_GLIMMER_CUDA_THREADS><<<dim3((dimension + SPARK_MUSE_GLIMMER_CUDA_THREADS - 1u) / SPARK_MUSE_GLIMMER_CUDA_THREADS,row_count),SPARK_MUSE_GLIMMER_CUDA_THREADS,0,stream>>>((const uint16_t *)a_bf16,(const uint16_t *)b_bf16,(uint16_t *)output_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchLinear(cudaStream_t stream, const void *weight_bf16, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, const uint32_t *dense_row_offset)
{
	LmGemmArguments gemm;
	int32_t status;
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_count = 1u;
	gemm.group_row_offset = dense_row_offset;
	gemm.input_dimension = input_dimension;
	gemm.output_dimension = output_dimension;
	gemm.output_bf16 = output_bf16;
	status = LmGemmLaunchTileK<LmBf16Format,SPARK_MUSE_GLIMMER_CUDA_TILE_N,SPARK_MUSE_GLIMMER_CUDA_STAGES,SPARK_MUSE_GLIMMER_CUDA_WARPS>(
		&gemm,input_bf16,weight_bf16,row_count,row_count,1u,1u,
		input_dimension,output_dimension,multiprocessors,false,stream);
	return(status == LM_LAUNCH_OK ? cudaSuccess : cudaErrorLaunchFailure);
}

extern "C" cudaError_t SparkMuseGlimmerLaunchLinearScores(cudaStream_t stream, const void *weight_bf16, const void *input_bf16, float *scores_f32, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, const uint32_t *dense_row_offset)
{
	LmGemmArguments gemm;
	int32_t status;
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_count = 1u;
	gemm.group_row_offset = dense_row_offset;
	gemm.input_dimension = input_dimension;
	gemm.output_dimension = output_dimension;
	gemm.output_f32 = scores_f32;
	status = LmGemmLaunchTileK<LmBf16Format,SPARK_MUSE_GLIMMER_CUDA_TILE_N,SPARK_MUSE_GLIMMER_CUDA_STAGES,SPARK_MUSE_GLIMMER_CUDA_WARPS>(
		&gemm,input_bf16,weight_bf16,row_count,row_count,1u,1u,
		input_dimension,output_dimension,multiprocessors,false,stream);
	return(status == LM_LAUNCH_OK ? cudaSuccess : cudaErrorLaunchFailure);
}

extern "C" cudaError_t SparkMuseGlimmerLaunchSplitQkv(cudaStream_t stream, const void *fused_bf16, void *query_gate_bf16, void *key_bf16, void *value_bf16, uint32_t row_count, uint32_t tp_degree)
{
	LmQkvLayout layout;
	layout.query_dimension = SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_QUERY_GATE_DIMENSION(tp_degree);
	layout.key_dimension = SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_DIMENSION(tp_degree);
	layout.value_dimension = SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_DIMENSION(tp_degree);
	layout.rope_dimension = SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION;
	layout.head_dimension = SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION;
	LmSplitQkvKernel<SPARK_MUSE_GLIMMER_CUDA_THREADS><<<row_count,SPARK_MUSE_GLIMMER_CUDA_THREADS,0,stream>>>((const uint16_t *)fused_bf16,layout,(uint16_t *)query_gate_bf16,(uint16_t *)key_bf16,(uint16_t *)value_bf16,row_count,1.0f);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchSplitQueryGate(cudaStream_t stream, const void *query_gate_bf16, void *query_bf16, void *gate_bf16, uint32_t row_count, uint32_t local_head_count)
{
	LmSplitQueryGateKernel<SPARK_MUSE_GLIMMER_CUDA_THREADS><<<row_count,SPARK_MUSE_GLIMMER_CUDA_THREADS,0,stream>>>((const uint16_t *)query_gate_bf16,(uint16_t *)query_bf16,(uint16_t *)gate_bf16,local_head_count,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchQkNorm(cudaStream_t stream, const void *input_bf16, void *output_bf16, uint32_t head_count, float head_multiply, uint32_t row_count)
{
	dim3 grid(head_count,row_count);
	LmHeadRmsNormKernel<SPARK_MUSE_GLIMMER_CUDA_HEAD_NORM_THREADS><<<grid,SPARK_MUSE_GLIMMER_CUDA_HEAD_NORM_THREADS,0,stream>>>((const uint16_t *)input_bf16,(const uint16_t *)0,(uint16_t *)output_bf16,row_count,head_count,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_ATTN_QK_NORM_EPSILON,head_multiply);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchRope(cudaStream_t stream, void *rows_bf16, const uint32_t *positions, uint32_t head_count, uint32_t row_count)
{
	LmRopePerHeadKernel<SPARK_MUSE_GLIMMER_CUDA_THREADS><<<dim3(row_count,head_count),SPARK_MUSE_GLIMMER_CUDA_THREADS,0,stream>>>((uint16_t *)rows_bf16,positions,head_count,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_ATTN_ROPE_THETA);
	return(cudaGetLastError());
}

extern "C" uint32_t SparkMuseGlimmerKvViewBytes(void)
{
	return((uint32_t)sizeof(LmKvView));
}

extern "C" cudaError_t SparkMuseGlimmerLaunchKvStore(cudaStream_t stream, const void *views_handle, uint32_t layer_index, const void *key_bf16, const void *value_bf16, const uint32_t *sequence_of_row, const uint32_t *positions, uint32_t row_count, uint32_t local_kv_head_count)
{
	if ( local_kv_head_count == 1u )
		LmGqaKvStoreKernel<MuseGlimmerKv,SPARK_MUSE_GLIMMER_CUDA_THREADS,1u,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION><<<row_count,SPARK_MUSE_GLIMMER_CUDA_THREADS,0,stream>>>(((const LmKvView *)views_handle)[layer_index],(const uint16_t *)key_bf16,(const uint16_t *)value_bf16,sequence_of_row,positions,row_count);
	else
		LmGqaKvStoreKernel<MuseGlimmerKv2,SPARK_MUSE_GLIMMER_CUDA_THREADS,2u,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION><<<row_count,SPARK_MUSE_GLIMMER_CUDA_THREADS,0,stream>>>(((const LmKvView *)views_handle)[layer_index],(const uint16_t *)key_bf16,(const uint16_t *)value_bf16,sequence_of_row,positions,row_count);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchWindowPositions(cudaStream_t stream, const uint32_t *sequence_of_row, const uint32_t *context_lengths, const uint32_t *positions, uint32_t row_count, uint32_t *window_positions)
{
	LmBuildSlidingWindowPositionsKernel<SPARK_MUSE_GLIMMER_CUDA_THREADS><<<row_count,SPARK_MUSE_GLIMMER_CUDA_THREADS,0,stream>>>(sequence_of_row,context_lengths,positions,row_count,SPARK_MUSE_GLIMMER_MODEL_SLIDING_WINDOW,window_positions);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchAttentionDecode(cudaStream_t stream, const void *views_handle, uint32_t layer_index, const void *query_bf16, const uint32_t *sequence_of_row, const uint32_t *context_lengths, const uint32_t *window_positions, const uint32_t *positions, void *head_out_bf16, uint32_t row_count, uint32_t local_head_count, uint32_t local_kv_head_count)
{
	uint32_t selected_count = window_positions != 0 ? SPARK_MUSE_GLIMMER_MODEL_SLIDING_WINDOW : 0u;
	if ( local_kv_head_count == 1u )
		LmGqaAttentionDecodeKernel<MuseGlimmerKv,SPARK_MUSE_GLIMMER_CUDA_THREADS,1u,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION><<<dim3(row_count,local_head_count),SPARK_MUSE_GLIMMER_CUDA_THREADS,0,stream>>>((const uint16_t *)query_bf16,((const LmKvView *)views_handle)[layer_index],sequence_of_row,context_lengths,window_positions,selected_count,local_head_count,SPARK_MUSE_GLIMMER_MODEL_ATTN_SCALE,(uint16_t *)head_out_bf16,positions);
	else
		LmGqaAttentionDecodeKernel<MuseGlimmerKv2,SPARK_MUSE_GLIMMER_CUDA_THREADS,2u,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION><<<dim3(row_count,local_head_count),SPARK_MUSE_GLIMMER_CUDA_THREADS,0,stream>>>((const uint16_t *)query_bf16,((const LmKvView *)views_handle)[layer_index],sequence_of_row,context_lengths,window_positions,selected_count,local_head_count,SPARK_MUSE_GLIMMER_MODEL_ATTN_SCALE,(uint16_t *)head_out_bf16,positions);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchOutputGate(cudaStream_t stream, void *head_out_bf16, const void *gate_bf16, uint32_t row_count, uint32_t local_query_dimension)
{
	LmOutputGateKernel<SPARK_MUSE_GLIMMER_CUDA_THREADS><<<row_count,SPARK_MUSE_GLIMMER_CUDA_THREADS,0,stream>>>((uint16_t *)head_out_bf16,(const uint16_t *)gate_bf16,local_query_dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchSiluMul(cudaStream_t stream, const void *gate_up_bf16, void *intermediate_bf16, uint32_t row_count, uint32_t local_intermediate)
{
	LmSiluMulKernel<SPARK_MUSE_GLIMMER_CUDA_THREADS><<<row_count,SPARK_MUSE_GLIMMER_CUDA_THREADS,0,stream>>>((const uint16_t *)gate_up_bf16,(uint16_t *)intermediate_bf16,local_intermediate,true);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchTpCombineAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkMuseGlimmerTpCombineAddKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t pairs = ((uint64_t)row_count * dimension + 1u) >> 1u;
	SparkMuseGlimmerResidualAddKernel<<<(uint32_t)((pairs + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,delta_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SparkMuseGlimmerLaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count)
{
	SparkLmHeadArgmaxKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,head_weight_bf16,token_ids,output_token_ids,row_count,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,candidate_count);
	return(cudaGetLastError());
}
