#pragma once

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/stage_head_shape.h"
#include "runtime/launch.h"
#include "sparkpipe/spark_head_screen.h"
#include "sparkpipe/spark_lm_certified_launch.h"
#include <stdint.h>


template<uint32_t THREADS>
__global__ static void LmStageEmbeddingKernel(const uint16_t *__restrict__ embed_weight, const uint32_t *__restrict__ token_ids,
	uint16_t *__restrict__ hidden_bf16, uint32_t hidden, uint32_t vocab_slice_offset, uint32_t vocab_slice_rows)
{
	const uint32_t row = blockIdx.y;
	const uint32_t local = token_ids[row] - vocab_slice_offset;
	const uint16_t *source = local < vocab_slice_rows ? embed_weight + (uint64_t)local * hidden : 0;
	for ( uint32_t k = blockIdx.x * THREADS + threadIdx.x; k < hidden; k += gridDim.x * THREADS )
		hidden_bf16[(uint64_t)row * hidden + k] = source != 0 ? source[k] : 0u;
}

static int32_t LmStageEmbedding(const uint16_t *embed_weight, const uint32_t *token_ids, uint16_t *hidden_bf16, uint32_t rows,
	uint32_t hidden, uint32_t vocab_slice_offset, uint32_t vocab_slice_rows, cudaStream_t stream)
{
	LM_LAUNCH((LmStageEmbeddingKernel<LM_STAGE_HEAD_THREADS>), dim3((hidden + LM_STAGE_HEAD_THREADS - 1u) / LM_STAGE_HEAD_THREADS,rows),
		LM_STAGE_HEAD_THREADS, 0, stream, embed_weight,token_ids,hidden_bf16,hidden,vocab_slice_offset,vocab_slice_rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

typedef struct LmStageHeadRows
{
	const uint16_t *hidden_bf16;
	uint16_t *normed_bf16;
	const void *norm_weight;
	const void *head_weight;
	uint32_t hidden;
	float epsilon;
	uint32_t norm_f32;
	uint32_t vocab_slice_rows;
	uint32_t rank_offset;
	float *candidate_score;
	uint32_t *candidate_token;
	uint32_t *output_token;
	float *output_score;
} LmStageHeadRows;

__global__ static void LmStageHeadRankTokenKernel(uint32_t *tokens, uint32_t rows, uint32_t rank_offset)
{
	const uint32_t row = blockIdx.y;
	if ( row < rows && threadIdx.x == 0u && tokens[row] != 0xFFFFFFFFu )
		tokens[row] += rank_offset;
}

static int32_t LmStageNormRows(const LmStageHeadRows &head, uint32_t rows, cudaStream_t stream)
{
	if ( head.norm_f32 != 0u )
		LM_LAUNCH((LmFusedResidualRmsNormKernel<LM_STAGE_HEAD_THREADS,float>), rows, LM_STAGE_HEAD_THREADS, (head.hidden + 8u) * sizeof(float), stream,
			head.hidden_bf16,0,(const float *)head.norm_weight,0,head.normed_bf16,head.hidden,head.hidden,head.epsilon);
	else
		LM_LAUNCH((LmFusedResidualRmsNormKernel<LM_STAGE_HEAD_THREADS,uint16_t>), rows, LM_STAGE_HEAD_THREADS, (head.hidden + 8u) * sizeof(float), stream,
			head.hidden_bf16,0,(const uint16_t *)head.norm_weight,0,head.normed_bf16,head.hidden,head.hidden,head.epsilon);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t LmStageHeadSlice(const LmStageHeadRows &head, uint32_t rows, cudaStream_t stream)
{
	const uint32_t tiles = (head.vocab_slice_rows + LM_STAGE_HEAD_TILE - 1u) / LM_STAGE_HEAD_TILE;
	if ( LmStageNormRows(head,rows,stream) != LM_LAUNCH_OK )
		return(LM_LAUNCH_ERR_LAUNCH);
	LM_LAUNCH((LmHeadCandidateKernel<LM_STAGE_HEAD_THREADS,LM_STAGE_HEAD_TILE>), dim3(tiles,rows), LM_STAGE_HEAD_THREADS, 0, stream,
		head.normed_bf16,(const uint16_t *)head.head_weight,0,head.candidate_score,head.candidate_token,rows,head.hidden,head.vocab_slice_rows);
	LM_LAUNCH((LmHeadCommitKernel<LM_STAGE_HEAD_THREADS>), rows, LM_STAGE_HEAD_THREADS, 0, stream,
		head.candidate_score,head.candidate_token,tiles,head.output_token,head.output_score,rows);
	if ( cudaPeekAtLastError() != cudaSuccess )
		return(LM_LAUNCH_ERR_LAUNCH);
	if ( head.rank_offset == 0u || rows == 0u )
		return(LM_LAUNCH_OK);
	LM_LAUNCH((LmStageHeadRankTokenKernel), dim3(1u,rows), LM_STAGE_HEAD_THREADS, 0, stream, head.output_token,rows,head.rank_offset);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

typedef struct LmStageHeadCertified
{
	const uint8_t *payload;
	const float *scale;
	const float *norm;
	void *scratch;
	uint32_t *candidates;
	uint32_t *screened;
} LmStageHeadCertified;

static int32_t LmStageHeadCertifiedSlice(const LmStageHeadRows &head, const LmStageHeadCertified &certified, uint32_t rows, cudaStream_t stream)
{
	if ( head.hidden_bf16 == 0 || head.normed_bf16 == 0 || head.norm_weight == 0 || head.head_weight == 0 ||
		head.output_token == 0 || head.output_score == 0 || certified.payload == 0 || certified.scale == 0 ||
		certified.norm == 0 || certified.scratch == 0 || certified.candidates == 0 || certified.screened == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( LmStageNormRows(head,rows,stream) != LM_LAUNCH_OK )
		return(LM_LAUNCH_ERR_LAUNCH);
	return(SparkLmHostLaunchHeadCertifiedFp8RowsWithScore(stream,head.normed_bf16,head.head_weight,certified.payload,
		certified.scale,certified.norm,certified.scratch,certified.candidates,certified.screened,head.output_token,
		head.output_score,head.rank_offset,rows,head.vocab_slice_rows,head.hidden) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static __device__ __forceinline__ uint32_t LmStageOrderedScore(float score)
{
	uint32_t bits;
	if ( score != score )
		return 0u;
	bits = __float_as_uint(score == 0.0f ? 0.0f : score);
	return bits ^ ((bits & 0x80000000u) != 0u ? 0xFFFFFFFFu : 0x80000000u);
}

static __device__ __forceinline__ float LmStageOrderedScoreInverse(uint32_t ordered)
{
	return __uint_as_float(ordered ^ ((ordered & 0x80000000u) != 0u ? 0x80000000u : 0xFFFFFFFFu));
}

__global__ static void LmStageMaxlocPackKernel(const float *scores, const uint32_t *tokens, uint64_t *maxloc, uint32_t rows)
{
	const uint32_t row = blockIdx.y;
	if ( row < rows && threadIdx.x == 0u )
		maxloc[row] = ((uint64_t)LmStageOrderedScore(scores[row]) << 32u) | (uint64_t)(0xFFFFFFFFu - tokens[row]);
}

__global__ static void LmStageMaxlocUnpackKernel(const uint64_t *maxloc, uint32_t *tokens, float *scores, uint32_t rows)
{
	const uint32_t row = blockIdx.y;
	if ( row < rows && threadIdx.x == 0u )
	{
		tokens[row] = 0xFFFFFFFFu - (uint32_t)maxloc[row];
		scores[row] = LmStageOrderedScoreInverse((uint32_t)(maxloc[row] >> 32u));
	}
}

static int32_t LmStageMaxlocPack(const float *scores, const uint32_t *tokens, uint64_t *maxloc, uint32_t rows, cudaStream_t stream)
{
	if ( rows == 0u )
		return LM_LAUNCH_OK;
	LM_LAUNCH((LmStageMaxlocPackKernel), dim3(1u,rows), LM_STAGE_HEAD_THREADS, 0, stream, scores,tokens,maxloc,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t LmStageMaxlocUnpack(const uint64_t *maxloc, uint32_t *tokens, float *scores, uint32_t rows, cudaStream_t stream)
{
	if ( rows == 0u )
		return LM_LAUNCH_OK;
	LM_LAUNCH((LmStageMaxlocUnpackKernel), dim3(1u,rows), LM_STAGE_HEAD_THREADS, 0, stream, maxloc,tokens,scores,rows);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

__global__ static void LmStageLastRowsGatherKernel(const uint32_t *row_begin, const uint32_t *row_indices,
	const uint16_t *hidden_bf16, uint16_t *out_bf16, uint32_t hidden)
{
	const uint4 *source = (const uint4 *)(hidden_bf16 + (uint64_t)row_indices[row_begin[blockIdx.x + 1u] - 1u] * hidden);
	uint4 *target = (uint4 *)(out_bf16 + (uint64_t)blockIdx.x * hidden);
	for ( uint32_t k = threadIdx.x; k < hidden / 8u; k += blockDim.x )
		target[k] = source[k];
}

__global__ static void LmStageLastRowsScatterKernel(const uint32_t *row_begin, const uint32_t *row_indices,
	const uint32_t *token, const float *score, uint32_t *out_token, float *out_score, uint32_t sequences)
{
	const uint32_t sequence = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t row;
	if ( sequence >= sequences )
		return;
	row = row_indices[row_begin[sequence + 1u] - 1u];
	out_token[row] = token[sequence];
	out_score[row] = score[sequence];
}

__global__ static void LmStageRowsGatherKernel(const uint32_t *rows, const uint16_t *hidden_bf16, uint16_t *out_bf16, uint32_t hidden)
{
	const uint64_t source = (uint64_t)rows[blockIdx.x] * hidden, target = (uint64_t)blockIdx.x * hidden;
	for ( uint32_t k = threadIdx.x; k < hidden; k += blockDim.x )
		out_bf16[target + k] = hidden_bf16[source + k];
}
