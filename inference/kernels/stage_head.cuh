#pragma once

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/stage_head_shape.h"
#include "runtime/launch.h"
#include "sparkpipe/spark_head_screen.h"
#include "sparkpipe/spark_lm_certified_launch.h"
#include <stdint.h>
#include "sparkpipe/spark_gpu.h"


static __device__ __forceinline__ uint32_t LmStageClearNegZeroPair(uint32_t pair)
{
	const uint32_t lo = (pair & 0xffffu) == 0x8000u ? 0u : (pair & 0xffffu);
	const uint32_t hi = (pair & 0xffff0000u) == 0x80000000u ? 0u : (pair & 0xffff0000u);
	return(lo | hi);
}

template<uint32_t THREADS>
__global__ static void LmStageEmbeddingKernel(const uint16_t *__restrict__ embed_weight, const uint32_t *__restrict__ token_ids,
	uint16_t *__restrict__ hidden_bf16, uint32_t hidden, uint32_t vocab_slice_offset, uint32_t vocab_slice_rows)
{
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.y;
	const uint32_t local = token_ids[row] - vocab_slice_offset;
	const uint16_t *source = local < vocab_slice_rows ? embed_weight + (uint64_t)local * hidden : 0;
	uint16_t *target = hidden_bf16 + (uint64_t)row * hidden;
	if ( ((hidden & 7u) == 0u) && LmIsAligned16(target) && (source == 0 || LmIsAligned16(source)) )
	{
		const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
		const uint32_t vecs = hidden >> 3u;
		uint4 *dst4 = (uint4 *)target;
		const uint4 *src4 = (const uint4 *)source;
		for ( uint32_t v = blockIdx.x * THREADS + threadIdx.x; v < vecs; v += gridDim.x * THREADS )
			dst4[v] = src4 != 0 ? __ldg(src4 + v) : zero;
		return;
	}
	for ( uint32_t k = blockIdx.x * THREADS + threadIdx.x; k < hidden; k += gridDim.x * THREADS )
		target[k] = source != 0 ? source[k] : 0u;
}

static int32_t LmStageEmbedding(const uint16_t *embed_weight, const uint32_t *token_ids, uint16_t *hidden_bf16, uint32_t rows,
	uint32_t hidden, uint32_t vocab_slice_offset, uint32_t vocab_slice_rows, SparkGpuStream stream)
{
	const uint32_t vec_ok = ((hidden & 7u) == 0u) && LmIsAligned16(embed_weight) && LmIsAligned16(hidden_bf16);
	const uint32_t items = vec_ok != 0u ? (hidden >> 3u) : hidden;
	LM_LAUNCH_DEPENDENT((LmStageEmbeddingKernel<LM_STAGE_HEAD_THREADS>), dim3((items + LM_STAGE_HEAD_THREADS - 1u) / LM_STAGE_HEAD_THREADS,rows),
		LM_STAGE_HEAD_THREADS, 0, stream, embed_weight,token_ids,hidden_bf16,hidden,vocab_slice_offset,vocab_slice_rows);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

template<uint32_t THREADS>
__global__ static void LmStageEmbeddingAliasKernel(const uint16_t *__restrict__ embed_weight, const uint32_t *__restrict__ token_ids,
	uint16_t *__restrict__ hidden_bf16, uint32_t hidden, uint32_t vocab_slice_offset, uint32_t vocab_slice_rows,
	uint32_t alias_first, uint32_t alias_count, uint32_t alias_row)
{
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.y;
	const uint32_t token = token_ids[row] - alias_first < alias_count ? alias_row : token_ids[row];
	const uint32_t local = token - vocab_slice_offset;
	const uint16_t *source = local < vocab_slice_rows ? embed_weight + (uint64_t)local * hidden : 0;
	uint16_t *target = hidden_bf16 + (uint64_t)row * hidden;
	if ( ((hidden & 7u) == 0u) && LmIsAligned16(target) && (source == 0 || LmIsAligned16(source)) )
	{
		const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
		const uint32_t vecs = hidden >> 3u;
		uint4 *dst4 = (uint4 *)target;
		const uint4 *src4 = (const uint4 *)source;
		for ( uint32_t v = blockIdx.x * THREADS + threadIdx.x; v < vecs; v += gridDim.x * THREADS )
			dst4[v] = src4 != 0 ? __ldg(src4 + v) : zero;
		return;
	}
	for ( uint32_t k = blockIdx.x * THREADS + threadIdx.x; k < hidden; k += gridDim.x * THREADS )
		target[k] = source != 0 ? source[k] : 0u;
}

static int32_t LmStageEmbeddingAlias(const uint16_t *embed_weight, const uint32_t *token_ids, uint16_t *hidden_bf16, uint32_t rows,
	uint32_t hidden, uint32_t vocab_slice_offset, uint32_t vocab_slice_rows, uint32_t alias_first, uint32_t alias_count, uint32_t alias_row,
	SparkGpuStream stream)
{
	const uint32_t vec_ok = ((hidden & 7u) == 0u) && LmIsAligned16(embed_weight) && LmIsAligned16(hidden_bf16);
	const uint32_t items = vec_ok != 0u ? (hidden >> 3u) : hidden;
	LM_LAUNCH_DEPENDENT((LmStageEmbeddingAliasKernel<LM_STAGE_HEAD_THREADS>), dim3((items + LM_STAGE_HEAD_THREADS - 1u) / LM_STAGE_HEAD_THREADS,rows),
		LM_STAGE_HEAD_THREADS, 0, stream, embed_weight,token_ids,hidden_bf16,hidden,vocab_slice_offset,vocab_slice_rows,alias_first,alias_count,alias_row);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

template<uint32_t THREADS>
__global__ static void LmStageEmbeddingWholeKernel(const uint16_t *__restrict__ table, const uint32_t *__restrict__ token_ids,
	uint16_t *__restrict__ hidden_bf16, uint32_t hidden, uint32_t vocab, uint32_t alias_first, uint32_t alias_count, uint32_t alias_row)
{
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.y;
	const uint32_t token = token_ids[row] - alias_first < alias_count ? alias_row : token_ids[row];
	const uint16_t *source = token < vocab ? table + (uint64_t)token * hidden : 0;
	uint16_t *target = hidden_bf16 + (uint64_t)row * hidden;
	if ( ((hidden & 7u) == 0u) && LmIsAligned16(target) && (source == 0 || LmIsAligned16(source)) )
	{
		const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
		const uint32_t vecs = hidden >> 3u;
		uint4 *dst4 = (uint4 *)target;
		const uint4 *src4 = (const uint4 *)source;
		for ( uint32_t v = blockIdx.x * THREADS + threadIdx.x; v < vecs; v += gridDim.x * THREADS )
		{
			if ( src4 == 0 )
				dst4[v] = zero;
			else
			{
				const uint4 raw = __ldg(src4 + v);
				dst4[v] = make_uint4(
					LmStageClearNegZeroPair(raw.x),
					LmStageClearNegZeroPair(raw.y),
					LmStageClearNegZeroPair(raw.z),
					LmStageClearNegZeroPair(raw.w));
			}
		}
		return;
	}
	for ( uint32_t k = blockIdx.x * THREADS + threadIdx.x; k < hidden; k += gridDim.x * THREADS )
	{
		const uint16_t value = source != 0 ? source[k] : 0u;
		target[k] = value == 0x8000u ? 0u : value;
	}
}

static int32_t LmStageEmbeddingWhole(const uint16_t *table, const uint32_t *token_ids, uint16_t *hidden_bf16, uint32_t rows, uint32_t hidden,
	uint32_t vocab, uint32_t alias_first, uint32_t alias_count, uint32_t alias_row, SparkGpuStream stream)
{
	const uint32_t vec_ok = ((hidden & 7u) == 0u) && LmIsAligned16(table) && LmIsAligned16(hidden_bf16);
	const uint32_t items = vec_ok != 0u ? (hidden >> 3u) : hidden;
	LM_LAUNCH_DEPENDENT((LmStageEmbeddingWholeKernel<LM_STAGE_HEAD_THREADS>), dim3((items + LM_STAGE_HEAD_THREADS - 1u) / LM_STAGE_HEAD_THREADS,rows),
		LM_STAGE_HEAD_THREADS, 0, stream, table,token_ids,hidden_bf16,hidden,vocab,alias_first,alias_count,alias_row);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
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
	uint32_t norm_round_first;
	uint32_t prenormed;
	uint32_t vocab_slice_rows;
	uint32_t rank_offset;
	float *candidate_score;
	uint32_t *candidate_token;
	uint32_t *output_token;
	float *output_score;
} LmStageHeadRows;

__global__ static void LmStageHeadRankTokenKernel(uint32_t *tokens, uint32_t rows, uint32_t rank_offset)
{
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.y;
	if ( row < rows && threadIdx.x == 0u && tokens[row] != 0xFFFFFFFFu )
		tokens[row] += rank_offset;
}

static int32_t LmStageNormRows(const LmStageHeadRows &head, uint32_t rows, SparkGpuStream stream)
{
	if ( head.prenormed != 0u )
		return(head.normed_bf16 == head.hidden_bf16 ? LM_LAUNCH_OK : LM_LAUNCH_ERR_SHAPE);
	if ( head.norm_f32 != 0u && head.norm_round_first != 0u )
		LM_LAUNCH_DEPENDENT((LmFusedResidualRmsNormKernel<LM_STAGE_HEAD_THREADS,float,true>), rows, LM_STAGE_HEAD_THREADS, (head.hidden + 8u) * sizeof(float), stream,
			head.hidden_bf16,(const uint16_t *)0,(const float *)head.norm_weight,(uint16_t *)0,head.normed_bf16,head.hidden,head.hidden,head.epsilon);
	else if ( head.norm_f32 != 0u )
		LM_LAUNCH_DEPENDENT((LmFusedResidualRmsNormKernel<LM_STAGE_HEAD_THREADS,float>), rows, LM_STAGE_HEAD_THREADS, (head.hidden + 8u) * sizeof(float), stream,
			head.hidden_bf16,(const uint16_t *)0,(const float *)head.norm_weight,(uint16_t *)0,head.normed_bf16,head.hidden,head.hidden,head.epsilon);
	else if ( head.norm_round_first != 0u )
		LM_LAUNCH_DEPENDENT((LmFusedResidualRmsNormKernel<LM_STAGE_HEAD_THREADS,uint16_t,true>), rows, LM_STAGE_HEAD_THREADS, (head.hidden + 8u) * sizeof(float), stream,
			head.hidden_bf16,(const uint16_t *)0,(const uint16_t *)head.norm_weight,(uint16_t *)0,head.normed_bf16,head.hidden,head.hidden,head.epsilon);
	else
		LM_LAUNCH_DEPENDENT((LmFusedResidualRmsNormKernel<LM_STAGE_HEAD_THREADS,uint16_t>), rows, LM_STAGE_HEAD_THREADS, (head.hidden + 8u) * sizeof(float), stream,
			head.hidden_bf16,(const uint16_t *)0,(const uint16_t *)head.norm_weight,(uint16_t *)0,head.normed_bf16,head.hidden,head.hidden,head.epsilon);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t LmStageHeadSlice(const LmStageHeadRows &head, uint32_t rows, SparkGpuStream stream)
{
	const uint32_t tiles = (head.vocab_slice_rows + LM_STAGE_HEAD_TILE - 1u) / LM_STAGE_HEAD_TILE;
	if ( LmStageNormRows(head,rows,stream) != LM_LAUNCH_OK )
		return(LM_LAUNCH_ERR_LAUNCH);
	LM_LAUNCH_DEPENDENT((LmHeadCandidateKernel<LM_STAGE_HEAD_THREADS,LM_STAGE_HEAD_TILE>), dim3(tiles,rows), LM_STAGE_HEAD_THREADS, 0, stream,
		head.normed_bf16,(const uint16_t *)head.head_weight,(const uint32_t *)0,head.candidate_score,head.candidate_token,rows,head.hidden,head.vocab_slice_rows);
	LM_LAUNCH_DEPENDENT((LmHeadCommitKernel<LM_STAGE_HEAD_THREADS>), rows, LM_STAGE_HEAD_THREADS, 0, stream,
		head.candidate_score,head.candidate_token,tiles,head.output_token,head.output_score,rows);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		return(LM_LAUNCH_ERR_LAUNCH);
	if ( head.rank_offset == 0u || rows == 0u )
		return(LM_LAUNCH_OK);
	LM_LAUNCH_DEPENDENT((LmStageHeadRankTokenKernel), dim3(1u,rows), LM_STAGE_HEAD_THREADS, 0, stream, head.output_token,rows,head.rank_offset);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

__global__ static void LmStageSoftcapKernel(float *values, uint64_t count, float cap, float scale)
{
	LmDependentWait();
	LmDependentRelease();
	const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	if ( LmIsAligned16(values) )
	{
		const uint64_t vecs = count >> 2u;
		float4 *v4 = (float4 *)values;
		for ( uint64_t v = tid; v < vecs; v += stride )
		{
			float4 x = v4[v];
			if ( scale != 0.0f ) { x.x *= scale; x.y *= scale; x.z *= scale; x.w *= scale; }
			if ( cap != 0.0f )
			{
				x.x = SparkMathSoftcap(x.x, cap);
				x.y = SparkMathSoftcap(x.y, cap);
				x.z = SparkMathSoftcap(x.z, cap);
				x.w = SparkMathSoftcap(x.w, cap);
			}
			v4[v] = x;
		}
		for ( uint64_t index = (vecs << 2u) + tid; index < count; index += stride )
		{
			const float value = scale != 0.0f ? values[index] * scale : values[index];
			values[index] = cap != 0.0f ? SparkMathSoftcap(value, cap) : value;
		}
		return;
	}
	for ( uint64_t index = tid; index < count; index += stride )
	{
		const float value = scale != 0.0f ? values[index] * scale : values[index];
		values[index] = cap != 0.0f ? SparkMathSoftcap(value,cap) : value;
	}
}

static int32_t LmStageSoftcap(float *values, uint64_t count, float cap, float scale, SparkGpuStream stream)
{
	const uint64_t items = LmIsAligned16(values) ? ((count + 3u) >> 2u) : count;
	const uint64_t blocks = (items + LM_STAGE_HEAD_THREADS - 1u) / LM_STAGE_HEAD_THREADS;
	if ( (cap == 0.0f && scale == 0.0f) || count == 0u )
		return LM_LAUNCH_OK;
	LM_LAUNCH_DEPENDENT((LmStageSoftcapKernel), (uint32_t)(blocks < 4096u ? blocks : 4096u), LM_STAGE_HEAD_THREADS, 0, stream, values,count,cap,scale);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
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

static int32_t LmStageHeadCertifiedSlice(const LmStageHeadRows &head, const LmStageHeadCertified &certified, uint32_t rows, SparkGpuStream stream)
{
	if ( head.hidden_bf16 == 0 || head.normed_bf16 == 0 || (head.norm_weight == 0 && head.prenormed == 0u) || head.head_weight == 0 ||
		head.output_token == 0 || head.output_score == 0 || certified.payload == 0 || certified.scale == 0 ||
		certified.norm == 0 || certified.scratch == 0 || certified.candidates == 0 || certified.screened == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( LmStageNormRows(head,rows,stream) != LM_LAUNCH_OK )
		return(LM_LAUNCH_ERR_LAUNCH);
	return(SparkLmHostLaunchHeadCertifiedFp8RowsWithScore(stream,head.normed_bf16,head.head_weight,certified.payload,
		certified.scale,certified.norm,certified.scratch,certified.candidates,certified.screened,head.output_token,
		head.output_score,head.rank_offset,rows,head.vocab_slice_rows,head.hidden) == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
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
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.y;
	if ( row < rows && threadIdx.x == 0u )
		maxloc[row] = ((uint64_t)LmStageOrderedScore(scores[row]) << 32u) | (uint64_t)(0xFFFFFFFFu - tokens[row]);
}

__global__ static void LmStageMaxlocUnpackKernel(const uint64_t *maxloc, uint32_t *tokens, float *scores, uint32_t rows)
{
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.y;
	if ( row < rows && threadIdx.x == 0u )
	{
		tokens[row] = 0xFFFFFFFFu - (uint32_t)maxloc[row];
		scores[row] = LmStageOrderedScoreInverse((uint32_t)(maxloc[row] >> 32u));
	}
}

static int32_t LmStageMaxlocPack(const float *scores, const uint32_t *tokens, uint64_t *maxloc, uint32_t rows, SparkGpuStream stream)
{
	if ( rows == 0u )
		return LM_LAUNCH_OK;
	LM_LAUNCH_DEPENDENT((LmStageMaxlocPackKernel), dim3(1u,rows), LM_STAGE_HEAD_THREADS, 0, stream, scores,tokens,maxloc,rows);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t LmStageMaxlocUnpack(const uint64_t *maxloc, uint32_t *tokens, float *scores, uint32_t rows, SparkGpuStream stream)
{
	if ( rows == 0u )
		return LM_LAUNCH_OK;
	LM_LAUNCH_DEPENDENT((LmStageMaxlocUnpackKernel), dim3(1u,rows), LM_STAGE_HEAD_THREADS, 0, stream, maxloc,tokens,scores,rows);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

__global__ static void LmStageLastRowsGatherKernel(const uint32_t *row_begin, const uint32_t *row_indices,
	const uint16_t *hidden_bf16, uint16_t *out_bf16, uint32_t hidden)
{
	LmDependentWait();
	LmDependentRelease();
	const uint4 *source = (const uint4 *)(hidden_bf16 + (uint64_t)row_indices[row_begin[blockIdx.x + 1u] - 1u] * hidden);
	uint4 *target = (uint4 *)(out_bf16 + (uint64_t)blockIdx.x * hidden);
	for ( uint32_t k = threadIdx.x; k < hidden / 8u; k += blockDim.x )
		target[k] = source[k];
}

__global__ static void LmStageLastRowsScatterKernel(const uint32_t *row_begin, const uint32_t *row_indices,
	const uint32_t *token, const float *score, uint32_t *out_token, float *out_score, uint32_t sequences)
{
	LmDependentWait();
	LmDependentRelease();
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
	LmDependentWait();
	LmDependentRelease();
	const uint16_t *source = hidden_bf16 + (uint64_t)rows[blockIdx.x] * hidden;
	uint16_t *target = out_bf16 + (uint64_t)blockIdx.x * hidden;
	if ( ((hidden & 7u) == 0u) && LmIsAligned16(source) && LmIsAligned16(target) )
	{
		const uint4 *src4 = (const uint4 *)source;
		uint4 *dst4 = (uint4 *)target;
		for ( uint32_t k = threadIdx.x; k < (hidden >> 3u); k += blockDim.x )
			dst4[k] = src4[k];
		return;
	}
	for ( uint32_t k = threadIdx.x; k < hidden; k += blockDim.x )
		target[k] = source[k];
}
