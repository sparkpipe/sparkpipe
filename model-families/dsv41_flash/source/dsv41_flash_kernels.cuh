#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/vec.cuh"
#include "sparkpipe/spark_kv_shard.h"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"
#include "skl2_dsv41_embedded.h"

struct D41Skl2BundleTag {};

template<typename... Args>
static inline SparkGpuError D41Skl2Launch(uint32_t id, dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, Args... args)
{
	return LmLaunchEmbedded<D41Skl2BundleTag>(kD41Skl2Cubin, sizeof(kD41Skl2Cubin), (const char *)kD41Skl2Manifest,
		sizeof(kD41Skl2Manifest) - 1u, kD41Skl2Names, (uint32_t)D41SKL2_COUNT, id, grid, block, shared, stream, args...);
}

#define SPARK_DSV41_FLASH_KV_FP4_GROUP 16u
#define SPARK_DSV41_FLASH_KV_FP4_AMAX_FLOOR (6.0f * 0.001953125f)
#define SPARK_DSV41_FLASH_ENGRAM_GATE_FLOOR 1e-6f
#define SPARK_DSV41_FLASH_ENGRAM_THREADS 256u
#define SPARK_DSV41_FLASH_CANDIDATE_THREADS 256u
#define SPARK_DSV41_FLASH_FP8_BLOCK 32u
#define SPARK_DSV41_FLASH_LINEAR_WARPS 8u
#define SPARK_DSV41_FLASH_ROW_THREADS 256u
#define SPARK_DSV41_FLASH_ATTENTION_MAX_KEYS 1024u
#define SPARK_DSV41_FLASH_MAX_EXPERTS 512u
#define SPARK_DSV41_FLASH_MAX_ROUTED 16u

static __device__ __forceinline__ uint16_t SparkDsv41FlashFp8QdqLane(float value)
{
	float amax = fabsf(value);
	for (uint32_t offset = SPARK_DSV41_FLASH_FP8_BLOCK / 2u; offset != 0u; offset >>= 1u) amax = fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,offset));
	float scale = SparkMathFp8QdqScale(fmaxf(0.0f,amax));
	return(LmFloatToBf16(LmE4m3ToFloat(LmFloatToE4m3(fminf(fmaxf(value / scale,-LM_E4M3_MAX),LM_E4M3_MAX))) * scale));
}

struct SparkDsv41FlashFp8QdqLaneOp { static __device__ __forceinline__ uint16_t Apply(float value) { return SparkDsv41FlashFp8QdqLane(value); } };

static inline SparkGpuError SparkDsv41FlashLaunchKvFp4Qdq(SparkGpuStream stream, uint16_t *data_bf16, uint64_t row_stride, uint32_t row_count, uint32_t width)
{
	if ( data_bf16 == 0 || row_count == 0u || width == 0u || width % SPARK_DSV41_FLASH_KV_FP4_GROUP != 0u || row_stride < width ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	if ( (row_stride & 7u) == 0u && (((uintptr_t)data_bf16) & 15u) == 0u ) {
		const uint64_t octets = (uint64_t)row_count * (width >> 3u);
		return D41Skl2Launch(D41SKL2_d41_kv_fp4_qdq_vec_256, (uint32_t)((octets + 255u) / 256u), 256u, 0, stream, data_bf16, row_stride, row_count, width);
	}
	const uint64_t groups = (uint64_t)row_count * (width / SPARK_DSV41_FLASH_KV_FP4_GROUP);
	return D41Skl2Launch(D41SKL2_d41_kv_fp4_qdq_256, (uint32_t)((groups + 255u) / 256u), 256u, 0, stream, data_bf16, row_stride, row_count, width);
}

static inline SparkGpuError SparkDsv41FlashLaunchEngramGate(SparkGpuStream stream, uint16_t *streams_bf16, const uint16_t *kv_bf16,
	const uint16_t *q_weight_bf16, const uint16_t *k_weight_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension, float epsilon)
{
	if ( streams_bf16 == 0 || kv_bf16 == 0 || q_weight_bf16 == 0 || k_weight_bf16 == 0 || row_count == 0u || hc == 0u || dimension == 0u ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return D41Skl2Launch(D41SKL2_d41_engram_gate_256, dim3(row_count, hc), SPARK_DSV41_FLASH_ENGRAM_THREADS, 0, stream, streams_bf16, kv_bf16, q_weight_bf16, k_weight_bf16, hc, dimension, epsilon);
}

static inline SparkGpuError SparkDsv41FlashLaunchCandidateMask(SparkGpuStream stream, float *scores_f32, const uint32_t *widths, uint64_t row_stride,
	float *block_scores_f32, uint64_t block_stride, uint32_t row_count, uint32_t block_size, uint32_t topk_blocks)
{
	if ( scores_f32 == 0 || widths == 0 || block_scores_f32 == 0 || row_count == 0u || block_size == 0u || topk_blocks == 0u || block_stride * block_size < row_stride ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return D41Skl2Launch(D41SKL2_d41_candidate_mask_256, row_count, SPARK_DSV41_FLASH_CANDIDATE_THREADS, 0, stream, scores_f32, widths, row_stride, block_scores_f32, block_stride, block_size, topk_blocks);
}

static inline SparkGpuError SparkDsv41FlashLaunchShardBlockMax(SparkGpuStream stream, const float *scores_f32, uint64_t score_stride, SparkKvShard shard,
	const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *row_position, float *block_scores_f32, uint32_t block_stride, uint32_t row_count)
{
	if ( scores_f32 == 0 || sequence_of_row == 0 || context_length == 0 || row_position == 0 || block_scores_f32 == 0 || row_count == 0u ||
		shard.grain == 0u || shard.degree == 0u || shard.rank >= shard.degree || (uint64_t)block_stride * shard.grain < score_stride ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return D41Skl2Launch(D41SKL2_d41_shard_block_max_256, row_count, SPARK_DSV41_FLASH_CANDIDATE_THREADS, 0, stream, scores_f32, score_stride, shard, sequence_of_row, context_length, row_position, block_scores_f32, block_stride);
}

static inline SparkGpuError SparkDsv41FlashLaunchShardBlockMask(SparkGpuStream stream, uint32_t global_stride, const uint32_t *selected, uint32_t keep,
	SparkKvShard blocks, uint8_t *mask, uint32_t block_stride, uint32_t row_count)
{
	if ( selected == 0 || mask == 0 || row_count == 0u || keep == 0u || blocks.degree == 0u || blocks.rank >= blocks.degree ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return D41Skl2Launch(D41SKL2_d41_shard_block_mask_256, row_count, SPARK_DSV41_FLASH_CANDIDATE_THREADS, 0, stream, global_stride, selected, keep, blocks, mask, block_stride);
}

static inline SparkGpuError SparkDsv41FlashLaunchShardBlockApply(SparkGpuStream stream, float *scores_f32, uint64_t score_stride, uint32_t grain,
	const uint8_t *mask, uint32_t block_stride, uint32_t row_count)
{
	if ( scores_f32 == 0 || mask == 0 || row_count == 0u || grain == 0u || (uint64_t)block_stride * grain < score_stride ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	const uint64_t spans = (score_stride + 16u * SPARK_DSV41_FLASH_CANDIDATE_THREADS - 1u) / (16u * SPARK_DSV41_FLASH_CANDIDATE_THREADS);
	return D41Skl2Launch(D41SKL2_d41_shard_block_apply_256, dim3((uint32_t)(spans < 256u ? spans : 256u), row_count), SPARK_DSV41_FLASH_CANDIDATE_THREADS, 0, stream, scores_f32, score_stride, grain, mask, block_stride);
}

static inline SparkGpuError SparkDsv41FlashLaunchFp8BlockLinear(SparkGpuStream stream, const uint8_t *weight_e4m3, const uint8_t *scale_ue8m0,
	const uint16_t *input_bf16, float *activation_scratch_f32, uint16_t *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	if ( weight_e4m3 == 0 || scale_ue8m0 == 0 || input_bf16 == 0 || activation_scratch_f32 == 0 || output_bf16 == 0 || row_count == 0u || input_dimension == 0u ||
		input_dimension % SPARK_DSV41_FLASH_FP8_BLOCK != 0u || output_dimension % SPARK_DSV41_FLASH_FP8_BLOCK != 0u || ((uintptr_t)weight_e4m3 % 16u) != 0u )
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	uint64_t groups = (uint64_t)row_count * (input_dimension / SPARK_DSV41_FLASH_FP8_BLOCK);
	SparkGpuError status = D41Skl2Launch(D41SKL2_d41_act_quant_256, (uint32_t)((groups + 255u) / 256u), 256u, 0, stream, input_bf16, activation_scratch_f32, row_count, input_dimension);
	if ( status != SPARK_GPU_SUCCESS ) return(status);
	return D41Skl2Launch(D41SKL2_d41_fp8_block_linear_256, dim3((output_dimension + SPARK_DSV41_FLASH_LINEAR_WARPS - 1u) / SPARK_DSV41_FLASH_LINEAR_WARPS, row_count),
		SPARK_DSV41_FLASH_LINEAR_WARPS * 32u, 0, stream, weight_e4m3, scale_ue8m0, activation_scratch_f32, output_bf16, input_dimension, output_dimension);
}

static inline SparkGpuError SparkDsv41FlashLaunchRmsNorm(SparkGpuStream stream, const uint16_t *input_bf16, const uint16_t *weight_bf16, uint16_t *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	if ( input_bf16 == 0 || weight_bf16 == 0 || output_bf16 == 0 || row_count == 0u || dimension == 0u ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return D41Skl2Launch(D41SKL2_d41_rms_norm_256, row_count, SPARK_DSV41_FLASH_ROW_THREADS, 0, stream, input_bf16, weight_bf16, output_bf16, dimension, epsilon);
}

static inline SparkGpuError SparkDsv41FlashLaunchRope(SparkGpuStream stream, uint16_t *data_bf16, const float *cos_sin_f32, uint32_t head_count, uint32_t head_dimension, uint32_t rope_dimension, uint32_t inverse)
{
	if ( data_bf16 == 0 || cos_sin_f32 == 0 || head_count == 0u || rope_dimension == 0u || rope_dimension > head_dimension || rope_dimension / 2u > 1024u ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return D41Skl2Launch(D41SKL2_d41_rope_op_256, head_count, 256u, 0, stream, data_bf16, cos_sin_f32, head_count, head_dimension, rope_dimension, inverse);
}

static inline SparkGpuError SparkDsv41FlashLaunchFp8QdqCopy(SparkGpuStream stream, const uint16_t *source_bf16, uint16_t *data_bf16, uint64_t element_count)
{
	if ( source_bf16 == 0 || data_bf16 == 0 || element_count == 0u || element_count % SPARK_DSV41_FLASH_FP8_BLOCK != 0u ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	uint32_t groups = (uint32_t)(element_count / SPARK_DSV41_FLASH_FP8_BLOCK);
	return D41Skl2Launch(D41SKL2_d41_fp8_qdq_256, (groups + 7u) / 8u, 8u * SPARK_DSV41_FLASH_FP8_BLOCK, 0, stream, source_bf16, data_bf16, groups);
}

static inline SparkGpuError SparkDsv41FlashLaunchFp8Qdq(SparkGpuStream stream, uint16_t *data_bf16, uint64_t element_count) { return(SparkDsv41FlashLaunchFp8QdqCopy(stream, data_bf16, data_bf16, element_count)); }

static inline SparkGpuError SparkDsv41FlashLaunchSinkAttention(SparkGpuStream stream, const uint16_t *query_bf16, const uint16_t *keys_bf16, const int32_t *indices, uint32_t index_count, const float *sink_f32, uint16_t *output_bf16, uint32_t head_count, uint32_t head_dimension, float softmax_scale)
{
	if ( query_bf16 == 0 || keys_bf16 == 0 || indices == 0 || sink_f32 == 0 || output_bf16 == 0 || index_count == 0u || index_count > SPARK_DSV41_FLASH_ATTENTION_MAX_KEYS || head_count == 0u || head_dimension == 0u ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return D41Skl2Launch(D41SKL2_d41_sink_attention_256, head_count, SPARK_DSV41_FLASH_ROW_THREADS, 0, stream, query_bf16, keys_bf16, indices, index_count, sink_f32, output_bf16, head_dimension, softmax_scale);
}

static inline SparkGpuError SparkDsv41FlashLaunchGroupedFp8Linear(SparkGpuStream stream, const uint8_t *weight_e4m3, const uint8_t *scale_ue8m0, const uint16_t *input_bf16, uint16_t *output_bf16, uint32_t input_dimension, uint32_t output_dimension, uint32_t outputs_per_group)
{
	if ( weight_e4m3 == 0 || scale_ue8m0 == 0 || input_bf16 == 0 || output_bf16 == 0 || input_dimension % SPARK_DSV41_FLASH_FP8_BLOCK != 0u || output_dimension % SPARK_DSV41_FLASH_FP8_BLOCK != 0u || outputs_per_group == 0u || output_dimension % outputs_per_group != 0u ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return D41Skl2Launch(D41SKL2_d41_grouped_fp8_linear_256, (output_dimension + SPARK_DSV41_FLASH_LINEAR_WARPS - 1u) / SPARK_DSV41_FLASH_LINEAR_WARPS, SPARK_DSV41_FLASH_LINEAR_WARPS * 32u, 0, stream, weight_e4m3, scale_ue8m0, input_bf16, output_bf16, input_dimension, output_dimension, outputs_per_group);
}

static inline SparkGpuError SparkDsv41FlashLaunchGate(SparkGpuStream stream, const uint16_t *input_bf16, const uint16_t *weight_bf16, const float *bias_f32, uint32_t *indices_out, float *weights_out, uint32_t expert_count, uint32_t dimension, uint32_t topk, float route_scale)
{
	if ( input_bf16 == 0 || weight_bf16 == 0 || bias_f32 == 0 || indices_out == 0 || weights_out == 0 || expert_count == 0u || expert_count > SPARK_DSV41_FLASH_MAX_EXPERTS || topk == 0u || topk > SPARK_DSV41_FLASH_MAX_ROUTED || topk > expert_count ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return D41Skl2Launch(D41SKL2_d41_gate_256, 1u, SPARK_DSV41_FLASH_ROW_THREADS, 0, stream, input_bf16, weight_bf16, bias_f32, indices_out, weights_out, expert_count, dimension, topk, route_scale);
}

static inline SparkGpuError SparkDsv41FlashLaunchMxfp4Linear(SparkGpuStream stream, const uint8_t *payload_e2m1, const uint8_t *scale_ue8m0, const uint16_t *input_bf16, float *activation_scratch_f32, uint16_t *output_bf16, uint32_t input_dimension, uint32_t output_dimension)
{
	if ( payload_e2m1 == 0 || scale_ue8m0 == 0 || input_bf16 == 0 || activation_scratch_f32 == 0 || output_bf16 == 0 || input_dimension == 0u || input_dimension % 32u != 0u || output_dimension == 0u || ((uintptr_t)payload_e2m1 % 16u) != 0u ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	SparkGpuError status = D41Skl2Launch(D41SKL2_d41_act_quant_256, (input_dimension / 32u + 255u) / 256u, 256u, 0, stream, input_bf16, activation_scratch_f32, 1u, input_dimension);
	if ( status != SPARK_GPU_SUCCESS ) return(status);
	return D41Skl2Launch(D41SKL2_d41_mxfp4_linear_256, (output_dimension + SPARK_DSV41_FLASH_LINEAR_WARPS - 1u) / SPARK_DSV41_FLASH_LINEAR_WARPS, SPARK_DSV41_FLASH_LINEAR_WARPS * 32u, 0, stream, payload_e2m1, scale_ue8m0, activation_scratch_f32, output_bf16, input_dimension, output_dimension);
}

static inline SparkGpuError SparkDsv41FlashLaunchSwiglu(SparkGpuStream stream, const uint16_t *gate_bf16, const uint16_t *up_bf16, uint16_t *output_bf16, uint32_t width, float limit, float weight)
{
	if ( gate_bf16 == 0 || up_bf16 == 0 || output_bf16 == 0 || width == 0u ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return D41Skl2Launch(D41SKL2_d41_swiglu_256, (width + 255u) / 256u, 256u, 0, stream, gate_bf16, up_bf16, output_bf16, width, limit, weight);
}

static inline SparkGpuError SparkDsv41FlashLaunchAccumulate(SparkGpuStream stream, float *accumulator_f32, const uint16_t *value_bf16, uint32_t width)
{ return D41Skl2Launch(D41SKL2_d41_accumulate_256, (width + 255u) / 256u, 256u, 0, stream, accumulator_f32, value_bf16, width); }

static inline SparkGpuError SparkDsv41FlashLaunchNarrow(SparkGpuStream stream, const float *input_f32, uint16_t *output_bf16, uint32_t width)
{ return D41Skl2Launch(D41SKL2_d41_narrow_256, (width + 255u) / 256u, 256u, 0, stream, input_f32, output_bf16, width); }

static inline SparkGpuError SparkDsv41FlashLaunchBf16Linear(SparkGpuStream stream, const uint16_t *weight_bf16, const uint16_t *input_bf16, float *output_f32, uint16_t *output_bf16, uint32_t input_dimension, uint32_t output_dimension)
{
	if ( weight_bf16 == 0 || input_bf16 == 0 || (output_f32 == 0 && output_bf16 == 0) || input_dimension == 0u || output_dimension == 0u ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return D41Skl2Launch(D41SKL2_d41_bf16_linear_256, (output_dimension + SPARK_DSV41_FLASH_LINEAR_WARPS - 1u) / SPARK_DSV41_FLASH_LINEAR_WARPS, SPARK_DSV41_FLASH_LINEAR_WARPS * 32u, 0, stream, weight_bf16, input_bf16, output_f32, output_bf16, input_dimension, output_dimension);
}

static inline SparkGpuError SparkDsv41FlashLaunchCompressPool(SparkGpuStream stream, const float *kv_state_f32, const float *score_state_f32, uint16_t *pooled_bf16, uint32_t ratio, uint32_t width)
{ return D41Skl2Launch(D41SKL2_d41_compress_pool_256, (width + 255u) / 256u, 256u, 0, stream, kv_state_f32, score_state_f32, pooled_bf16, ratio, width); }

static inline SparkGpuError SparkDsv41FlashLaunchFp4Pow2QdqScalar(SparkGpuStream stream, uint16_t *data_bf16, uint64_t group_count, uint32_t group_size)
{ return D41Skl2Launch(D41SKL2_d41_fp4_pow2_qdq_256, (uint32_t)((group_count + 255u) / 256u), 256u, 0, stream, data_bf16, (uint32_t)group_count, group_size); }

static inline void SparkDsv41FlashLaunchFp4Pow2Qdq(SparkGpuStream stream, uint16_t *data_bf16, uint64_t group_count, uint32_t group_size)
{
	if ( group_size == 32u && (((uintptr_t)data_bf16) & 15u) == 0u ) {
		const uint64_t octets = group_count * 4u;
		(void)D41Skl2Launch(D41SKL2_d41_fp4_pow2_qdq_vec_256, (uint32_t)((octets + 255u) / 256u), 256u, 0, stream, data_bf16, octets);
		return;
	}
	(void)SparkDsv41FlashLaunchFp4Pow2QdqScalar(stream, data_bf16, group_count, group_size);
}

static inline SparkGpuError SparkDsv41FlashLaunchIndexScore(SparkGpuStream stream, const uint16_t *query_bf16, const uint16_t *keys_bf16, const uint16_t *head_weights_bf16, float *scores_f32, uint32_t key_count, uint32_t head_count, uint32_t head_dimension)
{
	if ( query_bf16 == 0 || keys_bf16 == 0 || head_weights_bf16 == 0 || scores_f32 == 0 || key_count == 0u || head_count == 0u || head_dimension == 0u ) return(SPARK_GPU_ERROR_INVALID_VALUE);
	return D41Skl2Launch(D41SKL2_d41_index_score_ref_256, (key_count + 7u) / 8u, 256u, 0, stream, query_bf16, keys_bf16, head_weights_bf16, scores_f32, key_count, head_count, head_dimension);
}
