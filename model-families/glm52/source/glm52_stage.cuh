#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/launch.h"
#include "runtime/gemm.cuh"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_expert_planes.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_row_layout.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include "sparkpipe/spark_glm52_stagepack.h"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_prefill.cuh"
#include "inference/kernels/attn_rope_heads.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_split.h"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/index_score.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/norm_cluster.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/project_chain.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/row_digest.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/topk_exact_plan.h"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/weight_codec.cuh"

#define GLM_THREADS 256u
#define GLM_HIDDEN SPARK_GLM52_MODEL_HIDDEN_DIMENSION
#define GLM_LAYERS SPARK_GLM52_MODEL_LAYER_COUNT
#define GLM_FIRST_ROUTED SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER
#define GLM_VOCAB SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT
#define GLM_HEADS SPARK_GLM52_MODEL_HEAD_COUNT
#define GLM_LATENT SPARK_GLM52_MODEL_LATENT_DIMENSION
#define GLM_ROPE SPARK_GLM52_MODEL_ROPE_DIMENSION
#define GLM_NOPE SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION
#define GLM_VALUE SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION
#define GLM_QUERY_A SPARK_GLM52_MODEL_QUERY_A_DIMENSION
#define GLM_SLOT (GLM_LATENT + GLM_ROPE)
#define GLM_SLOT_BYTES (GLM_SLOT * 2u)
#define GLM_PAGE_SLOTS 64u
#define GLM_SELECTED SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT
#define GLM_MAX_POSITIONS SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS
#define GLM_INDEX_HEADS SPARK_GLM52_MODEL_DSA_INDEX_HEAD_COUNT
#define GLM_INDEX_DIM SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION
#define GLM_INDEX_QUERY (GLM_INDEX_HEADS * GLM_INDEX_DIM)
#define GLM_EXPERTS SPARK_GLM52_MODEL_MOE_EXPERT_COUNT
#define GLM_TOP_K SPARK_GLM52_MODEL_MOE_TOP_K
#define GLM_EXPERT_INTER SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION
#define GLM_DENSE_INTER SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION
#define GLM_RMS_EPSILON SPARK_GLM52_MODEL_RMS_NORM_EPSILON
#define GLM_INDEX_EPSILON SPARK_GLM52_MODEL_DSA_INDEX_NORM_EPSILON
#define GLM_INDEX_SCALE SPARK_GLM52_MODEL_DSA_INDEX_SOFTMAX_SCALE
#define GLM_ROPE_THETA SPARK_GLM52_MODEL_ROPE_THETA
#define GLM_QK_SCALE SPARK_GLM52_MODEL_QK_SCALE
#define GLM_ROUTED_SCALE SPARK_GLM52_MODEL_MOE_ROUTED_SCALING_FACTOR
#define GLM_GRANULE 8u
#define GLM_GATHER_WIDTH (GLM_QUERY_A + GLM_SLOT)
#define GLM_TILE_N 128u
#define GLM_TILE_STAGES 2u
#define GLM_TILE_WARPS 8u
#define GLM_STREAM_EXPERT_ROWS 256u
#define GLM_UNIT_BYTES ((uint64_t)GLM_HIDDEN * 2u)
#define GLM_CANDIDATE_BYTES 8u
#define GLM_PARTIAL_FLOATS (GLM_LATENT + 2u)
#define GLM_SLOT_ALIGN 3u
#define GLM_SCATTER_ROWS 64u
#define GLM_SPLIT_PARTITIONS 16u
#define GLM_PREFILL_BLOCK_ROWS 4u
#define GLM_PREFILL_SPANS 8u
#define GLM_PREFILL_TABLE_BLOCKS(rows) (((rows) + GLM_PREFILL_BLOCK_ROWS - 1u) / GLM_PREFILL_BLOCK_ROWS + GLM_PREFILL_SPANS)
#define GLM_PLAN_ARRAYS 3u
#define GLM_PLAN_WORDS 2u
#define GLM_DIGESTS 1u
#define GLM_DIGEST_WORDS (3u * GLM_DIGESTS)
#define GLM_ERROR_WORDS 6u
#define GLM_SPLIT_THRESHOLD 64u
#define GLM_TENSOR_KINDS SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT
#define GLM_MAX_TENSORS 2048u
#define GLM_EXPERT_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3

static_assert(LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS == GLM_SPLIT_PARTITIONS, "split-partials sizing matches the kernel's partition cap");
static_assert(GLM_QUERY_A % GLM_GRANULE == 0u && GLM_SLOT % GLM_GRANULE == 0u, "projection slices start on 16-byte columns");

struct GlmKv
{
	static constexpr uint32_t kSlotBytes = GLM_SLOT_BYTES;
	static constexpr uint32_t kPageSlots = GLM_PAGE_SLOTS;
	static constexpr uint32_t kPageBytes = GLM_SLOT_BYTES * GLM_PAGE_SLOTS * GLM_LAYERS;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / GLM_PAGE_SLOTS; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % GLM_PAGE_SLOTS; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + GLM_PAGE_SLOTS - 1u) / GLM_PAGE_SLOTS; }
	static __host__ __device__ constexpr uint64_t PoolBytes(uint64_t pages) { return pages * (uint64_t)kPageBytes; }
};
using GlmIndexKv = LmKvLatent<16u, GLM_INDEX_DIM, 0u, GLM_PAGE_SLOTS>;
typedef LmKvGeometry<GLM_SLOT_BYTES, 1u, true> GlmKvGathered;

static inline uint32_t GlmFullIndexer(uint32_t layer)
{
	return SparkGlm52StagePackLayerHasFullIndexer(layer);
}

typedef struct GlmLayer
{
	const uint16_t *attn_norm;
	const uint16_t *q_a;
	const uint16_t *q_a_norm;
	const uint16_t *q_b;
	const uint16_t *kv_a;
	const uint16_t *kv_a_norm;
	const uint16_t *kv_b_key;
	const uint16_t *kv_b_value;
	const uint16_t *o_proj;
	const uint16_t *post_norm;
	const uint16_t *index_q;
	const uint16_t *index_k;
	const uint16_t *index_head;
	const uint16_t *index_norm_weight;
	const uint16_t *index_norm_bias;
	const uint16_t *dense_gate_up;
	const uint16_t *dense_down;
	const uint16_t *router;
	const float *router_bias;
	const uint16_t *shared_gate_up;
	const uint16_t *shared_down;
	uint64_t up_gate_payload;
	uint64_t up_gate_scale;
	uint64_t down_payload;
	uint64_t down_scale;
	uint32_t index_ordinal;
} GlmLayer;

typedef struct GlmBuffers
{
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t attn_heads;
	uint32_t q_b_rows;
	uint32_t attn_output_columns;
	uint32_t dense_gate_up_rows;
	uint32_t dense_intermediate;
	uint32_t expert_w1_rows;
	uint32_t expert_intermediate;
	uint32_t shared_gate_up_rows;
	uint32_t shared_intermediate;
	const GlmLayer *weights;
	const void *expert_w1_weight;
	const void *expert_w1_scale;
	const void *expert_w2_weight;
	const void *expert_w2_scale;
	uint16_t *hidden_bf16;
	uint16_t *residual_bf16;
	uint16_t *normed_bf16;
	uint16_t *q_compressed_bf16;
	uint16_t *q_bf16;
	uint16_t *query_latent_bf16;
	uint16_t *query_rope_bf16;
	uint16_t *index_query_bf16;
	uint16_t *index_key_bf16;
	uint16_t *index_head_weight_bf16;
	uint16_t *kv_slot_bf16;
	uint16_t *attention_latent_bf16;
	uint16_t *attention_value_bf16;
	uint16_t *attention_out_bf16;
	uint16_t *gate_up_bf16;
	uint16_t *intermediate_bf16;
	uint16_t *expert_out_bf16;
	uint16_t *shared_out_bf16;
	float *router_logits;
	float *selection_scores;
	uint32_t *route_expert;
	float *route_weight;
	uint32_t *route_source_token;
	uint32_t *route_packed_row;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	LmKvView cache;
	LmKvView index_cache;
	const uint32_t *sequence_of_row;
	const uint32_t *context_length;
	const uint32_t *positions;
	const uint32_t *row_positions;
	uint32_t *selected_positions;
	uint32_t selected_position_count;
	uint32_t selection_rows;
	uint32_t *prefill_union_positions;
	uint8_t *prefill_union_masks;
	uint32_t *prefill_union_counts;
	uint64_t prefill_union_entries;
	float *topk_scratch_values;
	uint32_t *topk_scratch_positions;
	uint64_t topk_scratch_entries;
	float *attention_split_partials;
	uint64_t attention_split_partial_blocks;
	uint32_t decode_split_context_threshold;
	uint32_t single_sequence_rows;
	const uint32_t *prefill_block_table;
	uint32_t prefill_table_blocks;
	uint16_t *projection_gather_bf16;
	uint16_t *projection_local_bf16;
	SparkKvShard kv_shard;
	LmKvShardView cache_shard;
	LmKvShardView index_shard;
	LmKvView cache_gathered;
	uint8_t *kv_gather_latent_pack;
	uint8_t *kv_gather_latent_tail;
	uint32_t *kv_select_local;
	uint2 *kv_select_candidates;
	uint32_t kv_select_keep;
	uint32_t kv_select_rows;
	const uint32_t *kv_gather_sequences;
	const uint32_t *kv_gather_offset;
	const uint32_t *kv_gather_old_bound;
	const uint32_t *kv_gather_rows_zero;
	const uint32_t *kv_gather_context_word;
	unsigned long long *kv_gather_digest;
	uint32_t kv_gather_old_keys;
	uint16_t *shard_send_bf16;
	uint64_t shard_candidate_offset;
	const uint16_t *shard_received_bf16;
	uint64_t shard_rank_stride_bytes;
	float *shard_partials;
	float *shard_partials_received;
	uint64_t shard_partial_stride;
	uint32_t *index_local_selected;
} GlmBuffers;

static inline void GlmProjectionSlice(uint32_t total, uint32_t tp_degree, uint32_t tp_rank, uint32_t *first, uint32_t *count)
{
	const uint32_t granules = total / GLM_GRANULE, base = granules / tp_degree, extra = granules % tp_degree;
	*count = GLM_GRANULE * (base + (tp_rank < extra ? 1u : 0u));
	*first = GLM_GRANULE * (tp_rank * base + (tp_rank < extra ? tp_rank : extra));
}

static inline uint32_t GlmProjectionSliceMax(uint32_t total, uint32_t tp_degree)
{
	return GLM_GRANULE * ((total / GLM_GRANULE + tp_degree - 1u) / tp_degree);
}

static inline uint32_t GlmProjectionSliceWidth(uint32_t tp_degree)
{
	return GlmProjectionSliceMax(GLM_QUERY_A, tp_degree) + GlmProjectionSliceMax(GLM_SLOT, tp_degree);
}

static __device__ __forceinline__ void GlmProjectionSource(uint32_t column, uint32_t total, uint32_t tp_degree, uint32_t *rank, uint32_t *offset)
{
	const uint32_t granules = total / GLM_GRANULE, base = granules / tp_degree, extra = granules % tp_degree;
	const uint32_t granule = column / GLM_GRANULE, boundary = extra * (base + 1u);
	uint32_t local;
	if ( granule < boundary )
	{
		*rank = granule / (base + 1u);
		local = granule % (base + 1u);
	}
	else
	{
		*rank = extra + (granule - boundary) / base;
		local = (granule - boundary) % base;
	}
	*offset = local * GLM_GRANULE + column % GLM_GRANULE;
}

static __global__ void GlmProjectionUnpackKernel(const uint16_t *gathered, uint16_t *query_out, uint16_t *latent_out, uint32_t rows,
	uint32_t tp_degree, uint32_t width, uint32_t query_max)
{
	const uint32_t row = blockIdx.x;
	uint32_t rank, offset;
	for ( uint32_t column = threadIdx.x; column < GLM_GATHER_WIDTH; column += blockDim.x )
	{
		if ( column < GLM_QUERY_A )
		{
			GlmProjectionSource(column, GLM_QUERY_A, tp_degree, &rank, &offset);
			query_out[(uint64_t)row * GLM_QUERY_A + column] = gathered[((uint64_t)rank * rows + row) * width + offset];
		}
		else
		{
			GlmProjectionSource(column - GLM_QUERY_A, GLM_SLOT, tp_degree, &rank, &offset);
			latent_out[(uint64_t)row * GLM_SLOT + column - GLM_QUERY_A] = gathered[((uint64_t)rank * rows + row) * width + query_max + offset];
		}
	}
}

static __global__ void GlmWaveContextKernel(const uint32_t *slots, const uint32_t *positions, uint32_t *context_lengths, uint32_t rows)
{
	uint32_t row;
	if ( blockIdx.x != 0u )
		return;
	for ( row = threadIdx.x; row < rows; row += blockDim.x )
		context_lengths[slots[row]] = 0u;
	__syncthreads();
	for ( row = threadIdx.x; row < rows; row += blockDim.x )
		atomicMax(&context_lengths[slots[row]], positions[row] + 1u);
}

static __global__ void GlmGatherPlanKernel(const uint32_t *slots, const uint32_t *positions, uint32_t rows, uint32_t capacity, uint32_t *plan,
	LmKvAccessError *error)
{
	uint32_t sequence, row, first, end, contiguous = 1u;
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	sequence = slots[0];
	first = positions[0];
	end = positions[0] + 1u;
	for ( row = 1u; row < rows; row++ )
	{
		if ( slots[row] != sequence || positions[row] < positions[row - 1u] )
		{
			LmFrameErrorReport(error, LM_FRAME_ERROR_PAGE_TABLE_OUT_OF_RANGE, LM_KV_ACCESS_WRITE, row, slots[row], positions[row], positions[row - 1u]);
			return;
		}
		if ( positions[row] - positions[row - 1u] > 1u )
			contiguous = 0u;
		end = positions[row] + 1u;
	}
	plan[0] = sequence;
	plan[capacity + sequence] = 0u;
	plan[2u * capacity + sequence] = contiguous != 0u ? first : end;
	plan[3u * capacity] = contiguous != 0u ? first : end;
	plan[3u * capacity + 1u] = end;
}

static __global__ void GlmPadRowsKernel(const uint32_t *positions, const uint32_t *slots, uint32_t *padded_positions, uint32_t *padded_slots,
	uint32_t rows, uint32_t padded)
{
	for ( uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < padded; row += gridDim.x * blockDim.x )
	{
		padded_positions[row] = positions[row < rows ? row : rows - 1u];
		padded_slots[row] = slots[row < rows ? row : rows - 1u];
	}
}

static __global__ void GlmRowsGatherKernel(const uint32_t *order, const uint16_t *source, uint16_t *target, uint32_t width)
{
	const uint64_t from = (uint64_t)order[blockIdx.x] * width, to = (uint64_t)blockIdx.x * width;
	for ( uint32_t index = threadIdx.x; index < width; index += blockDim.x )
		target[to + index] = source[from + index];
}

static __global__ void GlmRowsScatterKernel(const uint32_t *order, const uint16_t *source, uint16_t *target, uint32_t width)
{
	const uint64_t from = (uint64_t)blockIdx.x * width, to = (uint64_t)order[blockIdx.x] * width;
	for ( uint32_t index = threadIdx.x; index < width; index += blockDim.x )
		target[to + index] = source[from + index];
}

template<class Format>
static int32_t GlmSkinny(const void *weight, const uint16_t *activation, uint16_t *output_bf16, float *output_f32, uint32_t rows,
	uint32_t input_dimension, uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset, cudaStream_t stream)
{
	LmSkinnyArguments args;
	int32_t status;
	memset(&args, 0, sizeof(args));
	args.weight = (const uint8_t *)weight;
	args.activation = activation;
	args.output_bf16 = output_bf16;
	args.output_f32 = output_f32;
	args.scale = LmScaleTensorNone();
	args.rows = rows;
	args.input_dimension = input_dimension;
	args.output_dimension = output_dimension;
	args.output_row_stride = output_row_stride != 0u ? output_row_stride : output_dimension;
	args.output_column_offset = output_column_offset;
	status = LmSkinnyValidate<Format>(&args);
	if ( status != LM_LAUNCH_OK )
		return status;
	return LmSkinnyLaunchRows<Format>(&args, input_dimension / LmSkinnyFormat<Format>::kElements, stream);
}

static int32_t GlmLinear(const uint16_t *activation, const void *weight, uint16_t *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset, uint32_t multiprocessors, cudaStream_t stream)
{
	if ( activation == 0 || weight == 0 || output == 0 || rows == 0u || input_dimension == 0u || output_dimension == 0u || multiprocessors == 0u )
		return LM_LAUNCH_ERR_SHAPE;
	if ( rows <= LM_SKINNY_ROWS_WIDE )
		return GlmSkinny<LmBf16Format>(weight, activation, output, 0, rows, input_dimension, output_dimension, output_row_stride, output_column_offset, stream);
	return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), activation, output, 0, rows, input_dimension, output_dimension,
		output_row_stride, output_column_offset, multiprocessors, stream);
}

static int32_t GlmLaunched(void)
{
	return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t GlmIndexerKeys(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	int32_t status = GlmLinear(b->normed_bf16, b->weights->index_k, b->index_key_bf16, rows, GLM_HIDDEN, GLM_INDEX_DIM, GLM_INDEX_DIM, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmLayerNormKernel<GLM_THREADS,uint16_t>), rows, GLM_THREADS, (GLM_INDEX_DIM + 8u) * sizeof(float), stream,
		b->index_key_bf16, b->weights->index_norm_weight, b->weights->index_norm_bias, b->index_key_bf16, GLM_INDEX_DIM, GLM_INDEX_DIM, GLM_INDEX_EPSILON);
	LM_LAUNCH((LmRopeKernel<GLM_THREADS,LM_ROPE_INTERLEAVED>), rows, GLM_THREADS, 0, stream,
		b->index_key_bf16, b->positions, GLM_INDEX_DIM, 0u, GLM_ROPE, GLM_ROPE_THETA);
	LM_LAUNCH((LmKvShardStoreKernel<GlmIndexKv,GLM_THREADS>), rows, GLM_THREADS, 0, stream,
		b->index_shard, b->index_key_bf16, b->sequence_of_row, b->positions, rows, GLM_INDEX_DIM);
	return GlmLaunched();
}

static int32_t GlmIndexerQuery(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	int32_t status = GlmLinear(b->q_compressed_bf16, b->weights->index_q, b->index_query_bf16, rows, GLM_QUERY_A, GLM_INDEX_QUERY, GLM_INDEX_QUERY, 0u, sms, stream);
	if ( status == LM_LAUNCH_OK )
		status = GlmLinear(b->normed_bf16, b->weights->index_head, b->index_head_weight_bf16, rows, GLM_HIDDEN, GLM_INDEX_HEADS, GLM_INDEX_HEADS, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmRopePerHeadKernel<GLM_THREADS,LM_ROPE_INTERLEAVED>), dim3(rows,GLM_INDEX_HEADS), GLM_THREADS, 0, stream,
		b->index_query_bf16, b->positions, GLM_INDEX_HEADS, GLM_INDEX_DIM, 0u, GLM_ROPE, GLM_ROPE_THETA);
	return GlmLaunched();
}

static uint32_t GlmIndexerChunkRows(const GlmBuffers *b, uint32_t rows)
{
	return b->selection_rows != 0u && b->selection_rows < rows ? b->selection_rows : rows;
}

static int32_t GlmIndexerScore(const GlmBuffers *b, LmKvShardView pages, uint32_t rows, uint32_t score_stride, uint32_t keep, uint32_t *selected,
	uint2 *candidates, cudaStream_t stream)
{
	const uint32_t chunk_rows = GlmIndexerChunkRows(b, rows);
	for ( uint32_t first = 0u; first < rows; first += chunk_rows )
	{
		const uint32_t count = rows - first < chunk_rows ? rows - first : chunk_rows;
		if ( LmWeightedSparseScoreLaunch<GlmIndexKv,GLM_INDEX_HEADS,GLM_INDEX_DIM>(b->index_query_bf16 + (uint64_t)first * GLM_INDEX_QUERY,
				b->index_head_weight_bf16 + (uint64_t)first * GLM_INDEX_HEADS, pages, b->sequence_of_row + first, b->context_length,
				b->row_positions != 0 ? b->row_positions + first : 0, count, score_stride, GLM_INDEX_SCALE / sqrtf((float)GLM_INDEX_HEADS),
				b->selection_scores, stream) != cudaSuccess ||
			LmTopkExactLaunch<GLM_THREADS>(b->selection_scores, count, score_stride, keep, LM_TOPK_EXACT_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS,
				b->topk_scratch_values, b->topk_scratch_positions, b->topk_scratch_entries, selected + (uint64_t)first * keep, stream) != cudaSuccess )
			return LM_LAUNCH_ERR_LAUNCH;
		if ( candidates != 0 && LmIndexShardCandidatePackLaunch<GLM_THREADS>(b->kv_shard, b->selection_scores, score_stride,
				selected + (uint64_t)first * keep, keep, count, candidates + (uint64_t)first * keep, stream) != cudaSuccess )
			return LM_LAUNCH_ERR_LAUNCH;
	}
	return GlmLaunched();
}

static int32_t GlmAttentionProject(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	uint32_t q_first, q_count, kv_first, kv_count, width, query_max;
	int32_t status = LM_LAUNCH_OK;
	LmFusedResidualRmsNormLaunch<GLM_THREADS,uint16_t>(rows, stream,
		b->hidden_bf16, b->residual_bf16, b->weights->attn_norm, b->residual_bf16, b->normed_bf16, GLM_HIDDEN, GLM_HIDDEN, GLM_RMS_EPSILON);
	GlmProjectionSlice(GLM_QUERY_A, b->tp_degree, b->tp_rank, &q_first, &q_count);
	GlmProjectionSlice(GLM_SLOT, b->tp_degree, b->tp_rank, &kv_first, &kv_count);
	width = GlmProjectionSliceWidth(b->tp_degree);
	query_max = GlmProjectionSliceMax(GLM_QUERY_A, b->tp_degree);
	if ( q_count != 0u && kv_count != 0u )
	{
		const LmSkinnyDenseTarget targets[2] = {
			{b->weights->q_a + (uint64_t)q_first * GLM_HIDDEN, b->projection_local_bf16, 0, q_count, width, 0u},
			{b->weights->kv_a + (uint64_t)kv_first * GLM_HIDDEN, b->projection_local_bf16, 0, kv_count, width, query_max}};
		status = LmSkinnyDenseMulti<LmBf16Format>(targets, 2u, b->normed_bf16, rows, GLM_HIDDEN, stream);
		if ( status != LM_LAUNCH_ERR_SHAPE )
			return status;
	}
	status = LM_LAUNCH_OK;
	if ( q_count != 0u )
		status = GlmLinear(b->normed_bf16, b->weights->q_a + (uint64_t)q_first * GLM_HIDDEN, b->projection_local_bf16, rows, GLM_HIDDEN, q_count, width, 0u, sms, stream);
	if ( status == LM_LAUNCH_OK && kv_count != 0u )
		status = GlmLinear(b->normed_bf16, b->weights->kv_a + (uint64_t)kv_first * GLM_HIDDEN, b->projection_local_bf16, rows, GLM_HIDDEN, kv_count, width, query_max, sms, stream);
	return status;
}

static int32_t GlmAttentionPrologue(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	int32_t status;
	GlmProjectionUnpackKernel<<<rows, GLM_THREADS, 0, stream>>>(b->projection_gather_bf16, b->q_compressed_bf16, b->kv_slot_bf16, rows, b->tp_degree,
		GlmProjectionSliceWidth(b->tp_degree), GlmProjectionSliceMax(GLM_QUERY_A, b->tp_degree));
	if ( cudaPeekAtLastError() != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	LmFusedResidualRmsNormLaunch<GLM_THREADS,uint16_t>(rows, stream,
		b->q_compressed_bf16, 0, b->weights->q_a_norm, 0, b->q_compressed_bf16, GLM_QUERY_A, GLM_QUERY_A, GLM_RMS_EPSILON);
	status = GlmLinear(b->q_compressed_bf16, b->weights->q_b, b->q_bf16, rows, GLM_QUERY_A, b->q_b_rows, b->q_b_rows, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LmFusedResidualRmsNormLaunch<GLM_THREADS,uint16_t>(rows, stream,
		b->kv_slot_bf16, 0, b->weights->kv_a_norm, 0, b->kv_slot_bf16, GLM_LATENT, GLM_SLOT, GLM_RMS_EPSILON);
	LM_LAUNCH((LmExtractRopePerHeadKernel<GLM_THREADS,LM_ROPE_INTERLEAVED>), dim3(rows,b->attn_heads), GLM_THREADS, 0, stream,
		b->q_bf16, b->query_rope_bf16, b->positions, b->attn_heads, GLM_NOPE + GLM_ROPE, GLM_NOPE, GLM_ROPE, GLM_ROPE_THETA);
	LM_LAUNCH((LmRopeKernel<GLM_THREADS,LM_ROPE_INTERLEAVED>), rows, GLM_THREADS, 0, stream,
		b->kv_slot_bf16, b->positions, GLM_SLOT, GLM_LATENT, GLM_ROPE, GLM_ROPE_THETA);
	if ( LmPerHeadProjectChainLaunch<GLM_THREADS,GLM_NOPE,GLM_LATENT,GLM_NOPE + GLM_ROPE,0u>(b->q_bf16, b->weights->kv_b_key + (uint64_t)b->tp_rank *
			b->attn_heads * GLM_LATENT * GLM_NOPE, b->query_latent_bf16, b->attn_heads, rows, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	return LM_LAUNCH_OK;
}

static int32_t GlmLatentStore(const GlmBuffers *b, uint32_t rows, cudaStream_t stream)
{
	LM_LAUNCH((LmKvShardStoreKernel<GlmKv,GLM_THREADS>), rows, GLM_THREADS, 0, stream,
		b->cache_shard, b->kv_slot_bf16, b->sequence_of_row, b->positions, rows, GLM_SLOT);
	return GlmLaunched();
}

static int32_t GlmLatentAttentionGathered(const GlmBuffers *b, LmKvView cache, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	const uint32_t *selected = context > GLM_SELECTED ? b->selected_positions : 0;
	const uint32_t selected_count = context > GLM_SELECTED ? b->selected_position_count : 0u;
	if ( (b->single_sequence_rows != 0u || b->prefill_block_table != 0) && rows > LM_SKINNY_ROWS && selected == 0 )
		return LmLatentAttentionPrefillLaunch<GlmKvGathered,GLM_LATENT,GLM_ROPE>(b->query_latent_bf16, b->query_rope_bf16, cache, b->sequence_of_row,
			b->row_positions, b->attn_heads, GLM_QK_SCALE, b->attention_latent_bf16, rows, b->prefill_block_table, b->prefill_table_blocks, stream) ==
			cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	if ( (b->single_sequence_rows != 0u || b->prefill_block_table != 0) && rows > LM_SKINNY_ROWS && selected != 0 && b->prefill_union_positions != 0 )
		return LmLatentAttentionSparsePrefillLaunch<GlmKvGathered,GLM_LATENT,GLM_ROPE,GLM_SELECTED>(b->query_latent_bf16, b->query_rope_bf16, cache,
			b->sequence_of_row, b->row_positions, b->attn_heads, GLM_QK_SCALE, b->attention_latent_bf16, rows, b->prefill_block_table,
			b->prefill_table_blocks, selected, selected_count, b->prefill_union_positions, b->prefill_union_masks, b->prefill_union_counts,
			b->prefill_union_entries, stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	return LmLatentRopeHeadsSplitLaunch<GlmKvGathered,GLM_THREADS,GLM_LATENT,GLM_ROPE,true>(b->query_latent_bf16, b->query_rope_bf16, cache,
		b->sequence_of_row, b->context_length, selected, selected_count, b->attn_heads, GLM_QK_SCALE, b->attention_latent_bf16, b->row_positions,
		rows, context > GLM_SELECTED ? GLM_SELECTED : context, b->decode_split_context_threshold, b->attention_split_partials,
		(uint32_t)b->attention_split_partial_blocks, sms, stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t GlmAttentionOutput(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	if ( LmPerHeadProjectChainLaunch<GLM_THREADS,GLM_LATENT,GLM_VALUE>(b->attention_latent_bf16, b->weights->kv_b_value + (uint64_t)b->tp_rank *
			b->attn_heads * GLM_VALUE * GLM_LATENT, b->attention_value_bf16, b->attn_heads, rows, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	return GlmLinear(b->attention_value_bf16, b->weights->o_proj, b->attention_out_bf16, rows, b->attn_output_columns, GLM_HIDDEN, GLM_HIDDEN, 0u, sms, stream);
}

static uint32_t GlmShardIndexing(uint32_t layer, uint32_t context)
{
	return GlmFullIndexer(layer) != 0u && context > GLM_SELECTED ? 1u : 0u;
}

static uint32_t GlmShardKeep(SparkKvShard shard, uint32_t context)
{
	const uint32_t stride = SparkKvShardGatherKeys(shard, context);
	return stride < GLM_SELECTED ? stride : GLM_SELECTED;
}

static int32_t GlmShardStoreAll(const GlmBuffers *b, uint32_t rows, uint32_t layer, uint32_t sms, cudaStream_t stream)
{
	int32_t status = GlmAttentionPrologue(b, rows, sms, stream);
	if ( status == LM_LAUNCH_OK && GlmFullIndexer(layer) != 0u )
		status = GlmIndexerKeys(b, rows, sms, stream);
	if ( status == LM_LAUNCH_OK )
		status = GlmLatentStore(b, rows, stream);
	return status;
}

static int32_t GlmShardScatterPre(const GlmBuffers *b, uint32_t rows, uint32_t context, uint32_t layer, uint32_t sms, cudaStream_t stream)
{
	int32_t status = GlmShardStoreAll(b, rows, layer, sms, stream);
	if ( status == LM_LAUNCH_OK && GlmShardIndexing(layer, context) != 0u )
	{
		status = GlmIndexerQuery(b, rows, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = GlmIndexerScore(b, b->index_shard, rows, SparkKvShardGatherKeys(b->kv_shard, context), GlmShardKeep(b->kv_shard, context),
				b->index_local_selected, (uint2 *)((uint8_t *)b->shard_send_bf16 + b->shard_candidate_offset), stream);
	}
	if ( status != LM_LAUNCH_OK )
		return status;
	return LmLatentShardQueryPackLaunch<GLM_LATENT,GLM_ROPE>(b->query_latent_bf16, b->query_rope_bf16, b->attn_heads, rows, b->shard_send_bf16, stream) ==
		cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t GlmShardScatterMid(const GlmBuffers *b, uint32_t rows, uint32_t context, uint32_t layer, cudaStream_t stream)
{
	if ( GlmShardIndexing(layer, context) != 0u )
	{
		const uint32_t keep = GlmShardKeep(b->kv_shard, context), chunk_rows = GlmIndexerChunkRows(b, rows);
		const uint2 *candidates = (const uint2 *)((const uint8_t *)b->shard_received_bf16 + b->shard_candidate_offset);
		for ( uint32_t first = 0u; first < rows; first += chunk_rows )
		{
			const uint32_t count = rows - first < chunk_rows ? rows - first : chunk_rows;
			if ( LmIndexShardCandidateScatterLaunch<GLM_THREADS>(candidates + (uint64_t)first * keep, b->shard_rank_stride_bytes / sizeof(uint2),
					b->kv_shard.degree, keep, count, context, b->selection_scores, stream) != cudaSuccess ||
				LmTopkExactLaunch<GLM_THREADS>(b->selection_scores, count, context, GLM_SELECTED, LM_TOPK_EXACT_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS,
					b->topk_scratch_values, b->topk_scratch_positions, b->topk_scratch_entries, b->selected_positions + (uint64_t)first * GLM_SELECTED,
					stream) != cudaSuccess )
				return LM_LAUNCH_ERR_LAUNCH;
		}
	}
	return LmLatentShardPartialLaunch<GlmKv,LmKvShardView,GLM_LATENT,GLM_ROPE>(b->cache_shard, b->shard_received_bf16,
		b->shard_rank_stride_bytes / sizeof(uint16_t), b->attn_heads, b->sequence_of_row, b->context_length, b->row_positions,
		context > GLM_SELECTED ? b->selected_positions : 0, GLM_SELECTED, GLM_SELECTED, GLM_QK_SCALE, b->shard_partials, b->shard_partial_stride,
		rows, stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t GlmShardScatterPost(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	if ( LmLatentShardMergeLaunch<GLM_LATENT>(b->shard_partials_received, b->shard_partial_stride, b->kv_shard.degree, b->attn_heads,
			b->attention_latent_bf16, rows, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	return GlmAttentionOutput(b, rows, sms, stream);
}

static int32_t GlmShardGatherPre(const GlmBuffers *b, uint32_t rows, uint32_t context, uint32_t layer, uint32_t sms, cudaStream_t stream)
{
	const uint32_t indexing = GlmShardIndexing(layer, context);
	int32_t status = GlmShardStoreAll(b, rows, layer, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( cudaMemcpyAsync(b->kv_gather_latent_tail, b->kv_slot_bf16, (size_t)rows * GLM_SLOT_BYTES, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
		LmRowDigestLaunch<GLM_THREADS>(b->kv_slot_bf16, GLM_SLOT, b->positions, rows, layer, b->kv_gather_digest, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	if ( indexing != 0u )
	{
		status = GlmIndexerQuery(b, b->kv_select_rows, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = GlmIndexerScore(b, b->index_shard, b->kv_select_rows, SparkKvShardGatherKeys(b->kv_shard, context), b->kv_select_keep,
				b->kv_select_local, b->kv_select_candidates, stream);
		if ( status != LM_LAUNCH_OK )
			return status;
	}
	if ( b->kv_gather_old_keys == 0u )
		return LM_LAUNCH_OK;
	return LmKvShardGatherPackLaunch<GlmKv,GLM_THREADS>(b->cache_shard, b->kv_gather_sequences, b->kv_gather_offset, b->kv_gather_old_bound, 1u,
		b->kv_gather_old_keys, b->kv_gather_latent_pack, stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t GlmShardGatherPost(const GlmBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	GlmBuffers local = *b;
	int32_t status;
	local.sequence_of_row = b->kv_gather_rows_zero;
	local.context_length = b->kv_gather_context_word;
	status = GlmLatentAttentionGathered(&local, local.cache_gathered, rows, context, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	return GlmAttentionOutput(b, rows, sms, stream);
}

static int32_t GlmDenseMlp(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	int32_t status;
	LmFusedResidualRmsNormLaunch<GLM_THREADS,uint16_t>(rows, stream,
		b->attention_out_bf16, b->residual_bf16, b->weights->post_norm, b->residual_bf16, b->normed_bf16, GLM_HIDDEN, GLM_HIDDEN, GLM_RMS_EPSILON);
	status = GlmLinear(b->normed_bf16, b->weights->dense_gate_up, b->gate_up_bf16, rows, GLM_HIDDEN, b->dense_gate_up_rows, b->dense_gate_up_rows, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmSiluMulKernel<GLM_THREADS>), rows, GLM_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16, b->dense_intermediate, false);
	return GlmLinear(b->intermediate_bf16, b->weights->dense_down, b->hidden_bf16, rows, b->dense_intermediate, GLM_HIDDEN, GLM_HIDDEN, 0u, sms, stream);
}

static int32_t GlmMoeRoute(const GlmBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	int32_t status;
	LmFusedResidualRmsNormLaunch<GLM_THREADS,uint16_t>(rows, stream,
		b->attention_out_bf16, b->residual_bf16, b->weights->post_norm, b->residual_bf16, b->normed_bf16, GLM_HIDDEN, GLM_HIDDEN, GLM_RMS_EPSILON);
	status = rows <= LM_SKINNY_ROWS_WIDE
		? GlmSkinny<LmBf16Format>(b->weights->router, b->normed_bf16, 0, b->router_logits, rows, GLM_HIDDEN, GLM_EXPERTS, 0u, 0u, stream)
		: LmStreamGemmDense<LmBf16Format>(b->weights->router, LmScaleTensorNone(), b->normed_bf16, 0, b->router_logits, rows, GLM_HIDDEN, GLM_EXPERTS,
			0u, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK || GlmLaunched() != LM_LAUNCH_OK )
		return status != LM_LAUNCH_OK ? status : LM_LAUNCH_ERR_LAUNCH;
	if ( LmTopkRouteLaunch<GLM_THREADS,GLM_TOP_K,true,LM_TOPK_SCORE_SIGMOID>(rows, b->router_logits, GLM_EXPERTS, b->route_expert, b->route_weight,
			b->weights->router_bias, 0, GLM_ROUTED_SCALE, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	status = LmRouteBuild<GLM_THREADS,GLM_EXPERTS>(b->route_expert, rows, packed_rows, GLM_TOP_K, b->group_row_offset, b->route_packed_row,
		b->route_source_token, b->expert_w1_rows, GLM_HIDDEN, GLM_TILE_N, b->group_tile_prefix_w1, b->group_tile_prefix_w2, stream);
	return status != LM_LAUNCH_OK ? status : GlmLaunched();
}

static int32_t GlmExpertsGateUp(const GlmBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	using Format = typename LmWeightCodec<GLM_EXPERT_CODEC>::Format;
	LmGemmArguments gemm;
	int32_t status;
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmWeightCodecScaleTensor<GLM_EXPERT_CODEC>(b->expert_w1_scale, GLM_EXPERTS, b->expert_w1_rows, GLM_HIDDEN);
	if ( rows >= GLM_STREAM_EXPERT_ROWS )
	{
		status = LmStreamGemmGrouped<Format>(b->expert_w1_weight, gemm.scale_b, b->normed_bf16, b->gate_up_bf16, b->group_row_offset,
			b->route_source_token, GLM_EXPERTS, packed_rows, GLM_HIDDEN, b->expert_w1_rows, sms, stream);
		return status == LM_LAUNCH_OK && cudaPeekAtLastError() != cudaSuccess ? LM_LAUNCH_ERR_LAUNCH : status;
	}
	status = rows == 1u ? LmSkinnyExperts<Format>(b->expert_w1_weight, gemm.scale_b, b->normed_bf16, b->gate_up_bf16, b->route_expert,
		b->route_packed_row, packed_rows, GLM_TOP_K, 0u, GLM_HIDDEN, b->expert_w1_rows, stream) : LM_LAUNCH_ERR_SHAPE;
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmSkinnyGroupedExperts<Format>(b->expert_w1_weight, gemm.scale_b, b->normed_bf16, b->gate_up_bf16, b->group_row_offset,
			b->route_source_token, GLM_EXPERTS, packed_rows, 0u, GLM_HIDDEN, b->expert_w1_rows, stream);
	if ( status != LM_LAUNCH_ERR_SHAPE )
		return status == LM_LAUNCH_OK && cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : status == LM_LAUNCH_OK ? LM_LAUNCH_ERR_LAUNCH : status;
	gemm.prefix_built = 1u;
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix_w1;
	gemm.source_row_map = b->route_source_token;
	gemm.source_row_count = rows;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmWeightOnlyIndirectLaunch<Format,GLM_TILE_N,GLM_TILE_STAGES,GLM_TILE_WARPS>(&gemm, b->normed_bf16, b->expert_w1_weight, packed_rows,
		rows, GLM_TOP_K, GLM_EXPERTS, GLM_HIDDEN, b->expert_w1_rows, sms, stream);
	return status != LM_LAUNCH_OK ? status : GlmLaunched();
}

static int32_t GlmExpertsDown(const GlmBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	using Format = typename LmWeightCodec<GLM_EXPERT_CODEC>::Format;
	LmGemmArguments gemm;
	int32_t status;
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmWeightCodecScaleTensor<GLM_EXPERT_CODEC>(b->expert_w2_scale, GLM_EXPERTS, GLM_HIDDEN, b->expert_intermediate);
	if ( rows >= GLM_STREAM_EXPERT_ROWS )
	{
		status = LmStreamGemmGrouped<Format>(b->expert_w2_weight, gemm.scale_b, b->intermediate_bf16, b->expert_out_bf16, b->group_row_offset, 0,
			GLM_EXPERTS, packed_rows, b->expert_intermediate, GLM_HIDDEN, sms, stream);
		return status == LM_LAUNCH_OK && cudaPeekAtLastError() != cudaSuccess ? LM_LAUNCH_ERR_LAUNCH : status;
	}
	status = LmSkinnyExperts<Format>(b->expert_w2_weight, gemm.scale_b, b->intermediate_bf16, b->expert_out_bf16, b->route_expert, b->route_packed_row,
		packed_rows, GLM_TOP_K, 1u, b->expert_intermediate, GLM_HIDDEN, stream);
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmSkinnyGroupedExperts<Format>(b->expert_w2_weight, gemm.scale_b, b->intermediate_bf16, b->expert_out_bf16, b->group_row_offset,
			b->route_source_token, GLM_EXPERTS, packed_rows, 1u, b->expert_intermediate, GLM_HIDDEN, stream);
	if ( status != LM_LAUNCH_ERR_SHAPE )
		return status == LM_LAUNCH_OK && cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : status == LM_LAUNCH_OK ? LM_LAUNCH_ERR_LAUNCH : status;
	gemm.prefix_built = 1u;
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix_w2;
	gemm.output_bf16 = b->expert_out_bf16;
	status = LmGemmWeightOnlyLaunch<Format,GLM_TILE_N,GLM_TILE_STAGES,GLM_TILE_WARPS>(&gemm, b->intermediate_bf16, b->expert_w2_weight, packed_rows, rows,
		GLM_TOP_K, GLM_EXPERTS, b->expert_intermediate, GLM_HIDDEN, sms, true, stream);
	return status != LM_LAUNCH_OK ? status : GlmLaunched();
}

static int32_t GlmMoeExperts(const GlmBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	int32_t status = GlmExpertsGateUp(b, rows, packed_rows, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmSiluMulKernel<GLM_THREADS>), packed_rows, GLM_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16, b->expert_intermediate, false);
	status = GlmExpertsDown(b, rows, packed_rows, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmMoeFinalizeKernel<GLM_THREADS>), dim3((GLM_HIDDEN + GLM_THREADS - 1u) / GLM_THREADS, rows), GLM_THREADS, 0, stream,
		b->expert_out_bf16, b->route_packed_row, b->route_weight, b->hidden_bf16, rows, GLM_TOP_K, GLM_HIDDEN);
	status = GlmLinear(b->normed_bf16, b->weights->shared_gate_up, b->gate_up_bf16, rows, GLM_HIDDEN, b->shared_gate_up_rows, b->shared_gate_up_rows, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmSiluMulKernel<GLM_THREADS>), rows, GLM_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16, b->shared_intermediate, false);
	status = GlmLinear(b->intermediate_bf16, b->weights->shared_down, b->shared_out_bf16, rows, b->shared_intermediate, GLM_HIDDEN, GLM_HIDDEN, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmAddRowsKernel<GLM_THREADS>), dim3((GLM_HIDDEN + GLM_THREADS - 1u) / GLM_THREADS, rows), GLM_THREADS, 0, stream,
		b->hidden_bf16, b->shared_out_bf16, b->hidden_bf16, rows, GLM_HIDDEN);
	return GlmLaunched();
}

typedef struct GlmWave
{
	uint32_t first;
	uint32_t rows;
	uint32_t context;
	uint32_t ordered;
	uint32_t gather;
	uint32_t old_bound;
	uint32_t query_units;
	uint32_t indexing_units;
	uint32_t partial_units;
	uint64_t candidate_offset;
	SparkKvShardSectionLayout latent;
	uint32_t select_keep;
	uint32_t select_slice;
	uint32_t select_rows;
} GlmWave;

typedef struct GlmRegime
{
	const uint32_t *positions;
	uint32_t max_positions;
} GlmRegime;

template<class G>
struct GlmModel
{
	struct State
	{
		const SparkStageRunnerServices *services;
		uint32_t tp_degree;
		uint32_t tp_rank;
		uint32_t heads;
		uint32_t max_rows;
		uint32_t scatter_rows;
		uint32_t slots;
		uint32_t max_positions;
		uint32_t selection_rows;
		uint64_t topk_entries;
		uint64_t union_entries;
		uint64_t file_bytes;
		uint32_t entry_count;
		SparkGlm52StagePackHeader header;
		SparkGlm52StagePackEntry entries[GLM_MAX_TENSORS];
		SparkKvShard shard;
		SparkKvShardSectionLayout latent_max;
		uint64_t latent_bytes;
		uint64_t tail_slot;
		uint32_t query_units_max;
		uint32_t indexing_units_max;
		uint32_t partial_units_max;
		uint32_t select_rows_max;
		GlmLayer layers[GLM_LAYERS];
		uint64_t w1_offset[GLM_LAYERS];
		uint64_t w2_offset[GLM_LAYERS];
		const uint16_t *final_norm;
		void *arena;
		uint64_t arena_bytes;
		uint16_t *hidden;
		uint16_t *residual;
		uint16_t *normed;
		uint16_t *q_compressed;
		uint16_t *q;
		uint16_t *query_latent;
		uint16_t *query_rope;
		uint16_t *index_query;
		uint16_t *index_key;
		uint16_t *index_head;
		uint16_t *kv_slot;
		uint16_t *attention_latent;
		uint16_t *attention_value;
		uint16_t *attention_out;
		uint16_t *projection_local;
		uint16_t *projection_gather;
		uint16_t *gate_up;
		uint16_t *intermediate;
		uint16_t *expert_out;
		uint16_t *shared_out;
		float *router_logits;
		float *selection_scores;
		float *split_partials;
		uint32_t *selected;
		float *topk_values;
		uint32_t *topk_positions;
		uint32_t *route_expert;
		float *route_weight;
		uint32_t *route_source;
		uint32_t *route_packed;
		uint32_t *group_offset;
		uint32_t *tile_prefix_w1;
		uint32_t *tile_prefix_w2;
		uint32_t *positions;
		uint32_t *wave_slots;
		uint32_t *order;
		uint32_t *padded_positions;
		uint32_t *padded_slots;
		uint32_t *context_lengths;
		uint32_t *block_table;
		uint32_t *union_positions;
		uint8_t *union_masks;
		uint32_t *union_counts;
		uint16_t *shard_send;
		uint16_t *shard_received;
		float *shard_partials;
		float *shard_partials_received;
		uint32_t *index_local_selected;
		uint8_t *latent_pack;
		uint8_t *latent_pool;
		uint32_t *latent_remap;
		uint32_t *select_local;
		uint64_t *select_candidates;
		uint64_t *select_received;
		uint32_t *select_merged;
		uint32_t *rows_zero;
		uint32_t *plan;
		unsigned long long *digest;
		LmKvAccessError *kv_error;
		uint32_t *host_order;
		uint32_t *host_positions;
		uint32_t *host_slots;
		uint32_t *host_staging;
		LmKvAccessError *host_error;
		uint8_t *kv_pool;
		uint64_t kv_layer_stride;
		uint8_t *index_pool;
		uint64_t index_layer_stride;
		const uint32_t *page_table;
		uint32_t page_table_stride;
		uint32_t pool_page_count;
		uint32_t sequence_count;
		uint32_t kv_attached;
		uint64_t steps;
		uint64_t waves;
		uint64_t gather_waves;
	};

	static uint32_t Units(uint64_t bytes)
	{
		return (uint32_t)((bytes + GLM_UNIT_BYTES - 1u) / GLM_UNIT_BYTES);
	}

	static void ScatterSizes(const State *state, uint32_t rows, uint32_t *query_units, uint32_t *indexing_units, uint32_t *partial_units,
		uint64_t *candidate_offset)
	{
		const uint64_t query = (uint64_t)rows * state->heads * GLM_SLOT * sizeof(uint16_t);
		*candidate_offset = query;
		*query_units = Units(query);
		*indexing_units = Units(query + (uint64_t)rows * GLM_SELECTED * GLM_CANDIDATE_BYTES);
		*partial_units = Units((uint64_t)rows * state->heads * GLM_PARTIAL_FLOATS * sizeof(float));
	}

	static SparkKvShardSectionLayout GatherLayout(const State *state, uint32_t old_bound)
	{
		return SparkKvShardSectionLayoutBuild(SparkKvShardGatherKeys(state->shard, old_bound), GLM_SLOT_BYTES, GLM_UNIT_BYTES, GLM_SLOT_ALIGN,
			state->max_rows / GLM_SLOT_ALIGN * GLM_SLOT_ALIGN);
	}

	static uint32_t RowRegime(void *context, uint32_t row)
	{
		const GlmRegime *regime = (const GlmRegime *)context;
		const uint32_t tokens = regime->positions[row] + 1u;
		if ( tokens > GLM_SELECTED )
			return 2u;
		return LmLatentAttentionContextSplits(tokens, GLM_SPLIT_THRESHOLD) != 0u ? 1u : 0u;
	}

	static uint32_t SelectionRegime(void *context, uint32_t row)
	{
		const GlmRegime *regime = (const GlmRegime *)context;
		return regime->positions[row] + 1u > GLM_SELECTED ? 1u : 0u;
	}

	static uint32_t OldBound(const uint32_t *positions, uint32_t rows)
	{
		uint32_t row, first = positions[0], end = positions[0] + 1u, contiguous = 1u;
		for ( row = 1u; row < rows; row++ )
		{
			if ( positions[row] < positions[row - 1u] )
				return UINT32_MAX;
			if ( positions[row] - positions[row - 1u] > 1u )
				contiguous = 0u;
			end = positions[row] + 1u;
		}
		return contiguous != 0u ? first : end;
	}

	static SparkStatus Plan(State *state, uint64_t *bytes, uint8_t *base)
	{
		const uint64_t rows = state->max_rows, packed = rows * GLM_TOP_K, hidden = rows * GLM_HIDDEN * sizeof(uint16_t);
		const uint64_t width = GlmProjectionSliceWidth(state->tp_degree), heads = state->heads, degree = state->tp_degree;
		struct { void **target; uint64_t bytes; } plan[] =
		{
			{ (void **)&state->hidden, hidden },
			{ (void **)&state->residual, hidden },
			{ (void **)&state->normed, hidden },
			{ (void **)&state->q_compressed, rows * GLM_QUERY_A * sizeof(uint16_t) },
			{ (void **)&state->q, rows * heads * (GLM_NOPE + GLM_ROPE) * sizeof(uint16_t) },
			{ (void **)&state->query_latent, rows * heads * GLM_LATENT * sizeof(uint16_t) },
			{ (void **)&state->query_rope, rows * heads * GLM_ROPE * sizeof(uint16_t) },
			{ (void **)&state->index_query, rows * GLM_INDEX_QUERY * sizeof(uint16_t) },
			{ (void **)&state->index_key, rows * GLM_INDEX_DIM * sizeof(uint16_t) },
			{ (void **)&state->index_head, rows * GLM_INDEX_HEADS * sizeof(uint16_t) },
			{ (void **)&state->kv_slot, rows * GLM_SLOT * sizeof(uint16_t) },
			{ (void **)&state->attention_latent, rows * heads * GLM_LATENT * sizeof(uint16_t) },
			{ (void **)&state->attention_value, rows * heads * GLM_VALUE * sizeof(uint16_t) },
			{ (void **)&state->attention_out, hidden },
			{ (void **)&state->projection_local, rows * width * sizeof(uint16_t) },
			{ (void **)&state->projection_gather, degree * rows * width * sizeof(uint16_t) },
			{ (void **)&state->gate_up, rows * (GLM_TOP_K * 2u * GLM_EXPERT_INTER / degree) * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * (GLM_TOP_K * GLM_EXPERT_INTER / degree) * sizeof(uint16_t) },
			{ (void **)&state->expert_out, packed * GLM_HIDDEN * sizeof(uint16_t) },
			{ (void **)&state->shared_out, hidden },
			{ (void **)&state->router_logits, rows * GLM_EXPERTS * sizeof(float) },
			{ (void **)&state->selection_scores, (uint64_t)state->selection_rows * state->max_positions * sizeof(float) },
			{ (void **)&state->split_partials, rows * heads * GLM_SPLIT_PARTITIONS * GLM_PARTIAL_FLOATS * sizeof(float) },
			{ (void **)&state->selected, rows * GLM_SELECTED * sizeof(uint32_t) },
			{ (void **)&state->topk_values, state->topk_entries * sizeof(float) },
			{ (void **)&state->topk_positions, state->topk_entries * sizeof(uint32_t) },
			{ (void **)&state->route_expert, packed * sizeof(uint32_t) },
			{ (void **)&state->route_weight, packed * sizeof(float) },
			{ (void **)&state->route_source, packed * sizeof(uint32_t) },
			{ (void **)&state->route_packed, packed * sizeof(uint32_t) },
			{ (void **)&state->group_offset, (GLM_EXPERTS + 1u) * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_w1, (GLM_EXPERTS + 1u) * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_w2, (GLM_EXPERTS + 1u) * sizeof(uint32_t) },
			{ (void **)&state->positions, rows * sizeof(uint32_t) },
			{ (void **)&state->wave_slots, rows * sizeof(uint32_t) },
			{ (void **)&state->order, rows * sizeof(uint32_t) },
			{ (void **)&state->padded_positions, (uint64_t)state->select_rows_max * sizeof(uint32_t) },
			{ (void **)&state->padded_slots, (uint64_t)state->select_rows_max * sizeof(uint32_t) },
			{ (void **)&state->context_lengths, (uint64_t)state->slots * sizeof(uint32_t) },
			{ (void **)&state->block_table, (uint64_t)GLM_PREFILL_TABLE_BLOCKS(state->max_rows) * sizeof(uint32_t) },
			{ (void **)&state->union_positions, state->union_entries * sizeof(uint32_t) },
			{ (void **)&state->union_masks, state->union_entries * sizeof(uint8_t) },
			{ (void **)&state->union_counts, state->union_entries != 0u ? (uint64_t)GLM_PREFILL_TABLE_BLOCKS(state->max_rows) * sizeof(uint32_t) : 0u },
			{ (void **)&state->shard_send, (uint64_t)state->indexing_units_max * GLM_UNIT_BYTES },
			{ (void **)&state->shard_received, degree * state->indexing_units_max * GLM_UNIT_BYTES },
			{ (void **)&state->shard_partials, degree * state->partial_units_max * GLM_UNIT_BYTES },
			{ (void **)&state->shard_partials_received, degree * state->partial_units_max * GLM_UNIT_BYTES },
			{ (void **)&state->index_local_selected, (uint64_t)state->scatter_rows * GLM_SELECTED * sizeof(uint32_t) },
			{ (void **)&state->latent_pack, state->latent_bytes },
			{ (void **)&state->latent_pool, degree * state->latent_bytes + rows * GLM_SLOT_BYTES },
			{ (void **)&state->latent_remap, (uint64_t)state->max_positions * sizeof(uint32_t) },
			{ (void **)&state->select_local, (uint64_t)state->select_rows_max * GLM_SELECTED * sizeof(uint32_t) },
			{ (void **)&state->select_candidates, (uint64_t)state->select_rows_max * GLM_SELECTED * sizeof(uint64_t) },
			{ (void **)&state->select_received, (uint64_t)state->select_rows_max * GLM_SELECTED * sizeof(uint64_t) },
			{ (void **)&state->select_merged, (uint64_t)state->select_rows_max / degree * GLM_SELECTED * sizeof(uint32_t) },
			{ (void **)&state->rows_zero, rows * sizeof(uint32_t) },
			{ (void **)&state->plan, ((uint64_t)GLM_PLAN_ARRAYS * state->slots + GLM_PLAN_WORDS) * sizeof(uint32_t) },
			{ (void **)&state->digest, GLM_DIGEST_WORDS * sizeof(unsigned long long) },
			{ (void **)&state->kv_error, sizeof(LmKvAccessError) },
		};
		uint64_t cursor = 0u;
		for ( uint32_t index = 0u; index < sizeof(plan) / sizeof(plan[0]); ++index )
		{
			cursor = (cursor + 255u) & ~255ull;
			if ( base != 0 )
				*plan[index].target = plan[index].bytes != 0u ? base + cursor : 0;
			cursor += plan[index].bytes;
		}
		*bytes = cursor;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Round(State *state, cudaStream_t stream, uint32_t operation, uint32_t rows, uint32_t row_elements, const void *local, void *full)
	{
		return state->services->round(state->services->context, (void *)stream, operation, SPARK_STAGE_RUNNER_ROUND_ALL, rows, row_elements, local, full);
	}

	static uint32_t ContractBytes(uint8_t digest[SPARK_GLM52_STAGEPACK_SHA256_BYTES])
	{
		const char *text = G::Contract();
		if ( strlen(text) != 2u * SPARK_GLM52_STAGEPACK_SHA256_BYTES )
			return 0u;
		for ( uint32_t index = 0u; index < SPARK_GLM52_STAGEPACK_SHA256_BYTES; index++ )
		{
			uint32_t value = 0u;
			for ( uint32_t nibble = 0u; nibble < 2u; nibble++ )
			{
				const char c = text[2u * index + nibble];
				const uint32_t digit = c >= '0' && c <= '9' ? (uint32_t)(c - '0') : c >= 'a' && c <= 'f' ? (uint32_t)(c - 'a' + 10) : 16u;
				if ( digit > 15u )
					return 0u;
				value = value * 16u + digit;
			}
			digest[index] = (uint8_t)value;
		}
		return 1u;
	}

	static uint32_t BytesZero(const uint8_t *bytes, uint32_t count)
	{
		for ( uint32_t index = 0u; index < count; index++ )
			if ( bytes[index] != 0u )
				return 0u;
		return 1u;
	}

	static uint32_t RangesOverlap(uint64_t a, uint64_t a_bytes, uint64_t b, uint64_t b_bytes)
	{
		return a_bytes != 0u && b_bytes != 0u && a < b + b_bytes && b < a + a_bytes ? 1u : 0u;
	}

	static SparkStatus PackFail(const char *path, const char *reason)
	{
		fprintf(stderr, "%s: pack %s refused: %s\n", G::Tag(), path, reason);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}

	static SparkStatus PackRead(State *state, const char *path)
	{
		SparkGlm52StagePackHeader *header = &state->header;
		uint8_t contract[SPARK_GLM52_STAGEPACK_SHA256_BYTES];
		uint64_t layer_seen[GLM_LAYERS], global_seen = 0u, directory_end;
		FILE *file = fopen(path, "rb");
		long end;
		if ( file == 0 )
			return PackFail(path, "cannot open");
		memset(layer_seen, 0, sizeof(layer_seen));
		if ( fread(header, 1u, sizeof(*header), file) != sizeof(*header) || fseek(file, 0, SEEK_END) != 0 || (end = ftell(file)) <= 0 )
		{
			fclose(file);
			return PackFail(path, "header unreadable");
		}
		state->file_bytes = (uint64_t)end;
		if ( header->magic != SPARK_GLM52_STAGEPACK_MAGIC || header->format_version != SPARK_GLM52_STAGEPACK_FORMAT_VERSION ||
			header->header_bytes != SPARK_GLM52_STAGEPACK_HEADER_BYTES || header->directory_entry_bytes != SPARK_GLM52_STAGEPACK_ENTRY_BYTES ||
			header->codec_abi_version != SPARK_WEIGHT_CODEC_ABI_VERSION )
		{
			fclose(file);
			return PackFail(path, "not a version 3 glm52 stage pack");
		}
		if ( (header->flags & ~SPARK_GLM52_STAGEPACK_KNOWN_FLAGS) != 0u || (header->flags & SPARK_GLM52_STAGEPACK_FLAG_MTP) != 0u )
		{
			fclose(file);
			return PackFail(path, "pack carries an MTP layer or unknown flags; the TP16 stage runs the 78 decoder layers only");
		}
		if ( SparkGlm52StagePackHeaderTpDegree(header) != state->tp_degree || SparkGlm52StagePackHeaderTpRank(header) != state->tp_rank ||
			header->tensor_count == 0u || header->tensor_count > GLM_MAX_TENSORS || header->stage_count != 1u || header->stage_index != 0u ||
			header->first_layer_index != 0u || header->layer_count != GLM_LAYERS || header->total_layer_count != GLM_LAYERS ||
			header->hidden_dimension != GLM_HIDDEN || header->vocab_count != GLM_VOCAB || header->routed_expert_count != GLM_EXPERTS )
		{
			fprintf(stderr, "%s: pack %s is tp %u rank %u, %u tensors, stage %u/%u layers %u+%u; the runner is tp %u rank %u\n", G::Tag(), path,
				SparkGlm52StagePackHeaderTpDegree(header), SparkGlm52StagePackHeaderTpRank(header), header->tensor_count, header->stage_index,
				header->stage_count, header->first_layer_index, header->layer_count, state->tp_degree, state->tp_rank);
			fclose(file);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if ( header->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 || header->expert_weight_codec != GLM_EXPERT_CODEC ||
			header->kv_cache_codec != SPARK_WEIGHT_CODEC_BF16 )
		{
			fclose(file);
			return PackFail(path, "codecs are not bf16 linear, fp8 experts, bf16 kv");
		}
		if ( ContractBytes(contract) == 0u || memcmp(header->contract_sha256, contract, sizeof(contract)) != 0 ||
			header->model_revision[SPARK_GLM52_STAGEPACK_MODEL_REVISION_BYTES - 1u] != '\0' || strcmp(header->model_revision, G::Revision()) != 0 ||
			BytesZero(header->source_config_sha256, sizeof(header->source_config_sha256)) != 0u ||
			BytesZero(header->pack_recipe_sha256, sizeof(header->pack_recipe_sha256)) != 0u )
		{
			fclose(file);
			return PackFail(path, "contract, model revision or source identity does not match this build");
		}
		directory_end = header->directory_offset + (uint64_t)header->tensor_count * header->directory_entry_bytes;
		if ( header->file_bytes != state->file_bytes || header->directory_offset < header->header_bytes ||
			header->directory_offset % SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES != 0u || directory_end > state->file_bytes ||
			fseek(file, (long)header->directory_offset, SEEK_SET) != 0 ||
			fread(state->entries, header->directory_entry_bytes, header->tensor_count, file) != header->tensor_count )
		{
			fclose(file);
			return PackFail(path, "directory out of bounds or unreadable");
		}
		fclose(file);
		state->entry_count = header->tensor_count;
		for ( uint32_t index = 0u; index < state->entry_count; index++ )
		{
			const SparkGlm52StagePackEntry *entry = &state->entries[index];
			SparkGlm52StagePackTensorShape shape;
			uint64_t payload, scale;
			if ( SparkGlm52StagePackExpectedShape(entry->tensor_kind, entry->layer_index, GLM_EXPERT_CODEC, state->tp_degree, &shape) < 0 ||
				entry->payload_type != shape.payload_type || entry->weight_codec != shape.weight_codec || entry->scale_encoding != shape.scale_encoding ||
				entry->group_count != shape.group_count || entry->rows != shape.rows || entry->columns != shape.columns )
				return PackFail(path, "a tensor has an unexpected kind, layer or shape");
			payload = SparkGlm52StagePackExpectedPayloadBytes(&shape);
			scale = SparkGlm52StagePackExpectedScaleBytes(&shape);
			if ( payload == 0u || entry->payload_bytes != payload || entry->scale_bytes != scale ||
				entry->payload_offset % SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->payload_offset < directory_end ||
				entry->payload_offset > state->file_bytes || entry->payload_bytes > state->file_bytes - entry->payload_offset ||
				(scale == 0u && entry->scale_offset != 0u) ||
				(scale != 0u && (entry->scale_offset % SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->scale_offset < directory_end ||
					entry->scale_offset > state->file_bytes || entry->scale_bytes > state->file_bytes - entry->scale_offset)) )
				return PackFail(path, "a tensor's bytes or offsets do not match its shape");
			if ( entry->layer_index == SPARK_GLM52_STAGEPACK_GLOBAL_LAYER )
			{
				if ( (global_seen & (UINT64_C(1) << entry->tensor_kind)) != 0u )
					return PackFail(path, "duplicate global tensor");
				global_seen |= UINT64_C(1) << entry->tensor_kind;
			}
			else
			{
				if ( (layer_seen[entry->layer_index] & (UINT64_C(1) << entry->tensor_kind)) != 0u )
					return PackFail(path, "duplicate layer tensor");
				layer_seen[entry->layer_index] |= UINT64_C(1) << entry->tensor_kind;
			}
			for ( uint32_t other = 0u; other < index; other++ )
			{
				const SparkGlm52StagePackEntry *right = &state->entries[other];
				if ( RangesOverlap(entry->payload_offset, entry->payload_bytes, right->payload_offset, right->payload_bytes) != 0u ||
					RangesOverlap(entry->payload_offset, entry->payload_bytes, right->scale_offset, right->scale_bytes) != 0u ||
					RangesOverlap(entry->scale_offset, entry->scale_bytes, right->payload_offset, right->payload_bytes) != 0u ||
					RangesOverlap(entry->scale_offset, entry->scale_bytes, right->scale_offset, right->scale_bytes) != 0u )
					return PackFail(path, "two tensors overlap");
			}
			if ( RangesOverlap(entry->payload_offset, entry->payload_bytes, entry->scale_offset, entry->scale_bytes) != 0u )
				return PackFail(path, "a payload overlaps its scales");
		}
		if ( global_seen != ((UINT64_C(1) << SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING) | (UINT64_C(1) << SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM) |
			(UINT64_C(1) << SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD)) )
			return PackFail(path, "embedding, final norm or lm_head missing");
		for ( uint32_t layer = 0u; layer < GLM_LAYERS; layer++ )
		{
			uint64_t expected = 0u;
			SparkGlm52StagePackTensorShape shape;
			for ( uint32_t kind = SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM; kind < GLM_TENSOR_KINDS; kind++ )
				if ( SparkGlm52StagePackExpectedShape(kind, layer, GLM_EXPERT_CODEC, state->tp_degree, &shape) == 0 )
					expected |= UINT64_C(1) << kind;
			if ( layer_seen[layer] != expected )
			{
				fprintf(stderr, "%s: pack %s layer %u has tensor mask %llx, the model needs %llx\n", G::Tag(), path, layer,
					(unsigned long long)layer_seen[layer], (unsigned long long)expected);
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			}
		}
		return SPARK_STATUS_OK;
	}

	static const SparkGlm52StagePackEntry *Find(const State *state, uint32_t kind, uint32_t layer)
	{
		for ( uint32_t index = 0u; index < state->entry_count; index++ )
			if ( state->entries[index].tensor_kind == kind && state->entries[index].layer_index == layer )
				return &state->entries[index];
		return 0;
	}

	static void Close(void *model)
	{
		State *state = (State *)model;
		if ( state == 0 )
			return;
		if ( state->arena != 0 )
			(void)cudaFree(state->arena);
		if ( state->host_staging != 0 )
			(void)cudaFreeHost(state->host_staging);
		free(state);
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry)
	{
		const SparkStageRunnerConfiguration *configuration = request->configuration;
		const SparkGlm52StagePackEntry *embed, *norm, *head;
		State *state;
		SparkStatus status;
		if ( configuration->stage_count != 1u || configuration->tp_degree != G::kTpDegree || configuration->max_input_row_count == 0u ||
			configuration->max_input_row_count % G::kTpDegree != 0u || configuration->state_budget_bytes == 0u ||
			configuration->kv_pages_per_sequence == 0u )
		{
			fprintf(stderr, "%s: the stage needs one TP%u stage, a row capacity that is a multiple of %u, a state budget and KV pages\n", G::Tag(),
				G::kTpDegree, G::kTpDegree);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		state = (State *)calloc(1u, sizeof(*state));
		if ( state == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		*model = state;
		state->services = request->services;
		state->tp_degree = configuration->tp_degree;
		state->tp_rank = configuration->tp_rank;
		state->heads = GLM_HEADS / state->tp_degree;
		state->max_rows = configuration->max_input_row_count;
		state->scatter_rows = state->max_rows < GLM_SCATTER_ROWS ? state->max_rows : GLM_SCATTER_ROWS;
		state->slots = configuration->resident_sequence_capacity > configuration->max_active_sequence_count ?
			configuration->resident_sequence_capacity : configuration->max_active_sequence_count;
		if ( configuration->kv_pages_per_sequence == 0u || (uint64_t)configuration->kv_pages_per_sequence * GLM_PAGE_SLOTS > GLM_MAX_POSITIONS )
		{
			fprintf(stderr, "%s: %u KV pages per sequence give %llu positions; the model allows 1 to %u\n", G::Tag(), configuration->kv_pages_per_sequence,
				(unsigned long long)configuration->kv_pages_per_sequence * GLM_PAGE_SLOTS, GLM_MAX_POSITIONS);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		state->max_positions = configuration->kv_pages_per_sequence * GLM_PAGE_SLOTS;
		state->shard.degree = state->tp_degree;
		state->shard.rank = state->tp_rank;
		state->shard.grain = 1u;
		if ( state->heads != GLM_PREFILL_BLOCK_ROWS || SparkKvShardValid(state->shard, GLM_PAGE_SLOTS) == 0u )
		{
			fprintf(stderr, "%s: TP%u gives %u heads per rank; the packed prefill tables and the context split need 4\n", G::Tag(), state->tp_degree, state->heads);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		status = PackRead(state, configuration->rank_pack_path);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->selection_rows = LmTopkExactSelectionRows(state->max_rows, state->max_positions);
		state->topk_entries = LmTopkExactScratchEntries(state->selection_rows, state->max_positions, GLM_SELECTED);
		state->union_entries = state->max_positions > GLM_SELECTED ?
			(uint64_t)GLM_PREFILL_TABLE_BLOCKS(state->max_rows) * GLM_PREFILL_BLOCK_ROWS * GLM_SELECTED : 0u;
		{
			uint64_t candidate_offset;
			ScatterSizes(state, state->scatter_rows, &state->query_units_max, &state->indexing_units_max, &state->partial_units_max, &candidate_offset);
		}
		state->latent_max = GatherLayout(state, state->max_positions);
		if ( state->latent_max.chunks == 0u )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		state->latent_bytes = (uint64_t)state->latent_max.chunks * state->latent_max.chunk_bytes;
		state->tail_slot = state->tp_degree * state->latent_bytes / state->latent_max.slot_bytes;
		state->select_rows_max = (state->max_rows + state->tp_degree - 1u) / state->tp_degree * state->tp_degree;
		status = Plan(state, &state->arena_bytes, 0);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( state->arena_bytes > configuration->state_budget_bytes )
		{
			fprintf(stderr, "%s: rank state %llu bytes is over the %llu-byte budget\n", G::Tag(), (unsigned long long)state->arena_bytes,
				(unsigned long long)configuration->state_budget_bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		embed = Find(state, SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING, SPARK_GLM52_STAGEPACK_GLOBAL_LAYER);
		norm = Find(state, SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM, SPARK_GLM52_STAGEPACK_GLOBAL_LAYER);
		head = Find(state, SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD, SPARK_GLM52_STAGEPACK_GLOBAL_LAYER);
		memset(geometry, 0, sizeof(*geometry));
		geometry->hidden = GLM_HIDDEN;
		geometry->vocab = GLM_VOCAB;
		geometry->total_layers = GLM_LAYERS;
		geometry->first_layer = 0u;
		geometry->layer_count = GLM_LAYERS;
		geometry->rms_epsilon = GLM_RMS_EPSILON;
		geometry->pack_bytes = state->file_bytes;
		geometry->embed_offset = embed->payload_offset;
		geometry->embed_bytes = embed->payload_bytes;
		geometry->embed_rows = embed->rows;
		geometry->head_norm_offset = norm->payload_offset;
		geometry->head_norm_bytes = norm->payload_bytes;
		geometry->head_offset = head->payload_offset;
		geometry->head_bytes = head->payload_bytes;
		geometry->head_rows = head->rows;
		geometry->head_norm_f32 = 0u;
		geometry->head_prenormed = 1u;
		geometry->kv_layer_count = GLM_LAYERS;
		geometry->kv_layer_page_bytes = (uint64_t)GLM_PAGE_SLOTS * GLM_SLOT_BYTES;
		geometry->kv_second_layer_count = G::kIndexLayers;
		geometry->kv_second_layer_page_bytes = GlmIndexKv::kPageBytes;
		geometry->kv_shard = state->shard;
		geometry->experts = GLM_EXPERTS;
		geometry->top_k = GLM_TOP_K;
		for ( uint32_t layer = GLM_FIRST_ROUTED; layer < GLM_LAYERS; layer++ )
		{
			state->w1_offset[layer] = Find(state, SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE, layer)->payload_offset;
			state->w2_offset[layer] = Find(state, SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN, layer)->payload_offset;
		}
		geometry->expert_w1_offset = state->w1_offset;
		geometry->expert_w2_offset = state->w2_offset;
		return SPARK_STATUS_OK;
	}

	static SparkStatus ManifestCheck(void *model, const struct SparkWeightdManifest *manifest)
	{
		const State *state = (const State *)model;
		uint64_t expected = 0u;
		for ( uint32_t index = 0u; index < state->entry_count; index++ )
		{
			const SparkGlm52StagePackEntry *entry = &state->entries[index];
			SparkStatus status;
			if ( entry->tensor_kind != SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE && entry->tensor_kind != SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN )
				continue;
			status = SparkExpertPlanesCovered(manifest, entry->tensor_kind, entry->layer_index, entry->weight_codec, entry->scale_encoding,
				entry->group_count, entry->payload_offset, entry->payload_bytes, entry->scale_offset, entry->scale_bytes, &expected);
			if ( status != SPARK_STATUS_OK )
			{
				fprintf(stderr, "%s: the .experts manifest does not cover layer %u kind %u\n", G::Tag(), entry->layer_index, entry->tensor_kind);
				return status;
			}
		}
		if ( expected != manifest->range_count )
		{
			fprintf(stderr, "%s: the .experts manifest has %u ranges, the pack's experts need %llu\n", G::Tag(), manifest->range_count,
				(unsigned long long)expected);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Slice(SparkWeightdLazyPack *lazy_pack, const SparkGlm52StagePackEntry *entry, const void **field)
	{
		const void *bound = 0;
		if ( SparkWeightdLazyPackSlice(lazy_pack, entry->payload_offset, entry->payload_bytes, &bound) != SPARK_STATUS_OK || bound == 0 )
		{
			fprintf(stderr, "%s: weightd has no resident bytes for tensor kind %u layer %u\n", G::Tag(), entry->tensor_kind, entry->layer_index);
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		}
		*field = bound;
		return SPARK_STATUS_OK;
	}

	static SparkStatus BindEntry(State *state, SparkWeightdLazyPack *lazy_pack, const SparkGlm52StagePackEntry *entry)
	{
		GlmLayer *layer = entry->layer_index != SPARK_GLM52_STAGEPACK_GLOBAL_LAYER ? &state->layers[entry->layer_index] : 0;
		const void *pointer = 0;
		if ( entry->tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE )
		{
			layer->up_gate_payload = entry->payload_offset;
			layer->up_gate_scale = entry->scale_offset;
			return SPARK_STATUS_OK;
		}
		if ( entry->tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN )
		{
			layer->down_payload = entry->payload_offset;
			layer->down_scale = entry->scale_offset;
			return SPARK_STATUS_OK;
		}
		if ( entry->tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING || entry->tensor_kind == SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD )
			return SPARK_STATUS_OK;
		if ( Slice(lazy_pack, entry, &pointer) != SPARK_STATUS_OK )
			return SPARK_STATUS_PARSE_ERROR;
		switch ( entry->tensor_kind )
		{
		case SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM: layer->attn_norm = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_Q_A: layer->q_a = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_Q_A_NORM: layer->q_a_norm = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_Q_B: layer->q_b = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_KV_A: layer->kv_a = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_KV_A_NORM: layer->kv_a_norm = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED: layer->kv_b_key = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_KV_B_VALUE: layer->kv_b_value = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_OUTPUT: layer->o_proj = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_POST_ATTN_NORM: layer->post_norm = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q: layer->index_q = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_K: layer->index_k = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_HEAD: layer->index_head = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT: layer->index_norm_weight = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_BIAS: layer->index_norm_bias = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP: layer->dense_gate_up = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN: layer->dense_down = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_ROUTER: layer->router = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_ROUTER_CORRECTION: layer->router_bias = (const float *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_GATE_UP: layer->shared_gate_up = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN: layer->shared_down = (const uint16_t *)pointer; break;
		default: SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Bind(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		uint32_t ordinal = 0u;
		(void)geometry;
		for ( uint32_t index = 0u; index < state->entry_count; index++ )
			if ( BindEntry(state, lazy_pack, &state->entries[index]) != SPARK_STATUS_OK )
				return SPARK_STATUS_PARSE_ERROR;
		for ( uint32_t layer = 0u; layer < GLM_LAYERS; layer++ )
			state->layers[layer].index_ordinal = GlmFullIndexer(layer) != 0u ? ordinal++ : UINT32_MAX;
		if ( ordinal != G::kIndexLayers || state->final_norm == 0 )
		{
			fprintf(stderr, "%s: %u indexer layers bound, the model has %u\n", G::Tag(), ordinal, G::kIndexLayers);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if ( cudaMalloc(&state->arena, state->arena_bytes) != cudaSuccess )
		{
			state->arena = 0;
			fprintf(stderr, "%s: %llu bytes of rank state do not fit\n", G::Tag(), (unsigned long long)state->arena_bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		if ( cudaMemset(state->arena, 0, state->arena_bytes) != cudaSuccess || Plan(state, &state->arena_bytes, (uint8_t *)state->arena) != SPARK_STATUS_OK ||
			cudaHostAlloc((void **)&state->host_staging, 3u * (uint64_t)state->max_rows * sizeof(uint32_t) + sizeof(LmKvAccessError) + 64u,
				cudaHostAllocPortable) != cudaSuccess )
		{
			state->host_staging = 0;
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		state->host_order = state->host_staging;
		state->host_positions = state->host_order + state->max_rows;
		state->host_slots = state->host_positions + state->max_rows;
		state->host_error = (LmKvAccessError *)(void *)(state->host_slots + state->max_rows);
		fprintf(stderr, "%s rank=%u arena_bytes=%llu rows=%u slots=%u positions=%u scatter_units=%u/%u/%u gather_latent=%u x %u units select_rows=%u\n",
			G::Tag(), state->tp_rank, (unsigned long long)state->arena_bytes, state->max_rows, state->slots, state->max_positions,
			state->query_units_max, state->indexing_units_max, state->partial_units_max, state->latent_max.chunks, state->latent_max.chunk_units,
			state->select_rows_max);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AttachKv(void *model, const SparkStageRunnerKv *kv)
	{
		State *state = (State *)model;
		if ( kv == 0 || kv->layer_count != GLM_LAYERS || kv->layer_page_bytes != (uint64_t)GLM_PAGE_SLOTS * GLM_SLOT_BYTES ||
			kv->layer_stride_bytes != (uint64_t)GLM_PAGE_SLOTS * GLM_SLOT_BYTES / state->tp_degree || kv->second_pool == 0 ||
			kv->second_layer_count != G::kIndexLayers || kv->second_layer_page_bytes != GlmIndexKv::kPageBytes || kv->sequence_count > state->slots ||
			kv->context_shard.degree != state->shard.degree || kv->context_shard.rank != state->shard.rank || kv->context_shard.grain != state->shard.grain )
		{
			fprintf(stderr, "%s: KV attach refused: the binding is not the page-major latent region plus the layer-major index region of this rank\n", G::Tag());
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		state->kv_pool = kv->pool;
		state->kv_layer_stride = kv->layer_stride_bytes;
		state->index_pool = kv->second_pool;
		state->index_layer_stride = kv->second_layer_stride_bytes;
		state->page_table = kv->page_table;
		state->page_table_stride = kv->page_table_stride;
		state->pool_page_count = kv->pool_page_count;
		state->sequence_count = kv->sequence_count;
		state->kv_attached = 1u;
		return SPARK_STATUS_OK;
	}

	static void View(const State *state, LmKvView *view, uint8_t *pool)
	{
		view->pool = pool;
		view->page_table = state->page_table;
		view->page_table_stride = state->page_table_stride;
		view->sequence_count = state->sequence_count;
		view->pool_page_count = state->pool_page_count;
		view->access_error = state->kv_error;
	}

	static void BindLayer(const State *state, const GlmWave *wave, uint32_t layer, uint16_t *hidden, const uint32_t *positions, const uint32_t *slots,
		GlmBuffers *b)
	{
		const GlmLayer *weights = &state->layers[layer];
		memset(b, 0, sizeof(*b));
		b->tp_degree = state->tp_degree;
		b->tp_rank = state->tp_rank;
		b->attn_heads = state->heads;
		b->q_b_rows = state->heads * (GLM_NOPE + GLM_ROPE);
		b->attn_output_columns = state->heads * GLM_VALUE;
		b->dense_gate_up_rows = 2u * GLM_DENSE_INTER / state->tp_degree;
		b->dense_intermediate = GLM_DENSE_INTER / state->tp_degree;
		b->expert_w1_rows = 2u * GLM_EXPERT_INTER / state->tp_degree;
		b->expert_intermediate = GLM_EXPERT_INTER / state->tp_degree;
		b->shared_gate_up_rows = 2u * GLM_EXPERT_INTER / state->tp_degree;
		b->shared_intermediate = GLM_EXPERT_INTER / state->tp_degree;
		b->weights = weights;
		b->hidden_bf16 = hidden;
		b->residual_bf16 = state->residual;
		b->normed_bf16 = state->normed;
		b->q_compressed_bf16 = state->q_compressed;
		b->q_bf16 = state->q;
		b->query_latent_bf16 = state->query_latent;
		b->query_rope_bf16 = state->query_rope;
		b->index_query_bf16 = state->index_query;
		b->index_key_bf16 = state->index_key;
		b->index_head_weight_bf16 = state->index_head;
		b->kv_slot_bf16 = state->kv_slot;
		b->attention_latent_bf16 = state->attention_latent;
		b->attention_value_bf16 = state->attention_value;
		b->attention_out_bf16 = state->attention_out;
		b->gate_up_bf16 = state->gate_up;
		b->intermediate_bf16 = state->intermediate;
		b->expert_out_bf16 = state->expert_out;
		b->shared_out_bf16 = state->shared_out;
		b->router_logits = state->router_logits;
		b->selection_scores = state->selection_scores;
		b->route_expert = state->route_expert;
		b->route_weight = state->route_weight;
		b->route_source_token = state->route_source;
		b->route_packed_row = state->route_packed;
		b->group_row_offset = state->group_offset;
		b->group_tile_prefix_w1 = state->tile_prefix_w1;
		b->group_tile_prefix_w2 = state->tile_prefix_w2;
		b->sequence_of_row = slots;
		b->context_length = state->context_lengths;
		b->positions = positions;
		b->row_positions = positions;
		b->selected_positions = state->selected;
		b->selected_position_count = GLM_SELECTED;
		b->selection_rows = state->selection_rows;
		b->prefill_union_positions = state->union_positions;
		b->prefill_union_masks = state->union_masks;
		b->prefill_union_counts = state->union_counts;
		b->prefill_union_entries = state->union_entries;
		b->topk_scratch_values = state->topk_values;
		b->topk_scratch_positions = state->topk_positions;
		b->topk_scratch_entries = state->topk_entries;
		b->attention_split_partials = state->split_partials;
		b->attention_split_partial_blocks = (uint64_t)state->max_rows * state->heads * GLM_SPLIT_PARTITIONS;
		b->decode_split_context_threshold = GLM_SPLIT_THRESHOLD;
		b->single_sequence_rows = wave->ordered;
		b->prefill_block_table = wave->ordered != 0u ? state->block_table : 0;
		b->prefill_table_blocks = wave->ordered != 0u ? GLM_PREFILL_TABLE_BLOCKS(wave->rows) : 0u;
		b->projection_local_bf16 = state->projection_local;
		b->projection_gather_bf16 = state->projection_gather;
		View(state, &b->cache, state->kv_pool + (uint64_t)layer * state->kv_layer_stride);
		if ( weights->index_ordinal != UINT32_MAX )
			View(state, &b->index_cache, state->index_pool + (uint64_t)weights->index_ordinal * state->index_layer_stride);
		b->kv_shard = state->shard;
		b->cache_shard.pages = b->cache;
		b->cache_shard.shard = state->shard;
		b->index_shard.pages = b->index_cache;
		b->index_shard.shard = state->shard;
		if ( wave->gather != 0u )
		{
			b->kv_gather_latent_pack = state->latent_pack;
			b->kv_gather_latent_tail = state->latent_pool + state->tail_slot * GLM_SLOT_BYTES;
			b->kv_select_local = state->select_local;
			b->kv_select_candidates = (uint2 *)state->select_candidates;
			b->kv_select_keep = wave->select_keep;
			b->kv_select_rows = wave->select_rows;
			b->kv_gather_sequences = state->plan;
			b->kv_gather_offset = state->plan + state->slots;
			b->kv_gather_old_bound = state->plan + 2u * state->slots;
			b->kv_gather_rows_zero = state->rows_zero;
			b->kv_gather_context_word = state->plan + 3u * state->slots + 1u;
			b->kv_gather_digest = state->digest;
			b->kv_gather_old_keys = (uint32_t)wave->latent.keys;
			b->cache_gathered.pool = state->latent_pool;
			b->cache_gathered.page_table = state->latent_remap;
			b->cache_gathered.page_table_stride = state->max_positions;
			b->cache_gathered.sequence_count = 1u;
			b->cache_gathered.pool_page_count = (uint32_t)(state->tail_slot + state->max_rows);
			b->cache_gathered.access_error = state->kv_error;
			return;
		}
		b->shard_send_bf16 = state->shard_send;
		b->shard_candidate_offset = wave->candidate_offset;
		b->shard_received_bf16 = state->shard_received;
		b->shard_rank_stride_bytes = (uint64_t)(GlmShardIndexing(layer, wave->context) != 0u ? wave->indexing_units : wave->query_units) * GLM_UNIT_BYTES;
		b->shard_partials = state->shard_partials;
		b->shard_partials_received = state->shard_partials_received;
		b->shard_partial_stride = (uint64_t)wave->partial_units * (GLM_UNIT_BYTES / sizeof(float));
		b->index_local_selected = state->index_local_selected;
	}

	static SparkStatus Fail(const char *site, uint32_t layer, int32_t code)
	{
		fprintf(stderr, "%s: %s failed layer=%u code=%d cuda=%s\n", G::Tag(), site, layer, code, cudaGetErrorString(cudaGetLastError()));
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}

	static SparkStatus GatherExchange(State *state, const GlmWave *wave, GlmBuffers *b, uint32_t layer, cudaStream_t stream)
	{
		SparkStatus status = SPARK_STATUS_OK;
		if ( GlmShardIndexing(layer, wave->context) != 0u )
		{
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, 2u * wave->select_slice, 2u * wave->select_keep,
				state->select_candidates, state->select_received);
			if ( status != SPARK_STATUS_OK )
				return status;
			if ( LmIndexShardCandidateScatterLaunch<GLM_THREADS>((const uint2 *)state->select_received, (uint64_t)wave->select_slice * wave->select_keep,
					state->tp_degree, wave->select_keep, wave->select_slice, wave->context, state->selection_scores, stream) != cudaSuccess ||
				LmTopkExactLaunch<GLM_THREADS>(state->selection_scores, wave->select_slice, wave->context, GLM_SELECTED, LM_TOPK_EXACT_CHUNK,
					LM_TOPK_EXACT_CHUNKED_ROWS, state->topk_values, state->topk_positions, state->topk_entries, state->select_merged, stream) != cudaSuccess )
				return Fail("kv select merge", layer, LM_LAUNCH_ERR_LAUNCH);
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, wave->select_slice, 2u * GLM_SELECTED, state->select_merged,
				state->selected);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		for ( uint32_t chunk = 0u; status == SPARK_STATUS_OK && chunk < wave->latent.chunks; chunk++ )
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, wave->latent.chunk_units, 0u,
				state->latent_pack + (uint64_t)chunk * wave->latent.chunk_bytes, state->latent_pool + (uint64_t)chunk * state->tp_degree * wave->latent.chunk_bytes);
		(void)b;
		return status;
	}

	static SparkStatus Layer(State *state, const GlmWave *wave, uint32_t layer, uint16_t *hidden, const uint32_t *positions, const uint32_t *slots,
		uint32_t sms, cudaStream_t stream)
	{
		const uint32_t rows = wave->rows, packed = rows * GLM_TOP_K;
		GlmBuffers b;
		SparkStatus status;
		int32_t code;
		BindLayer(state, wave, layer, hidden, positions, slots, &b);
		code = GlmAttentionProject(&b, rows, sms, stream);
		if ( code != LM_LAUNCH_OK )
			return Fail("attention projection", layer, code);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, GlmProjectionSliceWidth(state->tp_degree),
			state->projection_local, state->projection_gather);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( wave->gather != 0u )
		{
			code = GlmShardGatherPre(&b, rows, wave->context, layer, sms, stream);
			if ( code != LM_LAUNCH_OK )
				return Fail("gather pre", layer, code);
			status = GatherExchange(state, wave, &b, layer, stream);
			if ( status != SPARK_STATUS_OK )
				return status;
			code = GlmShardGatherPost(&b, rows, wave->context, sms, stream);
			if ( code != LM_LAUNCH_OK )
				return Fail("gather post", layer, code);
		}
		else
		{
			code = GlmShardScatterPre(&b, rows, wave->context, layer, sms, stream);
			if ( code != LM_LAUNCH_OK )
				return Fail("scatter pre", layer, code);
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER,
				GlmShardIndexing(layer, wave->context) != 0u ? wave->indexing_units : wave->query_units, 0u, state->shard_send, state->shard_received);
			if ( status != SPARK_STATUS_OK )
				return status;
			code = GlmShardScatterMid(&b, rows, wave->context, layer, stream);
			if ( code != LM_LAUNCH_OK )
				return Fail("scatter mid", layer, code);
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, wave->partial_units, 0u, state->shard_partials,
				state->shard_partials_received);
			if ( status != SPARK_STATUS_OK )
				return status;
			code = GlmShardScatterPost(&b, rows, sms, stream);
			if ( code != LM_LAUNCH_OK )
				return Fail("scatter post", layer, code);
		}
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->attention_out, state->attention_out);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( layer < GLM_FIRST_ROUTED )
			code = GlmDenseMlp(&b, rows, sms, stream);
		else
		{
			const void *w1 = 0, *w2 = 0;
			code = GlmMoeRoute(&b, rows, packed, sms, stream);
			if ( code != LM_LAUNCH_OK )
				return Fail("moe route", layer, code);
			status = state->services->expert_weights(state->services->context, layer, state->group_offset, packed, &w1, &w2);
			if ( status != SPARK_STATUS_OK || w1 == 0 || w2 == 0 )
			{
				fprintf(stderr, "%s: expert weights for layer %u are not available status=%d\n", G::Tag(), layer, (int)status);
				return status != SPARK_STATUS_OK ? status : SPARK_STATUS_INTERNAL_ERROR;
			}
			b.expert_w1_weight = w1;
			b.expert_w1_scale = (const uint8_t *)w1 - state->w1_offset[layer] + state->layers[layer].up_gate_scale;
			b.expert_w2_weight = w2;
			b.expert_w2_scale = (const uint8_t *)w2 - state->w2_offset[layer] + state->layers[layer].down_scale;
			code = GlmMoeExperts(&b, rows, packed, sms, stream);
		}
		if ( code != LM_LAUNCH_OK )
			return Fail("mlp", layer, code);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, hidden, hidden);
		if ( status == SPARK_STATUS_OK && state->services->layer_done != 0 )
			state->services->layer_done(state->services->context, layer);
		return status;
	}

	static void Measure(const State *state, GlmWave *wave, const uint32_t *host_positions)
	{
		uint32_t row;
		if ( wave->ordered == 0u )
		{
			wave->context = state->max_positions;
			ScatterSizes(state, wave->rows, &wave->query_units, &wave->indexing_units, &wave->partial_units, &wave->candidate_offset);
			return;
		}
		wave->context = 0u;
		for ( row = 0u; row < wave->rows; row++ )
			if ( host_positions[wave->first + row] + 1u > wave->context )
				wave->context = host_positions[wave->first + row] + 1u;
		wave->gather = wave->rows > LM_SKINNY_ROWS ? 1u : 0u;
		if ( wave->gather != 0u )
		{
			const uint32_t old_bound = OldBound(host_positions + wave->first, wave->rows);
			wave->old_bound = old_bound != UINT32_MAX ? old_bound : wave->context;
			wave->latent = GatherLayout(state, wave->old_bound);
			wave->select_keep = GlmShardKeep(state->shard, wave->context);
			wave->select_slice = (wave->rows + state->tp_degree - 1u) / state->tp_degree;
			wave->select_rows = wave->select_slice * state->tp_degree;
			return;
		}
		ScatterSizes(state, wave->rows, &wave->query_units, &wave->indexing_units, &wave->partial_units, &wave->candidate_offset);
	}

	static SparkStatus Wave(State *state, const GlmWave *wave, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = wave->rows;
		uint16_t *hidden;
		const uint32_t *positions, *slots;
		SparkStatus status = SPARK_STATUS_OK;
		if ( wave->ordered != 0u )
		{
			hidden = state->hidden;
			positions = state->positions + wave->first;
			slots = state->wave_slots + wave->first;
			GlmRowsGatherKernel<<<rows, GLM_THREADS, 0, stream>>>(state->order + wave->first, step->hidden_bf16, hidden, GLM_HIDDEN);
		}
		else
		{
			hidden = step->hidden_bf16 + (uint64_t)wave->first * GLM_HIDDEN;
			positions = step->positions + wave->first;
			slots = step->sequence_of_row + wave->first;
		}
		if ( cudaMemsetAsync(state->residual, 0, (size_t)rows * GLM_HIDDEN * sizeof(uint16_t), stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		GlmWaveContextKernel<<<1u, GLM_THREADS, 0, stream>>>(slots, positions, state->context_lengths, rows);
		if ( wave->gather != 0u )
		{
			GlmGatherPlanKernel<<<1u, 1u, 0, stream>>>(slots, positions, rows, state->slots, state->plan, state->kv_error);
			if ( cudaMemsetAsync(state->digest, 0, GLM_DIGESTS * sizeof(unsigned long long), stream) != cudaSuccess ||
				LmKvShardRemapLaunch<GLM_THREADS>(state->shard, wave->latent, state->plan + 3u * state->slots, state->max_positions, state->latent_remap,
					stream) != cudaSuccess ||
				LmKvRowsRemapLaunch<GLM_THREADS>(positions, rows, (uint32_t)state->tail_slot, state->max_positions, state->latent_remap, state->kv_error,
					stream) != cudaSuccess )
				return Fail("gather plan", 0u, LM_LAUNCH_ERR_LAUNCH);
		}
		if ( wave->ordered != 0u )
		{
			LmKvView cache;
			View(state, &cache, state->kv_pool);
			if ( LmPrefillBlockTableLaunch<GLM_PREFILL_BLOCK_ROWS>(cache, slots, rows, state->block_table, GLM_PREFILL_TABLE_BLOCKS(rows), 0, GLM_PREFILL_SPANS,
					stream) != cudaSuccess )
				return Fail("prefill block table", 0u, LM_LAUNCH_ERR_LAUNCH);
		}
		if ( wave->gather != 0u && wave->select_rows > rows )
		{
			GlmPadRowsKernel<<<1u, GLM_THREADS, 0, stream>>>(positions, slots, state->padded_positions, state->padded_slots, rows, wave->select_rows);
			positions = state->padded_positions;
			slots = state->padded_slots;
		}
		if ( cudaPeekAtLastError() != cudaSuccess )
			return Fail("wave metadata", 0u, LM_LAUNCH_ERR_LAUNCH);
		for ( uint32_t layer = 0u; layer < GLM_LAYERS && status == SPARK_STATUS_OK; layer++ )
			status = Layer(state, wave, layer, hidden, positions, slots, step->multiprocessors, stream);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( wave->gather != 0u )
		{
			if ( LmRowDigestFinishLaunch(state->digest, GLM_DIGESTS, state->digest + GLM_DIGESTS, stream) != cudaSuccess )
				return Fail("kv digest", GLM_LAYERS, LM_LAUNCH_ERR_LAUNCH);
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64, 2u * GLM_DIGESTS, 0u, state->digest + GLM_DIGESTS,
				state->digest + GLM_DIGESTS);
			if ( status != SPARK_STATUS_OK )
				return status;
			if ( LmRowDigestCheckLaunch(state->digest + GLM_DIGESTS, GLM_DIGESTS, state->kv_error, stream) != cudaSuccess )
				return Fail("kv digest check", GLM_LAYERS, LM_LAUNCH_ERR_LAUNCH);
		}
		LmFusedResidualRmsNormLaunch<GLM_THREADS,uint16_t>(rows, stream,
			hidden, state->residual, state->final_norm, 0, state->normed, GLM_HIDDEN, GLM_HIDDEN, GLM_RMS_EPSILON);
		if ( wave->ordered != 0u )
			GlmRowsScatterKernel<<<rows, GLM_THREADS, 0, stream>>>(state->order + wave->first, state->normed, step->hidden_bf16, GLM_HIDDEN);
		else if ( cudaMemcpyAsync(hidden, state->normed, (size_t)rows * GLM_HIDDEN * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( cudaPeekAtLastError() != cudaSuccess )
			return Fail("final norm", GLM_LAYERS, LM_LAUNCH_ERR_LAUNCH);
		state->waves++;
		state->gather_waves += wave->gather;
		return SPARK_STATUS_OK;
	}

	static uint32_t KvErrorReport(const State *state)
	{
		const LmKvAccessError *error = state->host_error;
		if ( error->error_code == 0u )
			return 0u;
		fprintf(stderr, "%s: KV access error code=%u kind=%u row=%u sequence=%u position=%u page=%u\n", G::Tag(), error->error_code,
			error->access_kind, error->row, error->sequence, error->position, error->page);
		return 1u;
	}

	static void CUDART_CB KvErrorCheck(void *user)
	{
		const State *state = (const State *)user;
		if ( KvErrorReport(state) != 0u )
			state->services->fail(state->services->context, SPARK_STATUS_INTERNAL_ERROR);
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *stream_void)
	{
		State *state = (State *)model;
		cudaStream_t stream = (cudaStream_t)stream_void;
		const uint32_t rows = step->rows;
		const uint32_t ordered = step->last_rows_only != 0u || rows != step->sequences ? 1u : 0u;
		GlmRegime regime;
		GlmWave wave;
		SparkStatus status = SPARK_STATUS_OK;
		if ( rows == 0u || rows > state->max_rows || step->sequences > state->slots || state->kv_attached == 0u || step->positions == 0 ||
			step->sequence_of_row == 0 || (ordered != 0u && (step->host_positions == 0 || step->host_sequence_of_row == 0)) )
		{
			fprintf(stderr, "%s: step refused rows=%u sequences=%u kv=%u ordered=%u host_rows=%s\n", G::Tag(), rows, step->sequences, state->kv_attached,
				ordered, step->host_positions != 0 && step->host_sequence_of_row != 0 ? "yes" : "missing");
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		if ( cudaMemsetAsync(state->kv_error, 0, sizeof(LmKvAccessError), stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( ordered != 0u )
		{
			if ( SparkRowLayoutLaneMajorOrder(rows, step->host_sequence_of_row, state->host_order) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			for ( uint32_t row = 0u; row < rows; row++ )
			{
				state->host_positions[row] = step->host_positions[state->host_order[row]];
				state->host_slots[row] = step->host_sequence_of_row[state->host_order[row]];
			}
			if ( cudaMemcpyAsync(state->order, state->host_order, (size_t)rows * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
				cudaMemcpyAsync(state->positions, state->host_positions, (size_t)rows * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
				cudaMemcpyAsync(state->wave_slots, state->host_slots, (size_t)rows * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			regime.positions = state->host_positions;
			regime.max_positions = state->max_positions;
		}
		for ( uint32_t first = 0u; first < rows && status == SPARK_STATUS_OK; first += wave.rows )
		{
			memset(&wave, 0, sizeof(wave));
			wave.first = first;
			wave.ordered = ordered;
			if ( ordered != 0u )
			{
				wave.rows = SparkRowLayoutPackedSpanWaveRowCount(first, rows, state->host_slots, SelectionRegime, &regime, state->max_rows, 1u);
				if ( wave.rows <= LM_SKINNY_ROWS )
					wave.rows = SparkRowLayoutPackedSpanWaveRowCount(first, rows, state->host_slots, RowRegime, &regime, state->max_rows, 1u);
			}
			else
				wave.rows = rows - first < state->scatter_rows ? rows - first : state->scatter_rows;
			if ( wave.rows == 0u )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			Measure(state, &wave, state->host_positions);
			status = Wave(state, &wave, step, stream);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( cudaMemcpyAsync(state->host_error, state->kv_error, sizeof(LmKvAccessError), cudaMemcpyDeviceToHost, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		state->steps++;
		if ( ordered == 0u )
		{
			if ( cudaLaunchHostFunc(stream, KvErrorCheck, state) != cudaSuccess )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			return SPARK_STATUS_OK;
		}
		if ( cudaStreamSynchronize(stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( KvErrorReport(state) != 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static uint32_t GraphKey(void *model, const SparkStageRunnerStep *step)
	{
		(void)model;
		return step->last_rows_only == 0u && step->rows == step->sequences ? 1u : 0u;
	}

	static SparkStatus RecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream)
	{
		(void)model; (void)to_buffer; (void)slot; (void)buffer; (void)bytes; (void)stream;
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}

	static SparkStatus ResetSlot(void *model, uint32_t slot, void *stream)
	{
		(void)model; (void)slot; (void)stream;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Sideband(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows, void *stream)
	{
		(void)model; (void)to_buffer; (void)hidden; (void)buffer; (void)bytes; (void)rows; (void)stream;
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}

	static const void *ProbeBuffers(void *model)
	{
		return model;
	}

	static void Report(void *model, uint32_t rank)
	{
		const State *state = (const State *)model;
		fprintf(stderr, "%s rank=%u steps=%llu waves=%llu gather_waves=%llu arena_bytes=%llu\n", G::ReportTag(), rank, (unsigned long long)state->steps,
			(unsigned long long)state->waves, (unsigned long long)state->gather_waves, (unsigned long long)state->arena_bytes);
	}

	static const SparkStageRunnerModelInterface *Interface(void)
	{
		static const SparkStageRunnerModelInterface interface_table =
		{
			SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
			G::Tag(),
			G::WeightdModel(),
			G::Revision(),
			Open,
			ManifestCheck,
			Bind,
			Close,
			Step,
			Sideband,
			AttachKv,
			RecurrentCopy,
			ResetSlot,
			ProbeBuffers,
			Report,
			GraphKey
		};
		return &interface_table;
	}
};
