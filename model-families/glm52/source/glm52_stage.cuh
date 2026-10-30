#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

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
#include "sparkpipe/spark_glm52_family.h"
#include "sparkpipe/spark_glm52_stagepack.h"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_prefill.cuh"
#include "inference/kernels/attn_rope_heads.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_split.h"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/index_score.cuh"
#include "inference/kernels/index_select.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/norm.cuh"
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
#include "glm52_row_shard.cuh"

#define GLM_THREADS 256u
#define GLM_PAGE_SLOTS 64u
#define GLM_GRANULE 8u
#define GLM_TILE_N 128u
#define GLM_TILE_STAGES 2u
#define GLM_TILE_WARPS 8u
#define GLM_STREAM_EXPERT_ROWS 256u
#define GLM_LANE_ROWS 1024u
#define GLM_CANDIDATE_BYTES 8u
#define GLM_SCATTER_ROWS 256u
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

struct GlmDefaults
{
	static constexpr uint32_t kQueryGate = 0u;
	static constexpr bool kQueryConcat = false;
	static constexpr uint32_t kNormGateRank = 0u;
	static constexpr uint32_t kExpertGroups = 1u;
	static constexpr uint32_t kTopGroups = 1u;
	static constexpr float kRopeFactor = 1.0f;
	static constexpr float kRopeBetaFast = 0.0f;
	static constexpr float kRopeBetaSlow = 0.0f;
	static constexpr uint32_t kRopeOriginalPositions = 0u;
};

template<class G>
struct GlmShape : G
{
	static constexpr uint32_t kSlot = G::kLatent + G::kRope;
	static constexpr uint32_t kSlotBytes = kSlot * 2u;
	static constexpr uint32_t kIndexQuery = G::kIndexHeads * G::kIndexDim;
	static constexpr uint32_t kGatherWidth = G::kQueryA + kSlot;
	static constexpr uint64_t kUnitBytes = (uint64_t)G::kHidden * 2u;
	static constexpr uint32_t kSlotAlign = (uint32_t)(kSlotBytes / std::gcd(kUnitBytes, (uint64_t)kSlotBytes));
	static constexpr uint32_t kPartialFloats = G::kLatent + 2u;
	static constexpr uint32_t kQueryHead = G::kNope + G::kRope + G::kQueryGate;
	static constexpr uint32_t kQueryInput = G::kQueryConcat ? 2u * G::kQueryA : G::kQueryA;
	static constexpr uint32_t kMlpScratchInter = G::kTopK * G::kExpertInter > G::kDenseInter ? G::kTopK * G::kExpertInter : G::kDenseInter;
	static constexpr bool kRopeTable = G::kRopeFactor > 1.0f;
	static_assert(G::kQueryA % GLM_GRANULE == 0u && kSlot % GLM_GRANULE == 0u, "projection slices start on 16-byte columns");
	static_assert(G::kQueryGate == 0u || G::kQueryGate == G::kValue, "the attention output gate covers each head's value");
	static_assert(G::kExpertGroups >= 1u && G::kExperts % G::kExpertGroups == 0u && G::kTopGroups <= G::kExpertGroups, "expert groups split the experts evenly");
	static_assert(G::kRopeFactor <= 1.0f || (G::kRopeBetaFast > 0.0f && G::kRopeBetaSlow > 0.0f && G::kRopeOriginalPositions != 0u),
		"a geometry that turns YaRN on chooses its betas and original positions");
};

struct Glm52Dimensions : GlmDefaults
{
	static constexpr uint32_t kHidden = SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
	static constexpr uint32_t kLayers = SPARK_GLM52_MODEL_LAYER_COUNT;
	static constexpr uint32_t kFirstRouted = SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER;
	static constexpr uint32_t kVocab = SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT;
	static constexpr uint32_t kHeads = SPARK_GLM52_MODEL_HEAD_COUNT;
	static constexpr uint32_t kLatent = SPARK_GLM52_MODEL_LATENT_DIMENSION;
	static constexpr uint32_t kRope = SPARK_GLM52_MODEL_ROPE_DIMENSION;
	static constexpr uint32_t kNope = SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION;
	static constexpr uint32_t kValue = SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION;
	static constexpr uint32_t kQueryA = SPARK_GLM52_MODEL_QUERY_A_DIMENSION;
	static constexpr uint32_t kSelected = SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT;
	static constexpr uint32_t kMaxPositions = SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS;
	static constexpr uint32_t kIndexHeads = SPARK_GLM52_MODEL_DSA_INDEX_HEAD_COUNT;
	static constexpr uint32_t kIndexDim = SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION;
	static constexpr uint32_t kExperts = SPARK_GLM52_MODEL_MOE_EXPERT_COUNT;
	static constexpr uint32_t kTopK = SPARK_GLM52_MODEL_MOE_TOP_K;
	static constexpr uint32_t kExpertInter = SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION;
	static constexpr uint32_t kDenseInter = SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION;
	static constexpr float kRmsEpsilon = SPARK_GLM52_MODEL_RMS_NORM_EPSILON;
	static constexpr float kIndexEpsilon = SPARK_GLM52_MODEL_DSA_INDEX_NORM_EPSILON;
	static constexpr float kIndexScale = SPARK_GLM52_MODEL_DSA_INDEX_SOFTMAX_SCALE;
	static constexpr float kRopeTheta = SPARK_GLM52_MODEL_ROPE_THETA;
	static constexpr float kQkScale = SPARK_GLM52_MODEL_QK_SCALE;
	static constexpr float kRoutedScale = SPARK_GLM52_MODEL_MOE_ROUTED_SCALING_FACTOR;
	static uint32_t FullIndexer(uint32_t layer) { return SparkGlm52StagePackLayerHasFullIndexer(layer); }
	static const SparkGlm52StagePackGeometry *PackGeometry(void) { return SparkGlm52StagePackGeometryGlm52(); }
	static constexpr uint32_t kExpertCodec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	static constexpr uint32_t kLinearCodec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	static constexpr LmRopePairing kIndexPairing = LM_ROPE_INTERLEAVED;
};

static inline void GlmYarnFrequencies(float *table, uint32_t rope, float theta, float factor, float beta_fast, float beta_slow, uint32_t original_positions)
{
	const double pi = 3.14159265358979323846;
	double low = floor((double)rope * log((double)original_positions / ((double)beta_fast * 2.0 * pi)) / (2.0 * log((double)theta)));
	double high = ceil((double)rope * log((double)original_positions / ((double)beta_slow * 2.0 * pi)) / (2.0 * log((double)theta)));
	low = low > 0.0 ? low : 0.0;
	high = high < (double)(rope - 1u) ? high : (double)(rope - 1u);
	if ( low == high )
		high += 0.001;
	for ( uint32_t index = 0u; index < rope / 2u; index++ )
	{
		const float frequency = powf(theta, (float)(2u * index) / (float)rope);
		const float extrapolation = 1.0f / frequency, interpolation = 1.0f / (factor * frequency);
		float ramp = ((float)index - (float)low) / ((float)high - (float)low);
		ramp = ramp < 0.0f ? 0.0f : ramp > 1.0f ? 1.0f : ramp;
		const float extrapolation_factor = 1.0f - ramp;
		table[index] = interpolation * (1.0f - extrapolation_factor) + extrapolation * extrapolation_factor;
	}
}

static_assert(LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS == GLM_SPLIT_PARTITIONS, "split-partials sizing matches the kernel's partition cap");

template<class S>
struct GlmKv
{
	static constexpr uint32_t kSlotBytes = S::kSlotBytes;
	static constexpr uint32_t kPageSlots = GLM_PAGE_SLOTS;
	static constexpr uint32_t kPageBytes = S::kSlotBytes * GLM_PAGE_SLOTS * S::kLayers;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / GLM_PAGE_SLOTS; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % GLM_PAGE_SLOTS; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + GLM_PAGE_SLOTS - 1u) / GLM_PAGE_SLOTS; }
	static __host__ __device__ constexpr uint64_t PoolBytes(uint64_t pages) { return pages * (uint64_t)kPageBytes; }
};
template<class S>
using GlmIndexKv = LmKvLatent<16u, S::kIndexDim, 0u, GLM_PAGE_SLOTS>;
template<class S>
using GlmKvGathered = LmKvGeometry<S::kSlotBytes, 1u, true>;

template<class S>
static inline uint32_t GlmFullIndexer(uint32_t layer)
{
	return S::FullIndexer(layer);
}

typedef struct GlmLinearWeight
{
	const uint8_t *weight;
	const float *scale;
} GlmLinearWeight;

typedef struct GlmLayer
{
	const uint16_t *attn_norm;
	GlmLinearWeight q_a;
	const uint16_t *q_a_norm;
	GlmLinearWeight q_b;
	GlmLinearWeight kv_a;
	const uint16_t *kv_a_norm;
	const uint16_t *kv_b_key;
	const uint16_t *kv_b_value;
	GlmLinearWeight o_proj;
	const uint16_t *post_norm;
	GlmLinearWeight index_q;
	GlmLinearWeight index_k;
	const uint16_t *index_head;
	const uint16_t *index_norm_weight;
	const uint16_t *index_norm_bias;
	GlmLinearWeight dense_gate_up;
	GlmLinearWeight dense_down;
	const uint16_t *router;
	const float *router_bias;
	GlmLinearWeight shared_gate_up;
	GlmLinearWeight shared_down;
	const uint16_t *attn_gate_down;
	const uint16_t *attn_gate_up;
	const uint16_t *post_gate_down;
	const uint16_t *post_gate_up;
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
	uint32_t prenormed;
	uint16_t *q_compressed_bf16;
	uint16_t *q_input_bf16;
	const float *rope_table;
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
	float *gate_bottleneck;
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
	uint32_t *index_bitmaps;
	uint32_t index_bitmap_words;
	LmKvAccessError *index_error;
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

template<class S>
static inline uint32_t GlmProjectionSliceWidth(uint32_t tp_degree)
{
	return GlmProjectionSliceMax(S::kQueryA, tp_degree) + GlmProjectionSliceMax(S::kSlot, tp_degree);
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

template<class S>
static __global__ __launch_bounds__(GLM_THREADS, 1) void GlmProjectionNormKernel(const uint16_t *__restrict__ gathered, const uint16_t *__restrict__ query_norm,
	const uint16_t *__restrict__ latent_norm, const uint32_t *__restrict__ positions, uint16_t *__restrict__ query_out, uint16_t *__restrict__ slot_out,
	uint32_t rows, uint32_t tp_degree, uint32_t width, uint32_t query_max, uint32_t rope_dimension, float theta, float epsilon,
	uint16_t *__restrict__ query_input_out, const float *__restrict__ rope_table)
{
	__shared__ float query[S::kQueryA];
	__shared__ float latent[S::kSlot];
	__shared__ float reduction[GLM_THREADS / LM_WARP_LANES];
	const uint32_t row = blockIdx.x, half = rope_dimension / 2u;
	uint16_t *query_row = query_out + (uint64_t)row * S::kQueryA;
	uint32_t rank, offset, index;
	float total = 0.0f, scale, value, position;
	LmDependentRelease();
	for ( index = threadIdx.x; index < S::kQueryA; index += GLM_THREADS )
	{
		GlmProjectionSource(index, S::kQueryA, tp_degree, &rank, &offset);
		value = LmBf16ToFloat(gathered[((uint64_t)rank * rows + row) * width + offset]);
		query[index] = value;
		total += value * value;
	}
	total = LmBlockSum<GLM_THREADS>(total, reduction);
	scale = SparkMathRmsScale(total, (float)S::kQueryA, epsilon);
	for ( index = threadIdx.x; index < S::kQueryA; index += GLM_THREADS )
		query_row[index] = LmFloatToBf16(query[index] * scale * LmScalarToFloat(query_norm[index]));
	if constexpr ( S::kQueryConcat )
		for ( index = threadIdx.x; index < S::kQueryA; index += GLM_THREADS )
		{
			query_input_out[(uint64_t)row * S::kQueryInput + index] = query_row[index];
			query_input_out[(uint64_t)row * S::kQueryInput + S::kQueryA + index] = LmFloatToBf16(query[index]);
		}
	total = 0.0f;
	for ( index = threadIdx.x; index < S::kSlot; index += GLM_THREADS )
	{
		GlmProjectionSource(index, S::kSlot, tp_degree, &rank, &offset);
		value = LmBf16ToFloat(gathered[((uint64_t)rank * rows + row) * width + query_max + offset]);
		latent[index] = value;
		if ( index < S::kLatent )
			total += value * value;
	}
	total = LmBlockSum<GLM_THREADS>(total, reduction);
	scale = SparkMathRmsScale(total, (float)S::kLatent, epsilon);
	for ( index = threadIdx.x; index < S::kSlot; index += GLM_THREADS )
		slot_out[(uint64_t)row * S::kSlot + index] = index < S::kLatent ? LmFloatToBf16(latent[index] * scale * LmScalarToFloat(latent_norm[index])) :
			LmFloatToBf16(latent[index]);
	__syncthreads();
	position = (float)positions[row];
	for ( index = threadIdx.x; index < half; index += GLM_THREADS )
		if constexpr ( S::kRopeTable )
			LmRopeRotate<LM_ROPE_INTERLEAVED>(slot_out, (uint64_t)row * S::kSlot + S::kLatent, index, half, position * rope_table[index]);
		else
			LmRopeRotate<LM_ROPE_INTERLEAVED>(slot_out, (uint64_t)row * S::kSlot + S::kLatent, index, half,
				position * __powf(theta, -2.0f * (float)index / (float)rope_dimension));
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
static int32_t GlmSkinny(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, float *output_f32, uint32_t rows,
	uint32_t input_dimension, uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset, cudaStream_t stream)
{
	LmSkinnyArguments args;
	int32_t status;
	memset(&args, 0, sizeof(args));
	args.weight = (const uint8_t *)weight;
	args.activation = activation;
	args.output_bf16 = output_bf16;
	args.output_f32 = output_f32;
	args.scale = scale;
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

static int32_t GlmLinearBf16(const uint16_t *activation, const void *weight, uint16_t *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset, uint32_t multiprocessors, cudaStream_t stream)
{
	if ( activation == 0 || weight == 0 || output == 0 || rows == 0u || input_dimension == 0u || output_dimension == 0u || multiprocessors == 0u )
		return LM_LAUNCH_ERR_SHAPE;
	if ( rows <= LM_SKINNY_ROWS_WIDE )
		return GlmSkinny<LmBf16Format>(weight, LmScaleTensorNone(), activation, output, 0, rows, input_dimension, output_dimension, output_row_stride,
			output_column_offset, stream);
	return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), activation, output, 0, rows, input_dimension, output_dimension,
		output_row_stride, output_column_offset, multiprocessors, stream);
}

template<class S>
using GlmLinearFormat = typename LmWeightCodec<S::kLinearCodec>::Format;

template<class S>
static GlmLinearWeight GlmLinearRows(GlmLinearWeight weight, uint32_t first_row, uint32_t input_dimension)
{
	weight.weight += (uint64_t)first_row * input_dimension * GlmLinearFormat<S>::kStoredBits / 8u;
	weight.scale += (uint64_t)first_row * ((input_dimension + GlmLinearFormat<S>::kScaleGroup - 1u) / GlmLinearFormat<S>::kScaleGroup);
	return weight;
}

template<class S>
static LmScaleTensor GlmLinearScale(GlmLinearWeight weight, uint32_t input_dimension, uint32_t output_dimension)
{
	return LmWeightCodecScaleTensor<S::kLinearCodec>(weight.scale, 1u, output_dimension, input_dimension);
}

template<class S>
static int32_t GlmLinear(const uint16_t *activation, GlmLinearWeight weight, uint16_t *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset, uint32_t multiprocessors, cudaStream_t stream)
{
	const LmScaleTensor scale = GlmLinearScale<S>(weight, input_dimension, output_dimension);
	if ( activation == 0 || weight.weight == 0 || weight.scale == 0 || output == 0 || rows == 0u || input_dimension == 0u || output_dimension == 0u ||
		multiprocessors == 0u )
		return LM_LAUNCH_ERR_SHAPE;
	if ( rows <= LM_SKINNY_ROWS_WIDE )
		return GlmSkinny<GlmLinearFormat<S>>(weight.weight, scale, activation, output, 0, rows, input_dimension, output_dimension, output_row_stride,
			output_column_offset, stream);
	return LmStreamGemmDense<GlmLinearFormat<S>>(weight.weight, scale, activation, output, 0, rows, input_dimension, output_dimension,
		output_row_stride, output_column_offset, multiprocessors, stream);
}

static int32_t GlmLaunched(void)
{
	return cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<class S>
static int32_t GlmIndexerKeys(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	int32_t status = GlmLinear<S>(b->normed_bf16, b->weights->index_k, b->index_key_bf16, rows, S::kHidden, S::kIndexDim, S::kIndexDim, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmLayerNormKernel<GLM_THREADS,uint16_t>), rows, GLM_THREADS, (S::kIndexDim + 8u) * sizeof(float), stream,
		b->index_key_bf16, b->weights->index_norm_weight, b->weights->index_norm_bias, b->index_key_bf16, S::kIndexDim, S::kIndexDim, S::kIndexEpsilon);
	LM_LAUNCH((LmRopeKernel<GLM_THREADS,S::kIndexPairing,S::kRopeTable>), rows, GLM_THREADS, 0, stream,
		b->index_key_bf16, b->positions, S::kIndexDim, 0u, S::kRope, S::kRopeTheta, b->rope_table);
	LM_LAUNCH((LmKvShardStoreKernel<GlmIndexKv<S>,GLM_THREADS>), rows, GLM_THREADS, 0, stream,
		b->index_shard, b->index_key_bf16, b->sequence_of_row, b->positions, rows, S::kIndexDim);
	return GlmLaunched();
}

template<class S>
static int32_t GlmIndexerQuery(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	int32_t status = GlmLinear<S>(b->q_compressed_bf16, b->weights->index_q, b->index_query_bf16, rows, S::kQueryA, S::kIndexQuery, S::kIndexQuery, 0u, sms, stream);
	if ( status == LM_LAUNCH_OK )
		status = GlmLinearBf16(b->normed_bf16, b->weights->index_head, b->index_head_weight_bf16, rows, S::kHidden, S::kIndexHeads, S::kIndexHeads, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if constexpr ( S::kRopeTable )
		LM_LAUNCH((LmRopePerHeadKernel<GLM_THREADS,S::kIndexPairing>), dim3(rows,S::kIndexHeads), GLM_THREADS, 0, stream,
			b->index_query_bf16, b->positions, S::kIndexHeads, S::kIndexDim, S::kRope, S::kRopeTheta, b->rope_table, 1.0f, 0u);
	else
		LM_LAUNCH((LmRopePerHeadKernel<GLM_THREADS,S::kIndexPairing>), dim3(rows,S::kIndexHeads), GLM_THREADS, 0, stream,
			b->index_query_bf16, b->positions, S::kIndexHeads, S::kIndexDim, 0u, S::kRope, S::kRopeTheta);
	return GlmLaunched();
}

static uint32_t GlmIndexerChunkRows(const GlmBuffers *b, uint32_t rows)
{
	return b->selection_rows != 0u && b->selection_rows < rows ? b->selection_rows : rows;
}

template<class S>
static int32_t GlmIndexerScore(const GlmBuffers *b, LmKvShardView pages, uint32_t rows, uint32_t score_stride, uint32_t keep, uint32_t *selected,
	uint2 *candidates, cudaStream_t stream)
{
	const uint32_t chunk_rows = GlmIndexerChunkRows(b, rows);
	for ( uint32_t first = 0u; first < rows; first += chunk_rows )
	{
		const uint32_t count = rows - first < chunk_rows ? rows - first : chunk_rows;
		if ( LmWeightedSparseScoreLaunch<GlmIndexKv<S>,S::kIndexHeads,S::kIndexDim,LmKvShardView>(b->index_query_bf16 + (uint64_t)first * S::kIndexQuery,
				b->index_head_weight_bf16 + (uint64_t)first * S::kIndexHeads, pages, b->sequence_of_row + first, b->context_length,
				b->row_positions != 0 ? b->row_positions + first : 0, count, score_stride, S::kIndexScale / sqrtf((float)S::kIndexHeads),
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

template<class S>
static int32_t GlmIndexerScoreSelect(const GlmBuffers *b, uint32_t rows, uint32_t score_stride, uint32_t keep, uint2 *candidates, cudaStream_t stream)
{
	const uint32_t chunk_rows = GlmIndexerChunkRows(b, rows);
	for ( uint32_t first = 0u; first < rows; first += chunk_rows )
	{
		const uint32_t count = rows - first < chunk_rows ? rows - first : chunk_rows;
		if ( LmWeightedSparseScoreLaunch<GlmIndexKv<S>,S::kIndexHeads,S::kIndexDim,LmKvShardView>(b->index_query_bf16 + (uint64_t)first * S::kIndexQuery,
				b->index_head_weight_bf16 + (uint64_t)first * S::kIndexHeads, b->index_shard, b->sequence_of_row + first, b->context_length,
				b->row_positions + first, count, score_stride, S::kIndexScale / sqrtf((float)S::kIndexHeads), b->selection_scores, stream) != cudaSuccess ||
			LmIndexSelectLocalLaunch<LM_INDEX_SELECT_THREADS>(b->kv_shard, b->selection_scores, score_stride, b->sequence_of_row + first, b->context_length,
				b->row_positions + first, keep, S::kSelected, count, candidates + (uint64_t)first * keep, b->index_error, stream) != cudaSuccess )
			return LM_LAUNCH_ERR_LAUNCH;
	}
	return GlmLaunched();
}

template<class S>
static int32_t GlmNormGate(uint16_t *rows_bf16, const uint16_t *down, const uint16_t *up, float *bottleneck, uint32_t rows, cudaStream_t stream)
{
	if constexpr ( S::kNormGateRank != 0u )
		if ( down != 0 && up != 0 && rows != 0u )
		{
			if ( bottleneck == 0 )
				return LM_LAUNCH_ERR_SHAPE;
			LM_LAUNCH((LmLowRankGateDownKernel<GLM_THREADS,S::kNormGateRank>), dim3(S::kNormGateRank, rows), GLM_THREADS, 0, stream, rows_bf16, down, bottleneck, S::kHidden);
			LM_LAUNCH((LmLowRankGateApplyKernel<GLM_THREADS,S::kNormGateRank>), dim3((S::kHidden + GLM_THREADS - 1u) / GLM_THREADS, rows), GLM_THREADS, 0, stream,
				rows_bf16, bottleneck, up, S::kHidden);
		}
	return GlmLaunched();
}

template<class S>
static int32_t GlmAttentionProject(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	uint32_t q_first, q_count, kv_first, kv_count, width, query_max;
	int32_t status = LM_LAUNCH_OK;
	if ( b->prenormed == 0u )
	{
		LM_LAUNCH((LmFusedResidualRmsNormKernel<GLM_THREADS,uint16_t>), rows, GLM_THREADS, (S::kHidden + 8u) * sizeof(float), stream,
			b->hidden_bf16, b->residual_bf16, b->weights->attn_norm, b->residual_bf16, b->normed_bf16, S::kHidden, S::kHidden, S::kRmsEpsilon);
		if ( GlmNormGate<S>(b->normed_bf16, b->weights->attn_gate_down, b->weights->attn_gate_up, b->gate_bottleneck, rows, stream) != LM_LAUNCH_OK )
			return LM_LAUNCH_ERR_LAUNCH;
	}
	GlmProjectionSlice(S::kQueryA, b->tp_degree, b->tp_rank, &q_first, &q_count);
	GlmProjectionSlice(S::kSlot, b->tp_degree, b->tp_rank, &kv_first, &kv_count);
	width = GlmProjectionSliceWidth<S>(b->tp_degree);
	query_max = GlmProjectionSliceMax(S::kQueryA, b->tp_degree);
	if ( q_count != 0u && kv_count != 0u )
	{
		const GlmLinearWeight q_a = GlmLinearRows<S>(b->weights->q_a, q_first, S::kHidden), kv_a = GlmLinearRows<S>(b->weights->kv_a, kv_first, S::kHidden);
		const LmSkinnyDenseTarget targets[2] = {
			{q_a.weight, b->projection_local_bf16, 0, q_count, width, 0u, GlmLinearScale<S>(q_a, S::kHidden, q_count)},
			{kv_a.weight, b->projection_local_bf16, 0, kv_count, width, query_max, GlmLinearScale<S>(kv_a, S::kHidden, kv_count)}};
		status = LmSkinnyDenseMulti<GlmLinearFormat<S>>(targets, 2u, b->normed_bf16, rows, S::kHidden, stream);
		if ( status != LM_LAUNCH_ERR_SHAPE )
			return status;
	}
	status = LM_LAUNCH_OK;
	if ( q_count != 0u )
		status = GlmLinear<S>(b->normed_bf16, GlmLinearRows<S>(b->weights->q_a, q_first, S::kHidden), b->projection_local_bf16, rows, S::kHidden, q_count, width, 0u, sms, stream);
	if ( status == LM_LAUNCH_OK && kv_count != 0u )
		status = GlmLinear<S>(b->normed_bf16, GlmLinearRows<S>(b->weights->kv_a, kv_first, S::kHidden), b->projection_local_bf16, rows, S::kHidden, kv_count, width, query_max, sms, stream);
	return status;
}

template<class S>
static int32_t GlmAttentionPrologue(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	int32_t status;
	GlmProjectionNormKernel<S><<<rows, GLM_THREADS, 0, stream>>>(b->projection_gather_bf16, b->weights->q_a_norm, b->weights->kv_a_norm, b->positions,
		b->q_compressed_bf16, b->kv_slot_bf16, rows, b->tp_degree, GlmProjectionSliceWidth<S>(b->tp_degree), GlmProjectionSliceMax(S::kQueryA, b->tp_degree),
		S::kRope, S::kRopeTheta, S::kRmsEpsilon, b->q_input_bf16, b->rope_table);
	if ( cudaPeekAtLastError() != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	status = GlmLinear<S>(S::kQueryConcat ? b->q_input_bf16 : b->q_compressed_bf16, b->weights->q_b, b->q_bf16, rows, S::kQueryInput, b->q_b_rows, b->q_b_rows, 0u,
		sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmExtractRopePerHeadKernel<GLM_THREADS,LM_ROPE_INTERLEAVED,S::kRopeTable>), dim3(rows,b->attn_heads), GLM_THREADS, 0, stream,
		b->q_bf16, b->query_rope_bf16, b->positions, b->attn_heads, S::kQueryHead, S::kNope, S::kRope, S::kRopeTheta, b->rope_table);
	if ( LmPerHeadProjectChainLaunch<GLM_THREADS,S::kNope,S::kLatent,S::kQueryHead,0u>(b->q_bf16, b->weights->kv_b_key + (uint64_t)b->tp_rank *
			b->attn_heads * S::kLatent * S::kNope, b->query_latent_bf16, b->attn_heads, rows, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	return LM_LAUNCH_OK;
}

template<class S>
static int32_t GlmLatentStore(const GlmBuffers *b, uint32_t rows, cudaStream_t stream)
{
	LM_LAUNCH((LmKvShardStoreKernel<GlmKv<S>,GLM_THREADS>), rows, GLM_THREADS, 0, stream,
		b->cache_shard, b->kv_slot_bf16, b->sequence_of_row, b->positions, rows, S::kSlot);
	return GlmLaunched();
}

template<class S>
static int32_t GlmLatentAttentionGathered(const GlmBuffers *b, LmKvView cache, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	const uint32_t *selected = context > S::kSelected ? b->selected_positions : 0;
	const uint32_t selected_count = context > S::kSelected ? b->selected_position_count : 0u;
	if ( (b->single_sequence_rows != 0u || b->prefill_block_table != 0) && rows > LM_SKINNY_ROWS && selected == 0 )
		return LmLatentAttentionPrefillLaunch<GlmKvGathered<S>,S::kLatent,S::kRope>(b->query_latent_bf16, b->query_rope_bf16, cache, b->sequence_of_row,
			b->row_positions, b->attn_heads, S::kQkScale, b->attention_latent_bf16, rows, b->prefill_block_table, b->prefill_table_blocks, stream) ==
			cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	if ( (b->single_sequence_rows != 0u || b->prefill_block_table != 0) && rows > LM_SKINNY_ROWS && selected != 0 && b->prefill_union_positions != 0 )
		return LmLatentAttentionSparsePrefillLaunch<GlmKvGathered<S>,S::kLatent,S::kRope,S::kSelected>(b->query_latent_bf16, b->query_rope_bf16, cache,
			b->sequence_of_row, b->row_positions, b->attn_heads, S::kQkScale, b->attention_latent_bf16, rows, b->prefill_block_table,
			b->prefill_table_blocks, selected, selected_count, b->prefill_union_positions, b->prefill_union_masks, b->prefill_union_counts,
			b->prefill_union_entries, stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	return LmLatentRopeHeadsSplitLaunch<GlmKvGathered<S>,GLM_THREADS,S::kLatent,S::kRope,true>(b->query_latent_bf16, b->query_rope_bf16, cache,
		b->sequence_of_row, b->context_length, selected, selected_count, b->attn_heads, S::kQkScale, b->attention_latent_bf16, b->row_positions,
		rows, context > S::kSelected ? S::kSelected : context, b->decode_split_context_threshold, b->attention_split_partials,
		(uint32_t)b->attention_split_partial_blocks, sms, stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<class S>
static int32_t GlmAttentionOutput(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	if ( LmPerHeadProjectChainLaunch<GLM_THREADS,S::kLatent,S::kValue>(b->attention_latent_bf16, b->weights->kv_b_value + (uint64_t)b->tp_rank *
			b->attn_heads * S::kValue * S::kLatent, b->attention_value_bf16, b->attn_heads, rows, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	if constexpr ( S::kQueryGate != 0u )
		LM_LAUNCH((LmHeadSigmoidGateKernel<GLM_THREADS>), dim3(rows,b->attn_heads), GLM_THREADS, 0, stream,
			b->attention_value_bf16, b->q_bf16, b->attn_heads, S::kValue, S::kQueryHead, S::kNope + S::kRope);
	return GlmLinear<S>(b->attention_value_bf16, b->weights->o_proj, b->attention_out_bf16, rows, b->attn_output_columns, S::kHidden, S::kHidden, 0u, sms, stream);
}

template<class S>
static uint32_t GlmShardIndexing(uint32_t layer, uint32_t context)
{
	return GlmFullIndexer<S>(layer) != 0u && context > S::kSelected ? 1u : 0u;
}

template<class S>
static uint32_t GlmShardKeep(SparkKvShard shard, uint32_t context)
{
	const uint32_t stride = SparkKvShardGatherKeys(shard, context);
	return stride < S::kSelected ? stride : S::kSelected;
}

template<class S>
static int32_t GlmShardStoreAll(const GlmBuffers *b, uint32_t rows, uint32_t layer, uint32_t sms, cudaStream_t stream)
{
	int32_t status = GlmAttentionPrologue<S>(b, rows, sms, stream);
	if ( status == LM_LAUNCH_OK && GlmFullIndexer<S>(layer) != 0u )
		status = GlmIndexerKeys<S>(b, rows, sms, stream);
	if ( status == LM_LAUNCH_OK )
		status = GlmLatentStore<S>(b, rows, stream);
	return status;
}

template<class S>
static int32_t GlmShardScatterPre(const GlmBuffers *b, uint32_t rows, uint32_t context, uint32_t layer, uint32_t sms, cudaStream_t stream)
{
	int32_t status = GlmShardStoreAll<S>(b, rows, layer, sms, stream);
	if ( status == LM_LAUNCH_OK && GlmShardIndexing<S>(layer, context) != 0u )
	{
		status = GlmIndexerQuery<S>(b, rows, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = GlmIndexerScoreSelect<S>(b, rows, SparkKvShardGatherKeys(b->kv_shard, context), GlmShardKeep<S>(b->kv_shard, context),
				(uint2 *)((uint8_t *)b->shard_send_bf16 + b->shard_candidate_offset), stream);
	}
	if ( status != LM_LAUNCH_OK )
		return status;
	return LmLatentShardQueryPackLaunch<S::kLatent,S::kRope>(b->query_latent_bf16, b->query_rope_bf16, b->attn_heads, rows, b->shard_send_bf16, stream) ==
		cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<class S>
static int32_t GlmShardScatterMid(const GlmBuffers *b, uint32_t rows, uint32_t context, uint32_t layer, cudaStream_t stream)
{
	if ( GlmShardIndexing<S>(layer, context) != 0u )
	{
		const uint32_t keep = GlmShardKeep<S>(b->kv_shard, context), chunk_rows = GlmIndexerChunkRows(b, rows);
		const uint2 *candidates = (const uint2 *)((const uint8_t *)b->shard_received_bf16 + b->shard_candidate_offset);
		for ( uint32_t first = 0u; first < rows; first += chunk_rows )
		{
			const uint32_t count = rows - first < chunk_rows ? rows - first : chunk_rows;
			if ( LmIndexSelectMergeLaunch<LM_INDEX_SELECT_THREADS>(candidates + (uint64_t)first * keep, b->shard_rank_stride_bytes / sizeof(uint2),
					b->kv_shard, keep, b->sequence_of_row + first, b->context_length, b->row_positions + first, S::kSelected, S::kSelected, count, b->index_bitmap_words,
					b->index_bitmaps, b->selected_positions + (uint64_t)first * S::kSelected, b->index_error, stream) != cudaSuccess )
				return LM_LAUNCH_ERR_LAUNCH;
		}
	}
	return LmLatentShardPartialLaunch<GlmKv<S>,LmKvShardView,S::kLatent,S::kRope>(b->cache_shard, b->shard_received_bf16,
		b->shard_rank_stride_bytes / sizeof(uint16_t), b->attn_heads, b->sequence_of_row, b->context_length, b->row_positions,
		context > S::kSelected ? b->selected_positions : 0, S::kSelected, S::kSelected, S::kQkScale, b->shard_partials, b->shard_partial_stride,
		rows, stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<class S>
static int32_t GlmShardScatterPost(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	if ( LmLatentShardMergeLaunch<S::kLatent>(b->shard_partials_received, b->shard_partial_stride, b->kv_shard.degree, b->attn_heads,
			b->attention_latent_bf16, rows, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	return GlmAttentionOutput<S>(b, rows, sms, stream);
}

template<class S>
static int32_t GlmShardGatherPre(const GlmBuffers *b, uint32_t rows, uint32_t context, uint32_t layer, uint32_t sms, cudaStream_t stream)
{
	const uint32_t indexing = GlmShardIndexing<S>(layer, context);
	int32_t status = GlmShardStoreAll<S>(b, rows, layer, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	if ( cudaMemcpyAsync(b->kv_gather_latent_tail, b->kv_slot_bf16, (size_t)rows * S::kSlotBytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
		LmRowDigestLaunch<GLM_THREADS>(b->kv_slot_bf16, S::kSlot, b->positions, rows, layer, b->kv_gather_digest, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	if ( indexing != 0u )
	{
		status = GlmIndexerQuery<S>(b, b->kv_select_rows, sms, stream);
		if ( status == LM_LAUNCH_OK )
			status = GlmIndexerScore<S>(b, b->index_shard, b->kv_select_rows, SparkKvShardGatherKeys(b->kv_shard, context), b->kv_select_keep,
				b->kv_select_local, b->kv_select_candidates, stream);
		if ( status != LM_LAUNCH_OK )
			return status;
	}
	if ( b->kv_gather_old_keys == 0u )
		return LM_LAUNCH_OK;
	return LmKvShardGatherPackLaunch<GlmKv<S>,GLM_THREADS>(b->cache_shard, b->kv_gather_sequences, b->kv_gather_offset, b->kv_gather_old_bound, 1u,
		b->kv_gather_old_keys, b->kv_gather_latent_pack, stream) == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

template<class S>
static int32_t GlmShardGatherPost(const GlmBuffers *b, uint32_t rows, uint32_t context, uint32_t sms, cudaStream_t stream)
{
	GlmBuffers local = *b;
	int32_t status;
	local.sequence_of_row = b->kv_gather_rows_zero;
	local.context_length = b->kv_gather_context_word;
	status = GlmLatentAttentionGathered<S>(&local, local.cache_gathered, rows, context, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	return GlmAttentionOutput<S>(b, rows, sms, stream);
}

template<class S>
static int32_t GlmDenseMlp(const GlmBuffers *b, uint32_t rows, uint32_t sms, cudaStream_t stream)
{
	int32_t status;
	if ( b->prenormed == 0u )
		LM_LAUNCH((LmFusedResidualRmsNormKernel<GLM_THREADS,uint16_t>), rows, GLM_THREADS, (S::kHidden + 8u) * sizeof(float), stream,
			b->attention_out_bf16, b->residual_bf16, b->weights->post_norm, b->residual_bf16, b->normed_bf16, S::kHidden, S::kHidden, S::kRmsEpsilon);
	status = GlmLinear<S>(b->normed_bf16, b->weights->dense_gate_up, b->gate_up_bf16, rows, S::kHidden, b->dense_gate_up_rows, b->dense_gate_up_rows, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmSiluMulKernel<GLM_THREADS>), rows, GLM_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16, b->dense_intermediate, false);
	return GlmLinear<S>(b->intermediate_bf16, b->weights->dense_down, b->hidden_bf16, rows, b->dense_intermediate, S::kHidden, S::kHidden, 0u, sms, stream);
}

template<class S>
static int32_t GlmMoeRoute(const GlmBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	int32_t status;
	if ( b->prenormed == 0u )
	{
		LM_LAUNCH((LmFusedResidualRmsNormKernel<GLM_THREADS,uint16_t>), rows, GLM_THREADS, (S::kHidden + 8u) * sizeof(float), stream,
			b->attention_out_bf16, b->residual_bf16, b->weights->post_norm, b->residual_bf16, b->normed_bf16, S::kHidden, S::kHidden, S::kRmsEpsilon);
		if ( GlmNormGate<S>(b->normed_bf16, b->weights->post_gate_down, b->weights->post_gate_up, b->gate_bottleneck, rows, stream) != LM_LAUNCH_OK )
			return LM_LAUNCH_ERR_LAUNCH;
	}
	status = rows <= LM_SKINNY_ROWS_WIDE
		? GlmSkinny<LmBf16Format>(b->weights->router, LmScaleTensorNone(), b->normed_bf16, 0, b->router_logits, rows, S::kHidden, S::kExperts, 0u, 0u, stream)
		: LmStreamGemmDense<LmBf16Format>(b->weights->router, LmScaleTensorNone(), b->normed_bf16, 0, b->router_logits, rows, S::kHidden, S::kExperts,
			0u, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK || GlmLaunched() != LM_LAUNCH_OK )
		return status != LM_LAUNCH_OK ? status : LM_LAUNCH_ERR_LAUNCH;
	if constexpr ( S::kExpertGroups > 1u )
	{
		LM_LAUNCH((LmTopkSmallKernel<GLM_THREADS,S::kTopK,true,S::kExpertGroups,S::kTopGroups,LM_TOPK_SCORE_SIGMOID>), rows, GLM_THREADS,
			2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t), stream, b->router_logits, S::kExperts, b->route_expert, b->route_weight, b->weights->router_bias,
			(const uint16_t *)0, S::kRoutedScale);
		if ( cudaPeekAtLastError() != cudaSuccess )
			return LM_LAUNCH_ERR_LAUNCH;
	}
	else if ( LmTopkRouteLaunch<GLM_THREADS,S::kTopK,true,LM_TOPK_SCORE_SIGMOID>(rows, b->router_logits, S::kExperts, b->route_expert, b->route_weight,
			b->weights->router_bias, 0, S::kRoutedScale, stream) != cudaSuccess )
		return LM_LAUNCH_ERR_LAUNCH;
	status = LmRouteBuild<GLM_THREADS,S::kExperts>(b->route_expert, rows, packed_rows, S::kTopK, b->group_row_offset, b->route_packed_row,
		b->route_source_token, b->expert_w1_rows, S::kHidden, GLM_TILE_N, b->group_tile_prefix_w1, b->group_tile_prefix_w2, stream);
	return status != LM_LAUNCH_OK ? status : GlmLaunched();
}

template<class S>
static int32_t GlmExpertsGateUp(const GlmBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	using Format = typename LmWeightCodec<S::kExpertCodec>::Format;
	LmGemmArguments gemm;
	int32_t status;
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmWeightCodecScaleTensor<S::kExpertCodec>(b->expert_w1_scale, S::kExperts, b->expert_w1_rows, S::kHidden);
	if ( rows >= GLM_STREAM_EXPERT_ROWS )
	{
		status = LmStreamGemmGrouped<Format>(b->expert_w1_weight, gemm.scale_b, b->normed_bf16, b->gate_up_bf16, b->group_row_offset,
			b->route_source_token, S::kExperts, packed_rows, S::kHidden, b->expert_w1_rows, sms, stream);
		return status == LM_LAUNCH_OK && cudaPeekAtLastError() != cudaSuccess ? LM_LAUNCH_ERR_LAUNCH : status;
	}
	status = rows == 1u ? LmSkinnyExperts<Format>(b->expert_w1_weight, gemm.scale_b, b->normed_bf16, b->gate_up_bf16, b->route_expert,
		b->route_packed_row, packed_rows, S::kTopK, 0u, S::kHidden, b->expert_w1_rows, stream) : LM_LAUNCH_ERR_SHAPE;
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmSkinnyGroupedExperts<Format>(b->expert_w1_weight, gemm.scale_b, b->normed_bf16, b->gate_up_bf16, b->group_row_offset,
			b->route_source_token, S::kExperts, packed_rows, 0u, S::kHidden, b->expert_w1_rows, stream);
	if ( status != LM_LAUNCH_ERR_SHAPE )
		return status == LM_LAUNCH_OK && cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : status == LM_LAUNCH_OK ? LM_LAUNCH_ERR_LAUNCH : status;
	gemm.prefix_built = 1u;
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix_w1;
	gemm.source_row_map = b->route_source_token;
	gemm.source_row_count = rows;
	gemm.output_bf16 = b->gate_up_bf16;
	status = LmGemmWeightOnlyIndirectLaunch<Format,GLM_TILE_N,GLM_TILE_STAGES,GLM_TILE_WARPS>(&gemm, b->normed_bf16, b->expert_w1_weight, packed_rows,
		rows, S::kTopK, S::kExperts, S::kHidden, b->expert_w1_rows, sms, stream);
	return status != LM_LAUNCH_OK ? status : GlmLaunched();
}

template<class S>
static int32_t GlmExpertsDown(const GlmBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	using Format = typename LmWeightCodec<S::kExpertCodec>::Format;
	LmGemmArguments gemm;
	int32_t status;
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmWeightCodecScaleTensor<S::kExpertCodec>(b->expert_w2_scale, S::kExperts, S::kHidden, b->expert_intermediate);
	if ( rows >= GLM_STREAM_EXPERT_ROWS )
	{
		status = LmStreamGemmGrouped<Format>(b->expert_w2_weight, gemm.scale_b, b->intermediate_bf16, b->expert_out_bf16, b->group_row_offset, 0,
			S::kExperts, packed_rows, b->expert_intermediate, S::kHidden, sms, stream);
		return status == LM_LAUNCH_OK && cudaPeekAtLastError() != cudaSuccess ? LM_LAUNCH_ERR_LAUNCH : status;
	}
	status = LmSkinnyExperts<Format>(b->expert_w2_weight, gemm.scale_b, b->intermediate_bf16, b->expert_out_bf16, b->route_expert, b->route_packed_row,
		packed_rows, S::kTopK, 1u, b->expert_intermediate, S::kHidden, stream);
	if ( status == LM_LAUNCH_ERR_SHAPE )
		status = LmSkinnyGroupedExperts<Format>(b->expert_w2_weight, gemm.scale_b, b->intermediate_bf16, b->expert_out_bf16, b->group_row_offset,
			b->route_source_token, S::kExperts, packed_rows, 1u, b->expert_intermediate, S::kHidden, stream);
	if ( status != LM_LAUNCH_ERR_SHAPE )
		return status == LM_LAUNCH_OK && cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : status == LM_LAUNCH_OK ? LM_LAUNCH_ERR_LAUNCH : status;
	gemm.prefix_built = 1u;
	gemm.group_row_offset = b->group_row_offset;
	gemm.group_tile_prefix = b->group_tile_prefix_w2;
	gemm.output_bf16 = b->expert_out_bf16;
	status = LmGemmWeightOnlyLaunch<Format,GLM_TILE_N,GLM_TILE_STAGES,GLM_TILE_WARPS>(&gemm, b->intermediate_bf16, b->expert_w2_weight, packed_rows, rows,
		S::kTopK, S::kExperts, b->expert_intermediate, S::kHidden, sms, true, stream);
	return status != LM_LAUNCH_OK ? status : GlmLaunched();
}

template<class S>
static int32_t GlmMoeExperts(const GlmBuffers *b, uint32_t rows, uint32_t packed_rows, uint32_t sms, cudaStream_t stream)
{
	int32_t status = GlmExpertsGateUp<S>(b, rows, packed_rows, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmSiluMulKernel<GLM_THREADS>), packed_rows, GLM_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16, b->expert_intermediate, false);
	status = GlmExpertsDown<S>(b, rows, packed_rows, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmMoeFinalizeKernel<GLM_THREADS>), dim3((S::kHidden + GLM_THREADS - 1u) / GLM_THREADS, rows), GLM_THREADS, 0, stream,
		b->expert_out_bf16, b->route_packed_row, b->route_weight, b->hidden_bf16, rows, S::kTopK, S::kHidden);
	status = GlmLinear<S>(b->normed_bf16, b->weights->shared_gate_up, b->gate_up_bf16, rows, S::kHidden, b->shared_gate_up_rows, b->shared_gate_up_rows, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmSiluMulKernel<GLM_THREADS>), rows, GLM_THREADS, 0, stream, b->gate_up_bf16, b->intermediate_bf16, b->shared_intermediate, false);
	status = GlmLinear<S>(b->intermediate_bf16, b->weights->shared_down, b->shared_out_bf16, rows, b->shared_intermediate, S::kHidden, S::kHidden, 0u, sms, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	LM_LAUNCH((LmAddRowsKernel<GLM_THREADS>), dim3((S::kHidden + GLM_THREADS - 1u) / GLM_THREADS, rows), GLM_THREADS, 0, stream,
		b->hidden_bf16, b->shared_out_bf16, b->hidden_bf16, rows, S::kHidden);
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
	typedef GlmShape<G> S;

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
		GlmLayer layers[S::kLayers];
		uint64_t w1_offset[S::kLayers];
		uint64_t w2_offset[S::kLayers];
		const uint16_t *final_norm;
		void *arena;
		uint64_t arena_bytes;
		void *lane_arena;
		uint32_t *lane_host_staging;
		void *lane_states[SPARK_STAGE_RUNNER_LANES];
		uint64_t lane_base[SPARK_STAGE_RUNNER_LANES][3];
		uint32_t lane_bound[SPARK_STAGE_RUNNER_LANES];
		uint16_t *hidden;
		uint16_t *residual;
		uint16_t *normed;
		uint16_t *row_share;
		uint16_t *row_normed;
		uint16_t *q_compressed;
		uint16_t *q_input;
		float *rope_table;
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
	float *gate_bottleneck;
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
		uint32_t *index_bitmaps;
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
		LmKvAccessError *error_words;
		LmKvAccessError *error_device;
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

	struct Frame
	{
		const SparkStageRunnerStep *step;
		cudaStream_t stream;
		uint32_t ordered;
		uint32_t first;
		GlmRegime regime;
		GlmWave wave;
		uint16_t *hidden;
		const uint32_t *positions;
		const uint32_t *slots;
		GlmBuffers b;
	};

	static uint32_t Units(uint64_t bytes)
	{
		return (uint32_t)((bytes + S::kUnitBytes - 1u) / S::kUnitBytes);
	}

	static void ScatterSizes(const State *state, uint32_t rows, uint32_t *query_units, uint32_t *indexing_units, uint32_t *partial_units,
		uint64_t *candidate_offset)
	{
		const uint64_t query = (uint64_t)rows * state->heads * S::kSlot * sizeof(uint16_t);
		*candidate_offset = query;
		*query_units = Units(query);
		*indexing_units = Units(query + (uint64_t)rows * S::kSelected * GLM_CANDIDATE_BYTES);
		*partial_units = Units((uint64_t)rows * state->heads * S::kPartialFloats * sizeof(float));
	}

	static SparkKvShardSectionLayout GatherLayout(const State *state, uint32_t old_bound)
	{
		return SparkKvShardSectionLayoutBuild(SparkKvShardGatherKeys(state->shard, old_bound), S::kSlotBytes, S::kUnitBytes, S::kSlotAlign,
			state->max_rows / S::kSlotAlign * S::kSlotAlign);
	}

	static uint32_t RowRegime(void *context, uint32_t row)
	{
		const GlmRegime *regime = (const GlmRegime *)context;
		const uint32_t tokens = regime->positions[row] + 1u;
		if ( tokens > S::kSelected )
			return 2u;
		return LmLatentAttentionContextSplits(tokens, GLM_SPLIT_THRESHOLD) != 0u ? 1u : 0u;
	}

	static uint32_t SelectionRegime(void *context, uint32_t row)
	{
		const GlmRegime *regime = (const GlmRegime *)context;
		return regime->positions[row] + 1u > S::kSelected ? 1u : 0u;
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
		const uint64_t rows = state->max_rows, packed = rows * S::kTopK, degree = state->tp_degree;
		const uint64_t share = (rows + degree - 1u) / degree, hidden = share * degree * S::kHidden * sizeof(uint16_t);
		const uint64_t width = GlmProjectionSliceWidth<S>(state->tp_degree), heads = state->heads;
		struct { void **target; uint64_t bytes; } plan[] =
		{
			{ (void **)&state->hidden, hidden },
			{ (void **)&state->residual, hidden },
			{ (void **)&state->normed, hidden },
			{ (void **)&state->row_share, share * S::kHidden * sizeof(uint16_t) },
			{ (void **)&state->row_normed, share * S::kHidden * sizeof(uint16_t) },
			{ (void **)&state->q_compressed, rows * S::kQueryA * sizeof(uint16_t) },
			{ (void **)&state->q_input, S::kQueryConcat ? rows * S::kQueryInput * sizeof(uint16_t) : 0u },
			{ (void **)&state->q, rows * heads * S::kQueryHead * sizeof(uint16_t) },
			{ (void **)&state->query_latent, rows * heads * S::kLatent * sizeof(uint16_t) },
			{ (void **)&state->query_rope, rows * heads * S::kRope * sizeof(uint16_t) },
			{ (void **)&state->index_query, rows * S::kIndexQuery * sizeof(uint16_t) },
			{ (void **)&state->index_key, rows * S::kIndexDim * sizeof(uint16_t) },
			{ (void **)&state->index_head, rows * S::kIndexHeads * sizeof(uint16_t) },
			{ (void **)&state->kv_slot, rows * S::kSlot * sizeof(uint16_t) },
			{ (void **)&state->attention_latent, rows * heads * S::kLatent * sizeof(uint16_t) },
			{ (void **)&state->attention_value, rows * heads * S::kValue * sizeof(uint16_t) },
			{ (void **)&state->attention_out, hidden },
			{ (void **)&state->projection_local, rows * width * sizeof(uint16_t) },
			{ (void **)&state->projection_gather, degree * rows * width * sizeof(uint16_t) },
			{ (void **)&state->gate_up, rows * (2u * S::kMlpScratchInter / degree) * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * (S::kMlpScratchInter / degree) * sizeof(uint16_t) },
			{ (void **)&state->expert_out, packed * S::kHidden * sizeof(uint16_t) },
			{ (void **)&state->shared_out, hidden },
			{ (void **)&state->router_logits, rows * S::kExperts * sizeof(float) },
			{ (void **)&state->selection_scores, (uint64_t)state->selection_rows * state->max_positions * sizeof(float) },
			{ (void **)&state->gate_bottleneck, rows * S::kNormGateRank * sizeof(float) },
			{ (void **)&state->split_partials, rows * heads * GLM_SPLIT_PARTITIONS * S::kPartialFloats * sizeof(float) },
			{ (void **)&state->selected, rows * S::kSelected * sizeof(uint32_t) },
			{ (void **)&state->topk_values, state->topk_entries * sizeof(float) },
			{ (void **)&state->topk_positions, state->topk_entries * sizeof(uint32_t) },
			{ (void **)&state->route_expert, packed * sizeof(uint32_t) },
			{ (void **)&state->route_weight, packed * sizeof(float) },
			{ (void **)&state->route_source, packed * sizeof(uint32_t) },
			{ (void **)&state->route_packed, packed * sizeof(uint32_t) },
			{ (void **)&state->group_offset, (S::kExperts + 1u) * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_w1, (S::kExperts + 1u) * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_w2, (S::kExperts + 1u) * sizeof(uint32_t) },
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
			{ (void **)&state->shard_send, (uint64_t)state->indexing_units_max * S::kUnitBytes },
			{ (void **)&state->shard_received, degree * state->indexing_units_max * S::kUnitBytes },
			{ (void **)&state->shard_partials, degree * state->partial_units_max * S::kUnitBytes },
			{ (void **)&state->shard_partials_received, degree * state->partial_units_max * S::kUnitBytes },
			{ (void **)&state->index_bitmaps, (uint64_t)state->scatter_rows * 2u * LmIndexSelectBitmapWords(state->max_positions) * sizeof(uint32_t) },
			{ (void **)&state->latent_pack, state->latent_bytes },
			{ (void **)&state->latent_pool, degree * state->latent_bytes + rows * S::kSlotBytes },
			{ (void **)&state->latent_remap, (uint64_t)state->max_positions * sizeof(uint32_t) },
			{ (void **)&state->select_local, (uint64_t)state->select_rows_max * S::kSelected * sizeof(uint32_t) },
			{ (void **)&state->select_candidates, (uint64_t)state->select_rows_max * S::kSelected * sizeof(uint64_t) },
			{ (void **)&state->select_received, (uint64_t)state->select_rows_max * S::kSelected * sizeof(uint64_t) },
			{ (void **)&state->select_merged, (uint64_t)state->select_rows_max / degree * S::kSelected * sizeof(uint32_t) },
			{ (void **)&state->rows_zero, rows * sizeof(uint32_t) },
			{ (void **)&state->plan, ((uint64_t)GLM_PLAN_ARRAYS * state->slots + GLM_PLAN_WORDS) * sizeof(uint32_t) },
			{ (void **)&state->digest, GLM_DIGEST_WORDS * sizeof(unsigned long long) },
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
		uint64_t layer_seen[S::kLayers], global_seen = 0u, directory_end;
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
			header->first_layer_index != 0u || header->layer_count != S::kLayers || header->total_layer_count != S::kLayers ||
			header->hidden_dimension != S::kHidden || header->vocab_count != S::kVocab || header->routed_expert_count != S::kExperts )
		{
			fprintf(stderr, "%s: pack %s is tp %u rank %u, %u tensors, stage %u/%u layers %u+%u; the runner is tp %u rank %u\n", G::Tag(), path,
				SparkGlm52StagePackHeaderTpDegree(header), SparkGlm52StagePackHeaderTpRank(header), header->tensor_count, header->stage_index,
				header->stage_count, header->first_layer_index, header->layer_count, state->tp_degree, state->tp_rank);
			fclose(file);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if ( header->linear_weight_codec != S::kLinearCodec || header->expert_weight_codec != S::kExpertCodec ||
			header->kv_cache_codec != SPARK_WEIGHT_CODEC_BF16 )
		{
			fclose(file);
			return PackFail(path, "codecs are not fp8 linear, fp8 experts, bf16 kv");
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
			if ( SparkGlm52StagePackExpectedShapeFor(S::PackGeometry(), entry->tensor_kind, entry->layer_index, S::kLinearCodec, S::kExpertCodec, state->tp_degree, &shape) < 0 ||
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
		for ( uint32_t layer = 0u; layer < S::kLayers; layer++ )
		{
			uint64_t expected = 0u;
			SparkGlm52StagePackTensorShape shape;
			for ( uint32_t kind = SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM; kind < GLM_TENSOR_KINDS; kind++ )
				if ( SparkGlm52StagePackExpectedShapeFor(S::PackGeometry(), kind, layer, S::kLinearCodec, S::kExpertCodec, state->tp_degree, &shape) == 0 )
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
		if ( state->rope_table != 0 )
			(void)cudaFree(state->rope_table);
		if ( state->host_staging != 0 )
			(void)cudaFreeHost(state->host_staging);
		if ( state->lane_arena != 0 )
			(void)cudaFree(state->lane_arena);
		if ( state->lane_host_staging != 0 )
			(void)cudaFreeHost(state->lane_host_staging);
		if ( state->error_words != 0 )
			(void)cudaFreeHost(state->error_words);
		for ( uint32_t lane = 0u; lane < SPARK_STAGE_RUNNER_LANES; ++lane )
			free(state->lane_states[lane]);
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
		state->heads = S::kHeads / state->tp_degree;
		state->max_rows = configuration->max_input_row_count;
		state->scatter_rows = state->max_rows < GLM_SCATTER_ROWS ? state->max_rows : GLM_SCATTER_ROWS;
		state->slots = configuration->resident_sequence_capacity > configuration->max_active_sequence_count ?
			configuration->resident_sequence_capacity : configuration->max_active_sequence_count;
		if ( configuration->kv_pages_per_sequence == 0u || (uint64_t)configuration->kv_pages_per_sequence * GLM_PAGE_SLOTS > S::kMaxPositions )
		{
			fprintf(stderr, "%s: %u KV pages per sequence give %llu positions; the model allows 1 to %u\n", G::Tag(), configuration->kv_pages_per_sequence,
				(unsigned long long)configuration->kv_pages_per_sequence * GLM_PAGE_SLOTS, S::kMaxPositions);
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
		state->selection_rows = LmTopkExactGatherSelectionRows(state->max_rows, state->max_positions, state->tp_degree);
		state->topk_entries = LmTopkExactScratchEntries(state->selection_rows, state->max_positions, S::kSelected);
		state->union_entries = state->max_positions > S::kSelected ?
			(uint64_t)GLM_PREFILL_TABLE_BLOCKS(state->max_rows) * GLM_PREFILL_BLOCK_ROWS * S::kSelected : 0u;
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
		if ( 2u * state->arena_bytes > configuration->state_budget_bytes )
		{
			fprintf(stderr, "%s: rank state %llu bytes for two lanes is over the %llu-byte budget\n", G::Tag(), 2ull * (unsigned long long)state->arena_bytes,
				(unsigned long long)configuration->state_budget_bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		embed = Find(state, SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING, SPARK_GLM52_STAGEPACK_GLOBAL_LAYER);
		norm = Find(state, SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM, SPARK_GLM52_STAGEPACK_GLOBAL_LAYER);
		head = Find(state, SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD, SPARK_GLM52_STAGEPACK_GLOBAL_LAYER);
		memset(geometry, 0, sizeof(*geometry));
		geometry->hidden = S::kHidden;
		geometry->vocab = S::kVocab;
		geometry->total_layers = S::kLayers;
		geometry->first_layer = 0u;
		geometry->layer_count = S::kLayers;
		geometry->rms_epsilon = S::kRmsEpsilon;
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
		geometry->kv_layer_count = S::kLayers;
		geometry->kv_layer_page_bytes = (uint64_t)GLM_PAGE_SLOTS * S::kSlotBytes;
		geometry->kv_second_layer_count = G::kIndexLayers;
		geometry->kv_second_layer_page_bytes = GlmIndexKv<S>::kPageBytes;
		geometry->kv_shard = state->shard;
		geometry->experts = S::kExperts;
		geometry->top_k = S::kTopK;
		for ( uint32_t layer = S::kFirstRouted; layer < S::kLayers; layer++ )
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

	static SparkStatus SliceLinear(SparkWeightdLazyPack *lazy_pack, const SparkGlm52StagePackEntry *entry, GlmLinearWeight *field)
	{
		const void *payload = 0, *scale = 0;
		SparkGlm52StagePackEntry scales = *entry;
		scales.payload_offset = entry->scale_offset;
		scales.payload_bytes = entry->scale_bytes;
		if ( entry->weight_codec != S::kLinearCodec || entry->scale_bytes == 0u || Slice(lazy_pack, entry, &payload) != SPARK_STATUS_OK ||
			Slice(lazy_pack, &scales, &scale) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		field->weight = (const uint8_t *)payload;
		field->scale = (const float *)scale;
		return SPARK_STATUS_OK;
	}

	static GlmLinearWeight *LinearField(GlmLayer *layer, uint32_t kind)
	{
		switch ( kind )
		{
		case SPARK_GLM52_STAGEPACK_TENSOR_Q_A: return &layer->q_a;
		case SPARK_GLM52_STAGEPACK_TENSOR_Q_B: return &layer->q_b;
		case SPARK_GLM52_STAGEPACK_TENSOR_KV_A: return &layer->kv_a;
		case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_OUTPUT: return &layer->o_proj;
		case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q: return &layer->index_q;
		case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_K: return &layer->index_k;
		case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP: return &layer->dense_gate_up;
		case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN: return &layer->dense_down;
		case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_GATE_UP: return &layer->shared_gate_up;
		case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN: return &layer->shared_down;
		default: return 0;
		}
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
		if ( SparkGlm52StagePackKindIsLinear(entry->tensor_kind) != 0u )
			return SliceLinear(lazy_pack, entry, LinearField(layer, entry->tensor_kind));
		if ( Slice(lazy_pack, entry, &pointer) != SPARK_STATUS_OK )
			return SPARK_STATUS_PARSE_ERROR;
		switch ( entry->tensor_kind )
		{
		case SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM: layer->attn_norm = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_Q_A_NORM: layer->q_a_norm = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_KV_A_NORM: layer->kv_a_norm = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED: layer->kv_b_key = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_KV_B_VALUE: layer->kv_b_value = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_POST_ATTN_NORM: layer->post_norm = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_HEAD: layer->index_head = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT: layer->index_norm_weight = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_BIAS: layer->index_norm_bias = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_ROUTER: layer->router = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_ROUTER_CORRECTION: layer->router_bias = (const float *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM_GATE_DOWN: layer->attn_gate_down = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM_GATE_UP: layer->attn_gate_up = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_POST_NORM_GATE_DOWN: layer->post_gate_down = (const uint16_t *)pointer; break;
		case SPARK_GLM52_STAGEPACK_TENSOR_POST_NORM_GATE_UP: layer->post_gate_up = (const uint16_t *)pointer; break;
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
		for ( uint32_t layer = 0u; layer < S::kLayers; layer++ )
			state->layers[layer].index_ordinal = GlmFullIndexer<S>(layer) != 0u ? ordinal++ : UINT32_MAX;
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
			cudaHostAlloc((void **)&state->host_staging, 3u * (uint64_t)state->max_rows * sizeof(uint32_t) + 64u,
				cudaHostAllocPortable) != cudaSuccess )
		{
			state->host_staging = 0;
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		Staging(state, state->host_staging);
		if ( cudaHostAlloc((void **)&state->error_words, SPARK_STAGE_RUNNER_LANES * sizeof(LmKvAccessError), cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess )
		{
			state->error_words = 0;
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		memset(state->error_words, 0, SPARK_STAGE_RUNNER_LANES * sizeof(LmKvAccessError));
		if ( cudaHostGetDevicePointer((void **)&state->error_device, state->error_words, 0u) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		state->kv_error = state->error_device;
		if ( cudaMalloc(&state->lane_arena, state->arena_bytes) != cudaSuccess )
		{
			state->lane_arena = 0;
			fprintf(stderr, "%s: %llu bytes of second lane state do not fit\n", G::Tag(), (unsigned long long)state->arena_bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		if ( cudaMemset(state->lane_arena, 0, state->arena_bytes) != cudaSuccess ||
			cudaHostAlloc((void **)&state->lane_host_staging, 3u * (uint64_t)state->max_rows * sizeof(uint32_t) + 64u,
				cudaHostAllocPortable) != cudaSuccess )
		{
			state->lane_host_staging = 0;
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		for ( uint32_t lane = 0u; lane < SPARK_STAGE_RUNNER_LANES; ++lane )
			if ( (state->lane_states[lane] = calloc(1u, sizeof(State))) == 0 )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if constexpr ( S::kRopeTable )
		{
			float frequencies[S::kRope / 2u];
			GlmYarnFrequencies(frequencies, S::kRope, S::kRopeTheta, S::kRopeFactor, S::kRopeBetaFast, S::kRopeBetaSlow, S::kRopeOriginalPositions);
			if ( cudaMalloc((void **)&state->rope_table, sizeof(frequencies)) != cudaSuccess ||
				cudaMemcpy(state->rope_table, frequencies, sizeof(frequencies), cudaMemcpyHostToDevice) != cudaSuccess )
			{
				fprintf(stderr, "%s: the rope frequency table does not fit\n", G::Tag());
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			}
		}
		fprintf(stderr, "%s rank=%u arena_bytes=%llu rows=%u slots=%u positions=%u scatter_units=%u/%u/%u gather_latent=%u x %u units select_rows=%u\n",
			G::Tag(), state->tp_rank, (unsigned long long)state->arena_bytes, state->max_rows, state->slots, state->max_positions,
			state->query_units_max, state->indexing_units_max, state->partial_units_max, state->latent_max.chunks, state->latent_max.chunk_units,
			state->select_rows_max);
		return SPARK_STATUS_OK;
	}

	static void Staging(State *state, uint32_t *staging)
	{
		state->host_staging = staging;
		state->host_order = staging;
		state->host_positions = state->host_order + state->max_rows;
		state->host_slots = state->host_positions + state->max_rows;
	}

	static void LaneFold(State *state)
	{
		for ( uint32_t lane = 0u; lane < SPARK_STAGE_RUNNER_LANES; ++lane )
		{
			State *copy = (State *)state->lane_states[lane];
			if ( state->lane_bound[lane] == 0u || copy == 0 )
				continue;
			state->steps += copy->steps - state->lane_base[lane][0];
			state->waves += copy->waves - state->lane_base[lane][1];
			state->gather_waves += copy->gather_waves - state->lane_base[lane][2];
			state->lane_base[lane][0] = copy->steps;
			state->lane_base[lane][1] = copy->waves;
			state->lane_base[lane][2] = copy->gather_waves;
		}
	}

	static SparkStatus Lane(void *model, uint32_t lane, const SparkStageRunnerStep *step, const SparkStageRunnerServices *services, void **lane_model)
	{
		State *state = (State *)model;
		State *copy;
		GlmRegime regime;
		uint64_t bytes = 0u;
		if ( lane >= SPARK_STAGE_RUNNER_LANES || step == 0 || services == 0 || lane_model == 0 || state->lane_states[lane] == 0 ||
			state->lane_arena == 0 || state->lane_host_staging == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( step->last_rows_only == 0u || step->host_positions == 0 || step->host_sequence_of_row == 0 || step->rows < GLM_LANE_ROWS ||
			step->rows > state->max_rows )
			return SPARK_STATUS_UNSUPPORTED;
		regime.positions = step->host_positions;
		regime.max_positions = state->max_positions;
		if ( SparkRowLayoutPackedSpanWaveRowCount(0u, step->rows, step->host_sequence_of_row, SelectionRegime, &regime, state->max_rows, 1u) != step->rows )
			return SPARK_STATUS_UNSUPPORTED;
		LaneFold(state);
		copy = (State *)state->lane_states[lane];
		memcpy(copy, state, sizeof(*copy));
		copy->services = services;
		if ( lane != 0u )
		{
			if ( Plan(copy, &bytes, (uint8_t *)state->lane_arena) != SPARK_STATUS_OK || bytes != state->arena_bytes )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			Staging(copy, state->lane_host_staging);
			copy->kv_error = state->error_device + lane;
		}
		state->lane_bound[lane] = 1u;
		state->lane_base[lane][0] = copy->steps;
		state->lane_base[lane][1] = copy->waves;
		state->lane_base[lane][2] = copy->gather_waves;
		*lane_model = copy;
		return SPARK_STATUS_OK;
	}

	static SparkStatus AttachKv(void *model, const SparkStageRunnerKv *kv)
	{
		State *state = (State *)model;
		if ( kv == 0 || kv->pool == 0 || kv->page_table == 0 || kv->pool_page_count == 0u || kv->sequence_count == 0u ||
			kv->page_table_stride < state->max_positions / GLM_PAGE_SLOTS ||
			state->shard.degree == 0u || kv->second_layer_stride_bytes < (uint64_t)kv->pool_page_count * (GlmIndexKv<S>::kPageBytes / state->shard.degree) ||
			kv->layer_count != S::kLayers || kv->layer_page_bytes != (uint64_t)GLM_PAGE_SLOTS * S::kSlotBytes ||
			kv->layer_stride_bytes != (uint64_t)GLM_PAGE_SLOTS * S::kSlotBytes / state->tp_degree || kv->second_pool == 0 ||
			kv->second_layer_count != G::kIndexLayers || kv->second_layer_page_bytes != GlmIndexKv<S>::kPageBytes || kv->sequence_count > state->slots ||
			kv->context_shard.degree != state->shard.degree || kv->context_shard.rank != state->shard.rank || kv->context_shard.grain != state->shard.grain )
		{
			fprintf(stderr, "%s: KV attach refused: the binding is not the page-major latent region plus the layer-major index region of this rank, "
				"with a page table of at least %u pages per sequence and index layers spanning every pool page\n", G::Tag(), state->max_positions / GLM_PAGE_SLOTS);
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
		b->q_b_rows = state->heads * S::kQueryHead;
		b->attn_output_columns = state->heads * S::kValue;
		b->dense_gate_up_rows = 2u * S::kDenseInter / state->tp_degree;
		b->dense_intermediate = S::kDenseInter / state->tp_degree;
		b->expert_w1_rows = 2u * S::kExpertInter / state->tp_degree;
		b->expert_intermediate = S::kExpertInter / state->tp_degree;
		b->shared_gate_up_rows = 2u * S::kExpertInter / state->tp_degree;
		b->shared_intermediate = S::kExpertInter / state->tp_degree;
		b->weights = weights;
		b->hidden_bf16 = hidden;
		b->residual_bf16 = state->residual;
		b->normed_bf16 = state->normed;
		b->q_compressed_bf16 = state->q_compressed;
		b->q_input_bf16 = state->q_input;
		b->rope_table = state->rope_table;
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
		b->gate_bottleneck = state->gate_bottleneck;
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
		b->selected_position_count = S::kSelected;
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
			b->kv_gather_latent_tail = state->latent_pool + state->tail_slot * S::kSlotBytes;
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
		b->shard_rank_stride_bytes = (uint64_t)(GlmShardIndexing<S>(layer, wave->context) != 0u ? wave->indexing_units : wave->query_units) * S::kUnitBytes;
		b->shard_partials = state->shard_partials;
		b->shard_partials_received = state->shard_partials_received;
		b->shard_partial_stride = (uint64_t)wave->partial_units * (S::kUnitBytes / sizeof(float));
		b->index_bitmaps = state->index_bitmaps;
		b->index_bitmap_words = LmIndexSelectBitmapWords(state->max_positions);
		b->index_error = state->kv_error;
	}

	static SparkStatus Fail(const char *site, uint32_t layer, int32_t code)
	{
		fprintf(stderr, "%s: %s failed layer=%u code=%d cuda=%s\n", G::Tag(), site, layer, code, cudaGetErrorString(cudaGetLastError()));
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}

	static void RowState(const State *state, uint32_t layer, uint32_t phase)
	{
		if ( state->services->row_state != 0 )
			state->services->row_state(state->services->context, layer, phase);
	}

	static SparkStatus GatherExchange(State *state, const GlmWave *wave, GlmBuffers *b, uint32_t layer, cudaStream_t stream)
	{
		SparkStatus status = SPARK_STATUS_OK;
		if ( GlmShardIndexing<S>(layer, wave->context) != 0u )
		{
			if ( wave->select_slice > state->selection_rows || wave->context > state->max_positions )
			{
				fprintf(stderr, "%s: gather wave scores %u rows of %u positions but the selection buffer holds %u rows of %u (refused)\n",
					G::Tag(), wave->select_slice, wave->context, state->selection_rows, state->max_positions);
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			}
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, 2u * wave->select_slice, 2u * wave->select_keep,
				state->select_candidates, state->select_received);
			if ( status != SPARK_STATUS_OK )
				return status;
			if ( LmIndexShardCandidateScatterLaunch<GLM_THREADS>((const uint2 *)state->select_received, (uint64_t)wave->select_slice * wave->select_keep,
					state->tp_degree, wave->select_keep, wave->select_slice, wave->context, state->selection_scores, stream) != cudaSuccess ||
				LmTopkExactLaunch<GLM_THREADS>(state->selection_scores, wave->select_slice, wave->context, S::kSelected, LM_TOPK_EXACT_CHUNK,
					LM_TOPK_EXACT_CHUNKED_ROWS, state->topk_values, state->topk_positions, state->topk_entries, state->select_merged, stream) != cudaSuccess )
				return Fail("kv select merge", layer, LM_LAUNCH_ERR_LAUNCH);
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, wave->select_slice, 2u * S::kSelected, state->select_merged,
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

	static uint32_t RowSharded(const GlmWave *wave)
	{
		return wave->ordered != 0u && wave->rows >= GLM_ROW_SHARD_ROWS ? 1u : 0u;
	}

	static SparkStatus RowShard(State *state, const GlmWave *wave, uint32_t layer, const uint16_t *partial, const uint16_t *weight, uint32_t keep_residual,
		const uint16_t *gate_down, const uint16_t *gate_up, cudaStream_t stream)
	{
		const GlmRowShare row = GlmRowShareOf(wave->rows, state->tp_degree, state->tp_rank);
		SparkStatus status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_REDUCE_SCATTER_SUM_BF16, row.share * state->tp_degree, 0u, partial,
			state->row_share);
		if ( status != SPARK_STATUS_OK )
			return status;
		GlmRowShareNorm<GLM_THREADS>(&row, state->row_share, state->residual, weight, keep_residual, state->row_normed, S::kHidden, S::kRmsEpsilon, stream);
		if ( GlmLaunched() != LM_LAUNCH_OK || GlmNormGate<S>(state->row_normed, gate_down, gate_up, state->gate_bottleneck, row.count, stream) != LM_LAUNCH_OK )
			return Fail("row shard norm", layer, LM_LAUNCH_ERR_LAUNCH);
		return Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, row.share, 0u, state->row_normed, state->normed);
	}

	static SparkStatus Attention(State *state, Frame &frame, uint32_t layer)
	{
		const GlmWave *wave = &frame.wave;
		const uint32_t rows = wave->rows, sharded = RowSharded(wave), sms = frame.step->multiprocessors;
		cudaStream_t stream = frame.stream;
		GlmBuffers *b = &frame.b;
		SparkStatus status;
		int32_t code;
		BindLayer(state, wave, layer, frame.hidden, frame.positions, frame.slots, b);
		b->prenormed = sharded != 0u && layer != 0u ? 1u : 0u;
		code = GlmAttentionProject<S>(b, rows, sms, stream);
		if ( code != LM_LAUNCH_OK )
			return Fail("attention projection", layer, code);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, GlmProjectionSliceWidth<S>(state->tp_degree),
			state->projection_local, state->projection_gather);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( wave->gather != 0u )
		{
			RowState(state, layer, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			code = GlmShardGatherPre<S>(b, rows, wave->context, layer, sms, stream);
			RowState(state, layer, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			if ( code != LM_LAUNCH_OK )
				return Fail("gather pre", layer, code);
			status = GatherExchange(state, wave, b, layer, stream);
			if ( status != SPARK_STATUS_OK )
				return status;
			code = GlmShardGatherPost<S>(b, rows, wave->context, sms, stream);
			if ( code != LM_LAUNCH_OK )
				return Fail("gather post", layer, code);
		}
		else
		{
			RowState(state, layer, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			code = GlmShardScatterPre<S>(b, rows, wave->context, layer, sms, stream);
			if ( code != LM_LAUNCH_OK )
				return Fail("scatter pre", layer, code);
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER,
				GlmShardIndexing<S>(layer, wave->context) != 0u ? wave->indexing_units : wave->query_units, 0u, state->shard_send, state->shard_received);
			if ( status != SPARK_STATUS_OK )
				return status;
			code = GlmShardScatterMid<S>(b, rows, wave->context, layer, stream);
			RowState(state, layer, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			if ( code != LM_LAUNCH_OK )
				return Fail("scatter mid", layer, code);
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, wave->partial_units, 0u, state->shard_partials,
				state->shard_partials_received);
			if ( status != SPARK_STATUS_OK )
				return status;
			code = GlmShardScatterPost<S>(b, rows, sms, stream);
			if ( code != LM_LAUNCH_OK )
				return Fail("scatter post", layer, code);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus AttentionResidual(State *state, Frame &frame, uint32_t layer)
	{
		const GlmWave *wave = &frame.wave;
		return RowSharded(wave) != 0u ? RowShard(state, wave, layer, state->attention_out, state->layers[layer].post_norm, 1u,
			state->layers[layer].post_gate_down, state->layers[layer].post_gate_up, frame.stream) :
			Round(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, wave->rows, 0u, state->attention_out, state->attention_out);
	}

	static SparkStatus DenseMlp(State *state, Frame &frame, uint32_t layer)
	{
		int32_t code;
		frame.b.prenormed = RowSharded(&frame.wave);
		code = GlmDenseMlp<S>(&frame.b, frame.wave.rows, frame.step->multiprocessors, frame.stream);
		if ( code != LM_LAUNCH_OK )
			return Fail("mlp", layer, code);
		return SPARK_STATUS_OK;
	}

	static SparkStatus RoutedMlp(State *state, Frame &frame, uint32_t layer)
	{
		const uint32_t rows = frame.wave.rows, packed = rows * S::kTopK, sms = frame.step->multiprocessors;
		cudaStream_t stream = frame.stream;
		GlmBuffers *b = &frame.b;
		const void *w1 = 0, *w2 = 0;
		SparkStatus status;
		int32_t code;
		b->prenormed = RowSharded(&frame.wave);
		code = GlmMoeRoute<S>(b, rows, packed, sms, stream);
		if ( code != LM_LAUNCH_OK )
			return Fail("moe route", layer, code);
		status = state->services->expert_weights(state->services->context, layer, state->group_offset, packed, &w1, &w2);
		if ( status != SPARK_STATUS_OK || w1 == 0 || w2 == 0 )
		{
			fprintf(stderr, "%s: expert weights for layer %u are not available status=%d\n", G::Tag(), layer, (int)status);
			return status != SPARK_STATUS_OK ? status : SPARK_STATUS_INTERNAL_ERROR;
		}
		b->expert_w1_weight = w1;
		b->expert_w1_scale = (const uint8_t *)w1 - state->w1_offset[layer] + state->layers[layer].up_gate_scale;
		b->expert_w2_weight = w2;
		b->expert_w2_scale = (const uint8_t *)w2 - state->w2_offset[layer] + state->layers[layer].down_scale;
		code = GlmMoeExperts<S>(b, rows, packed, sms, stream);
		if ( code != LM_LAUNCH_OK )
			return Fail("mlp", layer, code);
		return SPARK_STATUS_OK;
	}

	static SparkStatus MlpResidual(State *state, Frame &frame, uint32_t layer)
	{
		const GlmWave *wave = &frame.wave;
		const uint32_t last = layer + 1u == S::kLayers ? 1u : 0u;
		return RowSharded(wave) != 0u ? RowShard(state, wave, layer, frame.hidden, last != 0u ? state->final_norm : state->layers[layer + 1u].attn_norm,
			last == 0u ? 1u : 0u, last != 0u ? 0 : state->layers[layer + 1u].attn_gate_down, last != 0u ? 0 : state->layers[layer + 1u].attn_gate_up,
			frame.stream) : Round(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, wave->rows, 0u, frame.hidden, frame.hidden);
	}

	static SparkStatus LayerDone(State *state, Frame &frame, uint32_t layer)
	{
		(void)frame;
		if ( state->services->layer_done != 0 )
			state->services->layer_done(state->services->context, layer);
		return SPARK_STATUS_OK;
	}

	static uint32_t GatherCheaper(const State *state, const GlmWave *wave)
	{
		uint32_t query_units, indexing_units, partial_units;
		uint64_t candidate_offset;
		const uint64_t peers = state->tp_degree > 1u ? state->tp_degree - 1u : 0u;
		ScatterSizes(state, wave->rows, &query_units, &indexing_units, &partial_units, &candidate_offset);
		return SparkKvShardExchangeCostNs(wave->latent.chunks, peers * wave->latent.keys * S::kSlotBytes) <=
			SparkKvShardExchangeCostNs(2u, peers * ((uint64_t)indexing_units + partial_units) * S::kUnitBytes) ? 1u : 0u;
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
			wave->select_keep = GlmShardKeep<S>(state->shard, wave->context);
			wave->select_slice = (wave->rows + state->tp_degree - 1u) / state->tp_degree;
			wave->select_rows = wave->select_slice * state->tp_degree;
			if ( wave->rows > state->scatter_rows || GatherCheaper(state, wave) != 0u )
				return;
			wave->gather = 0u;
		}
		ScatterSizes(state, wave->rows, &wave->query_units, &wave->indexing_units, &wave->partial_units, &wave->candidate_offset);
	}

	static SparkStatus WaveBegin(State *state, Frame &frame)
	{
		const GlmWave *wave = &frame.wave;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		const uint32_t rows = wave->rows;
		uint16_t *hidden;
		const uint32_t *positions, *slots;
		if ( wave->ordered != 0u )
		{
			hidden = state->hidden;
			positions = state->positions + wave->first;
			slots = state->wave_slots + wave->first;
			GlmRowsGatherKernel<<<rows, GLM_THREADS, 0, stream>>>(state->order + wave->first, step->hidden_bf16, hidden, S::kHidden);
		}
		else
		{
			hidden = step->hidden_bf16 + (uint64_t)wave->first * S::kHidden;
			positions = step->positions + wave->first;
			slots = step->sequence_of_row + wave->first;
		}
		if ( cudaMemsetAsync(state->residual, 0, (size_t)rows * S::kHidden * sizeof(uint16_t), stream) != cudaSuccess )
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
		frame.hidden = hidden;
		frame.positions = positions;
		frame.slots = slots;
		return SPARK_STATUS_OK;
	}

	static SparkStatus WaveEnd(State *state, Frame &frame)
	{
		const GlmWave *wave = &frame.wave;
		const SparkStageRunnerStep *step = frame.step;
		cudaStream_t stream = frame.stream;
		const uint32_t rows = wave->rows;
		uint16_t *hidden = frame.hidden;
		SparkStatus status;
		if ( wave->gather != 0u )
		{
			if ( LmRowDigestFinishLaunch(state->digest, GLM_DIGESTS, state->digest + GLM_DIGESTS, stream) != cudaSuccess )
				return Fail("kv digest", S::kLayers, LM_LAUNCH_ERR_LAUNCH);
			status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64, 2u * GLM_DIGESTS, 0u, state->digest + GLM_DIGESTS,
				state->digest + GLM_DIGESTS);
			if ( status != SPARK_STATUS_OK )
				return status;
			if ( LmRowDigestCheckLaunch(state->digest + GLM_DIGESTS, GLM_DIGESTS, state->kv_error, stream) != cudaSuccess )
				return Fail("kv digest check", S::kLayers, LM_LAUNCH_ERR_LAUNCH);
		}
		if ( RowSharded(wave) == 0u )
			LM_LAUNCH((LmFusedResidualRmsNormKernel<GLM_THREADS,uint16_t>), rows, GLM_THREADS, (S::kHidden + 8u) * sizeof(float), stream,
				hidden, state->residual, state->final_norm, 0, state->normed, S::kHidden, S::kHidden, S::kRmsEpsilon);
		if ( wave->ordered != 0u )
			GlmRowsScatterKernel<<<rows, GLM_THREADS, 0, stream>>>(state->order + wave->first, state->normed, step->hidden_bf16, S::kHidden);
		else if ( cudaMemcpyAsync(hidden, state->normed, (size_t)rows * S::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( cudaPeekAtLastError() != cudaSuccess )
			return Fail("final norm", S::kLayers, LM_LAUNCH_ERR_LAUNCH);
		state->waves++;
		state->gather_waves += wave->gather;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Check(void *model)
	{
		const State *state = (const State *)model;
		SparkStatus status = SPARK_STATUS_OK;
		for ( uint32_t lane = 0u; state->error_words != 0 && lane < SPARK_STAGE_RUNNER_LANES; ++lane )
		{
			const volatile LmKvAccessError *error = state->error_words + lane;
			if ( error->error_code == 0u )
				continue;
			fprintf(stderr, "%s: KV access error lane=%u code=%u kind=%u row=%u sequence=%u position=%u page=%u\n", G::Tag(), lane, error->error_code,
				error->access_kind, error->row, error->sequence, error->position, error->page);
			status = SPARK_STATUS_INTERNAL_ERROR;
		}
		if ( status != SPARK_STATUS_OK )
			SPARK_FAIL(status);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream_void, Frame *frame)
	{
		cudaStream_t stream = (cudaStream_t)stream_void;
		const uint32_t rows = step->rows;
		const uint32_t ordered = step->last_rows_only != 0u || (rows != step->sequences && (FrameShape(step) == 0u || step->graph_key != 2u)) ? 1u : 0u;
		frame->step = step;
		frame->stream = stream;
		frame->ordered = ordered;
		frame->first = 0u;
		memset(&frame->wave, 0, sizeof(frame->wave));
		if ( rows == 0u || rows > state->max_rows || step->sequences > state->slots || state->kv_attached == 0u || step->positions == 0 ||
			step->sequence_of_row == 0 || (ordered != 0u && (step->host_positions == 0 || step->host_sequence_of_row == 0)) )
		{
			fprintf(stderr, "%s: step refused rows=%u sequences=%u kv=%u ordered=%u host_rows=%s\n", G::Tag(), rows, step->sequences, state->kv_attached,
				ordered, step->host_positions != 0 && step->host_sequence_of_row != 0 ? "yes" : "missing");
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
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
			frame->regime.positions = state->host_positions;
			frame->regime.max_positions = state->max_positions;
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus NextWave(State *state, Frame &frame, bool *more)
	{
		const uint32_t rows = frame.step->rows;
		GlmWave *wave = &frame.wave;
		frame.first += wave->rows;
		*more = frame.first < rows;
		if ( !*more )
			return SPARK_STATUS_OK;
		memset(wave, 0, sizeof(*wave));
		wave->first = frame.first;
		wave->ordered = frame.ordered;
		if ( frame.ordered != 0u )
		{
			wave->rows = SparkRowLayoutPackedSpanWaveRowCount(frame.first, rows, state->host_slots, SelectionRegime, &frame.regime, state->max_rows, 1u);
			if ( wave->rows <= LM_SKINNY_ROWS )
				wave->rows = SparkRowLayoutPackedSpanWaveRowCount(frame.first, rows, state->host_slots, RowRegime, &frame.regime, state->max_rows, 1u);
		}
		else
			wave->rows = rows - frame.first < state->scatter_rows ? rows - frame.first : state->scatter_rows;
		if ( wave->rows == 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		Measure(state, wave, state->host_positions);
		return SPARK_STATUS_OK;
	}

	static SparkStatus End(State *state, Frame &frame)
	{
		state->steps++;
		if ( frame.ordered != 0u && cudaStreamSynchronize(frame.stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *stream)
	{
		return GlmSchedule<G, GlmModel>::Step((State *)model, step, stream);
	}

	static uint32_t FrameShape(const SparkStageRunnerStep *step)
	{
		return step->last_rows_only == 0u && step->sequences == 1u && step->rows > 1u && step->rows <= LM_SKINNY_ROWS ? 1u : 0u;
	}

	static uint32_t GraphKey(void *model, const SparkStageRunnerStep *step)
	{
		(void)model;
		if ( step->last_rows_only == 0u && step->rows == step->sequences )
			return 1u;
		return FrameShape(step) != 0u ? 2u : 0u;
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
		State *state = (State *)model;
		LaneFold(state);
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
			GraphKey,
			Lane,
			0,
			0,
			0,
			Check
		};
		return &interface_table;
	}
};
