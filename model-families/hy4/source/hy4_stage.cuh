#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/launch.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_named_pack.h"
#include "sparkpipe/spark_named_pack_bind.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"
#include "inference/kernels/formats/mxfp8.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/scale.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/topk_exact_plan.h"
#include "inference/kernels/topk_group.cuh"
#include "runtime/gemm.cuh"

#define HY_THREADS 256u
#define HY_HC_ROWS 4u
#define HY_SCALE_GROUP 32u
#define HY_PACK_ALIGNMENT 128u
#define HY_SELECT_CHUNK 64u
#define HY_HC_LANES 256u
#define HY_HC_SLICE 32u
#define HY_HC_LOADERS 128u
#define HY_HC_SPLIT_ROWS 16u

template<class G>
struct HyShape
{
	static constexpr uint32_t kRankHeads = G::kHeads / G::kTpDegree;
	static constexpr uint32_t kQkDim = G::kNope + G::kRope;
	static constexpr uint32_t kSlot = G::kLatent + G::kRope;
	static constexpr uint32_t kRecord = LM_LATENT_SHARD_RECORD_FLOATS(G::kLatent);
	static constexpr uint32_t kResidual = G::kStreams * G::kHidden;
	static constexpr uint32_t kIndexWidth = G::kIndexHeads * G::kIndexDim;
	static constexpr uint32_t kCandidateSplit = (G::kIndexTopk * 4u + G::kHidden - 1u) / G::kHidden;
	using Kv = LmKvGeometry<kSlot * 2u, G::kPageSlots, true>;
	using IndexKv = LmKvGeometry<G::kIndexDim * 2u, G::kPageSlots, true>;
	static_assert(G::kHeads % G::kTpDegree == 0u && G::kExperts % G::kTpDegree == 0u, "heads and experts split evenly over the ranks");
	static_assert(kRankHeads * kSlot <= G::kHidden, "a rank's packed queries fit one collective row");
	static_assert(G::kIndexTopk % kCandidateSplit == 0u, "the candidate rows split evenly into collective rows");
};

typedef struct HyScaled
{
	const uint8_t *weight;
	const uint8_t *scale;
} HyScaled;

typedef struct HyLayer
{
	const float *attn_hc_fn;
	const float *attn_hc_base;
	const float *attn_hc_scale;
	const float *mlp_hc_fn;
	const float *mlp_hc_base;
	const float *mlp_hc_scale;
	const uint16_t *input_norm;
	const uint16_t *post_norm;
	HyScaled q_a;
	const uint16_t *q_norm;
	HyScaled q_b;
	HyScaled kv_a;
	const uint16_t *kv_norm;
	const uint16_t *uk;
	const uint16_t *uv;
	const uint16_t *gate;
	HyScaled out;
	const float *sinks;
	HyScaled index_q;
	HyScaled index_k;
	const float *index_k_norm;
	const float *index_k_bias;
	const uint16_t *index_weights;
	HyScaled ffn_gate_up;
	HyScaled ffn_down;
	const uint16_t *router;
	const float *router_bias;
	HyScaled moe_w1;
	HyScaled moe_w3;
	HyScaled moe_w2;
	HyScaled shared_gate_up;
	HyScaled shared_down;
	uint32_t index_slot;
} HyLayer;

template<class G>
struct HyState
{
	const SparkStageRunnerServices *services;
	SparkNamedPack pack;
	uint32_t pack_open;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t max_rows;
	uint32_t slots;
	uint32_t ffn_rows;
	uint32_t shared_rows;
	uint32_t expert_first;
	uint32_t context_capacity;
	uint32_t local_capacity;
	uint64_t topk_entries;
	uint64_t partial_stride;
	HyLayer layers[G::kLayers];
	const float *head_hc_fn;
	const float *head_hc_base;
	const float *head_hc_scale;
	void *arena;
	uint64_t arena_bytes;
	uint16_t *streams;
	float *hc_post;
	float *hc_partial;
	uint16_t *mixed;
	uint16_t *normed;
	uint16_t *partial;
	uint16_t *q_resid;
	uint16_t *q;
	uint16_t *q_latent;
	uint16_t *query;
	uint16_t *kv;
	uint16_t *gathered;
	float *send;
	float *received;
	uint16_t *latent;
	uint16_t *value;
	uint16_t *gate;
	uint16_t *index_q;
	uint16_t *index_k;
	float *index_weights;
	float *index_scores;
	float *index_topk_values;
	uint32_t *index_topk_positions;
	uint32_t *index_local_selected;
	uint2 *index_candidates;
	uint2 *index_gathered;
	float *index_full_scores;
	uint32_t *selected;
	uint16_t *gate_up;
	uint16_t *intermediate;
	float *router_logits;
	uint32_t *route_expert;
	uint32_t *route_global;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	uint32_t *tile_prefix_up;
	uint32_t *tile_prefix_down;
	float *packed_weight;
	uint16_t *expert_gate;
	uint16_t *expert_up;
	uint16_t *expert_act;
	uint16_t *expert_out;
	uint16_t *shared_gate_up;
	uint16_t *shared_act;
	uint16_t *shared_out;
	LmKvAccessError *kv_error;
	LmKvShardView kv_views[G::kLayers];
	LmKvShardView index_views[G::kIndexLayers];
	uint32_t kv_attached;
};

__global__ static void HyExpandStreamsKernel(const uint16_t *hidden, uint16_t *streams, uint32_t streams_count, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < streams_count * width; index += gridDim.x * blockDim.x )
		streams[(uint64_t)row * streams_count * width + index] = hidden[(uint64_t)row * width + index % width];
}

template<uint32_t STREAMS, uint32_t MIXES>
static __device__ __forceinline__ void HyHcWeights(const float *total, uint32_t flat, const float *base, const float *scale, float norm_epsilon,
	float hc_epsilon, float post_magnitude, float *weights, float *post)
{
	const float inverse = rsqrtf(total[MIXES] / (float)flat + norm_epsilon);
	for ( uint32_t s = 0u; s < STREAMS; ++s )
	{
		weights[s] = SparkMathSigmoid(total[s] * inverse * scale[0] + base[s]) + hc_epsilon;
		if ( MIXES > STREAMS && post != 0 )
			post[s] = SparkMathScaledSigmoid(post_magnitude,total[STREAMS + s] * inverse * scale[1] + base[STREAMS + s]) + hc_epsilon;
	}
}

template<uint32_t THREADS, uint32_t STREAMS, uint32_t MIXES>
__global__ __launch_bounds__(THREADS, 1)
void HyHcPreKernel(const uint16_t *streams, const float *fn, const float *base, const float *scale, float *post, uint16_t *mixed,
	uint32_t rows, uint32_t width, float norm_epsilon, float hc_epsilon, float post_magnitude)
{
	constexpr uint32_t WARPS = THREADS / 32u;
	__shared__ float reduction[HY_HC_ROWS][MIXES + 1u][WARPS];
	__shared__ float weights[HY_HC_ROWS][STREAMS];
	const uint32_t first = blockIdx.x * HY_HC_ROWS, flat = STREAMS * width, lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	float dot[HY_HC_ROWS][MIXES + 1u];
	for ( uint32_t r = 0u; r < HY_HC_ROWS; ++r )
		for ( uint32_t m = 0u; m <= MIXES; ++m )
			dot[r][m] = 0.0f;
	for ( uint32_t k = threadIdx.x; k < flat; k += THREADS )
	{
		float coefficient[MIXES];
		for ( uint32_t m = 0u; m < MIXES; ++m )
			coefficient[m] = fn[(uint64_t)m * flat + k];
		for ( uint32_t r = 0u; r < HY_HC_ROWS; ++r )
		{
			if ( first + r >= rows )
				break;
			const float x = LmBf16ToFloat(streams[(uint64_t)(first + r) * flat + k]);
			for ( uint32_t m = 0u; m < MIXES; ++m )
				dot[r][m] = fmaf(x, coefficient[m], dot[r][m]);
			dot[r][MIXES] = fmaf(x, x, dot[r][MIXES]);
		}
	}
	for ( uint32_t r = 0u; r < HY_HC_ROWS; ++r )
		for ( uint32_t m = 0u; m <= MIXES; ++m )
		{
			float value = dot[r][m];
			for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				value += __shfl_xor_sync(0xffffffffu, value, offset);
			if ( lane == 0u )
				reduction[r][m][warp] = value;
		}
	__syncthreads();
	if ( threadIdx.x < HY_HC_ROWS && first + threadIdx.x < rows )
	{
		const uint32_t r = threadIdx.x;
		float total[MIXES + 1u];
		for ( uint32_t m = 0u; m <= MIXES; ++m )
		{
			total[m] = 0.0f;
			for ( uint32_t w = 0u; w < WARPS; ++w )
				total[m] += reduction[r][m][w];
		}
		HyHcWeights<STREAMS,MIXES>(total, flat, base, scale, norm_epsilon, hc_epsilon, post_magnitude, weights[r],
			MIXES > STREAMS ? post + (uint64_t)(first + r) * STREAMS : (float *)0);
	}
	__syncthreads();
	for ( uint32_t r = 0u; r < HY_HC_ROWS; ++r )
	{
		if ( first + r >= rows )
			break;
		for ( uint32_t d = threadIdx.x; d < width; d += THREADS )
		{
			float total = 0.0f;
			for ( uint32_t s = 0u; s < STREAMS; ++s )
				total = fmaf(weights[r][s], LmBf16ToFloat(streams[(uint64_t)(first + r) * flat + s * width + d]), total);
			mixed[(uint64_t)(first + r) * width + d] = LmFloatToBf16(total);
		}
	}
}

template<uint32_t STREAMS, uint32_t WIDTH>
__global__ __launch_bounds__(HY_HC_LOADERS)
void HyHcDotKernel(const uint16_t *streams, const float *fn, float *partial, uint32_t mixes)
{
	constexpr uint32_t FLAT = STREAMS * WIDTH, STEPS = FLAT / HY_HC_LANES;
	static_assert(FLAT % HY_HC_LANES == 0u, "every lane walks the same number of columns");
	__shared__ float coefficient[STEPS][HY_HC_SLICE];
	__shared__ float value[STEPS][HY_HC_SLICE];
	const uint32_t mix = blockIdx.x, lane_first = blockIdx.y * HY_HC_SLICE, row = blockIdx.z;
	const uint16_t *source = streams + (uint64_t)row * FLAT;
	for ( uint32_t index = threadIdx.x; index < STEPS * HY_HC_SLICE; index += HY_HC_LOADERS )
	{
		const uint32_t step = index / HY_HC_SLICE, lane = index % HY_HC_SLICE, k = step * HY_HC_LANES + lane_first + lane;
		value[step][lane] = LmBf16ToFloat(source[k]);
		coefficient[step][lane] = mix < mixes - 1u ? fn[(uint64_t)mix * FLAT + k] : 0.0f;
	}
	__syncthreads();
	if ( threadIdx.x < HY_HC_SLICE )
	{
		float dot = 0.0f;
		if ( mix < mixes - 1u )
			for ( uint32_t step = 0u; step < STEPS; ++step )
				dot = fmaf(value[step][threadIdx.x], coefficient[step][threadIdx.x], dot);
		else
			for ( uint32_t step = 0u; step < STEPS; ++step )
				dot = fmaf(value[step][threadIdx.x], value[step][threadIdx.x], dot);
		partial[((uint64_t)row * mixes + mix) * HY_HC_LANES + lane_first + threadIdx.x] = dot;
	}
}

template<uint32_t STREAMS, uint32_t MIXES, uint32_t WIDTH>
__global__ __launch_bounds__(HY_HC_LANES, 1)
void HyHcMixKernel(const uint16_t *streams, const float *partial, const float *base, const float *scale, float *post, uint16_t *mixed,
	float norm_epsilon, float hc_epsilon, float post_magnitude)
{
	constexpr uint32_t WARPS = HY_HC_LANES / 32u, FLAT = STREAMS * WIDTH;
	__shared__ float reduction[MIXES + 1u][WARPS];
	__shared__ float weights[STREAMS];
	const uint32_t row = blockIdx.y, lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	for ( uint32_t m = 0u; m <= MIXES; ++m )
	{
		float value = partial[((uint64_t)row * (MIXES + 1u) + m) * HY_HC_LANES + threadIdx.x];
		for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
			value += __shfl_xor_sync(0xffffffffu, value, offset);
		if ( lane == 0u )
			reduction[m][warp] = value;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		float total[MIXES + 1u];
		for ( uint32_t m = 0u; m <= MIXES; ++m )
		{
			total[m] = 0.0f;
			for ( uint32_t w = 0u; w < WARPS; ++w )
				total[m] += reduction[m][w];
		}
		HyHcWeights<STREAMS,MIXES>(total, FLAT, base, scale, norm_epsilon, hc_epsilon, post_magnitude, weights,
			MIXES > STREAMS && blockIdx.x == 0u ? post + (uint64_t)row * STREAMS : (float *)0);
	}
	__syncthreads();
	for ( uint32_t d = blockIdx.x * HY_HC_LANES + threadIdx.x; d < WIDTH; d += gridDim.x * HY_HC_LANES )
	{
		float total = 0.0f;
		for ( uint32_t s = 0u; s < STREAMS; ++s )
			total = fmaf(weights[s], LmBf16ToFloat(streams[(uint64_t)row * FLAT + s * WIDTH + d]), total);
		mixed[(uint64_t)row * WIDTH + d] = LmFloatToBf16(total);
	}
}

__global__ static void HyHcPostKernel(uint16_t *streams, const float *post, const uint16_t *output, uint32_t streams_count, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < streams_count * width; index += gridDim.x * blockDim.x )
	{
		const uint64_t at = (uint64_t)row * streams_count * width + index;
		streams[at] = LmFloatToBf16(fmaf(post[(uint64_t)row * streams_count + index / width], LmBf16ToFloat(output[(uint64_t)row * width + index % width]),
			LmBf16ToFloat(streams[at])));
	}
}

template<uint32_t THREADS, uint32_t WIDTH>
__global__ __launch_bounds__(THREADS, 1)
void HyRmsNormKernel(const uint16_t *input, const uint16_t *weight, uint16_t *output, uint32_t input_stride, uint32_t output_stride, float epsilon)
{
	constexpr uint32_t PER_THREAD = (WIDTH + THREADS - 1u) / THREADS;
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	const uint16_t *source = input + (uint64_t)blockIdx.x * input_stride;
	uint16_t *target = output + (uint64_t)blockIdx.x * output_stride;
	float values[PER_THREAD];
	float total = 0.0f;
#pragma unroll
	for ( uint32_t step = 0u; step < PER_THREAD; ++step )
	{
		const uint32_t index = threadIdx.x + step * THREADS;
		values[step] = index < WIDTH ? LmBf16ToFloat(source[index]) : 0.0f;
	}
#pragma unroll
	for ( uint32_t step = 0u; step < PER_THREAD; ++step )
		if ( threadIdx.x + step * THREADS < WIDTH )
			total += values[step] * values[step];
	const float inverse = rsqrtf(LmBlockSum<THREADS>(total, reduction) / (float)WIDTH + epsilon);
#pragma unroll
	for ( uint32_t step = 0u; step < PER_THREAD; ++step )
	{
		const uint32_t index = threadIdx.x + step * THREADS;
		if ( index < WIDTH )
			target[index] = LmFloatToBf16(LmBf16ToFloat(weight[index]) * LmBf16ToFloat(LmFloatToBf16(values[step] * inverse)));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void HyLayerNormKernel(uint16_t *rows, const float *weight, const float *bias, uint32_t width, float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint16_t *row = rows + (uint64_t)blockIdx.x * width;
	float total = 0.0f, squares = 0.0f;
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
		total += LmBf16ToFloat(row[index]);
	const float mean = LmBlockSum<THREADS>(total, reduction) / (float)width;
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
	{
		const float centred = LmBf16ToFloat(row[index]) - mean;
		squares += centred * centred;
	}
	const float inverse = rsqrtf(LmBlockSum<THREADS>(squares, reduction) / (float)width + epsilon);
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
		row[index] = LmFloatToBf16((LmBf16ToFloat(row[index]) - mean) * inverse * weight[index] + bias[index]);
}

__global__ static void HyQueryPackKernel(const uint16_t *latent, const uint16_t *q, uint16_t *query, uint32_t heads, uint32_t latent_dim,
	uint32_t nope, uint32_t rope)
{
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	const uint64_t pair = (uint64_t)row * heads + head;
	for ( uint32_t element = threadIdx.x; element < latent_dim + rope; element += blockDim.x )
		query[pair * (latent_dim + rope) + element] = element < latent_dim ? latent[pair * latent_dim + element] :
			q[pair * (nope + rope) + nope + element - latent_dim];
}

__global__ static void HySigmoidGateKernel(uint16_t *values, const uint16_t *gate_logits, uint64_t count)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		const float gate = LmBf16ToFloat(LmFloatToBf16(SparkMathSigmoid(LmBf16ToFloat(gate_logits[index]))));
		values[index] = LmFloatToBf16(LmBf16ToFloat(values[index]) * gate);
	}
}

__global__ static void HyScaleRowsKernel(float *values, uint64_t count, float scale)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
		values[index] *= scale;
}

template<class Geometry, uint32_t HEADS, uint32_t DIM>
__global__ static void HyIndexScoreKernel(LmKvShardView view, const uint16_t *query, const float *weights, const uint32_t *sequence_of_row,
	const uint32_t *context_length, const uint32_t *positions, float *scores, uint32_t stride, uint32_t dense_limit)
{
	__shared__ float shared_query[HEADS * DIM];
	__shared__ float shared_weight[HEADS];
	const uint32_t row = blockIdx.y, sequence = sequence_of_row[row];
	uint32_t limit = context_length[sequence];
	if ( positions[row] + 1u < limit )
		limit = positions[row] + 1u;
	const uint32_t local_count = SparkKvShardLocalKeys(view.shard, limit);
	for ( uint32_t index = threadIdx.x; index < HEADS * DIM; index += blockDim.x )
		shared_query[index] = LmBf16ToFloat(query[(uint64_t)row * HEADS * DIM + index]);
	for ( uint32_t index = threadIdx.x; index < HEADS; index += blockDim.x )
		shared_weight[index] = weights[(uint64_t)row * HEADS + index];
	__syncthreads();
	for ( uint32_t local = blockIdx.x * blockDim.x + threadIdx.x; local < stride; local += gridDim.x * blockDim.x )
	{
		float score = -INFINITY;
		if ( limit > dense_limit && local < local_count )
		{
			const uint16_t *key = (const uint16_t *)LmKvShardSlotRequired<Geometry>(view, sequence,
				SparkKvShardLocalPosition(view.shard, local), row, LM_KV_ACCESS_READ);
			if ( key != 0 )
			{
				float key_values[DIM];
				for ( uint32_t d = 0u; d < DIM; ++d )
					key_values[d] = LmBf16ToFloat(key[d]);
				score = 0.0f;
				for ( uint32_t head = 0u; head < HEADS; ++head )
				{
					float dot = 0.0f;
					for ( uint32_t d = 0u; d < DIM; ++d )
						dot = fmaf(shared_query[head * DIM + d], key_values[d], dot);
					score = fmaf(shared_weight[head], fmaxf(dot, 0.0f), score);
				}
			}
		}
		scores[(uint64_t)row * stride + local] = score;
	}
}

template<class G>
struct HyModel
{
	using S = HyShape<G>;
	using State = HyState<G>;

	static uint32_t Blocks(uint64_t elements)
	{
		const uint64_t blocks = (elements + HY_THREADS - 1u) / HY_THREADS;
		return (uint32_t)(blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u);
	}

	template<class Format>
	static int32_t Project(const void *weight, LmScaleTensor scale, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		if ( rows <= LM_SKINNY_ROWS_WIDE )
		{
			LmSkinnyArguments args;
			int32_t status;
			memset(&args, 0, sizeof(args));
			args.weight = (const uint8_t *)weight;
			args.activation = source;
			args.output_bf16 = destination;
			args.output_f32 = destination_f32;
			args.scale = scale;
			args.rows = rows;
			args.input_dimension = input_dimension;
			args.output_dimension = output_dimension;
			status = LmSkinnyLaunch<Format>(&args, stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		return LmStreamGemmDense<Format>(weight, scale, source, destination, destination_f32, rows, input_dimension, output_dimension, 0u, 0u,
			multiprocessors, stream);
	}

	static LmScaleTensor Scales(const uint8_t *scale, uint32_t groups, uint32_t rows, uint32_t input_dimension)
	{
		return LmScaleTensorBuild(scale, LM_SCALE_ENCODING_UE8M0, groups, rows, input_dimension, 1u, HY_SCALE_GROUP);
	}

	static int32_t Linear(const HyScaled &weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		return Project<LmMxfp8>(weight.weight, Scales(weight.scale, 1u, output_dimension, input_dimension), source, destination, (float *)0, rows,
			input_dimension, output_dimension, multiprocessors, stream);
	}

	static int32_t Dense16(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, cudaStream_t stream)
	{
		return Project<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension, output_dimension,
			multiprocessors, stream);
	}

	static SparkStatus Round(State *state, cudaStream_t stream, uint32_t operation, uint32_t rows, uint32_t row_elements, const void *local, void *full)
	{
		return state->services->round(state->services->context, (void *)stream, operation, SPARK_STAGE_RUNNER_ROUND_ALL,
			rows, row_elements, local, full);
	}

	static SparkStatus Launched(void)
	{
		return cudaPeekAtLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	}

	static SparkStatus Select(State *state, const HyLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors;
		const LmKvShardView *view = &state->index_views[layer->index_slot];
		SparkStatus status;
		if ( Linear(layer->index_q, state->q_resid, state->index_q, rows, G::kQLora, S::kIndexWidth, sms, stream) != LM_LAUNCH_OK ||
			Linear(layer->index_k, state->normed, state->index_k, rows, G::kHidden, G::kIndexDim, sms, stream) != LM_LAUNCH_OK ||
			Dense16(layer->index_weights, state->normed, (uint16_t *)0, state->index_weights, rows, G::kHidden, G::kIndexHeads, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		HyLayerNormKernel<HY_THREADS><<<rows, HY_THREADS, 0, stream>>>(state->index_k, layer->index_k_norm, layer->index_k_bias, G::kIndexDim, G::kIndexNormEpsilon);
		LM_LAUNCH((LmRopePerHeadKernel<HY_THREADS,LM_ROPE_INTERLEAVED>), dim3(rows,G::kIndexHeads), HY_THREADS, 0, stream,
			state->index_q, step->positions, G::kIndexHeads, G::kIndexDim, G::kRope, G::kRopeTheta, (const float *)0, 1.0f, G::kIndexDim - G::kRope);
		LM_LAUNCH((LmRopePerHeadKernel<HY_THREADS,LM_ROPE_INTERLEAVED>), dim3(rows,1u), HY_THREADS, 0, stream,
			state->index_k, step->positions, 1u, G::kIndexDim, G::kRope, G::kRopeTheta, (const float *)0, 1.0f, G::kIndexDim - G::kRope);
		HyScaleRowsKernel<<<Blocks((uint64_t)rows * G::kIndexHeads), HY_THREADS, 0, stream>>>(state->index_weights, (uint64_t)rows * G::kIndexHeads,
			(float)(1.0 / sqrt((double)G::kIndexHeads) / sqrt((double)G::kIndexDim)));
		LM_LAUNCH((LmKvShardStoreKernel<typename S::IndexKv,HY_THREADS>), rows, HY_THREADS, 0, stream, *view, state->index_k, step->sequence_of_row,
			step->positions, rows, G::kIndexDim);
		HyIndexScoreKernel<typename S::IndexKv,G::kIndexHeads,G::kIndexDim><<<dim3((state->local_capacity + HY_THREADS - 1u) / HY_THREADS, rows), HY_THREADS, 0, stream>>>(*view,
			state->index_q, state->index_weights, step->sequence_of_row, step->context_length, step->positions, state->index_scores,
			state->local_capacity, G::kIndexTopk);
		if ( Launched() != SPARK_STATUS_OK ||
			LmTopkExactLaunch<HY_THREADS>(state->index_scores, rows, state->local_capacity, G::kIndexTopk, LM_TOPK_EXACT_CHUNK,
				LM_TOPK_EXACT_CHUNKED_ROWS, state->index_topk_values, state->index_topk_positions, state->topk_entries,
				state->index_local_selected, stream) != cudaSuccess ||
			LmIndexShardCandidatePackLaunch<HY_THREADS>(view->shard, state->index_scores, state->local_capacity, state->index_local_selected,
				G::kIndexTopk, rows, state->index_candidates, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, S::kCandidateSplit * rows,
			G::kIndexTopk / S::kCandidateSplit * (uint32_t)(sizeof(uint2) / sizeof(uint16_t)), state->index_candidates, state->index_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		for ( uint32_t first = 0u; first < rows; first += HY_SELECT_CHUNK )
		{
			const uint32_t count = rows - first < HY_SELECT_CHUNK ? rows - first : HY_SELECT_CHUNK;
			if ( LmIndexShardCandidateScatterLaunch<HY_THREADS>(state->index_gathered + (uint64_t)first * G::kIndexTopk, (uint64_t)rows * G::kIndexTopk,
					state->tp_degree, G::kIndexTopk, count, state->context_capacity, state->index_full_scores, stream) != cudaSuccess ||
				LmTopkExactLaunch<HY_THREADS>(state->index_full_scores, count, state->context_capacity, G::kIndexTopk, LM_TOPK_EXACT_CHUNK,
					LM_TOPK_EXACT_CHUNKED_ROWS, state->index_topk_values, state->index_topk_positions, state->topk_entries,
					state->selected + (uint64_t)first * G::kIndexTopk, stream) != cudaSuccess )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		return Launched();
	}

	static SparkStatus Attention(State *state, const HyLayer *layer, uint32_t index, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, degree = state->tp_degree;
		const uint64_t query_stride = (uint64_t)rows * S::kRankHeads * S::kSlot;
		const uint64_t record_unit = G::kHidden * sizeof(uint16_t) / sizeof(float);
		const uint64_t partial_stride = ((uint64_t)rows * S::kRankHeads * S::kRecord + record_unit - 1u) / record_unit * record_unit;
		const LmKvShardView *view = &state->kv_views[index];
		SparkStatus status;
		if ( state->kv_attached == 0u || partial_stride > state->partial_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Linear(layer->q_a, state->normed, state->q_resid, rows, G::kHidden, G::kQLora, sms, stream) != LM_LAUNCH_OK ||
			Linear(layer->kv_a, state->normed, state->kv, rows, G::kHidden, S::kSlot, sms, stream) != LM_LAUNCH_OK ||
			Dense16(layer->gate, state->normed, state->gate, (float *)0, rows, G::kHidden, S::kRankHeads * G::kValueDim, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		HyRmsNormKernel<HY_THREADS,G::kQLora><<<rows, HY_THREADS, 0, stream>>>(state->q_resid, layer->q_norm, state->q_resid, G::kQLora, G::kQLora,
			G::kEpsilon);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(layer->q_b, state->q_resid, state->q, rows, G::kQLora, S::kRankHeads * S::kQkDim, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmPerHeadProjectSplitKernel<HY_THREADS,G::kNope,G::kLatent,S::kQkDim,0u>), dim3(rows,S::kRankHeads,G::kLatent / LM_PER_HEAD_SPLIT_OUTPUTS),
			HY_THREADS, 0, stream, state->q, layer->uk, state->q_latent, S::kRankHeads, rows);
		LM_LAUNCH((LmRopePerHeadKernel<HY_THREADS,LM_ROPE_INTERLEAVED>), dim3(rows,S::kRankHeads), HY_THREADS, 0, stream,
			state->q, step->positions, S::kRankHeads, S::kQkDim, G::kRope, G::kRopeTheta, (const float *)0, 1.0f, G::kNope);
		HyQueryPackKernel<<<dim3(rows, S::kRankHeads), HY_THREADS, 0, stream>>>(state->q_latent, state->q, state->query, S::kRankHeads, G::kLatent,
			G::kNope, G::kRope);
		HyRmsNormKernel<HY_THREADS,G::kLatent><<<rows, HY_THREADS, 0, stream>>>(state->kv, layer->kv_norm, state->kv, S::kSlot, S::kSlot, G::kEpsilon);
		LM_LAUNCH((LmRopePerHeadKernel<HY_THREADS,LM_ROPE_INTERLEAVED>), dim3(rows,1u), HY_THREADS, 0, stream,
			state->kv, step->positions, 1u, S::kSlot, G::kRope, G::kRopeTheta, (const float *)0, 1.0f, G::kLatent);
		LM_LAUNCH((LmKvShardStoreKernel<typename S::Kv,HY_THREADS>), rows, HY_THREADS, 0, stream, *view, state->kv, step->sequence_of_row,
			step->positions, rows, S::kSlot);
		if ( Launched() != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( G::IndexerLayer(index) )
		{
			status = Select(state, layer, step, stream);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, S::kRankHeads * S::kSlot, state->query, state->gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmLatentShardAttendLaunch<typename S::Kv,LmKvShardView,G::kLatent,G::kRope>(*view, state->gathered, query_stride, S::kRankHeads,
				step->sequence_of_row, step->context_length, step->positions, state->selected, G::kIndexTopk, G::kIndexTopk,
				(float)(1.0 / sqrt((double)S::kQkDim)), state->send, partial_stride, rows, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)(partial_stride / record_unit), 0u, state->send,
			state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmLatentShardMergeLaunch<G::kLatent>(state->received, partial_stride, degree, S::kRankHeads, state->latent, rows, stream,
				layer->sinks + (uint64_t)state->tp_rank * S::kRankHeads) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmPerHeadProjectSplitKernel<HY_THREADS,G::kLatent,G::kValueDim>), dim3(rows,S::kRankHeads,G::kValueDim / LM_PER_HEAD_SPLIT_OUTPUTS),
			HY_THREADS, 0, stream, state->latent, layer->uv, state->value, S::kRankHeads, rows);
		HySigmoidGateKernel<<<Blocks((uint64_t)rows * S::kRankHeads * G::kValueDim), HY_THREADS, 0, stream>>>(state->value, state->gate,
			(uint64_t)rows * S::kRankHeads * G::kValueDim);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(layer->out, state->value, state->partial, rows, S::kRankHeads * G::kValueDim, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Dense(State *state, const HyLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors;
		if ( Linear(layer->ffn_gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * state->ffn_rows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmSiluMulLimitKernel<HY_THREADS>), rows, HY_THREADS, 0, stream, state->gate_up, state->intermediate, state->ffn_rows, true, INFINITY);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(layer->ffn_down, state->intermediate, state->partial, rows, state->ffn_rows, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Moe(State *state, const HyLayer *layer, const SparkStageRunnerStep *step, cudaStream_t stream)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, routes = rows * G::kTopK;
		LmLocalExpertRoutes expert_routes;
		if ( Dense16(layer->router, state->normed, (uint16_t *)0, state->router_logits, rows, G::kHidden, G::kExperts, sms, stream) != LM_LAUNCH_OK ||
			LmTopkGroupLimitedLaunch<HY_THREADS,G::kExperts,1u,1u,G::kTopK>(rows, state->router_logits, layer->router_bias, state->route_expert,
				state->route_weight, G::kRouteScale, stream) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_expert, state->route_weight, state->route_global,
			routes, state->expert_first, G::kLocalExperts);
		if ( LmRouteBuild<HY_THREADS,G::kLocalExperts + 1u>(state->route_expert, rows, routes, G::kTopK, state->group_row_offset,
			state->route_packed_row, state->route_source_token, G::kMoeInter, G::kHidden, LM_MOE_GEMM_TILE_N, state->tile_prefix_up,
			state->tile_prefix_down, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmPackedRouteWeightKernel<<<(routes + 255u) / 256u, 256u, 0, stream>>>(state->route_packed_row, state->route_weight, state->packed_weight, routes);
		expert_routes.group_row_offset = state->group_row_offset;
		expert_routes.route_source_token = state->route_source_token;
		expert_routes.tile_prefix_up = state->tile_prefix_up;
		expert_routes.tile_prefix_down = state->tile_prefix_down;
		expert_routes.rows = rows;
		expert_routes.routes = routes;
		expert_routes.top_k = G::kTopK;
		expert_routes.local_experts = G::kLocalExperts;
		expert_routes.multiprocessors = sms;
		if ( Launched() != SPARK_STATUS_OK ||
			LmLocalExpertsUp<LmMxfp8>(expert_routes, layer->moe_w1.weight, Scales(layer->moe_w1.scale, G::kLocalExperts, G::kMoeInter, G::kHidden),
				state->normed, state->expert_gate, G::kHidden, G::kMoeInter, stream) != LM_LAUNCH_OK ||
			LmLocalExpertsUp<LmMxfp8>(expert_routes, layer->moe_w3.weight, Scales(layer->moe_w3.scale, G::kLocalExperts, G::kMoeInter, G::kHidden),
				state->normed, state->expert_up, G::kHidden, G::kMoeInter, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmSwigluPackedKernel<<<dim3(Blocks(G::kMoeInter), routes), HY_THREADS, 0, stream>>>(state->expert_gate, state->expert_up, state->packed_weight,
			state->expert_act, G::kMoeInter, G::kSwigluLimit);
		if ( Launched() != SPARK_STATUS_OK ||
			LmLocalExpertsDown<LmMxfp8>(expert_routes, layer->moe_w2.weight, Scales(layer->moe_w2.scale, G::kLocalExperts, G::kHidden, G::kMoeInter),
				state->expert_act, state->expert_out, G::kMoeInter, G::kHidden, stream) != LM_LAUNCH_OK ||
			Linear(layer->shared_gate_up, state->normed, state->shared_gate_up, rows, G::kHidden, 2u * state->shared_rows, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmSiluMulLimitKernel<HY_THREADS>), rows, HY_THREADS, 0, stream, state->shared_gate_up, state->shared_act, state->shared_rows, true,
			INFINITY);
		if ( Launched() != SPARK_STATUS_OK ||
			Linear(layer->shared_down, state->shared_act, state->shared_out, rows, state->shared_rows, G::kHidden, sms, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LmMoeLocalFinalizeKernel<<<dim3(Blocks(G::kHidden), rows), HY_THREADS, 0, stream>>>(state->expert_out, state->route_packed_row,
			state->route_expert, state->shared_out, state->partial, G::kTopK, G::kLocalExperts, G::kHidden);
		return Launched();
	}

	template<uint32_t MIXES>
	static void HcPre(State *state, const float *fn, const float *base, const float *scale, float *post, uint16_t *mixed, uint32_t rows, float magnitude,
		cudaStream_t stream)
	{
		if ( rows <= HY_HC_SPLIT_ROWS )
		{
			HyHcDotKernel<G::kStreams,G::kHidden><<<dim3(MIXES + 1u, HY_HC_LANES / HY_HC_SLICE, rows), HY_HC_LOADERS, 0, stream>>>(state->streams, fn,
				state->hc_partial, MIXES + 1u);
			HyHcMixKernel<G::kStreams,MIXES,G::kHidden><<<dim3((G::kHidden + HY_HC_LANES - 1u) / HY_HC_LANES, rows), HY_HC_LANES, 0, stream>>>(state->streams,
				state->hc_partial, base, scale, post, mixed, G::kEpsilon, G::kHcEpsilon, magnitude);
			return;
		}
		HyHcPreKernel<HY_THREADS,G::kStreams,MIXES><<<(rows + HY_HC_ROWS - 1u) / HY_HC_ROWS, HY_THREADS, 0, stream>>>(state->streams, fn, base, scale, post,
			mixed, rows, G::kHidden, G::kEpsilon, G::kHcEpsilon, magnitude);
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *stream_void)
	{
		State *state = (State *)model;
		cudaStream_t stream = (cudaStream_t)stream_void;
		const uint32_t rows = step->rows;
		uint16_t *hidden = step->hidden_bf16;
		SparkStatus status = SPARK_STATUS_OK;
		if ( rows == 0u || rows > state->max_rows || step->sequences > state->slots )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		HyExpandStreamsKernel<<<dim3(Blocks(S::kResidual), rows), HY_THREADS, 0, stream>>>(hidden, state->streams, G::kStreams, G::kHidden);
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const HyLayer *layer = &state->layers[index];
			HcPre<2u * G::kStreams>(state, layer->attn_hc_fn, layer->attn_hc_base, layer->attn_hc_scale, state->hc_post, state->mixed, rows,
				G::kHcMagnitude, stream);
			HyRmsNormKernel<HY_THREADS,G::kHidden><<<rows, HY_THREADS, 0, stream>>>(state->mixed, layer->input_norm, state->normed, G::kHidden, G::kHidden,
				G::kEpsilon);
			status = Launched();
			if ( status == SPARK_STATUS_OK )
				status = Attention(state, layer, index, step, stream);
			if ( status == SPARK_STATUS_OK )
				status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
			if ( status != SPARK_STATUS_OK )
				break;
			HyHcPostKernel<<<dim3(Blocks(S::kResidual), rows), HY_THREADS, 0, stream>>>(state->streams, state->hc_post, state->partial, G::kStreams, G::kHidden);
			HcPre<2u * G::kStreams>(state, layer->mlp_hc_fn, layer->mlp_hc_base, layer->mlp_hc_scale, state->hc_post, state->mixed, rows,
				G::kHcMagnitude, stream);
			HyRmsNormKernel<HY_THREADS,G::kHidden><<<rows, HY_THREADS, 0, stream>>>(state->mixed, layer->post_norm, state->normed, G::kHidden, G::kHidden,
				G::kEpsilon);
			status = Launched();
			if ( status == SPARK_STATUS_OK )
				status = index < G::kDenseLayers ? Dense(state, layer, step, stream) : Moe(state, layer, step, stream);
			if ( status == SPARK_STATUS_OK )
				status = Round(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->partial, state->partial);
			if ( status != SPARK_STATUS_OK )
				break;
			HyHcPostKernel<<<dim3(Blocks(S::kResidual), rows), HY_THREADS, 0, stream>>>(state->streams, state->hc_post, state->partial, G::kStreams, G::kHidden);
			status = Launched();
			if ( status == SPARK_STATUS_OK && state->services->layer_done != 0 )
				state->services->layer_done(state->services->context, index);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		HcPre<G::kStreams>(state, state->head_hc_fn, state->head_hc_base, state->head_hc_scale, (float *)0, hidden, rows, 0.0f, stream);
		return Launched();
	}

	static SparkStatus Plan(State *state, uint64_t *bytes, uint8_t *base)
	{
		const uint64_t rows = state->max_rows, routes = rows * G::kTopK, groups = G::kLocalExperts + 2u;
		const uint64_t width = rows * G::kHidden * sizeof(uint16_t), chunk = rows < HY_SELECT_CHUNK ? rows : HY_SELECT_CHUNK;
		struct { void **target; uint64_t bytes; } plan[] =
		{
			{ (void **)&state->streams, rows * S::kResidual * sizeof(uint16_t) },
			{ (void **)&state->hc_post, rows * G::kStreams * sizeof(float) },
			{ (void **)&state->hc_partial, (rows < HY_HC_SPLIT_ROWS ? rows : HY_HC_SPLIT_ROWS) * (2u * G::kStreams + 1u) * HY_HC_LANES * sizeof(float) },
			{ (void **)&state->mixed, width },
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, width },
			{ (void **)&state->q_resid, rows * G::kQLora * sizeof(uint16_t) },
			{ (void **)&state->q, rows * S::kRankHeads * S::kQkDim * sizeof(uint16_t) },
			{ (void **)&state->q_latent, rows * S::kRankHeads * G::kLatent * sizeof(uint16_t) },
			{ (void **)&state->query, rows * S::kRankHeads * S::kSlot * sizeof(uint16_t) },
			{ (void **)&state->kv, rows * S::kSlot * sizeof(uint16_t) },
			{ (void **)&state->gathered, (uint64_t)state->tp_degree * rows * S::kRankHeads * S::kSlot * sizeof(uint16_t) },
			{ (void **)&state->send, (uint64_t)state->tp_degree * state->partial_stride * sizeof(float) },
			{ (void **)&state->received, (uint64_t)state->tp_degree * state->partial_stride * sizeof(float) },
			{ (void **)&state->latent, rows * S::kRankHeads * G::kLatent * sizeof(uint16_t) },
			{ (void **)&state->value, rows * S::kRankHeads * G::kValueDim * sizeof(uint16_t) },
			{ (void **)&state->gate, rows * S::kRankHeads * G::kValueDim * sizeof(uint16_t) },
			{ (void **)&state->index_q, rows * S::kIndexWidth * sizeof(uint16_t) },
			{ (void **)&state->index_k, rows * G::kIndexDim * sizeof(uint16_t) },
			{ (void **)&state->index_weights, rows * G::kIndexHeads * sizeof(float) },
			{ (void **)&state->index_scores, rows * state->local_capacity * sizeof(float) },
			{ (void **)&state->index_topk_values, state->topk_entries * sizeof(float) },
			{ (void **)&state->index_topk_positions, state->topk_entries * sizeof(uint32_t) },
			{ (void **)&state->index_local_selected, rows * G::kIndexTopk * sizeof(uint32_t) },
			{ (void **)&state->index_candidates, rows * G::kIndexTopk * sizeof(uint2) },
			{ (void **)&state->index_gathered, (uint64_t)state->tp_degree * rows * G::kIndexTopk * sizeof(uint2) },
			{ (void **)&state->index_full_scores, chunk * state->context_capacity * sizeof(float) },
			{ (void **)&state->selected, rows * G::kIndexTopk * sizeof(uint32_t) },
			{ (void **)&state->gate_up, rows * 2u * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->router_logits, rows * G::kExperts * sizeof(float) },
			{ (void **)&state->route_expert, routes * sizeof(uint32_t) },
			{ (void **)&state->route_global, routes * sizeof(uint32_t) },
			{ (void **)&state->route_weight, routes * sizeof(float) },
			{ (void **)&state->group_row_offset, groups * sizeof(uint32_t) },
			{ (void **)&state->route_packed_row, routes * sizeof(uint32_t) },
			{ (void **)&state->route_source_token, routes * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_up, groups * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_down, groups * sizeof(uint32_t) },
			{ (void **)&state->packed_weight, routes * sizeof(float) },
			{ (void **)&state->expert_gate, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_up, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_act, routes * G::kMoeInter * sizeof(uint16_t) },
			{ (void **)&state->expert_out, routes * G::kHidden * sizeof(uint16_t) },
			{ (void **)&state->shared_gate_up, rows * 2u * state->shared_rows * sizeof(uint16_t) },
			{ (void **)&state->shared_act, rows * state->shared_rows * sizeof(uint16_t) },
			{ (void **)&state->shared_out, width },
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

	static void Close(void *model)
	{
		State *state = (State *)model;
		if ( state == 0 )
			return;
		if ( state->arena != 0 )
			(void)cudaFree(state->arena);
		if ( state->pack_open != 0u )
			SparkNamedPackClose(&state->pack);
		free(state);
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry)
	{
		const SparkStageRunnerConfiguration *configuration = request->configuration;
		State *state;
		SparkStatus status;
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, experts = 0u, local = 0u, top_k = 0u, embed_rows = 0u;
		const uint64_t record_unit = G::kHidden * sizeof(uint16_t) / sizeof(float);
		if ( configuration->stage_count != 1u || configuration->tp_degree != G::kTpDegree || configuration->max_input_row_count == 0u ||
			configuration->state_budget_bytes == 0u || configuration->kv_pages_per_sequence == 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state = (State *)calloc(1u, sizeof(*state));
		if ( state == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		*model = state;
		state->services = request->services;
		state->tp_degree = configuration->tp_degree;
		state->tp_rank = configuration->tp_rank;
		state->max_rows = configuration->max_input_row_count;
		state->slots = configuration->resident_sequence_capacity > configuration->max_active_sequence_count ?
			configuration->resident_sequence_capacity : configuration->max_active_sequence_count;
		state->context_capacity = configuration->kv_pages_per_sequence * G::kPageSlots;
		state->local_capacity = ((state->context_capacity + state->tp_degree - 1u) / state->tp_degree + 63u) / 64u * 64u;
		state->context_capacity = (state->context_capacity + 63u) / 64u * 64u;
		state->local_capacity = state->local_capacity > G::kIndexTopk ? state->local_capacity : G::kIndexTopk;
		state->context_capacity = state->context_capacity > G::kIndexTopk ? state->context_capacity : G::kIndexTopk;
		{
			const uint64_t local_entries = LmTopkExactScratchEntries(state->max_rows, state->local_capacity, G::kIndexTopk);
			const uint64_t global_entries = LmTopkExactScratchEntries(HY_SELECT_CHUNK, state->context_capacity, G::kIndexTopk);
			state->topk_entries = local_entries > global_entries ? local_entries : global_entries;
		}
		status = SparkNamedPackOpen(configuration->rank_pack_path, SPARK_NAMED_PACK_MAGIC, SPARK_NAMED_PACK_VERSION, HY_PACK_ALIGNMENT, &state->pack);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->pack_open = 1u;
		if ( SparkNamedPackConfigU32(&state->pack, "tp_degree", &degree) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "tp_rank", &rank) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "hidden", &hidden) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "layers", &layers) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "vocab_rows", &vocab_rows) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "ffn_rows", &state->ffn_rows) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "shared_rows", &state->shared_rows) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "experts", &experts) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "local_experts", &local) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "expert_first", &state->expert_first) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "top_k", &top_k) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			vocab_rows * degree != G::kVocab || state->ffn_rows % 64u != 0u || state->shared_rows % 64u != 0u || experts != G::kExperts ||
			local != G::kLocalExperts || top_k != G::kTopK || state->expert_first != rank * G::kLocalExperts )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u experts %u/%u first %u top_k %u, the runner is tp %u rank %u\n",
				G::Tag(), degree, rank, hidden, layers, experts, local, state->expert_first, top_k, configuration->tp_degree, configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		state->partial_stride = ((uint64_t)state->max_rows * S::kRankHeads * S::kRecord + record_unit - 1u) / record_unit * record_unit;
		status = Plan(state, &state->arena_bytes, 0);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( state->arena_bytes > configuration->state_budget_bytes )
		{
			fprintf(stderr, "%s: rank state %llu bytes is over the %llu-byte budget\n", G::Tag(), (unsigned long long)state->arena_bytes,
				(unsigned long long)configuration->state_budget_bytes);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		memset(geometry, 0, sizeof(*geometry));
		geometry->hidden = G::kHidden;
		geometry->vocab = G::kVocab;
		geometry->total_layers = G::kLayers;
		geometry->layer_count = G::kLayers;
		geometry->rms_epsilon = G::kEpsilon;
		geometry->pack_bytes = state->pack.file_bytes;
		geometry->kv_layer_count = G::kLayers;
		geometry->kv_layer_page_bytes = S::Kv::kPageBytes;
		geometry->kv_second_layer_count = G::kIndexLayers;
		geometry->kv_second_layer_page_bytes = S::IndexKv::kPageBytes;
		geometry->kv_shard.degree = state->tp_degree;
		geometry->kv_shard.rank = state->tp_rank;
		geometry->kv_shard.grain = 1u;
		geometry->head_norm_f32 = 1u;
		if ( SparkNamedPackTensorRange(&state->pack, G::Tag(), "embed", "bf16", &geometry->embed_offset, &geometry->embed_bytes, &embed_rows) != SPARK_STATUS_OK ||
			SparkNamedPackTensorRange(&state->pack, G::Tag(), "head_norm", "f32", &geometry->head_norm_offset, &geometry->head_norm_bytes, 0) != SPARK_STATUS_OK ||
			SparkNamedPackTensorRange(&state->pack, G::Tag(), "head", "bf16", &geometry->head_offset, &geometry->head_bytes, &geometry->head_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( embed_rows != vocab_rows || geometry->head_rows != vocab_rows || geometry->embed_bytes != (uint64_t)vocab_rows * G::kHidden * sizeof(uint16_t) ||
			geometry->head_bytes != (uint64_t)vocab_rows * G::kHidden * sizeof(uint16_t) || geometry->head_norm_bytes != (uint64_t)G::kHidden * sizeof(float) )
		{
			fprintf(stderr, "%s: pack embed %u rows %llu bytes, head %u rows %llu bytes, head norm %llu bytes; the rank holds %u vocab rows of %u\n", G::Tag(),
				embed_rows, (unsigned long long)geometry->embed_bytes, geometry->head_rows, (unsigned long long)geometry->head_bytes,
				(unsigned long long)geometry->head_norm_bytes, vocab_rows, G::kHidden);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		geometry->embed_rows = embed_rows;
		return SPARK_STATUS_OK;
	}

	static SparkStatus ManifestCheck(void *model, const struct SparkWeightdManifest *manifest)
	{
		(void)model;
		if ( manifest == 0 || manifest->range_count != 0u )
		{
			fprintf(stderr, "%s: the pack carries its experts as plain tensors, so the weightd manifest has no expert ranges\n", G::Tag());
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Slice(State *state, SparkWeightdLazyPack *lazy_pack, const char *key, const char *kind, uint64_t expected_bytes, const void **field)
	{
		return SparkNamedPackBindSized(&state->pack, lazy_pack, G::Tag(), key, kind, expected_bytes, field);
	}

	template<class T>
	static SparkStatus Bind(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, const char *kind, T **field, uint64_t elements)
	{
		char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		snprintf(key, sizeof(key), "layers.%u.%s", index, name);
		return Slice(state, lazy_pack, key, kind, elements * sizeof(T), (const void **)field);
	}

	static SparkStatus BindScaled(State *state, SparkWeightdLazyPack *lazy_pack, uint32_t index, const char *name, HyScaled *field, uint64_t rows,
		uint32_t columns)
	{
		char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		SparkStatus status;
		if ( columns % HY_SCALE_GROUP != 0u )
		{
			fprintf(stderr, "%s: layer %u %s has %u columns, not whole %u-column scale groups\n", G::Tag(), index, name, columns, HY_SCALE_GROUP);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		status = Bind(state, lazy_pack, index, name, "fp8", &field->weight, rows * columns);
		snprintf(key, sizeof(key), "%s_scale", name);
		return status == SPARK_STATUS_OK ? Bind(state, lazy_pack, index, key, "u8", &field->scale, rows * (columns / HY_SCALE_GROUP)) : status;
	}

	static SparkStatus BindModel(void *model, SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		uint32_t indexers = 0u;
		(void)geometry;
		if ( Slice(state, lazy_pack, "head_hc_fn", "f32", (uint64_t)G::kStreams * S::kResidual * sizeof(float), (const void **)&state->head_hc_fn) != SPARK_STATUS_OK ||
			Slice(state, lazy_pack, "head_hc_base", "f32", (uint64_t)G::kStreams * sizeof(float), (const void **)&state->head_hc_base) != SPARK_STATUS_OK ||
			Slice(state, lazy_pack, "head_hc_scale", "f32", sizeof(float), (const void **)&state->head_hc_scale) != SPARK_STATUS_OK )
			return SPARK_STATUS_PARSE_ERROR;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			HyLayer *layer = &state->layers[index];
			SparkStatus status = Bind(state, lazy_pack, index, "attn_hc_fn", "f32", &layer->attn_hc_fn, (uint64_t)2u * G::kStreams * S::kResidual);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "attn_hc_base", "f32", &layer->attn_hc_base, 2u * G::kStreams);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "attn_hc_scale", "f32", &layer->attn_hc_scale, 2u);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "mlp_hc_fn", "f32", &layer->mlp_hc_fn, (uint64_t)2u * G::kStreams * S::kResidual);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "mlp_hc_base", "f32", &layer->mlp_hc_base, 2u * G::kStreams);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "mlp_hc_scale", "f32", &layer->mlp_hc_scale, 2u);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "input_norm", "bf16", &layer->input_norm, G::kHidden);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "post_norm", "bf16", &layer->post_norm, G::kHidden);
			if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "q_a", &layer->q_a, G::kQLora, G::kHidden);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "q_norm", "bf16", &layer->q_norm, G::kQLora);
			if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "q_b", &layer->q_b, (uint64_t)S::kRankHeads * S::kQkDim, G::kQLora);
			if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "kv_a", &layer->kv_a, S::kSlot, G::kHidden);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "kv_norm", "bf16", &layer->kv_norm, G::kLatent);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "uk", "bf16", &layer->uk, (uint64_t)S::kRankHeads * G::kLatent * G::kNope);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "uv", "bf16", &layer->uv, (uint64_t)S::kRankHeads * G::kValueDim * G::kLatent);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "gate", "bf16", &layer->gate, (uint64_t)S::kRankHeads * G::kValueDim * G::kHidden);
			if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "out", &layer->out, G::kHidden, S::kRankHeads * G::kValueDim);
			if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "sinks", "f32", &layer->sinks, G::kHeads);
			if ( G::IndexerLayer(index) )
			{
				if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "index_q", &layer->index_q, S::kIndexWidth, G::kQLora);
				if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "index_k", &layer->index_k, G::kIndexDim, G::kHidden);
				if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "index_k_norm", "f32", &layer->index_k_norm, G::kIndexDim);
				if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "index_k_bias", "f32", &layer->index_k_bias, G::kIndexDim);
				if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "index_weights", "bf16", &layer->index_weights, (uint64_t)G::kIndexHeads * G::kHidden);
				layer->index_slot = indexers++;
			}
			else
				layer->index_slot = indexers == 0u ? 0u : indexers - 1u;
			if ( index < G::kDenseLayers )
			{
				if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "ffn_gate_up", &layer->ffn_gate_up, 2u * (uint64_t)state->ffn_rows, G::kHidden);
				if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "ffn_down", &layer->ffn_down, G::kHidden, state->ffn_rows);
			}
			else
			{
				if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "router", "bf16", &layer->router, (uint64_t)G::kExperts * G::kHidden);
				if ( status == SPARK_STATUS_OK ) status = Bind(state, lazy_pack, index, "router_bias", "f32", &layer->router_bias, G::kExperts);
				if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "moe_w1", &layer->moe_w1, (uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
				if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "moe_w3", &layer->moe_w3, (uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
				if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "moe_w2", &layer->moe_w2, (uint64_t)G::kLocalExperts * G::kHidden, G::kMoeInter);
				if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "shared_gate_up", &layer->shared_gate_up, 2u * (uint64_t)state->shared_rows, G::kHidden);
				if ( status == SPARK_STATUS_OK ) status = BindScaled(state, lazy_pack, index, "shared_down", &layer->shared_down, G::kHidden, state->shared_rows);
			}
			if ( status != SPARK_STATUS_OK )
				return SPARK_STATUS_PARSE_ERROR;
		}
		if ( indexers != G::kIndexLayers || !G::IndexerLayer(0u) )
		{
			fprintf(stderr, "%s: %u indexer layers, the model has %u and the first layer must run one\n", G::Tag(), indexers, G::kIndexLayers);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if ( cudaMalloc(&state->arena, state->arena_bytes) != cudaSuccess )
		{
			state->arena = 0;
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		if ( cudaMemset(state->arena, 0, state->arena_bytes) != cudaSuccess || Plan(state, &state->arena_bytes, (uint8_t *)state->arena) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AttachKv(void *model, const SparkStageRunnerKv *kv)
	{
		State *state = (State *)model;
		if ( kv == 0 || kv->layer_count != G::kLayers || kv->layer_page_bytes != S::Kv::kPageBytes || kv->second_pool == 0 ||
			kv->second_layer_count != G::kIndexLayers || kv->second_layer_page_bytes != S::IndexKv::kPageBytes || kv->sequence_count > state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t layer = 0u; layer < G::kLayers; ++layer )
			if ( LmKvShardViewInitialize<typename S::Kv>(&state->kv_views[layer], kv->pool + (uint64_t)layer * kv->layer_stride_bytes, kv->page_table,
				kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, kv->context_shard) != 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t layer = 0u; layer < G::kIndexLayers; ++layer )
			if ( LmKvShardViewInitialize<typename S::IndexKv>(&state->index_views[layer], kv->second_pool + (uint64_t)layer * kv->second_layer_stride_bytes,
				kv->page_table, kv->page_table_stride, kv->sequence_count, kv->pool_page_count, state->kv_error, kv->context_shard) != 0 )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state->kv_attached = 1u;
		return SPARK_STATUS_OK;
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
		fprintf(stderr, "%s rank=%u arena_bytes=%llu slots=%u rows=%u ffn_rows=%u shared_rows=%u expert_first=%u context=%u\n", G::ReportTag(), rank,
			(unsigned long long)state->arena_bytes, state->slots, state->max_rows, state->ffn_rows, state->shared_rows, state->expert_first,
			state->context_capacity);
	}

	static const SparkStageRunnerModelInterface *Interface(void)
	{
		static const SparkStageRunnerModelInterface interface_table =
		{
			SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
			G::Tag(),
			G::Name(),
			"official",
			Open,
			ManifestCheck,
			BindModel,
			Close,
			Step,
			Sideband,
			AttachKv,
			RecurrentCopy,
			ResetSlot,
			ProbeBuffers,
			Report
		};
		return &interface_table;
	}
};
