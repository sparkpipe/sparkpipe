#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/launch.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "inference/runner/stage_model_cuda.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/mxfp8.cuh"
#include "inference/kernels/hc.cuh"
#include "inference/kernels/index_score.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/project_chain.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/scale.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/topk_exact_plan.h"
#include "inference/kernels/topk_group.cuh"
#include "runtime/gemm.cuh"

#include "sparkpipe/spark_hy4_family.h"
#include "sparkpipe/spark_gpu.h"

#define HY_THREADS 256u
#define HY_SCALE_GROUP 32u
#define HY_FP8_BLOCK 128u
#define HY_SELECT_CHUNK 64u

enum HyWeightCodec : uint8_t
{
	HY_CODEC_MXFP8 = 0u,
	HY_CODEC_FP8_BLOCK128 = 1u,
	HY_CODEC_BF16 = 2u
};

template<class G>
struct HyShape : HyLayout<G>
{
	static constexpr uint32_t kRecord = LM_LATENT_SHARD_RECORD_FLOATS(G::kLatent);
	using Kv = LmKvGeometry<HyLayout<G>::kSlot * 2u, G::kPageSlots, true>;
	using IndexKv = LmKvGeometry<G::kIndexDim * 2u, G::kPageSlots, true>;
	static_assert(HyLayout<G>::kRankHeads * HyLayout<G>::kSlot <= G::kHidden, "a rank's packed queries fit one collective row");
};

typedef struct HyScaled
{
	const void *weight;
	const void *scale;
	uint8_t codec;
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
	HyScaled gate;
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
struct HyState : CudaStagePackState
{
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
	LmKvShardView kv_views[G::kLayers];
	LmKvShardView index_views[G::kIndexLayers];
};

template<uint32_t STREAMS, uint32_t WIDTH>
__global__ __launch_bounds__(HY_THREADS, 2)
static void HyHcExpandKernel(const uint16_t *__restrict__ hidden, uint16_t *__restrict__ streams)
{
	static_assert((WIDTH & 7u) == 0u, "hidden width is whole 16-byte vectors");
	constexpr uint32_t VECS = WIDTH >> 3u;
	const uint32_t row = blockIdx.y;
	const uint4 *src4 = (const uint4 *)(hidden + (uint64_t)row * WIDTH);
	uint4 *dst4 = (uint4 *)(streams + (uint64_t)row * STREAMS * WIDTH);
	LmDependentWait();
	for ( uint32_t v = blockIdx.x * HY_THREADS + threadIdx.x; v < VECS; v += gridDim.x * HY_THREADS )
	{
		const uint4 x = src4[v];
		#pragma unroll
		for ( uint32_t s = 0u; s < STREAMS; ++s )
			dst4[(uint64_t)s * VECS + v] = x;
	}
	LmDependentRelease();
}

template<uint32_t STREAMS, uint32_t WIDTH>
__global__ __launch_bounds__(HY_THREADS, 2)
static void HyHcPostKernel(uint16_t *__restrict__ streams, const float *__restrict__ post, const uint16_t *__restrict__ output)
{
	static_assert((WIDTH & 7u) == 0u, "hidden width is whole 16-byte vectors");
	constexpr uint32_t VECS = WIDTH >> 3u;
	const uint32_t row = blockIdx.y;
	const uint4 *out4 = (const uint4 *)(output + (uint64_t)row * WIDTH);
	uint4 *st4 = (uint4 *)(streams + (uint64_t)row * STREAMS * WIDTH);
	float p[STREAMS];
	LmDependentWait();
	#pragma unroll
	for ( uint32_t s = 0u; s < STREAMS; ++s )
		p[s] = __ldg(post + (uint64_t)row * STREAMS + s);
	for ( uint32_t v = blockIdx.x * HY_THREADS + threadIdx.x; v < VECS; v += gridDim.x * HY_THREADS )
	{
		float ov[8];
		LmLoadBf16x8AsFloat(out4[v], ov);
		#pragma unroll
		for ( uint32_t s = 0u; s < STREAMS; ++s )
		{
			float sv[8];
			LmLoadBf16x8AsFloat(st4[(uint64_t)s * VECS + v], sv);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				sv[k] = fmaf(p[s], ov[k], sv[k]);
			st4[(uint64_t)s * VECS + v] = LmPackUint4FloatToBf16(sv);
		}
	}
	LmDependentRelease();
}

template<uint32_t STREAMS, uint32_t MIXES, uint32_t WIDTH, bool HAS_POST>
__global__ __launch_bounds__(HY_THREADS, 2)
static void HyHcPostDotSmallKernel(uint16_t *__restrict__ streams, const float *__restrict__ post_in, const uint16_t *__restrict__ partial_in,
	const float *__restrict__ fn, float *__restrict__ hc_partial)
{
	constexpr uint32_t FLAT = STREAMS * WIDTH, STEPS = FLAT / LM_HC_MIX_LANES;
	static_assert(FLAT % LM_HC_MIX_LANES == 0u && MIXES == 8u, "8 mixes map cleanly onto 8 warps across 256 lanes");
	__shared__ float s_x[8][32];
	__shared__ float s_p[STREAMS];
	const uint32_t slice = blockIdx.x, row = blockIdx.y;
	const uint32_t warp = threadIdx.x >> 5u, lane32 = threadIdx.x & 31u;
	const uint32_t lane256 = slice * 32u + lane32;
	uint16_t *st_row = streams + (uint64_t)row * FLAT;
	const uint16_t *part_row = HAS_POST ? partial_in + (uint64_t)row * WIDTH : (const uint16_t *)0;
	LmDependentWait();
	if constexpr ( HAS_POST )
	{
		if ( threadIdx.x < STREAMS )
			s_p[threadIdx.x] = post_in[(uint64_t)row * STREAMS + threadIdx.x];
		__syncthreads();
	}
	float dot = 0.0f, sq = 0.0f;
	for ( uint32_t s_base = 0u; s_base < STEPS; s_base += 8u )
	{
		const uint32_t load_step = s_base + warp;
		if ( load_step < STEPS )
		{
			const uint32_t k = load_step * LM_HC_MIX_LANES + lane256;
			float xv = LmBf16ToFloat(st_row[k]);
			if constexpr ( HAS_POST )
			{
				const uint32_t st = k / WIDTH, col = k % WIDTH;
				const uint16_t upd = LmFloatToBf16(fmaf(s_p[st], LmBf16ToFloat(part_row[col]), xv));
				st_row[k] = upd;
				xv = LmBf16ToFloat(upd);
			}
			s_x[warp][lane32] = xv;
		}
		__syncthreads();
		const uint32_t rem = STEPS - s_base < 8u ? STEPS - s_base : 8u;
		#pragma unroll
		for ( uint32_t u = 0u; u < 8u; ++u )
		{
			if ( u < rem )
			{
				const uint32_t k = (s_base + u) * LM_HC_MIX_LANES + lane256;
				const float xv = s_x[u][lane32];
				dot = fmaf(xv, __ldg(fn + (uint64_t)warp * FLAT + k), dot);
				if ( warp == 0u )
					sq = fmaf(xv, xv, sq);
			}
		}
		if ( s_base + 8u < STEPS )
			__syncthreads();
	}
	hc_partial[((uint64_t)row * (MIXES + 1u) + warp) * LM_HC_MIX_LANES + lane256] = dot;
	if ( warp == 0u )
		hc_partial[((uint64_t)row * (MIXES + 1u) + MIXES) * LM_HC_MIX_LANES + lane256] = sq;
	LmDependentRelease();
}

template<uint32_t STREAMS, uint32_t WIDTH>
static __device__ __forceinline__ void HyHcMixRmsNormRow(const uint16_t *__restrict__ st_row, const float weights[STREAMS],
	const uint16_t *__restrict__ norm_weight, uint16_t *__restrict__ out_row, float *__restrict__ sq_wave, float *__restrict__ norm_red,
	float norm_epsilon)
{
	static_assert((WIDTH & 7u) == 0u && WIDTH <= HY_THREADS * LM_RMS_NORM_STAGED, "hidden width fits in staged vectorized waves");
	constexpr uint32_t VEC_SLOTS = (WIDTH + HY_THREADS * 8u - 1u) / (HY_THREADS * 8u);
	uint4 staged[VEC_SLOTS];
	float total = 0.0f;
	#pragma unroll
	for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
	{
		const uint32_t wave_base = slot * HY_THREADS * 8u;
		const uint32_t offset = wave_base + threadIdx.x * 8u;
		if ( offset < WIDTH )
		{
			float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			#pragma unroll
			for ( uint32_t s = 0u; s < STREAMS; ++s )
			{
				float sv[8];
				LmLoadBf16x8AsFloat(*(const uint4 *)(st_row + (uint64_t)s * WIDTH + offset), sv);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					acc[k] = fmaf(weights[s], sv[k], acc[k]);
			}
			const uint4 packed = LmPackUint4FloatToBf16(acc);
			staged[slot] = packed;
			float vals[8], sq[8];
			LmLoadBf16x8AsFloat(packed, vals);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				sq[k] = vals[k] * vals[k];
			LmStoreFloat8(sq_wave + threadIdx.x * 8u, sq);
		}
		else
			staged[slot] = make_uint4(0u, 0u, 0u, 0u);
		__syncthreads();
		#pragma unroll
		for ( uint32_t s = 0u; s < 8u; ++s )
			if ( wave_base + s * HY_THREADS + threadIdx.x < WIDTH )
				total += sq_wave[s * HY_THREADS + threadIdx.x];
		if ( slot + 1u < VEC_SLOTS )
			__syncthreads();
	}
	const float inv = SparkMathRmsScale(LmBlockSumOnce<HY_THREADS>(total, norm_red), (float)WIDTH, norm_epsilon);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
	{
		const uint32_t offset = (slot * HY_THREADS + threadIdx.x) * 8u;
		if ( offset < WIDTH )
		{
			float x[8], w[8], out[8];
			LmLoadBf16x8AsFloat(staged[slot], x);
			LmLoadWeight8(norm_weight, offset, w);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out[k] = SparkMathNormGainRounded(x[k] * inv, w[k]);
			*(uint4 *)(out_row + offset) = LmPackUint4FloatToBf16(out);
		}
	}
}

template<uint32_t STREAMS, uint32_t MIXES, uint32_t WIDTH>
__global__ __launch_bounds__(HY_THREADS, 2)
static void HyHcMixNormSmallKernel(const uint16_t *__restrict__ streams, const float *__restrict__ hc_partial, const float *__restrict__ base,
	const float *__restrict__ scale, float *__restrict__ post_out, const uint16_t *__restrict__ norm_weight, uint16_t *__restrict__ normed,
	float norm_epsilon, float hc_epsilon, float post_magnitude)
{
	constexpr uint32_t WARPS = HY_THREADS / 32u, FLAT = STREAMS * WIDTH;
	__shared__ float reduction[MIXES + 1u][WARPS];
	__shared__ float weights[STREAMS];
	__shared__ float sq_wave[HY_THREADS * 8u];
	__shared__ float norm_red[WARPS];
	const uint32_t row = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	LmDependentWait();
	#pragma unroll
	for ( uint32_t m = 0u; m <= MIXES; ++m )
	{
		float value = hc_partial[((uint64_t)row * (MIXES + 1u) + m) * LM_HC_MIX_LANES + threadIdx.x];
		#pragma unroll
		for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
			value += __shfl_xor_sync(0xffffffffu, value, offset);
		if ( lane == 0u )
			reduction[m][warp] = value;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		float total[MIXES + 1u];
		#pragma unroll
		for ( uint32_t m = 0u; m <= MIXES; ++m )
		{
			total[m] = 0.0f;
			#pragma unroll
			for ( uint32_t w = 0u; w < WARPS; ++w )
				total[m] += reduction[m][w];
		}
		LmHcMixGates<STREAMS, MIXES>(total, FLAT, base, scale, norm_epsilon, hc_epsilon, post_magnitude, weights, post_out + (uint64_t)row * STREAMS);
	}
	__syncthreads();
	float w_reg[STREAMS];
	#pragma unroll
	for ( uint32_t s = 0u; s < STREAMS; ++s )
		w_reg[s] = weights[s];
	HyHcMixRmsNormRow<STREAMS, WIDTH>(streams + (uint64_t)row * FLAT, w_reg, norm_weight, normed + (uint64_t)row * WIDTH, sq_wave, norm_red, norm_epsilon);
	LmDependentRelease();
}

template<uint32_t STREAMS, uint32_t MIXES, uint32_t WIDTH, bool HAS_POST>
__global__ __launch_bounds__(HY_THREADS, 1)
static void HyHcPostMixNormWideKernel(uint16_t *__restrict__ streams, const float *__restrict__ post_in, const uint16_t *__restrict__ partial_in,
	const float *__restrict__ fn, const float *__restrict__ base, const float *__restrict__ scale, float *__restrict__ post_out,
	const uint16_t *__restrict__ norm_weight, uint16_t *__restrict__ normed, uint32_t rows, float norm_epsilon, float hc_epsilon, float post_magnitude)
{
	constexpr uint32_t WARPS = HY_THREADS / 32u, FLAT = STREAMS * WIDTH, RPB = 4u;
	__shared__ float reduction[RPB][MIXES + 1u][WARPS];
	__shared__ float weights[RPB][STREAMS];
	__shared__ float post_prev[RPB][STREAMS];
	__shared__ float sq_wave[HY_THREADS * 8u];
	__shared__ float norm_red[WARPS];
	const uint32_t first = blockIdx.x * RPB, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	LmDependentWait();
	if constexpr ( HAS_POST )
	{
		if ( threadIdx.x < RPB * STREAMS )
		{
			const uint32_t r = threadIdx.x / STREAMS, s = threadIdx.x % STREAMS;
			post_prev[r][s] = first + r < rows ? post_in[(uint64_t)(first + r) * STREAMS + s] : 0.0f;
		}
		__syncthreads();
	}
	float dot[RPB][MIXES + 1u];
	#pragma unroll
	for ( uint32_t r = 0u; r < RPB; ++r )
		#pragma unroll
		for ( uint32_t m = 0u; m <= MIXES; ++m )
			dot[r][m] = 0.0f;
	for ( uint32_t k = threadIdx.x; k < FLAT; k += HY_THREADS )
	{
		float coeff[MIXES];
		#pragma unroll
		for ( uint32_t m = 0u; m < MIXES; ++m )
			coeff[m] = __ldg(fn + (uint64_t)m * FLAT + k);
		const uint32_t st = HAS_POST ? (k / WIDTH) : 0u;
		const uint32_t col = HAS_POST ? (k % WIDTH) : 0u;
		#pragma unroll
		for ( uint32_t r = 0u; r < RPB; ++r )
		{
			if ( first + r < rows )
			{
				const uint64_t at = (uint64_t)(first + r) * FLAT + k;
				float x = LmBf16ToFloat(streams[at]);
				if constexpr ( HAS_POST )
				{
					const uint16_t upd = LmFloatToBf16(fmaf(post_prev[r][st], LmBf16ToFloat(partial_in[(uint64_t)(first + r) * WIDTH + col]), x));
					streams[at] = upd;
					x = LmBf16ToFloat(upd);
				}
				#pragma unroll
				for ( uint32_t m = 0u; m < MIXES; ++m )
					dot[r][m] = fmaf(x, coeff[m], dot[r][m]);
				dot[r][MIXES] = fmaf(x, x, dot[r][MIXES]);
			}
		}
	}
	#pragma unroll
	for ( uint32_t r = 0u; r < RPB; ++r )
		#pragma unroll
		for ( uint32_t m = 0u; m <= MIXES; ++m )
		{
			float value = dot[r][m];
			#pragma unroll
			for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				value += __shfl_xor_sync(0xffffffffu, value, offset);
			if ( lane == 0u )
				reduction[r][m][warp] = value;
		}
	__syncthreads();
	if ( threadIdx.x < RPB && first + threadIdx.x < rows )
	{
		const uint32_t r = threadIdx.x;
		float total[MIXES + 1u];
		#pragma unroll
		for ( uint32_t m = 0u; m <= MIXES; ++m )
		{
			total[m] = 0.0f;
			#pragma unroll
			for ( uint32_t w = 0u; w < WARPS; ++w )
				total[m] += reduction[r][m][w];
		}
		LmHcMixGates<STREAMS, MIXES>(total, FLAT, base, scale, norm_epsilon, hc_epsilon, post_magnitude, weights[r],
			post_out + (uint64_t)(first + r) * STREAMS);
	}
	__syncthreads();
	for ( uint32_t r = 0u; r < RPB; ++r )
	{
		if ( first + r >= rows )
			break;
		float w_reg[STREAMS];
		#pragma unroll
		for ( uint32_t s = 0u; s < STREAMS; ++s )
			w_reg[s] = weights[r][s];
		HyHcMixRmsNormRow<STREAMS, WIDTH>(streams + (uint64_t)(first + r) * FLAT, w_reg, norm_weight, normed + (uint64_t)(first + r) * WIDTH,
			sq_wave, norm_red, norm_epsilon);
		if ( r + 1u < RPB && first + r + 1u < rows )
			__syncthreads();
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(32u, 4)
static void HyQueryRopePackDirectKernel(const uint16_t *__restrict__ q_in, const uint32_t *__restrict__ positions, uint16_t *__restrict__ query_out)
{
	using S = HyShape<G>;
	constexpr uint32_t HALF = G::kRope / 2u;
	static_assert(HALF == 32u, "one warp covers all interleaved rotary pairs");
	const uint32_t row = blockIdx.x, head = blockIdx.y, pair = threadIdx.x;
	const uint64_t src_base = ((uint64_t)row * S::kRankHeads + head) * S::kQkDim + G::kNope;
	const uint64_t dst_base = ((uint64_t)row * S::kRankHeads + head) * S::kSlot + G::kLatent;
	LmDependentWait();
	const float angle = (float)positions[row] * __powf(G::kRopeTheta, -2.0f * (float)pair / (float)G::kRope);
	const uint32_t packed_in = *(const uint32_t *)(q_in + src_base + 2u * pair);
	const float x0 = LmBf16ToFloat((uint16_t)(packed_in & 0xffffu));
	const float x1 = LmBf16ToFloat((uint16_t)(packed_in >> 16u));
	float s, c;
	sincosf(angle, &s, &c);
	const float y0 = SparkMathRoundBf16(x0 * c) - SparkMathRoundBf16(x1 * s);
	const float y1 = SparkMathRoundBf16(x1 * c) + SparkMathRoundBf16(x0 * s);
	*(uint32_t *)(query_out + dst_base + 2u * pair) = LmPackBf16Pair(y0, y1);
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(HY_THREADS, 2)
static void HyKvPrepStoreKernel(const uint16_t *__restrict__ kv_in, const uint16_t *__restrict__ kv_norm, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ store_positions, const uint32_t *__restrict__ sequence_of_row, LmKvShardView kv_view)
{
	using S = HyShape<G>;
	constexpr uint32_t LAT = G::kLatent, HALF = G::kRope / 2u;
	constexpr uint32_t SLOTS = (LAT + HY_THREADS - 1u) / HY_THREADS;
	__shared__ float reduction[LmBlockWarps<HY_THREADS>()];
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	const uint32_t store_pos = store_positions[row];
	if ( SparkKvShardOwns(kv_view.shard, store_pos) != 0u )
	{
		uint16_t *slot = (uint16_t *)LmKvShardSlotRequired<typename S::Kv>(kv_view, sequence_of_row[row], store_pos, row, LM_KV_ACCESS_WRITE);
		if ( slot != 0 )
		{
			const uint16_t *src = kv_in + (uint64_t)row * S::kSlot;
			float vals[SLOTS], total = 0.0f;
			#pragma unroll
			for ( uint32_t s = 0u; s < SLOTS; ++s )
			{
				const uint32_t idx = threadIdx.x + s * HY_THREADS;
				if ( idx < LAT )
				{
					const float v = LmBf16ToFloat(src[idx]);
					vals[s] = v;
					total += v * v;
				}
			}
			const float scale = SparkMathRmsScale(LmBlockSumOnce<HY_THREADS>(total, reduction), (float)LAT, G::kEpsilon);
			#pragma unroll
			for ( uint32_t s = 0u; s < SLOTS; ++s )
			{
				const uint32_t idx = threadIdx.x + s * HY_THREADS;
				if ( idx < LAT )
					slot[idx] = LmFloatToBf16(SparkMathNormGainRounded(vals[s] * scale, LmBf16ToFloat(kv_norm[idx])));
			}
			if ( threadIdx.x < HALF )
			{
				const uint32_t pair = threadIdx.x;
				const float angle = (float)positions[row] * __powf(G::kRopeTheta, -2.0f * (float)pair / (float)G::kRope);
				const uint32_t packed_in = *(const uint32_t *)(src + LAT + 2u * pair);
				const float x0 = LmBf16ToFloat((uint16_t)(packed_in & 0xffffu));
				const float x1 = LmBf16ToFloat((uint16_t)(packed_in >> 16u));
				float sv, cv;
				sincosf(angle, &sv, &cv);
				const float y0 = SparkMathRoundBf16(x0 * cv) - SparkMathRoundBf16(x1 * sv);
				const float y1 = SparkMathRoundBf16(x1 * cv) + SparkMathRoundBf16(x0 * sv);
				*(uint32_t *)(slot + LAT + 2u * pair) = LmPackBf16Pair(y0, y1);
			}
		}
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(128u, 2)
static void HyIndexPrepStoreKernel(uint16_t *__restrict__ index_q, const uint16_t *__restrict__ index_k, const float *__restrict__ k_norm,
	const float *__restrict__ k_bias, float *__restrict__ index_weights, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ store_positions, const uint32_t *__restrict__ sequence_of_row, LmKvShardView index_view, float weight_scale)
{
	using S = HyShape<G>;
	constexpr uint32_t HALF = G::kRope / 2u;
	constexpr uint32_t ROPE_OFF = G::kIndexDim - G::kRope;
	constexpr uint32_t D = G::kIndexDim;
	static_assert(D == 128u && HALF == 32u, "indexer head dimension is 128 with 64 rotary dimensions");
	__shared__ float reduction[LmBlockWarps<128u>()];
	const uint32_t task = blockIdx.x, row = blockIdx.y;
	LmDependentWait();
	if ( task < G::kIndexHeads )
	{
		if ( threadIdx.x < HALF )
		{
			const uint32_t pair = threadIdx.x;
			uint16_t *base = index_q + ((uint64_t)row * G::kIndexHeads + task) * D + ROPE_OFF;
			const float angle = (float)positions[row] * __powf(G::kRopeTheta, -2.0f * (float)pair / (float)G::kRope);
			const uint32_t packed_in = *(const uint32_t *)(base + 2u * pair);
			const float x0 = LmBf16ToFloat((uint16_t)(packed_in & 0xffffu));
			const float x1 = LmBf16ToFloat((uint16_t)(packed_in >> 16u));
			float sv, cv;
			sincosf(angle, &sv, &cv);
			const float y0 = SparkMathRoundBf16(x0 * cv) - SparkMathRoundBf16(x1 * sv);
			const float y1 = SparkMathRoundBf16(x1 * cv) + SparkMathRoundBf16(x0 * sv);
			*(uint32_t *)(base + 2u * pair) = LmPackBf16Pair(y0, y1);
		}
	}
	else
	{
		if ( threadIdx.x < G::kIndexHeads )
			index_weights[(uint64_t)row * G::kIndexHeads + threadIdx.x] *= weight_scale;
		const uint32_t store_pos = store_positions[row];
		if ( SparkKvShardOwns(index_view.shard, store_pos) != 0u )
		{
			uint16_t *slot = (uint16_t *)LmKvShardSlotRequired<typename S::IndexKv>(index_view, sequence_of_row[row], store_pos, row, LM_KV_ACCESS_WRITE);
			if ( slot != 0 )
			{
				const uint16_t *k_row = index_k + (uint64_t)row * D;
				float total = 0.0f, squares = 0.0f;
				float vals[8];
				const uint32_t vectors = D >> 3u;
				if ( threadIdx.x < vectors )
				{
					LmLoadBf16x8AsFloat(*(const uint4 *)(k_row + threadIdx.x * 8u), vals);
					#pragma unroll
					for ( uint32_t k = 0u; k < 8u; ++k )
						total += vals[k];
				}
				const float mean = LmBlockSum<128u>(total, reduction) / (float)D;
				if ( threadIdx.x < vectors )
				{
					#pragma unroll
					for ( uint32_t k = 0u; k < 8u; ++k )
					{
						const float centred = vals[k] - mean;
						squares += centred * centred;
					}
				}
				const float inverse = SparkMathRmsScale(LmBlockSumOnce<128u>(squares, reduction), (float)D, G::kIndexNormEpsilon);
				__shared__ __align__(16) uint16_t s_k[D];
				if ( threadIdx.x < vectors )
				{
					const uint32_t offset = threadIdx.x * 8u;
					float w_f[8], b_f[8], out_f[8];
					LmLoadFloat8(k_norm + offset, w_f);
					LmLoadFloat8(k_bias + offset, b_f);
					#pragma unroll
					for ( uint32_t k = 0u; k < 8u; ++k )
						out_f[k] = (vals[k] - mean) * inverse * w_f[k] + b_f[k];
					*(uint4 *)(s_k + offset) = LmPackUint4FloatToBf16(out_f);
				}
				__syncthreads();
				if ( threadIdx.x < HALF )
				{
					const uint32_t pair = threadIdx.x;
					const float angle = (float)positions[row] * __powf(G::kRopeTheta, -2.0f * (float)pair / (float)G::kRope);
					LmRopeRotate<LM_ROPE_INTERLEAVED, LM_ROPE_ROUNDED_TERMS>(s_k, ROPE_OFF, pair, HALF, angle, 1.0f);
				}
				__syncthreads();
				if ( threadIdx.x < vectors )
					((uint4 *)slot)[threadIdx.x] = ((const uint4 *)s_k)[threadIdx.x];
			}
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(HY_THREADS, 2)
static void HyIndexPackDirectKernel(SparkKvShard shard, const float *__restrict__ local_scores, uint32_t score_stride, uint32_t valid_keep,
	uint32_t keep, uint2 *__restrict__ candidates)
{
	const uint32_t row = blockIdx.x;
	const uint2 empty = make_uint2(__float_as_uint(-INFINITY), LM_INDEX_SHARD_NO_POSITION);
	LmDependentWait();
	for ( uint32_t entry = threadIdx.x; entry < keep; entry += HY_THREADS )
	{
		if ( entry < valid_keep && entry < score_stride )
		{
			const float score = local_scores[(uint64_t)row * score_stride + entry];
			candidates[(uint64_t)row * keep + entry] = make_uint2(__float_as_uint(score),
				score > -INFINITY ? SparkKvShardLocalPosition(shard, entry) : LM_INDEX_SHARD_NO_POSITION);
		}
		else
			candidates[(uint64_t)row * keep + entry] = empty;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(HY_THREADS, 2)
static void HyIndexFillDirectKernel(uint32_t context_capacity, uint32_t keep, uint32_t *__restrict__ selected)
{
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	for ( uint32_t entry = threadIdx.x; entry < keep; entry += HY_THREADS )
		selected[(uint64_t)row * keep + entry] = entry < context_capacity ? entry : LM_INDEX_SHARD_NO_POSITION;
	LmDependentRelease();
}

__global__ __launch_bounds__(HY_THREADS, 2)
static void HyIndexScatterKernel(const uint2 *__restrict__ gathered, uint64_t rank_stride_candidates, uint32_t degree, uint32_t keep,
	uint32_t valid_keep, uint32_t score_stride, float *__restrict__ scores)
{
	const uint32_t row = blockIdx.x;
	float *dst = scores + (uint64_t)row * score_stride;
	const uint32_t vecs = score_stride >> 2u;
	const float4 neg_inf = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
	LmDependentWait();
	for ( uint32_t v = threadIdx.x; v < vecs; v += HY_THREADS )
		((float4 *)dst)[v] = neg_inf;
	for ( uint32_t i = (vecs << 2u) + threadIdx.x; i < score_stride; i += HY_THREADS )
		dst[i] = -INFINITY;
	__syncthreads();
	const uint32_t total = degree * valid_keep;
	for ( uint32_t idx = threadIdx.x; idx < total; idx += HY_THREADS )
	{
		const uint32_t source = idx / valid_keep;
		const uint32_t entry = idx - source * valid_keep;
		const uint2 candidate = gathered[(uint64_t)source * rank_stride_candidates + (uint64_t)row * keep + entry];
		if ( candidate.y < score_stride )
			dst[candidate.y] = __uint_as_float(candidate.x);
	}
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void HyOutputGateKernel(uint16_t *__restrict__ output_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t dimension)
{
	const uint64_t base = (uint64_t)blockIdx.x * dimension;
	LmDependentWait();
	LmGateUpRowBody<THREADS>(gate_bf16 + base, output_bf16 + base, output_bf16 + base, dimension, LmSigmoidGateOp<true>{});
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void HySiluMulKernel(const uint16_t *__restrict__ gate_up, uint16_t *__restrict__ output, uint32_t width)
{
	const uint64_t base = (uint64_t)blockIdx.y * 2u * width;
	LmDependentWait();
	LmBinaryBf16GridBody<THREADS>(gate_up + base, gate_up + base + width, output + (uint64_t)blockIdx.y * width, width, LmSiluLimitOp{INFINITY});
	LmDependentRelease();
}

template<uint32_t EXPERTS, uint32_t TOPK>
static __device__ __forceinline__ void HySigmoidTopkWarp(const float *__restrict__ logits, const float *__restrict__ bias, uint32_t row,
	uint32_t expert_first, uint32_t local_experts, float scale, uint32_t *__restrict__ fault, uint32_t *__restrict__ out_exp,
	float *__restrict__ out_wt, uint32_t lane)
{
	constexpr uint32_t PER_LANE = (EXPERTS + 31u) / 32u;
	float sc[PER_LANE];
	unsigned long long key[PER_LANE];
	const uint64_t row_base = (uint64_t)row * EXPERTS;
	#pragma unroll
	for ( uint32_t j = 0u; j < PER_LANE; ++j )
	{
		const uint32_t e = lane + j * 32u;
		if ( e < EXPERTS )
		{
			const float s = SparkMathSigmoid(logits[row_base + e]);
			const float sel = s + (bias != 0 ? bias[e] : 0.0f);
			sc[j] = s;
			key[j] = ((unsigned long long)LmTopkRouteKey(sel + 0.0f) << 32u) | (0xffffffffu - e);
		}
		else
		{
			sc[j] = 0.0f;
			key[j] = 0ull;
		}
	}
	float total = 0.0f;
	#pragma unroll
	for ( uint32_t pick = 0u; pick < TOPK; ++pick )
	{
		unsigned long long best = 0ull;
		#pragma unroll
		for ( uint32_t j = 0u; j < PER_LANE; ++j )
			best = key[j] > best ? key[j] : best;
		#pragma unroll
		for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		{
			const unsigned long long other = __shfl_xor_sync(0xffffffffu, best, offset);
			best = other > best ? other : best;
		}
		const uint32_t e = 0xffffffffu - (uint32_t)best;
		const uint32_t owner_lane = e & 31u;
		float picked_s = 0.0f;
		#pragma unroll
		for ( uint32_t j = 0u; j < PER_LANE; ++j )
			if ( lane + j * 32u == e )
			{
				picked_s = sc[j];
				key[j] = 0ull;
			}
		picked_s = __shfl_sync(0xffffffffu, picked_s, (int)owner_lane);
		total += picked_s;
		if ( lane == 0u )
		{
			out_exp[pick] = e;
			out_wt[pick] = picked_s;
			LmTopkRouteFault(fault, (uint32_t)(best >> 32u));
		}
	}
	if ( lane == 0u )
	{
		const float denom = total + 1e-20f;
		#pragma unroll
		for ( uint32_t pick = 0u; pick < TOPK; ++pick )
		{
			const uint32_t ge = out_exp[pick];
			if ( ge >= expert_first && ge < expert_first + local_experts )
			{
				out_exp[pick] = ge - expert_first;
				out_wt[pick] = out_wt[pick] / denom * scale;
			}
			else
			{
				out_exp[pick] = local_experts;
				out_wt[pick] = 0.0f;
			}
		}
	}
}

template<class G>
__global__ __launch_bounds__(HY_THREADS, 1)
static void HySigmoidRouteSmallKernel(const float *__restrict__ logits, const float *__restrict__ bias, uint32_t rows, uint32_t expert_first,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ packed_expert,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	float *__restrict__ packed_weight, uint32_t tile_m, uint32_t n_up, uint32_t *__restrict__ tile_prefix_up, uint32_t n_down,
	uint32_t *__restrict__ tile_prefix_down)
{
	__shared__ uint32_t s_exp[32];
	__shared__ float s_wt[32];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, routes = rows * G::kTopK;
	LmDependentWait();
	if ( warp < rows )
		HySigmoidTopkWarp<G::kExperts, G::kTopK>(logits, bias, warp, expert_first, G::kLocalExperts, G::kRouteScale, fault,
			s_exp + warp * G::kTopK, s_wt + warp * G::kTopK, lane);
	__syncthreads();
	if ( threadIdx.x < routes )
	{
		route_expert[threadIdx.x] = s_exp[threadIdx.x];
		route_weight[threadIdx.x] = s_wt[threadIdx.x];
	}
	LmRouteBuildSmallBody<HY_THREADS, G::kLocalExperts + 1u>(s_exp, routes, G::kTopK, group_row_offset, route_packed_row, route_source_token,
		tile_m, n_up, tile_prefix_up, n_down, tile_prefix_down);
	__syncthreads();
	if ( threadIdx.x < routes )
	{
		const uint32_t packed = route_packed_row[threadIdx.x];
		packed_weight[packed] = s_wt[threadIdx.x];
		packed_expert[packed] = s_exp[threadIdx.x];
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(HY_THREADS, 2)
static void HySigmoidTopkLocalKernel(const float *__restrict__ logits, const float *__restrict__ bias, uint32_t rows, uint32_t expert_first,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight)
{
	constexpr uint32_t WARPS = HY_THREADS / LM_WARP_LANES;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t row = blockIdx.x * WARPS + warp;
	LmDependentWait();
	if ( row < rows )
		HySigmoidTopkWarp<G::kExperts, G::kTopK>(logits, bias, row, expert_first, G::kLocalExperts, G::kRouteScale, fault,
			route_expert + (uint64_t)row * G::kTopK, route_weight + (uint64_t)row * G::kTopK, lane);
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t EXPERTS>
__global__ __launch_bounds__(THREADS, 1)
static void HyRouteBuildPackedKernel(const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, uint32_t routes,
	uint32_t top_k, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	float *__restrict__ packed_weight, uint32_t *__restrict__ packed_expert, uint32_t tile_m, uint32_t neuron_tiles_up,
	uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	LmDependentWait();
	LmRouteBuildBlock<THREADS, EXPERTS>(route_expert, routes, top_k, group_row_offset, route_packed_row, route_source_token, tile_m,
		neuron_tiles_up, tile_prefix_up, neuron_tiles_down, tile_prefix_down);
	__syncthreads();
	for ( uint32_t route = threadIdx.x; route < routes; route += THREADS )
	{
		const uint32_t packed = route_packed_row[route];
		packed_weight[packed] = route_weight[route];
		packed_expert[packed] = route_expert[route];
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(HY_THREADS, 2)
static void HyMoeSwigluPackedKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, const float *__restrict__ weight,
	const uint32_t *__restrict__ group_row_offset, uint32_t local_experts, uint16_t *__restrict__ output, uint32_t width, float limit)
{
	const uint32_t pr = blockIdx.y;
	LmDependentWait();
	if ( pr < group_row_offset[local_experts] )
		LmPackedGateUpBody(gate, up, output, width, LmSwigluPackedOp{weight[pr], limit});
	LmDependentRelease();
}

template<bool FP8_F32>
static __device__ __forceinline__ float HyLoadScale(const void *scales, uint64_t index)
{
	if constexpr ( FP8_F32 )
		return __ldg((const float *)scales + index);
	else
		return LmUe8m0ToFloat(__ldg((const uint8_t *)scales + index));
}

template<class G, bool FP8_F32>
__global__ __launch_bounds__(HY_THREADS, 2)
static void HyMoeB1W1W3SwigluKernel(const uint8_t *__restrict__ w1, const void *__restrict__ s1, const uint8_t *__restrict__ w3,
	const void *__restrict__ s3, const uint16_t *__restrict__ activation, const float *__restrict__ packed_weight,
	const uint32_t *__restrict__ packed_expert, const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ output)
{
	constexpr uint32_t CHUNKS = G::kHidden / 16u;
	constexpr uint32_t SC_GROUP = FP8_F32 ? HY_FP8_BLOCK : HY_SCALE_GROUP;
	constexpr uint32_t SC_SHIFT = FP8_F32 ? 3u : 1u;
	constexpr uint32_t SC_STRIDE = G::kHidden / SC_GROUP;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = HY_THREADS / LANES;
	constexpr uint32_t N = G::kMoeInter;
	const uint32_t pr = blockIdx.y;
	LmDependentWait();
	if ( pr < __ldg(group_row_offset + G::kLocalExperts) )
	{
		const uint32_t exp = __ldg(packed_expert + pr);
		const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
		const uint32_t col0 = (blockIdx.x * GROUPS + group) * 2u;
		if ( col0 < N )
		{
			const bool has1 = col0 + 1u < N;
			const uint64_t row_base = (uint64_t)exp * N + col0;
			const uint4 *w1_0 = (const uint4 *)w1 + row_base * CHUNKS;
			const uint4 *w3_0 = (const uint4 *)w3 + row_base * CHUNKS;
			const uint4 *w1_1 = has1 ? w1_0 + CHUNKS : w1_0;
			const uint4 *w3_1 = has1 ? w3_0 + CHUNKS : w3_0;
			const uint64_t sc_base0 = row_base * SC_STRIDE;
			const uint64_t sc_base1 = has1 ? sc_base0 + SC_STRIDE : sc_base0;
			float g0 = 0.0f, u0 = 0.0f, g1 = 0.0f, u1 = 0.0f;
			for ( uint32_t c = sub; c < CHUNKS; c += LANES )
			{
				const LmSkinnyWide act = LmSkinnyFormat<LmFp8>::Load(activation + c * 16u);
				const uint32_t sc_idx = c >> SC_SHIFT;
				const uint4 vw1_0 = __ldcs(w1_0 + c);
				const uint4 vw3_0 = __ldcs(w3_0 + c);
				const uint4 vw1_1 = __ldcs(w1_1 + c);
				const uint4 vw3_1 = __ldcs(w3_1 + c);
				g0 = fmaf(HyLoadScale<FP8_F32>(s1, sc_base0 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(vw1_0, act), g0);
				u0 = fmaf(HyLoadScale<FP8_F32>(s3, sc_base0 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(vw3_0, act), u0);
				g1 = fmaf(HyLoadScale<FP8_F32>(s1, sc_base1 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(vw1_1, act), g1);
				u1 = fmaf(HyLoadScale<FP8_F32>(s3, sc_base1 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(vw3_1, act), u1);
			}
			#pragma unroll
			for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
			{
				g0 += __shfl_down_sync(0xffffffffu, g0, offset);
				u0 += __shfl_down_sync(0xffffffffu, u0, offset);
				g1 += __shfl_down_sync(0xffffffffu, g1, offset);
				u1 += __shfl_down_sync(0xffffffffu, u1, offset);
			}
			if ( sub == 0u )
			{
				const float wt = __ldg(packed_weight + pr);
				output[(uint64_t)pr * N + col0] = LmFloatToBf16(SparkMathGateLimitedGlu(SparkMathRoundBf16(g0), SparkMathRoundBf16(u0), G::kSwigluLimit) * wt);
				if ( has1 )
					output[(uint64_t)pr * N + col0 + 1u] = LmFloatToBf16(SparkMathGateLimitedGlu(SparkMathRoundBf16(g1), SparkMathRoundBf16(u1), G::kSwigluLimit) * wt);
			}
		}
	}
	LmDependentRelease();
}

template<class G, bool FP8_F32>
__global__ __launch_bounds__(HY_THREADS, 2)
static void HyMoeB1W2FinalizeKernel(const uint8_t *__restrict__ w2, const void *__restrict__ s2, const uint16_t *__restrict__ expert_act,
	const uint32_t *__restrict__ route_packed_row, const uint32_t *__restrict__ route_expert, const uint16_t *__restrict__ shared,
	uint16_t *__restrict__ output)
{
	constexpr uint32_t K = G::kMoeInter;
	constexpr uint32_t CHUNKS = K / 16u;
	constexpr uint32_t SC_GROUP = FP8_F32 ? HY_FP8_BLOCK : HY_SCALE_GROUP;
	constexpr uint32_t SC_SHIFT = FP8_F32 ? 3u : 1u;
	constexpr uint32_t SC_STRIDE = K / SC_GROUP;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = HY_THREADS / LANES;
	constexpr uint32_t N = G::kHidden;
	__shared__ uint32_t s_exp[G::kTopK], s_pr[G::kTopK], s_count;
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		uint32_t cnt = 0u;
		for ( uint32_t k = 0u; k < G::kTopK; ++k )
		{
			const uint32_t e = route_expert[k];
			if ( e < G::kLocalExperts )
			{
				s_exp[cnt] = e;
				s_pr[cnt] = route_packed_row[k];
				cnt++;
			}
		}
		s_count = cnt;
	}
	__syncthreads();
	const uint32_t col0 = (blockIdx.x * GROUPS + (threadIdx.x / LANES)) * 2u;
	const uint32_t sub = threadIdx.x & (LANES - 1u);
	if ( col0 < N )
	{
		const uint32_t count = s_count;
		const bool has1 = col0 + 1u < N;
		if ( count == 0u )
		{
			if ( sub == 0u )
			{
				output[col0] = shared[col0];
				if ( has1 )
					output[col0 + 1u] = shared[col0 + 1u];
			}
		}
		else
		{
			float total0 = 0.0f, total1 = 0.0f;
			for ( uint32_t k = 0u; k < count; ++k )
			{
				const uint16_t *a_row = expert_act + (uint64_t)s_pr[k] * K;
				const uint64_t row_base = (uint64_t)s_exp[k] * N + col0;
				const uint4 *w0 = (const uint4 *)w2 + row_base * CHUNKS;
				const uint4 *w1 = has1 ? w0 + CHUNKS : w0;
				const uint64_t sc_base0 = row_base * SC_STRIDE;
				const uint64_t sc_base1 = has1 ? sc_base0 + SC_STRIDE : sc_base0;
				float acc0 = 0.0f, acc1 = 0.0f;
				for ( uint32_t c = sub; c < CHUNKS; c += LANES )
				{
					const LmSkinnyWide act = LmSkinnyFormat<LmFp8>::Load(a_row + c * 16u);
					const uint32_t sc_idx = c >> SC_SHIFT;
					const uint4 vw0 = __ldcs(w0 + c);
					const uint4 vw1 = __ldcs(w1 + c);
					acc0 = fmaf(HyLoadScale<FP8_F32>(s2, sc_base0 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(vw0, act), acc0);
					acc1 = fmaf(HyLoadScale<FP8_F32>(s2, sc_base1 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(vw1, act), acc1);
				}
				#pragma unroll
				for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
				{
					acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
					acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
				}
				if ( sub == 0u )
				{
					total0 += SparkMathRoundBf16(acc0);
					total1 += SparkMathRoundBf16(acc1);
				}
			}
			if ( sub == 0u )
			{
				total0 += LmBf16ToFloat(shared[col0]);
				output[col0] = LmFloatToBf16(total0);
				if ( has1 )
				{
					total1 += LmBf16ToFloat(shared[col0 + 1u]);
					output[col0 + 1u] = LmFloatToBf16(total1);
				}
			}
		}
	}
	LmDependentRelease();
}

template<class G>
struct HyModel : CudaStage<HyModel<G>>
{
	using S = HyShape<G>;
	using State = HyState<G>;
	using Layer = HyLayer;
	using Schedule = HySchedule<G, HyModel>;
	using Kv = CudaStageKv<CudaStageKvViews<typename S::Kv, &State::kv_views, 0u, G::kLayers>,
		CudaStageKvViews<typename S::IndexKv, &State::index_views, 1u, G::kIndexLayers>>;
	using LaneShared = CudaStageFields<&State::kv_error>;

	static constexpr bool kLanes = false;
	static constexpr bool kKeepRows = true;
	static constexpr bool kFusedPostNorm = true;
	static constexpr uint32_t kTpDegree = G::kTpDegree;
	static constexpr uint32_t kHidden = G::kHidden;
	static constexpr uint32_t kVocab = G::kVocab;
	static constexpr uint32_t kLayers = G::kLayers;
	static constexpr float kEpsilon = G::kEpsilon;
	static constexpr uint32_t kPackAlignment = HY_PACK_ALIGNMENT;

	struct Frame
	{
		const SparkStageRunnerStep *step;
		SparkGpuStream stream;
	};

	static const char *Tag(void) { return G::Tag(); }
	static const char *ReportTag(void) { return G::ReportTag(); }
	static const char *WeightdModel(void) { return G::Name(); }
	static LmScaleTensor Scales(const HyScaled &weight, uint32_t groups, uint32_t rows, uint32_t input_dimension)
	{
		if ( weight.codec == HY_CODEC_FP8_BLOCK128 )
			return LmScaleTensorBuild(weight.scale, LM_SCALE_ENCODING_F32, groups, rows, input_dimension, 1u, HY_FP8_BLOCK);
		if ( weight.codec == HY_CODEC_MXFP8 )
			return LmScaleTensorBuild(weight.scale, LM_SCALE_ENCODING_UE8M0, groups, rows, input_dimension, 1u, HY_SCALE_GROUP);
		return LmScaleTensorNone();
	}

	static int32_t SkinnyMulti(uint8_t codec, const LmSkinnyDenseTarget *targets, uint32_t count, const uint16_t *source, uint32_t rows,
		uint32_t input_dimension, SparkGpuStream stream, uint32_t exact)
	{
		if ( codec == HY_CODEC_BF16 )
			return LmSkinnyDenseMulti<LmBf16Format>(targets, count, source, rows, input_dimension, stream, exact);
		if ( codec == HY_CODEC_FP8_BLOCK128 )
			return LmSkinnyDenseMulti<LmFp8>(targets, count, source, rows, input_dimension, stream, exact);
		return LmSkinnyDenseMulti<LmMxfp8>(targets, count, source, rows, input_dimension, stream, exact);
	}

	static int32_t Linear(const HyScaled &weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		if ( weight.codec == HY_CODEC_BF16 )
			return LmDenseRows<LmBf16Format>(weight.weight, LmScaleTensorNone(), source, destination, (float *)0, rows, input_dimension,
				output_dimension, exact, step->multiprocessors, stream);
		if ( weight.codec == HY_CODEC_FP8_BLOCK128 )
			return LmDenseRows<LmFp8>(weight.weight, Scales(weight, 1u, output_dimension, input_dimension), source, destination, (float *)0, rows,
				input_dimension, output_dimension, exact, step->multiprocessors, stream);
		return LmDenseRows<LmMxfp8>(weight.weight, Scales(weight, 1u, output_dimension, input_dimension), source, destination, (float *)0, rows,
			input_dimension, output_dimension, exact, step->multiprocessors, stream);
	}

	static int32_t LinearPair(const HyScaled &first, const HyScaled &second, const uint16_t *source, uint16_t *first_destination,
		uint16_t *second_destination, uint32_t rows, uint32_t input_dimension, uint32_t first_dimension, uint32_t second_dimension,
		const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t row_exact = step->commit == 0u ? 1u : 0u;
		if ( first.codec == second.codec && (rows <= LM_SKINNY_ROWS_WIDE || row_exact != 0u) )
		{
			const LmSkinnyDenseTarget targets[2] = {
				{ first.weight, first_destination, (float *)0, first_dimension, first_dimension, 0u, Scales(first, 1u, first_dimension, input_dimension) },
				{ second.weight, second_destination, (float *)0, second_dimension, second_dimension, 0u, Scales(second, 1u, second_dimension, input_dimension) }
			};
			const int32_t status = SkinnyMulti(first.codec, targets, 2u, source, rows, input_dimension, stream, row_exact);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		const int32_t status = Linear(first, source, first_destination, rows, input_dimension, first_dimension, step, stream);
		return status != LM_LAUNCH_OK ? status : Linear(second, source, second_destination, rows, input_dimension, second_dimension, step, stream);
	}

	static int32_t Dense16(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension, output_dimension,
			step->commit == 0u ? 1u : 0u, step->multiprocessors, stream);
	}

	template<class Format>
	static int32_t ExpertProjectionFormat(const LmLocalExpertRoutes &routes, const HyScaled &weight, const uint16_t *activation, uint16_t *output,
		uint32_t down, uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, SparkGpuStream stream)
	{
		const LmScaleTensor scale = Scales(weight, G::kLocalExperts, output_dimension, input_dimension);
		if ( exact == 0u && routes.rows >= 16u )
		{
			const int32_t launch = LmStreamGemmGrouped<Format>(weight.weight, scale, activation, output, routes.group_row_offset,
				down != 0u ? (const uint32_t *)0 : routes.route_source_token, routes.local_experts, routes.routes, input_dimension, output_dimension,
				routes.multiprocessors, stream);
			if ( launch != LM_LAUNCH_ERR_SHAPE )
				return launch;
		}
		return down != 0u ? LmLocalExpertsDown<Format>(routes, weight.weight, scale, activation, output, input_dimension, output_dimension, stream, exact) :
			LmLocalExpertsUp<Format>(routes, weight.weight, scale, activation, output, input_dimension, output_dimension, stream, exact);
	}

	static int32_t ExpertProjection(const LmLocalExpertRoutes &routes, const HyScaled &weight, const uint16_t *activation, uint16_t *output, uint32_t down,
		uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, SparkGpuStream stream)
	{
		if ( weight.codec == HY_CODEC_BF16 )
			return ExpertProjectionFormat<LmBf16Format>(routes, weight, activation, output, down, input_dimension, output_dimension, exact, stream);
		if ( weight.codec == HY_CODEC_FP8_BLOCK128 )
			return ExpertProjectionFormat<LmFp8>(routes, weight, activation, output, down, input_dimension, output_dimension, exact, stream);
		return ExpertProjectionFormat<LmMxfp8>(routes, weight, activation, output, down, input_dimension, output_dimension, exact, stream);
	}

	static SparkStatus Select(State *state, const HyLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream, bool q_k_done)
	{
		const uint32_t rows = step->rows;
		const LmKvShardView *view = &state->index_views[layer->index_slot];
		SparkStatus status;
		if ( !q_k_done )
		{
			if ( Linear(layer->index_q, state->q_resid, state->index_q, rows, G::kQLora, S::kIndexWidth, step, stream) != LM_LAUNCH_OK ||
				Linear(layer->index_k, state->normed, state->index_k, rows, G::kHidden, G::kIndexDim, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( Dense16(layer->index_weights, state->normed, (uint16_t *)0, state->index_weights, rows, G::kHidden, G::kIndexHeads, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((HyIndexPrepStoreKernel<G>), dim3(G::kIndexHeads + 1u, rows), 128u, 0, stream,
			state->index_q, state->index_k, layer->index_k_norm, layer->index_k_bias, state->index_weights, step->positions,
			step->tree != 0u ? step->slots : step->positions, step->sequence_of_row, *view,
			(float)(1.0 / sqrt((double)G::kIndexHeads) / sqrt((double)G::kIndexDim)));
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			LmWeightedSparseScoreLaunch<typename S::IndexKv, G::kIndexHeads, G::kIndexDim, LmKvShardView, false, float>(state->index_q, state->index_weights,
				*view, step->sequence_of_row, step->context_length, step->positions, rows, state->local_capacity, 1.0f, state->index_scores, stream,
				step->tree != 0u ? step->fork_positions : (const uint32_t *)0, step->tree != 0u ? step->slot_offsets : (const uint32_t *)0) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		const uint32_t per_rank = (state->context_capacity + state->tp_degree - 1u) / state->tp_degree;
		const uint32_t valid_keep = state->local_capacity <= G::kIndexTopk && per_rank < G::kIndexTopk ? per_rank : G::kIndexTopk;
		if ( state->local_capacity <= G::kIndexTopk )
		{
			(void)LM_LAUNCH_DEPENDENT((HyIndexPackDirectKernel), rows, HY_THREADS, 0, stream, view->shard, state->index_scores, state->local_capacity,
				valid_keep, G::kIndexTopk, state->index_candidates);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( LmTopkExactLaunch<HY_THREADS>(state->index_scores, rows, state->local_capacity, G::kIndexTopk, LM_TOPK_EXACT_CHUNK,
				LM_TOPK_EXACT_CHUNKED_ROWS, state->index_topk_values, state->index_topk_positions, state->topk_entries,
				state->index_local_selected, stream) != SPARK_GPU_SUCCESS ||
			LmIndexShardCandidatePackLaunch<HY_THREADS>(view->shard, state->index_scores, state->local_capacity, state->index_local_selected,
				G::kIndexTopk, rows, state->index_candidates, stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, S::kCandidateSplit * rows,
			G::kIndexTopk / S::kCandidateSplit * (uint32_t)(sizeof(uint2) / sizeof(uint16_t)), state->index_candidates, state->index_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( state->context_capacity <= G::kIndexTopk )
		{
			(void)LM_LAUNCH_DEPENDENT((HyIndexFillDirectKernel), rows, HY_THREADS, 0, stream, state->context_capacity, G::kIndexTopk, state->selected);
			return CudaStageLaunched();
		}
		if constexpr ( (G::kIndexTopk & (G::kIndexTopk - 1u)) == 0u )
		{
			if ( state->context_capacity >= (uint32_t)state->tp_degree * G::kIndexTopk && (rows >= 16u || state->context_capacity >= 65536u) )
			{
				if ( LmIndexShardCandidateTopkLaunch<HY_THREADS, G::kIndexTopk>(state->index_gathered, (uint64_t)rows * G::kIndexTopk, state->tp_degree,
						G::kIndexTopk, rows, state->context_capacity, G::kIndexTopk, 1u, state->selected, stream) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				return CudaStageLaunched();
			}
		}
		for ( uint32_t first = 0u; first < rows; first += HY_SELECT_CHUNK )
		{
			const uint32_t count = rows - first < HY_SELECT_CHUNK ? rows - first : HY_SELECT_CHUNK;
			(void)LM_LAUNCH_DEPENDENT((HyIndexScatterKernel), count, HY_THREADS, 0, stream, state->index_gathered + (uint64_t)first * G::kIndexTopk,
				(uint64_t)rows * G::kIndexTopk, state->tp_degree, G::kIndexTopk, valid_keep, state->context_capacity, state->index_full_scores);
			if ( CudaStageLaunched() != SPARK_STATUS_OK ||
				LmTopkExactLaunch<HY_THREADS>(state->index_full_scores, count, state->context_capacity, G::kIndexTopk, LM_TOPK_EXACT_CHUNK,
					LM_TOPK_EXACT_CHUNKED_ROWS, state->index_topk_values, state->index_topk_positions, state->topk_entries,
					state->selected + (uint64_t)first * G::kIndexTopk, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		return CudaStageLaunched();
	}

	static SparkStatus Attention(State *state, const Frame &frame, const HyLayer &entry, uint32_t index)
	{
		const HyLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, degree = state->tp_degree, exact = step->commit == 0u ? 1u : 0u;
		const uint64_t query_stride = (uint64_t)rows * S::kRankHeads * S::kSlot;
		const uint64_t record_unit = G::kHidden * sizeof(uint16_t) / sizeof(float);
		const uint64_t partial_stride = ((uint64_t)rows * S::kRankHeads * S::kRecord + record_unit - 1u) / record_unit * record_unit;
		const LmKvShardView *view = &state->kv_views[index];
		const bool has_idx = G::IndexerLayer(index);
		bool idx_k_done = false, idx_q_done = false;
		SparkStatus status;
		if ( state->kv_attached == 0u || partial_stride > state->partial_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( has_idx && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) && layer->q_a.codec == layer->kv_a.codec && layer->q_a.codec == layer->index_k.codec )
		{
			const LmSkinnyDenseTarget targets[3] = {
				{ layer->q_a.weight, state->q_resid, (float *)0, G::kQLora, G::kQLora, 0u, Scales(layer->q_a, 1u, G::kQLora, G::kHidden) },
				{ layer->kv_a.weight, state->kv, (float *)0, S::kSlot, S::kSlot, 0u, Scales(layer->kv_a, 1u, S::kSlot, G::kHidden) },
				{ layer->index_k.weight, state->index_k, (float *)0, G::kIndexDim, G::kIndexDim, 0u, Scales(layer->index_k, 1u, G::kIndexDim, G::kHidden) }
			};
			if ( SkinnyMulti(layer->q_a.codec, targets, 3u, state->normed, rows, G::kHidden, stream, exact) == LM_LAUNCH_OK )
				idx_k_done = true;
		}
		if ( !idx_k_done && LinearPair(layer->q_a, layer->kv_a, state->normed, state->q_resid, state->kv, rows, G::kHidden, G::kQLora, S::kSlot, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( Linear(layer->gate, state->normed, state->gate, rows, G::kHidden, S::kRankHeads * G::kValueDim, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmBf16RmsNormKernel<HY_THREADS>), rows, HY_THREADS, (G::kQLora + LmBlockWarps<HY_THREADS>()) * sizeof(float), stream, state->q_resid,
			layer->q_norm, state->q_resid, G::kQLora, G::kQLora, G::kEpsilon);
		if ( CudaStageLaunched() != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( has_idx && idx_k_done && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) && layer->q_b.codec == layer->index_q.codec )
		{
			const LmSkinnyDenseTarget targets[2] = {
				{ layer->q_b.weight, state->q, (float *)0, S::kRankHeads * S::kQkDim, S::kRankHeads * S::kQkDim, 0u, Scales(layer->q_b, 1u, S::kRankHeads * S::kQkDim, G::kQLora) },
				{ layer->index_q.weight, state->index_q, (float *)0, S::kIndexWidth, S::kIndexWidth, 0u, Scales(layer->index_q, 1u, S::kIndexWidth, G::kQLora) }
			};
			if ( SkinnyMulti(layer->q_b.codec, targets, 2u, state->q_resid, rows, G::kQLora, stream, exact) == LM_LAUNCH_OK )
				idx_q_done = true;
		}
		if ( !idx_q_done && Linear(layer->q_b, state->q_resid, state->q, rows, G::kQLora, S::kRankHeads * S::kQkDim, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( LmPerHeadProjectChainLaunch<HY_THREADS, G::kNope, G::kLatent, S::kQkDim, 0u, S::kSlot>(state->q, layer->uk, state->query, S::kRankHeads,
				rows, stream, exact) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((HyQueryRopePackDirectKernel<G>), dim3(rows, S::kRankHeads), 32u, 0, stream, state->q, step->positions, state->query);
		(void)LM_LAUNCH_DEPENDENT((HyKvPrepStoreKernel<G>), rows, HY_THREADS, 0, stream, state->kv, layer->kv_norm, step->positions,
			step->tree != 0u ? step->slots : step->positions, step->sequence_of_row, *view);
		if ( CudaStageLaunched() != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( has_idx )
		{
			status = Select(state, layer, step, stream, idx_k_done && idx_q_done);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, S::kRankHeads * S::kSlot, state->query, state->gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmLatentShardAttendLaunch<typename S::Kv,LmKvShardView,G::kLatent,G::kRope>(*view, state->gathered, query_stride, S::kRankHeads,
				step->sequence_of_row, step->context_length, step->positions, state->selected, G::kIndexTopk, G::kIndexTopk,
				(float)(1.0 / sqrt((double)S::kQkDim)), state->send, partial_stride, rows, stream,
				step->tree != 0u ? step->fork_positions : (const uint32_t *)0, step->tree != 0u ? step->slot_offsets : (const uint32_t *)0) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)(partial_stride / record_unit), 0u, state->send,
			state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmLatentShardMergeLaunch<G::kLatent>(state->received, partial_stride, degree, S::kRankHeads, state->latent, rows, stream,
				layer->sinks + (uint64_t)state->tp_rank * S::kRankHeads) != SPARK_GPU_SUCCESS ||
			LmPerHeadProjectChainLaunch<HY_THREADS,G::kLatent,G::kValueDim>(state->latent, layer->uv, state->value, S::kRankHeads, rows, stream, exact) !=
				SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((HyOutputGateKernel<HY_THREADS>), rows, HY_THREADS, 0, stream, state->value, state->gate, S::kRankHeads * G::kValueDim);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->out, state->value, state->partial, rows, S::kRankHeads * G::kValueDim, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Dense(State *state, const Frame &frame, const HyLayer &entry)
	{
		const HyLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows;
		if ( Linear(layer->ffn_gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * state->ffn_rows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((HySiluMulKernel<HY_THREADS>), dim3(LmVecBlocks<HY_THREADS>(state->ffn_rows), rows), HY_THREADS, 0, stream,
			state->gate_up, state->intermediate, state->ffn_rows);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->ffn_down, state->intermediate, state->partial, rows, state->ffn_rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Experts(State *state, const Frame &frame, const HyLayer &entry)
	{
		const HyLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, routes = rows * G::kTopK, exact = step->commit == 0u ? 1u : 0u;
		const uint16_t *normed = state->normed;
		if ( Dense16(layer->router, normed, (uint16_t *)0, state->router_logits, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( routes <= 32u )
		{
			const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
			const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			(void)LM_LAUNCH_DEPENDENT((HySigmoidRouteSmallKernel<G>), 1u, HY_THREADS, 0, stream, state->router_logits, layer->router_bias, rows,
				state->expert_first, step->route_fault, state->route_expert, state->route_weight, state->route_global, state->group_row_offset,
				state->route_packed_row, state->route_source_token, state->packed_weight, tile_m, n_up, state->tile_prefix_up, n_down,
				state->tile_prefix_down);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			constexpr uint32_t WARPS = HY_THREADS / LM_WARP_LANES;
			const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
			const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			(void)LM_LAUNCH_DEPENDENT((HySigmoidTopkLocalKernel<G>), (rows + WARPS - 1u) / WARPS, HY_THREADS, 0, stream, state->router_logits,
				layer->router_bias, rows, state->expert_first, step->route_fault, state->route_expert, state->route_weight);
			(void)LM_LAUNCH_DEPENDENT((HyRouteBuildPackedKernel<HY_THREADS, G::kLocalExperts + 1u>), 1u, HY_THREADS, 0, stream,
				state->route_expert, state->route_weight, routes, G::kTopK, state->group_row_offset, state->route_packed_row,
				state->route_source_token, state->packed_weight, state->route_global, tile_m, n_up, state->tile_prefix_up, n_down,
				state->tile_prefix_down);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( Linear(layer->shared_gate_up, normed, state->shared_gate_up, rows, G::kHidden, 2u * state->shared_rows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((HySiluMulKernel<HY_THREADS>), dim3(LmVecBlocks<HY_THREADS>(state->shared_rows), rows), HY_THREADS, 0, stream,
			state->shared_gate_up, state->shared_act, state->shared_rows);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->shared_down, state->shared_act, state->shared_out, rows, state->shared_rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( rows == 1u && layer->moe_w1.codec != HY_CODEC_BF16 )
		{
			constexpr uint32_t UP_CHUNKS = G::kHidden / 16u;
			constexpr uint32_t UP_LANES = UP_CHUNKS >= 32u ? 32u : (UP_CHUNKS >= 16u ? 16u : 8u);
			constexpr uint32_t UP_COLS = (HY_THREADS / UP_LANES) * 2u;
			const uint32_t up_blocks = (G::kMoeInter + UP_COLS - 1u) / UP_COLS;
			constexpr uint32_t DN_CHUNKS = G::kMoeInter / 16u;
			constexpr uint32_t DN_LANES = DN_CHUNKS >= 32u ? 32u : (DN_CHUNKS >= 16u ? 16u : 8u);
			constexpr uint32_t DN_COLS = (HY_THREADS / DN_LANES) * 2u;
			const uint32_t dn_blocks = (G::kHidden + DN_COLS - 1u) / DN_COLS;
			if ( layer->moe_w1.codec == HY_CODEC_FP8_BLOCK128 )
			{
				(void)LM_LAUNCH_DEPENDENT((HyMoeB1W1W3SwigluKernel<G, true>), dim3(up_blocks, routes), HY_THREADS, 0, stream,
					(const uint8_t *)layer->moe_w1.weight, layer->moe_w1.scale, (const uint8_t *)layer->moe_w3.weight, layer->moe_w3.scale,
					normed, state->packed_weight, state->route_global, state->group_row_offset, state->expert_act);
				(void)LM_LAUNCH_DEPENDENT((HyMoeB1W2FinalizeKernel<G, true>), dn_blocks, HY_THREADS, 0, stream,
					(const uint8_t *)layer->moe_w2.weight, layer->moe_w2.scale, state->expert_act, state->route_packed_row, state->route_expert,
					state->shared_out, state->partial);
			}
			else
			{
				(void)LM_LAUNCH_DEPENDENT((HyMoeB1W1W3SwigluKernel<G, false>), dim3(up_blocks, routes), HY_THREADS, 0, stream,
					(const uint8_t *)layer->moe_w1.weight, layer->moe_w1.scale, (const uint8_t *)layer->moe_w3.weight, layer->moe_w3.scale,
					normed, state->packed_weight, state->route_global, state->group_row_offset, state->expert_act);
				(void)LM_LAUNCH_DEPENDENT((HyMoeB1W2FinalizeKernel<G, false>), dn_blocks, HY_THREADS, 0, stream,
					(const uint8_t *)layer->moe_w2.weight, layer->moe_w2.scale, state->expert_act, state->route_packed_row, state->route_expert,
					state->shared_out, state->partial);
			}
			return CudaStageLaunched();
		}
		LmLocalExpertRoutes expert_routes;
		expert_routes.group_row_offset = state->group_row_offset;
		expert_routes.route_source_token = state->route_source_token;
		expert_routes.tile_prefix_up = state->tile_prefix_up;
		expert_routes.tile_prefix_down = state->tile_prefix_down;
		expert_routes.rows = rows;
		expert_routes.routes = routes;
		expert_routes.top_k = G::kTopK;
		expert_routes.local_experts = G::kLocalExperts;
		expert_routes.multiprocessors = step->multiprocessors;
		if ( ExpertProjection(expert_routes, layer->moe_w1, normed, state->expert_gate, 0u, G::kHidden, G::kMoeInter, exact, stream) != LM_LAUNCH_OK ||
			ExpertProjection(expert_routes, layer->moe_w3, normed, state->expert_up, 0u, G::kHidden, G::kMoeInter, exact, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((HyMoeSwigluPackedKernel), dim3(LmVecBlocks<HY_THREADS>(G::kMoeInter), routes), HY_THREADS, 0, stream,
			state->expert_gate, state->expert_up, state->packed_weight, state->group_row_offset, G::kLocalExperts, state->expert_act, G::kMoeInter,
			G::kSwigluLimit);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			ExpertProjection(expert_routes, layer->moe_w2, state->expert_act, state->expert_out, 1u, G::kMoeInter, G::kHidden, exact, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmMoeLocalFinalizeKernel), dim3(LmVecBlocks<HY_THREADS>(G::kHidden), rows), HY_THREADS, 0, stream, state->expert_out,
			state->route_packed_row, state->route_expert, state->shared_out, state->partial, G::kTopK, G::kLocalExperts, G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus HyperBegin(State *state, const Frame &frame)
	{
		(void)LM_LAUNCH_DEPENDENT((HyHcExpandKernel<G::kStreams, G::kHidden>), dim3(LmVecBlocks<HY_THREADS>(G::kHidden), frame.step->rows), HY_THREADS,
			0, frame.stream, frame.step->hidden_bf16, state->streams);
		return CudaStageLaunched();
	}

	template<bool HAS_POST>
	static SparkStatus HyperStep(State *state, const Frame &frame, const HyLayer &layer, HyHyperPart part)
	{
		const bool ffn = part == HY_FFN_PART;
		const uint32_t rows = frame.step->rows;
		const float *fn = ffn ? layer.mlp_hc_fn : layer.attn_hc_fn;
		const float *hc_scale = ffn ? layer.mlp_hc_scale : layer.attn_hc_scale;
		const float *hc_base = ffn ? layer.mlp_hc_base : layer.attn_hc_base;
		const uint16_t *norm_w = ffn ? layer.post_norm : layer.input_norm;
		if ( rows <= LM_HC_MIX_SPLIT_ROWS )
		{
			(void)LM_LAUNCH_DEPENDENT((HyHcPostDotSmallKernel<G::kStreams, 2u * G::kStreams, G::kHidden, HAS_POST>), dim3(8u, rows), HY_THREADS, 0,
				frame.stream, state->streams, state->hc_post, state->partial, fn, state->hc_partial);
			(void)LM_LAUNCH_DEPENDENT((HyHcMixNormSmallKernel<G::kStreams, 2u * G::kStreams, G::kHidden>), rows, HY_THREADS, 0, frame.stream,
				state->streams, state->hc_partial, hc_base, hc_scale, state->hc_post, norm_w, state->normed, G::kEpsilon, G::kHcEpsilon, G::kHcMagnitude);
			return CudaStageLaunched();
		}
		(void)LM_LAUNCH_DEPENDENT((HyHcPostMixNormWideKernel<G::kStreams, 2u * G::kStreams, G::kHidden, HAS_POST>), (rows + 3u) / 4u, HY_THREADS, 0,
			frame.stream, state->streams, state->hc_post, state->partial, fn, hc_base, hc_scale, state->hc_post, norm_w, state->normed, rows,
			G::kEpsilon, G::kHcEpsilon, G::kHcMagnitude);
		return CudaStageLaunched();
	}

	static SparkStatus HyperNorm(State *state, const Frame &frame, const HyLayer &layer, HyHyperPart part)
	{
		return HyperStep<false>(state, frame, layer, part);
	}

	static SparkStatus HyperPostNorm(State *state, const Frame &frame, const HyLayer &layer, HyHyperPart part)
	{
		return HyperStep<true>(state, frame, layer, part);
	}

	static SparkStatus HyperPost(State *state, const Frame &frame)
	{
		(void)LM_LAUNCH_DEPENDENT((HyHcPostKernel<G::kStreams, G::kHidden>), dim3(LmVecBlocks<HY_THREADS>(G::kHidden), frame.step->rows), HY_THREADS, 0,
			frame.stream, state->streams, state->hc_post, state->partial);
		return CudaStageLaunched();
	}

	static SparkStatus HyperEnd(State *state, const Frame &frame)
	{
		LmHcMix<HY_THREADS,G::kStreams,G::kStreams,G::kHidden>(state->streams, state->head_hc_fn, state->head_hc_scale, state->head_hc_base,
			state->hc_partial, (float *)0, frame.step->hidden_bf16, frame.step->rows, G::kEpsilon, G::kHcEpsilon, 0.0f, frame.stream);
		return CudaStageLaunched();
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream, Frame *frame)
	{
		if ( step->tree != 0u && (step->commit != 0u || step->sequences != 1u || step->slots == 0 || step->fork_positions == 0 || step->slot_offsets == 0) )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		return CudaStage<HyModel>::Begin(state, step, stream, frame);
	}

	static SparkStatus KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream_void)
	{
		State *state = (State *)model;
		SparkGpuStream stream = (SparkGpuStream)stream_void;
		if ( state == 0 || step == 0 || kept == 0 || step->rows == 0u || step->rows > state->max_rows || step->sequences == 0u || step->sequences > state->slots )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( step->tree == 0u )
			return SPARK_STATUS_OK;
		if ( state->kv_attached == 0u || step->commit != 0u || step->sequences != 1u || step->slots == 0 || step->sequence_of_row == 0 || step->sequence_row_indices == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			LM_LAUNCH((LmKvShardPathMoveKernel<typename S::Kv,HY_THREADS>), step->rows, HY_THREADS, 0, stream,
				state->kv_views[index], step->sequence_of_row, step->sequence_row_indices, kept, step->slots, S::kSlot);
		for ( uint32_t index = 0u; index < G::kIndexLayers; ++index )
			LM_LAUNCH((LmKvShardPathMoveKernel<typename S::IndexKv,HY_THREADS>), step->rows, HY_THREADS, 0, stream,
				state->index_views[index], step->sequence_of_row, step->sequence_row_indices, kept, step->slots, G::kIndexDim);
		return CudaStageLaunched();
	}

	static SparkStatus AllReduce(State *state, const Frame &frame) { return CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, frame.step->rows, 0u, state->partial, state->partial); }

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, routes = rows * G::kTopK, groups = G::kLocalExperts + 2u;
		const uint64_t width = rows * G::kHidden * sizeof(uint16_t), chunk = rows < HY_SELECT_CHUNK ? rows : HY_SELECT_CHUNK;
		const CudaStageSlice plan[] =
		{
			{ (void **)&state->streams, rows * S::kResidual * sizeof(uint16_t) },
			{ (void **)&state->hc_post, rows * G::kStreams * sizeof(float) },
			{ (void **)&state->hc_partial, LmHcMixPartialBytes(rows, 2u * G::kStreams) },
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
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		return CudaStagePlace(plan, base);
	}

	static SparkStatus Shape(State *state)
	{
		const uint64_t record_unit = G::kHidden * sizeof(uint16_t) / sizeof(float);
		state->local_capacity = ((state->context_capacity + state->tp_degree - 1u) / state->tp_degree + 63u) / 64u * 64u;
		state->context_capacity = (state->context_capacity + 63u) / 64u * 64u;
		state->local_capacity = state->local_capacity > G::kIndexTopk ? state->local_capacity : G::kIndexTopk;
		state->context_capacity = state->context_capacity > G::kIndexTopk ? state->context_capacity : G::kIndexTopk;
		{
			const uint64_t local_entries = LmTopkExactScratchEntries(state->max_rows, state->local_capacity, G::kIndexTopk);
			const uint64_t global_entries = LmTopkExactScratchEntries(HY_SELECT_CHUNK, state->context_capacity, G::kIndexTopk);
			state->topk_entries = local_entries > global_entries ? local_entries : global_entries;
		}
		state->partial_stride = ((uint64_t)state->max_rows * S::kRankHeads * S::kRecord + record_unit - 1u) / record_unit * record_unit;
		return SPARK_STATUS_OK;
	}

	static uint32_t Accepts(const SparkStageRunnerConfiguration *configuration) { return configuration->kv_pages_per_sequence != 0u ? 1u : 0u; }
	static SparkStatus Prepare(State *state, const SparkStageRunnerConfiguration *configuration) { state->context_capacity = configuration->kv_pages_per_sequence * G::kPageSlots; return SPARK_STATUS_OK; }

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, experts = 0u, local = 0u, top_k = 0u;
		const CudaStagePackValue values[] =
		{
			{ "tp_degree", &degree }, { "tp_rank", &rank }, { "hidden", &hidden }, { "layers", &layers }, { "vocab_rows", &vocab_rows },
			{ "ffn_rows", &state->ffn_rows }, { "shared_rows", &state->shared_rows }, { "experts", &experts }, { "local_experts", &local },
			{ "expert_first", &state->expert_first }, { "top_k", &top_k }
		};
		if ( CudaStagePackValues(&state->pack, values) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			vocab_rows * degree != G::kVocab || state->ffn_rows % 64u != 0u || state->shared_rows % 64u != 0u || experts != G::kExperts ||
			local != G::kLocalExperts || top_k != G::kTopK || state->expert_first != rank * G::kLocalExperts )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u experts %u/%u first %u top_k %u, the runner is tp %u rank %u\n",
				G::Tag(), degree, rank, hidden, layers, experts, local, state->expert_first, top_k, configuration->tp_degree, configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		const uint32_t vocab_rows = G::kVocab / state->tp_degree;
		geometry->tree_verify = 1u;
		if ( CudaStage<HyModel>::HeadRanges(state, geometry, false, false) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( geometry->embed_rows != vocab_rows || geometry->head_rows != vocab_rows ||
			geometry->embed_bytes != (uint64_t)vocab_rows * G::kHidden * sizeof(uint16_t) ||
			geometry->head_bytes != (uint64_t)vocab_rows * G::kHidden * sizeof(uint16_t) || geometry->head_norm_bytes != (uint64_t)G::kHidden * sizeof(float) )
		{
			fprintf(stderr, "%s: pack embed %u rows %llu bytes, head %u rows %llu bytes, head norm %llu bytes; the rank holds %u vocab rows of %u\n", G::Tag(),
				geometry->embed_rows, (unsigned long long)geometry->embed_bytes, geometry->head_rows, (unsigned long long)geometry->head_bytes,
				(unsigned long long)geometry->head_norm_bytes, vocab_rows, G::kHidden);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus GlobalTensors(State *state, CudaStageTensors *tensors)
	{
		tensors->Add("head_hc_fn", "f32", (uint64_t)G::kStreams * S::kResidual * sizeof(float), &state->head_hc_fn);
		tensors->Add("head_hc_base", "f32", (uint64_t)G::kStreams * sizeof(float), &state->head_hc_base);
		tensors->Add("head_hc_scale", "f32", (uint64_t)sizeof(float), &state->head_hc_scale);
		return SPARK_STATUS_OK;
	}

	template<class T>
	static void Add(CudaStageTensors *tensors, const char *name, const char *kind, T **field, uint64_t elements) { tensors->Add(name, kind, elements * sizeof(T), field); }

	static SparkStatus AddScaled(State *state, CudaStageTensors *tensors, uint32_t index, const char *name, HyScaled *field, uint64_t rows, uint32_t columns)
	{
		char full_name[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		SparkNamedPackEntry entry;
		snprintf(full_name, sizeof(full_name), "layers.%u.%s", index, name);
		if ( SparkNamedPackEntryLoad(&state->pack, full_name, &entry) == SPARK_STATUS_OK )
		{
			if ( strcmp(entry.kind, "bf16") == 0 )
			{
				field->codec = HY_CODEC_BF16;
				field->scale = 0;
				tensors->Add(name, "bf16", rows * columns * sizeof(uint16_t), &field->weight);
				return SPARK_STATUS_OK;
			}
			if ( strcmp(entry.kind, "fp8_e4m3") == 0 )
			{
				if ( columns % HY_FP8_BLOCK != 0u )
				{
					fprintf(stderr, "%s: layer %u %s has %u columns, not whole %u-column fp8 blocks\n", G::Tag(), index, name, columns, HY_FP8_BLOCK);
					SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
				}
				field->codec = HY_CODEC_FP8_BLOCK128;
				tensors->Add(name, "fp8_e4m3", rows * columns, &field->weight);
				tensors->AddScaled(name, ".scale", "f32", rows * (columns / HY_FP8_BLOCK) * sizeof(float), &field->scale);
				return SPARK_STATUS_OK;
			}
		}
		if ( columns % HY_SCALE_GROUP != 0u )
		{
			fprintf(stderr, "%s: layer %u %s has %u columns, not whole %u-column scale groups\n", G::Tag(), index, name, columns, HY_SCALE_GROUP);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		field->codec = HY_CODEC_MXFP8;
		tensors->Add(name, "fp8", rows * columns, &field->weight);
		tensors->AddScaled(name, "_scale", "u8", rows * (columns / HY_SCALE_GROUP), &field->scale);
		return SPARK_STATUS_OK;
	}

	static SparkStatus LayerTensors(State *state, uint32_t index, CudaStageTensors *t)
	{
		HyLayer *layer = &state->layers[index];
		SparkStatus status = SPARK_STATUS_OK;
		Add(t, "attn_hc_fn", "f32", &layer->attn_hc_fn, (uint64_t)2u * G::kStreams * S::kResidual);
		Add(t, "attn_hc_base", "f32", &layer->attn_hc_base, 2u * G::kStreams);
		Add(t, "attn_hc_scale", "f32", &layer->attn_hc_scale, 2u);
		Add(t, "mlp_hc_fn", "f32", &layer->mlp_hc_fn, (uint64_t)2u * G::kStreams * S::kResidual);
		Add(t, "mlp_hc_base", "f32", &layer->mlp_hc_base, 2u * G::kStreams);
		Add(t, "mlp_hc_scale", "f32", &layer->mlp_hc_scale, 2u);
		Add(t, "input_norm", "bf16", &layer->input_norm, G::kHidden);
		Add(t, "post_norm", "bf16", &layer->post_norm, G::kHidden);
		if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "q_a", &layer->q_a, G::kQLora, G::kHidden);
		Add(t, "q_norm", "bf16", &layer->q_norm, G::kQLora);
		if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "q_b", &layer->q_b, (uint64_t)S::kRankHeads * S::kQkDim, G::kQLora);
		if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "kv_a", &layer->kv_a, S::kSlot, G::kHidden);
		Add(t, "kv_norm", "bf16", &layer->kv_norm, G::kLatent);
		Add(t, "uk", "bf16", &layer->uk, (uint64_t)S::kRankHeads * G::kLatent * G::kNope);
		Add(t, "uv", "bf16", &layer->uv, (uint64_t)S::kRankHeads * G::kValueDim * G::kLatent);
		if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "gate", &layer->gate, (uint64_t)S::kRankHeads * G::kValueDim, G::kHidden);
		if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "out", &layer->out, G::kHidden, S::kRankHeads * G::kValueDim);
		Add(t, "sinks", "f32", &layer->sinks, G::kHeads);
		if ( G::IndexerLayer(index) )
		{
			if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "index_q", &layer->index_q, S::kIndexWidth, G::kQLora);
			if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "index_k", &layer->index_k, G::kIndexDim, G::kHidden);
			Add(t, "index_k_norm", "f32", &layer->index_k_norm, G::kIndexDim);
			Add(t, "index_k_bias", "f32", &layer->index_k_bias, G::kIndexDim);
			Add(t, "index_weights", "bf16", &layer->index_weights, (uint64_t)G::kIndexHeads * G::kHidden);
		}
		if ( index < G::kDenseLayers )
		{
			if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "ffn_gate_up", &layer->ffn_gate_up, 2u * (uint64_t)state->ffn_rows, G::kHidden);
			if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "ffn_down", &layer->ffn_down, G::kHidden, state->ffn_rows);
		}
		else
		{
			Add(t, "router", "bf16", &layer->router, (uint64_t)G::kExperts * G::kHidden);
			Add(t, "router_bias", "f32", &layer->router_bias, G::kExperts);
			if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "moe_w1", &layer->moe_w1, (uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
			if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "moe_w3", &layer->moe_w3, (uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
			if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "moe_w2", &layer->moe_w2, (uint64_t)G::kLocalExperts * G::kHidden, G::kMoeInter);
			if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "shared_gate_up", &layer->shared_gate_up, 2u * (uint64_t)state->shared_rows, G::kHidden);
			if ( status == SPARK_STATUS_OK ) status = AddScaled(state, t, index, "shared_down", &layer->shared_down, G::kHidden, state->shared_rows);
		}
		return status;
	}

	static SparkStatus Arrange(State *state)
	{
		uint32_t indexers = 0u;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			state->layers[index].index_slot = G::IndexerLayer(index) ? indexers++ : indexers == 0u ? 0u : indexers - 1u;
		if ( indexers != G::kIndexLayers || !G::IndexerLayer(0u) )
		{
			fprintf(stderr, "%s: %u indexer layers, the model has %u and the first layer must run one\n", G::Tag(), indexers, G::kIndexLayers);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static uint32_t ReportFields(const State *state, CudaStageReportField *fields)
	{
		const CudaStageReportField report[] =
		{
			{ "arena_bytes", state->arena_bytes }, { "slots", state->slots }, { "rows", state->max_rows }, { "ffn_rows", state->ffn_rows },
			{ "shared_rows", state->shared_rows }, { "expert_first", state->expert_first }, { "context", state->context_capacity }
		};
		return CudaStageReportOut(fields, report);
	}
};
