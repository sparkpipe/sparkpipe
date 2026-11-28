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
#include "inference/kernels/frame_error.cuh"
#include "inference/kernels/hc.cuh"
#include "inference/kernels/index_score.cuh"
#include "inference/kernels/index_select.cuh"
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
#define HY_SELECT_CHUNK 256u

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
	uint32_t bitmap_words;
	uint32_t indexing_units_max;
	uint64_t partial_stride;
	HyLayer layers[G::kLayers];
	const float *head_hc_fn;
	const float *head_hc_base;
	const float *head_hc_scale;
	uint16_t *streams;
	float *hc_post;
	float *hc_partial;
	uint16_t *normed;
	uint16_t *partial;
	uint16_t *q_resid;
	uint16_t *q;
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
	uint32_t *index_bitmaps;
	uint32_t *selected;
	uint32_t *split_positions;
	uint32_t *split_offsets;
	uint32_t *gather_seq;
	uint32_t *gather_offset;
	uint32_t *gather_context;
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
__global__ __launch_bounds__(128u, 4)
static void HyHcPostDotSmallKernel(uint16_t *__restrict__ streams, const float *__restrict__ post_in, const uint16_t *__restrict__ partial_in,
	const float *__restrict__ fn, float *__restrict__ hc_partial)
{
	constexpr uint32_t FLAT = STREAMS * WIDTH, VECS = WIDTH >> 3u, FLAT_VECS = STREAMS * VECS;
	constexpr uint32_t STEPS = (FLAT_VECS + LM_HC_MIX_LANES - 1u) / LM_HC_MIX_LANES;
	static_assert((WIDTH & 7u) == 0u && MIXES == 8u, "8 mixes map cleanly onto 8 half-warps across 16 lane slices");
	__shared__ uint4 s_x[8][16];
	__shared__ float s_p[STREAMS];
	const uint32_t row = blockIdx.x, slice = blockIdx.y;
	const uint32_t mix = threadIdx.x >> 4u, lane16 = threadIdx.x & 15u;
	const uint32_t lane256 = slice * 16u + lane16;
	uint4 *st4 = (uint4 *)(streams + (uint64_t)row * FLAT);
	const uint4 *part4 = HAS_POST ? (const uint4 *)(partial_in + (uint64_t)row * WIDTH) : (const uint4 *)0;
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
		const uint32_t load_step = s_base + mix;
		if ( load_step < STEPS )
		{
			const uint32_t kv = load_step * LM_HC_MIX_LANES + lane256;
			if ( kv < FLAT_VECS )
			{
				uint4 raw4 = st4[kv];
				if constexpr ( HAS_POST )
				{
					const uint32_t st = kv / VECS, v = kv % VECS;
					float xv[8], pv[8];
					LmLoadBf16x8AsFloat(raw4, xv);
					LmLoadBf16x8AsFloat(__ldg(part4 + v), pv);
					const float sp = s_p[st];
					#pragma unroll
					for ( uint32_t e = 0u; e < 8u; ++e )
						xv[e] = fmaf(sp, pv[e], xv[e]);
					raw4 = LmPackUint4FloatToBf16(xv);
					st4[kv] = raw4;
				}
				s_x[mix][lane16] = raw4;
			}
		}
		__syncthreads();
		const uint32_t rem = STEPS - s_base < 8u ? STEPS - s_base : 8u;
		#pragma unroll
		for ( uint32_t u = 0u; u < 8u; ++u )
		{
			if ( u < rem )
			{
				const uint32_t kv = (s_base + u) * LM_HC_MIX_LANES + lane256;
				if ( kv < FLAT_VECS )
				{
					float xv[8];
					LmLoadBf16x8AsFloat(s_x[u][lane16], xv);
					const uint64_t k_elem = (uint64_t)kv * 8u;
					const float4 f0 = __ldg((const float4 *)(fn + (uint64_t)mix * FLAT + k_elem));
					const float4 f1 = __ldg((const float4 *)(fn + (uint64_t)mix * FLAT + k_elem + 4u));
					dot = fmaf(xv[0], f0.x, dot);
					dot = fmaf(xv[1], f0.y, dot);
					dot = fmaf(xv[2], f0.z, dot);
					dot = fmaf(xv[3], f0.w, dot);
					dot = fmaf(xv[4], f1.x, dot);
					dot = fmaf(xv[5], f1.y, dot);
					dot = fmaf(xv[6], f1.z, dot);
					dot = fmaf(xv[7], f1.w, dot);
					if ( mix == 0u )
					{
						#pragma unroll
						for ( uint32_t e = 0u; e < 8u; ++e )
							sq = fmaf(xv[e], xv[e], sq);
					}
				}
			}
		}
		if ( s_base + 8u < STEPS )
			__syncthreads();
	}
	hc_partial[((uint64_t)row * (MIXES + 1u) + mix) * LM_HC_MIX_LANES + lane256] = dot;
	if ( mix == 0u )
		hc_partial[((uint64_t)row * (MIXES + 1u) + MIXES) * LM_HC_MIX_LANES + lane256] = sq;
	LmDependentRelease();
}

template<uint32_t STREAMS, uint32_t MIXES, uint32_t WIDTH, bool HAS_POST, uint32_t TILE_ROWS>
__global__ __launch_bounds__(128u, 4)
static void HyHcPostDotTileKernel(uint16_t *__restrict__ streams, const float *__restrict__ post_in, const uint16_t *__restrict__ partial_in,
	const float *__restrict__ fn, float *__restrict__ hc_partial, uint32_t rows)
{
	constexpr uint32_t FLAT = STREAMS * WIDTH, VECS = WIDTH >> 3u, FLAT_VECS = STREAMS * VECS;
	constexpr uint32_t STEPS = (FLAT_VECS + LM_HC_MIX_LANES - 1u) / LM_HC_MIX_LANES;
	static_assert((WIDTH & 7u) == 0u && MIXES == 8u && TILE_ROWS == 8u,
		"multi-row HC dot kernel reuses fn loads across 8 rows with exact lane order");
	__shared__ uint4 s_x[8][16];
	__shared__ float s_p[TILE_ROWS][STREAMS];
	const uint32_t row_base = blockIdx.x * TILE_ROWS, slice = blockIdx.y;
	const uint32_t mix = threadIdx.x >> 4u, lane16 = threadIdx.x & 15u;
	const uint32_t lane256 = slice * 16u + lane16;
	const uint32_t cnt = row_base < rows ? (rows - row_base < TILE_ROWS ? rows - row_base : TILE_ROWS) : 0u;
	LmDependentWait();
	if ( cnt == 0u )
	{
		LmDependentRelease();
		return;
	}
	if constexpr ( HAS_POST )
	{
		if ( threadIdx.x < cnt * STREAMS )
		{
			const uint32_t r = threadIdx.x / STREAMS, s = threadIdx.x % STREAMS;
			s_p[r][s] = post_in[(uint64_t)(row_base + r) * STREAMS + s];
		}
		__syncthreads();
	}
	float dot[TILE_ROWS], sq[TILE_ROWS];
	#pragma unroll
	for ( uint32_t r = 0u; r < TILE_ROWS; ++r )
		dot[r] = sq[r] = 0.0f;
	for ( uint32_t step = 0u; step < STEPS; ++step )
	{
		const uint32_t kv = step * LM_HC_MIX_LANES + lane256;
		const bool valid_kv = kv < FLAT_VECS;
		float4 f0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		float4 f1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if ( valid_kv )
		{
			const uint64_t k_elem = (uint64_t)kv * 8u;
			f0 = __ldg((const float4 *)(fn + (uint64_t)mix * FLAT + k_elem));
			f1 = __ldg((const float4 *)(fn + (uint64_t)mix * FLAT + k_elem + 4u));
		}
		if ( valid_kv && mix < cnt )
		{
			const uint32_t row = row_base + mix;
			uint4 *st4 = (uint4 *)(streams + (uint64_t)row * FLAT);
			uint4 raw4 = st4[kv];
			if constexpr ( HAS_POST )
			{
				const uint32_t st = kv / VECS, v = kv % VECS;
				const uint4 *part4 = (const uint4 *)(partial_in + (uint64_t)row * WIDTH);
				float xv[8], pv[8];
				LmLoadBf16x8AsFloat(raw4, xv);
				LmLoadBf16x8AsFloat(__ldg(part4 + v), pv);
				const float sp = s_p[mix][st];
				#pragma unroll
				for ( uint32_t e = 0u; e < 8u; ++e )
					xv[e] = fmaf(sp, pv[e], xv[e]);
				raw4 = LmPackUint4FloatToBf16(xv);
				st4[kv] = raw4;
			}
			s_x[mix][lane16] = raw4;
		}
		__syncthreads();
		if ( valid_kv )
		{
			#pragma unroll
			for ( uint32_t r = 0u; r < TILE_ROWS; ++r )
			{
				if ( r < cnt )
				{
					float xv[8];
					LmLoadBf16x8AsFloat(s_x[r][lane16], xv);
					dot[r] = fmaf(xv[0], f0.x, dot[r]);
					dot[r] = fmaf(xv[1], f0.y, dot[r]);
					dot[r] = fmaf(xv[2], f0.z, dot[r]);
					dot[r] = fmaf(xv[3], f0.w, dot[r]);
					dot[r] = fmaf(xv[4], f1.x, dot[r]);
					dot[r] = fmaf(xv[5], f1.y, dot[r]);
					dot[r] = fmaf(xv[6], f1.z, dot[r]);
					dot[r] = fmaf(xv[7], f1.w, dot[r]);
					if ( mix == 0u )
					{
						#pragma unroll
						for ( uint32_t e = 0u; e < 8u; ++e )
							sq[r] = fmaf(xv[e], xv[e], sq[r]);
					}
				}
			}
		}
		if ( step + 1u < STEPS )
			__syncthreads();
	}
	#pragma unroll
	for ( uint32_t r = 0u; r < TILE_ROWS; ++r )
	{
		if ( r < cnt )
		{
			const uint32_t row = row_base + r;
			hc_partial[((uint64_t)row * (MIXES + 1u) + mix) * LM_HC_MIX_LANES + lane256] = dot[r];
			if ( mix == 0u )
				hc_partial[((uint64_t)row * (MIXES + 1u) + MIXES) * LM_HC_MIX_LANES + lane256] = sq[r];
		}
	}
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
static void HyHcFusedSinglePassKernel(uint16_t *__restrict__ streams, const float *__restrict__ post_in, const uint16_t *__restrict__ partial_in,
	const float *__restrict__ fn, const float *__restrict__ base, const float *__restrict__ scale, float *__restrict__ post_out,
	const uint16_t *__restrict__ norm_weight, uint16_t *__restrict__ normed, float norm_epsilon, float hc_epsilon, float post_magnitude)
{
	constexpr uint32_t FLAT = STREAMS * WIDTH;
	constexpr uint32_t VECS = WIDTH >> 3u;
	constexpr uint32_t FLAT_VECS = STREAMS * VECS;
	constexpr uint32_t STEPS = FLAT_VECS / HY_THREADS;
	constexpr uint32_t WARPS = HY_THREADS / 32u;
	constexpr uint32_t VEC_SLOTS = WIDTH / (HY_THREADS * 8u);
	static_assert(HY_THREADS == 256u && STREAMS == 4u && MIXES == 8u && (WIDTH % (HY_THREADS * 8u)) == 0u, "fused HC matches HY4 4-stream width");
	extern __shared__ __align__(128) uint8_t hc_fused_smem[];
	uint4 *s_st = (uint4 *)hc_fused_smem;
	__shared__ float s_red[MIXES + 1u][WARPS];
	__shared__ float s_w[STREAMS];
	__shared__ float s_p[STREAMS];
	__shared__ float s_sq[HY_THREADS * 8u];
	const uint32_t row = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	uint4 *st4 = (uint4 *)(streams + (uint64_t)row * FLAT);
	const uint4 *part4 = HAS_POST ? (const uint4 *)(partial_in + (uint64_t)row * WIDTH) : (const uint4 *)0;
	LmDependentWait();
	if constexpr ( HAS_POST )
	{
		if ( threadIdx.x < STREAMS )
			s_p[threadIdx.x] = post_in[(uint64_t)row * STREAMS + threadIdx.x];
		__syncthreads();
		const float p0 = s_p[0], p1 = s_p[1], p2 = s_p[2], p3 = s_p[3];
		#pragma unroll
		for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
		{
			const uint32_t v = slot * HY_THREADS + threadIdx.x;
			float pv[8];
			LmLoadBf16x8AsFloat(__ldcs(part4 + v), pv);
			#pragma unroll
			for ( uint32_t s = 0u; s < STREAMS; ++s )
			{
				const float sp = s == 0u ? p0 : (s == 1u ? p1 : (s == 2u ? p2 : p3));
				const uint32_t kv = s * VECS + v;
				float xv[8];
				LmLoadBf16x8AsFloat(__ldcs(st4 + kv), xv);
				#pragma unroll
				for ( uint32_t e = 0u; e < 8u; ++e )
					xv[e] = fmaf(sp, pv[e], xv[e]);
				const uint4 packed = LmPackUint4FloatToBf16(xv);
				__stcs(st4 + kv, packed);
				s_st[kv] = packed;
			}
		}
	}
	else
	{
		#pragma unroll
		for ( uint32_t step = 0u; step < STEPS; ++step )
		{
			const uint32_t kv = step * HY_THREADS + threadIdx.x;
			s_st[kv] = __ldcs(st4 + kv);
		}
	}
	__syncthreads();
	float dot[MIXES] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	float sq = 0.0f;
	for ( uint32_t step = 0u; step < STEPS; ++step )
	{
		const uint32_t kv = step * HY_THREADS + threadIdx.x;
		float xv[8];
		LmLoadBf16x8AsFloat(s_st[kv], xv);
		#pragma unroll
		for ( uint32_t e = 0u; e < 8u; ++e )
			sq = fmaf(xv[e], xv[e], sq);
		const uint64_t k_elem = (uint64_t)kv * 8u;
		#pragma unroll
		for ( uint32_t m = 0u; m < MIXES; ++m )
		{
			const float4 f0 = __ldg((const float4 *)(fn + (uint64_t)m * FLAT + k_elem));
			const float4 f1 = __ldg((const float4 *)(fn + (uint64_t)m * FLAT + k_elem + 4u));
			dot[m] = fmaf(xv[0], f0.x, dot[m]);
			dot[m] = fmaf(xv[1], f0.y, dot[m]);
			dot[m] = fmaf(xv[2], f0.z, dot[m]);
			dot[m] = fmaf(xv[3], f0.w, dot[m]);
			dot[m] = fmaf(xv[4], f1.x, dot[m]);
			dot[m] = fmaf(xv[5], f1.y, dot[m]);
			dot[m] = fmaf(xv[6], f1.z, dot[m]);
			dot[m] = fmaf(xv[7], f1.w, dot[m]);
		}
	}
	#pragma unroll
	for ( uint32_t m = 0u; m < MIXES; ++m )
	{
		float val = dot[m];
		#pragma unroll
		for ( uint32_t off = 16u; off > 0u; off >>= 1u )
			val += __shfl_xor_sync(0xffffffffu, val, off);
		if ( lane == 0u )
			s_red[m][warp] = val;
	}
	{
		float val = sq;
		#pragma unroll
		for ( uint32_t off = 16u; off > 0u; off >>= 1u )
			val += __shfl_xor_sync(0xffffffffu, val, off);
		if ( lane == 0u )
			s_red[MIXES][warp] = val;
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
				total[m] += s_red[m][w];
		}
		LmHcMixGates<STREAMS, MIXES>(total, FLAT, base, scale, norm_epsilon, hc_epsilon, post_magnitude, s_w, post_out + (uint64_t)row * STREAMS);
	}
	__syncthreads();
	const float w0 = s_w[0], w1 = s_w[1], w2 = s_w[2], w3 = s_w[3];
	uint4 staged[VEC_SLOTS];
	float total_sq = 0.0f;
	#pragma unroll
	for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
	{
		const uint32_t v = slot * HY_THREADS + threadIdx.x;
		float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		float sv[8];
		LmLoadBf16x8AsFloat(s_st[0u * VECS + v], sv);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k ) acc[k] = fmaf(w0, sv[k], acc[k]);
		LmLoadBf16x8AsFloat(s_st[1u * VECS + v], sv);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k ) acc[k] = fmaf(w1, sv[k], acc[k]);
		LmLoadBf16x8AsFloat(s_st[2u * VECS + v], sv);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k ) acc[k] = fmaf(w2, sv[k], acc[k]);
		LmLoadBf16x8AsFloat(s_st[3u * VECS + v], sv);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k ) acc[k] = fmaf(w3, sv[k], acc[k]);
		staged[slot] = LmPackUint4FloatToBf16(acc);
		float vals[8], sqv[8];
		LmLoadBf16x8AsFloat(staged[slot], vals);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			sqv[k] = vals[k] * vals[k];
		LmStoreFloat8(s_sq + threadIdx.x * 8u, sqv);
		__syncthreads();
		#pragma unroll
		for ( uint32_t s = 0u; s < 8u; ++s )
			total_sq += s_sq[s * HY_THREADS + threadIdx.x];
		if ( slot + 1u < VEC_SLOTS )
			__syncthreads();
	}
	const float inv = SparkMathRmsScale(LmBlockSumOnce<HY_THREADS>(total_sq, s_red[0]), (float)WIDTH, norm_epsilon);
	uint4 *out4 = (uint4 *)(normed + (uint64_t)row * WIDTH);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
	{
		const uint32_t v = slot * HY_THREADS + threadIdx.x;
		float x[8], w[8], out[8];
		LmLoadBf16x8AsFloat(staged[slot], x);
		LmLoadWeight8(norm_weight, v * 8u, w);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			out[k] = SparkMathNormGainRounded(x[k] * inv, w[k]);
		__stcs(out4 + v, LmPackUint4FloatToBf16(out));
	}
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

template<class G>
__global__ __launch_bounds__(128u, 4)
static void HyIndexWeightsKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation, float *__restrict__ output,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ positions)
{
	const uint32_t row = blockIdx.y;
	LmDependentWait();
	const uint32_t pos = positions[row];
	uint32_t bound = context_length[sequence_of_row[row]];
	if ( pos + 1u < bound )
		bound = pos + 1u;
	if ( bound <= G::kIndexTopk )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t head = blockIdx.x * 4u + warp;
	if ( head < G::kIndexHeads )
	{
		constexpr uint32_t CHUNKS = G::kHidden / 8u;
		const uint4 *w_row = (const uint4 *)(weight + (uint64_t)head * G::kHidden);
		const uint4 *a_row = (const uint4 *)(activation + (uint64_t)row * G::kHidden);
		float acc = 0.0f;
		for ( uint32_t c = lane; c < CHUNKS; c += 32u )
		{
			float af[8];
			LmSkinnyUnpackBf16x8(__ldg(a_row + c), af);
			acc = LmSkinnyBf16ChunkUnpacked(__ldcs(w_row + c), af, acc);
		}
		#pragma unroll
		for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
			acc += __shfl_down_sync(0xffffffffu, acc, offset);
		if ( lane == 0u )
			output[(uint64_t)row * G::kIndexHeads + head] = acc;
	}
	LmDependentRelease();
}

template<class G, uint8_t CODEC>
__global__ __launch_bounds__(128u, 4)
static void HyIndexQSmallKernel(const void *__restrict__ weight, const void *__restrict__ scales, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ positions, uint32_t rows)
{
	using S = HyShape<G>;
	constexpr uint32_t K = G::kQLora;
	constexpr uint32_t N = S::kIndexWidth;
	constexpr uint32_t ELEMS = CODEC == HY_CODEC_BF16 ? 8u : 16u;
	constexpr uint32_t CHUNKS = K / ELEMS;
	constexpr uint32_t SC_GROUP = CODEC == HY_CODEC_FP8_BLOCK128 ? HY_FP8_BLOCK : HY_SCALE_GROUP;
	constexpr uint32_t SC_SHIFT = CODEC == HY_CODEC_FP8_BLOCK128 ? 3u : 1u;
	constexpr uint32_t SC_STRIDE = K / SC_GROUP;
	constexpr uint32_t RG = 8u;
	__shared__ uint32_t s_rows[RG], s_cnt;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t col0 = (blockIdx.x * 4u + warp) * 2u;
	LmDependentWait();
	if ( col0 < N )
	{
		const bool has1 = col0 + 1u < N;
		const uint4 *w0 = (const uint4 *)weight + (uint64_t)col0 * CHUNKS;
		const uint4 *w1 = has1 ? w0 + CHUNKS : w0;
		const uint64_t sc_base0 = (uint64_t)col0 * SC_STRIDE;
		const uint64_t sc_base1 = has1 ? sc_base0 + SC_STRIDE : sc_base0;
		for ( uint32_t r_base = 0u; r_base < rows; r_base += RG )
		{
			if ( threadIdx.x == 0u )
			{
				uint32_t cnt = 0u;
				const uint32_t lim = rows - r_base < RG ? rows - r_base : RG;
				for ( uint32_t r = 0u; r < lim; ++r )
				{
					const uint32_t row = r_base + r;
					const uint32_t pos = positions[row];
					uint32_t bound = context_length[sequence_of_row[row]];
					if ( pos + 1u < bound )
						bound = pos + 1u;
					if ( bound > G::kIndexTopk )
						s_rows[cnt++] = row;
				}
				s_cnt = cnt;
			}
			__syncthreads();
			const uint32_t cnt = s_cnt;
			if ( cnt != 0u )
			{
				const uint16_t *act_ptr[RG];
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					act_ptr[r] = activation + (uint64_t)(r < cnt ? s_rows[r] : 0u) * K;
				float a0[RG], a1[RG];
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					a0[r] = a1[r] = 0.0f;
				for ( uint32_t c = lane; c < CHUNKS; c += 32u )
				{
					const uint4 vw0 = __ldcs(w0 + c);
					const uint4 vw1 = __ldcs(w1 + c);
					if constexpr ( CODEC == HY_CODEC_BF16 )
					{
						const uint32_t c_off = c * 8u;
						#pragma unroll
						for ( uint32_t r = 0u; r < RG; ++r )
						{
							if ( r < cnt )
							{
								float af[8];
								LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(act_ptr[r] + c_off)), af);
								a0[r] = LmSkinnyBf16ChunkUnpacked(vw0, af, a0[r]);
								a1[r] = LmSkinnyBf16ChunkUnpacked(vw1, af, a1[r]);
							}
						}
					}
					else
					{
						constexpr bool FP8_F32 = CODEC == HY_CODEC_FP8_BLOCK128;
						const uint32_t c_off = c * 16u;
						const uint32_t sc_idx = c >> SC_SHIFT;
						const float sc0 = HyLoadScale<FP8_F32>(scales, sc_base0 + sc_idx);
						const float sc1 = HyLoadScale<FP8_F32>(scales, sc_base1 + sc_idx);
						const LmSkinnyFormat<LmFp8>::Decoded d0 = LmSkinnyFormat<LmFp8>::Decode(vw0);
						const LmSkinnyFormat<LmFp8>::Decoded d1 = LmSkinnyFormat<LmFp8>::Decode(vw1);
						#pragma unroll
						for ( uint32_t r = 0u; r < RG; ++r )
						{
							if ( r < cnt )
							{
								const LmSkinnyWide act = LmSkinnyFormat<LmFp8>::Load(act_ptr[r] + c_off);
								a0[r] = fmaf(sc0, LmSkinnyFormat<LmFp8>::ChunkDecoded(d0, act), a0[r]);
								a1[r] = fmaf(sc1, LmSkinnyFormat<LmFp8>::ChunkDecoded(d1, act), a1[r]);
							}
						}
					}
				}
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
				{
					if ( r < cnt )
					{
						#pragma unroll
						for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
						{
							a0[r] += __shfl_down_sync(0xffffffffu, a0[r], offset);
							a1[r] += __shfl_down_sync(0xffffffffu, a1[r], offset);
						}
						if ( lane == 0u )
						{
							const uint32_t row = s_rows[r];
							if ( has1 )
								*(uint32_t *)(output + (uint64_t)row * N + col0) = LmPackBf16Pair(a0[r], a1[r]);
							else
								output[(uint64_t)row * N + col0] = LmFloatToBf16(a0[r]);
						}
					}
				}
			}
			if ( r_base + RG < rows )
				__syncthreads();
		}
	}
	LmDependentRelease();
}

template<class G, class Format>
__global__ __launch_bounds__(LM_STREAM_THREADS, 2)
static void HyIndexQMmaSkipKernel(const uint8_t *__restrict__ weight, LmScaleTensor scale, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ positions, uint32_t rows)
{
	using S = HyShape<G>;
	using Weight = LmStreamWeight<Format>;
	constexpr uint32_t TILE_M = 16u, TILE_N = 64u;
	constexpr uint32_t TILE_K = Weight::kScaled ? 128u : 64u;
	constexpr uint32_t STAGES = 2u;
	constexpr uint32_t K = G::kQLora, N = S::kIndexWidth;
	constexpr uint32_t K_TILES = K / TILE_K;
	constexpr uint32_t A_ROW_BYTES = TILE_K * 2u, A_ROW_CHUNKS = A_ROW_BYTES / 16u;
	constexpr uint32_t B_ROW_BYTES = TILE_K * Weight::kElementBytes, B_ROW_CHUNKS = B_ROW_BYTES / 16u;
	constexpr uint32_t A_BYTES = TILE_M * A_ROW_BYTES, B_BYTES = TILE_N * B_ROW_BYTES;
	constexpr uint32_t STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * A_ROW_CHUNKS;
	constexpr uint32_t B_CHUNKS = TILE_N * B_ROW_CHUNKS;
	constexpr uint32_t B_PER_THREAD = B_CHUNKS / LM_STREAM_THREADS;
	constexpr uint32_t SCALE_K = LmStreamScaleK<Format>();
	constexpr uint32_t SCALE_BLOCKS = Weight::kScaled ? TILE_K / SCALE_K : 1u;
	constexpr bool PACK4_UE8M0 = LmStreamPack4Ue8m0<Format>();
	static_assert(K % TILE_K == 0u && N % TILE_N == 0u && A_CHUNKS <= LM_STREAM_THREADS && B_CHUNKS % LM_STREAM_THREADS == 0u,
		"index_q geometry divides 16x64 skip MMA tile cleanly");
	__shared__ __align__(128) uint8_t smem[STAGES * STAGE_BYTES];
	__shared__ uint32_t s_any;
	const uint32_t row_base = blockIdx.y * TILE_M;
	const uint32_t row_limit = row_base + TILE_M < rows ? row_base + TILE_M : rows;
	LmDependentWait();
	if ( threadIdx.x < 32u )
	{
		const uint32_t r = row_base + threadIdx.x;
		uint32_t sparse = 0u;
		if ( r < row_limit )
		{
			const uint32_t pos = positions[r];
			uint32_t bound = context_length[sequence_of_row[r]];
			if ( pos + 1u < bound )
				bound = pos + 1u;
			sparse = bound > G::kIndexTopk ? 1u : 0u;
		}
		const uint32_t mask = __ballot_sync(0xffffffffu, sparse != 0u);
		if ( threadIdx.x == 0u )
			s_any = mask;
	}
	__syncthreads();
	if ( s_any == 0u )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t neuron_base = blockIdx.x * TILE_N;
	const uint64_t weight_row_bytes = (uint64_t)K * Weight::kElementBytes;
	const uint32_t b_row0 = threadIdx.x / B_ROW_CHUNKS, b_col0 = threadIdx.x % B_ROW_CHUNKS;
	const uint32_t b_row_step = LM_STREAM_THREADS / B_ROW_CHUNKS;
	const uint64_t b_ptr_step = (uint64_t)b_row_step * weight_row_bytes;
	const uint32_t a_row = threadIdx.x / A_ROW_CHUNKS, a_col = threadIdx.x % A_ROW_CHUNKS;
	const bool a_valid = threadIdx.x < A_CHUNKS && row_base + a_row < row_limit;
	const uint16_t *a_src = activation + (uint64_t)(a_valid ? row_base + a_row : 0u) * K + a_col * 8u;
	const uint8_t *b_base = weight + (uint64_t)(neuron_base + b_row0) * weight_row_bytes + b_col0 * 16u;
	const uint32_t a_swiz = a_row * A_ROW_BYTES + LmStreamSwizzle8(a_row, a_col) * 16u;
	const uint32_t ldm_a_row = lane & 15u, ldm_a_col = (lane >> 4u) & 1u;
	const uint32_t column = neuron_base + warp * 8u + quad * 2u;
	auto issue_k = [&](uint32_t st_off)
	{
		const uint32_t stage = shared_base + st_off;
		if ( threadIdx.x < A_CHUNKS )
			LmStreamCopy(stage + a_swiz, a_src, a_valid ? 16u : 0u);
		a_src += TILE_K;
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_THREAD; ++q )
		{
			const uint32_t row = b_row0 + q * b_row_step;
			LmStreamCopy(stage + A_BYTES + Weight::template Offset<B_ROW_BYTES>(row, b_col0), b_base + q * b_ptr_step, 16u);
		}
		b_base += TILE_K * Weight::kElementBytes;
	};
	float total[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
	issue_k(sp_off);
	LmStreamCommit();
	kp++;
	sp_off ^= STAGE_BYTES;
	for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
	{
		LmStreamWait<0u>();
		__syncthreads();
		if ( kp < K_TILES )
		{
			issue_k(sp_off);
			kp++;
			sp_off ^= STAGE_BYTES;
		}
		LmStreamCommit();
		const uint32_t stage_a = shared_base + sc_off;
		const uint32_t stage_b = stage_a + A_BYTES;
		if constexpr ( PACK4_UE8M0 )
		{
			uint32_t sp0 = LmStreamScalePack4(scale, 0u, column, kt * SCALE_BLOCKS);
			uint32_t sp1 = LmStreamScalePack4(scale, 0u, column + 1u, kt * SCALE_BLOCKS);
			#pragma unroll
			for ( uint32_t step = 0u; step < 4u; ++step )
			{
				const float sc0 = LmUe8m0ToFloat((uint8_t)(sp0 & 0xffu));
				const float sc1 = LmUe8m0ToFloat((uint8_t)(sp1 & 0xffu));
				sp0 >>= 8u;
				sp1 >>= 8u;
				uint32_t b[4], a0[4], a1[4];
				Weight::template Fragment<B_ROW_BYTES>(stage_b, warp * 8u, step, lane, b);
				LmStreamLdMatrix4(a0, stage_a + ldm_a_row * A_ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + ldm_a_col) * 16u);
				LmStreamLdMatrix4(a1, stage_a + ldm_a_row * A_ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + 2u + ldm_a_col) * 16u);
				float p[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
				uint32_t bf0[2] = { b[0], b[1] }, bf1[2] = { b[2], b[3] };
				LmMmaBf16(p, a0, bf0);
				LmMmaBf16(p, a1, bf1);
				total[0] = fmaf(p[0], sc0, total[0]);
				total[1] = fmaf(p[1], sc1, total[1]);
				total[2] = fmaf(p[2], sc0, total[2]);
				total[3] = fmaf(p[3], sc1, total[3]);
			}
		}
		else if constexpr ( Weight::kScaled )
		{
			float p[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			#pragma unroll
			for ( uint32_t step = 0u; step < 4u; ++step )
			{
				uint32_t b[4], a0[4], a1[4];
				Weight::template Fragment<B_ROW_BYTES>(stage_b, warp * 8u, step, lane, b);
				LmStreamLdMatrix4(a0, stage_a + ldm_a_row * A_ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + ldm_a_col) * 16u);
				LmStreamLdMatrix4(a1, stage_a + ldm_a_row * A_ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + 2u + ldm_a_col) * 16u);
				uint32_t bf0[2] = { b[0], b[1] }, bf1[2] = { b[2], b[3] };
				LmMmaBf16(p, a0, bf0);
				LmMmaBf16(p, a1, bf1);
			}
			const float sc0 = LmStreamScaleLoad<Format>(scale, 0u, column, kt);
			const float sc1 = LmStreamScaleLoad<Format>(scale, 0u, column + 1u, kt);
			total[0] = fmaf(p[0], sc0, total[0]);
			total[1] = fmaf(p[1], sc1, total[1]);
			total[2] = fmaf(p[2], sc0, total[2]);
			total[3] = fmaf(p[3], sc1, total[3]);
		}
		else
		{
			#pragma unroll
			for ( uint32_t step = 0u; step < 2u; ++step )
			{
				uint32_t b[4], a0[4], a1[4];
				Weight::template Fragment<B_ROW_BYTES>(stage_b, warp * 8u, step, lane, b);
				LmStreamLdMatrix4(a0, stage_a + ldm_a_row * A_ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + ldm_a_col) * 16u);
				LmStreamLdMatrix4(a1, stage_a + ldm_a_row * A_ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + 2u + ldm_a_col) * 16u);
				uint32_t bf0[2] = { b[0], b[1] }, bf1[2] = { b[2], b[3] };
				LmMmaBf16(total, a0, bf0);
				LmMmaBf16(total, a1, bf1);
			}
		}
		sc_off ^= STAGE_BYTES;
	}
	LmStreamWait<0u>();
	const uint32_t r0 = row_base + quad_row, r1 = r0 + 8u;
	if ( r0 < row_limit )
		*(uint32_t *)(output + (uint64_t)r0 * N + column) = LmPackBf16Pair(total[0], total[1]);
	if ( r1 < row_limit )
		*(uint32_t *)(output + (uint64_t)r1 * N + column) = LmPackBf16Pair(total[2], total[3]);
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t WIDTH>
__global__ __launch_bounds__(THREADS, 4)
static void HyQNormKernel(uint16_t *__restrict__ q_resid, const uint16_t *__restrict__ weight_bf16, float epsilon)
{
	static_assert((WIDTH & 7u) == 0u && WIDTH <= THREADS * 8u, "q_norm fits in one 8-element vector per thread");
	constexpr uint32_t VECS = WIDTH >> 3u;
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	__shared__ float sq_wave[WIDTH];
	uint16_t *row_ptr = q_resid + (uint64_t)blockIdx.x * WIDTH;
	uint4 raw = make_uint4(0u, 0u, 0u, 0u);
	float total = 0.0f;
	LmDependentWait();
	if ( threadIdx.x < VECS )
	{
		float vals[8], sq[8];
		raw = *(const uint4 *)(row_ptr + threadIdx.x * 8u);
		LmLoadBf16x8AsFloat(raw, vals);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			sq[k] = vals[k] * vals[k];
		LmStoreFloat8(sq_wave + threadIdx.x * 8u, sq);
	}
	__syncthreads();
	for ( uint32_t idx = threadIdx.x; idx < WIDTH; idx += THREADS )
		total += sq_wave[idx];
	const float inv = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)WIDTH, epsilon);
	if ( threadIdx.x < VECS )
	{
		const uint32_t offset = threadIdx.x * 8u;
		float x[8], w[8], out[8];
		LmLoadBf16x8AsFloat(raw, x);
		LmLoadWeight8(weight_bf16, offset, w);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			out[k] = SparkMathNormGainRounded(x[k] * inv, w[k]);
		*(uint4 *)(row_ptr + offset) = LmPackUint4FloatToBf16(out);
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(32u, 4)
static void HyUkRopeB1Kernel(const uint16_t *__restrict__ q_in, const uint16_t *__restrict__ uk_weight, const uint32_t *__restrict__ positions,
	uint16_t *__restrict__ query_out)
{
	using S = HyShape<G>;
	constexpr uint32_t IN_DIM = G::kNope, OUT_DIM = G::kLatent, OUTPUTS = 16u, K_TILE = 64u;
	constexpr uint32_t OUT_BLOCKS = OUT_DIM / OUTPUTS, VECTORS = IN_DIM / 8u, TILE_VECS = K_TILE / 8u, STRIDE = K_TILE + 8u;
	constexpr uint32_t HALF = G::kRope / 2u;
	static_assert(IN_DIM % K_TILE == 0u && OUT_DIM % OUTPUTS == 0u && HALF == 32u, "uk B1 and rotary dimensions match one warp");
	__shared__ __align__(16) uint16_t input_tile[IN_DIM];
	__shared__ __align__(16) uint16_t weight_tile[2][OUTPUTS][STRIDE];
	const uint32_t head = blockIdx.y;
	LmDependentWait();
	if ( blockIdx.x == OUT_BLOCKS )
	{
		const uint32_t pair = threadIdx.x;
		const uint64_t src_base = (uint64_t)head * S::kQkDim + G::kNope;
		const uint64_t dst_base = (uint64_t)head * S::kSlot + G::kLatent;
		const float angle = (float)positions[0] * __powf(G::kRopeTheta, -2.0f * (float)pair / (float)G::kRope);
		const uint32_t packed_in = *(const uint32_t *)(q_in + src_base + 2u * pair);
		const float x0 = LmBf16ToFloat((uint16_t)(packed_in & 0xffffu));
		const float x1 = LmBf16ToFloat((uint16_t)(packed_in >> 16u));
		float sv, cv;
		sincosf(angle, &sv, &cv);
		const float y0 = SparkMathRoundBf16(x0 * cv) - SparkMathRoundBf16(x1 * sv);
		const float y1 = SparkMathRoundBf16(x1 * cv) + SparkMathRoundBf16(x0 * sv);
		*(uint32_t *)(query_out + dst_base + 2u * pair) = LmPackBf16Pair(y0, y1);
		LmDependentRelease();
		return;
	}
	const uint32_t first_output = blockIdx.x * OUTPUTS;
	const uint16_t *weight_base = uk_weight + ((uint64_t)head * OUT_DIM + first_output) * IN_DIM;
	for ( uint32_t i = threadIdx.x; i < VECTORS; i += 32u )
		((uint4 *)input_tile)[i] = __ldg((const uint4 *)(q_in + (uint64_t)head * S::kQkDim) + i);
	float total = 0.0f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	const uint32_t w_base_s = (uint32_t)__cvta_generic_to_shared(weight_tile);
	constexpr uint32_t stage_bytes = OUTPUTS * STRIDE * 2u;
	auto issue_tile = [&](uint32_t stage, uint32_t k0)
	{
		const uint32_t dst_stage = w_base_s + stage * stage_bytes;
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < OUTPUTS * TILE_VECS; idx += 32u )
		{
			const uint32_t r = idx / TILE_VECS, c = idx % TILE_VECS;
			LmProjectCpAsync16(dst_stage + (r * STRIDE + c * 8u) * 2u, (const uint4 *)(weight_base + (uint64_t)r * IN_DIM + k0) + c, 16u);
		}
		LmProjectCpAsyncCommit();
	};
	issue_tile(0u, 0u);
	uint32_t stage = 0u;
	for ( uint32_t k0 = 0u; k0 < IN_DIM; k0 += K_TILE )
	{
		if ( k0 + K_TILE < IN_DIM )
			issue_tile(stage ^ 1u, k0 + K_TILE);
		if ( k0 + K_TILE < IN_DIM )
			LmProjectCpAsyncWait<1u>();
		else
			LmProjectCpAsyncWait<0u>();
		__syncwarp();
		if ( threadIdx.x < OUTPUTS )
		{
			#pragma unroll
			for ( uint32_t k = 0u; k < K_TILE; k += 8u )
				total = LmDotBf16x8(*(const uint4 *)(input_tile + k0 + k), *(const uint4 *)(weight_tile[stage][threadIdx.x] + k), total);
		}
		stage ^= 1u;
		if ( k0 + K_TILE < IN_DIM )
			__syncwarp();
	}
#else
	for ( uint32_t k0 = 0u; k0 < IN_DIM; k0 += K_TILE )
	{
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < OUTPUTS * TILE_VECS; idx += 32u )
		{
			const uint32_t r = idx / TILE_VECS, c = idx % TILE_VECS;
			*(uint4 *)(weight_tile[0][r] + c * 8u) = __ldcs((const uint4 *)(weight_base + (uint64_t)r * IN_DIM + k0) + c);
		}
		__syncwarp();
		if ( threadIdx.x < OUTPUTS )
		{
			#pragma unroll
			for ( uint32_t k = 0u; k < K_TILE; k += 8u )
				total = LmDotBf16x8(*(const uint4 *)(input_tile + k0 + k), *(const uint4 *)(weight_tile[0][threadIdx.x] + k), total);
		}
		if ( k0 + K_TILE < IN_DIM )
			__syncwarp();
	}
#endif
	if ( threadIdx.x < OUTPUTS )
		query_out[(uint64_t)head * S::kSlot + first_output + threadIdx.x] = LmFloatToBf16(total);
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(32u, 4)
static void HyUvGateB1Kernel(const uint16_t *__restrict__ latent_bf16, const uint16_t *__restrict__ uv_weight, const uint16_t *__restrict__ gate_bf16,
	uint16_t *__restrict__ value_out)
{
	constexpr uint32_t IN_DIM = G::kLatent, OUT_DIM = G::kValueDim, OUTPUTS = 16u, K_TILE = 64u;
	constexpr uint32_t VECTORS = IN_DIM / 8u, TILE_VECS = K_TILE / 8u, STRIDE = K_TILE + 8u;
	static_assert(IN_DIM % K_TILE == 0u && OUT_DIM % OUTPUTS == 0u, "uv B1 tiles divide K and output dimensions");
	__shared__ __align__(16) uint16_t input_tile[IN_DIM];
	__shared__ __align__(16) uint16_t weight_tile[2][OUTPUTS][STRIDE];
	const uint32_t first_output = blockIdx.x * OUTPUTS, head = blockIdx.y;
	const uint16_t *weight_base = uv_weight + ((uint64_t)head * OUT_DIM + first_output) * IN_DIM;
	LmDependentWait();
	for ( uint32_t i = threadIdx.x; i < VECTORS; i += 32u )
		((uint4 *)input_tile)[i] = __ldg((const uint4 *)(latent_bf16 + (uint64_t)head * IN_DIM) + i);
	float total = 0.0f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	const uint32_t w_base_s = (uint32_t)__cvta_generic_to_shared(weight_tile);
	constexpr uint32_t stage_bytes = OUTPUTS * STRIDE * 2u;
	auto issue_tile = [&](uint32_t stage, uint32_t k0)
	{
		const uint32_t dst_stage = w_base_s + stage * stage_bytes;
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < OUTPUTS * TILE_VECS; idx += 32u )
		{
			const uint32_t r = idx / TILE_VECS, c = idx % TILE_VECS;
			LmProjectCpAsync16(dst_stage + (r * STRIDE + c * 8u) * 2u, (const uint4 *)(weight_base + (uint64_t)r * IN_DIM + k0) + c, 16u);
		}
		LmProjectCpAsyncCommit();
	};
	issue_tile(0u, 0u);
	uint32_t stage = 0u;
	for ( uint32_t k0 = 0u; k0 < IN_DIM; k0 += K_TILE )
	{
		if ( k0 + K_TILE < IN_DIM )
			issue_tile(stage ^ 1u, k0 + K_TILE);
		if ( k0 + K_TILE < IN_DIM )
			LmProjectCpAsyncWait<1u>();
		else
			LmProjectCpAsyncWait<0u>();
		__syncwarp();
		if ( threadIdx.x < OUTPUTS )
		{
			#pragma unroll
			for ( uint32_t k = 0u; k < K_TILE; k += 8u )
				total = LmDotBf16x8(*(const uint4 *)(input_tile + k0 + k), *(const uint4 *)(weight_tile[stage][threadIdx.x] + k), total);
		}
		stage ^= 1u;
		if ( k0 + K_TILE < IN_DIM )
			__syncwarp();
	}
#else
	for ( uint32_t k0 = 0u; k0 < IN_DIM; k0 += K_TILE )
	{
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < OUTPUTS * TILE_VECS; idx += 32u )
		{
			const uint32_t r = idx / TILE_VECS, c = idx % TILE_VECS;
			*(uint4 *)(weight_tile[0][r] + c * 8u) = __ldcs((const uint4 *)(weight_base + (uint64_t)r * IN_DIM + k0) + c);
		}
		__syncwarp();
		if ( threadIdx.x < OUTPUTS )
		{
			#pragma unroll
			for ( uint32_t k = 0u; k < K_TILE; k += 8u )
				total = LmDotBf16x8(*(const uint4 *)(input_tile + k0 + k), *(const uint4 *)(weight_tile[0][threadIdx.x] + k), total);
		}
		if ( k0 + K_TILE < IN_DIM )
			__syncwarp();
	}
#endif
	if ( threadIdx.x < OUTPUTS )
	{
		const uint64_t dst_idx = (uint64_t)head * OUT_DIM + first_output + threadIdx.x;
		const float g = LmBf16ToFloat(__ldg(gate_bf16 + dst_idx));
		const float v = SparkMathRoundBf16(total);
		value_out[dst_idx] = LmFloatToBf16(v * SparkMathRoundBf16(SparkMathSigmoid(g)));
	}
	LmDependentRelease();
}

template<class G, uint32_t MAX_ROWS, uint32_t THREADS>
__global__ __launch_bounds__(THREADS)
static void HyUvGateChainKernel(const uint16_t *__restrict__ latent_bf16, const uint16_t *__restrict__ uv_weight, const uint16_t *__restrict__ gate_bf16,
	uint16_t *__restrict__ value_out, uint32_t rows)
{
	using S = HyShape<G>;
	constexpr uint32_t IN_DIM = G::kLatent, OUT_DIM = G::kValueDim, OUTPUTS = 16u;
	constexpr uint32_t VECTORS = IN_DIM / 8u, STRIDE = IN_DIM + 8u;
	__shared__ __align__(16) uint16_t weight_tile[OUTPUTS][STRIDE];
	__shared__ __align__(16) uint16_t input_tile[MAX_ROWS][STRIDE];
	const uint32_t first_output = blockIdx.x * OUTPUTS, head = blockIdx.y;
	const uint4 *weight = (const uint4 *)(uv_weight + ((uint64_t)head * OUT_DIM + first_output) * IN_DIM);
	LmDependentWait();
	for ( uint32_t index = threadIdx.x; index < OUTPUTS * VECTORS; index += THREADS )
		*(uint4 *)(weight_tile[index / VECTORS] + (index % VECTORS) * 8u) = __ldg(weight + index);
	for ( uint32_t index = threadIdx.x; index < rows * VECTORS; index += THREADS )
	{
		const uint32_t r = index / VECTORS, k = (index % VECTORS) * 8u;
		*(uint4 *)(input_tile[r] + k) = *(const uint4 *)(latent_bf16 + ((uint64_t)r * S::kRankHeads + head) * IN_DIM + k);
	}
	__syncthreads();
	const uint32_t output = threadIdx.x % OUTPUTS, row = threadIdx.x / OUTPUTS;
	if ( row < rows && row < MAX_ROWS )
	{
		float total = 0.0f;
		#pragma unroll 4
		for ( uint32_t k = 0u; k < IN_DIM; k += 8u )
			total = LmDotBf16x8(*(const uint4 *)(input_tile[row] + k), *(const uint4 *)(weight_tile[output] + k), total);
		const uint64_t dst_idx = ((uint64_t)row * S::kRankHeads + head) * OUT_DIM + first_output + output;
		const float g = LmBf16ToFloat(__ldg(gate_bf16 + dst_idx));
		const float v = SparkMathRoundBf16(total);
		value_out[dst_idx] = LmFloatToBf16(v * SparkMathRoundBf16(SparkMathSigmoid(g)));
	}
	LmDependentRelease();
}

template<class G>
__global__ __launch_bounds__(LM_PROJECT_MMA_THREADS)
static void HyUvGateMmaKernel(const uint16_t *__restrict__ latent_bf16, const uint16_t *__restrict__ uv_weight, const uint16_t *__restrict__ gate_bf16,
	uint16_t *__restrict__ value_out, uint32_t rows)
{
	using S = HyShape<G>;
	constexpr uint32_t IN_DIM = G::kLatent, OUT_DIM = G::kValueDim, HEADS = S::kRankHeads;
	constexpr uint32_t WORDS = LM_PROJECT_MMA_K / 8u, TILES = LM_PROJECT_MMA_OUTPUTS / LM_MMA16_N;
	constexpr uint32_t ROW_BYTES = LM_PROJECT_MMA_K * 2u;
	constexpr uint32_t A_STAGE_BYTES = LM_PROJECT_MMA_ROWS * ROW_BYTES;
	constexpr uint32_t B_STAGE_BYTES = LM_PROJECT_MMA_OUTPUTS * ROW_BYTES;
	constexpr uint32_t STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;
	const uint32_t first_output = blockIdx.x * LM_PROJECT_MMA_OUTPUTS, head = blockIdx.y, first_row = blockIdx.z * LM_PROJECT_MMA_ROWS;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, warp_row = (threadIdx.x / LM_WARP_LANES) * LM_MMA16_M;
	const uint16_t *weight = uv_weight + ((uint64_t)head * OUT_DIM + first_output) * IN_DIM;
	float accumulator[TILES][4];
	#pragma unroll
	for ( uint32_t tile = 0u; tile < TILES; ++tile )
		#pragma unroll
		for ( uint32_t reg = 0u; reg < 4u; ++reg )
			accumulator[tile][reg] = 0.0f;
	LmDependentWait();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	__shared__ __align__(128) uint8_t shared_stages[2u * STAGE_BYTES];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(shared_stages);
	const uint32_t k_tiles = IN_DIM / LM_PROJECT_MMA_K;
	const uint32_t ldm_a_row = warp_row + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	const uint32_t ldm_b_row = (lane & 7u) + (((lane >> 4u) & 1u) * 8u), ldm_b_col = (lane >> 3u) & 1u;
	auto issue_k = [&](uint32_t stage_off, uint32_t k0)
	{
		const uint32_t stage_a = shared_base + stage_off;
		const uint32_t stage_b = stage_a + A_STAGE_BYTES;
		#pragma unroll
		for ( uint32_t index = threadIdx.x; index < LM_PROJECT_MMA_ROWS * WORDS; index += LM_PROJECT_MMA_THREADS )
		{
			const uint32_t r = index / WORDS, c = index % WORDS;
			const uint32_t valid = first_row + r < rows ? 16u : 0u;
			const uint16_t *src = valid != 0u ? latent_bf16 + ((uint64_t)(first_row + r) * HEADS + head) * IN_DIM + k0 + c * 8u : latent_bf16;
			LmProjectCpAsync16(stage_a + r * ROW_BYTES + LmProjectMmaSwizzle8(r, c) * 16u, src, valid);
		}
		#pragma unroll
		for ( uint32_t index = threadIdx.x; index < LM_PROJECT_MMA_OUTPUTS * WORDS; index += LM_PROJECT_MMA_THREADS )
		{
			const uint32_t r = index / WORDS, c = index % WORDS;
			LmProjectCpAsync16(stage_b + r * ROW_BYTES + LmProjectMmaSwizzle8(r, c) * 16u, weight + (uint64_t)r * IN_DIM + k0 + c * 8u, 16u);
		}
		LmProjectCpAsyncCommit();
	};
	issue_k(0u, 0u);
	uint32_t cur_off = 0u;
	for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
	{
		if ( kt + 1u < k_tiles )
			issue_k(cur_off ^ STAGE_BYTES, (kt + 1u) * LM_PROJECT_MMA_K);
		if ( kt + 1u < k_tiles )
			LmProjectCpAsyncWait<1u>();
		else
			LmProjectCpAsyncWait<0u>();
		__syncthreads();
		const uint32_t stage_a = shared_base + cur_off;
		const uint32_t stage_b = stage_a + A_STAGE_BYTES;
		uint32_t a[4], b_tmp[4];
		#pragma unroll
		for ( uint32_t ks = 0u; ks < LM_PROJECT_MMA_K / LM_MMA16_K; ++ks )
		{
			LmProjectLdMatrix4(a, stage_a + ldm_a_row * ROW_BYTES + LmProjectMmaSwizzle8(ldm_a_row, ks * 2u + ldm_a_col) * 16u);
			#pragma unroll
			for ( uint32_t tile = 0u; tile < TILES; tile += 2u )
			{
				const uint32_t b_row = tile * LM_MMA16_N + ldm_b_row;
				LmProjectLdMatrix4(b_tmp, stage_b + b_row * ROW_BYTES + LmProjectMmaSwizzle8(b_row, ks * 2u + ldm_b_col) * 16u);
				uint32_t b0[2] = { b_tmp[0], b_tmp[1] }, b1[2] = { b_tmp[2], b_tmp[3] };
				LmMmaBf16(accumulator[tile], a, b0);
				LmMmaBf16(accumulator[tile + 1u], a, b1);
			}
		}
		cur_off ^= STAGE_BYTES;
		if ( kt + 1u < k_tiles )
			__syncthreads();
	}
#else
	__shared__ __align__(16) uint16_t input_tile[LM_PROJECT_MMA_ROWS * LM_PROJECT_MMA_PITCH];
	__shared__ __align__(16) uint16_t weight_tile[LM_PROJECT_MMA_OUTPUTS * LM_PROJECT_MMA_PITCH];
	for ( uint32_t k = 0u; k < IN_DIM; k += LM_PROJECT_MMA_K )
	{
		for ( uint32_t index = threadIdx.x; index < LM_PROJECT_MMA_ROWS * WORDS; index += LM_PROJECT_MMA_THREADS )
		{
			const uint32_t r = index / WORDS, c = (index % WORDS) * 8u;
			*(uint4 *)(input_tile + r * LM_PROJECT_MMA_PITCH + c) = first_row + r < rows ?
				__ldg((const uint4 *)(latent_bf16 + ((uint64_t)(first_row + r) * HEADS + head) * IN_DIM + k + c)) : make_uint4(0u, 0u, 0u, 0u);
		}
		for ( uint32_t index = threadIdx.x; index < LM_PROJECT_MMA_OUTPUTS * WORDS; index += LM_PROJECT_MMA_THREADS )
		{
			const uint32_t r = index / WORDS, c = (index % WORDS) * 8u;
			*(uint4 *)(weight_tile + r * LM_PROJECT_MMA_PITCH + c) = __ldg((const uint4 *)(weight + (uint64_t)r * IN_DIM + k + c));
		}
		__syncthreads();
		for ( uint32_t step = 0u; step < LM_PROJECT_MMA_K; step += LM_MMA16_K )
		{
			uint32_t a[4], b[2];
			for ( uint32_t reg = 0u; reg < 4u; ++reg )
				a[reg] = *(const uint32_t *)(input_tile + (warp_row + LmMma16OperandARow(lane, reg)) * LM_PROJECT_MMA_PITCH + step + LmMma16OperandAK(lane, reg));
			for ( uint32_t tile = 0u; tile < TILES; ++tile )
			{
				for ( uint32_t reg = 0u; reg < 2u; ++reg )
					b[reg] = *(const uint32_t *)(weight_tile + (tile * LM_MMA16_N + LmMma16OperandBRow(lane)) * LM_PROJECT_MMA_PITCH + step + LmMma16OperandBK(lane, reg));
				LmMmaBf16(accumulator[tile], a, b);
			}
		}
		__syncthreads();
	}
#endif
	#pragma unroll
	for ( uint32_t tile = 0u; tile < TILES; ++tile )
		#pragma unroll
		for ( uint32_t reg = 0u; reg < 4u; reg += 2u )
		{
			const uint32_t row = first_row + warp_row + LmMmaAccumulatorRow(lane, reg);
			const uint32_t column = first_output + tile * LM_MMA16_N + LmMmaAccumulatorColumn(lane, reg);
			if ( row < rows )
			{
				const uint64_t dst_off = ((uint64_t)row * HEADS + head) * OUT_DIM + column;
				const uint32_t gp = __ldg((const uint32_t *)(gate_bf16 + dst_off));
				const float g0 = LmBf16ToFloat((uint16_t)(gp & 0xffffu));
				const float g1 = LmBf16ToFloat((uint16_t)(gp >> 16u));
				const float v0 = SparkMathRoundBf16(accumulator[tile][reg]) * SparkMathRoundBf16(SparkMathSigmoid(g0));
				const float v1 = SparkMathRoundBf16(accumulator[tile][reg + 1u]) * SparkMathRoundBf16(SparkMathSigmoid(g1));
				*(uint32_t *)(value_out + dst_off) = LmPackBf16Pair(v0, v1);
			}
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

template<class G, bool WRITE_CUR_KV = false>
__global__ __launch_bounds__(HY_THREADS, 2)
static void HyKvPrepStoreKernel(uint16_t *__restrict__ kv_in, const uint16_t *__restrict__ kv_norm, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ store_positions, const uint32_t *__restrict__ sequence_of_row, LmKvShardView kv_view)
{
	using S = HyShape<G>;
	constexpr uint32_t LAT = G::kLatent, HALF = G::kRope / 2u;
	constexpr uint32_t SLOTS = (LAT + HY_THREADS - 1u) / HY_THREADS;
	__shared__ float reduction[LmBlockWarps<HY_THREADS>()];
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	const uint32_t store_pos = store_positions[row];
	const bool owns = SparkKvShardOwns(kv_view.shard, store_pos) != 0u;
	if ( owns || WRITE_CUR_KV )
	{
		uint16_t *slot = owns ? (uint16_t *)LmKvShardSlotRequired<typename S::Kv>(kv_view, sequence_of_row[row], store_pos, row, LM_KV_ACCESS_WRITE) : 0;
		if ( slot != 0 || WRITE_CUR_KV )
		{
			uint16_t *src = kv_in + (uint64_t)row * S::kSlot;
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
				{
					const uint16_t out_v = LmFloatToBf16(SparkMathNormGainRounded(vals[s] * scale, LmBf16ToFloat(kv_norm[idx])));
					if ( slot != 0 )
						slot[idx] = out_v;
					if constexpr ( WRITE_CUR_KV )
						src[idx] = out_v;
				}
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
				const uint32_t packed_out = LmPackBf16Pair(y0, y1);
				if ( slot != 0 )
					*(uint32_t *)(slot + LAT + 2u * pair) = packed_out;
				if constexpr ( WRITE_CUR_KV )
					*(uint32_t *)(src + LAT + 2u * pair) = packed_out;
			}
		}
	}
	LmDependentRelease();
}

template<class G, bool KEY_ONLY>
__global__ __launch_bounds__(128u, 2)
static void HyIndexPrepStoreKernel(uint16_t *__restrict__ index_q, const uint16_t *__restrict__ index_k, const float *__restrict__ k_norm,
	const float *__restrict__ k_bias, float *__restrict__ index_weights, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ store_positions, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	LmKvShardView index_view, float weight_scale)
{
	using S = HyShape<G>;
	constexpr uint32_t HALF = G::kRope / 2u;
	constexpr uint32_t ROPE_OFF = G::kIndexDim - G::kRope;
	constexpr uint32_t D = G::kIndexDim;
	static_assert(D == 128u && HALF == 32u, "indexer head dimension is 128 with 64 rotary dimensions");
	__shared__ float reduction[LmBlockWarps<128u>()];
	const uint32_t task = KEY_ONLY ? G::kIndexHeads : blockIdx.x, row = KEY_ONLY ? blockIdx.x : blockIdx.y;
	LmDependentWait();
	if ( !KEY_ONLY && task < G::kIndexHeads )
	{
		const uint32_t pos = positions[row];
		uint32_t bound = context_length[sequence_of_row[row]];
		if ( pos + 1u < bound )
			bound = pos + 1u;
		if ( bound > G::kIndexTopk && threadIdx.x < HALF )
		{
			const uint32_t pair = threadIdx.x;
			uint16_t *base = index_q + ((uint64_t)row * G::kIndexHeads + task) * D + ROPE_OFF;
			const float angle = (float)pos * __powf(G::kRopeTheta, -2.0f * (float)pair / (float)G::kRope);
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
		if ( !KEY_ONLY && threadIdx.x < G::kIndexHeads )
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

__global__ __launch_bounds__(32u, 1)
static void HyGatherMetaKernel(const uint32_t *__restrict__ sequence_of_row, uint32_t context, uint32_t *__restrict__ gather_seq,
	uint32_t *__restrict__ gather_offset, uint32_t *__restrict__ gather_context)
{
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		const uint32_t seq = sequence_of_row[0];
		gather_seq[0] = seq;
		gather_offset[seq] = 0u;
		gather_context[seq] = context;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(HY_THREADS, 2)
static void HyPadCandidatesKernel(uint2 *__restrict__ candidates, uint32_t rows, uint32_t padded_rows, uint32_t keep)
{
	const uint64_t first = (uint64_t)rows * keep;
	const uint64_t total = (uint64_t)padded_rows * keep;
	const uint2 empty = make_uint2(0u, LM_INDEX_SHARD_NO_POSITION);
	LmDependentWait();
	for ( uint64_t i = first + (uint64_t)blockIdx.x * HY_THREADS + threadIdx.x; i < total; i += (uint64_t)gridDim.x * HY_THREADS )
		candidates[i] = empty;
	LmDependentRelease();
}

template<uint32_t TOPK>
__global__ __launch_bounds__(HY_THREADS, 2)
static void HyFillDenseSelectedKernel(const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ positions, uint32_t *__restrict__ selected)
{
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	const uint32_t pos = positions[row];
	uint32_t bound = context_length[sequence_of_row[row]];
	if ( pos + 1u < bound )
		bound = pos + 1u;
	if ( bound <= TOPK )
	{
		uint32_t *dst = selected + (uint64_t)row * TOPK;
		for ( uint32_t i = threadIdx.x; i < TOPK; i += HY_THREADS )
			dst[i] = i < bound ? i : LM_INDEX_SHARD_NO_POSITION;
	}
	LmDependentRelease();
}

template<class G, uint32_t ROWS_PER_BLOCK>
__global__ __launch_bounds__(LM_INDEX_SCORE_THREADS)
static void HyIndexScoreSkipKernel(const uint16_t *__restrict__ index_query_bf16, const float *__restrict__ head_weight,
	LmKvShardView index_view, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position, uint32_t rows, uint32_t score_stride, float qk_scale, float *__restrict__ scores,
	const uint32_t *__restrict__ fork_positions, const uint32_t *__restrict__ slot_offsets, uint32_t dense_limit)
{
	using S = HyShape<G>;
	constexpr uint32_t HEAD_TILES = G::kIndexHeads / 16u, STEPS = G::kIndexDim / 16u;
	constexpr uint32_t WARPS = LM_INDEX_SCORE_WARPS;
	LmDependentWait();
	const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	const uint32_t group = lane >> 2u, pair = lane & 3u;
	const uint32_t row = ROWS_PER_BLOCK == 1u ? blockIdx.y : (blockIdx.y * WARPS + warp);
	if ( scores == 0 || row >= rows )
	{
		LmDependentRelease();
		return;
	}
	const LmKvView &index_cache = LmIndexScorePages(index_view);
	const uint32_t valid = sequence_of_row != 0 && context_length != 0 && index_query_bf16 != 0 && head_weight != 0 && LmKvViewIsConfigured(index_cache) ? 1u : 0u;
	const uint32_t sequence = valid != 0u ? sequence_of_row[row] : 0xffffffffu;
	if ( valid == 0u || sequence >= index_cache.sequence_count )
	{
		if ( lane == 0u && blockIdx.x == 0u && (ROWS_PER_BLOCK != 1u || warp == 0u) )
			LmKvReportRequiredAccessFailure(index_cache, valid == 0u ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
				LM_KV_ACCESS_READ, row, sequence, 0u, 0xffffffffu);
		LmDependentRelease();
		return;
	}
	uint32_t bound = context_length[sequence];
	if ( row_position != 0 && row_position[row] + 1u < bound )
		bound = row_position[row] + 1u;
	if ( dense_limit != 0u && bound <= dense_limit )
	{
		LmDependentRelease();
		return;
	}
	bound = LmIndexScoreKeys(index_view, bound);
	const uint32_t limit = bound < score_stride ? bound : score_stride;
	const uint32_t total_tiles = (limit + LM_INDEX_SCORE_TILE_POSITIONS - 1u) / LM_INDEX_SCORE_TILE_POSITIONS;
	const uint32_t tile0 = ROWS_PER_BLOCK == 1u ? (blockIdx.x * WARPS + warp) : blockIdx.x;
	const uint32_t tile_step = ROWS_PER_BLOCK == 1u ? (gridDim.x * WARPS) : gridDim.x;
	if ( tile0 >= total_tiles )
	{
		LmDependentRelease();
		return;
	}
	uint32_t a[HEAD_TILES][STEPS][4];
	float weight[2u * HEAD_TILES];
	const uint16_t *query = index_query_bf16 + (uint64_t)row * G::kIndexHeads * G::kIndexDim;
	#pragma unroll
	for ( uint32_t ht = 0u; ht < HEAD_TILES; ++ht )
		#pragma unroll
		for ( uint32_t step = 0u; step < STEPS; ++step )
		{
			const uint16_t *top = query + (uint64_t)(ht * 16u + group) * G::kIndexDim + step * 16u + pair * 2u;
			const uint16_t *bottom = top + 8u * G::kIndexDim;
			a[ht][step][0] = __ldg((const uint32_t *)top);
			a[ht][step][1] = __ldg((const uint32_t *)bottom);
			a[ht][step][2] = __ldg((const uint32_t *)(top + 8u));
			a[ht][step][3] = __ldg((const uint32_t *)(bottom + 8u));
		}
	#pragma unroll
	for ( uint32_t ht = 0u; ht < HEAD_TILES; ++ht )
	{
		weight[2u * ht] = LmIndexScoreWeight(head_weight[(uint64_t)row * G::kIndexHeads + ht * 16u + group]);
		weight[2u * ht + 1u] = LmIndexScoreWeight(head_weight[(uint64_t)row * G::kIndexHeads + ht * 16u + group + 8u]);
	}
	const uint32_t fork = fork_positions != 0 ? fork_positions[row] : 0xffffffffu;
	const uint32_t offset = slot_offsets != 0 ? slot_offsets[row] : 0u;
	uint32_t b[STEPS][2], next[STEPS][2];
	const uint32_t first0 = tile0 * LM_INDEX_SCORE_TILE_POSITIONS;
	LmIndexScoreKeyFragments<STEPS>(first0 + group < limit ?
		LmIndexScoreSlot<typename S::IndexKv>(index_view, sequence, first0 + group, row, fork, offset) : 0, pair, next);
	for ( uint32_t tile = tile0; tile < total_tiles; tile += tile_step )
	{
		const uint32_t first = tile * LM_INDEX_SCORE_TILE_POSITIONS;
		const uint32_t next_tile = tile + tile_step;
		const uint32_t ahead = next_tile * LM_INDEX_SCORE_TILE_POSITIONS + group;
		float accumulator[HEAD_TILES][4] = {};
		#pragma unroll
		for ( uint32_t step = 0u; step < STEPS; ++step )
		{
			b[step][0] = next[step][0];
			b[step][1] = next[step][1];
		}
		if ( next_tile < total_tiles )
			LmIndexScoreKeyFragments<STEPS>(ahead < limit ?
				LmIndexScoreSlot<typename S::IndexKv>(index_view, sequence, ahead, row, fork, offset) : 0, pair, next);
		#pragma unroll
		for ( uint32_t step = 0u; step < STEPS; ++step )
			#pragma unroll
			for ( uint32_t ht = 0u; ht < HEAD_TILES; ++ht )
				LmMmaBf16(accumulator[ht], a[ht][step], b[step]);
		LmIndexScoreStore<HEAD_TILES, false>(accumulator, weight, 1u, bound, first + pair * 2u, group, row, score_stride, qk_scale, scores);
	}
	LmDependentRelease();
}

static __device__ __forceinline__ void HyPrefillCopyL2(uint32_t destination, const void *source, uint32_t bytes)
{
	asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n" :: "r"(destination), "l"(source), "r"(bytes) : "memory");
}

template<class Geometry, uint32_t HEADS, uint32_t LATENT, uint32_t ROPE, bool SPARSE>
__global__ __launch_bounds__(LM_PREFILL_ATTN_THREADS, 1)
static void HyGatherPrefillKernel(
	const uint16_t *__restrict__ query_bf16,
	LmKvShardGatherAllView cache,
	const uint16_t *__restrict__ kv_cur,
	uint32_t old_ctx,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ row_position,
	const float *__restrict__ sinks,
	float qk_scale,
	uint16_t *__restrict__ output_bf16,
	uint32_t rows,
	const uint32_t *__restrict__ union_positions,
	const uint8_t *__restrict__ union_masks,
	const uint32_t *__restrict__ union_counts,
	uint32_t union_capacity)
{
	constexpr uint32_t QUERIES = LM_PREFILL_ATTN_QUERIES;
	constexpr uint32_t DIM = LATENT + ROPE;
	constexpr uint32_t ROW_BYTES = DIM * 2u;
	constexpr uint32_t CHUNKS = DIM / 8u;
	constexpr uint32_t TILE = LM_PREFILL_ATTN_TILE;
	constexpr uint32_t K_STEPS = DIM / 16u;
	constexpr uint32_t HALF_STEPS = K_STEPS / 2u;
	constexpr uint32_t ROWS_PER_BLOCK = QUERIES / HEADS;
	constexpr uint32_t KEY_GROUPS = LM_PREFILL_ATTN_WARPS / 2u;
	constexpr uint32_t KEY_FRAGS = TILE / (KEY_GROUPS * 8u);
	constexpr uint32_t WARP_COLUMNS = LATENT / LM_PREFILL_ATTN_WARPS;
	constexpr uint32_t COLUMN_FRAGS = WARP_COLUMNS / 8u;
	constexpr uint32_t QUERY_KEYS = QUERIES * TILE / LM_PREFILL_ATTN_THREADS;
	constexpr uint32_t QUERY_THREADS = TILE / QUERY_KEYS;
	static_assert(HEADS == 4u && ROWS_PER_BLOCK == 4u && QUERIES == 16u && KEY_FRAGS == 1u && QUERY_KEYS == 2u,
		"4-head 4-row grouped sparse prefill fills one 16-query MMA tile per CTA");
	extern __shared__ __align__(128) uint8_t hy_prefill_smem[];
	uint8_t *keys = hy_prefill_smem;
	float *score_parts = (float *)(keys + 2u * TILE * ROW_BYTES);
	uint16_t *weight_high = (uint16_t *)(score_parts + 2u * QUERIES * LM_PREFILL_ATTN_PARTIAL_STRIDE);
	float *running_max = (float *)(weight_high + QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE);
	float *running_sum = running_max + QUERIES;
	float *rescale = running_sum + QUERIES;
	const uint8_t **slots = (const uint8_t **)(rescale + QUERIES);
	uint32_t *key_rows = (uint32_t *)(slots + 2u * TILE);
	LmDependentWait();
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t half = (warp / KEY_GROUPS) & 1u, key_group = warp % KEY_GROUPS;
	const uint32_t table = gridDim.x - 1u - blockIdx.x;
	const uint32_t *tile_positions = SPARSE ? union_positions + (uint64_t)table * union_capacity : (const uint32_t *)0;
	const uint8_t *tile_masks = SPARSE ? union_masks + (uint64_t)table * union_capacity : (const uint8_t *)0;
	const uint32_t first_row = table * ROWS_PER_BLOCK;
	const uint32_t row_end = LmPrefillMin(rows, first_row + ROWS_PER_BLOCK);
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	const uint32_t query_base = key_base + TILE * ROW_BYTES;
	uint32_t query_frag[HALF_STEPS][4];
	float output[COLUMN_FRAGS][4];
	uint32_t row_limit[ROWS_PER_BLOCK];
	const float scale_log2 = qk_scale * 1.4426950408889634f;
	const uint32_t sequence = __ldg(sequence_of_row + first_row);
	uint32_t first_position_min = 0xffffffffu, last_position = 0u;
	for ( uint32_t r = first_row; r < row_end; ++r )
	{
		const uint32_t rpos = __ldg(row_position + r);
		first_position_min = rpos < first_position_min ? rpos : first_position_min;
		last_position = rpos > last_position ? rpos : last_position;
	}
	const uint32_t listed_keys = SPARSE ? __ldg(union_counts + table) : last_position + 1u;
	const uint32_t tiles = (listed_keys + TILE - 1u) / TILE;
	#pragma unroll
	for ( uint32_t r = 0u; r < ROWS_PER_BLOCK; ++r )
		row_limit[r] = first_row + r < row_end ? __ldg(row_position + first_row + r) : 0u;
	auto listing = [&](uint32_t which, uint32_t &position, uint32_t &members)
	{
		const uint32_t listed = which * TILE + threadIdx.x;
		position = !SPARSE ? listed : (listed < listed_keys ? __ldg(tile_positions + listed) : 0xffffffffu);
		members = !SPARSE ? 0xffffffffu : (listed < listed_keys ? (uint32_t)__ldg(tile_masks + listed) : 0u);
	};
	auto resolve = [&](uint32_t position, uint32_t members, uint32_t &valid) -> const uint8_t *
	{
		valid = 0u;
		#pragma unroll
		for ( uint32_t r = 0u; r < ROWS_PER_BLOCK; ++r )
			valid |= (first_row + r < row_end && ((members >> r) & 1u) != 0u && position <= row_limit[r] ? 1u : 0u) << r;
		if ( position > last_position )
			return (const uint8_t *)0;
		if ( kv_cur != 0 && position >= old_ctx && position - old_ctx < rows )
			return (const uint8_t *)(kv_cur + (uint64_t)(position - old_ctx) * DIM);
		return LmKvShardSlotRequired<Geometry>(cache, sequence, position, first_row, LM_KV_ACCESS_READ);
	};
	auto copy = [&](uint32_t which)
	{
		const uint32_t buffer = which & 1u;
		constexpr uint32_t SLOTS_PER_WARP = TILE / LM_PREFILL_ATTN_WARPS;
		static_assert(CHUNKS == 72u && SLOTS_PER_WARP == 4u, "hy4 gather prefill copies 4 576-bf16 slots per warp");
		const uint32_t slot_base = warp * SLOTS_PER_WARP;
		const uint32_t buf_base = key_base + (buffer * TILE + slot_base) * ROW_BYTES;
		const uint8_t *s0 = slots[buffer * TILE + slot_base + 0u];
		const uint8_t *s1 = slots[buffer * TILE + slot_base + 1u];
		const uint8_t *s2 = slots[buffer * TILE + slot_base + 2u];
		const uint8_t *s3 = slots[buffer * TILE + slot_base + 3u];
		const uint32_t b0 = s0 != 0 ? 16u : 0u;
		const uint32_t b1 = s1 != 0 ? 16u : 0u;
		const uint32_t b2 = s2 != 0 ? 16u : 0u;
		const uint32_t b3 = s3 != 0 ? 16u : 0u;
		const uint8_t *p0 = s0 != 0 ? s0 : (const uint8_t *)query_bf16;
		const uint8_t *p1 = s1 != 0 ? s1 : (const uint8_t *)query_bf16;
		const uint8_t *p2 = s2 != 0 ? s2 : (const uint8_t *)query_bf16;
		const uint8_t *p3 = s3 != 0 ? s3 : (const uint8_t *)query_bf16;
		HyPrefillCopyL2(buf_base + 0u * ROW_BYTES + LmPrefillSwizzle(slot_base + 0u, lane) * 16u, p0 + lane * 16u, b0);
		HyPrefillCopyL2(buf_base + 0u * ROW_BYTES + LmPrefillSwizzle(slot_base + 0u, lane + 32u) * 16u, p0 + (lane + 32u) * 16u, b0);
		HyPrefillCopyL2(buf_base + 1u * ROW_BYTES + LmPrefillSwizzle(slot_base + 1u, lane) * 16u, p1 + lane * 16u, b1);
		HyPrefillCopyL2(buf_base + 1u * ROW_BYTES + LmPrefillSwizzle(slot_base + 1u, lane + 32u) * 16u, p1 + (lane + 32u) * 16u, b1);
		HyPrefillCopyL2(buf_base + 2u * ROW_BYTES + LmPrefillSwizzle(slot_base + 2u, lane) * 16u, p2 + lane * 16u, b2);
		HyPrefillCopyL2(buf_base + 2u * ROW_BYTES + LmPrefillSwizzle(slot_base + 2u, lane + 32u) * 16u, p2 + (lane + 32u) * 16u, b2);
		HyPrefillCopyL2(buf_base + 3u * ROW_BYTES + LmPrefillSwizzle(slot_base + 3u, lane) * 16u, p3 + lane * 16u, b3);
		HyPrefillCopyL2(buf_base + 3u * ROW_BYTES + LmPrefillSwizzle(slot_base + 3u, lane + 32u) * 16u, p3 + (lane + 32u) * 16u, b3);
		const uint32_t ts = lane >> 3u, tp = 64u + (lane & 7u);
		const uint8_t *pt = ts == 0u ? p0 : (ts == 1u ? p1 : (ts == 2u ? p2 : p3));
		const uint32_t bt = ts == 0u ? b0 : (ts == 1u ? b1 : (ts == 2u ? b2 : b3));
		HyPrefillCopyL2(buf_base + ts * ROW_BYTES + LmPrefillSwizzle(slot_base + ts, tp) * 16u, pt + tp * 16u, bt);
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	for ( uint32_t idx = threadIdx.x; idx < QUERIES * CHUNKS; idx += LM_PREFILL_ATTN_THREADS )
	{
		const uint32_t q = idx / CHUNKS, chunk = idx % CHUNKS;
		const uint32_t row = first_row + q / HEADS;
		const uint16_t *src = row >= row_end ? query_bf16 : query_bf16 + ((uint64_t)row * HEADS + q % HEADS) * DIM + chunk * 8u;
		LmPrefillCopy(query_base + q * ROW_BYTES + LmPrefillSwizzle(q, chunk) * 16u, src, row >= row_end ? 0u : 16u);
	}
	asm volatile("cp.async.commit_group;\n" ::: "memory");
	uint32_t next_position = 0xffffffffu, next_members = 0u;
	if ( threadIdx.x < TILE )
	{
		uint32_t valid, p0, m0, p1 = 0xffffffffu, m1 = 0u;
		if ( tiles > 0u ) listing(0u, p0, m0);
		if ( tiles > 1u ) listing(1u, p1, m1);
		if ( tiles > 2u ) listing(2u, next_position, next_members);
		if ( tiles > 0u ) { slots[threadIdx.x] = resolve(p0, m0, valid); key_rows[threadIdx.x] = valid; }
		if ( tiles > 1u ) { slots[TILE + threadIdx.x] = resolve(p1, m1, valid); key_rows[TILE + threadIdx.x] = valid; }
	}
	__syncthreads();
	if ( tiles > 0u )
	{
		copy(0u);
		asm volatile("cp.async.wait_group 1;\n" ::: "memory");
	}
	else
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	{
		const uint32_t a_row = (matrix & 1u) * 8u + within;
		#pragma unroll
		for ( uint32_t step = 0u; step < HALF_STEPS; ++step )
			LmPrefillLoadX4(query_base + a_row * ROW_BYTES + LmPrefillSwizzle(a_row, (half * HALF_STEPS + step) * 2u + (matrix >> 1u)) * 16u,
				query_frag[step]);
	}
	if ( threadIdx.x < QUERIES )
	{
		running_max[threadIdx.x] = -INFINITY;
		running_sum[threadIdx.x] = 0.0f;
	}
	#pragma unroll
	for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; ++frag )
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;
	__syncthreads();
	for ( uint32_t tile = 0u; tile < tiles; ++tile )
	{
		const uint32_t buffer = tile & 1u;
		const uint32_t tile_base = key_base + buffer * TILE * ROW_BYTES;
		uint32_t ahead_position = 0xffffffffu, ahead_members = 0u;
		if ( tile + 1u < tiles )
		{
			copy(tile + 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		if ( threadIdx.x < TILE && tile + 3u < tiles )
			listing(tile + 3u, ahead_position, ahead_members);
		__syncthreads();
		{
			float score[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			uint32_t b[2];
			const uint32_t b_row = key_group * 8u + within;
			#pragma unroll
			for ( uint32_t step = 0u; step < HALF_STEPS; ++step )
			{
				const uint32_t b_chunk = (half * HALF_STEPS + step) * 2u + (matrix & 1u);
				LmPrefillLoadX2(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, b_chunk) * 16u, b);
				LmMmaBf16(score, query_frag[step], b);
			}
			const uint32_t base_col = key_group * 8u + (lane & 3u) * 2u;
			const uint32_t row_lo = half * QUERIES + (lane >> 2u);
			*(float2 *)(score_parts + row_lo * LM_PREFILL_ATTN_PARTIAL_STRIDE + base_col) = make_float2(score[0], score[1]);
			*(float2 *)(score_parts + (row_lo + 8u) * LM_PREFILL_ATTN_PARTIAL_STRIDE + base_col) = make_float2(score[2], score[3]);
		}
		__syncthreads();
		{
			const uint32_t own = threadIdx.x / QUERY_THREADS, part = threadIdx.x % QUERY_THREADS;
			const uint32_t key_base_idx = part * QUERY_KEYS, own_row = own / HEADS;
			const bool full_dense_tile = !SPARSE && (first_row + own_row < row_end) && ((tile + 1u) * TILE - 1u <= first_position_min);
			const float2 a = *(const float2 *)(score_parts + own * LM_PREFILL_ATTN_PARTIAL_STRIDE + key_base_idx);
			const float2 b = *(const float2 *)(score_parts + (QUERIES + own) * LM_PREFILL_ATTN_PARTIAL_STRIDE + key_base_idx);
			float v0 = a.x + b.x, v1 = a.y + b.y;
			if ( !full_dense_tile )
			{
				v0 = ((key_rows[buffer * TILE + key_base_idx] >> own_row) & 1u) != 0u ? v0 : -INFINITY;
				v1 = ((key_rows[buffer * TILE + key_base_idx + 1u] >> own_row) & 1u) != 0u ? v1 : -INFINITY;
			}
			float peak = fmaxf(v0, v1);
			#pragma unroll
			for ( uint32_t shift = 1u; shift < QUERY_THREADS; shift <<= 1u )
				peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, shift));
			const float previous_max = running_max[own];
			const float next_max = fmaxf(previous_max, peak);
			const float scale = previous_max == -INFINITY ? 0.0f : SparkMathExp2Fast((previous_max - next_max) * scale_log2);
			const float shift_val = next_max == -INFINITY ? 0.0f : next_max * scale_log2;
			const float w0 = full_dense_tile ? SparkMathExp2Fast(fmaf(v0, scale_log2, -shift_val)) :
				(next_max == -INFINITY || v0 == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(v0, scale_log2, -shift_val)));
			const float w1 = full_dense_tile ? SparkMathExp2Fast(fmaf(v1, scale_log2, -shift_val)) :
				(next_max == -INFINITY || v1 == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(v1, scale_log2, -shift_val)));
			*(uint32_t *)(weight_high + own * LM_PREFILL_ATTN_WEIGHT_STRIDE + key_base_idx) = LmPackBf16Pair(w0, w1);
			float total = w0 + w1;
			#pragma unroll
			for ( uint32_t shift = 1u; shift < QUERY_THREADS; shift <<= 1u )
				total += __shfl_xor_sync(0xffffffffu, total, shift);
			if ( part == 0u )
			{
				running_sum[own] = fmaf(running_sum[own], scale, total);
				running_max[own] = next_max;
				rescale[own] = scale;
			}
		}
		__syncthreads();
		{
			uint32_t a_high[4], b[4], first[2], second[2];
			const float low_scale = rescale[lane >> 2u], high_scale = rescale[(lane >> 2u) + 8u];
			if ( low_scale != 1.0f || high_scale != 1.0f )
			{
				#pragma unroll
				for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; ++frag )
				{
					output[frag][0] *= low_scale;
					output[frag][1] *= low_scale;
					output[frag][2] *= high_scale;
					output[frag][3] *= high_scale;
				}
			}
			#pragma unroll
			for ( uint32_t step = 0u; step < TILE / 16u; ++step )
			{
				const uint32_t a_row = (matrix & 1u) * 8u + within, a_col = step * 16u + (matrix >> 1u) * 8u;
				LmPrefillLoadX4((uint32_t)__cvta_generic_to_shared(weight_high + a_row * LM_PREFILL_ATTN_WEIGHT_STRIDE + a_col), a_high);
				#pragma unroll
				for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; frag += 2u )
				{
					const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
					const uint32_t b_chunk = (warp * WARP_COLUMNS + frag * 8u) / 8u + (matrix >> 1u);
					LmPrefillLoadX4Trans(tile_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, b_chunk) * 16u, b);
					first[0] = b[0]; first[1] = b[1];
					second[0] = b[2]; second[1] = b[3];
					LmMmaBf16(output[frag], a_high, first);
					LmMmaBf16(output[frag + 1u], a_high, second);
				}
			}
		}
		if ( threadIdx.x < TILE && tile + 2u < tiles )
		{
			uint32_t valid;
			slots[buffer * TILE + threadIdx.x] = resolve(next_position, next_members, valid);
			key_rows[buffer * TILE + threadIdx.x] = valid;
		}
		next_position = ahead_position;
		next_members = ahead_members;
		__syncthreads();
	}
	const uint32_t q_lo = lane >> 2u, q_hi = q_lo + 8u;
	const uint32_t row_lo = first_row + q_lo / HEADS, row_hi = first_row + q_hi / HEADS;
	const float m_lo = running_max[q_lo] == -INFINITY ? -INFINITY : running_max[q_lo] * qk_scale;
	const float m_hi = running_max[q_hi] == -INFINITY ? -INFINITY : running_max[q_hi] * qk_scale;
	const float sink_lo = __ldg(sinks + q_lo % HEADS);
	const float sink_hi = __ldg(sinks + q_hi % HEADS);
	const float gm_lo = fmaxf(m_lo, sink_lo), gm_hi = fmaxf(m_hi, sink_hi);
	const float s_lo = m_lo == -INFINITY ? 0.0f : __expf(m_lo - gm_lo);
	const float s_hi = m_hi == -INFINITY ? 0.0f : __expf(m_hi - gm_hi);
	const float inv_lo = row_lo < row_end ? s_lo / fmaxf(running_sum[q_lo] * s_lo + __expf(sink_lo - gm_lo), 1.0e-20f) : 0.0f;
	const float inv_hi = row_hi < row_end ? s_hi / fmaxf(running_sum[q_hi] * s_hi + __expf(sink_hi - gm_hi), 1.0e-20f) : 0.0f;
	#pragma unroll
	for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; ++frag )
	{
		const uint32_t col = warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u;
		if ( row_lo < row_end )
			__stcs((uint32_t *)(output_bf16 + ((uint64_t)row_lo * HEADS + q_lo % HEADS) * LATENT + col),
				LmPackBf16Pair(output[frag][0] * inv_lo, output[frag][1] * inv_lo));
		if ( row_hi < row_end )
			__stcs((uint32_t *)(output_bf16 + ((uint64_t)row_hi * HEADS + q_hi % HEADS) * LATENT + col),
				LmPackBf16Pair(output[frag][2] * inv_hi, output[frag][3] * inv_hi));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(HY_THREADS, 2)
static void HySplitSelectedKernel(SparkKvShard shard, const uint32_t *__restrict__ selected, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ positions, uint32_t topk, uint32_t *__restrict__ split_positions,
	uint32_t *__restrict__ split_offsets)
{
	constexpr uint32_t WARPS = HY_THREADS / 32u;
	__shared__ uint32_t tally[WARPS];
	const uint32_t row = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, below = (1u << lane) - 1u;
	LmDependentWait();
	const uint32_t pos = positions[row];
	uint32_t bound = context_length[sequence_of_row[row]];
	if ( pos + 1u < bound )
		bound = pos + 1u;
	uint32_t count = 0u;
	if ( bound > topk )
	{
		const uint32_t *src = selected + (uint64_t)row * topk;
		uint32_t *dst = split_positions + (uint64_t)row * topk;
		for ( uint32_t base = 0u; base < topk; base += HY_THREADS )
		{
			const uint32_t idx = base + threadIdx.x;
			const uint32_t val = idx < topk ? src[idx] : LM_INDEX_SHARD_NO_POSITION;
			const uint32_t keep = idx < topk && val <= pos && SparkKvShardOwns(shard, val) != 0u ? 1u : 0u;
			const uint32_t mask = __ballot_sync(0xffffffffu, keep != 0u);
			if ( lane == 0u )
				tally[warp] = (uint32_t)__popc(mask);
			__syncthreads();
			uint32_t before = 0u, total = 0u;
			#pragma unroll
			for ( uint32_t w = 0u; w < WARPS; ++w )
			{
				before += w < warp ? tally[w] : 0u;
				total += tally[w];
			}
			if ( keep != 0u )
				dst[count + before + (uint32_t)__popc(mask & below)] = val;
			count += total;
			if ( base + HY_THREADS < topk )
				__syncthreads();
		}
	}
	if ( threadIdx.x == 0u )
	{
		uint32_t *off = split_offsets + (uint64_t)row * (shard.degree + 1u) + shard.rank;
		off[0] = 0u;
		off[1] = count;
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

template<uint32_t RG>
static __device__ __forceinline__ void HyFp8AccumLoadedRows(uint4 weight, float scale, const LmSkinnyWide *acts, uint32_t cnt, float *acc)
{
	const LmSkinnyFormat<LmFp8>::Decoded dec = LmSkinnyFormat<LmFp8>::Decode(weight);
	#pragma unroll
	for ( uint32_t r = 0u; r < RG; ++r )
	{
		if ( r < cnt )
			acc[r] = fmaf(scale, LmSkinnyFormat<LmFp8>::ChunkDecoded(dec, acts[r]), acc[r]);
	}
}

template<uint32_t RG>
static __device__ __forceinline__ void HyFp8AccumIndexedRows(uint4 weight, float scale, const uint16_t *activation, const uint32_t *src_row,
	uint32_t stride, uint32_t c_off, uint32_t cnt, float *acc)
{
	const LmSkinnyFormat<LmFp8>::Decoded dec = LmSkinnyFormat<LmFp8>::Decode(weight);
	#pragma unroll
	for ( uint32_t r = 0u; r < RG; ++r )
	{
		if ( r < cnt )
		{
			const LmSkinnyWide act = LmSkinnyFormat<LmFp8>::Load(activation + (uint64_t)src_row[r] * stride + c_off);
			acc[r] = fmaf(scale, LmSkinnyFormat<LmFp8>::ChunkDecoded(dec, act), acc[r]);
		}
	}
}

template<uint32_t RG>
static __device__ __forceinline__ void HyFp8AccumContigRows(uint4 weight, float scale, const uint16_t *base, uint32_t stride, uint32_t cnt, float *acc)
{
	const LmSkinnyFormat<LmFp8>::Decoded dec = LmSkinnyFormat<LmFp8>::Decode(weight);
	#pragma unroll
	for ( uint32_t r = 0u; r < RG; ++r )
	{
		if ( r < cnt )
		{
			const LmSkinnyWide act = LmSkinnyFormat<LmFp8>::Load(base + (uint64_t)r * stride);
			acc[r] = fmaf(scale, LmSkinnyFormat<LmFp8>::ChunkDecoded(dec, act), acc[r]);
		}
	}
}

template<class G, uint8_t CODEC>
__global__ __launch_bounds__(64u, 4)
static void HySharedGateUpSiluKernel(const void *__restrict__ weight, const void *__restrict__ scales, const uint16_t *__restrict__ activation,
	uint16_t *__restrict__ output, uint32_t shared_rows, uint32_t rows)
{
	constexpr uint32_t K = G::kHidden;
	constexpr uint32_t ELEMS = CODEC == HY_CODEC_BF16 ? 8u : 16u;
	constexpr uint32_t CHUNKS = K / ELEMS;
	constexpr uint32_t SC_GROUP = CODEC == HY_CODEC_FP8_BLOCK128 ? HY_FP8_BLOCK : HY_SCALE_GROUP;
	constexpr uint32_t SC_SHIFT = CODEC == HY_CODEC_FP8_BLOCK128 ? 3u : 1u;
	constexpr uint32_t SC_STRIDE = K / SC_GROUP;
	constexpr uint32_t RG = 4u;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t col = blockIdx.x * 2u + warp;
	LmDependentWait();
	if ( col < shared_rows )
	{
		const uint4 *wg = (const uint4 *)weight + (uint64_t)col * CHUNKS;
		const uint4 *wu = (const uint4 *)weight + (uint64_t)(col + shared_rows) * CHUNKS;
		const uint64_t scg_base = (uint64_t)col * SC_STRIDE;
		const uint64_t scu_base = (uint64_t)(col + shared_rows) * SC_STRIDE;
		for ( uint32_t r_base = 0u; r_base < rows; r_base += RG )
		{
			const uint32_t cnt = rows - r_base < RG ? rows - r_base : RG;
			float ag[RG], au[RG];
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				ag[r] = au[r] = 0.0f;
			for ( uint32_t c = lane; c < CHUNKS; c += 32u )
			{
				if constexpr ( CODEC == HY_CODEC_BF16 )
				{
					const uint4 vwg = __ldcs(wg + c);
					const uint4 vwu = __ldcs(wu + c);
					const uint16_t *base = activation + (uint64_t)r_base * K + c * 8u;
					#pragma unroll
					for ( uint32_t r = 0u; r < RG; ++r )
					{
						if ( r < cnt )
						{
							float af[8];
							LmSkinnyUnpackBf16x8(__ldg((const uint4 *)(base + (uint64_t)r * K)), af);
							ag[r] = LmSkinnyBf16ChunkUnpacked(vwg, af, ag[r]);
							au[r] = LmSkinnyBf16ChunkUnpacked(vwu, af, au[r]);
						}
					}
				}
				else
				{
					constexpr bool FP8_F32 = CODEC == HY_CODEC_FP8_BLOCK128;
					const uint32_t sc_idx = c >> SC_SHIFT;
					const uint16_t *base = activation + (uint64_t)r_base * K + c * 16u;
					LmSkinnyWide acts[RG];
					#pragma unroll
					for ( uint32_t r = 0u; r < RG; ++r )
						if ( r < cnt )
							acts[r] = LmSkinnyFormat<LmFp8>::Load(base + (uint64_t)r * K);
					HyFp8AccumLoadedRows<RG>(__ldcs(wg + c), HyLoadScale<FP8_F32>(scales, scg_base + sc_idx), acts, cnt, ag);
					HyFp8AccumLoadedRows<RG>(__ldcs(wu + c), HyLoadScale<FP8_F32>(scales, scu_base + sc_idx), acts, cnt, au);
				}
			}
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
			{
				if ( r < cnt )
				{
					#pragma unroll
					for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
					{
						ag[r] += __shfl_down_sync(0xffffffffu, ag[r], offset);
						au[r] += __shfl_down_sync(0xffffffffu, au[r], offset);
					}
					if ( lane == 0u )
					{
						const float g = SparkMathRoundBf16(ag[r]);
						const float u = SparkMathRoundBf16(au[r]);
						output[(uint64_t)(r_base + r) * shared_rows + col] = LmFloatToBf16(SparkMathClampedGlu(SparkMathSiluFast(g), u, INFINITY));
					}
				}
			}
		}
	}
	LmDependentRelease();
}

template<class G, bool FP8_F32>
__global__ __launch_bounds__(128u, 4)
static void HyMoeFastW1W3SwigluKernel(const uint8_t *__restrict__ w1, const void *__restrict__ s1, const uint8_t *__restrict__ w3,
	const void *__restrict__ s3, const uint16_t *__restrict__ activation, const float *__restrict__ packed_weight,
	const uint32_t *__restrict__ route_source_token, const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ output, uint32_t rows)
{
	constexpr uint32_t THREADS = 128u;
	constexpr uint32_t CHUNKS = G::kHidden / 16u;
	constexpr uint32_t SC_GROUP = FP8_F32 ? HY_FP8_BLOCK : HY_SCALE_GROUP;
	constexpr uint32_t SC_SHIFT = FP8_F32 ? 3u : 1u;
	constexpr uint32_t SC_STRIDE = G::kHidden / SC_GROUP;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = THREADS / LANES;
	constexpr uint32_t N = G::kMoeInter;
	constexpr uint32_t COLS = GROUPS * 2u;
	constexpr uint32_t N_BLOCKS = (N + COLS - 1u) / COLS;
	constexpr uint32_t RG = 4u;
	__shared__ uint32_t s_prefix[G::kLocalExperts + 1u];
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		uint32_t sum = 0u;
		s_prefix[0] = 0u;
		for ( uint32_t e = 0u; e < G::kLocalExperts; ++e )
		{
			const uint32_t cnt = __ldg(group_row_offset + e + 1u) - __ldg(group_row_offset + e);
			sum += (cnt + RG - 1u) / RG;
			s_prefix[e + 1u] = sum;
		}
	}
	__syncthreads();
	const uint32_t total_tiles = s_prefix[G::kLocalExperts] * N_BLOCKS;
	const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		const uint32_t chunk_id = tile / N_BLOCKS;
		const uint32_t nb = tile - chunk_id * N_BLOCKS;
		const uint32_t col0 = (nb * GROUPS + group) * 2u;
		if ( col0 < N )
		{
			uint32_t lo = 0u, hi = G::kLocalExperts;
			while ( lo + 1u < hi )
			{
				const uint32_t mid = (lo + hi) >> 1u;
				if ( s_prefix[mid] <= chunk_id )
					lo = mid;
				else
					hi = mid;
			}
			const uint32_t exp = lo;
			const uint32_t exp_first = __ldg(group_row_offset + exp);
			const uint32_t exp_end = __ldg(group_row_offset + exp + 1u);
			const uint32_t r0 = exp_first + (chunk_id - s_prefix[exp]) * RG;
			const uint32_t cnt = exp_end - r0 < RG ? exp_end - r0 : RG;
			uint32_t src_row[RG];
			#pragma unroll
			for ( uint32_t r = 0u; r < RG; ++r )
				src_row[r] = (r < cnt && rows > 1u) ? __ldg(route_source_token + r0 + r) : 0u;
			const bool has1 = col0 + 1u < N;
			const uint64_t row_base = (uint64_t)exp * N + col0;
			const uint4 *w1_0 = (const uint4 *)w1 + row_base * CHUNKS;
			const uint4 *w3_0 = (const uint4 *)w3 + row_base * CHUNKS;
			const uint64_t w_off1 = has1 ? CHUNKS : 0u;
			const uint64_t sc_base0 = row_base * SC_STRIDE;
			const uint64_t sc_base1 = has1 ? sc_base0 + SC_STRIDE : sc_base0;
			if ( cnt == 1u )
			{
				const uint16_t *act0 = activation + (uint64_t)src_row[0] * G::kHidden;
				float lg0 = 0.0f, lu0 = 0.0f, lg1 = 0.0f, lu1 = 0.0f;
				for ( uint32_t c = sub; c < CHUNKS; c += LANES )
				{
					const LmSkinnyWide act = LmSkinnyFormat<LmFp8>::Load(act0 + c * 16u);
					const uint32_t sc_idx = c >> SC_SHIFT;
					lg0 = fmaf(HyLoadScale<FP8_F32>(s1, sc_base0 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w1_0 + c), act), lg0);
					lu0 = fmaf(HyLoadScale<FP8_F32>(s3, sc_base0 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w3_0 + c), act), lu0);
					lg1 = fmaf(HyLoadScale<FP8_F32>(s1, sc_base1 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w1_0 + w_off1 + c), act), lg1);
					lu1 = fmaf(HyLoadScale<FP8_F32>(s3, sc_base1 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w3_0 + w_off1 + c), act), lu1);
				}
				#pragma unroll
				for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
				{
					lg0 += __shfl_down_sync(0xffffffffu, lg0, offset);
					lu0 += __shfl_down_sync(0xffffffffu, lu0, offset);
					lg1 += __shfl_down_sync(0xffffffffu, lg1, offset);
					lu1 += __shfl_down_sync(0xffffffffu, lu1, offset);
				}
				if ( sub == 0u )
				{
					const float wt = __ldg(packed_weight + r0);
					const float v0 = SparkMathGateLimitedGlu(SparkMathRoundBf16(lg0), SparkMathRoundBf16(lu0), G::kSwigluLimit) * wt;
					if ( has1 )
					{
						const float v1 = SparkMathGateLimitedGlu(SparkMathRoundBf16(lg1), SparkMathRoundBf16(lu1), G::kSwigluLimit) * wt;
						*(uint32_t *)(output + (uint64_t)r0 * N + col0) = LmPackBf16Pair(v0, v1);
					}
					else
						output[(uint64_t)r0 * N + col0] = LmFloatToBf16(v0);
				}
			}
			else
			{
				float g0[RG], u0[RG], g1[RG], u1[RG];
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					g0[r] = u0[r] = g1[r] = u1[r] = 0.0f;
				for ( uint32_t c = sub; c < CHUNKS; c += LANES )
				{
					const uint32_t c_off = c * 16u, sc_idx = c >> SC_SHIFT;
					LmSkinnyWide acts[RG];
					#pragma unroll
					for ( uint32_t r = 0u; r < RG; ++r )
						if ( r < cnt )
							acts[r] = LmSkinnyFormat<LmFp8>::Load(activation + (uint64_t)src_row[r] * G::kHidden + c_off);
					HyFp8AccumLoadedRows<RG>(__ldcs(w1_0 + c), HyLoadScale<FP8_F32>(s1, sc_base0 + sc_idx), acts, cnt, g0);
					HyFp8AccumLoadedRows<RG>(__ldcs(w3_0 + c), HyLoadScale<FP8_F32>(s3, sc_base0 + sc_idx), acts, cnt, u0);
					HyFp8AccumLoadedRows<RG>(__ldcs(w1_0 + w_off1 + c), HyLoadScale<FP8_F32>(s1, sc_base1 + sc_idx), acts, cnt, g1);
					HyFp8AccumLoadedRows<RG>(__ldcs(w3_0 + w_off1 + c), HyLoadScale<FP8_F32>(s3, sc_base1 + sc_idx), acts, cnt, u1);
				}
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
				{
					if ( r < cnt )
					{
						#pragma unroll
						for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
						{
							g0[r] += __shfl_down_sync(0xffffffffu, g0[r], offset);
							u0[r] += __shfl_down_sync(0xffffffffu, u0[r], offset);
							g1[r] += __shfl_down_sync(0xffffffffu, g1[r], offset);
							u1[r] += __shfl_down_sync(0xffffffffu, u1[r], offset);
						}
						if ( sub == 0u )
						{
							const float wt = __ldg(packed_weight + r0 + r);
							const float v0 = SparkMathGateLimitedGlu(SparkMathRoundBf16(g0[r]), SparkMathRoundBf16(u0[r]), G::kSwigluLimit) * wt;
							if ( has1 )
							{
								const float v1 = SparkMathGateLimitedGlu(SparkMathRoundBf16(g1[r]), SparkMathRoundBf16(u1[r]), G::kSwigluLimit) * wt;
								*(uint32_t *)(output + (uint64_t)(r0 + r) * N + col0) = LmPackBf16Pair(v0, v1);
							}
							else
								output[(uint64_t)(r0 + r) * N + col0] = LmFloatToBf16(v0);
						}
					}
				}
			}
		}
	}
	LmDependentRelease();
}

template<class G, bool FP8_F32>
__global__ __launch_bounds__(128u, 4)
static void HyMoeFastW2Kernel(const uint8_t *__restrict__ w2, const void *__restrict__ s2, const uint16_t *__restrict__ expert_act,
	const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ output)
{
	constexpr uint32_t THREADS = 128u;
	constexpr uint32_t K = G::kMoeInter;
	constexpr uint32_t CHUNKS = K / 16u;
	constexpr uint32_t SC_GROUP = FP8_F32 ? HY_FP8_BLOCK : HY_SCALE_GROUP;
	constexpr uint32_t SC_SHIFT = FP8_F32 ? 3u : 1u;
	constexpr uint32_t SC_STRIDE = K / SC_GROUP;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = THREADS / LANES;
	constexpr uint32_t N = G::kHidden;
	constexpr uint32_t COLS = GROUPS * 4u;
	constexpr uint32_t N_BLOCKS = (N + COLS - 1u) / COLS;
	constexpr uint32_t RG = 4u;
	__shared__ uint32_t s_prefix[G::kLocalExperts + 1u];
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		uint32_t sum = 0u;
		s_prefix[0] = 0u;
		for ( uint32_t e = 0u; e < G::kLocalExperts; ++e )
		{
			const uint32_t cnt = __ldg(group_row_offset + e + 1u) - __ldg(group_row_offset + e);
			sum += (cnt + RG - 1u) / RG;
			s_prefix[e + 1u] = sum;
		}
	}
	__syncthreads();
	const uint32_t total_tiles = s_prefix[G::kLocalExperts] * N_BLOCKS;
	const uint32_t group = threadIdx.x / LANES, sub = threadIdx.x & (LANES - 1u);
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		const uint32_t chunk_id = tile / N_BLOCKS;
		const uint32_t nb = tile - chunk_id * N_BLOCKS;
		const uint32_t col0 = (nb * GROUPS + group) * 4u;
		if ( col0 < N )
		{
			uint32_t lo = 0u, hi = G::kLocalExperts;
			while ( lo + 1u < hi )
			{
				const uint32_t mid = (lo + hi) >> 1u;
				if ( s_prefix[mid] <= chunk_id )
					lo = mid;
				else
					hi = mid;
			}
			const uint32_t exp = lo;
			const uint32_t exp_first = __ldg(group_row_offset + exp);
			const uint32_t exp_end = __ldg(group_row_offset + exp + 1u);
			const uint32_t r0 = exp_first + (chunk_id - s_prefix[exp]) * RG;
			const uint32_t cnt = exp_end - r0 < RG ? exp_end - r0 : RG;
			const uint16_t *act_base = expert_act + (uint64_t)r0 * K;
			const bool has1 = col0 + 1u < N, has2 = col0 + 2u < N, has3 = col0 + 3u < N;
			const uint64_t row_base = (uint64_t)exp * N + col0;
			const uint4 *w0 = (const uint4 *)w2 + row_base * CHUNKS;
			const uint64_t wo1 = has1 ? CHUNKS : 0u;
			const uint64_t wo2 = has2 ? 2u * CHUNKS : 0u;
			const uint64_t wo3 = has3 ? 3u * CHUNKS : 0u;
			const uint64_t sc_base0 = row_base * SC_STRIDE;
			const uint64_t sc_base1 = has1 ? sc_base0 + SC_STRIDE : sc_base0;
			const uint64_t sc_base2 = has2 ? sc_base0 + 2u * SC_STRIDE : sc_base0;
			const uint64_t sc_base3 = has3 ? sc_base0 + 3u * SC_STRIDE : sc_base0;
			if ( cnt == 1u )
			{
				float la0 = 0.0f, la1 = 0.0f, la2 = 0.0f, la3 = 0.0f;
				for ( uint32_t c = sub; c < CHUNKS; c += LANES )
				{
					const LmSkinnyWide act = LmSkinnyFormat<LmFp8>::Load(act_base + c * 16u);
					const uint32_t sc_idx = c >> SC_SHIFT;
					la0 = fmaf(HyLoadScale<FP8_F32>(s2, sc_base0 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w0 + c), act), la0);
					la1 = fmaf(HyLoadScale<FP8_F32>(s2, sc_base1 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w0 + wo1 + c), act), la1);
					la2 = fmaf(HyLoadScale<FP8_F32>(s2, sc_base2 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w0 + wo2 + c), act), la2);
					la3 = fmaf(HyLoadScale<FP8_F32>(s2, sc_base3 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w0 + wo3 + c), act), la3);
				}
				#pragma unroll
				for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
				{
					la0 += __shfl_down_sync(0xffffffffu, la0, offset);
					la1 += __shfl_down_sync(0xffffffffu, la1, offset);
					la2 += __shfl_down_sync(0xffffffffu, la2, offset);
					la3 += __shfl_down_sync(0xffffffffu, la3, offset);
				}
				if ( sub == 0u )
				{
					uint16_t *dst = output + (uint64_t)r0 * N + col0;
					if ( has3 )
						*(uint2 *)dst = make_uint2(LmPackBf16Pair(la0, la1), LmPackBf16Pair(la2, la3));
					else
					{
						dst[0] = LmFloatToBf16(la0);
						if ( has1 ) dst[1] = LmFloatToBf16(la1);
						if ( has2 ) dst[2] = LmFloatToBf16(la2);
					}
				}
			}
			else
			{
				float a0[RG], a1[RG], a2[RG], a3[RG];
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
					a0[r] = a1[r] = a2[r] = a3[r] = 0.0f;
				for ( uint32_t c = sub; c < CHUNKS; c += LANES )
				{
					const uint32_t sc_idx = c >> SC_SHIFT;
					const uint16_t *cb = act_base + c * 16u;
					LmSkinnyWide acts[RG];
					#pragma unroll
					for ( uint32_t r = 0u; r < RG; ++r )
						if ( r < cnt )
							acts[r] = LmSkinnyFormat<LmFp8>::Load(cb + (uint64_t)r * K);
					HyFp8AccumLoadedRows<RG>(__ldcs(w0 + c), HyLoadScale<FP8_F32>(s2, sc_base0 + sc_idx), acts, cnt, a0);
					HyFp8AccumLoadedRows<RG>(__ldcs(w0 + wo1 + c), HyLoadScale<FP8_F32>(s2, sc_base1 + sc_idx), acts, cnt, a1);
					HyFp8AccumLoadedRows<RG>(__ldcs(w0 + wo2 + c), HyLoadScale<FP8_F32>(s2, sc_base2 + sc_idx), acts, cnt, a2);
					HyFp8AccumLoadedRows<RG>(__ldcs(w0 + wo3 + c), HyLoadScale<FP8_F32>(s2, sc_base3 + sc_idx), acts, cnt, a3);
				}
				#pragma unroll
				for ( uint32_t r = 0u; r < RG; ++r )
				{
					if ( r < cnt )
					{
						#pragma unroll
						for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
						{
							a0[r] += __shfl_down_sync(0xffffffffu, a0[r], offset);
							a1[r] += __shfl_down_sync(0xffffffffu, a1[r], offset);
							a2[r] += __shfl_down_sync(0xffffffffu, a2[r], offset);
							a3[r] += __shfl_down_sync(0xffffffffu, a3[r], offset);
						}
						if ( sub == 0u )
						{
							uint16_t *dst = output + (uint64_t)(r0 + r) * N + col0;
							if ( has3 )
								*(uint2 *)dst = make_uint2(LmPackBf16Pair(a0[r], a1[r]), LmPackBf16Pair(a2[r], a3[r]));
							else
							{
								dst[0] = LmFloatToBf16(a0[r]);
								if ( has1 ) dst[1] = LmFloatToBf16(a1[r]);
								if ( has2 ) dst[2] = LmFloatToBf16(a2[r]);
							}
						}
					}
				}
			}
		}
	}
	LmDependentRelease();
}

template<class G, bool FP8_F32>
__global__ __launch_bounds__(128u, 4)
static void HyMoeSmallW2FinalizeKernel(const uint8_t *__restrict__ w2, const void *__restrict__ s2, const uint16_t *__restrict__ expert_act,
	const uint32_t *__restrict__ route_packed_row, const uint32_t *__restrict__ route_expert, const uint16_t *__restrict__ shared,
	uint16_t *__restrict__ output)
{
	constexpr uint32_t THREADS = 128u;
	constexpr uint32_t K = G::kMoeInter;
	constexpr uint32_t CHUNKS = K / 16u;
	constexpr uint32_t SC_GROUP = FP8_F32 ? HY_FP8_BLOCK : HY_SCALE_GROUP;
	constexpr uint32_t SC_SHIFT = FP8_F32 ? 3u : 1u;
	constexpr uint32_t SC_STRIDE = K / SC_GROUP;
	constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
	constexpr uint32_t GROUPS = THREADS / LANES;
	constexpr uint32_t N = G::kHidden;
	__shared__ uint32_t s_exp[G::kTopK], s_pr[G::kTopK], s_count;
	const uint32_t row = blockIdx.y;
	const uint16_t *sh_row = shared + (uint64_t)row * N;
	uint16_t *out_row = output + (uint64_t)row * N;
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		const uint32_t rbase = row * G::kTopK;
		uint32_t cnt = 0u;
		for ( uint32_t k = 0u; k < G::kTopK; ++k )
		{
			const uint32_t e = route_expert[rbase + k];
			if ( e < G::kLocalExperts )
			{
				s_exp[cnt] = e;
				s_pr[cnt] = route_packed_row[rbase + k];
				cnt++;
			}
		}
		s_count = cnt;
	}
	__syncthreads();
	const uint32_t col0 = (blockIdx.x * GROUPS + (threadIdx.x / LANES)) * 4u;
	const uint32_t sub = threadIdx.x & (LANES - 1u);
	if ( col0 < N )
	{
		const uint32_t count = s_count;
		const bool has1 = col0 + 1u < N, has2 = col0 + 2u < N, has3 = col0 + 3u < N;
		if ( count == 0u )
		{
			if ( sub == 0u )
			{
				if ( has3 )
					*(uint2 *)(out_row + col0) = *(const uint2 *)(sh_row + col0);
				else
				{
					out_row[col0] = sh_row[col0];
					if ( has1 ) out_row[col0 + 1u] = sh_row[col0 + 1u];
					if ( has2 ) out_row[col0 + 2u] = sh_row[col0 + 2u];
				}
			}
		}
		else
		{
			float total0 = 0.0f, total1 = 0.0f, total2 = 0.0f, total3 = 0.0f;
			for ( uint32_t k = 0u; k < count; ++k )
			{
				const uint16_t *a_row = expert_act + (uint64_t)s_pr[k] * K;
				const uint64_t row_base = (uint64_t)s_exp[k] * N + col0;
				const uint4 *w0 = (const uint4 *)w2 + row_base * CHUNKS;
				const uint4 *w1 = has1 ? w0 + CHUNKS : w0;
				const uint4 *w2p = has2 ? w0 + 2u * CHUNKS : w0;
				const uint4 *w3p = has3 ? w0 + 3u * CHUNKS : w0;
				const uint64_t sc_base0 = row_base * SC_STRIDE;
				const uint64_t sc_base1 = has1 ? sc_base0 + SC_STRIDE : sc_base0;
				const uint64_t sc_base2 = has2 ? sc_base0 + 2u * SC_STRIDE : sc_base0;
				const uint64_t sc_base3 = has3 ? sc_base0 + 3u * SC_STRIDE : sc_base0;
				float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
				for ( uint32_t c = sub; c < CHUNKS; c += LANES )
				{
					const LmSkinnyWide act = LmSkinnyFormat<LmFp8>::Load(a_row + c * 16u);
					const uint32_t sc_idx = c >> SC_SHIFT;
					acc0 = fmaf(HyLoadScale<FP8_F32>(s2, sc_base0 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w0 + c), act), acc0);
					acc1 = fmaf(HyLoadScale<FP8_F32>(s2, sc_base1 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w1 + c), act), acc1);
					acc2 = fmaf(HyLoadScale<FP8_F32>(s2, sc_base2 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w2p + c), act), acc2);
					acc3 = fmaf(HyLoadScale<FP8_F32>(s2, sc_base3 + sc_idx), LmSkinnyFormat<LmFp8>::Chunk(__ldcs(w3p + c), act), acc3);
				}
				#pragma unroll
				for ( uint32_t offset = LANES >> 1u; offset > 0u; offset >>= 1u )
				{
					acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
					acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
					acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
					acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
				}
				if ( sub == 0u )
				{
					total0 += SparkMathRoundBf16(acc0);
					total1 += SparkMathRoundBf16(acc1);
					total2 += SparkMathRoundBf16(acc2);
					total3 += SparkMathRoundBf16(acc3);
				}
			}
			if ( sub == 0u )
			{
				if ( has3 )
				{
					const uint2 sh = *(const uint2 *)(sh_row + col0);
					total0 += __uint_as_float(sh.x << 16u);
					total1 += __uint_as_float(sh.x & 0xffff0000u);
					total2 += __uint_as_float(sh.y << 16u);
					total3 += __uint_as_float(sh.y & 0xffff0000u);
					*(uint2 *)(out_row + col0) = make_uint2(LmPackBf16Pair(total0, total1), LmPackBf16Pair(total2, total3));
				}
				else
				{
					out_row[col0] = LmFloatToBf16(total0 + LmBf16ToFloat(sh_row[col0]));
					if ( has1 ) out_row[col0 + 1u] = LmFloatToBf16(total1 + LmBf16ToFloat(sh_row[col0 + 1u]));
					if ( has2 ) out_row[col0 + 2u] = LmFloatToBf16(total2 + LmBf16ToFloat(sh_row[col0 + 2u]));
				}
			}
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(HY_THREADS, 4)
static void HyMoeFinalizeKernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row,
	const uint32_t *__restrict__ route_expert, const uint16_t *__restrict__ shared, uint16_t *__restrict__ output, uint32_t top_k,
	uint32_t local_experts, uint32_t width)
{
	LmMoeLocalFinalizeBody<false>(packed, route_packed_row, route_expert, (const float *)0, shared, output, top_k, local_experts, width);
}

static __device__ __forceinline__ uint32_t HyE4m3PairScaledFast(uint32_t word, uint32_t selector, uint32_t sc2)
{
	const uint32_t placed = __byte_perm(word, 0u, selector);
	const uint32_t bits = ((placed >> 4u) & 0x07f007f0u) | (placed & 0x80008000u);
	uint32_t product;
	asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(product) : "r"(bits), "r"(sc2));
	return product;
}

static __device__ __forceinline__ void HyMxfp8DecodePairScaled(uint32_t r0, uint32_t r1, uint32_t lane, uint8_t e8m0, uint32_t b[4])
{
	const uint32_t quad = lane & 3u, src_lo = (lane & ~3u) | (quad >> 1u), src_hi = src_lo | 2u;
	const uint32_t sel = (quad & 1u) != 0u ? 0x3424u : 0x1404u;
	const uint32_t w0 = __shfl_sync(0xffffffffu, r0, src_lo);
	const uint32_t w1 = __shfl_sync(0xffffffffu, r0, src_hi);
	const uint32_t w2 = __shfl_sync(0xffffffffu, r1, src_lo);
	const uint32_t w3 = __shfl_sync(0xffffffffu, r1, src_hi);
	const uint32_t first_sc = e8m0 <= 134u ? ((((uint32_t)e8m0 + 120u) << 7u) | (((uint32_t)e8m0 + 120u) << 23u)) : 0x7b807b80u;
	b[0] = HyE4m3PairScaledFast(w0, sel, first_sc);
	b[1] = HyE4m3PairScaledFast(w1, sel, first_sc);
	b[2] = HyE4m3PairScaledFast(w2, sel, first_sc);
	b[3] = HyE4m3PairScaledFast(w3, sel, first_sc);
	if ( __builtin_expect(e8m0 > 134u, 0) )
	{
		const uint32_t sc2 = e8m0 < 255u ? (((uint32_t)e8m0 << 7u) | ((uint32_t)e8m0 << 23u)) : 0x7fc07fc0u;
		asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(b[0]) : "r"(b[0]), "r"(sc2));
		asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(b[1]) : "r"(b[1]), "r"(sc2));
		asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(b[2]) : "r"(b[2]), "r"(sc2));
		asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(b[3]) : "r"(b[3]), "r"(sc2));
	}
}

template<uint32_t B_ROW_BYTES>
static __device__ __forceinline__ void HyMxfp8FragmentScaled(uint32_t stage_s, uint32_t base_row, uint32_t step, uint32_t lane, uint8_t e8m0, uint32_t b[4])
{
	const uint32_t row = base_row + (lane & 7u), col = step * 2u + ((lane >> 3u) & 1u);
	uint32_t raw[2];
	LmStreamLdMatrix2(raw, stage_s + LmStreamWeight<LmMxfp8>::template Offset<B_ROW_BYTES>(row, col));
	HyMxfp8DecodePairScaled(raw[0], raw[1], lane, e8m0, b);
}

template<uint32_t B_ROW_BYTES>
static __device__ __forceinline__ void HyMxfp8FragmentPairScaled(uint32_t stage_s, uint32_t base_row, uint32_t step, uint32_t lane,
	uint8_t e8m0_0, uint8_t e8m0_1, uint32_t b0[4], uint32_t b1[4])
{
	const uint32_t row = base_row + (lane & 7u) + (((lane >> 4u) & 1u) * 8u), col = step * 2u + ((lane >> 3u) & 1u);
	uint32_t raw[4];
	LmStreamLdMatrix4(raw, stage_s + LmStreamWeight<LmMxfp8>::template Offset<B_ROW_BYTES>(row, col));
	HyMxfp8DecodePairScaled(raw[0], raw[1], lane, e8m0_0, b0);
	HyMxfp8DecodePairScaled(raw[2], raw[3], lane, e8m0_1, b1);
}

template<uint32_t STAGES, uint32_t STAGE_BYTES>
static __device__ __forceinline__ void HyAdvanceStage(uint32_t &offset)
{
	if constexpr ( STAGES == 2u )
		offset ^= STAGE_BYTES;
	else
	{
		offset += STAGE_BYTES;
		if ( offset == STAGES * STAGE_BYTES )
			offset = 0u;
	}
}

static __device__ __forceinline__ void HyCopyL2_256(uint32_t dst, const void *src, uint32_t bytes)
{
	asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}

template<class Format, uint32_t MF, uint32_t NF, uint32_t A_ROW_BYTES, uint32_t B_ROW_BYTES, uint32_t WARP_N>
static __device__ __forceinline__ void HyScaledTileMma(uint32_t stage_a, uint32_t stage_b, uint32_t ldm_a_row, uint32_t ldm_a_col,
	uint32_t warp_n, uint32_t lane, uint32_t group, uint32_t neuron_base, uint32_t output_dim, uint32_t kt, const LmScaleTensor &scale,
	float total[MF][NF][4])
{
	using Weight = LmStreamWeight<Format>;
	constexpr bool PACK4_UE8M0 = LmStreamPack4Ue8m0<Format>();
	const uint32_t quad_row = lane >> 2u, quad = lane & 3u;
	if constexpr ( PACK4_UE8M0 )
	{
		uint32_t sp[NF];
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
		{
			const uint32_t col_b = neuron_base + warp_n * WARP_N + j * 8u + quad_row;
			sp[j] = col_b < output_dim ? LmStreamScalePack4(scale, group, col_b, kt * 4u) : 0u;
		}
		#pragma unroll
		for ( uint32_t step = 0u; step < 4u; ++step )
		{
			uint32_t b[NF][4];
			if constexpr ( (NF % 2u) == 0u )
			{
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; j += 2u )
				{
					HyMxfp8FragmentPairScaled<B_ROW_BYTES>(stage_b, warp_n * WARP_N + j * 8u, step, lane,
						(uint8_t)(sp[j] & 0xffu), (uint8_t)(sp[j + 1u] & 0xffu), b[j], b[j + 1u]);
					sp[j] >>= 8u;
					sp[j + 1u] >>= 8u;
				}
			}
			else
			{
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					HyMxfp8FragmentScaled<B_ROW_BYTES>(stage_b, warp_n * WARP_N + j * 8u, step, lane, (uint8_t)(sp[j] & 0xffu), b[j]);
					sp[j] >>= 8u;
				}
			}
			uint32_t a_cur[4], a_nxt[4];
			LmStreamLdMatrix4(a_cur, stage_a + ldm_a_row * A_ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + ldm_a_col) * 16u);
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
			{
				const uint32_t row = ldm_a_row + i * 16u;
				LmStreamLdMatrix4(a_nxt, stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + 2u + ldm_a_col) * 16u);
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					uint32_t b_lo[2] = { b[j][0], b[j][1] };
					LmMmaBf16(total[i][j], a_cur, b_lo);
				}
				if constexpr ( MF > 1u )
				{
					if ( i + 1u < MF )
					{
						const uint32_t next_row = ldm_a_row + (i + 1u) * 16u;
						LmStreamLdMatrix4(a_cur, stage_a + next_row * A_ROW_BYTES + LmStreamSwizzle8(next_row, step * 4u + ldm_a_col) * 16u);
					}
				}
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					uint32_t b_hi[2] = { b[j][2], b[j][3] };
					LmMmaBf16(total[i][j], a_nxt, b_hi);
				}
			}
		}
	}
	else
	{
		float partial[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				partial[i][j][0] = partial[i][j][1] = partial[i][j][2] = partial[i][j][3] = 0.0f;
		#pragma unroll
		for ( uint32_t step = 0u; step < 4u; ++step )
		{
			uint32_t b[NF][4];
			if constexpr ( (NF % 2u) == 0u )
			{
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; j += 2u )
					Weight::template FragmentPair<B_ROW_BYTES>(stage_b, warp_n * WARP_N + j * 8u, step, lane, b[j], b[j + 1u]);
			}
			else
			{
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
					Weight::template Fragment<B_ROW_BYTES>(stage_b, warp_n * WARP_N + j * 8u, step, lane, b[j]);
			}
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
			{
				const uint32_t row = ldm_a_row + i * 16u;
				uint32_t a_frag[4];
				LmStreamLdMatrix4(a_frag, stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + ldm_a_col) * 16u);
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					uint32_t b_lo[2] = { b[j][0], b[j][1] };
					LmMmaBf16(partial[i][j], a_frag, b_lo);
				}
				LmStreamLdMatrix4(a_frag, stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + 2u + ldm_a_col) * 16u);
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					uint32_t b_hi[2] = { b[j][2], b[j][3] };
					LmMmaBf16(partial[i][j], a_frag, b_hi);
				}
			}
		}
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
		{
			const uint32_t col = neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
			const float sc0 = col < output_dim ? LmStreamScaleLoad<Format>(scale, group, col, kt) : 0.0f;
			const float sc1 = col + 1u < output_dim ? LmStreamScaleLoad<Format>(scale, group, col + 1u, kt) : 0.0f;
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
			{
				total[i][j][0] = fmaf(partial[i][j][0], sc0, total[i][j][0]);
				total[i][j][1] = fmaf(partial[i][j][1], sc1, total[i][j][1]);
				total[i][j][2] = fmaf(partial[i][j][2], sc0, total[i][j][2]);
				total[i][j][3] = fmaf(partial[i][j][3], sc1, total[i][j][3]);
			}
		}
	}
}

template<class Format, uint32_t MF, uint32_t NF, uint32_t A_ROW_BYTES, uint32_t B_ROW_BYTES, uint32_t WARP_N>
static __device__ __forceinline__ void HyScaledTileMmaDual(uint32_t stage_a, uint32_t stage_b1, uint32_t stage_b3, uint32_t ldm_a_row,
	uint32_t ldm_a_col, uint32_t warp_n, uint32_t lane, uint32_t group, uint32_t neuron_base, uint32_t output_dim, uint32_t kt,
	const LmScaleTensor &s1, const LmScaleTensor &s3, float g_tot[MF][NF][4], float u_tot[MF][NF][4])
{
	using Weight = LmStreamWeight<Format>;
	constexpr bool PACK4_UE8M0 = LmStreamPack4Ue8m0<Format>();
	const uint32_t quad_row = lane >> 2u, quad = lane & 3u;
	if constexpr ( PACK4_UE8M0 )
	{
		uint32_t sp1[NF], sp3[NF];
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
		{
			const uint32_t col_b = neuron_base + warp_n * WARP_N + j * 8u + quad_row;
			sp1[j] = col_b < output_dim ? LmStreamScalePack4(s1, group, col_b, kt * 4u) : 0u;
			sp3[j] = col_b < output_dim ? LmStreamScalePack4(s3, group, col_b, kt * 4u) : 0u;
		}
		#pragma unroll
		for ( uint32_t step = 0u; step < 4u; ++step )
		{
			uint32_t b1[NF][4], b3[NF][4];
			if constexpr ( (NF % 2u) == 0u )
			{
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; j += 2u )
				{
					HyMxfp8FragmentPairScaled<B_ROW_BYTES>(stage_b1, warp_n * WARP_N + j * 8u, step, lane,
						(uint8_t)(sp1[j] & 0xffu), (uint8_t)(sp1[j + 1u] & 0xffu), b1[j], b1[j + 1u]);
					HyMxfp8FragmentPairScaled<B_ROW_BYTES>(stage_b3, warp_n * WARP_N + j * 8u, step, lane,
						(uint8_t)(sp3[j] & 0xffu), (uint8_t)(sp3[j + 1u] & 0xffu), b3[j], b3[j + 1u]);
					sp1[j] >>= 8u;
					sp1[j + 1u] >>= 8u;
					sp3[j] >>= 8u;
					sp3[j + 1u] >>= 8u;
				}
			}
			else
			{
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					HyMxfp8FragmentScaled<B_ROW_BYTES>(stage_b1, warp_n * WARP_N + j * 8u, step, lane, (uint8_t)(sp1[j] & 0xffu), b1[j]);
					HyMxfp8FragmentScaled<B_ROW_BYTES>(stage_b3, warp_n * WARP_N + j * 8u, step, lane, (uint8_t)(sp3[j] & 0xffu), b3[j]);
					sp1[j] >>= 8u;
					sp3[j] >>= 8u;
				}
			}
			uint32_t a_cur[4], a_nxt[4];
			LmStreamLdMatrix4(a_cur, stage_a + ldm_a_row * A_ROW_BYTES + LmStreamSwizzle8(ldm_a_row, step * 4u + ldm_a_col) * 16u);
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
			{
				const uint32_t row = ldm_a_row + i * 16u;
				LmStreamLdMatrix4(a_nxt, stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + 2u + ldm_a_col) * 16u);
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					uint32_t b1_lo[2] = { b1[j][0], b1[j][1] };
					uint32_t b3_lo[2] = { b3[j][0], b3[j][1] };
					LmMmaBf16(g_tot[i][j], a_cur, b1_lo);
					LmMmaBf16(u_tot[i][j], a_cur, b3_lo);
				}
				if constexpr ( MF > 1u )
				{
					if ( i + 1u < MF )
					{
						const uint32_t next_row = ldm_a_row + (i + 1u) * 16u;
						LmStreamLdMatrix4(a_cur, stage_a + next_row * A_ROW_BYTES + LmStreamSwizzle8(next_row, step * 4u + ldm_a_col) * 16u);
					}
				}
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					uint32_t b1_hi[2] = { b1[j][2], b1[j][3] };
					uint32_t b3_hi[2] = { b3[j][2], b3[j][3] };
					LmMmaBf16(g_tot[i][j], a_nxt, b1_hi);
					LmMmaBf16(u_tot[i][j], a_nxt, b3_hi);
				}
			}
		}
	}
	else
	{
		float gp[MF][NF][4], up[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				gp[i][j][0] = gp[i][j][1] = gp[i][j][2] = gp[i][j][3] =
					up[i][j][0] = up[i][j][1] = up[i][j][2] = up[i][j][3] = 0.0f;
		#pragma unroll
		for ( uint32_t step = 0u; step < 4u; ++step )
		{
			uint32_t b1[NF][4], b3[NF][4];
			if constexpr ( (NF % 2u) == 0u )
			{
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; j += 2u )
				{
					Weight::template FragmentPair<B_ROW_BYTES>(stage_b1, warp_n * WARP_N + j * 8u, step, lane, b1[j], b1[j + 1u]);
					Weight::template FragmentPair<B_ROW_BYTES>(stage_b3, warp_n * WARP_N + j * 8u, step, lane, b3[j], b3[j + 1u]);
				}
			}
			else
			{
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					Weight::template Fragment<B_ROW_BYTES>(stage_b1, warp_n * WARP_N + j * 8u, step, lane, b1[j]);
					Weight::template Fragment<B_ROW_BYTES>(stage_b3, warp_n * WARP_N + j * 8u, step, lane, b3[j]);
				}
			}
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
			{
				const uint32_t row = ldm_a_row + i * 16u;
				uint32_t a_frag[4];
				LmStreamLdMatrix4(a_frag, stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + ldm_a_col) * 16u);
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					uint32_t b1_lo[2] = { b1[j][0], b1[j][1] };
					uint32_t b3_lo[2] = { b3[j][0], b3[j][1] };
					LmMmaBf16(gp[i][j], a_frag, b1_lo);
					LmMmaBf16(up[i][j], a_frag, b3_lo);
				}
				LmStreamLdMatrix4(a_frag, stage_a + row * A_ROW_BYTES + LmStreamSwizzle8(row, step * 4u + 2u + ldm_a_col) * 16u);
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					uint32_t b1_hi[2] = { b1[j][2], b1[j][3] };
					uint32_t b3_hi[2] = { b3[j][2], b3[j][3] };
					LmMmaBf16(gp[i][j], a_frag, b1_hi);
					LmMmaBf16(up[i][j], a_frag, b3_hi);
				}
			}
		}
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
		{
			const uint32_t col = neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
			const float sc10 = col < output_dim ? LmStreamScaleLoad<Format>(s1, group, col, kt) : 0.0f;
			const float sc11 = col + 1u < output_dim ? LmStreamScaleLoad<Format>(s1, group, col + 1u, kt) : 0.0f;
			const float sc30 = col < output_dim ? LmStreamScaleLoad<Format>(s3, group, col, kt) : 0.0f;
			const float sc31 = col + 1u < output_dim ? LmStreamScaleLoad<Format>(s3, group, col + 1u, kt) : 0.0f;
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
			{
				g_tot[i][j][0] = fmaf(gp[i][j][0], sc10, g_tot[i][j][0]);
				g_tot[i][j][1] = fmaf(gp[i][j][1], sc11, g_tot[i][j][1]);
				g_tot[i][j][2] = fmaf(gp[i][j][2], sc10, g_tot[i][j][2]);
				g_tot[i][j][3] = fmaf(gp[i][j][3], sc11, g_tot[i][j][3]);
				u_tot[i][j][0] = fmaf(up[i][j][0], sc30, u_tot[i][j][0]);
				u_tot[i][j][1] = fmaf(up[i][j][1], sc31, u_tot[i][j][1]);
				u_tot[i][j][2] = fmaf(up[i][j][2], sc30, u_tot[i][j][2]);
				u_tot[i][j][3] = fmaf(up[i][j][3], sc31, u_tot[i][j][3]);
			}
		}
	}
}

template<class G, class Format, uint32_t TILE_M, uint32_t WARPS_M, uint32_t WARPS_N, uint32_t STAGES>
__global__ __launch_bounds__(WARPS_M * WARPS_N * 32u, TILE_M <= 32u ? 2 : 1)
static void HyMoeMmaW1W3SwigluKernel(const uint8_t *__restrict__ w1, LmScaleTensor s1, const uint8_t *__restrict__ w3, LmScaleTensor s3,
	const uint16_t *__restrict__ activation, const float *__restrict__ packed_weight, const uint32_t *__restrict__ route_source_token,
	const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ output)
{
	using Weight = LmStreamWeight<Format>;
	constexpr uint32_t TILE_N = 64u, TILE_K = 128u;
	constexpr uint32_t THREADS = WARPS_M * WARPS_N * 32u;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t K = G::kHidden, N = G::kMoeInter;
	constexpr uint32_t K_TILES = K / TILE_K, N_TILES = N / TILE_N;
	constexpr uint32_t A_ROW_BYTES = TILE_K * 2u, A_ROW_CHUNKS = A_ROW_BYTES / 16u;
	constexpr uint32_t B_ROW_BYTES = TILE_K, B_ROW_CHUNKS = B_ROW_BYTES / 16u;
	constexpr uint32_t A_BYTES = TILE_M * A_ROW_BYTES, B_BYTES = TILE_N * B_ROW_BYTES;
	constexpr uint32_t STAGE_BYTES = A_BYTES + 2u * B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * A_ROW_CHUNKS, A_PER_THREAD = A_CHUNKS / THREADS;
	constexpr uint32_t B_CHUNKS = TILE_N * B_ROW_CHUNKS, B_PER_THREAD = B_CHUNKS / THREADS;
	static_assert(Weight::kScaled && Weight::kElementBytes == 1u && THREADS == 256u && K % TILE_K == 0u && N % TILE_N == 0u &&
		A_CHUNKS % THREADS == 0u && B_CHUNKS % THREADS == 0u && STAGES >= 2u,
		"fused FP8/MXFP8 MoE W1W3 MMA kernel matches expert geometry");
	extern __shared__ __align__(128) uint8_t hy_moe_smem[];
	uint32_t *s_prefix = (uint32_t *)(hy_moe_smem + STAGES * STAGE_BYTES);
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(hy_moe_smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t b_row0 = threadIdx.x / B_ROW_CHUNKS, b_col0 = threadIdx.x % B_ROW_CHUNKS;
	const uint32_t b_row_step = THREADS / B_ROW_CHUNKS;
	const uint64_t b_ptr_step = (uint64_t)b_row_step * K;
	const uint64_t exp_bytes = (uint64_t)N * K;
	const uint32_t ldm_a_row = warp_m * WARP_M + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		uint32_t sum = 0u;
		s_prefix[0] = 0u;
		for ( uint32_t e = 0u; e < G::kLocalExperts; ++e )
		{
			const uint32_t cnt = __ldg(group_row_offset + e + 1u) - __ldg(group_row_offset + e);
			sum += ((cnt + TILE_M - 1u) / TILE_M) * N_TILES;
			s_prefix[e + 1u] = sum;
		}
	}
	__syncthreads();
	const uint32_t total_tiles = s_prefix[G::kLocalExperts];
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t lo = 0u, hi = G::kLocalExperts;
		while ( lo + 1u < hi )
		{
			const uint32_t mid = (lo + hi) >> 1u;
			if ( s_prefix[mid] <= tile )
				lo = mid;
			else
				hi = mid;
		}
		const uint32_t exp = lo;
		const uint32_t exp_first = __ldg(group_row_offset + exp);
		const uint32_t exp_limit = __ldg(group_row_offset + exp + 1u);
		const uint32_t m_tiles = (exp_limit - exp_first + TILE_M - 1u) / TILE_M;
		const uint32_t within = tile - s_prefix[exp];
		const uint32_t n_tile = within / m_tiles;
		const uint32_t m_tile = within - n_tile * m_tiles;
		const uint32_t row_base = exp_first + m_tile * TILE_M;
		const uint32_t neuron_base = n_tile * TILE_N;
		const uint16_t *a_src[A_PER_THREAD];
		uint32_t a_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_THREAD; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * THREADS;
			const uint32_t r = chunk / A_ROW_CHUNKS, c = chunk % A_ROW_CHUNKS;
			const uint32_t packed = row_base + r;
			const uint32_t valid = packed < exp_limit ? 1u : 0u;
			const uint32_t src_row = valid != 0u ? __ldg(route_source_token + packed) : 0u;
			a_src[q] = activation + (uint64_t)src_row * K + c * 8u;
			a_mask |= valid << q;
		}
		const uint64_t b_off = (uint64_t)exp * exp_bytes + (uint64_t)(neuron_base + b_row0) * K + b_col0 * 16u;
		const uint8_t *b1_base = w1 + b_off;
		const uint8_t *b3_base = w3 + b_off;
		auto issue_k = [&](uint32_t st_off)
		{
			const uint32_t stage = shared_base + st_off;
			#pragma unroll
			for ( uint32_t q = 0u; q < A_PER_THREAD; ++q )
			{
				const uint32_t chunk = threadIdx.x + q * THREADS;
				const uint32_t r = chunk / A_ROW_CHUNKS, c = chunk % A_ROW_CHUNKS;
				HyCopyL2_256(stage + r * A_ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], ((a_mask >> q) & 1u) << 4u);
				a_src[q] += TILE_K;
			}
			#pragma unroll
			for ( uint32_t q = 0u; q < B_PER_THREAD; ++q )
			{
				const uint32_t r = b_row0 + q * b_row_step;
				const uint32_t off = Weight::template Offset<B_ROW_BYTES>(r, b_col0);
				HyCopyL2_256(stage + A_BYTES + off, b1_base + q * b_ptr_step, 16u);
				HyCopyL2_256(stage + A_BYTES + B_BYTES + off, b3_base + q * b_ptr_step, 16u);
			}
			b1_base += TILE_K;
			b3_base += TILE_K;
		};
		float g_tot[MF][NF][4], u_tot[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				g_tot[i][j][0] = g_tot[i][j][1] = g_tot[i][j][2] = g_tot[i][j][3] =
					u_tot[i][j][0] = u_tot[i][j][1] = u_tot[i][j][2] = u_tot[i][j][3] = 0.0f;
		uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( kp < K_TILES )
				issue_k(sp_off);
			LmStreamCommit();
			kp++;
			HyAdvanceStage<STAGES, STAGE_BYTES>(sp_off);
		}
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			LmStreamWait<STAGES - 2u>();
			__syncthreads();
			if ( kp < K_TILES )
			{
				issue_k(sp_off);
				kp++;
				HyAdvanceStage<STAGES, STAGE_BYTES>(sp_off);
			}
			LmStreamCommit();
			const uint32_t stage_a = shared_base + sc_off;
			HyScaledTileMmaDual<Format, MF, NF, A_ROW_BYTES, B_ROW_BYTES, WARP_N>(stage_a, stage_a + A_BYTES, stage_a + A_BYTES + B_BYTES,
				ldm_a_row, ldm_a_col, warp_n, lane, exp, neuron_base, N, kt, s1, s3, g_tot, u_tot);
			HyAdvanceStage<STAGES, STAGE_BYTES>(sc_off);
		}
		LmStreamWait<0u>();
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; e += 2u )
				{
					const uint32_t r = row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					if ( r < exp_limit )
					{
						const uint32_t col = neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
						const float wt = __ldg(packed_weight + r);
						const float v0 = SparkMathGateLimitedGlu(SparkMathRoundBf16(g_tot[i][j][e]), SparkMathRoundBf16(u_tot[i][j][e]), G::kSwigluLimit) * wt;
						const float v1 = SparkMathGateLimitedGlu(SparkMathRoundBf16(g_tot[i][j][e + 1u]), SparkMathRoundBf16(u_tot[i][j][e + 1u]), G::kSwigluLimit) * wt;
						*(uint32_t *)(output + (uint64_t)r * N + col) = LmPackBf16Pair(v0, v1);
					}
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
	LmDependentRelease();
}

template<class G, class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS_M, uint32_t WARPS_N, uint32_t STAGES>
__global__ __launch_bounds__(WARPS_M * WARPS_N * 32u, TILE_M <= 32u ? 2 : 1)
static void HyMoeMmaW2Kernel(const uint8_t *__restrict__ w2, LmScaleTensor s2, const uint16_t *__restrict__ activation,
	const uint32_t *__restrict__ group_row_offset, uint16_t *__restrict__ output)
{
	using Weight = LmStreamWeight<Format>;
	constexpr uint32_t TILE_K = 128u;
	constexpr uint32_t THREADS = WARPS_M * WARPS_N * 32u;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t K = G::kMoeInter, N = G::kHidden;
	constexpr uint32_t K_TILES = K / TILE_K, N_TILES = N / TILE_N;
	constexpr uint32_t A_ROW_BYTES = TILE_K * 2u, A_ROW_CHUNKS = A_ROW_BYTES / 16u;
	constexpr uint32_t B_ROW_BYTES = TILE_K, B_ROW_CHUNKS = B_ROW_BYTES / 16u;
	constexpr uint32_t A_BYTES = TILE_M * A_ROW_BYTES, B_BYTES = TILE_N * B_ROW_BYTES;
	constexpr uint32_t STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_CHUNKS = TILE_M * A_ROW_CHUNKS, A_PER_THREAD = A_CHUNKS / THREADS;
	constexpr uint32_t B_CHUNKS = TILE_N * B_ROW_CHUNKS, B_PER_THREAD = B_CHUNKS / THREADS;
	static_assert(Weight::kScaled && Weight::kElementBytes == 1u && THREADS == 256u && K % TILE_K == 0u && N % TILE_N == 0u &&
		A_CHUNKS % THREADS == 0u && B_CHUNKS % THREADS == 0u && STAGES >= 2u,
		"FP8/MXFP8 MoE W2 MMA kernel matches expert geometry");
	extern __shared__ __align__(128) uint8_t hy_moe_w2_smem[];
	uint32_t *s_prefix = (uint32_t *)(hy_moe_w2_smem + STAGES * STAGE_BYTES);
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(hy_moe_w2_smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t b_row0 = threadIdx.x / B_ROW_CHUNKS, b_col0 = threadIdx.x % B_ROW_CHUNKS;
	const uint32_t b_row_step = THREADS / B_ROW_CHUNKS;
	const uint64_t b_ptr_step = (uint64_t)b_row_step * K;
	const uint64_t exp_bytes = (uint64_t)N * K;
	const uint32_t ldm_a_row = warp_m * WARP_M + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		uint32_t sum = 0u;
		s_prefix[0] = 0u;
		for ( uint32_t e = 0u; e < G::kLocalExperts; ++e )
		{
			const uint32_t cnt = __ldg(group_row_offset + e + 1u) - __ldg(group_row_offset + e);
			sum += ((cnt + TILE_M - 1u) / TILE_M) * N_TILES;
			s_prefix[e + 1u] = sum;
		}
	}
	__syncthreads();
	const uint32_t total_tiles = s_prefix[G::kLocalExperts];
	for ( uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x )
	{
		uint32_t lo = 0u, hi = G::kLocalExperts;
		while ( lo + 1u < hi )
		{
			const uint32_t mid = (lo + hi) >> 1u;
			if ( s_prefix[mid] <= tile )
				lo = mid;
			else
				hi = mid;
		}
		const uint32_t exp = lo;
		const uint32_t exp_first = __ldg(group_row_offset + exp);
		const uint32_t exp_limit = __ldg(group_row_offset + exp + 1u);
		const uint32_t m_tiles = (exp_limit - exp_first + TILE_M - 1u) / TILE_M;
		const uint32_t within = tile - s_prefix[exp];
		const uint32_t n_tile = within / m_tiles;
		const uint32_t m_tile = within - n_tile * m_tiles;
		const uint32_t row_base = exp_first + m_tile * TILE_M;
		const uint32_t neuron_base = n_tile * TILE_N;
		const uint16_t *a_src[A_PER_THREAD];
		uint32_t a_mask = 0u;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_THREAD; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * THREADS;
			const uint32_t r = chunk / A_ROW_CHUNKS, c = chunk % A_ROW_CHUNKS;
			const uint32_t packed = row_base + r;
			const uint32_t valid = packed < exp_limit ? 1u : 0u;
			a_src[q] = activation + (uint64_t)(valid != 0u ? packed : 0u) * K + c * 8u;
			a_mask |= valid << q;
		}
		const uint8_t *b_base = w2 + (uint64_t)exp * exp_bytes + (uint64_t)(neuron_base + b_row0) * K + b_col0 * 16u;
		auto issue_k = [&](uint32_t st_off)
		{
			const uint32_t stage = shared_base + st_off;
			#pragma unroll
			for ( uint32_t q = 0u; q < A_PER_THREAD; ++q )
			{
				const uint32_t chunk = threadIdx.x + q * THREADS;
				const uint32_t r = chunk / A_ROW_CHUNKS, c = chunk % A_ROW_CHUNKS;
				HyCopyL2_256(stage + r * A_ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], ((a_mask >> q) & 1u) << 4u);
				a_src[q] += TILE_K;
			}
			#pragma unroll
			for ( uint32_t q = 0u; q < B_PER_THREAD; ++q )
			{
				const uint32_t r = b_row0 + q * b_row_step;
				HyCopyL2_256(stage + A_BYTES + Weight::template Offset<B_ROW_BYTES>(r, b_col0), b_base + q * b_ptr_step, 16u);
			}
			b_base += TILE_K;
		};
		float total[MF][NF][4];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				total[i][j][0] = total[i][j][1] = total[i][j][2] = total[i][j][3] = 0.0f;
		uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( kp < K_TILES )
				issue_k(sp_off);
			LmStreamCommit();
			kp++;
			HyAdvanceStage<STAGES, STAGE_BYTES>(sp_off);
		}
		for ( uint32_t kt = 0u; kt < K_TILES; ++kt )
		{
			LmStreamWait<STAGES - 2u>();
			__syncthreads();
			if ( kp < K_TILES )
			{
				issue_k(sp_off);
				kp++;
				HyAdvanceStage<STAGES, STAGE_BYTES>(sp_off);
			}
			LmStreamCommit();
			const uint32_t stage_a = shared_base + sc_off;
			HyScaledTileMma<Format, MF, NF, A_ROW_BYTES, B_ROW_BYTES, WARP_N>(stage_a, stage_a + A_BYTES, ldm_a_row, ldm_a_col, warp_n, lane, exp,
				neuron_base, N, kt, s2, total);
			HyAdvanceStage<STAGES, STAGE_BYTES>(sc_off);
		}
		LmStreamWait<0u>();
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i )
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; ++j )
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; e += 2u )
				{
					const uint32_t r = row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					if ( r < exp_limit )
					{
						const uint32_t col = neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
						*(uint32_t *)(output + (uint64_t)r * N + col) = LmPackBf16Pair(total[i][j][e], total[i][j][e + 1u]);
					}
				}
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
	LmDependentRelease();
}

struct HyMultiLinearArgs
{
	const uint8_t *w[4];
	LmScaleTensor sc[4];
	uint16_t *out[4];
	uint32_t n[4];
	uint32_t tprefix[5];
	uint32_t count;
};

template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS_M, uint32_t WARPS_N, uint32_t STAGES>
__global__ __launch_bounds__(256u, (STAGES * (TILE_M * 256u + TILE_N * 128u) <= 49152u) ? 2 : 1)
static void HyMultiLinearMmaKernel(HyMultiLinearArgs args, const uint16_t *__restrict__ activation, uint32_t rows, uint32_t K)
{
	using Weight = LmStreamWeight<Format>;
	constexpr uint32_t TILE_K = 128u, THREADS = 256u;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_ROW_BYTES = 256u, A_ROW_CHUNKS = 16u, B_ROW_BYTES = 128u, B_ROW_CHUNKS = 8u;
	constexpr uint32_t A_BYTES = TILE_M * A_ROW_BYTES, B_BYTES = TILE_N * B_ROW_BYTES, STAGE_BYTES = A_BYTES + B_BYTES;
	constexpr uint32_t A_PER_THREAD = (TILE_M * A_ROW_CHUNKS) / THREADS, B_PER_THREAD = (TILE_N * B_ROW_CHUNKS) / THREADS;
	extern __shared__ __align__(128) uint8_t hy_mlin_smem[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(hy_mlin_smem);
	const uint32_t n_tile_global = blockIdx.x, row_base = blockIdx.y * TILE_M;
	LmDependentWait();
	if ( row_base >= rows )
	{
		LmDependentRelease();
		return;
	}
	uint32_t tgt = 0u;
	#pragma unroll
	for ( uint32_t t = 1u; t < 4u; ++t )
		if ( t < args.count && n_tile_global >= args.tprefix[t] )
			tgt = t;
	const uint32_t neuron_base = (n_tile_global - args.tprefix[tgt]) * TILE_N, N = args.n[tgt];
	const uint8_t *weight = args.w[tgt];
	const LmScaleTensor scale = args.sc[tgt];
	uint16_t *output = args.out[tgt];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, quad_row = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t b_row0 = threadIdx.x / B_ROW_CHUNKS, b_col0 = threadIdx.x % B_ROW_CHUNKS, b_row_step = THREADS / B_ROW_CHUNKS;
	const uint64_t b_ptr_step = (uint64_t)b_row_step * K;
	const uint32_t ldm_a_row = warp_m * WARP_M + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u, k_tiles = K / TILE_K;
	const uint16_t *a_src[A_PER_THREAD];
	uint32_t a_mask = 0u, b_mask = 0u;
	#pragma unroll
	for ( uint32_t q = 0u; q < A_PER_THREAD; ++q )
	{
		const uint32_t chunk = threadIdx.x + q * THREADS, r = chunk / A_ROW_CHUNKS, c = chunk % A_ROW_CHUNKS;
		const uint32_t row = row_base + r, valid = row < rows ? 1u : 0u;
		a_src[q] = activation + (uint64_t)(valid != 0u ? row : 0u) * K + c * 8u;
		a_mask |= valid << q;
	}
	#pragma unroll
	for ( uint32_t q = 0u; q < B_PER_THREAD; ++q )
		if ( neuron_base + b_row0 + q * b_row_step < N )
			b_mask |= 1u << q;
	const uint8_t *b_base = weight + (uint64_t)(neuron_base + b_row0) * K + b_col0 * 16u;
	auto issue_k = [&](uint32_t st_off)
	{
		const uint32_t stage = shared_base + st_off;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PER_THREAD; ++q )
		{
			const uint32_t chunk = threadIdx.x + q * THREADS, r = chunk / A_ROW_CHUNKS, c = chunk % A_ROW_CHUNKS;
			HyCopyL2_256(stage + r * A_ROW_BYTES + LmStreamSwizzle8(r, c) * 16u, a_src[q], ((a_mask >> q) & 1u) << 4u);
			a_src[q] += TILE_K;
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PER_THREAD; ++q )
		{
			const uint32_t r = b_row0 + q * b_row_step;
			const uint32_t bytes = ((b_mask >> q) & 1u) << 4u;
			HyCopyL2_256(stage + A_BYTES + Weight::template Offset<B_ROW_BYTES>(r, b_col0), bytes != 0u ? b_base + q * b_ptr_step : weight, bytes);
		}
		b_base += TILE_K;
	};
	float total[MF][NF][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			total[i][j][0] = total[i][j][1] = total[i][j][2] = total[i][j][3] = 0.0f;
	uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
	#pragma unroll
	for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
	{
		if ( kp < k_tiles )
			issue_k(sp_off);
		LmStreamCommit();
		kp++;
		HyAdvanceStage<STAGES, STAGE_BYTES>(sp_off);
	}
	for ( uint32_t kt = 0u; kt < k_tiles; ++kt )
	{
		LmStreamWait<STAGES - 2u>();
		__syncthreads();
		if ( kp < k_tiles )
		{
			issue_k(sp_off);
			kp++;
			HyAdvanceStage<STAGES, STAGE_BYTES>(sp_off);
		}
		LmStreamCommit();
		const uint32_t stage_a = shared_base + sc_off;
		HyScaledTileMma<Format, MF, NF, A_ROW_BYTES, B_ROW_BYTES, WARP_N>(stage_a, stage_a + A_BYTES, ldm_a_row, ldm_a_col, warp_n, lane, 0u,
			neuron_base, N, kt, scale, total);
		HyAdvanceStage<STAGES, STAGE_BYTES>(sc_off);
	}
	LmStreamWait<0u>();
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; e += 2u )
			{
				const uint32_t r = row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
				const uint32_t col = neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
				if ( r < rows && col + 1u < N )
					*(uint32_t *)(output + (uint64_t)r * N + col) = LmPackBf16Pair(total[i][j][e], total[i][j][e + 1u]);
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

	static constexpr bool kLanes = true;
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

	template<class Fn>
	static auto DispatchFormat(uint8_t codec, Fn fn)
	{
		if ( codec == HY_CODEC_BF16 ) return fn(LmBf16Format{});
		if ( codec == HY_CODEC_FP8_BLOCK128 ) return fn(LmFp8{});
		return fn(LmMxfp8{});
	}

	template<class Fn>
	static void DispatchCodecTag(uint8_t codec, Fn fn)
	{
		if ( codec == HY_CODEC_BF16 ) fn(std::integral_constant<uint32_t, HY_CODEC_BF16>{});
		else if ( codec == HY_CODEC_FP8_BLOCK128 ) fn(std::integral_constant<uint32_t, HY_CODEC_FP8_BLOCK128>{});
		else fn(std::integral_constant<uint32_t, HY_CODEC_MXFP8>{});
	}

	static int32_t SkinnyMulti(uint8_t codec, const LmSkinnyDenseTarget *targets, uint32_t count, const uint16_t *source, uint32_t rows,
		uint32_t input_dimension, SparkGpuStream stream, uint32_t exact, uint32_t skip_part = LM_SKINNY_MULTI_MAX,
		const SparkStageRunnerStep *skip_step = 0, uint32_t skip_max_context = 0u)
	{
		const uint32_t *seq = skip_step != 0 ? skip_step->sequence_of_row : (const uint32_t *)0;
		const uint32_t *ctx = skip_step != 0 ? skip_step->context_length : (const uint32_t *)0;
		const uint32_t *pos = skip_step != 0 ? skip_step->positions : (const uint32_t *)0;
		return DispatchFormat(codec, [&](auto fmt)
		{
			return LmSkinnyDenseMulti<decltype(fmt)>(targets, count, source, rows, input_dimension, stream, exact, 0u, skip_part, seq, ctx, pos, skip_max_context);
		});
	}

	static bool PrefillUnified(const SparkStageRunnerStep *step) { return CudaStagePrefillUnified(step); }

	template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS_M, uint32_t WARPS_N, uint32_t STAGES>
	static int32_t LaunchMultiLinearMma(const HyMultiLinearArgs &args, const uint16_t *source, uint32_t rows, uint32_t K, SparkGpuStream stream)
	{
		static CudaStageSmemGrant grant;
		constexpr uint32_t SMEM_BYTES = STAGES * (TILE_M * 256u + TILE_N * 128u);
		if ( !grant.Grant((const void *)HyMultiLinearMmaKernel<Format, TILE_M, TILE_N, WARPS_M, WARPS_N, STAGES>, SMEM_BYTES, 49152u) )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		const dim3 grid(args.tprefix[args.count], (rows + TILE_M - 1u) / TILE_M);
		(void)LM_LAUNCH_DEPENDENT((HyMultiLinearMmaKernel<Format, TILE_M, TILE_N, WARPS_M, WARPS_N, STAGES>), grid, 256u, SMEM_BYTES, stream,
			args, source, rows, K);
		return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	template<class Format>
	static int32_t DispatchMultiLinearMma(const HyScaled *weights, uint16_t *const *dests, const uint32_t *dims, uint32_t count,
		const uint16_t *source, uint32_t rows, uint32_t K, SparkGpuStream stream)
	{
		uint32_t n64 = 0u, n128 = 0u;
		for ( uint32_t t = 0u; t < count; ++t )
		{
			n64 += (dims[t] + 63u) / 64u;
			n128 += (dims[t] + 127u) / 128u;
		}
		const bool use_128 = rows > 96u && n128 * ((rows + 127u) / 128u) >= 48u;
		const uint32_t tile_n = use_128 ? 128u : 64u;
		HyMultiLinearArgs args = {};
		args.count = count;
		args.tprefix[0] = 0u;
		for ( uint32_t t = 0u; t < count; ++t )
		{
			args.w[t] = (const uint8_t *)weights[t].weight;
			args.sc[t] = Scales(weights[t], 1u, dims[t], K);
			args.out[t] = dests[t];
			args.n[t] = dims[t];
			args.tprefix[t + 1u] = args.tprefix[t] + (dims[t] + tile_n - 1u) / tile_n;
		}
		if ( use_128 )
			return LaunchMultiLinearMma<Format, 128u, 128u, 2u, 4u, 2u>(args, source, rows, K, stream);
		if ( rows <= 16u )
			return LaunchMultiLinearMma<Format, 16u, 64u, 1u, 8u, 3u>(args, source, rows, K, stream);
		if ( rows <= 48u || n64 * ((rows + 63u) / 64u) < 48u )
			return LaunchMultiLinearMma<Format, 32u, 64u, 2u, 4u, 3u>(args, source, rows, K, stream);
		return LaunchMultiLinearMma<Format, 64u, 64u, 2u, 4u, 2u>(args, source, rows, K, stream);
	}

	static int32_t Linear(const HyScaled &weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream, uint32_t skip_max_context = 0u)
	{
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		const bool prefill = PrefillUnified(step);
		const uint32_t min_n_16 = weight.codec == HY_CODEC_BF16 ? 4096u : 2048u;
		const bool force_stream = prefill || (exact == 0u && (rows > 16u || (rows > 8u && output_dimension >= min_n_16)));
		if ( !force_stream && skip_max_context != 0u && rows == 1u )
		{
			const LmSkinnyDenseTarget target = { weight.weight, destination, (float *)0, output_dimension, output_dimension, 0u,
				Scales(weight, 1u, output_dimension, input_dimension) };
			const int32_t status = SkinnyMulti(weight.codec, &target, 1u, source, rows, input_dimension, stream, exact, 0u, step, skip_max_context);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		if ( force_stream )
		{
			if ( weight.codec == HY_CODEC_MXFP8 && input_dimension >= 2048u && (input_dimension % 128u) == 0u &&
				output_dimension >= 1536u && (output_dimension & 1u) == 0u )
			{
				const HyScaled w1[1] = { weight };
				uint16_t *d1[1] = { destination };
				const uint32_t n1[1] = { output_dimension };
				return DispatchMultiLinearMma<LmMxfp8>(w1, d1, n1, 1u, source, rows, input_dimension, stream);
			}
			const LmScaleTensor scale = Scales(weight, 1u, output_dimension, input_dimension);
			const int32_t status = DispatchFormat(weight.codec, [&](auto fmt)
			{
				return LmStreamGemmDense<decltype(fmt)>(weight.weight, scale, source, destination, (float *)0, rows, input_dimension,
					output_dimension, 0u, 0u, step->multiprocessors, stream);
			});
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		const LmScaleTensor scale = Scales(weight, 1u, output_dimension, input_dimension);
		return DispatchFormat(weight.codec, [&](auto fmt)
		{
			return LmDenseRows<decltype(fmt)>(weight.weight, scale, source, destination, (float *)0, rows, input_dimension, output_dimension,
				exact, step->multiprocessors, stream);
		});
	}

	static int32_t LinearPair(const HyScaled &first, const HyScaled &second, const uint16_t *source, uint16_t *first_destination,
		uint16_t *second_destination, uint32_t rows, uint32_t input_dimension, uint32_t first_dimension, uint32_t second_dimension,
		const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t row_exact = step->commit == 0u ? 1u : 0u;
		const uint32_t pair_rows = first.codec == HY_CODEC_BF16 ? 16u : 8u;
		if ( !PrefillUnified(step) && first.codec == second.codec && (rows <= pair_rows || row_exact != 0u) )
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
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream, uint32_t skip_max_context = 0u)
	{
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		const bool prefill = PrefillUnified(step);
		const bool force_stream = prefill || (exact == 0u && (rows > 16u || (rows > 8u && output_dimension >= 4096u)));
		if ( !force_stream && skip_max_context != 0u && rows == 1u )
		{
			LmSkinnyArguments args;
			memset(&args, 0, sizeof(args));
			args.weight = (const uint8_t *)weight;
			args.activation = source;
			args.output_bf16 = destination;
			args.output_f32 = destination_f32;
			args.scale = LmScaleTensorNone();
			args.rows = rows;
			args.input_dimension = input_dimension;
			args.output_dimension = output_dimension;
			args.skip_sequence_of_row = step->sequence_of_row;
			args.skip_context_length = step->context_length;
			args.skip_row_position = step->positions;
			args.skip_max_context = skip_max_context;
			const int32_t status = LmSkinnyLaunch<LmBf16Format>(&args, stream, exact);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		if ( force_stream )
		{
			const int32_t status = LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows,
				input_dimension, output_dimension, 0u, 0u, step->multiprocessors, stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension, output_dimension,
			exact, step->multiprocessors, stream);
	}

	template<class Format>
	static int32_t ExpertProjectionFormat(const LmLocalExpertRoutes &routes, const HyScaled &weight, const uint16_t *activation, uint16_t *output,
		uint32_t down, uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, SparkGpuStream stream, uint32_t force_stream = 0u)
	{
		const LmScaleTensor scale = Scales(weight, G::kLocalExperts, output_dimension, input_dimension);
		if ( exact == 0u && (routes.rows >= 16u || force_stream != 0u) )
		{
			const uint32_t expected_local = (uint32_t)(((uint64_t)routes.routes * G::kLocalExperts + G::kExperts - 1u) / G::kExperts);
			const int32_t launch = LmStreamGemmGrouped<Format>(weight.weight, scale, activation, output, routes.group_row_offset,
				down != 0u ? (const uint32_t *)0 : routes.route_source_token, routes.local_experts, expected_local, input_dimension, output_dimension,
				routes.multiprocessors, stream);
			if ( launch != LM_LAUNCH_ERR_SHAPE )
				return launch;
		}
		return down != 0u ? LmLocalExpertsDown<Format>(routes, weight.weight, scale, activation, output, input_dimension, output_dimension, stream, exact) :
			LmLocalExpertsUp<Format>(routes, weight.weight, scale, activation, output, input_dimension, output_dimension, stream, exact);
	}

	static int32_t ExpertProjection(const LmLocalExpertRoutes &routes, const HyScaled &weight, const uint16_t *activation, uint16_t *output, uint32_t down,
		uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, SparkGpuStream stream, uint32_t force_stream = 0u)
	{
		return DispatchFormat(weight.codec, [&](auto fmt)
		{
			return ExpertProjectionFormat<decltype(fmt)>(routes, weight, activation, output, down, input_dimension, output_dimension, exact, stream, force_stream);
		});
	}

	static SparkStatus MergeSelected(State *state, SparkKvShard shard, const uint2 *candidates, uint64_t rank_stride, uint32_t keep,
		const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t rows = step->rows;
		for ( uint32_t first = 0u; first < rows; first += HY_SELECT_CHUNK )
		{
			const uint32_t count = rows - first < HY_SELECT_CHUNK ? rows - first : HY_SELECT_CHUNK;
			if ( LmIndexSelectMergeLaunch<LM_INDEX_SELECT_THREADS>(candidates + (uint64_t)first * keep, rank_stride, shard, keep,
					step->sequence_of_row + first, step->context_length, step->positions + first, G::kIndexTopk, G::kIndexTopk, count,
					state->bitmap_words, state->index_bitmaps, state->selected + (uint64_t)first * G::kIndexTopk, state->kv_error, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		(void)LM_LAUNCH_DEPENDENT((HySplitSelectedKernel), rows, HY_THREADS, 0, stream, shard, state->selected, step->sequence_of_row,
			step->context_length, step->positions, G::kIndexTopk, state->split_positions, state->split_offsets);
		return CudaStageLaunched();
	}

	static SparkStatus ProjectAttnIn(State *state, const HyLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream, bool has_idx,
		bool host_dense)
	{
		const uint32_t rows = step->rows, exact = step->commit == 0u ? 1u : 0u;
		const bool prefill = PrefillUnified(step);
		bool gate_done = false, idx_k_done = false, q_kv_done = false, kv_done = false;
		const uint32_t multi_rows = layer->q_a.codec == HY_CODEC_BF16 ? 16u : 8u;
		if ( !prefill && (rows <= multi_rows || exact != 0u) )
		{
			LmSkinnyDenseTarget targets[4];
			uint32_t count = 0u;
			if ( layer->q_a.codec == layer->kv_a.codec )
			{
				targets[count++] = { layer->q_a.weight, state->q_resid, (float *)0, G::kQLora, G::kQLora, 0u, Scales(layer->q_a, 1u, G::kQLora, G::kHidden) };
				targets[count++] = { layer->kv_a.weight, state->kv, (float *)0, S::kSlot, S::kSlot, 0u, Scales(layer->kv_a, 1u, S::kSlot, G::kHidden) };
				const bool add_gate = layer->gate.codec == layer->q_a.codec;
				if ( add_gate )
					targets[count++] = { layer->gate.weight, state->gate, (float *)0, S::kRankHeads * G::kValueDim, S::kRankHeads * G::kValueDim, 0u,
						Scales(layer->gate, 1u, S::kRankHeads * G::kValueDim, G::kHidden) };
				const bool add_ik = has_idx && layer->index_k.codec == layer->q_a.codec;
				if ( add_ik )
					targets[count++] = { layer->index_k.weight, state->index_k, (float *)0, G::kIndexDim, G::kIndexDim, 0u,
						Scales(layer->index_k, 1u, G::kIndexDim, G::kHidden) };
				if ( SkinnyMulti(layer->q_a.codec, targets, count, state->normed, rows, G::kHidden, stream, exact) == LM_LAUNCH_OK )
				{
					q_kv_done = true;
					kv_done = true;
					gate_done = add_gate;
					idx_k_done = add_ik;
				}
			}
		}
		else if ( (prefill || (exact == 0u && rows > 8u)) && layer->q_a.codec != HY_CODEC_BF16 && layer->q_a.codec == layer->kv_a.codec &&
			(G::kHidden % 128u) == 0u )
		{
			HyScaled ws[4];
			uint16_t *ds[4];
			uint32_t ns[4];
			uint32_t count = 0u;
			ws[count] = layer->q_a; ds[count] = state->q_resid; ns[count++] = G::kQLora;
			ws[count] = layer->kv_a; ds[count] = state->kv; ns[count++] = S::kSlot;
			const bool add_gate = layer->gate.codec == layer->q_a.codec;
			if ( add_gate )
			{
				ws[count] = layer->gate; ds[count] = state->gate; ns[count++] = S::kRankHeads * G::kValueDim;
			}
			const bool add_ik = has_idx && layer->index_k.codec == layer->q_a.codec;
			if ( add_ik )
			{
				ws[count] = layer->index_k; ds[count] = state->index_k; ns[count++] = G::kIndexDim;
			}
			const int32_t status = layer->q_a.codec == HY_CODEC_FP8_BLOCK128 ?
				DispatchMultiLinearMma<LmFp8>(ws, ds, ns, count, state->normed, rows, G::kHidden, stream) :
				DispatchMultiLinearMma<LmMxfp8>(ws, ds, ns, count, state->normed, rows, G::kHidden, stream);
			if ( status != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			q_kv_done = true;
			kv_done = true;
			gate_done = add_gate;
			idx_k_done = add_ik;
		}
		else if ( !prefill && exact == 0u && rows <= 16u && has_idx && layer->kv_a.codec == layer->index_k.codec )
		{
			const LmSkinnyDenseTarget targets[2] = {
				{ layer->kv_a.weight, state->kv, (float *)0, S::kSlot, S::kSlot, 0u, Scales(layer->kv_a, 1u, S::kSlot, G::kHidden) },
				{ layer->index_k.weight, state->index_k, (float *)0, G::kIndexDim, G::kIndexDim, 0u, Scales(layer->index_k, 1u, G::kIndexDim, G::kHidden) }
			};
			if ( SkinnyMulti(layer->kv_a.codec, targets, 2u, state->normed, rows, G::kHidden, stream, 0u) == LM_LAUNCH_OK )
			{
				kv_done = true;
				idx_k_done = true;
			}
		}
		if ( !q_kv_done )
		{
			if ( kv_done )
			{
				if ( Linear(layer->q_a, state->normed, state->q_resid, rows, G::kHidden, G::kQLora, step, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else if ( LinearPair(layer->q_a, layer->kv_a, state->normed, state->q_resid, state->kv, rows, G::kHidden, G::kQLora, S::kSlot, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( !gate_done && Linear(layer->gate, state->normed, state->gate, rows, G::kHidden, S::kRankHeads * G::kValueDim, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( has_idx && !idx_k_done && Linear(layer->index_k, state->normed, state->index_k, rows, G::kHidden, G::kIndexDim, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( has_idx && !host_dense )
		{
			if ( !prefill && (rows <= 64u || exact != 0u) )
			{
				(void)LM_LAUNCH_DEPENDENT((HyIndexWeightsKernel<G>), dim3((G::kIndexHeads + 3u) / 4u, rows), 128u, 0, stream,
					layer->index_weights, state->normed, state->index_weights, step->sequence_of_row, step->context_length, step->positions);
				if ( CudaStageLaunched() != SPARK_STATUS_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else if ( Dense16(layer->index_weights, state->normed, (uint16_t *)0, state->index_weights, rows, G::kHidden, G::kIndexHeads, step, stream,
					rows == 1u ? G::kIndexTopk : 0u) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Attention(State *state, const Frame &frame, const HyLayer &entry, uint32_t index)
	{
		const HyLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, degree = state->tp_degree, exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		const uint32_t force_mma = prefill_unified ? 1u : 0u;
		const uint64_t unit_bytes = (uint64_t)G::kHidden * sizeof(uint16_t);
		const uint64_t record_unit = unit_bytes / sizeof(float);
		const uint64_t partial_stride = ((uint64_t)rows * S::kRankHeads * S::kRecord + record_unit - 1u) / record_unit * record_unit;
		const LmKvShardView *view = &state->kv_views[index];
		const bool has_idx = G::IndexerLayer(index);
		const bool host_dense = step->gather_context != 0u && step->gather_context <= G::kIndexTopk;
		const bool run_idx = has_idx && !host_dense;
		SparkStatus status;
		if ( state->kv_attached == 0u || partial_stride > state->partial_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		status = ProjectAttnIn(state, layer, step, stream, has_idx, host_dense);
		if ( status != SPARK_STATUS_OK )
			return status;
		(void)LM_LAUNCH_DEPENDENT((HyQNormKernel<HY_THREADS, G::kQLora>), rows, HY_THREADS, 0, stream, state->q_resid, layer->q_norm, G::kEpsilon);
		if ( CudaStageLaunched() != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( Linear(layer->q_b, state->q_resid, state->q, rows, G::kQLora, S::kRankHeads * S::kQkDim, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( run_idx )
		{
			if ( !prefill_unified && (rows <= 8u || exact != 0u) )
			{
				const uint32_t blocks = (S::kIndexWidth + 7u) / 8u;
				DispatchCodecTag(layer->index_q.codec, [&](auto tag)
				{
					(void)LM_LAUNCH_DEPENDENT((HyIndexQSmallKernel<G, decltype(tag)::value>), blocks, 128u, 0, stream, layer->index_q.weight,
						layer->index_q.scale, state->q_resid, state->index_q, step->sequence_of_row, step->context_length, step->positions, rows);
				});
				if ( CudaStageLaunched() != SPARK_STATUS_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else if ( !prefill_unified && exact == 0u && rows <= 64u )
			{
				const dim3 grid(S::kIndexWidth / 64u, (rows + 15u) / 16u);
				const LmScaleTensor scale = Scales(layer->index_q, 1u, S::kIndexWidth, G::kQLora);
				DispatchFormat(layer->index_q.codec, [&](auto fmt)
				{
					(void)LM_LAUNCH_DEPENDENT((HyIndexQMmaSkipKernel<G, decltype(fmt)>), grid, LM_STREAM_THREADS, 0, stream,
						(const uint8_t *)layer->index_q.weight, scale, state->q_resid, state->index_q, step->sequence_of_row,
						step->context_length, step->positions, rows);
					return 0;
				});
				if ( CudaStageLaunched() != SPARK_STATUS_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else if ( Linear(layer->index_q, state->q_resid, state->index_q, rows, G::kQLora, S::kIndexWidth, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( rows == 1u && !prefill_unified )
		{
			(void)LM_LAUNCH_DEPENDENT((HyUkRopeB1Kernel<G>), dim3(G::kLatent / 16u + 1u, S::kRankHeads), 32u, 0, stream, state->q, layer->uk,
				step->positions, state->query);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			if ( LmPerHeadProjectChainLaunch<HY_THREADS, G::kNope, G::kLatent, S::kQkDim, 0u, S::kSlot>(state->q, layer->uk, state->query, S::kRankHeads,
					rows, stream, exact, force_mma) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((HyQueryRopePackDirectKernel<G>), dim3(rows, S::kRankHeads), 32u, 0, stream, state->q, step->positions, state->query);
		}
		const uint32_t gather_ctx = step->gather_context != 0u && step->gather_context <= state->context_capacity ? step->gather_context : 0u;
		const uint32_t old_ctx = gather_ctx > rows ? gather_ctx - rows : 0u;
		const uint64_t select_rows_max = (((uint64_t)state->max_rows + degree - 1u) / degree) * degree;
		const uint64_t comm_min = (uint64_t)degree * state->partial_stride * sizeof(float);
		const uint64_t cand_min = select_rows_max * G::kIndexTopk * sizeof(uint2);
		const uint64_t send_cap_bytes = comm_min > cand_min ? comm_min : cand_min;
		const uint32_t sparse_blocks = (rows + 3u) / 4u;
		const uint32_t union_cap = 4u * G::kIndexTopk;
		const uint64_t union_entries = (uint64_t)sparse_blocks * union_cap;
		const uint64_t pos_bytes = ((union_entries * sizeof(uint32_t) + 127u) / 128u) * 128u;
		const uint64_t mask_bytes = ((union_entries * sizeof(uint8_t) + 127u) / 128u) * 128u;
		const uint64_t cnt_bytes = (uint64_t)sparse_blocks * sizeof(uint32_t);
		const bool union_fits = pos_bytes + mask_bytes + cnt_bytes <= send_cap_bytes;
		const uint32_t prior_ctx = (!host_dense && !union_fits) ? gather_ctx : old_ctx;
		const uint32_t prior_local_keys = prior_ctx != 0u ? SparkKvShardGatherKeys(view->shard, prior_ctx) : 0u;
		const uint32_t kv_pack_units = prior_local_keys != 0u ?
			(uint32_t)(((uint64_t)prior_local_keys * S::Kv::kSlotBytes + unit_bytes - 1u) / unit_bytes) : 0u;
		const uint64_t kv_rank_stride = (uint64_t)kv_pack_units * unit_bytes;
		const uint32_t select_slice = (rows + degree - 1u) / degree;
		const uint32_t idx_keep = gather_ctx != 0u ?
			(SparkKvShardGatherKeys(view->shard, gather_ctx) < G::kIndexTopk ? SparkKvShardGatherKeys(view->shard, gather_ctx) : G::kIndexTopk) : 0u;
		const bool idx_fits = host_dense || (uint64_t)degree * select_slice * idx_keep * sizeof(uint2) <= send_cap_bytes;
		const bool use_kv_gather = prefill_unified && gather_ctx >= rows && kv_pack_units <= state->indexing_units_max &&
			kv_rank_stride <= send_cap_bytes && idx_fits;
		CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		if ( use_kv_gather )
			(void)LM_LAUNCH_DEPENDENT((HyKvPrepStoreKernel<G, true>), rows, HY_THREADS, 0, stream, state->kv, layer->kv_norm, step->positions,
				step->tree != 0u ? step->slots : step->positions, step->sequence_of_row, *view);
		else
			(void)LM_LAUNCH_DEPENDENT((HyKvPrepStoreKernel<G, false>), rows, HY_THREADS, 0, stream, state->kv, layer->kv_norm, step->positions,
				step->tree != 0u ? step->slots : step->positions, step->sequence_of_row, *view);
		if ( has_idx )
		{
			const LmKvShardView *idx_view = &state->index_views[layer->index_slot];
			const float weight_scale = (float)(1.0 / sqrt((double)G::kIndexHeads) / sqrt((double)G::kIndexDim));
			if ( host_dense )
				(void)LM_LAUNCH_DEPENDENT((HyIndexPrepStoreKernel<G, true>), rows, 128u, 0, stream, state->index_q, state->index_k,
					layer->index_k_norm, layer->index_k_bias, state->index_weights, step->positions,
					step->tree != 0u ? step->slots : step->positions, step->sequence_of_row, step->context_length, *idx_view, weight_scale);
			else
				(void)LM_LAUNCH_DEPENDENT((HyIndexPrepStoreKernel<G, false>), dim3(G::kIndexHeads + 1u, rows), 128u, 0, stream, state->index_q,
					state->index_k, layer->index_k_norm, layer->index_k_bias, state->index_weights, step->positions,
					step->tree != 0u ? step->slots : step->positions, step->sequence_of_row, step->context_length, *idx_view, weight_scale);
		}
		CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( CudaStageLaunched() != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( use_kv_gather )
		{
			if ( run_idx )
			{
				const LmKvShardView *idx_view = &state->index_views[layer->index_slot];
				const uint32_t local_keys = SparkKvShardGatherKeys(idx_view->shard, gather_ctx);
				const uint32_t score_stride = local_keys > 0u && local_keys <= state->local_capacity ? local_keys : state->local_capacity;
				const uint32_t keep = score_stride < G::kIndexTopk ? score_stride : G::kIndexTopk;
				uint2 *cand_local = (uint2 *)state->received;
				if ( LmWeightedSparseScoreLaunch<typename S::IndexKv, G::kIndexHeads, G::kIndexDim, LmKvShardView, false, float>(state->index_q,
						state->index_weights, *idx_view, step->sequence_of_row, step->context_length, step->positions, rows, score_stride, 1.0f,
						state->index_scores, stream, (const uint32_t *)0, (const uint32_t *)0, G::kIndexTopk) != SPARK_GPU_SUCCESS ||
					LmIndexSelectLocalLaunch<LM_INDEX_SELECT_THREADS>(idx_view->shard, state->index_scores, score_stride, step->sequence_of_row,
						step->context_length, step->positions, keep, G::kIndexTopk, rows, cand_local, state->kv_error, stream) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				if ( degree > 1u )
				{
					const uint32_t padded_rows = select_slice * degree;
					if ( padded_rows > rows )
					{
						const uint64_t pad_elems = (uint64_t)(padded_rows - rows) * keep;
						(void)LM_LAUNCH_DEPENDENT((HyPadCandidatesKernel), (uint32_t)((pad_elems + HY_THREADS - 1u) / HY_THREADS), HY_THREADS, 0,
							stream, cand_local, rows, padded_rows, keep);
						if ( CudaStageLaunched() != SPARK_STATUS_OK )
							SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					}
					status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, 2u * select_slice, 2u * keep,
						state->received, state->send);
					if ( status != SPARK_STATUS_OK )
						return status;
					const uint32_t own_first = state->tp_rank * select_slice;
					const uint32_t own_rows = own_first < rows ? (rows - own_first < select_slice ? rows - own_first : select_slice) : 0u;
					for ( uint32_t first = 0u; first < own_rows; first += HY_SELECT_CHUNK )
					{
						const uint32_t count = own_rows - first < HY_SELECT_CHUNK ? own_rows - first : HY_SELECT_CHUNK;
						if ( LmIndexSelectMergeLaunch<LM_INDEX_SELECT_THREADS>((const uint2 *)state->send + (uint64_t)first * keep,
								(uint64_t)select_slice * keep, idx_view->shard, keep, step->sequence_of_row + own_first + first,
								step->context_length, step->positions + own_first + first, G::kIndexTopk, G::kIndexTopk, count,
								state->bitmap_words, state->index_bitmaps, state->split_positions + (uint64_t)first * G::kIndexTopk,
								state->kv_error, stream) != SPARK_GPU_SUCCESS )
							SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					}
					status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, select_slice, 2u * G::kIndexTopk,
						state->split_positions, state->selected);
					if ( status != SPARK_STATUS_OK )
						return status;
				}
				else
				{
					for ( uint32_t first = 0u; first < rows; first += HY_SELECT_CHUNK )
					{
						const uint32_t count = rows - first < HY_SELECT_CHUNK ? rows - first : HY_SELECT_CHUNK;
						if ( LmIndexSelectMergeLaunch<LM_INDEX_SELECT_THREADS>(cand_local + (uint64_t)first * keep, (uint64_t)rows * keep,
								idx_view->shard, keep, step->sequence_of_row + first, step->context_length, step->positions + first, G::kIndexTopk,
								G::kIndexTopk, count, state->bitmap_words, state->index_bitmaps, state->selected + (uint64_t)first * G::kIndexTopk,
								state->kv_error, stream) != SPARK_GPU_SUCCESS )
							SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					}
				}
			}
			LmKvShardGatherAllView all_view = {};
			if ( prior_ctx != 0u )
			{
				(void)LM_LAUNCH_DEPENDENT((HyGatherMetaKernel), 1u, 32u, 0, stream, step->sequence_of_row, prior_ctx, state->gather_seq,
					state->gather_offset, state->gather_context);
				if ( CudaStageLaunched() != SPARK_STATUS_OK ||
					LmKvShardGatherPackLaunch<typename S::Kv, HY_THREADS>(*view, state->gather_seq, state->gather_offset, state->gather_context, 1u,
						prior_local_keys, (uint8_t *)state->send, stream) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, kv_pack_units, 0u, state->send, state->gathered);
				if ( status != SPARK_STATUS_OK )
					return status;
				if ( LmKvShardGatherAllViewInitialize<typename S::Kv>(&all_view, view->pages, view->shard, (const uint8_t *)state->gathered,
						kv_rank_stride, state->gather_offset, state->gather_context) != 0 )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			const uint64_t local_partial_stride = (uint64_t)rows * S::kRankHeads * S::kRecord;
			const float *rank_sinks = layer->sinks + (uint64_t)state->tp_rank * S::kRankHeads;
			if ( !host_dense )
			{
				if ( union_fits )
				{
					uint8_t *ubase = (uint8_t *)state->received;
					uint32_t *u_pos = (uint32_t *)ubase;
					uint8_t *u_mask = ubase + pos_bytes;
					uint32_t *u_cnt = (uint32_t *)(ubase + pos_bytes + mask_bytes);
					constexpr uint32_t sp_shared = LmPrefillAttnSharedBytes<S::kSlot, LM_PREFILL_ATTN_QUERIES>();
					static CudaStageSmemGrant sp_grant;
					if ( !sp_grant.Grant((const void *)HyGatherPrefillKernel<typename S::Kv, S::kRankHeads, G::kLatent, G::kRope, true>, sp_shared) )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					if ( has_idx )
					{
						(void)LM_LAUNCH_DEPENDENT((HyFillDenseSelectedKernel<G::kIndexTopk>), rows, HY_THREADS, 0, stream,
							step->sequence_of_row, step->context_length, step->positions, state->selected);
						(void)LM_LAUNCH_DEPENDENT((LmPrefillSparseUnionKernel<4u, G::kIndexTopk>), sparse_blocks, LM_PREFILL_UNION_THREADS, 0, stream,
							state->selected, step->positions, rows, (const uint32_t *)0, union_cap, u_pos, u_mask, u_cnt);
					}
					(void)LM_LAUNCH_DEPENDENT((HyGatherPrefillKernel<typename S::Kv, S::kRankHeads, G::kLatent, G::kRope, true>), sparse_blocks,
						LM_PREFILL_ATTN_THREADS, sp_shared, stream, state->query, all_view, state->kv, old_ctx, step->sequence_of_row,
						step->positions, rank_sinks, (float)(1.0 / sqrt((double)S::kQkDim)), state->latent, rows, u_pos, u_mask, u_cnt, union_cap);
					if ( CudaStageLaunched() != SPARK_STATUS_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				else
				{
					(void)LM_LAUNCH_DEPENDENT((HySplitSelectedKernel), rows, HY_THREADS, 0, stream, all_view.shard, state->selected,
						step->sequence_of_row, step->context_length, step->positions, G::kIndexTopk, state->split_positions, state->split_offsets);
					if ( CudaStageLaunched() != SPARK_STATUS_OK ||
						LmLatentShardAttendHeadsLaunch<typename S::Kv, LmKvShardGatherAllView, G::kLatent, G::kRope>(all_view, state->query,
							(uint64_t)rows * S::kRankHeads * S::kSlot, S::kRankHeads, S::kRankHeads, step->sequence_of_row, step->context_length,
							step->positions, state->selected, G::kIndexTopk, G::kIndexTopk, (float)(1.0 / sqrt((double)S::kQkDim)), state->send,
							local_partial_stride, S::kRankHeads, 1u, rows, stream, state->split_positions, state->split_offsets, 4u) != SPARK_GPU_SUCCESS ||
						LmLatentShardMergeLaunch<G::kLatent>(state->send, local_partial_stride, 1u, S::kRankHeads, state->latent, rows, stream,
							rank_sinks) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
			}
			else
			{
				const uint32_t dense_blocks = (rows + 3u) / 4u;
				constexpr uint32_t dn_shared = LmPrefillAttnSharedBytes<S::kSlot, LM_PREFILL_ATTN_QUERIES>();
				static CudaStageSmemGrant dn_grant;
				if ( !dn_grant.Grant((const void *)HyGatherPrefillKernel<typename S::Kv, S::kRankHeads, G::kLatent, G::kRope, false>, dn_shared) )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				(void)LM_LAUNCH_DEPENDENT((HyGatherPrefillKernel<typename S::Kv, S::kRankHeads, G::kLatent, G::kRope, false>), dense_blocks,
					LM_PREFILL_ATTN_THREADS, dn_shared, stream, state->query, all_view, state->kv, old_ctx, step->sequence_of_row, step->positions,
					rank_sinks, (float)(1.0 / sqrt((double)S::kQkDim)), state->latent, rows, (const uint32_t *)0, (const uint8_t *)0,
					(const uint32_t *)0, 0u);
				if ( CudaStageLaunched() != SPARK_STATUS_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		else
		{
			uint64_t query_stride = (uint64_t)rows * S::kRankHeads * S::kSlot;
			if ( run_idx )
			{
				const LmKvShardView *idx_view = &state->index_views[layer->index_slot];
				const uint32_t ctx = step->gather_context != 0u && step->gather_context <= state->context_capacity ? step->gather_context : state->context_capacity;
				const uint32_t local_keys = SparkKvShardGatherKeys(idx_view->shard, ctx);
				const uint32_t score_stride = local_keys > 0u && local_keys <= state->local_capacity ? local_keys : state->local_capacity;
				const uint32_t keep = score_stride < G::kIndexTopk ? score_stride : G::kIndexTopk;
				const uint64_t cand_offset = (uint64_t)rows * S::kRankHeads * S::kSlot * sizeof(uint16_t);
				const uint32_t gather_units = (uint32_t)((cand_offset + (uint64_t)rows * keep * sizeof(uint2) + unit_bytes - 1u) / unit_bytes);
				uint2 *cand_local = (uint2 *)((uint8_t *)state->query + cand_offset);
				const uint32_t *forks = step->tree != 0u ? step->fork_positions : (const uint32_t *)0;
				const uint32_t *offs = step->tree != 0u ? step->slot_offsets : (const uint32_t *)0;
				if ( rows <= 8u )
				{
					const uint32_t raw_bx = (score_stride + LM_INDEX_SCORE_WARPS * LM_INDEX_SCORE_TILE_POSITIONS - 1u) /
						(LM_INDEX_SCORE_WARPS * LM_INDEX_SCORE_TILE_POSITIONS);
					const uint32_t bx = raw_bx < 48u ? raw_bx : 48u;
					(void)LM_LAUNCH_DEPENDENT((HyIndexScoreSkipKernel<G, 1u>), dim3(bx, rows), LM_INDEX_SCORE_THREADS, 0, stream, state->index_q,
						state->index_weights, *idx_view, step->sequence_of_row, step->context_length, step->positions, rows, score_stride, 1.0f,
						state->index_scores, forks, offs, G::kIndexTopk);
				}
				else
				{
					const uint32_t raw_bx = (score_stride + LM_INDEX_SCORE_TILE_POSITIONS - 1u) / LM_INDEX_SCORE_TILE_POSITIONS;
					const uint32_t bx = raw_bx < 48u ? raw_bx : 48u;
					const uint32_t by = (rows + LM_INDEX_SCORE_WARPS - 1u) / LM_INDEX_SCORE_WARPS;
					(void)LM_LAUNCH_DEPENDENT((HyIndexScoreSkipKernel<G, LM_INDEX_SCORE_WARPS>), dim3(bx, by), LM_INDEX_SCORE_THREADS, 0, stream,
						state->index_q, state->index_weights, *idx_view, step->sequence_of_row, step->context_length, step->positions, rows,
						score_stride, 1.0f, state->index_scores, forks, offs, G::kIndexTopk);
				}
				if ( CudaStageLaunched() != SPARK_STATUS_OK ||
					LmIndexSelectLocalLaunch<LM_INDEX_SELECT_THREADS>(idx_view->shard, state->index_scores, score_stride, step->sequence_of_row,
						step->context_length, step->positions, keep, G::kIndexTopk, rows, cand_local, state->kv_error, stream) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, gather_units, 0u, state->query, state->gathered);
				if ( status != SPARK_STATUS_OK )
					return status;
				query_stride = (uint64_t)gather_units * G::kHidden;
				status = MergeSelected(state, idx_view->shard, (const uint2 *)((const uint8_t *)state->gathered + cand_offset),
					(uint64_t)gather_units * (unit_bytes / sizeof(uint2)), keep, step, stream);
				if ( status != SPARK_STATUS_OK )
					return status;
			}
			else
			{
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, S::kRankHeads * S::kSlot, state->query, state->gathered);
				if ( status != SPARK_STATUS_OK )
					return status;
			}
			if ( LmLatentShardAttendHeadsLaunch<typename S::Kv, LmKvShardView, G::kLatent, G::kRope>(*view, state->gathered, query_stride, S::kRankHeads,
					S::kRankHeads * degree, step->sequence_of_row, step->context_length, step->positions, host_dense ? (const uint32_t *)0 : state->selected,
					G::kIndexTopk, G::kIndexTopk, (float)(1.0 / sqrt((double)S::kQkDim)), state->send, partial_stride, S::kRankHeads, 1u, rows, stream,
					host_dense ? (const uint32_t *)0 : state->split_positions, host_dense ? (const uint32_t *)0 : state->split_offsets, 4u,
					step->tree != 0u ? step->fork_positions : (const uint32_t *)0, step->tree != 0u ? step->slot_offsets : (const uint32_t *)0) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)(partial_stride / record_unit), 0u, state->send,
				state->received);
			if ( status != SPARK_STATUS_OK )
				return status;
			if ( LmLatentShardMergeLaunch<G::kLatent>(state->received, partial_stride, degree, S::kRankHeads, state->latent, rows, stream,
					layer->sinks + (uint64_t)state->tp_rank * S::kRankHeads) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( rows == 1u && !prefill_unified )
			(void)LM_LAUNCH_DEPENDENT((HyUvGateB1Kernel<G>), dim3(G::kValueDim / 16u, S::kRankHeads), 32u, 0, stream, state->latent, layer->uv,
				state->gate, state->value);
		else if ( !prefill_unified && rows <= LM_PROJECT_CHAIN_ROWS )
		{
			if ( rows == 2u )
				(void)LM_LAUNCH_DEPENDENT((HyUvGateChainKernel<G, 2u, 64u>), dim3(G::kValueDim / 16u, S::kRankHeads), 64u, 0, stream, state->latent,
					layer->uv, state->gate, state->value, rows);
			else if ( rows <= 4u )
				(void)LM_LAUNCH_DEPENDENT((HyUvGateChainKernel<G, 4u, 128u>), dim3(G::kValueDim / 16u, S::kRankHeads), 128u, 0, stream, state->latent,
					layer->uv, state->gate, state->value, rows);
			else
				(void)LM_LAUNCH_DEPENDENT((HyUvGateChainKernel<G, LM_PROJECT_CHAIN_ROWS, 128u>), dim3(G::kValueDim / 16u, S::kRankHeads), 128u, 0,
					stream, state->latent, layer->uv, state->gate, state->value, rows);
		}
		else if ( exact == 0u )
			(void)LM_LAUNCH_DEPENDENT((HyUvGateMmaKernel<G>), dim3(G::kValueDim / LM_PROJECT_MMA_OUTPUTS, S::kRankHeads,
				(rows + LM_PROJECT_MMA_ROWS - 1u) / LM_PROJECT_MMA_ROWS), LM_PROJECT_MMA_THREADS, 0, stream, state->latent, layer->uv, state->gate,
				state->value, rows);
		else
		{
			if ( LmPerHeadProjectChainLaunch<HY_THREADS, G::kLatent, G::kValueDim>(state->latent, layer->uv, state->value, S::kRankHeads, rows, stream,
					exact, force_mma) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((HyOutputGateKernel<HY_THREADS>), rows, HY_THREADS, 0, stream, state->value, state->gate, S::kRankHeads * G::kValueDim);
		}
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

	template<class Format, uint32_t TILE_M, uint32_t WARPS_M, uint32_t WARPS_N, uint32_t STAGES>
	static SparkStatus LaunchFusedMoeMma(const State *state, const HyLayer *layer, const uint16_t *normed, uint32_t expected_local, uint32_t sms,
		SparkGpuStream stream)
	{
		static CudaStageSmemGrant grant;
		constexpr uint32_t THREADS = WARPS_M * WARPS_N * 32u;
		constexpr uint32_t SMEM_BYTES = STAGES * (TILE_M * 256u + 2u * 64u * 128u) + (G::kLocalExperts + 1u) * (uint32_t)sizeof(uint32_t);
		uint32_t blocks = 0u;
		if ( !grant.GrantBlocks((const void *)HyMoeMmaW1W3SwigluKernel<G, Format, TILE_M, WARPS_M, WARPS_N, STAGES>, SMEM_BYTES, THREADS, sms, &blocks, 49152u) )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		const uint64_t tile_bound = ((uint64_t)expected_local / TILE_M + G::kLocalExperts) * (G::kMoeInter / 64u);
		if ( tile_bound < blocks )
			blocks = (uint32_t)tile_bound;
		(void)LM_LAUNCH_DEPENDENT((HyMoeMmaW1W3SwigluKernel<G, Format, TILE_M, WARPS_M, WARPS_N, STAGES>), blocks, THREADS, SMEM_BYTES, stream,
			(const uint8_t *)layer->moe_w1.weight, Scales(layer->moe_w1, G::kLocalExperts, G::kMoeInter, G::kHidden),
			(const uint8_t *)layer->moe_w3.weight, Scales(layer->moe_w3, G::kLocalExperts, G::kMoeInter, G::kHidden),
			normed, state->packed_weight, state->route_source_token, state->group_row_offset, state->expert_act);
		return CudaStageLaunched();
	}

	template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS_M, uint32_t WARPS_N, uint32_t STAGES>
	static SparkStatus LaunchMoeW2Mma(const State *state, const HyLayer *layer, uint32_t expected_local, uint32_t sms, SparkGpuStream stream)
	{
		static CudaStageSmemGrant grant;
		constexpr uint32_t THREADS = WARPS_M * WARPS_N * 32u;
		constexpr uint32_t SMEM_BYTES = STAGES * (TILE_M * 256u + TILE_N * 128u) + (G::kLocalExperts + 1u) * (uint32_t)sizeof(uint32_t);
		uint32_t blocks = 0u;
		if ( !grant.GrantBlocks((const void *)HyMoeMmaW2Kernel<G, Format, TILE_M, TILE_N, WARPS_M, WARPS_N, STAGES>, SMEM_BYTES, THREADS, sms, &blocks, 49152u) )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		const uint64_t tile_bound = ((uint64_t)expected_local / TILE_M + G::kLocalExperts) * (G::kHidden / TILE_N);
		if ( tile_bound < blocks )
			blocks = (uint32_t)tile_bound;
		(void)LM_LAUNCH_DEPENDENT((HyMoeMmaW2Kernel<G, Format, TILE_M, TILE_N, WARPS_M, WARPS_N, STAGES>), blocks, THREADS, SMEM_BYTES, stream,
			(const uint8_t *)layer->moe_w2.weight, Scales(layer->moe_w2, G::kLocalExperts, G::kHidden, G::kMoeInter),
			state->expert_act, state->group_row_offset, state->expert_out);
		return CudaStageLaunched();
	}

	template<class Format>
	static SparkStatus DispatchFusedMoePair(const State *state, const HyLayer *layer, const uint16_t *normed, uint32_t expected_local, uint32_t sms,
		SparkGpuStream stream)
	{
		const uint32_t mean_rows = (expected_local + G::kLocalExperts - 1u) / G::kLocalExperts;
		SparkStatus st;
		if ( mean_rows > 40u )
			st = LaunchFusedMoeMma<Format, 96u, 2u, 4u, 2u>(state, layer, normed, expected_local, sms, stream);
		else if ( mean_rows > 24u )
			st = LaunchFusedMoeMma<Format, 64u, 2u, 4u, 3u>(state, layer, normed, expected_local, sms, stream);
		else if ( mean_rows > 10u )
			st = LaunchFusedMoeMma<Format, 32u, 2u, 4u, 2u>(state, layer, normed, expected_local, sms, stream);
		else
			st = LaunchFusedMoeMma<Format, 16u, 1u, 8u, 2u>(state, layer, normed, expected_local, sms, stream);
		if ( st != SPARK_STATUS_OK )
			return st;
		if ( mean_rows > 40u )
			return LaunchMoeW2Mma<Format, 96u, 64u, 2u, 4u, 3u>(state, layer, expected_local, sms, stream);
		if ( mean_rows > 24u )
			return LaunchMoeW2Mma<Format, 64u, 128u, 2u, 4u, 3u>(state, layer, expected_local, sms, stream);
		if ( mean_rows > 10u )
			return LaunchMoeW2Mma<Format, 32u, 128u, 2u, 4u, 2u>(state, layer, expected_local, sms, stream);
		return LaunchMoeW2Mma<Format, 16u, 128u, 1u, 8u, 2u>(state, layer, expected_local, sms, stream);
	}

	static SparkStatus SharedFfn(State *state, const HyLayer *layer, const uint16_t *normed, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t rows = step->rows, exact = step->commit == 0u ? 1u : 0u;
		const bool prefill = PrefillUnified(step);
		if ( !prefill && (rows <= 8u || exact != 0u) )
		{
			const uint32_t blocks = (state->shared_rows + 1u) / 2u;
			DispatchCodecTag(layer->shared_gate_up.codec, [&](auto tag)
			{
				(void)LM_LAUNCH_DEPENDENT((HySharedGateUpSiluKernel<G, decltype(tag)::value>), blocks, 64u, 0, stream, layer->shared_gate_up.weight,
					layer->shared_gate_up.scale, normed, state->shared_act, state->shared_rows, rows);
			});
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			if ( Linear(layer->shared_gate_up, normed, state->shared_gate_up, rows, G::kHidden, 2u * state->shared_rows, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((HySiluMulKernel<HY_THREADS>), dim3(LmVecBlocks<HY_THREADS>(state->shared_rows), rows), HY_THREADS, 0, stream,
				state->shared_gate_up, state->shared_act, state->shared_rows);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( Linear(layer->shared_down, state->shared_act, state->shared_out, rows, state->shared_rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
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
		const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
		const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
		const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
		if ( routes <= 32u )
			(void)LM_LAUNCH_DEPENDENT((HySigmoidRouteSmallKernel<G>), 1u, HY_THREADS, 0, stream, state->router_logits, layer->router_bias, rows,
				state->expert_first, step->route_fault, state->route_expert, state->route_weight, state->route_global, state->group_row_offset,
				state->route_packed_row, state->route_source_token, state->packed_weight, tile_m, n_up, state->tile_prefix_up, n_down,
				state->tile_prefix_down);
		else
		{
			constexpr uint32_t WARPS = HY_THREADS / LM_WARP_LANES;
			(void)LM_LAUNCH_DEPENDENT((HySigmoidTopkLocalKernel<G>), (rows + WARPS - 1u) / WARPS, HY_THREADS, 0, stream, state->router_logits,
				layer->router_bias, rows, state->expert_first, step->route_fault, state->route_expert, state->route_weight);
			(void)LM_LAUNCH_DEPENDENT((HyRouteBuildPackedKernel<HY_THREADS, G::kLocalExperts + 1u>), 1u, HY_THREADS, 0, stream,
				state->route_expert, state->route_weight, routes, G::kTopK, state->group_row_offset, state->route_packed_row,
				state->route_source_token, state->packed_weight, state->route_global, tile_m, n_up, state->tile_prefix_up, n_down,
				state->tile_prefix_down);
		}
		if ( CudaStageLaunched() != SPARK_STATUS_OK || SharedFfn(state, layer, normed, step, stream) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		const uint32_t force_stream = PrefillUnified(step) ? 1u : 0u;
		const uint32_t sms = step->multiprocessors != 0u ? step->multiprocessors : 48u;
		const uint32_t expected_local = (uint32_t)(((uint64_t)routes * G::kLocalExperts + G::kExperts - 1u) / G::kExperts);
		if ( force_stream == 0u && (exact != 0u || rows <= 8u) &&
			layer->moe_w1.codec != HY_CODEC_BF16 && layer->moe_w1.codec == layer->moe_w3.codec && layer->moe_w1.codec == layer->moe_w2.codec )
		{
			const uint32_t persistent_blocks = sms * 4u;
			auto run_fast = [&](auto fp8_tag)
			{
				constexpr bool FP8 = decltype(fp8_tag)::value;
				(void)LM_LAUNCH_DEPENDENT((HyMoeFastW1W3SwigluKernel<G, FP8>), persistent_blocks, 128u, 0, stream,
					(const uint8_t *)layer->moe_w1.weight, layer->moe_w1.scale, (const uint8_t *)layer->moe_w3.weight, layer->moe_w3.scale,
					normed, state->packed_weight, state->route_source_token, state->group_row_offset, state->expert_act, rows);
				if ( rows == 1u )
				{
					constexpr uint32_t DN_CHUNKS = G::kMoeInter / 16u;
					constexpr uint32_t DN_LANES = DN_CHUNKS >= 32u ? 32u : (DN_CHUNKS >= 16u ? 16u : 8u);
					constexpr uint32_t DN_COLS = (128u / DN_LANES) * 4u;
					(void)LM_LAUNCH_DEPENDENT((HyMoeSmallW2FinalizeKernel<G, FP8>), dim3((G::kHidden + DN_COLS - 1u) / DN_COLS, 1u), 128u, 0, stream,
						(const uint8_t *)layer->moe_w2.weight, layer->moe_w2.scale, state->expert_act, state->route_packed_row, state->route_expert,
						state->shared_out, state->partial);
				}
				else
				{
					(void)LM_LAUNCH_DEPENDENT((HyMoeFastW2Kernel<G, FP8>), persistent_blocks, 128u, 0, stream,
						(const uint8_t *)layer->moe_w2.weight, layer->moe_w2.scale, state->expert_act, state->group_row_offset, state->expert_out);
					(void)LM_LAUNCH_DEPENDENT((HyMoeFinalizeKernel), dim3(LmVecBlocks<HY_THREADS>(G::kHidden), rows), HY_THREADS, 0, stream,
						state->expert_out, state->route_packed_row, state->route_expert, state->shared_out, state->partial, G::kTopK, G::kLocalExperts,
						G::kHidden);
				}
			};
			if ( layer->moe_w1.codec == HY_CODEC_FP8_BLOCK128 ) run_fast(std::true_type{});
			else run_fast(std::false_type{});
			return CudaStageLaunched();
		}
		const LmLocalExpertRoutes expert_routes = CudaStageBuildExpertRoutes<LmLocalExpertRoutes>(state, rows, G::kTopK, G::kLocalExperts, step->multiprocessors);
		if ( exact == 0u && (force_stream != 0u || rows > 8u) && layer->moe_w1.codec != HY_CODEC_BF16 &&
			layer->moe_w1.codec == layer->moe_w3.codec && layer->moe_w1.codec == layer->moe_w2.codec )
		{
			const SparkStatus mma_status = layer->moe_w1.codec == HY_CODEC_FP8_BLOCK128 ?
				DispatchFusedMoePair<LmFp8>(state, layer, normed, expected_local, sms, stream) :
				DispatchFusedMoePair<LmMxfp8>(state, layer, normed, expected_local, sms, stream);
			if ( mma_status != SPARK_STATUS_OK )
				return mma_status;
			(void)LM_LAUNCH_DEPENDENT((HyMoeFinalizeKernel), dim3(LmVecBlocks<HY_THREADS>(G::kHidden), rows), HY_THREADS, 0, stream, state->expert_out,
				state->route_packed_row, state->route_expert, state->shared_out, state->partial, G::kTopK, G::kLocalExperts, G::kHidden);
			return CudaStageLaunched();
		}
		if ( ExpertProjection(expert_routes, layer->moe_w1, normed, state->expert_gate, 0u, G::kHidden, G::kMoeInter, exact, stream, force_stream) != LM_LAUNCH_OK ||
			ExpertProjection(expert_routes, layer->moe_w3, normed, state->expert_up, 0u, G::kHidden, G::kMoeInter, exact, stream, force_stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((HyMoeSwigluPackedKernel), dim3(LmVecBlocks<HY_THREADS>(G::kMoeInter), routes), HY_THREADS, 0, stream,
			state->expert_gate, state->expert_up, state->packed_weight, state->group_row_offset, G::kLocalExperts, state->expert_act, G::kMoeInter,
			G::kSwigluLimit);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			ExpertProjection(expert_routes, layer->moe_w2, state->expert_act, state->expert_out, 1u, G::kMoeInter, G::kHidden, exact, stream, force_stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((HyMoeFinalizeKernel), dim3(LmVecBlocks<HY_THREADS>(G::kHidden), rows), HY_THREADS, 0, stream, state->expert_out,
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
		static_assert((uint64_t)G::kTopK * G::kHidden * sizeof(uint16_t) >= (uint64_t)(2u * G::kStreams + 1u) * LM_HC_MIX_LANES * sizeof(float),
			"expert_out holds per-row HC dot partials when rows exceed split limit");
		const bool ffn = part == HY_FFN_PART;
		const uint32_t rows = frame.step->rows;
		const float *fn = ffn ? layer.mlp_hc_fn : layer.attn_hc_fn;
		const float *hc_scale = ffn ? layer.mlp_hc_scale : layer.attn_hc_scale;
		const float *hc_base = ffn ? layer.mlp_hc_base : layer.attn_hc_base;
		const uint16_t *norm_w = ffn ? layer.post_norm : layer.input_norm;
		if constexpr ( G::kStreams == 4u && (G::kHidden % (HY_THREADS * 8u)) == 0u )
		{
			if ( rows > 512u )
			{
				static CudaStageSmemGrant hc_grant;
				constexpr uint32_t HC_SMEM = G::kStreams * G::kHidden * (uint32_t)sizeof(uint16_t);
				if ( !hc_grant.Grant((const void *)HyHcFusedSinglePassKernel<G::kStreams, 2u * G::kStreams, G::kHidden, HAS_POST>, HC_SMEM) )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				(void)LM_LAUNCH_DEPENDENT((HyHcFusedSinglePassKernel<G::kStreams, 2u * G::kStreams, G::kHidden, HAS_POST>), rows, HY_THREADS, HC_SMEM,
					frame.stream, state->streams, state->hc_post, state->partial, fn, hc_base, hc_scale, state->hc_post, norm_w, state->normed,
					G::kEpsilon, G::kHcEpsilon, G::kHcMagnitude);
				return CudaStageLaunched();
			}
		}
		float *hc_buf = rows <= LM_HC_MIX_SPLIT_ROWS ? state->hc_partial : (float *)state->expert_out;
		if ( rows <= 16u )
			(void)LM_LAUNCH_DEPENDENT((HyHcPostDotSmallKernel<G::kStreams, 2u * G::kStreams, G::kHidden, HAS_POST>), dim3(rows, 16u), 128u, 0,
				frame.stream, state->streams, state->hc_post, state->partial, fn, hc_buf);
		else
			(void)LM_LAUNCH_DEPENDENT((HyHcPostDotTileKernel<G::kStreams, 2u * G::kStreams, G::kHidden, HAS_POST, 8u>), dim3((rows + 7u) / 8u, 16u),
				128u, 0, frame.stream, state->streams, state->hc_post, state->partial, fn, hc_buf, rows);
		(void)LM_LAUNCH_DEPENDENT((HyHcMixNormSmallKernel<G::kStreams, 2u * G::kStreams, G::kHidden>), rows, HY_THREADS, 0, frame.stream,
			state->streams, hc_buf, hc_base, hc_scale, state->hc_post, norm_w, state->normed, G::kEpsilon, G::kHcEpsilon, G::kHcMagnitude);
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
		const SparkStatus status = CudaStageCheckTreeBegin(step);
		return status != SPARK_STATUS_OK ? status : CudaStage<HyModel>::Begin(state, step, stream, frame);
	}

	static SparkStatus KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream_void)
	{
		State *state = (State *)model;
		SparkGpuStream stream = (SparkGpuStream)stream_void;
		const SparkStatus status = CudaStageKeepRowsCheck(state, step, kept);
		if ( status != SPARK_STATUS_OK || step->tree == 0u )
			return status;
		CudaStageKeepMoveViews<typename S::Kv, HY_THREADS>(state, state->kv_views, G::kLayers, S::kSlot, step->rows, step, kept, stream);
		CudaStageKeepMoveViews<typename S::IndexKv, HY_THREADS>(state, state->index_views, G::kIndexLayers, G::kIndexDim, step->rows, step, kept, stream);
		return CudaStageLaunched();
	}

	static SparkStatus AllReduce(State *state, const Frame &frame) { return CudaStageAllReduceSum(state, frame.stream, state->partial, frame.step->rows); }

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, routes = rows * G::kTopK, groups = G::kLocalExperts + 2u;
		const uint64_t unit_bytes = (uint64_t)G::kHidden * sizeof(uint16_t);
		const uint64_t width = rows * unit_bytes, chunk = rows < HY_SELECT_CHUNK ? rows : HY_SELECT_CHUNK;
		const uint64_t select_rows = ((rows + state->tp_degree - 1u) / state->tp_degree) * state->tp_degree;
		const uint64_t comm_min = (uint64_t)state->tp_degree * state->partial_stride * sizeof(float);
		const uint64_t cand_min = select_rows * G::kIndexTopk * sizeof(uint2);
		const uint64_t comm_bytes = comm_min > cand_min ? comm_min : cand_min;
		const CudaStageSlice plan[] =
		{
			{ (void **)&state->streams, rows * S::kResidual * sizeof(uint16_t) },
			{ (void **)&state->hc_post, rows * G::kStreams * sizeof(float) },
			{ (void **)&state->hc_partial, LmHcMixPartialBytes(rows, 2u * G::kStreams) },
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, width },
			{ (void **)&state->q_resid, rows * G::kQLora * sizeof(uint16_t) },
			{ (void **)&state->q, rows * S::kRankHeads * S::kQkDim * sizeof(uint16_t) },
			{ (void **)&state->query, (uint64_t)state->indexing_units_max * unit_bytes },
			{ (void **)&state->kv, rows * S::kSlot * sizeof(uint16_t) },
			{ (void **)&state->gathered, (uint64_t)state->tp_degree * state->indexing_units_max * unit_bytes },
			{ (void **)&state->send, comm_bytes },
			{ (void **)&state->received, comm_bytes },
			{ (void **)&state->latent, rows * S::kRankHeads * G::kLatent * sizeof(uint16_t) },
			{ (void **)&state->value, rows * S::kRankHeads * G::kValueDim * sizeof(uint16_t) },
			{ (void **)&state->gate, rows * S::kRankHeads * G::kValueDim * sizeof(uint16_t) },
			{ (void **)&state->index_q, rows * S::kIndexWidth * sizeof(uint16_t) },
			{ (void **)&state->index_k, rows * G::kIndexDim * sizeof(uint16_t) },
			{ (void **)&state->index_weights, rows * G::kIndexHeads * sizeof(float) },
			{ (void **)&state->index_scores, rows * state->local_capacity * sizeof(float) },
			{ (void **)&state->index_bitmaps, chunk * 2u * state->bitmap_words * sizeof(uint32_t) },
			{ (void **)&state->selected, select_rows * G::kIndexTopk * sizeof(uint32_t) },
			{ (void **)&state->split_positions, select_rows * G::kIndexTopk * sizeof(uint32_t) },
			{ (void **)&state->split_offsets, rows * (state->tp_degree + 1u) * sizeof(uint32_t) },
			{ (void **)&state->gather_seq, sizeof(uint32_t) },
			{ (void **)&state->gather_offset, (uint64_t)state->slots * sizeof(uint32_t) },
			{ (void **)&state->gather_context, (uint64_t)state->slots * sizeof(uint32_t) },
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
		const uint64_t unit_bytes = (uint64_t)G::kHidden * sizeof(uint16_t);
		const uint64_t record_unit = unit_bytes / sizeof(float);
		state->local_capacity = ((state->context_capacity + state->tp_degree - 1u) / state->tp_degree + 63u) / 64u * 64u;
		state->context_capacity = (state->context_capacity + 63u) / 64u * 64u;
		state->local_capacity = state->local_capacity > G::kIndexTopk ? state->local_capacity : G::kIndexTopk;
		state->context_capacity = state->context_capacity > G::kIndexTopk ? state->context_capacity : G::kIndexTopk;
		state->bitmap_words = LmIndexSelectBitmapWords(state->context_capacity);
		const uint64_t query_max_bytes = (uint64_t)state->max_rows * S::kRankHeads * S::kSlot * sizeof(uint16_t);
		const uint64_t cand_max_bytes = (uint64_t)state->max_rows * G::kIndexTopk * sizeof(uint2);
		state->indexing_units_max = (uint32_t)((query_max_bytes + cand_max_bytes + unit_bytes - 1u) / unit_bytes);
		state->partial_stride = ((uint64_t)state->max_rows * S::kRankHeads * S::kRecord + record_unit - 1u) / record_unit * record_unit;
		return SPARK_STATUS_OK;
	}

	static uint32_t Accepts(const SparkStageRunnerConfiguration *configuration) { return configuration->kv_pages_per_sequence != 0u ? 1u : 0u; }
	static SparkStatus Prepare(State *state, const SparkStageRunnerConfiguration *configuration) { state->context_capacity = configuration->kv_pages_per_sequence * G::kPageSlots; return SPARK_STATUS_OK; }

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		const uint32_t expected_first = configuration->tp_rank * G::kLocalExperts;
		const CudaStagePackCheck checks[] =
		{
			{ "tp_degree", configuration->tp_degree }, { "tp_rank", configuration->tp_rank }, { "hidden", G::kHidden }, { "layers", G::kLayers },
			{ "vocab_rows", G::kVocab / configuration->tp_degree }, { "experts", G::kExperts }, { "local_experts", G::kLocalExperts },
			{ "expert_first", expected_first }, { "top_k", G::kTopK }
		};
		const CudaStagePackValue widths[] = { { "ffn_rows", &state->ffn_rows }, { "shared_rows", &state->shared_rows } };
		SparkStatus status = CudaStagePackExpect(G::Tag(), &state->pack, checks);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( CudaStagePackValues(&state->pack, widths) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( state->ffn_rows % 64u != 0u || state->shared_rows % 64u != 0u )
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		state->expert_first = expected_first;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		geometry->tree_verify = 1u;
		return CudaStage<HyModel>::HeadCheck(state, geometry, false, false, true);
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
				field->codec = HY_CODEC_FP8_BLOCK128;
				return tensors->AddFp8Block(G::Tag(), index, name, rows, columns, HY_FP8_BLOCK, &field->weight, &field->scale);
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
