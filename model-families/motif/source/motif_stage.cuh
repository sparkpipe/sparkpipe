#pragma once

#include <atomic>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/gemm.cuh"
#include "runtime/launch.h"
#include "inference/kernels/attn_prefill.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/diff_attn.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/hc.cuh"
#include "inference/kernels/hc_site.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/poly_norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/project_chain.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_group.cuh"
#include "inference/runner/stage_model_cuda.cuh"
#include "sparkpipe/spark_motif_family.h"
#include "sparkpipe/spark_rope_plan.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_gpu.h"

#define MO_HIDDEN SPARK_MOTIF_MODEL_HIDDEN_DIMENSION
#define MO_LAYERS SPARK_MOTIF_MODEL_LAYER_COUNT
#define MO_VOCAB SPARK_MOTIF_MODEL_VOCABULARY_SIZE
#define MO_EPS SPARK_MOTIF_MODEL_RMS_NORM_EPSILON
#define MO_GROUPS SPARK_MOTIF_MODEL_NOISE_HEAD_COUNT
#define MO_GROUP SPARK_MOTIF_MODEL_HEAD_GROUP_SIZE
#define MO_SIGNAL (MO_GROUP - 1u)
#define MO_HEAD SPARK_MOTIF_MODEL_HEAD_DIMENSION
#define MO_ROPE SPARK_MOTIF_MODEL_ROPE_DIMENSION
#define MO_NOPE SPARK_MOTIF_MODEL_NOPE_DIMENSION
#define MO_VALUE SPARK_MOTIF_MODEL_VALUE_DIMENSION
#define MO_QA SPARK_MOTIF_MODEL_QUERY_LORA_RANK
#define MO_LATENT SPARK_MOTIF_MODEL_KV_LORA_RANK
#define MO_SLOT SPARK_MOTIF_MODEL_LATENT_ROW_DIMENSION
#define MO_WINDOW SPARK_MOTIF_MODEL_WINDOW_TOKENS
#define MO_EXPERTS SPARK_MOTIF_MODEL_MOE_EXPERT_COUNT
#define MO_TOP_K SPARK_MOTIF_MODEL_MOE_TOP_K
#define MO_EXPERT_INTER SPARK_MOTIF_MODEL_MOE_EXPERT_INTERMEDIATE
#define MO_DENSE_INTER SPARK_MOTIF_MODEL_DENSE_INTERMEDIATE
#define MO_THREADS 256u
#define MO_RECORD LM_LATENT_SHARD_RECORD_FLOATS(MO_LATENT)
#define MO_PREFILL_Q_TILE 16u
#define MO_PREFILL_KV_TILE 32u

static_assert(MO_HEAD == MO_NOPE + MO_ROPE && MO_SLOT == MO_LATENT + MO_ROPE, "a query head is nope | rope and a latent slot is latent | rope");
static_assert(SPARK_MOTIF_MODEL_HC_STREAMS == LM_HC_SITE_STREAMS && SPARK_MOTIF_MODEL_HC_MIX_DIMENSION == LM_HC_SITE_MIX, "four mHC streams");
static_assert(MO_THREADS == LM_HC_SITE_THREADS, "the HC site runs the stage block size");

using MotifKv = LmKvGeometry<SPARK_MOTIF_MODEL_LATENT_ROW_BYTES, SPARK_MOTIF_MODEL_KV_PAGE_SLOTS, true>;

struct MotifHcPolicy
{
	static constexpr float kNormEpsilon = SPARK_MOTIF_MODEL_HC_NORM_EPSILON;

	static __device__ void Gates(const float *mixes, const float *scale3, const float *base, float *pre, float *post)
	{
		const uint32_t lane = threadIdx.x % LM_WARP_LANES;
		const float clamp = SPARK_MOTIF_MODEL_HC_GATE_CLAMP;
		if ( lane < LM_HC_SITE_STREAMS )
		{
			pre[lane] = SparkMathSigmoid(SparkMathClamp(mixes[lane] * scale3[0] + base[lane], clamp));
			post[lane] = SparkMathScaledSigmoid(SPARK_MOTIF_MODEL_HC_POST_COEFFICIENT, SparkMathClamp(mixes[LM_HC_SITE_STREAMS + lane] * scale3[1] +
				base[LM_HC_SITE_STREAMS + lane], clamp));
		}
	}

	static __device__ void Comb(const float *mixes, const float *scale3, const float *base, float *comb_out)
	{
		const uint32_t lane = threadIdx.x % LM_WARP_LANES, cell = lane & 15u;
		const float clamp = SPARK_MOTIF_MODEL_HC_RESIDUAL_CLAMP, floor = SPARK_MOTIF_MODEL_HC_SUM_FLOOR;
		float value = mixes[2u * LM_HC_SITE_STREAMS + cell] * scale3[2] + base[2u * LM_HC_SITE_STREAMS + cell];
		value = SPARK_MATH_EXP(SparkMathClamp(value, clamp));
		for ( uint32_t iteration = 0u; iteration < SPARK_MOTIF_MODEL_HC_SINKHORN_ITERATIONS; ++iteration )
		{
			value /= fmaxf(LmHcRowSum(value, lane), floor);
			value /= fmaxf(LmHcColumnSum(value, lane), floor);
		}
		if ( lane < LM_HC_SITE_STREAMS * LM_HC_SITE_STREAMS )
			comb_out[cell] = value;
	}
};

__global__ __launch_bounds__(MO_THREADS, 2)
static void MotifHcFinishNormKernel(const float *__restrict__ partials, const uint16_t *__restrict__ streams_bf16,
	const float *__restrict__ scale3_f32, const float *__restrict__ base_f32, float *__restrict__ post_f32, float *__restrict__ comb_f32,
	const float *__restrict__ norm_weight, uint16_t *__restrict__ normed_bf16, uint32_t rows)
{
	using S = LmHcSiteShape<MO_HIDDEN>;
	constexpr uint32_t SLOTS = MO_HIDDEN / (MO_THREADS * 8u);
	static_assert(MO_HIDDEN % (MO_THREADS * 8u) == 0u && SLOTS == 2u, "motif hidden covers two vectorized spans per thread");
	__shared__ float mixes[LM_HC_SITE_MIX + 1u];
	__shared__ float pre[LM_HC_SITE_STREAMS];
	__shared__ float reduction[MO_THREADS / LM_WARP_LANES];
	const uint32_t row = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
	const float *row_partials = partials + (uint64_t)row * LM_HC_SITE_CLUSTER * (LM_HC_SITE_MIX + 1u);
	const uint16_t *row_streams = streams_bf16 + (uint64_t)row * S::kFlat;
	if ( threadIdx.x < LM_HC_SITE_MIX + 1u )
	{
		float total = 0.0f;
		#pragma unroll
		for ( uint32_t rank = 0u; rank < LM_HC_SITE_CLUSTER; ++rank )
			total += row_partials[rank * (LM_HC_SITE_MIX + 1u) + threadIdx.x];
		mixes[threadIdx.x] = total;
	}
	__syncthreads();
	if ( warp == 0u )
	{
		const float inverse = SparkMathRmsScale(mixes[LM_HC_SITE_MIX], (float)S::kFlat, MotifHcPolicy::kNormEpsilon);
		if ( lane < LM_HC_SITE_MIX )
			mixes[lane] = mixes[lane] * inverse;
		__syncwarp();
		MotifHcPolicy::Gates(mixes, scale3_f32, base_f32, pre, post_f32 + (uint64_t)row * LM_HC_SITE_STREAMS);
		__syncwarp();
		MotifHcPolicy::Comb(mixes, scale3_f32, base_f32, comb_f32 + (uint64_t)row * LM_HC_SITE_STREAMS * LM_HC_SITE_STREAMS);
	}
	__syncthreads();
	const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
	uint4 staged[SLOTS];
	float sum_sq = 0.0f;
	#pragma unroll
	for ( uint32_t slot = 0u; slot < SLOTS; ++slot )
	{
		const uint32_t off = (slot * MO_THREADS + threadIdx.x) * 8u;
		float f0[8], f1[8], f2[8], f3[8], out[8];
		LmLoadBf16x8AsFloat(__ldg((const uint4 *)(row_streams + 0u * MO_HIDDEN + off)), f0);
		LmLoadBf16x8AsFloat(__ldg((const uint4 *)(row_streams + 1u * MO_HIDDEN + off)), f1);
		LmLoadBf16x8AsFloat(__ldg((const uint4 *)(row_streams + 2u * MO_HIDDEN + off)), f2);
		LmLoadBf16x8AsFloat(__ldg((const uint4 *)(row_streams + 3u * MO_HIDDEN + off)), f3);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			out[k] = (((0.0f + p0 * f0[k]) + p1 * f1[k]) + p2 * f2[k]) + p3 * f3[k];
		staged[slot] = LmPackUint4FloatToBf16(out);
		sum_sq = LmSumSqBf16x8(staged[slot], sum_sq);
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<MO_THREADS>(sum_sq, reduction), (float)MO_HIDDEN, MO_EPS);
	uint16_t *out_row = normed_bf16 + (uint64_t)row * MO_HIDDEN;
	#pragma unroll
	for ( uint32_t slot = 0u; slot < SLOTS; ++slot )
	{
		const uint32_t off = (slot * MO_THREADS + threadIdx.x) * 8u;
		float x[8], w[8], out[8];
		LmLoadBf16x8AsFloat(staged[slot], x);
		LmLoadWeight8(norm_weight, off, w);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			out[k] = SparkMathNormGainRounded(x[k] * scale, w[k]);
		*(uint4 *)(out_row + off) = LmPackUint4FloatToBf16(out);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(MO_THREADS, 2)
static void MotifHcPostInPlaceKernel(const uint16_t *__restrict__ out_bf16, const float *__restrict__ post_f32,
	const float *__restrict__ comb_f32, uint16_t *__restrict__ streams_bf16, uint32_t rows)
{
	constexpr uint32_t SLOTS = MO_HIDDEN / (MO_THREADS * 8u);
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
	float p[LM_HC_SITE_STREAMS], c[LM_HC_SITE_STREAMS][LM_HC_SITE_STREAMS];
	#pragma unroll
	for ( uint32_t s = 0u; s < LM_HC_SITE_STREAMS; ++s )
		p[s] = __ldg(post_f32 + (uint64_t)row * LM_HC_SITE_STREAMS + s);
	#pragma unroll
	for ( uint32_t s = 0u; s < LM_HC_SITE_STREAMS; ++s )
		#pragma unroll
		for ( uint32_t src = 0u; src < LM_HC_SITE_STREAMS; ++src )
			c[s][src] = __ldg(comb_f32 + (uint64_t)row * 16u + s * LM_HC_SITE_STREAMS + src);
	const uint16_t *o_row = out_bf16 + (uint64_t)row * MO_HIDDEN;
	uint16_t *s_row = streams_bf16 + (uint64_t)row * LM_HC_SITE_STREAMS * MO_HIDDEN;
	#pragma unroll
	for ( uint32_t slot = 0u; slot < SLOTS; ++slot )
	{
		const uint32_t off = (slot * MO_THREADS + threadIdx.x) * 8u;
		float x[8], r[LM_HC_SITE_STREAMS][8];
		LmLoadBf16x8AsFloat(__ldg((const uint4 *)(o_row + off)), x);
		#pragma unroll
		for ( uint32_t src = 0u; src < LM_HC_SITE_STREAMS; ++src )
			LmLoadBf16x8AsFloat(*(const uint4 *)(s_row + (uint64_t)src * MO_HIDDEN + off), r[src]);
		#pragma unroll
		for ( uint32_t s = 0u; s < LM_HC_SITE_STREAMS; ++s )
		{
			float res[8];
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				float v = 0.0f;
				#pragma unroll
				for ( uint32_t src = 0u; src < LM_HC_SITE_STREAMS; ++src )
					v = fmaf(c[s][src], r[src][k], v);
				res[k] = fmaf(p[s], x[k], v);
			}
			*(uint4 *)(s_row + (uint64_t)s * MO_HIDDEN + off) = LmPackUint4FloatToBf16(res);
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t DIM, bool ROUND_FIRST>
__global__ __launch_bounds__(THREADS, 4)
static void MotifRmsNormVecKernel(uint16_t *__restrict__ data, const float *__restrict__ weight, uint32_t rows, uint32_t stride)
{
	static_assert(DIM == THREADS * 8u, "one vectorized chunk per thread");
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	if ( row < rows )
	{
		uint16_t *row_ptr = data + (uint64_t)row * stride;
		const uint32_t off = threadIdx.x * 8u;
		const uint4 v = *(const uint4 *)(row_ptr + off);
		const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(LmSumSqBf16x8(v, 0.0f), reduction), (float)DIM, MO_EPS);
		float x[8], w[8], out[8];
		LmLoadBf16x8AsFloat(v, x);
		LmLoadWeight8(weight, off, w);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			out[k] = ROUND_FIRST ? SparkMathNormGainRounded(x[k] * scale, w[k]) : (x[k] * scale * w[k]);
		*(uint4 *)(row_ptr + off) = LmPackUint4FloatToBf16(out);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 4)
static void MotifKvNormRopeStoreKernel(uint16_t *__restrict__ slot_bf16, const float *__restrict__ kv_norm,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ store_positions, const uint32_t *__restrict__ sequence_of_row,
	const float *__restrict__ inv_freq, float theta, LmKvShardView kv_view, uint32_t rows)
{
	static_assert(MO_LATENT == 512u && MO_ROPE == 64u && MO_SLOT == 576u, "motif latent slot geometry");
	__shared__ float reduction[4u];
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	if ( row < rows )
	{
		uint16_t *row_slot = slot_bf16 + (uint64_t)row * MO_SLOT;
		uint4 lat_v = make_uint4(0u, 0u, 0u, 0u);
		float sq = 0.0f;
		if ( threadIdx.x < 64u )
		{
			lat_v = *(const uint4 *)(row_slot + threadIdx.x * 8u);
			sq = LmSumSqBf16x8(lat_v, 0.0f);
		}
		const float scale = SparkMathRmsScale(LmBlockSumOnce<128u>(sq, reduction), (float)MO_LATENT, MO_EPS);
		const uint32_t store_pos = __ldg(store_positions + row);
		const bool owns = SparkKvShardOwns(kv_view.shard, store_pos) != 0u;
		uint16_t *kv_dst = owns ? (uint16_t *)LmKvShardSlotRequired<MotifKv>(kv_view, __ldg(sequence_of_row + row), store_pos, row, LM_KV_ACCESS_WRITE) : (uint16_t *)0;
		if ( threadIdx.x < 64u )
		{
			const uint32_t off = threadIdx.x * 8u;
			float x[8], w[8], out[8];
			LmLoadBf16x8AsFloat(lat_v, x);
			LmLoadWeight8(kv_norm, off, w);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out[k] = SparkMathNormGainRounded(x[k] * scale, w[k]);
			const uint4 packed = LmPackUint4FloatToBf16(out);
			*(uint4 *)(row_slot + off) = packed;
			if ( kv_dst != 0 )
				*(uint4 *)(kv_dst + off) = packed;
		}
		else if ( threadIdx.x < 96u )
		{
			const uint32_t pair = threadIdx.x - 64u;
			const float pos = (float)__ldg(positions + row);
			const float angle = pos * (inv_freq != 0 ? __ldg(inv_freq + pair) : __powf(theta, -2.0f * (float)pair / (float)MO_ROPE));
			float lo = LmBf16ToFloat(row_slot[MO_LATENT + pair]);
			float hi = LmBf16ToFloat(row_slot[MO_LATENT + 32u + pair]);
			LmRopePairRounded<LM_ROPE_BF16_TRIG>(&lo, &hi, angle, 1.0f);
			const uint16_t b_lo = LmFloatToBf16(lo), b_hi = LmFloatToBf16(hi);
			row_slot[MO_LATENT + pair] = b_lo;
			row_slot[MO_LATENT + 32u + pair] = b_hi;
			if ( kv_dst != 0 )
			{
				kv_dst[MO_LATENT + pair] = b_lo;
				kv_dst[MO_LATENT + 32u + pair] = b_hi;
			}
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(32u, 8)
static void MotifQueryRopeStoreKernel(const uint16_t *__restrict__ q_bf16, const uint32_t *__restrict__ positions,
	const float *__restrict__ inv_freq, float theta, uint16_t *__restrict__ query_bf16, uint32_t heads, uint32_t rows)
{
	const uint32_t row = blockIdx.x, head = blockIdx.y, pair = threadIdx.x;
	LmDependentWait();
	if ( row < rows && head < heads && pair < MO_ROPE / 2u )
	{
		const uint16_t *src = q_bf16 + ((uint64_t)row * heads + head) * MO_HEAD + MO_NOPE;
		uint16_t *dst = query_bf16 + ((uint64_t)row * heads + head) * MO_SLOT + MO_LATENT;
		const float pos = (float)__ldg(positions + row);
		const float angle = pos * (inv_freq != 0 ? __ldg(inv_freq + pair) : __powf(theta, -2.0f * (float)pair / (float)MO_ROPE));
		float lo = LmBf16ToFloat(src[pair]);
		float hi = LmBf16ToFloat(src[32u + pair]);
		LmRopePairRounded<LM_ROPE_BF16_TRIG>(&lo, &hi, angle, 1.0f);
		dst[pair] = LmFloatToBf16(lo);
		dst[32u + pair] = LmFloatToBf16(hi);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(MO_THREADS, 2)
static void MotifWindowListKernel(const uint32_t *__restrict__ positions, uint32_t *__restrict__ list, uint32_t rows)
{
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	if ( row < rows )
	{
		const uint32_t position = positions[row], first = position + 1u >= MO_WINDOW ? position + 1u - MO_WINDOW : 0u;
		for ( uint32_t index = threadIdx.x; index < MO_WINDOW; index += blockDim.x )
			list[(uint64_t)row * MO_WINDOW + index] = first + index <= position ? first + index : position;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 4)
static void MotifPriorKvDirectKernel(LmKvShardView kv_view, const uint32_t *__restrict__ sequence_of_row, uint32_t win_begin,
	uint32_t prior_keys, uint16_t *__restrict__ kv_prior)
{
	constexpr uint32_t SLOT_V = MO_SLOT >> 3u;
	static_assert(SLOT_V == 72u && SLOT_V <= 128u, "one thread per 16-byte vector of a latent slot");
	const uint32_t k_idx = blockIdx.x;
	LmDependentWait();
	if ( k_idx < prior_keys && threadIdx.x < SLOT_V )
	{
		const uint32_t seq = __ldg(sequence_of_row);
		const uint32_t pos = win_begin + k_idx;
		const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<MotifKv>(kv_view, seq, pos, 0u, LM_KV_ACCESS_READ);
		if ( slot != 0 )
			((uint4 *)(kv_prior + (uint64_t)k_idx * MO_SLOT))[threadIdx.x] = __ldg((const uint4 *)slot + threadIdx.x);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 4)
static void MotifPriorKvPackKernel(LmKvShardView kv_view, const uint32_t *__restrict__ sequence_of_row, uint32_t win_begin,
	uint32_t old_ctx, uint32_t b0, uint32_t max_peer_keys, uint16_t *__restrict__ send_slots)
{
	constexpr uint32_t SLOT_V = MO_SLOT >> 3u;
	const uint32_t idx = blockIdx.x;
	const uint32_t degree = kv_view.shard.degree, grain = kv_view.shard.grain, my_rank = kv_view.shard.rank;
	const uint32_t first_b = b0 + ((my_rank + degree - (b0 % degree)) % degree);
	const uint32_t pos = (first_b + (idx / grain) * degree) * grain + (idx % grain);
	LmDependentWait();
	if ( idx < max_peer_keys && pos >= win_begin && pos < old_ctx && threadIdx.x < SLOT_V )
	{
		const uint32_t seq = __ldg(sequence_of_row);
		const uint16_t *slot = (const uint16_t *)LmKvShardSlotRequired<MotifKv>(kv_view, seq, pos, 0u, LM_KV_ACCESS_READ);
		if ( slot != 0 )
			((uint4 *)(send_slots + (uint64_t)idx * MO_SLOT))[threadIdx.x] = __ldg((const uint4 *)slot + threadIdx.x);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 4)
static void MotifPriorKvUnpackKernel(SparkKvShard shard, const uint8_t *__restrict__ gathered_bytes, uint64_t ag_stride_bytes,
	uint32_t win_begin, uint32_t old_ctx, uint32_t b0, uint32_t max_peer_keys, uint16_t *__restrict__ kv_prior)
{
	constexpr uint32_t SLOT_V = MO_SLOT >> 3u;
	const uint32_t idx = blockIdx.x, src_rank = blockIdx.y;
	const uint32_t degree = shard.degree, grain = shard.grain;
	const uint32_t first_b = b0 + ((src_rank + degree - (b0 % degree)) % degree);
	const uint32_t pos = (first_b + (idx / grain) * degree) * grain + (idx % grain);
	LmDependentWait();
	if ( idx < max_peer_keys && src_rank < degree && pos >= win_begin && pos < old_ctx && threadIdx.x < SLOT_V )
	{
		const uint32_t k_idx = pos - win_begin;
		const uint4 *s4 = (const uint4 *)(gathered_bytes + (uint64_t)src_rank * ag_stride_bytes) + (uint64_t)idx * SLOT_V;
		uint4 *d4 = (uint4 *)(kv_prior + (uint64_t)k_idx * MO_SLOT);
		d4[threadIdx.x] = __ldg(s4 + threadIdx.x);
	}
	LmDependentRelease();
}

static __host__ __device__ constexpr uint32_t MotifLocalPrefillSharedBytes(void)
{
	return 2u * MO_PREFILL_KV_TILE * MO_SLOT * 2u
		+ 2u * MO_PREFILL_Q_TILE * (MO_PREFILL_KV_TILE + 8u) * 4u
		+ MO_PREFILL_Q_TILE * (MO_PREFILL_KV_TILE + 8u) * 2u
		+ 2u * MO_PREFILL_Q_TILE * 4u;
}

__global__ __launch_bounds__(MO_THREADS, 1)
static void MotifMlaLocalPrefillKernel(const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ kv_prior,
	const uint16_t *__restrict__ kv_cur, const uint32_t *__restrict__ positions, uint32_t rows, uint32_t rank_heads,
	uint32_t win_begin, uint32_t prior_keys, uint32_t window, float qk_scale, uint16_t *__restrict__ merged_out)
{
	constexpr uint32_t Q_TILE = MO_PREFILL_Q_TILE;
	constexpr uint32_t KV_TILE = MO_PREFILL_KV_TILE;
	constexpr uint32_t DIM = MO_SLOT;
	constexpr uint32_t LATENT = MO_LATENT;
	constexpr uint32_t ROW_BYTES = DIM * 2u;
	constexpr uint32_t TILE_BYTES = KV_TILE * ROW_BYTES;
	constexpr uint32_t CHUNKS = DIM / 8u;
	constexpr uint32_t HALF_K_STEPS = (DIM / 16u) / 2u;
	constexpr uint32_t SCORE_STRIDE = KV_TILE + 8u;
	constexpr uint32_t WARP_COLS = LATENT / 8u;
	constexpr uint32_t VAL_FRAGS = WARP_COLS / 8u;
	static_assert(DIM == 576u && LATENT == 512u && HALF_K_STEPS == 18u && VAL_FRAGS == 8u, "motif MLA MMA geometry");
	extern __shared__ __align__(128) uint8_t mo_prefill_smem[];
	uint8_t *kv_smem = mo_prefill_smem;
	float *score_parts = (float *)(kv_smem + 2u * TILE_BYTES);
	uint16_t *weights_smem = (uint16_t *)(score_parts + 2u * Q_TILE * SCORE_STRIDE);
	float *row_rescale = (float *)(weights_smem + Q_TILE * SCORE_STRIDE);
	float *row_inv_sum = row_rescale + Q_TILE;
	const uint32_t kv_base0 = (uint32_t)__cvta_generic_to_shared(kv_smem);
	const uint32_t kv_base1 = kv_base0 + TILE_BYTES;
	const uint32_t w_base = (uint32_t)__cvta_generic_to_shared(weights_smem);
	const uint32_t local_h = blockIdx.x, q_block = blockIdx.y;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t k_half = warp & 1u, k_grp = warp >> 1u;
	const uint32_t row_first = q_block * Q_TILE;
	const uint32_t row_end = row_first + Q_TILE < rows ? row_first + Q_TILE : rows;
	const uint32_t total_kv = prior_keys + rows;
	const float scale_log2 = qk_scale * 1.4426950408889634f;
	LmDependentWait();
	if ( row_first >= rows )
	{
		LmDependentRelease();
		return;
	}
	for ( uint32_t idx = threadIdx.x; idx < Q_TILE * CHUNKS; idx += MO_THREADS )
	{
		const uint32_t r = idx / CHUNKS, c = idx % CHUNKS;
		const bool vr = row_first + r < row_end;
		const uint16_t *src = vr ? query_bf16 + ((uint64_t)(row_first + r) * rank_heads + local_h) * DIM + c * 8u : query_bf16;
		LmPrefillCopy(kv_base0 + r * ROW_BYTES + LmPrefillSwizzle(r, c) * 16u, src, vr ? 16u : 0u);
	}
	asm volatile("cp.async.commit_group;\n" ::: "memory");
	asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	uint32_t q_frag[HALF_K_STEPS][4];
	{
		const uint32_t a_row = (matrix & 1u) * 8u + within;
		#pragma unroll
		for ( uint32_t ks = 0u; ks < HALF_K_STEPS; ++ks )
		{
			const uint32_t a_chunk = 2u * (k_half * HALF_K_STEPS + ks) + (matrix >> 1u);
			LmPrefillLoadX4(kv_base0 + a_row * ROW_BYTES + LmPrefillSwizzle(a_row, a_chunk) * 16u, q_frag[ks]);
		}
	}
	__syncthreads();
	const uint32_t min_pos = __ldg(positions + row_first);
	const uint32_t max_pos = __ldg(positions + row_end - 1u);
	const uint32_t cta_start = (window != 0u && min_pos + 1u > window) ? (min_pos + 1u - window) : 0u;
	const uint32_t cta_end = max_pos + 1u;
	const uint32_t tile_first = cta_start > win_begin ? (cta_start - win_begin) / KV_TILE : 0u;
	const uint32_t tile_end = cta_end > win_begin ? (cta_end - win_begin + KV_TILE - 1u) / KV_TILE : 0u;
	const uint32_t s_row0 = warp * 2u, s_row1 = warp * 2u + 1u;
	const bool s_valid0 = row_first + s_row0 < row_end, s_valid1 = row_first + s_row1 < row_end;
	const uint32_t s_pos0 = s_valid0 ? __ldg(positions + row_first + s_row0) : 0u;
	const uint32_t s_pos1 = s_valid1 ? __ldg(positions + row_first + s_row1) : 0u;
	const uint32_t s_start0 = (window != 0u && s_pos0 + 1u > window) ? (s_pos0 + 1u - window) : 0u;
	const uint32_t s_start1 = (window != 0u && s_pos1 + 1u > window) ? (s_pos1 + 1u - window) : 0u;
	float rmax0 = -INFINITY, rmax1 = -INFINITY, rsum0 = 0.0f, rsum1 = 0.0f;
	float output[VAL_FRAGS][4];
	#pragma unroll
	for ( uint32_t f = 0u; f < VAL_FRAGS; ++f )
		output[f][0] = output[f][1] = output[f][2] = output[f][3] = 0.0f;
	auto copy_kv = [&](uint32_t tile, uint32_t dst_base)
	{
		for ( uint32_t idx = threadIdx.x; idx < KV_TILE * CHUNKS; idx += MO_THREADS )
		{
			const uint32_t s = idx / CHUNKS, c = idx % CHUNKS;
			const uint32_t k_idx = tile * KV_TILE + s;
			const bool valid = k_idx < total_kv;
			const uint16_t *src = valid ?
				(k_idx < prior_keys ? kv_prior + (uint64_t)k_idx * DIM + c * 8u : kv_cur + (uint64_t)(k_idx - prior_keys) * DIM + c * 8u) :
				query_bf16;
			LmPrefillCopy(dst_base + s * ROW_BYTES + LmPrefillSwizzle(s, c) * 16u, src, valid ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	if ( tile_first < tile_end )
		copy_kv(tile_first, kv_base0);
	uint32_t stage = 0u;
	const uint32_t q_lo = lane >> 2u, q_hi = q_lo + 8u;
	const uint32_t k_col = k_grp * 8u + (lane & 3u) * 2u;
	for ( uint32_t tile = tile_first; tile < tile_end; ++tile )
	{
		const uint32_t cur_base = stage == 0u ? kv_base0 : kv_base1;
		const uint32_t nxt_base = stage == 0u ? kv_base1 : kv_base0;
		if ( tile + 1u < tile_end )
			copy_kv(tile + 1u, nxt_base);
		if ( tile + 1u < tile_end )
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		float sc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
		const uint32_t b_row = k_grp * 8u + (lane & 7u);
		#pragma unroll
		for ( uint32_t ks = 0u; ks < HALF_K_STEPS; ++ks )
		{
			uint32_t b[2];
			const uint32_t b_chunk = 2u * (k_half * HALF_K_STEPS + ks) + ((lane >> 3u) & 1u);
			LmPrefillLoadX2(cur_base + b_row * ROW_BYTES + LmPrefillSwizzle(b_row, b_chunk) * 16u, b);
			LmMmaBf16(sc, q_frag[ks], b);
		}
		float *my_sp = score_parts + k_half * Q_TILE * SCORE_STRIDE;
		my_sp[q_lo * SCORE_STRIDE + k_col] = sc[0];
		my_sp[q_lo * SCORE_STRIDE + k_col + 1u] = sc[1];
		my_sp[q_hi * SCORE_STRIDE + k_col] = sc[2];
		my_sp[q_hi * SCORE_STRIDE + k_col + 1u] = sc[3];
		__syncthreads();
		const uint32_t key_pos = win_begin + tile * KV_TILE + lane;
		{
			const bool vis = s_valid0 && key_pos >= s_start0 && key_pos <= s_pos0;
			const float s = vis ? (score_parts[s_row0 * SCORE_STRIDE + lane] + score_parts[(Q_TILE + s_row0) * SCORE_STRIDE + lane]) : -INFINITY;
			float peak = s;
			#pragma unroll
			for ( uint32_t off = 16u; off > 0u; off >>= 1u )
				peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, off));
			const float next_max = fmaxf(rmax0, peak);
			const float rescale = rmax0 == -INFINITY ? 0.0f : SparkMathExp2Fast((rmax0 - next_max) * scale_log2);
			rmax0 = next_max;
			const float wt = (next_max == -INFINITY || s == -INFINITY) ? 0.0f : SparkMathExp2Fast(fmaf(s, scale_log2, -next_max * scale_log2));
			float wsum = wt;
			#pragma unroll
			for ( uint32_t off = 16u; off > 0u; off >>= 1u )
				wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
			rsum0 = rsum0 * rescale + wsum;
			weights_smem[s_row0 * SCORE_STRIDE + lane] = LmFloatToBf16(wt);
			if ( lane == 0u )
				row_rescale[s_row0] = rescale;
		}
		{
			const bool vis = s_valid1 && key_pos >= s_start1 && key_pos <= s_pos1;
			const float s = vis ? (score_parts[s_row1 * SCORE_STRIDE + lane] + score_parts[(Q_TILE + s_row1) * SCORE_STRIDE + lane]) : -INFINITY;
			float peak = s;
			#pragma unroll
			for ( uint32_t off = 16u; off > 0u; off >>= 1u )
				peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, off));
			const float next_max = fmaxf(rmax1, peak);
			const float rescale = rmax1 == -INFINITY ? 0.0f : SparkMathExp2Fast((rmax1 - next_max) * scale_log2);
			rmax1 = next_max;
			const float wt = (next_max == -INFINITY || s == -INFINITY) ? 0.0f : SparkMathExp2Fast(fmaf(s, scale_log2, -next_max * scale_log2));
			float wsum = wt;
			#pragma unroll
			for ( uint32_t off = 16u; off > 0u; off >>= 1u )
				wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
			rsum1 = rsum1 * rescale + wsum;
			weights_smem[s_row1 * SCORE_STRIDE + lane] = LmFloatToBf16(wt);
			if ( lane == 0u )
				row_rescale[s_row1] = rescale;
		}
		__syncthreads();
		const float rs_lo = row_rescale[q_lo], rs_hi = row_rescale[q_hi];
		if ( rs_lo != 1.0f || rs_hi != 1.0f )
		{
			#pragma unroll
			for ( uint32_t f = 0u; f < VAL_FRAGS; ++f )
			{
				output[f][0] *= rs_lo;
				output[f][1] *= rs_lo;
				output[f][2] *= rs_hi;
				output[f][3] *= rs_hi;
			}
		}
		const uint32_t w_row = (matrix & 1u) * 8u + within;
		const uint32_t w_half = (matrix >> 1u) * 8u;
		#pragma unroll
		for ( uint32_t p_step = 0u; p_step < 2u; ++p_step )
		{
			uint32_t a_p[4];
			LmPrefillLoadX4(w_base + (w_row * SCORE_STRIDE + p_step * 16u + w_half) * 2u, a_p);
			#pragma unroll
			for ( uint32_t f = 0u; f < VAL_FRAGS; f += 2u )
			{
				uint32_t b[4], v0[2], v1[2];
				const uint32_t v_row = p_step * 16u + (matrix & 1u) * 8u + within;
				const uint32_t v_chunk = warp * VAL_FRAGS + f + (matrix >> 1u);
				LmPrefillLoadX4Trans(cur_base + v_row * ROW_BYTES + LmPrefillSwizzle(v_row, v_chunk) * 16u, b);
				v0[0] = b[0]; v0[1] = b[1];
				v1[0] = b[2]; v1[1] = b[3];
				LmMmaBf16(output[f], a_p, v0);
				LmMmaBf16(output[f + 1u], a_p, v1);
			}
		}
		stage ^= 1u;
		if ( tile + 1u < tile_end )
			__syncthreads();
	}
	if ( lane == 0u )
	{
		row_inv_sum[s_row0] = s_valid0 && rsum0 > 0.0f ? (1.0f / rsum0) : 0.0f;
		row_inv_sum[s_row1] = s_valid1 && rsum1 > 0.0f ? (1.0f / rsum1) : 0.0f;
	}
	__syncthreads();
	const uint32_t out_lo = row_first + q_lo, out_hi = row_first + q_hi;
	const float inv_lo = row_inv_sum[q_lo], inv_hi = row_inv_sum[q_hi];
	#pragma unroll
	for ( uint32_t f = 0u; f < VAL_FRAGS; ++f )
	{
		const uint32_t col = warp * WARP_COLS + f * 8u + (lane & 3u) * 2u;
		if ( out_lo < row_end )
			__stcs((uint32_t *)(merged_out + ((uint64_t)out_lo * rank_heads + local_h) * LATENT + col),
				LmPackBf16Pair(output[f][0] * inv_lo, output[f][1] * inv_lo));
		if ( out_hi < row_end )
			__stcs((uint32_t *)(merged_out + ((uint64_t)out_hi * rank_heads + local_h) * LATENT + col),
				LmPackBf16Pair(output[f][2] * inv_hi, output[f][3] * inv_hi));
	}
	LmDependentRelease();
}

template<uint32_t LOCAL_EXPERTS>
__global__ __launch_bounds__(MO_THREADS, 1)
static void MotifRoutePrepareKernel(uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ route_global,
	uint32_t routes, uint32_t expert_first, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row,
	uint32_t *__restrict__ route_source_token, uint32_t *__restrict__ param_of_row, uint32_t tile_m, uint32_t neuron_tiles_up,
	uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	LmDependentWait();
	for ( uint32_t route = threadIdx.x; route < routes; route += MO_THREADS )
	{
		const uint32_t expert = route_expert[route];
		route_global[route] = expert;
		if ( expert >= expert_first && expert < expert_first + LOCAL_EXPERTS )
			route_expert[route] = expert - expert_first;
		else
		{
			route_expert[route] = LOCAL_EXPERTS;
			route_weight[route] = 0.0f;
		}
	}
	__syncthreads();
	LmRouteBuildBlock<MO_THREADS, LOCAL_EXPERTS + 1u>(route_expert, routes, MO_TOP_K, group_row_offset, route_packed_row, route_source_token, tile_m,
		neuron_tiles_up, tile_prefix_up, neuron_tiles_down, tile_prefix_down);
	__syncthreads();
	for ( uint32_t route = threadIdx.x; route < routes; route += MO_THREADS )
	{
		const uint32_t exp = route_expert[route];
		param_of_row[route_packed_row[route]] = exp < LOCAL_EXPERTS ? exp : 0u;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(MO_THREADS, 2)
static void MotifPolyNormActiveKernel(const uint16_t *__restrict__ gate_up_bf16, const float *__restrict__ params_f32,
	const uint32_t *__restrict__ param_of_row, const uint32_t *__restrict__ group_row_offset, uint32_t local_experts,
	uint16_t *__restrict__ out_bf16, uint32_t width)
{
	__shared__ float reduction[3u][MO_THREADS / LM_WARP_LANES];
	__shared__ __align__(16) uint16_t raw_wave[MO_THREADS * 8u];
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	if ( row >= __ldg(group_row_offset + local_experts) )
	{
		LmDependentRelease();
		return;
	}
	const uint16_t *gate = gate_up_bf16 + (uint64_t)row * 2u * width, *up = gate + width;
	const float *params = params_f32 + (uint64_t)LM_POLY_NORM_PARAMS * __ldg(param_of_row + row);
	const float clamp = SPARK_MOTIF_MODEL_HIDDEN_CLAMP, bias_clamp = SPARK_MOTIF_MODEL_POLYNORM_ROUTED_BIAS_CLAMP;
	const float out_scale = SPARK_MOTIF_MODEL_POLYNORM_OUTPUT_SCALE, epsilon = SPARK_MOTIF_MODEL_POLYNORM_EPSILON;
	const float w0 = SparkMathSigmoid(__ldg(params + 0u)), w1 = SparkMathSigmoid(__ldg(params + 1u)), w2 = SparkMathSigmoid(__ldg(params + 2u));
	const float bias = SparkMathClamp(__ldg(params + 3u), bias_clamp);
	float square = 0.0f, fourth = 0.0f, sixth = 0.0f;
	const uint4 g0 = LmPolyNormWaveAccum<MO_THREADS>(gate, raw_wave, 0u, width, clamp, &square, &fourth, &sixth);
	LmPolyNormBlockSum3<MO_THREADS>(&square, &fourth, &sixth, reduction);
	const float count = (float)width;
	const float d1 = sqrtf(square / count + epsilon), d2 = sqrtf(fourth / count + epsilon), d3 = sqrtf(sixth / count + epsilon);
	const uint32_t off = threadIdx.x * 8u;
	if ( off < width )
		LmPolyNormStore8<true>(g0, __ldg((const uint4 *)(up + off)), clamp, d1, d2, d3, w0, w1, w2, bias, out_scale,
			out_bf16 + (uint64_t)row * width + off);
	LmDependentRelease();
}

struct MotifLayer
{
	const float *attn_norm, *ffn_norm;
	const float *hc_attn_fn, *hc_attn_scale, *hc_attn_base, *hc_ffn_fn, *hc_ffn_scale, *hc_ffn_base;
	const uint16_t *q_a, *kv_a, *lambda, *q_b, *gate, *out, *uk, *uv;
	const float *q_norm, *kv_norm;
	const uint16_t *ffn_gate_up, *ffn_down, *router, *moe_w13, *moe_w2, *shared_gate_up, *shared_down;
	const float *ffn_poly, *expert_bias, *moe_poly, *shared_poly;
	uint32_t window, owner;
};

template<class G>
struct MotifState : CudaStagePackState
{
	static constexpr uint32_t kGroups = MO_GROUPS / G::kTpDegree;
	static constexpr uint32_t kHeads = kGroups * MO_GROUP;
	static constexpr uint32_t kSignal = kGroups * MO_SIGNAL;
	static constexpr uint32_t kLocalExperts = MO_EXPERTS / G::kTpDegree;
	uint32_t expert_first, collective_capacity;
	uint64_t steps, prefill_rows;
	MotifLayer layers[MO_LAYERS];
	uint16_t *streams, *snapshot, *collapsed, *normed, *q_c, *q, *q_latent, *query, *gathered, *slot, *merged, *values, *lambda, *gate, *diff, *partial;
	float *hc_mixes, *hc_pre, *hc_post, *hc_comb, *partials, *received, *router_logits, *route_weight, *full_inv_freq;
	uint32_t *windows, *route_expert, *route_global, *group_row_offset, *route_packed_row, *route_source_token, *tile_prefix_up, *tile_prefix_down;
	uint32_t *param_of_row;
	uint16_t *expert_gate_up, *expert_act, *expert_out, *mlp_gate_up, *mlp_act, *shared_out;
	LmKvShardView views[MO_LAYERS];
};

template<class G>
struct MotifStage : CudaStage<MotifStage<G>>
{
	using State = MotifState<G>;
	using Schedule = MotifSchedule<G, MotifStage>;
	using Kv = CudaStageKv<CudaStageKvViews<MotifKv, &State::views, 0u, MO_LAYERS>>;
	using LaneShared = CudaStageFields<&State::full_inv_freq, &State::kv_error>;
	static constexpr uint32_t kGroups = State::kGroups, kHeads = State::kHeads, kSignal = State::kSignal, kLocalExperts = State::kLocalExperts;
	static constexpr bool kLanes = true;
	static constexpr bool kKeepRows = true;
	static constexpr uint32_t kTpDegree = G::kTpDegree;
	static constexpr uint32_t kHidden = MO_HIDDEN;
	static constexpr uint32_t kVocab = MO_VOCAB;
	static constexpr uint32_t kLayers = MO_LAYERS;
	static constexpr float kEpsilon = MO_EPS;
	static constexpr uint32_t kPackAlignment = G::kPackAlignment;

	struct Frame
	{
		const SparkStageRunnerStep *step;
		SparkGpuStream stream;
		const MotifLayer *layer;
		uint32_t index;
	};

	static const char *Tag(void) { return G::Tag(); }
	static const char *ReportTag(void) { return G::ReportTag(); }
	static const char *WeightdModel(void) { return G::WeightdModel(); }
	static const char *WeightdRevision(void) { return G::WeightdRevision(); }
	static uint32_t QuerySequences(uint32_t rows) { return (uint32_t)(((uint64_t)rows * kHeads * MO_SLOT + MO_HIDDEN - 1u) / MO_HIDDEN); }
	static uint32_t RecordSequences(uint32_t rows) { return (uint32_t)(((uint64_t)rows * kHeads * MO_RECORD * 2u + MO_HIDDEN - 1u) / MO_HIDDEN); }

	static bool PrefillUnified(const SparkStageRunnerStep *step) { return CudaStagePrefillUnified(step); }

	static SparkStatus Check(int32_t status, const char *site, uint32_t layer)
	{
		if ( status == LM_LAUNCH_OK && SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS )
			return SPARK_STATUS_OK;
		fprintf(stderr, "%s: %s failed at layer %u status=%d cuda=%s\n", G::Tag(), site, layer, (int)status, SparkGpuGetErrorString(SparkGpuPeekAtLastError()));
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}

	static int32_t LinearOut(const uint16_t *weight, const uint16_t *source, uint16_t *dst_bf16, float *dst_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		if ( PrefillUnified(step) )
			return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), source, dst_bf16, dst_f32, rows, input_dimension, output_dimension,
				0u, 0u, step->multiprocessors, stream);
		return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), source, dst_bf16, dst_f32, rows, input_dimension, output_dimension,
			step->commit == 0u ? 1u : 0u, step->multiprocessors, stream);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		return LinearOut(weight, source, destination, (float *)0, rows, input_dimension, output_dimension, step, stream);
	}

	static int32_t LinearFloat(const uint16_t *weight, const uint16_t *source, float *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		return LinearOut(weight, source, (uint16_t *)0, destination, rows, input_dimension, output_dimension, step, stream);
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream, Frame *frame)
	{
		if ( state->kv_attached == 0u || step->rows == 0u || step->rows > state->max_rows || step->sequences == 0u || step->sequences > state->slots )
		{
			fprintf(stderr, "%s: step refused rows=%u sequences=%u kv_attached=%u\n", G::Tag(), step->rows, step->sequences, state->kv_attached);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		const SparkStatus status = CudaStageCheckTreeBegin(step);
		if ( status != SPARK_STATUS_OK )
			return status;
		frame->step = step;
		frame->stream = (SparkGpuStream)stream;
		frame->layer = 0;
		frame->index = 0u;
		return SPARK_STATUS_OK;
	}

	static SparkStatus HyperBegin(State *state, Frame &frame)
	{
		(void)LM_LAUNCH_DEPENDENT((LmHcExpandStreamsKernel<MO_HIDDEN>), dim3((MO_HIDDEN + MO_THREADS - 1u) / MO_THREADS, frame.step->rows), MO_THREADS, 0, frame.stream,
			frame.step->hidden_bf16, state->streams, frame.step->rows);
		return CudaStageLaunched();
	}

	static void BindLayer(State *state, Frame &frame, uint32_t layer)
	{
		frame.layer = &state->layers[layer];
		frame.index = layer;
	}

	static SparkStatus HyperNorm(State *state, Frame &frame, uint32_t layer, uint32_t part)
	{
		const MotifLayer *w = frame.layer;
		const uint32_t rows = frame.step->rows;
		const bool attention = part == MOTIF_ATTENTION_PART;
		const float *fn = attention ? w->hc_attn_fn : w->hc_ffn_fn;
		const float *scale = attention ? w->hc_attn_scale : w->hc_ffn_scale;
		const float *base = attention ? w->hc_attn_base : w->hc_ffn_base;
		const float *nw = attention ? w->attn_norm : w->ffn_norm;
		float *partials = (float *)state->snapshot;
		const uint32_t row_tiles = rows >= 512u ? 8u : (rows >= 128u ? 4u : (rows >= 32u ? 2u : 1u));
		if ( rows <= 4u )
			(void)LM_LAUNCH_DEPENDENT((LmHcSiteMultiDotKernel<MO_HIDDEN, 6u, false>), dim3(LM_HC_SITE_CLUSTER, 6u, 1u), MO_THREADS, 0, frame.stream,
				state->streams, (const uint16_t *)0, (const float *)0, (const float *)0, (uint16_t *)0, fn, partials, rows);
		else
			(void)LM_LAUNCH_DEPENDENT((LmHcSiteMultiDotKernel<MO_HIDDEN, 3u, false>), dim3(LM_HC_SITE_CLUSTER, 3u, row_tiles), MO_THREADS, 0, frame.stream,
				state->streams, (const uint16_t *)0, (const float *)0, (const float *)0, (uint16_t *)0, fn, partials, rows);
		if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
			return Check(LM_LAUNCH_ERR_LAUNCH, "hc dot", layer);
		(void)LM_LAUNCH_DEPENDENT((MotifHcFinishNormKernel), rows, MO_THREADS, 0, frame.stream,
			partials, state->streams, scale, base, state->hc_post, state->hc_comb, nw, state->normed, rows);
		return Check(LM_LAUNCH_OK, "hc finish norm", layer);
	}

	static SparkGpuError Attend(State *state, const SparkStageRunnerStep *step, uint32_t layer, bool window, uint64_t query_stride,
		uint64_t record_stride, SparkGpuStream stream)
	{
		const uint32_t rows = step->rows, degree = state->tp_degree, total_heads = degree * kHeads;
		const float scale = window ? SPARK_MOTIF_MODEL_WINDOW_SOFTMAX_SCALE : SPARK_MOTIF_MODEL_FULL_SOFTMAX_SCALE;
		const uint32_t *selected = window ? state->windows : (const uint32_t *)0;
		const uint32_t selected_count = window ? MO_WINDOW : 0u;
		if ( rows > LM_SKINNY_ROWS_WIDE && step->commit != 0u )
		{
			if ( !window && step->sequences == 1u && (total_heads % (LM_LATENT_SHARD_QUERIES / 2u)) == 0u )
				return LmLatentShardPrefillRowsLaunch<MotifKv, LmKvShardView, MO_LATENT, MO_ROPE, 2u>(state->views[layer], state->gathered,
					query_stride, kHeads, total_heads, step->sequence_of_row, step->context_length, step->positions, scale, state->partials,
					record_stride, rows, 1u, 0u, stream, LmLatentShardDense(kHeads));
			if ( (total_heads % LM_LATENT_SHARD_QUERIES) == 0u || total_heads < LM_LATENT_SHARD_QUERIES )
				return LmLatentShardAttendLaunch<MotifKv, LmKvShardView, MO_LATENT, MO_ROPE>(state->views[layer], state->gathered, query_stride,
					kHeads, step->sequence_of_row, step->context_length, step->positions, selected, selected_count, selected_count, scale,
					state->partials, record_stride, rows, stream);
			const uint32_t max_ranks = (LM_LATENT_SHARD_QUERIES - 1u) / kHeads;
			if ( max_ranks != 0u )
			{
				for ( uint32_t rank_first = 0u; rank_first < degree; rank_first += max_ranks )
				{
					const uint32_t chunk = degree - rank_first < max_ranks ? degree - rank_first : max_ranks;
					const SparkGpuError error = LmLatentShardAttendHeadsLaunch<MotifKv, LmKvShardView, MO_LATENT, MO_ROPE>(state->views[layer],
						state->gathered + (uint64_t)rank_first * query_stride, query_stride, kHeads, chunk * kHeads, step->sequence_of_row,
						step->context_length, step->positions, selected, selected_count, selected_count, scale,
						state->partials + (uint64_t)rank_first * record_stride, record_stride, kHeads, 1u, rows, stream);
					if ( error != SPARK_GPU_SUCCESS )
						return error;
				}
				return SPARK_GPU_SUCCESS;
			}
		}
		return LmLatentShardPartialLaunch<MotifKv, LmKvShardView, MO_LATENT, MO_ROPE>(state->views[layer], state->gathered, query_stride, kHeads,
			step->sequence_of_row, step->context_length, step->positions, selected, selected_count, selected_count, scale, state->partials,
			record_stride, rows, stream, step->tree != 0u ? step->fork_positions : (const uint32_t *)0,
			step->tree != 0u ? step->slot_offsets : (const uint32_t *)0);
	}

	static SparkStatus Attention(State *state, Frame &frame, uint32_t layer, bool window)
	{
		const MotifLayer *w = frame.layer;
		const SparkStageRunnerStep *step = frame.step;
		const uint32_t rows = step->rows, degree = state->tp_degree;
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		SparkGpuStream stream = frame.stream;
		const float *inv_freq = window ? (const float *)0 : state->full_inv_freq;
		const float theta = window ? SPARK_MOTIF_MODEL_WINDOW_ROPE_THETA : SPARK_MOTIF_MODEL_ROPE_THETA;
		const uint64_t query_stride = (uint64_t)QuerySequences(rows) * MO_HIDDEN, record_stride = (uint64_t)RecordSequences(rows) * MO_HIDDEN / 2u;
		SparkStatus status = SPARK_STATUS_OK;
		CudaStageRowState(state, layer, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		bool fused_in = false;
		if ( !prefill_unified && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
		{
			const LmSkinnyDenseTarget targets[3] =
			{
				{ w->q_a, state->q_c, (float *)0, MO_QA, MO_QA, 0u, LmScaleTensorNone() },
				{ w->lambda, state->lambda, (float *)0, MO_GROUPS * MO_SIGNAL, MO_GROUPS * MO_SIGNAL, 0u, LmScaleTensorNone() },
				{ w->kv_a, state->slot, (float *)0, MO_SLOT, MO_SLOT, 0u, LmScaleTensorNone() }
			};
			if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 3u, state->normed, rows, MO_HIDDEN, stream, exact, 0u) == LM_LAUNCH_OK )
				fused_in = true;
		}
		if ( !fused_in )
		{
			status = Check(Linear(w->q_a, state->normed, state->q_c, rows, MO_HIDDEN, MO_QA, step, stream), "q_a", layer);
			if ( status == SPARK_STATUS_OK )
				status = Check(Linear(w->lambda, state->normed, state->lambda, rows, MO_HIDDEN, MO_GROUPS * MO_SIGNAL, step, stream), "lambda", layer);
			if ( status == SPARK_STATUS_OK )
				status = Check(Linear(w->kv_a, state->normed, state->slot, rows, MO_HIDDEN, MO_SLOT, step, stream), "kv_a", layer);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		(void)LM_LAUNCH_DEPENDENT((MotifRmsNormVecKernel<128u, MO_QA, false>), rows, 128u, 0, stream, state->q_c, w->q_norm, rows, MO_QA);
		status = Check(LM_LAUNCH_OK, "q norm", layer);
		if ( status != SPARK_STATUS_OK )
			return status;
		bool fused_q = false;
		if ( !prefill_unified && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
		{
			const LmSkinnyDenseTarget targets[2] =
			{
				{ w->q_b, state->q, (float *)0, kHeads * MO_HEAD, kHeads * MO_HEAD, 0u, LmScaleTensorNone() },
				{ w->gate, state->gate, (float *)0, kSignal * MO_VALUE, kSignal * MO_VALUE, 0u, LmScaleTensorNone() }
			};
			if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 2u, state->q_c, rows, MO_QA, stream, exact, 0u) == LM_LAUNCH_OK )
				fused_q = true;
		}
		if ( !fused_q )
		{
			status = Check(Linear(w->q_b, state->q_c, state->q, rows, MO_QA, kHeads * MO_HEAD, step, stream), "q_b", layer);
			if ( status == SPARK_STATUS_OK )
				status = Check(Linear(w->gate, state->q_c, state->gate, rows, MO_QA, kSignal * MO_VALUE, step, stream), "gate", layer);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		(void)LM_LAUNCH_DEPENDENT((MotifKvNormRopeStoreKernel), rows, 128u, 0, stream, state->slot, w->kv_norm, step->positions,
			step->tree != 0u ? step->slots : step->positions, step->sequence_of_row, inv_freq, theta, state->views[layer], rows);
		if ( LmPerHeadProjectChainLaunch<MO_THREADS, MO_NOPE, MO_LATENT, MO_HEAD, 0u, MO_SLOT>(state->q, w->uk, state->query, kHeads, rows, stream,
				exact, prefill_unified ? 1u : 0u) != SPARK_GPU_SUCCESS )
			return Check(LM_LAUNCH_ERR_LAUNCH, "query absorb", layer);
		(void)LM_LAUNCH_DEPENDENT((MotifQueryRopeStoreKernel), dim3(rows, kHeads), 32u, 0, stream, state->q, step->positions, inv_freq, theta,
			state->query, kHeads, rows);
		status = Check(LM_LAUNCH_OK, "kv store", layer);
		CudaStageRowState(state, layer, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( status != SPARK_STATUS_OK )
			return status;
		const uint32_t total_ctx = step->gather_context;
		const uint32_t old_ctx = total_ctx > rows ? total_ctx - rows : 0u;
		const uint32_t win_len = window ? MO_WINDOW : 0u;
		const uint32_t raw_win_begin = (win_len != 0u && old_ctx + 1u > win_len) ? (old_ctx + 1u - win_len) : 0u;
		const uint32_t win_begin = raw_win_begin & ~(MO_PREFILL_KV_TILE - 1u);
		const uint32_t prior_keys = old_ctx > win_begin ? old_ctx - win_begin : 0u;
		const uint32_t grain = state->views[layer].shard.grain;
		const uint32_t b0 = win_begin / grain;
		const uint32_t b1 = (old_ctx + grain - 1u) / grain;
		const uint32_t max_peer_keys = prior_keys != 0u ? (((b1 - b0 + degree - 1u) / degree) * grain) : 0u;
		const uint64_t slot_bytes = (uint64_t)MO_SLOT * sizeof(uint16_t);
		const uint64_t unit_bytes = (uint64_t)MO_HIDDEN * sizeof(uint16_t);
		const uint32_t ag_units = (degree > 1u && prior_keys != 0u) ? (uint32_t)(((uint64_t)max_peer_keys * slot_bytes + unit_bytes - 1u) / unit_bytes) : 0u;
		const uint64_t ag_stride_bytes = (uint64_t)ag_units * unit_bytes;
		const uint64_t prior_bytes = (uint64_t)prior_keys * slot_bytes;
		const uint64_t buf_cap_bytes = (uint64_t)degree * RecordSequences(state->max_rows) * unit_bytes;
		const bool use_local_prefill = prefill_unified && total_ctx >= rows &&
			(state->collective_capacity == 0u || ag_units <= state->collective_capacity) &&
			(uint64_t)degree * ag_stride_bytes <= buf_cap_bytes && prior_bytes <= buf_cap_bytes;
		if ( use_local_prefill )
		{
			uint16_t *kv_prior = (uint16_t *)state->partials;
			if ( prior_keys != 0u )
			{
				if ( degree > 1u )
				{
					(void)LM_LAUNCH_DEPENDENT((MotifPriorKvPackKernel), max_peer_keys, 128u, 0, stream,
						state->views[layer], step->sequence_of_row, win_begin, old_ctx, b0, max_peer_keys, kv_prior);
					if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
						return Check(LM_LAUNCH_ERR_LAUNCH, "prior kv pack", layer);
					status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, ag_units, 0u, state->partials, state->received);
					if ( status != SPARK_STATUS_OK )
						return status;
					(void)LM_LAUNCH_DEPENDENT((MotifPriorKvUnpackKernel), dim3(max_peer_keys, degree), 128u, 0, stream,
						state->views[layer].shard, (const uint8_t *)state->received, ag_stride_bytes, win_begin, old_ctx, b0, max_peer_keys, kv_prior);
					if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
						return Check(LM_LAUNCH_ERR_LAUNCH, "prior kv unpack", layer);
				}
				else
				{
					(void)LM_LAUNCH_DEPENDENT((MotifPriorKvDirectKernel), prior_keys, 128u, 0, stream,
						state->views[layer], step->sequence_of_row, win_begin, prior_keys, kv_prior);
					if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
						return Check(LM_LAUNCH_ERR_LAUNCH, "prior kv direct", layer);
				}
			}
			constexpr uint32_t shared_bytes = MotifLocalPrefillSharedBytes();
			if constexpr ( shared_bytes > 48000u )
			{
				static CudaStageSmemGrant lp_grant;
				if ( !lp_grant.Grant((const void *)MotifMlaLocalPrefillKernel, shared_bytes) )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			const float scale = window ? SPARK_MOTIF_MODEL_WINDOW_SOFTMAX_SCALE : SPARK_MOTIF_MODEL_FULL_SOFTMAX_SCALE;
			(void)LM_LAUNCH_DEPENDENT((MotifMlaLocalPrefillKernel), dim3(kHeads, (rows + MO_PREFILL_Q_TILE - 1u) / MO_PREFILL_Q_TILE),
				MO_THREADS, shared_bytes, stream, state->query, kv_prior, state->slot, step->positions, rows, kHeads, win_begin, prior_keys,
				win_len, scale, state->merged);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				return Check(LM_LAUNCH_ERR_LAUNCH, "local prefill attn", layer);
		}
		else
		{
			if ( window )
				(void)LM_LAUNCH_DEPENDENT((MotifWindowListKernel), rows, MO_THREADS, 0, stream, step->positions, state->windows, rows);
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, QuerySequences(rows), 0u, state->query, state->gathered);
			if ( status != SPARK_STATUS_OK )
				return status;
			if ( Attend(state, step, layer, window, query_stride, record_stride, stream) != SPARK_GPU_SUCCESS )
				return Check(LM_LAUNCH_ERR_LAUNCH, "attention partial", layer);
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, RecordSequences(rows), 0u, state->partials, state->received);
			if ( status != SPARK_STATUS_OK )
				return status;
			if ( LmLatentShardMergeLaunch<MO_LATENT>(state->received, record_stride, degree, kHeads, state->merged, rows, stream) != SPARK_GPU_SUCCESS )
				return Check(LM_LAUNCH_ERR_LAUNCH, "attention merge", layer);
		}
		if ( LmPerHeadProjectChainLaunch<MO_THREADS, MO_LATENT, MO_VALUE>(state->merged, w->uv, state->values, kHeads, rows, stream,
				exact, prefill_unified ? 1u : 0u) != SPARK_GPU_SUCCESS ||
			LmDifferentialGateLaunch(state->values, state->lambda + (uint64_t)state->tp_rank * kSignal, MO_GROUPS * MO_SIGNAL, state->gate, state->diff,
				rows, kGroups, MO_SIGNAL, MO_VALUE, stream) != SPARK_GPU_SUCCESS )
			return Check(LM_LAUNCH_ERR_LAUNCH, "attention out prep", layer);
		return Check(Linear(w->out, state->diff, state->partial, rows, kSignal * MO_VALUE, MO_HIDDEN, step, stream), "attention out", layer);
	}

	static SparkStatus FullAttention(State *state, Frame &frame, uint32_t layer) { return Attention(state, frame, layer, false); }
	static SparkStatus WindowAttention(State *state, Frame &frame, uint32_t layer) { return Attention(state, frame, layer, true); }
	static SparkStatus AllReduce(State *state, Frame &frame) { return CudaStageAllReduceSum(state, frame.stream, state->partial, frame.step->rows); }

	static SparkStatus KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream_void)
	{
		State *state = (State *)model;
		const SparkStatus status = CudaStageKeepRowsCheck(state, step, kept);
		if ( status != SPARK_STATUS_OK || step->tree == 0u )
			return status;
		CudaStageKeepMoveViews<MotifKv, MO_THREADS>(state, state->views, MO_LAYERS, MO_SLOT, step->rows, step, kept, (SparkGpuStream)stream_void);
		return CudaStageLaunched();
	}

	static SparkStatus HyperPost(State *state, Frame &frame, uint32_t layer, uint32_t part)
	{
		(void)part;
		(void)LM_LAUNCH_DEPENDENT((MotifHcPostInPlaceKernel), frame.step->rows, MO_THREADS, 0, frame.stream,
			state->partial, state->hc_post, state->hc_comb, state->streams, frame.step->rows);
		return Check(LM_LAUNCH_OK, "hc post", layer);
	}

	static SparkStatus OwnedMlp(State *state, Frame &frame, uint32_t layer, const uint16_t *gate_up, const uint16_t *down, const float *poly,
		uint32_t inter, uint16_t *output)
	{
		const SparkStageRunnerStep *step = frame.step;
		const uint32_t rows = step->rows;
		SparkStatus status = Check(Linear(gate_up, state->normed, state->mlp_gate_up, rows, MO_HIDDEN, 2u * inter, step, frame.stream), "mlp gate_up", layer);
		if ( status == SPARK_STATUS_OK && LmPolyNormMulLaunch<false>(state->mlp_gate_up, poly, (const uint32_t *)0, state->mlp_act, rows, inter,
				SPARK_MOTIF_MODEL_HIDDEN_CLAMP, SPARK_MOTIF_MODEL_POLYNORM_ROUTED_BIAS_CLAMP, SPARK_MOTIF_MODEL_POLYNORM_OUTPUT_SCALE,
				SPARK_MOTIF_MODEL_POLYNORM_EPSILON, frame.stream) != SPARK_GPU_SUCCESS )
			status = Check(LM_LAUNCH_ERR_LAUNCH, "mlp polynorm", layer);
		if ( status == SPARK_STATUS_OK )
			status = Check(Linear(down, state->mlp_act, output, rows, inter, MO_HIDDEN, step, frame.stream), "mlp down", layer);
		return status;
	}

	static SparkStatus DenseMlp(State *state, Frame &frame, uint32_t layer)
	{
		const MotifLayer *w = frame.layer;
		SparkStatus status = SPARK_STATUS_OK;
		if ( w->owner != 0u )
			status = OwnedMlp(state, frame, layer, w->ffn_gate_up, w->ffn_down, w->ffn_poly, MO_DENSE_INTER, state->partial);
		else if ( SparkGpuMemsetAsync(state->partial, 0, (uint64_t)frame.step->rows * MO_HIDDEN * sizeof(uint16_t), frame.stream) != SPARK_GPU_SUCCESS )
			status = Check(LM_LAUNCH_ERR_LAUNCH, "dense mlp clear", layer);
		if ( state->services->layer_done != 0 )
			state->services->layer_done(state->services->context, layer);
		return status;
	}

	static SparkStatus RoutedMlp(State *state, Frame &frame, uint32_t layer)
	{
		const MotifLayer *w = frame.layer;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, sms = step->multiprocessors, routes = rows * MO_TOP_K;
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		const bool prefill_unified = PrefillUnified(step);
		SparkStatus status = Check(LinearFloat(w->router, state->normed, state->router_logits, rows, MO_HIDDEN, MO_EXPERTS, step, stream), "router", layer);
		if ( status == SPARK_STATUS_OK && LmTopkGroupLimitedLaunch<MO_THREADS, MO_EXPERTS, 1u, 1u, MO_TOP_K>(rows, state->router_logits, w->expert_bias,
				state->route_expert, state->route_weight, SPARK_MOTIF_MODEL_MOE_ROUTED_SCALE, step->route_fault, stream) != SPARK_GPU_SUCCESS )
			status = Check(LM_LAUNCH_ERR_LAUNCH, "router top-k", layer);
		if ( status != SPARK_STATUS_OK )
			return status;
		const uint32_t tile_m = LmLaunchGroupedTileM(rows, MO_TOP_K, kLocalExperts + 1u);
		const uint32_t n_up = (2u * MO_EXPERT_INTER + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
		const uint32_t n_down = (MO_HIDDEN + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
		(void)LM_LAUNCH_DEPENDENT((MotifRoutePrepareKernel<kLocalExperts>), 1u, MO_THREADS, 0, stream,
			state->route_expert, state->route_weight, state->route_global, routes, state->expert_first,
			state->group_row_offset, state->route_packed_row, state->route_source_token, state->param_of_row,
			tile_m, n_up, state->tile_prefix_up, n_down, state->tile_prefix_down);
		if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
			return Check(LM_LAUNCH_ERR_LAUNCH, "route prepare", layer);
		const LmLocalExpertRoutes expert_routes = CudaStageBuildExpertRoutes<LmLocalExpertRoutes>(state, rows, MO_TOP_K, kLocalExperts, sms);
		if ( prefill_unified )
		{
			status = Check(LmStreamGemmGrouped<LmBf16Format>(w->moe_w13, LmScaleTensorNone(), state->normed, state->expert_gate_up,
				state->group_row_offset, state->route_source_token, kLocalExperts, routes, MO_HIDDEN, 2u * MO_EXPERT_INTER, sms, stream), "experts up", layer);
		}
		else
		{
			status = Check(LmLocalExpertsUp<LmBf16Format>(expert_routes, w->moe_w13, LmScaleTensorNone(), state->normed, state->expert_gate_up, MO_HIDDEN,
				2u * MO_EXPERT_INTER, stream, exact), "experts up", layer);
		}
		if ( status == SPARK_STATUS_OK )
		{
			(void)LM_LAUNCH_DEPENDENT((MotifPolyNormActiveKernel), routes, MO_THREADS, 0, stream,
				state->expert_gate_up, w->moe_poly, state->param_of_row, state->group_row_offset, kLocalExperts, state->expert_act, MO_EXPERT_INTER);
			status = Check(LM_LAUNCH_OK, "experts polynorm", layer);
		}
		if ( status == SPARK_STATUS_OK )
		{
			if ( prefill_unified )
				status = Check(LmStreamGemmGrouped<LmBf16Format>(w->moe_w2, LmScaleTensorNone(), state->expert_act, state->expert_out,
					state->group_row_offset, (const uint32_t *)0, kLocalExperts, routes, MO_EXPERT_INTER, MO_HIDDEN, sms, stream), "experts down", layer);
			else
				status = Check(LmLocalExpertsDown<LmBf16Format>(expert_routes, w->moe_w2, LmScaleTensorNone(), state->expert_act, state->expert_out,
					MO_EXPERT_INTER, MO_HIDDEN, stream, exact), "experts down", layer);
		}
		if ( status == SPARK_STATUS_OK && w->owner != 0u )
			status = OwnedMlp(state, frame, layer, w->shared_gate_up, w->shared_down, w->shared_poly, MO_EXPERT_INTER, state->shared_out);
		if ( status == SPARK_STATUS_OK )
		{
			(void)LM_LAUNCH_DEPENDENT((LmMoeLocalWeightedFinalizeKernel), dim3(LmVecBlocks<MO_THREADS>(MO_HIDDEN), rows), MO_THREADS, 0, stream,
				state->expert_out, state->route_packed_row, state->route_expert, state->route_weight, w->owner != 0u ? state->shared_out : (const uint16_t *)0,
				state->partial, MO_TOP_K, kLocalExperts, MO_HIDDEN);
			status = Check(LM_LAUNCH_OK, "experts finalize", layer);
		}
		if ( state->services->layer_done != 0 )
			state->services->layer_done(state->services->context, layer);
		return status;
	}

	static SparkStatus HyperEnd(State *state, Frame &frame)
	{
		const uint32_t rows = frame.step->rows;
		(void)LM_LAUNCH_DEPENDENT((LmHcStreamsMeanKernel<MO_HIDDEN>), rows, MO_THREADS, 0, frame.stream, state->streams, frame.step->hidden_bf16, rows);
		if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		state->steps++;
		if ( rows > frame.step->sequences )
			state->prefill_rows += rows;
		return SPARK_STATUS_OK;
	}

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, routes = rows * MO_TOP_K, groups = kLocalExperts + 2u, degree = state->tp_degree, b = sizeof(uint16_t);
		const CudaStageSlice plan[] =
		{
			{ (void **)&state->streams, rows * LM_HC_SITE_STREAMS * MO_HIDDEN * b },
			{ (void **)&state->snapshot, rows * LM_HC_SITE_STREAMS * MO_HIDDEN * b },
			{ (void **)&state->collapsed, rows * MO_HIDDEN * b },
			{ (void **)&state->normed, rows * MO_HIDDEN * b },
			{ (void **)&state->hc_mixes, rows * LM_HC_SITE_MIX * sizeof(float) },
			{ (void **)&state->hc_pre, rows * LM_HC_SITE_STREAMS * sizeof(float) },
			{ (void **)&state->hc_post, rows * LM_HC_SITE_STREAMS * sizeof(float) },
			{ (void **)&state->hc_comb, rows * LM_HC_SITE_STREAMS * LM_HC_SITE_STREAMS * sizeof(float) },
			{ (void **)&state->q_c, rows * MO_QA * b },
			{ (void **)&state->q, rows * kHeads * MO_HEAD * b },
			{ (void **)&state->q_latent, rows * kHeads * MO_LATENT * b },
			{ (void **)&state->query, (uint64_t)QuerySequences(state->max_rows) * MO_HIDDEN * b },
			{ (void **)&state->gathered, degree * QuerySequences(state->max_rows) * MO_HIDDEN * b },
			{ (void **)&state->slot, rows * MO_SLOT * b },
			{ (void **)&state->windows, rows * MO_WINDOW * sizeof(uint32_t) },
			{ (void **)&state->partials, degree * RecordSequences(state->max_rows) * MO_HIDDEN * b },
			{ (void **)&state->received, degree * RecordSequences(state->max_rows) * MO_HIDDEN * b },
			{ (void **)&state->merged, rows * kHeads * MO_LATENT * b },
			{ (void **)&state->values, rows * kHeads * MO_VALUE * b },
			{ (void **)&state->lambda, rows * MO_GROUPS * MO_SIGNAL * b },
			{ (void **)&state->gate, rows * kSignal * MO_VALUE * b },
			{ (void **)&state->diff, rows * kSignal * MO_VALUE * b },
			{ (void **)&state->partial, rows * MO_HIDDEN * b },
			{ (void **)&state->router_logits, rows * MO_EXPERTS * sizeof(float) },
			{ (void **)&state->route_expert, routes * sizeof(uint32_t) },
			{ (void **)&state->route_global, routes * sizeof(uint32_t) },
			{ (void **)&state->route_weight, routes * sizeof(float) },
			{ (void **)&state->group_row_offset, groups * sizeof(uint32_t) },
			{ (void **)&state->route_packed_row, routes * sizeof(uint32_t) },
			{ (void **)&state->route_source_token, routes * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_up, groups * sizeof(uint32_t) },
			{ (void **)&state->tile_prefix_down, groups * sizeof(uint32_t) },
			{ (void **)&state->param_of_row, routes * sizeof(uint32_t) },
			{ (void **)&state->expert_gate_up, routes * 2u * MO_EXPERT_INTER * b },
			{ (void **)&state->expert_act, routes * MO_EXPERT_INTER * b },
			{ (void **)&state->expert_out, routes * MO_HIDDEN * b },
			{ (void **)&state->mlp_gate_up, rows * 2u * MO_DENSE_INTER * b },
			{ (void **)&state->mlp_act, rows * MO_DENSE_INTER * b },
			{ (void **)&state->shared_out, rows * MO_HIDDEN * b },
			{ (void **)&state->full_inv_freq, lane == 0u ? (MO_ROPE / 2u) * sizeof(float) : 0u },
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		return CudaStagePlace(plan, base);
	}

	static uint32_t Accepts(const SparkStageRunnerConfiguration *configuration) { return configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_BF16 ? 1u : 0u; }

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		const uint32_t expected_first = configuration->tp_rank * kLocalExperts;
		const CudaStagePackCheck checks[] =
		{
			{ "tp_degree", configuration->tp_degree }, { "tp_rank", configuration->tp_rank }, { "hidden", MO_HIDDEN }, { "layers", MO_LAYERS },
			{ "vocab_rows", MO_VOCAB / configuration->tp_degree }, { "head_groups", kGroups }, { "head_group_first", configuration->tp_rank * kGroups },
			{ "experts", MO_EXPERTS }, { "local_experts", kLocalExperts }, { "expert_first", expected_first }
		};
		const SparkStatus status = CudaStagePackExpect(G::Tag(), &state->pack, checks);
		if ( status != SPARK_STATUS_OK )
			return status;
		state->expert_first = expected_first;
		state->collective_capacity = configuration->device_collective != 0 ? configuration->device_collective->max_active_sequence_count : 0u;
		if ( state->max_rows > state->collective_capacity || QuerySequences(state->max_rows) > state->collective_capacity ||
			RecordSequences(state->max_rows) > state->collective_capacity )
		{
			fprintf(stderr, "%s: %u rows need collectives of %u query and %u record sequences; the collective carries %u\n", G::Tag(), state->max_rows,
				QuerySequences(state->max_rows), RecordSequences(state->max_rows), state->collective_capacity);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		geometry->tree_verify = 1u;
		geometry->head_norm_round_first = 1u;
		return CudaStage<MotifStage>::HeadCheck(state, geometry, false, false, false);
	}

	template<class T>
	static void Tensor(CudaStageTensors *tensors, const char *name, const char *kind, T **field, uint64_t bytes) { tensors->Add(name, kind, bytes, field); }

	static SparkStatus LayerTensors(State *state, uint32_t index, CudaStageTensors *t)
	{
		const uint64_t h = MO_HIDDEN, b = sizeof(uint16_t), f = sizeof(float), mix = LM_HC_SITE_MIX * LM_HC_SITE_STREAMS * h;
		MotifLayer *w = &state->layers[index];
		w->window = MotifLayout::Window(index) ? 1u : 0u;
		w->owner = index % state->tp_degree == state->tp_rank ? 1u : 0u;
		Tensor(t, "attn_norm", "f32", &w->attn_norm, h * f);
		Tensor(t, "ffn_norm", "f32", &w->ffn_norm, h * f);
		Tensor(t, "hc_attn_fn", "f32", &w->hc_attn_fn, mix * f);
		Tensor(t, "hc_attn_scale", "f32", &w->hc_attn_scale, 3u * f);
		Tensor(t, "hc_attn_base", "f32", &w->hc_attn_base, LM_HC_SITE_MIX * f);
		Tensor(t, "hc_ffn_fn", "f32", &w->hc_ffn_fn, mix * f);
		Tensor(t, "hc_ffn_scale", "f32", &w->hc_ffn_scale, 3u * f);
		Tensor(t, "hc_ffn_base", "f32", &w->hc_ffn_base, LM_HC_SITE_MIX * f);
		Tensor(t, "attn_q_a", "bf16", &w->q_a, (uint64_t)MO_QA * h * b);
		Tensor(t, "attn_q_norm", "f32", &w->q_norm, (uint64_t)MO_QA * f);
		Tensor(t, "attn_kv_a", "bf16", &w->kv_a, (uint64_t)MO_SLOT * h * b);
		Tensor(t, "attn_kv_norm", "f32", &w->kv_norm, (uint64_t)MO_LATENT * f);
		Tensor(t, "attn_lambda", "bf16", &w->lambda, (uint64_t)MO_GROUPS * MO_SIGNAL * h * b);
		Tensor(t, "attn_q_b", "bf16", &w->q_b, (uint64_t)kHeads * MO_HEAD * MO_QA * b);
		Tensor(t, "attn_gate", "bf16", &w->gate, (uint64_t)kSignal * MO_VALUE * MO_QA * b);
		Tensor(t, "attn_out", "bf16", &w->out, h * kSignal * MO_VALUE * b);
		Tensor(t, "attn_uk", "bf16", &w->uk, (uint64_t)kHeads * MO_LATENT * MO_NOPE * b);
		Tensor(t, "attn_uv", "bf16", &w->uv, (uint64_t)kHeads * MO_VALUE * MO_LATENT * b);
		if ( !MotifLayout::Routed(index) && w->owner != 0u )
		{
			Tensor(t, "ffn_gate_up", "bf16", &w->ffn_gate_up, 2ull * MO_DENSE_INTER * h * b);
			Tensor(t, "ffn_down", "bf16", &w->ffn_down, h * MO_DENSE_INTER * b);
			Tensor(t, "ffn_poly", "f32", &w->ffn_poly, LM_POLY_NORM_PARAMS * f);
		}
		if ( MotifLayout::Routed(index) )
		{
			Tensor(t, "router_proj", "bf16", &w->router, (uint64_t)MO_EXPERTS * h * b);
			Tensor(t, "expert_bias", "f32", &w->expert_bias, (uint64_t)MO_EXPERTS * f);
			Tensor(t, "moe_w13", "bf16", &w->moe_w13, (uint64_t)kLocalExperts * 2u * MO_EXPERT_INTER * h * b);
			Tensor(t, "moe_w2", "bf16", &w->moe_w2, (uint64_t)kLocalExperts * h * MO_EXPERT_INTER * b);
			Tensor(t, "moe_poly", "f32", &w->moe_poly, (uint64_t)kLocalExperts * LM_POLY_NORM_PARAMS * f);
			if ( w->owner != 0u )
			{
				Tensor(t, "shared_gate_up", "bf16", &w->shared_gate_up, 2ull * MO_EXPERT_INTER * h * b);
				Tensor(t, "shared_down", "bf16", &w->shared_down, h * MO_EXPERT_INTER * b);
				Tensor(t, "shared_poly", "f32", &w->shared_poly, LM_POLY_NORM_PARAMS * f);
			}
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Upload(State *state)
	{
		float inv_freq[MO_ROPE / 2u];
		SparkRopeYarnRampTable(inv_freq, MO_ROPE, SPARK_MOTIF_MODEL_ROPE_THETA, SPARK_MOTIF_MODEL_YARN_FACTOR, SPARK_MOTIF_MODEL_YARN_BETA_FAST,
			SPARK_MOTIF_MODEL_YARN_BETA_SLOW, SPARK_MOTIF_MODEL_YARN_ORIGINAL_CONTEXT);
		if ( SparkGpuMemcpy(state->full_inv_freq, inv_freq, sizeof(inv_freq), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
	}

	static uint32_t ReportFields(const State *state, CudaStageReportField *fields)
	{
		const CudaStageReportField report[] =
		{
			{ "arena_bytes", state->arena_bytes }, { "lane_bytes", state->lane_bytes }, { "slots", state->slots }, { "rows", state->max_rows },
			{ "expert_first", state->expert_first }, { "steps", state->steps }, { "prefill_rows", state->prefill_rows }
		};
		return CudaStageReportOut(fields, report);
	}
};
