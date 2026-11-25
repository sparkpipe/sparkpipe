#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "runtime/launch.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "inference/runner/stage_model_cuda.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_group.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "runtime/gemm.cuh"
#include "sparkpipe/spark_rope_plan.h"
#include "sparkpipe/spark_gemma4_family.h"
#include "sparkpipe/spark_gpu.h"

#define GM_THREADS 256u

template<class G>
struct GemmaShape : GemmaLayout<G>
{
	using SlidingKvGeometry = LmKvHeads<16u, G::kSlidingKvHeads, G::kSlidingDim, G::kPageSlots>;
	using FullKvGeometry = LmKvHeads<16u, G::kFullKvHeads, G::kFullDim, G::kPageSlots>;
};

typedef struct GemmaLayer
{
	const uint16_t *qkv;
	const uint16_t *out;
	const uint16_t *gate_up;
	const uint16_t *down;
	const float *input_norm;
	const float *post_attn_norm;
	const float *pre_ff_norm;
	const float *post_ff_norm;
	const float *layer_scalar;
	const float *q_norm;
	const float *k_norm;
	const uint16_t *gate;
	const uint16_t *router_proj;
	const float *router_scale;
	const float *expert_scale;
	const uint16_t *moe_w1;
	const uint16_t *moe_w3;
	const uint16_t *moe_w2;
	const float *post_ff_norm_1;
	const float *pre_ff_norm_2;
	const float *post_ff_norm_2;
	uint32_t kv_index;
	const float *expert_bias;
	uint32_t ffn_rows;
	uint32_t gate_up_rows;
	uint32_t window;
	float swiglu_limit;
} GemmaLayer;

template<class G>
struct GemmaState : CudaStagePackState
{
	uint32_t ffn_rows;
	uint32_t expert_first;
	uint32_t router_fused_done;
	uint64_t record_stride;
	GemmaLayer layers[G::kLayers];
	uint16_t *normed;
	uint16_t *partial;
	uint16_t *attn_local;
	uint16_t *attn_gathered;
	uint16_t *attn_full;
	uint16_t *query;
	uint16_t *key;
	uint16_t *value;
	uint16_t *slot;
	float *send;
	float *received;
	uint16_t *merged;
	uint16_t *gate_logits;
	uint16_t *gate_up;
	uint16_t *intermediate;
	float *full_inv_freq;
	uint16_t *moe_normed;
	uint16_t *router_logits;
	float *router_probs;
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
	LmKvShardView sliding_views[GemmaShape<G>::kSlidingLayers];
	LmKvShardView full_views[GemmaShape<G>::kFullLayers];
};

static __device__ __forceinline__ float GemmaWarpSum(float value)
{
	#pragma unroll
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		value += __shfl_down_sync(0xffffffffu, value, offset);
	return __shfl_sync(0xffffffffu, value, 0u);
}

static __device__ __forceinline__ float GemmaHalfWarpSum(float value)
{
	#pragma unroll
	for ( uint32_t offset = 8u; offset > 0u; offset >>= 1u )
		value += __shfl_down_sync(0xffffffffu, value, offset);
	return value;
}

static __device__ __forceinline__ uint4 GemmaLoadRankMajor8(const uint4 *__restrict__ gathered4, uint32_t rows, uint32_t row, uint32_t local_v,
	uint32_t logical_v)
{
	const uint32_t rank = logical_v / local_v;
	const uint32_t elem = logical_v % local_v;
	return gathered4[(((uint64_t)rank * rows) + row) * local_v + elem];
}

template<uint32_t HD, bool WEIGHTED, bool ROUND_FIRST>
static __device__ __forceinline__ void GemmaHeadNormVec(const uint4 *in_v, const float *__restrict__ weight, uint4 *out_v, uint32_t lane,
	float epsilon, float head_multiply)
{
	constexpr uint32_t HD_V = HD >> 3u;
	constexpr uint32_t SLOTS = (HD_V + 31u) / 32u;
	float total = 0.0f;
	if constexpr ( SLOTS == 1u )
	{
		const float sq = lane < HD_V ? LmSumSqBf16x8(in_v[0], 0.0f) : 0.0f;
		total = GemmaWarpSum(sq);
	}
	else
	{
		#pragma unroll
		for ( uint32_t s = 0u; s < SLOTS; ++s )
		{
			const uint32_t c = s * 32u + lane;
			const float sq = c < HD_V ? LmSumSqBf16x8(in_v[s], 0.0f) : 0.0f;
			total += GemmaWarpSum(sq);
		}
	}
	const float scale = SparkMathRmsScale(total, (float)HD, epsilon);
	#pragma unroll
	for ( uint32_t s = 0u; s < SLOTS; ++s )
	{
		const uint32_t c = s * 32u + lane;
		if ( c < HD_V )
		{
			float x[8], w[8], out[8];
			LmLoadBf16x8AsFloat(in_v[s], x);
			if constexpr ( WEIGHTED )
				LmLoadWeight8(weight, c << 3u, w);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				float v = x[k] * scale;
				if constexpr ( WEIGHTED )
					v = ROUND_FIRST ? SparkMathNormGainRounded(v, w[k]) : v * w[k];
				out[k] = SparkMathRoundBf16(v) * head_multiply;
			}
			out_v[s] = LmPackUint4FloatToBf16(out);
		}
	}
}

template<class G, bool FULL>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaAttnPrepStoreKernel(const uint16_t *__restrict__ gathered, const float *__restrict__ q_norm, const float *__restrict__ k_norm,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ store_positions, const uint32_t *__restrict__ sequence_of_row,
	const float *__restrict__ full_inv_freq, LmKvShardView kv_view, uint16_t *__restrict__ query_out, uint32_t rows, uint32_t local_rows)
{
	using S = GemmaShape<G>;
	using Kv = typename std::conditional<FULL, typename S::FullKvGeometry, typename S::SlidingKvGeometry>::type;
	constexpr uint32_t HD = FULL ? G::kFullDim : G::kSlidingDim;
	constexpr uint32_t KVH = FULL ? G::kFullKvHeads : G::kSlidingKvHeads;
	constexpr uint32_t HEADS = FULL ? G::kFullHeads : G::kSlidingHeads;
	constexpr uint32_t QW = FULL ? S::kFullQ : S::kSlidingQ;
	constexpr uint32_t KVW = FULL ? S::kFullKv : S::kSlidingKv;
	constexpr uint32_t HD_V = HD >> 3u;
	constexpr uint32_t Q_V = QW >> 3u;
	constexpr uint32_t KV_V = KVW >> 3u;
	constexpr uint32_t SLOTS = (HD_V + 31u) / 32u;
	constexpr uint32_t ROPE_W = FULL ? S::kFullRopeWidth : HD;
	constexpr uint32_t ROPE_V = ROPE_W >> 3u;
	constexpr uint32_t HALF = ROPE_W / 2u;
	constexpr bool DO_ROPE = !FULL || G::kFullRope;
	constexpr bool KEY_IS_VAL = FULL && G::kKeyIsValue;
	constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
	constexpr uint32_t TASKS = HEADS + 2u * KVH;
	__shared__ __align__(16) uint16_t rope_smem[WARPS][DO_ROPE ? ROPE_W : 1u];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t task = blockIdx.x * WARPS + warp, row = blockIdx.y;
	LmDependentWait();
	if ( task < TASKS && row < rows )
	{
		const uint4 *gathered4 = (const uint4 *)gathered;
		const uint32_t local_v = local_rows >> 3u;
		if ( task < HEADS )
		{
			const uint32_t h = task;
			uint4 in_v[SLOTS], out_v[SLOTS];
			#pragma unroll
			for ( uint32_t s = 0u; s < SLOTS; ++s )
			{
				const uint32_t c = s * 32u + lane;
				if ( c < HD_V )
					in_v[s] = GemmaLoadRankMajor8(gathered4, rows, row, local_v, h * HD_V + c);
			}
			GemmaHeadNormVec<HD, G::kQkNormWeighted, G::kNormRoundFirst>(in_v, q_norm, out_v, lane, G::kNormEpsilon, G::kQueryMultiply);
			if constexpr ( DO_ROPE )
			{
				#pragma unroll
				for ( uint32_t s = 0u; s < SLOTS; ++s )
				{
					const uint32_t c = s * 32u + lane;
					if ( c < ROPE_V )
						*(uint4 *)(rope_smem[warp] + (c << 3u)) = out_v[s];
				}
				__syncwarp();
				const float pos = (float)positions[row];
				for ( uint32_t pair = lane; pair < HALF; pair += 32u )
				{
					const float angle = pos * (FULL ? full_inv_freq[pair] : __powf(G::kSlidingTheta, -2.0f * (float)pair / (float)ROPE_W));
					LmRopeRotate<LM_ROPE_HALF_SPLIT, LM_ROPE_ROUNDED_TERMS>(rope_smem[warp], 0u, pair, HALF, angle, FULL ? G::kFullRopeScale : 1.0f);
				}
				__syncwarp();
				#pragma unroll
				for ( uint32_t s = 0u; s < SLOTS; ++s )
				{
					const uint32_t c = s * 32u + lane;
					if ( c < ROPE_V )
						out_v[s] = *(const uint4 *)(rope_smem[warp] + (c << 3u));
				}
			}
			uint4 *q4 = (uint4 *)query_out + ((uint64_t)row * HEADS + h) * HD_V;
			#pragma unroll
			for ( uint32_t s = 0u; s < SLOTS; ++s )
			{
				const uint32_t c = s * 32u + lane;
				if ( c < HD_V )
					q4[c] = out_v[s];
			}
		}
		else
		{
			const uint32_t store_pos = store_positions[row];
			if ( SparkKvShardOwns(kv_view.shard, store_pos) != 0u )
			{
				uint16_t *slot = (uint16_t *)LmKvShardSlotRequired<Kv>(kv_view, sequence_of_row[row], store_pos, row, LM_KV_ACCESS_WRITE);
				if ( slot != 0 )
				{
					if ( task < HEADS + KVH )
					{
						const uint32_t kh = task - HEADS;
						uint4 in_v[SLOTS], out_v[SLOTS];
						#pragma unroll
						for ( uint32_t s = 0u; s < SLOTS; ++s )
						{
							const uint32_t c = s * 32u + lane;
							if ( c < HD_V )
								in_v[s] = GemmaLoadRankMajor8(gathered4, rows, row, local_v, Q_V + kh * HD_V + c);
						}
						GemmaHeadNormVec<HD, G::kQkNormWeighted, G::kNormRoundFirst>(in_v, k_norm, out_v, lane, G::kNormEpsilon, 1.0f);
						if constexpr ( DO_ROPE )
						{
							#pragma unroll
							for ( uint32_t s = 0u; s < SLOTS; ++s )
							{
								const uint32_t c = s * 32u + lane;
								if ( c < ROPE_V )
									*(uint4 *)(rope_smem[warp] + (c << 3u)) = out_v[s];
							}
							__syncwarp();
							const float pos = (float)positions[row];
							for ( uint32_t pair = lane; pair < HALF; pair += 32u )
							{
								const float angle = pos * (FULL ? full_inv_freq[pair] : __powf(G::kSlidingTheta, -2.0f * (float)pair / (float)ROPE_W));
								LmRopeRotate<LM_ROPE_HALF_SPLIT, LM_ROPE_ROUNDED_TERMS>(rope_smem[warp], 0u, pair, HALF, angle, FULL ? G::kFullRopeScale : 1.0f);
							}
							__syncwarp();
							#pragma unroll
							for ( uint32_t s = 0u; s < SLOTS; ++s )
							{
								const uint32_t c = s * 32u + lane;
								if ( c < ROPE_V )
									out_v[s] = *(const uint4 *)(rope_smem[warp] + (c << 3u));
							}
						}
						uint4 *k4 = (uint4 *)slot + kh * HD_V;
						#pragma unroll
						for ( uint32_t s = 0u; s < SLOTS; ++s )
						{
							const uint32_t c = s * 32u + lane;
							if ( c < HD_V )
								k4[c] = out_v[s];
						}
					}
					else
					{
						const uint32_t vh = task - HEADS - KVH;
						const uint32_t src_base = (KEY_IS_VAL ? Q_V : Q_V + KV_V) + vh * HD_V;
						uint4 in_v[SLOTS], out_v[SLOTS];
						#pragma unroll
						for ( uint32_t s = 0u; s < SLOTS; ++s )
						{
							const uint32_t c = s * 32u + lane;
							if ( c < HD_V )
								in_v[s] = GemmaLoadRankMajor8(gathered4, rows, row, local_v, src_base + c);
						}
						if constexpr ( G::kValueNorm )
							GemmaHeadNormVec<HD, false, false>(in_v, (const float *)0, out_v, lane, G::kNormEpsilon, 1.0f);
						uint4 *v4 = (uint4 *)slot + KV_V + vh * HD_V;
						#pragma unroll
						for ( uint32_t s = 0u; s < SLOTS; ++s )
						{
							const uint32_t c = s * 32u + lane;
							if ( c < HD_V )
								v4[c] = G::kValueNorm ? out_v[s] : in_v[s];
						}
					}
				}
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t HIDDEN>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaSmallNBf16ProjKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ activation, uint16_t *__restrict__ output,
	uint32_t rows, uint32_t output_dim)
{
	constexpr uint32_t VECS = HIDDEN >> 3u;
	constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
	__shared__ __align__(16) uint4 s_x[VECS];
	const uint32_t row = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	if ( row < rows )
	{
		const uint4 *x4 = (const uint4 *)(activation + (uint64_t)row * HIDDEN);
		for ( uint32_t c = threadIdx.x; c < VECS; c += GM_THREADS )
			s_x[c] = x4[c];
		__syncthreads();
		for ( uint32_t n = warp; n < output_dim; n += WARPS )
		{
			const uint4 *w4 = (const uint4 *)(weight + (uint64_t)n * HIDDEN);
			float acc = 0.0f;
			for ( uint32_t c = lane; c < VECS; c += LM_WARP_LANES )
				acc = LmDotBf16x8(s_x[c], __ldg(&w4[c]), acc);
			#pragma unroll
			for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				acc += __shfl_down_sync(0xffffffffu, acc, offset);
			if ( lane == 0u )
				output[(uint64_t)row * output_dim + n] = LmFloatToBf16(acc);
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaOutputGateKernel(uint16_t *__restrict__ output_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t dimension)
{
	const uint64_t base = (uint64_t)blockIdx.x * dimension;
	LmDependentWait();
	LmGateUpRowBody<THREADS>(gate_bf16 + base, output_bf16 + base, output_bf16 + base, dimension, LmSigmoidGateOp<true>{});
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaHeadGateBroadcastKernel(uint16_t *__restrict__ heads_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t heads,
	uint32_t head_dimension, uint32_t gate_stride)
{
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	uint16_t *base = heads_bf16 + ((((uint64_t)row * heads) + head) * head_dimension);
	LmDependentWait();
	const float gate = SparkMathSoftplusGate(LmBf16ToFloat(gate_bf16[((uint64_t)row * gate_stride) + head]));
	for ( uint32_t index = threadIdx.x * 8u; index < head_dimension; index += THREADS * 8u )
	{
		float v[8];
		LmLoadBf16x8AsFloat(*(const uint4 *)(base + index), v);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k )
			v[k] *= gate;
		*(uint4 *)(base + index) = LmPackUint4FloatToBf16(v);
	}
	LmDependentRelease();
}

struct GemmaGeluTanhOp
{
	__device__ __forceinline__ float operator()(float g, float u) const
	{
		return SparkMathRoundBf16(SparkMathGeluTanh(g)) * u;
	}
};

template<uint32_t THREADS, bool ROUND_FIRST, bool HAS_ADD>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaAddRmsNormKernel(const uint16_t *__restrict__ input_bf16, uint16_t *__restrict__ residual_bf16, const float *__restrict__ weight,
	uint16_t *__restrict__ output_bf16, uint32_t width, float epsilon)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	const uint64_t base = (uint64_t)blockIdx.x * width;
	const uint16_t *in_row = input_bf16 + base;
	uint16_t *res_row = HAS_ADD ? residual_bf16 + base : (uint16_t *)0;
	uint16_t *out_row = output_bf16 + base;
	uint4 staged[4];
	float total = 0.0f;
	LmDependentWait();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < 4u; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			uint4 v = *(const uint4 *)(in_row + offset);
			if constexpr ( HAS_ADD )
			{
				float a[8], b[8];
				LmLoadBf16x8AsFloat(v, a);
				LmLoadBf16x8AsFloat(*(const uint4 *)(res_row + offset), b);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					a[k] += b[k];
				v = LmPackUint4FloatToBf16(a);
				*(uint4 *)(res_row + offset) = v;
			}
			staged[slot] = v;
			total = LmSumSqBf16x8(v, total);
		}
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)width, epsilon);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < 4u; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			float x[8], w[8], out[8];
			LmLoadBf16x8AsFloat(staged[slot], x);
			LmLoadWeight8(weight, offset, w);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out[k] = ROUND_FIRST ? SparkMathNormGainRounded(x[k] * scale, w[k]) : x[k] * scale * w[k];
			*(uint4 *)(out_row + offset) = LmPackUint4FloatToBf16(out);
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS, bool ROUND_FIRST, bool HAS_SCALAR, bool TRIPLE_OUT>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaPostNormAddNextKernel(uint16_t *__restrict__ residual, const uint16_t *__restrict__ partial, const float *__restrict__ post_weight,
	const float *__restrict__ scalar, const float *__restrict__ next_weight, uint16_t *__restrict__ next_out,
	const float *__restrict__ router_scale, uint16_t *__restrict__ router_out, const float *__restrict__ moe_weight,
	uint16_t *__restrict__ moe_out, uint32_t width, float post_eps, float norm_eps, float router_root)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	const uint64_t base = (uint64_t)blockIdx.x * width;
	const uint16_t *part_row = partial + base;
	uint16_t *res_row = residual + base;
	uint4 staged[4];
	float total = 0.0f;
	LmDependentWait();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < 4u; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			staged[slot] = *(const uint4 *)(part_row + offset);
			total = LmSumSqBf16x8(staged[slot], total);
		}
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)width, post_eps);
	const float mult = HAS_SCALAR ? scalar[0] : 1.0f;
	float total2 = 0.0f;
	#pragma unroll
	for ( uint32_t slot = 0u; slot < 4u; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			float p[8], w[8], r[8], out[8];
			LmLoadBf16x8AsFloat(staged[slot], p);
			LmLoadWeight8(post_weight, offset, w);
			LmLoadBf16x8AsFloat(*(const uint4 *)(res_row + offset), r);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				const float added = SparkMathRoundBf16(r[k] + SparkMathRoundBf16(p[k] * scale * w[k]));
				out[k] = HAS_SCALAR ? added * mult : added;
			}
			const uint4 packed = LmPackUint4FloatToBf16(out);
			*(uint4 *)(res_row + offset) = packed;
			staged[slot] = packed;
			if ( next_out != 0 )
				total2 = LmSumSqBf16x8(packed, total2);
		}
	}
	if ( next_out != 0 )
	{
		__syncthreads();
		const float scale2 = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total2, reduction), (float)width, norm_eps);
		uint16_t *n_row = next_out + base;
		uint16_t *rt_row = TRIPLE_OUT ? router_out + base : (uint16_t *)0;
		uint16_t *m_row = TRIPLE_OUT ? moe_out + base : (uint16_t *)0;
		#pragma unroll
		for ( uint32_t slot = 0u; slot < 4u; ++slot )
		{
			const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
			if ( offset < width )
			{
				float x[8], w[8], out[8];
				LmLoadBf16x8AsFloat(staged[slot], x);
				LmLoadWeight8(next_weight, offset, w);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					out[k] = ROUND_FIRST ? SparkMathNormGainRounded(x[k] * scale2, w[k]) : x[k] * scale2 * w[k];
				*(uint4 *)(n_row + offset) = LmPackUint4FloatToBf16(out);
				if constexpr ( TRIPLE_OUT )
				{
					float rs[8], mw[8], rt[8], mo[8];
					LmLoadWeight8(router_scale, offset, rs);
					LmLoadWeight8(moe_weight, offset, mw);
					#pragma unroll
					for ( uint32_t k = 0u; k < 8u; ++k )
					{
						rt[k] = SparkMathRoundBf16(SparkMathRoundBf16(x[k] * scale2) * rs[k]) * router_root;
						mo[k] = ROUND_FIRST ? SparkMathNormGainRounded(x[k] * scale2, mw[k]) : x[k] * scale2 * mw[k];
					}
					*(uint4 *)(rt_row + offset) = LmPackUint4FloatToBf16(rt);
					*(uint4 *)(m_row + offset) = LmPackUint4FloatToBf16(mo);
				}
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS, bool ROUND_FIRST>
__global__ __launch_bounds__(THREADS, 1)
static void GemmaMoeCombineNextKernel(uint16_t *__restrict__ residual, const uint16_t *__restrict__ dense, const uint16_t *__restrict__ moe,
	const float *__restrict__ dense_norm, const float *__restrict__ moe_norm, const float *__restrict__ post_norm, const float *__restrict__ scalar,
	const float *__restrict__ next_weight, uint16_t *__restrict__ next_out, uint32_t width, float post_eps, float norm_eps)
{
	__shared__ float reduction[2u * LmBlockWarps<THREADS>()];
	const uint64_t base = (uint64_t)blockIdx.x * width;
	const uint16_t *dense_row = dense + base, *moe_row = moe + base;
	uint16_t *res_row = residual + base;
	uint4 d_staged[2], m_staged[2], c_staged[2];
	float dense_total = 0.0f, moe_total = 0.0f, total = 0.0f;
	LmDependentWait();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < 2u; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			d_staged[slot] = *(const uint4 *)(dense_row + offset);
			m_staged[slot] = *(const uint4 *)(moe_row + offset);
			dense_total = LmSumSqBf16x8(d_staged[slot], dense_total);
			moe_total = LmSumSqBf16x8(m_staged[slot], moe_total);
		}
	}
	LmBlockSum2Once<THREADS>(&dense_total, &moe_total, reduction);
	__syncthreads();
	const float dense_scale = SparkMathRmsScale(dense_total, (float)width, post_eps);
	const float moe_scale = SparkMathRmsScale(moe_total, (float)width, post_eps);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < 2u; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			float dv[8], mv[8], dn[8], mn[8], comb[8];
			LmLoadBf16x8AsFloat(d_staged[slot], dv);
			LmLoadBf16x8AsFloat(m_staged[slot], mv);
			LmLoadFloat8(dense_norm + offset, dn);
			LmLoadFloat8(moe_norm + offset, mn);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				const float c = SparkMathRoundBf16(SparkMathRoundBf16(dv[k] * dense_scale * dn[k]) + SparkMathRoundBf16(mv[k] * moe_scale * mn[k]));
				comb[k] = c;
				total += c * c;
			}
			c_staged[slot] = LmPackUint4FloatToBf16(comb);
		}
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)width, post_eps);
	const float multiplier = scalar[0];
	float total2 = 0.0f;
	#pragma unroll
	for ( uint32_t slot = 0u; slot < 2u; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < width )
		{
			float comb[8], pn[8], rv[8], out[8];
			LmLoadBf16x8AsFloat(c_staged[slot], comb);
			LmLoadFloat8(post_norm + offset, pn);
			LmLoadBf16x8AsFloat(*(const uint4 *)(res_row + offset), rv);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out[k] = SparkMathRoundBf16(rv[k] + SparkMathRoundBf16(comb[k] * scale * pn[k])) * multiplier;
			const uint4 packed = LmPackUint4FloatToBf16(out);
			*(uint4 *)(res_row + offset) = packed;
			c_staged[slot] = packed;
			if ( next_out != 0 )
				total2 = LmSumSqBf16x8(packed, total2);
		}
	}
	if ( next_out != 0 )
	{
		__syncthreads();
		const float scale2 = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total2, reduction), (float)width, norm_eps);
		uint16_t *n_row = next_out + base;
		#pragma unroll
		for ( uint32_t slot = 0u; slot < 2u; ++slot )
		{
			const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
			if ( offset < width )
			{
				float x[8], w[8], out[8];
				LmLoadBf16x8AsFloat(c_staged[slot], x);
				LmLoadWeight8(next_weight, offset, w);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					out[k] = ROUND_FIRST ? SparkMathNormGainRounded(x[k] * scale2, w[k]) : x[k] * scale2 * w[k];
				*(uint4 *)(n_row + offset) = LmPackUint4FloatToBf16(out);
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaGeluTanhMulKernel(const uint16_t *__restrict__ gate_up, uint16_t *__restrict__ output, uint32_t width)
{
	const uint64_t base = (uint64_t)blockIdx.y * 2u * width;
	LmDependentWait();
	LmBinaryBf16GridBody<THREADS>(gate_up + base, gate_up + base + width, output + (uint64_t)blockIdx.y * width, width, GemmaGeluTanhOp{});
	LmDependentRelease();
}

template<uint32_t THREADS, bool HAS_LIMIT>
__global__ __launch_bounds__(THREADS, 2)
static void GemmaSiluMulKernel(const uint16_t *__restrict__ gate_up, uint16_t *__restrict__ output, uint32_t width, float limit)
{
	const uint64_t base = (uint64_t)blockIdx.y * 2u * width;
	LmDependentWait();
	if constexpr ( HAS_LIMIT )
		LmBinaryBf16GridBody<THREADS>(gate_up + base, gate_up + base + width, output + (uint64_t)blockIdx.y * width, width, LmSiluGateLimitOp{limit});
	else
		LmBinaryBf16GridBody<THREADS>(gate_up + base, gate_up + base + width, output + (uint64_t)blockIdx.y * width, width, LmStridedGateUpOp<true>{});
	LmDependentRelease();
}

template<uint32_t EXPERTS, uint32_t TOPK, bool BF16_INPUT>
static __device__ __forceinline__ void GemmaSigmoidTopkWarp(const void *logits_ptr, const float *__restrict__ bias, uint32_t row,
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
			const float lv = BF16_INPUT ? LmBf16ToFloat(((const uint16_t *)logits_ptr)[row_base + e]) : ((const float *)logits_ptr)[row_base + e];
			const float s = SparkMathSigmoid(lv);
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
	uint32_t chosen_e[TOPK];
	float chosen_s[TOPK];
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
		chosen_e[pick] = e;
		chosen_s[pick] = picked_s;
		total += picked_s;
		if ( lane == 0u )
			LmTopkRouteFault(fault, (uint32_t)(best >> 32u));
	}
	if ( lane == 0u )
	{
		const float denom = total + 1e-20f;
		#pragma unroll
		for ( uint32_t pick = 0u; pick < TOPK; ++pick )
		{
			const uint32_t ge = chosen_e[pick];
			if ( ge >= expert_first && ge < expert_first + local_experts )
			{
				out_exp[pick] = ge - expert_first;
				out_wt[pick] = chosen_s[pick] / denom * scale;
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
__global__ __launch_bounds__(GM_THREADS, 1)
static void GemmaSigmoidRouteSmallKernel(const void *__restrict__ logits, const float *__restrict__ bias, uint32_t rows, uint32_t expert_first,
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
		GemmaSigmoidTopkWarp<G::kExperts, G::kTopK, !G::kRouterFloat>(logits, bias, warp, expert_first, G::kLocalExperts, G::kRoutedScale, fault,
			s_exp + warp * G::kTopK, s_wt + warp * G::kTopK, lane);
	__syncthreads();
	if ( threadIdx.x < routes )
	{
		route_expert[threadIdx.x] = s_exp[threadIdx.x];
		route_weight[threadIdx.x] = s_wt[threadIdx.x];
	}
	LmRouteBuildSmallBody<GM_THREADS, G::kLocalExperts + 1u>(s_exp, routes, G::kTopK, group_row_offset, route_packed_row, route_source_token,
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
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaSigmoidTopkLocalKernel(const void *__restrict__ logits, const float *__restrict__ bias, uint32_t rows, uint32_t expert_first,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight)
{
	constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t row = blockIdx.x * WARPS + warp;
	LmDependentWait();
	if ( row < rows )
		GemmaSigmoidTopkWarp<G::kExperts, G::kTopK, !G::kRouterFloat>(logits, bias, row, expert_first, G::kLocalExperts, G::kRoutedScale, fault,
			route_expert + (uint64_t)row * G::kTopK, route_weight + (uint64_t)row * G::kTopK, lane);
	LmDependentRelease();
}

template<class G>
static __device__ __forceinline__ void GemmaSoftmaxShared(const uint16_t *__restrict__ logits, uint32_t row, float *__restrict__ s_probs,
	float *__restrict__ reduction)
{
	constexpr uint32_t WIDTH = G::kExperts;
	constexpr uint32_t per_thread = (WIDTH + GM_THREADS - 1u) / GM_THREADS;
	const uint64_t row_base = (uint64_t)row * WIDTH;
	float top = -INFINITY, sum = 0.0f;
	float staged[per_thread];
	#pragma unroll
	for ( uint32_t slot = 0u; slot < per_thread; ++slot )
	{
		const uint32_t index = threadIdx.x + slot * GM_THREADS;
		staged[slot] = index < WIDTH ? LmBf16ToFloat(logits[row_base + index]) : -INFINITY;
		top = fmaxf(top, staged[slot]);
	}
	top = LmBlockMax<GM_THREADS>(top, reduction);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < per_thread; ++slot )
	{
		staged[slot] = threadIdx.x + slot * GM_THREADS < WIDTH ? expf(staged[slot] - top) : 0.0f;
		sum += staged[slot];
	}
	sum = LmBlockSumOnce<GM_THREADS>(sum, reduction);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < per_thread; ++slot )
		if ( threadIdx.x + slot * GM_THREADS < WIDTH )
			s_probs[threadIdx.x + slot * GM_THREADS] = staged[slot] / sum;
}

template<class G>
__global__ __launch_bounds__(GM_THREADS, 1)
static void GemmaSoftmaxRouteSmallKernel(const uint16_t *__restrict__ logits, const float *__restrict__ expert_scale, uint32_t rows,
	uint32_t expert_first, uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight,
	uint32_t *__restrict__ packed_expert, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row,
	uint32_t *__restrict__ route_source_token, float *__restrict__ packed_weight, uint32_t tile_m, uint32_t n_up,
	uint32_t *__restrict__ tile_prefix_up, uint32_t n_down, uint32_t *__restrict__ tile_prefix_down)
{
	__shared__ float s_probs[G::kExperts];
	__shared__ float reduction[LmBlockWarps<GM_THREADS>()];
	__shared__ uint32_t s_exp[32];
	__shared__ float s_wt[32];
	const uint32_t routes = rows * G::kTopK, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	for ( uint32_t r = 0u; r < rows; ++r )
	{
		GemmaSoftmaxShared<G>(logits, r, s_probs, reduction);
		__syncthreads();
		if ( warp == 0u )
			LmTopkWarpRow<G::kTopK, true, LM_TOPK_SCORE_IDENTITY>(s_probs, G::kExperts, s_exp + r * G::kTopK, s_wt + r * G::kTopK,
				(const float *)0, (const uint16_t *)0, 1.0f, fault, 0u, lane);
		__syncthreads();
	}
	if ( threadIdx.x < routes )
	{
		const uint32_t ge = s_exp[threadIdx.x];
		const float wt = s_wt[threadIdx.x] * expert_scale[ge];
		if ( ge >= expert_first && ge < expert_first + G::kLocalExperts )
		{
			s_exp[threadIdx.x] = ge - expert_first;
			s_wt[threadIdx.x] = wt;
		}
		else
		{
			s_exp[threadIdx.x] = G::kLocalExperts;
			s_wt[threadIdx.x] = 0.0f;
		}
		route_expert[threadIdx.x] = s_exp[threadIdx.x];
		route_weight[threadIdx.x] = s_wt[threadIdx.x];
	}
	__syncthreads();
	LmRouteBuildSmallBody<GM_THREADS, G::kLocalExperts + 1u>(s_exp, routes, G::kTopK, group_row_offset, route_packed_row, route_source_token,
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
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaSoftmaxTopkLocalKernel(const uint16_t *__restrict__ logits, const float *__restrict__ expert_scale, uint32_t expert_first,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight)
{
	__shared__ float s_probs[G::kExperts];
	__shared__ float reduction[LmBlockWarps<GM_THREADS>()];
	const uint32_t row = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	GemmaSoftmaxShared<G>(logits, row, s_probs, reduction);
	__syncthreads();
	if ( warp == 0u )
	{
		const uint64_t base = (uint64_t)row * G::kTopK;
		LmTopkWarpRow<G::kTopK, true, LM_TOPK_SCORE_IDENTITY>(s_probs, G::kExperts, route_expert + base, route_weight + base,
			(const float *)0, (const uint16_t *)0, 1.0f, fault, 0u, lane);
		if ( lane < G::kTopK )
		{
			const uint32_t ge = route_expert[base + lane];
			const float wt = route_weight[base + lane] * expert_scale[ge];
			if ( ge >= expert_first && ge < expert_first + G::kLocalExperts )
			{
				route_expert[base + lane] = ge - expert_first;
				route_weight[base + lane] = wt;
			}
			else
			{
				route_expert[base + lane] = G::kLocalExperts;
				route_weight[base + lane] = 0.0f;
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t EXPERTS>
__global__ __launch_bounds__(THREADS, 1)
static void GemmaRouteBuildPackedKernel(const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, uint32_t routes,
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

template<bool GELU>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaMoeActPackedKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, const float *__restrict__ weight,
	const uint32_t *__restrict__ group_row_offset, uint32_t local_experts, uint16_t *__restrict__ output, uint32_t width, float limit)
{
	const uint32_t pr = blockIdx.y;
	LmDependentWait();
	if ( pr < group_row_offset[local_experts] )
	{
		const uint64_t row_base = (uint64_t)pr * width;
		const uint4 *g4 = (const uint4 *)(gate + row_base);
		const uint4 *u4 = (const uint4 *)(up + row_base);
		uint4 *o4 = (uint4 *)(output + row_base);
		const float wt = weight[pr];
		const uint32_t chunks = width >> 3u;
		for ( uint32_t c = blockIdx.x * GM_THREADS + threadIdx.x; c < chunks; c += gridDim.x * GM_THREADS )
		{
			float gv[8], uv[8];
			LmLoadBf16x8AsFloat(g4[c], gv);
			LmLoadBf16x8AsFloat(u4[c], uv);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				if constexpr ( GELU )
					gv[k] = SparkMathRoundBf16(SparkMathGeluTanhScaleFirst(gv[k])) * uv[k] * wt;
				else
					gv[k] = SparkMathGateLimitedGlu(gv[k], uv[k], limit) * wt;
			}
			o4[c] = LmPackUint4FloatToBf16(gv);
		}
	}
	LmDependentRelease();
}

template<bool HAS_SHARED>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaMoeLocalFinalizeKernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row,
	const uint32_t *__restrict__ route_expert, const uint16_t *__restrict__ shared, uint16_t *__restrict__ output, uint32_t top_k,
	uint32_t local_experts, uint32_t width)
{
	__shared__ uint32_t active_rows[LM_MOE_LOCAL_MAX_TOPK], active_count;
	const uint32_t row = blockIdx.y;
	const uint64_t row_base = (uint64_t)row * width;
	uint16_t *out_row = output + row_base;
	const uint16_t *shared_row = HAS_SHARED ? shared + row_base : (const uint16_t *)0;
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		uint32_t count = 0u;
		for ( uint32_t k = 0u; k < top_k; ++k )
		{
			const uint32_t route = row * top_k + k;
			if ( route_expert[route] < local_experts )
				active_rows[count++] = route_packed_row[route];
		}
		active_count = count;
	}
	__syncthreads();
	const uint32_t count = active_count;
	const uint32_t chunks = width >> 3u;
	const uint4 *packed4 = (const uint4 *)packed;
	const uint4 *shared4 = HAS_SHARED ? (const uint4 *)shared_row : (const uint4 *)0;
	uint4 *out4 = (uint4 *)out_row;
	if ( count == 0u )
	{
		const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
		for ( uint32_t c = blockIdx.x * GM_THREADS + threadIdx.x; c < chunks; c += gridDim.x * GM_THREADS )
			out4[c] = HAS_SHARED ? __ldcs(shared4 + c) : zero;
	}
	else
	{
		for ( uint32_t c = blockIdx.x * GM_THREADS + threadIdx.x; c < chunks; c += gridDim.x * GM_THREADS )
		{
			float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			for ( uint32_t k = 0u; k < count; ++k )
				LmAddBf16x8(acc, __ldcs(packed4 + (uint64_t)active_rows[k] * chunks + c));
			if constexpr ( HAS_SHARED )
				LmAddBf16x8(acc, __ldcs(shared4 + c));
			out4[c] = LmPackUint4FloatToBf16(acc);
		}
	}
	LmDependentRelease();
}

template<class G, bool GELU>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaMoeB1W1W3ActKernel(const uint16_t *__restrict__ w1, const uint16_t *__restrict__ w3, const uint16_t *__restrict__ activation,
	const float *__restrict__ packed_weight, const uint32_t *__restrict__ packed_expert, const uint32_t *__restrict__ group_row_offset,
	uint16_t *__restrict__ output, float limit)
{
	constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
	constexpr uint32_t K_V = G::kHidden >> 3u;
	constexpr uint32_t N = G::kMoeInter;
	const uint32_t pr = blockIdx.y;
	LmDependentWait();
	if ( pr < __ldg(group_row_offset + G::kLocalExperts) )
	{
		const uint32_t exp = __ldg(packed_expert + pr);
		const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
		const uint32_t col0 = (blockIdx.x * WARPS + warp) * 2u;
		if ( col0 < N )
		{
			const bool has1 = col0 + 1u < N;
			const uint64_t w_off0 = ((uint64_t)exp * N + col0) * K_V;
			const uint4 *w1_0 = (const uint4 *)w1 + w_off0;
			const uint4 *w3_0 = (const uint4 *)w3 + w_off0;
			const uint4 *w1_1 = has1 ? w1_0 + K_V : w1_0;
			const uint4 *w3_1 = has1 ? w3_0 + K_V : w3_0;
			const uint4 *act_row = (const uint4 *)activation;
			if ( lane == 0u )
			{
				LmPrefetchL2(w1_0, K_V * 16u);
				LmPrefetchL2(w3_0, K_V * 16u);
				if ( has1 )
				{
					LmPrefetchL2(w1_1, K_V * 16u);
					LmPrefetchL2(w3_1, K_V * 16u);
				}
			}
			float g0 = 0.0f, u0 = 0.0f, g1 = 0.0f, u1 = 0.0f;
			#pragma unroll 4
			for ( uint32_t c = lane; c < K_V; c += LM_WARP_LANES )
			{
				const uint4 va = __ldg(act_row + c);
				const uint4 vw1_0 = __ldcs(w1_0 + c);
				const uint4 vw3_0 = __ldcs(w3_0 + c);
				const uint4 vw1_1 = __ldcs(w1_1 + c);
				const uint4 vw3_1 = __ldcs(w3_1 + c);
				g0 = fmaf(1.0f, LmSkinnyBf16Chunk(vw1_0, va, 0.0f), g0);
				u0 = fmaf(1.0f, LmSkinnyBf16Chunk(vw3_0, va, 0.0f), u0);
				g1 = fmaf(1.0f, LmSkinnyBf16Chunk(vw1_1, va, 0.0f), g1);
				u1 = fmaf(1.0f, LmSkinnyBf16Chunk(vw3_1, va, 0.0f), u1);
			}
			#pragma unroll
			for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
			{
				g0 += __shfl_down_sync(0xffffffffu, g0, offset);
				u0 += __shfl_down_sync(0xffffffffu, u0, offset);
				g1 += __shfl_down_sync(0xffffffffu, g1, offset);
				u1 += __shfl_down_sync(0xffffffffu, u1, offset);
			}
			if ( lane == 0u )
			{
				const float wt = __ldg(packed_weight + pr);
				const float rg0 = SparkMathRoundBf16(g0), ru0 = SparkMathRoundBf16(u0);
				output[(uint64_t)pr * N + col0] = LmFloatToBf16((GELU ? SparkMathRoundBf16(SparkMathGeluTanhScaleFirst(rg0)) * ru0 : SparkMathGateLimitedGlu(rg0, ru0, limit)) * wt);
				if ( has1 )
				{
					const float rg1 = SparkMathRoundBf16(g1), ru1 = SparkMathRoundBf16(u1);
					output[(uint64_t)pr * N + col0 + 1u] = LmFloatToBf16((GELU ? SparkMathRoundBf16(SparkMathGeluTanhScaleFirst(rg1)) * ru1 : SparkMathGateLimitedGlu(rg1, ru1, limit)) * wt);
				}
			}
		}
	}
	LmDependentRelease();
}

template<class G, bool HAS_SHARED>
__global__ __launch_bounds__(GM_THREADS, 2)
static void GemmaMoeB1W2FinalizeKernel(const uint16_t *__restrict__ w2, const uint16_t *__restrict__ expert_act,
	const uint32_t *__restrict__ route_packed_row, const uint32_t *__restrict__ route_expert, const uint16_t *__restrict__ shared,
	uint16_t *__restrict__ output)
{
	constexpr uint32_t K = G::kMoeInter;
	constexpr uint32_t K_V = K >> 3u;
	constexpr uint32_t N = G::kHidden;
	constexpr bool HALF_WARP = (K == 128u);
	constexpr uint32_t SUB_LANES = HALF_WARP ? 16u : 32u;
	constexpr uint32_t NPG = HALF_WARP ? 4u : 2u;
	constexpr uint32_t GROUPS = GM_THREADS / SUB_LANES;
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
	const uint32_t col0 = (blockIdx.x * GROUPS + (threadIdx.x / SUB_LANES)) * NPG;
	const uint32_t sub = threadIdx.x & (SUB_LANES - 1u);
	if ( col0 < N )
	{
		const uint32_t count = s_count;
		if ( count == 0u )
		{
			if ( sub == 0u )
				#pragma unroll
				for ( uint32_t n = 0u; n < NPG; ++n )
					if ( col0 + n < N )
						output[col0 + n] = HAS_SHARED ? shared[col0 + n] : 0u;
		}
		else if constexpr ( HALF_WARP )
		{
			float total[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			const uint4 *w_cur = (const uint4 *)w2 + ((uint64_t)s_exp[0] * N + col0) * K_V + sub;
			uint4 va_cur = __ldg((const uint4 *)expert_act + (uint64_t)s_pr[0] * K_V + sub);
			uint4 w0_cur = __ldcs(w_cur);
			uint4 w1_cur = __ldcs(w_cur + K_V);
			uint4 w2_cur = __ldcs(w_cur + 2u * K_V);
			uint4 w3_cur = __ldcs(w_cur + 3u * K_V);
			for ( uint32_t k = 0u; k < count; ++k )
			{
				uint4 va_nxt, w0_nxt, w1_nxt, w2_nxt, w3_nxt;
				if ( k + 1u < count )
				{
					const uint4 *w_nxt = (const uint4 *)w2 + ((uint64_t)s_exp[k + 1u] * N + col0) * K_V + sub;
					va_nxt = __ldg((const uint4 *)expert_act + (uint64_t)s_pr[k + 1u] * K_V + sub);
					w0_nxt = __ldcs(w_nxt);
					w1_nxt = __ldcs(w_nxt + K_V);
					w2_nxt = __ldcs(w_nxt + 2u * K_V);
					w3_nxt = __ldcs(w_nxt + 3u * K_V);
				}
				const float d0 = GemmaHalfWarpSum(LmSkinnyBf16Chunk(w0_cur, va_cur, 0.0f));
				const float d1 = GemmaHalfWarpSum(LmSkinnyBf16Chunk(w1_cur, va_cur, 0.0f));
				const float d2 = GemmaHalfWarpSum(LmSkinnyBf16Chunk(w2_cur, va_cur, 0.0f));
				const float d3 = GemmaHalfWarpSum(LmSkinnyBf16Chunk(w3_cur, va_cur, 0.0f));
				if ( sub == 0u )
				{
					total[0] += SparkMathRoundBf16(d0);
					total[1] += SparkMathRoundBf16(d1);
					total[2] += SparkMathRoundBf16(d2);
					total[3] += SparkMathRoundBf16(d3);
				}
				if ( k + 1u < count )
				{
					va_cur = va_nxt;
					w0_cur = w0_nxt;
					w1_cur = w1_nxt;
					w2_cur = w2_nxt;
					w3_cur = w3_nxt;
				}
			}
			if ( sub == 0u )
			{
				#pragma unroll
				for ( uint32_t n = 0u; n < 4u; ++n )
				{
					if constexpr ( HAS_SHARED )
						total[n] += LmBf16ToFloat(shared[col0 + n]);
					output[col0 + n] = LmFloatToBf16(total[n]);
				}
			}
		}
		else
		{
			float total[2] = { 0.0f, 0.0f };
			for ( uint32_t k = 0u; k < count; ++k )
			{
				const uint4 *a_row = (const uint4 *)expert_act + (uint64_t)s_pr[k] * K_V;
				const uint4 *w0_row = (const uint4 *)w2 + ((uint64_t)s_exp[k] * N + col0) * K_V;
				const uint4 *w1_row = w0_row + K_V;
				float acc0 = 0.0f, acc1 = 0.0f;
				for ( uint32_t c = sub; c < K_V; c += LM_WARP_LANES )
				{
					const uint4 va = __ldg(a_row + c);
					acc0 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w0_row + c), va, 0.0f), acc0);
					acc1 = fmaf(1.0f, LmSkinnyBf16Chunk(__ldcs(w1_row + c), va, 0.0f), acc1);
				}
				#pragma unroll
				for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				{
					acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
					acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
				}
				if ( sub == 0u )
				{
					total[0] += SparkMathRoundBf16(acc0);
					total[1] += SparkMathRoundBf16(acc1);
				}
			}
			if ( sub == 0u )
			{
				#pragma unroll
				for ( uint32_t n = 0u; n < 2u; ++n )
				{
					if constexpr ( HAS_SHARED )
						total[n] += LmBf16ToFloat(shared[col0 + n]);
					output[col0 + n] = LmFloatToBf16(total[n]);
				}
			}
		}
	}
	LmDependentRelease();
}

template<class G>
struct GemmaModel : CudaStage<GemmaModel<G>>
{
	using S = GemmaShape<G>;
	using State = GemmaState<G>;
	using Layer = GemmaLayer;
	using Schedule = GemmaSchedule<G, GemmaModel>;
	using Kv = CudaStageKv<CudaStageKvViews<typename S::SlidingKvGeometry, &State::sliding_views, 0u, S::kSlidingLayers>,
		CudaStageKvViews<typename S::FullKvGeometry, &State::full_views, 1u, S::kFullLayers>>;
	using LaneShared = CudaStageFields<&State::full_inv_freq, &State::kv_error>;

	static constexpr bool kLanes = true;
	static constexpr bool kKeepRows = true;
	static constexpr bool kFusedPostNorm = true;
	static constexpr uint32_t kTpDegree = G::kTpDegree;
	static constexpr uint32_t kHidden = G::kHidden;
	static constexpr uint32_t kVocab = G::kVocab;
	static constexpr uint32_t kLayers = G::kLayers;
	static constexpr float kEpsilon = G::kNormEpsilon;
	static constexpr uint32_t kPackAlignment = GEMMA_PACK_ALIGNMENT;

	static_assert(G::kHidden <= GM_THREADS * 32u && (G::kHidden & 7u) == 0u, "hidden width fits four vectorized register slots per thread");
	static_assert(!G::kMoe || G::kSigmoidRouter || S::kMaxFused >= G::kHidden, "attn_full holds router_input during FFN");

	struct Frame
	{
		const SparkStageRunnerStep *step;
		SparkGpuStream stream;
	};

	static const char *Tag(void) { return G::Tag(); }
	static const char *ReportTag(void) { return G::ReportTag(); }
	static const char *WeightdModel(void) { return G::Name(); }

	static int32_t Project(const uint16_t *weight, const uint16_t *source, uint16_t *destination, float *destination_f32, uint32_t rows,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), source, destination, destination_f32, rows, input_dimension, output_dimension,
			step->commit == 0u ? 1u : 0u, step->multiprocessors, stream);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, uint16_t *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		return Project(weight, source, destination, (float *)0, rows, input_dimension, output_dimension, step, stream);
	}

	static int32_t Linear(const uint16_t *weight, const uint16_t *source, float *destination, uint32_t rows, uint32_t input_dimension,
		uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		return Project(weight, source, (uint16_t *)0, destination, rows, input_dimension, output_dimension, step, stream);
	}

	static int32_t ExpertProjection(const LmLocalExpertRoutes &routes, const uint16_t *weight, const uint16_t *activation, uint16_t *output, uint32_t down,
		uint32_t input_dimension, uint32_t output_dimension, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t exact = step->commit == 0u ? 1u : 0u;
		if ( exact == 0u && routes.rows > 1u )
		{
			const bool use_stream = down != 0u ? (routes.routes >= 8u) : (routes.rows >= 16u || routes.routes >= 128u);
			if ( use_stream )
			{
				const int32_t launch = LmStreamGemmGrouped<LmBf16Format>(weight, LmScaleTensorNone(), activation, output, routes.group_row_offset,
					down != 0u ? (const uint32_t *)0 : routes.route_source_token, routes.local_experts, routes.routes, input_dimension, output_dimension,
					routes.multiprocessors, stream);
				if ( launch != LM_LAUNCH_ERR_SHAPE )
					return launch;
			}
		}
		return down != 0u ? LmLocalExpertsDown<LmBf16Format>(routes, weight, LmScaleTensorNone(), activation, output, input_dimension, output_dimension, stream, exact) :
			LmLocalExpertsUp<LmBf16Format>(routes, weight, LmScaleTensorNone(), activation, output, input_dimension, output_dimension, stream, exact);
	}

	static void ExpertRoutes(const State *state, const SparkStageRunnerStep *step, uint32_t rows, LmLocalExpertRoutes *routes)
	{
		routes->group_row_offset = state->group_row_offset;
		routes->route_source_token = state->route_source_token;
		routes->tile_prefix_up = state->tile_prefix_up;
		routes->tile_prefix_down = state->tile_prefix_down;
		routes->rows = rows;
		routes->routes = rows * G::kTopK;
		routes->top_k = G::kTopK;
		routes->local_experts = G::kLocalExperts;
		routes->multiprocessors = step->multiprocessors;
	}

	static SparkStatus BuildRoutesPacked(State *state, uint32_t rows, uint32_t routes, SparkGpuStream stream)
	{
		const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
		const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
		const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
		(void)LM_LAUNCH_DEPENDENT((GemmaRouteBuildPackedKernel<GM_THREADS, G::kLocalExperts + 1u>), 1u, GM_THREADS, 0, stream,
			state->route_expert, state->route_weight, routes, G::kTopK, state->group_row_offset, state->route_packed_row, state->route_source_token,
			state->packed_weight, state->route_global, tile_m, n_up, state->tile_prefix_up, n_down, state->tile_prefix_down);
		return CudaStageLaunched();
	}

	template<bool FULL>
	static SparkStatus Attention(State *state, const Frame &frame, const GemmaLayer &entry)
	{
		const GemmaLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		using Kv = typename std::conditional<FULL, typename S::FullKvGeometry, typename S::SlidingKvGeometry>::type;
		constexpr uint32_t HD = FULL ? G::kFullDim : G::kSlidingDim, KVH = FULL ? G::kFullKvHeads : G::kSlidingKvHeads;
		constexpr uint32_t QW = FULL ? S::kFullQ : S::kSlidingQ;
		constexpr uint32_t FUSED = FULL ? S::kFullFused : S::kSlidingFused;
		constexpr uint32_t HEADS = FULL ? G::kFullHeads : G::kSlidingHeads;
		constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
		constexpr uint32_t TASKS = HEADS + 2u * KVH;
		const uint32_t rows = step->rows, degree = state->tp_degree, local_rows = FUSED / degree;
		const uint32_t width = QW / degree, exact = step->commit == 0u ? 1u : 0u;
		const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(HEADS, HD, degree) * sizeof(float) + G::kHidden * sizeof(uint16_t) - 1u) /
			(G::kHidden * sizeof(uint16_t));
		const uint64_t stride = units * G::kHidden * sizeof(uint16_t) / sizeof(float);
		const LmKvShardView *view = FULL ? &state->full_views[layer->kv_index] : &state->sliding_views[layer->kv_index];
		bool gate_done = false;
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if constexpr ( G::kOutputGate || G::kHeadGate )
		{
			const uint32_t gate_out = G::kOutputGate ? width : (FULL ? S::kFullGateRows : S::kSlidingGateRows);
			const bool allow_multi = local_rows > LM_SKINNY_SPLIT_MAX_OUTPUTS || (local_rows + gate_out <= LM_SKINNY_SPLIT_MAX_OUTPUTS) ||
				(exact == 0u && rows > 1u);
			if ( allow_multi && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
			{
				const uint32_t allow_split = local_rows + gate_out <= LM_SKINNY_SPLIT_MAX_OUTPUTS ? 1u : 0u;
				const LmSkinnyDenseTarget targets[2] =
				{
					{ layer->qkv, state->attn_local, (float *)0, local_rows, local_rows, 0u, LmScaleTensorNone() },
					{ layer->gate, state->gate_logits, (float *)0, gate_out, gate_out, 0u, LmScaleTensorNone() }
				};
				if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 2u, state->normed, rows, G::kHidden, stream, exact, allow_split) == LM_LAUNCH_OK )
					gate_done = true;
			}
		}
		if ( !gate_done && Linear(layer->qkv, state->normed, state->attn_local, rows, G::kHidden, local_rows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, local_rows, state->attn_local, state->attn_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		(void)LM_LAUNCH_DEPENDENT((GemmaAttnPrepStoreKernel<G, FULL>), dim3((TASKS + WARPS - 1u) / WARPS, rows), GM_THREADS, 0, stream,
			state->attn_gathered, G::kQkNormWeighted ? layer->q_norm : (const float *)0, G::kQkNormWeighted ? layer->k_norm : (const float *)0,
			step->positions, step->tree != 0u ? step->slots : step->positions, step->sequence_of_row,
			FULL ? state->full_inv_freq : (const float *)0, *view, state->query, rows, local_rows);
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			LmGqaShardPartialLaunch<Kv,LmKvShardView,KVH,HD,HD>(*view, state->query, HEADS, step->sequence_of_row, step->context_length,
				step->positions, G::kAttentionScale, state->send, stride, rows, stream, (const uint32_t *)0, 0u, 1u, FULL ? 0u : layer->window,
				step->tree != 0u ? step->fork_positions : (const uint32_t *)0, step->tree != 0u ? step->slot_offsets : (const uint32_t *)0) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( LmGqaShardMergeLaunch<HD>(state->received, stride, degree, HEADS, state->tp_rank, state->merged, rows, stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kOutputGate )
		{
			if ( !gate_done && Linear(layer->gate, state->normed, state->gate_logits, rows, G::kHidden, width, step, stream) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			(void)LM_LAUNCH_DEPENDENT((GemmaOutputGateKernel<GM_THREADS>), rows, GM_THREADS, 0, stream, state->merged, state->gate_logits, width);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if constexpr ( G::kHeadGate )
		{
			constexpr uint32_t GATE_ROWS = FULL ? S::kFullGateRows : S::kSlidingGateRows;
			if ( !gate_done )
			{
				if ( exact == 0u && GATE_ROWS <= 16u )
					(void)LM_LAUNCH_DEPENDENT((GemmaSmallNBf16ProjKernel<G::kHidden>), rows, GM_THREADS, 0, stream,
						layer->gate, state->normed, state->gate_logits, rows, GATE_ROWS);
				else if ( Linear(layer->gate, state->normed, state->gate_logits, rows, G::kHidden, GATE_ROWS, step, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			(void)LM_LAUNCH_DEPENDENT((GemmaHeadGateBroadcastKernel<128u>), dim3(rows, width / HD), 128u, 0, stream,
				state->merged, state->gate_logits, width / HD, HD, GATE_ROWS);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( Linear(layer->out, state->merged, state->partial, rows, width, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Mlp(State *state, const Frame &frame, const GemmaLayer &entry)
	{
		const GemmaLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, ffn_rows = layer->ffn_rows, exact = step->commit == 0u ? 1u : 0u;
		bool fused_router = false;
		if constexpr ( G::kMoe && G::kSigmoidRouter )
		{
			const uint32_t gu_out = 2u * ffn_rows;
			const bool allow_multi = gu_out > LM_SKINNY_SPLIT_MAX_OUTPUTS || (gu_out + G::kExperts <= LM_SKINNY_SPLIT_MAX_OUTPUTS) ||
				(exact == 0u && rows > 1u);
			if ( allow_multi && layer->router_proj != 0 && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
			{
				const uint32_t allow_split = gu_out + G::kExperts <= LM_SKINNY_SPLIT_MAX_OUTPUTS ? 1u : 0u;
				const LmSkinnyDenseTarget targets[2] =
				{
					{ layer->gate_up, state->gate_up, (float *)0, gu_out, gu_out, 0u, LmScaleTensorNone() },
					{ layer->router_proj, G::kRouterFloat ? (uint16_t *)0 : state->router_logits, G::kRouterFloat ? state->router_probs : (float *)0,
						G::kExperts, G::kExperts, 0u, LmScaleTensorNone() }
				};
				if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 2u, state->normed, rows, G::kHidden, stream, exact, allow_split) == LM_LAUNCH_OK )
					fused_router = true;
			}
		}
		state->router_fused_done = fused_router ? 1u : 0u;
		if ( !fused_router && Linear(layer->gate_up, state->normed, state->gate_up, rows, G::kHidden, 2u * ffn_rows, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kGeluMlp )
			(void)LM_LAUNCH_DEPENDENT((GemmaGeluTanhMulKernel<GM_THREADS>), dim3(LmVecBlocks<GM_THREADS>(ffn_rows), rows), GM_THREADS, 0, stream,
				state->gate_up, state->intermediate, ffn_rows);
		else if ( layer->swiglu_limit != INFINITY )
			(void)LM_LAUNCH_DEPENDENT((GemmaSiluMulKernel<GM_THREADS, true>), dim3(LmVecBlocks<GM_THREADS>(ffn_rows), rows), GM_THREADS, 0, stream,
				state->gate_up, state->intermediate, ffn_rows, layer->swiglu_limit);
		else
			(void)LM_LAUNCH_DEPENDENT((GemmaSiluMulKernel<GM_THREADS, false>), dim3(LmVecBlocks<GM_THREADS>(ffn_rows), rows), GM_THREADS, 0, stream,
				state->gate_up, state->intermediate, ffn_rows, INFINITY);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(layer->down, state->intermediate, state->partial, rows, ffn_rows, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	template<bool GELU, bool HAS_SHARED>
	static SparkStatus RunExpertsBody(State *state, const GemmaLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream,
		const uint16_t *act_in, uint16_t *output)
	{
		const uint32_t rows = step->rows, routes = rows * G::kTopK;
		if ( rows == 1u )
		{
			constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
			const uint32_t w1_blocks = (G::kMoeInter + WARPS * 2u - 1u) / (WARPS * 2u);
			(void)LM_LAUNCH_DEPENDENT((GemmaMoeB1W1W3ActKernel<G, GELU>), dim3(w1_blocks, routes), GM_THREADS, 0, stream,
				layer->moe_w1, layer->moe_w3, act_in, state->packed_weight, state->route_global, state->group_row_offset,
				state->expert_act, layer->swiglu_limit);
			constexpr uint32_t W2_SUB = G::kMoeInter == 128u ? 16u : 32u;
			constexpr uint32_t W2_NPG = G::kMoeInter == 128u ? 4u : 2u;
			constexpr uint32_t W2_COLS = (GM_THREADS / W2_SUB) * W2_NPG;
			const uint32_t w2_blocks = (G::kHidden + W2_COLS - 1u) / W2_COLS;
			(void)LM_LAUNCH_DEPENDENT((GemmaMoeB1W2FinalizeKernel<G, HAS_SHARED>), w2_blocks, GM_THREADS, 0, stream,
				layer->moe_w2, state->expert_act, state->route_packed_row, state->route_expert, HAS_SHARED ? state->partial : (const uint16_t *)0, output);
			return CudaStageLaunched();
		}
		LmLocalExpertRoutes expert_routes;
		ExpertRoutes(state, step, rows, &expert_routes);
		if ( ExpertProjection(expert_routes, layer->moe_w1, act_in, state->expert_gate, 0u, G::kHidden, G::kMoeInter, step, stream) != LM_LAUNCH_OK ||
			ExpertProjection(expert_routes, layer->moe_w3, act_in, state->expert_up, 0u, G::kHidden, G::kMoeInter, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((GemmaMoeActPackedKernel<GELU>), dim3(LmVecBlocks<GM_THREADS>(G::kMoeInter), routes), GM_THREADS, 0, stream,
			state->expert_gate, state->expert_up, state->packed_weight, state->group_row_offset, G::kLocalExperts, state->expert_act, G::kMoeInter,
			layer->swiglu_limit);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			ExpertProjection(expert_routes, layer->moe_w2, state->expert_act, state->expert_out, 1u, G::kMoeInter, G::kHidden, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		(void)LM_LAUNCH_DEPENDENT((GemmaMoeLocalFinalizeKernel<HAS_SHARED>), dim3(LmVecBlocks<GM_THREADS>(G::kHidden), rows), GM_THREADS, 0, stream,
			state->expert_out, state->route_packed_row, state->route_expert, HAS_SHARED ? state->partial : (const uint16_t *)0, output,
			G::kTopK, G::kLocalExperts, G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus ExpertRows(State *state, const GemmaLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t rows = step->rows, routes = rows * G::kTopK;
		uint16_t *output = state->partial + (uint64_t)rows * G::kHidden;
		if ( Linear(layer->router_proj, state->attn_full, state->router_logits, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( routes <= 32u )
		{
			const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
			const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			(void)LM_LAUNCH_DEPENDENT((GemmaSoftmaxRouteSmallKernel<G>), 1u, GM_THREADS, 0, stream, state->router_logits, layer->expert_scale, rows,
				state->expert_first, step->route_fault, state->route_expert, state->route_weight, state->route_global, state->group_row_offset,
				state->route_packed_row, state->route_source_token, state->packed_weight, tile_m, n_up, state->tile_prefix_up, n_down, state->tile_prefix_down);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			(void)LM_LAUNCH_DEPENDENT((GemmaSoftmaxTopkLocalKernel<G>), rows, GM_THREADS, 0, stream, state->router_logits, layer->expert_scale,
				state->expert_first, step->route_fault, state->route_expert, state->route_weight);
			if ( CudaStageLaunched() != SPARK_STATUS_OK || BuildRoutesPacked(state, rows, routes, stream) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		return RunExpertsBody<true, false>(state, layer, step, stream, state->moe_normed, output);
	}

	static SparkStatus SigmoidExpertRows(State *state, const GemmaLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream)
	{
		const uint32_t rows = step->rows, routes = rows * G::kTopK;
		const uint16_t *normed = state->normed;
		uint16_t *output = state->partial + (uint64_t)rows * G::kHidden;
		const bool already_projected = state->router_fused_done != 0u;
		state->router_fused_done = 0u;
		if ( !already_projected )
		{
			if constexpr ( G::kRouterFloat )
			{
				if ( Linear(layer->router_proj, normed, state->router_probs, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				if ( Linear(layer->router_proj, normed, state->router_logits, rows, G::kHidden, G::kExperts, step, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		const void *logits_ptr = G::kRouterFloat ? (const void *)state->router_probs : (const void *)state->router_logits;
		if ( routes <= 32u )
		{
			const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
			const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			(void)LM_LAUNCH_DEPENDENT((GemmaSigmoidRouteSmallKernel<G>), 1u, GM_THREADS, 0, stream, logits_ptr, layer->expert_bias, rows,
				state->expert_first, step->route_fault, state->route_expert, state->route_weight, state->route_global, state->group_row_offset,
				state->route_packed_row, state->route_source_token, state->packed_weight, tile_m, n_up, state->tile_prefix_up, n_down, state->tile_prefix_down);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			constexpr uint32_t WARPS = GM_THREADS / LM_WARP_LANES;
			(void)LM_LAUNCH_DEPENDENT((GemmaSigmoidTopkLocalKernel<G>), (rows + WARPS - 1u) / WARPS, GM_THREADS, 0, stream, logits_ptr,
				layer->expert_bias, rows, state->expert_first, step->route_fault, state->route_expert, state->route_weight);
			if ( CudaStageLaunched() != SPARK_STATUS_OK || BuildRoutesPacked(state, rows, routes, stream) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		return RunExpertsBody<false, true>(state, layer, step, stream, normed, output);
	}

	static SparkStatus Experts(State *state, const Frame &frame, const GemmaLayer &entry) { return ExpertRows(state, &entry, frame.step, frame.stream); }
	static SparkStatus SigmoidExperts(State *state, const Frame &frame, const GemmaLayer &entry) { return SigmoidExpertRows(state, &entry, frame.step, frame.stream); }
	static uint16_t *BranchRows(State *state, const Frame &frame, GemmaBranch branch) { return state->partial + (uint64_t)branch * frame.step->rows * G::kHidden; }

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream, Frame *frame)
	{
		if ( step->tree != 0u && (step->commit != 0u || step->sequences != 1u || step->slots == 0 || step->fork_positions == 0 || step->slot_offsets == 0) )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		state->router_fused_done = 0u;
		return CudaStage<GemmaModel>::Begin(state, step, stream, frame);
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
		for ( uint32_t index = 0u; index < S::kSlidingLayers; ++index )
		{
			CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			LM_LAUNCH((LmKvShardPathMoveKernel<typename S::SlidingKvGeometry,GM_THREADS>), step->rows, GM_THREADS, 0, stream,
				state->sliding_views[index], step->sequence_of_row, step->sequence_row_indices, kept, step->slots, 2u * S::kSlidingKv);
			CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		}
		for ( uint32_t index = 0u; index < S::kFullLayers; ++index )
		{
			CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			LM_LAUNCH((LmKvShardPathMoveKernel<typename S::FullKvGeometry,GM_THREADS>), step->rows, GM_THREADS, 0, stream,
				state->full_views[index], step->sequence_of_row, step->sequence_row_indices, kept, step->slots, 2u * S::kFullKv);
			CudaStageRowState(state, index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		}
		return CudaStageLaunched();
	}

	static SparkStatus EmbedNorm(State *state, const Frame &frame)
	{
		uint16_t *hidden = frame.step->hidden_bf16;
		(void)state;
		LM_LAUNCH((LmHeadRmsNormKernel<GM_THREADS,float>), dim3(1u,frame.step->rows), GM_THREADS, 0, frame.stream,
			hidden, (const float *)0, hidden, frame.step->rows, 1u, G::kHidden, G::kNormEpsilon, 1.0f);
		return SPARK_STATUS_OK;
	}

	static SparkStatus EmbedScale(State *state, const Frame &frame, float scale)
	{
		const uint64_t width = (uint64_t)frame.step->rows * G::kHidden;
		(void)state;
		LM_LAUNCH((LmScaleElementsKernel<uint16_t>), LmVecBlocks<GM_THREADS>(width), GM_THREADS, 0, frame.stream, frame.step->hidden_bf16, width, scale);
		return SPARK_STATUS_OK;
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, const float *weight)
	{
		(void)LM_LAUNCH_DEPENDENT((GemmaAddRmsNormKernel<GM_THREADS, G::kNormRoundFirst, false>), frame.step->rows, GM_THREADS, 0, frame.stream,
			frame.step->hidden_bf16, (uint16_t *)0, weight, state->normed, G::kHidden, G::kNormEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AddRmsNorm(State *state, const Frame &frame, GemmaBranch branch, const float *weight)
	{
		(void)LM_LAUNCH_DEPENDENT((GemmaAddRmsNormKernel<GM_THREADS, G::kNormRoundFirst, true>), frame.step->rows, GM_THREADS, 0, frame.stream,
			BranchRows(state, frame, branch), frame.step->hidden_bf16, weight, state->normed, G::kHidden, G::kNormEpsilon);
		return SPARK_STATUS_OK;
	}

	static SparkStatus PostAttnNorm(State *state, const Frame &frame, const GemmaLayer &layer)
	{
		if constexpr ( G::kMoe && !G::kSigmoidRouter )
			(void)LM_LAUNCH_DEPENDENT((GemmaPostNormAddNextKernel<GM_THREADS, G::kNormRoundFirst, false, true>), frame.step->rows, GM_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->partial, layer.post_attn_norm, (const float *)0, layer.pre_ff_norm, state->normed,
				layer.router_scale, state->attn_full, layer.pre_ff_norm_2, state->moe_normed, G::kHidden, G::kPostEpsilon, G::kNormEpsilon,
				(float)(1.0 / sqrt((double)G::kHidden)));
		else
			(void)LM_LAUNCH_DEPENDENT((GemmaPostNormAddNextKernel<GM_THREADS, G::kNormRoundFirst, false, false>), frame.step->rows, GM_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->partial, layer.post_attn_norm, (const float *)0, layer.pre_ff_norm, state->normed,
				(const float *)0, (uint16_t *)0, (const float *)0, (uint16_t *)0, G::kHidden, G::kPostEpsilon, G::kNormEpsilon, 1.0f);
		return SPARK_STATUS_OK;
	}

	static SparkStatus PostFfnNorm(State *state, const Frame &frame, const GemmaLayer &layer, const float *next_norm)
	{
		const uint32_t rows = frame.step->rows;
		if constexpr ( G::kMoe )
			(void)LM_LAUNCH_DEPENDENT((GemmaMoeCombineNextKernel<GM_THREADS, G::kNormRoundFirst>), rows, GM_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->partial, state->partial + (uint64_t)rows * G::kHidden, layer.post_ff_norm_1, layer.post_ff_norm_2,
				layer.post_ff_norm, layer.layer_scalar, next_norm, next_norm != 0 ? state->normed : (uint16_t *)0, G::kHidden, G::kPostEpsilon,
				G::kNormEpsilon);
		else if constexpr ( G::kLayerScalar )
			(void)LM_LAUNCH_DEPENDENT((GemmaPostNormAddNextKernel<GM_THREADS, G::kNormRoundFirst, true, false>), rows, GM_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->partial, layer.post_ff_norm, layer.layer_scalar, next_norm, next_norm != 0 ? state->normed : (uint16_t *)0,
				(const float *)0, (uint16_t *)0, (const float *)0, (uint16_t *)0, G::kHidden, G::kPostEpsilon, G::kNormEpsilon, 1.0f);
		else
			(void)LM_LAUNCH_DEPENDENT((GemmaPostNormAddNextKernel<GM_THREADS, G::kNormRoundFirst, false, false>), rows, GM_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->partial, layer.post_ff_norm, (const float *)0, next_norm, next_norm != 0 ? state->normed : (uint16_t *)0,
				(const float *)0, (uint16_t *)0, (const float *)0, (uint16_t *)0, G::kHidden, G::kPostEpsilon, G::kNormEpsilon, 1.0f);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AddRows(State *state, const Frame &frame, GemmaBranch branch)
	{
		const uint64_t width = (uint64_t)frame.step->rows * G::kHidden;
		LM_LAUNCH((LmAddElementsKernel), LmVecBlocks<GM_THREADS>(width), GM_THREADS, 0, frame.stream, frame.step->hidden_bf16, BranchRows(state, frame, branch), width);
		return SPARK_STATUS_OK;
	}

	static SparkStatus AllReduce(State *state, const Frame &frame, GemmaBranch branch, uint32_t blocks)
	{
		uint16_t *rows = BranchRows(state, frame, branch);
		return CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, blocks * frame.step->rows, 0u, rows, rows);
	}

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, width = rows * G::kHidden * sizeof(uint16_t), routes = rows * G::kTopK;
		const uint64_t groups = G::kMoe ? G::kLocalExperts + 2u : 0u;
		const CudaStageSlice plan[] =
		{
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, G::kMoe ? 2u * width : width },
			{ (void **)&state->attn_local, rows * (S::kMaxFused / state->tp_degree) * sizeof(uint16_t) },
			{ (void **)&state->attn_gathered, rows * S::kMaxFused * sizeof(uint16_t) },
			{ (void **)&state->attn_full, rows * S::kMaxFused * sizeof(uint16_t) },
			{ (void **)&state->query, rows * S::kMaxQ * sizeof(uint16_t) },
			{ (void **)&state->key, rows * S::kMaxKv * sizeof(uint16_t) },
			{ (void **)&state->value, rows * S::kMaxKv * sizeof(uint16_t) },
			{ (void **)&state->slot, rows * 2u * S::kMaxKv * sizeof(uint16_t) },
			{ (void **)&state->send, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->received, (uint64_t)state->tp_degree * state->record_stride * sizeof(float) },
			{ (void **)&state->merged, rows * (S::kMaxQ / state->tp_degree) * sizeof(uint16_t) },
			{ (void **)&state->gate_logits, G::kOutputGate ? rows * (S::kMaxQ / state->tp_degree) * sizeof(uint16_t) : G::kHeadGate ? rows * S::kMaxGateRows * sizeof(uint16_t) : 0u },
			{ (void **)&state->gate_up, rows * 2u * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->intermediate, rows * state->ffn_rows * sizeof(uint16_t) },
			{ (void **)&state->full_inv_freq, lane == 0u ? (G::kFullDim / 2u) * sizeof(float) : 0u },
			{ (void **)&state->moe_normed, G::kMoe ? width : 0u },
			{ (void **)&state->router_logits, rows * G::kExperts * sizeof(uint16_t) },
			{ (void **)&state->router_probs, rows * G::kExperts * sizeof(float) },
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
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		return CudaStagePlace(plan, base);
	}

	static SparkStatus Shape(State *state)
	{
		const uint64_t full = LmGqaShardRecordFloats(G::kFullHeads, G::kFullDim, state->tp_degree);
		const uint64_t sliding = LmGqaShardRecordFloats(G::kSlidingHeads, G::kSlidingDim, state->tp_degree);
		const uint64_t row_bytes = G::kHidden * sizeof(uint16_t);
		state->record_stride = (((uint64_t)state->max_rows * (full > sliding ? full : sliding) * sizeof(float) + row_bytes - 1u) / row_bytes) * row_bytes /
			sizeof(float);
		return SPARK_STATUS_OK;
	}

	static uint32_t Accepts(const SparkStageRunnerConfiguration *configuration) { return configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_BF16 ? 1u : 0u; }

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, window = 0u;
		const CudaStagePackValue values[] =
		{
			{ "tp_degree", &degree }, { "tp_rank", &rank }, { "hidden", &hidden }, { "layers", &layers }, { "vocab_rows", &vocab_rows },
			{ "ffn_rows", &state->ffn_rows }, { "sliding_window", &window }
		};
		if ( CudaStagePackValues(&state->pack, values) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			vocab_rows * degree != G::kVocab || window != G::kWindow || state->ffn_rows == 0u || state->ffn_rows % 64u != 0u )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u window %u, the runner is tp %u rank %u\n", G::Tag(), degree, rank,
				hidden, layers, window, configuration->tp_degree, configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if ( G::kMoe )
		{
			uint32_t experts = 0u, local = 0u, top_k = 0u;
			const CudaStagePackValue moe[] =
			{
				{ "experts", &experts }, { "local_experts", &local }, { "expert_first", &state->expert_first }, { "top_k", &top_k }
			};
			if ( CudaStagePackValues(&state->pack, moe) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			if ( experts != G::kExperts || local != G::kLocalExperts || top_k != G::kTopK ||
				state->expert_first != (G::kLocalExperts == G::kExperts ? 0u : rank * G::kLocalExperts) )
			{
				fprintf(stderr, "%s: pack experts %u local %u first %u top_k %u do not match the model\n", G::Tag(), experts, local,
					state->expert_first, top_k);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		geometry->tree_verify = 1u;
		geometry->head_norm_round_first = G::kNormRoundFirst ? 1u : 0u;
		geometry->logit_softcap = G::kLogitSoftcap;
		geometry->logit_scale = G::kLogitScale;
		if constexpr ( G::kEmbedAliasCount != 0u )
		{
			geometry->embed_alias_first = G::kEmbedAliasFirst;
			geometry->embed_alias_count = G::kEmbedAliasCount;
			geometry->embed_alias_row = G::kEmbedAliasRow;
		}
		if ( CudaStage<GemmaModel>::HeadRanges(state, geometry, !G::kUntiedHead, false) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( geometry->embed_bytes != (uint64_t)geometry->embed_rows * G::kHidden * sizeof(uint16_t) ||
			geometry->head_bytes != (uint64_t)geometry->head_rows * G::kHidden * sizeof(uint16_t) || geometry->head_norm_bytes != (uint64_t)G::kHidden * sizeof(float) )
		{
			fprintf(stderr, "%s: pack embed %u rows %llu bytes, head %u rows %llu bytes, head norm %llu bytes do not match the stage geometry\n", G::Tag(),
				geometry->embed_rows, (unsigned long long)geometry->embed_bytes, geometry->head_rows, (unsigned long long)geometry->head_bytes,
				(unsigned long long)geometry->head_norm_bytes);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus LayerTensors(State *state, uint32_t index, CudaStageTensors *tensors)
	{
		GemmaLayer *layer = &state->layers[index];
		const bool full = S::Full(index);
		const uint64_t hidden = G::kHidden, norm = hidden * sizeof(float);
		char key[SPARK_NAMED_PACK_MAX_NAME_BYTES];
		uint64_t offset, bytes;
		uint32_t gate_rows = 0u;
		snprintf(key, sizeof(key), "layers.%u.ffn_gate_up", index);
		layer->gate_up_rows = 0u;
		if ( SparkNamedPackTensorRange(&state->pack, G::Tag(), key, "bf16", &offset, &bytes, &layer->gate_up_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		tensors->Add("attn_qkv", "bf16", (uint64_t)((full ? S::kFullFused : S::kSlidingFused) / state->tp_degree) * hidden * sizeof(uint16_t), &layer->qkv);
		tensors->Add("attn_out", "bf16", hidden * ((full ? S::kFullQ : S::kSlidingQ) / state->tp_degree) * sizeof(uint16_t), &layer->out);
		tensors->Add("ffn_gate_up", "bf16", (uint64_t)layer->gate_up_rows * hidden * sizeof(uint16_t), &layer->gate_up);
		tensors->Add("ffn_down", "bf16", hidden * (layer->gate_up_rows / 2u) * sizeof(uint16_t), &layer->down);
		tensors->Add("input_norm", "f32", norm, &layer->input_norm);
		if ( !G::kPreNorm )
			tensors->Add("post_attn_norm", "f32", norm, &layer->post_attn_norm);
		tensors->Add("pre_ff_norm", "f32", norm, &layer->pre_ff_norm);
		if ( !G::kPreNorm )
			tensors->Add("post_ff_norm", "f32", norm, &layer->post_ff_norm);
		if ( G::kLayerScalar )
			tensors->Add("layer_scalar", "f32", (uint64_t)sizeof(float), &layer->layer_scalar);
		if ( G::kQkNormWeighted )
		{
			tensors->Add("attn_q_norm", "f32", (uint64_t)(full ? G::kFullDim : G::kSlidingDim) * sizeof(float), &layer->q_norm);
			tensors->Add("attn_k_norm", "f32", (uint64_t)(full ? G::kFullDim : G::kSlidingDim) * sizeof(float), &layer->k_norm);
		}
		if ( G::kOutputGate )
			tensors->Add("attn_gate", "bf16", ((full ? S::kFullQ : S::kSlidingQ) / state->tp_degree) * hidden * sizeof(uint16_t), &layer->gate);
		if ( G::kHeadGate )
		{
			snprintf(key, sizeof(key), "layers.%u.attn_gate", index);
			if ( SparkNamedPackTensorRange(&state->pack, G::Tag(), key, "bf16", &offset, &bytes, &gate_rows) != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			if ( gate_rows != (full ? S::kFullGateRows : S::kSlidingGateRows) )
			{
				fprintf(stderr, "%s: layer %u attn_gate has %u rows, the rank's padded head count is %u\n", G::Tag(), index, gate_rows,
					full ? S::kFullGateRows : S::kSlidingGateRows);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
			tensors->Add("attn_gate", "bf16", (uint64_t)gate_rows * hidden * sizeof(uint16_t), &layer->gate);
		}
		if ( (G::kSigmoidRouter && index >= G::kFirstMoeLayer) || (G::kMoe && !G::kSigmoidRouter) )
		{
			const uint64_t experts = (uint64_t)G::kLocalExperts * G::kMoeInter * hidden * sizeof(uint16_t);
			tensors->Add("router_proj", "bf16", (uint64_t)G::kExperts * hidden * sizeof(uint16_t), &layer->router_proj);
			if ( G::kSigmoidRouter )
				tensors->Add("expert_bias", "f32", (uint64_t)G::kExperts * sizeof(float), &layer->expert_bias);
			else
			{
				tensors->Add("router_scale", "f32", norm, &layer->router_scale);
				tensors->Add("expert_scale", "f32", (uint64_t)G::kExperts * sizeof(float), &layer->expert_scale);
			}
			tensors->Add("moe_w1", "bf16", experts, &layer->moe_w1);
			tensors->Add("moe_w3", "bf16", experts, &layer->moe_w3);
			tensors->Add("moe_w2", "bf16", experts, &layer->moe_w2);
			if ( !G::kSigmoidRouter )
			{
				tensors->Add("post_ff_norm_1", "f32", norm, &layer->post_ff_norm_1);
				tensors->Add("pre_ff_norm_2", "f32", norm, &layer->pre_ff_norm_2);
				tensors->Add("post_ff_norm_2", "f32", norm, &layer->post_ff_norm_2);
			}
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus LayerBound(State *state, uint32_t index)
	{
		GemmaLayer *layer = &state->layers[index];
		layer->ffn_rows = layer->gate_up_rows / 2u;
		if ( layer->gate_up_rows % 128u != 0u || layer->ffn_rows == 0u || layer->ffn_rows > state->ffn_rows )
		{
			fprintf(stderr, "%s: layer %u ffn_gate_up has %u rows; a rank holds gate|up in 64-row blocks of at most %u rows each\n", G::Tag(),
				index, layer->gate_up_rows, state->ffn_rows);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Arrange(State *state)
	{
		uint32_t sliding = 0u, full = 0u;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
		{
			GemmaLayer *layer = &state->layers[index];
			layer->kv_index = S::Full(index) ? full++ : sliding++;
			layer->window = S::Window(index);
			layer->swiglu_limit = S::SwigluLimit(index);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Upload(State *state)
	{
		float inv_freq[G::kFullDim / 2u];
		for ( uint32_t pair = 0u; pair < G::kFullDim / 2u; ++pair )
		{
			if constexpr ( G::kFullRopePairs != 0u )
				inv_freq[pair] = pair < G::kFullRopePairs ? (float)(1.0 / pow((double)G::kFullTheta, 2.0 * pair / (double)G::kFullDim)) : 0.0f;
			else
				inv_freq[pair] = 0.0f;
		}
		if constexpr ( G::kYarnFactor > 1.0f )
		{
			const SparkRopePlanDomain yarn = { G::kFullTheta, G::kYarnFactor, G::kYarnOriginal, G::kYarnBetaFast, G::kYarnBetaSlow, G::kFullRopeScale,
				S::kFullRopeWidth };
			memset(inv_freq, 0, sizeof(inv_freq));
			SparkRopePlanBuildYarnInvFrequency(&yarn, inv_freq);
			if ( SparkRopePlanYarnTableIsValid(&yarn, inv_freq) == 0u )
			{
				fprintf(stderr, "%s: the YaRN inverse-frequency table is invalid\n", G::Tag());
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			}
		}
		if ( SparkGpuMemcpy(state->full_inv_freq, inv_freq, sizeof(inv_freq), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
	}

	static uint32_t ReportFields(const State *state, CudaStageReportField *fields)
	{
		const CudaStageReportField report[] =
		{
			{ "arena_bytes", state->arena_bytes }, { "slots", state->slots }, { "rows", state->max_rows }, { "ffn_rows", state->ffn_rows },
			{ "expert_first", state->expert_first }
		};
		return CudaStageReportOut(fields, report);
	}
};
