#pragma once

#include <stdint.h>
#include <math.h>
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/vec.cuh"

#ifdef __CUDACC__
static __global__ void LmRouteLocalKernel(uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ route_global, uint32_t routes, uint32_t first_expert, uint32_t local_experts)
{
	const uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
	if ( route >= routes ) return;
	const uint32_t expert = route_expert[route];
	route_global[route] = expert;
	if ( expert >= first_expert && expert < first_expert + local_experts ) route_expert[route] = expert - first_expert;
	else { route_expert[route] = local_experts; route_weight[route] = 0.0f; }
}

static __device__ __forceinline__ void LmPackedRouteWeight(const uint32_t *route_packed_row, const float *route_weight, float *packed_weight, uint32_t routes, uint32_t route)
{
	if ( route < routes ) packed_weight[route_packed_row[route]] = route_weight[route];
}

static __global__ void LmPackedRouteWeightKernel(const uint32_t *__restrict__ route_packed_row, const float *__restrict__ route_weight, float *__restrict__ packed_weight, uint32_t routes)
{
	LmPackedRouteWeight(route_packed_row, route_weight, packed_weight, routes, blockIdx.x * blockDim.x + threadIdx.x);
}

static __device__ __forceinline__ uint16_t LmSwigluPackedValue(uint16_t gate, uint16_t up, float weight, float limit)
{
	return(LmFloatToBf16(SparkMathGateLimitedGlu(LmBf16ToFloat(gate), LmBf16ToFloat(up), limit) * weight));
}

struct LmSwigluPackedOp { float weight, limit; __device__ __forceinline__ float operator()(float g, float u) const { return(SparkMathGateLimitedGlu(g, u, limit) * weight); } };
struct LmGeluTanhPackedOp { float weight; __device__ __forceinline__ float operator()(float g, float u) const { return(SparkMathRoundBf16(SparkMathGeluTanhScaleFirst(g)) * u * weight); } };

template<class Op>
static __device__ __forceinline__ void LmPackedGateUpBody(const uint16_t *gate, const uint16_t *up, uint16_t *output, uint32_t width, Op op)
{
	const uint64_t row_base = (uint64_t)blockIdx.y * width;
	const uint16_t *gate_row = gate + row_base, *up_row = up + row_base;
	uint16_t *output_row = output + row_base;
	const uint32_t total_threads = gridDim.x * blockDim.x;
	if ( LmIsAligned16(gate_row) && LmIsAligned16(up_row) && LmIsAligned16(output_row) && (width & 7u) == 0u && width >= total_threads * 8u )
	{
		const uint4 *g4 = (const uint4 *)gate_row, *u4 = (const uint4 *)up_row;
		uint4 *o4 = (uint4 *)output_row;
		for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < (width >> 3u); c += total_threads)
		{
			float g[8], u[8];
			LmLoadBf16x8AsFloat(g4[c], g);
			LmLoadBf16x8AsFloat(u4[c], u);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) g[k] = op(g[k], u[k]);
			o4[c] = LmPackUint4FloatToBf16(g);
		}
		return;
	}
	for (uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += total_threads)
		output_row[element] = LmFloatToBf16(op(LmBf16ToFloat(gate_row[element]), LmBf16ToFloat(up_row[element])));
}

static __global__ void LmSwigluPackedKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, const float *__restrict__ weight, uint16_t *__restrict__ output, uint32_t width, float limit)
{
	LmPackedGateUpBody(gate, up, output, width, LmSwigluPackedOp{weight[blockIdx.y], limit});
}

static __global__ void LmGeluTanhPackedKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, const float *__restrict__ weight, uint16_t *__restrict__ output, uint32_t width)
{
	LmPackedGateUpBody(gate, up, output, width, LmGeluTanhPackedOp{weight[blockIdx.y]});
}

template<uint32_t THREADS, uint32_t WIDTH>
__global__ static void LmRouterSoftmaxKernel(const uint16_t *logits, float *probs)
{
	constexpr uint32_t per_thread = (WIDTH + THREADS - 1u) / THREADS;
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	const uint64_t row_base = (uint64_t)blockIdx.x * WIDTH;
	float top = -INFINITY, sum = 0.0f;
	if constexpr ( per_thread <= 8u )
	{
		float staged[per_thread];
		#pragma unroll
		for (uint32_t slot = 0u; slot < per_thread; ++slot)
		{
			const uint32_t index = threadIdx.x + slot * THREADS;
			staged[slot] = index < WIDTH ? LmBf16ToFloat(logits[row_base + index]) : -INFINITY;
			top = fmaxf(top, staged[slot]);
		}
		top = LmBlockMax<THREADS>(top, reduction);
		#pragma unroll
		for (uint32_t slot = 0u; slot < per_thread; ++slot)
		{
			staged[slot] = threadIdx.x + slot * THREADS < WIDTH ? expf(staged[slot] - top) : 0.0f;
			sum += staged[slot];
		}
		sum = LmBlockSumOnce<THREADS>(sum, reduction);
		#pragma unroll
		for (uint32_t slot = 0u; slot < per_thread; ++slot)
			if ( threadIdx.x + slot * THREADS < WIDTH ) probs[row_base + threadIdx.x + slot * THREADS] = staged[slot] / sum;
	}
	else
	{
		for (uint32_t index = threadIdx.x; index < WIDTH; index += THREADS) top = fmaxf(top, LmBf16ToFloat(logits[row_base + index]));
		top = LmBlockMax<THREADS>(top, reduction);
		for (uint32_t index = threadIdx.x; index < WIDTH; index += THREADS) sum += expf(LmBf16ToFloat(logits[row_base + index]) - top);
		sum = LmBlockSumOnce<THREADS>(sum, reduction);
		for (uint32_t index = threadIdx.x; index < WIDTH; index += THREADS) probs[row_base + index] = expf(LmBf16ToFloat(logits[row_base + index]) - top) / sum;
	}
}

#define LM_MOE_LOCAL_MAX_TOPK 32u

template<bool WEIGHTED>
static __device__ __forceinline__ void LmMoeLocalFinalizeBody(const uint16_t *packed, const uint32_t *route_packed_row, const uint32_t *route_expert, const float *route_weight, const uint16_t *shared, uint16_t *output, uint32_t top_k, uint32_t local_experts, uint32_t width)
{
	__shared__ uint32_t active_rows[LM_MOE_LOCAL_MAX_TOPK], active_count;
	__shared__ float active_weights[WEIGHTED ? LM_MOE_LOCAL_MAX_TOPK : 1u];
	const uint32_t row = blockIdx.y;
	const uint64_t row_base = (uint64_t)row * width;
	uint16_t *out_row = output + row_base;
	const uint16_t *shared_row = shared != 0 ? shared + row_base : 0;
	if ( top_k <= LM_MOE_LOCAL_MAX_TOPK )
	{
		if ( threadIdx.x == 0u )
		{
			uint32_t count = 0u;
			for (uint32_t k = 0u; k < top_k; ++k)
			{
				const uint32_t route = row * top_k + k;
				if ( route_expert[route] < local_experts ) { active_rows[count] = route_packed_row[route]; if constexpr ( WEIGHTED ) active_weights[count] = route_weight[route]; ++count; }
			}
			active_count = count;
		}
		__syncthreads();
		const uint32_t count = active_count;
		if ( (width & 7u) == 0u && LmIsAligned16(packed) && LmIsAligned16(out_row) && (shared_row == 0 || LmIsAligned16(shared_row)) )
		{
			const uint32_t chunks = width >> 3u;
			const uint4 *packed4 = (const uint4 *)packed, *shared4 = (const uint4 *)shared_row;
			uint4 *out4 = (uint4 *)out_row;
			for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < chunks; c += gridDim.x * blockDim.x)
			{
				float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
				for (uint32_t k = 0u; k < count; ++k)
				{
					const uint4 v = packed4[(uint64_t)active_rows[k] * chunks + c];
					if constexpr ( WEIGHTED ) LmFmaBf16x8(acc, v, active_weights[k]);
					else LmAddBf16x8(acc, v);
				}
				if ( shared4 != 0 ) LmAddBf16x8(acc, shared4[c]);
				out4[c] = LmPackUint4FloatToBf16(acc);
			}
			return;
		}
		for (uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x)
		{
			float total = 0.0f;
			for (uint32_t k = 0u; k < count; ++k)
			{
				const float v = LmBf16ToFloat(packed[(uint64_t)active_rows[k] * width + element]);
				total = WEIGHTED ? fmaf(v, active_weights[k], total) : total + v;
			}
			if ( shared_row != 0 ) total += LmBf16ToFloat(shared_row[element]);
			out_row[element] = LmFloatToBf16(total);
		}
		return;
	}
	for (uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x)
	{
		float total = 0.0f;
		for (uint32_t k = 0u; k < top_k; ++k)
		{
			const uint32_t route = row * top_k + k;
			if ( route_expert[route] < local_experts )
			{
				const float v = LmBf16ToFloat(packed[(uint64_t)route_packed_row[route] * width + element]);
				total = WEIGHTED ? fmaf(v, route_weight[route], total) : total + v;
			}
		}
		if ( shared_row != 0 ) total += LmBf16ToFloat(shared_row[element]);
		out_row[element] = LmFloatToBf16(total);
	}
}

static __global__ void LmMoeLocalWeightedFinalizeKernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row, const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, const uint16_t *__restrict__ shared, uint16_t *__restrict__ output, uint32_t top_k, uint32_t local_experts, uint32_t width)
{
	LmMoeLocalFinalizeBody<true>(packed, route_packed_row, route_expert, route_weight, shared, output, top_k, local_experts, width);
}

static __global__ void LmMoeLocalFinalizeKernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row, const uint32_t *__restrict__ route_expert, const uint16_t *__restrict__ shared, uint16_t *__restrict__ output, uint32_t top_k, uint32_t local_experts, uint32_t width)
{
	LmMoeLocalFinalizeBody<false>(packed, route_packed_row, route_expert, (const float *)0, shared, output, top_k, local_experts, width);
}
#endif
