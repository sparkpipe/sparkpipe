#pragma once

#include <stdint.h>
#include <math.h>

#include "inference/kernels/dtype.cuh"

#ifdef __CUDACC__
static __global__ void LmRouteLocalKernel(uint32_t *__restrict__ route_expert, float *__restrict__ route_weight,
	uint32_t *__restrict__ route_global, uint32_t routes, uint32_t first_expert, uint32_t local_experts)
{
	const uint32_t route = blockIdx.x * blockDim.x + threadIdx.x;
	if ( route >= routes )
		return;
	const uint32_t expert = route_expert[route];
	route_global[route] = expert;
	if ( expert >= first_expert && expert < first_expert + local_experts )
		route_expert[route] = expert - first_expert;
	else
	{
		route_expert[route] = local_experts;
		route_weight[route] = 0.0f;
	}
}

static __device__ __forceinline__ void LmPackedRouteWeight(const uint32_t *route_packed_row, const float *route_weight, float *packed_weight,
	uint32_t routes, uint32_t route)
{
	if ( route < routes )
		packed_weight[route_packed_row[route]] = route_weight[route];
}

static __global__ void LmPackedRouteWeightKernel(const uint32_t *__restrict__ route_packed_row, const float *__restrict__ route_weight,
	float *__restrict__ packed_weight, uint32_t routes)
{
	LmPackedRouteWeight(route_packed_row, route_weight, packed_weight, routes, blockIdx.x * blockDim.x + threadIdx.x);
}

static __device__ __forceinline__ uint16_t LmSwigluPackedValue(uint16_t gate, uint16_t up, float weight, float limit)
{
	const float g = fminf(LmBf16ToFloat(gate), limit);
	const float u = fminf(fmaxf(LmBf16ToFloat(up), -limit), limit);
	return LmFloatToBf16(SparkMathSilu(g) * u * weight);
}

static __device__ __forceinline__ uint32_t LmSwigluPackedPair(uint32_t gate, uint32_t up, float weight, float limit)
{
	const uint16_t low = LmSwigluPackedValue((uint16_t)gate, (uint16_t)up, weight, limit);
	const uint16_t high = LmSwigluPackedValue((uint16_t)(gate >> 16u), (uint16_t)(up >> 16u), weight, limit);
	return((uint32_t)low | ((uint32_t)high << 16u));
}

static __global__ void LmSwigluPackedKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, const float *__restrict__ weight,
	uint16_t *__restrict__ output, uint32_t width, float limit)
{
	const uint32_t row = blockIdx.y;
	const float row_weight = weight[row];
	const uint64_t row_base = (uint64_t)row * width;
	const uint16_t *gate_row = gate + row_base, *up_row = up + row_base;
	uint16_t *output_row = output + row_base;
	const uint32_t total_threads = gridDim.x * blockDim.x;
	if ( (((uintptr_t)gate_row | (uintptr_t)up_row | (uintptr_t)output_row) & 15u) == 0u && (width & 7u) == 0u && width >= total_threads * 8u )
	{
		const uint32_t chunks = width >> 3u;
		const uint4 *g4 = (const uint4 *)gate_row, *u4 = (const uint4 *)up_row;
		uint4 *o4 = (uint4 *)output_row;
		for ( uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < chunks; c += total_threads )
		{
			const uint4 gv = g4[c], uv = u4[c];
			o4[c] = make_uint4(
				LmSwigluPackedPair(gv.x, uv.x, row_weight, limit),
				LmSwigluPackedPair(gv.y, uv.y, row_weight, limit),
				LmSwigluPackedPair(gv.z, uv.z, row_weight, limit),
				LmSwigluPackedPair(gv.w, uv.w, row_weight, limit));
		}
		return;
	}
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += total_threads )
		output_row[element] = LmSwigluPackedValue(gate_row[element], up_row[element], row_weight, limit);
}

static __device__ __forceinline__ uint16_t LmGeluTanhPackedValue(uint16_t gate, uint16_t up, float weight)
{
	const float g = LmBf16ToFloat(gate);
	const float activated = LmBf16ToFloat(LmFloatToBf16(SparkMathGeluTanhScaleFirst(g)));
	return(LmFloatToBf16(activated * LmBf16ToFloat(up) * weight));
}

static __device__ __forceinline__ uint32_t LmGeluTanhPackedPair(uint32_t gate, uint32_t up, float weight)
{
	const uint16_t low = LmGeluTanhPackedValue((uint16_t)gate, (uint16_t)up, weight);
	const uint16_t high = LmGeluTanhPackedValue((uint16_t)(gate >> 16u), (uint16_t)(up >> 16u), weight);
	return((uint32_t)low | ((uint32_t)high << 16u));
}

static __global__ void LmGeluTanhPackedKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, const float *__restrict__ weight,
	uint16_t *__restrict__ output, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	const float row_weight = weight[row];
	const uint64_t row_base = (uint64_t)row * width;
	const uint16_t *gate_row = gate + row_base, *up_row = up + row_base;
	uint16_t *output_row = output + row_base;
	const uint32_t total_threads = gridDim.x * blockDim.x;
	if ( (((uintptr_t)gate_row | (uintptr_t)up_row | (uintptr_t)output_row) & 15u) == 0u && (width & 7u) == 0u && width >= total_threads * 8u )
	{
		const uint32_t chunks = width >> 3u;
		const uint4 *g4 = (const uint4 *)gate_row, *u4 = (const uint4 *)up_row;
		uint4 *o4 = (uint4 *)output_row;
		for ( uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < chunks; c += total_threads )
		{
			const uint4 gv = g4[c], uv = u4[c];
			o4[c] = make_uint4(
				LmGeluTanhPackedPair(gv.x, uv.x, row_weight),
				LmGeluTanhPackedPair(gv.y, uv.y, row_weight),
				LmGeluTanhPackedPair(gv.z, uv.z, row_weight),
				LmGeluTanhPackedPair(gv.w, uv.w, row_weight));
		}
		return;
	}
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += total_threads )
		output_row[element] = LmGeluTanhPackedValue(gate_row[element], up_row[element], row_weight);
}

template<uint32_t THREADS, uint32_t WIDTH>
__global__ static void LmRouterSoftmaxKernel(const uint16_t *logits, float *probs)
{
	constexpr uint32_t warps = THREADS / 32u;
	constexpr uint32_t per_thread = (WIDTH + THREADS - 1u) / THREADS;
	__shared__ float reduce_max[warps];
	__shared__ float reduce_sum[warps];
	const uint32_t row = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	const uint64_t row_base = (uint64_t)row * WIDTH;
	if constexpr ( per_thread <= 8u )
	{
		float staged[per_thread];
		float top = -INFINITY, sum = 0.0f;
		#pragma unroll
		for ( uint32_t slot = 0u; slot < per_thread; ++slot )
		{
			const uint32_t index = threadIdx.x + slot * THREADS;
			const float value = index < WIDTH ? LmBf16ToFloat(logits[row_base + index]) : -INFINITY;
			staged[slot] = value;
			top = fmaxf(top, value);
		}
		#pragma unroll
		for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
			top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset));
		if constexpr ( warps > 1u )
		{
			if ( lane == 0u )
				reduce_max[warp] = top;
			__syncthreads();
			top = lane < warps ? reduce_max[lane] : -INFINITY;
			#pragma unroll
			for ( uint32_t offset = warps / 2u; offset > 0u; offset >>= 1u )
				top = fmaxf(top, __shfl_down_sync(0xffffffffu, top, offset));
			top = __shfl_sync(0xffffffffu, top, 0u);
		}
		#pragma unroll
		for ( uint32_t slot = 0u; slot < per_thread; ++slot )
		{
			const uint32_t index = threadIdx.x + slot * THREADS;
			const float value = index < WIDTH ? expf(staged[slot] - top) : 0.0f;
			staged[slot] = value;
			sum += value;
		}
		#pragma unroll
		for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
			sum += __shfl_xor_sync(0xffffffffu, sum, offset);
		if constexpr ( warps > 1u )
		{
			if ( lane == 0u )
				reduce_sum[warp] = sum;
			__syncthreads();
			sum = lane < warps ? reduce_sum[lane] : 0.0f;
			#pragma unroll
			for ( uint32_t offset = warps / 2u; offset > 0u; offset >>= 1u )
				sum += __shfl_down_sync(0xffffffffu, sum, offset);
			sum = __shfl_sync(0xffffffffu, sum, 0u);
		}
		#pragma unroll
		for ( uint32_t slot = 0u; slot < per_thread; ++slot )
		{
			const uint32_t index = threadIdx.x + slot * THREADS;
			if ( index < WIDTH )
				probs[row_base + index] = staged[slot] / sum;
		}
		return;
	}
	float top = -INFINITY, sum = 0.0f;
	for ( uint32_t index = threadIdx.x; index < WIDTH; index += blockDim.x )
		top = fmaxf(top, LmBf16ToFloat(logits[row_base + index]));
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset));
	if constexpr ( warps > 1u )
	{
		if ( lane == 0u )
			reduce_max[warp] = top;
		__syncthreads();
		top = lane < warps ? reduce_max[lane] : -INFINITY;
		for ( uint32_t offset = warps / 2u; offset > 0u; offset >>= 1u )
			top = fmaxf(top, __shfl_down_sync(0xffffffffu, top, offset));
		top = __shfl_sync(0xffffffffu, top, 0u);
	}
	for ( uint32_t index = threadIdx.x; index < WIDTH; index += blockDim.x )
		sum += expf(LmBf16ToFloat(logits[row_base + index]) - top);
	for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		sum += __shfl_xor_sync(0xffffffffu, sum, offset);
	if constexpr ( warps > 1u )
	{
		if ( lane == 0u )
			reduce_sum[warp] = sum;
		__syncthreads();
		sum = lane < warps ? reduce_sum[lane] : 0.0f;
		for ( uint32_t offset = warps / 2u; offset > 0u; offset >>= 1u )
			sum += __shfl_down_sync(0xffffffffu, sum, offset);
		sum = __shfl_sync(0xffffffffu, sum, 0u);
	}
	for ( uint32_t index = threadIdx.x; index < WIDTH; index += blockDim.x )
		probs[row_base + index] = expf(LmBf16ToFloat(logits[row_base + index]) - top) / sum;
}

#define LM_MOE_LOCAL_MAX_TOPK 32u

static __global__ void LmMoeLocalWeightedFinalizeKernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row,
	const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, const uint16_t *__restrict__ shared, uint16_t *__restrict__ output,
	uint32_t top_k, uint32_t local_experts, uint32_t width)
{
	__shared__ uint32_t active_rows[LM_MOE_LOCAL_MAX_TOPK];
	__shared__ float active_weights[LM_MOE_LOCAL_MAX_TOPK];
	__shared__ uint32_t active_count;
	const uint32_t row = blockIdx.y;
	if ( top_k <= LM_MOE_LOCAL_MAX_TOPK )
	{
		if ( threadIdx.x == 0u )
		{
			uint32_t count = 0u;
			for ( uint32_t k = 0u; k < top_k; ++k )
			{
				const uint32_t route = row * top_k + k;
				if ( route_expert[route] < local_experts )
				{
					active_rows[count] = route_packed_row[route];
					active_weights[count] = route_weight[route];
					++count;
				}
			}
			active_count = count;
		}
		__syncthreads();
		const uint32_t count = active_count;
		const uint64_t row_base = (uint64_t)row * width;
		uint16_t *out_row = output + row_base;
		const uint16_t *shared_row = shared != 0 ? shared + row_base : 0;
		if ( (width & 7u) == 0u && (((uintptr_t)packed | (uintptr_t)out_row | (uintptr_t)shared_row) & 15u) == 0u )
		{
			const uint32_t chunks = width >> 3u;
			const uint4 *packed4 = (const uint4 *)packed;
			const uint4 *shared4 = (const uint4 *)shared_row;
			uint4 *out4 = (uint4 *)out_row;
			for ( uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < chunks; c += gridDim.x * blockDim.x )
			{
				float acc[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
				for ( uint32_t k = 0u; k < count; ++k )
				{
					const uint4 v = packed4[(uint64_t)active_rows[k] * chunks + c];
					const float w = active_weights[k];
					acc[0] = fmaf(__uint_as_float(v.x << 16u), w, acc[0]);
					acc[1] = fmaf(__uint_as_float(v.x & 0xffff0000u), w, acc[1]);
					acc[2] = fmaf(__uint_as_float(v.y << 16u), w, acc[2]);
					acc[3] = fmaf(__uint_as_float(v.y & 0xffff0000u), w, acc[3]);
					acc[4] = fmaf(__uint_as_float(v.z << 16u), w, acc[4]);
					acc[5] = fmaf(__uint_as_float(v.z & 0xffff0000u), w, acc[5]);
					acc[6] = fmaf(__uint_as_float(v.w << 16u), w, acc[6]);
					acc[7] = fmaf(__uint_as_float(v.w & 0xffff0000u), w, acc[7]);
				}
				if ( shared4 != 0 )
				{
					const uint4 s = shared4[c];
					acc[0] += __uint_as_float(s.x << 16u);
					acc[1] += __uint_as_float(s.x & 0xffff0000u);
					acc[2] += __uint_as_float(s.y << 16u);
					acc[3] += __uint_as_float(s.y & 0xffff0000u);
					acc[4] += __uint_as_float(s.z << 16u);
					acc[5] += __uint_as_float(s.z & 0xffff0000u);
					acc[6] += __uint_as_float(s.w << 16u);
					acc[7] += __uint_as_float(s.w & 0xffff0000u);
				}
				out4[c] = make_uint4(
					(uint32_t)LmFloatToBf16(acc[0]) | ((uint32_t)LmFloatToBf16(acc[1]) << 16u),
					(uint32_t)LmFloatToBf16(acc[2]) | ((uint32_t)LmFloatToBf16(acc[3]) << 16u),
					(uint32_t)LmFloatToBf16(acc[4]) | ((uint32_t)LmFloatToBf16(acc[5]) << 16u),
					(uint32_t)LmFloatToBf16(acc[6]) | ((uint32_t)LmFloatToBf16(acc[7]) << 16u));
			}
			return;
		}
		for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
		{
			float total = 0.0f;
			for ( uint32_t k = 0u; k < count; ++k )
				total = fmaf(LmBf16ToFloat(packed[(uint64_t)active_rows[k] * width + element]), active_weights[k], total);
			if ( shared_row != 0 )
				total += LmBf16ToFloat(shared_row[element]);
			out_row[element] = LmFloatToBf16(total);
		}
		return;
	}
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		float total = 0.0f;
		for ( uint32_t k = 0u; k < top_k; ++k )
		{
			const uint32_t route = row * top_k + k;
			if ( route_expert[route] < local_experts )
				total = fmaf(LmBf16ToFloat(packed[(uint64_t)route_packed_row[route] * width + element]), route_weight[route], total);
		}
		if ( shared != 0 )
			total += LmBf16ToFloat(shared[(uint64_t)row * width + element]);
		output[(uint64_t)row * width + element] = LmFloatToBf16(total);
	}
}

static __global__ void LmMoeLocalFinalizeKernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row,
	const uint32_t *__restrict__ route_expert, const uint16_t *__restrict__ shared, uint16_t *__restrict__ output, uint32_t top_k,
	uint32_t local_experts, uint32_t width)
{
	__shared__ uint32_t active_rows[LM_MOE_LOCAL_MAX_TOPK];
	__shared__ uint32_t active_count;
	const uint32_t row = blockIdx.y;
	if ( top_k <= LM_MOE_LOCAL_MAX_TOPK )
	{
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
		const uint64_t row_base = (uint64_t)row * width;
		uint16_t *out_row = output + row_base;
		const uint16_t *shared_row = shared != 0 ? shared + row_base : 0;
		if ( (width & 7u) == 0u && (((uintptr_t)packed | (uintptr_t)out_row | (uintptr_t)shared_row) & 15u) == 0u )
		{
			const uint32_t chunks = width >> 3u;
			const uint4 *packed4 = (const uint4 *)packed;
			const uint4 *shared4 = (const uint4 *)shared_row;
			uint4 *out4 = (uint4 *)out_row;
			for ( uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < chunks; c += gridDim.x * blockDim.x )
			{
				float acc[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
				for ( uint32_t k = 0u; k < count; ++k )
				{
					const uint4 v = packed4[(uint64_t)active_rows[k] * chunks + c];
					acc[0] += __uint_as_float(v.x << 16u);
					acc[1] += __uint_as_float(v.x & 0xffff0000u);
					acc[2] += __uint_as_float(v.y << 16u);
					acc[3] += __uint_as_float(v.y & 0xffff0000u);
					acc[4] += __uint_as_float(v.z << 16u);
					acc[5] += __uint_as_float(v.z & 0xffff0000u);
					acc[6] += __uint_as_float(v.w << 16u);
					acc[7] += __uint_as_float(v.w & 0xffff0000u);
				}
				if ( shared4 != 0 )
				{
					const uint4 s = shared4[c];
					acc[0] += __uint_as_float(s.x << 16u);
					acc[1] += __uint_as_float(s.x & 0xffff0000u);
					acc[2] += __uint_as_float(s.y << 16u);
					acc[3] += __uint_as_float(s.y & 0xffff0000u);
					acc[4] += __uint_as_float(s.z << 16u);
					acc[5] += __uint_as_float(s.z & 0xffff0000u);
					acc[6] += __uint_as_float(s.w << 16u);
					acc[7] += __uint_as_float(s.w & 0xffff0000u);
				}
				out4[c] = make_uint4(
					(uint32_t)LmFloatToBf16(acc[0]) | ((uint32_t)LmFloatToBf16(acc[1]) << 16u),
					(uint32_t)LmFloatToBf16(acc[2]) | ((uint32_t)LmFloatToBf16(acc[3]) << 16u),
					(uint32_t)LmFloatToBf16(acc[4]) | ((uint32_t)LmFloatToBf16(acc[5]) << 16u),
					(uint32_t)LmFloatToBf16(acc[6]) | ((uint32_t)LmFloatToBf16(acc[7]) << 16u));
			}
			return;
		}
		for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
		{
			float total = 0.0f;
			for ( uint32_t k = 0u; k < count; ++k )
				total += LmBf16ToFloat(packed[(uint64_t)active_rows[k] * width + element]);
			if ( shared_row != 0 )
				total += LmBf16ToFloat(shared_row[element]);
			out_row[element] = LmFloatToBf16(total);
		}
		return;
	}
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		float total = 0.0f;
		for ( uint32_t k = 0u; k < top_k; ++k )
		{
			const uint32_t route = row * top_k + k;
			if ( route_expert[route] < local_experts )
				total += LmBf16ToFloat(packed[(uint64_t)route_packed_row[route] * width + element]);
		}
		if ( shared != 0 )
			total += LmBf16ToFloat(shared[(uint64_t)row * width + element]);
		output[(uint64_t)row * width + element] = LmFloatToBf16(total);
	}
}
#endif
