#pragma once

#include "inference/kernels/dtype.cuh"
#include <math.h>
#include <stdint.h>

template<uint32_t THREADS>
static __host__ __device__ constexpr uint32_t LmBlockWarps(void) { return(THREADS < 32u ? 1u : THREADS / 32u); }

template<uint32_t THREADS>
static __host__ __device__ __forceinline__ uint32_t LmGridBlocks(uint64_t elements) { return((uint32_t)((elements + THREADS - 1u) / THREADS)); }

template<uint32_t THREADS>
static __host__ __device__ __forceinline__ uint32_t LmVecBlocks(uint64_t elements) { return((uint32_t)((((elements + 7u) >> 3u) + THREADS - 1u) / THREADS)); }

static __host__ __device__ __forceinline__ bool LmIsAligned16(const void *pointer) { return(((uintptr_t)pointer & 15u) == 0u); }
static __device__ __forceinline__ bool LmIsAligned8(const void *pointer) { return(((uintptr_t)pointer & 7u) == 0u); }

static __device__ __forceinline__ uint32_t LmPackBf16Pair(float low, float high)
{
	return((uint32_t)LmFloatToBf16(low) | ((uint32_t)LmFloatToBf16(high) << 16u));
}

static __device__ __forceinline__ void LmUnpackUint4Bf16(uint4 packed, uint16_t values[8])
{
	const uint32_t w[4] = {packed.x, packed.y, packed.z, packed.w};
	#pragma unroll
	for (uint32_t i = 0u; i < 4u; ++i) { values[2u * i] = (uint16_t)w[i]; values[2u * i + 1u] = (uint16_t)(w[i] >> 16u); }
}

static __device__ __forceinline__ uint4 LmPackUint4Bf16(const uint16_t values[8])
{
	return(make_uint4((uint32_t)values[0] | ((uint32_t)values[1] << 16u), (uint32_t)values[2] | ((uint32_t)values[3] << 16u),
		(uint32_t)values[4] | ((uint32_t)values[5] << 16u), (uint32_t)values[6] | ((uint32_t)values[7] << 16u)));
}

static __device__ __forceinline__ uint4 LmPackUint4FloatToBf16(const float values[8])
{
	return(make_uint4(LmPackBf16Pair(values[0], values[1]), LmPackBf16Pair(values[2], values[3]),
		LmPackBf16Pair(values[4], values[5]), LmPackBf16Pair(values[6], values[7])));
}

static __device__ __forceinline__ void LmLoadBf16x8AsFloat(uint4 packed, float values[8])
{
	const uint32_t w[4] = {packed.x, packed.y, packed.z, packed.w};
	#pragma unroll
	for (uint32_t i = 0u; i < 4u; ++i) { values[2u * i] = __uint_as_float(w[i] << 16u); values[2u * i + 1u] = __uint_as_float(w[i] & 0xffff0000u); }
}

static __device__ __forceinline__ void LmAddBf16x8(float acc[8], uint4 packed)
{
	float v[8];
	LmLoadBf16x8AsFloat(packed, v);
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; ++k) acc[k] += v[k];
}

static __device__ __forceinline__ void LmFmaBf16x8(float acc[8], uint4 packed, float weight)
{
	float v[8];
	LmLoadBf16x8AsFloat(packed, v);
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; ++k) acc[k] = fmaf(v[k], weight, acc[k]);
}

static __device__ __forceinline__ float LmDotBf16x8(uint4 a, uint4 b, float total)
{
	float av[8], bv[8];
	LmLoadBf16x8AsFloat(a, av);
	LmLoadBf16x8AsFloat(b, bv);
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; ++k) total = fmaf(av[k], bv[k], total);
	return(total);
}

static __device__ __forceinline__ float LmSumSqBf16x8(uint4 packed, float total)
{
	float v[8];
	LmLoadBf16x8AsFloat(packed, v);
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; ++k) total += v[k] * v[k];
	return(total);
}

static __device__ __forceinline__ void LmStoreFloat8(float *destination, const float values[8])
{
	*(float4 *)(destination) = {values[0], values[1], values[2], values[3]};
	*(float4 *)(destination + 4u) = {values[4], values[5], values[6], values[7]};
}

static __device__ __forceinline__ void LmLoadFloat8(const float *source, float values[8])
{
	const float4 lo = *(const float4 *)(source), hi = *(const float4 *)(source + 4u);
	values[0] = lo.x; values[1] = lo.y; values[2] = lo.z; values[3] = lo.w;
	values[4] = hi.x; values[5] = hi.y; values[6] = hi.z; values[7] = hi.w;
}

template<class Weight>
static __device__ __forceinline__ void LmLoadWeight8(const Weight *weight, uint32_t offset, float values[8])
{
	if constexpr ( sizeof(Weight) == 2u ) LmLoadBf16x8AsFloat(*(const uint4 *)(weight + offset), values);
	else LmLoadFloat8(weight + offset, values);
}

struct LmReduceAddOp { __device__ __forceinline__ float operator()(float a, float b) const { return(a + b); } };
struct LmReduceMaxOp { __device__ __forceinline__ float operator()(float a, float b) const { return(fmaxf(a, b)); } };

template<uint32_t THREADS, class Op>
static __device__ __forceinline__ float LmBlockReduceOnce(float value, float identity, float *shared, Op op)
{
	constexpr uint32_t lanes = THREADS < 32u ? THREADS : 32u, warps = THREADS / lanes;
	const uint32_t lane = threadIdx.x & (lanes - 1u), warp = threadIdx.x >> 5u;
	#pragma unroll
	for (uint32_t offset = lanes / 2u; offset > 0u; offset >>= 1u)
		value = op(value, __shfl_down_sync(0xffffffffu, value, offset));
	if constexpr ( warps == 1u ) return(__shfl_sync(0xffffffffu, value, 0u));
	if ( lane == 0u ) shared[warp] = value;
	__syncthreads();
	value = lane < warps ? shared[lane] : identity;
	#pragma unroll
	for (uint32_t offset = warps / 2u; offset > 0u; offset >>= 1u)
		value = op(value, __shfl_down_sync(0xffffffffu, value, offset));
	return(__shfl_sync(0xffffffffu, value, 0u));
}

template<uint32_t THREADS>
static __device__ __forceinline__ float LmBlockSumOnce(float value, float *shared) { return(LmBlockReduceOnce<THREADS>(value, 0.0f, shared, LmReduceAddOp{})); }

template<uint32_t THREADS>
static __device__ __forceinline__ float LmBlockSum(float value, float *shared)
{
	value = LmBlockSumOnce<THREADS>(value, shared);
	if constexpr ( LmBlockWarps<THREADS>() > 1u ) __syncthreads();
	return(value);
}

template<uint32_t THREADS>
static __device__ __forceinline__ void LmBlockSum2Once(float *first, float *second, float *shared)
{
	constexpr uint32_t lanes = THREADS < 32u ? THREADS : 32u, warps = THREADS / lanes;
	const uint32_t lane = threadIdx.x & (lanes - 1u), warp = threadIdx.x >> 5u;
	float a = *first, b = *second;
	#pragma unroll
	for (uint32_t offset = lanes / 2u; offset > 0u; offset >>= 1u)
	{
		a += __shfl_down_sync(0xffffffffu, a, offset);
		b += __shfl_down_sync(0xffffffffu, b, offset);
	}
	if constexpr ( warps == 1u ) { *first = __shfl_sync(0xffffffffu, a, 0u); *second = __shfl_sync(0xffffffffu, b, 0u); return; }
	if ( lane == 0u ) { shared[warp] = a; shared[warps + warp] = b; }
	__syncthreads();
	a = lane < warps ? shared[lane] : 0.0f;
	b = lane < warps ? shared[warps + lane] : 0.0f;
	#pragma unroll
	for (uint32_t offset = warps / 2u; offset > 0u; offset >>= 1u)
	{
		a += __shfl_down_sync(0xffffffffu, a, offset);
		b += __shfl_down_sync(0xffffffffu, b, offset);
	}
	*first = __shfl_sync(0xffffffffu, a, 0u);
	*second = __shfl_sync(0xffffffffu, b, 0u);
}

template<uint32_t THREADS>
static __device__ __forceinline__ float LmBlockMaxOnce(float value, float *shared) { return(LmBlockReduceOnce<THREADS>(value, -INFINITY, shared, LmReduceMaxOp{})); }

template<uint32_t THREADS>
static __device__ __forceinline__ float LmBlockMax(float value, float *shared)
{
	value = LmBlockMaxOnce<THREADS>(value, shared);
	if constexpr ( LmBlockWarps<THREADS>() > 1u ) __syncthreads();
	return(value);
}

template<uint32_t THREADS>
static __device__ __forceinline__ void LmWarpSoftmax(const float *score, float *weight, uint32_t count)
{
	constexpr uint32_t lanes = THREADS < 32u ? 1u : 32u;
	if ( threadIdx.x < lanes )
	{
		float local_max = -INFINITY, local_sum = 0.0f;
		for (uint32_t i = threadIdx.x; i < count; i += lanes) local_max = fmaxf(local_max, score[i]);
		#pragma unroll
		for (uint32_t offset = lanes / 2u; offset > 0u; offset >>= 1u) local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
		for (uint32_t i = threadIdx.x; i < count; i += lanes) { const float w = __expf(score[i] - local_max); weight[i] = w; local_sum += w; }
		#pragma unroll
		for (uint32_t offset = lanes / 2u; offset > 0u; offset >>= 1u) local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
		for (uint32_t i = threadIdx.x; i < count; i += lanes) weight[i] /= local_sum;
	}
	__syncthreads();
}

template<uint32_t THREADS>
static __device__ __forceinline__ bool LmVecElement(uint32_t dimension, uint32_t *element, uint32_t *stride = 0)
{
	if constexpr ( (THREADS & 7u) == 0u )
		if ( (uint64_t)gridDim.x * THREADS >= dimension )
		{
			if ( threadIdx.x >= (THREADS >> 3u) ) return(false);
			*element = (blockIdx.x * THREADS) + (threadIdx.x << 3u);
			if ( stride != 0 ) *stride = dimension;
			return(*element < dimension);
		}
	*element = ((blockIdx.x * THREADS) + threadIdx.x) << 3u;
	if ( stride != 0 ) *stride = (gridDim.x * THREADS) << 3u;
	return(*element < dimension);
}

template<uint32_t THREADS>
static __device__ __forceinline__ void LmCopySliceBf16(const uint16_t *source, uint16_t *destination, uint32_t width)
{
	if ( ((width & 7u) == 0u) && LmIsAligned16(source) && LmIsAligned16(destination) )
	{
		for (uint32_t index = threadIdx.x * 8u; index < width; index += THREADS * 8u)
			*(uint4 *)(destination + index) = *(const uint4 *)(source + index);
		return;
	}
	for (uint32_t index = threadIdx.x; index < width; index += THREADS) destination[index] = source[index];
}

template<uint32_t THREADS, class Op>
static __device__ __forceinline__ void LmGateUpRowBody(const uint16_t *gate, const uint16_t *up, uint16_t *output, uint32_t dimension, Op op)
{
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(gate) && LmIsAligned16(up) && LmIsAligned16(output) )
	{
		for (uint32_t index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
		{
			float g[8], u[8], out[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(gate + index), g);
			LmLoadBf16x8AsFloat(*(const uint4 *)(up + index), u);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) out[k] = op(g[k], u[k]);
			*(uint4 *)(output + index) = LmPackUint4FloatToBf16(out);
		}
		return;
	}
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS)
		output[index] = LmFloatToBf16(op(LmBf16ToFloat(gate[index]), LmBf16ToFloat(up[index])));
}

template<uint32_t THREADS = 0u, class Op>
static __device__ __forceinline__ void LmUnaryBf16GridBody(uint16_t *values, uint64_t count, Op op)
{
	const uint32_t block = THREADS != 0u ? THREADS : blockDim.x;
	const uint64_t tid = (uint64_t)blockIdx.x * block + threadIdx.x, stride = (uint64_t)gridDim.x * block;
	if ( LmIsAligned16(values) )
	{
		const uint64_t chunks = count >> 3u;
		uint4 *v4 = (uint4 *)values;
		for (uint64_t c = tid; c < chunks; c += stride)
		{
			float x[8];
			LmLoadBf16x8AsFloat(v4[c], x);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) x[k] = op(x[k]);
			v4[c] = LmPackUint4FloatToBf16(x);
		}
		for (uint64_t index = (chunks << 3u) + tid; index < count; index += stride) values[index] = LmFloatToBf16(op(LmBf16ToFloat(values[index])));
		return;
	}
	for (uint64_t index = tid; index < count; index += stride) values[index] = LmFloatToBf16(op(LmBf16ToFloat(values[index])));
}

template<uint32_t THREADS = 0u, class Op>
static __device__ __forceinline__ void LmBinaryBf16GridBody(const uint16_t *a, const uint16_t *b, uint16_t *out, uint64_t count, Op op)
{
	const uint32_t block = THREADS != 0u ? THREADS : blockDim.x;
	const uint64_t tid = (uint64_t)blockIdx.x * block + threadIdx.x, stride = (uint64_t)gridDim.x * block;
	if ( LmIsAligned16(a) && LmIsAligned16(b) && LmIsAligned16(out) )
	{
		const uint64_t chunks = count >> 3u;
		const uint4 *a4 = (const uint4 *)a, *b4 = (const uint4 *)b;
		uint4 *o4 = (uint4 *)out;
		for (uint64_t c = tid; c < chunks; c += stride)
		{
			float av[8], bv[8];
			LmLoadBf16x8AsFloat(a4[c], av);
			LmLoadBf16x8AsFloat(b4[c], bv);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) av[k] = op(av[k], bv[k]);
			o4[c] = LmPackUint4FloatToBf16(av);
		}
		for (uint64_t index = (chunks << 3u) + tid; index < count; index += stride) out[index] = LmFloatToBf16(op(LmBf16ToFloat(a[index]), LmBf16ToFloat(b[index])));
		return;
	}
	for (uint64_t index = tid; index < count; index += stride) out[index] = LmFloatToBf16(op(LmBf16ToFloat(a[index]), LmBf16ToFloat(b[index])));
}
