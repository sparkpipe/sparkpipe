#pragma once

#include <math.h>
#include <stdint.h>

#include "dtype.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "runtime/launch.h"

#define LM_HC_MIX_ROWS 4u
#define LM_HC_MIX_LANES 256u
#define LM_HC_MIX_SLICE 32u
#define LM_HC_MIX_LOADERS 128u
#define LM_HC_MIX_SPLIT_ROWS 16u
#define LM_HC_SINKHORN_THREADS 256u
#define LM_HC_SINKHORN_WARPS (LM_HC_SINKHORN_THREADS / 32u)
#define LM_HC_SINKHORN_UNROLL 16u
#define LM_HC_SINKHORN_ROWS 4u
#define LM_HC_SINKHORN_ROWS_FROM 32u

template<uint32_t STREAMS>
__global__ void LmHcExpandFlatKernel(const uint16_t *hidden, uint16_t *streams, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < STREAMS * width; index += gridDim.x * blockDim.x )
		streams[(uint64_t)row * STREAMS * width + index] = hidden[(uint64_t)row * width + index % width];
}

template<uint32_t STREAMS, uint32_t MIXES>
static __device__ __forceinline__ void LmHcMixGates(const float *total, uint32_t flat, const float *base, const float *scale, float norm_epsilon,
	float hc_epsilon, float post_magnitude, float *weights, float *post)
{
	const float inverse = SparkMathRmsScale(total[MIXES], (float)flat, norm_epsilon);
	for ( uint32_t s = 0u; s < STREAMS; ++s )
	{
		weights[s] = SparkMathSigmoid(total[s] * inverse * scale[0] + base[s]) + hc_epsilon;
		if ( MIXES > STREAMS && post != 0 )
			post[s] = SparkMathScaledSigmoid(post_magnitude,total[STREAMS + s] * inverse * scale[1] + base[STREAMS + s]) + hc_epsilon;
	}
}

template<uint32_t THREADS, uint32_t STREAMS, uint32_t MIXES>
__global__ __launch_bounds__(THREADS, 1)
void LmHcMixKernel(const uint16_t *streams, const float *fn, const float *base, const float *scale, float *post, uint16_t *mixed,
	uint32_t rows, uint32_t width, float norm_epsilon, float hc_epsilon, float post_magnitude)
{
	constexpr uint32_t WARPS = THREADS / 32u;
	__shared__ float reduction[LM_HC_MIX_ROWS][MIXES + 1u][WARPS];
	__shared__ float weights[LM_HC_MIX_ROWS][STREAMS];
	const uint32_t first = blockIdx.x * LM_HC_MIX_ROWS, flat = STREAMS * width, lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	float dot[LM_HC_MIX_ROWS][MIXES + 1u];
	for ( uint32_t r = 0u; r < LM_HC_MIX_ROWS; ++r )
		for ( uint32_t m = 0u; m <= MIXES; ++m )
			dot[r][m] = 0.0f;
	for ( uint32_t k = threadIdx.x; k < flat; k += THREADS )
	{
		float coefficient[MIXES];
		for ( uint32_t m = 0u; m < MIXES; ++m )
			coefficient[m] = fn[(uint64_t)m * flat + k];
		for ( uint32_t r = 0u; r < LM_HC_MIX_ROWS; ++r )
		{
			if ( first + r >= rows )
				break;
			const float x = LmBf16ToFloat(streams[(uint64_t)(first + r) * flat + k]);
			for ( uint32_t m = 0u; m < MIXES; ++m )
				dot[r][m] = fmaf(x, coefficient[m], dot[r][m]);
			dot[r][MIXES] = fmaf(x, x, dot[r][MIXES]);
		}
	}
	for ( uint32_t r = 0u; r < LM_HC_MIX_ROWS; ++r )
		for ( uint32_t m = 0u; m <= MIXES; ++m )
		{
			float value = dot[r][m];
			for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
				value += __shfl_xor_sync(0xffffffffu, value, offset);
			if ( lane == 0u )
				reduction[r][m][warp] = value;
		}
	__syncthreads();
	if ( threadIdx.x < LM_HC_MIX_ROWS && first + threadIdx.x < rows )
	{
		const uint32_t r = threadIdx.x;
		float total[MIXES + 1u];
		for ( uint32_t m = 0u; m <= MIXES; ++m )
		{
			total[m] = 0.0f;
			for ( uint32_t w = 0u; w < WARPS; ++w )
				total[m] += reduction[r][m][w];
		}
		LmHcMixGates<STREAMS,MIXES>(total, flat, base, scale, norm_epsilon, hc_epsilon, post_magnitude, weights[r],
			MIXES > STREAMS ? post + (uint64_t)(first + r) * STREAMS : (float *)0);
	}
	__syncthreads();
	for ( uint32_t r = 0u; r < LM_HC_MIX_ROWS; ++r )
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
__global__ __launch_bounds__(LM_HC_MIX_LOADERS)
void LmHcMixDotKernel(const uint16_t *streams, const float *fn, float *partial, uint32_t mixes)
{
	constexpr uint32_t FLAT = STREAMS * WIDTH, STEPS = FLAT / LM_HC_MIX_LANES;
	static_assert(FLAT % LM_HC_MIX_LANES == 0u, "every lane walks the same number of columns");
	__shared__ float coefficient[STEPS][LM_HC_MIX_SLICE];
	__shared__ float value[STEPS][LM_HC_MIX_SLICE];
	const uint32_t mix = blockIdx.x, lane_first = blockIdx.y * LM_HC_MIX_SLICE, row = blockIdx.z;
	const uint16_t *source = streams + (uint64_t)row * FLAT;
	for ( uint32_t index = threadIdx.x; index < STEPS * LM_HC_MIX_SLICE; index += LM_HC_MIX_LOADERS )
	{
		const uint32_t step = index / LM_HC_MIX_SLICE, lane = index % LM_HC_MIX_SLICE, k = step * LM_HC_MIX_LANES + lane_first + lane;
		value[step][lane] = LmBf16ToFloat(source[k]);
		coefficient[step][lane] = mix < mixes - 1u ? fn[(uint64_t)mix * FLAT + k] : 0.0f;
	}
	__syncthreads();
	if ( threadIdx.x < LM_HC_MIX_SLICE )
	{
		float dot = 0.0f;
		if ( mix < mixes - 1u )
			for ( uint32_t step = 0u; step < STEPS; ++step )
				dot = fmaf(value[step][threadIdx.x], coefficient[step][threadIdx.x], dot);
		else
			for ( uint32_t step = 0u; step < STEPS; ++step )
				dot = fmaf(value[step][threadIdx.x], value[step][threadIdx.x], dot);
		partial[((uint64_t)row * mixes + mix) * LM_HC_MIX_LANES + lane_first + threadIdx.x] = dot;
	}
}

template<uint32_t STREAMS, uint32_t MIXES, uint32_t WIDTH>
__global__ __launch_bounds__(LM_HC_MIX_LANES, 1)
void LmHcMixRowKernel(const uint16_t *streams, const float *partial, const float *base, const float *scale, float *post, uint16_t *mixed,
	float norm_epsilon, float hc_epsilon, float post_magnitude)
{
	constexpr uint32_t WARPS = LM_HC_MIX_LANES / 32u, FLAT = STREAMS * WIDTH;
	__shared__ float reduction[MIXES + 1u][WARPS];
	__shared__ float weights[STREAMS];
	const uint32_t row = blockIdx.y, lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	for ( uint32_t m = 0u; m <= MIXES; ++m )
	{
		float value = partial[((uint64_t)row * (MIXES + 1u) + m) * LM_HC_MIX_LANES + threadIdx.x];
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
		LmHcMixGates<STREAMS,MIXES>(total, FLAT, base, scale, norm_epsilon, hc_epsilon, post_magnitude, weights,
			MIXES > STREAMS && blockIdx.x == 0u ? post + (uint64_t)row * STREAMS : (float *)0);
	}
	__syncthreads();
	for ( uint32_t d = blockIdx.x * LM_HC_MIX_LANES + threadIdx.x; d < WIDTH; d += gridDim.x * LM_HC_MIX_LANES )
	{
		float total = 0.0f;
		for ( uint32_t s = 0u; s < STREAMS; ++s )
			total = fmaf(weights[s], LmBf16ToFloat(streams[(uint64_t)row * FLAT + s * WIDTH + d]), total);
		mixed[(uint64_t)row * WIDTH + d] = LmFloatToBf16(total);
	}
}

static inline uint64_t LmHcMixPartialBytes(uint64_t rows, uint32_t mixes)
{
	return (rows < LM_HC_MIX_SPLIT_ROWS ? rows : LM_HC_MIX_SPLIT_ROWS) * (mixes + 1u) * LM_HC_MIX_LANES * sizeof(float);
}

template<uint32_t THREADS, uint32_t STREAMS, uint32_t MIXES, uint32_t WIDTH>
static void LmHcMix(const uint16_t *streams, const float *fn, const float *scale, const float *base, float *partial, float *post, uint16_t *mixed,
	uint32_t rows, float norm_epsilon, float hc_epsilon, float magnitude, cudaStream_t stream)
{
	if ( rows <= LM_HC_MIX_SPLIT_ROWS )
	{
		LM_LAUNCH((LmHcMixDotKernel<STREAMS,WIDTH>), dim3(MIXES + 1u, LM_HC_MIX_LANES / LM_HC_MIX_SLICE, rows), LM_HC_MIX_LOADERS, 0, stream, streams, fn,
			partial, MIXES + 1u);
		LM_LAUNCH((LmHcMixRowKernel<STREAMS,MIXES,WIDTH>), dim3((WIDTH + LM_HC_MIX_LANES - 1u) / LM_HC_MIX_LANES, rows), LM_HC_MIX_LANES, 0, stream, streams,
			partial, base, scale, post, mixed, norm_epsilon, hc_epsilon, magnitude);
		return;
	}
	LM_LAUNCH((LmHcMixKernel<THREADS,STREAMS,MIXES>), (rows + LM_HC_MIX_ROWS - 1u) / LM_HC_MIX_ROWS, THREADS, 0, stream, streams, fn, base, scale, post, mixed,
		rows, WIDTH, norm_epsilon, hc_epsilon, magnitude);
}

template<uint32_t STREAMS>
__global__ void LmHcPostKernel(uint16_t *streams, const float *post, const uint16_t *output, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < STREAMS * width; index += gridDim.x * blockDim.x )
	{
		const uint64_t at = (uint64_t)row * STREAMS * width + index;
		streams[at] = LmFloatToBf16(fmaf(post[(uint64_t)row * STREAMS + index / width], LmBf16ToFloat(output[(uint64_t)row * width + index % width]),
			LmBf16ToFloat(streams[at])));
	}
}

template<uint32_t STREAMS>
struct LmHcSinkhornShape
{
	static constexpr uint32_t kMix = (2u + STREAMS) * STREAMS;
	static constexpr uint32_t kTasks = kMix + 1u;
	static constexpr uint32_t kPartials = kTasks * LM_HC_SINKHORN_WARPS;
};

template<uint32_t STREAMS>
__global__ void LmHcSinkhornPartialKernel(const uint16_t *__restrict__ streams, const float *__restrict__ fn, float *__restrict__ partials,
	uint32_t width)
{
	using S = LmHcSinkhornShape<STREAMS>;
	const uint32_t task = blockIdx.x, row = blockIdx.y, flat = STREAMS * width, lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	const uint16_t *source = streams + (uint64_t)row * flat;
	const float *weight = task < S::kMix ? fn + (uint64_t)task * flat : 0;
	float total = 0.0f;
	for ( uint32_t base = threadIdx.x; base < flat; base += LM_HC_SINKHORN_UNROLL * LM_HC_SINKHORN_THREADS )
	{
		float value[LM_HC_SINKHORN_UNROLL], factor[LM_HC_SINKHORN_UNROLL];
#pragma unroll
		for ( uint32_t step = 0u; step < LM_HC_SINKHORN_UNROLL; ++step )
		{
			const uint32_t element = base + step * LM_HC_SINKHORN_THREADS;
			value[step] = element < flat ? LmBf16ToFloat(source[element]) : 0.0f;
			factor[step] = element >= flat ? 0.0f : weight != 0 ? weight[element] : value[step];
		}
#pragma unroll
		for ( uint32_t step = 0u; step < LM_HC_SINKHORN_UNROLL; ++step )
			if ( base + step * LM_HC_SINKHORN_THREADS < flat )
				total += factor[step] * value[step];
	}
	for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
		total += __shfl_down_sync(0xffffffffu, total, offset);
	if ( lane == 0u )
		partials[((uint64_t)row * S::kTasks + task) * LM_HC_SINKHORN_WARPS + warp] = total;
}

template<uint32_t STREAMS, uint32_t ROWS>
__global__ void LmHcSinkhornPartialRowsKernel(const uint16_t *__restrict__ streams, const float *__restrict__ fn, float *__restrict__ partials,
	uint32_t width, uint32_t rows)
{
	using S = LmHcSinkhornShape<STREAMS>;
	const uint32_t first_row = blockIdx.x * ROWS, flat = STREAMS * width, lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	float total[S::kTasks][ROWS];
	for ( uint32_t task = 0u; task < S::kTasks; ++task )
		for ( uint32_t r = 0u; r < ROWS; ++r )
			total[task][r] = 0.0f;
	for ( uint32_t element = threadIdx.x; element < flat; element += LM_HC_SINKHORN_THREADS )
	{
		float value[ROWS];
		for ( uint32_t r = 0u; r < ROWS; ++r )
			value[r] = first_row + r < rows ? LmBf16ToFloat(streams[(uint64_t)(first_row + r) * flat + element]) : 0.0f;
		for ( uint32_t task = 0u; task < S::kMix; ++task )
		{
			const float factor = fn[(uint64_t)task * flat + element];
			for ( uint32_t r = 0u; r < ROWS; ++r )
				total[task][r] += factor * value[r];
		}
		for ( uint32_t r = 0u; r < ROWS; ++r )
			total[S::kMix][r] += value[r] * value[r];
	}
	for ( uint32_t task = 0u; task < S::kTasks; ++task )
		for ( uint32_t r = 0u; r < ROWS; ++r )
		{
			float sum = total[task][r];
			for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
				sum += __shfl_down_sync(0xffffffffu, sum, offset);
			if ( lane == 0u && first_row + r < rows )
				partials[((uint64_t)(first_row + r) * S::kTasks + task) * LM_HC_SINKHORN_WARPS + warp] = sum;
		}
}

template<uint32_t STREAMS>
static __device__ __forceinline__ float LmHcSinkhornLineSum(float value, uint32_t first, uint32_t stride)
{
	float sum = 0.0f;
	for ( uint32_t index = 0u; index < STREAMS; ++index )
		sum += __shfl_sync(0xffffffffu, value, first + index * stride);
	return sum;
}

template<uint32_t STREAMS>
__global__ void LmHcSinkhornKernel(const float *__restrict__ partials, const float *__restrict__ scale, const float *__restrict__ base,
	float *__restrict__ pre, float *__restrict__ post, float *__restrict__ comb_out, uint32_t width, float rms_epsilon, uint32_t iterations,
	float epsilon)
{
	using S = LmHcSinkhornShape<STREAMS>;
	__shared__ float mix[S::kTasks];
	const uint32_t row = blockIdx.x, lane = threadIdx.x, r = lane / STREAMS, c = lane % STREAMS;
	if ( lane < S::kTasks )
	{
		const float *part = partials + ((uint64_t)row * S::kTasks + lane) * LM_HC_SINKHORN_WARPS;
		float value = 0.0f;
		for ( uint32_t warp = 0u; warp < LM_HC_SINKHORN_WARPS; ++warp )
			value += part[warp];
		mix[lane] = value;
	}
	__syncwarp();
	const float inverse = rsqrtf(mix[S::kMix] / (float)(STREAMS * width) + rms_epsilon);
	const float *m = mix;
	if ( lane < STREAMS )
		pre[(uint64_t)row * STREAMS + lane] = 1.0f / (1.0f + expf(-((m[lane] * inverse) * scale[0] + base[lane]))) + epsilon;
	else if ( lane < 2u * STREAMS )
		post[(uint64_t)row * STREAMS + lane - STREAMS] = 2.0f / (1.0f + expf(-((m[lane] * inverse) * scale[1] + base[lane])));
	const uint32_t cell = lane % (STREAMS * STREAMS);
	float comb = (m[2u * STREAMS + cell] * inverse) * scale[2] + base[2u * STREAMS + cell];
	float top = -INFINITY;
	for ( uint32_t index = 0u; index < STREAMS; ++index )
		top = fmaxf(top, __shfl_sync(0xffffffffu, comb, r * STREAMS + index));
	comb = expf(comb - top);
	comb = comb / LmHcSinkhornLineSum<STREAMS>(comb, r * STREAMS, 1u) + epsilon;
	comb /= LmHcSinkhornLineSum<STREAMS>(comb, c, STREAMS) + epsilon;
	for ( uint32_t iteration = 1u; iteration < iterations; ++iteration )
	{
		comb /= LmHcSinkhornLineSum<STREAMS>(comb, r * STREAMS, 1u) + epsilon;
		comb /= LmHcSinkhornLineSum<STREAMS>(comb, c, STREAMS) + epsilon;
	}
	if ( lane < STREAMS * STREAMS )
		comb_out[(uint64_t)row * STREAMS * STREAMS + lane] = comb;
}

template<uint32_t STREAMS>
static void LmHcSinkhornMixes(const uint16_t *streams, const float *fn, const float *scale, const float *base, float *partials, float *pre, float *post,
	float *comb, uint32_t rows, uint32_t width, float rms_epsilon, uint32_t iterations, float epsilon, cudaStream_t stream)
{
	if ( rows >= LM_HC_SINKHORN_ROWS_FROM )
		LM_LAUNCH((LmHcSinkhornPartialRowsKernel<STREAMS,LM_HC_SINKHORN_ROWS>), (rows + LM_HC_SINKHORN_ROWS - 1u) / LM_HC_SINKHORN_ROWS, LM_HC_SINKHORN_THREADS,
			0, stream, streams, fn, partials, width, rows);
	else
		LM_LAUNCH((LmHcSinkhornPartialKernel<STREAMS>), dim3(LmHcSinkhornShape<STREAMS>::kTasks, rows), LM_HC_SINKHORN_THREADS, 0, stream, streams, fn,
			partials, width);
	LM_LAUNCH((LmHcSinkhornKernel<STREAMS>), rows, 32u, 0, stream, partials, scale, base, pre, post, comb, width, rms_epsilon, iterations, epsilon);
}

template<uint32_t STREAMS>
static __device__ __forceinline__ uint16_t LmHcCollapseElement(const uint16_t *__restrict__ streams, const float *__restrict__ pre, uint32_t row,
	uint32_t width, uint32_t element)
{
	float total = 0.0f;
	for ( uint32_t copy = 0u; copy < STREAMS; ++copy )
		total += pre[(uint64_t)row * STREAMS + copy] * LmBf16ToFloat(streams[((uint64_t)row * STREAMS + copy) * width + element]);
	return LmFloatToBf16(total);
}

template<uint32_t STREAMS>
__global__ void LmHcCollapseKernel(const uint16_t *__restrict__ streams, const float *__restrict__ pre, uint16_t *__restrict__ output,
	uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
		output[(uint64_t)row * width + element] = LmHcCollapseElement<STREAMS>(streams, pre, row, width, element);
}

template<uint32_t STREAMS>
__global__ void LmHcPostBf16Kernel(const uint16_t *out_bf16,const uint16_t *snapshot_bf16,const float *post_f32,const float *comb_f32,uint16_t *streams_bf16,uint32_t rows,uint32_t hc,uint32_t width)
{
	__shared__ float post[STREAMS],comb[STREAMS * STREAMS];
	uint32_t row = blockIdx.x,element,stream,source,index;
	float residual[STREAMS],out,value,contribution;
	if ( row >= rows || hc == 0u || hc > STREAMS )
		return;
	for (index=threadIdx.x; index<hc; index+=blockDim.x)
		post[index] = LmBf16ToFloat(LmFloatToBf16(post_f32[((uint64_t)row * hc) + index]));
	for (index=threadIdx.x; index<(hc * hc); index+=blockDim.x)
		comb[index] = LmBf16ToFloat(LmFloatToBf16(comb_f32[((uint64_t)row * hc * hc) + index]));
	__syncthreads();
	for (element=(blockIdx.y * blockDim.x) + threadIdx.x; element<width; element+=blockDim.x * gridDim.y)
	{
		out = LmBf16ToFloat(out_bf16[((uint64_t)row * width) + element]);
		for (source=0u; source<hc; source++)
			residual[source] = LmBf16ToFloat(snapshot_bf16[((((uint64_t)row * hc) + source) * width) + element]);
		for (stream=0u; stream<hc; stream++)
		{
			value = 0.0f;
			for (source=0u; source<hc; source++)
				value = fmaf(comb[(source * hc) + stream],residual[source],value);
			value = LmBf16ToFloat(LmFloatToBf16(value));
			contribution = LmBf16ToFloat(LmFloatToBf16(post[stream] * out));
			streams_bf16[((((uint64_t)row * hc) + stream) * width) + element] = LmFloatToBf16(contribution + value);
		}
	}
}

template<uint32_t STREAMS>
__global__ void LmHcPostF32Kernel(const uint16_t *out_bf16,const uint16_t *snapshot_bf16,const float *post_f32,const float *comb_f32,uint16_t *streams_bf16,uint32_t rows,uint32_t hc,uint32_t width)
{
	__shared__ float post[STREAMS],comb[STREAMS * STREAMS];
	uint32_t row = blockIdx.x,element,stream,source,index;
	float residual[STREAMS],out,value;
	if ( row >= rows || hc == 0u || hc > STREAMS )
		return;
	for (index=threadIdx.x; index<hc; index+=blockDim.x)
		post[index] = post_f32[((uint64_t)row * hc) + index];
	for (index=threadIdx.x; index<(hc * hc); index+=blockDim.x)
		comb[index] = comb_f32[((uint64_t)row * hc * hc) + index];
	__syncthreads();
	for (element=(blockIdx.y * blockDim.x) + threadIdx.x; element<width; element+=blockDim.x * gridDim.y)
	{
		out = LmBf16ToFloat(out_bf16[((uint64_t)row * width) + element]);
		for (source=0u; source<hc; source++)
			residual[source] = LmBf16ToFloat(snapshot_bf16[((((uint64_t)row * hc) + source) * width) + element]);
		for (stream=0u; stream<hc; stream++)
		{
			value = 0.0f;
			for (source=0u; source<hc; source++)
				value = fmaf(comb[(stream * hc) + source],residual[source],value);
			streams_bf16[((((uint64_t)row * hc) + stream) * width) + element] = LmFloatToBf16(fmaf(post[stream],out,value));
		}
	}
}

template<uint32_t STREAMS>
__global__ void LmHcPostF32SourceMajorKernel(const uint16_t *__restrict__ sublayer, const float *__restrict__ post,
	const float *__restrict__ comb, uint16_t *streams, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		const float x = LmBf16ToFloat(sublayer[(uint64_t)row * width + element]);
		float source[STREAMS];
		for ( uint32_t copy = 0u; copy < STREAMS; ++copy )
			source[copy] = LmBf16ToFloat(streams[((uint64_t)row * STREAMS + copy) * width + element]);
		for ( uint32_t copy = 0u; copy < STREAMS; ++copy )
		{
			float total = 0.0f;
			for ( uint32_t from = 0u; from < STREAMS; ++from )
				total += comb[(uint64_t)row * STREAMS * STREAMS + from * STREAMS + copy] * source[from];
			streams[((uint64_t)row * STREAMS + copy) * width + element] = LmFloatToBf16(post[(uint64_t)row * STREAMS + copy] * x + total);
		}
	}
}
