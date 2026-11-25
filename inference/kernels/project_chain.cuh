#pragma once

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/project.cuh"
#include <stdint.h>
#include "sparkpipe/spark_gpu.h"

#define LM_PROJECT_CHAIN_THREADS 256u
#define LM_PROJECT_CHAIN_OUTPUTS 8u
#define LM_PROJECT_CHAIN_ROWS 8u
#define LM_PROJECT_MMA_THREADS 128u
#define LM_PROJECT_MMA_ROWS 64u
#define LM_PROJECT_MMA_OUTPUTS 64u
#define LM_PROJECT_MMA_K 64u
#define LM_PROJECT_MMA_PITCH (LM_PROJECT_MMA_K + 8u)

template<uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM, uint32_t INPUT_OFFSET, uint32_t OUTPUTS = LM_PROJECT_CHAIN_OUTPUTS, uint32_t MAX_ROWS = LM_PROJECT_CHAIN_ROWS, uint32_t THREADS = LM_PROJECT_CHAIN_THREADS, uint32_t OUT_HEAD_STRIDE = OUT_DIM>
__global__ __launch_bounds__(THREADS)
void LmPerHeadProjectChainKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t heads, uint32_t rows)
{
	constexpr uint32_t vectors = IN_DIM / 8u, stride = IN_DIM + 8u;
	static_assert(IN_DIM % 8u == 0u && INPUT_OFFSET % 8u == 0u && INPUT_HEAD_DIM % 8u == 0u && OUT_DIM % OUTPUTS == 0u && INPUT_OFFSET + IN_DIM <= INPUT_HEAD_DIM, "per-head chain tiles load whole 16-byte words of one head");
	__shared__ __align__(16) uint16_t weight_tile[OUTPUTS][stride];
	__shared__ __align__(16) uint16_t input_tile[MAX_ROWS][stride];
	const uint32_t first_output = blockIdx.x * OUTPUTS, head = blockIdx.y;
	const uint4 *weight = (const uint4 *)(weight_bf16 + ((uint64_t)head * OUT_DIM + first_output) * IN_DIM);
	uint32_t index, row, output, k;
	float total;
	LmDependentRelease();
	for (index = threadIdx.x; index < OUTPUTS * vectors; index += THREADS)
		*(uint4 *)(weight_tile[index / vectors] + (index % vectors) * 8u) = __ldg(weight + index);
	LmDependentWait();
	for (index = threadIdx.x; index < rows * vectors; index += THREADS)
	{
		row = index / vectors;
		k = (index % vectors) * 8u;
		*(uint4 *)(input_tile[row] + k) = *(const uint4 *)(input_bf16 + ((uint64_t)row * heads + head) * INPUT_HEAD_DIM + INPUT_OFFSET + k);
	}
	__syncthreads();
	output = threadIdx.x % OUTPUTS;
	row = threadIdx.x / OUTPUTS;
	if (row >= rows || row >= MAX_ROWS)
		return;
	total = 0.0f;
	#pragma unroll 4
	for (k = 0u; k < IN_DIM; k += 8u)
		total = LmDotBf16x8(*(const uint4 *)(input_tile[row] + k), *(const uint4 *)(weight_tile[output] + k), total);
	output_bf16[((uint64_t)row * heads + head) * OUT_HEAD_STRIDE + first_output + output] = LmFloatToBf16(total);
}

static __device__ __forceinline__ uint32_t LmProjectMmaSwizzle8(uint32_t row, uint32_t chunk)
{
	return(chunk ^ (((row & 1u) << 2u) | ((row >> 1u) & 3u)));
}

static __device__ __forceinline__ void LmProjectCpAsync16(uint32_t destination, const void *source, uint32_t bytes)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(destination), "l"(source), "r"(bytes) : "memory");
#else
	(void)destination;
	(void)source;
	(void)bytes;
#endif
}

static __device__ __forceinline__ void LmProjectCpAsyncCommit(void)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template<uint32_t PENDING>
static __device__ __forceinline__ void LmProjectCpAsyncWait(void)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	asm volatile("cp.async.wait_group %0;\n" :: "n"(PENDING) : "memory");
#endif
}

static __device__ __forceinline__ void LmProjectLdMatrix4(uint32_t out[4], uint32_t address)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
		: "r"(address));
#else
	(void)out;
	(void)address;
#endif
}

template<uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM, uint32_t INPUT_OFFSET, uint32_t OUTPUTS, uint32_t K_TILE, uint32_t THREADS, uint32_t OUT_HEAD_STRIDE = OUT_DIM>
__global__ __launch_bounds__(THREADS)
void LmPerHeadProjectB1Kernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16)
{
	constexpr uint32_t vectors = IN_DIM / 8u, tile_vecs = K_TILE / 8u, stride = K_TILE + 8u;
	static_assert(IN_DIM % K_TILE == 0u && K_TILE % 8u == 0u && OUT_DIM % OUTPUTS == 0u && INPUT_OFFSET % 8u == 0u && INPUT_HEAD_DIM % 8u == 0u && INPUT_OFFSET + IN_DIM <= INPUT_HEAD_DIM, "per-head B1 tiles divide K and output dimensions");
	__shared__ __align__(16) uint16_t input_tile[IN_DIM];
	__shared__ __align__(16) uint16_t weight_tile[2][OUTPUTS][stride];
	const uint32_t first_output = blockIdx.x * OUTPUTS, head = blockIdx.y;
	const uint16_t *weight_base = weight_bf16 + ((uint64_t)head * OUT_DIM + first_output) * IN_DIM;
	LmDependentWait();
	for (uint32_t i = threadIdx.x; i < vectors; i += THREADS)
		((uint4 *)input_tile)[i] = __ldg((const uint4 *)(input_bf16 + (uint64_t)head * INPUT_HEAD_DIM + INPUT_OFFSET) + i);
	float total = 0.0f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	const uint32_t w_base_s = (uint32_t)__cvta_generic_to_shared(weight_tile);
	constexpr uint32_t stage_bytes = OUTPUTS * stride * 2u;
	auto issue_tile = [&](uint32_t stage, uint32_t k0)
	{
		const uint32_t dst_stage = w_base_s + stage * stage_bytes;
		#pragma unroll
		for (uint32_t idx = threadIdx.x; idx < OUTPUTS * tile_vecs; idx += THREADS)
		{
			const uint32_t r = idx / tile_vecs, c = idx % tile_vecs;
			LmProjectCpAsync16(dst_stage + (r * stride + c * 8u) * 2u, (const uint4 *)(weight_base + (uint64_t)r * IN_DIM + k0) + c, 16u);
		}
		LmProjectCpAsyncCommit();
	};
	issue_tile(0u, 0u);
	uint32_t stage = 0u;
	for (uint32_t k0 = 0u; k0 < IN_DIM; k0 += K_TILE)
	{
		if (k0 + K_TILE < IN_DIM)
			issue_tile(stage ^ 1u, k0 + K_TILE);
		if (k0 + K_TILE < IN_DIM)
			LmProjectCpAsyncWait<1u>();
		else
			LmProjectCpAsyncWait<0u>();
		if constexpr (THREADS == 32u)
			__syncwarp();
		else
			__syncthreads();
		if (threadIdx.x < OUTPUTS)
		{
			#pragma unroll
			for (uint32_t k = 0u; k < K_TILE; k += 8u)
				total = LmDotBf16x8(*(const uint4 *)(input_tile + k0 + k), *(const uint4 *)(weight_tile[stage][threadIdx.x] + k), total);
		}
		stage ^= 1u;
		if (k0 + K_TILE < IN_DIM)
		{
			if constexpr (THREADS == 32u)
				__syncwarp();
			else
				__syncthreads();
		}
	}
#else
	for (uint32_t k0 = 0u; k0 < IN_DIM; k0 += K_TILE)
	{
		#pragma unroll
		for (uint32_t idx = threadIdx.x; idx < OUTPUTS * tile_vecs; idx += THREADS)
		{
			const uint32_t r = idx / tile_vecs, c = idx % tile_vecs;
			*(uint4 *)(weight_tile[0][r] + c * 8u) = __ldcs((const uint4 *)(weight_base + (uint64_t)r * IN_DIM + k0) + c);
		}
		__syncthreads();
		if (threadIdx.x < OUTPUTS)
		{
			#pragma unroll
			for (uint32_t k = 0u; k < K_TILE; k += 8u)
				total = LmDotBf16x8(*(const uint4 *)(input_tile + k0 + k), *(const uint4 *)(weight_tile[0][threadIdx.x] + k), total);
		}
		if (k0 + K_TILE < IN_DIM)
			__syncthreads();
	}
#endif
	if (threadIdx.x < OUTPUTS)
		output_bf16[(uint64_t)head * OUT_HEAD_STRIDE + first_output + threadIdx.x] = LmFloatToBf16(total);
	LmDependentRelease();
}

template<uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM, uint32_t INPUT_OFFSET, uint32_t OUT_HEAD_STRIDE = OUT_DIM>
__global__ __launch_bounds__(LM_PROJECT_MMA_THREADS)
void LmPerHeadProjectMmaKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t heads, uint32_t rows)
{
	static_assert(IN_DIM % LM_PROJECT_MMA_K == 0u && OUT_DIM % LM_PROJECT_MMA_OUTPUTS == 0u && INPUT_OFFSET % 8u == 0u && INPUT_HEAD_DIM % 8u == 0u &&
		INPUT_OFFSET + IN_DIM <= INPUT_HEAD_DIM, "per-head tensor-core tiles load whole 16-byte words of one head");
	static_assert(LM_PROJECT_MMA_ROWS == (LM_PROJECT_MMA_THREADS / LM_WARP_LANES) * LM_MMA16_M, "every warp owns 16 rows of the tile");
	constexpr uint32_t words = LM_PROJECT_MMA_K / 8u, tiles = LM_PROJECT_MMA_OUTPUTS / LM_MMA16_N;
	constexpr uint32_t row_bytes = LM_PROJECT_MMA_K * 2u;
	constexpr uint32_t a_stage_bytes = LM_PROJECT_MMA_ROWS * row_bytes;
	constexpr uint32_t b_stage_bytes = LM_PROJECT_MMA_OUTPUTS * row_bytes;
	constexpr uint32_t stage_bytes = a_stage_bytes + b_stage_bytes;
	const uint32_t first_output = blockIdx.x * LM_PROJECT_MMA_OUTPUTS, head = blockIdx.y, first_row = blockIdx.z * LM_PROJECT_MMA_ROWS;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, warp_row = (threadIdx.x / LM_WARP_LANES) * LM_MMA16_M;
	const uint16_t *weight = weight_bf16 + ((uint64_t)head * OUT_DIM + first_output) * IN_DIM;
	float accumulator[tiles][4];
	uint32_t tile, reg, row, column;
	#pragma unroll
	for (tile = 0u; tile < tiles; ++tile)
		#pragma unroll
		for (reg = 0u; reg < 4u; ++reg)
			accumulator[tile][reg] = 0.0f;
	LmDependentWait();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	__shared__ __align__(128) uint8_t shared_stages[2u * stage_bytes];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(shared_stages);
	const uint32_t k_tiles = IN_DIM / LM_PROJECT_MMA_K;
	const uint32_t ldm_a_row = warp_row + (lane & 15u), ldm_a_col = (lane >> 4u) & 1u;
	const uint32_t ldm_b_row = (lane & 7u) + (((lane >> 4u) & 1u) * 8u), ldm_b_col = (lane >> 3u) & 1u;
	auto issue_k = [&](uint32_t stage_off, uint32_t k0)
	{
		const uint32_t stage_a = shared_base + stage_off;
		const uint32_t stage_b = stage_a + a_stage_bytes;
		#pragma unroll
		for (uint32_t index = threadIdx.x; index < LM_PROJECT_MMA_ROWS * words; index += LM_PROJECT_MMA_THREADS)
		{
			const uint32_t r = index / words, c = index % words;
			const uint32_t valid = first_row + r < rows ? 16u : 0u;
			const uint16_t *src = valid != 0u
				? input_bf16 + ((uint64_t)(first_row + r) * heads + head) * INPUT_HEAD_DIM + INPUT_OFFSET + k0 + c * 8u
				: input_bf16;
			LmProjectCpAsync16(stage_a + r * row_bytes + LmProjectMmaSwizzle8(r, c) * 16u, src, valid);
		}
		#pragma unroll
		for (uint32_t index = threadIdx.x; index < LM_PROJECT_MMA_OUTPUTS * words; index += LM_PROJECT_MMA_THREADS)
		{
			const uint32_t r = index / words, c = index % words;
			LmProjectCpAsync16(stage_b + r * row_bytes + LmProjectMmaSwizzle8(r, c) * 16u, weight + (uint64_t)r * IN_DIM + k0 + c * 8u, 16u);
		}
		LmProjectCpAsyncCommit();
	};
	issue_k(0u, 0u);
	uint32_t cur_off = 0u;
	for (uint32_t kt = 0u; kt < k_tiles; ++kt)
	{
		if (kt + 1u < k_tiles)
			issue_k(cur_off ^ stage_bytes, (kt + 1u) * LM_PROJECT_MMA_K);
		if (kt + 1u < k_tiles)
			LmProjectCpAsyncWait<1u>();
		else
			LmProjectCpAsyncWait<0u>();
		__syncthreads();
		const uint32_t stage_a = shared_base + cur_off;
		const uint32_t stage_b = stage_a + a_stage_bytes;
		uint32_t a[4], b_tmp[4];
		#pragma unroll
		for (uint32_t ks = 0u; ks < LM_PROJECT_MMA_K / LM_MMA16_K; ++ks)
		{
			LmProjectLdMatrix4(a, stage_a + ldm_a_row * row_bytes + LmProjectMmaSwizzle8(ldm_a_row, ks * 2u + ldm_a_col) * 16u);
			#pragma unroll
			for (tile = 0u; tile < tiles; tile += 2u)
			{
				const uint32_t b_row = tile * LM_MMA16_N + ldm_b_row;
				LmProjectLdMatrix4(b_tmp, stage_b + b_row * row_bytes + LmProjectMmaSwizzle8(b_row, ks * 2u + ldm_b_col) * 16u);
				uint32_t b0[2] = {b_tmp[0], b_tmp[1]};
				uint32_t b1[2] = {b_tmp[2], b_tmp[3]};
				LmMmaBf16(accumulator[tile], a, b0);
				LmMmaBf16(accumulator[tile + 1u], a, b1);
			}
		}
		cur_off ^= stage_bytes;
		if (kt + 1u < k_tiles)
			__syncthreads();
	}
#else
	__shared__ __align__(16) uint16_t input_tile[LM_PROJECT_MMA_ROWS * LM_PROJECT_MMA_PITCH];
	__shared__ __align__(16) uint16_t weight_tile[LM_PROJECT_MMA_OUTPUTS * LM_PROJECT_MMA_PITCH];
	uint32_t index, k, step, a[4], b[2];
	for (k = 0u; k < IN_DIM; k += LM_PROJECT_MMA_K)
	{
		for (index = threadIdx.x; index < LM_PROJECT_MMA_ROWS * words; index += LM_PROJECT_MMA_THREADS)
		{
			row = index / words;
			column = (index % words) * 8u;
			*(uint4 *)(input_tile + row * LM_PROJECT_MMA_PITCH + column) = first_row + row < rows ?
				__ldg((const uint4 *)(input_bf16 + ((uint64_t)(first_row + row) * heads + head) * INPUT_HEAD_DIM + INPUT_OFFSET + k + column)) : make_uint4(0u, 0u, 0u, 0u);
		}
		for (index = threadIdx.x; index < LM_PROJECT_MMA_OUTPUTS * words; index += LM_PROJECT_MMA_THREADS)
		{
			row = index / words;
			column = (index % words) * 8u;
			*(uint4 *)(weight_tile + row * LM_PROJECT_MMA_PITCH + column) = __ldg((const uint4 *)(weight + (uint64_t)row * IN_DIM + k + column));
		}
		__syncthreads();
		for (step = 0u; step < LM_PROJECT_MMA_K; step += LM_MMA16_K)
		{
			for (reg = 0u; reg < 4u; ++reg)
				a[reg] = *(const uint32_t *)(input_tile + (warp_row + LmMma16OperandARow(lane, reg)) * LM_PROJECT_MMA_PITCH + step + LmMma16OperandAK(lane, reg));
			for (tile = 0u; tile < tiles; ++tile)
			{
				for (reg = 0u; reg < 2u; ++reg)
					b[reg] = *(const uint32_t *)(weight_tile + (tile * LM_MMA16_N + LmMma16OperandBRow(lane)) * LM_PROJECT_MMA_PITCH + step + LmMma16OperandBK(lane, reg));
				LmMmaBf16(accumulator[tile], a, b);
			}
		}
		__syncthreads();
	}
#endif
	#pragma unroll
	for (tile = 0u; tile < tiles; ++tile)
		#pragma unroll
		for (reg = 0u; reg < 4u; reg += 2u)
		{
			row = first_row + warp_row + LmMmaAccumulatorRow(lane, reg);
			column = first_output + tile * LM_MMA16_N + LmMmaAccumulatorColumn(lane, reg);
			if ( row < rows )
				*(uint32_t *)(output_bf16 + ((uint64_t)row * heads + head) * OUT_HEAD_STRIDE + column) = LmPackBf16Pair(accumulator[tile][reg], accumulator[tile][reg + 1u]);
		}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM = IN_DIM, uint32_t INPUT_OFFSET = 0u, uint32_t OUT_HEAD_STRIDE = OUT_DIM>
static inline SparkGpuError LmPerHeadProjectChainLaunch(const uint16_t *input_bf16, const uint16_t *weight_bf16, uint16_t *output_bf16, uint32_t heads, uint32_t rows, SparkGpuStream stream,
	uint32_t row_exact = 0u)
{
	if (rows == 0u || (rows > LM_PROJECT_CHAIN_ROWS && row_exact != 0u))
		return LmPerHeadProjectRowsLaunch<THREADS, IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET, OUT_HEAD_STRIDE>(input_bf16, weight_bf16, output_bf16, heads, rows, stream);
	if (rows > LM_PROJECT_CHAIN_ROWS)
	{
		LM_LAUNCH_DEPENDENT((LmPerHeadProjectMmaKernel<IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET, OUT_HEAD_STRIDE>), dim3(OUT_DIM / LM_PROJECT_MMA_OUTPUTS, heads, (rows + LM_PROJECT_MMA_ROWS - 1u) / LM_PROJECT_MMA_ROWS),
			LM_PROJECT_MMA_THREADS, 0, stream, input_bf16, weight_bf16, output_bf16, heads, rows);
		return SparkGpuPeekAtLastError();
	}
	if (rows == 1u)
	{
		if constexpr (IN_DIM % 64u == 0u && OUT_DIM % 16u == 0u)
			if ((OUT_DIM / 16u) * heads >= 48u)
			{
				LM_LAUNCH_DEPENDENT((LmPerHeadProjectB1Kernel<IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET, 16u, 64u, 32u, OUT_HEAD_STRIDE>), dim3(OUT_DIM / 16u, heads, 1u), 32u, 0, stream, input_bf16, weight_bf16, output_bf16);
				return SparkGpuPeekAtLastError();
			}
		if constexpr (OUT_DIM % 16u == 0u)
			if ((OUT_DIM / 16u) * heads >= 48u)
			{
				LM_LAUNCH_DEPENDENT((LmPerHeadProjectChainKernel<IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET, 16u, 1u, 64u, OUT_HEAD_STRIDE>), dim3(OUT_DIM / 16u, heads, 1u), 64u, 0, stream, input_bf16, weight_bf16, output_bf16, heads, rows);
				return SparkGpuPeekAtLastError();
			}
		LM_LAUNCH_DEPENDENT((LmPerHeadProjectChainKernel<IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET, LM_PROJECT_CHAIN_OUTPUTS, 1u, 64u, OUT_HEAD_STRIDE>), dim3(OUT_DIM / LM_PROJECT_CHAIN_OUTPUTS, heads, 1u), 64u, 0, stream, input_bf16, weight_bf16, output_bf16, heads, rows);
		return SparkGpuPeekAtLastError();
	}
	if constexpr (OUT_DIM % 16u == 0u)
	{
		if (rows == 2u)
			LM_LAUNCH_DEPENDENT((LmPerHeadProjectChainKernel<IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET, 16u, 2u, IN_DIM >= 256u ? 64u : 32u, OUT_HEAD_STRIDE>), dim3(OUT_DIM / 16u, heads, 1u), IN_DIM >= 256u ? 64u : 32u, 0, stream, input_bf16, weight_bf16, output_bf16, heads, rows);
		else if (rows <= 4u)
			LM_LAUNCH_DEPENDENT((LmPerHeadProjectChainKernel<IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET, 16u, 4u, IN_DIM >= 256u ? 128u : 64u, OUT_HEAD_STRIDE>), dim3(OUT_DIM / 16u, heads, 1u), IN_DIM >= 256u ? 128u : 64u, 0, stream, input_bf16, weight_bf16, output_bf16, heads, rows);
		else
			LM_LAUNCH_DEPENDENT((LmPerHeadProjectChainKernel<IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET, 16u, LM_PROJECT_CHAIN_ROWS, 128u, OUT_HEAD_STRIDE>), dim3(OUT_DIM / 16u, heads, 1u), 128u, 0, stream, input_bf16, weight_bf16, output_bf16, heads, rows);
		return SparkGpuPeekAtLastError();
	}
	LM_LAUNCH_DEPENDENT((LmPerHeadProjectChainKernel<IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET, LM_PROJECT_CHAIN_OUTPUTS, LM_PROJECT_CHAIN_ROWS, 64u, OUT_HEAD_STRIDE>), dim3(OUT_DIM / LM_PROJECT_CHAIN_OUTPUTS, heads, 1u), 64u, 0, stream, input_bf16, weight_bf16, output_bf16, heads, rows);
	return SparkGpuPeekAtLastError();
}
