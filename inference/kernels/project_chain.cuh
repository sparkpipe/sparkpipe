#pragma once

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/project.cuh"
#include <stdint.h>

#define LM_PROJECT_CHAIN_THREADS 256u
#define LM_PROJECT_CHAIN_OUTPUTS 8u
#define LM_PROJECT_CHAIN_ROWS 8u
#define LM_PROJECT_MMA_THREADS 128u
#define LM_PROJECT_MMA_ROWS 64u
#define LM_PROJECT_MMA_OUTPUTS 64u
#define LM_PROJECT_MMA_K 64u
#define LM_PROJECT_MMA_PITCH (LM_PROJECT_MMA_K + 8u)

template<uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM, uint32_t INPUT_OFFSET>
__global__ __launch_bounds__(LM_PROJECT_CHAIN_THREADS, 1)
void LmPerHeadProjectChainKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t heads, uint32_t rows)
{
	static_assert(IN_DIM % 8u == 0u && INPUT_OFFSET % 8u == 0u && INPUT_HEAD_DIM % 8u == 0u && OUT_DIM % LM_PROJECT_CHAIN_OUTPUTS == 0u && INPUT_OFFSET + IN_DIM <= INPUT_HEAD_DIM, "per-head chain tiles load whole 16-byte words of one head");
	__shared__ float weight_tile[LM_PROJECT_CHAIN_OUTPUTS][IN_DIM + 1u];
	__shared__ float input_tile[LM_PROJECT_CHAIN_ROWS][IN_DIM + 1u];
	const uint32_t first_output = blockIdx.x * LM_PROJECT_CHAIN_OUTPUTS, head = blockIdx.y;
	const uint4 *weight = (const uint4 *)(weight_bf16 + ((uint64_t)head * OUT_DIM + first_output) * IN_DIM);
	uint32_t index, word, row, output, k;
	uint32_t bits[4];
	uint4 packed;
	float total;
	LmDependentRelease();
	for (index = threadIdx.x; index < LM_PROJECT_CHAIN_OUTPUTS * IN_DIM / 8u; index += LM_PROJECT_CHAIN_THREADS)
	{
		packed = __ldg(weight + index);
		bits[0] = packed.x; bits[1] = packed.y; bits[2] = packed.z; bits[3] = packed.w;
		output = (index * 8u) / IN_DIM;
		k = (index * 8u) % IN_DIM;
		for (word = 0u; word < 4u; ++word)
		{
			weight_tile[output][k + word * 2u] = __uint_as_float(bits[word] << 16u);
			weight_tile[output][k + word * 2u + 1u] = __uint_as_float(bits[word] & 0xffff0000u);
		}
	}
	LmDependentWait();
	for (index = threadIdx.x; index < rows * IN_DIM / 8u; index += LM_PROJECT_CHAIN_THREADS)
	{
		row = (index * 8u) / IN_DIM;
		k = (index * 8u) % IN_DIM;
		packed = *(const uint4 *)(input_bf16 + ((uint64_t)row * heads + head) * INPUT_HEAD_DIM + INPUT_OFFSET + k);
		bits[0] = packed.x; bits[1] = packed.y; bits[2] = packed.z; bits[3] = packed.w;
		for (word = 0u; word < 4u; ++word)
		{
			input_tile[row][k + word * 2u] = __uint_as_float(bits[word] << 16u);
			input_tile[row][k + word * 2u + 1u] = __uint_as_float(bits[word] & 0xffff0000u);
		}
	}
	__syncthreads();
	output = threadIdx.x % LM_PROJECT_CHAIN_OUTPUTS;
	row = threadIdx.x / LM_PROJECT_CHAIN_OUTPUTS;
	if (row >= rows)
		return;
	total = 0.0f;
	#pragma unroll 8
	for (k = 0u; k < IN_DIM; ++k)
		total = fmaf(input_tile[row][k], weight_tile[output][k], total);
	output_bf16[((uint64_t)row * heads + head) * OUT_DIM + first_output + output] = LmFloatToBf16(total);
}

template<uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM, uint32_t INPUT_OFFSET>
__global__ __launch_bounds__(LM_PROJECT_MMA_THREADS)
void LmPerHeadProjectMmaKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t heads, uint32_t rows)
{
	static_assert(IN_DIM % LM_PROJECT_MMA_K == 0u && OUT_DIM % LM_PROJECT_MMA_OUTPUTS == 0u && INPUT_OFFSET % 8u == 0u && INPUT_HEAD_DIM % 8u == 0u &&
		INPUT_OFFSET + IN_DIM <= INPUT_HEAD_DIM, "per-head tensor-core tiles load whole 16-byte words of one head");
	static_assert(LM_PROJECT_MMA_ROWS == (LM_PROJECT_MMA_THREADS / LM_WARP_LANES) * LM_MMA16_M, "every warp owns 16 rows of the tile");
	constexpr uint32_t words = LM_PROJECT_MMA_K / 8u, tiles = LM_PROJECT_MMA_OUTPUTS / LM_MMA16_N;
	__shared__ __align__(16) uint16_t input_tile[LM_PROJECT_MMA_ROWS * LM_PROJECT_MMA_PITCH];
	__shared__ __align__(16) uint16_t weight_tile[LM_PROJECT_MMA_OUTPUTS * LM_PROJECT_MMA_PITCH];
	const uint32_t first_output = blockIdx.x * LM_PROJECT_MMA_OUTPUTS, head = blockIdx.y, first_row = blockIdx.z * LM_PROJECT_MMA_ROWS;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, warp_row = (threadIdx.x / LM_WARP_LANES) * LM_MMA16_M;
	const uint16_t *weight = weight_bf16 + ((uint64_t)head * OUT_DIM + first_output) * IN_DIM;
	float accumulator[tiles][4];
	uint32_t index, k, step, tile, reg, row, column, a[4], b[2];
	for (tile = 0u; tile < tiles; ++tile)
		for (reg = 0u; reg < 4u; ++reg)
			accumulator[tile][reg] = 0.0f;
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
	for (tile = 0u; tile < tiles; ++tile)
		for (reg = 0u; reg < 4u; reg += 2u)
		{
			row = first_row + warp_row + LmMmaAccumulatorRow(lane, reg);
			column = first_output + tile * LM_MMA16_N + LmMmaAccumulatorColumn(lane, reg);
			if ( row < rows )
				*(uint32_t *)(output_bf16 + ((uint64_t)row * heads + head) * OUT_DIM + column) = LmPackBf16Pair(accumulator[tile][reg], accumulator[tile][reg + 1u]);
		}
}

template<uint32_t THREADS, uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM = IN_DIM, uint32_t INPUT_OFFSET = 0u>
static inline cudaError_t LmPerHeadProjectChainLaunch(const uint16_t *input_bf16, const uint16_t *weight_bf16, uint16_t *output_bf16, uint32_t heads, uint32_t rows, cudaStream_t stream)
{
	if (rows == 0u)
		return LmPerHeadProjectRowsLaunch<THREADS, IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET>(input_bf16, weight_bf16, output_bf16, heads, rows, stream);
	if (rows > LM_PROJECT_CHAIN_ROWS)
	{
		LM_LAUNCH((LmPerHeadProjectMmaKernel<IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET>), dim3(OUT_DIM / LM_PROJECT_MMA_OUTPUTS, heads, (rows + LM_PROJECT_MMA_ROWS - 1u) / LM_PROJECT_MMA_ROWS),
			LM_PROJECT_MMA_THREADS, 0, stream, input_bf16, weight_bf16, output_bf16, heads, rows);
		return cudaPeekAtLastError();
	}
	LM_LAUNCH_DEPENDENT((LmPerHeadProjectChainKernel<IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET>), dim3(OUT_DIM / LM_PROJECT_CHAIN_OUTPUTS, heads, 1u), LM_PROJECT_CHAIN_THREADS, 0, stream, input_bf16, weight_bf16, output_bf16, heads, rows);
	return cudaPeekAtLastError();
}
