#pragma once


#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/rows_tile.cuh"
#include <stdint.h>

struct LmQkvLayout
{
	uint32_t query_dimension;
	uint32_t key_dimension;
	uint32_t value_dimension;
	uint32_t rope_dimension;
	uint32_t head_dimension;
};

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSplitQkvKernel(const uint16_t *__restrict__ fused_bf16, LmQkvLayout layout, uint16_t *__restrict__ query_bf16, uint16_t *__restrict__ key_bf16, uint16_t *__restrict__ value_bf16, uint32_t rows, float value_scale)
{
	uint32_t row = blockIdx.x,index;
	uint32_t total = layout.query_dimension + layout.key_dimension + layout.value_dimension;
	uint64_t base = (uint64_t)row * total;
	if ( row >= rows )
		return;
	for (index = threadIdx.x; index < layout.query_dimension; index += THREADS)
		query_bf16[((uint64_t)row * layout.query_dimension) + index] = fused_bf16[base + index];
	for (index = threadIdx.x; index < layout.key_dimension; index += THREADS)
		key_bf16[((uint64_t)row * layout.key_dimension) + index] =
			fused_bf16[base + layout.query_dimension + index];
	for (index = threadIdx.x; index < layout.value_dimension; index += THREADS)
		value_bf16[((uint64_t)row * layout.value_dimension) + index] =
			LmFloatToBf16(LmBf16ToFloat(
				fused_bf16[base + layout.query_dimension + layout.key_dimension + index])
				* value_scale);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSplitQueryGateKernel(const uint16_t *__restrict__ fused_bf16, uint16_t *__restrict__ query_bf16, uint16_t *__restrict__ gate_bf16, uint32_t heads, uint32_t head_dimension, uint32_t rows)
{
	uint32_t row = blockIdx.x,index,head,element;
	uint64_t fused_base = (uint64_t)row * heads * 2u * head_dimension;
	uint64_t half_base = (uint64_t)row * heads * head_dimension;
	if ( row >= rows )
		return;
	for (index = threadIdx.x; index < heads * head_dimension; index += THREADS)
	{
		head = index / head_dimension;
		element = index % head_dimension;
		query_bf16[half_base + index] =
			fused_bf16[fused_base + (head * 2u * head_dimension) + element];
		gate_bf16[half_base + index] =
			fused_bf16[fused_base + (head * 2u * head_dimension) + head_dimension + element];
	}
}

template<uint32_t THREADS, LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmRopePerHeadKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t heads, uint32_t head_dimension, uint32_t rope_dimension, float theta, const float *inv_freq_table = 0, float attention_scale = 1.0f, uint32_t rope_offset = 0xffffffffu)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint32_t half = rope_dimension / 2u;
	uint64_t base = (((uint64_t)row * heads) + head) * head_dimension
		+ (rope_offset == 0xffffffffu ? (head_dimension - rope_dimension) : rope_offset);
	float position = (float)positions[row];
	for (index = threadIdx.x; index < half; index += THREADS)
		LmRopeRotate<PAIRING>(rows_bf16,base,index,half,
			position * (inv_freq_table != 0 ? inv_freq_table[index]
				: __powf(theta,-2.0f * (float)index / (float)rope_dimension)),
			attention_scale);
}

template<
	uint32_t THREADS,
	uint32_t IN_DIM,
	uint32_t OUT_DIM,
	uint32_t INPUT_HEAD_DIM = IN_DIM,
	uint32_t INPUT_OFFSET = 0u>
__global__ __launch_bounds__(THREADS, 1)
void LmPerHeadProjectKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t heads, uint32_t rows)
{
	__shared__ float shared_input[IN_DIM];
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint64_t input_base,weight_base,output_base;
	static_assert(INPUT_OFFSET + IN_DIM <= INPUT_HEAD_DIM,
		"per-head input slice exceeds its source head");
	if ( row >= rows || head >= heads )
		return;
	input_base = ((((uint64_t)row * heads) + head) * INPUT_HEAD_DIM)
		+ INPUT_OFFSET;
	weight_base = (uint64_t)head * OUT_DIM * IN_DIM;
	output_base = (((uint64_t)row * heads) + head) * OUT_DIM;
	for (index = threadIdx.x; index < IN_DIM; index += THREADS)
		shared_input[index] = LmBf16ToFloat(input_bf16[input_base + index]);
	__syncthreads();
	for (index = threadIdx.x; index < OUT_DIM; index += THREADS)
	{
		float total = 0.0f;
		uint32_t element;
		for (element = 0u; element < IN_DIM; ++element)
			total += shared_input[element]
				* LmBf16ToFloat(weight_bf16[weight_base + (index * IN_DIM) + element]);
		output_bf16[output_base + index] = LmFloatToBf16(total);
	}
}

#define LM_PER_HEAD_SPLIT_OUTPUTS 16u

template<uint32_t THREADS, uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM = IN_DIM, uint32_t INPUT_OFFSET = 0u>
__global__ __launch_bounds__(THREADS)
void LmPerHeadProjectSplitKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t heads, uint32_t rows)
{
	constexpr uint32_t words = IN_DIM / 2u, stride = words + 1u, vectors = IN_DIM / 8u;
	static_assert((OUT_DIM % LM_PER_HEAD_SPLIT_OUTPUTS) == 0u && (IN_DIM % 8u) == 0u,
		"the split covers whole output slices and 16-byte weight vectors");
	static_assert(INPUT_OFFSET + IN_DIM <= INPUT_HEAD_DIM, "per-head input slice exceeds its source head");
	__shared__ float shared_input[IN_DIM];
	__shared__ uint32_t tile[LM_PER_HEAD_SPLIT_OUTPUTS * stride];
	const uint32_t row = blockIdx.x, head = blockIdx.y, first = blockIdx.z * LM_PER_HEAD_SPLIT_OUTPUTS;
	uint32_t index, element, word;
	uint64_t input_base, output_base;
	const uint4 *weight;
	float total = 0.0f;
	if ( row >= rows || head >= heads )
		return;
	input_base = ((((uint64_t)row * heads) + head) * INPUT_HEAD_DIM) + INPUT_OFFSET;
	output_base = (((uint64_t)row * heads) + head) * OUT_DIM + first;
	weight = (const uint4 *)(weight_bf16 + (uint64_t)head * OUT_DIM * IN_DIM + (uint64_t)first * IN_DIM);
	for (index = threadIdx.x; index < IN_DIM; index += THREADS)
		shared_input[index] = LmBf16ToFloat(input_bf16[input_base + index]);
	for (index = threadIdx.x; index < LM_PER_HEAD_SPLIT_OUTPUTS * vectors; index += THREADS)
	{
		uint4 packed = weight[index];
		uint32_t *destination = &tile[(index / vectors) * stride + (index % vectors) * 4u];
		destination[0] = packed.x;
		destination[1] = packed.y;
		destination[2] = packed.z;
		destination[3] = packed.w;
	}
	__syncthreads();
	if ( threadIdx.x >= LM_PER_HEAD_SPLIT_OUTPUTS )
		return;
	for (element = 0u; element < words; ++element)
	{
		word = tile[threadIdx.x * stride + element];
		total += shared_input[2u * element] * LmBf16ToFloat((uint16_t)(word & 0xffffu));
		total += shared_input[2u * element + 1u] * LmBf16ToFloat((uint16_t)(word >> 16u));
	}
	output_bf16[output_base + threadIdx.x] = LmFloatToBf16(total);
}

template<uint32_t THREADS, uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM, uint32_t INPUT_OFFSET, uint32_t ROWS>
__global__ __launch_bounds__(THREADS, 1)
void LmPerHeadProjectRowsKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t heads, uint32_t rows)
{
	constexpr uint32_t per_thread = ROWS / (THREADS / LM_ROWS_TILE_N);
	static_assert(IN_DIM % LM_ROWS_TILE_K == 0u && OUT_DIM % LM_ROWS_TILE_N == 0u && INPUT_OFFSET + IN_DIM <= INPUT_HEAD_DIM, "per-head projection tiles must divide the head");
	__shared__ float weight_tile[LM_ROWS_TILE_N][LM_ROWS_TILE_K + 1u];
	__shared__ float input_tile[ROWS][LM_ROWS_TILE_K];
	const uint32_t first_index = blockIdx.x * LM_ROWS_TILE_N, head = blockIdx.y, first_row = blockIdx.z * ROWS, lane = threadIdx.x % LM_ROWS_TILE_N, group = threadIdx.x / LM_ROWS_TILE_N;
	const LmRowsTileOperand operand = {input_bf16 + ((uint64_t)head * INPUT_HEAD_DIM) + INPUT_OFFSET, weight_bf16 + ((uint64_t)head * OUT_DIM * IN_DIM), 0, (uint64_t)heads * INPUT_HEAD_DIM, IN_DIM, rows, OUT_DIM, IN_DIM};
	uint32_t j, row;
	float total[per_thread];
	LmRowsTileDot<THREADS, ROWS>(&operand, weight_tile, input_tile, first_index, first_row, total);
	for (j = 0u; j < per_thread; ++j)
	{
		row = first_row + group * per_thread + j;
		if ( row < rows )
			output_bf16[((((uint64_t)row * heads) + head) * OUT_DIM) + first_index + lane] = LmFloatToBf16(total[j]);
	}
}

template<uint32_t THREADS, uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM = IN_DIM, uint32_t INPUT_OFFSET = 0u>
static inline cudaError_t LmPerHeadProjectRowsLaunch(const uint16_t *input_bf16, const uint16_t *weight_bf16, uint16_t *output_bf16, uint32_t heads, uint32_t rows, cudaStream_t stream)
{
	constexpr uint32_t few = THREADS / LM_ROWS_TILE_N, many = 4u * few;
	if ( rows <= few )
		LM_LAUNCH((LmPerHeadProjectRowsKernel<THREADS, IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET, few>), dim3(OUT_DIM / LM_ROWS_TILE_N, heads, 1u), THREADS, 0, stream, input_bf16, weight_bf16, output_bf16, heads, rows);
	else
		LM_LAUNCH((LmPerHeadProjectRowsKernel<THREADS, IN_DIM, OUT_DIM, INPUT_HEAD_DIM, INPUT_OFFSET, many>), dim3(OUT_DIM / LM_ROWS_TILE_N, heads, (rows + many - 1u) / many), THREADS, 0, stream, input_bf16, weight_bf16, output_bf16, heads, rows);
	return(cudaPeekAtLastError());
}

template<
	uint32_t THREADS,
	LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT,
	bool TABLE = false>
__global__ __launch_bounds__(THREADS, 1)
void LmExtractRopePerHeadKernel(
	const uint16_t *__restrict__ input_bf16,
	uint16_t *__restrict__ output_bf16,
	const uint32_t *__restrict__ positions,
	uint32_t heads,
	uint32_t input_head_dimension,
	uint32_t input_offset,
	uint32_t rope_dimension,
	float theta,
	const float *__restrict__ inv_freq_table = 0)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint32_t half = rope_dimension / 2u;
	uint64_t input_base,output_base;
	float position;
	if ( head >= heads || input_offset > input_head_dimension ||
		rope_dimension > input_head_dimension - input_offset )
		return;
	input_base = ((((uint64_t)row * heads) + head) * input_head_dimension)
		+ input_offset;
	output_base = (((uint64_t)row * heads) + head) * rope_dimension;
	position = (float)positions[row];
	for (index = threadIdx.x; index < half; index += THREADS)
	{
		uint32_t low_offset,high_offset;
		float low,high,angle;
		low_offset = PAIRING == LM_ROPE_INTERLEAVED ? index * 2u : index;
		high_offset = PAIRING == LM_ROPE_INTERLEAVED
			? (index * 2u) + 1u : half + index;
		low = LmBf16ToFloat(input_bf16[input_base + low_offset]);
		high = LmBf16ToFloat(input_bf16[input_base + high_offset]);
		if constexpr ( TABLE )
			angle = position * inv_freq_table[index];
		else
			angle = position * __powf(
				theta,-2.0f * (float)index / (float)rope_dimension);
		LmRopePair(&low,&high,angle);
		output_bf16[output_base + low_offset] = LmFloatToBf16(low);
		output_bf16[output_base + high_offset] = LmFloatToBf16(high);
	}
}
