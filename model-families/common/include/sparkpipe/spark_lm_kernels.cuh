#pragma once

#include <stdio.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>

#include "inference/kernels/activation.cuh"
#include "sparkpipe/spark_head_screen.h"

/*
 * Shared device kernels for the sparkpipe model-driver family.
 *
 * Extracted verbatim from the audited K3 resident decode stage (2026-07-18
 * production audit) and renamed SparkLm*: every driver module includes this
 * header inside its own translation unit, so each module gets its own
 * internal-linkage instantiation. One source, zero ABI coupling between
 * modules, zero runtime cost. The MXFP4 group size is a template parameter,
 * never a defaulted macro, so a module that disagrees with the pack format
 * fails to build instead of silently decoding garbage.
 *
 * Consumers today: qwen36, dsv4, mimo25 (from first line). The K3 module
 * retrofits onto this header at its PP v2 pass; until then the K3 copies of
 * these functions remain the audited originals this file was taken from.
 */

#define SPARK_LM_WARP_LANES 32u
#define SPARK_LM_CTA_THREADS 256u
#define SPARK_LM_CTA_WARPS (SPARK_LM_CTA_THREADS / SPARK_LM_WARP_LANES)
#define SPARK_LM_SCALAR_CTA_THREADS 1024u
#define SPARK_LM_SCALAR_CTA_WARPS \
	(SPARK_LM_SCALAR_CTA_THREADS / SPARK_LM_WARP_LANES)
#define SPARK_LM_SM121_B1_DENSE_W13_CTA_THREADS 1024u
#define SPARK_LM_SM121_B1_DENSE_W13_CTA_WARPS \
	(SPARK_LM_SM121_B1_DENSE_W13_CTA_THREADS / SPARK_LM_WARP_LANES)

#define SPARK_LM_PAIR_POLICY_FLAT_8 0u
#define SPARK_LM_PAIR_POLICY_FLAT_16 1u
#define SPARK_LM_FP8_PAIR_WIDE_MINIMUM_WORK 1572864ull
#define SPARK_LM_BF16_PAIR_WIDE_MINIMUM_WORK 524288ull

#define SPARK_LM_WEIGHT_FORMAT_BF16 0u
#define SPARK_LM_WEIGHT_FORMAT_F32 1u
#define SPARK_LM_WEIGHT_FORMAT_U32 2u
#define SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1 3u
#define SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 4u
#define SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 5u

#define SPARK_LM_EXPERT_TILE_POLICY_ALL_WARPS 0u
#define SPARK_LM_EXPERT_TILE_POLICY_SOFTWARE_PIPELINED 1u
#define SPARK_LM_EXPERT_TILE_POLICY_AUTOMATIC 2u
#define SPARK_LM_EXPERT_TILE_POLICY SPARK_LM_EXPERT_TILE_POLICY_AUTOMATIC
#define SPARK_LM_WEIGHT_READ_AHEAD_SECTOR_BYTES 32u
#if SPARK_LM_EXPERT_TILE_POLICY > SPARK_LM_EXPERT_TILE_POLICY_AUTOMATIC
#error "SPARK_LM_EXPERT_TILE_POLICY is invalid"
#endif

static __device__ __forceinline__ float SparkLmBf16ToFloat(const void *source, uint64_t index)
{
	return(__bfloat162float(((const __nv_bfloat16 *)source)[index]));
}

static __device__ __forceinline__ void SparkLmFloatToBf16(void *destination, uint64_t index, float value)
{
	((__nv_bfloat16 *)destination)[index] = __float2bfloat16(value);
}

/*
 * Read immutable device weights on an auxiliary stream so an exposed
 * collective window can populate cache without changing model arithmetic.
 * Every thread publishes its checksum into private scratch, preventing the
 * compiler from deleting the vector loads while keeping inference outputs
 * completely disjoint from this hint path. One uint4 load warms each measured
 * 32-byte GB10 cache sector; the small auxiliary range is read completely.
 */
static __global__ void SparkLmWeightReadAheadKernel(
	const uint4 *payload,uint64_t sector_count,const uint4 *auxiliary_payload,
	uint64_t auxiliary_vector_count,uint32_t *sink_u32)
{
	uint64_t thread_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	uint64_t index;
	uint32_t checksum = 0u;
	uint4 value;
	for (index=thread_index; index<sector_count; index+=stride)
	{
		value = payload[index *
			(SPARK_LM_WEIGHT_READ_AHEAD_SECTOR_BYTES / sizeof(uint4))];
		checksum ^= value.x ^ value.y ^ value.z ^ value.w;
	}
	for (index=thread_index; index<auxiliary_vector_count; index+=stride)
	{
		value = auxiliary_payload[index];
		checksum ^= value.x ^ value.y ^ value.z ^ value.w;
	}
	sink_u32[thread_index] = checksum;
}

static inline cudaError_t SparkLmHostLaunchWeightReadAhead(
	cudaStream_t stream,const void *payload,uint64_t bytes,
	const void *auxiliary_payload,uint64_t auxiliary_bytes,uint32_t *sink_u32,
	uint32_t block_capacity)
{
	uint64_t sector_count,auxiliary_vector_count,touch_count,required_blocks;
	uint32_t block_count;
	if ( stream == 0 || payload == 0 || sink_u32 == 0 || block_capacity == 0u ||
		bytes == 0u ||
		(bytes % SPARK_LM_WEIGHT_READ_AHEAD_SECTOR_BYTES) != 0u ||
		((uintptr_t)payload % SPARK_LM_WEIGHT_READ_AHEAD_SECTOR_BYTES) != 0u ||
		((auxiliary_payload == 0) != (auxiliary_bytes == 0u)) ||
		(auxiliary_bytes % sizeof(uint4)) != 0u ||
		(auxiliary_payload != 0 &&
			((uintptr_t)auxiliary_payload % alignof(uint4)) != 0u) )
		return(cudaErrorInvalidValue);
	sector_count = bytes / SPARK_LM_WEIGHT_READ_AHEAD_SECTOR_BYTES;
	auxiliary_vector_count = auxiliary_bytes / sizeof(uint4);
	if ( auxiliary_vector_count > UINT64_MAX - sector_count )
		return(cudaErrorInvalidValue);
	touch_count = sector_count + auxiliary_vector_count;
	if ( touch_count > UINT64_MAX - (SPARK_LM_CTA_THREADS - 1u) )
		return(cudaErrorInvalidValue);
	required_blocks = (touch_count + SPARK_LM_CTA_THREADS - 1u) /
		SPARK_LM_CTA_THREADS;
	block_count = required_blocks < block_capacity ? (uint32_t)required_blocks :
		block_capacity;
	SparkLmWeightReadAheadKernel<<<block_count,SPARK_LM_CTA_THREADS,0u,stream>>>(
		(const uint4 *)payload,sector_count,(const uint4 *)auxiliary_payload,
		auxiliary_vector_count,sink_u32);
	return(cudaGetLastError());
}

// E2M1 nibble to float: sign, 2 exponent bits (bias 1), 1 mantissa bit.
static __device__ __forceinline__ float SparkLmDecodeE2m1(uint32_t nibble)
{
	const float magnitude[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
	float value = magnitude[nibble & 7u];
	return((nibble & 8u) != 0u ? -value : value);
}

// E4M3fn byte to float: 4 exponent bits bias 7, 3 mantissa bits, finite
// only, 0x7f/0xff are NaN -> zero; subnormals at 2^-9 granularity.
static __device__ __forceinline__ float SparkLmDecodeE4m3(uint32_t byte_value)
{
	uint32_t exponent = (byte_value >> 3u) & 0x0fu,mantissa = byte_value & 7u;
	float sign = (byte_value & 0x80u) != 0u ? -1.0f : 1.0f,magnitude;
	if ( (byte_value & 0x7fu) == 0x7fu )
		return(0.0f);
	if ( exponent == 0u )
		magnitude = (float)mantissa * 0.001953125f;
	else
		magnitude = (1.0f + ((float)mantissa * 0.125f)) * exp2f((float)(int32_t)exponent - 7.0f);
	return(sign * magnitude);
}

// E8M0 scale byte: pure power of two, bias 127, 0xff reserved as NaN -> zero.
static __device__ __forceinline__ float SparkLmDecodeE8m0(uint32_t byte_value)
{
	if ( byte_value == 0xffu )
		return(0.0f);
	if ( byte_value == 0u )
		return(__uint_as_float(0x00400000u));
	return(__uint_as_float(byte_value << 23u));
}

static __device__ __forceinline__ float SparkLmSigmoid(float value)
{
	return(1.0f / (1.0f + __expf(-value)));
}

static __device__ __forceinline__ float SparkLmSoftplus(float value)
{
	if ( value > 20.0f )
		return(value);
	return(log1pf(__expf(value)));
}

static __device__ __forceinline__ float SparkLmSwish(float value)
{
	return(value * SparkLmSigmoid(value));
}

/*
 * Vectorized memory primitives. Every model dimension in the family is a
 * multiple of 64, every device buffer is cudaMalloc-aligned and every row
 * stride is even, so 4-byte and 16-byte lane loads are always legal; the
 * dot loops still carry scalar tails so the kernels stay correct for any
 * dimension. One 4-byte load per lane turns a warp's weight traffic into
 * full 128-byte transactions - the difference between two and eight
 * sectors per request on every quantized format.
 */
static __device__ __forceinline__ float2 SparkLmLoadBf16Pair(const void *source, uint64_t pair_index)
{
	uint32_t raw = __ldg(((const uint32_t *)source) + pair_index);
	return(__bfloat1622float2(*(const __nv_bfloat162 *)&raw));
}

static __device__ __forceinline__ void SparkLmStoreBf16Pair(void *destination, uint64_t pair_index, float low, float high)
{
	__nv_bfloat162 packed = __floats2bfloat162_rn(low,high);
	((uint32_t *)destination)[pair_index] = *(const uint32_t *)&packed;
}

static __device__ __forceinline__ float SparkLmWarpReduceSum(float value)
{
	uint32_t offset;
	for (offset = SPARK_LM_WARP_LANES / 2u; offset != 0u; offset >>= 1u)
		value += __shfl_down_sync(0xffffffffu,value,offset);
	return(value);
}


static __device__ __forceinline__ uint64_t SparkLmOrderedTopKKey(
    float score,
    uint32_t candidate_index)
{
    uint32_t score_bits;
    uint32_t ordered_score;

    if (isnan(score))
    {
        return 0u;
    }
    if (score == 0.0f)
    {
        score = 0.0f;
    }
    score_bits = __float_as_uint(score);
    ordered_score = score_bits ^
        ((score_bits & 0x80000000u) != 0u ? 0xffffffffu : 0x80000000u);
    return ((uint64_t)ordered_score << 32u) |
        (uint64_t)(0xffffffffu - candidate_index);
}

template <uint32_t CAPACITY>
static __device__ void SparkLmBitonicSortKeysAscending(uint64_t *shared_keys)
{
    uint32_t bitonic_size;
    uint32_t bitonic_stride;
    uint32_t candidate_index;

    static_assert(CAPACITY != 0u && (CAPACITY & (CAPACITY - 1u)) == 0u,
        "bitonic capacity must be a power of two");
    for (bitonic_size = 2u; bitonic_size <= CAPACITY; bitonic_size <<= 1u)
    {
        for (bitonic_stride = bitonic_size >> 1u;
             bitonic_stride != 0u;
             bitonic_stride >>= 1u)
        {
            for (candidate_index = threadIdx.x;
                 candidate_index < CAPACITY;
                 candidate_index += blockDim.x)
            {
                uint32_t partner_index;

                partner_index = candidate_index ^ bitonic_stride;
                if (partner_index > candidate_index)
                {
                    uint64_t current_key;
                    uint64_t partner_key;
                    uint32_t ascending;

                    current_key = shared_keys[candidate_index];
                    partner_key = shared_keys[partner_index];
                    ascending =
                        (candidate_index & bitonic_size) == 0u ? 1u : 0u;
                    if ((ascending != 0u && current_key > partner_key) ||
                        (ascending == 0u && current_key < partner_key))
                    {
                        shared_keys[candidate_index] = partner_key;
                        shared_keys[partner_index] = current_key;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Block sum over blockDim.x threads; scratch must hold SPARK_LM_CTA_WARPS floats.
static __device__ float SparkLmBlockReduceSum(float value, float *scratch)
{
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES,warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t warp_count = (blockDim.x + SPARK_LM_WARP_LANES - 1u) / SPARK_LM_WARP_LANES;
	value = SparkLmWarpReduceSum(value);
	if ( lane == 0u )
		scratch[warp] = value;
	__syncthreads();
	value = (threadIdx.x < warp_count) ? scratch[threadIdx.x] : 0.0f;
	if ( warp == 0u )
		value = SparkLmWarpReduceSum(value);
	if ( threadIdx.x == 0u )
		scratch[0] = value;
	__syncthreads();
	value = scratch[0];
	__syncthreads();
	return(value);
}

// Thread-0 softmax over a small shared scalar table; every thread leaves with
// the normalized weights visible.
static __device__ void SparkLmSharedSoftmax(const float *logits, float *weights, uint32_t count)
{
	uint32_t candidate;
	float maximum,total;
	if ( threadIdx.x == 0u )
	{
		maximum = logits[0];
		for (candidate = 1; candidate < count; candidate++)
			if ( logits[candidate] > maximum )
				maximum = logits[candidate];
		total = 0.0f;
		for (candidate = 0; candidate < count; candidate++)
		{
			weights[candidate] = __expf(logits[candidate] - maximum);
			total += weights[candidate];
		}
		for (candidate = 0; candidate < count; candidate++)
			weights[candidate] /= total;
	}
	__syncthreads();
}

static __device__ __forceinline__ void SparkLmRmsNormRow(
	const void *input_bf16,const void *gain_bf16,void *output_bf16,
	uint32_t row,uint32_t dimension,float epsilon,float *staged_input,
	float *reduce_scratch)
{
	uint32_t element;
	uint64_t row_offset,tail_index;
	float sum_squares = 0.0f,inverse_rms,tail;
	float2 pair_value,gain_value;
	row_offset = ((uint64_t)row * (uint64_t)dimension) >> 1u;
	for (element=threadIdx.x; element<(dimension >> 1u); element+=blockDim.x)
	{
		pair_value = SparkLmLoadBf16Pair(input_bf16,row_offset + element);
		staged_input[element << 1u] = pair_value.x;
		staged_input[(element << 1u) + 1u] = pair_value.y;
		sum_squares = fmaf(pair_value.x,pair_value.x,sum_squares);
		sum_squares = fmaf(pair_value.y,pair_value.y,sum_squares);
	}
	if ( (dimension & 1u) != 0u && threadIdx.x == 0u )
	{
		tail_index = ((uint64_t)row * dimension) + dimension - 1u;
		tail = SparkLmBf16ToFloat(input_bf16,tail_index);
		staged_input[dimension - 1u] = tail;
		sum_squares = fmaf(tail,tail,sum_squares);
	}
	sum_squares = SparkLmBlockReduceSum(sum_squares,reduce_scratch);
	inverse_rms = rsqrtf((sum_squares / (float)dimension) + epsilon);
	__syncthreads();
	for (element=threadIdx.x; element<(dimension >> 1u); element+=blockDim.x)
	{
		gain_value = SparkLmLoadBf16Pair(gain_bf16,element);
		SparkLmStoreBf16Pair(output_bf16,row_offset + element,
			(staged_input[element << 1u] * inverse_rms) * gain_value.x,
			(staged_input[(element << 1u) + 1u] * inverse_rms) * gain_value.y);
	}
	if ( (dimension & 1u) != 0u && threadIdx.x == 0u )
	{
		tail_index = ((uint64_t)row * dimension) + dimension - 1u;
		SparkLmFloatToBf16(output_bf16,tail_index,
			staged_input[dimension - 1u] * inverse_rms *
			SparkLmBf16ToFloat(gain_bf16,dimension - 1u));
	}
}

static __global__ void SparkLmRmsNormKernel(const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	extern __shared__ float staged_input[];
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x;
	if ( row >= row_count )
		return;
	SparkLmRmsNormRow(input_bf16,gain_bf16,output_bf16,row,dimension,epsilon,
		staged_input,reduce_scratch);
}

// Fused residual-add + RMS-norm. The per-layer sequence hidden += delta then
// normalized = rmsnorm(hidden) is two full-hidden passes, each a global
// read+write. This folds the add into the norm: hidden and delta are read,
// the sum is formed in registers, the UPDATED hidden is written back, and
// the norm computes on the sum in the same pass - one read+write of hidden
// instead of two, on every layer, accuracy-identical. A null delta_bf16
// makes this a plain norm, so the single kernel serves both and the plain
// path stays byte-identical. The sum is recomputed in the second element
// loop from hidden (now holding the sum) rather than restaged, keeping the
// register footprint flat.
static __global__ void SparkLmFusedResidualRmsNormKernel(void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
    extern __shared__ float staged_hidden[];
    __shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
    uint32_t row = blockIdx.x;
    uint32_t element;
    uint64_t row_offset;
    float sum_squares = 0.0f;
    float inverse_rms;
    float sum_x;
    float sum_y;
    float2 hidden_value;
    float2 delta_value;
    float2 gain_value;

    if (row >= row_count)
    {
        return;
    }
    row_offset = ((uint64_t)row * (uint64_t)dimension) >> 1u;
    for (element = threadIdx.x; element < (dimension >> 1u); element += blockDim.x)
    {
        hidden_value = SparkLmLoadBf16Pair(hidden_bf16, row_offset + element);
        if (delta_bf16 != 0)
        {
            delta_value = SparkLmLoadBf16Pair(delta_bf16, row_offset + element);
            sum_x = hidden_value.x + delta_value.x;
            sum_y = hidden_value.y + delta_value.y;
            SparkLmStoreBf16Pair(hidden_bf16, row_offset + element, sum_x, sum_y);
        }
        else
        {
            sum_x = hidden_value.x;
            sum_y = hidden_value.y;
        }
        staged_hidden[element << 1u] = sum_x;
        staged_hidden[(element << 1u) + 1u] = sum_y;
        sum_squares = fmaf(sum_x, sum_x, sum_squares);
        sum_squares = fmaf(sum_y, sum_y, sum_squares);
    }
    if ((dimension & 1u) != 0u && threadIdx.x == 0u)
    {
        uint64_t tail_index = ((uint64_t)row * dimension) + dimension - 1u;
        float tail = SparkLmBf16ToFloat(hidden_bf16, tail_index);
        if (delta_bf16 != 0)
        {
            tail += SparkLmBf16ToFloat(delta_bf16, tail_index);
            SparkLmFloatToBf16(hidden_bf16, tail_index, tail);
        }
        staged_hidden[dimension - 1u] = tail;
        sum_squares = fmaf(tail, tail, sum_squares);
    }
    sum_squares = SparkLmBlockReduceSum(sum_squares, reduce_scratch);
    inverse_rms = rsqrtf((sum_squares / (float)dimension) + epsilon);
    __syncthreads();

    for (element = threadIdx.x; element < (dimension >> 1u); element += blockDim.x)
    {
        gain_value = SparkLmLoadBf16Pair(gain_bf16, element);
        SparkLmStoreBf16Pair(
            output_bf16,
            row_offset + element,
            (staged_hidden[element << 1u] * inverse_rms) * gain_value.x,
            (staged_hidden[(element << 1u) + 1u] * inverse_rms) * gain_value.y);
    }
    if ((dimension & 1u) != 0u && threadIdx.x == 0u)
    {
        uint64_t tail_index = ((uint64_t)row * dimension) + dimension - 1u;
        SparkLmFloatToBf16(
            output_bf16,
            tail_index,
            staged_hidden[dimension - 1u] * inverse_rms * SparkLmBf16ToFloat(gain_bf16, dimension - 1u));
    }
}

/*
 * Row-major linear, one warp per output neuron, eight neurons in flight per
 * block, activations staged once in shared memory. The weight branch is
 * launch-uniform (bf16 or MXFP4 nibble pairs with one E8M0 scale per group),
 * so both formats share the loop. Weight fetch is the bound at decode batch
 * sizes, which this layout keeps fully coalesced. GROUP_SIZE is the MXFP4
 * scale group and must match the stage pack; it is a template parameter so a
 * mismatch is a compile error at the launch site, never a silent default.
 */
static __device__ __forceinline__ float SparkLmDotRowBf16(const float *shared_input, const void *weight_payload, uint32_t neuron, uint32_t input_dimension, uint32_t lane)
{
	uint64_t pair_row = ((uint64_t)neuron * input_dimension) >> 1u;
	uint32_t pair_count = input_dimension >> 1u,pair,element;
	float accumulator = 0.0f;
	float2 pair_value;
	#pragma unroll 4
	for (pair = lane; pair < pair_count; pair += SPARK_LM_WARP_LANES)
	{
		pair_value = SparkLmLoadBf16Pair(weight_payload,pair_row + pair);
		accumulator = fmaf(shared_input[pair << 1u],pair_value.x,accumulator);
		accumulator = fmaf(shared_input[(pair << 1u) + 1u],pair_value.y,accumulator);
	}
	for (element = (pair_count << 1u) + lane; element < input_dimension; element += SPARK_LM_WARP_LANES)
		accumulator += (shared_input[element] * SparkLmBf16ToFloat(weight_payload,((uint64_t)neuron * input_dimension) + element));
	return(accumulator);
}

static __device__ __forceinline__ void SparkLmDotRowBf16Pair(
	const float *shared_input,const void *first_weight,
	const void *second_weight,uint32_t neuron,uint32_t input_dimension,
	uint32_t lane,float *first_total,float *second_total)
{
	uint64_t pair_row = ((uint64_t)neuron * input_dimension) >> 1u;
	uint32_t pair_count = input_dimension >> 1u,pair,element;
	float first = 0.0f,second = 0.0f;
	float2 first_pair,second_pair;
	#pragma unroll 4
	for (pair=lane; pair<pair_count; pair+=SPARK_LM_WARP_LANES)
	{
		first_pair = SparkLmLoadBf16Pair(first_weight,pair_row + pair);
		second_pair = SparkLmLoadBf16Pair(second_weight,pair_row + pair);
		first = fmaf(shared_input[pair << 1u],first_pair.x,first);
		first = fmaf(shared_input[(pair << 1u) + 1u],first_pair.y,first);
		second = fmaf(shared_input[pair << 1u],second_pair.x,second);
		second = fmaf(shared_input[(pair << 1u) + 1u],second_pair.y,second);
	}
	for (element=(pair_count << 1u) + lane; element<input_dimension;
		element+=SPARK_LM_WARP_LANES)
	{
		first += shared_input[element] * SparkLmBf16ToFloat(first_weight,
			((uint64_t)neuron * input_dimension) + element);
		second += shared_input[element] * SparkLmBf16ToFloat(second_weight,
			((uint64_t)neuron * input_dimension) + element);
	}
	*first_total = first;
	*second_total = second;
}

typedef union
{
	uint32_t bits;
	__half2 values;
} SparkLmHalf2Bits;

// Decode full payload words with native packed converts. The scalar GEMV
// consumes float pairs, but it must not pay one transcendental/table path per
// weight while the target can widen the packed checkpoint representation.
static __device__ __forceinline__ void SparkLmDecodeE2m1x8Half2(uint32_t packed, uint32_t decoded[4])
{
#if LM_SM121_NATIVE_COMPUTE_PTX
	asm volatile(
		"{\n\t"
		".reg .b8 b0, b1, b2, b3;\n\t"
		"mov.b32 {b0, b1, b2, b3}, %4;\n\t"
		"cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
		"cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
		"cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
		"cvt.rn.f16x2.e2m1x2 %3, b3;\n\t"
		"}"
		: "=r"(decoded[0]),"=r"(decoded[1]),"=r"(decoded[2]),"=r"(decoded[3])
		: "r"(packed));
#else
	SparkLmHalf2Bits value;
	uint32_t pair;
	#pragma unroll
	for (pair=0u; pair<4u; pair++)
	{
		value.values = __floats2half2_rn(SparkLmDecodeE2m1((packed >> (pair * 8u)) & 0x0fu),SparkLmDecodeE2m1((packed >> ((pair * 8u) + 4u)) & 0x0fu));
		decoded[pair] = value.bits;
	}
#endif
}

static __device__ __forceinline__ void SparkLmDecodeE4m3x4Half2(uint32_t packed, uint32_t decoded[2])
{
#if LM_SM121_NATIVE_COMPUTE_PTX
	uint32_t invalid_high = (((packed & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u);
	uint32_t invalid_mask = invalid_high | (invalid_high - (invalid_high >> 7u));
	packed &= ~invalid_mask;
	asm volatile(
		"{\n\t"
		".reg .b16 lo, hi;\n\t"
		"mov.b32 {lo, hi}, %2;\n\t"
		"cvt.rn.f16x2.e4m3x2 %0, lo;\n\t"
		"cvt.rn.f16x2.e4m3x2 %1, hi;\n\t"
		"}"
		: "=r"(decoded[0]),"=r"(decoded[1])
		: "r"(packed));
#else
	SparkLmHalf2Bits value;
	uint32_t pair;
	#pragma unroll
	for (pair=0u; pair<2u; pair++)
	{
		value.values = __floats2half2_rn(SparkLmDecodeE4m3((packed >> (pair * 16u)) & 0xffu),SparkLmDecodeE4m3((packed >> ((pair * 16u) + 8u)) & 0xffu));
		decoded[pair] = value.bits;
	}
#endif
}

// One 4-byte load carries eight E2M1 elements; an eight-element run at an
// eight-aligned base never crosses a scale group (eight divides every
// GROUP_SIZE in use), so each run costs one scale decode.
template <uint32_t GROUP_SIZE>
static __device__ __forceinline__ float SparkLmDotRowMxfp4(const float *shared_input, const void *weight_payload, const uint8_t *weight_scale_e8m0, uint32_t neuron, uint32_t input_dimension, uint32_t lane)
{
	uint64_t run_row = ((uint64_t)neuron * input_dimension) >> 3u,scale_row = (uint64_t)neuron * (input_dimension / GROUP_SIZE);
	uint32_t run_count = input_dimension >> 3u,run,pair,packed,decoded[4];
	float accumulator = 0.0f,scale_value;
	float2 values;
	SparkLmHalf2Bits half2_bits;
	#pragma unroll 2
	for (run = lane; run < run_count; run += SPARK_LM_WARP_LANES)
	{
		packed = __ldg(((const uint32_t *)weight_payload) + run_row + run);
		scale_value = SparkLmDecodeE8m0(weight_scale_e8m0[scale_row + ((run << 3u) / GROUP_SIZE)]);
		SparkLmDecodeE2m1x8Half2(packed,decoded);
		#pragma unroll
		for (pair = 0; pair < 4u; pair++)
		{
			half2_bits.bits = decoded[pair];
			values = __half22float2(half2_bits.values);
			accumulator = fmaf(shared_input[(run << 3u) + (pair << 1u)],values.x * scale_value,accumulator);
			accumulator = fmaf(shared_input[(run << 3u) + (pair << 1u) + 1u],values.y * scale_value,accumulator);
		}
	}
	return(accumulator);
}

template <uint32_t GROUP_SIZE>
static __device__ __forceinline__ void SparkLmDotRowMxfp4Pair(const float *shared_input, const void *first_payload, const uint8_t *first_scale_e8m0, const void *second_payload, const uint8_t *second_scale_e8m0, uint32_t neuron, uint32_t input_dimension, uint32_t lane, float *first_total, float *second_total)
{
	uint64_t run_row = ((uint64_t)neuron * input_dimension) >> 3u,scale_row = (uint64_t)neuron * (input_dimension / GROUP_SIZE);
	uint32_t run_count = input_dimension >> 3u,run,pair,first_packed,second_packed,first_decoded[4],second_decoded[4];
	float first_value = 0.0f,second_value = 0.0f,first_scale,second_scale;
	float2 values;
	SparkLmHalf2Bits half2_bits;
	#pragma unroll 2
	for (run=lane; run<run_count; run+=SPARK_LM_WARP_LANES)
	{
		first_packed = __ldg(((const uint32_t *)first_payload) + run_row + run);
		second_packed = __ldg(((const uint32_t *)second_payload) + run_row + run);
		first_scale = SparkLmDecodeE8m0(first_scale_e8m0[scale_row + ((run << 3u) / GROUP_SIZE)]);
		second_scale = SparkLmDecodeE8m0(second_scale_e8m0[scale_row + ((run << 3u) / GROUP_SIZE)]);
		SparkLmDecodeE2m1x8Half2(first_packed,first_decoded);
		SparkLmDecodeE2m1x8Half2(second_packed,second_decoded);
		#pragma unroll
		for (pair=0u; pair<4u; pair++)
		{
			half2_bits.bits = first_decoded[pair];
			values = __half22float2(half2_bits.values);
			first_value = fmaf(shared_input[(run << 3u) + (pair << 1u)],values.x * first_scale,first_value);
			first_value = fmaf(shared_input[(run << 3u) + (pair << 1u) + 1u],values.y * first_scale,first_value);
			half2_bits.bits = second_decoded[pair];
			values = __half22float2(half2_bits.values);
			second_value = fmaf(shared_input[(run << 3u) + (pair << 1u)],values.x * second_scale,second_value);
			second_value = fmaf(shared_input[(run << 3u) + (pair << 1u) + 1u],values.y * second_scale,second_value);
		}
	}
	*first_total = first_value;
	*second_total = second_value;
}

// FP8 weights with F32 scales on [BLOCK, BLOCK] 2-D tiles - the MiMo
// checkpoint layout: element (r, c) multiplies scale[(r/B)*(C/B) + c/B].
template <uint32_t BLOCK>
static __device__ __forceinline__ float SparkLmDotRowFp8F32(const float *shared_input, const void *weight_payload, const float *weight_scale_f32, uint32_t neuron, uint32_t input_dimension, uint32_t lane)
{
	uint64_t run_row = ((uint64_t)neuron * input_dimension) >> 2u,scale_row = ((uint64_t)(neuron / BLOCK)) * (input_dimension / BLOCK);
	uint32_t run_count = input_dimension >> 2u,run,pair,packed,decoded[2];
	float accumulator = 0.0f,scale_value;
	float2 values;
	SparkLmHalf2Bits half2_bits;
	#pragma unroll 4
	for (run = lane; run < run_count; run += SPARK_LM_WARP_LANES)
	{
		packed = __ldg(((const uint32_t *)weight_payload) + run_row + run);
		scale_value = __ldg(weight_scale_f32 + scale_row + ((run << 2u) / BLOCK));
		/* Native SM121 e4m3x2 decode (cvt.rn.f16x2.e4m3x2) into f16 pairs:
		 * e4m3 is exactly representable in f16, so this is bit-lossless versus
		 * the old per-byte SparkLmDecodeE4m3, whose exp2f() transcendental made
		 * the FP8 dot compute-bound instead of memory-bound. */
		SparkLmDecodeE4m3x4Half2(packed,decoded);
		#pragma unroll
		for (pair = 0u; pair < 2u; pair++)
		{
			half2_bits.bits = decoded[pair];
			values = __half22float2(half2_bits.values);
			accumulator = fmaf(shared_input[(run << 2u) + (pair << 1u)],values.x * scale_value,accumulator);
			accumulator = fmaf(shared_input[(run << 2u) + (pair << 1u) + 1u],values.y * scale_value,accumulator);
		}
	}
	return(accumulator);
}

// FP8 weights: one e4m3 byte per element, one E8M0 scale per GROUP_SIZE
// columns per row - the DeepSeek block-quantized layout.
template <uint32_t GROUP_SIZE>
static __device__ __forceinline__ float SparkLmDotRowFp8(const float *shared_input, const void *weight_payload, const uint8_t *weight_scale_e8m0, uint32_t neuron, uint32_t input_dimension, uint32_t lane)
{
	uint64_t run_row = ((uint64_t)neuron * input_dimension) >> 2u,scale_row = (uint64_t)neuron * (input_dimension / GROUP_SIZE);
	uint32_t run_count = input_dimension >> 2u,run,pair,packed,decoded[2];
	float accumulator = 0.0f,scale_value;
	float2 values;
	SparkLmHalf2Bits half2_bits;
	#pragma unroll 4
	for (run = lane; run < run_count; run += SPARK_LM_WARP_LANES)
	{
		packed = __ldcs(((const uint32_t *)weight_payload) + run_row + run);
		scale_value = SparkLmDecodeE8m0(weight_scale_e8m0[scale_row + ((run << 2u) / GROUP_SIZE)]);
		SparkLmDecodeE4m3x4Half2(packed,decoded);
		#pragma unroll
		for (pair = 0; pair < 2u; pair++)
		{
			half2_bits.bits = decoded[pair];
			values = __half22float2(half2_bits.values);
			accumulator = fmaf(shared_input[(run << 2u) + (pair << 1u)],values.x * scale_value,accumulator);
			accumulator = fmaf(shared_input[(run << 2u) + (pair << 1u) + 1u],values.y * scale_value,accumulator);
		}
	}
	return(accumulator);
}

template <uint32_t GROUP_SIZE>
static __device__ __forceinline__ void SparkLmDotRowFp8Pair(
	const float *shared_input,const void *first_payload,
	const uint8_t *first_scale_e8m0,const void *second_payload,
	const uint8_t *second_scale_e8m0,uint32_t neuron,
	uint32_t input_dimension,uint32_t lane,float *first_total,
	float *second_total)
{
	uint64_t run_row = ((uint64_t)neuron * input_dimension) >> 2u;
	uint64_t scale_row = (uint64_t)neuron * (input_dimension / GROUP_SIZE);
	uint32_t run_count = input_dimension >> 2u,run,pair;
	uint32_t first_packed,second_packed,first_decoded[2],second_decoded[2];
	float first = 0.0f,second = 0.0f,first_scale,second_scale;
	float2 values;
	SparkLmHalf2Bits half2_bits;
	#pragma unroll 4
	for (run=lane; run<run_count; run+=SPARK_LM_WARP_LANES)
	{
		first_packed = __ldg(((const uint32_t *)first_payload) + run_row + run);
		second_packed = __ldg(((const uint32_t *)second_payload) + run_row + run);
		first_scale = SparkLmDecodeE8m0(first_scale_e8m0[scale_row +
			((run << 2u) / GROUP_SIZE)]);
		second_scale = SparkLmDecodeE8m0(second_scale_e8m0[scale_row +
			((run << 2u) / GROUP_SIZE)]);
		SparkLmDecodeE4m3x4Half2(first_packed,first_decoded);
		SparkLmDecodeE4m3x4Half2(second_packed,second_decoded);
		#pragma unroll
		for (pair=0u; pair<2u; pair++)
		{
			half2_bits.bits = first_decoded[pair];
			values = __half22float2(half2_bits.values);
			first = fmaf(shared_input[(run << 2u) + (pair << 1u)],
				values.x * first_scale,first);
			first = fmaf(shared_input[(run << 2u) + (pair << 1u) + 1u],
				values.y * first_scale,first);
			half2_bits.bits = second_decoded[pair];
			values = __half22float2(half2_bits.values);
			second = fmaf(shared_input[(run << 2u) + (pair << 1u)],
				values.x * second_scale,second);
			second = fmaf(shared_input[(run << 2u) + (pair << 1u) + 1u],
				values.y * second_scale,second);
		}
	}
	*first_total = first;
	*second_total = second;
}

// Token embedding gather is a pure row copy: 16-byte vector moves, no
// conversion round trip, scalar tail for a non-multiple-of-eight hidden.
static __global__ void SparkLmEmbeddingGatherKernel(const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t hidden_dimension)
{
	uint32_t row = blockIdx.x,element,vector_count = hidden_dimension >> 3u;
	uint64_t source_offset,destination_offset;
	if ( row >= row_count )
		return;
	source_offset = (uint64_t)token_ids[row] * hidden_dimension;
	destination_offset = (uint64_t)row * hidden_dimension;
	for (element = threadIdx.x; element < vector_count; element += blockDim.x)
		((uint4 *)hidden_bf16)[(destination_offset >> 3u) + element] = __ldg(((const uint4 *)embedding_bf16) + (source_offset >> 3u) + element);
	for (element = (vector_count << 3u) + threadIdx.x; element < hidden_dimension; element += blockDim.x)
		SparkLmFloatToBf16(hidden_bf16,destination_offset + element,SparkLmBf16ToFloat(embedding_bf16,source_offset + element));
}

template <uint32_t GROUP_SIZE>
static __device__ __forceinline__ float SparkLmDotLinearRow(
	uint32_t weight_format,
	const float *shared_input,
	const void *weight_payload,
	const void *weight_scale,
	uint32_t neuron,
	uint32_t input_dimension,
	uint32_t lane)
{
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_BF16 )
		return(SparkLmDotRowBf16(shared_input,weight_payload,neuron,
			input_dimension,lane));
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 )
		return(SparkLmDotRowFp8<GROUP_SIZE>(shared_input,weight_payload,
			(const uint8_t *)weight_scale,neuron,input_dimension,lane));
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		return(SparkLmDotRowFp8F32<128u>(shared_input,weight_payload,
			(const float *)weight_scale,neuron,input_dimension,lane));
	return(SparkLmDotRowMxfp4<GROUP_SIZE>(shared_input,weight_payload,
		(const uint8_t *)weight_scale,neuron,input_dimension,lane));
}

template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC,uint32_t CTA_WARPS>
static __global__ void SparkLmLinearKernel(uint32_t weight_format, const void *weight_payload, const void *weight_scale, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	extern __shared__ float shared_input[];
	uint32_t row = blockIdx.x,neuron_base =
		blockIdx.y * CTA_WARPS;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t neuron,element;
	float accumulator;
	float2 stage_pair;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (input_dimension >> 1u); element += blockDim.x)
	{
		stage_pair = SparkLmLoadBf16Pair(input_bf16,(((uint64_t)row * input_dimension) >> 1u) + element);
		shared_input[element << 1u] = stage_pair.x;
		shared_input[(element << 1u) + 1u] = stage_pair.y;
	}
	for (element = ((input_dimension >> 1u) << 1u) + threadIdx.x; element < input_dimension; element += blockDim.x)
		shared_input[element] = SparkLmBf16ToFloat(input_bf16,((uint64_t)row * input_dimension) + element);
	__syncthreads();
	if constexpr ( ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0 )
	{
		LmActivationFp8QdqFloatRow<ACTIVATION_CODEC>(shared_input,input_dimension);
		__syncthreads();
	}
	neuron = neuron_base + warp;
	if ( neuron < output_dimension )
	{
		accumulator = SparkLmDotLinearRow<GROUP_SIZE>(weight_format,
			shared_input,weight_payload,weight_scale,neuron,input_dimension,lane);
		accumulator = SparkLmWarpReduceSum(accumulator);
		if ( lane == 0u )
			SparkLmFloatToBf16(output_bf16,
				((uint64_t)row * output_dimension) + neuron,accumulator);
	}
}

template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC,uint32_t CTA_WARPS>
static __global__ void SparkLmFp8LinearPairKernel(
	const void *first_payload,const uint8_t *first_scale,
	const void *second_payload,const uint8_t *second_scale,
	const void *input_bf16,void *first_output_bf16,void *second_output_bf16,
	uint32_t input_dimension,uint32_t first_output_dimension,
	uint32_t second_output_dimension)
{
	extern __shared__ float shared_input[];
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t neuron = blockIdx.x * CTA_WARPS + warp;
	uint32_t element,pair_count = input_dimension >> 1u,second_neuron;
	float accumulator;
	float2 input_pair;
	for (element=threadIdx.x; element<pair_count; element+=blockDim.x)
	{
		input_pair = SparkLmLoadBf16Pair(input_bf16,element);
		shared_input[element << 1u] = input_pair.x;
		shared_input[(element << 1u) + 1u] = input_pair.y;
	}
	for (element=(pair_count << 1u) + threadIdx.x;
		element<input_dimension; element+=blockDim.x)
		shared_input[element] = SparkLmBf16ToFloat(input_bf16,element);
	__syncthreads();
	if constexpr ( ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0 )
	{
		LmActivationFp8QdqFloatRow<ACTIVATION_CODEC>(shared_input,
			input_dimension);
		__syncthreads();
	}
	if ( neuron < first_output_dimension )
	{
		accumulator = SparkLmDotRowFp8<GROUP_SIZE>(shared_input,first_payload,
			first_scale,neuron,input_dimension,lane);
		accumulator = SparkLmWarpReduceSum(accumulator);
		if ( lane == 0u )
			SparkLmFloatToBf16(first_output_bf16,neuron,accumulator);
	}
	else if ( neuron < first_output_dimension + second_output_dimension )
	{
		second_neuron = neuron - first_output_dimension;
		accumulator = SparkLmDotRowFp8<GROUP_SIZE>(shared_input,second_payload,
			second_scale,second_neuron,input_dimension,lane);
		accumulator = SparkLmWarpReduceSum(accumulator);
		if ( lane == 0u )
			SparkLmFloatToBf16(second_output_bf16,second_neuron,accumulator);
	}
}

template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC=SPARK_ACTIVATION_CODEC_NONE>
static __global__ void SparkLmStridedLinearKernel(uint32_t weight_format, const void *weight_payload, const uint8_t *weight_scale, uint64_t weight_payload_group_stride_bytes, uint64_t weight_scale_group_stride_bytes, const void *input_bf16, uint64_t input_row_stride, uint32_t input_offset, uint32_t input_group_stride, void *output_bf16, uint64_t output_row_stride, uint32_t output_offset, uint32_t output_group_stride, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	extern __shared__ float shared_input[];
	uint32_t row = blockIdx.x,group = blockIdx.z,neuron_base = blockIdx.y * SPARK_LM_CTA_WARPS;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES,neuron,element;
	const uint8_t *payload = (const uint8_t *)weight_payload + ((uint64_t)group * weight_payload_group_stride_bytes);
	const uint8_t *scale = weight_scale != 0 ? weight_scale + ((uint64_t)group * weight_scale_group_stride_bytes) : 0;
	float accumulator;
	input_offset += group * input_group_stride;
	output_offset += group * output_group_stride;
	if ( row >= row_count )
		return;
	for (element=threadIdx.x; element<input_dimension; element+=blockDim.x)
		shared_input[element] = SparkLmBf16ToFloat(input_bf16,((uint64_t)row * input_row_stride) + input_offset + element);
	__syncthreads();
	if constexpr ( ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0 )
	{
		LmActivationFp8QdqFloatRow<ACTIVATION_CODEC>(shared_input,input_dimension);
		__syncthreads();
	}
	neuron = neuron_base + warp;
	if ( neuron < output_dimension )
	{
		accumulator = SparkLmDotLinearRow<GROUP_SIZE>(weight_format,
			shared_input,payload,scale,neuron,input_dimension,lane);
		accumulator = SparkLmWarpReduceSum(accumulator);
		if ( lane == 0u )
			SparkLmFloatToBf16(output_bf16,
				((uint64_t)row * output_row_stride) + output_offset + neuron,
				accumulator);
	}
}

template <uint32_t CTA_WARPS>
static __global__ void SparkLmBf16LinearPairKernel(
	const void *first_weight,const void *second_weight,const void *input_bf16,
	void *first_output_bf16,void *second_output_bf16,uint32_t row_count,
	uint32_t input_dimension,uint32_t output_dimension)
{
	extern __shared__ float shared_input[];
	uint32_t row = blockIdx.x;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t combined_neuron = blockIdx.y * CTA_WARPS + warp;
	uint32_t neuron;
	uint32_t element,pair,pair_count = input_dimension >> 1u;
	const void *weight;
	void *output_bf16;
	float accumulator;
	float2 input_pair;
	if ( row >= row_count )
		return;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		input_pair = SparkLmLoadBf16Pair(input_bf16,
			(((uint64_t)row * input_dimension) >> 1u) + pair);
		shared_input[pair << 1u] = input_pair.x;
		shared_input[(pair << 1u) + 1u] = input_pair.y;
	}
	for (element=(pair_count << 1u) + threadIdx.x;
		element<input_dimension; element+=blockDim.x)
		shared_input[element] = SparkLmBf16ToFloat(input_bf16,
			((uint64_t)row * input_dimension) + element);
	__syncthreads();
	if ( combined_neuron >= output_dimension * 2u )
		return;
	if ( combined_neuron < output_dimension )
	{
		neuron = combined_neuron;
		weight = first_weight;
		output_bf16 = first_output_bf16;
	}
	else
	{
		neuron = combined_neuron - output_dimension;
		weight = second_weight;
		output_bf16 = second_output_bf16;
	}
	accumulator = SparkLmDotRowBf16(shared_input,weight,neuron,
		input_dimension,lane);
	accumulator = SparkLmWarpReduceSum(accumulator);
	if ( lane == 0u )
		SparkLmFloatToBf16(output_bf16,
			((uint64_t)row * output_dimension) + neuron,accumulator);
}

/*
 * Expert-batched machinery: the serving plane groups a decode batch's
 * routed rows by expert (PR497's expert-queue regime emits exactly these
 * per-expert row lists), and the driver turns each group into batched
 * launches over an indirected row map. Slots are dense per group; the
 * scatter applies the routing weight to the expert OUTPUT (or the caller
 * pre-applies it at the intermediate) and adds into the true row. No
 * atomics: within one expert's group every target row is distinct.
 */
template <uint32_t GROUP_SIZE>
static __global__ void SparkLmGatherLinearKernel(uint32_t weight_format, const void *weight_payload, const void *weight_scale, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count, uint32_t input_dimension, uint32_t output_dimension)
{
	extern __shared__ float shared_input[];
	uint32_t slot = blockIdx.x,neuron_base = blockIdx.y * SPARK_LM_CTA_WARPS;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES,neuron,element,source_row;
	float accumulator;
	float2 stage_pair;
	if ( slot >= slot_count )
		return;
	source_row = input_row_map != 0 ? input_row_map[slot] : slot;
	for (element = threadIdx.x; element < (input_dimension >> 1u); element += blockDim.x)
	{
		stage_pair = SparkLmLoadBf16Pair(input_bf16,(((uint64_t)source_row * input_dimension) >> 1u) + element);
		shared_input[element << 1u] = stage_pair.x;
		shared_input[(element << 1u) + 1u] = stage_pair.y;
	}
	for (element = ((input_dimension >> 1u) << 1u) + threadIdx.x; element < input_dimension; element += blockDim.x)
		shared_input[element] = SparkLmBf16ToFloat(input_bf16,((uint64_t)source_row * input_dimension) + element);
	__syncthreads();
	neuron = neuron_base + warp;
	if ( neuron >= output_dimension )
		return;
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_BF16 )
		accumulator = SparkLmDotRowBf16(shared_input,weight_payload,neuron,input_dimension,lane);
	else if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 )
		accumulator = SparkLmDotRowFp8<GROUP_SIZE>(shared_input,weight_payload,(const uint8_t *)weight_scale,neuron,input_dimension,lane);
	else if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		accumulator = SparkLmDotRowFp8F32<128u>(shared_input,weight_payload,(const float *)weight_scale,neuron,input_dimension,lane);
	else
		accumulator = SparkLmDotRowMxfp4<GROUP_SIZE>(shared_input,weight_payload,(const uint8_t *)weight_scale,neuron,input_dimension,lane);
	accumulator = SparkLmWarpReduceSum(accumulator);
	if ( lane == 0u )
		SparkLmFloatToBf16(output_bf16,((uint64_t)slot * output_dimension) + neuron,accumulator);
}

// destination[row_map[slot]] += source[slot] * weights[weight_map[slot]];
// weight_map (and the weight scaling) optional for a plain scatter add.
static __global__ void SparkLmScatterScaledAddKernel(void *destination_bf16, const void *source_bf16, const uint32_t *row_map, const float *weights_f32, const uint32_t *weight_map, uint32_t slot_count, uint32_t width)
{
	uint32_t slot = blockIdx.x,element,target;
	uint64_t source_offset,target_offset;
	float weight = 1.0f;
	float2 destination_pair,source_pair;
	if ( slot >= slot_count )
		return;
	target = row_map != 0 ? row_map[slot] : slot;
	if ( weights_f32 != 0 )
		weight = weights_f32[weight_map != 0 ? weight_map[slot] : slot];
	source_offset = ((uint64_t)slot * width) >> 1u;
	target_offset = ((uint64_t)target * width) >> 1u;
	for (element = threadIdx.x; element < (width >> 1u); element += blockDim.x)
	{
		destination_pair = SparkLmLoadBf16Pair(destination_bf16,target_offset + element);
		source_pair = SparkLmLoadBf16Pair(source_bf16,source_offset + element);
		SparkLmStoreBf16Pair(destination_bf16,target_offset + element,fmaf(source_pair.x,weight,destination_pair.x),fmaf(source_pair.y,weight,destination_pair.y));
	}
	for (element = ((width >> 1u) << 1u) + threadIdx.x; element < width; element += blockDim.x)
		SparkLmFloatToBf16(destination_bf16,((uint64_t)target * width) + element,SparkLmBf16ToFloat(destination_bf16,((uint64_t)target * width) + element) + SparkLmBf16ToFloat(source_bf16,((uint64_t)slot * width) + element) * weight);
}

// One weight element decoded to float for the tile loader, every format
// through the same dot-helper arithmetic.
template <uint32_t GROUP_SIZE>
static __device__ __forceinline__ float SparkLmWeightElement(uint32_t weight_format, const void *weight_payload, const void *weight_scale, uint32_t neuron, uint32_t element, uint32_t input_dimension)
{
	uint64_t row_offset;
	uint32_t packed;
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_BF16 )
		return(SparkLmBf16ToFloat(weight_payload,((uint64_t)neuron * input_dimension) + element));
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 )
		return(SparkLmDecodeE4m3(((const uint8_t *)weight_payload)[((uint64_t)neuron * input_dimension) + element]) * SparkLmDecodeE8m0(((const uint8_t *)weight_scale)[((uint64_t)neuron * (input_dimension / GROUP_SIZE)) + (element / GROUP_SIZE)]));
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		return(SparkLmDecodeE4m3(((const uint8_t *)weight_payload)[((uint64_t)neuron * input_dimension) + element]) * ((const float *)weight_scale)[(((uint64_t)(neuron / 128u)) * (input_dimension / 128u)) + (element / 128u)]);
	row_offset = (uint64_t)neuron * (input_dimension / 2u);
	packed = ((const uint8_t *)weight_payload)[row_offset + (element >> 1u)];
	return(SparkLmDecodeE2m1((element & 1u) != 0u ? packed >> 4u : packed & 0x0fu) * SparkLmDecodeE8m0(((const uint8_t *)weight_scale)[((uint64_t)neuron * (input_dimension / GROUP_SIZE)) + (element / GROUP_SIZE)]));
}

/*
 * Argmax over a precomputed logits matrix, the second phase of the
 * two-phase head: the tile GEMM writes logits [row][candidate] with the
 * head weights read once per SIXTEEN rows instead of once per row - the
 * old one-block-per-row argmax streamed the entire vocab-by-hidden
 * matrix per row, sixteen times the traffic of this split at a full row
 * tile. Warps stripe the candidates with paired loads, ties go to the
 * lower index, and the optional id table maps restricted candidates.
 */
// Warp-then-block argmax reduce with ties to the lower index; thread
// zero leaves the winner in the scratch slots' zeroth entries.
static __device__ void SparkLmArgmaxReduce(float running_best, uint32_t running_candidate, float *best_score, uint32_t *best_candidate)
{
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES,offset,candidate,shuffle_candidate;
	float shuffle_score;
	for (offset = SPARK_LM_WARP_LANES / 2u; offset != 0u; offset >>= 1u)
	{
		shuffle_score = __shfl_down_sync(0xffffffffu,running_best,offset);
		shuffle_candidate = __shfl_down_sync(0xffffffffu,running_candidate,offset);
		if ( shuffle_score > running_best || (shuffle_score == running_best && shuffle_candidate < running_candidate) )
		{
			running_best = shuffle_score;
			running_candidate = shuffle_candidate;
		}
	}
	if ( lane == 0u )
	{
		best_score[warp] = running_best;
		best_candidate[warp] = running_candidate;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		for (candidate = 1; candidate < SPARK_LM_CTA_WARPS; candidate++)
			if ( best_score[candidate] > best_score[0] || (best_score[candidate] == best_score[0] && best_candidate[candidate] < best_candidate[0]) )
			{
				best_score[0] = best_score[candidate];
				best_candidate[0] = best_candidate[candidate];
			}
	}
	__syncthreads();
}

// Block max of a per-thread scalar into a shared float, warp shuffles
// then a thread-zero scan; both barriers included.
static __device__ void SparkLmAttnBlockScalarMax(float local_maximum, float *scratch, float *shared_maximum)
{
	uint32_t offset,warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES;
	float value;
	for (offset = SPARK_LM_WARP_LANES / 2u; offset != 0u; offset >>= 1u)
	{
		value = __shfl_down_sync(0xffffffffu,local_maximum,offset);
		if ( value > local_maximum )
			local_maximum = value;
	}
	if ( lane == 0u )
		scratch[warp] = local_maximum;
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		for (offset = 1; offset < SPARK_LM_CTA_WARPS; offset++)
			if ( scratch[offset] > scratch[0] )
				scratch[0] = scratch[offset];
		*shared_maximum = scratch[0];
	}
	__syncthreads();
}

/*
 * Screened full-vocabulary head: an MXFP4 shadow of the lm_head (built
 * once at initialize, one third the bytes) produces coarse logits
 * through the tensor-core tile; the screen then keeps only candidates
 * whose coarse logit PLUS its certified error bound reaches the best
 * guaranteed lower bound, and the exact bf16 rescore touches just those
 * rows. The bound is |exact - coarse| <= ||hidden||_2 * e_n with e_n
 * the neuron's quantization error norm precomputed at shadow build, so
 * the true argmax is provably in the candidate set and the emitted
 * token EQUALS the reference argmax, deterministically. Rows whose set
 * overflows the cap fall back to the exact full scan on device - no
 * host round trip anywhere. Idle compute buys a ~4x cut of the head
 * stage's dominant memory stream.
 */
#define SPARK_LM_HEAD_SCREEN_CAP 4096u
#define SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT 128u
#define SPARK_LM_HEAD_FALLBACK_ROW_GROUP_MAX 4u
#define SPARK_LM_HEAD_FALLBACK_SHARED_BYTES 32768u

// Rounding slack for the SCREEN only: the certified e_n bound covers
// the fp4 weight error, this covers the bf16 store of the coarse logit
// and the tile's own accumulate rounding, relative-aware so large
// logits stay sound. Applied on both sides of the keep test.
static __device__ __forceinline__ float SparkLmHeadScreenSlack(float coarse)
{
	return(1.0f + (fabsf(coarse) * 0.0078125f));
}
#define SPARK_LM_HEAD_SHADOW_GROUP 32u

// Nearest E2M1 code for value / scale, magnitude set {0,.5,1,1.5,2,3,4,6}.
static __device__ __forceinline__ uint32_t SparkLmEncodeE2m1(float value)
{
	const float edges[7] = {0.25f,0.75f,1.25f,1.75f,2.5f,3.5f,5.0f};
	uint32_t sign = value < 0.0f ? 8u : 0u,code = 0u,edge;
	float magnitude = fabsf(value);
	for (edge = 0; edge < 7u; edge++)
		if ( magnitude >= edges[edge] )
			code = edge + 1u;
	return(sign | code);
}

// One warp per neuron row: per-group absmax to an E8M0 scale (smallest
// power of two whose 6.0 span covers the group), nibble encode, and the
// accumulated squared error reduced into the neuron's certified bound.
static __global__ void SparkLmHeadShadowQuantizeKernel(const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale_e8m0, float *error_norm_f32, uint32_t candidate_count, uint32_t hidden_dimension)
{
	uint32_t neuron = (blockIdx.x * SPARK_LM_CTA_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES);
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES,group,element,packed,nibble,exponent;
	int32_t clamped;
	uint64_t row_base = (uint64_t)neuron * hidden_dimension;
	float absmax,scale_value,exact,coded,error_squares = 0.0f;
	if ( neuron >= candidate_count )
		return;
	for (group = lane; group < (hidden_dimension / SPARK_LM_HEAD_SHADOW_GROUP); group += SPARK_LM_WARP_LANES)
	{
		absmax = 0.0f;
		for (element = 0; element < SPARK_LM_HEAD_SHADOW_GROUP; element++)
			absmax = fmaxf(absmax,fabsf(SparkLmBf16ToFloat(head_bf16,row_base + (group * SPARK_LM_HEAD_SHADOW_GROUP) + element)));
		exponent = 127u;
		if ( absmax > 0.0f )
		{
			clamped = (int32_t)ceilf(log2f(absmax / 6.0f));
			clamped = clamped < -127 ? -127 : (clamped > 127 ? 127 : clamped);
			exponent = (uint32_t)(clamped + 127);
		}
		shadow_scale_e8m0[((uint64_t)neuron * (hidden_dimension / SPARK_LM_HEAD_SHADOW_GROUP)) + group] = (uint8_t)exponent;
		scale_value = SparkLmDecodeE8m0(exponent);
		for (element = 0; element < SPARK_LM_HEAD_SHADOW_GROUP; element += 8u)
		{
			packed = 0u;
			for (nibble = 0; nibble < 8u; nibble++)
			{
				exact = SparkLmBf16ToFloat(head_bf16,row_base + (group * SPARK_LM_HEAD_SHADOW_GROUP) + element + nibble);
				packed |= SparkLmEncodeE2m1(scale_value > 0.0f ? exact / scale_value : 0.0f) << (nibble << 2u);
				coded = SparkLmDecodeE2m1((packed >> (nibble << 2u)) & 0x0fu) * scale_value;
				error_squares = fmaf(exact - coded,exact - coded,error_squares);
			}
			((uint32_t *)shadow_payload)[((row_base + (group * SPARK_LM_HEAD_SHADOW_GROUP) + element) >> 3u)] = packed;
		}
	}
	error_squares = SparkLmWarpReduceSum(error_squares);
	if ( lane == 0u )
		error_norm_f32[neuron] = sqrtf(error_squares);
}

// Screen one row: the hidden norm prices the bound, pass one takes the
// best guaranteed lower bound L = max(coarse - s*e), pass two appends
// every candidate whose upper bound reaches L. Overflow leaves the
// count past the cap and the exact fallback owns the row.
static __global__ void SparkLmHeadScreenKernel(const void *hidden_bf16, const void *coarse_logits_bf16, const float *error_norm_f32, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension)
{
	__shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
	__shared__ float shared_norm,shared_bound;
	__shared__ uint32_t shared_cursor;
	uint32_t row = blockIdx.x,element,candidate,slot;
	uint64_t row_base = (uint64_t)row * candidate_count;
	float sum_squares = 0.0f,lower = -3.0e38f,coarse,priced;
	float2 pair_value;
	if ( row >= row_count )
		return;
	for (element = threadIdx.x; element < (hidden_dimension >> 1u); element += blockDim.x)
	{
		pair_value = SparkLmLoadBf16Pair(hidden_bf16,(((uint64_t)row * hidden_dimension) >> 1u) + element);
		sum_squares = fmaf(pair_value.x,pair_value.x,fmaf(pair_value.y,pair_value.y,sum_squares));
	}
	sum_squares = SparkLmBlockReduceSum(sum_squares,reduce_scratch);
	if ( threadIdx.x == 0u )
	{
		shared_norm = sqrtf(sum_squares);
		shared_cursor = 0u;
	}
	__syncthreads();
	for (candidate = threadIdx.x; candidate < candidate_count; candidate += blockDim.x)
	{
		coarse = SparkLmBf16ToFloat(coarse_logits_bf16,row_base + candidate);
		priced = coarse - (shared_norm * __ldg(error_norm_f32 + candidate)) - SparkLmHeadScreenSlack(coarse);
		lower = priced > lower ? priced : lower;
	}
	SparkLmAttnBlockScalarMax(lower,reduce_scratch,&shared_bound);
	for (candidate = threadIdx.x; candidate < candidate_count; candidate += blockDim.x)
	{
		coarse = SparkLmBf16ToFloat(coarse_logits_bf16,row_base + candidate);
		if ( coarse + (shared_norm * __ldg(error_norm_f32 + candidate)) + SparkLmHeadScreenSlack(coarse) < shared_bound )
			continue;
		slot = atomicAdd(&shared_cursor,1u);
		if ( slot < SPARK_LM_HEAD_SCREEN_CAP )
			candidate_ids[((uint64_t)row * SPARK_LM_HEAD_SCREEN_CAP) + slot] = candidate;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
		candidate_counts[row] = shared_cursor;
}

static __device__ void SparkLmHeadStageHidden(const void *hidden_bf16, float *hidden_shared, uint32_t row, uint32_t hidden_dimension)
{
	uint32_t element;
	float2 pair;
	for (element=threadIdx.x; element<(hidden_dimension >> 1u); element+=blockDim.x)
	{
		pair = SparkLmLoadBf16Pair(hidden_bf16,(((uint64_t)row * hidden_dimension) >> 1u) + element);
		hidden_shared[element << 1u] = pair.x;
		hidden_shared[(element << 1u) + 1u] = pair.y;
	}
	__syncthreads();
}

static __device__ void SparkLmHeadExactArgmaxRange(const float *hidden_shared, const void *head_weight_bf16, const uint32_t *candidate_ids, uint32_t first_slot, uint32_t slot_count, uint32_t hidden_dimension, float *best_score, uint32_t *best_candidate)
{
	uint32_t end_slot = first_slot + slot_count,lane = threadIdx.x % SPARK_LM_WARP_LANES,neuron,slot,warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t running_candidate = UINT32_MAX;
	float running_best = -3.0e38f,score;
	for (slot=first_slot + warp; slot<end_slot; slot+=SPARK_LM_CTA_WARPS)
	{
		neuron = candidate_ids != 0 ? candidate_ids[slot] : slot;
		score = SparkLmWarpReduceSum(SparkLmDotRowBf16(hidden_shared,head_weight_bf16,neuron,hidden_dimension,lane));
		score = __shfl_sync(0xffffffffu,score,0);
		if ( lane == 0u && (score > running_best || (score == running_best && neuron < running_candidate)) )
		{
			running_best = score;
			running_candidate = neuron;
		}
	}
	SparkLmArgmaxReduce(running_best,running_candidate,best_score,best_candidate);
}

static __device__ void SparkLmHeadStageHiddenGroup(const void *hidden_bf16, uint32_t *hidden_shared_pairs, uint32_t first_row, uint32_t group_row_count, uint32_t hidden_dimension)
{
	uint32_t group_pair,pair,pairs_per_row,row;
	uint64_t source_pair;
	pairs_per_row = hidden_dimension >> 1u;
	for (group_pair=threadIdx.x; group_pair<group_row_count * pairs_per_row; group_pair+=blockDim.x)
	{
		row = group_pair / pairs_per_row;
		pair = group_pair - (row * pairs_per_row);
		source_pair = ((((uint64_t)first_row + row) * hidden_dimension) >> 1u) + pair;
		hidden_shared_pairs[group_pair] = __ldg(((const uint32_t *)hidden_bf16) + source_pair);
	}
	__syncthreads();
}

static __device__ __forceinline__ float2 SparkLmHeadLoadSharedBf16Pair(const uint32_t *hidden_shared_pairs, uint32_t pair)
{
	uint32_t raw = hidden_shared_pairs[pair];
	return(__bfloat1622float2(*(const __nv_bfloat162 *)&raw));
}

static __device__ __forceinline__ void SparkLmHeadExactArgmaxRowGroup(const uint32_t *hidden_shared_pairs, const void *head_weight_bf16, uint32_t overflow_mask, uint32_t first_slot, uint32_t slot_count, uint32_t group_row_count, uint32_t hidden_dimension, float *running_best, uint32_t *running_candidate)
{
	uint32_t end_slot = first_slot + slot_count,lane = threadIdx.x % SPARK_LM_WARP_LANES,neuron,pair,pair_count,row,slot,warp = threadIdx.x / SPARK_LM_WARP_LANES;
	float accumulator[SPARK_LM_HEAD_FALLBACK_ROW_GROUP_MAX],score;
	float2 hidden_pair,weight_pair;
	pair_count = hidden_dimension >> 1u;
	for (slot=first_slot + warp; slot<end_slot; slot+=SPARK_LM_CTA_WARPS)
	{
		neuron = slot;
		#pragma unroll
		for (row=0u; row<SPARK_LM_HEAD_FALLBACK_ROW_GROUP_MAX; row++) accumulator[row] = 0.0f;
		#pragma unroll 4
		for (pair=lane; pair<pair_count; pair+=SPARK_LM_WARP_LANES)
		{
			weight_pair = SparkLmLoadBf16Pair(head_weight_bf16,(((uint64_t)neuron * hidden_dimension) >> 1u) + pair);
			#pragma unroll
			for (row=0u; row<SPARK_LM_HEAD_FALLBACK_ROW_GROUP_MAX; row++)
			{
				if ( row >= group_row_count || (overflow_mask & (1u << row)) == 0u ) continue;
				hidden_pair = SparkLmHeadLoadSharedBf16Pair(hidden_shared_pairs,(row * pair_count) + pair);
				accumulator[row] = fmaf(hidden_pair.x,weight_pair.x,accumulator[row]);
				accumulator[row] = fmaf(hidden_pair.y,weight_pair.y,accumulator[row]);
			}
		}
		#pragma unroll
		for (row=0u; row<SPARK_LM_HEAD_FALLBACK_ROW_GROUP_MAX; row++)
		{
			score = SparkLmWarpReduceSum(accumulator[row]);
			score = __shfl_sync(0xffffffffu,score,0);
			if ( lane == 0u && row < group_row_count && (overflow_mask & (1u << row)) != 0u && (score > running_best[row] || (score == running_best[row] && neuron < running_candidate[row])) )
			{
				running_best[row] = score;
				running_candidate[row] = neuron;
			}
		}
	}
}

// Overflow rows retain exact bf16 scoring, but stripe the vocabulary over
// enough CTAs to use the GPU instead of assigning the full scan to one SM.
static __global__ void SparkLmHeadFallbackRescoreKernel(const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *candidate_counts, float *partial_scores, uint32_t *partial_candidates, uint32_t row_count, uint32_t hidden_dimension, uint32_t candidate_count)
{
	extern __shared__ float hidden_shared[];
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS];
	uint32_t chunk = blockIdx.y,end_slot,first_slot,row = blockIdx.x;
	uint64_t partial;
	if ( row >= row_count || candidate_counts[row] <= SPARK_LM_HEAD_SCREEN_CAP )
		return;
	first_slot = (uint32_t)(((uint64_t)candidate_count * chunk) / SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT);
	end_slot = (uint32_t)(((uint64_t)candidate_count * (chunk + 1u)) / SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT);
	SparkLmHeadStageHidden(hidden_bf16,hidden_shared,row,hidden_dimension);
	SparkLmHeadExactArgmaxRange(hidden_shared,head_weight_bf16,0,first_slot,end_slot - first_slot,hidden_dimension,best_score,best_candidate);
	if ( threadIdx.x == 0u )
	{
		partial = ((uint64_t)row * SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT) + chunk;
		partial_scores[partial] = best_score[0];
		partial_candidates[partial] = best_candidate[0];
	}
}

// Wider batches group rows so each exact bf16 weight pair feeds several
// independent dot products. Per-row FMA and warp-reduction order is unchanged.
static __global__ void SparkLmHeadGroupedFallbackRescoreKernel(const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *candidate_counts, float *partial_scores, uint32_t *partial_candidates, uint32_t row_count, uint32_t hidden_dimension, uint32_t candidate_count, uint32_t rows_per_group)
{
	extern __shared__ uint32_t hidden_shared_pairs[];
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS],overflow_mask;
	uint32_t chunk = blockIdx.y,end_slot,first_row = blockIdx.x * rows_per_group,first_slot,group_row_count,row;
	float running_best[SPARK_LM_HEAD_FALLBACK_ROW_GROUP_MAX];
	uint32_t running_candidate[SPARK_LM_HEAD_FALLBACK_ROW_GROUP_MAX];
	if ( first_row >= row_count ) return;
	group_row_count = row_count - first_row < rows_per_group ? row_count - first_row : rows_per_group;
	if ( threadIdx.x == 0u )
	{
		overflow_mask = 0u;
		for (row=0u; row<group_row_count; row++)
			if ( candidate_counts[first_row + row] > SPARK_LM_HEAD_SCREEN_CAP ) overflow_mask |= 1u << row;
	}
	__syncthreads();
	if ( overflow_mask == 0u ) return;
	SparkLmHeadStageHiddenGroup(hidden_bf16,hidden_shared_pairs,first_row,group_row_count,hidden_dimension);
	first_slot = (uint32_t)(((uint64_t)candidate_count * chunk) / SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT);
	end_slot = (uint32_t)(((uint64_t)candidate_count * (chunk + 1u)) / SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT);
	#pragma unroll
	for (row=0u; row<SPARK_LM_HEAD_FALLBACK_ROW_GROUP_MAX; row++)
	{
		running_best[row] = -3.0e38f;
		running_candidate[row] = UINT32_MAX;
	}
	SparkLmHeadExactArgmaxRowGroup(hidden_shared_pairs,head_weight_bf16,overflow_mask,first_slot,end_slot - first_slot,group_row_count,hidden_dimension,running_best,running_candidate);
	#pragma unroll
	for (row=0u; row<SPARK_LM_HEAD_FALLBACK_ROW_GROUP_MAX; row++)
	{
		SparkLmArgmaxReduce(running_best[row],running_candidate[row],best_score,best_candidate);
		if ( threadIdx.x == 0u && row < group_row_count && (overflow_mask & (1u << row)) != 0u )
		{
			uint64_t partial = ((uint64_t)(first_row + row) * SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT) + chunk;
			partial_scores[partial] = best_score[0];
			partial_candidates[partial] = best_candidate[0];
		}
	}
}

// Screened rows rescore their compact list. Overflow rows reduce the exact
// chunk winners with the same score and lower-token tie rule.
static __global__ void SparkLmHeadRescoreArgmaxKernel(const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *candidate_ids, const uint32_t *candidate_counts, const float *partial_scores, const uint32_t *partial_candidates, uint32_t *output_token_ids, float *output_scores, uint32_t candidate_offset, uint32_t row_count, uint32_t hidden_dimension)
{
	extern __shared__ float hidden_shared[];
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS];
	uint32_t count,partial,row = blockIdx.x,running_candidate = UINT32_MAX;
	float running_best = -3.0e38f,score;
	if ( row >= row_count )
		return;
	count = candidate_counts[row];
	if ( count <= SPARK_LM_HEAD_SCREEN_CAP )
	{
		SparkLmHeadStageHidden(hidden_bf16,hidden_shared,row,hidden_dimension);
		SparkLmHeadExactArgmaxRange(hidden_shared,head_weight_bf16,candidate_ids + ((uint64_t)row * SPARK_LM_HEAD_SCREEN_CAP),0u,count,hidden_dimension,best_score,best_candidate);
	}
	else
	{
		for (partial=threadIdx.x; partial<SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT; partial+=blockDim.x)
		{
			score = partial_scores[((uint64_t)row * SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT) + partial];
			if ( score > running_best || (score == running_best && partial_candidates[((uint64_t)row * SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT) + partial] < running_candidate) )
			{
				running_best = score;
				running_candidate = partial_candidates[((uint64_t)row * SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT) + partial];
			}
		}
		SparkLmArgmaxReduce(running_best,running_candidate,best_score,best_candidate);
	}
	if ( threadIdx.x == 0u )
	{
		output_token_ids[row] = best_candidate[0] + candidate_offset;
		if ( output_scores != 0 )
			output_scores[row] = best_score[0];
	}
}

/*
 * B1 certified FP8 head screen. The FP8 copy is screening data only: every
 * retained candidate is rescored against the untouched BF16 target head and
 * the emitted score is that exact BF16 dot product. One norm per 32-element
 * group certifies
 *
 *   |dot_bf16 - dot_fp8| <= sum_g ||h_g||_2 ||w_g - q_g||_2
 *
 * and also includes a Higham gamma bound for both FP32 accumulation paths.
 * All stored norms and screen endpoints round outward. The full rank-local
 * vocabulary is a valid candidate capacity at B1, so this path never needs an
 * overflow compatibility scan.
 */
#define SPARK_LM_HEAD_CERTIFIED_FP8_THREADS 1024u
#define SPARK_LM_HEAD_CERTIFIED_FP8_WARPS 32u

static __device__ __forceinline__ double SparkLmHeadWarpReduceDouble(double value)
{
	uint32_t offset;
	for (offset=SPARK_LM_WARP_LANES / 2u; offset!=0u; offset>>=1u)
		value += __shfl_down_sync(0xffffffffu,value,offset);
	return(value);
}

static __device__ __forceinline__ float SparkLmHeadWarpReduceMax(float value)
{
	uint32_t offset;
	float other;
	for (offset=SPARK_LM_WARP_LANES / 2u; offset!=0u; offset>>=1u)
	{
		other = __shfl_down_sync(0xffffffffu,value,offset);
		value = other > value ? other : value;
	}
	return(value);
}

static __device__ __forceinline__ float SparkLmHeadRoundDoubleUp(double value)
{
	float rounded = (float)value;
	if ( (double)rounded < value )
		rounded = nextafterf(rounded,INFINITY);
	return(rounded);
}

static __device__ __forceinline__ double SparkLmHeadFloatGamma(uint32_t operations)
{
	double unit = 0x1p-24;
	return(((double)operations * unit) / (1.0 - ((double)operations * unit)));
}

static __device__ __forceinline__ uint32_t SparkLmHeadUint4Word(uint4 value,uint32_t index)
{
	if ( index == 0u ) return(value.x);
	if ( index == 1u ) return(value.y);
	if ( index == 2u ) return(value.z);
	return(value.w);
}

static __global__ void SparkLmHeadCertifiedFp8QuantizeKernel(const void *head_bf16,uint8_t *shadow_payload,float *shadow_scale_f32,float *cert_norm_f32,uint32_t candidate_count,uint32_t hidden_dimension)
{
	uint32_t neuron = (blockIdx.x * SPARK_LM_CTA_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES),lane = threadIdx.x % SPARK_LM_WARP_LANES,group,groups = hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE;
	uint64_t group_base,row_base = (uint64_t)neuron * hidden_dimension;
	double delta,error_square,error_norm,shadow_norm,shadow_square,weight_norm,weight_square,cert;
	double exact_gamma = SparkLmHeadFloatGamma((hidden_dimension / SPARK_LM_WARP_LANES) + 5u),shadow_gamma = SparkLmHeadFloatGamma((2u * hidden_dimension / SPARK_LM_WARP_LANES) + 5u);
	float absmax,exact,quantized,scale;
	if ( neuron >= candidate_count ) return;
	for (group=0u; group<groups; group++)
	{
		group_base = row_base + ((uint64_t)group * SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE);
		exact = SparkLmBf16ToFloat(head_bf16,group_base + lane);
		absmax = SparkLmHeadWarpReduceMax(fabsf(exact));
		absmax = __shfl_sync(0xffffffffu,absmax,0u);
		scale = absmax > 0.0f ? absmax / LM_E4M3_MAX : 1.0f;
		shadow_payload[group_base + lane] = LmFloatToE4m3(exact / scale);
		quantized = LmE4m3ToFloat(shadow_payload[group_base + lane]) * scale;
		delta = (double)exact - (double)quantized;
		error_square = SparkLmHeadWarpReduceDouble(delta * delta);
		weight_square = SparkLmHeadWarpReduceDouble((double)exact * (double)exact);
		shadow_square = SparkLmHeadWarpReduceDouble((double)quantized * (double)quantized);
		if ( lane == 0u )
		{
			error_norm = sqrt(error_square * (1.0 + 1.0e-12));
			weight_norm = sqrt(weight_square * (1.0 + 1.0e-12));
			shadow_norm = sqrt(shadow_square * (1.0 + 1.0e-12));
			cert = (error_norm + (exact_gamma * weight_norm) + (shadow_gamma * shadow_norm)) * (1.0 + 1.0e-12);
			shadow_scale_f32[((uint64_t)neuron * groups) + group] = scale;
			cert_norm_f32[((uint64_t)neuron * groups) + group] = SparkLmHeadRoundDoubleUp(cert);
		}
	}
}

static __global__ void SparkLmHeadCertifiedHiddenNormKernel(const void *hidden_bf16,float *hidden_norm_f32,uint32_t hidden_dimension)
{
	uint32_t group = (blockIdx.x * SPARK_LM_CTA_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES),lane = threadIdx.x % SPARK_LM_WARP_LANES,groups = hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE;
	double value,sum;
	if ( group >= groups ) return;
	value = (double)SparkLmBf16ToFloat(hidden_bf16,((uint64_t)group * SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE) + lane);
	sum = SparkLmHeadWarpReduceDouble(value * value);
	if ( lane == 0u )
		hidden_norm_f32[group] = SparkLmHeadRoundDoubleUp(sqrt(sum * (1.0 + 1.0e-12)));
}

static __global__ void SparkLmHeadCertifiedFp8ScoreKernel(const void *hidden_bf16,const uint8_t *shadow_payload,const float *shadow_scale_f32,const float *cert_norm_f32,const float *hidden_norm_f32,float *coarse_scores,float *bounds,uint32_t candidate_count,uint32_t hidden_dimension)
{
	uint32_t neuron = (blockIdx.x * SPARK_LM_HEAD_CERTIFIED_FP8_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES),lane = threadIdx.x % SPARK_LM_WARP_LANES,group,pair,tile,tiles = hidden_dimension / 512u,word;
	uint64_t norm_base = (uint64_t)neuron * (hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE),shadow_base = ((uint64_t)neuron * hidden_dimension) / sizeof(uint4);
	float accumulator = 0.0f,bound = 0.0f,scale;
	float2 hidden_pair,shadow_pair;
	uint4 hidden_first,hidden_second,shadow_vector;
	if ( neuron >= candidate_count ) return;
	for (tile=0u; tile<tiles; tile++)
	{
		group = (tile * 16u) + (lane >> 1u);
		scale = (lane & 1u) == 0u ? __ldg(shadow_scale_f32 + norm_base + group) : 0.0f;
		scale = __shfl_sync(0xffffffffu,scale,lane & ~1u);
		shadow_vector = __ldg(((const uint4 *)shadow_payload) + shadow_base + ((uint64_t)tile * SPARK_LM_WARP_LANES) + lane);
		hidden_first = __ldg(((const uint4 *)hidden_bf16) + ((uint64_t)tile * 64u) + ((uint64_t)lane * 2u));
		hidden_second = __ldg(((const uint4 *)hidden_bf16) + ((uint64_t)tile * 64u) + ((uint64_t)lane * 2u) + 1u);
		#pragma unroll
		for (pair=0u; pair<8u; pair++)
		{
			word = SparkLmHeadUint4Word(shadow_vector,pair >> 1u);
			shadow_pair = LmE4m3PairToFloat2((uint16_t)(word >> ((pair & 1u) << 4u)));
			word = pair < 4u ? SparkLmHeadUint4Word(hidden_first,pair) : SparkLmHeadUint4Word(hidden_second,pair - 4u);
			hidden_pair = __bfloat1622float2(*(const __nv_bfloat162 *)&word);
			accumulator = fmaf(hidden_pair.x,shadow_pair.x * scale,accumulator);
			accumulator = fmaf(hidden_pair.y,shadow_pair.y * scale,accumulator);
		}
		if ( (lane & 1u) == 0u )
			bound = fmaf(__ldg(hidden_norm_f32 + group),__ldg(cert_norm_f32 + norm_base + group),bound);
	}
	accumulator = SparkLmWarpReduceSum(accumulator);
	bound = SparkLmWarpReduceSum(bound);
	if ( lane == 0u )
	{
		coarse_scores[neuron] = accumulator;
		bound *= 1.0f + (float)SparkLmHeadFloatGamma(tiles + 5u);
		bounds[neuron] = nextafterf(bound,INFINITY);
	}
}

static __global__ void SparkLmHeadCertifiedScreenKernel(const float *coarse_scores,const float *bounds,uint32_t *candidate_ids,uint32_t *candidate_count,uint32_t vocabulary_count)
{
	__shared__ float scratch[SPARK_LM_CTA_WARPS];
	__shared__ float best_lower;
	__shared__ uint32_t cursor;
	uint32_t candidate,lane = threadIdx.x % SPARK_LM_WARP_LANES,warp = threadIdx.x / SPARK_LM_WARP_LANES;
	float lower = -3.0e38f,other,upper;
	for (candidate=threadIdx.x; candidate<vocabulary_count; candidate+=blockDim.x)
	{
		other = nextafterf(coarse_scores[candidate] - bounds[candidate],-INFINITY);
		lower = other > lower ? other : lower;
	}
	lower = SparkLmHeadWarpReduceMax(lower);
	if ( lane == 0u ) scratch[warp] = lower;
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		best_lower = scratch[0];
		for (candidate=1u; candidate<SPARK_LM_CTA_WARPS; candidate++) best_lower = scratch[candidate] > best_lower ? scratch[candidate] : best_lower;
		cursor = 0u;
	}
	__syncthreads();
	for (candidate=threadIdx.x; candidate<vocabulary_count; candidate+=blockDim.x)
	{
		upper = nextafterf(coarse_scores[candidate] + bounds[candidate],INFINITY);
		if ( upper >= best_lower ) candidate_ids[atomicAdd(&cursor,1u)] = candidate;
	}
	__syncthreads();
	if ( threadIdx.x == 0u ) *candidate_count = cursor;
}

static __global__ void SparkLmHeadCertifiedRescoreKernel(const void *hidden_bf16,const void *head_weight_bf16,const uint32_t *candidate_ids,const uint32_t *candidate_count,float *partial_scores,uint32_t *partial_candidates,uint32_t hidden_dimension)
{
	extern __shared__ float hidden_shared[];
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS];
	uint32_t count = *candidate_count,lane = threadIdx.x % SPARK_LM_WARP_LANES,slot = (blockIdx.x * SPARK_LM_CTA_WARPS) + (threadIdx.x / SPARK_LM_WARP_LANES),candidate,running_candidate = UINT32_MAX;
	float running_best = -3.0e38f,score;
	SparkLmHeadStageHidden(hidden_bf16,hidden_shared,0u,hidden_dimension);
	for (; slot<count; slot+=SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT * SPARK_LM_CTA_WARPS)
	{
		candidate = candidate_ids[slot];
		score = SparkLmWarpReduceSum(SparkLmDotRowBf16(hidden_shared,head_weight_bf16,candidate,hidden_dimension,lane));
		score = __shfl_sync(0xffffffffu,score,0u);
		if ( lane == 0u && (score > running_best || (score == running_best && candidate < running_candidate)) )
		{
			running_best = score;
			running_candidate = candidate;
		}
	}
	SparkLmArgmaxReduce(running_best,running_candidate,best_score,best_candidate);
	if ( threadIdx.x == 0u )
	{
		partial_scores[blockIdx.x] = best_score[0];
		partial_candidates[blockIdx.x] = best_candidate[0];
	}
}

static __global__ void SparkLmHeadCertifiedReduceKernel(const float *partial_scores,const uint32_t *partial_candidates,uint32_t *output_token_id,float *output_score,uint32_t candidate_offset)
{
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS];
	uint32_t partial,running_candidate = UINT32_MAX;
	float running_best = -3.0e38f,score;
	for (partial=threadIdx.x; partial<SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT; partial+=blockDim.x)
	{
		score = partial_scores[partial];
		if ( score > running_best || (score == running_best && partial_candidates[partial] < running_candidate) )
		{
			running_best = score;
			running_candidate = partial_candidates[partial];
		}
	}
	SparkLmArgmaxReduce(running_best,running_candidate,best_score,best_candidate);
	if ( threadIdx.x == 0u )
	{
		*output_token_id = best_candidate[0] + candidate_offset;
		*output_score = best_score[0];
	}
}



/*
 * Single-pass decode attention for one (row, head): eight warps stripe
 * the key range, each warp computing one key's logit cooperatively -
 * lanes pair-load the key so every K fetch is a full-width transaction -
 * and folding it into a per-warp online softmax: running max, running
 * denominator, and the value accumulation held in registers, four lanes
 * of float2 covering up to 256 value channels. One QK pass total, no
 * atomics anywhere; the eight partials merge once through shared at the
 * end, where the optional attention sink joins the denominator. Ring
 * addressing when window_slots is nonzero, dense otherwise; kv head =
 * head / group_size on split K and V caches.
 */
#define SPARK_LM_ATTN_MAX_VALUE_PAIRS_PER_LANE 4u

static __device__ __forceinline__ float SparkLmAttnKeyLogit(const float *q_shared, const void *k_cache_bf16, uint64_t key_base, uint32_t head_pairs, uint32_t lane, float scale)
{
    uint32_t pair;
    float accumulator = 0.0f;
    float2 pair_value;

    for (pair = lane; pair < head_pairs; pair += SPARK_LM_WARP_LANES)
    {
        pair_value = SparkLmLoadBf16Pair(k_cache_bf16, (key_base >> 1u) + pair);
        accumulator = fmaf(q_shared[pair << 1u], pair_value.x, accumulator);
        accumulator = fmaf(q_shared[(pair << 1u) + 1u], pair_value.y, accumulator);
    }
    accumulator = SparkLmWarpReduceSum(accumulator);
    return __shfl_sync(0xffffffffu, accumulator, 0) * scale;
}

// Cross-warp merge and store: block max over the warp partials, the
// denominator with the optional sink folded in, then the per-element
// weighted recombination of the staged accumulators.
static __device__ void SparkLmAttnMergeStore(float *merge_max, float *merge_den, const float *merge_acc, const float *sink_f32, uint32_t head, uint32_t value_dim, void *out_bf16, uint64_t out_base)
{
    uint32_t element;
    uint32_t partial;
    float merged;

    if (threadIdx.x == 0u)
    {
        float block_max = merge_max[0];
        float block_den;

        for (partial = 1u; partial < SPARK_LM_CTA_WARPS; ++partial)
        {
            block_max = merge_max[partial] > block_max ? merge_max[partial] : block_max;
        }
        block_den = sink_f32 != 0 ? __expf(sink_f32[head] - block_max) : 0.0f;
        for (partial = 0u; partial < SPARK_LM_CTA_WARPS; ++partial)
        {
            float scale = __expf(merge_max[partial] - block_max);
            merge_den[partial] *= scale;
            merge_max[partial] = scale;
            block_den += merge_den[partial];
        }
        merge_den[0] = 1.0f / block_den;
    }
    __syncthreads();

    for (element = threadIdx.x; element < value_dim; element += blockDim.x)
    {
        merged = 0.0f;
        for (partial = 0u; partial < SPARK_LM_CTA_WARPS; ++partial)
        {
            merged = fmaf(
                merge_acc[(partial * value_dim) + element],
                merge_max[partial],
                merged);
        }
        SparkLmFloatToBf16(out_bf16, out_base + element, merged * merge_den[0]);
    }
}

// Stage the head's query into shared, zero the merge scratch and the
// register accumulators.
static __device__ void SparkLmAttnStage(const void *q_bf16, uint64_t q_row_stride, uint32_t row, uint32_t head, uint32_t head_dim, uint32_t value_dim, float *q_shared, float *merge_acc, float2 *accumulator)
{
	uint32_t element,pair;
	for (element = threadIdx.x; element < head_dim; element += blockDim.x)
		q_shared[element] = SparkLmBf16ToFloat(q_bf16,((uint64_t)row * q_row_stride) + ((uint64_t)head * head_dim) + element);
	for (element = threadIdx.x; element < SPARK_LM_CTA_WARPS * value_dim; element += blockDim.x)
		merge_acc[element] = 0.0f;
	for (pair = 0; pair < SPARK_LM_ATTN_MAX_VALUE_PAIRS_PER_LANE; pair++)
		accumulator[pair] = make_float2(0.0f,0.0f);
}

static __global__ void SparkLmAttnDecodeKernel(const void *q_bf16, uint64_t q_row_stride, const void *k_cache_bf16, const void *v_cache_bf16, uint64_t k_lane_stride, uint64_t v_lane_stride, uint64_t k_slot_stride, uint64_t v_slot_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, const float *sink_f32, float scale, void *out_bf16, uint32_t row_count, uint32_t head_count, uint32_t group_size, uint32_t head_dim, uint32_t value_dim, uint32_t window_slots)
{
	extern __shared__ float attn_shared[];
	float *q_shared = attn_shared,*merge_acc = attn_shared + head_dim;
	__shared__ float merge_max[SPARK_LM_CTA_WARPS],merge_den[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x,head = blockIdx.y,kv_head = head / group_size;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t head_pairs = head_dim >> 1u,value_pairs = value_dim >> 1u,pairs_per_lane = (value_pairs + SPARK_LM_WARP_LANES - 1u) / SPARK_LM_WARP_LANES;
	uint64_t position,first_key,key,k_base,v_base,slot;
	float running_max = -3.0e38f,running_den = 0.0f,logit,rescale,weight;
	float2 accumulator[SPARK_LM_ATTN_MAX_VALUE_PAIRS_PER_LANE],pair_value;
	uint32_t pair,element;
	if ( row >= row_count || head >= head_count )
		return;
	position = row_positions[row];
	first_key = window_slots != 0u && position + 1u > window_slots ? position + 1u - window_slots : 0u;
	k_base = ((uint64_t)row_lane_indices[row] * k_lane_stride) + ((uint64_t)kv_head * head_dim);
	v_base = ((uint64_t)row_lane_indices[row] * v_lane_stride) + ((uint64_t)kv_head * value_dim);
	SparkLmAttnStage(q_bf16,q_row_stride,row,head,head_dim,value_dim,q_shared,merge_acc,accumulator);
	__syncthreads();
	for (key = first_key + warp; key <= position; key += SPARK_LM_CTA_WARPS)
	{
		slot = window_slots != 0u ? key % window_slots : key;
		logit = SparkLmAttnKeyLogit(q_shared,k_cache_bf16,k_base + (slot * k_slot_stride),head_pairs,lane,scale);
		rescale = logit > running_max ? __expf(running_max - logit) : 1.0f;
		weight = logit > running_max ? 1.0f : __expf(logit - running_max);
		running_max = logit > running_max ? logit : running_max;
		running_den = fmaf(running_den,rescale,weight);
		for (pair = 0; pair < pairs_per_lane; pair++)
			if ( (pair * SPARK_LM_WARP_LANES) + lane < value_pairs )
			{
				pair_value = SparkLmLoadBf16Pair(v_cache_bf16,((v_base + (slot * v_slot_stride)) >> 1u) + (pair * SPARK_LM_WARP_LANES) + lane);
				accumulator[pair].x = fmaf(accumulator[pair].x,rescale,weight * pair_value.x);
				accumulator[pair].y = fmaf(accumulator[pair].y,rescale,weight * pair_value.y);
			}
	}
	if ( lane == 0u )
	{
		merge_max[warp] = running_max;
		merge_den[warp] = running_den;
	}
	for (pair = 0; pair < pairs_per_lane; pair++)
		if ( (pair * SPARK_LM_WARP_LANES) + lane < value_pairs )
		{
			element = ((pair * SPARK_LM_WARP_LANES) + lane) << 1u;
			merge_acc[(warp * value_dim) + element] = accumulator[pair].x;
			merge_acc[(warp * value_dim) + element + 1u] = accumulator[pair].y;
		}
	__syncthreads();
	SparkLmAttnMergeStore(merge_max,merge_den,merge_acc,sink_f32,head,value_dim,out_bf16,(((uint64_t)row * head_count) + head) * value_dim);
}

#define SPARK_LM_GROUPED_ATTN_MAX_HEADS_PER_CTA 4u

template <uint32_t HEADS_PER_CTA>
static __global__ void SparkLmGroupedAttnDecodeKernel(
    const void *q_bf16,
    uint64_t q_row_stride,
    const void *k_cache_bf16,
    const void *v_cache_bf16,
    uint64_t k_lane_stride,
    uint64_t v_lane_stride,
    uint64_t k_slot_stride,
    uint64_t v_slot_stride,
    const uint32_t *row_lane_indices,
    const uint64_t *row_positions,
    const float *sink_f32,
    float scale,
    void *out_bf16,
    uint32_t row_count,
    uint32_t head_count,
    uint32_t group_size,
    uint32_t head_dim,
    uint32_t value_dim,
    uint32_t window_slots)
{
    extern __shared__ float grouped_attn_shared[];
    __shared__ float merge_max[
        HEADS_PER_CTA * SPARK_LM_CTA_WARPS];
    __shared__ float merge_den[
        HEADS_PER_CTA * SPARK_LM_CTA_WARPS];
    __shared__ float inverse_denominator[HEADS_PER_CTA];
    float *query_shared;
    float *merge_accumulator;
    float running_max[HEADS_PER_CTA];
    float running_denominator[HEADS_PER_CTA];
    float2 accumulator[
        HEADS_PER_CTA][SPARK_LM_ATTN_MAX_VALUE_PAIRS_PER_LANE];
    uint32_t row;
    uint32_t kv_head_count;
    uint32_t chunks_per_kv_head;
    uint32_t kv_head;
    uint32_t head_chunk;
    uint32_t first_head;
    uint32_t active_head_count;
    uint32_t warp_index;
    uint32_t lane_index;
    uint32_t head_pairs;
    uint32_t value_pairs;
    uint32_t pairs_per_lane;
    uint32_t local_head;
    uint32_t pair_index;
    uint32_t element_index;
    uint64_t position;
    uint64_t first_key;
    uint64_t key_index;
    uint64_t key_base;
    uint64_t value_base;
    uint64_t slot_index;

    static_assert(
        HEADS_PER_CTA > 0u &&
        HEADS_PER_CTA <= SPARK_LM_GROUPED_ATTN_MAX_HEADS_PER_CTA,
        "grouped attention head count must fit the retained register layout");
    row = blockIdx.x;
    kv_head_count = head_count / group_size;
    chunks_per_kv_head =
        (group_size + HEADS_PER_CTA - 1u) / HEADS_PER_CTA;
    kv_head = blockIdx.y / chunks_per_kv_head;
    head_chunk = blockIdx.y - (kv_head * chunks_per_kv_head);
    first_head = (kv_head * group_size) + (head_chunk * HEADS_PER_CTA);
    if (row >= row_count || kv_head >= kv_head_count || first_head >= head_count)
    {
        return;
    }
    active_head_count = head_count - first_head;
    if (active_head_count > HEADS_PER_CTA)
    {
        active_head_count = HEADS_PER_CTA;
    }
    if (active_head_count > group_size - (head_chunk * HEADS_PER_CTA))
    {
        active_head_count = group_size - (head_chunk * HEADS_PER_CTA);
    }

    query_shared = grouped_attn_shared;
    merge_accumulator = query_shared + (HEADS_PER_CTA * head_dim);
    warp_index = threadIdx.x / SPARK_LM_WARP_LANES;
    lane_index = threadIdx.x % SPARK_LM_WARP_LANES;
    head_pairs = head_dim >> 1u;
    value_pairs = value_dim >> 1u;
    pairs_per_lane =
        (value_pairs + SPARK_LM_WARP_LANES - 1u) /
        SPARK_LM_WARP_LANES;

    element_index = threadIdx.x;
    while (element_index < active_head_count * head_dim)
    {
        local_head = element_index / head_dim;
        query_shared[element_index] = SparkLmBf16ToFloat(
            q_bf16,
            ((uint64_t)row * q_row_stride) +
                ((uint64_t)(first_head + local_head) * head_dim) +
                (element_index - (local_head * head_dim)));
        element_index += blockDim.x;
    }
    element_index = threadIdx.x;
    while (element_index <
        HEADS_PER_CTA * SPARK_LM_CTA_WARPS * value_dim)
    {
        merge_accumulator[element_index] = 0.0f;
        element_index += blockDim.x;
    }
    for (local_head = 0u; local_head < HEADS_PER_CTA; ++local_head)
    {
        running_max[local_head] = -3.0e38f;
        running_denominator[local_head] = 0.0f;
        for (pair_index = 0u;
             pair_index < SPARK_LM_ATTN_MAX_VALUE_PAIRS_PER_LANE;
             ++pair_index)
        {
            accumulator[local_head][pair_index] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    position = row_positions[row];
    first_key = window_slots != 0u && position + 1u > window_slots
        ? position + 1u - window_slots
        : 0u;
    key_base =
        ((uint64_t)row_lane_indices[row] * k_lane_stride) +
        ((uint64_t)kv_head * head_dim);
    value_base =
        ((uint64_t)row_lane_indices[row] * v_lane_stride) +
        ((uint64_t)kv_head * value_dim);
    for (key_index = first_key + warp_index;
         key_index <= position;
         key_index += SPARK_LM_CTA_WARPS)
    {
        float local_logit[HEADS_PER_CTA];
        float logit[HEADS_PER_CTA];
        float rescale[HEADS_PER_CTA];
        float weight[HEADS_PER_CTA];
        float2 key_pair;

        slot_index = window_slots != 0u
            ? key_index % window_slots
            : key_index;
        for (local_head = 0u; local_head < HEADS_PER_CTA; ++local_head)
        {
            local_logit[local_head] = 0.0f;
        }
        for (pair_index = lane_index;
             pair_index < head_pairs;
             pair_index += SPARK_LM_WARP_LANES)
        {
            key_pair = SparkLmLoadBf16Pair(
                k_cache_bf16,
                ((key_base + (slot_index * k_slot_stride)) >> 1u) +
                    pair_index);
            for (local_head = 0u;
                 local_head < active_head_count;
                 ++local_head)
            {
                uint32_t query_element;

                query_element = pair_index << 1u;
                local_logit[local_head] = fmaf(
                    query_shared[(local_head * head_dim) + query_element],
                    key_pair.x,
                    local_logit[local_head]);
                local_logit[local_head] = fmaf(
                    query_shared[
                        (local_head * head_dim) + query_element + 1u],
                    key_pair.y,
                    local_logit[local_head]);
            }
        }
        for (local_head = 0u;
             local_head < active_head_count;
             ++local_head)
        {
            logit[local_head] = __shfl_sync(
                0xffffffffu,
                SparkLmWarpReduceSum(local_logit[local_head]),
                0) * scale;
            rescale[local_head] = 0.0f;
            weight[local_head] = 0.0f;
            if (lane_index == 0u)
            {
                rescale[local_head] =
                    logit[local_head] > running_max[local_head]
                    ? __expf(running_max[local_head] - logit[local_head])
                    : 1.0f;
                weight[local_head] =
                    logit[local_head] > running_max[local_head]
                    ? 1.0f
                    : __expf(logit[local_head] - running_max[local_head]);
                running_max[local_head] =
                    logit[local_head] > running_max[local_head]
                    ? logit[local_head]
                    : running_max[local_head];
                running_denominator[local_head] = fmaf(
                    running_denominator[local_head],
                    rescale[local_head],
                    weight[local_head]);
            }
            rescale[local_head] = __shfl_sync(
                0xffffffffu,rescale[local_head],0);
            weight[local_head] = __shfl_sync(
                0xffffffffu,weight[local_head],0);
        }
        for (pair_index = 0u;
             pair_index < pairs_per_lane;
             ++pair_index)
        {
            uint32_t value_pair_index;

            value_pair_index =
                (pair_index * SPARK_LM_WARP_LANES) + lane_index;
            if (value_pair_index < value_pairs)
            {
                float2 value_pair;

                value_pair = SparkLmLoadBf16Pair(
                    v_cache_bf16,
                    ((value_base + (slot_index * v_slot_stride)) >> 1u) +
                        value_pair_index);
                for (local_head = 0u;
                     local_head < active_head_count;
                     ++local_head)
                {
                    accumulator[local_head][pair_index].x = fmaf(
                        accumulator[local_head][pair_index].x,
                        rescale[local_head],
                        weight[local_head] * value_pair.x);
                    accumulator[local_head][pair_index].y = fmaf(
                        accumulator[local_head][pair_index].y,
                        rescale[local_head],
                        weight[local_head] * value_pair.y);
                }
            }
        }
    }

    if (lane_index == 0u)
    {
        for (local_head = 0u;
             local_head < active_head_count;
             ++local_head)
        {
            merge_max[(local_head * SPARK_LM_CTA_WARPS) + warp_index] =
                running_max[local_head];
            merge_den[(local_head * SPARK_LM_CTA_WARPS) + warp_index] =
                running_denominator[local_head];
        }
    }
    for (local_head = 0u;
         local_head < active_head_count;
         ++local_head)
    {
        for (pair_index = 0u;
             pair_index < pairs_per_lane;
             ++pair_index)
        {
            uint32_t value_pair_index;

            value_pair_index =
                (pair_index * SPARK_LM_WARP_LANES) + lane_index;
            if (value_pair_index < value_pairs)
            {
                uint32_t output_element;
                uint64_t merge_base;

                output_element = value_pair_index << 1u;
                merge_base =
                    (((uint64_t)local_head * SPARK_LM_CTA_WARPS) +
                     warp_index) *
                    value_dim;
                merge_accumulator[merge_base + output_element] =
                    accumulator[local_head][pair_index].x;
                merge_accumulator[merge_base + output_element + 1u] =
                    accumulator[local_head][pair_index].y;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x < active_head_count)
    {
        float block_max;
        float block_denominator;
        uint32_t partial_index;
        uint32_t actual_head;

        local_head = threadIdx.x;
        actual_head = first_head + local_head;
        block_max = merge_max[local_head * SPARK_LM_CTA_WARPS];
        for (partial_index = 1u;
             partial_index < SPARK_LM_CTA_WARPS;
             ++partial_index)
        {
            float partial_max;

            partial_max = merge_max[
                (local_head * SPARK_LM_CTA_WARPS) + partial_index];
            block_max = partial_max > block_max ? partial_max : block_max;
        }
        block_denominator = sink_f32 != 0
            ? __expf(sink_f32[actual_head] - block_max)
            : 0.0f;
        for (partial_index = 0u;
             partial_index < SPARK_LM_CTA_WARPS;
             ++partial_index)
        {
            uint32_t partial_offset;
            float partial_scale;

            partial_offset =
                (local_head * SPARK_LM_CTA_WARPS) + partial_index;
            partial_scale = __expf(merge_max[partial_offset] - block_max);
            merge_max[partial_offset] = partial_scale;
            block_denominator += merge_den[partial_offset] * partial_scale;
        }
        inverse_denominator[local_head] = 1.0f / block_denominator;
    }
    __syncthreads();

    element_index = threadIdx.x;
    while (element_index < active_head_count * value_dim)
    {
        float merged_value;
        uint32_t output_element;
        uint32_t partial_index;
        uint32_t actual_head;

        local_head = element_index / value_dim;
        output_element = element_index - (local_head * value_dim);
        actual_head = first_head + local_head;
        merged_value = 0.0f;
        for (partial_index = 0u;
             partial_index < SPARK_LM_CTA_WARPS;
             ++partial_index)
        {
            uint64_t merge_base;
            uint32_t partial_offset;

            merge_base =
                (((uint64_t)local_head * SPARK_LM_CTA_WARPS) +
                 partial_index) *
                value_dim;
            partial_offset =
                (local_head * SPARK_LM_CTA_WARPS) + partial_index;
            merged_value = fmaf(
                merge_accumulator[merge_base + output_element],
                merge_max[partial_offset],
                merged_value);
        }
        SparkLmFloatToBf16(
            out_bf16,
            (((uint64_t)row * head_count) + actual_head) * value_dim +
                output_element,
            merged_value * inverse_denominator[local_head]);
        element_index += blockDim.x;
    }
}

template <uint32_t HEADS_PER_CTA>
static cudaError_t SparkLmHostLaunchGroupedAttnDecode(
    cudaStream_t stream,
    const void *q_bf16,
    uint64_t q_row_stride,
    const void *k_cache_bf16,
    const void *v_cache_bf16,
    uint64_t k_lane_stride,
    uint64_t v_lane_stride,
    uint64_t k_slot_stride,
    uint64_t v_slot_stride,
    const uint32_t *row_lane_indices,
    const uint64_t *row_positions,
    const float *sink_f32,
    float scale,
    void *out_bf16,
    uint32_t row_count,
    uint32_t head_count,
    uint32_t group_size,
    uint32_t head_dim,
    uint32_t value_dim,
    uint32_t window_slots)
{
    dim3 grid;
    uint32_t kv_head_count;
    uint32_t chunks_per_kv_head;
    uint32_t shared_float_count;

    if (stream == 0 || q_bf16 == 0 || k_cache_bf16 == 0 ||
        v_cache_bf16 == 0 || row_lane_indices == 0 ||
        row_positions == 0 || out_bf16 == 0 || row_count == 0u ||
        head_count == 0u || group_size == 0u ||
        head_count % group_size != 0u || head_dim == 0u ||
        value_dim == 0u || (head_dim & 1u) != 0u ||
        (value_dim & 1u) != 0u ||
        ((value_dim >> 1u) + SPARK_LM_WARP_LANES - 1u) /
                SPARK_LM_WARP_LANES >
            SPARK_LM_ATTN_MAX_VALUE_PAIRS_PER_LANE)
    {
        return cudaErrorInvalidValue;
    }
    kv_head_count = head_count / group_size;
    chunks_per_kv_head =
        (group_size + HEADS_PER_CTA - 1u) / HEADS_PER_CTA;
    grid = dim3(row_count, kv_head_count * chunks_per_kv_head);
    shared_float_count =
        (HEADS_PER_CTA * head_dim) +
        (HEADS_PER_CTA * SPARK_LM_CTA_WARPS * value_dim);
    SparkLmGroupedAttnDecodeKernel<HEADS_PER_CTA><<<
        grid,
        SPARK_LM_CTA_THREADS,
        (size_t)shared_float_count * sizeof(float),
        stream>>>(
        q_bf16,
        q_row_stride,
        k_cache_bf16,
        v_cache_bf16,
        k_lane_stride,
        v_lane_stride,
        k_slot_stride,
        v_slot_stride,
        row_lane_indices,
        row_positions,
        sink_f32,
        scale,
        out_bf16,
        row_count,
        head_count,
        group_size,
        head_dim,
        value_dim,
        window_slots);
    return cudaGetLastError();
}

/*
 * Head grouping trades KV traffic against CTA count, and the right trade
 * depends entirely on batch. Grouping four query heads per CTA reads each KV
 * head a quarter as often, which is what you want once rows alone fill the
 * machine. At small batch it is backwards: mimo25 at one row yields sixteen
 * CTAs against forty-eight SMs, so two thirds of the GPU idles while each CTA
 * walks the whole context. One head per CTA gives sixty-four CTAs and a
 * quarter the shared memory, and the extra KV rereads are nearly free at one
 * row because a single row's KV slice is small.
 *
 * This picks the LARGEST grouping - least KV traffic - that still fills the
 * machine, and only falls to narrower groups when it would not. Splitting the
 * context across CTAs as well (an exact online-softmax merge over partials)
 * would add parallelism beyond this, but needs a workspace and a second pass;
 * it is worth doing only once this no longer fills the SMs.
 */
#define SPARK_LM_ATTN_TARGET_CTAS 96u

static inline cudaError_t SparkLmHostLaunchAdaptiveAttnDecode(cudaStream_t stream, const void *q_bf16, uint64_t q_row_stride, const void *k_cache_bf16, const void *v_cache_bf16, uint64_t k_lane_stride, uint64_t v_lane_stride, uint64_t k_slot_stride, uint64_t v_slot_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, const float *sink_f32, float scale, void *out_bf16, uint32_t row_count, uint32_t head_count, uint32_t group_size, uint32_t head_dim, uint32_t value_dim, uint32_t window_slots)
{
	uint32_t kv_head_count,ctas_group_four,ctas_group_two;
	if ( group_size == 0u || head_count % group_size != 0u )
		return(cudaErrorInvalidValue);
	kv_head_count = head_count / group_size;
	ctas_group_four = row_count * kv_head_count * ((group_size + 3u) / 4u);
	ctas_group_two = row_count * kv_head_count * ((group_size + 1u) / 2u);
	if ( ctas_group_four >= SPARK_LM_ATTN_TARGET_CTAS )
		return(SparkLmHostLaunchGroupedAttnDecode<4u>(stream,q_bf16,q_row_stride,k_cache_bf16,v_cache_bf16,k_lane_stride,v_lane_stride,k_slot_stride,v_slot_stride,row_lane_indices,row_positions,sink_f32,scale,out_bf16,row_count,head_count,group_size,head_dim,value_dim,window_slots));
	if ( ctas_group_two >= SPARK_LM_ATTN_TARGET_CTAS )
		return(SparkLmHostLaunchGroupedAttnDecode<2u>(stream,q_bf16,q_row_stride,k_cache_bf16,v_cache_bf16,k_lane_stride,v_lane_stride,k_slot_stride,v_slot_stride,row_lane_indices,row_positions,sink_f32,scale,out_bf16,row_count,head_count,group_size,head_dim,value_dim,window_slots));
	return(SparkLmHostLaunchGroupedAttnDecode<1u>(stream,q_bf16,q_row_stride,k_cache_bf16,v_cache_bf16,k_lane_stride,v_lane_stride,k_slot_stride,v_slot_stride,row_lane_indices,row_positions,sink_f32,scale,out_bf16,row_count,head_count,group_size,head_dim,value_dim,window_slots));
}


#include <mma.h>

#define SPARK_LM_TILE 16u
#define SPARK_LM_TILE_N 128u
#define SPARK_LM_TILE_K 64u

/*
 * Native SM121 decode compute.
 *
 * This is intentionally separate from the portable weight-only GEMM below.
 * The portable path decodes a packed weight to BF16 before mma, which is a
 * useful compatibility implementation but is not a native MX deployment
 * route.  DSV4 calls only this section for its qualified B1/B8/B1024 shapes.
 * Every quantized multiply reaches one of the architecture-gated block-scaled
 * atoms in inference/kernels/mma.cuh; unsupported shapes return an error at the
 * host wrapper and unsupported device code traps rather than falling back.
 */
#define SPARK_LM_SM121_NATIVE_TILE_M 16u
#define SPARK_LM_SM121_NATIVE_WIDE_TILE_M 64u
#define SPARK_LM_SM121_NATIVE_TILE_N 128u
#define SPARK_LM_SM121_NATIVE_MMA_N 8u
#define SPARK_LM_SM121_NATIVE_K 32u
#define SPARK_LM_SM121_NATIVE_WEIGHT_FP8 8u
#define SPARK_LM_SM121_NATIVE_WEIGHT_MXFP4 4u
// With one routed row per selected expert, N32 gives W13 exactly two CTAs per
// SM across the six selected TP4 experts. W2 uses N128 and four CTAs per SM:
// that leaves one complete task per CTA while halving activation restaging.
// Larger decode buckets keep the block-scaled MMA schedule.
#define SPARK_LM_SM121_B1_EXPERT_W13_TILE_N 32u
#define SPARK_LM_SM121_B1_EXPERT_W2_TILE_N 128u
#define SPARK_LM_SM121_B1_EXPERT_BLOCKS_PER_SM 2u
#define SPARK_LM_SM121_B1_EXPERT_W2_BLOCKS_PER_SM 4u

static inline uint32_t SparkLmSm121NativeDecodeShape(uint32_t rows)
{
	/* 7 = the DSpark serving block (k=7 drafts); it rides the generic
	 * NATIVE_TILE_N config alongside 5 (the trained block). */
	return(rows == 1u || rows == 5u || rows == 7u || rows == 8u || rows == 16u || rows == 32u || rows == 64u || rows == 1024u ? 1u : 0u);
}

static inline uint32_t SparkLmSm121ExpertW13TileN(uint32_t rows)
{
	return(rows == 1u ? SPARK_LM_SM121_B1_EXPERT_W13_TILE_N :
		SPARK_LM_SM121_NATIVE_TILE_N);
}

static inline uint32_t SparkLmSm121ExpertW2TileN(uint32_t rows)
{
	return(rows == 1u ? SPARK_LM_SM121_B1_EXPERT_W2_TILE_N :
		SPARK_LM_SM121_NATIVE_TILE_N);
}

static __device__ __forceinline__ uint8_t SparkLmSm121E8m0ScaleCode(float amax)
{
	float exponent = ceilf(log2f(fmaxf(amax,1.0e-4f) / LM_E4M3_MAX));
	int32_t code = (int32_t)exponent + 127;
	code = code < 0 ? 0 : (code > 254 ? 254 : code);
	return((uint8_t)code);
}

static __device__ __forceinline__ float SparkLmSm121E8m0ScaleValue(uint8_t code)
{
	return(exp2f((float)(int32_t)code - 127.0f));
}

static __device__ __forceinline__ float SparkLmSm121Bf16Round(float value)
{
	return(__bfloat162float(__float2bfloat16(value)));
}

/*
 * Quantize TILE_M BF16 rows to one native MXFP8 K32 operand tile.  A warp owns
 * a row, so the scale is the exact row-block amax and the payload written to
 * shared memory is the E4M3 byte representation consumed by mma.  Indirect
 * routed rows follow source_row_map; ragged rows become numeric zero and never
 * read beyond either the route map or the source tensor.
 */
template<uint32_t TILE_M>
static __device__ void SparkLmSm121StageMxf8(
	const void *input_bf16,
	uint64_t input_row_stride,
	uint32_t input_column_offset,
	const uint32_t *source_row_map,
	uint32_t source_row_count,
	uint32_t packed_row_base,
	uint32_t packed_row_limit,
	uint32_t k_base,
	uint8_t *payload_e4m3,
	uint8_t *scale_e8m0)
{
	static_assert(TILE_M == SPARK_LM_SM121_NATIVE_TILE_M ||
		TILE_M == SPARK_LM_SM121_NATIVE_WIDE_TILE_M,
		"SM121 MXFP8 staging has only the qualified tile heights");
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t local_row,packed_row,source_row;
	float value,amax,scale;
	uint8_t scale_code;
	for (local_row = warp; local_row < TILE_M;
		local_row += SPARK_LM_CTA_WARPS)
	{
		packed_row = packed_row_base + local_row;
		if ( packed_row < packed_row_limit )
		{
			source_row = source_row_map != 0
				? __ldg(source_row_map + packed_row) : packed_row;
			if ( source_row >= source_row_count )
				asm volatile("trap;\n");
			value = SparkLmBf16ToFloat(input_bf16,
				((uint64_t)source_row * input_row_stride) +
				input_column_offset + k_base + lane);
			amax = LmActivationWarpMax(fabsf(value));
			amax = __shfl_sync(0xffffffffu,amax,0u);
			scale_code = SparkLmSm121E8m0ScaleCode(amax);
			scale = SparkLmSm121E8m0ScaleValue(scale_code);
			payload_e4m3[(local_row * SPARK_LM_SM121_NATIVE_K) + lane] =
				LmFloatToE4m3(value / scale);
			if ( lane == 0u )
				scale_e8m0[local_row] = scale_code;
		}
		else
		{
			payload_e4m3[(local_row * SPARK_LM_SM121_NATIVE_K) + lane] = 0u;
			if ( lane == 0u )
				scale_e8m0[local_row] = 127u;
		}
	}
}

static __device__ __forceinline__ void SparkLmSm121LoadMxf8A(
	const uint8_t *payload_e4m3,
	uint32_t row_base,
	uint32_t lane,
	uint32_t a[4])
{
	uint32_t reg,byte_index,packed;
	#pragma unroll
	for (reg = 0u; reg < 4u; ++reg)
	{
		packed = 0u;
		#pragma unroll
		for (byte_index = 0u; byte_index < 4u; ++byte_index)
			packed |= (uint32_t)payload_e4m3[
				(row_base + LmMma8OperandARow(lane,reg)) *
				SPARK_LM_SM121_NATIVE_K +
				LmMma8OperandAByte(lane,reg) + byte_index]
				<< (byte_index * 8u);
		a[reg] = packed;
	}
}

static __device__ __forceinline__ uint32_t SparkLmSm121ScaleA(
	const uint8_t *scale_e8m0,
	uint32_t row_base,
	uint32_t lane)
{
	return((uint32_t)scale_e8m0[
		row_base + (8u * (lane % 2u)) + (lane / 4u)]);
}

static __device__ __forceinline__ void SparkLmSm121LoadMxf4B(
	const uint8_t *payload_e2m1,
	uint32_t neuron_base,
	uint32_t input_dimension,
	uint32_t k_base,
	uint32_t lane,
	uint32_t b[2])
{
	uint32_t reg,byte_index,k,packed,code_byte;
	uint32_t neuron = neuron_base + LmMma8OperandBRow(lane);
	uint64_t row_base = (uint64_t)neuron * (input_dimension >> 1u);
	#pragma unroll
	for (reg = 0u; reg < 2u; ++reg)
	{
		packed = 0u;
		#pragma unroll
		for (byte_index = 0u; byte_index < 4u; ++byte_index)
		{
			k = k_base + LmMma8OperandBByte(lane,reg) + byte_index;
			code_byte = __ldg(payload_e2m1 + row_base + (k >> 1u));
			/*
			 * kind::mxf8f6f4 requires E2M1 in bits [5:2] of each
			 * 8-bit register container.  The checkpoint stays nibble-packed
			 * until here; these are the required two padding bits per side.
			 */
			packed |= (((code_byte >> ((k & 1u) * 4u)) & 15u) << 2u)
				<< (byte_index * 8u);
		}
		b[reg] = packed;
	}
}

static __device__ __forceinline__ void SparkLmSm121LoadMxf8B(
	const uint8_t *payload_e4m3,
	uint32_t neuron_base,
	uint32_t input_dimension,
	uint32_t k_base,
	uint32_t lane,
	uint32_t b[2])
{
	uint32_t reg,byte_index,packed;
	uint32_t neuron = neuron_base + LmMma8OperandBRow(lane);
	uint64_t row_base = (uint64_t)neuron * input_dimension;
	#pragma unroll
	for (reg = 0u; reg < 2u; ++reg)
	{
		packed = 0u;
		#pragma unroll
		for (byte_index = 0u; byte_index < 4u; ++byte_index)
			packed |= (uint32_t)__ldg(payload_e4m3 + row_base + k_base +
				LmMma8OperandBByte(lane,reg) + byte_index)
				<< (byte_index * 8u);
		b[reg] = packed;
	}
}

template<uint32_t WEIGHT_BITS>
static __device__ __forceinline__ void SparkLmSm121LoadB(
	const uint8_t *payload,
	uint32_t neuron,
	uint32_t input_dimension,
	uint32_t k_base,
	uint32_t lane,
	uint32_t b[2])
{
	static_assert(WEIGHT_BITS == SPARK_LM_SM121_NATIVE_WEIGHT_FP8 ||
		WEIGHT_BITS == SPARK_LM_SM121_NATIVE_WEIGHT_MXFP4,
		"native SM121 weight width is not qualified");
	if constexpr ( WEIGHT_BITS == SPARK_LM_SM121_NATIVE_WEIGHT_MXFP4 )
		SparkLmSm121LoadMxf4B(payload,neuron,input_dimension,k_base,lane,b);
	else
		SparkLmSm121LoadMxf8B(payload,neuron,input_dimension,k_base,lane,b);
}

template<uint32_t WEIGHT_BITS>
static __device__ __forceinline__ uint32_t SparkLmSm121ScaleB(
	const uint8_t *scale_e8m0,
	uint32_t neuron_base,
	uint32_t input_dimension,
	uint32_t k_base,
	uint32_t lane)
{
	constexpr uint32_t group = WEIGHT_BITS ==
		SPARK_LM_SM121_NATIVE_WEIGHT_MXFP4 ? 32u : 128u;
	uint32_t neuron = neuron_base + LmMma8OperandBRow(lane);
	return((uint32_t)__ldg(scale_e8m0 +
		(uint64_t)neuron * (input_dimension / group) + (k_base / group)));
}

template<uint32_t WEIGHT_BITS>
static __device__ __forceinline__ void SparkLmSm121Mma(
	float accumulator[4],
	const uint32_t a[4],
	const uint32_t b[2],
	uint32_t scale_a,
	uint32_t scale_b)
{
	if constexpr ( WEIGHT_BITS == SPARK_LM_SM121_NATIVE_WEIGHT_MXFP4 )
		LmMmaMxf8Mxf4(accumulator,a,b,scale_a,scale_b);
	else
		LmMmaMxf8Mxf8(accumulator,a,b,scale_a,scale_b);
}

static __device__ __forceinline__ uint32_t SparkLmSm121GroupOfTile(
	const uint32_t *tile_prefix,
	uint32_t group_count,
	uint32_t tile)
{
	uint32_t low = 0u,high = group_count;
	while ( low + 1u < high )
	{
		uint32_t middle = low + ((high - low) >> 1u);
		if ( __ldg(tile_prefix + middle) <= tile )
			low = middle;
		else
			high = middle;
	}
	return(low);
}

static __device__ __forceinline__ void SparkLmSm121StageB1Activation(
	const void *input_bf16,
	uint32_t row,
	uint32_t row_stride,
	uint32_t width,
	float *shared_input)
{
	uint32_t element;
	for (element=threadIdx.x; element<width; element+=blockDim.x)
		shared_input[element] = SparkLmBf16ToFloat(input_bf16,
			((uint64_t)row * row_stride) + element);
	__syncthreads();
	LmActivationFp8QdqFloatRow<
		SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0>(shared_input,width);
	__syncthreads();
}

template<uint32_t TILE_N>
static __device__ __forceinline__ void SparkLmSm121B1ExpertTask(
	const uint32_t *group_row_offset,
	const uint32_t *group_tile_prefix,
	uint32_t group_count,
	uint32_t output_dimension,
	uint32_t task,
	uint32_t *group,
	uint32_t *row,
	uint32_t *neuron_base)
{
	uint32_t in_group,neuron_tiles;
	*group = SparkLmSm121GroupOfTile(group_tile_prefix,group_count,task);
	in_group = task - __ldg(group_tile_prefix + *group);
	neuron_tiles = output_dimension / TILE_N;
	*row = __ldg(group_row_offset + *group) +
		(in_group / neuron_tiles) * SPARK_LM_SM121_NATIVE_TILE_M;
	*neuron_base = (in_group % neuron_tiles) * TILE_N;
}

template<uint32_t TILE_M,uint32_t WEIGHT_BITS>
static __global__ __launch_bounds__(SPARK_LM_CTA_THREADS,1)
void SparkLmSm121NativeLinearKernel(
	const uint8_t *weight_payload,
	const uint8_t *weight_scale_e8m0,
	uint64_t weight_payload_group_stride_bytes,
	uint64_t weight_scale_group_stride_bytes,
	const void *input_bf16,
	uint64_t input_row_stride,
	uint32_t input_offset,
	uint32_t input_group_stride,
	void *output_bf16,
	uint64_t output_row_stride,
	uint32_t output_offset,
	uint32_t output_group_stride,
	uint32_t group_count,
	uint32_t row_count,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	__shared__ uint8_t activation_e4m3[TILE_M * SPARK_LM_SM121_NATIVE_K];
	__shared__ uint8_t activation_scale_e8m0[TILE_M];
	float total[2][4] = {};
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t group = blockIdx.z;
	uint32_t row_base = blockIdx.x * TILE_M;
	uint32_t neuron_base = blockIdx.y * SPARK_LM_SM121_NATIVE_TILE_N +
		warp * (2u * SPARK_LM_SM121_NATIVE_MMA_N);
	const uint8_t *group_payload = weight_payload +
		(uint64_t)group * weight_payload_group_stride_bytes;
	const uint8_t *group_scale = weight_scale_e8m0 +
		(uint64_t)group * weight_scale_group_stride_bytes;
	uint32_t k_base,ni,entry,a[4],b[2],scale_a,scale_b,row,column;
	for (k_base = 0u; k_base < input_dimension;
		k_base += SPARK_LM_SM121_NATIVE_K)
	{
		SparkLmSm121StageMxf8<TILE_M>(input_bf16,input_row_stride,
			input_offset + group * input_group_stride,0,row_count,row_base,
			row_count,k_base,activation_e4m3,activation_scale_e8m0);
		__syncthreads();
		SparkLmSm121LoadMxf8A(activation_e4m3,0u,lane,a);
		scale_a = SparkLmSm121ScaleA(activation_scale_e8m0,0u,lane);
		#pragma unroll
		for (ni = 0u; ni < 2u; ++ni)
		{
			uint32_t fragment_neuron = neuron_base +
				ni * SPARK_LM_SM121_NATIVE_MMA_N;
			SparkLmSm121LoadB<WEIGHT_BITS>(group_payload,fragment_neuron,
				input_dimension,k_base,lane,b);
			scale_b = SparkLmSm121ScaleB<WEIGHT_BITS>(group_scale,
				fragment_neuron,input_dimension,k_base,lane);
			SparkLmSm121Mma<WEIGHT_BITS>(total[ni],a,b,scale_a,scale_b);
		}
		__syncthreads();
	}
	#pragma unroll
	for (ni = 0u; ni < 2u; ++ni)
		#pragma unroll
		for (entry = 0u; entry < 4u; ++entry)
		{
			row = row_base + LmMmaAccumulatorRow(lane,entry);
			column = neuron_base + ni * SPARK_LM_SM121_NATIVE_MMA_N +
				LmMmaAccumulatorColumn(lane,entry);
			if ( row < row_count && column < output_dimension )
				SparkLmFloatToBf16(output_bf16,
					(uint64_t)row * output_row_stride + output_offset +
					group * output_group_stride + column,total[ni][entry]);
		}
}

template<uint32_t TILE_M,uint32_t WEIGHT_BITS>
static __global__ __launch_bounds__(SPARK_LM_CTA_THREADS,1)
void SparkLmSm121FusedDenseW13Kernel(
	const uint8_t *w1_payload,
	const uint8_t *w1_scale_e8m0,
	const uint8_t *w3_payload,
	const uint8_t *w3_scale_e8m0,
	const void *input_bf16,
	void *activated_bf16,
	uint32_t row_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	float limit)
{
	__shared__ uint8_t activation_e4m3[TILE_M * SPARK_LM_SM121_NATIVE_K];
	__shared__ uint8_t activation_scale_e8m0[TILE_M];
	float gate_total[2][4] = {};
	float up_total[2][4] = {};
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t row_base = blockIdx.x * TILE_M;
	uint32_t neuron_base = blockIdx.y * SPARK_LM_SM121_NATIVE_TILE_N +
		warp * (2u * SPARK_LM_SM121_NATIVE_MMA_N);
	uint32_t k_base,ni,entry,a[4],b1[2],b3[2],scale_a,scale_b1,scale_b3;
	uint32_t row,column;
	for (k_base = 0u; k_base < input_dimension;
		k_base += SPARK_LM_SM121_NATIVE_K)
	{
		SparkLmSm121StageMxf8<TILE_M>(input_bf16,input_dimension,0u,0,
			row_count,row_base,row_count,k_base,activation_e4m3,
			activation_scale_e8m0);
		__syncthreads();
		SparkLmSm121LoadMxf8A(activation_e4m3,0u,lane,a);
		scale_a = SparkLmSm121ScaleA(activation_scale_e8m0,0u,lane);
		#pragma unroll
		for (ni = 0u; ni < 2u; ++ni)
		{
			uint32_t fragment_neuron = neuron_base +
				ni * SPARK_LM_SM121_NATIVE_MMA_N;
			SparkLmSm121LoadB<WEIGHT_BITS>(w1_payload,fragment_neuron,
				input_dimension,k_base,lane,b1);
			SparkLmSm121LoadB<WEIGHT_BITS>(w3_payload,fragment_neuron,
				input_dimension,k_base,lane,b3);
			scale_b1 = SparkLmSm121ScaleB<WEIGHT_BITS>(w1_scale_e8m0,
				fragment_neuron,input_dimension,k_base,lane);
			scale_b3 = SparkLmSm121ScaleB<WEIGHT_BITS>(w3_scale_e8m0,
				fragment_neuron,input_dimension,k_base,lane);
			SparkLmSm121Mma<WEIGHT_BITS>(gate_total[ni],a,b1,scale_a,scale_b1);
			SparkLmSm121Mma<WEIGHT_BITS>(up_total[ni],a,b3,scale_a,scale_b3);
		}
		__syncthreads();
	}
	#pragma unroll
	for (ni = 0u; ni < 2u; ++ni)
		#pragma unroll
		for (entry = 0u; entry < 4u; ++entry)
		{
			float gate,up;
			row = row_base + LmMmaAccumulatorRow(lane,entry);
			column = neuron_base + ni * SPARK_LM_SM121_NATIVE_MMA_N +
				LmMmaAccumulatorColumn(lane,entry);
			if ( row >= row_count || column >= output_dimension )
				continue;
			/* GA boundary: projection -> BF16 -> clamp/SiLU/product -> BF16. */
			gate = SparkLmSm121Bf16Round(gate_total[ni][entry]);
			up = SparkLmSm121Bf16Round(up_total[ni][entry]);
			if ( limit > 0.0f )
			{
				gate = gate > limit ? limit : gate;
				up = up > limit ? limit : (up < -limit ? -limit : up);
			}
			SparkLmFloatToBf16(activated_bf16,
				(uint64_t)row * output_dimension + column,
				SparkLmSwish(gate) * up);
		}
}

template<uint32_t TILE_M>
static __global__ __launch_bounds__(SPARK_LM_CTA_THREADS,1)
void SparkLmSm121FusedExpertW13Kernel(
	const uint8_t *w1_payload,
	const uint8_t *w1_scale_e8m0,
	const uint8_t *w3_payload,
	const uint8_t *w3_scale_e8m0,
	const void *input_bf16,
	const uint32_t *source_row_map,
	const uint32_t *group_row_offset,
	const uint32_t *group_tile_prefix,
	void *activated_bf16,
	uint32_t source_row_count,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	float limit)
{
	__shared__ uint8_t activation_e4m3[TILE_M * SPARK_LM_SM121_NATIVE_K];
	__shared__ uint8_t activation_scale_e8m0[TILE_M];
	constexpr uint32_t m_fragments = TILE_M / SPARK_LM_SM121_NATIVE_TILE_M;
	float gate_total[m_fragments][2][4] = {};
	float up_total[m_fragments][2][4] = {};
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t neuron_tiles = output_dimension / SPARK_LM_SM121_NATIVE_TILE_N;
	uint32_t total_tiles = __ldg(group_tile_prefix + group_count);
	uint32_t tile,group,in_group,row_base,row_limit,neuron_base,k_base,mi,ni;
	uint32_t entry,a[4],b1[2],b3[2],scale_a,scale_b1,scale_b3,row,column;
	for (tile = blockIdx.x; tile < total_tiles; tile += gridDim.x)
	{
		group = SparkLmSm121GroupOfTile(group_tile_prefix,group_count,tile);
		in_group = tile - __ldg(group_tile_prefix + group);
		row_base = __ldg(group_row_offset + group) +
			(in_group / neuron_tiles) * TILE_M;
		row_limit = __ldg(group_row_offset + group + 1u);
		neuron_base = (in_group % neuron_tiles) * SPARK_LM_SM121_NATIVE_TILE_N +
			warp * (2u * SPARK_LM_SM121_NATIVE_MMA_N);
		#pragma unroll
		for (mi = 0u; mi < m_fragments; ++mi)
			#pragma unroll
			for (ni = 0u; ni < 2u; ++ni)
				#pragma unroll
				for (entry = 0u; entry < 4u; ++entry)
				{
					gate_total[mi][ni][entry] = 0.0f;
					up_total[mi][ni][entry] = 0.0f;
				}
		const uint64_t payload_group_stride =
			(uint64_t)output_dimension * input_dimension / 2u;
		const uint64_t scale_group_stride =
			(uint64_t)output_dimension * (input_dimension / 32u);
		const uint8_t *group_w1 = w1_payload + group * payload_group_stride;
		const uint8_t *group_w3 = w3_payload + group * payload_group_stride;
		const uint8_t *group_s1 = w1_scale_e8m0 + group * scale_group_stride;
		const uint8_t *group_s3 = w3_scale_e8m0 + group * scale_group_stride;
		for (k_base = 0u; k_base < input_dimension;
			k_base += SPARK_LM_SM121_NATIVE_K)
		{
			SparkLmSm121StageMxf8<TILE_M>(input_bf16,input_dimension,0u,
				source_row_map,source_row_count,row_base,row_limit,k_base,
				activation_e4m3,activation_scale_e8m0);
			__syncthreads();
			#pragma unroll
			for (ni = 0u; ni < 2u; ++ni)
			{
				uint32_t fragment_neuron = neuron_base +
					ni * SPARK_LM_SM121_NATIVE_MMA_N;
				SparkLmSm121LoadMxf4B(group_w1,fragment_neuron,
					input_dimension,k_base,lane,b1);
				SparkLmSm121LoadMxf4B(group_w3,fragment_neuron,
					input_dimension,k_base,lane,b3);
				scale_b1 = SparkLmSm121ScaleB<4u>(group_s1,fragment_neuron,
					input_dimension,k_base,lane);
				scale_b3 = SparkLmSm121ScaleB<4u>(group_s3,fragment_neuron,
					input_dimension,k_base,lane);
				#pragma unroll
				for (mi = 0u; mi < m_fragments; ++mi)
				{
					SparkLmSm121LoadMxf8A(activation_e4m3,
						mi * SPARK_LM_SM121_NATIVE_TILE_M,lane,a);
					scale_a = SparkLmSm121ScaleA(activation_scale_e8m0,
						mi * SPARK_LM_SM121_NATIVE_TILE_M,lane);
					LmMmaMxf8Mxf4(gate_total[mi][ni],a,b1,scale_a,scale_b1);
					LmMmaMxf8Mxf4(up_total[mi][ni],a,b3,scale_a,scale_b3);
				}
			}
			__syncthreads();
		}
		#pragma unroll
		for (mi = 0u; mi < m_fragments; ++mi)
			#pragma unroll
			for (ni = 0u; ni < 2u; ++ni)
				#pragma unroll
				for (entry = 0u; entry < 4u; ++entry)
				{
					float gate,up;
					row = row_base + mi * SPARK_LM_SM121_NATIVE_TILE_M +
						LmMmaAccumulatorRow(lane,entry);
					column = neuron_base + ni * SPARK_LM_SM121_NATIVE_MMA_N +
						LmMmaAccumulatorColumn(lane,entry);
					if ( row >= row_limit || column >= output_dimension )
						continue;
					gate = SparkLmSm121Bf16Round(gate_total[mi][ni][entry]);
					up = SparkLmSm121Bf16Round(up_total[mi][ni][entry]);
					if ( limit > 0.0f )
					{
						gate = gate > limit ? limit : gate;
						up = up > limit ? limit : (up < -limit ? -limit : up);
					}
					SparkLmFloatToBf16(activated_bf16,
						(uint64_t)row * output_dimension + column,
						SparkLmSwish(gate) * up);
				}
		__syncthreads();
	}
}

template<uint32_t TILE_N>
static __device__ __forceinline__ void SparkLmSm121B1ExpertW13Task(
	const uint8_t *w1_payload,
	const uint8_t *w1_scale_e8m0,
	const uint8_t *w3_payload,
	const uint8_t *w3_scale_e8m0,
	const void *input_bf16,
	const uint32_t *source_row_map,
	void *activated_bf16,
	uint32_t source_row_count,
	uint32_t group,
	uint32_t row,
	uint32_t neuron_base,
	uint32_t input_dimension,
	uint32_t output_dimension,
	float limit,
	float *shared_input)
{
	uint64_t payload_stride = (uint64_t)output_dimension * input_dimension / 2u;
	uint64_t scale_stride = (uint64_t)output_dimension * (input_dimension / 32u);
	const uint8_t *group_w1 = w1_payload + ((uint64_t)group * payload_stride);
	const uint8_t *group_w3 = w3_payload + ((uint64_t)group * payload_stride);
	const uint8_t *group_s1 = w1_scale_e8m0 + ((uint64_t)group * scale_stride);
	const uint8_t *group_s3 = w3_scale_e8m0 + ((uint64_t)group * scale_stride);
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t source_row = __ldg(source_row_map + row),neuron;
	float gate,up;
	if ( source_row >= source_row_count )
		asm volatile("trap;\n");
	SparkLmSm121StageB1Activation(input_bf16,source_row,input_dimension,
		input_dimension,shared_input);
	for (neuron=neuron_base + warp; neuron<neuron_base + TILE_N;
		neuron+=SPARK_LM_CTA_WARPS)
	{
		SparkLmDotRowMxfp4Pair<32u>(shared_input,group_w1,group_s1,group_w3,
			group_s3,neuron,input_dimension,lane,&gate,&up);
		gate = SparkLmWarpReduceSum(gate);
		up = SparkLmWarpReduceSum(up);
		if ( lane == 0u )
		{
			gate = SparkLmSm121Bf16Round(gate);
			up = SparkLmSm121Bf16Round(up);
			gate = gate > limit ? limit : gate;
			up = up > limit ? limit : (up < -limit ? -limit : up);
			SparkLmFloatToBf16(activated_bf16,
				((uint64_t)row * output_dimension) + neuron,
				SparkLmSwish(gate) * up);
		}
	}
	__syncthreads();
}

template<uint32_t TILE_N>
static __global__ __launch_bounds__(SPARK_LM_CTA_THREADS,1)
void SparkLmSm121B1ExpertW13Kernel(
	const uint8_t *w1_payload,
	const uint8_t *w1_scale_e8m0,
	const uint8_t *w3_payload,
	const uint8_t *w3_scale_e8m0,
	const void *input_bf16,
	const uint32_t *source_row_map,
	const uint32_t *group_row_offset,
	const uint32_t *group_tile_prefix,
	void *activated_bf16,
	uint32_t source_row_count,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	float limit)
{
	extern __shared__ float shared_input[];
	uint32_t task,group,row,neuron_base,row_limit,row_end;
	uint32_t total_tasks = __ldg(group_tile_prefix + group_count);
	for (task=blockIdx.x; task<total_tasks; task+=gridDim.x)
	{
		SparkLmSm121B1ExpertTask<TILE_N>(group_row_offset,group_tile_prefix,
			group_count,output_dimension,task,&group,&row,&neuron_base);
		/* The route build groups rows in TILE_M-strided tiles; the B1 task
		 * covers ONE row per launch, so walk the tile's consecutive packed
		 * rows here (rows==1 batches: exactly one row, unchanged). */
		row_limit = __ldg(group_row_offset + group + 1u);
		row_end = row + SPARK_LM_SM121_NATIVE_TILE_M;
		if ( row_end > row_limit )
			row_end = row_limit;
		for (; row < row_end; row++)
		{
			if ( row >= row_limit )
				asm volatile("trap;\n");
			SparkLmSm121B1ExpertW13Task<TILE_N>(w1_payload,w1_scale_e8m0,
				w3_payload,w3_scale_e8m0,input_bf16,source_row_map,
				activated_bf16,source_row_count,group,row,neuron_base,
				input_dimension,output_dimension,limit,shared_input);
		}
	}
}

template<uint32_t TILE_N>
static __device__ __forceinline__ void SparkLmSm121B1ExpertW2Task(
	const uint8_t *weight_payload,
	const uint8_t *weight_scale_e8m0,
	const void *input_bf16,
	void *output_bf16,
	uint32_t group,
	uint32_t row,
	uint32_t neuron_base,
	uint32_t input_dimension,
	uint32_t output_dimension,
	float *shared_input)
{
	uint64_t payload_stride = (uint64_t)output_dimension * input_dimension / 2u;
	uint64_t scale_stride = (uint64_t)output_dimension * (input_dimension / 32u);
	const uint8_t *group_payload = weight_payload + ((uint64_t)group * payload_stride);
	const uint8_t *group_scale = weight_scale_e8m0 + ((uint64_t)group * scale_stride);
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t neuron;
	float value;
	SparkLmSm121StageB1Activation(input_bf16,row,input_dimension,input_dimension,
		shared_input);
	for (neuron=neuron_base + warp; neuron<neuron_base + TILE_N;
		neuron+=SPARK_LM_CTA_WARPS)
	{
		value = SparkLmDotRowMxfp4<32u>(shared_input,group_payload,group_scale,
			neuron,input_dimension,lane);
		value = SparkLmWarpReduceSum(value);
		if ( lane == 0u )
			SparkLmFloatToBf16(output_bf16,
				((uint64_t)row * output_dimension) + neuron,value);
	}
	__syncthreads();
}

template<uint32_t TILE_N>
static __global__ __launch_bounds__(SPARK_LM_CTA_THREADS,1)
void SparkLmSm121B1ExpertW2Kernel(
	const uint8_t *weight_payload,
	const uint8_t *weight_scale_e8m0,
	const void *input_bf16,
	const uint32_t *group_row_offset,
	const uint32_t *group_tile_prefix,
	void *output_bf16,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	extern __shared__ float shared_input[];
	uint32_t task,group,row,neuron_base,row_limit,row_end;
	uint32_t total_tasks = __ldg(group_tile_prefix + group_count);
	for (task=blockIdx.x; task<total_tasks; task+=gridDim.x)
	{
		SparkLmSm121B1ExpertTask<TILE_N>(group_row_offset,group_tile_prefix,
			group_count,output_dimension,task,&group,&row,&neuron_base);
		/* Walk the tile's consecutive packed rows (see the W13 kernel). */
		row_limit = __ldg(group_row_offset + group + 1u);
		row_end = row + SPARK_LM_SM121_NATIVE_TILE_M;
		if ( row_end > row_limit )
			row_end = row_limit;
		for (; row < row_end; row++)
		{
			if ( row >= row_limit )
				asm volatile("trap;\n");
			SparkLmSm121B1ExpertW2Task<TILE_N>(weight_payload,
				weight_scale_e8m0,input_bf16,output_bf16,group,row,
				neuron_base,input_dimension,output_dimension,shared_input);
		}
	}
}

template<uint32_t WEIGHT_BITS>
static __global__ __launch_bounds__(SPARK_LM_SM121_B1_DENSE_W13_CTA_THREADS,1)
void SparkLmSm121FusedDenseW13GemvKernel(
	const uint8_t *w1_payload,
	const uint8_t *w1_scale_e8m0,
	const uint8_t *w3_payload,
	const uint8_t *w3_scale_e8m0,
	const void *input_bf16,
	void *activated_bf16,
	uint32_t input_dimension,
	uint32_t output_dimension,
	float limit)
{
	extern __shared__ float shared_input[];
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t neuron = blockIdx.x * SPARK_LM_SM121_B1_DENSE_W13_CTA_WARPS +
		warp,element;
	float gate,up;
	for (element=threadIdx.x; element<input_dimension; element+=blockDim.x)
		shared_input[element] = SparkLmBf16ToFloat(input_bf16,element);
	__syncthreads();
	LmActivationFp8QdqFloatRow<SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0>(shared_input,input_dimension);
	__syncthreads();
	if ( neuron >= output_dimension )
		return;
	if constexpr ( WEIGHT_BITS == SPARK_LM_SM121_NATIVE_WEIGHT_FP8 )
	{
		gate = SparkLmDotRowFp8<128u>(shared_input,w1_payload,w1_scale_e8m0,neuron,input_dimension,lane);
		up = SparkLmDotRowFp8<128u>(shared_input,w3_payload,w3_scale_e8m0,neuron,input_dimension,lane);
	}
	else
	{
		gate = SparkLmDotRowMxfp4<32u>(shared_input,w1_payload,w1_scale_e8m0,neuron,input_dimension,lane);
		up = SparkLmDotRowMxfp4<32u>(shared_input,w3_payload,w3_scale_e8m0,neuron,input_dimension,lane);
	}
	gate = SparkLmWarpReduceSum(gate);
	up = SparkLmWarpReduceSum(up);
	if ( lane == 0u )
	{
		gate = SparkLmSm121Bf16Round(gate);
		up = SparkLmSm121Bf16Round(up);
		gate = gate > limit ? limit : gate;
		up = up > limit ? limit : (up < -limit ? -limit : up);
		SparkLmFloatToBf16(activated_bf16,neuron,SparkLmSwish(gate) * up);
	}
}

template<uint32_t TILE_M>
static __global__ __launch_bounds__(SPARK_LM_CTA_THREADS,1)
void SparkLmSm121ExpertW2Kernel(
	const uint8_t *weight_payload,
	const uint8_t *weight_scale_e8m0,
	const void *input_bf16,
	const uint32_t *group_row_offset,
	const uint32_t *group_tile_prefix,
	void *output_bf16,
	uint32_t packed_row_count,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	__shared__ uint8_t activation_e4m3[TILE_M * SPARK_LM_SM121_NATIVE_K];
	__shared__ uint8_t activation_scale_e8m0[TILE_M];
	constexpr uint32_t m_fragments = TILE_M / SPARK_LM_SM121_NATIVE_TILE_M;
	float total[m_fragments][2][4] = {};
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
	uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint32_t neuron_tiles = output_dimension / SPARK_LM_SM121_NATIVE_TILE_N;
	uint32_t total_tiles = __ldg(group_tile_prefix + group_count);
	uint32_t tile,group,in_group,row_base,row_limit,neuron_base,k_base,mi,ni;
	uint32_t entry,a[4],b[2],scale_a,scale_b,row,column;
	for (tile = blockIdx.x; tile < total_tiles; tile += gridDim.x)
	{
		group = SparkLmSm121GroupOfTile(group_tile_prefix,group_count,tile);
		in_group = tile - __ldg(group_tile_prefix + group);
		row_base = __ldg(group_row_offset + group) +
			(in_group / neuron_tiles) * TILE_M;
		row_limit = __ldg(group_row_offset + group + 1u);
		neuron_base = (in_group % neuron_tiles) * SPARK_LM_SM121_NATIVE_TILE_N +
			warp * (2u * SPARK_LM_SM121_NATIVE_MMA_N);
		#pragma unroll
		for (mi = 0u; mi < m_fragments; ++mi)
			#pragma unroll
			for (ni = 0u; ni < 2u; ++ni)
				#pragma unroll
				for (entry = 0u; entry < 4u; ++entry)
					total[mi][ni][entry] = 0.0f;
		const uint64_t payload_group_stride =
			(uint64_t)output_dimension * input_dimension / 2u;
		const uint64_t scale_group_stride =
			(uint64_t)output_dimension * (input_dimension / 32u);
		const uint8_t *group_payload = weight_payload +
			group * payload_group_stride;
		const uint8_t *group_scale = weight_scale_e8m0 +
			group * scale_group_stride;
		for (k_base = 0u; k_base < input_dimension;
			k_base += SPARK_LM_SM121_NATIVE_K)
		{
			SparkLmSm121StageMxf8<TILE_M>(input_bf16,input_dimension,0u,0,
				packed_row_count,row_base,row_limit,k_base,activation_e4m3,
				activation_scale_e8m0);
			__syncthreads();
			#pragma unroll
			for (ni = 0u; ni < 2u; ++ni)
			{
				uint32_t fragment_neuron = neuron_base +
					ni * SPARK_LM_SM121_NATIVE_MMA_N;
				SparkLmSm121LoadMxf4B(group_payload,fragment_neuron,
					input_dimension,k_base,lane,b);
				scale_b = SparkLmSm121ScaleB<4u>(group_scale,fragment_neuron,
					input_dimension,k_base,lane);
				#pragma unroll
				for (mi = 0u; mi < m_fragments; ++mi)
				{
					SparkLmSm121LoadMxf8A(activation_e4m3,
						mi * SPARK_LM_SM121_NATIVE_TILE_M,lane,a);
					scale_a = SparkLmSm121ScaleA(activation_scale_e8m0,
						mi * SPARK_LM_SM121_NATIVE_TILE_M,lane);
					LmMmaMxf8Mxf4(total[mi][ni],a,b,scale_a,scale_b);
				}
			}
			__syncthreads();
		}
		#pragma unroll
		for (mi = 0u; mi < m_fragments; ++mi)
			#pragma unroll
			for (ni = 0u; ni < 2u; ++ni)
				#pragma unroll
				for (entry = 0u; entry < 4u; ++entry)
				{
					row = row_base + mi * SPARK_LM_SM121_NATIVE_TILE_M +
						LmMmaAccumulatorRow(lane,entry);
					column = neuron_base + ni * SPARK_LM_SM121_NATIVE_MMA_N +
						LmMmaAccumulatorColumn(lane,entry);
					if ( row < row_limit && column < output_dimension )
						SparkLmFloatToBf16(output_bf16,
							(uint64_t)row * output_dimension + column,
							total[mi][ni][entry]);
				}
		__syncthreads();
	}
}

template<uint32_t WEIGHT_BITS>
static inline cudaError_t SparkLmHostLaunchSm121NativeLinear(
	cudaStream_t stream,
	const void *weight_payload,
	const uint8_t *weight_scale_e8m0,
	uint64_t weight_payload_group_stride_bytes,
	uint64_t weight_scale_group_stride_bytes,
	const void *input_bf16,
	uint64_t input_row_stride,
	uint32_t input_offset,
	uint32_t input_group_stride,
	void *output_bf16,
	uint64_t output_row_stride,
	uint32_t output_offset,
	uint32_t output_group_stride,
	uint32_t group_count,
	uint32_t row_count,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	if ( weight_payload == 0 || weight_scale_e8m0 == 0 || input_bf16 == 0 ||
		output_bf16 == 0 || group_count == 0u ||
		SparkLmSm121NativeDecodeShape(row_count) == 0u ||
		input_dimension == 0u || input_dimension % 128u != 0u ||
		output_dimension == 0u ||
		output_dimension % SPARK_LM_SM121_NATIVE_TILE_N != 0u ||
		input_row_stride < (uint64_t)input_offset + input_dimension +
			(uint64_t)(group_count - 1u) * input_group_stride ||
		output_row_stride < (uint64_t)output_offset + output_dimension +
			(uint64_t)(group_count - 1u) * output_group_stride )
		return(cudaErrorInvalidValue);
	dim3 grid((row_count + SPARK_LM_SM121_NATIVE_TILE_M - 1u) /
		SPARK_LM_SM121_NATIVE_TILE_M,
		output_dimension / SPARK_LM_SM121_NATIVE_TILE_N,group_count);
	SparkLmSm121NativeLinearKernel<SPARK_LM_SM121_NATIVE_TILE_M,WEIGHT_BITS>
		<<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(
			(const uint8_t *)weight_payload,weight_scale_e8m0,
			weight_payload_group_stride_bytes,weight_scale_group_stride_bytes,
			input_bf16,input_row_stride,input_offset,input_group_stride,
			output_bf16,output_row_stride,output_offset,output_group_stride,
			group_count,row_count,input_dimension,output_dimension);
	return(cudaGetLastError());
}

template<uint32_t WEIGHT_BITS>
static inline cudaError_t SparkLmHostLaunchSm121FusedDenseW13(
	cudaStream_t stream,
	const void *w1_payload,
	const uint8_t *w1_scale_e8m0,
	const void *w3_payload,
	const uint8_t *w3_scale_e8m0,
	const void *input_bf16,
	void *activated_bf16,
	uint32_t row_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	float limit)
{
	if ( w1_payload == 0 || w1_scale_e8m0 == 0 || w3_payload == 0 ||
		w3_scale_e8m0 == 0 || input_bf16 == 0 || activated_bf16 == 0 ||
		SparkLmSm121NativeDecodeShape(row_count) == 0u ||
		input_dimension == 0u || input_dimension % 128u != 0u ||
		output_dimension == 0u ||
		output_dimension % SPARK_LM_SM121_NATIVE_TILE_N != 0u || limit <= 0.0f )
		return(cudaErrorInvalidValue);
	/* Multi-row must be bit-identical to the certified 1-row math: the
	 * native tiled kernel quantizes the activation with per-TILE scales
	 * (a tile spans rows), so a rows>1 batch diverges from the 1-row
	 * GEMV's per-row quantization. Launch the exact GEMV once per row,
	 * base pointers advanced by the row strides. */
	{
		uint32_t row;
		for (row = 0u; row < row_count; row++)
		{
			const uint8_t *row_input = (const uint8_t *)input_bf16 +
				(uint64_t)row * input_dimension * 2u;
			uint8_t *row_activated = (uint8_t *)activated_bf16 +
				(uint64_t)row * output_dimension * 2u;
			SparkLmSm121FusedDenseW13GemvKernel<WEIGHT_BITS>
				<<<((output_dimension +
					SPARK_LM_SM121_B1_DENSE_W13_CTA_WARPS -
					1u) / SPARK_LM_SM121_B1_DENSE_W13_CTA_WARPS),
					SPARK_LM_SM121_B1_DENSE_W13_CTA_THREADS,
					input_dimension * sizeof(float),stream>>>(
					(const uint8_t *)w1_payload,w1_scale_e8m0,
					(const uint8_t *)w3_payload,w3_scale_e8m0,row_input,
					row_activated,input_dimension,output_dimension,limit);
		}
		return(cudaGetLastError());
	}
}

static inline cudaError_t SparkLmHostLaunchSm121FusedExpertW13(
	cudaStream_t stream,
	const void *w1_payload,
	const uint8_t *w1_scale_e8m0,
	const void *w3_payload,
	const uint8_t *w3_scale_e8m0,
	const void *input_bf16,
	const uint32_t *source_row_map,
	const uint32_t *group_row_offset,
	const uint32_t *group_tile_prefix,
	void *activated_bf16,
	uint32_t rows,
	uint32_t top_k,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	float limit,
	uint32_t multiprocessor_count)
{
	uint32_t block_count;
	if ( w1_payload == 0 || w1_scale_e8m0 == 0 || w3_payload == 0 ||
		w3_scale_e8m0 == 0 || input_bf16 == 0 || source_row_map == 0 ||
		group_row_offset == 0 || group_tile_prefix == 0 ||
		activated_bf16 == 0 || SparkLmSm121NativeDecodeShape(rows) == 0u ||
		top_k == 0u || group_count == 0u || input_dimension % 128u != 0u ||
		output_dimension == 0u ||
		output_dimension % SPARK_LM_SM121_NATIVE_TILE_N != 0u ||
		limit <= 0.0f || multiprocessor_count == 0u ||
		multiprocessor_count > UINT32_MAX /
			SPARK_LM_SM121_B1_EXPERT_BLOCKS_PER_SM )
		return(cudaErrorInvalidValue);
	/* All row counts use the certified B1 exact kernel (per-row activation
	 * quantization): the native tiled kernel's per-tile scales diverge for
	 * rows>1 batches. The B1 kernel enumerates every (group,row,neuron-
	 * tile) task via the shared prefix structures; the per-neuron math is
	 * tile-size independent, so instantiate it with the tile N the route
	 * build actually used for the prefixes (32 for the 1-row cooperative
	 * build, 128 for the batched build - see SparkLmSm121ExpertW13TileN). */
	block_count = multiprocessor_count *
		SPARK_LM_SM121_B1_EXPERT_BLOCKS_PER_SM;
	if ( rows == 1u )
		SparkLmSm121B1ExpertW13Kernel<SPARK_LM_SM121_B1_EXPERT_W13_TILE_N>
			<<<block_count,SPARK_LM_CTA_THREADS,
				input_dimension * sizeof(float),stream>>>(
				(const uint8_t *)w1_payload,w1_scale_e8m0,
				(const uint8_t *)w3_payload,w3_scale_e8m0,input_bf16,
				source_row_map,group_row_offset,group_tile_prefix,
				activated_bf16,rows,group_count,input_dimension,
				output_dimension,limit);
	else
		SparkLmSm121B1ExpertW13Kernel<SPARK_LM_SM121_NATIVE_TILE_N>
			<<<block_count,SPARK_LM_CTA_THREADS,
				input_dimension * sizeof(float),stream>>>(
				(const uint8_t *)w1_payload,w1_scale_e8m0,
				(const uint8_t *)w3_payload,w3_scale_e8m0,input_bf16,
				source_row_map,group_row_offset,group_tile_prefix,
				activated_bf16,rows,group_count,input_dimension,
				output_dimension,limit);
	return(cudaGetLastError());
}

static inline cudaError_t SparkLmHostLaunchSm121ExpertW2(
	cudaStream_t stream,
	const void *weight_payload,
	const uint8_t *weight_scale_e8m0,
	const void *input_bf16,
	const uint32_t *group_row_offset,
	const uint32_t *group_tile_prefix,
	void *output_bf16,
	uint32_t rows,
	uint32_t top_k,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	uint32_t multiprocessor_count)
{
	uint32_t block_count,packed_rows;
	if ( weight_payload == 0 || weight_scale_e8m0 == 0 || input_bf16 == 0 ||
		group_row_offset == 0 || group_tile_prefix == 0 || output_bf16 == 0 ||
		SparkLmSm121NativeDecodeShape(rows) == 0u || top_k == 0u ||
		rows > UINT32_MAX / top_k || group_count == 0u ||
		input_dimension % 128u != 0u || output_dimension == 0u ||
		output_dimension % SPARK_LM_SM121_NATIVE_TILE_N != 0u ||
		multiprocessor_count == 0u || multiprocessor_count > UINT32_MAX /
			SPARK_LM_SM121_B1_EXPERT_W2_BLOCKS_PER_SM )
		return(cudaErrorInvalidValue);
	packed_rows = rows * top_k;
	(void)packed_rows;
	/* All row counts use the certified B1 exact kernel (per-row activation
	 * quantization); the native tiled kernel's per-tile scales diverge for
	 * rows>1 batches. The B1 kernel enumerates all (group,row,neuron-tile)
	 * tasks via the shared prefix structures. */
	block_count = multiprocessor_count *
		SPARK_LM_SM121_B1_EXPERT_W2_BLOCKS_PER_SM;
	SparkLmSm121B1ExpertW2Kernel<SPARK_LM_SM121_B1_EXPERT_W2_TILE_N>
		<<<block_count,SPARK_LM_CTA_THREADS,
			input_dimension * sizeof(float),stream>>>(
			(const uint8_t *)weight_payload,weight_scale_e8m0,input_bf16,
			group_row_offset,group_tile_prefix,output_bf16,group_count,
			input_dimension,output_dimension);
	return(cudaGetLastError());
}

/*
 * Decode a 32-element contiguous run of one weight row into bf16, the
 * tile stager's unit: vector payload loads (4-byte lanes, 16-byte for
 * bf16), one scale fetch per in-group run. The format is launch-uniform
 * so the branch never diverges.
 */
template <uint32_t GROUP_SIZE>
static __device__ __forceinline__ void SparkLmTileDecodeRun(uint32_t weight_format, const void *weight_payload, const void *weight_scale, uint32_t neuron, uint32_t k_base, uint32_t input_dimension, __nv_bfloat16 *destination)
{
	uint32_t chunk,byte_index,packed;
	float scale_value;
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_BF16 )
	{
		#pragma unroll
		for (chunk = 0; chunk < 4u; chunk++)
			((uint4 *)destination)[chunk] = __ldg(((const uint4 *)weight_payload) + ((((uint64_t)neuron * input_dimension) + k_base) >> 3u) + chunk);
		return;
	}
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1 )
	{
		#pragma unroll
		for (chunk = 0; chunk < 4u; chunk++)
		{
			packed = __ldg(((const uint32_t *)weight_payload) + ((((uint64_t)neuron * input_dimension) + k_base) >> 3u) + chunk);
			scale_value = SparkLmDecodeE8m0(((const uint8_t *)weight_scale)[((uint64_t)neuron * (input_dimension / GROUP_SIZE)) + ((k_base + (chunk << 3u)) / GROUP_SIZE)]);
			#pragma unroll
			for (byte_index = 0; byte_index < 8u; byte_index++)
				destination[(chunk << 3u) + byte_index] = __float2bfloat16(SparkLmDecodeE2m1((packed >> (byte_index << 2u)) & 0x0fu) * scale_value);
		}
		return;
	}
	#pragma unroll
	for (chunk = 0; chunk < 8u; chunk++)
	{
		packed = __ldg(((const uint32_t *)weight_payload) + ((((uint64_t)neuron * input_dimension) + k_base) >> 2u) + chunk);
		if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
			scale_value = __ldg(((const float *)weight_scale) + (((uint64_t)(neuron / 128u)) * (input_dimension / 128u)) + ((k_base + (chunk << 2u)) / 128u));
		else
			scale_value = SparkLmDecodeE8m0(((const uint8_t *)weight_scale)[((uint64_t)neuron * (input_dimension / GROUP_SIZE)) + ((k_base + (chunk << 2u)) / GROUP_SIZE)]);
		#pragma unroll
		for (byte_index = 0; byte_index < 4u; byte_index++)
			destination[(chunk << 2u) + byte_index] = __float2bfloat16(SparkLmDecodeE4m3((packed >> (byte_index << 3u)) & 0xffu) * scale_value);
	}
}

// CONTRACT: k_base + SPARK_LM_TILE_K must not exceed input_dimension. Rows
// are bounded by slot_count and zero-filled past it, but K is NOT bounded -
// a partial trailing K tile reads past the row. Callers whose width is not a
// multiple of SPARK_LM_TILE_K must not use the tile path; see
// SparkLmHostLaunchBatchedLinear, which routes those to the scalar kernel.
static __device__ __forceinline__ void SparkLmTileStageInput(const void *input_bf16, const uint32_t *input_row_map, uint32_t slot_base, uint32_t slot_count, uint32_t k_base, uint32_t input_dimension, __nv_bfloat16 *tile)
{
	uint32_t entry,slot,source_row;
	float2 pair_value;
	for (entry = threadIdx.x; entry < (SPARK_LM_TILE * SPARK_LM_TILE_K) >> 1u; entry += blockDim.x)
	{
		slot = slot_base + (entry / (SPARK_LM_TILE_K >> 1u));
		if ( slot < slot_count )
		{
			source_row = input_row_map != 0 ? input_row_map[slot] : slot;
			pair_value = SparkLmLoadBf16Pair(input_bf16,((((uint64_t)source_row * input_dimension) + k_base) >> 1u) + (entry % (SPARK_LM_TILE_K >> 1u)));
			((__nv_bfloat162 *)tile)[entry] = __floats2bfloat162_rn(pair_value.x,pair_value.y);
		}
		else
			((__nv_bfloat162 *)tile)[entry] = __floats2bfloat162_rn(0.0f,0.0f);
	}
}

template <uint32_t GROUP_SIZE>
static __device__ __forceinline__ void SparkLmTileStageWeightAll(
    uint32_t weight_format,
    const void *weight_payload,
    const void *weight_scale,
    uint32_t neuron_base,
    uint32_t k_base,
    uint32_t input_dimension,
    uint32_t output_dimension,
    __nv_bfloat16 *tile_weight)
{
    uint32_t stage_neuron;
    uint32_t stage_k;
    uint32_t entry;

    stage_neuron =
        neuron_base + (threadIdx.x & (SPARK_LM_TILE_N - 1u));
    stage_k = (threadIdx.x >> 7u) << 5u;
    if (stage_neuron < output_dimension)
    {
        SparkLmTileDecodeRun<GROUP_SIZE>(
            weight_format,
            weight_payload,
            weight_scale,
            stage_neuron,
            k_base + stage_k,
            input_dimension,
            tile_weight +
                ((threadIdx.x & (SPARK_LM_TILE_N - 1u)) *
                 SPARK_LM_TILE_K) +
                stage_k);
    }
    else
    {
        for (entry = 0u; entry < 32u; ++entry)
        {
            tile_weight[
                ((threadIdx.x & (SPARK_LM_TILE_N - 1u)) *
                 SPARK_LM_TILE_K) +
                stage_k + entry] = __float2bfloat16(0.0f);
        }
    }
}

static __device__ __forceinline__ void SparkLmTileStageInputProducerGroup(
    const void *input_bf16,
    const uint32_t *input_row_map,
    uint32_t slot_base,
    uint32_t slot_count,
    uint32_t k_base,
    uint32_t input_dimension,
    uint32_t producer_warp_base,
    __nv_bfloat16 *tile_input)
{
    uint32_t warp_index;
    uint32_t lane_index;
    uint32_t producer_thread_index;
    uint32_t entry;

    warp_index = threadIdx.x / SPARK_LM_WARP_LANES;
    if (warp_index < producer_warp_base ||
        warp_index >= producer_warp_base + 4u)
    {
        return;
    }
    lane_index = threadIdx.x % SPARK_LM_WARP_LANES;
    producer_thread_index =
        ((warp_index - producer_warp_base) * SPARK_LM_WARP_LANES) +
        lane_index;
    for (entry = producer_thread_index;
         entry < (SPARK_LM_TILE * SPARK_LM_TILE_K) >> 1u;
         entry += 4u * SPARK_LM_WARP_LANES)
    {
        uint32_t slot;

        slot = slot_base + (entry / (SPARK_LM_TILE_K >> 1u));
        if (slot < slot_count)
        {
            float2 pair_value;
            uint32_t source_row;

            source_row = input_row_map != 0 ? input_row_map[slot] : slot;
            pair_value = SparkLmLoadBf16Pair(
                input_bf16,
                ((((uint64_t)source_row * input_dimension) + k_base) >> 1u) +
                    (entry % (SPARK_LM_TILE_K >> 1u)));
            reinterpret_cast<__nv_bfloat162 *>(tile_input)[entry] =
                __floats2bfloat162_rn(pair_value.x, pair_value.y);
        }
        else
        {
            reinterpret_cast<__nv_bfloat162 *>(tile_input)[entry] =
                __floats2bfloat162_rn(0.0f, 0.0f);
        }
    }
}

template <uint32_t GROUP_SIZE>
static __device__ __forceinline__ void SparkLmTileStageWeightProducerHalf(
    uint32_t weight_format,
    const void *weight_payload,
    const void *weight_scale,
    uint32_t neuron_base,
    uint32_t k_base,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t producer_warp_base,
    uint32_t neuron_half,
    __nv_bfloat16 *tile_weight)
{
    uint32_t warp_index;
    uint32_t lane_index;
    uint32_t producer_thread_index;
    uint32_t neuron_local;
    uint32_t stage_k;
    uint32_t stage_neuron;
    uint32_t entry;

    warp_index = threadIdx.x / SPARK_LM_WARP_LANES;
    if (warp_index < producer_warp_base ||
        warp_index >= producer_warp_base + 4u)
    {
        return;
    }
    lane_index = threadIdx.x % SPARK_LM_WARP_LANES;
    producer_thread_index =
        ((warp_index - producer_warp_base) * SPARK_LM_WARP_LANES) +
        lane_index;
    neuron_local =
        (neuron_half * (SPARK_LM_TILE_N / 2u)) +
        (producer_thread_index & ((SPARK_LM_TILE_N / 2u) - 1u));
    stage_k =
        (producer_thread_index / (SPARK_LM_TILE_N / 2u)) * 32u;
    stage_neuron = neuron_base + neuron_local;
    if (stage_neuron < output_dimension)
    {
        SparkLmTileDecodeRun<GROUP_SIZE>(
            weight_format,
            weight_payload,
            weight_scale,
            stage_neuron,
            k_base + stage_k,
            input_dimension,
            tile_weight + (neuron_local * SPARK_LM_TILE_K) + stage_k);
    }
    else
    {
        for (entry = 0u; entry < 32u; ++entry)
        {
            tile_weight[
                (neuron_local * SPARK_LM_TILE_K) + stage_k + entry] =
                __float2bfloat16(0.0f);
        }
    }
}

/*
 * Row-tiled expert GEMM on tensor cores, all eight warps computing: a
 * 16-slot by 128-neuron block accumulator, K staged 64 wide in shared.
 * Each warp owns a 16-neuron column slice; the weight stage decodes each
 * neuron's K-run ONCE into shared bf16 - thread t owns 32 contiguous
 * elements of neuron t mod 128 - and the tile is reused by all sixteen
 * gathered rows. Missing rows and neurons stage zeros; stores are
 * guarded, so any slot count and output width are served.
 */
template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC>
static __device__ void SparkLmExpertTileBodyAllWarps(
    uint32_t weight_format,
    const void *weight_payload,
    const void *weight_scale,
    const void *input_bf16,
    const uint32_t *input_row_map,
    void *output_bf16,
    uint32_t slot_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t slot_base,
    uint32_t neuron_base)
{
    __shared__ __nv_bfloat16 tile_input[
        SPARK_LM_TILE * SPARK_LM_TILE_K];
    __shared__ __nv_bfloat16 tile_weight[
        SPARK_LM_TILE_N * SPARK_LM_TILE_K];
    __shared__ float tile_output[
        SPARK_LM_TILE][SPARK_LM_TILE_N + 8u];
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        16,
        16,
        16,
        __nv_bfloat16,
        nvcuda::wmma::row_major> fragment_input;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        16,
        16,
        16,
        __nv_bfloat16,
        nvcuda::wmma::col_major> fragment_weight;
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator,
        16,
        16,
        16,
        float> fragment_accumulator;
    uint32_t warp_index;
    uint32_t k_base;
    uint32_t k_step;
    uint32_t entry;
    uint32_t slot;
    uint32_t neuron;

    warp_index = threadIdx.x / SPARK_LM_WARP_LANES;
    nvcuda::wmma::fill_fragment(fragment_accumulator, 0.0f);
    for (k_base = 0u;
         k_base < input_dimension;
         k_base += SPARK_LM_TILE_K)
    {
		if constexpr ( ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0 )
			LmActivationStageFp8Qdq<SPARK_LM_TILE,SPARK_LM_TILE_K,false,ACTIVATION_CODEC>(input_bf16,input_row_map,slot_count,slot_base,slot_count,k_base,input_dimension,tile_input,0u,SPARK_LM_CTA_WARPS);
		else
			SparkLmTileStageInput(input_bf16,input_row_map,slot_base,slot_count,k_base,input_dimension,tile_input);
        SparkLmTileStageWeightAll<GROUP_SIZE>(
            weight_format,
            weight_payload,
            weight_scale,
            neuron_base,
            k_base,
            input_dimension,
            output_dimension,
            tile_weight);
        __syncthreads();
        #pragma unroll
        for (k_step = 0u;
             k_step < SPARK_LM_TILE_K / SPARK_LM_TILE;
             ++k_step)
        {
            nvcuda::wmma::load_matrix_sync(
                fragment_input,
                tile_input + (k_step * SPARK_LM_TILE),
                SPARK_LM_TILE_K);
            nvcuda::wmma::load_matrix_sync(
                fragment_weight,
                tile_weight +
                    (warp_index * SPARK_LM_TILE * SPARK_LM_TILE_K) +
                    (k_step * SPARK_LM_TILE),
                SPARK_LM_TILE_K);
            nvcuda::wmma::mma_sync(
                fragment_accumulator,
                fragment_input,
                fragment_weight,
                fragment_accumulator);
        }
        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(
        &tile_output[0][warp_index * SPARK_LM_TILE],
        fragment_accumulator,
        SPARK_LM_TILE_N + 8u,
        nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (entry = threadIdx.x;
         entry < SPARK_LM_TILE * SPARK_LM_TILE_N;
         entry += blockDim.x)
    {
        slot = slot_base + (entry / SPARK_LM_TILE_N);
        neuron = neuron_base + (entry % SPARK_LM_TILE_N);
        if (slot < slot_count && neuron < output_dimension)
        {
            SparkLmFloatToBf16(
                output_bf16,
                ((uint64_t)slot * output_dimension) + neuron,
                tile_output[
                    entry / SPARK_LM_TILE_N][entry % SPARK_LM_TILE_N]);
        }
    }
}

template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC>
static __device__ void SparkLmExpertTileBodySoftwarePipelined(
    uint32_t weight_format,
    const void *weight_payload,
    const void *weight_scale,
    const void *input_bf16,
    const uint32_t *input_row_map,
    void *output_bf16,
    uint32_t slot_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t slot_base,
    uint32_t neuron_base)
{
    __shared__ __nv_bfloat16 tile_input[2u][
        SPARK_LM_TILE * SPARK_LM_TILE_K];
    __shared__ __nv_bfloat16 tile_weight[2u][
        SPARK_LM_TILE_N * SPARK_LM_TILE_K];
    __shared__ float tile_output[SPARK_LM_TILE][SPARK_LM_TILE_N + 8u];
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        16,
        16,
        16,
        __nv_bfloat16,
        nvcuda::wmma::row_major> frag_input;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        16,
        16,
        16,
        __nv_bfloat16,
        nvcuda::wmma::col_major> frag_weight;
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator,
        16,
        16,
        16,
        float> frag_accum;
    uint32_t warp_index;
    uint32_t current_buffer;
    uint32_t next_buffer;
    uint32_t k_base;
    uint32_t next_k_base;
    uint32_t k_step;
    uint32_t entry;
    uint32_t slot;
    uint32_t neuron;

    warp_index = threadIdx.x / SPARK_LM_WARP_LANES;
    current_buffer = 0u;
	if constexpr ( ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0 )
		LmActivationStageFp8Qdq<SPARK_LM_TILE,SPARK_LM_TILE_K,false,ACTIVATION_CODEC>(input_bf16,input_row_map,slot_count,slot_base,slot_count,0u,input_dimension,tile_input[current_buffer],0u,SPARK_LM_CTA_WARPS);
	else
		SparkLmTileStageInput(input_bf16,input_row_map,slot_base,slot_count,0u,input_dimension,tile_input[current_buffer]);
    SparkLmTileStageWeightAll<GROUP_SIZE>(
        weight_format,
        weight_payload,
        weight_scale,
        neuron_base,
        0u,
        input_dimension,
        output_dimension,
        tile_weight[current_buffer]);
    nvcuda::wmma::fill_fragment(frag_accum, 0.0f);
    __syncthreads();

    for (k_base = 0u;
         k_base < input_dimension;
         k_base += SPARK_LM_TILE_K)
    {
        next_k_base = k_base + SPARK_LM_TILE_K;
        next_buffer = current_buffer ^ 1u;

        if (warp_index < 4u)
        {
            for (k_step = 0u;
                 k_step < SPARK_LM_TILE_K / SPARK_LM_TILE;
                 ++k_step)
            {
                nvcuda::wmma::load_matrix_sync(
                    frag_input,
                    tile_input[current_buffer] +
                        (k_step * SPARK_LM_TILE),
                    SPARK_LM_TILE_K);
                nvcuda::wmma::load_matrix_sync(
                    frag_weight,
                    tile_weight[current_buffer] +
                        (warp_index * SPARK_LM_TILE * SPARK_LM_TILE_K) +
                        (k_step * SPARK_LM_TILE),
                    SPARK_LM_TILE_K);
                nvcuda::wmma::mma_sync(
                    frag_accum,
                    frag_input,
                    frag_weight,
                    frag_accum);
            }
        }
        else if (next_k_base < input_dimension)
        {
			if constexpr ( ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0 )
				LmActivationStageFp8Qdq<SPARK_LM_TILE,SPARK_LM_TILE_K,false,ACTIVATION_CODEC>(input_bf16,input_row_map,slot_count,slot_base,slot_count,next_k_base,input_dimension,tile_input[next_buffer],4u,4u);
			else
				SparkLmTileStageInputProducerGroup(input_bf16,input_row_map,slot_base,slot_count,next_k_base,input_dimension,4u,tile_input[next_buffer]);
            SparkLmTileStageWeightProducerHalf<GROUP_SIZE>(
                weight_format,
                weight_payload,
                weight_scale,
                neuron_base,
                next_k_base,
                input_dimension,
                output_dimension,
                4u,
                0u,
                tile_weight[next_buffer]);
        }
        __syncthreads();

        if (warp_index >= 4u)
        {
            for (k_step = 0u;
                 k_step < SPARK_LM_TILE_K / SPARK_LM_TILE;
                 ++k_step)
            {
                nvcuda::wmma::load_matrix_sync(
                    frag_input,
                    tile_input[current_buffer] +
                        (k_step * SPARK_LM_TILE),
                    SPARK_LM_TILE_K);
                nvcuda::wmma::load_matrix_sync(
                    frag_weight,
                    tile_weight[current_buffer] +
                        (warp_index * SPARK_LM_TILE * SPARK_LM_TILE_K) +
                        (k_step * SPARK_LM_TILE),
                    SPARK_LM_TILE_K);
                nvcuda::wmma::mma_sync(
                    frag_accum,
                    frag_input,
                    frag_weight,
                    frag_accum);
            }
        }
        else if (next_k_base < input_dimension)
        {
            SparkLmTileStageWeightProducerHalf<GROUP_SIZE>(
                weight_format,
                weight_payload,
                weight_scale,
                neuron_base,
                next_k_base,
                input_dimension,
                output_dimension,
                0u,
                1u,
                tile_weight[next_buffer]);
        }
        __syncthreads();
        current_buffer = next_buffer;
    }

    nvcuda::wmma::store_matrix_sync(
        &tile_output[0][warp_index * SPARK_LM_TILE],
        frag_accum,
        SPARK_LM_TILE_N + 8u,
        nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (entry = threadIdx.x;
         entry < SPARK_LM_TILE * SPARK_LM_TILE_N;
         entry += blockDim.x)
    {
        slot = slot_base + (entry / SPARK_LM_TILE_N);
        neuron = neuron_base + (entry % SPARK_LM_TILE_N);
        if (slot < slot_count && neuron < output_dimension)
        {
            SparkLmFloatToBf16(
                output_bf16,
                ((uint64_t)slot * output_dimension) + neuron,
                tile_output[
                    entry / SPARK_LM_TILE_N][entry % SPARK_LM_TILE_N]);
        }
    }
}

/* Dense tile with an inner M-group loop: one CTA owns M_GROUP m-tiles of
 * one n-tile and stages each k-stage's weight strip ONCE, reusing it for
 * every m-tile's MMA. The plain grid stages the same weight strip once per
 * m-tile, so B=256 (16 m-blocks) re-reads a dense weight 16x; this cuts
 * the amplification to ceil(m_blocks/M_GROUP). BF16 only - the qwen38
 * dense spine; scaled formats keep the original grid. Caller: qwen38's
 * SparkQwen38LaunchLinear via SparkLmHostLaunchBatchedLinearMloop. */
#define SPARK_LM_MLOOP_GROUP 8u
template <uint32_t GROUP_SIZE>
static __global__ void SparkLmExpertTileMloopKernel(const void *weight_payload, const void *input_bf16, void *output_bf16, uint32_t slot_count, uint32_t input_dimension, uint32_t output_dimension)
{
    __shared__ __nv_bfloat16 tile_input[2u][
        SPARK_LM_MLOOP_GROUP * SPARK_LM_TILE * SPARK_LM_TILE_K];
    __shared__ __nv_bfloat16 tile_weight[2u][
        SPARK_LM_TILE_N * SPARK_LM_TILE_K];
    __shared__ float tile_output[SPARK_LM_TILE][SPARK_LM_TILE_N + 8u];
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        16,
        16,
        16,
        __nv_bfloat16,
        nvcuda::wmma::row_major> frag_input;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        16,
        16,
        16,
        __nv_bfloat16,
        nvcuda::wmma::col_major> frag_weight;
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator,
        16,
        16,
        16,
        float> frag_accum[SPARK_LM_MLOOP_GROUP];
    uint32_t warp_index = threadIdx.x / SPARK_LM_WARP_LANES;
    uint32_t slot_base = blockIdx.x * (SPARK_LM_MLOOP_GROUP * SPARK_LM_TILE);
    uint32_t neuron_base = blockIdx.y * SPARK_LM_TILE_N;
    uint32_t valid_m,row_limit,m,entry,row,k_base,k_step,next_k_base;
    uint32_t current_buffer = 0u,next_buffer,slot,neuron;
    if (slot_base >= slot_count)
        return;
    row_limit = slot_count - slot_base;
    valid_m = (row_limit + SPARK_LM_TILE - 1u) / SPARK_LM_TILE;
    if (valid_m > SPARK_LM_MLOOP_GROUP)
        valid_m = SPARK_LM_MLOOP_GROUP;
    /* Prologue: stage every valid m-tile's first input strip and the
     * weight strip for k=0. Invalid tiles stage zeros so their accum
     * fragments stay zero and the epilogue's slot bound drops them. */
    for (m = 0u; m < valid_m; ++m)
        for (entry = threadIdx.x; entry < SPARK_LM_TILE * SPARK_LM_TILE_K; entry += blockDim.x)
        {
            row = slot_base + (m * SPARK_LM_TILE) + (entry / SPARK_LM_TILE_K);
            k_base = entry % SPARK_LM_TILE_K;
            tile_input[0u][(m * SPARK_LM_TILE * SPARK_LM_TILE_K) + entry] =
                __float2bfloat16(SparkLmBf16ToFloat(input_bf16,((uint64_t)row * input_dimension) + k_base));
        }
    for (m = valid_m; m < SPARK_LM_MLOOP_GROUP; ++m)
        for (entry = threadIdx.x; entry < SPARK_LM_TILE * SPARK_LM_TILE_K; entry += blockDim.x)
            tile_input[0u][(m * SPARK_LM_TILE * SPARK_LM_TILE_K) + entry] = __float2bfloat16(0.0f);
    SparkLmTileStageWeightAll<GROUP_SIZE>(
        SPARK_LM_WEIGHT_FORMAT_BF16,
        weight_payload,
        0,
        neuron_base,
        0u,
        input_dimension,
        output_dimension,
        tile_weight[0u]);
    for (m = 0u; m < SPARK_LM_MLOOP_GROUP; ++m)
        nvcuda::wmma::fill_fragment(frag_accum[m],0.0f);
    __syncthreads();

    for (k_base = 0u; k_base < input_dimension; k_base += SPARK_LM_TILE_K)
    {
        next_k_base = k_base + SPARK_LM_TILE_K;
        next_buffer = current_buffer ^ 1u;

        if (warp_index < 4u)
        {
            for (m = 0u; m < SPARK_LM_MLOOP_GROUP; ++m)
                for (k_step = 0u; k_step < SPARK_LM_TILE_K / SPARK_LM_TILE; ++k_step)
                {
                    nvcuda::wmma::load_matrix_sync(
                        frag_input,
                        tile_input[current_buffer] +
                            (m * SPARK_LM_TILE * SPARK_LM_TILE_K) +
                            (k_step * SPARK_LM_TILE),
                        SPARK_LM_TILE_K);
                    nvcuda::wmma::load_matrix_sync(
                        frag_weight,
                        tile_weight[current_buffer] +
                            (warp_index * SPARK_LM_TILE * SPARK_LM_TILE_K) +
                            (k_step * SPARK_LM_TILE),
                        SPARK_LM_TILE_K);
                    nvcuda::wmma::mma_sync(
                        frag_accum[m],
                        frag_input,
                        frag_weight,
                        frag_accum[m]);
                }
        }
        else if (next_k_base < input_dimension)
        {
            for (m = 0u; m < valid_m; ++m)
                for (entry = threadIdx.x; entry < SPARK_LM_TILE * SPARK_LM_TILE_K; entry += blockDim.x)
                {
                    row = slot_base + (m * SPARK_LM_TILE) + (entry / SPARK_LM_TILE_K);
                    k_step = entry % SPARK_LM_TILE_K;
                    tile_input[next_buffer][(m * SPARK_LM_TILE * SPARK_LM_TILE_K) + entry] =
                        __float2bfloat16(SparkLmBf16ToFloat(input_bf16,((uint64_t)row * input_dimension) + next_k_base + k_step));
                }
            SparkLmTileStageWeightProducerHalf<GROUP_SIZE>(
                SPARK_LM_WEIGHT_FORMAT_BF16,
                weight_payload,
                0,
                neuron_base,
                next_k_base,
                input_dimension,
                output_dimension,
                4u,
                0u,
                tile_weight[next_buffer]);
        }
        __syncthreads();

        if (warp_index >= 4u)
        {
            for (m = 0u; m < SPARK_LM_MLOOP_GROUP; ++m)
                for (k_step = 0u; k_step < SPARK_LM_TILE_K / SPARK_LM_TILE; ++k_step)
                {
                    nvcuda::wmma::load_matrix_sync(
                        frag_input,
                        tile_input[current_buffer] +
                            (m * SPARK_LM_TILE * SPARK_LM_TILE_K) +
                            (k_step * SPARK_LM_TILE),
                        SPARK_LM_TILE_K);
                    nvcuda::wmma::load_matrix_sync(
                        frag_weight,
                        tile_weight[current_buffer] +
                            (warp_index * SPARK_LM_TILE * SPARK_LM_TILE_K) +
                            (k_step * SPARK_LM_TILE),
                        SPARK_LM_TILE_K);
                    nvcuda::wmma::mma_sync(
                        frag_accum[m],
                        frag_input,
                        frag_weight,
                        frag_accum[m]);
                }
        }
        else if (next_k_base < input_dimension)
        {
            SparkLmTileStageWeightProducerHalf<GROUP_SIZE>(
                SPARK_LM_WEIGHT_FORMAT_BF16,
                weight_payload,
                0,
                neuron_base,
                next_k_base,
                input_dimension,
                output_dimension,
                0u,
                1u,
                tile_weight[next_buffer]);
        }
        __syncthreads();
        current_buffer = next_buffer;
    }

    for (m = 0u; m < valid_m; ++m)
    {
        nvcuda::wmma::store_matrix_sync(
            &tile_output[0][warp_index * SPARK_LM_TILE],
            frag_accum[m],
            SPARK_LM_TILE_N + 8u,
            nvcuda::wmma::mem_row_major);
        __syncthreads();
        for (entry = threadIdx.x;
             entry < SPARK_LM_TILE * SPARK_LM_TILE_N;
             entry += blockDim.x)
        {
            slot = slot_base + (m * SPARK_LM_TILE) + (entry / SPARK_LM_TILE_N);
            neuron = neuron_base + (entry % SPARK_LM_TILE_N);
            if (slot < slot_count && neuron < output_dimension)
                SparkLmFloatToBf16(
                    output_bf16,
                    ((uint64_t)slot * output_dimension) + neuron,
                    tile_output[entry / SPARK_LM_TILE_N][entry % SPARK_LM_TILE_N]);
        }
        __syncthreads();
    }
}

/* Grouped-expert m-loop: grid.x is ONE (n-tiles x experts only), each CTA
 * walks its expert's row group in chunks of M_GROUP m-tiles, and each
 * k-stage's weight strip is staged ONCE and shared across the chunk. The
 * plain grid (m_blocks x n_tiles x experts) launched ~122K empty CTAs at
 * B=256 (m-tiles beyond a group of ~5 rows) at ~1.25 us of launch/retire
 * each - the measured 233 GB/s collapse. Caller: qwen38 grouped FP8
 * experts via SparkLmHostLaunchGroupedExpertTileMloop. */
template <uint32_t GROUP_SIZE>
static __global__ void SparkLmExpertTileAllMloopKernel(uint32_t weight_format, const void *payload_base, const void *scale_base, uint64_t payload_expert_stride_bytes, uint64_t scale_expert_stride_bytes, const void *input_bf16, const uint32_t *grouped_rows, const uint32_t *expert_offsets, void *output_bf16, uint32_t input_dimension, uint32_t output_dimension, uint32_t expert_count)
{
    /* Expert-stride loop: one CTA walks several experts so the launch uses
     * a fixed CTA budget instead of 512 experts x n_tiles whose empty CTAs
     * dominate small batches (measured ~1.25 us each). */
    uint32_t expert;
    for (expert = blockIdx.z; expert < expert_count; expert += gridDim.z)
    {
    const uint32_t M_GROUP = 8u;
    __shared__ __nv_bfloat16 tile_input[2u][
        M_GROUP * SPARK_LM_TILE * SPARK_LM_TILE_K];
    __shared__ __nv_bfloat16 tile_weight[2u][
        SPARK_LM_TILE_N * SPARK_LM_TILE_K];
    __shared__ float tile_output[SPARK_LM_TILE][SPARK_LM_TILE_N + 8u];
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        16,
        16,
        16,
        __nv_bfloat16,
        nvcuda::wmma::row_major> frag_input;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        16,
        16,
        16,
        __nv_bfloat16,
        nvcuda::wmma::col_major> frag_weight;
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator,
        16,
        16,
        16,
        float> frag_accum[M_GROUP];
    uint32_t warp_index = threadIdx.x / SPARK_LM_WARP_LANES;
    uint32_t neuron_base = blockIdx.y * SPARK_LM_TILE_N;
    uint32_t offset,count,chunk_base,chunk_m,m,row_limit,entry,row,k_base,k_step,next_k_base;
    uint32_t current_buffer,next_buffer,slot,neuron;
    const uint8_t *payload;
    const uint8_t *scale;
    const uint32_t *row_map;
    void *output;
    offset = expert_offsets[expert];
    count = expert_offsets[expert + 1u] - offset;
    if (count == 0u)
        continue;
    payload = (const uint8_t *)payload_base + ((uint64_t)expert * payload_expert_stride_bytes);
    scale = (const uint8_t *)scale_base + ((uint64_t)expert * scale_expert_stride_bytes);
    row_map = grouped_rows != 0 ? grouped_rows + offset : 0;
    output = (void *)((uint8_t *)output_bf16 + ((uint64_t)offset * output_dimension * 2u));
    current_buffer = 0u;
    for (chunk_base = 0u; chunk_base < count; chunk_base += M_GROUP * SPARK_LM_TILE)
    {
        chunk_m = (count - chunk_base + SPARK_LM_TILE - 1u) / SPARK_LM_TILE;
        if (chunk_m > M_GROUP)
            chunk_m = M_GROUP;
        /* Prologue: stage the chunk's input strips and the k=0 weight. */
        for (m = 0u; m < chunk_m; ++m)
        {
            row_limit = count - (chunk_base + (m * SPARK_LM_TILE));
            for (entry = threadIdx.x; entry < SPARK_LM_TILE * SPARK_LM_TILE_K; entry += blockDim.x)
            {
                row = (entry / SPARK_LM_TILE_K) + (m * SPARK_LM_TILE);
                k_base = entry % SPARK_LM_TILE_K;
                if (row < row_limit)
                    tile_input[0u][(m * SPARK_LM_TILE * SPARK_LM_TILE_K) + entry] =
                        __float2bfloat16(SparkLmBf16ToFloat(input_bf16,((uint64_t)(row_map != 0 ? row_map[chunk_base + row] : (offset + chunk_base + row)) * input_dimension) + k_base));
                else
                    tile_input[0u][(m * SPARK_LM_TILE * SPARK_LM_TILE_K) + entry] = __float2bfloat16(0.0f);
            }
        }
        for (m = chunk_m; m < M_GROUP; ++m)
            for (entry = threadIdx.x; entry < SPARK_LM_TILE * SPARK_LM_TILE_K; entry += blockDim.x)
                tile_input[0u][(m * SPARK_LM_TILE * SPARK_LM_TILE_K) + entry] = __float2bfloat16(0.0f);
        SparkLmTileStageWeightAll<GROUP_SIZE>(
            weight_format,payload,scale,neuron_base,0u,input_dimension,
            output_dimension,tile_weight[0u]);
        for (m = 0u; m < M_GROUP; ++m)
            nvcuda::wmma::fill_fragment(frag_accum[m],0.0f);
        __syncthreads();

        for (k_base = 0u; k_base < input_dimension; k_base += SPARK_LM_TILE_K)
        {
            next_k_base = k_base + SPARK_LM_TILE_K;
            next_buffer = current_buffer ^ 1u;

            if (warp_index < 4u)
            {
                for (m = 0u; m < chunk_m; ++m)
                    for (k_step = 0u; k_step < SPARK_LM_TILE_K / SPARK_LM_TILE; ++k_step)
                    {
                        nvcuda::wmma::load_matrix_sync(
                            frag_input,
                            tile_input[current_buffer] +
                                (m * SPARK_LM_TILE * SPARK_LM_TILE_K) +
                                (k_step * SPARK_LM_TILE),
                            SPARK_LM_TILE_K);
                        nvcuda::wmma::load_matrix_sync(
                            frag_weight,
                            tile_weight[current_buffer] +
                                (warp_index * SPARK_LM_TILE * SPARK_LM_TILE_K) +
                                (k_step * SPARK_LM_TILE),
                            SPARK_LM_TILE_K);
                        nvcuda::wmma::mma_sync(
                            frag_accum[m],frag_input,frag_weight,frag_accum[m]);
                    }
            }
            else if (next_k_base < input_dimension)
            {
                for (m = 0u; m < chunk_m; ++m)
                {
                    row_limit = count - (chunk_base + (m * SPARK_LM_TILE));
                    for (entry = threadIdx.x; entry < SPARK_LM_TILE * SPARK_LM_TILE_K; entry += blockDim.x)
                    {
                        row = (entry / SPARK_LM_TILE_K) + (m * SPARK_LM_TILE);
                        k_step = entry % SPARK_LM_TILE_K;
                        if (row < row_limit)
                            tile_input[next_buffer][(m * SPARK_LM_TILE * SPARK_LM_TILE_K) + entry] =
                                __float2bfloat16(SparkLmBf16ToFloat(input_bf16,((uint64_t)(row_map != 0 ? row_map[chunk_base + row] : (offset + chunk_base + row)) * input_dimension) + next_k_base + k_step));
                    }
                }
                SparkLmTileStageWeightProducerHalf<GROUP_SIZE>(
                    weight_format,payload,scale,neuron_base,next_k_base,
                    input_dimension,output_dimension,4u,0u,
                    tile_weight[next_buffer]);
            }
            __syncthreads();

            if (warp_index >= 4u)
            {
                for (m = 0u; m < chunk_m; ++m)
                    for (k_step = 0u; k_step < SPARK_LM_TILE_K / SPARK_LM_TILE; ++k_step)
                    {
                        nvcuda::wmma::load_matrix_sync(
                            frag_input,
                            tile_input[current_buffer] +
                                (m * SPARK_LM_TILE * SPARK_LM_TILE_K) +
                                (k_step * SPARK_LM_TILE),
                            SPARK_LM_TILE_K);
                        nvcuda::wmma::load_matrix_sync(
                            frag_weight,
                            tile_weight[current_buffer] +
                                (warp_index * SPARK_LM_TILE * SPARK_LM_TILE_K) +
                                (k_step * SPARK_LM_TILE),
                            SPARK_LM_TILE_K);
                        nvcuda::wmma::mma_sync(
                            frag_accum[m],frag_input,frag_weight,frag_accum[m]);
                    }
            }
            else if (next_k_base < input_dimension)
            {
                SparkLmTileStageWeightProducerHalf<GROUP_SIZE>(
                    weight_format,payload,scale,neuron_base,next_k_base,
                    input_dimension,output_dimension,0u,1u,
                    tile_weight[next_buffer]);
            }
            __syncthreads();
            current_buffer = next_buffer;
        }

        for (m = 0u; m < chunk_m; ++m)
        {
            nvcuda::wmma::store_matrix_sync(
                &tile_output[0][warp_index * SPARK_LM_TILE],
                frag_accum[m],
                SPARK_LM_TILE_N + 8u,
                nvcuda::wmma::mem_row_major);
            __syncthreads();
            row_limit = count - (chunk_base + (m * SPARK_LM_TILE));
            for (entry = threadIdx.x; entry < SPARK_LM_TILE * SPARK_LM_TILE_N; entry += blockDim.x)
            {
                row = entry / SPARK_LM_TILE_N;
                slot = chunk_base + (m * SPARK_LM_TILE) + row;
                neuron = neuron_base + (entry % SPARK_LM_TILE_N);
                if (row < row_limit && neuron < output_dimension)
                    SparkLmFloatToBf16(
                        output,
                        ((uint64_t)slot * output_dimension) + neuron,
                        tile_output[row][entry % SPARK_LM_TILE_N]);
            }
            __syncthreads();
        }
    }
    }
}

// Body dispatch keeps independently selectable all-warp and software-pipelined
// BF16 schedules, chosen by SPARK_LM_EXPERT_TILE_POLICY. Both are live: the
// AUTOMATIC default selects between them at runtime on input_dimension, so
// neither can be removed without a measurement showing one dominates.
//
// The direct-FP8 branch that used to sit here selected spark_lm_fp8_tile.cuh
// under SPARK_LM_FP8_TILE. That macro was defined nowhere in the tree, so the
// branch was unreachable and the header it included was never compiled. Both
// are removed; spark_lm_group_gemm.cuh is the FP8 path, with its fragment
// mapping verified against CUTLASS rather than hand-derived.
template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC=SPARK_ACTIVATION_CODEC_NONE>
static __device__ void SparkLmExpertTileDispatch(uint32_t weight_format, const void *weight_payload, const void *weight_scale, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count, uint32_t input_dimension, uint32_t output_dimension, uint32_t slot_base, uint32_t neuron_base)
{
#if SPARK_LM_EXPERT_TILE_POLICY == SPARK_LM_EXPERT_TILE_POLICY_ALL_WARPS
    SparkLmExpertTileBodyAllWarps<GROUP_SIZE,ACTIVATION_CODEC>(
        weight_format,
        weight_payload,
        weight_scale,
        input_bf16,
        input_row_map,
        output_bf16,
        slot_count,
        input_dimension,
        output_dimension,
        slot_base,
        neuron_base);
#elif SPARK_LM_EXPERT_TILE_POLICY == SPARK_LM_EXPERT_TILE_POLICY_SOFTWARE_PIPELINED
    SparkLmExpertTileBodySoftwarePipelined<GROUP_SIZE,ACTIVATION_CODEC>(
        weight_format,
        weight_payload,
        weight_scale,
        input_bf16,
        input_row_map,
        output_bf16,
        slot_count,
        input_dimension,
        output_dimension,
        slot_base,
        neuron_base);
#else
    if (input_dimension <= SPARK_LM_TILE_K)
    {
        SparkLmExpertTileBodyAllWarps<GROUP_SIZE,ACTIVATION_CODEC>(
            weight_format,
            weight_payload,
            weight_scale,
            input_bf16,
            input_row_map,
            output_bf16,
            slot_count,
            input_dimension,
            output_dimension,
            slot_base,
            neuron_base);
    }
    else
    {
        SparkLmExpertTileBodySoftwarePipelined<GROUP_SIZE,ACTIVATION_CODEC>(
            weight_format,
            weight_payload,
            weight_scale,
            input_bf16,
            input_row_map,
            output_bf16,
            slot_count,
            input_dimension,
            output_dimension,
            slot_base,
            neuron_base);
    }
#endif
}

template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC=SPARK_ACTIVATION_CODEC_NONE>
static __global__ void SparkLmExpertTileKernel(uint32_t weight_format, const void *weight_payload, const void *weight_scale, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count, uint32_t input_dimension, uint32_t output_dimension)
{
	SparkLmExpertTileDispatch<GROUP_SIZE,ACTIVATION_CODEC>(weight_format,weight_payload,weight_scale,input_bf16,input_row_map,output_bf16,slot_count,input_dimension,output_dimension,blockIdx.x * SPARK_LM_TILE,blockIdx.y * SPARK_LM_TILE_N);
}

/*
 * One-row grouped expert tile: the tensor-core path is deliberately M=16,
 * so a B1 group still pads fifteen rows and performs the same weight tile
 * work.  This path keeps the route grouping and the device prefix, but makes
 * one CTA own a complete 128-neuron tile.  It stages the source activation
 * once, then has its eight warps walk the sixteen scalar neurons in that tile.
 * The result is exact with the scalar linear arithmetic and avoids reloading
 * the same activation once per eight-neuron sub-tile.
 */
static __device__ __forceinline__ uint32_t SparkLmGroupedScalarGroupOfTile(const uint32_t *group_tile_prefix, uint32_t group_count, uint32_t tile_index)
{
	uint32_t low = 0u,high = group_count,middle;
	while ( low + 1u < high )
	{
		middle = (low + high) >> 1u;
		if ( group_tile_prefix[middle] <= tile_index )
			low = middle;
		else
			high = middle;
	}
	return(low);
}

template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC=SPARK_ACTIVATION_CODEC_NONE>
static __global__ void SparkLmGroupedScalarLinearKernel(uint32_t weight_format, const void *payload_base, const uint8_t *scale_base, uint64_t payload_group_stride_bytes, uint64_t scale_group_stride_bytes, const void *input_bf16, const uint32_t *source_row_map, uint32_t source_row_count, const uint32_t *group_row_offset, const uint32_t *group_tile_prefix, void *output_bf16, uint32_t group_count, uint32_t input_dimension, uint32_t output_dimension)
{
	extern __shared__ float shared_input[];
	const uint32_t subtile_count = (SPARK_LM_TILE_N + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS;
	uint32_t task,group,in_group,neuron_tiles,row_tile,neuron_tile,row_base,row_limit,local_row,row,source_row,element,subtile,neuron,warp,lane;
	const void *group_payload;
	const uint8_t *group_scale;
	float accumulator;
	for (task = blockIdx.x; task < group_tile_prefix[group_count]; task += gridDim.x)
	{
		group = SparkLmGroupedScalarGroupOfTile(group_tile_prefix,group_count,task);
		in_group = task - group_tile_prefix[group];
		neuron_tiles = (output_dimension + SPARK_LM_TILE_N - 1u) / SPARK_LM_TILE_N;
		row_tile = in_group / neuron_tiles;
		neuron_tile = in_group % neuron_tiles;
		row_base = group_row_offset[group] + row_tile * SPARK_LM_TILE;
		row_limit = group_row_offset[group + 1u];
		group_payload = (const uint8_t *)payload_base + ((uint64_t)group * payload_group_stride_bytes);
		group_scale = scale_base != 0 ? scale_base + ((uint64_t)group * scale_group_stride_bytes) : 0;
		for (local_row = 0u; local_row < SPARK_LM_TILE && row_base + local_row < row_limit; local_row++)
		{
			row = row_base + local_row;
			source_row = source_row_map != 0 ? source_row_map[row] : row;
			if ( source_row >= source_row_count )
			{
				asm volatile("trap;\n");
				return;
			}
			for (element = threadIdx.x; element < input_dimension; element += blockDim.x)
				shared_input[element] = SparkLmBf16ToFloat(input_bf16,((uint64_t)source_row * input_dimension) + element);
			__syncthreads();
			if constexpr ( ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0 )
			{
				LmActivationFp8QdqFloatRow<ACTIVATION_CODEC>(shared_input,input_dimension);
				__syncthreads();
			}
			warp = threadIdx.x / SPARK_LM_WARP_LANES;
			lane = threadIdx.x % SPARK_LM_WARP_LANES;
			for (subtile = 0u; subtile < subtile_count; subtile++)
			{
				neuron = (neuron_tile * SPARK_LM_TILE_N) + (subtile * SPARK_LM_CTA_WARPS) + warp;
				if ( neuron < output_dimension )
				{
					if ( weight_format == SPARK_LM_WEIGHT_FORMAT_BF16 )
						accumulator = SparkLmDotRowBf16(shared_input,group_payload,neuron,input_dimension,lane);
					else if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 )
						accumulator = SparkLmDotRowFp8<GROUP_SIZE>(shared_input,group_payload,group_scale,neuron,input_dimension,lane);
					else if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
						accumulator = SparkLmDotRowFp8F32<128u>(shared_input,group_payload,(const float *)group_scale,neuron,input_dimension,lane);
					else
						accumulator = SparkLmDotRowMxfp4<GROUP_SIZE>(shared_input,group_payload,group_scale,neuron,input_dimension,lane);
					accumulator = SparkLmWarpReduceSum(accumulator);
					if ( lane == 0u )
						SparkLmFloatToBf16(output_bf16,((uint64_t)row * output_dimension) + neuron,accumulator);
				}
			}
			__syncthreads();
		}
	}
}

/*
 * All-expert tile: gridDim.z spans the routed expert table, the group
 * offsets live on DEVICE, and empty or out-of-range tiles exit in a few
 * cycles - the whole routed w1/w3/w2 phase becomes ONE launch with no
 * host knowledge of the grouping. Identity mapping (row_map zero) shifts
 * the input base by the group offset so the w2 shape works unchanged.
 * gridDim.x MUST cover the worst-case group - the FULL pair count, not
 * the row count: hash-routed layers can send several of one row's ranks
 * to the same expert, so a group is bounded only by the pair total.
 */
template <uint32_t GROUP_SIZE>
static __global__ void SparkLmExpertTileAllKernel(uint32_t weight_format, const void *payload_base, const void *scale_base, uint64_t payload_expert_stride_bytes, uint64_t scale_expert_stride_bytes, const void *input_bf16, const uint32_t *grouped_rows, const uint32_t *expert_offsets, void *output_bf16, uint32_t input_dimension, uint32_t output_dimension)
{
	uint32_t expert = blockIdx.z;
	uint32_t offset = expert_offsets[expert],count = expert_offsets[expert + 1u] - offset;
	const void *payload = (const uint8_t *)payload_base + ((uint64_t)expert * payload_expert_stride_bytes);
	const void *scale = (const uint8_t *)scale_base + ((uint64_t)expert * scale_expert_stride_bytes);
	const void *input = grouped_rows != 0 ? input_bf16 : (const void *)((const uint8_t *)input_bf16 + ((uint64_t)offset * input_dimension * 2u));
	const uint32_t *row_map = grouped_rows != 0 ? grouped_rows + offset : 0;
	void *output = (void *)((uint8_t *)output_bf16 + ((uint64_t)offset * output_dimension * 2u));
	if ( (blockIdx.x * SPARK_LM_TILE) >= count )
		return;
	SparkLmExpertTileDispatch<GROUP_SIZE>(weight_format,payload,scale,input,row_map,output,count,input_dimension,output_dimension,blockIdx.x * SPARK_LM_TILE,blockIdx.y * SPARK_LM_TILE_N);
}

/*
 * Device grouping: ONE single-block kernel replaces the per-layer host
 * round trip - histogram, exclusive prefix, and an atomic scatter whose
 * inverse map restores canonical route order for the pair reduce. Indices
 * come from the driver's
 * own gate select and are in range by construction. Kills the stream
 * synchronize every MoE layer paid and makes the step graph-capturable.
 */
#define SPARK_LM_MOE_MAX_EXPERTS 1024u

static __global__ void SparkLmMoeGroupKernel(const uint32_t *pair_expert_ids, uint32_t pair_count, uint32_t expert_count, uint32_t experts_per_token, uint32_t *expert_offsets, uint32_t *grouped_rows, uint32_t *grouped_weight_slots, uint32_t *inverse_map)
{
	__shared__ uint32_t counts[SPARK_LM_MOE_MAX_EXPERTS];
	__shared__ uint32_t cursors[SPARK_LM_MOE_MAX_EXPERTS];
	uint32_t pair,expert,slot,running;
	for (expert = threadIdx.x; expert < expert_count; expert += blockDim.x)
		counts[expert] = 0u;
	__syncthreads();
	for (pair = threadIdx.x; pair < pair_count; pair += blockDim.x)
		atomicAdd(&counts[pair_expert_ids[pair]],1u);
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		running = 0u;
		for (expert = 0; expert < expert_count; expert++)
		{
			cursors[expert] = running;
			expert_offsets[expert] = running;
			running += counts[expert];
		}
		expert_offsets[expert_count] = running;
	}
	__syncthreads();
	for (pair = threadIdx.x; pair < pair_count; pair += blockDim.x)
	{
		slot = atomicAdd(&cursors[pair_expert_ids[pair]],1u);
		grouped_rows[slot] = pair / experts_per_token;
		grouped_weight_slots[slot] = pair;
		inverse_map[pair] = slot;
	}
}

// Race-free replacement for the per-group output scatter: every pair's
// expert output sits at its grouped slot, so one block per row sums the
// row's contributions through the inverse map and accumulates once.
#define SPARK_LM_MOE_MAX_TOPK 16u

static __global__ void SparkLmMoePairReduceKernel(const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint32_t row_count, uint32_t experts_per_token, uint32_t width)
{
	uint32_t row = blockIdx.x,element,rank;
	uint64_t pair_base = (uint64_t)row * experts_per_token;
	uint64_t rank_pair_base[SPARK_LM_MOE_MAX_TOPK];
	float rank_weight[SPARK_LM_MOE_MAX_TOPK];
	float2 pair_value,accum_pair;
	if ( row >= row_count )
		return;
	for (rank = 0; rank < experts_per_token; rank++)
	{
		rank_pair_base[rank] = ((uint64_t)__ldg(inverse_map + pair_base + rank) * width) >> 1u;
		rank_weight[rank] = pair_weights_f32 != 0 ? __ldg(pair_weights_f32 + pair_base + rank) : 1.0f;
	}
	for (element = threadIdx.x; element < (width >> 1u); element += blockDim.x)
	{
		accum_pair = SparkLmLoadBf16Pair(accum_bf16,(((uint64_t)row * width) >> 1u) + element);
		for (rank = 0; rank < experts_per_token; rank++)
		{
			pair_value = SparkLmLoadBf16Pair(slot_out_bf16,rank_pair_base[rank] + element);
			accum_pair.x = fmaf(rank_weight[rank],pair_value.x,accum_pair.x);
			accum_pair.y = fmaf(rank_weight[rank],pair_value.y,accum_pair.y);
		}
		SparkLmStoreBf16Pair(accum_bf16,(((uint64_t)row * width) >> 1u) + element,accum_pair.x,accum_pair.y);
	}
}

/*
 * TP rank shards occupy a column slice of a full hidden row.  Keeping the
 * destination base, full row stride, and rank-column offset separate prevents
 * B>1 from treating adjacent shards of row zero as subsequent rows.  Routed
 * expert outputs remain packed at the local shard width; only the destination
 * accumulation is strided.
 */
static __global__ void SparkLmMoePairReduceStridedKernel(
	const void *slot_out_bf16,
	const uint32_t *inverse_map,
	const float *pair_weights_f32,
	void *accum_bf16,
	uint64_t accum_row_stride,
	uint32_t accum_offset,
	uint32_t row_count,
	uint32_t experts_per_token,
	uint32_t width)
{
	uint32_t row = blockIdx.x,element,rank;
	uint64_t pair_base = (uint64_t)row * experts_per_token;
	uint64_t accum_pair_base =
		((uint64_t)row * accum_row_stride + accum_offset) >> 1u;
	uint64_t rank_pair_base[SPARK_LM_MOE_MAX_TOPK];
	float rank_weight[SPARK_LM_MOE_MAX_TOPK];
	float2 pair_value,accum_pair;
	if ( row >= row_count )
		return;
	for (rank = 0u; rank < experts_per_token; ++rank)
	{
		rank_pair_base[rank] =
			((uint64_t)__ldg(inverse_map + pair_base + rank) * width) >> 1u;
		rank_weight[rank] = pair_weights_f32 != 0
			? __ldg(pair_weights_f32 + pair_base + rank) : 1.0f;
	}
	for (element = threadIdx.x; element < (width >> 1u);
		element += blockDim.x)
	{
		accum_pair = SparkLmLoadBf16Pair(accum_bf16,
			accum_pair_base + element);
		for (rank = 0u; rank < experts_per_token; ++rank)
		{
			pair_value = SparkLmLoadBf16Pair(slot_out_bf16,
				rank_pair_base[rank] + element);
			accum_pair.x = fmaf(rank_weight[rank],pair_value.x,accum_pair.x);
			accum_pair.y = fmaf(rank_weight[rank],pair_value.y,accum_pair.y);
		}
		SparkLmStoreBf16Pair(accum_bf16,accum_pair_base + element,
			accum_pair.x,accum_pair.y);
	}
}

static __global__ void SparkLmMoePairReduceOverwriteKernel(const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *output_bf16, uint32_t row_count, uint32_t experts_per_token, uint32_t width)
{
    uint32_t row = blockIdx.x;
    uint32_t element;
    uint32_t rank;
    uint64_t pair_base = (uint64_t)row * experts_per_token;
    uint64_t rank_pair_base[SPARK_LM_MOE_MAX_TOPK];
    float rank_weight[SPARK_LM_MOE_MAX_TOPK];
    float2 pair_value;
    float2 output_pair;

    if (row >= row_count)
    {
        return;
    }
    for (rank = 0u; rank < experts_per_token; ++rank)
    {
        rank_pair_base[rank] =
            ((uint64_t)__ldg(inverse_map + pair_base + rank) * width) >> 1u;
        rank_weight[rank] = pair_weights_f32 != 0
            ? __ldg(pair_weights_f32 + pair_base + rank)
            : 1.0f;
    }
    for (element = threadIdx.x; element < (width >> 1u); element += blockDim.x)
    {
        output_pair.x = 0.0f;
        output_pair.y = 0.0f;
        for (rank = 0u; rank < experts_per_token; ++rank)
        {
            pair_value = SparkLmLoadBf16Pair(
                slot_out_bf16,
                rank_pair_base[rank] + element);
            output_pair.x = fmaf(rank_weight[rank], pair_value.x, output_pair.x);
            output_pair.y = fmaf(rank_weight[rank], pair_value.y, output_pair.y);
        }
        SparkLmStoreBf16Pair(
            output_bf16,
            (((uint64_t)row * width) >> 1u) + element,
            output_pair.x,
            output_pair.y);
    }
}




/*
 * Fused LM head: matvec against the row's hidden and a running argmax, no
 * logits tensor ever materialized. One block per row; each warp owns a
 * stripe of candidates, keeps its running best in registers, and the block
 * reduces bests through shared memory. token_ids may be null, in which case
 * the winning candidate INDEX is written (dense-vocab head); non-null maps
 * through a restricted-vocabulary id table.
 */
static __global__ void SparkLmHeadArgmaxKernel(const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t hidden_dimension, uint32_t candidate_count)
{
	__shared__ float best_score[SPARK_LM_CTA_WARPS];
	__shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS];
	uint32_t row = blockIdx.x;
	uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES,lane = threadIdx.x % SPARK_LM_WARP_LANES;
	uint64_t row_offset;
	uint32_t candidate,element,winner;
	float accumulator,warp_best_score;
	uint32_t warp_best_candidate;
	if ( row >= row_count )
		return;
	row_offset = (uint64_t)row * (uint64_t)hidden_dimension;
	warp_best_score = -3.0e38f;
	warp_best_candidate = 0u;
	for (candidate = warp; candidate < candidate_count; candidate += SPARK_LM_CTA_WARPS)
	{
		accumulator = 0.0f;
		for (element = lane; element < hidden_dimension; element += SPARK_LM_WARP_LANES)
			accumulator += (SparkLmBf16ToFloat(hidden_bf16,row_offset + element) * SparkLmBf16ToFloat(head_weight_bf16,((uint64_t)candidate * hidden_dimension) + element));
		accumulator = SparkLmWarpReduceSum(accumulator);
		accumulator = __shfl_sync(0xffffffffu,accumulator,0);
		if ( accumulator > warp_best_score || (accumulator == warp_best_score && candidate < warp_best_candidate) )
		{
			warp_best_score = accumulator;
			warp_best_candidate = candidate;
		}
	}
	if ( lane == 0u )
	{
		best_score[warp] = warp_best_score;
		best_candidate[warp] = warp_best_candidate;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		winner = 0u;
		for (candidate = 1; candidate < SPARK_LM_CTA_WARPS; candidate++)
			if ( best_score[candidate] > best_score[winner] || (best_score[candidate] == best_score[winner] && best_candidate[candidate] < best_candidate[winner]) )
				winner = candidate;
		output_token_ids[row] = token_ids != 0 ? token_ids[best_candidate[winner]] : best_candidate[winner];
	}
}

/*
 * Shared host-side launch pipelines - each driver wraps these in a
 * two-line extern with its own constants, so the sequence lives once.
 */
template <uint32_t GROUP_SIZE>
static cudaError_t SparkLmHostLaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension)
{
	uint32_t blocks = (candidate_count + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS;
	(void)sizeof(char[GROUP_SIZE == SPARK_LM_HEAD_SHADOW_GROUP ? 1 : -1]);
	SparkLmHeadShadowQuantizeKernel<<<blocks,SPARK_LM_CTA_THREADS,0,stream>>>(head_bf16,shadow_payload,shadow_scale,error_norm,candidate_count,hidden_dimension);
	return(cudaGetLastError());
}

static inline cudaError_t SparkLmHostLaunchHeadCertifiedFp8Quantize(
	cudaStream_t stream,const void *head_bf16,uint8_t *shadow_payload,
	float *shadow_scale_f32,float *cert_norm_f32,uint32_t candidate_count,
	uint32_t hidden_dimension)
{
	uint32_t blocks;
	if ( head_bf16 == 0 || shadow_payload == 0 || shadow_scale_f32 == 0 ||
		cert_norm_f32 == 0 || candidate_count == 0u || hidden_dimension == 0u ||
		(hidden_dimension % SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE) != 0u )
		return(cudaErrorInvalidValue);
	blocks = (candidate_count + SPARK_LM_CTA_WARPS - 1u) /
		SPARK_LM_CTA_WARPS;
	SparkLmHeadCertifiedFp8QuantizeKernel<<<blocks,SPARK_LM_CTA_THREADS,0u,
		stream>>>(head_bf16,shadow_payload,shadow_scale_f32,cert_norm_f32,
		candidate_count,hidden_dimension);
	return(cudaGetLastError());
}

typedef struct SparkLmHeadCertifiedFp8Scratch
{
	float *coarse_scores;
	float *bounds;
	float *hidden_norms;
	float *partial_scores;
	uint32_t *partial_candidates;
} SparkLmHeadCertifiedFp8Scratch;

static inline SparkLmHeadCertifiedFp8Scratch SparkLmHeadCertifiedFp8ScratchView(
	void *scratch,uint32_t vocabulary_count,uint32_t hidden_dimension)
{
	SparkLmHeadCertifiedFp8Scratch view;
	view.coarse_scores = (float *)scratch;
	view.bounds = view.coarse_scores + vocabulary_count;
	view.hidden_norms = view.bounds + vocabulary_count;
	view.partial_scores = view.hidden_norms +
		(hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE);
	view.partial_candidates = (uint32_t *)(view.partial_scores +
		SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT);
	return(view);
}

static inline cudaError_t SparkLmHostLaunchHeadCertifiedFp8B1WithScore(
	cudaStream_t stream,const void *hidden_bf16,const void *head_weight_bf16,
	const uint8_t *shadow_payload,const float *shadow_scale_f32,
	const float *cert_norm_f32,void *scratch,uint32_t *candidate_ids,
	uint32_t *candidate_count,uint32_t *output_token_id,float *output_score,
	uint32_t candidate_offset,uint32_t row_count,uint32_t vocabulary_count,
	uint32_t hidden_dimension)
{
	uint32_t groups = hidden_dimension / SPARK_HEAD_CERTIFIED_FP8_GROUP_SIZE;
	uint32_t norm_blocks = (groups + SPARK_LM_CTA_WARPS - 1u) /
		SPARK_LM_CTA_WARPS;
	uint32_t score_blocks = (vocabulary_count +
		SPARK_LM_HEAD_CERTIFIED_FP8_WARPS - 1u) /
		SPARK_LM_HEAD_CERTIFIED_FP8_WARPS;
	SparkLmHeadCertifiedFp8Scratch view;
	if ( hidden_bf16 == 0 || head_weight_bf16 == 0 || shadow_payload == 0 ||
		shadow_scale_f32 == 0 || cert_norm_f32 == 0 || scratch == 0 ||
		candidate_ids == 0 || candidate_count == 0 || output_token_id == 0 ||
		output_score == 0 || row_count != 1u || vocabulary_count == 0u ||
		hidden_dimension == 0u || (hidden_dimension % 512u) != 0u )
		return(cudaErrorInvalidValue);
	view = SparkLmHeadCertifiedFp8ScratchView(scratch,vocabulary_count,
		hidden_dimension);
	SparkLmHeadCertifiedHiddenNormKernel<<<norm_blocks,SPARK_LM_CTA_THREADS,0u,
		stream>>>(hidden_bf16,view.hidden_norms,hidden_dimension);
	SparkLmHeadCertifiedFp8ScoreKernel<<<score_blocks,
		SPARK_LM_HEAD_CERTIFIED_FP8_THREADS,0u,stream>>>(hidden_bf16,
		shadow_payload,shadow_scale_f32,cert_norm_f32,view.hidden_norms,
		view.coarse_scores,view.bounds,vocabulary_count,hidden_dimension);
	SparkLmHeadCertifiedScreenKernel<<<1u,SPARK_LM_CTA_THREADS,0u,stream>>>(
		view.coarse_scores,view.bounds,candidate_ids,candidate_count,
		vocabulary_count);
	SparkLmHeadCertifiedRescoreKernel<<<SPARK_HEAD_CERTIFIED_FP8_PARTIAL_COUNT,
		SPARK_LM_CTA_THREADS,hidden_dimension * sizeof(float),stream>>>(
		hidden_bf16,head_weight_bf16,candidate_ids,candidate_count,
		view.partial_scores,view.partial_candidates,hidden_dimension);
	SparkLmHeadCertifiedReduceKernel<<<1u,SPARK_LM_CTA_THREADS,0u,stream>>>(
		view.partial_scores,view.partial_candidates,output_token_id,output_score,
		candidate_offset);
	return(cudaGetLastError());
}

static inline cudaError_t SparkLmHostLaunchHeadDirectArgmaxWithScore(
	cudaStream_t stream,const void *hidden_bf16,const void *head_weight_bf16,
	void *scratch,uint32_t *candidate_counts,uint32_t *output_token_ids,
	float *output_scores,uint32_t candidate_offset,uint32_t row_count,
	uint32_t candidate_count,uint32_t hidden_dimension)
{
	float *partial_scores = (float *)scratch;
	uint32_t *partial_candidates = (uint32_t *)(partial_scores +
		((uint64_t)row_count * SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT));
	dim3 fallback_grid(row_count,SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT);
	uint32_t shared_bytes = hidden_dimension * (uint32_t)sizeof(float);
	cudaError_t error;
	error = cudaMemsetAsync(candidate_counts,0xff,
		(uint64_t)row_count * sizeof(candidate_counts[0]),stream);
	if ( error != cudaSuccess )
		return(error);
	SparkLmHeadFallbackRescoreKernel<<<fallback_grid,SPARK_LM_CTA_THREADS,
		shared_bytes,stream>>>(hidden_bf16,head_weight_bf16,candidate_counts,
		partial_scores,partial_candidates,row_count,hidden_dimension,
		candidate_count);
	SparkLmHeadRescoreArgmaxKernel<<<row_count,SPARK_LM_CTA_THREADS,
		shared_bytes,stream>>>(hidden_bf16,head_weight_bf16,0,candidate_counts,
		partial_scores,partial_candidates,output_token_ids,output_scores,
		candidate_offset,row_count,hidden_dimension);
	return(cudaGetLastError());
}

static inline cudaError_t SparkLmHostLaunchHeadScreenedArgmaxWithScore(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, float *output_scores, uint32_t candidate_offset, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension)
{
    dim3 tile_grid(
        (row_count + SPARK_LM_TILE - 1u) / SPARK_LM_TILE,
        (candidate_count + SPARK_LM_TILE_N - 1u) / SPARK_LM_TILE_N);
    dim3 fallback_grid(row_count,SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT);
    float *partial_scores = (float *)logits_bf16;
    uint32_t *partial_candidates = (uint32_t *)(partial_scores + ((uint64_t)row_count * SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT));
    uint32_t grouped_rows = SPARK_LM_HEAD_FALLBACK_SHARED_BYTES / (hidden_dimension * (uint32_t)sizeof(uint16_t));
	uint32_t rescore_shared_bytes = hidden_dimension * (uint32_t)sizeof(float);
	grouped_rows = grouped_rows > SPARK_LM_HEAD_FALLBACK_ROW_GROUP_MAX ? SPARK_LM_HEAD_FALLBACK_ROW_GROUP_MAX : grouped_rows;
	grouped_rows = grouped_rows > row_count ? row_count : grouped_rows;
	if ( row_count == 1u )
		return(SparkLmHostLaunchHeadDirectArgmaxWithScore(stream,hidden_bf16,
			head_weight_bf16,logits_bf16,candidate_counts,output_token_ids,
			output_scores,candidate_offset,row_count,candidate_count,
			hidden_dimension));

	/*
	 * DSV4's exact decode shapes use the native packed shadow projection.  The
	 * compatibility branches remain for other model families and non-native
	 * batch shapes; DSV4's wrapper rejects those shapes before this function.
	 */
	if ( SparkLmSm121NativeDecodeShape(row_count) != 0u &&
		(hidden_dimension % 128u) == 0u &&
		(candidate_count % SPARK_LM_SM121_NATIVE_TILE_N) == 0u )
	{
		cudaError_t native_status = SparkLmHostLaunchSm121NativeLinear<
			SPARK_LM_SM121_NATIVE_WEIGHT_MXFP4>(stream,shadow_payload,
			shadow_scale,(uint64_t)candidate_count * hidden_dimension / 2u,
			(uint64_t)candidate_count * (hidden_dimension /
				SPARK_LM_HEAD_SHADOW_GROUP),hidden_bf16,hidden_dimension,0u,0u,
			logits_bf16,candidate_count,0u,0u,1u,row_count,
			hidden_dimension,candidate_count);
		if ( native_status != cudaSuccess )
			return(native_status);
	}
	else if ( row_count < SPARK_LM_TILE )
	{
		dim3 shadow_grid(
			row_count,
			(candidate_count + SPARK_LM_SCALAR_CTA_WARPS - 1u) /
			SPARK_LM_SCALAR_CTA_WARPS);
		uint32_t shadow_shared_bytes = hidden_dimension * (uint32_t)sizeof(float);
		SparkLmLinearKernel<SPARK_LM_HEAD_SHADOW_GROUP,
			SPARK_ACTIVATION_CODEC_NONE,SPARK_LM_SCALAR_CTA_WARPS><<<
			shadow_grid,
			SPARK_LM_SCALAR_CTA_THREADS,
			shadow_shared_bytes,
			stream>>>(
			SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1,
			shadow_payload,
			shadow_scale,
			hidden_bf16,
			logits_bf16,
			row_count,
			hidden_dimension,
			candidate_count);
	}
	else
	{
		SparkLmExpertTileKernel<SPARK_LM_HEAD_SHADOW_GROUP><<<
			tile_grid,
			SPARK_LM_CTA_THREADS,
			0u,
			stream>>>(
				SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1,
				shadow_payload,
				shadow_scale,
				hidden_bf16,
				0,
				logits_bf16,
				row_count,
				hidden_dimension,
				candidate_count);
	}
    SparkLmHeadScreenKernel<<<row_count, SPARK_LM_CTA_THREADS, 0u, stream>>>(
        hidden_bf16,
        logits_bf16,
        error_norm,
        candidate_ids,
        candidate_counts,
        row_count,
        candidate_count,
        hidden_dimension);
	if ( grouped_rows <= 1u || (hidden_dimension & 1u) != 0u )
	{
		SparkLmHeadFallbackRescoreKernel<<<fallback_grid,SPARK_LM_CTA_THREADS,rescore_shared_bytes,stream>>>(hidden_bf16,head_weight_bf16,candidate_counts,partial_scores,partial_candidates,row_count,hidden_dimension,candidate_count);
	}
	else
	{
		dim3 grouped_grid((row_count + grouped_rows - 1u) / grouped_rows,SPARK_LM_HEAD_FALLBACK_CHUNK_COUNT);
		uint32_t grouped_shared_bytes = grouped_rows * hidden_dimension * (uint32_t)sizeof(uint16_t);
		SparkLmHeadGroupedFallbackRescoreKernel<<<grouped_grid,SPARK_LM_CTA_THREADS,grouped_shared_bytes,stream>>>(hidden_bf16,head_weight_bf16,candidate_counts,partial_scores,partial_candidates,row_count,hidden_dimension,candidate_count,grouped_rows);
	}
    SparkLmHeadRescoreArgmaxKernel<<<
        row_count,
        SPARK_LM_CTA_THREADS,
        rescore_shared_bytes,
        stream>>>(
            hidden_bf16,
            head_weight_bf16,
            candidate_ids,
            candidate_counts,
            partial_scores,
            partial_candidates,
            output_token_ids,
			output_scores,
			candidate_offset,
            row_count,
            hidden_dimension);
    return cudaGetLastError();
}

static inline cudaError_t SparkLmHostLaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension)
{
	return(SparkLmHostLaunchHeadScreenedArgmaxWithScore(stream,hidden_bf16,
		head_weight_bf16,shadow_payload,shadow_scale,error_norm,logits_bf16,
		candidate_ids,candidate_counts,output_token_ids,0,0u,row_count,
		candidate_count,hidden_dimension));
}

static inline cudaError_t SparkLmHostLaunchMoeGroup(cudaStream_t stream, const uint32_t *pair_expert_ids, uint32_t pair_count, uint32_t expert_count, uint32_t experts_per_token, uint32_t *expert_offsets, uint32_t *grouped_rows, uint32_t *grouped_weight_slots, uint32_t *inverse_map)
{
	SparkLmMoeGroupKernel<<<1u,SPARK_LM_CTA_THREADS,0,stream>>>(pair_expert_ids,pair_count,expert_count,experts_per_token,expert_offsets,grouped_rows,grouped_weight_slots,inverse_map);
	return(cudaGetLastError());
}

// Size-aware batched dense linear/QKVO projection. The tile path is also the
// fast path for tiny aligned batches: it zero-fills the padded M rows and
// keeps one weight tile resident while tensor cores process the live row. The
// scalar fallback remains for widths with a partial K tile or a scale layout
// that the tile decoder does not implement.
//
//   B < SPARK_LM_TILE  (tiny): the tensor tile for aligned production shapes;
//     scalar only for the explicitly unsupported layouts. This avoids making
//     B1 pay one scalar CTA per 128 output neurons for every projection.
//   SPARK_LM_TILE <= B <= READ_ONCE: one tile pass, weight read once across
//     the batch. Per-token cost is flat B16..B128 - one kernel serves it.
//   B > READ_ONCE (wide): the SAME tile, but grid.x rasterizes M as the
//     FAST axis so adjacent M-blocks of one N-tile reuse the weight strip
//     from L2 instead of re-reading DRAM. This is the measured knob - the
//     deleted per-size wide-warp variants lost to grid rasterization, not
//     to a kernel-per-size. ceil(B/TILE) M-blocks per N-tile, M fastest.
//
// All three inherit the FP8 tensor path under SPARK_LM_FP8_TILE (the tile
// dispatch), and every dimension is parametric so the shape carries to the
// next model generation unchanged.
//
/*
 * Hard validation of the tile's input contract. The tile does not fail on
 * bad input, it silently produces wrong numbers, in two ways. A width that
 * is not a multiple of the K tile leaves a partial trailing K tile whose
 * stagers bound rows and neurons but never K, so it folds whatever follows
 * the row into the dot product. And a weight format the decoder has no
 * branch for - F32 and U32 - falls past the BF16 and MXFP4 early returns
 * into the FP8 path and is read as packed E4M3 bytes.
 *
 * Both are latent today: every shipped projection width is K-aligned and no
 * live view carries F32 or U32. Both would be silent wrong tokens the moment
 * that changes, with nothing in the output to reveal it. They fail loudly
 * here instead, naming the offending value, because a plausible wrong answer
 * costs far more than a refused launch. The tiny-batch scalar path carries
 * explicit tails and handles any width, so only the tile regime is bound by
 * the K-multiple rule.
 */
static inline cudaError_t SparkLmValidateLinearContract(uint32_t weight_format, uint32_t row_count, uint32_t input_dimension)
{
	if ( weight_format != SPARK_LM_WEIGHT_FORMAT_BF16 && weight_format != SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1 && weight_format != SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 && weight_format != SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
	{
		fprintf(stderr,"spark_lm_kernels: linear dispatch got weight_format %u, which no decoder branch handles\n",weight_format);
		return(cudaErrorInvalidValue);
	}
	if ( row_count >= SPARK_LM_TILE && (input_dimension % SPARK_LM_TILE_K) != 0u )
	{
		fprintf(stderr,"spark_lm_kernels: linear dispatch got input_dimension %u, not a multiple of the K tile %u\n",input_dimension,(unsigned)SPARK_LM_TILE_K);
		return(cudaErrorInvalidValue);
	}
	return(cudaSuccess);
}

template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC=SPARK_ACTIVATION_CODEC_NONE>
static inline cudaError_t SparkLmHostLaunchGroupedScalarLinear(cudaStream_t stream, uint32_t weight_format, const void *payload_base, const uint8_t *scale_base, uint64_t payload_group_stride_bytes, uint64_t scale_group_stride_bytes, const void *input_bf16, const uint32_t *source_row_map, uint32_t source_row_count, const uint32_t *group_row_offset, const uint32_t *group_tile_prefix, void *output_bf16, uint32_t group_count, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessor_count)
{
	uint32_t block_count,neuron_tiles;
	size_t shared_bytes;
	if ( payload_base == 0 || input_bf16 == 0 || group_row_offset == 0 || group_tile_prefix == 0 || output_bf16 == 0 || source_row_count == 0u || group_count == 0u || input_dimension == 0u || output_dimension == 0u || multiprocessor_count == 0u || payload_group_stride_bytes == 0u || (source_row_map == 0 && source_row_count == 0u) )
		return(cudaErrorInvalidValue);
	if ( weight_format != SPARK_LM_WEIGHT_FORMAT_BF16 && weight_format != SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1 && weight_format != SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 && weight_format != SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
		return(cudaErrorInvalidValue);
	if ( weight_format != SPARK_LM_WEIGHT_FORMAT_BF16 && (scale_base == 0 || scale_group_stride_bytes == 0u) )
		return(cudaErrorInvalidValue);
	if ( SparkActivationCodecIsKnown(ACTIVATION_CODEC) == 0u || (ACTIVATION_CODEC != SPARK_ACTIVATION_CODEC_NONE && (input_dimension % SparkActivationCodecGroupSize(ACTIVATION_CODEC)) != 0u) )
		return(cudaErrorInvalidValue);
	neuron_tiles = (output_dimension + SPARK_LM_TILE_N - 1u) / SPARK_LM_TILE_N;
	if ( neuron_tiles == 0u )
		return(cudaErrorInvalidValue);
	block_count = multiprocessor_count * 2u;
	if ( block_count == 0u )
		block_count = 1u;
	shared_bytes = (size_t)input_dimension * sizeof(float);
	SparkLmGroupedScalarLinearKernel<GROUP_SIZE,ACTIVATION_CODEC><<<block_count,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(weight_format,payload_base,scale_base,payload_group_stride_bytes,scale_group_stride_bytes,input_bf16,source_row_map,source_row_count,group_row_offset,group_tile_prefix,output_bf16,group_count,input_dimension,output_dimension);
	return(cudaGetLastError());
}

// The tile also REQUIRES input_dimension to be a multiple of the K tile. Its
// stagers bound rows and neurons but never K, so a trailing partial K tile
// stages whatever follows the row in memory and folds it into the dot
// product - wrong output, no crash. Expert widths are always K-aligned so
// this never bit the expert path, but a dense projection can have any width
// and the next model generation may well introduce one. Non-aligned widths
// therefore take the scalar path, which carries explicit scalar tails and
// handles arbitrary dimensions exactly.
template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC=SPARK_ACTIVATION_CODEC_NONE>
static inline cudaError_t SparkLmHostLaunchBatchedLinear(cudaStream_t stream, uint32_t weight_format, const void *weight_payload, const void *weight_scale, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	uint32_t m_blocks = (row_count + SPARK_LM_TILE - 1u) / SPARK_LM_TILE;
	uint32_t n_tiles = (output_dimension + SPARK_LM_TILE_N - 1u) / SPARK_LM_TILE_N;
	cudaError_t contract = SparkLmValidateLinearContract(weight_format,row_count,input_dimension);
	if ( contract != cudaSuccess )
		return(contract);
	if ( SparkActivationCodecIsKnown(ACTIVATION_CODEC) == 0u )
		return(cudaErrorInvalidValue);
	if ( ACTIVATION_CODEC != SPARK_ACTIVATION_CODEC_NONE && (input_dimension % SparkActivationCodecGroupSize(ACTIVATION_CODEC)) != 0u )
		return(cudaErrorInvalidValue);
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 &&
		(weight_scale == 0 || (input_dimension % 128u) != 0u ||
			(output_dimension % 128u) != 0u) )
		return(cudaErrorInvalidValue);
	if ( row_count < SPARK_LM_TILE )
	{
		uint32_t shared_bytes = input_dimension * (uint32_t)sizeof(float);
		dim3 scalar_grid(row_count,(output_dimension +
			SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
		SparkLmLinearKernel<GROUP_SIZE,ACTIVATION_CODEC,
			SPARK_LM_CTA_WARPS><<<scalar_grid,
			SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(weight_format,
			weight_payload,weight_scale,input_bf16,output_bf16,row_count,
			input_dimension,output_dimension);
		return(cudaGetLastError());
	}
	// B >= TILE: the tensor tile spans BOTH the read-once (B<=128) and the
	// wide (B>128) regimes in one launch. grid.x is the M axis (blockIdx.x
	// -> M in the tile body), so the default rasterization already walks all
	// M-blocks of one N-tile consecutively - adjacent M-blocks reuse the
	// weight strip warm in L2, which IS the "make M the fast grid" knob the
	// scaling analysis names. That is why the per-size wide-warp variants
	// were deleted rather than kept: the tile subsumes them. No third path
	// is built because the measurement shows none pays - per-token cost is
	// flat B16..B128 and the wide regime is the same kernel with more
	// M-blocks. A future in-block M-chunk loop, if B256+ profiling demands
	// it, changes the tile body, not this dispatcher.
	dim3 tile_grid(m_blocks,n_tiles);
	SparkLmExpertTileKernel<GROUP_SIZE,ACTIVATION_CODEC><<<tile_grid,SPARK_LM_CTA_THREADS,0,stream>>>(weight_format,weight_payload,weight_scale,input_bf16,0,output_bf16,row_count,input_dimension,output_dimension);
	return(cudaGetLastError());
}

static inline uint32_t SparkLmSm121B1Bf16LinearPairPolicy(
	uint32_t row_count,uint32_t input_dimension,uint32_t output_dimension)
{
	uint64_t work = (uint64_t)input_dimension * output_dimension;
	if ( row_count != 1u || work < SPARK_LM_BF16_PAIR_WIDE_MINIMUM_WORK )
		return(SPARK_LM_PAIR_POLICY_FLAT_8);
	return(SPARK_LM_PAIR_POLICY_FLAT_16);
}

static inline uint32_t SparkLmSm121B1Fp8LinearPairPolicy(
	uint32_t row_count,uint32_t input_dimension,
	uint32_t combined_output_dimension)
{
	uint64_t work = (uint64_t)input_dimension * combined_output_dimension;
	return(row_count == 1u && work >= SPARK_LM_FP8_PAIR_WIDE_MINIMUM_WORK ?
		SPARK_LM_PAIR_POLICY_FLAT_16 : SPARK_LM_PAIR_POLICY_FLAT_8);
}

/* M-group dense launch: BF16 only, at least two m-tiles, K past one
 * tile so the pipelined loop engages. Caller: qwen38 dense linears. */
static inline cudaError_t SparkLmHostLaunchBatchedLinearMloop(cudaStream_t stream, const void *weight_payload, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	uint32_t m_blocks = (row_count + SPARK_LM_TILE - 1u) / SPARK_LM_TILE;
	uint32_t m_groups = (m_blocks + SPARK_LM_MLOOP_GROUP - 1u) / SPARK_LM_MLOOP_GROUP;
	dim3 grid(m_groups,(output_dimension + SPARK_LM_TILE_N - 1u) / SPARK_LM_TILE_N);
	if ( input_dimension % SPARK_LM_TILE_K != 0u )
		return(cudaErrorInvalidValue);
	SparkLmExpertTileMloopKernel<32u><<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(weight_payload,input_bf16,output_bf16,row_count,input_dimension,output_dimension);
	return(cudaGetLastError());
}

/* Grouped m-loop launch: qwen38 grouped FP8 experts. */
static inline cudaError_t SparkLmHostLaunchGroupedExpertTileMloop(cudaStream_t stream, uint32_t weight_format, const void *payload_base, const void *scale_base, uint64_t payload_stride, uint64_t scale_stride, const void *input_bf16, const uint32_t *grouped_rows, const uint32_t *expert_offsets, void *output_bf16, uint32_t input_dimension, uint32_t output_dimension, uint32_t expert_count)
{
	/* Expert-stride CTA budget: 64 columns x n_tiles CTAs, each walking up
	 * to expert_count/64 experts - empty experts cost a few cycles instead
	 * of a launch. */
	uint32_t budget = expert_count < 64u ? expert_count : 64u;
	dim3 grid(1u,(output_dimension + SPARK_LM_TILE_N - 1u) / SPARK_LM_TILE_N,budget);
	SparkLmExpertTileAllMloopKernel<32u><<<grid,SPARK_LM_CTA_THREADS,0,stream>>>(weight_format,payload_base,scale_base,payload_stride,scale_stride,input_bf16,grouped_rows,expert_offsets,output_bf16,input_dimension,output_dimension,expert_count);
	return(cudaGetLastError());
}

static inline cudaError_t SparkLmHostLaunchBf16LinearPair(
	cudaStream_t stream,const void *first_weight,const void *second_weight,
	const void *input_bf16,void *first_output_bf16,void *second_output_bf16,
	uint32_t row_count,uint32_t input_dimension,uint32_t output_dimension)
{
	dim3 grid;
	cudaError_t error;
	uint32_t policy;
	if ( first_weight == 0 || second_weight == 0 || input_bf16 == 0 ||
		first_output_bf16 == 0 || second_output_bf16 == 0 || row_count == 0u ||
		input_dimension == 0u || output_dimension == 0u )
		return(cudaErrorInvalidValue);
	if ( row_count < SPARK_LM_TILE )
	{
		if ( output_dimension > UINT32_MAX / 2u )
			return(cudaErrorInvalidValue);
		policy = SparkLmSm121B1Bf16LinearPairPolicy(row_count,input_dimension,
			output_dimension);
		if ( policy == SPARK_LM_PAIR_POLICY_FLAT_16 )
		{
			grid = dim3(row_count,(output_dimension * 2u + 15u) / 16u);
			SparkLmBf16LinearPairKernel<16u><<<grid,512u,
				input_dimension * sizeof(float),stream>>>(first_weight,second_weight,
				input_bf16,first_output_bf16,second_output_bf16,row_count,
				input_dimension,output_dimension);
			return(cudaGetLastError());
		}
		grid = dim3(row_count,(output_dimension * 2u +
			SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
		SparkLmBf16LinearPairKernel<SPARK_LM_CTA_WARPS><<<grid,
			SPARK_LM_CTA_THREADS,
			input_dimension * sizeof(float),stream>>>(first_weight,
			second_weight,input_bf16,first_output_bf16,second_output_bf16,
			row_count,input_dimension,output_dimension);
		return(cudaGetLastError());
	}
	error = SparkLmHostLaunchBatchedLinear<32u,SPARK_ACTIVATION_CODEC_NONE>(
		stream,SPARK_LM_WEIGHT_FORMAT_BF16,first_weight,0,input_bf16,
		first_output_bf16,row_count,input_dimension,output_dimension);
	if ( error != cudaSuccess )
		return(error);
	return(SparkLmHostLaunchBatchedLinear<32u,SPARK_ACTIVATION_CODEC_NONE>(
		stream,SPARK_LM_WEIGHT_FORMAT_BF16,second_weight,0,input_bf16,
		second_output_bf16,row_count,input_dimension,output_dimension));
}

/*
 * True B1 is a matrix-vector product: padding it to the M16 tensor-core atom
 * multiplies the activation work by sixteen and is slower on GB10.  Keep the
 * native tensor path for the qualified B8/B1024 buckets; only B1 selects the
 * bandwidth-oriented GEMV shared by model families.
 */
template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC=SPARK_ACTIVATION_CODEC_NONE>
static inline cudaError_t SparkLmHostLaunchSm121DecodeLinear(cudaStream_t stream, uint32_t weight_format, const void *weight_payload, const void *weight_scale, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	dim3 grid;
	if ( weight_payload == 0 || input_bf16 == 0 || output_bf16 == 0 || SparkLmSm121NativeDecodeShape(row_count) == 0u || input_dimension == 0u || output_dimension == 0u )
		return(cudaErrorInvalidValue);
	if ( row_count == 1u )
		return(SparkLmHostLaunchBatchedLinear<GROUP_SIZE,ACTIVATION_CODEC>(stream,weight_format,weight_payload,weight_scale,input_bf16,output_bf16,row_count,input_dimension,output_dimension));
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 )
	{
		if ( weight_scale == 0 || input_dimension % 128u != 0u || output_dimension % SPARK_LM_SM121_NATIVE_TILE_N != 0u )
			return(cudaErrorInvalidValue);
		return(SparkLmHostLaunchSm121NativeLinear<SPARK_LM_SM121_NATIVE_WEIGHT_FP8>(stream,weight_payload,(const uint8_t *)weight_scale,(uint64_t)output_dimension * input_dimension,(uint64_t)output_dimension * (input_dimension / 128u),input_bf16,input_dimension,0u,0u,output_bf16,output_dimension,0u,0u,1u,row_count,input_dimension,output_dimension));
	}
	if ( weight_format != SPARK_LM_WEIGHT_FORMAT_BF16 || (input_dimension % SPARK_LM_TILE_K) != 0u )
		return(cudaErrorInvalidValue);
	grid = dim3((row_count + SPARK_LM_TILE - 1u) / SPARK_LM_TILE,(output_dimension + SPARK_LM_TILE_N - 1u) / SPARK_LM_TILE_N);
	SparkLmExpertTileKernel<GROUP_SIZE,ACTIVATION_CODEC><<<grid,SPARK_LM_CTA_THREADS,0u,stream>>>(weight_format,weight_payload,weight_scale,input_bf16,0,output_bf16,row_count,input_dimension,output_dimension);
	return(cudaGetLastError());
}

template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC>
static inline cudaError_t SparkLmHostLaunchSm121DecodeLinearPair(
	cudaStream_t stream,const void *first_payload,const uint8_t *first_scale,
	uint32_t first_output_dimension,const void *second_payload,
	const uint8_t *second_scale,uint32_t second_output_dimension,
	const void *input_bf16,void *first_output_bf16,void *second_output_bf16,
	uint32_t row_count,uint32_t input_dimension)
{
	uint32_t combined_output_dimension;
	if ( first_payload == 0 || first_scale == 0 || second_payload == 0 ||
		second_scale == 0 || input_bf16 == 0 || first_output_bf16 == 0 ||
		second_output_bf16 == 0 || first_output_dimension == 0u ||
		second_output_dimension == 0u || input_dimension == 0u ||
		(input_dimension % GROUP_SIZE) != 0u ||
		SparkActivationCodecIsKnown(ACTIVATION_CODEC) == 0u )
		return(cudaErrorInvalidValue);
	if ( first_output_dimension > UINT32_MAX - second_output_dimension )
		return(cudaErrorInvalidValue);
	combined_output_dimension = first_output_dimension + second_output_dimension;
	if ( row_count != 1u )
	{
		/* Multi-row must be bit-identical to the certified 1-row math:
		 * the verify anchor (row 0) is diffed against the 1-row lean
		 * baseline, and the two-launch batched route accumulates
		 * fp16-level noise (first divergent tensor: delta_wq_a,
		 * measured). Run the exact pair kernel once per row instead of
		 * two batched linears: same shared-input load, same dot, same
		 * writeback as the rows==1 path, row by row. */
		uint64_t input_row_bytes = (uint64_t)input_dimension * 2u;
		uint64_t first_row_bytes = (uint64_t)first_output_dimension * 2u;
		uint64_t second_row_bytes = (uint64_t)second_output_dimension * 2u;
		uint32_t row,policy;
		for (row = 0u; row < row_count; row++)
		{
			const uint8_t *row_input = (const uint8_t *)input_bf16 +
				(uint64_t)row * input_row_bytes;
			uint8_t *row_first = (uint8_t *)first_output_bf16 +
				(uint64_t)row * first_row_bytes;
			uint8_t *row_second = (uint8_t *)second_output_bf16 +
				(uint64_t)row * second_row_bytes;
			policy = SparkLmSm121B1Fp8LinearPairPolicy(1u,
				input_dimension,combined_output_dimension);
			if ( policy == SPARK_LM_PAIR_POLICY_FLAT_16 )
				SparkLmFp8LinearPairKernel<GROUP_SIZE,ACTIVATION_CODEC,
					16u><<<(combined_output_dimension + 15u) / 16u,512u,
					input_dimension * sizeof(float),stream>>>(
					first_payload,first_scale,second_payload,second_scale,
					row_input,row_first,row_second,input_dimension,
					first_output_dimension,second_output_dimension);
			else
				SparkLmFp8LinearPairKernel<GROUP_SIZE,ACTIVATION_CODEC,
					SPARK_LM_CTA_WARPS><<<
					(combined_output_dimension + SPARK_LM_CTA_WARPS - 1u) /
					SPARK_LM_CTA_WARPS,SPARK_LM_CTA_THREADS,
					input_dimension * sizeof(float),stream>>>(
					first_payload,first_scale,second_payload,second_scale,
					row_input,row_first,row_second,input_dimension,
					first_output_dimension,second_output_dimension);
		}
		return(cudaGetLastError());
	}
	if ( SparkLmSm121B1Fp8LinearPairPolicy(row_count,input_dimension,
		combined_output_dimension) == SPARK_LM_PAIR_POLICY_FLAT_16 )
		SparkLmFp8LinearPairKernel<GROUP_SIZE,ACTIVATION_CODEC,16u><<<
			(combined_output_dimension + 15u) / 16u,512u,
			input_dimension * sizeof(float),stream>>>(first_payload,first_scale,
			second_payload,second_scale,input_bf16,first_output_bf16,
			second_output_bf16,input_dimension,first_output_dimension,
			second_output_dimension);
	else
		SparkLmFp8LinearPairKernel<GROUP_SIZE,ACTIVATION_CODEC,
			SPARK_LM_CTA_WARPS><<<
			(combined_output_dimension + SPARK_LM_CTA_WARPS - 1u) /
			SPARK_LM_CTA_WARPS,SPARK_LM_CTA_THREADS,
			input_dimension * sizeof(float),stream>>>(first_payload,first_scale,
			second_payload,second_scale,input_bf16,first_output_bf16,
			second_output_bf16,input_dimension,first_output_dimension,
			second_output_dimension);
	return(cudaGetLastError());
}

template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC=SPARK_ACTIVATION_CODEC_NONE>
static inline cudaError_t SparkLmHostLaunchSm121StridedDecodeLinear(cudaStream_t stream, uint32_t weight_format, const void *weight_payload, const uint8_t *weight_scale, uint64_t weight_payload_group_stride_bytes, uint64_t weight_scale_group_stride_bytes, const void *input_bf16, uint64_t input_row_stride, uint32_t input_offset, uint32_t input_group_stride, void *output_bf16, uint64_t output_row_stride, uint32_t output_offset, uint32_t output_group_stride, uint32_t group_count, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	dim3 grid;
	if ( weight_payload == 0 || input_bf16 == 0 || output_bf16 == 0 || group_count == 0u || SparkLmSm121NativeDecodeShape(row_count) == 0u || input_dimension == 0u || output_dimension == 0u )
		return(cudaErrorInvalidValue);
	if ( row_count == 1u )
	{
		if ( SparkActivationCodecIsKnown(ACTIVATION_CODEC) == 0u || (ACTIVATION_CODEC != SPARK_ACTIVATION_CODEC_NONE && (input_dimension % SparkActivationCodecGroupSize(ACTIVATION_CODEC)) != 0u) )
			return(cudaErrorInvalidValue);
		grid = dim3(1u,(output_dimension + SPARK_LM_CTA_WARPS - 1u) /
			SPARK_LM_CTA_WARPS,group_count);
		SparkLmStridedLinearKernel<GROUP_SIZE,ACTIVATION_CODEC><<<grid,
			SPARK_LM_CTA_THREADS,input_dimension * (uint32_t)sizeof(float),stream>>>(
			weight_format,weight_payload,weight_scale,
			weight_payload_group_stride_bytes,weight_scale_group_stride_bytes,
			input_bf16,input_row_stride,input_offset,input_group_stride,output_bf16,
			output_row_stride,output_offset,output_group_stride,row_count,
			input_dimension,output_dimension);
		return(cudaGetLastError());
	}
	if ( weight_format != SPARK_LM_WEIGHT_FORMAT_FP8_E4M3 || weight_scale == 0 ||
		SparkActivationCodecIsKnown(ACTIVATION_CODEC) == 0u ||
		(ACTIVATION_CODEC != SPARK_ACTIVATION_CODEC_NONE &&
		 (input_dimension % SparkActivationCodecGroupSize(ACTIVATION_CODEC)) != 0u) )
		return(cudaErrorInvalidValue);
	/* Multi-row must be bit-identical to the certified 1-row math: the
	 * native MXFP8 route quantizes the BF16 activation per-K-block,
	 * which diverges from the 1-row strided kernel and corrupts the
	 * verify anchor's output composition (wo_a -> o_ranks). Launch the
	 * exact strided kernel once per row, base pointers advanced by the
	 * row strides (strides are in BF16 elements). */
	{
		uint32_t row;
		for (row = 0u; row < row_count; row++)
		{
			const uint8_t *row_input = (const uint8_t *)input_bf16 +
				(uint64_t)row * input_row_stride * 2u;
			uint8_t *row_output = (uint8_t *)output_bf16 +
				(uint64_t)row * output_row_stride * 2u;
			grid = dim3(1u,(output_dimension + SPARK_LM_CTA_WARPS - 1u) /
				SPARK_LM_CTA_WARPS,group_count);
			SparkLmStridedLinearKernel<GROUP_SIZE,ACTIVATION_CODEC><<<grid,
				SPARK_LM_CTA_THREADS,
				input_dimension * (uint32_t)sizeof(float),stream>>>(
				weight_format,weight_payload,weight_scale,
				weight_payload_group_stride_bytes,
				weight_scale_group_stride_bytes,
				row_input,input_row_stride,input_offset,input_group_stride,
				row_output,output_row_stride,output_offset,
				output_group_stride,1u,input_dimension,output_dimension);
		}
		return(cudaGetLastError());
	}
}

static inline cudaError_t SparkLmHostLaunchMoePairReduce(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint32_t row_count, uint32_t experts_per_token, uint32_t width)
{
	SparkLmMoePairReduceKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(slot_out_bf16,inverse_map,pair_weights_f32,accum_bf16,row_count,experts_per_token,width);
	return(cudaGetLastError());
}

static inline cudaError_t SparkLmHostLaunchMoePairReduceStrided(
	cudaStream_t stream,
	const void *slot_out_bf16,
	const uint32_t *inverse_map,
	const float *pair_weights_f32,
	void *accum_bf16,
	uint64_t accum_row_stride,
	uint32_t accum_offset,
	uint32_t row_count,
	uint32_t experts_per_token,
	uint32_t width)
{
	if ( slot_out_bf16 == 0 || inverse_map == 0 || accum_bf16 == 0 ||
		row_count == 0u || experts_per_token == 0u ||
		experts_per_token > SPARK_LM_MOE_MAX_TOPK || width == 0u ||
		(width & 1u) != 0u || (accum_row_stride & 1u) != 0u ||
		(accum_offset & 1u) != 0u ||
		accum_row_stride < (uint64_t)accum_offset + width )
		return(cudaErrorInvalidValue);
	SparkLmMoePairReduceStridedKernel<<<row_count,SPARK_LM_CTA_THREADS,0,
		stream>>>(slot_out_bf16,inverse_map,pair_weights_f32,accum_bf16,
		accum_row_stride,accum_offset,row_count,experts_per_token,width);
	return(cudaGetLastError());
}

static inline cudaError_t SparkLmHostLaunchMoePairReduceOverwrite(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *output_bf16, uint32_t row_count, uint32_t experts_per_token, uint32_t width)
{
    SparkLmMoePairReduceOverwriteKernel<<<row_count, SPARK_LM_CTA_THREADS, 0u, stream>>>(
        slot_out_bf16,
        inverse_map,
        pair_weights_f32,
        output_bf16,
        row_count,
        experts_per_token,
        width);
    return cudaGetLastError();
}
