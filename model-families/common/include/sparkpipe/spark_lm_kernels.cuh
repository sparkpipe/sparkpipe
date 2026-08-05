#pragma once

#include <stdio.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>

#include "inference/kernels/activation.cuh"

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
	return(exp2f((float)(int32_t)byte_value - 127.0f));
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

static __global__ void SparkLmRmsNormKernel(const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
    extern __shared__ float staged_input[];
    __shared__ float reduce_scratch[SPARK_LM_CTA_WARPS];
    uint32_t row = blockIdx.x;
    uint32_t element;
    uint64_t row_offset;
    float sum_squares = 0.0f;
    float inverse_rms;
    float2 pair_value;
    float2 gain_value;

    if (row >= row_count)
    {
        return;
    }
    row_offset = ((uint64_t)row * (uint64_t)dimension) >> 1u;
    for (element = threadIdx.x; element < (dimension >> 1u); element += blockDim.x)
    {
        pair_value = SparkLmLoadBf16Pair(input_bf16, row_offset + element);
        staged_input[element << 1u] = pair_value.x;
        staged_input[(element << 1u) + 1u] = pair_value.y;
        sum_squares = fmaf(pair_value.x, pair_value.x, sum_squares);
        sum_squares = fmaf(pair_value.y, pair_value.y, sum_squares);
    }
    if ((dimension & 1u) != 0u && threadIdx.x == 0u)
    {
        float tail = SparkLmBf16ToFloat(input_bf16, ((uint64_t)row * dimension) + dimension - 1u);
        staged_input[dimension - 1u] = tail;
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
            (staged_input[element << 1u] * inverse_rms) * gain_value.x,
            (staged_input[(element << 1u) + 1u] * inverse_rms) * gain_value.y);
    }
    if ((dimension & 1u) != 0u && threadIdx.x == 0u)
    {
        uint64_t tail_index = ((uint64_t)row * dimension) + dimension - 1u;
        SparkLmFloatToBf16(
            output_bf16,
            tail_index,
            staged_input[dimension - 1u] * inverse_rms * SparkLmBf16ToFloat(gain_bf16, dimension - 1u));
    }
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

// One 4-byte load carries eight E2M1 elements; an eight-element run at an
// eight-aligned base never crosses a scale group (eight divides every
// GROUP_SIZE in use), so each run costs one scale decode.
template <uint32_t GROUP_SIZE>
static __device__ __forceinline__ float SparkLmDotRowMxfp4(const float *shared_input, const void *weight_payload, const uint8_t *weight_scale_e8m0, uint32_t neuron, uint32_t input_dimension, uint32_t lane)
{
	uint64_t run_row = ((uint64_t)neuron * input_dimension) >> 3u,scale_row = (uint64_t)neuron * (input_dimension / GROUP_SIZE);
	uint32_t run_count = input_dimension >> 3u,run,nibble_pair,packed;
	float accumulator = 0.0f,scale_value;
	#pragma unroll 2
	for (run = lane; run < run_count; run += SPARK_LM_WARP_LANES)
	{
		packed = __ldg(((const uint32_t *)weight_payload) + run_row + run);
		scale_value = SparkLmDecodeE8m0(weight_scale_e8m0[scale_row + ((run << 3u) / GROUP_SIZE)]);
		#pragma unroll
		for (nibble_pair = 0; nibble_pair < 4u; nibble_pair++)
		{
			accumulator = fmaf(shared_input[(run << 3u) + (nibble_pair << 1u)],SparkLmDecodeE2m1((packed >> (nibble_pair << 3u)) & 0x0fu) * scale_value,accumulator);
			accumulator = fmaf(shared_input[(run << 3u) + (nibble_pair << 1u) + 1u],SparkLmDecodeE2m1((packed >> ((nibble_pair << 3u) + 4u)) & 0x0fu) * scale_value,accumulator);
		}
	}
	return(accumulator);
}

// FP8 weights with F32 scales on [BLOCK, BLOCK] 2-D tiles - the MiMo
// checkpoint layout: element (r, c) multiplies scale[(r/B)*(C/B) + c/B].
template <uint32_t BLOCK>
static __device__ __forceinline__ float SparkLmDotRowFp8F32(const float *shared_input, const void *weight_payload, const float *weight_scale_f32, uint32_t neuron, uint32_t input_dimension, uint32_t lane)
{
	uint64_t run_row = ((uint64_t)neuron * input_dimension) >> 2u,scale_row = ((uint64_t)(neuron / BLOCK)) * (input_dimension / BLOCK);
	uint32_t run_count = input_dimension >> 2u,run,byte_index,packed;
	float accumulator = 0.0f,scale_value;
	#pragma unroll 4
	for (run = lane; run < run_count; run += SPARK_LM_WARP_LANES)
	{
		packed = __ldg(((const uint32_t *)weight_payload) + run_row + run);
		scale_value = __ldg(weight_scale_f32 + scale_row + ((run << 2u) / BLOCK));
		#pragma unroll
		for (byte_index = 0; byte_index < 4u; byte_index++)
			accumulator = fmaf(shared_input[(run << 2u) + byte_index],SparkLmDecodeE4m3((packed >> (byte_index << 3u)) & 0xffu) * scale_value,accumulator);
	}
	return(accumulator);
}

// FP8 weights: one e4m3 byte per element, one E8M0 scale per GROUP_SIZE
// columns per row - the DeepSeek block-quantized layout.
template <uint32_t GROUP_SIZE>
static __device__ __forceinline__ float SparkLmDotRowFp8(const float *shared_input, const void *weight_payload, const uint8_t *weight_scale_e8m0, uint32_t neuron, uint32_t input_dimension, uint32_t lane)
{
	uint64_t run_row = ((uint64_t)neuron * input_dimension) >> 2u,scale_row = (uint64_t)neuron * (input_dimension / GROUP_SIZE);
	uint32_t run_count = input_dimension >> 2u,run,byte_index,packed;
	float accumulator = 0.0f,scale_value;
	#pragma unroll 4
	for (run = lane; run < run_count; run += SPARK_LM_WARP_LANES)
	{
		packed = __ldg(((const uint32_t *)weight_payload) + run_row + run);
		scale_value = SparkLmDecodeE8m0(weight_scale_e8m0[scale_row + ((run << 2u) / GROUP_SIZE)]);
		#pragma unroll
		for (byte_index = 0; byte_index < 4u; byte_index++)
			accumulator = fmaf(shared_input[(run << 2u) + byte_index],SparkLmDecodeE4m3((packed >> (byte_index << 3u)) & 0xffu) * scale_value,accumulator);
	}
	return(accumulator);
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

template <uint32_t GROUP_SIZE,uint32_t ACTIVATION_CODEC=SPARK_ACTIVATION_CODEC_NONE>
static __global__ void SparkLmLinearKernel(uint32_t weight_format, const void *weight_payload, const void *weight_scale, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension)
{
	extern __shared__ float shared_input[];
	uint32_t row = blockIdx.x,neuron_base = blockIdx.y * SPARK_LM_CTA_WARPS;
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
		SparkLmFloatToBf16(output_bf16,((uint64_t)row * output_dimension) + neuron,accumulator);
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

// Exact bf16 argmax over either the screened candidate list or, when
// candidate_ids is null, the full range - the overflow fallback. A row
// runs in exactly one of the two launches: candidate mode owns counts
// within the cap, full mode owns rows past it.
static __global__ void SparkLmHeadRescoreArgmaxKernel(const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *candidate_ids, const uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t hidden_dimension, uint32_t candidate_count)
{
    extern __shared__ float rescore_shared[];
    float *hidden_shared = rescore_shared;
    __shared__ float best_score[SPARK_LM_CTA_WARPS];
    __shared__ uint32_t best_candidate[SPARK_LM_CTA_WARPS];
    uint32_t row = blockIdx.x;
    uint32_t warp = threadIdx.x / SPARK_LM_WARP_LANES;
    uint32_t lane = threadIdx.x % SPARK_LM_WARP_LANES;
    uint32_t count;
    uint32_t bound;
    uint32_t slot;
    uint32_t neuron;
    uint32_t element;
    float running_best = -3.0e38f;
    float score;
    uint32_t running_candidate = UINT32_MAX;
    float2 stage_pair;
    uint32_t use_screened_candidates;

    if (row >= row_count)
    {
        return;
    }
    count = candidate_counts[row];
    use_screened_candidates = count <= SPARK_LM_HEAD_SCREEN_CAP ? 1u : 0u;
    bound = use_screened_candidates != 0u ? count : candidate_count;
    for (element = threadIdx.x; element < (hidden_dimension >> 1u); element += blockDim.x)
    {
        stage_pair = SparkLmLoadBf16Pair(
            hidden_bf16,
            (((uint64_t)row * hidden_dimension) >> 1u) + element);
        hidden_shared[element << 1u] = stage_pair.x;
        hidden_shared[(element << 1u) + 1u] = stage_pair.y;
    }
    __syncthreads();

    for (slot = warp; slot < bound; slot += SPARK_LM_CTA_WARPS)
    {
        neuron = use_screened_candidates != 0u
            ? candidate_ids[((uint64_t)row * SPARK_LM_HEAD_SCREEN_CAP) + slot]
            : slot;
        score = SparkLmWarpReduceSum(
            SparkLmDotRowBf16(hidden_shared, head_weight_bf16, neuron, hidden_dimension, lane));
        score = __shfl_sync(0xffffffffu, score, 0);
        if (lane == 0u &&
            (score > running_best ||
             (score == running_best && neuron < running_candidate)))
        {
            running_best = score;
            running_candidate = neuron;
        }
    }
    SparkLmArgmaxReduce(running_best, running_candidate, best_score, best_candidate);
    if (threadIdx.x == 0u)
    {
        output_token_ids[row] = best_candidate[0];
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
                    entry / SPARK_LM_TILE][entry % SPARK_LM_TILE_N]);
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
	if ( (blockIdx.x * SPARK_LM_TILE) >= count )
		return;
	SparkLmExpertTileDispatch<GROUP_SIZE>(weight_format,payload,scale,input,grouped_rows != 0 ? grouped_rows + offset : 0,(void *)((uint8_t *)output_bf16 + ((uint64_t)offset * output_dimension * 2u)),count,input_dimension,output_dimension,blockIdx.x * SPARK_LM_TILE,blockIdx.y * SPARK_LM_TILE_N);
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

static inline cudaError_t SparkLmHostLaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension)
{
    dim3 tile_grid(
        (row_count + SPARK_LM_TILE - 1u) / SPARK_LM_TILE,
        (candidate_count + SPARK_LM_TILE_N - 1u) / SPARK_LM_TILE_N);
    uint32_t rescore_shared_bytes = hidden_dimension * (uint32_t)sizeof(float);

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
    SparkLmHeadScreenKernel<<<row_count, SPARK_LM_CTA_THREADS, 0u, stream>>>(
        hidden_bf16,
        logits_bf16,
        error_norm,
        candidate_ids,
        candidate_counts,
        row_count,
        candidate_count,
        hidden_dimension);
    SparkLmHeadRescoreArgmaxKernel<<<
        row_count,
        SPARK_LM_CTA_THREADS,
        rescore_shared_bytes,
        stream>>>(
            hidden_bf16,
            head_weight_bf16,
            candidate_ids,
            candidate_counts,
            output_token_ids,
            row_count,
            hidden_dimension,
            candidate_count);
    return cudaGetLastError();
}

static inline cudaError_t SparkLmHostLaunchMoeGroup(cudaStream_t stream, const uint32_t *pair_expert_ids, uint32_t pair_count, uint32_t expert_count, uint32_t experts_per_token, uint32_t *expert_offsets, uint32_t *grouped_rows, uint32_t *grouped_weight_slots, uint32_t *inverse_map)
{
	SparkLmMoeGroupKernel<<<1u,SPARK_LM_CTA_THREADS,0,stream>>>(pair_expert_ids,pair_count,expert_count,experts_per_token,expert_offsets,grouped_rows,grouped_weight_slots,inverse_map);
	return(cudaGetLastError());
}

// Size-aware batched dense linear/QKVO projection. The measured regimes
// (GLM52_B128_SCALING_ROOT_CAUSE) are three, not a continuum, so this
// deliberately breaks DRY into exactly three paths with two crossovers -
// more paths would not pay because per-token cost is flat within a regime:
//
//   B < SPARK_LM_TILE  (tiny): the scalar one-warp-per-neuron kernel. B1
//     decode is memory bound; a tensor tile over a near-empty M wastes the
//     fragment and the padded rows. Unchanged, correct here.
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
		dim3 scalar_grid(row_count,(output_dimension + SPARK_LM_CTA_WARPS - 1u) / SPARK_LM_CTA_WARPS);
		uint32_t shared_bytes = input_dimension * (uint32_t)sizeof(float);
		SparkLmLinearKernel<GROUP_SIZE,ACTIVATION_CODEC><<<scalar_grid,SPARK_LM_CTA_THREADS,shared_bytes,stream>>>(weight_format,weight_payload,weight_scale,input_bf16,output_bf16,row_count,input_dimension,output_dimension);
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

static inline cudaError_t SparkLmHostLaunchMoePairReduce(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint32_t row_count, uint32_t experts_per_token, uint32_t width)
{
	SparkLmMoePairReduceKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(slot_out_bf16,inverse_map,pair_weights_f32,accum_bf16,row_count,experts_per_token,width);
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
