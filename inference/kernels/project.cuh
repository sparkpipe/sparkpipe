#pragma once


#include "runtime/gemm.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/attn.cuh"
#include <stdint.h>

struct LmLowRankWeights
{
	const void *down_weight;
	const void *down_scale;
	const void *norm_weight;
	const void *up_weight;
	const void *up_scale;
	uint32_t input_dimension;
	uint32_t rank;
	uint32_t output_dimension;
	float norm_epsilon;
};

struct LmLowRankScratch
{
	uint8_t *input_codes;
	uint8_t *input_scales;
	uint16_t *compressed_bf16;
	uint8_t *compressed_codes;
	uint8_t *compressed_scales;
	const uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
};

template<class Format>
static LmScaleTensor LmProjectionWeightScale(
    const void *scale_data,
    uint32_t output_dimension,
    uint32_t input_dimension)
{
    if constexpr (Format::kScaleGroup == 0u)
    {
        return scale_data == 0
            ? LmScaleTensorNone()
            : LmScaleTensorInvalid(LM_SCALE_ENCODING_NONE);
    }
    else
    {
        return LmScaleTensorBlockF32(
            scale_data,
            1u,
            output_dimension,
            input_dimension,
            Format::kScaleGroup,
            Format::kScaleGroup);
    }
}

template<class Format>
static const void *LmProjectionPrepareInput(
    const uint16_t *input_bf16,
    const uint32_t *source_row_map,
    uint8_t *input_codes,
    uint8_t *input_scales,
    uint32_t rows,
    uint32_t input_dimension,
    uint32_t threads,
    LmScaleTensor *scale_out,
    cudaStream_t stream)
{
    if constexpr (Format::kScaleGroup == 0u)
    {
        *scale_out = LmScaleTensorNone();
        return input_bf16;
    }
    else
    {
        if (input_codes == 0 || input_scales == 0 ||
            (input_dimension % Format::kScaleGroup) != 0u)
        {
            *scale_out = LmScaleTensorInvalid(LM_SCALE_ENCODING_UE4M3);
            return 0;
        }
        LM_LAUNCH(
            (LmQuantiseRowsKernel<Format,256u>),
            dim3(rows,input_dimension / Format::kScaleGroup),
            threads,
            (Format::kScaleGroup + 8u) * sizeof(float),
            stream,
            input_bf16,
            source_row_map,
            input_codes,
            input_scales,
            rows,
            input_dimension);
        *scale_out = LmScaleTensorRowsUe4m3(
            input_scales,
            rows,
            input_dimension,
            Format::kScaleGroup);
        return input_codes;
    }
}

template<class Format>
static int32_t LmLowRankProject(
    const LmLowRankWeights *weights,
    const LmLowRankScratch *scratch,
    const uint16_t *input_bf16,
    uint16_t *output_bf16,
    uint32_t rows,
    uint32_t threads,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    LmGemmArguments gemm;
    const void *activation;
    int32_t status;

    if (weights == 0 || scratch == 0 || input_bf16 == 0 ||
        output_bf16 == 0 || rows == 0u || weights->input_dimension == 0u ||
        weights->rank == 0u || weights->output_dimension == 0u ||
        scratch->compressed_bf16 == 0 ||
        scratch->dense_row_offset == 0 || scratch->dense_tile_prefix == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }

    memset(&gemm,0,sizeof(gemm));
    activation = LmProjectionPrepareInput<Format>(
        input_bf16,
        0,
        scratch->input_codes,
        scratch->input_scales,
        rows,
        weights->input_dimension,
        threads,
        &gemm.scale_a,
        stream);
    if (activation == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    gemm.scale_b = LmProjectionWeightScale<Format>(
        weights->down_scale,
        weights->rank,
        weights->input_dimension);
    gemm.group_row_offset = scratch->dense_row_offset;
    gemm.group_tile_prefix = scratch->dense_tile_prefix;
    gemm.output_bf16 = scratch->compressed_bf16;
    status = LmGemmLaunch<
        Format,128u,Format::kTileK,LM_PIPELINE_STAGES,8u>(
            &gemm,
            activation,
            weights->down_weight,
            rows,
            rows,
            1u,
            1u,
            weights->input_dimension,
            weights->rank,
            multiprocessors,
            false,
            stream);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }

    LM_LAUNCH(
        (LmFusedResidualRmsNormKernel<256u,uint16_t>),
        rows,
        threads,
        (weights->rank + 8u) * sizeof(float),
        stream,
        scratch->compressed_bf16,
        0,
        (const uint16_t *)weights->norm_weight,
        0,
        scratch->compressed_bf16,
        weights->rank,
        weights->rank,
        weights->norm_epsilon);

    memset(&gemm,0,sizeof(gemm));
    activation = LmProjectionPrepareInput<Format>(
        scratch->compressed_bf16,
        0,
        scratch->compressed_codes,
        scratch->compressed_scales,
        rows,
        weights->rank,
        threads,
        &gemm.scale_a,
        stream);
    if (activation == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    gemm.scale_b = LmProjectionWeightScale<Format>(
        weights->up_scale,
        weights->output_dimension,
        weights->rank);
    gemm.group_row_offset = scratch->dense_row_offset;
    gemm.group_tile_prefix = scratch->dense_tile_prefix;
    gemm.output_bf16 = output_bf16;
    return LmGemmLaunch<
        Format,128u,Format::kTileK,LM_PIPELINE_STAGES,8u>(
            &gemm,
            activation,
            weights->up_weight,
            rows,
            rows,
            1u,
            1u,
            weights->rank,
            weights->output_dimension,
            multiprocessors,
            false,
            stream);
}

struct LmAbsorbedWeights
{
	const void *query_latent_weight;
	const void *query_latent_scale;
	const void *query_rope_weight;
	const void *query_rope_scale;
	const void *key_rope_weight;
	const void *key_rope_scale;
	const void *kv_latent_weight;
	const void *kv_latent_scale;
	uint32_t input_dimension;
	uint32_t query_latent_dimension;
	uint32_t query_rope_dimension;
	uint32_t key_rope_dimension;
	uint32_t kv_latent_dimension;
};

struct LmAbsorbedOutputs
{
	uint16_t *query_latent_bf16;
	uint16_t *query_rope_bf16;
	uint16_t *key_rope_bf16;
	uint16_t *kv_latent_bf16;
};


template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmJoinRowsKernel(
    const uint16_t *__restrict__ left_bf16,
    uint32_t left_dimension,
    const uint16_t *__restrict__ right_bf16,
    uint32_t right_dimension,
    uint16_t *__restrict__ output_bf16,
    uint32_t row_count)
{
    uint32_t row = blockIdx.x;
    uint32_t output_dimension = left_dimension + right_dimension;
    uint32_t column;

    if (row >= row_count)
    {
        return;
    }
    for (column = threadIdx.x; column < output_dimension; column += THREADS)
    {
        if (column < left_dimension)
        {
            output_bf16[((uint64_t)row * output_dimension) + column] =
                left_bf16[((uint64_t)row * left_dimension) + column];
        }
        else
        {
            uint32_t right_column = column - left_dimension;
            output_bf16[((uint64_t)row * output_dimension) + column] =
                right_bf16[((uint64_t)row * right_dimension) + right_column];
        }
    }
}

template<class Format>
static int32_t LmAbsorbedProject(
    const LmAbsorbedWeights *weights,
    const LmAbsorbedOutputs *out,
    const uint16_t *input_bf16,
    const uint8_t *input_codes,
    const LmScaleTensor *input_scale,
    const uint32_t *dense_row_offset,
    uint32_t *dense_tile_prefix,
    uint32_t rows,
    uint32_t multiprocessors,
    cudaStream_t stream)
{
    struct LmAbsorbedPass
    {
        const void *weight;
        const void *scale;
        uint16_t *output;
        uint32_t width;
    };
    const LmAbsorbedPass pass[4] =
    {
        { weights->query_latent_weight, weights->query_latent_scale,
            out->query_latent_bf16, weights->query_latent_dimension },
        { weights->query_rope_weight, weights->query_rope_scale,
            out->query_rope_bf16, weights->query_rope_dimension },
        { weights->key_rope_weight, weights->key_rope_scale,
            out->key_rope_bf16, weights->key_rope_dimension },
        { weights->kv_latent_weight, weights->kv_latent_scale,
            out->kv_latent_bf16, weights->kv_latent_dimension }
    };
    const void *activation;
    LmGemmArguments gemm;
    int32_t status;
    uint32_t index;

    if (weights == 0 || out == 0 || input_bf16 == 0 ||
        dense_row_offset == 0 || dense_tile_prefix == 0 || rows == 0u)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    if constexpr (Format::kScaleGroup == 0u)
    {
        activation = input_bf16;
    }
    else
    {
        if (input_codes == 0 || input_scale == 0 ||
            LmScaleTensorIsValid(input_scale) == 0u)
        {
            return LM_LAUNCH_ERR_SHAPE;
        }
        activation = input_codes;
    }

    for (index = 0u; index < 4u; ++index)
    {
        if (pass[index].weight == 0 || pass[index].output == 0 ||
            pass[index].width == 0u)
        {
            return LM_LAUNCH_ERR_SHAPE;
        }
        memset(&gemm,0,sizeof(gemm));
        gemm.scale_a = Format::kScaleGroup == 0u
            ? LmScaleTensorNone()
            : *input_scale;
        gemm.scale_b = LmProjectionWeightScale<Format>(
            pass[index].scale,
            pass[index].width,
            weights->input_dimension);
        gemm.group_row_offset = dense_row_offset;
        gemm.group_tile_prefix = dense_tile_prefix;
        gemm.output_bf16 = pass[index].output;
        status = LmGemmLaunch<
            Format,128u,Format::kTileK,LM_PIPELINE_STAGES,8u>(
                &gemm,
                activation,
                pass[index].weight,
                rows,
                rows,
                1u,
                1u,
                weights->input_dimension,
                pass[index].width,
                multiprocessors,
                false,
                stream);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
    }
    return LM_LAUNCH_OK;
}

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
void LmRopePerHeadKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t heads, uint32_t head_dimension, uint32_t rope_dimension, float theta)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint32_t half = rope_dimension / 2u;
	uint64_t base = (((uint64_t)row * heads) + head) * head_dimension
		+ (head_dimension - rope_dimension);
	float position = (float)positions[row];
	for (index = threadIdx.x; index < half; index += THREADS)
		LmRopeRotate<PAIRING>(rows_bf16,base,index,half,
			position * __powf(theta,-2.0f * (float)index / (float)rope_dimension));
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

template<
	uint32_t THREADS,
	LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmExtractRopePerHeadKernel(
	const uint16_t *__restrict__ input_bf16,
	uint16_t *__restrict__ output_bf16,
	const uint32_t *__restrict__ positions,
	uint32_t heads,
	uint32_t input_head_dimension,
	uint32_t input_offset,
	uint32_t rope_dimension,
	float theta)
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
		angle = position * __powf(
			theta,-2.0f * (float)index / (float)rope_dimension);
		LmRopePair(&low,&high,angle);
		output_bf16[output_base + low_offset] = LmFloatToBf16(low);
		output_bf16[output_base + high_offset] = LmFloatToBf16(high);
	}
}
