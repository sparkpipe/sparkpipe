#pragma once

#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/int6.cuh"
#include "inference/kernels/formats/int7.cuh"
#include "inference/kernels/formats/int8.cuh"
#include "inference/kernels/formats/mxfp4.cuh"
#include "inference/kernels/formats/nvfp4.cuh"
#include "inference/kernels/scale.cuh"
#include "sparkpipe/spark_weight_codec.h"

template<uint32_t Codec>
struct LmWeightCodec;

#define LM_WEIGHT_CODEC(code,format,scale_encoding) \
	template<> struct LmWeightCodec<code> \
	{ \
		using Format = format; \
		static constexpr uint32_t kCodec = code; \
		static constexpr uint32_t kScaleEncoding = scale_encoding; \
	}

LM_WEIGHT_CODEC(SPARK_WEIGHT_CODEC_BF16,LmBf16Format,LM_SCALE_ENCODING_NONE);
LM_WEIGHT_CODEC(SPARK_WEIGHT_CODEC_INT6,LmInt6,LM_SCALE_ENCODING_F32);
LM_WEIGHT_CODEC(SPARK_WEIGHT_CODEC_INT7,LmInt7,LM_SCALE_ENCODING_F32);
LM_WEIGHT_CODEC(SPARK_WEIGHT_CODEC_INT8,LmInt8,LM_SCALE_ENCODING_F32);
LM_WEIGHT_CODEC(SPARK_WEIGHT_CODEC_FP8_E4M3,LmFp8,LM_SCALE_ENCODING_F32);
LM_WEIGHT_CODEC(SPARK_WEIGHT_CODEC_NVFP4_E2M1,LmNvfp4,LM_SCALE_ENCODING_UE4M3_F32_GLOBAL);
LM_WEIGHT_CODEC(SPARK_WEIGHT_CODEC_MXFP4_E2M1,LmMxfp4,LM_SCALE_ENCODING_UE8M0);

#undef LM_WEIGHT_CODEC

static_assert(SPARK_WEIGHT_SCALE_ENCODING_NONE == LM_SCALE_ENCODING_NONE,
	"public and CUDA scale encodings disagree");
static_assert(SPARK_WEIGHT_SCALE_ENCODING_F32 == LM_SCALE_ENCODING_F32,
	"public and CUDA scale encodings disagree");
static_assert(SPARK_WEIGHT_SCALE_ENCODING_UE4M3 == LM_SCALE_ENCODING_UE4M3,
	"public and CUDA scale encodings disagree");
static_assert(SPARK_WEIGHT_SCALE_ENCODING_E8M0 == LM_SCALE_ENCODING_UE8M0,
	"public and CUDA scale encodings disagree");
static_assert(SPARK_WEIGHT_SCALE_ENCODING_UE4M3_F32_GLOBAL == LM_SCALE_ENCODING_UE4M3_F32_GLOBAL,
	"public and CUDA scale encodings disagree");

template<uint32_t Codec>
static LmScaleTensor LmWeightCodecScaleTensor(
	const void *data,
	uint32_t group_count,
	uint32_t row_count,
	uint32_t input_dimension)
{
	using Trait = LmWeightCodec<Codec>;
	using Format = typename Trait::Format;
	if constexpr ( Trait::kScaleEncoding == LM_SCALE_ENCODING_NONE )
		return(data == 0 ? LmScaleTensorNone() : LmScaleTensorInvalid(Trait::kScaleEncoding));
	if constexpr ( Trait::kScaleEncoding == LM_SCALE_ENCODING_UE4M3_F32_GLOBAL )
		return(LmScaleTensorBlockUe4m3F32Global(
			(const uint8_t *)data + ((uint64_t)group_count * sizeof(float)),
			data,
			group_count,
			row_count,
			input_dimension,
			Format::kScaleGroup));
	return(LmScaleTensorBuild(data,Trait::kScaleEncoding,group_count,row_count,input_dimension,1u,Format::kScaleGroup));
}
