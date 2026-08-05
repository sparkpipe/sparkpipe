#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_WEIGHT_CODEC_ABI_VERSION 1u

typedef enum SparkWeightScaleEncoding
{
	SPARK_WEIGHT_SCALE_ENCODING_NONE = 0,
	SPARK_WEIGHT_SCALE_ENCODING_F32 = 1,
	SPARK_WEIGHT_SCALE_ENCODING_UE4M3 = 2,
	SPARK_WEIGHT_SCALE_ENCODING_E8M0 = 3,
	SPARK_WEIGHT_SCALE_ENCODING_UE4M3_F32_GLOBAL = 4
} SparkWeightScaleEncoding;

typedef enum SparkWeightCodec
{
	SPARK_WEIGHT_CODEC_NONE = 0,
	SPARK_WEIGHT_CODEC_BF16 = 1,
	SPARK_WEIGHT_CODEC_INT6 = 2,
	SPARK_WEIGHT_CODEC_INT7 = 3,
	SPARK_WEIGHT_CODEC_INT8 = 4,
	SPARK_WEIGHT_CODEC_FP8_E4M3 = 5,
	SPARK_WEIGHT_CODEC_NVFP4_E2M1 = 6,
	SPARK_WEIGHT_CODEC_MXFP4_E2M1 = 7
} SparkWeightCodec;

static inline uint32_t SparkWeightCodecIsKnown(uint32_t codec)
{
	return(codec >= SPARK_WEIGHT_CODEC_BF16 && codec <= SPARK_WEIGHT_CODEC_MXFP4_E2M1 ? 1u : 0u);
}

static inline uint32_t SparkWeightCodecStoredBits(uint32_t codec)
{
	switch ( codec )
	{
	case SPARK_WEIGHT_CODEC_BF16:
		return(16u);
	case SPARK_WEIGHT_CODEC_INT6:
		return(6u);
	case SPARK_WEIGHT_CODEC_INT7:
		return(7u);
	case SPARK_WEIGHT_CODEC_INT8:
	case SPARK_WEIGHT_CODEC_FP8_E4M3:
		return(8u);
	case SPARK_WEIGHT_CODEC_NVFP4_E2M1:
	case SPARK_WEIGHT_CODEC_MXFP4_E2M1:
		return(4u);
	default:
		return(0u);
	}
}

static inline uint32_t SparkWeightCodecScaleEncoding(uint32_t codec)
{
	switch ( codec )
	{
	case SPARK_WEIGHT_CODEC_BF16:
		return(SPARK_WEIGHT_SCALE_ENCODING_NONE);
	case SPARK_WEIGHT_CODEC_INT6:
	case SPARK_WEIGHT_CODEC_INT7:
	case SPARK_WEIGHT_CODEC_INT8:
	case SPARK_WEIGHT_CODEC_FP8_E4M3:
		return(SPARK_WEIGHT_SCALE_ENCODING_F32);
	case SPARK_WEIGHT_CODEC_NVFP4_E2M1:
		return(SPARK_WEIGHT_SCALE_ENCODING_UE4M3_F32_GLOBAL);
	case SPARK_WEIGHT_CODEC_MXFP4_E2M1:
		return(SPARK_WEIGHT_SCALE_ENCODING_E8M0);
	default:
		return(UINT32_MAX);
	}
}

static inline uint32_t SparkWeightCodecScaleGroupSize(uint32_t codec)
{
	switch ( codec )
	{
	case SPARK_WEIGHT_CODEC_BF16:
		return(0u);
	case SPARK_WEIGHT_CODEC_INT6:
		return(32u);
	case SPARK_WEIGHT_CODEC_INT7:
	case SPARK_WEIGHT_CODEC_INT8:
	case SPARK_WEIGHT_CODEC_FP8_E4M3:
		return(128u);
	case SPARK_WEIGHT_CODEC_NVFP4_E2M1:
		return(16u);
	case SPARK_WEIGHT_CODEC_MXFP4_E2M1:
		return(32u);
	default:
		return(0u);
	}
}

static inline uint32_t SparkWeightScaleEncodingBytes(uint32_t encoding)
{
	switch ( encoding )
	{
	case SPARK_WEIGHT_SCALE_ENCODING_NONE:
		return(0u);
	case SPARK_WEIGHT_SCALE_ENCODING_F32:
		return(4u);
	case SPARK_WEIGHT_SCALE_ENCODING_UE4M3:
	case SPARK_WEIGHT_SCALE_ENCODING_E8M0:
	case SPARK_WEIGHT_SCALE_ENCODING_UE4M3_F32_GLOBAL:
		return(1u);
	default:
		return(0u);
	}
}

static inline uint64_t SparkWeightCodecPayloadBytes(uint32_t codec,uint64_t rows,uint64_t columns)
{
	uint64_t bits;
	bits = (uint64_t)SparkWeightCodecStoredBits(codec);
	if ( bits == 0u || rows == 0u || columns == 0u || rows > (UINT64_MAX / columns) || (rows * columns) > (UINT64_MAX / bits) )
		return(0u);
	bits *= rows * columns;
	return((bits + 7u) / 8u);
}

static inline uint64_t SparkWeightCodecScaleBytes(uint32_t codec,uint64_t groups,uint64_t rows,uint64_t columns)
{
	uint64_t entries,blocks;
	uint32_t block,bytes;
	block = SparkWeightCodecScaleGroupSize(codec);
	bytes = SparkWeightScaleEncodingBytes(SparkWeightCodecScaleEncoding(codec));
	if ( codec == SPARK_WEIGHT_CODEC_BF16 )
		return(0u);
	if ( block == 0u || bytes == 0u || groups == 0u || rows == 0u || columns == 0u )
		return(0u);
	blocks = (columns + block - 1u) / block;
	if ( groups > (UINT64_MAX / rows) || groups * rows > (UINT64_MAX / blocks) )
		return(0u);
	entries = groups * rows * blocks;
	if ( entries > (UINT64_MAX / bytes) )
		return(0u);
	entries *= bytes;
	if ( codec == SPARK_WEIGHT_CODEC_NVFP4_E2M1 )
	{
		if ( groups > (UINT64_MAX / sizeof(float)) || entries > UINT64_MAX - (groups * sizeof(float)) )
			return(0u);
		entries += groups * sizeof(float);
	}
	return(entries);
}

#ifdef __cplusplus
}
#endif
