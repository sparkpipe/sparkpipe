#include <assert.h>
#include <stdint.h>

#include "sparkpipe/spark_weight_codec.h"

typedef struct SparkTestWeightCodecCase
{
	uint32_t codec;
	uint32_t stored_bits;
	uint32_t scale_encoding;
	uint32_t scale_group;
} SparkTestWeightCodecCase;

static void SparkTestCodecCases(void)
{
	static const SparkTestWeightCodecCase cases[] = {
		{SPARK_WEIGHT_CODEC_BF16,16u,SPARK_WEIGHT_SCALE_ENCODING_NONE,0u},
		{SPARK_WEIGHT_CODEC_INT6,6u,SPARK_WEIGHT_SCALE_ENCODING_F32,32u},
		{SPARK_WEIGHT_CODEC_INT7,7u,SPARK_WEIGHT_SCALE_ENCODING_F32,128u},
		{SPARK_WEIGHT_CODEC_INT8,8u,SPARK_WEIGHT_SCALE_ENCODING_F32,128u},
		{SPARK_WEIGHT_CODEC_FP8_E4M3,8u,SPARK_WEIGHT_SCALE_ENCODING_F32,128u},
		{SPARK_WEIGHT_CODEC_NVFP4_E2M1,4u,SPARK_WEIGHT_SCALE_ENCODING_UE4M3_F32_GLOBAL,16u},
		{SPARK_WEIGHT_CODEC_MXFP4_E2M1,4u,SPARK_WEIGHT_SCALE_ENCODING_E8M0,32u}
	};
	uint32_t index;
	for ( index = 0u; index < sizeof(cases) / sizeof(cases[0]); ++index )
	{
		assert(SparkWeightCodecIsKnown(cases[index].codec) != 0u);
		assert(SparkWeightCodecStoredBits(cases[index].codec) == cases[index].stored_bits);
		assert(SparkWeightCodecScaleEncoding(cases[index].codec) == cases[index].scale_encoding);
		assert(SparkWeightCodecScaleGroupSize(cases[index].codec) == cases[index].scale_group);
	}
	assert(SparkWeightCodecIsKnown(SPARK_WEIGHT_CODEC_NONE) == 0u);
	assert(SparkWeightCodecStoredBits(UINT32_MAX) == 0u);
}

static void SparkTestByteGeometry(void)
{
	assert(SparkWeightCodecPayloadBytes(SPARK_WEIGHT_CODEC_INT6,2u,128u) == 192u);
	assert(SparkWeightCodecPayloadBytes(SPARK_WEIGHT_CODEC_INT7,2u,128u) == 224u);
	assert(SparkWeightCodecPayloadBytes(SPARK_WEIGHT_CODEC_INT8,2u,128u) == 256u);
	assert(SparkWeightCodecPayloadBytes(SPARK_WEIGHT_CODEC_NVFP4_E2M1,2u,128u) == 128u);
	assert(SparkWeightCodecScaleBytes(SPARK_WEIGHT_CODEC_INT6,3u,2u,128u) == 96u);
	assert(SparkWeightCodecScaleBytes(SPARK_WEIGHT_CODEC_INT7,3u,2u,128u) == 24u);
	assert(SparkWeightCodecScaleBytes(SPARK_WEIGHT_CODEC_NVFP4_E2M1,3u,2u,128u) == 60u);
	assert(SparkWeightCodecScaleBytes(SPARK_WEIGHT_CODEC_MXFP4_E2M1,3u,2u,128u) == 24u);
	assert(SparkWeightCodecPayloadBytes(SPARK_WEIGHT_CODEC_NONE,1u,1u) == 0u);
}

static void SparkTestActivationCodecs(void)
{
	assert(SparkActivationCodecIsKnown(SPARK_ACTIVATION_CODEC_NONE) != 0u);
	assert(SparkActivationCodecIsKnown(SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0) != 0u);
	assert(SparkActivationCodecIsKnown(UINT32_MAX) == 0u);
	assert(SparkActivationCodecGroupSize(SPARK_ACTIVATION_CODEC_NONE) == 0u);
	assert(SparkActivationCodecGroupSize(SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0) == 128u);
}

int main(void)
{
	SparkTestCodecCases();
	SparkTestByteGeometry();
	SparkTestActivationCodecs();
	return(0);
}
