#include <assert.h>
#include <stdint.h>

#include "spark_glm52_stagepack_format.h"

static void SparkTestHeaderLayout(void)
{
	assert(SPARK_GLM52_STAGEPACK_HEADER_BYTES == 264u);
	assert(SPARK_GLM52_STAGEPACK_ENTRY_BYTES == 64u);
	assert(SPARK_GLM52_STAGEPACK_FORMAT_VERSION == 3u);
	assert(SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES == 256u);
}

static void SparkTestExpertCodec(uint32_t codec,uint32_t bits,uint32_t group)
{
	SparkGlm52StagePackTensorShape shape;
	uint64_t elements,payload_bytes,scale_bytes,expected_scale_bytes;
	assert(SparkGlm52StagePackExpectedShape(
		SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE,
		3u,
		codec,
		1u,
		&shape) == 0);
	assert(shape.payload_type == SPARK_GLM52_STAGEPACK_PAYLOAD_PACKED_WEIGHT);
	assert(shape.weight_codec == codec);
	assert(SparkWeightCodecStoredBits(codec) == bits);
	assert(SparkWeightCodecScaleGroupSize(codec) == group);
	elements = (uint64_t)shape.group_count * shape.rows * shape.columns;
	payload_bytes = SparkGlm52StagePackExpectedPayloadBytes(&shape);
	assert(payload_bytes == ((elements * bits) / 8u));
	scale_bytes = SparkGlm52StagePackExpectedScaleBytes(&shape);
	expected_scale_bytes = (uint64_t)shape.group_count * shape.rows *
		(shape.columns / group) *
		(codec >= SPARK_WEIGHT_CODEC_NVFP4_E2M1 ? 1u : 4u);
	if ( codec == SPARK_WEIGHT_CODEC_NVFP4_E2M1 )
		expected_scale_bytes += (uint64_t)shape.group_count * sizeof(float);
	assert(scale_bytes == expected_scale_bytes);
}

static void SparkTestAllExpertCodecs(void)
{
	SparkTestExpertCodec(SPARK_WEIGHT_CODEC_INT6,6u,32u);
	SparkTestExpertCodec(SPARK_WEIGHT_CODEC_INT7,7u,128u);
	SparkTestExpertCodec(SPARK_WEIGHT_CODEC_INT8,8u,128u);
	SparkTestExpertCodec(SPARK_WEIGHT_CODEC_FP8_E4M3,8u,128u);
	SparkTestExpertCodec(SPARK_WEIGHT_CODEC_NVFP4_E2M1,4u,16u);
	SparkTestExpertCodec(SPARK_WEIGHT_CODEC_MXFP4_E2M1,4u,32u);
}

static void SparkTestShapeContract(void)
{
	SparkGlm52StagePackTensorShape shape;
	assert(SparkGlm52StagePackExpectedShape(
		SPARK_GLM52_STAGEPACK_TENSOR_Q_A,
		0u,
		SPARK_WEIGHT_CODEC_INT8,
		1u,
		&shape) == 0);
	assert(shape.payload_type == SPARK_GLM52_STAGEPACK_PAYLOAD_BF16);
	assert(shape.weight_codec == SPARK_WEIGHT_CODEC_BF16);
	assert(shape.rows == SPARK_GLM52_MODEL_QUERY_A_DIMENSION);
	assert(shape.columns == SPARK_GLM52_MODEL_HIDDEN_DIMENSION);
	assert(SparkGlm52StagePackExpectedShape(
		SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN,
		3u,
		SPARK_WEIGHT_CODEC_BF16,
		1u,
		&shape) == 0);
	/* The bf16 expert arm: native-precision experts carry the BF16
	 * payload and NO scale plane (the glm53full bf16 pack contract). */
	assert(shape.payload_type == SPARK_GLM52_STAGEPACK_PAYLOAD_BF16);
	assert(shape.weight_codec == SPARK_WEIGHT_CODEC_BF16);
	assert(shape.scale_encoding == SPARK_WEIGHT_SCALE_ENCODING_NONE);
	assert(shape.group_count == SPARK_GLM52_MODEL_MOE_EXPERT_COUNT);
	assert(shape.rows == SPARK_GLM52_MODEL_HIDDEN_DIMENSION);
	assert(shape.columns == SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION);
	assert(SparkGlm52StagePackExpectedPayloadBytes(&shape) ==
		(uint64_t)SPARK_GLM52_MODEL_MOE_EXPERT_COUNT *
		SPARK_GLM52_MODEL_HIDDEN_DIMENSION *
		SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION * 2u);
	assert(SparkGlm52StagePackExpectedScaleBytes(&shape) == 0u);
	assert(SparkGlm52StagePackExpectedShape(
		SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN,
		3u,
		SPARK_WEIGHT_CODEC_NONE,
		1u,
		&shape) == -5);
	assert(SparkGlm52StagePackExpectedShape(
		SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q,
		4u,
		SPARK_WEIGHT_CODEC_INT8,
		1u,
		&shape) == -3);
	assert(SparkGlm52StagePackExpectedShape(
		SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN,
		3u,
		SPARK_WEIGHT_CODEC_INT8,
		1u,
		&shape) == -4);
}

int main(void)
{
	SparkTestHeaderLayout();
	SparkTestAllExpertCodecs();
	SparkTestShapeContract();
	return(0);
}
