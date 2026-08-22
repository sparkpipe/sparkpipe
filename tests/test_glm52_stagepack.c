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
	payload_bytes = SparkStagePackPayloadBytes(&shape);
	assert(payload_bytes == ((elements * bits) / 8u));
	scale_bytes = SparkStagePackScaleBytes(&shape);
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

/* Resolve one kind at one layer for a TP degree; picks a layer that is valid
 * for the kind's placement class (global / dense / routed / indexer). */
static int32_t SparkTestResolveKind(uint32_t tensor_kind,uint32_t expert_codec,uint32_t tp_degree,SparkGlm52StagePackTensorShape *shape)
{
	uint32_t layer;
	if ( SparkGlm52StagePackKindIsGlobal(tensor_kind) != 0u )
		layer = SPARK_GLM52_STAGEPACK_GLOBAL_LAYER;
	else if ( SparkGlm52StagePackKindIsRouted(tensor_kind) != 0u )
		layer = SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER;
	else
		layer = 0u; /* dense and per-layer kinds are valid here; layer 0 has a full indexer */
	return(SparkGlm52StagePackExpectedShape(tensor_kind,layer,expert_codec,tp_degree,shape));
}

/* The TP sharding policy is duplicated in tools/glm52_resident_stagepack.py
 * (shard="rows"/"cols" call sites); the C table and that mirror are bound
 * only by convention, so pin the table here: any policy edit must update
 * this test AND the Python packer in the same change. Also proves each
 * sharded axis stays degree-divisible and group_count never shards. */
static void SparkTestTpShardPolicy(void)
{
	SparkGlm52StagePackTensorShape base,sharded;
	static const uint32_t row_sharded[] = {
		SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING,
		SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD,
		SPARK_GLM52_STAGEPACK_TENSOR_Q_B,
		SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP,
		SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE,
		SPARK_GLM52_STAGEPACK_TENSOR_SHARED_GATE_UP
	};
	static const uint32_t column_sharded[] = {
		SPARK_GLM52_STAGEPACK_TENSOR_ATTN_OUTPUT,
		SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN,
		SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN,
		SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN
	};
	uint32_t kind,degree,index;
	for (kind = 0u; kind < SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT; kind++)
	{
		assert(SparkTestResolveKind(kind,SPARK_WEIGHT_CODEC_INT8,1u,&base) == 0);
		for (degree = 2u; degree <= 8u; degree *= 2u)
		{
			assert(SparkTestResolveKind(kind,SPARK_WEIGHT_CODEC_INT8,degree,&sharded) == 0);
			assert(sharded.group_count == base.group_count);
			switch ( SparkGlm52StagePackTpShardPolicy(kind) )
			{
			case SPARK_STAGE_PACK_SHARD_ROWS:
				assert(sharded.rows == base.rows / degree && sharded.columns == base.columns);
				break;
			case SPARK_STAGE_PACK_SHARD_COLUMNS:
				assert(sharded.rows == base.rows && sharded.columns == base.columns / degree);
				break;
			default:
				assert(sharded.rows == base.rows && sharded.columns == base.columns);
				break;
			}
		}
	}
	for (index = 0u; index < sizeof(row_sharded) / sizeof(row_sharded[0]); index++)
		assert(SparkGlm52StagePackTpShardPolicy(row_sharded[index]) == SPARK_STAGE_PACK_SHARD_ROWS);
	for (index = 0u; index < sizeof(column_sharded) / sizeof(column_sharded[0]); index++)
		assert(SparkGlm52StagePackTpShardPolicy(column_sharded[index]) == SPARK_STAGE_PACK_SHARD_COLUMNS);
	/* Shared mechanics: zero degree rejected, non-divisible axis rejected
	 * before dividing, unknown policy rejected. */
	assert(SparkGlm52StagePackExpectedShape(SPARK_GLM52_STAGEPACK_TENSOR_Q_B,0u,SPARK_WEIGHT_CODEC_INT8,0u,&sharded) == -1);
	assert(SparkStagePackApplyShard(&base,SPARK_STAGE_PACK_SHARD_ROWS,0u) == -1);
	base.rows = 3u;
	assert(SparkStagePackApplyShard(&base,SPARK_STAGE_PACK_SHARD_ROWS,4u) == -2);
	assert(base.rows == 3u);
	base.columns = 5u;
	assert(SparkStagePackApplyShard(&base,SPARK_STAGE_PACK_SHARD_COLUMNS,4u) == -2);
	assert(base.columns == 5u);
	assert(SparkStagePackApplyShard(&base,7u,1u) == -3);
}

int main(void)
{
	SparkTestHeaderLayout();
	SparkTestAllExpertCodecs();
	SparkTestShapeContract();
	SparkTestTpShardPolicy();
	return(0);
}
