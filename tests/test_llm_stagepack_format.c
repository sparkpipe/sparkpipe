#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "llm_defines.h"
#include "common/common_stagepack_format_ext.h"

int32_t SparkTestNegativeControls(void);

static void SparkTestWireLayout(void)
{
	assert(SPARK_STAGEPACK_ENTRY_BYTES == 56u);
	assert(sizeof(SparkStagePackEntry) == SPARK_STAGEPACK_ENTRY_BYTES);
	assert(SPARK_STAGEPACK_HEADER_BYTES == 120u);
	assert(SPARK_STAGEPACK_PAYLOAD_ALIGNMENT == 256u);
	assert(SPARK_LLM_STAGEPACK_MAGIC == 0x50533451u);
	assert(SPARK_LLM_STAGEPACK_FORMAT_VERSION == 2u);
	assert(SPARK_STAGEPACK_LAYER_GLOBAL == UINT32_MAX);
	assert(SPARK_STAGEPACK_LAYER_MTP == UINT32_MAX - 1u);
	assert(SparkStagePackFamilySpecCheck(&SparkLlmStagePackFamilySpec) == 0);
	assert(SparkLlmStagePackKindTable.row_count == 36u);
}

static void SparkTestExpectedTensorCounts(void)
{
	const SparkStagePackFamilySpec *spec = &SparkLlmStagePackFamilySpec;
	assert(SparkStagePackFamilyExpectedTensorCount(spec,0u,4u,0u) == 101u);
	assert(SparkStagePackFamilyExpectedTensorCount(spec,0u,4u,1u) == 111u);
	assert(SparkStagePackFamilyExpectedTensorCount(spec,0u,48u,0u) == 1236u);
	assert(SparkStagePackFamilyExpectedTensorCount(spec,0u,48u,1u) == 1246u);
	assert(SparkStagePackFamilyExpectedTensorCount(spec,4u,4u,0u) == 100u);
	assert(SparkStagePackFamilyExpectedTensorCount(spec,4u,4u,1u) == 100u);
	assert(SparkStagePackFamilyExpectedTensorCount(spec,4u,48u,0u) == 1200u);
	assert(SparkStagePackFamilyExpectedTensorCount(spec,4u,48u,1u) == 1200u);
}

static void SparkTestExpectedGeometry(void)
{
	SparkStagePackHeaderCommon header;
	SparkStagePackFamilyExpectedGeometry(&SparkLlmStagePackFamilySpec,&header,0u,4u,1u);
	assert(header.magic == 0x50533451u);
	assert(header.format_version == 2u);
	assert(header.header_bytes == 120u);
	assert(header.directory_entry_bytes == 56u);
	assert(header.tensor_count == 111u);
	assert(header.hidden_dimension == 2560u);
	assert(header.layer_count == 4u);
	assert(header.total_layer_count == 48u);
	assert(header.attention_period == 4u);
	assert(header.full_attention_phase == 3u);
	assert(header.gdn_key_head_count == 16u);
	assert(header.gdn_value_head_count == 48u);
	assert(header.gdn_head_key_dimension == 128u);
	assert(header.gdn_head_value_dimension == 128u);
	assert(header.gdn_conv_kernel == 4u);
	assert(header.attn_query_head_count == 24u);
	assert(header.attn_kv_head_count == 2u);
	assert(header.attn_head_dimension == 256u);
	assert(header.attn_rope_dimension == 64u);
	assert(header.routed_expert_count == 512u);
	assert(header.experts_per_token == 10u);
	assert(header.expert_intermediate_dimension == 640u);
	assert(header.output_vocab_count == 248320u);
	assert(header.mxfp4_group_size == 32u);
	assert(header.mtp_layer_count == SPARK_LLM_MTP_LAYER_COUNT);
	assert(SparkStagePackHeaderMatches(&header,&header) == 0);
}

static void SparkTestShapes(void)
{
	const SparkStagePackFamilySpec *spec = &SparkLlmStagePackFamilySpec;
	const SparkStagePackKindTable *table = &SparkLlmStagePackKindTable;
	SparkStagePackTensorShape shape;
	struct
	{
		uint32_t kind;
		uint32_t rows;
		uint32_t columns;
		uint32_t natural_format;
		uint32_t layer_class;
	} vectors[] =
	{
		{SPARK_LLM_STAGEPACK_TENSOR_EMBEDDING,248320u,2560u,0u,0u},
		{SPARK_LLM_STAGEPACK_TENSOR_FINAL_NORM,1u,10240u,0u,0u},
		{SPARK_LLM_STAGEPACK_TENSOR_LM_HEAD,248320u,2560u,0u,0u},
		{SPARK_LLM_STAGEPACK_TENSOR_ATTN_QUERY,12288u,2560u,0u,3u},
		{SPARK_LLM_STAGEPACK_TENSOR_ATTN_KEY,512u,2560u,0u,3u},
		{SPARK_LLM_STAGEPACK_TENSOR_ATTN_OUTPUT,2560u,6144u,0u,3u},
		{SPARK_LLM_STAGEPACK_TENSOR_ATTN_QUERY_NORM,1u,256u,0u,3u},
		{SPARK_LLM_STAGEPACK_TENSOR_ATTN_HC_DOWN,320u,10240u,0u,1u},
		{SPARK_LLM_STAGEPACK_TENSOR_ATTN_HC_INJECT,4u,10240u,0u,1u},
		{SPARK_LLM_STAGEPACK_TENSOR_MLP_HC_UP,10240u,320u,0u,1u},
		{SPARK_LLM_STAGEPACK_TENSOR_INDEXER_QK,640u,2560u,0u,3u},
		{SPARK_LLM_STAGEPACK_TENSOR_INDEXER_K_NORM,1u,128u,0u,3u},
		{SPARK_LLM_STAGEPACK_TENSOR_MIXER_DOWN,320u,10240u,0u,0u},
		{SPARK_LLM_STAGEPACK_TENSOR_MTP_FC,2560u,5120u,0u,0u},
		{SPARK_LLM_STAGEPACK_TENSOR_MTP_MIXER_UP,10240u,320u,0u,0u},
		{SPARK_LLM_STAGEPACK_TENSOR_PLE_KEY,10240u,2560u,0u,4u},
		{SPARK_LLM_STAGEPACK_TENSOR_PLE_CONV,10240u,4u,0u,4u},
		{SPARK_LLM_STAGEPACK_TENSOR_PLE_MULTIPLIERS,1u,3u,7u,4u},
		{SPARK_LLM_STAGEPACK_TENSOR_PLE_NGRAM,320001536u,160u,0u,4u},
		{SPARK_LLM_STAGEPACK_TENSOR_PLE_HEAD_VOCABS,1u,16u,7u,4u},
		{SPARK_STAGEPACK_TENSOR_MOE_GATE,512u,2560u,0u,1u},
		{SPARK_STAGEPACK_TENSOR_MOE_W1,327680u,2560u,4u,1u},
		{SPARK_STAGEPACK_TENSOR_GDN_QKV,10240u,2560u,0u,2u},
		{SPARK_STAGEPACK_TENSOR_GDN_BETA,48u,2560u,0u,2u},
		{SPARK_STAGEPACK_TENSOR_GDN_A_LOG,1u,48u,1u,2u},
		{SPARK_STAGEPACK_TENSOR_GDN_NORM,1u,128u,0u,2u},
		{SPARK_STAGEPACK_TENSOR_ATTENTION_NORM,1u,10240u,0u,1u}
	};
	uint32_t index;
	for (index = 0u; index < sizeof(vectors) / sizeof(vectors[0]); index++)
	{
		assert(SparkStagePackFamilyTensorShapeOf(spec,table,vectors[index].kind,&shape) == 0);
		assert(shape.rows == vectors[index].rows);
		assert(shape.columns == vectors[index].columns);
		assert(shape.natural_format == vectors[index].natural_format);
		assert(shape.layer_class == vectors[index].layer_class);
	}
}

static void SparkTestResolvedShape(void)
{
	const SparkStagePackFamilySpec *spec = &SparkLlmStagePackFamilySpec;
	const SparkStagePackKindTable *table = &SparkLlmStagePackKindTable;
	SparkStagePackTensorShape shape;
	assert(SparkStagePackFamilyResolvedShape(spec,table,SPARK_LLM_STAGEPACK_TENSOR_EMBEDDING,SPARK_STAGEPACK_LAYER_GLOBAL,1u,&shape) == 0);
	assert(SparkStagePackFamilyResolvedShape(spec,table,SPARK_LLM_STAGEPACK_TENSOR_EMBEDDING,0u,0u,&shape) == -2);
	assert(SparkStagePackFamilyResolvedShape(spec,table,SPARK_LLM_STAGEPACK_TENSOR_ATTN_QUERY,3u,0u,&shape) == 0);
	assert(SparkStagePackFamilyResolvedShape(spec,table,SPARK_LLM_STAGEPACK_TENSOR_ATTN_QUERY,0u,0u,&shape) == -5);
	assert(SparkStagePackFamilyResolvedShape(spec,table,SPARK_STAGEPACK_TENSOR_GDN_QKV,0u,0u,&shape) == 0);
	assert(SparkStagePackFamilyResolvedShape(spec,table,SPARK_STAGEPACK_TENSOR_GDN_QKV,3u,0u,&shape) == -4);
	assert(SparkStagePackFamilyResolvedShape(spec,table,SPARK_STAGEPACK_TENSOR_GDN_QKV,48u,0u,&shape) == -3);
	assert(SparkStagePackFamilyResolvedShape(spec,table,SPARK_STAGEPACK_TENSOR_GDN_QKV,SPARK_STAGEPACK_LAYER_MTP,0u,&shape) == -6);
	assert(SparkStagePackFamilyResolvedShape(spec,table,SPARK_LLM_STAGEPACK_TENSOR_ATTN_HC_DOWN,SPARK_STAGEPACK_LAYER_MTP,0u,&shape) == 0);
	assert(SparkStagePackFamilyResolvedShape(spec,table,SPARK_LLM_STAGEPACK_TENSOR_EMBEDDING,SPARK_STAGEPACK_LAYER_MTP,0u,&shape) == -6);
	assert(SparkStagePackFamilyResolvedShape(spec,table,SPARK_LLM_STAGEPACK_TENSOR_PLE_CONV,SPARK_LLM_PLE_LAYER_INDEX,0u,&shape) == 0);
	assert(SparkStagePackFamilyResolvedShape(spec,table,SPARK_LLM_STAGEPACK_TENSOR_PLE_CONV,2u,0u,&shape) == -7);
}

static void SparkTestNarrowShape(void)
{
	const SparkStagePackFamilySpec *spec = &SparkLlmStagePackFamilySpec;
	const SparkStagePackKindTable *table = &SparkLlmStagePackKindTable;
	SparkStagePackTensorShape shape;
	assert(SparkStagePackFamilyTensorShapeOf(spec,table,SPARK_LLM_STAGEPACK_TENSOR_ATTN_QUERY,&shape) == 0);
	SparkStagePackFamilyNarrowShape(spec,table,SPARK_LLM_STAGEPACK_TENSOR_ATTN_QUERY,&shape,4u,1u);
	assert(shape.rows == 3072u);
	assert(SparkStagePackFamilyTensorShapeOf(spec,table,SPARK_STAGEPACK_TENSOR_GDN_QKV,&shape) == 0);
	SparkStagePackFamilyNarrowShape(spec,table,SPARK_STAGEPACK_TENSOR_GDN_QKV,&shape,4u,1u);
	assert(shape.rows == 2560u);
	assert(SparkStagePackFamilyTensorShapeOf(spec,table,SPARK_STAGEPACK_TENSOR_MOE_GATE,&shape) == 0);
	SparkStagePackFamilyNarrowShape(spec,table,SPARK_STAGEPACK_TENSOR_MOE_GATE,&shape,4u,1u);
	assert(shape.rows == 128u);
	assert(SparkStagePackFamilyTensorShapeOf(spec,table,SPARK_LLM_STAGEPACK_TENSOR_PLE_NGRAM,&shape) == 0);
	SparkStagePackFamilyNarrowShape(spec,table,SPARK_LLM_STAGEPACK_TENSOR_PLE_NGRAM,&shape,4u,1u);
	assert(shape.rows == 80000384u);
	assert(SparkStagePackFamilyTensorShapeOf(spec,table,SPARK_LLM_STAGEPACK_TENSOR_EMBEDDING,&shape) == 0);
	SparkStagePackFamilyNarrowShape(spec,table,SPARK_LLM_STAGEPACK_TENSOR_EMBEDDING,&shape,4u,1u);
	assert(shape.rows == 62080u);
	assert(SparkStagePackFamilyTensorShapeOf(spec,table,SPARK_STAGEPACK_TENSOR_GDN_A_LOG,&shape) == 0);
	SparkStagePackFamilyNarrowShape(spec,table,SPARK_STAGEPACK_TENSOR_GDN_A_LOG,&shape,4u,1u);
	assert(shape.columns == 12u);
}

static void SparkTestPayloadAndScaleBytes(void)
{
	const SparkStagePackFamilySpec *spec = &SparkLlmStagePackFamilySpec;
	assert(SparkStagePackFamilyPayloadBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_BF16,2u,3u) == 12u);
	assert(SparkStagePackFamilyPayloadBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_F32,2u,3u) == 24u);
	assert(SparkStagePackFamilyPayloadBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_U32,2u,3u) == 24u);
	assert(SparkStagePackFamilyPayloadBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_I64,2u,3u) == 48u);
	assert(SparkStagePackFamilyPayloadBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_MXFP4_E2M1,32u,32u) == 512u);
	assert(SparkStagePackFamilyPayloadBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128,256u,256u) == 65536u);
	assert(SparkStagePackFamilyPayloadBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_E8M0B128,256u,256u) == 65536u);
	assert(SparkStagePackFamilyPayloadBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_NVFP4_PACKED,640u,2560u) == 819200u);
	assert(SparkStagePackFamilyScaleBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_MXFP4_E2M1,32u,2560u) == 2560u);
	assert(SparkStagePackFamilyScaleBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128,2560u,2560u) == 1600u);
	assert(SparkStagePackFamilyScaleBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_E8M0B128,2560u,2560u) == 51200u);
	assert(SparkStagePackFamilyScaleBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_NVFP4_PACKED,640u,2560u) == 102408u);
	assert(SparkStagePackFamilyScaleBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_NVFP4_PACKED,2560u,640u) == 102408u);
	assert(SparkStagePackFamilyScaleBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_NVFP4_PACKED,100u,100u) == 0u);
	assert(SparkStagePackFamilyScaleBytes(spec,SPARK_STAGEPACK_FORMAT_WEIGHT_BF16,2u,3u) == 0u);
}

int32_t main(void)
{
	SparkTestWireLayout();
	SparkTestExpectedTensorCounts();
	SparkTestExpectedGeometry();
	SparkTestShapes();
	SparkTestResolvedShape();
	SparkTestNarrowShape();
	SparkTestPayloadAndScaleBytes();
	assert(SparkTestNegativeControls() == 0);
	printf("test_llm_stagepack_format: all vectors from llm_defines.h pass\n");
	return(0);
}
