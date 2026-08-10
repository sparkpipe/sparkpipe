#include "sparkpipe/spark_dsv4_parallel_shape.h"

#include <stddef.h>
#include <string.h>

static uint64_t SparkDsv4TpHashBytes(
	uint64_t hash,
	const void *data,
	uint32_t data_bytes)
{
	const uint8_t *bytes;
	uint32_t index;
	bytes = (const uint8_t *)data;
	for (index=0u; index<data_bytes; index++)
	{
		hash ^= bytes[index];
		hash *= UINT64_C(1099511628211);
	}
	return(hash);
}

uint64_t SparkDsv4TpConfigurationHash(
	const SparkDsv4TpShapeDescriptor *shape,
	const SparkDsv4TpNodeConfig *config)
{
	uint64_t hash;
	if ( shape == 0 || config == 0 )
		return(0u);
	hash = UINT64_C(1469598103934665603);
	hash = SparkDsv4TpHashBytes(hash,shape,sizeof(*shape));
	hash = SparkDsv4TpHashBytes(hash,config,(uint32_t)offsetof(SparkDsv4TpNodeConfig,configuration_hash));
	return(hash);
}

SparkStatus SparkDsv4TpDeriveNodeConfig(
	const SparkDsv4TpShapeDescriptor *shape,
	SparkDsv4TpNodeConfig *config_out)
{
	uint32_t degree;
	if ( shape == 0 || config_out == 0 ||
		shape->abi_version != SPARK_DSV4_PARALLEL_SHAPE_ABI_VERSION )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	degree = shape->tp_degree;
	if ( degree != 1u && degree != SPARK_DSV4_PARALLEL_SHAPE_MAX_TP_DEGREE )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( shape->tp_rank >= degree || shape->pp_stage_count != 1u ||
		shape->pp_stage_index != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT % degree != 0u ||
		SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION % degree != 0u ||
		SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION % degree != 0u ||
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION % degree != 0u ||
		SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION % degree != 0u ||
		SPARK_DSV4_MODEL_VOCAB_COUNT % degree != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(config_out,0,sizeof(*config_out));
	config_out->abi_version = SPARK_DSV4_PARALLEL_SHAPE_ABI_VERSION;
	config_out->tp_degree = degree;
	config_out->tp_rank = shape->tp_rank;
	config_out->first_layer_index = 0u;
	config_out->layer_count = SPARK_DSV4_MODEL_LAYER_COUNT;
	config_out->query_heads_per_rank =
		SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT / degree;
	config_out->query_output_elements_per_rank =
		SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION / degree;
	config_out->output_composition_input_elements_per_rank =
		SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION / degree;
	config_out->output_hidden_rows_per_rank =
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION / degree;
	config_out->expert_intermediate_per_rank =
		SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION / degree;
	config_out->vocabulary_rows_per_rank =
		SPARK_DSV4_MODEL_VOCAB_COUNT / degree;
	config_out->configuration_hash =
		SparkDsv4TpConfigurationHash(shape,config_out);
	return(SPARK_STATUS_OK);
}
