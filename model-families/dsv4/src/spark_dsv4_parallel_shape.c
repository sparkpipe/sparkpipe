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

static int SparkDsv4TpDegreeIsSupported(uint32_t degree)
{
	return degree == 1u || degree == 2u || degree == 4u ||
		degree == 8u || degree == SPARK_DSV4_PARALLEL_SHAPE_MAX_TP_DEGREE;
}

static SparkStatus SparkDsv4TpDeriveLayerSlice(
	uint32_t stage_count,
	uint32_t stage_index,
	uint32_t *first_layer_out,
	uint32_t *layer_count_out)
{
	uint32_t base_layer_count;
	uint32_t remainder;
	if ( stage_count == 0u ||
		stage_count > SPARK_DSV4_PARALLEL_SHAPE_MAX_PP_DEGREE ||
		stage_count > SPARK_DSV4_MODEL_LAYER_COUNT ||
		stage_index >= stage_count || first_layer_out == 0 ||
		layer_count_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	base_layer_count = SPARK_DSV4_MODEL_LAYER_COUNT / stage_count;
	remainder = SPARK_DSV4_MODEL_LAYER_COUNT % stage_count;
	*layer_count_out = base_layer_count +
		(stage_index < remainder ? 1u : 0u);
	*first_layer_out = stage_index * base_layer_count +
		(stage_index < remainder ? stage_index : remainder);
	return(SPARK_STATUS_OK);
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
	uint32_t groups_per_rank;
	uint32_t ranks_per_group;
	uint32_t world_size;
	SparkStatus status;
	if ( shape == 0 || config_out == 0 ||
		shape->abi_version != SPARK_DSV4_PARALLEL_SHAPE_ABI_VERSION )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	degree = shape->tp_degree;
	if ( !SparkDsv4TpDegreeIsSupported(degree) )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( shape->tp_rank >= degree || shape->pp_stage_count == 0u ||
		shape->pp_stage_count > SPARK_DSV4_PARALLEL_SHAPE_MAX_PP_DEGREE ||
		shape->pp_stage_index >= shape->pp_stage_count ||
		degree > SPARK_DSV4_PARALLEL_SHAPE_MAX_WORLD_SIZE /
			shape->pp_stage_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	world_size = degree * shape->pp_stage_count;
	if ( world_size > SPARK_DSV4_PARALLEL_SHAPE_MAX_WORLD_SIZE )
		return(SPARK_STATUS_UNSUPPORTED);
	ranks_per_group = degree > SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT ?
		degree / SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT : 1u;
	groups_per_rank = degree < SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT ?
		SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT / degree : 1u;
	if ( SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT % degree != 0u ||
		SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION % degree != 0u ||
		SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT %
			(degree < SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT ? degree :
			SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT) != 0u ||
		SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION % ranks_per_group != 0u ||
		SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION % degree != 0u ||
		SPARK_DSV4_MODEL_OUTPUT_LORA_RANK == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(config_out,0,sizeof(*config_out));
	config_out->abi_version = SPARK_DSV4_PARALLEL_SHAPE_ABI_VERSION;
	config_out->tp_degree = degree;
	config_out->tp_rank = shape->tp_rank;
	status = SparkDsv4TpDeriveLayerSlice(shape->pp_stage_count,
		shape->pp_stage_index,&config_out->first_layer_index,
		&config_out->layer_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	config_out->query_heads_per_rank =
		SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT / degree;
	config_out->query_output_elements_per_rank =
		SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION / degree;
	config_out->output_group_count_per_rank = groups_per_rank;
	config_out->output_group_input_elements_per_rank =
		SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION / ranks_per_group;
	config_out->output_lora_elements_per_rank = groups_per_rank *
		SPARK_DSV4_MODEL_OUTPUT_LORA_RANK;
	config_out->output_hidden_rows_per_rank =
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	config_out->expert_intermediate_per_rank =
		SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION / degree;
	config_out->vocabulary_rows_per_rank =
		SPARK_DSV4_MODEL_VOCAB_COUNT;
	config_out->world_size = world_size;
	config_out->world_rank = shape->pp_stage_index * degree + shape->tp_rank;
	config_out->configuration_hash =
		SparkDsv4TpConfigurationHash(shape,config_out);
	return(SPARK_STATUS_OK);
}
