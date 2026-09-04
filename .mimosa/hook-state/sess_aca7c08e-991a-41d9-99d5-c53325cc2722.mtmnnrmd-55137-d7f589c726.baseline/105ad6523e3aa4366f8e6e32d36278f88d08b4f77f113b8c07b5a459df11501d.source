#pragma once

#include <stdint.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DSV4_PARALLEL_SHAPE_ABI_VERSION 4u
#define SPARK_DSV4_PARALLEL_SHAPE_MAX_TP_DEGREE 16u
#define SPARK_DSV4_PARALLEL_SHAPE_MAX_PP_DEGREE 16u
#define SPARK_DSV4_PARALLEL_SHAPE_MAX_WORLD_SIZE 16u

typedef struct SparkDsv4TpShapeDescriptor
{
	uint32_t abi_version;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t pp_stage_count;
	uint32_t pp_stage_index;
} SparkDsv4TpShapeDescriptor;

typedef struct SparkDsv4TpNodeConfig
{
	uint32_t abi_version;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t query_heads_per_rank;
	uint32_t query_output_elements_per_rank;
	uint32_t output_group_count_per_rank;
	uint32_t output_group_input_elements_per_rank;
	uint32_t output_lora_elements_per_rank;
	uint32_t output_hidden_rows_per_rank;
	uint32_t expert_intermediate_per_rank;
	uint32_t vocabulary_row_start;
	uint32_t vocabulary_rows_per_rank;
	uint32_t world_size;
	uint32_t world_rank;
	uint64_t configuration_hash;
} SparkDsv4TpNodeConfig;

SparkStatus SparkDsv4TpDeriveNodeConfig(
	const SparkDsv4TpShapeDescriptor *shape,
	SparkDsv4TpNodeConfig *config_out);

uint64_t SparkDsv4TpConfigurationHash(
	const SparkDsv4TpShapeDescriptor *shape,
	const SparkDsv4TpNodeConfig *config);

#ifdef __cplusplus
}
#endif
