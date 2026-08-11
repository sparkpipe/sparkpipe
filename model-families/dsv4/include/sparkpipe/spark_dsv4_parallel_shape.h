#pragma once

#include <stdint.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DSV4_PARALLEL_SHAPE_ABI_VERSION 1u
#define SPARK_DSV4_PARALLEL_SHAPE_MAX_TP_DEGREE 16u

/*
 * The TP descriptor is deliberately separate from the PP deployment
 * descriptor.  TP16 is one complete model replica per rank with dimensions
 * sharded inside every layer; it is not twelve or sixteen pipeline slices.
 * A non-one PP stage count is therefore refused until a distinct TPxPP
 * contract is added instead of silently changing the topology semantics.
 */
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
	uint32_t output_composition_input_elements_per_rank;
	uint32_t output_hidden_rows_per_rank;
	uint32_t expert_intermediate_per_rank;
	uint32_t vocabulary_rows_per_rank;
	uint32_t reserved0;
	uint64_t configuration_hash;
} SparkDsv4TpNodeConfig;

/*
 * Derive every dimension that a rank-sharded DSV4 kernel must agree on.
 * Current supported TP shapes are degree one and TP16, both over the full 43
 * layer model.  Degree one is retained for a byte-identical control pack;
 * higher degrees fail closed unless every split dimension divides exactly.
 */
SparkStatus SparkDsv4TpDeriveNodeConfig(
	const SparkDsv4TpShapeDescriptor *shape,
	SparkDsv4TpNodeConfig *config_out);

uint64_t SparkDsv4TpConfigurationHash(
	const SparkDsv4TpShapeDescriptor *shape,
	const SparkDsv4TpNodeConfig *config);

#ifdef __cplusplus
}
#endif
