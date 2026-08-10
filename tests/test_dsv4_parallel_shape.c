#include "sparkpipe/spark_dsv4_parallel_shape.h"

#include <assert.h>
#include <string.h>

static SparkDsv4TpShapeDescriptor SparkDsv4TestShape(
	uint32_t degree,
	uint32_t rank)
{
	SparkDsv4TpShapeDescriptor shape;
	memset(&shape,0,sizeof(shape));
	shape.abi_version = SPARK_DSV4_PARALLEL_SHAPE_ABI_VERSION;
	shape.tp_degree = degree;
	shape.tp_rank = rank;
	shape.pp_stage_count = 1u;
	shape.pp_stage_index = 0u;
	return(shape);
}

static void SparkDsv4TestTp16(void)
{
	SparkDsv4TpShapeDescriptor shape;
	SparkDsv4TpNodeConfig config;
	shape = SparkDsv4TestShape(16u,15u);
	assert(SparkDsv4TpDeriveNodeConfig(&shape,&config) == SPARK_STATUS_OK);
	assert(config.first_layer_index == 0u);
	assert(config.layer_count == 43u);
	assert(config.query_heads_per_rank == 4u);
	assert(config.query_output_elements_per_rank == 2048u);
	assert(config.output_composition_input_elements_per_rank == 256u);
	assert(config.output_hidden_rows_per_rank == 256u);
	assert(config.expert_intermediate_per_rank == 128u);
	assert(config.vocabulary_rows_per_rank == 8080u);
	assert(config.configuration_hash != 0u);
}

static void SparkDsv4TestHashSeparatesRanks(void)
{
	SparkDsv4TpShapeDescriptor shape_a,shape_b;
	SparkDsv4TpNodeConfig config_a,config_b;
	shape_a = SparkDsv4TestShape(16u,0u);
	shape_b = SparkDsv4TestShape(16u,1u);
	assert(SparkDsv4TpDeriveNodeConfig(&shape_a,&config_a) == SPARK_STATUS_OK);
	assert(SparkDsv4TpDeriveNodeConfig(&shape_b,&config_b) == SPARK_STATUS_OK);
	assert(config_a.configuration_hash != config_b.configuration_hash);
}

static void SparkDsv4TestFailsClosed(void)
{
	SparkDsv4TpShapeDescriptor shape;
	SparkDsv4TpNodeConfig config;
	shape = SparkDsv4TestShape(8u,0u);
	assert(SparkDsv4TpDeriveNodeConfig(&shape,&config) == SPARK_STATUS_UNSUPPORTED);
	shape = SparkDsv4TestShape(16u,16u);
	assert(SparkDsv4TpDeriveNodeConfig(&shape,&config) == SPARK_STATUS_INVALID_ARGUMENT);
	shape = SparkDsv4TestShape(16u,0u);
	shape.pp_stage_count = 2u;
	assert(SparkDsv4TpDeriveNodeConfig(&shape,&config) == SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
	SparkDsv4TestTp16();
	SparkDsv4TestHashSeparatesRanks();
	SparkDsv4TestFailsClosed();
	return(0);
}
