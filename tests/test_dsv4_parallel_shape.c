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
	assert(config.output_group_count_per_rank == 1u);
	assert(config.output_group_input_elements_per_rank == 2048u);
	assert(config.output_lora_elements_per_rank == 1024u);
	assert(config.output_hidden_rows_per_rank == 4096u);
	assert(config.expert_intermediate_per_rank == 128u);
	assert(config.vocabulary_row_start == 121216u);
	assert(config.vocabulary_rows_per_rank == 8064u);
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

static void SparkDsv4TestTp4(void)
{
	SparkDsv4TpShapeDescriptor shape;
	SparkDsv4TpNodeConfig config;
	shape = SparkDsv4TestShape(4u,3u);
	assert(SparkDsv4TpDeriveNodeConfig(&shape,&config) == SPARK_STATUS_OK);
	assert(config.query_heads_per_rank == 16u);
	assert(config.query_output_elements_per_rank == 8192u);
	assert(config.output_group_count_per_rank == 2u);
	assert(config.output_group_input_elements_per_rank == 4096u);
	assert(config.output_lora_elements_per_rank == 2048u);
	assert(config.output_hidden_rows_per_rank == 4096u);
	assert(config.expert_intermediate_per_rank == 512u);
	assert(config.vocabulary_row_start == 97024u);
	assert(config.vocabulary_rows_per_rank == 32256u);
}

static void SparkDsv4TestVocabularyCoverage(void)
{
	SparkDsv4TpShapeDescriptor shape;
	SparkDsv4TpNodeConfig config;
	uint32_t degree,rank,row;
	for (degree=1u; degree<=16u; degree<<=1u)
	{
		row = 0u;
		for (rank=0u; rank<degree; rank++)
		{
			shape = SparkDsv4TestShape(degree,rank);
			assert(SparkDsv4TpDeriveNodeConfig(&shape,&config) ==
				SPARK_STATUS_OK);
			assert(config.vocabulary_row_start == row);
			assert(config.vocabulary_rows_per_rank % 128u == 0u);
			row += config.vocabulary_rows_per_rank;
		}
		assert(row == 129280u);
	}
}

static void SparkDsv4TestFailsClosed(void)
{
	SparkDsv4TpShapeDescriptor shape;
	SparkDsv4TpNodeConfig config;
	shape = SparkDsv4TestShape(3u,0u);
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
	SparkDsv4TestTp4();
	SparkDsv4TestVocabularyCoverage();
	SparkDsv4TestHashSeparatesRanks();
	SparkDsv4TestFailsClosed();
	return(0);
}
