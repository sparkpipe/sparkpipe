#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_glm52_shape_config.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPARK_TEST_SHAPE_FULL_ROOT "/tmp/spark_shape_full_pack"
#define SPARK_TEST_SHAPE_NODE_ROOT "/tmp/spark_shape_node_pack"

static void SparkTestShapeDescriptor(SparkGlm52TpShapeDescriptor *shape,uint32_t tp_degree,uint32_t tp_rank,uint32_t pp_stage_count,uint32_t pp_stage_index)
{
	memset(shape,0,sizeof(*shape));
	shape->abi_version = SPARK_GLM52_TP_SHARD_ABI_VERSION;
	shape->tp_degree = tp_degree;
	shape->tp_rank = tp_rank;
	shape->pp_stage_count = pp_stage_count;
	shape->pp_stage_index = pp_stage_index;
}

static void SparkTestShapeGeometry(SparkGlm52TpModelGeometry *geometry)
{
	memset(geometry,0,sizeof(*geometry));
	geometry->abi_version = SPARK_GLM52_TP_SHARD_ABI_VERSION;
	geometry->head_count = 64u;
	geometry->q_b_head_block = 256u;
	geometry->kv_b_head_block = 448u;
	geometry->o_proj_head_block = 256u;
}

static void SparkTestShapeInputs(SparkGlm52ShapeModelInputs *inputs)
{
	memset(inputs,0,sizeof(*inputs));
	inputs->abi_version = SPARK_GLM52_SHAPE_CONFIG_ABI_VERSION;
	inputs->total_layer_count = SPARK_GLM52_MODEL_LAYER_COUNT;
	inputs->hidden_dimension = SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
	inputs->moe_intermediate_dimension = SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION;
	inputs->dense_intermediate_dimension = SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION;
	inputs->kv_latent_plus_rope_dimension = SPARK_GLM52_MODEL_KV_A_DIMENSION;
	inputs->kv_bytes_per_element = sizeof(uint8_t);
}

// RING reproduces the deployed 6-layer stages; TP4 x PP3 gives 26-layer
// quarter-width stages; TP8 x PP1 is the widest clean shape with all 78
// layers at one-eighth width. KV per token is stage depth times the latent
// row and independent of the TP degree.
static void SparkTestShapeDerivation(void)
{
	SparkGlm52TpShapeDescriptor shape;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52ShapeModelInputs inputs;
	SparkGlm52ShapeNodeConfig config;
	SparkTestShapeGeometry(&geometry);
	SparkTestShapeInputs(&inputs);
	SparkTestShapeDescriptor(&shape,1u,0u,13u,5u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config) == SPARK_STATUS_OK);
	assert(config.first_layer_index == 30u);
	assert(config.layer_count == 6u);
	assert(config.heads_per_rank == 64u);
	assert(config.kv_bytes_per_token == 6u * SPARK_GLM52_MODEL_KV_A_DIMENSION);
	SparkTestShapeDescriptor(&shape,4u,2u,3u,1u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config) == SPARK_STATUS_OK);
	assert(config.first_layer_index == 26u);
	assert(config.layer_count == 26u);
	assert(config.heads_per_rank == 16u);
	assert(config.moe_intermediate_per_rank == 512u);
	assert(config.dense_intermediate_per_rank == 3072u);
	assert(config.q_b_output_per_rank == 16u * 256u);
	assert(config.kv_b_output_per_rank == 16u * 448u);
	assert(config.o_proj_input_per_rank == 16u * 256u);
	assert(config.kv_bytes_per_token == 26u * SPARK_GLM52_MODEL_KV_A_DIMENSION);
	SparkTestShapeDescriptor(&shape,16u,15u,1u,0u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config) == SPARK_STATUS_OK);
	assert(config.layer_count == 78u);
	assert(config.heads_per_rank == 4u);
	assert(config.moe_intermediate_per_rank == 128u);
	assert(config.dense_intermediate_per_rank == 768u);
	SparkTestShapeDescriptor(&shape,8u,7u,1u,0u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config) == SPARK_STATUS_OK);
	assert(config.first_layer_index == 0u);
	assert(config.layer_count == 78u);
	assert(config.heads_per_rank == 8u);
	assert(config.kv_bytes_per_token == SPARK_GLM52_MODEL_LAYER_COUNT * SPARK_GLM52_MODEL_KV_A_DIMENSION);
}

static void SparkTestShapeFailsClosed(void)
{
	SparkGlm52TpShapeDescriptor shape;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52ShapeModelInputs inputs;
	SparkGlm52ShapeNodeConfig config;
	SparkTestShapeGeometry(&geometry);
	SparkTestShapeInputs(&inputs);
	// A stage count that does not divide the layers evenly is rejected.
	SparkTestShapeDescriptor(&shape,1u,0u,5u,0u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config) == SPARK_STATUS_INVALID_ARGUMENT);
	// Degrees outside 1, 2, 4, 8 are rejected.
	SparkTestShapeDescriptor(&shape,3u,0u,3u,0u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config) == SPARK_STATUS_INVALID_ARGUMENT);
	SparkTestShapeDescriptor(&shape,13u,0u,1u,0u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config) == SPARK_STATUS_INVALID_ARGUMENT);
	// Out-of-range stage index and rank are rejected.
	SparkTestShapeDescriptor(&shape,1u,0u,13u,13u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config) == SPARK_STATUS_INVALID_ARGUMENT);
	SparkTestShapeDescriptor(&shape,4u,4u,3u,0u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestShapeHashSeparates(void)
{
	SparkGlm52TpShapeDescriptor shape;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52ShapeModelInputs inputs;
	SparkGlm52ShapeNodeConfig config_a,config_b,config_c;
	SparkTestShapeGeometry(&geometry);
	SparkTestShapeInputs(&inputs);
	SparkTestShapeDescriptor(&shape,4u,1u,3u,0u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config_a) == SPARK_STATUS_OK);
	SparkTestShapeDescriptor(&shape,4u,2u,3u,0u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config_b) == SPARK_STATUS_OK);
	SparkTestShapeDescriptor(&shape,4u,1u,3u,1u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config_c) == SPARK_STATUS_OK);
	assert(config_a.configuration_hash != config_b.configuration_hash);
	assert(config_a.configuration_hash != config_c.configuration_hash);
	SparkTestShapeDescriptor(&shape,4u,1u,3u,0u);
	assert(SparkGlm52ShapeDeriveNodeConfig(&shape,&geometry,&inputs,&config_b) == SPARK_STATUS_OK);
	assert(config_a.configuration_hash == config_b.configuration_hash);
}

// End to end: fabricate a full pack, generate the tp2 rank-1 node pack, and
// verify the standard resolver loads the node pack whole with byte content
// equal to the rank's shard of the original tensors. Nothing downstream of
// generation knows about sharding.
static void SparkTestShapeNodePackRoundTrip(void)
{
	SparkGlm52TpShapeDescriptor shape;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52StagePackTensorSpec specs[2];
	SparkGlm52StagePackTensorSpec node_spec;
	SparkGlm52StagePackTensorRegion region;
	uint16_t gate[32],down[32],scratch[64],loaded[16],expected[16];
	FILE *stream;
	char path[256];
	uint32_t element_index,row_index;
	for (element_index = 0u; element_index < 32u; ++element_index)
	{
		gate[element_index] = (uint16_t)(element_index + 1u);
		down[element_index] = (uint16_t)(0x9000u + element_index);
	}
	assert(system("mkdir -p " SPARK_TEST_SHAPE_FULL_ROOT) == 0);
	snprintf(path,sizeof(path),"%s/tensors.bin",SPARK_TEST_SHAPE_FULL_ROOT);
	stream = fopen(path,"wb");
	assert(stream != 0);
	assert(fwrite(gate,sizeof(gate[0]),sizeof(gate) / sizeof(gate[0]),stream) == sizeof(gate) / sizeof(gate[0]));
	assert(fwrite(down,sizeof(down[0]),sizeof(down) / sizeof(down[0]),stream) == sizeof(down) / sizeof(down[0]));
	fclose(stream);
	snprintf(path,sizeof(path),"%s/%s",SPARK_TEST_SHAPE_FULL_ROOT,SPARK_GLM52_STAGEPACK_INDEX_FILE);
	stream = fopen(path,"w");
	assert(stream != 0);
	fprintf(stream,
		"{\n  \"format\": \"%s\",\n  \"tensor_map\": {\n"
		"    \"model.layers.0.mlp.gate_proj.weight\": {\n"
		"      \"file\": \"tensors.bin\", \"dtype\": \"BF16\",\n"
		"      \"shape\": [8, 4], \"offset\": 0, \"bytes\": 64},\n"
		"    \"model.layers.0.mlp.down_proj.weight\": {\n"
		"      \"file\": \"tensors.bin\", \"dtype\": \"BF16\",\n"
		"      \"shape\": [4, 8], \"offset\": 64, \"bytes\": 64}\n"
		"  }\n}\n",
		SPARK_GLM52_STAGEPACK_FORMAT);
	fclose(stream);
	memset(specs,0,sizeof(specs));
	specs[0].abi_version = SPARK_GLM52_STAGEPACK_ABI_VERSION;
	specs[0].rank = 2u;
	specs[0].bytes_per_element = sizeof(uint16_t);
	specs[0].shape[0] = 8u;
	specs[0].shape[1] = 4u;
	specs[0].tensor_name = "model.layers.0.mlp.gate_proj.weight";
	specs[0].dtype = "BF16";
	specs[1] = specs[0];
	specs[1].shape[0] = 4u;
	specs[1].shape[1] = 8u;
	specs[1].tensor_name = "model.layers.0.mlp.down_proj.weight";
	SparkTestShapeGeometry(&geometry);
	geometry.head_count = 2u;
	SparkTestShapeDescriptor(&shape,2u,1u,1u,0u);
	assert(system("rm -rf " SPARK_TEST_SHAPE_NODE_ROOT) == 0);
	assert(SparkGlm52ShapeWriteNodeStagePack(SPARK_TEST_SHAPE_FULL_ROOT,SPARK_TEST_SHAPE_NODE_ROOT,&shape,&geometry,specs,2u,scratch,sizeof(scratch)) == SPARK_STATUS_OK);
	// The node pack's gate is rows 4..7 of the original, loaded whole through
	// the standard resolver with the sharded shape.
	node_spec = specs[0];
	node_spec.shape[0] = 4u;
	assert(SparkGlm52StagePackResolveTensor(SPARK_TEST_SHAPE_NODE_ROOT,&node_spec,&region) == SPARK_STATUS_OK);
	assert(region.tensor_bytes == 32u);
	stream = fopen(region.file_path,"rb");
	assert(stream != 0);
	assert(fseek(stream,(long)region.file_offset,SEEK_SET) == 0);
	assert(fread(loaded,sizeof(loaded[0]),sizeof(loaded) / sizeof(loaded[0]),stream) == sizeof(loaded) / sizeof(loaded[0]));
	fclose(stream);
	assert(memcmp(loaded,gate + 16u,16u * sizeof(uint16_t)) == 0);
	// The node pack's down is the second half of every original row.
	node_spec = specs[1];
	node_spec.shape[1] = 4u;
	assert(SparkGlm52StagePackResolveTensor(SPARK_TEST_SHAPE_NODE_ROOT,&node_spec,&region) == SPARK_STATUS_OK);
	assert(region.tensor_bytes == 32u);
	stream = fopen(region.file_path,"rb");
	assert(stream != 0);
	assert(fseek(stream,(long)region.file_offset,SEEK_SET) == 0);
	assert(fread(loaded,sizeof(loaded[0]),sizeof(loaded) / sizeof(loaded[0]),stream) == sizeof(loaded) / sizeof(loaded[0]));
	fclose(stream);
	for (row_index = 0u; row_index < 4u; ++row_index)
		for (element_index = 0u; element_index < 4u; ++element_index)
			expected[row_index * 4u + element_index] = down[row_index * 8u + 4u + element_index];
	assert(memcmp(loaded,expected,sizeof(expected)) == 0);
}

int main(void)
{
	SparkTestShapeDerivation();
	SparkTestShapeFailsClosed();
	SparkTestShapeHashSeparates();
	SparkTestShapeNodePackRoundTrip();
	return 0;
}
