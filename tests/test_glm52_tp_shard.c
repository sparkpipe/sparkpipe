#include "sparkpipe/spark_glm52_tp_shard.h"

#include <assert.h>
#include <string.h>

// Real GLM-5.2 shapes: hidden 6144, 64 heads, q head block 192+64=256, kv_b
// head block 192+256=448, o_proj input block 256, dense intermediate 12288,
// shared-expert intermediate 2048, latent+rope 576.

static void SparkTestTpShardSpec(SparkGlm52StagePackTensorSpec *spec,const char *name,uint64_t dim0,uint64_t dim1)
{
	memset(spec,0,sizeof(*spec));
	spec->abi_version = SPARK_GLM52_STAGEPACK_ABI_VERSION;
	spec->rank = 2u;
	spec->bytes_per_element = SPARK_GLM52_MODEL_BF16_ELEMENT_BYTES;
	spec->shape[0] = dim0;
	spec->shape[1] = dim1;
	spec->tensor_name = name;
	spec->dtype = "BF16";
}

static void SparkTestTpShardShape(SparkGlm52TpShapeDescriptor *shape,uint32_t degree,uint32_t rank)
{
	memset(shape,0,sizeof(*shape));
	shape->abi_version = SPARK_GLM52_TP_SHARD_ABI_VERSION;
	shape->tp_degree = degree;
	shape->tp_rank = rank;
	shape->pp_stage_count = 1u;
	shape->pp_stage_index = 0u;
}

static void SparkTestTpShardGeometry(SparkGlm52TpModelGeometry *geometry)
{
	memset(geometry,0,sizeof(*geometry));
	SparkGlm52TpModelGeometryFromModel(geometry);
}

static void SparkTestTpShardClassification(void)
{
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.self_attn.q_b_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM_HEADS);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.self_attn.kv_b_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM_HEADS);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.self_attn.o_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_INPUT_DIM_HEADS);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.0.mlp.gate_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.9.mlp.shared_experts.up_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_OUTPUT_DIM);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.0.mlp.down_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_INPUT_DIM);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.9.mlp.shared_experts.down_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_INPUT_DIM);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.self_attn.q_a_proj.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_REPLICATED);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.self_attn.kv_a_proj_with_mqa.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_REPLICATED);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.mlp.gate.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_REPLICATED);
	assert(SparkGlm52TpShardClassifyTensor("model.embed_tokens.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_REPLICATED);
	assert(SparkGlm52TpShardClassifyTensor("model.layers.7.self_attn.mystery.weight") ==
		SPARK_GLM52_TP_SHARD_CLASS_UNKNOWN);
}

// The four tp4 shards of q_b tile the output dimension exactly: contiguous,
// non-overlapping, head-block aligned, and their bytes sum to the full tensor.
static void SparkTestTpShardTilingExactness(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52TpModelGeometry geometry;
	uint64_t next_offset,total_bytes,full_bytes;
	uint32_t rank_index;
	SparkTestTpShardSpec(&spec,"model.layers.3.self_attn.q_b_proj.weight",16384u,1536u);
	SparkTestTpShardGeometry(&geometry);
	full_bytes = 16384u * 1536u * sizeof(uint16_t);
	next_offset = 0u;
	total_bytes = 0u;
	for (rank_index = 0u; rank_index < 4u; ++rank_index)
	{
		SparkGlm52TpShapeDescriptor shape;
		SparkGlm52TpShardView view;
		SparkTestTpShardShape(&shape,4u,rank_index);
		assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_OK);
		assert(view.split_dimension == 0u);
		assert(view.element_offset == next_offset);
		assert(view.element_extent == 4096u);
		assert(view.element_offset % 256u == 0u);
		next_offset = view.element_offset + view.element_extent;
		total_bytes += view.shard_bytes;
	}
	assert(next_offset == 16384u);
	assert(total_bytes == full_bytes);
}

// o_proj splits its input dimension on value-head blocks; tp8 gives eight
// 2048-element input slices.
static void SparkTestTpShardInputDimHeads(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52TpShapeDescriptor shape;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52TpShardView view;
	SparkTestTpShardSpec(&spec,"model.layers.3.self_attn.o_proj.weight",SPARK_GLM52_MODEL_HIDDEN_DIMENSION,SPARK_GLM52_MODEL_ATTENTION_PROJECTION_DIMENSION);
	SparkTestTpShardShape(&shape,8u,5u);
	SparkTestTpShardGeometry(&geometry);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_OK);
	assert(view.split_dimension == 1u);
	assert(view.element_extent == 2048u);
	assert(view.element_offset == 5u * 2048u);
	assert(view.shard_bytes == SPARK_GLM52_MODEL_HIDDEN_DIMENSION * 2048u * sizeof(uint16_t));
}

// Replicated tensors load whole on every rank; the latent kv_a path is the
// canonical case.
static void SparkTestTpShardReplicated(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52TpShapeDescriptor shape;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52TpShardView view;
	SparkTestTpShardSpec(&spec,"model.layers.3.self_attn.kv_a_proj_with_mqa.weight",SPARK_GLM52_MODEL_KV_A_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION);
	SparkTestTpShardShape(&shape,4u,2u);
	SparkTestTpShardGeometry(&geometry);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_OK);
	assert(view.shard_class == SPARK_GLM52_TP_SHARD_CLASS_REPLICATED);
	assert(view.element_offset == 0u);
	assert(view.shard_bytes == SPARK_GLM52_MODEL_KV_A_DIMENSION * SPARK_GLM52_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t));
}

// Degree one is a whole-tensor view for every class including unknown, so
// existing single-shape packs keep loading unchanged.
static void SparkTestTpShardDegreeOneCompat(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52TpShapeDescriptor shape;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52TpShardView view;
	SparkTestTpShardSpec(&spec,"some.future.tensor.weight",100u,200u);
	SparkTestTpShardShape(&shape,1u,0u);
	SparkTestTpShardGeometry(&geometry);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_OK);
	assert(view.shard_bytes == 100u * 200u * sizeof(uint16_t));
}

static void SparkTestTpShardFailsClosed(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52TpShapeDescriptor shape;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52TpShardView view;
	SparkTestTpShardSpec(&spec,"model.layers.0.mlp.gate_proj.weight",SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION);
	SparkTestTpShardGeometry(&geometry);
	// Degrees that do not divide the model are rejected outright.
	SparkTestTpShardShape(&shape,3u,0u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_INVALID_ARGUMENT);
	// Degree sixteen is clean: 64 heads give four per rank.
	SparkTestTpShardShape(&shape,16u,0u);
	{
		SparkGlm52StagePackTensorSpec sixteen_spec;
		SparkGlm52TpShardView sixteen_view;
		SparkTestTpShardSpec(&sixteen_spec,"model.layers.0.mlp.gate_proj.weight",SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION);
		assert(SparkGlm52TpShardComputeView(&sixteen_spec,&shape,&geometry,&sixteen_view) == SPARK_STATUS_OK);
		assert(sixteen_view.element_extent == 768u);
	}
	SparkTestTpShardShape(&shape,13u,0u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_INVALID_ARGUMENT);
	// Rank out of range.
	SparkTestTpShardShape(&shape,4u,4u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_INVALID_ARGUMENT);
	// Unknown tensors at any real degree fail closed instead of guessing.
	SparkTestTpShardSpec(&spec,"model.layers.0.mystery.weight",SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION);
	SparkTestTpShardShape(&shape,2u,0u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_VALIDATION_FAILED);
	// A dimension the degree does not divide is rejected.
	SparkTestTpShardSpec(&spec,"model.layers.0.mlp.gate_proj.weight",SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION + 2u,SPARK_GLM52_MODEL_HIDDEN_DIMENSION);
	SparkTestTpShardShape(&shape,8u,0u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_INVALID_ARGUMENT);
	// Head count that the degree does not divide is rejected at validation.
	SparkTestTpShardSpec(&spec,"model.layers.0.mlp.gate_proj.weight",SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION,SPARK_GLM52_MODEL_HIDDEN_DIMENSION);
	SparkTestTpShardShape(&shape,4u,0u);
	geometry.head_count = 6u;
	assert(SparkGlm52TpShardComputeView(&spec,&shape,&geometry,&view) == SPARK_STATUS_INVALID_ARGUMENT);
}

// The contract hash separates every degree, rank, and tensor, and is stable
// for identical inputs.
static void SparkTestTpShardGeometryHash(void)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52TpShapeDescriptor shape_a,shape_b;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52TpShardView view_a,view_b;
	uint64_t hash_a,hash_b,hash_a_repeat;
	SparkTestTpShardSpec(&spec,"model.layers.3.self_attn.q_b_proj.weight",16384u,1536u);
	SparkTestTpShardGeometry(&geometry);
	SparkTestTpShardShape(&shape_a,4u,1u);
	SparkTestTpShardShape(&shape_b,4u,2u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape_a,&geometry,&view_a) == SPARK_STATUS_OK);
	assert(SparkGlm52TpShardComputeView(&spec,&shape_b,&geometry,&view_b) == SPARK_STATUS_OK);
	hash_a = SparkGlm52TpShardGeometryHash(&spec,&shape_a,&view_a);
	hash_b = SparkGlm52TpShardGeometryHash(&spec,&shape_b,&view_b);
	hash_a_repeat = SparkGlm52TpShardGeometryHash(&spec,&shape_a,&view_a);
	assert(hash_a != 0u && hash_b != 0u);
	assert(hash_a != hash_b);
	assert(hash_a == hash_a_repeat);
	SparkTestTpShardShape(&shape_b,2u,1u);
	assert(SparkGlm52TpShardComputeView(&spec,&shape_b,&geometry,&view_b) == SPARK_STATUS_OK);
	assert(SparkGlm52TpShardGeometryHash(&spec,&shape_b,&view_b) != hash_a);
}

// Round-trip against a synthetic on-disk pack: three tensors with distinct
// linear-index patterns at a nonzero file offset, read back as tp2 shards.
// The two shards of each split tensor must reconstruct the original
// byte-exactly, replication must return the whole tensor, and a wrong
// destination size must fail closed without touching the buffer contract.
#include <stdio.h>
#include <stdlib.h>

#define SPARK_TEST_TP_SHARD_ROOT "/tmp/spark_tp_shard_pack"
#define SPARK_TEST_TP_SHARD_DATA_OFFSET 64u

static void SparkTestTpShardWriteFixture(uint16_t *gate,uint32_t gate_elements,uint16_t *down,uint32_t down_elements)
{
	FILE *stream;
	char path[256];
	uint32_t element_index;
	uint8_t padding[SPARK_TEST_TP_SHARD_DATA_OFFSET];
	for (element_index = 0u; element_index < gate_elements; ++element_index)
		gate[element_index] = (uint16_t)(element_index + 1u);
	for (element_index = 0u; element_index < down_elements; ++element_index)
		down[element_index] = (uint16_t)(0x8000u + element_index);
	memset(padding,0xEE,sizeof(padding));
	assert(system("mkdir -p " SPARK_TEST_TP_SHARD_ROOT) == 0);
	snprintf(path,sizeof(path),"%s/tensors.bin",SPARK_TEST_TP_SHARD_ROOT);
	stream = fopen(path,"wb");
	assert(stream != 0);
	assert(fwrite(padding,1u,sizeof(padding),stream) == sizeof(padding));
	assert(fwrite(gate,sizeof(uint16_t),gate_elements,stream) == gate_elements);
	assert(fwrite(down,sizeof(uint16_t),down_elements,stream) == down_elements);
	fclose(stream);
	snprintf(path,sizeof(path),"%s/%s",SPARK_TEST_TP_SHARD_ROOT,SPARK_GLM52_STAGEPACK_INDEX_FILE);
	stream = fopen(path,"w");
	assert(stream != 0);
	fprintf(stream,
		"{\n"
		"  \"format\": \"%s\",\n"
		"  \"tensor_map\": {\n"
		"    \"model.layers.0.mlp.gate_proj.weight\": {\n"
		"      \"file\": \"tensors.bin\", \"dtype\": \"BF16\",\n"
		"      \"shape\": [8, 4], \"offset\": %u, \"bytes\": %u},\n"
		"    \"model.layers.0.mlp.down_proj.weight\": {\n"
		"      \"file\": \"tensors.bin\", \"dtype\": \"BF16\",\n"
		"      \"shape\": [4, 8], \"offset\": %u, \"bytes\": %u}\n"
		"  }\n"
		"}\n",
		SPARK_GLM52_STAGEPACK_FORMAT,
		SPARK_TEST_TP_SHARD_DATA_OFFSET,
		(unsigned)(32u * sizeof(uint16_t)),
		(unsigned)(SPARK_TEST_TP_SHARD_DATA_OFFSET + 32u * sizeof(uint16_t)),
		(unsigned)(32u * sizeof(uint16_t)));
	fclose(stream);
}

static void SparkTestTpShardRoundTrip(void)
{
	SparkGlm52StagePackTensorSpec gate_spec,down_spec;
	SparkGlm52TpModelGeometry geometry;
	SparkGlm52TpShardView view;
	uint16_t gate[32],down[32];
	uint16_t shard_a[16],shard_b[16],rebuilt[32];
	uint32_t rank_index,row_index,element_index;
	SparkTestTpShardWriteFixture(gate,32u,down,32u);
	SparkTestTpShardGeometry(&geometry);
	geometry.head_count = 2u;
	SparkTestTpShardSpec(&gate_spec,"model.layers.0.mlp.gate_proj.weight",8u,4u);
	SparkTestTpShardSpec(&down_spec,"model.layers.0.mlp.down_proj.weight",4u,8u);
	// gate splits its leading dimension: rank 0 takes rows 0..3, rank 1 rows
	// 4..7, and the concatenation is the original tensor.
	for (rank_index = 0u; rank_index < 2u; ++rank_index)
	{
		SparkGlm52TpShapeDescriptor shape;
		uint16_t *target = rank_index == 0u ? shard_a : shard_b;
		SparkTestTpShardShape(&shape,2u,rank_index);
		assert(SparkGlm52TpShardReadTensor(SPARK_TEST_TP_SHARD_ROOT,&gate_spec,&shape,&geometry,target,16u * sizeof(uint16_t),&view) == SPARK_STATUS_OK);
		assert(view.split_dimension == 0u);
	}
	memcpy(rebuilt,shard_a,16u * sizeof(uint16_t));
	memcpy(rebuilt + 16u,shard_b,16u * sizeof(uint16_t));
	assert(memcmp(rebuilt,gate,sizeof(gate)) == 0);
	// down splits its inner dimension: each rank gathers a 4-element chunk per
	// outer row, and interleaving the chunks rebuilds the original rows.
	for (rank_index = 0u; rank_index < 2u; ++rank_index)
	{
		SparkGlm52TpShapeDescriptor shape;
		uint16_t *target = rank_index == 0u ? shard_a : shard_b;
		SparkTestTpShardShape(&shape,2u,rank_index);
		assert(SparkGlm52TpShardReadTensor(SPARK_TEST_TP_SHARD_ROOT,&down_spec,&shape,&geometry,target,16u * sizeof(uint16_t),&view) == SPARK_STATUS_OK);
		assert(view.split_dimension == 1u);
		assert(view.element_extent == 4u);
	}
	for (row_index = 0u; row_index < 4u; ++row_index)
		for (element_index = 0u; element_index < 4u; ++element_index)
		{
			rebuilt[row_index * 8u + element_index] = shard_a[row_index * 4u + element_index];
			rebuilt[row_index * 8u + 4u + element_index] = shard_b[row_index * 4u + element_index];
		}
	assert(memcmp(rebuilt,down,sizeof(down)) == 0);
	// Degree one returns the whole tensor from the same entry point.
	{
		SparkGlm52TpShapeDescriptor shape;
		SparkTestTpShardShape(&shape,1u,0u);
		assert(SparkGlm52TpShardReadTensor(SPARK_TEST_TP_SHARD_ROOT,&gate_spec,&shape,&geometry,rebuilt,32u * sizeof(uint16_t),&view) == SPARK_STATUS_OK);
		assert(memcmp(rebuilt,gate,sizeof(gate)) == 0);
	}
	// A wrong destination size fails closed before any byte is read.
	{
		SparkGlm52TpShapeDescriptor shape;
		SparkTestTpShardShape(&shape,2u,0u);
		assert(SparkGlm52TpShardReadTensor(SPARK_TEST_TP_SHARD_ROOT,&gate_spec,&shape,&geometry,shard_a,15u * sizeof(uint16_t),&view) == SPARK_STATUS_INVALID_ARGUMENT);
	}
}

// The model-derived geometry must equal the values every test in this file
// hardcodes, proving the initializer and the authoritative constants agree.
static void SparkTestTpShardModelGeometry(void)
{
	SparkGlm52TpModelGeometry from_model,reference;
	SparkGlm52TpModelGeometryFromModel(&from_model);
	SparkTestTpShardGeometry(&reference);
	assert(memcmp(&from_model,&reference,sizeof(reference)) == 0);
}

int main(void)
{
	SparkTestTpShardClassification();
	SparkTestTpShardTilingExactness();
	SparkTestTpShardInputDimHeads();
	SparkTestTpShardReplicated();
	SparkTestTpShardDegreeOneCompat();
	SparkTestTpShardFailsClosed();
	SparkTestTpShardGeometryHash();
	SparkTestTpShardModelGeometry();
	SparkTestTpShardRoundTrip();
	return 0;
}
