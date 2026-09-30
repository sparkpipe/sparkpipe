#undef NDEBUG
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "sparkpipe/spark_glm52_kv_shard.h"
#include "sparkpipe/spark_weightd.h"
#include "spark_glm52_stagepack_format.h"

#define TEST_DEGREE 16u
#define TEST_SLOT SPARK_WEIGHTD_MESH_SLOT_BYTES

static void TestRounds(uint32_t rows,uint32_t query,uint32_t scatter,uint32_t peer)
{
	SparkGlm52KvShardLayerRounds abi8,abi9;
	assert(SparkGlm52KvShardLayerRoundsBuild(rows,TEST_DEGREE,TEST_SLOT,0u,&abi8) == 1u);
	assert(SparkGlm52KvShardLayerRoundsBuild(rows,TEST_DEGREE,TEST_SLOT,TEST_SLOT,&abi9) == 1u);
	printf("rows %4u: query all-gather %u round(s), partial all-to-all %2u (slot slices) / %u (per-peer slots), wire %.2f + %.2f MB per layer per rank, %.0f MB per step\n",
		rows,abi8.query_rounds,abi8.partial_rounds,abi9.partial_rounds,(double)abi8.query_wire_bytes / 1.0e6,(double)abi8.partial_wire_bytes / 1.0e6,
		(double)(abi8.query_wire_bytes + abi8.partial_wire_bytes) * SPARK_GLM52_MODEL_LAYER_COUNT / 1.0e6);
	assert(abi8.query_rounds == query && abi9.query_rounds == query);
	assert(abi8.partial_rounds == scatter);
	assert(abi9.partial_rounds == peer);
	assert(abi8.query_wire_bytes == abi9.query_wire_bytes && abi8.partial_wire_bytes == abi9.partial_wire_bytes);
	assert(abi8.query_wire_bytes == (uint64_t)(TEST_DEGREE - 1u) * SparkGlm52KvShardQuerySequences(rows,TEST_DEGREE) * SPARK_GLM52_KV_SHARD_UNIT * 2u);
}

static void TestCapacity(void)
{
	static const uint32_t sequences[] = {16u,64u,128u,256u,64u,128u,256u,64u};
	static const uint32_t positions[] = {2048u,2048u,2048u,2048u,32768u,16384u,8192u,131072u};
	const uint64_t budget = UINT64_C(24) << 30;
	uint32_t index;
	for (index=0u; index<sizeof(sequences) / sizeof(sequences[0]); index++)
	{
		uint64_t tokens = (uint64_t)sequences[index] * positions[index];
		uint64_t replicated = tokens * SparkGlm52KvShardTokenBytes(1u);
		uint64_t latent = tokens * SparkGlm52KvShardTokenBytes(TEST_DEGREE);
		printf("%3u x %6u: replicated %6.1f GiB/rank %s | latent 1/16 + index replicated %5.1f GiB/rank %s\n",
			sequences[index],positions[index],(double)replicated / (1u << 30),replicated <= budget ? "fits" : "over",
			(double)latent / (1u << 30),latent <= budget ? "fits" : "over");
	}
	assert((uint64_t)16u * 2048u * SparkGlm52KvShardTokenBytes(1u) <= budget);
	assert((uint64_t)64u * 2048u * SparkGlm52KvShardTokenBytes(1u) <= budget);
	assert((uint64_t)128u * 2048u * SparkGlm52KvShardTokenBytes(1u) <= budget);
	assert((uint64_t)256u * 2048u * SparkGlm52KvShardTokenBytes(1u) > budget);
	assert((uint64_t)64u * 131072u * SparkGlm52KvShardTokenBytes(TEST_DEGREE) > budget);
	assert((uint64_t)64u * 32768u * SparkGlm52KvShardTokenBytes(TEST_DEGREE) <= budget);
	assert((uint64_t)256u * 8192u * SparkGlm52KvShardTokenBytes(TEST_DEGREE) <= budget);
}

int main(void)
{
	uint32_t layer,index_layers = 0u,rows;
	for (layer=0u; layer<SPARK_GLM52_MODEL_LAYER_COUNT; layer++)
		index_layers += SparkGlm52StagePackLayerHasFullIndexer(layer);
	assert(SparkGlm52KvShardIndexLayers() == index_layers);
	assert(index_layers == 21u);
	assert(SparkGlm52KvShardHeads(TEST_DEGREE) == 4u && SparkGlm52KvShardHeads(3u) == 0u);
	assert(SparkGlm52KvShardTokenBytes(1u) == 95232u);
	assert(SparkGlm52KvShardTokenBytes(TEST_DEGREE) == 5616u + 5376u);
	assert(SparkGlm52KvShardFits(1u,16u) == 0u && SparkGlm52KvShardFits(3u,16u) == 0u && SparkGlm52KvShardFits(TEST_DEGREE,0u) == 0u);
	for (rows=1u; rows<=1024u; rows++)
		assert(SparkGlm52KvShardFits(TEST_DEGREE,rows) == 1u);
	assert(SparkGlm52KvShardQuerySequences(1u,TEST_DEGREE) == 1u && SparkGlm52KvShardPartialSequences(1u,TEST_DEGREE) == 1u);
	assert(SparkGlm52KvShardQuerySequences(64u,TEST_DEGREE) == 24u && SparkGlm52KvShardPartialSequences(64u,TEST_DEGREE) == 43u);
	TestRounds(1u,1u,1u,1u);
	TestRounds(8u,1u,5u,1u);
	TestRounds(16u,1u,9u,1u);
	TestRounds(32u,1u,17u,2u);
	TestRounds(64u,2u,33u,3u);
	TestRounds(128u,3u,65u,5u);
	TestRounds(256u,5u,129u,9u);
	TestCapacity();
	printf("PASS glm52 kv shard plan: 4 heads per rank at TP16, 10,992 B/token/rank sharded vs 95,232 replicated, round and wire plan for slot slices and per-peer slots\n");
	return(0);
}
