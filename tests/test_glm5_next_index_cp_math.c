#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_glm5_next_index_cp.h"

#define TEST_POOLS_MAX 6000u

static uint8_t test_seen[TEST_POOLS_MAX];

static void TestCover(uint32_t pools,uint32_t degree)
{
	uint32_t rank,local,pool,stride,count;
	stride = SparkGlm5NextIndexCpLocalStride(pools,degree);
	memset(test_seen,0,sizeof(test_seen));
	count = 0u;
	for (rank=0u; rank<degree; rank++)
		for (local=0u; local<stride; local++)
		{
			pool = SparkGlm5NextIndexCpGlobalPool(local,rank,degree);
			if ( pool >= pools )
				continue;
			assert(test_seen[pool] == 0u);
			test_seen[pool] = 1u;
			count++;
			assert(SparkGlm5NextIndexCpOwner(pool,degree) == rank);
			assert(SparkGlm5NextIndexCpLocalPool(pool,degree) == local);
		}
	assert(count == pools);
	assert(degree <= 1u || (uint64_t)stride * degree < (uint64_t)pools + (uint64_t)degree * SPARK_GLM5_NEXT_INDEX_CP_POOLS_PER_PAGE);
}

int main(void)
{
	static const uint32_t degrees[] = {1u,2u,3u,4u,5u,8u,15u,16u};
	uint32_t degree,pools,rows;
	for (degree=0u; degree<sizeof(degrees) / sizeof(degrees[0]); degree++)
		for (pools=1u; pools<TEST_POOLS_MAX; pools+=pools < 600u ? 1u : 37u)
			TestCover(pools,degrees[degree]);
	assert(SparkGlm5NextIndexCpActive(SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K,16u) == 0u);
	assert(SparkGlm5NextIndexCpActive(SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K + 1u,16u) == 1u);
	assert(SparkGlm5NextIndexCpActive(1u << 20u,1u) == 0u);
	for (rows=1u; rows<=128u; rows++)
		assert(SparkGlm5NextIndexCpFits(131072u,16u,rows) == 1u);
	assert(SparkGlm5NextIndexCpFits(262144u,16u,128u) == 0u);
	assert(SparkGlm5NextIndexCpFits(32768u,2u,8u) == 0u);
	assert(SparkGlm5NextIndexCpFits(16384u,2u,8u) == 1u);
	assert(SparkGlm5NextIndexCpGatherSequences(128u,512u) == 32u);
	puts("test_glm5_next_index_cp_math: ownership covers every pool exactly once for degrees 1-16; gather fits up to 128K context at TP16");
	return(0);
}
