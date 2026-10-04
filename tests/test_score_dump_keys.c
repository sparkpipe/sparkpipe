#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "sparkpipe/spark_score_dump.h"

int main(void)
{
	SparkScoreDumpKeys keys;
	uint64_t first[4],second,restored;
	uint32_t position;
	assert(SparkScoreDumpKeysInitialize(&keys,2u,8u) == SPARK_STATUS_OK);
	for (position=0u; position<4u; position++)
		assert(SparkScoreDumpKeysAdvance(&keys,0u,100u,position,10u + position,&first[position]) == 1u);
	assert(SparkScoreDumpKeysAdvance(&keys,0u,200u,2u,12u,&restored) == 0u);
	assert(SparkScoreDumpKeysAdvance(&keys,0u,200u,3u,13u,&restored) == 0u);
	assert(SparkScoreDumpKeysAdvance(&keys,0u,200u,0u,10u,&second) == 1u && second == first[0]);
	assert(SparkScoreDumpKeysAdvance(&keys,0u,200u,1u,11u,&second) == 1u && second == first[1]);
	assert(SparkScoreDumpKeysAdvance(&keys,1u,300u,0u,10u,&second) == 1u && second == first[0]);
	assert(SparkScoreDumpKeysAdvance(&keys,1u,300u,5u,15u,&second) == 0u);
	SparkScoreDumpKeysDestroy(&keys);
	puts("PASS score-dump key chains restart keyless when a slot changes sequence mid-stream");
	return 0;
}
