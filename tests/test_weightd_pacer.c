#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "sparkpipe/spark_weightd_pacer.h"

typedef struct TestClock
{
	uint64_t now;
	uint32_t sleeps;
} TestClock;

static uint64_t TestNow(void *context)
{
	return(((TestClock *)context)->now);
}

static void TestSleep(void *context,uint64_t nanoseconds)
{
	TestClock *clock = (TestClock *)context;
	clock->now += nanoseconds;
	clock->sleeps++;
}

int main(void)
{
	SparkWeightdPacer pacer;
	TestClock clock = {1000u,0u};
	uint64_t chunk = UINT64_C(4194304),rate = UINT64_C(1073741824),start,index;
	assert(SparkWeightdPacerInitialize(&pacer,rate,TestNow,TestSleep,&clock) == SPARK_STATUS_OK);
	for (index=0u; index<64u; index++)
		assert(SparkWeightdPacerTake(&pacer,chunk,0u) == SPARK_STATUS_OK);
	assert(clock.sleeps == 0u && pacer.paced_bytes == 0u && clock.now == 1000u);
	start = clock.now;
	for (index=0u; index<256u; index++)
	{
		assert(SparkWeightdPacerTake(&pacer,chunk,1u) == SPARK_STATUS_OK);
		assert((unsigned __int128)index * chunk * UINT64_C(1000000000) <= (unsigned __int128)(clock.now - start) * rate);
		clock.now += 1000u;
	}
	assert(pacer.paced_bytes == 256u * chunk && clock.sleeps != 0u);
	assert((clock.now - start) >= (uint64_t)((unsigned __int128)255u * chunk * UINT64_C(1000000000) / rate));
	clock.sleeps = 0u;
	clock.now += UINT64_C(5000000000);
	assert(SparkWeightdPacerTake(&pacer,chunk,1u) == SPARK_STATUS_OK && clock.sleeps == 0u);
	assert(SparkWeightdPacerTake(&pacer,chunk,0u) == SPARK_STATUS_OK);
	assert(SparkWeightdPacerTake(&pacer,chunk,1u) == SPARK_STATUS_OK && clock.sleeps == 0u);
	assert(SparkWeightdPacerInitialize(&pacer,0u,TestNow,TestSleep,&clock) == SPARK_STATUS_OK);
	assert(SparkWeightdPacerTake(&pacer,chunk,0u) == SPARK_STATUS_OK);
	assert(SparkWeightdPacerTake(&pacer,chunk,1u) == SPARK_STATUS_CAPACITY_EXCEEDED && pacer.refused_count == 1u);
	puts("PASS weightd pacer: serving loads stay at or under the cap, idle loads run unpaced, and a serving load without a cap is refused");
	return(0);
}
