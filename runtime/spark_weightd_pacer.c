#include "sparkpipe/spark_weightd_pacer.h"

#include <string.h>

#include "sparkpipe/spark_error_site.h"

SparkStatus SparkWeightdPacerInitialize(SparkWeightdPacer *pacer,uint64_t bytes_per_second,SparkWeightdPacerClock clock,SparkWeightdPacerSleep sleep,void *context)
{
	if ( pacer == 0 || clock == 0 || sleep == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(pacer,0,sizeof(*pacer));
	pacer->bytes_per_second = bytes_per_second;
	pacer->clock = clock;
	pacer->sleep = sleep;
	pacer->context = context;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdPacerTake(SparkWeightdPacer *pacer,uint64_t bytes,uint32_t serving)
{
	unsigned __int128 duration;
	uint64_t now,start;
	if ( pacer == 0 || pacer->clock == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	now = pacer->clock(pacer->context);
	if ( serving == 0u )
	{
		pacer->next_ns = now;
		return(SPARK_STATUS_OK);
	}
	if ( pacer->bytes_per_second == 0u )
	{
		pacer->refused_count++;
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	start = pacer->next_ns > now ? pacer->next_ns : now;
	if ( start > now )
	{
		pacer->sleep(pacer->context,start - now);
		pacer->wait_ns += start - now;
	}
	duration = (unsigned __int128)bytes * UINT64_C(1000000000) / pacer->bytes_per_second;
	pacer->next_ns = duration > (unsigned __int128)(UINT64_MAX - start) ? UINT64_MAX : start + (uint64_t)duration;
	pacer->paced_bytes += bytes;
	return(SPARK_STATUS_OK);
}
