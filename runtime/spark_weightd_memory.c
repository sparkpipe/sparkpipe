#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_error_site.h"

#include <stdio.h>

SparkStatus SparkWeightdMemInfoRead(const char *path,uint64_t *total_bytes,uint64_t *available_bytes)
{
	char line[256];
	unsigned long long value;
	uint32_t found = 0u;
	FILE *file = fopen(path,"r");
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	while ( found != 3u && fgets(line,sizeof(line),file) != 0 )
	{
		if ( sscanf(line,"MemTotal: %llu kB",&value) == 1 )
		{
			*total_bytes = (uint64_t)value * 1024u;
			found |= 1u;
		}
		else if ( sscanf(line,"MemAvailable: %llu kB",&value) == 1 )
		{
			*available_bytes = (uint64_t)value * 1024u;
			found |= 2u;
		}
	}
	(void)fclose(file);
	if ( found != 3u )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return(SPARK_STATUS_OK);
}

uint64_t SparkWeightdKvMemoryMargin(uint64_t total_bytes)
{
	return(total_bytes / 100u * SPARK_WEIGHTD_KV_EARLYOOM_PERCENT + SPARK_WEIGHTD_KV_RUNTIME_HEADROOM_BYTES);
}
