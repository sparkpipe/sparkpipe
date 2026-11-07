#ifndef SPARK_HUGE_PAGES_H
#define SPARK_HUGE_PAGES_H

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static inline uint64_t SparkHugePageBytes(const void *address, uint64_t bytes)
{
	const uint64_t first = (uint64_t)(uintptr_t)address, last = first + bytes;
	uint64_t begin,end,kilobytes,total = 0u;
	uint32_t inside = 0u;
	char line[256];
	FILE *maps = fopen("/proc/self/smaps","r");
	if ( maps == 0 )
		return 0u;
	while ( fgets(line,sizeof(line),maps) != 0 )
	{
		if ( sscanf(line,"%" SCNx64 "-%" SCNx64 " ",&begin,&end) == 2 )
			inside = begin < last && end > first ? 1u : 0u;
		else if ( inside != 0u && sscanf(line,"AnonHugePages: %" SCNu64 " kB",&kilobytes) == 1 )
			total += kilobytes * 1024u;
	}
	(void)fclose(maps);
	return total;
}

#endif
