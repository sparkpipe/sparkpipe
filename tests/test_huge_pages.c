#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "sparkpipe/spark_huge_pages.h"

#define HUGE_REGION_BYTES (64ull << 20)
#define HUGE_ALIGN (2ull << 20)

static int HugeModeAllowsAdvice(void)
{
	char mode[128] = "";
	FILE *file = fopen("/sys/kernel/mm/transparent_hugepage/enabled","r");
	if ( file == 0 )
		return 0;
	if ( fgets(mode,sizeof(mode),file) == 0 )
		mode[0] = 0;
	(void)fclose(file);
	return strstr(mode,"[never]") == 0 && mode[0] != 0;
}

int main(void)
{
	uint8_t *region = (uint8_t *)aligned_alloc(HUGE_ALIGN,HUGE_REGION_BYTES);
	uint64_t held;
	if ( region == 0 )
		return 1;
	if ( SparkHugePageBytes(region,HUGE_REGION_BYTES) != 0u )
	{
		fprintf(stderr,"FAIL an untouched region reports huge pages\n");
		return 1;
	}
	if ( !HugeModeAllowsAdvice() )
	{
		printf("PASS test_huge_pages: this kernel never gives huge pages, and an untouched region reports none\n");
		return 0;
	}
	if ( madvise(region,HUGE_REGION_BYTES,MADV_HUGEPAGE) != 0 )
		return 1;
	memset(region,1,HUGE_REGION_BYTES);
	held = SparkHugePageBytes(region,HUGE_REGION_BYTES);
	if ( held == 0u || held > HUGE_REGION_BYTES || (held % HUGE_ALIGN) != 0u )
	{
		fprintf(stderr,"FAIL a touched region asking for huge pages reports %llu bytes of them\n",(unsigned long long)held);
		return 1;
	}
	if ( SparkHugePageBytes(region + HUGE_REGION_BYTES + (1ull << 40),4096u) != 0u )
	{
		fprintf(stderr,"FAIL an address outside every mapping reports huge pages\n");
		return 1;
	}
	printf("PASS test_huge_pages: a touched region reports %llu MiB of huge pages\n",(unsigned long long)(held >> 20));
	free(region);
	return 0;
}
