#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_error_site.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/file.h>
#include <unistd.h>

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

SparkStatus SparkWeightdNodeMemoryAdmit(const char *meminfo_path,uint64_t bytes,const char *what,const char *model)
{
	uint64_t total = 0u,available = 0u,margin;
	SparkStatus status = SparkWeightdMemInfoRead(meminfo_path,&total,&available);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"node memory refused: %s for model=%s needs %llu bytes and %s gives no MemTotal and MemAvailable\n",
		    what,model,(unsigned long long)bytes,meminfo_path);
		SPARK_RETURN(status);
	}
	margin = SparkWeightdKvMemoryMargin(total);
	if ( available >= margin && bytes <= available - margin )
		return(SPARK_STATUS_OK);
	fprintf(stderr,"node memory refused: %s for model=%s needs %llu bytes beside the %llu-byte margin and MemAvailable is %llu of %llu; other engines on this node, or one still exiting, hold the rest\n",
	    what,model,(unsigned long long)bytes,(unsigned long long)margin,(unsigned long long)available,(unsigned long long)total);
	SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
}

SparkStatus SparkWeightdNodeMemoryLock(const char *lock_path,int *lock_fd)
{
	int fd = open(lock_path,O_RDONLY | O_CLOEXEC);
	if ( fd < 0 && errno == ENOENT )
		fd = open(lock_path,O_RDONLY | O_CREAT | O_CLOEXEC,0644);
	if ( fd < 0 && errno == EACCES )
		fd = open(lock_path,O_RDONLY | O_CLOEXEC);
	if ( fd < 0 )
	{
		fprintf(stderr,"node memory lock %s cannot open: errno=%d\n",lock_path,errno);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	while ( flock(fd,LOCK_EX) != 0 )
		if ( errno != EINTR )
		{
			fprintf(stderr,"node memory lock %s cannot lock: errno=%d\n",lock_path,errno);
			(void)close(fd);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
	*lock_fd = fd;
	return(SPARK_STATUS_OK);
}

void SparkWeightdNodeMemoryUnlock(int lock_fd)
{
	(void)flock(lock_fd,LOCK_UN);
	(void)close(lock_fd);
}
