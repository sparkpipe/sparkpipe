#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_gpu.h"

#define PAGES_BUDDYINFO "/proc/buddyinfo"
#define PAGES_MEMINFO "/proc/meminfo"
#define PAGES_TEXT_BYTES 16384u
#define PAGES_ORDERS_MAX 32u
#define PAGES_WAIT_STEP_NS 100000000ull
#define PAGES_WAIT_STEPS 20u

static uint32_t PagesOrder(uint64_t page_bytes, uint64_t huge_bytes)
{
	uint32_t order = 0u;
	while ( order < PAGES_ORDERS_MAX && (page_bytes << order) < huge_bytes )
		order++;
	return(order);
}

static uint64_t PagesLine(const char *line, uint32_t order)
{
	const char *zone = strstr(line,"zone");
	char name[32];
	uint64_t blocks = 0u;
	uint32_t index = 0u;
	int used = 0;
	if ( zone == 0 || sscanf(zone,"zone %31s%n",name,&used) != 1 || strcmp(name,"Movable") == 0 )
		return(0u);
	zone += used;
	while ( index < PAGES_ORDERS_MAX )
	{
		char *end = 0;
		const unsigned long long count = strtoull(zone,&end,10);
		if ( end == zone )
			break;
		if ( index >= order )
			blocks += (uint64_t)count << (index - order);
		zone = end;
		index++;
	}
	return(blocks);
}

uint64_t SparkNvrmFreeHugeBlocks(const char *buddyinfo, uint64_t page_bytes, uint64_t huge_bytes)
{
	const uint32_t order = PagesOrder(page_bytes,huge_bytes);
	const char *line = buddyinfo;
	uint64_t blocks = 0u;
	if ( buddyinfo == 0 || page_bytes == 0u || huge_bytes == 0u || order >= PAGES_ORDERS_MAX || (page_bytes << order) != huge_bytes )
		return(0u);
	while ( line != 0 && *line != '\0' )
	{
		const char *next = strchr(line,'\n');
		char copy[512];
		const size_t length = next != 0 ? (size_t)(next - line) : strlen(line);
		if ( length < sizeof(copy) )
		{
			memcpy(copy,line,length);
			copy[length] = '\0';
			blocks += PagesLine(copy,order);
		}
		line = next != 0 ? next + 1 : 0;
	}
	return(blocks);
}

static int PagesRead(const char *path, char *text, size_t capacity)
{
	const int fd = open(path,O_RDONLY | O_CLOEXEC);
	size_t done = 0u;
	if ( fd < 0 )
		return(0);
	while ( done + 1u < capacity )
	{
		const ssize_t got = read(fd,text + done,capacity - 1u - done);
		if ( got <= 0 )
			break;
		done += (size_t)got;
	}
	(void)close(fd);
	text[done] = '\0';
	return(done != 0u);
}

static uint64_t PagesAvailable(char *text)
{
	const char *line = PagesRead(PAGES_MEMINFO,text,PAGES_TEXT_BYTES) ? strstr(text,"MemAvailable:") : 0;
	return(line != 0 ? strtoull(line + strlen("MemAvailable:"),0,10) << 10 : 0u);
}

SparkStatus SparkNvrmHugePagesReady(uint64_t bytes, const char *owner)
{
	const long page = sysconf(_SC_PAGESIZE);
	const uint64_t needed = (bytes + SPARK_NVRM_HUGE_PAGE_BYTES - 1u) / SPARK_NVRM_HUGE_PAGE_BYTES;
	const struct timespec step = { 0, (long)PAGES_WAIT_STEP_NS };
	char *text = (char *)malloc(PAGES_TEXT_BYTES);
	char reason[128];
	uint64_t blocks = 0u;
	uint32_t attempt;
	if ( text == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",owner,0,"out of host memory to read the free page lists"));
	for (attempt = 0u; attempt <= PAGES_WAIT_STEPS; attempt++)
	{
		if ( page <= 0 || !PagesRead(PAGES_BUDDYINFO,text,PAGES_TEXT_BYTES) )
		{
			free(text);
			return(SPARK_STATUS_OK);
		}
		blocks = SparkNvrmFreeHugeBlocks(text,(uint64_t)page,SPARK_NVRM_HUGE_PAGE_BYTES);
		if ( blocks >= needed )
		{
			free(text);
			return(SPARK_STATUS_OK);
		}
		if ( attempt < PAGES_WAIT_STEPS )
			(void)nanosleep(&step,0);
	}
	snprintf(reason,sizeof(reason),"memory too fragmented: %llu free 2 MiB blocks after 2 s, %llu needed, %llu MiB available",
		(unsigned long long)blocks,(unsigned long long)needed,(unsigned long long)(PagesAvailable(text) >> 20));
	free(text);
	return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",owner,0,reason));
}
