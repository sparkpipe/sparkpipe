#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_device.h"
#include "runtime/device/device_backend.h"
#include "runtime/device/nvrm/nvrm_gpu.h"

#define PAGES_TEXT_BYTES 16384u
#define PAGES_ORDERS_MAX 32u
#define PAGES_ZONE "Normal"

static uint32_t PagesOrder(uint64_t page_bytes, uint64_t huge_bytes)
{
	uint32_t order = 0u;
	while ( order < PAGES_ORDERS_MAX && (page_bytes << order) < huge_bytes )
		order++;
	return(order);
}

static uint32_t PagesLine(const char *line, uint32_t order, uint64_t *blocks)
{
	const char *zone = strstr(line,"zone");
	char name[32];
	uint32_t index = 0u;
	int used = 0;
	if ( zone == 0 || sscanf(zone,"zone %31s%n",name,&used) != 1 || strcmp(name,PAGES_ZONE) != 0 )
		return(0u);
	zone += used;
	while ( index < PAGES_ORDERS_MAX )
	{
		char *end = 0;
		const unsigned long long count = strtoull(zone,&end,10);
		if ( end == zone )
			break;
		if ( index >= order )
			*blocks += (uint64_t)count << (index - order);
		zone = end;
		index++;
	}
	return(index > order ? 1u : 0u);
}

uint32_t SparkNvrmFreeHugeBlocks(const char *buddyinfo, uint64_t page_bytes, uint64_t huge_bytes, uint64_t *blocks)
{
	const uint32_t order = PagesOrder(page_bytes,huge_bytes);
	const char *line = buddyinfo;
	uint32_t zones = 0u;
	*blocks = 0u;
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
			zones += PagesLine(copy,order,blocks);
		}
		line = next != 0 ? next + 1 : 0;
	}
	if ( zones == 0u )
		*blocks = 0u;
	return(zones);
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

static uint64_t PagesReserve(const char *path, char *text)
{
	if ( !PagesRead(path,text,PAGES_TEXT_BYTES) )
		return(0u);
	return((strtoull(text,0,10) * 1024u + SPARK_NVRM_HUGE_PAGE_BYTES - 1u) / SPARK_NVRM_HUGE_PAGE_BYTES);
}

uint64_t SparkNvrmHugePagesPull(uint64_t blocks)
{
#if defined(MADV_HUGEPAGE)
	const uint64_t huge = SPARK_NVRM_HUGE_PAGE_BYTES;
	const size_t bytes = (size_t)((blocks + 1u) * huge);
	volatile uint8_t *base;
	uint8_t *raw;
	uint64_t index;
	if ( blocks == 0u )
		return(0u);
	raw = (uint8_t *)mmap(0,bytes,PROT_READ | PROT_WRITE,MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,-1,0);
	if ( raw == MAP_FAILED )
		return(0u);
	base = (volatile uint8_t *)(((uintptr_t)raw + huge - 1u) & ~(uintptr_t)(huge - 1u));
	if ( madvise((void *)(uintptr_t)base,(size_t)(blocks * huge),MADV_HUGEPAGE) != 0 )
		blocks = 0u;
	for (index = 0u; index < blocks; index++)
		base[index * huge] = 1u;
	(void)munmap(raw,bytes);
	return(blocks);
#else
	(void)blocks;
	return(0u);
#endif
}

SparkStatus SparkNvrmHugePagesWait(const SparkNvrmHugeSource *source, uint64_t bytes, const char *owner)
{
	const long system_page = sysconf(_SC_PAGESIZE);
	const uint64_t page = source->page_bytes != 0u ? source->page_bytes : system_page > 0 ? (uint64_t)system_page : 0u;
	const uint64_t needed = (bytes + SPARK_NVRM_HUGE_PAGE_BYTES - 1u) / SPARK_NVRM_HUGE_PAGE_BYTES;
	const struct timespec step = { (time_t)(source->poll_ns / 1000000000u), (long)(source->poll_ns % 1000000000u) };
	const uint64_t start = SparkMonotonicNs();
	char *text = (char *)malloc(PAGES_TEXT_BYTES);
	char reason[256];
	uint64_t blocks = 0u,reserve = 0u;
	uint32_t pulled = 0u;
	if ( text == 0 )
		return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",owner,0,"out of host memory to read the free page lists"));
	for (;;)
	{
		if ( page == 0u || !PagesRead(source->buddyinfo,text,PAGES_TEXT_BYTES) || SparkNvrmFreeHugeBlocks(text,page,SPARK_NVRM_HUGE_PAGE_BYTES,&blocks) == 0u )
		{
			free(text);
			return(SPARK_STATUS_OK);
		}
		reserve = PagesReserve(source->min_free_kbytes,text);
		if ( blocks >= needed + reserve )
		{
			free(text);
			return(SPARK_STATUS_OK);
		}
		if ( SparkMonotonicNs() - start >= source->wait_ns )
			break;
		if ( source->pull != 0u && pulled == 0u )
		{
			(void)SparkNvrmHugePagesPull(needed + reserve);
			pulled = 1u;
			continue;
		}
		pulled = 0u;
		(void)nanosleep(&step,0);
	}
	free(text);
	snprintf(reason,sizeof(reason),"a %llu MiB object of 2 MiB pages needs %llu free 2 MiB blocks above the %llu-block min_free reserve; %llu were free after %llu s",
		(unsigned long long)((bytes + (1u << 20) - 1u) >> 20),(unsigned long long)needed,(unsigned long long)reserve,(unsigned long long)blocks,
		(unsigned long long)(source->wait_ns / 1000000000u));
	return(SparkDeviceRecordError(SPARK_STATUS_CAPACITY_EXCEEDED,"allocate",owner,0,reason));
}

SparkStatus SparkNvrmHugePagesReady(uint64_t bytes, const char *owner)
{
	static const SparkNvrmHugeSource kernel = { "/proc/buddyinfo","/proc/sys/vm/min_free_kbytes",0u,SPARK_NVRM_HUGE_WAIT_NS,SPARK_NVRM_HUGE_POLL_NS,1u };
	return(SparkNvrmHugePagesWait(&kernel,bytes,owner));
}
