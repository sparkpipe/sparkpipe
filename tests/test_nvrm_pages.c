#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_gpu.h"

static uint32_t pages_failures;

static void Check(int condition, const char *what)
{
	if ( condition )
		return;
	printf("FAIL %s\n",what);
	pages_failures++;
}

static uint64_t Blocks(const char *buddyinfo, uint64_t page_bytes, uint64_t huge_bytes, uint32_t *zones)
{
	uint64_t blocks = 99u;
	*zones = SparkNvrmFreeHugeBlocks(buddyinfo,page_bytes,huge_bytes,&blocks);
	return(blocks);
}

static void TestCount(void)
{
	static const char four_kib[] =
		"Node 0, zone      DMA      1      1      1      1      1      1      1      1      1      2      3\n"
		"Node 0, zone    DMA32      0      0      0      0      0      0      0      0      0      5      0\n"
		"Node 0, zone   Normal    100    200    300     40     50     60     70     80     90      7     11\n"
		"Node 0, zone  Movable      0      0      0      0      0      0      0      0      0    100    100\n";
	static const char deep_orders[] =
		"Node 0, zone      DMA     14      9     10      8      8     12     12      8      9      8      7      7      4     15 \n"
		"Node 0, zone   Normal  62328 102479  44487  28217  29382  23274  64667   9807   4225   2874   1907   1048    692    573 \n";
	static const char two_nodes[] =
		"Node 0, zone   Normal      0      0      0      0      0      0      0      0      0      3      1\n"
		"Node 1, zone   Normal      0      0      0      0      0      0      0      0      0      4      2\n";
	static const char large_pages[] = "Node 0, zone   Normal      5      4      3      2      1      6      7      8      9     10     11     12     13     14\n";
	uint32_t zones = 0u;
	Check(Blocks(four_kib,4096u,2u << 20,&zones) == 29u && zones == 1u,
		"4 KiB pages: the Normal zone's order-9 blocks count once and order-10 blocks twice; DMA, DMA32 and Movable count nothing");
	Check(Blocks(deep_orders,4096u,2u << 20,&zones) == 25584u && zones == 1u,
		"4 KiB pages up to order 13 (a GB10's lists): each larger Normal block counts as the 2 MiB blocks it holds, DMA's do not");
	Check(Blocks(two_nodes,4096u,2u << 20,&zones) == 13u && zones == 2u,"the Normal zones of every node add up");
	Check(Blocks(large_pages,65536u,2u << 20,&zones) == 6652u && zones == 1u,"64 KiB pages: 2 MiB is order 5");
	Check(Blocks("",4096u,2u << 20,&zones) == 0u && zones == 0u && Blocks("Node 0, zone\n",4096u,2u << 20,&zones) == 0u && zones == 0u &&
		Blocks("garbage\nNode 0, zone Normal x 1 2\n",4096u,2u << 20,&zones) == 0u && zones == 0u &&
		Blocks("Node 0, zone      DMA      1      1      1      1      1      1      1      1      1      2      3\n",4096u,2u << 20,&zones) == 0u && zones == 0u,
		"empty, malformed and Normal-less lists count no zone");
	Check(Blocks(four_kib,4096u,3u << 20,&zones) == 0u && zones == 0u && Blocks(four_kib,0u,2u << 20,&zones) == 0u && zones == 0u,
		"a block size that is not a power-of-two number of pages counts nothing");
}

typedef struct PagesFiles
{
	char directory[64];
	char buddyinfo[96];
	char reserve[96];
} PagesFiles;

static void PagesWrite(const char *path, const char *text)
{
	char temporary[128];
	FILE *file;
	snprintf(temporary,sizeof(temporary),"%s.new",path);
	file = fopen(temporary,"w");
	if ( file == 0 )
		return;
	fputs(text,file);
	fclose(file);
	(void)rename(temporary,path);
}

static void PagesNormal(const char *path, uint64_t blocks)
{
	char text[512];
	snprintf(text,sizeof(text),"Node 0, zone      DMA      0      0      0      0      0      0      0      0      0    400      0\n"
		"Node 0, zone   Normal    100    200    300     40     50     60     70     80     90 %6llu      0\n",(unsigned long long)blocks);
	PagesWrite(path,text);
}

static int PagesOpen(PagesFiles *files)
{
	snprintf(files->directory,sizeof(files->directory),"/tmp/test_nvrm_pages.XXXXXX");
	if ( mkdtemp(files->directory) == 0 )
		return(0);
	snprintf(files->buddyinfo,sizeof(files->buddyinfo),"%s/buddyinfo",files->directory);
	snprintf(files->reserve,sizeof(files->reserve),"%s/min_free_kbytes",files->directory);
	PagesWrite(files->reserve,"1048576\n");
	return(1);
}

static void PagesClose(PagesFiles *files)
{
	(void)unlink(files->buddyinfo);
	(void)unlink(files->reserve);
	(void)rmdir(files->directory);
}

typedef struct PagesLater
{
	const char *path;
	uint64_t blocks;
	uint64_t delay_ns;
} PagesLater;

static void *PagesLaterRun(void *argument)
{
	const PagesLater *later = (const PagesLater *)argument;
	const struct timespec delay = { (time_t)(later->delay_ns / 1000000000u), (long)(later->delay_ns % 1000000000u) };
	(void)nanosleep(&delay,0);
	PagesNormal(later->path,later->blocks);
	return(0);
}

static void TestWait(void)
{
	PagesFiles files;
	SparkNvrmHugeSource source;
	pthread_t thread;
	PagesLater later;
	uint64_t start,elapsed;
	if ( !PagesOpen(&files) )
	{
		Check(0,"make a directory for the fake kernel files");
		return;
	}
	source.buddyinfo = files.buddyinfo;
	source.min_free_kbytes = files.reserve;
	source.page_bytes = 4096u;
	source.poll_ns = SPARK_NVRM_HUGE_POLL_NS;
	source.wait_ns = 1000000000u;
	source.pull = 0u;
	PagesNormal(files.buddyinfo,544u);
	start = SparkMonotonicNs();
	Check(SparkNvrmHugePagesWait(&source,64u << 20,"pages") == SPARK_STATUS_OK && SparkMonotonicNs() - start < SPARK_NVRM_HUGE_POLL_NS,
		"a 64 MiB object goes ahead at once when the Normal zone has its 32 blocks above the 512-block reserve of a 1 GiB min_free_kbytes");
	PagesNormal(files.buddyinfo,543u);
	start = SparkMonotonicNs();
	Check(SparkNvrmHugePagesWait(&source,64u << 20,"pages") == SPARK_STATUS_CAPACITY_EXCEEDED &&
		strstr(SparkDeviceLastError()->text,"a 64 MiB object of 2 MiB pages needs 32 free 2 MiB blocks above the 512-block min_free reserve; 543 were free after 1 s") != 0,
		"one block short of the target, the object is refused after the bound, naming its need, the reserve and the free blocks");
	elapsed = SparkMonotonicNs() - start;
	Check(elapsed >= source.wait_ns && elapsed < source.wait_ns + 500000000u,"the refusal comes at the bound, not before and not long after");
	later.path = files.buddyinfo;
	later.blocks = 544u;
	later.delay_ns = 200000000u;
	source.wait_ns = 5000000000ull;
	start = SparkMonotonicNs();
	Check(pthread_create(&thread,0,PagesLaterRun,&later) == 0,"start the fake compaction");
	Check(SparkNvrmHugePagesWait(&source,64u << 20,"pages") == SPARK_STATUS_OK,"the object goes ahead once compaction frees its blocks");
	elapsed = SparkMonotonicNs() - start;
	(void)pthread_join(thread,0);
	Check(elapsed >= later.delay_ns && elapsed < later.delay_ns + 300000000u,"the wait ends within a few polls of the blocks appearing");
	PagesWrite(files.reserve,"45056\n");
	PagesNormal(files.buddyinfo,54u);
	Check(SparkNvrmHugePagesWait(&source,64u << 20,"pages") == SPARK_STATUS_OK,"the boot reserve of 45056 KiB holds back 22 blocks");
	source.wait_ns = 0u;
	PagesNormal(files.buddyinfo,53u);
	Check(SparkNvrmHugePagesWait(&source,64u << 20,"pages") == SPARK_STATUS_CAPACITY_EXCEEDED &&
		strstr(SparkDeviceLastError()->text,"needs 32 free 2 MiB blocks above the 22-block min_free reserve; 53 were free") != 0,"and 53 blocks are one short");
	PagesWrite(files.buddyinfo,"Node 0, zone      DMA      0      0      0      0      0      0      0      0      0    900      0\n");
	Check(SparkNvrmHugePagesWait(&source,64u << 20,"pages") == SPARK_STATUS_OK,"lists without a Normal zone leave the decision to the module");
	(void)unlink(files.buddyinfo);
	Check(SparkNvrmHugePagesWait(&source,64u << 20,"pages") == SPARK_STATUS_OK,"so does an unreadable buddyinfo");
	PagesClose(&files);
}

static void TestPull(void)
{
#if defined(__linux__)
	Check(SparkNvrmHugePagesPull(4u) == 4u && SparkNvrmHugePagesPull(0u) == 0u,"the pull faults the 2 MiB blocks it is asked for in a huge-page mapping and releases them");
#else
	Check(SparkNvrmHugePagesPull(4u) == 0u,"without huge-page advice the pull makes nothing");
#endif
}

int main(void)
{
	TestCount();
	TestWait();
	TestPull();
	if ( pages_failures != 0u )
	{
		printf("FAIL (%u)\n",pages_failures);
		return(1);
	}
	printf("PASS the free 2 MiB blocks come from the Normal zones' buddy lists, an object of 2 MiB pages waits until they cover it and the min_free reserve, then goes ahead or is refused at the bound, and the pull faults and releases huge pages\n");
	return(0);
}
