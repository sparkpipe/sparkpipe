#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_wait.h"

#define LARGE_MARGIN_BYTES (8ull << 30)
#define LARGE_HELD_BYTES (4ull << 30)
#define LARGE_CHECK_BYTES (64ull << 20)

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			const SparkDeviceError *last = SparkDeviceLastError(); \
			printf("FAIL %s:%d %s (last error: %s %s)\n",__FILE__,__LINE__,what,last->operation,last->text); \
			failures++; \
		} \
	} while (0)

static SparkDeviceRange At(const SparkDeviceBuffer *buffer, uint64_t offset)
{
	SparkDeviceRange range;
	range.buffer = buffer;
	range.offset = offset;
	return(range);
}

static uint64_t Now(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static uint64_t Available(void)
{
	char line[128];
	unsigned long long kilobytes = 0u;
	FILE *meminfo = fopen("/proc/meminfo","r");
	if ( meminfo == 0 )
		return(0u);
	while ( fgets(line,sizeof(line),meminfo) != 0 )
		if ( sscanf(line,"MemAvailable: %llu kB",&kilobytes) == 1 )
			break;
	fclose(meminfo);
	return((uint64_t)kilobytes << 10);
}

static uint64_t Wrong(const uint8_t *bytes, uint64_t begin, uint64_t end, uint8_t value)
{
	const uint64_t pattern = value * 0x0101010101010101ull;
	uint64_t index = begin,wrong = 0u;
	while ( index < end && (index & 7u) != 0u )
		wrong += bytes[index++] != value;
	for ( ; index + 8u <= end; index += 8u )
	{
		uint64_t word;
		memcpy(&word,bytes + index,sizeof(word));
		wrong += word != pattern;
	}
	while ( index < end )
		wrong += bytes[index++] != value;
	return(wrong);
}

static double Rate(uint64_t bytes, uint64_t ns)
{
	return(ns != 0u ? (double)bytes / (double)ns : 0.0);
}

static void TestSize(SparkDeviceQueue *queue, uint64_t bytes)
{
	SparkDeviceBuffer device,host;
	const uint64_t mib = bytes >> 20;
	uint64_t start,allocate_ns = 0u,fill_ns = 0u,copy_ns = 0u;
	uint8_t *view;
	char what[192];
	memset(&device,0,sizeof(device));
	memset(&host,0,sizeof(host));
	start = Now();
	snprintf(what,sizeof(what),"allocate %llu MiB of device-private memory, which the copy engine zero-fills",(unsigned long long)mib);
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,"large device") == SPARK_STATUS_OK,what);
	allocate_ns = Now() - start;
	snprintf(what,sizeof(what),"allocate %llu MiB of host-visible memory",(unsigned long long)mib);
	CHECK(SparkDeviceAllocate(&host,SPARK_DEVICE_SPACE_SHARED_COHERENT,bytes,"large host") == SPARK_STATUS_OK,what);
	view = (uint8_t *)host.host;
	if ( device.bytes != 0u && view != 0 )
	{
		view[0] = 0x11u;
		view[bytes - 1u] = 0x22u;
		snprintf(what,sizeof(what),"fill %llu MiB less two bytes at an odd offset",(unsigned long long)mib);
		CHECK(SparkDeviceFill(queue,At(&host,1u),0x5au,bytes - 2u) == SPARK_STATUS_OK && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
			what);
		snprintf(what,sizeof(what),"the odd-offset fill of %llu MiB reaches every byte it names and no other",(unsigned long long)mib);
		CHECK(view[0] == 0x11u && view[bytes - 1u] == 0x22u && Wrong(view,1u,bytes - 1u,0x5au) == 0u,what);
		start = Now();
		snprintf(what,sizeof(what),"copy %llu MiB of device-private memory to host-visible memory in one call",(unsigned long long)mib);
		CHECK(SparkDeviceCopy(queue,At(&host,0u),At(&device,0u),bytes) == SPARK_STATUS_OK && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
			what);
		copy_ns = Now() - start;
		snprintf(what,sizeof(what),"the allocation's zero fill reached all %llu MiB, and the copy of all of it arrives",(unsigned long long)mib);
		CHECK(Wrong(view,0u,bytes,0u) == 0u,what);
		start = Now();
		snprintf(what,sizeof(what),"fill all %llu MiB of device-private memory in one call",(unsigned long long)mib);
		CHECK(SparkDeviceFill(queue,At(&device,0u),0xc3u,bytes) == SPARK_STATUS_OK && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,what);
		fill_ns = Now() - start;
		snprintf(what,sizeof(what),"copy the filled %llu MiB back in one call",(unsigned long long)mib);
		CHECK(SparkDeviceCopy(queue,At(&host,0u),At(&device,0u),bytes) == SPARK_STATUS_OK && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
			what);
		snprintf(what,sizeof(what),"the fill of all %llu MiB, copied back whole, arrives everywhere",(unsigned long long)mib);
		CHECK(Wrong(view,0u,bytes,0xc3u) == 0u,what);
		printf("%llu MiB: allocate with zero fill %.2f s, fill %.1f GB/s, copy %.1f GB/s\n",(unsigned long long)mib,(double)allocate_ns / 1e9,
			Rate(bytes,fill_ns),Rate(bytes,copy_ns));
	}
	if ( device.bytes != 0u )
		CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK,"free the device-private memory");
	if ( host.bytes != 0u )
		CHECK(SparkDeviceFree(&host) == SPARK_STATUS_OK,"free the host-visible memory");
}

static int TestSizes(void)
{
	static const uint64_t sizes[] = {8ull << 30, (21ull << 30) / 2u, 16ull << 30};
	SparkDeviceQueue *queue = 0;
	uint32_t index;
	const uint64_t needed = 2u * sizes[2] + LARGE_MARGIN_BYTES,available = Available();
	if ( available < needed )
	{
		printf("FAIL the large transfers need %llu GiB of available memory; the host has %llu GiB\n",(unsigned long long)(needed >> 30),
			(unsigned long long)(available >> 30));
		return(1);
	}
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue,"large transfers") != SPARK_STATUS_OK )
	{
		printf("FAIL set up the device: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	for (index = 0u; index < sizeof(sizes) / sizeof(sizes[0]); index++)
		TestSize(queue,sizes[index]);
	CHECK(SparkDeviceQueueDestroy(queue) == SPARK_STATUS_OK && SparkDeviceShutdown() == SPARK_STATUS_OK,"close the device");
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS fills and copies of 8, 10.5 and 16 GiB go through in one call each, in lines of at most 2^31 bytes, and reach every byte\n");
	return(0);
}

static int TestHeld(void)
{
	SparkDeviceBuffer device,host;
	SparkDeviceQueue *queue = 0;
	SparkStatus status;
	memset(&device,0,sizeof(device));
	memset(&host,0,sizeof(host));
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK )
	{
		printf("FAIL set up the device: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	SparkNvrmWaitStallSet(0u);
	status = SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,LARGE_HELD_BYTES,"held");
	SparkNvrmWaitStallSet(SPARK_NVRM_WAIT_STALL_NS);
	printf("allocation whose wait gave up: %s\n",SparkDeviceLastError()->text);
	CHECK(status == SPARK_STATUS_IO_ERROR && device.bytes == 0u && strstr(SparkDeviceLastError()->text,"no GPU progress") != 0 &&
		strstr(SparkDeviceLastError()->text,"stay held") != 0,
		"a zero fill still running when its wait gives up fails the allocation, keeps the memory held and says so");
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_OK,"the device closes once the held memory's fill has finished, and releases it then");
	CHECK(SparkDeviceInitialize(0) == SPARK_STATUS_OK && SparkDeviceQueueCreate(&queue,"after held") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&host,SPARK_DEVICE_SPACE_SHARED_COHERENT,LARGE_CHECK_BYTES,"after held") == SPARK_STATUS_OK &&
		SparkDeviceFill(queue,At(&host,0u),0x7eu,LARGE_CHECK_BYTES) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK && Wrong((const uint8_t *)host.host,0u,LARGE_CHECK_BYTES,0x7eu) == 0u &&
		SparkDeviceFree(&host) == SPARK_STATUS_OK && SparkDeviceQueueDestroy(queue) == SPARK_STATUS_OK && SparkDeviceShutdown() == SPARK_STATUS_OK,
		"the device opens again and fills memory after the held allocation was released");
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS an allocation whose zero fill outlives its wait stays held until the fill ends, and the device works after it\n");
	return(0);
}

int main(int argc, char **argv)
{
	if ( argc == 2 && strcmp(argv[1],"sizes") == 0 )
		return(TestSizes());
	if ( argc == 2 && strcmp(argv[1],"held") == 0 )
		return(TestHeld());
	printf("usage: %s sizes|held\n",argv[0]);
	return(2);
}
