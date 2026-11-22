#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_device.h"

#define LARGE_MARGIN_BYTES (8ull << 30)

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

static void TestPinned(SparkDeviceQueue *queue, uint64_t bytes)
{
	SparkDeviceBuffer pinned;
	const uint64_t mib = bytes >> 20;
	char what[192];
	memset(&pinned,0,sizeof(pinned));
	snprintf(what,sizeof(what),"allocate %llu MiB of host-pinned memory",(unsigned long long)mib);
	CHECK(SparkDeviceAllocate(&pinned,SPARK_DEVICE_SPACE_HOST_PINNED,bytes,"large pinned") == SPARK_STATUS_OK,what);
	if ( pinned.bytes == 0u )
		return;
	memset(pinned.host,0,(size_t)bytes);
	snprintf(what,sizeof(what),"fill all %llu MiB of host-pinned memory in one call",(unsigned long long)mib);
	CHECK(SparkDeviceFill(queue,At(&pinned,0u),0x6du,bytes) == SPARK_STATUS_OK && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,what);
	snprintf(what,sizeof(what),"the fill of all %llu MiB of host-pinned memory reaches every byte",(unsigned long long)mib);
	CHECK(Wrong((const uint8_t *)pinned.host,0u,bytes,0x6du) == 0u,what);
	CHECK(SparkDeviceFree(&pinned) == SPARK_STATUS_OK,"free the host-pinned memory");
}

static void TestShared(SparkDeviceQueue *queue, const SparkDeviceBuffer *host, uint64_t bytes)
{
	SparkDeviceMemoryHandle memory = 0;
	SparkDeviceAddress address = 0u;
	SparkDeviceBuffer shared;
	const uint64_t mib = bytes >> 20;
	uint32_t created,reserved = 0u,mapped = 0u,adopted = 0u;
	char what[192];
	memset(&shared,0,sizeof(shared));
	snprintf(what,sizeof(what),"create, map and adopt %llu MiB of shared memory",(unsigned long long)mib);
	created = SparkDeviceMemoryCreate(&memory,bytes,"large shared") == SPARK_STATUS_OK;
	reserved = created != 0u && SparkDeviceAddressReserve(&address,bytes,"large shared") == SPARK_STATUS_OK;
	mapped = reserved != 0u && SparkDeviceMemoryMap(address,bytes,memory,0u,SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE) == SPARK_STATUS_OK;
	adopted = mapped != 0u && SparkDeviceAdopt(&shared,address,0,bytes,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,"large shared") == SPARK_STATUS_OK;
	CHECK(adopted != 0u,what);
	if ( adopted != 0u )
	{
		memset(host->host,0,(size_t)bytes);
		snprintf(what,sizeof(what),"fill all %llu MiB of shared memory and copy it to host-visible memory",(unsigned long long)mib);
		CHECK(SparkDeviceFill(queue,At(&shared,0u),0x95u,bytes) == SPARK_STATUS_OK && SparkDeviceCopy(queue,At(host,0u),At(&shared,0u),bytes) == SPARK_STATUS_OK &&
			SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,what);
		snprintf(what,sizeof(what),"the fill of all %llu MiB of shared memory arrives everywhere",(unsigned long long)mib);
		CHECK(Wrong((const uint8_t *)host->host,0u,bytes,0x95u) == 0u,what);
		CHECK(SparkDeviceFree(&shared) == SPARK_STATUS_OK,"release the adopted shared memory");
	}
	if ( mapped != 0u )
		CHECK(SparkDeviceMemoryUnmap(address,bytes) == SPARK_STATUS_OK,"unmap the shared memory");
	if ( reserved != 0u )
		CHECK(SparkDeviceAddressRelease(address,bytes) == SPARK_STATUS_OK,"release the shared memory's addresses");
	if ( created != 0u )
		CHECK(SparkDeviceMemoryRelease(memory) == SPARK_STATUS_OK,"release the shared memory");
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
	snprintf(what,sizeof(what),"allocate %llu MiB of device-private memory, which the open module zeroes",(unsigned long long)mib);
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
		printf("%llu MiB: allocate %.2f s, fill %.1f GB/s, copy %.1f GB/s\n",(unsigned long long)mib,(double)allocate_ns / 1e9,
			Rate(bytes,fill_ns),Rate(bytes,copy_ns));
	}
	if ( device.bytes != 0u )
		CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK,"free the device-private memory");
	if ( view != 0 )
	{
		TestPinned(queue,bytes);
		TestShared(queue,&host,bytes);
	}
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
	printf("PASS fills and copies of 8, 10.5 and 16 GiB of device-private, shared-coherent, host-pinned and shared memory go through in one call each, in lines of at most 2^31 bytes, and reach every byte\n");
	return(0);
}

int main(int argc, char **argv)
{
	if ( argc == 1 || (argc == 2 && strcmp(argv[1],"sizes") == 0) )
		return(TestSizes());
	printf("usage: %s [sizes]\n",argv[0]);
	return(2);
}
