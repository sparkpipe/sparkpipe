#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_device.h"

#define LARGE_MARGIN_BYTES (8ull << 30)
#define BOUNDARY_BYTES (9ull << 30)
#define BOUNDARY_WINDOW_BYTES (64ull << 10)
#define BOUNDARY_WINDOWS 4u
#define BOUNDARY_SLOTS_BYTES (6u * BOUNDARY_WINDOW_BYTES)

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

typedef struct BoundaryWindow
{
	uint64_t offset;
	uint64_t bytes;
} BoundaryWindow;

static void BoundaryWindows(uint64_t bytes, BoundaryWindow windows[BOUNDARY_WINDOWS])
{
	windows[0].offset = 0u;
	windows[0].bytes = BOUNDARY_WINDOW_BYTES;
	windows[1].offset = (4ull << 30) - BOUNDARY_WINDOW_BYTES;
	windows[1].bytes = 2u * BOUNDARY_WINDOW_BYTES;
	windows[2].offset = (8ull << 30) - BOUNDARY_WINDOW_BYTES;
	windows[2].bytes = 2u * BOUNDARY_WINDOW_BYTES;
	windows[3].offset = bytes - BOUNDARY_WINDOW_BYTES;
	windows[3].bytes = BOUNDARY_WINDOW_BYTES;
}

static uint32_t BoundaryWord(uint64_t offset, uint32_t seed)
{
	const uint64_t mixed = ((offset >> 2) ^ ((uint64_t)seed << 40)) * 0x9e3779b97f4a7c15ull;
	return((uint32_t)(mixed >> 32) ^ (uint32_t)(offset >> 2));
}

static void BoundaryWrite(uint8_t *cpu, const BoundaryWindow *window, uint32_t seed)
{
	uint64_t offset;
	for (offset = window->offset; offset < window->offset + window->bytes; offset += sizeof(uint32_t))
	{
		const uint32_t word = BoundaryWord(offset,seed);
		memcpy(cpu + offset,&word,sizeof(word));
	}
}

static uint64_t BoundaryFirstWrong(const uint8_t *copied, const BoundaryWindow *window, uint32_t seed, uint64_t *wrong_words)
{
	uint64_t offset,first = UINT64_MAX;
	*wrong_words = 0u;
	for (offset = 0u; offset < window->bytes; offset += sizeof(uint32_t))
	{
		uint32_t word;
		memcpy(&word,copied + offset,sizeof(word));
		if ( word != BoundaryWord(window->offset + offset,seed) )
		{
			first = first == UINT64_MAX ? window->offset + offset : first;
			(*wrong_words)++;
		}
	}
	return(first);
}

static void BoundaryCheck(SparkDeviceQueue *queue, const char *kind, const SparkDeviceBuffer *object, uint8_t *cpu, uint32_t seed,
	const SparkDeviceBuffer *device, const SparkDeviceBuffer *pageable)
{
	BoundaryWindow windows[BOUNDARY_WINDOWS];
	uint64_t slot = 0u;
	uint32_t index,copied = 1u;
	char what[256];
	BoundaryWindows(BOUNDARY_BYTES,windows);
	for (index = 0u; index < BOUNDARY_WINDOWS; index++)
		BoundaryWrite(cpu,&windows[index],seed);
	memset(pageable->host,0,(size_t)BOUNDARY_SLOTS_BYTES);
	for (index = 0u; index < BOUNDARY_WINDOWS && copied != 0u; index++)
	{
		copied = SparkDeviceCopy(queue,At(device,slot),At(object,windows[index].offset),windows[index].bytes) == SPARK_STATUS_OK;
		slot += windows[index].bytes;
	}
	copied = copied != 0u && SparkDeviceCopy(queue,At(pageable,0u),At(device,0u),slot) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK;
	snprintf(what,sizeof(what),"the GPU copies the windows of 9 GiB of %s to device memory and on to pageable memory",kind);
	CHECK(copied != 0u,what);
	if ( copied == 0u )
		return;
	slot = 0u;
	for (index = 0u; index < BOUNDARY_WINDOWS; index++)
	{
		uint64_t wrong = 0u;
		const uint64_t first = BoundaryFirstWrong((const uint8_t *)pageable->host + slot,&windows[index],seed,&wrong);
		snprintf(what,sizeof(what),"the GPU reads 9 GiB of %s at [%llu, %llu) as the CPU wrote it: %llu of %llu words differ, the first at offset %llu",kind,
			(unsigned long long)windows[index].offset,(unsigned long long)(windows[index].offset + windows[index].bytes),(unsigned long long)wrong,
			(unsigned long long)(windows[index].bytes / sizeof(uint32_t)),(unsigned long long)(first == UINT64_MAX ? 0u : first));
		CHECK(wrong == 0u,what);
		slot += windows[index].bytes;
	}
}

static uint32_t BoundaryAllocated(SparkDeviceQueue *queue, SparkDeviceSpace space, const char *kind, uint32_t seed, const SparkDeviceBuffer *device,
	const SparkDeviceBuffer *pageable, uint32_t needs_host)
{
	SparkDeviceBuffer object;
	uint32_t checked = 0u;
	char what[192];
	memset(&object,0,sizeof(object));
	snprintf(what,sizeof(what),"allocate 9 GiB of %s",kind);
	CHECK(SparkDeviceAllocate(&object,space,BOUNDARY_BYTES,kind) == SPARK_STATUS_OK,what);
	if ( object.bytes == 0u )
		return(0u);
	if ( object.host != 0 )
	{
		BoundaryCheck(queue,kind,&object,(uint8_t *)object.host,seed,device,pageable);
		checked = 1u;
	}
	else if ( needs_host != 0u )
	{
		snprintf(what,sizeof(what),"9 GiB of %s has a host view",kind);
		CHECK(0,what);
	}
	CHECK(SparkDeviceFree(&object) == SPARK_STATUS_OK,"free the boundary object");
	return(checked);
}

static uint32_t BoundaryCreatedHost(SparkDeviceQueue *queue, uint32_t seed, const SparkDeviceBuffer *device, const SparkDeviceBuffer *pageable)
{
	SparkDeviceMemoryHandle memory = 0;
	SparkDeviceAddress address = 0u;
	SparkDeviceBuffer view;
	uint32_t created,reserved = 0u,mapped = 0u,adopted = 0u;
	memset(&view,0,sizeof(view));
	created = SparkDeviceMemoryCreateHost(&memory,BOUNDARY_BYTES,"boundary created host") == SPARK_STATUS_OK;
	reserved = created != 0u && SparkDeviceAddressReserve(&address,BOUNDARY_BYTES,"boundary created host") == SPARK_STATUS_OK;
	mapped = reserved != 0u && SparkDeviceMemoryMap(address,BOUNDARY_BYTES,memory,0u,
		SPARK_DEVICE_ACCESS_HOST_READ | SPARK_DEVICE_ACCESS_HOST_WRITE | SPARK_DEVICE_ACCESS_DEVICE_READ) == SPARK_STATUS_OK;
	adopted = mapped != 0u && SparkDeviceAdopt(&view,address,0,BOUNDARY_BYTES,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,"boundary created host") == SPARK_STATUS_OK;
	CHECK(adopted != 0u,"create 9 GiB of host memory, map it with host access and adopt the mapping");
	if ( adopted != 0u )
	{
		BoundaryCheck(queue,"created host memory",&view,(uint8_t *)(uintptr_t)address,seed,device,pageable);
		CHECK(SparkDeviceFree(&view) == SPARK_STATUS_OK,"release the adopted created host memory");
	}
	if ( mapped != 0u )
		CHECK(SparkDeviceMemoryUnmap(address,BOUNDARY_BYTES) == SPARK_STATUS_OK,"unmap the created host memory");
	if ( reserved != 0u )
		CHECK(SparkDeviceAddressRelease(address,BOUNDARY_BYTES) == SPARK_STATUS_OK,"release the created host memory's addresses");
	if ( created != 0u )
		CHECK(SparkDeviceMemoryRelease(memory) == SPARK_STATUS_OK,"release the created host memory");
	return(adopted);
}

static int TestBoundaries(void)
{
	SparkDeviceQueue *queue = 0;
	SparkDeviceBuffer device,pageable;
	const SparkDeviceProperties *properties = 0;
	const uint64_t needed = BOUNDARY_BYTES + LARGE_MARGIN_BYTES,available = Available();
	const int before = failures;
	uint32_t device_private = 0u;
	memset(&device,0,sizeof(device));
	memset(&pageable,0,sizeof(pageable));
	if ( available < needed )
	{
		printf("FAIL the boundary windows need %llu GiB of available memory; the host has %llu GiB\n",(unsigned long long)(needed >> 30),
			(unsigned long long)(available >> 30));
		return(1);
	}
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || (properties = SparkDeviceGetProperties()) == 0 ||
		SparkDeviceQueueCreate(&queue,"boundary windows") != SPARK_STATUS_OK )
	{
		printf("FAIL set up the device: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,BOUNDARY_SLOTS_BYTES,"boundary slots") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&pageable,SPARK_DEVICE_SPACE_HOST_PAGEABLE,BOUNDARY_SLOTS_BYTES,"boundary pageable") == SPARK_STATUS_OK,
		"allocate the device memory and the pageable memory the windows go through");
	if ( device.bytes != 0u && pageable.bytes != 0u )
	{
		(void)BoundaryAllocated(queue,SPARK_DEVICE_SPACE_HOST_PINNED,"host-pinned memory",0x51u,&device,&pageable,1u);
		(void)BoundaryAllocated(queue,SPARK_DEVICE_SPACE_SHARED_COHERENT,"shared-coherent memory",0x52u,&device,&pageable,1u);
		(void)BoundaryCreatedHost(queue,0x53u,&device,&pageable);
		if ( (properties->properties & SPARK_DEVICE_PROPERTY_UNIFIED_MEMORY) != 0u )
			device_private = BoundaryAllocated(queue,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,"device-private memory",0x54u,&device,&pageable,0u);
	}
	if ( pageable.bytes != 0u )
		CHECK(SparkDeviceFree(&pageable) == SPARK_STATUS_OK,"free the pageable memory");
	if ( device.bytes != 0u )
		CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK,"free the device memory");
	CHECK(SparkDeviceQueueDestroy(queue) == SPARK_STATUS_OK && SparkDeviceShutdown() == SPARK_STATUS_OK,"close the device");
	if ( failures != before )
	{
		printf("\nFAIL boundary windows (%d)\n",failures - before);
		return(1);
	}
	printf("PASS the GPU reads 9 GiB of host-pinned, shared-coherent and created host memory%s as the CPU wrote it at [0, 64 KiB), 4 GiB and 8 GiB "
		"+- 64 KiB and the last 64 KiB, through device memory into pageable memory\n",device_private != 0u ? ", and of device-private memory," : "");
	return(0);
}

static int TestSizes(void)
{
	static const uint64_t sizes[] = {8ull << 30, (21ull << 30) / 2u, 16ull << 30};
	SparkDeviceQueue *queue = 0;
	uint32_t index;
	const uint64_t needed = 2u * sizes[2] + LARGE_MARGIN_BYTES,available = Available();
	const int before = failures;
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
	if ( failures != before )
	{
		printf("\nFAIL (%d)\n",failures - before);
		return(1);
	}
	printf("PASS fills and copies of 8, 10.5 and 16 GiB of device-private, shared-coherent, host-pinned and shared memory go through in one call each, in lines of at most 2^31 bytes, and reach every byte\n");
	return(0);
}

int main(int argc, char **argv)
{
	if ( argc == 1 )
	{
		const int boundaries = TestBoundaries();
		return(TestSizes() != 0 || boundaries != 0 ? 1 : 0);
	}
	if ( argc == 2 && strcmp(argv[1],"sizes") == 0 )
		return(TestSizes());
	if ( argc == 2 && strcmp(argv[1],"boundaries") == 0 )
		return(TestBoundaries());
	printf("usage: %s [sizes|boundaries]\n",argv[0]);
	return(2);
}
