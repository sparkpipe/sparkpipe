#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"

#define MEMORY_ODD_BYTES 70001u
#define MEMORY_FRESH_BYTES (1ull << 30)
#define MEMORY_TAIL_BYTES 4096u
#define MEMORY_LARGE_BYTES ((1ull << 31) + 8192u + 5u)
#define MEMORY_FILE_BYTES ((1u << 20) + 123u)
#define MEMORY_FILE_OFFSET (4096u + 17u)
#define MEMORY_QUEUE_CYCLES 64u
#define MEMORY_PAGEABLE_BYTES ((20ull << 20) + 13u)

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

static void Pattern(uint8_t *bytes, uint64_t count, uint32_t seed)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
		bytes[index] = (uint8_t)((index * 131u + seed * 7u + (index >> 8)) & 0xffu);
}

static void TestSpaces(void)
{
	const SparkDeviceProperties *properties = SparkDeviceGetProperties();
	SparkDeviceBuffer device,shared,pinned,pageable,again;
	CHECK(strcmp(properties->backend,"nvrm") == 0 && properties->compute_units != 0u && properties->max_buffer_bytes != 0u,
		"the nvrm backend reports itself, its processors and its memory");
	printf("device: %s, %s, %u processors, %llu bytes\n",properties->name,properties->target,properties->compute_units,
		(unsigned long long)properties->max_buffer_bytes);
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,1000u,"spaces") == SPARK_STATUS_OK && device.host == 0 && device.device != 0u,
		"device-private memory has a device address and no host view");
	CHECK(SparkDeviceAllocate(&shared,SPARK_DEVICE_SPACE_SHARED_COHERENT,1000u,"spaces") == SPARK_STATUS_OK &&
		(uint64_t)(uintptr_t)shared.host == shared.device,"shared memory has one address for host and device");
	CHECK(SparkDeviceAllocate(&pinned,SPARK_DEVICE_SPACE_HOST_PINNED,1000u,"spaces") == SPARK_STATUS_OK &&
		(uint64_t)(uintptr_t)pinned.host == pinned.device,"pinned memory has one address for host and device");
	CHECK(SparkDeviceAllocate(&pageable,SPARK_DEVICE_SPACE_HOST_PAGEABLE,1000u,"spaces") == SPARK_STATUS_OK && pageable.host != 0 && pageable.device == 0u,
		"pageable memory is host only");
	{
		const SparkDeviceAddress freed = device.device;
		CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK,"free device memory");
		CHECK(SparkDeviceAllocate(&again,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,1000u,"spaces") == SPARK_STATUS_OK && again.device == freed,
			"a freed device address is handed out again");
		CHECK(SparkDeviceFree(&again) == SPARK_STATUS_OK,"free it again");
	}
	CHECK(SparkDeviceFree(&shared) == SPARK_STATUS_OK && SparkDeviceFree(&pinned) == SPARK_STATUS_OK && SparkDeviceFree(&pageable) == SPARK_STATUS_OK,
		"free every space");
}

static void TestFresh(SparkDeviceQueue *queue)
{
	SparkDeviceBuffer device,back;
	uint8_t zeros[MEMORY_TAIL_BYTES];
	memset(zeros,0,sizeof(zeros));
	CHECK(SparkDeviceAllocate(&back,SPARK_DEVICE_SPACE_HOST_PINNED,MEMORY_TAIL_BYTES,"fresh") == SPARK_STATUS_OK,"allocate the readback");
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,MEMORY_FRESH_BYTES,"fresh") == SPARK_STATUS_OK,"allocate 1 GiB of device memory");
	memset(back.host,0xee,MEMORY_TAIL_BYTES);
	CHECK(SparkDeviceCopy(queue,At(&back,0u),At(&device,MEMORY_FRESH_BYTES - MEMORY_TAIL_BYTES),MEMORY_TAIL_BYTES) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK && memcmp(back.host,zeros,MEMORY_TAIL_BYTES) == 0,
		"new device memory reads as zeros to its last byte");
	CHECK(SparkDeviceFree(&device) == SPARK_STATUS_OK && SparkDeviceFree(&back) == SPARK_STATUS_OK,"free the fresh buffers");
}

static void TestCopies(SparkDeviceQueue *queue)
{
	SparkDeviceBuffer pinned,device,shared,back,registered;
	uint8_t *expected = (uint8_t *)malloc(MEMORY_ODD_BYTES);
	uint8_t *host = 0;
	CHECK(expected != 0 && posix_memalign((void **)&host,4096u,MEMORY_ODD_BYTES + 4096u) == 0,"host memory for the copies");
	if ( expected == 0 || host == 0 )
		return;
	CHECK(SparkDeviceAllocate(&pinned,SPARK_DEVICE_SPACE_HOST_PINNED,MEMORY_ODD_BYTES,"copies") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,MEMORY_ODD_BYTES,"copies") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&shared,SPARK_DEVICE_SPACE_SHARED_COHERENT,MEMORY_ODD_BYTES + 7u,"copies") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&back,SPARK_DEVICE_SPACE_HOST_PINNED,MEMORY_ODD_BYTES,"copies") == SPARK_STATUS_OK,"allocate the copy buffers");
	CHECK(SparkDeviceRegister(&registered,host + 13u,MEMORY_ODD_BYTES,SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE,"copies") ==
		SPARK_STATUS_OK && registered.device != 0u,"register host memory that starts off a page");
	Pattern((uint8_t *)pinned.host,MEMORY_ODD_BYTES,1u);
	memcpy(expected,pinned.host,MEMORY_ODD_BYTES);
	CHECK(SparkDeviceCopy(queue,At(&device,0u),At(&pinned,0u),MEMORY_ODD_BYTES) == SPARK_STATUS_OK &&
		SparkDeviceCopy(queue,At(&shared,7u),At(&device,0u),MEMORY_ODD_BYTES) == SPARK_STATUS_OK &&
		SparkDeviceCopy(queue,At(&registered,0u),At(&shared,7u),MEMORY_ODD_BYTES) == SPARK_STATUS_OK &&
		SparkDeviceCopy(queue,At(&back,0u),At(&registered,0u),MEMORY_ODD_BYTES) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"copy pinned, device, shared at an offset, registered and pinned again");
	CHECK(memcmp(back.host,expected,MEMORY_ODD_BYTES) == 0 && memcmp(host + 13u,expected,MEMORY_ODD_BYTES) == 0,
		"70001 bytes survive every space, and the registered memory holds them for the host too");
	CHECK(SparkDeviceFill(queue,At(&device,0u),0x11u,MEMORY_ODD_BYTES) == SPARK_STATUS_OK &&
		SparkDeviceFill(queue,At(&device,3u),0x5au,1001u) == SPARK_STATUS_OK &&
		SparkDeviceCopy(queue,At(&back,0u),At(&device,0u),MEMORY_ODD_BYTES) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"fill device memory, then an unaligned 1001 bytes inside it");
	{
		const uint8_t *bytes = (const uint8_t *)back.host;
		uint32_t index,wrong = 0u;
		for (index = 0u; index < MEMORY_ODD_BYTES; index++)
			wrong += bytes[index] != (index >= 3u && index < 1004u ? 0x5au : 0x11u);
		CHECK(wrong == 0u,"each fill reaches exactly its bytes, the unaligned head and tail included");
	}
	CHECK(SparkDeviceCopy(queue,At(&device,0u),At(&device,100u),100u) == SPARK_STATUS_OK && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) ==
		SPARK_STATUS_OK,"disjoint ranges of one buffer copy");
	CHECK(SparkDeviceFree(&registered) == SPARK_STATUS_OK && SparkDeviceFree(&pinned) == SPARK_STATUS_OK && SparkDeviceFree(&device) == SPARK_STATUS_OK &&
		SparkDeviceFree(&shared) == SPARK_STATUS_OK && SparkDeviceFree(&back) == SPARK_STATUS_OK,"free the copy buffers");
	free(host);
	free(expected);
}

static void TestLarge(SparkDeviceQueue *queue)
{
	SparkDeviceBuffer source,destination;
	uint64_t index;
	CHECK(SparkDeviceAllocate(&source,SPARK_DEVICE_SPACE_SHARED_COHERENT,MEMORY_LARGE_BYTES,"large") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&destination,SPARK_DEVICE_SPACE_SHARED_COHERENT,MEMORY_LARGE_BYTES,"large") == SPARK_STATUS_OK,"allocate two 2 GiB+ buffers");
	if ( source.host == 0 || destination.host == 0 )
		return;
	for (index = 0u; index < MEMORY_LARGE_BYTES; index += 4096u)
		((uint8_t *)source.host)[index] = (uint8_t)(index >> 12);
	((uint8_t *)source.host)[MEMORY_LARGE_BYTES - 1u] = 0xa5u;
	CHECK(SparkDeviceCopy(queue,At(&destination,0u),At(&source,0u),MEMORY_LARGE_BYTES) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"copy more than 2^31 bytes at once");
	CHECK(memcmp(destination.host,source.host,(size_t)MEMORY_LARGE_BYTES) == 0,"a copy split into 2^31-byte lines arrives whole");
	CHECK(SparkDeviceFill(queue,At(&destination,1u),0x3cu,MEMORY_LARGE_BYTES - 2u) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"fill more than 2^31 bytes at once");
	{
		const uint8_t *bytes = (const uint8_t *)destination.host;
		uint64_t wrong = 0u;
		for (index = 1u; index < MEMORY_LARGE_BYTES - 1u; index++)
			wrong += bytes[index] != 0x3cu;
		CHECK(wrong == 0u && bytes[0] == ((const uint8_t *)source.host)[0] && bytes[MEMORY_LARGE_BYTES - 1u] == 0xa5u,
			"the large fill reaches every byte it names and no other");
	}
	CHECK(SparkDeviceFree(&source) == SPARK_STATUS_OK && SparkDeviceFree(&destination) == SPARK_STATUS_OK,"free the large buffers");
}

static void TestFile(SparkDeviceQueue *queue)
{
	char path[] = "/tmp/test_nvrm_memory.XXXXXX";
	const int fd = mkstemp(path);
	uint8_t *contents = (uint8_t *)malloc(MEMORY_FILE_BYTES);
	SparkDeviceBuffer mapped,device,back;
	const uint64_t bytes = MEMORY_FILE_BYTES - MEMORY_FILE_OFFSET;
	CHECK(fd >= 0 && contents != 0,"a scratch file");
	if ( fd < 0 || contents == 0 )
		return;
	unlink(path);
	Pattern(contents,MEMORY_FILE_BYTES,9u);
	CHECK(write(fd,contents,MEMORY_FILE_BYTES) == (ssize_t)MEMORY_FILE_BYTES,"write the file");
	CHECK(SparkDeviceMapFile(&mapped,fd,MEMORY_FILE_OFFSET,bytes,"file") == SPARK_STATUS_OK && mapped.device != 0u &&
		memcmp(mapped.host,contents + MEMORY_FILE_OFFSET,(size_t)bytes) == 0,"map a file range that starts off a page");
	CHECK(SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,"file") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&back,SPARK_DEVICE_SPACE_HOST_PINNED,bytes,"file") == SPARK_STATUS_OK,"allocate the file copies");
	CHECK(SparkDeviceCopy(queue,At(&device,0u),At(&mapped,0u),bytes) == SPARK_STATUS_OK &&
		SparkDeviceCopy(queue,At(&back,0u),At(&device,0u),bytes) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"the device reads the mapped file");
	CHECK(memcmp(back.host,contents + MEMORY_FILE_OFFSET,(size_t)bytes) == 0,"the device sees the file's bytes from the mapped offset");
	CHECK(SparkDeviceFill(queue,At(&mapped,0u),0u,16u) == SPARK_STATUS_INVALID_ARGUMENT,"a mapped file is never a destination");
	CHECK(SparkDeviceFree(&mapped) == SPARK_STATUS_OK && SparkDeviceFree(&device) == SPARK_STATUS_OK && SparkDeviceFree(&back) == SPARK_STATUS_OK,
		"free the file buffers");
	close(fd);
	free(contents);
}

static void TestPageable(SparkDeviceQueue *queue)
{
	SparkDeviceBuffer source,device,back,pinned;
	uint64_t index,wrong = 0u;
	CHECK(SparkDeviceAllocate(&source,SPARK_DEVICE_SPACE_HOST_PAGEABLE,MEMORY_PAGEABLE_BYTES,"pageable") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&device,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,MEMORY_PAGEABLE_BYTES,"pageable") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&back,SPARK_DEVICE_SPACE_HOST_PAGEABLE,MEMORY_PAGEABLE_BYTES,"pageable") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&pinned,SPARK_DEVICE_SPACE_HOST_PINNED,MEMORY_PAGEABLE_BYTES,"pageable") == SPARK_STATUS_OK,"allocate the pageable buffers");
	if ( source.host == 0 || back.host == 0 || pinned.host == 0 )
		return;
	Pattern((uint8_t *)source.host,MEMORY_PAGEABLE_BYTES,5u);
	CHECK(SparkDeviceCopy(queue,At(&device,3u),At(&source,0u),MEMORY_PAGEABLE_BYTES - 3u) == SPARK_STATUS_OK &&
		SparkDeviceCopy(queue,At(&back,0u),At(&device,3u),MEMORY_PAGEABLE_BYTES - 3u) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"pageable to device and back, in several bounce-buffer chunks");
	CHECK(memcmp(back.host,source.host,(size_t)(MEMORY_PAGEABLE_BYTES - 3u)) == 0,"the bytes survive the staged copies");
	CHECK(SparkDeviceCopy(queue,At(&pinned,0u),At(&source,0u),MEMORY_PAGEABLE_BYTES) == SPARK_STATUS_OK &&
		SparkDeviceFill(queue,At(&back,1u),0x77u,1000u) == SPARK_STATUS_OK &&
		SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"pageable to pinned memory, and a pageable fill, in queue order");
	CHECK(memcmp(pinned.host,source.host,(size_t)MEMORY_PAGEABLE_BYTES) == 0,"a pageable to pinned copy runs on the host in queue order");
	for (index = 0u; index < 1002u; index++)
		wrong += ((uint8_t *)back.host)[index] != (index >= 1u && index < 1001u ? 0x77u : ((uint8_t *)source.host)[index]);
	CHECK(wrong == 0u,"a pageable fill reaches exactly its bytes");
	CHECK(SparkDeviceFree(&source) == SPARK_STATUS_OK && SparkDeviceFree(&device) == SPARK_STATUS_OK && SparkDeviceFree(&back) == SPARK_STATUS_OK &&
		SparkDeviceFree(&pinned) == SPARK_STATUS_OK,"free the pageable buffers");
}

static void TestQueues(void)
{
	SparkDeviceQueue *first = 0,*second = 0;
	SparkDeviceBuffer source,left,right,pageable;
	uint32_t cycle,created = 0u;
	for (cycle = 0u; cycle < MEMORY_QUEUE_CYCLES; cycle++)
	{
		SparkDeviceQueue *queue = 0;
		created += SparkDeviceQueueCreate(&queue,"cycle") == SPARK_STATUS_OK && SparkDeviceQueueDestroy(queue) == SPARK_STATUS_OK;
	}
	CHECK(created == MEMORY_QUEUE_CYCLES,"64 queues open and close in turn");
	CHECK(SparkDeviceQueueCreate(&first,"first") == SPARK_STATUS_OK && SparkDeviceQueueCreate(&second,"second") == SPARK_STATUS_OK,"two queues at once");
	CHECK(SparkDeviceAllocate(&source,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,MEMORY_FRESH_BYTES,"queues") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&left,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,MEMORY_FRESH_BYTES,"queues") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&right,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,MEMORY_FRESH_BYTES,"queues") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&pageable,SPARK_DEVICE_SPACE_HOST_PAGEABLE,4096u,"queues") == SPARK_STATUS_OK,"buffers for both queues");
	CHECK(SparkDeviceCopy(first,At(&left,0u),At(&source,0u),MEMORY_FRESH_BYTES) == SPARK_STATUS_OK &&
		SparkDeviceCopy(second,At(&right,0u),At(&source,0u),MEMORY_FRESH_BYTES) == SPARK_STATUS_OK,"both queues copy at once");
	{
		const SparkStatus early = SparkDeviceQueueQuery(first);
		CHECK(early == SPARK_STATUS_OK || early == SPARK_STATUS_PENDING,"a query answers without waiting");
	}
	CHECK(SparkDeviceQueueWait(first,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK && SparkDeviceQueueWait(second,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK &&
		SparkDeviceQueueQuery(first) == SPARK_STATUS_OK,"both queues finish, and a finished queue answers at once");
	CHECK(SparkDeviceQueueCapture(first,1u) == SPARK_STATUS_UNSUPPORTED,"a queue the backend owns is never captured");
	CHECK(SparkDeviceFree(&source) == SPARK_STATUS_OK && SparkDeviceFree(&left) == SPARK_STATUS_OK && SparkDeviceFree(&right) == SPARK_STATUS_OK &&
		SparkDeviceFree(&pageable) == SPARK_STATUS_OK,"free the queue buffers");
	CHECK(SparkDeviceQueueDestroy(first) == SPARK_STATUS_OK && SparkDeviceQueueDestroy(second) == SPARK_STATUS_OK,"destroy both queues");
}

int main(void)
{
	SparkDeviceConfiguration configuration;
	SparkDeviceQueue *queue = 0;
	if ( access("/dev/nvidiactl",F_OK) != 0 )
	{
		printf("SKIP test_nvrm_memory (no open kernel module on this host)\n");
		return(0);
	}
	memset(&configuration,0,sizeof(configuration));
	if ( SparkDeviceInitialize(&configuration) != SPARK_STATUS_OK )
	{
		printf("FAIL initialize: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	TestSpaces();
	CHECK(SparkDeviceQueueCreate(&queue,"memory") == SPARK_STATUS_OK,"create a queue");
	if ( queue != 0 )
	{
		TestFresh(queue);
		TestCopies(queue);
		TestLarge(queue);
		TestFile(queue);
		TestPageable(queue);
		CHECK(SparkDeviceQueueDestroy(queue) == SPARK_STATUS_OK,"destroy the queue");
	}
	TestQueues();
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_OK,"shut the device down");
	CHECK(SparkDeviceInitialize(&configuration) == SPARK_STATUS_OK && SparkDeviceShutdown() == SPARK_STATUS_OK,"the device opens again after a shutdown");
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS nvrm memory spaces, registered, file-backed and pageable memory, copy-engine copies and fills, and queues\n");
	return(0);
}
