#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_launch.h"

#ifndef SPARK_NVRM_FIXTURE_CUBIN
#define SPARK_NVRM_FIXTURE_CUBIN "tests/fixtures/nvrm/fixture_kernels.sm_121a.cubin"
#endif

#define ORDER_BYTES (64ull << 20)
#define ORDER_ROUNDS 32u
#define ORDER_SETS 32u
#define ORDER_SET_BYTES (8ull << 20)
#define ORDER_SLOTS 64u
#define ORDER_THREADS 256u
#define ORDER_BLOCKS_PER_SM 4u
#define ORDER_ELEMENT_BYTES 16u
#define ORDER_TIMEOUT_NS 30000000000ull
#define ORDER_FENCE_START 0xffffffe0ull
#define ORDER_STALE 0xeeu

typedef struct CopyArguments { uint64_t source; uint64_t destination; uint64_t count; } CopyArguments;
typedef struct FillArguments { uint64_t destination; uint64_t count; uint32_t value; } FillArguments;
#define FILL_ARGUMENT_BYTES ((uint32_t)(offsetof(FillArguments,value) + sizeof(uint32_t)))

typedef struct Order
{
	SparkNvrmGpu gpu;
	SparkNvrmChannel queue;
	SparkNvrmProgram program;
	const SparkNvrmCubinKernel *copy;
	const SparkNvrmCubinKernel *fill;
	SparkNvrmMemory slots;
	uint32_t slot_bytes;
	uint32_t slot_next;
	uint32_t blocks;
	SparkNvrmMemory first;
	SparkNvrmMemory second;
	SparkNvrmMemory third;
	SparkNvrmMemory staging;
	SparkNvrmMemory sets[3u * ORDER_SETS];
} Order;

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s\n",what); \
			failures++; \
		} \
	} while (0)

static int Failed(const char *step)
{
	const SparkDeviceError *error = SparkDeviceLastError();
	printf("FAIL %s: %s [%s, %s, code %d]\n",step,error->text,error->operation,error->owner,error->backend_code);
	failures++;
	return(1);
}

static uint8_t *ReadFile(const char *path, uint64_t *bytes)
{
	FILE *file = fopen(path,"rb");
	uint8_t *data;
	long length;
	if ( file == 0 )
		return(0);
	fseek(file,0,SEEK_END);
	length = ftell(file);
	fseek(file,0,SEEK_SET);
	data = (uint8_t *)malloc((size_t)length);
	if ( data != 0 && fread(data,1u,(size_t)length,file) != (size_t)length )
	{
		free(data);
		data = 0;
	}
	fclose(file);
	*bytes = (uint64_t)length;
	return(data);
}

static int Launch(Order *order, const SparkNvrmCubinKernel *kernel, const void *arguments, uint32_t bytes, uint64_t *value)
{
	SparkNvrmLaunchRequest request;
	const uint32_t slot = order->slot_next++ % ORDER_SLOTS;
	memset(&request,0,sizeof(request));
	request.kernel = kernel;
	request.grid[0] = order->blocks;
	request.grid[1] = request.grid[2] = 1u;
	request.block[0] = ORDER_THREADS;
	request.block[1] = request.block[2] = 1u;
	request.arguments = arguments;
	request.argument_bytes = bytes;
	request.slot = order->slots.cpu + (uint64_t)slot * order->slot_bytes;
	request.slot_address = order->slots.address + (uint64_t)slot * order->slot_bytes;
	request.slot_bytes = order->slot_bytes;
	if ( SparkNvrmLaunch(&order->gpu,&order->queue,&order->program,&request,value,"order") != SPARK_STATUS_OK )
		return(Failed("queue a launch"));
	return(0);
}

static int KernelCopy(Order *order, const SparkNvrmMemory *source, const SparkNvrmMemory *destination, uint64_t bytes, uint64_t *value)
{
	CopyArguments arguments;
	arguments.source = source->address;
	arguments.destination = destination->address;
	arguments.count = bytes / ORDER_ELEMENT_BYTES;
	return(Launch(order,order->copy,&arguments,(uint32_t)sizeof(arguments),value));
}

static int KernelFill(Order *order, const SparkNvrmMemory *destination, uint64_t bytes, uint32_t pattern, uint64_t *value)
{
	FillArguments arguments;
	memset(&arguments,0,sizeof(arguments));
	arguments.destination = destination->address;
	arguments.count = bytes / ORDER_ELEMENT_BYTES;
	arguments.value = pattern;
	return(Launch(order,order->fill,&arguments,FILL_ARGUMENT_BYTES,value));
}

static int Wait(Order *order, uint64_t value)
{
	if ( SparkNvrmWaitValue((const volatile uint64_t *)(void *)order->queue.fence.cpu,value,ORDER_TIMEOUT_NS,"queue_wait","order") != SPARK_STATUS_OK )
		return(Failed("wait for the queue"));
	return(0);
}

static int Stale(Order *order, const SparkNvrmMemory *memory, uint64_t bytes)
{
	uint64_t value = 0u;
	if ( SparkNvrmFill(&order->gpu,&order->queue,memory->address,ORDER_STALE,bytes,&value,"order") != SPARK_STATUS_OK )
		return(Failed("mark a buffer stale"));
	return(0);
}

static int Readback(Order *order, const SparkNvrmMemory *memory, uint64_t bytes)
{
	uint64_t value = 0u;
	if ( SparkNvrmCopy(&order->gpu,&order->queue,order->staging.address,memory->address,bytes,&value,"order") != SPARK_STATUS_OK )
		return(Failed("copy a result to host memory"));
	return(Wait(order,value));
}

static uint64_t WrongBytes(const SparkNvrmMemory *memory, uint64_t bytes, uint8_t expected)
{
	const volatile uint8_t *at = memory->cpu;
	uint64_t index,wrong = 0u;
	for (index = 0u; index < bytes; index++)
		wrong += at[index] != expected;
	return(wrong);
}

static uint64_t WrongPattern(const SparkNvrmMemory *memory, uint64_t bytes, uint32_t pattern)
{
	const volatile uint32_t *at = (const volatile uint32_t *)(const void *)memory->cpu;
	uint64_t index,wrong = 0u;
	for (index = 0u; index < bytes / ORDER_ELEMENT_BYTES; index++)
		wrong += at[4u * index] != pattern || at[4u * index + 1u] != (pattern ^ 1u) || at[4u * index + 2u] != (pattern ^ 2u) ||
			at[4u * index + 3u] != (pattern ^ (uint32_t)index);
	return(wrong);
}

static void TestCopyThenKernel(Order *order)
{
	uint32_t round,late = 0u;
	uint64_t worst = 0u;
	for (round = 0u; round < ORDER_ROUNDS; round++)
	{
		const uint8_t byte = (uint8_t)(0x11u + 37u * round);
		uint64_t value = 0u,wrong;
		if ( Stale(order,&order->second,ORDER_BYTES) != 0 )
			return;
		if ( SparkNvrmFill(&order->gpu,&order->queue,order->first.address,byte,ORDER_BYTES,&value,"order") != SPARK_STATUS_OK )
		{
			Failed("fill on the copy engine");
			return;
		}
		if ( KernelCopy(order,&order->first,&order->second,ORDER_BYTES,&value) != 0 || Readback(order,&order->second,ORDER_BYTES) != 0 )
			return;
		wrong = WrongBytes(&order->staging,ORDER_BYTES,byte);
		late += wrong != 0u;
		worst = wrong > worst ? wrong : worst;
	}
	printf("copy then kernel: %u of %u rounds read stale bytes (worst %llu of %llu)\n",late,ORDER_ROUNDS,(unsigned long long)worst,
		(unsigned long long)ORDER_BYTES);
	CHECK(late == 0u,"a kernel queued after a copy-engine fill reads every byte the fill wrote");
}

static void TestKernelThenCopy(Order *order)
{
	uint32_t round,late = 0u;
	uint64_t worst = 0u;
	for (round = 0u; round < ORDER_ROUNDS; round++)
	{
		const uint32_t pattern = 0x51000000u + 0x1010u * round;
		uint64_t value = 0u,wrong;
		if ( Stale(order,&order->third,ORDER_BYTES) != 0 || KernelFill(order,&order->second,ORDER_BYTES,pattern,&value) != 0 )
			return;
		if ( SparkNvrmCopy(&order->gpu,&order->queue,order->third.address,order->second.address,ORDER_BYTES,&value,"order") != SPARK_STATUS_OK )
		{
			Failed("copy on the copy engine");
			return;
		}
		if ( Readback(order,&order->third,ORDER_BYTES) != 0 )
			return;
		wrong = WrongPattern(&order->staging,ORDER_BYTES,pattern);
		late += wrong != 0u;
		worst = wrong > worst ? wrong : worst;
	}
	printf("kernel then copy: %u of %u rounds copied stale elements (worst %llu of %llu)\n",late,ORDER_ROUNDS,(unsigned long long)worst,
		(unsigned long long)(ORDER_BYTES / ORDER_ELEMENT_BYTES));
	CHECK(late == 0u,"a copy-engine copy queued after a kernel copies everything the kernel wrote");
}

static void TestAlternation(Order *order)
{
	uint64_t value = 0u,wrong_kernel = 0u,wrong_copy = 0u;
	uint32_t set,late = 0u;
	for (set = 0u; set < 3u * ORDER_SETS; set++)
		if ( Stale(order,&order->sets[set],ORDER_SET_BYTES) != 0 )
			return;
	for (set = 0u; set < ORDER_SETS; set++)
	{
		SparkNvrmMemory *first = &order->sets[3u * set],*second = &order->sets[3u * set + 1u],*third = &order->sets[3u * set + 2u];
		const uint8_t byte = (uint8_t)(0x21u + 11u * set);
		if ( SparkNvrmFill(&order->gpu,&order->queue,first->address,byte,ORDER_SET_BYTES,&value,"order") != SPARK_STATUS_OK )
		{
			Failed("fill on the copy engine");
			return;
		}
		if ( KernelCopy(order,first,second,ORDER_SET_BYTES,&value) != 0 )
			return;
		if ( SparkNvrmCopy(&order->gpu,&order->queue,third->address,second->address,ORDER_SET_BYTES,&value,"order") != SPARK_STATUS_OK )
		{
			Failed("copy on the copy engine");
			return;
		}
	}
	if ( Wait(order,value) != 0 )
		return;
	for (set = 0u; set < ORDER_SETS; set++)
	{
		const uint8_t byte = (uint8_t)(0x21u + 11u * set);
		uint64_t kernel,copy;
		if ( Readback(order,&order->sets[3u * set + 1u],ORDER_SET_BYTES) != 0 )
			return;
		kernel = WrongBytes(&order->staging,ORDER_SET_BYTES,byte);
		if ( Readback(order,&order->sets[3u * set + 2u],ORDER_SET_BYTES) != 0 )
			return;
		copy = WrongBytes(&order->staging,ORDER_SET_BYTES,byte);
		wrong_kernel += kernel;
		wrong_copy += copy;
		late += kernel != 0u || copy != 0u;
	}
	printf("alternation: %u of %u sets wrong (%llu stale bytes after the kernel, %llu after the copy), queue at %llu\n",late,ORDER_SETS,
		(unsigned long long)wrong_kernel,(unsigned long long)wrong_copy,(unsigned long long)value);
	CHECK(late == 0u,"back-to-back fill, kernel, copy, fill, ... with no host wait keeps every dependency in both directions");
	CHECK(value > 0xffffffffull,"the queue's 64-bit fence crossed 2^32 and every engine kept releasing it in order");
}

static int Prepare(Order *order)
{
	uint64_t bytes = 0u;
	uint8_t *blob = ReadFile(SPARK_NVRM_FIXTURE_CUBIN,&bytes);
	uint32_t set;
	if ( blob == 0 )
	{
		printf("FAIL cannot read %s\n",SPARK_NVRM_FIXTURE_CUBIN);
		failures++;
		return(1);
	}
	if ( SparkNvrmGpuOpen(&order->gpu,0u,SPARK_NVRM_MEMORY_HOST,"order") != SPARK_STATUS_OK )
		return(Failed("open the GPU"));
	if ( SparkNvrmProgramLoad(&order->gpu,blob,bytes,&order->program,"order") != SPARK_STATUS_OK )
		return(Failed("load the fixture"));
	free(blob);
	order->copy = SparkNvrmCubinKernelNamed(&order->program.cubin,"SparkNvrmStreamCopy");
	order->fill = SparkNvrmCubinKernelNamed(&order->program.cubin,"SparkNvrmStreamFill");
	if ( order->copy == 0 || order->fill == 0 )
		return(Failed("find the stream kernels"));
	order->blocks = order->gpu.gpcs * order->gpu.tpcs_per_gpc * order->gpu.sms_per_tpc * ORDER_BLOCKS_PER_SM;
	order->slot_bytes = SparkNvrmLaunchSlotBytes(order->copy) > SparkNvrmLaunchSlotBytes(order->fill) ? SparkNvrmLaunchSlotBytes(order->copy) :
		SparkNvrmLaunchSlotBytes(order->fill);
	if ( SparkNvrmChannelCreate(&order->gpu,&order->queue,SPARK_NVRM_MEMORY_HOST,"order") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&order->gpu,(uint64_t)ORDER_SLOTS * order->slot_bytes,SPARK_NVRM_MEMORY_HOST,&order->slots,"order") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&order->gpu,ORDER_BYTES,SPARK_NVRM_MEMORY_SYSTEM,&order->first,"order") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&order->gpu,ORDER_BYTES,SPARK_NVRM_MEMORY_SYSTEM,&order->second,"order") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&order->gpu,ORDER_BYTES,SPARK_NVRM_MEMORY_SYSTEM,&order->third,"order") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&order->gpu,ORDER_BYTES,SPARK_NVRM_MEMORY_HOST,&order->staging,"order") != SPARK_STATUS_OK )
		return(Failed("create the queue and its buffers"));
	for (set = 0u; set < 3u * ORDER_SETS; set++)
		if ( SparkNvrmMemoryAllocate(&order->gpu,ORDER_SET_BYTES,SPARK_NVRM_MEMORY_SYSTEM,&order->sets[set],"order") != SPARK_STATUS_OK )
			return(Failed("allocate the alternation buffers"));
	__atomic_store_n((volatile uint64_t *)(void *)order->queue.fence.cpu,ORDER_FENCE_START,__ATOMIC_RELEASE);
	order->queue.fence_value = ORDER_FENCE_START;
	return(0);
}

static void Release(Order *order)
{
	uint32_t set;
	for (set = 0u; set < 3u * ORDER_SETS; set++)
		if ( order->sets[set].bytes != 0u )
			SparkNvrmMemoryRelease(&order->gpu,&order->sets[set]);
	if ( order->staging.bytes != 0u )
		SparkNvrmMemoryRelease(&order->gpu,&order->staging);
	if ( order->third.bytes != 0u )
		SparkNvrmMemoryRelease(&order->gpu,&order->third);
	if ( order->second.bytes != 0u )
		SparkNvrmMemoryRelease(&order->gpu,&order->second);
	if ( order->first.bytes != 0u )
		SparkNvrmMemoryRelease(&order->gpu,&order->first);
	if ( order->slots.bytes != 0u )
		SparkNvrmMemoryRelease(&order->gpu,&order->slots);
	SparkNvrmChannelDestroy(&order->gpu,&order->queue);
	SparkNvrmProgramRelease(&order->gpu,&order->program);
	SparkNvrmGpuClose(&order->gpu);
}

int main(void)
{
	static Order order;
	if ( access("/dev/nvidiactl",F_OK) != 0 )
	{
		printf("SKIP test_nvrm_queue_order (no open kernel module on this host)\n");
		return(0);
	}
	if ( Prepare(&order) == 0 )
	{
		TestCopyThenKernel(&order);
		TestKernelThenCopy(&order);
		TestAlternation(&order);
	}
	Release(&order);
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS one queue orders its copy engine and its compute engine in both directions, round after round\n");
	return(0);
}
