#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_launch.h"

#define BANDWIDTH_THREADS 256u
#define BANDWIDTH_BLOCKS_PER_SM 4u
#define BANDWIDTH_TIMEOUT_NS 30000000000ull
#define BANDWIDTH_SAMPLES 4096u
#define BANDWIDTH_ROUNDS_MAX 256u

typedef struct Bench
{
	SparkNvrmGpu gpu;
	SparkNvrmProgram program;
	const SparkNvrmCubinKernel *read;
	const SparkNvrmCubinKernel *copy;
	const SparkNvrmCubinKernel *fill;
	SparkNvrmMemory slots;
	SparkNvrmMemory semaphore;
	SparkNvrmMemory sink;
	uint32_t slot_bytes;
	uint64_t sequence;
	uint32_t blocks;
} Bench;

static int Failed(const char *step)
{
	const SparkDeviceError *error = SparkDeviceLastError();
	printf("FAIL %s: %s [%s, owner %s, code %d]\n",step,error->text,error->operation,error->owner,error->backend_code);
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

static int CompareTimes(const void *left, const void *right)
{
	const uint64_t a = *(const uint64_t *)left,b = *(const uint64_t *)right;
	return(a < b ? -1 : a > b);
}

static int Run(Bench *bench, const SparkNvrmCubinKernel *kernel, const void *arguments, uint32_t bytes, uint64_t *gpu_nanoseconds)
{
	SparkNvrmChannel *channel = &bench->gpu.compute;
	volatile uint64_t *semaphore = (volatile uint64_t *)(void *)bench->semaphore.cpu;
	SparkNvrmLaunchRequest request;
	const uint64_t first = channel->push_used,value = ++bench->sequence;
	memset(&request,0,sizeof(request));
	request.kernel = kernel;
	request.grid[0] = bench->blocks;
	request.grid[1] = request.grid[2] = 1u;
	request.block[0] = BANDWIDTH_THREADS;
	request.block[1] = request.block[2] = 1u;
	request.arguments = arguments;
	request.argument_bytes = bytes;
	request.slot = bench->slots.cpu;
	request.slot_address = bench->slots.address;
	request.slot_bytes = bench->slot_bytes;
	if ( SparkNvrmSemaphoreRelease(channel,bench->semaphore.address,value,1u,"bandwidth") != SPARK_STATUS_OK ||
		SparkNvrmLaunchQueue(&bench->gpu,&bench->program,&request,"bandwidth") != SPARK_STATUS_OK ||
		SparkNvrmSemaphoreRelease(channel,bench->semaphore.address + 16u,value,1u,"bandwidth") != SPARK_STATUS_OK ||
		SparkNvrmSubmit(&bench->gpu,channel,first,"bandwidth") != SPARK_STATUS_OK )
		return(Failed("submit a measured launch"));
	if ( SparkNvrmWaitValue(&semaphore[2],value,BANDWIDTH_TIMEOUT_NS,"kernel_launch","bandwidth") != SPARK_STATUS_OK )
		return(Failed("wait for a measured launch"));
	*gpu_nanoseconds = semaphore[3] - semaphore[1];
	if ( channel->push_used > channel->push.bytes / 2u )
		channel->push_used = 0u;
	return(0);
}

static int Measure(Bench *bench, const char *label, const SparkNvrmCubinKernel *kernel, const void *arguments, uint32_t bytes,
	uint64_t moved, uint32_t rounds)
{
	uint64_t times[BANDWIDTH_ROUNDS_MAX];
	uint32_t round;
	for (round = 0u; round < 3u; round++)
		if ( Run(bench,kernel,arguments,bytes,&times[0]) != 0 )
			return(1);
	for (round = 0u; round < rounds; round++)
		if ( Run(bench,kernel,arguments,bytes,&times[round]) != 0 )
			return(1);
	qsort(times,rounds,sizeof(times[0]),CompareTimes);
	printf("  %-5s median %8.2f GB/s  best %8.2f GB/s  (%u rounds, %.3f ms median)\n",label,(double)moved / (double)times[rounds / 2u],
		(double)moved / (double)times[0],rounds,(double)times[rounds / 2u] / 1e6);
	return(0);
}

static int MeasurePath(Bench *bench, SparkNvrmMemoryKind kind, const char *name, uint64_t bytes, uint32_t rounds)
{
	SparkNvrmMemory source,destination;
	uint8_t arguments[24];
	const uint64_t count = bytes / 16u;
	const uint32_t value = 0x13572468u;
	uint64_t nanoseconds = 0u,address;
	uint32_t index;
	int result;
	printf("path %s, %llu MiB per buffer\n",name,(unsigned long long)(bytes >> 20));
	if ( SparkNvrmMemoryAllocate(&bench->gpu,bytes,kind,&source,"source") != SPARK_STATUS_OK )
		return(Failed("allocate the source"));
	if ( SparkNvrmMemoryAllocate(&bench->gpu,bytes,kind,&destination,"destination") != SPARK_STATUS_OK )
	{
		SparkNvrmMemoryRelease(&bench->gpu,&source);
		return(Failed("allocate the destination"));
	}
	printf("  pages %u bytes, source at %#llx\n",source.page_bytes,(unsigned long long)source.address);
	memcpy(arguments,&source.address,8u);
	memcpy(arguments + 8u,&count,8u);
	memcpy(arguments + 16u,&value,4u);
	result = Run(bench,bench->fill,arguments,20u,&nanoseconds);
	if ( result == 0 )
	{
		address = bench->sink.address;
		memcpy(arguments + 16u,&address,8u);
		result = Measure(bench,"read",bench->read,arguments,24u,bytes,rounds);
	}
	if ( result == 0 )
	{
		memcpy(arguments + 8u,&destination.address,8u);
		memcpy(arguments + 16u,&count,8u);
		result = Measure(bench,"copy",bench->copy,arguments,24u,2u * bytes,rounds);
	}
	for (index = 0u; result == 0 && index < BANDWIDTH_SAMPLES; index++)
	{
		const uint64_t at = ((uint64_t)index * 2654435761u) % count;
		if ( memcmp(source.cpu + at * 16u,destination.cpu + at * 16u,16u) != 0 || ((const uint32_t *)(void *)source.cpu)[at * 4u] != value )
		{
			printf("FAIL the copy differs from its source at element %llu\n",(unsigned long long)at);
			result = 1;
		}
	}
	SparkNvrmMemoryRelease(&bench->gpu,&destination);
	SparkNvrmMemoryRelease(&bench->gpu,&source);
	return(result);
}

int main(int argc, char **argv)
{
	static Bench bench;
	const char *path = "tests/fixtures/nvrm/fixture_kernels.sm_121a.cubin",*only = "all";
	uint64_t bytes = 1024ull << 20,blob_bytes = 0u;
	uint32_t rounds = 20u;
	uint8_t *blob;
	int index,result = 0;
	for (index = 1; index < argc; index++)
	{
		if ( strcmp(argv[index],"--cubin") == 0 && index + 1 < argc )
			path = argv[++index];
		else if ( strcmp(argv[index],"--mib") == 0 && index + 1 < argc )
			bytes = strtoull(argv[++index],0,10) << 20;
		else if ( strcmp(argv[index],"--rounds") == 0 && index + 1 < argc )
			rounds = (uint32_t)strtoul(argv[++index],0,10);
		else if ( strcmp(argv[index],"--path") == 0 && index + 1 < argc )
			only = argv[++index];
		else
		{
			printf("usage: %s [--cubin PATH] [--mib N] [--rounds N] [--path host|system|uncached|all]\n",argv[0]);
			return(2);
		}
	}
	if ( rounds == 0u || rounds > BANDWIDTH_ROUNDS_MAX || bytes == 0u )
	{
		printf("FAIL rounds must be 1 to %u and the buffer at least 1 MiB\n",BANDWIDTH_ROUNDS_MAX);
		return(2);
	}
	blob = ReadFile(path,&blob_bytes);
	if ( blob == 0 )
	{
		printf("FAIL cannot read %s\n",path);
		return(1);
	}
	if ( SparkNvrmGpuOpen(&bench.gpu,0u,SPARK_NVRM_MEMORY_HOST,"bandwidth") != SPARK_STATUS_OK )
		return(Failed("open the GPU"));
	if ( SparkNvrmProgramLoad(&bench.gpu,blob,blob_bytes,&bench.program,"bandwidth") != SPARK_STATUS_OK )
		return(Failed("load the kernel image"));
	free(blob);
	bench.read = SparkNvrmCubinKernelNamed(&bench.program.cubin,"SparkNvrmStreamRead");
	bench.copy = SparkNvrmCubinKernelNamed(&bench.program.cubin,"SparkNvrmStreamCopy");
	bench.fill = SparkNvrmCubinKernelNamed(&bench.program.cubin,"SparkNvrmStreamFill");
	if ( bench.read == 0 || bench.copy == 0 || bench.fill == 0 )
	{
		printf("FAIL %s lacks the stream kernels\n",path);
		return(1);
	}
	bench.slot_bytes = SparkNvrmLaunchSlotBytes(bench.read);
	bench.slot_bytes = SparkNvrmLaunchSlotBytes(bench.copy) > bench.slot_bytes ? SparkNvrmLaunchSlotBytes(bench.copy) : bench.slot_bytes;
	bench.slot_bytes = SparkNvrmLaunchSlotBytes(bench.fill) > bench.slot_bytes ? SparkNvrmLaunchSlotBytes(bench.fill) : bench.slot_bytes;
	bench.blocks = bench.gpu.gpcs * bench.gpu.tpcs_per_gpc * bench.gpu.sms_per_tpc * BANDWIDTH_BLOCKS_PER_SM;
	if ( SparkNvrmMemoryAllocate(&bench.gpu,bench.slot_bytes,SPARK_NVRM_MEMORY_HOST,&bench.slots,"slots") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&bench.gpu,SPARK_NVRM_PAGE_BYTES,SPARK_NVRM_MEMORY_HOST,&bench.semaphore,"semaphore") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&bench.gpu,SPARK_NVRM_PAGE_BYTES,SPARK_NVRM_MEMORY_HOST,&bench.sink,"sink") != SPARK_STATUS_OK )
		return(Failed("allocate the launch memory"));
	printf("nvrm memory bandwidth: %u SMs, %u blocks of %u threads\n",bench.gpu.gpcs * bench.gpu.tpcs_per_gpc * bench.gpu.sms_per_tpc,bench.blocks,
		BANDWIDTH_THREADS);
	if ( result == 0 && (strcmp(only,"all") == 0 || strcmp(only,"host") == 0) )
		result = MeasurePath(&bench,SPARK_NVRM_MEMORY_HOST,"host memory registered by OS descriptor",bytes,rounds);
	if ( result == 0 && (strcmp(only,"all") == 0 || strcmp(only,"system") == 0) )
		result = MeasurePath(&bench,SPARK_NVRM_MEMORY_SYSTEM,"system memory allocated by the module, GPU cached",bytes,rounds);
	if ( result == 0 && (strcmp(only,"all") == 0 || strcmp(only,"uncached") == 0) )
		result = MeasurePath(&bench,SPARK_NVRM_MEMORY_SYSTEM_UNCACHED,"system memory allocated by the module, GPU uncached",bytes,rounds);
	SparkNvrmMemoryRelease(&bench.gpu,&bench.sink);
	SparkNvrmMemoryRelease(&bench.gpu,&bench.semaphore);
	SparkNvrmMemoryRelease(&bench.gpu,&bench.slots);
	SparkNvrmProgramRelease(&bench.gpu,&bench.program);
	SparkNvrmGpuClose(&bench.gpu);
	if ( result == 0 )
		printf("PASS every path measured and every sampled copy matches its source\n");
	return(result);
}
