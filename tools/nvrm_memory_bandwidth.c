#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_launch.h"

#define BANDWIDTH_THREADS 256u
#define BANDWIDTH_BLOCKS_PER_SM 4u
#define BANDWIDTH_TIMEOUT_NS 30000000000ull
#define BANDWIDTH_SAMPLES 4096u
#define BANDWIDTH_ROUNDS_MAX 256u
#define BANDWIDTH_ELEMENT_BYTES (4u * sizeof(uint32_t))

typedef struct ReadArguments { uint64_t source; uint64_t count; uint64_t sink; } ReadArguments;
typedef struct CopyArguments { uint64_t source; uint64_t destination; uint64_t count; } CopyArguments;
typedef struct FillArguments { uint64_t destination; uint64_t count; uint32_t value; } FillArguments;
#define FILL_ARGUMENT_BYTES ((uint32_t)(offsetof(FillArguments,value) + sizeof(uint32_t)))

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

static int CompareTimes(const void *left, const void *right)
{
	const uint64_t a = *(const uint64_t *)left,b = *(const uint64_t *)right;
	return(a < b ? -1 : a > b);
}

static uint64_t NowNanoseconds(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000u + (uint64_t)now.tv_nsec);
}

static int EngineCopy(Bench *bench, uint64_t destination, uint64_t source, uint64_t bytes, uint64_t *nanoseconds)
{
	const uint64_t started = NowNanoseconds();
	uint64_t value = 0u;
	if ( SparkNvrmCopy(&bench->gpu,&bench->gpu.compute,destination,source,bytes,&value,"bandwidth") != SPARK_STATUS_OK )
		return(Failed("queue a copy-engine copy"));
	if ( SparkNvrmWaitValue((const volatile uint64_t *)(void *)bench->gpu.compute.fence.cpu,value,BANDWIDTH_TIMEOUT_NS,"copy","bandwidth") != SPARK_STATUS_OK )
		return(Failed("wait for a copy-engine copy"));
	*nanoseconds = NowNanoseconds() - started;
	return(0);
}

static int Run(Bench *bench, const SparkNvrmCubinKernel *kernel, const void *arguments, uint32_t bytes, uint64_t *gpu_nanoseconds)
{
	SparkNvrmChannel *channel = &bench->gpu.compute;
	volatile uint64_t *semaphore = (volatile uint64_t *)(void *)bench->semaphore.cpu;
	SparkNvrmLaunchRequest request;
	const uint64_t value = ++bench->sequence;
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
		SparkNvrmLaunchQueue(&bench->gpu,channel,&bench->program,&request,"bandwidth") != SPARK_STATUS_OK ||
		SparkNvrmSemaphoreRelease(channel,bench->semaphore.address + 16u,value,1u,"bandwidth") != SPARK_STATUS_OK ||
		SparkNvrmSubmit(&bench->gpu,channel,"bandwidth") != SPARK_STATUS_OK )
		return(Failed("submit a measured launch"));
	if ( SparkNvrmWaitValue(&semaphore[2],value,BANDWIDTH_TIMEOUT_NS,"kernel_launch","bandwidth") != SPARK_STATUS_OK )
		return(Failed("wait for a measured launch"));
	*gpu_nanoseconds = semaphore[3] - semaphore[1];
	return(0);
}

static void Report(const char *label, uint64_t *times, uint32_t rounds, uint64_t moved)
{
	qsort(times,rounds,sizeof(times[0]),CompareTimes);
	printf("  %-5s median %8.2f GB/s  best %8.2f GB/s  (%u rounds, %.3f ms median)\n",label,(double)moved / (double)times[rounds / 2u],
		(double)moved / (double)times[0],rounds,(double)times[rounds / 2u] / 1e6);
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
	Report(label,times,rounds,moved);
	return(0);
}

static int Sampled(const SparkNvrmMemory *expected, const SparkNvrmMemory *seen, uint64_t count, const char *what)
{
	uint32_t index;
	for (index = 0u; index < BANDWIDTH_SAMPLES; index++)
	{
		const uint64_t at = ((uint64_t)index * 2654435761u) % count;
		if ( memcmp(expected->cpu + at * BANDWIDTH_ELEMENT_BYTES,seen->cpu + at * BANDWIDTH_ELEMENT_BYTES,BANDWIDTH_ELEMENT_BYTES) != 0 )
		{
			printf("FAIL %s differs from its source at element %llu\n",what,(unsigned long long)at);
			return(1);
		}
	}
	return(0);
}

static int MeasurePcie(Bench *bench, SparkNvrmMemoryKind kind, const char *name, uint64_t bytes, uint32_t rounds)
{
	SparkNvrmMemory host,back,video;
	uint64_t to_video[BANDWIDTH_ROUNDS_MAX],to_host[BANDWIDTH_ROUNDS_MAX],index;
	uint32_t round;
	int result = 0;
	printf("copy engine between %s and video memory, %llu MiB per copy\n",name,(unsigned long long)(bytes >> 20));
	if ( SparkNvrmMemoryAllocate(&bench->gpu,bytes,kind,&host,"pcie host") != SPARK_STATUS_OK )
		return(Failed("allocate the host side"));
	if ( SparkNvrmMemoryAllocate(&bench->gpu,bytes,kind,&back,"pcie back") != SPARK_STATUS_OK )
	{
		SparkNvrmMemoryRelease(&bench->gpu,&host);
		return(Failed("allocate the host return"));
	}
	if ( SparkNvrmMemoryAllocate(&bench->gpu,bytes,SPARK_NVRM_MEMORY_VIDEO,&video,"pcie video") != SPARK_STATUS_OK )
	{
		SparkNvrmMemoryRelease(&bench->gpu,&back);
		SparkNvrmMemoryRelease(&bench->gpu,&host);
		return(Failed("allocate the video side"));
	}
	printf("  host pages %u bytes, video pages %u bytes\n",host.page_bytes,video.page_bytes);
	for (index = 0u; index < bytes / sizeof(uint64_t); index++)
		((uint64_t *)(void *)host.cpu)[index] = index * 0x9e3779b97f4a7c15ull;
	for (round = 0u; result == 0 && round < 2u; round++)
		result = EngineCopy(bench,video.address,host.address,bytes,&to_video[0]) || EngineCopy(bench,back.address,video.address,bytes,&to_host[0]);
	for (round = 0u; result == 0 && round < rounds; round++)
		result = EngineCopy(bench,video.address,host.address,bytes,&to_video[round]) || EngineCopy(bench,back.address,video.address,bytes,&to_host[round]);
	if ( result == 0 )
	{
		Report("to video",to_video,rounds,bytes);
		Report("to host",to_host,rounds,bytes);
		result = Sampled(&host,&back,bytes / BANDWIDTH_ELEMENT_BYTES,"the round trip through video memory");
	}
	SparkNvrmMemoryRelease(&bench->gpu,&video);
	SparkNvrmMemoryRelease(&bench->gpu,&back);
	SparkNvrmMemoryRelease(&bench->gpu,&host);
	return(result);
}

static int VerifyVideo(Bench *bench, const SparkNvrmMemory *source, const SparkNvrmMemory *destination, uint64_t bytes, uint32_t value)
{
	SparkNvrmMemory from,to;
	uint64_t nanoseconds = 0u;
	uint32_t index,first;
	int result;
	if ( SparkNvrmMemoryAllocate(&bench->gpu,bytes,SPARK_NVRM_MEMORY_HOST,&from,"verify source") != SPARK_STATUS_OK )
		return(Failed("allocate the host copy of the source"));
	if ( SparkNvrmMemoryAllocate(&bench->gpu,bytes,SPARK_NVRM_MEMORY_HOST,&to,"verify destination") != SPARK_STATUS_OK )
	{
		SparkNvrmMemoryRelease(&bench->gpu,&from);
		return(Failed("allocate the host copy of the destination"));
	}
	result = EngineCopy(bench,from.address,source->address,bytes,&nanoseconds) || EngineCopy(bench,to.address,destination->address,bytes,&nanoseconds);
	if ( result == 0 )
		result = Sampled(&from,&to,bytes / BANDWIDTH_ELEMENT_BYTES,"the copy");
	for (index = 0u; result == 0 && index < BANDWIDTH_SAMPLES; index++)
	{
		memcpy(&first,from.cpu + (((uint64_t)index * 2654435761u) % (bytes / BANDWIDTH_ELEMENT_BYTES)) * BANDWIDTH_ELEMENT_BYTES,sizeof(first));
		if ( first != value )
		{
			printf("FAIL the fill did not reach video memory\n");
			result = 1;
		}
	}
	SparkNvrmMemoryRelease(&bench->gpu,&to);
	SparkNvrmMemoryRelease(&bench->gpu,&from);
	return(result);
}

static int MeasurePath(Bench *bench, SparkNvrmMemoryKind kind, const char *name, uint64_t bytes, uint32_t rounds)
{
	SparkNvrmMemory source,destination;
	const uint64_t count = bytes / BANDWIDTH_ELEMENT_BYTES;
	const uint32_t value = 0x13572468u;
	FillArguments fill;
	ReadArguments read;
	CopyArguments copy;
	uint64_t nanoseconds = 0u;
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
	memset(&fill,0,sizeof(fill));
	fill.destination = source.address;
	fill.count = count;
	fill.value = value;
	read.source = source.address;
	read.count = count;
	read.sink = bench->sink.address;
	copy.source = source.address;
	copy.destination = destination.address;
	copy.count = count;
	result = Run(bench,bench->fill,&fill,FILL_ARGUMENT_BYTES,&nanoseconds);
	if ( result == 0 )
		result = Measure(bench,"read",bench->read,&read,(uint32_t)sizeof(read),bytes,rounds);
	if ( result == 0 )
		result = Measure(bench,"copy",bench->copy,&copy,(uint32_t)sizeof(copy),2u * bytes,rounds);
	if ( result == 0 && kind == SPARK_NVRM_MEMORY_VIDEO )
		result = VerifyVideo(bench,&source,&destination,bytes,value);
	for (index = 0u; result == 0 && kind != SPARK_NVRM_MEMORY_VIDEO && index < BANDWIDTH_SAMPLES; index++)
	{
		const uint64_t at = ((uint64_t)index * 2654435761u) % count;
		const uint8_t *from = source.cpu + at * BANDWIDTH_ELEMENT_BYTES,*to = destination.cpu + at * BANDWIDTH_ELEMENT_BYTES;
		uint32_t first;
		memcpy(&first,from,sizeof(first));
		if ( memcmp(from,to,BANDWIDTH_ELEMENT_BYTES) != 0 || first != value )
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
	uint64_t bytes = 1024ull << 20;
	uint32_t rounds = 20u;
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
			printf("usage: %s [--cubin PATH] [--mib N] [--rounds N] [--path host|system|uncached|video|pcie|all]\n",argv[0]);
			return(2);
		}
	}
	if ( rounds == 0u || rounds > BANDWIDTH_ROUNDS_MAX || bytes == 0u )
	{
		printf("FAIL rounds must be 1 to %u and the buffer at least 1 MiB\n",BANDWIDTH_ROUNDS_MAX);
		return(2);
	}
	if ( SparkNvrmGpuOpen(&bench.gpu,0u,SPARK_NVRM_MEMORY_HOST,"bandwidth") != SPARK_STATUS_OK )
		return(Failed("open the GPU"));
	if ( SparkNvrmProgramLoadFile(&bench.gpu,path,&bench.program,"bandwidth") != SPARK_STATUS_OK )
		return(Failed("load the kernel image and its manifest"));
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
	bench.blocks = bench.gpu.sm_count * BANDWIDTH_BLOCKS_PER_SM;
	if ( SparkNvrmMemoryAllocate(&bench.gpu,bench.slot_bytes,SPARK_NVRM_MEMORY_HOST,&bench.slots,"slots") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&bench.gpu,SPARK_NVRM_PAGE_BYTES,SPARK_NVRM_MEMORY_HOST,&bench.semaphore,"semaphore") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&bench.gpu,SPARK_NVRM_PAGE_BYTES,SPARK_NVRM_MEMORY_HOST,&bench.sink,"sink") != SPARK_STATUS_OK )
		return(Failed("allocate the launch memory"));
	printf("nvrm memory bandwidth: %u SMs, %u blocks of %u threads\n",bench.gpu.sm_count,bench.blocks,
		BANDWIDTH_THREADS);
	if ( result == 0 && (strcmp(only,"all") == 0 || strcmp(only,"host") == 0) )
		result = MeasurePath(&bench,SPARK_NVRM_MEMORY_HOST,"host memory registered by OS descriptor",bytes,rounds);
	if ( result == 0 && (strcmp(only,"all") == 0 || strcmp(only,"system") == 0) )
		result = MeasurePath(&bench,SPARK_NVRM_MEMORY_SYSTEM,"system memory allocated by the module, GPU cached",bytes,rounds);
	if ( result == 0 && (strcmp(only,"all") == 0 || strcmp(only,"uncached") == 0) )
		result = MeasurePath(&bench,SPARK_NVRM_MEMORY_SYSTEM_UNCACHED,"system memory allocated by the module, GPU uncached",bytes,rounds);
	if ( result == 0 && ((strcmp(only,"all") == 0 && bench.gpu.integrated == 0u) || strcmp(only,"video") == 0) )
		result = MeasurePath(&bench,SPARK_NVRM_MEMORY_VIDEO,"video memory allocated by the module",bytes,rounds);
	if ( result == 0 && ((strcmp(only,"all") == 0 && bench.gpu.integrated == 0u) || strcmp(only,"pcie") == 0) )
		result = MeasurePcie(&bench,SPARK_NVRM_MEMORY_SYSTEM,"system memory allocated by the module",bytes,rounds);
	if ( result == 0 && ((strcmp(only,"all") == 0 && bench.gpu.integrated == 0u) || strcmp(only,"pcie") == 0) )
		result = MeasurePcie(&bench,SPARK_NVRM_MEMORY_HOST,"host memory registered by OS descriptor",bytes,rounds);
	SparkNvrmMemoryRelease(&bench.gpu,&bench.sink);
	SparkNvrmMemoryRelease(&bench.gpu,&bench.semaphore);
	SparkNvrmMemoryRelease(&bench.gpu,&bench.slots);
	SparkNvrmProgramRelease(&bench.gpu,&bench.program);
	SparkNvrmGpuClose(&bench.gpu);
	if ( result == 0 )
		printf("PASS every path measured and every sampled copy matches its source\n");
	return(result);
}
