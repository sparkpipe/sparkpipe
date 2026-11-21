#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_launch.h"
#include "runtime/device/nvrm/nvrm_version.h"

#define LIGHT_BLOCKS 4u
#define LIGHT_THREADS 128u
#define LIGHT_ROUNDS 16u
#define LIGHT_TIMEOUT_NS 5000000000ull

typedef struct Light
{
	SparkNvrmGpu gpu;
	SparkNvrmProgram program;
	const SparkNvrmCubinKernel *kernel;
	SparkNvrmMemory output;
	SparkNvrmMemory launch;
	SparkNvrmMemory semaphore;
} Light;

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

static void Describe(const SparkNvrmGpu *gpu)
{
	uint32_t index;
	printf("module %s, GPU minor %u instance %u, architecture %#x implementation %#x\n",gpu->rm.loaded_version,gpu->minor,gpu->instance,
		gpu->architecture,gpu->implementation);
	printf("classes: compute %#x channel %#x copy %#x usermode %#x\n",gpu->compute_class,gpu->gpfifo_class,gpu->copy_class,gpu->usermode_class);
	printf("%u GPCs x %u TPCs x %u SMs, %u warps per SM, SM version %#x, instruction set version %#x\nuuid ",gpu->gpcs,gpu->tpcs_per_gpc,
		gpu->sms_per_tpc,gpu->warps_per_sm,gpu->sm_version,gpu->sass_version);
	for (index = 0u; index < 16u; index++)
		printf("%02x",gpu->uuid[index]);
	printf("\n");
}

static int Load(Light *light, const char *path)
{
	uint64_t bytes = 0u;
	uint8_t *blob = ReadFile(path,&bytes);
	SparkStatus status;
	if ( blob == 0 )
	{
		printf("FAIL cannot read %s\n",path);
		return(1);
	}
	status = SparkNvrmProgramLoad(&light->gpu,blob,bytes,&light->program,"first light");
	free(blob);
	if ( status != SPARK_STATUS_OK )
		return(Failed("load the kernel image"));
	light->kernel = SparkNvrmCubinKernelNamed(&light->program.cubin,"SparkNvrmFirstLight");
	if ( light->kernel == 0 )
	{
		printf("FAIL %s holds no SparkNvrmFirstLight kernel\n",path);
		return(1);
	}
	if ( SparkNvrmMemoryAllocate(&light->gpu,LIGHT_BLOCKS * LIGHT_THREADS * sizeof(uint32_t),SPARK_NVRM_MEMORY_HOST,&light->output,"output") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&light->gpu,SparkNvrmLaunchSlotBytes(light->kernel),SPARK_NVRM_MEMORY_HOST,&light->launch,"launch") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&light->gpu,SPARK_NVRM_PAGE_BYTES,SPARK_NVRM_MEMORY_HOST,&light->semaphore,"semaphore") != SPARK_STATUS_OK )
		return(Failed("allocate launch memory"));
	return(0);
}

static int Launch(Light *light, uint32_t value, uint64_t round, uint64_t *nanoseconds, uint64_t *gpu_nanoseconds)
{
	SparkNvrmChannel *channel = &light->gpu.compute;
	volatile uint64_t *semaphores = (volatile uint64_t *)(void *)light->semaphore.cpu;
	SparkNvrmLaunchRequest request;
	uint8_t arguments[sizeof(uint64_t) + sizeof(uint32_t)];
	uint32_t index;
	const uint64_t first = channel->push_used,output = light->output.address;
	uint64_t start;
	memcpy(arguments,&output,sizeof(output));
	memcpy(arguments + sizeof(output),&value,sizeof(value));
	memset(light->output.cpu,0xee,(size_t)LIGHT_BLOCKS * LIGHT_THREADS * sizeof(uint32_t));
	memset(&request,0,sizeof(request));
	request.kernel = light->kernel;
	request.grid[0] = LIGHT_BLOCKS;
	request.grid[1] = request.grid[2] = 1u;
	request.block[0] = LIGHT_THREADS;
	request.block[1] = request.block[2] = 1u;
	request.arguments = arguments;
	request.argument_bytes = sizeof(arguments);
	request.slot = light->launch.cpu;
	request.slot_address = light->launch.address;
	request.slot_bytes = (uint32_t)light->launch.bytes;
	request.release_address = light->semaphore.address;
	request.release_payload = round;
	start = SparkMonotonicNs();
	if ( SparkNvrmSemaphoreRelease(channel,light->semaphore.address + 32u,round,1u,"first light") != SPARK_STATUS_OK ||
		SparkNvrmLaunchQueue(&light->gpu,&light->program,&request,"first light") != SPARK_STATUS_OK ||
		SparkNvrmSemaphoreRelease(channel,light->semaphore.address + 16u,round,1u,"first light") != SPARK_STATUS_OK ||
		SparkNvrmSubmit(&light->gpu,channel,first,"first light") != SPARK_STATUS_OK )
		return(Failed("submit the launch"));
	if ( SparkNvrmWaitValue(&semaphores[0],round,LIGHT_TIMEOUT_NS,"kernel_launch","grid release") != SPARK_STATUS_OK )
		return(Failed("wait for the grid's release"));
	if ( SparkNvrmWaitValue(&semaphores[2],round,LIGHT_TIMEOUT_NS,"kernel_launch","host release") != SPARK_STATUS_OK )
		return(Failed("wait for the channel's release"));
	*nanoseconds = SparkMonotonicNs() - start;
	*gpu_nanoseconds = semaphores[3] - semaphores[5];
	for (index = 0u; index < LIGHT_BLOCKS * LIGHT_THREADS; index++)
	{
		const uint32_t got = ((const volatile uint32_t *)(void *)light->output.cpu)[index];
		if ( got != (value ^ index) )
		{
			printf("FAIL output[%u] is %#x, the kernel writes %#x\n",index,got,value ^ index);
			return(1);
		}
	}
	return(0);
}

static void Release(Light *light)
{
	if ( light->semaphore.bytes != 0u )
		SparkNvrmMemoryRelease(&light->gpu,&light->semaphore);
	if ( light->launch.bytes != 0u )
		SparkNvrmMemoryRelease(&light->gpu,&light->launch);
	if ( light->output.bytes != 0u )
		SparkNvrmMemoryRelease(&light->gpu,&light->output);
	SparkNvrmProgramRelease(&light->gpu,&light->program);
	SparkNvrmGpuClose(&light->gpu);
}

int main(int argc, char **argv)
{
	static Light light;
	const char *path = "tests/fixtures/nvrm/fixture_kernels.sm_121a.cubin";
	SparkNvrmMemoryKind control = SPARK_NVRM_MEMORY_HOST;
	uint64_t total = 0u,best = UINT64_MAX,gpu_total = 0u,round;
	int index,result = 0;
	for (index = 1; index < argc; index++)
	{
		if ( strcmp(argv[index],"--cubin") == 0 && index + 1 < argc )
			path = argv[++index];
		else if ( strcmp(argv[index],"--control") == 0 && index + 1 < argc && (strcmp(argv[index + 1],"host") == 0 || strcmp(argv[index + 1],"system") == 0) )
			control = strcmp(argv[++index],"system") == 0 ? SPARK_NVRM_MEMORY_SYSTEM : SPARK_NVRM_MEMORY_HOST;
		else
		{
			printf("usage: %s [--cubin PATH] [--control host|system]\n",argv[0]);
			return(2);
		}
	}
	printf("nvrm first light: backend for module %s, kernel image %s, channel control memory %s\n",SPARK_NVRM_MODULE_VERSION,path,
		control == SPARK_NVRM_MEMORY_SYSTEM ? "system" : "host");
	if ( SparkNvrmGpuOpen(&light.gpu,0u,control,"first light") != SPARK_STATUS_OK )
		return(Failed("open the GPU"));
	Describe(&light.gpu);
	result = Load(&light,path);
	for (round = 1u; result == 0 && round <= LIGHT_ROUNDS; round++)
	{
		uint64_t nanoseconds = 0u,gpu_nanoseconds = 0u;
		result = Launch(&light,0x5a5a0000u + (uint32_t)round,round,&nanoseconds,&gpu_nanoseconds);
		total += nanoseconds;
		gpu_total += gpu_nanoseconds;
		best = nanoseconds < best ? nanoseconds : best;
	}
	if ( result == 0 )
		printf("PASS first light: %u launches of %u x %u threads wrote every word; host submit to completion %.1f us mean, %.1f us best; "
			"GPU timestamps around the grid %.1f us mean\n",LIGHT_ROUNDS,LIGHT_BLOCKS,LIGHT_THREADS,(double)total / LIGHT_ROUNDS / 1000.0,
			(double)best / 1000.0,(double)gpu_total / LIGHT_ROUNDS / 1000.0);
	Release(&light);
	return(result);
}
