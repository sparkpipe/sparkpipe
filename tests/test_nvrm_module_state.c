#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/nvrm_launch.h"
#include "tests/nvrm/fixture_kernels.h"

#ifndef SPARK_NVRM_FIXTURE_CUBIN
#define SPARK_NVRM_FIXTURE_CUBIN "tests/fixtures/nvrm/fixture_kernels.sm_121a.cubin"
#endif

#define STATE_TIMEOUT_NS 10000000000ull
#define STATE_TABLE_THREADS 8u
#define STATE_FIRST 0x1234abcdu
#define STATE_SECOND 0x0badf00du

typedef struct SwapArguments { uint64_t output; uint32_t value; } SwapArguments;
#define SWAP_ARGUMENT_BYTES ((uint32_t)(offsetof(SwapArguments,value) + sizeof(uint32_t)))

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s\n",what); \
			failures++; \
		} \
	} while (0)

typedef struct State
{
	SparkNvrmGpu gpu;
	SparkNvrmProgram program;
	SparkNvrmMemory output;
	SparkNvrmMemory slot;
} State;

static int Failed(const char *step)
{
	const SparkDeviceError *error = SparkDeviceLastError();
	printf("FAIL %s: %s [%s, %s, code %d]\n",step,error->text,error->operation,error->owner,error->backend_code);
	failures++;
	return(1);
}

static int Run(State *state, const char *name, uint32_t threads, const void *arguments, uint32_t bytes)
{
	SparkNvrmLaunchRequest request;
	uint64_t value = 0u;
	memset(&request,0,sizeof(request));
	request.kernel = SparkNvrmCubinKernelNamed(&state->program.cubin,name);
	if ( request.kernel == 0 )
	{
		printf("FAIL the fixture has no kernel %s\n",name);
		failures++;
		return(1);
	}
	request.grid[0] = request.grid[1] = request.grid[2] = 1u;
	request.block[0] = threads;
	request.block[1] = request.block[2] = 1u;
	request.arguments = arguments;
	request.argument_bytes = bytes;
	request.slot = state->slot.cpu;
	request.slot_address = state->slot.address;
	request.slot_bytes = (uint32_t)state->slot.bytes;
	if ( SparkNvrmLaunch(&state->gpu,&state->gpu.compute,&state->program,&request,&value,"state") != SPARK_STATUS_OK ||
		SparkNvrmWaitValue((const volatile uint64_t *)(void *)state->gpu.compute.fence.cpu,value,STATE_TIMEOUT_NS,"kernel_launch","state") != SPARK_STATUS_OK )
		return(Failed(name));
	return(0);
}

int main(void)
{
	static State state;
	static const uint32_t table[SPARK_NVRM_FIXTURE_TABLE_WORDS] = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u};
	volatile uint32_t *output;
	SwapArguments swap;
	uint32_t index,wrong = 0u;
	if ( access("/dev/nvidiactl",F_OK) != 0 )
	{
		printf("SKIP test_nvrm_module_state (no open kernel module on this host)\n");
		return(0);
	}
	if ( SparkNvrmGpuOpen(&state.gpu,0u,SPARK_NVRM_MEMORY_HOST,"state") != SPARK_STATUS_OK )
		return(Failed("open the GPU"));
	if ( SparkNvrmProgramLoadFile(&state.gpu,SPARK_NVRM_FIXTURE_CUBIN,&state.program,"state") != SPARK_STATUS_OK )
		return(Failed("load the fixture"));
	if ( SparkNvrmMemoryAllocate(&state.gpu,SPARK_NVRM_PAGE_BYTES,SPARK_NVRM_MEMORY_HOST,&state.output,"state") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&state.gpu,SPARK_NVRM_PAGE_BYTES,SPARK_NVRM_MEMORY_HOST,&state.slot,"state") != SPARK_STATUS_OK )
		return(Failed("allocate the output and the launch slot"));
	output = (volatile uint32_t *)(void *)state.output.cpu;
	memset(&swap,0,sizeof(swap));
	swap.output = state.output.address;
	swap.value = STATE_FIRST;
	output[0] = 0xffffffffu;
	if ( Run(&state,"SparkNvrmGlobalSwap",1u,&swap,SWAP_ARGUMENT_BYTES) == 0 )
		CHECK(output[0] == 0u,"a module's __device__ variable starts zeroed, at the address its relocations name");
	swap.value = STATE_SECOND;
	if ( Run(&state,"SparkNvrmGlobalSwap",1u,&swap,SWAP_ARGUMENT_BYTES) == 0 )
		CHECK(output[0] == STATE_FIRST,"the variable keeps what the previous launch stored");
	if ( Run(&state,"SparkNvrmConstantTable",STATE_TABLE_THREADS,&swap.output,(uint32_t)sizeof(swap.output)) == 0 )
	{
		for (index = 0u; index < STATE_TABLE_THREADS; index++)
			wrong += output[index] != table[index % SPARK_NVRM_FIXTURE_TABLE_WORDS];
		CHECK(wrong == 0u,"a module's initialized __constant__ table reads back in every thread");
	}
	SparkNvrmMemoryRelease(&state.gpu,&state.slot);
	SparkNvrmMemoryRelease(&state.gpu,&state.output);
	SparkNvrmProgramRelease(&state.gpu,&state.program);
	SparkNvrmGpuClose(&state.gpu);
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS a module's device variables start zeroed and persist across launches, and its constant table reads back\n");
	return(0);
}
