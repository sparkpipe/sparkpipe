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
#define STATE_DEEP_THREADS 1024u
#define STATE_DEEP_GROUPS 512u
#define STATE_DEEP_PICK 3u

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
	SparkNvrmMemory deep;
} State;

static int Failed(const char *step)
{
	const SparkDeviceError *error = SparkDeviceLastError();
	printf("FAIL %s: %s [%s, %s, code %d]\n",step,error->text,error->operation,error->owner,error->backend_code);
	failures++;
	return(1);
}

static int RunGrid(State *state, const char *name, uint32_t groups, uint32_t threads, const void *arguments, uint32_t bytes)
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
	request.grid[0] = groups;
	request.grid[1] = request.grid[2] = 1u;
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

static int Run(State *state, const char *name, uint32_t threads, const void *arguments, uint32_t bytes)
{
	return(RunGrid(state,name,1u,threads,arguments,bytes));
}

static uint32_t FrameTotal(uint32_t lane)
{
	return(SPARK_NVRM_FIXTURE_FRAME_WORDS * lane * SPARK_NVRM_FIXTURE_DEEP_STRIDE + SPARK_NVRM_FIXTURE_FRAME_WORDS * (SPARK_NVRM_FIXTURE_FRAME_WORDS - 1u) / 2u);
}

static uint32_t DeepFrameWrong(State *state, uint32_t groups)
{
	const volatile uint32_t *words = (const volatile uint32_t *)(const void *)state->deep.cpu;
	SwapArguments call;
	uint32_t lane,wrong = 0u;
	memset(&call,0,sizeof(call));
	call.output = state->deep.address;
	call.value = STATE_DEEP_PICK;
	memset(state->deep.cpu,0xff,(size_t)groups * STATE_DEEP_THREADS * sizeof(uint32_t));
	if ( RunGrid(state,"SparkNvrmDeepFrame",groups,STATE_DEEP_THREADS,&call,SWAP_ARGUMENT_BYTES) != 0 )
		return(UINT32_MAX);
	for (lane = 0u; lane < groups * STATE_DEEP_THREADS; lane++)
		wrong += words[lane] != FrameTotal(lane);
	return(wrong);
}

static void TestStack(State *state)
{
	const SparkNvrmCubinKernel *frame = SparkNvrmCubinKernelNamed(&state->program.cubin,"SparkNvrmDeepFrame");
	const SparkNvrmCubinKernel *call = SparkNvrmCubinKernelNamed(&state->program.cubin,"SparkNvrmCallStack");
	SparkNvrmCubinKernel greedy;
	SparkNvrmLaunchRequest request;
	uint32_t wrong;
	CHECK(frame != 0 && call != 0 && frame->stack_bytes > SPARK_NVRM_FIXTURE_FRAME_WORDS * sizeof(uint32_t) - 1u && frame->frame_bytes == frame->stack_bytes &&
		state->gpu.local_bytes_per_thread >= call->stack_bytes + SPARK_NVRM_LOCAL_EXTRA_BYTES,
		"loading the module provisions local memory for its largest stack, a call chain's included");
	wrong = DeepFrameWrong(state,1u);
	CHECK(wrong == 0u,"every thread of a full group gets its own 4400-byte stack frame");
	if ( wrong == 0u )
	{
		wrong = DeepFrameWrong(state,STATE_DEEP_GROUPS);
		CHECK(wrong == 0u,"so does every thread of a grid that fills the GPU");
	}
	if ( frame == 0 || call == 0 )
		return;
	memset(&request,0,sizeof(request));
	request.kernel = call;
	request.grid[0] = request.grid[1] = request.grid[2] = 1u;
	request.block[0] = request.block[1] = request.block[2] = 1u;
	request.slot = state->slot.cpu;
	request.slot_address = state->slot.address;
	request.slot_bytes = (uint32_t)state->slot.bytes;
	CHECK(call->frame_bytes == 0u && call->stack_bytes > 0u &&
		SparkNvrmLaunchQueue(&state->gpu,&state->gpu.compute,&state->program,&request,"state") == SPARK_STATUS_UNSUPPORTED &&
		strstr(SparkDeviceLastError()->text,"frames of functions it calls") != 0,
		"a kernel whose callees keep stack frames of their own is refused, not run");
	greedy = *frame;
	greedy.stack_bytes = greedy.frame_bytes = state->gpu.local_bytes_per_thread;
	request.kernel = &greedy;
	CHECK(SparkNvrmLaunchQueue(&state->gpu,&state->gpu.compute,&state->program,&request,"state") == SPARK_STATUS_INVALID_ARGUMENT &&
		strstr(SparkDeviceLastError()->text,"local memory") != 0,"a kernel whose stack needs more than the provisioned local memory is refused");
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
		SparkNvrmMemoryAllocate(&state.gpu,SPARK_NVRM_PAGE_BYTES,SPARK_NVRM_MEMORY_HOST,&state.slot,"state") != SPARK_STATUS_OK ||
		SparkNvrmMemoryAllocate(&state.gpu,(uint64_t)STATE_DEEP_GROUPS * STATE_DEEP_THREADS * sizeof(uint32_t),SPARK_NVRM_MEMORY_HOST,&state.deep,"state") != SPARK_STATUS_OK )
		return(Failed("allocate the outputs and the launch slot"));
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
	TestStack(&state);
	SparkNvrmMemoryRelease(&state.gpu,&state.deep);
	SparkNvrmMemoryRelease(&state.gpu,&state.slot);
	SparkNvrmMemoryRelease(&state.gpu,&state.output);
	SparkNvrmProgramRelease(&state.gpu,&state.program);
	SparkNvrmGpuClose(&state.gpu);
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS a module's device variables start zeroed and persist across launches, its constant table reads back, its stack frames fit the local memory provisioned for them, and kernels it cannot run are refused\n");
	return(0);
}
