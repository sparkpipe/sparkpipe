#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_device.h"

#define PROGRAM_CAPACITY 70000u
#define PROGRAM_WIDTH 256u

static int failures;

#define CHECK(condition, what) do { if ( !(condition) ) { const SparkDeviceError *last = SparkDeviceLastError(); \
	printf("FAIL %s:%d %s (last device error: %s %s)\n", __FILE__, __LINE__, what, last->operation, last->text); failures++; } } while ( 0 )

typedef struct Gate
{
	pthread_mutex_t mutex;
	pthread_cond_t changed;
	int open;
} Gate;

static void GateWait(void *context)
{
	Gate *gate = (Gate *)context;
	pthread_mutex_lock(&gate->mutex);
	while ( gate->open == 0 )
		pthread_cond_wait(&gate->changed,&gate->mutex);
	pthread_mutex_unlock(&gate->mutex);
}

static void GateOpen(Gate *gate)
{
	pthread_mutex_lock(&gate->mutex);
	gate->open = 1;
	pthread_cond_broadcast(&gate->changed);
	pthread_mutex_unlock(&gate->mutex);
}

__global__ static void ProgramWrite(uint32_t *out, uint32_t count, uint32_t seed, uint64_t salt)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		out[index] = index * 2654435761u + seed + (uint32_t)(salt >> 7);
}

__global__ static void ProgramMix(const uint32_t *in, uint32_t *out, uint32_t count, uint32_t seed)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		out[index] = (in[index] ^ seed) * 747796405u + 2891336453u;
}

__global__ static void ProgramSlowWrite(uint32_t *out, uint32_t count, uint32_t seed, long long spin)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const long long start = clock64();
	while ( clock64() - start < spin )
		;
	if ( index < count )
		out[index] = index * 2654435761u + seed;
}

__global__ static void ProgramSlowMix(const uint32_t *in, uint32_t *out, uint32_t count, uint32_t seed, long long spin)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const long long start = clock64();
	while ( clock64() - start < spin )
		;
	if ( index < count )
		out[index] = (in[index] ^ seed) * 747796405u + 2891336453u;
}

static SparkDeviceRange At(const SparkDeviceBuffer *buffer, uint64_t offset)
{
	SparkDeviceRange range;
	range.buffer = buffer;
	range.offset = offset;
	return(range);
}

static SparkStatus Patch(SparkDeviceQueue *queue, uint32_t value, uint32_t target, uint32_t argument, uint64_t divide)
{
	SparkDevicePatch patch;
	memset(&patch,0,sizeof(patch));
	patch.value = value;
	patch.target = target;
	patch.argument = argument;
	patch.transform.multiply = 1;
	patch.transform.divide = divide;
	patch.transform.scale = 1;
	return(SparkDeviceProgramPatch(queue,&patch));
}

static SparkStatus PatchConstant(SparkDeviceQueue *queue, uint32_t value, uint32_t offset, uint32_t bytes)
{
	SparkDevicePatch patch;
	memset(&patch,0,sizeof(patch));
	patch.value = value;
	patch.target = SPARK_DEVICE_PROGRAM_TARGET_CONSTANT;
	patch.offset = offset;
	patch.bytes = bytes;
	return(SparkDeviceProgramPatch(queue,&patch));
}

static void TestEvents(SparkDeviceQueue *queue, cudaStream_t stream, SparkDeviceQueue *other, cudaStream_t other_stream)
{
	enum { COUNT = 65536, ROUNDS = 4 };
	const long long spin = 4000000ll;
	SparkDeviceBuffer written,mixed,landed;
	SparkDeviceEvent *ready = 0,*done = 0;
	SparkDeviceProgram *program = 0;
	SparkDeviceProgramInfo info;
	uint32_t round,index,wrong = 0u;
	CHECK(SparkDeviceAllocate(&written,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,COUNT * 4u,"event written") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&mixed,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,COUNT * 4u,"event mixed") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&landed,SPARK_DEVICE_SPACE_HOST_PINNED,COUNT * 4u,"event landed") == SPARK_STATUS_OK &&
		SparkDeviceEventCreate(&ready) == SPARK_STATUS_OK && SparkDeviceEventCreate(&done) == SPARK_STATUS_OK,"allocate the event test's buffers and events");
	CHECK(SparkDeviceProgramRecord(queue,"cuda event program") == SPARK_STATUS_OK && SparkDeviceQueueWaitEvent(queue,ready) == SPARK_STATUS_OK,
		"record a wait for an event the other stream records");
	ProgramSlowMix<<<COUNT / PROGRAM_WIDTH,PROGRAM_WIDTH,0,stream>>>((const uint32_t *)(uintptr_t)written.device,(uint32_t *)(uintptr_t)mixed.device,COUNT,0u,spin);
	CHECK(Patch(queue,0u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,3u,1u) == SPARK_STATUS_OK && SparkDeviceEventRecord(done,queue) == SPARK_STATUS_OK,
		"patch the mixer's seed and record the program's own event");
	CHECK(SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_OK && SparkDeviceProgramGetInfo(program,&info) == SPARK_STATUS_OK &&
		info.nodes == 3u && info.kernels == 1u && info.values == 1u,"the program holds a wait, one kernel and a record");
	CHECK(SparkDeviceEventDestroy(ready) == SPARK_STATUS_BUSY && SparkDeviceEventDestroy(done) == SPARK_STATUS_BUSY,
		"a live program keeps the events it waits on and records");
	for ( round = 0u; program != 0 && round < ROUNDS; round++ )
	{
		const uint32_t written_seed = 0x9e3779b9u * (round + 1u);
		const uint64_t seed = 101u + round * 7919u;
		memset(landed.host,0,COUNT * 4u);
		ProgramSlowWrite<<<COUNT / PROGRAM_WIDTH,PROGRAM_WIDTH,0,other_stream>>>((uint32_t *)(uintptr_t)written.device,COUNT,written_seed,spin * 3ll);
		CHECK(SparkDeviceEventRecord(ready,other) == SPARK_STATUS_OK && SparkDeviceProgramRun(program,queue,&seed,1u) == SPARK_STATUS_OK,
			"the other stream records its slow write and the program runs behind it");
		CHECK(SparkDeviceQueueWaitEvent(other,done) == SPARK_STATUS_OK && SparkDeviceCopy(other,At(&landed,0u),At(&mixed,0u),COUNT * 4u) == SPARK_STATUS_OK &&
			SparkDeviceQueueWait(other,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"the other stream waits for the program's event and copies its output");
		for ( index = 0u; index < COUNT; index++ )
			wrong += ((const uint32_t *)landed.host)[index] != (((index * 2654435761u + written_seed) ^ (uint32_t)seed) * 747796405u + 2891336453u);
	}
	CHECK(wrong == 0u,"each run waited for the other stream's write, and the other stream waited for each run");
	CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK && program != 0 && SparkDeviceProgramDestroy(program) == SPARK_STATUS_OK,
		"destroy the event program");
	CHECK(SparkDeviceEventDestroy(ready) == SPARK_STATUS_OK && SparkDeviceEventDestroy(done) == SPARK_STATUS_OK && SparkDeviceFree(&written) == SPARK_STATUS_OK &&
		SparkDeviceFree(&mixed) == SPARK_STATUS_OK && SparkDeviceFree(&landed) == SPARK_STATUS_OK,"the events and buffers go once the program is gone");
}

static void Eager(cudaStream_t stream, uint32_t *first, uint32_t *second, uint32_t count, uint32_t seed, const SparkDeviceBuffer *copy,
	SparkDeviceQueue *queue, const SparkDeviceBuffer *device_second)
{
	const uint32_t groups = (count + PROGRAM_WIDTH - 1u) / PROGRAM_WIDTH;
	ProgramWrite<<<groups,PROGRAM_WIDTH,0,stream>>>(first,count,seed,0x5151ull);
	ProgramMix<<<groups,PROGRAM_WIDTH,0,stream>>>(first,second,count,seed);
	CHECK(SparkDeviceCopy(queue,At(copy,0u),At(device_second,0u),PROGRAM_CAPACITY * 4u) == SPARK_STATUS_OK,"copy the eager result back");
}

int main(void)
{
	static const uint64_t values[][2] = {{11u, 70000u}, {977u, 1u}, {31337u, 4097u}, {31337u, 4097u}, {5u, 256u}};
	SparkDeviceQueue *queue = 0,*other = 0;
	SparkDeviceBuffer first,second,program_copy,eager_copy;
	SparkDeviceProgram *program = 0;
	SparkDeviceProgramInfo info;
	cudaStream_t stream = 0,other_stream = 0;
	Gate gate;
	uint32_t run,index,wrong = 0u;
	pthread_mutex_init(&gate.mutex,0);
	pthread_cond_init(&gate.changed,0);
	gate.open = 0;
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) != cudaSuccess ||
		cudaStreamCreateWithFlags(&other_stream,cudaStreamNonBlocking) != cudaSuccess )
	{
		printf("FAIL initialize: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	CHECK(SparkDeviceQueueAdopt(&queue,(void *)stream,"program stream") == SPARK_STATUS_OK &&
		SparkDeviceQueueAdopt(&other,(void *)other_stream,"program other stream") == SPARK_STATUS_OK,"adopt the two streams");
	CHECK(SparkDeviceAllocate(&first,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,PROGRAM_CAPACITY * 4u,"program first") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&second,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,PROGRAM_CAPACITY * 4u,"program second") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&program_copy,SPARK_DEVICE_SPACE_HOST_PINNED,PROGRAM_CAPACITY * 4u,"program copy") == SPARK_STATUS_OK &&
		SparkDeviceAllocate(&eager_copy,SPARK_DEVICE_SPACE_HOST_PINNED,PROGRAM_CAPACITY * 4u,"eager copy") == SPARK_STATUS_OK,"allocate the buffers");
	CHECK(SparkDeviceProgramRecord(queue,"cuda program") == SPARK_STATUS_OK,"start recording the stream");
	ProgramWrite<<<1u,PROGRAM_WIDTH,0,stream>>>((uint32_t *)(uintptr_t)first.device,1u,0u,0x5151ull);
	CHECK(Patch(queue,1u,SPARK_DEVICE_PROGRAM_TARGET_GRID_X,0u,PROGRAM_WIDTH) == SPARK_STATUS_OK &&
		Patch(queue,1u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,1u,1u) == SPARK_STATUS_OK &&
		Patch(queue,0u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,2u,1u) == SPARK_STATUS_OK,"patch the writer's grid, count and seed");
	ProgramMix<<<1u,PROGRAM_WIDTH,0,stream>>>((const uint32_t *)(uintptr_t)first.device,(uint32_t *)(uintptr_t)second.device,1u,0u);
	CHECK(Patch(queue,1u,SPARK_DEVICE_PROGRAM_TARGET_GRID_X,0u,PROGRAM_WIDTH) == SPARK_STATUS_OK &&
		Patch(queue,1u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,2u,1u) == SPARK_STATUS_OK &&
		Patch(queue,0u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,3u,1u) == SPARK_STATUS_OK,"patch the mixer's grid, count and seed");
	CHECK(Patch(queue,0u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,4u,1u) == SPARK_STATUS_INVALID_ARGUMENT,"a patch names one of the kernel's arguments");
	CHECK(PatchConstant(queue,0u,18u,4u) == SPARK_STATUS_INVALID_ARGUMENT && PatchConstant(queue,0u,20u,8u) == SPARK_STATUS_INVALID_ARGUMENT,
		"a constant patch covers exactly one parameter");
	CHECK(SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_INVALID_ARGUMENT && program == 0,"a refused patch breaks the recording");
	CHECK(SparkDeviceProgramRecord(queue,"cuda program") == SPARK_STATUS_OK,"record again");
	ProgramWrite<<<1u,PROGRAM_WIDTH,0,stream>>>((uint32_t *)(uintptr_t)first.device,1u,0u,0x5151ull);
	CHECK(Patch(queue,1u,SPARK_DEVICE_PROGRAM_TARGET_GRID_X,0u,PROGRAM_WIDTH) == SPARK_STATUS_OK &&
		Patch(queue,1u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,1u,1u) == SPARK_STATUS_OK &&
		Patch(queue,0u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,2u,1u) == SPARK_STATUS_OK,"patch the writer again");
	ProgramMix<<<1u,PROGRAM_WIDTH,0,stream>>>((const uint32_t *)(uintptr_t)first.device,(uint32_t *)(uintptr_t)second.device,1u,0u);
	CHECK(Patch(queue,1u,SPARK_DEVICE_PROGRAM_TARGET_GRID_X,0u,PROGRAM_WIDTH) == SPARK_STATUS_OK &&
		Patch(queue,1u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,2u,1u) == SPARK_STATUS_OK &&
		Patch(queue,0u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,3u,1u) == SPARK_STATUS_OK,"patch the mixer again");
	CHECK(SparkDeviceCopy(queue,At(&program_copy,0u),At(&second,0u),PROGRAM_CAPACITY * 4u) == SPARK_STATUS_OK,"record a device-layer copy");
	CHECK(Patch(queue,0u,SPARK_DEVICE_PROGRAM_TARGET_GRID_X,0u,1u) == SPARK_STATUS_INVALID_ARGUMENT,"a patch must follow a kernel launch");
	CHECK(SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_INVALID_ARGUMENT && program == 0,"and the refusal breaks that recording too");
	CHECK(SparkDeviceProgramRecord(queue,"cuda program") == SPARK_STATUS_OK,"record the program");
	ProgramWrite<<<1u,PROGRAM_WIDTH,0,stream>>>((uint32_t *)(uintptr_t)first.device,1u,0u,0x5151ull);
	CHECK(Patch(queue,1u,SPARK_DEVICE_PROGRAM_TARGET_GRID_X,0u,PROGRAM_WIDTH) == SPARK_STATUS_OK &&
		Patch(queue,1u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,1u,1u) == SPARK_STATUS_OK &&
		Patch(queue,0u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,2u,1u) == SPARK_STATUS_OK,"patch the writer");
	ProgramMix<<<1u,PROGRAM_WIDTH,0,stream>>>((const uint32_t *)(uintptr_t)first.device,(uint32_t *)(uintptr_t)second.device,1u,0u);
	CHECK(Patch(queue,1u,SPARK_DEVICE_PROGRAM_TARGET_GRID_X,0u,PROGRAM_WIDTH) == SPARK_STATUS_OK &&
		Patch(queue,1u,SPARK_DEVICE_PROGRAM_TARGET_ARGUMENT,2u,1u) == SPARK_STATUS_OK &&
		PatchConstant(queue,0u,20u,4u) == SPARK_STATUS_OK,"patch the mixer, its seed by the parameter's place");
	CHECK(SparkDeviceCopy(queue,At(&program_copy,0u),At(&second,0u),PROGRAM_CAPACITY * 4u) == SPARK_STATUS_OK,"record the copy back");
	CHECK(SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_OK && program != 0,"finish the program");
	CHECK(SparkDeviceProgramGetInfo(program,&info) == SPARK_STATUS_OK && info.kernels == 2u && info.values == 2u && info.places == 6u && info.nodes == 3u,
		"the program holds two captured kernels, six places for two values and the copy, three nodes in all");
	CHECK(SparkDeviceFree(&program_copy) == SPARK_STATUS_BUSY,"a live program keeps the buffers its device-layer operations use");
	for ( run = 0u; program != 0 && run < sizeof(values) / sizeof(values[0]); run++ )
	{
		SparkDeviceQueue *target = run % 2u == 0u ? queue : other;
		memset(program_copy.host,0,PROGRAM_CAPACITY * 4u);
		memset(eager_copy.host,0,PROGRAM_CAPACITY * 4u);
		CHECK(cudaMemsetAsync((void *)(uintptr_t)second.device,0,PROGRAM_CAPACITY * 4u,target == queue ? stream : other_stream) == cudaSuccess,
			"clear the device result");
		CHECK(SparkDeviceProgramRun(program,target,values[run],2u) == SPARK_STATUS_OK && SparkDeviceQueueWait(target,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
			"run the program with new values, on either stream");
		CHECK(cudaMemsetAsync((void *)(uintptr_t)second.device,0,PROGRAM_CAPACITY * 4u,stream) == cudaSuccess,"clear the device result again");
		Eager(stream,(uint32_t *)(uintptr_t)first.device,(uint32_t *)(uintptr_t)second.device,(uint32_t)values[run][1],(uint32_t)values[run][0],
			&eager_copy,queue,&second);
		CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"wait for the eager launches");
		for ( index = 0u; index < PROGRAM_CAPACITY; index++ )
			wrong += ((const uint32_t *)program_copy.host)[index] != ((const uint32_t *)eager_copy.host)[index];
	}
	CHECK(wrong == 0u,"every run writes exactly the bytes the same launches write eagerly");
	CHECK(SparkDeviceQueueHostCallback(queue,GateWait,&gate) == SPARK_STATUS_OK && program != 0 &&
		SparkDeviceProgramRun(program,queue,values[0],2u) == SPARK_STATUS_OK,"hold the stream, then queue a run behind the hold");
	CHECK(SparkDeviceProgramDestroy(program) == SPARK_STATUS_BUSY,"a program with a queued run is not destroyed");
	GateOpen(&gate);
	CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK && SparkDeviceProgramDestroy(program) == SPARK_STATUS_OK,
		"once the run is done the program is destroyed");
	TestEvents(queue,stream,other,other_stream);
	CHECK(SparkDeviceFree(&first) == SPARK_STATUS_OK && SparkDeviceFree(&second) == SPARK_STATUS_OK && SparkDeviceFree(&program_copy) == SPARK_STATUS_OK &&
		SparkDeviceFree(&eager_copy) == SPARK_STATUS_OK,"free the buffers");
	CHECK(SparkDeviceQueueDestroy(queue) == SPARK_STATUS_OK && SparkDeviceQueueDestroy(other) == SPARK_STATUS_OK && SparkDeviceShutdown() == SPARK_STATUS_OK,
		"release the queues and the device");
	if ( failures != 0 )
	{
		printf("\nFAIL device_program_cuda (%d)\n",failures);
		return(1);
	}
	printf("PASS device_program_cuda: recorded CUDA launches run as one graph per program, patched per run, bit-identical to eager launches, "
		"and a program's event waits and records order it against another stream\n");
	return(0);
}
