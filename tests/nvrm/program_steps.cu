#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

#define STEPS_THREADS 128u
#define STEPS_GROUPS 64u
#define STEPS_WORDS (STEPS_GROUPS * STEPS_THREADS)
#define STEPS_ROUNDS 120u
#define STEPS_RUNS 4u

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s: %s / %s\n",what,SparkGpuGetErrorString(SparkGpuGetLastError()),SparkDeviceLastError()->text); \
			failures++; \
		} \
	} while (0)

__global__ void StepPlain(uint32_t *data, uint32_t salt)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	data[index] = data[index] * 3u + salt;
}

__global__ void StepDependent(uint32_t *data, uint32_t salt)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t value;
	asm volatile("griddepcontrol.wait;" ::: "memory");
	value = data[index];
	data[index] = (value ^ (value >> 7)) * 5u + salt;
}

__global__ void __cluster_dims__(2, 1, 1) StepCluster(uint32_t *data, uint32_t salt)
{
	__shared__ uint32_t mine,theirs;
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t rank;
	asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
	if ( threadIdx.x == 0u )
		mine = data[blockIdx.x * blockDim.x];
	asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
	if ( threadIdx.x == 0u )
	{
		uint32_t local = (uint32_t)__cvta_generic_to_shared(&mine),remote,peer;
		asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(rank ^ 1u));
		asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(peer) : "r"(remote) : "memory");
		theirs = peer;
	}
	asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
	data[index] = data[index] + theirs * 7u + rank + salt;
}

static void StepsCount(void *context)
{
	__atomic_add_fetch((uint32_t *)context,1u,__ATOMIC_SEQ_CST);
}

static uint64_t StepsNow(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static uint64_t StepsDigest(const uint32_t *words, uint32_t count)
{
	const uint8_t *bytes = (const uint8_t *)words;
	uint64_t digest = 1469598103934665603ull;
	uint32_t index;
	for ( index = 0u; index < count * sizeof(uint32_t); index++ )
		digest = (digest ^ bytes[index]) * 1099511628211ull;
	return(digest);
}

static int StepsIssue(SparkGpuStream stream, uint32_t *data, uint32_t *calls)
{
	uint32_t round;
	for ( round = 0u; round < STEPS_ROUNDS; round++ )
	{
		LM_LAUNCH((StepPlain),STEPS_GROUPS,STEPS_THREADS,0u,stream,data,round);
		if ( LM_LAUNCH_DEPENDENT((StepDependent),STEPS_GROUPS,STEPS_THREADS,0u,stream,data,round + 1000u) != SPARK_GPU_SUCCESS )
			return(0);
		LM_LAUNCH((StepCluster),STEPS_GROUPS,STEPS_THREADS,0u,stream,data,round + 2000u);
		if ( round == STEPS_ROUNDS / 2u && SparkGpuLaunchHostFunc(stream,StepsCount,calls) != SPARK_GPU_SUCCESS )
			return(0);
	}
	return(SparkGpuGetLastError() == SPARK_GPU_SUCCESS);
}

int main(int argc, char **argv)
{
	const char *flavour = argc > 1 ? argv[1] : "this flavour";
	SparkGpuStream stream = 0;
	SparkDeviceQueue *queue = 0;
	SparkDeviceProgram *program = 0;
	SparkDeviceProgramInfo info;
	SparkStatus initialized;
	uint32_t *program_data = 0,*eager_data = 0,*start,*from_program,*from_eager,calls = 0u,run,index,wrong = 0u;
	uint64_t run_ns = 0u,issue_ns = 0u,begin;
	start = (uint32_t *)malloc(STEPS_WORDS * sizeof(uint32_t));
	from_program = (uint32_t *)malloc(STEPS_WORDS * sizeof(uint32_t));
	from_eager = (uint32_t *)malloc(STEPS_WORDS * sizeof(uint32_t));
	for ( index = 0u; start != 0 && index < STEPS_WORDS; index++ )
		start[index] = index * 2654435761u + 12345u;
	initialized = SparkDeviceInitialize(0);
	CHECK(start != 0 && from_program != 0 && from_eager != 0 && SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) == SPARK_GPU_SUCCESS &&
		SparkGpuMalloc(&program_data,STEPS_WORDS * sizeof(uint32_t)) == SPARK_GPU_SUCCESS &&
		SparkGpuMalloc(&eager_data,STEPS_WORDS * sizeof(uint32_t)) == SPARK_GPU_SUCCESS &&
		(initialized == SPARK_STATUS_OK || initialized == SPARK_STATUS_DUPLICATE) &&
		SparkDeviceQueueAdopt(&queue,(void *)stream,"program steps") == SPARK_STATUS_OK,"set up the stream, its device queue and two buffers");
	if ( failures != 0 )
		return(1);
	CHECK(SparkGpuMemcpy(program_data,start,STEPS_WORDS * sizeof(uint32_t),SPARK_GPU_COPY_HOST_TO_DEVICE) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpy(eager_data,start,STEPS_WORDS * sizeof(uint32_t),SPARK_GPU_COPY_HOST_TO_DEVICE) == SPARK_GPU_SUCCESS,"fill both buffers alike");
	CHECK(SparkDeviceProgramRecord(queue,"program steps") == SPARK_STATUS_OK && StepsIssue(stream,program_data,&calls) != 0 &&
		SparkDeviceProgramFinish(queue,&program) == SPARK_STATUS_OK && program != 0,"record plain, dependent and cluster launches and a host callback");
	CHECK(SparkDeviceProgramGetInfo(program,&info) == SPARK_STATUS_OK && info.kernels == 3u * STEPS_ROUNDS && info.values == 0u &&
		__atomic_load_n(&calls,__ATOMIC_SEQ_CST) == 0u,"the program holds every launch, and recording ran nothing");
	for ( run = 0u; failures == 0 && run < STEPS_RUNS; run++ )
	{
		begin = StepsNow();
		CHECK(SparkDeviceProgramRun(program,queue,0,0u) == SPARK_STATUS_OK,"run the program");
		run_ns += StepsNow() - begin;
		begin = StepsNow();
		CHECK(StepsIssue(stream,eager_data,&calls) != 0,"issue the same steps eagerly");
		issue_ns += StepsNow() - begin;
	}
	CHECK(SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpy(from_program,program_data,STEPS_WORDS * sizeof(uint32_t),SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpy(from_eager,eager_data,STEPS_WORDS * sizeof(uint32_t),SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS,"read both buffers");
	for ( index = 0u; index < STEPS_WORDS; index++ )
		wrong += from_program[index] != from_eager[index] ? 1u : 0u;
	CHECK(wrong == 0u && memcmp(from_program,start,STEPS_WORDS * sizeof(uint32_t)) != 0,"every run of the program writes the bytes the eager steps write");
	CHECK(__atomic_load_n(&calls,__ATOMIC_SEQ_CST) == 2u * STEPS_RUNS,"each program run and each eager step calls the host callback once");
	CHECK(SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK && SparkDeviceProgramDestroy(program) == SPARK_STATUS_OK &&
		SparkDeviceQueueDestroy(queue) == SPARK_STATUS_OK,"once the queue reports the runs done, destroy the program and the queue");
	printf("digest program steps %016llx\n",(unsigned long long)StepsDigest(from_program,STEPS_WORDS));
	printf("program steps %s: %u launches and a host callback; host %.1f us per program run, %.1f us per eager issue\n",flavour,3u * STEPS_ROUNDS,
		(double)run_ns / 1e3 / STEPS_RUNS,(double)issue_ns / 1e3 / STEPS_RUNS);
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS program steps %s: a program of plain, dependent and cluster launches and a host callback writes the eager bytes on every run\n",flavour);
	return(0);
}
