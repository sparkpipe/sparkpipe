#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"
#include "runtime/device/nvrm/nvrm_launch.h"

#define SEGMENT_BATCH 1000u
#define PROGRAM_LAUNCHES 64u
#define COMPETE_SAMPLES_MAX 200000u

__global__ void SegmentStep(uint32_t *counter, uint64_t spin_ns)
{
	asm volatile("griddepcontrol.wait;" ::: "memory");
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
	{
		uint64_t start,now;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
		do
			asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
		while ( now - start < spin_ns );
		counter[0] += 1u;
	}
}

__global__ void CompeteStep(uint32_t *counter)
{
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
		counter[0] += 1u;
}

static uint64_t Now(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static int CompareTimes(const void *left, const void *right)
{
	const uint64_t a = *(const uint64_t *)left,b = *(const uint64_t *)right;
	return(a < b ? -1 : a > b);
}

static int Chain(uint64_t launches, double seconds, uint32_t segment, uint64_t spin_ns)
{
	SparkGpuStream stream = 0;
	uint32_t *counter = 0,seen = 0u;
	uint64_t done = 0u,start;
	SparkNvrmLatchSegmentSet(segment);
	if ( SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS || SparkGpuMalloc(&counter,sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMemset(counter,0,sizeof(uint32_t)) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL set up the chain: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	start = Now();
	while ( (launches != 0u && done < launches) || (launches == 0u && (double)(Now() - start) < seconds * 1e9) )
	{
		uint32_t index;
		for ( index = 0u; index < SEGMENT_BATCH && (launches == 0u || done < launches); index++,done++ )
			if ( LM_LAUNCH_DEPENDENT((SegmentStep),1,32,0u,stream,counter,spin_ns) != SPARK_GPU_SUCCESS )
			{
				printf("FAIL dependent launch %llu: %s\n",(unsigned long long)done,SparkDeviceLastError()->text);
				return(1);
			}
	}
	if ( SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpy(&seen,counter,sizeof(seen),SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL finish the chain: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	printf("chain: %llu dependent launches of %.0f us in segments of %u, %.3f us each, counter %u\n",(unsigned long long)done,(double)spin_ns / 1e3,SparkNvrmLatchSegment(),
		(double)(Now() - start) / 1e3 / (double)done,seen);
	if ( seen != (uint32_t)done )
	{
		printf("FAIL the chain counted %u of %llu launches\n",seen,(unsigned long long)done);
		return(1);
	}
	SparkGpuFree(counter);
	SparkGpuStreamDestroy(stream);
	return(0);
}

static int ChainProgram(uint64_t launches, uint32_t segment, uint64_t spin_ns)
{
	SparkGpuStream stream = 0;
	SparkDeviceQueue *queue = 0;
	SparkDeviceProgram *program = 0;
	SparkStatus initialized = SparkDeviceInitialize(0);
	uint32_t *counter = 0,seen = 0u,index;
	uint64_t runs = (launches + PROGRAM_LAUNCHES - 1u) / PROGRAM_LAUNCHES,run,start;
	SparkNvrmLatchSegmentSet(segment);
	if ( (initialized != SPARK_STATUS_OK && initialized != SPARK_STATUS_DUPLICATE) ||
		SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS || SparkGpuMalloc(&counter,sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMemset(counter,0,sizeof(uint32_t)) != SPARK_GPU_SUCCESS || SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS ||
		SparkDeviceQueueAdopt(&queue,(void *)stream,"latch segments") != SPARK_STATUS_OK || SparkDeviceProgramRecord(queue,"latch segments") != SPARK_STATUS_OK )
	{
		printf("FAIL set up the program chain: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	for ( index = 0u; index < PROGRAM_LAUNCHES; index++ )
		LM_LAUNCH((SegmentStep),1,32,0u,stream,counter,spin_ns);
	if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS || SparkDeviceProgramFinish(queue,&program) != SPARK_STATUS_OK )
	{
		printf("FAIL record the program chain: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	start = Now();
	for ( run = 0u; run < runs; run++ )
		if ( SparkDeviceProgramRun(program,queue,0,0u) != SPARK_STATUS_OK )
		{
			printf("FAIL program run %llu: %s\n",(unsigned long long)run,SparkDeviceLastError()->text);
			return(1);
		}
	if ( SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK ||
		SparkGpuMemcpy(&seen,counter,sizeof(seen),SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL finish the program chain: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	printf("program chain: %llu runs of %u launches of %.0f us in segments of %u, %.3f us each, counter %u\n",(unsigned long long)runs,PROGRAM_LAUNCHES,
		(double)spin_ns / 1e3,SparkNvrmLatchSegment(),(double)(Now() - start) / 1e3 / (double)(runs * PROGRAM_LAUNCHES),seen);
	if ( seen != (uint32_t)(runs * PROGRAM_LAUNCHES) || SparkDeviceProgramDestroy(program) != SPARK_STATUS_OK || SparkDeviceQueueDestroy(queue) != SPARK_STATUS_OK )
	{
		printf("FAIL the program chain counted %u of %llu launches\n",seen,(unsigned long long)(runs * PROGRAM_LAUNCHES));
		return(1);
	}
	SparkGpuFree(counter);
	SparkGpuStreamDestroy(stream);
	return(0);
}

static int Compete(double seconds, double wait_p99_ms)
{
	static uint64_t waits[COMPETE_SAMPLES_MAX];
	SparkGpuStream stream = 0;
	uint32_t *counter = 0;
	uint64_t start,count = 0u;
	if ( SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS || SparkGpuMalloc(&counter,sizeof(uint32_t)) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL set up the competitor: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	start = Now();
	while ( (double)(Now() - start) < seconds * 1e9 && count < COMPETE_SAMPLES_MAX )
	{
		const uint64_t submitted = Now();
		LM_LAUNCH((CompeteStep),1,32,0u,stream,counter);
		if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS || SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS )
		{
			printf("FAIL competitor launch %llu: %s\n",(unsigned long long)count,SparkDeviceLastError()->text);
			return(1);
		}
		waits[count++] = Now() - submitted;
	}
	qsort(waits,(size_t)count,sizeof(waits[0]),CompareTimes);
	printf("compete: %llu launches in %.1f s, wait p50 %.1f us, p99 %.1f us, max %.1f us\n",(unsigned long long)count,(double)(Now() - start) / 1e9,
		(double)waits[count / 2u] / 1e3,(double)waits[count * 99u / 100u] / 1e3,(double)waits[count - 1u] / 1e3);
	SparkGpuFree(counter);
	SparkGpuStreamDestroy(stream);
	if ( (double)waits[count * 99u / 100u] / 1e6 > wait_p99_ms )
	{
		printf("FAIL the competitor's 99th percentile wait was %.3f ms, over %.3f ms\n",(double)waits[count * 99u / 100u] / 1e6,wait_p99_ms);
		return(1);
	}
	return(0);
}

int main(int argc, char **argv)
{
	const uint64_t spin_ns = argc == 5 ? strtoull(argv[4],0,10) * 1000ull : 0u;
	if ( (argc == 4 || argc == 5) && strcmp(argv[1],"chain") == 0 )
		return(Chain(strtoull(argv[2],0,10),0.0,(uint32_t)strtoul(argv[3],0,10),spin_ns));
	if ( (argc == 4 || argc == 5) && strcmp(argv[1],"chain-seconds") == 0 )
		return(Chain(0u,strtod(argv[2],0),(uint32_t)strtoul(argv[3],0,10),spin_ns));
	if ( (argc == 4 || argc == 5) && strcmp(argv[1],"program") == 0 )
		return(ChainProgram(strtoull(argv[2],0,10),(uint32_t)strtoul(argv[3],0,10),spin_ns));
	if ( argc == 4 && strcmp(argv[1],"compete") == 0 )
		return(Compete(strtod(argv[2],0),strtod(argv[3],0)));
	printf("usage: %s chain LAUNCHES SEGMENT [SPIN_US] | chain-seconds SECONDS SEGMENT [SPIN_US] | program LAUNCHES SEGMENT [SPIN_US] | compete SECONDS P99_WAIT_MS\n",argv[0]);
	return(2);
}
