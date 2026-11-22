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
#define TRIGGER_PAIRS 32u
#define STALE_PARENT_GROUPS 48u
#define STALE_CHILD_GROUPS 96u
#define STALE_PAIRS 32u

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

__global__ void TriggerParent(uint32_t *word, uint32_t value, uint64_t spin_ns)
{
	asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
	{
		uint64_t start,now;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
		do
			asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
		while ( now - start < spin_ns );
		*(volatile uint32_t *)word = value;
	}
}

__global__ void TriggerPlain(const uint32_t *word, uint32_t *seen, uint32_t slot)
{
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
		seen[slot] = *(const volatile uint32_t *)word;
}

__global__ void TriggerWaiting(const uint32_t *word, uint32_t *seen, uint32_t slot)
{
	asm volatile("griddepcontrol.wait;" ::: "memory");
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
		seen[slot] = *(const volatile uint32_t *)word;
}

static __device__ __forceinline__ uint32_t StaleLoad(const uint32_t *word)
{
	uint32_t value;
	asm volatile("ld.global.ca.u32 %0,[%1];" : "=r"(value) : "l"(word) : "memory");
	return(value);
}

__global__ void StaleParent(uint32_t *word, uint32_t value, uint64_t spin_ns, uint32_t trigger, uint32_t *sink)
{
	if ( trigger != 0u )
		asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
	if ( threadIdx.x != 0u )
		return;
	if ( blockIdx.x != 0u )
	{
		sink[blockIdx.x] = StaleLoad(word);
		return;
	}
	uint64_t start,now;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	do
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
	while ( now - start < spin_ns );
	word[0] = value;
}

__global__ void StaleChild(const uint32_t *word, uint32_t *seen, uint32_t slot, uint32_t wait)
{
	if ( wait != 0u )
		asm volatile("griddepcontrol.wait;" ::: "memory");
	if ( threadIdx.x == 0u )
		seen[slot * STALE_CHILD_GROUPS + blockIdx.x] = StaleLoad(word);
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
		if ( LM_LAUNCH_DEPENDENT((SegmentStep),1,32,0u,stream,counter,spin_ns) != SPARK_GPU_SUCCESS )
		{
			printf("FAIL record dependent launch %u: %s\n",index,SparkDeviceLastError()->text);
			return(1);
		}
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
	printf("program chain: %llu runs of %u dependent launches of %.0f us in segments of %u, %.3f us each, counter %u\n",(unsigned long long)runs,PROGRAM_LAUNCHES,
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

static int TriggerIssue(SparkGpuStream stream, uint32_t *word, uint32_t *seen, uint64_t spin_ns)
{
	uint32_t pair;
	for ( pair = 0u; pair < TRIGGER_PAIRS; pair++ )
	{
		LM_LAUNCH((TriggerParent),1,32,0u,stream,word,pair + 1u,spin_ns);
		if ( (pair & 1u) == 0u )
			LM_LAUNCH((TriggerPlain),1,32,0u,stream,(const uint32_t *)word,seen,pair);
		else if ( LM_LAUNCH_DEPENDENT((TriggerWaiting),1,32,0u,stream,(const uint32_t *)word,seen,pair) != SPARK_GPU_SUCCESS )
			return(0);
	}
	return(SparkGpuGetLastError() == SPARK_GPU_SUCCESS);
}

static int Trigger(uint64_t runs, uint64_t spin_ns)
{
	static const char *const ways[2] = {"eager","program"};
	SparkGpuStream stream = 0;
	SparkDeviceQueue *queue = 0;
	SparkDeviceProgram *program = 0;
	SparkStatus initialized = SparkDeviceInitialize(0);
	uint32_t *word = 0,*seen = 0,host[TRIGGER_PAIRS],pair,way;
	uint64_t run,late[2][2] = {{0u,0u},{0u,0u}},first[2][2][2];
	memset(first,0,sizeof(first));
	if ( (initialized != SPARK_STATUS_OK && initialized != SPARK_STATUS_DUPLICATE) ||
		SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS || SparkGpuMalloc(&word,sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&seen,TRIGGER_PAIRS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS || SparkGpuMemset(word,0,sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS || SparkDeviceQueueAdopt(&queue,(void *)stream,"latch trigger") != SPARK_STATUS_OK ||
		SparkDeviceProgramRecord(queue,"latch trigger") != SPARK_STATUS_OK || TriggerIssue(stream,word,seen,spin_ns) == 0 ||
		SparkDeviceProgramFinish(queue,&program) != SPARK_STATUS_OK )
	{
		printf("FAIL set up the trigger chain: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	for ( run = 0u; run < runs; run++ )
		for ( way = 0u; way < 2u; way++ )
		{
			if ( SparkGpuMemset(seen,0xff,TRIGGER_PAIRS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
				(way == 0u ? TriggerIssue(stream,word,seen,spin_ns) == 0 : SparkDeviceProgramRun(program,queue,0,0u) != SPARK_STATUS_OK) ||
				SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS ||
				SparkGpuMemcpy(host,seen,sizeof(host),SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
			{
				printf("FAIL trigger run %llu %s: %s\n",(unsigned long long)run,ways[way],SparkDeviceLastError()->text);
				return(1);
			}
			for ( pair = 0u; pair < TRIGGER_PAIRS; pair++ )
				if ( host[pair] != pair + 1u && late[way][pair & 1u]++ == 0u )
				{
					first[way][pair & 1u][0] = pair;
					first[way][pair & 1u][1] = host[pair];
				}
		}
	for ( way = 0u; way < 2u; way++ )
	{
		printf("trigger %s: %llu runs of %u parents that trigger their dependents first and write %.0f us later\n",ways[way],(unsigned long long)runs,
			TRIGGER_PAIRS,(double)spin_ns / 1e3);
		printf("  plain children read a value from before their parent's write %llu of %llu times",(unsigned long long)late[way][0],
			(unsigned long long)(runs * TRIGGER_PAIRS / 2u));
		if ( late[way][0] != 0u )
			printf(" (first: child %llu read %llu)",(unsigned long long)first[way][0][0],(unsigned long long)first[way][0][1]);
		printf("\n  waiting dependent children did %llu of %llu times",(unsigned long long)late[way][1],(unsigned long long)(runs * TRIGGER_PAIRS / 2u));
		if ( late[way][1] != 0u )
			printf(" (first: child %llu read %llu)",(unsigned long long)first[way][1][0],(unsigned long long)first[way][1][1]);
		printf("\n");
	}
	if ( SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK || SparkDeviceProgramDestroy(program) != SPARK_STATUS_OK ||
		SparkDeviceQueueDestroy(queue) != SPARK_STATUS_OK )
	{
		printf("FAIL finish the trigger chain: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	SparkGpuFree(word);
	SparkGpuFree(seen);
	SparkGpuStreamDestroy(stream);
	if ( late[0][0] + late[0][1] + late[1][0] + late[1][1] != 0u )
	{
		printf("FAIL a child read its parent's output before the parent wrote it: a plain launch waits for its parent to end, and a dependent launch's griddepcontrol.wait does\n");
		return(1);
	}
	return(0);
}

static int StaleIssue(SparkGpuStream stream, uint32_t *word, uint32_t *seen, uint32_t *sink, uint64_t spin_ns)
{
	uint32_t pair;
	for ( pair = 0u; pair < STALE_PAIRS; pair++ )
	{
		const uint32_t trigger = (pair & 1u) == 0u ? 1u : 0u,wait = (pair & 2u) != 0u ? 1u : 0u;
		LM_LAUNCH((StaleParent),STALE_PARENT_GROUPS,32,0u,stream,word,pair + 1u,spin_ns,trigger,sink);
		if ( wait == 0u )
			LM_LAUNCH((StaleChild),STALE_CHILD_GROUPS,32,0u,stream,(const uint32_t *)word,seen,pair,wait);
		else if ( LM_LAUNCH_DEPENDENT((StaleChild),STALE_CHILD_GROUPS,32,0u,stream,(const uint32_t *)word,seen,pair,wait) != SPARK_GPU_SUCCESS )
			return(0);
	}
	return(SparkGpuGetLastError() == SPARK_GPU_SUCCESS);
}

static int Stale(uint64_t runs, uint64_t spin_ns)
{
	static const char *const ways[2] = {"eager","program"};
	static uint32_t host[STALE_PAIRS * STALE_CHILD_GROUPS];
	SparkGpuStream stream = 0;
	SparkDeviceQueue *queue = 0;
	SparkDeviceProgram *program = 0;
	SparkStatus initialized = SparkDeviceInitialize(0);
	uint32_t *word = 0,*seen = 0,*sink = 0,pair,group,way,kind;
	uint64_t run,late[2][4],reads[2][4];
	memset(late,0,sizeof(late));
	memset(reads,0,sizeof(reads));
	if ( (initialized != SPARK_STATUS_OK && initialized != SPARK_STATUS_DUPLICATE) ||
		SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS || SparkGpuMalloc(&word,sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&seen,sizeof(host)) != SPARK_GPU_SUCCESS || SparkGpuMalloc(&sink,STALE_PARENT_GROUPS * sizeof(uint32_t)) != SPARK_GPU_SUCCESS ||
		SparkGpuMemset(word,0,sizeof(uint32_t)) != SPARK_GPU_SUCCESS || SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS ||
		SparkDeviceQueueAdopt(&queue,(void *)stream,"latch stale") != SPARK_STATUS_OK || SparkDeviceProgramRecord(queue,"latch stale") != SPARK_STATUS_OK ||
		StaleIssue(stream,word,seen,sink,spin_ns) == 0 || SparkDeviceProgramFinish(queue,&program) != SPARK_STATUS_OK )
	{
		printf("FAIL set up the stale-line chain: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	for ( run = 0u; run < runs; run++ )
		for ( way = 0u; way < 2u; way++ )
		{
			if ( SparkGpuMemset(seen,0xff,sizeof(host)) != SPARK_GPU_SUCCESS ||
				(way == 0u ? StaleIssue(stream,word,seen,sink,spin_ns) == 0 : SparkDeviceProgramRun(program,queue,0,0u) != SPARK_STATUS_OK) ||
				SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS || SparkGpuMemcpy(host,seen,sizeof(host),SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
			{
				printf("FAIL stale-line run %llu %s: %s\n",(unsigned long long)run,ways[way],SparkDeviceLastError()->text);
				return(1);
			}
			for ( pair = 0u; pair < STALE_PAIRS; pair++ )
				for ( group = 0u; group < STALE_CHILD_GROUPS; group++ )
				{
					kind = pair & 3u;
					reads[way][kind]++;
					if ( host[pair * STALE_CHILD_GROUPS + group] != pair + 1u )
						late[way][kind]++;
				}
		}
	for ( way = 0u; way < 2u; way++ )
	{
		printf("stale %s: %llu runs; a parent of %u groups whose other groups cache the word, then group 0 writes it %.0f us later; children of %u groups read it cached\n",
			ways[way],(unsigned long long)runs,STALE_PARENT_GROUPS,(double)spin_ns / 1e3,STALE_CHILD_GROUPS);
		for ( kind = 0u; kind < 4u; kind++ )
			printf("  parent %s, %s child: %llu of %llu reads saw the value from before the parent's write\n",(kind & 1u) == 0u ? "triggers first" : "never triggers",
				(kind & 2u) != 0u ? "waiting dependent" : "plain",(unsigned long long)late[way][kind],(unsigned long long)reads[way][kind]);
	}
	if ( SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK || SparkDeviceProgramDestroy(program) != SPARK_STATUS_OK ||
		SparkDeviceQueueDestroy(queue) != SPARK_STATUS_OK )
	{
		printf("FAIL finish the stale-line chain: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	SparkGpuFree(word);
	SparkGpuFree(seen);
	SparkGpuFree(sink);
	SparkGpuStreamDestroy(stream);
	for ( way = 0u; way < 2u; way++ )
		for ( kind = 0u; kind < 4u; kind++ )
			if ( late[way][kind] != 0u )
			{
				printf("FAIL a child read a cached line from before its parent's write\n");
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
	if ( argc == 4 && strcmp(argv[1],"trigger") == 0 )
		return(Trigger(strtoull(argv[2],0,10),strtoull(argv[3],0,10) * 1000ull));
	if ( argc == 4 && strcmp(argv[1],"stale") == 0 )
		return(Stale(strtoull(argv[2],0,10),strtoull(argv[3],0,10) * 1000ull));
	printf("usage: %s chain LAUNCHES SEGMENT [SPIN_US] | chain-seconds SECONDS SEGMENT [SPIN_US] | program LAUNCHES SEGMENT [SPIN_US] | compete SECONDS P99_WAIT_MS | trigger RUNS SPIN_US | stale RUNS SPIN_US\n",argv[0]);
	return(2);
}
