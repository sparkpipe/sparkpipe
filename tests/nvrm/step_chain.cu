#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

#define CHAIN_MAX_NODES 2627u
#define CHAIN_RUNS 10
#define CHAIN_WARMUP 2
#define CHAIN_WORK_NS 2000ull
#define CHAIN_THREADS 128u

__global__ void ChainNode(unsigned long long *starts, unsigned long long *ends, uint32_t node, unsigned long long work_ns)
{
	unsigned long long start,now;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	do
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
	while ( now - start < work_ns );
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		atomicMin(&starts[node],start);
		atomicMax(&ends[node],now);
	}
}

static uint64_t ChainNow(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

typedef struct ChainResult
{
	double issue_us;
	double wall_us;
	double span_us;
	double idle_per_node_us;
	double busy_us;
} ChainResult;

static int ChainIssue(SparkGpuStream stream, unsigned long long *starts, unsigned long long *ends, uint32_t nodes, int blocks)
{
	uint32_t node;
	for (node = 0u; node < nodes; node++)
		LM_LAUNCH((ChainNode),blocks,CHAIN_THREADS,0,stream,starts,ends,node,CHAIN_WORK_NS);
	return(SparkGpuGetLastError() == SPARK_GPU_SUCCESS);
}

static int ChainMeasure(SparkGpuStream stream, SparkGpuGraphExec exec, unsigned long long *starts, unsigned long long *ends, unsigned long long *host_starts,
	unsigned long long *host_ends, uint32_t nodes, int blocks, ChainResult *result)
{
	int run;
	memset(result,0,sizeof(*result));
	for (run = -CHAIN_WARMUP; run < CHAIN_RUNS; run++)
	{
		uint64_t begin,issued,done;
		unsigned long long first = ~0ull,last = 0ull,busy = 0ull;
		uint32_t node;
		if ( SparkGpuMemsetAsync(starts,0xff,nodes * sizeof(*starts),stream) != SPARK_GPU_SUCCESS ||
			SparkGpuMemsetAsync(ends,0,nodes * sizeof(*ends),stream) != SPARK_GPU_SUCCESS || SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS )
			return(0);
		begin = ChainNow();
		if ( exec != 0 ? SparkGpuGraphLaunch(exec,stream) != SPARK_GPU_SUCCESS : ChainIssue(stream,starts,ends,nodes,blocks) == 0 )
			return(0);
		issued = ChainNow();
		if ( SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS )
			return(0);
		done = ChainNow();
		if ( SparkGpuMemcpy(host_starts,starts,nodes * sizeof(*starts),SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS ||
			SparkGpuMemcpy(host_ends,ends,nodes * sizeof(*ends),SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
			return(0);
		for (node = 0u; node < nodes; node++)
		{
			first = host_starts[node] < first ? host_starts[node] : first;
			last = host_ends[node] > last ? host_ends[node] : last;
			busy += host_ends[node] - host_starts[node];
		}
		if ( run < 0 )
			continue;
		result->issue_us += (double)(issued - begin) / 1e3 / CHAIN_RUNS;
		result->wall_us += (double)(done - begin) / 1e3 / CHAIN_RUNS;
		result->span_us += (double)(last - first) / 1e3 / CHAIN_RUNS;
		result->busy_us += (double)busy / 1e3 / CHAIN_RUNS;
	}
	result->idle_per_node_us = nodes > 1u ? (result->span_us - result->busy_us) / (double)(nodes - 1u) : 0.0;
	return(1);
}

static void ChainPrint(const char *flavour, const char *mode, uint32_t nodes, int blocks, const ChainResult *result)
{
	printf("chain %s %s nodes %u blocks %d: host issue %.1f us (%.3f us per node), wall %.1f us, GPU span %.1f us, kernels busy %.1f us, "
		"GPU idle %.3f us per node boundary\n",flavour,mode,nodes,blocks,result->issue_us,result->issue_us / nodes,result->wall_us,result->span_us,result->busy_us,
		result->idle_per_node_us);
}

int main(int argc, char **argv)
{
	static const uint32_t counts[] = {16u, 256u, 1600u, CHAIN_MAX_NODES};
	const char *flavour = argc > 1 ? argv[1] : "this flavour";
	SparkGpuStream stream = 0;
	unsigned long long *starts = 0,*ends = 0,*host_starts,*host_ends;
	int sms = 0,widths[2],width;
	uint32_t count;
	host_starts = (unsigned long long *)calloc(CHAIN_MAX_NODES,sizeof(*host_starts));
	host_ends = (unsigned long long *)calloc(CHAIN_MAX_NODES,sizeof(*host_ends));
	if ( host_starts == 0 || host_ends == 0 || SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc(&starts,CHAIN_MAX_NODES * sizeof(*starts)) != SPARK_GPU_SUCCESS || SparkGpuMalloc(&ends,CHAIN_MAX_NODES * sizeof(*ends)) != SPARK_GPU_SUCCESS ||
		SparkGpuDeviceGetAttribute(&sms,SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT,0) != SPARK_GPU_SUCCESS || sms <= 0 )
	{
		printf("FAIL set up the chain: %s\n",SparkGpuGetErrorString(SparkGpuGetLastError()));
		return(1);
	}
	widths[0] = 1;
	widths[1] = sms;
	for (width = 0; width < 2; width++)
		for (count = 0u; count < sizeof(counts) / sizeof(counts[0]); count++)
		{
			ChainResult result;
			SparkGpuGraph graph = 0;
			SparkGpuGraphExec exec = 0;
			if ( ChainMeasure(stream,0,starts,ends,host_starts,host_ends,counts[count],widths[width],&result) == 0 )
			{
				printf("FAIL an eager chain of %u nodes: %s\n",counts[count],SparkGpuGetErrorString(SparkGpuGetLastError()));
				return(1);
			}
			ChainPrint(flavour,"eager",counts[count],widths[width],&result);
			if ( SparkGpuStreamBeginCapture(stream,SPARK_GPU_CAPTURE_MODE_THREAD_LOCAL) != SPARK_GPU_SUCCESS )
			{
				(void)SparkGpuGetLastError();
				printf("chain %s graph nodes %u blocks %d: capture refused (%s)\n",flavour,counts[count],widths[width],"no graphs in this flavour");
				continue;
			}
			if ( ChainIssue(stream,starts,ends,counts[count],widths[width]) == 0 || SparkGpuStreamEndCapture(stream,&graph) != SPARK_GPU_SUCCESS ||
				SparkGpuGraphInstantiate(&exec,graph,0) != SPARK_GPU_SUCCESS || SparkGpuGraphUpload(exec,stream) != SPARK_GPU_SUCCESS ||
				ChainMeasure(stream,exec,starts,ends,host_starts,host_ends,counts[count],widths[width],&result) == 0 )
			{
				printf("FAIL a graph of %u nodes: %s\n",counts[count],SparkGpuGetErrorString(SparkGpuGetLastError()));
				return(1);
			}
			ChainPrint(flavour,"graph",counts[count],widths[width],&result);
			(void)SparkGpuGraphExecDestroy(exec);
			(void)SparkGpuGraphDestroy(graph);
		}
	return(0);
}
