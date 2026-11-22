#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>

#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

#define LATENCY_ROUNDS 2000
#define LATENCY_WARMUP 100

__global__ void LatencyHold(unsigned long long hold_ns)
{
	unsigned long long start,now;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	do
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
	while ( now - start < hold_ns );
}

static unsigned long long LatencyNow(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((unsigned long long)now.tv_sec * 1000000000ull + (unsigned long long)now.tv_nsec);
}

static double LatencyCpuUs(void)
{
	struct rusage usage;
	getrusage(RUSAGE_SELF,&usage);
	return((double)(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1e6 + (double)(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec));
}

static int LatencyCompare(const void *a, const void *b)
{
	const long long x = *(const long long *)a,y = *(const long long *)b;
	return(x < y ? -1 : x > y);
}

int main(int argc, char **argv)
{
	static const unsigned long long holds[] = {0ull, 20000ull, 90000ull, 150000ull, 400000ull, 2000000ull};
	static long long values[LATENCY_ROUNDS];
	const char *flavour = argc > 1 ? argv[1] : "this flavour";
	SparkGpuStream stream = 0;
	unsigned int hold;
	if ( SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL create a stream: %s\n",SparkGpuGetErrorString(SparkGpuGetLastError()));
		return(1);
	}
	for (hold = 0u; hold < sizeof(holds) / sizeof(holds[0]); hold++)
	{
		const int rounds = holds[hold] >= 1000000ull ? LATENCY_ROUNDS / 10 : LATENCY_ROUNDS;
		double sum = 0.0,cpu = 0.0;
		int index;
		for (index = -LATENCY_WARMUP; index < rounds; index++)
		{
			if ( index == 0 )
				cpu = LatencyCpuUs();
			const unsigned long long start = LatencyNow();
			LM_LAUNCH((LatencyHold),1,1,0,stream,holds[hold]);
			if ( SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS || SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
			{
				printf("FAIL launch and synchronize: %s\n",SparkGpuGetErrorString(SparkGpuGetLastError()));
				return(1);
			}
			if ( index >= 0 )
			{
				values[index] = (long long)(LatencyNow() - start - holds[hold]);
				sum += (double)values[index];
			}
		}
		cpu = LatencyCpuUs() - cpu;
		qsort(values,(size_t)rounds,sizeof(values[0]),LatencyCompare);
		printf("latency %s hold %llu us: launch to synchronize past the hold median %.2f us p90 %.2f us mean %.2f us; process CPU %.1f us per round\n",flavour,
			holds[hold] / 1000ull,values[rounds / 2] / 1000.0,values[rounds * 9 / 10] / 1000.0,sum / rounds / 1000.0,cpu / rounds);
	}
	return(SparkGpuStreamDestroy(stream) == SPARK_GPU_SUCCESS ? 0 : 1);
}
