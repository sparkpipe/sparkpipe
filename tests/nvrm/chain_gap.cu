#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"
#include "inference/kernels/dependent_launch.cuh"

#define GAP_CHAIN 136
#define GAP_ROUNDS 300
#define GAP_WARMUP 30

__global__ void GapStamp(unsigned long long *stamps, unsigned int index, unsigned long long hold_ns, unsigned int dependent)
{
	unsigned long long start, now;
	if ( dependent != 0u )
		LmDependentWait();
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
	if ( threadIdx.x == 0 )
		atomicMin(&stamps[2u * index], start);
	do
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
	while ( now - start < hold_ns );
	__syncthreads();
	if ( threadIdx.x == 0 )
		atomicMax(&stamps[2u * index + 1u], now);
}

static unsigned long long GapNow(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (unsigned long long)now.tv_sec * 1000000000ull + (unsigned long long)now.tv_nsec;
}

static int GapCompare(const void *a, const void *b)
{
	const double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

static double GapMedian(double *values, int count)
{
	qsort(values, (size_t)count, sizeof(values[0]), GapCompare);
	return values[count / 2];
}

int main(int argc, char **argv)
{
	static const unsigned long long holds[] = {0ull, 5000ull, 20000ull};
	static const unsigned int blocks[] = {1u, 170u};
	static unsigned long long host_stamps[2 * GAP_CHAIN], initial[2 * GAP_CHAIN];
	static double gaps[GAP_ROUNDS], spans[GAP_ROUNDS], walls[GAP_ROUNDS], submits[GAP_ROUNDS], tails[GAP_ROUNDS];
	const char *flavour = argc > 1 ? argv[1] : "this flavour";
	unsigned long long missing = 0ull, overlapped = 0ull, kernels = 0ull;
	SparkGpuStream stream = 0;
	unsigned long long *stamps = 0;
	for ( int index = 0; index < GAP_CHAIN; index++ )
	{
		initial[2 * index] = ~0ull;
		initial[2 * index + 1] = 0ull;
	}
	if ( SparkGpuStreamCreateWithFlags(&stream, SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS ||
		SparkGpuMalloc((void **)&stamps, sizeof(initial)) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL set up: %s\n", SparkGpuGetErrorString(SparkGpuGetLastError()));
		return 1;
	}
	for ( unsigned int dependent = 0u; dependent < 2u; dependent++ )
		for ( unsigned int shape = 0u; shape < sizeof(blocks) / sizeof(blocks[0]); shape++ )
			for ( unsigned int hold = 0u; hold < sizeof(holds) / sizeof(holds[0]); hold++ )
			{
				for ( int round = -GAP_WARMUP; round < GAP_ROUNDS; round++ )
				{
					unsigned long long started, submitted, finished;
					double gap_sum = 0.0;
					if ( SparkGpuMemcpy(stamps, initial, sizeof(initial), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
					{
						printf("FAIL reset stamps\n");
						return 1;
					}
					started = GapNow();
					for ( unsigned int index = 0u; index < GAP_CHAIN; index++ )
					{
						if ( dependent != 0u && index != 0u )
							(void)LM_LAUNCH_DEPENDENT((GapStamp), blocks[shape], 128, 0, stream, stamps, index, holds[hold], 1u);
						else
							LM_LAUNCH((GapStamp), blocks[shape], 128, 0, stream, stamps, index, holds[hold], 0u);
					}
					submitted = GapNow();
					if ( SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS || SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
					{
						printf("FAIL launch and synchronize: %s\n", SparkGpuGetErrorString(SparkGpuGetLastError()));
						return 1;
					}
					finished = GapNow();
					if ( SparkGpuMemcpy(host_stamps, stamps, sizeof(host_stamps), SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
					{
						printf("FAIL read stamps\n");
						return 1;
					}
					for ( int index = 0; index < GAP_CHAIN; index++ )
					{
						kernels++;
						if ( host_stamps[2 * index] == ~0ull || host_stamps[2 * index + 1] == 0ull )
							missing++;
						else if ( index != 0 && host_stamps[2 * index] < host_stamps[2 * index - 1] )
							overlapped++;
					}
					if ( round < 0 )
						continue;
					for ( int index = 1; index < GAP_CHAIN; index++ )
						gap_sum += (double)((long long)host_stamps[2 * index] - (long long)host_stamps[2 * index - 1]);
					gaps[round] = gap_sum / (GAP_CHAIN - 1) / 1e3;
					spans[round] = (double)(host_stamps[2 * GAP_CHAIN - 1] - host_stamps[0]) / 1e3;
					walls[round] = (double)(finished - started) / 1e3;
					submits[round] = (double)(submitted - started) / 1e3;
					tails[round] = walls[round] - submits[round];
				}
				printf("chain %s %s blocks=%u hold_us=%llu launches=%d: gap between kernels median %.2f us; GPU span %.1f us; host submit %.1f us; "
					"sync wait %.1f us; wall %.1f us\n",
					flavour, dependent != 0u ? "dependent" : "plain", blocks[shape], holds[hold] / 1000ull, GAP_CHAIN, GapMedian(gaps, GAP_ROUNDS),
					GapMedian(spans, GAP_ROUNDS), GapMedian(submits, GAP_ROUNDS), GapMedian(tails, GAP_ROUNDS), GapMedian(walls, GAP_ROUNDS));
			}
	if ( SparkGpuFree(stamps) != SPARK_GPU_SUCCESS || SparkGpuStreamDestroy(stream) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL tear down: %s\n", SparkGpuGetErrorString(SparkGpuGetLastError()));
		return 1;
	}
	if ( missing != 0ull || overlapped != 0ull )
	{
		printf("FAIL chain %s: of %llu kernels, %llu left no stamps and %llu started before the kernel ahead of them ended\n", flavour, kernels, missing,
			overlapped);
		return 1;
	}
	printf("PASS chain %s: all %llu kernels ran, each starting after the kernel ahead of it ended, in plain and dependent chains of %d launches\n",
		flavour, kernels, GAP_CHAIN);
	return 0;
}
