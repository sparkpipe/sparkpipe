#include "tests/host_cuda/lm_host_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "inference/kernels/dtype.cuh"
#define __CUDACC__ 1
#include "inference/kernels/norm.cuh"
#include "inference/kernels/topk.cuh"
#include "runtime/launch.h"
#include "tests/host_cuda/lm_host_threads.cuh"
LmHostDim3 blockDim,gridDim;
#include "inference/kernels/topk_exact.cuh"

#define HOST_THREADS 64u
#define HOST_MAX_ROWS 3u
#define HOST_MAX_N 8192u
#define HOST_MAX_K 600u
#define HOST_REPEATS 4u

static uint32_t host_state = 13579u;

static float HostUniform(void)
{
	host_state = host_state * 1664525u + 1013904223u;
	return (float)(host_state >> 8u) / 16777216.0f;
}

static uint32_t HostKey(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return bits ^ ((bits >> 31u) != 0u ? 0xffffffffu : 0x80000000u);
}

static void HostReference(const float *scores, uint32_t n, uint32_t k, uint32_t *out)
{
	static uint8_t taken[HOST_MAX_N];
	uint32_t pick,index,best,emitted;
	memset(taken,0,sizeof(taken));
	for (pick = 0u; pick < k && pick < n; ++pick)
	{
		best = UINT32_MAX;
		for (index = 0u; index < n; ++index)
			if ( taken[index] == 0u && (best == UINT32_MAX || HostKey(scores[index]) > HostKey(scores[best])) )
				best = index;
		if ( best >= HOST_MAX_N )
			break;
		taken[best] = 1u;
	}
	for (emitted = 0u,index = 0u; index < n; ++index)
		if ( taken[index] != 0u )
			out[emitted++] = index;
	for (; emitted < k; ++emitted)
		out[emitted] = 0xffffffffu;
}

static void HostCheck(const char *name, const float *scores, uint32_t rows, uint32_t n, uint32_t k)
{
	static uint32_t got[HOST_MAX_ROWS * HOST_MAX_K],first[HOST_MAX_ROWS * HOST_MAX_K],want[HOST_MAX_K];
	uint32_t repeat,row,index;
	for (repeat = 0u; repeat < HOST_REPEATS; ++repeat)
	{
		memset(got,0xab,sizeof(got));
		LM_LAUNCH((LmTopkExactKernel<HOST_THREADS>),dim3(rows),HOST_THREADS,0,0,scores,n,k,got);
		if ( repeat == 0u )
			memcpy(first,got,sizeof(got));
		else if ( memcmp(first,got,(size_t)rows * k * sizeof(uint32_t)) != 0 )
		{
			fprintf(stderr,"FAIL %s: repeat %u selected a different set or order\n",name,repeat);
			exit(1);
		}
	}
	for (row = 0u; row < rows; ++row)
	{
		HostReference(scores + (uint64_t)row * n,n,k,want);
		for (index = 0u; index < k; ++index)
			if ( first[row * k + index] != want[index] )
			{
				fprintf(stderr,"FAIL %s: row %u slot %u holds %u, exact top-%u in index order holds %u\n",name,row,index,first[row * k + index],k,want[index]);
				exit(1);
			}
	}
	printf("PASS %s rows=%u n=%u k=%u\n",name,rows,n,k);
}

static void HostHighBucketLast(void)
{
	static float scores[8];
	uint32_t index;
	for (index = 0u; index < 7u; ++index)
		scores[index] = 1.0f + 0.01f * (float)index;
	scores[7] = 5.0f;
	HostCheck("higher bucket after a full threshold bucket",scores,1u,8u,2u);
}

static void HostQuantizedTies(void)
{
	static float scores[HOST_MAX_ROWS * 1000u];
	uint32_t index;
	for (index = 0u; index < HOST_MAX_ROWS * 1000u; ++index)
		scores[index] = floorf(HostUniform() * 20.0f) / 20.0f - 0.5f;
	HostCheck("ties at the threshold keep the lowest indices",scores,HOST_MAX_ROWS,1000u,100u);
}

static void HostPoolsBeyondContext(void)
{
	static float scores[600];
	uint32_t index;
	for (index = 0u; index < 600u; ++index)
		scores[index] = index < 550u ? HostUniform() * 4.0f - 2.0f : -INFINITY;
	HostCheck("pools past the context never displace valid pools",scores,1u,600u,512u);
}

static void HostShortRow(void)
{
	static float scores[5] = {0.5f,-1.0f,2.0f,0.0f,1.0f};
	HostCheck("fewer candidates than k",scores,1u,5u,8u);
}

static void HostLongContext(void)
{
	static float scores[HOST_MAX_N];
	uint32_t index;
	for (index = 0u; index < HOST_MAX_N; ++index)
		scores[index] = HostUniform() * 8.0f - 4.0f;
	HostCheck("32K-token context of 4-token pools",scores,1u,HOST_MAX_N,512u);
}

int main(void)
{
	HostHighBucketLast();
	HostQuantizedTies();
	HostPoolsBeyondContext();
	HostShortRow();
	HostLongContext();
	return 0;
}
