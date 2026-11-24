#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "inference/kernels/diff_attn.cuh"
#include "inference/kernels/poly_norm.cuh"
#include "runtime/launch.h"

#define LAUNCH_BENCH_BF16_ELEMENTS 2048u
#define LAUNCH_BENCH_WORDS 16u

__global__ void LaunchBenchEmptyKernel(uint32_t *sink, uint32_t value)
{
	if ( value == UINT32_MAX && threadIdx.x == 0u )
		sink[0] = value;
}

static double LaunchBenchSeconds(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

typedef struct LaunchBenchBuffers
{
	uint16_t *bf16;
	float *params;
	uint32_t *rows;
	uint32_t *sink;
}
LaunchBenchBuffers;

static int LaunchBenchIssue(int which, const LaunchBenchBuffers *b, cudaStream_t stream)
{
	switch ( which )
	{
	case 0:
		LM_LAUNCH((LaunchBenchEmptyKernel), 1u, 32u, 0u, stream, b->sink, 0u);
		return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
	case 1:
		return LM_LAUNCH_DEPENDENT((LaunchBenchEmptyKernel), 1u, 32u, 0u, stream, b->sink, 0u) == cudaSuccess ? 0 : 1;
	case 2:
		return LmDifferentialGateLaunch(b->bf16, b->bf16 + 64, 1u, b->bf16 + 128, b->bf16 + 192, 1u, 1u, 1u, 1u, stream) == cudaSuccess ? 0 : 1;
	default:
		return LmPolyNormMulLaunch<false>(b->bf16, b->params, b->rows, b->bf16 + 256, 1u, 1u, 0.0f, 0.0f, 1.0f, 1e-6f, stream) == cudaSuccess ? 0 : 1;
	}
}

static const char *const LaunchBenchNames[] = {"empty", "empty_dependent", "differential_gate", "poly_norm"};

int main(int argc, char **argv)
{
	const uint32_t launches = argc > 1 ? (uint32_t)strtoul(argv[1], 0, 10) : 20000u;
	const uint32_t repeats = argc > 2 ? (uint32_t)strtoul(argv[2], 0, 10) : 9u;
	LaunchBenchBuffers buffers;
	cudaStream_t stream;
	int which;

	if ( launches == 0u || repeats == 0u )
	{
		fprintf(stderr, "usage: %s [launches] [repeats]\n", argv[0]);
		return 2;
	}
	if ( cudaMalloc((void **)&buffers.bf16, LAUNCH_BENCH_BF16_ELEMENTS * sizeof(uint16_t)) != cudaSuccess
		|| cudaMalloc((void **)&buffers.params, LAUNCH_BENCH_WORDS * sizeof(float)) != cudaSuccess
		|| cudaMalloc((void **)&buffers.rows, LAUNCH_BENCH_WORDS * sizeof(uint32_t)) != cudaSuccess
		|| cudaMalloc((void **)&buffers.sink, LAUNCH_BENCH_WORDS * sizeof(uint32_t)) != cudaSuccess
		|| cudaMemset(buffers.bf16, 0, LAUNCH_BENCH_BF16_ELEMENTS * sizeof(uint16_t)) != cudaSuccess
		|| cudaMemset(buffers.params, 0, LAUNCH_BENCH_WORDS * sizeof(float)) != cudaSuccess
		|| cudaMemset(buffers.rows, 0, LAUNCH_BENCH_WORDS * sizeof(uint32_t)) != cudaSuccess
		|| cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess )
	{
		fprintf(stderr, "launch bench: device setup failed: %s\n", cudaGetErrorString(cudaGetLastError()));
		return 1;
	}
	for ( which = 0; which < 4; which++ )
	{
		std::vector<double> issue, total;
		uint32_t repeat, index;
		for ( index = 0u; index < 1000u; index++ )
			if ( LaunchBenchIssue(which, &buffers, stream) != 0 )
				break;
		if ( index != 1000u || cudaStreamSynchronize(stream) != cudaSuccess )
		{
			fprintf(stderr, "launch bench: %s warm-up failed: %s\n", LaunchBenchNames[which], cudaGetErrorString(cudaGetLastError()));
			return 1;
		}
		for ( repeat = 0u; repeat < repeats; repeat++ )
		{
			const double start = LaunchBenchSeconds();
			double issued;
			for ( index = 0u; index < launches; index++ )
				if ( LaunchBenchIssue(which, &buffers, stream) != 0 )
					break;
			issued = LaunchBenchSeconds();
			if ( index != launches || cudaStreamSynchronize(stream) != cudaSuccess )
			{
				fprintf(stderr, "launch bench: %s failed: %s\n", LaunchBenchNames[which], cudaGetErrorString(cudaGetLastError()));
				return 1;
			}
			issue.push_back((issued - start) * 1e9 / launches);
			total.push_back((LaunchBenchSeconds() - start) * 1e9 / launches);
		}
		std::sort(issue.begin(), issue.end());
		std::sort(total.begin(), total.end());
		printf("%-18s launches %u repeats %u issue_ns median %.1f min %.1f total_ns median %.1f min %.1f\n", LaunchBenchNames[which], launches, repeats,
			issue[issue.size() / 2u], issue[0], total[total.size() / 2u], total[0]);
	}
	cudaStreamDestroy(stream);
	cudaFree(buffers.sink);
	cudaFree(buffers.rows);
	cudaFree(buffers.params);
	cudaFree(buffers.bf16);
	return 0;
}
