#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/norm.cuh"

#define PROBE_THREADS 256u
#define PROBE_TOP_K 8u
#define PROBE_REPEATS 50u

static int probe_failures;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

static uint64_t ProbeMix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	return z ^ (z >> 31u);
}

__global__ void ProbeFillKernel(uint16_t *values, uint64_t count, uint64_t salt)
{
	for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		uint64_t z = (index ^ salt) * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
		z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
		z ^= z >> 31u;
		values[index] = (uint16_t)(0x3800u + (z & 0x7ffu) + ((z >> 20u) & 1u) * 0x8000u);
	}
}

static void ProbeCase(uint32_t tokens, uint32_t dimension)
{
	const uint64_t packed = (uint64_t)tokens * PROBE_TOP_K;
	std::vector<uint32_t> route(packed);
	std::vector<float> weight(packed);
	std::vector<uint16_t> two_pass((uint64_t)tokens * dimension), fused((uint64_t)tokens * dimension);
	uint16_t *expert_out, *addend, *hidden, *fused_out;
	uint32_t *route_device;
	float *weight_device, two_ms, fused_ms;
	cudaEvent_t begin, end;
	for (uint64_t index = 0u; index < packed; index++)
		route[index] = (uint32_t)index;
	for (uint64_t index = packed; index > 1u; index--)
	{
		const uint64_t other = ProbeMix(index * 31u + tokens) % index;
		const uint32_t keep = route[index - 1u];
		route[index - 1u] = route[other];
		route[other] = keep;
	}
	for (uint64_t index = 0u; index < packed; index++)
		weight[index] = (float)(ProbeMix(index ^ 0x5151u) % 100000u) / 100000.0f * 0.4f;
	PROBE_CUDA(cudaMalloc((void **)&expert_out, packed * dimension * 2u));
	PROBE_CUDA(cudaMalloc((void **)&addend, (uint64_t)tokens * dimension * 2u));
	PROBE_CUDA(cudaMalloc((void **)&hidden, (uint64_t)tokens * dimension * 2u));
	PROBE_CUDA(cudaMalloc((void **)&fused_out, (uint64_t)tokens * dimension * 2u));
	PROBE_CUDA(cudaMalloc((void **)&route_device, packed * 4u));
	PROBE_CUDA(cudaMalloc((void **)&weight_device, packed * 4u));
	PROBE_CUDA(cudaMemcpy(route_device, route.data(), packed * 4u, cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(weight_device, weight.data(), packed * 4u, cudaMemcpyHostToDevice));
	ProbeFillKernel<<<512, 256>>>(expert_out, packed * dimension, tokens * 7u + dimension);
	ProbeFillKernel<<<512, 256>>>(addend, (uint64_t)tokens * dimension, 1ull << 40u);
	PROBE_CUDA(cudaMemset(fused_out, 0xff, (uint64_t)tokens * dimension * 2u));
	const dim3 scalar((dimension + PROBE_THREADS - 1u) / PROBE_THREADS, LmLaunchGridRows(tokens));
	const dim3 vector((dimension / LM_MOE_FINALIZE_ADD_ELEMENTS + PROBE_THREADS - 1u) / PROBE_THREADS, LmLaunchGridRows(tokens));
	auto two = [&]() {
		LmMoeFinalizeKernel<PROBE_THREADS><<<scalar, PROBE_THREADS>>>(expert_out, route_device, weight_device, hidden, tokens, PROBE_TOP_K, dimension);
		LmAddRowsKernel<PROBE_THREADS><<<scalar, PROBE_THREADS>>>(hidden, addend, hidden, tokens, dimension);
	};
	auto one = [&]() {
		LmMoeFinalizeAddKernel<PROBE_THREADS><<<vector, PROBE_THREADS>>>(expert_out, route_device, weight_device, addend, fused_out, tokens, PROBE_TOP_K, dimension);
	};
	two();
	one();
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(two_pass.data(), hidden, two_pass.size() * 2u, cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(fused.data(), fused_out, fused.size() * 2u, cudaMemcpyDeviceToHost));
	if ( memcmp(two_pass.data(), fused.data(), two_pass.size() * 2u) != 0 )
	{
		uint64_t index = 0u;
		while ( two_pass[index] == fused[index] )
			index++;
		printf("FAIL %u tokens x %u: the fused combine differs from finalize then add at %llu (%04x vs %04x)\n", tokens, dimension, (unsigned long long)index, fused[index],
			two_pass[index]);
		probe_failures++;
	}
	PROBE_CUDA(cudaEventCreate(&begin));
	PROBE_CUDA(cudaEventCreate(&end));
	PROBE_CUDA(cudaEventRecord(begin));
	for (uint32_t repeat = 0u; repeat < PROBE_REPEATS; repeat++)
		two();
	PROBE_CUDA(cudaEventRecord(end));
	PROBE_CUDA(cudaEventSynchronize(end));
	PROBE_CUDA(cudaEventElapsedTime(&two_ms, begin, end));
	PROBE_CUDA(cudaEventRecord(begin));
	for (uint32_t repeat = 0u; repeat < PROBE_REPEATS; repeat++)
		one();
	PROBE_CUDA(cudaEventRecord(end));
	PROBE_CUDA(cudaEventSynchronize(end));
	PROBE_CUDA(cudaEventElapsedTime(&fused_ms, begin, end));
	printf("%s %u tokens x %u, top %u: bit-equal to finalize then add, us two kernels %.1f fused %.1f\n", probe_failures == 0 ? "PASS" : "FAIL", tokens, dimension,
		PROBE_TOP_K, two_ms * 1000.0f / PROBE_REPEATS, fused_ms * 1000.0f / PROBE_REPEATS);
	cudaEventDestroy(begin);
	cudaEventDestroy(end);
	cudaFree(expert_out); cudaFree(addend); cudaFree(hidden); cudaFree(fused_out); cudaFree(route_device); cudaFree(weight_device);
}

int main(void)
{
	ProbeCase(1u, 6144u);
	ProbeCase(7u, 7168u);
	ProbeCase(300u, 6144u);
	ProbeCase(2048u, 6144u);
	ProbeCase(4096u, 7168u);
	ProbeCase(70000u, 64u);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n", probe_failures);
		return 1;
	}
	printf("PASS the fused expert combine adds the shared expert in the same pass and is bit-equal to finalize then add\n");
	return 0;
}
