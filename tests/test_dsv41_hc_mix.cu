#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stage_kernels.cuh"

#define WIDTH 5120u
#define FLAT (D41K_HC * WIDTH)

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)

static uint64_t Mix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, 4u);
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static uint32_t RunCase(uint32_t rows, uint32_t perturb)
{
	std::vector<uint16_t> streams((size_t)rows * FLAT);
	std::vector<float> fn((size_t)D41K_HC_MIX * FLAT);
	for ( size_t index = 0u; index < streams.size(); ++index )
		streams[index] = Bf16((float)((int32_t)((Mix(index ^ 0x77u) >> 40u) & 0xfffu) - 2048) / 256.0f);
	for ( size_t index = 0u; index < fn.size(); ++index )
		fn[index] = (float)((int32_t)((Mix(index ^ 0x1234u) >> 36u) & 0xfffffu) - 524288) / 4194304.0f;
	const size_t partial_floats = (size_t)rows * D41K_HC_TASKS * D41K_HC_WARPS;
	uint16_t *streams_d;
	float *fn_d, *reference_d, *candidate_d;
	CUDA(cudaMalloc((void **)&streams_d, streams.size() * sizeof(uint16_t)));
	CUDA(cudaMalloc((void **)&fn_d, fn.size() * sizeof(float)));
	CUDA(cudaMalloc((void **)&reference_d, partial_floats * sizeof(float)));
	CUDA(cudaMalloc((void **)&candidate_d, partial_floats * sizeof(float)));
	CUDA(cudaMemcpy(streams_d, streams.data(), streams.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(fn_d, fn.data(), fn.size() * sizeof(float), cudaMemcpyHostToDevice));
	D41HcMixPartialKernel<<<dim3(D41K_HC_TASKS, rows), D41K_THREADS>>>(streams_d, fn_d, reference_d, WIDTH);
	CUDA(cudaGetLastError());
	if ( perturb != 0u )
	{
		const uint16_t bumped = (uint16_t)(streams[(size_t)(rows - 1u) * FLAT + 777u] ^ 0x0040u);
		CUDA(cudaMemcpy(streams_d + (size_t)(rows - 1u) * FLAT + 777u, &bumped, sizeof(bumped), cudaMemcpyHostToDevice));
	}
	D41HcMixRowsKernel<D41K_HC_MIX_ROWS><<<(rows + D41K_HC_MIX_ROWS - 1u) / D41K_HC_MIX_ROWS, D41K_THREADS>>>(streams_d, fn_d, candidate_d, WIDTH,
		rows);
	CUDA(cudaGetLastError());
	CUDA(cudaDeviceSynchronize());
	std::vector<float> reference(partial_floats), candidate(partial_floats);
	CUDA(cudaMemcpy(reference.data(), reference_d, partial_floats * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpy(candidate.data(), candidate_d, partial_floats * sizeof(float), cudaMemcpyDeviceToHost));
	size_t differing = 0u;
	for ( size_t index = 0u; index < partial_floats; ++index )
		differing += memcmp(&reference[index], &candidate[index], sizeof(float)) != 0 ? 1u : 0u;
	cudaFree(streams_d); cudaFree(fn_d); cudaFree(reference_d); cudaFree(candidate_d);
	printf("case rows %u%s: %zu of %zu partials differ\n", rows, perturb != 0u ? " (perturbed stream)" : "", differing, partial_floats);
	return differing != 0u ? 1u : 0u;
}

int main(void)
{
	const uint32_t cases[] = {1u, 3u, 32u, 37u, 1024u};
	uint32_t failures = 0u;
	for ( uint32_t rows : cases )
		failures += RunCase(rows, 0u);
	const uint32_t sensitive = RunCase(37u, 1u);
	if ( failures != 0u || sensitive == 0u )
	{
		printf("FAIL dsv41 hc mix: %u cases differ from the task-parallel mix, perturbed stream %s\n", failures, sensitive != 0u ? "detected" : "not detected");
		return 1;
	}
	printf("PASS dsv41 hc mix: row-blocked partials are the same bits as the task-parallel partials\n");
	return 0;
}
