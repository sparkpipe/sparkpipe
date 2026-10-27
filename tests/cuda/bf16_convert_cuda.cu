#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "inference/kernels/kernel_math.cuh"

#define SAMPLES 8u

struct Counts
{
	unsigned long long library;
	unsigned long long hardware;
	unsigned long long truncated;
	unsigned int library_seen;
	unsigned int hardware_seen;
	uint32_t library_input[SAMPLES];
	uint32_t hardware_input[SAMPLES];
};

static __device__ __forceinline__ uint16_t Contract(uint32_t bits)
{
	if ( (bits & 0x7fffffffu) > 0x7f800000u )
		return 0x7fffu;
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

__global__ void Exhaust(Counts *counts)
{
	unsigned long long library = 0u, hardware = 0u, truncated = 0u;
	const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index <= UINT64_C(0xffffffff); index += stride )
	{
		const uint32_t bits = (uint32_t)index;
		const float value = __uint_as_float(bits);
		const uint16_t want = Contract(bits);
		if ( SparkMathFloatToBf16(value) != want )
		{
			const unsigned int slot = atomicAdd(&counts->library_seen, 1u);
			if ( slot < SAMPLES )
				counts->library_input[slot] = bits;
			library++;
		}
		if ( (uint16_t)(bits >> 16u) != want )
			truncated++;
		if ( __bfloat16_as_ushort(__float2bfloat16_rn(value)) != want )
		{
			const unsigned int slot = atomicAdd(&counts->hardware_seen, 1u);
			if ( slot < SAMPLES )
				counts->hardware_input[slot] = bits;
			hardware++;
		}
	}
	if ( library != 0u )
		atomicAdd(&counts->library, library);
	if ( hardware != 0u )
		atomicAdd(&counts->hardware, hardware);
	if ( truncated != 0u )
		atomicAdd(&counts->truncated, truncated);
}

int main(void)
{
	Counts *device = 0, host;
	if ( cudaMalloc(&device, sizeof(Counts)) != cudaSuccess || cudaMemset(device, 0, sizeof(Counts)) != cudaSuccess )
	{
		fprintf(stderr, "FAIL bf16_convert_cuda: no device memory\n");
		return 1;
	}
	Exhaust<<<4096, 256>>>(device);
	if ( cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(&host, device, sizeof(host), cudaMemcpyDeviceToHost) != cudaSuccess )
	{
		fprintf(stderr, "FAIL bf16_convert_cuda: the exhaustive kernel did not run\n");
		return 1;
	}
	printf("hardware cvt.rn.bf16.f32 differs from the contract on %llu of 4294967296 inputs", host.hardware);
	for ( unsigned int index = 0u; index < host.hardware_seen && index < SAMPLES; ++index )
		printf(" 0x%08x", host.hardware_input[index]);
	printf("\n");
	if ( host.truncated < UINT64_C(1) << 30u )
	{
		fprintf(stderr, "FAIL bf16_convert_cuda: truncation differs from the contract on only %llu inputs; the comparison does not see wrong conversions\n",
			host.truncated);
		return 1;
	}
	if ( host.library != 0u )
	{
		fprintf(stderr, "FAIL bf16_convert_cuda: SparkMathFloatToBf16 differs from round-to-nearest-even with every NaN as 0x7fff on %llu inputs, e.g.",
			host.library);
		for ( unsigned int index = 0u; index < host.library_seen && index < SAMPLES; ++index )
			fprintf(stderr, " 0x%08x", host.library_input[index]);
		fprintf(stderr, "\n");
		return 1;
	}
	printf("PASS bf16_convert_cuda: SparkMathFloatToBf16 rounds all 4294967296 float bit patterns to nearest even and maps every NaN to 0x7fff; truncation, the control, differs on %llu\n", host.truncated);
	return 0;
}
