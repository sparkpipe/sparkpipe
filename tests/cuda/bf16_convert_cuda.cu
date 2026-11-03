#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "inference/kernels/dtype.cuh"
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

static __device__ __forceinline__ bool Same(float got, float want)
{
	return(isnan(got) ? isnan(want) : __float_as_uint(got) == __float_as_uint(want));
}

__global__ void Decode(uint32_t *bad)
{
	const uint32_t code = threadIdx.x, magnitude = code & 0x7fu;
	const float e4m3 = LmE4m3ToFloat((uint8_t)code);
	if ( !Same(SparkMathE4m3ToFloat(code), e4m3) )
		atomicAdd(&bad[0], 1u);
	if ( !Same(SparkMathUe4m3ToFloat(code), magnitude == 0x7fu ? LM_E4M3_MAX : fabsf(LmE4m3ToFloat((uint8_t)magnitude))) )
		atomicAdd(&bad[1], 1u);
	if ( !Same(SparkMathUe8m0ToFloat(code), code == 0xffu ? __uint_as_float(0x7fffffffu) : ldexpf(1.0f, (int)code - 127)) )
		atomicAdd(&bad[2], 1u);
	if ( code < 16u && !Same(SparkMathE2m1ToFloat(code), LmE2m1PairToFloat((uint8_t)(code | (code << 4u))).x) )
		atomicAdd(&bad[3], 1u);
}

int main(void)
{
	Counts *device = 0, host;
	uint32_t *decode = 0, decode_bad[4];
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
	if ( cudaMalloc(&decode, sizeof(decode_bad)) != cudaSuccess || cudaMemset(decode, 0, sizeof(decode_bad)) != cudaSuccess )
	{
		fprintf(stderr, "FAIL bf16_convert_cuda: no device memory for the decoders\n");
		return 1;
	}
	Decode<<<1, 256>>>(decode);
	if ( cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(decode_bad, decode, sizeof(decode_bad), cudaMemcpyDeviceToHost) != cudaSuccess )
	{
		fprintf(stderr, "FAIL bf16_convert_cuda: the decoder kernel did not run\n");
		return 1;
	}
	if ( decode_bad[0] != 0u || decode_bad[1] != 0u || decode_bad[2] != 0u || decode_bad[3] != 0u )
	{
		fprintf(stderr, "FAIL bf16_convert_cuda: the shared decoders differ from the hardware conversions: E4M3 %u, UE4M3 %u, UE8M0 %u of 256 codes, E2M1 %u of 16\n",
			decode_bad[0], decode_bad[1], decode_bad[2], decode_bad[3]);
		return 1;
	}
	printf("PASS bf16_convert_cuda: SparkMathE4m3ToFloat, SparkMathUe4m3ToFloat, SparkMathUe8m0ToFloat and SparkMathE2m1ToFloat equal cvt.rn.f16x2.e4m3x2 and cvt.rn.f16x2.e2m1x2 on every code\n");
	return 0;
}
