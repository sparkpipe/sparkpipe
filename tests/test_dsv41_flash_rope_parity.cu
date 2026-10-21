#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stage_kernels.cuh"

#define HEADS 3u
#define WIDTH (2u * D41K_ROPE_PAIRS)
#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)

static const uint32_t Positions[] = { 0u, 1u, 2u, 7u, 127u, 128u, 4095u, 65535u, 65536u, 100000u, 131071u, 262143u, 327679u, 524287u,
	737279u, 999999u, 1048575u };
#define ROWS ((uint32_t)(sizeof(Positions) / sizeof(Positions[0])))

static __global__ void TurnKernel(const uint32_t *positions, const float *table, float *sine, float *cosine)
{
	const uint32_t row = blockIdx.x, pair = threadIdx.x;
	D41RopeTurn(positions[row], 0, table[pair], 0u, &sine[row * D41K_ROPE_PAIRS + pair], &cosine[row * D41K_ROPE_PAIRS + pair]);
}

static void Write(FILE *file, const void *device, size_t bytes)
{
	void *host = malloc(bytes);
	REQUIRE(host != 0);
	CUDA(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
	REQUIRE(fwrite(host, 1u, bytes, file) == bytes);
	free(host);
}

int main(int argc, char **argv)
{
	const size_t values = (size_t)ROWS * HEADS * WIDTH;
	uint16_t input[ROWS * HEADS * WIDTH];
	uint32_t *positions;
	float *table, *sine, *cosine;
	uint16_t *data;
	uint64_t state = 0x243F6A8885A308D3ull;
	FILE *file;
	REQUIRE(argc == 2);
	for ( size_t index = 0u; index < values; ++index )
	{
		state = state * 6364136223846793005ull + 1442695040888963407ull;
		const uint32_t exponent = 120u + (uint32_t)((state >> 40u) % 12u);
		input[index] = (uint16_t)(((state >> 63u) << 15u) | (exponent << 7u) | ((state >> 20u) & 0x7fu));
	}
	CUDA(cudaMalloc(&positions, sizeof(Positions)));
	CUDA(cudaMalloc(&table, 2u * D41K_ROPE_PAIRS * sizeof(float)));
	CUDA(cudaMalloc(&sine, (size_t)ROWS * D41K_ROPE_PAIRS * sizeof(float)));
	CUDA(cudaMalloc(&cosine, (size_t)ROWS * D41K_ROPE_PAIRS * sizeof(float)));
	CUDA(cudaMalloc(&data, values * sizeof(uint16_t)));
	CUDA(cudaMemcpy(positions, Positions, sizeof(Positions), cudaMemcpyHostToDevice));
	CUDA(D41LaunchFrequencies(table, 10000.0, 160000.0, 16.0, 65536.0, 32.0, 1.0));
	CUDA(cudaDeviceSynchronize());
	file = fopen(argv[1], "wb");
	REQUIRE(file != 0);
	REQUIRE(fwrite(Positions, 1u, sizeof(Positions), file) == sizeof(Positions));
	REQUIRE(fwrite(input, 1u, sizeof(input), file) == sizeof(input));
	Write(file, table, 2u * D41K_ROPE_PAIRS * sizeof(float));
	for ( uint32_t which = 0u; which < 2u; ++which )
	{
		const float *frequency = table + which * D41K_ROPE_PAIRS;
		TurnKernel<<<ROWS, D41K_ROPE_PAIRS>>>(positions, frequency, sine, cosine);
		CUDA(cudaGetLastError());
		Write(file, sine, (size_t)ROWS * D41K_ROPE_PAIRS * sizeof(float));
		Write(file, cosine, (size_t)ROWS * D41K_ROPE_PAIRS * sizeof(float));
		for ( uint32_t inverse = 0u; inverse < 2u; ++inverse )
		{
			CUDA(cudaMemcpy(data, input, sizeof(input), cudaMemcpyHostToDevice));
			D41RopeRowsKernel<<<dim3(ROWS, HEADS), D41K_ROPE_PAIRS>>>(data, HEADS, WIDTH, 0u, positions, 0, frequency, inverse, 0);
			CUDA(cudaGetLastError());
			Write(file, data, values * sizeof(uint16_t));
		}
	}
	REQUIRE(fclose(file) == 0);
	printf("rows %u heads %u pairs %u\n", ROWS, HEADS, D41K_ROPE_PAIRS);
	return 0;
}
