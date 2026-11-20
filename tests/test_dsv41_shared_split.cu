#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "model-families/dsv41_flash/source/dsv41_flash_stage_kernels.cuh"

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)

#define RANKS 16u
#define ROWS 9u
#define WIDTH 768u
#define TOPK 6u
#define LOCAL 24u
#define PACKED_ROWS (ROWS * TOPK)

static uint32_t Next(uint64_t *state)
{
	*state = *state * 6364136223846793005ull + 1442695040888963407ull;
	return (uint32_t)(*state >> 33u);
}

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static float Float(uint16_t value)
{
	const uint32_t bits = (uint32_t)value << 16u;
	float out;
	memcpy(&out, &bits, sizeof(out));
	return out;
}

static float Uniform(uint64_t *state)
{
	return ((float)(Next(state) & 0xffffffu) / 16777216.0f - 0.5f) * 4.0f;
}

static void CheckParts(uint32_t parts, uint32_t granules, uint32_t granule)
{
	uint32_t next = 0u;
	for ( uint32_t part = 0u; part < parts; ++part )
	{
		uint32_t first, width;
		D41SharedPart(part, parts, granules, granule, &first, &width);
		REQUIRE(first == next && width >= granule && width % granule == 0u);
		REQUIRE(width <= (granules + parts - 1u) / parts * granule);
		next = first + width;
	}
	REQUIRE(next == granules * granule);
}

typedef struct RankInputs
{
	std::vector<uint32_t> expert, packed_row;
	std::vector<uint16_t> packed;
	std::vector<float> shared;
} RankInputs;

static void MakeRank(uint64_t *seed, RankInputs *in)
{
	in->expert.resize(ROWS * TOPK);
	in->packed_row.resize(ROWS * TOPK);
	in->packed.resize((uint64_t)PACKED_ROWS * WIDTH);
	in->shared.resize((uint64_t)ROWS * WIDTH);
	for ( uint32_t route = 0u; route < ROWS * TOPK; ++route )
	{
		in->expert[route] = Next(seed) % (LOCAL * 3u);
		in->packed_row[route] = (route * 7u + 3u) % PACKED_ROWS;
	}
	for ( uint64_t index = 0u; index < in->packed.size(); ++index )
		in->packed[index] = Bf16(Uniform(seed));
	for ( uint64_t index = 0u; index < in->shared.size(); ++index )
		in->shared[index] = Uniform(seed) * 0.03125f;
}

static void Partials(const RankInputs *in, uint32_t row_first, uint32_t rows, float *device_partial, uint32_t *device_expert,
	uint32_t *device_packed_row, uint16_t *device_packed, float *device_shared)
{
	CUDA(cudaMemcpy(device_expert, in->expert.data() + (uint64_t)row_first * TOPK, (uint64_t)rows * TOPK * sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(device_packed_row, in->packed_row.data() + (uint64_t)row_first * TOPK, (uint64_t)rows * TOPK * sizeof(uint32_t),
		cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(device_packed, in->packed.data(), in->packed.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(device_shared, in->shared.data() + (uint64_t)row_first * WIDTH, (uint64_t)rows * WIDTH * sizeof(float), cudaMemcpyHostToDevice));
	D41MoeFinalizeF32Kernel<<<dim3((WIDTH + 255u) / 256u, rows), 256u>>>(device_packed, device_packed_row, device_expert, device_shared,
		device_partial, TOPK, LOCAL, WIDTH);
	CUDA(cudaGetLastError());
}

static std::vector<uint16_t> Combine(const std::vector<RankInputs> &ranks, uint32_t row_first, uint32_t rows)
{
	uint32_t *expert, *packed_row;
	uint16_t *packed, *out;
	float *shared, *gathered;
	std::vector<uint16_t> result((uint64_t)rows * WIDTH);
	CUDA(cudaMalloc(&expert, (uint64_t)ROWS * TOPK * sizeof(uint32_t)));
	CUDA(cudaMalloc(&packed_row, (uint64_t)ROWS * TOPK * sizeof(uint32_t)));
	CUDA(cudaMalloc(&packed, (uint64_t)PACKED_ROWS * WIDTH * sizeof(uint16_t)));
	CUDA(cudaMalloc(&shared, (uint64_t)ROWS * WIDTH * sizeof(float)));
	CUDA(cudaMalloc(&gathered, (uint64_t)RANKS * ROWS * WIDTH * sizeof(float)));
	CUDA(cudaMalloc(&out, (uint64_t)ROWS * WIDTH * sizeof(uint16_t)));
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
		Partials(&ranks[rank], row_first, rows, gathered + (uint64_t)rank * rows * WIDTH, expert, packed_row, packed, shared);
	D41SumSourcesF32Kernel<<<64u, 256u>>>(gathered, out, (uint64_t)rows * WIDTH, RANKS);
	CUDA(cudaGetLastError());
	CUDA(cudaMemcpy(result.data(), out, result.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	CUDA(cudaFree(expert));
	CUDA(cudaFree(packed_row));
	CUDA(cudaFree(packed));
	CUDA(cudaFree(shared));
	CUDA(cudaFree(gathered));
	CUDA(cudaFree(out));
	return result;
}

static uint16_t HostElement(const std::vector<RankInputs> &ranks, uint32_t row, uint32_t element)
{
	float sum = 0.0f;
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
	{
		const RankInputs *in = &ranks[rank];
		float total = 0.0f;
		for ( uint32_t k = 0u; k < TOPK; ++k )
		{
			const uint32_t route = row * TOPK + k;
			if ( in->expert[route] < LOCAL )
				total += Float(in->packed[(uint64_t)in->packed_row[route] * WIDTH + element]);
		}
		sum += total + in->shared[(uint64_t)row * WIDTH + element];
	}
	return Bf16(sum);
}

int main(void)
{
	uint64_t seed = 0x5eed5a1ad5eedull;
	std::vector<RankInputs> ranks(RANKS);
	CheckParts(16u, 18u, 128u);
	CheckParts(4u, 18u, 128u);
	CheckParts(8u, 18u, 128u);
	CheckParts(1u, 18u, 128u);
	{
		uint32_t first, width;
		D41SharedPart(0u, 16u, 18u, 128u, &first, &width);
		REQUIRE(first == 0u && width == 256u);
		D41SharedPart(2u, 16u, 18u, 128u, &first, &width);
		REQUIRE(first == 512u && width == 128u);
		D41SharedPart(15u, 16u, 18u, 128u, &first, &width);
		REQUIRE(first == 2176u && width == 128u);
	}
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
		MakeRank(&seed, &ranks[rank]);
	const std::vector<uint16_t> whole = Combine(ranks, 0u, ROWS);
	uint32_t local_routes = 0u;
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
		for ( uint32_t route = 0u; route < ROWS * TOPK; ++route )
			local_routes += ranks[rank].expert[route] < LOCAL ? 1u : 0u;
	REQUIRE(local_routes > 0u && local_routes < RANKS * ROWS * TOPK);
	for ( uint32_t row = 0u; row < ROWS; ++row )
		for ( uint32_t element = 0u; element < WIDTH; ++element )
			REQUIRE(whole[(uint64_t)row * WIDTH + element] == HostElement(ranks, row, element));
	for ( uint32_t row = 0u; row < ROWS; ++row )
	{
		const std::vector<uint16_t> single = Combine(ranks, row, 1u);
		REQUIRE(memcmp(single.data(), whole.data() + (uint64_t)row * WIDTH, (uint64_t)WIDTH * sizeof(uint16_t)) == 0);
	}
	printf("PASS dsv41 shared split: the shared expert's granules split over 16, 8, 4 and 1 ranks cover it once, each rank's FP32 partial (local "
		"routes plus its shared slice) summed in rank order rounds once to the same bits as the host, and a one-row step equals its row of a "
		"nine-row step\n");
	return 0;
}
