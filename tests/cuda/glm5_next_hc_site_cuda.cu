#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "glm5_next_stage.cuh"

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)

static uint64_t Mix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

static float Uniform(uint64_t seed, uint64_t index)
{
	return (float)(Mix(seed * 0x100000001ull + index) & 0xffffffu) / 8388608.0f - 1.0f;
}

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, 4u);
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return (uint16_t)(bits >> 16u);
}

struct Outputs
{
	float *mixes, *pre, *post, *comb;
	uint16_t *collapsed, *snapshot;
};

static void Allocate(Outputs *o, uint32_t rows)
{
	CUDA(cudaMalloc(&o->mixes, (uint64_t)rows * G5N_HC_MIX * 4u));
	CUDA(cudaMalloc(&o->pre, (uint64_t)rows * G5N_HC * 4u));
	CUDA(cudaMalloc(&o->post, (uint64_t)rows * G5N_HC * 4u));
	CUDA(cudaMalloc(&o->comb, (uint64_t)rows * G5N_HC * G5N_HC * 4u));
	CUDA(cudaMalloc(&o->collapsed, (uint64_t)rows * G5N_HIDDEN * 2u));
	CUDA(cudaMalloc(&o->snapshot, (uint64_t)rows * G5N_HC_FLAT * 2u));
	CUDA(cudaMemset(o->mixes, 0xa5, (uint64_t)rows * G5N_HC_MIX * 4u));
	CUDA(cudaMemset(o->pre, 0xa5, (uint64_t)rows * G5N_HC * 4u));
	CUDA(cudaMemset(o->post, 0xa5, (uint64_t)rows * G5N_HC * 4u));
	CUDA(cudaMemset(o->comb, 0xa5, (uint64_t)rows * G5N_HC * G5N_HC * 4u));
	CUDA(cudaMemset(o->collapsed, 0xa5, (uint64_t)rows * G5N_HIDDEN * 2u));
	CUDA(cudaMemset(o->snapshot, 0xa5, (uint64_t)rows * G5N_HC_FLAT * 2u));
}

static void Release(Outputs *o)
{
	cudaFree(o->mixes); cudaFree(o->pre); cudaFree(o->post); cudaFree(o->comb); cudaFree(o->collapsed); cudaFree(o->snapshot);
}

static std::vector<uint8_t> Host(const void *device, uint64_t bytes)
{
	std::vector<uint8_t> host(bytes);
	CUDA(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
	return host;
}

static uint32_t Differ(const void *a, const void *b, uint64_t bytes)
{
	return Host(a, bytes) != Host(b, bytes) ? 1u : 0u;
}

template<uint32_t ROWS>
static void Run(const uint16_t *streams, const float *fn, const float *scale, const float *base, Outputs *o, uint32_t rows)
{
	CUDA(Glm5NextHcSiteLaunch<ROWS>(streams, fn, scale, base, o->mixes, o->pre, o->post, o->comb, o->collapsed, o->snapshot, rows, 0));
	CUDA(cudaDeviceSynchronize());
}

template<uint32_t ROWS>
static float Time(const uint16_t *streams, const float *fn, const float *scale, const float *base, Outputs *o, uint32_t rows)
{
	cudaEvent_t start, stop;
	float ms = 0.0f;
	CUDA(cudaEventCreate(&start));
	CUDA(cudaEventCreate(&stop));
	Run<ROWS>(streams, fn, scale, base, o, rows);
	CUDA(cudaEventRecord(start));
	for ( uint32_t i = 0u; i < 20u; i++ )
		CUDA(Glm5NextHcSiteLaunch<ROWS>(streams, fn, scale, base, o->mixes, o->pre, o->post, o->comb, o->collapsed, o->snapshot, rows, 0));
	CUDA(cudaEventRecord(stop));
	CUDA(cudaEventSynchronize(stop));
	CUDA(cudaEventElapsedTime(&ms, start, stop));
	return 1000.0f * ms / 20.0f;
}

int main(int argc, char **argv)
{
	static const uint32_t kRows[] = {1u, 2u, 3u, 4u, 5u, 7u, 8u, 63u, 130u, 2049u};
	const uint32_t max_rows = 2049u;
	std::vector<uint16_t> streams_host((uint64_t)max_rows * G5N_HC_FLAT);
	std::vector<float> fn_host((uint64_t)G5N_HC_MIX * G5N_HC_FLAT), base_host(G5N_HC_MIX), scale_host(3u);
	uint16_t *streams;
	float *fn, *base, *scale;
	uint32_t failures = 0u;
	for ( uint64_t i = 0; i < streams_host.size(); i++ )
		streams_host[i] = Bf16(4.0f * Uniform(1u, i));
	for ( uint64_t i = 0; i < fn_host.size(); i++ )
		fn_host[i] = 0.02f * Uniform(2u, i);
	for ( uint32_t i = 0; i < G5N_HC_MIX; i++ )
		base_host[i] = 0.5f * Uniform(3u, i);
	for ( uint32_t i = 0; i < 3u; i++ )
		scale_host[i] = 0.5f + 0.25f * Uniform(4u, i);
	CUDA(cudaMalloc(&streams, streams_host.size() * 2u));
	CUDA(cudaMalloc(&fn, fn_host.size() * 4u));
	CUDA(cudaMalloc(&base, base_host.size() * 4u));
	CUDA(cudaMalloc(&scale, scale_host.size() * 4u));
	CUDA(cudaMemcpy(streams, streams_host.data(), streams_host.size() * 2u, cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(fn, fn_host.data(), fn_host.size() * 4u, cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(base, base_host.data(), base_host.size() * 4u, cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(scale, scale_host.data(), scale_host.size() * 4u, cudaMemcpyHostToDevice));
	if ( argc > 2 && strcmp(argv[1], "--dump") == 0 )
	{
		Outputs single;
		FILE *out = fopen(argv[2], "wb");
		REQUIRE(out != 0);
		Allocate(&single, max_rows);
		Run<1u>(streams, fn, scale, base, &single, max_rows);
		std::vector<uint8_t> parts[6] = {Host(single.mixes, (uint64_t)max_rows * G5N_HC_MIX * 4u), Host(single.pre, (uint64_t)max_rows * G5N_HC * 4u),
			Host(single.post, (uint64_t)max_rows * G5N_HC * 4u), Host(single.comb, (uint64_t)max_rows * G5N_HC * G5N_HC * 4u),
			Host(single.collapsed, (uint64_t)max_rows * G5N_HIDDEN * 2u), Host(single.snapshot, (uint64_t)max_rows * G5N_HC_FLAT * 2u)};
		for ( auto &part : parts )
			REQUIRE(fwrite(part.data(), 1u, part.size(), out) == part.size());
		fclose(out);
		printf("dumped one-row-cluster outputs for %u rows to %s\n", max_rows, argv[2]);
		return 0;
	}
	for ( uint32_t rows : kRows )
	{
		Outputs single, grouped;
		Allocate(&single, rows);
		Allocate(&grouped, rows);
		Run<1u>(streams, fn, scale, base, &single, rows);
		Run<G5N_HC_ROWS>(streams, fn, scale, base, &grouped, rows);
		const uint32_t differ = Differ(single.mixes, grouped.mixes, (uint64_t)rows * G5N_HC_MIX * 4u) +
			Differ(single.pre, grouped.pre, (uint64_t)rows * G5N_HC * 4u) + Differ(single.post, grouped.post, (uint64_t)rows * G5N_HC * 4u) +
			Differ(single.comb, grouped.comb, (uint64_t)rows * G5N_HC * G5N_HC * 4u) +
			Differ(single.collapsed, grouped.collapsed, (uint64_t)rows * G5N_HIDDEN * 2u) +
			Differ(single.snapshot, grouped.snapshot, (uint64_t)rows * G5N_HC_FLAT * 2u);
		printf("rows %4u: %u of 6 outputs differ between %u-row and one-row clusters\n", rows, differ, G5N_HC_ROWS);
		failures += differ;
		Release(&single);
		Release(&grouped);
	}
	{
		Outputs timing;
		Allocate(&timing, 2048u);
		const float one = Time<1u>(streams, fn, scale, base, &timing, 2048u), grouped = Time<G5N_HC_ROWS>(streams, fn, scale, base, &timing, 2048u);
		printf("2048 rows: one-row clusters %.1f us, %u-row clusters %.1f us\n", one, G5N_HC_ROWS, grouped);
		Release(&timing);
	}
	if ( failures != 0u )
	{
		printf("FAIL glm5_next hc site: %u outputs differ\n", failures);
		return 1;
	}
	printf("PASS glm5_next hc site: %u-row clusters equal one-row clusters bit for bit at every row count\n", G5N_HC_ROWS);
	return 0;
}
