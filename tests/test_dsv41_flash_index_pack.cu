#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stage_kernels.cuh"

#define INDEX_HEADS 32u
#define INDEX_DIM 128u
#define THREADS 256u

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)

static uint16_t Value(uint32_t rank, uint32_t row, uint32_t element, uint32_t salt)
{
	uint64_t x = ((uint64_t)rank << 40u ^ (uint64_t)row << 20u ^ element) * 0x9E3779B97F4A7C15ull ^ salt;
	x ^= x >> 31u;
	return (uint16_t)(x | 1u);
}

static void RoundTrip(uint32_t degree, uint32_t rows)
{
	const uint32_t heads = INDEX_HEADS / degree, query_width = heads * INDEX_DIM, pack = D41K_INDEX_PACK(heads, INDEX_DIM);
	const uint64_t rank_stride = (uint64_t)rows * pack;
	std::vector<uint16_t> gathered((uint64_t)degree * rank_stride, 0u), query((uint64_t)rows * INDEX_HEADS * INDEX_DIM), weight((uint64_t)rows * INDEX_HEADS);
	uint16_t *d_query, *d_weight, *d_pack, *d_gathered, *d_query_full, *d_weight_full;
	REQUIRE(pack % 8u == 0u && pack >= query_width + heads);
	CUDA(cudaMalloc(&d_query, (uint64_t)rows * query_width * 2u));
	CUDA(cudaMalloc(&d_weight, (uint64_t)rows * heads * 2u));
	CUDA(cudaMalloc(&d_pack, rank_stride * 2u));
	CUDA(cudaMalloc(&d_gathered, (uint64_t)degree * rank_stride * 2u));
	CUDA(cudaMalloc(&d_query_full, (uint64_t)rows * INDEX_HEADS * INDEX_DIM * 2u));
	CUDA(cudaMalloc(&d_weight_full, (uint64_t)rows * INDEX_HEADS * 2u));
	for ( uint32_t rank = 0u; rank < degree; ++rank )
	{
		std::vector<uint16_t> local_query((uint64_t)rows * query_width), local_weight((uint64_t)rows * heads);
		for ( uint32_t row = 0u; row < rows; ++row )
		{
			for ( uint32_t element = 0u; element < query_width; ++element )
				local_query[(uint64_t)row * query_width + element] = Value(rank, row, element, 0x51u);
			for ( uint32_t head = 0u; head < heads; ++head )
				local_weight[(uint64_t)row * heads + head] = Value(rank, row, head, 0xa7u);
		}
		CUDA(cudaMemcpy(d_query, local_query.data(), local_query.size() * 2u, cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(d_weight, local_weight.data(), local_weight.size() * 2u, cudaMemcpyHostToDevice));
		D41IndexPackKernel<<<rows, THREADS>>>(d_query, d_weight, d_pack, query_width, heads, pack);
		CUDA(cudaGetLastError());
		CUDA(cudaMemcpy(d_gathered + (uint64_t)rank * rank_stride, d_pack, rank_stride * 2u, cudaMemcpyDeviceToDevice));
	}
	D41IndexUnpackKernel<<<dim3(rows, degree), THREADS>>>(d_gathered, rank_stride, d_query_full, d_weight_full, query_width, heads, pack);
	CUDA(cudaGetLastError());
	CUDA(cudaDeviceSynchronize());
	CUDA(cudaMemcpy(query.data(), d_query_full, query.size() * 2u, cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpy(weight.data(), d_weight_full, weight.size() * 2u, cudaMemcpyDeviceToHost));
	for ( uint32_t row = 0u; row < rows; ++row )
		for ( uint32_t rank = 0u; rank < degree; ++rank )
		{
			for ( uint32_t element = 0u; element < query_width; ++element )
				REQUIRE(query[((uint64_t)row * degree + rank) * query_width + element] == Value(rank, row, element, 0x51u));
			for ( uint32_t head = 0u; head < heads; ++head )
				REQUIRE(weight[((uint64_t)row * degree + rank) * heads + head] == Value(rank, row, head, 0xa7u));
		}
	CUDA(cudaFree(d_query));
	CUDA(cudaFree(d_weight));
	CUDA(cudaFree(d_pack));
	CUDA(cudaFree(d_gathered));
	CUDA(cudaFree(d_query_full));
	CUDA(cudaFree(d_weight_full));
	printf("index pack TP%u rows %u: %u heads x %u dims + %u weights per rank in a %u-element pack, every query and weight back at its global head\n",
		degree, rows, heads, INDEX_DIM, heads, pack);
}

int main(void)
{
	const uint32_t degrees[] = { 16u, 8u, 4u, 2u, 1u };
	const uint32_t rows[] = { 1u, 3u, 64u, 2048u };
	for ( uint32_t degree : degrees )
		for ( uint32_t count : rows )
			RoundTrip(degree, count);
	printf("PASS dsv41 index pack: at every tensor-parallel degree a rank's index query pack carries all its heads' queries and weights through the gather\n");
	return 0;
}
