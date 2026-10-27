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

template<class T>
static T *Upload(const std::vector<T> &host)
{
	T *device = 0;
	CUDA(cudaMalloc((void **)&device, host.size() * sizeof(T)));
	CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
	return device;
}

static uint32_t failures;

static void Case(uint32_t context, uint32_t rows, uint32_t degree, uint32_t rank, uint32_t early, bool timed)
{
	const uint32_t pages = (context + G5N_PAGE_SLOTS - 1u) / G5N_PAGE_SLOTS, pools = SparkGlm5NextIndexCpPools(context);
	const uint32_t local_stride = SparkGlm5NextIndexCpLocalStride(pools, degree);
	std::vector<uint32_t> table(pages), token_sequence(context, 0u), token_position(context), sequence(rows, 0u), position(rows), length(1, context);
	std::vector<uint16_t> packed((uint64_t)context * G5N_INDEX_PACKED), query((uint64_t)rows * G5N_INDEX_HEADS * G5N_INDEX_DIM),
		weight((uint64_t)rows * G5N_INDEX_HEADS);
	std::vector<float> ape(G5N_KPOOL * G5N_INDEX_DIM);
	LmKvAccessError *error = 0;
	LmKvShardView view;
	uint8_t *pool_bytes = 0;
	float *single = 0, *grouped = 0;
	for ( uint32_t page = 0u; page < pages; page++ )
		table[page] = (page * 7u + 3u) % pages;
	for ( uint32_t token = 0u; token < context; token++ )
	{
		token_position[token] = token;
		for ( uint32_t element = 0u; element < 2u * G5N_INDEX_DIM; element++ )
			packed[(uint64_t)token * G5N_INDEX_PACKED + element] = Bf16(2.0f * Uniform(1u, (uint64_t)token * G5N_INDEX_PACKED + element));
		packed[(uint64_t)token * G5N_INDEX_PACKED + 2u * G5N_INDEX_DIM] = 0x3F80u;
	}
	for ( uint64_t index = 0u; index < query.size(); index++ )
		query[index] = Bf16(Uniform(2u, index));
	for ( uint64_t index = 0u; index < weight.size(); index++ )
		weight[index] = Bf16(0.5f + 0.5f * Uniform(3u, index));
	for ( uint64_t index = 0u; index < ape.size(); index++ )
		ape[index] = 0.25f * Uniform(4u, index);
	for ( uint32_t row = 0u; row < rows; row++ )
		position[row] = row < early ? (uint32_t)(Mix(5u + row) % context) : context - rows + row;
	CUDA(cudaMallocManaged((void **)&error, sizeof(*error)));
	LmKvAccessErrorReset(error);
	const SparkKvShard shard = SparkGlm5NextIndexCpShard(rank, degree);
	CUDA(cudaMalloc((void **)&pool_bytes, SparkKvShardPoolBytes(shard, G5N_PAGE_SLOTS, Glm5NextIndexKv::kSlotBytes, pages)));
	uint32_t *d_table = Upload(table), *d_token_sequence = Upload(token_sequence), *d_token_position = Upload(token_position);
	uint32_t *d_sequence = Upload(sequence), *d_position = Upload(position), *d_length = Upload(length);
	uint16_t *d_packed = Upload(packed), *d_query = Upload(query), *d_weight = Upload(weight);
	float *d_ape = Upload(ape);
	REQUIRE(LmKvShardViewInitialize<Glm5NextIndexKv>(&view, pool_bytes, d_table, pages, 1u, pages, error, shard) == 0);
	LmKvShardStoreKernel<Glm5NextIndexKv, G5N_THREADS><<<context, G5N_THREADS>>>(view, d_packed, d_token_sequence, d_token_position, context,
		G5N_INDEX_PACKED);
	CUDA(cudaMalloc((void **)&single, (uint64_t)rows * local_stride * sizeof(float)));
	CUDA(cudaMalloc((void **)&grouped, (uint64_t)rows * local_stride * sizeof(float)));
	CUDA(cudaMemset(single, 0xa5, (uint64_t)rows * local_stride * sizeof(float)));
	CUDA(cudaMemset(grouped, 0xa5, (uint64_t)rows * local_stride * sizeof(float)));
	auto run_single = [&]() {
		Glm5NextPoolScoreKernel<G5N_THREADS, G5N_INDEX_DIM, G5N_KPOOL, G5N_INDEX_HEADS><<<dim3(local_stride, rows), G5N_THREADS>>>(d_query, d_weight,
			view, d_sequence, d_length, d_position, d_ape, pools, rank, degree, local_stride, SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE,
			SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE, single);
	};
	auto run_grouped = [&]() {
		Glm5NextPoolScoreRowsKernel<G5N_THREADS, G5N_INDEX_DIM, G5N_KPOOL, G5N_INDEX_HEADS, G5N_POOL_BLOCK_POOLS, G5N_POOL_BLOCK_ROWS>
			<<<dim3((local_stride + G5N_POOL_BLOCK_POOLS - 1u) / G5N_POOL_BLOCK_POOLS, (rows + G5N_POOL_BLOCK_ROWS - 1u) / G5N_POOL_BLOCK_ROWS),
			G5N_THREADS>>>(d_query, d_weight, view, d_sequence, d_length, d_position, d_ape, pools, rank, degree, local_stride,
			SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE, SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE, grouped, rows);
	};
	run_single();
	run_grouped();
	CUDA(cudaDeviceSynchronize());
	std::vector<float> a((uint64_t)rows * local_stride), b((uint64_t)rows * local_stride);
	CUDA(cudaMemcpy(a.data(), single, a.size() * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpy(b.data(), grouped, b.size() * sizeof(float), cudaMemcpyDeviceToHost));
	uint64_t differ = 0u, finite = 0u;
	for ( uint64_t index = 0u; index < a.size(); index++ )
	{
		differ += memcmp(&a[index], &b[index], sizeof(float)) != 0;
		finite += a[index] > -INFINITY && a[index] < INFINITY;
	}
	REQUIRE(error->error_code == LM_FRAME_ERROR_NONE);
	printf("context %6u rows %4u degree %2u rank %2u early %2u: %llu of %llu scores differ (%llu finite)", context, rows, degree, rank, early,
		(unsigned long long)differ, (unsigned long long)a.size(), (unsigned long long)finite);
	if ( timed )
	{
		cudaEvent_t start, stop;
		float single_ms = 0.0f, grouped_ms = 0.0f;
		CUDA(cudaEventCreate(&start));
		CUDA(cudaEventCreate(&stop));
		CUDA(cudaEventRecord(start));
		for ( uint32_t i = 0u; i < 5u; i++ )
			run_single();
		CUDA(cudaEventRecord(stop));
		CUDA(cudaEventSynchronize(stop));
		CUDA(cudaEventElapsedTime(&single_ms, start, stop));
		CUDA(cudaEventRecord(start));
		for ( uint32_t i = 0u; i < 5u; i++ )
			run_grouped();
		CUDA(cudaEventRecord(stop));
		CUDA(cudaEventSynchronize(stop));
		CUDA(cudaEventElapsedTime(&grouped_ms, start, stop));
		printf("; one pool per block %.1f us, %u pools x %u rows per block %.1f us", 1000.0f * single_ms / 5.0f, G5N_POOL_BLOCK_POOLS,
			G5N_POOL_BLOCK_ROWS, 1000.0f * grouped_ms / 5.0f);
	}
	printf("\n");
	failures += differ != 0u || finite == 0u;
	cudaFree(pool_bytes); cudaFree(single); cudaFree(grouped); cudaFree(error);
	cudaFree(d_table); cudaFree(d_token_sequence); cudaFree(d_token_position); cudaFree(d_sequence); cudaFree(d_position); cudaFree(d_length);
	cudaFree(d_packed); cudaFree(d_query); cudaFree(d_weight); cudaFree(d_ape);
}

int main(void)
{
	Case(1024u, 64u, 1u, 0u, 0u, false);
	Case(4096u, 100u, 16u, 0u, 7u, false);
	Case(4096u, 77u, 16u, 5u, 20u, false);
	Case(9000u, 130u, 16u, 15u, 3u, false);
	Case(16384u, 2048u, 16u, 3u, 0u, true);
	Case(65536u, 2048u, 16u, 9u, 0u, true);
	if ( failures != 0u )
	{
		printf("FAIL glm5_next pool score: %u cases differ or score nothing\n", failures);
		return 1;
	}
	printf("PASS glm5_next pool score: the multi-row kernel equals the one-pool kernel bit for bit\n");
	return 0;
}
