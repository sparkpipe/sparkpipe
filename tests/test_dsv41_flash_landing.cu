#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "runtime/launch.h"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stage_kernels.cuh"

#define DEGREE 16u
#define LOCAL 4u
#define HEADS (LOCAL * DEGREE)
#define LATENT 512u
#define PACK 264u
#define RECORD LM_LATENT_SHARD_RECORD_FLOATS(LATENT)
#define GRAIN 8u
#define PAGE_TOKENS 256u
#define RING 8u
#define WINDOW 128u
#define TOPK 512u
#define SLOT 1280u


#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)

static __global__ void Fill(uint16_t *values, uint64_t count, uint64_t salt)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		uint64_t z = (index ^ salt) * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
		z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
		z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
		z ^= z >> 31u;
		values[index] = LmFloatToBf16((float)((int32_t)((z >> 40u) & 0xfffu) - 2048) / 512.0f);
	}
}

template<class T>
static T *Device(const std::vector<T> &host)
{
	T *device;
	CUDA(cudaMalloc((void **)&device, host.size() * sizeof(T) + 16u));
	CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
	return device;
}

template<class T>
static std::vector<T> Host(const T *device, size_t count)
{
	std::vector<T> host(count);
	CUDA(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
	return host;
}

static void Settled(const LmKvAccessError *error_d, const char *phase)
{
	LmKvAccessError error;
	CUDA(cudaDeviceSynchronize());
	CUDA(cudaMemcpy(&error, error_d, sizeof(error), cudaMemcpyDeviceToHost));
	if ( error.error_code != LM_FRAME_ERROR_NONE )
	{
		fprintf(stderr, "FAIL %s: KV access error code %u kind %u row %u sequence %u position %u page %u\n", phase, error.error_code, error.access_kind,
			error.row, error.sequence, error.position, error.page);
		exit(1);
	}
}

template<uint32_t RATIO>
static uint32_t Case(uint32_t rows, uint32_t first_position)
{
	using Kv = LmKvGeometry<SLOT, PAGE_TOKENS / RATIO, true>;
	const uint32_t PAGE = PAGE_TOKENS / RATIO;
	const uint32_t context = first_position + rows, keys = context / RATIO;
	const uint32_t pages = (keys + PAGE - 1u) / PAGE;
	const uint32_t landing_keys = (uint32_t)SparkKvShardGatherKeys(SparkKvShard{DEGREE, 0u, GRAIN}, keys);
	const uint64_t stride = (uint64_t)rows * LOCAL * RECORD, query_stride = (uint64_t)rows * (LOCAL * LATENT + PACK);
	const uint64_t landing_stride = (uint64_t)landing_keys * SLOT;
	const float scale = 0.0441941738f;
	std::vector<uint32_t> table(pages), positions(rows), zeros(rows, 0u), ordinal(rows), indices(rows), begin = {0u, rows}, slot = {0u};
	std::vector<uint32_t> bound(rows), length = {keys}, key_positions(keys), key_sequences(keys, 0u), selected((uint64_t)rows * TOPK), plan = {0u, 0u, keys};
	std::vector<float> sinks(HEADS);
	uint32_t mismatched_records = 0u, mismatched_outputs = 0u, mismatched_split = 0u;
	uint32_t *split_selected, *split_offsets;
	for ( uint32_t page = 0u; page < pages; ++page )
		table[page] = (page * 5u + 1u) % pages;
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		positions[row] = first_position + row;
		ordinal[row] = indices[row] = row;
		bound[row] = (positions[row] + 1u) / RATIO - 1u;
		const uint32_t visible = bound[row] + 1u;
		for ( uint32_t entry = 0u; entry < TOPK; ++entry )
			selected[(uint64_t)row * TOPK + entry] = (row * 13u + entry * 389u) % visible;
	}
	for ( uint32_t key = 0u; key < keys; ++key )
		key_positions[key] = key;
	for ( uint32_t head = 0u; head < HEADS; ++head )
		sinks[head] = 0.25f * (float)(head % 7u) - 0.5f;
	uint32_t *table_d = Device(table), *positions_d = Device(positions), *zeros_d = Device(zeros), *ordinal_d = Device(ordinal);
	uint32_t *indices_d = Device(indices), *begin_d = Device(begin), *slot_d = Device(slot), *bound_d = Device(bound), *length_d = Device(length);
	uint32_t *key_positions_d = Device(key_positions), *key_sequences_d = Device(key_sequences), *selected_d = Device(selected), *plan_d = Device(plan);
	float *sinks_d = Device(sinks);
	uint16_t *values, *queries, *gathered, *wave, *rings, *ring_landing, *outputs_scatter, *outputs_local;
	uint8_t *landing;
	float *sent, *received_scatter, *received_local, *received_split;
	LmKvAccessError *error_d, error;
	std::vector<uint8_t *> pools(DEGREE);
	std::vector<LmKvShardView> views(DEGREE);
	CUDA(cudaMalloc((void **)&values, (uint64_t)keys * (SLOT / 2u) * 2u));
	CUDA(cudaMalloc((void **)&queries, (uint64_t)DEGREE * rows * LOCAL * LATENT * 2u));
	CUDA(cudaMalloc((void **)&gathered, (uint64_t)DEGREE * query_stride * 2u));
	CUDA(cudaMalloc((void **)&wave, (uint64_t)rows * LATENT * 2u));
	CUDA(cudaMalloc((void **)&rings, (uint64_t)DEGREE * RING * LATENT * 2u));
	CUDA(cudaMalloc((void **)&ring_landing, (uint64_t)DEGREE * RING * LATENT * 2u));
	CUDA(cudaMalloc((void **)&landing, (uint64_t)DEGREE * landing_stride));
	CUDA(cudaMalloc((void **)&sent, (uint64_t)DEGREE * DEGREE * stride * 4u));
	CUDA(cudaMalloc((void **)&received_scatter, (uint64_t)DEGREE * DEGREE * stride * 4u));
	CUDA(cudaMalloc((void **)&received_local, (uint64_t)DEGREE * DEGREE * stride * 4u));
	CUDA(cudaMalloc((void **)&outputs_scatter, (uint64_t)DEGREE * rows * LOCAL * LATENT * 2u));
	CUDA(cudaMalloc((void **)&outputs_local, (uint64_t)DEGREE * rows * LOCAL * LATENT * 2u));
	CUDA(cudaMalloc((void **)&error_d, sizeof(LmKvAccessError)));
	LmKvAccessErrorReset(&error);
	CUDA(cudaMemcpy(error_d, &error, sizeof(error), cudaMemcpyHostToDevice));
	CUDA(cudaMemset(gathered, 0, (uint64_t)DEGREE * query_stride * 2u));
	Fill<<<512, 256>>>(values, (uint64_t)keys * (SLOT / 2u), 1u);
	Fill<<<512, 256>>>(queries, (uint64_t)DEGREE * rows * LOCAL * LATENT, 2u);
	Fill<<<64, 256>>>(wave, (uint64_t)rows * LATENT, 3u);
	Fill<<<64, 256>>>(rings, (uint64_t)DEGREE * RING * LATENT, 4u);
	CUDA(cudaGetLastError());
	for ( uint32_t rank = 0u; rank < DEGREE; ++rank )
	{
		const SparkKvShard shard = {DEGREE, rank, GRAIN};
		const uint64_t bytes = SparkKvShardPoolBytes(shard, PAGE, SLOT, pages);
		CUDA(cudaMalloc((void **)&pools[rank], bytes));
		CUDA(cudaMemset(pools[rank], 0, bytes));
		REQUIRE(LmKvShardViewInitialize<Kv>(&views[rank], pools[rank], table_d, pages, 1u, pages, error_d, shard) == 0);
		LmKvShardStoreKernel<Kv, 256u><<<keys, 256>>>(views[rank], values, key_sequences_d, key_positions_d, keys, SLOT / 2u);
		CUDA(cudaMemcpy(gathered + (uint64_t)rank * query_stride, queries + (uint64_t)rank * rows * LOCAL * LATENT,
			(uint64_t)rows * LOCAL * LATENT * 2u, cudaMemcpyDeviceToDevice));
	}
	Settled(error_d, "store");
	for ( uint32_t source = 0u; source < DEGREE; ++source )
	{
		float *records = sent + (uint64_t)source * DEGREE * stride;
		REQUIRE((LmLatentShardAttendHeadsLaunch<Kv, LmKvShardView, LATENT, 0u>(views[source], gathered, query_stride, LOCAL, HEADS, zeros_d,
			length_d, bound_d, selected_d, TOPK, TOPK, scale, records, stride, LOCAL, 1u, rows, 0)) == cudaSuccess);
		D41WindowFoldKernel<<<dim3(rows, HEADS), D41K_THREADS>>>(records, stride, LOCAL, gathered, query_stride, wave,
			rings + (uint64_t)source * RING * LATENT, positions_d, zeros_d, ordinal_d, begin_d, indices_d, slot_d, 0u, RING, WINDOW, DEGREE, source,
			scale, LOCAL, 1u);
		CUDA(cudaGetLastError());
		REQUIRE((LmKvShardGatherPackRangeLaunch<Kv, 256u>(views[source], plan_d, plan_d + 1u, plan_d + 2u, 1u, 0u, landing_keys,
			landing + (uint64_t)source * landing_stride, 0)) == cudaSuccess);
		CUDA(cudaMemcpy(ring_landing + (uint64_t)source * RING * LATENT, rings + (uint64_t)source * RING * LATENT, RING * LATENT * 2u,
			cudaMemcpyDeviceToDevice));
	}
	Settled(error_d, "scatter partials and pack");
	for ( uint32_t target = 0u; target < DEGREE; ++target )
		for ( uint32_t source = 0u; source < DEGREE; ++source )
			CUDA(cudaMemcpy(received_scatter + ((uint64_t)target * DEGREE + source) * stride, sent + ((uint64_t)source * DEGREE + target) * stride,
				stride * 4u, cudaMemcpyDeviceToDevice));
	for ( uint32_t rank = 0u; rank < DEGREE; ++rank )
	{
		float *received = received_local + (uint64_t)rank * DEGREE * stride;
		const uint16_t *query = queries + (uint64_t)rank * rows * LOCAL * LATENT;
		for ( uint32_t source = 0u; source < DEGREE; ++source )
		{
			LmKvShardGatherView view;
			SparkKvShard shard = views[source].shard;
			REQUIRE(shard.rank == source);
			REQUIRE(LmKvShardGatherViewInitialize<Kv>(&view, views[source].pages, shard, landing + (uint64_t)source * landing_stride, plan_d + 1u,
				plan_d + 2u) == 0 && LmKvShardGatherViewWindow(&view, 0u, landing_keys) == 0);
			REQUIRE((LmLatentShardAttendHeadsLaunch<Kv, LmKvShardGatherView, LATENT, 0u>(view, query, (uint64_t)rows * LOCAL * LATENT, LOCAL, LOCAL,
				zeros_d, length_d, bound_d, selected_d, TOPK, TOPK, scale, received + (uint64_t)source * stride, stride, LOCAL, 1u, rows, 0)) == cudaSuccess);
			D41WindowFoldKernel<<<dim3(rows, LOCAL), D41K_THREADS>>>(received + (uint64_t)source * stride, stride, LOCAL, query,
				(uint64_t)rows * LOCAL * LATENT, wave, ring_landing + (uint64_t)source * RING * LATENT, positions_d, zeros_d, ordinal_d, begin_d, indices_d,
				slot_d, 0u, RING, WINDOW, DEGREE, source, scale, LOCAL, 1u);
			CUDA(cudaGetLastError());
		}
		D41MergeSinkKernel<<<dim3(rows, LOCAL), 256>>>(received_scatter + (uint64_t)rank * DEGREE * stride, stride, DEGREE, LOCAL, sinks_d + rank * LOCAL,
			outputs_scatter + (uint64_t)rank * rows * LOCAL * LATENT);
		D41MergeSinkKernel<<<dim3(rows, LOCAL), 256>>>(received, stride, DEGREE, LOCAL, sinks_d + rank * LOCAL,
			outputs_local + (uint64_t)rank * rows * LOCAL * LATENT);
		CUDA(cudaGetLastError());
	}
	{
		CUDA(cudaMalloc((void **)&split_selected, (uint64_t)rows * TOPK * 4u));
		CUDA(cudaMalloc((void **)&split_offsets, (uint64_t)rows * (DEGREE + 1u) * 4u));
		CUDA(cudaMalloc((void **)&received_split, (uint64_t)DEGREE * DEGREE * stride * 4u));
		REQUIRE(LmLatentShardSplitLaunch(views[0].shard, selected_d, TOPK, bound_d, split_selected, split_offsets, rows, 0) == cudaSuccess);
		for ( uint32_t rank = 0u; rank < DEGREE; ++rank )
		{
			float *received = received_split + (uint64_t)rank * DEGREE * stride;
			const uint16_t *query = queries + (uint64_t)rank * rows * LOCAL * LATENT;
			for ( uint32_t source = 0u; source < DEGREE; ++source )
			{
				LmKvShardGatherView view;
				SparkKvShard shard = views[source].shard;
				REQUIRE(LmKvShardGatherViewInitialize<Kv>(&view, views[source].pages, shard, landing + (uint64_t)source * landing_stride, plan_d + 1u,
					plan_d + 2u) == 0 && LmKvShardGatherViewWindow(&view, 0u, landing_keys) == 0);
				REQUIRE((LmLatentShardAttendHeadsLaunch<Kv, LmKvShardGatherView, LATENT, 0u>(view, query, (uint64_t)rows * LOCAL * LATENT, LOCAL, LOCAL,
					zeros_d, length_d, bound_d, selected_d, TOPK, TOPK, scale, received + (uint64_t)source * stride, stride, LOCAL, 1u, rows, 0,
					split_selected, split_offsets, 1u)) == cudaSuccess);
				D41WindowFoldKernel<<<dim3(rows, LOCAL), D41K_THREADS>>>(received + (uint64_t)source * stride, stride, LOCAL, query,
					(uint64_t)rows * LOCAL * LATENT, wave, ring_landing + (uint64_t)source * RING * LATENT, positions_d, zeros_d, ordinal_d, begin_d, indices_d,
					slot_d, 0u, RING, WINDOW, DEGREE, source, scale, LOCAL, 1u);
				CUDA(cudaGetLastError());
			}
		}
	}
	CUDA(cudaDeviceSynchronize());
	CUDA(cudaMemcpy(&error, error_d, sizeof(error), cudaMemcpyDeviceToHost));
	REQUIRE(error.error_code == LM_FRAME_ERROR_NONE);
	{
		const std::vector<float> a = Host(received_scatter, (size_t)DEGREE * DEGREE * stride), b = Host(received_local, (size_t)DEGREE * DEGREE * stride);
		const std::vector<uint16_t> c = Host(outputs_scatter, (size_t)DEGREE * rows * LOCAL * LATENT), d = Host(outputs_local, (size_t)DEGREE * rows * LOCAL * LATENT);
		for ( size_t record = 0u; record < a.size() / RECORD; ++record )
			mismatched_records += memcmp(&a[record * RECORD], &b[record * RECORD], RECORD * sizeof(float)) != 0 ? 1u : 0u;
		for ( size_t element = 0u; element < c.size(); ++element )
			mismatched_outputs += c[element] != d[element] ? 1u : 0u;
		const std::vector<float> e = Host(received_split, (size_t)DEGREE * DEGREE * stride);
		for ( size_t record = 0u; record < b.size() / RECORD; ++record )
			mismatched_split += memcmp(&b[record * RECORD], &e[record * RECORD], RECORD * sizeof(float)) != 0 ? 1u : 0u;
	}
	printf("%s ratio %u rows %u positions %u-%u compressed keys %u landing keys %u: %u of %u records and %u of %u outputs differ; split lists change %u records\n",
		mismatched_records == 0u && mismatched_outputs == 0u && mismatched_split == 0u ? "ok  " : "FAIL", RATIO, rows, first_position, context - 1u, keys, landing_keys,
		mismatched_records, (uint32_t)((uint64_t)DEGREE * DEGREE * rows * LOCAL), mismatched_outputs, (uint32_t)((uint64_t)DEGREE * rows * LOCAL * LATENT),
		mismatched_split);
	for ( uint32_t rank = 0u; rank < DEGREE; ++rank )
		cudaFree(pools[rank]);
	cudaFree(table_d); cudaFree(positions_d); cudaFree(zeros_d); cudaFree(ordinal_d); cudaFree(indices_d); cudaFree(begin_d); cudaFree(slot_d);
	cudaFree(bound_d); cudaFree(length_d); cudaFree(key_positions_d); cudaFree(key_sequences_d); cudaFree(selected_d); cudaFree(plan_d); cudaFree(sinks_d);
	cudaFree(values); cudaFree(queries); cudaFree(gathered); cudaFree(wave); cudaFree(rings); cudaFree(ring_landing); cudaFree(landing);
	cudaFree(sent); cudaFree(received_scatter); cudaFree(received_local); cudaFree(outputs_scatter); cudaFree(outputs_local); cudaFree(error_d);
	cudaFree(split_selected); cudaFree(split_offsets); cudaFree(received_split);
	return mismatched_records + mismatched_outputs + mismatched_split;
}

int main(void)
{
	uint32_t failures = 0u;
	failures += Case<2u>(17u, 300u);
	failures += Case<2u>(64u, 2000u);
	failures += Case<2u>(200u, 3000u);
	failures += Case<2u>(256u, 1500u);
	failures += Case<1u>(17u, 300u);
	failures += Case<1u>(64u, 2000u);
	failures += Case<1u>(200u, 3000u);
	failures += Case<1u>(256u, 1500u);
	failures += Case<1u>(256u, 9000u);
	if ( failures != 0u )
		return 1;
	printf("PASS dsv41 landing: attention over a landed KV copy of a ratio-2 cache and of the full-length ratio-1 cache, for a rank's own heads, equals the all-to-all records and merged output bit for bit, with or without split selection lists and one head per block\n");
	return 0;
}
