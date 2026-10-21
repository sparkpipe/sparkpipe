#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_stage_kernels.cuh"

#define DEGREE 16u
#define LATENT 512u
#define RING 8u
#define WINDOW 128u

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)

static __global__ void ReferenceWindowFoldKernel(float *__restrict__ records, uint64_t destination_stride, uint32_t heads_per_destination,
	const uint16_t *__restrict__ query, uint64_t query_rank_stride, const uint16_t *__restrict__ wave_kv,
	const uint16_t *__restrict__ ring, const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane,
	const uint32_t *__restrict__ row_ordinal, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices,
	const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows, uint32_t window, uint32_t degree, uint32_t rank,
	float qk_scale, uint32_t record_heads, uint32_t record_copies)
{
	__shared__ float score_shared[D41K_THREADS / 32u];
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	const uint32_t position = positions[row], lane = row_lane[row];
	const uint32_t wave_first = position - row_ordinal[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const uint32_t first = position + 1u > window ? position + 1u - window : 0u;
	const uint32_t destination = head / heads_per_destination, local = head % heads_per_destination;
	const uint16_t *q = query + (uint64_t)destination * query_rank_stride + ((uint64_t)row * heads_per_destination + local) * D41K_LATENT;
	const uint64_t record_offset = ((uint64_t)row * record_heads + head % record_heads) * D41K_RECORD;
	float *record = records + (uint64_t)(head / record_heads) * record_copies * destination_stride + record_offset;
	float running_max = record[0], running_sum = record[1];
	float accumulator[D41K_LATENT / D41K_THREADS];
	for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
		accumulator[e] = record[2u + threadIdx.x + e * D41K_THREADS];
	for ( uint32_t key = first; key <= position; ++key )
	{
		if ( key % degree != rank )
			continue;
		const uint16_t *k = key >= wave_first ? wave_kv + (uint64_t)lane_rows[key - wave_first] * D41K_LATENT :
			ring + (uint64_t)recurrent_slot[lane] * ring_slot_stride + (uint64_t)((key / degree) % ring_rows) * D41K_LATENT;
		float partial = 0.0f, value[D41K_LATENT / D41K_THREADS];
		for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
		{
			value[e] = LmBf16ToFloat(k[threadIdx.x + e * D41K_THREADS]);
			partial += LmBf16ToFloat(q[threadIdx.x + e * D41K_THREADS]) * value[e];
		}
		const float score = SparkDsv41FlashBlockSum(partial, score_shared) * qk_scale;
		const float top = fmaxf(running_max, score);
		const float keep = running_max == -INFINITY ? 0.0f : __expf(running_max - top), add = __expf(score - top);
		running_sum = running_sum * keep + add;
		for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
			accumulator[e] = accumulator[e] * keep + add * value[e];
		running_max = top;
	}
	__syncthreads();
	for ( uint32_t copy = 0u; copy < record_copies; ++copy )
	{
		float *target = record + (uint64_t)copy * destination_stride;
		for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
			target[2u + threadIdx.x + e * D41K_THREADS] = accumulator[e];
		if ( threadIdx.x == 0u )
		{
			target[0] = running_max;
			target[1] = running_sum;
		}
	}
}

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

static std::vector<uint16_t> Values(size_t count, uint64_t salt)
{
	std::vector<uint16_t> values(count);
	for ( size_t index = 0u; index < count; ++index )
		values[index] = Bf16((float)((int32_t)((Mix(index ^ salt) >> 40u) & 0xfffu) - 2048) / 512.0f);
	return values;
}

template<class T>
static T *Device(const std::vector<T> &host)
{
	T *device;
	CUDA(cudaMalloc((void **)&device, host.size() * sizeof(T) + 16u));
	CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
	return device;
}

struct Case
{
	uint32_t rows, heads, heads_per_destination, record_heads, record_copies, rank, first_position, ring_stride;
};

static uint32_t RunCase(const Case &c, uint32_t perturb)
{
	const uint32_t destinations = c.heads / c.heads_per_destination;
	const uint64_t query_rank_stride = (uint64_t)c.rows * c.heads_per_destination * LATENT;
	const uint64_t destination_stride = (uint64_t)c.rows * c.record_heads * D41K_RECORD;
	const uint64_t record_floats = (uint64_t)(c.heads / c.record_heads) * c.record_copies * destination_stride;
	std::vector<uint16_t> query = Values((size_t)destinations * query_rank_stride, 11u + c.rows);
	std::vector<uint16_t> wave = Values((size_t)c.rows * LATENT, 23u + c.rank);
	std::vector<uint16_t> ring = Values((size_t)2u * RING * LATENT, 37u + c.first_position);
	std::vector<uint32_t> positions(c.rows), lanes(c.rows, 0u), ordinals(c.rows), begin = {0u, c.rows}, indices(c.rows), slot = {1u};
	for ( uint32_t row = 0u; row < c.rows; ++row )
	{
		positions[row] = c.first_position + row;
		ordinals[row] = row;
		indices[row] = row;
	}
	std::vector<float> records(record_floats);
	for ( uint64_t index = 0u; index < record_floats; ++index )
		records[index] = (float)((int32_t)((Mix(index ^ 0x5151u) >> 40u) & 0xffffu) - 32768) / 4096.0f;
	for ( uint64_t base = 0u; base < record_floats; base += D41K_RECORD )
	{
		const uint64_t z = Mix(base);
		records[base] = (z & 3u) == 0u ? -INFINITY : (float)((int32_t)((z >> 8u) & 0xffu) - 128) / 16.0f;
		records[base + 1u] = records[base] == -INFINITY ? 0.0f : (float)((z >> 20u) & 0xffu) / 8.0f;
	}
	uint16_t *query_d = Device(query), *wave_d = Device(wave), *ring_d = Device(ring);
	uint32_t *positions_d = Device(positions), *lanes_d = Device(lanes), *ordinals_d = Device(ordinals), *begin_d = Device(begin),
		*indices_d = Device(indices), *slot_d = Device(slot);
	float *reference_d = Device(records), *candidate_d = Device(records);
	const float scale = 1.0f / sqrtf((float)LATENT);
	ReferenceWindowFoldKernel<<<dim3(c.rows, c.heads), D41K_THREADS>>>(reference_d, destination_stride, c.heads_per_destination, query_d,
		query_rank_stride, wave_d, ring_d, positions_d, lanes_d, ordinals_d, begin_d, indices_d, slot_d, c.ring_stride, RING, WINDOW, DEGREE, c.rank,
		scale, c.record_heads, c.record_copies);
	CUDA(cudaGetLastError());
	if ( perturb != 0u )
	{
		uint16_t bumped = (uint16_t)(ring[(uint64_t)c.ring_stride + 3u] ^ 0x0100u);
		CUDA(cudaMemcpy(ring_d + c.ring_stride + 3u, &bumped, sizeof(bumped), cudaMemcpyHostToDevice));
		bumped = (uint16_t)(wave[3u] ^ 0x0100u);
		CUDA(cudaMemcpy(wave_d + 3u, &bumped, sizeof(bumped), cudaMemcpyHostToDevice));
	}
	D41WindowFoldKernel<<<dim3(c.rows, c.heads), D41K_THREADS>>>(candidate_d, destination_stride, c.heads_per_destination, query_d,
		query_rank_stride, wave_d, ring_d, positions_d, lanes_d, ordinals_d, begin_d, indices_d, slot_d, c.ring_stride, RING, WINDOW, DEGREE, c.rank,
		scale, c.record_heads, c.record_copies);
	CUDA(cudaGetLastError());
	CUDA(cudaDeviceSynchronize());
	std::vector<float> reference(record_floats), candidate(record_floats);
	CUDA(cudaMemcpy(reference.data(), reference_d, record_floats * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpy(candidate.data(), candidate_d, record_floats * sizeof(float), cudaMemcpyDeviceToHost));
	uint64_t differing = 0u;
	for ( uint64_t index = 0u; index < record_floats; ++index )
		differing += memcmp(&reference[index], &candidate[index], sizeof(float)) != 0 ? 1u : 0u;
	cudaFree(query_d); cudaFree(wave_d); cudaFree(ring_d); cudaFree(positions_d); cudaFree(lanes_d); cudaFree(ordinals_d);
	cudaFree(begin_d); cudaFree(indices_d); cudaFree(slot_d); cudaFree(reference_d); cudaFree(candidate_d);
	printf("case rows %u heads %u record_heads %u copies %u rank %u first %u%s: %llu of %llu floats differ\n", c.rows, c.heads, c.record_heads,
		c.record_copies, c.rank, c.first_position, perturb != 0u ? " (perturbed keys)" : "", (unsigned long long)differing,
		(unsigned long long)record_floats);
	return differing != 0u ? 1u : 0u;
}

int main(void)
{
	const Case cases[] = {
		{1u, 64u, 4u, 4u, 1u, 0u, 9999u, RING * LATENT},
		{1u, 64u, 4u, 8u, 2u, 7u, 70000u, RING * LATENT},
		{6u, 64u, 4u, 8u, 2u, 15u, 4096u, RING * LATENT},
		{17u, 64u, 4u, 4u, 1u, 3u, 123457u, RING * LATENT},
		{1024u, 4u, 4u, 4u, 1u, 9u, 33000u, 0u},
		{256u, 4u, 4u, 4u, 1u, 0u, 5u, 0u},
		{3u, 64u, 4u, 4u, 1u, 14u, 2u, RING * LATENT},
	};
	uint32_t failures = 0u;
	for ( const Case &c : cases )
		failures += RunCase(c, 0u);
	const uint32_t sensitive = RunCase(cases[1], 1u);
	if ( failures != 0u || sensitive == 0u )
	{
		printf("FAIL dsv41 window fold: %u cases differ from the sequential fold, perturbed keys %s\n", failures, sensitive != 0u ? "detected" : "not detected");
		return 1;
	}
	printf("PASS dsv41 window fold: batched scores fold every case to the same bits as the sequential fold\n");
	return 0;
}
