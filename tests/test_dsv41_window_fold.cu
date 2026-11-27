#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "model-families/dsv41_flash/source/dsv41_flash_stage_kernels.cuh"
#include "model-families/dsv41_flash/source/dsv41_flash_attend.cuh"
#include "inference/kernels/index_score.cuh"
#include "inference/kernels/mx_gemm.cuh"
#include "inference/kernels/ngram_hash.cuh"

#define DEGREE 16u
#define LATENT 512u
#define RING 8u
#define WINDOW 128u
#define WAVE_WORST 0.03125
#define WAVE_MEAN 0.001

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
			accumulator[e] = accumulator[e] * keep + SparkMathAttentionValueWeight(add) * value[e];
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
	uint32_t rows, heads, heads_per_destination, record_heads, record_copies, rank, first_position, ring_stride, whole;
};

static uint32_t RunCase(const Case &c, uint32_t perturb)
{
	const uint32_t destinations = c.heads / c.heads_per_destination;
	const uint64_t query_rank_stride = (uint64_t)c.rows * c.heads_per_destination * LATENT;
	const uint64_t destination_stride = (uint64_t)c.rows * c.record_heads * D41K_RECORD;
	const uint64_t record_floats = (uint64_t)(c.heads / c.record_heads) * c.record_copies * destination_stride;
	std::vector<uint16_t> query = Values((size_t)destinations * query_rank_stride, 11u + c.rows);
	std::vector<uint16_t> wave = Values((size_t)c.rows * LATENT, 23u + c.rank);
	const uint32_t degree = c.whole != 0u ? 1u : DEGREE, ring_rows = c.whole != 0u ? WINDOW : RING;
	std::vector<uint16_t> ring = Values((size_t)2u * ring_rows * LATENT, 37u + c.first_position);
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
		query_rank_stride, wave_d, ring_d, positions_d, lanes_d, ordinals_d, begin_d, indices_d, slot_d, c.ring_stride, ring_rows, WINDOW, degree, c.rank,
		scale, c.record_heads, c.record_copies);
	CUDA(cudaGetLastError());
	if ( perturb != 0u )
	{
		uint16_t bumped = (uint16_t)(ring[(uint64_t)c.ring_stride + 3u] ^ 0x0100u);
		CUDA(cudaMemcpy(ring_d + c.ring_stride + 3u, &bumped, sizeof(bumped), cudaMemcpyHostToDevice));
		bumped = (uint16_t)(wave[3u] ^ 0x0100u);
		CUDA(cudaMemcpy(wave_d + 3u, &bumped, sizeof(bumped), cudaMemcpyHostToDevice));
	}
	if ( c.whole == 1u )
		D41WindowFoldWholeKernel<<<dim3(c.rows, c.heads), D41K_THREADS>>>(candidate_d, c.heads, query_d, wave_d, ring_d, positions_d, lanes_d,
			ordinals_d, begin_d, indices_d, slot_d, c.ring_stride, WINDOW, scale);
	else if ( c.whole == 3u )
		D41WindowFoldClusterKernel<<<dim3(D41K_WINDOW_PARTS, c.heads, c.rows), D41K_THREADS>>>(candidate_d, c.heads, query_d, wave_d, ring_d, positions_d,
			lanes_d, ordinals_d, begin_d, indices_d, slot_d, c.ring_stride, WINDOW, scale);
	else
		D41WindowFoldKernel<<<dim3(c.rows, c.heads), D41K_THREADS>>>(candidate_d, destination_stride, c.heads_per_destination, query_d,
			query_rank_stride, wave_d, ring_d, positions_d, lanes_d, ordinals_d, begin_d, indices_d, slot_d, c.ring_stride, ring_rows, WINDOW, degree,
			c.rank, scale, c.record_heads, c.record_copies);
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
	printf("case rows %u heads %u record_heads %u copies %u rank %u first %u%s%s: %llu of %llu floats differ\n", c.rows, c.heads, c.record_heads,
		c.record_copies, c.rank, c.first_position, c.whole == 1u ? " (whole window, one pass)" : c.whole == 2u ? " (whole window, chunks)" : c.whole == 3u ? " (whole window, cluster)" : "",
		perturb != 0u ? " (perturbed keys)" : "",
		(unsigned long long)differing, (unsigned long long)record_floats);
	return differing != 0u ? 1u : 0u;
}

struct Wave
{
	uint32_t lanes, count[2], first[2], heads;
};

static float FromBf16(uint16_t value)
{
	const uint32_t bits = (uint32_t)value << 16u;
	float result;
	memcpy(&result, &bits, 4u);
	return result;
}

template<class T>
static T *Scratch(size_t count)
{
	T *device;
	CUDA(cudaMalloc((void **)&device, count * sizeof(T) + 16u));
	CUDA(cudaMemset(device, 0x5a, count * sizeof(T)));
	return device;
}

static uint32_t coded_mismatch = 0u;

static uint32_t RunWave(const Wave &w, uint32_t perturb)
{
	const uint32_t rows = w.count[0] + (w.lanes > 1u ? w.count[1] : 0u), heads = w.heads, packed = rows * heads;
	const uint64_t stride = (uint64_t)rows * heads * D41K_RECORD;
	std::vector<uint16_t> query = Values((size_t)packed * LATENT, 41u + rows);
	std::vector<uint16_t> wave = Values((size_t)rows * LATENT, 43u + w.first[0]);
	std::vector<uint16_t> ring = Values((size_t)2u * WINDOW * LATENT, 47u + w.first[1]);
	std::vector<uint32_t> positions(rows), lanes(rows), ordinals(rows), begin = {0u, w.count[0], rows}, indices(rows), slot = {1u, 0u};
	std::vector<float> sink(heads), records(stride, 0.0f);
	for ( uint32_t lane = 0u, at = 0u; lane < w.lanes; ++lane )
		for ( uint32_t ordinal = 0u; ordinal < w.count[lane]; ++ordinal, ++at )
		{
			const uint32_t row = rows - 1u - at;
			indices[at] = row;
			positions[row] = w.first[lane] + ordinal;
			lanes[row] = lane;
			ordinals[row] = ordinal;
		}
	for ( uint32_t head = 0u; head < heads; ++head )
		sink[head] = (float)((int32_t)(Mix(head ^ 0x77u) & 0xffu) - 128) / 32.0f;
	for ( uint64_t base = 0u; base < stride; base += D41K_RECORD )
		records[base] = -INFINITY;
	uint16_t *query_d = Device(query), *wave_d = Device(wave), *ring_d = Device(ring);
	uint32_t *positions_d = Device(positions), *lanes_d = Device(lanes), *ordinals_d = Device(ordinals), *begin_d = Device(begin),
		*indices_d = Device(indices), *slot_d = Device(slot);
	float *sink_d = Device(sink), *sequential_d = Device(records);
	uint16_t *expected_d = Scratch<uint16_t>((size_t)packed * LATENT), *merged_d = Scratch<uint16_t>((size_t)packed * LATENT);
	const float scale = 1.0f / sqrtf((float)LATENT);
	LmKvShardView none;
	memset(&none, 0, sizeof(none));
	ReferenceWindowFoldKernel<<<dim3(rows, heads), D41K_THREADS>>>(sequential_d, stride, heads, query_d, (uint64_t)packed * LATENT, wave_d, ring_d,
		positions_d, lanes_d, ordinals_d, begin_d, indices_d, slot_d, WINDOW * LATENT, WINDOW, WINDOW, 1u, 0u, scale, heads, 1u);
	D41MergeSinkKernel<<<dim3(rows, heads), D41K_THREADS>>>(sequential_d, stride, 1u, heads, sink_d, expected_d);
	CUDA(cudaGetLastError());
	if ( perturb != 0u )
	{
		const uint16_t bumped = (uint16_t)(wave[(size_t)(rows / 2u) * LATENT + 5u] ^ 0x0100u);
		CUDA(cudaMemcpy(wave_d + (size_t)(rows / 2u) * LATENT + 5u, &bumped, sizeof(bumped), cudaMemcpyHostToDevice));
	}
	std::vector<float> frequency(D41K_ROPE_PAIRS);
	for ( uint32_t pair = 0u; pair < D41K_ROPE_PAIRS; ++pair )
		frequency[pair] = powf(10000.0f, -(float)pair / (float)D41K_ROPE_PAIRS);
	float *frequency_d = Device(frequency);
	const uint32_t group_heads = heads / 2u, width = group_heads * LATENT;
	uint8_t *codes_d = Scratch<uint8_t>((size_t)rows * heads * LATENT + (size_t)rows * heads * LATENT / 32u);
	uint8_t *reference_d = Scratch<uint8_t>((size_t)rows * heads * LATENT + (size_t)rows * heads * LATENT / 32u);
	uint16_t *roped_d = Scratch<uint16_t>((size_t)packed * LATENT);
	CUDA((D41AttendLaunch<LmKvGeometry<1280u, 64u, true>, LATENT>(none, 0u, query_d, (const uint32_t *)0, (const uint32_t *)0, (const uint32_t *)0,
		(const uint32_t *)0, 0u, wave_d, ring_d, WINDOW * LATENT, WINDOW, positions_d, lanes_d, begin_d, indices_d, slot_d, sink_d, scale, merged_d,
		(const float *)0, 0u, 0u, (uint8_t *)0, rows, rows, 0u, 0)));
	CUDA((D41AttendLaunch<LmKvGeometry<1280u, 64u, true>, LATENT>(none, 0u, query_d, (const uint32_t *)0, (const uint32_t *)0, (const uint32_t *)0,
		(const uint32_t *)0, 0u, wave_d, ring_d, WINDOW * LATENT, WINDOW, positions_d, lanes_d, begin_d, indices_d, slot_d, sink_d, scale, (uint16_t *)0,
		frequency_d, LATENT - 2u * D41K_ROPE_PAIRS, group_heads, codes_d, rows, rows, 0u, 0)));
	CUDA(cudaMemcpy(roped_d, merged_d, (size_t)packed * LATENT * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
	D41RopeRowsKernel<<<dim3(rows, heads), D41K_ROPE_PAIRS>>>(roped_d, heads, LATENT, LATENT - 2u * D41K_ROPE_PAIRS, positions_d, 0, frequency_d, 1u,
		(const uint32_t *)0);
	for ( uint32_t group = 0u; group < 2u; ++group )
		REQUIRE(LmMxQuantizeRows(roped_d + (size_t)group * width, (uint64_t)heads * LATENT, reference_d + (size_t)group * rows * width,
			reference_d + (size_t)2u * rows * width + (size_t)group * rows * (width / 32u), rows, width, 0) == LM_LAUNCH_OK);
	CUDA(cudaGetLastError());
	CUDA(cudaDeviceSynchronize());
	if ( perturb == 0u )
	{
		const size_t bytes = (size_t)rows * heads * LATENT + (size_t)rows * heads * LATENT / 32u;
		std::vector<uint8_t> coded(bytes), reference(bytes);
		CUDA(cudaMemcpy(coded.data(), codes_d, bytes, cudaMemcpyDeviceToHost));
		CUDA(cudaMemcpy(reference.data(), reference_d, bytes, cudaMemcpyDeviceToHost));
		uint64_t differ = 0u;
		for ( size_t index = 0u; index < bytes; ++index )
			differ += coded[index] != reference[index] ? 1u : 0u;
		printf("attend codes lanes %u rows %u: %llu of %llu code and scale bytes differ from rope then quantize\n", w.lanes, rows,
			(unsigned long long)differ, (unsigned long long)bytes);
		coded_mismatch += differ != 0u ? 1u : 0u;
		const uint32_t live = rows - rows / 3u;
		std::vector<uint8_t> partial(bytes);
		CUDA(cudaMemset(codes_d, 0xa5, bytes));
		CUDA((D41AttendLaunch<LmKvGeometry<1280u, 64u, true>, LATENT>(none, 0u, query_d, (const uint32_t *)0, (const uint32_t *)0, (const uint32_t *)0,
			(const uint32_t *)0, 0u, wave_d, ring_d, WINDOW * LATENT, WINDOW, positions_d, lanes_d, begin_d, indices_d, slot_d, sink_d, scale, (uint16_t *)0,
			frequency_d, LATENT - 2u * D41K_ROPE_PAIRS, group_heads, codes_d, rows, live, 0u, 0)));
		CUDA(cudaDeviceSynchronize());
		CUDA(cudaMemcpy(partial.data(), codes_d, bytes, cudaMemcpyDeviceToHost));
		uint64_t wrong = 0u;
		for ( uint32_t group = 0u; group < 2u; ++group )
			for ( uint32_t row = 0u; row < rows; ++row )
			{
				const size_t code = ((size_t)group * rows + row) * width, scale_at = (size_t)2u * rows * width + ((size_t)group * rows + row) * (width / 32u);
				for ( size_t byte = 0u; byte < width; ++byte )
					wrong += partial[code + byte] != (row < live ? coded[code + byte] : (uint8_t)0xa5u) ? 1u : 0u;
				for ( size_t byte = 0u; byte < width / 32u; ++byte )
					wrong += partial[scale_at + byte] != (row < live ? coded[scale_at + byte] : (uint8_t)0xa5u) ? 1u : 0u;
			}
		printf("attend codes lanes %u rows %u live %u: %llu bytes differ from the full launch or touch rows past the live count\n", w.lanes, rows, live,
			(unsigned long long)wrong);
		coded_mismatch += wrong != 0u ? 1u : 0u;
		const size_t part_elements = (size_t)packed * LATENT;
		std::vector<uint16_t> pair_query = query, second = Values(part_elements, 53u + rows);
		pair_query.insert(pair_query.end(), second.begin(), second.end());
		std::vector<float> pair_sink = sink;
		for ( uint32_t head = 0u; head < heads; ++head )
			pair_sink.push_back((float)((int32_t)(Mix(head ^ 0x99u) & 0xffu) - 128) / 32.0f);
		uint16_t *pair_query_d = Device(pair_query), *single_out_d = Scratch<uint16_t>(2u * part_elements), *pair_out_d = Scratch<uint16_t>(2u * part_elements);
		float *pair_sink_d = Device(pair_sink);
		uint8_t *single_codes_d = Scratch<uint8_t>(2u * bytes), *pair_codes_d = Scratch<uint8_t>(2u * bytes);
		for ( uint32_t part = 0u; part < 2u; ++part )
		{
			CUDA((D41AttendLaunch<LmKvGeometry<1280u, 64u, true>, LATENT>(none, 0u, pair_query_d + part * part_elements, (const uint32_t *)0,
				(const uint32_t *)0, (const uint32_t *)0, (const uint32_t *)0, 0u, wave_d, ring_d, WINDOW * LATENT, WINDOW, positions_d, lanes_d, begin_d,
				indices_d, slot_d, pair_sink_d + part * heads, scale, single_out_d + part * part_elements, (const float *)0, 0u, 0u, (uint8_t *)0, rows, rows,
				0u, 0)));
			CUDA((D41AttendLaunch<LmKvGeometry<1280u, 64u, true>, LATENT>(none, 0u, pair_query_d + part * part_elements, (const uint32_t *)0,
				(const uint32_t *)0, (const uint32_t *)0, (const uint32_t *)0, 0u, wave_d, ring_d, WINDOW * LATENT, WINDOW, positions_d, lanes_d, begin_d,
				indices_d, slot_d, pair_sink_d + part * heads, scale, (uint16_t *)0, frequency_d, LATENT - 2u * D41K_ROPE_PAIRS, group_heads,
				single_codes_d + part * bytes, rows, rows, 0u, 0)));
		}
		CUDA((D41AttendLaunch<LmKvGeometry<1280u, 64u, true>, LATENT, 2u>(none, 0u, pair_query_d, (const uint32_t *)0, (const uint32_t *)0,
			(const uint32_t *)0, (const uint32_t *)0, 0u, wave_d, ring_d, WINDOW * LATENT, WINDOW, positions_d, lanes_d, begin_d, indices_d, slot_d,
			pair_sink_d, scale, pair_out_d, (const float *)0, 0u, 0u, (uint8_t *)0, rows, rows, 0u, 0, part_elements, part_elements, 0u)));
		CUDA((D41AttendLaunch<LmKvGeometry<1280u, 64u, true>, LATENT, 2u>(none, 0u, pair_query_d, (const uint32_t *)0, (const uint32_t *)0,
			(const uint32_t *)0, (const uint32_t *)0, 0u, wave_d, ring_d, WINDOW * LATENT, WINDOW, positions_d, lanes_d, begin_d, indices_d, slot_d,
			pair_sink_d, scale, (uint16_t *)0, frequency_d, LATENT - 2u * D41K_ROPE_PAIRS, group_heads, pair_codes_d, rows, rows, 0u, 0, part_elements, 0u,
			bytes)));
		CUDA(cudaDeviceSynchronize());
		std::vector<uint16_t> single_out(2u * part_elements), pair_out(2u * part_elements);
		std::vector<uint8_t> single_codes(2u * bytes), pair_codes(2u * bytes);
		CUDA(cudaMemcpy(single_out.data(), single_out_d, single_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		CUDA(cudaMemcpy(pair_out.data(), pair_out_d, pair_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		CUDA(cudaMemcpy(single_codes.data(), single_codes_d, single_codes.size(), cudaMemcpyDeviceToHost));
		CUDA(cudaMemcpy(pair_codes.data(), pair_codes_d, pair_codes.size(), cudaMemcpyDeviceToHost));
		uint64_t outputs_differ = 0u, codes_differ = 0u;
		for ( size_t index = 0u; index < single_out.size(); ++index )
			outputs_differ += single_out[index] != pair_out[index] ? 1u : 0u;
		for ( size_t index = 0u; index < single_codes.size(); ++index )
			codes_differ += single_codes[index] != pair_codes[index] ? 1u : 0u;
		printf("attend two query tiles a pass lanes %u rows %u: %llu of %llu outputs and %llu of %llu code bytes differ from one tile a launch\n", w.lanes,
			rows, (unsigned long long)outputs_differ, (unsigned long long)single_out.size(), (unsigned long long)codes_differ,
			(unsigned long long)single_codes.size());
		coded_mismatch += outputs_differ + codes_differ != 0u ? 1u : 0u;
		cudaFree(pair_query_d); cudaFree(single_out_d); cudaFree(pair_out_d); cudaFree(pair_sink_d); cudaFree(single_codes_d); cudaFree(pair_codes_d);
	}
	cudaFree(frequency_d); cudaFree(codes_d); cudaFree(reference_d); cudaFree(roped_d);
	std::vector<uint16_t> expected((size_t)packed * LATENT), merged((size_t)packed * LATENT);
	CUDA(cudaMemcpy(expected.data(), expected_d, expected.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpy(merged.data(), merged_d, merged.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	double worst = 0.0, total = 0.0;
	uint64_t same = 0u;
	for ( size_t index = 0u; index < expected.size(); ++index )
	{
		const double difference = fabs((double)FromBf16(expected[index]) - (double)FromBf16(merged[index]));
		worst = difference > worst ? difference : worst;
		total += difference;
		same += expected[index] == merged[index] ? 1u : 0u;
	}
	const double mean = total / (double)expected.size();
	cudaFree(query_d); cudaFree(wave_d); cudaFree(ring_d); cudaFree(positions_d); cudaFree(lanes_d); cudaFree(ordinals_d); cudaFree(begin_d);
	cudaFree(indices_d); cudaFree(slot_d); cudaFree(sink_d); cudaFree(sequential_d); cudaFree(expected_d); cudaFree(merged_d);
	printf("attend lanes %u rows %u+%u first %u/%u heads %u%s: worst %.5f mean %.6f, %llu of %llu outputs the same bits\n", w.lanes, w.count[0],
		w.lanes > 1u ? w.count[1] : 0u, w.first[0], w.first[1], heads, perturb != 0u ? " (perturbed key)" : "", worst, mean,
		(unsigned long long)same, (unsigned long long)expected.size());
	return worst > WAVE_WORST || mean > WAVE_MEAN ? 1u : 0u;
}

static uint32_t InitCase(uint32_t rows, uint32_t heads, uint32_t destinations, uint32_t pad)
{
	const uint64_t span = (uint64_t)rows * heads * D41K_RECORD, stride = span + pad, total = stride * destinations;
	std::vector<uint32_t> host(total);
	float *device = 0;
	uint32_t wrong = 0u;
	CUDA(cudaMalloc(&device, total * sizeof(float)));
	CUDA(cudaMemset(device, 0x7f, total * sizeof(float)));
	D41InitRecords(device, stride, rows, heads, destinations, 0);
	CUDA(cudaGetLastError());
	CUDA(cudaDeviceSynchronize());
	CUDA(cudaMemcpy(host.data(), device, total * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA(cudaFree(device));
	for ( uint64_t index = 0u; index < total; ++index )
	{
		const uint64_t offset = index % stride;
		const uint32_t expected = offset >= span ? 0x7f7f7f7fu : offset % D41K_RECORD == 0u ? 0xff800000u : 0u;
		wrong += host[index] != expected ? 1u : 0u;
	}
	return wrong;
}

struct TreeSpec
{
	uint32_t rows, anchor, context, stride;
	std::vector<uint32_t> parent, positions, slots, fork_positions, slot_offsets;
};

static TreeSpec BuildTreeSpec(const std::vector<uint32_t> &parent, uint32_t anchor, uint32_t slot_grain)
{
	const uint32_t rows = (uint32_t)parent.size();
	TreeSpec spec;
	spec.rows = rows;
	spec.anchor = anchor;
	spec.parent = parent;
	spec.positions.resize(rows);
	spec.slots.resize(rows);
	spec.fork_positions.resize(rows);
	spec.slot_offsets.resize(rows);
	std::vector<uint32_t> first_child(rows, D41K_NO_ROW), depth(rows, 0u), branch(rows, 0u), fork_pos(rows, 0u);
	uint32_t branches = 0u, max_depth = 0u, extent = 0u;
	for ( uint32_t r = 1u; r < rows; ++r )
	{
		const uint32_t p = parent[r];
		depth[r] = depth[p] + 1u;
		if ( first_child[p] == D41K_NO_ROW )
		{
			first_child[p] = r;
			branch[r] = branch[p];
			fork_pos[r] = fork_pos[p];
		}
		else
		{
			branches++;
			branch[r] = branches;
			fork_pos[r] = anchor + depth[r];
		}
		if ( depth[r] > max_depth )
			max_depth = depth[r];
	}
	spec.stride = slot_grain > 1u ? (max_depth + 1u + slot_grain - 1u) / slot_grain * slot_grain : max_depth + 1u;
	for ( uint32_t r = 0u; r < rows; ++r )
	{
		spec.positions[r] = anchor + depth[r];
		spec.slot_offsets[r] = branch[r] * spec.stride;
		spec.slots[r] = spec.positions[r] + spec.slot_offsets[r];
		spec.fork_positions[r] = branch[r] == 0u ? spec.positions[r] + 1u : fork_pos[r];
		if ( depth[r] + spec.slot_offsets[r] > extent )
			extent = depth[r] + spec.slot_offsets[r];
	}
	spec.context = anchor + extent + 1u;
	return spec;
}

static std::vector<uint32_t> TreePathToRow(const std::vector<uint32_t> &parent, uint32_t row)
{
	std::vector<uint32_t> rev;
	for ( uint32_t cur = row; cur != D41K_NO_ROW; cur = parent[cur] )
		rev.push_back(cur);
	return std::vector<uint32_t>(rev.rbegin(), rev.rend());
}

static uint32_t RunTreeNgramCase(uint32_t anchor)
{
	const uint32_t orders = 3u, heads = 8u, cols = orders * heads, vocab = 64u, pad = 7;
	const std::vector<uint32_t> parent = { D41K_NO_ROW, 0u, 1u, 2u, 3u, 0u, 5u, 6u, 1u, 8u };
	const TreeSpec tree = BuildTreeSpec(parent, anchor, 8u);
	std::vector<int32_t> token_map(vocab);
	for ( uint32_t t = 0u; t < vocab; ++t )
		token_map[t] = (t % 11u == 0u) ? -1 : (int32_t)(100u + (t * 7u) % 53u);
	std::vector<int64_t> multipliers = { 1315423911ll, 2654435761ll, 2246822519ll, 3266489917ll };
	std::vector<int64_t> primes(cols), offsets(cols);
	for ( uint32_t c = 0u; c < cols; ++c )
	{
		primes[c] = 1009ll + (int64_t)c * 37ll;
		offsets[c] = (int64_t)c * 2048ll;
	}
	std::vector<uint32_t> tokens(tree.rows);
	for ( uint32_t r = 0u; r < tree.rows; ++r )
		tokens[r] = (uint32_t)((Mix(r + anchor * 19u + 7u)) % vocab);
	std::vector<int32_t> init_hist = { 112, -1, 141, 0 };
	if ( anchor < 3u )
		init_hist = { 103, 109, 117, 0 };
	int32_t *map_d = Device(token_map), *hist_d = Device(init_hist);
	int64_t *mult_d = Device(multipliers), *prime_d = Device(primes), *off_d = Device(offsets);
	uint32_t *tok_d = Device(tokens), *pos_d = Device(tree.positions), *par_d = Device(tree.parent);
	std::vector<uint32_t> slot_v = { 0u };
	uint32_t *slot_d = Device(slot_v);
	int64_t *tree_ids_d = Scratch<int64_t>((size_t)tree.rows * cols);
	D41TreeNgramHashKernel<<<1, 64>>>(tok_d, map_d, pos_d, par_d, slot_d, hist_d, 4u, mult_d, prime_d, off_d, tree_ids_d, tree.rows, orders, heads, pad);
	CUDA(cudaDeviceSynchronize());
	std::vector<int64_t> tree_ids((size_t)tree.rows * cols);
	CUDA(cudaMemcpy(tree_ids.data(), tree_ids_d, tree_ids.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
	uint64_t mismatches = 0u;
	uint32_t *seq_tok_d = Scratch<uint32_t>(1u), *seq_pos_d = Scratch<uint32_t>(1u), *seq_lane_d = Device(std::vector<uint32_t>{ 0u });
	uint32_t *seq_ord_d = Device(std::vector<uint32_t>{ 0u }), *seq_beg_d = Device(std::vector<uint32_t>{ 0u, 1u }), *seq_idx_d = Device(std::vector<uint32_t>{ 0u });
	int32_t *seq_hist_d = Scratch<int32_t>(4u);
	int64_t *seq_ids_d = Scratch<int64_t>(cols);
	for ( uint32_t r = 0u; r < tree.rows; ++r )
	{
		const std::vector<uint32_t> path = TreePathToRow(tree.parent, r);
		CUDA(cudaMemcpy(seq_hist_d, init_hist.data(), 4u * sizeof(int32_t), cudaMemcpyHostToDevice));
		std::vector<int64_t> last_ids(cols);
		for ( uint32_t step = 0u; step < path.size(); ++step )
		{
			const uint32_t pr = path[step];
			CUDA(cudaMemcpy(seq_tok_d, &tokens[pr], sizeof(uint32_t), cudaMemcpyHostToDevice));
			CUDA(cudaMemcpy(seq_pos_d, &tree.positions[pr], sizeof(uint32_t), cudaMemcpyHostToDevice));
			LmNgramHashKernel<<<1, 64>>>(seq_tok_d, map_d, seq_pos_d, seq_lane_d, seq_ord_d, seq_beg_d, seq_idx_d, slot_d, seq_hist_d, 4u, mult_d, prime_d, off_d,
				seq_ids_d, 1u, orders, heads, pad, -1, 0u);
			LmNgramHistoryKernel<<<1, 64>>>(seq_tok_d, map_d, seq_beg_d, seq_idx_d, slot_d, seq_hist_d, 4u, 1u, orders, 1u, (const uint32_t *)0);
		}
		CUDA(cudaDeviceSynchronize());
		CUDA(cudaMemcpy(last_ids.data(), seq_ids_d, cols * sizeof(int64_t), cudaMemcpyDeviceToHost));
		for ( uint32_t c = 0u; c < cols; ++c )
			mismatches += tree_ids[(size_t)r * cols + c] != last_ids[c] ? 1u : 0u;
	}
	const std::vector<std::vector<uint32_t>> accepts = { { 0u, 1u, 2u }, { 0u, 5u, 6u }, { 0u, 1u, 8u, 9u } };
	const uint32_t post_tokens[2] = { 17u, 22u };
	for ( const auto &acc : accepts )
	{
		const uint32_t kept = (uint32_t)acc.size();
		std::vector<uint32_t> pb = { 0u, kept }, kv = { kept };
		uint32_t *pb_d = Device(pb), *pr_d = Device(acc), *kv_d = Device(kv);
		CUDA(cudaMemcpy(hist_d, init_hist.data(), 4u * sizeof(int32_t), cudaMemcpyHostToDevice));
		LmNgramHistoryKernel<<<1, 64>>>(tok_d, map_d, pb_d, pr_d, slot_d, hist_d, 4u, 1u, orders, 1u, kv_d);
		CUDA(cudaMemcpy(seq_hist_d, init_hist.data(), 4u * sizeof(int32_t), cudaMemcpyHostToDevice));
		for ( uint32_t pr : acc )
		{
			CUDA(cudaMemcpy(seq_tok_d, &tokens[pr], sizeof(uint32_t), cudaMemcpyHostToDevice));
			LmNgramHistoryKernel<<<1, 64>>>(seq_tok_d, map_d, seq_beg_d, seq_idx_d, slot_d, seq_hist_d, 4u, 1u, orders, 1u, (const uint32_t *)0);
		}
		for ( uint32_t extra = 0u; extra < 2u; ++extra )
		{
			const uint32_t p = anchor + kept + extra;
			CUDA(cudaMemcpy(seq_tok_d, &post_tokens[extra], sizeof(uint32_t), cudaMemcpyHostToDevice));
			CUDA(cudaMemcpy(seq_pos_d, &p, sizeof(uint32_t), cudaMemcpyHostToDevice));
			LmNgramHashKernel<<<1, 64>>>(seq_tok_d, map_d, seq_pos_d, seq_lane_d, seq_ord_d, seq_beg_d, seq_idx_d, slot_d, hist_d, 4u, mult_d, prime_d, off_d,
				tree_ids_d, 1u, orders, heads, pad, -1, 0u);
			LmNgramHistoryKernel<<<1, 64>>>(seq_tok_d, map_d, seq_beg_d, seq_idx_d, slot_d, hist_d, 4u, 1u, orders, 1u, (const uint32_t *)0);
			LmNgramHashKernel<<<1, 64>>>(seq_tok_d, map_d, seq_pos_d, seq_lane_d, seq_ord_d, seq_beg_d, seq_idx_d, slot_d, seq_hist_d, 4u, mult_d, prime_d, off_d,
				seq_ids_d, 1u, orders, heads, pad, -1, 0u);
			LmNgramHistoryKernel<<<1, 64>>>(seq_tok_d, map_d, seq_beg_d, seq_idx_d, slot_d, seq_hist_d, 4u, 1u, orders, 1u, (const uint32_t *)0);
			CUDA(cudaDeviceSynchronize());
			std::vector<int64_t> a(cols), b(cols);
			CUDA(cudaMemcpy(a.data(), tree_ids_d, cols * sizeof(int64_t), cudaMemcpyDeviceToHost));
			CUDA(cudaMemcpy(b.data(), seq_ids_d, cols * sizeof(int64_t), cudaMemcpyDeviceToHost));
			for ( uint32_t c = 0u; c < cols; ++c )
				mismatches += a[c] != b[c] ? 1u : 0u;
		}
		cudaFree(pb_d); cudaFree(pr_d); cudaFree(kv_d);
	}
	cudaFree(map_d); cudaFree(hist_d); cudaFree(mult_d); cudaFree(prime_d); cudaFree(off_d);
	cudaFree(tok_d); cudaFree(pos_d); cudaFree(par_d); cudaFree(slot_d); cudaFree(tree_ids_d);
	cudaFree(seq_tok_d); cudaFree(seq_pos_d); cudaFree(seq_lane_d); cudaFree(seq_ord_d);
	cudaFree(seq_beg_d); cudaFree(seq_idx_d); cudaFree(seq_hist_d); cudaFree(seq_ids_d);
	printf("tree engram anchor %u: %llu hash mismatches across tree rows and post-keep decode\n", anchor, (unsigned long long)mismatches);
	return mismatches != 0u ? 1u : 0u;
}

template<uint32_t RATIO>
static uint32_t RunTreeCsa2Case(uint32_t anchor, uint32_t degree, uint32_t slot_grain)
{
	using Kv = LmKvGeometry<1280u, 256u / RATIO, true>;
	constexpr uint32_t GRAIN = 8u, INDEX_HEADS = 32u, INDEX_DIM = 128u, TOPK = 512u;
	const uint32_t heads = degree == 1u ? 16u : 64u, local_heads = heads / degree, ring_rows = WINDOW / degree;
	const float qk_scale = 1.0f / sqrtf((float)LATENT);
	const std::vector<uint32_t> parent = { D41K_NO_ROW, 0u, 1u, 2u, 3u, 0u, 5u, 6u, 1u, 8u, 2u, 10u };
	const TreeSpec tree = BuildTreeSpec(parent, anchor, slot_grain);
	const uint32_t max_context = tree.context + 8u, max_keys = max_context / RATIO + 4u;
	const uint32_t pages = (max_keys + Kv::kPageSlots - 1u) / Kv::kPageSlots;
	const uint32_t score_stride = ((SparkKvShardGatherKeys(SparkKvShard{ degree, 0u, GRAIN }, max_keys) + 63u) / 64u) * 64u;
	std::vector<uint32_t> table(pages);
	for ( uint32_t p = 0u; p < pages; ++p )
		table[p] = pages - 1u - p;
	const uint32_t total_pool_tokens = anchor + tree.rows + 2u;
	std::vector<float> all_c_kv((size_t)total_pool_tokens * LATENT), all_c_score((size_t)total_pool_tokens * LATENT);
	for ( size_t i = 0u; i < all_c_kv.size(); ++i )
	{
		all_c_kv[i] = (float)((int32_t)((Mix(i ^ (0x101u + anchor)) >> 40u) & 0xfffu) - 2048) / 512.0f;
		all_c_score[i] = (float)((int32_t)((Mix(i ^ (0x202u + RATIO)) >> 40u) & 0xfffu) - 2048) / 1024.0f;
	}
	std::vector<uint16_t> all_idx_key = Values((size_t)total_pool_tokens * INDEX_DIM, 0x303u + anchor);
	std::vector<uint16_t> all_wave = Values((size_t)total_pool_tokens * LATENT, 0x404u + anchor);
	std::vector<uint16_t> all_query = Values((size_t)(tree.rows + 2u) * heads * LATENT, 0x505u + anchor);
	std::vector<uint16_t> all_idx_q = Values((size_t)(tree.rows + 2u) * INDEX_HEADS * INDEX_DIM, 0x606u + anchor);
	std::vector<uint16_t> all_idx_w = Values((size_t)(tree.rows + 2u) * INDEX_HEADS, 0x707u + anchor);
	std::vector<float> sink(heads);
	for ( uint32_t h = 0u; h < heads; ++h )
		sink[h] = (float)((int32_t)(Mix(h ^ 0x808u) & 0xffu) - 128) / 32.0f;
	uint32_t *table_d = Device(table);
	float *sink_d = Device(sink);
	LmKvAccessError *err_d;
	CUDA(cudaMalloc((void **)&err_d, sizeof(LmKvAccessError)));
	LmKvAccessError err;
	LmKvAccessErrorReset(&err);
	CUDA(cudaMemcpy(err_d, &err, sizeof(err), cudaMemcpyHostToDevice));
	const uint64_t pool_bytes = SparkKvShardPoolBytes(SparkKvShard{ degree, 0u, GRAIN }, Kv::kPageSlots, 1280u, pages);
	std::vector<uint8_t *> base_pools(degree), work_pools(degree);
	std::vector<LmKvShardView> base_views(degree), base_idx_views(degree), work_views(degree), work_idx_views(degree);
	for ( uint32_t r = 0u; r < degree; ++r )
	{
		const SparkKvShard sh = { degree, r, GRAIN };
		CUDA(cudaMalloc((void **)&base_pools[r], pool_bytes));
		CUDA(cudaMemset(base_pools[r], 0, pool_bytes));
		CUDA(cudaMalloc((void **)&work_pools[r], pool_bytes));
		REQUIRE(LmKvShardViewInitialize<Kv>(&base_views[r], base_pools[r], table_d, pages, 1u, pages, err_d, sh) == 0);
		REQUIRE(LmKvShardViewInitialize<Kv>(&base_idx_views[r], base_pools[r] + LATENT * sizeof(uint16_t), table_d, pages, 1u, pages, err_d, sh) == 0);
		REQUIRE(LmKvShardViewInitialize<Kv>(&work_views[r], work_pools[r], table_d, pages, 1u, pages, err_d, sh) == 0);
		REQUIRE(LmKvShardViewInitialize<Kv>(&work_idx_views[r], work_pools[r] + LATENT * sizeof(uint16_t), table_d, pages, 1u, pages, err_d, sh) == 0);
	}
	uint16_t *base_rings_d = Scratch<uint16_t>((size_t)degree * ring_rows * LATENT);
	uint16_t *work_rings_d = Scratch<uint16_t>((size_t)degree * ring_rows * LATENT);
	CUDA(cudaMemset(base_rings_d, 0, (size_t)degree * ring_rows * LATENT * sizeof(uint16_t)));
	float *base_cstate_d = Scratch<float>(2u * LATENT), *work_cstate_d = Scratch<float>(2u * LATENT);
	CUDA(cudaMemset(base_cstate_d, 0, 2u * LATENT * sizeof(float)));
	float *all_c_kv_d = Device(all_c_kv), *all_c_score_d = Device(all_c_score);
	uint16_t *all_idx_key_d = Device(all_idx_key), *all_wave_d = Device(all_wave);
	uint16_t *all_query_d = Device(all_query), *all_idx_q_d = Device(all_idx_q), *all_idx_w_d = Device(all_idx_w);
	uint16_t *pooled_d = Scratch<uint16_t>((size_t)anchor * LATENT);
	uint32_t *emit_d = Scratch<uint32_t>(anchor + tree.rows);
	std::vector<uint32_t> pre_pos(anchor), pre_lane(anchor, 0u), pre_ord(anchor), pre_beg = { 0u, anchor }, pre_idx(anchor), seq0 = { 0u };
	for ( uint32_t i = 0u; i < anchor; ++i )
		pre_pos[i] = pre_ord[i] = pre_idx[i] = i;
	uint32_t *pre_pos_d = Device(pre_pos), *pre_lane_d = Device(pre_lane), *pre_ord_d = Device(pre_ord), *pre_beg_d = Device(pre_beg), *pre_idx_d = Device(pre_idx), *seq0_d = Device(seq0);
	D41CompressorRowsKernel<<<anchor, D41K_THREADS>>>(all_c_kv_d, all_c_score_d, pooled_d, emit_d, base_cstate_d, pre_pos_d, pre_lane_d, pre_ord_d, pre_beg_d, pre_idx_d, seq0_d, RATIO, 2u * LATENT, LATENT);
	if ( RATIO > 1u )
		D41CompressorSaveLaunch(anchor, D41K_THREADS, (SparkGpuStream)0, all_c_kv_d, all_c_score_d, base_cstate_d, pre_pos_d, pre_lane_d, pre_ord_d, pre_beg_d, seq0_d, anchor, RATIO, 2u * LATENT, LATENT, 1u, (const uint32_t *)0);
	for ( uint32_t r = 0u; r < degree; ++r )
	{
		D41StoreCompressedKernel<Kv><<<anchor, D41K_THREADS>>>(base_views[r], pooled_d, all_idx_key_d, emit_d, pre_lane_d, INDEX_DIM);
		D41WindowRingStoreKernel<<<anchor, D41K_THREADS>>>(all_wave_d, base_rings_d + (size_t)r * ring_rows * LATENT, pre_pos_d, pre_lane_d, pre_ord_d, pre_beg_d, seq0_d,
			(uint64_t)ring_rows * LATENT, ring_rows, WINDOW, degree, r, anchor, 1u, (const uint32_t *)0);
	}
	CUDA(cudaDeviceSynchronize());
	CUDA(cudaMemcpy(&err, err_d, sizeof(err), cudaMemcpyDeviceToHost));
	REQUIRE(err.error_code == LM_FRAME_ERROR_NONE);
	cudaFree(pooled_d); cudaFree(pre_pos_d); cudaFree(pre_lane_d); cudaFree(pre_ord_d); cudaFree(pre_beg_d); cudaFree(pre_idx_d);
	auto cloneBase = [&]()
	{
		for ( uint32_t r = 0u; r < degree; ++r )
			CUDA(cudaMemcpy(work_pools[r], base_pools[r], pool_bytes, cudaMemcpyDeviceToDevice));
		CUDA(cudaMemcpy(work_rings_d, base_rings_d, (size_t)degree * ring_rows * LATENT * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
		CUDA(cudaMemcpy(work_cstate_d, base_cstate_d, 2u * LATENT * sizeof(float), cudaMemcpyDeviceToDevice));
	};
	cloneBase();
	std::vector<uint32_t> t_lane(tree.rows, 0u), t_ord(tree.rows), t_beg = { 0u, tree.rows }, t_idx(tree.rows), t_ctx = { tree.context };
	for ( uint32_t r = 0u; r < tree.rows; ++r )
		t_ord[r] = t_idx[r] = r;
	uint32_t *t_pos_d = Device(tree.positions), *t_slots_d = Device(tree.slots), *t_fork_d = Device(tree.fork_positions), *t_off_d = Device(tree.slot_offsets), *t_par_d = Device(tree.parent);
	uint32_t *t_lane_d = Device(t_lane), *t_ord_d = Device(t_ord), *t_beg_d = Device(t_beg), *t_idx_d = Device(t_idx), *t_ctx_d = Device(t_ctx);
	uint32_t *t_bound_d = Scratch<uint32_t>(tree.rows), *t_len_d = Scratch<uint32_t>(1u), *t_cfork_d = Scratch<uint32_t>(tree.rows), *t_coff_d = Scratch<uint32_t>(tree.rows);
	uint16_t *t_pooled_d = Scratch<uint16_t>((size_t)tree.rows * LATENT);
	D41CompressedEmitPositionLaunch(1, 256, (SparkGpuStream)0, t_pos_d, t_bound_d, tree.rows, RATIO);
	D41CompressedLengthLaunch(1, 256, (SparkGpuStream)0, t_ctx_d, t_len_d, 1u, RATIO);
	D41CompressedTreeMapLaunch(1, 256, (SparkGpuStream)0, t_fork_d, t_off_d, t_cfork_d, t_coff_d, tree.rows, RATIO);
	const float *t_ckv = all_c_kv_d + (size_t)anchor * LATENT, *t_cscore = all_c_score_d + (size_t)anchor * LATENT;
	const uint16_t *t_ikey = all_idx_key_d + (size_t)anchor * INDEX_DIM, *t_wave = all_wave_d + (size_t)anchor * LATENT;
	D41CompressorRowsKernel<<<tree.rows, D41K_THREADS>>>(t_ckv, t_cscore, t_pooled_d, emit_d, work_cstate_d, t_pos_d, t_lane_d, t_ord_d, t_beg_d, t_idx_d, seq0_d, RATIO, 2u * LATENT, LATENT, t_slots_d, t_par_d);
	for ( uint32_t r = 0u; r < degree; ++r )
		D41StoreCompressedKernel<Kv><<<tree.rows, D41K_THREADS>>>(work_views[r], t_pooled_d, t_ikey, emit_d, t_lane_d, INDEX_DIM);
	float *t_scores_d = Scratch<float>((size_t)degree * tree.rows * score_stride);
	const uint64_t t_rec_stride = (uint64_t)tree.rows * local_heads * D41K_RECORD;
	const uint64_t t_q_stride = (uint64_t)tree.rows * local_heads * LATENT;
	uint16_t *t_q_rank_d = Scratch<uint16_t>((size_t)degree * t_q_stride);
	std::vector<uint16_t> t_q_rank((size_t)degree * t_q_stride);
	for ( uint32_t r = 0u; r < degree; ++r )
		for ( uint32_t row = 0u; row < tree.rows; ++row )
			for ( uint32_t h = 0u; h < local_heads; ++h )
				memcpy(&t_q_rank[(size_t)r * t_q_stride + ((size_t)row * local_heads + h) * LATENT],
					&all_query[((size_t)row * heads + r * local_heads + h) * LATENT], LATENT * sizeof(uint16_t));
	CUDA(cudaMemcpy(t_q_rank_d, t_q_rank.data(), t_q_rank.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
	float *t_recs_d = Scratch<float>((size_t)degree * degree * t_rec_stride);
	float *t_recv_d = Scratch<float>((size_t)degree * degree * t_rec_stride);
	uint16_t *t_out_d = Scratch<uint16_t>((size_t)degree * tree.rows * local_heads * LATENT);
	std::vector<uint32_t> t_sel((size_t)tree.rows * TOPK);
	for ( uint32_t row = 0u; row < tree.rows; ++row )
	{
		const uint32_t vis = (tree.positions[row] + 1u) / RATIO;
		for ( uint32_t k = 0u; k < TOPK; ++k )
			t_sel[(size_t)row * TOPK + k] = vis != 0u ? (k * 13u + row * 7u) % vis : 0u;
	}
	uint32_t *t_sel_d = Device(t_sel);
	for ( uint32_t r = 0u; r < degree; ++r )
	{
		REQUIRE((LmWeightedSparseScoreLaunch<Kv, INDEX_HEADS, INDEX_DIM, LmKvShardView, true>(all_idx_q_d, all_idx_w_d, work_idx_views[r], t_lane_d, t_len_d, t_bound_d,
			tree.rows, score_stride, 1.0f, t_scores_d + (size_t)r * tree.rows * score_stride, 0, t_cfork_d, t_coff_d)) == SPARK_GPU_SUCCESS);
		REQUIRE((LmLatentShardAttendHeadsLaunch<Kv, LmKvShardView, LATENT, 0u>(work_views[r], t_q_rank_d, t_q_stride, local_heads, heads, t_lane_d, t_len_d, t_bound_d,
			t_sel_d, TOPK, TOPK, qk_scale, t_recs_d + (size_t)r * degree * t_rec_stride, t_rec_stride, local_heads, 1u, tree.rows, 0,
			(const uint32_t *)0, (const uint32_t *)0, 4u, t_cfork_d, t_coff_d)) == SPARK_GPU_SUCCESS);
		D41WindowFoldKernel<<<dim3(tree.rows, heads), D41K_THREADS>>>(t_recs_d + (size_t)r * degree * t_rec_stride, t_rec_stride, local_heads, t_q_rank_d, t_q_stride,
			t_wave, work_rings_d + (size_t)r * ring_rows * LATENT, t_pos_d, t_lane_d, t_ord_d, t_beg_d, t_idx_d, seq0_d, (uint64_t)ring_rows * LATENT, ring_rows, WINDOW,
			degree, r, qk_scale, local_heads, 1u, t_par_d);
	}
	for ( uint32_t dst = 0u; dst < degree; ++dst )
		for ( uint32_t src = 0u; src < degree; ++src )
			CUDA(cudaMemcpy(t_recv_d + ((size_t)dst * degree + src) * t_rec_stride, t_recs_d + ((size_t)src * degree + dst) * t_rec_stride, t_rec_stride * sizeof(float), cudaMemcpyDeviceToDevice));
	for ( uint32_t dst = 0u; dst < degree; ++dst )
		D41MergeSinkKernel<<<dim3(tree.rows, local_heads), D41K_THREADS>>>(t_recv_d + (size_t)dst * degree * t_rec_stride, t_rec_stride, degree, local_heads,
			sink_d + dst * local_heads, t_out_d + (size_t)dst * tree.rows * local_heads * LATENT);
	uint16_t *t_attend_d = 0;
	if ( degree == 1u )
	{
		t_attend_d = Scratch<uint16_t>((size_t)tree.rows * heads * LATENT);
		REQUIRE((D41AttendLaunch<Kv, LATENT, 1u>(work_views[0], 1u, t_q_rank_d, t_lane_d, t_len_d, t_bound_d, t_sel_d, TOPK, t_wave, work_rings_d,
			(uint64_t)ring_rows * LATENT, ring_rows, t_pos_d, t_lane_d, t_beg_d, t_idx_d, seq0_d, sink_d, qk_scale, t_attend_d, (const float *)0, 0u, 0u,
			(uint8_t *)0, tree.rows, tree.rows, 0u, 0, 0u, 0u, 0u, t_cfork_d, t_coff_d, t_par_d)) == SPARK_GPU_SUCCESS);
	}
	CUDA(cudaDeviceSynchronize());
	CUDA(cudaMemcpy(&err, err_d, sizeof(err), cudaMemcpyDeviceToHost));
	REQUIRE(err.error_code == LM_FRAME_ERROR_NONE);
	std::vector<float> tree_scores((size_t)degree * tree.rows * score_stride);
	std::vector<uint16_t> tree_out((size_t)degree * tree.rows * local_heads * LATENT), tree_attend(degree == 1u ? (size_t)tree.rows * heads * LATENT : 1u);
	CUDA(cudaMemcpy(tree_scores.data(), t_scores_d, tree_scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpy(tree_out.data(), t_out_d, tree_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	if ( degree == 1u )
		CUDA(cudaMemcpy(tree_attend.data(), t_attend_d, tree_attend.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	uint64_t score_diffs = 0u, out_diffs = 0u, attend_diffs = 0u, post_diffs = 0u;
	uint32_t *s_pos_d = Scratch<uint32_t>(1u), *s_ctx_d = Scratch<uint32_t>(1u), *s_bound_d = Scratch<uint32_t>(1u), *s_len_d = Scratch<uint32_t>(1u);
	uint32_t *s_lane_d = Device(std::vector<uint32_t>{ 0u }), *s_ord_d = Device(std::vector<uint32_t>{ 0u }), *s_beg_d = Device(std::vector<uint32_t>{ 0u, 1u }), *s_idx_d = Device(std::vector<uint32_t>{ 0u });
	uint32_t *s_sel_d = Scratch<uint32_t>(TOPK);
	uint16_t *s_pooled_d = Scratch<uint16_t>(LATENT), *s_q_d = Scratch<uint16_t>((size_t)heads * LATENT), *s_out_d = Scratch<uint16_t>((size_t)heads * LATENT);
	uint16_t *s_attend_d = Scratch<uint16_t>((size_t)heads * LATENT);
	const uint64_t s_rec_stride = (uint64_t)local_heads * D41K_RECORD;
	float *s_scores_d = Scratch<float>((size_t)degree * score_stride), *s_recs_d = Scratch<float>((size_t)degree * degree * s_rec_stride), *s_recv_d = Scratch<float>((size_t)degree * degree * s_rec_stride);
	auto runOneSeq = [&](uint32_t tok_offset, uint32_t q_row, uint32_t pos, const uint32_t *sel_host, bool eval_out)
	{
		const uint32_t ctx = pos + 1u;
		CUDA(cudaMemcpy(s_pos_d, &pos, sizeof(uint32_t), cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(s_ctx_d, &ctx, sizeof(uint32_t), cudaMemcpyHostToDevice));
		D41CompressedEmitPositionLaunch(1, 64, (SparkGpuStream)0, s_pos_d, s_bound_d, 1u, RATIO);
		D41CompressedLengthLaunch(1, 64, (SparkGpuStream)0, s_ctx_d, s_len_d, 1u, RATIO);
		D41CompressorRowsKernel<<<1, D41K_THREADS>>>(all_c_kv_d + (size_t)tok_offset * LATENT, all_c_score_d + (size_t)tok_offset * LATENT, s_pooled_d, emit_d,
			work_cstate_d, s_pos_d, s_lane_d, s_ord_d, s_beg_d, s_idx_d, seq0_d, RATIO, 2u * LATENT, LATENT);
		if ( RATIO > 1u )
			D41CompressorSaveLaunch(1, D41K_THREADS, (SparkGpuStream)0, all_c_kv_d + (size_t)tok_offset * LATENT, all_c_score_d + (size_t)tok_offset * LATENT, work_cstate_d,
				s_pos_d, s_lane_d, s_ord_d, s_beg_d, seq0_d, 1u, RATIO, 2u * LATENT, LATENT, 1u, (const uint32_t *)0);
		for ( uint32_t r = 0u; r < degree; ++r )
			D41StoreCompressedKernel<Kv><<<1, D41K_THREADS>>>(work_views[r], s_pooled_d, all_idx_key_d + (size_t)tok_offset * INDEX_DIM, emit_d, s_lane_d, INDEX_DIM);
		if ( eval_out )
		{
			CUDA(cudaMemcpy(s_sel_d, sel_host, TOPK * sizeof(uint32_t), cudaMemcpyHostToDevice));
			CUDA(cudaMemcpy(s_q_d, all_query_d + (size_t)q_row * heads * LATENT, (size_t)heads * LATENT * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
			for ( uint32_t r = 0u; r < degree; ++r )
			{
				REQUIRE((LmWeightedSparseScoreLaunch<Kv, INDEX_HEADS, INDEX_DIM, LmKvShardView, true>(all_idx_q_d + (size_t)q_row * INDEX_HEADS * INDEX_DIM,
					all_idx_w_d + (size_t)q_row * INDEX_HEADS, work_idx_views[r], s_lane_d, s_len_d, s_bound_d, 1u, score_stride, 1.0f,
					s_scores_d + (size_t)r * score_stride, 0)) == SPARK_GPU_SUCCESS);
				REQUIRE((LmLatentShardAttendHeadsLaunch<Kv, LmKvShardView, LATENT, 0u>(work_views[r], s_q_d, (uint64_t)local_heads * LATENT, local_heads, heads,
					s_lane_d, s_len_d, s_bound_d, s_sel_d, TOPK, TOPK, qk_scale, s_recs_d + (size_t)r * degree * s_rec_stride, s_rec_stride, local_heads, 1u, 1u, 0)) == SPARK_GPU_SUCCESS);
				D41WindowFoldKernel<<<dim3(1u, heads), D41K_THREADS>>>(s_recs_d + (size_t)r * degree * s_rec_stride, s_rec_stride, local_heads, s_q_d,
					(uint64_t)local_heads * LATENT, all_wave_d + (size_t)tok_offset * LATENT, work_rings_d + (size_t)r * ring_rows * LATENT, s_pos_d, s_lane_d,
					s_ord_d, s_beg_d, s_idx_d, seq0_d, (uint64_t)ring_rows * LATENT, ring_rows, WINDOW, degree, r, qk_scale, local_heads, 1u);
			}
			for ( uint32_t dst = 0u; dst < degree; ++dst )
				for ( uint32_t src = 0u; src < degree; ++src )
					CUDA(cudaMemcpy(s_recv_d + ((size_t)dst * degree + src) * s_rec_stride, s_recs_d + ((size_t)src * degree + dst) * s_rec_stride, s_rec_stride * sizeof(float), cudaMemcpyDeviceToDevice));
			for ( uint32_t dst = 0u; dst < degree; ++dst )
				D41MergeSinkKernel<<<dim3(1u, local_heads), D41K_THREADS>>>(s_recv_d + (size_t)dst * degree * s_rec_stride, s_rec_stride, degree, local_heads,
					sink_d + dst * local_heads, s_out_d + (size_t)dst * local_heads * LATENT);
			if ( degree == 1u )
				REQUIRE((D41AttendLaunch<Kv, LATENT, 1u>(work_views[0], 1u, s_q_d, s_lane_d, s_len_d, s_bound_d, s_sel_d, TOPK, all_wave_d + (size_t)tok_offset * LATENT,
					work_rings_d, (uint64_t)ring_rows * LATENT, ring_rows, s_pos_d, s_lane_d, s_beg_d, s_idx_d, seq0_d, sink_d, qk_scale, s_attend_d,
					(const float *)0, 0u, 0u, (uint8_t *)0, 1u, 1u, 0u, 0)) == SPARK_GPU_SUCCESS);
		}
		for ( uint32_t r = 0u; r < degree; ++r )
			D41WindowRingStoreKernel<<<1, D41K_THREADS>>>(all_wave_d + (size_t)tok_offset * LATENT, work_rings_d + (size_t)r * ring_rows * LATENT, s_pos_d, s_lane_d, s_ord_d,
				s_beg_d, seq0_d, (uint64_t)ring_rows * LATENT, ring_rows, WINDOW, degree, r, 1u, 1u, (const uint32_t *)0);
	};
	for ( uint32_t row = 0u; row < tree.rows; ++row )
	{
		cloneBase();
		const std::vector<uint32_t> path = TreePathToRow(tree.parent, row);
		for ( uint32_t step = 0u; step < path.size(); ++step )
			runOneSeq(anchor + path[step], path[step], tree.positions[path[step]], &t_sel[(size_t)row * TOPK], step + 1u == path.size());
		CUDA(cudaDeviceSynchronize());
		std::vector<float> s_scores((size_t)degree * score_stride);
		std::vector<uint16_t> s_out((size_t)heads * LATENT), s_att((size_t)heads * LATENT);
		CUDA(cudaMemcpy(s_scores.data(), s_scores_d, s_scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA(cudaMemcpy(s_out.data(), s_out_d, s_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		for ( uint32_t r = 0u; r < degree; ++r )
			for ( uint32_t k = 0u; k < score_stride; ++k )
				score_diffs += memcmp(&tree_scores[((size_t)r * tree.rows + row) * score_stride + k], &s_scores[(size_t)r * score_stride + k], sizeof(float)) != 0 ? 1u : 0u;
		for ( uint32_t r = 0u; r < degree; ++r )
			for ( uint32_t e = 0u; e < local_heads * LATENT; ++e )
				out_diffs += tree_out[((size_t)r * tree.rows + row) * local_heads * LATENT + e] != s_out[(size_t)r * local_heads * LATENT + e] ? 1u : 0u;
		if ( degree == 1u )
		{
			CUDA(cudaMemcpy(s_att.data(), s_attend_d, s_att.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
			for ( uint32_t e = 0u; e < heads * LATENT; ++e )
				attend_diffs += tree_attend[(size_t)row * heads * LATENT + e] != s_att[e] ? 1u : 0u;
		}
	}
	const std::vector<std::vector<uint32_t>> accepts = { { 0u, 1u, 2u }, { 0u, 5u }, { 0u, 5u, 6u, 7u }, { 0u, 1u, 8u, 9u }, { 0u, 1u, 2u, 10u, 11u } };
	for ( const auto &acc : accepts )
	{
		const uint32_t kept = (uint32_t)acc.size();
		cloneBase();
		D41CompressorRowsKernel<<<tree.rows, D41K_THREADS>>>(t_ckv, t_cscore, t_pooled_d, emit_d, work_cstate_d, t_pos_d, t_lane_d, t_ord_d, t_beg_d, t_idx_d, seq0_d, RATIO, 2u * LATENT, LATENT, t_slots_d, t_par_d);
		for ( uint32_t r = 0u; r < degree; ++r )
			D41StoreCompressedKernel<Kv><<<tree.rows, D41K_THREADS>>>(work_views[r], t_pooled_d, t_ikey, emit_d, t_lane_d, INDEX_DIM);
		uint32_t *pr_d = Device(acc), *kv_d = Device(std::vector<uint32_t>{ kept });
		for ( uint32_t r = 0u; r < degree; ++r )
		{
			D41WindowRingStoreKernel<<<tree.rows, D41K_THREADS>>>(t_wave, work_rings_d + (size_t)r * ring_rows * LATENT, t_pos_d, t_lane_d, t_ord_d, t_beg_d, seq0_d,
				(uint64_t)ring_rows * LATENT, ring_rows, WINDOW, degree, r, tree.rows, 1u, kv_d, pr_d);
			D41CompressedPathMoveKernel<Kv, D41K_THREADS><<<tree.rows, D41K_THREADS>>>(work_views[r], t_lane_d, t_pos_d, pr_d, kv_d, t_slots_d, RATIO, 1280u / 2u);
		}
		if ( RATIO > 1u )
			D41CompressorSaveLaunch(tree.rows, D41K_THREADS, (SparkGpuStream)0, t_ckv, t_cscore, work_cstate_d, t_pos_d, t_lane_d, t_ord_d, t_beg_d, seq0_d, tree.rows, RATIO, 2u * LATENT, LATENT, 1u, kv_d, pr_d);
		std::vector<std::vector<float>> keep_scores(2u);
		std::vector<std::vector<uint16_t>> keep_outs(2u);
		std::vector<std::vector<uint32_t>> extra_sels(2u, std::vector<uint32_t>(TOPK));
		for ( uint32_t extra = 0u; extra < 2u; ++extra )
		{
			const uint32_t p = anchor + kept + extra, vis = (p + 1u) / RATIO;
			for ( uint32_t k = 0u; k < TOPK; ++k )
				extra_sels[extra][k] = vis != 0u ? (k * 17u + extra * 5u) % vis : 0u;
			runOneSeq(anchor + tree.rows + extra, tree.rows + extra, p, extra_sels[extra].data(), true);
			CUDA(cudaDeviceSynchronize());
			keep_scores[extra].resize((size_t)degree * score_stride);
			keep_outs[extra].resize((size_t)heads * LATENT);
			CUDA(cudaMemcpy(keep_scores[extra].data(), s_scores_d, keep_scores[extra].size() * sizeof(float), cudaMemcpyDeviceToHost));
			CUDA(cudaMemcpy(keep_outs[extra].data(), s_out_d, keep_outs[extra].size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		}
		cloneBase();
		for ( uint32_t pr : acc )
			runOneSeq(anchor + pr, pr, tree.positions[pr], extra_sels[0].data(), false);
		for ( uint32_t extra = 0u; extra < 2u; ++extra )
		{
			const uint32_t p = anchor + kept + extra;
			runOneSeq(anchor + tree.rows + extra, tree.rows + extra, p, extra_sels[extra].data(), true);
			CUDA(cudaDeviceSynchronize());
			std::vector<float> ref_scores((size_t)degree * score_stride);
			std::vector<uint16_t> ref_outs((size_t)heads * LATENT);
			CUDA(cudaMemcpy(ref_scores.data(), s_scores_d, ref_scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
			CUDA(cudaMemcpy(ref_outs.data(), s_out_d, ref_outs.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
			for ( size_t i = 0u; i < ref_scores.size(); ++i )
				post_diffs += memcmp(&keep_scores[extra][i], &ref_scores[i], sizeof(float)) != 0 ? 1u : 0u;
			for ( size_t i = 0u; i < ref_outs.size(); ++i )
				post_diffs += keep_outs[extra][i] != ref_outs[i] ? 1u : 0u;
		}
		cudaFree(pr_d); cudaFree(kv_d);
	}
	CUDA(cudaMemcpy(&err, err_d, sizeof(err), cudaMemcpyDeviceToHost));
	REQUIRE(err.error_code == LM_FRAME_ERROR_NONE);
	for ( uint32_t r = 0u; r < degree; ++r )
	{
		cudaFree(base_pools[r]);
		cudaFree(work_pools[r]);
	}
	cudaFree(table_d); cudaFree(sink_d); cudaFree(err_d); cudaFree(base_rings_d); cudaFree(work_rings_d);
	cudaFree(base_cstate_d); cudaFree(work_cstate_d); cudaFree(all_c_kv_d); cudaFree(all_c_score_d);
	cudaFree(all_idx_key_d); cudaFree(all_wave_d); cudaFree(all_query_d); cudaFree(all_idx_q_d); cudaFree(all_idx_w_d);
	cudaFree(emit_d); cudaFree(seq0_d); cudaFree(t_pos_d); cudaFree(t_slots_d); cudaFree(t_fork_d); cudaFree(t_off_d); cudaFree(t_par_d);
	cudaFree(t_lane_d); cudaFree(t_ord_d); cudaFree(t_beg_d); cudaFree(t_idx_d); cudaFree(t_ctx_d); cudaFree(t_bound_d); cudaFree(t_len_d);
	cudaFree(t_cfork_d); cudaFree(t_coff_d); cudaFree(t_pooled_d); cudaFree(t_scores_d); cudaFree(t_q_rank_d); cudaFree(t_recs_d);
	cudaFree(t_recv_d); cudaFree(t_out_d); cudaFree(t_sel_d); if ( t_attend_d != 0 ) cudaFree(t_attend_d);
	cudaFree(s_pos_d); cudaFree(s_ctx_d); cudaFree(s_bound_d); cudaFree(s_len_d); cudaFree(s_lane_d); cudaFree(s_ord_d); cudaFree(s_beg_d);
	cudaFree(s_idx_d); cudaFree(s_sel_d); cudaFree(s_pooled_d); cudaFree(s_q_d); cudaFree(s_out_d); cudaFree(s_attend_d);
	cudaFree(s_scores_d); cudaFree(s_recs_d); cudaFree(s_recv_d);
	printf("tree csa2 ratio %u degree %u grain %u anchor %u: score_diffs %llu out_diffs %llu attend_diffs %llu post_keep_diffs %llu\n",
		RATIO, degree, slot_grain, anchor, (unsigned long long)score_diffs, (unsigned long long)out_diffs,
		(unsigned long long)attend_diffs, (unsigned long long)post_diffs);
	return (score_diffs + out_diffs + attend_diffs + post_diffs) != 0u ? 1u : 0u;
}

static uint32_t RunTreeGraphCase(void)
{
	using Kv = LmKvGeometry<1280u, 128u, true>;
	constexpr uint32_t ROWS = 12u, HEADS = 16u, INDEX_HEADS = 32u, INDEX_DIM = 128u, TOPK = 512u, ORDERS = 3u, EHEADS = 8u, ECOLS = ORDERS * EHEADS;
	const uint32_t anchor = 201u, pages = 8u, score_stride = 512u;
	const float qk_scale = 1.0f / sqrtf((float)LATENT);
	const std::vector<uint32_t> parent_a = { D41K_NO_ROW, 0u, 1u, 2u, 3u, 0u, 5u, 6u, 1u, 8u, 2u, 10u };
	const std::vector<uint32_t> parent_b = { D41K_NO_ROW, 0u, 1u, 2u, 0u, 4u, 5u, 6u, 2u, 8u, 1u, 10u };
	const std::vector<uint32_t> parent_chain = { D41K_NO_ROW, 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u };
	const TreeSpec tree_a = BuildTreeSpec(parent_a, anchor, 8u), tree_b = BuildTreeSpec(parent_b, anchor + 1u, 8u), chain = BuildTreeSpec(parent_chain, anchor, 8u);
	std::vector<uint32_t> table(pages);
	for ( uint32_t p = 0u; p < pages; ++p )
		table[p] = p;
	uint32_t *table_d = Device(table);
	LmKvAccessError *err_d;
	CUDA(cudaMalloc((void **)&err_d, sizeof(LmKvAccessError)));
	LmKvAccessError err;
	LmKvAccessErrorReset(&err);
	CUDA(cudaMemcpy(err_d, &err, sizeof(err), cudaMemcpyHostToDevice));
	uint8_t *pool_d;
	CUDA(cudaMalloc((void **)&pool_d, (uint64_t)pages * Kv::kPageBytes));
	CUDA(cudaMemset(pool_d, 0, (uint64_t)pages * Kv::kPageBytes));
	LmKvShardView view, idx_view;
	REQUIRE(LmKvShardViewInitialize<Kv>(&view, pool_d, table_d, pages, 1u, pages, err_d, SparkKvShard{ 1u, 0u, 8u }) == 0);
	REQUIRE(LmKvShardViewInitialize<Kv>(&idx_view, pool_d + LATENT * sizeof(uint16_t), table_d, pages, 1u, pages, err_d, SparkKvShard{ 1u, 0u, 8u }) == 0);
	std::vector<float> ckv(ROWS * LATENT, 0.25f), cscore(ROWS * LATENT, 0.5f), cstate(2u * LATENT, 0.125f), sink(HEADS, -0.25f);
	for ( size_t i = 0u; i < ckv.size(); ++i )
		ckv[i] = (float)((int32_t)((Mix(i ^ 0x111u) >> 40u) & 0xfffu) - 2048) / 512.0f;
	std::vector<uint16_t> ikey = Values(ROWS * INDEX_DIM, 0x222u), wave = Values(ROWS * LATENT, 0x333u), ring = Values(WINDOW * LATENT, 0x444u);
	std::vector<uint16_t> query = Values(ROWS * HEADS * LATENT, 0x555u), iq = Values(ROWS * INDEX_HEADS * INDEX_DIM, 0x666u), iw = Values(ROWS * INDEX_HEADS, 0x777u);
	std::vector<uint32_t> lane(ROWS, 0u), ord(ROWS), beg = { 0u, ROWS }, idx(ROWS), seq0 = { 0u }, sel(ROWS * TOPK, 0u), tok(ROWS);
	for ( uint32_t r = 0u; r < ROWS; ++r )
	{
		ord[r] = idx[r] = r;
		tok[r] = 3u + r * 5u;
	}
	std::vector<int32_t> hist = { 10, 20, 30, 0 };
	std::vector<int64_t> mult = { 1315423911ll, 2654435761ll, 2246822519ll, 3266489917ll }, primes(ECOLS, 1009ll), offs(ECOLS, 0ll);
	float *ckv_d = Device(ckv), *cscore_d = Device(cscore), *cstate_d = Device(cstate), *sink_d = Device(sink);
	uint16_t *ikey_d = Device(ikey), *wave_d = Device(wave), *ring_d = Device(ring), *query_d = Device(query), *iq_d = Device(iq), *iw_d = Device(iw);
	uint32_t *lane_d = Device(lane), *ord_d = Device(ord), *beg_d = Device(beg), *idx_d = Device(idx), *seq0_d = Device(seq0), *sel_d = Device(sel), *tok_d = Device(tok);
	int32_t *hist_d = Device(hist);
	int64_t *mult_d = Device(mult), *primes_d = Device(primes), *offs_d = Device(offs), *ids_d = Scratch<int64_t>(ROWS * ECOLS);
	uint32_t *pos_d = Scratch<uint32_t>(ROWS), *slots_d = Scratch<uint32_t>(ROWS), *fork_d = Scratch<uint32_t>(ROWS), *off_d = Scratch<uint32_t>(ROWS), *par_d = Scratch<uint32_t>(ROWS);
	uint32_t *bound_d = Scratch<uint32_t>(ROWS), *ctx_d = Scratch<uint32_t>(1u), *len_d = Scratch<uint32_t>(1u), *cfork_d = Scratch<uint32_t>(ROWS), *coff_d = Scratch<uint32_t>(ROWS), *emit_d = Scratch<uint32_t>(ROWS);
	uint16_t *pooled_d = Scratch<uint16_t>(ROWS * LATENT), *out_d = Scratch<uint16_t>(ROWS * HEADS * LATENT);
	const uint64_t rec_stride = (uint64_t)ROWS * HEADS * D41K_RECORD;
	float *scores_d = Scratch<float>(ROWS * score_stride), *recs_d = Scratch<float>(rec_stride);
	cudaStream_t stream;
	CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	auto runPipeline = [&](bool is_tree)
	{
		D41CompressedEmitPositionLaunch(1, 256, stream, pos_d, bound_d, ROWS, 2u);
		D41CompressedLengthLaunch(1, 256, stream, ctx_d, len_d, 1u, 2u);
		if ( is_tree )
		{
			D41CompressedTreeMapLaunch(1, 256, stream, fork_d, off_d, cfork_d, coff_d, ROWS, 2u);
			D41TreeNgramHashKernel<<<1, 64, 0, stream>>>(tok_d, (const int32_t *)0, pos_d, par_d, seq0_d, hist_d, 4u, mult_d, primes_d, offs_d, ids_d, ROWS, ORDERS, EHEADS, 7);
		}
		else
			LmNgramHashKernel<<<1, 64, 0, stream>>>(tok_d, (const int32_t *)0, pos_d, lane_d, ord_d, beg_d, idx_d, seq0_d, hist_d, 4u, mult_d, primes_d, offs_d, ids_d, ROWS, ORDERS, EHEADS, 7, -1, 0u);
		D41CompressorRowsKernel<<<ROWS, D41K_THREADS, 0, stream>>>(ckv_d, cscore_d, pooled_d, emit_d, cstate_d, pos_d, lane_d, ord_d, beg_d, idx_d, seq0_d, 2u, 2u * LATENT, LATENT,
			is_tree ? slots_d : (const uint32_t *)0, is_tree ? par_d : (const uint32_t *)0);
		D41StoreCompressedKernel<Kv><<<ROWS, D41K_THREADS, 0, stream>>>(view, pooled_d, ikey_d, emit_d, lane_d, INDEX_DIM);
		REQUIRE((LmWeightedSparseScoreLaunch<Kv, INDEX_HEADS, INDEX_DIM, LmKvShardView, true>(iq_d, iw_d, idx_view, lane_d, len_d, bound_d, ROWS, score_stride, 1.0f,
			scores_d, stream, is_tree ? cfork_d : (const uint32_t *)0, is_tree ? coff_d : (const uint32_t *)0)) == SPARK_GPU_SUCCESS);
		REQUIRE((LmLatentShardAttendHeadsLaunch<Kv, LmKvShardView, LATENT, 0u>(view, query_d, (uint64_t)ROWS * HEADS * LATENT, HEADS, HEADS, lane_d, len_d, bound_d,
			sel_d, TOPK, TOPK, qk_scale, recs_d, rec_stride, HEADS, 1u, ROWS, stream, (const uint32_t *)0, (const uint32_t *)0, 4u,
			is_tree ? cfork_d : (const uint32_t *)0, is_tree ? coff_d : (const uint32_t *)0)) == SPARK_GPU_SUCCESS);
		D41WindowFoldKernel<<<dim3(ROWS, HEADS), D41K_THREADS, 0, stream>>>(recs_d, rec_stride, HEADS, query_d, (uint64_t)ROWS * HEADS * LATENT, wave_d, ring_d,
			pos_d, lane_d, ord_d, beg_d, idx_d, seq0_d, (uint64_t)WINDOW * LATENT, WINDOW, WINDOW, 1u, 0u, qk_scale, HEADS, 1u, is_tree ? par_d : (const uint32_t *)0);
		D41MergeSinkKernel<<<dim3(ROWS, HEADS), D41K_THREADS, 0, stream>>>(recs_d, rec_stride, 1u, HEADS, sink_d, out_d);
	};
	auto loadSpec = [&](const TreeSpec &s)
	{
		CUDA(cudaMemcpyAsync(pos_d, s.positions.data(), ROWS * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
		CUDA(cudaMemcpyAsync(slots_d, s.slots.data(), ROWS * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
		CUDA(cudaMemcpyAsync(fork_d, s.fork_positions.data(), ROWS * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
		CUDA(cudaMemcpyAsync(off_d, s.slot_offsets.data(), ROWS * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
		CUDA(cudaMemcpyAsync(par_d, s.parent.data(), ROWS * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
		CUDA(cudaMemcpyAsync(ctx_d, &s.context, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
	};
	loadSpec(tree_a);
	runPipeline(true);
	CUDA(cudaStreamSynchronize(stream));
	cudaGraph_t tree_graph, chain_graph;
	cudaGraphExec_t tree_exec, chain_exec;
	CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
	runPipeline(true);
	CUDA(cudaStreamEndCapture(stream, &tree_graph));
	CUDA(cudaGraphInstantiate(&tree_exec, tree_graph, 0, 0, 0));
	loadSpec(chain);
	CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
	runPipeline(false);
	CUDA(cudaStreamEndCapture(stream, &chain_graph));
	CUDA(cudaGraphInstantiate(&chain_exec, chain_graph, 0, 0, 0));
	uint64_t diffs = 0u;
	const TreeSpec *specs[4] = { &tree_a, &chain, &tree_b, &chain };
	const bool is_trees[4] = { true, false, true, false };
	for ( uint32_t step = 0u; step < 4u; ++step )
	{
		loadSpec(*specs[step]);
		CUDA(cudaGraphLaunch(is_trees[step] ? tree_exec : chain_exec, stream));
		CUDA(cudaStreamSynchronize(stream));
		std::vector<int64_t> g_ids(ROWS * ECOLS), e_ids(ROWS * ECOLS);
		std::vector<float> g_scores(ROWS * score_stride), e_scores(ROWS * score_stride);
		std::vector<uint16_t> g_out(ROWS * HEADS * LATENT), e_out(ROWS * HEADS * LATENT);
		CUDA(cudaMemcpy(g_ids.data(), ids_d, g_ids.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
		CUDA(cudaMemcpy(g_scores.data(), scores_d, g_scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA(cudaMemcpy(g_out.data(), out_d, g_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		loadSpec(*specs[step]);
		runPipeline(is_trees[step]);
		CUDA(cudaStreamSynchronize(stream));
		CUDA(cudaMemcpy(e_ids.data(), ids_d, e_ids.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
		CUDA(cudaMemcpy(e_scores.data(), scores_d, e_scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA(cudaMemcpy(e_out.data(), out_d, e_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		for ( size_t i = 0u; i < g_ids.size(); ++i )
			diffs += g_ids[i] != e_ids[i] ? 1u : 0u;
		for ( size_t i = 0u; i < g_scores.size(); ++i )
			diffs += memcmp(&g_scores[i], &e_scores[i], sizeof(float)) != 0 ? 1u : 0u;
		for ( size_t i = 0u; i < g_out.size(); ++i )
			diffs += g_out[i] != e_out[i] ? 1u : 0u;
	}
	cudaGraphExecDestroy(tree_exec); cudaGraphDestroy(tree_graph);
	cudaGraphExecDestroy(chain_exec); cudaGraphDestroy(chain_graph);
	cudaStreamDestroy(stream);
	cudaFree(table_d); cudaFree(err_d); cudaFree(pool_d); cudaFree(ckv_d); cudaFree(cscore_d); cudaFree(cstate_d); cudaFree(sink_d);
	cudaFree(ikey_d); cudaFree(wave_d); cudaFree(ring_d); cudaFree(query_d); cudaFree(iq_d); cudaFree(iw_d);
	cudaFree(lane_d); cudaFree(ord_d); cudaFree(beg_d); cudaFree(idx_d); cudaFree(seq0_d); cudaFree(sel_d); cudaFree(tok_d);
	cudaFree(hist_d); cudaFree(mult_d); cudaFree(primes_d); cudaFree(offs_d); cudaFree(ids_d);
	cudaFree(pos_d); cudaFree(slots_d); cudaFree(fork_d); cudaFree(off_d); cudaFree(par_d);
	cudaFree(bound_d); cudaFree(ctx_d); cudaFree(len_d); cudaFree(cfork_d); cudaFree(coff_d); cudaFree(emit_d);
	cudaFree(pooled_d); cudaFree(out_d); cudaFree(scores_d); cudaFree(recs_d);
	printf("tree cuda graph alternating replay (0 recaptures, 4 steps): %llu bit diffs vs eager\n", (unsigned long long)diffs);
	return diffs != 0u ? 1u : 0u;
}

int main(void)
{
	const Case cases[] = {
		{1u, 64u, 4u, 4u, 1u, 0u, 9999u, RING * LATENT, 0u},
		{1u, 64u, 4u, 8u, 2u, 7u, 70000u, RING * LATENT, 0u},
		{6u, 64u, 4u, 8u, 2u, 15u, 4096u, RING * LATENT, 0u},
		{17u, 64u, 4u, 4u, 1u, 3u, 123457u, RING * LATENT, 0u},
		{1024u, 4u, 4u, 4u, 1u, 9u, 33000u, 0u, 0u},
		{256u, 4u, 4u, 4u, 1u, 0u, 5u, 0u, 0u},
		{3u, 64u, 4u, 4u, 1u, 14u, 2u, RING * LATENT, 0u},
		{1u, 16u, 16u, 16u, 1u, 0u, 9999u, WINDOW * LATENT, 1u},
		{6u, 16u, 16u, 16u, 1u, 0u, 70000u, WINDOW * LATENT, 1u},
		{300u, 16u, 16u, 16u, 1u, 0u, 5u, WINDOW * LATENT, 1u},
		{2u, 8u, 8u, 8u, 1u, 0u, 0u, WINDOW * LATENT, 1u},
		{1024u, 16u, 16u, 16u, 1u, 0u, 33000u, 0u, 1u},
		{300u, 16u, 16u, 16u, 1u, 0u, 5u, WINDOW * LATENT, 2u},
		{1024u, 16u, 16u, 16u, 1u, 0u, 33000u, 0u, 2u},
		{17u, 8u, 8u, 8u, 1u, 0u, 123457u, WINDOW * LATENT, 2u},
		{1u, 4u, 4u, 4u, 1u, 0u, 9999u, WINDOW * LATENT, 3u},
		{1u, 4u, 4u, 4u, 1u, 0u, 70000u, WINDOW * LATENT, 3u},
		{1u, 4u, 4u, 4u, 1u, 0u, 5u, WINDOW * LATENT, 3u},
		{6u, 16u, 16u, 16u, 1u, 0u, 70000u, WINDOW * LATENT, 3u},
		{2u, 8u, 8u, 8u, 1u, 0u, 0u, WINDOW * LATENT, 3u},
		{300u, 16u, 16u, 16u, 1u, 0u, 5u, WINDOW * LATENT, 3u},
		{1u, 16u, 16u, 16u, 1u, 0u, 9999u, WINDOW * LATENT, 3u},
		{1u, 16u, 16u, 16u, 1u, 0u, 70000u, WINDOW * LATENT, 1u},
		{1u, 16u, 16u, 16u, 1u, 0u, 70000u, WINDOW * LATENT, 3u},
	};
	uint32_t failures = 0u;
	for ( const Case &c : cases )
		failures += RunCase(c, 0u);
	const uint32_t sensitive = RunCase(cases[1], 1u) != 0u && RunCase(cases[8], 1u) != 0u && RunCase(cases[18], 1u) != 0u ? 1u : 0u;
	const Wave waves[] = {
		{1u, {17u, 0u}, {0u, 0u}, 16u},
		{1u, {128u, 0u}, {5u, 0u}, 16u},
		{1u, {1024u, 0u}, {33000u, 0u}, 16u},
		{2u, {300u, 77u}, {5u, 70000u}, 16u},
		{2u, {129u, 255u}, {126u, 127u}, 16u},
	};
	uint32_t waves_off = 0u;
	for ( const Wave &w : waves )
		waves_off += RunWave(w, 0u);
	const uint32_t wave_sensitive = RunWave(waves[3], 1u);
	const uint32_t unset = InitCase(2048u, 4u, 16u, 0u) + InitCase(4100u, 4u, 16u, 0u) + InitCase(3u, 8u, 16u, 37u);
	uint32_t tree_failures = 0u;
	tree_failures += RunTreeNgramCase(1u);
	tree_failures += RunTreeNgramCase(200u);
	tree_failures += RunTreeCsa2Case<2u>(110u, 16u, 256u);
	tree_failures += RunTreeCsa2Case<2u>(200u, 16u, 256u);
	tree_failures += RunTreeCsa2Case<2u>(201u, 16u, 256u);
	tree_failures += RunTreeCsa2Case<2u>(1200u, 16u, 256u);
	tree_failures += RunTreeCsa2Case<2u>(200u, 1u, 8u);
	tree_failures += RunTreeCsa2Case<2u>(201u, 1u, 8u);
	tree_failures += RunTreeCsa2Case<2u>(1200u, 1u, 8u);
	tree_failures += RunTreeCsa2Case<1u>(200u, 16u, 256u);
	tree_failures += RunTreeCsa2Case<1u>(201u, 16u, 256u);
	tree_failures += RunTreeCsa2Case<1u>(200u, 1u, 8u);
	tree_failures += RunTreeCsa2Case<1u>(600u, 1u, 8u);
	tree_failures += RunTreeGraphCase();
	if ( failures != 0u || sensitive == 0u || unset != 0u || waves_off != 0u || wave_sensitive == 0u || coded_mismatch != 0u || tree_failures != 0u )
	{
		printf("FAIL dsv41 window fold: %u cases differ from the sequential fold, perturbed keys %s, %u record words not initialized as -inf/0 or padding overwritten, %u fused waves off the sequential fold's attention output, a perturbed wave key %s, %u coded waves off rope then quantize or off one tile a launch, %u tree cases failed\n",
			failures, sensitive != 0u ? "detected" : "not detected", unset, waves_off, wave_sensitive != 0u ? "detected" : "not detected", coded_mismatch, tree_failures);
		return 1;
	}
	printf("PASS dsv41 window fold: batched scores fold every case to the same bits as the sequential fold, with the window cut sixteen ways and whole on one rank, one block a head or eight clustered blocks a head; the fused attention of a wave gives the sequential fold's attention output to within the weights' BF16 rounding for one and two sequences, shuffled rows, early positions and ring keys, and its MXFP8 output is the inverse rope and quantization of that output byte for byte, also over fewer live rows with the rest untouched, and two query tiles a pass give one tile a launch's outputs and codes bit for bit; record initialization sets every record of 2048- and 4100-row waves at TP16 to -inf/0 and leaves padding alone\n");
	return 0;
}
