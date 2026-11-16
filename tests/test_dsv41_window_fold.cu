#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "model-families/dsv41_flash/source/dsv41_flash_stage_kernels.cuh"
#include "model-families/dsv41_flash/source/dsv41_flash_attend.cuh"
#include "inference/kernels/mx_gemm.cuh"

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
		c.record_copies, c.rank, c.first_position, c.whole == 1u ? " (whole window, one pass)" : c.whole == 2u ? " (whole window, chunks)" : "",
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
	};
	uint32_t failures = 0u;
	for ( const Case &c : cases )
		failures += RunCase(c, 0u);
	const uint32_t sensitive = RunCase(cases[1], 1u) != 0u && RunCase(cases[8], 1u) != 0u ? 1u : 0u;
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
	if ( failures != 0u || sensitive == 0u || unset != 0u || waves_off != 0u || wave_sensitive == 0u || coded_mismatch != 0u )
	{
		printf("FAIL dsv41 window fold: %u cases differ from the sequential fold, perturbed keys %s, %u record words not initialized as -inf/0 or padding overwritten, %u fused waves off the sequential fold's attention output, a perturbed wave key %s, %u coded waves off rope then quantize\n",
			failures, sensitive != 0u ? "detected" : "not detected", unset, waves_off, wave_sensitive != 0u ? "detected" : "not detected", coded_mismatch);
		return 1;
	}
	printf("PASS dsv41 window fold: batched scores fold every case to the same bits as the sequential fold, with the window cut sixteen ways and whole on one rank; the fused attention of a wave gives the sequential fold's attention output to within the weights' BF16 rounding for one and two sequences, shuffled rows, early positions and ring keys, and its MXFP8 output is the inverse rope and quantization of that output byte for byte, also over fewer live rows with the rest untouched; record initialization sets every record of 2048- and 4100-row waves at TP16 to -inf/0 and leaves padding alone\n");
	return 0;
}
