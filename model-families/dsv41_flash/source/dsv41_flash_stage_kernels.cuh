#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/ngram_hash.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "dsv41_flash_kernels.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"

#define D41K_THREADS 256u
#define D41K_HC 4u
#define D41K_ROPE_PAIRS 32u
#define D41K_LATENT 512u
#define D41K_RECORD (D41K_LATENT + 2u)
#define D41K_NO_ROW 0xffffffffu
#define D41K_FOLD_KEYS 8u

static __global__ void D41FrequencyKernel(float *__restrict__ table, float theta, float inverse_factor, float low, float inverse_span,
	uint32_t yarn)
{
	const uint32_t pair = threadIdx.x;
	if ( pair >= D41K_ROPE_PAIRS )
		return;
	const float frequency = __frcp_rn(powf(theta, __fmul_rn((float)(2u * pair), 1.0f / (float)(2u * D41K_ROPE_PAIRS))));
	if ( yarn == 0u )
	{
		table[pair] = frequency;
		return;
	}
	const float ramp = fminf(fmaxf(__fmul_rn(__fsub_rn((float)pair, low), inverse_span), 0.0f), 1.0f);
	const float smooth = __fsub_rn(1.0f, ramp);
	table[pair] = __fadd_rn(__fmul_rn(__fmul_rn(frequency, inverse_factor), __fsub_rn(1.0f, smooth)), __fmul_rn(frequency, smooth));
}

static inline SparkGpuError D41LaunchFrequencies(float *table, double theta, double compress_theta, double factor, double original,
	double beta_fast, double beta_slow)
{
	const double rope = 2.0 * D41K_ROPE_PAIRS;
	const double low_raw = rope * log(original / (beta_fast * 2.0 * M_PI)) / (2.0 * log(compress_theta));
	const double high_raw = rope * log(original / (beta_slow * 2.0 * M_PI)) / (2.0 * log(compress_theta));
	const double low = floor(low_raw) > 0.0 ? floor(low_raw) : 0.0;
	const double high = ceil(high_raw) < rope - 1.0 ? ceil(high_raw) : rope - 1.0;
	const double span = high - low > 1e-3 ? high - low : 1e-3;
	LM_LAUNCH((D41FrequencyKernel), 1, D41K_ROPE_PAIRS, 0, 0, table, (float)theta, 1.0f / (float)factor, 0.0f, 0.0f, 0u);
	LM_LAUNCH((D41FrequencyKernel), 1, D41K_ROPE_PAIRS, 0, 0, table + D41K_ROPE_PAIRS, (float)compress_theta, 1.0f / (float)factor, (float)low,
		1.0f / (float)span, 1u);
	return SparkGpuGetLastError();
}

static __device__ __forceinline__ void D41RopeTurn(uint32_t position, int32_t shift, float inverse_frequency, uint32_t inverse, float *sine,
	float *cosine)
{
	const float angle = __fmul_rn((float)((int64_t)position + shift), inverse_frequency);
	*sine = sinf(angle);
	*cosine = cosf(angle);
	if ( inverse != 0u )
		*sine = -*sine;
}

static __device__ __forceinline__ uint16_t D41RopeReal(float real, float imaginary, float sine, float cosine)
{
	return LmFloatToBf16(__fmaf_rn(real, cosine, -__fmul_rn(imaginary, sine)));
}

static __device__ __forceinline__ uint16_t D41RopeImaginary(float real, float imaginary, float sine, float cosine)
{
	return LmFloatToBf16(__fmaf_rn(real, sine, __fmul_rn(imaginary, cosine)));
}

static __device__ __forceinline__ void D41RopePair(uint16_t *value, uint32_t position, int32_t shift, float inverse_frequency, uint32_t inverse)
{
	float sine, cosine;
	D41RopeTurn(position, shift, inverse_frequency, inverse, &sine, &cosine);
	const float real = LmBf16ToFloat(value[0]), imaginary = LmBf16ToFloat(value[1]);
	value[0] = D41RopeReal(real, imaginary, sine, cosine);
	value[1] = D41RopeImaginary(real, imaginary, sine, cosine);
}

static __global__ void D41RopeRowsKernel(uint16_t *__restrict__ data, uint32_t heads, uint32_t head_stride, uint32_t rope_offset,
	const uint32_t *__restrict__ positions, int32_t shift, const float *__restrict__ inverse_frequency, uint32_t inverse,
	const uint32_t *__restrict__ row_valid)
{
	const uint32_t row = blockIdx.x, head = blockIdx.y, pair = threadIdx.x;
	if ( pair >= D41K_ROPE_PAIRS || head >= heads || (row_valid != 0 && row_valid[row] == D41K_NO_ROW) )
		return;
	D41RopePair(data + ((uint64_t)row * heads + head) * head_stride + rope_offset + 2u * pair, positions[row], shift, inverse_frequency[pair], inverse);
}

static __global__ void D41Fp8SliceKernel(const uint8_t *__restrict__ payload, const uint8_t *__restrict__ block_scale, uint8_t *__restrict__ weight,
	uint8_t *__restrict__ row_scale, uint16_t *__restrict__ wide, uint32_t columns, uint32_t row_first, uint32_t column_first,
	uint32_t output_columns)
{
	const uint32_t row = row_first + blockIdx.x, scale_columns = columns / 32u, output_groups = output_columns / 32u;
	for ( uint32_t group = threadIdx.x; group < output_groups; group += blockDim.x )
		row_scale[(uint64_t)blockIdx.x * output_groups + group] = block_scale[(uint64_t)(row / 32u) * scale_columns + column_first / 32u + group];
	if ( weight != 0 )
		for ( uint32_t index = threadIdx.x; index < output_columns; index += blockDim.x )
			weight[(uint64_t)blockIdx.x * output_columns + index] = payload[(uint64_t)row * columns + column_first + index];
	if ( wide != 0 )
		for ( uint32_t index = threadIdx.x; index < output_columns; index += blockDim.x )
			wide[(uint64_t)blockIdx.x * output_columns + index] = LmFloatToBf16(LmE4m3ToFloat(payload[(uint64_t)row * columns + column_first + index]) *
				LmE8m0ToFloat(block_scale[(uint64_t)(row / 32u) * scale_columns + (column_first + index) / 32u]));
}

static __global__ void D41CompressorRowsKernel(const float *__restrict__ kv, const float *__restrict__ score, uint16_t *__restrict__ pooled,
	uint32_t *__restrict__ emit_row, float *__restrict__ state, const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane,
	const uint32_t *__restrict__ row_ordinal, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices,
	const uint32_t *__restrict__ recurrent_slot, uint32_t ratio, uint32_t state_stride_floats, uint32_t width)
{
	const uint32_t row = blockIdx.x;
	const uint32_t position = positions[row];
	if ( ratio == 1u )
	{
		for ( uint32_t c = threadIdx.x; c < width; c += blockDim.x )
			pooled[(uint64_t)row * width + c] = LmFloatToBf16(kv[(uint64_t)row * width + c]);
		if ( threadIdx.x == 0u )
			emit_row[row] = position;
		return;
	}
	if ( (position + 1u) % ratio != 0u )
	{
		if ( threadIdx.x == 0u )
			emit_row[row] = D41K_NO_ROW;
		return;
	}
	const uint32_t lane = row_lane[row], ordinal = row_ordinal[row];
	const uint32_t previous = ordinal > 0u ? sequence_row_indices[sequence_row_begin[lane] + ordinal - 1u] : row;
	const float *slot_state = state + (uint64_t)recurrent_slot[lane] * state_stride_floats;
	for ( uint32_t c = threadIdx.x; c < width; c += blockDim.x )
	{
		const float kv0 = ordinal > 0u ? kv[(uint64_t)previous * width + c] : slot_state[c];
		const float score0 = ordinal > 0u ? score[(uint64_t)previous * width + c] : slot_state[width + c];
		const float kv1 = kv[(uint64_t)row * width + c], score1 = score[(uint64_t)row * width + c];
		pooled[(uint64_t)row * width + c] = LmFloatToBf16(SparkMathSoftmaxPool2(kv0, score0, kv1, score1));
	}
	if ( threadIdx.x == 0u )
		emit_row[row] = position / ratio;
}

static __global__ void D41CompressorSaveKernel(const float *__restrict__ kv, const float *__restrict__ score, float *__restrict__ state,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ recurrent_slot, uint32_t rows, uint32_t ratio,
	uint32_t state_stride_floats, uint32_t width, uint32_t commit, const uint32_t *__restrict__ lane_count)
{
	const uint32_t row = blockIdx.x;
	if ( commit == 0u || row >= rows || (positions[row] + 1u) % ratio == 0u )
		return;
	const uint32_t lane = row_lane[row];
	const uint32_t count = lane_count != 0 ? lane_count[lane] : sequence_row_begin[lane + 1u] - sequence_row_begin[lane];
	if ( row_ordinal[row] + 1u != count )
		return;
	float *slot_state = state + (uint64_t)recurrent_slot[lane] * state_stride_floats;
	for ( uint32_t c = threadIdx.x; c < width; c += blockDim.x )
	{
		slot_state[c] = kv[(uint64_t)row * width + c];
		slot_state[width + c] = score[(uint64_t)row * width + c];
	}
}

static __global__ void D41CompressedEmitPositionKernel(const uint32_t *__restrict__ positions, uint32_t *__restrict__ bound_position,
	uint32_t rows, uint32_t ratio)
{
	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row >= rows )
		return;
	const uint32_t visible = (positions[row] + 1u) / ratio;
	bound_position[row] = visible == 0u ? D41K_NO_ROW : visible - 1u;
}

static __global__ void D41RowBeginKernel(uint32_t *__restrict__ row_begin, uint32_t rows)
{
	row_begin[0] = 0u;
	row_begin[1] = rows;
}

static __global__ void D41CompressedLengthKernel(const uint32_t *__restrict__ context_length, uint32_t *__restrict__ compressed_length,
	uint32_t sequences, uint32_t ratio)
{
	const uint32_t sequence = blockIdx.x * blockDim.x + threadIdx.x;
	if ( sequence < sequences )
		compressed_length[sequence] = context_length[sequence] / ratio;
}

template<class Geometry>
static __global__ void D41StoreCompressedKernel(LmKvShardView view, const uint16_t *__restrict__ latent, const uint16_t *__restrict__ index_key,
	const uint32_t *__restrict__ emit_row, const uint32_t *__restrict__ sequence_of_row, uint32_t index_width)
{
	const uint32_t row = blockIdx.x;
	const uint32_t position = emit_row[row];
	if ( position == D41K_NO_ROW || SparkKvShardOwns(view.shard, position) == 0u )
		return;
	uint8_t *slot = (uint8_t *)LmKvShardSlotRequired<Geometry>(view, sequence_of_row[row], position, row, LM_KV_ACCESS_WRITE);
	if ( slot == 0 )
		return;
	uint16_t *target = (uint16_t *)slot;
	for ( uint32_t c = threadIdx.x; c < D41K_LATENT; c += blockDim.x )
		target[c] = latent[(uint64_t)row * D41K_LATENT + c];
	if ( index_key != 0 )
		for ( uint32_t c = threadIdx.x; c < index_width; c += blockDim.x )
			target[D41K_LATENT + c] = index_key[(uint64_t)row * index_width + c];
}

static __global__ void D41GatherPlanKernel(uint32_t *__restrict__ plan, uint32_t capacity, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ length)
{
	const uint32_t sequence = sequence_of_row[0];
	plan[0] = sequence;
	plan[capacity + sequence] = 0u;
	plan[2u * capacity + sequence] = length[sequence];
}

static __global__ void D41RingPackKernel(const uint16_t *__restrict__ rings, const uint32_t *__restrict__ recurrent_slot, uint64_t slot_stride,
	uint32_t ring_elements, uint16_t *__restrict__ send)
{
	const uint16_t *ring = rings + (uint64_t)blockIdx.x * ring_elements + (uint64_t)recurrent_slot[0] * slot_stride;
	uint16_t *target = send + (uint64_t)blockIdx.x * ring_elements;
	for ( uint32_t element = threadIdx.x; element < ring_elements; element += blockDim.x )
		target[element] = ring[element];
}

static __global__ void D41InitRecordsKernel(float *__restrict__ records, uint64_t destination_stride, uint32_t span, uint32_t destinations)
{
	for ( uint32_t destination = 0u; destination < destinations; ++destination )
		for ( uint32_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < span; offset += gridDim.x * blockDim.x )
			records[(uint64_t)destination * destination_stride + offset] = offset % D41K_RECORD == 0u ? -INFINITY : 0.0f;
}

static inline void D41InitRecords(float *records, uint64_t destination_stride, uint32_t rows, uint32_t heads_per_destination, uint32_t destinations,
	SparkGpuStream stream)
{
	const uint32_t span = rows * heads_per_destination * D41K_RECORD, blocks = (span + D41K_THREADS - 1u) / D41K_THREADS;
	LM_LAUNCH((D41InitRecordsKernel), blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u, D41K_THREADS, 0, stream, records, destination_stride, span, destinations);
}

static __global__ void D41WindowFoldKernel(float *__restrict__ records, uint64_t destination_stride, uint32_t heads_per_destination,
	const uint16_t *__restrict__ query, uint64_t query_rank_stride, const uint16_t *__restrict__ wave_kv,
	const uint16_t *__restrict__ ring, const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane,
	const uint32_t *__restrict__ row_ordinal, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices,
	const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows, uint32_t window, uint32_t degree, uint32_t rank,
	float qk_scale, uint32_t record_heads, uint32_t record_copies)
{
	__shared__ float warp_sums[D41K_FOLD_KEYS][D41K_THREADS / 32u];
	__shared__ float key_sums[D41K_FOLD_KEYS];
	const uint32_t row = blockIdx.x, head = blockIdx.y, lane_index = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	const uint32_t position = positions[row], lane = row_lane[row];
	const uint32_t wave_first = position - row_ordinal[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const uint32_t first = position + 1u > window ? position + 1u - window : 0u;
	const uint32_t destination = head / heads_per_destination, local = head % heads_per_destination;
	const uint16_t *q = query + (uint64_t)destination * query_rank_stride + ((uint64_t)row * heads_per_destination + local) * D41K_LATENT;
	const uint64_t record_offset = ((uint64_t)row * record_heads + head % record_heads) * D41K_RECORD;
	float *record = records + (uint64_t)(head / record_heads) * record_copies * destination_stride + record_offset;
	float running_max = record[0], running_sum = record[1];
	float accumulator[D41K_LATENT / D41K_THREADS], query_value[D41K_LATENT / D41K_THREADS];
	for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
	{
		accumulator[e] = record[2u + threadIdx.x + e * D41K_THREADS];
		query_value[e] = LmBf16ToFloat(q[threadIdx.x + e * D41K_THREADS]);
	}
	uint32_t key = first + (rank + degree - first % degree) % degree;
	while ( key <= position )
	{
		float value[D41K_FOLD_KEYS][D41K_LATENT / D41K_THREADS];
		uint32_t count = 0u;
		for ( ; count < D41K_FOLD_KEYS && key + count * degree <= position; ++count )
		{
			const uint32_t at = key + count * degree;
			const uint16_t *k = at >= wave_first ? wave_kv + (uint64_t)lane_rows[at - wave_first] * D41K_LATENT :
				ring + (uint64_t)recurrent_slot[lane] * ring_slot_stride + (uint64_t)((at / degree) % ring_rows) * D41K_LATENT;
			float partial = 0.0f;
			for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
			{
				value[count][e] = LmBf16ToFloat(k[threadIdx.x + e * D41K_THREADS]);
				partial += query_value[e] * value[count][e];
			}
			for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
				partial += __shfl_down_sync(0xffffffffu, partial, offset);
			if ( lane_index == 0u )
				warp_sums[count][warp] = partial;
		}
		__syncthreads();
		if ( threadIdx.x < count )
		{
			float total = 0.0f;
			for ( uint32_t w = 0u; w < D41K_THREADS / 32u; ++w )
				total += warp_sums[threadIdx.x][w];
			key_sums[threadIdx.x] = total;
		}
		__syncthreads();
		for ( uint32_t j = 0u; j < count; ++j )
		{
			const float score = key_sums[j] * qk_scale;
			const float top = fmaxf(running_max, score);
			const float keep = running_max == -INFINITY ? 0.0f : __expf(running_max - top), add = __expf(score - top);
			running_sum = running_sum * keep + add;
			for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
				accumulator[e] = accumulator[e] * keep + SparkMathAttentionValueWeight(add) * value[j][e];
			running_max = top;
		}
		__syncthreads();
		key += count * degree;
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

#define D41K_WINDOW_KEYS 128u
#define D41K_WINDOW_BATCH 16u

static __device__ __forceinline__ const uint16_t *D41WindowKey(const uint16_t *__restrict__ wave_kv, const uint16_t *__restrict__ slot_ring,
	const uint32_t *__restrict__ lane_rows, uint32_t wave_first, uint32_t window, uint32_t at)
{
	return at >= wave_first ? wave_kv + (uint64_t)lane_rows[at - wave_first] * D41K_LATENT : slot_ring + (uint64_t)(at % window) * D41K_LATENT;
}

#define D41K_WINDOW_PARTS 8u
#define D41K_WINDOW_PART_KEYS (D41K_WINDOW_KEYS / D41K_WINDOW_PARTS)
#define D41K_WINDOW_PART_SLICE (D41K_LATENT / D41K_WINDOW_PARTS)

static __device__ __forceinline__ float D41WarpPrefixMax(float value, uint32_t lane, float *warp_total)
{
	float inclusive = value;
	for ( uint32_t offset = 1u; offset < 32u; offset <<= 1u )
	{
		const float other = __shfl_up_sync(0xffffffffu, inclusive, offset);
		if ( lane >= offset )
			inclusive = fmaxf(inclusive, other);
	}
	if ( lane == 31u )
		*warp_total = inclusive;
	return __shfl_up_sync(0xffffffffu, inclusive, 1u);
}

static __device__ __forceinline__ void D41WindowPrefixWeights(float score, float carried_max, float prev_in_warp, const float *warp_max,
	uint32_t warp, uint32_t lane_index, uint32_t count, float *keep, float *added, float *folded)
{
	if ( threadIdx.x >= count )
		return;
	float before = carried_max;
	for ( uint32_t w = 0u; w < warp; ++w )
		before = fmaxf(before, warp_max[w]);
	if ( lane_index != 0u )
		before = fmaxf(before, prev_in_warp);
	const float top = fmaxf(before, score);
	keep[threadIdx.x] = before == -INFINITY ? 0.0f : __expf(before - top);
	added[threadIdx.x] = __expf(score - top);
	if ( threadIdx.x + 1u == count )
		folded[0] = top;
}

static __global__ void D41WindowFoldWholeKernel(float *__restrict__ records, uint32_t heads, const uint16_t *__restrict__ query,
	const uint16_t *__restrict__ wave_kv, const uint16_t *__restrict__ ring, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal, const uint32_t *__restrict__ sequence_row_begin,
	const uint32_t *__restrict__ sequence_row_indices, const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t window,
	float qk_scale)
{
	static_assert(D41K_WINDOW_KEYS % 32u == 0u && D41K_WINDOW_KEYS <= D41K_THREADS, "window keys fill whole warps");
	__shared__ float warp_sums[D41K_WINDOW_KEYS][D41K_THREADS / 32u];
	__shared__ float warp_max[D41K_WINDOW_KEYS / 32u];
	__shared__ float keep[D41K_WINDOW_KEYS];
	__shared__ float added[D41K_WINDOW_KEYS];
	__shared__ float folded[2];
	const uint32_t row = blockIdx.x, head = blockIdx.y, lane_index = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	const uint32_t position = positions[row], lane = row_lane[row];
	const uint32_t wave_first = position - row_ordinal[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const uint32_t first = position + 1u > window ? position + 1u - window : 0u, count = position + 1u - first;
	const uint16_t *q = query + ((uint64_t)row * heads + head) * D41K_LATENT;
	const uint16_t *slot_ring = ring + (uint64_t)recurrent_slot[lane] * ring_slot_stride;
	float *record = records + ((uint64_t)row * heads + head) * D41K_RECORD;
	const float carried_max = record[0], carried_sum = record[1];
	float accumulator[D41K_LATENT / D41K_THREADS], query_value[D41K_LATENT / D41K_THREADS];
	uint16_t value[D41K_WINDOW_BATCH][D41K_LATENT / D41K_THREADS];
	for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
	{
		accumulator[e] = record[2u + threadIdx.x + e * D41K_THREADS];
		query_value[e] = LmBf16ToFloat(q[threadIdx.x + e * D41K_THREADS]);
	}
	for ( uint32_t base = 0u; base < count; base += D41K_WINDOW_BATCH )
	{
		const uint32_t batch = count - base < D41K_WINDOW_BATCH ? count - base : D41K_WINDOW_BATCH;
		for ( uint32_t j = 0u; j < batch; ++j )
		{
			const uint16_t *k = D41WindowKey(wave_kv, slot_ring, lane_rows, wave_first, window, first + base + j);
			for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
				value[j][e] = k[threadIdx.x + e * D41K_THREADS];
		}
		for ( uint32_t j = 0u; j < batch; ++j )
		{
			float partial = 0.0f;
			for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
				partial += query_value[e] * LmBf16ToFloat(value[j][e]);
			for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
				partial += __shfl_down_sync(0xffffffffu, partial, offset);
			if ( lane_index == 0u )
				warp_sums[base + j][warp] = partial;
		}
	}
	__syncthreads();
	float score = -INFINITY, prev_in_warp = -INFINITY;
	if ( threadIdx.x < D41K_WINDOW_KEYS )
	{
		if ( threadIdx.x < count )
		{
			float total = 0.0f;
			for ( uint32_t w = 0u; w < D41K_THREADS / 32u; ++w )
				total += warp_sums[threadIdx.x][w];
			score = total * qk_scale;
		}
		prev_in_warp = D41WarpPrefixMax(score, lane_index, &warp_max[warp]);
	}
	__syncthreads();
	D41WindowPrefixWeights(score, carried_max, prev_in_warp, warp_max, warp, lane_index, count, keep, added, folded);
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		float running_sum = carried_sum;
		for ( uint32_t j = 0u; j < count; ++j )
			running_sum = running_sum * keep[j] + added[j];
		folded[1] = running_sum;
		if ( count == 0u )
			folded[0] = carried_max;
	}
	for ( uint32_t base = 0u; base < count; base += D41K_WINDOW_BATCH )
	{
		const uint32_t batch = count - base < D41K_WINDOW_BATCH ? count - base : D41K_WINDOW_BATCH;
		for ( uint32_t j = 0u; j < batch; ++j )
		{
			const uint16_t *k = D41WindowKey(wave_kv, slot_ring, lane_rows, wave_first, window, first + base + j);
			for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
				value[j][e] = k[threadIdx.x + e * D41K_THREADS];
		}
		for ( uint32_t j = 0u; j < batch; ++j )
		{
			const float kept = keep[base + j], weight = SparkMathAttentionValueWeight(added[base + j]);
			for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
				accumulator[e] = accumulator[e] * kept + weight * LmBf16ToFloat(value[j][e]);
		}
	}
	__syncthreads();
	for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
		record[2u + threadIdx.x + e * D41K_THREADS] = accumulator[e];
	if ( threadIdx.x == 0u )
	{
		record[0] = folded[0];
		record[1] = folded[1];
	}
}

static __global__ void __cluster_dims__(D41K_WINDOW_PARTS, 1, 1) __launch_bounds__(D41K_THREADS) D41WindowFoldClusterKernel(float *__restrict__ records,
	uint32_t heads, const uint16_t *__restrict__ query, const uint16_t *__restrict__ wave_kv, const uint16_t *__restrict__ ring,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices, const uint32_t *__restrict__ recurrent_slot,
	uint64_t ring_slot_stride, uint32_t window, float qk_scale)
{
	static_assert(D41K_WINDOW_KEYS % D41K_WINDOW_PARTS == 0u && D41K_LATENT % D41K_WINDOW_PARTS == 0u && D41K_WINDOW_PART_SLICE <= D41K_THREADS &&
		D41K_WINDOW_KEYS <= D41K_THREADS && D41K_WINDOW_KEYS % 32u == 0u, "each part scores a whole run of keys and folds a whole slice of the latent");
	__shared__ float warp_sums[D41K_WINDOW_PART_KEYS][D41K_THREADS / 32u];
	__shared__ float part_scores[D41K_WINDOW_PART_KEYS];
	__shared__ float warp_max[D41K_WINDOW_KEYS / 32u];
	__shared__ float keep[D41K_WINDOW_KEYS];
	__shared__ float added[D41K_WINDOW_KEYS];
	__shared__ float folded[2];
	cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
	const uint32_t part = blockIdx.x, head = blockIdx.y, row = blockIdx.z, lane_index = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	const uint32_t position = positions[row], lane = row_lane[row];
	const uint32_t wave_first = position - row_ordinal[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const uint32_t first = position + 1u > window ? position + 1u - window : 0u, count = position + 1u - first;
	const uint32_t key_first = part * D41K_WINDOW_PART_KEYS;
	const uint32_t key_count = count > key_first ? (count - key_first < D41K_WINDOW_PART_KEYS ? count - key_first : D41K_WINDOW_PART_KEYS) : 0u;
	const uint16_t *q = query + ((uint64_t)row * heads + head) * D41K_LATENT;
	const uint16_t *slot_ring = ring + (uint64_t)recurrent_slot[lane] * ring_slot_stride;
	float *record = records + ((uint64_t)row * heads + head) * D41K_RECORD;
	const float carried_max = record[0], carried_sum = record[1];
	float query_value[D41K_LATENT / D41K_THREADS];
	uint16_t value[D41K_WINDOW_PART_KEYS][D41K_LATENT / D41K_THREADS];
	for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
		query_value[e] = LmBf16ToFloat(q[threadIdx.x + e * D41K_THREADS]);
	for ( uint32_t j = 0u; j < key_count; ++j )
	{
		const uint16_t *k = D41WindowKey(wave_kv, slot_ring, lane_rows, wave_first, window, first + key_first + j);
		for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
			value[j][e] = k[threadIdx.x + e * D41K_THREADS];
	}
	for ( uint32_t j = 0u; j < key_count; ++j )
	{
		float partial = 0.0f;
		for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
			partial += query_value[e] * LmBf16ToFloat(value[j][e]);
		for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
			partial += __shfl_down_sync(0xffffffffu, partial, offset);
		if ( lane_index == 0u )
			warp_sums[j][warp] = partial;
	}
	__syncthreads();
	if ( threadIdx.x < key_count )
	{
		float total = 0.0f;
		for ( uint32_t w = 0u; w < D41K_THREADS / 32u; ++w )
			total += warp_sums[threadIdx.x][w];
		part_scores[threadIdx.x] = total * qk_scale;
	}
	cluster.sync();
	float score = -INFINITY, prev_in_warp = -INFINITY;
	if ( threadIdx.x < D41K_WINDOW_KEYS )
	{
		if ( threadIdx.x < count )
			score = cluster.map_shared_rank(part_scores, threadIdx.x / D41K_WINDOW_PART_KEYS)[threadIdx.x % D41K_WINDOW_PART_KEYS];
		prev_in_warp = D41WarpPrefixMax(score, lane_index, &warp_max[warp]);
	}
	cluster.sync();
	D41WindowPrefixWeights(score, carried_max, prev_in_warp, warp_max, warp, lane_index, count, keep, added, folded);
	__syncthreads();
	if ( part == 0u && threadIdx.x == 0u )
	{
		float running_sum = carried_sum;
		for ( uint32_t j = 0u; j < count; ++j )
			running_sum = running_sum * keep[j] + added[j];
		record[0] = count == 0u ? carried_max : folded[0];
		record[1] = running_sum;
	}
	if ( threadIdx.x < D41K_WINDOW_PART_SLICE )
	{
		const uint32_t e = part * D41K_WINDOW_PART_SLICE + threadIdx.x;
		float accumulator = record[2u + e];
		for ( uint32_t base = 0u; base < count; base += D41K_WINDOW_BATCH )
		{
			const uint32_t batch = count - base < D41K_WINDOW_BATCH ? count - base : D41K_WINDOW_BATCH;
			uint16_t slice[D41K_WINDOW_BATCH];
			for ( uint32_t j = 0u; j < batch; ++j )
				slice[j] = D41WindowKey(wave_kv, slot_ring, lane_rows, wave_first, window, first + base + j)[e];
			for ( uint32_t j = 0u; j < batch; ++j )
			{
				const float kept = keep[base + j], weight = SparkMathAttentionValueWeight(added[base + j]);
				accumulator = accumulator * kept + weight * LmBf16ToFloat(slice[j]);
			}
		}
		record[2u + e] = accumulator;
	}
}

static __device__ __forceinline__ void D41WindowRingStoreRow(const uint16_t *__restrict__ wave_kv, uint16_t *__restrict__ ring,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows,
	uint32_t window, uint32_t degree, uint32_t rank, uint32_t rows, uint32_t commit, const uint32_t *__restrict__ lane_count, uint32_t row)
{
	if ( commit == 0u || row >= rows || positions[row] % degree != rank )
		return;
	const uint32_t lane = row_lane[row];
	const uint32_t count = lane_count != 0 ? lane_count[lane] : sequence_row_begin[lane + 1u] - sequence_row_begin[lane];
	if ( row_ordinal[row] >= count || row_ordinal[row] + window < count )
		return;
	uint16_t *target = ring + (uint64_t)recurrent_slot[lane] * ring_slot_stride +
		(uint64_t)((positions[row] / degree) % ring_rows) * D41K_LATENT;
	for ( uint32_t c = threadIdx.x; c < D41K_LATENT; c += blockDim.x )
		target[c] = wave_kv[(uint64_t)row * D41K_LATENT + c];
}

static __global__ void D41WindowRingStoreKernel(const uint16_t *__restrict__ wave_kv, uint16_t *__restrict__ ring,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows,
	uint32_t window, uint32_t degree, uint32_t rank, uint32_t rows, uint32_t commit, const uint32_t *__restrict__ lane_count)
{
	D41WindowRingStoreRow(wave_kv, ring, positions, row_lane, row_ordinal, sequence_row_begin, recurrent_slot, ring_slot_stride, ring_rows, window, degree,
		rank, rows, commit, lane_count, blockIdx.x);
}

#define D41K_GRID_ITEMS 32768u

static __device__ __forceinline__ uint32_t D41PackBf16(float low, float high)
{
	return (uint32_t)LmFloatToBf16(low) | ((uint32_t)LmFloatToBf16(high) << 16u);
}

static __device__ __forceinline__ uint32_t D41DequantE4m3x2(uint32_t bytes, float scale)
{
	return D41PackBf16(LmE4m3ToFloat((uint8_t)(bytes & 0xffu)) * scale, LmE4m3ToFloat((uint8_t)((bytes >> 8u) & 0xffu)) * scale);
}

static __device__ __forceinline__ uint4 D41DequantE4m3x8(uint2 codes, float scale)
{
	return make_uint4(D41DequantE4m3x2(codes.x, scale), D41DequantE4m3x2(codes.x >> 16u, scale),
		D41DequantE4m3x2(codes.y, scale), D41DequantE4m3x2(codes.y >> 16u, scale));
}

static __device__ __forceinline__ void D41MergeSinkHead(const float *__restrict__ partials, uint64_t source_stride, uint32_t sources, uint32_t heads,
	const float *__restrict__ sink, uint16_t *output, uint32_t row, uint32_t head, float *scales, float *denominator_shared)
{
	const uint64_t base = ((uint64_t)row * heads + head) * D41K_RECORD;
	float max_local = -1.0e30f, den_local = 0.0f;
	if ( threadIdx.x < 32u )
	{
		if ( threadIdx.x < sources )
		{
			max_local = partials[(uint64_t)threadIdx.x * source_stride + base];
			den_local = partials[(uint64_t)threadIdx.x * source_stride + base + 1u];
		}
		float top = max_local;
		for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
			top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset));
		float scale_local = 0.0f;
		if ( threadIdx.x < sources )
		{
			scale_local = max_local == -INFINITY ? 0.0f : __expf(max_local - top);
			scales[threadIdx.x] = scale_local;
		}
		float denominator = 0.0f;
		for ( uint32_t source = 0u; source < sources; ++source )
		{
			const float d = __shfl_sync(0xffffffffu, den_local, source);
			const float s = __shfl_sync(0xffffffffu, scale_local, source);
			denominator = fmaf(d, s, denominator);
		}
		if ( threadIdx.x == 0u )
			*denominator_shared = denominator + __expf(sink[head] - top);
	}
	__syncthreads();
	const float denom = *denominator_shared;
	const float *first_src = partials + base + 2u;
	if ( (((uintptr_t)first_src | (uintptr_t)output) & 15u) == 0u && (sources == 1u || (source_stride & 3u) == 0u) )
	{
		const uint64_t stride_f4 = source_stride >> 2u;
		const float4 *pq = (const float4 *)first_src;
		uint2 *oq = (uint2 *)output;
		for ( uint32_t v = threadIdx.x; v < D41K_LATENT / 4u; v += blockDim.x )
		{
			float4 m = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			for ( uint32_t source = 0u; source < sources; ++source )
			{
				const float s = scales[source];
				const float4 p = pq[(uint64_t)source * stride_f4 + v];
				m.x = fmaf(p.x, s, m.x);
				m.y = fmaf(p.y, s, m.y);
				m.z = fmaf(p.z, s, m.z);
				m.w = fmaf(p.w, s, m.w);
			}
			oq[v] = make_uint2(D41PackBf16(m.x / denom, m.y / denom), D41PackBf16(m.z / denom, m.w / denom));
		}
		return;
	}
	if ( (((uintptr_t)first_src | (uintptr_t)output) & 7u) == 0u && (sources == 1u || (source_stride & 1u) == 0u) )
	{
		const uint64_t stride_f2 = source_stride >> 1u;
		const float2 *pq = (const float2 *)first_src;
		uint32_t *oq = (uint32_t *)output;
		for ( uint32_t v = threadIdx.x; v < D41K_LATENT / 2u; v += blockDim.x )
		{
			float2 m = make_float2(0.0f, 0.0f);
			for ( uint32_t source = 0u; source < sources; ++source )
			{
				const float s = scales[source];
				const float2 p = pq[(uint64_t)source * stride_f2 + v];
				m.x = fmaf(p.x, s, m.x);
				m.y = fmaf(p.y, s, m.y);
			}
			oq[v] = D41PackBf16(m.x / denom, m.y / denom);
		}
		return;
	}
	for ( uint32_t element = threadIdx.x; element < D41K_LATENT; element += blockDim.x )
	{
		float merged = 0.0f;
		for ( uint32_t source = 0u; source < sources; ++source )
			merged = fmaf(first_src[(uint64_t)source * source_stride + element], scales[source], merged);
		output[element] = LmFloatToBf16(merged / denom);
	}
}

static __global__ void D41MergeSinkKernel(const float *__restrict__ partials, uint64_t source_stride, uint32_t sources, uint32_t heads,
	const float *__restrict__ sink, uint16_t *__restrict__ output)
{
	__shared__ float scales[16];
	__shared__ float denominator_shared;
	D41MergeSinkHead(partials, source_stride, sources, heads, sink, output + ((uint64_t)blockIdx.x * heads + blockIdx.y) * D41K_LATENT, blockIdx.x,
		blockIdx.y, scales, &denominator_shared);
}

template<uint32_t THREADS>
static __device__ __forceinline__ void D41OutputHead(const float *__restrict__ partials, uint64_t source_stride, uint32_t heads,
	const float *__restrict__ sink, const uint32_t *__restrict__ positions, const float *__restrict__ inverse_frequency, uint16_t *group_input,
	uint32_t group_heads, uint32_t rows, uint32_t row, uint32_t head, float *shared)
{
	static_assert(THREADS % 32u == 0u && D41K_LATENT % THREADS == 0u, "every warp quantizes whole 32-element blocks of the head");
	uint16_t *merged = (uint16_t *)(shared + 32u);
	D41MergeSinkHead(partials, source_stride, 1u, heads, sink, merged, row, head, shared, shared + 16u);
	__syncthreads();
	if ( threadIdx.x < D41K_ROPE_PAIRS )
		D41RopePair(merged + D41K_LATENT - 2u * D41K_ROPE_PAIRS + 2u * threadIdx.x, positions[row], 0, inverse_frequency[threadIdx.x], 1u);
	__syncthreads();
	uint16_t *target = group_input + ((uint64_t)(head / group_heads) * rows + row) * group_heads * D41K_LATENT + (head % group_heads) * D41K_LATENT;
	for ( uint32_t element = threadIdx.x; element < D41K_LATENT; element += THREADS )
		target[element] = SparkDsv41FlashFp8QdqLane(LmBf16ToFloat(merged[element]));
}

static __device__ __forceinline__ uint32_t D41SwigluPair(uint32_t g_pair, uint32_t u_pair, float limit)
{
	const float g0 = fminf(LmBf16ToFloat((uint16_t)g_pair), limit);
	const float u0 = fminf(fmaxf(LmBf16ToFloat((uint16_t)u_pair), -limit), limit);
	const float g1 = fminf(LmBf16ToFloat((uint16_t)(g_pair >> 16u)), limit);
	const float u1 = fminf(fmaxf(LmBf16ToFloat((uint16_t)(u_pair >> 16u)), -limit), limit);
	return (uint32_t)LmFloatToBf16((g0 / (1.0f + expf(-g0))) * u0) |
		((uint32_t)LmFloatToBf16((g1 / (1.0f + expf(-g1))) * u1) << 16u);
}

static __global__ void D41SwigluRowsKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ output,
	uint32_t width, float limit)
{
	const uint32_t row = blockIdx.y;
	const uint64_t row_base = (uint64_t)row * width;
	const uint16_t *g_row = gate + row_base, *u_row = up + row_base;
	uint16_t *o_row = output + row_base;
	if ( (width & 7u) == 0u && (((uintptr_t)g_row | (uintptr_t)u_row | (uintptr_t)o_row) & 15u) == 0u )
	{
		const uint32_t quads = width >> 3u;
		const uint4 *gq = (const uint4 *)g_row, *uq = (const uint4 *)u_row;
		uint4 *oq = (uint4 *)o_row;
		for ( uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < quads; q += gridDim.x * blockDim.x )
		{
			const uint4 gv = gq[q], uv = uq[q];
			uint4 ov;
			ov.x = D41SwigluPair(gv.x, uv.x, limit);
			ov.y = D41SwigluPair(gv.y, uv.y, limit);
			ov.z = D41SwigluPair(gv.z, uv.z, limit);
			ov.w = D41SwigluPair(gv.w, uv.w, limit);
			oq[q] = ov;
		}
		return;
	}
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		const float g = fminf(LmBf16ToFloat(g_row[element]), limit);
		const float u = fminf(fmaxf(LmBf16ToFloat(u_row[element]), -limit), limit);
		o_row[element] = LmFloatToBf16((g / (1.0f + expf(-g))) * u);
	}
}

static __device__ __forceinline__ uint32_t D41AddPair(uint32_t a, uint32_t b)
{
	const uint32_t lo = (uint32_t)LmFloatToBf16(LmBf16ToFloat((uint16_t)a) + LmBf16ToFloat((uint16_t)b));
	const uint32_t hi = (uint32_t)LmFloatToBf16(LmBf16ToFloat((uint16_t)(a >> 16u)) + LmBf16ToFloat((uint16_t)(b >> 16u)));
	return lo | (hi << 16u);
}

static __global__ void D41AddRowsKernel(uint16_t *__restrict__ hidden, const uint16_t *__restrict__ addend, uint64_t elements)
{
	if ( (((uintptr_t)hidden | (uintptr_t)addend) & 15u) == 0u )
	{
		const uint64_t quads = elements >> 3u, tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (uint64_t)gridDim.x * blockDim.x;
		uint4 *hq = (uint4 *)hidden;
		const uint4 *aq = (const uint4 *)addend;
		for ( uint64_t q = tid; q < quads; q += stride )
		{
			const uint4 hv = hq[q], av = aq[q];
			uint4 ov;
			ov.x = D41AddPair(hv.x, av.x);
			ov.y = D41AddPair(hv.y, av.y);
			ov.z = D41AddPair(hv.z, av.z);
			ov.w = D41AddPair(hv.w, av.w);
			hq[q] = ov;
		}
		for ( uint64_t index = (quads << 3u) + tid; index < elements; index += stride )
			hidden[index] = LmFloatToBf16(LmBf16ToFloat(hidden[index]) + LmBf16ToFloat(addend[index]));
		return;
	}
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
		hidden[index] = LmFloatToBf16(LmBf16ToFloat(hidden[index]) + LmBf16ToFloat(addend[index]));
}

static __global__ void D41GroupInputKernel(const uint16_t *__restrict__ gathered, uint16_t *__restrict__ group_input, uint32_t rows,
	uint32_t width_per_rank, uint32_t first_rank, uint32_t ranks)
{
	const uint32_t row = blockIdx.y;
	if ( (width_per_rank & 7u) == 0u && (((uintptr_t)gathered | (uintptr_t)group_input) & 15u) == 0u )
	{
		const uint32_t quads = width_per_rank >> 3u, span = quads * ranks;
		const uint4 *gq = (const uint4 *)gathered;
		uint4 *oq = (uint4 *)group_input;
		for ( uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < span; q += gridDim.x * blockDim.x )
		{
			const uint32_t rank = first_rank + q / quads, col = q % quads;
			oq[(uint64_t)row * span + q] = gq[((uint64_t)rank * rows + row) * quads + col];
		}
		return;
	}
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width_per_rank * ranks; index += gridDim.x * blockDim.x )
	{
		const uint32_t rank = first_rank + index / width_per_rank, element = index % width_per_rank;
		group_input[(uint64_t)row * width_per_rank * ranks + index] = gathered[((uint64_t)rank * rows + row) * width_per_rank + element];
	}
}

static __global__ void D41RankRowsKernel(const uint4 *__restrict__ gathered, uint4 *__restrict__ output, uint32_t rows, uint32_t quads,
	uint32_t first_rank, uint32_t ranks)
{
	const uint32_t row = blockIdx.y;
	const uint64_t span = (uint64_t)ranks * quads;
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < span; index += (uint64_t)gridDim.x * blockDim.x )
	{
		const uint64_t rank = first_rank + index / quads, column = index % quads;
		output[(uint64_t)row * span + index] = gathered[(rank * rows + row) * quads + column];
	}
}

static __global__ void D41ColumnsKernel(const uint16_t *__restrict__ source, uint32_t source_stride, uint16_t *__restrict__ target,
	uint32_t target_stride, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	const uint16_t *src_row = source + (uint64_t)row * source_stride;
	uint16_t *dst_row = target + (uint64_t)row * target_stride;
	if ( (width & 7u) == 0u && (((uintptr_t)src_row | (uintptr_t)dst_row) & 15u) == 0u )
	{
		const uint32_t quads = width >> 3u;
		const uint4 *sq = (const uint4 *)src_row;
		uint4 *dq = (uint4 *)dst_row;
		for ( uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < quads; q += gridDim.x * blockDim.x )
			dq[q] = sq[q];
		return;
	}
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += gridDim.x * blockDim.x )
		dst_row[index] = src_row[index];
}

static __global__ void D41IndexPackKernel(const uint16_t *__restrict__ query, const uint16_t *__restrict__ weight, uint16_t *__restrict__ packed,
	uint32_t query_width, uint32_t weight_width, uint32_t pack_width)
{
	const uint32_t row = blockIdx.x;
	const uint16_t *q_row = query + (uint64_t)row * query_width, *w_row = weight + (uint64_t)row * weight_width;
	uint16_t *p_row = packed + (uint64_t)row * pack_width;
	if ( (query_width & 7u) == 0u && (((uintptr_t)q_row | (uintptr_t)p_row) & 15u) == 0u )
	{
		const uint32_t q_quads = query_width >> 3u;
		const uint4 *qq = (const uint4 *)q_row;
		uint4 *pq = (uint4 *)p_row;
		for ( uint32_t i = threadIdx.x; i < q_quads; i += blockDim.x )
			pq[i] = qq[i];
		for ( uint32_t element = query_width + threadIdx.x; element < pack_width; element += blockDim.x )
			p_row[element] = element < query_width + weight_width ? w_row[element - query_width] : (uint16_t)0u;
		return;
	}
	for ( uint32_t element = threadIdx.x; element < pack_width; element += blockDim.x )
		p_row[element] = element < query_width ? q_row[element] :
			element < query_width + weight_width ? w_row[element - query_width] : (uint16_t)0u;
}

static __global__ void D41IndexUnpackKernel(const uint16_t *__restrict__ packed, uint64_t rank_stride, uint16_t *__restrict__ query,
	uint16_t *__restrict__ weight, uint32_t query_width, uint32_t weight_width, uint32_t pack_width)
{
	const uint32_t row = blockIdx.x, rank = blockIdx.y, ranks = gridDim.y;
	const uint16_t *source = packed + (uint64_t)rank * rank_stride + (uint64_t)row * pack_width;
	uint16_t *q_dst = query + ((uint64_t)row * ranks + rank) * query_width;
	uint16_t *w_dst = weight + ((uint64_t)row * ranks + rank) * weight_width;
	if ( (query_width & 7u) == 0u && (((uintptr_t)source | (uintptr_t)q_dst) & 15u) == 0u )
	{
		const uint32_t q_quads = query_width >> 3u;
		const uint4 *sq = (const uint4 *)source;
		uint4 *qq = (uint4 *)q_dst;
		for ( uint32_t i = threadIdx.x; i < q_quads; i += blockDim.x )
			qq[i] = sq[i];
		for ( uint32_t element = threadIdx.x; element < weight_width; element += blockDim.x )
			w_dst[element] = source[query_width + element];
		return;
	}
	for ( uint32_t element = threadIdx.x; element < query_width + weight_width; element += blockDim.x )
		if ( element < query_width )
			q_dst[element] = source[element];
		else
			w_dst[element - query_width] = source[element];
}

static __device__ __forceinline__ uint32_t D41ScalePair(uint32_t v, float factor)
{
	const uint32_t lo = (uint32_t)LmFloatToBf16(LmBf16ToFloat((uint16_t)v) * factor);
	const uint32_t hi = (uint32_t)LmFloatToBf16(LmBf16ToFloat((uint16_t)(v >> 16u)) * factor);
	return lo | (hi << 16u);
}

static __global__ void D41ScaleRowsKernel(uint16_t *__restrict__ data, uint64_t elements, float factor)
{
	if ( ((uintptr_t)data & 15u) == 0u )
	{
		const uint64_t quads = elements >> 3u, tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (uint64_t)gridDim.x * blockDim.x;
		uint4 *dq = (uint4 *)data;
		for ( uint64_t q = tid; q < quads; q += stride )
		{
			const uint4 dv = dq[q];
			uint4 ov;
			ov.x = D41ScalePair(dv.x, factor);
			ov.y = D41ScalePair(dv.y, factor);
			ov.z = D41ScalePair(dv.z, factor);
			ov.w = D41ScalePair(dv.w, factor);
			dq[q] = ov;
		}
		for ( uint64_t index = (quads << 3u) + tid; index < elements; index += stride )
			data[index] = LmFloatToBf16(LmBf16ToFloat(data[index]) * factor);
		return;
	}
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
		data[index] = LmFloatToBf16(LmBf16ToFloat(data[index]) * factor);
}

static inline dim3 D41ItemGrid(uint32_t blocks, uint32_t items)
{
	const uint32_t span = items < D41K_GRID_ITEMS ? (items != 0u ? items : 1u) : D41K_GRID_ITEMS;
	return dim3(blocks, span, (items + span - 1u) / span);
}

static __global__ void D41MoeSendKernel(const uint16_t *__restrict__ gate_up, const float *__restrict__ packed_weight,
	const uint16_t *__restrict__ shared, uint16_t *__restrict__ send, uint32_t routes, uint32_t rows, uint32_t slice, uint32_t hidden_slice,
	uint32_t degree, uint64_t destination_stride, float limit, uint32_t route_elements, uint32_t codes)
{
	const uint32_t item = blockIdx.z * gridDim.y + blockIdx.y, element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( item < routes )
	{
		const uint32_t route = item;
		if ( element >= slice )
			return;
		const uint16_t value = LmSwigluPackedValue(gate_up[(uint64_t)route * 2u * slice + element], gate_up[(uint64_t)route * 2u * slice + slice + element],
			packed_weight[route], limit);
		if ( codes == 0u )
		{
			for ( uint32_t destination = 0u; destination < degree; ++destination )
				send[(uint64_t)destination * destination_stride + (uint64_t)route * slice + element] = value;
			return;
		}
		const float activation = LmBf16ToFloat(value);
		float amax = fabsf(activation);
		for ( uint32_t offset = SPARK_DSV41_FLASH_FP8_BLOCK / 2u; offset != 0u; offset >>= 1u )
			amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
		const float scale = SparkMathFp8QdqScale(fmaxf(0.0f, amax));
		const uint8_t code = LmFloatToE4m3(fminf(fmaxf(activation / scale, -LM_E4M3_MAX), LM_E4M3_MAX));
		const uint8_t exponent = (uint8_t)(__float_as_uint(scale) >> 23u);
		for ( uint32_t destination = 0u; destination < degree; ++destination )
		{
			uint8_t *bytes = (uint8_t *)(send + (uint64_t)destination * destination_stride);
			bytes[(uint64_t)route * slice + element] = code;
			if ( element % SPARK_DSV41_FLASH_FP8_BLOCK == 0u )
				bytes[(uint64_t)routes * slice + (uint64_t)route * (slice / SPARK_DSV41_FLASH_FP8_BLOCK) + element / SPARK_DSV41_FLASH_FP8_BLOCK] = exponent;
		}
		return;
	}
	const uint32_t row = item - routes;
	if ( row >= rows || element >= hidden_slice * degree )
		return;
	send[(uint64_t)(element / hidden_slice) * destination_stride + (uint64_t)routes * route_elements + (uint64_t)row * hidden_slice +
		element % hidden_slice] = shared[(uint64_t)row * hidden_slice * degree + element];
}

static __global__ void D41ActAssembleQdqKernel(const uint16_t *__restrict__ received, uint16_t *__restrict__ act, uint32_t slice, uint32_t width,
	uint64_t source_stride, uint32_t routes, uint32_t codes)
{
	const uint32_t route = blockIdx.z * gridDim.y + blockIdx.y, element = blockIdx.x * blockDim.x + threadIdx.x, source = element / slice;
	if ( route >= routes || element >= width )
		return;
	if ( codes != 0u )
	{
		const uint8_t *bytes = (const uint8_t *)(received + (uint64_t)source * source_stride);
		const uint32_t within = element - source * slice;
		const uint32_t exponent = bytes[(uint64_t)routes * slice + (uint64_t)route * (slice / SPARK_DSV41_FLASH_FP8_BLOCK) +
			within / SPARK_DSV41_FLASH_FP8_BLOCK];
		act[(uint64_t)route * width + element] = LmFloatToBf16(LmE4m3ToFloat(bytes[(uint64_t)route * slice + within]) * __uint_as_float(exponent << 23u));
		return;
	}
	act[(uint64_t)route * width + element] = SparkDsv41FlashFp8QdqLane(LmBf16ToFloat(
		received[(uint64_t)source * source_stride + (uint64_t)route * slice + element - source * slice]));
}

static __global__ void D41ActAssembleCodesKernel(const uint16_t *__restrict__ received, uint8_t *__restrict__ codes, uint32_t slice, uint32_t width,
	uint64_t source_stride, uint32_t routes)
{
	const uint32_t route = blockIdx.z * gridDim.y + blockIdx.y, vector = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t code_vectors = width / 16u, blocks = slice / SPARK_DSV41_FLASH_FP8_BLOCK;
	if ( route >= routes || vector >= code_vectors + width / (2u * SPARK_DSV41_FLASH_FP8_BLOCK) )
		return;
	if ( vector < code_vectors )
	{
		const uint32_t element = vector * 16u, source = element / slice;
		const uint8_t *bytes = (const uint8_t *)(received + (uint64_t)source * source_stride);
		*(uint4 *)(codes + (uint64_t)route * width + element) = *(const uint4 *)(bytes + (uint64_t)route * slice + element - source * slice);
		return;
	}
	const uint32_t block = (vector - code_vectors) * 2u, source = block / blocks;
	const uint8_t *bytes = (const uint8_t *)(received + (uint64_t)source * source_stride);
	*(uint16_t *)(codes + (uint64_t)routes * width + (uint64_t)route * (width / SPARK_DSV41_FLASH_FP8_BLOCK) + block) =
		*(const uint16_t *)(bytes + (uint64_t)routes * slice + (uint64_t)route * blocks + block - source * blocks);
}

static __global__ void D41SharedSendKernel(const uint16_t *__restrict__ shared, uint16_t *__restrict__ send, uint32_t rows, uint32_t width,
	uint32_t degree, uint64_t destination_stride)
{
	const uint32_t row = blockIdx.y, element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row >= rows || element >= width )
		return;
	const float activation = LmBf16ToFloat(shared[(uint64_t)row * width + element]);
	float amax = fabsf(activation);
	for ( uint32_t offset = SPARK_DSV41_FLASH_FP8_BLOCK / 2u; offset != 0u; offset >>= 1u )
		amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
	const float scale = SparkMathFp8QdqScale(fmaxf(0.0f, amax));
	const uint8_t code = LmFloatToE4m3(fminf(fmaxf(activation / scale, -LM_E4M3_MAX), LM_E4M3_MAX));
	const uint8_t exponent = (uint8_t)(__float_as_uint(scale) >> 23u);
	for ( uint32_t destination = 0u; destination < degree; ++destination )
	{
		uint8_t *bytes = (uint8_t *)(send + (uint64_t)destination * destination_stride);
		bytes[(uint64_t)row * width + element] = code;
		if ( element % SPARK_DSV41_FLASH_FP8_BLOCK == 0u )
			bytes[(uint64_t)rows * width + (uint64_t)row * (width / SPARK_DSV41_FLASH_FP8_BLOCK) + element / SPARK_DSV41_FLASH_FP8_BLOCK] = exponent;
	}
}

static __global__ void D41SharedDequantKernel(const uint8_t *__restrict__ bytes, uint16_t *__restrict__ act, uint32_t rows, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	if ( row >= rows )
		return;
	const uint8_t *src = bytes + (uint64_t)row * width;
	const uint8_t *scales = bytes + (uint64_t)rows * width + (uint64_t)row * (width / SPARK_DSV41_FLASH_FP8_BLOCK);
	uint16_t *dst = act + (uint64_t)row * width;
	if ( (width & 7u) == 0u && ((uintptr_t)src & 7u) == 0u && ((uintptr_t)dst & 15u) == 0u )
	{
		const uint32_t quads = width >> 3u;
		const uint2 *sq = (const uint2 *)src;
		uint4 *dq = (uint4 *)dst;
		for ( uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < quads; q += gridDim.x * blockDim.x )
		{
			const uint32_t exponent = scales[(q * 8u) / SPARK_DSV41_FLASH_FP8_BLOCK];
			dq[q] = D41DequantE4m3x8(sq[q], __uint_as_float(exponent << 23u));
		}
		return;
	}
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		const uint32_t exponent = scales[element / SPARK_DSV41_FLASH_FP8_BLOCK];
		dst[element] = LmFloatToBf16(LmE4m3ToFloat(src[element]) * __uint_as_float(exponent << 23u));
	}
}

static __global__ void D41FfnPackKernel(const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, uint8_t *__restrict__ block,
	uint32_t own, uint32_t routed, uint32_t hidden, uint32_t top_k)
{
	const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t ids = (uint64_t)own * (hidden + hidden / SPARK_DSV41_FLASH_FP8_BLOCK);
	if ( item >= routed * top_k )
		return;
	((uint32_t *)(block + ids))[item] = route_expert[item];
	((float *)(block + ids + (uint64_t)own * top_k * sizeof(uint32_t)))[item] = route_weight[item];
}

static __global__ void D41FfnUnpackKernel(const uint8_t *__restrict__ received, uint16_t *__restrict__ xq, uint32_t *__restrict__ route_expert,
	float *__restrict__ route_weight, uint32_t own, uint32_t hidden, uint32_t top_k, uint64_t block_bytes)
{
	const uint32_t row = blockIdx.y, source = row / own, index = row - source * own;
	const uint8_t *block = received + (uint64_t)source * block_bytes;
	const uint64_t ids = (uint64_t)own * (hidden + hidden / SPARK_DSV41_FLASH_FP8_BLOCK);
	for ( uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < top_k; k += gridDim.x * blockDim.x )
	{
		route_expert[(uint64_t)row * top_k + k] = ((const uint32_t *)(block + ids))[(uint64_t)index * top_k + k];
		route_weight[(uint64_t)row * top_k + k] =
			((const float *)(block + ids + (uint64_t)own * top_k * sizeof(uint32_t)))[(uint64_t)index * top_k + k];
	}
	const uint8_t *src = block + (uint64_t)index * hidden;
	const uint8_t *scales = block + (uint64_t)own * hidden + (uint64_t)index * (hidden / SPARK_DSV41_FLASH_FP8_BLOCK);
	uint16_t *dst = xq + (uint64_t)row * hidden;
	if ( (hidden & 7u) == 0u && ((uintptr_t)src & 7u) == 0u && ((uintptr_t)dst & 15u) == 0u )
	{
		const uint32_t quads = hidden >> 3u;
		const uint2 *sq = (const uint2 *)src;
		uint4 *dq = (uint4 *)dst;
		for ( uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < quads; q += gridDim.x * blockDim.x )
		{
			const uint32_t exponent = scales[(q * 8u) / SPARK_DSV41_FLASH_FP8_BLOCK];
			dq[q] = D41DequantE4m3x8(sq[q], __uint_as_float(exponent << 23u));
		}
		return;
	}
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < hidden; element += gridDim.x * blockDim.x )
	{
		const uint32_t exponent = scales[element / SPARK_DSV41_FLASH_FP8_BLOCK];
		dst[element] = LmFloatToBf16(LmE4m3ToFloat(src[element]) * __uint_as_float(exponent << 23u));
	}
}

#define D41K_BACKUP_FLAG 0x80000000u
#define D41K_BACKUP_EXPERT_ROUTES 93u
#define D41K_BACKUP_ROUNDS 64u
#define D41K_BACKUP_PAIR_SLACK (2u * D41K_BACKUP_EXPERT_ROUTES)
#define D41K_SLOT_WARPS 32u

template <uint32_t RANKS, uint32_t EXPERTS>
struct D41ExchangeCounts
{
	uint32_t pair[RANKS * RANKS];
	uint16_t routes[EXPERTS];
	uint8_t sole[RANKS * EXPERTS];
};

template <uint32_t RANKS, uint32_t EXPERTS>
struct D41ExchangeTotals
{
	uint32_t pair[RANKS * RANKS];
	uint32_t routes[EXPERTS];
	uint16_t sole[RANKS * EXPERTS];
};

template <uint32_t RANKS, uint32_t EXPERTS>
static __global__ void D41ExchangeCountsKernel(const uint32_t *__restrict__ route_expert, uint32_t rows, uint32_t first, uint32_t top_k,
	const uint8_t *__restrict__ owner, uint32_t backups, D41ExchangeCounts<RANKS, EXPERTS> *__restrict__ counts)
{
	__shared__ uint32_t pair[RANKS * RANKS], routes[EXPERTS], sole[RANKS * EXPERTS];
	for ( uint32_t i = threadIdx.x; i < RANKS * EXPERTS; i += blockDim.x )
		sole[i] = 0u;
	for ( uint32_t i = threadIdx.x; i < EXPERTS; i += blockDim.x )
		routes[i] = 0u;
	for ( uint32_t i = threadIdx.x; i < RANKS * RANKS; i += blockDim.x )
		pair[i] = 0u;
	__syncthreads();
	for ( uint32_t row = threadIdx.x; row < rows; row += blockDim.x )
	{
		const uint32_t target = (first + row) % RANKS;
		uint32_t mask = 0u, twice = 0u;
		for ( uint32_t k = 0u; k < top_k; ++k )
		{
			const uint32_t home = 1u << owner[route_expert[(uint64_t)row * top_k + k]];
			twice |= mask & home;
			mask |= home;
		}
		for ( uint32_t rank = 0u; rank < RANKS; ++rank )
			if ( ((mask >> rank) & 1u) != 0u )
				atomicAdd(&pair[target * RANKS + rank], 1u);
		for ( uint32_t k = 0u; k < top_k && backups != 0u; ++k )
		{
			const uint32_t expert = route_expert[(uint64_t)row * top_k + k];
			atomicAdd(&routes[expert], 1u);
			if ( ((twice >> owner[expert]) & 1u) == 0u )
				atomicAdd(&sole[target * EXPERTS + expert], 1u);
		}
	}
	__syncthreads();
	for ( uint32_t i = threadIdx.x; i < RANKS * RANKS; i += blockDim.x )
		counts->pair[i] = pair[i];
	for ( uint32_t i = threadIdx.x; i < EXPERTS && backups != 0u; i += blockDim.x )
		counts->routes[i] = (uint16_t)routes[i];
	for ( uint32_t i = threadIdx.x; i < RANKS * EXPERTS && backups != 0u; i += blockDim.x )
		counts->sole[i] = (uint8_t)sole[i];
}

template <uint32_t RANKS, uint32_t EXPERTS>
static __global__ void D41ExchangeTotalsKernel(const uint8_t *__restrict__ gathered, uint64_t stride, uint32_t backups,
	D41ExchangeTotals<RANKS, EXPERTS> *__restrict__ totals)
{
	const uint32_t items = RANKS * RANKS + (backups != 0u ? EXPERTS + RANKS * EXPERTS : 0u);
	for ( uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < items; item += gridDim.x * blockDim.x )
	{
		uint32_t total = 0u;
		for ( uint32_t sender = 0u; sender < RANKS; ++sender )
		{
			const D41ExchangeCounts<RANKS, EXPERTS> *counts = (const D41ExchangeCounts<RANKS, EXPERTS> *)(gathered + sender * stride);
			total += item < RANKS * RANKS ? counts->pair[item] : item < RANKS * RANKS + EXPERTS ? (uint32_t)counts->routes[item - RANKS * RANKS] :
				(uint32_t)counts->sole[item - RANKS * RANKS - EXPERTS];
		}
		if ( item < RANKS * RANKS )
			totals->pair[item] = total;
		else if ( item < RANKS * RANKS + EXPERTS )
			totals->routes[item - RANKS * RANKS] = total;
		else
			totals->sole[item - RANKS * RANKS - EXPERTS] = (uint16_t)total;
	}
}

template <uint32_t RANKS, uint32_t EXPERTS, uint32_t LOCAL, uint32_t COPIES>
static inline uint32_t D41BackupPlan(const D41ExchangeTotals<RANKS, EXPERTS> *totals, const uint8_t *owner, const uint8_t *backup,
	const uint16_t *expert_at, uint32_t limit, uint16_t *quota, uint32_t *pair)
{
	uint64_t load[RANKS] = {0};
	uint32_t avail[EXPERTS], left[RANKS * EXPERTS], moved[COPIES][EXPERTS], any = 0u;
	uint8_t paid[COPIES][EXPERTS];
	memcpy(pair, totals->pair, sizeof(totals->pair));
	memset(quota, 0, (size_t)COPIES * RANKS * EXPERTS * sizeof(uint16_t));
	memset(moved, 0, sizeof(moved));
	memset(paid, 0, sizeof(paid));
	for ( uint32_t expert = 0u; expert < EXPERTS; ++expert )
	{
		uint32_t total = 0u, usable = 0u;
		for ( uint32_t target = 0u; target < RANKS; ++target )
		{
			total += totals->sole[target * EXPERTS + expert];
			left[target * EXPERTS + expert] = totals->sole[target * EXPERTS + expert];
		}
		for ( uint32_t copy = 0u; copy < COPIES; ++copy )
			usable |= backup[copy * EXPERTS + expert] < RANKS && backup[copy * EXPERTS + expert] != owner[expert] ? 1u : 0u;
		avail[expert] = usable != 0u ? total : 0u;
		if ( totals->routes[expert] != 0u )
			load[owner[expert]] += D41K_BACKUP_EXPERT_ROUTES + totals->routes[expert];
	}
	for ( uint32_t round = 0u; round < D41K_BACKUP_ROUNDS * COPIES; ++round )
	{
		uint32_t busiest = 0u, best = EXPERTS, best_copy = 0u;
		for ( uint32_t rank = 1u; rank < RANKS; ++rank )
			busiest = load[rank] > load[busiest] ? rank : busiest;
		for ( uint32_t place = 0u; place < LOCAL; ++place )
		{
			const uint32_t expert = expert_at[busiest * LOCAL + place];
			for ( uint32_t copy = 0u; copy < COPIES && expert < EXPERTS && avail[expert] != 0u; ++copy )
			{
				const uint32_t target = backup[copy * EXPERTS + expert];
				if ( target >= RANKS || target == owner[expert] )
					continue;
				if ( best == EXPERTS || load[target] < load[backup[best_copy * EXPERTS + best]] ||
					(load[target] == load[backup[best_copy * EXPERTS + best]] && avail[expert] > avail[best]) )
				{
					best = expert;
					best_copy = copy;
				}
			}
		}
		if ( best == EXPERTS )
			break;
		const uint32_t target = backup[best_copy * EXPERTS + best];
		const uint64_t extra = paid[best_copy][best] != 0u ? 0u : D41K_BACKUP_EXPERT_ROUTES, other = load[target] + extra;
		const uint64_t half = load[busiest] > other ? (load[busiest] - other) / 2u : 0u;
		const uint32_t take = half < avail[best] ? (uint32_t)half : avail[best];
		if ( take == 0u )
			break;
		load[busiest] -= take;
		load[target] += take + extra;
		paid[best_copy][best] = 1u;
		avail[best] -= take;
		moved[best_copy][best] += take;
		any = 1u;
	}
	for ( uint32_t copy = 0u; copy < COPIES && any != 0u; ++copy )
		for ( uint32_t expert = 0u; expert < EXPERTS; ++expert )
		{
			uint32_t given = 0u, remaining = 0u;
			uint16_t *share = quota + (uint64_t)copy * RANKS * EXPERTS;
			if ( moved[copy][expert] == 0u )
				continue;
			for ( uint32_t target = 0u; target < RANKS; ++target )
				remaining += left[target * EXPERTS + expert];
			for ( uint32_t target = 0u; target < RANKS; ++target )
			{
				const uint32_t part = (uint32_t)((uint64_t)moved[copy][expert] * left[target * EXPERTS + expert] / remaining);
				share[target * EXPERTS + expert] = (uint16_t)part;
				given += part;
			}
			for ( uint32_t target = 0u; given < moved[copy][expert]; target = (target + 1u) % RANKS )
				if ( share[target * EXPERTS + expert] < left[target * EXPERTS + expert] )
				{
					share[target * EXPERTS + expert]++;
					given++;
				}
			for ( uint32_t target = 0u; target < RANKS; ++target )
			{
				left[target * EXPERTS + expert] -= share[target * EXPERTS + expert];
				pair[target * RANKS + owner[expert]] -= share[target * EXPERTS + expert];
				pair[target * RANKS + backup[copy * EXPERTS + expert]] += share[target * EXPERTS + expert];
			}
		}
	uint64_t ceiling = 0u;
	for ( uint32_t rank = 0u; rank < RANKS; ++rank )
		ceiling = load[rank] > ceiling ? load[rank] : ceiling;
	ceiling += D41K_BACKUP_PAIR_SLACK;
	for ( ;; )
	{
		uint32_t top = 0u, from = 0u, to = 0u, best = EXPERTS, best_copy = 0u;
		for ( uint32_t index = 0u; index < RANKS * RANKS; ++index )
			if ( pair[index] > top )
			{
				top = pair[index];
				from = index / RANKS;
				to = index % RANKS;
			}
		for ( uint32_t place = 0u; place < LOCAL; ++place )
		{
			const uint32_t expert = expert_at[to * LOCAL + place];
			if ( expert >= EXPERTS || left[from * EXPERTS + expert] == 0u )
				continue;
			for ( uint32_t copy = 0u; copy < COPIES; ++copy )
			{
				const uint32_t target = backup[copy * EXPERTS + expert];
				if ( target >= RANKS || target == owner[expert] || pair[from * RANKS + target] + 1u >= top ||
					load[target] + 1u + (paid[copy][expert] != 0u ? 0u : D41K_BACKUP_EXPERT_ROUTES) > ceiling )
					continue;
				if ( best == EXPERTS || pair[from * RANKS + target] < pair[from * RANKS + backup[best_copy * EXPERTS + best]] )
				{
					best = expert;
					best_copy = copy;
				}
			}
		}
		if ( best == EXPERTS )
			break;
		const uint32_t target = backup[best_copy * EXPERTS + best];
		quota[((uint64_t)best_copy * RANKS + from) * EXPERTS + best]++;
		left[from * EXPERTS + best]--;
		pair[from * RANKS + to]--;
		pair[from * RANKS + target]++;
		load[to] -= 1u;
		load[target] += 1u + (paid[best_copy][best] != 0u ? 0u : D41K_BACKUP_EXPERT_ROUTES);
		paid[best_copy][best] = 1u;
		any = 1u;
	}
	for ( uint32_t index = 0u; index < RANKS * RANKS && any != 0u; ++index )
		if ( pair[index] > limit )
		{
			memcpy(pair, totals->pair, sizeof(totals->pair));
			memset(quota, 0, (size_t)COPIES * RANKS * EXPERTS * sizeof(uint16_t));
			return 0u;
		}
	return any;
}

template <uint32_t RANKS, uint32_t EXPERTS, uint32_t TOPK, uint32_t COPIES>
static __global__ __launch_bounds__(D41K_SLOT_WARPS * 32u) void D41ExchangeSlotsKernel(const uint32_t *__restrict__ route_expert, uint32_t rows,
	const uint8_t *__restrict__ owner, const uint8_t *__restrict__ backup, const uint16_t *__restrict__ quota, uint32_t capacity,
	uint32_t *__restrict__ slot, uint32_t *__restrict__ count)
{
	constexpr uint32_t WORDS = (RANKS + 3u) / 4u;
	static_assert(RANKS <= 16u && TOPK * 32u < 256u, "a row names its rank in four bits and a warp's records to a rank fit eight");
	__shared__ uint16_t sole_rank[D41K_SLOT_WARPS][EXPERTS];
	__shared__ uint16_t sole_base[EXPERTS];
	__shared__ uint32_t target_rank[D41K_SLOT_WARPS][RANKS];
	__shared__ uint32_t target_base[RANKS];
	const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	for ( uint32_t i = threadIdx.x; i < EXPERTS; i += blockDim.x )
		sole_base[i] = 0u;
	if ( threadIdx.x < RANKS )
		target_base[threadIdx.x] = 0u;
	for ( uint32_t chunk = 0u; chunk < rows; chunk += blockDim.x )
	{
		const uint32_t row = chunk + threadIdx.x;
		uint32_t expert[TOPK], candidate[TOPK], before[TOPK], peers[TOPK], counter[WORDS], inclusive[WORDS], mask = 0u, twice = 0u;
		uint64_t target = 0u;
		for ( uint32_t i = threadIdx.x; i < D41K_SLOT_WARPS * EXPERTS; i += blockDim.x )
			sole_rank[i / EXPERTS][i % EXPERTS] = 0u;
		for ( uint32_t k = 0u; k < TOPK; ++k )
		{
			expert[k] = row < rows ? route_expert[(uint64_t)row * TOPK + k] : EXPERTS;
			const uint32_t home = expert[k] < EXPERTS ? 1u << owner[expert[k]] : 0u;
			twice |= mask & home;
			mask |= home;
		}
		for ( uint32_t rank = 0u; rank < RANKS; ++rank )
			target |= (uint64_t)rank << (4u * rank);
		for ( uint32_t k = 0u; k < TOPK; ++k )
		{
			uint32_t planned = 0u;
			for ( uint32_t copy = 0u; copy < COPIES && quota != 0 && expert[k] < EXPERTS; ++copy )
				planned += quota[copy * EXPERTS + expert[k]];
			candidate[k] = planned != 0u && ((twice >> owner[expert[k]]) & 1u) == 0u ? expert[k] : EXPERTS;
			before[k] = peers[k] = 0u;
		}
		for ( uint32_t source = 0u; source < 32u && quota != 0; ++source )
			for ( uint32_t j = 0u; j < TOPK; ++j )
			{
				const uint32_t other = __shfl_sync(0xffffffffu, candidate[j], source);
				for ( uint32_t k = 0u; k < TOPK; ++k )
					if ( candidate[k] != EXPERTS && other == candidate[k] )
					{
						before[k] += source < lane ? 1u : 0u;
						peers[k] += 1u;
					}
			}
		__syncthreads();
		for ( uint32_t k = 0u; k < TOPK; ++k )
			if ( candidate[k] != EXPERTS && before[k] == 0u )
				sole_rank[warp][candidate[k]] = (uint16_t)peers[k];
		__syncthreads();
		for ( uint32_t e = threadIdx.x; e < EXPERTS && quota != 0; e += blockDim.x )
		{
			uint32_t running = sole_base[e];
			for ( uint32_t w = 0u; w < D41K_SLOT_WARPS; ++w )
			{
				const uint32_t here = sole_rank[w][e];
				sole_rank[w][e] = (uint16_t)running;
				running += here;
			}
			sole_base[e] = (uint16_t)running;
		}
		__syncthreads();
		for ( uint32_t k = 0u; k < TOPK; ++k )
			if ( candidate[k] != EXPERTS )
			{
				const uint32_t place = (uint32_t)sole_rank[warp][candidate[k]] + before[k], home = owner[candidate[k]];
				uint32_t bound = 0u;
				for ( uint32_t copy = 0u; copy < COPIES; ++copy )
				{
					const uint32_t start = bound;
					bound += quota[copy * EXPERTS + candidate[k]];
					if ( place >= start && place < bound )
						target = (target & ~(0xfull << (4u * home))) | ((uint64_t)backup[copy * EXPERTS + candidate[k]] << (4u * home));
				}
			}
		for ( uint32_t w = 0u; w < WORDS; ++w )
			counter[w] = 0u;
		for ( uint32_t home = 0u; home < RANKS; ++home )
			if ( ((mask >> home) & 1u) != 0u )
			{
				const uint32_t to = (uint32_t)(target >> (4u * home)) & 0xfu;
				counter[to / 4u] += 1u << (8u * (to % 4u));
			}
		for ( uint32_t w = 0u; w < WORDS; ++w )
		{
			inclusive[w] = counter[w];
			for ( uint32_t step = 1u; step < 32u; step <<= 1u )
			{
				const uint32_t add = __shfl_up_sync(0xffffffffu, inclusive[w], step);
				inclusive[w] += lane >= step ? add : 0u;
			}
		}
		if ( lane == 31u )
			for ( uint32_t to = 0u; to < RANKS; ++to )
				target_rank[warp][to] = (inclusive[to / 4u] >> (8u * (to % 4u))) & 0xffu;
		__syncthreads();
		if ( threadIdx.x < RANKS )
		{
			uint32_t running = target_base[threadIdx.x];
			for ( uint32_t w = 0u; w < D41K_SLOT_WARPS; ++w )
			{
				const uint32_t here = target_rank[w][threadIdx.x];
				target_rank[w][threadIdx.x] = running;
				running += here;
			}
			target_base[threadIdx.x] = running;
		}
		__syncthreads();
		if ( row < rows )
		{
			uint32_t used[WORDS];
			for ( uint32_t w = 0u; w < WORDS; ++w )
				used[w] = inclusive[w] - counter[w];
			for ( uint32_t home = 0u; home < RANKS; ++home )
			{
				uint32_t at = UINT32_MAX;
				if ( ((mask >> home) & 1u) != 0u )
				{
					const uint32_t to = (uint32_t)(target >> (4u * home)) & 0xfu;
					at = to * capacity + target_rank[warp][to] + ((used[to / 4u] >> (8u * (to % 4u))) & 0xffu);
					used[to / 4u] += 1u << (8u * (to % 4u));
				}
				slot[(uint64_t)row * RANKS + home] = at;
			}
		}
		__syncthreads();
	}
	if ( threadIdx.x < RANKS )
		count[threadIdx.x] = target_base[threadIdx.x];
}

static __device__ __forceinline__ void D41RouteLocal(uint32_t *route_expert, float *route_weight, uint32_t *route_global, uint32_t routes,
	const uint8_t *__restrict__ group, uint32_t local_experts, uint32_t groups, uint32_t experts, uint32_t route)
{
	if ( route >= routes )
		return;
	const uint32_t value = route_expert[route], expert = value & ~D41K_BACKUP_FLAG;
	const uint32_t local = expert < experts ? group[expert] : groups;
	route_global[route] = expert;
	if ( (value & D41K_BACKUP_FLAG) != 0u ? local >= local_experts && local < groups : local < local_experts )
		route_expert[route] = local;
	else
	{
		route_expert[route] = groups;
		route_weight[route] = 0.0f;
	}
}

static __global__ void D41RouteLocalKernel(uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ route_global,
	uint32_t routes, const uint8_t *__restrict__ group, uint32_t local_experts, uint32_t groups, uint32_t experts)
{
	D41RouteLocal(route_expert, route_weight, route_global, routes, group, local_experts, groups, experts, blockIdx.x * blockDim.x + threadIdx.x);
}

template<uint32_t THREADS, uint32_t EXPERTS>
static __device__ __forceinline__ void D41RouteTail(uint32_t *route_expert, float *route_weight, uint32_t *route_global, uint32_t routes,
	const uint8_t *__restrict__ group, uint32_t local_experts, uint32_t experts, uint32_t top_k, uint32_t *group_row_offset, uint32_t *route_packed_row,
	uint32_t *route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *tile_prefix_up, uint32_t neuron_tiles_down,
	uint32_t *tile_prefix_down, float *packed_weight)
{
	for ( uint32_t route = threadIdx.x; route < routes; route += THREADS )
		D41RouteLocal(route_expert, route_weight, route_global, routes, group, local_experts, local_experts, experts, route);
	__syncthreads();
	LmRouteBuildBlock<THREADS, EXPERTS>(route_expert, routes, top_k, group_row_offset, route_packed_row, route_source_token, tile_m, neuron_tiles_up,
		tile_prefix_up, neuron_tiles_down, tile_prefix_down);
	__syncthreads();
	for ( uint32_t route = threadIdx.x; route < routes; route += THREADS )
		LmPackedRouteWeight(route_packed_row, route_weight, packed_weight, routes, route);
}

static __global__ void D41ExchangePackKernel(const uint8_t *__restrict__ codes, const uint8_t *__restrict__ scales,
	const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, const uint32_t *__restrict__ slot, const uint8_t *__restrict__ owner,
	uint32_t ranks, uint32_t capacity, uint32_t hidden, uint32_t top_k, uint32_t experts, uint64_t record_bytes, uint8_t *__restrict__ send)
{
	const uint32_t row = blockIdx.x, home = blockIdx.y, at = slot[(uint64_t)row * ranks + home];
	const uint32_t scale_bytes = hidden / SPARK_DSV41_FLASH_FP8_BLOCK;
	if ( at == UINT32_MAX )
		return;
	const uint32_t backup = at / capacity != home ? 1u : 0u;
	uint8_t *record = send + (uint64_t)at * record_bytes;
	for ( uint32_t i = threadIdx.x; i < hidden / 16u; i += blockDim.x )
		((uint4 *)record)[i] = ((const uint4 *)(codes + (uint64_t)row * hidden))[i];
	for ( uint32_t i = threadIdx.x; i < scale_bytes; i += blockDim.x )
		record[hidden + i] = scales[(uint64_t)row * scale_bytes + i];
	for ( uint32_t k = threadIdx.x; k < top_k; k += blockDim.x )
	{
		const uint32_t expert = route_expert[(uint64_t)row * top_k + k];
		((uint32_t *)(record + hidden + scale_bytes))[k] = backup == 0u ? expert : owner[expert] == home ? expert | D41K_BACKUP_FLAG : experts;
		((float *)(record + hidden + scale_bytes + top_k * sizeof(uint32_t)))[k] = route_weight[(uint64_t)row * top_k + k];
	}
}

template <uint32_t RANKS>
struct D41ReceivedRows
{
	uint32_t first[RANKS + 1u];
};

template <uint32_t RANKS>
static __device__ __forceinline__ uint32_t D41ReceivedSource(const D41ReceivedRows<RANKS> &received, uint32_t row)
{
	uint32_t source = 0u;
	while ( source + 1u < RANKS && received.first[source + 1u] <= row )
		++source;
	return source;
}

static __device__ __forceinline__ void D41AddBf16x8(float *total, uint4 value)
{
	const uint32_t words[4] = { value.x, value.y, value.z, value.w };
	for ( uint32_t i = 0u; i < 4u; ++i )
	{
		total[2u * i] += LmBf16ToFloat((uint16_t)(words[i] & 0xffffu));
		total[2u * i + 1u] += LmBf16ToFloat((uint16_t)(words[i] >> 16u));
	}
}

static __device__ __forceinline__ uint4 D41RoundBf16x8(const float *total)
{
	return make_uint4(D41PackBf16(total[0], total[1]), D41PackBf16(total[2], total[3]), D41PackBf16(total[4], total[5]),
		D41PackBf16(total[6], total[7]));
}

template <uint32_t RANKS>
static __global__ void D41ExchangeUnpackKernel(const uint8_t *__restrict__ received_bytes, D41ReceivedRows<RANKS> received, uint32_t capacity,
	uint32_t hidden, uint32_t top_k, uint64_t record_bytes, uint8_t *__restrict__ codes, uint8_t *__restrict__ scales,
	uint32_t *__restrict__ route_expert, float *__restrict__ route_weight)
{
	const uint32_t row = blockIdx.y, source = D41ReceivedSource<RANKS>(received, row), index = row - received.first[source];
	const uint32_t vector = blockIdx.x * blockDim.x + threadIdx.x, scale_bytes = hidden / SPARK_DSV41_FLASH_FP8_BLOCK;
	const uint8_t *record = received_bytes + ((uint64_t)source * capacity + index) * record_bytes;
	if ( vector < top_k )
	{
		route_expert[(uint64_t)row * top_k + vector] = ((const uint32_t *)(record + hidden + scale_bytes))[vector];
		route_weight[(uint64_t)row * top_k + vector] = ((const float *)(record + hidden + scale_bytes + top_k * sizeof(uint32_t)))[vector];
	}
	if ( codes != 0 && vector * 16u < hidden )
		((uint4 *)(codes + (uint64_t)row * hidden))[vector] = ((const uint4 *)record)[vector];
	if ( scales != 0 && vector < scale_bytes )
		scales[(uint64_t)row * scale_bytes + vector] = record[hidden + vector];
}

template <uint32_t RANKS>
static __global__ void D41RecordPlaceKernel(uint32_t *__restrict__ source_token, uint32_t count, D41ReceivedRows<RANKS> received, uint32_t capacity)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= count )
		return;
	const uint32_t row = source_token[index];
	if ( row >= received.first[RANKS] )
		return;
	const uint32_t source = D41ReceivedSource<RANKS>(received, row);
	source_token[index] = source * capacity + row - received.first[source];
}

template <uint32_t RANKS>
static __global__ void D41ExchangeFinalizeKernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row,
	const uint32_t *__restrict__ route_expert, D41ReceivedRows<RANKS> received, uint32_t capacity, uint32_t top_k, uint32_t local_experts,
	uint32_t width, uint32_t skip_single, uint16_t *__restrict__ output)
{
	const uint32_t row = blockIdx.y, source = D41ReceivedSource<RANKS>(received, row), index = row - received.first[source];
	uint4 *out = (uint4 *)(output + ((uint64_t)source * capacity + index) * width);
	if ( skip_single != 0u )
	{
		uint32_t local = 0u;
		for ( uint32_t k = 0u; k < top_k; ++k )
			local += route_expert[row * top_k + k] < local_experts ? 1u : 0u;
		if ( local == 1u )
			return;
	}
	for ( uint32_t vector = blockIdx.x * blockDim.x + threadIdx.x; vector * 8u < width; vector += gridDim.x * blockDim.x )
	{
		float total[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		for ( uint32_t k = 0u; k < top_k; ++k )
		{
			const uint32_t route = row * top_k + k;
			if ( route_expert[route] < local_experts )
				D41AddBf16x8(total, ((const uint4 *)(packed + (uint64_t)route_packed_row[route] * width))[vector]);
		}
		out[vector] = D41RoundBf16x8(total);
	}
}

template <uint32_t RANKS>
static __global__ void D41FinalizeMapKernel(const uint32_t *__restrict__ route_expert, const uint32_t *__restrict__ route_packed_row,
	D41ReceivedRows<RANKS> received, uint32_t capacity, uint32_t top_k, uint32_t local_experts, uint32_t rows, uint32_t mapped,
	uint32_t *__restrict__ output_row)
{
	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t local = 0u, only = 0u;
	if ( row >= rows )
		return;
	for ( uint32_t k = 0u; k < top_k; ++k )
		if ( route_expert[row * top_k + k] < local_experts )
		{
			local++;
			only = row * top_k + k;
		}
	if ( local == 1u )
	{
		const uint32_t source = D41ReceivedSource<RANKS>(received, row);
		output_row[route_packed_row[only]] = mapped | (source * capacity + row - received.first[source]);
		return;
	}
	for ( uint32_t k = 0u; k < top_k; ++k )
		if ( route_expert[row * top_k + k] < local_experts )
			output_row[route_packed_row[row * top_k + k]] = route_packed_row[row * top_k + k];
}

static __host__ __device__ __forceinline__ uint32_t D41ShuffleLive(uint32_t source, uint32_t rows, uint32_t own)
{
	const uint32_t first = source * own;
	return first < rows ? (rows - first < own ? rows - first : own) : 0u;
}

static __host__ __device__ __forceinline__ uint32_t D41ShuffleCount(uint32_t source, uint32_t target, uint32_t rows, uint32_t own, uint32_t ranks)
{
	const uint32_t live = D41ShuffleLive(source, rows, own), offset = (target + ranks - (source * own) % ranks) % ranks;
	return live > offset ? (live - offset - 1u) / ranks + 1u : 0u;
}

static __global__ void D41ShufflePackKernel(const uint8_t *__restrict__ codes, const uint8_t *__restrict__ scales,
	const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, uint32_t first, uint32_t ranks, uint32_t capacity,
	uint32_t hidden, uint32_t top_k, uint64_t record_bytes, uint8_t *__restrict__ send)
{
	const uint32_t row = blockIdx.x, target = (first + row) % ranks, index = row / ranks;
	const uint32_t scale_bytes = hidden / SPARK_DSV41_FLASH_FP8_BLOCK;
	uint8_t *record = send + ((uint64_t)target * capacity + index) * record_bytes;
	for ( uint32_t i = threadIdx.x; i < hidden / 16u; i += blockDim.x )
		((uint4 *)record)[i] = ((const uint4 *)(codes + (uint64_t)row * hidden))[i];
	for ( uint32_t i = threadIdx.x; i < scale_bytes; i += blockDim.x )
		record[hidden + i] = scales[(uint64_t)row * scale_bytes + i];
	for ( uint32_t k = threadIdx.x; k < top_k; k += blockDim.x )
	{
		((uint32_t *)(record + hidden + scale_bytes))[k] = route_expert[(uint64_t)row * top_k + k];
		((float *)(record + hidden + scale_bytes + top_k * sizeof(uint32_t)))[k] = route_weight[(uint64_t)row * top_k + k];
	}
}

template <uint32_t RANKS>
static __global__ void D41ShuffleUnpackKernel(const uint8_t *__restrict__ received_bytes, D41ReceivedRows<RANKS> received, uint32_t capacity,
	uint32_t hidden, uint32_t top_k, uint64_t record_bytes, uint8_t *__restrict__ codes, uint8_t *__restrict__ scales,
	uint32_t *__restrict__ route_expert, float *__restrict__ route_weight)
{
	const uint32_t row = blockIdx.y, source = D41ReceivedSource<RANKS>(received, row), index = row - received.first[source];
	const uint32_t scale_bytes = hidden / SPARK_DSV41_FLASH_FP8_BLOCK;
	const uint8_t *record = received_bytes + ((uint64_t)source * capacity + index) * record_bytes;
	const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x, lanes = gridDim.x * blockDim.x;
	for ( uint32_t i = lane; i < hidden / 16u; i += lanes )
		((uint4 *)(codes + (uint64_t)row * hidden))[i] = ((const uint4 *)record)[i];
	for ( uint32_t i = lane; i < scale_bytes; i += lanes )
		scales[(uint64_t)row * scale_bytes + i] = record[hidden + i];
	if ( lane < top_k )
	{
		route_expert[(uint64_t)row * top_k + lane] = ((const uint32_t *)(record + hidden + scale_bytes))[lane];
		route_weight[(uint64_t)row * top_k + lane] = ((const float *)(record + hidden + scale_bytes + top_k * sizeof(uint32_t)))[lane];
	}
}

template <uint32_t RANKS>
static __global__ void D41ShuffleSumKernel(const uint16_t *__restrict__ landing, const uint32_t *__restrict__ slot,
	D41ReceivedRows<RANKS> received, uint32_t shuffle, uint32_t hidden, uint16_t *__restrict__ output)
{
	const uint32_t row = blockIdx.y, source = D41ReceivedSource<RANKS>(received, row), index = row - received.first[source];
	uint4 *out = (uint4 *)(output + ((uint64_t)source * shuffle + index) * hidden);
	for ( uint32_t vector = blockIdx.x * blockDim.x + threadIdx.x; vector * 8u < hidden; vector += gridDim.x * blockDim.x )
	{
		float total[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		for ( uint32_t rank = 0u; rank < RANKS; ++rank )
		{
			const uint32_t at = slot[(uint64_t)row * RANKS + rank];
			if ( at != UINT32_MAX )
				D41AddBf16x8(total, ((const uint4 *)(landing + (uint64_t)at * hidden))[vector]);
		}
		out[vector] = D41RoundBf16x8(total);
	}
}

static __global__ void D41ShuffleFinishKernel(const uint16_t *__restrict__ back, const uint16_t *__restrict__ shared, uint32_t live, uint32_t first,
	uint32_t ranks, uint32_t shuffle, uint32_t hidden, uint16_t *__restrict__ output)
{
	const uint32_t row = blockIdx.y, target = (first + row) % ranks, index = row / ranks;
	for ( uint32_t vector = blockIdx.x * blockDim.x + threadIdx.x; vector * 8u < hidden; vector += gridDim.x * blockDim.x )
	{
		if ( row >= live )
		{
			((uint4 *)(output + (uint64_t)row * hidden))[vector] = make_uint4(0u, 0u, 0u, 0u);
			continue;
		}
		const uint4 moe = ((const uint4 *)(back + ((uint64_t)target * shuffle + index) * hidden))[vector];
		const uint4 extra = ((const uint4 *)(shared + (uint64_t)row * hidden))[vector];
		const uint32_t moe_words[4] = { moe.x, moe.y, moe.z, moe.w }, extra_words[4] = { extra.x, extra.y, extra.z, extra.w };
		uint32_t sum[4];
		for ( uint32_t i = 0u; i < 4u; ++i )
			sum[i] = D41PackBf16(LmBf16ToFloat((uint16_t)(moe_words[i] & 0xffffu)) + LmBf16ToFloat((uint16_t)(extra_words[i] & 0xffffu)),
				LmBf16ToFloat((uint16_t)(moe_words[i] >> 16u)) + LmBf16ToFloat((uint16_t)(extra_words[i] >> 16u)));
		((uint4 *)(output + (uint64_t)row * hidden))[vector] = make_uint4(sum[0], sum[1], sum[2], sum[3]);
	}
}

static __global__ void D41WindowAssembleKernel(const uint16_t *__restrict__ own_kv, const uint16_t *__restrict__ edges, uint16_t *__restrict__ window_kv,
	uint32_t own, uint32_t edge, uint32_t first)
{
	const uint32_t block = blockIdx.x;
	const uint32_t rank = block < own ? 0u : (block - own) / edge, index = block < own ? block : (block - own) - rank * edge;
	const uint4 *source = (const uint4 *)(block < own ? own_kv + (uint64_t)block * D41K_LATENT : edges + (uint64_t)(block - own) * D41K_LATENT);
	uint4 *target = (uint4 *)(window_kv + (uint64_t)(block < own ? first + block : rank * own + own - edge + index) * D41K_LATENT);
	for ( uint32_t i = threadIdx.x; i < D41K_LATENT / 8u; i += blockDim.x )
		target[i] = source[i];
}

static_assert(D41K_HC * sizeof(float) == sizeof(uint4), "a row's four stream weights travel as one 16-byte vector after its streams");

static __host__ __device__ inline uint32_t D41TailSpan(uint32_t source, uint32_t target, uint32_t rows, uint32_t source_first, uint32_t own,
	uint32_t first, uint32_t tail_own, uint32_t *low)
{
	const uint32_t tail_rows = rows - first, source_rows = rows - source_first;
	const uint32_t source_begin = source_first + (source * own < source_rows ? source * own : source_rows);
	const uint32_t source_end = source_first + ((source + 1u) * own < source_rows ? (source + 1u) * own : source_rows);
	const uint32_t target_begin = first + (target * tail_own < tail_rows ? target * tail_own : tail_rows);
	const uint32_t target_end = first + ((target + 1u) * tail_own < tail_rows ? (target + 1u) * tail_own : tail_rows);
	const uint32_t begin = source_begin > target_begin ? source_begin : target_begin;
	const uint32_t end = source_end < target_end ? source_end : target_end;
	*low = begin;
	return end > begin ? end - begin : 0u;
}

static __global__ void D41TailPackKernel(const uint16_t *__restrict__ streams, const float *__restrict__ pre, uint32_t rank, uint32_t rows,
	uint32_t source_first, uint32_t own, uint32_t first, uint32_t tail_own, uint32_t capacity, uint32_t width, uint4 *__restrict__ send,
	uint4 *__restrict__ send_pre)
{
	const uint32_t target = blockIdx.z, slot = blockIdx.y, vectors = width / 8u;
	uint32_t low;
	if ( slot >= D41TailSpan(rank, target, rows, source_first, own, first, tail_own, &low) )
		return;
	const uint64_t row = (uint64_t)low + slot - source_first, record = (uint64_t)target * capacity + slot;
	const uint4 *source = (const uint4 *)(streams + row * width);
	uint4 *target_streams = send + record * vectors;
	for ( uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < vectors; i += gridDim.x * blockDim.x )
		target_streams[i] = source[i];
	if ( blockIdx.x == 0u && threadIdx.x == 0u )
		send_pre[record] = *(const uint4 *)(pre + row * D41K_HC);
}

static __global__ void D41TailUnpackKernel(const uint4 *__restrict__ received, const uint4 *__restrict__ received_pre, uint32_t rank, uint32_t rows,
	uint32_t source_first, uint32_t own, uint32_t first, uint32_t tail_own, uint32_t capacity, uint32_t width, uint16_t *__restrict__ streams,
	float *__restrict__ pre)
{
	const uint32_t source = blockIdx.z, slot = blockIdx.y, vectors = width / 8u;
	uint32_t low;
	if ( slot >= D41TailSpan(source, rank, rows, source_first, own, first, tail_own, &low) )
		return;
	const uint64_t row = (uint64_t)low + slot - first, record = (uint64_t)source * capacity + slot;
	const uint4 *source_streams = received + record * vectors;
	uint4 *target = (uint4 *)(streams + row * width);
	for ( uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < vectors; i += gridDim.x * blockDim.x )
		target[i] = source_streams[i];
	if ( blockIdx.x == 0u && threadIdx.x == 0u )
		*(uint4 *)(pre + row * D41K_HC) = received_pre[record];
}

static __global__ void D41TailRowsPackKernel(const uint4 *__restrict__ rows_data, uint32_t row_vectors, uint32_t record_vectors, uint32_t rank,
	uint32_t rows, uint32_t source_first, uint32_t own, uint32_t first, uint32_t tail_own, uint32_t capacity, uint4 *__restrict__ send)
{
	const uint32_t target = blockIdx.z, slot = blockIdx.y;
	uint32_t low;
	if ( slot >= D41TailSpan(rank, target, rows, source_first, own, first, tail_own, &low) )
		return;
	const uint4 *source = rows_data + (uint64_t)(low + slot) * row_vectors;
	uint4 *record = send + ((uint64_t)target * capacity + slot) * record_vectors;
	for ( uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < row_vectors; i += gridDim.x * blockDim.x )
		record[i] = source[i];
}

static __global__ void D41TailRowsUnpackKernel(const uint4 *__restrict__ received, uint32_t row_vectors, uint32_t record_vectors, uint32_t rank,
	uint32_t rows, uint32_t source_first, uint32_t own, uint32_t first, uint32_t tail_own, uint32_t capacity, uint4 *__restrict__ rows_data)
{
	const uint32_t source = blockIdx.z, slot = blockIdx.y;
	uint32_t low;
	if ( slot >= D41TailSpan(source, rank, rows, source_first, own, first, tail_own, &low) )
		return;
	const uint4 *record = received + ((uint64_t)source * capacity + slot) * record_vectors;
	uint4 *target = rows_data + (uint64_t)(low + slot) * row_vectors;
	for ( uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < row_vectors; i += gridDim.x * blockDim.x )
		target[i] = record[i];
}

#define D41K_ENGRAM_DIM 256u
#define D41K_ENGRAM_SCALES (D41K_ENGRAM_DIM / 32u)
#define D41K_ENGRAM_RECORD_BYTES (D41K_ENGRAM_DIM + 16u)

static __device__ __forceinline__ uint32_t D41EngramSource(int64_t id, uint64_t part, uint64_t table_rows)
{
	return id >= 0 && (uint64_t)id < table_rows ? (uint32_t)((uint64_t)id / part) : UINT32_MAX;
}

template<uint32_t RANKS>
static __global__ void D41EngramSlotsKernel(const int64_t *__restrict__ ids, uint32_t rows, uint32_t own, uint32_t columns, uint64_t part,
	uint64_t table_rows, uint32_t *__restrict__ slot, uint32_t *__restrict__ counts)
{
	__shared__ uint32_t totals[32][RANKS];
	__shared__ uint32_t base[RANKS];
	const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u, warps = blockDim.x >> 5u, owner = blockIdx.x;
	const uint32_t first = owner * own < rows ? owner * own : rows, last = first + own < rows ? first + own : rows;
	const uint64_t begin = (uint64_t)first * columns, end = (uint64_t)last * columns;
	if ( threadIdx.x < RANKS )
		base[threadIdx.x] = 0u;
	__syncthreads();
	for ( uint64_t chunk = begin; chunk < end; chunk += blockDim.x )
	{
		const uint64_t entry = chunk + threadIdx.x;
		const uint32_t source = entry < end ? D41EngramSource(ids[entry], part, table_rows) : UINT32_MAX;
		uint32_t before = 0u;
		for ( uint32_t rank = 0u; rank < RANKS; ++rank )
		{
			const uint32_t vote = __ballot_sync(0xffffffffu, source == rank);
			before = source == rank ? __popc(vote & ((1u << lane) - 1u)) : before;
			if ( lane == 0u )
				totals[warp][rank] = __popc(vote);
		}
		__syncthreads();
		if ( threadIdx.x < RANKS )
		{
			uint32_t running = base[threadIdx.x];
			for ( uint32_t w = 0u; w < warps; ++w )
			{
				const uint32_t here = totals[w][threadIdx.x];
				totals[w][threadIdx.x] = running;
				running += here;
			}
			base[threadIdx.x] = running;
		}
		__syncthreads();
		if ( entry < end )
			slot[entry] = source < RANKS ? totals[warp][source] + before : UINT32_MAX;
		__syncthreads();
	}
	if ( threadIdx.x < RANKS )
		counts[threadIdx.x * RANKS + owner] = base[threadIdx.x];
}

static __global__ void D41EngramPackKernel(const int64_t *__restrict__ ids, const uint32_t *__restrict__ slot, const uint8_t *__restrict__ payload,
	const uint8_t *__restrict__ scale, uint32_t own, uint32_t columns, uint64_t first_row, uint64_t local_rows, uint32_t capacity,
	uint8_t *__restrict__ send)
{
	const uint32_t row = blockIdx.x, lane = threadIdx.x & 31u;
	for ( uint32_t column = threadIdx.x >> 5u; column < columns; column += blockDim.x >> 5u )
	{
		const uint64_t entry = (uint64_t)row * columns + column;
		const int64_t id = ids[entry];
		if ( id < (int64_t)first_row || id >= (int64_t)(first_row + local_rows) )
			continue;
		const uint64_t local = (uint64_t)(id - (int64_t)first_row);
		uint8_t *record = send + ((uint64_t)(row / own) * capacity + slot[entry]) * D41K_ENGRAM_RECORD_BYTES;
		if ( lane < D41K_ENGRAM_DIM / 16u )
			((uint4 *)record)[lane] = ((const uint4 *)(payload + local * D41K_ENGRAM_DIM))[lane];
		else if ( lane == D41K_ENGRAM_DIM / 16u )
			((uint2 *)(record + D41K_ENGRAM_DIM))[0] = ((const uint2 *)(scale + local * D41K_ENGRAM_SCALES))[0];
	}
}

static __global__ void D41EngramUnpackKernel(const int64_t *__restrict__ ids, const uint32_t *__restrict__ slot, const uint8_t *__restrict__ received,
	uint32_t rows, uint32_t first, uint32_t columns, uint64_t part, uint64_t table_rows, uint32_t capacity, uint16_t *__restrict__ embed)
{
	const uint32_t row = first + blockIdx.x, column = blockIdx.y;
	const uint64_t entry = (uint64_t)row * columns + column;
	const uint32_t source = row < rows ? D41EngramSource(ids[entry], part, table_rows) : UINT32_MAX;
	const uint8_t *record = source != UINT32_MAX ? received + ((uint64_t)source * capacity + slot[entry]) * D41K_ENGRAM_RECORD_BYTES : 0;
	uint16_t *target = embed + ((uint64_t)blockIdx.x * columns + column) * D41K_ENGRAM_DIM;
	for ( uint32_t c = threadIdx.x; c < D41K_ENGRAM_DIM; c += blockDim.x )
		target[c] = record != 0 ? LmFloatToBf16(LmE4m3ToFloat(record[c]) * LmE8m0ToFloat(record[D41K_ENGRAM_DIM + c / 32u])) : (uint16_t)0u;
}

static __global__ void D41SumSourcesKernel(const uint16_t *__restrict__ partial, uint16_t *__restrict__ output, uint32_t pairs, uint64_t source_pairs,
	uint32_t sources)
{
	const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
	float x = 0.0f, y = 0.0f;
	if ( pair >= pairs )
		return;
	for ( uint32_t source = 0u; source < sources; ++source )
	{
		const uint32_t packed = ((const uint32_t *)partial)[source * source_pairs + pair];
		x += __uint_as_float(packed << 16u);
		y += __uint_as_float(packed & 0xffff0000u);
	}
	((uint32_t *)output)[pair] = ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(y)) << 16u) | (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(x));
}

static __global__ void D41FinalizeSliceKernel(const uint16_t *__restrict__ slice_out, const uint16_t *__restrict__ received,
	const uint32_t *__restrict__ route_expert, const uint32_t *__restrict__ route_packed_row, uint16_t *__restrict__ output, uint32_t top_k,
	uint32_t local_experts, uint32_t degree, uint32_t hidden_slice, uint64_t source_stride, uint64_t shared_offset,
	const uint16_t *__restrict__ shared_partial)
{
	const uint32_t row = blockIdx.y;
	const uint32_t aligned = (hidden_slice & 7u) == 0u && (((uintptr_t)slice_out | (uintptr_t)output) & 15u) == 0u &&
		(shared_partial != 0 ? ((uintptr_t)shared_partial & 15u) == 0u :
			(((uintptr_t)received & 15u) == 0u && (source_stride & 7u) == 0u && (shared_offset & 7u) == 0u)) ? 1u : 0u;
	if ( aligned != 0u )
	{
		const uint32_t quads = hidden_slice >> 3u;
		uint4 *out_q = (uint4 *)(output + (uint64_t)row * hidden_slice);
		for ( uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < quads; q += gridDim.x * blockDim.x )
		{
			float sum[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			for ( uint32_t source = 0u; source < degree; ++source )
			{
				float total[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
				for ( uint32_t k = 0u; k < top_k; ++k )
				{
					const uint32_t route = row * top_k + k;
					if ( route_expert[route] / local_experts == source )
						D41AddBf16x8(total, ((const uint4 *)(slice_out + (uint64_t)route_packed_row[route] * hidden_slice))[q]);
				}
				const uint4 sh = shared_partial != 0 ?
					((const uint4 *)(shared_partial + ((uint64_t)source * gridDim.y + row) * hidden_slice))[q] :
					((const uint4 *)(received + (uint64_t)source * source_stride + shared_offset + (uint64_t)row * hidden_slice))[q];
				D41AddBf16x8(total, sh);
				for ( uint32_t i = 0u; i < 8u; ++i )
					sum[i] += LmBf16ToFloat(LmFloatToBf16(total[i]));
			}
			out_q[q] = D41RoundBf16x8(sum);
		}
		return;
	}
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < hidden_slice; element += gridDim.x * blockDim.x )
	{
		float sum = 0.0f;
		for ( uint32_t source = 0u; source < degree; ++source )
		{
			float total = 0.0f;
			for ( uint32_t k = 0u; k < top_k; ++k )
			{
				const uint32_t route = row * top_k + k;
				if ( route_expert[route] / local_experts == source )
					total += LmBf16ToFloat(slice_out[(uint64_t)route_packed_row[route] * hidden_slice + element]);
			}
			total += LmBf16ToFloat(shared_partial != 0 ? shared_partial[((uint64_t)source * gridDim.y + row) * hidden_slice + element] :
				received[(uint64_t)source * source_stride + shared_offset + (uint64_t)row * hidden_slice + element]);
			sum += LmBf16ToFloat(LmFloatToBf16(total));
		}
		output[(uint64_t)row * hidden_slice + element] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
	}
}

static __global__ void D41SliceGatherKernel(const uint16_t *__restrict__ gathered, uint16_t *__restrict__ output, uint32_t rows, uint32_t hidden_slice,
	uint32_t degree)
{
	const uint32_t row = blockIdx.y, element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element >= hidden_slice * degree )
		return;
	output[(uint64_t)row * hidden_slice * degree + element] =
		gathered[((uint64_t)(element / hidden_slice) * rows + row) * hidden_slice + element % hidden_slice];
}

static __global__ void D41ExpertActKernel(const uint16_t *__restrict__ gate_up, const float *__restrict__ packed_weight,
	const uint32_t *__restrict__ group_row_offset, uint32_t local_experts, uint8_t *__restrict__ codes, uint16_t *__restrict__ act, uint32_t routes,
	uint32_t width, float limit)
{
	const uint32_t element = blockIdx.x * blockDim.x + threadIdx.x, local = group_row_offset[local_experts];
	const uint32_t count = local < routes ? local : routes;
	if ( element >= width )
		return;
	for ( uint32_t route = blockIdx.y; route < count; route += gridDim.y )
	{
		const uint16_t value = LmSwigluPackedValue(gate_up[(uint64_t)route * 2u * width + element], gate_up[(uint64_t)route * 2u * width + width + element],
			packed_weight[route], limit);
		const float activation = LmBf16ToFloat(value);
		float amax = fabsf(activation);
		for ( uint32_t offset = SPARK_DSV41_FLASH_FP8_BLOCK / 2u; offset != 0u; offset >>= 1u )
			amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
		const float scale = SparkMathFp8QdqScale(fmaxf(0.0f, amax));
		const uint8_t code = LmFloatToE4m3(fminf(fmaxf(activation / scale, -LM_E4M3_MAX), LM_E4M3_MAX));
		const uint32_t exponent = (uint8_t)(__float_as_uint(scale) >> 23u);
		if ( act != 0 )
		{
			act[(uint64_t)route * width + element] = LmFloatToBf16(LmE4m3ToFloat(code) * __uint_as_float(exponent << 23u));
			continue;
		}
		codes[(uint64_t)route * width + element] = code;
		if ( element % SPARK_DSV41_FLASH_FP8_BLOCK == 0u )
			codes[(uint64_t)routes * width + (uint64_t)route * (width / SPARK_DSV41_FLASH_FP8_BLOCK) + element / SPARK_DSV41_FLASH_FP8_BLOCK] =
				(uint8_t)exponent;
	}
}

static void D41SharedPart(uint32_t part, uint32_t parts, uint32_t granules, uint32_t granule, uint32_t *first, uint32_t *width)
{
	const uint32_t base = granules / parts, extra = granules % parts;
	*first = (part * base + (part < extra ? part : extra)) * granule;
	*width = (base + (part < extra ? 1u : 0u)) * granule;
}

static __global__ void D41MoeFinalizeF32Kernel(const uint16_t *__restrict__ packed, const uint32_t *__restrict__ route_packed_row,
	const uint32_t *__restrict__ route_expert, const float *__restrict__ shared, float *__restrict__ output, uint32_t top_k, uint32_t local_experts,
	uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		float total = 0.0f;
		for ( uint32_t k = 0u; k < top_k; ++k )
		{
			const uint32_t route = row * top_k + k;
			if ( route_expert[route] < local_experts )
				total += LmBf16ToFloat(packed[(uint64_t)route_packed_row[route] * width + element]);
		}
		output[(uint64_t)row * width + element] = total + shared[(uint64_t)row * width + element];
	}
}

static __global__ void D41SumSourcesF32Kernel(const float *__restrict__ gathered, uint16_t *__restrict__ output, uint64_t elements, uint32_t degree)
{
	if ( (elements & 7u) == 0u && (((uintptr_t)gathered | (uintptr_t)output) & 15u) == 0u )
	{
		const uint64_t quads = elements >> 3u, stride_f4 = elements >> 2u;
		const float4 *gq = (const float4 *)gathered;
		uint4 *oq = (uint4 *)output;
		for ( uint64_t q = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; q < quads; q += (uint64_t)gridDim.x * blockDim.x )
		{
			float4 s0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), s1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			for ( uint32_t source = 0u; source < degree; ++source )
			{
				const uint64_t off = (uint64_t)source * stride_f4 + (q << 1u);
				const float4 v0 = gq[off], v1 = gq[off + 1u];
				s0.x += v0.x; s0.y += v0.y; s0.z += v0.z; s0.w += v0.w;
				s1.x += v1.x; s1.y += v1.y; s1.z += v1.z; s1.w += v1.w;
			}
			oq[q] = make_uint4(D41PackBf16(s0.x, s0.y), D41PackBf16(s0.z, s0.w), D41PackBf16(s1.x, s1.y), D41PackBf16(s1.z, s1.w));
		}
		return;
	}
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
	{
		float sum = 0.0f;
		for ( uint32_t source = 0u; source < degree; ++source )
			sum += gathered[(uint64_t)source * elements + index];
		output[index] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
	}
}

static __global__ void D41SumGatheredKernel(const uint16_t *__restrict__ gathered, uint16_t *__restrict__ output, uint64_t elements, uint32_t degree)
{
	if ( (elements & 7u) == 0u && (((uintptr_t)gathered | (uintptr_t)output) & 15u) == 0u )
	{
		const uint64_t quads = elements >> 3u;
		const uint4 *gq = (const uint4 *)gathered;
		uint4 *oq = (uint4 *)output;
		for ( uint64_t q = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; q < quads; q += (uint64_t)gridDim.x * blockDim.x )
		{
			float sum[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			for ( uint32_t source = 0u; source < degree; ++source )
				D41AddBf16x8(sum, gq[(uint64_t)source * quads + q]);
			oq[q] = D41RoundBf16x8(sum);
		}
		return;
	}
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
	{
		float sum = 0.0f;
		for ( uint32_t source = 0u; source < degree; ++source )
			sum += LmBf16ToFloat(gathered[(uint64_t)source * elements + index]);
		output[index] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
	}
}
