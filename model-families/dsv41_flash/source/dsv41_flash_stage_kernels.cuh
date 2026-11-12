#pragma once

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/ngram_hash.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "dsv41_flash_kernels.cuh"

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

static inline cudaError_t D41LaunchFrequencies(float *table, double theta, double compress_theta, double factor, double original,
	double beta_fast, double beta_slow)
{
	const double rope = 2.0 * D41K_ROPE_PAIRS;
	const double low_raw = rope * log(original / (beta_fast * 2.0 * M_PI)) / (2.0 * log(compress_theta));
	const double high_raw = rope * log(original / (beta_slow * 2.0 * M_PI)) / (2.0 * log(compress_theta));
	const double low = floor(low_raw) > 0.0 ? floor(low_raw) : 0.0;
	const double high = ceil(high_raw) < rope - 1.0 ? ceil(high_raw) : rope - 1.0;
	const double span = high - low > 1e-3 ? high - low : 1e-3;
	D41FrequencyKernel<<<1, D41K_ROPE_PAIRS>>>(table, (float)theta, 1.0f / (float)factor, 0.0f, 0.0f, 0u);
	D41FrequencyKernel<<<1, D41K_ROPE_PAIRS>>>(table + D41K_ROPE_PAIRS, (float)compress_theta, 1.0f / (float)factor, (float)low,
		1.0f / (float)span, 1u);
	return cudaGetLastError();
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

static __global__ void D41RopeRowsKernel(uint16_t *__restrict__ data, uint32_t heads, uint32_t head_stride, uint32_t rope_offset,
	const uint32_t *__restrict__ positions, int32_t shift, const float *__restrict__ inverse_frequency, uint32_t inverse,
	const uint32_t *__restrict__ row_valid)
{
	const uint32_t row = blockIdx.x, head = blockIdx.y, pair = threadIdx.x;
	if ( pair >= D41K_ROPE_PAIRS || head >= heads || (row_valid != 0 && row_valid[row] == D41K_NO_ROW) )
		return;
	float sine, cosine;
	D41RopeTurn(positions[row], shift, inverse_frequency[pair], inverse, &sine, &cosine);
	uint16_t *value = data + ((uint64_t)row * heads + head) * head_stride + rope_offset + 2u * pair;
	const float real = LmBf16ToFloat(value[0]), imaginary = LmBf16ToFloat(value[1]);
	value[0] = D41RopeReal(real, imaginary, sine, cosine);
	value[1] = D41RopeImaginary(real, imaginary, sine, cosine);
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
	cudaStream_t stream)
{
	const uint32_t span = rows * heads_per_destination * D41K_RECORD, blocks = (span + D41K_THREADS - 1u) / D41K_THREADS;
	D41InitRecordsKernel<<<blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u, D41K_THREADS, 0, stream>>>(records, destination_stride, span, destinations);
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

static __global__ void D41WindowFoldWholeKernel(float *__restrict__ records, uint32_t heads, const uint16_t *__restrict__ query,
	const uint16_t *__restrict__ wave_kv, const uint16_t *__restrict__ ring, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal, const uint32_t *__restrict__ sequence_row_begin,
	const uint32_t *__restrict__ sequence_row_indices, const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t window,
	float qk_scale)
{
	__shared__ float warp_sums[D41K_WINDOW_KEYS][D41K_THREADS / 32u];
	__shared__ float scores[D41K_WINDOW_KEYS];
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
	if ( threadIdx.x < count )
	{
		float total = 0.0f;
		for ( uint32_t w = 0u; w < D41K_THREADS / 32u; ++w )
			total += warp_sums[threadIdx.x][w];
		scores[threadIdx.x] = total * qk_scale;
	}
	__syncthreads();
	if ( threadIdx.x < count )
	{
		float before = carried_max;
		for ( uint32_t j = 0u; j < threadIdx.x; ++j )
			before = fmaxf(before, scores[j]);
		const float top = fmaxf(before, scores[threadIdx.x]);
		keep[threadIdx.x] = before == -INFINITY ? 0.0f : __expf(before - top);
		added[threadIdx.x] = __expf(scores[threadIdx.x] - top);
		if ( threadIdx.x + 1u == count )
			folded[0] = top;
	}
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

static __global__ void D41WindowRingStoreKernel(const uint16_t *__restrict__ wave_kv, uint16_t *__restrict__ ring,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows,
	uint32_t window, uint32_t degree, uint32_t rank, uint32_t rows, uint32_t commit, const uint32_t *__restrict__ lane_count)
{
	const uint32_t row = blockIdx.x;
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

#define D41K_FOLD_BLOCK 128u
#define D41K_FOLD_SPAN (2u * D41K_FOLD_BLOCK)
#define D41K_FOLD_WARPS (D41K_THREADS / 32u)

static __global__ void D41FoldPlanKernel(const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices,
	uint32_t lanes, uint32_t rows, uint32_t heads, uint32_t groups, uint32_t *__restrict__ group_offset, uint32_t *__restrict__ source_map,
	uint32_t *__restrict__ row_packed, uint32_t *__restrict__ block_lane, uint32_t *__restrict__ block_first)
{
	if ( threadIdx.x == 0u )
	{
		uint32_t block = 0u;
		for ( uint32_t lane = 0u; lane < lanes; ++lane )
		{
			const uint32_t begin = sequence_row_begin[lane], count = sequence_row_begin[lane + 1u] - begin;
			for ( uint32_t first = 0u; first < count && block < groups; first += D41K_FOLD_BLOCK, ++block )
			{
				group_offset[block] = (begin + first) * heads;
				block_lane[block] = lane;
				block_first[block] = first;
			}
		}
		for ( ; block < groups; ++block )
		{
			group_offset[block] = rows * heads;
			block_lane[block] = D41K_NO_ROW;
			block_first[block] = 0u;
		}
		group_offset[groups] = rows * heads;
	}
	for ( uint32_t packed = threadIdx.x; packed < rows; packed += blockDim.x )
	{
		const uint32_t row = sequence_row_indices[packed];
		row_packed[row] = packed;
		for ( uint32_t head = 0u; head < heads; ++head )
			source_map[packed * heads + head] = row * heads + head;
	}
}

static __global__ void D41FoldKeysKernel(const uint16_t *__restrict__ wave_kv, const uint16_t *__restrict__ ring,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices,
	const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows, const uint32_t *__restrict__ block_lane,
	const uint32_t *__restrict__ block_first, uint16_t *__restrict__ keys, uint16_t *__restrict__ keys_transposed)
{
	const uint32_t block = blockIdx.y, column = blockIdx.x, lane = block_lane[block];
	if ( lane == D41K_NO_ROW )
		return;
	const uint32_t begin = sequence_row_begin[lane], count = sequence_row_begin[lane + 1u] - begin;
	const uint32_t wave_first = positions[sequence_row_indices[begin]];
	const int64_t position = (int64_t)wave_first + block_first[block] + column - (int64_t)(D41K_FOLD_BLOCK - 1u);
	const uint16_t *source = 0;
	if ( position >= (int64_t)wave_first )
	{
		if ( position - wave_first < count )
			source = wave_kv + (uint64_t)sequence_row_indices[begin + (uint32_t)(position - wave_first)] * D41K_LATENT;
	}
	else if ( position >= 0 )
		source = ring + (uint64_t)recurrent_slot[lane] * ring_slot_stride + (uint64_t)((uint32_t)position % ring_rows) * D41K_LATENT;
	uint16_t *row_out = keys + ((uint64_t)block * D41K_FOLD_SPAN + column) * D41K_LATENT;
	uint16_t *column_out = keys_transposed + (uint64_t)block * D41K_LATENT * D41K_FOLD_SPAN + column;
	for ( uint32_t element = threadIdx.x; element < D41K_LATENT; element += blockDim.x )
	{
		const uint16_t value = source != 0 ? source[element] : (uint16_t)0u;
		row_out[element] = value;
		column_out[(uint64_t)element * D41K_FOLD_SPAN] = value;
	}
}

static __global__ void D41FoldWeightsKernel(const float *__restrict__ scores, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ sequence_row_indices, const uint32_t *__restrict__ row_ordinal, uint32_t heads, uint32_t packed_rows,
	float qk_scale, uint16_t *__restrict__ weights, float *__restrict__ records)
{
	const uint32_t lane = threadIdx.x % 32u, packed = blockIdx.x * D41K_FOLD_WARPS + threadIdx.x / 32u;
	if ( packed >= packed_rows )
		return;
	const uint32_t row = sequence_row_indices[packed / heads], position = positions[row], within = row_ordinal[row] % D41K_FOLD_BLOCK;
	const uint32_t last = within + D41K_FOLD_BLOCK - 1u, first = position >= D41K_FOLD_BLOCK - 1u ? within : last - position;
	const float *score = scores + (uint64_t)packed * D41K_FOLD_SPAN;
	uint16_t *weight = weights + (uint64_t)packed * D41K_FOLD_SPAN;
	float value[D41K_FOLD_SPAN / 32u], top = -INFINITY, sum = 0.0f;
	for ( uint32_t part = 0u; part < D41K_FOLD_SPAN / 32u; ++part )
	{
		const uint32_t column = part * 32u + lane;
		value[part] = column >= first && column <= last ? score[column] * qk_scale : -INFINITY;
		top = fmaxf(top, value[part]);
	}
	for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
		top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset));
	for ( uint32_t part = 0u; part < D41K_FOLD_SPAN / 32u; ++part )
	{
		const float add = value[part] == -INFINITY ? 0.0f : __expf(value[part] - top);
		sum += add;
		weight[part * 32u + lane] = LmFloatToBf16(SparkMathAttentionValueWeight(add));
	}
	for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
		sum += __shfl_xor_sync(0xffffffffu, sum, offset);
	if ( lane == 0u )
	{
		records[(uint64_t)packed * D41K_RECORD] = top;
		records[(uint64_t)packed * D41K_RECORD + 1u] = sum;
	}
}

typedef struct D41WindowFoldPlan
{
	uint32_t *group_offset, *source_map, *row_packed, *block_lane, *block_first;
	uint16_t *keys, *keys_transposed, *weights;
	float *scores;
	uint32_t groups;
} D41WindowFoldPlan;

static inline uint32_t D41WindowFoldGroups(uint32_t rows, uint32_t lanes)
{
	return rows / D41K_FOLD_BLOCK + lanes;
}

static inline cudaError_t D41WindowFoldPlanLaunch(D41WindowFoldPlan *plan, const uint32_t *sequence_row_begin, const uint32_t *sequence_row_indices,
	uint32_t lanes, uint32_t rows, uint32_t heads, cudaStream_t stream)
{
	plan->groups = D41WindowFoldGroups(rows, lanes);
	D41FoldPlanKernel<<<1u, D41K_THREADS, 0, stream>>>(sequence_row_begin, sequence_row_indices, lanes, rows, heads, plan->groups, plan->group_offset,
		plan->source_map, plan->row_packed, plan->block_lane, plan->block_first);
	return cudaPeekAtLastError();
}

static inline int32_t D41WindowFoldWaveLaunch(const D41WindowFoldPlan *plan, const uint16_t *query, const uint16_t *wave_kv, const uint16_t *ring,
	const uint32_t *positions, const uint32_t *row_ordinal, const uint32_t *sequence_row_begin, const uint32_t *sequence_row_indices,
	const uint32_t *recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows, uint32_t rows, uint32_t heads, float qk_scale, float *window,
	uint32_t multiprocessors, cudaStream_t stream)
{
	const uint32_t packed = rows * heads;
	LmStreamGemmArguments args;
	int32_t status;
	D41FoldKeysKernel<<<dim3(D41K_FOLD_SPAN, plan->groups), D41K_THREADS, 0, stream>>>(wave_kv, ring, positions, sequence_row_begin,
		sequence_row_indices, recurrent_slot, ring_slot_stride, ring_rows, plan->block_lane, plan->block_first, plan->keys, plan->keys_transposed);
	memset(&args, 0, sizeof(args));
	args.activation = query;
	args.weight = (const uint8_t *)plan->keys;
	args.group_row_offset = plan->group_offset;
	args.source_row_map = plan->source_map;
	args.output_f32 = plan->scores;
	args.weight_group_bytes = (uint64_t)D41K_FOLD_SPAN * D41K_LATENT * sizeof(uint16_t);
	args.rows = packed;
	args.groups = plan->groups;
	args.input_dimension = D41K_LATENT;
	args.output_dimension = D41K_FOLD_SPAN;
	args.output_row_stride = D41K_FOLD_SPAN;
	status = LmStreamGemmLaunch<LmBf16Format>(&args, multiprocessors, stream);
	if ( status != LM_LAUNCH_OK )
		return status;
	D41FoldWeightsKernel<<<(packed + D41K_FOLD_WARPS - 1u) / D41K_FOLD_WARPS, D41K_THREADS, 0, stream>>>(plan->scores, positions, sequence_row_indices,
		row_ordinal, heads, packed, qk_scale, plan->weights, window);
	args.activation = plan->weights;
	args.weight = (const uint8_t *)plan->keys_transposed;
	args.source_row_map = 0;
	args.output_f32 = window;
	args.input_dimension = D41K_FOLD_SPAN;
	args.output_dimension = D41K_LATENT;
	args.output_row_stride = D41K_RECORD;
	args.output_column_offset = 2u;
	return LmStreamGemmLaunch<LmBf16Format>(&args, multiprocessors, stream);
}

static __global__ void D41MergeWindowSinkKernel(const float *__restrict__ records, const float *__restrict__ window,
	const uint32_t *__restrict__ row_packed, uint32_t heads, const float *__restrict__ sink, uint16_t *__restrict__ output)
{
	__shared__ float scales[2];
	__shared__ float denominator_shared;
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	const uint64_t base = ((uint64_t)row * heads + head) * D41K_RECORD, window_base = ((uint64_t)row_packed[row] * heads + head) * D41K_RECORD;
	if ( threadIdx.x == 0u )
	{
		const float listed = records[base], near = window[window_base];
		const float top = fmaxf(fmaxf(-1.0e30f, listed), near);
		float denominator = 0.0f;
		scales[0] = listed == -INFINITY ? 0.0f : __expf(listed - top);
		scales[1] = near == -INFINITY ? 0.0f : __expf(near - top);
		denominator = fmaf(records[base + 1u], scales[0], denominator);
		denominator = fmaf(window[window_base + 1u], scales[1], denominator);
		denominator_shared = denominator + __expf(sink[head] - top);
	}
	__syncthreads();
	for ( uint32_t element = threadIdx.x; element < D41K_LATENT; element += blockDim.x )
	{
		float merged = fmaf(records[base + 2u + element], scales[0], 0.0f);
		merged = fmaf(window[window_base + 2u + element], scales[1], merged);
		output[((uint64_t)row * heads + head) * D41K_LATENT + element] = LmFloatToBf16(merged / denominator_shared);
	}
}

static __global__ void D41MergeSinkKernel(const float *__restrict__ partials, uint64_t source_stride, uint32_t sources, uint32_t heads,
	const float *__restrict__ sink, uint16_t *__restrict__ output)
{
	__shared__ float scales[16];
	__shared__ float denominator_shared;
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	const uint64_t base = ((uint64_t)row * heads + head) * D41K_RECORD;
	if ( threadIdx.x == 0u )
	{
		float top = -1.0e30f, denominator = 0.0f;
		for ( uint32_t source = 0u; source < sources; ++source )
			top = fmaxf(top, partials[(uint64_t)source * source_stride + base]);
		for ( uint32_t source = 0u; source < sources; ++source )
		{
			const float maximum = partials[(uint64_t)source * source_stride + base];
			scales[source] = maximum == -INFINITY ? 0.0f : __expf(maximum - top);
			denominator = fmaf(partials[(uint64_t)source * source_stride + base + 1u], scales[source], denominator);
		}
		denominator_shared = denominator + __expf(sink[head] - top);
	}
	__syncthreads();
	for ( uint32_t element = threadIdx.x; element < D41K_LATENT; element += blockDim.x )
	{
		float merged = 0.0f;
		for ( uint32_t source = 0u; source < sources; ++source )
			merged = fmaf(partials[(uint64_t)source * source_stride + base + 2u + element], scales[source], merged);
		output[((uint64_t)row * heads + head) * D41K_LATENT + element] = LmFloatToBf16(merged / denominator_shared);
	}
}

static __global__ void D41SwigluRowsKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ output,
	uint32_t width, float limit)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		const float g = fminf(LmBf16ToFloat(gate[(uint64_t)row * width + element]), limit);
		const float u = fminf(fmaxf(LmBf16ToFloat(up[(uint64_t)row * width + element]), -limit), limit);
		output[(uint64_t)row * width + element] = LmFloatToBf16((g / (1.0f + expf(-g))) * u);
	}
}

static __global__ void D41AddRowsKernel(uint16_t *__restrict__ hidden, const uint16_t *__restrict__ addend, uint64_t elements)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
		hidden[index] = LmFloatToBf16(LmBf16ToFloat(hidden[index]) + LmBf16ToFloat(addend[index]));
}

static __global__ void D41GroupInputKernel(const uint16_t *__restrict__ gathered, uint16_t *__restrict__ group_input, uint32_t rows,
	uint32_t width_per_rank, uint32_t first_rank, uint32_t ranks)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width_per_rank * ranks; index += gridDim.x * blockDim.x )
	{
		const uint32_t rank = first_rank + index / width_per_rank, element = index % width_per_rank;
		group_input[(uint64_t)row * width_per_rank * ranks + index] = gathered[((uint64_t)rank * rows + row) * width_per_rank + element];
	}
}

static __global__ void D41ColumnsKernel(const uint16_t *__restrict__ source, uint32_t source_stride, uint16_t *__restrict__ target,
	uint32_t target_stride, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += gridDim.x * blockDim.x )
		target[(uint64_t)row * target_stride + index] = source[(uint64_t)row * source_stride + index];
}

static __global__ void D41NaiveLinearKernel(const uint16_t *__restrict__ weight, const uint16_t *__restrict__ input, uint16_t *__restrict__ output,
	float *__restrict__ output_f32, uint32_t input_dimension, uint32_t output_dimension)
{
	const uint32_t lane = threadIdx.x % 32u, warp = threadIdx.x / 32u, row = blockIdx.y;
	const uint32_t neuron = blockIdx.x * (blockDim.x / 32u) + warp;
	if ( neuron >= output_dimension )
		return;
	const uint16_t *w = weight + (uint64_t)neuron * input_dimension, *x = input + (uint64_t)row * input_dimension;
	float total = 0.0f;
	for ( uint32_t k = lane; k < input_dimension; k += 32u )
		total += LmBf16ToFloat(w[k]) * LmBf16ToFloat(x[k]);
	for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
		total += __shfl_down_sync(0xffffffffu, total, offset);
	if ( lane != 0u )
		return;
	if ( output != 0 )
		output[(uint64_t)row * output_dimension + neuron] = LmFloatToBf16(total);
	if ( output_f32 != 0 )
		output_f32[(uint64_t)row * output_dimension + neuron] = total;
}

static __global__ void D41IndexPackKernel(const uint16_t *__restrict__ query, const uint16_t *__restrict__ weight, uint16_t *__restrict__ packed,
	uint32_t query_width, uint32_t weight_width, uint32_t pack_width)
{
	const uint32_t row = blockIdx.x;
	for ( uint32_t element = threadIdx.x; element < pack_width; element += blockDim.x )
		packed[(uint64_t)row * pack_width + element] = element < query_width ? query[(uint64_t)row * query_width + element] :
			element < query_width + weight_width ? weight[(uint64_t)row * weight_width + element - query_width] : (uint16_t)0u;
}

static __global__ void D41IndexUnpackKernel(const uint16_t *__restrict__ packed, uint64_t rank_stride, uint16_t *__restrict__ query,
	uint16_t *__restrict__ weight, uint32_t query_width, uint32_t weight_width, uint32_t pack_width)
{
	const uint32_t row = blockIdx.x, rank = blockIdx.y, ranks = gridDim.y;
	const uint16_t *source = packed + (uint64_t)rank * rank_stride + (uint64_t)row * pack_width;
	for ( uint32_t element = threadIdx.x; element < query_width + weight_width; element += blockDim.x )
		if ( element < query_width )
			query[((uint64_t)row * ranks + rank) * query_width + element] = source[element];
		else
			weight[((uint64_t)row * ranks + rank) * weight_width + element - query_width] = source[element];
}

static __global__ void D41ScaleRowsKernel(uint16_t *__restrict__ data, uint64_t elements, float factor)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
		data[index] = LmFloatToBf16(LmBf16ToFloat(data[index]) * factor);
}

static __global__ void D41MoeSendKernel(const uint16_t *__restrict__ gate_up, const float *__restrict__ packed_weight,
	const uint16_t *__restrict__ shared, uint16_t *__restrict__ send, uint32_t routes, uint32_t slice, uint32_t hidden_slice, uint32_t degree,
	uint64_t destination_stride, float limit, uint32_t route_elements, uint32_t codes)
{
	const uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( blockIdx.y < routes )
	{
		const uint32_t route = blockIdx.y;
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
	const uint32_t row = blockIdx.y - routes;
	if ( element >= hidden_slice * degree )
		return;
	send[(uint64_t)(element / hidden_slice) * destination_stride + (uint64_t)routes * route_elements + (uint64_t)row * hidden_slice +
		element % hidden_slice] = shared[(uint64_t)row * hidden_slice * degree + element];
}

static __global__ void D41ActAssembleQdqKernel(const uint16_t *__restrict__ received, uint16_t *__restrict__ act, uint32_t slice, uint32_t width,
	uint64_t source_stride, uint32_t routes, uint32_t codes)
{
	const uint32_t route = blockIdx.y, element = blockIdx.x * blockDim.x + threadIdx.x, source = element / slice;
	if ( element >= width )
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

static __global__ void D41FinalizeSliceKernel(const uint16_t *__restrict__ slice_out, const uint16_t *__restrict__ received,
	const uint32_t *__restrict__ route_expert, const uint32_t *__restrict__ route_packed_row, uint16_t *__restrict__ output, uint32_t top_k,
	uint32_t local_experts, uint32_t degree, uint32_t hidden_slice, uint64_t source_stride, uint64_t shared_offset)
{
	const uint32_t row = blockIdx.y, element = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	if ( element >= hidden_slice )
		return;
	for ( uint32_t source = 0u; source < degree; ++source )
	{
		float total = 0.0f;
		for ( uint32_t k = 0u; k < top_k; ++k )
		{
			const uint32_t route = row * top_k + k;
			if ( route_expert[route] / local_experts == source )
				total += LmBf16ToFloat(slice_out[(uint64_t)route_packed_row[route] * hidden_slice + element]);
		}
		total += LmBf16ToFloat(received[(uint64_t)source * source_stride + shared_offset + (uint64_t)row * hidden_slice + element]);
		sum += LmBf16ToFloat(LmFloatToBf16(total));
	}
	output[(uint64_t)row * hidden_slice + element] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
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
