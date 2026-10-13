#pragma once

#include <math.h>
#include <stdint.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/ngram_hash.cuh"
#include "spark_dsv41_flash_kernels.cuh"

#define D41K_THREADS 256u
#define D41K_HC 4u
#define D41K_HC_MIX ((2u + D41K_HC) * D41K_HC)
#define D41K_HC_TASKS (D41K_HC_MIX + 1u)
#define D41K_HC_WARPS (D41K_THREADS / 32u)
#define D41K_HC_UNROLL 16u
#define D41K_ROPE_PAIRS 32u
#define D41K_LATENT 512u
#define D41K_RECORD (D41K_LATENT + 2u)
#define D41K_NO_ROW 0xffffffffu

static __global__ void D41HcMixPartialKernel(const uint16_t *__restrict__ streams, const float *__restrict__ fn, float *__restrict__ partials,
	uint32_t width)
{
	const uint32_t task = blockIdx.x, row = blockIdx.y, flat = D41K_HC * width, lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	const uint16_t *source = streams + (uint64_t)row * flat;
	const float *weight = task < D41K_HC_MIX ? fn + (uint64_t)task * flat : 0;
	float total = 0.0f;
	for ( uint32_t base = threadIdx.x; base < flat; base += D41K_HC_UNROLL * D41K_THREADS )
	{
		float value[D41K_HC_UNROLL], factor[D41K_HC_UNROLL];
#pragma unroll
		for ( uint32_t step = 0u; step < D41K_HC_UNROLL; ++step )
		{
			const uint32_t element = base + step * D41K_THREADS;
			value[step] = element < flat ? LmBf16ToFloat(source[element]) : 0.0f;
			factor[step] = element >= flat ? 0.0f : weight != 0 ? weight[element] : value[step];
		}
#pragma unroll
		for ( uint32_t step = 0u; step < D41K_HC_UNROLL; ++step )
			if ( base + step * D41K_THREADS < flat )
				total += factor[step] * value[step];
	}
	for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
		total += __shfl_down_sync(0xffffffffu, total, offset);
	if ( lane == 0u )
		partials[((uint64_t)row * D41K_HC_TASKS + task) * D41K_HC_WARPS + warp] = total;
}

static __device__ __forceinline__ float D41HcLineSum(float value, uint32_t first, uint32_t stride)
{
	float sum = 0.0f;
	for ( uint32_t index = 0u; index < D41K_HC; ++index )
		sum += __shfl_sync(0xffffffffu, value, first + index * stride);
	return sum;
}

static __global__ void D41HcSinkhornRowsKernel(const float *__restrict__ partials, const float *__restrict__ scale, const float *__restrict__ base,
	float *__restrict__ pre, float *__restrict__ post, float *__restrict__ comb_out, uint32_t width, float rms_epsilon, uint32_t iterations,
	float epsilon)
{
	__shared__ float mix[D41K_HC_TASKS];
	const uint32_t row = blockIdx.x, lane = threadIdx.x, r = lane / D41K_HC, c = lane % D41K_HC;
	if ( lane < D41K_HC_TASKS )
	{
		const float *part = partials + ((uint64_t)row * D41K_HC_TASKS + lane) * D41K_HC_WARPS;
		float value = 0.0f;
		for ( uint32_t warp = 0u; warp < D41K_HC_WARPS; ++warp )
			value += part[warp];
		mix[lane] = value;
	}
	__syncwarp();
	const float inverse = rsqrtf(mix[D41K_HC_MIX] / (float)(D41K_HC * width) + rms_epsilon);
	const float *m = mix;
	if ( lane < D41K_HC )
		pre[(uint64_t)row * D41K_HC + lane] = 1.0f / (1.0f + expf(-((m[lane] * inverse) * scale[0] + base[lane]))) + epsilon;
	else if ( lane < 2u * D41K_HC )
		post[(uint64_t)row * D41K_HC + lane - D41K_HC] = 2.0f / (1.0f + expf(-((m[lane] * inverse) * scale[1] + base[lane])));
	const uint32_t cell = lane % (D41K_HC * D41K_HC);
	float comb = (m[2u * D41K_HC + cell] * inverse) * scale[2] + base[2u * D41K_HC + cell];
	float top = -INFINITY;
	for ( uint32_t index = 0u; index < D41K_HC; ++index )
		top = fmaxf(top, __shfl_sync(0xffffffffu, comb, r * D41K_HC + index));
	comb = expf(comb - top);
	comb = comb / D41HcLineSum(comb, r * D41K_HC, 1u) + epsilon;
	comb /= D41HcLineSum(comb, c, D41K_HC) + epsilon;
	for ( uint32_t iteration = 1u; iteration < iterations; ++iteration )
	{
		comb /= D41HcLineSum(comb, r * D41K_HC, 1u) + epsilon;
		comb /= D41HcLineSum(comb, c, D41K_HC) + epsilon;
	}
	if ( lane < D41K_HC * D41K_HC )
		comb_out[(uint64_t)row * D41K_HC * D41K_HC + lane] = comb;
}

static __global__ void D41HcPreRowsKernel(const uint16_t *__restrict__ streams, const float *__restrict__ pre, uint16_t *__restrict__ output,
	uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		float total = 0.0f;
		for ( uint32_t copy = 0u; copy < D41K_HC; ++copy )
			total += pre[(uint64_t)row * D41K_HC + copy] * LmBf16ToFloat(streams[((uint64_t)row * D41K_HC + copy) * width + element]);
		output[(uint64_t)row * width + element] = LmFloatToBf16(total);
	}
}

static __global__ void D41HcPostRowsKernel(const uint16_t *__restrict__ sublayer, const uint16_t *__restrict__ residual,
	const float *__restrict__ post, const float *__restrict__ comb, uint16_t *__restrict__ streams, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
	{
		const float x = LmBf16ToFloat(sublayer[(uint64_t)row * width + element]);
		float source[D41K_HC];
		for ( uint32_t copy = 0u; copy < D41K_HC; ++copy )
			source[copy] = LmBf16ToFloat(residual[((uint64_t)row * D41K_HC + copy) * width + element]);
		for ( uint32_t copy = 0u; copy < D41K_HC; ++copy )
		{
			float total = 0.0f;
			for ( uint32_t from = 0u; from < D41K_HC; ++from )
				total += comb[(uint64_t)row * D41K_HC * D41K_HC + from * D41K_HC + copy] * source[from];
			streams[((uint64_t)row * D41K_HC + copy) * width + element] = LmFloatToBf16(post[(uint64_t)row * D41K_HC + copy] * x + total);
		}
	}
}

static __global__ void D41ReplicateStreamsKernel(const uint16_t *__restrict__ hidden, uint16_t *__restrict__ streams, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t element = blockIdx.x * blockDim.x + threadIdx.x; element < width; element += gridDim.x * blockDim.x )
		for ( uint32_t copy = 0u; copy < D41K_HC; ++copy )
			streams[((uint64_t)row * D41K_HC + copy) * width + element] = hidden[(uint64_t)row * width + element];
}

static __global__ void D41RopeRowsKernel(uint16_t *__restrict__ data, uint32_t heads, uint32_t head_stride, uint32_t rope_offset,
	const uint32_t *__restrict__ positions, int32_t shift, const double *__restrict__ inverse_frequency, uint32_t inverse,
	const uint32_t *__restrict__ row_valid)
{
	const uint32_t row = blockIdx.x, head = blockIdx.y, pair = threadIdx.x;
	if ( pair >= D41K_ROPE_PAIRS || head >= heads || (row_valid != 0 && row_valid[row] == D41K_NO_ROW) )
		return;
	const double angle = (double)((int64_t)positions[row] + shift) * inverse_frequency[pair];
	double sine, cosine;
	sincos(angle, &sine, &cosine);
	if ( inverse != 0u )
		sine = -sine;
	uint16_t *value = data + ((uint64_t)row * heads + head) * head_stride + rope_offset + 2u * pair;
	const float real = LmBf16ToFloat(value[0]), imaginary = LmBf16ToFloat(value[1]);
	value[0] = LmFloatToBf16((float)((double)real * cosine - (double)imaginary * sine));
	value[1] = LmFloatToBf16((float)((double)imaginary * cosine + (double)real * sine));
}

static __global__ void D41DequantFp8BlockKernel(const uint8_t *__restrict__ payload, const uint8_t *__restrict__ scale, uint16_t *__restrict__ output,
	uint32_t columns, uint32_t row_first, uint32_t column_first, uint32_t output_columns)
{
	const uint32_t row = row_first + blockIdx.y;
	const uint32_t scale_columns = columns / 32u;
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < output_columns; index += gridDim.x * blockDim.x )
	{
		const uint32_t column = column_first + index;
		const float factor = LmE8m0ToFloat(scale[(uint64_t)(row / 32u) * scale_columns + column / 32u]);
		output[(uint64_t)blockIdx.y * output_columns + index] = LmFloatToBf16(LmE4m3ToFloat(payload[(uint64_t)row * columns + column]) * factor);
	}
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
		const float top = fmaxf(score0, score1);
		const float e0 = expf(score0 - top), e1 = expf(score1 - top);
		pooled[(uint64_t)row * width + c] = LmFloatToBf16((kv0 * e0 + kv1 * e1) / (e0 + e1));
	}
	if ( threadIdx.x == 0u )
		emit_row[row] = position / ratio;
}

static __global__ void D41CompressorSaveKernel(const float *__restrict__ kv, const float *__restrict__ score, float *__restrict__ state,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ recurrent_slot, uint32_t rows, uint32_t ratio,
	uint32_t state_stride_floats, uint32_t width, uint32_t commit)
{
	const uint32_t row = blockIdx.x;
	if ( commit == 0u || row >= rows || (positions[row] + 1u) % ratio == 0u )
		return;
	const uint32_t lane = row_lane[row];
	if ( row_ordinal[row] + 1u != sequence_row_begin[lane + 1u] - sequence_row_begin[lane] )
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

static __global__ void D41InitRecordsKernel(float *__restrict__ records, uint64_t destination_stride, uint32_t rows, uint32_t heads_per_destination,
	uint32_t destinations)
{
	const uint64_t pair = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if ( pair >= (uint64_t)rows * heads_per_destination * destinations )
		return;
	const uint32_t destination = (uint32_t)(pair / ((uint64_t)rows * heads_per_destination));
	float *record = records + destination * destination_stride + (pair % ((uint64_t)rows * heads_per_destination)) * D41K_RECORD;
	record[0] = -INFINITY;
	record[1] = 0.0f;
	for ( uint32_t element = 0u; element < D41K_LATENT; ++element )
		record[2u + element] = 0.0f;
}

static __global__ void D41WindowFoldKernel(float *__restrict__ records, uint64_t destination_stride, uint32_t heads_per_destination,
	const uint16_t *__restrict__ query, uint64_t query_rank_stride, const uint16_t *__restrict__ wave_kv,
	const uint16_t *__restrict__ ring, const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane,
	const uint32_t *__restrict__ row_ordinal, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices,
	const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows, uint32_t window, uint32_t degree, uint32_t rank,
	float qk_scale)
{
	__shared__ float score_shared[D41K_THREADS / 32u];
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	const uint32_t position = positions[row], lane = row_lane[row];
	const uint32_t wave_first = position - row_ordinal[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const uint32_t first = position + 1u > window ? position + 1u - window : 0u;
	const uint32_t destination = head / heads_per_destination, local = head % heads_per_destination;
	const uint16_t *q = query + (uint64_t)destination * query_rank_stride + ((uint64_t)row * heads_per_destination + local) * D41K_LATENT;
	float *record = records + destination * destination_stride + ((uint64_t)row * heads_per_destination + local) * D41K_RECORD;
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
	for ( uint32_t e = 0u; e < D41K_LATENT / D41K_THREADS; ++e )
		record[2u + threadIdx.x + e * D41K_THREADS] = accumulator[e];
	if ( threadIdx.x == 0u )
	{
		record[0] = running_max;
		record[1] = running_sum;
	}
}

static __global__ void D41WindowRingStoreKernel(const uint16_t *__restrict__ wave_kv, uint16_t *__restrict__ ring,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows,
	uint32_t window, uint32_t degree, uint32_t rank, uint32_t rows, uint32_t commit)
{
	const uint32_t row = blockIdx.x;
	if ( commit == 0u || row >= rows || positions[row] % degree != rank )
		return;
	const uint32_t lane = row_lane[row];
	if ( row_ordinal[row] + window < sequence_row_begin[lane + 1u] - sequence_row_begin[lane] )
		return;
	uint16_t *target = ring + (uint64_t)recurrent_slot[lane] * ring_slot_stride +
		(uint64_t)((positions[row] / degree) % ring_rows) * D41K_LATENT;
	for ( uint32_t c = threadIdx.x; c < D41K_LATENT; c += blockDim.x )
		target[c] = wave_kv[(uint64_t)row * D41K_LATENT + c];
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

static __global__ void D41ScaleRowsKernel(uint16_t *__restrict__ data, uint64_t elements, float factor)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
		data[index] = LmFloatToBf16(LmBf16ToFloat(data[index]) * factor);
}
