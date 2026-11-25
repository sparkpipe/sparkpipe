#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "inference/kernels/cell_mx.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/fused/arrival.cuh"
#include "inference/kernels/hc.cuh"
#include "inference/kernels/index_score.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/ngram_hash.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/topk_warp.cuh"
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
	const uint32_t bits = *(const uint32_t *)value;
	const float real = __uint_as_float(bits << 16u), imaginary = __uint_as_float(bits & 0xffff0000u);
	*(uint32_t *)value = (uint32_t)D41RopeReal(real, imaginary, sine, cosine) | ((uint32_t)D41RopeImaginary(real, imaginary, sine, cosine) << 16u);
}

static __global__ void D41RopeTableKernel(const uint32_t *__restrict__ positions, const float *__restrict__ freq, float2 *__restrict__ full_sc,
	float2 *__restrict__ comp_sc, uint32_t rows)
{
	const uint32_t row = blockIdx.x, pair = threadIdx.x;
	LmDependentWait();
	LmDependentRelease();
	if ( row >= rows || pair >= D41K_ROPE_PAIRS )
		return;
	const uint32_t pos = positions[row];
	float s0, c0, s1, c1;
	D41RopeTurn(pos, 0, freq[pair], 0u, &s0, &c0);
	D41RopeTurn(pos, 0, freq[D41K_ROPE_PAIRS + pair], 0u, &s1, &c1);
	full_sc[(uint64_t)row * D41K_ROPE_PAIRS + pair] = make_float2(s0, c0);
	comp_sc[(uint64_t)row * D41K_ROPE_PAIRS + pair] = make_float2(s1, c1);
}

static __device__ __forceinline__ void D41RopePairSc(uint16_t *value, float2 sc, uint32_t inverse)
{
	const float sine = inverse != 0u ? -sc.x : sc.x, cosine = sc.y;
	const uint32_t bits = *(const uint32_t *)value;
	const float real = __uint_as_float(bits << 16u), imaginary = __uint_as_float(bits & 0xffff0000u);
	*(uint32_t *)value = (uint32_t)D41RopeReal(real, imaginary, sine, cosine) | ((uint32_t)D41RopeImaginary(real, imaginary, sine, cosine) << 16u);
}

static __global__ void D41RopeRowsKernel(uint16_t *__restrict__ data, uint32_t heads, uint32_t head_stride, uint32_t rope_offset,
	const uint32_t *__restrict__ positions, int32_t shift, const float *__restrict__ inverse_frequency, uint32_t inverse,
	const uint32_t *__restrict__ row_valid, const float2 *__restrict__ rope_sc = 0)
{
	const uint32_t row = blockIdx.x, head = blockIdx.y, pair = threadIdx.x;
	LmDependentWait();
	if ( pair < D41K_ROPE_PAIRS && head < heads && (row_valid == 0 || row_valid[row] != D41K_NO_ROW) )
	{
		uint16_t *target = data + ((uint64_t)row * heads + head) * head_stride + rope_offset + 2u * pair;
		if ( rope_sc != 0 && shift == 0 )
			D41RopePairSc(target, rope_sc[(uint64_t)row * D41K_ROPE_PAIRS + pair], inverse);
		else
			D41RopePair(target, positions[row], shift, inverse_frequency[pair], inverse);
	}
	LmDependentRelease();
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
	const uint32_t *__restrict__ recurrent_slot, uint32_t ratio, uint32_t state_stride_floats, uint32_t width,
	const uint32_t *__restrict__ slots = 0, const uint32_t *__restrict__ parent_rows = 0, const uint16_t *__restrict__ norm = 0,
	float norm_eps = 0.0f, uint32_t save_inline = 0u)
{
	__shared__ float s_val[D41K_LATENT];
	__shared__ float s_red[D41K_THREADS / 32u];
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	const uint32_t position = positions[row], target_slot = slots != 0 ? slots[row] : position;
	if ( ratio == 1u )
	{
		float ss = 0.0f;
		for ( uint32_t c = threadIdx.x; c < width; c += blockDim.x )
		{
			const uint16_t bf = LmFloatToBf16(kv[(uint64_t)row * width + c]);
			if ( norm != 0 ) { const float v = LmBf16ToFloat(bf); s_val[c] = v; ss += v * v; }
			else pooled[(uint64_t)row * width + c] = bf;
		}
		if ( norm != 0 )
		{
			const float sc = SparkMathRmsScale(LmBlockSum<D41K_THREADS>(ss, s_red), (float)width, norm_eps);
			for ( uint32_t c = threadIdx.x; c < width; c += blockDim.x )
				pooled[(uint64_t)row * width + c] = LmFloatToBf16(s_val[c] * sc * LmScalarToFloat(norm[c]));
		}
		if ( threadIdx.x == 0u ) emit_row[row] = target_slot;
		LmDependentRelease();
		return;
	}
	const uint32_t lane = row_lane[row];
	float *slot_state = state + (uint64_t)recurrent_slot[lane] * state_stride_floats;
	if ( (position + 1u) % ratio != 0u )
	{
		if ( save_inline != 0u )
			for ( uint32_t c = threadIdx.x; c < width; c += blockDim.x )
			{
				slot_state[c] = kv[(uint64_t)row * width + c];
				slot_state[width + c] = score[(uint64_t)row * width + c];
			}
		if ( threadIdx.x == 0u ) emit_row[row] = D41K_NO_ROW;
		LmDependentRelease();
		return;
	}
	const uint32_t ordinal = row_ordinal[row];
	const uint32_t previous = parent_rows != 0 ? parent_rows[row] :
		(ordinal > 0u ? sequence_row_indices[sequence_row_begin[lane] + ordinal - 1u] : D41K_NO_ROW);
	const uint32_t in_wave = previous != D41K_NO_ROW ? 1u : 0u;
	float ss = 0.0f;
	for ( uint32_t c = threadIdx.x; c < width; c += blockDim.x )
	{
		const float kv0 = in_wave != 0u ? kv[(uint64_t)previous * width + c] : slot_state[c];
		const float score0 = in_wave != 0u ? score[(uint64_t)previous * width + c] : slot_state[width + c];
		const float kv1 = kv[(uint64_t)row * width + c], score1 = score[(uint64_t)row * width + c];
		const uint16_t bf = LmFloatToBf16(SparkMathSoftmaxPool2(kv0, score0, kv1, score1));
		if ( norm != 0 ) { const float v = LmBf16ToFloat(bf); s_val[c] = v; ss += v * v; }
		else pooled[(uint64_t)row * width + c] = bf;
	}
	if ( norm != 0 )
	{
		const float sc = SparkMathRmsScale(LmBlockSum<D41K_THREADS>(ss, s_red), (float)width, norm_eps);
		for ( uint32_t c = threadIdx.x; c < width; c += blockDim.x )
			pooled[(uint64_t)row * width + c] = LmFloatToBf16(s_val[c] * sc * LmScalarToFloat(norm[c]));
	}
	if ( threadIdx.x == 0u ) emit_row[row] = target_slot / ratio;
	LmDependentRelease();
}

static __global__ void D41CompressorSaveKernel(const float *__restrict__ kv, const float *__restrict__ score, float *__restrict__ state,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ recurrent_slot, uint32_t rows, uint32_t ratio,
	uint32_t state_stride_floats, uint32_t width, uint32_t commit, const uint32_t *__restrict__ lane_count,
	const uint32_t *__restrict__ path_rows = 0)
{
	LmDependentWait();
	LmDependentRelease();
	if ( commit == 0u )
		return;
	uint32_t row = blockIdx.x, lane;
	if ( path_rows != 0 )
	{
		const uint32_t count = lane_count != 0 ? lane_count[0] : 0u;
		if ( blockIdx.x != 0u || count == 0u )
			return;
		row = path_rows[count - 1u];
		if ( (positions[row] + 1u) % ratio == 0u )
			return;
		lane = 0u;
	}
	else
	{
		if ( row >= rows || (positions[row] + 1u) % ratio == 0u )
			return;
		lane = row_lane[row];
		const uint32_t count = lane_count != 0 ? lane_count[lane] : sequence_row_begin[lane + 1u] - sequence_row_begin[lane];
		if ( row_ordinal[row] + 1u != count )
			return;
	}
	float *slot_state = state + (uint64_t)recurrent_slot[lane] * state_stride_floats;
	for ( uint32_t c = threadIdx.x; c < width; c += blockDim.x )
	{
		slot_state[c] = kv[(uint64_t)row * width + c];
		slot_state[width + c] = score[(uint64_t)row * width + c];
	}
}

static __global__ void D41CompressedTreeMapKernel(const uint32_t *__restrict__ fork_positions, const uint32_t *__restrict__ slot_offsets,
	uint32_t *__restrict__ compressed_fork, uint32_t *__restrict__ compressed_offset, uint32_t rows, uint32_t ratio)
{
	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row >= rows )
		return;
	compressed_fork[row] = fork_positions[row] == D41K_NO_ROW ? D41K_NO_ROW : fork_positions[row] / ratio;
	compressed_offset[row] = slot_offsets[row] / ratio;
}

template<class Geometry, uint32_t THREADS>
static __global__ __launch_bounds__(THREADS, 1)
void D41CompressedPathMoveKernel(LmKvShardView view, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ path_rows, const uint32_t *__restrict__ path_count, const uint32_t *__restrict__ slots, uint32_t ratio,
	uint32_t elements)
{
	const uint32_t index = blockIdx.x;
	if ( index >= path_count[0] )
		return;
	const uint32_t row = path_rows[index], position = positions[row];
	if ( (position + 1u) % ratio != 0u )
		return;
	const uint32_t source = slots[row] / ratio, destination = position / ratio;
	if ( source == destination || SparkKvShardOwns(view.shard, destination) == 0u )
		return;
	const uint8_t *from = LmKvShardSlotRequired<Geometry>(view, sequence_of_row[row], source, row, LM_KV_ACCESS_READ);
	uint8_t *to = (uint8_t *)LmKvShardSlotRequired<Geometry>(view, sequence_of_row[row], destination, row, LM_KV_ACCESS_WRITE);
	if ( from == 0 || to == 0 )
		return;
	for ( uint32_t element = threadIdx.x; element < elements; element += THREADS )
		((uint16_t *)to)[element] = ((const uint16_t *)from)[element];
}

static __global__ void D41TreeNgramHashKernel(const uint32_t *__restrict__ token_ids, const int32_t *__restrict__ token_map,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ parent_rows, const uint32_t *__restrict__ recurrent_slot,
	const int32_t *__restrict__ history, uint32_t history_stride, const int64_t *__restrict__ multipliers, const int64_t *__restrict__ primes,
	const int64_t *__restrict__ offsets, int64_t *__restrict__ ids, uint32_t rows, uint32_t orders, uint32_t heads, int32_t pad)
{
	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row >= rows )
		return;
	const uint32_t position = positions[row];
	const int32_t *carried = history + (uint64_t)recurrent_slot[0] * history_stride;
	uint32_t blocked = 0u, current = row, past = 0u;
	int64_t rolling = 0;
	for ( uint32_t shift = 0u; shift <= orders; ++shift )
	{
		int32_t source = -1;
		if ( position >= shift )
		{
			if ( current != D41K_NO_ROW )
			{
				const uint32_t token = token_ids[current];
				source = token_map != 0 ? token_map[token] : (int32_t)token;
				current = parent_rows[current];
			}
			else
				source = carried[past++];
		}
		if ( position < shift || source == -1 )
			blocked = 1u;
		const int64_t value = blocked != 0u ? pad : source;
		if ( shift == 0u )
			rolling = value * multipliers[0];
		else
		{
			rolling ^= value * multipliers[shift];
			for ( uint32_t head = 0u; head < heads; ++head )
			{
				const uint32_t column = (shift - 1u) * heads + head;
				ids[(uint64_t)row * orders * heads + column] = rolling % primes[column] + offsets[column];
			}
		}
	}
}

static __global__ void D41CompressedEmitPositionKernel(const uint32_t *__restrict__ positions, uint32_t *__restrict__ bound_position,
	uint32_t rows, uint32_t ratio)
{
	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	LmDependentWait();
	LmDependentRelease();
	if ( row >= rows )
		return;
	const uint32_t visible = (positions[row] + 1u) / ratio;
	bound_position[row] = visible == 0u ? D41K_NO_ROW : visible - 1u;
}

static __global__ void D41CompressedLengthKernel(const uint32_t *__restrict__ context_length, uint32_t *__restrict__ compressed_length,
	uint32_t sequences, uint32_t ratio)
{
	const uint32_t sequence = blockIdx.x * blockDim.x + threadIdx.x;
	LmDependentWait();
	LmDependentRelease();
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

template<class Geometry>
static __global__ void D41CompressFinishStoreKernel(LmKvShardView view, const uint16_t *__restrict__ latent, const uint16_t *__restrict__ index_key,
	const uint16_t *__restrict__ idx_kn, const uint32_t *__restrict__ positions, int32_t shift, const float *__restrict__ freq,
	const uint32_t *__restrict__ emit_row, const uint32_t *__restrict__ sequence_of_row, uint32_t index_width, float eps)
{
	__shared__ float s_idx[128];
	__shared__ float s_red[D41K_THREADS / 32u];
	__shared__ __align__(16) uint16_t s_lat[D41K_LATENT];
	__shared__ __align__(16) uint16_t s_k[128];
	const uint32_t row = blockIdx.x;
	LmDependentWait();
	const uint32_t pos = emit_row[row];
	if ( pos == D41K_NO_ROW || SparkKvShardOwns(view.shard, pos) == 0u ) { LmDependentRelease(); return; }
	float ss = 0.0f;
	if ( threadIdx.x < index_width )
	{
		const float v = LmBf16ToFloat(index_key[(uint64_t)row * index_width + threadIdx.x]);
		s_idx[threadIdx.x] = v;
		ss = v * v;
	}
	if ( threadIdx.x < D41K_LATENT / 8u )
		((uint4 *)s_lat)[threadIdx.x] = ((const uint4 *)(latent + (uint64_t)row * D41K_LATENT))[threadIdx.x];
	const float sc = SparkMathRmsScale(LmBlockSum<D41K_THREADS>(ss, s_red), (float)index_width, eps);
	if ( threadIdx.x < index_width )
		s_k[threadIdx.x] = LmFloatToBf16(s_idx[threadIdx.x] * sc * LmScalarToFloat(idx_kn[threadIdx.x]));
	__syncthreads();
	if ( threadIdx.x < D41K_ROPE_PAIRS )
	{
		float sine, cosine;
		D41RopeTurn(positions[row], shift, freq[threadIdx.x], 0u, &sine, &cosine);
		const uint32_t ko = index_width - 2u * D41K_ROPE_PAIRS + 2u * threadIdx.x;
		const float kr = LmBf16ToFloat(s_k[ko]), ki = LmBf16ToFloat(s_k[ko + 1u]);
		s_k[ko] = D41RopeReal(kr, ki, sine, cosine);
		s_k[ko + 1u] = D41RopeImaginary(kr, ki, sine, cosine);
		const uint32_t lo = D41K_LATENT - 2u * D41K_ROPE_PAIRS + 2u * threadIdx.x;
		const float lr = LmBf16ToFloat(s_lat[lo]), li = LmBf16ToFloat(s_lat[lo + 1u]);
		s_lat[lo] = D41RopeReal(lr, li, sine, cosine);
		s_lat[lo + 1u] = D41RopeImaginary(lr, li, sine, cosine);
	}
	__syncthreads();
	uint16_t *target = (uint16_t *)LmKvShardSlotRequired<Geometry>(view, sequence_of_row[row], pos, row, LM_KV_ACCESS_WRITE);
	if ( target == 0 ) { LmDependentRelease(); return; }
	if ( threadIdx.x < D41K_LATENT / 8u )
	{
		float v[8], amax = 0.0f;
		LmLoadBf16x8AsFloat(((const uint4 *)s_lat)[threadIdx.x], v);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k ) amax = fmaxf(amax, fabsf(v[k]));
		amax = fmaxf(fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1u)), SPARK_DSV41_FLASH_KV_FP4_AMAX_FLOOR);
		const float qsc = LmE4m3ToFloat(LmFloatToE4m3(amax / LM_E2M1_MAX));
		#pragma unroll
		for ( uint32_t k = 0u; k < 4u; ++k )
		{
			const float2 r = LmE2m1PairToFloat(LmFloatPairToE2m1(fminf(fmaxf(v[2u * k] / qsc, -LM_E2M1_MAX), LM_E2M1_MAX),
				fminf(fmaxf(v[2u * k + 1u] / qsc, -LM_E2M1_MAX), LM_E2M1_MAX)));
			v[2u * k] = r.x * qsc; v[2u * k + 1u] = r.y * qsc;
		}
		((uint4 *)target)[threadIdx.x] = LmPackUint4FloatToBf16(v);
	}
	else if ( threadIdx.x < D41K_LATENT / 8u + 16u )
	{
		const uint32_t q = threadIdx.x - D41K_LATENT / 8u;
		float v[8], amax = 0.0f;
		LmLoadBf16x8AsFloat(((const uint4 *)s_k)[q], v);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k ) amax = fmaxf(amax, fabsf(v[k]));
		amax = fmaxf(amax, __shfl_xor_sync(0x0000ffffu, amax, 1u));
		amax = fmaxf(amax, __shfl_xor_sync(0x0000ffffu, amax, 2u));
		const float qsc = SparkMathFp4Pow2QdqScale(amax), inv = __uint_as_float(0x7f000000u - __float_as_uint(qsc));
		#pragma unroll
		for ( uint32_t k = 0u; k < 4u; ++k )
		{
			const float2 r = LmE2m1PairToFloat(LmFloatPairToE2m1(fminf(fmaxf(v[2u * k] * inv, -LM_E2M1_MAX), LM_E2M1_MAX),
				fminf(fmaxf(v[2u * k + 1u] * inv, -LM_E2M1_MAX), LM_E2M1_MAX)));
			v[2u * k] = r.x * qsc; v[2u * k + 1u] = r.y * qsc;
		}
		((uint4 *)(target + D41K_LATENT))[q] = LmPackUint4FloatToBf16(v);
	}
	LmDependentRelease();
}

static __global__ void D41IndexQueryPrepKernel(uint16_t *__restrict__ query, uint16_t *__restrict__ weight, const uint32_t *__restrict__ positions,
	const float *__restrict__ freq, uint32_t heads, uint32_t dim, float weight_scale, const uint32_t *__restrict__ skip_seq,
	const uint32_t *__restrict__ skip_len, const uint32_t *__restrict__ skip_bound, uint32_t skip_max_context, const float2 *__restrict__ rope_sc = 0)
{
	__shared__ __align__(16) uint16_t s_q[128];
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	LmDependentWait();
	if ( skip_max_context != 0u && skip_seq != 0 && skip_len != 0 )
	{
		uint32_t b = skip_len[skip_seq[row]];
		if ( skip_bound != 0 ) { const uint32_t rp = skip_bound[row]; b = rp == D41K_NO_ROW ? 0u : (rp + 1u < b ? rp + 1u : b); }
		if ( b <= skip_max_context ) { LmDependentRelease(); return; }
	}
	uint16_t *dst = query + ((uint64_t)row * heads + head) * dim;
	if ( threadIdx.x < 16u ) ((uint4 *)s_q)[threadIdx.x] = ((const uint4 *)dst)[threadIdx.x];
	__syncthreads();
	if ( threadIdx.x < D41K_ROPE_PAIRS )
	{
		uint16_t *rp = s_q + dim - 2u * D41K_ROPE_PAIRS + 2u * threadIdx.x;
		if ( rope_sc != 0 ) D41RopePairSc(rp, rope_sc[(uint64_t)row * D41K_ROPE_PAIRS + threadIdx.x], 0u);
		else D41RopePair(rp, positions[row], 0, freq[threadIdx.x], 0u);
	}
	__syncthreads();
	if ( threadIdx.x < 16u )
	{
		float v[8], amax = 0.0f;
		LmLoadBf16x8AsFloat(((const uint4 *)s_q)[threadIdx.x], v);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k ) amax = fmaxf(amax, fabsf(v[k]));
		amax = fmaxf(amax, __shfl_xor_sync(0x0000ffffu, amax, 1u));
		amax = fmaxf(amax, __shfl_xor_sync(0x0000ffffu, amax, 2u));
		const float qsc = SparkMathFp4Pow2QdqScale(amax), inv = __uint_as_float(0x7f000000u - __float_as_uint(qsc));
		#pragma unroll
		for ( uint32_t k = 0u; k < 4u; ++k )
		{
			const float2 r = LmE2m1PairToFloat(LmFloatPairToE2m1(fminf(fmaxf(v[2u * k] * inv, -LM_E2M1_MAX), LM_E2M1_MAX),
				fminf(fmaxf(v[2u * k + 1u] * inv, -LM_E2M1_MAX), LM_E2M1_MAX)));
			v[2u * k] = r.x * qsc; v[2u * k + 1u] = r.y * qsc;
		}
		((uint4 *)dst)[threadIdx.x] = LmPackUint4FloatToBf16(v);
	}
	if ( head == 0u && weight != 0 && threadIdx.x < heads )
		weight[(uint64_t)row * heads + threadIdx.x] = LmFloatToBf16(LmBf16ToFloat(weight[(uint64_t)row * heads + threadIdx.x]) * weight_scale);
	LmDependentRelease();
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
	LmDependentWait();
	LmDependentRelease();
	for ( uint32_t destination = 0u; destination < destinations; ++destination )
		for ( uint32_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < span; offset += gridDim.x * blockDim.x )
			records[(uint64_t)destination * destination_stride + offset] = offset % D41K_RECORD == 0u ? -INFINITY : 0.0f;
}

static inline void D41InitRecords(float *records, uint64_t destination_stride, uint32_t rows, uint32_t heads_per_destination, uint32_t destinations,
	SparkGpuStream stream)
{
	const uint32_t span = rows * heads_per_destination * D41K_RECORD, blocks = (span + D41K_THREADS - 1u) / D41K_THREADS;
	LM_LAUNCH_DEPENDENT((D41InitRecordsKernel), blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u, D41K_THREADS, 0, stream, records, destination_stride, span, destinations);
}

#define D41K_WINDOW_KEYS 128u
#define D41K_WINDOW_BATCH 16u

static __device__ __forceinline__ void D41WindowTreeRows(const uint32_t *__restrict__ parent_rows, uint32_t row, uint32_t *tree_rows)
{
	if ( parent_rows == 0 )
		return;
	if ( threadIdx.x == 0u )
	{
		uint32_t current = row;
		for ( uint32_t back = 0u; back < D41K_WINDOW_KEYS && current != D41K_NO_ROW; ++back )
		{
			tree_rows[back] = current;
			current = parent_rows[current];
		}
	}
	__syncthreads();
}

static __device__ __forceinline__ uint32_t D41WindowWaveRow(const uint32_t *__restrict__ lane_rows, const uint32_t *__restrict__ tree_rows,
	const uint32_t *__restrict__ parent_rows, uint32_t position, uint32_t wave_first, uint32_t at)
{
	return parent_rows != 0 ? tree_rows[position - at] : lane_rows[at - wave_first];
}

static __global__ void D41WindowFoldKernel(float *__restrict__ records, uint64_t destination_stride, uint32_t heads_per_destination,
	const uint16_t *__restrict__ query, uint64_t query_rank_stride, const uint16_t *__restrict__ wave_kv,
	const uint16_t *__restrict__ ring, const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane,
	const uint32_t *__restrict__ row_ordinal, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices,
	const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows, uint32_t window, uint32_t degree, uint32_t rank,
	float qk_scale, uint32_t record_heads, uint32_t record_copies, const uint32_t *__restrict__ parent_rows = 0)
{
	__shared__ float warp_sums[D41K_FOLD_KEYS][D41K_THREADS / 32u];
	__shared__ float key_sums[D41K_FOLD_KEYS];
	__shared__ uint32_t tree_rows[D41K_WINDOW_KEYS];
	const uint32_t row = blockIdx.x, head = blockIdx.y, lane_index = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	LmDependentWait();
	LmDependentRelease();
	const uint32_t position = positions[row], lane = row_lane[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const uint32_t wave_first = parent_rows != 0 ? positions[lane_rows[0]] : position - row_ordinal[row];
	const uint32_t first = position + 1u > window ? position + 1u - window : 0u;
	const uint32_t destination = head / heads_per_destination, local = head % heads_per_destination;
	const uint16_t *q = query + (uint64_t)destination * query_rank_stride + ((uint64_t)row * heads_per_destination + local) * D41K_LATENT;
	const uint64_t record_offset = ((uint64_t)row * record_heads + head % record_heads) * D41K_RECORD;
	float *record = records + (uint64_t)(head / record_heads) * record_copies * destination_stride + record_offset;
	float running_max = record[0], running_sum = record[1];
	float accumulator[D41K_LATENT / D41K_THREADS], query_value[D41K_LATENT / D41K_THREADS];
	D41WindowTreeRows(parent_rows, row, tree_rows);
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
			const uint16_t *k = at >= wave_first ? wave_kv + (uint64_t)D41WindowWaveRow(lane_rows, tree_rows, parent_rows, position, wave_first, at) * D41K_LATENT :
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

static __global__ __launch_bounds__(D41K_THREADS) void D41WindowFoldWholeKernel(float *__restrict__ records, uint32_t heads, const uint16_t *__restrict__ query,
	const uint16_t *__restrict__ wave_kv, const uint16_t *__restrict__ ring, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal, const uint32_t *__restrict__ sequence_row_begin,
	const uint32_t *__restrict__ sequence_row_indices, const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t window,
	float qk_scale)
{
	static_assert(D41K_WINDOW_KEYS % 32u == 0u && D41K_WINDOW_KEYS <= D41K_THREADS && D41K_LATENT == 2u * D41K_THREADS, "window keys fill whole warps");
	__shared__ const uint16_t *kptr[D41K_WINDOW_KEYS];
	__shared__ float warp_sums[D41K_WINDOW_KEYS][D41K_THREADS / 32u];
	__shared__ float warp_max[D41K_WINDOW_KEYS / 32u];
	__shared__ float keep[D41K_WINDOW_KEYS];
	__shared__ float added[D41K_WINDOW_KEYS];
	__shared__ float folded[2];
	const uint32_t row = blockIdx.x, head = blockIdx.y, lane_index = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	LmDependentWait();
	const uint32_t position = positions[row], lane = row_lane[row];
	const uint32_t wave_first = position - row_ordinal[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const uint32_t first = position + 1u > window ? position + 1u - window : 0u, count = position + 1u - first;
	const uint16_t *q = query + ((uint64_t)row * heads + head) * D41K_LATENT;
	const uint16_t *slot_ring = ring + (uint64_t)recurrent_slot[lane] * ring_slot_stride;
	float *record = records + ((uint64_t)row * heads + head) * D41K_RECORD;
	const float carried_max = record[0], carried_sum = record[1];
	float acc0 = record[2u + threadIdx.x], acc1 = record[2u + threadIdx.x + D41K_THREADS];
	const float q0 = LmBf16ToFloat(__ldg(q + threadIdx.x)), q1 = LmBf16ToFloat(__ldg(q + threadIdx.x + D41K_THREADS));
	if ( threadIdx.x < count )
		kptr[threadIdx.x] = D41WindowKey(wave_kv, slot_ring, lane_rows, wave_first, window, first + threadIdx.x);
	__syncthreads();
	uint32_t j = 0u;
	for ( ; j + 3u < count; j += 4u )
	{
		const uint16_t *k0 = kptr[j], *k1 = kptr[j + 1u], *k2 = kptr[j + 2u], *k3 = kptr[j + 3u];
		const uint16_t a0 = __ldg(k0 + threadIdx.x), b0 = __ldg(k0 + threadIdx.x + D41K_THREADS);
		const uint16_t a1 = __ldg(k1 + threadIdx.x), b1 = __ldg(k1 + threadIdx.x + D41K_THREADS);
		const uint16_t a2 = __ldg(k2 + threadIdx.x), b2 = __ldg(k2 + threadIdx.x + D41K_THREADS);
		const uint16_t a3 = __ldg(k3 + threadIdx.x), b3 = __ldg(k3 + threadIdx.x + D41K_THREADS);
		float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
		p0 += q0 * LmBf16ToFloat(a0); p0 += q1 * LmBf16ToFloat(b0);
		p1 += q0 * LmBf16ToFloat(a1); p1 += q1 * LmBf16ToFloat(b1);
		p2 += q0 * LmBf16ToFloat(a2); p2 += q1 * LmBf16ToFloat(b2);
		p3 += q0 * LmBf16ToFloat(a3); p3 += q1 * LmBf16ToFloat(b3);
		#pragma unroll
		for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
		{
			p0 += __shfl_down_sync(0xffffffffu, p0, offset);
			p1 += __shfl_down_sync(0xffffffffu, p1, offset);
			p2 += __shfl_down_sync(0xffffffffu, p2, offset);
			p3 += __shfl_down_sync(0xffffffffu, p3, offset);
		}
		if ( lane_index == 0u )
		{
			warp_sums[j][warp] = p0;
			warp_sums[j + 1u][warp] = p1;
			warp_sums[j + 2u][warp] = p2;
			warp_sums[j + 3u][warp] = p3;
		}
	}
	for ( ; j < count; ++j )
	{
		const uint16_t *k = kptr[j];
		float partial = 0.0f;
		partial += q0 * LmBf16ToFloat(__ldg(k + threadIdx.x));
		partial += q1 * LmBf16ToFloat(__ldg(k + threadIdx.x + D41K_THREADS));
		#pragma unroll
		for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
			partial += __shfl_down_sync(0xffffffffu, partial, offset);
		if ( lane_index == 0u )
			warp_sums[j][warp] = partial;
	}
	__syncthreads();
	float score = -INFINITY, prev_in_warp = -INFINITY;
	if ( threadIdx.x < D41K_WINDOW_KEYS )
	{
		if ( threadIdx.x < count )
		{
			float total = 0.0f;
			#pragma unroll
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
		for ( uint32_t i = 0u; i < count; ++i )
			running_sum = running_sum * keep[i] + added[i];
		folded[1] = running_sum;
		if ( count == 0u )
			folded[0] = carried_max;
	}
	j = 0u;
	for ( ; j + 3u < count; j += 4u )
	{
		const uint16_t *k0 = kptr[j], *k1 = kptr[j + 1u], *k2 = kptr[j + 2u], *k3 = kptr[j + 3u];
		const uint16_t a0 = __ldg(k0 + threadIdx.x), b0 = __ldg(k0 + threadIdx.x + D41K_THREADS);
		const uint16_t a1 = __ldg(k1 + threadIdx.x), b1 = __ldg(k1 + threadIdx.x + D41K_THREADS);
		const uint16_t a2 = __ldg(k2 + threadIdx.x), b2 = __ldg(k2 + threadIdx.x + D41K_THREADS);
		const uint16_t a3 = __ldg(k3 + threadIdx.x), b3 = __ldg(k3 + threadIdx.x + D41K_THREADS);
		const float kp0 = keep[j], wt0 = SparkMathAttentionValueWeight(added[j]);
		const float kp1 = keep[j + 1u], wt1 = SparkMathAttentionValueWeight(added[j + 1u]);
		const float kp2 = keep[j + 2u], wt2 = SparkMathAttentionValueWeight(added[j + 2u]);
		const float kp3 = keep[j + 3u], wt3 = SparkMathAttentionValueWeight(added[j + 3u]);
		acc0 = acc0 * kp0 + wt0 * LmBf16ToFloat(a0);
		acc1 = acc1 * kp0 + wt0 * LmBf16ToFloat(b0);
		acc0 = acc0 * kp1 + wt1 * LmBf16ToFloat(a1);
		acc1 = acc1 * kp1 + wt1 * LmBf16ToFloat(b1);
		acc0 = acc0 * kp2 + wt2 * LmBf16ToFloat(a2);
		acc1 = acc1 * kp2 + wt2 * LmBf16ToFloat(b2);
		acc0 = acc0 * kp3 + wt3 * LmBf16ToFloat(a3);
		acc1 = acc1 * kp3 + wt3 * LmBf16ToFloat(b3);
	}
	for ( ; j < count; ++j )
	{
		const uint16_t *k = kptr[j];
		const float kept = keep[j], weight = SparkMathAttentionValueWeight(added[j]);
		acc0 = acc0 * kept + weight * LmBf16ToFloat(__ldg(k + threadIdx.x));
		acc1 = acc1 * kept + weight * LmBf16ToFloat(__ldg(k + threadIdx.x + D41K_THREADS));
	}
	__syncthreads();
	record[2u + threadIdx.x] = acc0;
	record[2u + threadIdx.x + D41K_THREADS] = acc1;
	if ( threadIdx.x == 0u )
	{
		record[0] = folded[0];
		record[1] = folded[1];
	}
	LmDependentRelease();
}

template<uint32_t BLOCK_HEADS = 4u>
static __global__ __launch_bounds__(D41K_THREADS) void D41WindowFoldWholeTiledKernel(float *__restrict__ records, uint32_t heads,
	const uint16_t *__restrict__ query, const uint16_t *__restrict__ wave_kv, const uint16_t *__restrict__ ring, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal, const uint32_t *__restrict__ sequence_row_begin,
	const uint32_t *__restrict__ sequence_row_indices, const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t window,
	float qk_scale)
{
	static_assert((BLOCK_HEADS == 2u || BLOCK_HEADS == 4u) && D41K_LATENT == 2u * D41K_THREADS && D41K_WINDOW_KEYS <= D41K_THREADS, "tiled window fold processes two or four heads per block");
	constexpr uint32_t TILE = 16u;
	__shared__ const uint16_t *kptr[D41K_WINDOW_KEYS];
	__shared__ __align__(16) uint16_t s_k[TILE][D41K_LATENT];
	__shared__ float warp_sums[BLOCK_HEADS][TILE][D41K_THREADS / 32u];
	__shared__ float s_keep[BLOCK_HEADS][TILE];
	__shared__ float s_wt[BLOCK_HEADS][TILE];
	__shared__ float s_max[BLOCK_HEADS];
	__shared__ float s_sum[BLOCK_HEADS];
	const uint32_t row = blockIdx.x, hbase = blockIdx.y * BLOCK_HEADS, lane_index = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	LmDependentWait();
	const uint32_t position = positions[row], lane = row_lane[row];
	const uint32_t wave_first = position - row_ordinal[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const uint32_t first = position + 1u > window ? position + 1u - window : 0u, count = position + 1u - first;
	const uint16_t *slot_ring = ring + (uint64_t)recurrent_slot[lane] * ring_slot_stride;
	float q0[BLOCK_HEADS], q1[BLOCK_HEADS], acc0[BLOCK_HEADS], acc1[BLOCK_HEADS];
	#pragma unroll
	for ( uint32_t h = 0u; h < BLOCK_HEADS; ++h )
	{
		const uint16_t *qp = query + ((uint64_t)row * heads + hbase + h) * D41K_LATENT;
		q0[h] = LmBf16ToFloat(__ldg(qp + threadIdx.x));
		q1[h] = LmBf16ToFloat(__ldg(qp + threadIdx.x + D41K_THREADS));
		const float *rec = records + ((uint64_t)row * heads + hbase + h) * D41K_RECORD;
		acc0[h] = __ldg(rec + 2u + threadIdx.x);
		acc1[h] = __ldg(rec + 2u + threadIdx.x + D41K_THREADS);
		if ( threadIdx.x == h ) { s_max[h] = __ldg(rec); s_sum[h] = __ldg(rec + 1u); }
	}
	if ( threadIdx.x < count )
		kptr[threadIdx.x] = D41WindowKey(wave_kv, slot_ring, lane_rows, wave_first, window, first + threadIdx.x);
	__syncthreads();
	for ( uint32_t j = 0u; j < count; j += TILE )
	{
		const uint32_t n = count - j < TILE ? count - j : TILE;
		for ( uint32_t idx = threadIdx.x; idx < n * (D41K_LATENT / 8u); idx += D41K_THREADS )
		{
			const uint32_t t = idx >> 6u, qv = idx & 63u;
			((uint4 *)s_k[t])[qv] = __ldg((const uint4 *)kptr[j + t] + qv);
		}
		__syncthreads();
		for ( uint32_t t = 0u; t < n; ++t )
		{
			const float a = LmBf16ToFloat(s_k[t][threadIdx.x]), b = LmBf16ToFloat(s_k[t][threadIdx.x + D41K_THREADS]);
			#pragma unroll
			for ( uint32_t h = 0u; h < BLOCK_HEADS; ++h )
			{
				float p = q0[h] * a + q1[h] * b;
				#pragma unroll
				for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
					p += __shfl_down_sync(0xffffffffu, p, offset);
				if ( lane_index == 0u ) warp_sums[h][t][warp] = p;
			}
		}
		__syncthreads();
		if ( threadIdx.x < BLOCK_HEADS )
		{
			const uint32_t h = threadIdx.x;
			float rmax = s_max[h], rsum = s_sum[h];
			for ( uint32_t t = 0u; t < n; ++t )
			{
				float tot = 0.0f;
				#pragma unroll
				for ( uint32_t w = 0u; w < D41K_THREADS / 32u; ++w ) tot += warp_sums[h][t][w];
				const float sc = tot * qk_scale, top = fmaxf(rmax, sc);
				const float kp = rmax == -INFINITY ? 0.0f : __expf(rmax - top), ad = __expf(sc - top);
				rsum = rsum * kp + ad; rmax = top;
				s_keep[h][t] = kp; s_wt[h][t] = SparkMathAttentionValueWeight(ad);
			}
			s_max[h] = rmax; s_sum[h] = rsum;
		}
		__syncthreads();
		for ( uint32_t t = 0u; t < n; ++t )
		{
			const float va = LmBf16ToFloat(s_k[t][threadIdx.x]), vb = LmBf16ToFloat(s_k[t][threadIdx.x + D41K_THREADS]);
			#pragma unroll
			for ( uint32_t h = 0u; h < BLOCK_HEADS; ++h )
			{
				const float kp = s_keep[h][t], wt = s_wt[h][t];
				acc0[h] = acc0[h] * kp + wt * va;
				acc1[h] = acc1[h] * kp + wt * vb;
			}
		}
		if ( j + TILE < count ) __syncthreads();
	}
	#pragma unroll
	for ( uint32_t h = 0u; h < BLOCK_HEADS; ++h )
	{
		float *rec = records + ((uint64_t)row * heads + hbase + h) * D41K_RECORD;
		rec[2u + threadIdx.x] = acc0[h];
		rec[2u + threadIdx.x + D41K_THREADS] = acc1[h];
		if ( threadIdx.x == h ) { rec[0] = s_max[h]; rec[1] = s_sum[h]; }
	}
	LmDependentRelease();
}

static __global__ void __cluster_dims__(D41K_WINDOW_PARTS, 1, 1) __launch_bounds__(D41K_THREADS) D41WindowFoldClusterKernel(float *__restrict__ records,
	uint32_t heads, const uint16_t *__restrict__ query, const uint16_t *__restrict__ wave_kv, const uint16_t *__restrict__ ring,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices, const uint32_t *__restrict__ recurrent_slot,
	uint64_t ring_slot_stride, uint32_t window, float qk_scale)
{
	static_assert(D41K_WINDOW_KEYS % D41K_WINDOW_PARTS == 0u && D41K_LATENT % D41K_WINDOW_PARTS == 0u && D41K_WINDOW_PART_SLICE <= D41K_THREADS &&
		D41K_WINDOW_KEYS <= D41K_THREADS && D41K_WINDOW_KEYS % 32u == 0u && D41K_LATENT == 2u * D41K_THREADS, "each part scores a whole run of keys and folds a whole slice of the latent");
	__shared__ const uint16_t *kptr[D41K_WINDOW_KEYS];
	__shared__ float warp_sums[D41K_WINDOW_PART_KEYS][D41K_THREADS / 32u];
	__shared__ float part_scores[D41K_WINDOW_PART_KEYS];
	__shared__ float warp_max[D41K_WINDOW_KEYS / 32u];
	__shared__ float keep[D41K_WINDOW_KEYS];
	__shared__ float added[D41K_WINDOW_KEYS];
	__shared__ float folded[2];
	cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
	const uint32_t part = blockIdx.x, head = blockIdx.y, row = blockIdx.z, lane_index = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	LmDependentWait();
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
	const float q0 = LmBf16ToFloat(__ldg(q + threadIdx.x)), q1 = LmBf16ToFloat(__ldg(q + threadIdx.x + D41K_THREADS));
	if ( threadIdx.x < count )
		kptr[threadIdx.x] = D41WindowKey(wave_kv, slot_ring, lane_rows, wave_first, window, first + threadIdx.x);
	__syncthreads();
	uint32_t j = 0u;
	for ( ; j + 3u < key_count; j += 4u )
	{
		const uint16_t *k0 = kptr[key_first + j], *k1 = kptr[key_first + j + 1u];
		const uint16_t *k2 = kptr[key_first + j + 2u], *k3 = kptr[key_first + j + 3u];
		const uint16_t a0 = __ldg(k0 + threadIdx.x), b0 = __ldg(k0 + threadIdx.x + D41K_THREADS);
		const uint16_t a1 = __ldg(k1 + threadIdx.x), b1 = __ldg(k1 + threadIdx.x + D41K_THREADS);
		const uint16_t a2 = __ldg(k2 + threadIdx.x), b2 = __ldg(k2 + threadIdx.x + D41K_THREADS);
		const uint16_t a3 = __ldg(k3 + threadIdx.x), b3 = __ldg(k3 + threadIdx.x + D41K_THREADS);
		float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
		p0 += q0 * LmBf16ToFloat(a0); p0 += q1 * LmBf16ToFloat(b0);
		p1 += q0 * LmBf16ToFloat(a1); p1 += q1 * LmBf16ToFloat(b1);
		p2 += q0 * LmBf16ToFloat(a2); p2 += q1 * LmBf16ToFloat(b2);
		p3 += q0 * LmBf16ToFloat(a3); p3 += q1 * LmBf16ToFloat(b3);
		#pragma unroll
		for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
		{
			p0 += __shfl_down_sync(0xffffffffu, p0, offset);
			p1 += __shfl_down_sync(0xffffffffu, p1, offset);
			p2 += __shfl_down_sync(0xffffffffu, p2, offset);
			p3 += __shfl_down_sync(0xffffffffu, p3, offset);
		}
		if ( lane_index == 0u )
		{
			warp_sums[j][warp] = p0;
			warp_sums[j + 1u][warp] = p1;
			warp_sums[j + 2u][warp] = p2;
			warp_sums[j + 3u][warp] = p3;
		}
	}
	for ( ; j < key_count; ++j )
	{
		const uint16_t *k = kptr[key_first + j];
		float partial = 0.0f;
		partial += q0 * LmBf16ToFloat(__ldg(k + threadIdx.x));
		partial += q1 * LmBf16ToFloat(__ldg(k + threadIdx.x + D41K_THREADS));
		#pragma unroll
		for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
			partial += __shfl_down_sync(0xffffffffu, partial, offset);
		if ( lane_index == 0u )
			warp_sums[j][warp] = partial;
	}
	__syncthreads();
	if ( threadIdx.x < key_count )
	{
		float total = 0.0f;
		#pragma unroll
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
		for ( uint32_t i = 0u; i < count; ++i )
			running_sum = running_sum * keep[i] + added[i];
		record[0] = count == 0u ? carried_max : folded[0];
		record[1] = running_sum;
	}
	if ( threadIdx.x < D41K_WINDOW_PART_SLICE )
	{
		const uint32_t e = part * D41K_WINDOW_PART_SLICE + threadIdx.x;
		float accumulator = record[2u + e];
		uint32_t i = 0u;
		for ( ; i + 3u < count; i += 4u )
		{
			const uint16_t s0 = __ldg(kptr[i] + e);
			const uint16_t s1 = __ldg(kptr[i + 1u] + e);
			const uint16_t s2 = __ldg(kptr[i + 2u] + e);
			const uint16_t s3 = __ldg(kptr[i + 3u] + e);
			const float kp0 = keep[i], wt0 = SparkMathAttentionValueWeight(added[i]);
			const float kp1 = keep[i + 1u], wt1 = SparkMathAttentionValueWeight(added[i + 1u]);
			const float kp2 = keep[i + 2u], wt2 = SparkMathAttentionValueWeight(added[i + 2u]);
			const float kp3 = keep[i + 3u], wt3 = SparkMathAttentionValueWeight(added[i + 3u]);
			accumulator = accumulator * kp0 + wt0 * LmBf16ToFloat(s0);
			accumulator = accumulator * kp1 + wt1 * LmBf16ToFloat(s1);
			accumulator = accumulator * kp2 + wt2 * LmBf16ToFloat(s2);
			accumulator = accumulator * kp3 + wt3 * LmBf16ToFloat(s3);
		}
		for ( ; i < count; ++i )
		{
			const float kept = keep[i], weight = SparkMathAttentionValueWeight(added[i]);
			accumulator = accumulator * kept + weight * LmBf16ToFloat(__ldg(kptr[i] + e));
		}
		record[2u + e] = accumulator;
	}
	LmDependentRelease();
}

static __device__ __forceinline__ void D41WindowRingStoreRow(const uint16_t *__restrict__ wave_kv, uint16_t *__restrict__ ring,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows,
	uint32_t window, uint32_t degree, uint32_t rank, uint32_t rows, uint32_t commit, const uint32_t *__restrict__ lane_count, uint32_t row,
	const uint32_t *__restrict__ path_rows = 0)
{
	if ( commit == 0u || row >= rows )
		return;
	uint32_t lane, ordinal, count;
	if ( path_rows != 0 )
	{
		count = lane_count != 0 ? lane_count[0] : 0u;
		ordinal = row;
		if ( ordinal >= count || ordinal + window < count )
			return;
		row = path_rows[ordinal];
		lane = 0u;
	}
	else
	{
		lane = row_lane[row];
		count = lane_count != 0 ? lane_count[lane] : sequence_row_begin[lane + 1u] - sequence_row_begin[lane];
		ordinal = row_ordinal[row];
		if ( ordinal >= count || ordinal + window < count )
			return;
	}
	if ( positions[row] % degree != rank )
		return;
	uint16_t *target = ring + (uint64_t)recurrent_slot[lane] * ring_slot_stride +
		(uint64_t)((positions[row] / degree) % ring_rows) * D41K_LATENT;
	const uint16_t *src = wave_kv + (uint64_t)row * D41K_LATENT;
	if ( LmIsAligned16(target) && LmIsAligned16(src) )
	{
		for ( uint32_t q = threadIdx.x; q < D41K_LATENT / 8u; q += blockDim.x )
			((uint4 *)target)[q] = ((const uint4 *)src)[q];
		return;
	}
	for ( uint32_t c = threadIdx.x; c < D41K_LATENT; c += blockDim.x )
		target[c] = src[c];
}

static __global__ void D41WindowRingStoreKernel(const uint16_t *__restrict__ wave_kv, uint16_t *__restrict__ ring,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows,
	uint32_t window, uint32_t degree, uint32_t rank, uint32_t rows, uint32_t commit, const uint32_t *__restrict__ lane_count,
	const uint32_t *__restrict__ path_rows = 0)
{
	LmDependentWait();
	LmDependentRelease();
	D41WindowRingStoreRow(wave_kv, ring, positions, row_lane, row_ordinal, sequence_row_begin, recurrent_slot, ring_slot_stride, ring_rows, window, degree,
		rank, rows, commit, lane_count, blockIdx.x, path_rows);
}

#define D41K_GRID_ITEMS 32768u

static __device__ __forceinline__ uint32_t D41DequantE4m3x2(uint32_t bytes, float scale)
{
	return LmPackBf16Pair(LmE4m3ToFloat((uint8_t)(bytes & 0xffu)) * scale, LmE4m3ToFloat((uint8_t)((bytes >> 8u) & 0xffu)) * scale);
}

static __device__ __forceinline__ uint4 D41DequantE4m3x8(uint2 codes, float scale)
{
	return make_uint4(D41DequantE4m3x2(codes.x, scale), D41DequantE4m3x2(codes.x >> 16u, scale),
		D41DequantE4m3x2(codes.y, scale), D41DequantE4m3x2(codes.y >> 16u, scale));
}

static __device__ __forceinline__ void D41DequantRowE4m3(const uint8_t *__restrict__ src, const uint8_t *__restrict__ scales,
	uint16_t *__restrict__ dst, uint32_t width)
{
	if ( (width & 7u) == 0u && LmIsAligned8(src) && LmIsAligned16(dst) )
	{
		for ( uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < (width >> 3u); q += gridDim.x * blockDim.x )
			((uint4 *)dst)[q] = D41DequantE4m3x8(((const uint2 *)src)[q], __uint_as_float((uint32_t)scales[(q * 8u) / SPARK_DSV41_FLASH_FP8_BLOCK] << 23u));
		return;
	}
	for ( uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += gridDim.x * blockDim.x )
		dst[i] = LmFloatToBf16(LmE4m3ToFloat(src[i]) * __uint_as_float((uint32_t)scales[i / SPARK_DSV41_FLASH_FP8_BLOCK] << 23u));
}

static __device__ __forceinline__ uint8_t D41QuantFp8Block(uint16_t value, uint8_t *exponent)
{
	const float activation = LmBf16ToFloat(value);
	float amax = fabsf(activation);
	for ( uint32_t offset = SPARK_DSV41_FLASH_FP8_BLOCK / 2u; offset != 0u; offset >>= 1u )
		amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
	const float scale = SparkMathFp8QdqScale(fmaxf(0.0f, amax));
	*exponent = (uint8_t)(__float_as_uint(scale) >> 23u);
	return LmFloatToE4m3(fminf(fmaxf(activation / scale, -LM_E4M3_MAX), LM_E4M3_MAX));
}

static __device__ __forceinline__ uint4 D41Fp8Qdq8(const float vals[8])
{
	float amax = 0.0f, q[8];
	#pragma unroll
	for ( uint32_t k = 0u; k < 8u; ++k ) amax = fmaxf(amax, fabsf(vals[k]));
	amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1u));
	amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2u));
	const float scale = SparkMathFp8QdqScale(fmaxf(0.0f, amax));
	#pragma unroll
	for ( uint32_t k = 0u; k < 8u; ++k )
		q[k] = LmE4m3ToFloat(LmFloatToE4m3(fminf(fmaxf(vals[k] / scale, -LM_E4M3_MAX), LM_E4M3_MAX))) * scale;
	return LmPackUint4FloatToBf16(q);
}

template<uint32_t THREADS>
static __device__ __forceinline__ float D41RmsSharedScale(const float *values, float *reduction, uint32_t dimension, float epsilon)
{
	float total = 0.0f;
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < dimension; index += THREADS ) { const float val = values[index]; total += val * val; }
	return SparkMathRmsScale(LmBlockSum<THREADS>(total, reduction), (float)dimension, epsilon);
}

static __device__ __forceinline__ void D41Fp8Quant8(const float vals[8], uint2 *codes, uint8_t *exponent)
{
	const unsigned mask = __activemask();
	float amax = 0.0f;
	#pragma unroll
	for ( uint32_t k = 0u; k < 8u; ++k ) amax = fmaxf(amax, fabsf(vals[k]));
	amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 1u));
	amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 2u));
	const float scale = SparkMathFp8QdqScale(fmaxf(0.0f, amax));
	const uint32_t sbits = __float_as_uint(scale);
	const float inv = __uint_as_float(0x7f000000u - sbits);
	const uint32_t p0 = LmFloatPairToE4m3(vals[0] * inv, vals[1] * inv), p1 = LmFloatPairToE4m3(vals[2] * inv, vals[3] * inv);
	const uint32_t p2 = LmFloatPairToE4m3(vals[4] * inv, vals[5] * inv), p3 = LmFloatPairToE4m3(vals[6] * inv, vals[7] * inv);
	*codes = make_uint2(p0 | (p1 << 16u), p2 | (p3 << 16u));
	*exponent = (uint8_t)(sbits >> 23u);
}

static __device__ __forceinline__ uint4 D41Swiglu8(uint4 g_raw, uint4 u_raw, float weight, float limit, float out[8])
{
	float g[8], u[8];
	LmLoadBf16x8AsFloat(g_raw, g);
	LmLoadBf16x8AsFloat(u_raw, u);
	#pragma unroll
	for ( uint32_t k = 0u; k < 8u; ++k ) out[k] = SparkMathGateLimitedGlu(g[k], u[k], limit) * weight;
	const uint4 packed = LmPackUint4FloatToBf16(out);
	LmLoadBf16x8AsFloat(packed, out);
	return packed;
}

static __device__ __forceinline__ void D41MergeSinkHead(const float *__restrict__ partials, uint64_t source_stride, uint32_t sources, uint32_t heads,
	const float *__restrict__ sink, uint16_t *output, uint32_t row, uint32_t head, float *scales, float *denominator_shared)
{
	const uint64_t base = ((uint64_t)row * heads + head) * D41K_RECORD;
	float max_local = -1.0e30f, den_local = 0.0f;
	if ( threadIdx.x < 32u )
	{
		if ( threadIdx.x < sources ) { max_local = partials[(uint64_t)threadIdx.x * source_stride + base]; den_local = partials[(uint64_t)threadIdx.x * source_stride + base + 1u]; }
		float top = max_local, scale_local = 0.0f, denominator = 0.0f;
		for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u ) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset));
		if ( threadIdx.x < sources ) { scale_local = max_local == -INFINITY ? 0.0f : __expf(max_local - top); scales[threadIdx.x] = scale_local; }
		for ( uint32_t s = 0u; s < sources; ++s ) denominator = fmaf(__shfl_sync(0xffffffffu, den_local, s), __shfl_sync(0xffffffffu, scale_local, s), denominator);
		if ( threadIdx.x == 0u ) *denominator_shared = denominator + __expf(sink[head] - top);
	}
	__syncthreads();
	const float denom = *denominator_shared;
	const float *first_src = partials + base + 2u;
	if ( LmIsAligned16(first_src) && LmIsAligned16(output) && (sources == 1u || (source_stride & 3u) == 0u) )
	{
		const uint64_t stride_f4 = source_stride >> 2u;
		for ( uint32_t v = threadIdx.x; v < D41K_LATENT / 4u; v += blockDim.x )
		{
			float m[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
			for ( uint32_t s = 0u; s < sources; ++s )
			{
				const float sc = scales[s];
				const float4 p = ((const float4 *)first_src)[(uint64_t)s * stride_f4 + v];
				m[0] = fmaf(p.x, sc, m[0]); m[1] = fmaf(p.y, sc, m[1]); m[2] = fmaf(p.z, sc, m[2]); m[3] = fmaf(p.w, sc, m[3]);
			}
			((uint2 *)output)[v] = make_uint2(LmPackBf16Pair(m[0] / denom, m[1] / denom), LmPackBf16Pair(m[2] / denom, m[3] / denom));
		}
		return;
	}
	if ( LmIsAligned8(first_src) && LmIsAligned8(output) && (sources == 1u || (source_stride & 1u) == 0u) )
	{
		const uint64_t stride_f2 = source_stride >> 1u;
		for ( uint32_t v = threadIdx.x; v < D41K_LATENT / 2u; v += blockDim.x )
		{
			float m0 = 0.0f, m1 = 0.0f;
			for ( uint32_t s = 0u; s < sources; ++s )
			{
				const float sc = scales[s];
				const float2 p = ((const float2 *)first_src)[(uint64_t)s * stride_f2 + v];
				m0 = fmaf(p.x, sc, m0); m1 = fmaf(p.y, sc, m1);
			}
			((uint32_t *)output)[v] = LmPackBf16Pair(m0 / denom, m1 / denom);
		}
		return;
	}
	for ( uint32_t e = threadIdx.x; e < D41K_LATENT; e += blockDim.x )
	{
		float merged = 0.0f;
		for ( uint32_t s = 0u; s < sources; ++s ) merged = fmaf(first_src[(uint64_t)s * source_stride + e], scales[s], merged);
		output[e] = LmFloatToBf16(merged / denom);
	}
}

static __global__ void D41MergeSinkKernel(const float *__restrict__ partials, uint64_t source_stride, uint32_t sources, uint32_t heads,
	const float *__restrict__ sink, uint16_t *__restrict__ output)
{
	__shared__ float scales[16];
	__shared__ float denominator_shared;
	LmDependentWait();
	LmDependentRelease();
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

template<uint32_t THREADS>
static __global__ __launch_bounds__(THREADS) void D41SpreadOutputKernel(const float *__restrict__ partials, uint64_t source_stride, uint32_t heads,
	const float *__restrict__ sink, const uint32_t *__restrict__ positions, const float *__restrict__ inverse_frequency, uint16_t *__restrict__ output,
	uint32_t qdq)
{
	__shared__ float shared[32u + D41K_LATENT / 2u];
	uint16_t *merged = (uint16_t *)(shared + 32u);
	LmDependentWait();
	D41MergeSinkHead(partials, source_stride, 1u, heads, sink, merged, blockIdx.x, blockIdx.y, shared, shared + 16u);
	__syncthreads();
	if ( threadIdx.x < D41K_ROPE_PAIRS )
		D41RopePair(merged + D41K_LATENT - 2u * D41K_ROPE_PAIRS + 2u * threadIdx.x, positions[blockIdx.x], 0, inverse_frequency[threadIdx.x], 1u);
	__syncthreads();
	uint16_t *target = output + ((uint64_t)blockIdx.x * heads + blockIdx.y) * D41K_LATENT;
	for ( uint32_t element = threadIdx.x; element < D41K_LATENT; element += THREADS )
		target[element] = qdq != 0u ? SparkDsv41FlashFp8QdqLane(LmBf16ToFloat(merged[element])) : merged[element];
	LmDependentRelease();
}

template<uint32_t THREADS>
static __global__ __launch_bounds__(THREADS) void D41SpreadOutputRingKernel(const float *__restrict__ partials, uint64_t source_stride, uint32_t heads,
	const float *__restrict__ sink, const uint32_t *__restrict__ positions, const float *__restrict__ inverse_frequency, uint16_t *__restrict__ output,
	uint32_t qdq, const uint16_t *__restrict__ wave_kv, uint16_t *__restrict__ ring, const uint32_t *__restrict__ row_lane,
	const uint32_t *__restrict__ row_ordinal, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ recurrent_slot,
	uint64_t ring_slot_stride, uint32_t ring_rows, uint32_t window, uint32_t rows, uint32_t commit)
{
	__shared__ float shared[32u + D41K_LATENT / 2u];
	LmDependentWait();
	if ( blockIdx.x < rows * heads )
	{
		const uint32_t row = blockIdx.x / heads, head = blockIdx.x % heads;
		uint16_t *merged = (uint16_t *)(shared + 32u);
		D41MergeSinkHead(partials, source_stride, 1u, heads, sink, merged, row, head, shared, shared + 16u);
		__syncthreads();
		if ( threadIdx.x < D41K_ROPE_PAIRS )
			D41RopePair(merged + D41K_LATENT - 2u * D41K_ROPE_PAIRS + 2u * threadIdx.x, positions[row], 0, inverse_frequency[threadIdx.x], 1u);
		__syncthreads();
		uint16_t *target = output + ((uint64_t)row * heads + head) * D41K_LATENT;
		for ( uint32_t element = threadIdx.x; element < D41K_LATENT; element += THREADS )
			target[element] = qdq != 0u ? SparkDsv41FlashFp8QdqLane(LmBf16ToFloat(merged[element])) : merged[element];
	}
	else
	{
		const uint32_t row = blockIdx.x - rows * heads;
		D41WindowRingStoreRow(wave_kv, ring, positions, row_lane, row_ordinal, sequence_row_begin, recurrent_slot, ring_slot_stride, ring_rows, window,
			1u, 0u, rows, commit, (const uint32_t *)0, row);
	}
	LmDependentRelease();
}

template<uint32_t HEADS, uint32_t BLOCK_HEADS = 2u>
static __global__ __launch_bounds__(D41K_THREADS) void D41WindowFoldSpreadKernel(const float *__restrict__ records, uint32_t has_carried,
	const uint16_t *__restrict__ query, const uint16_t *__restrict__ wave_kv, uint16_t *__restrict__ ring, const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal, const uint32_t *__restrict__ sequence_row_begin,
	const uint32_t *__restrict__ sequence_row_indices, const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows,
	uint32_t window, float qk_scale, const float *__restrict__ sink, const float2 *__restrict__ rope_sc, uint16_t *__restrict__ output,
	uint32_t qdq, uint32_t rows, uint32_t commit)
{
	static_assert(HEADS % BLOCK_HEADS == 0u && BLOCK_HEADS == 2u && D41K_LATENT == 2u * D41K_THREADS && D41K_WINDOW_KEYS <= D41K_THREADS,
		"window fold spread processes two heads per block");
	constexpr uint32_t TILE = 16u;
	__shared__ const uint16_t *kptr[D41K_WINDOW_KEYS];
	__shared__ __align__(16) uint16_t s_k[TILE][D41K_LATENT];
	__shared__ float warp_sums[BLOCK_HEADS][TILE][D41K_THREADS / 32u];
	__shared__ float s_keep[BLOCK_HEADS][TILE];
	__shared__ float s_wt[BLOCK_HEADS][TILE];
	__shared__ float s_max[BLOCK_HEADS];
	__shared__ float s_sum[BLOCK_HEADS];
	const uint32_t row = blockIdx.x, hbase = blockIdx.y * BLOCK_HEADS, lane_index = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	LmDependentWait();
	const uint32_t position = positions[row], lane = row_lane[row];
	const uint32_t wave_first = position - row_ordinal[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const uint32_t first = position + 1u > window ? position + 1u - window : 0u, count = position + 1u - first;
	const uint16_t *slot_ring = ring + (uint64_t)recurrent_slot[lane] * ring_slot_stride;
	const uint16_t *qp0 = query + ((uint64_t)row * HEADS + hbase) * D41K_LATENT;
	const uint16_t *qp1 = qp0 + D41K_LATENT;
	const float q00 = LmBf16ToFloat(__ldg(qp0 + threadIdx.x)), q10 = LmBf16ToFloat(__ldg(qp0 + threadIdx.x + D41K_THREADS));
	const float q01 = LmBf16ToFloat(__ldg(qp1 + threadIdx.x)), q11 = LmBf16ToFloat(__ldg(qp1 + threadIdx.x + D41K_THREADS));
	float acc00 = 0.0f, acc10 = 0.0f, acc01 = 0.0f, acc11 = 0.0f;
	if ( has_carried != 0u )
	{
		const float *r0 = records + ((uint64_t)row * HEADS + hbase) * D41K_RECORD;
		const float *r1 = r0 + D41K_RECORD;
		acc00 = __ldg(r0 + 2u + threadIdx.x); acc10 = __ldg(r0 + 2u + threadIdx.x + D41K_THREADS);
		acc01 = __ldg(r1 + 2u + threadIdx.x); acc11 = __ldg(r1 + 2u + threadIdx.x + D41K_THREADS);
		if ( threadIdx.x == 0u ) { s_max[0] = __ldg(r0); s_sum[0] = __ldg(r0 + 1u); }
		else if ( threadIdx.x == 1u ) { s_max[1] = __ldg(r1); s_sum[1] = __ldg(r1 + 1u); }
	}
	else if ( threadIdx.x < BLOCK_HEADS ) { s_max[threadIdx.x] = -INFINITY; s_sum[threadIdx.x] = 0.0f; }
	if ( threadIdx.x < count )
		kptr[threadIdx.x] = D41WindowKey(wave_kv, slot_ring, lane_rows, wave_first, window, first + threadIdx.x);
	__syncthreads();
	for ( uint32_t j = 0u; j < count; j += TILE )
	{
		const uint32_t n = count - j < TILE ? count - j : TILE;
		for ( uint32_t idx = threadIdx.x; idx < n * (D41K_LATENT / 8u); idx += D41K_THREADS )
		{
			const uint32_t t = idx >> 6u, qv = idx & 63u;
			((uint4 *)s_k[t])[qv] = __ldg((const uint4 *)kptr[j + t] + qv);
		}
		__syncthreads();
		for ( uint32_t t = 0u; t < n; t += 2u )
		{
			const uint32_t rem = n - t > 1u ? 2u : 1u;
			const float a0 = LmBf16ToFloat(s_k[t][threadIdx.x]), b0 = LmBf16ToFloat(s_k[t][threadIdx.x + D41K_THREADS]);
			const float a1 = rem > 1u ? LmBf16ToFloat(s_k[t + 1u][threadIdx.x]) : 0.0f;
			const float b1 = rem > 1u ? LmBf16ToFloat(s_k[t + 1u][threadIdx.x + D41K_THREADS]) : 0.0f;
			float p00 = q00 * a0 + q10 * b0, p01 = q00 * a1 + q10 * b1;
			float p10 = q01 * a0 + q11 * b0, p11 = q01 * a1 + q11 * b1;
			#pragma unroll
			for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
			{
				p00 += __shfl_down_sync(0xffffffffu, p00, offset);
				p01 += __shfl_down_sync(0xffffffffu, p01, offset);
				p10 += __shfl_down_sync(0xffffffffu, p10, offset);
				p11 += __shfl_down_sync(0xffffffffu, p11, offset);
			}
			if ( lane_index == 0u )
			{
				warp_sums[0][t][warp] = p00; warp_sums[1][t][warp] = p10;
				if ( rem > 1u ) { warp_sums[0][t + 1u][warp] = p01; warp_sums[1][t + 1u][warp] = p11; }
			}
		}
		__syncthreads();
		if ( threadIdx.x < BLOCK_HEADS )
		{
			const uint32_t h = threadIdx.x;
			float rmax = s_max[h], rsum = s_sum[h];
			for ( uint32_t t = 0u; t < n; ++t )
			{
				float tot = 0.0f;
				#pragma unroll
				for ( uint32_t w = 0u; w < D41K_THREADS / 32u; ++w ) tot += warp_sums[h][t][w];
				const float sc = tot * qk_scale, top = fmaxf(rmax, sc);
				const float kp = rmax == -INFINITY ? 0.0f : __expf(rmax - top), ad = __expf(sc - top);
				rsum = rsum * kp + ad; rmax = top;
				s_keep[h][t] = kp; s_wt[h][t] = SparkMathAttentionValueWeight(ad);
			}
			s_max[h] = rmax; s_sum[h] = rsum;
		}
		__syncthreads();
		for ( uint32_t t = 0u; t < n; ++t )
		{
			const float va = LmBf16ToFloat(s_k[t][threadIdx.x]), vb = LmBf16ToFloat(s_k[t][threadIdx.x + D41K_THREADS]);
			const float kp0 = s_keep[0][t], wt0 = s_wt[0][t], kp1 = s_keep[1][t], wt1 = s_wt[1][t];
			acc00 = acc00 * kp0 + wt0 * va; acc10 = acc10 * kp0 + wt0 * vb;
			acc01 = acc01 * kp1 + wt1 * va; acc11 = acc11 * kp1 + wt1 * vb;
		}
		if ( j + TILE < count ) __syncthreads();
	}
	const float top0 = fmaxf(-1.0e30f, s_max[0]), sc0 = s_max[0] == -INFINITY ? 0.0f : __expf(s_max[0] - top0);
	const float den0 = fmaf(s_sum[0], sc0, 0.0f) + __expf(sink[hbase] - top0);
	const float top1 = fmaxf(-1.0e30f, s_max[1]), sc1 = s_max[1] == -INFINITY ? 0.0f : __expf(s_max[1] - top1);
	const float den1 = fmaf(s_sum[1], sc1, 0.0f) + __expf(sink[hbase + 1u] - top1);
	s_k[0][threadIdx.x] = LmFloatToBf16(fmaf(acc00, sc0, 0.0f) / den0);
	s_k[0][threadIdx.x + D41K_THREADS] = LmFloatToBf16(fmaf(acc10, sc0, 0.0f) / den0);
	s_k[1][threadIdx.x] = LmFloatToBf16(fmaf(acc01, sc1, 0.0f) / den1);
	s_k[1][threadIdx.x + D41K_THREADS] = LmFloatToBf16(fmaf(acc11, sc1, 0.0f) / den1);
	__syncthreads();
	if ( threadIdx.x < D41K_ROPE_PAIRS )
	{
		const float2 sc = rope_sc[(uint64_t)row * D41K_ROPE_PAIRS + threadIdx.x];
		const float sine = -sc.x, cosine = sc.y;
		const uint32_t off = D41K_LATENT - 2u * D41K_ROPE_PAIRS + 2u * threadIdx.x;
		#pragma unroll
		for ( uint32_t h = 0u; h < BLOCK_HEADS; ++h )
		{
			const float r = LmBf16ToFloat(s_k[h][off]), im = LmBf16ToFloat(s_k[h][off + 1u]);
			s_k[h][off] = D41RopeReal(r, im, sine, cosine);
			s_k[h][off + 1u] = D41RopeImaginary(r, im, sine, cosine);
		}
	}
	__syncthreads();
	#pragma unroll
	for ( uint32_t h = 0u; h < BLOCK_HEADS; ++h )
	{
		uint16_t *target = output + ((uint64_t)row * HEADS + hbase + h) * D41K_LATENT;
		if ( qdq != 0u )
		{
			target[threadIdx.x] = SparkDsv41FlashFp8QdqLane(LmBf16ToFloat(s_k[h][threadIdx.x]));
			target[threadIdx.x + D41K_THREADS] = SparkDsv41FlashFp8QdqLane(LmBf16ToFloat(s_k[h][threadIdx.x + D41K_THREADS]));
		}
		else if ( threadIdx.x < D41K_LATENT / 8u )
			((uint4 *)target)[threadIdx.x] = ((const uint4 *)s_k[h])[threadIdx.x];
	}
	if ( blockIdx.y == 0u )
		D41WindowRingStoreRow(wave_kv, ring, positions, row_lane, row_ordinal, sequence_row_begin, recurrent_slot, ring_slot_stride, ring_rows, window,
			1u, 0u, rows, commit, (const uint32_t *)0, row);
	LmDependentRelease();
}

template<uint32_t HEADS>
static __global__ void __cluster_dims__(D41K_WINDOW_PARTS, 1, 1) __launch_bounds__(D41K_THREADS)
D41WindowFoldSpreadClusterKernel(const float *__restrict__ records, uint32_t has_carried, const uint16_t *__restrict__ query,
	const uint16_t *__restrict__ wave_kv, uint16_t *__restrict__ ring, const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane,
	const uint32_t *__restrict__ row_ordinal, const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices,
	const uint32_t *__restrict__ recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows, uint32_t window, float qk_scale,
	const float *__restrict__ sink, const float2 *__restrict__ rope_sc, uint16_t *__restrict__ output, uint32_t qdq, uint32_t rows, uint32_t commit)
{
	static_assert(HEADS == 4u && D41K_WINDOW_PARTS == 8u && D41K_WINDOW_PART_SLICE == 64u && D41K_WINDOW_PART_SLICE == 2u * D41K_ROPE_PAIRS,
		"last cluster part holds the whole rotary slice and two whole fp8 groups");
	__shared__ const uint16_t *kptr[D41K_WINDOW_KEYS];
	__shared__ float warp_sums[HEADS][D41K_WINDOW_PART_KEYS][D41K_THREADS / 32u];
	__shared__ float part_scores[HEADS][D41K_WINDOW_PART_KEYS];
	__shared__ float warp_max[D41K_WINDOW_KEYS / 32u];
	__shared__ float keep[HEADS][D41K_WINDOW_KEYS];
	__shared__ float added[HEADS][D41K_WINDOW_KEYS];
	__shared__ float folded_max[HEADS];
	__shared__ float folded_sum[HEADS];
	__shared__ float s_carried_sum[HEADS];
	__shared__ float ftmp_s[2];
	__shared__ uint16_t s_slice[HEADS][D41K_WINDOW_PART_SLICE];
	cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
	const uint32_t part = blockIdx.x, row = blockIdx.y, lane_index = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	LmDependentWait();
	const uint32_t position = positions[row], lane = row_lane[row];
	const uint32_t wave_first = position - row_ordinal[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const uint32_t first = position + 1u > window ? position + 1u - window : 0u, count = position + 1u - first;
	const uint32_t key_first = part * D41K_WINDOW_PART_KEYS;
	const uint32_t key_count = count > key_first ? (count - key_first < D41K_WINDOW_PART_KEYS ? count - key_first : D41K_WINDOW_PART_KEYS) : 0u;
	const uint16_t *slot_ring = ring + (uint64_t)recurrent_slot[lane] * ring_slot_stride;
	float q0[HEADS], q1[HEADS], carried_max[HEADS];
	#pragma unroll
	for ( uint32_t h = 0u; h < HEADS; ++h )
	{
		const uint16_t *q = query + ((uint64_t)row * HEADS + h) * D41K_LATENT;
		q0[h] = LmBf16ToFloat(__ldg(q + threadIdx.x));
		q1[h] = LmBf16ToFloat(__ldg(q + threadIdx.x + D41K_THREADS));
		if ( has_carried != 0u )
		{
			const float *rec = records + ((uint64_t)row * HEADS + h) * D41K_RECORD;
			carried_max[h] = __ldg(rec);
			if ( threadIdx.x == h ) s_carried_sum[h] = __ldg(rec + 1u);
		}
		else
		{
			carried_max[h] = -INFINITY;
			if ( threadIdx.x == h ) s_carried_sum[h] = 0.0f;
		}
	}
	if ( threadIdx.x < count )
		kptr[threadIdx.x] = D41WindowKey(wave_kv, slot_ring, lane_rows, wave_first, window, first + threadIdx.x);
	__syncthreads();
	uint32_t j = 0u;
	for ( ; j + 3u < key_count; j += 4u )
	{
		const uint16_t *k0 = kptr[key_first + j], *k1 = kptr[key_first + j + 1u], *k2 = kptr[key_first + j + 2u], *k3 = kptr[key_first + j + 3u];
		const uint16_t a0 = __ldg(k0 + threadIdx.x), b0 = __ldg(k0 + threadIdx.x + D41K_THREADS);
		const uint16_t a1 = __ldg(k1 + threadIdx.x), b1 = __ldg(k1 + threadIdx.x + D41K_THREADS);
		const uint16_t a2 = __ldg(k2 + threadIdx.x), b2 = __ldg(k2 + threadIdx.x + D41K_THREADS);
		const uint16_t a3 = __ldg(k3 + threadIdx.x), b3 = __ldg(k3 + threadIdx.x + D41K_THREADS);
		#pragma unroll
		for ( uint32_t h = 0u; h < HEADS; ++h )
		{
			float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
			p0 += q0[h] * LmBf16ToFloat(a0); p0 += q1[h] * LmBf16ToFloat(b0);
			p1 += q0[h] * LmBf16ToFloat(a1); p1 += q1[h] * LmBf16ToFloat(b1);
			p2 += q0[h] * LmBf16ToFloat(a2); p2 += q1[h] * LmBf16ToFloat(b2);
			p3 += q0[h] * LmBf16ToFloat(a3); p3 += q1[h] * LmBf16ToFloat(b3);
			#pragma unroll
			for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
			{
				p0 += __shfl_down_sync(0xffffffffu, p0, offset);
				p1 += __shfl_down_sync(0xffffffffu, p1, offset);
				p2 += __shfl_down_sync(0xffffffffu, p2, offset);
				p3 += __shfl_down_sync(0xffffffffu, p3, offset);
			}
			if ( lane_index == 0u )
			{
				warp_sums[h][j][warp] = p0; warp_sums[h][j + 1u][warp] = p1;
				warp_sums[h][j + 2u][warp] = p2; warp_sums[h][j + 3u][warp] = p3;
			}
		}
	}
	for ( ; j < key_count; ++j )
	{
		const uint16_t *k = kptr[key_first + j];
		const uint16_t a = __ldg(k + threadIdx.x), b = __ldg(k + threadIdx.x + D41K_THREADS);
		#pragma unroll
		for ( uint32_t h = 0u; h < HEADS; ++h )
		{
			float partial = 0.0f;
			partial += q0[h] * LmBf16ToFloat(a); partial += q1[h] * LmBf16ToFloat(b);
			#pragma unroll
			for ( uint32_t offset = 16u; offset != 0u; offset >>= 1u )
				partial += __shfl_down_sync(0xffffffffu, partial, offset);
			if ( lane_index == 0u ) warp_sums[h][j][warp] = partial;
		}
	}
	__syncthreads();
	if ( threadIdx.x < HEADS * D41K_WINDOW_PART_KEYS )
	{
		const uint32_t h = threadIdx.x / D41K_WINDOW_PART_KEYS, kj = threadIdx.x % D41K_WINDOW_PART_KEYS;
		if ( kj < key_count )
		{
			float total = 0.0f;
			#pragma unroll
			for ( uint32_t w = 0u; w < D41K_THREADS / 32u; ++w ) total += warp_sums[h][kj][w];
			part_scores[h][kj] = total * qk_scale;
		}
	}
	cluster.sync();
	#pragma unroll
	for ( uint32_t h = 0u; h < HEADS; ++h )
	{
		float score = -INFINITY, prev_in_warp = -INFINITY;
		if ( threadIdx.x < D41K_WINDOW_KEYS )
		{
			if ( threadIdx.x < count )
			{
				const float (*remote)[D41K_WINDOW_PART_KEYS] = cluster.map_shared_rank(part_scores, threadIdx.x / D41K_WINDOW_PART_KEYS);
				score = remote[h][threadIdx.x % D41K_WINDOW_PART_KEYS];
			}
			prev_in_warp = D41WarpPrefixMax(score, lane_index, &warp_max[warp]);
		}
		__syncthreads();
		D41WindowPrefixWeights(score, carried_max[h], prev_in_warp, warp_max, warp, lane_index, count, keep[h], added[h], ftmp_s);
		if ( threadIdx.x + 1u == count ) folded_max[h] = ftmp_s[0];
		else if ( count == 0u && threadIdx.x == 0u ) folded_max[h] = carried_max[h];
		__syncthreads();
	}
	if ( threadIdx.x < HEADS )
	{
		const uint32_t h = threadIdx.x;
		float running_sum = s_carried_sum[h];
		for ( uint32_t i = 0u; i < count; ++i ) running_sum = running_sum * keep[h][i] + added[h][i];
		folded_sum[h] = running_sum;
	}
	__syncthreads();
	if ( threadIdx.x < D41K_WINDOW_PART_SLICE )
	{
		const uint32_t e = part * D41K_WINDOW_PART_SLICE + threadIdx.x;
		float acc[HEADS];
		#pragma unroll
		for ( uint32_t h = 0u; h < HEADS; ++h )
			acc[h] = has_carried != 0u ? __ldg(records + ((uint64_t)row * HEADS + h) * D41K_RECORD + 2u + e) : 0.0f;
		uint32_t i = 0u;
		for ( ; i + 3u < count; i += 4u )
		{
			const float v0 = LmBf16ToFloat(__ldg(kptr[i] + e));
			const float v1 = LmBf16ToFloat(__ldg(kptr[i + 1u] + e));
			const float v2 = LmBf16ToFloat(__ldg(kptr[i + 2u] + e));
			const float v3 = LmBf16ToFloat(__ldg(kptr[i + 3u] + e));
			#pragma unroll
			for ( uint32_t h = 0u; h < HEADS; ++h )
			{
				acc[h] = acc[h] * keep[h][i] + SparkMathAttentionValueWeight(added[h][i]) * v0;
				acc[h] = acc[h] * keep[h][i + 1u] + SparkMathAttentionValueWeight(added[h][i + 1u]) * v1;
				acc[h] = acc[h] * keep[h][i + 2u] + SparkMathAttentionValueWeight(added[h][i + 2u]) * v2;
				acc[h] = acc[h] * keep[h][i + 3u] + SparkMathAttentionValueWeight(added[h][i + 3u]) * v3;
			}
		}
		for ( ; i < count; ++i )
		{
			const float v = LmBf16ToFloat(__ldg(kptr[i] + e));
			#pragma unroll
			for ( uint32_t h = 0u; h < HEADS; ++h )
				acc[h] = acc[h] * keep[h][i] + SparkMathAttentionValueWeight(added[h][i]) * v;
		}
		#pragma unroll
		for ( uint32_t h = 0u; h < HEADS; ++h )
		{
			const float rmax = folded_max[h], rsum = folded_sum[h];
			const float top = fmaxf(-1.0e30f, rmax);
			const float sc = rmax == -INFINITY ? 0.0f : __expf(rmax - top);
			const float denom = fmaf(rsum, sc, 0.0f) + __expf(sink[h] - top);
			s_slice[h][threadIdx.x] = LmFloatToBf16(fmaf(acc[h], sc, 0.0f) / denom);
		}
		if ( part == D41K_WINDOW_PARTS - 1u )
		{
			asm volatile("bar.sync 2, 64;\n" ::: "memory");
			if ( threadIdx.x < D41K_ROPE_PAIRS )
			{
				const float2 sc = rope_sc[(uint64_t)row * D41K_ROPE_PAIRS + threadIdx.x];
				const float sine = -sc.x, cosine = sc.y;
				#pragma unroll
				for ( uint32_t h = 0u; h < HEADS; ++h )
				{
					const float r = LmBf16ToFloat(s_slice[h][2u * threadIdx.x]), im = LmBf16ToFloat(s_slice[h][2u * threadIdx.x + 1u]);
					s_slice[h][2u * threadIdx.x] = D41RopeReal(r, im, sine, cosine);
					s_slice[h][2u * threadIdx.x + 1u] = D41RopeImaginary(r, im, sine, cosine);
				}
			}
			asm volatile("bar.sync 2, 64;\n" ::: "memory");
		}
		#pragma unroll
		for ( uint32_t h = 0u; h < HEADS; ++h )
		{
			uint16_t val = s_slice[h][threadIdx.x];
			if ( qdq != 0u ) val = SparkDsv41FlashFp8QdqLane(LmBf16ToFloat(val));
			output[((uint64_t)row * HEADS + h) * D41K_LATENT + e] = val;
		}
	}
	if ( part == 0u )
		D41WindowRingStoreRow(wave_kv, ring, positions, row_lane, row_ordinal, sequence_row_begin, recurrent_slot, ring_slot_stride, ring_rows, window,
			1u, 0u, rows, commit, (const uint32_t *)0, row);
	LmDependentRelease();
}

static __global__ void D41SwigluRowsKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ output, uint32_t width, float limit)
{ LmDependentWait(); LmDependentRelease(); const uint64_t row_base = (uint64_t)blockIdx.y * width; LmBinaryBf16GridBody(gate + row_base, up + row_base, output + row_base, width, LmSiluGateLimitOp{limit}); }

static __global__ void D41SwigluQdqRowsKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ output, uint32_t width, float limit)
{
	const uint32_t octets = width >> 3u, q = blockIdx.x * blockDim.x + threadIdx.x;
	LmDependentWait();
	if ( q < octets )
	{
		const uint64_t base = (uint64_t)blockIdx.y * octets + q;
		float v[8];
		D41Swiglu8(((const uint4 *)gate)[base], ((const uint4 *)up)[base], 1.0f, limit, v);
		((uint4 *)output)[base] = D41Fp8Qdq8(v);
	}
	LmDependentRelease();
}

static __global__ void D41SwigluMxKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint8_t *__restrict__ codes,
	uint8_t *__restrict__ scales, uint32_t width, float limit)
{
	const uint32_t octets = width >> 3u, q = blockIdx.x * blockDim.x + threadIdx.x;
	LmDependentWait();
	if ( q < octets )
	{
		const uint64_t base = (uint64_t)blockIdx.y * octets + q;
		float v[8]; uint2 packed; uint8_t sc;
		D41Swiglu8(((const uint4 *)gate)[base], ((const uint4 *)up)[base], 1.0f, limit, v);
		D41Fp8Quant8(v, &packed, &sc);
		((uint2 *)codes)[base] = packed;
		if ( (q & 3u) == 0u ) scales[base >> 2u] = sc;
	}
	LmDependentRelease();
}

static __global__ void D41RankMajorMxKernel(const uint16_t *__restrict__ gathered, uint8_t *__restrict__ codes, uint8_t *__restrict__ scales,
	uint32_t rows, uint32_t width, uint32_t ranks, uint32_t first_rank)
{
	const uint32_t total_octets = (ranks * width) >> 3u, q = blockIdx.x * blockDim.x + threadIdx.x;
	LmDependentWait();
	if ( q < total_octets )
	{
		const uint32_t width_octets = width >> 3u, r = q / width_octets, within = q - r * width_octets;
		float v[8]; uint2 packed; uint8_t sc;
		LmLoadBf16x8AsFloat(((const uint4 *)gathered)[((uint64_t)(first_rank + r) * rows + blockIdx.y) * width_octets + within], v);
		D41Fp8Quant8(v, &packed, &sc);
		const uint64_t dst = (uint64_t)blockIdx.y * total_octets + q;
		((uint2 *)codes)[dst] = packed;
		if ( (q & 3u) == 0u ) scales[dst >> 2u] = sc;
	}
	LmDependentRelease();
}

static __global__ void D41RankMajorQdqKernel(const uint16_t *__restrict__ gathered, uint16_t *__restrict__ target,
	uint32_t rows, uint32_t width, uint32_t ranks, uint32_t first_rank)
{
	const uint32_t total_octets = (ranks * width) >> 3u, q = blockIdx.x * blockDim.x + threadIdx.x;
	LmDependentWait();
	if ( q < total_octets )
	{
		const uint32_t width_octets = width >> 3u, r = q / width_octets, within = q - r * width_octets;
		float v[8];
		LmLoadBf16x8AsFloat(((const uint4 *)gathered)[((uint64_t)(first_rank + r) * rows + blockIdx.y) * width_octets + within], v);
		((uint4 *)target)[(uint64_t)blockIdx.y * total_octets + q] = D41Fp8Qdq8(v);
	}
	LmDependentRelease();
}

template<uint32_t STREAMS, uint32_t MODE>
static __global__ __launch_bounds__(256)
void D41HcPostFusedKernel(const void *__restrict__ gathered, const float *__restrict__ post, const float *__restrict__ comb,
	uint16_t *__restrict__ streams, uint32_t rows, uint32_t width, uint32_t degree, uint32_t slice_width)
{
	__shared__ float sp[STREAMS], sc[STREAMS][STREAMS];
	const uint32_t row = blockIdx.y, octets = width >> 3u;
	LmDependentWait();
	if ( threadIdx.x < STREAMS ) sp[threadIdx.x] = __ldg(post + (uint64_t)row * STREAMS + threadIdx.x);
	if ( threadIdx.x < STREAMS * STREAMS )
		sc[threadIdx.x / STREAMS][threadIdx.x % STREAMS] = __ldg(comb + (uint64_t)row * STREAMS * STREAMS + threadIdx.x);
	__syncthreads();
	for ( uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < octets; q += gridDim.x * blockDim.x )
	{
		uint4 sub;
		if constexpr ( MODE == 1u )
		{
			const uint32_t sw_oct = slice_width >> 3u, r = q / sw_oct, within = q - r * sw_oct;
			sub = __ldg((const uint4 *)gathered + ((uint64_t)r * rows + row) * sw_oct + within);
		}
		else if constexpr ( MODE == 2u )
		{
			const float *gf = (const float *)gathered;
			const uint64_t elems = (uint64_t)rows * width, base = (uint64_t)row * width + ((uint64_t)q << 3u);
			float s[8] = {}, v[8];
			for ( uint32_t src = 0u; src < degree; ++src )
			{
				LmLoadFloat8(gf + (uint64_t)src * elems + base, v);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k ) s[k] += v[k];
			}
			sub = LmPackUint4FloatToBf16(s);
		}
		else
		{
			const uint4 *gb = (const uint4 *)gathered;
			const uint64_t row_oct = (uint64_t)rows * octets, base = (uint64_t)row * octets + q;
			float s[8] = {};
			for ( uint32_t src = 0u; src < degree; ++src ) LmAddBf16x8(s, gb[(uint64_t)src * row_oct + base]);
			sub = LmPackUint4FloatToBf16(s);
		}
		const uint32_t offset = q << 3u;
		uint4 sraw[STREAMS];
		#pragma unroll
		for ( uint32_t copy = 0u; copy < STREAMS; ++copy )
			sraw[copy] = *(const uint4 *)(streams + ((uint64_t)row * STREAMS + copy) * width + offset);
		const uint32_t subw[4] = { sub.x, sub.y, sub.z, sub.w };
		for ( uint32_t copy = 0u; copy < STREAMS; ++copy )
		{
			const float pc = sp[copy];
			uint32_t outw[4];
			#pragma unroll
			for ( uint32_t w = 0u; w < 4u; ++w )
			{
				float t0 = 0.0f, t1 = 0.0f;
				#pragma unroll
				for ( uint32_t from = 0u; from < STREAMS; ++from )
				{
					const uint32_t sw = w == 0u ? sraw[from].x : w == 1u ? sraw[from].y : w == 2u ? sraw[from].z : sraw[from].w;
					const float cf = sc[from][copy];
					t0 += cf * __uint_as_float(sw << 16u);
					t1 += cf * __uint_as_float(sw & 0xffff0000u);
				}
				const float x0 = __uint_as_float(subw[w] << 16u), x1 = __uint_as_float(subw[w] & 0xffff0000u);
				outw[w] = LmPackBf16Pair(pc * x0 + t0, pc * x1 + t1);
			}
			*(uint4 *)(streams + ((uint64_t)row * STREAMS + copy) * width + offset) = make_uint4(outw[0], outw[1], outw[2], outw[3]);
		}
	}
	LmDependentRelease();
}

static __global__ void D41ColumnsKernel(const uint16_t *__restrict__ source, uint32_t source_stride, uint16_t *__restrict__ target, uint32_t target_stride, uint32_t width)
{
	const uint16_t *src = source + (uint64_t)blockIdx.y * source_stride;
	uint16_t *dst = target + (uint64_t)blockIdx.y * target_stride;
	if ( (width & 7u) == 0u && LmIsAligned16(src) && LmIsAligned16(dst) ) {
		for ( uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < (width >> 3u); q += gridDim.x * blockDim.x ) ((uint4 *)dst)[q] = ((const uint4 *)src)[q];
		return;
	}
	for ( uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < width; i += gridDim.x * blockDim.x ) dst[i] = src[i];
}

static __global__ void D41IndexPackKernel(const uint16_t *__restrict__ query, const uint16_t *__restrict__ weight, uint16_t *__restrict__ packed,
	uint32_t query_width, uint32_t weight_width, uint32_t pack_width)
{
	LmDependentWait();
	LmDependentRelease();
	uint16_t *p_row = packed + (uint64_t)blockIdx.x * pack_width;
	LmCopySliceBf16<D41K_THREADS>(query + (uint64_t)blockIdx.x * query_width, p_row, query_width);
	for ( uint32_t i = threadIdx.x; i < pack_width - query_width; i += D41K_THREADS ) p_row[query_width + i] = i < weight_width ? weight[(uint64_t)blockIdx.x * weight_width + i] : (uint16_t)0u;
}

static __global__ void D41IndexUnpackKernel(const uint16_t *__restrict__ packed, uint64_t rank_stride, uint16_t *__restrict__ query,
	uint16_t *__restrict__ weight, uint32_t query_width, uint32_t weight_width, uint32_t pack_width)
{
	LmDependentWait();
	LmDependentRelease();
	const uint16_t *src = packed + (uint64_t)blockIdx.y * rank_stride + (uint64_t)blockIdx.x * pack_width;
	LmCopySliceBf16<D41K_THREADS>(src, query + ((uint64_t)blockIdx.x * gridDim.y + blockIdx.y) * query_width, query_width);
	LmCopySliceBf16<D41K_THREADS>(src + query_width, weight + ((uint64_t)blockIdx.x * gridDim.y + blockIdx.y) * weight_width, weight_width);
}

static inline dim3 D41ItemGrid(uint32_t blocks, uint32_t items) { const uint32_t span = items < D41K_GRID_ITEMS ? (items != 0u ? items : 1u) : D41K_GRID_ITEMS; return dim3(blocks, span, (items + span - 1u) / span); }

static __global__ void D41MoeSendKernel(const uint16_t *__restrict__ gate_up, const float *__restrict__ packed_weight,
	const uint16_t *__restrict__ shared, uint16_t *__restrict__ send, uint32_t routes, uint32_t rows, uint32_t slice, uint32_t hidden_slice,
	uint32_t degree, uint64_t destination_stride, float limit, uint32_t route_elements, uint32_t codes)
{
	const uint32_t item = blockIdx.z * gridDim.y + blockIdx.y, element = blockIdx.x * blockDim.x + threadIdx.x;
	LmDependentWait();
	LmDependentRelease();
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
		uint8_t exponent;
		const uint8_t code = D41QuantFp8Block(value, &exponent);
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

static __global__ void D41ActAssembleMxKernel(const uint16_t *__restrict__ received, uint8_t *__restrict__ codes, uint32_t slice, uint32_t width,
	uint64_t source_stride, uint32_t routes)
{
	const uint32_t route = blockIdx.z * gridDim.y + blockIdx.y, q = blockIdx.x * blockDim.x + threadIdx.x, octets = width >> 3u;
	LmDependentWait();
	LmDependentRelease();
	if ( route >= routes || q >= octets ) return;
	const uint32_t element = q << 3u, source = element / slice, within = element - source * slice;
	float v[8];
	LmLoadBf16x8AsFloat(*(const uint4 *)(received + (uint64_t)source * source_stride + (uint64_t)route * slice + within), v);
	uint2 packed;
	uint8_t sc;
	D41Fp8Quant8(v, &packed, &sc);
	*(uint2 *)(codes + (uint64_t)route * width + element) = packed;
	if ( (q & 3u) == 0u ) codes[(uint64_t)routes * width + (uint64_t)route * (width / SPARK_DSV41_FLASH_FP8_BLOCK) + (q >> 2u)] = sc;
}

static __global__ void D41ActAssembleQdqKernel(const uint16_t *__restrict__ received, uint16_t *__restrict__ act, uint32_t slice, uint32_t width,
	uint64_t source_stride, uint32_t routes, uint32_t codes)
{
	const uint32_t route = blockIdx.z * gridDim.y + blockIdx.y, element = blockIdx.x * blockDim.x + threadIdx.x, source = element / slice;
	LmDependentWait();
	LmDependentRelease();
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

static __global__ void D41MoeSendVecKernel(const uint16_t *__restrict__ gate_up, const float *__restrict__ packed_weight,
	const uint16_t *__restrict__ shared, uint16_t *__restrict__ send, uint32_t routes, uint32_t rows, uint32_t slice, uint32_t hidden_slice,
	uint32_t degree, uint64_t destination_stride, float limit, uint32_t route_elements, uint32_t codes)
{
	const uint32_t item = blockIdx.z * gridDim.y + blockIdx.y, q = blockIdx.x * blockDim.x + threadIdx.x;
	LmDependentWait();
	LmDependentRelease();
	if ( item < routes )
	{
		const uint32_t route = item, octets = slice >> 3u;
		if ( q >= octets )
			return;
		const uint64_t g_base = (uint64_t)route * 2u * octets + q;
		float v[8];
		const uint4 bf = D41Swiglu8(((const uint4 *)gate_up)[g_base], ((const uint4 *)gate_up)[g_base + octets], packed_weight[route], limit, v);
		if ( codes == 0u )
		{
			for ( uint32_t d = 0u; d < degree; ++d )
				((uint4 *)(send + (uint64_t)d * destination_stride + (uint64_t)route * slice))[q] = bf;
			return;
		}
		uint2 packed;
		uint8_t sc;
		D41Fp8Quant8(v, &packed, &sc);
		const uint32_t s_blocks = slice / SPARK_DSV41_FLASH_FP8_BLOCK;
		for ( uint32_t d = 0u; d < degree; ++d )
		{
			uint8_t *bytes = (uint8_t *)(send + (uint64_t)d * destination_stride);
			*(uint2 *)(bytes + (uint64_t)route * slice + (q << 3u)) = packed;
			if ( (q & 3u) == 0u )
				bytes[(uint64_t)routes * slice + (uint64_t)route * s_blocks + (q >> 2u)] = sc;
		}
		return;
	}
	const uint32_t row = item - routes, h_oct = hidden_slice >> 3u;
	if ( row >= rows || q >= h_oct * degree )
		return;
	const uint32_t d = q / h_oct, within = q - d * h_oct;
	((uint4 *)(send + (uint64_t)d * destination_stride + (uint64_t)routes * route_elements + (uint64_t)row * hidden_slice))[within] =
		((const uint4 *)(shared + (uint64_t)row * hidden_slice * degree))[q];
}

static __global__ void D41ActAssembleQdqVecKernel(const uint16_t *__restrict__ received, uint16_t *__restrict__ act, uint32_t slice, uint32_t width,
	uint64_t source_stride, uint32_t routes, uint32_t codes)
{
	const uint32_t route = blockIdx.z * gridDim.y + blockIdx.y, q = blockIdx.x * blockDim.x + threadIdx.x, octets = width >> 3u;
	LmDependentWait();
	LmDependentRelease();
	if ( route >= routes || q >= octets )
		return;
	const uint32_t element = q << 3u, source = element / slice, within = element - source * slice;
	if ( codes != 0u )
	{
		const uint8_t *bytes = (const uint8_t *)(received + (uint64_t)source * source_stride);
		const uint32_t s_blocks = slice / SPARK_DSV41_FLASH_FP8_BLOCK;
		const uint32_t exponent = bytes[(uint64_t)routes * slice + (uint64_t)route * s_blocks + within / SPARK_DSV41_FLASH_FP8_BLOCK];
		const uint2 c = *(const uint2 *)(bytes + (uint64_t)route * slice + within);
		((uint4 *)(act + (uint64_t)route * width))[q] = D41DequantE4m3x8(c, __uint_as_float(exponent << 23u));
		return;
	}
	float v[8];
	LmLoadBf16x8AsFloat(*(const uint4 *)(received + (uint64_t)source * source_stride + (uint64_t)route * slice + within), v);
	((uint4 *)(act + (uint64_t)route * width))[q] = D41Fp8Qdq8(v);
}

static __global__ void D41ActAssembleCodesKernel(const uint16_t *__restrict__ received, uint8_t *__restrict__ codes, uint32_t slice, uint32_t width,
	uint64_t source_stride, uint32_t routes)
{
	const uint32_t route = blockIdx.z * gridDim.y + blockIdx.y, vector = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t code_vectors = width / 16u, blocks = slice / SPARK_DSV41_FLASH_FP8_BLOCK;
	LmDependentWait();
	LmDependentRelease();
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

static __global__ void D41SharedSendKernel(const uint16_t *__restrict__ gate, const uint16_t *__restrict__ up, uint16_t *__restrict__ send,
	uint32_t rows, uint32_t width, uint32_t degree, uint64_t destination_stride, float limit)
{
	const uint32_t row = blockIdx.y, q = blockIdx.x * blockDim.x + threadIdx.x, octets = width >> 3u;
	LmDependentWait();
	LmDependentRelease();
	if ( row >= rows || q >= octets ) return;
	const uint64_t base = (uint64_t)row * octets + q;
	float v[8];
	D41Swiglu8(((const uint4 *)gate)[base], ((const uint4 *)up)[base], 1.0f, limit, v);
	uint2 packed;
	uint8_t sc;
	D41Fp8Quant8(v, &packed, &sc);
	const uint32_t s_blocks = width / SPARK_DSV41_FLASH_FP8_BLOCK;
	for ( uint32_t d = 0u; d < degree; ++d )
	{
		uint8_t *bytes = (uint8_t *)(send + (uint64_t)d * destination_stride);
		*(uint2 *)(bytes + (uint64_t)row * width + (q << 3u)) = packed;
		if ( (q & 3u) == 0u ) bytes[(uint64_t)rows * width + (uint64_t)row * s_blocks + (q >> 2u)] = sc;
	}
}

static __global__ void D41SharedDequantAllKernel(const uint16_t *__restrict__ received, uint16_t *__restrict__ act, uint32_t rows, uint64_t stride,
	uint32_t degree, uint32_t total_granules, uint32_t granule)
{
	const uint32_t row = blockIdx.y, source = blockIdx.z;
	LmDependentWait();
	LmDependentRelease();
	if ( row >= rows || source >= degree )
		return;
	const uint32_t base = total_granules / degree, extra = total_granules % degree;
	const uint32_t first = (source * base + (source < extra ? source : extra)) * granule;
	const uint32_t width = (base + (source < extra ? 1u : 0u)) * granule;
	const uint8_t *bytes = (const uint8_t *)(received + (uint64_t)source * stride);
	D41DequantRowE4m3(bytes + (uint64_t)row * width, bytes + (uint64_t)rows * width + (uint64_t)row * (width / SPARK_DSV41_FLASH_FP8_BLOCK),
		act + (uint64_t)first * rows + (uint64_t)row * width, width);
}

static __global__ void D41L2PrefetchKernel(const uint8_t *__restrict__ ptr0, uint64_t bytes0, const uint8_t *__restrict__ ptr1, uint64_t bytes1)
{
	constexpr uint32_t PIECE = 16384u;
	const uint64_t stride = (uint64_t)gridDim.x * blockDim.x * PIECE;
	const uint64_t start = (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) * PIECE;
	if ( ptr0 != 0 )
		for ( uint64_t off = start; off < bytes0; off += stride )
			LmPrefetchL2(ptr0 + off, (uint32_t)(bytes0 - off < PIECE ? bytes0 - off : PIECE));
	if ( ptr1 != 0 )
		for ( uint64_t off = start; off < bytes1; off += stride )
			LmPrefetchL2(ptr1 + off, (uint32_t)(bytes1 - off < PIECE ? bytes1 - off : PIECE));
}

static __global__ void D41SharedDequantKernel(const uint8_t *__restrict__ bytes, uint16_t *__restrict__ act, uint32_t rows, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	LmDependentWait();
	LmDependentRelease();
	if ( row >= rows )
		return;
	D41DequantRowE4m3(bytes + (uint64_t)row * width, bytes + (uint64_t)rows * width + (uint64_t)row * (width / SPARK_DSV41_FLASH_FP8_BLOCK),
		act + (uint64_t)row * width, width);
}

static __global__ void D41FfnPackKernel(const uint32_t *__restrict__ route_expert, const float *__restrict__ route_weight, uint8_t *__restrict__ block,
	uint32_t own, uint32_t routed, uint32_t hidden, uint32_t top_k)
{
	const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t ids = (uint64_t)own * (hidden + hidden / SPARK_DSV41_FLASH_FP8_BLOCK);
	LmDependentWait();
	LmDependentRelease();
	if ( item >= routed * top_k )
		return;
	((uint32_t *)(block + ids))[item] = route_expert[item];
	((float *)(block + ids + (uint64_t)own * top_k * sizeof(uint32_t)))[item] = route_weight[item];
}

static __global__ void D41FfnUnpackKernel(const uint8_t *__restrict__ received, uint16_t *__restrict__ xq, uint8_t *__restrict__ mx_codes,
	uint8_t *__restrict__ mx_scales, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t own, uint32_t hidden,
	uint32_t top_k, uint64_t block_bytes)
{
	const uint32_t row = blockIdx.y, source = row / own, index = row - source * own;
	const uint8_t *block = received + (uint64_t)source * block_bytes;
	const uint32_t scale_bytes = hidden / SPARK_DSV41_FLASH_FP8_BLOCK;
	const uint64_t ids = (uint64_t)own * (hidden + scale_bytes);
	LmDependentWait();
	LmDependentRelease();
	for ( uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < top_k; k += gridDim.x * blockDim.x )
	{
		route_expert[(uint64_t)row * top_k + k] = ((const uint32_t *)(block + ids))[(uint64_t)index * top_k + k];
		route_weight[(uint64_t)row * top_k + k] =
			((const float *)(block + ids + (uint64_t)own * top_k * sizeof(uint32_t)))[(uint64_t)index * top_k + k];
	}
	if ( mx_codes != 0 )
	{
		const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x, lanes = gridDim.x * blockDim.x;
		for ( uint32_t i = lane; i < hidden / 16u; i += lanes )
			((uint4 *)(mx_codes + (uint64_t)row * hidden))[i] = ((const uint4 *)(block + (uint64_t)index * hidden))[i];
		for ( uint32_t i = lane; i < scale_bytes / 4u; i += lanes )
			((uint32_t *)(mx_scales + (uint64_t)row * scale_bytes))[i] =
				((const uint32_t *)(block + (uint64_t)own * hidden + (uint64_t)index * scale_bytes))[i];
		return;
	}
	D41DequantRowE4m3(block + (uint64_t)index * hidden, block + (uint64_t)own * hidden + (uint64_t)index * scale_bytes, xq + (uint64_t)row * hidden, hidden);
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
	uint32_t direct_pair[RANKS];
	uint16_t routes[EXPERTS];
	uint16_t direct_sole[EXPERTS];
	uint8_t sole[RANKS * EXPERTS];
};

template <uint32_t RANKS, uint32_t EXPERTS>
struct D41ExchangeTotals
{
	uint32_t pair[RANKS * RANKS];
	uint32_t direct_pair[RANKS * RANKS];
	uint32_t routes[EXPERTS];
	uint16_t sole[RANKS * EXPERTS];
	uint16_t direct_sole[RANKS * EXPERTS];
};

template <uint32_t RANKS, uint32_t EXPERTS>
static __global__ void D41ExchangeCountsKernel(const uint32_t *__restrict__ route_expert, uint32_t rows, uint32_t first, uint32_t top_k,
	const uint8_t *__restrict__ owner, uint32_t backups, D41ExchangeCounts<RANKS, EXPERTS> *__restrict__ counts)
{
	__shared__ uint32_t pair[RANKS * RANKS], direct_pair[RANKS], routes[EXPERTS], direct_sole[EXPERTS], sole[RANKS * EXPERTS];
	for ( uint32_t i = threadIdx.x; i < RANKS * EXPERTS; i += blockDim.x ) sole[i] = 0u;
	for ( uint32_t i = threadIdx.x; i < EXPERTS; i += blockDim.x ) routes[i] = direct_sole[i] = 0u;
	for ( uint32_t i = threadIdx.x; i < RANKS * RANKS; i += blockDim.x ) pair[i] = 0u;
	if ( threadIdx.x < RANKS ) direct_pair[threadIdx.x] = 0u;
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
			{
				atomicAdd(&pair[target * RANKS + rank], 1u);
				atomicAdd(&direct_pair[rank], 1u);
			}
		for ( uint32_t k = 0u; k < top_k && backups != 0u; ++k )
		{
			const uint32_t expert = route_expert[(uint64_t)row * top_k + k];
			atomicAdd(&routes[expert], 1u);
			if ( ((twice >> owner[expert]) & 1u) == 0u )
			{
				atomicAdd(&sole[target * EXPERTS + expert], 1u);
				atomicAdd(&direct_sole[expert], 1u);
			}
		}
	}
	__syncthreads();
	for ( uint32_t i = threadIdx.x; i < RANKS * RANKS; i += blockDim.x ) counts->pair[i] = pair[i];
	if ( threadIdx.x < RANKS ) counts->direct_pair[threadIdx.x] = direct_pair[threadIdx.x];
	for ( uint32_t i = threadIdx.x; i < EXPERTS && backups != 0u; i += blockDim.x )
	{
		counts->routes[i] = (uint16_t)routes[i];
		counts->direct_sole[i] = (uint16_t)direct_sole[i];
	}
	for ( uint32_t i = threadIdx.x; i < RANKS * EXPERTS && backups != 0u; i += blockDim.x ) counts->sole[i] = (uint8_t)sole[i];
}

template <uint32_t RANKS, uint32_t EXPERTS>
static __global__ void D41ExchangeTotalsKernel(const uint8_t *__restrict__ gathered, uint64_t stride, uint32_t backups,
	D41ExchangeTotals<RANKS, EXPERTS> *__restrict__ totals)
{
	constexpr uint32_t P = RANKS * RANKS, S = RANKS * EXPERTS;
	const uint32_t items = 2u * P + (backups != 0u ? EXPERTS + 2u * S : 0u);
	for ( uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < items; item += gridDim.x * blockDim.x )
	{
		if ( item >= P && item < 2u * P )
		{
			const uint32_t idx = item - P;
			totals->direct_pair[idx] = ((const D41ExchangeCounts<RANKS, EXPERTS> *)(gathered + (idx / RANKS) * stride))->direct_pair[idx % RANKS];
			continue;
		}
		if ( item >= 2u * P + EXPERTS + S )
		{
			const uint32_t idx = item - (2u * P + EXPERTS + S);
			totals->direct_sole[idx] = ((const D41ExchangeCounts<RANKS, EXPERTS> *)(gathered + (idx / EXPERTS) * stride))->direct_sole[idx % EXPERTS];
			continue;
		}
		uint32_t total = 0u;
		for ( uint32_t sender = 0u; sender < RANKS; ++sender )
		{
			const D41ExchangeCounts<RANKS, EXPERTS> *counts = (const D41ExchangeCounts<RANKS, EXPERTS> *)(gathered + sender * stride);
			total += item < P ? counts->pair[item] : item < 2u * P + EXPERTS ? (uint32_t)counts->routes[item - 2u * P] :
				(uint32_t)counts->sole[item - 2u * P - EXPERTS];
		}
		if ( item < P ) totals->pair[item] = total;
		else if ( item < 2u * P + EXPERTS ) totals->routes[item - 2u * P] = total;
		else totals->sole[item - 2u * P - EXPERTS] = (uint16_t)total;
	}
}

template <uint32_t RANKS, uint32_t EXPERTS, uint32_t LOCAL, uint32_t COPIES>
static inline uint32_t D41BackupPlanFor(const uint32_t *base_pair, const uint32_t *routes, const uint16_t *sole, const uint8_t *owner,
	const uint8_t *backup, const uint16_t *expert_at, uint32_t limit, uint16_t *quota, uint32_t *pair)
{
	uint64_t load[RANKS] = {0};
	uint32_t avail[EXPERTS], left[RANKS * EXPERTS], moved[COPIES][EXPERTS], any = 0u;
	uint8_t paid[COPIES][EXPERTS];
	memcpy(pair, base_pair, (size_t)RANKS * RANKS * sizeof(uint32_t));
	memset(quota, 0, (size_t)COPIES * RANKS * EXPERTS * sizeof(uint16_t));
	memset(moved, 0, sizeof(moved));
	memset(paid, 0, sizeof(paid));
	for ( uint32_t expert = 0u; expert < EXPERTS; ++expert )
	{
		uint32_t total = 0u, usable = 0u;
		for ( uint32_t target = 0u; target < RANKS; ++target )
		{
			total += sole[target * EXPERTS + expert];
			left[target * EXPERTS + expert] = sole[target * EXPERTS + expert];
		}
		for ( uint32_t copy = 0u; copy < COPIES; ++copy )
			usable |= backup[copy * EXPERTS + expert] < RANKS && backup[copy * EXPERTS + expert] != owner[expert] ? 1u : 0u;
		avail[expert] = usable != 0u ? total : 0u;
		if ( routes[expert] != 0u )
			load[owner[expert]] += D41K_BACKUP_EXPERT_ROUTES + routes[expert];
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
			memcpy(pair, base_pair, (size_t)RANKS * RANKS * sizeof(uint32_t));
			memset(quota, 0, (size_t)COPIES * RANKS * EXPERTS * sizeof(uint16_t));
			return 0u;
		}
	return any;
}

template <uint32_t RANKS, uint32_t EXPERTS, uint32_t LOCAL, uint32_t COPIES>
static inline uint32_t D41BackupPlan(const D41ExchangeTotals<RANKS, EXPERTS> *totals, const uint8_t *owner, const uint8_t *backup,
	const uint16_t *expert_at, uint32_t limit, uint16_t *quota, uint32_t *pair)
{
	return D41BackupPlanFor<RANKS, EXPERTS, LOCAL, COPIES>(totals->pair, totals->routes, totals->sole, owner, backup, expert_at, limit, quota, pair);
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
	LmDependentWait();
	LmDependentRelease();
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

template<uint32_t THREADS, uint32_t HELD_PLUS_ONE>
static __device__ __forceinline__ void D41RouteScan3Warp(uint32_t *s_count, uint32_t *s_warp_tot, uint32_t *__restrict__ group_row_offset,
	uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	static_assert(HELD_PLUS_ONE <= 96u, "balanced route groups fit three warps");
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	if ( warp < 3u )
	{
		const uint32_t e = threadIdx.x;
		const uint32_t c = e < HELD_PLUS_ONE ? s_count[e] : 0u;
		const uint32_t mt = c != 0u ? (c + tile_m - 1u) / tile_m : 0u;
		const uint32_t up = mt * neuron_tiles_up, dn = mt * neuron_tiles_down;
		uint32_t inc_c = c, inc_u = up, inc_d = dn;
		#pragma unroll
		for ( uint32_t off = 1u; off < 32u; off <<= 1u )
		{
			const uint32_t ac = __shfl_up_sync(0xffffffffu, inc_c, off);
			const uint32_t au = __shfl_up_sync(0xffffffffu, inc_u, off);
			const uint32_t ad = __shfl_up_sync(0xffffffffu, inc_d, off);
			if ( lane >= off ) { inc_c += ac; inc_u += au; inc_d += ad; }
		}
		if ( lane == 31u ) { s_warp_tot[warp] = inc_c; s_warp_tot[4u + warp] = inc_u; s_warp_tot[8u + warp] = inc_d; }
		asm volatile("bar.sync 3, 96;\n" ::: "memory");
		uint32_t bc = inc_c - c, bu = inc_u - up, bd = inc_d - dn;
		if ( warp >= 1u ) { bc += s_warp_tot[0]; bu += s_warp_tot[4]; bd += s_warp_tot[8]; }
		if ( warp >= 2u ) { bc += s_warp_tot[1]; bu += s_warp_tot[5]; bd += s_warp_tot[9]; }
		if ( e < HELD_PLUS_ONE )
		{
			s_count[e] = bc;
			group_row_offset[e] = bc;
			if ( tile_prefix_up != 0 && tile_prefix_down != 0 )
			{
				tile_prefix_up[e] = bu;
				tile_prefix_down[e] = bd;
			}
			if ( e + 1u == HELD_PLUS_ONE )
			{
				group_row_offset[HELD_PLUS_ONE] = bc + c;
				if ( tile_prefix_up != 0 && tile_prefix_down != 0 )
				{
					tile_prefix_up[HELD_PLUS_ONE] = bu + up;
					tile_prefix_down[HELD_PLUS_ONE] = bd + dn;
				}
			}
		}
	}
}

template<uint32_t RANKS, uint32_t EXPERTS, uint32_t LOCAL, uint32_t THREADS, uint32_t HELD_PLUS_ONE>
static __global__ __launch_bounds__(THREADS, 1)
void D41BalanceRouteKernel(uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ route_global,
	uint32_t rows, uint32_t top_k, uint32_t rank, uint32_t copies, uint32_t row_invariant, const uint8_t *__restrict__ owner,
	const uint8_t *__restrict__ backup, const uint8_t *__restrict__ group, uint32_t *__restrict__ group_row_offset,
	uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up,
	uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down, float *__restrict__ packed_weight)
{
	constexpr uint32_t CHUNK = 512u, WARPS = THREADS / 32u;
	__shared__ uint8_t s_owner[EXPERTS];
	__shared__ uint8_t s_backup[2u * EXPERTS];
	__shared__ uint8_t s_group[EXPERTS];
	__shared__ uint8_t expert_rank[EXPERTS];
	__shared__ uint16_t s_local[CHUNK];
	__shared__ uint32_t s_rank_load[RANKS];
	__shared__ uint32_t s_unseen;
	__shared__ uint32_t s_count[HELD_PLUS_ONE];
	__shared__ uint32_t s_warp_tot[12];
	__shared__ uint16_t s_warp_cnt[WARPS][HELD_PLUS_ONE];
	const uint32_t routes = rows * top_k, groups = LOCAL + copies * LOCAL;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	for ( uint32_t e = threadIdx.x; e < EXPERTS; e += THREADS )
	{
		s_owner[e] = owner[e];
		s_group[e] = group[e];
		expert_rank[e] = 255u;
	}
	for ( uint32_t i = threadIdx.x; i < copies * EXPERTS; i += THREADS )
		s_backup[i] = backup[i];
	if ( threadIdx.x < RANKS ) s_rank_load[threadIdx.x] = 0u;
	if ( threadIdx.x == 0u ) s_unseen = EXPERTS;
	for ( uint32_t i = threadIdx.x; i < HELD_PLUS_ONE; i += THREADS ) s_count[i] = 0u;
	for ( uint32_t i = threadIdx.x; i < WARPS * HELD_PLUS_ONE; i += THREADS )
		s_warp_cnt[i / HELD_PLUS_ONE][i % HELD_PLUS_ONE] = 0u;
	__syncthreads();
	if ( row_invariant != 0u )
	{
		for ( uint32_t row = threadIdx.x; row < rows; row += THREADS )
		{
			uint64_t load = 0ull;
			for ( uint32_t k = 0u; k < top_k; ++k )
			{
				const uint32_t route = row * top_k + k, expert = route_expert[route];
				route_global[route] = expert;
				uint32_t chosen = RANKS;
				if ( expert < EXPERTS )
				{
					chosen = s_owner[expert];
					uint32_t chosen_load = (uint32_t)((load >> (4u * chosen)) & 0xfull);
					for ( uint32_t copy = 0u; copy < copies; ++copy )
					{
						const uint32_t cand = s_backup[copy * EXPERTS + expert];
						if ( cand < RANKS )
						{
							const uint32_t cand_load = (uint32_t)((load >> (4u * cand)) & 0xfull);
							if ( cand_load < chosen_load ) { chosen = cand; chosen_load = cand_load; }
						}
					}
					load += 1ull << (4u * chosen);
				}
				const uint32_t loc = chosen == rank && expert < EXPERTS && s_group[expert] < groups ? (uint32_t)s_group[expert] : groups;
				route_expert[route] = loc;
				if ( loc == groups ) route_weight[route] = 0.0f;
				atomicAdd(&s_count[loc], 1u);
			}
		}
	}
	else
	{
		for ( uint32_t base = 0u; base < routes && s_unseen != 0u; base += CHUNK )
		{
			const uint32_t count = routes - base < CHUNK ? routes - base : CHUNK;
			for ( uint32_t i = threadIdx.x; i < count; i += THREADS )
			{
				const uint32_t e = route_expert[base + i];
				s_local[i] = (uint16_t)(e < EXPERTS ? e : EXPERTS);
			}
			__syncthreads();
			if ( warp == 0u )
			{
				uint32_t unseen = s_unseen, pos = 0u;
				while ( pos < count && unseen != 0u )
				{
					const uint32_t idx = pos + lane;
					const uint32_t e_lane = idx < count ? (uint32_t)s_local[idx] : EXPERTS;
					const uint32_t r_lane = e_lane < EXPERTS ? (uint32_t)expert_rank[e_lane] : RANKS;
					const uint32_t miss_mask = __ballot_sync(0xffffffffu, e_lane < EXPERTS && r_lane >= RANKS);
					const uint32_t advance = miss_mask == 0u ? (count - pos < 32u ? count - pos : 32u) : (uint32_t)(__ffs((int)miss_mask) - 1);
					const uint32_t hit_r = lane < advance && e_lane < EXPERTS ? r_lane : RANKS;
					const uint32_t same = __match_any_sync(0xffffffffu, hit_r);
					if ( hit_r < RANKS && lane == (uint32_t)(__ffs((int)same) - 1) )
						s_rank_load[hit_r] += (uint32_t)__popc(same);
					__syncwarp(0xffffffffu);
					if ( miss_mask != 0u )
					{
						if ( lane == 0u )
						{
							const uint32_t expert = s_local[pos + advance];
							uint32_t chosen = s_owner[expert], chosen_load = s_rank_load[chosen];
							for ( uint32_t copy = 0u; copy < copies; ++copy )
							{
								const uint32_t cand = s_backup[copy * EXPERTS + expert];
								if ( cand < RANKS && s_rank_load[cand] < chosen_load ) { chosen = cand; chosen_load = s_rank_load[cand]; }
							}
							expert_rank[expert] = (uint8_t)chosen;
							s_rank_load[chosen] = chosen_load + 17u;
						}
						__syncwarp(0xffffffffu);
						--unseen;
						pos += advance + 1u;
					}
					else
						pos += advance;
				}
				if ( lane == 0u ) s_unseen = unseen;
			}
			__syncthreads();
		}
		for ( uint32_t base = 0u; base < routes; base += THREADS )
		{
			const uint32_t r = base + threadIdx.x;
			uint32_t loc = HELD_PLUS_ONE;
			if ( r < routes )
			{
				const uint32_t e = route_expert[r];
				route_global[r] = e;
				loc = e < EXPERTS && expert_rank[e] == rank && s_group[e] < groups ? (uint32_t)s_group[e] : groups;
				route_expert[r] = loc;
				if ( loc == groups ) route_weight[r] = 0.0f;
			}
			const uint32_t same = __match_any_sync(0xffffffffu, loc);
			if ( loc < HELD_PLUS_ONE && lane == (uint32_t)(__ffs((int)same) - 1) )
				atomicAdd(&s_count[loc], (uint32_t)__popc(same));
		}
	}
	__syncthreads();
	if constexpr ( HELD_PLUS_ONE <= 96u )
		D41RouteScan3Warp<THREADS, HELD_PLUS_ONE>(s_count, s_warp_tot, group_row_offset, tile_m, neuron_tiles_up, tile_prefix_up, neuron_tiles_down,
			tile_prefix_down);
	else
		LmRouteBuildPrefix<THREADS, HELD_PLUS_ONE>(s_count, group_row_offset, tile_m, neuron_tiles_up, tile_prefix_up, neuron_tiles_down, tile_prefix_down);
	__syncthreads();
	const uint32_t below = lane != 0u ? 0xffffffffu >> (32u - lane) : 0u;
	for ( uint32_t base = 0u; base < routes; base += THREADS )
	{
		const uint32_t route = base + threadIdx.x;
		const uint32_t e = route < routes ? route_expert[route] : HELD_PLUS_ONE;
		const float w = route < routes ? route_weight[route] : 0.0f;
		const uint32_t same = __match_any_sync(0xffffffffu, e);
		if ( e < HELD_PLUS_ONE && lane == (uint32_t)(__ffs((int)same) - 1) )
			s_warp_cnt[warp][e] = (uint16_t)__popc(same);
		__syncthreads();
		if ( route < routes )
		{
			uint32_t packed = s_count[e] + (uint32_t)__popc(same & below);
			#pragma unroll
			for ( uint32_t earlier = 0u; earlier < WARPS; ++earlier )
				if ( earlier < warp ) packed += s_warp_cnt[earlier][e];
			route_packed_row[route] = packed;
			route_source_token[packed] = route / top_k;
			packed_weight[packed] = w;
		}
		if ( base + THREADS < routes )
		{
			__syncthreads();
			for ( uint32_t g = threadIdx.x; g < HELD_PLUS_ONE; g += THREADS )
			{
				uint32_t sum = 0u;
				#pragma unroll
				for ( uint32_t wr = 0u; wr < WARPS; ++wr ) { sum += s_warp_cnt[wr][g]; s_warp_cnt[wr][g] = 0u; }
				s_count[g] += sum;
			}
			__syncthreads();
		}
	}
	LmDependentRelease();
}

template<uint32_t RANKS, uint32_t EXPERTS, uint32_t LOCAL, uint32_t THREADS, uint32_t HELD_PLUS_ONE, uint32_t TOP_K, uint32_t MAX_COPIES = 2u, bool DO_TOPK = true, uint32_t MAX_ROWS = 256u>
static __device__ __forceinline__ void D41TopkBalanceRouteBody(const float *__restrict__ router_logits, const float *__restrict__ router_bias, float route_scale,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ route_global,
	uint32_t rows, uint32_t rank, uint32_t copies, uint32_t row_invariant, const uint8_t *__restrict__ owner, const uint8_t *__restrict__ backup,
	const uint8_t *__restrict__ group, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row,
	uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up,
	uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down, float *__restrict__ packed_weight, const uint32_t *__restrict__ row_valid = 0)
{
	static_assert(HELD_PLUS_ONE <= 96u && THREADS == 256u && RANKS <= 16u, "balanced route groups fit three warps");
	constexpr uint32_t WARPS = THREADS / 32u, MAX_ROUTES = MAX_ROWS * TOP_K;
	__shared__ uint8_t s_owner[EXPERTS];
	__shared__ uint8_t s_backup[MAX_COPIES * EXPERTS];
	__shared__ uint8_t s_group[EXPERTS];
	__shared__ uint8_t expert_rank[EXPERTS];
	__shared__ uint16_t s_local[MAX_ROUTES];
	__shared__ float s_weight[MAX_ROUTES];
	__shared__ uint32_t s_count[HELD_PLUS_ONE];
	__shared__ uint32_t s_warp_tot[12];
	__shared__ uint16_t s_warp_cnt[WARPS][HELD_PLUS_ONE];
	__shared__ uint32_t s_rank_load[RANKS];
	const uint32_t routes = rows * TOP_K, groups = LOCAL + copies * LOCAL;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	for ( uint32_t e = threadIdx.x; e < EXPERTS; e += THREADS )
	{
		s_owner[e] = owner[e];
		s_group[e] = group[e];
		expert_rank[e] = 255u;
	}
	for ( uint32_t i = threadIdx.x; i < copies * EXPERTS; i += THREADS )
		s_backup[i] = backup[i];
	if ( threadIdx.x < RANKS ) s_rank_load[threadIdx.x] = 0u;
	if ( threadIdx.x < HELD_PLUS_ONE )
		s_count[threadIdx.x] = 0u;
	for ( uint32_t i = threadIdx.x; i < WARPS * HELD_PLUS_ONE; i += THREADS )
		s_warp_cnt[i / HELD_PLUS_ONE][i % HELD_PLUS_ONE] = 0u;
	if constexpr ( DO_TOPK )
	{
		for ( uint32_t row = warp; row < rows; row += WARPS )
		{
			if ( row_valid != 0 && row_valid[row] == 0u )
			{
				if ( lane < TOP_K )
				{
					route_expert[(uint64_t)row * TOP_K + lane] = 0xffffffffu;
					route_weight[(uint64_t)row * TOP_K + lane] = 0.0f;
				}
				continue;
			}
			LmTopkWarpRow<TOP_K, true, LM_TOPK_SCORE_SQRT_SOFTPLUS>(router_logits, EXPERTS, route_expert, route_weight, router_bias, (const uint16_t *)0,
				route_scale, fault, row, lane);
		}
		__syncthreads();
	}
	for ( uint32_t r = threadIdx.x; r < routes; r += THREADS )
	{
		const uint32_t e = route_expert[r];
		const uint32_t dead = (row_valid != 0 && row_valid[r / TOP_K] == 0u) || e >= EXPERTS ? 1u : 0u;
		s_local[r] = (uint16_t)(dead != 0u ? EXPERTS : e);
		route_global[r] = dead != 0u ? 0xffffffffu : e;
		s_weight[r] = dead != 0u ? 0.0f : route_weight[r];
	}
	__syncthreads();
	if ( row_invariant != 0u )
	{
		if ( threadIdx.x < rows )
		{
			const uint32_t row = threadIdx.x;
			uint64_t load = 0ull;
			#pragma unroll
			for ( uint32_t k = 0u; k < TOP_K; ++k )
			{
				const uint32_t route = row * TOP_K + k, expert = s_local[route];
				uint32_t chosen = RANKS;
				if ( expert < EXPERTS )
				{
					chosen = s_owner[expert];
					uint32_t chosen_load = (uint32_t)((load >> (4u * chosen)) & 0xfull);
					for ( uint32_t copy = 0u; copy < copies; ++copy )
					{
						const uint32_t cand = s_backup[copy * EXPERTS + expert];
						if ( cand < RANKS )
						{
							const uint32_t cand_load = (uint32_t)((load >> (4u * cand)) & 0xfull);
							if ( cand_load < chosen_load ) { chosen = cand; chosen_load = cand_load; }
						}
					}
					load += 1ull << (4u * chosen);
				}
				if ( chosen == rank && expert < EXPERTS && s_group[expert] < groups )
					s_local[route] = (uint16_t)s_group[expert];
				else
				{
					s_local[route] = (uint16_t)groups;
					s_weight[route] = 0.0f;
				}
			}
		}
	}
	else if ( routes <= 96u )
	{
		if ( threadIdx.x == 0u )
			for ( uint32_t route = 0u; route < routes; ++route )
			{
				const uint32_t expert = s_local[route];
				if ( expert >= EXPERTS ) continue;
				const uint32_t cached = expert_rank[expert];
				if ( cached < RANKS ) { s_rank_load[cached] += 1u; continue; }
				uint32_t chosen = s_owner[expert], chosen_load = s_rank_load[chosen];
				for ( uint32_t copy = 0u; copy < copies; ++copy )
				{
					const uint32_t cand = s_backup[copy * EXPERTS + expert];
					if ( cand < RANKS && s_rank_load[cand] < chosen_load ) { chosen = cand; chosen_load = s_rank_load[cand]; }
				}
				expert_rank[expert] = (uint8_t)chosen;
				s_rank_load[chosen] = chosen_load + 17u;
			}
	}
	else if ( warp == 0u )
	{
		uint32_t unseen = EXPERTS, pos = 0u;
		while ( pos < routes && unseen != 0u )
		{
			const uint32_t idx = pos + lane;
			const uint32_t e_lane = idx < routes ? (uint32_t)s_local[idx] : EXPERTS;
			const uint32_t r_lane = e_lane < EXPERTS ? (uint32_t)expert_rank[e_lane] : RANKS;
			const uint32_t miss_mask = __ballot_sync(0xffffffffu, e_lane < EXPERTS && r_lane >= RANKS);
			const uint32_t advance = miss_mask == 0u ? (routes - pos < 32u ? routes - pos : 32u) : (uint32_t)(__ffs((int)miss_mask) - 1);
			const uint32_t hit_r = lane < advance && e_lane < EXPERTS ? r_lane : RANKS;
			const uint32_t same = __match_any_sync(0xffffffffu, hit_r);
			if ( hit_r < RANKS && lane == (uint32_t)(__ffs((int)same) - 1) )
				s_rank_load[hit_r] += (uint32_t)__popc(same);
			__syncwarp(0xffffffffu);
			if ( miss_mask != 0u )
			{
				if ( lane == 0u )
				{
					const uint32_t expert = s_local[pos + advance];
					uint32_t chosen = s_owner[expert], chosen_load = s_rank_load[chosen];
					for ( uint32_t copy = 0u; copy < copies; ++copy )
					{
						const uint32_t cand = s_backup[copy * EXPERTS + expert];
						if ( cand < RANKS && s_rank_load[cand] < chosen_load ) { chosen = cand; chosen_load = s_rank_load[cand]; }
					}
					expert_rank[expert] = (uint8_t)chosen;
					s_rank_load[chosen] = chosen_load + 17u;
				}
				__syncwarp(0xffffffffu);
				--unseen;
				pos += advance + 1u;
			}
			else
				pos += advance;
		}
	}
	__syncthreads();
	if ( row_invariant == 0u )
	{
		for ( uint32_t r = threadIdx.x; r < routes; r += THREADS )
		{
			const uint32_t expert = s_local[r];
			if ( expert < EXPERTS && expert_rank[expert] == rank && s_group[expert] < groups )
				s_local[r] = (uint16_t)s_group[expert];
			else
			{
				s_local[r] = (uint16_t)groups;
				s_weight[r] = 0.0f;
			}
		}
		__syncthreads();
	}
	for ( uint32_t r = threadIdx.x; r < routes; r += THREADS )
	{
		route_expert[r] = (uint32_t)s_local[r];
		route_weight[r] = s_weight[r];
	}
	for ( uint32_t index = threadIdx.x - lane; index < routes; index += THREADS )
	{
		const uint32_t e = index + lane < routes ? (uint32_t)s_local[index + lane] : HELD_PLUS_ONE;
		const uint32_t same = __match_any_sync(0xffffffffu, e);
		if ( e < HELD_PLUS_ONE && lane == (uint32_t)(__ffs((int)same) - 1) )
			atomicAdd(&s_count[e], (uint32_t)__popc(same));
	}
	__syncthreads();
	D41RouteScan3Warp<THREADS, HELD_PLUS_ONE>(s_count, s_warp_tot, group_row_offset, tile_m, neuron_tiles_up, tile_prefix_up, neuron_tiles_down,
		tile_prefix_down);
	__syncthreads();
	const uint32_t below = lane != 0u ? 0xffffffffu >> (32u - lane) : 0u;
	for ( uint32_t base = 0u; base < routes; base += THREADS )
	{
		const uint32_t route = base + threadIdx.x;
		const uint32_t e = route < routes ? (uint32_t)s_local[route] : HELD_PLUS_ONE;
		const uint32_t same = __match_any_sync(0xffffffffu, e);
		if ( e < HELD_PLUS_ONE && lane == (uint32_t)(__ffs((int)same) - 1) )
			s_warp_cnt[warp][e] = (uint16_t)__popc(same);
		__syncthreads();
		if ( route < routes )
		{
			uint32_t packed = s_count[e] + (uint32_t)__popc(same & below);
			#pragma unroll
			for ( uint32_t earlier = 0u; earlier < WARPS; ++earlier )
				if ( earlier < warp ) packed += s_warp_cnt[earlier][e];
			route_packed_row[route] = packed;
			route_source_token[packed] = route / TOP_K;
			packed_weight[packed] = s_weight[route];
		}
		if ( base + THREADS < routes )
		{
			__syncthreads();
			if ( threadIdx.x < HELD_PLUS_ONE )
			{
				uint32_t sum = 0u;
				#pragma unroll
				for ( uint32_t w = 0u; w < WARPS; ++w ) { sum += s_warp_cnt[w][threadIdx.x]; s_warp_cnt[w][threadIdx.x] = 0u; }
				s_count[threadIdx.x] += sum;
			}
			__syncthreads();
		}
	}
}

template<uint32_t RANKS, uint32_t EXPERTS, uint32_t LOCAL, uint32_t THREADS, uint32_t HELD_PLUS_ONE, uint32_t TOP_K, uint32_t MAX_COPIES = 2u, bool DO_TOPK = true>
static __global__ __launch_bounds__(THREADS, 1)
void D41TopkBalanceRouteKernel(const float *__restrict__ router_logits, const float *__restrict__ router_bias, float route_scale,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight, uint32_t *__restrict__ route_global,
	uint32_t rows, uint32_t rank, uint32_t copies, uint32_t row_invariant, const uint8_t *__restrict__ owner, const uint8_t *__restrict__ backup,
	const uint8_t *__restrict__ group, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row,
	uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up,
	uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down, float *__restrict__ packed_weight)
{
	LmDependentWait();
	D41TopkBalanceRouteBody<RANKS, EXPERTS, LOCAL, THREADS, HELD_PLUS_ONE, TOP_K, MAX_COPIES, DO_TOPK, 256u>(router_logits, router_bias, route_scale,
		fault, route_expert, route_weight, route_global, rows, rank, copies, row_invariant, owner, backup, group, group_row_offset, route_packed_row,
		route_source_token, tile_m, neuron_tiles_up, tile_prefix_up, neuron_tiles_down, tile_prefix_down, packed_weight, (const uint32_t *)0);
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t HIDDEN>
static __device__ __forceinline__ void D41CollapseNormBlock(const uint16_t *__restrict__ streams, const float *__restrict__ collapse_pre,
	const uint16_t *__restrict__ norm, uint16_t *__restrict__ x, uint16_t *__restrict__ xq, uint8_t *__restrict__ mx_codes,
	uint8_t *__restrict__ mx_scales, uint32_t row, float rms_eps, uint16_t *s_row, float *s_red)
{
	constexpr uint32_t FLAT = D41K_HC * HIDDEN, OCTETS = HIDDEN / 8u;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const float p[D41K_HC] = { collapse_pre[(uint64_t)row * D41K_HC + 0u], collapse_pre[(uint64_t)row * D41K_HC + 1u],
		collapse_pre[(uint64_t)row * D41K_HC + 2u], collapse_pre[(uint64_t)row * D41K_HC + 3u] };
	const uint16_t *st = streams + (uint64_t)row * FLAT;
	for ( uint32_t q = threadIdx.x; q < OCTETS; q += THREADS )
	{
		float v[8];
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k ) v[k] = 0.0f;
		#pragma unroll
		for ( uint32_t copy = 0u; copy < D41K_HC; ++copy )
		{
			float s[8];
			LmLoadBf16x8AsFloat(__ldg((const uint4 *)(st + copy * HIDDEN) + q), s);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k ) v[k] += p[copy] * s[k];
		}
		((uint4 *)s_row)[q] = LmPackUint4FloatToBf16(v);
	}
	__syncthreads();
	if constexpr ( THREADS == 128u )
	{
		float t0 = 0.0f, t1 = 0.0f;
		for ( uint32_t off = 0u; off < HIDDEN; off += 256u )
		{
			const float a = LmBf16ToFloat(s_row[off + threadIdx.x]), b = LmBf16ToFloat(s_row[off + 128u + threadIdx.x]);
			t0 += a * a; t1 += b * b;
		}
		#pragma unroll
		for ( uint32_t off = 16u; off != 0u; off >>= 1u )
		{
			t0 += __shfl_down_sync(0xffffffffu, t0, off);
			t1 += __shfl_down_sync(0xffffffffu, t1, off);
		}
		if ( lane == 0u ) { s_red[warp] = t0; s_red[4u + warp] = t1; }
	}
	else
	{
		float acc = 0.0f;
		for ( uint32_t off = threadIdx.x; off < HIDDEN; off += 256u )
		{
			const float a = LmBf16ToFloat(s_row[off]);
			acc += a * a;
		}
		#pragma unroll
		for ( uint32_t off = 16u; off != 0u; off >>= 1u ) acc += __shfl_down_sync(0xffffffffu, acc, off);
		if ( lane == 0u ) s_red[warp] = acc;
	}
	__syncthreads();
	if ( threadIdx.x < 32u )
	{
		float s = lane < 8u ? s_red[lane] : 0.0f;
		#pragma unroll
		for ( uint32_t off = 4u; off != 0u; off >>= 1u ) s += __shfl_down_sync(0xffffffffu, s, off);
		if ( lane == 0u ) s_red[0] = s;
	}
	__syncthreads();
	const float rscale = SparkMathRmsScale(s_red[0], (float)HIDDEN, rms_eps);
	const uint64_t rbase = (uint64_t)row * HIDDEN;
	for ( uint32_t q = threadIdx.x; q < OCTETS; q += THREADS )
	{
		float v[8], w[8];
		LmLoadBf16x8AsFloat(((const uint4 *)s_row)[q], v);
		LmLoadBf16x8AsFloat(__ldg((const uint4 *)norm + q), w);
		#pragma unroll
		for ( uint32_t k = 0u; k < 8u; ++k ) v[k] = v[k] * rscale * w[k];
		const uint4 normed = LmPackUint4FloatToBf16(v);
		if ( x != 0 ) ((uint4 *)(x + rbase))[q] = normed;
		LmLoadBf16x8AsFloat(normed, v);
		if ( mx_codes != 0 )
		{
			uint2 packed; uint8_t sc;
			D41Fp8Quant8(v, &packed, &sc);
			((uint2 *)(mx_codes + rbase))[q] = packed;
			if ( (q & 3u) == 0u ) mx_scales[(rbase >> 5u) + (q >> 2u)] = sc;
		}
		else
			((uint4 *)(xq + rbase))[q] = D41Fp8Qdq8(v);
	}
}

template<uint32_t MAX_R, uint32_t UNROLL, uint32_t SPLITS, uint32_t HIDDEN>
static __global__ __launch_bounds__(256u / SPLITS)
void D41HcMixNormKernel(const uint16_t *__restrict__ streams, const float *__restrict__ fn, const float *__restrict__ scale,
	const float *__restrict__ base, float *__restrict__ partials, float *__restrict__ pre, float *__restrict__ post, float *__restrict__ comb,
	uint32_t *__restrict__ counters, const float *__restrict__ collapse_pre, const uint16_t *__restrict__ norm, uint16_t *__restrict__ x,
	uint16_t *__restrict__ xq, uint8_t *__restrict__ mx_codes, uint8_t *__restrict__ mx_scales, uint32_t rows, float rms_eps, uint32_t hc_iters,
	float hc_eps)
{
	using S = LmHcSinkhornShape<D41K_HC>;
	constexpr uint32_t THREADS = 256u / SPLITS, WARPS = THREADS / 32u, FLAT = D41K_HC * HIDDEN, STEPS = FLAT / 256u;
	constexpr uint32_t TASK_BLOCKS = SPLITS * S::kTasks;
	static_assert((SPLITS == 1u || SPLITS == 2u || SPLITS == 4u) && FLAT % 256u == 0u && STEPS % UNROLL == 0u && LM_HC_SINKHORN_WARPS == 8u, "hidden splits evenly across warps");
	union SharedStorage
	{
		float mix[WARPS][S::kTasks];
		struct { uint16_t row[HIDDEN]; float red[8]; } cn;
	};
	__shared__ __align__(16) SharedStorage smem;
	const uint32_t chunks = (rows + MAX_R - 1u) / MAX_R, sink_blocks = chunks * TASK_BLOCKS;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	if ( blockIdx.x < sink_blocks )
	{
		const uint32_t chunk = blockIdx.x / TASK_BLOCKS, local_id = blockIdx.x - chunk * TASK_BLOCKS;
		const uint32_t task = local_id / SPLITS;
		const uint32_t gwarp = (local_id % SPLITS) * WARPS + warp;
		const uint32_t gt = (gwarp << 5u) | lane, row0 = chunk * MAX_R, live = rows - row0 < MAX_R ? rows - row0 : MAX_R;
		float total[MAX_R];
		#pragma unroll
		for ( uint32_t r = 0u; r < MAX_R; ++r ) total[r] = 0.0f;
		if ( task < S::kMix )
		{
			const float *w = fn + (uint64_t)task * FLAT + gt;
			const uint16_t *s0 = streams + (uint64_t)row0 * FLAT + gt;
			for ( uint32_t step = 0u; step < STEPS; step += UNROLL )
			{
				float fac[UNROLL];
				uint16_t raw[UNROLL][MAX_R];
				#pragma unroll
				for ( uint32_t u = 0u; u < UNROLL; ++u )
				{
					const uint32_t off = (step + u) * 256u;
					fac[u] = __ldcs(w + off);
					#pragma unroll
					for ( uint32_t r = 0u; r < MAX_R; ++r )
						raw[u][r] = r < live ? __ldg(s0 + (uint64_t)r * FLAT + off) : 0u;
				}
				#pragma unroll
				for ( uint32_t u = 0u; u < UNROLL; ++u )
					#pragma unroll
					for ( uint32_t r = 0u; r < MAX_R; ++r )
						if ( r < live ) total[r] += fac[u] * LmBf16ToFloat(raw[u][r]);
			}
		}
		else
		{
			const uint16_t *s0 = streams + (uint64_t)row0 * FLAT + gt;
			for ( uint32_t step = 0u; step < STEPS; step += UNROLL )
			{
				uint16_t raw[UNROLL][MAX_R];
				#pragma unroll
				for ( uint32_t u = 0u; u < UNROLL; ++u )
				{
					const uint32_t off = (step + u) * 256u;
					#pragma unroll
					for ( uint32_t r = 0u; r < MAX_R; ++r )
						raw[u][r] = r < live ? __ldg(s0 + (uint64_t)r * FLAT + off) : 0u;
				}
				#pragma unroll
				for ( uint32_t u = 0u; u < UNROLL; ++u )
					#pragma unroll
					for ( uint32_t r = 0u; r < MAX_R; ++r )
						if ( r < live ) { const float v = LmBf16ToFloat(raw[u][r]); total[r] += v * v; }
			}
		}
		#pragma unroll
		for ( uint32_t r = 0u; r < MAX_R; ++r )
			if ( r < live )
			{
				float sum = total[r];
				#pragma unroll
				for ( uint32_t off = 16u; off != 0u; off >>= 1u ) sum += __shfl_down_sync(0xffffffffu, sum, off);
				if ( lane == 0u ) partials[((uint64_t)(row0 + r) * S::kTasks + task) * 8u + gwarp] = sum;
			}
		if ( LmFusedLastArrival(counters + chunk, TASK_BLOCKS) )
			for ( uint32_t r = warp; r < live; r += WARPS )
				LmHcSinkhornRow<D41K_HC>(partials, scale, base, pre, post, comb, HIDDEN, rms_eps, hc_iters, hc_eps, row0 + r, lane, smem.mix[warp]);
		LmDependentRelease();
		return;
	}
	const uint32_t row = blockIdx.x - sink_blocks;
	if ( row < rows )
		D41CollapseNormBlock<THREADS, HIDDEN>(streams, collapse_pre, norm, x, xq, mx_codes, mx_scales, row, rms_eps, smem.cn.row, smem.cn.red);
	LmDependentRelease();
}

template<uint32_t MAX_R, uint32_t HIDDEN>
static __global__ __launch_bounds__(256)
void D41HcMixNormStagedKernel(const uint16_t *__restrict__ streams, const float *__restrict__ fn, const float *__restrict__ scale,
	const float *__restrict__ base, float *__restrict__ partials, float *__restrict__ pre, float *__restrict__ post, float *__restrict__ comb,
	uint32_t *__restrict__ counters, const float *__restrict__ collapse_pre, const uint16_t *__restrict__ norm, uint16_t *__restrict__ x,
	uint16_t *__restrict__ xq, uint8_t *__restrict__ mx_codes, uint8_t *__restrict__ mx_scales, uint32_t rows, float rms_eps, uint32_t hc_iters,
	float hc_eps)
{
	using S = LmHcSinkhornShape<D41K_HC>;
	constexpr uint32_t FLAT = D41K_HC * HIDDEN, STEPS = FLAT / 256u, STAGES = 3u;
	constexpr uint32_t FN_VECS = S::kMix * 8u, STR_VECS = MAX_R * 4u, FN_BYTES = S::kMix * 32u * 4u;
	constexpr uint32_t STAGE_BYTES = FN_BYTES + MAX_R * 32u * 2u;
	static_assert(FLAT % 256u == 0u && HIDDEN % 256u == 0u && FN_VECS + STR_VECS <= 256u && LM_HC_SINKHORN_WARPS == 8u, "staged sinkhorn fits one async pass");
	union SharedStorage
	{
		uint8_t stage[STAGES * STAGE_BYTES];
		float mix[8][S::kTasks];
		struct { uint16_t row[HIDDEN]; float red[8]; } cn;
	};
	__shared__ __align__(16) SharedStorage smem;
	const uint32_t chunks = (rows + MAX_R - 1u) / MAX_R, sink_blocks = chunks * 8u;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	if ( blockIdx.x < sink_blocks )
	{
		const uint32_t chunk = blockIdx.x >> 3u, gwarp = blockIdx.x & 7u, row0 = chunk * MAX_R;
		const uint32_t live = rows - row0 < MAX_R ? rows - row0 : MAX_R;
		const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem.stage);
		float total[4][MAX_R];
		#pragma unroll
		for ( uint32_t t = 0u; t < 4u; ++t )
			#pragma unroll
			for ( uint32_t r = 0u; r < MAX_R; ++r )
				total[t][r] = 0.0f;
		auto produce = [&](uint32_t step)
		{
			const uint32_t dst = sbase + (step % STAGES) * STAGE_BYTES, elem_base = step * 256u + gwarp * 32u;
			if ( threadIdx.x < FN_VECS )
			{
				const uint32_t m = threadIdx.x >> 3u, v = threadIdx.x & 7u;
				asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst + threadIdx.x * 16u),
					"l"(fn + (uint64_t)m * FLAT + elem_base + v * 4u) : "memory");
			}
			else if ( threadIdx.x < FN_VECS + STR_VECS )
			{
				const uint32_t u = threadIdx.x - FN_VECS, r = u >> 2u, v = u & 3u, bytes = r < live ? 16u : 0u;
				const void *src = r < live ? (const void *)(streams + (uint64_t)(row0 + r) * FLAT + elem_base + v * 8u) : (const void *)streams;
				asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst + FN_BYTES + u * 16u), "l"(src), "r"(bytes) : "memory");
			}
		};
		for ( uint32_t p = 0u; p + 1u < STAGES; ++p ) { produce(p); asm volatile("cp.async.commit_group;\n" ::: "memory"); }
		for ( uint32_t step = 0u; step < STEPS; ++step )
		{
			asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2u) : "memory");
			__syncthreads();
			if ( step + STAGES - 1u < STEPS ) produce(step + STAGES - 1u);
			asm volatile("cp.async.commit_group;\n" ::: "memory");
			const uint8_t *slot = smem.stage + (step % STAGES) * STAGE_BYTES;
			const float *facs = (const float *)slot;
			const uint16_t *vals = (const uint16_t *)(slot + FN_BYTES);
			float v[MAX_R];
			#pragma unroll
			for ( uint32_t r = 0u; r < MAX_R; ++r ) v[r] = r < live ? LmBf16ToFloat(vals[r * 32u + lane]) : 0.0f;
			#pragma unroll
			for ( uint32_t t = 0u; t < 4u; ++t )
			{
				const uint32_t task = warp + t * 8u;
				if ( task < S::kMix )
				{
					const float f = facs[task * 32u + lane];
					#pragma unroll
					for ( uint32_t r = 0u; r < MAX_R; ++r ) total[t][r] += f * v[r];
				}
				else if ( task == S::kMix )
					#pragma unroll
					for ( uint32_t r = 0u; r < MAX_R; ++r ) total[t][r] += v[r] * v[r];
			}
		}
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		#pragma unroll
		for ( uint32_t t = 0u; t < 4u; ++t )
		{
			const uint32_t task = warp + t * 8u;
			if ( task < S::kTasks )
				#pragma unroll
				for ( uint32_t r = 0u; r < MAX_R; ++r )
					if ( r < live )
					{
						float sum = total[t][r];
						#pragma unroll
						for ( uint32_t off = 16u; off != 0u; off >>= 1u ) sum += __shfl_down_sync(0xffffffffu, sum, off);
						if ( lane == 0u ) partials[((uint64_t)(row0 + r) * S::kTasks + task) * 8u + gwarp] = sum;
					}
		}
		if ( LmFusedLastArrival(counters + chunk, 8u) )
			for ( uint32_t r = warp; r < live; r += 8u )
				LmHcSinkhornRow<D41K_HC>(partials, scale, base, pre, post, comb, HIDDEN, rms_eps, hc_iters, hc_eps, row0 + r, lane, smem.mix[warp]);
		LmDependentRelease();
		return;
	}
	const uint32_t row = blockIdx.x - sink_blocks;
	if ( row < rows )
		D41CollapseNormBlock<256u, HIDDEN>(streams, collapse_pre, norm, x, xq, mx_codes, mx_scales, row, rms_eps, smem.cn.row, smem.cn.red);
	LmDependentRelease();
}

template<uint32_t MAX_R, uint32_t HIDDEN>
static __global__ __launch_bounds__(256, 2)
void D41HcMixNormK32Kernel(const uint16_t *__restrict__ streams, const float *__restrict__ fn, const float *__restrict__ scale,
	const float *__restrict__ base, float *__restrict__ partials, float *__restrict__ pre, float *__restrict__ post, float *__restrict__ comb,
	uint32_t *__restrict__ counters, const float *__restrict__ collapse_pre, const uint16_t *__restrict__ norm, uint16_t *__restrict__ x,
	uint16_t *__restrict__ xq, uint8_t *__restrict__ mx_codes, uint8_t *__restrict__ mx_scales, uint32_t rows, float rms_eps, uint32_t hc_iters,
	float hc_eps)
{
	using S = LmHcSinkhornShape<D41K_HC>;
	constexpr uint32_t THREADS = 256u, WARPS = 8u, K_SPLITS = 32u, TASKS_PER_WARP = S::kMix / WARPS;
	constexpr uint32_t FLAT = D41K_HC * HIDDEN, SLICE_K = FLAT / K_SPLITS, ITERS = SLICE_K / 128u;
	constexpr uint32_t STR_VECS = MAX_R * 16u;
	static_assert(S::kMix == 24u && S::kTasks == 25u && FLAT % (K_SPLITS * 128u) == 0u && STR_VECS <= THREADS,
		"k32 sinkhorn splits hidden evenly into 128-wide chunks");
	union SharedStorage
	{
		uint16_t s_str[2][MAX_R][128];
		float mix[WARPS][S::kTasks];
		struct { uint16_t row[HIDDEN]; float red[8]; } cn;
	};
	__shared__ __align__(16) SharedStorage smem;
	const uint32_t chunks = (rows + MAX_R - 1u) / MAX_R, sink_blocks = chunks * K_SPLITS;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	if ( blockIdx.x < sink_blocks )
	{
		const uint32_t chunk = blockIdx.x >> 5u, ksplit = blockIdx.x & 31u, row0 = chunk * MAX_R;
		const uint32_t live = rows - row0 < MAX_R ? rows - row0 : MAX_R;
		const uint32_t k_base = ksplit * SLICE_K;
		const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem.s_str);
		constexpr uint32_t BUF_BYTES = MAX_R * 128u * 2u;
		const uint32_t cp_r = threadIdx.x >> 4u, cp_c = threadIdx.x & 15u;
		const uint32_t cp_active = threadIdx.x < STR_VECS && cp_r < live ? 1u : 0u;
		const uint16_t *cp_src = cp_active != 0u ? streams + (uint64_t)(row0 + cp_r) * FLAT + k_base + cp_c * 8u : (const uint16_t *)0;
		const uint32_t cp_dst_off = threadIdx.x * 16u;
		auto issue = [&](uint32_t iter)
		{
			if ( cp_active != 0u )
				asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sbase + (iter & 1u) * BUF_BYTES + cp_dst_off),
					"l"(cp_src + iter * 128u) : "memory");
			asm volatile("cp.async.commit_group;\n" ::: "memory");
		};
		float total[TASKS_PER_WARP][MAX_R], norm_sq[MAX_R];
		#pragma unroll
		for ( uint32_t r = 0u; r < MAX_R; ++r )
		{
			norm_sq[r] = 0.0f;
			#pragma unroll
			for ( uint32_t t = 0u; t < TASKS_PER_WARP; ++t ) total[t][r] = 0.0f;
		}
		issue(0u);
		for ( uint32_t iter = 0u; iter < ITERS; ++iter )
		{
			const uint32_t k_off = k_base + iter * 128u;
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
			__syncthreads();
			if ( iter + 1u < ITERS ) issue(iter + 1u);
			float4 f[TASKS_PER_WARP];
			#pragma unroll
			for ( uint32_t t = 0u; t < TASKS_PER_WARP; ++t )
				f[t] = __ldcs((const float4 *)(fn + (uint64_t)(warp + t * WARPS) * FLAT + k_off) + lane);
			const uint16_t (*cur)[128] = smem.s_str[iter & 1u];
			#pragma unroll
			for ( uint32_t r = 0u; r < MAX_R; ++r )
				if ( r < live )
				{
					const uint2 s = ((const uint2 *)cur[r])[lane];
					const float v0 = LmBf16ToFloat((uint16_t)(s.x & 0xffffu));
					const float v1 = LmBf16ToFloat((uint16_t)(s.x >> 16u));
					const float v2 = LmBf16ToFloat((uint16_t)(s.y & 0xffffu));
					const float v3 = LmBf16ToFloat((uint16_t)(s.y >> 16u));
					#pragma unroll
					for ( uint32_t t = 0u; t < TASKS_PER_WARP; ++t )
						total[t][r] = fmaf(f[t].x, v0, fmaf(f[t].y, v1, fmaf(f[t].z, v2, fmaf(f[t].w, v3, total[t][r]))));
					if ( warp == 0u )
						norm_sq[r] = fmaf(v0, v0, fmaf(v1, v1, fmaf(v2, v2, fmaf(v3, v3, norm_sq[r]))));
				}
		}
		#pragma unroll
		for ( uint32_t r = 0u; r < MAX_R; ++r )
			if ( r < live )
			{
				#pragma unroll
				for ( uint32_t t = 0u; t < TASKS_PER_WARP; ++t )
				{
					float sum = total[t][r];
					#pragma unroll
					for ( uint32_t off = 16u; off != 0u; off >>= 1u ) sum += __shfl_down_sync(0xffffffffu, sum, off);
					if ( lane == 0u ) partials[((uint64_t)(row0 + r) * S::kTasks + warp + t * WARPS) * K_SPLITS + ksplit] = sum;
				}
				if ( warp == 0u )
				{
					float ns = norm_sq[r];
					#pragma unroll
					for ( uint32_t off = 16u; off != 0u; off >>= 1u ) ns += __shfl_down_sync(0xffffffffu, ns, off);
					if ( lane == 0u ) partials[((uint64_t)(row0 + r) * S::kTasks + S::kMix) * K_SPLITS + ksplit] = ns;
				}
			}
		if ( LmFusedLastArrival(counters + chunk, K_SPLITS) )
		{
			const uint32_t sr = lane / D41K_HC, sc = lane % D41K_HC, cell = lane % (D41K_HC * D41K_HC);
			for ( uint32_t r = warp; r < live; r += WARPS )
			{
				const uint32_t row = row0 + r;
				if ( lane < S::kTasks )
				{
					const float *part = partials + ((uint64_t)row * S::kTasks + lane) * K_SPLITS;
					float val = 0.0f;
					#pragma unroll 4
					for ( uint32_t k = 0u; k < K_SPLITS; ++k ) val += part[k];
					smem.mix[warp][lane] = val;
				}
				__syncwarp();
				const float *m = smem.mix[warp];
				const float inverse = rsqrtf(m[S::kMix] / (float)FLAT + rms_eps);
				if ( lane < D41K_HC )
					pre[(uint64_t)row * D41K_HC + lane] = 1.0f / (1.0f + expf(-((m[lane] * inverse) * scale[0] + base[lane]))) + hc_eps;
				else if ( lane < 2u * D41K_HC )
					post[(uint64_t)row * D41K_HC + lane - D41K_HC] = 2.0f / (1.0f + expf(-((m[lane] * inverse) * scale[1] + base[lane])));
				float cv = (m[2u * D41K_HC + cell] * inverse) * scale[2] + base[2u * D41K_HC + cell];
				float top = -INFINITY;
				#pragma unroll
				for ( uint32_t idx = 0u; idx < D41K_HC; ++idx )
					top = fmaxf(top, __shfl_sync(0xffffffffu, cv, sr * D41K_HC + idx));
				cv = expf(cv - top);
				cv = cv / LmHcSinkhornLineSum<D41K_HC>(cv, sr * D41K_HC, 1u) + hc_eps;
				cv /= LmHcSinkhornLineSum<D41K_HC>(cv, sc, D41K_HC) + hc_eps;
				for ( uint32_t it = 1u; it < hc_iters; ++it )
				{
					cv /= LmHcSinkhornLineSum<D41K_HC>(cv, sr * D41K_HC, 1u) + hc_eps;
					cv /= LmHcSinkhornLineSum<D41K_HC>(cv, sc, D41K_HC) + hc_eps;
				}
				if ( lane < D41K_HC * D41K_HC )
					comb[(uint64_t)row * D41K_HC * D41K_HC + lane] = cv;
			}
		}
		LmDependentRelease();
		return;
	}
	const uint32_t row = blockIdx.x - sink_blocks;
	if ( row < rows )
		D41CollapseNormBlock<THREADS, HIDDEN>(streams, collapse_pre, norm, x, xq, mx_codes, mx_scales, row, rms_eps, smem.cn.row, smem.cn.red);
	LmDependentRelease();
}

template<uint32_t QLORA>
static __global__ __launch_bounds__(128)
void D41UnpackQCodesKernel(const uint8_t *__restrict__ received, uint8_t *__restrict__ codes, uint8_t *__restrict__ scales,
	uint32_t own, uint64_t segment)
{
	constexpr uint32_t CODE_VECS = QLORA / 16u, SCALE_WORDS = QLORA / 128u;
	static_assert(QLORA % 128u == 0u && CODE_VECS + SCALE_WORDS <= 128u, "q_coded unpack fits one block");
	const uint32_t row = blockIdx.y, source = row / own, local = row - source * own;
	const uint8_t *seg = received + (uint64_t)source * segment;
	LmDependentWait();
	if ( threadIdx.x < CODE_VECS )
		((uint4 *)(codes + (uint64_t)row * QLORA))[threadIdx.x] = ((const uint4 *)(seg + (uint64_t)local * QLORA))[threadIdx.x];
	else if ( threadIdx.x < CODE_VECS + SCALE_WORDS )
	{
		const uint32_t j = threadIdx.x - CODE_VECS;
		((uint32_t *)(scales + (uint64_t)row * (QLORA / 32u)))[j] =
			((const uint32_t *)(seg + (uint64_t)own * QLORA + (uint64_t)local * (QLORA / 32u)))[j];
	}
	LmDependentRelease();
}

template<uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS_M, uint32_t WARPS_N, uint32_t STAGES, bool OUT_F32>
static __global__ __launch_bounds__(WARPS_M * WARPS_N * 32u)
void D41MxGemmSmallKernel(const uint8_t *__restrict__ act, const uint8_t *__restrict__ act_scale,
	const uint8_t *__restrict__ w0, const uint8_t *__restrict__ ws0, void *__restrict__ out0, uint32_t dim0,
	const uint8_t *__restrict__ w1, const uint8_t *__restrict__ ws1, void *__restrict__ out1, uint32_t dim1,
	uint32_t rows, uint32_t input_dim, uint32_t out_stride0 = 0u, uint32_t out_col0 = 0u,
	const uint32_t *__restrict__ skip_seq = 0, const uint32_t *__restrict__ skip_len = 0,
	const uint32_t *__restrict__ skip_bound = 0, uint32_t skip_max_context = 0u)
{
	constexpr uint32_t THREADS = WARPS_M * WARPS_N * 32u, TILE_K = 128u, BLK = 32u;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M, WARP_N = TILE_N / WARPS_N, MF = WARP_M / 16u, NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * TILE_K, B_BYTES = TILE_N * TILE_K;
	constexpr uint32_t STAGE_BYTES = A_BYTES + B_BYTES + (TILE_M + TILE_N) * 4u;
	constexpr uint32_t A_PASSES = (TILE_M * (TILE_K / 16u)) / THREADS, B_PASSES = (TILE_N * (TILE_K / 16u)) / THREADS;
	static_assert(TILE_M * (TILE_K / 16u) % THREADS == 0u && TILE_N * (TILE_K / 16u) % THREADS == 0u && TILE_M + TILE_N <= THREADS,
		"small mx gemm tile divides evenly across threads");
	__shared__ __align__(128) uint8_t smem[STAGES * STAGE_BYTES];
	const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, group = lane >> 2u, quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N, warp_n = warp % WARPS_N;
	const uint32_t ntiles0 = (dim0 + TILE_N - 1u) / TILE_N, ntiles = ntiles0 + (dim1 != 0u ? (dim1 + TILE_N - 1u) / TILE_N : 0u);
	const uint32_t nt = blockIdx.x, row_base = blockIdx.y * TILE_M;
	if ( nt >= ntiles || row_base >= rows ) return;
	const uint32_t is_second = nt >= ntiles0 ? 1u : 0u;
	const uint8_t *w = is_second != 0u ? w1 : w0;
	const uint8_t *ws = is_second != 0u ? ws1 : ws0;
	void *out_ptr = is_second != 0u ? out1 : out0;
	const uint32_t out_dim = is_second != 0u ? dim1 : dim0;
	const uint32_t out_stride = is_second == 0u && out_stride0 != 0u ? out_stride0 : out_dim;
	const uint32_t out_col = is_second == 0u ? out_col0 : 0u;
	const uint32_t n_base = (is_second != 0u ? nt - ntiles0 : nt) * TILE_N;
	const uint32_t k_tiles = input_dim / TILE_K, sc_cols = input_dim / BLK;
	LmDependentWait();
	if ( skip_max_context != 0u && skip_seq != 0 && skip_len != 0 && (is_second != 0u || dim1 == 0u) )
	{
		const uint32_t row_end = row_base + TILE_M < rows ? row_base + TILE_M : rows;
		uint32_t any_sparse = 0u;
		for ( uint32_t r = row_base; r < row_end; ++r )
		{
			uint32_t b = __ldg(skip_len + __ldg(skip_seq + r));
			if ( skip_bound != 0 )
			{
				const uint32_t rp = __ldg(skip_bound + r);
				b = rp == D41K_NO_ROW ? 0u : (rp + 1u < b ? rp + 1u : b);
			}
			if ( b > skip_max_context ) { any_sparse = 1u; break; }
		}
		if ( any_sparse == 0u ) { LmDependentRelease(); return; }
	}
	auto swz = [](uint32_t r, uint32_t c) { return r * TILE_K + ((c ^ (r & 7u)) << 4u); };
	auto issue = [&](uint32_t k)
	{
		const uint32_t st = sbase + (k % STAGES) * STAGE_BYTES;
		#pragma unroll
		for ( uint32_t q = 0u; q < A_PASSES; ++q )
		{
			const uint32_t ch = threadIdx.x + q * THREADS, r = ch >> 3u, live = row_base + r < rows ? 16u : 0u;
			const void *src = act + (uint64_t)(live != 0u ? row_base + r : 0u) * input_dim + (uint64_t)k * TILE_K + (ch & 7u) * 16u;
			asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(st + swz(r, ch & 7u)), "l"(src), "r"(live) : "memory");
		}
		#pragma unroll
		for ( uint32_t q = 0u; q < B_PASSES; ++q )
		{
			const uint32_t ch = threadIdx.x + q * THREADS, r = ch >> 3u, live = n_base + r < out_dim ? 16u : 0u;
			const void *src = w + (uint64_t)(live != 0u ? n_base + r : 0u) * input_dim + (uint64_t)k * TILE_K + (ch & 7u) * 16u;
			asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(st + A_BYTES + swz(r, ch & 7u)), "l"(src), "r"(live) : "memory");
		}
		if ( threadIdx.x < TILE_M )
		{
			const uint32_t sr = row_base + threadIdx.x < rows ? row_base + threadIdx.x : rows - 1u;
			asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(st + A_BYTES + B_BYTES + threadIdx.x * 4u),
				"l"(act_scale + (uint64_t)sr * sc_cols + (uint64_t)k * 4u) : "memory");
		}
		else if ( threadIdx.x < TILE_M + TILE_N )
		{
			const uint32_t nr = n_base + (threadIdx.x - TILE_M) < out_dim ? n_base + (threadIdx.x - TILE_M) : out_dim - 1u;
			asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(st + A_BYTES + B_BYTES + threadIdx.x * 4u),
				"l"(ws + (uint64_t)nr * sc_cols + (uint64_t)k * 4u) : "memory");
		}
	};
	float total[MF][NF][4];
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			total[i][j][0] = total[i][j][1] = total[i][j][2] = total[i][j][3] = 0.0f;
	for ( uint32_t p = 0u; p + 1u < STAGES; ++p )
	{
		if ( p < k_tiles ) issue(p);
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	}
	for ( uint32_t k = 0u; k < k_tiles; ++k )
	{
		asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2u) : "memory");
		__syncthreads();
		if ( k + STAGES - 1u < k_tiles ) issue(k + STAGES - 1u);
		asm volatile("cp.async.commit_group;\n" ::: "memory");
		const uint32_t sa = sbase + (k % STAGES) * STAGE_BYTES, sb = sa + A_BYTES;
		const uint32_t *words = (const uint32_t *)(smem + (k % STAGES) * STAGE_BYTES + A_BYTES + B_BYTES);
		uint32_t scale_a[MF], scale_b[NF];
		#pragma unroll
		for ( uint32_t i = 0u; i < MF; ++i ) scale_a[i] = words[warp_m * WARP_M + i * 16u + group + ((lane & 1u) << 3u)];
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j ) scale_b[j] = words[TILE_M + warp_n * WARP_N + j * 8u + group];
		#pragma unroll
		for ( uint32_t step = 0u; step < 4u; ++step )
		{
			uint32_t a[MF][4], b[NF][2];
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
				asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
					: "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3])
					: "r"(sa + swz(warp_m * WARP_M + i * 16u + ((lane >> 3u) & 1u) * 8u + (lane & 7u), step * 2u + (lane >> 4u))));
			#pragma unroll
			for ( uint32_t j = 0u; j < NF; j += 2u )
			{
				uint32_t v0, v1, v2, v3;
				asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
					: "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
					: "r"(sb + swz(warp_n * WARP_N + j * 8u + (lane >> 4u) * 8u + (lane & 7u), step * 2u + ((lane >> 3u) & 1u))));
				b[j][0] = v0; b[j][1] = v1; b[j + 1u][0] = v2; b[j + 1u][1] = v3;
			}
			#pragma unroll
			for ( uint32_t i = 0u; i < MF; ++i )
				#pragma unroll
				for ( uint32_t j = 0u; j < NF; ++j )
				{
					float block[4];
					LmMmaMxf8Block(block, a[i], b[j], scale_a[i], (uint16_t)step, scale_b[j], (uint16_t)step);
					total[i][j][0] += block[0]; total[i][j][1] += block[1];
					total[i][j][2] += block[2]; total[i][j][3] += block[3];
				}
		}
	}
	asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	#pragma unroll
	for ( uint32_t i = 0u; i < MF; ++i )
		#pragma unroll
		for ( uint32_t j = 0u; j < NF; ++j )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; e += 2u )
			{
				const uint32_t r = row_base + warp_m * WARP_M + i * 16u + group + (e >> 1u) * 8u;
				const uint32_t col = n_base + warp_n * WARP_N + j * 8u + quad * 2u;
				if ( r < rows && col < out_dim )
				{
					if constexpr ( OUT_F32 )
						*(float2 *)((float *)out_ptr + (uint64_t)r * out_stride + out_col + col) = make_float2(total[i][j][e], total[i][j][e + 1u]);
					else
						*(uint32_t *)((uint16_t *)out_ptr + (uint64_t)r * out_stride + out_col + col) =
							(uint32_t)LmFloatToBf16(total[i][j][e]) | ((uint32_t)LmFloatToBf16(total[i][j][e + 1u]) << 16u);
				}
			}
	LmDependentRelease();
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
	if ( (scale_bytes & 15u) == 0u )
		for ( uint32_t i = threadIdx.x; i < scale_bytes / 16u; i += blockDim.x )
			((uint4 *)(record + hidden))[i] = ((const uint4 *)(scales + (uint64_t)row * scale_bytes))[i];
	else
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
				LmAddBf16x8(total, ((const uint4 *)(packed + (uint64_t)route_packed_row[route] * width))[vector]);
		}
		out[vector] = LmPackUint4FloatToBf16(total);
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
	if ( (scale_bytes & 15u) == 0u )
		for ( uint32_t i = threadIdx.x; i < scale_bytes / 16u; i += blockDim.x )
			((uint4 *)(record + hidden))[i] = ((const uint4 *)(scales + (uint64_t)row * scale_bytes))[i];
	else
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
	if ( (scale_bytes & 15u) == 0u )
		for ( uint32_t i = lane; i < scale_bytes / 16u; i += lanes )
			((uint4 *)(scales + (uint64_t)row * scale_bytes))[i] = ((const uint4 *)(record + hidden))[i];
	else
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
				LmAddBf16x8(total, ((const uint4 *)(landing + (uint64_t)at * hidden))[vector]);
		}
		out[vector] = LmPackUint4FloatToBf16(total);
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
		float sum[8];
		LmLoadBf16x8AsFloat(((const uint4 *)(back + ((uint64_t)target * shuffle + index) * hidden))[vector], sum);
		LmAddBf16x8(sum, ((const uint4 *)(shared + (uint64_t)row * hidden))[vector]);
		((uint4 *)(output + (uint64_t)row * hidden))[vector] = LmPackUint4FloatToBf16(sum);
	}
}

template <uint32_t RANKS>
static __global__ void D41DirectFinishKernel(const uint16_t *__restrict__ landing, const uint32_t *__restrict__ slot,
	const uint16_t *__restrict__ shared, uint32_t live, uint32_t hidden, uint16_t *__restrict__ output)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t vector = blockIdx.x * blockDim.x + threadIdx.x; vector * 8u < hidden; vector += gridDim.x * blockDim.x )
	{
		if ( row >= live ) { ((uint4 *)(output + (uint64_t)row * hidden))[vector] = make_uint4(0u, 0u, 0u, 0u); continue; }
		float total[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		for ( uint32_t rank = 0u; rank < RANKS; ++rank )
		{
			const uint32_t at = slot[(uint64_t)row * RANKS + rank];
			if ( at != UINT32_MAX ) LmAddBf16x8(total, ((const uint4 *)(landing + (uint64_t)at * hidden))[vector]);
		}
		LmLoadBf16x8AsFloat(LmPackUint4FloatToBf16(total), total);
		LmAddBf16x8(total, ((const uint4 *)(shared + (uint64_t)row * hidden))[vector]);
		((uint4 *)(output + (uint64_t)row * hidden))[vector] = LmPackUint4FloatToBf16(total);
	}
}

static __global__ void D41WindowAssembleKernel(const uint16_t *__restrict__ own_kv, const uint16_t *__restrict__ edges, uint16_t *__restrict__ window_kv,
	uint32_t own, uint32_t edge, uint32_t first)
{
	const uint32_t block = blockIdx.x;
	const uint32_t rank = block < own ? 0u : (block - own) / edge, index = block < own ? block : (block - own) - rank * edge;
	const uint4 *source = (const uint4 *)(block < own ? own_kv + (uint64_t)block * D41K_LATENT : edges + (uint64_t)(block - own) * D41K_LATENT);
	uint4 *target = (uint4 *)(window_kv + (uint64_t)(block < own ? first + block : rank * own + own - edge + index) * D41K_LATENT);
	LmDependentWait();
	LmDependentRelease();
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

static __global__ void D41FinalizeSliceKernel(const uint16_t *__restrict__ slice_out, const uint16_t *__restrict__ received,
	const uint32_t *__restrict__ route_expert, const uint32_t *__restrict__ route_packed_row, uint16_t *__restrict__ output, uint32_t top_k,
	uint32_t local_experts, uint32_t degree, uint32_t hidden_slice, uint64_t source_stride, uint64_t shared_offset,
	const uint16_t *__restrict__ shared_partial)
{
	__shared__ uint32_t s_src[16];
	__shared__ uint32_t s_pr[16];
	const uint32_t row = blockIdx.y, total_experts = local_experts * degree;
	LmDependentWait();
	LmDependentRelease();
	if ( top_k <= 16u )
	{
		if ( threadIdx.x < top_k )
		{
			const uint32_t route = row * top_k + threadIdx.x, e = route_expert[route];
			s_src[threadIdx.x] = e < total_experts ? e / local_experts : degree;
			s_pr[threadIdx.x] = route_packed_row[route];
		}
		__syncthreads();
	}
	const bool aligned = (hidden_slice & 7u) == 0u && LmIsAligned16(slice_out) && LmIsAligned16(output) &&
		(shared_partial != 0 ? LmIsAligned16(shared_partial) : (LmIsAligned16(received) && (source_stride & 7u) == 0u && (shared_offset & 7u) == 0u));
	if ( aligned )
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
					const uint32_t rsrc = top_k <= 16u ? s_src[k] : (route_expert[row * top_k + k] < total_experts ? route_expert[row * top_k + k] / local_experts : degree);
					if ( rsrc == source )
					{
						const uint32_t pr = top_k <= 16u ? s_pr[k] : route_packed_row[row * top_k + k];
						LmAddBf16x8(total, ((const uint4 *)(slice_out + (uint64_t)pr * hidden_slice))[q]);
					}
				}
				const uint4 sh = shared_partial != 0 ?
					((const uint4 *)(shared_partial + ((uint64_t)source * gridDim.y + row) * hidden_slice))[q] :
					((const uint4 *)(received + (uint64_t)source * source_stride + shared_offset + (uint64_t)row * hidden_slice))[q];
				LmAddBf16x8(total, sh);
				for ( uint32_t i = 0u; i < 8u; ++i )
					sum[i] += LmBf16ToFloat(LmFloatToBf16(total[i]));
			}
			out_q[q] = LmPackUint4FloatToBf16(sum);
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
				const uint32_t route = row * top_k + k, e = route_expert[route];
				if ( e < total_experts && e / local_experts == source )
					total += LmBf16ToFloat(slice_out[(uint64_t)route_packed_row[route] * hidden_slice + element]);
			}
			total += LmBf16ToFloat(shared_partial != 0 ? shared_partial[((uint64_t)source * gridDim.y + row) * hidden_slice + element] :
				received[(uint64_t)source * source_stride + shared_offset + (uint64_t)row * hidden_slice + element]);
			sum += LmBf16ToFloat(LmFloatToBf16(total));
		}
		output[(uint64_t)row * hidden_slice + element] = LmFloatToBf16(sum);
	}
}

static __global__ void D41SliceGatherKernel(const uint16_t *__restrict__ gathered, uint16_t *__restrict__ output, uint32_t rows, uint32_t hidden_slice,
	uint32_t degree)
{
	const uint32_t row = blockIdx.y, element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < hidden_slice * degree )
		output[(uint64_t)row * hidden_slice * degree + element] =
			gathered[((uint64_t)(element / hidden_slice) * rows + row) * hidden_slice + element % hidden_slice];
}

static __global__ void D41ExpertActKernel(const uint16_t *__restrict__ gate_up, const float *__restrict__ packed_weight,
	const uint32_t *__restrict__ group_row_offset, uint32_t local_experts, uint8_t *__restrict__ codes, uint16_t *__restrict__ act, uint32_t routes,
	uint32_t width, float limit)
{
	const uint32_t q = blockIdx.x * blockDim.x + threadIdx.x, octets = width >> 3u, s_blocks = width / SPARK_DSV41_FLASH_FP8_BLOCK;
	LmDependentWait();
	const uint32_t local = group_row_offset[local_experts], count = local < routes ? local : routes;
	if ( q < octets )
		for ( uint32_t route = blockIdx.y; route < count; route += gridDim.y )
		{
			const uint64_t g_base = (uint64_t)route * 2u * octets + q;
			float v[8];
			D41Swiglu8(((const uint4 *)gate_up)[g_base], ((const uint4 *)gate_up)[g_base + octets], packed_weight[route], limit, v);
			uint2 packed;
			uint8_t sc;
			D41Fp8Quant8(v, &packed, &sc);
			if ( act != 0 )
			{
				((uint4 *)act)[(uint64_t)route * octets + q] = D41DequantE4m3x8(packed, __uint_as_float((uint32_t)sc << 23u));
				continue;
			}
			((uint2 *)codes)[(uint64_t)route * octets + q] = packed;
			if ( (q & 3u) == 0u ) codes[(uint64_t)routes * width + (uint64_t)route * s_blocks + (q >> 2u)] = sc;
		}
	LmDependentRelease();
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
	__shared__ uint32_t active_rows[16];
	__shared__ uint32_t active_count;
	const uint32_t row = blockIdx.y;
	LmDependentWait();
	if ( threadIdx.x == 0u )
	{
		uint32_t n = 0u;
		for ( uint32_t k = 0u; k < top_k && n < 16u; ++k )
		{
			const uint32_t route = row * top_k + k;
			if ( route_expert[route] < local_experts )
				active_rows[n++] = route_packed_row[route];
		}
		active_count = n;
	}
	__syncthreads();
	const uint32_t count = active_count;
	if ( top_k <= 16u && (width & 7u) == 0u && LmIsAligned16(packed) && LmIsAligned16(shared) && LmIsAligned16(output) )
	{
		const uint32_t octets = width >> 3u;
		for ( uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < octets; q += gridDim.x * blockDim.x )
		{
			const uint32_t offset = q << 3u;
			float total[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, v[8], sh[8];
			for ( uint32_t i = 0u; i < count; ++i )
			{
				LmLoadBf16x8AsFloat(__ldg((const uint4 *)(packed + (uint64_t)active_rows[i] * width + offset)), v);
				#pragma unroll
				for ( uint32_t j = 0u; j < 8u; ++j ) total[j] += v[j];
			}
			LmLoadFloat8(shared + (uint64_t)row * width + offset, sh);
			#pragma unroll
			for ( uint32_t j = 0u; j < 8u; ++j ) total[j] += sh[j];
			LmStoreFloat8(output + (uint64_t)row * width + offset, total);
		}
		LmDependentRelease();
		return;
	}
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
	LmDependentRelease();
}

static __global__ void D41SumSourcesF32Kernel(const float *__restrict__ gathered, uint16_t *__restrict__ output, uint64_t elements, uint32_t degree)
{
	LmDependentWait();
	if ( (elements & 7u) == 0u && LmIsAligned16(gathered) && LmIsAligned16(output) )
	{
		for ( uint64_t q = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; q < (elements >> 3u); q += (uint64_t)gridDim.x * blockDim.x )
		{
			float s[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, v[8];
			for ( uint32_t source = 0u; source < degree; ++source )
			{
				LmLoadFloat8(gathered + (uint64_t)source * elements + (q << 3u), v);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k ) s[k] += v[k];
			}
			((uint4 *)output)[q] = LmPackUint4FloatToBf16(s);
		}
		LmDependentRelease();
		return;
	}
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
	{
		float sum = 0.0f;
		for ( uint32_t source = 0u; source < degree; ++source ) sum += gathered[(uint64_t)source * elements + index];
		output[index] = LmFloatToBf16(sum);
	}
	LmDependentRelease();
}

static __global__ void D41SumGatheredKernel(const uint16_t *__restrict__ gathered, uint16_t *__restrict__ output, uint64_t elements, uint32_t degree)
{
	LmDependentWait();
	LmDependentRelease();
	if ( (elements & 7u) == 0u && LmIsAligned16(gathered) && LmIsAligned16(output) )
	{
		const uint64_t quads = elements >> 3u;
		for ( uint64_t q = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; q < quads; q += (uint64_t)gridDim.x * blockDim.x )
		{
			float sum[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
			for ( uint32_t source = 0u; source < degree; ++source ) LmAddBf16x8(sum, ((const uint4 *)gathered)[(uint64_t)source * quads + q]);
			((uint4 *)output)[q] = LmPackUint4FloatToBf16(sum);
		}
		return;
	}
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * blockDim.x )
	{
		float sum = 0.0f;
		for ( uint32_t source = 0u; source < degree; ++source ) sum += LmBf16ToFloat(gathered[(uint64_t)source * elements + index]);
		output[index] = LmFloatToBf16(sum);
	}
}

static __device__ __forceinline__ void D41PackFp4Block8(const float v[8], uint32_t *out_nibbles, uint8_t *out_scale)
{
	float amax = 0.0f;
	const unsigned mask = __activemask();
	#pragma unroll
	for ( uint32_t k = 0u; k < 8u; ++k ) amax = fmaxf(amax, fabsf(v[k]));
	amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 1u)); amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 2u));
	const uint32_t sbits = __float_as_uint(SparkMathFp4Pow2QdqScale(amax));
	const float inv = __uint_as_float(0x7f000000u - sbits);
	uint32_t packed = 0u;
	#pragma unroll
	for ( uint32_t k = 0u; k < 4u; ++k )
		packed |= (uint32_t)LmFloatPairToE2m1(fminf(fmaxf(v[2u * k] * inv, -6.0f), 6.0f), fminf(fmaxf(v[2u * k + 1u] * inv, -6.0f), 6.0f)) << (8u * k);
	*out_nibbles = packed; *out_scale = (uint8_t)(sbits >> 23u);
}

static __global__ void D41PackIndexQueriesMxfp4Kernel(const uint16_t *__restrict__ src, uint32_t *__restrict__ codes, uint8_t *__restrict__ scales, uint64_t octets)
{
	const uint64_t q = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	LmDependentWait();
	LmDependentRelease();
	if ( q >= octets ) return;
	float v[8]; uint32_t packed; uint8_t sc;
	LmLoadBf16x8AsFloat(((const uint4 *)src)[q], v);
	D41PackFp4Block8(v, &packed, &sc);
	codes[q] = packed;
	if ( (q & 3u) == 0u ) scales[q >> 2u] = sc;
}

template<class Geometry>
static __global__ void D41PackIndexKeysMxfp4Kernel(LmKvShardView view, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ length,
	uint32_t stride, uint32_t *__restrict__ out_codes, uint32_t *__restrict__ out_scales)
{
	const uint32_t key = (blockIdx.x * blockDim.x + threadIdx.x) >> 4u, lane16 = threadIdx.x & 15u;
	LmDependentWait();
	LmDependentRelease();
	if ( key >= stride ) return;
	const uint32_t seq = sequence_of_row[0], limit = SparkKvShardLocalKeys(view.shard, length[seq]);
	if ( key >= limit ) return;
	const uint8_t *slot = LmKvShardSlotRequired<Geometry>(view, seq, SparkKvShardLocalPosition(view.shard, key), 0u, LM_KV_ACCESS_READ);
	float v[8] = {}; uint32_t packed; uint8_t sc;
	if ( slot != 0 ) LmLoadBf16x8AsFloat(((const uint4 *)slot)[lane16], v);
	D41PackFp4Block8(v, &packed, &sc);
	out_codes[(uint64_t)key * 16u + lane16] = packed;
	const unsigned mask = __activemask();
	uint32_t sw = (uint32_t)sc | (__shfl_down_sync(mask, (uint32_t)sc, 4u) << 8u) | (__shfl_down_sync(mask, (uint32_t)sc, 8u) << 16u) | (__shfl_down_sync(mask, (uint32_t)sc, 12u) << 24u);
	if ( lane16 == 0u ) out_scales[key] = sw;
}

template<bool ROUND_BF16>
static __global__ __launch_bounds__(256)
void D41IndexScoreMxfp4Kernel(const uint32_t *__restrict__ q_codes, const uint32_t *__restrict__ q_scales, const uint16_t *__restrict__ head_weight,
	const uint32_t *__restrict__ k_codes, const uint32_t *__restrict__ k_scales, SparkKvShard shard, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ length, const uint32_t *__restrict__ row_bound, const uint8_t *__restrict__ block_mask, uint32_t block_stride,
	uint32_t rows, uint32_t score_stride, float *__restrict__ scores, uint32_t dense_limit = 0u)
{
	constexpr uint32_t STAGE_KEYS = 64u, PITCH_U32 = 20u;
	__shared__ __align__(16) uint32_t s_codes[2][STAGE_KEYS * PITCH_U32];
	__shared__ uint32_t s_scales[2][STAGE_KEYS];
	const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u, g = lane >> 2u, c = lane & 3u;
	const uint32_t block_first = blockIdx.x * (STAGE_KEYS * 4u), row = blockIdx.y * 8u + warp, s_base = (uint32_t)__cvta_generic_to_shared(s_codes);
	LmDependentWait();
	const uint32_t ctx_len = length[sequence_of_row[0]], ctx_keys = SparkKvShardLocalKeys(shard, ctx_len);
	if ( dense_limit != 0u && (ctx_len <= dense_limit || block_first >= ctx_keys) ) { LmDependentRelease(); return; }
	uint32_t a[2][2][4], sa[2], rbound = 0u; float w[4];
	if ( row < rows ) {
		uint32_t rb = ctx_len;
		if ( row_bound != 0 ) { const uint32_t b = row_bound[row]; rb = b == 0xffffffffu ? 0u : (b + 1u < rb ? b + 1u : rb); }
		rbound = SparkKvShardLocalKeys(shard, rb);
		const uint32_t *qc = q_codes + (uint64_t)row * 512u, *qs = q_scales + (uint64_t)row * 32u, srow = (lane & 1u) * 8u + g;
		const uint16_t *hw = head_weight + (uint64_t)row * 32u;
		#pragma unroll
		for ( uint32_t ht = 0u; ht < 2u; ++ht ) {
			const uint32_t h0 = ht * 16u + g, h1 = h0 + 8u;
			a[ht][0][0] = qc[h0 * 16u + c]; a[ht][0][1] = qc[h1 * 16u + c]; a[ht][0][2] = qc[h0 * 16u + 4u + c]; a[ht][0][3] = qc[h1 * 16u + 4u + c];
			a[ht][1][0] = qc[h0 * 16u + 8u + c]; a[ht][1][1] = qc[h1 * 16u + 8u + c]; a[ht][1][2] = qc[h0 * 16u + 12u + c]; a[ht][1][3] = qc[h1 * 16u + 12u + c];
			sa[ht] = qs[ht * 16u + srow]; w[2u * ht] = LmBf16ToFloat(hw[h0]); w[2u * ht + 1u] = LmBf16ToFloat(hw[h1]);
		}
	}
	const uint32_t lk = threadIdx.x >> 2u, lq = threadIdx.x & 3u, l_off = (lk * PITCH_U32 + lq * 4u) * 4u, mat_off = ((lane & 7u) * PITCH_U32 + (lane >> 3u) * 4u) * 4u;
	auto produce = [&](uint32_t ch, uint32_t st) {
		const uint32_t kf = block_first + ch * STAGE_KEYS, gk = kf + lk, live = (gk < ctx_keys && gk < score_stride) ? 16u : 0u;
		const uint32_t dst = s_base + st * STAGE_KEYS * PITCH_U32 * 4u + l_off;
		const void *src = live != 0u ? (const void *)(k_codes + (uint64_t)gk * 16u + lq * 4u) : (const void *)k_codes;
		asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(live) : "memory");
		if ( threadIdx.x < STAGE_KEYS ) s_scales[st][threadIdx.x] = (kf + threadIdx.x < ctx_keys && kf + threadIdx.x < score_stride) ? k_scales[kf + threadIdx.x] : 0x7f7f7f7fu;
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	produce(0u, 0u);
	for ( uint32_t ch = 0u; ch < 4u; ++ch ) {
		const uint32_t st = ch & 1u;
		if ( ch < 3u ) produce(ch + 1u, st ^ 1u);
		asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		if ( ch == 3u ) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		const uint32_t cf = block_first + ch * STAGE_KEYS, st_base = s_base + st * STAGE_KEYS * PITCH_U32 * 4u + mat_off;
		#pragma unroll
		for ( uint32_t t = 0u; t < STAGE_KEYS / 8u; ++t ) {
			const uint32_t first = cf + t * 8u, blk = first >> 3u, written = first + c * 2u;
			if ( first >= score_stride ) break;
			if ( row >= rows ) continue;
			if ( first >= rbound || (block_mask != 0 && (blk >= block_stride || block_mask[(uint64_t)row * block_stride + blk] == 0u)) ) {
				if ( g == 0u && written + 1u < score_stride ) *(float2 *)(scores + (uint64_t)row * score_stride + written) = make_float2(-INFINITY, -INFINITY);
				continue;
			}
			uint32_t bv[4];
			asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(bv[0]), "=r"(bv[1]), "=r"(bv[2]), "=r"(bv[3]) : "r"(st_base + t * 8u * PITCH_U32 * 4u));
			const uint32_t b0[2] = { bv[0], bv[1] }, b1[2] = { bv[2], bv[3] }, sb = s_scales[st][t * 8u + g];
			float acc0[4] = {}, acc1[4] = {}, low, high;
			LmMmaMxfp4(acc0, a[0][0], b0, sa[0] & 0xffffu, sb & 0xffffu); LmMmaMxfp4(acc0, a[0][1], b1, sa[0] >> 16u, sb >> 16u);
			LmMmaMxfp4(acc1, a[1][0], b0, sa[1] & 0xffffu, sb & 0xffffu); LmMmaMxfp4(acc1, a[1][1], b1, sa[1] >> 16u, sb >> 16u);
			if constexpr ( ROUND_BF16 ) {
				auto rbf = [](float x) { return LmBf16ToFloat(LmFloatToBf16(x)); };
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e ) { acc0[e] = fmaxf(rbf(acc0[e]), 0.0f); acc1[e] = fmaxf(rbf(acc1[e]), 0.0f); }
				low = __fadd_rn(__fadd_rn(__fadd_rn(rbf(__fmul_rn(w[0], acc0[0])), rbf(__fmul_rn(w[1], acc0[2]))), rbf(__fmul_rn(w[2], acc1[0]))), rbf(__fmul_rn(w[3], acc1[2])));
				high = __fadd_rn(__fadd_rn(__fadd_rn(rbf(__fmul_rn(w[0], acc0[1])), rbf(__fmul_rn(w[1], acc0[3]))), rbf(__fmul_rn(w[2], acc1[1]))), rbf(__fmul_rn(w[3], acc1[3])));
			} else {
				#pragma unroll
				for ( uint32_t e = 0u; e < 4u; ++e ) { acc0[e] = fmaxf(acc0[e], 0.0f); acc1[e] = fmaxf(acc1[e], 0.0f); }
				low = __fmaf_rn(w[3], acc1[2], __fmaf_rn(w[2], acc1[0], __fmaf_rn(w[1], acc0[2], __fmul_rn(w[0], acc0[0]))));
				high = __fmaf_rn(w[3], acc1[3], __fmaf_rn(w[2], acc1[1], __fmaf_rn(w[1], acc0[3], __fmul_rn(w[0], acc0[1]))));
			}
			for ( uint32_t off = 4u; off < 32u; off <<= 1u ) { low += __shfl_xor_sync(0xffffffffu, low, off); high += __shfl_xor_sync(0xffffffffu, high, off); }
			if ( g == 0u && written + 1u < score_stride )
				*(float2 *)(scores + (uint64_t)row * score_stride + written) = make_float2(
					written < rbound ? (ROUND_BF16 ? LmBf16ToFloat(LmFloatToBf16(low)) : low) : -INFINITY,
					written + 1u < rbound ? (ROUND_BF16 ? LmBf16ToFloat(LmFloatToBf16(high)) : high) : -INFINITY);
		}
		__syncthreads();
	}
	LmDependentRelease();
}

static __device__ __forceinline__ uint32_t D41RowActiveKeys(uint32_t row, const uint32_t *seq, const uint32_t *len, const uint32_t *bound)
{
	uint32_t active = len[seq[row]];
	if ( bound != 0 ) { const uint32_t b = bound[row]; active = b == D41K_NO_ROW ? 0u : (b + 1u < active ? b + 1u : active); }
	return active;
}

static __global__ void D41WholeBlockMaxKernel(const float *__restrict__ scores, uint64_t score_stride, const uint32_t *__restrict__ seq,
	const uint32_t *__restrict__ len, const uint32_t *__restrict__ bound, float *__restrict__ block_scores, uint32_t block_stride,
	uint32_t grain, uint32_t dense_rows)
{
	__shared__ float s_vals[256u * 8u];
	const uint32_t row = blockIdx.y;
	LmDependentWait();
	LmDependentRelease();
	const uint32_t active = D41RowActiveKeys(row, seq, len, bound);
	if ( dense_rows != 0u && active <= dense_rows ) return;
	const float *src = scores + (uint64_t)row * score_stride;
	float *dst = block_scores + (uint64_t)row * block_stride;
	const uint32_t b0 = blockIdx.x * blockDim.x, b = b0 + threadIdx.x;
	if ( grain == 8u && blockDim.x == 256u && LmIsAligned16(src) )
	{
		const uint32_t base_k = b0 * 8u;
		if ( base_k >= active )
		{
			if ( b < block_stride ) dst[b] = -INFINITY;
			return;
		}
		#pragma unroll
		for ( uint32_t pass = 0u; pass < 2u; ++pass )
		{
			const uint32_t q = pass * 256u + threadIdx.x, off = q * 4u, gk = base_k + off;
			if ( gk + 3u < active && gk + 3u < score_stride )
				((float4 *)s_vals)[q] = ((const float4 *)src)[gk >> 2u];
			else
			{
				s_vals[off] = gk < active && gk < score_stride ? src[gk] : -INFINITY;
				s_vals[off + 1u] = gk + 1u < active && gk + 1u < score_stride ? src[gk + 1u] : -INFINITY;
				s_vals[off + 2u] = gk + 2u < active && gk + 2u < score_stride ? src[gk + 2u] : -INFINITY;
				s_vals[off + 3u] = gk + 3u < active && gk + 3u < score_stride ? src[gk + 3u] : -INFINITY;
			}
		}
		__syncthreads();
		if ( b >= block_stride ) return;
		const uint32_t o = threadIdx.x * 8u;
		float mx = fmaxf(fmaxf(s_vals[o], s_vals[o + 1u]), fmaxf(s_vals[o + 2u], s_vals[o + 3u]));
		mx = fmaxf(mx, fmaxf(fmaxf(s_vals[o + 4u], s_vals[o + 5u]), fmaxf(s_vals[o + 6u], s_vals[o + 7u])));
		dst[b] = mx;
		return;
	}
	if ( b >= block_stride ) return;
	const uint32_t first = b * grain;
	if ( first >= active ) { dst[b] = -INFINITY; return; }
	float mx = -INFINITY;
	const uint32_t last = first + grain < active ? first + grain : active;
	for ( uint32_t i = first; i < last; ++i ) mx = fmaxf(mx, src[i]);
	dst[b] = mx;
}

static __global__ void D41WholeBlockApplyKernel(float *__restrict__ scores, uint64_t score_stride, uint32_t grain, const uint8_t *__restrict__ mask,
	uint32_t block_stride, const uint32_t *__restrict__ seq, const uint32_t *__restrict__ len, const uint32_t *__restrict__ bound, uint32_t dense_rows)
{
	const uint32_t row = blockIdx.y;
	const uint32_t active = D41RowActiveKeys(row, seq, len, bound);
	if ( dense_rows != 0u && active <= dense_rows ) return;
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= active || i >= score_stride ) return;
	const uint32_t b = i / grain;
	if ( b >= block_stride || mask[(uint64_t)row * block_stride + b] == 0u )
		scores[(uint64_t)row * score_stride + i] = -INFINITY;
}

static __global__ void D41WholeBlockMaskKernel(const float *__restrict__ block_scores, const uint32_t *__restrict__ chosen, uint32_t keep,
	uint8_t *__restrict__ mask, uint32_t block_stride, const uint32_t *__restrict__ seq = 0, const uint32_t *__restrict__ len = 0,
	const uint32_t *__restrict__ bound = 0, uint32_t grain = 0u, uint32_t dense_rows = 0u)
{
	uint8_t *line = mask + (uint64_t)blockIdx.x * block_stride;
	LmDependentWait();
	LmDependentRelease();
	if ( dense_rows != 0u && seq != 0 && len != 0 )
	{
		const uint32_t active = D41RowActiveKeys(blockIdx.x, seq, len, bound);
		if ( active <= dense_rows )
		{
			const uint32_t active_blocks = (active + grain - 1u) / grain;
			for ( uint32_t b = threadIdx.x; b < block_stride; b += blockDim.x ) line[b] = b < active_blocks ? 1u : 0u;
			return;
		}
	}
	const float *scores = block_scores + (uint64_t)blockIdx.x * block_stride;
	for ( uint32_t b = threadIdx.x; b < block_stride; b += blockDim.x ) line[b] = 0u;
	__syncthreads();
	for ( uint32_t e = threadIdx.x; e < keep; e += blockDim.x ) { const uint32_t b = chosen[(uint64_t)blockIdx.x * keep + e]; if ( b < block_stride && scores[b] > -INFINITY ) line[b] = 1u; }
}

template<class Geometry, bool ROUND_BF16 = false>
static __global__ __launch_bounds__(256)
void D41IndexScoreDirectKernel(const uint16_t *__restrict__ index_query_bf16, const uint16_t *__restrict__ head_weight,
	LmKvShardView index_view, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position, const uint8_t *__restrict__ block_mask, uint32_t block_stride,
	uint32_t rows, uint32_t score_stride, float qk_scale, float *__restrict__ scores,
	const uint32_t *__restrict__ fork_positions = 0, const uint32_t *__restrict__ slot_offsets = 0, uint32_t dense_limit = 0u)
{
	constexpr uint32_t WARPS = 8u, BLOCK_TILES = 32u, BLOCK_KEYS = BLOCK_TILES * 8u, PITCH_U4 = 17u;
	__shared__ __align__(16) uint4 s_k[WARPS][8][PITCH_U4];
	const uint32_t row = blockIdx.y, block_first = blockIdx.x * BLOCK_KEYS;
	if ( scores == 0 || row >= rows ) return;
	LmDependentWait();
	const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u, group = lane >> 2u, pair = lane & 3u;
	const uint32_t valid_view = sequence_of_row != 0 && context_length != 0 && index_query_bf16 != 0 && head_weight != 0 &&
		LmKvViewIsConfigured(index_view.pages) ? 1u : 0u;
	const uint32_t seq = valid_view != 0u ? sequence_of_row[row] : 0xffffffffu;
	uint32_t valid = valid_view != 0u && seq < index_view.pages.sequence_count ? 1u : 0u;
	uint32_t bound = 0u;
	if ( valid == 0u )
	{
		if ( lane == 0u && blockIdx.x == 0u && warp == 0u )
			LmKvReportRequiredAccessFailure(index_view.pages, valid_view == 0u ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
				LM_KV_ACCESS_READ, row, seq, block_first, 0xffffffffu);
	}
	else
	{
		bound = context_length[seq];
		if ( row_position != 0 )
		{
			const uint32_t rp = row_position[row];
			bound = rp == D41K_NO_ROW ? 0u : (rp + 1u < bound ? rp + 1u : bound);
		}
		if ( dense_limit != 0u && bound <= dense_limit ) { LmDependentRelease(); return; }
		bound = SparkKvShardLocalKeys(index_view.shard, bound);
		if ( dense_limit != 0u && block_first >= bound ) { LmDependentRelease(); return; }
	}
	uint32_t a[2][8][4];
	float weight[4];
	if ( valid != 0u && block_first < bound )
	{
		const uint16_t *query = index_query_bf16 + (uint64_t)row * 32u * 128u;
		#pragma unroll
		for ( uint32_t ht = 0u; ht < 2u; ++ht )
		{
			#pragma unroll
			for ( uint32_t step = 0u; step < 8u; ++step )
			{
				const uint16_t *top = query + (uint64_t)(ht * 16u + group) * 128u + step * 16u + pair * 2u;
				const uint16_t *bottom = top + 8u * 128u;
				a[ht][step][0] = *(const uint32_t *)top;
				a[ht][step][1] = *(const uint32_t *)bottom;
				a[ht][step][2] = *(const uint32_t *)(top + 8u);
				a[ht][step][3] = *(const uint32_t *)(bottom + 8u);
			}
			weight[2u * ht] = LmBf16ToFloat(head_weight[(uint64_t)row * 32u + ht * 16u + group]);
			weight[2u * ht + 1u] = LmBf16ToFloat(head_weight[(uint64_t)row * 32u + ht * 16u + group + 8u]);
		}
	}
	const uint32_t fork = valid != 0u && fork_positions != 0 ? fork_positions[row] : 0xffffffffu;
	const uint32_t offset = valid != 0u && slot_offsets != 0 ? slot_offsets[row] : 0u;
	const uint32_t half = lane >> 4u, lane16 = lane & 15u;
	for ( uint32_t tile = warp; tile < BLOCK_TILES; tile += WARPS )
	{
		const uint32_t first = block_first + tile * 8u, blk = first >> 3u, written = first + pair * 2u;
		if ( first >= score_stride ) break;
		if ( valid == 0u || first >= bound )
		{
			if ( dense_limit == 0u && group == 0u && written < score_stride )
			{
				if ( written + 1u < score_stride )
					*(float2 *)(scores + (uint64_t)row * score_stride + written) = make_float2(-INFINITY, -INFINITY);
				else
					scores[(uint64_t)row * score_stride + written] = -INFINITY;
			}
			continue;
		}
		if ( block_mask != 0 && (blk >= block_stride || block_mask[(uint64_t)row * block_stride + blk] == 0u) )
		{
			if ( group == 0u && written < score_stride )
			{
				if ( written + 1u < score_stride )
					*(float2 *)(scores + (uint64_t)row * score_stride + written) = make_float2(-INFINITY, -INFINITY);
				else
					scores[(uint64_t)row * score_stride + written] = -INFINITY;
			}
			continue;
		}
		#pragma unroll
		for ( uint32_t p = 0u; p < 4u; ++p )
		{
			const uint32_t ki = p * 2u + half, gk = first + ki;
			const uint8_t *slot = gk < bound ? LmIndexScoreSlot<Geometry>(index_view, seq, gk, row, fork, offset) : 0;
			s_k[warp][ki][lane16] = slot != 0 ? __ldg((const uint4 *)slot + lane16) : make_uint4(0u, 0u, 0u, 0u);
		}
		__syncwarp();
		float acc[2][4] = {};
		const uint32_t *krow = (const uint32_t *)s_k[warp][group];
		#pragma unroll
		for ( uint32_t step = 0u; step < 8u; ++step )
		{
			uint32_t b[2] = { krow[step * 8u + pair], krow[step * 8u + 4u + pair] };
			LmMmaBf16(acc[0], a[0][step], b);
			LmMmaBf16(acc[1], a[1][step], b);
		}
		LmIndexScoreStore<2u, ROUND_BF16>(acc, weight, 1u, bound, written, group, row, score_stride, qk_scale, scores);
	}
	LmDependentRelease();
}

static __device__ __forceinline__ void D41TopkHistKey(uint32_t key, uint32_t mask, uint32_t prefix, uint32_t shift, uint32_t field, uint32_t valid,
	uint32_t lane, uint32_t *hist)
{
	const uint32_t b = valid != 0u && (key & mask) == prefix ? ((key >> shift) & field) : 0xffffffffu;
	const uint32_t peers = __match_any_sync(0xffffffffu, b);
	if ( b != 0xffffffffu && lane == (uint32_t)__ffs((int)peers) - 1u )
		atomicAdd(&hist[b], (uint32_t)__popc(peers));
}

static __device__ __forceinline__ void D41TopkDescendSelect(const uint32_t local[8], uint32_t need, uint32_t warp, uint32_t lane,
	uint32_t *s_warp, uint32_t *s_choice)
{
	uint32_t run = 0u;
	#pragma unroll
	for ( uint32_t p = 0u; p < 8u; ++p ) run += local[p];
	uint32_t inc = run;
	#pragma unroll
	for ( uint32_t off = 1u; off < 32u; off <<= 1u )
	{
		const uint32_t add = __shfl_up_sync(0xffffffffu, inc, off);
		if ( lane >= off ) inc += add;
	}
	if ( lane == 31u ) s_warp[warp] = inc;
	__syncthreads();
	uint32_t before = inc - run;
	#pragma unroll
	for ( uint32_t w = 0u; w < 8u; ++w )
		if ( w < warp ) before += s_warp[w];
	#pragma unroll
	for ( uint32_t p = 0u; p < 8u; ++p )
	{
		if ( before < need && before + local[p] >= need )
		{
			s_choice[0] = 2047u - (threadIdx.x * 8u + p);
			s_choice[1] = before;
			s_choice[2] = local[p];
		}
		before += local[p];
	}
	__syncthreads();
}

static __device__ __forceinline__ void D41TopkEmit4(const uint32_t ks[4], uint32_t base_idx, uint32_t eff_n, uint32_t mask, uint32_t prefix,
	uint32_t need, uint32_t warp, uint32_t lane, uint32_t *s_scan, uint32_t *out, uint32_t *emitted, uint32_t *seen_eq)
{
	const uint32_t g0 = (base_idx + 0u < eff_n && (ks[0] & mask) > prefix) ? 1u : 0u;
	const uint32_t e0 = (base_idx + 0u < eff_n && (ks[0] & mask) == prefix) ? 1u : 0u;
	const uint32_t g1 = (base_idx + 1u < eff_n && (ks[1] & mask) > prefix) ? 1u : 0u;
	const uint32_t e1 = (base_idx + 1u < eff_n && (ks[1] & mask) == prefix) ? 1u : 0u;
	const uint32_t g2 = (base_idx + 2u < eff_n && (ks[2] & mask) > prefix) ? 1u : 0u;
	const uint32_t e2 = (base_idx + 2u < eff_n && (ks[2] & mask) == prefix) ? 1u : 0u;
	const uint32_t g3 = (base_idx + 3u < eff_n && (ks[3] & mask) > prefix) ? 1u : 0u;
	const uint32_t e3 = (base_idx + 3u < eff_n && (ks[3] & mask) == prefix) ? 1u : 0u;
	const uint32_t gc = g0 + g1 + g2 + g3, ec = e0 + e1 + e2 + e3;
	uint32_t gi = gc, ei = ec;
	#pragma unroll
	for ( uint32_t off = 1u; off < 32u; off <<= 1u )
	{
		const uint32_t ga = __shfl_up_sync(0xffffffffu, gi, off), ea = __shfl_up_sync(0xffffffffu, ei, off);
		if ( lane >= off ) { gi += ga; ei += ea; }
	}
	if ( lane == 31u ) { s_scan[warp] = gi; s_scan[8u + warp] = ei; }
	__syncthreads();
	uint32_t gb = gi - gc, eb = ei - ec, gtot = 0u, etot = 0u;
	#pragma unroll
	for ( uint32_t w = 0u; w < 8u; ++w )
	{
		const uint32_t gw = s_scan[w], ew = s_scan[8u + w];
		if ( w < warp ) { gb += gw; eb += ew; }
		gtot += gw; etot += ew;
	}
	const uint32_t rem = need > *seen_eq ? need - *seen_eq : 0u, em = *emitted;
	if ( g0 || (e0 && eb < rem) ) out[em + gb + (eb < rem ? eb : rem)] = base_idx + 0u;
	gb += g0; eb += e0;
	if ( g1 || (e1 && eb < rem) ) out[em + gb + (eb < rem ? eb : rem)] = base_idx + 1u;
	gb += g1; eb += e1;
	if ( g2 || (e2 && eb < rem) ) out[em + gb + (eb < rem ? eb : rem)] = base_idx + 2u;
	gb += g2; eb += e2;
	if ( g3 || (e3 && eb < rem) ) out[em + gb + (eb < rem ? eb : rem)] = base_idx + 3u;
	*emitted = em + gtot + (etot < rem ? etot : rem);
	*seen_eq += etot;
	__syncthreads();
}

static __device__ __forceinline__ void D41LoadKeys4(const float *__restrict__ row, uint32_t idx, uint32_t eff_n, bool aligned16, uint32_t ks[4])
{
	if ( aligned16 && idx + 3u < eff_n )
	{
		const float4 v = __ldg((const float4 *)(row + idx));
		ks[0] = LmTopkKey(v.x); ks[1] = LmTopkKey(v.y); ks[2] = LmTopkKey(v.z); ks[3] = LmTopkKey(v.w);
	}
	else
	{
		ks[0] = idx + 0u < eff_n ? LmTopkKey(__ldg(row + idx + 0u)) : 0u;
		ks[1] = idx + 1u < eff_n ? LmTopkKey(__ldg(row + idx + 1u)) : 0u;
		ks[2] = idx + 2u < eff_n ? LmTopkKey(__ldg(row + idx + 2u)) : 0u;
		ks[3] = idx + 3u < eff_n ? LmTopkKey(__ldg(row + idx + 3u)) : 0u;
	}
}

static __device__ __forceinline__ void D41TopkHistKeyPass(uint32_t key, uint32_t mask, uint32_t prefix, uint32_t shift, uint32_t field, uint32_t valid,
	uint32_t *hist)
{
	if ( valid != 0u && (key & mask) == prefix )
		atomicAdd(&hist[(key >> shift) & field], 1u);
}

static __device__ __forceinline__ void D41TopkSelectReg4096(const float *__restrict__ row, uint32_t eff_n, uint32_t k, uint32_t *__restrict__ out,
	uint32_t *s_hist, uint32_t *s_warp, uint32_t *s_choice, uint32_t *s_scan)
{
	const bool aligned16 = LmIsAligned16(row);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	uint32_t rk[4][4];
	#pragma unroll
	for ( uint32_t c = 0u; c < 4u; ++c )
		D41LoadKeys4(row, c * 1024u + threadIdx.x * 4u, eff_n, aligned16, rk[c]);
	uint32_t prefix = 0u, mask = 0u, need = k, count = eff_n;
	for ( uint32_t pass = 0u; pass < 3u; ++pass )
	{
		const uint32_t shift = pass == 0u ? 21u : (pass == 1u ? 10u : 0u), field = pass < 2u ? 0x7ffu : 0x3ffu;
		#pragma unroll
		for ( uint32_t p = 0u; p < 8u; ++p ) s_hist[p * 256u + threadIdx.x] = 0u;
		__syncthreads();
		#pragma unroll
		for ( uint32_t c = 0u; c < 4u; ++c )
		{
			const uint32_t bidx = c * 1024u + threadIdx.x * 4u;
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; ++e )
			{
				if ( pass == 0u ) D41TopkHistKey(rk[c][e], mask, prefix, shift, field, bidx + e < eff_n ? 1u : 0u, lane, s_hist);
				else D41TopkHistKeyPass(rk[c][e], mask, prefix, shift, field, bidx + e < eff_n ? 1u : 0u, s_hist);
			}
		}
		__syncthreads();
		uint32_t local[8];
		#pragma unroll
		for ( uint32_t p = 0u; p < 8u; ++p ) local[p] = s_hist[2047u - (threadIdx.x * 8u + p)];
		D41TopkDescendSelect(local, need, warp, lane, s_warp, s_choice);
		prefix |= s_choice[0] << shift;
		mask |= field << shift;
		need -= s_choice[1];
		count = s_choice[2];
		if ( count == need ) break;
	}
	uint32_t emitted = 0u, seen_eq = 0u;
	#pragma unroll
	for ( uint32_t c = 0u; c < 4u; ++c )
		if ( c * 1024u < eff_n )
			D41TopkEmit4(rk[c], c * 1024u + threadIdx.x * 4u, eff_n, mask, prefix, need, warp, lane, s_scan, out, &emitted, &seen_eq);
}

static __device__ __forceinline__ void D41TopkSelectStream(const float *__restrict__ row, uint32_t eff_n, uint32_t k, uint32_t *__restrict__ out,
	uint32_t *s_hist, uint32_t *s_warp, uint32_t *s_choice, uint32_t *s_scan)
{
	const bool aligned16 = LmIsAligned16(row);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	uint32_t prefix = 0u, mask = 0u, need = k, count = eff_n;
	for ( uint32_t pass = 0u; pass < 3u; ++pass )
	{
		const uint32_t shift = pass == 0u ? 21u : (pass == 1u ? 10u : 0u), field = pass < 2u ? 0x7ffu : 0x3ffu;
		#pragma unroll
		for ( uint32_t p = 0u; p < 8u; ++p ) s_hist[p * 256u + threadIdx.x] = 0u;
		__syncthreads();
		for ( uint32_t base = 0u; base < eff_n; base += 1024u )
		{
			uint32_t ks[4];
			const uint32_t bidx = base + threadIdx.x * 4u;
			D41LoadKeys4(row, bidx, eff_n, aligned16, ks);
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; ++e )
			{
				if ( pass == 0u ) D41TopkHistKey(ks[e], mask, prefix, shift, field, bidx + e < eff_n ? 1u : 0u, lane, s_hist);
				else D41TopkHistKeyPass(ks[e], mask, prefix, shift, field, bidx + e < eff_n ? 1u : 0u, s_hist);
			}
		}
		__syncthreads();
		uint32_t local[8];
		#pragma unroll
		for ( uint32_t p = 0u; p < 8u; ++p ) local[p] = s_hist[2047u - (threadIdx.x * 8u + p)];
		D41TopkDescendSelect(local, need, warp, lane, s_warp, s_choice);
		prefix |= s_choice[0] << shift;
		mask |= field << shift;
		need -= s_choice[1];
		count = s_choice[2];
		if ( count == need ) break;
	}
	uint32_t emitted = 0u, seen_eq = 0u;
	for ( uint32_t base = 0u; base < eff_n; base += 1024u )
	{
		uint32_t ks[4];
		const uint32_t bidx = base + threadIdx.x * 4u;
		D41LoadKeys4(row, bidx, eff_n, aligned16, ks);
		D41TopkEmit4(ks, bidx, eff_n, mask, prefix, need, warp, lane, s_scan, out, &emitted, &seen_eq);
	}
}

static __global__ __launch_bounds__(256)
void D41TopkExactFastKernel(const float *__restrict__ scores, uint32_t n, uint32_t k, uint32_t *__restrict__ out_indices,
	const uint32_t *__restrict__ seq = 0, const uint32_t *__restrict__ len = 0, const uint32_t *__restrict__ bound = 0,
	uint32_t dense_limit = 0u, uint32_t pool_stride = 1u)
{
	__shared__ uint32_t s_hist[2048];
	__shared__ uint32_t s_warp[8];
	__shared__ uint32_t s_choice[3];
	__shared__ uint32_t s_scan[16];
	LmDependentWait();
	uint32_t skip = 0u;
	const uint32_t eff_n = LmTopkExactRowActive(blockIdx.x, n, seq, len, bound, dense_limit, pool_stride, &skip);
	if ( skip != 0u ) { LmDependentRelease(); return; }
	uint32_t *out = out_indices + (uint64_t)blockIdx.x * k;
	if ( eff_n <= k )
	{
		for ( uint32_t i = threadIdx.x; i < k; i += 256u ) out[i] = i < eff_n ? i : 0xffffffffu;
		LmDependentRelease();
		return;
	}
	const float *row = scores + (uint64_t)blockIdx.x * n;
	if ( eff_n <= 4096u ) D41TopkSelectReg4096(row, eff_n, k, out, s_hist, s_warp, s_choice, s_scan);
	else D41TopkSelectStream(row, eff_n, k, out, s_hist, s_warp, s_choice, s_scan);
	LmDependentRelease();
}

static __global__ __launch_bounds__(256)
void D41TopkExactChunkRegKernel(const float *__restrict__ scores, uint32_t n, uint32_t k, uint32_t chunk,
	float *__restrict__ out_values, uint32_t *__restrict__ out_positions,
	const uint32_t *__restrict__ seq = 0, const uint32_t *__restrict__ len = 0, const uint32_t *__restrict__ bound = 0,
	uint32_t dense_limit = 0u, uint32_t pool_stride = 1u)
{
	__shared__ uint32_t s_hist[2048];
	__shared__ uint32_t s_warp[8];
	__shared__ uint32_t s_choice[3];
	__shared__ uint32_t s_scan[16];
	LmDependentWait();
	const uint32_t row_idx = blockIdx.y, cid = blockIdx.x, first = cid * chunk;
	uint32_t skip = 0u;
	const uint32_t eff_n = LmTopkExactRowActive(row_idx, n, seq, len, bound, dense_limit, pool_stride, &skip);
	if ( skip != 0u || eff_n <= k || first >= eff_n ) { LmDependentRelease(); return; }
	const uint32_t length = eff_n - first < chunk ? eff_n - first : chunk;
	const float *row = scores + (uint64_t)row_idx * n + first;
	const uint64_t base = ((uint64_t)row_idx * gridDim.x + cid) * k;
	uint32_t *pos_out = out_positions + base;
	float *val_out = out_values + base;
	if ( length <= k )
	{
		for ( uint32_t i = threadIdx.x; i < k; i += 256u )
		{
			val_out[i] = i < length ? row[i] : -INFINITY;
			pos_out[i] = i < length ? first + i : 0xffffffffu;
		}
		LmDependentRelease();
		return;
	}
	if ( length <= 4096u ) D41TopkSelectReg4096(row, length, k, pos_out, s_hist, s_warp, s_choice, s_scan);
	else D41TopkSelectStream(row, length, k, pos_out, s_hist, s_warp, s_choice, s_scan);
	__syncthreads();
	for ( uint32_t i = threadIdx.x; i < k; i += 256u )
	{
		const uint32_t loc = pos_out[i];
		val_out[i] = loc != 0xffffffffu ? row[loc] : -INFINITY;
		pos_out[i] = loc != 0xffffffffu ? first + loc : 0xffffffffu;
	}
	LmDependentRelease();
}

static __global__ __launch_bounds__(256)
void D41TopkExactGatherRegKernel(const float *__restrict__ scores, const float *__restrict__ values, const uint32_t *__restrict__ positions,
	uint32_t n, uint32_t k, uint32_t chunk, uint32_t max_chunks, uint32_t *__restrict__ out_indices,
	const uint32_t *__restrict__ seq = 0, const uint32_t *__restrict__ len = 0, const uint32_t *__restrict__ bound = 0,
	uint32_t dense_limit = 0u, uint32_t pool_stride = 1u)
{
	__shared__ uint32_t s_hist[2048];
	__shared__ uint32_t s_warp[8];
	__shared__ uint32_t s_choice[3];
	__shared__ uint32_t s_scan[16];
	LmDependentWait();
	const uint32_t row_idx = blockIdx.x;
	uint32_t skip = 0u;
	const uint32_t eff_n = LmTopkExactRowActive(row_idx, n, seq, len, bound, dense_limit, pool_stride, &skip);
	if ( skip != 0u ) { LmDependentRelease(); return; }
	uint32_t *out = out_indices + (uint64_t)row_idx * k;
	if ( eff_n <= k )
	{
		for ( uint32_t i = threadIdx.x; i < k; i += 256u ) out[i] = i < eff_n ? i : 0xffffffffu;
		LmDependentRelease();
		return;
	}
	const uint32_t active_chunks = (eff_n + chunk - 1u) / chunk;
	const uint32_t cand_n = active_chunks * k;
	const uint64_t base = (uint64_t)row_idx * max_chunks * k;
	const float *row_vals = values + base;
	const uint32_t *row_pos = positions + base;
	if ( active_chunks == 1u )
	{
		for ( uint32_t i = threadIdx.x; i < k; i += 256u ) out[i] = row_pos[i];
		LmDependentRelease();
		return;
	}
	if ( cand_n <= 4096u ) D41TopkSelectReg4096(row_vals, cand_n, k, out, s_hist, s_warp, s_choice, s_scan);
	else D41TopkSelectStream(row_vals, cand_n, k, out, s_hist, s_warp, s_choice, s_scan);
	__syncthreads();
	for ( uint32_t i = threadIdx.x; i < k; i += 256u )
		out[i] = out[i] != 0xffffffffu ? row_pos[out[i]] : 0xffffffffu;
	(void)scores;
	LmDependentRelease();
}

static inline SparkGpuError D41TopkExactLaunch(const float *scores, uint32_t rows, uint32_t n, uint32_t k, uint32_t *out_indices,
	SparkGpuStream stream, const uint32_t *seq = 0, const uint32_t *len = 0, const uint32_t *bound = 0, uint32_t dense_limit = 0u,
	uint32_t pool_stride = 1u, float *topk_values = 0, uint32_t *topk_positions = 0, uint64_t topk_entries = 0u)
{
	if ( rows == 0u ) return SPARK_GPU_SUCCESS;
	if ( rows <= 16u && n > 4096u && topk_values != 0 && topk_positions != 0 )
	{
		const uint32_t chunk = (uint64_t)rows * ((n + 4095u) / 4096u) * k <= topk_entries ? 4096u :
			(uint64_t)rows * ((n + 8191u) / 8192u) * k <= topk_entries ? 8192u : 0u;
		if ( chunk != 0u )
		{
			const uint32_t chunks = (n + chunk - 1u) / chunk;
			LM_LAUNCH_DEPENDENT((D41TopkExactChunkRegKernel), dim3(chunks, rows), 256u, 0, stream, scores, n, k, chunk, topk_values, topk_positions,
				seq, len, bound, dense_limit, pool_stride);
			LM_LAUNCH_DEPENDENT((D41TopkExactGatherRegKernel), rows, 256u, 0, stream, scores, topk_values, topk_positions, n, k, chunk, chunks, out_indices,
				seq, len, bound, dense_limit, pool_stride);
			return SparkGpuPeekAtLastError();
		}
	}
	LM_LAUNCH_DEPENDENT((D41TopkExactFastKernel), rows, 256u, 0, stream, scores, n, k, out_indices, seq, len, bound, dense_limit, pool_stride);
	return SparkGpuPeekAtLastError();
}

template<uint32_t EXPERTS, uint32_t TOP_K, uint32_t MAX_ROUTES = 128u>
static __device__ __forceinline__ void D41TopkRoutePackedBody(const float *__restrict__ router_logits, const float *__restrict__ router_bias, float route_scale,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up,
	uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down, float *__restrict__ packed_weight, uint32_t rows,
	const uint32_t *__restrict__ row_valid = 0)
{
	static_assert(EXPERTS <= 512u && TOP_K <= 8u, "fused route build covers up to 512 experts");
	__shared__ uint32_t s_off[512];
	__shared__ uint32_t s_up[512];
	__shared__ uint32_t s_dn[512];
	__shared__ uint32_t s_warp[8];
	__shared__ uint32_t s_re[MAX_ROUTES];
	__shared__ float s_rw[MAX_ROUTES];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, routes = rows * TOP_K;
	for ( uint32_t i = threadIdx.x; i < 512u; i += 256u ) s_off[i] = 0u;
	for ( uint32_t row = warp; row < rows; row += 8u )
	{
		if ( row_valid != 0 && row_valid[row] == 0u )
		{
			if ( lane < TOP_K )
			{
				route_expert[(uint64_t)row * TOP_K + lane] = 0xffffffffu;
				route_weight[(uint64_t)row * TOP_K + lane] = 0.0f;
			}
			continue;
		}
		LmTopkWarpRow<TOP_K, true, LM_TOPK_SCORE_SQRT_SOFTPLUS>(router_logits, EXPERTS, route_expert, route_weight, router_bias,
			(const uint16_t *)0, route_scale, fault, row, lane);
	}
	__syncthreads();
	if ( threadIdx.x < routes )
	{
		const uint32_t e = route_expert[threadIdx.x];
		const uint32_t dead = (row_valid != 0 && row_valid[threadIdx.x / TOP_K] == 0u) || e >= EXPERTS ? 1u : 0u;
		s_re[threadIdx.x] = dead != 0u ? 0xffffffffu : e;
		s_rw[threadIdx.x] = dead != 0u ? 0.0f : route_weight[threadIdx.x];
		if ( dead == 0u ) atomicAdd(&s_off[e], 1u);
	}
	__syncthreads();
	auto scan512 = [&](uint32_t *arr) -> uint32_t
	{
		const uint32_t v0 = threadIdx.x < EXPERTS ? arr[threadIdx.x] : 0u;
		const uint32_t v1 = threadIdx.x + 256u < EXPERTS ? arr[threadIdx.x + 256u] : 0u;
		uint32_t x0 = v0;
		#pragma unroll
		for ( uint32_t d = 1u; d < 32u; d <<= 1u ) { const uint32_t u = __shfl_up_sync(0xffffffffu, x0, d); if ( lane >= d ) x0 += u; }
		if ( lane == 31u ) s_warp[warp] = x0;
		__syncthreads();
		if ( warp == 0u )
		{
			uint32_t w = lane < 8u ? s_warp[lane] : 0u;
			#pragma unroll
			for ( uint32_t d = 1u; d < 8u; d <<= 1u ) { const uint32_t u = __shfl_up_sync(0xffffffffu, w, d); if ( lane >= d ) w += u; }
			if ( lane < 8u ) s_warp[lane] = w;
		}
		__syncthreads();
		const uint32_t tot0 = s_warp[7];
		const uint32_t off0 = (warp > 0u ? s_warp[warp - 1u] : 0u) + x0 - v0;
		__syncthreads();
		uint32_t x1 = v1;
		#pragma unroll
		for ( uint32_t d = 1u; d < 32u; d <<= 1u ) { const uint32_t u = __shfl_up_sync(0xffffffffu, x1, d); if ( lane >= d ) x1 += u; }
		if ( lane == 31u ) s_warp[warp] = x1;
		__syncthreads();
		if ( warp == 0u )
		{
			uint32_t w = lane < 8u ? s_warp[lane] : 0u;
			#pragma unroll
			for ( uint32_t d = 1u; d < 8u; d <<= 1u ) { const uint32_t u = __shfl_up_sync(0xffffffffu, w, d); if ( lane >= d ) w += u; }
			if ( lane < 8u ) s_warp[lane] = w;
		}
		__syncthreads();
		const uint32_t tot1 = s_warp[7];
		const uint32_t off1 = tot0 + (warp > 0u ? s_warp[warp - 1u] : 0u) + x1 - v1;
		__syncthreads();
		if ( threadIdx.x < EXPERTS ) arr[threadIdx.x] = off0;
		if ( threadIdx.x + 256u < EXPERTS ) arr[threadIdx.x + 256u] = off1;
		return tot0 + tot1;
	};
	for ( uint32_t i = threadIdx.x; i < EXPERTS; i += 256u )
	{
		const uint32_t c = s_off[i];
		const uint32_t mt = c != 0u ? (c + tile_m - 1u) / tile_m : 0u;
		s_up[i] = mt * neuron_tiles_up;
		s_dn[i] = mt * neuron_tiles_down;
	}
	__syncthreads();
	const uint32_t tot_off = scan512(s_off);
	const uint32_t tot_up = scan512(s_up);
	const uint32_t tot_dn = scan512(s_dn);
	for ( uint32_t i = threadIdx.x; i < EXPERTS; i += 256u )
	{
		group_row_offset[i] = s_off[i];
		tile_prefix_up[i] = s_up[i];
		tile_prefix_down[i] = s_dn[i];
	}
	if ( threadIdx.x == 0u )
	{
		group_row_offset[EXPERTS] = tot_off;
		tile_prefix_up[EXPERTS] = tot_up;
		tile_prefix_down[EXPERTS] = tot_dn;
	}
	__syncthreads();
	if ( threadIdx.x < routes )
	{
		const uint32_t r = threadIdx.x, e = s_re[r];
		if ( e < EXPERTS )
		{
			uint32_t ord = 0u;
			for ( uint32_t prev = 0u; prev < r; ++prev )
				if ( s_re[prev] == e ) ++ord;
			const uint32_t packed = s_off[e] + ord;
			route_packed_row[r] = packed;
			route_source_token[packed] = r / TOP_K;
			packed_weight[packed] = s_rw[r];
		}
		else
			route_packed_row[r] = routes;
	}
}

template<uint32_t EXPERTS, uint32_t TOP_K>
static __global__ __launch_bounds__(256)
void D41TopkRoutePackedKernel(const float *__restrict__ router_logits, const float *__restrict__ router_bias, float route_scale,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up,
	uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down, float *__restrict__ packed_weight, uint32_t rows,
	const uint32_t *__restrict__ row_valid = 0)
{
	LmDependentWait();
	D41TopkRoutePackedBody<EXPERTS, TOP_K, 128u>(router_logits, router_bias, route_scale, fault, route_expert, route_weight,
		group_row_offset, route_packed_row, route_source_token, tile_m, neuron_tiles_up, tile_prefix_up, neuron_tiles_down,
		tile_prefix_down, packed_weight, rows, row_valid);
	LmDependentRelease();
}

static __device__ __forceinline__ float D41Bf16Dot8(uint4 weight, uint4 activation)
{
	float a0 = __uint_as_float(weight.x << 16u) * __uint_as_float(activation.x << 16u);
	float a1 = __uint_as_float(weight.x & 0xffff0000u) * __uint_as_float(activation.x & 0xffff0000u);
	a0 = fmaf(__uint_as_float(weight.y << 16u), __uint_as_float(activation.y << 16u), a0);
	a1 = fmaf(__uint_as_float(weight.y & 0xffff0000u), __uint_as_float(activation.y & 0xffff0000u), a1);
	a0 = fmaf(__uint_as_float(weight.z << 16u), __uint_as_float(activation.z << 16u), a0);
	a1 = fmaf(__uint_as_float(weight.z & 0xffff0000u), __uint_as_float(activation.z & 0xffff0000u), a1);
	a0 = fmaf(__uint_as_float(weight.w << 16u), __uint_as_float(activation.w << 16u), a0);
	a1 = fmaf(__uint_as_float(weight.w & 0xffff0000u), __uint_as_float(activation.w & 0xffff0000u), a1);
	return a0 + a1;
}

template<uint32_t EXPERTS, uint32_t HIDDEN, uint32_t MAX_R>
static __device__ __forceinline__ void D41RouterGateSplitBlock(const uint16_t *__restrict__ router_w, const uint16_t *__restrict__ x,
	float *__restrict__ router_logits, uint32_t rows, const uint32_t *__restrict__ row_valid)
{
	constexpr uint32_t CHUNKS = HIDDEN / 8u, WARPS = 8u, SPAN = CHUNKS / WARPS;
	static_assert(CHUNKS % WARPS == 0u && SPAN <= 96u, "router hidden splits evenly across eight warps");
	__shared__ float partial[WARPS][MAX_R];
	const uint32_t e = blockIdx.x, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, first = warp * SPAN;
	const uint4 *w_base = (const uint4 *)router_w, *x_base = (const uint4 *)x;
	const uint4 *wr = w_base + (uint64_t)e * CHUNKS + first;
	if ( lane == 0u )
		LmPrefetchL2(wr, SPAN * 16u);
	LmDependentWait();
	uint4 w0 = {}, w1 = {}, w2 = {};
	if ( lane < SPAN ) w0 = __ldcs(wr + lane);
	if ( lane + 32u < SPAN ) w1 = __ldcs(wr + lane + 32u);
	if ( lane + 64u < SPAN ) w2 = __ldcs(wr + lane + 64u);
	float acc[MAX_R];
	#pragma unroll
	for ( uint32_t r = 0u; r < MAX_R; ++r )
	{
		const uint32_t row = r < rows ? r : 0u;
		const uint4 *xr = x_base + (uint64_t)row * CHUNKS + first;
		float s = 0.0f;
		if ( lane < SPAN ) s += D41Bf16Dot8(w0, __ldg(xr + lane));
		if ( lane + 32u < SPAN ) s += D41Bf16Dot8(w1, __ldg(xr + lane + 32u));
		if ( lane + 64u < SPAN ) s += D41Bf16Dot8(w2, __ldg(xr + lane + 64u));
		acc[r] = s;
	}
	#pragma unroll
	for ( uint32_t off = 16u; off > 0u; off >>= 1u )
		#pragma unroll
		for ( uint32_t r = 0u; r < MAX_R; ++r )
			acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
	if ( lane == 0u )
		#pragma unroll
		for ( uint32_t r = 0u; r < MAX_R; ++r ) partial[warp][r] = acc[r];
	__syncthreads();
	if ( warp == 0u && lane < rows )
	{
		float total = 0.0f;
		if ( row_valid == 0 || row_valid[lane] != 0u )
		{
			#pragma unroll
			for ( uint32_t w = 0u; w < WARPS; ++w ) total += partial[w][lane];
		}
		router_logits[(uint64_t)lane * EXPERTS + e] = total;
	}
}

template<uint32_t RANKS, uint32_t EXPERTS, uint32_t LOCAL, uint32_t THREADS, uint32_t HELD_PLUS_ONE, uint32_t TOP_K, uint32_t MAX_COPIES, uint32_t HIDDEN, uint32_t MAX_R>
static __global__ __launch_bounds__(THREADS)
void D41FusedRouterBalanceKernel(const uint16_t *__restrict__ router_w, const uint16_t *__restrict__ x, float *__restrict__ router_logits,
	const float *__restrict__ router_bias, float route_scale, uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert,
	float *__restrict__ route_weight, uint32_t *__restrict__ route_global, uint32_t rows, uint32_t rank, uint32_t copies, uint32_t row_invariant,
	const uint8_t *__restrict__ owner, const uint8_t *__restrict__ backup, const uint8_t *__restrict__ group, uint32_t *__restrict__ group_row_offset,
	uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up,
	uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down, float *__restrict__ packed_weight,
	uint32_t *__restrict__ counter, const uint32_t *__restrict__ row_valid = 0)
{
	D41RouterGateSplitBlock<EXPERTS, HIDDEN, MAX_R>(router_w, x, router_logits, rows, row_valid);
	if ( LmFusedLastArrival(counter, EXPERTS) )
		D41TopkBalanceRouteBody<RANKS, EXPERTS, LOCAL, THREADS, HELD_PLUS_ONE, TOP_K, MAX_COPIES, true, MAX_R>(router_logits, router_bias, route_scale,
			fault, route_expert, route_weight, route_global, rows, rank, copies, row_invariant, owner, backup, group, group_row_offset, route_packed_row,
			route_source_token, tile_m, neuron_tiles_up, tile_prefix_up, neuron_tiles_down, tile_prefix_down, packed_weight, row_valid);
	LmDependentRelease();
}

template<uint32_t EXPERTS, uint32_t TOP_K, uint32_t HIDDEN, uint32_t MAX_R>
static __global__ __launch_bounds__(256)
void D41FusedRouterPackedKernel(const uint16_t *__restrict__ router_w, const uint16_t *__restrict__ x, float *__restrict__ router_logits,
	const float *__restrict__ router_bias, float route_scale, uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert,
	float *__restrict__ route_weight, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row,
	uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up,
	uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down, float *__restrict__ packed_weight, uint32_t rows,
	uint32_t *__restrict__ counter, const uint32_t *__restrict__ row_valid = 0)
{
	D41RouterGateSplitBlock<EXPERTS, HIDDEN, MAX_R>(router_w, x, router_logits, rows, row_valid);
	if ( LmFusedLastArrival(counter, EXPERTS) )
		D41TopkRoutePackedBody<EXPERTS, TOP_K, MAX_R * TOP_K>(router_logits, router_bias, route_scale, fault, route_expert, route_weight,
			group_row_offset, route_packed_row, route_source_token, tile_m, neuron_tiles_up, tile_prefix_up, neuron_tiles_down,
			tile_prefix_down, packed_weight, rows, row_valid);
	LmDependentRelease();
}

template<uint32_t EXPERTS>
static __global__ __launch_bounds__(256)
void D41CompactActiveExpertsKernel(const uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ active_out)
{
	__shared__ uint32_t s_warp[8];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	LmDependentWait();
	const uint32_t a0 = threadIdx.x < EXPERTS && group_row_offset[threadIdx.x + 1u] > group_row_offset[threadIdx.x] ? 1u : 0u;
	const uint32_t a1 = threadIdx.x + 256u < EXPERTS && group_row_offset[threadIdx.x + 257u] > group_row_offset[threadIdx.x + 256u] ? 1u : 0u;
	const uint32_t m0 = __ballot_sync(0xffffffffu, a0 != 0u);
	if ( lane == 0u ) s_warp[warp] = (uint32_t)__popc(m0);
	__syncthreads();
	if ( warp == 0u )
	{
		uint32_t w = lane < 8u ? s_warp[lane] : 0u;
		#pragma unroll
		for ( uint32_t d = 1u; d < 8u; d <<= 1u ) { const uint32_t u = __shfl_up_sync(0xffffffffu, w, d); if ( lane >= d ) w += u; }
		if ( lane < 8u ) s_warp[lane] = w;
	}
	__syncthreads();
	const uint32_t tot0 = s_warp[7];
	const uint32_t off0 = (warp > 0u ? s_warp[warp - 1u] : 0u) + (uint32_t)__popc(m0 & ((1u << lane) - 1u));
	if ( a0 != 0u ) active_out[off0] = threadIdx.x;
	__syncthreads();
	const uint32_t m1 = __ballot_sync(0xffffffffu, a1 != 0u);
	if ( lane == 0u ) s_warp[warp] = (uint32_t)__popc(m1);
	__syncthreads();
	if ( warp == 0u )
	{
		uint32_t w = lane < 8u ? s_warp[lane] : 0u;
		#pragma unroll
		for ( uint32_t d = 1u; d < 8u; d <<= 1u ) { const uint32_t u = __shfl_up_sync(0xffffffffu, w, d); if ( lane >= d ) w += u; }
		if ( lane < 8u ) s_warp[lane] = w;
	}
	__syncthreads();
	const uint32_t tot1 = s_warp[7];
	const uint32_t off1 = tot0 + (warp > 0u ? s_warp[warp - 1u] : 0u) + (uint32_t)__popc(m1 & ((1u << lane) - 1u));
	if ( a1 != 0u ) active_out[off1] = threadIdx.x + 256u;
	if ( threadIdx.x == 0u ) active_out[EXPERTS] = tot0 + tot1;
	LmDependentRelease();
}

struct D41MxBatchDesc
{
	const uint8_t *act[4];
	const uint8_t *act_scale[4];
	const uint8_t *w[4];
	const uint8_t *ws[4];
	uint16_t *out[4];
	uint32_t in_dim[4];
	uint32_t out_dim[4];
	uint32_t out_stride[4];
	uint32_t out_col[4];
	uint32_t tile_end[4];
};

template<uint32_t STAGES>
static __global__ __launch_bounds__(128, 3)
void D41MxGemmBatch16Kernel(const __grid_constant__ D41MxBatchDesc desc, uint32_t rows)
{
	constexpr uint32_t TILE_M = 16u, TILE_N = 16u, TILE_K = 128u, BLK = 32u;
	constexpr uint32_t A_BYTES = TILE_M * TILE_K, B_BYTES = TILE_N * TILE_K;
	constexpr uint32_t STAGE_BYTES = A_BYTES + B_BYTES + (TILE_M + TILE_N) * 4u;
	__shared__ __align__(128) uint8_t smem[STAGES * STAGE_BYTES];
	const uint32_t nt = blockIdx.x, row_base = blockIdx.y * TILE_M;
	uint32_t grp = 0u;
	if ( nt >= desc.tile_end[0] )
	{
		grp = 1u;
		if ( nt >= desc.tile_end[1] )
		{
			grp = 2u;
			if ( nt >= desc.tile_end[2] ) grp = 3u;
		}
	}
	const uint32_t n_base = (nt - (grp > 0u ? desc.tile_end[grp - 1u] : 0u)) * TILE_N;
	const uint32_t out_dim = desc.out_dim[grp], input_dim = desc.in_dim[grp];
	if ( n_base >= out_dim || row_base >= rows ) return;
	const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
	const uint32_t warp_k = threadIdx.x >> 5u, lane = threadIdx.x & 31u, group = lane >> 2u, quad = lane & 3u;
	const uint8_t *act = desc.act[grp], *act_scale = desc.act_scale[grp];
	const uint8_t *w = desc.w[grp], *ws = desc.ws[grp];
	uint16_t *out_ptr = desc.out[grp];
	const uint32_t out_stride = desc.out_stride[grp], out_col = desc.out_col[grp];
	const uint32_t k_tiles = input_dim / TILE_K, sc_cols = input_dim / BLK;
	LmDependentWait();
	auto swz = [](uint32_t r, uint32_t c) { return r * TILE_K + ((c ^ (r & 7u)) << 4u); };
	const uint32_t ch = threadIdx.x, r_cp = ch >> 3u, c_cp = ch & 7u;
	const uint32_t a_live = row_base + r_cp < rows ? 16u : 0u;
	const uint32_t b_live = n_base + r_cp < out_dim ? 16u : 0u;
	const uint8_t *a_row = act + (uint64_t)(a_live != 0u ? row_base + r_cp : 0u) * input_dim + c_cp * 16u;
	const uint8_t *b_row = w + (uint64_t)(b_live != 0u ? n_base + r_cp : 0u) * input_dim + c_cp * 16u;
	const uint32_t sa_dst = swz(r_cp, c_cp), sb_dst = A_BYTES + sa_dst;
	const uint8_t *sc_src = threadIdx.x < TILE_M ?
		act_scale + (uint64_t)(row_base + threadIdx.x < rows ? row_base + threadIdx.x : rows - 1u) * sc_cols :
		threadIdx.x < TILE_M + TILE_N ?
		ws + (uint64_t)(n_base + (threadIdx.x - TILE_M) < out_dim ? n_base + (threadIdx.x - TILE_M) : out_dim - 1u) * sc_cols :
		(const uint8_t *)0;
	auto issue = [&](uint32_t k)
	{
		const uint32_t st = sbase + (k % STAGES) * STAGE_BYTES;
		asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(st + sa_dst), "l"(a_row + (uint64_t)k * TILE_K), "r"(a_live) : "memory");
		asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(st + sb_dst), "l"(b_row + (uint64_t)k * TILE_K), "r"(b_live) : "memory");
		if ( sc_src != 0 )
			asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(st + A_BYTES + B_BYTES + threadIdx.x * 4u),
				"l"(sc_src + (uint64_t)k * 4u) : "memory");
	};
	float total[2][4];
	#pragma unroll
	for ( uint32_t j = 0u; j < 2u; ++j ) total[j][0] = total[j][1] = total[j][2] = total[j][3] = 0.0f;
	for ( uint32_t p = 0u; p + 1u < STAGES; ++p )
	{
		if ( p < k_tiles ) issue(p);
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	}
	const uint32_t a_ldm_r = ((lane >> 3u) & 1u) * 8u + (lane & 7u), a_ldm_c = warp_k * 2u + (lane >> 4u);
	const uint32_t b_ldm_r = (lane >> 4u) * 8u + (lane & 7u), b_ldm_c = warp_k * 2u + ((lane >> 3u) & 1u);
	const uint32_t a_ldm_off = swz(a_ldm_r, a_ldm_c), b_ldm_off = A_BYTES + swz(b_ldm_r, b_ldm_c);
	const uint32_t sa_idx = group + ((lane & 1u) << 3u), sb0_idx = TILE_M + group, sb1_idx = TILE_M + 8u + group;
	for ( uint32_t k = 0u; k < k_tiles; ++k )
	{
		asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2u) : "memory");
		__syncthreads();
		if ( k + STAGES - 1u < k_tiles ) issue(k + STAGES - 1u);
		asm volatile("cp.async.commit_group;\n" ::: "memory");
		const uint32_t st = sbase + (k % STAGES) * STAGE_BYTES;
		const uint32_t *words = (const uint32_t *)(smem + (k % STAGES) * STAGE_BYTES + A_BYTES + B_BYTES);
		const uint32_t scale_a = words[sa_idx], scale_b0 = words[sb0_idx], scale_b1 = words[sb1_idx];
		uint32_t a[4], b0[2], b1[2];
		asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
			: "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(st + a_ldm_off));
		asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
			: "=r"(b0[0]), "=r"(b0[1]), "=r"(b1[0]), "=r"(b1[1]) : "r"(st + b_ldm_off));
		float blk0[4], blk1[4];
		LmMmaMxf8Block(blk0, a, b0, scale_a, (uint16_t)warp_k, scale_b0, (uint16_t)warp_k);
		LmMmaMxf8Block(blk1, a, b1, scale_a, (uint16_t)warp_k, scale_b1, (uint16_t)warp_k);
		total[0][0] += blk0[0]; total[0][1] += blk0[1]; total[0][2] += blk0[2]; total[0][3] += blk0[3];
		total[1][0] += blk1[0]; total[1][1] += blk1[1]; total[1][2] += blk1[2]; total[1][3] += blk1[3];
	}
	asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	float *ep = (float *)smem;
	if ( warp_k != 0u )
	{
		#pragma unroll
		for ( uint32_t j = 0u; j < 2u; ++j )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; ++e )
				ep[(((warp_k - 1u) * 2u + j) * 4u + e) * 32u + lane] = total[j][e];
	}
	__syncthreads();
	if ( warp_k == 0u )
	{
		#pragma unroll
		for ( uint32_t j = 0u; j < 2u; ++j )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; e += 2u )
			{
				const uint32_t r = row_base + group + (e >> 1u) * 8u;
				const uint32_t col = n_base + j * 8u + quad * 2u;
				if ( r < rows && col < out_dim )
				{
					const float v0 = total[j][e] + ep[((0u * 2u + j) * 4u + e) * 32u + lane]
						+ ep[((1u * 2u + j) * 4u + e) * 32u + lane] + ep[((2u * 2u + j) * 4u + e) * 32u + lane];
					const float v1 = total[j][e + 1u] + ep[((0u * 2u + j) * 4u + e + 1u) * 32u + lane]
						+ ep[((1u * 2u + j) * 4u + e + 1u) * 32u + lane] + ep[((2u * 2u + j) * 4u + e + 1u) * 32u + lane];
					*(uint32_t *)(out_ptr + (uint64_t)r * out_stride + out_col + col) = LmPackBf16Pair(v0, v1);
				}
			}
	}
	LmDependentRelease();
}

template<uint32_t WARP_CHUNKS, uint32_t STAGES, uint32_t EXPERTS, uint32_t SLICE, uint32_t INTER, uint32_t OUT_DIM>
static __global__ __launch_bounds__(256, 2)
void D41CellMxW2DirectKernel(const uint8_t *__restrict__ w2_cells, const uint16_t *__restrict__ recv_base,
	uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ active_experts,
	uint32_t routes, uint64_t recv_stride)
{
	constexpr uint32_t THREADS = 256u, TILE_M = 2u * WARP_CHUNKS * 16u, COPIES = TILE_M / 32u;
	constexpr uint32_t TILE_K = 128u, K_STEPS = 4u, K_ITERS = INTER / TILE_K;
	constexpr uint32_t TOTAL_CELLS = OUT_DIM / 16u, N_BLOCKS = TOTAL_CELLS / 8u;
	constexpr uint32_t CELL_VECS = 17u, CELL_BYTES = CELL_VECS * 16u;
	constexpr uint32_t TILE_VECS = 8u * CELL_VECS, TILE_BYTES = 8u * CELL_BYTES;
	constexpr uint32_t W_VECS = K_STEPS * TILE_VECS, W_BYTES = K_STEPS * TILE_BYTES;
	constexpr uint32_t W_PASSES = (W_VECS + THREADS - 1u) / THREADS;
	constexpr uint32_t A_BYTES = TILE_M * TILE_K, STAGE_BYTES = A_BYTES + W_BYTES;
	constexpr uint32_t S_WORDS = 2u * TILE_M;
	constexpr uint64_t TILE_STRIDE = (uint64_t)TOTAL_CELLS * CELL_BYTES;
	constexpr uint64_t GROUP_BYTES = (uint64_t)(INTER / 32u) * TILE_STRIDE;
	constexpr uint32_t S_BLOCKS = SLICE / SPARK_DSV41_FLASH_FP8_BLOCK;
	static_assert(OUT_DIM % 128u == 0u && INTER % 128u == 0u && SLICE % 64u == 0u && W_PASSES == 3u && TILE_M % 32u == 0u, "direct w2 recv geometry");
	__shared__ __align__(128) uint8_t smem[STAGES * STAGE_BYTES + S_WORDS * 4u];
	const uint32_t active_idx = blockIdx.x / N_BLOCKS, n_block = blockIdx.x - active_idx * N_BLOCKS;
	LmDependentWait();
	LmDependentRelease();
	const uint32_t active_count = __ldg(active_experts + EXPERTS);
	if ( active_idx >= active_count )
		return;
	const uint32_t group = __ldg(active_experts + active_idx);
	const uint32_t start = __ldg(group_row_offset + group), end = __ldg(group_row_offset + group + 1u);
	if ( start >= end )
		return;
	const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
	uint32_t *s_scratch = (uint32_t *)(smem + STAGES * STAGE_BYTES);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, g = lane >> 2u, c = lane & 3u;
	const uint32_t warp_m = warp >> 2u, warp_n = warp & 3u;
	const uint32_t chunk = threadIdx.x & 7u, row_base = threadIdx.x >> 3u;
	const uint32_t warp_row_base = warp_m * (WARP_CHUNKS * 16u);
	const uint32_t block_cell0 = n_block * 8u, cell_first = block_cell0 + warp_n * 2u;
	const uint8_t *group_cells = w2_cells + (uint64_t)group * GROUP_BYTES + (uint64_t)block_cell0 * CELL_BYTES;
	uint32_t w_off[W_PASSES];
	#pragma unroll
	for ( uint32_t p = 0u; p < W_PASSES; ++p )
	{
		const uint32_t v = threadIdx.x + p * THREADS;
		if ( v < W_VECS )
		{
			const uint32_t kt = v / TILE_VECS, in_kt = v - kt * TILE_VECS;
			w_off[p] = (uint32_t)((uint64_t)kt * TILE_STRIDE + in_kt * 16u);
		}
		else
			w_off[p] = 0u;
	}
	const uint32_t shfl_lo = (lane & ~3u) | (c >> 1u), shfl_hi = shfl_lo | 2u;
	const uint32_t perm_sel = (c & 1u) != 0u ? 0x7632u : 0x5410u;
	const uint32_t ldm_row = warp_row_base + ((lane >> 3u) & 1u) * 8u + (lane & 7u), ldm_col = lane >> 4u;
	const uint32_t s_row_idx = warp_row_base + g + ((lane & 1u) << 3u);
	const uint32_t warp_w_off = warp_n * 2u * CELL_BYTES;
	const uint32_t w_lane_off = warp_w_off + lane * 4u, ws_lane_off = warp_w_off + 256u + g;
	const uint8_t *r0 = (const uint8_t *)(recv_base + 0ull * recv_stride);
	const uint8_t *r1 = (const uint8_t *)(recv_base + 1ull * recv_stride);
	const uint8_t *r2 = (const uint8_t *)(recv_base + 2ull * recv_stride);
	const uint8_t *r3 = (const uint8_t *)(recv_base + 3ull * recv_stride);
	const uint64_t sc_off_base = (uint64_t)routes * SLICE;
	for ( uint32_t first = start; first < end; first += TILE_M )
	{
		const uint32_t count = end - first < TILE_M ? end - first : TILE_M;
		const uint32_t sc_route = first + threadIdx.x, sc_live = threadIdx.x < count ? 1u : 0u;
		const uint8_t *w_base = group_cells;
		float acc[WARP_CHUNKS][4][4];
		#pragma unroll
		for ( uint32_t m = 0u; m < WARP_CHUNKS; ++m )
			#pragma unroll
			for ( uint32_t f = 0u; f < 4u; ++f ) acc[m][f][0] = acc[m][f][1] = acc[m][f][2] = acc[m][f][3] = 0.0f;
		auto produce = [&](uint32_t k_idx, uint32_t st_off)
		{
			const uint32_t stage_s = sbase + st_off, w_stage = stage_s + A_BYTES;
			const uint32_t elem = k_idx * TILE_K + chunk * 16u;
			const uint32_t src_r = elem / SLICE, within = elem - src_r * SLICE;
			const uint8_t *rb = src_r == 0u ? r0 : src_r == 1u ? r1 : src_r == 2u ? r2 : r3;
			#pragma unroll
			for ( uint32_t q = 0u; q < COPIES; ++q )
			{
				const uint32_t lr = row_base + q * 32u, live = lr < count ? 1u : 0u;
				LmCellMxCopy(stage_s + LmCellMxSwizzle(lr, chunk), live != 0u ? rb + (uint64_t)(first + lr) * SLICE + within : rb, live << 4u);
			}
			#pragma unroll
			for ( uint32_t p = 0u; p < W_PASSES; ++p )
				if ( threadIdx.x + p * THREADS < W_VECS )
					LmCellMxCopy(w_stage + (threadIdx.x + p * THREADS) * 16u, w_base + w_off[p], 16u);
			w_base += (uint64_t)K_STEPS * TILE_STRIDE;
		};
		uint32_t kp = 0u, sp_off = 0u, sc_off = 0u;
		auto advance = [](uint32_t &off) { off += STAGE_BYTES; if ( off == STAGES * STAGE_BYTES ) off = 0u; };
		#pragma unroll
		for ( uint32_t s = 0u; s + 1u < STAGES; ++s )
		{
			if ( kp < K_ITERS ) produce(kp, sp_off);
			LmCellMmaCommit();
			++kp;
			advance(sp_off);
		}
		const uint32_t active_m = warp_row_base < count ? 1u : 0u;
		for ( uint32_t k_step = 0u; k_step < K_ITERS; ++k_step )
		{
			const uint32_t s_buf = (k_step & 1u) * TILE_M;
			if ( threadIdx.x < TILE_M )
			{
				uint32_t sw = 0u;
				if ( sc_live != 0u )
				{
					const uint32_t b0 = k_step * 4u, sr0 = b0 / S_BLOCKS, sb0 = b0 - sr0 * S_BLOCKS;
					const uint32_t b2 = b0 + 2u, sr2 = b2 / S_BLOCKS, sb2 = b2 - sr2 * S_BLOCKS;
					const uint8_t *rb0 = sr0 == 0u ? r0 : sr0 == 1u ? r1 : sr0 == 2u ? r2 : r3;
					const uint8_t *rb2 = sr2 == 0u ? r0 : sr2 == 1u ? r1 : sr2 == 2u ? r2 : r3;
					const uint32_t lo = *(const uint16_t *)(rb0 + sc_off_base + (uint64_t)sc_route * S_BLOCKS + sb0);
					const uint32_t hi = *(const uint16_t *)(rb2 + sc_off_base + (uint64_t)sc_route * S_BLOCKS + sb2);
					sw = lo | (hi << 16u);
				}
				s_scratch[s_buf + threadIdx.x] = sw;
			}
			LmCellMmaWait<STAGES - 2u>();
			__syncthreads();
			if ( kp < K_ITERS )
			{
				produce(kp, sp_off);
				++kp;
				advance(sp_off);
			}
			LmCellMmaCommit();
			if ( active_m != 0u )
			{
				const uint32_t stage_s = sbase + sc_off, w_stage = stage_s + A_BYTES;
				uint32_t scale_word[WARP_CHUNKS];
				#pragma unroll
				for ( uint32_t m = 0u; m < WARP_CHUNKS; ++m )
					scale_word[m] = s_scratch[s_buf + s_row_idx + m * 16u];
				#pragma unroll 1
				for ( uint32_t step = 0u; step < K_STEPS; ++step )
				{
					const uint32_t w_tile = w_stage + step * TILE_BYTES + w_lane_off;
					const uint32_t ws_tile = w_stage + step * TILE_BYTES + ws_lane_off;
					const uint32_t a_col = step * 2u + ldm_col;
					uint32_t b[4][2], scale_b[4], a[2][4];
					#pragma unroll
					for ( uint32_t f = 0u; f < 4u; ++f )
					{
						const uint32_t raw = LmCellMxShared32(w_tile + (f / 2u) * CELL_BYTES + (f % 2u) * 128u);
						LmCellMxE2m1(__byte_perm(__shfl_sync(0xffffffffu, raw, shfl_lo), __shfl_sync(0xffffffffu, raw, shfl_hi), perm_sel), b[f][0], b[f][1]);
						scale_b[f] = LmCellMxShared8(ws_tile + (f / 2u) * CELL_BYTES + (f % 2u) * 8u);
					}
					LmCellMxMatrix4(stage_s + LmCellMxSwizzle(ldm_row, a_col), a[0]);
					#pragma unroll
					for ( uint32_t m = 0u; m < WARP_CHUNKS; ++m )
					{
						const uint32_t cur = m & 1u, nxt = cur ^ 1u;
						if ( m + 1u < WARP_CHUNKS )
							LmCellMxMatrix4(stage_s + LmCellMxSwizzle(ldm_row + (m + 1u) * 16u, a_col), a[nxt]);
						#pragma unroll
						for ( uint32_t f = 0u; f < 4u; ++f )
						{
							float blk[4];
							LmMmaMxf8Mxf4Block<0u>(blk, a[cur], b[f], scale_word[m], scale_b[f]);
							acc[m][f][0] += blk[0]; acc[m][f][1] += blk[1]; acc[m][f][2] += blk[2]; acc[m][f][3] += blk[3];
						}
						scale_word[m] >>= 8u;
					}
				}
			}
			advance(sc_off);
		}
		LmCellMmaWait<0u>();
		if ( active_m != 0u )
		{
			#pragma unroll
			for ( uint32_t m = 0u; m < WARP_CHUNKS; ++m )
				#pragma unroll
				for ( uint32_t h = 0u; h < 2u; ++h )
				{
					const uint32_t lr = warp_row_base + m * 16u + g + h * 8u;
					if ( lr >= count ) continue;
					uint16_t *row_out = output_bf16 + (uint64_t)(first + lr) * OUT_DIM;
					#pragma unroll
					for ( uint32_t f = 0u; f < 4u; ++f )
						*(uint32_t *)(row_out + (cell_first + (f >> 1u)) * 16u + (f & 1u) * 8u + c * 2u) =
							LmPackBf16Pair(acc[m][f][h * 2u], acc[m][f][h * 2u + 1u]);
				}
		}
		if ( first + TILE_M < end ) __syncthreads();
	}
}


