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

static inline SparkGpuError D41Fp8SliceLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint8_t *payload, const uint8_t *block_scale,
	uint8_t *weight, uint8_t *row_scale, uint16_t *wide, uint32_t columns, uint32_t row_first, uint32_t column_first, uint32_t output_columns)
{ return D41Skl2Launch(D41SKL2_d41_fp8_slice_256, grid, block, 0, stream, payload, block_scale, weight, row_scale, wide, columns, row_first, column_first, output_columns); }

static inline SparkGpuError D41CompressorRowsLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const float *kv, const float *score, uint16_t *pooled,
	uint32_t *emit_row, float *state, const uint32_t *positions, const uint32_t *row_lane, const uint32_t *row_ordinal,
	const uint32_t *sequence_row_begin, const uint32_t *sequence_row_indices, const uint32_t *recurrent_slot, uint32_t ratio,
	uint32_t state_stride_floats, uint32_t width, const uint32_t *slots = 0, const uint32_t *parent_rows = 0,
	const uint16_t *norm = 0, float norm_eps = 0.0f, uint32_t save_inline = 0u)
{
	return D41Skl2Launch(D41SKL2_d41_compressor_rows_256, grid, block, 0, stream, kv, score, pooled, emit_row, state, positions, row_lane, row_ordinal,
		sequence_row_begin, sequence_row_indices, recurrent_slot, ratio, state_stride_floats, width, slots, parent_rows, norm, norm_eps, save_inline);
}

static inline SparkGpuError D41CompressorSaveLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const float *kv, const float *score, float *state,
	const uint32_t *positions, const uint32_t *row_lane, const uint32_t *row_ordinal, const uint32_t *sequence_row_begin, const uint32_t *recurrent_slot,
	uint32_t rows, uint32_t ratio, uint32_t state_stride_floats, uint32_t width, uint32_t commit, const uint32_t *lane_count, const uint32_t *path_rows = 0)
{ return D41Skl2Launch(D41SKL2_d41_compressor_save_256, grid, block, 0, stream, kv, score, state, positions, row_lane, row_ordinal, sequence_row_begin, recurrent_slot, rows, ratio, state_stride_floats, width, commit, lane_count, path_rows); }

static inline SparkGpuError D41CompressedTreeMapLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint32_t *fork_positions,
	const uint32_t *slot_offsets, uint32_t *compressed_fork, uint32_t *compressed_offset, uint32_t rows, uint32_t ratio)
{ return D41Skl2Launch(D41SKL2_d41_compressed_tree_map_256, grid, block, 0, stream, fork_positions, slot_offsets, compressed_fork, compressed_offset, rows, ratio); }

template<class Geometry>
static inline SparkGpuError D41CompressedPathMoveLaunch(dim3 grid, dim3 block, SparkGpuStream stream, LmKvShardView view,
	const uint32_t *sequence_of_row, const uint32_t *positions, const uint32_t *path_rows, const uint32_t *path_count,
	const uint32_t *slots, uint32_t ratio, uint32_t elements)
{
	return D41Skl2Launch(D41SKL2_d41_compressed_path_move_256, grid, block, 0, stream, view, sequence_of_row, positions, path_rows, path_count,
		slots, ratio, elements, (uint32_t)Geometry::kPageSlots, (uint32_t)Geometry::kSlotBytes);
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

static inline SparkGpuError D41CompressedEmitPositionLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint32_t *positions,
	uint32_t *bound_position, uint32_t rows, uint32_t ratio)
{ return D41Skl2Launch(D41SKL2_d41_compressed_emit_position_256, grid, block, 0, stream, positions, bound_position, rows, ratio); }

static inline SparkGpuError D41CompressedLengthLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint32_t *context_length,
	uint32_t *compressed_length, uint32_t sequences, uint32_t ratio)
{ return D41Skl2Launch(D41SKL2_d41_compressed_length_256, grid, block, 0, stream, context_length, compressed_length, sequences, ratio); }

template<class Geometry>
static inline SparkGpuError D41StoreCompressedLaunch(dim3 grid, dim3 block, SparkGpuStream stream, LmKvShardView view,
	const uint16_t *latent, const uint16_t *index_key, const uint32_t *emit_row, const uint32_t *sequence_of_row, uint32_t index_width)
{
	return D41Skl2Launch(D41SKL2_d41_store_compressed_256, grid, block, 0, stream, view, latent, index_key, emit_row, sequence_of_row,
		index_width, (uint32_t)Geometry::kPageSlots, (uint32_t)Geometry::kSlotBytes);
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

static inline SparkGpuError D41GatherPlanLaunch(SparkGpuStream stream, uint32_t *plan, uint32_t capacity, const uint32_t *sequence_of_row,
	const uint32_t *length)
{ return D41Skl2Launch(D41SKL2_d41_gather_plan_32, 1u, 32u, 0, stream, plan, capacity, sequence_of_row, length); }

static inline SparkGpuError D41RingPackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *rings, const uint32_t *recurrent_slot,
	uint64_t slot_stride, uint32_t ring_elements, uint16_t *send)
{ return D41Skl2Launch(D41SKL2_d41_ring_pack_256, grid, block, 0, stream, rings, recurrent_slot, slot_stride, ring_elements, send); }

static inline void D41InitRecords(float *records, uint64_t destination_stride, uint32_t rows, uint32_t heads_per_destination, uint32_t destinations,
	SparkGpuStream stream)
{
	const uint32_t span = rows * heads_per_destination * D41K_RECORD, blocks = (span + D41K_THREADS - 1u) / D41K_THREADS;
	D41Skl2Launch(D41SKL2_d41_init_records_256, blocks < 1024u ? (blocks == 0u ? 1u : blocks) : 1024u, D41K_THREADS, 0, stream, records, destination_stride, span, destinations);
}

#define D41K_WINDOW_KEYS 128u
#define D41K_WINDOW_BATCH 16u

static inline SparkGpuError D41WindowFoldLaunch(dim3 grid, dim3 block, SparkGpuStream stream, float *records, uint64_t destination_stride,
	uint32_t heads_per_destination, const uint16_t *query, uint64_t query_rank_stride, const uint16_t *wave_kv, const uint16_t *ring,
	const uint32_t *positions, const uint32_t *row_lane, const uint32_t *row_ordinal, const uint32_t *sequence_row_begin,
	const uint32_t *sequence_row_indices, const uint32_t *recurrent_slot, uint64_t ring_slot_stride, uint32_t ring_rows, uint32_t window,
	uint32_t degree, uint32_t rank, float qk_scale, uint32_t record_heads, uint32_t record_copies, const uint32_t *parent_rows = 0)
{
	return D41Skl2Launch(D41SKL2_d41_window_fold_256, grid, block, 0, stream, records, destination_stride, heads_per_destination, query,
		query_rank_stride, wave_kv, ring, positions, row_lane, row_ordinal, sequence_row_begin, sequence_row_indices, recurrent_slot,
		ring_slot_stride, ring_rows, window, degree, rank, qk_scale, record_heads, record_copies, parent_rows);
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

static inline SparkGpuError D41WindowFoldWholeLaunch(dim3 grid, dim3 block, SparkGpuStream stream, float *records, uint32_t heads,
	const uint16_t *query, const uint16_t *wave_kv, const uint16_t *ring, const uint32_t *positions, const uint32_t *row_lane,
	const uint32_t *row_ordinal, const uint32_t *sequence_row_begin, const uint32_t *sequence_row_indices, const uint32_t *recurrent_slot,
	uint64_t ring_slot_stride, uint32_t window, float qk_scale)
{
	return D41Skl2Launch(D41SKL2_d41_window_fold_whole_256, grid, block, 0, stream, records, heads, query, wave_kv, ring, positions, row_lane,
		row_ordinal, sequence_row_begin, sequence_row_indices, recurrent_slot, ring_slot_stride, window, qk_scale);
}

template<uint32_t BLOCK_HEADS = 4u>
static inline SparkGpuError D41WindowFoldWholeTiledLaunch(dim3 grid, dim3 block, SparkGpuStream stream, float *records, uint32_t heads,
	const uint16_t *query, const uint16_t *wave_kv, const uint16_t *ring, const uint32_t *positions, const uint32_t *row_lane,
	const uint32_t *row_ordinal, const uint32_t *sequence_row_begin, const uint32_t *sequence_row_indices, const uint32_t *recurrent_slot,
	uint64_t ring_slot_stride, uint32_t window, float qk_scale)
{
	static_assert(BLOCK_HEADS == 4u, "tiled window fold skl2 supports 4 heads per block");
	return D41Skl2Launch(D41SKL2_d41_window_fold_whole_tiled4_256, grid, block, 0, stream, records, heads, query, wave_kv, ring, positions,
		row_lane, row_ordinal, sequence_row_begin, sequence_row_indices, recurrent_slot, ring_slot_stride, window, qk_scale);
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
static inline SparkGpuError D41WindowFoldSpreadLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const float *records, uint32_t has_carried,
	const uint16_t *query, const uint16_t *wave_kv, uint16_t *ring, const uint32_t *positions, const uint32_t *row_lane,
	const uint32_t *row_ordinal, const uint32_t *sequence_row_begin, const uint32_t *sequence_row_indices, const uint32_t *recurrent_slot,
	uint64_t ring_slot_stride, uint32_t ring_rows, uint32_t window, float qk_scale, const float *sink, const float2 *rope_sc,
	uint16_t *output, uint32_t qdq, uint32_t rows, uint32_t commit)
{
	static_assert(HEADS % BLOCK_HEADS == 0u && BLOCK_HEADS == 2u, "window fold spread skl2 processes two heads per block");
	return D41Skl2Launch(D41SKL2_d41_window_fold_spread_256, grid, block, 0, stream, records, has_carried, HEADS, query, wave_kv, ring, positions,
		row_lane, row_ordinal, sequence_row_begin, sequence_row_indices, recurrent_slot, ring_slot_stride, ring_rows, window, qk_scale,
		sink, rope_sc, output, qdq, rows, commit);
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

static inline SparkGpuError D41SwigluRowsLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *gate, const uint16_t *up, uint16_t *output, uint32_t width, float limit)
{ return D41Skl2Launch(D41SKL2_d41_swiglu_rows_256, grid, block, 0, stream, gate, up, output, width, limit); }

static inline SparkGpuError D41SwigluQdqRowsLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *gate, const uint16_t *up, uint16_t *output, uint32_t width, float limit)
{ return D41Skl2Launch(D41SKL2_d41_swiglu_qdq_rows_256, grid, block, 0, stream, gate, up, output, width, limit); }

static inline SparkGpuError D41SwigluMxLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *gate, const uint16_t *up, uint8_t *codes, uint8_t *scales, uint32_t width, float limit)
{ return D41Skl2Launch(D41SKL2_d41_swiglu_mx_256, grid, block, 0, stream, gate, up, codes, scales, width, limit); }

static inline SparkGpuError D41RankMajorMxLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *gathered, uint8_t *codes, uint8_t *scales, uint32_t rows, uint32_t width, uint32_t ranks, uint32_t first_rank)
{ return D41Skl2Launch(D41SKL2_d41_rank_major_mx_256, grid, block, 0, stream, gathered, codes, scales, rows, width, ranks, first_rank); }

static inline SparkGpuError D41RankMajorQdqLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *gathered, uint16_t *target, uint32_t rows, uint32_t width, uint32_t ranks, uint32_t first_rank)
{ return D41Skl2Launch(D41SKL2_d41_rank_major_qdq_256, grid, block, 0, stream, gathered, target, rows, width, ranks, first_rank); }

template<uint32_t STREAMS, uint32_t MODE>
static inline SparkGpuError D41HcPostFusedLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const void *gathered, const float *post, const float *comb, uint16_t *streams, uint32_t rows, uint32_t width, uint32_t degree, uint32_t slice_width)
{
	static_assert(STREAMS == 4u && MODE >= 1u && MODE <= 3u, "hc post fused skl2 supports 4 streams and modes 1..3");
	const uint32_t id = MODE == 1u ? (uint32_t)D41SKL2_d41_hc_post_fused_m1_256 : MODE == 2u ? (uint32_t)D41SKL2_d41_hc_post_fused_m2_256 : (uint32_t)D41SKL2_d41_hc_post_fused_m3_256;
	return D41Skl2Launch(id, grid, block, 0, stream, gathered, post, comb, streams, rows, width, degree, slice_width);
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

static inline SparkGpuError D41IndexPackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *query, const uint16_t *weight, uint16_t *packed, uint32_t query_width, uint32_t weight_width, uint32_t pack_width)
{ return D41Skl2Launch(D41SKL2_d41_index_pack_256, grid, block, 0, stream, query, weight, packed, query_width, weight_width, pack_width); }

static inline SparkGpuError D41IndexUnpackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *packed, uint64_t rank_stride, uint16_t *query, uint16_t *weight, uint32_t query_width, uint32_t weight_width, uint32_t pack_width)
{ return D41Skl2Launch(D41SKL2_d41_index_unpack_256, grid, block, 0, stream, packed, rank_stride, query, weight, query_width, weight_width, pack_width); }

static inline dim3 D41ItemGrid(uint32_t blocks, uint32_t items) { const uint32_t span = items < D41K_GRID_ITEMS ? (items != 0u ? items : 1u) : D41K_GRID_ITEMS; return dim3(blocks, span, (items + span - 1u) / span); }

static inline SparkGpuError D41MoeSendLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *gate_up, const float *packed_weight, const uint16_t *shared, uint16_t *send, uint32_t routes, uint32_t rows, uint32_t slice, uint32_t hidden_slice, uint32_t degree, uint64_t destination_stride, float limit, uint32_t route_elements, uint32_t codes)
{ return D41Skl2Launch(D41SKL2_d41_moe_send_256, grid, block, 0, stream, gate_up, packed_weight, shared, send, routes, rows, slice, hidden_slice, degree, destination_stride, limit, route_elements, codes); }

static inline SparkGpuError D41ActAssembleMxLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *received, uint8_t *codes, uint32_t slice, uint32_t width, uint64_t source_stride, uint32_t routes)
{ return D41Skl2Launch(D41SKL2_d41_act_assemble_mx_256, grid, block, 0, stream, received, codes, slice, width, source_stride, routes); }

static inline SparkGpuError D41ActAssembleQdqLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *received, uint16_t *act, uint32_t slice, uint32_t width, uint64_t source_stride, uint32_t routes, uint32_t codes)
{ return D41Skl2Launch(D41SKL2_d41_act_assemble_qdq_256, grid, block, 0, stream, received, act, slice, width, source_stride, routes, codes); }

static inline SparkGpuError D41MoeSendVecLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *gate_up, const float *packed_weight, const uint16_t *shared, uint16_t *send, uint32_t routes, uint32_t rows, uint32_t slice, uint32_t hidden_slice, uint32_t degree, uint64_t destination_stride, float limit, uint32_t route_elements, uint32_t codes)
{ return D41Skl2Launch(D41SKL2_d41_moe_send_vec_256, grid, block, 0, stream, gate_up, packed_weight, shared, send, routes, rows, slice, hidden_slice, degree, destination_stride, limit, route_elements, codes); }

static inline SparkGpuError D41ActAssembleQdqVecLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *received, uint16_t *act, uint32_t slice, uint32_t width, uint64_t source_stride, uint32_t routes, uint32_t codes)
{ return D41Skl2Launch(D41SKL2_d41_act_assemble_qdq_vec_256, grid, block, 0, stream, received, act, slice, width, source_stride, routes, codes); }

static inline SparkGpuError D41ActAssembleCodesLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *received, uint8_t *codes, uint32_t slice, uint32_t width, uint64_t source_stride, uint32_t routes)
{ return D41Skl2Launch(D41SKL2_d41_act_assemble_codes_256, grid, block, 0, stream, received, codes, slice, width, source_stride, routes); }

static inline SparkGpuError D41SharedSendLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *gate, const uint16_t *up, uint16_t *send, uint32_t rows, uint32_t width, uint32_t degree, uint64_t destination_stride, float limit)
{ return D41Skl2Launch(D41SKL2_d41_shared_send_256, grid, block, 0, stream, gate, up, send, rows, width, degree, destination_stride, limit); }

static inline SparkGpuError D41SharedDequantAllLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *received, uint16_t *act, uint32_t rows, uint64_t stride, uint32_t degree, uint32_t total_granules, uint32_t granule)
{ return D41Skl2Launch(D41SKL2_d41_shared_dequant_all_256, grid, block, 0, stream, received, act, rows, stride, degree, total_granules, granule); }

static inline SparkGpuError D41L2PrefetchLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint8_t *ptr0, uint64_t bytes0, const uint8_t *ptr1, uint64_t bytes1)
{ return D41Skl2Launch(D41SKL2_d41_l2_prefetch_64, grid, block, 0, stream, ptr0, bytes0, ptr1, bytes1); }

static inline SparkGpuError D41WaveRowIndicesLaunch(dim3 grid, dim3 block, SparkGpuStream stream, uint32_t *indices, uint32_t count)
{ return D41Skl2Launch(D41SKL2_d41_wave_row_indices_256, grid, block, 0, stream, indices, count); }

static inline SparkGpuError D41RmsNormQdqLaunch(dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, const uint16_t *input,
	const uint16_t *streams, const float *pre, const uint16_t *weight, uint16_t *output, uint16_t *quantized, uint32_t dimension,
	float epsilon, uint8_t *mx_codes = 0, uint8_t *mx_scales = 0)
{
	return D41Skl2Launch(D41SKL2_d41_rms_norm_qdq_256, grid, block, shared, stream, input, streams, pre, weight, output, quantized,
		dimension, epsilon, mx_codes, mx_scales);
}

static inline SparkGpuError D41QKvNormRopeQdqLaunch(dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, const uint16_t *q_in,
	const uint16_t *q_w, uint16_t *q_qdq, uint8_t *mx_codes, uint8_t *mx_scales, uint32_t q_dim, uint32_t q_parts, const uint16_t *kv_in,
	const uint16_t *kv_w, const uint32_t *positions, const float *inv_freq, uint16_t *kv_out, uint32_t kv_dim, uint32_t rope_offset,
	float epsilon, const float2 *rope_sc = 0)
{
	return D41Skl2Launch(D41SKL2_d41_qkv_norm_rope_qdq_256, grid, block, shared, stream, q_in, q_w, q_qdq, mx_codes, mx_scales, q_dim, q_parts,
		kv_in, kv_w, positions, inv_freq, kv_out, kv_dim, rope_offset, epsilon, rope_sc);
}

static inline SparkGpuError D41SharedDequantLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint8_t *bytes, uint16_t *act, uint32_t rows, uint32_t width)
{ return D41Skl2Launch(D41SKL2_d41_shared_dequant_256, grid, block, 0, stream, bytes, act, rows, width); }

static inline SparkGpuError D41FfnPackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint32_t *route_expert, const float *route_weight, uint8_t *block_out, uint32_t own, uint32_t routed, uint32_t hidden, uint32_t top_k)
{ return D41Skl2Launch(D41SKL2_d41_ffn_pack_256, grid, block, 0, stream, route_expert, route_weight, block_out, own, routed, hidden, top_k); }

static inline SparkGpuError D41FfnUnpackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint8_t *received, uint16_t *xq, uint8_t *mx_codes, uint8_t *mx_scales, uint32_t *route_expert, float *route_weight, uint32_t own, uint32_t hidden, uint32_t top_k, uint64_t block_bytes)
{ return D41Skl2Launch(D41SKL2_d41_ffn_unpack_256, grid, block, 0, stream, received, xq, mx_codes, mx_scales, route_expert, route_weight, own, hidden, top_k, block_bytes); }

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
static inline SparkGpuError D41ExchangeCountsLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint32_t *route_expert, uint32_t rows,
	uint32_t first, uint32_t top_k, const uint8_t *owner, uint32_t backups, D41ExchangeCounts<RANKS, EXPERTS> *counts)
{
	static_assert((RANKS == 4u || RANKS == 16u) && EXPERTS == 384u, "exchange counts skl2 supports 4 or 16 ranks and 384 experts");
	return D41Skl2Launch(RANKS == 4u ? D41SKL2_d41_exchange_counts_r4_e384 : D41SKL2_d41_exchange_counts_r16_e384, grid, block, 0, stream,
		route_expert, rows, first, top_k, owner, backups, (uint8_t *)counts);
}

template <uint32_t RANKS, uint32_t EXPERTS>
static inline SparkGpuError D41ExchangeTotalsLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint8_t *gathered, uint64_t stride, uint32_t backups, D41ExchangeTotals<RANKS, EXPERTS> *totals)
{
	static_assert((RANKS == 4u || RANKS == 16u) && EXPERTS == 384u, "exchange totals skl2 supports 4 or 16 ranks and 384 experts");
	return D41Skl2Launch(RANKS == 4u ? D41SKL2_d41_exchange_totals_r4_e384 : D41SKL2_d41_exchange_totals_r16_e384, grid, block, 0, stream, gathered, stride, backups, totals);
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
static inline SparkGpuError D41ExchangeSlotsLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint32_t *route_expert, uint32_t rows,
	const uint8_t *owner, const uint8_t *backup, const uint16_t *quota, uint32_t capacity, uint32_t *slot, uint32_t *count)
{
	static_assert((RANKS == 4u || RANKS == 16u) && EXPERTS == 384u && TOPK == 6u && (COPIES == 1u || COPIES == 2u), "exchange slots skl2 supports r4/r16 e384 k6 c1/c2");
	return D41Skl2Launch(RANKS == 4u ? D41SKL2_d41_exchange_slots_r4_e384 : D41SKL2_d41_exchange_slots_r16_e384, grid, block, 0, stream,
		route_expert, rows, owner, backup, quota, capacity, slot, count, COPIES);
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

template<uint32_t MAX_R, uint32_t K_SPLITS, uint32_t HIDDEN>
static __device__ __forceinline__ void D41HcMixNormKSplitBody(const uint16_t *__restrict__ streams, const float *__restrict__ fn,
	const float *__restrict__ scale, const float *__restrict__ base, float *__restrict__ partials, float *__restrict__ pre,
	float *__restrict__ post, float *__restrict__ comb, uint32_t *__restrict__ counters, const float *__restrict__ collapse_pre,
	const uint16_t *__restrict__ norm, uint16_t *__restrict__ x, uint16_t *__restrict__ xq, uint8_t *__restrict__ mx_codes,
	uint8_t *__restrict__ mx_scales, uint32_t rows, float rms_eps, uint32_t hc_iters, float hc_eps)
{
	using S = LmHcSinkhornShape<D41K_HC>;
	constexpr uint32_t THREADS = 256u, WARPS = 8u, TASKS_PER_WARP = S::kMix / WARPS;
	constexpr uint32_t FLAT = D41K_HC * HIDDEN, SLICE_K = FLAT / K_SPLITS, ITERS = SLICE_K / 128u;
	constexpr uint32_t STR_VECS = MAX_R * 16u;
	static_assert(S::kMix == 24u && S::kTasks == 25u && FLAT % (K_SPLITS * 128u) == 0u && STR_VECS <= THREADS,
		"ksplit sinkhorn splits hidden evenly into 128-wide chunks");
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
		const uint32_t chunk = blockIdx.x / K_SPLITS, ksplit = blockIdx.x - chunk * K_SPLITS, row0 = chunk * MAX_R;
		const uint32_t live = rows - row0 < MAX_R ? rows - row0 : MAX_R;
		const uint32_t k_base = ksplit * SLICE_K;
		float total[TASKS_PER_WARP][MAX_R], norm_sq[MAX_R];
		#pragma unroll
		for ( uint32_t r = 0u; r < MAX_R; ++r )
		{
			norm_sq[r] = 0.0f;
			#pragma unroll
			for ( uint32_t t = 0u; t < TASKS_PER_WARP; ++t ) total[t][r] = 0.0f;
		}
		if constexpr ( MAX_R == 1u )
		{
			const uint16_t *str_row = streams + (uint64_t)row0 * FLAT + k_base;
			#pragma unroll
			for ( uint32_t iter = 0u; iter < ITERS; ++iter )
			{
				const uint32_t k_off = k_base + iter * 128u;
				const uint2 s = __ldg((const uint2 *)(str_row + iter * 128u) + lane);
				const float v0 = LmBf16ToFloat((uint16_t)(s.x & 0xffffu));
				const float v1 = LmBf16ToFloat((uint16_t)(s.x >> 16u));
				const float v2 = LmBf16ToFloat((uint16_t)(s.y & 0xffffu));
				const float v3 = LmBf16ToFloat((uint16_t)(s.y >> 16u));
				#pragma unroll
				for ( uint32_t t = 0u; t < TASKS_PER_WARP; ++t )
				{
					const float4 ft = __ldcs((const float4 *)(fn + (uint64_t)(warp + t * WARPS) * FLAT + k_off) + lane);
					total[t][0] = fmaf(ft.x, v0, fmaf(ft.y, v1, fmaf(ft.z, v2, fmaf(ft.w, v3, total[t][0]))));
				}
				if ( warp == 0u )
					norm_sq[0] = fmaf(v0, v0, fmaf(v1, v1, fmaf(v2, v2, fmaf(v3, v3, norm_sq[0]))));
			}
		}
		else
		{
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

template<uint32_t MAX_R, uint32_t HIDDEN>
static __global__ __launch_bounds__(256, 2)
void D41HcMixNormK32Kernel(const uint16_t *__restrict__ streams, const float *__restrict__ fn, const float *__restrict__ scale,
	const float *__restrict__ base, float *__restrict__ partials, float *__restrict__ pre, float *__restrict__ post, float *__restrict__ comb,
	uint32_t *__restrict__ counters, const float *__restrict__ collapse_pre, const uint16_t *__restrict__ norm, uint16_t *__restrict__ x,
	uint16_t *__restrict__ xq, uint8_t *__restrict__ mx_codes, uint8_t *__restrict__ mx_scales, uint32_t rows, float rms_eps, uint32_t hc_iters,
	float hc_eps)
{
	D41HcMixNormKSplitBody<MAX_R, 32u, HIDDEN>(streams, fn, scale, base, partials, pre, post, comb, counters, collapse_pre, norm, x, xq,
		mx_codes, mx_scales, rows, rms_eps, hc_iters, hc_eps);
}

template<uint32_t MAX_R, uint32_t HIDDEN>
static __global__ __launch_bounds__(256, 2)
void D41HcMixNormK40Kernel(const uint16_t *__restrict__ streams, const float *__restrict__ fn, const float *__restrict__ scale,
	const float *__restrict__ base, float *__restrict__ partials, float *__restrict__ pre, float *__restrict__ post, float *__restrict__ comb,
	uint32_t *__restrict__ counters, const float *__restrict__ collapse_pre, const uint16_t *__restrict__ norm, uint16_t *__restrict__ x,
	uint16_t *__restrict__ xq, uint8_t *__restrict__ mx_codes, uint8_t *__restrict__ mx_scales, uint32_t rows, float rms_eps, uint32_t hc_iters,
	float hc_eps)
{
	D41HcMixNormKSplitBody<MAX_R, 40u, HIDDEN>(streams, fn, scale, base, partials, pre, post, comb, counters, collapse_pre, norm, x, xq,
		mx_codes, mx_scales, rows, rms_eps, hc_iters, hc_eps);
}

template<uint32_t QLORA>
static inline SparkGpuError D41UnpackQCodesLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint8_t *received, uint8_t *codes, uint8_t *scales, uint32_t own, uint64_t segment)
{ return D41Skl2Launch(D41SKL2_d41_unpack_q_codes_128, grid, block, 0, stream, received, codes, scales, own, segment, QLORA); }

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

static inline SparkGpuError D41ExchangePackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint8_t *codes, const uint8_t *scales, const uint32_t *route_expert, const float *route_weight, const uint32_t *slot, const uint8_t *owner, uint32_t ranks, uint32_t capacity, uint32_t hidden, uint32_t top_k, uint32_t experts, uint64_t record_bytes, uint8_t *send)
{ return D41Skl2Launch(D41SKL2_d41_exchange_pack_256, grid, block, 0, stream, codes, scales, route_expert, route_weight, slot, owner, ranks, capacity, hidden, top_k, experts, record_bytes, send); }

template <uint32_t RANKS>
struct D41ReceivedRows
{
	uint32_t first[RANKS + 1u];
};

template <uint32_t RANKS>
static inline SparkGpuError D41ExchangeUnpackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint8_t *received_bytes, D41ReceivedRows<RANKS> received, uint32_t capacity, uint32_t hidden, uint32_t top_k, uint64_t record_bytes, uint8_t *codes, uint8_t *scales, uint32_t *route_expert, float *route_weight)
{
	static_assert(RANKS == 4u || RANKS == 16u, "exchange unpack skl2 supports 4 or 16 ranks");
	return D41Skl2Launch(RANKS == 4u ? D41SKL2_d41_exchange_unpack_r4 : D41SKL2_d41_exchange_unpack_r16, grid, block, 0, stream, received_bytes, received, capacity, hidden, top_k, record_bytes, codes, scales, route_expert, route_weight);
}

template <uint32_t RANKS>
static inline SparkGpuError D41RecordPlaceLaunch(dim3 grid, dim3 block, SparkGpuStream stream, uint32_t *source_token, uint32_t count, D41ReceivedRows<RANKS> received, uint32_t capacity)
{
	static_assert(RANKS == 4u || RANKS == 16u, "record place skl2 supports 4 or 16 ranks");
	return D41Skl2Launch(RANKS == 4u ? D41SKL2_d41_record_place_r4 : D41SKL2_d41_record_place_r16, grid, block, 0, stream, source_token, count, received, capacity);
}

template <uint32_t RANKS>
static inline SparkGpuError D41ExchangeFinalizeLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *packed, const uint32_t *route_packed_row, const uint32_t *route_expert, D41ReceivedRows<RANKS> received, uint32_t capacity, uint32_t top_k, uint32_t local_experts, uint32_t width, uint32_t skip_single, uint16_t *output)
{
	static_assert(RANKS == 4u || RANKS == 16u, "exchange finalize skl2 supports 4 or 16 ranks");
	return D41Skl2Launch(RANKS == 4u ? D41SKL2_d41_exchange_finalize_r4 : D41SKL2_d41_exchange_finalize_r16, grid, block, 0, stream, packed, route_packed_row, route_expert, received, capacity, top_k, local_experts, width, skip_single, output);
}

template <uint32_t RANKS>
static inline SparkGpuError D41FinalizeMapLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint32_t *route_expert, const uint32_t *route_packed_row, D41ReceivedRows<RANKS> received, uint32_t capacity, uint32_t top_k, uint32_t local_experts, uint32_t rows, uint32_t mapped, uint32_t *output_row)
{
	static_assert(RANKS == 4u || RANKS == 16u, "finalize map skl2 supports 4 or 16 ranks");
	return D41Skl2Launch(RANKS == 4u ? D41SKL2_d41_finalize_map_r4 : D41SKL2_d41_finalize_map_r16, grid, block, 0, stream, route_expert, route_packed_row, received, capacity, top_k, local_experts, rows, mapped, output_row);
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

static inline SparkGpuError D41ShufflePackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint8_t *codes, const uint8_t *scales, const uint32_t *route_expert, const float *route_weight, uint32_t first, uint32_t ranks, uint32_t capacity, uint32_t hidden, uint32_t top_k, uint64_t record_bytes, uint8_t *send)
{ return D41Skl2Launch(D41SKL2_d41_shuffle_pack_256, grid, block, 0, stream, codes, scales, route_expert, route_weight, first, ranks, capacity, hidden, top_k, record_bytes, send); }

template <uint32_t RANKS>
static inline SparkGpuError D41ShuffleUnpackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint8_t *received_bytes, D41ReceivedRows<RANKS> received, uint32_t capacity, uint32_t hidden, uint32_t top_k, uint64_t record_bytes, uint8_t *codes, uint8_t *scales, uint32_t *route_expert, float *route_weight)
{
	static_assert(RANKS == 4u || RANKS == 16u, "shuffle unpack skl2 supports 4 or 16 ranks");
	return D41Skl2Launch(RANKS == 4u ? D41SKL2_d41_shuffle_unpack_r4 : D41SKL2_d41_shuffle_unpack_r16, grid, block, 0, stream, received_bytes, received, capacity, hidden, top_k, record_bytes, codes, scales, route_expert, route_weight);
}

template <uint32_t RANKS>
static inline SparkGpuError D41ShuffleSumLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *landing, const uint32_t *slot, D41ReceivedRows<RANKS> received, uint32_t shuffle, uint32_t hidden, uint16_t *output)
{
	static_assert(RANKS == 4u || RANKS == 16u, "shuffle sum skl2 supports 4 or 16 ranks");
	return D41Skl2Launch(RANKS == 4u ? D41SKL2_d41_shuffle_sum_r4 : D41SKL2_d41_shuffle_sum_r16, grid, block, 0, stream, landing, slot, received, shuffle, hidden, output);
}

static inline SparkGpuError D41ShuffleFinishLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *back, const uint16_t *shared, uint32_t live, uint32_t first, uint32_t ranks, uint32_t shuffle, uint32_t hidden, uint16_t *output)
{ return D41Skl2Launch(D41SKL2_d41_shuffle_finish_256, grid, block, 0, stream, back, shared, live, first, ranks, shuffle, hidden, output); }

template <uint32_t RANKS>
static inline SparkGpuError D41DirectFinishLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *landing, const uint32_t *slot, const uint16_t *shared, uint32_t live, uint32_t hidden, uint16_t *output)
{
	static_assert(RANKS == 4u || RANKS == 16u, "direct finish skl2 supports 4 or 16 ranks");
	return D41Skl2Launch(RANKS == 4u ? D41SKL2_d41_direct_finish_r4 : D41SKL2_d41_direct_finish_r16, grid, block, 0, stream, landing, slot, shared, live, hidden, output);
}

static inline SparkGpuError D41WindowAssembleLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *own_kv, const uint16_t *edges, uint16_t *window_kv, uint32_t own, uint32_t edge, uint32_t first)
{ return D41Skl2Launch(D41SKL2_d41_window_assemble_256, grid, block, 0, stream, own_kv, edges, window_kv, own, edge, first); }

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

static inline SparkGpuError D41TailPackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *streams, const float *pre, uint32_t rank, uint32_t rows, uint32_t source_first, uint32_t own, uint32_t first, uint32_t tail_own, uint32_t capacity, uint32_t width, uint4 *send, uint4 *send_pre)
{ return D41Skl2Launch(D41SKL2_d41_tail_pack_256, grid, block, 0, stream, streams, pre, rank, rows, source_first, own, first, tail_own, capacity, width, send, send_pre); }

static inline SparkGpuError D41TailUnpackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint4 *received, const uint4 *received_pre, uint32_t rank, uint32_t rows, uint32_t source_first, uint32_t own, uint32_t first, uint32_t tail_own, uint32_t capacity, uint32_t width, uint16_t *streams, float *pre)
{ return D41Skl2Launch(D41SKL2_d41_tail_unpack_256, grid, block, 0, stream, received, received_pre, rank, rows, source_first, own, first, tail_own, capacity, width, streams, pre); }

static inline SparkGpuError D41TailRowsPackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint4 *rows_data, uint32_t row_vectors, uint32_t record_vectors, uint32_t rank, uint32_t rows, uint32_t source_first, uint32_t own, uint32_t first, uint32_t tail_own, uint32_t capacity, uint4 *send)
{ return D41Skl2Launch(D41SKL2_d41_tail_rows_pack_256, grid, block, 0, stream, rows_data, row_vectors, record_vectors, rank, rows, source_first, own, first, tail_own, capacity, send); }

static inline SparkGpuError D41TailRowsUnpackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint4 *received, uint32_t row_vectors, uint32_t record_vectors, uint32_t rank, uint32_t rows, uint32_t source_first, uint32_t own, uint32_t first, uint32_t tail_own, uint32_t capacity, uint4 *rows_data)
{ return D41Skl2Launch(D41SKL2_d41_tail_rows_unpack_256, grid, block, 0, stream, received, row_vectors, record_vectors, rank, rows, source_first, own, first, tail_own, capacity, rows_data); }

#define D41K_ENGRAM_DIM 256u
#define D41K_ENGRAM_SCALES (D41K_ENGRAM_DIM / 32u)
#define D41K_ENGRAM_RECORD_BYTES (D41K_ENGRAM_DIM + 16u)

template<uint32_t RANKS>
static inline SparkGpuError D41EngramSlotsLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const int64_t *ids, uint32_t rows,
	uint32_t own, uint32_t columns, uint64_t part, uint64_t table_rows, uint32_t *slot, uint32_t *counts)
{
	static_assert(RANKS == 4u || RANKS == 16u, "engram slots skl2 supports 4 or 16 ranks");
	return D41Skl2Launch(RANKS == 4u ? D41SKL2_d41_engram_slots_r4 : D41SKL2_d41_engram_slots_r16, grid, block, 0, stream,
		ids, rows, own, columns, part, table_rows, slot, counts);
}

static inline SparkGpuError D41EngramPackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const int64_t *ids, const uint32_t *slot,
	const uint8_t *payload, const uint8_t *scale, uint32_t own, uint32_t columns, uint64_t first_row, uint64_t local_rows,
	uint32_t capacity, uint8_t *send)
{
	return D41Skl2Launch(D41SKL2_d41_engram_pack_256, grid, block, 0, stream, ids, slot, payload, scale, own, columns, first_row, local_rows, capacity, send);
}

static inline SparkGpuError D41EngramUnpackLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const int64_t *ids, const uint32_t *slot,
	const uint8_t *received, uint32_t rows, uint32_t first, uint32_t columns, uint64_t part, uint64_t table_rows, uint32_t capacity, uint16_t *embed)
{
	return D41Skl2Launch(D41SKL2_d41_engram_unpack_256, grid, block, 0, stream, ids, slot, received, rows, first, columns, part, table_rows, capacity, embed);
}

static inline SparkGpuError D41FinalizeSliceLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *slice_out, const uint16_t *received, const uint32_t *route_expert, const uint32_t *route_packed_row, uint16_t *output, uint32_t top_k, uint32_t local_experts, uint32_t degree, uint32_t hidden_slice, uint64_t source_stride, uint64_t shared_offset, const uint16_t *shared_partial)
{ return D41Skl2Launch(D41SKL2_d41_finalize_slice_256, grid, block, 0, stream, slice_out, received, route_expert, route_packed_row, output, top_k, local_experts, degree, hidden_slice, source_stride, shared_offset, shared_partial); }

static inline SparkGpuError D41SliceGatherLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *gathered, uint16_t *output, uint32_t rows, uint32_t hidden_slice, uint32_t degree)
{ return D41Skl2Launch(D41SKL2_d41_slice_gather_256, grid, block, 0, stream, gathered, output, rows, hidden_slice, degree); }

static inline SparkGpuError D41ExpertActLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *gate_up, const float *packed_weight, const uint32_t *group_row_offset, uint32_t local_experts, uint8_t *codes, uint16_t *act, uint32_t routes, uint32_t width, float limit)
{ return D41Skl2Launch(D41SKL2_d41_expert_act_256, grid, block, 0, stream, gate_up, packed_weight, group_row_offset, local_experts, codes, act, routes, width, limit); }

static void D41SharedPart(uint32_t part, uint32_t parts, uint32_t granules, uint32_t granule, uint32_t *first, uint32_t *width)
{
	const uint32_t base = granules / parts, extra = granules % parts;
	*first = (part * base + (part < extra ? part : extra)) * granule;
	*width = (base + (part < extra ? 1u : 0u)) * granule;
}

static inline SparkGpuError D41MoeFinalizeF32Launch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *packed, const uint32_t *route_packed_row, const uint32_t *route_expert, const float *shared, float *output, uint32_t top_k, uint32_t local_experts, uint32_t width)
{ return D41Skl2Launch(D41SKL2_d41_moe_finalize_f32_256, grid, block, 0, stream, packed, route_packed_row, route_expert, shared, output, top_k, local_experts, width); }

static inline SparkGpuError D41MoeFinalizeF32ToBf16Launch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *packed, const uint32_t *route_packed_row, const uint32_t *route_expert, const float *shared, uint16_t *output, uint32_t top_k, uint32_t local_experts, uint32_t width)
{ return D41Skl2Launch(D41SKL2_d41_moe_finalize_f32_bf16_256, grid, block, 0, stream, packed, route_packed_row, route_expert, shared, output, top_k, local_experts, width); }

static inline SparkGpuError D41SumSourcesF32Launch(dim3 grid, dim3 block, SparkGpuStream stream, const float *gathered, uint16_t *output, uint64_t elements, uint32_t degree)
{ return D41Skl2Launch(D41SKL2_d41_sum_sources_f32_256, grid, block, 0, stream, gathered, output, elements, degree); }

static inline SparkGpuError D41SumGatheredLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *gathered, uint16_t *output, uint64_t elements, uint32_t degree)
{ return D41Skl2Launch(D41SKL2_d41_sum_gathered_256, grid, block, 0, stream, gathered, output, elements, degree); }

static inline SparkGpuError D41PackIndexQueriesMxfp4Launch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *src, uint32_t *codes, uint8_t *scales, uint64_t octets)
{ return D41Skl2Launch(D41SKL2_d41_pack_index_queries_mxfp4_256, grid, block, 0, stream, src, codes, scales, octets); }

template<class Geometry>
static inline SparkGpuError D41PackIndexKeysMxfp4Launch(dim3 grid, dim3 block, SparkGpuStream stream, LmKvShardView view,
	const uint32_t *sequence_of_row, const uint32_t *length, uint32_t stride, uint32_t *out_codes, uint32_t *out_scales)
{
	return D41Skl2Launch(D41SKL2_d41_pack_index_keys_mxfp4_256, grid, block, 0, stream, view, sequence_of_row, length, stride,
		out_codes, out_scales, (uint32_t)Geometry::kPageSlots, (uint32_t)Geometry::kSlotBytes);
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

static inline SparkGpuError D41WholeBlockMaxLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const float *scores, uint64_t score_stride, const uint32_t *seq, const uint32_t *len, const uint32_t *bound, float *block_scores, uint32_t block_stride, uint32_t grain, uint32_t dense_rows)
{ return D41Skl2Launch(D41SKL2_d41_whole_block_max_256, grid, block, 0, stream, scores, score_stride, seq, len, bound, block_scores, block_stride, grain, dense_rows); }

static inline SparkGpuError D41WholeBlockApplyLaunch(dim3 grid, dim3 block, SparkGpuStream stream, float *scores, uint64_t score_stride, uint32_t grain, const uint8_t *mask, uint32_t block_stride, const uint32_t *seq, const uint32_t *len, const uint32_t *bound, uint32_t dense_rows)
{ return D41Skl2Launch(D41SKL2_d41_whole_block_apply_256, grid, block, 0, stream, scores, score_stride, grain, mask, block_stride, seq, len, bound, dense_rows); }

static inline SparkGpuError D41WholeBlockMaskLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const float *block_scores, const uint32_t *chosen, uint32_t keep, uint8_t *mask, uint32_t block_stride, const uint32_t *seq = 0, const uint32_t *len = 0, const uint32_t *bound = 0, uint32_t grain = 0u, uint32_t dense_rows = 0u)
{ return D41Skl2Launch(D41SKL2_d41_whole_block_mask_256, grid, block, 0, stream, block_scores, chosen, keep, mask, block_stride, seq, len, bound, grain, dense_rows); }

template<class Geometry, bool ROUND_BF16 = false>
static inline SparkGpuError D41IndexScoreDirectLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint16_t *index_query_bf16,
	const uint16_t *head_weight, LmKvShardView index_view, const uint32_t *sequence_of_row, const uint32_t *context_length,
	const uint32_t *row_position, const uint8_t *block_mask, uint32_t block_stride, uint32_t rows, uint32_t score_stride,
	float qk_scale, float *scores, const uint32_t *fork_positions = 0, const uint32_t *slot_offsets = 0, uint32_t dense_limit = 0u)
{
	static_assert(!ROUND_BF16, "index score direct skl2 uses unrounded f32 scoring");
	return D41Skl2Launch(D41SKL2_d41_index_score_direct_256, grid, block, 0, stream, index_query_bf16, head_weight, index_view,
		sequence_of_row, context_length, row_position, block_mask, block_stride, rows, score_stride, qk_scale, scores,
		fork_positions, slot_offsets, dense_limit, (uint32_t)Geometry::kPageSlots, (uint32_t)Geometry::kSlotBytes);
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
			SparkGpuError status = D41Skl2Launch(D41SKL2_d41_topk_exact_chunk_reg_256, dim3(chunks, rows), 256u, 0, stream, scores, topk_values, topk_positions, n, k, chunk, chunks, (uint32_t *)0,
				seq, len, bound, dense_limit, pool_stride);
			if ( status != SPARK_GPU_SUCCESS ) return status;
			return D41Skl2Launch(D41SKL2_d41_topk_exact_gather_reg_256, rows, 256u, 0, stream, scores, topk_values, topk_positions, n, k, chunk, chunks, out_indices,
				seq, len, bound, dense_limit, pool_stride);
		}
	}
	return D41Skl2Launch(D41SKL2_d41_topk_exact_fast_256, rows, 256u, 0, stream, scores, (const float *)0, (const uint32_t *)0, n, k, 0u, 0u, out_indices, seq, len, bound, dense_limit, pool_stride);
}

template<uint32_t EXPERTS, uint32_t TOP_K, uint32_t MAX_ROUTES = 128u>
static __device__ __forceinline__ void D41TopkRoutePackedBody(const float *__restrict__ router_logits, const float *__restrict__ router_bias, float route_scale,
	uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token,
	uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up,
	uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down, float *__restrict__ packed_weight, uint32_t rows,
	const uint32_t *__restrict__ row_valid = 0, uint32_t *__restrict__ active_out = 0)
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
	const uint32_t act0 = threadIdx.x < EXPERTS && s_off[threadIdx.x] != 0u ? 1u : 0u;
	const uint32_t act1 = threadIdx.x + 256u < EXPERTS && s_off[threadIdx.x + 256u] != 0u ? 1u : 0u;
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
	if ( active_out == 0 )
	{
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
	}
	else
	{
		const uint32_t tot_off = scan512(s_off);
		for ( uint32_t i = threadIdx.x; i < EXPERTS; i += 256u ) group_row_offset[i] = s_off[i];
		if ( threadIdx.x == 0u ) group_row_offset[EXPERTS] = tot_off;
		const uint32_t m0 = __ballot_sync(0xffffffffu, act0 != 0u);
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
		if ( act0 != 0u ) active_out[off0] = threadIdx.x;
		__syncthreads();
		const uint32_t m1 = __ballot_sync(0xffffffffu, act1 != 0u);
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
		if ( act1 != 0u ) active_out[off1] = threadIdx.x + 256u;
		if ( threadIdx.x == 0u ) active_out[EXPERTS] = tot0 + tot1;
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
	const uint32_t *__restrict__ row_valid = 0, uint32_t *__restrict__ active_out = 0)
{
	LmDependentWait();
	D41TopkRoutePackedBody<EXPERTS, TOP_K, 128u>(router_logits, router_bias, route_scale, fault, route_expert, route_weight,
		group_row_offset, route_packed_row, route_source_token, tile_m, neuron_tiles_up, tile_prefix_up, neuron_tiles_down,
		tile_prefix_down, packed_weight, rows, row_valid, active_out);
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

template<uint32_t EXPERTS, uint32_t HIDDEN, uint32_t MAX_R, uint32_t EPB = 1u>
static __device__ __forceinline__ void D41RouterGateSplitBlock(const uint16_t *__restrict__ router_w, const uint16_t *__restrict__ x,
	float *__restrict__ router_logits, uint32_t rows, const uint32_t *__restrict__ row_valid)
{
	constexpr uint32_t CHUNKS = HIDDEN / 8u, WARPS = 8u, SPAN = CHUNKS / WARPS;
	static_assert(CHUNKS % WARPS == 0u && SPAN <= 96u && EXPERTS % EPB == 0u, "router hidden splits evenly across eight warps");
	__shared__ float partial[WARPS][MAX_R == 1u ? EPB : MAX_R];
	const uint32_t e0 = blockIdx.x * EPB, warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, first = warp * SPAN;
	const uint4 *w_base = (const uint4 *)router_w, *x_base = (const uint4 *)x;
	if ( lane == 0u )
		#pragma unroll
		for ( uint32_t ep = 0u; ep < EPB; ++ep )
			LmPrefetchL2(w_base + (uint64_t)(e0 + ep) * CHUNKS + first, SPAN * 16u);
	LmDependentWait();
	if constexpr ( MAX_R == 1u )
	{
		const uint4 *xr = x_base + first;
		const uint4 x0 = lane < SPAN ? __ldg(xr + lane) : make_uint4(0u, 0u, 0u, 0u);
		const uint4 x1 = lane + 32u < SPAN ? __ldg(xr + lane + 32u) : make_uint4(0u, 0u, 0u, 0u);
		const uint4 x2 = lane + 64u < SPAN ? __ldg(xr + lane + 64u) : make_uint4(0u, 0u, 0u, 0u);
		#pragma unroll
		for ( uint32_t ep = 0u; ep < EPB; ++ep )
		{
			const uint4 *wr = w_base + (uint64_t)(e0 + ep) * CHUNKS + first;
			float s = 0.0f;
			if ( lane < SPAN ) s += D41Bf16Dot8(__ldcs(wr + lane), x0);
			if ( lane + 32u < SPAN ) s += D41Bf16Dot8(__ldcs(wr + lane + 32u), x1);
			if ( lane + 64u < SPAN ) s += D41Bf16Dot8(__ldcs(wr + lane + 64u), x2);
			#pragma unroll
			for ( uint32_t off = 16u; off > 0u; off >>= 1u )
				s += __shfl_xor_sync(0xffffffffu, s, off);
			if ( lane == 0u ) partial[warp][ep] = s;
		}
		__syncthreads();
		if ( warp == 0u && lane < EPB )
		{
			float total = 0.0f;
			if ( row_valid == 0 || row_valid[0] != 0u )
			{
				#pragma unroll
				for ( uint32_t w = 0u; w < WARPS; ++w ) total += partial[w][lane];
			}
			router_logits[e0 + lane] = total;
		}
		return;
	}
	for ( uint32_t ep = 0u; ep < EPB; ++ep )
	{
		const uint32_t e = e0 + ep;
		const uint4 *wr = w_base + (uint64_t)e * CHUNKS + first;
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
		if ( ep + 1u < EPB ) __syncthreads();
	}
}

template<uint32_t RANKS, uint32_t EXPERTS, uint32_t LOCAL, uint32_t THREADS, uint32_t HELD_PLUS_ONE, uint32_t TOP_K, uint32_t MAX_COPIES, uint32_t HIDDEN, uint32_t MAX_R, uint32_t EPB = 1u>
static __global__ __launch_bounds__(THREADS)
void D41FusedRouterBalanceKernel(const uint16_t *__restrict__ router_w, const uint16_t *__restrict__ x, float *__restrict__ router_logits,
	const float *__restrict__ router_bias, float route_scale, uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert,
	float *__restrict__ route_weight, uint32_t *__restrict__ route_global, uint32_t rows, uint32_t rank, uint32_t copies, uint32_t row_invariant,
	const uint8_t *__restrict__ owner, const uint8_t *__restrict__ backup, const uint8_t *__restrict__ group, uint32_t *__restrict__ group_row_offset,
	uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up,
	uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down, float *__restrict__ packed_weight,
	uint32_t *__restrict__ counter, const uint32_t *__restrict__ row_valid = 0)
{
	D41RouterGateSplitBlock<EXPERTS, HIDDEN, MAX_R, EPB>(router_w, x, router_logits, rows, row_valid);
	if ( LmFusedLastArrival(counter, EXPERTS / EPB) )
		D41TopkBalanceRouteBody<RANKS, EXPERTS, LOCAL, THREADS, HELD_PLUS_ONE, TOP_K, MAX_COPIES, true, MAX_R>(router_logits, router_bias, route_scale,
			fault, route_expert, route_weight, route_global, rows, rank, copies, row_invariant, owner, backup, group, group_row_offset, route_packed_row,
			route_source_token, tile_m, neuron_tiles_up, tile_prefix_up, neuron_tiles_down, tile_prefix_down, packed_weight, row_valid);
	LmDependentRelease();
}

template<uint32_t EXPERTS, uint32_t TOP_K, uint32_t HIDDEN, uint32_t MAX_R, uint32_t EPB = 1u>
static __global__ __launch_bounds__(256)
void D41FusedRouterPackedKernel(const uint16_t *__restrict__ router_w, const uint16_t *__restrict__ x, float *__restrict__ router_logits,
	const float *__restrict__ router_bias, float route_scale, uint32_t *__restrict__ fault, uint32_t *__restrict__ route_expert,
	float *__restrict__ route_weight, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row,
	uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up,
	uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down, float *__restrict__ packed_weight, uint32_t rows,
	uint32_t *__restrict__ counter, const uint32_t *__restrict__ row_valid = 0, uint32_t *__restrict__ active_out = 0)
{
	D41RouterGateSplitBlock<EXPERTS, HIDDEN, MAX_R, EPB>(router_w, x, router_logits, rows, row_valid);
	if ( LmFusedLastArrival(counter, EXPERTS / EPB) )
		D41TopkRoutePackedBody<EXPERTS, TOP_K, MAX_R * TOP_K>(router_logits, router_bias, route_scale, fault, route_expert, route_weight,
			group_row_offset, route_packed_row, route_source_token, tile_m, neuron_tiles_up, tile_prefix_up, neuron_tiles_down,
			tile_prefix_down, packed_weight, rows, row_valid, active_out);
	LmDependentRelease();
}

template<uint32_t EXPERTS>
static inline SparkGpuError D41CompactActiveExpertsLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint32_t *group_row_offset, uint32_t *active_out, uint32_t count = EXPERTS)
{ return D41Skl2Launch(D41SKL2_d41_compact_active_experts_256, grid, block, 0, stream, group_row_offset, active_out, EXPERTS, count); }

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
	const uint32_t *skip_seq;
	const uint32_t *skip_len;
	const uint32_t *skip_bound;
	uint32_t skip_max;
	uint32_t skip_grp;
};

template<uint32_t STAGES, uint32_t TILE_N = 16u>
static __global__ __launch_bounds__(128, 4)
void D41MxGemmBatch16Kernel(const __grid_constant__ D41MxBatchDesc desc, uint32_t rows)
{
	constexpr uint32_t TILE_M = 16u, TILE_K = 128u, BLK = 32u, N_FRAGS = TILE_N / 8u;
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
	if ( desc.skip_max != 0u && desc.skip_seq != 0 && desc.skip_len != 0 && grp == desc.skip_grp )
	{
		const uint32_t row_end = row_base + TILE_M < rows ? row_base + TILE_M : rows;
		uint32_t any_sparse = 0u;
		for ( uint32_t r = row_base; r < row_end; ++r )
		{
			uint32_t b = __ldg(desc.skip_len + __ldg(desc.skip_seq + r));
			if ( desc.skip_bound != 0 )
			{
				const uint32_t rp = __ldg(desc.skip_bound + r);
				b = rp == D41K_NO_ROW ? 0u : (rp + 1u < b ? rp + 1u : b);
			}
			if ( b > desc.skip_max ) { any_sparse = 1u; break; }
		}
		if ( any_sparse == 0u ) { LmDependentRelease(); return; }
	}
	auto swz = [](uint32_t r, uint32_t c) { return r * TILE_K + ((c ^ (r & 7u)) << 4u); };
	const uint32_t ch = threadIdx.x, r_cp = ch >> 3u, c_cp = ch & 7u;
	const uint32_t a_live = row_base + r_cp < rows ? 16u : 0u;
	const uint32_t b_live = r_cp < TILE_N && n_base + r_cp < out_dim ? 16u : 0u;
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
		asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(st + sa_dst), "l"(a_row + (uint64_t)k * TILE_K), "r"(a_live) : "memory");
		if constexpr ( TILE_N == 16u )
			asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n" :: "r"(st + sb_dst), "l"(b_row + (uint64_t)k * TILE_K), "r"(b_live) : "memory");
		else if ( r_cp < TILE_N )
			asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\n" :: "r"(st + sb_dst), "l"(b_row + (uint64_t)k * TILE_K), "r"(b_live) : "memory");
		if ( sc_src != 0 )
			asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(st + A_BYTES + B_BYTES + threadIdx.x * 4u),
				"l"(sc_src + (uint64_t)k * 4u) : "memory");
	};
	float total[N_FRAGS][4];
	#pragma unroll
	for ( uint32_t j = 0u; j < N_FRAGS; ++j ) total[j][0] = total[j][1] = total[j][2] = total[j][3] = 0.0f;
	for ( uint32_t p = 0u; p + 1u < STAGES; ++p )
	{
		if ( p < k_tiles ) issue(p);
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	}
	const uint32_t a_ldm_r = ((lane >> 3u) & 1u) * 8u + (lane & 7u), a_ldm_c = warp_k * 2u + (lane >> 4u);
	const uint32_t b_ldm_r = (TILE_N == 16u ? (lane >> 4u) * 8u : 0u) + (lane & 7u), b_ldm_c = warp_k * 2u + ((lane >> 3u) & 1u);
	const uint32_t a_ldm_off = swz(a_ldm_r, a_ldm_c), b_ldm_off = A_BYTES + swz(b_ldm_r, b_ldm_c);
	const uint32_t sa_idx = group + ((lane & 1u) << 3u), sb0_idx = TILE_M + group, sb1_idx = TILE_M + (TILE_N == 16u ? 8u : 0u) + group;
	for ( uint32_t k = 0u; k < k_tiles; ++k )
	{
		asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2u) : "memory");
		__syncthreads();
		if ( k + STAGES - 1u < k_tiles ) issue(k + STAGES - 1u);
		asm volatile("cp.async.commit_group;\n" ::: "memory");
		const uint32_t st = sbase + (k % STAGES) * STAGE_BYTES;
		const uint32_t *words = (const uint32_t *)(smem + (k % STAGES) * STAGE_BYTES + A_BYTES + B_BYTES);
		const uint32_t scale_a = words[sa_idx], scale_b0 = words[sb0_idx];
		uint32_t a[4], b0[2];
		asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
			: "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(st + a_ldm_off));
		if constexpr ( TILE_N == 16u )
		{
			const uint32_t scale_b1 = words[sb1_idx];
			uint32_t b1[2];
			asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
				: "=r"(b0[0]), "=r"(b0[1]), "=r"(b1[0]), "=r"(b1[1]) : "r"(st + b_ldm_off));
			float blk0[4], blk1[4];
			LmMmaMxf8Block(blk0, a, b0, scale_a, (uint16_t)warp_k, scale_b0, (uint16_t)warp_k);
			LmMmaMxf8Block(blk1, a, b1, scale_a, (uint16_t)warp_k, scale_b1, (uint16_t)warp_k);
			total[0][0] += blk0[0]; total[0][1] += blk0[1]; total[0][2] += blk0[2]; total[0][3] += blk0[3];
			total[1][0] += blk1[0]; total[1][1] += blk1[1]; total[1][2] += blk1[2]; total[1][3] += blk1[3];
		}
		else
		{
			asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
				: "=r"(b0[0]), "=r"(b0[1]) : "r"(st + b_ldm_off));
			float blk0[4];
			LmMmaMxf8Block(blk0, a, b0, scale_a, (uint16_t)warp_k, scale_b0, (uint16_t)warp_k);
			total[0][0] += blk0[0]; total[0][1] += blk0[1]; total[0][2] += blk0[2]; total[0][3] += blk0[3];
		}
	}
	asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	float *ep = (float *)smem;
	if ( warp_k != 0u )
	{
		#pragma unroll
		for ( uint32_t j = 0u; j < N_FRAGS; ++j )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; ++e )
				ep[(((warp_k - 1u) * N_FRAGS + j) * 4u + e) * 32u + lane] = total[j][e];
	}
	__syncthreads();
	if ( warp_k == 0u )
	{
		#pragma unroll
		for ( uint32_t j = 0u; j < N_FRAGS; ++j )
			#pragma unroll
			for ( uint32_t e = 0u; e < 4u; e += 2u )
			{
				const uint32_t r = row_base + group + (e >> 1u) * 8u;
				const uint32_t col = n_base + j * 8u + quad * 2u;
				if ( r < rows && col < out_dim )
				{
					const float v0 = total[j][e] + ep[((0u * N_FRAGS + j) * 4u + e) * 32u + lane]
						+ ep[((1u * N_FRAGS + j) * 4u + e) * 32u + lane] + ep[((2u * N_FRAGS + j) * 4u + e) * 32u + lane];
					const float v1 = total[j][e + 1u] + ep[((0u * N_FRAGS + j) * 4u + e + 1u) * 32u + lane]
						+ ep[((1u * N_FRAGS + j) * 4u + e + 1u) * 32u + lane] + ep[((2u * N_FRAGS + j) * 4u + e + 1u) * 32u + lane];
					*(uint32_t *)(out_ptr + (uint64_t)r * out_stride + out_col + col) = LmPackBf16Pair(v0, v1);
				}
			}
	}
	LmDependentRelease();
}

struct D41SharedPartialsDesc
{
	const uint8_t *w[4];
	const uint8_t *ws[4];
	uint16_t *out[4];
	uint32_t width[4];
};

template<uint32_t MAX_R, uint32_t OUT_DIM, uint32_t MAX_WIDTH>
static inline SparkGpuError D41SharedPartialsFusedLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const D41SharedPartialsDesc desc,
	const uint16_t *received, uint64_t stride, uint32_t rows)
{
	static_assert((MAX_R == 1u || MAX_R == 8u) && MAX_WIDTH <= 640u && OUT_DIM % 16u == 0u, "shared partials skl2 supports r1/r8 w<=640");
	return D41Skl2Launch(MAX_R == 1u ? D41SKL2_d41_shared_partials_r1_256 : D41SKL2_d41_shared_partials_r8_256, grid, block, 0, stream,
		desc, received, stride, rows, OUT_DIM);
}

template<uint32_t MODE_RECV, uint32_t CELLS_PER_BLK, uint32_t WARP_CHUNKS, uint32_t STAGES, uint32_t EXPERTS, uint32_t SLICE, uint32_t IN_DIM, uint32_t OUT_DIM>
static __device__ __forceinline__ void D41CellMxDirectBody(const uint8_t *__restrict__ w_cells, const uint8_t *__restrict__ act_codes,
	const uint8_t *__restrict__ act_scales, const uint16_t *__restrict__ recv_base, uint16_t *__restrict__ output_bf16,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ route_source_token, const uint32_t *__restrict__ active_experts,
	uint32_t routes, uint64_t recv_stride, const uint8_t *__restrict__ backup_cells = 0, uint32_t split_at = 0u)
{
	constexpr uint32_t WARPS_M = CELLS_PER_BLK == 8u ? 2u : 1u, WARPS_N = 4u, THREADS = WARPS_M * WARPS_N * 32u;
	constexpr uint32_t CELLS_PER_WARP = CELLS_PER_BLK / WARPS_N, N_FRAGS = CELLS_PER_WARP * 2u;
	constexpr uint32_t TILE_M = WARPS_M * WARP_CHUNKS * 16u, ROW_STEP = THREADS / 8u, COPIES = TILE_M / ROW_STEP;
	constexpr uint32_t TILE_K = 128u, K_STEPS = 4u, K_ITERS = IN_DIM / TILE_K;
	constexpr uint32_t TOTAL_CELLS = OUT_DIM / 16u, N_BLOCKS = TOTAL_CELLS / CELLS_PER_BLK;
	constexpr uint32_t CELL_VECS = 17u, CELL_BYTES = CELL_VECS * 16u;
	constexpr uint32_t TILE_VECS = CELLS_PER_BLK * CELL_VECS, TILE_BYTES = CELLS_PER_BLK * CELL_BYTES;
	constexpr uint32_t W_VECS = K_STEPS * TILE_VECS, W_BYTES = K_STEPS * TILE_BYTES;
	constexpr uint32_t W_PASSES = (W_VECS + THREADS - 1u) / THREADS;
	constexpr uint32_t A_BYTES = TILE_M * TILE_K, STAGE_BYTES = A_BYTES + W_BYTES;
	constexpr uint32_t S_WORDS = 2u * TILE_M;
	constexpr uint64_t TILE_STRIDE = (uint64_t)TOTAL_CELLS * CELL_BYTES;
	constexpr uint64_t GROUP_BYTES = (uint64_t)(IN_DIM / 32u) * TILE_STRIDE;
	constexpr uint32_t S_BLOCKS = SLICE / SPARK_DSV41_FLASH_FP8_BLOCK;
	static_assert(OUT_DIM % (CELLS_PER_BLK * 16u) == 0u && IN_DIM % 128u == 0u && (MODE_RECV != 1u || SLICE % 64u == 0u) && W_PASSES == 3u && TILE_M % 16u == 0u,
		"direct cell mx geometry");
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
	const uint32_t block_cell0 = n_block * CELLS_PER_BLK, cell_first = block_cell0 + warp_n * CELLS_PER_WARP;
	const uint8_t *group_cells = (backup_cells != 0 && group >= split_at ?
		backup_cells + (uint64_t)(group - split_at) * GROUP_BYTES :
		w_cells + (uint64_t)group * GROUP_BYTES) + (uint64_t)block_cell0 * CELL_BYTES;
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
	const uint32_t warp_w_off = warp_n * CELLS_PER_WARP * CELL_BYTES;
	const uint32_t w_lane_off = warp_w_off + lane * 4u, ws_lane_off = warp_w_off + 256u + g;
	const uint8_t *r0 = MODE_RECV == 1u ? (const uint8_t *)(recv_base + 0ull * recv_stride) : (const uint8_t *)0;
	const uint8_t *r1 = MODE_RECV == 1u ? (const uint8_t *)(recv_base + 1ull * recv_stride) : (const uint8_t *)0;
	const uint8_t *r2 = MODE_RECV == 1u ? (const uint8_t *)(recv_base + 2ull * recv_stride) : (const uint8_t *)0;
	const uint8_t *r3 = MODE_RECV == 1u ? (const uint8_t *)(recv_base + 3ull * recv_stride) : (const uint8_t *)0;
	const uint64_t sc_off_base = (uint64_t)routes * SLICE;
	for ( uint32_t first = start; first < end; first += TILE_M )
	{
		const uint32_t count = end - first < TILE_M ? end - first : TILE_M;
		const uint32_t sc_route = first + threadIdx.x, sc_live = threadIdx.x < count ? 1u : 0u;
		uint32_t a_tok[COPIES];
		if constexpr ( MODE_RECV != 1u )
		{
			#pragma unroll
			for ( uint32_t q = 0u; q < COPIES; ++q )
			{
				const uint32_t lr = row_base + q * ROW_STEP;
				a_tok[q] = lr < count ? (MODE_RECV == 0u ? __ldg(route_source_token + first + lr) : first + lr) : 0u;
			}
		}
		const uint32_t sc_tok = MODE_RECV != 1u && sc_live != 0u ? (MODE_RECV == 0u ? __ldg(route_source_token + sc_route) : sc_route) : 0u;
		const uint8_t *w_base = group_cells;
		float acc[WARP_CHUNKS][N_FRAGS][4];
		#pragma unroll
		for ( uint32_t m = 0u; m < WARP_CHUNKS; ++m )
			#pragma unroll
			for ( uint32_t f = 0u; f < N_FRAGS; ++f ) acc[m][f][0] = acc[m][f][1] = acc[m][f][2] = acc[m][f][3] = 0.0f;
		auto produce = [&](uint32_t k_idx, uint32_t st_off)
		{
			const uint32_t stage_s = sbase + st_off, w_stage = stage_s + A_BYTES;
			const uint32_t elem = k_idx * TILE_K + chunk * 16u;
			if constexpr ( MODE_RECV != 1u )
			{
				#pragma unroll
				for ( uint32_t q = 0u; q < COPIES; ++q )
				{
					const uint32_t lr = row_base + q * ROW_STEP, live = lr < count ? 1u : 0u;
					LmCellMxCopy(stage_s + LmCellMxSwizzle(lr, chunk),
						live != 0u ? act_codes + (uint64_t)a_tok[q] * IN_DIM + elem : act_codes, live << 4u);
				}
			}
			else
			{
				const uint32_t src_r = elem / SLICE, within = elem - src_r * SLICE;
				const uint8_t *rb = src_r == 0u ? r0 : src_r == 1u ? r1 : src_r == 2u ? r2 : r3;
				#pragma unroll
				for ( uint32_t q = 0u; q < COPIES; ++q )
				{
					const uint32_t lr = row_base + q * ROW_STEP, live = lr < count ? 1u : 0u;
					LmCellMxCopy(stage_s + LmCellMxSwizzle(lr, chunk), live != 0u ? rb + (uint64_t)(first + lr) * SLICE + within : rb, live << 4u);
				}
			}
			#pragma unroll
			for ( uint32_t p = 0u; p < W_PASSES; ++p )
				if ( threadIdx.x + p * THREADS < W_VECS )
					LmCellMxCopy256(w_stage + (threadIdx.x + p * THREADS) * 16u, w_base + w_off[p], 16u);
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
					if constexpr ( MODE_RECV != 1u )
						sw = *(const uint32_t *)(act_scales + (uint64_t)sc_tok * (IN_DIM / 32u) + k_step * 4u);
					else
					{
						const uint32_t b0 = k_step * 4u, sr0 = b0 / S_BLOCKS, sb0 = b0 - sr0 * S_BLOCKS;
						const uint32_t b2 = b0 + 2u, sr2 = b2 / S_BLOCKS, sb2 = b2 - sr2 * S_BLOCKS;
						const uint8_t *rb0 = sr0 == 0u ? r0 : sr0 == 1u ? r1 : sr0 == 2u ? r2 : r3;
						const uint8_t *rb2 = sr2 == 0u ? r0 : sr2 == 1u ? r1 : sr2 == 2u ? r2 : r3;
						const uint32_t lo = *(const uint16_t *)(rb0 + sc_off_base + (uint64_t)sc_route * S_BLOCKS + sb0);
						const uint32_t hi = *(const uint16_t *)(rb2 + sc_off_base + (uint64_t)sc_route * S_BLOCKS + sb2);
						sw = lo | (hi << 16u);
					}
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
					uint32_t b[N_FRAGS][2], scale_b[N_FRAGS], a[2][4];
					#pragma unroll
					for ( uint32_t f = 0u; f < N_FRAGS; ++f )
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
						for ( uint32_t f = 0u; f < N_FRAGS; ++f )
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
					for ( uint32_t f = 0u; f < N_FRAGS; ++f )
						*(uint32_t *)(row_out + (cell_first + (f >> 1u)) * 16u + (f & 1u) * 8u + c * 2u) =
							LmPackBf16Pair(acc[m][f][h * 2u], acc[m][f][h * 2u + 1u]);
				}
		}
		if ( first + TILE_M < end ) __syncthreads();
	}
}

template<uint32_t WARP_CHUNKS, uint32_t STAGES, uint32_t EXPERTS, uint32_t SLICE, uint32_t INTER, uint32_t OUT_DIM>
static __global__ __launch_bounds__(256, 2)
void D41CellMxW2DirectKernel(const uint8_t *__restrict__ w2_cells, const uint16_t *__restrict__ recv_base,
	uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ active_experts,
	uint32_t routes, uint64_t recv_stride)
{
	D41CellMxDirectBody<1u, 8u, WARP_CHUNKS, STAGES, EXPERTS, SLICE, INTER, OUT_DIM>(w2_cells, (const uint8_t *)0, (const uint8_t *)0,
		recv_base, output_bf16, group_row_offset, (const uint32_t *)0, active_experts, routes, recv_stride);
}

template<uint32_t WARP_CHUNKS, uint32_t STAGES, uint32_t EXPERTS, uint32_t IN_DIM, uint32_t OUT_DIM>
static __global__ __launch_bounds__(256, 2)
void D41CellMxW13DirectKernel(const uint8_t *__restrict__ w13_cells, const uint8_t *__restrict__ act_codes,
	const uint8_t *__restrict__ act_scales, uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ group_row_offset,
	const uint32_t *__restrict__ route_source_token, const uint32_t *__restrict__ active_experts, uint32_t routes)
{
	D41CellMxDirectBody<0u, 8u, WARP_CHUNKS, STAGES, EXPERTS, 128u, IN_DIM, OUT_DIM>(w13_cells, act_codes, act_scales,
		(const uint16_t *)0, output_bf16, group_row_offset, route_source_token, active_experts, routes, 0ull);
}

template<uint32_t WARP_CHUNKS, uint32_t STAGES, uint32_t EXPERTS, uint32_t IN_DIM, uint32_t OUT_DIM>
static __global__ __launch_bounds__(256, 2)
void D41CellMxSplitW13DirectKernel(const uint8_t *__restrict__ w13_cells, const uint8_t *__restrict__ backup_cells, uint32_t split_at,
	const uint8_t *__restrict__ act_codes, const uint8_t *__restrict__ act_scales, uint16_t *__restrict__ output_bf16,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ route_source_token, const uint32_t *__restrict__ active_experts,
	uint32_t routes)
{
	D41CellMxDirectBody<0u, 8u, WARP_CHUNKS, STAGES, EXPERTS, 128u, IN_DIM, OUT_DIM>(w13_cells, act_codes, act_scales,
		(const uint16_t *)0, output_bf16, group_row_offset, route_source_token, active_experts, routes, 0ull, backup_cells, split_at);
}

template<uint32_t WARP_CHUNKS, uint32_t STAGES, uint32_t EXPERTS, uint32_t IN_DIM, uint32_t OUT_DIM>
static __global__ __launch_bounds__(256, 2)
void D41CellMxSplitW2DirectKernel(const uint8_t *__restrict__ w2_cells, const uint8_t *__restrict__ backup_cells, uint32_t split_at,
	const uint8_t *__restrict__ act_codes, const uint8_t *__restrict__ act_scales, uint16_t *__restrict__ output_bf16,
	const uint32_t *__restrict__ group_row_offset, const uint32_t *__restrict__ active_experts, uint32_t routes)
{
	D41CellMxDirectBody<2u, 8u, WARP_CHUNKS, STAGES, EXPERTS, 128u, IN_DIM, OUT_DIM>(w2_cells, act_codes, act_scales,
		(const uint16_t *)0, output_bf16, group_row_offset, (const uint32_t *)0, active_experts, routes, 0ull, backup_cells, split_at);
}

template<uint32_t EXPERTS, uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t MODE_RECV, uint32_t SLICE, uint32_t CELLS_PER_BLK = 4u, uint32_t STAGES = 3u>
static inline SparkGpuError D41CellFp4GemvLaunch(dim3 grid, dim3 block, SparkGpuStream stream, const uint8_t *cells, const uint16_t *act_bf16,
	const uint16_t *recv_base, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token,
	const uint32_t *active_experts, uint32_t routes, uint64_t recv_stride)
{
	static_assert((CELLS_PER_BLK == 4u || CELLS_PER_BLK == 8u) && (MODE_RECV == 0u || MODE_RECV == 1u) && STAGES >= 2u, "cell fp4 gemv skl2 geometry");
	const uint32_t id = CELLS_PER_BLK == 4u
		? (MODE_RECV == 0u ? (uint32_t)D41SKL2_d41_cell_fp4_gemv_c4_m0 : (uint32_t)D41SKL2_d41_cell_fp4_gemv_c4_m1)
		: (MODE_RECV == 0u ? (uint32_t)D41SKL2_d41_cell_fp4_gemv_c8_m0 : (uint32_t)D41SKL2_d41_cell_fp4_gemv_c8_m1);
	return D41Skl2Launch(id, grid, block, 0, stream, cells, act_bf16, recv_base, output_bf16, group_row_offset, route_source_token,
		active_experts, routes, recv_stride, EXPERTS, IN_DIM, OUT_DIM, SLICE);
}


