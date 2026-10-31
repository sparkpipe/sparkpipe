#pragma once

#include "runtime/launch.h"
#include "inference/kernels/attn_prefill.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "inference/kernels/mma.cuh"
#include <math.h>
#include <stdint.h>

#define LM_FP8_ATTENTION_WARPS 8u
#define LM_FP8_ATTENTION_THREADS (LM_FP8_ATTENTION_WARPS * 32u)
#define LM_FP8_ATTENTION_QUERIES (LM_FP8_ATTENTION_WARPS * 16u)
#define LM_FP8_ATTENTION_TILE 64u
#define LM_FP8_ATTENTION_KEY_ROW 256u
#define LM_FP8_ATTENTION_MAX 448.0f
#define LM_FP8_ATTENTION_AMAX_THREADS 256u
#define LM_FP8_ATTENTION_AMAX_KEYS 512u

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static __host__ __device__ constexpr bool LmFp8AttentionFits(void)
{
	return(HEAD_DIM % 32u == 0u && HEAD_DIM <= LM_FP8_ATTENTION_KEY_ROW && VALUE_DIM % 16u == 0u && VALUE_DIM <= 128u);
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static __host__ __device__ constexpr uint32_t LmFp8AttentionSharedBytes(void)
{
	return(2u * LM_FP8_ATTENTION_TILE * LM_FP8_ATTENTION_KEY_ROW + 2u * VALUE_DIM * LM_FP8_ATTENTION_TILE + 2u * LM_FP8_ATTENTION_TILE * 4u + 16u);
}

static __host__ __device__ __forceinline__ uint64_t LmFp8AttentionTiles(uint64_t keys)
{
	return((keys + LM_FP8_ATTENTION_TILE - 1u) / LM_FP8_ATTENTION_TILE);
}

static __host__ __device__ __forceinline__ uint32_t LmFp8AttentionValueKey(uint32_t logical)
{
	const uint32_t base = logical & ~31u, within = logical & 31u, half = within / 16u, word = within % 16u, quad = word / 4u, byte = word % 4u;
	return(base + half * 16u + 2u * quad + (byte % 2u) + 8u * (byte / 2u));
}

static __device__ __forceinline__ float LmFp8AttentionScale(float amax)
{
	return(amax > 0.0f ? amax / LM_FP8_ATTENTION_MAX : 1.0f);
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(LM_FP8_ATTENTION_AMAX_THREADS, 1)
void LmFp8AttentionValueAmaxKernel(const uint16_t *__restrict__ slots, uint64_t slot_pitch, uint32_t heads, uint32_t keys, uint32_t *__restrict__ value_amax_bits)
{
	const uint32_t first = blockIdx.x * LM_FP8_ATTENTION_AMAX_KEYS, last = first + LM_FP8_ATTENTION_AMAX_KEYS < keys ? first + LM_FP8_ATTENTION_AMAX_KEYS : keys;
	for (uint32_t column = threadIdx.x; column < heads * VALUE_DIM; column += blockDim.x)
	{
		float amax = 0.0f;
		for (uint32_t key = first; key < last; ++key)
			amax = fmaxf(amax,fabsf(LmBf16ToFloat(slots[(uint64_t)key * slot_pitch + (uint64_t)heads * HEAD_DIM + column])));
		atomicMax(&value_amax_bits[column],__float_as_uint(amax));
	}
}

template<uint32_t HEAD_DIM>
__global__ __launch_bounds__(32, 1)
void LmFp8AttentionKeyKernel(const uint16_t *__restrict__ slots, uint64_t slot_pitch, uint32_t heads, uint32_t keys, uint32_t key_capacity,
	uint8_t *__restrict__ keys_e4m3, float *__restrict__ key_scale)
{
	const uint32_t key = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
	const uint16_t *row = slots + (uint64_t)key * slot_pitch + (uint64_t)head * HEAD_DIM;
	uint8_t *out = keys_e4m3 + ((uint64_t)head * key_capacity + key) * HEAD_DIM;
	float amax = 0.0f, scale;
	if ( key >= keys )
		return;
	for (uint32_t d = lane; d < HEAD_DIM; d += 32u)
		amax = fmaxf(amax,fabsf(LmBf16ToFloat(row[d])));
	for (uint32_t offset = 16u; offset != 0u; offset >>= 1u)
		amax = fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,offset));
	scale = LmFp8AttentionScale(amax);
	for (uint32_t d = lane; d < HEAD_DIM; d += 32u)
		out[d] = LmFloatToE4m3(LmBf16ToFloat(row[d]) / scale);
	if ( lane == 0u )
		key_scale[(uint64_t)head * key_capacity + key] = scale;
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(LM_FP8_ATTENTION_THREADS, 1)
void LmFp8AttentionValueKernel(const uint16_t *__restrict__ slots, uint64_t slot_pitch, uint32_t heads, uint32_t keys, const uint32_t *__restrict__ value_amax_bits,
	uint8_t *__restrict__ values_e4m3)
{
	__shared__ float tile_values[LM_FP8_ATTENTION_TILE][VALUE_DIM + 1u];
	const uint32_t tile = blockIdx.x, head = blockIdx.y, tiles = (uint32_t)LmFp8AttentionTiles(keys);
	uint8_t *out = values_e4m3 + ((uint64_t)head * tiles + tile) * VALUE_DIM * LM_FP8_ATTENTION_TILE;
	for (uint32_t index = threadIdx.x; index < LM_FP8_ATTENTION_TILE * VALUE_DIM; index += blockDim.x)
	{
		const uint32_t key = tile * LM_FP8_ATTENTION_TILE + index / VALUE_DIM, dim = index % VALUE_DIM;
		tile_values[index / VALUE_DIM][dim] = key < keys ?
			LmBf16ToFloat(slots[(uint64_t)key * slot_pitch + (uint64_t)heads * HEAD_DIM + (uint64_t)head * VALUE_DIM + dim]) : 0.0f;
	}
	__syncthreads();
	for (uint32_t index = threadIdx.x; index < LM_FP8_ATTENTION_TILE * VALUE_DIM; index += blockDim.x)
	{
		const uint32_t dim = index / LM_FP8_ATTENTION_TILE, logical = index % LM_FP8_ATTENTION_TILE, chunk = logical / 16u;
		const uint32_t place = dim * LM_FP8_ATTENTION_TILE + ((chunk ^ ((dim >> 1u) & 3u)) * 16u) + logical % 16u;
		out[place] = LmFloatToE4m3(tile_values[LmFp8AttentionValueKey(logical)][dim] / LmFp8AttentionScale(__uint_as_float(value_amax_bits[head * VALUE_DIM + dim])));
	}
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(LM_FP8_ATTENTION_THREADS, 1)
void LmFp8AttentionPartialKernel(
	const uint16_t *__restrict__ query_bf16,
	uint32_t heads,
	const uint32_t *__restrict__ row_position,
	uint32_t rows,
	const uint8_t *__restrict__ keys_e4m3,
	const float *__restrict__ key_scale,
	const uint8_t *__restrict__ values_e4m3,
	const uint32_t *__restrict__ value_amax_bits,
	uint32_t key_first,
	uint32_t key_count,
	uint32_t key_capacity,
	float qk_scale,
	float *__restrict__ send)
{
	constexpr uint32_t TILE = LM_FP8_ATTENTION_TILE, KEY_ROW = LM_FP8_ATTENTION_KEY_ROW, K_STEPS = HEAD_DIM / 32u, KEY_CHUNKS = HEAD_DIM / 16u;
	constexpr uint32_t KEY_FRAGS = TILE / 8u, P_STEPS = TILE / 32u, VALUE_FRAGS = VALUE_DIM / 8u, VALUE_TILE_BYTES = VALUE_DIM * TILE;
	static_assert(LmFp8AttentionFits<HEAD_DIM,VALUE_DIM>(), "a warp holds its 16 queries, a score tile and the value accumulators in registers");
	static_assert(LmFp8AttentionSharedBytes<HEAD_DIM,VALUE_DIM>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the key and value tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t lm_fp8_attention_shared[];
	uint8_t *key_tiles = lm_fp8_attention_shared;
	uint8_t *value_tiles = key_tiles + 2u * TILE * KEY_ROW;
	float *key_scales = (float *)(value_tiles + 2u * VALUE_TILE_BYTES);
	uint32_t *reach = (uint32_t *)(key_scales + 2u * TILE);
	const uint32_t warp = threadIdx.x / 32u, lane = threadIdx.x % 32u, group = lane / 4u, quad = lane % 4u, head = blockIdx.y;
	const uint32_t first_row = blockIdx.x * LM_FP8_ATTENTION_QUERIES + warp * 16u, tiles_all = (uint32_t)LmFp8AttentionTiles(key_count);
	const uint32_t width = LmGqaShardWidth(heads,VALUE_DIM,1u), record = LmGqaShardRecordFloats(heads,VALUE_DIM,1u);
	const float scale_log2 = qk_scale * 1.4426950408889634f;
	uint32_t query[K_STEPS][4], row[2], visible_end[2], tile_end;
	float query_scale[2], running_max[2] = {-INFINITY,-INFINITY}, running_sum[2] = {0.0f,0.0f}, output[VALUE_FRAGS][4];
	bool valid[2];

	for (uint32_t side = 0u; side < 2u; ++side)
	{
		row[side] = first_row + group + 8u * side;
		valid[side] = row[side] < rows;
		visible_end[side] = 0u;
		if ( valid[side] && row_position[row[side]] >= key_first )
			visible_end[side] = row_position[row[side]] - key_first + 1u < key_count ? row_position[row[side]] - key_first + 1u : key_count;
	}
	if ( threadIdx.x == 0u )
		reach[0] = 0u;
	__syncthreads();
	if ( (visible_end[0] | visible_end[1]) != 0u )
		atomicMax(&reach[0],visible_end[0] > visible_end[1] ? visible_end[0] : visible_end[1]);
	{
		float amax[2] = {0.0f,0.0f};
		#pragma unroll
		for (uint32_t step = 0u; step < K_STEPS; ++step)
			#pragma unroll
			for (uint32_t index = 0u; index < 4u; ++index)
				#pragma unroll
				for (uint32_t byte = 0u; byte < 4u; ++byte)
					if ( valid[index % 2u] )
						amax[index % 2u] = fmaxf(amax[index % 2u],fabsf(LmBf16ToFloat(query_bf16[((uint64_t)row[index % 2u] * heads + head) * HEAD_DIM +
							step * 32u + 4u * quad + (index / 2u) * 16u + byte])));
		for (uint32_t side = 0u; side < 2u; ++side)
		{
			amax[side] = fmaxf(amax[side],__shfl_xor_sync(0xffffffffu,amax[side],1u));
			amax[side] = fmaxf(amax[side],__shfl_xor_sync(0xffffffffu,amax[side],2u));
			query_scale[side] = LmFp8AttentionScale(amax[side]);
		}
		#pragma unroll
		for (uint32_t step = 0u; step < K_STEPS; ++step)
			#pragma unroll
			for (uint32_t index = 0u; index < 4u; ++index)
			{
				const uint32_t side = index % 2u;
				const uint16_t *source = query_bf16 + ((uint64_t)row[side] * heads + head) * HEAD_DIM + step * 32u + 4u * quad + (index / 2u) * 16u;
				query[step][index] = valid[side] ?
					(uint32_t)LmFloatPairToE4m3(LmBf16ToFloat(source[0]) / query_scale[side],LmBf16ToFloat(source[1]) / query_scale[side]) |
					((uint32_t)LmFloatPairToE4m3(LmBf16ToFloat(source[2]) / query_scale[side],LmBf16ToFloat(source[3]) / query_scale[side]) << 16u) : 0u;
			}
	}
	#pragma unroll
	for (uint32_t frag = 0u; frag < VALUE_FRAGS; ++frag)
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;
	__syncthreads();
	tile_end = (uint32_t)LmFp8AttentionTiles(reach[0]);

	auto load = [&](uint32_t tile, uint32_t buffer)
	{
		const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(key_tiles + buffer * TILE * KEY_ROW);
		const uint32_t value_base = (uint32_t)__cvta_generic_to_shared(value_tiles + buffer * VALUE_TILE_BYTES);
		const uint8_t *value_source = values_e4m3 + ((uint64_t)head * tiles_all + tile) * VALUE_TILE_BYTES;
		for (uint32_t item = threadIdx.x; item < TILE * KEY_CHUNKS; item += LM_FP8_ATTENTION_THREADS)
		{
			const uint32_t key = item / KEY_CHUNKS, chunk = item % KEY_CHUNKS, local = tile * TILE + key;
			LmPrefillCopy(key_base + key * KEY_ROW + LmPrefillSwizzle(key,chunk) * 16u,
				local < key_count ? (const void *)(keys_e4m3 + ((uint64_t)head * key_capacity + local) * HEAD_DIM + chunk * 16u) : (const void *)keys_e4m3,
				local < key_count ? 16u : 0u);
		}
		for (uint32_t item = threadIdx.x; item < VALUE_TILE_BYTES / 16u; item += LM_FP8_ATTENTION_THREADS)
			LmPrefillCopy(value_base + item * 16u,value_source + item * 16u,16u);
		if ( threadIdx.x < TILE )
			key_scales[buffer * TILE + threadIdx.x] = tile * TILE + threadIdx.x < key_count ? key_scale[(uint64_t)head * key_capacity + tile * TILE + threadIdx.x] : 0.0f;
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};

	if ( tile_end != 0u )
		load(0u,0u);
	for (uint32_t tile = 0u; tile < tile_end; ++tile)
	{
		const uint32_t buffer = tile & 1u;
		const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(key_tiles + buffer * TILE * KEY_ROW);
		const uint32_t value_base = (uint32_t)__cvta_generic_to_shared(value_tiles + buffer * VALUE_TILE_BYTES);
		float score[KEY_FRAGS][4], peak[2], rescale[2], shift[2];
		uint16_t weight[KEY_FRAGS][2];
		if ( tile + 1u < tile_end )
		{
			load(tile + 1u,buffer ^ 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		#pragma unroll
		for (uint32_t frag = 0u; frag < KEY_FRAGS; ++frag)
			score[frag][0] = score[frag][1] = score[frag][2] = score[frag][3] = 0.0f;
		#pragma unroll
		for (uint32_t step = 0u; step < K_STEPS; ++step)
			#pragma unroll
			for (uint32_t frag = 0u; frag < KEY_FRAGS; frag += 2u)
			{
				const uint32_t matrix = lane / 8u, key = (frag + matrix / 2u) * 8u + lane % 8u, chunk = 2u * step + (matrix % 2u);
				uint32_t b[4];
				LmPrefillLoadX4(key_base + key * KEY_ROW + LmPrefillSwizzle(key,chunk) * 16u,b);
				LmMmaE4m3(score[frag],query[step],b);
				LmMmaE4m3(score[frag + 1u],query[step],b + 2);
			}
		peak[0] = peak[1] = -INFINITY;
		#pragma unroll
		for (uint32_t frag = 0u; frag < KEY_FRAGS; ++frag)
			#pragma unroll
			for (uint32_t entry = 0u; entry < 4u; ++entry)
			{
				const uint32_t key = frag * 8u + 2u * quad + (entry & 1u), side = entry / 2u;
				const float value = score[frag][entry] * query_scale[side] * key_scales[buffer * TILE + key];
				score[frag][entry] = tile * TILE + key < visible_end[side] ? value : -INFINITY;
				peak[side] = fmaxf(peak[side],score[frag][entry]);
			}
		for (uint32_t side = 0u; side < 2u; ++side)
		{
			float next;
			peak[side] = fmaxf(peak[side],__shfl_xor_sync(0xffffffffu,peak[side],1u));
			peak[side] = fmaxf(peak[side],__shfl_xor_sync(0xffffffffu,peak[side],2u));
			next = fmaxf(running_max[side],peak[side]);
			rescale[side] = running_max[side] == -INFINITY ? 0.0f : SparkMathExp2Fast((running_max[side] - next) * scale_log2);
			running_max[side] = next;
			shift[side] = next == -INFINITY ? 0.0f : next * scale_log2;
			running_sum[side] *= rescale[side];
		}
		#pragma unroll
		for (uint32_t frag = 0u; frag < VALUE_FRAGS; ++frag)
		{
			output[frag][0] *= rescale[0];
			output[frag][1] *= rescale[0];
			output[frag][2] *= rescale[1];
			output[frag][3] *= rescale[1];
		}
		#pragma unroll
		for (uint32_t frag = 0u; frag < KEY_FRAGS; ++frag)
			#pragma unroll
			for (uint32_t side = 0u; side < 2u; ++side)
			{
				float2 rounded;
				weight[frag][side] = LmFloatPairToE4m3(SparkMathExp2Fast(fmaf(score[frag][2u * side],scale_log2,-shift[side])) * LM_FP8_ATTENTION_MAX,
					SparkMathExp2Fast(fmaf(score[frag][2u * side + 1u],scale_log2,-shift[side])) * LM_FP8_ATTENTION_MAX);
				rounded = LmE4m3PairToFloat2Pure(weight[frag][side]);
				running_sum[side] += rounded.x + rounded.y;
			}
		#pragma unroll
		for (uint32_t step = 0u; step < P_STEPS; ++step)
		{
			uint32_t a[4];
			#pragma unroll
			for (uint32_t index = 0u; index < 4u; ++index)
			{
				const uint32_t frag = 4u * step + (index / 2u) * 2u, side = index % 2u;
				a[index] = (uint32_t)weight[frag][side] | ((uint32_t)weight[frag + 1u][side] << 16u);
			}
			#pragma unroll
			for (uint32_t frag = 0u; frag < VALUE_FRAGS; frag += 2u)
			{
				const uint32_t matrix = lane / 8u, dim = (frag + matrix / 2u) * 8u + lane % 8u, chunk = 2u * step + (matrix % 2u);
				uint32_t b[4];
				LmPrefillLoadX4(value_base + dim * TILE + ((chunk ^ ((dim >> 1u) & 3u)) * 16u),b);
				LmMmaE4m3(output[frag],a,b);
				LmMmaE4m3(output[frag + 1u],a,b + 2);
			}
		}
		__syncthreads();
	}
	for (uint32_t side = 0u; side < 2u; ++side)
	{
		running_sum[side] += __shfl_xor_sync(0xffffffffu,running_sum[side],1u);
		running_sum[side] += __shfl_xor_sync(0xffffffffu,running_sum[side],2u);
	}
	for (uint32_t side = 0u; side < 2u; ++side)
	{
		float *base = send + (uint64_t)row[side] * record;
		if ( !valid[side] )
			continue;
		if ( quad == 0u )
		{
			base[width + 2u * head] = running_max[side] == -INFINITY ? -INFINITY : running_max[side] * qk_scale;
			base[width + 2u * head + 1u] = running_sum[side] / LM_FP8_ATTENTION_MAX;
		}
		#pragma unroll
		for (uint32_t frag = 0u; frag < VALUE_FRAGS; ++frag)
			#pragma unroll
			for (uint32_t entry = 0u; entry < 2u; ++entry)
			{
				const uint32_t dim = frag * 8u + 2u * quad + entry;
				base[head * VALUE_DIM + dim] = output[frag][2u * side + entry] * LmFp8AttentionScale(__uint_as_float(value_amax_bits[head * VALUE_DIM + dim])) /
					LM_FP8_ATTENTION_MAX;
			}
	}
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static inline cudaError_t LmFp8AttentionQuantizeLaunch(
	const uint16_t *slots,
	uint64_t slot_pitch,
	uint32_t heads,
	uint32_t keys,
	uint32_t key_capacity,
	uint8_t *keys_e4m3,
	float *key_scale,
	uint8_t *values_e4m3,
	uint32_t *value_amax_bits,
	cudaStream_t stream)
{
	cudaError_t error;
	if ( slots == 0 || keys_e4m3 == 0 || key_scale == 0 || values_e4m3 == 0 || value_amax_bits == 0 || heads == 0u || keys == 0u || keys > key_capacity ||
		slot_pitch < (uint64_t)heads * (HEAD_DIM + VALUE_DIM) )
		return(cudaErrorInvalidValue);
	error = cudaMemsetAsync(value_amax_bits,0,(size_t)heads * VALUE_DIM * sizeof(uint32_t),stream);
	if ( error != cudaSuccess )
		return(error);
	LmFp8AttentionValueAmaxKernel<HEAD_DIM,VALUE_DIM><<<(keys + LM_FP8_ATTENTION_AMAX_KEYS - 1u) / LM_FP8_ATTENTION_AMAX_KEYS,LM_FP8_ATTENTION_AMAX_THREADS,0,stream>>>(
		slots,slot_pitch,heads,keys,value_amax_bits);
	LmFp8AttentionKeyKernel<HEAD_DIM><<<dim3(keys,heads),32u,0,stream>>>(slots,slot_pitch,heads,keys,key_capacity,keys_e4m3,key_scale);
	LmFp8AttentionValueKernel<HEAD_DIM,VALUE_DIM><<<dim3((uint32_t)LmFp8AttentionTiles(keys),heads),LM_FP8_ATTENTION_THREADS,0,stream>>>(
		slots,slot_pitch,heads,keys,value_amax_bits,values_e4m3);
	return(cudaPeekAtLastError());
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static inline cudaError_t LmFp8AttentionPartialLaunch(
	const uint16_t *query_bf16,
	uint32_t heads,
	const uint32_t *row_position,
	uint32_t rows,
	const uint8_t *keys_e4m3,
	const float *key_scale,
	const uint8_t *values_e4m3,
	const uint32_t *value_amax_bits,
	uint32_t key_first,
	uint32_t key_count,
	uint32_t key_capacity,
	float qk_scale,
	float *send,
	uint64_t send_stride,
	cudaStream_t stream)
{
	constexpr uint32_t shared = LmFp8AttentionSharedBytes<HEAD_DIM,VALUE_DIM>();
	static bool granted = false;
	if ( rows == 0u || heads == 0u || query_bf16 == 0 || row_position == 0 || keys_e4m3 == 0 || key_scale == 0 || values_e4m3 == 0 || value_amax_bits == 0 ||
		send == 0 || key_count == 0u || key_count > key_capacity || send_stride < (uint64_t)rows * LmGqaShardRecordFloats(heads,VALUE_DIM,1u) )
		return(cudaErrorInvalidValue);
	if ( !granted )
	{
		const cudaError_t error = cudaFuncSetAttribute((const void *)LmFp8AttentionPartialKernel<HEAD_DIM,VALUE_DIM>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shared);
		if ( error != cudaSuccess )
			return(error);
		granted = true;
	}
	LmFp8AttentionPartialKernel<HEAD_DIM,VALUE_DIM><<<dim3((rows + LM_FP8_ATTENTION_QUERIES - 1u) / LM_FP8_ATTENTION_QUERIES,heads),LM_FP8_ATTENTION_THREADS,shared,stream>>>(
		query_bf16,heads,row_position,rows,keys_e4m3,key_scale,values_e4m3,value_amax_bits,key_first,key_count,key_capacity,qk_scale,send);
	return(cudaPeekAtLastError());
}
