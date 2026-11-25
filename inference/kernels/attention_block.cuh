#pragma once

#include "runtime/launch.h"
#include "inference/kernels/attn_prefill.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "inference/kernels/mma.cuh"
#include <math.h>
#include <stdint.h>
#include "sparkpipe/spark_gpu.h"

#define LM_ATTENTION_BLOCK_WARPS 8u
#define LM_ATTENTION_BLOCK_THREADS (LM_ATTENTION_BLOCK_WARPS * 32u)
#define LM_ATTENTION_BLOCK_QUERIES (LM_ATTENTION_BLOCK_WARPS * 16u)
#define LM_ATTENTION_BLOCK_TILE 64u

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static __host__ __device__ constexpr bool LmAttentionBlockFits(void)
{
	return(HEAD_DIM % 16u == 0u && VALUE_DIM % 16u == 0u && HEAD_DIM <= 192u && VALUE_DIM <= 128u);
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static __host__ __device__ constexpr uint32_t LmAttentionBlockSharedBytes(void)
{
	return(2u * LM_ATTENTION_BLOCK_TILE * (HEAD_DIM + VALUE_DIM) * 2u);
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(LM_ATTENTION_BLOCK_THREADS, 1)
void LmAttentionBlockPartialKernel(
	const uint16_t *__restrict__ query_bf16,
	uint32_t heads,
	const uint32_t *__restrict__ row_position,
	uint32_t rows,
	const uint16_t *__restrict__ slots,
	uint64_t slot_pitch,
	uint32_t key_first,
	uint32_t key_count,
	float qk_scale,
	float *__restrict__ send)
{
	constexpr uint32_t TILE = LM_ATTENTION_BLOCK_TILE, KEY_BYTES = HEAD_DIM * 2u, VALUE_BYTES = VALUE_DIM * 2u;
	constexpr uint32_t KEY_CHUNKS = HEAD_DIM / 8u, VALUE_CHUNKS = VALUE_DIM / 8u, K_STEPS = HEAD_DIM / 16u, KEY_FRAGS = TILE / 8u, P_STEPS = TILE / 16u, VALUE_FRAGS = VALUE_DIM / 8u;
	constexpr uint32_t KEYS_PER_WARP = TILE / LM_ATTENTION_BLOCK_WARPS;
	static_assert(LmAttentionBlockFits<HEAD_DIM,VALUE_DIM>(), "a warp holds its 16 queries, a score tile and the value accumulators in registers");
	static_assert(LmAttentionBlockSharedBytes<HEAD_DIM,VALUE_DIM>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the key and value tiles exceed the shared memory an SM has");
	static_assert(TILE % LM_ATTENTION_BLOCK_WARPS == 0u, "each warp loads whole keys of a tile");
	extern __shared__ __align__(128) uint8_t lm_attention_block_shared[];
	__shared__ uint32_t reach;
	LmDependentWait();
	uint8_t *key_tiles = lm_attention_block_shared, *value_tiles = lm_attention_block_shared + 2u * TILE * KEY_BYTES;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, matrix = lane >> 3u, within = lane & 7u, head = blockIdx.y;
	const uint32_t first_row = blockIdx.x * LM_ATTENTION_BLOCK_QUERIES + warp * 16u;
	const uint32_t width = LmGqaShardWidth(heads,VALUE_DIM,1u), record = LmGqaShardRecordFloats(heads,VALUE_DIM,1u);
	const float scale_log2 = qk_scale * 1.4426950408889634f;
	uint32_t query[K_STEPS][4], row[2], visible_end[2], tiles;
	float running_max[2] = {-INFINITY,-INFINITY}, running_sum[2] = {0.0f,0.0f}, output[VALUE_FRAGS][4];
	bool valid[2];

	#pragma unroll
	for (uint32_t side = 0u; side < 2u; ++side)
	{
		row[side] = first_row + (lane >> 2u) + 8u * side;
		valid[side] = row[side] < rows;
		visible_end[side] = 0u;
		if ( valid[side] )
		{
			const uint32_t pos = __ldg(row_position + row[side]);
			if ( pos >= key_first )
				visible_end[side] = pos - key_first + 1u < key_count ? pos - key_first + 1u : key_count;
		}
	}
	if ( threadIdx.x == 0u )
		reach = 0u;
	__syncthreads();
	{
		const uint32_t warp_reach = __reduce_max_sync(0xffffffffu,visible_end[0] > visible_end[1] ? visible_end[0] : visible_end[1]);
		if ( lane == 0u && warp_reach != 0u )
			atomicMax(&reach,warp_reach);
	}
	#pragma unroll
	for (uint32_t step = 0u; step < K_STEPS; ++step)
		#pragma unroll
		for (uint32_t index = 0u; index < 4u; ++index)
		{
			const uint32_t side = index & 1u;
			query[step][index] = valid[side] ? __ldg((const uint32_t *)(query_bf16 + ((uint64_t)row[side] * heads + head) * HEAD_DIM + step * 16u + 2u * (lane & 3u) +
				8u * (index >> 1u))) : 0u;
		}
	#pragma unroll
	for (uint32_t frag = 0u; frag < VALUE_FRAGS; ++frag)
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;
	__syncthreads();
	tiles = (reach + TILE - 1u) / TILE;
	const uint32_t key_tiles_base = (uint32_t)__cvta_generic_to_shared(key_tiles);
	const uint32_t value_tiles_base = (uint32_t)__cvta_generic_to_shared(value_tiles);
	const uint64_t head_key_offset = (uint64_t)head * HEAD_DIM;
	const uint64_t head_value_offset = (uint64_t)heads * HEAD_DIM + (uint64_t)head * VALUE_DIM;
	const uint32_t warp_visible_max = __reduce_max_sync(0xffffffffu,visible_end[0] > visible_end[1] ? visible_end[0] : visible_end[1]);
	const uint32_t warp_visible_min = __reduce_min_sync(0xffffffffu,visible_end[0] < visible_end[1] ? visible_end[0] : visible_end[1]);

	auto load = [&](uint32_t tile, uint32_t buffer)
	{
		const uint32_t key_base = key_tiles_base + buffer * TILE * KEY_BYTES;
		const uint32_t value_base = value_tiles_base + buffer * TILE * VALUE_BYTES;
		const uint32_t warp_key0 = warp * KEYS_PER_WARP;
		if constexpr ( KEY_CHUNKS == 24u && VALUE_CHUNKS == 16u && KEYS_PER_WARP == 8u )
		{
			#pragma unroll
			for (uint32_t g = 0u; g < 2u; ++g)
			{
				const uint32_t k0 = warp_key0 + g * 4u;
				const uint32_t l0 = tile * TILE + k0;
				const uint16_t *s0 = slots + (uint64_t)(l0 < key_count ? l0 : 0u) * slot_pitch;
				const uint16_t *s1 = slots + (uint64_t)(l0 + 1u < key_count ? l0 + 1u : 0u) * slot_pitch;
				const uint16_t *s2 = slots + (uint64_t)(l0 + 2u < key_count ? l0 + 2u : 0u) * slot_pitch;
				const uint16_t *s3 = slots + (uint64_t)(l0 + 3u < key_count ? l0 + 3u : 0u) * slot_pitch;
				{
					const uint32_t kk = lane < 24u ? k0 : k0 + 1u;
					const uint32_t pp = lane < 24u ? lane : lane - 24u;
					const uint16_t *ss = lane < 24u ? s0 : s1;
					const uint32_t bytes = (tile * TILE + kk < key_count) ? 16u : 0u;
					LmPrefillCopy(key_base + kk * KEY_BYTES + LmPrefillSwizzle(kk,pp) * 16u,ss + head_key_offset + pp * 8u,bytes);
				}
				{
					const uint32_t kk = lane < 16u ? k0 + 1u : k0 + 2u;
					const uint32_t pp = lane < 16u ? lane + 8u : lane - 16u;
					const uint16_t *ss = lane < 16u ? s1 : s2;
					const uint32_t bytes = (tile * TILE + kk < key_count) ? 16u : 0u;
					LmPrefillCopy(key_base + kk * KEY_BYTES + LmPrefillSwizzle(kk,pp) * 16u,ss + head_key_offset + pp * 8u,bytes);
				}
				{
					const uint32_t kk = lane < 8u ? k0 + 2u : k0 + 3u;
					const uint32_t pp = lane < 8u ? lane + 16u : lane - 8u;
					const uint16_t *ss = lane < 8u ? s2 : s3;
					const uint32_t bytes = (tile * TILE + kk < key_count) ? 16u : 0u;
					LmPrefillCopy(key_base + kk * KEY_BYTES + LmPrefillSwizzle(kk,pp) * 16u,ss + head_key_offset + pp * 8u,bytes);
				}
				{
					const uint32_t kk = lane < 16u ? k0 : k0 + 1u;
					const uint32_t pp = lane & 15u;
					const uint16_t *ss = lane < 16u ? s0 : s1;
					const uint32_t bytes = (tile * TILE + kk < key_count) ? 16u : 0u;
					LmPrefillCopy(value_base + kk * VALUE_BYTES + LmPrefillSwizzle(kk,pp) * 16u,ss + head_value_offset + pp * 8u,bytes);
				}
				{
					const uint32_t kk = lane < 16u ? k0 + 2u : k0 + 3u;
					const uint32_t pp = lane & 15u;
					const uint16_t *ss = lane < 16u ? s2 : s3;
					const uint32_t bytes = (tile * TILE + kk < key_count) ? 16u : 0u;
					LmPrefillCopy(value_base + kk * VALUE_BYTES + LmPrefillSwizzle(kk,pp) * 16u,ss + head_value_offset + pp * 8u,bytes);
				}
			}
		}
		else if constexpr ( KEY_CHUNKS == 16u && VALUE_CHUNKS == 16u )
		{
			#pragma unroll
			for (uint32_t k = 0u; k < KEYS_PER_WARP; ++k)
			{
				const uint32_t kk = warp_key0 + k;
				const uint32_t local = tile * TILE + kk;
				const uint16_t *slot = slots + (uint64_t)(local < key_count ? local : 0u) * slot_pitch;
				const uint32_t pp = lane & 15u;
				const uint32_t bytes = local < key_count ? 16u : 0u;
				if ( lane < 16u )
					LmPrefillCopy(key_base + kk * KEY_BYTES + LmPrefillSwizzle(kk,pp) * 16u,slot + head_key_offset + pp * 8u,bytes);
				else
					LmPrefillCopy(value_base + kk * VALUE_BYTES + LmPrefillSwizzle(kk,pp) * 16u,slot + head_value_offset + pp * 8u,bytes);
			}
		}
		else
		{
			for (uint32_t item = threadIdx.x; item < TILE * (KEY_CHUNKS + VALUE_CHUNKS); item += LM_ATTENTION_BLOCK_THREADS)
			{
				const uint32_t key = item / (KEY_CHUNKS + VALUE_CHUNKS), piece = item % (KEY_CHUNKS + VALUE_CHUNKS), local = tile * TILE + key;
				const uint16_t *slot = slots + (uint64_t)(local < key_count ? local : 0u) * slot_pitch;
				if ( piece < KEY_CHUNKS )
					LmPrefillCopy(key_base + key * KEY_BYTES + LmPrefillSwizzle(key,piece) * 16u,slot + head_key_offset + piece * 8u,local < key_count ? 16u : 0u);
				else
					LmPrefillCopy(value_base + key * VALUE_BYTES + LmPrefillSwizzle(key,piece - KEY_CHUNKS) * 16u,
						slot + head_value_offset + (piece - KEY_CHUNKS) * 8u,local < key_count ? 16u : 0u);
			}
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};

	if ( tiles != 0u )
		load(0u,0u);
	for (uint32_t tile = 0u; tile < tiles; ++tile)
	{
		const uint32_t buffer = tile & 1u;
		const uint32_t tile_keys = key_tiles_base + buffer * TILE * KEY_BYTES;
		const uint32_t tile_values = value_tiles_base + buffer * TILE * VALUE_BYTES;
		if ( tile + 1u < tiles )
		{
			load(tile + 1u,buffer ^ 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		if ( tile * TILE < warp_visible_max )
		{
			float score[KEY_FRAGS][4], peak[2], rescale[2], shift[2];
			#pragma unroll
			for (uint32_t frag = 0u; frag < KEY_FRAGS; ++frag)
				score[frag][0] = score[frag][1] = score[frag][2] = score[frag][3] = 0.0f;
			#pragma unroll
			for (uint32_t step = 0u; step < K_STEPS; ++step)
				#pragma unroll
				for (uint32_t frag = 0u; frag < KEY_FRAGS; frag += 2u)
				{
					const uint32_t b_row = (frag + (matrix >> 1u)) * 8u + within, b_chunk = 2u * step + (matrix & 1u);
					uint32_t b[4];
					LmPrefillLoadX4(tile_keys + b_row * KEY_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u,b);
					LmMmaBf16(score[frag],query[step],b);
					LmMmaBf16(score[frag + 1u],query[step],b + 2);
				}
			peak[0] = peak[1] = -INFINITY;
			if ( (tile + 1u) * TILE <= warp_visible_min )
			{
				#pragma unroll
				for (uint32_t frag = 0u; frag < KEY_FRAGS; ++frag)
				{
					peak[0] = fmaxf(peak[0],fmaxf(score[frag][0],score[frag][1]));
					peak[1] = fmaxf(peak[1],fmaxf(score[frag][2],score[frag][3]));
				}
			}
			else
			{
				#pragma unroll
				for (uint32_t frag = 0u; frag < KEY_FRAGS; ++frag)
					#pragma unroll
					for (uint32_t entry = 0u; entry < 4u; ++entry)
					{
						const uint32_t key = tile * TILE + frag * 8u + 2u * (lane & 3u) + (entry & 1u), side = entry >> 1u;
						score[frag][entry] = key < visible_end[side] ? score[frag][entry] : -INFINITY;
						peak[side] = fmaxf(peak[side],score[frag][entry]);
					}
			}
			#pragma unroll
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
			if ( rescale[0] != 1.0f || rescale[1] != 1.0f )
			{
				#pragma unroll
				for (uint32_t frag = 0u; frag < VALUE_FRAGS; ++frag)
				{
					output[frag][0] *= rescale[0];
					output[frag][1] *= rescale[0];
					output[frag][2] *= rescale[1];
					output[frag][3] *= rescale[1];
				}
			}
			#pragma unroll
			for (uint32_t frag = 0u; frag < KEY_FRAGS; ++frag)
				#pragma unroll
				for (uint32_t entry = 0u; entry < 4u; ++entry)
				{
					const float weight = SparkMathExp2Fast(fmaf(score[frag][entry],scale_log2,-shift[entry >> 1u]));
					running_sum[entry >> 1u] += weight;
					score[frag][entry] = weight;
				}
			#pragma unroll
			for (uint32_t step = 0u; step < P_STEPS; ++step)
			{
				uint32_t weights[4];
				#pragma unroll
				for (uint32_t index = 0u; index < 4u; ++index)
				{
					const float *pair = &score[2u * step + (index >> 1u)][2u * (index & 1u)];
					weights[index] = LmPackBf16Pair(pair[0],pair[1]);
				}
				#pragma unroll
				for (uint32_t frag = 0u; frag < VALUE_FRAGS; frag += 2u)
				{
					const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within, b_chunk = frag + (matrix >> 1u);
					uint32_t b[4];
					LmPrefillLoadX4Trans(tile_values + b_row * VALUE_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u,b);
					LmMmaBf16(output[frag],weights,b);
					LmMmaBf16(output[frag + 1u],weights,b + 2);
				}
			}
		}
		__syncthreads();
	}
	#pragma unroll
	for (uint32_t side = 0u; side < 2u; ++side)
	{
		running_sum[side] += __shfl_xor_sync(0xffffffffu,running_sum[side],1u);
		running_sum[side] += __shfl_xor_sync(0xffffffffu,running_sum[side],2u);
	}
	#pragma unroll
	for (uint32_t side = 0u; side < 2u; ++side)
	{
		float *base = send + (uint64_t)row[side] * record;
		if ( !valid[side] )
			continue;
		if ( (lane & 3u) == 0u )
		{
			base[width + 2u * head] = running_max[side] == -INFINITY ? -INFINITY : running_max[side] * qk_scale;
			base[width + 2u * head + 1u] = running_sum[side];
		}
		#pragma unroll
		for (uint32_t frag = 0u; frag < VALUE_FRAGS; ++frag)
			*(float2 *)(base + head * VALUE_DIM + frag * 8u + 2u * (lane & 3u)) = make_float2(output[frag][2u * side],output[frag][2u * side + 1u]);
	}
	LmDependentRelease();
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static inline SparkGpuError LmAttentionBlockPartialLaunch(
	const uint16_t *query_bf16,
	uint32_t heads,
	const uint32_t *row_position,
	uint32_t rows,
	const uint16_t *slots,
	uint64_t slot_pitch,
	uint32_t key_first,
	uint32_t key_count,
	float qk_scale,
	float *send,
	uint64_t send_stride,
	SparkGpuStream stream)
{
	constexpr uint32_t shared = LmAttentionBlockSharedBytes<HEAD_DIM,VALUE_DIM>();
	static bool granted = false;
	if ( rows == 0u || heads == 0u || query_bf16 == 0 || row_position == 0 || slots == 0 || send == 0 || key_count == 0u ||
		slot_pitch < (uint64_t)heads * (HEAD_DIM + VALUE_DIM) ||
		send_stride < (uint64_t)rows * LmGqaShardRecordFloats(heads,VALUE_DIM,1u) )
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	if ( !granted )
	{
		const SparkGpuError error = SparkGpuFuncSetAttribute((const void *)LmAttentionBlockPartialKernel<HEAD_DIM,VALUE_DIM>,SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES,(int)shared);
		if ( error != SPARK_GPU_SUCCESS )
			return(error);
		granted = true;
	}
	LM_LAUNCH_DEPENDENT((LmAttentionBlockPartialKernel<HEAD_DIM,VALUE_DIM>),dim3((rows + LM_ATTENTION_BLOCK_QUERIES - 1u) / LM_ATTENTION_BLOCK_QUERIES,heads),LM_ATTENTION_BLOCK_THREADS,shared,stream,
		query_bf16,heads,row_position,rows,slots,slot_pitch,key_first,key_count,qk_scale,send);
	return(SparkGpuPeekAtLastError());
}
