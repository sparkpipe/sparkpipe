#pragma once

#include "inference/kernels/attn_prefill.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "runtime/launch.h"
#include <math.h>
#include <stdint.h>

#define LM_GQA_SHARD_MAX_GROUP 32u
#define LM_GQA_SHARD_TILED_WARPS 8u
#define LM_GQA_SHARD_TILED_THREADS (LM_GQA_SHARD_TILED_WARPS * 32u)
#define LM_GQA_SHARD_TILED_QUERIES (LM_GQA_SHARD_TILED_WARPS * 16u)
#define LM_GQA_SHARD_TILED_TILE 64u

static __host__ __device__ __forceinline__ uint32_t LmGqaShardWidth(uint32_t heads, uint32_t value_dim, uint32_t degree)
{
	return(heads * value_dim / degree);
}

static __host__ __device__ __forceinline__ uint32_t LmGqaShardHeadSlots(uint32_t heads, uint32_t value_dim, uint32_t degree)
{
	return((LmGqaShardWidth(heads,value_dim,degree) + value_dim - 1u) / value_dim + 1u);
}

static __host__ __device__ __forceinline__ uint32_t LmGqaShardRecordFloats(uint32_t heads, uint32_t value_dim, uint32_t degree)
{
	return(LmGqaShardWidth(heads,value_dim,degree) + 2u * LmGqaShardHeadSlots(heads,value_dim,degree));
}

static __host__ __forceinline__ uint32_t LmGqaShardGeometryValid(uint32_t heads, uint32_t kv_heads, uint32_t value_dim, uint32_t degree)
{
	return(degree != 0u && degree <= SPARK_KV_SHARD_MAX_DEGREE && kv_heads != 0u && heads >= kv_heads &&
		(heads % kv_heads) == 0u && heads / kv_heads <= LM_GQA_SHARD_MAX_GROUP &&
		((uint64_t)heads * value_dim) % degree == 0u ? 1u : 0u);
}

template<uint32_t COUNT>
static __device__ __forceinline__ void LmGqaShardLoad(const uint16_t *__restrict__ source, float *values)
{
	uint32_t index;
	#pragma unroll
	for (index = 0u; index < COUNT; index += 2u)
	{
		const uint32_t word = *(const uint32_t *)(source + index);
		values[index] = __uint_as_float(word << 16u);
		values[index + 1u] = __uint_as_float(word & 0xffff0000u);
	}
}

template<class Geometry, class Pages, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(LM_GQA_SHARD_MAX_GROUP * 32u, 1)
void LmGqaShardPartialKernel(
	Pages cache,
	const uint16_t *__restrict__ query_bf16,
	uint32_t heads,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position,
	float qk_scale,
	float *__restrict__ send,
	uint64_t destination_stride,
	uint32_t rows,
	const uint32_t *__restrict__ block_mask,
	uint32_t mask_words,
	uint32_t block_tokens,
	uint32_t window)
{
	constexpr uint32_t KEYS = HEAD_DIM / 32u;
	constexpr uint32_t VALUES = VALUE_DIM / 32u;
	static_assert(HEAD_DIM % 64u == 0u && VALUE_DIM % 64u == 0u, "each lane reads whole bf16 pairs of a head");
	static_assert(Geometry::kSlotBytes == KV_HEADS * (HEAD_DIM + VALUE_DIM) * 2u,
		"a slot is [K: kv heads x head_dim][V: kv heads x value_dim] bf16 and nothing else");
	const uint32_t row = blockIdx.x, kv_head = blockIdx.y, warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
	const uint32_t group = heads / KV_HEADS, head = kv_head * group + warp;
	const uint32_t degree = cache.shard.degree;
	const uint32_t width = LmGqaShardWidth(heads,VALUE_DIM,degree), slots = LmGqaShardHeadSlots(heads,VALUE_DIM,degree);
	const uint32_t record = width + 2u * slots;
	float query[KEYS], key[KEYS], value[VALUES], accumulator[VALUES];
	float running_max = -INFINITY, running_sum = 0.0f, score, previous, rescale, weight;
	uint32_t sequence, limit, count, local, position, index, destination, first, last, element;
	const uint8_t *slot;

	if ( row >= rows || warp >= group )
		return;
	sequence = sequence_of_row[row];
	if ( !LmKvViewIsConfigured(cache.pages) || sequence >= cache.pages.sequence_count )
	{
		LmKvReportRequiredAccessFailure(cache.pages,
			!LmKvViewIsConfigured(cache.pages) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
			LM_KV_ACCESS_READ,row,sequence,0xffffffffu,0xffffffffu);
		return;
	}
	if ( (heads % KV_HEADS) != 0u || group > LM_GQA_SHARD_MAX_GROUP || ((uint64_t)heads * VALUE_DIM) % degree != 0u )
	{
		LmKvReportRequiredAccessFailure(cache.pages,LM_KV_ACCESS_ERROR_INVALID_GQA_GEOMETRY,LM_KV_ACCESS_READ,row,sequence,head,heads);
		return;
	}
	limit = context_length[sequence];
	if ( row_position != 0 && row_position[row] != LM_KV_POSITION_UNUSED && row_position[row] + 1u < limit )
		limit = row_position[row] + 1u;
	LmGqaShardLoad<KEYS>(query_bf16 + ((uint64_t)row * heads + head) * HEAD_DIM + lane * KEYS,query);
	#pragma unroll
	for (index = 0u; index < VALUES; ++index)
		accumulator[index] = 0.0f;
	count = SparkKvShardLocalKeys(cache.shard,limit);
	for (local = window != 0u && limit > window ? SparkKvShardLocalKeys(cache.shard,limit - window) : 0u; local < count; ++local)
	{
		position = SparkKvShardLocalPosition(cache.shard,local);
		if ( block_mask != 0 && position / block_tokens < limit / block_tokens &&
			((block_mask[(uint64_t)row * mask_words + position / block_tokens / 32u] >> ((position / block_tokens) % 32u)) & 1u) == 0u )
			continue;
		slot = LmKvShardSlotRequired<Geometry>(cache,sequence,position,row,LM_KV_ACCESS_READ);
		if ( slot == 0 )
			return;
		LmGqaShardLoad<KEYS>((const uint16_t *)slot + kv_head * HEAD_DIM + lane * KEYS,key);
		LmGqaShardLoad<VALUES>((const uint16_t *)slot + KV_HEADS * HEAD_DIM + kv_head * VALUE_DIM + lane * VALUES,value);
		score = 0.0f;
		#pragma unroll
		for (index = 0u; index < KEYS; ++index)
			score = fmaf(query[index],key[index],score);
		#pragma unroll
		for (index = 16u; index > 0u; index >>= 1u)
			score += __shfl_xor_sync(0xffffffffu,score,index);
		score *= qk_scale;
		previous = running_max;
		running_max = fmaxf(previous,score);
		rescale = __expf(previous - running_max);
		weight = __expf(score - running_max);
		running_sum = fmaf(running_sum,rescale,weight);
		#pragma unroll
		for (index = 0u; index < VALUES; ++index)
			accumulator[index] = fmaf(accumulator[index],rescale,weight * value[index]);
	}
	first = head * VALUE_DIM;
	last = first + VALUE_DIM;
	for (destination = first / width; destination < degree && destination * width < last; ++destination)
	{
		float *base = send + (uint64_t)destination * destination_stride + (uint64_t)row * record;
		const uint32_t slot_index = head - (destination * width) / VALUE_DIM;
		if ( lane == 0u )
		{
			base[width + 2u * slot_index] = running_max;
			base[width + 2u * slot_index + 1u] = running_sum;
		}
		#pragma unroll
		for (index = 0u; index < VALUES; ++index)
		{
			element = first + lane * VALUES + index;
			if ( element >= destination * width && element < (destination + 1u) * width )
				base[element - destination * width] = accumulator[index];
		}
	}
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static __host__ __device__ constexpr bool LmGqaShardTiledFits(void)
{
	return(HEAD_DIM % 64u == 0u && VALUE_DIM % 64u == 0u && HEAD_DIM <= 192u && VALUE_DIM <= 128u);
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static __host__ __device__ constexpr uint32_t LmGqaShardTiledSharedBytes(void)
{
	return(2u * LM_GQA_SHARD_TILED_TILE * (HEAD_DIM + VALUE_DIM) * 2u + 2u * LM_GQA_SHARD_TILED_TILE * 8u + 4u * 4u);
}

template<class Geometry, class Pages, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(LM_GQA_SHARD_TILED_THREADS, 1)
void LmGqaShardTiledKernel(
	Pages cache,
	const uint16_t *__restrict__ query_bf16,
	uint32_t heads,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position,
	float qk_scale,
	float *__restrict__ send,
	uint64_t destination_stride,
	uint32_t rows,
	const uint32_t *__restrict__ block_mask,
	uint32_t mask_words,
	uint32_t block_tokens,
	uint32_t window)
{
	constexpr uint32_t TILE = LM_GQA_SHARD_TILED_TILE;
	constexpr uint32_t KEY_BYTES = HEAD_DIM * 2u, VALUE_BYTES = VALUE_DIM * 2u;
	constexpr uint32_t KEY_CHUNKS = HEAD_DIM / 8u, VALUE_CHUNKS = VALUE_DIM / 8u;
	constexpr uint32_t K_STEPS = HEAD_DIM / 16u, KEY_FRAGS = TILE / 8u, P_STEPS = TILE / 16u, VALUE_FRAGS = VALUE_DIM / 8u;
	static_assert(LmGqaShardTiledFits<HEAD_DIM,VALUE_DIM>(), "a warp holds its 16 queries, a 64-key score tile and the value accumulators in registers");
	static_assert(Geometry::kSlotBytes == KV_HEADS * (HEAD_DIM + VALUE_DIM) * 2u,
		"a slot is [K: kv heads x head_dim][V: kv heads x value_dim] bf16 and nothing else");
	static_assert(LmGqaShardTiledSharedBytes<HEAD_DIM,VALUE_DIM>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the key and value tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t lm_gqa_tiled_shared[];
	uint8_t *keys = lm_gqa_tiled_shared;
	uint8_t *values = keys + 2u * TILE * KEY_BYTES;
	const uint8_t **slots = (const uint8_t **)(values + 2u * TILE * VALUE_BYTES);
	uint32_t *run = (uint32_t *)(slots + 2u * TILE);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, matrix = lane >> 3u, within = lane & 7u;
	const uint32_t kv_head = blockIdx.y, group = heads / KV_HEADS, degree = cache.shard.degree;
	const uint32_t vectors = rows * group, block_first = blockIdx.x * LM_GQA_SHARD_TILED_QUERIES;
	const uint32_t block_last = (block_first + LM_GQA_SHARD_TILED_QUERIES < vectors ? block_first + LM_GQA_SHARD_TILED_QUERIES : vectors) - 1u;
	const uint32_t row_last = block_last / group;
	const bool warp_active = block_first + warp * 16u < vectors;
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys), value_base = (uint32_t)__cvta_generic_to_shared(values);
	const uint32_t width = LmGqaShardWidth(heads,VALUE_DIM,degree), slot_count = LmGqaShardHeadSlots(heads,VALUE_DIM,degree);
	const uint32_t record = width + 2u * slot_count;
	uint32_t query[K_STEPS][4], vector[2], row[2], head[2], limit[2], count[2], begin[2], cursor, run_end, sequence, low, high, tile_low, tile_end, tile, index, step, frag, entry, side;
	float output[VALUE_FRAGS][4], running_max[2] = { -INFINITY, -INFINITY }, running_sum[2] = { 0.0f, 0.0f };
	bool valid[2], member[2];

	if ( block_first >= vectors )
		return;
	for (side = 0u; side < 2u; ++side)
	{
		vector[side] = block_first + warp * 16u + (lane >> 2u) + 8u * side;
		valid[side] = vector[side] < vectors;
		row[side] = valid[side] ? vector[side] / group : 0u;
		head[side] = kv_head * group + (valid[side] ? vector[side] % group : 0u);
		limit[side] = count[side] = begin[side] = 0u;
	}
	#pragma unroll
	for (step = 0u; step < K_STEPS; ++step)
		#pragma unroll
		for (index = 0u; index < 4u; ++index)
		{
			side = index % 2u;
			query[step][index] = valid[side] ? *(const uint32_t *)(query_bf16 + ((uint64_t)row[side] * heads + head[side]) * HEAD_DIM +
				step * 16u + 2u * (lane & 3u) + 8u * (index / 2u)) : 0u;
		}
	#pragma unroll
	for (frag = 0u; frag < VALUE_FRAGS; ++frag)
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;

	auto lookup = [&](uint32_t which) -> const uint8_t *
	{
		const uint32_t local = which * TILE + threadIdx.x;
		return(local < high && local >= low ? LmKvShardSlotRequired<Geometry>(cache,sequence,SparkKvShardLocalPosition(cache.shard,local),cursor,LM_KV_ACCESS_READ) : 0);
	};
	auto copy = [&](uint32_t which)
	{
		const uint32_t buffer = which & 1u;
		uint32_t item, key, piece;
		for (item = threadIdx.x; item < TILE * (KEY_CHUNKS + VALUE_CHUNKS); item += LM_GQA_SHARD_TILED_THREADS)
		{
			key = item / (KEY_CHUNKS + VALUE_CHUNKS);
			piece = item % (KEY_CHUNKS + VALUE_CHUNKS);
			const uint8_t *slot = slots[buffer * TILE + key];
			if ( piece < KEY_CHUNKS )
				LmPrefillCopy(key_base + (buffer * TILE + key) * KEY_BYTES + LmPrefillSwizzle(key,piece) * 16u,
					slot != 0 ? (const void *)(slot + (kv_head * HEAD_DIM + piece * 8u) * 2u) : (const void *)query_bf16, slot != 0 ? 16u : 0u);
			else
				LmPrefillCopy(value_base + (buffer * TILE + key) * VALUE_BYTES + LmPrefillSwizzle(key,piece - KEY_CHUNKS) * 16u,
					slot != 0 ? (const void *)(slot + (KV_HEADS * HEAD_DIM + kv_head * VALUE_DIM + (piece - KEY_CHUNKS) * 8u) * 2u) : (const void *)query_bf16,
					slot != 0 ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};

	for (cursor = block_first / group; cursor <= row_last; cursor = run_end + 1u)
	{
		if ( threadIdx.x == 0u )
		{
			run[0] = sequence_of_row[cursor];
			for (run_end = cursor; run_end < row_last && sequence_of_row[run_end + 1u] == run[0]; ++run_end)
			{
			}
			run[1] = run_end;
			run[2] = 0xffffffffu;
			run[3] = 0u;
		}
		__syncthreads();
		sequence = run[0];
		run_end = run[1];
		if ( !LmKvViewIsConfigured(cache.pages) || sequence >= cache.pages.sequence_count )
		{
			if ( threadIdx.x == 0u )
				LmKvReportRequiredAccessFailure(cache.pages,
					!LmKvViewIsConfigured(cache.pages) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
					LM_KV_ACCESS_READ,cursor,sequence,0xffffffffu,0xffffffffu);
			return;
		}
		for (side = 0u; side < 2u; ++side)
		{
			member[side] = valid[side] && row[side] >= cursor && row[side] <= run_end;
			if ( !member[side] )
				continue;
			limit[side] = context_length[sequence];
			if ( row_position != 0 && row_position[row[side]] != LM_KV_POSITION_UNUSED && row_position[row[side]] + 1u < limit[side] )
				limit[side] = row_position[row[side]] + 1u;
			count[side] = SparkKvShardLocalKeys(cache.shard,limit[side]);
			begin[side] = window != 0u && limit[side] > window ? SparkKvShardLocalKeys(cache.shard,limit[side] - window) : 0u;
			if ( (lane & 3u) == 0u && begin[side] < count[side] )
			{
				atomicMin(&run[2],begin[side]);
				atomicMax(&run[3],count[side]);
			}
		}
		__syncthreads();
		low = run[2];
		high = run[3];
		tile_low = low < high ? low / TILE : 0u;
		tile_end = low < high ? (high + TILE - 1u) / TILE : 0u;
		if ( threadIdx.x < TILE && tile_low < tile_end )
		{
			slots[(tile_low & 1u) * TILE + threadIdx.x] = lookup(tile_low);
			if ( tile_low + 1u < tile_end )
				slots[((tile_low + 1u) & 1u) * TILE + threadIdx.x] = lookup(tile_low + 1u);
		}
		__syncthreads();
		if ( tile_low < tile_end )
			copy(tile_low);
		for (tile = tile_low; tile < tile_end; ++tile)
		{
			const uint32_t buffer = tile & 1u;
			const uint32_t tile_keys = key_base + buffer * TILE * KEY_BYTES, tile_values = value_base + buffer * TILE * VALUE_BYTES;
			const uint8_t *ahead = threadIdx.x < TILE && tile + 2u < tile_end ? lookup(tile + 2u) : 0;
			float score[KEY_FRAGS][4], peak[2], next_max[2], rescale[2];
			uint32_t b[4], first[2], second[2], high_bits[4], low_bits[4];
			if ( tile + 1u < tile_end )
			{
				copy(tile + 1u);
				asm volatile("cp.async.wait_group 1;\n" ::: "memory");
			}
			else
				asm volatile("cp.async.wait_group 0;\n" ::: "memory");
			__syncthreads();
			if ( !warp_active )
			{
				if ( threadIdx.x < TILE && tile + 2u < tile_end )
					slots[buffer * TILE + threadIdx.x] = ahead;
				__syncthreads();
				continue;
			}
			#pragma unroll
			for (frag = 0u; frag < KEY_FRAGS; ++frag)
				score[frag][0] = score[frag][1] = score[frag][2] = score[frag][3] = 0.0f;
			#pragma unroll
			for (step = 0u; step < K_STEPS; ++step)
				#pragma unroll
				for (frag = 0u; frag < KEY_FRAGS; frag += 2u)
				{
					const uint32_t b_row = (frag + (matrix >> 1u)) * 8u + within, b_chunk = 2u * step + (matrix & 1u);
					LmPrefillLoadX4(tile_keys + b_row * KEY_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u,b);
					first[0] = b[0]; first[1] = b[1];
					second[0] = b[2]; second[1] = b[3];
					LmMmaBf16(score[frag],query[step],first);
					LmMmaBf16(score[frag + 1u],query[step],second);
				}
			peak[0] = peak[1] = -INFINITY;
			#pragma unroll
			for (frag = 0u; frag < KEY_FRAGS; ++frag)
				#pragma unroll
				for (entry = 0u; entry < 4u; ++entry)
				{
					const uint32_t local = tile * TILE + frag * 8u + 2u * (lane & 3u) + (entry & 1u), which = entry / 2u;
					bool visible = member[which] && local >= begin[which] && local < count[which];
					if ( visible && block_mask != 0 )
					{
						const uint32_t position = SparkKvShardLocalPosition(cache.shard,local);
						visible = position / block_tokens >= limit[which] / block_tokens ||
							((block_mask[(uint64_t)row[which] * mask_words + position / block_tokens / 32u] >> ((position / block_tokens) % 32u)) & 1u) != 0u;
					}
					score[frag][entry] = visible ? score[frag][entry] * qk_scale : -INFINITY;
					peak[which] = fmaxf(peak[which],score[frag][entry]);
				}
			for (side = 0u; side < 2u; ++side)
			{
				peak[side] = fmaxf(peak[side],__shfl_xor_sync(0xffffffffu,peak[side],1u));
				peak[side] = fmaxf(peak[side],__shfl_xor_sync(0xffffffffu,peak[side],2u));
				next_max[side] = fmaxf(running_max[side],peak[side]);
				rescale[side] = running_max[side] == -INFINITY ? 0.0f : __expf(running_max[side] - next_max[side]);
				running_max[side] = next_max[side];
				running_sum[side] *= rescale[side];
			}
			#pragma unroll
			for (frag = 0u; frag < VALUE_FRAGS; ++frag)
			{
				output[frag][0] *= rescale[0];
				output[frag][1] *= rescale[0];
				output[frag][2] *= rescale[1];
				output[frag][3] *= rescale[1];
			}
			#pragma unroll
			for (frag = 0u; frag < KEY_FRAGS; ++frag)
				#pragma unroll
				for (entry = 0u; entry < 4u; ++entry)
				{
					const float weight = score[frag][entry] == -INFINITY ? 0.0f : __expf(score[frag][entry] - running_max[entry / 2u]);
					running_sum[entry / 2u] += weight;
					score[frag][entry] = weight;
				}
			#pragma unroll
			for (step = 0u; step < P_STEPS; ++step)
			{
				#pragma unroll
				for (index = 0u; index < 4u; ++index)
				{
					const float *pair = &score[2u * step + index / 2u][2u * (index % 2u)];
					const uint16_t high_first = LmFloatToBf16(pair[0]), high_second = LmFloatToBf16(pair[1]);
					high_bits[index] = (uint32_t)high_first | ((uint32_t)high_second << 16u);
					low_bits[index] = LmPackBf16Pair(pair[0] - __uint_as_float((uint32_t)high_first << 16u),pair[1] - __uint_as_float((uint32_t)high_second << 16u));
				}
				#pragma unroll
				for (frag = 0u; frag < VALUE_FRAGS; frag += 2u)
				{
					const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within, b_chunk = frag + (matrix >> 1u);
					LmPrefillLoadX4Trans(tile_values + b_row * VALUE_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u,b);
					first[0] = b[0]; first[1] = b[1];
					second[0] = b[2]; second[1] = b[3];
					LmMmaBf16(output[frag],high_bits,first);
					LmMmaBf16(output[frag],low_bits,first);
					LmMmaBf16(output[frag + 1u],high_bits,second);
					LmMmaBf16(output[frag + 1u],low_bits,second);
				}
			}
			if ( threadIdx.x < TILE && tile + 2u < tile_end )
				slots[buffer * TILE + threadIdx.x] = ahead;
			__syncthreads();
		}
		__syncthreads();
	}
	for (side = 0u; side < 2u; ++side)
	{
		running_sum[side] += __shfl_xor_sync(0xffffffffu,running_sum[side],1u);
		running_sum[side] += __shfl_xor_sync(0xffffffffu,running_sum[side],2u);
	}
	for (side = 0u; side < 2u; ++side)
	{
		uint32_t destination, element;
		const uint32_t start = head[side] * VALUE_DIM, finish = start + VALUE_DIM;
		if ( !valid[side] )
			continue;
		for (destination = start / width; destination < degree && destination * width < finish; ++destination)
		{
			float *base = send + (uint64_t)destination * destination_stride + (uint64_t)row[side] * record;
			if ( (lane & 3u) == 0u )
			{
				base[width + 2u * (head[side] - (destination * width) / VALUE_DIM)] = running_max[side];
				base[width + 2u * (head[side] - (destination * width) / VALUE_DIM) + 1u] = running_sum[side];
			}
			#pragma unroll
			for (frag = 0u; frag < VALUE_FRAGS; ++frag)
				#pragma unroll
				for (entry = 0u; entry < 2u; ++entry)
				{
					element = start + frag * 8u + 2u * (lane & 3u) + entry;
					if ( element >= destination * width && element < (destination + 1u) * width )
						base[element - destination * width] = output[frag][2u * side + entry];
				}
		}
	}
}

template<uint32_t VALUE_DIM>
__global__ __launch_bounds__(256, 1)
void LmGqaShardMergeKernel(
	const float *__restrict__ received,
	uint64_t source_stride,
	uint32_t sources,
	uint32_t heads,
	uint32_t rank,
	uint16_t *__restrict__ output_bf16,
	uint32_t rows,
	const float *__restrict__ sinks)
{
	const uint32_t row = blockIdx.x;
	const uint32_t width = LmGqaShardWidth(heads,VALUE_DIM,sources), slots = LmGqaShardHeadSlots(heads,VALUE_DIM,sources);
	const uint32_t record = width + 2u * slots, origin = (rank * width) / VALUE_DIM;
	uint32_t element, source, slot_index;
	float global_max, source_max, scale, denominator, merged, sink = 0.0f;
	const float *base;

	if ( row >= rows )
		return;
	for (element = threadIdx.x; element < width; element += blockDim.x)
	{
		slot_index = (rank * width + element) / VALUE_DIM - origin;
		global_max = -INFINITY;
		for (source = 0u; source < sources; ++source)
		{
			source_max = received[(uint64_t)source * source_stride + (uint64_t)row * record + width + 2u * slot_index];
			global_max = source_max > global_max ? source_max : global_max;
		}
		if ( sinks != 0 )
		{
			sink = sinks[(rank * width + element) / VALUE_DIM];
			global_max = sink > global_max ? sink : global_max;
		}
		denominator = 0.0f;
		merged = 0.0f;
		for (source = 0u; source < sources; ++source)
		{
			base = received + (uint64_t)source * source_stride + (uint64_t)row * record;
			source_max = base[width + 2u * slot_index];
			scale = global_max == -INFINITY || source_max == -INFINITY ? 0.0f : __expf(source_max - global_max);
			denominator = fmaf(base[width + 2u * slot_index + 1u],scale,denominator);
			merged = fmaf(base[element],scale,merged);
		}
		if ( sinks != 0 )
			denominator += __expf(sink - global_max);
		output_bf16[(uint64_t)row * width + element] = LmFloatToBf16(merged / fmaxf(denominator,1.0e-20f));
	}
}

template<class Geometry, class Pages, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static inline cudaError_t LmGqaShardPartialLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint32_t heads,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *send,
	uint64_t destination_stride,
	uint32_t rows,
	cudaStream_t stream,
	const uint32_t *block_mask = 0,
	uint32_t mask_words = 0u,
	uint32_t block_tokens = 1u,
	uint32_t window = 0u)
{
	const uint32_t degree = cache.shard.degree;
	if ( rows == 0u || query_bf16 == 0 || send == 0 || sequence_of_row == 0 || context_length == 0 ||
		LmGqaShardGeometryValid(heads,KV_HEADS,VALUE_DIM,degree) == 0u ||
		SparkKvShardValid(cache.shard,Geometry::kPageSlots) == 0u ||
		destination_stride < (uint64_t)rows * LmGqaShardRecordFloats(heads,VALUE_DIM,degree) ||
		(block_mask != 0 && (mask_words == 0u || block_tokens == 0u)) )
		return(cudaErrorInvalidValue);
	if constexpr ( LmGqaShardTiledFits<HEAD_DIM,VALUE_DIM>() )
	{
		constexpr uint32_t shared = LmGqaShardTiledSharedBytes<HEAD_DIM,VALUE_DIM>();
		static bool granted = false;
		if ( !granted )
		{
			const cudaError_t error = cudaFuncSetAttribute((const void *)LmGqaShardTiledKernel<Geometry,Pages,KV_HEADS,HEAD_DIM,VALUE_DIM>,
				cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shared);
			if ( error != cudaSuccess )
				return(error);
			granted = true;
		}
		LmGqaShardTiledKernel<Geometry,Pages,KV_HEADS,HEAD_DIM,VALUE_DIM><<<dim3((rows * (heads / KV_HEADS) + LM_GQA_SHARD_TILED_QUERIES - 1u) /
			LM_GQA_SHARD_TILED_QUERIES,KV_HEADS),LM_GQA_SHARD_TILED_THREADS,shared,stream>>>(
			cache,query_bf16,heads,sequence_of_row,context_length,row_position,qk_scale,send,destination_stride,rows,block_mask,mask_words,block_tokens,window);
		return(cudaPeekAtLastError());
	}
	LmGqaShardPartialKernel<Geometry,Pages,KV_HEADS,HEAD_DIM,VALUE_DIM><<<dim3(rows,KV_HEADS),(heads / KV_HEADS) * 32u,0,stream>>>(
		cache,query_bf16,heads,sequence_of_row,context_length,row_position,qk_scale,send,destination_stride,rows,block_mask,mask_words,block_tokens,window);
	return(cudaPeekAtLastError());
}

template<uint32_t VALUE_DIM>
static inline cudaError_t LmGqaShardMergeLaunch(
	const float *received,
	uint64_t source_stride,
	uint32_t sources,
	uint32_t heads,
	uint32_t rank,
	uint16_t *output_bf16,
	uint32_t rows,
	cudaStream_t stream,
	const float *sinks = 0)
{
	if ( rows == 0u || received == 0 || output_bf16 == 0 || sources == 0u || sources > SPARK_KV_SHARD_MAX_DEGREE ||
		rank >= sources || ((uint64_t)heads * VALUE_DIM) % sources != 0u ||
		source_stride < (uint64_t)rows * LmGqaShardRecordFloats(heads,VALUE_DIM,sources) )
		return(cudaErrorInvalidValue);
	LM_LAUNCH((LmGqaShardMergeKernel<VALUE_DIM>),rows,256u,0,stream,received,source_stride,sources,heads,rank,output_bf16,rows,sinks);
	return(cudaPeekAtLastError());
}
