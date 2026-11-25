#pragma once

#include "inference/kernels/attn_prefill.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/shard_reduce.cuh"
#include "runtime/launch.h"
#include <math.h>
#include <stdint.h>
#include "sparkpipe/spark_gpu.h"

#define LM_GQA_SHARD_MAX_GROUP 32u
#define LM_GQA_SHARD_TILED_WARPS 8u
#define LM_GQA_SHARD_TILED_THREADS (LM_GQA_SHARD_TILED_WARPS * 32u)
#define LM_GQA_SHARD_TILED_QUERIES (LM_GQA_SHARD_TILED_WARPS * 16u)
#define LM_GQA_SHARD_TILED_VALUE_PART 128u

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
	uint32_t window,
	const uint32_t *__restrict__ fork_positions,
	const uint32_t *__restrict__ slot_offsets)
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
	uint32_t sequence, limit, count, local, position, index, destination, first, last, element, fork, offset, start, phase, begin_key, end_key, shift, semantic;
	const uint8_t *slot;

	if ( row >= rows || warp >= group )
		return;
	sequence = sequence_of_row[row];
	if ( !LmKvCheckSequence(cache.pages,row,sequence) )
		return;
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
	fork = fork_positions != 0 && fork_positions[row] < limit ? fork_positions[row] : limit;
	offset = slot_offsets != 0 ? slot_offsets[row] : 0u;
	start = window != 0u && limit > window ? limit - window : 0u;
	for (phase = 0u; phase < (fork < limit ? 2u : 1u); ++phase)
	{
		begin_key = phase == 0u ? (start < fork ? start : fork) : (start > fork ? start : fork) + offset;
		end_key = phase == 0u ? fork : limit + offset;
		shift = phase == 0u ? 0u : offset;
		count = SparkKvShardLocalKeys(cache.shard,end_key);
		count = count < LmKvKeyWindowEnd(cache) ? count : LmKvKeyWindowEnd(cache);
		for (local = SparkKvShardLocalKeys(cache.shard,begin_key), local = local > LmKvKeyWindowBegin(cache) ? local : LmKvKeyWindowBegin(cache);
			local < count; ++local)
		{
			position = SparkKvShardLocalPosition(cache.shard,local);
			semantic = position - shift;
			if ( block_mask != 0 && semantic / block_tokens < limit / block_tokens &&
				((block_mask[(uint64_t)row * mask_words + semantic / block_tokens / 32u] >> ((semantic / block_tokens) % 32u)) & 1u) == 0u )
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
				accumulator[index] = fmaf(accumulator[index],rescale,SparkMathAttentionValueWeight(weight) * value[index]);
		}
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

template<uint32_t HEAD_DIM>
static __host__ __device__ constexpr uint32_t LmGqaShardTiledTile(void)
{
	return(HEAD_DIM <= 192u ? 64u : 32u);
}

template<uint32_t VALUE_DIM>
static __host__ __device__ constexpr uint32_t LmGqaShardTiledValuePart(void)
{
	return(VALUE_DIM < LM_GQA_SHARD_TILED_VALUE_PART ? VALUE_DIM : LM_GQA_SHARD_TILED_VALUE_PART);
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM, uint32_t GROUP_WARPS>
static __host__ __device__ constexpr uint32_t LmGqaShardTiledStages(uint32_t block_warps, bool single_row)
{
	return(single_row && block_warps == LM_GQA_SHARD_TILED_WARPS &&
		2u * LmGqaShardTiledTile<HEAD_DIM>() * (HEAD_DIM + LmGqaShardTiledValuePart<VALUE_DIM>() * GROUP_WARPS) * 2u +
		(GROUP_WARPS > 1u ? GROUP_WARPS * (LmGqaShardTiledTile<HEAD_DIM>() / 8u) * 512u : 0u) +
		2u * LmGqaShardTiledTile<HEAD_DIM>() * 8u + 6u * 4u <= LM_PREFILL_ATTN_SHARED_LIMIT ? 2u : 1u);
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM, uint32_t GROUP_WARPS>
static __host__ __device__ constexpr uint32_t LmGqaShardTiledSharedFor(uint32_t block_warps, bool single_row = false, uint32_t stage_override = 0u)
{
	return((stage_override != 0u ? stage_override : LmGqaShardTiledStages<HEAD_DIM,VALUE_DIM,GROUP_WARPS>(block_warps,single_row)) *
		LmGqaShardTiledTile<HEAD_DIM>() * (HEAD_DIM + LmGqaShardTiledValuePart<VALUE_DIM>() * GROUP_WARPS) * 2u +
		(GROUP_WARPS > 1u ? (single_row ? GROUP_WARPS : block_warps) * (LmGqaShardTiledTile<HEAD_DIM>() / 8u) * 512u : 0u) +
		2u * LmGqaShardTiledTile<HEAD_DIM>() * 8u + 6u * 4u);
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static __host__ __device__ constexpr uint32_t LmGqaShardTiledValueGroups(void)
{
	return((VALUE_DIM / LmGqaShardTiledValuePart<VALUE_DIM>()) % 4u == 0u && ((HEAD_DIM / 16u) % 4u) == 0u &&
		LmGqaShardTiledSharedFor<HEAD_DIM,VALUE_DIM,4u>(LM_GQA_SHARD_TILED_WARPS) <= LM_PREFILL_ATTN_SHARED_LIMIT ? 4u :
		(VALUE_DIM / LmGqaShardTiledValuePart<VALUE_DIM>()) % 2u == 0u && ((HEAD_DIM / 16u) % 2u) == 0u &&
		LmGqaShardTiledSharedFor<HEAD_DIM,VALUE_DIM,2u>(LM_GQA_SHARD_TILED_WARPS) <= LM_PREFILL_ATTN_SHARED_LIMIT ? 2u : 1u);
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static __host__ __device__ constexpr bool LmGqaShardTiledFits(void)
{
	return(HEAD_DIM % 64u == 0u && VALUE_DIM % 64u == 0u && HEAD_DIM <= 512u && VALUE_DIM % LmGqaShardTiledValuePart<VALUE_DIM>() == 0u);
}

template<uint32_t HEAD_DIM, uint32_t VALUE_DIM, uint32_t BLOCK_WARPS = LM_GQA_SHARD_TILED_WARPS, bool SINGLE_ROW = false, uint32_t STAGE_OVERRIDE = 0u>
static __host__ __device__ constexpr uint32_t LmGqaShardTiledSharedBytes(void)
{
	return(LmGqaShardTiledSharedFor<HEAD_DIM,VALUE_DIM,LmGqaShardTiledValueGroups<HEAD_DIM,VALUE_DIM>()>(BLOCK_WARPS,SINGLE_ROW,STAGE_OVERRIDE));
}

template<class Geometry, class Pages, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM, uint32_t BLOCK_WARPS = LM_GQA_SHARD_TILED_WARPS, bool SINGLE_ROW = false, uint32_t STAGE_OVERRIDE = 0u>
__global__ __launch_bounds__(BLOCK_WARPS * 32u, 1)
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
	uint32_t window,
	const uint32_t *__restrict__ fork_positions,
	const uint32_t *__restrict__ slot_offsets)
{
	constexpr uint32_t TILE = LmGqaShardTiledTile<HEAD_DIM>(), PART = LmGqaShardTiledValuePart<VALUE_DIM>();
	constexpr uint32_t V_GROUPS = LmGqaShardTiledValueGroups<HEAD_DIM,VALUE_DIM>();
	constexpr uint32_t STAGES = STAGE_OVERRIDE != 0u ? STAGE_OVERRIDE : LmGqaShardTiledStages<HEAD_DIM,VALUE_DIM,V_GROUPS>(BLOCK_WARPS,SINGLE_ROW);
	constexpr uint32_t BLOCK_THREADS = BLOCK_WARPS * 32u;
	constexpr uint32_t Q_TILES = SINGLE_ROW ? 1u : (BLOCK_WARPS / V_GROUPS);
	constexpr uint32_t BLOCK_QUERIES = Q_TILES * 16u;
	constexpr uint32_t BLOCK_VDIM = PART * V_GROUPS;
	constexpr uint32_t KEY_BYTES = HEAD_DIM * 2u, VALUE_BYTES = BLOCK_VDIM * 2u;
	constexpr uint32_t KEY_CHUNKS = HEAD_DIM / 8u, VALUE_CHUNKS = BLOCK_VDIM / 8u;
	constexpr uint32_t K_STEPS = HEAD_DIM / 16u, WARP_K_STEPS = K_STEPS / V_GROUPS;
	constexpr uint32_t KEY_FRAGS = TILE / 8u, P_STEPS = TILE / 16u, VALUE_FRAGS = PART / 8u;
	constexpr uint32_t SCORE_STRIDE = KEY_FRAGS * 4u * 32u;
	constexpr bool QUERY_RESIDENT = WARP_K_STEPS <= 16u;
	constexpr uint32_t QUERY_STEPS = QUERY_RESIDENT ? WARP_K_STEPS : 1u;
	static_assert(LmGqaShardTiledFits<HEAD_DIM,VALUE_DIM>(), "a warp holds a score tile and the accumulators of one value part in registers, and its 16 queries when they are at most 256 wide");
	static_assert(Geometry::kSlotBytes == KV_HEADS * (HEAD_DIM + VALUE_DIM) * 2u,
		"a slot is [K: kv heads x head_dim][V: kv heads x value_dim] bf16 and nothing else");
	static_assert(LmGqaShardTiledSharedBytes<HEAD_DIM,VALUE_DIM,BLOCK_WARPS,SINGLE_ROW,STAGE_OVERRIDE>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the key and value tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t lm_gqa_tiled_shared[];
	uint8_t *keys = lm_gqa_tiled_shared;
	uint8_t *values = keys + STAGES * TILE * KEY_BYTES;
	float *score_buf = (float *)(values + STAGES * TILE * VALUE_BYTES);
	const uint8_t **slots = (const uint8_t **)(score_buf + (V_GROUPS > 1u ? (uint32_t)Q_TILES * V_GROUPS * SCORE_STRIDE : 0u));
	uint32_t *run = (uint32_t *)(slots + 2u * TILE);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, matrix = lane >> 3u, within = lane & 7u;
	const uint32_t q_tile = warp / V_GROUPS, v_part = warp % V_GROUPS;
	const uint32_t kv_head = blockIdx.y, part_block = blockIdx.z, group = heads / KV_HEADS, degree = cache.shard.degree;
	const uint32_t head_tiles = SINGLE_ROW ? ((group + 15u) >> 4u) : 1u;
	const uint32_t single_row_id = SINGLE_ROW ? (blockIdx.x / head_tiles) : 0u;
	const uint32_t single_head_base = SINGLE_ROW ? ((blockIdx.x % head_tiles) * 16u) : 0u;
	const uint32_t vectors = rows * group, block_first = SINGLE_ROW ? 0u : blockIdx.x * BLOCK_QUERIES;
	const uint32_t block_last = SINGLE_ROW ? 0u : ((block_first + BLOCK_QUERIES < vectors ? block_first + BLOCK_QUERIES : vectors) - 1u);
	const uint32_t row_last = SINGLE_ROW ? single_row_id : (block_last / group);
	const bool warp_active = SINGLE_ROW ? (warp < V_GROUPS) : (block_first + q_tile * 16u < vectors);
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys), value_base = (uint32_t)__cvta_generic_to_shared(values);
	const uint32_t width = LmGqaShardWidth(heads,VALUE_DIM,degree), slot_count = LmGqaShardHeadSlots(heads,VALUE_DIM,degree);
	const uint32_t record = width + 2u * slot_count;
	const uint32_t vec_base = SINGLE_ROW ? (single_head_base + (lane >> 2u)) : (block_first + q_tile * 16u + (lane >> 2u));
	uint32_t query[QUERY_STEPS][4], limit[2], count[2], begin[2], extra_begin[2], extra_end[2], offset[2], cursor, run_end, sequence, low, high, branch_cut, branch_shift, trunk_high, branch_low, tile_low, tile_end, tile, index, step, frag, entry, side;
	float output[VALUE_FRAGS][4], running_max[2] = { -INFINITY, -INFINITY }, running_sum[2] = { 0.0f, 0.0f };
	const float scale_log2 = qk_scale * 1.4426950408889634f;
	bool valid[2], member[2];

	if ( SINGLE_ROW ? (single_row_id >= rows) : (block_first >= vectors) )
		return;
	#pragma unroll
	for (side = 0u; side < 2u; ++side)
	{
		const uint32_t v = vec_base + 8u * side;
		valid[side] = SINGLE_ROW ? (warp_active && v < group) : (v < vectors);
		limit[side] = count[side] = begin[side] = extra_begin[side] = extra_end[side] = offset[side] = 0u;
	}
	auto query_fragment = [&](uint32_t at, uint32_t *fragment)
	{
		#pragma unroll
		for (uint32_t item = 0u; item < 4u; ++item)
		{
			const uint32_t s = item & 1u;
			const uint32_t v = vec_base + 8u * s;
			const uint32_t r = SINGLE_ROW ? single_row_id : (v / group);
			const uint32_t h = kv_head * group + (SINGLE_ROW ? v : (v % group));
			fragment[item] = valid[s] ? *(const uint32_t *)(query_bf16 + ((uint64_t)r * heads + h) * HEAD_DIM +
				at * 16u + 2u * (lane & 3u) + 8u * (item >> 1u)) : 0u;
		}
	};
	if constexpr ( QUERY_RESIDENT )
	{
		if ( warp_active )
		{
			#pragma unroll
			for (step = 0u; step < WARP_K_STEPS; ++step)
				query_fragment(v_part * WARP_K_STEPS + step,query[step]);
		}
	}
	#pragma unroll
	for (frag = 0u; frag < VALUE_FRAGS; ++frag)
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;

	auto lookup_at = [&](uint32_t which, uint32_t slot_idx) -> const uint8_t *
	{
		const uint32_t local = which * TILE + slot_idx;
		if ( local < low || local >= high )
			return(0);
		if constexpr ( SINGLE_ROW )
		{
			const uint32_t mapped = local >= branch_cut ? local + branch_shift : local;
			return(LmKvShardSlotRequired<Geometry>(cache,sequence,SparkKvShardLocalPosition(cache.shard,mapped),cursor,LM_KV_ACCESS_READ));
		}
		if ( local >= trunk_high && local < branch_low )
			return(0);
		return(LmKvShardSlotRequired<Geometry>(cache,sequence,SparkKvShardLocalPosition(cache.shard,local),cursor,LM_KV_ACCESS_READ));
	};
	auto lookup = [&](uint32_t which) -> const uint8_t *
	{
		return(lookup_at(which,threadIdx.x));
	};
	auto copy_k = [&](uint32_t which)
	{
		const uint32_t slot_buf = which & 1u;
		const uint32_t stage_base = key_base + (STAGES == 2u ? slot_buf * TILE * KEY_BYTES : 0u);
		uint32_t item, key, piece;
		for (item = threadIdx.x; item < TILE * KEY_CHUNKS; item += BLOCK_THREADS)
		{
			key = item / KEY_CHUNKS;
			piece = item % KEY_CHUNKS;
			const uint8_t *slot = slots[slot_buf * TILE + key];
			LmPrefillCopy(stage_base + key * KEY_BYTES + LmPrefillSwizzle(key,piece) * 16u,
				slot != 0 ? (const void *)(slot + (kv_head * HEAD_DIM + piece * 8u) * 2u) : (const void *)query_bf16, slot != 0 ? 16u : 0u);
		}
		if constexpr ( STAGES == 1u )
			asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	auto copy_v = [&](uint32_t which)
	{
		const uint32_t slot_buf = which & 1u;
		const uint32_t stage_base = value_base + (STAGES == 2u ? slot_buf * TILE * VALUE_BYTES : 0u);
		uint32_t item, key, piece;
		for (item = threadIdx.x; item < TILE * VALUE_CHUNKS; item += BLOCK_THREADS)
		{
			key = item / VALUE_CHUNKS;
			piece = item % VALUE_CHUNKS;
			const uint8_t *slot = slots[slot_buf * TILE + key];
			LmPrefillCopy(stage_base + key * VALUE_BYTES + LmPrefillSwizzle(key,piece) * 16u,
				slot != 0 ? (const void *)(slot + (KV_HEADS * HEAD_DIM + kv_head * VALUE_DIM + part_block * BLOCK_VDIM + piece * 8u) * 2u) : (const void *)query_bf16,
				slot != 0 ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};

	for (cursor = SINGLE_ROW ? single_row_id : (block_first / group); cursor <= row_last; cursor = run_end + 1u)
	{
		if constexpr ( SINGLE_ROW )
		{
			sequence = sequence_of_row[single_row_id];
			run_end = single_row_id;
			if ( !LmKvCheckSequence(cache.pages,single_row_id,sequence,0xffffffffu,threadIdx.x == 0u) )
				return;
			uint32_t lim = context_length[sequence];
			if ( row_position != 0 && row_position[single_row_id] != LM_KV_POSITION_UNUSED && row_position[single_row_id] + 1u < lim )
				lim = row_position[single_row_id] + 1u;
			const uint32_t fork = fork_positions != 0 && fork_positions[single_row_id] < lim ? fork_positions[single_row_id] : lim;
			const uint32_t off = slot_offsets != 0 ? slot_offsets[single_row_id] : 0u;
			const uint32_t st = window != 0u && lim > window ? lim - window : 0u;
			uint32_t cnt = SparkKvShardLocalKeys(cache.shard,fork);
			cnt = cnt < LmKvKeyWindowEnd(cache) ? cnt : LmKvKeyWindowEnd(cache);
			uint32_t beg = SparkKvShardLocalKeys(cache.shard,st < fork ? st : fork);
			beg = beg > LmKvKeyWindowBegin(cache) ? beg : LmKvKeyWindowBegin(cache);
			uint32_t ebeg = 0u, eend = 0u;
			if ( fork < lim )
			{
				ebeg = SparkKvShardLocalKeys(cache.shard,(st > fork ? st : fork) + off);
				ebeg = ebeg > LmKvKeyWindowBegin(cache) ? ebeg : LmKvKeyWindowBegin(cache);
				eend = SparkKvShardLocalKeys(cache.shard,lim + off);
				eend = eend < LmKvKeyWindowEnd(cache) ? eend : LmKvKeyWindowEnd(cache);
			}
			const uint32_t branch_len = ebeg < eend ? eend - ebeg : 0u;
			if ( beg < cnt )
			{
				low = beg;
				high = cnt + branch_len;
				branch_cut = cnt;
				branch_shift = branch_len != 0u ? ebeg - cnt : 0u;
			}
			else if ( branch_len != 0u )
			{
				uint32_t bbase = SparkKvShardLocalKeys(cache.shard,st);
				bbase = bbase > LmKvKeyWindowBegin(cache) ? bbase : LmKvKeyWindowBegin(cache);
				if ( bbase <= ebeg )
				{
					low = bbase;
					high = bbase + branch_len;
					branch_cut = bbase;
					branch_shift = ebeg - bbase;
				}
				else
				{
					low = ebeg;
					high = eend;
					branch_cut = ebeg;
					branch_shift = 0u;
				}
			}
			else
			{
				low = 0xffffffffu;
				high = 0u;
				branch_cut = 0u;
				branch_shift = 0u;
			}
			trunk_high = 0u;
			branch_low = 0xffffffffu;
			#pragma unroll
			for (side = 0u; side < 2u; ++side)
			{
				member[side] = valid[side];
				limit[side] = lim;
				count[side] = low < high ? high : 0u;
				begin[side] = low < high ? low : 0u;
				extra_begin[side] = 0u;
				extra_end[side] = 0u;
				offset[side] = off;
			}
		}
		else
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
				run[4] = 0u;
				run[5] = 0xffffffffu;
			}
			__syncthreads();
			sequence = run[0];
			run_end = run[1];
			if ( !LmKvCheckSequence(cache.pages,cursor,sequence,0xffffffffu,threadIdx.x == 0u) )
				return;
			#pragma unroll
			for (side = 0u; side < 2u; ++side)
			{
				uint32_t fork, start;
				const uint32_t r = valid[side] ? ((vec_base + 8u * side) / group) : 0u;
				member[side] = valid[side] && r >= cursor && r <= run_end;
				extra_begin[side] = extra_end[side] = offset[side] = 0u;
				if ( !member[side] )
					continue;
				limit[side] = context_length[sequence];
				if ( row_position != 0 && row_position[r] != LM_KV_POSITION_UNUSED && row_position[r] + 1u < limit[side] )
					limit[side] = row_position[r] + 1u;
				fork = fork_positions != 0 && fork_positions[r] < limit[side] ? fork_positions[r] : limit[side];
				offset[side] = slot_offsets != 0 ? slot_offsets[r] : 0u;
				start = window != 0u && limit[side] > window ? limit[side] - window : 0u;
				count[side] = SparkKvShardLocalKeys(cache.shard,fork);
				count[side] = count[side] < LmKvKeyWindowEnd(cache) ? count[side] : LmKvKeyWindowEnd(cache);
				begin[side] = SparkKvShardLocalKeys(cache.shard,start < fork ? start : fork);
				begin[side] = begin[side] > LmKvKeyWindowBegin(cache) ? begin[side] : LmKvKeyWindowBegin(cache);
				if ( fork < limit[side] )
				{
					extra_begin[side] = SparkKvShardLocalKeys(cache.shard,(start > fork ? start : fork) + offset[side]);
					extra_begin[side] = extra_begin[side] > LmKvKeyWindowBegin(cache) ? extra_begin[side] : LmKvKeyWindowBegin(cache);
					extra_end[side] = SparkKvShardLocalKeys(cache.shard,limit[side] + offset[side]);
					extra_end[side] = extra_end[side] < LmKvKeyWindowEnd(cache) ? extra_end[side] : LmKvKeyWindowEnd(cache);
				}
				if ( v_part == 0u && (lane & 3u) == 0u && begin[side] < count[side] )
				{
					atomicMin(&run[2],begin[side]);
					atomicMax(&run[3],count[side]);
					atomicMax(&run[4],count[side]);
				}
				if ( v_part == 0u && (lane & 3u) == 0u && extra_begin[side] < extra_end[side] )
				{
					atomicMin(&run[2],extra_begin[side]);
					atomicMax(&run[3],extra_end[side]);
					atomicMin(&run[5],extra_begin[side]);
				}
			}
			__syncthreads();
			low = run[2];
			high = run[3];
			trunk_high = run[4];
			branch_low = run[5];
			branch_cut = 0u;
			branch_shift = 0u;
		}
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
		{
			copy_k(tile_low);
			if constexpr ( STAGES == 2u )
				copy_v(tile_low);
		}
		for (tile = tile_low; tile < tile_end; ++tile)
		{
			const uint32_t slot_buf = tile & 1u;
			const uint32_t tile_keys = key_base + (STAGES == 2u ? slot_buf * TILE * KEY_BYTES : 0u);
			const uint32_t tile_values = value_base + (STAGES == 2u ? slot_buf * TILE * VALUE_BYTES : 0u);
			const uint8_t *ahead = 0;
			float score[KEY_FRAGS][4], peak[2], next_max[2], rescale[2], shift[2];
			uint32_t b[4], first[2], second[2], weights[4];
			bool whole;
			if constexpr ( STAGES == 2u )
			{
				if ( tile + 1u < tile_end )
				{
					copy_k(tile + 1u);
					copy_v(tile + 1u);
					asm volatile("cp.async.wait_group 1;\n" ::: "memory");
				}
				else
					asm volatile("cp.async.wait_group 0;\n" ::: "memory");
				__syncthreads();
				constexpr uint32_t SLOT_BASE = BLOCK_THREADS >= 128u + TILE ? 128u : (SINGLE_ROW && BLOCK_THREADS >= V_GROUPS * 32u + TILE ? V_GROUPS * 32u : 0u);
				if ( threadIdx.x >= SLOT_BASE && threadIdx.x < SLOT_BASE + TILE && tile + 2u < tile_end )
					slots[slot_buf * TILE + (threadIdx.x - SLOT_BASE)] = lookup_at(tile + 2u,threadIdx.x - SLOT_BASE);
			}
			else
			{
				asm volatile("cp.async.wait_group 0;\n" ::: "memory");
				__syncthreads();
				copy_v(tile);
				if ( threadIdx.x < TILE && tile + 2u < tile_end )
					ahead = lookup(tile + 2u);
			}
			if ( warp_active )
			{
				#pragma unroll
				for (frag = 0u; frag < KEY_FRAGS; ++frag)
					score[frag][0] = score[frag][1] = score[frag][2] = score[frag][3] = 0.0f;
				#pragma unroll
				for (step = 0u; step < WARP_K_STEPS; ++step)
				{
					const uint32_t k_step = v_part * WARP_K_STEPS + step;
					uint32_t *fragment = query[QUERY_RESIDENT ? step : 0u];
					if constexpr ( !QUERY_RESIDENT )
						query_fragment(k_step,fragment);
					#pragma unroll
					for (frag = 0u; frag < KEY_FRAGS; frag += 2u)
					{
						const uint32_t b_row = (frag + (matrix >> 1u)) * 8u + within, b_chunk = 2u * k_step + (matrix & 1u);
						LmPrefillLoadX4(tile_keys + b_row * KEY_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u,b);
						first[0] = b[0]; first[1] = b[1];
						second[0] = b[2]; second[1] = b[3];
						LmMmaBf16(score[frag],fragment,first);
						LmMmaBf16(score[frag + 1u],fragment,second);
					}
				}
				if constexpr ( V_GROUPS > 1u )
				{
					float *my_score = score_buf + ((q_tile * V_GROUPS + v_part) * SCORE_STRIDE);
					#pragma unroll
					for (frag = 0u; frag < KEY_FRAGS; ++frag)
						#pragma unroll
						for (entry = 0u; entry < 4u; ++entry)
							my_score[(frag * 4u + entry) * 32u + lane] = score[frag][entry];
				}
			}
			if constexpr ( STAGES == 1u )
			{
				asm volatile("cp.async.wait_group 0;\n" ::: "memory");
				__syncthreads();
				if ( tile + 1u < tile_end )
					copy_k(tile + 1u);
				if ( threadIdx.x < TILE && tile + 2u < tile_end )
					slots[slot_buf * TILE + threadIdx.x] = ahead;
			}
			else if constexpr ( V_GROUPS > 1u )
				__syncthreads();
			if ( warp_active )
			{
				if constexpr ( V_GROUPS > 1u )
				{
					const float *tile_score = score_buf + (q_tile * V_GROUPS * SCORE_STRIDE);
					#pragma unroll
					for (frag = 0u; frag < KEY_FRAGS; ++frag)
						#pragma unroll
						for (entry = 0u; entry < 4u; ++entry)
						{
							const uint32_t off = (frag * 4u + entry) * 32u + lane;
							float s = tile_score[off];
							#pragma unroll
							for (uint32_t g = 1u; g < V_GROUPS; ++g)
								s += tile_score[g * SCORE_STRIDE + off];
							score[frag][entry] = s;
						}
				}
				peak[0] = peak[1] = -INFINITY;
				whole = __all_sync(0xffffffffu,block_mask == 0 && member[0] && member[1] &&
					((begin[0] <= tile * TILE && tile * TILE + TILE <= count[0]) || (extra_begin[0] <= tile * TILE && tile * TILE + TILE <= extra_end[0])) &&
					((begin[1] <= tile * TILE && tile * TILE + TILE <= count[1]) || (extra_begin[1] <= tile * TILE && tile * TILE + TILE <= extra_end[1])));
				if ( whole )
				{
					#pragma unroll
					for (frag = 0u; frag < KEY_FRAGS; ++frag)
						#pragma unroll
						for (entry = 0u; entry < 4u; ++entry)
							peak[entry >> 1u] = fmaxf(peak[entry >> 1u],score[frag][entry]);
				}
				else
				{
					#pragma unroll
					for (frag = 0u; frag < KEY_FRAGS; ++frag)
						#pragma unroll
						for (entry = 0u; entry < 4u; ++entry)
						{
							const uint32_t local = tile * TILE + frag * 8u + 2u * (lane & 3u) + (entry & 1u), which = entry >> 1u;
							const bool branch = SINGLE_ROW ? (local >= branch_cut) : (local >= extra_begin[which] && local < extra_end[which]);
							bool visible = member[which] && ((local >= begin[which] && local < count[which]) || (!SINGLE_ROW && branch));
							if ( visible && block_mask != 0 )
							{
								const uint32_t r = SINGLE_ROW ? single_row_id : ((vec_base + 8u * which) / group);
								const uint32_t mapped = SINGLE_ROW && branch ? local + branch_shift : local;
								const uint32_t position = SparkKvShardLocalPosition(cache.shard,mapped) - (branch ? offset[which] : 0u);
								visible = position / block_tokens >= limit[which] / block_tokens ||
									((block_mask[(uint64_t)r * mask_words + position / block_tokens / 32u] >> ((position / block_tokens) % 32u)) & 1u) != 0u;
							}
							score[frag][entry] = visible ? score[frag][entry] : -INFINITY;
							peak[which] = fmaxf(peak[which],score[frag][entry]);
						}
				}
				#pragma unroll
				for (side = 0u; side < 2u; ++side)
				{
					peak[side] = fmaxf(peak[side],__shfl_xor_sync(0xffffffffu,peak[side],1u));
					peak[side] = fmaxf(peak[side],__shfl_xor_sync(0xffffffffu,peak[side],2u));
					next_max[side] = fmaxf(running_max[side],peak[side]);
					rescale[side] = running_max[side] == -INFINITY ? 0.0f : SparkMathExp2Fast((running_max[side] - next_max[side]) * scale_log2);
					running_max[side] = next_max[side];
					shift[side] = next_max[side] == -INFINITY ? 0.0f : next_max[side] * scale_log2;
					running_sum[side] *= rescale[side];
				}
				if ( rescale[0] != 1.0f || rescale[1] != 1.0f )
				{
					#pragma unroll
					for (frag = 0u; frag < VALUE_FRAGS; ++frag)
					{
						output[frag][0] *= rescale[0];
						output[frag][1] *= rescale[0];
						output[frag][2] *= rescale[1];
						output[frag][3] *= rescale[1];
					}
				}
				#pragma unroll
				for (frag = 0u; frag < KEY_FRAGS; ++frag)
					#pragma unroll
					for (entry = 0u; entry < 4u; ++entry)
					{
						const float weight = SparkMathExp2Fast(fmaf(score[frag][entry],scale_log2,-shift[entry >> 1u]));
						running_sum[entry >> 1u] += weight;
						score[frag][entry] = weight;
					}
				#pragma unroll
				for (step = 0u; step < P_STEPS; ++step)
				{
					#pragma unroll
					for (index = 0u; index < 4u; ++index)
					{
						const float *pair = &score[2u * step + (index >> 1u)][2u * (index & 1u)];
						weights[index] = LmPackBf16Pair(pair[0],pair[1]);
					}
					#pragma unroll
					for (frag = 0u; frag < VALUE_FRAGS; frag += 2u)
					{
						const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within, b_chunk = v_part * (PART / 8u) + frag + (matrix >> 1u);
						LmPrefillLoadX4Trans(tile_values + b_row * VALUE_BYTES + LmPrefillSwizzle(b_row,b_chunk) * 16u,b);
						first[0] = b[0]; first[1] = b[1];
						second[0] = b[2]; second[1] = b[3];
						LmMmaBf16(output[frag],weights,first);
						LmMmaBf16(output[frag + 1u],weights,second);
					}
				}
			}
			if constexpr ( STAGES == 2u )
				__syncthreads();
		}
		if constexpr ( !SINGLE_ROW )
			__syncthreads();
	}
	if ( !warp_active )
		return;
	#pragma unroll
	for (side = 0u; side < 2u; ++side)
	{
		running_sum[side] += __shfl_xor_sync(0xffffffffu,running_sum[side],1u);
		running_sum[side] += __shfl_xor_sync(0xffffffffu,running_sum[side],2u);
	}
	#pragma unroll
	for (side = 0u; side < 2u; ++side)
	{
		uint32_t destination, element;
		if ( !valid[side] )
			continue;
		const uint32_t v = vec_base + 8u * side;
		const uint32_t r = SINGLE_ROW ? single_row_id : (v / group);
		const uint32_t h = kv_head * group + (SINGLE_ROW ? v : (v % group));
		const uint32_t start = h * VALUE_DIM, finish = start + VALUE_DIM, columns = start + (part_block * V_GROUPS + v_part) * PART;
		for (destination = start / width; destination < degree && destination * width < finish; ++destination)
		{
			float *base = send + (uint64_t)destination * destination_stride + (uint64_t)r * record;
			if ( part_block == 0u && v_part == 0u && (lane & 3u) == 0u )
			{
				base[width + 2u * (h - (destination * width) / VALUE_DIM)] = running_max[side] == -INFINITY ? -INFINITY : running_max[side] * qk_scale;
				base[width + 2u * (h - (destination * width) / VALUE_DIM) + 1u] = running_sum[side];
			}
			#pragma unroll
			for (frag = 0u; frag < VALUE_FRAGS; ++frag)
				#pragma unroll
				for (entry = 0u; entry < 2u; ++entry)
				{
					element = columns + frag * 8u + 2u * (lane & 3u) + entry;
					if ( element >= destination * width && element < (destination + 1u) * width )
						base[element - destination * width] = output[frag][2u * side + entry];
				}
		}
	}
}

#define LM_GQA_SHARD_MERGE_SLOTS 64u

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
	__shared__ float scales[LM_GQA_SHARD_MERGE_SLOTS][SPARK_KV_SHARD_MAX_DEGREE];
	__shared__ float denom_shared[LM_GQA_SHARD_MERGE_SLOTS];
	const uint32_t row = blockIdx.x;
	const uint32_t width = LmGqaShardWidth(heads,VALUE_DIM,sources), slots = LmGqaShardHeadSlots(heads,VALUE_DIM,sources);
	const uint32_t record = width + 2u * slots, origin = (rank * width) / VALUE_DIM;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	if ( row >= rows || width == 0u )
		return;
	const uint32_t used_slots = (rank * width + width - 1u) / VALUE_DIM - origin + 1u;
	const float *row_base = received + (uint64_t)row * record;
	uint16_t *out_row = output_bf16 + (uint64_t)row * width;
	for (uint32_t slot_base = 0u; slot_base < used_slots; slot_base += LM_GQA_SHARD_MERGE_SLOTS)
	{
		const uint32_t slot_end = slot_base + LM_GQA_SHARD_MERGE_SLOTS < used_slots ? slot_base + LM_GQA_SHARD_MERGE_SLOTS : used_slots;
		const uint32_t chunk_slots = slot_end - slot_base;
		const uint32_t elem_begin = slot_base == 0u ? 0u : (origin + slot_base) * VALUE_DIM - rank * width;
		const uint32_t elem_end = slot_end == used_slots ? width : (origin + slot_end) * VALUE_DIM - rank * width;
		for (uint32_t local_slot = warp; local_slot < chunk_slots; local_slot += 8u)
		{
			const uint32_t slot_index = slot_base + local_slot;
			const float sink = sinks != 0 ? sinks[origin + slot_index] : -INFINITY;
			float gm = -INFINITY;
			float den = LmWarpMergeScales(row_base + width + 2u * slot_index, source_stride, sources, scales[local_slot], &gm, sink);
			if ( lane == 0u )
			{
				if ( sinks != 0 )
					den += __expf(sink - gm);
				denom_shared[local_slot] = fmaxf(den,1.0e-20f);
			}
		}
		__syncthreads();
		LmWeightedSourcesMergeRangeBf16<VALUE_DIM, 256u>(
			row_base, source_stride, sources, elem_begin, elem_end, out_row,
			[&](uint32_t elem) { return (rank * width + elem) / VALUE_DIM - origin - slot_base; },
			[&](uint32_t slot, uint32_t s) { return scales[slot][s]; },
			[&](uint32_t slot) { return denom_shared[slot]; },
			rank * width);
		if ( slot_end < used_slots )
			__syncthreads();
	}
}

template<class Geometry, class Pages, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static inline SparkGpuError LmGqaShardPartialLaunch(
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
	SparkGpuStream stream,
	const uint32_t *block_mask = 0,
	uint32_t mask_words = 0u,
	uint32_t block_tokens = 1u,
	uint32_t window = 0u,
	const uint32_t *fork_positions = 0,
	const uint32_t *slot_offsets = 0)
{
	const uint32_t degree = cache.shard.degree;
	if ( rows == 0u || query_bf16 == 0 || send == 0 || sequence_of_row == 0 || context_length == 0 || (fork_positions == 0) != (slot_offsets == 0) ||
		LmGqaShardGeometryValid(heads,KV_HEADS,VALUE_DIM,degree) == 0u ||
		SparkKvShardValid(cache.shard,Geometry::kPageSlots) == 0u ||
		destination_stride < (uint64_t)rows * LmGqaShardRecordFloats(heads,VALUE_DIM,degree) ||
		(block_mask != 0 && (mask_words == 0u || block_tokens == 0u)) )
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	if constexpr ( LmGqaShardTiledFits<HEAD_DIM,VALUE_DIM>() )
	{
		constexpr uint32_t PART = LmGqaShardTiledValuePart<VALUE_DIM>();
		constexpr uint32_t V_GROUPS = LmGqaShardTiledValueGroups<HEAD_DIM,VALUE_DIM>();
		constexpr uint32_t Z_PARTS = VALUE_DIM / (PART * V_GROUPS);
		const uint32_t group = heads / KV_HEADS;
		if ( rows <= 64u )
		{
			const uint32_t head_tiles = (group + 15u) >> 4u;
			if ( (uint64_t)rows * head_tiles * KV_HEADS * Z_PARTS <= 24u )
			{
				constexpr uint32_t DECODE_WARPS = LM_GQA_SHARD_TILED_WARPS;
				constexpr uint32_t shared = LmGqaShardTiledSharedBytes<HEAD_DIM,VALUE_DIM,DECODE_WARPS,true>();
				static bool granted_decode_wide = false;
				if ( !granted_decode_wide )
				{
					const SparkGpuError error = SparkGpuFuncSetAttribute((const void *)LmGqaShardTiledKernel<Geometry,Pages,KV_HEADS,HEAD_DIM,VALUE_DIM,DECODE_WARPS,true>,
						SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES,(int)shared);
					if ( error != SPARK_GPU_SUCCESS )
						return(error);
					granted_decode_wide = true;
				}
				LM_LAUNCH((LmGqaShardTiledKernel<Geometry,Pages,KV_HEADS,HEAD_DIM,VALUE_DIM,DECODE_WARPS,true>),
					dim3(rows * head_tiles,KV_HEADS,Z_PARTS),DECODE_WARPS * 32u,shared,stream,
					cache,query_bf16,heads,sequence_of_row,context_length,row_position,qk_scale,send,destination_stride,rows,block_mask,mask_words,block_tokens,window,
					fork_positions,slot_offsets);
				return(SparkGpuPeekAtLastError());
			}
			constexpr uint32_t DECODE_WARPS = 4u;
			constexpr uint32_t shared = LmGqaShardTiledSharedBytes<HEAD_DIM,VALUE_DIM,DECODE_WARPS,true>();
			static bool granted_decode = false;
			if ( !granted_decode )
			{
				const SparkGpuError error = SparkGpuFuncSetAttribute((const void *)LmGqaShardTiledKernel<Geometry,Pages,KV_HEADS,HEAD_DIM,VALUE_DIM,DECODE_WARPS,true>,
					SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES,(int)shared);
				if ( error != SPARK_GPU_SUCCESS )
					return(error);
				granted_decode = true;
			}
			LM_LAUNCH((LmGqaShardTiledKernel<Geometry,Pages,KV_HEADS,HEAD_DIM,VALUE_DIM,DECODE_WARPS,true>),
				dim3(rows * head_tiles,KV_HEADS,Z_PARTS),DECODE_WARPS * 32u,shared,stream,
				cache,query_bf16,heads,sequence_of_row,context_length,row_position,qk_scale,send,destination_stride,rows,block_mask,mask_words,block_tokens,window,
				fork_positions,slot_offsets);
			return(SparkGpuPeekAtLastError());
		}
		constexpr uint32_t PREFILL_QUERIES = (LM_GQA_SHARD_TILED_WARPS / V_GROUPS) * 16u;
		constexpr uint32_t shared = LmGqaShardTiledSharedBytes<HEAD_DIM,VALUE_DIM,LM_GQA_SHARD_TILED_WARPS>();
		static bool granted = false;
		if ( !granted )
		{
			const SparkGpuError error = SparkGpuFuncSetAttribute((const void *)LmGqaShardTiledKernel<Geometry,Pages,KV_HEADS,HEAD_DIM,VALUE_DIM,LM_GQA_SHARD_TILED_WARPS,false>,
				SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES,(int)shared);
			if ( error != SPARK_GPU_SUCCESS )
				return(error);
			granted = true;
		}
		LM_LAUNCH((LmGqaShardTiledKernel<Geometry,Pages,KV_HEADS,HEAD_DIM,VALUE_DIM,LM_GQA_SHARD_TILED_WARPS,false>),
			dim3((rows * group + PREFILL_QUERIES - 1u) / PREFILL_QUERIES,KV_HEADS,Z_PARTS),LM_GQA_SHARD_TILED_THREADS,shared,stream,
			cache,query_bf16,heads,sequence_of_row,context_length,row_position,qk_scale,send,destination_stride,rows,block_mask,mask_words,block_tokens,window,
			fork_positions,slot_offsets);
		return(SparkGpuPeekAtLastError());
	}
	LM_LAUNCH((LmGqaShardPartialKernel<Geometry,Pages,KV_HEADS,HEAD_DIM,VALUE_DIM>),dim3(rows,KV_HEADS),(heads / KV_HEADS) * 32u,0,stream,
		cache,query_bf16,heads,sequence_of_row,context_length,row_position,qk_scale,send,destination_stride,rows,block_mask,mask_words,block_tokens,window,
		fork_positions,slot_offsets);
	return(SparkGpuPeekAtLastError());
}

template<uint32_t VALUE_DIM>
static inline SparkGpuError LmGqaShardMergeLaunch(
	const float *received,
	uint64_t source_stride,
	uint32_t sources,
	uint32_t heads,
	uint32_t rank,
	uint16_t *output_bf16,
	uint32_t rows,
	SparkGpuStream stream,
	const float *sinks = 0)
{
	if ( rows == 0u || received == 0 || output_bf16 == 0 || sources == 0u || sources > SPARK_KV_SHARD_MAX_DEGREE ||
		rank >= sources || ((uint64_t)heads * VALUE_DIM) % sources != 0u ||
		source_stride < (uint64_t)rows * LmGqaShardRecordFloats(heads,VALUE_DIM,sources) )
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	LM_LAUNCH((LmGqaShardMergeKernel<VALUE_DIM>),rows,256u,0,stream,received,source_stride,sources,heads,rank,output_bf16,rows,sinks);
	return(SparkGpuPeekAtLastError());
}
