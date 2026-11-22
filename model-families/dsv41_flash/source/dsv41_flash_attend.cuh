#pragma once

#include "inference/kernels/attn_shard_prefill.cuh"
#include "dsv41_flash_stage_kernels.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"

#define D41A_QUERIES 16u
#define D41A_TILE 32u
#define D41A_WARPS 8u
#define D41A_CONSUMERS (D41A_WARPS * 32u)
#define D41A_THREADS ((D41A_WARPS + 2u) * 32u)
#define D41A_WINDOW 128u
#define D41A_WINDOW_TILES (D41A_WINDOW / D41A_TILE)
#define D41A_PLANS 4u
#define D41A_PLAN_WORDS 16u
#define D41A_SCORE_STRIDE (D41A_TILE + 1u)
#define D41A_WEIGHT_STRIDE (D41A_TILE + 8u)
#define D41A_PLANNER D41A_WARPS
#define D41A_ISSUER (D41A_WARPS + 1u)

template<uint32_t DIM, uint32_t QTILES>
static __host__ __device__ constexpr uint32_t D41AttendFixedBytes(void)
{
	return 2u * D41A_TILE * (DIM * 2u + LM_LATENT_SHARD_ROW_PAD) + 2u * QTILES * D41A_QUERIES * D41A_SCORE_STRIDE * 4u +
		QTILES * D41A_QUERIES * D41A_WEIGHT_STRIDE * 2u + 4u * QTILES * D41A_QUERIES * 4u + (5u + 2u * D41A_PLANS) * 8u + D41A_PLANS * D41A_PLAN_WORDS * 4u;
}

template<uint32_t DIM, uint32_t QTILES>
static __host__ __device__ inline uint64_t D41AttendSharedBytes(uint32_t listed)
{
	return D41AttendFixedBytes<DIM, QTILES>() + ((uint64_t)listed + 2u * (D41A_THREADS / 32u)) * sizeof(uint32_t);
}

static __device__ __forceinline__ void D41AttendBarrier(void)
{
	asm volatile("bar.sync 1, %0;\n" :: "n"(D41A_CONSUMERS) : "memory");
}

template<class Geometry, uint32_t DIM, uint32_t QTILES>
__global__ __launch_bounds__(D41A_THREADS, 1)
void D41AttendKernel(
	LmKvShardView cache,
	uint32_t compressed,
	const uint16_t *__restrict__ query,
	uint64_t query_part_stride,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ length,
	const uint32_t *__restrict__ bound,
	const uint32_t *__restrict__ selected,
	uint32_t topk,
	const uint16_t *__restrict__ wave_kv,
	const uint16_t *__restrict__ ring,
	uint64_t ring_slot_stride,
	uint32_t ring_rows,
	const uint32_t *__restrict__ positions,
	const uint32_t *__restrict__ row_lane,
	const uint32_t *__restrict__ sequence_row_begin,
	const uint32_t *__restrict__ sequence_row_indices,
	const uint32_t *__restrict__ recurrent_slot,
	const float *__restrict__ sink,
	float qk_scale,
	uint16_t *__restrict__ output,
	uint64_t output_part_stride,
	const float *__restrict__ frequency,
	uint32_t rope_offset,
	uint32_t group_heads,
	uint8_t *__restrict__ codes,
	uint64_t codes_part_stride,
	uint32_t rows,
	uint32_t live,
	uint32_t row_base)
{
	constexpr uint32_t ROW_BYTES = DIM * 2u;
	constexpr uint32_t PITCH = ROW_BYTES + LM_LATENT_SHARD_ROW_PAD;
	constexpr uint32_t CHUNKS = DIM / 8u;
	constexpr uint32_t QUERIES = D41A_QUERIES;
	constexpr uint32_t QT = QTILES * QUERIES;
	constexpr uint32_t TILE = D41A_TILE;
	constexpr uint32_t HALF_STEPS = DIM / 32u;
	constexpr uint32_t WARP_COLUMNS = DIM / D41A_WARPS;
	constexpr uint32_t COLUMN_FRAGS = WARP_COLUMNS / 8u;
	constexpr uint32_t PLANS = D41A_PLANS, PLAN_WORDS = D41A_PLAN_WORDS;
	static_assert(DIM % 64u == 0u && HALF_STEPS % 2u == 0u, "each warp multiplies an even number of 16-wide steps over half the width");
	static_assert(WARP_COLUMNS % 16u == 0u, "each warp owns whole 16-column value slices");
	static_assert(PITCH % 128u == 16u, "a padded row starts four banks after the row before it");
	static_assert(Geometry::kSlotBytes >= ROW_BYTES, "the slot holds the latent");
	static_assert(QTILES == 1u || QTILES == 2u, "a warp's keys split evenly over the query tiles");
	static_assert(D41AttendFixedBytes<DIM, QTILES>() + 4096u <= LM_PREFILL_ATTN_SHARED_LIMIT, "the tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t d41a_shared[];
	uint8_t *keys = d41a_shared;
	float *score_parts = (float *)(keys + 2u * TILE * PITCH);
	uint16_t *weight_high = (uint16_t *)(score_parts + 2u * QT * D41A_SCORE_STRIDE);
	float *running_max = (float *)(weight_high + QT * D41A_WEIGHT_STRIDE);
	float *running_sum = running_max + QT;
	float *rescale = running_sum + QT;
	float *denominator = rescale + QT;
	uint64_t *full = (uint64_t *)(denominator + QT);
	uint64_t *empty = full + 2u;
	uint64_t *planned = empty + 2u;
	uint64_t *read = planned + PLANS;
	uint64_t *listed_free = read + PLANS;
	uint32_t *plans = (uint32_t *)(listed_free + 1u);
	uint32_t *list_shared = plans + PLANS * PLAN_WORDS;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	const uint32_t qk_half = warp & 1u, query_tile = (warp >> 1u) % QTILES, key_group = (warp >> 1u) / QTILES;
	const uint32_t own = threadIdx.x / 16u, part = threadIdx.x % 16u;
	uint32_t block, ordinal, used, first_compacted = 0u;

	if (threadIdx.x == 0u)
	{
		LmMbarrierInit(&full[0], 1u);
		LmMbarrierInit(&full[1], 1u);
		LmMbarrierInit(&empty[0], D41A_WARPS);
		LmMbarrierInit(&empty[1], D41A_WARPS);
		for (uint32_t index = 0u; index < PLANS; index++)
		{
			LmMbarrierInit(&planned[index], 1u);
			LmMbarrierInit(&read[index], D41A_WARPS + 1u);
		}
		LmMbarrierInit(listed_free, 1u);
		LmMbarrierInitFence();
	}
	if (compressed != 0u && selected != 0 && blockIdx.x < live && LmKvViewIsConfigured(cache.pages) &&
		sequence_of_row[blockIdx.x] < cache.pages.sequence_count)
	{
		const uint32_t first_bound = bound[blockIdx.x];
		uint32_t first_dense = length[sequence_of_row[blockIdx.x]];
		if (first_bound == 0xffffffffu)
			first_dense = 0u;
		else if (first_bound < first_dense)
			first_dense = first_bound + 1u;
		if (first_dense > topk)
			first_compacted = LmLatentShardCompactBlock<D41A_THREADS>(cache.shard, selected + (uint64_t)blockIdx.x * topk, topk, first_bound, list_shared,
				list_shared + topk);
	}
	__syncthreads();
	if (warp == D41A_PLANNER)
	{
		uint32_t compactions = 0u;
		for (block = blockIdx.x, ordinal = 0u; block < live; block += gridDim.x, ordinal++)
		{
			const uint32_t record = ordinal % PLANS, row = block;
			const uint32_t step_lane = row_lane[row], begin = sequence_row_begin[step_lane];
			const uint32_t position = positions[row_base + row], wave_first = positions[sequence_row_indices[begin]];
			const uint32_t sequence = compressed != 0u ? sequence_of_row[row] : 0u;
			const uint32_t valid = compressed == 0u || (LmKvViewIsConfigured(cache.pages) && sequence < cache.pages.sequence_count) ? 1u : 0u;
			uint32_t listed = 0u, count = 0u, window_begin = 0u, limit = 0u, tile_first = 0u, tile_end = 0u;
			if (compressed != 0u && valid != 0u)
			{
				const uint32_t row_bound = bound[row];
				uint32_t dense = length[sequence];
				if (row_bound == 0xffffffffu)
					dense = 0u;
				else if (row_bound < dense)
					dense = row_bound + 1u;
				listed = selected != 0 && dense > topk ? 1u : 0u;
				if (listed != 0u)
				{
					if (compactions > 0u)
						LmMbarrierWait(listed_free, (compactions - 1u) & 1u);
					count = ordinal == 0u ? first_compacted : LmLatentShardCompactWarp(cache.shard, selected + (uint64_t)row * topk, topk, row_bound,
						list_shared);
					compactions++;
				}
				else
					count = SparkKvShardLocalKeys(cache.shard, dense);
				window_begin = listed != 0u ? 0u : LmKvKeyWindowBegin(cache);
				limit = listed != 0u ? count : LmKvKeyWindowEnd(cache);
				limit = count < limit ? count : limit;
				tile_first = window_begin / TILE;
				tile_end = limit > window_begin ? (limit + TILE - 1u) / TILE : tile_first;
			}
			if (ordinal >= PLANS)
				LmMbarrierWait(&read[record], ((ordinal / PLANS) - 1u) & 1u);
			if (lane == 0u)
			{
				uint32_t *plan = plans + record * PLAN_WORDS;
				plan[0] = D41A_WINDOW_TILES + tile_end - tile_first;
				plan[1] = position >= D41A_WINDOW - 1u ? 0u : D41A_WINDOW - 1u - position;
				plan[2] = limit;
				plan[3] = valid;
				plan[4] = listed;
				plan[5] = sequence;
				plan[6] = position;
				plan[7] = wave_first;
				plan[8] = begin;
				plan[9] = recurrent_slot[step_lane];
				plan[10] = window_begin;
				plan[11] = tile_first;
				LmMbarrierArrive(&planned[record]);
			}
			__syncwarp();
		}
		return;
	}
	if (warp == D41A_ISSUER)
	{
		uint32_t zeroed[2] = { 0u, 0u };
		used = 0u;
		for (block = blockIdx.x, ordinal = 0u; block < live; block += gridDim.x, ordinal++)
		{
			const uint32_t record = ordinal % PLANS, row = block;
			const uint32_t *plan = plans + record * PLAN_WORDS;
			uint32_t tiles, window_from, limit, listed, sequence, position, wave_first, begin, slot, window_begin, tile_first;
			LmMbarrierWait(&planned[record], (ordinal / PLANS) & 1u);
			tiles = plan[0];
			window_from = plan[1];
			limit = plan[2];
			listed = plan[4];
			sequence = plan[5];
			position = plan[6];
			wave_first = plan[7];
			begin = plan[8];
			slot = plan[9];
			window_begin = plan[10];
			tile_first = plan[11];
			LmSharedReadsDone();
			if (lane == 0u)
				LmMbarrierArrive(&read[record]);
			for (uint32_t tile = 0u; tile < tiles; tile++, used++)
			{
				const uint32_t buffer = used & 1u, uses = used >> 1u;
				const uint8_t *source = 0;
				if (tile < D41A_WINDOW_TILES)
				{
					const uint32_t key = tile * TILE + lane;
					if (key >= window_from)
					{
						const uint32_t at = position + key - (D41A_WINDOW - 1u);
						source = (const uint8_t *)(at >= wave_first ? wave_kv + (uint64_t)sequence_row_indices[begin + at - wave_first] * DIM :
							ring + (uint64_t)slot * ring_slot_stride + (uint64_t)(at % ring_rows) * DIM);
					}
				}
				else
				{
					const uint32_t local = (tile_first + tile - D41A_WINDOW_TILES) * TILE + lane;
					if (local < limit && local >= window_begin)
						source = LmKvShardSlotRequired<Geometry>(cache, sequence,
							listed != 0u ? list_shared[local] : SparkKvShardLocalPosition(cache.shard, local), row, LM_KV_ACCESS_READ);
				}
				const uint32_t present = __ballot_sync(0xffffffffu, source != 0);
				uint32_t clear = ~present & ~zeroed[buffer];
				if (uses > 0u)
					LmMbarrierWait(&empty[buffer], (uses - 1u) & 1u);
				zeroed[buffer] = ~present;
				if (source != 0)
					LmTmaLoadBulk1d(keys + (buffer * TILE + lane) * PITCH, source, &full[buffer], ROW_BYTES);
				for (; clear != 0u; clear &= clear - 1u)
					for (uint32_t index = lane; index < CHUNKS; index += 32u)
						*(uint4 *)(keys + (buffer * TILE + (uint32_t)(__ffs(clear) - 1)) * PITCH + index * 16u) = make_uint4(0u, 0u, 0u, 0u);
				LmMbarrierInitFence();
				__syncwarp();
				if (lane == 0u)
					LmMbarrierArriveExpect(&full[buffer], (uint32_t)__popc(present) * ROW_BYTES);
			}
			if (listed != 0u)
			{
				LmSharedReadsDone();
				if (lane == 0u)
					LmMbarrierArrive(listed_free);
				__syncwarp();
			}
		}
		return;
	}
	uint32_t query_frag[HALF_STEPS][4];
	float accumulator[QTILES][COLUMN_FRAGS][4];
	const uint16_t *tile_query = query + query_tile * query_part_stride;
	auto fetch = [&](uint32_t row)
	{
		const uint32_t low = lane >> 2u, high = low + 8u;
		const uint32_t *low_query = (const uint32_t *)(tile_query + ((uint64_t)row * QUERIES + low) * DIM) + qk_half * HALF_STEPS * 8u + (lane & 3u);
		const uint32_t *high_query = (const uint32_t *)(tile_query + ((uint64_t)row * QUERIES + high) * DIM) + qk_half * HALF_STEPS * 8u + (lane & 3u);
#pragma unroll
		for (uint32_t step = 0u; step < HALF_STEPS; step++)
		{
			query_frag[step][0] = low_query[step * 8u];
			query_frag[step][1] = high_query[step * 8u];
			query_frag[step][2] = low_query[step * 8u + 4u];
			query_frag[step][3] = high_query[step * 8u + 4u];
		}
	};
	used = 0u;
	if (blockIdx.x < live)
		fetch(blockIdx.x);
	for (block = blockIdx.x, ordinal = 0u; block < live; block += gridDim.x, ordinal++)
	{
		const uint32_t record = ordinal % PLANS, row = block, following = block + gridDim.x;
		const uint32_t *plan = plans + record * PLAN_WORDS;
		uint32_t tiles, window_from, limit, valid, window_begin, tile_first;
		LmMbarrierWait(&planned[record], (ordinal / PLANS) & 1u);
		tiles = plan[0];
		window_from = plan[1];
		limit = plan[2];
		valid = plan[3];
		window_begin = plan[10];
		tile_first = plan[11];
		LmSharedReadsDone();
		if (lane == 0u)
			LmMbarrierArrive(&read[record]);
		if (valid == 0u && threadIdx.x == 0u)
			LmKvReportRequiredAccessFailure(cache.pages, !LmKvViewIsConfigured(cache.pages) ? LM_KV_ACCESS_ERROR_INVALID_VIEW :
				LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence_of_row[row], 0xffffffffu, 0xffffffffu);
		if (threadIdx.x < QT)
		{
			running_max[threadIdx.x] = -INFINITY;
			running_sum[threadIdx.x] = 0.0f;
		}
#pragma unroll
		for (uint32_t qt = 0u; qt < QTILES; qt++)
#pragma unroll
			for (uint32_t frag = 0u; frag < COLUMN_FRAGS; frag++)
				accumulator[qt][frag][0] = accumulator[qt][frag][1] = accumulator[qt][frag][2] = accumulator[qt][frag][3] = 0.0f;
		for (uint32_t tile = 0u; tile < tiles; tile++, used++)
		{
			const uint32_t buffer = used & 1u;
			const uint32_t tile_base = key_base + buffer * TILE * PITCH;
			LmMbarrierWait(&full[buffer], (used >> 1u) & 1u);
#pragma unroll
			for (uint32_t key_tile = 0u; key_tile < QTILES; key_tile++)
			{
				float score[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
				uint32_t b[4], first[2], second[2];
				const uint32_t key_first = (key_group * QTILES + key_tile) * 8u;
				const uint32_t b_row = key_first + within;
#pragma unroll
				for (uint32_t pair = 0u; pair < HALF_STEPS / 2u; pair++)
				{
					const uint32_t b_chunk = (qk_half * HALF_STEPS + 2u * pair + (matrix >> 1u)) * 2u + (matrix & 1u);
					LmPrefillLoadX4(tile_base + b_row * PITCH + b_chunk * 16u, b);
					first[0] = b[0]; first[1] = b[1];
					second[0] = b[2]; second[1] = b[3];
					LmMmaBf16(score, query_frag[2u * pair], first);
					LmMmaBf16(score, query_frag[2u * pair + 1u], second);
				}
#pragma unroll
				for (uint32_t index = 0u; index < 4u; index++)
					score_parts[(qk_half * QT + query_tile * QUERIES + (lane >> 2u) + (index >> 1u) * 8u) * D41A_SCORE_STRIDE + key_first + (lane & 3u) * 2u +
						(index & 1u)] = score[index];
			}
			if (tile + 1u == tiles && following < live)
				fetch(following);
			D41AttendBarrier();
#pragma unroll
			for (uint32_t qt = 0u; qt < QTILES; qt++)
			{
				const uint32_t mine = qt * QUERIES + own;
				float value[2], weight[2], peak, previous_max, next_max, scale, total;
#pragma unroll
				for (uint32_t index = 0u; index < 2u; index++)
				{
					const uint32_t key = part + index * 16u;
					uint32_t keep;
					value[index] = (score_parts[mine * D41A_SCORE_STRIDE + key] + score_parts[(QT + mine) * D41A_SCORE_STRIDE + key]) * qk_scale;
					if (tile < D41A_WINDOW_TILES)
						keep = tile * TILE + key >= window_from ? 1u : 0u;
					else
					{
						const uint32_t local = (tile_first + tile - D41A_WINDOW_TILES) * TILE + key;
						keep = local < limit && local >= window_begin ? 1u : 0u;
					}
					value[index] = keep != 0u ? value[index] : -INFINITY;
				}
				peak = fmaxf(value[0], value[1]);
#pragma unroll
				for (uint32_t shift = 8u; shift > 0u; shift >>= 1u)
					peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, shift));
				previous_max = running_max[mine];
				next_max = fmaxf(previous_max, peak);
				scale = previous_max == -INFINITY ? 0.0f : __expf(previous_max - next_max);
#pragma unroll
				for (uint32_t index = 0u; index < 2u; index++)
				{
					weight[index] = next_max == -INFINITY || value[index] == -INFINITY ? 0.0f : __expf(value[index] - next_max);
					weight_high[mine * D41A_WEIGHT_STRIDE + part + index * 16u] = LmFloatToBf16(weight[index]);
				}
				total = weight[0] + weight[1];
#pragma unroll
				for (uint32_t shift = 8u; shift > 0u; shift >>= 1u)
					total += __shfl_xor_sync(0xffffffffu, total, shift);
				__syncwarp();
				if (part == 0u)
				{
					running_sum[mine] = running_sum[mine] * scale + total;
					running_max[mine] = next_max;
					rescale[mine] = scale;
				}
			}
			D41AttendBarrier();
			{
				uint32_t a_high[QTILES][4], b[4], first[2], second[2];
#pragma unroll
				for (uint32_t qt = 0u; qt < QTILES; qt++)
				{
					const float low_scale = rescale[qt * QUERIES + (lane >> 2u)], high_scale = rescale[qt * QUERIES + (lane >> 2u) + 8u];
#pragma unroll
					for (uint32_t frag = 0u; frag < COLUMN_FRAGS; frag++)
					{
						accumulator[qt][frag][0] *= low_scale;
						accumulator[qt][frag][1] *= low_scale;
						accumulator[qt][frag][2] *= high_scale;
						accumulator[qt][frag][3] *= high_scale;
					}
				}
#pragma unroll
				for (uint32_t step = 0u; step < TILE / 16u; step++)
				{
					const uint32_t a_row = (matrix & 1u) * 8u + within, a_column = step * 16u + (matrix >> 1u) * 8u;
#pragma unroll
					for (uint32_t qt = 0u; qt < QTILES; qt++)
						LmPrefillLoadX4((uint32_t)__cvta_generic_to_shared(weight_high + (qt * QUERIES + a_row) * D41A_WEIGHT_STRIDE + a_column), a_high[qt]);
#pragma unroll
					for (uint32_t frag = 0u; frag < COLUMN_FRAGS; frag += 2u)
					{
						const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
						const uint32_t b_chunk = (warp * WARP_COLUMNS + frag * 8u) / 8u + (matrix >> 1u);
						LmPrefillLoadX4Trans(tile_base + b_row * PITCH + b_chunk * 16u, b);
						first[0] = b[0]; first[1] = b[1];
						second[0] = b[2]; second[1] = b[3];
#pragma unroll
						for (uint32_t qt = 0u; qt < QTILES; qt++)
						{
							LmMmaBf16(accumulator[qt][frag], a_high[qt], first);
							LmMmaBf16(accumulator[qt][frag + 1u], a_high[qt], second);
						}
					}
				}
			}
			LmSharedReadsDone();
			if (lane == 0u)
				LmMbarrierArrive(&empty[buffer]);
		}
		if (tiles == 0u && following < live)
			fetch(following);
		if (threadIdx.x < QT)
		{
			const float maximum = running_max[threadIdx.x], top = fmaxf(-1.0e30f, maximum);
			const float kept = maximum == -INFINITY ? 0.0f : __expf(maximum - top);
			rescale[threadIdx.x] = kept;
			denominator[threadIdx.x] = fmaf(running_sum[threadIdx.x], kept, 0.0f) + __expf(sink[threadIdx.x] - top);
		}
		D41AttendBarrier();
#pragma unroll
		for (uint32_t step = 0u; step < 2u * QTILES; step++)
		{
			const uint32_t qt = step / 2u, index = (step % 2u) * 2u;
			const uint32_t q = (lane >> 2u) + (index >> 1u) * 8u;
			const float kept = rescale[qt * QUERIES + q], divisor = denominator[qt * QUERIES + q];
			uint16_t value[COLUMN_FRAGS][2];
#pragma unroll
			for (uint32_t frag = 0u; frag < COLUMN_FRAGS; frag++)
			{
				value[frag][0] = LmFloatToBf16(fmaf(accumulator[qt][frag][index], kept, 0.0f) / divisor);
				value[frag][1] = LmFloatToBf16(fmaf(accumulator[qt][frag][index + 1u], kept, 0.0f) / divisor);
			}
			if (codes == 0)
			{
				uint16_t *target = output + qt * output_part_stride + ((uint64_t)row * QUERIES + q) * DIM + warp * WARP_COLUMNS + (lane & 3u) * 2u;
#pragma unroll
				for (uint32_t frag = 0u; frag < COLUMN_FRAGS; frag++)
					*(uint32_t *)(target + frag * 8u) = (uint32_t)value[frag][0] | ((uint32_t)value[frag][1] << 16u);
				continue;
			}
			if (warp * WARP_COLUMNS >= rope_offset)
			{
				const uint32_t position = positions[row_base + row];
#pragma unroll
				for (uint32_t frag = 0u; frag < COLUMN_FRAGS; frag++)
				{
					const uint32_t pair = (warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u - rope_offset) / 2u;
					const float real = LmBf16ToFloat(value[frag][0]), imaginary = LmBf16ToFloat(value[frag][1]);
					float sine, cosine;
					D41RopeTurn(position, 0, frequency[pair], 1u, &sine, &cosine);
					value[frag][0] = D41RopeReal(real, imaginary, sine, cosine);
					value[frag][1] = D41RopeImaginary(real, imaginary, sine, cosine);
				}
			}
			{
				const uint32_t group = q / group_heads, width = group_heads * DIM;
				const uint32_t base = (q % group_heads) * DIM + warp * WARP_COLUMNS;
				uint8_t *part_codes = codes + qt * codes_part_stride;
				uint8_t *group_codes = part_codes + ((uint64_t)group * rows + row) * width;
				uint8_t *group_scales = part_codes + (uint64_t)(QUERIES / group_heads) * rows * width + ((uint64_t)group * rows + row) * (width / 32u);
#pragma unroll
				for (uint32_t block = 0u; block < COLUMN_FRAGS / 4u; block++)
				{
					float amax = 0.0f;
#pragma unroll
					for (uint32_t frag = block * 4u; frag < block * 4u + 4u; frag++)
						amax = fmaxf(amax, fmaxf(fabsf(LmBf16ToFloat(value[frag][0])), fabsf(LmBf16ToFloat(value[frag][1]))));
					amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1u));
					amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2u));
					const float scale = SparkMathFp8QdqScale(fmaxf(0.0f, amax));
#pragma unroll
					for (uint32_t frag = block * 4u; frag < block * 4u + 4u; frag++)
					{
						const uint32_t column = base + frag * 8u + (lane & 3u) * 2u;
						const uint32_t low = LmFloatToE4m3(fminf(fmaxf(LmBf16ToFloat(value[frag][0]) / scale, -LM_E4M3_MAX), LM_E4M3_MAX));
						const uint32_t high = LmFloatToE4m3(fminf(fmaxf(LmBf16ToFloat(value[frag][1]) / scale, -LM_E4M3_MAX), LM_E4M3_MAX));
						*(uint16_t *)(group_codes + column) = (uint16_t)(low | (high << 8u));
					}
					if ((lane & 3u) == 0u)
						group_scales[(base + block * 32u) / 32u] = (uint8_t)(__float_as_uint(scale) >> 23u);
				}
			}
		}
	}
}

template<class Geometry, uint32_t DIM, uint32_t QTILES = 1u>
static inline SparkGpuError D41AttendLaunch(LmKvShardView cache, uint32_t compressed, const uint16_t *query, const uint32_t *sequence_of_row,
	const uint32_t *length, const uint32_t *bound, const uint32_t *selected, uint32_t topk, const uint16_t *wave_kv, const uint16_t *ring,
	uint64_t ring_slot_stride, uint32_t ring_rows, const uint32_t *positions, const uint32_t *row_lane, const uint32_t *sequence_row_begin,
	const uint32_t *sequence_row_indices, const uint32_t *recurrent_slot, const float *sink, float qk_scale, uint16_t *output, const float *frequency,
	uint32_t rope_offset, uint32_t group_heads, uint8_t *codes, uint32_t rows, uint32_t live, uint32_t row_base, SparkGpuStream stream,
	uint64_t query_part_stride = 0u, uint64_t output_part_stride = 0u, uint64_t codes_part_stride = 0u)
{
	const uint64_t shared = D41AttendSharedBytes<DIM, QTILES>(compressed != 0u && selected != 0 ? topk : 0u);
	static bool granted = false;
	if (live == 0u)
		return SPARK_GPU_SUCCESS;
	if (rows == 0u || live > rows || query == 0 || (output == 0) == (codes == 0) || (QTILES > 1u && query_part_stride == 0u) ||
		(QTILES > 1u && (output != 0 ? output_part_stride : codes_part_stride) == 0u) ||
		(codes != 0 && (frequency == 0 || group_heads == 0u || D41A_QUERIES % group_heads != 0u || rope_offset % (DIM / D41A_WARPS) != 0u ||
		rope_offset >= DIM || (DIM - rope_offset) / 2u > D41K_ROPE_PAIRS)) || wave_kv == 0 || ring == 0 || positions == 0 || row_lane == 0 || sequence_row_begin == 0 ||
		sequence_row_indices == 0 || recurrent_slot == 0 || sink == 0 || ring_rows == 0u || shared > LM_PREFILL_ATTN_SHARED_LIMIT ||
		(compressed != 0u && (length == 0 || bound == 0 || sequence_of_row == 0 || SparkKvShardValid(cache.shard, Geometry::kPageSlots) == 0u)))
		return SPARK_GPU_ERROR_INVALID_VALUE;
	if (!granted)
	{
		const SparkGpuError error = SparkGpuFuncSetAttribute((const void *)D41AttendKernel<Geometry, DIM, QTILES>, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES,
			(int)LM_PREFILL_ATTN_SHARED_LIMIT);
		if (error != SPARK_GPU_SUCCESS)
			return error;
		granted = true;
	}
	const uint32_t resident = LmLatentShardPrefillMultiprocessors();
	const uint32_t grid = live < resident ? live : resident;
	LM_LAUNCH((D41AttendKernel<Geometry, DIM, QTILES>), grid, D41A_THREADS, shared, stream, cache, compressed, query, query_part_stride, sequence_of_row, length, bound,
		selected, topk, wave_kv, ring, ring_slot_stride, ring_rows, positions, row_lane, sequence_row_begin, sequence_row_indices, recurrent_slot, sink,
		qk_scale, output, output_part_stride, frequency, rope_offset, group_heads, codes, codes_part_stride, rows, live, row_base);
	return SparkGpuPeekAtLastError();
}
