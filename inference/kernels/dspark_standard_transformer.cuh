#ifndef SPARKPIPE_DSPARK_STANDARD_TRANSFORMER_CUH
#define SPARKPIPE_DSPARK_STANDARD_TRANSFORMER_CUH

#include "inference/kernels/dspark_draft_stage_common.cuh"

enum DsparkAttnKind
{
	DSPARK_ATTN_RING_SPLIT_MHA64 = 0,
	DSPARK_ATTN_RING_SPLIT_GQA128 = 1,
	DSPARK_ATTN_RING_SINGLE_PASS = 2,
	DSPARK_ATTN_PAGED_GQA128 = 3,
};

struct DsparkStandardLayer
{
	const uint16_t *input_norm;
	const uint16_t *post_norm;
	const uint16_t *q;
	const uint16_t *k;
	const uint16_t *v;
	const uint16_t *o;
	const uint16_t *q_norm;
	const uint16_t *k_norm;
	const uint16_t *gate;
	const uint16_t *up;
	const uint16_t *down;
};

template<uint32_t BLOCK, uint32_t WINDOW>
struct DsparkRingDraft
{
	uint64_t seed;
	uint16_t *cache;
	uint32_t positions[BLOCK];
	uint32_t anchor;
	uint32_t mask;
	uint32_t position;
	uint32_t top_k;
	float inverse_temperature;
	float nucleus;
	float scale;
	uint32_t start;
	uint32_t block_at;
	uint32_t splits;
	uint32_t drafts;
	uint32_t masked;
	uint8_t valid[WINDOW];
};

template<uint32_t BLOCK, uint32_t RUNS_MAX>
struct DsparkPagedDraft
{
	uint64_t seed;
	uint32_t positions[BLOCK];
	uint32_t anchor;
	uint32_t mask;
	uint32_t position;
	uint32_t top_k;
	float inverse_temperature;
	float nucleus;
	float scale;
	uint32_t held;
	uint32_t splits;
	uint32_t split_rows;
	uint32_t drafts;
	uint32_t runs;
	uint32_t run_first[RUNS_MAX];
	uint32_t run_position[RUNS_MAX];
};

template<uint32_t HEADS, uint32_t BLOCK, uint32_t COLUMNS, uint32_t HEAD_DIM, uint32_t SPLIT_KEYS, uint32_t SPLITS, uint32_t ATTN_TILE,
	typename DraftT>
__global__ __launch_bounds__(BLOCK * DSPARK_WARP, 2)
void DsparkRingSplitMha64Kernel(const uint16_t *__restrict__ q, uint64_t keys_offset, uint64_t values_offset, const DraftT *__restrict__ draft,
	float scale, float *__restrict__ partials)
{
	LmDependentWait();
	constexpr uint32_t HALF = HEAD_DIM / 2u;
	constexpr uint32_t STRIDE = HALF + 4u;
	__shared__ __align__(16) uint32_t s_k[2][ATTN_TILE][STRIDE];
	__shared__ __align__(16) uint32_t s_v[2][ATTN_TILE][STRIDE];
	__shared__ float s_scores[BLOCK][ATTN_TILE];
	__shared__ uint8_t s_vis[2][ATTN_TILE];
	const uint32_t head = blockIdx.x, split = blockIdx.y;
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	const uint32_t first = draft->start + split * SPLIT_KEYS, end = draft->block_at;
	const uint32_t stop = first + SPLIT_KEYS < end ? first + SPLIT_KEYS : end;
	const uint32_t tiles = first < stop ? (stop - first + ATTN_TILE - 1u) / ATTN_TILE : 0u;
	const uint32_t masked = draft->masked;
	const uint16_t *head_keys = draft->cache + keys_offset + (uint64_t)head * COLUMNS * HEAD_DIM;
	const uint16_t *head_values = draft->cache + values_offset + (uint64_t)head * COLUMNS * HEAD_DIM;
	const uint32_t q_pair = __ldg((const uint32_t *)(q + ((uint64_t)warp * HEADS + head) * HEAD_DIM + 2u * lane));
	const float q0 = LmBf16ToFloat((uint16_t)(q_pair & 0xffffu));
	const float q1 = LmBf16ToFloat((uint16_t)(q_pair >> 16u));
	float running_max = -INFINITY, running_sum = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
	float *out = partials + (((uint64_t)head * SPLITS + split) * BLOCK + warp) * (HEAD_DIM + 2u);
	const uint32_t load_key = threadIdx.x >> 3u, load_chunk = threadIdx.x & 7u;
	auto stage = [&](uint32_t tile_idx, uint32_t buf)
	{
		const uint32_t tile_first = first + tile_idx * ATTN_TILE;
		const uint32_t col = tile_first + load_key;
		const uint32_t bytes = col < stop ? 16u : 0u;
		const uint32_t safe_col = col < stop ? col : 0u;
		SparkDrafterStage16((uint32_t)__cvta_generic_to_shared(&s_k[buf][load_key][load_chunk * 4u]),
			head_keys + (uint64_t)safe_col * HEAD_DIM + load_chunk * 8u, bytes);
		SparkDrafterStage16((uint32_t)__cvta_generic_to_shared(&s_v[buf][load_key][load_chunk * 4u]),
			head_values + (uint64_t)safe_col * HEAD_DIM + load_chunk * 8u, bytes);
		SparkDrafterCommitGroup();
		if ( threadIdx.x < ATTN_TILE )
			s_vis[buf][threadIdx.x] = tile_first + threadIdx.x < stop && (masked == 0u || draft->valid[tile_first + threadIdx.x] != 0u) ? 1u : 0u;
	};
	if ( tiles != 0u )
		stage(0u, 0u);
	for ( uint32_t tile = 0u; tile < tiles; tile++ )
	{
		const uint32_t buf = tile & 1u;
		if ( tile + 1u < tiles )
		{
			stage(tile + 1u, buf ^ 1u);
			SparkDrafterWaitGroup<1u>();
		}
		else
			SparkDrafterWaitGroup<0u>();
		__syncthreads();
		for ( uint32_t t = 0u; t < ATTN_TILE; t++ )
		{
			const uint32_t kp = s_k[buf][t][lane];
			const float k0 = LmBf16ToFloat((uint16_t)(kp & 0xffffu));
			const float k1 = LmBf16ToFloat((uint16_t)(kp >> 16u));
			const float dot = DsparkWarpSum(fmaf(q0, k0, q1 * k1));
			if ( lane == 0u )
				s_scores[warp][t] = s_vis[buf][t] != 0u ? dot * scale : -INFINITY;
		}
		__syncwarp();
		{
			const float sc = s_scores[warp][lane];
			const float tile_max = DsparkWarpMax(sc);
			const float next_max = fmaxf(running_max, tile_max);
			const float rescale = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
			const float weight = sc == -INFINITY ? 0.0f : expf(sc - next_max);
			const float tile_sum = DsparkWarpSum(weight);
			running_sum = running_sum * rescale + tile_sum;
			running_max = next_max;
			acc0 *= rescale;
			acc1 *= rescale;
			s_scores[warp][lane] = weight;
		}
		__syncwarp();
		for ( uint32_t t = 0u; t < ATTN_TILE; t++ )
		{
			const float wt = s_scores[warp][t];
			const uint32_t vp = s_v[buf][t][lane];
			acc0 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp & 0xffffu)), acc0);
			acc1 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp >> 16u)), acc1);
		}
		__syncthreads();
	}
	out[2u * lane] = acc0;
	out[2u * lane + 1u] = acc1;
	if ( lane == 0u )
	{
		out[HEAD_DIM] = running_max;
		out[HEAD_DIM + 1u] = running_sum;
	}
	LmDependentRelease();
}

template<uint32_t PAGE_POSITIONS, uint32_t HEAD_DIM>
static __device__ __forceinline__ const uint16_t *DsparkPagedKeyRow(uint16_t *const *__restrict__ pages, uint64_t layer_head_offset,
	const uint32_t *__restrict__ run_first, const uint32_t *__restrict__ run_position, uint32_t runs, uint32_t index, uint32_t end)
{
	uint32_t low = 0u, high = runs, position;
	if ( index >= end )
		return 0;
	while ( high - low > 1u )
	{
		const uint32_t middle = (low + high) >> 1u;
		if ( run_first[middle] <= index )
			low = middle;
		else
			high = middle;
	}
	position = run_position[low] + (index - run_first[low]);
	return pages[position / PAGE_POSITIONS] + layer_head_offset + (uint64_t)(position % PAGE_POSITIONS) * HEAD_DIM;
}

template<uint32_t HEADS, uint32_t KV_HEADS, uint32_t ATTN_Q_GROUP, uint32_t BLOCK, uint32_t COLUMNS, uint32_t HEAD_DIM, uint64_t LAYER_KV,
	uint32_t SPLIT_KEYS, uint32_t SPLITS, uint32_t ATTN_TILE, uint32_t PAGE_POSITIONS, uint64_t PAGE_LAYER, uint64_t PAGE_HALF_LAYER, bool PAGED,
	typename DraftT>
__global__ __launch_bounds__(DSPARK_THREADS, 2)
void DsparkSplitGqa128Kernel(const uint16_t *__restrict__ q, uint16_t *const *__restrict__ pages, uint32_t layer, const DraftT *__restrict__ draft,
	float scale, float *__restrict__ partials)
{
	constexpr uint32_t HALF = HEAD_DIM / 2u;
	constexpr uint32_t GROUP = HEADS / KV_HEADS;
	constexpr uint32_t ATTN_Q_GROUPS = GROUP / ATTN_Q_GROUP;
	__shared__ __align__(16) uint32_t s_k[2][ATTN_TILE][HALF];
	__shared__ __align__(16) uint32_t s_v[2][ATTN_TILE][HALF];
	__shared__ float s_scores[BLOCK][ATTN_TILE];
	__shared__ uint8_t s_vis[ATTN_TILE];
	LmDependentWait();
	const uint32_t kv_head = blockIdx.x / ATTN_Q_GROUPS, q_group = blockIdx.x % ATTN_Q_GROUPS, split = blockIdx.y;
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	const uint32_t q_head_base = kv_head * GROUP + q_group * ATTN_Q_GROUP;
	const uint32_t load_key = threadIdx.x >> 3u, load_chunk = threadIdx.x & 7u;
	uint32_t first, stop, masked = 0u;
	uint64_t layer_head_offset = 0u;
	const uint16_t *head_keys = 0, *head_values = 0;
	if constexpr ( PAGED )
	{
		const uint32_t held = draft->held, split_rows = draft->split_rows;
		layer_head_offset = (uint64_t)layer * PAGE_LAYER + (uint64_t)kv_head * PAGE_POSITIONS * HEAD_DIM;
		first = split * split_rows;
		stop = first + split_rows < held ? first + split_rows : held;
	}
	else
	{
		first = draft->start + split * SPLIT_KEYS;
		stop = first + SPLIT_KEYS < draft->block_at ? first + SPLIT_KEYS : draft->block_at;
		masked = draft->masked;
		head_keys = draft->cache + (uint64_t)(2u * layer) * LAYER_KV + (uint64_t)kv_head * COLUMNS * HEAD_DIM;
		head_values = head_keys + LAYER_KV;
	}
	const uint32_t tiles = first < stop ? (stop - first + ATTN_TILE - 1u) / ATTN_TILE : 0u;
	float q0[ATTN_Q_GROUP], q1[ATTN_Q_GROUP], q2[ATTN_Q_GROUP], q3[ATTN_Q_GROUP];
	float running_max[ATTN_Q_GROUP], running_sum[ATTN_Q_GROUP];
	float acc0[ATTN_Q_GROUP], acc1[ATTN_Q_GROUP], acc2[ATTN_Q_GROUP], acc3[ATTN_Q_GROUP];
	if ( warp < BLOCK )
	{
#pragma unroll
		for ( uint32_t g = 0u; g < ATTN_Q_GROUP; g++ )
		{
			const uint16_t *q_row = q + ((uint64_t)warp * HEADS + q_head_base + g) * HEAD_DIM;
			if constexpr ( PAGED )
			{
				const uint2 q_quad = *(const uint2 *)(q_row + 4u * lane);
				q0[g] = LmBf16ToFloat((uint16_t)(q_quad.x & 0xffffu));
				q1[g] = LmBf16ToFloat((uint16_t)(q_quad.x >> 16u));
				q2[g] = LmBf16ToFloat((uint16_t)(q_quad.y & 0xffffu));
				q3[g] = LmBf16ToFloat((uint16_t)(q_quad.y >> 16u));
			}
			else
			{
				const uint32_t qp0 = *(const uint32_t *)(q_row + 2u * lane);
				const uint32_t qp1 = *(const uint32_t *)(q_row + HALF + 2u * lane);
				q0[g] = LmBf16ToFloat((uint16_t)(qp0 & 0xffffu));
				q1[g] = LmBf16ToFloat((uint16_t)(qp0 >> 16u));
				q2[g] = LmBf16ToFloat((uint16_t)(qp1 & 0xffffu));
				q3[g] = LmBf16ToFloat((uint16_t)(qp1 >> 16u));
			}
			running_max[g] = -INFINITY;
			running_sum[g] = 0.0f;
			acc0[g] = acc1[g] = acc2[g] = acc3[g] = 0.0f;
		}
	}
	auto stage_tile = [&](uint32_t buf, uint32_t tile_first)
	{
		const uint16_t *src_k = 0, *src_v = 0;
		if constexpr ( PAGED )
		{
			src_k = DsparkPagedKeyRow<PAGE_POSITIONS, HEAD_DIM>(pages, layer_head_offset, draft->run_first, draft->run_position, draft->runs,
				tile_first + load_key, stop);
			if ( src_k != 0 )
				src_v = src_k + PAGE_HALF_LAYER;
		}
		else
		{
			const uint32_t col = tile_first + load_key;
			if ( col < stop )
			{
				src_k = head_keys + (uint64_t)col * HEAD_DIM;
				src_v = head_values + (uint64_t)col * HEAD_DIM;
			}
		}
		if ( src_k != 0 )
		{
			SparkDrafterStage16((uint32_t)__cvta_generic_to_shared(&s_k[buf][load_key][load_chunk * 8u]), src_k + load_chunk * 16u, 16u);
			SparkDrafterStage16((uint32_t)__cvta_generic_to_shared(&s_k[buf][load_key][load_chunk * 8u + 4u]), src_k + load_chunk * 16u + 8u, 16u);
			SparkDrafterStage16((uint32_t)__cvta_generic_to_shared(&s_v[buf][load_key][load_chunk * 8u]), src_v + load_chunk * 16u, 16u);
			SparkDrafterStage16((uint32_t)__cvta_generic_to_shared(&s_v[buf][load_key][load_chunk * 8u + 4u]), src_v + load_chunk * 16u + 8u, 16u);
		}
		SparkDrafterCommitGroup();
	};
	if ( tiles > 0u )
	{
		stage_tile(0u, first);
		for ( uint32_t tile = 0u; tile < tiles; tile++ )
		{
			const uint32_t cur = tile & 1u, nxt = cur ^ 1u;
			const uint32_t tile_first = first + tile * ATTN_TILE;
			const uint32_t tile_keys = stop - tile_first < ATTN_TILE ? stop - tile_first : ATTN_TILE;
			if ( tile + 1u < tiles )
			{
				stage_tile(nxt, tile_first + ATTN_TILE);
				SparkDrafterWaitGroup<1>();
			}
			else
				SparkDrafterWaitGroup<0>();
			if constexpr ( !PAGED )
				if ( threadIdx.x < ATTN_TILE )
					s_vis[threadIdx.x] = threadIdx.x < tile_keys && (masked == 0u || draft->valid[tile_first + threadIdx.x] != 0u) ? 1u : 0u;
			__syncthreads();
			if ( warp < BLOCK )
			{
#pragma unroll
				for ( uint32_t g = 0u; g < ATTN_Q_GROUP; g++ )
				{
					for ( uint32_t t = 0u; t < tile_keys; t++ )
					{
						float dot;
						if constexpr ( PAGED )
						{
							const uint2 kp = *(const uint2 *)&s_k[cur][t][2u * lane];
							dot = fmaf(q0[g], LmBf16ToFloat((uint16_t)(kp.x & 0xffffu)),
								fmaf(q1[g], LmBf16ToFloat((uint16_t)(kp.x >> 16u)),
									fmaf(q2[g], LmBf16ToFloat((uint16_t)(kp.y & 0xffffu)), q3[g] * LmBf16ToFloat((uint16_t)(kp.y >> 16u)))));
						}
						else
						{
							const uint32_t kp0 = s_k[cur][t][lane];
							const uint32_t kp1 = s_k[cur][t][DSPARK_WARP + lane];
							const float d0 = fmaf(q0[g], LmBf16ToFloat((uint16_t)(kp0 & 0xffffu)), q1[g] * LmBf16ToFloat((uint16_t)(kp0 >> 16u)));
							const float d1 = fmaf(q2[g], LmBf16ToFloat((uint16_t)(kp1 & 0xffffu)), q3[g] * LmBf16ToFloat((uint16_t)(kp1 >> 16u)));
							dot = d0 + d1;
						}
						dot = DsparkWarpSum(dot);
						if ( lane == 0u )
						{
							if constexpr ( PAGED )
								s_scores[warp][t] = dot * scale;
							else
								s_scores[warp][t] = s_vis[t] != 0u ? dot * scale : -INFINITY;
						}
					}
					__syncwarp();
					{
						const float sc = lane < tile_keys ? s_scores[warp][lane] : -INFINITY;
						const float tile_max = DsparkWarpMax(sc);
						const float next_max = fmaxf(running_max[g], tile_max);
						const float rescale = running_max[g] == -INFINITY ? 0.0f : expf(running_max[g] - next_max);
						const float weight = sc == -INFINITY ? 0.0f : expf(sc - next_max);
						const float tile_sum = DsparkWarpSum(weight);
						running_sum[g] = running_sum[g] * rescale + tile_sum;
						running_max[g] = next_max;
						acc0[g] *= rescale;
						acc1[g] *= rescale;
						acc2[g] *= rescale;
						acc3[g] *= rescale;
						s_scores[warp][lane] = weight;
					}
					__syncwarp();
					for ( uint32_t t = 0u; t < tile_keys; t++ )
					{
						const float wt = s_scores[warp][t];
						if constexpr ( PAGED )
						{
							const uint2 vp = *(const uint2 *)&s_v[cur][t][2u * lane];
							acc0[g] = fmaf(wt, LmBf16ToFloat((uint16_t)(vp.x & 0xffffu)), acc0[g]);
							acc1[g] = fmaf(wt, LmBf16ToFloat((uint16_t)(vp.x >> 16u)), acc1[g]);
							acc2[g] = fmaf(wt, LmBf16ToFloat((uint16_t)(vp.y & 0xffffu)), acc2[g]);
							acc3[g] = fmaf(wt, LmBf16ToFloat((uint16_t)(vp.y >> 16u)), acc3[g]);
						}
						else
						{
							const uint32_t vp0 = s_v[cur][t][lane];
							const uint32_t vp1 = s_v[cur][t][DSPARK_WARP + lane];
							acc0[g] = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 & 0xffffu)), acc0[g]);
							acc1[g] = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 >> 16u)), acc1[g]);
							acc2[g] = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 & 0xffffu)), acc2[g]);
							acc3[g] = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 >> 16u)), acc3[g]);
						}
					}
				}
			}
			__syncthreads();
		}
	}
	if ( warp < BLOCK )
	{
#pragma unroll
		for ( uint32_t g = 0u; g < ATTN_Q_GROUP; g++ )
		{
			float *out = partials + (((uint64_t)(q_head_base + g) * SPLITS + split) * BLOCK + warp) * (HEAD_DIM + 2u);
			if constexpr ( PAGED )
			{
				out[4u * lane + 0u] = acc0[g];
				out[4u * lane + 1u] = acc1[g];
				out[4u * lane + 2u] = acc2[g];
				out[4u * lane + 3u] = acc3[g];
			}
			else
			{
				out[2u * lane] = acc0[g];
				out[2u * lane + 1u] = acc1[g];
				out[HALF + 2u * lane] = acc2[g];
				out[HALF + 2u * lane + 1u] = acc3[g];
			}
			if ( lane == 0u )
			{
				out[HEAD_DIM] = running_max[g];
				out[HEAD_DIM + 1u] = running_sum[g];
			}
		}
	}
	LmDependentRelease();
}

template<uint32_t HEADS, uint32_t HEAD_DIM, uint32_t BLOCK, uint32_t COLUMNS, uint64_t LAYER_KV, uint32_t ATTN_TILE, typename DraftT>
__global__ __launch_bounds__(BLOCK * DSPARK_WARP, 2)
void DsparkRingSinglePassAttentionKernel(const uint16_t *__restrict__ q, const DraftT *__restrict__ draft, uint32_t layer, float scale,
	uint16_t *__restrict__ output)
{
	constexpr uint32_t HALF = HEAD_DIM / 2u;
	constexpr uint32_t Q_WIDTH = HEADS * HEAD_DIM;
	constexpr uint32_t ATTN_THREADS = BLOCK * DSPARK_WARP;
	constexpr uint32_t ATTN_VEC4S = HEAD_DIM / 8u;
	__shared__ __align__(16) uint32_t s_k[2][ATTN_TILE][HALF];
	__shared__ __align__(16) uint32_t s_v[2][ATTN_TILE][HALF];
	__shared__ float s_scores[BLOCK][ATTN_TILE];
	__shared__ uint8_t s_vis[ATTN_TILE];
	LmDependentWait();
	const uint32_t head = blockIdx.x;
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	const uint32_t start = draft->start, block_at = draft->block_at, masked = draft->masked;
	const uint32_t tiles = start < block_at ? (block_at - start + ATTN_TILE - 1u) / ATTN_TILE : 0u;
	const uint16_t *head_keys = draft->cache + (uint64_t)(2u * layer) * LAYER_KV + (uint64_t)head * COLUMNS * HEAD_DIM;
	const uint16_t *head_values = head_keys + LAYER_KV;
	const uint16_t *q_row = q + ((uint64_t)warp * HEADS + head) * HEAD_DIM;
	const uint32_t qp0 = *(const uint32_t *)(q_row + 2u * lane);
	const float q0 = LmBf16ToFloat((uint16_t)(qp0 & 0xffffu));
	const float q1 = LmBf16ToFloat((uint16_t)(qp0 >> 16u));
	float q2 = 0.0f, q3 = 0.0f;
	float running_max = -INFINITY, running_sum = 0.0f, acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
	uint16_t *out_row = output + (uint64_t)warp * Q_WIDTH + (uint64_t)head * HEAD_DIM;
	const uint32_t blk_stage = tiles & 1u;
	if ( lane < HALF - DSPARK_WARP )
	{
		const uint32_t qp1 = *(const uint32_t *)(q_row + 2u * (DSPARK_WARP + lane));
		q2 = LmBf16ToFloat((uint16_t)(qp1 & 0xffffu));
		q3 = LmBf16ToFloat((uint16_t)(qp1 >> 16u));
	}
	auto stage_keys = [&](uint32_t buf, uint32_t base_col, uint32_t nkeys)
	{
		for ( uint32_t idx = threadIdx.x; idx < nkeys * ATTN_VEC4S; idx += ATTN_THREADS )
		{
			const uint32_t load_key = idx / ATTN_VEC4S, load_chunk = idx % ATTN_VEC4S;
			const uint64_t off = (uint64_t)(base_col + load_key) * HEAD_DIM + load_chunk * 8u;
			SparkDrafterStage16((uint32_t)__cvta_generic_to_shared(&s_k[buf][load_key][load_chunk * 4u]), head_keys + off, 16u);
			SparkDrafterStage16((uint32_t)__cvta_generic_to_shared(&s_v[buf][load_key][load_chunk * 4u]), head_values + off, 16u);
		}
		SparkDrafterCommitGroup();
	};
	if ( tiles > 0u )
	{
		stage_keys(0u, start, block_at - start < ATTN_TILE ? block_at - start : ATTN_TILE);
		for ( uint32_t tile = 0u; tile < tiles; tile++ )
		{
			const uint32_t cur = tile & 1u, nxt = cur ^ 1u;
			const uint32_t tile_first = start + tile * ATTN_TILE;
			const uint32_t tile_keys = block_at - tile_first < ATTN_TILE ? block_at - tile_first : ATTN_TILE;
			if ( tile + 1u < tiles )
			{
				const uint32_t next_first = tile_first + ATTN_TILE;
				stage_keys(nxt, next_first, block_at - next_first < ATTN_TILE ? block_at - next_first : ATTN_TILE);
			}
			else
				stage_keys(nxt, block_at, BLOCK);
			SparkDrafterWaitGroup<1>();
			if ( threadIdx.x < ATTN_TILE )
				s_vis[threadIdx.x] = threadIdx.x < tile_keys && (masked == 0u || draft->valid[tile_first + threadIdx.x] != 0u) ? 1u : 0u;
			__syncthreads();
			for ( uint32_t t = 0u; t < tile_keys; t++ )
			{
				const uint32_t kp0 = s_k[cur][t][lane];
				float dot = fmaf(q0, LmBf16ToFloat((uint16_t)(kp0 & 0xffffu)), q1 * LmBf16ToFloat((uint16_t)(kp0 >> 16u)));
				if ( lane < HALF - DSPARK_WARP )
				{
					const uint32_t kp1 = s_k[cur][t][DSPARK_WARP + lane];
					dot = fmaf(q2, LmBf16ToFloat((uint16_t)(kp1 & 0xffffu)), fmaf(q3, LmBf16ToFloat((uint16_t)(kp1 >> 16u)), dot));
				}
				dot = DsparkWarpSum(dot);
				if ( lane == 0u )
					s_scores[warp][t] = s_vis[t] != 0u ? dot * scale : -INFINITY;
			}
			__syncwarp();
			{
				const float sc = lane < tile_keys ? s_scores[warp][lane] : -INFINITY;
				const float tile_max = DsparkWarpMax(sc);
				const float next_max = fmaxf(running_max, tile_max);
				const float rescale = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
				const float weight = sc == -INFINITY ? 0.0f : expf(sc - next_max);
				const float tile_sum = DsparkWarpSum(weight);
				running_sum = running_sum * rescale + tile_sum;
				running_max = next_max;
				acc0 *= rescale;
				acc1 *= rescale;
				acc2 *= rescale;
				acc3 *= rescale;
				s_scores[warp][lane] = weight;
			}
			__syncwarp();
			for ( uint32_t t = 0u; t < tile_keys; t++ )
			{
				const float wt = s_scores[warp][t];
				const uint32_t vp0 = s_v[cur][t][lane];
				acc0 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 & 0xffffu)), acc0);
				acc1 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 >> 16u)), acc1);
				if ( lane < HALF - DSPARK_WARP )
				{
					const uint32_t vp1 = s_v[cur][t][DSPARK_WARP + lane];
					acc2 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 & 0xffffu)), acc2);
					acc3 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 >> 16u)), acc3);
				}
			}
			__syncthreads();
		}
		SparkDrafterWaitGroup<0>();
		__syncthreads();
	}
	else
	{
		stage_keys(0u, block_at, BLOCK);
		SparkDrafterWaitGroup<0>();
		__syncthreads();
	}
	for ( uint32_t t = 0u; t <= warp; t++ )
	{
		const uint32_t kp0 = s_k[blk_stage][t][lane];
		float dot = fmaf(q0, LmBf16ToFloat((uint16_t)(kp0 & 0xffffu)), q1 * LmBf16ToFloat((uint16_t)(kp0 >> 16u)));
		if ( lane < HALF - DSPARK_WARP )
		{
			const uint32_t kp1 = s_k[blk_stage][t][DSPARK_WARP + lane];
			dot = fmaf(q2, LmBf16ToFloat((uint16_t)(kp1 & 0xffffu)), fmaf(q3, LmBf16ToFloat((uint16_t)(kp1 >> 16u)), dot));
		}
		dot = DsparkWarpSum(dot);
		if ( lane == 0u )
			s_scores[warp][t] = dot * scale;
	}
	__syncwarp();
	{
		const float sc = lane <= warp ? s_scores[warp][lane] : -INFINITY;
		const float tile_max = DsparkWarpMax(sc);
		const float next_max = fmaxf(running_max, tile_max);
		const float rescale = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
		const float weight = sc == -INFINITY ? 0.0f : expf(sc - next_max);
		const float tile_sum = DsparkWarpSum(weight);
		running_sum = running_sum * rescale + tile_sum;
		running_max = next_max;
		acc0 *= rescale;
		acc1 *= rescale;
		acc2 *= rescale;
		acc3 *= rescale;
		s_scores[warp][lane] = weight;
	}
	__syncwarp();
	for ( uint32_t t = 0u; t <= warp; t++ )
	{
		const float wt = s_scores[warp][t];
		const uint32_t vp0 = s_v[blk_stage][t][lane];
		acc0 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 & 0xffffu)), acc0);
		acc1 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp0 >> 16u)), acc1);
		if ( lane < HALF - DSPARK_WARP )
		{
			const uint32_t vp1 = s_v[blk_stage][t][DSPARK_WARP + lane];
			acc2 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 & 0xffffu)), acc2);
			acc3 = fmaf(wt, LmBf16ToFloat((uint16_t)(vp1 >> 16u)), acc3);
		}
	}
	out_row[2u * lane] = LmFloatToBf16(acc0 / running_sum);
	out_row[2u * lane + 1u] = LmFloatToBf16(acc1 / running_sum);
	if ( lane < HALF - DSPARK_WARP )
	{
		out_row[2u * (DSPARK_WARP + lane)] = LmFloatToBf16(acc2 / running_sum);
		out_row[2u * (DSPARK_WARP + lane) + 1u] = LmFloatToBf16(acc3 / running_sum);
	}
	LmDependentRelease();
}

template<uint32_t HEAD_DIM, uint32_t PAGE_POSITIONS, uint64_t PAGE_LAYER, bool TO_PAGES>
__global__ __launch_bounds__(HEAD_DIM, 4)
void DsparkPagedHeadNormRopeKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, const float *__restrict__ inverse,
	const uint32_t *__restrict__ positions, uint32_t heads, float epsilon, uint16_t *__restrict__ output, uint16_t *const *__restrict__ pages,
	uint32_t layer)
{
	constexpr uint32_t HALF = HEAD_DIM / 2u;
	__shared__ float reduction[HEAD_DIM / DSPARK_WARP];
	__shared__ float normed[HEAD_DIM];
	const uint32_t row = blockIdx.x / heads, head = blockIdx.x % heads, lane = threadIdx.x;
	const uint32_t position = positions[row];
	const float value = LmBf16ToFloat(input[((uint64_t)row * heads + head) * HEAD_DIM + lane]);
	float scale, cosine, sine, rotated;
	scale = rsqrtf(LmBlockSum<HEAD_DIM>(value * value, reduction) / (float)HEAD_DIM + epsilon);
	normed[lane] = LmBf16ToFloat(LmFloatToBf16(value * scale * LmBf16ToFloat(weight[lane])));
	__syncthreads();
	DsparkRotationUnit(position, inverse[lane % HALF], &cosine, &sine);
	rotated = lane < HALF ? normed[lane] * cosine - normed[lane + HALF] * sine : normed[lane] * cosine + normed[lane - HALF] * sine;
	if constexpr ( TO_PAGES )
	{
		uint16_t *page = pages[position / PAGE_POSITIONS];
		page[(uint64_t)layer * PAGE_LAYER + ((uint64_t)head * PAGE_POSITIONS + (position % PAGE_POSITIONS)) * HEAD_DIM + lane] =
			LmFloatToBf16(rotated);
	}
	else
		output[((uint64_t)row * heads + head) * HEAD_DIM + lane] = LmFloatToBf16(rotated);
	LmDependentRelease();
}

template<uint32_t KV_WIDTH, uint32_t HEAD_DIM, uint32_t PAGE_POSITIONS, uint64_t PAGE_LAYER, uint64_t PAGE_HALF_LAYER>
__global__ __launch_bounds__(DSPARK_THREADS, 2)
void DsparkPagedIngestStoreValuesKernel(const uint16_t *__restrict__ values, const uint32_t *__restrict__ positions,
	uint16_t *const *__restrict__ pages, uint32_t layer)
{
	const uint32_t row = blockIdx.x, position = positions[row];
	uint16_t *dst = pages[position / PAGE_POSITIONS] + (uint64_t)layer * PAGE_LAYER + PAGE_HALF_LAYER;
	const uint32_t slot = position % PAGE_POSITIONS;
	for ( uint32_t vec = threadIdx.x; vec < KV_WIDTH / 8u; vec += DSPARK_THREADS )
	{
		const uint32_t index = vec * 8u;
		const uint32_t head = index / HEAD_DIM;
		const uint32_t col = index % HEAD_DIM;
		*(uint4 *)(dst + ((uint64_t)head * PAGE_POSITIONS + slot) * HEAD_DIM + col) = *(const uint4 *)(values + (uint64_t)row * KV_WIDTH + index);
	}
	LmDependentRelease();
}

template<typename Cfg>
struct DsparkStandardEngine
{
	static constexpr uint32_t TAP_WIDTH = Cfg::TAPS * Cfg::HIDDEN;
	static constexpr uint32_t HALF = Cfg::HEAD_DIM / 2u;
	static constexpr uint32_t Q_WIDTH = Cfg::HEADS * Cfg::HEAD_DIM;
	static constexpr uint32_t KV_WIDTH = Cfg::KV_HEADS * Cfg::HEAD_DIM;
	static constexpr uint32_t GROUP = Cfg::HEADS / Cfg::KV_HEADS;
	static constexpr uint32_t ATTN_Q_GROUPS = GROUP / Cfg::ATTN_Q_GROUP;
	static constexpr uint32_t CONFIDENCE = Cfg::HIDDEN + Cfg::MARKOV;
	static constexpr uint32_t COLUMNS = Cfg::WINDOW + Cfg::BLOCK;
	static constexpr uint64_t LAYER_KV = (uint64_t)Cfg::KV_HEADS * COLUMNS * Cfg::HEAD_DIM;
	static constexpr uint64_t SEQ_ELEMENTS = 2u * Cfg::LAYERS * LAYER_KV;
	static constexpr uint64_t SEQ_BYTES = SEQ_ELEMENTS * sizeof(uint16_t);
	static constexpr uint64_t PAGE_HALF_LAYER = (uint64_t)Cfg::KV_HEADS * Cfg::PAGE_POSITIONS * Cfg::HEAD_DIM;
	static constexpr uint64_t PAGE_LAYER = 2u * PAGE_HALF_LAYER;
	static constexpr uint64_t PAGE_BYTES = (uint64_t)Cfg::LAYERS * PAGE_LAYER * sizeof(uint16_t);
	static constexpr uint32_t PARTIAL_SPLITS = Cfg::PAGED ? Cfg::SPLITS_MAX : (Cfg::SPLIT_KEYS != 0u ? (Cfg::WINDOW + Cfg::SPLIT_KEYS - 1u) / Cfg::SPLIT_KEYS : 0u);
	static constexpr bool HAS_PARTIALS = Cfg::ATTN_KIND != DSPARK_ATTN_RING_SINGLE_PASS;
	static constexpr bool HAS_D2T = Cfg::DRAFT_VOCAB != Cfg::VOCAB;

	using Draft = typename std::conditional<Cfg::PAGED, DsparkPagedDraft<Cfg::BLOCK, Cfg::RUNS_MAX>, DsparkRingDraft<Cfg::BLOCK, Cfg::WINDOW>>::type;
	using Picks = DsparkMarkovPicks<Cfg::BLOCK, Cfg::BRANCHES, Cfg::BRANCH_LEN>;
	using RingPool = DsparkRingPool<Cfg::WINDOW, Cfg::SEQUENCES_MAX>;
	using RingSeq = DsparkRingSequence<Cfg::WINDOW>;
	using PagedPool = DsparkPagedPool<Cfg::PAGE_POSITIONS, Cfg::PAGES_MAX, PAGE_BYTES, Cfg::PAGED ? Cfg::SEQUENCES_MAX : 1u, Cfg::RUNS_MAX>;

	struct Forward
	{
		Draft *host_draft;
		Draft *draft;
		uint16_t **host_pages;
		uint16_t **pages;
		Picks *host_picks;
		Picks *picks;
		uint16_t *hidden;
		uint16_t *x;
		uint16_t *q;
		uint16_t *q_roped;
		uint16_t *k;
		uint16_t *k_roped;
		uint16_t *v;
		float *partials;
		uint16_t *attended;
		uint16_t *projected;
		uint16_t *residual;
		uint16_t *gate;
		uint16_t *up;
		uint16_t *act;
		uint16_t *mlp;
		uint16_t *y;
		uint16_t *logits;
		SparkDrafterTopkBf16Workspace topk_ws;
		uint32_t *candidates;
		float *markov_row;
	};

	struct State
	{
		SparkWeightdLazyPack *pack;
		SparkGpuStream stream;
		SparkDrafterGraphCache graphs;
		uint32_t multiprocessors;
		uint32_t mask_token;
		float epsilon;
		float rope_gain;
		float softmax_scale;
		float *rope_inverse;
		const uint16_t *fc;
		const uint16_t *hidden_norm;
		const uint16_t *norm;
		const uint16_t *embed;
		const uint16_t *head;
		const int64_t *d2t;
		const uint16_t *markov_previous;
		const uint16_t *markov_next;
		const uint16_t *confidence_weight;
		const uint16_t *confidence_bias;
		DsparkStandardLayer layers[Cfg::LAYERS];
		Forward forward;
		uint16_t *host_taps;
		uint32_t *host_positions;
		uint16_t *taps;
		uint32_t *positions;
		uint16_t *features_raw;
		uint16_t *features;
		uint16_t *keys_raw;
		uint16_t *values_raw;
		uint16_t *caches;
		RingPool ring;
		PagedPool paged;
		uint64_t frames;
		uint64_t rows;
	};

	static uint16_t *RingKeys(const RingSeq *sequence, uint32_t layer)
	{
		return sequence->cache + (uint64_t)(2u * layer) * LAYER_KV;
	}

	static uint16_t *RingValues(const RingSeq *sequence, uint32_t layer)
	{
		return sequence->cache + (uint64_t)(2u * layer + 1u) * LAYER_KV;
	}

	static uint16_t *PagedKeySlot(const DsparkPagedSequence *sequence, uint32_t layer, uint32_t kv_head, uint64_t position)
	{
		uint16_t *page = sequence->pages[position / Cfg::PAGE_POSITIONS];
		return page + (uint64_t)layer * PAGE_LAYER + ((uint64_t)kv_head * Cfg::PAGE_POSITIONS + (position % Cfg::PAGE_POSITIONS)) * Cfg::HEAD_DIM;
	}

	static uint16_t *PagedValSlot(const DsparkPagedSequence *sequence, uint32_t layer, uint32_t kv_head, uint64_t position)
	{
		return PagedKeySlot(sequence, layer, kv_head, position) + PAGE_HALF_LAYER;
	}

	static SparkStatus Weights(State *state, const char *pack_path)
	{
		SparkJsonDocument document;
		uint64_t payload_base = 0u;
		SparkStatus status;
		memset(&document, 0, sizeof(document));
		status = DsparkLoadManifest(pack_path, &document, &payload_base);
		if ( status == SPARK_STATUS_OK )
			status = Cfg::CheckShape(state, &document);
		if ( status == SPARK_STATUS_OK )
		{
			const DsparkBf16Spec plain[] = {
				{Cfg::NAME_FC, (uint64_t)Cfg::HIDDEN * TAP_WIDTH, &state->fc},
				{Cfg::NAME_HIDDEN_NORM, Cfg::HIDDEN, &state->hidden_norm},
				{Cfg::NAME_NORM, Cfg::HIDDEN, &state->norm},
				{"target.embed", (uint64_t)Cfg::VOCAB * Cfg::HIDDEN, &state->embed},
				{"target.head", (uint64_t)Cfg::DRAFT_VOCAB * Cfg::HIDDEN, &state->head},
				{Cfg::NAME_MARKOV_W1, (uint64_t)Cfg::VOCAB * Cfg::MARKOV, &state->markov_previous},
				{Cfg::NAME_MARKOV_W2, (uint64_t)Cfg::DRAFT_VOCAB * Cfg::MARKOV, &state->markov_next},
				{Cfg::NAME_CONF_W, CONFIDENCE, &state->confidence_weight},
				{Cfg::NAME_CONF_B, 1u, &state->confidence_bias},
			};
			status = DsparkLoadBf16Table(state->pack, &document, payload_base, Cfg::LOG_TAG, plain, (uint32_t)(sizeof(plain) / sizeof(plain[0])));
		}
		if constexpr ( HAS_D2T )
		{
			if ( status == SPARK_STATUS_OK )
			{
				const void *d2t_slice = 0;
				status = DsparkTensorBytes(state->pack, &document, payload_base, Cfg::LOG_TAG, "d2t", (uint64_t)Cfg::DRAFT_VOCAB * sizeof(int64_t),
					&d2t_slice);
				if ( status == SPARK_STATUS_OK )
					state->d2t = (const int64_t *)d2t_slice;
			}
		}
		for ( uint32_t layer = 0u; layer < Cfg::LAYERS && status == SPARK_STATUS_OK; layer++ )
		{
			DsparkStandardLayer *weights = &state->layers[layer];
			const DsparkBf16Spec tensors[] = {
				{"input_layernorm.weight", Cfg::HIDDEN, &weights->input_norm},
				{"post_attention_layernorm.weight", Cfg::HIDDEN, &weights->post_norm},
				{"self_attn.q_proj.weight", (uint64_t)Q_WIDTH * Cfg::HIDDEN, &weights->q},
				{"self_attn.k_proj.weight", (uint64_t)KV_WIDTH * Cfg::HIDDEN, &weights->k},
				{"self_attn.v_proj.weight", (uint64_t)KV_WIDTH * Cfg::HIDDEN, &weights->v},
				{"self_attn.o_proj.weight", (uint64_t)Cfg::HIDDEN * Q_WIDTH, &weights->o},
				{"self_attn.q_norm.weight", Cfg::HEAD_DIM, &weights->q_norm},
				{"self_attn.k_norm.weight", Cfg::HEAD_DIM, &weights->k_norm},
				{"mlp.gate_proj.weight", (uint64_t)Cfg::MLP * Cfg::HIDDEN, &weights->gate},
				{"mlp.up_proj.weight", (uint64_t)Cfg::MLP * Cfg::HIDDEN, &weights->up},
				{"mlp.down_proj.weight", (uint64_t)Cfg::HIDDEN * Cfg::MLP, &weights->down},
			};
			status = DsparkLoadPrefixedBf16Table(state->pack, &document, payload_base, Cfg::LOG_TAG, Cfg::LAYER_PREFIX_FMT, layer, tensors,
				(uint32_t)(sizeof(tensors) / sizeof(tensors[0])));
		}
		SparkJsonDocumentDestroy(&document);
		return status;
	}

	static SparkStatus Scratch(State *state, uint64_t context_bytes)
	{
		Forward *forward = &state->forward;
		int multiprocessors = 0;
		if constexpr ( !Cfg::PAGED )
		{
			if ( !state->ring.Init(context_bytes, SEQ_ELEMENTS, &state->caches) )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		else
		{
			state->paged.context_bytes = context_bytes;
			if ( !DsparkAllocHost(&forward->host_pages, Cfg::PAGES_MAX) || !DSPARK_DEVICE(forward->pages, Cfg::PAGES_MAX) ||
				!DSPARK_DEVICE(forward->k_roped, Cfg::BLOCK * KV_WIDTH) )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		if constexpr ( HAS_PARTIALS )
			if ( !DSPARK_DEVICE(forward->partials, (uint64_t)Cfg::HEADS * PARTIAL_SPLITS * Cfg::BLOCK * (Cfg::HEAD_DIM + 2u)) )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
			!DsparkAllocHost(&state->host_taps, (size_t)Cfg::INGEST_ROWS * TAP_WIDTH) ||
			!DsparkAllocHost(&state->host_positions, Cfg::INGEST_ROWS) ||
			!DsparkAllocHost(&forward->host_draft, 1u) || !DsparkAllocHost(&forward->host_picks, 1u) ||
			!DSPARK_DEVICE(state->taps, (uint64_t)Cfg::INGEST_ROWS * TAP_WIDTH) || !DSPARK_DEVICE(state->positions, Cfg::INGEST_ROWS) ||
			!DSPARK_DEVICE(state->features_raw, (uint64_t)Cfg::INGEST_ROWS * Cfg::HIDDEN) ||
			!DSPARK_DEVICE(state->features, (uint64_t)Cfg::INGEST_ROWS * Cfg::HIDDEN) ||
			!DSPARK_DEVICE(state->keys_raw, (uint64_t)Cfg::INGEST_ROWS * KV_WIDTH) ||
			!DSPARK_DEVICE(state->values_raw, (uint64_t)Cfg::INGEST_ROWS * KV_WIDTH) || !DSPARK_DEVICE(forward->draft, 1u) ||
			!DSPARK_DEVICE(forward->picks, 1u) || !DSPARK_DEVICE(forward->hidden, Cfg::BLOCK * Cfg::HIDDEN) ||
			!DSPARK_DEVICE(forward->x, Cfg::BLOCK * Cfg::HIDDEN) || !DSPARK_DEVICE(forward->q, Cfg::BLOCK * Q_WIDTH) ||
			!DSPARK_DEVICE(forward->q_roped, Cfg::BLOCK * Q_WIDTH) || !DSPARK_DEVICE(forward->k, Cfg::BLOCK * KV_WIDTH) ||
			!DSPARK_DEVICE(forward->v, Cfg::BLOCK * KV_WIDTH) || !DSPARK_DEVICE(forward->attended, Cfg::BLOCK * Q_WIDTH) ||
			!DSPARK_DEVICE(forward->projected, Cfg::BLOCK * Cfg::HIDDEN) || !DSPARK_DEVICE(forward->residual, Cfg::BLOCK * Cfg::HIDDEN) ||
			!DSPARK_DEVICE(forward->gate, Cfg::BLOCK * Cfg::MLP) || !DSPARK_DEVICE(forward->up, Cfg::BLOCK * Cfg::MLP) ||
			!DSPARK_DEVICE(forward->act, Cfg::BLOCK * Cfg::MLP) || !DSPARK_DEVICE(forward->mlp, Cfg::BLOCK * Cfg::HIDDEN) ||
			!DSPARK_DEVICE(forward->y, Cfg::BLOCK * Cfg::HIDDEN) || !DSPARK_DEVICE(forward->logits, (uint64_t)Cfg::BLOCK * Cfg::DRAFT_VOCAB) ||
			SparkDrafterTopkBf16Alloc(&forward->topk_ws, Cfg::BLOCK) != SPARK_STATUS_OK ||
			!DSPARK_DEVICE(forward->candidates, Cfg::BLOCK * Cfg::CANDIDATES) ||
			!DSPARK_DEVICE(forward->markov_row, Cfg::BRANCHES * Cfg::CANDIDATES) )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		state->multiprocessors = (uint32_t)multiprocessors;
		return SPARK_STATUS_OK;
	}

	static SparkStatus IngestRows(State *state, RingSeq *sequence, uint32_t rows)
	{
		SparkStatus status = DsparkGemm(state->fc, state->taps, state->features_raw, rows, TAP_WIDTH, Cfg::HIDDEN, state->multiprocessors,
			state->stream);
		if ( status == SPARK_STATUS_OK )
			LM_LAUNCH((DsparkRmsKernel<Cfg::ROUND_RMS>), rows, DSPARK_THREADS, 0, state->stream, state->features_raw, state->hidden_norm,
				state->features, Cfg::HIDDEN, state->epsilon);
		for ( uint32_t layer = 0u; layer < Cfg::LAYERS && status == SPARK_STATUS_OK; layer++ )
		{
			status = DsparkGemm(state->layers[layer].k, state->features, state->keys_raw, rows, Cfg::HIDDEN, KV_WIDTH, state->multiprocessors,
				state->stream);
			if ( status == SPARK_STATUS_OK )
				status = DsparkGemm(state->layers[layer].v, state->features, state->values_raw, rows, Cfg::HIDDEN, KV_WIDTH, state->multiprocessors,
					state->stream);
			if ( status != SPARK_STATUS_OK )
				break;
			if constexpr ( !Cfg::PAGED )
			{
				LM_LAUNCH((DsparkHeadNormRopeRingKernel<Cfg::HEAD_DIM, Cfg::HEAD_THREADS, Cfg::WINDOW, COLUMNS, Cfg::HEAD_NORM_MODE, Cfg::USE_ROPE_GAIN,
						Draft>),
					rows * Cfg::KV_HEADS, Cfg::HEAD_THREADS, 0, state->stream, state->keys_raw, state->layers[layer].k_norm, state->rope_inverse,
					state->rope_gain, state->positions, Cfg::KV_HEADS, state->epsilon, (uint16_t *)0, RingKeys(sequence, layer), (const Draft *)0, 0u, 1u);
				LM_LAUNCH((DsparkStoreValuesRingKernel<KV_WIDTH, Cfg::HEAD_DIM, Cfg::WINDOW, COLUMNS, Draft>), rows, DSPARK_THREADS, 0, state->stream,
					state->values_raw, state->positions, (const Draft *)0, 0u, RingValues(sequence, layer));
			}
			else
			{
				LM_LAUNCH((DsparkPagedHeadNormRopeKernel<Cfg::HEAD_DIM, Cfg::PAGE_POSITIONS, PAGE_LAYER, true>), rows * Cfg::KV_HEADS, Cfg::HEAD_DIM,
					0, state->stream, state->keys_raw, state->layers[layer].k_norm, state->rope_inverse, state->positions, Cfg::KV_HEADS,
					state->epsilon, (uint16_t *)0, state->forward.pages, layer);
				LM_LAUNCH((DsparkPagedIngestStoreValuesKernel<KV_WIDTH, Cfg::HEAD_DIM, Cfg::PAGE_POSITIONS, PAGE_LAYER, PAGE_HALF_LAYER>), rows,
					DSPARK_THREADS, 0, state->stream, state->values_raw, state->positions, state->forward.pages, layer);
			}
		}
		if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
			status = SPARK_STATUS_INTERNAL_ERROR;
		return DsparkSettle(state->stream, status);
	}

	static SparkStatus Ingest(void *opaque, const SparkTapStreamReaderFrame *frame)
	{
		State *state = (State *)opaque;
		if constexpr ( !Cfg::PAGED )
			return state->ring.template IngestFrame<Cfg::TAPS, Cfg::HIDDEN, Cfg::INGEST_ROWS>(frame, state->host_taps, state->host_positions,
				state->taps, state->positions, state->stream, &state->frames, &state->rows,
				[&](RingSeq *sequence, uint32_t part) { return IngestRows(state, sequence, part); });
		else
			return state->paged.template IngestFrame<Cfg::TAPS, Cfg::HIDDEN, Cfg::INGEST_ROWS>(frame, state->host_taps, state->host_positions,
				state->taps, state->positions, state->stream, &state->frames, &state->rows,
				[&](DsparkPagedSequence *sequence) -> SparkStatus {
					for ( uint32_t page = 0u; page < sequence->page_count; page++ )
						state->forward.host_pages[page] = sequence->pages[page];
					if ( SparkGpuMemcpyAsync(state->forward.pages, state->forward.host_pages,
							(size_t)sequence->page_count * sizeof(*state->forward.host_pages), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) !=
						SPARK_GPU_SUCCESS )
					{
						(void)SparkGpuStreamSynchronize(state->stream);
						SPARK_FAIL(SPARK_STATUS_IO_ERROR);
					}
					return SPARK_STATUS_OK;
				},
				[&](DsparkPagedSequence *, uint64_t, uint32_t part) { return IngestRows(state, (RingSeq *)0, part); });
	}

	static SparkStatus PrepareRing(State *state, RingSeq *sequence, uint32_t anchor, uint64_t position, uint64_t checked,
		const SparkRowSampling *sampling, uint32_t drafts, uint32_t *splits_out)
	{
		Draft *draft = state->forward.host_draft;
		uint32_t splits = 0u;
		if ( anchor >= Cfg::VOCAB || position + Cfg::BLOCK > UINT32_MAX )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		memset(draft, 0, offsetof(Draft, valid));
		draft->cache = sequence->cache;
		for ( uint32_t row = 0u; row < Cfg::BLOCK; row++ )
			draft->positions[row] = (uint32_t)(position + row);
		draft->anchor = anchor;
		draft->mask = state->mask_token;
		draft->position = (uint32_t)position;
		draft->seed = sampling->seed;
		draft->inverse_temperature = sampling->inverse_temperature;
		draft->nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
		draft->scale = sampling->inverse_temperature > 0.0f ? sampling->inverse_temperature : 1.0f;
		draft->top_k = sampling->top_k;
		draft->drafts = drafts;
		RingPool::template Span<Cfg::BLOCK, Cfg::SLIDING, false>(sequence, checked, position, draft);
		if constexpr ( Cfg::SPLIT_KEYS != 0u )
		{
			const uint32_t span = draft->block_at - draft->start;
			splits = (span + Cfg::SPLIT_KEYS - 1u) / Cfg::SPLIT_KEYS;
			draft->splits = splits;
		}
		if ( splits_out != 0 )
			*splits_out = splits;
		if ( SparkGpuMemcpyAsync(state->forward.draft, draft, sizeof(*draft), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus PreparePaged(State *state, const DsparkPagedSequence *sequence, uint32_t anchor, uint64_t position,
		const SparkRowSampling *sampling, uint32_t drafts, uint32_t *splits_out)
	{
		Forward *forward = &state->forward;
		const SparkGpuStream stream = state->stream;
		const uint32_t pages = sequence != 0 ? sequence->page_count : 0u;
		uint32_t held = 0u, splits = 0u, split_rows = 0u;
		SparkStatus status;
		if ( anchor >= Cfg::VOCAB || position + Cfg::BLOCK > (uint64_t)Cfg::PAGES_MAX * Cfg::PAGE_POSITIONS )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		memset(forward->host_draft, 0, sizeof(*forward->host_draft));
		status = PagedPool::BuildRuns(sequence, position, forward->host_draft, &held);
		if ( status != SPARK_STATUS_OK )
			return status;
		for ( uint32_t row = 0u; row < Cfg::BLOCK; row++ )
			forward->host_draft->positions[row] = (uint32_t)(position + row);
		forward->host_draft->anchor = anchor;
		forward->host_draft->mask = state->mask_token;
		forward->host_draft->position = (uint32_t)position;
		if ( sampling != 0 )
		{
			forward->host_draft->seed = sampling->seed;
			forward->host_draft->top_k = sampling->top_k;
			forward->host_draft->inverse_temperature = sampling->inverse_temperature;
			forward->host_draft->nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
			forward->host_draft->scale = sampling->inverse_temperature > 0.0f ? sampling->inverse_temperature : 1.0f;
		}
		else
			forward->host_draft->scale = 1.0f;
		if ( held != 0u )
		{
			splits = (held + Cfg::SPLIT_ROWS_MIN - 1u) / Cfg::SPLIT_ROWS_MIN;
			splits = splits > Cfg::SPLITS_MAX ? Cfg::SPLITS_MAX : splits;
			split_rows = ((held + splits - 1u) / splits + Cfg::ATTN_TILE - 1u) / Cfg::ATTN_TILE * Cfg::ATTN_TILE;
			splits = (held + split_rows - 1u) / split_rows;
		}
		forward->host_draft->held = held;
		forward->host_draft->splits = splits;
		forward->host_draft->split_rows = split_rows;
		forward->host_draft->drafts = drafts;
		for ( uint32_t page = 0u; page < pages; page++ )
			forward->host_pages[page] = sequence->pages[page];
		if ( SparkGpuMemcpyAsync(forward->draft, forward->host_draft, sizeof(Draft), SPARK_GPU_COPY_HOST_TO_DEVICE, stream) != SPARK_GPU_SUCCESS ||
			(pages != 0u && SparkGpuMemcpyAsync(forward->pages, forward->host_pages, (size_t)pages * sizeof(*forward->host_pages),
				SPARK_GPU_COPY_HOST_TO_DEVICE, stream) != SPARK_GPU_SUCCESS) )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		*splits_out = splits;
		return SPARK_STATUS_OK;
	}

	static SparkStatus RunForward(State *state, uint32_t layers, uint32_t splits)
	{
		Forward *forward = &state->forward;
		const SparkGpuStream stream = state->stream;
		const uint32_t *positions = (const uint32_t *)((const uint8_t *)forward->draft + offsetof(Draft, positions));
		SparkStatus status = SPARK_STATUS_OK;
		LM_LAUNCH((DsparkEmbedBf16Kernel<Cfg::HIDDEN, Draft>), Cfg::BLOCK, DSPARK_THREADS, 0, stream, state->embed, forward->draft, forward->hidden);
		LM_LAUNCH((DsparkRmsKernel<Cfg::ROUND_RMS>), Cfg::BLOCK, DSPARK_THREADS, 0, stream, forward->hidden, state->layers[0].input_norm, forward->x,
			Cfg::HIDDEN, state->epsilon);
		for ( uint32_t layer = 0u; layer < layers && status == SPARK_STATUS_OK; layer++ )
		{
			const DsparkStandardLayer *weights = &state->layers[layer];
			status = DsparkGemm(weights->q, forward->x, forward->q, Cfg::BLOCK, Cfg::HIDDEN, Q_WIDTH, state->multiprocessors, stream);
			if ( status == SPARK_STATUS_OK )
				status = DsparkGemm(weights->k, forward->x, forward->k, Cfg::BLOCK, Cfg::HIDDEN, KV_WIDTH, state->multiprocessors, stream);
			if ( status == SPARK_STATUS_OK )
				status = DsparkGemm(weights->v, forward->x, forward->v, Cfg::BLOCK, Cfg::HIDDEN, KV_WIDTH, state->multiprocessors, stream);
			if ( status != SPARK_STATUS_OK )
				break;
			if constexpr ( !Cfg::PAGED )
			{
				const uint64_t keys_offset = (uint64_t)(2u * layer) * LAYER_KV;
				const uint64_t values_offset = (uint64_t)(2u * layer + 1u) * LAYER_KV;
				LM_LAUNCH((DsparkHeadNormRopeRingKernel<Cfg::HEAD_DIM, Cfg::HEAD_THREADS, Cfg::WINDOW, COLUMNS, Cfg::HEAD_NORM_MODE,
						Cfg::USE_ROPE_GAIN, Draft>),
					Cfg::BLOCK * Cfg::HEADS, Cfg::HEAD_THREADS, 0, stream, forward->q, weights->q_norm, state->rope_inverse, state->rope_gain,
					positions, Cfg::HEADS, state->epsilon, forward->q_roped, (uint16_t *)0, forward->draft, 0u, 0u);
				if ( LM_LAUNCH_DEPENDENT((DsparkHeadNormRopeRingKernel<Cfg::HEAD_DIM, Cfg::HEAD_THREADS, Cfg::WINDOW, COLUMNS, Cfg::HEAD_NORM_MODE,
							Cfg::USE_ROPE_GAIN, Draft>),
						Cfg::BLOCK * Cfg::KV_HEADS, Cfg::HEAD_THREADS, 0, stream, forward->k, weights->k_norm, state->rope_inverse, state->rope_gain,
						positions, Cfg::KV_HEADS, state->epsilon, (uint16_t *)0, (uint16_t *)0, forward->draft, keys_offset, 0u) !=
						SPARK_GPU_SUCCESS ||
					LM_LAUNCH_DEPENDENT((DsparkStoreValuesRingKernel<KV_WIDTH, Cfg::HEAD_DIM, Cfg::WINDOW, COLUMNS, Draft>), Cfg::BLOCK,
						DSPARK_THREADS, 0, stream, forward->v, (const uint32_t *)0, forward->draft, values_offset, (uint16_t *)0) !=
						SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				if constexpr ( Cfg::ATTN_KIND == DSPARK_ATTN_RING_SPLIT_MHA64 )
				{
					if ( splits != 0u &&
						LM_LAUNCH_DEPENDENT((DsparkRingSplitMha64Kernel<Cfg::HEADS, Cfg::BLOCK, COLUMNS, Cfg::HEAD_DIM, Cfg::SPLIT_KEYS,
								PARTIAL_SPLITS, Cfg::ATTN_TILE, Draft>),
							dim3(Cfg::HEADS, splits), Cfg::BLOCK * DSPARK_WARP, 0, stream, forward->q_roped, keys_offset, values_offset,
							forward->draft, state->softmax_scale, forward->partials) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					if ( LM_LAUNCH_DEPENDENT((DsparkAttentionMergeKernel<Cfg::HEADS, Cfg::KV_HEADS, Cfg::BLOCK, COLUMNS, Cfg::HEAD_DIM,
								PARTIAL_SPLITS, Cfg::CAUSAL, false, false, Draft>),
							Cfg::BLOCK * Cfg::HEADS, Cfg::HEAD_DIM, 0, stream, forward->q_roped, (const uint16_t *)0, (const uint16_t *)0,
							keys_offset, values_offset, forward->draft, state->softmax_scale, forward->partials, forward->attended) !=
						SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				else if constexpr ( Cfg::ATTN_KIND == DSPARK_ATTN_RING_SPLIT_GQA128 )
				{
					if ( splits != 0u &&
						LM_LAUNCH_DEPENDENT((DsparkSplitGqa128Kernel<Cfg::HEADS, Cfg::KV_HEADS, Cfg::ATTN_Q_GROUP, Cfg::BLOCK, COLUMNS, Cfg::HEAD_DIM,
								LAYER_KV, Cfg::SPLIT_KEYS, PARTIAL_SPLITS, Cfg::ATTN_TILE, 1u, 0u, 0u, false, Draft>),
							dim3(Cfg::KV_HEADS * ATTN_Q_GROUPS, splits), DSPARK_THREADS, 0, stream, forward->q_roped, (uint16_t *const *)0, layer,
							forward->draft, state->softmax_scale, forward->partials) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					if ( LM_LAUNCH_DEPENDENT((DsparkAttentionMergeKernel<Cfg::HEADS, Cfg::KV_HEADS, Cfg::BLOCK, COLUMNS, Cfg::HEAD_DIM,
								PARTIAL_SPLITS, Cfg::CAUSAL, false, false, Draft>),
							Cfg::BLOCK * Cfg::HEADS, Cfg::HEAD_DIM, 0, stream, forward->q_roped, (const uint16_t *)0, (const uint16_t *)0,
							keys_offset, values_offset, forward->draft, state->softmax_scale, forward->partials, forward->attended) !=
						SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				else if constexpr ( Cfg::ATTN_KIND == DSPARK_ATTN_RING_SINGLE_PASS )
				{
					if ( LM_LAUNCH_DEPENDENT((DsparkRingSinglePassAttentionKernel<Cfg::HEADS, Cfg::HEAD_DIM, Cfg::BLOCK, COLUMNS, LAYER_KV,
								Cfg::ATTN_TILE, Draft>),
							Cfg::HEADS, Cfg::BLOCK * DSPARK_WARP, 0, stream, forward->q_roped, forward->draft, layer, state->softmax_scale,
							forward->attended) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
			}
			else
			{
				LM_LAUNCH((DsparkPagedHeadNormRopeKernel<Cfg::HEAD_DIM, Cfg::PAGE_POSITIONS, PAGE_LAYER, false>), Cfg::BLOCK * Cfg::HEADS,
					Cfg::HEAD_DIM, 0, stream, forward->q, weights->q_norm, state->rope_inverse, positions, Cfg::HEADS, state->epsilon,
					forward->q_roped, (uint16_t *const *)0, 0u);
				LM_LAUNCH((DsparkPagedHeadNormRopeKernel<Cfg::HEAD_DIM, Cfg::PAGE_POSITIONS, PAGE_LAYER, false>), Cfg::BLOCK * Cfg::KV_HEADS,
					Cfg::HEAD_DIM, 0, stream, forward->k, weights->k_norm, state->rope_inverse, positions, Cfg::KV_HEADS, state->epsilon,
					forward->k_roped, (uint16_t *const *)0, 0u);
				if ( splits != 0u &&
					LM_LAUNCH_DEPENDENT((DsparkSplitGqa128Kernel<Cfg::HEADS, Cfg::KV_HEADS, Cfg::ATTN_Q_GROUP, Cfg::BLOCK, 0u, Cfg::HEAD_DIM, 0u, 0u,
							Cfg::SPLITS_MAX, Cfg::ATTN_TILE, Cfg::PAGE_POSITIONS, PAGE_LAYER, PAGE_HALF_LAYER, true, Draft>),
						dim3(Cfg::KV_HEADS * ATTN_Q_GROUPS, splits), DSPARK_THREADS, 0, stream, forward->q_roped, forward->pages, layer,
						forward->draft, state->softmax_scale, forward->partials) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				if ( LM_LAUNCH_DEPENDENT((DsparkAttentionMergeKernel<Cfg::HEADS, Cfg::KV_HEADS, Cfg::BLOCK, 0u, Cfg::HEAD_DIM, Cfg::SPLITS_MAX, false,
							true, false, Draft>),
						Cfg::BLOCK * Cfg::HEADS, Cfg::HEAD_DIM, 0, stream, forward->q_roped, forward->k_roped, forward->v, 0u, 0u, forward->draft,
						state->softmax_scale, forward->partials, forward->attended) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			status = DsparkGemm(weights->o, forward->attended, forward->projected, Cfg::BLOCK, Q_WIDTH, Cfg::HIDDEN, state->multiprocessors, stream);
			if ( status != SPARK_STATUS_OK )
				break;
			LM_LAUNCH((DsparkResidualRmsKernel<Cfg::ROUND_RMS>), Cfg::BLOCK, DSPARK_THREADS, 0, stream, forward->hidden, forward->projected,
				weights->post_norm, forward->residual, forward->x, Cfg::HIDDEN, state->epsilon);
			status = DsparkGemm(weights->gate, forward->x, forward->gate, Cfg::BLOCK, Cfg::HIDDEN, Cfg::MLP, state->multiprocessors, stream);
			if ( status == SPARK_STATUS_OK )
				status = DsparkGemm(weights->up, forward->x, forward->up, Cfg::BLOCK, Cfg::HIDDEN, Cfg::MLP, state->multiprocessors, stream);
			if ( status != SPARK_STATUS_OK )
				break;
			LM_LAUNCH((DsparkSwigluKernel), 2u * state->multiprocessors, DSPARK_THREADS, 0, stream, forward->gate, forward->up, forward->act,
				Cfg::BLOCK * Cfg::MLP);
			status = DsparkGemm(weights->down, forward->act, forward->mlp, Cfg::BLOCK, Cfg::MLP, Cfg::HIDDEN, state->multiprocessors, stream);
			if ( status != SPARK_STATUS_OK )
				break;
			if ( layer + 1u < layers )
				LM_LAUNCH((DsparkResidualRmsKernel<Cfg::ROUND_RMS>), Cfg::BLOCK, DSPARK_THREADS, 0, stream, forward->residual, forward->mlp,
					state->layers[layer + 1u].input_norm, forward->hidden, forward->x, Cfg::HIDDEN, state->epsilon);
			else if ( layers == Cfg::LAYERS )
				LM_LAUNCH((DsparkResidualRmsKernel<Cfg::ROUND_RMS>), Cfg::BLOCK, DSPARK_THREADS, 0, stream, forward->residual, forward->mlp,
					state->norm, forward->hidden, forward->y, Cfg::HIDDEN, state->epsilon);
			else
				LM_LAUNCH((DsparkResidualKernel), 2u * state->multiprocessors, DSPARK_THREADS, 0, stream, forward->residual, forward->mlp,
					forward->hidden, Cfg::BLOCK * Cfg::HIDDEN);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Heads(State *state, uint32_t branch_steps, uint32_t tree_enabled)
	{
		Forward *forward = &state->forward;
		return DsparkLaunchMarkovHeads<Cfg::BLOCK, Cfg::HIDDEN, Cfg::VOCAB, Cfg::DRAFT_VOCAB, Cfg::MARKOV, Cfg::CANDIDATES, Cfg::SAMPLED,
			Cfg::BRANCHES, HAS_D2T, Draft, Picks>(state->head, forward->y, forward->logits, &forward->topk_ws, forward->candidates,
			forward->markov_row, state->markov_previous, state->markov_next, state->d2t, state->confidence_weight, state->confidence_bias,
			forward->draft, forward->picks, forward->host_picks, branch_steps, tree_enabled, state->multiprocessors, state->stream);
	}

	static SparkStatus Close(void *opaque)
	{
		State *state = (State *)opaque;
		SparkStatus status = SPARK_STATUS_OK;
		if ( state == 0 )
			return SPARK_STATUS_OK;
		if ( state->stream != 0 )
			(void)SparkGpuStreamSynchronize(state->stream);
		SparkDrafterGraphCacheDestroy(&state->graphs);
		if constexpr ( Cfg::PAGED )
			state->paged.FreeAll();
		{
			Forward *f = &state->forward;
			void *buffers[] = {
				f->draft, f->pages, f->picks, f->hidden, f->x, f->q, f->q_roped, f->k, f->k_roped, f->v, f->partials, f->attended, f->projected,
				f->residual, f->gate, f->up, f->act, f->mlp, f->y, f->logits, f->candidates, f->markov_row, state->caches, state->taps,
				state->positions, state->features_raw, state->features, state->keys_raw, state->values_raw, state->rope_inverse,
			};
			void *host_buffers[] = { f->host_draft, (void *)f->host_pages, f->host_picks, state->host_taps, state->host_positions };
			SparkDrafterTopkBf16Free(&f->topk_ws);
			DsparkFreeDeviceBuffers(buffers, (uint32_t)(sizeof(buffers) / sizeof(buffers[0])));
			DsparkFreeHostBuffers(host_buffers, (uint32_t)(sizeof(host_buffers) / sizeof(host_buffers[0])));
		}
		if ( state->stream != 0 )
			(void)SparkGpuStreamDestroy(state->stream);
		if ( state->pack != 0 )
			status = SparkWeightdLazyPackDestroy(state->pack);
		free(state->ring.sequences);
		free(state);
		return status;
	}

	static SparkStatus Open(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
	{
		return DsparkDrafterOpen<State>(lane, Cfg::PAGED ? PAGE_BYTES : SEQ_BYTES,
			Cfg::PAGED ? "lane %s has a context cap below one context page" : "lane %s has a context cap below one sequence's window",
			Cfg::BAD_PACK_FMT, !Cfg::PAGED, state_out, bind, Close,
			[&](State *state) -> SparkStatus {
				SparkStatus status = Weights(state, lane->pack_path);
				if ( status == SPARK_STATUS_OK )
					status = Scratch(state, lane->context_bytes);
				if ( status == SPARK_STATUS_OK )
					status = DsparkSettle(state->stream, SPARK_STATUS_OK);
				return status;
			},
			[](const State *state) { return state->ring.capacity; });
	}

	static SparkStatus Propose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
	{
		State *state = (State *)opaque;
		const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
		const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
		const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
		uint32_t drafts = requested, splits = 0u;
		SparkStatus status;
		*count = 0u;
		if ( limit == 0u )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		RingSeq *ring_seq = 0;
		DsparkPagedSequence *paged_seq = 0;
		uint64_t checked = 0u;
		if constexpr ( !Cfg::PAGED )
		{
			const uint64_t first = request->position > Cfg::WINDOW ? request->position - Cfg::WINDOW : 0u;
			checked = first;
			status = state->ring.PrepareProposeContext(request, first, state->stream, &ring_seq, &checked,
				[&](RingSeq *dst, const RingSeq *src, uint32_t slot, uint32_t n) {
					return RingPool::template CopyKvRange<Cfg::LAYERS, Cfg::KV_HEADS, COLUMNS, Cfg::HEAD_DIM>(dst, src, slot, n, state->stream);
				});
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		else
		{
			paged_seq = state->paged.Find(request->sequence_id, 0u);
			if ( PagedPool::Covers(paged_seq, request->position) == 0u )
				return SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS;
			if ( paged_seq != 0 && PagedPool::Complete(paged_seq, request->position) == 0u )
			{
				status = state->paged.FillFromDonors(paged_seq, request,
					[&](DsparkPagedSequence *dst, const DsparkPagedSequence *src, uint64_t at, uint64_t end) -> SparkStatus {
						while ( at < end )
						{
							const uint64_t page_end = (at / Cfg::PAGE_POSITIONS + 1u) * Cfg::PAGE_POSITIONS;
							const uint64_t stop = end < page_end ? end : page_end;
							const size_t bytes = (size_t)(stop - at) * Cfg::HEAD_DIM * sizeof(uint16_t);
							for ( uint32_t layer = 0u; layer < Cfg::LAYERS; layer++ )
								for ( uint32_t head = 0u; head < Cfg::KV_HEADS; head++ )
									if ( SparkGpuMemcpyAsync(PagedKeySlot(dst, layer, head, at), PagedKeySlot(src, layer, head, at), bytes,
											SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS ||
										SparkGpuMemcpyAsync(PagedValSlot(dst, layer, head, at), PagedValSlot(src, layer, head, at), bytes,
											SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
										SPARK_FAIL(SPARK_STATUS_IO_ERROR);
							at = stop;
						}
						return SPARK_STATUS_OK;
					});
				if ( status != SPARK_STATUS_OK )
					return DsparkSettle(state->stream, status);
			}
			if ( paged_seq != 0 && PagedPool::TokensMatch(paged_seq, request) == 0u )
				return DsparkSettle(state->stream, SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT);
		}
		drafts = drafts < Cfg::BLOCK ? drafts : Cfg::BLOCK;
		drafts = drafts < limit - 1u ? drafts : limit - 1u;
		{
			const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
			const uint32_t tree_enabled = tree != 0u && drafts >= 2u ? 1u : 0u;
			const uint32_t branch_steps = drafts < Cfg::BRANCH_LEN ? drafts : Cfg::BRANCH_LEN;
			if constexpr ( !Cfg::PAGED )
				status = PrepareRing(state, ring_seq, request->anchor_token_id, request->position, checked, &request->sampling, drafts, &splits);
			else
				status = PreparePaged(state, paged_seq, request->anchor_token_id, request->position, &request->sampling, drafts, &splits);
			if ( status == SPARK_STATUS_OK )
			{
				const uint64_t graph_key = ((uint64_t)splits << 16u) | ((uint64_t)tree_enabled << 8u) | (uint64_t)branch_steps;
				status = SparkDrafterGraphRun(&state->graphs, state->stream, graph_key, [&]() -> SparkStatus {
					SparkStatus inner = RunForward(state, Cfg::LAYERS, splits);
					if ( inner == SPARK_STATUS_OK )
						inner = Heads(state, branch_steps, tree_enabled);
					return inner;
				});
			}
			status = DsparkSettle(state->stream, status);
			if ( status != SPARK_STATUS_OK )
				return status;
			*count = DsparkPackMarkovTree<Cfg::BLOCK, Cfg::BRANCHES>(request->anchor_token_id, source, state->forward.host_picks, tree, drafts,
				requested, limit, nodes);
		}
		return SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS;
	}

	static SparkStatus CacheRing(void *opaque, uint64_t sequence_id, uint32_t layer, uint16_t *keys, uint16_t *values, int64_t *positions)
	{
		State *state = (State *)opaque;
		if ( state == 0 )
			SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
		return state->ring.template CacheReadback<Cfg::LAYERS, LAYER_KV>(sequence_id, layer, keys, values, positions);
	}

	static SparkStatus CachePaged(void *opaque, uint64_t sequence_id, uint32_t layer, uint64_t first, uint32_t count, uint16_t *keys,
		uint16_t *values)
	{
		State *state = (State *)opaque;
		const DsparkPagedSequence *sequence = state != 0 ? state->paged.Find(sequence_id, 0u) : 0;
		uint64_t at = first, end = first + count;
		if ( sequence == 0 || layer >= Cfg::LAYERS || keys == 0 || values == 0 || end > sequence->held_end )
			SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
		while ( at < end )
		{
			const uint64_t page_end = (at / Cfg::PAGE_POSITIONS + 1u) * Cfg::PAGE_POSITIONS, stop = end < page_end ? end : page_end;
			const size_t bytes = (size_t)(stop - at) * Cfg::HEAD_DIM * sizeof(uint16_t);
			for ( uint32_t head = 0u; head < Cfg::KV_HEADS; head++ )
			{
				const uint64_t dst = ((uint64_t)head * count + (at - first)) * Cfg::HEAD_DIM;
				if ( DsparkReadback(keys + dst, PagedKeySlot(sequence, layer, head, at), bytes) != SPARK_STATUS_OK ||
					DsparkReadback(values + dst, PagedValSlot(sequence, layer, head, at), bytes) != SPARK_STATUS_OK )
					SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			}
			at = stop;
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus RunDiag(State *state, uint64_t sequence_id, uint32_t anchor, uint64_t position, uint32_t layers, SparkStatus missing_status)
	{
		uint32_t splits = 0u;
		SparkStatus status;
		if constexpr ( !Cfg::PAGED )
		{
			RingSeq *sequence = state != 0 ? state->ring.Live(sequence_id) : 0;
			const uint64_t first = position > Cfg::WINDOW ? position - Cfg::WINDOW : 0u;
			SparkRowSampling greedy;
			if ( sequence == 0 )
				SPARK_FAIL(missing_status);
			memset(&greedy, 0, sizeof(greedy));
			status = PrepareRing(state, sequence, anchor, position, RingPool::CheckedFrom(sequence, first, position), &greedy, 0u, &splits);
		}
		else
		{
			const DsparkPagedSequence *sequence = state != 0 ? state->paged.Find(sequence_id, 0u) : 0;
			if ( state == 0 || sequence == 0 )
				SPARK_FAIL(missing_status);
			status = PreparePaged(state, sequence, anchor, position, 0, 0u, &splits);
		}
		if ( status == SPARK_STATUS_OK )
			status = RunForward(state, layers, splits);
		return DsparkSettle(state->stream, status);
	}

	static SparkStatus ForwardDiag(void *opaque, uint64_t sequence_id, uint32_t anchor, uint64_t position, uint16_t *y, uint64_t *elapsed_ns)
	{
		State *state = (State *)opaque;
		const uint64_t started = SparkMonotonicNs();
		SparkStatus status = RunDiag(state, sequence_id, anchor, position, Cfg::LAYERS, SPARK_STATUS_NOT_FOUND);
		if ( elapsed_ns != 0 )
			*elapsed_ns = SparkMonotonicNs() - started;
		if ( status == SPARK_STATUS_OK && y != 0 )
			status = DsparkReadback(y, state->forward.y, (size_t)Cfg::BLOCK * Cfg::HIDDEN * sizeof(uint16_t));
		return status;
	}

	static SparkStatus TraceDiag(void *opaque, uint64_t sequence_id, uint32_t anchor, uint64_t position, uint32_t layers, uint16_t *hidden)
	{
		State *state = (State *)opaque;
		SparkStatus status;
		if ( state == 0 || hidden == 0 || layers == 0u || layers > Cfg::LAYERS )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		status = RunDiag(state, sequence_id, anchor, position, layers, SPARK_STATUS_INVALID_ARGUMENT);
		if ( status == SPARK_STATUS_OK )
			status = DsparkReadback(hidden, state->forward.hidden, (size_t)Cfg::BLOCK * Cfg::HIDDEN * sizeof(uint16_t));
		return status;
	}

	static const SparkDrafterModuleInterface *Interface()
	{
		static const SparkDrafterModuleInterface iface = {
			SPARK_DRAFTER_MODULE_ABI_VERSION,
			(uint32_t)sizeof(SparkDrafterModuleInterface),
			Cfg::INTERFACE_NAME,
			Open,
			Ingest,
			Propose,
			Close,
		};
		return &iface;
	}
};

#endif
