#include "inference/kernels/dspark_draft_stage_common.cuh"
#include "inference/kernels/mma.cuh"

#define K31_HIDDEN 7168u
#define K31_TAPS 5u
#define K31_TAP_WIDTH (K31_TAPS * K31_HIDDEN)
#define K31_LAYERS 5u
#define K31_KV_RANK 512u
#define K31_ROPE 64u
#define K31_LATENT (K31_KV_RANK + K31_ROPE)
#define K31_HEADS 96u
#define K31_NOPE 128u
#define K31_VALUE 128u
#define K31_QK (K31_NOPE + K31_ROPE)
#define K31_QUERY_RANK 1536u
#define K31_MLP 14336u
#define K31_BLOCK 7u
#define K31_BRANCHES 2u
#define K31_BRANCH_LEN 4u
#define K31_VOCAB 163840u
#define K31_QUERIES (K31_BLOCK * K31_HEADS)
#define K31_MARKOV 512u
#define K31_CANDIDATES 2048u
#define K31_SAMPLED 32u
#define K31_CONFIDENCE (K31_HIDDEN + K31_MARKOV)
#define K31_PAGE_POSITIONS 4096u
#define K31_PAGE_BYTES ((uint64_t)K31_LAYERS * K31_PAGE_POSITIONS * K31_LATENT * sizeof(uint16_t))
#define K31_PAGES_MAX 4096u
#define K31_SEQUENCES_MAX 64u
#define K31_INGEST_ROWS 256u
#define K31_CHUNK_BYTES 16u
#define K31_CHUNK_ELEMENTS (K31_CHUNK_BYTES / sizeof(uint16_t))
#define K31_ROW_BYTES (K31_LATENT * sizeof(uint16_t))
#define K31_ATTN_THREADS 256u
#define K31_ATTN_QUERIES 32u
#define K31_ATTN_TILE 32u
#define K31_ATTN_STRIDE (K31_ATTN_TILE + 8u)
#define K31_ATTN_TILES (K31_QUERIES / K31_ATTN_QUERIES)
#define K31_ATTN_SHARED_BYTES (2u * K31_ATTN_TILE * K31_ROW_BYTES + 2u * K31_ATTN_QUERIES * K31_ATTN_STRIDE * sizeof(float) + \
	K31_ATTN_QUERIES * K31_ATTN_STRIDE * sizeof(uint16_t) + 3u * K31_ATTN_QUERIES * sizeof(float) + 2u * K31_ATTN_TILE * sizeof(void *))
#define K31_MERGE_THREADS 128u
#define K31_SPLITS_MAX 8u
#define K31_SPLIT_ROWS_MIN 1024u
#define K31_RUNS_MAX 256u
#define K31_LOG2E 1.4426950408889634

typedef struct K31Layer
{
	const uint16_t *kv_a;
	const uint16_t *kv_a_norm;
	const uint16_t *input_norm;
	const uint16_t *post_norm;
	const uint16_t *q_a;
	const uint16_t *q_a_norm;
	const uint16_t *q_b;
	const uint16_t *kv_b;
	const uint16_t *o_proj;
	const uint16_t *gate;
	const uint16_t *up;
	const uint16_t *down;
} K31Layer;

using K31PagedPool = DsparkPagedPool<K31_PAGE_POSITIONS, K31_PAGES_MAX, K31_PAGE_BYTES, K31_SEQUENCES_MAX, K31_RUNS_MAX>;
using K31Picks = DsparkMarkovPicks<K31_BLOCK, K31_BRANCHES, K31_BRANCH_LEN>;

typedef struct K31Draft
{
	uint64_t seed;
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
	uint32_t positions[K31_BLOCK];
	uint32_t runs;
	uint32_t run_first[K31_RUNS_MAX];
	uint32_t run_position[K31_RUNS_MAX];
} K31Draft;

typedef struct K31Forward
{
	K31Draft *host_draft;
	K31Draft *draft;
	const uint16_t **host_pages;
	const uint16_t **pages;
	uint16_t **block_destinations;
	float *residual;
	uint16_t *x;
	uint16_t *q_a_raw;
	uint16_t *q_a;
	uint16_t *q;
	uint16_t *latent_raw;
	uint16_t *block_rows;
	uint16_t *queries;
	float *queries_f32;
	float *partials;
	float *stats;
	float *latent;
	uint16_t *values;
	uint16_t *attention;
	uint16_t *gate;
	uint16_t *up;
	uint16_t *act;
	uint16_t *mlp;
	uint16_t *y;
	uint16_t *logits;
	SparkDrafterTopkBf16Workspace topk_ws;
	uint32_t *candidates;
	float *markov_row;
	K31Picks *picks;
	K31Picks *host_picks;
} K31Forward;

typedef struct K31State
{
	SparkWeightdLazyPack *pack;
	SparkGpuStream stream;
	uint32_t multiprocessors;
	uint32_t mask_token;
	float epsilon;
	float rope_gain;
	float attention_scale;
	float *rope_inverse;
	const uint16_t *context_proj;
	const uint16_t *context_norm;
	const uint16_t *embed;
	const uint16_t *final_norm;
	const uint16_t *head;
	const uint16_t *markov_previous;
	const uint16_t *markov_next;
	const uint16_t *confidence_weight;
	const uint16_t *confidence_bias;
	K31Layer layers[K31_LAYERS];
	K31Forward forward;
	SparkDrafterGraphCache graphs;
	uint16_t **ingest_rows;
	uint16_t *host_taps;
	uint16_t *taps;
	uint16_t *features_raw;
	uint16_t *features;
	uint16_t *latent_raw;
	uint32_t *positions;
	uint32_t *host_positions;
	K31PagedPool paged;
	uint64_t frames;
	uint64_t rows;
} K31State;

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void K31LatentRowsKernel(const uint16_t *__restrict__ latent, const uint16_t *__restrict__ norm, const float *__restrict__ inverse, float gain,
	const uint32_t *__restrict__ positions, uint16_t *const *__restrict__ destinations, float epsilon)
{
	LmDependentWait();
	__shared__ float reduction[DSPARK_WARPS];
	const uint16_t *row = latent + (uint64_t)blockIdx.x * K31_LATENT;
	uint16_t *out = destinations[blockIdx.x];
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < K31_KV_RANK; index += DSPARK_THREADS )
	{
		const float value = LmBf16ToFloat(__ldg(row + index));
		total += value * value;
	}
	total = LmBlockSum<DSPARK_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)K31_KV_RANK + epsilon);
	for ( uint32_t index = threadIdx.x; index < K31_KV_RANK; index += DSPARK_THREADS )
		out[index] = LmFloatToBf16(LmBf16ToFloat(__ldg(row + index)) * scale * LmBf16ToFloat(__ldg(norm + index)));
	if ( threadIdx.x < K31_ROPE / 2u )
	{
		const uint32_t pair = threadIdx.x;
		float cosine, sine;
		DsparkRotation(positions[blockIdx.x], inverse[pair], gain, &cosine, &sine);
		const float even = LmBf16ToFloat(__ldg(row + K31_KV_RANK + 2u * pair)), odd = LmBf16ToFloat(__ldg(row + K31_KV_RANK + 2u * pair + 1u));
		out[K31_KV_RANK + 2u * pair] = LmFloatToBf16(even * cosine - odd * sine);
		out[K31_KV_RANK + 2u * pair + 1u] = LmFloatToBf16(even * sine + odd * cosine);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void K31ResidualNormKernel(float *__restrict__ residual, const uint16_t *__restrict__ addend, const uint16_t *__restrict__ weight,
	uint16_t *__restrict__ output, float epsilon)
{
	__shared__ float reduction[DSPARK_WARPS];
	float *row = residual + (uint64_t)blockIdx.x * K31_HIDDEN;
	const uint16_t *add = addend != 0 ? addend + (uint64_t)blockIdx.x * K31_HIDDEN : (const uint16_t *)0;
	uint16_t *out = output + (uint64_t)blockIdx.x * K31_HIDDEN;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < K31_HIDDEN; index += DSPARK_THREADS )
	{
		float value = row[index];
		if ( add != 0 )
		{
			value += LmBf16ToFloat(__ldg(add + index));
			row[index] = value;
		}
		total += value * value;
	}
	total = LmBlockSum<DSPARK_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)K31_HIDDEN + epsilon);
	for ( uint32_t index = threadIdx.x; index < K31_HIDDEN; index += DSPARK_THREADS )
		out[index] = LmFloatToBf16(row[index] * scale * LmBf16ToFloat(__ldg(weight + index)));
	LmDependentRelease();
}

__global__ __launch_bounds__(64u, 4)
void K31QueryKernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ kv_b, const float *__restrict__ inverse, float gain,
	const uint32_t *__restrict__ positions, uint16_t *__restrict__ queries, float *__restrict__ queries_f32)
{
	__shared__ float nope[K31_BLOCK][K31_NOPE];
	const uint32_t head = blockIdx.x >> 2u, part = blockIdx.x & 3u;
	const uint32_t col0 = part * 128u + 2u * threadIdx.x;
	float acc0[K31_BLOCK], acc1[K31_BLOCK];
	for ( uint32_t index = threadIdx.x; index < K31_BLOCK * K31_NOPE; index += 64u )
		nope[index / K31_NOPE][index % K31_NOPE] = LmBf16ToFloat(__ldg(q + (uint64_t)(index / K31_NOPE) * K31_HEADS * K31_QK + head * K31_QK + index % K31_NOPE));
#pragma unroll
	for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
		acc0[row] = acc1[row] = 0.0f;
	__syncthreads();
	const uint16_t *base_w = kv_b + (uint64_t)head * (K31_NOPE + K31_VALUE) * K31_KV_RANK + col0;
	uint32_t next_pair = __ldg((const uint32_t *)base_w);
	for ( uint32_t n = 0u; n < K31_NOPE; n++ )
	{
		const uint32_t pair = next_pair;
		if ( n + 1u < K31_NOPE )
			next_pair = __ldg((const uint32_t *)(base_w + (uint64_t)(n + 1u) * K31_KV_RANK));
		const float w0 = __uint_as_float(pair << 16u);
		const float w1 = __uint_as_float(pair & 0xffff0000u);
#pragma unroll
		for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
		{
			const float x = nope[row][n];
			acc0[row] += x * w0;
			acc1[row] += x * w1;
		}
	}
#pragma unroll
	for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
	{
		const uint64_t at = ((uint64_t)row * K31_HEADS + head) * K31_LATENT + col0;
		*(float2 *)(queries_f32 + at) = make_float2(acc0[row], acc1[row]);
		*(uint32_t *)(queries + at) = (uint32_t)LmFloatToBf16(acc0[row]) | ((uint32_t)LmFloatToBf16(acc1[row]) << 16u);
	}
	{
		const uint32_t idx = part * 64u + threadIdx.x;
		const uint32_t row = idx / (K31_ROPE / 2u), pair = idx % (K31_ROPE / 2u);
		const uint16_t *rope = q + (uint64_t)row * K31_HEADS * K31_QK + head * K31_QK + K31_NOPE;
		float cosine, sine;
		DsparkRotation(positions[row], inverse[pair], gain, &cosine, &sine);
		const float even = LmBf16ToFloat(__ldg(rope + 2u * pair)), odd = LmBf16ToFloat(__ldg(rope + 2u * pair + 1u));
		const float first = even * cosine - odd * sine, second = even * sine + odd * cosine;
		const uint64_t at = ((uint64_t)row * K31_HEADS + head) * K31_LATENT + K31_KV_RANK + 2u * pair;
		*(float2 *)(queries_f32 + at) = make_float2(first, second);
		*(uint32_t *)(queries + at) = (uint32_t)LmFloatToBf16(first) | ((uint32_t)LmFloatToBf16(second) << 16u);
	}
	LmDependentRelease();
}

static __device__ __forceinline__ const uint16_t *K31KeyRow(const uint16_t *const *__restrict__ pages, uint64_t layer_offset,
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
	return pages[position / K31_PAGE_POSITIONS] + layer_offset + (uint64_t)(position % K31_PAGE_POSITIONS) * K31_LATENT;
}

__global__ __launch_bounds__(K31_ATTN_THREADS, 1)
void K31ContextAttentionKernel(const uint16_t *__restrict__ queries, const uint16_t *const *__restrict__ pages, uint64_t layer_offset,
	const K31Draft *__restrict__ draft, float scale, float *__restrict__ partials, float *__restrict__ stats)
{
	LmDependentWait();
	const uint32_t *run_first = draft->run_first;
	const uint32_t *run_position = draft->run_position;
	const uint32_t runs = draft->runs, held = draft->held, split_rows = draft->split_rows;
	constexpr uint32_t CHUNKS = K31_LATENT / K31_CHUNK_ELEMENTS;
	constexpr uint32_t TILE = K31_ATTN_TILE;
	constexpr uint32_t QUERIES = K31_ATTN_QUERIES;
	constexpr uint32_t QUERY_TILES = QUERIES / 16u;
	constexpr uint32_t KEY_FRAGS = 2u;
	constexpr uint32_t KEY_GROUPS = TILE / 8u / KEY_FRAGS;
	constexpr uint32_t QUERY_KEYS = QUERIES * TILE / K31_ATTN_THREADS;
	constexpr uint32_t QUERY_THREADS = TILE / QUERY_KEYS;
	constexpr uint32_t HALF_STEPS = K31_LATENT / 32u;
	constexpr uint32_t WARP_COLUMNS = K31_KV_RANK / (K31_ATTN_THREADS / DSPARK_WARP);
	constexpr uint32_t COLUMN_FRAGS = WARP_COLUMNS / 8u;
	extern __shared__ __align__(128) uint8_t k31_attention_shared[];
	uint8_t *keys = k31_attention_shared;
	float *score_parts = (float *)(keys + 2u * TILE * K31_ROW_BYTES);
	uint16_t *weights = (uint16_t *)(score_parts + 2u * QUERIES * K31_ATTN_STRIDE);
	float *running_max = (float *)(weights + QUERIES * K31_ATTN_STRIDE);
	float *running_sum = running_max + QUERIES;
	float *rescale = running_sum + QUERIES;
	const uint16_t **slots = (const uint16_t **)(rescale + QUERIES);
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP, matrix = lane >> 3u, within = lane & 7u;
	const uint32_t query_tile = warp / (2u * KEY_GROUPS), half = (warp / KEY_GROUPS) & 1u, key_group = warp % KEY_GROUPS;
	const uint32_t first_query = blockIdx.x * QUERIES, split = blockIdx.y;
	const uint32_t begin = split * split_rows, end = begin + split_rows < held ? begin + split_rows : held;
	const uint32_t tiles = (end - begin + TILE - 1u) / TILE;
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys), query_base = key_base + TILE * K31_ROW_BYTES;
	uint32_t query_frag[HALF_STEPS][4];
	float output[QUERY_TILES][COLUMN_FRAGS][4];

	auto stage = [&](uint32_t which)
	{
		const uint32_t buffer = which & 1u;
		for ( uint32_t item = threadIdx.x; item < TILE * CHUNKS; item += K31_ATTN_THREADS )
		{
			const uint32_t slot_index = item / CHUNKS, piece = item % CHUNKS;
			const uint16_t *slot = slots[buffer * TILE + slot_index];
			const void *source = slot != 0 ? (const void *)(slot + piece * K31_CHUNK_ELEMENTS) : (const void *)queries;
			const uint32_t destination = key_base + (buffer * TILE + slot_index) * K31_ROW_BYTES + DsparkSwizzle(slot_index, piece) * K31_CHUNK_BYTES;
			DsparkStage16Cg(destination, source, slot != 0 ? K31_CHUNK_BYTES : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};

	for ( uint32_t index = threadIdx.x; index < QUERIES * CHUNKS; index += K31_ATTN_THREADS )
	{
		const uint32_t q = index / CHUNKS, chunk = index % CHUNKS;
		const uint16_t *source = queries + (uint64_t)(first_query + q) * K31_LATENT + chunk * K31_CHUNK_ELEMENTS;
		DsparkStage16Cg(query_base + q * K31_ROW_BYTES + DsparkSwizzle(q, chunk) * K31_CHUNK_BYTES, source, K31_CHUNK_BYTES);
	}
	asm volatile("cp.async.commit_group;\n" ::: "memory");
	if ( threadIdx.x < TILE )
	{
		slots[threadIdx.x] = K31KeyRow(pages, layer_offset, run_first, run_position, runs, begin + threadIdx.x, end);
		slots[TILE + threadIdx.x] = K31KeyRow(pages, layer_offset, run_first, run_position, runs, begin + TILE + threadIdx.x, end);
	}
	__syncthreads();
	stage(0u);
	asm volatile("cp.async.wait_group 1;\n" ::: "memory");
	__syncthreads();
	{
		const uint32_t a_row = query_tile * 16u + (matrix & 1u) * 8u + within;
#pragma unroll
		for ( uint32_t step = 0u; step < HALF_STEPS; step++ )
			DsparkLoadX4(query_base + a_row * K31_ROW_BYTES + DsparkSwizzle(a_row, (half * HALF_STEPS + step) * 2u + (matrix >> 1u)) * K31_CHUNK_BYTES,
				query_frag[step]);
	}
	if ( threadIdx.x < QUERIES )
	{
		running_max[threadIdx.x] = -INFINITY;
		running_sum[threadIdx.x] = 0.0f;
	}
#pragma unroll
	for ( uint32_t m = 0u; m < QUERY_TILES; m++ )
#pragma unroll
		for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; frag++ )
			output[m][frag][0] = output[m][frag][1] = output[m][frag][2] = output[m][frag][3] = 0.0f;
	__syncthreads();

	for ( uint32_t tile = 0u; tile < tiles; tile++ )
	{
		const uint32_t buffer = tile & 1u, tile_base = key_base + buffer * TILE * K31_ROW_BYTES, tile_first = begin + tile * TILE;
		if ( tile + 1u < tiles )
		{
			stage(tile + 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		{
			float score[KEY_FRAGS][4];
			uint32_t b[4], first[2], second[2];
			const uint32_t b_row = (key_group * KEY_FRAGS + (matrix >> 1u)) * 8u + within;
#pragma unroll
			for ( uint32_t frag = 0u; frag < KEY_FRAGS; frag++ )
				score[frag][0] = score[frag][1] = score[frag][2] = score[frag][3] = 0.0f;
#pragma unroll
			for ( uint32_t step = 0u; step < HALF_STEPS; step++ )
			{
				const uint32_t b_chunk = (half * HALF_STEPS + step) * 2u + (matrix & 1u);
				DsparkLoadX4(tile_base + b_row * K31_ROW_BYTES + DsparkSwizzle(b_row, b_chunk) * K31_CHUNK_BYTES, b);
				first[0] = b[0];
				first[1] = b[1];
				second[0] = b[2];
				second[1] = b[3];
				LmMmaBf16(score[0], query_frag[step], first);
				LmMmaBf16(score[1], query_frag[step], second);
			}
#pragma unroll
			for ( uint32_t frag = 0u; frag < KEY_FRAGS; frag++ )
#pragma unroll
				for ( uint32_t index = 0u; index < 4u; index++ )
					score_parts[(half * QUERIES + query_tile * 16u + (lane >> 2u) + (index >> 1u) * 8u) * K31_ATTN_STRIDE +
						(key_group * KEY_FRAGS + frag) * 8u + (lane & 3u) * 2u + (index & 1u)] = score[frag][index];
		}
		__syncthreads();
		{
			const uint32_t own = threadIdx.x / QUERY_THREADS, part = threadIdx.x % QUERY_THREADS;
			float value[QUERY_KEYS], peak = -INFINITY, total = 0.0f, previous, next, carry;
#pragma unroll
			for ( uint32_t index = 0u; index < QUERY_KEYS; index++ )
			{
				const uint32_t key = part + index * QUERY_THREADS;
				value[index] = tile_first + key < end
					? (score_parts[own * K31_ATTN_STRIDE + key] + score_parts[(QUERIES + own) * K31_ATTN_STRIDE + key]) * scale : -INFINITY;
				peak = fmaxf(peak, value[index]);
			}
#pragma unroll
			for ( uint32_t shift = QUERY_THREADS / 2u; shift > 0u; shift >>= 1u )
				peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, shift));
			previous = running_max[own];
			next = fmaxf(previous, peak);
			carry = previous == -INFINITY ? 0.0f : exp2f(previous - next);
#pragma unroll
			for ( uint32_t index = 0u; index < QUERY_KEYS; index++ )
			{
				const float weight = value[index] == -INFINITY ? 0.0f : exp2f(value[index] - next);
				weights[own * K31_ATTN_STRIDE + part + index * QUERY_THREADS] = LmFloatToBf16(weight);
				total += weight;
			}
#pragma unroll
			for ( uint32_t shift = QUERY_THREADS / 2u; shift > 0u; shift >>= 1u )
				total += __shfl_xor_sync(0xffffffffu, total, shift);
			__syncwarp();
			if ( part == 0u )
			{
				running_sum[own] = running_sum[own] * carry + total;
				running_max[own] = next;
				rescale[own] = carry;
			}
		}
		__syncthreads();
		{
			uint32_t a[QUERY_TILES][4], b[4], first[2], second[2];
#pragma unroll
			for ( uint32_t m = 0u; m < QUERY_TILES; m++ )
			{
				const float low = rescale[m * 16u + (lane >> 2u)], high = rescale[m * 16u + (lane >> 2u) + 8u];
#pragma unroll
				for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; frag++ )
				{
					output[m][frag][0] *= low;
					output[m][frag][1] *= low;
					output[m][frag][2] *= high;
					output[m][frag][3] *= high;
				}
			}
#pragma unroll
			for ( uint32_t step = 0u; step < TILE / 16u; step++ )
			{
#pragma unroll
				for ( uint32_t m = 0u; m < QUERY_TILES; m++ )
				{
					const uint32_t a_row = m * 16u + (matrix & 1u) * 8u + within, a_column = step * 16u + (matrix >> 1u) * 8u;
					DsparkLoadX4((uint32_t)__cvta_generic_to_shared(weights + a_row * K31_ATTN_STRIDE + a_column), a[m]);
				}
#pragma unroll
				for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; frag += 2u )
				{
					const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
					const uint32_t b_chunk = (warp * WARP_COLUMNS + frag * 8u) / 8u + (matrix >> 1u);
					DsparkLoadX4Trans(tile_base + b_row * K31_ROW_BYTES + DsparkSwizzle(b_row, b_chunk) * K31_CHUNK_BYTES, b);
					first[0] = b[0];
					first[1] = b[1];
					second[0] = b[2];
					second[1] = b[3];
#pragma unroll
					for ( uint32_t m = 0u; m < QUERY_TILES; m++ )
					{
						LmMmaBf16(output[m][frag], a[m], first);
						LmMmaBf16(output[m][frag + 1u], a[m], second);
					}
				}
			}
		}
		if ( threadIdx.x < TILE && tile + 2u < tiles )
			slots[buffer * TILE + threadIdx.x] = K31KeyRow(pages, layer_offset, run_first, run_position, runs, tile_first + 2u * TILE + threadIdx.x, end);
		__syncthreads();
	}
#pragma unroll
	for ( uint32_t m = 0u; m < QUERY_TILES; m++ )
#pragma unroll
		for ( uint32_t frag = 0u; frag < COLUMN_FRAGS; frag++ )
#pragma unroll
			for ( uint32_t index = 0u; index < 4u; index += 2u )
			{
				const uint32_t q = m * 16u + (lane >> 2u) + (index >> 1u) * 8u;
				const uint32_t column = warp * WARP_COLUMNS + frag * 8u + (lane & 3u) * 2u;
				*(float2 *)(partials + ((uint64_t)split * K31_QUERIES + first_query + q) * K31_KV_RANK + column) =
					make_float2(output[m][frag][index], output[m][frag][index + 1u]);
			}
	if ( threadIdx.x < QUERIES )
	{
		stats[((uint64_t)split * K31_QUERIES + first_query + threadIdx.x) * 2u] = running_max[threadIdx.x];
		stats[((uint64_t)split * K31_QUERIES + first_query + threadIdx.x) * 2u + 1u] = running_sum[threadIdx.x];
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(K31_MERGE_THREADS, 2)
void K31AttentionMergeKernel(const float *__restrict__ partials, const float *__restrict__ stats, const K31Draft *__restrict__ draft,
	const float *__restrict__ queries, const uint16_t *__restrict__ block_rows, float scale, float *__restrict__ latent)
{
	LmDependentWait();
	constexpr uint32_t PER = K31_KV_RANK / K31_MERGE_THREADS;
	__shared__ float reduction[K31_MERGE_THREADS / DSPARK_WARP];
	const uint32_t splits = draft->splits;
	const uint32_t query = blockIdx.x, row = query / K31_HEADS;
	const float *own = queries + (uint64_t)query * K31_LATENT;
	float accumulated[PER], scores[K31_BLOCK], best = -INFINITY, total = 0.0f, peak = -INFINITY, next, keep, denominator;
#pragma unroll
	for ( uint32_t column = 0u; column < PER; column++ )
		accumulated[column] = 0.0f;
	for ( uint32_t split = 0u; split < splits; split++ )
		best = fmaxf(best, stats[((uint64_t)split * K31_QUERIES + query) * 2u]);
	for ( uint32_t split = 0u; split < splits; split++ )
	{
		const float *part = stats + ((uint64_t)split * K31_QUERIES + query) * 2u;
		const float weight = exp2f(part[0] - best);
		total += weight * part[1];
#pragma unroll
		for ( uint32_t column = 0u; column < PER; column++ )
			accumulated[column] += weight * partials[((uint64_t)split * K31_QUERIES + query) * K31_KV_RANK + column * K31_MERGE_THREADS + threadIdx.x];
	}
#pragma unroll
	for ( uint32_t key = 0u; key < K31_BLOCK; key++ )
	{
		float dot = 0.0f;
		scores[key] = -INFINITY;
		if ( key > row )
			continue;
		for ( uint32_t element = threadIdx.x; element < K31_LATENT; element += K31_MERGE_THREADS )
			dot += own[element] * LmBf16ToFloat(__ldg(block_rows + (uint64_t)key * K31_LATENT + element));
		scores[key] = LmBlockSum<K31_MERGE_THREADS>(dot, reduction) * scale;
		peak = fmaxf(peak, scores[key]);
	}
	next = fmaxf(best, peak);
	keep = exp2f(best - next);
	denominator = total * keep;
#pragma unroll
	for ( uint32_t column = 0u; column < PER; column++ )
		accumulated[column] *= keep;
#pragma unroll
	for ( uint32_t key = 0u; key < K31_BLOCK; key++ )
	{
		if ( key > row )
			continue;
		const float weight = exp2f(scores[key] - next);
		denominator += weight;
#pragma unroll
		for ( uint32_t column = 0u; column < PER; column++ )
			accumulated[column] += weight * LmBf16ToFloat(__ldg(block_rows + (uint64_t)key * K31_LATENT + column * K31_MERGE_THREADS + threadIdx.x));
	}
#pragma unroll
	for ( uint32_t column = 0u; column < PER; column++ )
		latent[(uint64_t)query * K31_KV_RANK + column * K31_MERGE_THREADS + threadIdx.x] = accumulated[column] / denominator;
	LmDependentRelease();
}

__global__ __launch_bounds__(128u, 2)
void K31ValueKernel(const float *__restrict__ latent, const uint16_t *__restrict__ kv_b, uint16_t *__restrict__ values)
{
	LmDependentWait();
	__shared__ float rows[K31_BLOCK][K31_KV_RANK];
	const uint32_t head = blockIdx.x >> 2u, part = blockIdx.x & 3u;
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	for ( uint32_t index = threadIdx.x; index < K31_BLOCK * (K31_KV_RANK / 4u); index += 128u )
	{
		const uint32_t r = index / (K31_KV_RANK / 4u), c = (index % (K31_KV_RANK / 4u)) * 4u;
		*(float4 *)(rows[r] + c) = __ldg((const float4 *)(latent + ((uint64_t)r * K31_HEADS + head) * K31_KV_RANK + c));
	}
	__syncthreads();
	const uint32_t val_begin = part * (K31_VALUE / 4u), val_end = val_begin + (K31_VALUE / 4u);
	for ( uint32_t value = val_begin + warp; value < val_end; value += 4u )
	{
		const uint16_t *weight = kv_b + ((uint64_t)head * (K31_NOPE + K31_VALUE) + K31_NOPE + value) * K31_KV_RANK;
		float sum[K31_BLOCK];
#pragma unroll
		for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
			sum[row] = 0.0f;
		for ( uint32_t element = lane; element < K31_KV_RANK; element += DSPARK_WARP )
		{
			const float w = LmBf16ToFloat(__ldg(weight + element));
#pragma unroll
			for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
				sum[row] += rows[row][element] * w;
		}
#pragma unroll
		for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
			sum[row] = DsparkWarpSum(sum[row]);
		if ( lane == 0u )
#pragma unroll
			for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
				values[(uint64_t)row * K31_HEADS * K31_VALUE + head * K31_VALUE + value] = LmFloatToBf16(sum[row]);
	}
	LmDependentRelease();
}

static float K31ConfigFloat(const SparkJsonDocument *document, int32_t object, const char *name, float fallback)
{
	const int32_t token = DsparkMember(document, object, name);
	float value = fallback;
	if ( token >= 0 )
		(void)SparkJsonGetFloat(document, token, &value);
	return value;
}

static SparkStatus K31ShapeAndRope(K31State *state, const SparkJsonDocument *document)
{
	const int32_t config = DsparkMember(document, SparkJsonGetRootToken(document), "config");
	const int32_t checkpoint = DsparkMember(document, config, "checkpoint_config");
	const int32_t rope = DsparkMember(document, checkpoint, "rope_parameters");
	const int32_t mask = DsparkMember(document, checkpoint, "mask_token_id");
	const DsparkExpectedU64 expected[] = {
		{checkpoint, "hidden_size", K31_HIDDEN}, {checkpoint, "num_attention_heads", K31_HEADS}, {checkpoint, "q_lora_rank", K31_QUERY_RANK},
		{checkpoint, "kv_lora_rank", K31_KV_RANK}, {checkpoint, "qk_nope_head_dim", K31_NOPE}, {checkpoint, "qk_rope_head_dim", K31_ROPE},
		{checkpoint, "v_head_dim", K31_VALUE}, {checkpoint, "intermediate_size", K31_MLP}, {checkpoint, "num_hidden_layers", K31_LAYERS},
		{checkpoint, "num_target_layers", K31_TAPS}, {checkpoint, "block_size", K31_BLOCK}, {checkpoint, "vocab_size", K31_VOCAB},
		{checkpoint, "draft_vocab_size", K31_VOCAB}, {checkpoint, "markov_rank", K31_MARKOV},
	};
	uint64_t value = 0u;
	float host[K31_ROPE / 2u];
	SparkStatus status = DsparkCheckU64Table(document, "k31 dspark", expected, (uint32_t)(sizeof(expected) / sizeof(expected[0])));
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( mask < 0 || SparkJsonGetUInt64(document, mask, &value) != SPARK_STATUS_OK || value >= K31_VOCAB || rope < 0 )
		return SPARK_STATUS_SCHEMA_ERROR;
	state->mask_token = (uint32_t)value;
	{
		const double base = K31ConfigFloat(document, rope, "rope_theta", 10000.0f), factor = K31ConfigFloat(document, rope, "factor", 1.0f);
		const double original = K31ConfigFloat(document, rope, "original_max_position_embeddings", 4096.0f);
		const double fast = K31ConfigFloat(document, rope, "beta_fast", 32.0f), slow = K31ConfigFloat(document, rope, "beta_slow", 1.0f);
		const double mscale = K31ConfigFloat(document, rope, "mscale", 1.0f), mscale_all = K31ConfigFloat(document, rope, "mscale_all_dim", 0.0f);
		const double attention = K31ConfigFloat(document, rope, "attn_factor", 1.0f), dimension = (double)K31_ROPE;
		double low = floor(dimension * log(original / (fast * 2.0 * DSPARK_PI)) / (2.0 * log(base)));
		double high = ceil(dimension * log(original / (slow * 2.0 * DSPARK_PI)) / (2.0 * log(base)));
		state->epsilon = K31ConfigFloat(document, checkpoint, "rms_norm_eps", 1e-6f);
		low = low < 0.0 ? 0.0 : low;
		high = high > dimension - 1.0 ? dimension - 1.0 : high;
		if ( low == high )
			high += 0.001;
		for ( uint32_t pair = 0u; pair < K31_ROPE / 2u; pair++ )
		{
			const double extrapolated = 1.0 / pow(base, (2.0 * pair) / dimension);
			double ramp = ((double)pair - low) / (high - low);
			ramp = ramp < 0.0 ? 0.0 : ramp > 1.0 ? 1.0 : ramp;
			host[pair] = (float)(extrapolated / factor * ramp + extrapolated * (1.0 - ramp));
		}
		state->rope_gain = (float)(DsparkYarnScale(factor, mscale) / DsparkYarnScale(factor, mscale_all) * attention);
		state->attention_scale = (float)(DsparkYarnScale(factor, mscale_all) * DsparkYarnScale(factor, mscale_all) / sqrt((double)K31_QK) * K31_LOG2E);
	}
	if ( SparkGpuMalloc((void **)&state->rope_inverse, sizeof(host)) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpy(state->rope_inverse, host, sizeof(host), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SPARK_STATUS_OK;
}

static SparkStatus K31Weights(K31State *state, const char *pack_path)
{
	SparkJsonDocument document;
	uint64_t payload_base = 0u;
	SparkStatus status;
	memset(&document, 0, sizeof(document));
	status = DsparkLoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = K31ShapeAndRope(state, &document);
	if ( status == SPARK_STATUS_OK )
	{
		const DsparkBf16Spec plain[] = {
			{"context_proj.weight", (uint64_t)K31_HIDDEN * K31_TAP_WIDTH, &state->context_proj},
			{"context_norm.weight", K31_HIDDEN, &state->context_norm},
			{"final_norm.weight", K31_HIDDEN, &state->final_norm},
			{"target.embed", (uint64_t)K31_VOCAB * K31_HIDDEN, &state->embed},
			{"target.head", (uint64_t)K31_VOCAB * K31_HIDDEN, &state->head},
			{"markov_head.markov_w1.weight", (uint64_t)K31_VOCAB * K31_MARKOV, &state->markov_previous},
			{"markov_head.markov_w2.weight", (uint64_t)K31_VOCAB * K31_MARKOV, &state->markov_next},
			{"confidence_head.proj.weight", K31_CONFIDENCE, &state->confidence_weight},
			{"confidence_head.proj.bias", 1u, &state->confidence_bias},
		};
		status = DsparkLoadBf16Table(state->pack, &document, payload_base, "k31 dspark", plain, (uint32_t)(sizeof(plain) / sizeof(plain[0])));
	}
	for ( uint32_t layer = 0u; layer < K31_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		K31Layer *w = &state->layers[layer];
		const DsparkBf16Spec tensors[] = {
			{"self_attn.kv_a_proj_with_mqa.weight", (uint64_t)K31_LATENT * K31_HIDDEN, &w->kv_a},
			{"self_attn.kv_a_layernorm.weight", K31_KV_RANK, &w->kv_a_norm},
			{"input_layernorm.weight", K31_HIDDEN, &w->input_norm},
			{"post_attention_layernorm.weight", K31_HIDDEN, &w->post_norm},
			{"self_attn.q_a_proj.weight", (uint64_t)K31_QUERY_RANK * K31_HIDDEN, &w->q_a},
			{"self_attn.q_a_layernorm.weight", K31_QUERY_RANK, &w->q_a_norm},
			{"self_attn.q_b_proj.weight", (uint64_t)K31_HEADS * K31_QK * K31_QUERY_RANK, &w->q_b},
			{"self_attn.kv_b_proj.weight", (uint64_t)K31_HEADS * (K31_NOPE + K31_VALUE) * K31_KV_RANK, &w->kv_b},
			{"self_attn.o_proj.weight", (uint64_t)K31_HIDDEN * K31_HEADS * K31_VALUE, &w->o_proj},
			{"mlp.gate_proj.weight", (uint64_t)K31_MLP * K31_HIDDEN, &w->gate},
			{"mlp.up_proj.weight", (uint64_t)K31_MLP * K31_HIDDEN, &w->up},
			{"mlp.down_proj.weight", (uint64_t)K31_HIDDEN * K31_MLP, &w->down},
		};
		status = DsparkLoadPrefixedBf16Table(state->pack, &document, payload_base, "k31 dspark", "layers.%u.%s", layer, tensors,
			(uint32_t)(sizeof(tensors) / sizeof(tensors[0])));
	}
	SparkJsonDocumentDestroy(&document);
	return status;
}

static SparkStatus K31Scratch(K31State *state)
{
	K31Forward *f = &state->forward;
	uint16_t *destinations[K31_BLOCK];
	int multiprocessors = 0;
	if ( SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		SparkGpuFuncSetAttribute(K31ContextAttentionKernel, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)K31_ATTN_SHARED_BYTES) !=
			SPARK_GPU_SUCCESS ||
		!DsparkAllocHost(&state->host_taps, (size_t)K31_INGEST_ROWS * K31_TAP_WIDTH) ||
		!DsparkAllocHost(&state->host_positions, K31_INGEST_ROWS) || !DsparkAllocHost(&f->host_draft, 1u) ||
		!DsparkAllocHost(&f->host_pages, K31_PAGES_MAX) || !DsparkAllocHost(&f->host_picks, 1u) ||
		!DSPARK_DEVICE(state->taps, (uint64_t)K31_INGEST_ROWS * K31_TAP_WIDTH) ||
		!DSPARK_DEVICE(state->features_raw, (uint64_t)K31_INGEST_ROWS * K31_HIDDEN) ||
		!DSPARK_DEVICE(state->features, (uint64_t)K31_INGEST_ROWS * K31_HIDDEN) ||
		!DSPARK_DEVICE(state->latent_raw, (uint64_t)K31_INGEST_ROWS * K31_LATENT) || !DSPARK_DEVICE(state->positions, K31_INGEST_ROWS) ||
		!DSPARK_DEVICE(state->ingest_rows, (uint64_t)K31_LAYERS * K31_INGEST_ROWS) || !DSPARK_DEVICE(f->draft, 1u) ||
		!DSPARK_DEVICE(f->pages, K31_PAGES_MAX) || !DSPARK_DEVICE(f->block_destinations, K31_BLOCK) ||
		!DSPARK_DEVICE(f->residual, K31_BLOCK * K31_HIDDEN) || !DSPARK_DEVICE(f->x, K31_BLOCK * K31_HIDDEN) ||
		!DSPARK_DEVICE(f->q_a_raw, K31_BLOCK * K31_QUERY_RANK) || !DSPARK_DEVICE(f->q_a, K31_BLOCK * K31_QUERY_RANK) ||
		!DSPARK_DEVICE(f->q, K31_BLOCK * K31_HEADS * K31_QK) || !DSPARK_DEVICE(f->latent_raw, K31_BLOCK * K31_LATENT) ||
		!DSPARK_DEVICE(f->block_rows, K31_BLOCK * K31_LATENT) || !DSPARK_DEVICE(f->queries, (uint64_t)K31_QUERIES * K31_LATENT) ||
		!DSPARK_DEVICE(f->queries_f32, (uint64_t)K31_QUERIES * K31_LATENT) ||
		!DSPARK_DEVICE(f->partials, (uint64_t)K31_SPLITS_MAX * K31_QUERIES * K31_KV_RANK) ||
		!DSPARK_DEVICE(f->stats, (uint64_t)K31_SPLITS_MAX * K31_QUERIES * 2u) ||
		!DSPARK_DEVICE(f->latent, (uint64_t)K31_QUERIES * K31_KV_RANK) || !DSPARK_DEVICE(f->values, K31_BLOCK * K31_HEADS * K31_VALUE) ||
		!DSPARK_DEVICE(f->attention, K31_BLOCK * K31_HIDDEN) || !DSPARK_DEVICE(f->gate, K31_BLOCK * K31_MLP) ||
		!DSPARK_DEVICE(f->up, K31_BLOCK * K31_MLP) || !DSPARK_DEVICE(f->act, K31_BLOCK * K31_MLP) ||
		!DSPARK_DEVICE(f->mlp, K31_BLOCK * K31_HIDDEN) || !DSPARK_DEVICE(f->y, K31_BLOCK * K31_HIDDEN) ||
		!DSPARK_DEVICE(f->logits, (uint64_t)K31_BLOCK * K31_VOCAB) || SparkDrafterTopkBf16Alloc(&f->topk_ws, K31_BLOCK) != SPARK_STATUS_OK ||
		!DSPARK_DEVICE(f->candidates, K31_BLOCK * K31_CANDIDATES) || !DSPARK_DEVICE(f->markov_row, K31_BRANCHES * K31_CANDIDATES) ||
		!DSPARK_DEVICE(f->picks, 1u) )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
		destinations[row] = f->block_rows + (uint64_t)row * K31_LATENT;
	if ( SparkGpuMemcpy(f->block_destinations, destinations, sizeof(destinations), SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	state->multiprocessors = (uint32_t)multiprocessors;
	return SPARK_STATUS_OK;
}

static uint16_t *K31Row(const DsparkPagedSequence *sequence, uint32_t layer, uint64_t position)
{
	uint16_t *page = sequence->pages[position / K31_PAGE_POSITIONS];
	return page + ((uint64_t)layer * K31_PAGE_POSITIONS + position % K31_PAGE_POSITIONS) * K31_LATENT;
}

static SparkStatus K31WriteRows(K31State *state, DsparkPagedSequence *sequence, uint64_t first, uint32_t rows)
{
	uint16_t *host_rows[K31_LAYERS * K31_INGEST_ROWS];
	SparkStatus status = SPARK_STATUS_OK;
	for ( uint32_t layer = 0u; layer < K31_LAYERS; layer++ )
		for ( uint32_t row = 0u; row < rows; row++ )
			host_rows[(uint64_t)layer * rows + row] = K31Row(sequence, layer, first + row);
	if ( SparkGpuMemcpyAsync(state->ingest_rows, host_rows, (size_t)K31_LAYERS * rows * sizeof(uint16_t *), SPARK_GPU_COPY_HOST_TO_DEVICE,
			state->stream) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	status = DsparkGemm(state->context_proj, state->taps, state->features_raw, rows, K31_TAP_WIDTH, K31_HIDDEN, state->multiprocessors, state->stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((DsparkRmsKernel<false>), rows, DSPARK_THREADS, 0, state->stream, state->features_raw, state->context_norm, state->features, K31_HIDDEN,
		state->epsilon);
	for ( uint32_t layer = 0u; layer < K31_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		status = DsparkGemm(state->layers[layer].kv_a, state->features, state->latent_raw, rows, K31_HIDDEN, K31_LATENT, state->multiprocessors,
			state->stream);
		if ( status == SPARK_STATUS_OK )
			LM_LAUNCH((K31LatentRowsKernel), rows, DSPARK_THREADS, 0, state->stream, state->latent_raw, state->layers[layer].kv_a_norm,
				state->rope_inverse, state->rope_gain, state->positions, state->ingest_rows + (uint64_t)layer * rows, state->epsilon);
	}
	return DsparkSettle(state->stream, status);
}

static SparkStatus K31Ingest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	K31State *state = (K31State *)opaque;
	return state->paged.IngestFrame<K31_TAPS, K31_HIDDEN, K31_INGEST_ROWS>(frame, state->host_taps, state->host_positions, state->taps,
		state->positions, state->stream, &state->frames, &state->rows,
		[](DsparkPagedSequence *) { return SPARK_STATUS_OK; },
		[&](DsparkPagedSequence *sequence, uint64_t first, uint32_t part) { return K31WriteRows(state, sequence, first, part); });
}

static SparkStatus K31LayerForward(K31State *state, uint32_t layer, uint32_t splits)
{
	K31Forward *f = &state->forward;
	const K31Layer *w = &state->layers[layer];
	const uint32_t *positions = (const uint32_t *)((const uint8_t *)f->draft + offsetof(K31Draft, positions));
	const SparkGpuStream stream = state->stream;
	SparkStatus status;
	LM_LAUNCH((K31ResidualNormKernel), K31_BLOCK, DSPARK_THREADS, 0, stream, f->residual, layer == 0u ? (const uint16_t *)0 : f->mlp, w->input_norm,
		f->x, state->epsilon);
	status = DsparkGemm(w->q_a, f->x, f->q_a_raw, K31_BLOCK, K31_HIDDEN, K31_QUERY_RANK, state->multiprocessors, stream);
	if ( status == SPARK_STATUS_OK )
		status = DsparkGemm(w->kv_a, f->x, f->latent_raw, K31_BLOCK, K31_HIDDEN, K31_LATENT, state->multiprocessors, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((DsparkRmsKernel<false>), K31_BLOCK, DSPARK_THREADS, 0, stream, f->q_a_raw, w->q_a_norm, f->q_a, K31_QUERY_RANK, state->epsilon);
	status = DsparkGemm(w->q_b, f->q_a, f->q, K31_BLOCK, K31_QUERY_RANK, K31_HEADS * K31_QK, state->multiprocessors, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((K31QueryKernel), K31_HEADS * 4u, 64u, 0, stream, f->q, w->kv_b, state->rope_inverse, state->rope_gain, positions, f->queries,
		f->queries_f32);
	if ( LM_LAUNCH_DEPENDENT((K31LatentRowsKernel), K31_BLOCK, DSPARK_THREADS, 0, stream, f->latent_raw, w->kv_a_norm, state->rope_inverse,
			state->rope_gain, positions, f->block_destinations, state->epsilon) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( splits != 0u &&
		LM_LAUNCH_DEPENDENT((K31ContextAttentionKernel), dim3(K31_ATTN_TILES, splits), K31_ATTN_THREADS, K31_ATTN_SHARED_BYTES, stream, f->queries,
			f->pages, (uint64_t)layer * K31_PAGE_POSITIONS * K31_LATENT, f->draft, state->attention_scale, f->partials, f->stats) !=
			SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( LM_LAUNCH_DEPENDENT((K31AttentionMergeKernel), K31_QUERIES, K31_MERGE_THREADS, 0, stream, f->partials, f->stats, f->draft, f->queries_f32,
			f->block_rows, state->attention_scale, f->latent) != SPARK_GPU_SUCCESS ||
		LM_LAUNCH_DEPENDENT((K31ValueKernel), K31_HEADS * 4u, 128u, 0, stream, f->latent, w->kv_b, f->values) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	status = DsparkGemm(w->o_proj, f->values, f->attention, K31_BLOCK, K31_HEADS * K31_VALUE, K31_HIDDEN, state->multiprocessors, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((K31ResidualNormKernel), K31_BLOCK, DSPARK_THREADS, 0, stream, f->residual, f->attention, w->post_norm, f->x, state->epsilon);
	status = DsparkGemm(w->gate, f->x, f->gate, K31_BLOCK, K31_HIDDEN, K31_MLP, state->multiprocessors, stream);
	if ( status == SPARK_STATUS_OK )
		status = DsparkGemm(w->up, f->x, f->up, K31_BLOCK, K31_HIDDEN, K31_MLP, state->multiprocessors, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((DsparkSwigluKernel), state->multiprocessors, DSPARK_THREADS, 0, stream, f->gate, f->up, f->act, K31_BLOCK * K31_MLP);
	return DsparkGemm(w->down, f->act, f->mlp, K31_BLOCK, K31_MLP, K31_HIDDEN, state->multiprocessors, stream);
}

static SparkStatus K31ForwardReplay(K31State *state, uint32_t layers, uint32_t splits)
{
	K31Forward *f = &state->forward;
	SparkStatus status = SPARK_STATUS_OK;
	LM_LAUNCH((DsparkEmbedF32Kernel<K31_HIDDEN, K31Draft>), K31_BLOCK, DSPARK_THREADS, 0, state->stream, state->embed, f->draft, f->residual);
	for ( uint32_t layer = 0u; layer < layers && status == SPARK_STATUS_OK; layer++ )
		status = K31LayerForward(state, layer, splits);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( layers == K31_LAYERS )
		LM_LAUNCH((K31ResidualNormKernel), K31_BLOCK, DSPARK_THREADS, 0, state->stream, f->residual, f->mlp, state->final_norm, f->y, state->epsilon);
	return SPARK_STATUS_OK;
}

static SparkStatus K31PrepareForward(K31State *state, const DsparkPagedSequence *sequence, uint32_t anchor_token, uint64_t position, uint32_t layers,
	const SparkRowSampling *sampling, uint32_t drafts, uint32_t *splits_out)
{
	K31Forward *f = &state->forward;
	const uint32_t pages = sequence != 0 ? sequence->page_count : 0u;
	uint32_t held = 0u, splits = 0u, split_rows = 0u;
	SparkStatus status;
	if ( anchor_token >= K31_VOCAB || position + K31_BLOCK > (uint64_t)K31_PAGES_MAX * K31_PAGE_POSITIONS || layers == 0u || layers > K31_LAYERS )
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = K31PagedPool::BuildRuns(sequence, position, f->host_draft, &held);
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( uint32_t row = 0u; row < K31_BLOCK; row++ )
		f->host_draft->positions[row] = (uint32_t)(position + row);
	for ( uint32_t page = 0u; page < pages; page++ )
		f->host_pages[page] = sequence->pages[page];
	if ( held != 0u )
	{
		splits = (held + K31_SPLIT_ROWS_MIN - 1u) / K31_SPLIT_ROWS_MIN;
		splits = splits > K31_SPLITS_MAX ? K31_SPLITS_MAX : splits;
		split_rows = ((held + splits - 1u) / splits + K31_ATTN_TILE - 1u) / K31_ATTN_TILE * K31_ATTN_TILE;
		splits = (held + split_rows - 1u) / split_rows;
	}
	f->host_draft->anchor = anchor_token;
	f->host_draft->mask = state->mask_token;
	f->host_draft->position = (uint32_t)position;
	f->host_draft->held = held;
	f->host_draft->splits = splits;
	f->host_draft->split_rows = split_rows;
	f->host_draft->drafts = drafts;
	if ( sampling != 0 )
	{
		f->host_draft->seed = sampling->seed;
		f->host_draft->top_k = sampling->top_k;
		f->host_draft->inverse_temperature = sampling->inverse_temperature;
		f->host_draft->nucleus = sampling->nucleus;
		f->host_draft->scale = sampling->inverse_temperature > 0.0f ? sampling->inverse_temperature : 1.0f;
	}
	else
	{
		f->host_draft->seed = 0u;
		f->host_draft->top_k = 0u;
		f->host_draft->inverse_temperature = 0.0f;
		f->host_draft->nucleus = 0.0f;
		f->host_draft->scale = 1.0f;
	}
	if ( SparkGpuMemcpyAsync(f->draft, f->host_draft, sizeof(K31Draft), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS ||
		(pages != 0u && SparkGpuMemcpyAsync(f->pages, f->host_pages, (size_t)pages * sizeof(*f->host_pages), SPARK_GPU_COPY_HOST_TO_DEVICE,
			state->stream) != SPARK_GPU_SUCCESS) )
		return SPARK_STATUS_IO_ERROR;
	*splits_out = splits;
	return SPARK_STATUS_OK;
}

static SparkStatus K31ForwardRun(K31State *state, const DsparkPagedSequence *sequence, uint32_t anchor_token, uint64_t position, uint32_t layers)
{
	uint32_t splits = 0u;
	SparkStatus status = K31PrepareForward(state, sequence, anchor_token, position, layers, 0, K31_BLOCK, &splits);
	if ( status == SPARK_STATUS_OK )
		status = K31ForwardReplay(state, layers, splits);
	if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_INTERNAL_ERROR;
	return status;
}

static SparkStatus K31Close(void *opaque)
{
	K31State *state = (K31State *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state == 0 )
		return SPARK_STATUS_OK;
	if ( state->stream != 0 )
		(void)SparkGpuStreamSynchronize(state->stream);
	SparkDrafterGraphCacheDestroy(&state->graphs);
	state->paged.FreeAll();
	{
		K31Forward *f = &state->forward;
		void *buffers[] = {
			f->draft, f->pages, f->block_destinations, f->residual, f->x, f->q_a_raw, f->q_a, f->q, f->latent_raw, f->block_rows, f->queries,
			f->queries_f32, f->partials, f->stats, f->latent, f->values, f->attention, f->gate, f->up, f->act, f->mlp, f->y, f->logits,
			f->candidates, f->markov_row, f->picks, state->taps, state->features_raw, state->features, state->latent_raw, state->positions,
			state->ingest_rows, state->rope_inverse,
		};
		void *host_buffers[] = { f->host_draft, f->host_picks, (void *)f->host_pages, state->host_taps, state->host_positions };
		DsparkFreeDeviceBuffers(buffers, (uint32_t)(sizeof(buffers) / sizeof(buffers[0])));
		SparkDrafterTopkBf16Free(&f->topk_ws);
		DsparkFreeHostBuffers(host_buffers, (uint32_t)(sizeof(host_buffers) / sizeof(host_buffers[0])));
	}
	if ( state->stream != 0 )
		(void)SparkGpuStreamDestroy(state->stream);
	if ( state->pack != 0 )
		status = SparkWeightdLazyPackDestroy(state->pack);
	free(state);
	return status;
}

static SparkStatus K31Open(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	return DsparkDrafterOpen<K31State>(lane, K31_PAGE_BYTES, "lane %s has a context cap below one context page",
		"lane %s: the pack %s is not a K3.1 Pro DSpark pack this module reads", false, state_out, bind, K31Close,
		[&](K31State *state) -> SparkStatus {
			SparkStatus status;
			state->paged.context_bytes = lane->context_bytes;
			status = K31Weights(state, lane->pack_path);
			if ( status == SPARK_STATUS_OK )
				status = K31Scratch(state);
			return status;
		},
		[](const K31State *) { return 0u; });
}

static SparkStatus K31Propose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	K31State *state = (K31State *)opaque;
	DsparkPagedSequence *sequence = state->paged.Find(request->sequence_id, 0u);
	const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
	const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
	const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
	uint32_t drafts = requested;
	SparkStatus status;
	*count = 0u;
	if ( limit == 0u )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( K31PagedPool::Covers(sequence, request->position) == 0u )
		return SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS;
	if ( sequence != 0 && K31PagedPool::Complete(sequence, request->position) == 0u )
	{
		status = state->paged.FillFromDonors(sequence, request,
			[&](DsparkPagedSequence *dst, const DsparkPagedSequence *src, uint64_t at, uint64_t end) -> SparkStatus {
				while ( at < end )
				{
					const uint64_t page_end = (at / K31_PAGE_POSITIONS + 1u) * K31_PAGE_POSITIONS, stop = end < page_end ? end : page_end;
					for ( uint32_t layer = 0u; layer < K31_LAYERS; layer++ )
						if ( SparkGpuMemcpyAsync(K31Row(dst, layer, at), K31Row(src, layer, at), (size_t)(stop - at) * K31_ROW_BYTES,
								SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
							return SPARK_STATUS_IO_ERROR;
					at = stop;
				}
				return SPARK_STATUS_OK;
			});
		if ( status != SPARK_STATUS_OK )
			return SPARK_STATUS_IO_ERROR;
	}
	if ( sequence != 0 && K31PagedPool::TokensMatch(sequence, request) == 0u )
		return SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT;
	drafts = drafts < K31_BLOCK ? drafts : K31_BLOCK;
	drafts = drafts < limit - 1u ? drafts : limit - 1u;
	{
		const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
		const uint32_t branch_steps = tree != 0u ? (drafts < K31_BRANCH_LEN ? drafts : K31_BRANCH_LEN) : 0u;
		const uint32_t tree_enabled = tree != 0u && branch_steps >= 2u ? 1u : 0u;
		uint32_t splits = 0u;
		status = K31PrepareForward(state, sequence, request->anchor_token_id, request->position, K31_LAYERS, &request->sampling, drafts, &splits);
		if ( status == SPARK_STATUS_OK )
		{
			const uint64_t key = ((uint64_t)splits << 8u) | ((uint64_t)tree << 4u) | (uint64_t)branch_steps;
			status = SparkDrafterGraphRun(&state->graphs, state->stream, key, [&]() -> SparkStatus {
				SparkStatus inner = K31ForwardReplay(state, K31_LAYERS, splits);
				if ( inner == SPARK_STATUS_OK )
					inner = DsparkLaunchMarkovHeads<K31_BLOCK, K31_HIDDEN, K31_VOCAB, K31_VOCAB, K31_MARKOV, K31_CANDIDATES, K31_SAMPLED,
						K31_BRANCHES, false, K31Draft, K31Picks>(state->head, state->forward.y, state->forward.logits, &state->forward.topk_ws,
						state->forward.candidates, state->forward.markov_row, state->markov_previous, state->markov_next, (const int64_t *)0,
						state->confidence_weight, state->confidence_bias, state->forward.draft, state->forward.picks, state->forward.host_picks,
						branch_steps, tree_enabled, state->multiprocessors, state->stream);
				return inner;
			});
		}
		status = DsparkSettle(state->stream, status);
		if ( status != SPARK_STATUS_OK )
			return status;
		*count = DsparkPackMarkovTree<K31_BLOCK, K31_BRANCHES>(request->anchor_token_id, source, state->forward.host_picks, tree, drafts, requested,
			limit, nodes);
	}
	return SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS;
}

static const SparkDrafterModuleInterface k31_interface = {
	SPARK_DRAFTER_MODULE_ABI_VERSION,
	(uint32_t)sizeof(SparkDrafterModuleInterface),
	"k31-dspark",
	K31Open,
	K31Ingest,
	K31Propose,
	K31Close,
};

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &k31_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_k31_dspark_context_rows(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint64_t first, uint32_t count, uint16_t *host)
{
	K31State *state = (K31State *)opaque;
	DsparkPagedSequence *sequence = state != 0 ? state->paged.Find(sequence_id, 0u) : 0;
	if ( sequence == 0 || layer >= K31_LAYERS || first + count > sequence->held_end )
		return SPARK_STATUS_NOT_FOUND;
	for ( uint32_t row = 0u; row < count; row++ )
		if ( DsparkReadback(host + (uint64_t)row * K31_LATENT, K31Row(sequence, layer, first + row), K31_LATENT * sizeof(uint16_t)) !=
			SPARK_STATUS_OK )
			return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_k31_dspark_block_rows(void *opaque, uint64_t sequence_id, uint32_t anchor_token,
	uint64_t position, uint16_t *host, uint64_t *elapsed_ns)
{
	K31State *state = (K31State *)opaque;
	const DsparkPagedSequence *sequence = state != 0 ? state->paged.Find(sequence_id, 0u) : 0;
	const uint64_t started = SparkMonotonicNs();
	SparkStatus status;
	if ( state == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = K31ForwardRun(state, sequence, anchor_token, position, K31_LAYERS);
	status = DsparkSettle(state->stream, status);
	if ( elapsed_ns != 0 )
		*elapsed_ns = SparkMonotonicNs() - started;
	if ( status == SPARK_STATUS_OK && host != 0 )
		status = DsparkReadback(host, state->forward.y, (size_t)K31_BLOCK * K31_HIDDEN * sizeof(uint16_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_k31_dspark_block_trace(void *opaque, uint64_t sequence_id, uint32_t anchor_token,
	uint64_t position, uint32_t layers, float *queries, float *latent, uint16_t *block)
{
	K31State *state = (K31State *)opaque;
	const DsparkPagedSequence *sequence = state != 0 ? state->paged.Find(sequence_id, 0u) : 0;
	SparkStatus status;
	if ( state == 0 || queries == 0 || latent == 0 || block == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = K31ForwardRun(state, sequence, anchor_token, position, layers);
	status = DsparkSettle(state->stream, status);
	if ( status == SPARK_STATUS_OK )
		status = DsparkReadback(queries, state->forward.queries_f32, (size_t)K31_QUERIES * K31_LATENT * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = DsparkReadback(latent, state->forward.latent, (size_t)K31_QUERIES * K31_KV_RANK * sizeof(float));
	if ( status == SPARK_STATUS_OK )
		status = DsparkReadback(block, state->forward.block_rows, (size_t)K31_BLOCK * K31_LATENT * sizeof(uint16_t));
	return status;
}
