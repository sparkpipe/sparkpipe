#include "inference/kernels/metal/common.metal"

static_assert(SPARK_INDEXER_TOPK_BUCKETS == (1u << SPARK_INDEXER_TOPK_RADIX_BITS), "one bucket per radix digit");
static_assert(SPARK_INDEXER_TOPK_THREADS <= 32u * 32u, "one scan slot per simdgroup");

#define SPARK_INDEXER_TOPK_PREFIX 0u
#define SPARK_INDEXER_TOPK_MASK 1u
#define SPARK_INDEXER_TOPK_NEED 2u
#define SPARK_INDEXER_TOPK_EMITTED 3u
#define SPARK_INDEXER_TOPK_SEEN 4u
#define SPARK_INDEXER_TOPK_STATE 5u

static inline bool SparkIndexerSlot(constant SparkIndexerParams &p, device const uint *page_table, uint sequence, uint position,
	device atomic_uint *error, uint row, thread ulong &slot)
{
	const uint index = position / p.page_slots;
	const uint page = index < p.pages_per_sequence ? page_table[(ulong)sequence * p.pages_per_sequence + index] : p.pool_pages;
	if ( page >= p.pool_pages )
	{
		atomic_store_explicit(error, 1u + row, memory_order_relaxed);
		return false;
	}
	slot = (ulong)page * p.page_pitch + (ulong)(position % p.page_slots) * p.slot_pitch;
	return true;
}

static inline float SparkIndexerWeight(constant SparkIndexerParams &p, device const ushort *weights, uint row, uint head)
{
	const ulong at = (ulong)row * p.weight_stride + head;
	return p.weight_f32 != 0u ? ((device const float *)weights)[at] : SparkMathBf16ToFloat(weights[at]);
}

static inline uint SparkIndexerVisible(constant SparkIndexerParams &p, uint position)
{
	return (position + 1u) / max(p.key_ratio, 1u);
}

kernel void spark_indexer_store_keys(constant SparkIndexerParams &p [[buffer(0)]], device const ushort *keys [[buffer(1)]],
	device const float *norm_weight [[buffer(2)]], device const float *norm_bias [[buffer(3)]], device const uint *sequence_of_row [[buffer(4)]],
	device const uint *rope_positions [[buffer(5)]], device const uint *store_positions [[buffer(6)]], device const uint *page_table [[buffer(7)]],
	device ushort *pool [[buffer(8)]], device atomic_uint *error [[buffer(9)]], device const ushort *gates [[buffer(10)]],
	uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	threadgroup float normed[SPARK_INDEXER_KERNEL_MAX_DIM];
	if ( row >= p.rows || store_positions[row] == SPARK_INDEXER_KERNEL_NONE )
		return;
	const uint position = store_positions[row];
	device const ushort *x = keys + (ulong)row * p.key_stride;
	float total = 0.0f, total_squared = 0.0f;
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		const float value = SparkMathBf16ToFloat(x[i]);
		total += value;
		total_squared += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	total_squared = SparkGroupSum(total_squared, scratch, lane, simd, simds);
	const float count = float(p.head_dimension);
	const float mean = p.key_norm == SPARK_INDEXER_NORM_KERNEL_LAYER ? total / count : 0.0f;
	const float inverse = p.key_norm == SPARK_INDEXER_NORM_KERNEL_LAYER ? SparkMathLayerNormInverse(total, total_squared, count, p.key_norm_epsilon) :
		p.key_norm == SPARK_INDEXER_NORM_KERNEL_RMS ? SparkMathRmsScale(total_squared, count, p.key_norm_epsilon) : 1.0f;
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		const float value = SparkMathBf16ToFloat(x[i]);
		const float weight = p.has_weight != 0u ? norm_weight[i] : 1.0f, bias = p.has_bias != 0u ? norm_bias[i] : 0.0f;
		normed[i] = p.key_norm == SPARK_INDEXER_NORM_KERNEL_LAYER ? SparkMathRoundBf16(SparkMathLayerNormValue(value, mean, inverse, weight, bias)) :
			p.key_norm == SPARK_INDEXER_NORM_KERNEL_RMS ? SparkMathRoundBf16(value * inverse * weight) : value;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	ulong offset = 0ul;
	if ( !SparkIndexerSlot(p, page_table, sequence_of_row[row], position, error, row, offset) )
		return;
	device ushort *slot = pool + offset;
	const float angle_position = float(rope_positions[row]);
	const uint half_rope = p.rope_dimension / 2u;
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		float value = normed[i];
		if ( i < p.rope_dimension )
		{
			const bool low = p.rope_interleaved != 0u ? i % 2u == 0u : i < half_rope;
			const uint mate = p.rope_interleaved != 0u ? (low ? i + 1u : i - 1u) : (low ? i + half_rope : i - half_rope);
			const float angle = angle_position * p.inverse_frequency[p.rope_interleaved != 0u ? i / 2u : i % half_rope];
			const float cosine = SparkMathRopeCos(angle, p.rope_scale), sine = SparkMathRopeSin(angle, p.rope_scale);
			const float c = p.rope_rounding == SPARK_ROPE_ROUNDING_BF16_TRIG ? SparkMathRoundBf16(cosine) : cosine;
			const float s = p.rope_rounding == SPARK_ROPE_ROUNDING_BF16_TRIG ? SparkMathRoundBf16(sine) : sine;
			if ( p.rope_rounding == SPARK_ROPE_ROUNDING_TERMS )
				value = SparkMathRopeRoundedTerm(value, low ? -normed[mate] : normed[mate], c, s);
			else
				value = low ? SparkMathRopeLow(value, normed[mate], c, s) : SparkMathRopeHigh(normed[mate], value, c, s);
		}
		slot[i] = SparkMathFloatToBf16(value);
		if ( p.has_gate != 0u )
			slot[p.head_dimension + i] = gates[(ulong)row * p.gate_stride + i];
	}
}

kernel void spark_indexer_score(constant SparkIndexerParams &p [[buffer(0)]], device const ushort *query [[buffer(1)]],
	device const ushort *head_weight [[buffer(2)]], device const ushort *pool [[buffer(3)]], device const uint *page_table [[buffer(4)]],
	device const uint *sequence_of_row [[buffer(5)]], device const uint *row_positions [[buffer(6)]], device float *scores [[buffer(7)]],
	device atomic_uint *error [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup ushort shared_query[SPARK_INDEXER_KERNEL_MAX_QUERY];
	threadgroup float shared_weight[SPARK_INDEXER_KERNEL_MAX_HEADS];
	threadgroup float shared_key[SPARK_INDEXER_SCORE_SIMDS][SPARK_INDEXER_KERNEL_MAX_DIM];
	threadgroup float shared_relu[SPARK_INDEXER_SCORE_SIMDS][SPARK_INDEXER_KERNEL_MAX_HEADS];
	const uint row = group.y, first = group.x * SPARK_INDEXER_SCORE_BLOCK_POSITIONS;
	if ( row >= p.rows )
		return;
	const uint width = p.heads * p.head_dimension, tiles = p.heads / 16u, groups = SPARK_INDEXER_KERNEL_HEAD_GROUPS;
	const uint sequence = sequence_of_row[row], bound = SparkIndexerVisible(p, row_positions[row]);
	for ( uint i = tid; i < width; i += SPARK_INDEXER_SCORE_THREADS )
		shared_query[i] = query[(ulong)row * p.query_stride + i];
	for ( uint h = tid; h < p.heads; h += SPARK_INDEXER_SCORE_THREADS )
		shared_weight[h] = SparkIndexerWeight(p, head_weight, row, h);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint offset = simd; offset < SPARK_INDEXER_SCORE_BLOCK_POSITIONS; offset += SPARK_INDEXER_SCORE_SIMDS )
	{
		const uint position = first + offset;
		if ( position >= p.score_stride )
			break;
		float score = -INFINITY;
		ulong slot_offset = 0ul;
		if ( position < bound && SparkIndexerSlot(p, page_table, sequence, position, error, row, slot_offset) )
		{
			device const ushort *slot = pool + slot_offset;
			simdgroup_barrier(mem_flags::mem_threadgroup);
			for ( uint d = lane; d < p.head_dimension; d += 32u )
				shared_key[simd][d] = SparkMathBf16ToFloat(slot[d]);
			simdgroup_barrier(mem_flags::mem_threadgroup);
			for ( uint h = lane; h < p.heads; h += 32u )
			{
				threadgroup const ushort *q = shared_query + h * p.head_dimension;
				float dot = 0.0f;
				for ( uint d = 0u; d < p.head_dimension; d++ )
					dot = fma(SparkMathBf16ToFloat(q[d]), shared_key[simd][d], dot);
				shared_relu[simd][h] = fmax(dot, 0.0f);
			}
			simdgroup_barrier(mem_flags::mem_threadgroup);
			if ( p.weight_scale != 0.0f )
			{
				float total = 0.0f;
				for ( uint h = 0u; h < p.heads; h++ )
					total += fmax(shared_relu[simd][h] * p.score_scale, 0.0f) * shared_weight[h] * p.weight_scale;
				score = total;
			}
			else if ( p.heads_in_order != 0u )
			{
				float total = 0.0f;
				for ( uint h = 0u; h < p.heads; h++ )
					total += shared_weight[h] * shared_relu[simd][h];
				score = total * p.score_scale;
			}
			const uint g = lane % groups;
			float sum = shared_weight[g] * shared_relu[simd][g];
			sum = fma(shared_weight[g + groups], shared_relu[simd][g + groups], sum);
			for ( uint t = 1u; t < tiles; t++ )
			{
				sum = fma(shared_weight[16u * t + g], shared_relu[simd][16u * t + g], sum);
				sum = fma(shared_weight[16u * t + g + groups], shared_relu[simd][16u * t + g + groups], sum);
			}
			sum += simd_shuffle_xor(sum, ushort(1));
			sum += simd_shuffle_xor(sum, ushort(2));
			sum += simd_shuffle_xor(sum, ushort(4));
			if ( p.weight_scale == 0.0f && p.heads_in_order == 0u )
				score = sum * p.score_scale;
		}
		if ( lane == 0u )
			scores[(ulong)row * p.score_stride + position] = score;
	}
}

kernel void spark_indexer_score_pooled(constant SparkIndexerParams &p [[buffer(0)]], device const ushort *query [[buffer(1)]],
	device const ushort *head_weight [[buffer(2)]], device const ushort *pool [[buffer(3)]], device const uint *page_table [[buffer(4)]],
	device const uint *sequence_of_row [[buffer(5)]], device const uint *row_positions [[buffer(6)]], device float *scores [[buffer(7)]],
	device atomic_uint *error [[buffer(8)]], device const float *ape [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float pool_key[SPARK_INDEXER_KERNEL_MAX_DIM];
	threadgroup float scratch[32];
	threadgroup ulong slots[SPARK_INDEXER_KERNEL_MAX_RATIO];
	threadgroup uint failed;
	const uint row = group.y, pooled = group.x, ratio = max(p.key_ratio, 1u);
	if ( row >= p.rows || pooled >= p.score_stride )
		return;
	device float *score = scores + (ulong)row * p.score_stride + pooled;
	if ( pooled >= SparkIndexerVisible(p, row_positions[row]) )
	{
		if ( tid == 0u )
			*score = -INFINITY;
		return;
	}
	if ( tid == 0u )
	{
		failed = 0u;
		for ( uint i = 0u; i < ratio; i++ )
			if ( !SparkIndexerSlot(p, page_table, sequence_of_row[row], pooled * ratio + i, error, row, slots[i]) )
				failed = 1u;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( failed != 0u )
	{
		if ( tid == 0u )
			*score = -INFINITY;
		return;
	}
	for ( uint d = tid; d < p.head_dimension; d += threads )
	{
		float logits[SPARK_INDEXER_KERNEL_MAX_RATIO];
		float maximum = -INFINITY, sum = 0.0f, key = 0.0f;
		for ( uint i = 0u; i < ratio; i++ )
		{
			logits[i] = SparkMathBf16ToFloat(pool[slots[i] + p.head_dimension + d]) + ape[i * p.head_dimension + d];
			maximum = fmax(maximum, logits[i]);
		}
		for ( uint i = 0u; i < ratio; i++ )
			if ( logits[i] != -INFINITY )
				sum += SPARK_MATH_EXP_FAST(logits[i] - maximum);
		for ( uint i = 0u; i < ratio; i++ )
		{
			const float mix = logits[i] == -INFINITY ? 0.0f : SPARK_MATH_EXP_FAST(logits[i] - maximum) / fmax(sum, 1.0e-20f);
			if ( mix != 0.0f )
				key += mix * SparkMathBf16ToFloat(pool[slots[i] + d]);
		}
		pool_key[d] = key;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float total = 0.0f;
	for ( uint h = 0u; h < p.heads; h++ )
	{
		float dot = 0.0f;
		for ( uint d = tid; d < p.head_dimension; d += threads )
			dot += SparkMathBf16ToFloat(query[(ulong)row * p.query_stride + h * p.head_dimension + d]) * pool_key[d];
		dot = SparkGroupSum(dot, scratch, lane, simd, simds) * p.score_scale;
		if ( dot < 0.0f )
			dot = 0.0f;
		total += dot * SparkIndexerWeight(p, head_weight, row, h) * p.weight_scale;
	}
	if ( tid == 0u )
		*score = total;
}

kernel void spark_indexer_expand_pools(constant SparkIndexerParams &p [[buffer(0)]], device const uint *selected_pools [[buffer(1)]],
	device const uint *row_positions [[buffer(2)]], device uint *selected [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]])
{
	if ( row >= p.rows )
		return;
	const uint ratio = max(p.key_ratio, 1u), context = row_positions[row] + 1u, pooled = context - context % ratio, spread = p.topk * ratio;
	for ( uint index = tid; index < p.expand_width; index += threads )
	{
		uint position = SPARK_INDEXER_KERNEL_NONE;
		if ( index < spread )
		{
			const uint chosen = selected_pools[(ulong)row * p.topk + index / ratio];
			position = chosen == SPARK_INDEXER_KERNEL_NONE ? SPARK_INDEXER_KERNEL_NONE : chosen * ratio + index % ratio;
			if ( position >= pooled )
				position = SPARK_INDEXER_KERNEL_NONE;
		}
		else if ( pooled + index - spread < context )
			position = pooled + index - spread;
		selected[(ulong)row * p.expand_width + index] = position;
	}
}

kernel void spark_indexer_topk(constant SparkIndexerParams &p [[buffer(0)]], device const float *scores [[buffer(1)]],
	device uint *selected [[buffer(2)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup atomic_uint histogram[SPARK_INDEXER_TOPK_BUCKETS];
	threadgroup uint state[SPARK_INDEXER_TOPK_STATE];
	threadgroup uint scan_greater[32];
	threadgroup uint scan_equal[32];
	const uint n = p.score_stride, k = p.topk, threads = SPARK_INDEXER_TOPK_THREADS;
	if ( row >= p.rows )
		return;
	device const float *row_scores = scores + (ulong)row * n;
	device uint *out = selected + (ulong)row * k;
	if ( n <= k )
	{
		for ( uint i = tid; i < k; i += threads )
			out[i] = i < n ? i : SPARK_INDEXER_KERNEL_NONE;
		return;
	}
	if ( tid == 0u )
	{
		state[SPARK_INDEXER_TOPK_PREFIX] = 0u;
		state[SPARK_INDEXER_TOPK_MASK] = 0u;
		state[SPARK_INDEXER_TOPK_NEED] = k;
		state[SPARK_INDEXER_TOPK_EMITTED] = 0u;
		state[SPARK_INDEXER_TOPK_SEEN] = 0u;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint pass = 0u; pass < 32u / SPARK_INDEXER_TOPK_RADIX_BITS; pass++ )
	{
		const uint shift = 32u - SPARK_INDEXER_TOPK_RADIX_BITS * (pass + 1u);
		for ( uint i = tid; i < SPARK_INDEXER_TOPK_BUCKETS; i += threads )
			atomic_store_explicit(&histogram[i], 0u, memory_order_relaxed);
		threadgroup_barrier(mem_flags::mem_threadgroup);
		const uint prefix = state[SPARK_INDEXER_TOPK_PREFIX], mask = state[SPARK_INDEXER_TOPK_MASK];
		for ( uint i = tid; i < n; i += threads )
		{
			const uint key = SparkMathTopkKey(row_scores[i]);
			if ( (key & mask) == prefix )
				atomic_fetch_add_explicit(&histogram[(key >> shift) & (SPARK_INDEXER_TOPK_BUCKETS - 1u)], 1u, memory_order_relaxed);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( tid == 0u )
		{
			uint running = 0u, bucket = SPARK_INDEXER_TOPK_BUCKETS - 1u;
			while ( bucket > 0u && running + atomic_load_explicit(&histogram[bucket], memory_order_relaxed) < state[SPARK_INDEXER_TOPK_NEED] )
			{
				running += atomic_load_explicit(&histogram[bucket], memory_order_relaxed);
				bucket--;
			}
			state[SPARK_INDEXER_TOPK_PREFIX] |= bucket << shift;
			state[SPARK_INDEXER_TOPK_MASK] |= (SPARK_INDEXER_TOPK_BUCKETS - 1u) << shift;
			state[SPARK_INDEXER_TOPK_NEED] -= running;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	const uint threshold = state[SPARK_INDEXER_TOPK_PREFIX];
	for ( uint start = 0u; start < n; start += threads )
	{
		const uint index = start + tid;
		const uint key = index < n ? SparkMathTopkKey(row_scores[index]) : 0u;
		const uint greater = index < n && key > threshold ? 1u : 0u, equal = index < n && key == threshold ? 1u : 0u;
		uint greater_before = simd_prefix_exclusive_sum(greater), equal_before = simd_prefix_exclusive_sum(equal);
		const uint greater_simd = simd_sum(greater), equal_simd = simd_sum(equal);
		if ( lane == 0u )
		{
			scan_greater[simd] = greater_simd;
			scan_equal[simd] = equal_simd;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		uint greater_total = 0u, equal_total = 0u;
		for ( uint other = 0u; other < simds; other++ )
		{
			if ( other < simd )
			{
				greater_before += scan_greater[other];
				equal_before += scan_equal[other];
			}
			greater_total += scan_greater[other];
			equal_total += scan_equal[other];
		}
		const uint need = state[SPARK_INDEXER_TOPK_NEED], seen = state[SPARK_INDEXER_TOPK_SEEN], emitted = state[SPARK_INDEXER_TOPK_EMITTED];
		const uint remaining = need > seen ? need - seen : 0u;
		if ( greater != 0u || (equal != 0u && equal_before < remaining) )
			out[emitted + greater_before + min(equal_before, remaining)] = index;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( tid == 0u )
		{
			state[SPARK_INDEXER_TOPK_EMITTED] = emitted + greater_total + min(equal_total, remaining);
			state[SPARK_INDEXER_TOPK_SEEN] = seen + equal_total;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
}

kernel void spark_indexer_pool_keys(constant SparkIndexerParams &p [[buffer(0)]], device ushort *window [[buffer(1)]],
	device const uint *state_index [[buffer(2)]], device const uint *row_begin [[buffer(3)]], device const ushort *keys [[buffer(4)]],
	device ushort *pooled [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint d = group.x * group_size.x + tid, sequence = group.y, ratio = max(p.key_ratio, 1u), span = ratio - 1u;
	if ( d >= p.head_dimension || sequence >= p.sequences || ratio > SPARK_INDEXER_KERNEL_MAX_RATIO )
		return;
	device ushort *slot = window + (ulong)state_index[sequence] * span * p.head_dimension;
	float carried[SPARK_INDEXER_KERNEL_MAX_RATIO];
	for ( uint e = 0u; e < span; e++ )
		carried[e] = SparkMathBf16ToFloat(slot[(ulong)e * p.head_dimension + d]);
	const uint begin = row_begin[sequence], count = row_begin[sequence + 1u] - begin;
	for ( uint i = 0u; i < count; i++ )
	{
		float sum = 0.0f;
		for ( uint k = 0u; k < ratio; k++ )
			sum += i + k >= span ? SparkMathBf16ToFloat(keys[(ulong)(begin + i + k - span) * p.key_stride + d]) : carried[i + k];
		pooled[(ulong)(begin + i) * p.head_dimension + d] = SparkMathFloatToBf16(sum / float(ratio));
	}
	if ( p.commit == 0u )
		return;
	for ( uint e = 0u; e < span; e++ )
	{
		const uint combined = count + e;
		slot[(ulong)e * p.head_dimension + d] = combined >= span ? keys[(ulong)(begin + combined - span) * p.key_stride + d] :
			SparkMathFloatToBf16(carried[combined]);
	}
}
