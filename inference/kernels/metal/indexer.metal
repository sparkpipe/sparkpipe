#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "inference/kernels/metal/common.metal"

using namespace mpp::tensor_ops;

static_assert(SPARK_INDEXER_TOPK_BUCKETS == (1u << SPARK_INDEXER_TOPK_RADIX_BITS), "one bucket per radix digit");
static_assert(SPARK_INDEXER_TOPK_THREADS <= 32u * 32u, "one scan slot per simdgroup");

#define SPARK_INDEXER_TOPK_PREFIX 0u
#define SPARK_INDEXER_TOPK_MASK 1u
#define SPARK_INDEXER_TOPK_NEED 2u
#define SPARK_INDEXER_TOPK_EMITTED 3u
#define SPARK_INDEXER_TOPK_SEEN 4u
#define SPARK_INDEXER_TOPK_STATE 5u
#define SPARK_INDEXER_SCORE_TILE 32u

static inline float2 SparkIndexerGroupSum2(float2 value, threadgroup float2 *scratch, uint lane, uint simd, uint simds)
{
	value = simd_sum(value);
	if ( lane == 0u )
		scratch[simd] = value;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float2 total = lane < simds ? scratch[lane] : float2(0.0f);
	total = simd_sum(total);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return total;
}

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
	threadgroup float2 scratch[32];
	threadgroup float normed[SPARK_INDEXER_KERNEL_MAX_DIM];
	if ( row >= p.rows || store_positions[row] == SPARK_INDEXER_KERNEL_NONE )
		return;
	const uint position = store_positions[row], D = p.head_dimension;
	device const ushort *x = keys + (ulong)row * p.key_stride;
	if ( D <= threads )
	{
		const float value = tid < D ? SparkMathBf16ToFloat(x[tid]) : 0.0f;
		if ( p.key_norm != SPARK_INDEXER_NORM_KERNEL_NONE )
		{
			const float2 totals = SparkIndexerGroupSum2(float2(value, value * value), scratch, lane, simd, simds);
			const float count = float(D);
			const float mean = p.key_norm == SPARK_INDEXER_NORM_KERNEL_LAYER ? totals.x / count : 0.0f;
			const float inverse = p.key_norm == SPARK_INDEXER_NORM_KERNEL_LAYER ? SparkMathLayerNormInverse(totals.x, totals.y, count, p.key_norm_epsilon) :
				SparkMathRmsScale(totals.y, count, p.key_norm_epsilon);
			if ( tid < D )
			{
				const float weight = p.has_weight != 0u ? norm_weight[tid] : 1.0f, bias = p.has_bias != 0u ? norm_bias[tid] : 0.0f;
				normed[tid] = p.key_norm == SPARK_INDEXER_NORM_KERNEL_LAYER ? SparkMathRoundBf16(SparkMathLayerNormValue(value, mean, inverse, weight, bias)) :
					SparkMathRoundBf16(value * inverse * weight);
			}
		}
		else if ( tid < D )
			normed[tid] = value;
	}
	else
	{
		float2 totals = float2(0.0f);
		for ( uint i = tid; i < D; i += threads )
		{
			const float value = SparkMathBf16ToFloat(x[i]);
			totals += float2(value, value * value);
		}
		totals = SparkIndexerGroupSum2(totals, scratch, lane, simd, simds);
		const float count = float(D);
		const float mean = p.key_norm == SPARK_INDEXER_NORM_KERNEL_LAYER ? totals.x / count : 0.0f;
		const float inverse = p.key_norm == SPARK_INDEXER_NORM_KERNEL_LAYER ? SparkMathLayerNormInverse(totals.x, totals.y, count, p.key_norm_epsilon) :
			p.key_norm == SPARK_INDEXER_NORM_KERNEL_RMS ? SparkMathRmsScale(totals.y, count, p.key_norm_epsilon) : 1.0f;
		for ( uint i = tid; i < D; i += threads )
		{
			const float value = SparkMathBf16ToFloat(x[i]);
			const float weight = p.has_weight != 0u ? norm_weight[i] : 1.0f, bias = p.has_bias != 0u ? norm_bias[i] : 0.0f;
			normed[i] = p.key_norm == SPARK_INDEXER_NORM_KERNEL_LAYER ? SparkMathRoundBf16(SparkMathLayerNormValue(value, mean, inverse, weight, bias)) :
				p.key_norm == SPARK_INDEXER_NORM_KERNEL_RMS ? SparkMathRoundBf16(value * inverse * weight) : value;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	ulong offset = 0ul;
	if ( !SparkIndexerSlot(p, page_table, sequence_of_row[row], position, error, row, offset) )
		return;
	device ushort *slot = pool + offset;
	const float angle_position = float(rope_positions[row]);
	const uint half_rope = p.rope_dimension / 2u;
	for ( uint i = tid; i < D; i += threads )
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
			slot[D + i] = gates[(ulong)row * p.gate_stride + i];
	}
}

kernel void spark_indexer_score(constant SparkIndexerParams &p [[buffer(0)]], device bfloat *query [[buffer(1)]],
	device const ushort *head_weight [[buffer(2)]], device const ushort *pool [[buffer(3)]], device const uint *page_table [[buffer(4)]],
	device const uint *sequence_of_row [[buffer(5)]], device const uint *row_positions [[buffer(6)]], device float *scores [[buffer(7)]],
	device atomic_uint *error [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	const uint row = group.y, first = group.x * SPARK_INDEXER_SCORE_BLOCK_POSITIONS;
	if ( row >= p.rows )
		return;
	const uint sequence = sequence_of_row[row], bound = SparkIndexerVisible(p, row_positions[row]);
	if ( first >= bound )
	{
		for ( uint offset = tid; offset < SPARK_INDEXER_SCORE_BLOCK_POSITIONS; offset += SPARK_INDEXER_SCORE_THREADS )
			if ( first + offset < p.score_stride )
				scores[(ulong)row * p.score_stride + first + offset] = -INFINITY;
		return;
	}
	threadgroup float shared_weight[SPARK_INDEXER_KERNEL_MAX_HEADS];
	threadgroup union
	{
		struct
		{
			uint4 tile_keys[SPARK_INDEXER_SCORE_TILE * 16u];
			float tile_relu[SPARK_INDEXER_SCORE_TILE][64];
			uint tile_valid[SPARK_INDEXER_SCORE_TILE];
		} fast;
		struct
		{
			ushort shared_query[SPARK_INDEXER_KERNEL_MAX_QUERY];
			float shared_key[SPARK_INDEXER_SCORE_SIMDS][SPARK_INDEXER_KERNEL_MAX_DIM];
			float shared_relu[SPARK_INDEXER_SCORE_SIMDS][SPARK_INDEXER_KERNEL_MAX_HEADS];
		} slow;
	} smem;
	for ( uint h = tid; h < p.heads; h += SPARK_INDEXER_SCORE_THREADS )
		shared_weight[h] = SparkIndexerWeight(p, head_weight, row, h);
	if ( p.head_dimension == 128u && p.heads <= 64u )
	{
		constexpr auto desc = matmul2d_descriptor(64, int(SPARK_INDEXER_SCORE_TILE), 128, false, true, false);
		matmul2d<desc, execution_simdgroups<SPARK_INDEXER_SCORE_SIMDS>> operation;
		tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> q_tensor(query + (ulong)row * p.query_stride,
			dextents<int32_t, 2>(128, int(p.heads)), array<int32_t, 2>{ 1, 128 });
		tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> k_tensor((threadgroup bfloat *)smem.fast.tile_keys,
			dextents<int32_t, 2>(128, int(SPARK_INDEXER_SCORE_TILE)), array<int32_t, 2>{ 1, 128 });
		const uint tiles = p.heads / 16u, groups = SPARK_INDEXER_KERNEL_HEAD_GROUPS;
		for ( uint tile_base = 0u; tile_base < SPARK_INDEXER_SCORE_BLOCK_POSITIONS; tile_base += SPARK_INDEXER_SCORE_TILE )
		{
			const uint pos_base = first + tile_base;
			if ( pos_base >= p.score_stride )
				break;
			if ( pos_base >= bound )
			{
				if ( tid < SPARK_INDEXER_SCORE_TILE && pos_base + tid < p.score_stride )
					scores[(ulong)row * p.score_stride + pos_base + tid] = -INFINITY;
				continue;
			}
			for ( uint j = 0u; j < 2u; j++ )
			{
				const uint elem = tid + j * SPARK_INDEXER_SCORE_THREADS;
				const uint k_local = elem >> 4u, chunk = elem & 15u;
				const uint position = pos_base + k_local;
				ulong slot_offset = 0ul;
				const bool ok = position < bound && position < p.score_stride &&
					SparkIndexerSlot(p, page_table, sequence, position, error, row, slot_offset);
				if ( chunk == 0u )
					smem.fast.tile_valid[k_local] = ok ? 1u : 0u;
				if ( ok )
				{
					device const ushort *slot = pool + slot_offset + chunk * 8u;
					if ( ((ulong)slot & 15ul) == 0ul )
						smem.fast.tile_keys[elem] = *(device const uint4 *)slot;
					else
					{
						const uint w0 = uint(slot[0]) | (uint(slot[1]) << 16u);
						const uint w1 = uint(slot[2]) | (uint(slot[3]) << 16u);
						const uint w2 = uint(slot[4]) | (uint(slot[5]) << 16u);
						const uint w3 = uint(slot[6]) | (uint(slot[7]) << 16u);
						smem.fast.tile_keys[elem] = uint4(w0, w1, w2, w3);
					}
				}
				else
					smem.fast.tile_keys[elem] = uint4(0u);
			}
			threadgroup_barrier(mem_flags::mem_threadgroup);
			auto dots = operation.template get_destination_cooperative_tensor<decltype(q_tensor), decltype(k_tensor), float>();
			operation.run(q_tensor, k_tensor, dots);
			for ( uint16_t i = 0; i < dots.get_capacity(); ++i )
				if ( dots.is_valid_element(i) )
				{
					const auto idx = dots.get_multidimensional_index(i);
					if ( idx[1] < int(p.heads) )
						smem.fast.tile_relu[idx[0]][idx[1]] = fmax(dots[i], 0.0f);
				}
			threadgroup_barrier(mem_flags::mem_threadgroup);
			if ( p.weight_scale != 0.0f )
			{
				if ( tid < SPARK_INDEXER_SCORE_TILE && pos_base + tid < p.score_stride )
				{
					float score = -INFINITY;
					if ( smem.fast.tile_valid[tid] != 0u )
					{
						float total = 0.0f;
						for ( uint h = 0u; h < p.heads; h++ )
							total += fmax(smem.fast.tile_relu[tid][h] * p.score_scale, 0.0f) * shared_weight[h] * p.weight_scale;
						score = total;
					}
					scores[(ulong)row * p.score_stride + pos_base + tid] = score;
				}
			}
			else if ( p.heads_in_order != 0u )
			{
				if ( tid < SPARK_INDEXER_SCORE_TILE && pos_base + tid < p.score_stride )
				{
					float score = -INFINITY;
					if ( smem.fast.tile_valid[tid] != 0u )
					{
						float total = 0.0f;
						if ( p.round_bf16 != 0u )
						{
							for ( uint h = 0u; h < p.heads; h++ )
								total += SparkMathRoundBf16(shared_weight[h] * SparkMathRoundBf16(smem.fast.tile_relu[tid][h]));
							score = SparkMathRoundBf16(total * p.score_scale);
						}
						else
						{
							for ( uint h = 0u; h < p.heads; h++ )
								total += shared_weight[h] * smem.fast.tile_relu[tid][h];
							score = total * p.score_scale;
						}
					}
					scores[(ulong)row * p.score_stride + pos_base + tid] = score;
				}
			}
			else
			{
				const uint k_local = tid >> 3u, g = tid & 7u;
				if ( pos_base + k_local < p.score_stride )
				{
					float score = -INFINITY;
					if ( smem.fast.tile_valid[k_local] != 0u )
					{
						float sum = shared_weight[g] * smem.fast.tile_relu[k_local][g];
						sum = fma(shared_weight[g + groups], smem.fast.tile_relu[k_local][g + groups], sum);
						for ( uint t = 1u; t < tiles; t++ )
						{
							sum = fma(shared_weight[16u * t + g], smem.fast.tile_relu[k_local][16u * t + g], sum);
							sum = fma(shared_weight[16u * t + g + groups], smem.fast.tile_relu[k_local][16u * t + g + groups], sum);
						}
						sum += simd_shuffle_xor(sum, ushort(1));
						sum += simd_shuffle_xor(sum, ushort(2));
						sum += simd_shuffle_xor(sum, ushort(4));
						score = sum * p.score_scale;
					}
					if ( g == 0u )
						scores[(ulong)row * p.score_stride + pos_base + k_local] = score;
				}
			}
			threadgroup_barrier(mem_flags::mem_threadgroup);
		}
		return;
	}
	const uint width = p.heads * p.head_dimension, tiles = p.heads / 16u, groups = SPARK_INDEXER_KERNEL_HEAD_GROUPS;
	device const ushort *q_row = (device const ushort *)query + (ulong)row * p.query_stride;
	for ( uint i = tid; i < width; i += SPARK_INDEXER_SCORE_THREADS )
		smem.slow.shared_query[i] = q_row[i];
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
				smem.slow.shared_key[simd][d] = SparkMathBf16ToFloat(slot[d]);
			simdgroup_barrier(mem_flags::mem_threadgroup);
			for ( uint h = lane; h < p.heads; h += 32u )
			{
				threadgroup const ushort *q = smem.slow.shared_query + h * p.head_dimension;
				float dot = 0.0f;
				for ( uint d = 0u; d < p.head_dimension; d++ )
					dot = fma(SparkMathBf16ToFloat(q[d]), smem.slow.shared_key[simd][d], dot);
				smem.slow.shared_relu[simd][h] = fmax(dot, 0.0f);
			}
			simdgroup_barrier(mem_flags::mem_threadgroup);
			if ( p.weight_scale != 0.0f )
			{
				if ( lane == 0u )
				{
					float total = 0.0f;
					for ( uint h = 0u; h < p.heads; h++ )
						total += fmax(smem.slow.shared_relu[simd][h] * p.score_scale, 0.0f) * shared_weight[h] * p.weight_scale;
					score = total;
				}
			}
			else if ( p.heads_in_order != 0u )
			{
				if ( lane == 0u )
				{
					float total = 0.0f;
					if ( p.round_bf16 != 0u )
					{
						for ( uint h = 0u; h < p.heads; h++ )
							total += SparkMathRoundBf16(shared_weight[h] * SparkMathRoundBf16(smem.slow.shared_relu[simd][h]));
						score = SparkMathRoundBf16(total * p.score_scale);
					}
					else
					{
						for ( uint h = 0u; h < p.heads; h++ )
							total += shared_weight[h] * smem.slow.shared_relu[simd][h];
						score = total * p.score_scale;
					}
				}
			}
			else
			{
				const uint g = lane % groups;
				float sum = shared_weight[g] * smem.slow.shared_relu[simd][g];
				sum = fma(shared_weight[g + groups], smem.slow.shared_relu[simd][g + groups], sum);
				for ( uint t = 1u; t < tiles; t++ )
				{
					sum = fma(shared_weight[16u * t + g], smem.slow.shared_relu[simd][16u * t + g], sum);
					sum = fma(shared_weight[16u * t + g + groups], smem.slow.shared_relu[simd][16u * t + g + groups], sum);
				}
				sum += simd_shuffle_xor(sum, ushort(1));
				sum += simd_shuffle_xor(sum, ushort(2));
				sum += simd_shuffle_xor(sum, ushort(4));
				score = sum * p.score_scale;
			}
		}
		if ( lane == 0u )
			scores[(ulong)row * p.score_stride + position] = score;
	}
}

kernel void spark_indexer_score_pooled(constant SparkIndexerParams &p [[buffer(0)]], device const ushort *query [[buffer(1)]],
	device const ushort *head_weight [[buffer(2)]], device const ushort *pool [[buffer(3)]], device const uint *page_table [[buffer(4)]],
	device const uint *sequence_of_row [[buffer(5)]], device const uint *row_positions [[buffer(6)]], device float *scores [[buffer(7)]],
	device atomic_uint *error [[buffer(8)]], device const float *ape [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	const uint threads = group_size.x;
	threadgroup float pool_key[SPARK_INDEXER_KERNEL_MAX_DIM];
	threadgroup float scratch[32];
	threadgroup ulong slots[SPARK_INDEXER_KERNEL_MAX_RATIO];
	threadgroup uint failed[1];
	const uint row = group.y, pooled = group.x, ratio = max(p.key_ratio, 1u), D = p.head_dimension;
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
		failed[0] = 0u;
		for ( uint i = 0u; i < ratio; i++ )
		{
			ulong offset = 0ul;
			if ( !SparkIndexerSlot(p, page_table, sequence_of_row[row], pooled * ratio + i, error, row, offset) )
				failed[0] = 1u;
			slots[i] = offset;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( failed[0] != 0u )
	{
		if ( tid == 0u )
			*score = -INFINITY;
		return;
	}
	for ( uint d = tid; d < D; d += threads )
	{
		float logits[SPARK_INDEXER_KERNEL_MAX_RATIO];
		float maximum = -INFINITY, sum = 0.0f, key = 0.0f;
		for ( uint i = 0u; i < ratio; i++ )
		{
			logits[i] = SparkMathBf16ToFloat(pool[slots[i] + D + d]) + ape[i * D + d];
			maximum = fmax(maximum, logits[i]);
		}
		for ( uint i = 0u; i < ratio; i++ )
		{
			const float e = logits[i] == -INFINITY ? 0.0f : SPARK_MATH_EXP_FAST(logits[i] - maximum);
			logits[i] = e;
			sum += e;
		}
		const float inv_sum = 1.0f / fmax(sum, 1.0e-20f);
		for ( uint i = 0u; i < ratio; i++ )
		{
			const float mix = logits[i] * inv_sum;
			if ( mix != 0.0f )
				key = fma(mix, SparkMathBf16ToFloat(pool[slots[i] + d]), key);
		}
		pool_key[d] = key;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float local_total = 0.0f;
	device const ushort *q_row = query + (ulong)row * p.query_stride;
	const bool vec4 = (D & 3u) == 0u && ((ulong)q_row & 7ul) == 0ul;
	for ( uint h = simd; h < p.heads; h += simds )
	{
		float dsum = 0.0f;
		device const ushort *qh = q_row + h * D;
		if ( vec4 )
		{
			for ( uint d = lane * 4u; d < D; d += 128u )
			{
				const float4 qv = SparkUnpack4(*(device const uint2 *)(qh + d));
				const float4 kv = *(threadgroup const float4 *)(pool_key + d);
				dsum += dot(qv, kv);
			}
		}
		else
		{
			for ( uint d = lane; d < D; d += 32u )
				dsum = fma(SparkMathBf16ToFloat(qh[d]), pool_key[d], dsum);
		}
		float head_dot = simd_sum(dsum) * p.score_scale;
		if ( lane == 0u && head_dot > 0.0f )
			local_total = fma(head_dot, SparkIndexerWeight(p, head_weight, row, h) * p.weight_scale, local_total);
	}
	if ( lane == 0u )
		scratch[simd] = local_total;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( simd == 0u )
	{
		const float total = simd_sum(lane < simds ? scratch[lane] : 0.0f);
		if ( lane == 0u )
			*score = total;
	}
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
	threadgroup uint scan_greater_off[32];
	threadgroup uint scan_equal_off[32];
	for ( uint start = 0u; start < n; start += threads )
	{
		const uint index = start + tid;
		const uint key = index < n ? SparkMathTopkKey(row_scores[index]) : 0u;
		const uint greater = index < n && key > threshold ? 1u : 0u, equal = index < n && key == threshold ? 1u : 0u;
		const uint greater_in = simd_prefix_exclusive_sum(greater), equal_in = simd_prefix_exclusive_sum(equal);
		const uint greater_simd = simd_sum(greater), equal_simd = simd_sum(equal);
		if ( lane == 0u )
		{
			scan_greater[simd] = greater_simd;
			scan_equal[simd] = equal_simd;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( simd == 0u )
		{
			const uint gv = lane < simds ? scan_greater[lane] : 0u;
			const uint ev = lane < simds ? scan_equal[lane] : 0u;
			const uint gp = simd_prefix_exclusive_sum(gv), ep = simd_prefix_exclusive_sum(ev);
			const uint gt = simd_sum(gv), et = simd_sum(ev);
			if ( lane < simds )
			{
				scan_greater_off[lane] = gp;
				scan_equal_off[lane] = ep;
			}
			if ( lane == 0u )
			{
				const uint need = state[SPARK_INDEXER_TOPK_NEED], seen = state[SPARK_INDEXER_TOPK_SEEN], emitted = state[SPARK_INDEXER_TOPK_EMITTED];
				const uint remaining = need > seen ? need - seen : 0u;
				scan_greater[0] = emitted;
				scan_equal[0] = remaining;
				state[SPARK_INDEXER_TOPK_EMITTED] = emitted + gt + min(et, remaining);
				state[SPARK_INDEXER_TOPK_SEEN] = seen + et;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		const uint emitted = scan_greater[0], remaining = scan_equal[0];
		const uint greater_before = scan_greater_off[simd] + greater_in;
		const uint equal_before = scan_equal_off[simd] + equal_in;
		if ( greater != 0u || (equal != 0u && equal_before < remaining) )
			out[emitted + greater_before + min(equal_before, remaining)] = index;
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

kernel void spark_indexer_block_max(constant SparkIndexerParams &p [[buffer(0)]], device const float *scores [[buffer(1)]],
	device const uint *row_positions [[buffer(2)]], device float *blocks [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]])
{
	if ( row >= p.rows )
		return;
	const uint size = max(p.candidate_block, 1u), width = min(SparkIndexerVisible(p, row_positions[row]), p.score_stride);
	const uint count = (width + size - 1u) / size;
	device const float *row_scores = scores + (ulong)row * p.score_stride;
	device float *out = blocks + (ulong)row * p.block_stride;
	for ( uint block = tid; block < p.block_stride; block += threads )
	{
		float best = -INFINITY;
		if ( block < count )
		{
			if ( block == count - 1u )
				best = INFINITY;
			else
			{
				const uint start = block * size, finish = min(width, start + size);
				uint element = start;
				if ( (start & 3u) == 0u && ((ulong)(row_scores + start) & 15ul) == 0ul )
				{
					float4 b4 = float4(-INFINITY);
					for ( ; element + 4u <= finish; element += 4u )
						b4 = fmax(b4, *(device const float4 *)(row_scores + element));
					best = fmax(fmax(b4.x, b4.y), fmax(b4.z, b4.w));
				}
				for ( ; element < finish; element++ )
					best = fmax(best, row_scores[element]);
			}
		}
		out[block] = best;
	}
}

kernel void spark_indexer_candidate_mask(constant SparkIndexerParams &p [[buffer(0)]], device float *scores [[buffer(1)]],
	device const uint *row_positions [[buffer(2)]], device const uint *candidates [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]])
{
	if ( row >= p.rows )
		return;
	const uint size = max(p.candidate_block, 1u), width = min(SparkIndexerVisible(p, row_positions[row]), p.score_stride);
	const uint blocks = (width + size - 1u) / size;
	device const uint *list = candidates + (ulong)row * p.topk;
	device float *row_scores = scores + (ulong)row * p.score_stride;
	for ( uint block = tid; block < blocks; block += threads )
	{
		uint low = 0u, high = p.topk;
		while ( low < high )
		{
			const uint middle = (low + high) / 2u;
			if ( list[middle] < block )
				low = middle + 1u;
			else
				high = middle;
		}
		if ( low >= p.topk || list[low] != block )
		{
			const uint start = block * size, finish = min(width, start + size);
			uint element = start;
			if ( (start & 3u) == 0u && ((ulong)(row_scores + start) & 15ul) == 0ul )
				for ( ; element + 4u <= finish; element += 4u )
					*(device float4 *)(row_scores + element) = float4(-INFINITY);
			for ( ; element < finish; element++ )
				row_scores[element] = -INFINITY;
		}
	}
}
