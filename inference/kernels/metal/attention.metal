#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "inference/kernels/metal/common.metal"

using namespace mpp::tensor_ops;

#define SPARK_ATTENTION_BLOCK_SIMDS 4u

static_assert(SPARK_ATTENTION_BLOCK_KEYS == SPARK_MATH_FLASH_KEY_TILE, "prefill attention tiles keys as the shared flash walk does");

static inline bool SparkKvPage(device const uint *page_table, uint pages_per_sequence, uint page_slots, uint pool_pages, uint sequence,
	uint position, device atomic_uint *error, uint row, thread uint &page)
{
	const uint index = position / page_slots;
	page = index < pages_per_sequence ? page_table[(ulong)sequence * pages_per_sequence + index] : pool_pages;
	if ( page < pool_pages )
		return true;
	atomic_store_explicit(error, 1u + row, memory_order_relaxed);
	return false;
}

static inline uint SparkAttentionFirstKey(constant SparkAttentionParams &p, uint position)
{
	return p.window != 0u && position + 1u > p.window ? position + 1u - p.window : 0u;
}

static inline void SparkCopyBf16(device ushort *dst, device const ushort *src, uint count, uint tid, uint threads)
{
	if ( (count & 7u) == 0u && (((ulong)dst | (ulong)src) & 15ul) == 0ul )
	{
		const uint chunks = count >> 3u;
		device uint4 *d4 = (device uint4 *)dst;
		device const uint4 *s4 = (device const uint4 *)src;
		for ( uint i = tid; i < chunks; i += threads )
			d4[i] = s4[i];
		return;
	}
	for ( uint i = tid; i < count; i += threads )
		dst[i] = src[i];
}

template <bool SKIP>
kernel void spark_kv_store_paged(constant SparkKvParams &p [[buffer(0)]], device const ushort *key [[buffer(1)]],
	device const ushort *value [[buffer(2)]], device const uint *sequence_of_row [[buffer(3)]], device const uint *positions [[buffer(4)]],
	device const uint *page_table [[buffer(5)]], device ushort *pool [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint row = group.x, head = group.y, position = positions[row];
	if ( SKIP && position == SPARK_KV_SKIP )
		return;
	uint page;
	if ( !SparkKvPage(page_table, p.pages_per_sequence, p.page_slots, p.pool_pages, sequence_of_row[row], position, error, row, page) )
		return;
	const uint values = p.value_dimension != 0u ? p.value_dimension : p.head_dimension;
	device ushort *slot = pool + ((ulong)page * p.page_slots + position % p.page_slots) * (p.kv_heads * (p.head_dimension + values));
	SparkCopyBf16(slot + head * p.head_dimension, key + (ulong)row * p.key_stride + head * p.head_dimension, p.head_dimension, tid, threads);
	SparkCopyBf16(slot + p.kv_heads * p.head_dimension + head * values, value + (ulong)row * p.value_stride + p.value_offset + head * values, values, tid, threads);
}

#define SPARK_KV_STORE_INSTANCE(name, skip) \
	template [[host_name(name)]] kernel void spark_kv_store_paged<skip>(constant SparkKvParams &p [[buffer(0)]], device const ushort *key [[buffer(1)]], \
		device const ushort *value [[buffer(2)]], device const uint *sequence_of_row [[buffer(3)]], device const uint *positions [[buffer(4)]], \
		device const uint *page_table [[buffer(5)]], device ushort *pool [[buffer(6)]], device atomic_uint *error [[buffer(7)]], \
		uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);

SPARK_KV_STORE_INSTANCE("spark_kv_store_paged", false)
SPARK_KV_STORE_INSTANCE("spark_kv_store_paged_skip", true)

kernel void spark_kv_move_paged(constant SparkKvParams &p [[buffer(0)]], device const uint *sequence_of_row [[buffer(1)]],
	device const uint *sources [[buffer(2)]], device const uint *destinations [[buffer(3)]], device const uint *page_table [[buffer(4)]],
	device ushort *pool [[buffer(5)]], device atomic_uint *error [[buffer(6)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint row = group.x, head = group.y, source = sources[row], destination = destinations[row];
	uint source_page, destination_page;
	if ( source == destination )
		return;
	if ( !SparkKvPage(page_table, p.pages_per_sequence, p.page_slots, p.pool_pages, sequence_of_row[row], source, error, row, source_page) ||
		!SparkKvPage(page_table, p.pages_per_sequence, p.page_slots, p.pool_pages, sequence_of_row[row], destination, error, row, destination_page) )
		return;
	const uint values = p.value_dimension != 0u ? p.value_dimension : p.head_dimension;
	const uint slot_width = p.kv_heads * (p.head_dimension + values);
	device const ushort *from = pool + ((ulong)source_page * p.page_slots + source % p.page_slots) * slot_width;
	device ushort *to = pool + ((ulong)destination_page * p.page_slots + destination % p.page_slots) * slot_width;
	SparkCopyBf16(to + head * p.head_dimension, from + head * p.head_dimension, p.head_dimension, tid, threads);
	SparkCopyBf16(to + p.kv_heads * p.head_dimension + head * values, from + p.kv_heads * p.head_dimension + head * values, values, tid, threads);
}

template <uint DIM>
static inline void SparkLoadQueryCoalesced(device const ushort *base, uint lane, thread float4 *v4, thread float2 &v2)
{
	constexpr uint FULL8 = DIM / 256u;
	constexpr uint REM = DIM % 256u;
	if constexpr ( FULL8 > 0u )
		for ( uint c = 0u; c < FULL8; c++ )
			SparkUnpack8(*(device const uint4 *)(base + c * 256u + lane * 8u), v4[2u * c], v4[2u * c + 1u]);
	if constexpr ( REM >= 128u )
		v4[FULL8 * 2u] = SparkUnpack4(*(device const uint2 *)(base + FULL8 * 256u + lane * 4u));
	if constexpr ( (REM % 128u) == 64u )
	{
		const uint bits = *(device const uint *)(base + FULL8 * 256u + (REM >= 128u ? 128u : 0u) + lane * 2u);
		v2 = float2(as_type<bfloat2>(bits));
	}
	else
		v2 = float2(0.0f);
}

template <uint DIM>
static inline float SparkDotKeyCoalesced(device const ushort *base, uint lane, thread const float4 *q4, float2 q2)
{
	constexpr uint FULL8 = DIM / 256u;
	constexpr uint REM = DIM % 256u;
	float score = 0.0f;
	if constexpr ( FULL8 == 2u )
	{
		const uint4 r0 = *(device const uint4 *)(base + lane * 8u);
		const uint4 r1 = *(device const uint4 *)(base + 256u + lane * 8u);
		float4 k0l, k0h, k1l, k1h;
		SparkUnpack8(r0, k0l, k0h);
		SparkUnpack8(r1, k1l, k1h);
		score = (dot(q4[0], k0l) + dot(q4[1], k0h)) + (dot(q4[2], k1l) + dot(q4[3], k1h));
	}
	else if constexpr ( FULL8 == 1u )
	{
		float4 kl, kh;
		SparkUnpack8(*(device const uint4 *)(base + lane * 8u), kl, kh);
		score = dot(q4[0], kl) + dot(q4[1], kh);
	}
	if constexpr ( REM >= 128u )
		score += dot(q4[FULL8 * 2u], SparkUnpack4(*(device const uint2 *)(base + FULL8 * 256u + lane * 4u)));
	if constexpr ( (REM % 128u) == 64u )
	{
		const uint bits = *(device const uint *)(base + FULL8 * 256u + (REM >= 128u ? 128u : 0u) + lane * 2u);
		score += dot(q2, float2(as_type<bfloat2>(bits)));
	}
	return score;
}

template <uint DV>
static inline void SparkAccumValueCoalesced(device const ushort *base, uint lane, float vw, thread float4 *acc4)
{
	constexpr uint VFULL8 = DV / 256u;
	const float4 vw4 = float4(vw);
	if constexpr ( VFULL8 == 2u )
	{
		const uint4 r0 = *(device const uint4 *)(base + lane * 8u);
		const uint4 r1 = *(device const uint4 *)(base + 256u + lane * 8u);
		float4 v0l, v0h, v1l, v1h;
		SparkUnpack8(r0, v0l, v0h);
		SparkUnpack8(r1, v1l, v1h);
		acc4[0] = fma(vw4, v0l, acc4[0]);
		acc4[1] = fma(vw4, v0h, acc4[1]);
		acc4[2] = fma(vw4, v1l, acc4[2]);
		acc4[3] = fma(vw4, v1h, acc4[3]);
	}
	else if constexpr ( VFULL8 == 1u )
	{
		float4 vl, vh;
		SparkUnpack8(*(device const uint4 *)(base + lane * 8u), vl, vh);
		acc4[0] = fma(vw4, vl, acc4[0]);
		acc4[1] = fma(vw4, vh, acc4[1]);
	}
	else
		acc4[0] = fma(vw4, SparkUnpack4(*(device const uint2 *)(base + lane * 4u)), acc4[0]);
}

template <uint D, uint DV, bool EXTRA = false, bool GATHER = false>
kernel void spark_attention_decode(constant SparkAttentionParams &p [[buffer(0)]], device const ushort *query [[buffer(1)]],
	device const ushort *pool [[buffer(2)]], device const uint *page_table [[buffer(3)]], device const uint *sequence_of_row [[buffer(4)]],
	device const uint *positions [[buffer(5)]], device float *partial [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
	device const uint *row_keys [[buffer(8)]], device const uint *slot_offsets [[buffer(9)]],
	uint3 group [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	constexpr uint QFULL8 = D / 256u, QREM = D % 256u;
	constexpr uint Q4 = QFULL8 * 2u + (QREM >= 128u ? 1u : 0u);
	constexpr uint VFULL8 = DV / 256u, V4 = VFULL8 * 2u + (DV == 128u ? 1u : 0u);
	constexpr uint HALF_SIMDS = SPARK_ATTENTION_SIMDS / 2u;
	threadgroup float shared_score[SPARK_ATTENTION_DECODE_PARTITION_KEYS];
	threadgroup uint shared_slot[SPARK_ATTENTION_DECODE_PARTITION_KEYS];
	threadgroup float shared_max[SPARK_ATTENTION_SIMDS];
	threadgroup float shared_sum[4];
	threadgroup float shared_out[HALF_SIMDS][DV];
	const uint linear = group.y * grid.x + group.x;
	const uint part = linear / grid.y, head = linear % grid.y, row = group.z;
	const uint kv_head = (p.first_head + head) / p.group, sequence = sequence_of_row[row], length = positions[row] + 1u;
	const uint visible = GATHER ? length / max(p.gather_ratio, 1u) : length;
	const uint lowest = GATHER && p.gather_ratio > 1u ? 0u : SparkAttentionFirstKey(p, positions[row]);
	const uint key_part = !GATHER && p.window_partitions != 0u ? part + lowest / p.partition_keys : part;
	const uint begin = GATHER ? part * p.partition_keys : max(key_part * p.partition_keys, lowest);
	const uint end = GATHER ? min(p.gather_keys, part * p.partition_keys + p.partition_keys) : min(length, key_part * p.partition_keys + p.partition_keys);
	const uint count_keys = end > begin ? end - begin : 0u;
	const ulong slot_stride = (ulong)p.kv_heads * (D + DV);
	const uint fork_pos = !GATHER && p.tree != 0u ? row_keys[row] : 0u;
	const uint slot_off = !GATHER && p.tree != 0u ? slot_offsets[row] : 0u;
	float4 q4[Q4], acc4[V4];
	float2 q2;
	float local_max = -INFINITY;
	SparkLoadQueryCoalesced<D>(query + (ulong)row * p.query_stride + head * D, lane, q4, q2);
	for ( uint d = 0u; d < V4; d++ )
		acc4[d] = float4(0.0f);
	uint cached_page_idx = 0xffffffffu, cached_page = 0u;
	for ( uint key = begin + simd; key < end; key += SPARK_ATTENTION_SIMDS )
	{
		const uint chosen = GATHER ? row_keys[(ulong)row * p.gather_keys + key] : key;
		if ( GATHER && (chosen >= visible || chosen < lowest) )
		{
			if ( lane == 0u )
				shared_score[key - begin] = -INFINITY;
			continue;
		}
		const uint stored = !GATHER && p.tree != 0u && key >= fork_pos ? key + slot_off : chosen;
		const uint page_idx = stored / p.page_slots;
		uint page;
		if ( page_idx == cached_page_idx )
			page = cached_page;
		else
		{
			if ( !SparkKvPage(page_table, p.pages_per_sequence, p.page_slots, p.pool_pages, sequence, stored, error, row, page) )
			{
				if ( lane == 0u )
					shared_score[key - begin] = -INFINITY;
				continue;
			}
			cached_page_idx = page_idx;
			cached_page = page;
		}
		const uint slot_idx = page * p.page_slots + stored % p.page_slots;
		const float score = simd_sum(SparkDotKeyCoalesced<D>(pool + (ulong)slot_idx * slot_stride + kv_head * D, lane, q4, q2)) * p.scale;
		if ( lane == 0u )
		{
			shared_score[key - begin] = score;
			shared_slot[key - begin] = slot_idx;
		}
		local_max = max(local_max, score);
	}
	if ( lane == 0u )
		shared_max[simd] = local_max;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float partition_max = -INFINITY;
	for ( uint s = 0u; s < SPARK_ATTENTION_SIMDS; s++ )
		partition_max = max(partition_max, shared_max[s]);
	const uint tid = simd * 32u + lane;
	if ( simd < 4u )
	{
		float w = 0.0f;
		if ( tid < count_keys )
		{
			const float sc = shared_score[tid];
			w = sc == -INFINITY ? 0.0f : exp(sc - partition_max);
			shared_score[tid] = w == 0.0f ? 0.0f : SparkMathAttentionValueWeight(w);
		}
		const float ssum = simd_sum(w);
		if ( lane == 0u )
			shared_sum[simd] = ssum;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	const uint v_head_off = p.kv_heads * D + kv_head * DV;
	for ( uint key = begin + simd; key < end; key += SPARK_ATTENTION_SIMDS )
	{
		const float value_weight = shared_score[key - begin];
		if ( value_weight == 0.0f )
			continue;
		SparkAccumValueCoalesced<DV>(pool + (ulong)shared_slot[key - begin] * slot_stride + v_head_off, lane, value_weight, acc4);
	}
	if ( simd >= HALF_SIMDS )
	{
		const uint s_idx = simd - HALF_SIMDS;
		if constexpr ( VFULL8 > 0u )
			for ( uint c = 0u; c < VFULL8; c++ )
			{
				*(threadgroup float4 *)(shared_out[s_idx] + c * 256u + lane * 8u) = acc4[2u * c];
				*(threadgroup float4 *)(shared_out[s_idx] + c * 256u + lane * 8u + 4u) = acc4[2u * c + 1u];
			}
		else
			*(threadgroup float4 *)(shared_out[s_idx] + lane * 4u) = acc4[0];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( simd < HALF_SIMDS )
	{
		if constexpr ( VFULL8 > 0u )
			for ( uint c = 0u; c < VFULL8; c++ )
			{
				*(threadgroup float4 *)(shared_out[simd] + c * 256u + lane * 8u) = acc4[2u * c] +
					*(threadgroup const float4 *)(shared_out[simd] + c * 256u + lane * 8u);
				*(threadgroup float4 *)(shared_out[simd] + c * 256u + lane * 8u + 4u) = acc4[2u * c + 1u] +
					*(threadgroup const float4 *)(shared_out[simd] + c * 256u + lane * 8u + 4u);
			}
		else
			*(threadgroup float4 *)(shared_out[simd] + lane * 4u) = acc4[0] + *(threadgroup const float4 *)(shared_out[simd] + lane * 4u);
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( simd >= HALF_SIMDS )
		return;
	device float *record = partial + (((ulong)row * p.heads + head) * p.partitions + (EXTRA ? p.partition_base : 0u) + part) * (DV + 2u);
	if ( simd == 0u && lane == 0u )
	{
		record[0] = partition_max;
		record[1] = (shared_sum[0] + shared_sum[1]) + (shared_sum[2] + shared_sum[3]);
	}
	constexpr uint PER_SIMD = DV / HALF_SIMDS;
	constexpr uint CHUNKS_PER_LANE = PER_SIMD / 32u;
	const uint base_d = simd * PER_SIMD + lane * CHUNKS_PER_LANE;
	device float *dst = record + 2u + base_d;
	if constexpr ( CHUNKS_PER_LANE == 4u )
	{
		float4 sum4 = (*(threadgroup const float4 *)(shared_out[0] + base_d) + *(threadgroup const float4 *)(shared_out[1] + base_d)) +
			(*(threadgroup const float4 *)(shared_out[2] + base_d) + *(threadgroup const float4 *)(shared_out[3] + base_d));
		*(device float2 *)dst = sum4.xy;
		*(device float2 *)(dst + 2u) = sum4.zw;
	}
	else if constexpr ( CHUNKS_PER_LANE == 2u )
	{
		float2 sum2 = (*(threadgroup const float2 *)(shared_out[0] + base_d) + *(threadgroup const float2 *)(shared_out[1] + base_d)) +
			(*(threadgroup const float2 *)(shared_out[2] + base_d) + *(threadgroup const float2 *)(shared_out[3] + base_d));
		*(device float2 *)dst = sum2;
	}
	else
		dst[0] = (shared_out[0][base_d] + shared_out[1][base_d]) + (shared_out[2][base_d] + shared_out[3][base_d]);
}

template <uint D, uint G>
kernel void spark_attention_decode_gqa(constant SparkAttentionParams &p [[buffer(0)]], device const ushort *query [[buffer(1)]],
	device const ushort *pool [[buffer(2)]], device const uint *page_table [[buffer(3)]], device const uint *sequence_of_row [[buffer(4)]],
	device const uint *positions [[buffer(5)]], device float *partial [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
	device const uint *row_keys [[buffer(8)]], device const uint *slot_offsets [[buffer(9)]],
	uint3 group [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	constexpr uint HALF_SIMDS = SPARK_ATTENTION_SIMDS / 2u;
	constexpr uint VFULL8 = D / 256u;
	threadgroup float shared_score[G][SPARK_ATTENTION_DECODE_PARTITION_KEYS];
	threadgroup uint shared_slot[SPARK_ATTENTION_DECODE_PARTITION_KEYS];
	threadgroup float shared_max[G][SPARK_ATTENTION_SIMDS];
	threadgroup float shared_sum[G][4];
	threadgroup float shared_out[HALF_SIMDS][D];
	const uint linear = group.y * grid.x + group.x;
	const uint part = linear / grid.y, gblock = linear % grid.y, row = group.z;
	const uint head0 = gblock * G;
	const uint kv_head = (p.first_head + head0) / p.group, sequence = sequence_of_row[row], length = positions[row] + 1u;
	const uint lowest = SparkAttentionFirstKey(p, positions[row]);
	const uint key_part = p.window_partitions != 0u ? part + lowest / p.partition_keys : part;
	const uint begin = max(key_part * p.partition_keys, lowest);
	const uint end = min(length, key_part * p.partition_keys + p.partition_keys);
	const uint count_keys = end > begin ? end - begin : 0u;
	const ulong slot_stride = (ulong)p.kv_heads * (D + D);
	const uint fork_pos = p.tree != 0u ? row_keys[row] : 0u;
	const uint slot_off = p.tree != 0u ? slot_offsets[row] : 0u;
	{
		float4 q0[G], q1[G];
		float lmax[G];
		for ( uint g = 0u; g < G; g++ )
		{
			device const ushort *qp = query + (ulong)row * p.query_stride + (head0 + g) * D;
			if constexpr ( VFULL8 == 1u )
				SparkUnpack8(*(device const uint4 *)(qp + lane * 8u), q0[g], q1[g]);
			else
			{
				q0[g] = SparkUnpack4(*(device const uint2 *)(qp + lane * 4u));
				q1[g] = float4(0.0f);
			}
			lmax[g] = -INFINITY;
		}
		uint cached_page_idx = 0xffffffffu, cached_page = 0u;
		for ( uint key = begin + simd; key < end; key += SPARK_ATTENTION_SIMDS )
		{
			const uint stored = p.tree != 0u && key >= fork_pos ? key + slot_off : key;
			const uint page_idx = stored / p.page_slots;
			uint page;
			if ( page_idx == cached_page_idx )
				page = cached_page;
			else
			{
				if ( !SparkKvPage(page_table, p.pages_per_sequence, p.page_slots, p.pool_pages, sequence, stored, error, row, page) )
				{
					if ( lane == 0u )
						for ( uint g = 0u; g < G; g++ )
							shared_score[g][key - begin] = -INFINITY;
					continue;
				}
				cached_page_idx = page_idx;
				cached_page = page;
			}
			const uint slot_idx = page * p.page_slots + stored % p.page_slots;
			device const ushort *kp = pool + (ulong)slot_idx * slot_stride + kv_head * D;
			float4 kl, kh;
			if constexpr ( VFULL8 == 1u )
				SparkUnpack8(*(device const uint4 *)(kp + lane * 8u), kl, kh);
			else
			{
				kl = SparkUnpack4(*(device const uint2 *)(kp + lane * 4u));
				kh = float4(0.0f);
			}
			if ( lane == 0u )
				shared_slot[key - begin] = slot_idx;
			for ( uint g = 0u; g < G; g++ )
			{
				const float d = VFULL8 == 1u ? dot(q0[g], kl) + dot(q1[g], kh) : dot(q0[g], kl);
				const float score = simd_sum(d) * p.scale;
				if ( lane == 0u )
					shared_score[g][key - begin] = score;
				lmax[g] = max(lmax[g], score);
			}
		}
		if ( lane == 0u )
			for ( uint g = 0u; g < G; g++ )
				shared_max[g][simd] = lmax[g];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float pmax[G];
	const uint tid = simd * 32u + lane;
	for ( uint g = 0u; g < G; g++ )
	{
		float m = -INFINITY;
		for ( uint s = 0u; s < SPARK_ATTENTION_SIMDS; s++ )
			m = max(m, shared_max[g][s]);
		pmax[g] = m;
		if ( simd < 4u )
		{
			float w = 0.0f;
			if ( tid < count_keys )
			{
				const float sc = shared_score[g][tid];
				w = sc == -INFINITY ? 0.0f : exp(sc - m);
				shared_score[g][tid] = w == 0.0f ? 0.0f : SparkMathAttentionValueWeight(w);
			}
			const float ssum = simd_sum(w);
			if ( lane == 0u )
				shared_sum[g][simd] = ssum;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float4 acc0[G], acc1[G];
	for ( uint g = 0u; g < G; g++ )
	{
		acc0[g] = float4(0.0f);
		acc1[g] = float4(0.0f);
	}
	const uint v_head_off = p.kv_heads * D + kv_head * D;
	for ( uint key = begin + simd; key < end; key += SPARK_ATTENTION_SIMDS )
	{
		const uint kidx = key - begin;
		float vw[G];
		bool any_nz = false;
		for ( uint g = 0u; g < G; g++ )
		{
			vw[g] = shared_score[g][kidx];
			any_nz = any_nz || (vw[g] != 0.0f);
		}
		if ( !any_nz )
			continue;
		device const ushort *vp = pool + (ulong)shared_slot[kidx] * slot_stride + v_head_off;
		float4 vl, vh;
		if constexpr ( VFULL8 == 1u )
			SparkUnpack8(*(device const uint4 *)(vp + lane * 8u), vl, vh);
		else
		{
			vl = SparkUnpack4(*(device const uint2 *)(vp + lane * 4u));
			vh = float4(0.0f);
		}
		for ( uint g = 0u; g < G; g++ )
			if ( vw[g] != 0.0f )
			{
				const float4 vw4 = float4(vw[g]);
				acc0[g] = fma(vw4, vl, acc0[g]);
				if constexpr ( VFULL8 == 1u )
					acc1[g] = fma(vw4, vh, acc1[g]);
			}
	}
	constexpr uint PER_SIMD = D / HALF_SIMDS;
	constexpr uint CHUNKS_PER_LANE = PER_SIMD / 32u;
	const uint base_d = simd * PER_SIMD + lane * CHUNKS_PER_LANE;
	for ( uint g = 0u; g < G; g++ )
	{
		if ( g > 0u )
			threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( simd >= HALF_SIMDS )
		{
			const uint s_idx = simd - HALF_SIMDS;
			if constexpr ( VFULL8 == 1u )
			{
				*(threadgroup float4 *)(shared_out[s_idx] + lane * 8u) = acc0[g];
				*(threadgroup float4 *)(shared_out[s_idx] + lane * 8u + 4u) = acc1[g];
			}
			else
				*(threadgroup float4 *)(shared_out[s_idx] + lane * 4u) = acc0[g];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( simd < HALF_SIMDS )
		{
			if constexpr ( VFULL8 == 1u )
			{
				*(threadgroup float4 *)(shared_out[simd] + lane * 8u) = acc0[g] + *(threadgroup const float4 *)(shared_out[simd] + lane * 8u);
				*(threadgroup float4 *)(shared_out[simd] + lane * 8u + 4u) = acc1[g] + *(threadgroup const float4 *)(shared_out[simd] + lane * 8u + 4u);
			}
			else
				*(threadgroup float4 *)(shared_out[simd] + lane * 4u) = acc0[g] + *(threadgroup const float4 *)(shared_out[simd] + lane * 4u);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( simd < HALF_SIMDS )
		{
			device float *record = partial + (((ulong)row * p.heads + (head0 + g)) * p.partitions + part) * (D + 2u);
			if ( simd == 0u && lane == 0u )
			{
				record[0] = pmax[g];
				record[1] = (shared_sum[g][0] + shared_sum[g][1]) + (shared_sum[g][2] + shared_sum[g][3]);
			}
			device float *dst = record + 2u + base_d;
			if constexpr ( CHUNKS_PER_LANE == 2u )
			{
				float2 sum2 = (*(threadgroup const float2 *)(shared_out[0] + base_d) + *(threadgroup const float2 *)(shared_out[1] + base_d)) +
					(*(threadgroup const float2 *)(shared_out[2] + base_d) + *(threadgroup const float2 *)(shared_out[3] + base_d));
				*(device float2 *)dst = sum2;
			}
			else
				dst[0] = (shared_out[0][base_d] + shared_out[1][base_d]) + (shared_out[2][base_d] + shared_out[3][base_d]);
		}
	}
}

template <uint DK, uint D, bool EXTRA = false, bool GATHER = false>
kernel void spark_attention_combine(constant SparkAttentionParams &p [[buffer(0)]], device const float *partial [[buffer(1)]],
	device const ushort *gate [[buffer(2)]], device const uint *positions [[buffer(3)]], device ushort *output [[buffer(4)]],
	device const float *sinks [[buffer(5)]], device const uint *extra_positions [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	constexpr uint PAIRS = D / 2u;
	constexpr uint PART_GROUPS = 256u / PAIRS;
	threadgroup float scratch[32];
	threadgroup float factors[512];
	threadgroup float2 reduce_buf[PART_GROUPS > 1u ? PART_GROUPS : 1u][PART_GROUPS > 1u ? PAIRS : 1u];
	const uint threads = group_size.x;
	const uint head = group.x, row = group.y, length = positions[row] + 1u;
	const uint primary = EXTRA ? p.partitions - p.extra_partitions : p.partitions;
	const uint first_part = !GATHER && p.window_partitions != 0u ? SparkAttentionFirstKey(p, positions[row]) / p.partition_keys : 0u;
	const uint used = GATHER ? p.partitions : min(primary, (length + p.partition_keys - 1u) / p.partition_keys - first_part);
	const uint extra_used = EXTRA ? min(p.extra_partitions, (extra_positions[row] + 1u + p.partition_keys - 1u) / p.partition_keys) : 0u;
	const uint total_parts = used + extra_used;
	device const float *records = partial + ((ulong)row * p.heads + head) * p.partitions * (D + 2u);
	float local_max = -INFINITY;
	for ( uint part = tid; part < total_parts; part += threads )
	{
		const uint slot_part = part < used ? part : primary + part - used;
		local_max = max(local_max, records[slot_part * (D + 2u)]);
	}
	float total_max = SparkGroupMax(local_max, scratch, lane, simd, simds);
	const float sink = p.has_sinks != 0u ? sinks[head] : -INFINITY;
	total_max = max(total_max, sink);
	float local_sum = 0.0f;
	for ( uint part = tid; part < total_parts; part += threads )
	{
		const uint slot_part = part < used ? part : primary + part - used;
		device const float *record = records + slot_part * (D + 2u);
		const float factor = record[0] == -INFINITY ? 0.0f : (record[0] == total_max ? 1.0f : exp(record[0] - total_max));
		if ( part < 512u )
			factors[part] = factor;
		local_sum = fma(record[1], factor, local_sum);
	}
	float sum = SparkGroupSum(local_sum, scratch, lane, simd, simds);
	if ( p.has_sinks != 0u )
		sum += exp(sink - total_max);
	if constexpr ( PART_GROUPS > 1u )
	{
		const uint pg = tid / PAIRS, pair = tid % PAIRS, d = pair * 2u;
		float2 val = float2(0.0f);
		if ( pg < PART_GROUPS )
		{
			for ( uint part = pg; part < total_parts; part += PART_GROUPS )
			{
				const uint slot_part = part < used ? part : primary + part - used;
				device const float *record = records + slot_part * (D + 2u);
				const float factor = part < 512u ? factors[part] : (record[0] == -INFINITY ? 0.0f : exp(record[0] - total_max));
				val = fma(*(device const float2 *)(record + 2u + d), float2(factor), val);
			}
			reduce_buf[pg][pair] = val;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( tid >= PAIRS )
			return;
		float2 acc = reduce_buf[0][tid];
		for ( uint g = 1u; g < PART_GROUPS; g++ )
			acc += reduce_buf[g][tid];
		float a0 = SparkMathRoundBf16(acc.x / sum);
		float a1 = SparkMathRoundBf16(acc.y / sum);
		if ( p.output_gate != 0u )
		{
			device const ushort *gptr = gate + (ulong)row * p.gate_stride + p.gate_offset + head * p.gate_head_stride + tid * 2u;
			const float s0 = SparkMathSigmoid(SparkMathBf16ToFloat(gptr[0]));
			const float s1 = SparkMathSigmoid(SparkMathBf16ToFloat(gptr[1]));
			a0 *= p.output_gate == SPARK_ATTENTION_GATE_SIGMOID_BF16 ? SparkMathRoundBf16(s0) : s0;
			a1 *= p.output_gate == SPARK_ATTENTION_GATE_SIGMOID_BF16 ? SparkMathRoundBf16(s1) : s1;
		}
		device ushort *out_ptr = output + (ulong)row * p.output_stride + head * D + tid * 2u;
		if ( ((ulong)out_ptr & 3ul) == 0ul )
			*(device uint *)out_ptr = SparkPack2(a0, a1);
		else
		{
			out_ptr[0] = SparkMathFloatToBf16(a0);
			out_ptr[1] = SparkMathFloatToBf16(a1);
		}
		return;
	}
	for ( uint pair = tid; pair < PAIRS; pair += threads )
	{
		const uint d = pair * 2u;
		float2 val = float2(0.0f);
		for ( uint part = 0u; part < total_parts; part++ )
		{
			const uint slot_part = part < used ? part : primary + part - used;
			device const float *record = records + slot_part * (D + 2u);
			const float factor = part < 512u ? factors[part] : (record[0] == -INFINITY ? 0.0f : exp(record[0] - total_max));
			val = fma(*(device const float2 *)(record + 2u + d), float2(factor), val);
		}
		float a0 = SparkMathRoundBf16(val.x / sum);
		float a1 = SparkMathRoundBf16(val.y / sum);
		if ( p.output_gate != 0u )
		{
			device const ushort *gptr = gate + (ulong)row * p.gate_stride + p.gate_offset + head * p.gate_head_stride + d;
			const float s0 = SparkMathSigmoid(SparkMathBf16ToFloat(gptr[0]));
			const float s1 = SparkMathSigmoid(SparkMathBf16ToFloat(gptr[1]));
			a0 *= p.output_gate == SPARK_ATTENTION_GATE_SIGMOID_BF16 ? SparkMathRoundBf16(s0) : s0;
			a1 *= p.output_gate == SPARK_ATTENTION_GATE_SIGMOID_BF16 ? SparkMathRoundBf16(s1) : s1;
		}
		device ushort *out_ptr = output + (ulong)row * p.output_stride + head * D + d;
		if ( ((ulong)out_ptr & 3ul) == 0ul )
			*(device uint *)out_ptr = SparkPack2(a0, a1);
		else
		{
			out_ptr[0] = SparkMathFloatToBf16(a0);
			out_ptr[1] = SparkMathFloatToBf16(a1);
		}
	}
}

template <uint D, uint DV>
kernel void spark_attention_prefill_blocks(constant SparkAttentionParams &p [[buffer(0)]], device bfloat *query [[buffer(1)]],
	device bfloat *pool [[buffer(2)]], device const uint *page_table [[buffer(3)]], device const uint *sequence_of_row [[buffer(4)]],
	device const uint *positions [[buffer(5)]], device float *partial [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
	uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	constexpr uint ROWS = SPARK_ATTENTION_BLOCK_ROWS(D);
	constexpr uint BSIMDS = D > 256u ? 8u : SPARK_ATTENTION_BLOCK_SIMDS;
	threadgroup float block[ROWS * SPARK_ATTENTION_BLOCK_KEYS];
	threadgroup bfloat probability[ROWS * SPARK_ATTENTION_BLOCK_KEYS];
	threadgroup float row_max[ROWS];
	threadgroup float row_sum[ROWS];
	threadgroup float row_scale[ROWS];
	threadgroup uint any_rescale[1];
	const uint part = group.x, head = group.y, kv_head = (p.first_head + head) / p.group;
	const uint first_row = p.row_offset + group.z * ROWS;
	if ( first_row >= p.rows )
		return;
	const uint count = min(ROWS, p.rows - first_row), sequence = sequence_of_row[first_row];
	const uint first_position = positions[first_row], span_end = positions[first_row + count - 1u] + 1u;
	const uint begin = max(part * p.partition_keys, SparkAttentionFirstKey(p, first_position) / SPARK_ATTENTION_BLOCK_KEYS * SPARK_ATTENTION_BLOCK_KEYS);
	const uint end = min(span_end, part * p.partition_keys + p.partition_keys);
	const uint slot_stride = p.kv_heads * (D + DV);
	const float scale = p.scale;
	const uint max_window_low = SparkAttentionFirstKey(p, first_position + count - 1u);
	constexpr auto score_descriptor = matmul2d_descriptor(ROWS, SPARK_ATTENTION_BLOCK_KEYS, D, false, true, false);
	constexpr auto value_full_descriptor = matmul2d_descriptor(ROWS, DV, SPARK_ATTENTION_BLOCK_KEYS, false, false,
		false, matmul2d_descriptor::mode::multiply_accumulate);
	constexpr auto value_tail_descriptor = matmul2d_descriptor(ROWS, DV, static_cast<int>(dynamic_extent), false, false,
		false, matmul2d_descriptor::mode::multiply_accumulate);
	matmul2d<score_descriptor, execution_simdgroups<BSIMDS>> score_operation;
	matmul2d<value_full_descriptor, execution_simdgroups<BSIMDS>> value_full_operation;
	matmul2d<value_tail_descriptor, execution_simdgroups<BSIMDS>> value_tail_operation;
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> queries(query + (ulong)first_row * p.query_stride + head * D,
		dextents<int32_t, 2>(D, count), array<int32_t, 2>{ 1, int(p.query_stride) });
	tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> weights(probability, dextents<int32_t, 2>(SPARK_ATTENTION_BLOCK_KEYS, ROWS),
		array<int32_t, 2>{ 1, int(SPARK_ATTENTION_BLOCK_KEYS) });
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> first_values(pool, dextents<int32_t, 2>(DV, SPARK_ATTENTION_BLOCK_KEYS),
		array<int32_t, 2>{ 1, int(slot_stride) });
	auto attended = value_full_operation.template get_destination_cooperative_tensor<decltype(weights), decltype(first_values), float>();
	for ( uint16_t i = 0; i < attended.get_capacity(); ++i )
		if ( attended.is_valid_element(i) )
			attended[i] = 0.0f;
	if ( simd == 0u && lane < ROWS )
	{
		row_max[lane] = -INFINITY;
		row_sum[lane] = 0.0f;
	}
	const uint tiles = SparkMathFlashTiles(begin, end);
	for ( uint step = 0u; step < tiles; step++ )
	{
		const uint tile = SparkMathFlashTile(end, step);
		const uint keys = min(SPARK_ATTENTION_BLOCK_KEYS, end - tile);
		uint page;
		if ( !SparkKvPage(page_table, p.pages_per_sequence, p.page_slots, p.pool_pages, sequence, tile, error, first_row, page) )
			continue;
		device bfloat *slot = pool + ((ulong)page * p.page_slots + tile % p.page_slots) * slot_stride;
		tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> key_block(slot + kv_head * D, dextents<int32_t, 2>(D, keys),
			array<int32_t, 2>{ 1, int(slot_stride) });
		tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> value_block(slot + p.kv_heads * D + kv_head * DV, dextents<int32_t, 2>(DV, keys),
			array<int32_t, 2>{ 1, int(slot_stride) });
		auto scores = score_operation.template get_destination_cooperative_tensor<decltype(queries), decltype(key_block), float>();
		score_operation.run(queries, key_block, scores);
		for ( uint16_t i = 0; i < scores.get_capacity(); ++i )
			if ( scores.is_valid_element(i) )
			{
				const auto index = scores.get_multidimensional_index(i);
				block[index[1] * SPARK_ATTENTION_BLOCK_KEYS + index[0]] = scores[i];
			}
		if ( simd == 0u && lane == 0u )
			any_rescale[0] = 0u;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		const bool unmasked = keys == SPARK_ATTENTION_BLOCK_KEYS && tile + SPARK_ATTENTION_BLOCK_KEYS <= first_position + 1u && tile >= max_window_low;
		uint local_rescale = 0u;
		for ( uint r = simd; r < ROWS; r += BSIMDS )
		{
			float low_value, high_value;
			if ( unmasked )
			{
				low_value = r < count ? block[r * SPARK_ATTENTION_BLOCK_KEYS + lane] * scale : -INFINITY;
				high_value = r < count ? block[r * SPARK_ATTENTION_BLOCK_KEYS + lane + 32u] * scale : -INFINITY;
			}
			else
			{
				const uint limit = r < count ? first_position + r + 1u : 0u, low = r < count ? SparkAttentionFirstKey(p, first_position + r) : 0u;
				const uint low_key = tile + lane, high_key = tile + lane + 32u;
				low_value = low_key < limit && low_key >= low && lane < keys ? block[r * SPARK_ATTENTION_BLOCK_KEYS + lane] * scale : -INFINITY;
				high_value = high_key < limit && high_key >= low && lane + 32u < keys ? block[r * SPARK_ATTENTION_BLOCK_KEYS + lane + 32u] * scale : -INFINITY;
			}
			const float previous = row_max[r], tile_max = simd_max(max(low_value, high_value));
			const float next = max(previous, tile_max);
			const float correction = (next == -INFINITY || next == previous) ? 1.0f : exp(previous - next);
			low_value = next == -INFINITY ? 0.0f : exp(low_value - next);
			high_value = next == -INFINITY ? 0.0f : exp(high_value - next);
			const float total = simd_sum(low_value + high_value);
			probability[r * SPARK_ATTENTION_BLOCK_KEYS + lane] = as_type<bfloat>(SparkMathFloatToBf16(low_value));
			probability[r * SPARK_ATTENTION_BLOCK_KEYS + lane + 32u] = as_type<bfloat>(SparkMathFloatToBf16(high_value));
			if ( lane == 0u )
			{
				row_max[r] = next;
				row_sum[r] = row_sum[r] * correction + total;
				row_scale[r] = correction;
				if ( correction != 1.0f )
					local_rescale = 1u;
			}
		}
		if ( lane == 0u && local_rescale != 0u )
			any_rescale[0] = 1u;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( any_rescale[0] != 0u )
			for ( uint16_t i = 0; i < attended.get_capacity(); ++i )
				if ( attended.is_valid_element(i) )
					attended[i] *= row_scale[attended.get_multidimensional_index(i)[1]];
		if ( keys == SPARK_ATTENTION_BLOCK_KEYS )
			value_full_operation.run(weights, value_block, attended);
		else
		{
			tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tile_weights(probability, dextents<int32_t, 2>(keys, ROWS),
				array<int32_t, 2>{ 1, int(SPARK_ATTENTION_BLOCK_KEYS) });
			value_tail_operation.run(tile_weights, value_block, attended);
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint16_t i = 0; i < attended.get_capacity(); ++i )
		if ( attended.is_valid_element(i) )
		{
			const auto index = attended.get_multidimensional_index(i);
			const uint r = uint(index[1]);
			if ( r < count && part * p.partition_keys < first_position + r + 1u )
				partial[(((ulong)(first_row + r) * p.heads + head) * p.partitions + part) * (DV + 2u) + 2u + uint(index[0])] = attended[i];
		}
	if ( simd == 0u && lane < count && part * p.partition_keys < first_position + lane + 1u )
	{
		device float *record = partial + (((ulong)(first_row + lane) * p.heads + head) * p.partitions + part) * (DV + 2u);
		record[0] = row_max[lane];
		record[1] = row_sum[lane];
	}
}

#define SPARK_ATTENTION_INSTANCES(name, dk, dv) \
	template [[host_name("spark_attention_decode_d" name)]] kernel void spark_attention_decode<dk, dv>(constant SparkAttentionParams &p [[buffer(0)]], \
		device const ushort *query [[buffer(1)]], device const ushort *pool [[buffer(2)]], device const uint *page_table [[buffer(3)]], \
		device const uint *sequence_of_row [[buffer(4)]], device const uint *positions [[buffer(5)]], device float *partial [[buffer(6)]], \
		device atomic_uint *error [[buffer(7)]], device const uint *fork_positions [[buffer(8)]], device const uint *slot_offsets [[buffer(9)]], \
		uint3 group [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]], \
		uint simd [[simdgroup_index_in_threadgroup]]); \
	template [[host_name("spark_attention_combine_d" name)]] kernel void spark_attention_combine<dk, dv>(constant SparkAttentionParams &p [[buffer(0)]], \
		device const float *partial [[buffer(1)]], device const ushort *gate [[buffer(2)]], device const uint *positions [[buffer(3)]], \
		device ushort *output [[buffer(4)]], device const float *sinks [[buffer(5)]], device const uint *extra_positions [[buffer(6)]], \
		uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]], \
		uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]]); \
	template [[host_name("spark_attention_prefill_blocks_d" name)]] kernel void spark_attention_prefill_blocks<dk, dv>( \
		constant SparkAttentionParams &p [[buffer(0)]], device bfloat *query [[buffer(1)]], device bfloat *pool [[buffer(2)]], \
		device const uint *page_table [[buffer(3)]], device const uint *sequence_of_row [[buffer(4)]], device const uint *positions [[buffer(5)]], \
		device float *partial [[buffer(6)]], device atomic_uint *error [[buffer(7)]], uint3 group [[threadgroup_position_in_grid]], \
		uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]);

#define SPARK_ATTENTION_GATHER_INSTANCES(name, dk, dv) \
	template [[host_name("spark_attention_decode_gather_d" name)]] kernel void spark_attention_decode<dk, dv, false, true>( \
		constant SparkAttentionParams &p [[buffer(0)]], device const ushort *query [[buffer(1)]], device const ushort *pool [[buffer(2)]], \
		device const uint *page_table [[buffer(3)]], device const uint *sequence_of_row [[buffer(4)]], device const uint *positions [[buffer(5)]], \
		device float *partial [[buffer(6)]], device atomic_uint *error [[buffer(7)]], device const uint *row_keys [[buffer(8)]], \
		device const uint *slot_offsets [[buffer(9)]], uint3 group [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]], \
		uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]); \
	template [[host_name("spark_attention_combine_gather_d" name)]] kernel void spark_attention_combine<dk, dv, false, true>( \
		constant SparkAttentionParams &p [[buffer(0)]], device const float *partial [[buffer(1)]], device const ushort *gate [[buffer(2)]], \
		device const uint *positions [[buffer(3)]], device ushort *output [[buffer(4)]], device const float *sinks [[buffer(5)]], \
		device const uint *extra_positions [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
		uint2 group_size [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], \
		uint simds [[simdgroups_per_threadgroup]]);

SPARK_ATTENTION_INSTANCES("256", 256, 256)
SPARK_ATTENTION_INSTANCES("512", 512, 512)
SPARK_ATTENTION_INSTANCES("128", 128, 128)
SPARK_ATTENTION_INSTANCES("192v128", 192, 128)
SPARK_ATTENTION_INSTANCES("576v512", 576, 512)
#define SPARK_ATTENTION_GQA_INSTANCE(name, d, g) \
	template [[host_name(name)]] kernel void spark_attention_decode_gqa<d, g>(constant SparkAttentionParams &p [[buffer(0)]], \
		device const ushort *query [[buffer(1)]], device const ushort *pool [[buffer(2)]], device const uint *page_table [[buffer(3)]], \
		device const uint *sequence_of_row [[buffer(4)]], device const uint *positions [[buffer(5)]], device float *partial [[buffer(6)]], \
		device atomic_uint *error [[buffer(7)]], device const uint *fork_positions [[buffer(8)]], device const uint *slot_offsets [[buffer(9)]], \
		uint3 group [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]], \
		uint simd [[simdgroup_index_in_threadgroup]]);

SPARK_ATTENTION_GQA_INSTANCE("spark_attention_decode_gqa4_d256", 256, 4)
SPARK_ATTENTION_GQA_INSTANCE("spark_attention_decode_gqa6_d256", 256, 6)
SPARK_ATTENTION_GQA_INSTANCE("spark_attention_decode_gqa4_d128", 128, 4)
SPARK_ATTENTION_GATHER_INSTANCES("256", 256, 256)
SPARK_ATTENTION_GATHER_INSTANCES("512", 512, 512)
SPARK_ATTENTION_GATHER_INSTANCES("128", 128, 128)
SPARK_ATTENTION_GATHER_INSTANCES("192v128", 192, 128)
SPARK_ATTENTION_GATHER_INSTANCES("576v512", 576, 512)

template [[host_name("spark_attention_decode_extra_d512")]] kernel void spark_attention_decode<512, 512, true>(constant SparkAttentionParams &p [[buffer(0)]],
	device const ushort *query [[buffer(1)]], device const ushort *pool [[buffer(2)]], device const uint *page_table [[buffer(3)]],
	device const uint *sequence_of_row [[buffer(4)]], device const uint *positions [[buffer(5)]], device float *partial [[buffer(6)]],
	device atomic_uint *error [[buffer(7)]], device const uint *fork_positions [[buffer(8)]], device const uint *slot_offsets [[buffer(9)]],
	uint3 group [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]]);
template [[host_name("spark_attention_decode_extra_gather_d512")]] kernel void spark_attention_decode<512, 512, true, true>(
	constant SparkAttentionParams &p [[buffer(0)]], device const ushort *query [[buffer(1)]], device const ushort *pool [[buffer(2)]],
	device const uint *page_table [[buffer(3)]], device const uint *sequence_of_row [[buffer(4)]], device const uint *positions [[buffer(5)]],
	device float *partial [[buffer(6)]], device atomic_uint *error [[buffer(7)]], device const uint *row_keys [[buffer(8)]],
	device const uint *slot_offsets [[buffer(9)]], uint3 group [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]);
template [[host_name("spark_attention_combine_extra_d512")]] kernel void spark_attention_combine<512, 512, true>(constant SparkAttentionParams &p [[buffer(0)]],
	device const float *partial [[buffer(1)]], device const ushort *gate [[buffer(2)]], device const uint *positions [[buffer(3)]],
	device ushort *output [[buffer(4)]], device const float *sinks [[buffer(5)]], device const uint *extra_positions [[buffer(6)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]]);
