#include "inference/kernels/metal/common.metal"

#define SPARK_ATTENTION_DIM 256u
#define SPARK_ATTENTION_LANE_DIMS (SPARK_ATTENTION_DIM / 32u)
#define SPARK_ATTENTION_SIMDS 8u
#define SPARK_ATTENTION_TILE 32u

static inline device const ushort *SparkKvSlot(constant SparkAttentionParams &p, device const ushort *pool, device const uint *page_table,
	uint sequence, uint position)
{
	const uint page = page_table[(ulong)sequence * p.pages_per_sequence + position / p.page_slots];
	if ( page >= p.pool_pages )
		return (device const ushort *)0;
	return pool + ((ulong)page * p.page_slots + position % p.page_slots) * (2u * p.kv_heads * p.head_dimension);
}

static inline void SparkLoad8(device const ushort *source, thread float *values)
{
	const uint4 bits = *(device const uint4 *)source;
	values[0] = as_type<float>(bits.x << 16);
	values[1] = as_type<float>(bits.x & 0xffff0000u);
	values[2] = as_type<float>(bits.y << 16);
	values[3] = as_type<float>(bits.y & 0xffff0000u);
	values[4] = as_type<float>(bits.z << 16);
	values[5] = as_type<float>(bits.z & 0xffff0000u);
	values[6] = as_type<float>(bits.w << 16);
	values[7] = as_type<float>(bits.w & 0xffff0000u);
}

static inline void SparkLoad8(threadgroup const ushort *source, thread float *values)
{
	const uint4 bits = *(threadgroup const uint4 *)source;
	values[0] = as_type<float>(bits.x << 16);
	values[1] = as_type<float>(bits.x & 0xffff0000u);
	values[2] = as_type<float>(bits.y << 16);
	values[3] = as_type<float>(bits.y & 0xffff0000u);
	values[4] = as_type<float>(bits.z << 16);
	values[5] = as_type<float>(bits.z & 0xffff0000u);
	values[6] = as_type<float>(bits.w << 16);
	values[7] = as_type<float>(bits.w & 0xffff0000u);
}

kernel void spark_kv_store_paged(constant SparkKvParams &p [[buffer(0)]], device const ushort *key [[buffer(1)]],
	device const ushort *value [[buffer(2)]], device const uint *sequence_of_row [[buffer(3)]], device const uint *positions [[buffer(4)]],
	device const uint *page_table [[buffer(5)]], device ushort *pool [[buffer(6)]], device atomic_uint *error [[buffer(7)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint row = group.x, head = group.y, position = positions[row];
	const uint page = page_table[(ulong)sequence_of_row[row] * p.pages_per_sequence + position / p.page_slots];
	if ( page >= p.pool_pages || position / p.page_slots >= p.pages_per_sequence )
	{
		if ( tid == 0u )
			atomic_store_explicit(error, 1u + row, memory_order_relaxed);
		return;
	}
	device ushort *slot = pool + ((ulong)page * p.page_slots + position % p.page_slots) * (2u * p.kv_heads * p.head_dimension);
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		slot[head * p.head_dimension + i] = key[(ulong)row * p.key_stride + head * p.head_dimension + i];
		slot[(p.kv_heads + head) * p.head_dimension + i] = value[(ulong)row * p.value_stride + p.value_offset + head * p.head_dimension + i];
	}
}

kernel void spark_attention_decode(constant SparkAttentionParams &p [[buffer(0)]], device const ushort *query [[buffer(1)]],
	device const ushort *pool [[buffer(2)]], device const uint *page_table [[buffer(3)]], device const uint *sequence_of_row [[buffer(4)]],
	device const uint *positions [[buffer(5)]], device float *partial [[buffer(6)]],
	uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup float shared_max[SPARK_ATTENTION_SIMDS];
	threadgroup float shared_sum[SPARK_ATTENTION_SIMDS];
	threadgroup float shared_out[SPARK_ATTENTION_SIMDS][SPARK_ATTENTION_DIM];
	const uint part = group.x, head = group.y, row = group.z;
	const uint kv_head = head / (p.heads / p.kv_heads), sequence = sequence_of_row[row], length = positions[row] + 1u;
	const uint begin = part * p.partition_keys, end = min(length, begin + p.partition_keys);
	float q[SPARK_ATTENTION_LANE_DIMS], acc[SPARK_ATTENTION_LANE_DIMS];
	float running_max = -INFINITY, running_sum = 0.0f;
	SparkLoad8(query + (ulong)row * p.query_stride + head * SPARK_ATTENTION_DIM + lane * SPARK_ATTENTION_LANE_DIMS, q);
	for ( uint d = 0u; d < SPARK_ATTENTION_LANE_DIMS; d++ )
		acc[d] = 0.0f;
	for ( uint key = begin + simd; key < end; key += SPARK_ATTENTION_SIMDS )
	{
		device const ushort *slot = SparkKvSlot(p, pool, page_table, sequence, key);
		float k[SPARK_ATTENTION_LANE_DIMS], v[SPARK_ATTENTION_LANE_DIMS], score = 0.0f;
		if ( slot == (device const ushort *)0 )
			continue;
		SparkLoad8(slot + kv_head * SPARK_ATTENTION_DIM + lane * SPARK_ATTENTION_LANE_DIMS, k);
		SparkLoad8(slot + (p.kv_heads + kv_head) * SPARK_ATTENTION_DIM + lane * SPARK_ATTENTION_LANE_DIMS, v);
		for ( uint d = 0u; d < SPARK_ATTENTION_LANE_DIMS; d++ )
			score += q[d] * k[d];
		score = simd_sum(score) * p.scale;
		const float next_max = max(running_max, score);
		const float correction = exp(running_max - next_max), weight = exp(score - next_max);
		running_sum = running_sum * correction + weight;
		for ( uint d = 0u; d < SPARK_ATTENTION_LANE_DIMS; d++ )
			acc[d] = acc[d] * correction + weight * v[d];
		running_max = next_max;
	}
	if ( lane == 0u )
	{
		shared_max[simd] = running_max;
		shared_sum[simd] = running_sum;
	}
	for ( uint d = 0u; d < SPARK_ATTENTION_LANE_DIMS; d++ )
		shared_out[simd][lane * SPARK_ATTENTION_LANE_DIMS + d] = acc[d];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( simd != 0u )
		return;
	float total_max = -INFINITY, total_sum = 0.0f;
	for ( uint s = 0u; s < SPARK_ATTENTION_SIMDS; s++ )
		total_max = max(total_max, shared_max[s]);
	float out[SPARK_ATTENTION_LANE_DIMS];
	for ( uint d = 0u; d < SPARK_ATTENTION_LANE_DIMS; d++ )
		out[d] = 0.0f;
	for ( uint s = 0u; s < SPARK_ATTENTION_SIMDS; s++ )
	{
		const float factor = shared_max[s] == -INFINITY ? 0.0f : exp(shared_max[s] - total_max);
		total_sum += shared_sum[s] * factor;
		for ( uint d = 0u; d < SPARK_ATTENTION_LANE_DIMS; d++ )
			out[d] += shared_out[s][lane * SPARK_ATTENTION_LANE_DIMS + d] * factor;
	}
	device float *record = partial + (((ulong)row * p.heads + head) * p.partitions + part) * (SPARK_ATTENTION_DIM + 2u);
	if ( lane == 0u )
	{
		record[0] = total_max;
		record[1] = total_sum;
	}
	for ( uint d = 0u; d < SPARK_ATTENTION_LANE_DIMS; d++ )
		record[2u + lane * SPARK_ATTENTION_LANE_DIMS + d] = out[d];
}

kernel void spark_attention_combine(constant SparkAttentionParams &p [[buffer(0)]], device const float *partial [[buffer(1)]],
	device const ushort *gate [[buffer(2)]], device const uint *positions [[buffer(3)]], device ushort *output [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint head = group.x, row = group.y, length = positions[row] + 1u;
	const uint used = min(p.partitions, (length + p.partition_keys - 1u) / p.partition_keys);
	device const float *records = partial + ((ulong)row * p.heads + head) * p.partitions * (SPARK_ATTENTION_DIM + 2u);
	float total_max = -INFINITY;
	for ( uint part = 0u; part < used; part++ )
		total_max = max(total_max, records[part * (SPARK_ATTENTION_DIM + 2u)]);
	for ( uint d = tid; d < SPARK_ATTENTION_DIM; d += threads )
	{
		float sum = 0.0f, value = 0.0f;
		for ( uint part = 0u; part < used; part++ )
		{
			device const float *record = records + part * (SPARK_ATTENTION_DIM + 2u);
			const float factor = record[0] == -INFINITY ? 0.0f : exp(record[0] - total_max);
			sum += record[1] * factor;
			value += record[2u + d] * factor;
		}
		const float attended = SparkMathRoundBf16(value / sum);
		const float g = SparkMathBf16ToFloat(gate[(ulong)row * p.gate_stride + p.gate_offset + head * p.gate_head_stride + d]);
		output[(ulong)row * p.output_stride + head * SPARK_ATTENTION_DIM + d] = SparkMathFloatToBf16(attended * SparkMathSigmoid(g));
	}
}

kernel void spark_attention_prefill(constant SparkAttentionParams &p [[buffer(0)]], device const ushort *query [[buffer(1)]],
	device const ushort *pool [[buffer(2)]], device const uint *page_table [[buffer(3)]], device const uint *sequence_of_row [[buffer(4)]],
	device const uint *positions [[buffer(5)]], device float *partial [[buffer(6)]],
	uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup ushort tile_keys[SPARK_ATTENTION_TILE * SPARK_ATTENTION_DIM];
	threadgroup ushort tile_values[SPARK_ATTENTION_TILE * SPARK_ATTENTION_DIM];
	const uint part = group.x, head = group.y;
	const uint kv_head = head / (p.heads / p.kv_heads);
	const uint first_row = p.row_offset + group.z * SPARK_ATTENTION_SIMDS, row = first_row + simd;
	const uint last_row = min(p.rows, first_row + SPARK_ATTENTION_SIMDS) - 1u;
	const uint sequence = sequence_of_row[first_row];
	const uint valid = row < p.rows && sequence_of_row[min(row, p.rows - 1u)] == sequence ? 1u : 0u;
	const uint length = valid != 0u ? positions[row] + 1u : 0u;
	uint span_end = 0u;
	for ( uint r = first_row; r <= last_row; r++ )
		if ( sequence_of_row[r] == sequence )
			span_end = max(span_end, positions[r] + 1u);
	const uint begin = part * p.partition_keys, end = min(span_end, begin + p.partition_keys);
	float q[SPARK_ATTENTION_LANE_DIMS], acc[SPARK_ATTENTION_LANE_DIMS];
	float running_max = -INFINITY, running_sum = 0.0f;
	for ( uint d = 0u; d < SPARK_ATTENTION_LANE_DIMS; d++ )
	{
		acc[d] = 0.0f;
		q[d] = 0.0f;
	}
	if ( valid != 0u )
		SparkLoad8(query + (ulong)row * p.query_stride + head * SPARK_ATTENTION_DIM + lane * SPARK_ATTENTION_LANE_DIMS, q);
	for ( uint tile = begin; tile < end; tile += SPARK_ATTENTION_TILE )
	{
		const uint count = min(SPARK_ATTENTION_TILE, end - tile);
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( uint chunk = tid; chunk < count * (SPARK_ATTENTION_DIM / 8u); chunk += SPARK_ATTENTION_SIMDS * 32u )
		{
			const uint key = chunk / (SPARK_ATTENTION_DIM / 8u), offset = (chunk % (SPARK_ATTENTION_DIM / 8u)) * 8u;
			device const ushort *slot = SparkKvSlot(p, pool, page_table, sequence, tile + key);
			uint4 k_bits = uint4(0u), v_bits = uint4(0u);
			if ( slot != (device const ushort *)0 )
			{
				k_bits = *(device const uint4 *)(slot + kv_head * SPARK_ATTENTION_DIM + offset);
				v_bits = *(device const uint4 *)(slot + (p.kv_heads + kv_head) * SPARK_ATTENTION_DIM + offset);
			}
			*(threadgroup uint4 *)(tile_keys + key * SPARK_ATTENTION_DIM + offset) = k_bits;
			*(threadgroup uint4 *)(tile_values + key * SPARK_ATTENTION_DIM + offset) = v_bits;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( uint key = 0u; key < count && tile + key < length; key++ )
		{
			float k[SPARK_ATTENTION_LANE_DIMS], v[SPARK_ATTENTION_LANE_DIMS], score = 0.0f;
			SparkLoad8(tile_keys + key * SPARK_ATTENTION_DIM + lane * SPARK_ATTENTION_LANE_DIMS, k);
			SparkLoad8(tile_values + key * SPARK_ATTENTION_DIM + lane * SPARK_ATTENTION_LANE_DIMS, v);
			for ( uint d = 0u; d < SPARK_ATTENTION_LANE_DIMS; d++ )
				score += q[d] * k[d];
			score = simd_sum(score) * p.scale;
			const float next_max = max(running_max, score);
			const float correction = exp(running_max - next_max), weight = exp(score - next_max);
			running_sum = running_sum * correction + weight;
			for ( uint d = 0u; d < SPARK_ATTENTION_LANE_DIMS; d++ )
				acc[d] = acc[d] * correction + weight * v[d];
			running_max = next_max;
		}
	}
	if ( valid == 0u )
		return;
	device float *record = partial + (((ulong)row * p.heads + head) * p.partitions + part) * (SPARK_ATTENTION_DIM + 2u);
	if ( lane == 0u )
	{
		record[0] = running_max;
		record[1] = running_sum;
	}
	for ( uint d = 0u; d < SPARK_ATTENTION_LANE_DIMS; d++ )
		record[2u + lane * SPARK_ATTENTION_LANE_DIMS + d] = acc[d];
}
