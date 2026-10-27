#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "inference/kernels/metal/common.metal"

using namespace mpp::tensor_ops;

#define SPARK_ATTENTION_SIMDS 8u
#define SPARK_ATTENTION_BLOCK_KEYS 64u
#define SPARK_ATTENTION_BLOCK_SIMDS 4u

static inline device const ushort *SparkKvSlot(constant SparkAttentionParams &p, device const ushort *pool, device const uint *page_table,
	uint sequence, uint position)
{
	const uint page = page_table[(ulong)sequence * p.pages_per_sequence + position / p.page_slots];
	if ( page >= p.pool_pages )
		return (device const ushort *)0;
	return pool + ((ulong)page * p.page_slots + position % p.page_slots) * (2u * p.kv_heads * p.head_dimension);
}

static inline uint SparkAttentionFirstKey(constant SparkAttentionParams &p, uint position)
{
	return p.window != 0u && position + 1u > p.window ? position + 1u - p.window : 0u;
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

template <uint D>
kernel void spark_attention_decode(constant SparkAttentionParams &p [[buffer(0)]], device const ushort *query [[buffer(1)]],
	device const ushort *pool [[buffer(2)]], device const uint *page_table [[buffer(3)]], device const uint *sequence_of_row [[buffer(4)]],
	device const uint *positions [[buffer(5)]], device float *partial [[buffer(6)]],
	uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	constexpr uint LANE = D / 32u;
	threadgroup float shared_max[SPARK_ATTENTION_SIMDS];
	threadgroup float shared_sum[SPARK_ATTENTION_SIMDS];
	threadgroup float shared_out[SPARK_ATTENTION_SIMDS][D];
	const uint part = group.x, head = group.y, row = group.z;
	const uint kv_head = head / (p.heads / p.kv_heads), sequence = sequence_of_row[row], length = positions[row] + 1u;
	const uint begin = max(part * p.partition_keys, SparkAttentionFirstKey(p, positions[row])), end = min(length, part * p.partition_keys + p.partition_keys);
	float q[LANE], acc[LANE];
	float running_max = -INFINITY, running_sum = 0.0f;
	for ( uint chunk = 0u; chunk < LANE; chunk += 8u )
		SparkLoad8(query + (ulong)row * p.query_stride + head * D + lane * LANE + chunk, q + chunk);
	for ( uint d = 0u; d < LANE; d++ )
		acc[d] = 0.0f;
	for ( uint key = begin + simd; key < end; key += SPARK_ATTENTION_SIMDS )
	{
		device const ushort *slot = SparkKvSlot(p, pool, page_table, sequence, key);
		float k[LANE], v[LANE], score = 0.0f;
		if ( slot == (device const ushort *)0 )
			continue;
		for ( uint chunk = 0u; chunk < LANE; chunk += 8u )
		{
			SparkLoad8(slot + kv_head * D + lane * LANE + chunk, k + chunk);
			SparkLoad8(slot + (p.kv_heads + kv_head) * D + lane * LANE + chunk, v + chunk);
		}
		for ( uint d = 0u; d < LANE; d++ )
			score += q[d] * k[d];
		score = simd_sum(score) * p.scale;
		const float next_max = max(running_max, score);
		const float correction = exp(running_max - next_max), weight = exp(score - next_max);
		running_sum = running_sum * correction + weight;
		for ( uint d = 0u; d < LANE; d++ )
			acc[d] = acc[d] * correction + weight * v[d];
		running_max = next_max;
	}
	if ( lane == 0u )
	{
		shared_max[simd] = running_max;
		shared_sum[simd] = running_sum;
	}
	for ( uint d = 0u; d < LANE; d++ )
		shared_out[simd][lane * LANE + d] = acc[d];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( simd != 0u )
		return;
	float total_max = -INFINITY, total_sum = 0.0f;
	for ( uint s = 0u; s < SPARK_ATTENTION_SIMDS; s++ )
		total_max = max(total_max, shared_max[s]);
	float out[LANE];
	for ( uint d = 0u; d < LANE; d++ )
		out[d] = 0.0f;
	for ( uint s = 0u; s < SPARK_ATTENTION_SIMDS; s++ )
	{
		const float factor = shared_max[s] == -INFINITY ? 0.0f : exp(shared_max[s] - total_max);
		total_sum += shared_sum[s] * factor;
		for ( uint d = 0u; d < LANE; d++ )
			out[d] += shared_out[s][lane * LANE + d] * factor;
	}
	device float *record = partial + (((ulong)row * p.heads + head) * p.partitions + part) * (D + 2u);
	if ( lane == 0u )
	{
		record[0] = total_max;
		record[1] = total_sum;
	}
	for ( uint d = 0u; d < LANE; d++ )
		record[2u + lane * LANE + d] = out[d];
}

template <uint D>
kernel void spark_attention_combine(constant SparkAttentionParams &p [[buffer(0)]], device const float *partial [[buffer(1)]],
	device const ushort *gate [[buffer(2)]], device const uint *positions [[buffer(3)]], device ushort *output [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint head = group.x, row = group.y, length = positions[row] + 1u;
	const uint used = min(p.partitions, (length + p.partition_keys - 1u) / p.partition_keys);
	device const float *records = partial + ((ulong)row * p.heads + head) * p.partitions * (D + 2u);
	float total_max = -INFINITY;
	for ( uint part = 0u; part < used; part++ )
		total_max = max(total_max, records[part * (D + 2u)]);
	for ( uint d = tid; d < D; d += threads )
	{
		float sum = 0.0f, value = 0.0f;
		for ( uint part = 0u; part < used; part++ )
		{
			device const float *record = records + part * (D + 2u);
			const float factor = record[0] == -INFINITY ? 0.0f : exp(record[0] - total_max);
			sum += record[1] * factor;
			value += record[2u + d] * factor;
		}
		const float attended = SparkMathRoundBf16(value / sum);
		const float g = p.output_gate != 0u ? SparkMathSigmoid(SparkMathBf16ToFloat(gate[(ulong)row * p.gate_stride + p.gate_offset + head * p.gate_head_stride + d])) : 1.0f;
		output[(ulong)row * p.output_stride + head * D + d] = SparkMathFloatToBf16(p.output_gate != 0u ? attended * g : attended);
	}
}

template <uint D>
kernel void spark_attention_prefill_blocks(constant SparkAttentionParams &p [[buffer(0)]], device bfloat *query [[buffer(1)]],
	device bfloat *pool [[buffer(2)]], device const uint *page_table [[buffer(3)]], device const uint *sequence_of_row [[buffer(4)]],
	device const uint *positions [[buffer(5)]], device float *partial [[buffer(6)]],
	uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	constexpr uint ROWS = SPARK_ATTENTION_BLOCK_ROWS(D);
	threadgroup float block[ROWS * SPARK_ATTENTION_BLOCK_KEYS];
	threadgroup float row_max[ROWS];
	threadgroup float row_sum[ROWS];
	threadgroup float row_scale[ROWS];
	const uint part = group.x, head = group.y, kv_head = head / (p.heads / p.kv_heads);
	const uint first_row = p.row_offset + group.z * ROWS;
	if ( first_row >= p.rows )
		return;
	const uint count = min(ROWS, p.rows - first_row), sequence = sequence_of_row[first_row];
	const uint first_position = positions[first_row], span_end = positions[first_row + count - 1u] + 1u;
	const uint begin = max(part * p.partition_keys, SparkAttentionFirstKey(p, first_position) / SPARK_ATTENTION_BLOCK_KEYS * SPARK_ATTENTION_BLOCK_KEYS);
	const uint end = min(span_end, part * p.partition_keys + p.partition_keys);
	const uint slot_stride = 2u * p.kv_heads * D;
	const float scale = p.scale;
	constexpr auto score_descriptor = matmul2d_descriptor(ROWS, SPARK_ATTENTION_BLOCK_KEYS, D, false, true, false);
	constexpr auto value_descriptor = matmul2d_descriptor(ROWS, D, static_cast<int>(dynamic_extent), false, false,
		false, matmul2d_descriptor::mode::multiply_accumulate);
	matmul2d<score_descriptor, execution_simdgroups<SPARK_ATTENTION_BLOCK_SIMDS>> score_operation;
	matmul2d<value_descriptor, execution_simdgroups<SPARK_ATTENTION_BLOCK_SIMDS>> value_operation;
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> queries(query + (ulong)first_row * p.query_stride + head * D,
		dextents<int32_t, 2>(D, count), array<int32_t, 2>{ 1, int(p.query_stride) });
	tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> weights(block, dextents<int32_t, 2>(SPARK_ATTENTION_BLOCK_KEYS, ROWS),
		array<int32_t, 2>{ 1, int(SPARK_ATTENTION_BLOCK_KEYS) });
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> first_values(pool, dextents<int32_t, 2>(D, SPARK_ATTENTION_BLOCK_KEYS),
		array<int32_t, 2>{ 1, int(slot_stride) });
	auto attended = value_operation.template get_destination_cooperative_tensor<decltype(weights), decltype(first_values), float>();
	for ( uint16_t i = 0; i < attended.get_capacity(); ++i )
		if ( attended.is_valid_element(i) )
			attended[i] = 0.0f;
	if ( simd == 0u && lane < ROWS )
	{
		row_max[lane] = -INFINITY;
		row_sum[lane] = 0.0f;
	}
	for ( uint tile = begin; tile < end; tile += SPARK_ATTENTION_BLOCK_KEYS )
	{
		const uint keys = min(SPARK_ATTENTION_BLOCK_KEYS, end - tile);
		const uint page = page_table[(ulong)sequence * p.pages_per_sequence + tile / p.page_slots];
		device bfloat *slot = pool + ((ulong)min(page, p.pool_pages - 1u) * p.page_slots + tile % p.page_slots) * slot_stride;
		tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> key_block(slot + kv_head * D, dextents<int32_t, 2>(D, keys),
			array<int32_t, 2>{ 1, int(slot_stride) });
		tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> value_block(slot + (p.kv_heads + kv_head) * D, dextents<int32_t, 2>(D, keys),
			array<int32_t, 2>{ 1, int(slot_stride) });
		tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tile_weights(block, dextents<int32_t, 2>(keys, ROWS),
			array<int32_t, 2>{ 1, int(SPARK_ATTENTION_BLOCK_KEYS) });
		auto scores = score_operation.template get_destination_cooperative_tensor<decltype(queries), decltype(key_block), float>();
		threadgroup_barrier(mem_flags::mem_threadgroup);
		score_operation.run(queries, key_block, scores);
		for ( uint16_t i = 0; i < scores.get_capacity(); ++i )
			if ( scores.is_valid_element(i) )
			{
				const auto index = scores.get_multidimensional_index(i);
				block[index[1] * SPARK_ATTENTION_BLOCK_KEYS + index[0]] = scores[i];
			}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( uint r = simd; r < ROWS; r += SPARK_ATTENTION_BLOCK_SIMDS )
		{
			const uint limit = r < count ? first_position + r + 1u : 0u, low = r < count ? SparkAttentionFirstKey(p, first_position + r) : 0u;
			const uint low_key = tile + lane, high_key = tile + lane + 32u;
			float low_value = low_key < limit && low_key >= low && lane < keys ? block[r * SPARK_ATTENTION_BLOCK_KEYS + lane] * scale : -INFINITY;
			float high_value = high_key < limit && high_key >= low && lane + 32u < keys ? block[r * SPARK_ATTENTION_BLOCK_KEYS + lane + 32u] * scale : -INFINITY;
			const float previous = row_max[r], next = max(previous, simd_max(max(low_value, high_value)));
			const float correction = next == -INFINITY ? 1.0f : exp(previous - next);
			low_value = next == -INFINITY ? 0.0f : exp(low_value - next);
			high_value = next == -INFINITY ? 0.0f : exp(high_value - next);
			const float total = simd_sum(low_value + high_value);
			block[r * SPARK_ATTENTION_BLOCK_KEYS + lane] = low_value;
			block[r * SPARK_ATTENTION_BLOCK_KEYS + lane + 32u] = high_value;
			if ( lane == 0u )
			{
				row_max[r] = next;
				row_sum[r] = row_sum[r] * correction + total;
				row_scale[r] = correction;
			}
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( uint16_t i = 0; i < attended.get_capacity(); ++i )
			if ( attended.is_valid_element(i) )
				attended[i] *= row_scale[attended.get_multidimensional_index(i)[1]];
		value_operation.run(tile_weights, value_block, attended);
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint16_t i = 0; i < attended.get_capacity(); ++i )
		if ( attended.is_valid_element(i) )
		{
			const auto index = attended.get_multidimensional_index(i);
			const uint r = uint(index[1]);
			if ( r < count && part * p.partition_keys < first_position + r + 1u )
				partial[(((ulong)(first_row + r) * p.heads + head) * p.partitions + part) * (D + 2u) + 2u + uint(index[0])] = attended[i];
		}
	if ( simd == 0u && lane < count && part * p.partition_keys < first_position + lane + 1u )
	{
		device float *record = partial + (((ulong)(first_row + lane) * p.heads + head) * p.partitions + part) * (D + 2u);
		record[0] = row_max[lane];
		record[1] = row_sum[lane];
	}
}

#define SPARK_ATTENTION_INSTANCES(dim) \
	template [[host_name("spark_attention_decode_d" #dim)]] kernel void spark_attention_decode<dim>(constant SparkAttentionParams &p [[buffer(0)]], \
		device const ushort *query [[buffer(1)]], device const ushort *pool [[buffer(2)]], device const uint *page_table [[buffer(3)]], \
		device const uint *sequence_of_row [[buffer(4)]], device const uint *positions [[buffer(5)]], device float *partial [[buffer(6)]], \
		uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]); \
	template [[host_name("spark_attention_combine_d" #dim)]] kernel void spark_attention_combine<dim>(constant SparkAttentionParams &p [[buffer(0)]], \
		device const float *partial [[buffer(1)]], device const ushort *gate [[buffer(2)]], device const uint *positions [[buffer(3)]], \
		device ushort *output [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
		uint2 group_size [[threads_per_threadgroup]]); \
	template [[host_name("spark_attention_prefill_blocks_d" #dim)]] kernel void spark_attention_prefill_blocks<dim>( \
		constant SparkAttentionParams &p [[buffer(0)]], device bfloat *query [[buffer(1)]], device bfloat *pool [[buffer(2)]], \
		device const uint *page_table [[buffer(3)]], device const uint *sequence_of_row [[buffer(4)]], device const uint *positions [[buffer(5)]], \
		device float *partial [[buffer(6)]], uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
		uint simd [[simdgroup_index_in_threadgroup]]);

SPARK_ATTENTION_INSTANCES(256)
SPARK_ATTENTION_INSTANCES(512)
