#include "inference/kernels/metal/common.metal"

kernel void spark_ngram_hash(constant SparkNgramParams &p [[buffer(0)]], device const uint *token_ids [[buffer(1)]],
	device const uint *positions [[buffer(2)]], device const uint *row_begin [[buffer(3)]], device const uint *state_index [[buffer(4)]],
	device const int *history [[buffer(5)]], device const long *multipliers [[buffer(6)]], device const long *primes [[buffer(7)]],
	device const long *offsets [[buffer(8)]], device long *ids [[buffer(9)]], uint row [[thread_position_in_grid]])
{
	if ( row >= p.rows )
		return;
	uint lane = 0u;
	while ( lane + 1u < p.sequences && row_begin[lane + 1u] <= row )
		lane++;
	const uint begin = row_begin[lane], position = positions[row], into_wave = row - begin;
	device const int *carried = history + (ulong)state_index[lane] * p.history_stride;
	long tokens[SPARK_NGRAM_MAX_ORDERS + 1u];
	bool blocked = false;
	for ( uint shift = 0u; shift <= p.orders; shift++ )
	{
		int source = -1;
		if ( position >= shift )
			source = shift <= into_wave ? int(token_ids[begin + into_wave - shift]) : carried[shift - into_wave - 1u];
		if ( position < shift || (shift >= p.block_from && source == p.block_token) )
			blocked = true;
		tokens[shift] = blocked ? long(p.pad_token) : long(source);
	}
	long rolling = tokens[0] * multipliers[0];
	for ( uint order = 1u; order <= p.orders; order++ )
	{
		rolling ^= tokens[order] * multipliers[order];
		for ( uint head = 0u; head < p.heads; head++ )
		{
			const uint column = (order - 1u) * p.heads + head;
			ids[(ulong)row * p.orders * p.heads + column] = rolling % primes[column] + offsets[column];
		}
	}
}

kernel void spark_ngram_gather_fp8(constant SparkNgramParams &p [[buffer(0)]], device const long *ids [[buffer(1)]],
	device const uchar *payload [[buffer(2)]], device const float *scale [[buffer(3)]], device ushort *embed [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.x, column = group.y, columns = p.orders * p.heads;
	const long id = ids[(ulong)row * columns + column];
	device ushort *target = embed + ((ulong)row * columns + column) * p.head_dimension;
	const bool owned = id >= long(p.first_row) && id < long(p.first_row + p.table_rows);
	const ulong local = owned ? ulong(id - long(p.first_row)) : 0ul;
	const float factor = scale[0];
	for ( uint c = tid; c < p.head_dimension; c += group_size.x )
		target[c] = owned ? SparkMathFloatToBf16(SparkMathE4m3ToFloat(payload[local * p.head_dimension + c]) * factor) : ushort(0u);
}

kernel void spark_ngram_history(constant SparkNgramParams &p [[buffer(0)]], device const uint *token_ids [[buffer(1)]],
	device const uint *row_begin [[buffer(2)]], device const uint *state_index [[buffer(3)]], device int *history [[buffer(4)]],
	uint sequence [[thread_position_in_grid]])
{
	if ( p.commit == 0u || sequence >= p.sequences )
		return;
	const uint begin = row_begin[sequence], count = row_begin[sequence + 1u] - begin;
	if ( count == 0u )
		return;
	const uint into_wave = count - 1u;
	device int *slot = history + (ulong)state_index[sequence] * p.history_stride;
	int carried[SPARK_NGRAM_MAX_ORDERS], updated[SPARK_NGRAM_MAX_ORDERS];
	for ( uint shift = 0u; shift < p.orders; shift++ )
		carried[shift] = slot[shift];
	for ( uint shift = 0u; shift < p.orders; shift++ )
		updated[shift] = shift <= into_wave ? int(token_ids[begin + into_wave - shift]) : carried[shift - into_wave - 1u];
	for ( uint shift = 0u; shift < p.orders; shift++ )
		slot[shift] = updated[shift];
}
