#include "inference/kernels/metal/common.metal"

static inline int SparkNgramToken(device const uint *token_ids, device const int *token_map, uint row, bool mapped)
{
	return mapped ? token_map[token_ids[row]] : int(token_ids[row]);
}

template <bool TOKEN_MAP>
kernel void spark_ngram_hash(constant SparkNgramParams &p [[buffer(0)]], device const uint *token_ids [[buffer(1)]],
	device const uint *positions [[buffer(2)]], device const uint *row_begin [[buffer(3)]], device const uint *state_index [[buffer(4)]],
	device const int *history [[buffer(5)]], device const long *multipliers [[buffer(6)]], device const long *primes [[buffer(7)]],
	device const long *offsets [[buffer(8)]], device long *ids [[buffer(9)]], device const int *token_map [[buffer(10)]], uint row [[thread_position_in_grid]])
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
			source = shift <= into_wave ? SparkNgramToken(token_ids, token_map, begin + into_wave - shift, TOKEN_MAP) : carried[shift - into_wave - 1u];
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

#define SPARK_NGRAM_HASH_INSTANCE(name, mapped) \
	template [[host_name(name)]] kernel void spark_ngram_hash<mapped>(constant SparkNgramParams &p [[buffer(0)]], device const uint *token_ids [[buffer(1)]], \
		device const uint *positions [[buffer(2)]], device const uint *row_begin [[buffer(3)]], device const uint *state_index [[buffer(4)]], \
		device const int *history [[buffer(5)]], device const long *multipliers [[buffer(6)]], device const long *primes [[buffer(7)]], \
		device const long *offsets [[buffer(8)]], device long *ids [[buffer(9)]], device const int *token_map [[buffer(10)]], uint row [[thread_position_in_grid]]);

SPARK_NGRAM_HASH_INSTANCE("spark_ngram_hash", false)
SPARK_NGRAM_HASH_INSTANCE("spark_ngram_hash_mapped", true)

template <typename S>
kernel void spark_ngram_gather_fp8(constant SparkNgramParams &p [[buffer(0)]], device const long *ids [[buffer(1)]],
	device const uchar *payload [[buffer(2)]], device const S *scale [[buffer(3)]], device ushort *embed [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.x, column = group.y, columns = p.orders * p.heads;
	const long id = ids[(ulong)row * columns + column];
	device ushort *target = embed + ((ulong)row * columns + column) * p.head_dimension;
	const bool owned = id >= long(p.first_row) && id < long(p.first_row + p.table_rows);
	const ulong local = owned ? ulong(id - long(p.first_row)) : 0ul;
	constexpr bool block_scaled = is_same<S, uchar>::value;
	const float factor = block_scaled ? 0.0f : float(scale[0]);
	for ( uint c = tid; c < p.head_dimension; c += group_size.x )
	{
		const float block = block_scaled ? SparkMathUe8m0ToFloat(uchar(scale[local * (p.head_dimension / SPARK_MXFP8_GROUP) + c / SPARK_MXFP8_GROUP])) : factor;
		target[c] = owned ? SparkMathFloatToBf16(SparkMathE4m3ToFloat(payload[local * p.head_dimension + c]) * block) : ushort(0u);
	}
}

#define SPARK_NGRAM_GATHER_INSTANCE(name, type) \
	template [[host_name(name)]] kernel void spark_ngram_gather_fp8<type>(constant SparkNgramParams &p [[buffer(0)]], device const long *ids [[buffer(1)]], \
		device const uchar *payload [[buffer(2)]], device const type *scale [[buffer(3)]], device ushort *embed [[buffer(4)]], \
		uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);

SPARK_NGRAM_GATHER_INSTANCE("spark_ngram_gather_fp8", float)
SPARK_NGRAM_GATHER_INSTANCE("spark_ngram_gather_e8m0", uchar)

template <bool TOKEN_MAP>
kernel void spark_ngram_history(constant SparkNgramParams &p [[buffer(0)]], device const uint *token_ids [[buffer(1)]],
	device const uint *row_begin [[buffer(2)]], device const uint *state_index [[buffer(3)]], device int *history [[buffer(4)]],
	device const uint *row_map [[buffer(5)]], device const int *token_map [[buffer(6)]], uint sequence [[thread_position_in_grid]])
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
	{
		const uint row = begin + into_wave - shift;
		updated[shift] = shift <= into_wave ? SparkNgramToken(token_ids, token_map, p.mapped != 0u ? row_map[row] : row, TOKEN_MAP) : carried[shift - into_wave - 1u];
	}
	for ( uint shift = 0u; shift < p.orders; shift++ )
		slot[shift] = updated[shift];
}

#define SPARK_NGRAM_HISTORY_INSTANCE(name, mapped) \
	template [[host_name(name)]] kernel void spark_ngram_history<mapped>(constant SparkNgramParams &p [[buffer(0)]], device const uint *token_ids [[buffer(1)]], \
		device const uint *row_begin [[buffer(2)]], device const uint *state_index [[buffer(3)]], device int *history [[buffer(4)]], \
		device const uint *row_map [[buffer(5)]], device const int *token_map [[buffer(6)]], uint sequence [[thread_position_in_grid]]);

SPARK_NGRAM_HISTORY_INSTANCE("spark_ngram_history", false)
SPARK_NGRAM_HISTORY_INSTANCE("spark_ngram_history_mapped", true)
