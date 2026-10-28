#include "inference/kernels/metal/common.metal"

#define SPARK_CONV_TAPS_MAX 8u
#define SPARK_DELTA_COLUMNS 4u

kernel void spark_causal_conv_silu(constant SparkConvParams &p [[buffer(0)]], device ushort *window [[buffer(1)]],
	device const uint *state_index [[buffer(2)]], device const uint *row_begin [[buffer(3)]], device const ushort *input [[buffer(4)]],
	device const ushort *weight [[buffer(5)]], device ushort *output [[buffer(6)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint channel = group.x * threads + tid, sequence = group.y, taps_count = p.kernel_taps;
	const uint dilation = p.dilation != 0u ? p.dilation : 1u, span = SPARK_CONV_SPAN(taps_count, dilation);
	float taps[SPARK_CONV_SPAN_MAX];
	float weights[SPARK_CONV_TAPS_MAX];
	if ( channel >= p.channels || sequence >= p.sequences || taps_count > SPARK_CONV_TAPS_MAX || span > SPARK_CONV_SPAN_MAX )
		return;
	device ushort *slot = window + ((ulong)state_index[sequence] * p.channels + channel) * span;
	for ( uint entry = 0u; entry < span; entry++ )
		taps[entry] = SparkMathBf16ToFloat(slot[entry]);
	for ( uint tap = 0u; tap < taps_count; tap++ )
		weights[tap] = SparkMathBf16ToFloat(weight[channel * taps_count + tap]);
	for ( uint row = row_begin[sequence]; row < row_begin[sequence + 1u]; row++ )
	{
		float total = 0.0f;
		for ( uint entry = 0u; entry + 1u < span; entry++ )
			taps[entry] = taps[entry + 1u];
		taps[span - 1u] = SparkMathBf16ToFloat(input[(ulong)row * p.row_stride + channel]);
		for ( uint tap = 0u; tap < taps_count; tap++ )
			total += taps[tap * dilation] * weights[tap];
		total = SparkMathRoundBf16(total);
		output[(ulong)row * p.row_stride + channel] = SparkMathFloatToBf16(SparkMathSwishFast(total));
	}
	if ( p.commit != 0u )
		for ( uint entry = 0u; entry < span; entry++ )
			slot[entry] = SparkMathFloatToBf16(taps[entry]);
}

kernel void spark_gated_delta_prepare(constant SparkDeltaParams &p [[buffer(0)]], device const ushort *mix [[buffer(1)]],
	device const ushort *gates [[buffer(2)]], device const float *log_scale [[buffer(3)]], device const float *bias [[buffer(4)]],
	device float *query [[buffer(5)]], device float *key [[buffer(6)]], device float *decay [[buffer(7)]], device float *beta [[buffer(8)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	const uint threads = group_size.x;
	threadgroup float scratch[32];
	const uint head = group.x, row = group.y, D = p.key_dimension, per_key = p.value_heads / p.key_heads;
	device const ushort *row_mix = mix + (ulong)row * p.mix_stride;
	float q_sum = 0.0f, k_sum = 0.0f;
	for ( uint i = tid; i < D; i += threads )
	{
		const float q = SparkMathBf16ToFloat(row_mix[p.query_offset + head * D + i]);
		const float k = SparkMathBf16ToFloat(row_mix[p.key_offset + head * D + i]);
		q_sum += q * q;
		k_sum += k * k;
	}
	q_sum = SparkGroupSum(q_sum, scratch, lane, simd, simds);
	k_sum = SparkGroupSum(k_sum, scratch, lane, simd, simds);
	const float q_scale = rsqrt(q_sum + p.epsilon) * rsqrt(float(D)), k_scale = rsqrt(k_sum + p.epsilon);
	device float *q_out = query + ((ulong)row * p.key_heads + head) * D;
	device float *k_out = key + ((ulong)row * p.key_heads + head) * D;
	for ( uint i = tid; i < D; i += threads )
	{
		q_out[i] = SparkMathBf16ToFloat(row_mix[p.query_offset + head * D + i]) * q_scale;
		k_out[i] = SparkMathBf16ToFloat(row_mix[p.key_offset + head * D + i]) * k_scale;
	}
	if ( tid < per_key )
	{
		const uint value_head = head * per_key + tid;
		device const ushort *row_gates = gates + (ulong)row * p.gate_stride;
		const float shifted = SparkMathBf16ToFloat(row_gates[p.decay_offset + value_head]) + bias[value_head];
		decay[(ulong)row * p.value_heads + value_head] = SparkMathGatedDeltaDecay(log_scale[value_head],shifted);
		beta[(ulong)row * p.value_heads + value_head] = SparkMathSigmoid(SparkMathBf16ToFloat(row_gates[p.beta_offset + value_head]));
	}
}

kernel void spark_gated_delta_rule(constant SparkDeltaParams &p [[buffer(0)]], device float *state [[buffer(1)]],
	device const uint *state_index [[buffer(2)]], device const uint *row_begin [[buffer(3)]], device const float *query [[buffer(4)]],
	device const float *key [[buffer(5)]], device const float *decay [[buffer(6)]], device const float *beta [[buffer(7)]],
	device const ushort *mix [[buffer(8)]], device ushort *output [[buffer(9)]],
	uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	const uint column = group.x * SPARK_DELTA_COLUMNS + simd, head = group.y, sequence = group.z;
	const uint K = p.key_dimension, V = p.value_dimension, key_head = head / (p.value_heads / p.key_heads);
	if ( column >= V || sequence >= p.sequences || K != 128u )
		return;
	device float *s = state + (((ulong)state_index[sequence] * p.value_heads + head) * K) * V + column;
	float4 cell = float4(s[(lane * 4u) * V], s[(lane * 4u + 1u) * V], s[(lane * 4u + 2u) * V], s[(lane * 4u + 3u) * V]);
	for ( uint row = row_begin[sequence]; row < row_begin[sequence + 1u]; row++ )
	{
		const ulong vector = ((ulong)row * p.key_heads + key_head) * K + lane * 4u;
		const float4 k = *(device const float4 *)(key + vector);
		const float4 q = *(device const float4 *)(query + vector);
		const float factor = decay[(ulong)row * p.value_heads + head];
		const float strength = beta[(ulong)row * p.value_heads + head];
		const float value = SparkMathBf16ToFloat(mix[(ulong)row * p.mix_stride + p.value_offset + head * V + column]);
		cell *= factor;
		const float predicted = simd_sum(dot(cell, k));
		cell += (strength * (value - predicted)) * k;
		const float result = simd_sum(dot(cell, q));
		if ( lane == 0u )
			output[(ulong)row * p.output_stride + head * V + column] = SparkMathFloatToBf16(result);
	}
	if ( p.commit != 0u )
	{
		s[(lane * 4u) * V] = cell.x;
		s[(lane * 4u + 1u) * V] = cell.y;
		s[(lane * 4u + 2u) * V] = cell.z;
		s[(lane * 4u + 3u) * V] = cell.w;
	}
}
