#include "inference/kernels/metal/common.metal"

kernel void spark_hyper_expand(constant SparkHyperParams &p [[buffer(0)]], device const ushort *hidden [[buffer(1)]],
	device ushort *streams [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, index = group.x * group_size.x + tid;
	if ( index >= p.streams * p.width )
		return;
	streams[(ulong)row * p.streams * p.width + index] = hidden[(ulong)row * p.width + index % p.width];
}

kernel void spark_group_rmsnorm(constant SparkHyperParams &p [[buffer(0)]], device const ushort *input [[buffer(1)]],
	device const float *weight [[buffer(2)]], device ushort *output [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	const uint stream = group.x, row = group.y;
	const ulong base = ((ulong)row * p.streams + stream) * p.width;
	device const float *gain = weight + (ulong)stream * p.width;
	float total = 0.0f;
	for ( uint i = tid; i < p.width; i += group_size.x )
	{
		const float value = SparkMathBf16ToFloat(input[base + i]);
		total += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float scale = SparkMathRmsScale(total, float(p.width), p.epsilon);
	for ( uint i = tid; i < p.width; i += group_size.x )
		output[base + i] = SparkMathFloatToBf16(SparkMathBf16ToFloat(input[base + i]) * scale * gain[i]);
}

kernel void spark_hyper_gate(constant SparkHyperParams &p [[buffer(0)]], device ushort *values [[buffer(1)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, index = group.x * group_size.x + tid;
	if ( index >= p.width )
		return;
	const ulong at = (ulong)row * p.width + index;
	values[at] = SparkMathFloatToBf16(SparkMathHyperGate(SparkMathBf16ToFloat(values[at]), p.divisor));
}

kernel void spark_hyper_mix(constant SparkHyperParams &p [[buffer(0)]], device const ushort *up [[buffer(1)]],
	device const ushort *normed [[buffer(2)]], device ushort *mixed [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, element = group.x * group_size.x + tid;
	if ( element >= p.width )
		return;
	const ulong base = (ulong)row * p.streams * p.width;
	float total = 0.0f;
	for ( uint stream = 0u; stream < p.streams; stream++ )
	{
		const ulong at = base + (ulong)stream * p.width + element;
		const float gate = SparkMathHyperMixGate(SparkMathBf16ToFloat(up[at]));
		total += SparkMathHyperMixTerm(gate, SparkMathBf16ToFloat(normed[at]));
	}
	mixed[(ulong)row * p.width + element] = SparkMathFloatToBf16(total / float(p.streams));
}

kernel void spark_hyper_inject_weights(constant SparkHyperParams &p [[buffer(0)]], device const ushort *normed [[buffer(1)]],
	device const ushort *weight [[buffer(2)]], device float *inject [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	const uint total_width = p.streams * p.width;
	device const ushort *source = normed + (ulong)row * total_width;
	float sums[SPARK_HYPER_MAX_STREAMS];
	for ( uint stream = 0u; stream < p.streams; stream++ )
		sums[stream] = 0.0f;
	for ( uint i = tid; i < total_width; i += threads )
	{
		const float value = SparkMathBf16ToFloat(source[i]);
		for ( uint stream = 0u; stream < p.streams; stream++ )
			sums[stream] = fma(value, SparkMathBf16ToFloat(weight[(ulong)stream * total_width + i]), sums[stream]);
	}
	for ( uint stream = 0u; stream < p.streams; stream++ )
	{
		const float logit = SparkMathRoundBf16(SparkGroupSum(sums[stream], scratch, lane, simd, simds));
		if ( tid == 0u )
			inject[(ulong)row * p.streams + stream] = SparkMathHyperInjectWeight(logit, float(p.streams));
	}
}

kernel void spark_hyper_inject(constant SparkHyperParams &p [[buffer(0)]], device ushort *streams [[buffer(1)]],
	device const ushort *output [[buffer(2)]], device const float *inject [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, index = group.x * group_size.x + tid;
	if ( index >= p.streams * p.width )
		return;
	const ulong at = (ulong)row * p.streams * p.width + index;
	const float injected = SparkMathHyperInjected(SparkMathBf16ToFloat(output[(ulong)row * p.width + index % p.width]),
		inject[(ulong)row * p.streams + index / p.width]);
	streams[at] = SparkMathFloatToBf16(SparkMathBf16ToFloat(streams[at]) + injected);
}

kernel void spark_hyper_key_gate(constant SparkHyperParams &p [[buffer(0)]], device const ushort *key [[buffer(1)]],
	device const ushort *query [[buffer(2)]], device const ushort *value [[buffer(3)]], device ushort *gated [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	const uint stream = group.x, row = group.y;
	const ulong base = ((ulong)row * p.streams + stream) * p.width;
	float total = 0.0f;
	for ( uint i = tid; i < p.width; i += group_size.x )
		total += SparkMathRoundBf16(SparkMathBf16ToFloat(key[base + i]) * SparkMathBf16ToFloat(query[base + i]));
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float gate = SparkMathHyperKeyGate(total, p.width);
	for ( uint i = tid; i < p.width; i += group_size.x )
		gated[base + i] = SparkMathFloatToBf16(gate * SparkMathBf16ToFloat(value[(ulong)row * p.width + i]));
}

kernel void spark_hyper_add(constant SparkHyperParams &p [[buffer(0)]], device ushort *streams [[buffer(1)]],
	device const ushort *gated [[buffer(2)]], device const ushort *conv [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, index = group.x * group_size.x + tid;
	if ( index >= p.streams * p.width )
		return;
	const ulong at = (ulong)row * p.streams * p.width + index;
	const float added = SparkMathHyperAddend(SparkMathBf16ToFloat(gated[at]), SparkMathBf16ToFloat(conv[at]));
	streams[at] = SparkMathFloatToBf16(SparkMathBf16ToFloat(streams[at]) + added);
}
