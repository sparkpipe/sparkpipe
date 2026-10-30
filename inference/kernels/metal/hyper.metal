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

template <uint MIXES>
kernel void spark_hc_mix(constant SparkHcParams &p [[buffer(0)]], device const ushort *streams [[buffer(1)]], device const float *fn [[buffer(2)]],
	device const float *base [[buffer(3)]], device const float *scale [[buffer(4)]], device float *post [[buffer(5)]], device ushort *mixed [[buffer(6)]],
	uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float reduction[(MIXES + 1u) * 32u];
	threadgroup float weights[SPARK_HC_MAX_STREAMS];
	const uint flat = p.streams * p.width;
	device const ushort *source = streams + (ulong)row * flat;
	float dot[MIXES + 1u];
	for ( uint m = 0u; m <= MIXES; m++ )
		dot[m] = 0.0f;
	for ( uint k = tid; k < flat; k += threads )
	{
		const float x = SparkMathBf16ToFloat(source[k]);
		for ( uint m = 0u; m < MIXES; m++ )
			dot[m] = fma(x, fn[(ulong)m * flat + k], dot[m]);
		dot[MIXES] = fma(x, x, dot[MIXES]);
	}
	for ( uint m = 0u; m <= MIXES; m++ )
	{
		const float value = simd_sum(dot[m]);
		if ( lane == 0u )
			reduction[m * 32u + simd] = value;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( tid == 0u )
	{
		float total[MIXES + 1u];
		for ( uint m = 0u; m <= MIXES; m++ )
		{
			total[m] = 0.0f;
			for ( uint w = 0u; w < simds; w++ )
				total[m] += reduction[m * 32u + w];
		}
		const float inverse = SparkMathRmsScale(total[MIXES], float(flat), p.norm_epsilon);
		for ( uint s = 0u; s < p.streams; s++ )
		{
			weights[s] = SparkMathSigmoid(total[s] * inverse * scale[0] + base[s]) + p.hc_epsilon;
			if ( MIXES > p.streams )
				post[(ulong)row * p.streams + s] = SparkMathScaledSigmoid(p.post_magnitude, total[p.streams + s] * inverse * scale[1] + base[p.streams + s]) +
					p.hc_epsilon;
		}
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint d = tid; d < p.width; d += threads )
	{
		float total = 0.0f;
		for ( uint s = 0u; s < p.streams; s++ )
			total = fma(weights[s], SparkMathBf16ToFloat(source[s * p.width + d]), total);
		mixed[(ulong)row * p.width + d] = SparkMathFloatToBf16(total);
	}
}

template [[host_name("spark_hc_mix8")]] kernel void spark_hc_mix<8u>(constant SparkHcParams &p [[buffer(0)]], device const ushort *streams [[buffer(1)]],
	device const float *fn [[buffer(2)]], device const float *base [[buffer(3)]], device const float *scale [[buffer(4)]], device float *post [[buffer(5)]],
	device ushort *mixed [[buffer(6)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
	uint simds [[simdgroups_per_threadgroup]]);
template [[host_name("spark_hc_mix4")]] kernel void spark_hc_mix<4u>(constant SparkHcParams &p [[buffer(0)]], device const ushort *streams [[buffer(1)]],
	device const float *fn [[buffer(2)]], device const float *base [[buffer(3)]], device const float *scale [[buffer(4)]], device float *post [[buffer(5)]],
	device ushort *mixed [[buffer(6)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
	uint simds [[simdgroups_per_threadgroup]]);

kernel void spark_hc_post(constant SparkHcParams &p [[buffer(0)]], device ushort *streams [[buffer(1)]], device const float *post [[buffer(2)]],
	device const ushort *output [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, index = group.x * group_size.x + tid;
	if ( index >= p.streams * p.width )
		return;
	const ulong at = (ulong)row * p.streams * p.width + index;
	streams[at] = SparkMathFloatToBf16(fma(post[(ulong)row * p.streams + index / p.width], SparkMathBf16ToFloat(output[(ulong)row * p.width + index % p.width]),
		SparkMathBf16ToFloat(streams[at])));
}
