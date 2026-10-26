#include "inference/kernels/metal/common.metal"

kernel void spark_rmsnorm_residual(constant SparkNormParams &p [[buffer(0)]], device ushort *hidden [[buffer(1)]],
	device const ushort *addend [[buffer(2)]], device const float *weight [[buffer(3)]], device ushort *output [[buffer(4)]],
	uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	device ushort *x = hidden + (ulong)row * p.input_stride;
	device const ushort *a = addend + (ulong)row * p.input_stride;
	device ushort *y = output + (ulong)row * p.output_stride;
	float total = 0.0f;
	for ( uint i = tid; i < p.dimension; i += threads )
	{
		float value = SparkBf16ToFloat(x[i]);
		if ( p.has_addend != 0u )
			value = SparkRoundBf16(value + SparkBf16ToFloat(a[i]));
		total += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float scale = rsqrt(total / float(p.dimension) + p.epsilon);
	for ( uint i = tid; i < p.dimension; i += threads )
	{
		float value = SparkBf16ToFloat(x[i]);
		if ( p.has_addend != 0u )
		{
			value = SparkRoundBf16(value + SparkBf16ToFloat(a[i]));
			if ( p.write_sum != 0u )
				x[i] = SparkFloatToBf16(value);
		}
		y[i] = SparkFloatToBf16(value * scale * weight[i]);
	}
}

kernel void spark_rmsnorm_gate(constant SparkGatedNormParams &p [[buffer(0)]], device const ushort *input [[buffer(1)]],
	device const ushort *gate [[buffer(2)]], device const ushort *weight [[buffer(3)]], device ushort *output [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	const uint threads = group_size.x;
	threadgroup float scratch[32];
	const uint head = group.x, row = group.y;
	device const ushort *x = input + (ulong)row * p.input_stride + head * p.head_dimension;
	device const ushort *g = gate + (ulong)row * p.gate_stride + p.gate_offset + head * p.head_dimension;
	device ushort *y = output + (ulong)row * p.output_stride + head * p.head_dimension;
	float total = 0.0f;
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		const float value = SparkBf16ToFloat(x[i]);
		total += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float scale = rsqrt(total / float(p.head_dimension) + p.epsilon);
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		float value = SparkRoundBf16(SparkBf16ToFloat(x[i]) * scale);
		value = SparkRoundBf16(value * SparkBf16ToFloat(weight[i]));
		y[i] = SparkFloatToBf16(value * SparkSilu(SparkBf16ToFloat(g[i])));
	}
}

kernel void spark_head_rmsnorm_rope(constant SparkHeadRopeParams &p [[buffer(0)]], device const ushort *input [[buffer(1)]],
	device const float *weight [[buffer(2)]], device const uint *positions [[buffer(3)]], device ushort *output [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	const uint threads = group_size.x;
	threadgroup float scratch[32];
	threadgroup float normed[1024];
	const uint head = group.x, row = group.y, half_rope = p.rope_dimension / 2u;
	device const ushort *x = input + (ulong)row * p.input_stride + p.input_offset + head * p.input_head_stride;
	device ushort *y = output + (ulong)row * p.output_stride + head * p.head_dimension;
	float total = 0.0f;
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		const float value = SparkBf16ToFloat(x[i]);
		total += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float scale = rsqrt(total / float(p.head_dimension) + p.epsilon);
	for ( uint i = tid; i < p.head_dimension; i += threads )
		normed[i] = SparkRoundBf16(SparkBf16ToFloat(x[i]) * scale * weight[i]);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	const float position = float(positions[row]);
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		float value = normed[i];
		if ( i < p.rope_dimension )
		{
			const uint pair = i % half_rope;
			const float angle = position * p.inverse_frequency[pair];
			const float partner = i < half_rope ? -normed[i + half_rope] : normed[i - half_rope];
			value = value * cos(angle) + partner * sin(angle);
		}
		y[i] = SparkFloatToBf16(value);
	}
}
