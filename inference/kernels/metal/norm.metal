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
		float value = SparkMathBf16ToFloat(x[i]);
		if ( p.has_addend != 0u )
			value += SparkMathBf16ToFloat(a[i]);
		total += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float scale = SparkMathRmsScale(total, float(p.dimension), p.epsilon);
	for ( uint i = tid; i < p.dimension; i += threads )
	{
		float value = SparkMathBf16ToFloat(x[i]);
		if ( p.has_addend != 0u )
		{
			value += SparkMathBf16ToFloat(a[i]);
			if ( p.write_sum != 0u )
				x[i] = SparkMathFloatToBf16(value);
		}
		y[i] = SparkMathFloatToBf16(p.weightless != 0u ? value * scale : value * scale * weight[i]);
	}
}

kernel void spark_rmsnorm_bf16(constant SparkNormParams &p [[buffer(0)]], device const ushort *input [[buffer(1)]],
	device const ushort *weight [[buffer(2)]], device ushort *output [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	device const ushort *x = input + (ulong)row * p.input_stride;
	device ushort *y = output + (ulong)row * p.output_stride;
	float total = 0.0f;
	for ( uint i = tid; i < p.dimension; i += threads )
	{
		const float value = SparkMathBf16ToFloat(x[i]);
		total += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float scale = SparkMathRmsScale(total, float(p.dimension), p.epsilon);
	for ( uint i = tid; i < p.dimension; i += threads )
		y[i] = SparkMathFloatToBf16(SparkMathRoundBf16(SparkMathBf16ToFloat(x[i]) * scale) * SparkMathBf16ToFloat(weight[i]));
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
		const float value = SparkMathBf16ToFloat(x[i]);
		total += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float scale = SparkMathRmsScale(total, float(p.head_dimension), p.epsilon);
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		float value;
		if ( p.weight_f32 != 0u )
		{
			value = SparkMathBf16ToFloat(x[i]) * scale * ((device const float *)weight)[i];
			if ( p.round_norm != 0u )
				value = SparkMathRoundBf16(value);
		}
		else
			value = SparkMathRoundBf16(SparkMathRoundBf16(SparkMathBf16ToFloat(x[i]) * scale) * SparkMathBf16ToFloat(weight[i]));
		const float g_value = SparkMathBf16ToFloat(g[i]);
		y[i] = SparkMathFloatToBf16(value * (p.sigmoid_gate != 0u ? SparkMathSigmoidFast(g_value) : SparkMathSiluFast(g_value)));
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
	device ushort *y = output + (ulong)row * p.output_stride + head * (p.output_head_stride != 0u ? p.output_head_stride : p.head_dimension);
	float total = 0.0f;
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		const float value = SparkMathBf16ToFloat(x[i]);
		total += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float scale = p.plain != 0u ? 1.0f : SparkMathRmsScale(total, float(p.head_dimension), p.epsilon);
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		const float value = SparkMathRoundBf16(p.has_weight != 0u ? SparkMathBf16ToFloat(x[i]) * scale * weight[i] : SparkMathBf16ToFloat(x[i]) * scale);
		normed[i] = p.multiply != 0.0f ? SparkMathRoundBf16(value * p.multiply) : value;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	const float position = float(positions[row]);
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		float value = normed[i];
		if ( i < p.rope_dimension )
		{
			const bool low = p.interleaved != 0u ? i % 2u == 0u : i < half_rope;
			const uint mate = p.interleaved != 0u ? (low ? i + 1u : i - 1u) : (low ? i + half_rope : i - half_rope);
			const float angle = position * p.inverse_frequency[p.interleaved != 0u ? i / 2u : i % half_rope];
			const float c = SparkMathRopeCos(angle, p.scale), s = SparkMathRopeSin(angle, p.scale);
			if ( p.rope_rounded_terms != 0u )
			{
				const float partner = low ? -normed[mate] : normed[mate];
				value = SparkMathRoundBf16(SparkMathRoundBf16(value * SparkMathRoundBf16(c)) + SparkMathRoundBf16(partner * SparkMathRoundBf16(s)));
			}
			else
				value = low ? SparkMathRopeLow(value, normed[mate], c, s) : SparkMathRopeHigh(normed[mate], value, c, s);
		}
		y[i] = SparkMathFloatToBf16(value);
	}
}

static inline float SparkRmsScale(device const ushort *x, uint dimension, float epsilon, threadgroup float *scratch, uint tid, uint threads, uint lane,
	uint simd, uint simds)
{
	float total = 0.0f;
	for ( uint i = tid; i < dimension; i += threads )
	{
		const float value = SparkMathBf16ToFloat(x[i]);
		total += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	return(SparkMathRmsScale(total, float(dimension), epsilon));
}

kernel void spark_rmsnorm_branches(constant SparkBranchNormParams &p [[buffer(0)]], device ushort *hidden [[buffer(1)]],
	device const ushort *first [[buffer(2)]], device const ushort *second [[buffer(3)]], device const float *first_weight [[buffer(4)]],
	device const float *second_weight [[buffer(5)]], device const float *output_weight [[buffer(6)]], device const float *scalar [[buffer(7)]],
	uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	threadgroup float combined[SPARK_BRANCH_NORM_MAX_DIMENSION];
	device ushort *h = hidden + (ulong)row * p.hidden_stride;
	device const ushort *a = first + (ulong)row * p.branch_stride;
	const float scale_a = SparkRmsScale(a, p.dimension, p.epsilon, scratch, tid, threads, lane, simd, simds);
	if ( p.branches == 2u )
	{
		device const ushort *b = second + (ulong)row * p.branch_stride;
		const float scale_b = SparkRmsScale(b, p.dimension, p.epsilon, scratch, tid, threads, lane, simd, simds);
		float total = 0.0f;
		for ( uint i = tid; i < p.dimension; i += threads )
		{
			const float value = SparkMathRoundBf16(SparkMathRoundBf16(SparkMathBf16ToFloat(a[i]) * scale_a * first_weight[i]) +
				SparkMathRoundBf16(SparkMathBf16ToFloat(b[i]) * scale_b * second_weight[i]));
			combined[i] = value;
			total += value * value;
		}
		const float scale = SparkMathRmsScale(SparkGroupSum(total, scratch, lane, simd, simds), float(p.dimension), p.epsilon);
		for ( uint i = tid; i < p.dimension; i += threads )
		{
			const float added = SparkMathRoundBf16(SparkMathBf16ToFloat(h[i]) + SparkMathRoundBf16(combined[i] * scale * output_weight[i]));
			h[i] = SparkMathFloatToBf16(p.has_scalar != 0u ? added * scalar[0] : added);
		}
		return;
	}
	for ( uint i = tid; i < p.dimension; i += threads )
	{
		const float added = SparkMathRoundBf16(SparkMathBf16ToFloat(h[i]) + SparkMathRoundBf16(SparkMathBf16ToFloat(a[i]) * scale_a * output_weight[i]));
		h[i] = SparkMathFloatToBf16(p.has_scalar != 0u ? added * scalar[0] : added);
	}
}

kernel void spark_rmsnorm_scaled(constant SparkBranchNormParams &p [[buffer(0)]], device const ushort *input [[buffer(1)]],
	device const float *scale [[buffer(2)]], device ushort *output [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	device const ushort *x = input + (ulong)row * p.branch_stride;
	const float inverse = SparkRmsScale(x, p.dimension, p.epsilon, scratch, tid, threads, lane, simd, simds);
	for ( uint i = tid; i < p.dimension; i += threads )
	{
		const float normed = SparkMathRoundBf16(SparkMathBf16ToFloat(x[i]) * inverse);
		output[(ulong)row * p.hidden_stride + i] = SparkMathFloatToBf16(SparkMathRoundBf16(normed * scale[i]) * p.output_scale);
	}
}

static inline device const ushort *SparkAttnResSource(constant SparkAttnResParams &p, device const ushort *bank, device const ushort *partial, uint source,
	uint row)
{
	return source + 1u == p.sources ? partial + (ulong)row * p.dimension : bank + ((ulong)source * p.rows + row) * p.dimension;
}

kernel void spark_attnres_score(constant SparkAttnResParams &p [[buffer(0)]], device const ushort *bank [[buffer(1)]],
	device const ushort *partial [[buffer(2)]], device const ushort *weight [[buffer(3)]], device float *scores [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	const uint row = group.x, source = group.y, threads = group_size.x;
	device const ushort *values = SparkAttnResSource(p, bank, partial, source, row);
	float square = 0.0f, dot = 0.0f;
	for ( uint i = tid; i < p.dimension; i += threads )
	{
		const float value = SparkMathBf16ToFloat(values[i]);
		square += value * value;
	}
	square = SparkGroupSum(square, scratch, lane, simd, simds);
	const float inverse = SparkMathRmsScale(square, float(p.dimension), p.epsilon);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint i = tid; i < p.dimension; i += threads )
		dot = fma(SparkMathBf16ToFloat(values[i]) * inverse, SparkMathBf16ToFloat(weight[i]), dot);
	dot = SparkGroupSum(dot, scratch, lane, simd, simds);
	if ( tid == 0u )
		scores[(ulong)row * p.score_stride + source] = dot;
}

kernel void spark_attnres_mix(constant SparkAttnResParams &p [[buffer(0)]], device const ushort *bank [[buffer(1)]],
	device const ushort *partial [[buffer(2)]], device const float *scores [[buffer(3)]], device ushort *output [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	threadgroup float weights[SPARK_ATTNRES_MAX_SOURCES];
	const uint row = group.y, index = group.x * group_size.x + tid;
	if ( tid == 0u )
	{
		device const float *score = scores + (ulong)row * p.score_stride;
		float running_max = -INFINITY, running_sum = 0.0f;
		for ( uint source = 0u; source < p.sources; source++ )
			running_max = max(running_max, score[source]);
		for ( uint source = 0u; source < p.sources; source++ )
		{
			weights[source] = SPARK_MATH_EXP_FAST(score[source] - running_max);
			running_sum += weights[source];
		}
		for ( uint source = 0u; source < p.sources; source++ )
			weights[source] /= running_sum;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( index >= p.dimension )
		return;
	float total = 0.0f;
	for ( uint source = 0u; source < p.sources; source++ )
		total = fma(weights[source], SparkMathBf16ToFloat(SparkAttnResSource(p, bank, partial, source, row)[index]), total);
	output[(ulong)row * p.dimension + index] = SparkMathFloatToBf16(total);
}
