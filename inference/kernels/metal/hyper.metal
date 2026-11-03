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
static inline void SparkHcTotals(device const ushort *source, device const float *fn, uint flat, uint tid, uint threads, uint lane, uint simd, uint simds,
	threadgroup float *reduction, thread float *total)
{
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
	for ( uint m = 0u; m <= MIXES; m++ )
	{
		total[m] = 0.0f;
		for ( uint w = 0u; w < simds; w++ )
			total[m] += reduction[m * 32u + w];
	}
}

static inline void SparkHcCollapse(constant SparkHcParams &p, device const ushort *source, threadgroup const float *weights, device ushort *mixed, uint tid,
	uint threads)
{
	for ( uint d = tid; d < p.width; d += threads )
	{
		float total = 0.0f;
		for ( uint s = 0u; s < p.streams; s++ )
			total = fma(weights[s], SparkMathBf16ToFloat(source[s * p.width + d]), total);
		mixed[d] = SparkMathFloatToBf16(total);
	}
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
	float total[MIXES + 1u];
	SparkHcTotals<MIXES>(source, fn, flat, tid, threads, lane, simd, simds, reduction, total);
	if ( tid == 0u )
	{
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
	SparkHcCollapse(p, source, weights, mixed + (ulong)row * p.width, tid, threads);
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

kernel void spark_hc_site(constant SparkHcParams &p [[buffer(0)]], device const ushort *streams [[buffer(1)]], device const float *fn [[buffer(2)]],
	device const float *base [[buffer(3)]], device const float *scale [[buffer(4)]], device float *post [[buffer(5)]], device float *comb [[buffer(6)]],
	device ushort *mixed [[buffer(7)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
	uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float reduction[(SPARK_HC_COMB_MIXES + 1u) * 32u];
	threadgroup float weights[SPARK_HC_MAX_STREAMS];
	const uint flat = SPARK_HC_MAX_STREAMS * p.width;
	device const ushort *source = streams + (ulong)row * flat;
	float total[SPARK_HC_COMB_MIXES + 1u];
	SparkHcTotals<SPARK_HC_COMB_MIXES>(source, fn, flat, tid, threads, lane, simd, simds, reduction, total);
	if ( tid == 0u )
	{
		const float inverse = SparkMathRmsScale(total[SPARK_HC_COMB_MIXES], float(flat), p.norm_epsilon);
		float cell[SPARK_HC_MAX_STREAMS * SPARK_HC_MAX_STREAMS], sums[SPARK_HC_MAX_STREAMS];
		for ( uint s = 0u; s < SPARK_HC_MAX_STREAMS; s++ )
		{
			weights[s] = SparkMathSigmoidFast(total[s] * inverse * scale[0] + base[s]) + p.hc_epsilon;
			post[(ulong)row * SPARK_HC_MAX_STREAMS + s] = SparkMathScaledSigmoidFast(p.post_magnitude,
				total[SPARK_HC_MAX_STREAMS + s] * inverse * scale[1] + base[SPARK_HC_MAX_STREAMS + s]);
		}
		for ( uint c = 0u; c < SPARK_HC_MAX_STREAMS * SPARK_HC_MAX_STREAMS; c++ )
			cell[c] = total[2u * SPARK_HC_MAX_STREAMS + c] * inverse * scale[2] + base[2u * SPARK_HC_MAX_STREAMS + c];
		for ( uint i = 0u; i < SPARK_HC_MAX_STREAMS; i++ )
		{
			float maximum = -3.0e38f;
			for ( uint j = 0u; j < SPARK_HC_MAX_STREAMS; j++ )
				maximum = fmax(maximum, cell[i * SPARK_HC_MAX_STREAMS + j]);
			for ( uint j = 0u; j < SPARK_HC_MAX_STREAMS; j++ )
				cell[i * SPARK_HC_MAX_STREAMS + j] = SPARK_MATH_EXP_FAST(cell[i * SPARK_HC_MAX_STREAMS + j] - maximum);
			sums[i] = 0.0f;
			for ( uint j = 0u; j < SPARK_HC_MAX_STREAMS; j++ )
				sums[i] += cell[i * SPARK_HC_MAX_STREAMS + j];
		}
		for ( uint c = 0u; c < SPARK_HC_MAX_STREAMS * SPARK_HC_MAX_STREAMS; c++ )
			cell[c] = cell[c] / sums[c / SPARK_HC_MAX_STREAMS] + p.hc_epsilon;
		for ( uint iteration = 0u; iteration < p.iterations; iteration++ )
		{
			if ( iteration != 0u )
			{
				for ( uint i = 0u; i < SPARK_HC_MAX_STREAMS; i++ )
				{
					sums[i] = 0.0f;
					for ( uint j = 0u; j < SPARK_HC_MAX_STREAMS; j++ )
						sums[i] += cell[i * SPARK_HC_MAX_STREAMS + j];
				}
				for ( uint c = 0u; c < SPARK_HC_MAX_STREAMS * SPARK_HC_MAX_STREAMS; c++ )
					cell[c] /= sums[c / SPARK_HC_MAX_STREAMS] + p.hc_epsilon;
			}
			for ( uint j = 0u; j < SPARK_HC_MAX_STREAMS; j++ )
			{
				sums[j] = 0.0f;
				for ( uint i = 0u; i < SPARK_HC_MAX_STREAMS; i++ )
					sums[j] += cell[i * SPARK_HC_MAX_STREAMS + j];
			}
			for ( uint c = 0u; c < SPARK_HC_MAX_STREAMS * SPARK_HC_MAX_STREAMS; c++ )
				cell[c] /= sums[c % SPARK_HC_MAX_STREAMS] + p.hc_epsilon;
		}
		for ( uint c = 0u; c < SPARK_HC_MAX_STREAMS * SPARK_HC_MAX_STREAMS; c++ )
			comb[(ulong)row * SPARK_HC_MAX_STREAMS * SPARK_HC_MAX_STREAMS + c] = cell[c];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	SparkHcCollapse(p, source, weights, mixed + (ulong)row * p.width, tid, threads);
}

kernel void spark_hc_post_comb(constant SparkHcParams &p [[buffer(0)]], device ushort *streams [[buffer(1)]], device const float *post [[buffer(2)]],
	device const float *comb [[buffer(3)]], device const ushort *output [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, d = group.x * group_size.x + tid;
	if ( d >= p.width )
		return;
	device ushort *row_streams = streams + (ulong)row * SPARK_HC_MAX_STREAMS * p.width;
	const float out = SparkMathBf16ToFloat(output[(ulong)row * p.width + d]);
	float residual[SPARK_HC_MAX_STREAMS];
	for ( uint s = 0u; s < SPARK_HC_MAX_STREAMS; s++ )
		residual[s] = SparkMathBf16ToFloat(row_streams[s * p.width + d]);
	for ( uint t = 0u; t < SPARK_HC_MAX_STREAMS; t++ )
	{
		float value = 0.0f;
		for ( uint s = 0u; s < SPARK_HC_MAX_STREAMS; s++ )
			value = fma(SparkMathRoundBf16(comb[((ulong)row * SPARK_HC_MAX_STREAMS + s) * SPARK_HC_MAX_STREAMS + t]), residual[s], value);
		const float contribution = SparkMathRoundBf16(SparkMathRoundBf16(post[(ulong)row * SPARK_HC_MAX_STREAMS + t]) * out);
		row_streams[t * p.width + d] = SparkMathFloatToBf16(contribution + SparkMathRoundBf16(value));
	}
}

kernel void spark_stream_mean(constant SparkHcParams &p [[buffer(0)]], device const ushort *streams [[buffer(1)]], device ushort *output [[buffer(2)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, d = group.x * group_size.x + tid;
	if ( d >= p.width )
		return;
	device const ushort *source = streams + (ulong)row * p.streams * p.width + d;
	float total = 0.0f;
	for ( uint s = 0u; s < p.streams; s++ )
		total += SparkMathBf16ToFloat(source[(ulong)s * p.width]);
	output[(ulong)row * p.width + d] = SparkMathFloatToBf16(total / float(p.streams));
}
