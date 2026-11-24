#include "inference/kernels/metal/common.metal"

kernel void spark_hyper_expand(constant SparkHyperParams &p [[buffer(0)]], device const ushort *hidden [[buffer(1)]],
	device ushort *streams [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y;
	if ( (((ulong)hidden | (ulong)streams) & 15ul) == 0ul && (p.width & 7u) == 0u && (group_size.x & 7u) == 0u )
	{
		const uint vec_threads = group_size.x / 8u;
		if ( tid >= vec_threads )
			return;
		const uint v = group.x * vec_threads + tid, wvec = p.width / 8u;
		if ( v >= p.streams * wvec )
			return;
		((device uint4 *)streams)[(ulong)row * p.streams * wvec + v] = ((device const uint4 *)hidden)[(ulong)row * wvec + v % wvec];
		return;
	}
	const uint index = group.x * group_size.x + tid;
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
	const uint stream = group.x, row = group.y, threads = group_size.x;
	const ulong base = ((ulong)row * p.streams + stream) * p.width;
	device const ushort *x = input + base;
	device const float *gain = weight + (ulong)stream * p.width;
	device ushort *y = output + base;
	const uint vec = (((ulong)x | (ulong)gain | (ulong)y) & 15ul) == 0ul ? p.width / 8u : 0u, tail = vec * 8u;
	float4 cl = 0.0f, ch = 0.0f;
	float total = 0.0f;
	if ( vec <= threads )
	{
		if ( tid < vec )
		{
			SparkUnpack8(((device const uint4 *)x)[tid], cl, ch);
			total = dot(cl, cl) + dot(ch, ch);
		}
	}
	else
	{
		for ( uint v = tid; v < vec; v += threads )
		{
			float4 xl, xh;
			SparkUnpack8(((device const uint4 *)x)[v], xl, xh);
			total += dot(xl, xl) + dot(xh, xh);
		}
	}
	for ( uint i = tail + tid; i < p.width; i += threads )
	{
		const float value = SparkMathBf16ToFloat(x[i]);
		total += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float scale = SparkMathRmsScale(total, float(p.width), p.epsilon);
	if ( vec <= threads )
	{
		if ( tid < vec )
		{
			const float4 gl = ((device const float4 *)gain)[2u * tid], gh = ((device const float4 *)gain)[2u * tid + 1u];
			((device uint4 *)y)[tid] = SparkPack8(cl * scale * gl, ch * scale * gh);
		}
	}
	else
	{
		for ( uint v = tid; v < vec; v += threads )
		{
			float4 xl, xh;
			SparkUnpack8(((device const uint4 *)x)[v], xl, xh);
			const float4 gl = ((device const float4 *)gain)[2u * v], gh = ((device const float4 *)gain)[2u * v + 1u];
			((device uint4 *)y)[v] = SparkPack8(xl * scale * gl, xh * scale * gh);
		}
	}
	for ( uint i = tail + tid; i < p.width; i += threads )
		y[i] = SparkMathFloatToBf16(SparkMathBf16ToFloat(x[i]) * scale * gain[i]);
}

static inline float4 SparkHyperGate4(float4 v, float divisor)
{
	return float4(SparkMathHyperGate(v.x, divisor), SparkMathHyperGate(v.y, divisor), SparkMathHyperGate(v.z, divisor), SparkMathHyperGate(v.w, divisor));
}

kernel void spark_hyper_gate(constant SparkHyperParams &p [[buffer(0)]], device ushort *values [[buffer(1)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y;
	if ( ((ulong)values & 15ul) == 0ul && (p.width & 7u) == 0u && (group_size.x & 7u) == 0u )
	{
		const uint vec_threads = group_size.x / 8u;
		if ( tid >= vec_threads )
			return;
		const uint v = group.x * vec_threads + tid, wvec = p.width / 8u;
		if ( v >= wvec )
			return;
		const ulong at = (ulong)row * wvec + v;
		float4 xl, xh;
		SparkUnpack8(((device const uint4 *)values)[at], xl, xh);
		((device uint4 *)values)[at] = SparkPack8(SparkHyperGate4(xl, p.divisor), SparkHyperGate4(xh, p.divisor));
		return;
	}
	const uint index = group.x * group_size.x + tid;
	if ( index >= p.width )
		return;
	const ulong at = (ulong)row * p.width + index;
	values[at] = SparkMathFloatToBf16(SparkMathHyperGate(SparkMathBf16ToFloat(values[at]), p.divisor));
}

static inline float4 SparkHyperMixTerm4(float4 u, float4 n)
{
	const float4 g = float4(SparkMathHyperMixGate(u.x), SparkMathHyperMixGate(u.y), SparkMathHyperMixGate(u.z), SparkMathHyperMixGate(u.w));
	return SparkRoundBf16x4(g * n);
}

kernel void spark_hyper_mix(constant SparkHyperParams &p [[buffer(0)]], device const ushort *up [[buffer(1)]],
	device const ushort *normed [[buffer(2)]], device ushort *mixed [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y;
	if ( (((ulong)up | (ulong)normed | (ulong)mixed) & 15ul) == 0ul && (p.width & 7u) == 0u && (group_size.x & 7u) == 0u )
	{
		const uint vec_threads = group_size.x / 8u;
		if ( tid >= vec_threads )
			return;
		const uint v = group.x * vec_threads + tid, wvec = p.width / 8u;
		if ( v >= wvec )
			return;
		const ulong base = (ulong)row * p.streams * wvec;
		float4 tl = 0.0f, th = 0.0f;
		for ( uint stream = 0u; stream < p.streams; stream++ )
		{
			const ulong at = base + (ulong)stream * wvec + v;
			float4 ul, uh, nl, nh;
			SparkUnpack8(((device const uint4 *)up)[at], ul, uh);
			SparkUnpack8(((device const uint4 *)normed)[at], nl, nh);
			tl += SparkHyperMixTerm4(ul, nl);
			th += SparkHyperMixTerm4(uh, nh);
		}
		const float inv = float(p.streams);
		((device uint4 *)mixed)[(ulong)row * wvec + v] = SparkPack8(tl / inv, th / inv);
		return;
	}
	const uint element = group.x * group_size.x + tid;
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
	const uint vec = (((ulong)source | (ulong)weight) & 15ul) == 0ul && (total_width & 7u) == 0u ? total_width / 8u : 0u, tail = vec * 8u;
	for ( uint v = tid; v < vec; v += threads )
	{
		float4 xl, xh;
		SparkUnpack8(((device const uint4 *)source)[v], xl, xh);
		for ( uint stream = 0u; stream < p.streams; stream++ )
		{
			float4 wl, wh;
			SparkUnpack8(((device const uint4 *)weight)[(ulong)stream * vec + v], wl, wh);
			sums[stream] += dot(xl, wl) + dot(xh, wh);
		}
	}
	for ( uint i = tail + tid; i < total_width; i += threads )
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
	const uint row = group.y;
	if ( (((ulong)streams | (ulong)output) & 15ul) == 0ul && (p.width & 7u) == 0u && (group_size.x & 7u) == 0u )
	{
		const uint vec_threads = group_size.x / 8u;
		if ( tid >= vec_threads )
			return;
		const uint v = group.x * vec_threads + tid, wvec = p.width / 8u;
		if ( v >= p.streams * wvec )
			return;
		const ulong at = (ulong)row * p.streams * wvec + v;
		const float w = inject[(ulong)row * p.streams + v / wvec];
		float4 ol, oh, sl, sh;
		SparkUnpack8(((device const uint4 *)output)[(ulong)row * wvec + v % wvec], ol, oh);
		SparkUnpack8(((device const uint4 *)streams)[at], sl, sh);
		((device uint4 *)streams)[at] = SparkPack8(sl + SparkRoundBf16x4(ol * w), sh + SparkRoundBf16x4(oh * w));
		return;
	}
	const uint index = group.x * group_size.x + tid;
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
	const uint stream = group.x, row = group.y, threads = group_size.x;
	const ulong base = ((ulong)row * p.streams + stream) * p.width;
	device const ushort *k = key + base, *q = query + base, *val = value + (ulong)row * p.width;
	device ushort *g = gated + base;
	const uint vec = (((ulong)k | (ulong)q | (ulong)val | (ulong)g) & 15ul) == 0ul ? p.width / 8u : 0u, tail = vec * 8u;
	float total = 0.0f;
	for ( uint v = tid; v < vec; v += threads )
	{
		float4 kl, kh, ql, qh;
		SparkUnpack8(((device const uint4 *)k)[v], kl, kh);
		SparkUnpack8(((device const uint4 *)q)[v], ql, qh);
		const float4 pl = SparkRoundBf16x4(kl * ql), ph = SparkRoundBf16x4(kh * qh);
		total += (pl.x + pl.y) + (pl.z + pl.w) + (ph.x + ph.y) + (ph.z + ph.w);
	}
	for ( uint i = tail + tid; i < p.width; i += threads )
		total += SparkMathRoundBf16(SparkMathBf16ToFloat(k[i]) * SparkMathBf16ToFloat(q[i]));
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float gate = SparkMathHyperKeyGate(total, p.width);
	for ( uint v = tid; v < vec; v += threads )
	{
		float4 vl, vh;
		SparkUnpack8(((device const uint4 *)val)[v], vl, vh);
		((device uint4 *)g)[v] = SparkPack8(gate * vl, gate * vh);
	}
	for ( uint i = tail + tid; i < p.width; i += threads )
		g[i] = SparkMathFloatToBf16(gate * SparkMathBf16ToFloat(val[i]));
}

kernel void spark_hyper_add(constant SparkHyperParams &p [[buffer(0)]], device ushort *streams [[buffer(1)]],
	device const ushort *gated [[buffer(2)]], device const ushort *conv [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y;
	if ( (((ulong)streams | (ulong)gated | (ulong)conv) & 15ul) == 0ul && ((p.streams * p.width) & 7u) == 0u && (group_size.x & 7u) == 0u )
	{
		const uint vec_threads = group_size.x / 8u;
		if ( tid >= vec_threads )
			return;
		const uint v = group.x * vec_threads + tid, total_vec = p.streams * p.width / 8u;
		if ( v >= total_vec )
			return;
		const ulong at = (ulong)row * total_vec + v;
		float4 sl, sh, gl, gh, cl, ch;
		SparkUnpack8(((device const uint4 *)streams)[at], sl, sh);
		SparkUnpack8(((device const uint4 *)gated)[at], gl, gh);
		SparkUnpack8(((device const uint4 *)conv)[at], cl, ch);
		((device uint4 *)streams)[at] = SparkPack8(sl + SparkRoundBf16x4(gl + cl), sh + SparkRoundBf16x4(gh + ch));
		return;
	}
	const uint index = group.x * group_size.x + tid;
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
	float acc[MIXES + 1u];
	for ( uint m = 0u; m <= MIXES; m++ )
		acc[m] = 0.0f;
	const uint vec = ((ulong)source & 7ul) == 0ul && ((ulong)fn & 15ul) == 0ul && (flat & 3u) == 0u ? flat / 4u : 0u, tail = vec * 4u;
	for ( uint v = tid; v < vec; v += threads )
	{
		const float4 x = SparkUnpack4(((device const uint2 *)source)[v]);
		for ( uint m = 0u; m < MIXES; m++ )
			acc[m] += dot(x, ((device const float4 *)fn)[(ulong)m * vec + v]);
		acc[MIXES] += dot(x, x);
	}
	for ( uint k = tail + tid; k < flat; k += threads )
	{
		const float x = SparkMathBf16ToFloat(source[k]);
		for ( uint m = 0u; m < MIXES; m++ )
			acc[m] = fma(x, fn[(ulong)m * flat + k], acc[m]);
		acc[MIXES] = fma(x, x, acc[MIXES]);
	}
	for ( uint m = 0u; m <= MIXES; m++ )
	{
		const float value = simd_sum(acc[m]);
		if ( lane == 0u )
			reduction[m * 32u + simd] = value;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( tid == 0u )
	{
		for ( uint m = 0u; m <= MIXES; m++ )
		{
			total[m] = 0.0f;
			for ( uint w = 0u; w < simds; w++ )
				total[m] += reduction[m * 32u + w];
		}
	}
}

static inline void SparkHcCollapse(constant SparkHcParams &p, device const ushort *source, threadgroup const float *weights, device ushort *mixed, uint tid,
	uint threads)
{
	const uint vec = (((ulong)source | (ulong)mixed) & 15ul) == 0ul && (p.width & 7u) == 0u ? p.width / 8u : 0u, tail = vec * 8u;
	for ( uint v = tid; v < vec; v += threads )
	{
		float4 tl = 0.0f, th = 0.0f;
		for ( uint s = 0u; s < p.streams; s++ )
		{
			float4 sl, sh;
			SparkUnpack8(((device const uint4 *)(source + (ulong)s * p.width))[v], sl, sh);
			tl = fma(weights[s], sl, tl);
			th = fma(weights[s], sh, th);
		}
		((device uint4 *)mixed)[v] = SparkPack8(tl, th);
	}
	for ( uint d = tail + tid; d < p.width; d += threads )
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
	const uint row = group.y;
	if ( (((ulong)streams | (ulong)output) & 15ul) == 0ul && (p.width & 7u) == 0u && (group_size.x & 7u) == 0u )
	{
		const uint vec_threads = group_size.x / 8u;
		if ( tid >= vec_threads )
			return;
		const uint v = group.x * vec_threads + tid, wvec = p.width / 8u;
		if ( v >= p.streams * wvec )
			return;
		const ulong at = (ulong)row * p.streams * wvec + v;
		const float gain = post[(ulong)row * p.streams + v / wvec];
		float4 ol, oh, sl, sh;
		SparkUnpack8(((device const uint4 *)output)[(ulong)row * wvec + v % wvec], ol, oh);
		SparkUnpack8(((device const uint4 *)streams)[at], sl, sh);
		((device uint4 *)streams)[at] = SparkPack8(fma(gain, ol, sl), fma(gain, oh, sh));
		return;
	}
	const uint index = group.x * group_size.x + tid;
	if ( index >= p.streams * p.width )
		return;
	const ulong at = (ulong)row * p.streams * p.width + index;
	streams[at] = SparkMathFloatToBf16(fma(post[(ulong)row * p.streams + index / p.width], SparkMathBf16ToFloat(output[(ulong)row * p.width + index % p.width]),
		SparkMathBf16ToFloat(streams[at])));
}

template <bool LAGGED>
kernel void spark_hc_site(constant SparkHcParams &p [[buffer(0)]], device const ushort *streams [[buffer(1)]], device const float *fn [[buffer(2)]],
	device const float *base [[buffer(3)]], device const float *scale [[buffer(4)]], device float *post [[buffer(5)]], device float *comb [[buffer(6)]],
	device ushort *mixed [[buffer(7)]], device float *pre_state [[buffer(8)]], uint row [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float reduction[(SPARK_HC_COMB_MIXES + 1u) * 32u];
	threadgroup float weights[SPARK_HC_MAX_STREAMS];
	threadgroup float lagged[SPARK_HC_MAX_STREAMS];
	const uint flat = SPARK_HC_MAX_STREAMS * p.width;
	device const ushort *source = streams + (ulong)row * flat;
	float total[SPARK_HC_COMB_MIXES + 1u];
	if ( LAGGED && tid < SPARK_HC_MAX_STREAMS )
		lagged[tid] = pre_state[(ulong)row * SPARK_HC_MAX_STREAMS + tid];
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
		if ( LAGGED )
			for ( uint s = 0u; s < SPARK_HC_MAX_STREAMS; s++ )
				pre_state[(ulong)row * SPARK_HC_MAX_STREAMS + s] = weights[s];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	SparkHcCollapse(p, source, LAGGED ? lagged : weights, mixed + (ulong)row * p.width, tid, threads);
}

#define SPARK_HC_SITE_INSTANCE(name, lagged) \
	template [[host_name(name)]] kernel void spark_hc_site<lagged>(constant SparkHcParams &p [[buffer(0)]], device const ushort *streams [[buffer(1)]], \
		device const float *fn [[buffer(2)]], device const float *base [[buffer(3)]], device const float *scale [[buffer(4)]], device float *post [[buffer(5)]], \
		device float *comb [[buffer(6)]], device ushort *mixed [[buffer(7)]], device float *pre_state [[buffer(8)]], uint row [[threadgroup_position_in_grid]], \
		uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
		uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]]);

SPARK_HC_SITE_INSTANCE("spark_hc_site", false)
SPARK_HC_SITE_INSTANCE("spark_hc_site_lagged", true)

kernel void spark_hc_site_clamped(constant SparkHcParams &p [[buffer(0)]], device const ushort *streams [[buffer(1)]], device const float *fn [[buffer(2)]],
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
			weights[s] = SparkMathSigmoid(SparkMathClamp(total[s] * inverse * scale[0] + base[s], p.gate_clamp));
			post[(ulong)row * SPARK_HC_MAX_STREAMS + s] = SparkMathScaledSigmoid(p.post_magnitude,
				SparkMathClamp(total[SPARK_HC_MAX_STREAMS + s] * inverse * scale[1] + base[SPARK_HC_MAX_STREAMS + s], p.gate_clamp));
		}
		for ( uint c = 0u; c < SPARK_HC_MAX_STREAMS * SPARK_HC_MAX_STREAMS; c++ )
			cell[c] = SPARK_MATH_EXP(SparkMathClamp(total[2u * SPARK_HC_MAX_STREAMS + c] * inverse * scale[2] + base[2u * SPARK_HC_MAX_STREAMS + c],
				p.residual_clamp));
		for ( uint iteration = 0u; iteration < p.iterations; iteration++ )
		{
			for ( uint i = 0u; i < SPARK_HC_MAX_STREAMS; i++ )
			{
				sums[i] = 0.0f;
				for ( uint j = 0u; j < SPARK_HC_MAX_STREAMS; j++ )
					sums[i] += cell[i * SPARK_HC_MAX_STREAMS + j];
			}
			for ( uint c = 0u; c < SPARK_HC_MAX_STREAMS * SPARK_HC_MAX_STREAMS; c++ )
				cell[c] /= fmax(sums[c / SPARK_HC_MAX_STREAMS], p.sum_floor);
			for ( uint j = 0u; j < SPARK_HC_MAX_STREAMS; j++ )
			{
				sums[j] = 0.0f;
				for ( uint i = 0u; i < SPARK_HC_MAX_STREAMS; i++ )
					sums[j] += cell[i * SPARK_HC_MAX_STREAMS + j];
			}
			for ( uint c = 0u; c < SPARK_HC_MAX_STREAMS * SPARK_HC_MAX_STREAMS; c++ )
				cell[c] /= fmax(sums[c % SPARK_HC_MAX_STREAMS], p.sum_floor);
		}
		for ( uint i = 0u; i < SPARK_HC_MAX_STREAMS; i++ )
			for ( uint j = 0u; j < SPARK_HC_MAX_STREAMS; j++ )
				comb[((ulong)row * SPARK_HC_MAX_STREAMS + j) * SPARK_HC_MAX_STREAMS + i] = cell[i * SPARK_HC_MAX_STREAMS + j];
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	SparkHcCollapse(p, source, weights, mixed + (ulong)row * p.width, tid, threads);
}

template <bool ROUNDED>
kernel void spark_hc_post_comb(constant SparkHcParams &p [[buffer(0)]], device ushort *streams [[buffer(1)]], device const float *post [[buffer(2)]],
	device const float *comb [[buffer(3)]], device const ushort *output [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y;
	if ( (((ulong)streams | (ulong)output) & 15ul) == 0ul && (p.width & 7u) == 0u && (group_size.x & 7u) == 0u )
	{
		const uint vec_threads = group_size.x / 8u;
		if ( tid >= vec_threads )
			return;
		const uint v = group.x * vec_threads + tid, wvec = p.width / 8u;
		if ( v >= wvec )
			return;
		device uint4 *row_streams = (device uint4 *)streams + (ulong)row * SPARK_HC_MAX_STREAMS * wvec;
		float4 ol, oh;
		SparkUnpack8(((device const uint4 *)output)[(ulong)row * wvec + v], ol, oh);
		float4 rl[SPARK_HC_MAX_STREAMS], rh[SPARK_HC_MAX_STREAMS];
		for ( uint s = 0u; s < SPARK_HC_MAX_STREAMS; s++ )
			SparkUnpack8(row_streams[(ulong)s * wvec + v], rl[s], rh[s]);
		for ( uint t = 0u; t < SPARK_HC_MAX_STREAMS; t++ )
		{
			float4 vl = 0.0f, vh = 0.0f;
			for ( uint s = 0u; s < SPARK_HC_MAX_STREAMS; s++ )
			{
				const float weight = comb[((ulong)row * SPARK_HC_MAX_STREAMS + s) * SPARK_HC_MAX_STREAMS + t];
				const float w = ROUNDED ? SparkMathRoundBf16(weight) : weight;
				vl = fma(w, rl[s], vl);
				vh = fma(w, rh[s], vh);
			}
			const float gain = post[(ulong)row * SPARK_HC_MAX_STREAMS + t];
			if ( ROUNDED )
			{
				const float g = SparkMathRoundBf16(gain);
				row_streams[(ulong)t * wvec + v] = SparkPack8(SparkRoundBf16x4(g * ol) + SparkRoundBf16x4(vl), SparkRoundBf16x4(g * oh) + SparkRoundBf16x4(vh));
			}
			else
				row_streams[(ulong)t * wvec + v] = SparkPack8(fma(gain, ol, vl), fma(gain, oh, vh));
		}
		return;
	}
	const uint d = group.x * group_size.x + tid;
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
		{
			const float weight = comb[((ulong)row * SPARK_HC_MAX_STREAMS + s) * SPARK_HC_MAX_STREAMS + t];
			value = fma(ROUNDED ? SparkMathRoundBf16(weight) : weight, residual[s], value);
		}
		const float gain = post[(ulong)row * SPARK_HC_MAX_STREAMS + t];
		if ( ROUNDED )
			row_streams[t * p.width + d] = SparkMathFloatToBf16(SparkMathRoundBf16(SparkMathRoundBf16(gain) * out) + SparkMathRoundBf16(value));
		else
			row_streams[t * p.width + d] = SparkMathFloatToBf16(fma(gain, out, value));
	}
}

#define SPARK_HC_POST_COMB_INSTANCE(name, rounded) \
	template [[host_name(name)]] kernel void spark_hc_post_comb<rounded>(constant SparkHcParams &p [[buffer(0)]], device ushort *streams [[buffer(1)]], \
		device const float *post [[buffer(2)]], device const float *comb [[buffer(3)]], device const ushort *output [[buffer(4)]], \
		uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);

SPARK_HC_POST_COMB_INSTANCE("spark_hc_post_comb", true)
SPARK_HC_POST_COMB_INSTANCE("spark_hc_post_comb_f32", false)

kernel void spark_stream_mean(constant SparkHcParams &p [[buffer(0)]], device const ushort *streams [[buffer(1)]], device ushort *output [[buffer(2)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y;
	if ( (((ulong)streams | (ulong)output) & 15ul) == 0ul && (p.width & 7u) == 0u && (group_size.x & 7u) == 0u )
	{
		const uint vec_threads = group_size.x / 8u;
		if ( tid >= vec_threads )
			return;
		const uint v = group.x * vec_threads + tid, wvec = p.width / 8u;
		if ( v >= wvec )
			return;
		device const uint4 *source = (device const uint4 *)streams + (ulong)row * p.streams * wvec + v;
		float4 tl = 0.0f, th = 0.0f;
		for ( uint s = 0u; s < p.streams; s++ )
		{
			float4 sl, sh;
			SparkUnpack8(source[(ulong)s * wvec], sl, sh);
			tl += sl;
			th += sh;
		}
		const float inv = float(p.streams);
		((device uint4 *)output)[(ulong)row * wvec + v] = SparkPack8(tl / inv, th / inv);
		return;
	}
	const uint d = group.x * group_size.x + tid;
	if ( d >= p.width )
		return;
	device const ushort *source = streams + (ulong)row * p.streams * p.width + d;
	float total = 0.0f;
	for ( uint s = 0u; s < p.streams; s++ )
		total += SparkMathBf16ToFloat(source[(ulong)s * p.width]);
	output[(ulong)row * p.width + d] = SparkMathFloatToBf16(total / float(p.streams));
}

kernel void spark_hc_collapse(constant SparkHcParams &p [[buffer(0)]], device const ushort *streams [[buffer(1)]], device const float *pre [[buffer(2)]],
	device ushort *output [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y;
	if ( (((ulong)streams | (ulong)output) & 15ul) == 0ul && (p.width & 7u) == 0u && (group_size.x & 7u) == 0u )
	{
		const uint vec_threads = group_size.x / 8u;
		if ( tid >= vec_threads )
			return;
		const uint v = group.x * vec_threads + tid, wvec = p.width / 8u;
		if ( v >= wvec )
			return;
		device const uint4 *source = (device const uint4 *)streams + (ulong)row * p.streams * wvec + v;
		float4 tl = 0.0f, th = 0.0f;
		for ( uint s = 0u; s < p.streams; s++ )
		{
			float4 sl, sh;
			SparkUnpack8(source[(ulong)s * wvec], sl, sh);
			const float w = pre[(ulong)row * p.streams + s];
			tl = fma(w, sl, tl);
			th = fma(w, sh, th);
		}
		((device uint4 *)output)[(ulong)row * wvec + v] = SparkPack8(tl, th);
		return;
	}
	const uint d = group.x * group_size.x + tid;
	if ( d >= p.width )
		return;
	device const ushort *source = streams + (ulong)row * p.streams * p.width + d;
	float total = 0.0f;
	for ( uint s = 0u; s < p.streams; s++ )
		total = fma(pre[(ulong)row * p.streams + s], SparkMathBf16ToFloat(source[(ulong)s * p.width]), total);
	output[(ulong)row * p.width + d] = SparkMathFloatToBf16(total);
}

kernel void spark_stream_key_gate(constant SparkGateParams &p [[buffer(0)]], device ushort *streams [[buffer(1)]], device const ushort *kv [[buffer(2)]],
	device const ushort *query_weight [[buffer(3)]], device const ushort *key_weight [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	const uint threads = group_size.x, stream = group.x, row = group.y;
	device ushort *h = streams + ((ulong)row * p.streams + stream) * p.width;
	device const ushort *row_kv = kv + (ulong)row * (p.streams + 1u) * p.width;
	device const ushort *key = row_kv + (ulong)stream * p.width, *value = row_kv + (ulong)p.streams * p.width;
	device const ushort *qw = query_weight + (ulong)stream * p.width, *kw = key_weight + (ulong)stream * p.width;
	const uint vec = (((ulong)h | (ulong)key | (ulong)value | (ulong)qw | (ulong)kw) & 15ul) == 0ul ? p.width / 8u : 0u, tail = vec * 8u;
	float4 cl = 0.0f, ch = 0.0f;
	float h_square = 0.0f, key_square = 0.0f, acc = 0.0f;
	if ( vec <= threads )
	{
		if ( tid < vec )
		{
			float4 kl, kh, qwl, qwh, kwl, kwh;
			SparkUnpack8(((device const uint4 *)h)[tid], cl, ch);
			SparkUnpack8(((device const uint4 *)key)[tid], kl, kh);
			SparkUnpack8(((device const uint4 *)qw)[tid], qwl, qwh);
			SparkUnpack8(((device const uint4 *)kw)[tid], kwl, kwh);
			h_square = dot(cl, cl) + dot(ch, ch);
			key_square = dot(kl, kl) + dot(kh, kh);
			acc = dot(cl * (qwl * kwl), kl) + dot(ch * (qwh * kwh), kh);
		}
	}
	else
	{
		for ( uint v = tid; v < vec; v += threads )
		{
			float4 xl, xh, kl, kh, qwl, qwh, kwl, kwh;
			SparkUnpack8(((device const uint4 *)h)[v], xl, xh);
			SparkUnpack8(((device const uint4 *)key)[v], kl, kh);
			SparkUnpack8(((device const uint4 *)qw)[v], qwl, qwh);
			SparkUnpack8(((device const uint4 *)kw)[v], kwl, kwh);
			h_square += dot(xl, xl) + dot(xh, xh);
			key_square += dot(kl, kl) + dot(kh, kh);
			acc += dot(xl * (qwl * kwl), kl) + dot(xh * (qwh * kwh), kh);
		}
	}
	for ( uint i = tail + tid; i < p.width; i += threads )
	{
		const float x = SparkMathBf16ToFloat(h[i]), k = SparkMathBf16ToFloat(key[i]);
		const float weight = SparkMathBf16ToFloat(qw[i]) * SparkMathBf16ToFloat(kw[i]);
		h_square = fma(x, x, h_square);
		key_square = fma(k, k, key_square);
		acc = fma(x * weight, k, acc);
	}
	h_square = SparkGroupSum(h_square, scratch, lane, simd, simds);
	key_square = SparkGroupSum(key_square, scratch, lane, simd, simds);
	acc = SparkGroupSum(acc, scratch, lane, simd, simds);
	const float gate = SparkMathSignedSqrtGate(acc, h_square, key_square, p.width, p.epsilon, p.floor);
	if ( vec <= threads )
	{
		if ( tid < vec )
		{
			float4 vl, vh;
			SparkUnpack8(((device const uint4 *)value)[tid], vl, vh);
			((device uint4 *)h)[tid] = SparkPack8(fma(gate, vl, cl), fma(gate, vh, ch));
		}
	}
	else
	{
		for ( uint v = tid; v < vec; v += threads )
		{
			float4 xl, xh, vl, vh;
			SparkUnpack8(((device const uint4 *)h)[v], xl, xh);
			SparkUnpack8(((device const uint4 *)value)[v], vl, vh);
			((device uint4 *)h)[v] = SparkPack8(fma(gate, vl, xl), fma(gate, vh, xh));
		}
	}
	for ( uint i = tail + tid; i < p.width; i += threads )
		h[i] = SparkMathFloatToBf16(fma(gate, SparkMathBf16ToFloat(value[i]), SparkMathBf16ToFloat(h[i])));
}
