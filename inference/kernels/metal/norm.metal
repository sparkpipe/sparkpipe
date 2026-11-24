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
	const ulong mask = (ulong)x | (ulong)y | (p.has_addend != 0u ? (ulong)a : 0ul) | (p.weightless == 0u ? (ulong)weight : 0ul);
	const uint vec = (mask & 15ul) == 0ul ? p.dimension / 8u : 0u, tail = vec * 8u;
	float4 cl = 0.0f, ch = 0.0f;
	float total = 0.0f;
	if ( vec <= threads )
	{
		if ( tid < vec )
		{
			SparkUnpack8(((device const uint4 *)x)[tid], cl, ch);
			if ( p.has_addend != 0u )
			{
				float4 al, ah;
				SparkUnpack8(((device const uint4 *)a)[tid], al, ah);
				cl = SparkResidualSum4(cl, al);
				ch = SparkResidualSum4(ch, ah);
			}
			total = dot(cl, cl) + dot(ch, ch);
		}
	}
	else
	{
		for ( uint v = tid; v < vec; v += threads )
		{
			float4 xl, xh;
			SparkUnpack8(((device const uint4 *)x)[v], xl, xh);
			if ( p.has_addend != 0u )
			{
				float4 al, ah;
				SparkUnpack8(((device const uint4 *)a)[v], al, ah);
				xl = SparkResidualSum4(xl, al);
				xh = SparkResidualSum4(xh, ah);
			}
			total += dot(xl, xl) + dot(xh, xh);
		}
	}
	for ( uint i = tail + tid; i < p.dimension; i += threads )
	{
		float value = SparkMathBf16ToFloat(x[i]);
		if ( p.has_addend != 0u )
			value = SparkMathResidualSum(value, SparkMathBf16ToFloat(a[i]));
		total += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float scale = SparkMathRmsScale(total, float(p.dimension), p.epsilon);
	if ( vec <= threads )
	{
		if ( tid < vec )
		{
			if ( p.has_addend != 0u && p.write_sum != 0u )
				((device uint4 *)x)[tid] = SparkPack8(cl, ch);
			float4 ol = cl * scale, oh = ch * scale;
			if ( p.weightless == 0u )
			{
				const float4 wl = ((device const float4 *)weight)[2u * tid], wh = ((device const float4 *)weight)[2u * tid + 1u];
				ol = p.round_first != 0u ? SparkNormGainRounded4(ol, wl) : ol * wl;
				oh = p.round_first != 0u ? SparkNormGainRounded4(oh, wh) : oh * wh;
			}
			((device uint4 *)y)[tid] = SparkPack8(ol, oh);
		}
	}
	else
	{
		for ( uint v = tid; v < vec; v += threads )
		{
			float4 xl, xh;
			SparkUnpack8(((device const uint4 *)x)[v], xl, xh);
			if ( p.has_addend != 0u )
			{
				float4 al, ah;
				SparkUnpack8(((device const uint4 *)a)[v], al, ah);
				xl = SparkResidualSum4(xl, al);
				xh = SparkResidualSum4(xh, ah);
				if ( p.write_sum != 0u )
					((device uint4 *)x)[v] = SparkPack8(xl, xh);
			}
			float4 ol = xl * scale, oh = xh * scale;
			if ( p.weightless == 0u )
			{
				const float4 wl = ((device const float4 *)weight)[2u * v], wh = ((device const float4 *)weight)[2u * v + 1u];
				ol = p.round_first != 0u ? SparkNormGainRounded4(ol, wl) : ol * wl;
				oh = p.round_first != 0u ? SparkNormGainRounded4(oh, wh) : oh * wh;
			}
			((device uint4 *)y)[v] = SparkPack8(ol, oh);
		}
	}
	for ( uint i = tail + tid; i < p.dimension; i += threads )
	{
		float value = SparkMathBf16ToFloat(x[i]);
		if ( p.has_addend != 0u )
		{
			value = SparkMathResidualSum(value, SparkMathBf16ToFloat(a[i]));
			if ( p.write_sum != 0u )
				x[i] = SparkMathFloatToBf16(value);
		}
		y[i] = SparkMathFloatToBf16(p.weightless != 0u ? value * scale : p.round_first != 0u ? SparkMathNormGainRounded(value * scale, weight[i]) : value * scale * weight[i]);
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
	const uint vec = (((ulong)x | (ulong)weight | (ulong)y) & 15ul) == 0ul ? p.dimension / 8u : 0u, tail = vec * 8u;
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
	for ( uint i = tail + tid; i < p.dimension; i += threads )
	{
		const float value = SparkMathBf16ToFloat(x[i]);
		total += value * value;
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float scale = SparkMathRmsScale(total, float(p.dimension), p.epsilon);
	if ( vec <= threads )
	{
		if ( tid < vec )
		{
			float4 wl, wh;
			SparkUnpack8(((device const uint4 *)weight)[tid], wl, wh);
			((device uint4 *)y)[tid] = SparkPack8(SparkNormGainRounded4(cl * scale, wl), SparkNormGainRounded4(ch * scale, wh));
		}
	}
	else
	{
		for ( uint v = tid; v < vec; v += threads )
		{
			float4 xl, xh, wl, wh;
			SparkUnpack8(((device const uint4 *)x)[v], xl, xh);
			SparkUnpack8(((device const uint4 *)weight)[v], wl, wh);
			((device uint4 *)y)[v] = SparkPack8(SparkNormGainRounded4(xl * scale, wl), SparkNormGainRounded4(xh * scale, wh));
		}
	}
	for ( uint i = tail + tid; i < p.dimension; i += threads )
		y[i] = SparkMathFloatToBf16(SparkMathNormGainRounded(SparkMathBf16ToFloat(x[i]) * scale, SparkMathBf16ToFloat(weight[i])));
}

static inline float SparkGateElem(float x, float g, float w, float scale, constant SparkGatedNormParams &p)
{
	const float act = p.sigmoid_gate != 0u ? SparkMathSigmoidFast(g) : SparkMathSiluFast(g);
	if ( p.round_once != 0u && p.weight_f32 == 0u )
		return SparkMathRmsNormGateOnce(x, scale, w, act);
	float value;
	if ( p.weight_f32 != 0u )
	{
		value = x * scale * w;
		if ( p.round_norm != 0u )
			value = SparkMathRoundBf16(value);
	}
	else
		value = SparkMathRoundBf16(SparkMathRoundBf16(x * scale) * w);
	return value * act;
}

static inline float4 SparkGate4(float4 x, float4 g, float4 w, float scale, constant SparkGatedNormParams &p)
{
	return float4(SparkGateElem(x.x, g.x, w.x, scale, p), SparkGateElem(x.y, g.y, w.y, scale, p),
		SparkGateElem(x.z, g.z, w.z, scale, p), SparkGateElem(x.w, g.w, w.w, scale, p));
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
	const uint vec = p.head_dimension >= 256u && (((ulong)x | (ulong)g | (ulong)weight | (ulong)y) & 15ul) == 0ul ? p.head_dimension / 8u : 0u, tail = vec * 8u;
	float4 cl = 0.0f, ch = 0.0f;
	float cached = 0.0f, total = 0.0f;
	if ( vec != 0u )
	{
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
		for ( uint i = tail + tid; i < p.head_dimension; i += threads )
		{
			const float value = SparkMathBf16ToFloat(x[i]);
			total += value * value;
		}
	}
	else if ( p.head_dimension <= threads )
	{
		if ( tid < p.head_dimension )
		{
			cached = SparkMathBf16ToFloat(x[tid]);
			total = cached * cached;
		}
	}
	else
	{
		for ( uint i = tid; i < p.head_dimension; i += threads )
		{
			const float value = SparkMathBf16ToFloat(x[i]);
			total += value * value;
		}
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	const float scale = SparkMathRmsScale(total, float(p.head_dimension), p.epsilon);
	if ( vec != 0u )
	{
		if ( vec <= threads )
		{
			if ( tid < vec )
			{
				float4 gl, gh, wl, wh;
				SparkUnpack8(((device const uint4 *)g)[tid], gl, gh);
				if ( p.weight_f32 != 0u )
				{
					wl = ((device const float4 *)weight)[2u * tid];
					wh = ((device const float4 *)weight)[2u * tid + 1u];
				}
				else
					SparkUnpack8(((device const uint4 *)weight)[tid], wl, wh);
				((device uint4 *)y)[tid] = SparkPack8(SparkGate4(cl, gl, wl, scale, p), SparkGate4(ch, gh, wh, scale, p));
			}
		}
		else
		{
			for ( uint v = tid; v < vec; v += threads )
			{
				float4 xl, xh, gl, gh, wl, wh;
				SparkUnpack8(((device const uint4 *)x)[v], xl, xh);
				SparkUnpack8(((device const uint4 *)g)[v], gl, gh);
				if ( p.weight_f32 != 0u )
				{
					wl = ((device const float4 *)weight)[2u * v];
					wh = ((device const float4 *)weight)[2u * v + 1u];
				}
				else
					SparkUnpack8(((device const uint4 *)weight)[v], wl, wh);
				((device uint4 *)y)[v] = SparkPack8(SparkGate4(xl, gl, wl, scale, p), SparkGate4(xh, gh, wh, scale, p));
			}
		}
		for ( uint i = tail + tid; i < p.head_dimension; i += threads )
		{
			const float wv = p.weight_f32 != 0u ? ((device const float *)weight)[i] : SparkMathBf16ToFloat(weight[i]);
			y[i] = SparkMathFloatToBf16(SparkGateElem(SparkMathBf16ToFloat(x[i]), SparkMathBf16ToFloat(g[i]), wv, scale, p));
		}
		return;
	}
	if ( p.head_dimension <= threads )
	{
		if ( tid < p.head_dimension )
		{
			const float wv = p.weight_f32 != 0u ? ((device const float *)weight)[tid] : SparkMathBf16ToFloat(weight[tid]);
			y[tid] = SparkMathFloatToBf16(SparkGateElem(cached, SparkMathBf16ToFloat(g[tid]), wv, scale, p));
		}
		return;
	}
	for ( uint i = tid; i < p.head_dimension; i += threads )
	{
		const float wv = p.weight_f32 != 0u ? ((device const float *)weight)[i] : SparkMathBf16ToFloat(weight[i]);
		y[i] = SparkMathFloatToBf16(SparkGateElem(SparkMathBf16ToFloat(x[i]), SparkMathBf16ToFloat(g[i]), wv, scale, p));
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
	const ulong mask_x = (ulong)x | (p.has_weight != 0u ? (ulong)weight : 0ul);
	const uint vec_x = p.head_dimension >= 256u && (mask_x & 15ul) == 0ul ? p.head_dimension / 8u : 0u, tail_x = vec_x * 8u;
	float4 cl = 0.0f, ch = 0.0f;
	float tail_v = 0.0f;
	if ( vec_x != 0u )
	{
		if ( tid < vec_x )
			SparkUnpack8(((device const uint4 *)x)[tid], cl, ch);
		if ( tail_x + tid < p.head_dimension )
			tail_v = SparkMathBf16ToFloat(x[tail_x + tid]);
	}
	else if ( tid < p.head_dimension )
		tail_v = SparkMathBf16ToFloat(x[tid]);
	float scale = 1.0f;
	if ( p.plain == 0u )
	{
		float total = 0.0f;
		if ( vec_x != 0u )
		{
			if ( tid < vec_x )
				total = dot(cl, cl) + dot(ch, ch);
			if ( tail_x + tid < p.head_dimension )
				total += tail_v * tail_v;
		}
		else if ( tid < p.head_dimension )
			total = tail_v * tail_v;
		total = SparkGroupSum(total, scratch, lane, simd, simds);
		scale = SparkMathRmsScale(total, float(p.head_dimension), p.epsilon);
	}
	if ( vec_x != 0u )
	{
		if ( tid < vec_x )
		{
			float4 ol = cl * scale, oh = ch * scale;
			if ( p.has_weight != 0u )
			{
				const float4 wl = ((device const float4 *)weight)[2u * tid], wh = ((device const float4 *)weight)[2u * tid + 1u];
				ol = p.round_first != 0u ? SparkNormGainRounded4(ol, wl) : ol * wl;
				oh = p.round_first != 0u ? SparkNormGainRounded4(oh, wh) : oh * wh;
			}
			ol = SparkRoundBf16x4(ol);
			oh = SparkRoundBf16x4(oh);
			if ( p.multiply != 0.0f )
			{
				ol = SparkRoundBf16x4(ol * p.multiply);
				oh = SparkRoundBf16x4(oh * p.multiply);
			}
			const uint base = tid * 8u;
			normed[base] = ol.x;
			normed[base + 1u] = ol.y;
			normed[base + 2u] = ol.z;
			normed[base + 3u] = ol.w;
			normed[base + 4u] = oh.x;
			normed[base + 5u] = oh.y;
			normed[base + 6u] = oh.z;
			normed[base + 7u] = oh.w;
		}
		if ( tail_x + tid < p.head_dimension )
		{
			const uint i = tail_x + tid;
			const float scaled = tail_v * scale;
			const float value = SparkMathRoundBf16(p.has_weight == 0u ? scaled : p.round_first != 0u ? SparkMathNormGainRounded(scaled, weight[i]) : scaled * weight[i]);
			normed[i] = p.multiply != 0.0f ? SparkMathRoundBf16(value * p.multiply) : value;
		}
	}
	else if ( tid < p.head_dimension )
	{
		const float scaled = tail_v * scale;
		const float value = SparkMathRoundBf16(p.has_weight == 0u ? scaled : p.round_first != 0u ? SparkMathNormGainRounded(scaled, weight[tid]) : scaled * weight[tid]);
		normed[tid] = p.multiply != 0.0f ? SparkMathRoundBf16(value * p.multiply) : value;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	const float position = float(positions[row]);
	for ( uint i = tid; i < half_rope; i += threads )
	{
		const uint lo = p.interleaved != 0u ? 2u * i : i;
		const uint hi = p.interleaved != 0u ? 2u * i + 1u : i + half_rope;
		const float a = normed[lo], b = normed[hi];
		const float angle = position * p.inverse_frequency[i];
		const float cosine = SparkMathRopeCos(angle, p.scale), sine = SparkMathRopeSin(angle, p.scale);
		const float c = p.rope_rounding == SPARK_ROPE_ROUNDING_BF16_TRIG ? SparkMathRoundBf16(cosine) : cosine;
		const float s = p.rope_rounding == SPARK_ROPE_ROUNDING_BF16_TRIG ? SparkMathRoundBf16(sine) : sine;
		if ( p.rope_rounding == SPARK_ROPE_ROUNDING_TERMS )
		{
			y[lo] = SparkMathFloatToBf16(SparkMathRopeRoundedTerm(a, -b, c, s));
			y[hi] = SparkMathFloatToBf16(SparkMathRopeRoundedTerm(b, a, c, s));
		}
		else
		{
			y[lo] = SparkMathFloatToBf16(SparkMathRopeLow(a, b, c, s));
			y[hi] = SparkMathFloatToBf16(SparkMathRopeHigh(a, b, c, s));
		}
	}
	for ( uint i = p.rope_dimension + tid; i < p.head_dimension; i += threads )
		y[i] = SparkMathFloatToBf16(normed[i]);
}

static inline float SparkRmsScale(device const ushort *x, uint dimension, float epsilon, threadgroup float *scratch, uint tid, uint threads, uint lane,
	uint simd, uint simds)
{
	const uint vec = ((ulong)x & 15ul) == 0ul ? dimension / 8u : 0u, tail = vec * 8u;
	float total = 0.0f;
	for ( uint v = tid; v < vec; v += threads )
	{
		float4 xl, xh;
		SparkUnpack8(((device const uint4 *)x)[v], xl, xh);
		total += dot(xl, xl) + dot(xh, xh);
	}
	for ( uint i = tail + tid; i < dimension; i += threads )
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
	device ushort *h = hidden + (ulong)row * p.hidden_stride;
	device const ushort *a = first + (ulong)row * p.branch_stride;
	device const ushort *b = second + (ulong)row * p.branch_stride;
	const ulong mask = (ulong)h | (ulong)a | (ulong)output_weight | (p.branches == 2u ? (ulong)b | (ulong)first_weight | (ulong)second_weight : 0ul);
	if ( (mask & 15ul) == 0ul && (p.dimension & 7u) == 0u )
	{
		const uint vec = p.dimension / 8u;
		float4 al = 0.0f, ah = 0.0f;
		if ( tid < vec )
			SparkUnpack8(((device const uint4 *)a)[tid], al, ah);
		const float scale_a = SparkMathRmsScale(SparkGroupSum(tid < vec ? dot(al, al) + dot(ah, ah) : 0.0f, scratch, lane, simd, simds), float(p.dimension), p.epsilon);
		const float gain = p.has_scalar != 0u ? scalar[0] : 1.0f;
		if ( p.branches == 2u )
		{
			float4 bl = 0.0f, bh = 0.0f;
			if ( tid < vec )
				SparkUnpack8(((device const uint4 *)b)[tid], bl, bh);
			const float scale_b = SparkMathRmsScale(SparkGroupSum(tid < vec ? dot(bl, bl) + dot(bh, bh) : 0.0f, scratch, lane, simd, simds), float(p.dimension), p.epsilon);
			float4 cl = 0.0f, ch = 0.0f;
			if ( tid < vec )
			{
				const float4 w1l = ((device const float4 *)first_weight)[2u * tid], w1h = ((device const float4 *)first_weight)[2u * tid + 1u];
				const float4 w2l = ((device const float4 *)second_weight)[2u * tid], w2h = ((device const float4 *)second_weight)[2u * tid + 1u];
				cl = SparkRoundBf16x4(SparkRoundBf16x4(al * scale_a * w1l) + SparkRoundBf16x4(bl * scale_b * w2l));
				ch = SparkRoundBf16x4(SparkRoundBf16x4(ah * scale_a * w1h) + SparkRoundBf16x4(bh * scale_b * w2h));
			}
			const float scale = SparkMathRmsScale(SparkGroupSum(tid < vec ? dot(cl, cl) + dot(ch, ch) : 0.0f, scratch, lane, simd, simds), float(p.dimension), p.epsilon);
			if ( tid < vec )
			{
				float4 hl, hh;
				SparkUnpack8(((device const uint4 *)h)[tid], hl, hh);
				const float4 wol = ((device const float4 *)output_weight)[2u * tid], woh = ((device const float4 *)output_weight)[2u * tid + 1u];
				float4 ol = SparkRoundBf16x4(hl + SparkRoundBf16x4(cl * scale * wol));
				float4 oh = SparkRoundBf16x4(hh + SparkRoundBf16x4(ch * scale * woh));
				if ( p.has_scalar != 0u )
				{
					ol *= gain;
					oh *= gain;
				}
				((device uint4 *)h)[tid] = SparkPack8(ol, oh);
			}
			return;
		}
		if ( tid < vec )
		{
			float4 hl, hh;
			SparkUnpack8(((device const uint4 *)h)[tid], hl, hh);
			const float4 wol = ((device const float4 *)output_weight)[2u * tid], woh = ((device const float4 *)output_weight)[2u * tid + 1u];
			float4 ol = SparkRoundBf16x4(hl + SparkRoundBf16x4(al * scale_a * wol));
			float4 oh = SparkRoundBf16x4(hh + SparkRoundBf16x4(ah * scale_a * woh));
			if ( p.has_scalar != 0u )
			{
				ol *= gain;
				oh *= gain;
			}
			((device uint4 *)h)[tid] = SparkPack8(ol, oh);
		}
		return;
	}
	const float scale_a = SparkRmsScale(a, p.dimension, p.epsilon, scratch, tid, threads, lane, simd, simds);
	if ( p.branches == 2u )
	{
		const float scale_b = SparkRmsScale(b, p.dimension, p.epsilon, scratch, tid, threads, lane, simd, simds);
		float combined[8];
		float total = 0.0f;
		uint slot = 0u;
		for ( uint i = tid; i < p.dimension; i += threads )
		{
			const float value = SparkMathRoundBf16(SparkMathRoundBf16(SparkMathBf16ToFloat(a[i]) * scale_a * first_weight[i]) +
				SparkMathRoundBf16(SparkMathBf16ToFloat(b[i]) * scale_b * second_weight[i]));
			combined[slot++] = value;
			total += value * value;
		}
		const float scale = SparkMathRmsScale(SparkGroupSum(total, scratch, lane, simd, simds), float(p.dimension), p.epsilon);
		slot = 0u;
		for ( uint i = tid; i < p.dimension; i += threads )
		{
			const float added = SparkMathRoundBf16(SparkMathBf16ToFloat(h[i]) + SparkMathRoundBf16(combined[slot++] * scale * output_weight[i]));
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
	device ushort *y = output + (ulong)row * p.hidden_stride;
	const uint vec = (((ulong)x | (ulong)scale | (ulong)y) & 15ul) == 0ul ? p.dimension / 8u : 0u, tail = vec * 8u;
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
	for ( uint i = tail + tid; i < p.dimension; i += threads )
	{
		const float value = SparkMathBf16ToFloat(x[i]);
		total += value * value;
	}
	const float inverse = SparkMathRmsScale(SparkGroupSum(total, scratch, lane, simd, simds), float(p.dimension), p.epsilon);
	if ( vec <= threads )
	{
		if ( tid < vec )
		{
			const float4 sl = ((device const float4 *)scale)[2u * tid], sh = ((device const float4 *)scale)[2u * tid + 1u];
			const float4 ol = SparkRoundBf16x4(SparkRoundBf16x4(cl * inverse) * sl) * p.output_scale;
			const float4 oh = SparkRoundBf16x4(SparkRoundBf16x4(ch * inverse) * sh) * p.output_scale;
			((device uint4 *)y)[tid] = SparkPack8(ol, oh);
		}
	}
	else
	{
		for ( uint v = tid; v < vec; v += threads )
		{
			float4 xl, xh;
			SparkUnpack8(((device const uint4 *)x)[v], xl, xh);
			const float4 sl = ((device const float4 *)scale)[2u * v], sh = ((device const float4 *)scale)[2u * v + 1u];
			const float4 ol = SparkRoundBf16x4(SparkRoundBf16x4(xl * inverse) * sl) * p.output_scale;
			const float4 oh = SparkRoundBf16x4(SparkRoundBf16x4(xh * inverse) * sh) * p.output_scale;
			((device uint4 *)y)[v] = SparkPack8(ol, oh);
		}
	}
	for ( uint i = tail + tid; i < p.dimension; i += threads )
	{
		const float normed = SparkMathRoundBf16(SparkMathBf16ToFloat(x[i]) * inverse);
		y[i] = SparkMathFloatToBf16(SparkMathRoundBf16(normed * scale[i]) * p.output_scale);
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
	const uint vec = (((ulong)values | (ulong)weight) & 15ul) == 0ul ? p.dimension / 8u : 0u, tail = vec * 8u;
	float4 cl = 0.0f, ch = 0.0f;
	float square = 0.0f, acc = 0.0f;
	if ( vec <= threads )
	{
		if ( tid < vec )
		{
			SparkUnpack8(((device const uint4 *)values)[tid], cl, ch);
			square = dot(cl, cl) + dot(ch, ch);
		}
	}
	else
	{
		for ( uint v = tid; v < vec; v += threads )
		{
			float4 vl, vh;
			SparkUnpack8(((device const uint4 *)values)[v], vl, vh);
			square += dot(vl, vl) + dot(vh, vh);
		}
	}
	for ( uint i = tail + tid; i < p.dimension; i += threads )
	{
		const float value = SparkMathBf16ToFloat(values[i]);
		square += value * value;
	}
	square = SparkGroupSum(square, scratch, lane, simd, simds);
	const float inverse = SparkMathRmsScale(square, float(p.dimension), p.epsilon);
	if ( vec <= threads )
	{
		if ( tid < vec )
		{
			float4 wl, wh;
			SparkUnpack8(((device const uint4 *)weight)[tid], wl, wh);
			acc = dot(cl * inverse, wl) + dot(ch * inverse, wh);
		}
	}
	else
	{
		for ( uint v = tid; v < vec; v += threads )
		{
			float4 vl, vh, wl, wh;
			SparkUnpack8(((device const uint4 *)values)[v], vl, vh);
			SparkUnpack8(((device const uint4 *)weight)[v], wl, wh);
			acc += dot(vl * inverse, wl) + dot(vh * inverse, wh);
		}
	}
	for ( uint i = tail + tid; i < p.dimension; i += threads )
		acc = fma(SparkMathBf16ToFloat(values[i]) * inverse, SparkMathBf16ToFloat(weight[i]), acc);
	acc = SparkGroupSum(acc, scratch, lane, simd, simds);
	if ( tid == 0u )
		scores[(ulong)row * p.score_stride + source] = acc;
}

kernel void spark_attnres_mix(constant SparkAttnResParams &p [[buffer(0)]], device const ushort *bank [[buffer(1)]],
	device const ushort *partial [[buffer(2)]], device const float *scores [[buffer(3)]], device ushort *output [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	threadgroup float weights[SPARK_ATTNRES_MAX_SOURCES];
	const uint row = group.y;
	if ( (((ulong)bank | (ulong)partial | (ulong)output) & 15ul) == 0ul && (p.dimension & 7u) == 0u && group_size.x == 256u )
	{
		if ( tid >= 32u )
			return;
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
		simdgroup_barrier(mem_flags::mem_threadgroup);
		const uint v = group.x * 32u + tid;
		if ( v * 8u >= p.dimension )
			return;
		float4 tl = 0.0f, th = 0.0f;
		for ( uint source = 0u; source < p.sources; source++ )
		{
			float4 sl, sh;
			SparkUnpack8(((device const uint4 *)SparkAttnResSource(p, bank, partial, source, row))[v], sl, sh);
			const float w = weights[source];
			tl = fma(w, sl, tl);
			th = fma(w, sh, th);
		}
		((device uint4 *)(output + (ulong)row * p.dimension))[v] = SparkPack8(tl, th);
		return;
	}
	const uint index = group.x * group_size.x + tid;
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
