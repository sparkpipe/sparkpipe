#include "inference/kernels/metal/common.metal"

kernel void spark_gather_rows(constant SparkGatherParams &p [[buffer(0)]], device const ushort *table [[buffer(1)]],
	device const uint *ids [[buffer(2)]], device ushort *output [[buffer(3)]], device atomic_uint *error [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint row = group.y, chunk = group.x * threads + tid, id = ids[row] - p.first_row;
	if ( chunk * 8u >= p.width )
		return;
	if ( id >= p.table_rows && p.sliced != 0u )
	{
		*(device uint4 *)(output + (ulong)row * p.width + chunk * 8u) = uint4(0u);
		return;
	}
	if ( id >= p.table_rows )
	{
		atomic_store_explicit(error, 1u + row, memory_order_relaxed);
		return;
	}
	*(device uint4 *)(output + (ulong)row * p.width + chunk * 8u) = *(device const uint4 *)(table + (ulong)id * p.width + chunk * 8u);
}

kernel void spark_add_rows(constant SparkSiluMulParams &p [[buffer(0)]], device ushort *hidden [[buffer(1)]],
	device const ushort *addend [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint row = group.y, index = (group.x * threads + tid) * 8u;
	if ( index >= p.dimension )
		return;
	device ushort *h = hidden + (ulong)row * p.input_stride + index;
	device const ushort *a = addend + (ulong)row * p.output_stride + index;
	if ( index + 8u <= p.dimension && (((ulong)h | (ulong)a) & 15ul) == 0ul )
	{
		float4 hl, hh, al, ah;
		SparkUnpack8(*(device const uint4 *)h, hl, hh);
		SparkUnpack8(*(device const uint4 *)a, al, ah);
		*(device uint4 *)h = SparkPack8(hl + al, hh + ah);
		return;
	}
	for ( uint k = 0u; k < 8u && index + k < p.dimension; k++ )
		h[k] = SparkMathFloatToBf16(SparkMathBf16ToFloat(h[k]) + SparkMathBf16ToFloat(a[k]));
}

template <uint ACTIVATION>
static inline float SparkGatedActivate(float gate, float up, float limit, float up_limit)
{
	return ACTIVATION == SPARK_GATED_GELU_TANH ? SparkMathRoundBf16(SparkMathGeluTanh(gate)) * up :
		ACTIVATION == SPARK_GATED_SILU_ROUNDED ? SparkMathRoundBf16(SparkMathSilu(gate)) * up :
		ACTIVATION == SPARK_GATED_SILU_ROUNDED_CLAMP ? SparkMathClampedGlu(SparkMathRoundBf16(SparkMathSilu(gate)), up, limit) :
		ACTIVATION == SPARK_GATED_SILU_CLAMP ? SparkMathClampedGlu(SparkMathSiluFast(gate), up, limit) :
		ACTIVATION == SPARK_GATED_SILU_GATE_LIMIT ? SparkMathGateLimitedGlu(gate, up, limit) :
		ACTIVATION == SPARK_GATED_SITU ? SparkMathSitu(gate, up, limit, up_limit) : SparkMathSilu(gate) * up;
}

template <uint ACTIVATION>
kernel void spark_gated_mul(constant SparkSiluMulParams &p [[buffer(0)]], device const ushort *input [[buffer(1)]],
	device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint row = group.y, index = (group.x * threads + tid) * 4u;
	if ( index >= p.dimension )
		return;
	device const ushort *g_ptr = input + (ulong)row * p.input_stride + index;
	device const ushort *u_ptr = g_ptr + p.dimension;
	device ushort *dst = output + (ulong)row * p.output_stride + index;
	if ( index + 4u <= p.dimension && (((ulong)g_ptr | (ulong)u_ptr | (ulong)dst) & 7ul) == 0ul )
	{
		const float4 g = SparkUnpack4(*(device const uint2 *)g_ptr);
		const float4 u = SparkUnpack4(*(device const uint2 *)u_ptr);
		const float4 y = float4(
			SparkGatedActivate<ACTIVATION>(g.x, u.x, p.limit, p.up_limit),
			SparkGatedActivate<ACTIVATION>(g.y, u.y, p.limit, p.up_limit),
			SparkGatedActivate<ACTIVATION>(g.z, u.z, p.limit, p.up_limit),
			SparkGatedActivate<ACTIVATION>(g.w, u.w, p.limit, p.up_limit));
		*(device uint2 *)dst = SparkPack4(y);
		return;
	}
	for ( uint k = 0u; k < 4u && index + k < p.dimension; k++ )
		dst[k] = SparkMathFloatToBf16(SparkGatedActivate<ACTIVATION>(SparkMathBf16ToFloat(g_ptr[k]), SparkMathBf16ToFloat(u_ptr[k]), p.limit, p.up_limit));
}

template [[host_name("spark_silu_mul")]] kernel void spark_gated_mul<SPARK_GATED_SILU>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);
template [[host_name("spark_gelu_tanh_mul")]] kernel void spark_gated_mul<SPARK_GATED_GELU_TANH>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);
template [[host_name("spark_silu_rounded_mul")]] kernel void spark_gated_mul<SPARK_GATED_SILU_ROUNDED>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);
template [[host_name("spark_silu_clamp_mul")]] kernel void spark_gated_mul<SPARK_GATED_SILU_CLAMP>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);
template [[host_name("spark_silu_rounded_clamp_mul")]] kernel void spark_gated_mul<SPARK_GATED_SILU_ROUNDED_CLAMP>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);
template [[host_name("spark_silu_gate_limit_mul")]] kernel void spark_gated_mul<SPARK_GATED_SILU_GATE_LIMIT>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);
template [[host_name("spark_situ_mul")]] kernel void spark_gated_mul<SPARK_GATED_SITU>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);

template <uint ACTIVATION>
static inline float SparkGatedRoutedActivate(float gate, float up, float weight, float limit)
{
	return ACTIVATION == SPARK_GATED_GELU_TANH ? SparkMathRoundBf16(SparkMathGeluTanhScaleFirst(gate)) * up * weight :
		ACTIVATION == SPARK_GATED_SILU_CLAMP ? SparkMathClampedGlu(SparkMathSilu(gate), up, limit) * weight :
		ACTIVATION == SPARK_GATED_SILU_GATE_LIMIT ? SparkMathGateLimitedGlu(gate, up, limit) * weight : SparkMathSilu(gate) * up * weight;
}

template <uint ACTIVATION>
kernel void spark_gated_mul_routed(constant SparkSiluMulParams &p [[buffer(0)]], device const ushort *input [[buffer(1)]],
	device const float *weights [[buffer(2)]], device ushort *output [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint row = group.y, index = (group.x * threads + tid) * 4u;
	if ( index >= p.dimension )
		return;
	const float weight = weights[row];
	device const ushort *g_ptr = input + (ulong)row * p.input_stride + index;
	device const ushort *u_ptr = g_ptr + p.dimension;
	device ushort *dst = output + (ulong)row * p.output_stride + index;
	if ( index + 4u <= p.dimension && (((ulong)g_ptr | (ulong)u_ptr | (ulong)dst) & 7ul) == 0ul )
	{
		const float4 g = SparkUnpack4(*(device const uint2 *)g_ptr);
		const float4 u = SparkUnpack4(*(device const uint2 *)u_ptr);
		const float4 y = float4(
			SparkGatedRoutedActivate<ACTIVATION>(g.x, u.x, weight, p.limit),
			SparkGatedRoutedActivate<ACTIVATION>(g.y, u.y, weight, p.limit),
			SparkGatedRoutedActivate<ACTIVATION>(g.z, u.z, weight, p.limit),
			SparkGatedRoutedActivate<ACTIVATION>(g.w, u.w, weight, p.limit));
		*(device uint2 *)dst = SparkPack4(y);
		return;
	}
	for ( uint k = 0u; k < 4u && index + k < p.dimension; k++ )
		dst[k] = SparkMathFloatToBf16(SparkGatedRoutedActivate<ACTIVATION>(SparkMathBf16ToFloat(g_ptr[k]), SparkMathBf16ToFloat(u_ptr[k]), weight, p.limit));
}

template [[host_name("spark_silu_mul_routed")]] kernel void spark_gated_mul_routed<SPARK_GATED_SILU>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device const float *weights [[buffer(2)]], device ushort *output [[buffer(3)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);
template [[host_name("spark_gelu_tanh_mul_routed")]] kernel void spark_gated_mul_routed<SPARK_GATED_GELU_TANH>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device const float *weights [[buffer(2)]], device ushort *output [[buffer(3)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);
template [[host_name("spark_silu_clamp_mul_routed")]] kernel void spark_gated_mul_routed<SPARK_GATED_SILU_CLAMP>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device const float *weights [[buffer(2)]], device ushort *output [[buffer(3)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);
template [[host_name("spark_silu_gate_limit_mul_routed")]] kernel void spark_gated_mul_routed<SPARK_GATED_SILU_GATE_LIMIT>(
	constant SparkSiluMulParams &p [[buffer(0)]], device const ushort *input [[buffer(1)]], device const float *weights [[buffer(2)]],
	device ushort *output [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]]);

kernel void spark_scale_rows(constant SparkScaleParams &p [[buffer(0)]], device ushort *values [[buffer(1)]], uint index [[thread_position_in_grid]])
{
	const ulong base = (ulong)index * 8u;
	if ( base >= p.count )
		return;
	device ushort *v = values + base;
	if ( base + 8u <= p.count && ((ulong)v & 15ul) == 0ul )
	{
		float4 lo, hi;
		SparkUnpack8(*(device const uint4 *)v, lo, hi);
		*(device uint4 *)v = SparkPack8(lo * p.scale, hi * p.scale);
		return;
	}
	for ( ulong k = 0u; k < 8u && base + k < p.count; k++ )
		v[k] = SparkMathFloatToBf16(SparkMathBf16ToFloat(v[k]) * p.scale);
}

kernel void spark_softcap_rows(constant SparkScaleParams &p [[buffer(0)]], device float *values [[buffer(1)]], uint index [[thread_position_in_grid]])
{
	const ulong base = (ulong)index * 4u;
	if ( base >= p.count )
		return;
	device float *v = values + base;
	if ( base + 4u <= p.count && ((ulong)v & 15ul) == 0ul )
	{
		float4 x = *(device const float4 *)v;
		if ( p.multiply != 0.0f )
			x *= p.multiply;
		if ( p.scale != 0.0f )
			x = float4(SparkMathSoftcap(x.x, p.scale), SparkMathSoftcap(x.y, p.scale), SparkMathSoftcap(x.z, p.scale), SparkMathSoftcap(x.w, p.scale));
		*(device float4 *)v = x;
		return;
	}
	for ( ulong k = 0u; k < 4u && base + k < p.count; k++ )
	{
		const float value = p.multiply != 0.0f ? v[k] * p.multiply : v[k];
		v[k] = p.scale != 0.0f ? SparkMathSoftcap(value, p.scale) : value;
	}
}

static inline void SparkRouteProbabilities(constant SparkRouteParams &p, device const ushort *values, threadgroup float *probability,
	threadgroup float *scratch, device atomic_uint *error, uint row, uint tid, uint threads, uint lane, uint simd, uint simds)
{
	if ( p.experts <= threads )
	{
		float logit = -INFINITY;
		if ( tid < p.experts )
		{
			logit = SparkMathBf16ToFloat(values[tid]);
			if ( !isfinite(logit) )
				atomic_store_explicit(error, 1u + row, memory_order_relaxed);
		}
		const float best = SparkGroupMax(logit, scratch, lane, simd, simds);
		const float e = tid < p.experts ? exp(logit - best) : 0.0f;
		const float total = SparkGroupSum(e, scratch, lane, simd, simds);
		if ( tid < p.experts )
			probability[tid] = e / total;
		threadgroup_barrier(mem_flags::mem_threadgroup);
		return;
	}
	float best = -INFINITY;
	for ( uint i = tid; i < p.experts; i += threads )
	{
		const float logit = SparkMathBf16ToFloat(values[i]);
		if ( !isfinite(logit) )
			atomic_store_explicit(error, 1u + row, memory_order_relaxed);
		best = max(best, logit);
	}
	best = SparkGroupMax(best, scratch, lane, simd, simds);
	float total = 0.0f;
	for ( uint i = tid; i < p.experts; i += threads )
	{
		probability[i] = exp(SparkMathBf16ToFloat(values[i]) - best);
		total += probability[i];
	}
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	for ( uint i = tid; i < p.experts; i += threads )
		probability[i] /= total;
	threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline void SparkRouteWeights(constant SparkRouteParams &p, threadgroup const float *probability, threadgroup const uint *chosen,
	device const float *expert_scale, device uint *experts, device float *weights, uint row)
{
	const uint width = p.top_k <= 1u ? 1u : p.top_k <= 2u ? 2u : p.top_k <= 4u ? 4u : p.top_k <= 8u ? 8u : 16u;
	float tree[SPARK_ROUTE_MAX_TOP_K];
	for ( uint k = 0u; k < width; k++ )
		tree[k] = k < p.top_k ? probability[chosen[k]] : 0.0f;
	for ( uint step = width / 2u; step > 0u; step >>= 1u )
		for ( uint k = 0u; k < step; k++ )
			tree[k] = tree[k] + tree[k + step];
	const float total = tree[0] + 1e-20f;
	for ( uint k = 0u; k < p.top_k; k++ )
	{
		experts[(ulong)row * p.top_k + k] = chosen[k];
		weights[(ulong)row * p.top_k + k] = probability[chosen[k]] / total * expert_scale[chosen[k]];
	}
}

static inline bool SparkRouteAbove(float first, uint first_index, float second, uint second_index)
{
	if ( isnan(first) != isnan(second) )
		return isnan(second);
	if ( !isnan(first) && first != second )
		return first > second;
	return first_index < second_index;
}

kernel void spark_route_topk(constant SparkRouteParams &p [[buffer(0)]], device const ushort *logits [[buffer(1)]],
	device const float *expert_scale [[buffer(2)]], device uint *experts [[buffer(3)]], device float *weights [[buffer(4)]],
	device atomic_uint *error [[buffer(5)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	threadgroup float probability[SPARK_ROUTE_MAX_EXPERTS];
	threadgroup uchar used[SPARK_ROUTE_MAX_EXPERTS];
	threadgroup float simd_value[32];
	threadgroup uint simd_index[32];
	threadgroup uint chosen[SPARK_ROUTE_MAX_TOP_K];
	SparkRouteProbabilities(p, logits + (ulong)row * p.logit_stride, probability, scratch, error, row, tid, threads, lane, simd, simds);
	for ( uint i = tid; i < p.experts; i += threads )
		used[i] = 0u;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint pick = 0u; pick < p.top_k; pick++ )
	{
		float best_val = 0.0f;
		uint best_idx = 0xffffffffu;
		for ( uint i = tid; i < p.experts; i += threads )
		{
			if ( used[i] != 0u )
				continue;
			const float v = probability[i];
			if ( best_idx == 0xffffffffu || SparkRouteAbove(v, i, best_val, best_idx) )
			{
				best_val = v;
				best_idx = i;
			}
		}
		for ( uint offset = 16u; offset > 0u; offset >>= 1u )
		{
			const float ov = simd_shuffle_xor(best_val, offset);
			const uint oi = simd_shuffle_xor(best_idx, offset);
			if ( oi != 0xffffffffu && (best_idx == 0xffffffffu || SparkRouteAbove(ov, oi, best_val, best_idx)) )
			{
				best_val = ov;
				best_idx = oi;
			}
		}
		if ( lane == 0u )
		{
			simd_value[simd] = best_val;
			simd_index[simd] = best_idx;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( tid == 0u )
		{
			float tv = simd_value[0];
			uint ti = simd_index[0];
			for ( uint other = 1u; other < simds; other++ )
			{
				const uint oi = simd_index[other];
				if ( oi != 0xffffffffu && (ti == 0xffffffffu || SparkRouteAbove(simd_value[other], oi, tv, ti)) )
				{
					tv = simd_value[other];
					ti = oi;
				}
			}
			chosen[pick] = ti;
			used[ti] = 1u;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	if ( tid == 0u )
		SparkRouteWeights(p, probability, chosen, expert_scale, experts, weights, row);
}

kernel void spark_route_pinned(constant SparkRouteParams &p [[buffer(0)]], device const ushort *logits [[buffer(1)]],
	device const float *expert_scale [[buffer(2)]], device uint *experts [[buffer(3)]], device float *weights [[buffer(4)]],
	device const uint *pinned [[buffer(5)]], device atomic_uint *error [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]],
	uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
	uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	threadgroup float probability[SPARK_ROUTE_MAX_EXPERTS];
	threadgroup uint chosen[SPARK_ROUTE_MAX_TOP_K];
	SparkRouteProbabilities(p, logits + (ulong)row * p.logit_stride, probability, scratch, error, row, tid, threads, lane, simd, simds);
	if ( tid < p.top_k )
		chosen[tid] = pinned[(ulong)row * p.top_k + tid];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( tid == 0u )
		SparkRouteWeights(p, probability, chosen, expert_scale, experts, weights, row);
}

static inline float SparkHeadGateActivation(uint activation, float logit)
{
	return activation == SPARK_HEAD_GATE_SIGMOID_BF16 ? SparkMathRoundBf16(SparkMathSigmoid(logit)) :
		activation == SPARK_HEAD_GATE_SIGMOID ? SparkMathSigmoidFast(logit) : SparkMathSoftplusGate(logit);
}

kernel void spark_head_gate(constant SparkHeadGateParams &p [[buffer(0)]], device ushort *values [[buffer(1)]],
	device const ushort *logits [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint head = group.x, row = group.y, D = p.head_dimension;
	device ushort *base = values + ((ulong)row * p.heads + head) * D;
	if ( p.gate_head_stride != 0u )
	{
		device const ushort *gates = logits + (ulong)row * p.gate_stride + (ulong)head * p.gate_head_stride + p.gate_offset;
		if ( (D & 7u) == 0u && (((ulong)base | (ulong)gates) & 15ul) == 0ul )
		{
			const uint chunks = D >> 3u;
			for ( uint c = tid; c < chunks; c += group_size.x )
			{
				float4 vl, vh, gl, gh;
				SparkUnpack8(((device const uint4 *)base)[c], vl, vh);
				SparkUnpack8(((device const uint4 *)gates)[c], gl, gh);
				vl *= float4(SparkHeadGateActivation(p.activation, gl.x), SparkHeadGateActivation(p.activation, gl.y),
					SparkHeadGateActivation(p.activation, gl.z), SparkHeadGateActivation(p.activation, gl.w));
				vh *= float4(SparkHeadGateActivation(p.activation, gh.x), SparkHeadGateActivation(p.activation, gh.y),
					SparkHeadGateActivation(p.activation, gh.z), SparkHeadGateActivation(p.activation, gh.w));
				((device uint4 *)base)[c] = SparkPack8(vl, vh);
			}
			return;
		}
		for ( uint index = tid; index < D; index += group_size.x )
			base[index] = SparkMathFloatToBf16(SparkMathBf16ToFloat(base[index]) * SparkHeadGateActivation(p.activation, SparkMathBf16ToFloat(gates[index])));
		return;
	}
	const float gate = SparkHeadGateActivation(p.activation, SparkMathBf16ToFloat(logits[(ulong)row * p.gate_stride + head]));
	if ( (D & 7u) == 0u && ((ulong)base & 15ul) == 0ul )
	{
		const uint chunks = D >> 3u;
		for ( uint c = tid; c < chunks; c += group_size.x )
		{
			float4 vl, vh;
			SparkUnpack8(((device const uint4 *)base)[c], vl, vh);
			((device uint4 *)base)[c] = SparkPack8(vl * gate, vh * gate);
		}
		return;
	}
	for ( uint index = tid; index < D; index += group_size.x )
		base[index] = SparkMathFloatToBf16(SparkMathBf16ToFloat(base[index]) * gate);
}

kernel void spark_route_local(constant SparkRouteParams &p [[buffer(0)]], device uint *experts [[buffer(1)]], uint index [[thread_position_in_grid]])
{
	if ( index >= p.rows * p.top_k )
		return;
	const uint local = experts[index] - p.first_expert;
	experts[index] = local < p.local_experts ? local : SPARK_ROUTE_SKIP;
}

kernel void spark_route_combine(constant SparkRouteParams &p [[buffer(0)]], device const ushort *expert_out [[buffer(1)]],
	device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, index = (group.x * group_size.x + tid) * 4u;
	if ( index >= p.width )
		return;
	device ushort *dst = output + (ulong)row * p.width + index;
	if ( index + 4u <= p.width && (p.width & 3u) == 0u && (((ulong)expert_out | (ulong)dst) & 7ul) == 0ul )
	{
		float4 total = float4(0.0f);
		for ( uint k = 0u; k < p.top_k; k++ )
			total += SparkUnpack4(*(device const uint2 *)(expert_out + ((ulong)row * p.top_k + k) * p.width + index));
		*(device uint2 *)dst = SparkPack4(total);
		return;
	}
	for ( uint c = 0u; c < 4u && index + c < p.width; c++ )
	{
		float total = 0.0f;
		for ( uint k = 0u; k < p.top_k; k++ )
			total += SparkMathBf16ToFloat(expert_out[((ulong)row * p.top_k + k) * p.width + index + c]);
		dst[c] = SparkMathFloatToBf16(total);
	}
}

kernel void spark_route_combine_weighted(constant SparkRouteParams &p [[buffer(0)]], device const ushort *expert_out [[buffer(1)]],
	device const float *weights [[buffer(2)]], device ushort *output [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, index = (group.x * group_size.x + tid) * 4u;
	if ( index >= p.width )
		return;
	device ushort *dst = output + (ulong)row * p.width + index;
	if ( index + 4u <= p.width && (p.width & 3u) == 0u && (((ulong)expert_out | (ulong)dst) & 7ul) == 0ul )
	{
		float4 total = float4(0.0f);
		for ( uint k = 0u; k < p.top_k; k++ )
		{
			const ulong route = (ulong)row * p.top_k + k;
			total = fma(SparkUnpack4(*(device const uint2 *)(expert_out + route * p.width + index)), float4(weights[route]), total);
		}
		*(device uint2 *)dst = SparkPack4(total);
		return;
	}
	for ( uint c = 0u; c < 4u && index + c < p.width; c++ )
	{
		float total = 0.0f;
		for ( uint k = 0u; k < p.top_k; k++ )
		{
			const ulong route = (ulong)row * p.top_k + k;
			total = fma(SparkMathBf16ToFloat(expert_out[route * p.width + index + c]), weights[route], total);
		}
		dst[c] = SparkMathFloatToBf16(total);
	}
}

kernel void spark_route_combine_weighted_shared(constant SparkRouteParams &p [[buffer(0)]], device const ushort *expert_out [[buffer(1)]],
	device const float *weights [[buffer(2)]], device const ushort *shared [[buffer(3)]], device ushort *output [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, index = (group.x * group_size.x + tid) * 4u;
	if ( index >= p.width )
		return;
	device const ushort *sh = shared + (ulong)row * p.width + index;
	device ushort *dst = output + (ulong)row * p.width + index;
	if ( index + 4u <= p.width && (p.width & 3u) == 0u && (((ulong)expert_out | (ulong)sh | (ulong)dst) & 7ul) == 0ul )
	{
		float4 total = float4(0.0f);
		for ( uint k = 0u; k < p.top_k; k++ )
		{
			const ulong route = (ulong)row * p.top_k + k;
			total = fma(SparkUnpack4(*(device const uint2 *)(expert_out + route * p.width + index)), float4(weights[route]), total);
		}
		total += SparkUnpack4(*(device const uint2 *)sh);
		*(device uint2 *)dst = SparkPack4(total);
		return;
	}
	for ( uint c = 0u; c < 4u && index + c < p.width; c++ )
	{
		float total = 0.0f;
		for ( uint k = 0u; k < p.top_k; k++ )
		{
			const ulong route = (ulong)row * p.top_k + k;
			total = fma(SparkMathBf16ToFloat(expert_out[route * p.width + index + c]), weights[route], total);
		}
		total += SparkMathBf16ToFloat(sh[c]);
		dst[c] = SparkMathFloatToBf16(total);
	}
}

kernel void spark_route_combine_weighted_shared_f32(constant SparkRouteParams &p [[buffer(0)]], device const float *expert_out [[buffer(1)]],
	device const float *weights [[buffer(2)]], device const float *shared [[buffer(3)]], device float *output [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, index = (group.x * group_size.x + tid) * 4u;
	if ( index >= p.width )
		return;
	device const float *sh = shared + (ulong)row * p.width + index;
	device float *dst = output + (ulong)row * p.width + index;
	if ( index + 4u <= p.width && (p.width & 3u) == 0u && (((ulong)expert_out | (ulong)sh | (ulong)dst) & 15ul) == 0ul )
	{
		float4 total = float4(0.0f);
		for ( uint k = 0u; k < p.top_k; k++ )
		{
			const ulong route = (ulong)row * p.top_k + k;
			total = fma(*(device const float4 *)(expert_out + route * p.width + index), float4(weights[route]), total);
		}
		*(device float4 *)dst = total + *(device const float4 *)sh;
		return;
	}
	for ( uint c = 0u; c < 4u && index + c < p.width; c++ )
	{
		float total = 0.0f;
		for ( uint k = 0u; k < p.top_k; k++ )
		{
			const ulong route = (ulong)row * p.top_k + k;
			total = fma(expert_out[route * p.width + index + c], weights[route], total);
		}
		dst[c] = total + sh[c];
	}
}

template <uint SCORE>
static inline void SparkRouteScores(constant SparkRouteParams &p, device const float *logits, device const float *bias, threadgroup float *score,
	threadgroup float *selection, uint row, uint tid, uint threads)
{
	for ( uint expert = tid; expert < p.experts; expert += threads )
	{
		const float logit = logits[(ulong)row * p.logit_stride + expert];
		const float rounded = p.round_logits != 0u ? SparkMathRoundBf16(logit) : logit;
		const float value = SCORE == SPARK_ROUTE_SCORE_SQRT_SOFTPLUS ? SparkMathSqrtSoftplus(rounded) : SparkMathSigmoid(rounded);
		score[expert] = value;
		selection[expert] = value + bias[expert];
	}
}

static inline void SparkRouteScoredWeights(constant SparkRouteParams &p, threadgroup const float *score, threadgroup const uint *chosen,
	device uint *experts, device float *weights, uint row)
{
	float total = 0.0f;
	if ( p.tree_sum != 0u )
	{
		float tree[SPARK_ROUTE_MAX_TOP_K];
		uint width = 1u;
		while ( width < p.top_k )
			width <<= 1u;
		for ( uint pick = 0u; pick < width; pick++ )
			tree[pick] = pick < p.top_k ? score[chosen[pick]] : 0.0f;
		for ( uint step = width / 2u; step > 0u; step >>= 1u )
			for ( uint pick = 0u; pick < step; pick++ )
				tree[pick] = tree[pick] + tree[pick + step];
		total = tree[0];
	}
	else
		for ( uint pick = 0u; pick < p.top_k; pick++ )
			total += score[chosen[pick]];
	for ( uint pick = 0u; pick < p.top_k; pick++ )
	{
		experts[(ulong)row * p.top_k + pick] = chosen[pick];
		weights[(ulong)row * p.top_k + pick] = score[chosen[pick]] / (total + 1e-20f) * p.scale;
	}
}

template <uint SCORE>
kernel void spark_route_scored_topk(constant SparkRouteParams &p [[buffer(0)]], device const float *logits [[buffer(1)]],
	device const float *bias [[buffer(2)]], device uint *experts [[buffer(3)]], device float *weights [[buffer(4)]],
	uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float score[SPARK_ROUTE_MAX_EXPERTS];
	threadgroup float selection[SPARK_ROUTE_MAX_EXPERTS];
	threadgroup float simd_value[2][32];
	threadgroup uint simd_index[2][32];
	threadgroup uint chosen[SPARK_ROUTE_MAX_TOP_K];
	threadgroup float group_score[SPARK_ROUTE_MAX_GROUPS];
	threadgroup uint group_open[SPARK_ROUTE_MAX_GROUPS];
	if ( p.experts <= threads )
	{
		float my_sel = -INFINITY;
		if ( tid < p.experts )
		{
			const float logit = logits[(ulong)row * p.logit_stride + tid];
			const float rounded = p.round_logits != 0u ? SparkMathRoundBf16(logit) : logit;
			const float value = SCORE == SPARK_ROUTE_SCORE_SQRT_SOFTPLUS ? SparkMathSqrtSoftplus(rounded) : SparkMathSigmoid(rounded);
			score[tid] = value;
			my_sel = value + bias[tid];
			if ( p.groups > 1u )
				selection[tid] = my_sel;
		}
		if ( p.groups > 1u )
		{
			const uint per_group = p.experts / p.groups;
			if ( per_group == 32u )
			{
				if ( simd < p.groups )
				{
					float v1 = my_sel;
					uint i1 = lane;
					for ( uint offset = 16u; offset > 0u; offset >>= 1u )
					{
						const float ov = simd_shuffle_xor(v1, offset);
						const uint oi = simd_shuffle_xor(i1, offset);
						if ( ov > v1 || (ov == v1 && oi < i1) )
						{
							v1 = ov;
							i1 = oi;
						}
					}
					const float v2 = simd_max(lane == i1 ? -INFINITY : my_sel);
					if ( lane == 0u )
					{
						group_score[simd] = v1 + v2;
						group_open[simd] = 0u;
					}
				}
				threadgroup_barrier(mem_flags::mem_threadgroup);
			}
			else
			{
				threadgroup_barrier(mem_flags::mem_threadgroup);
				if ( tid < p.groups )
				{
					float first = -INFINITY, second = -INFINITY;
					for ( uint index = 0u; index < per_group; index++ )
					{
						const float value = selection[tid * per_group + index];
						if ( value > first )
						{
							second = first;
							first = value;
						}
						else if ( value > second )
							second = value;
					}
					group_score[tid] = first + second;
					group_open[tid] = 0u;
				}
				threadgroup_barrier(mem_flags::mem_threadgroup);
			}
			if ( tid == 0u )
				for ( uint pick = 0u; pick < p.topk_groups; pick++ )
				{
					uint best = p.groups;
					for ( uint candidate = 0u; candidate < p.groups; candidate++ )
						if ( group_open[candidate] == 0u && (best == p.groups || group_score[candidate] > group_score[best]) )
							best = candidate;
					group_open[best] = 1u;
				}
			threadgroup_barrier(mem_flags::mem_threadgroup);
			if ( tid < p.experts && group_open[tid / per_group] == 0u )
				my_sel = -INFINITY;
		}
		for ( uint pick = 0u; pick < p.top_k; pick++ )
		{
			float value = my_sel;
			uint index = tid < p.experts ? tid : 0xffffffffu;
			for ( uint offset = 16u; offset > 0u; offset >>= 1u )
			{
				const float other_value = simd_shuffle_xor(value, offset);
				const uint other_index = simd_shuffle_xor(index, offset);
				if ( other_value > value || (other_value == value && other_index < index) )
				{
					value = other_value;
					index = other_index;
				}
			}
			const uint buf = pick & 1u;
			if ( lane == 0u )
			{
				simd_value[buf][simd] = value;
				simd_index[buf][simd] = index;
			}
			threadgroup_barrier(mem_flags::mem_threadgroup);
			float gval = lane < simds ? simd_value[buf][lane] : -INFINITY;
			uint gidx = lane < simds ? simd_index[buf][lane] : 0xffffffffu;
			for ( uint offset = 16u; offset > 0u; offset >>= 1u )
			{
				const float ov = simd_shuffle_xor(gval, offset);
				const uint oi = simd_shuffle_xor(gidx, offset);
				if ( ov > gval || (ov == gval && oi < gidx) )
				{
					gval = ov;
					gidx = oi;
				}
			}
			if ( tid == gidx )
				my_sel = -INFINITY;
			if ( tid == 0u )
				chosen[pick] = gidx;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( tid == 0u )
			SparkRouteScoredWeights(p, score, chosen, experts, weights, row);
		return;
	}
	SparkRouteScores<SCORE>(p, logits, bias, score, selection, row, tid, threads);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( p.groups > 1u )
	{
		const uint per_group = p.experts / p.groups;
		if ( tid < p.groups )
		{
			float first = -INFINITY, second = -INFINITY;
			for ( uint index = 0u; index < per_group; index++ )
			{
				const float value = selection[tid * per_group + index];
				if ( value > first )
				{
					second = first;
					first = value;
				}
				else if ( value > second )
					second = value;
			}
			group_score[tid] = first + second;
			group_open[tid] = 0u;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( tid == 0u )
			for ( uint pick = 0u; pick < p.topk_groups; pick++ )
			{
				uint best = p.groups;
				for ( uint candidate = 0u; candidate < p.groups; candidate++ )
					if ( group_open[candidate] == 0u && (best == p.groups || group_score[candidate] > group_score[best]) )
						best = candidate;
				group_open[best] = 1u;
			}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( uint expert = tid; expert < p.experts; expert += threads )
			if ( group_open[expert / per_group] == 0u )
				selection[expert] = -INFINITY;
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	for ( uint pick = 0u; pick < p.top_k; pick++ )
	{
		float value = -INFINITY;
		uint index = 0xffffffffu;
		for ( uint expert = tid; expert < p.experts; expert += threads )
			if ( selection[expert] > value || (selection[expert] == value && expert < index) )
			{
				value = selection[expert];
				index = expert;
			}
		for ( uint offset = 16u; offset > 0u; offset >>= 1u )
		{
			const float other_value = simd_shuffle_xor(value, offset);
			const uint other_index = simd_shuffle_xor(index, offset);
			if ( other_value > value || (other_value == value && other_index < index) )
			{
				value = other_value;
				index = other_index;
			}
		}
		if ( lane == 0u )
		{
			simd_value[0][simd] = value;
			simd_index[0][simd] = index;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( tid == 0u )
		{
			value = simd_value[0][0];
			index = simd_index[0][0];
			for ( uint other = 1u; other < simds; other++ )
				if ( simd_value[0][other] > value || (simd_value[0][other] == value && simd_index[0][other] < index) )
				{
					value = simd_value[0][other];
					index = simd_index[0][other];
				}
			chosen[pick] = index;
			selection[index] = -INFINITY;
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	if ( tid == 0u )
		SparkRouteScoredWeights(p, score, chosen, experts, weights, row);
}

template <uint SCORE>
kernel void spark_route_scored_pinned(constant SparkRouteParams &p [[buffer(0)]], device const float *logits [[buffer(1)]],
	device const float *bias [[buffer(2)]], device uint *experts [[buffer(3)]], device float *weights [[buffer(4)]],
	device const uint *pinned [[buffer(5)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint threads [[threads_per_threadgroup]])
{
	threadgroup float score[SPARK_ROUTE_MAX_EXPERTS];
	threadgroup float selection[SPARK_ROUTE_MAX_EXPERTS];
	threadgroup uint chosen[SPARK_ROUTE_MAX_TOP_K];
	SparkRouteScores<SCORE>(p, logits, bias, score, selection, row, tid, threads);
	if ( tid < p.top_k )
		chosen[tid] = pinned[(ulong)row * p.top_k + tid];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( tid == 0u )
		SparkRouteScoredWeights(p, score, chosen, experts, weights, row);
}

#define SPARK_ROUTE_SCORED_INSTANCE(name, score) \
	template [[host_name(name)]] kernel void spark_route_scored_topk<score>(constant SparkRouteParams &p [[buffer(0)]], \
		device const float *logits [[buffer(1)]], device const float *bias [[buffer(2)]], device uint *experts [[buffer(3)]], \
		device float *weights [[buffer(4)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
		uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], \
		uint simds [[simdgroups_per_threadgroup]]);

SPARK_ROUTE_SCORED_INSTANCE("spark_route_sigmoid_topk", SPARK_ROUTE_SCORE_SIGMOID)
SPARK_ROUTE_SCORED_INSTANCE("spark_route_sqrt_softplus_topk", SPARK_ROUTE_SCORE_SQRT_SOFTPLUS)

#define SPARK_ROUTE_PINNED_INSTANCE(name, score) \
	template [[host_name(name)]] kernel void spark_route_scored_pinned<score>(constant SparkRouteParams &p [[buffer(0)]], \
		device const float *logits [[buffer(1)]], device const float *bias [[buffer(2)]], device uint *experts [[buffer(3)]], \
		device float *weights [[buffer(4)]], device const uint *pinned [[buffer(5)]], uint row [[threadgroup_position_in_grid]], \
		uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]);

SPARK_ROUTE_PINNED_INSTANCE("spark_route_sigmoid_pinned", SPARK_ROUTE_SCORE_SIGMOID)
SPARK_ROUTE_PINNED_INSTANCE("spark_route_sqrt_softplus_pinned", SPARK_ROUTE_SCORE_SQRT_SOFTPLUS)

kernel void spark_logprob_rows(constant SparkLogprobParams &p [[buffer(0)]], device const float *logits [[buffer(1)]],
	device const uint *targets [[buffer(2)]], device float *result [[buffer(3)]], device uint *argmax [[buffer(4)]],
	uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float scratch[32];
	threadgroup uint index_scratch[32];
	device const float *values = logits + (ulong)row * p.logit_stride;
	float best = -INFINITY;
	uint best_index = 0xffffffffu;
	const bool aligned4 = ((ulong)values & 15ul) == 0ul;
	const uint vec_end = aligned4 ? (p.vocabulary & ~3u) : 0u;
	if ( aligned4 )
	{
		device const float4 *v4 = (device const float4 *)values;
		const uint chunks = vec_end >> 2u;
		for ( uint c = tid; c < chunks; c += threads )
		{
			const float4 v = v4[c];
			const uint base = c << 2u;
			if ( v.x > best ) { best = v.x; best_index = base; }
			if ( v.y > best ) { best = v.y; best_index = base + 1u; }
			if ( v.z > best ) { best = v.z; best_index = base + 2u; }
			if ( v.w > best ) { best = v.w; best_index = base + 3u; }
		}
	}
	for ( uint i = vec_end + tid; i < p.vocabulary; i += threads )
		if ( values[i] > best )
		{
			best = values[i];
			best_index = i;
		}
	const float group_best = SparkGroupMax(best, scratch, lane, simd, simds);
	uint candidate = best == group_best ? best_index : 0xffffffffu;
	candidate = simd_min(candidate);
	if ( lane == 0u )
		index_scratch[simd] = candidate;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	uint chosen = lane < simds ? index_scratch[lane] : 0xffffffffu;
	chosen = simd_min(chosen);
	float total = 0.0f;
	if ( aligned4 )
	{
		device const float4 *v4 = (device const float4 *)values;
		const uint chunks = vec_end >> 2u;
		for ( uint c = tid; c < chunks; c += threads )
		{
			const float4 d = v4[c] - group_best;
			total += (exp(d.x) + exp(d.y)) + (exp(d.z) + exp(d.w));
		}
	}
	for ( uint i = vec_end + tid; i < p.vocabulary; i += threads )
		total += exp(values[i] - group_best);
	total = SparkGroupSum(total, scratch, lane, simd, simds);
	if ( tid == 0u )
	{
		const float normalizer = group_best + log(total);
		const uint target = targets[row];
		result[(ulong)row * 4u] = group_best - normalizer;
		result[(ulong)row * 4u + 1u] = normalizer;
		result[(ulong)row * 4u + 2u] = target < p.vocabulary ? values[target] - normalizer : -INFINITY;
		result[(ulong)row * 4u + 3u] = group_best;
		argmax[row] = chosen;
	}
}

kernel void spark_stream_read(constant SparkStreamParams &p [[buffer(0)]], device const uint4 *words [[buffer(1)]],
	device uint *sink [[buffer(2)]], uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint threads [[threads_per_threadgroup]])
{
	uint4 total = uint4(0u);
	const ulong stride = (ulong)p.groups * threads;
	for ( ulong i = (ulong)group * threads + tid; i < p.words; i += stride )
		total ^= words[i];
	const uint folded = total.x ^ total.y ^ total.z ^ total.w;
	if ( folded == 0x9e3779b9u )
		sink[group] = folded;
}

kernel void spark_route_combine_shared(constant SparkRouteParams &p [[buffer(0)]], device const ushort *expert_out [[buffer(1)]],
	device const ushort *shared [[buffer(2)]], device ushort *output [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, index = (group.x * group_size.x + tid) * 4u;
	if ( index >= p.width )
		return;
	device const ushort *sh = shared + (ulong)row * p.width + index;
	device ushort *dst = output + (ulong)row * p.width + index;
	if ( index + 4u <= p.width && (p.width & 3u) == 0u && (((ulong)expert_out | (ulong)sh | (ulong)dst) & 7ul) == 0ul )
	{
		float4 total = float4(0.0f);
		for ( uint k = 0u; k < p.top_k; k++ )
			total += SparkUnpack4(*(device const uint2 *)(expert_out + ((ulong)row * p.top_k + k) * p.width + index));
		total += SparkUnpack4(*(device const uint2 *)sh);
		*(device uint2 *)dst = SparkPack4(total);
		return;
	}
	for ( uint c = 0u; c < 4u && index + c < p.width; c++ )
	{
		float total = 0.0f;
		for ( uint k = 0u; k < p.top_k; k++ )
			total += SparkMathBf16ToFloat(expert_out[((ulong)row * p.top_k + k) * p.width + index + c]);
		total += SparkMathBf16ToFloat(sh[c]);
		dst[c] = SparkMathFloatToBf16(total);
	}
}

kernel void spark_sigmoid_gate_rows(constant SparkRouteParams &p [[buffer(0)]], device ushort *values [[buffer(1)]],
	device const ushort *logits [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, index = (group.x * group_size.x + tid) * 4u;
	if ( index >= p.width )
		return;
	const float gate = SparkMathSigmoid(SparkMathBf16ToFloat(logits[(ulong)row * p.logit_stride]));
	device ushort *v = values + (ulong)row * p.width + index;
	if ( index + 4u <= p.width && ((ulong)v & 7ul) == 0ul )
	{
		*(device uint2 *)v = SparkPack4(SparkUnpack4(*(device const uint2 *)v) * gate);
		return;
	}
	for ( uint c = 0u; c < 4u && index + c < p.width; c++ )
		v[c] = SparkMathFloatToBf16(SparkMathBf16ToFloat(v[c]) * gate);
}

kernel void spark_block_qdq(constant SparkQdqParams &p [[buffer(0)]], device const ushort *input [[buffer(1)]], device ushort *output [[buffer(2)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint size = SPARK_QDQ_BLOCK(p.format), block = group.x * group_size.x + tid, row = group.y;
	if ( row >= p.rows || block * size >= p.width )
		return;
	device const ushort *x = input + (ulong)row * p.input_stride + block * size;
	device ushort *y = output + (ulong)row * p.output_stride + block * size;
	if ( (size & 7u) == 0u && (((ulong)x | (ulong)y) & 15ul) == 0ul )
	{
		float4 vals[8];
		float4 amax4 = float4(0.0f);
		const uint chunks = size >> 3u;
		for ( uint c = 0u; c < chunks; c++ )
		{
			SparkUnpack8(((device const uint4 *)x)[c], vals[2u * c], vals[2u * c + 1u]);
			amax4 = fmax(amax4, fmax(fabs(vals[2u * c]), fabs(vals[2u * c + 1u])));
		}
		const float amax = fmax(fmax(amax4.x, amax4.y), fmax(amax4.z, amax4.w));
		const float scale = p.format == SPARK_QDQ_FP8 ? SparkMathFp8QdqScale(amax) : p.format == SPARK_QDQ_FP4 ? SparkMathFp4QdqScale(amax) :
			SparkMathFp4Pow2QdqScale(amax);
		for ( uint c = 0u; c < chunks; c++ )
		{
			float4 lo = vals[2u * c], hi = vals[2u * c + 1u];
			if ( p.format == SPARK_QDQ_FP8 )
			{
				lo = float4(SparkMathFp8Qdq(lo.x, scale), SparkMathFp8Qdq(lo.y, scale), SparkMathFp8Qdq(lo.z, scale), SparkMathFp8Qdq(lo.w, scale));
				hi = float4(SparkMathFp8Qdq(hi.x, scale), SparkMathFp8Qdq(hi.y, scale), SparkMathFp8Qdq(hi.z, scale), SparkMathFp8Qdq(hi.w, scale));
			}
			else
			{
				lo = float4(SparkMathFp4Qdq(lo.x, scale), SparkMathFp4Qdq(lo.y, scale), SparkMathFp4Qdq(lo.z, scale), SparkMathFp4Qdq(lo.w, scale));
				hi = float4(SparkMathFp4Qdq(hi.x, scale), SparkMathFp4Qdq(hi.y, scale), SparkMathFp4Qdq(hi.z, scale), SparkMathFp4Qdq(hi.w, scale));
			}
			((device uint4 *)y)[c] = SparkPack8(lo, hi);
		}
		return;
	}
	float values[32];
	float amax = 0.0f;
	for ( uint i = 0u; i < size; i++ )
	{
		values[i] = SparkMathBf16ToFloat(x[i]);
		amax = fmax(amax, fabs(values[i]));
	}
	const float scale = p.format == SPARK_QDQ_FP8 ? SparkMathFp8QdqScale(amax) : p.format == SPARK_QDQ_FP4 ? SparkMathFp4QdqScale(amax) :
		SparkMathFp4Pow2QdqScale(amax);
	for ( uint i = 0u; i < size; i++ )
		y[i] = SparkMathFloatToBf16(p.format == SPARK_QDQ_FP8 ? SparkMathFp8Qdq(values[i], scale) : SparkMathFp4Qdq(values[i], scale));
}

static inline uint SparkSequenceOfRow(device const uint *row_begin, uint sequences, uint row)
{
	uint lane = 0u;
	while ( lane + 1u < sequences && row_begin[lane + 1u] <= row )
		lane++;
	return lane;
}

kernel void spark_kv_compress(constant SparkCompressParams &p [[buffer(0)]], device const float *kv [[buffer(1)]], device const float *score [[buffer(2)]],
	device const float *state [[buffer(3)]], device const uint *positions [[buffer(4)]], device const uint *row_begin [[buffer(5)]],
	device const uint *state_index [[buffer(6)]], device ushort *pooled [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, c = group.x * group_size.x + tid;
	if ( row >= p.rows || c >= p.width )
		return;
	const ulong at = (ulong)row * p.width + c;
	if ( p.ratio == 1u )
	{
		pooled[at] = SparkMathFloatToBf16(kv[at]);
		return;
	}
	if ( (positions[row] + 1u) % p.ratio != 0u )
		return;
	const uint lane = SparkSequenceOfRow(row_begin, p.sequences, row);
	device const float *carried = state + (ulong)state_index[lane] * p.state_stride;
	const bool inside = row > row_begin[lane];
	const float first = inside ? kv[at - p.width] : carried[c], first_score = inside ? score[at - p.width] : carried[p.width + c];
	pooled[at] = SparkMathFloatToBf16(SparkMathSoftmaxPool2(first, first_score, kv[at], score[at]));
}

kernel void spark_kv_compress_save(constant SparkCompressParams &p [[buffer(0)]], device const float *kv [[buffer(1)]],
	device const float *score [[buffer(2)]], device float *state [[buffer(3)]], device const uint *positions [[buffer(4)]],
	device const uint *row_begin [[buffer(5)]], device const uint *state_index [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint lane = group.y, c = group.x * group_size.x + tid;
	if ( p.commit == 0u || lane >= p.sequences || c >= p.width || row_begin[lane + 1u] == row_begin[lane] )
		return;
	const uint row = row_begin[lane + 1u] - 1u;
	if ( (positions[row] + 1u) % p.ratio == 0u )
		return;
	device float *carried = state + (ulong)state_index[lane] * p.state_stride;
	carried[c] = kv[(ulong)row * p.width + c];
	carried[p.width + c] = score[(ulong)row * p.width + c];
}

kernel void spark_compressed_positions(constant SparkCompressParams &p [[buffer(0)]], device const uint *positions [[buffer(1)]],
	device uint *store [[buffer(2)]], device uint *rope [[buffer(3)]], device uint *last [[buffer(4)]], uint row [[thread_position_in_grid]])
{
	if ( row >= p.rows )
		return;
	const uint position = positions[row];
	store[row] = (position + 1u) % p.ratio == 0u ? position / p.ratio : SPARK_KV_SKIP;
	rope[row] = position + 1u - p.ratio;
	last[row] = (position + 1u) / p.ratio - 1u;
}

static inline float4 SparkGroupSum4(float4 value, threadgroup float4 *scratch, uint lane, uint simd, uint simds)
{
	value = simd_sum(value);
	if ( lane == 0u )
		scratch[simd] = value;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float4 total = lane < simds ? scratch[lane] : float4(0.0f);
	total = simd_sum(total);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	return total;
}

kernel void spark_poly_norm_mul(constant SparkPolyNormParams &p [[buffer(0)]], device const ushort *gate_up [[buffer(1)]],
	device const float *params [[buffer(2)]], device const uint *param_of_row [[buffer(3)]], device ushort *output [[buffer(4)]],
	uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	threadgroup float4 scratch[32];
	if ( row >= p.rows )
		return;
	device const ushort *gate = gate_up + (ulong)row * p.input_stride, *up = gate + p.width;
	device ushort *y = output + (ulong)row * p.output_stride;
	const uint index = p.indexed != 0u ? param_of_row[row] : 0u;
	if ( index >= p.param_count )
	{
		for ( uint i = tid; i < p.width; i += threads )
			y[i] = 0u;
		return;
	}
	const bool vec8 = (p.width & 7u) == 0u && (((ulong)gate | (ulong)up | (ulong)y) & 15ul) == 0ul;
	const uint chunks = p.width >> 3u;
	const bool cached = vec8 && chunks <= threads;
	float4 g_lo_c = float4(0.0f), g_hi_c = float4(0.0f);
	float4 sums = float4(0.0f);
	if ( vec8 )
	{
		if ( cached )
		{
			if ( tid < chunks )
			{
				SparkUnpack8(((device const uint4 *)gate)[tid], g_lo_c, g_hi_c);
				g_lo_c = clamp(g_lo_c, -p.clamp, p.clamp);
				g_hi_c = clamp(g_hi_c, -p.clamp, p.clamp);
				const float4 g2l = g_lo_c * g_lo_c, g2h = g_hi_c * g_hi_c;
				const float4 g3l = g2l * g_lo_c, g3h = g2h * g_hi_c;
				sums.x = dot(g_lo_c, g_lo_c) + dot(g_hi_c, g_hi_c);
				sums.y = dot(g2l, g2l) + dot(g2h, g2h);
				sums.z = dot(g3l, g3l) + dot(g3h, g3h);
			}
		}
		else
		{
			for ( uint c = tid; c < chunks; c += threads )
			{
				float4 gl, gh;
				SparkUnpack8(((device const uint4 *)gate)[c], gl, gh);
				gl = clamp(gl, -p.clamp, p.clamp);
				gh = clamp(gh, -p.clamp, p.clamp);
				const float4 g2l = gl * gl, g2h = gh * gh;
				const float4 g3l = g2l * gl, g3h = g2h * gh;
				sums.x += dot(gl, gl) + dot(gh, gh);
				sums.y += dot(g2l, g2l) + dot(g2h, g2h);
				sums.z += dot(g3l, g3l) + dot(g3h, g3h);
			}
		}
	}
	else
	{
		for ( uint i = tid; i < p.width; i += threads )
		{
			const float g = SparkMathClamp(SparkMathBf16ToFloat(gate[i]), p.clamp), g2 = g * g, g3 = g2 * g;
			sums.x += g * g;
			sums.y += g2 * g2;
			sums.z += g3 * g3;
		}
	}
	sums = SparkGroupSum4(sums, scratch, lane, simd, simds);
	const float count = float(p.width);
	const float d1 = sqrt(sums.x / count + p.epsilon), d2 = sqrt(sums.y / count + p.epsilon), d3 = sqrt(sums.z / count + p.epsilon);
	device const float *w = params + 4ul * index;
	const float w0 = SparkMathSigmoid(w[0]), w1 = SparkMathSigmoid(w[1]), w2 = SparkMathSigmoid(w[2]);
	const float bias = p.routed != 0u ? SparkMathClamp(w[3], p.bias_clamp) : w[3];
	if ( vec8 )
	{
		for ( uint c = tid; c < chunks; c += threads )
		{
			float4 gl, gh, ul, uh;
			if ( cached )
			{
				gl = g_lo_c;
				gh = g_hi_c;
			}
			else
			{
				SparkUnpack8(((device const uint4 *)gate)[c], gl, gh);
				gl = clamp(gl, -p.clamp, p.clamp);
				gh = clamp(gh, -p.clamp, p.clamp);
			}
			SparkUnpack8(((device const uint4 *)up)[c], ul, uh);
			ul = clamp(ul, -p.clamp, p.clamp);
			uh = clamp(uh, -p.clamp, p.clamp);
			float4 vl = float4(
				SparkMathPolyNorm(gl.x, d1, d2, d3, w0, w1, w2, bias) * ul.x,
				SparkMathPolyNorm(gl.y, d1, d2, d3, w0, w1, w2, bias) * ul.y,
				SparkMathPolyNorm(gl.z, d1, d2, d3, w0, w1, w2, bias) * ul.z,
				SparkMathPolyNorm(gl.w, d1, d2, d3, w0, w1, w2, bias) * ul.w);
			float4 vh = float4(
				SparkMathPolyNorm(gh.x, d1, d2, d3, w0, w1, w2, bias) * uh.x,
				SparkMathPolyNorm(gh.y, d1, d2, d3, w0, w1, w2, bias) * uh.y,
				SparkMathPolyNorm(gh.z, d1, d2, d3, w0, w1, w2, bias) * uh.z,
				SparkMathPolyNorm(gh.w, d1, d2, d3, w0, w1, w2, bias) * uh.w);
			if ( p.routed != 0u )
			{
				vl = clamp(vl, -p.clamp, p.clamp) * p.output_scale;
				vh = clamp(vh, -p.clamp, p.clamp) * p.output_scale;
			}
			else
			{
				vl = SparkRoundBf16x4(vl) * p.output_scale;
				vh = SparkRoundBf16x4(vh) * p.output_scale;
			}
			((device uint4 *)y)[c] = SparkPack8(vl, vh);
		}
		return;
	}
	for ( uint i = tid; i < p.width; i += threads )
	{
		const float g = SparkMathClamp(SparkMathBf16ToFloat(gate[i]), p.clamp), u = SparkMathClamp(SparkMathBf16ToFloat(up[i]), p.clamp);
		const float value = SparkMathPolyNorm(g, d1, d2, d3, w0, w1, w2, bias) * u;
		y[i] = p.routed != 0u ? SparkMathFloatToBf16(SparkMathClamp(value, p.clamp) * p.output_scale) :
			SparkMathFloatToBf16(SparkMathRoundBf16(value) * p.output_scale);
	}
}

kernel void spark_differential_gate(constant SparkDifferentialGateParams &p [[buffer(0)]], device const ushort *heads [[buffer(1)]],
	device const ushort *lambda [[buffer(2)]], device const ushort *gate [[buffer(3)]], device ushort *output [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, element = group.x * group_size.x + tid, width = p.groups * p.signal * p.dimension;
	if ( row >= p.rows || element >= width )
		return;
	const uint head = element / p.dimension, d = element % p.dimension, team = head / p.signal, member = head % p.signal;
	device const ushort *base = heads + (ulong)row * p.heads_stride + (ulong)team * (p.signal + 1u) * p.dimension;
	const float signal = SparkMathBf16ToFloat(base[(ulong)member * p.dimension + d]), noise = SparkMathBf16ToFloat(base[(ulong)p.signal * p.dimension + d]);
	const float weight = SparkMathBf16ToFloat(lambda[(ulong)row * p.lambda_stride + p.lambda_offset + head]);
	const float opened = SparkMathBf16ToFloat(gate[(ulong)row * p.gate_stride + element]);
	output[(ulong)row * p.output_stride + element] = SparkMathFloatToBf16(SparkMathDifferentialGate(signal, noise, weight, opened));
}
