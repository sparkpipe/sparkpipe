#include "inference/kernels/metal/common.metal"

kernel void spark_gather_rows(constant SparkGatherParams &p [[buffer(0)]], device const ushort *table [[buffer(1)]],
	device const uint *ids [[buffer(2)]], device ushort *output [[buffer(3)]], device atomic_uint *error [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint row = group.y, chunk = group.x * threads + tid, id = ids[row];
	if ( chunk * 8u >= p.width )
		return;
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
	const uint row = group.y, index = group.x * threads + tid;
	if ( index >= p.dimension )
		return;
	const ulong at = (ulong)row * p.input_stride + index;
	hidden[at] = SparkMathFloatToBf16(SparkMathBf16ToFloat(hidden[at]) + SparkMathBf16ToFloat(addend[(ulong)row * p.output_stride + index]));
}

template <uint ACTIVATION>
kernel void spark_gated_mul(constant SparkSiluMulParams &p [[buffer(0)]], device const ushort *input [[buffer(1)]],
	device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint row = group.y, index = group.x * threads + tid;
	if ( index >= p.dimension )
		return;
	device const ushort *source = input + (ulong)row * p.input_stride;
	const float gate = SparkMathBf16ToFloat(source[index]), up = SparkMathBf16ToFloat(source[p.dimension + index]);
	const float activated = ACTIVATION == SPARK_GATED_GELU_TANH ? SparkMathRoundBf16(SparkMathGeluTanh(gate)) * up : SparkMathSilu(gate) * up;
	output[(ulong)row * p.output_stride + index] = SparkMathFloatToBf16(activated);
}

template [[host_name("spark_silu_mul")]] kernel void spark_gated_mul<SPARK_GATED_SILU>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);
template [[host_name("spark_gelu_tanh_mul")]] kernel void spark_gated_mul<SPARK_GATED_GELU_TANH>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);

template <uint ACTIVATION>
kernel void spark_gated_mul_routed(constant SparkSiluMulParams &p [[buffer(0)]], device const ushort *input [[buffer(1)]],
	device const float *weights [[buffer(2)]], device ushort *output [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint row = group.y, index = group.x * threads + tid;
	if ( index >= p.dimension )
		return;
	device const ushort *source = input + (ulong)row * p.input_stride;
	const float gate = SparkMathBf16ToFloat(source[index]), up = SparkMathBf16ToFloat(source[p.dimension + index]), weight = weights[row];
	const float activated = ACTIVATION == SPARK_GATED_GELU_TANH ? SparkMathRoundBf16(SparkMathGeluTanhScaleFirst(gate)) * up * weight :
		SparkMathSilu(gate) * up * weight;
	output[(ulong)row * p.output_stride + index] = SparkMathFloatToBf16(activated);
}

template [[host_name("spark_silu_mul_routed")]] kernel void spark_gated_mul_routed<SPARK_GATED_SILU>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device const float *weights [[buffer(2)]], device ushort *output [[buffer(3)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);
template [[host_name("spark_gelu_tanh_mul_routed")]] kernel void spark_gated_mul_routed<SPARK_GATED_GELU_TANH>(constant SparkSiluMulParams &p [[buffer(0)]],
	device const ushort *input [[buffer(1)]], device const float *weights [[buffer(2)]], device ushort *output [[buffer(3)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]);

kernel void spark_scale_rows(constant SparkScaleParams &p [[buffer(0)]], device ushort *values [[buffer(1)]], uint index [[thread_position_in_grid]])
{
	if ( index < p.count )
		values[index] = SparkMathFloatToBf16(SparkMathBf16ToFloat(values[index]) * p.scale);
}

kernel void spark_softcap_rows(constant SparkScaleParams &p [[buffer(0)]], device float *values [[buffer(1)]], uint index [[thread_position_in_grid]])
{
	if ( index < p.count )
	{
		const float value = p.multiply != 0.0f ? values[index] * p.multiply : values[index];
		values[index] = p.scale != 0.0f ? SparkMathSoftcap(value, p.scale) : value;
	}
}

static inline void SparkRouteProbabilities(constant SparkRouteParams &p, device const ushort *values, threadgroup float *probability,
	threadgroup float *scratch, device atomic_uint *error, uint row, uint tid, uint threads, uint lane, uint simd, uint simds)
{
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
	threadgroup uint chosen[SPARK_ROUTE_MAX_TOP_K];
	SparkRouteProbabilities(p, logits + (ulong)row * p.logit_stride, probability, scratch, error, row, tid, threads, lane, simd, simds);
	for ( uint i = tid; i < p.experts; i += threads )
	{
		uint rank = 0u;
		for ( uint j = 0u; j < p.experts; j++ )
			rank += SparkRouteAbove(probability[j], j, probability[i], i) ? 1u : 0u;
		if ( rank < p.top_k )
			chosen[rank] = i;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
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

kernel void spark_route_combine(constant SparkRouteParams &p [[buffer(0)]], device const ushort *expert_out [[buffer(1)]],
	device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint row = group.y, index = group.x * group_size.x + tid;
	if ( index >= p.width )
		return;
	float total = 0.0f;
	for ( uint k = 0u; k < p.top_k; k++ )
		total += SparkMathBf16ToFloat(expert_out[((ulong)row * p.top_k + k) * p.width + index]);
	output[(ulong)row * p.width + index] = SparkMathFloatToBf16(total);
}

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
	for ( uint i = tid; i < p.vocabulary; i += threads )
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
	for ( uint i = tid; i < p.vocabulary; i += threads )
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
