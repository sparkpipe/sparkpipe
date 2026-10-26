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

kernel void spark_silu_mul(constant SparkSiluMulParams &p [[buffer(0)]], device const ushort *input [[buffer(1)]],
	device ushort *output [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint2 group_size [[threads_per_threadgroup]])
{
	const uint threads = group_size.x;
	const uint row = group.y, index = group.x * threads + tid;
	if ( index >= p.dimension )
		return;
	device const ushort *source = input + (ulong)row * p.input_stride;
	const float gate = SparkMathBf16ToFloat(source[index]), up = SparkMathBf16ToFloat(source[p.dimension + index]);
	output[(ulong)row * p.output_stride + index] = SparkMathFloatToBf16(SparkMathSilu(gate) * up);
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
