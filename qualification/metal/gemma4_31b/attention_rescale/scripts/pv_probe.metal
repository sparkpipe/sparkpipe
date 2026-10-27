#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void probe_pv(device const float *p [[buffer(1)]], device const bfloat *v [[buffer(2)]], device float *out [[buffer(3)]],
	uint tid [[thread_index_in_threadgroup]])
{
	threadgroup float block[32 * 64];
	threadgroup bfloat high[32 * 64];
	threadgroup bfloat low[32 * 64];
	for ( uint i = tid; i < 32u * 64u; i += 128u )
	{
		block[i] = p[i];
		high[i] = bfloat(p[i]);
		low[i] = bfloat(p[i] - float(high[i]));
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	constexpr auto desc = matmul2d_descriptor(32, 256, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
	matmul2d<desc, execution_simdgroups<4>> op;
	tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> weights(block, dextents<int32_t, 2>(64, 32), array<int32_t, 2>{ 1, 64 });
	tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> weights_high(high, dextents<int32_t, 2>(64, 32), array<int32_t, 2>{ 1, 64 });
	tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> weights_low(low, dextents<int32_t, 2>(64, 32), array<int32_t, 2>{ 1, 64 });
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> values((device bfloat *)v, dextents<int32_t, 2>(256, 64), array<int32_t, 2>{ 1, 256 });
	auto acc = op.template get_destination_cooperative_tensor<decltype(weights), decltype(values), float>();
	for ( uint16_t i = 0; i < acc.get_capacity(); ++i )
		if ( acc.is_valid_element(i) )
			acc[i] = 0.0f;
	op.run(weights, values, acc);
	for ( uint16_t i = 0; i < acc.get_capacity(); ++i )
		if ( acc.is_valid_element(i) )
		{
			const auto index = acc.get_multidimensional_index(i);
			out[index[1] * 256 + index[0]] = acc[i];
		}
	auto split = op.template get_destination_cooperative_tensor<decltype(weights_high), decltype(values), float>();
	for ( uint16_t i = 0; i < split.get_capacity(); ++i )
		if ( split.is_valid_element(i) )
			split[i] = 0.0f;
	op.run(weights_low, values, split);
	op.run(weights_high, values, split);
	for ( uint16_t i = 0; i < split.get_capacity(); ++i )
		if ( split.is_valid_element(i) )
		{
			const auto index = split.get_multidimensional_index(i);
			out[32 * 256 + index[1] * 256 + index[0]] = split[i];
		}
}
