#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void probe_pv_tiles(device const float *p [[buffer(1)]], device bfloat *v [[buffer(2)]], device float *out [[buffer(3)]],
	uint tid [[thread_index_in_threadgroup]])
{
	constexpr uint ROWS = 16, D = 512, TILE = 64, TILES = 16;
	threadgroup float block[ROWS * TILE];
	constexpr auto desc = matmul2d_descriptor(ROWS, D, static_cast<int>(dynamic_extent), false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
	matmul2d<desc, execution_simdgroups<4>> op;
	tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> weights(block, dextents<int32_t, 2>(TILE, ROWS), array<int32_t, 2>{ 1, int(TILE) });
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> first(v, dextents<int32_t, 2>(D, TILE), array<int32_t, 2>{ 1, int(D) });
	auto acc = op.template get_destination_cooperative_tensor<decltype(weights), decltype(first), float>();
	for ( uint16_t i = 0; i < acc.get_capacity(); ++i )
		if ( acc.is_valid_element(i) )
			acc[i] = 0.0f;
	for ( uint t = 0; t < TILES; t++ )
	{
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( uint i = tid; i < ROWS * TILE; i += 128u )
			block[i] = p[(i / TILE) * (TILE * TILES) + t * TILE + i % TILE];
		threadgroup_barrier(mem_flags::mem_threadgroup);
		tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> values(v + (ulong)t * TILE * D, dextents<int32_t, 2>(D, TILE), array<int32_t, 2>{ 1, int(D) });
		op.run(weights, values, acc);
	}
	for ( uint16_t i = 0; i < acc.get_capacity(); ++i )
		if ( acc.is_valid_element(i) )
		{
			const auto index = acc.get_multidimensional_index(i);
			out[index[1] * D + index[0]] = acc[i];
		}
}
