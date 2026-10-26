#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "inference/kernels/metal/common.metal"

using namespace mpp::tensor_ops;

#define SPARK_GEMM_BM 128
#define SPARK_GEMM_BN 64
#define SPARK_GEMM_SIMDS 4

template <bool F32>
kernel void spark_gemm(constant SparkGemmParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]], device bfloat *w [[buffer(2)]],
	device void *out [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]])
{
	const int M = int(p.rows), K = int(p.input_dimension), N = int(p.output_dimension);
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> left(x, dextents<int32_t, 2>(K, M), array<int32_t, 2>{ 1, int(p.input_stride) });
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> right(w, dextents<int32_t, 2>(K, N), array<int32_t, 2>{ 1, K });
	constexpr auto descriptor = matmul2d_descriptor(SPARK_GEMM_BM, SPARK_GEMM_BN, static_cast<int>(dynamic_extent), false, true, false);
	matmul2d<descriptor, execution_simdgroups<SPARK_GEMM_SIMDS>> operation;
	const int m0 = int(group.y) * SPARK_GEMM_BM, n0 = int(group.x) * SPARK_GEMM_BN;
	auto tile_left = left.slice(0, m0);
	auto tile_right = right.slice(0, n0);
	auto product = operation.template get_destination_cooperative_tensor<decltype(tile_left), decltype(tile_right), float>();
	for ( uint16_t i = 0; i < product.get_capacity(); ++i )
		if ( product.is_valid_element(i) )
			product[i] = 0.0f;
	operation.run(tile_left, tile_right, product);
	for ( uint16_t i = 0; i < product.get_capacity(); ++i )
		if ( product.is_valid_element(i) )
		{
			const auto index = product.get_multidimensional_index(i);
			const int n = n0 + index[0], m = m0 + index[1];
			if ( m < M && n < N )
			{
				const ulong at = (ulong)m * p.output_stride + n;
				if ( F32 )
					((device float *)out)[at] = product[i];
				else
					((device ushort *)out)[at] = SparkMathFloatToBf16(product[i]);
			}
		}
}

template [[host_name("spark_gemm")]] kernel void spark_gemm<false>(constant SparkGemmParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]],
	device bfloat *w [[buffer(2)]], device void *out [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]]);
template [[host_name("spark_gemm_f32")]] kernel void spark_gemm<true>(constant SparkGemmParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]],
	device bfloat *w [[buffer(2)]], device void *out [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]]);
