#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "inference/kernels/metal/common.metal"

using namespace mpp::tensor_ops;

#define SPARK_GEMM_BM 128
#define SPARK_GEMM_BN 64
#define SPARK_GEMM_SIMDS 4

template <bool F32>
kernel void spark_gemm(constant SparkGemvParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]], device bfloat *w [[buffer(2)]],
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

template [[host_name("spark_gemm")]] kernel void spark_gemm<false>(constant SparkGemvParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]],
	device bfloat *w [[buffer(2)]], device void *out [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]]);
template [[host_name("spark_gemm_f32")]] kernel void spark_gemm<true>(constant SparkGemvParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]],
	device bfloat *w [[buffer(2)]], device void *out [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]]);

template <uint BLOCK, bool MX>
kernel void spark_gemm_fp8_blocks(constant SparkGemvParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]], device bfloat *w [[buffer(2)]],
	device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]])
{
	const int M = int(p.rows), K = int(p.input_dimension), N = int(p.output_dimension);
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> left(x, dextents<int32_t, 2>(K, M), array<int32_t, 2>{ 1, int(p.input_stride) });
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> right(w, dextents<int32_t, 2>(K, N), array<int32_t, 2>{ 1, K });
	constexpr auto descriptor = matmul2d_descriptor(SPARK_GEMM_BM, SPARK_GEMM_BN, int(BLOCK), false, true, false);
	matmul2d<descriptor, execution_simdgroups<SPARK_GEMM_SIMDS>> operation;
	const int m0 = int(group.y) * SPARK_GEMM_BM, n0 = int(group.x) * SPARK_GEMM_BN;
	auto origin_left = left.slice(0, m0);
	auto origin_right = right.slice(0, n0);
	auto total = operation.template get_destination_cooperative_tensor<decltype(origin_left), decltype(origin_right), float>();
	auto partial = operation.template get_destination_cooperative_tensor<decltype(origin_left), decltype(origin_right), float>();
	for ( uint16_t i = 0; i < total.get_capacity(); ++i )
		if ( total.is_valid_element(i) )
			total[i] = 0.0f;
	for ( int k0 = 0; k0 < K; k0 += int(BLOCK) )
	{
		auto tile_left = left.slice(k0, m0);
		auto tile_right = right.slice(k0, n0);
		for ( uint16_t i = 0; i < partial.get_capacity(); ++i )
			if ( partial.is_valid_element(i) )
				partial[i] = 0.0f;
		operation.run(tile_left, tile_right, partial);
		for ( uint16_t i = 0; i < partial.get_capacity(); ++i )
			if ( partial.is_valid_element(i) )
			{
				const int n = n0 + partial.get_multidimensional_index(i)[0];
				if ( n < N )
					total[i] = fma(partial[i], SparkFp8Block<MX>(p, scale, ulong(n), uint(k0)), total[i]);
			}
	}
	for ( uint16_t i = 0; i < total.get_capacity(); ++i )
		if ( total.is_valid_element(i) )
		{
			const auto index = total.get_multidimensional_index(i);
			const int n = n0 + index[0], m = m0 + index[1];
			if ( m < M && n < N )
				out[(ulong)m * p.output_stride + n] = SparkMathFloatToBf16(total[i]);
		}
}

template [[host_name("spark_gemm_fp8_blocks")]] kernel void spark_gemm_fp8_blocks<SPARK_FP8_SCALE_K, false>(constant SparkGemvParams &p [[buffer(0)]],
	device bfloat *x [[buffer(1)]], device bfloat *w [[buffer(2)]], device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]]);
template [[host_name("spark_gemm_mxfp8_blocks")]] kernel void spark_gemm_fp8_blocks<SPARK_MXFP8_GROUP, true>(constant SparkGemvParams &p [[buffer(0)]],
	device bfloat *x [[buffer(1)]], device bfloat *w [[buffer(2)]], device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]]);

template <int ROWS>
kernel void spark_gemm_rows(constant SparkGemvParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]], device const uint4 *w [[buffer(2)]],
	device void *out [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]])
{
	constexpr uint per_output = SPARK_GEMM_ROWS_DEPTH / 8u, per_thread = SPARK_GEMM_ROWS_OUTPUTS * per_output / SPARK_GEMM_ROWS_THREADS;
	threadgroup uint4 tile[SPARK_GEMM_ROWS_OUTPUTS * per_output];
	const int M = int(p.rows), K = int(p.input_dimension), N = int(p.output_dimension);
	const int span = int(p.split_span), k0 = int(group.z) * span, k_end = min(K, k0 + span);
	const int m0 = int(group.x) * ROWS, n0 = int(group.y) * int(SPARK_GEMM_ROWS_OUTPUTS);
	constexpr auto descriptor = matmul2d_descriptor(SPARK_GEMM_ROWS_OUTPUTS, ROWS, SPARK_GEMM_ROWS_DEPTH, false, true, false,
		matmul2d_descriptor::mode::multiply_accumulate);
	matmul2d<descriptor, execution_simdgroups<SPARK_GEMM_ROWS_THREADS / 32u>> operation;
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> rows(x, dextents<int32_t, 2>(K, M), array<int32_t, 2>{ 1, int(p.input_stride) });
	tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> weights((threadgroup bfloat *)tile,
		dextents<int32_t, 2>(SPARK_GEMM_ROWS_DEPTH, SPARK_GEMM_ROWS_OUTPUTS), array<int32_t, 2>{ 1, int(SPARK_GEMM_ROWS_DEPTH) });
	auto first = rows.slice(k0, m0);
	auto product = operation.template get_destination_cooperative_tensor<decltype(weights), decltype(first), float>();
	uint4 next[per_thread];
	device const uint4 *source[per_thread];
	for ( uint16_t i = 0; i < product.get_capacity(); ++i )
		if ( product.is_valid_element(i) )
			product[i] = 0.0f;
	for ( uint j = 0u; j < per_thread; j++ )
	{
		const uint element = tid + j * SPARK_GEMM_ROWS_THREADS;
		source[j] = w + (ulong)min(n0 + int(element / per_output), N - 1) * (ulong)(K / 8) + element % per_output;
		next[j] = source[j][k0 / 8];
	}
	for ( int k = k0; k < k_end; k += int(SPARK_GEMM_ROWS_DEPTH) )
	{
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( uint j = 0u; j < per_thread; j++ )
			tile[tid + j * SPARK_GEMM_ROWS_THREADS] = next[j];
		threadgroup_barrier(mem_flags::mem_threadgroup);
		if ( k + int(SPARK_GEMM_ROWS_DEPTH) < k_end )
			for ( uint j = 0u; j < per_thread; j++ )
				next[j] = source[j][(k + int(SPARK_GEMM_ROWS_DEPTH)) / 8];
		auto slice = rows.slice(k, m0);
		operation.run(weights, slice, product);
	}
	for ( uint16_t i = 0; i < product.get_capacity(); ++i )
		if ( product.is_valid_element(i) )
		{
			const auto index = product.get_multidimensional_index(i);
			const int m = m0 + index[0], n = n0 + index[1];
			if ( m >= M || n >= N )
				continue;
			if ( p.splits > 1u )
				((device float *)out)[((ulong)group.z * p.rows + m) * p.output_dimension + n] = product[i];
			else if ( p.output_f32 != 0u )
				((device float *)out)[(ulong)m * p.output_stride + n] = product[i];
			else
				((device ushort *)out)[(ulong)m * p.output_stride + n] = SparkMathFloatToBf16(product[i]);
		}
}

template [[host_name("spark_gemm_rows16")]] kernel void spark_gemm_rows<16>(constant SparkGemvParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]],
	device const uint4 *w [[buffer(2)]], device void *out [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]]);
template [[host_name("spark_gemm_rows32")]] kernel void spark_gemm_rows<32>(constant SparkGemvParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]],
	device const uint4 *w [[buffer(2)]], device void *out [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]]);

kernel void spark_gemm_reduce(constant SparkGemvParams &p [[buffer(0)]], device const float *partial [[buffer(1)]], device void *out [[buffer(2)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint n = group.x * group_size.x + tid, m = group.y;
	if ( n >= p.output_dimension )
		return;
	float total = 0.0f;
	for ( uint split = 0u; split < p.splits; split++ )
		total += partial[((ulong)split * p.rows + m) * p.output_dimension + n];
	if ( p.output_f32 != 0u )
		((device float *)out)[(ulong)m * p.output_stride + n] = total;
	else
		((device ushort *)out)[(ulong)m * p.output_stride + n] = SparkMathFloatToBf16(total);
}
