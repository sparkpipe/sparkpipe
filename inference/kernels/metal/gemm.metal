#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "inference/kernels/metal/common.metal"

using namespace mpp::tensor_ops;

#define SPARK_GEMM_BM 128
#define SPARK_GEMM_BN 64
#define SPARK_GEMM_FP8_BM 64
#define SPARK_GEMM_MXFP8_BK 64
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
	operation.run(tile_left, tile_right, product);
	#pragma clang loop unroll(full)
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

kernel void spark_gemm_fp8_blocks(constant SparkGemvParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]], device bfloat *w [[buffer(2)]],
	device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]])
{
	threadgroup float tile_scale[2][SPARK_GEMM_BN];
	const int M = int(p.rows), K = int(p.input_dimension), N = int(p.output_dimension);
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> left(x, dextents<int32_t, 2>(K, M), array<int32_t, 2>{ 1, int(p.input_stride) });
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> right(w, dextents<int32_t, 2>(K, N), array<int32_t, 2>{ 1, K });
	constexpr auto descriptor = matmul2d_descriptor(SPARK_GEMM_FP8_BM, SPARK_GEMM_BN, int(SPARK_FP8_SCALE_K), false, true, false);
	matmul2d<descriptor, execution_simdgroups<SPARK_GEMM_SIMDS>> operation;
	const int m0 = int(group.y) * SPARK_GEMM_FP8_BM, n0 = int(group.x) * SPARK_GEMM_BN;
	auto origin_left = left.slice(0, m0);
	auto origin_right = right.slice(0, n0);
	auto total = operation.template get_destination_cooperative_tensor<decltype(origin_left), decltype(origin_right), float>();
	auto partial = operation.template get_destination_cooperative_tensor<decltype(origin_left), decltype(origin_right), float>();
	for ( uint16_t i = 0; i < total.get_capacity(); ++i )
		if ( total.is_valid_element(i) )
			total[i] = 0.0f;
	uint buf = 0u;
	for ( int k0 = 0; k0 < K; k0 += int(SPARK_FP8_SCALE_K), buf ^= 1u )
	{
		if ( tid < uint(SPARK_GEMM_BN) )
		{
			const int n = n0 + int(tid);
			tile_scale[buf][tid] = n < N ? SparkFp8Block<false>(p, scale, ulong(n), uint(k0)) : 0.0f;
		}
		auto tile_left = left.slice(k0, m0);
		auto tile_right = right.slice(k0, n0);
		operation.run(tile_left, tile_right, partial);
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( uint16_t i = 0; i < partial.get_capacity(); ++i )
			if ( partial.is_valid_element(i) )
				total[i] = fma(partial[i], tile_scale[buf][partial.get_multidimensional_index(i)[0]], total[i]);
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

kernel void spark_gemm_mxfp8_blocks(constant SparkGemvParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]], device const ushort *w [[buffer(2)]],
	device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
	uint tid [[thread_index_in_threadgroup]])
{
	constexpr uint per_row = SPARK_GEMM_MXFP8_BK / 8u;
	threadgroup uint4 tile[2][SPARK_GEMM_BN * per_row];
	const int M = int(p.rows), K = int(p.input_dimension), N = int(p.output_dimension);
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> left(x, dextents<int32_t, 2>(K, M), array<int32_t, 2>{ 1, int(p.input_stride) });
	tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> right0((threadgroup bfloat *)tile[0],
		dextents<int32_t, 2>(SPARK_GEMM_MXFP8_BK, SPARK_GEMM_BN), array<int32_t, 2>{ 1, SPARK_GEMM_MXFP8_BK });
	tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> right1((threadgroup bfloat *)tile[1],
		dextents<int32_t, 2>(SPARK_GEMM_MXFP8_BK, SPARK_GEMM_BN), array<int32_t, 2>{ 1, SPARK_GEMM_MXFP8_BK });
	constexpr auto descriptor = matmul2d_descriptor(SPARK_GEMM_BM, SPARK_GEMM_BN, SPARK_GEMM_MXFP8_BK, false, true, false,
		matmul2d_descriptor::mode::multiply_accumulate);
	matmul2d<descriptor, execution_simdgroups<SPARK_GEMM_SIMDS>> operation;
	const int m0 = int(group.y) * SPARK_GEMM_BM, n0 = int(group.x) * SPARK_GEMM_BN;
	auto origin_left = left.slice(0, m0);
	auto total = operation.template get_destination_cooperative_tensor<decltype(origin_left), decltype(right0), float>();
	#pragma clang loop unroll(full)
	for ( uint16_t i = 0; i < total.get_capacity(); ++i )
		if ( total.is_valid_element(i) )
			total[i] = 0.0f;
	const uint rn = tid >> 1u, half_k = tid & 1u;
	const uint rk_base = half_k * SPARK_MXFP8_GROUP, elem_base = rn * per_row + half_k * 4u;
	const int n = n0 + int(rn);
	uint buf = 0u;
	for ( int k0 = 0; k0 < K; k0 += SPARK_GEMM_MXFP8_BK, buf ^= 1u )
	{
		const int k = k0 + int(rk_base);
		if ( n < N && k < K )
		{
			const float sc = SparkMathUe8m0ToFloat(scale[(ulong)n * p.scale_stride + uint(k) / SPARK_MXFP8_GROUP]);
			device const uint4 *wp = (device const uint4 *)(w + (ulong)n * uint(K) + uint(k));
			for ( uint c = 0u; c < 4u; c++ )
			{
				float4 lo, hi;
				SparkUnpack8(wp[c], lo, hi);
				tile[buf][elem_base + c] = SparkPack8(lo * sc, hi * sc);
			}
		}
		else
			for ( uint c = 0u; c < 4u; c++ )
				tile[buf][elem_base + c] = uint4(0u);
		threadgroup_barrier(mem_flags::mem_threadgroup);
		auto tile_left = left.slice(k0, m0);
		if ( buf == 0u )
			operation.run(tile_left, right0, total);
		else
			operation.run(tile_left, right1, total);
	}
	#pragma clang loop unroll(full)
	for ( uint16_t i = 0; i < total.get_capacity(); ++i )
		if ( total.is_valid_element(i) )
		{
			const auto index = total.get_multidimensional_index(i);
			const int n_out = n0 + index[0], m_out = m0 + index[1];
			if ( m_out < M && n_out < N )
				out[(ulong)m_out * p.output_stride + n_out] = SparkMathFloatToBf16(total[i]);
		}
}

template <int ROWS>
kernel void spark_gemm_rows(constant SparkGemvParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]], device bfloat *w [[buffer(2)]],
	device void *out [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]])
{
	const int M = int(p.rows), K = int(p.input_dimension), N = int(p.output_dimension);
	const int span = int(p.split_span), k0 = int(group.z) * span, k_len = min(K, k0 + span) - k0;
	const int m0 = int(group.x) * ROWS, n0 = int(group.y) * int(SPARK_GEMM_ROWS_OUTPUTS);
	if ( k_len <= 0 )
		return;
	constexpr auto descriptor = matmul2d_descriptor(SPARK_GEMM_ROWS_OUTPUTS, ROWS, static_cast<int>(dynamic_extent), false, true, false);
	matmul2d<descriptor, execution_simdgroups<SPARK_GEMM_ROWS_THREADS / 32u>> operation;
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> rows(x + k0, dextents<int32_t, 2>(k_len, M), array<int32_t, 2>{ 1, int(p.input_stride) });
	tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> weights(w + k0, dextents<int32_t, 2>(k_len, N), array<int32_t, 2>{ 1, K });
	auto slice_w = weights.slice(0, n0);
	auto slice_x = rows.slice(0, m0);
	auto product = operation.template get_destination_cooperative_tensor<decltype(slice_w), decltype(slice_x), float>();
	operation.run(slice_w, slice_x, product);
	#pragma clang loop unroll(full)
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
	device bfloat *w [[buffer(2)]], device void *out [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]]);
template [[host_name("spark_gemm_rows32")]] kernel void spark_gemm_rows<32>(constant SparkGemvParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]],
	device bfloat *w [[buffer(2)]], device void *out [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]]);
template [[host_name("spark_gemm_rows64")]] kernel void spark_gemm_rows<64>(constant SparkGemvParams &p [[buffer(0)]], device bfloat *x [[buffer(1)]],
	device bfloat *w [[buffer(2)]], device void *out [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]]);

kernel void spark_gemm_reduce(constant SparkGemvParams &p [[buffer(0)]], device const float *partial [[buffer(1)]], device void *out [[buffer(2)]],
	uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]])
{
	const uint n0 = (group.x * group_size.x + tid) * 4u, m = group.y;
	if ( n0 >= p.output_dimension )
		return;
	const ulong row_stride = (ulong)p.rows * p.output_dimension;
	const ulong base = (ulong)m * p.output_dimension + n0;
	if ( n0 + 4u <= p.output_dimension && (p.output_dimension & 3u) == 0u )
	{
		float4 total = float4(0.0f);
		for ( uint split = 0u; split < p.splits; split++ )
			total += *(device const float4 *)(partial + (ulong)split * row_stride + base);
		const ulong at = (ulong)m * p.output_stride + n0;
		if ( p.output_f32 != 0u )
		{
			device float *dst = (device float *)out + at;
			if ( ((ulong)dst & 15ul) == 0ul )
				*(device float4 *)dst = total;
			else
			{
				dst[0] = total.x;
				dst[1] = total.y;
				dst[2] = total.z;
				dst[3] = total.w;
			}
		}
		else
		{
			device ushort *dst = (device ushort *)out + at;
			if ( ((ulong)dst & 7ul) == 0ul )
				*(device uint2 *)dst = SparkPack4(total);
			else
			{
				dst[0] = SparkMathFloatToBf16(total.x);
				dst[1] = SparkMathFloatToBf16(total.y);
				dst[2] = SparkMathFloatToBf16(total.z);
				dst[3] = SparkMathFloatToBf16(total.w);
			}
		}
		return;
	}
	for ( uint k = 0u; k < 4u && n0 + k < p.output_dimension; k++ )
	{
		float total = 0.0f;
		for ( uint split = 0u; split < p.splits; split++ )
			total += partial[(ulong)split * row_stride + base + k];
		const ulong at = (ulong)m * p.output_stride + n0 + k;
		if ( p.output_f32 != 0u )
			((device float *)out)[at] = total;
		else
			((device ushort *)out)[at] = SparkMathFloatToBf16(total);
	}
}
