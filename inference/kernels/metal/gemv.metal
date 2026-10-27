#include "inference/kernels/metal/common.metal"

#define SPARK_GEMV_SIMDS 4u
#define SPARK_GEMV_SPAN (SPARK_GEMV_SIMDS * 256u)

static inline void SparkUnpack8(uint4 bits, thread float4 &low, thread float4 &high)
{
	low = float4(as_type<float>(bits.x << 16), as_type<float>(bits.x & 0xffff0000u), as_type<float>(bits.y << 16), as_type<float>(bits.y & 0xffff0000u));
	high = float4(as_type<float>(bits.z << 16), as_type<float>(bits.z & 0xffff0000u), as_type<float>(bits.w << 16), as_type<float>(bits.w & 0xffff0000u));
}

template <uint R, uint O, bool F32>
static inline void SparkGemvRows(constant SparkGemvParams &p, device const ushort *x, device const ushort *w, device void *out, uint first,
	uint lane, uint simd, threadgroup float *partial)
{
	const uint K = p.input_dimension, N = p.output_dimension;
	float acc[O][R];
	device const ushort *rows[O];
	for ( uint i = 0u; i < O; i++ )
	{
		rows[i] = w + (ulong)min(first + i, N - 1u) * K;
		for ( uint r = 0u; r < R; r++ )
			acc[i][r] = 0.0f;
	}
	for ( uint k = (simd * 32u + lane) * 8u; k < K; k += SPARK_GEMV_SPAN )
	{
		float4 xl[R], xh[R];
		for ( uint r = 0u; r < R; r++ )
			SparkUnpack8(*(device const uint4 *)(x + (ulong)r * p.input_stride + k), xl[r], xh[r]);
		for ( uint i = 0u; i < O; i++ )
		{
			float4 wl, wh;
			SparkUnpack8(*(device const uint4 *)(rows[i] + k), wl, wh);
			for ( uint r = 0u; r < R; r++ )
				acc[i][r] += dot(wl, xl[r]) + dot(wh, xh[r]);
		}
	}
	for ( uint i = 0u; i < O; i++ )
		for ( uint r = 0u; r < R; r++ )
		{
			const float total = simd_sum(acc[i][r]);
			if ( lane == 0u )
				partial[(simd * O + i) * R + r] = total;
		}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( simd != 0u || lane >= O * R )
		return;
	const uint i = lane / R, r = lane % R;
	float total = partial[i * R + r];
	for ( uint s = 1u; s < SPARK_GEMV_SIMDS; s++ )
		total += partial[(s * O + i) * R + r];
	if ( first + i < N )
	{
		const ulong at = (ulong)r * p.output_stride + first + i;
		if ( F32 )
			((device float *)out)[at] = total;
		else
			((device ushort *)out)[at] = SparkMathFloatToBf16(total);
	}
}

template <uint R, uint O, bool F32>
kernel void spark_gemv(constant SparkGemvParams &p [[buffer(0)]], device const ushort *x [[buffer(1)]], device const ushort *w [[buffer(2)]],
	device void *out [[buffer(3)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup float partial[SPARK_GEMV_SIMDS * O * R];
	SparkGemvRows<R, O, F32>(p, x, w, out, group * O, lane, simd, partial);
}

kernel void spark_gemv_routed(constant SparkGemvParams &p [[buffer(0)]], device const ushort *x [[buffer(1)]], device const ushort *w [[buffer(2)]],
	device ushort *out [[buffer(3)]], device const uint *experts [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup float partial[SPARK_GEMV_SIMDS * SPARK_GEMV_OUTPUTS];
	const uint route = group.y;
	device const ushort *row = x + (ulong)(route / p.route_divisor) * p.input_stride;
	device const ushort *weights = w + (ulong)experts[route] * p.output_dimension * p.input_dimension;
	SparkGemvRows<1u, SPARK_GEMV_OUTPUTS, false>(p, row, weights, out + (ulong)route * p.output_stride, group.x * SPARK_GEMV_OUTPUTS, lane, simd,
		partial);
}

#define SPARK_GEMV_INSTANCE(rows, f32, name) \
	template [[host_name(name)]] kernel void spark_gemv<rows, SPARK_GEMV_ROW_OUTPUTS(rows), f32>(constant SparkGemvParams &p [[buffer(0)]], \
		device const ushort *x [[buffer(1)]], device const ushort *w [[buffer(2)]], device void *out [[buffer(3)]], \
		uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]);

SPARK_GEMV_INSTANCE(1u, false, "spark_gemv_r1")
SPARK_GEMV_INSTANCE(2u, false, "spark_gemv_r2")
SPARK_GEMV_INSTANCE(4u, false, "spark_gemv_r4")
SPARK_GEMV_INSTANCE(8u, false, "spark_gemv_r8")
SPARK_GEMV_INSTANCE(1u, true, "spark_gemv_f32_r1")
SPARK_GEMV_INSTANCE(2u, true, "spark_gemv_f32_r2")
SPARK_GEMV_INSTANCE(4u, true, "spark_gemv_f32_r4")
SPARK_GEMV_INSTANCE(8u, true, "spark_gemv_f32_r8")
