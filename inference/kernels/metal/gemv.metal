#include "inference/kernels/metal/common.metal"

#define SPARK_GEMV_SIMDS 4u
#define SPARK_GEMV_OUTPUTS 4u
#define SPARK_GEMV_SPAN 256u

static inline void SparkUnpack8(uint4 bits, thread float4 &low, thread float4 &high)
{
	low = float4(as_type<float>(bits.x << 16), as_type<float>(bits.x & 0xffff0000u), as_type<float>(bits.y << 16), as_type<float>(bits.y & 0xffff0000u));
	high = float4(as_type<float>(bits.z << 16), as_type<float>(bits.z & 0xffff0000u), as_type<float>(bits.w << 16), as_type<float>(bits.w & 0xffff0000u));
}

static inline float4 SparkGated4(float4 gate, float4 up)
{
	return float4(SparkRoundBf16(SparkSilu(gate.x) * up.x), SparkRoundBf16(SparkSilu(gate.y) * up.y),
		SparkRoundBf16(SparkSilu(gate.z) * up.z), SparkRoundBf16(SparkSilu(gate.w) * up.w));
}

template <uint R, bool F32, bool GATED>
kernel void spark_gemv(constant SparkGemvParams &p [[buffer(0)]], device const ushort *x [[buffer(1)]], device const ushort *w [[buffer(2)]],
	device void *out [[buffer(3)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]])
{
	const uint K = p.input_dimension, N = p.output_dimension;
	const uint first = (group * SPARK_GEMV_SIMDS + simd) * SPARK_GEMV_OUTPUTS;
	float acc[SPARK_GEMV_OUTPUTS][R];
	device const ushort *rows[SPARK_GEMV_OUTPUTS];
	for ( uint i = 0u; i < SPARK_GEMV_OUTPUTS; i++ )
	{
		rows[i] = w + (ulong)min(first + i, N - 1u) * K;
		for ( uint r = 0u; r < R; r++ )
			acc[i][r] = 0.0f;
	}
	for ( uint k = lane * 8u; k < K; k += SPARK_GEMV_SPAN )
	{
		float4 xl[R], xh[R];
		for ( uint r = 0u; r < R; r++ )
		{
			device const ushort *source = x + (ulong)r * p.input_stride + k;
			SparkUnpack8(*(device const uint4 *)source, xl[r], xh[r]);
			if ( GATED )
			{
				float4 ul, uh;
				SparkUnpack8(*(device const uint4 *)(source + K), ul, uh);
				xl[r] = SparkGated4(xl[r], ul);
				xh[r] = SparkGated4(xh[r], uh);
			}
		}
		for ( uint i = 0u; i < SPARK_GEMV_OUTPUTS; i++ )
		{
			float4 wl, wh;
			SparkUnpack8(*(device const uint4 *)(rows[i] + k), wl, wh);
			for ( uint r = 0u; r < R; r++ )
				acc[i][r] += dot(wl, xl[r]) + dot(wh, xh[r]);
		}
	}
	for ( uint i = 0u; i < SPARK_GEMV_OUTPUTS; i++ )
		for ( uint r = 0u; r < R; r++ )
		{
			const float total = simd_sum(acc[i][r]);
			if ( lane == 0u && first + i < N )
			{
				const ulong at = (ulong)r * p.output_stride + first + i;
				if ( F32 )
					((device float *)out)[at] = total;
				else
					((device ushort *)out)[at] = SparkFloatToBf16(total);
			}
		}
}

#define SPARK_GEMV_INSTANCE(rows, f32, gated, name) \
	template [[host_name(name)]] kernel void spark_gemv<rows, f32, gated>(constant SparkGemvParams &p [[buffer(0)]], \
		device const ushort *x [[buffer(1)]], device const ushort *w [[buffer(2)]], device void *out [[buffer(3)]], \
		uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]);

SPARK_GEMV_INSTANCE(1u, false, false, "spark_gemv_r1")
SPARK_GEMV_INSTANCE(2u, false, false, "spark_gemv_r2")
SPARK_GEMV_INSTANCE(4u, false, false, "spark_gemv_r4")
SPARK_GEMV_INSTANCE(8u, false, false, "spark_gemv_r8")
SPARK_GEMV_INSTANCE(1u, true, false, "spark_gemv_f32_r1")
SPARK_GEMV_INSTANCE(2u, true, false, "spark_gemv_f32_r2")
SPARK_GEMV_INSTANCE(4u, true, false, "spark_gemv_f32_r4")
SPARK_GEMV_INSTANCE(8u, true, false, "spark_gemv_f32_r8")
SPARK_GEMV_INSTANCE(1u, false, true, "spark_gemv_gated_r1")
SPARK_GEMV_INSTANCE(2u, false, true, "spark_gemv_gated_r2")
SPARK_GEMV_INSTANCE(4u, false, true, "spark_gemv_gated_r4")
SPARK_GEMV_INSTANCE(8u, false, true, "spark_gemv_gated_r8")
