#include "inference/kernels/metal/common.metal"

#define SPARK_GEMV_SIMDS 4u
#define SPARK_GEMV_SPAN (SPARK_GEMV_SIMDS * 256u)

constant uint kSparkE4m3Bits[256] = {
	0x00000000u, 0x3b000000u, 0x3b800000u, 0x3bc00000u, 0x3c000000u, 0x3c200000u, 0x3c400000u, 0x3c600000u,
	0x3c800000u, 0x3c900000u, 0x3ca00000u, 0x3cb00000u, 0x3cc00000u, 0x3cd00000u, 0x3ce00000u, 0x3cf00000u,
	0x3d000000u, 0x3d100000u, 0x3d200000u, 0x3d300000u, 0x3d400000u, 0x3d500000u, 0x3d600000u, 0x3d700000u,
	0x3d800000u, 0x3d900000u, 0x3da00000u, 0x3db00000u, 0x3dc00000u, 0x3dd00000u, 0x3de00000u, 0x3df00000u,
	0x3e000000u, 0x3e100000u, 0x3e200000u, 0x3e300000u, 0x3e400000u, 0x3e500000u, 0x3e600000u, 0x3e700000u,
	0x3e800000u, 0x3e900000u, 0x3ea00000u, 0x3eb00000u, 0x3ec00000u, 0x3ed00000u, 0x3ee00000u, 0x3ef00000u,
	0x3f000000u, 0x3f100000u, 0x3f200000u, 0x3f300000u, 0x3f400000u, 0x3f500000u, 0x3f600000u, 0x3f700000u,
	0x3f800000u, 0x3f900000u, 0x3fa00000u, 0x3fb00000u, 0x3fc00000u, 0x3fd00000u, 0x3fe00000u, 0x3ff00000u,
	0x40000000u, 0x40100000u, 0x40200000u, 0x40300000u, 0x40400000u, 0x40500000u, 0x40600000u, 0x40700000u,
	0x40800000u, 0x40900000u, 0x40a00000u, 0x40b00000u, 0x40c00000u, 0x40d00000u, 0x40e00000u, 0x40f00000u,
	0x41000000u, 0x41100000u, 0x41200000u, 0x41300000u, 0x41400000u, 0x41500000u, 0x41600000u, 0x41700000u,
	0x41800000u, 0x41900000u, 0x41a00000u, 0x41b00000u, 0x41c00000u, 0x41d00000u, 0x41e00000u, 0x41f00000u,
	0x42000000u, 0x42100000u, 0x42200000u, 0x42300000u, 0x42400000u, 0x42500000u, 0x42600000u, 0x42700000u,
	0x42800000u, 0x42900000u, 0x42a00000u, 0x42b00000u, 0x42c00000u, 0x42d00000u, 0x42e00000u, 0x42f00000u,
	0x43000000u, 0x43100000u, 0x43200000u, 0x43300000u, 0x43400000u, 0x43500000u, 0x43600000u, 0x43700000u,
	0x43800000u, 0x43900000u, 0x43a00000u, 0x43b00000u, 0x43c00000u, 0x43d00000u, 0x43e00000u, 0x7fc00000u,
	0x80000000u, 0xbb000000u, 0xbb800000u, 0xbbc00000u, 0xbc000000u, 0xbc200000u, 0xbc400000u, 0xbc600000u,
	0xbc800000u, 0xbc900000u, 0xbca00000u, 0xbcb00000u, 0xbcc00000u, 0xbcd00000u, 0xbce00000u, 0xbcf00000u,
	0xbd000000u, 0xbd100000u, 0xbd200000u, 0xbd300000u, 0xbd400000u, 0xbd500000u, 0xbd600000u, 0xbd700000u,
	0xbd800000u, 0xbd900000u, 0xbda00000u, 0xbdb00000u, 0xbdc00000u, 0xbdd00000u, 0xbde00000u, 0xbdf00000u,
	0xbe000000u, 0xbe100000u, 0xbe200000u, 0xbe300000u, 0xbe400000u, 0xbe500000u, 0xbe600000u, 0xbe700000u,
	0xbe800000u, 0xbe900000u, 0xbea00000u, 0xbeb00000u, 0xbec00000u, 0xbed00000u, 0xbee00000u, 0xbef00000u,
	0xbf000000u, 0xbf100000u, 0xbf200000u, 0xbf300000u, 0xbf400000u, 0xbf500000u, 0xbf600000u, 0xbf700000u,
	0xbf800000u, 0xbf900000u, 0xbfa00000u, 0xbfb00000u, 0xbfc00000u, 0xbfd00000u, 0xbfe00000u, 0xbff00000u,
	0xc0000000u, 0xc0100000u, 0xc0200000u, 0xc0300000u, 0xc0400000u, 0xc0500000u, 0xc0600000u, 0xc0700000u,
	0xc0800000u, 0xc0900000u, 0xc0a00000u, 0xc0b00000u, 0xc0c00000u, 0xc0d00000u, 0xc0e00000u, 0xc0f00000u,
	0xc1000000u, 0xc1100000u, 0xc1200000u, 0xc1300000u, 0xc1400000u, 0xc1500000u, 0xc1600000u, 0xc1700000u,
	0xc1800000u, 0xc1900000u, 0xc1a00000u, 0xc1b00000u, 0xc1c00000u, 0xc1d00000u, 0xc1e00000u, 0xc1f00000u,
	0xc2000000u, 0xc2100000u, 0xc2200000u, 0xc2300000u, 0xc2400000u, 0xc2500000u, 0xc2600000u, 0xc2700000u,
	0xc2800000u, 0xc2900000u, 0xc2a00000u, 0xc2b00000u, 0xc2c00000u, 0xc2d00000u, 0xc2e00000u, 0xc2f00000u,
	0xc3000000u, 0xc3100000u, 0xc3200000u, 0xc3300000u, 0xc3400000u, 0xc3500000u, 0xc3600000u, 0xc3700000u,
	0xc3800000u, 0xc3900000u, 0xc3a00000u, 0xc3b00000u, 0xc3c00000u, 0xc3d00000u, 0xc3e00000u, 0xffc00000u
};

constant float kSparkE2m1[16] = {
	0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
	-0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

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

static inline float4 SparkFp8Word(uint word)
{
	return float4(as_type<float>(kSparkE4m3Bits[word & 0xffu]), as_type<float>(kSparkE4m3Bits[(word >> 8u) & 0xffu]),
		as_type<float>(kSparkE4m3Bits[(word >> 16u) & 0xffu]), as_type<float>(kSparkE4m3Bits[word >> 24u]));
}

template <uint R, uint O, bool MX>
static inline void SparkFp8Rows(constant SparkGemvParams &p, device const ushort *x, device const uchar *w, device const uchar *scale,
	device ushort *out, uint first, uint lane, uint simd, threadgroup float *partial)
{
	const uint K = p.input_dimension, N = p.output_dimension;
	float acc[O][R];
	ulong rows[O];
	for ( uint i = 0u; i < O; i++ )
	{
		rows[i] = min(first + i, N - 1u);
		for ( uint r = 0u; r < R; r++ )
			acc[i][r] = 0.0f;
	}
	for ( uint k = (simd * 32u + lane) * SPARK_FP8_CHUNK; k < K; k += SPARK_GEMV_SIMDS * 32u * SPARK_FP8_CHUNK )
	{
		if ( R <= 2u )
		{
			float4 x0[R], x1[R], x2[R], x3[R];
			for ( uint r = 0u; r < R; r++ )
			{
				device const ushort *xr = x + (ulong)r * p.input_stride + k;
				SparkUnpack8(*(device const uint4 *)xr, x0[r], x1[r]);
				SparkUnpack8(*(device const uint4 *)(xr + 8u), x2[r], x3[r]);
			}
			for ( uint i = 0u; i < O; i++ )
			{
				const float block = SparkFp8Block<MX>(p, scale, rows[i], k);
				const uint4 bits = *(device const uint4 *)(w + rows[i] * K + k);
				const float4 w0 = SparkFp8Word(bits.x), w1 = SparkFp8Word(bits.y), w2 = SparkFp8Word(bits.z), w3 = SparkFp8Word(bits.w);
				for ( uint r = 0u; r < R; r++ )
					acc[i][r] = fma(block, (dot(w0, x0[r]) + dot(w1, x1[r])) + (dot(w2, x2[r]) + dot(w3, x3[r])), acc[i][r]);
			}
		}
		else
		{
			float4 w0[O], w1[O], w2[O], w3[O];
			float block[O];
			for ( uint i = 0u; i < O; i++ )
			{
				block[i] = SparkFp8Block<MX>(p, scale, rows[i], k);
				const uint4 bits = *(device const uint4 *)(w + rows[i] * K + k);
				w0[i] = SparkFp8Word(bits.x);
				w1[i] = SparkFp8Word(bits.y);
				w2[i] = SparkFp8Word(bits.z);
				w3[i] = SparkFp8Word(bits.w);
			}
			for ( uint r = 0u; r < R; r++ )
			{
				float4 x0, x1, x2, x3;
				device const ushort *xr = x + (ulong)r * p.input_stride + k;
				SparkUnpack8(*(device const uint4 *)xr, x0, x1);
				SparkUnpack8(*(device const uint4 *)(xr + 8u), x2, x3);
				for ( uint i = 0u; i < O; i++ )
					acc[i][r] = fma(block[i], (dot(w0[i], x0) + dot(w1[i], x1)) + (dot(w2[i], x2) + dot(w3[i], x3)), acc[i][r]);
			}
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
		out[(ulong)r * p.output_stride + first + i] = SparkMathFloatToBf16(total);
}

template <uint R, uint O, bool MX>
kernel void spark_gemv_fp8(constant SparkGemvParams &p [[buffer(0)]], device const ushort *x [[buffer(1)]], device const uchar *w [[buffer(2)]],
	device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup float partial[SPARK_GEMV_SIMDS * O * R];
	SparkFp8Rows<R, O, MX>(p, x, w, scale, out, group * O, lane, simd, partial);
}

template <bool MX>
kernel void spark_gemv_fp8_routed(constant SparkGemvParams &p [[buffer(0)]], device const ushort *x [[buffer(1)]], device const uchar *w [[buffer(2)]],
	device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]], device const uint *experts [[buffer(5)]],
	uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup float partial[SPARK_GEMV_SIMDS * SPARK_GEMV_OUTPUTS];
	const uint K = p.input_dimension, N = p.output_dimension, route = group.y, first = group.x * SPARK_GEMV_OUTPUTS;
	if ( experts[route] == SPARK_ROUTE_SKIP )
	{
		if ( simd == 0u && lane < SPARK_GEMV_OUTPUTS && first + lane < N )
			out[(ulong)route * p.output_stride + first + lane] = 0u;
		return;
	}
	const ulong expert = experts[route];
	SparkFp8Rows<1u, SPARK_GEMV_OUTPUTS, MX>(p, x + (ulong)(route / p.route_divisor) * p.input_stride, w + expert * N * K,
		scale + expert * N * p.scale_stride * (MX ? 1u : 4u), out + (ulong)route * p.output_stride, first, lane, simd, partial);
}

template [[host_name("spark_gemv_fp8_routed")]] kernel void spark_gemv_fp8_routed<false>(constant SparkGemvParams &p [[buffer(0)]],
	device const ushort *x [[buffer(1)]], device const uchar *w [[buffer(2)]], device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]],
	device const uint *experts [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]]);
template [[host_name("spark_gemv_mxfp8_routed")]] kernel void spark_gemv_fp8_routed<true>(constant SparkGemvParams &p [[buffer(0)]],
	device const ushort *x [[buffer(1)]], device const uchar *w [[buffer(2)]], device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]],
	device const uint *experts [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]]);

template <bool MX>
kernel void spark_dequant_fp8(constant SparkGemvParams &p [[buffer(0)]], device const uchar *w [[buffer(1)]], device const uchar *scale [[buffer(2)]],
	device ushort *out [[buffer(3)]], uint2 index [[thread_position_in_grid]])
{
	const uint k = index.x * 4u, row = index.y;
	if ( row >= p.output_dimension || k >= p.input_dimension )
		return;
	(void)scale;
	const uint bits = *(device const uint *)(w + (ulong)row * p.input_dimension + k);
	device ushort *y = out + (ulong)row * p.input_dimension + k;
	const float4 vals = SparkFp8Word(bits);
	if ( ((ulong)y & 7ul) == 0ul )
		*(device uint2 *)y = SparkPack4(vals);
	else
	{
		y[0] = SparkMathFloatToBf16(vals.x);
		y[1] = SparkMathFloatToBf16(vals.y);
		y[2] = SparkMathFloatToBf16(vals.z);
		y[3] = SparkMathFloatToBf16(vals.w);
	}
}

template [[host_name("spark_dequant_fp8")]] kernel void spark_dequant_fp8<false>(constant SparkGemvParams &p [[buffer(0)]],
	device const uchar *w [[buffer(1)]], device const uchar *scale [[buffer(2)]], device ushort *out [[buffer(3)]], uint2 index [[thread_position_in_grid]]);
template [[host_name("spark_dequant_mxfp8")]] kernel void spark_dequant_fp8<true>(constant SparkGemvParams &p [[buffer(0)]],
	device const uchar *w [[buffer(1)]], device const uchar *scale [[buffer(2)]], device ushort *out [[buffer(3)]], uint2 index [[thread_position_in_grid]]);

static inline float SparkE2m1Dot8(uint word, float4 x0, float4 x1)
{
	const float4 w0 = float4(kSparkE2m1[word & 15u], kSparkE2m1[(word >> 4u) & 15u], kSparkE2m1[(word >> 8u) & 15u], kSparkE2m1[(word >> 12u) & 15u]);
	const float4 w1 = float4(kSparkE2m1[(word >> 16u) & 15u], kSparkE2m1[(word >> 20u) & 15u], kSparkE2m1[(word >> 24u) & 15u], kSparkE2m1[word >> 28u]);
	return dot(w0, x0) + dot(w1, x1);
}

static inline float SparkE2m1Fma8(uint word, float4 x0, float4 x1, float acc)
{
	acc = fma(kSparkE2m1[word & 15u], x0.x, acc);
	acc = fma(kSparkE2m1[(word >> 4u) & 15u], x0.y, acc);
	acc = fma(kSparkE2m1[(word >> 8u) & 15u], x0.z, acc);
	acc = fma(kSparkE2m1[(word >> 12u) & 15u], x0.w, acc);
	acc = fma(kSparkE2m1[(word >> 16u) & 15u], x1.x, acc);
	acc = fma(kSparkE2m1[(word >> 20u) & 15u], x1.y, acc);
	acc = fma(kSparkE2m1[(word >> 24u) & 15u], x1.z, acc);
	return fma(kSparkE2m1[word >> 28u], x1.w, acc);
}

kernel void spark_gemv_mxfp4_routed(constant SparkGemvParams &p [[buffer(0)]], device const ushort *x [[buffer(1)]], device const uchar *w [[buffer(2)]],
	device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]], device const uint *experts [[buffer(5)]],
	uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup float partial[SPARK_GEMV_SIMDS * SPARK_GEMV_OUTPUTS];
	const uint K = p.input_dimension, N = p.output_dimension, route = group.y, first = group.x * SPARK_GEMV_OUTPUTS;
	if ( experts[route] == SPARK_ROUTE_SKIP )
	{
		if ( simd == 0u && lane < SPARK_GEMV_OUTPUTS && first + lane < N )
			out[(ulong)route * p.output_stride + first + lane] = 0u;
		return;
	}
	const ulong expert = experts[route];
	device const ushort *row = x + (ulong)(route / p.route_divisor) * p.input_stride;
	float acc[SPARK_GEMV_OUTPUTS];
	device const uchar *weights[SPARK_GEMV_OUTPUTS];
	device const uchar *scales[SPARK_GEMV_OUTPUTS];
	for ( uint i = 0u; i < SPARK_GEMV_OUTPUTS; i++ )
	{
		const ulong neuron = expert * N + min(first + i, N - 1u);
		weights[i] = w + neuron * (K / 2u);
		scales[i] = scale + neuron * (K / SPARK_MXFP4_GROUP);
		acc[i] = 0.0f;
	}
	for ( uint k = (simd * 32u + lane) * SPARK_MXFP4_GROUP; k < K; k += SPARK_GEMV_SIMDS * 32u * SPARK_MXFP4_GROUP )
	{
		float4 xs[8];
		for ( uint part = 0u; part < 4u; part++ )
			SparkUnpack8(*(device const uint4 *)(row + k + 8u * part), xs[2u * part], xs[2u * part + 1u]);
		for ( uint i = 0u; i < SPARK_GEMV_OUTPUTS; i++ )
		{
			const uint4 bits = *(device const uint4 *)(weights[i] + k / 2u);
			const float sum = (SparkE2m1Dot8(bits.x, xs[0], xs[1]) + SparkE2m1Dot8(bits.y, xs[2], xs[3])) +
				(SparkE2m1Dot8(bits.z, xs[4], xs[5]) + SparkE2m1Dot8(bits.w, xs[6], xs[7]));
			acc[i] = fma(SparkMathUe8m0ToFloat(scales[i][k / SPARK_MXFP4_GROUP]), sum, acc[i]);
		}
	}
	for ( uint i = 0u; i < SPARK_GEMV_OUTPUTS; i++ )
	{
		const float total = simd_sum(acc[i]);
		if ( lane == 0u )
			partial[simd * SPARK_GEMV_OUTPUTS + i] = total;
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if ( simd != 0u || lane >= SPARK_GEMV_OUTPUTS )
		return;
	float total = partial[lane];
	for ( uint s = 1u; s < SPARK_GEMV_SIMDS; s++ )
		total += partial[s * SPARK_GEMV_OUTPUTS + lane];
	if ( first + lane < N )
		out[(ulong)route * p.output_stride + first + lane] = SparkMathFloatToBf16(total);
}

template <bool F32>
kernel void spark_gemv_routed(constant SparkGemvParams &p [[buffer(0)]], device const ushort *x [[buffer(1)]], device const ushort *w [[buffer(2)]],
	device void *out [[buffer(3)]], device const uint *experts [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup float partial[SPARK_GEMV_SIMDS * SPARK_GEMV_OUTPUTS];
	const uint route = group.y;
	const ulong at = (ulong)route * p.output_stride;
	if ( experts[route] == SPARK_ROUTE_SKIP )
	{
		if ( simd == 0u && lane < SPARK_GEMV_OUTPUTS && group.x * SPARK_GEMV_OUTPUTS + lane < p.output_dimension )
		{
			if ( F32 )
				((device float *)out)[at + group.x * SPARK_GEMV_OUTPUTS + lane] = 0.0f;
			else
				((device ushort *)out)[at + group.x * SPARK_GEMV_OUTPUTS + lane] = 0u;
		}
		return;
	}
	device const ushort *row = x + (ulong)(route / p.route_divisor) * p.input_stride;
	device const ushort *weights = w + (ulong)experts[route] * p.output_dimension * p.input_dimension;
	device void *target = F32 ? (device void *)((device float *)out + at) : (device void *)((device ushort *)out + at);
	SparkGemvRows<1u, SPARK_GEMV_OUTPUTS, F32>(p, row, weights, target, group.x * SPARK_GEMV_OUTPUTS, lane, simd, partial);
}

template [[host_name("spark_gemv_routed")]] kernel void spark_gemv_routed<false>(constant SparkGemvParams &p [[buffer(0)]],
	device const ushort *x [[buffer(1)]], device const ushort *w [[buffer(2)]], device void *out [[buffer(3)]], device const uint *experts [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]);
template [[host_name("spark_gemv_routed_f32")]] kernel void spark_gemv_routed<true>(constant SparkGemvParams &p [[buffer(0)]],
	device const ushort *x [[buffer(1)]], device const ushort *w [[buffer(2)]], device void *out [[buffer(3)]], device const uint *experts [[buffer(4)]],
	uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]);

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

#define SPARK_GEMV_FP8_INSTANCE(rows, mx, name) \
	template [[host_name(name)]] kernel void spark_gemv_fp8<rows, SPARK_GEMV_ROW_OUTPUTS(rows), mx>(constant SparkGemvParams &p [[buffer(0)]], \
		device const ushort *x [[buffer(1)]], device const uchar *w [[buffer(2)]], device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]], \
		uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]);

SPARK_GEMV_FP8_INSTANCE(1u, false, "spark_gemv_fp8_r1")
SPARK_GEMV_FP8_INSTANCE(2u, false, "spark_gemv_fp8_r2")
SPARK_GEMV_FP8_INSTANCE(4u, false, "spark_gemv_fp8_r4")
SPARK_GEMV_FP8_INSTANCE(8u, false, "spark_gemv_fp8_r8")
SPARK_GEMV_FP8_INSTANCE(1u, true, "spark_gemv_mxfp8_r1")
SPARK_GEMV_FP8_INSTANCE(2u, true, "spark_gemv_mxfp8_r2")
SPARK_GEMV_FP8_INSTANCE(4u, true, "spark_gemv_mxfp8_r4")
SPARK_GEMV_FP8_INSTANCE(8u, true, "spark_gemv_mxfp8_r8")

kernel void spark_gemv_nvfp4_routed(constant SparkGemvParams &p [[buffer(0)]], device const ushort *x [[buffer(1)]], device const uchar *w [[buffer(2)]],
	device const uchar *scale [[buffer(3)]], device const float *global_scale [[buffer(4)]], device ushort *out [[buffer(5)]],
	device const uint *experts [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
	uint simd [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]])
{
	const uint K = p.input_dimension, N = p.output_dimension, route = group.y, chunks = K / SPARK_NVFP4_CHUNK;
	const uint lanes = SPARK_NVFP4_LANES(chunks), sub = lane % lanes;
	const uint neuron = (group.x * simds + simd) * (32u / lanes) + lane / lanes;
	if ( experts[route] == SPARK_ROUTE_SKIP )
	{
		if ( sub == 0u && neuron < N )
			out[(ulong)route * p.output_stride + neuron] = 0u;
		return;
	}
	const ulong expert = experts[route], at = expert * N + min(neuron, N - 1u);
	device const ushort *row = x + (ulong)(route / p.route_divisor) * p.input_stride;
	device const uchar *weights = w + at * (K / 2u);
	device const uchar *scales = scale + at * (K / SPARK_NVFP4_GROUP);
	const float global = global_scale[expert];
	float acc = 0.0f;
	for ( uint c = sub; c < chunks; c += lanes )
	{
		const uint4 bits = *(device const uint4 *)(weights + c * (SPARK_NVFP4_CHUNK / 2u));
		float4 xs[8];
		for ( uint part = 0u; part < 4u; part++ )
			SparkUnpack8(*(device const uint4 *)(row + c * SPARK_NVFP4_CHUNK + 8u * part), xs[2u * part], xs[2u * part + 1u]);
		const float low = SparkE2m1Fma8(bits.y, xs[2], xs[3], SparkE2m1Fma8(bits.x, xs[0], xs[1], 0.0f));
		const float high = SparkE2m1Fma8(bits.w, xs[6], xs[7], SparkE2m1Fma8(bits.z, xs[4], xs[5], 0.0f));
		const float low_scale = SparkMathUe4m3ToFloat(scales[2u * c]) * global, high_scale = SparkMathUe4m3ToFloat(scales[2u * c + 1u]) * global;
		acc = fma(high_scale, high, fma(low_scale, low, acc));
	}
	for ( uint step = lanes / 2u; step > 0u; step >>= 1u )
		acc += simd_shuffle_xor(acc, step);
	if ( sub == 0u && neuron < N )
		out[(ulong)route * p.output_stride + neuron] = SparkMathFloatToBf16(acc);
}
