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

static inline void SparkFp8Decode(uint4 bits, threadgroup const float *table, thread float *w)
{
	const uint words[4] = { bits.x, bits.y, bits.z, bits.w };
	for ( uint word = 0u; word < 4u; word++ )
		for ( uint j = 0u; j < 4u; j++ )
			w[word * 4u + j] = table[(words[word] >> (8u * j)) & 0xffu];
}

static inline float SparkFp8Chunk(thread const float *w, device const ushort *x)
{
	float4 xs[4];
	float sum = 0.0f;
	SparkUnpack8(*(device const uint4 *)x, xs[0], xs[1]);
	SparkUnpack8(*(device const uint4 *)(x + 8u), xs[2], xs[3]);
	for ( uint j = 0u; j < 16u; j++ )
		sum = fma(w[j], xs[j / 4u][j % 4u], sum);
	return(sum);
}

template <uint R, uint O, bool MX>
static inline void SparkFp8Rows(constant SparkGemvParams &p, device const ushort *x, device const uchar *w, device const uchar *scale,
	device ushort *out, uint first, uint lane, uint simd, threadgroup float *partial, threadgroup const float *table)
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
		for ( uint i = 0u; i < O; i++ )
		{
			float weights[SPARK_FP8_CHUNK];
			const float block = SparkFp8Block<MX>(p, scale, rows[i], k);
			SparkFp8Decode(*(device const uint4 *)(w + rows[i] * K + k), table, weights);
			for ( uint r = 0u; r < R; r++ )
				acc[i][r] = fma(block, SparkFp8Chunk(weights, x + (ulong)r * p.input_stride + k), acc[i][r]);
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

static inline void SparkFp8Table(threadgroup float *table, uint lane, uint simd)
{
	for ( uint code = simd * 32u + lane; code < 256u; code += SPARK_GEMV_SIMDS * 32u )
		table[code] = SparkMathE4m3ToFloat(code);
	threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint R, uint O, bool MX>
kernel void spark_gemv_fp8(constant SparkGemvParams &p [[buffer(0)]], device const ushort *x [[buffer(1)]], device const uchar *w [[buffer(2)]],
	device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup float partial[SPARK_GEMV_SIMDS * O * R];
	threadgroup float table[256];
	SparkFp8Table(table, lane, simd);
	SparkFp8Rows<R, O, MX>(p, x, w, scale, out, group * O, lane, simd, partial, table);
}

template <bool MX>
kernel void spark_gemv_fp8_routed(constant SparkGemvParams &p [[buffer(0)]], device const ushort *x [[buffer(1)]], device const uchar *w [[buffer(2)]],
	device const uchar *scale [[buffer(3)]], device ushort *out [[buffer(4)]], device const uint *experts [[buffer(5)]],
	uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup float partial[SPARK_GEMV_SIMDS * SPARK_GEMV_OUTPUTS];
	threadgroup float table[256];
	const uint K = p.input_dimension, N = p.output_dimension, route = group.y, first = group.x * SPARK_GEMV_OUTPUTS;
	if ( experts[route] == SPARK_ROUTE_SKIP )
	{
		if ( simd == 0u && lane < SPARK_GEMV_OUTPUTS && first + lane < N )
			out[(ulong)route * p.output_stride + first + lane] = 0u;
		return;
	}
	const ulong expert = experts[route];
	SparkFp8Table(table, lane, simd);
	SparkFp8Rows<1u, SPARK_GEMV_OUTPUTS, MX>(p, x + (ulong)(route / p.route_divisor) * p.input_stride, w + expert * N * K,
		scale + expert * N * p.scale_stride * (MX ? 1u : 4u), out + (ulong)route * p.output_stride, first, lane, simd, partial, table);
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
	for ( uint j = 0u; j < 4u; j++ )
		y[j] = SparkMathFloatToBf16(SparkMathE4m3ToFloat((bits >> (8u * j)) & 0xffu));
}

template [[host_name("spark_dequant_fp8")]] kernel void spark_dequant_fp8<false>(constant SparkGemvParams &p [[buffer(0)]],
	device const uchar *w [[buffer(1)]], device const uchar *scale [[buffer(2)]], device ushort *out [[buffer(3)]], uint2 index [[thread_position_in_grid]]);
template [[host_name("spark_dequant_mxfp8")]] kernel void spark_dequant_fp8<true>(constant SparkGemvParams &p [[buffer(0)]],
	device const uchar *w [[buffer(1)]], device const uchar *scale [[buffer(2)]], device ushort *out [[buffer(3)]], uint2 index [[thread_position_in_grid]]);

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
			const uint words[4] = { bits.x, bits.y, bits.z, bits.w };
			float sum = 0.0f;
			for ( uint j = 0u; j < SPARK_MXFP4_GROUP; j++ )
				sum = fma(SparkMathE2m1ToFloat((words[j / 8u] >> (4u * (j % 8u))) & 15u), xs[j / 4u][j % 4u], sum);
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

kernel void spark_gemv_routed(constant SparkGemvParams &p [[buffer(0)]], device const ushort *x [[buffer(1)]], device const ushort *w [[buffer(2)]],
	device ushort *out [[buffer(3)]], device const uint *experts [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup float partial[SPARK_GEMV_SIMDS * SPARK_GEMV_OUTPUTS];
	const uint route = group.y;
	if ( experts[route] == SPARK_ROUTE_SKIP )
	{
		if ( simd == 0u && lane < SPARK_GEMV_OUTPUTS && group.x * SPARK_GEMV_OUTPUTS + lane < p.output_dimension )
			out[(ulong)route * p.output_stride + group.x * SPARK_GEMV_OUTPUTS + lane] = 0u;
		return;
	}
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
		const uint words[4] = { bits.x, bits.y, bits.z, bits.w };
		float4 xs[8];
		for ( uint part = 0u; part < 4u; part++ )
			SparkUnpack8(*(device const uint4 *)(row + c * SPARK_NVFP4_CHUNK + 8u * part), xs[2u * part], xs[2u * part + 1u]);
		float low = 0.0f, high = 0.0f;
		for ( uint j = 0u; j < SPARK_NVFP4_GROUP; j++ )
			low = fma(SparkMathE2m1ToFloat((words[j / 8u] >> (4u * (j % 8u))) & 15u), xs[j / 4u][j % 4u], low);
		for ( uint j = SPARK_NVFP4_GROUP; j < SPARK_NVFP4_CHUNK; j++ )
			high = fma(SparkMathE2m1ToFloat((words[j / 8u] >> (4u * (j % 8u))) & 15u), xs[j / 4u][j % 4u], high);
		const float low_scale = SparkMathUe4m3ToFloat(scales[2u * c]) * global, high_scale = SparkMathUe4m3ToFloat(scales[2u * c + 1u]) * global;
		acc = fma(high_scale, high, fma(low_scale, low, acc));
	}
	for ( uint step = lanes / 2u; step > 0u; step >>= 1u )
		acc += simd_shuffle_xor(acc, step);
	if ( sub == 0u && neuron < N )
		out[(ulong)route * p.output_stride + neuron] = SparkMathFloatToBf16(acc);
}
