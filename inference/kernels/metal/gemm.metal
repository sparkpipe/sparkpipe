#include "inference/kernels/metal/common.metal"

#define SPARK_GEMM_BM 64u
#define SPARK_GEMM_BN 64u
#define SPARK_GEMM_BK 32u
#define SPARK_GEMM_THREADS 128u
#define SPARK_GEMM_CHUNKS (SPARK_GEMM_BM * SPARK_GEMM_BK / 8u)

static inline uint4 SparkGatedChunk(uint4 gate_bits, uint4 up_bits)
{
	float4 gl, gh, ul, uh;
	gl = float4(as_type<float>(gate_bits.x << 16), as_type<float>(gate_bits.x & 0xffff0000u), as_type<float>(gate_bits.y << 16), as_type<float>(gate_bits.y & 0xffff0000u));
	gh = float4(as_type<float>(gate_bits.z << 16), as_type<float>(gate_bits.z & 0xffff0000u), as_type<float>(gate_bits.w << 16), as_type<float>(gate_bits.w & 0xffff0000u));
	ul = float4(as_type<float>(up_bits.x << 16), as_type<float>(up_bits.x & 0xffff0000u), as_type<float>(up_bits.y << 16), as_type<float>(up_bits.y & 0xffff0000u));
	uh = float4(as_type<float>(up_bits.z << 16), as_type<float>(up_bits.z & 0xffff0000u), as_type<float>(up_bits.w << 16), as_type<float>(up_bits.w & 0xffff0000u));
	const float values[8] = { SparkSilu(gl.x) * ul.x, SparkSilu(gl.y) * ul.y, SparkSilu(gl.z) * ul.z, SparkSilu(gl.w) * ul.w,
		SparkSilu(gh.x) * uh.x, SparkSilu(gh.y) * uh.y, SparkSilu(gh.z) * uh.z, SparkSilu(gh.w) * uh.w };
	uint4 packed;
	packed.x = uint(SparkFloatToBf16(values[0])) | (uint(SparkFloatToBf16(values[1])) << 16);
	packed.y = uint(SparkFloatToBf16(values[2])) | (uint(SparkFloatToBf16(values[3])) << 16);
	packed.z = uint(SparkFloatToBf16(values[4])) | (uint(SparkFloatToBf16(values[5])) << 16);
	packed.w = uint(SparkFloatToBf16(values[6])) | (uint(SparkFloatToBf16(values[7])) << 16);
	return packed;
}

template <bool F32, bool GATED>
kernel void spark_gemm(constant SparkGemmParams &p [[buffer(0)]], device const ushort *x [[buffer(1)]], device const ushort *w [[buffer(2)]],
	device void *out [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
	uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
	threadgroup uint4 x_tile[SPARK_GEMM_CHUNKS];
	threadgroup uint4 w_tile[SPARK_GEMM_CHUNKS];
	threadgroup float c_tile[4u * 32u * 32u];
	const uint M = p.rows, N = p.output_dimension, K = p.input_dimension;
	const uint m0 = group.y * SPARK_GEMM_BM, n0 = group.x * SPARK_GEMM_BN;
	const uint sm = (simd / 2u) * 32u, sn = (simd % 2u) * 32u;
	threadgroup const bfloat *xs = (threadgroup const bfloat *)x_tile;
	threadgroup const bfloat *ws = (threadgroup const bfloat *)w_tile;
	simdgroup_matrix<float, 8, 8> acc[4][4];
	for ( uint i = 0u; i < 4u; i++ )
		for ( uint j = 0u; j < 4u; j++ )
			acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
	for ( uint k0 = 0u; k0 < K; k0 += SPARK_GEMM_BK )
	{
		for ( uint chunk = tid; chunk < SPARK_GEMM_CHUNKS; chunk += SPARK_GEMM_THREADS )
		{
			const uint row = chunk / (SPARK_GEMM_BK / 8u), column = (chunk % (SPARK_GEMM_BK / 8u)) * 8u;
			const uint m = m0 + row, n = n0 + row;
			uint4 value = uint4(0u);
			if ( m < M )
			{
				device const ushort *source = x + (ulong)m * p.input_stride + k0 + column;
				value = *(device const uint4 *)source;
				if ( GATED )
					value = SparkGatedChunk(value, *(device const uint4 *)(source + K));
			}
			x_tile[chunk] = value;
			w_tile[chunk] = n < N ? *(device const uint4 *)(w + (ulong)n * K + k0 + column) : uint4(0u);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
		for ( uint kk = 0u; kk < SPARK_GEMM_BK; kk += 8u )
		{
			simdgroup_matrix<bfloat, 8, 8> a[4], b[4];
			for ( uint i = 0u; i < 4u; i++ )
				simdgroup_load(a[i], xs + (sm + i * 8u) * SPARK_GEMM_BK + kk, SPARK_GEMM_BK);
			for ( uint j = 0u; j < 4u; j++ )
				simdgroup_load(b[j], ws + (sn + j * 8u) * SPARK_GEMM_BK + kk, SPARK_GEMM_BK, ulong2(0, 0), true);
			for ( uint i = 0u; i < 4u; i++ )
				for ( uint j = 0u; j < 4u; j++ )
					simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	threadgroup float *cs = c_tile + simd * 32u * 32u;
	for ( uint i = 0u; i < 4u; i++ )
		for ( uint j = 0u; j < 4u; j++ )
			simdgroup_store(acc[i][j], cs + (i * 8u) * 32u + j * 8u, 32u);
	simdgroup_barrier(mem_flags::mem_threadgroup);
	for ( uint e = lane; e < 32u * 32u; e += 32u )
	{
		const uint m = m0 + sm + e / 32u, n = n0 + sn + e % 32u;
		if ( m < M && n < N )
		{
			const ulong at = (ulong)m * p.output_stride + n;
			if ( F32 )
				((device float *)out)[at] = cs[e];
			else
				((device ushort *)out)[at] = SparkFloatToBf16(cs[e]);
		}
	}
}

#define SPARK_GEMM_INSTANCE(f32, gated, name) \
	template [[host_name(name)]] kernel void spark_gemm<f32, gated>(constant SparkGemmParams &p [[buffer(0)]], \
		device const ushort *x [[buffer(1)]], device const ushort *w [[buffer(2)]], device void *out [[buffer(3)]], \
		uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
		uint simd [[simdgroup_index_in_threadgroup]]);

SPARK_GEMM_INSTANCE(false, false, "spark_gemm")
SPARK_GEMM_INSTANCE(true, false, "spark_gemm_f32")
SPARK_GEMM_INSTANCE(false, true, "spark_gemm_gated")
