#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef struct __align__(32) RefU32x8
{
	uint32_t v[8];
} RefU32x8;

typedef struct __align__(16) RefU32x4
{
	uint32_t v[4];
} RefU32x4;

static __device__ __forceinline__ RefU32x8 RefLdg256Stream(const void *ptr)
{
	RefU32x8 r;
	asm volatile(
		"ld.global.L1::evict_first.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
		: "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]),
		  "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
		: "l"(ptr));
	return(r);
}

static __device__ __forceinline__ RefU32x8 RefLdg256Persist(const void *ptr)
{
	RefU32x8 r;
	asm volatile(
		"ld.global.L1::evict_last.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
		: "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]),
		  "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
		: "l"(ptr));
	return(r);
}

static __device__ __forceinline__ RefU32x4 RefLdg128Stream(const void *ptr)
{
	RefU32x4 r;
	asm volatile(
		"ld.global.L1::evict_first.v4.u32 {%0,%1,%2,%3}, [%4];"
		: "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
		: "l"(ptr));
	return(r);
}

static __device__ __forceinline__ RefU32x4 RefLdg128Persist(const void *ptr)
{
	RefU32x4 r;
	asm volatile(
		"ld.global.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];"
		: "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3])
		: "l"(ptr));
	return(r);
}

static __device__ __forceinline__ void RefStg256(void *ptr, const RefU32x8 &r)
{
	asm volatile(
		"st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
		:
		: "l"(ptr), "r"(r.v[0]), "r"(r.v[1]), "r"(r.v[2]), "r"(r.v[3]),
		  "r"(r.v[4]), "r"(r.v[5]), "r"(r.v[6]), "r"(r.v[7]));
}

static __device__ __forceinline__ void RefStg128(void *ptr, const RefU32x4 &r)
{
	asm volatile(
		"st.global.v4.u32 [%0], {%1,%2,%3,%4};"
		:
		: "l"(ptr), "r"(r.v[0]), "r"(r.v[1]), "r"(r.v[2]), "r"(r.v[3]));
}

static __device__ __forceinline__ float RefBf16Lo(uint32_t w)
{
	return(__uint_as_float(w << 16u));
}

static __device__ __forceinline__ float RefBf16Hi(uint32_t w)
{
	return(__uint_as_float(w & 0xffff0000u));
}

static __device__ __forceinline__ uint32_t RefPackBf16x2(float lo, float hi)
{
	uint32_t out;
	asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(hi), "f"(lo));
	return(out);
}

static __device__ __forceinline__ uint16_t RefF32ToBf16(float x)
{
	return((uint16_t)RefPackBf16x2(x,0.0f));
}

static __device__ __forceinline__ uint16_t RefPackE4m3x2(float lo, float hi)
{
	uint16_t out;
	asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(out) : "f"(hi), "f"(lo));
	return(out);
}

static __device__ __forceinline__ float RefExpFast(float x)
{
	float r;
	asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x * 1.4426950408889634f));
	return(r);
}

static __device__ __forceinline__ float RefRcpFast(float x)
{
	float r;
	asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
	return(r);
}

static __device__ __forceinline__ float RefWarpSum(float v)
{
	v += __shfl_xor_sync(0xffffffffu,v,16);
	v += __shfl_xor_sync(0xffffffffu,v,8);
	v += __shfl_xor_sync(0xffffffffu,v,4);
	v += __shfl_xor_sync(0xffffffffu,v,2);
	v += __shfl_xor_sync(0xffffffffu,v,1);
	return(v);
}

static __device__ __forceinline__ float RefWarpMax(float v)
{
	v = fmaxf(v,__shfl_xor_sync(0xffffffffu,v,16));
	v = fmaxf(v,__shfl_xor_sync(0xffffffffu,v,8));
	v = fmaxf(v,__shfl_xor_sync(0xffffffffu,v,4));
	v = fmaxf(v,__shfl_xor_sync(0xffffffffu,v,2));
	v = fmaxf(v,__shfl_xor_sync(0xffffffffu,v,1));
	return(v);
}

static __device__ __forceinline__ float RefQuadMax(float v)
{
	v = fmaxf(v,__shfl_xor_sync(0xffffffffu,v,2));
	v = fmaxf(v,__shfl_xor_sync(0xffffffffu,v,1));
	return(v);
}

static __device__ __forceinline__ float RefUe4m3ToF32(uint32_t b)
{
	const uint32_t exp = (b >> 3u) & 0xfu;
	const uint32_t mant = b & 0x7u;
	if ( exp == 0u )
		return((float)mant * (1.0f / 512.0f));
	return(__uint_as_float(((exp + 120u) << 23u) | (mant << 20u)));
}

static __device__ __forceinline__ float RefUe8m0ToF32(uint32_t b)
{
	const uint32_t eb = b & 0xffu;
	if ( eb == 255u )
		return(0.0f);
	if ( eb == 0u )
		return(__uint_as_float(0x00400000u));
	return(__uint_as_float(eb << 23u));
}

static __device__ __forceinline__ float2 RefE2m1Pair(uint32_t word, uint32_t shift)
{
	uint32_t fp16x2;
	asm("{\n\t.reg .b8 narrow;\n\tcvt.u8.u32 narrow, %1;\n\tcvt.rn.f16x2.e2m1x2 %0, narrow;\n\t}\n"
		: "=r"(fp16x2) : "r"(word >> shift));
	return(make_float2(__half2float(__ushort_as_half((uint16_t)(fp16x2 & 0xffffu))),
		__half2float(__ushort_as_half((uint16_t)(fp16x2 >> 16u)))));
}

extern "C" __global__ __launch_bounds__(256) void rmsnorm_bf16_vec16(
	const uint32_t *__restrict__ x_bf16,
	const uint32_t *__restrict__ weight_bf16,
	uint32_t *__restrict__ out_bf16,
	uint32_t rows,
	uint32_t chunks16,
	float eps,
	float inv_dim)
{
	__shared__ float smem[8];
	const uint32_t row = blockIdx.x;
	if ( row >= rows )
		return;
	const uint32_t tid = threadIdx.x;
	const uint32_t wid = tid >> 5u;
	const uint32_t lane = tid & 31u;
	const uint8_t *x_row = (const uint8_t *)x_bf16 + (uint64_t)row * chunks16 * 32u;
	uint8_t *o_row = (uint8_t *)out_bf16 + (uint64_t)row * chunks16 * 32u;
	float s0 = 0.0f,s1 = 0.0f,s2 = 0.0f,s3 = 0.0f;
	for (uint32_t c = tid; c < chunks16; c += 256u)
	{
		const RefU32x8 xv = RefLdg256Persist(x_row + (uint64_t)c * 32u);
		#pragma unroll
		for (uint32_t k = 0u; k < 4u; k++)
		{
			const float v0 = RefBf16Lo(xv.v[k * 2u]);
			const float v1 = RefBf16Hi(xv.v[k * 2u]);
			const float v2 = RefBf16Lo(xv.v[k * 2u + 1u]);
			const float v3 = RefBf16Hi(xv.v[k * 2u + 1u]);
			s0 = fmaf(v0,v0,s0);
			s1 = fmaf(v1,v1,s1);
			s2 = fmaf(v2,v2,s2);
			s3 = fmaf(v3,v3,s3);
		}
	}
	const float wsum = RefWarpSum((s0 + s1) + (s2 + s3));
	if ( lane == 0u )
		smem[wid] = wsum;
	__syncthreads();
	float bsum = 0.0f;
	#pragma unroll
	for (uint32_t w = 0u; w < 8u; w++)
		bsum += smem[w];
	const float inv_rms = rsqrtf(fmaf(bsum,inv_dim,eps));
	for (uint32_t c = tid; c < chunks16; c += 256u)
	{
		const RefU32x8 xv = RefLdg256Persist(x_row + (uint64_t)c * 32u);
		const RefU32x8 wv = RefLdg256Persist((const uint8_t *)weight_bf16 + (uint64_t)c * 32u);
		RefU32x8 ov;
		#pragma unroll
		for (uint32_t k = 0u; k < 8u; k++)
		{
			const float lo = RefBf16Lo(xv.v[k]) * inv_rms * RefBf16Lo(wv.v[k]);
			const float hi = RefBf16Hi(xv.v[k]) * inv_rms * RefBf16Hi(wv.v[k]);
			ov.v[k] = RefPackBf16x2(lo,hi);
		}
		RefStg256(o_row + (uint64_t)c * 32u,ov);
	}
}

extern "C" __global__ __launch_bounds__(256) void fused_residual_rmsnorm_bf16x8(
	const uint32_t *__restrict__ x_bf16,
	uint32_t *__restrict__ res_bf16,
	const uint32_t *__restrict__ weight_bf16,
	uint32_t *__restrict__ out_bf16,
	uint32_t rows,
	uint32_t chunks16,
	float eps,
	float inv_dim)
{
	__shared__ float smem[8];
	const uint32_t row = blockIdx.x;
	if ( row >= rows )
		return;
	const uint32_t tid = threadIdx.x;
	const uint32_t wid = tid >> 5u;
	const uint32_t lane = tid & 31u;
	const uint8_t *x_row = (const uint8_t *)x_bf16 + (uint64_t)row * chunks16 * 32u;
	uint8_t *r_row = (uint8_t *)res_bf16 + (uint64_t)row * chunks16 * 32u;
	uint8_t *o_row = (uint8_t *)out_bf16 + (uint64_t)row * chunks16 * 32u;
	float s0 = 0.0f,s1 = 0.0f,s2 = 0.0f,s3 = 0.0f;
	for (uint32_t c = tid; c < chunks16; c += 256u)
	{
		const RefU32x8 xv = RefLdg256Stream(x_row + (uint64_t)c * 32u);
		const RefU32x8 rv = RefLdg256Stream(r_row + (uint64_t)c * 32u);
		RefU32x8 hv;
		#pragma unroll
		for (uint32_t k = 0u; k < 4u; k++)
		{
			const float v0 = RefBf16Lo(xv.v[k * 2u]) + RefBf16Lo(rv.v[k * 2u]);
			const float v1 = RefBf16Hi(xv.v[k * 2u]) + RefBf16Hi(rv.v[k * 2u]);
			const float v2 = RefBf16Lo(xv.v[k * 2u + 1u]) + RefBf16Lo(rv.v[k * 2u + 1u]);
			const float v3 = RefBf16Hi(xv.v[k * 2u + 1u]) + RefBf16Hi(rv.v[k * 2u + 1u]);
			s0 = fmaf(v0,v0,s0);
			s1 = fmaf(v1,v1,s1);
			s2 = fmaf(v2,v2,s2);
			s3 = fmaf(v3,v3,s3);
			hv.v[k * 2u] = RefPackBf16x2(v0,v1);
			hv.v[k * 2u + 1u] = RefPackBf16x2(v2,v3);
		}
		RefStg256(r_row + (uint64_t)c * 32u,hv);
	}
	const float wsum = RefWarpSum((s0 + s1) + (s2 + s3));
	if ( lane == 0u )
		smem[wid] = wsum;
	__syncthreads();
	float bsum = 0.0f;
	#pragma unroll
	for (uint32_t w = 0u; w < 8u; w++)
		bsum += smem[w];
	const float inv_rms = rsqrtf(fmaf(bsum,inv_dim,eps));
	for (uint32_t c = tid; c < chunks16; c += 256u)
	{
		const RefU32x8 hv = RefLdg256Persist(r_row + (uint64_t)c * 32u);
		const RefU32x8 wv = RefLdg256Persist((const uint8_t *)weight_bf16 + (uint64_t)c * 32u);
		RefU32x8 ov;
		#pragma unroll
		for (uint32_t k = 0u; k < 8u; k++)
		{
			const float lo = RefBf16Lo(hv.v[k]) * inv_rms * RefBf16Lo(wv.v[k]);
			const float hi = RefBf16Hi(hv.v[k]) * inv_rms * RefBf16Hi(wv.v[k]);
			ov.v[k] = RefPackBf16x2(lo,hi);
		}
		RefStg256(o_row + (uint64_t)c * 32u,ov);
	}
}

extern "C" __global__ __launch_bounds__(256) void swiglu_packed_bf16x8(
	const uint32_t *__restrict__ gate_bf16,
	const uint32_t *__restrict__ up_bf16,
	uint32_t *__restrict__ out_bf16,
	uint32_t total_chunks16)
{
	const uint32_t idx = blockIdx.x * 256u + threadIdx.x;
	if ( idx >= total_chunks16 )
		return;
	const RefU32x8 gv = RefLdg256Stream((const uint8_t *)gate_bf16 + (uint64_t)idx * 32u);
	const RefU32x8 uv = RefLdg256Stream((const uint8_t *)up_bf16 + (uint64_t)idx * 32u);
	RefU32x8 ov;
	#pragma unroll
	for (uint32_t k = 0u; k < 4u; k++)
	{
		const float g0 = RefBf16Lo(gv.v[k * 2u]);
		const float g1 = RefBf16Hi(gv.v[k * 2u]);
		const float g2 = RefBf16Lo(gv.v[k * 2u + 1u]);
		const float g3 = RefBf16Hi(gv.v[k * 2u + 1u]);
		float e0,e1,e2,e3;
		asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e0) : "f"(g0 * -1.4426950408889634f));
		asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e1) : "f"(g1 * -1.4426950408889634f));
		asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e2) : "f"(g2 * -1.4426950408889634f));
		asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e3) : "f"(g3 * -1.4426950408889634f));
		asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(e0) : "f"(e0 + 1.0f));
		asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(e1) : "f"(e1 + 1.0f));
		asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(e2) : "f"(e2 + 1.0f));
		asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(e3) : "f"(e3 + 1.0f));
		ov.v[k * 2u] = RefPackBf16x2((g0 * e0) * RefBf16Lo(uv.v[k * 2u]),(g1 * e1) * RefBf16Hi(uv.v[k * 2u]));
		ov.v[k * 2u + 1u] = RefPackBf16x2((g2 * e2) * RefBf16Lo(uv.v[k * 2u + 1u]),(g3 * e3) * RefBf16Hi(uv.v[k * 2u + 1u]));
	}
	RefStg256((uint8_t *)out_bf16 + (uint64_t)idx * 32u,ov);
}

extern "C" __global__ __launch_bounds__(128) void skinny_bf16_b1(
	const uint32_t *__restrict__ weight_bf16,
	const uint32_t *__restrict__ act_bf16,
	uint16_t *__restrict__ out_bf16,
	uint32_t chunks16,
	uint32_t out_dim,
	uint32_t out_col_off)
{
	const uint32_t wid = threadIdx.x >> 5u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t row0 = ((blockIdx.x << 2u) + wid) << 1u;
	if ( row0 >= out_dim )
		return;
	const uint8_t *w_r0 = (const uint8_t *)weight_bf16 + (uint64_t)row0 * chunks16 * 32u;
	const uint8_t *w_r1 = w_r0 + (uint64_t)chunks16 * 32u;
	float a0 = 0.0f,a1 = 0.0f,a2 = 0.0f,a3 = 0.0f;
	float b0 = 0.0f,b1 = 0.0f,b2 = 0.0f,b3 = 0.0f;
	for (uint32_t c = lane; c < chunks16; c += 32u)
	{
		const RefU32x8 w0 = RefLdg256Stream(w_r0 + (uint64_t)c * 32u);
		const RefU32x8 w1 = RefLdg256Stream(w_r1 + (uint64_t)c * 32u);
		const RefU32x8 av = RefLdg256Persist((const uint8_t *)act_bf16 + (uint64_t)c * 32u);
		#pragma unroll
		for (uint32_t k = 0u; k < 4u; k++)
		{
			const float fx0 = RefBf16Lo(av.v[k * 2u]);
			const float fx1 = RefBf16Hi(av.v[k * 2u]);
			const float fx2 = RefBf16Lo(av.v[k * 2u + 1u]);
			const float fx3 = RefBf16Hi(av.v[k * 2u + 1u]);
			a0 = fmaf(RefBf16Lo(w0.v[k * 2u]),fx0,a0);
			a1 = fmaf(RefBf16Hi(w0.v[k * 2u]),fx1,a1);
			a2 = fmaf(RefBf16Lo(w0.v[k * 2u + 1u]),fx2,a2);
			a3 = fmaf(RefBf16Hi(w0.v[k * 2u + 1u]),fx3,a3);
			b0 = fmaf(RefBf16Lo(w1.v[k * 2u]),fx0,b0);
			b1 = fmaf(RefBf16Hi(w1.v[k * 2u]),fx1,b1);
			b2 = fmaf(RefBf16Lo(w1.v[k * 2u + 1u]),fx2,b2);
			b3 = fmaf(RefBf16Hi(w1.v[k * 2u + 1u]),fx3,b3);
		}
	}
	const float t0 = RefWarpSum((a0 + a1) + (a2 + a3));
	const float t1 = RefWarpSum((b0 + b1) + (b2 + b3));
	if ( lane == 0u )
	{
		out_bf16[out_col_off + row0] = RefF32ToBf16(t0);
		out_bf16[out_col_off + row0 + 1u] = RefF32ToBf16(t1);
	}
}

extern "C" __global__ __launch_bounds__(128) void skinny_nvfp4_b1(
	const uint32_t *__restrict__ weight,
	const uint32_t *__restrict__ act_bf16,
	const uint32_t *__restrict__ scales_ue4m3,
	const float *__restrict__ global_scale_ptr,
	uint16_t *__restrict__ out_bf16,
	uint32_t chunks32,
	uint32_t out_dim,
	uint32_t out_col_off)
{
	const uint32_t wid = threadIdx.x >> 5u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t row0 = ((blockIdx.x << 2u) + wid) << 1u;
	if ( row0 >= out_dim )
		return;
	const uint8_t *w_r0 = (const uint8_t *)weight + (uint64_t)row0 * chunks32 * 32u;
	const uint8_t *w_r1 = w_r0 + (uint64_t)chunks32 * 32u;
	const uint32_t *s_r0 = scales_ue4m3 + (uint64_t)row0 * chunks32;
	const uint32_t *s_r1 = s_r0 + chunks32;
	const float gscale = __ldg(global_scale_ptr);
	float a0 = 0.0f,a1 = 0.0f,b0 = 0.0f,b1 = 0.0f;
	for (uint32_t c = lane; c < chunks32; c += 32u)
	{
		const RefU32x8 w0 = RefLdg256Stream(w_r0 + (uint64_t)c * 32u);
		const RefU32x8 w1 = RefLdg256Stream(w_r1 + (uint64_t)c * 32u);
		const uint32_t sc0 = s_r0[c];
		const uint32_t sc1 = s_r1[c];
		const uint32_t ab = c << 2u;
		#pragma unroll
		for (uint32_t g = 0u; g < 4u; g++)
		{
			const RefU32x8 av = RefLdg256Persist((const uint8_t *)act_bf16 + (uint64_t)(ab + g) * 32u);
			const float sg0 = RefUe4m3ToF32((sc0 >> (g << 3u)) & 0xffu);
			const float sg1 = RefUe4m3ToF32((sc1 >> (g << 3u)) & 0xffu);
			float p0 = 0.0f,p1 = 0.0f,p2 = 0.0f,p3 = 0.0f;
			float q0 = 0.0f,q1 = 0.0f,q2 = 0.0f,q3 = 0.0f;
			#pragma unroll
			for (uint32_t k = 0u; k < 4u; k++)
			{
				const float x0 = RefBf16Lo(av.v[k * 2u]);
				const float x1 = RefBf16Hi(av.v[k * 2u]);
				const float x2 = RefBf16Lo(av.v[k * 2u + 1u]);
				const float x3 = RefBf16Hi(av.v[k * 2u + 1u]);
				const uint32_t w0w = (k < 2u) ? w0.v[g * 2u] : w0.v[g * 2u + 1u];
				const uint32_t w1w = (k < 2u) ? w1.v[g * 2u] : w1.v[g * 2u + 1u];
				const uint32_t sh = (k & 1u) << 4u;
				const float2 d0 = RefE2m1Pair(w0w,sh);
				const float2 d1 = RefE2m1Pair(w0w,sh + 8u);
				p0 = fmaf(d0.x,x0,p0);
				p1 = fmaf(d0.y,x1,p1);
				p2 = fmaf(d1.x,x2,p2);
				p3 = fmaf(d1.y,x3,p3);
				const float2 e0 = RefE2m1Pair(w1w,sh);
				const float2 e1 = RefE2m1Pair(w1w,sh + 8u);
				q0 = fmaf(e0.x,x0,q0);
				q1 = fmaf(e0.y,x1,q1);
				q2 = fmaf(e1.x,x2,q2);
				q3 = fmaf(e1.y,x3,q3);
			}
			if ( (g & 1u) == 0u )
			{
				a0 = fmaf(sg0,(p0 + p1) + (p2 + p3),a0);
				b0 = fmaf(sg1,(q0 + q1) + (q2 + q3),b0);
			}
			else
			{
				a1 = fmaf(sg0,(p0 + p1) + (p2 + p3),a1);
				b1 = fmaf(sg1,(q0 + q1) + (q2 + q3),b1);
			}
		}
	}
	const float t0 = RefWarpSum(a0 + a1) * gscale;
	const float t1 = RefWarpSum(b0 + b1) * gscale;
	if ( lane == 0u )
	{
		out_bf16[out_col_off + row0] = RefF32ToBf16(t0);
		out_bf16[out_col_off + row0 + 1u] = RefF32ToBf16(t1);
	}
}

extern "C" __global__ __launch_bounds__(128) void skinny_mxfp4_b1(
	const uint32_t *__restrict__ weight,
	const uint32_t *__restrict__ act_bf16,
	const uint16_t *__restrict__ scales_ue8m0,
	uint16_t *__restrict__ out_bf16,
	uint32_t chunks32,
	uint32_t out_dim,
	uint32_t out_col_off)
{
	const uint32_t wid = threadIdx.x >> 5u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t row0 = ((blockIdx.x << 2u) + wid) << 1u;
	if ( row0 >= out_dim )
		return;
	const uint8_t *w_r0 = (const uint8_t *)weight + (uint64_t)row0 * chunks32 * 32u;
	const uint8_t *w_r1 = w_r0 + (uint64_t)chunks32 * 32u;
	const uint16_t *s_r0 = scales_ue8m0 + (uint64_t)row0 * chunks32;
	const uint16_t *s_r1 = s_r0 + chunks32;
	float a0 = 0.0f,a1 = 0.0f,b0 = 0.0f,b1 = 0.0f;
	for (uint32_t c = lane; c < chunks32; c += 32u)
	{
		const RefU32x8 w0 = RefLdg256Stream(w_r0 + (uint64_t)c * 32u);
		const RefU32x8 w1 = RefLdg256Stream(w_r1 + (uint64_t)c * 32u);
		const uint32_t sc0 = (uint32_t)s_r0[c];
		const uint32_t sc1 = (uint32_t)s_r1[c];
		const uint32_t ab = c << 2u;
		#pragma unroll
		for (uint32_t g = 0u; g < 2u; g++)
		{
			float p0 = 0.0f,p1 = 0.0f,p2 = 0.0f,p3 = 0.0f;
			float q0 = 0.0f,q1 = 0.0f,q2 = 0.0f,q3 = 0.0f;
			#pragma unroll
			for (uint32_t h = 0u; h < 2u; h++)
			{
				const RefU32x8 av = RefLdg256Persist((const uint8_t *)act_bf16 + (uint64_t)(ab + g * 2u + h) * 32u);
				#pragma unroll
				for (uint32_t k = 0u; k < 4u; k++)
				{
					const float x0 = RefBf16Lo(av.v[k * 2u]);
					const float x1 = RefBf16Hi(av.v[k * 2u]);
					const float x2 = RefBf16Lo(av.v[k * 2u + 1u]);
					const float x3 = RefBf16Hi(av.v[k * 2u + 1u]);
					const uint32_t w0w = (k < 2u) ? w0.v[g * 4u + h * 2u] : w0.v[g * 4u + h * 2u + 1u];
					const uint32_t w1w = (k < 2u) ? w1.v[g * 4u + h * 2u] : w1.v[g * 4u + h * 2u + 1u];
					const uint32_t sh = (k & 1u) << 4u;
					const float2 d0 = RefE2m1Pair(w0w,sh);
					const float2 d1 = RefE2m1Pair(w0w,sh + 8u);
					p0 = fmaf(d0.x,x0,p0);
					p1 = fmaf(d0.y,x1,p1);
					p2 = fmaf(d1.x,x2,p2);
					p3 = fmaf(d1.y,x3,p3);
					const float2 e0 = RefE2m1Pair(w1w,sh);
					const float2 e1 = RefE2m1Pair(w1w,sh + 8u);
					q0 = fmaf(e0.x,x0,q0);
					q1 = fmaf(e0.y,x1,q1);
					q2 = fmaf(e1.x,x2,q2);
					q3 = fmaf(e1.y,x3,q3);
				}
			}
			const float sg0 = RefUe8m0ToF32(sc0 >> (g << 3u));
			const float sg1 = RefUe8m0ToF32(sc1 >> (g << 3u));
			if ( g == 0u )
			{
				a0 = fmaf(sg0,(p0 + p1) + (p2 + p3),a0);
				b0 = fmaf(sg1,(q0 + q1) + (q2 + q3),b0);
			}
			else
			{
				a1 = fmaf(sg0,(p0 + p1) + (p2 + p3),a1);
				b1 = fmaf(sg1,(q0 + q1) + (q2 + q3),b1);
			}
		}
	}
	const float t0 = RefWarpSum(a0 + a1);
	const float t1 = RefWarpSum(b0 + b1);
	if ( lane == 0u )
	{
		out_bf16[out_col_off + row0] = RefF32ToBf16(t0);
		out_bf16[out_col_off + row0 + 1u] = RefF32ToBf16(t1);
	}
}

extern "C" __global__ __launch_bounds__(256) void latent_attn_decode_512_64(
	const uint32_t *__restrict__ q_nope_bf16,
	const uint32_t *__restrict__ q_rope_bf16,
	const uint32_t *__restrict__ kv_nope_bf16,
	const uint32_t *__restrict__ kv_rope_bf16,
	uint16_t *__restrict__ out_bf16,
	uint32_t num_heads,
	uint32_t kv_len,
	float scale)
{
	__shared__ float smem_max[8];
	__shared__ float smem_sum[8];
	__shared__ float smem_out[4096];
	const uint32_t head = blockIdx.x;
	if ( head >= num_heads )
		return;
	const uint32_t wid = threadIdx.x >> 5u;
	const uint32_t lane = threadIdx.x & 31u;
	const RefU32x8 qn = RefLdg256Persist((const uint8_t *)q_nope_bf16 + ((uint64_t)head * 32u + lane) * 32u);
	const uint32_t qr = __ldg(q_rope_bf16 + (uint64_t)head * 32u + lane);
	const float qr0 = RefBf16Lo(qr);
	const float qr1 = RefBf16Hi(qr);
	float acc[16];
	#pragma unroll
	for (uint32_t i = 0u; i < 16u; i++)
		acc[i] = 0.0f;
	float m_val = -1.0e30f;
	float l_val = 0.0f;
	for (uint32_t tok = wid; tok < kv_len; tok += 8u)
	{
		const RefU32x8 kn = RefLdg256Stream((const uint8_t *)kv_nope_bf16 + ((uint64_t)tok * 32u + lane) * 32u);
		const uint32_t kr = kv_rope_bf16[(uint64_t)tok * 32u + lane];
		float d0 = fmaf(RefBf16Lo(qn.v[0]),RefBf16Lo(kn.v[0]),qr0 * RefBf16Lo(kr));
		float d1 = fmaf(RefBf16Hi(qn.v[0]),RefBf16Hi(kn.v[0]),qr1 * RefBf16Hi(kr));
		float d2 = RefBf16Lo(qn.v[1]) * RefBf16Lo(kn.v[1]);
		float d3 = RefBf16Hi(qn.v[1]) * RefBf16Hi(kn.v[1]);
		#pragma unroll
		for (uint32_t k = 1u; k < 4u; k++)
		{
			d0 = fmaf(RefBf16Lo(qn.v[k * 2u]),RefBf16Lo(kn.v[k * 2u]),d0);
			d1 = fmaf(RefBf16Hi(qn.v[k * 2u]),RefBf16Hi(kn.v[k * 2u]),d1);
			d2 = fmaf(RefBf16Lo(qn.v[k * 2u + 1u]),RefBf16Lo(kn.v[k * 2u + 1u]),d2);
			d3 = fmaf(RefBf16Hi(qn.v[k * 2u + 1u]),RefBf16Hi(kn.v[k * 2u + 1u]),d3);
		}
		const float logit = RefWarpSum((d0 + d1) + (d2 + d3)) * scale;
		const float m_new = fmaxf(m_val,logit);
		const float alpha = RefExpFast(m_val - m_new);
		const float p = RefExpFast(logit - m_new);
		l_val = fmaf(l_val,alpha,p);
		m_val = m_new;
		#pragma unroll
		for (uint32_t k = 0u; k < 8u; k++)
		{
			acc[k * 2u] = fmaf(p,RefBf16Lo(kn.v[k]),acc[k * 2u] * alpha);
			acc[k * 2u + 1u] = fmaf(p,RefBf16Hi(kn.v[k]),acc[k * 2u + 1u] * alpha);
		}
	}
	if ( lane == 0u )
	{
		smem_max[wid] = m_val;
		smem_sum[wid] = l_val;
	}
	const uint32_t sbase = wid * 512u + lane * 16u;
	#pragma unroll
	for (uint32_t i = 0u; i < 16u; i++)
		smem_out[sbase + i] = acc[i];
	__syncthreads();
	const uint32_t tid = threadIdx.x;
	float gmax = -1.0e30f;
	#pragma unroll
	for (uint32_t w = 0u; w < 8u; w++)
		gmax = fmaxf(gmax,smem_max[w]);
	float gsum = 0.0f,o0 = 0.0f,o1 = 0.0f;
	#pragma unroll
	for (uint32_t w = 0u; w < 8u; w++)
	{
		const float wt = RefExpFast(smem_max[w] - gmax);
		gsum = fmaf(smem_sum[w],wt,gsum);
		o0 = fmaf(smem_out[w * 512u + tid],wt,o0);
		o1 = fmaf(smem_out[w * 512u + 256u + tid],wt,o1);
	}
	const float inv_l = RefRcpFast(gsum);
	uint16_t *obase = out_bf16 + (uint64_t)head * 512u;
	obase[tid] = RefF32ToBf16(o0 * inv_l);
	obase[256u + tid] = RefF32ToBf16(o1 * inv_l);
}

extern "C" __global__ __launch_bounds__(256) void router_softmax_256(
	const float *__restrict__ logits,
	float *__restrict__ probs,
	uint32_t rows)
{
	const uint32_t wid = threadIdx.x >> 5u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t row = (blockIdx.x << 3u) + wid;
	if ( row >= rows )
		return;
	const RefU32x8 v = RefLdg256Persist((const uint8_t *)logits + ((uint64_t)row * 32u + lane) * 32u);
	float f[8];
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; k++)
		f[k] = __uint_as_float(v.v[k]);
	const float m0 = fmaxf(fmaxf(f[0],f[1]),fmaxf(f[2],f[3]));
	const float m1 = fmaxf(fmaxf(f[4],f[5]),fmaxf(f[6],f[7]));
	const float rmax = RefWarpMax(fmaxf(m0,m1));
	float e[8];
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; k++)
		e[k] = RefExpFast(f[k] - rmax);
	const float s = ((e[0] + e[1]) + (e[2] + e[3])) + ((e[4] + e[5]) + (e[6] + e[7]));
	const float inv_sum = RefRcpFast(RefWarpSum(s));
	RefU32x8 ov;
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; k++)
		ov.v[k] = __float_as_uint(e[k] * inv_sum);
	RefStg256((uint8_t *)probs + ((uint64_t)row * 32u + lane) * 32u,ov);
}

extern "C" __global__ __launch_bounds__(256) void router_softmax_bf16_256(
	const uint32_t *__restrict__ logits_bf16,
	float *__restrict__ probs,
	uint32_t rows)
{
	const uint32_t wid = threadIdx.x >> 5u;
	const uint32_t lane = threadIdx.x & 31u;
	const uint32_t row = (blockIdx.x << 3u) + wid;
	if ( row >= rows )
		return;
	const RefU32x4 v = RefLdg128Persist((const uint8_t *)logits_bf16 + ((uint64_t)row * 32u + lane) * 16u);
	float f[8];
	#pragma unroll
	for (uint32_t k = 0u; k < 4u; k++)
	{
		f[k * 2u] = RefBf16Lo(v.v[k]);
		f[k * 2u + 1u] = RefBf16Hi(v.v[k]);
	}
	const float m0 = fmaxf(fmaxf(f[0],f[1]),fmaxf(f[2],f[3]));
	const float m1 = fmaxf(fmaxf(f[4],f[5]),fmaxf(f[6],f[7]));
	const float rmax = RefWarpMax(fmaxf(m0,m1));
	float e[8];
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; k++)
		e[k] = RefExpFast(f[k] - rmax);
	const float s = ((e[0] + e[1]) + (e[2] + e[3])) + ((e[4] + e[5]) + (e[6] + e[7]));
	const float inv_sum = RefRcpFast(RefWarpSum(s));
	RefU32x8 ov;
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; k++)
		ov.v[k] = __float_as_uint(e[k] * inv_sum);
	RefStg256((uint8_t *)probs + ((uint64_t)row * 32u + lane) * 32u,ov);
}

extern "C" __global__ __launch_bounds__(256) void mx_quantize_e4m3_ue8m0(
	const uint32_t *__restrict__ src_bf16,
	uint32_t *__restrict__ dst_fp8,
	uint8_t *__restrict__ dst_scales,
	uint32_t total_blocks)
{
	const uint32_t tid = blockIdx.x * 256u + threadIdx.x;
	const uint32_t blk = tid >> 2u;
	if ( blk >= total_blocks )
		return;
	const uint32_t sub = tid & 3u;
	const RefU32x4 v = RefLdg128Stream((const uint8_t *)src_bf16 + (uint64_t)tid * 16u);
	const float f0 = RefBf16Lo(v.v[0]);
	const float f1 = RefBf16Hi(v.v[0]);
	const float f2 = RefBf16Lo(v.v[1]);
	const float f3 = RefBf16Hi(v.v[1]);
	const float f4 = RefBf16Lo(v.v[2]);
	const float f5 = RefBf16Hi(v.v[2]);
	const float f6 = RefBf16Lo(v.v[3]);
	const float f7 = RefBf16Hi(v.v[3]);
	float lmax = fmaxf(fmaxf(fabsf(f0),fabsf(f1)),fmaxf(fabsf(f2),fabsf(f3)));
	lmax = fmaxf(lmax,fmaxf(fmaxf(fabsf(f4),fabsf(f5)),fmaxf(fabsf(f6),fabsf(f7))));
	const float amax = RefQuadMax(lmax);
	uint32_t e8m0 = 0u;
	if ( amax > 0.0f )
	{
		const uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
		e8m0 = (bits + 0x007fffffu) >> 23u;
		if ( e8m0 > 254u )
			e8m0 = 254u;
	}
	const float inv_scale = (e8m0 == 0u) ? 0.0f : __uint_as_float((254u - e8m0) << 23u);
	const uint16_t p0 = RefPackE4m3x2(f0 * inv_scale,f1 * inv_scale);
	const uint16_t p1 = RefPackE4m3x2(f2 * inv_scale,f3 * inv_scale);
	const uint16_t p2 = RefPackE4m3x2(f4 * inv_scale,f5 * inv_scale);
	const uint16_t p3 = RefPackE4m3x2(f6 * inv_scale,f7 * inv_scale);
	uint2 out64;
	out64.x = (uint32_t)p0 | ((uint32_t)p1 << 16u);
	out64.y = (uint32_t)p2 | ((uint32_t)p3 << 16u);
	((uint2 *)dst_fp8)[tid] = out64;
	if ( sub == 0u )
		dst_scales[blk] = (uint8_t)e8m0;
}

extern "C" __global__ __launch_bounds__(256) void tp_f32_to_bf16_vec16(
	const uint32_t *__restrict__ src_f32,
	uint32_t *__restrict__ dst_bf16,
	uint32_t chunks16)
{
	const uint32_t chunks8 = chunks16 << 1u;
	const uint32_t i0 = (blockIdx.x << 9u) + threadIdx.x;
	if ( i0 >= chunks8 )
		return;
	const RefU32x8 a0 = RefLdg256Stream((const uint8_t *)src_f32 + (uint64_t)i0 * 32u);
	RefU32x4 o0;
	#pragma unroll
	for (uint32_t k = 0u; k < 4u; k++)
		o0.v[k] = RefPackBf16x2(__uint_as_float(a0.v[k * 2u]),__uint_as_float(a0.v[k * 2u + 1u]));
	RefStg128((uint8_t *)dst_bf16 + (uint64_t)i0 * 16u,o0);
	const uint32_t i1 = i0 + 256u;
	if ( i1 >= chunks8 )
		return;
	const RefU32x8 a1 = RefLdg256Stream((const uint8_t *)src_f32 + (uint64_t)i1 * 32u);
	RefU32x4 o1;
	#pragma unroll
	for (uint32_t k = 0u; k < 4u; k++)
		o1.v[k] = RefPackBf16x2(__uint_as_float(a1.v[k * 2u]),__uint_as_float(a1.v[k * 2u + 1u]));
	RefStg128((uint8_t *)dst_bf16 + (uint64_t)i1 * 16u,o1);
}

extern "C" __global__ __launch_bounds__(256) void tp_bf16_to_f32_vec16(
	const uint32_t *__restrict__ src_bf16,
	uint32_t *__restrict__ dst_f32,
	uint32_t chunks16)
{
	const uint32_t idx = blockIdx.x * 256u + threadIdx.x;
	if ( idx >= chunks16 )
		return;
	const RefU32x8 v = RefLdg256Stream((const uint8_t *)src_bf16 + (uint64_t)idx * 32u);
	RefU32x8 o0,o1;
	#pragma unroll
	for (uint32_t k = 0u; k < 4u; k++)
	{
		o0.v[k * 2u] = __float_as_uint(RefBf16Lo(v.v[k]));
		o0.v[k * 2u + 1u] = __float_as_uint(RefBf16Hi(v.v[k]));
		o1.v[k * 2u] = __float_as_uint(RefBf16Lo(v.v[4u + k]));
		o1.v[k * 2u + 1u] = __float_as_uint(RefBf16Hi(v.v[4u + k]));
	}
	RefStg256((uint8_t *)dst_f32 + (uint64_t)(idx * 2u) * 32u,o0);
	RefStg256((uint8_t *)dst_f32 + (uint64_t)(idx * 2u + 1u) * 32u,o1);
}

extern "C" __global__ __launch_bounds__(256) void device_program_copy_vec256(
	uint32_t *__restrict__ dst,
	const uint32_t *__restrict__ src,
	uint32_t chunks32)
{
	const uint32_t idx = blockIdx.x * 256u + threadIdx.x;
	if ( idx >= chunks32 )
		return;
	const RefU32x8 v = RefLdg256Stream((const uint8_t *)src + (uint64_t)idx * 32u);
	RefStg256((uint8_t *)dst + (uint64_t)idx * 32u,v);
}

#if defined(SKLC_NVRM_REF_MAIN)
#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_error_site.h"
#include "tests/nvrm/sklc_nvrm_bench_common.h"

#define REF_TRY(call) \
	do { \
		const SparkStatus ref_status = (call); \
		if ( ref_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(ref_status); \
	} while (0)

static SparkStatus RefUpload(SparkDeviceBuffer *dev, const void *src, uint64_t bytes, const char *owner)
{
	SparkDeviceBuffer host;
	SparkStatus status;
	SparkDeviceRange dst_r,src_r;
	REF_TRY(SparkDeviceAllocate(dev,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,owner));
	status = SparkDeviceAllocate(&host,SPARK_DEVICE_SPACE_HOST_PAGEABLE,bytes,owner);
	if ( status != SPARK_STATUS_OK )
	{
		SparkDeviceFree(dev);
		SPARK_FAIL(status);
	}
	memcpy(host.host,src,(size_t)bytes);
	dst_r.buffer = dev;
	dst_r.offset = 0u;
	src_r.buffer = &host;
	src_r.offset = 0u;
	status = SparkDeviceCopySync(dst_r,src_r,bytes);
	SparkDeviceFree(&host);
	if ( status != SPARK_STATUS_OK )
	{
		SparkDeviceFree(dev);
		SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus RefDownload(void *dst, const SparkDeviceBuffer *dev, uint64_t bytes, const char *owner)
{
	SparkDeviceBuffer host;
	SparkStatus status;
	SparkDeviceRange dst_r,src_r;
	REF_TRY(SparkDeviceAllocate(&host,SPARK_DEVICE_SPACE_HOST_PAGEABLE,bytes,owner));
	dst_r.buffer = &host;
	dst_r.offset = 0u;
	src_r.buffer = dev;
	src_r.offset = 0u;
	status = SparkDeviceCopySync(dst_r,src_r,bytes);
	if ( status == SPARK_STATUS_OK )
		memcpy(dst,host.host,(size_t)bytes);
	SparkDeviceFree(&host);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

template<typename LaunchFn>
static SparkStatus RefTimeKernel(SparkDeviceQueue *queue, LaunchFn launch_fn, uint64_t traffic_bytes, BenchMetrics *metrics)
{
	SparkDeviceEvent *ev_start = 0,*ev_stop = 0;
	double batch_us[BENCH_BATCH_TRIALS],lat_us[BENCH_LATENCY_ROUNDS];
	uint32_t i,t;
	REF_TRY(SparkDeviceEventCreate(&ev_start));
	REF_TRY(SparkDeviceEventCreate(&ev_stop));
	for (i = 0u; i < BENCH_WARMUP_ROUNDS; i++)
		launch_fn();
	REF_TRY(SparkDeviceQueueWait(queue,BENCH_TIMEOUT_NS));
	for (t = 0u; t < BENCH_BATCH_TRIALS; t++)
	{
		uint64_t ns = 0u;
		REF_TRY(SparkDeviceEventRecord(ev_start,queue));
		for (i = 0u; i < BENCH_BATCH_SIZE; i++)
			launch_fn();
		REF_TRY(SparkDeviceEventRecord(ev_stop,queue));
		REF_TRY(SparkDeviceEventWait(ev_stop,BENCH_TIMEOUT_NS));
		REF_TRY(SparkDeviceEventElapsed(ev_start,ev_stop,&ns));
		batch_us[t] = (double)ns / (1000.0 * (double)BENCH_BATCH_SIZE);
	}
	for (i = 0u; i < BENCH_LATENCY_ROUNDS; i++)
	{
		const uint64_t t0 = SparkMonotonicNs();
		launch_fn();
		REF_TRY(SparkDeviceQueueWait(queue,BENCH_TIMEOUT_NS));
		lat_us[i] = (double)(SparkMonotonicNs() - t0) / 1000.0;
	}
	SparkDeviceEventDestroy(ev_stop);
	SparkDeviceEventDestroy(ev_start);
	metrics->gpu_us = BenchMedian(batch_us,BENCH_BATCH_TRIALS);
	metrics->lat_us = BenchMedian(lat_us,BENCH_LATENCY_ROUNDS);
	metrics->traffic_bytes = traffic_bytes;
	metrics->gbps = ((double)traffic_bytes / (metrics->gpu_us * 1e-6)) / 1e9;
	return(SPARK_STATUS_OK);
}

static int RefWant(const char *only_kernel, const char *name)
{
	return(only_kernel == 0 || strcmp(only_kernel,name) == 0);
}

static int RefFinishRow(const char *label, const BenchMetrics *m)
{
	BenchPrintRow(label,m);
	fflush(stdout);
	if ( !BenchCheckParity(m) )
	{
		printf("FAIL numerical parity for %s (max_err_fp32=%.6f max_ulp_fp32=%u)\n",m->name,m->max_abs_err,m->max_ulp_err);
		return(1);
	}
	return(0);
}

int main(int argc, char **argv)
{
	const char *label = "nvcc+cuda",*dump_dir = 0,*only_kernel = 0;
	SparkDeviceConfiguration cfg = {0u, 0u};
	cudaStream_t stream = 0;
	SparkDeviceQueue *queue = 0;
	BenchMetrics row;
	uint32_t ran = 0u;
	int i,failed = 0;
	for (i = 1; i < argc; i++)
	{
		if ( strcmp(argv[i],"--label") == 0 && i + 1 < argc )
			label = argv[++i];
		else if ( strcmp(argv[i],"--dump") == 0 && i + 1 < argc )
			dump_dir = argv[++i];
		else if ( strcmp(argv[i],"--kernel") == 0 && i + 1 < argc )
			only_kernel = argv[++i];
		else
		{
			printf("usage: %s [--label NAME] [--dump DIR] [--kernel NAME]\n",argv[0]);
			return(2);
		}
	}
	if ( SparkDeviceInitialize(&cfg) != SPARK_STATUS_OK || cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) != cudaSuccess ||
		SparkDeviceQueueAdopt(&queue,(void *)stream,"cuda_bench") != SPARK_STATUS_OK )
	{
		const SparkDeviceError *e = SparkDeviceLastError();
		printf("FAIL setup: %s [%s owner=%s code=%d]\n",e->text,e->operation,e->owner,e->backend_code);
		return(1);
	}
	{
		const SparkDeviceProperties *p = SparkDeviceGetProperties();
		printf("INFO label=%s backend=%s target=%s device=\"%s\" cus=%u\n",label,p->backend,p->target,p->name,p->compute_units);
	}
	if ( failed == 0 && RefWant(only_kernel,"rmsnorm_bf16_vec16") )
	{
		const uint64_t x_bytes = (uint64_t)BENCH_RMS_ROWS * BENCH_RMS_DIM * 2u;
		const uint64_t w_bytes = (uint64_t)BENCH_RMS_DIM * 2u;
		uint16_t *hx = (uint16_t *)malloc((size_t)x_bytes);
		uint16_t *hw = (uint16_t *)malloc((size_t)w_bytes);
		uint16_t *ho = (uint16_t *)calloc(1u,(size_t)x_bytes);
		SparkDeviceBuffer dx,dw,dout;
		memset(&row,0,sizeof(row));
		row.name = "rmsnorm_bf16_vec16";
		BenchFillRmsInputs(hx,hw,BENCH_RMS_ROWS,BENCH_RMS_DIM);
		if ( RefUpload(&dx,hx,x_bytes,"rms_x") != SPARK_STATUS_OK || RefUpload(&dw,hw,w_bytes,"rms_w") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,x_bytes,"rms_o") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			auto fn = [&]() {
				rmsnorm_bf16_vec16<<<BENCH_RMS_ROWS,256,0,stream>>>(
					(const uint32_t *)(uintptr_t)dx.device,(const uint32_t *)(uintptr_t)dw.device,
					(uint32_t *)(uintptr_t)dout.device,BENCH_RMS_ROWS,BENCH_RMS_CHUNKS16,BENCH_RMS_EPS,1.0f / (float)BENCH_RMS_DIM);
			};
			if ( RefTimeKernel(queue,fn,x_bytes + w_bytes + x_bytes,&row) != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,x_bytes,"rms_o") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifyRms(hx,hw,ho,BENCH_RMS_ROWS,BENCH_RMS_DIM,BENCH_RMS_EPS,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBf16(dump_dir,row.name,ho,(size_t)BENCH_RMS_ROWS * BENCH_RMS_DIM);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&dout);
			SparkDeviceFree(&dw);
			SparkDeviceFree(&dx);
		}
		free(ho);
		free(hw);
		free(hx);
	}
	if ( failed == 0 && RefWant(only_kernel,"fused_residual_rmsnorm_bf16x8") )
	{
		const uint64_t x_bytes = (uint64_t)BENCH_RMS_ROWS * BENCH_RMS_DIM * 2u;
		const uint64_t w_bytes = (uint64_t)BENCH_RMS_DIM * 2u;
		uint16_t *hx = (uint16_t *)malloc((size_t)x_bytes);
		uint16_t *hr = (uint16_t *)malloc((size_t)x_bytes);
		uint16_t *hw = (uint16_t *)malloc((size_t)w_bytes);
		uint16_t *hro = (uint16_t *)calloc(1u,(size_t)x_bytes);
		uint16_t *ho = (uint16_t *)calloc(1u,(size_t)x_bytes);
		SparkDeviceBuffer dx,dr,dw,dout;
		memset(&row,0,sizeof(row));
		row.name = "fused_residual_rmsnorm_bf16x8";
		BenchFillFusedRmsInputs(hx,hr,hw,BENCH_RMS_ROWS,BENCH_RMS_DIM);
		if ( RefUpload(&dx,hx,x_bytes,"frms_x") != SPARK_STATUS_OK || RefUpload(&dr,hr,x_bytes,"frms_r") != SPARK_STATUS_OK ||
			RefUpload(&dw,hw,w_bytes,"frms_w") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,x_bytes,"frms_o") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			auto fn = [&]() {
				fused_residual_rmsnorm_bf16x8<<<BENCH_RMS_ROWS,256,0,stream>>>(
					(const uint32_t *)(uintptr_t)dx.device,(uint32_t *)(uintptr_t)dr.device,
					(const uint32_t *)(uintptr_t)dw.device,(uint32_t *)(uintptr_t)dout.device,
					BENCH_RMS_ROWS,BENCH_RMS_CHUNKS16,BENCH_RMS_EPS,1.0f / (float)BENCH_RMS_DIM);
			};
			fn();
			if ( SparkDeviceQueueWait(queue,BENCH_TIMEOUT_NS) != SPARK_STATUS_OK ||
				RefDownload(hro,&dr,x_bytes,"frms_ro") != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,x_bytes,"frms_o") != SPARK_STATUS_OK ||
				RefTimeKernel(queue,fn,x_bytes * 4u + w_bytes,&row) != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifyFusedRms(hx,hr,hw,hro,ho,BENCH_RMS_ROWS,BENCH_RMS_DIM,BENCH_RMS_EPS,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBf16(dump_dir,row.name,ho,(size_t)BENCH_RMS_ROWS * BENCH_RMS_DIM);
				BenchDumpBf16(dump_dir,"fused_residual_rmsnorm_bf16x8_res",hro,(size_t)BENCH_RMS_ROWS * BENCH_RMS_DIM);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&dout);
			SparkDeviceFree(&dw);
			SparkDeviceFree(&dr);
			SparkDeviceFree(&dx);
		}
		free(ho);
		free(hro);
		free(hw);
		free(hr);
		free(hx);
	}
	if ( failed == 0 && RefWant(only_kernel,"swiglu_packed_bf16x8") )
	{
		const uint64_t bytes = (uint64_t)BENCH_SWIGLU_ELEMS * 2u;
		uint16_t *hg = (uint16_t *)malloc((size_t)bytes);
		uint16_t *hu = (uint16_t *)malloc((size_t)bytes);
		uint16_t *ho = (uint16_t *)calloc(1u,(size_t)bytes);
		SparkDeviceBuffer dg,du,dout;
		memset(&row,0,sizeof(row));
		row.name = "swiglu_packed_bf16x8";
		BenchFillSwigluInputs(hg,hu,BENCH_SWIGLU_ELEMS);
		if ( RefUpload(&dg,hg,bytes,"swi_g") != SPARK_STATUS_OK || RefUpload(&du,hu,bytes,"swi_u") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,"swi_o") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			const uint32_t grid = (BENCH_SWIGLU_CHUNKS16 + 255u) / 256u;
			auto fn = [&]() {
				swiglu_packed_bf16x8<<<grid,256,0,stream>>>(
					(const uint32_t *)(uintptr_t)dg.device,(const uint32_t *)(uintptr_t)du.device,
					(uint32_t *)(uintptr_t)dout.device,BENCH_SWIGLU_CHUNKS16);
			};
			if ( RefTimeKernel(queue,fn,bytes * 3u,&row) != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,bytes,"swi_o") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifySwiglu(hg,hu,ho,BENCH_SWIGLU_ELEMS,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBf16(dump_dir,row.name,ho,(size_t)BENCH_SWIGLU_ELEMS);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&dout);
			SparkDeviceFree(&du);
			SparkDeviceFree(&dg);
		}
		free(ho);
		free(hu);
		free(hg);
	}
	if ( failed == 0 && RefWant(only_kernel,"skinny_bf16_b1") )
	{
		const uint64_t w_bytes = (uint64_t)BENCH_SKINNY_BF16_OUT * BENCH_SKINNY_BF16_K * 2u;
		const uint64_t a_bytes = (uint64_t)BENCH_SKINNY_BF16_K * 2u;
		const uint64_t o_bytes = (uint64_t)BENCH_SKINNY_BF16_OUT * 2u;
		uint16_t *hw = (uint16_t *)malloc((size_t)w_bytes);
		uint16_t *ha = (uint16_t *)malloc((size_t)a_bytes);
		uint16_t *ho = (uint16_t *)calloc(1u,(size_t)o_bytes);
		SparkDeviceBuffer dw,da,dout;
		memset(&row,0,sizeof(row));
		row.name = "skinny_bf16_b1";
		BenchFillSkinnyBf16Inputs(hw,ha,BENCH_SKINNY_BF16_OUT,BENCH_SKINNY_BF16_K);
		if ( RefUpload(&dw,hw,w_bytes,"sk_bf16_w") != SPARK_STATUS_OK || RefUpload(&da,ha,a_bytes,"sk_bf16_a") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,o_bytes,"sk_bf16_o") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			const uint32_t grid = (BENCH_SKINNY_BF16_OUT + 7u) / 8u;
			auto fn = [&]() {
				skinny_bf16_b1<<<grid,128,0,stream>>>(
					(const uint32_t *)(uintptr_t)dw.device,(const uint32_t *)(uintptr_t)da.device,
					(uint16_t *)(uintptr_t)dout.device,BENCH_SKINNY_BF16_CHUNKS16,BENCH_SKINNY_BF16_OUT,0u);
			};
			if ( RefTimeKernel(queue,fn,w_bytes + a_bytes + o_bytes,&row) != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,o_bytes,"sk_bf16_o") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifySkinnyBf16(hw,ha,ho,BENCH_SKINNY_BF16_OUT,BENCH_SKINNY_BF16_K,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBf16(dump_dir,row.name,ho,(size_t)BENCH_SKINNY_BF16_OUT);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&dout);
			SparkDeviceFree(&da);
			SparkDeviceFree(&dw);
		}
		free(ho);
		free(ha);
		free(hw);
	}
	if ( failed == 0 && RefWant(only_kernel,"skinny_nvfp4_b1") )
	{
		const uint64_t w_bytes = (uint64_t)BENCH_SKINNY_NVFP4_OUT * BENCH_SKINNY_NVFP4_CHUNKS32 * 32u;
		const uint64_t s_bytes = (uint64_t)BENCH_SKINNY_NVFP4_OUT * BENCH_SKINNY_NVFP4_CHUNKS32 * 4u;
		const uint64_t a_bytes = (uint64_t)BENCH_SKINNY_NVFP4_K * 2u;
		const uint64_t o_bytes = (uint64_t)BENCH_SKINNY_NVFP4_OUT * 2u;
		uint8_t *hw = (uint8_t *)malloc((size_t)w_bytes);
		uint8_t *hs = (uint8_t *)malloc((size_t)s_bytes);
		uint16_t *ha = (uint16_t *)malloc((size_t)a_bytes);
		uint16_t *ho = (uint16_t *)calloc(1u,(size_t)o_bytes);
		float hg = 0.0f;
		SparkDeviceBuffer dw,ds,da,dg,dout;
		memset(&row,0,sizeof(row));
		row.name = "skinny_nvfp4_b1";
		BenchFillSkinnyNvfp4Inputs(hw,hs,ha,&hg,BENCH_SKINNY_NVFP4_OUT,BENCH_SKINNY_NVFP4_K);
		if ( RefUpload(&dw,hw,w_bytes,"sk_fp4_w") != SPARK_STATUS_OK || RefUpload(&ds,hs,s_bytes,"sk_fp4_s") != SPARK_STATUS_OK ||
			RefUpload(&da,ha,a_bytes,"sk_fp4_a") != SPARK_STATUS_OK || RefUpload(&dg,&hg,sizeof(hg),"sk_fp4_g") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,o_bytes,"sk_fp4_o") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			const uint32_t grid = (BENCH_SKINNY_NVFP4_OUT + 7u) / 8u;
			auto fn = [&]() {
				skinny_nvfp4_b1<<<grid,128,0,stream>>>(
					(const uint32_t *)(uintptr_t)dw.device,(const uint32_t *)(uintptr_t)da.device,
					(const uint32_t *)(uintptr_t)ds.device,(const float *)(uintptr_t)dg.device,
					(uint16_t *)(uintptr_t)dout.device,BENCH_SKINNY_NVFP4_CHUNKS32,BENCH_SKINNY_NVFP4_OUT,0u);
			};
			if ( RefTimeKernel(queue,fn,w_bytes + s_bytes + a_bytes + o_bytes,&row) != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,o_bytes,"sk_fp4_o") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifySkinnyNvfp4(hw,hs,ha,hg,ho,BENCH_SKINNY_NVFP4_OUT,BENCH_SKINNY_NVFP4_K,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBf16(dump_dir,row.name,ho,(size_t)BENCH_SKINNY_NVFP4_OUT);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&dout);
			SparkDeviceFree(&dg);
			SparkDeviceFree(&da);
			SparkDeviceFree(&ds);
			SparkDeviceFree(&dw);
		}
		free(ho);
		free(ha);
		free(hs);
		free(hw);
	}
	if ( failed == 0 && RefWant(only_kernel,"skinny_mxfp4_b1") )
	{
		const uint64_t w_bytes = (uint64_t)BENCH_SKINNY_MXFP4_OUT * BENCH_SKINNY_MXFP4_CHUNKS32 * 32u;
		const uint64_t s_bytes = (uint64_t)BENCH_SKINNY_MXFP4_OUT * BENCH_SKINNY_MXFP4_CHUNKS32 * 2u;
		const uint64_t a_bytes = (uint64_t)BENCH_SKINNY_MXFP4_K * 2u;
		const uint64_t o_bytes = (uint64_t)BENCH_SKINNY_MXFP4_OUT * 2u;
		uint8_t *hw = (uint8_t *)malloc((size_t)w_bytes);
		uint8_t *hs = (uint8_t *)malloc((size_t)s_bytes);
		uint16_t *ha = (uint16_t *)malloc((size_t)a_bytes);
		uint16_t *ho = (uint16_t *)calloc(1u,(size_t)o_bytes);
		SparkDeviceBuffer dw,ds,da,dout;
		memset(&row,0,sizeof(row));
		row.name = "skinny_mxfp4_b1";
		BenchFillSkinnyMxfp4Inputs(hw,hs,ha,BENCH_SKINNY_MXFP4_OUT,BENCH_SKINNY_MXFP4_K);
		if ( RefUpload(&dw,hw,w_bytes,"sk_mx4_w") != SPARK_STATUS_OK || RefUpload(&ds,hs,s_bytes,"sk_mx4_s") != SPARK_STATUS_OK ||
			RefUpload(&da,ha,a_bytes,"sk_mx4_a") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,o_bytes,"sk_mx4_o") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			const uint32_t grid = (BENCH_SKINNY_MXFP4_OUT + 7u) / 8u;
			auto fn = [&]() {
				skinny_mxfp4_b1<<<grid,128,0,stream>>>(
					(const uint32_t *)(uintptr_t)dw.device,(const uint32_t *)(uintptr_t)da.device,
					(const uint16_t *)(uintptr_t)ds.device,(uint16_t *)(uintptr_t)dout.device,
					BENCH_SKINNY_MXFP4_CHUNKS32,BENCH_SKINNY_MXFP4_OUT,0u);
			};
			if ( RefTimeKernel(queue,fn,w_bytes + s_bytes + a_bytes + o_bytes,&row) != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,o_bytes,"sk_mx4_o") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifySkinnyMxfp4(hw,hs,ha,ho,BENCH_SKINNY_MXFP4_OUT,BENCH_SKINNY_MXFP4_K,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBf16(dump_dir,row.name,ho,(size_t)BENCH_SKINNY_MXFP4_OUT);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&dout);
			SparkDeviceFree(&da);
			SparkDeviceFree(&ds);
			SparkDeviceFree(&dw);
		}
		free(ho);
		free(ha);
		free(hs);
		free(hw);
	}
	if ( failed == 0 && RefWant(only_kernel,"latent_attn_decode_512_64") )
	{
		const uint64_t qn_bytes = (uint64_t)BENCH_ATTN_HEADS * BENCH_ATTN_NOPE * 2u;
		const uint64_t qr_bytes = (uint64_t)BENCH_ATTN_HEADS * BENCH_ATTN_ROPE * 2u;
		const uint64_t kn_bytes = (uint64_t)BENCH_ATTN_KV_LEN * BENCH_ATTN_NOPE * 2u;
		const uint64_t kr_bytes = (uint64_t)BENCH_ATTN_KV_LEN * BENCH_ATTN_ROPE * 2u;
		const uint64_t o_bytes = qn_bytes;
		const uint64_t traffic = (uint64_t)BENCH_ATTN_HEADS * (kn_bytes + kr_bytes) + qn_bytes + qr_bytes + o_bytes;
		uint16_t *hqn = (uint16_t *)malloc((size_t)qn_bytes);
		uint16_t *hqr = (uint16_t *)malloc((size_t)qr_bytes);
		uint16_t *hkn = (uint16_t *)malloc((size_t)kn_bytes);
		uint16_t *hkr = (uint16_t *)malloc((size_t)kr_bytes);
		uint16_t *ho = (uint16_t *)calloc(1u,(size_t)o_bytes);
		SparkDeviceBuffer dqn,dqr,dkn,dkr,dout;
		memset(&row,0,sizeof(row));
		row.name = "latent_attn_decode_512_64";
		BenchFillAttnInputs(hqn,hqr,hkn,hkr,BENCH_ATTN_HEADS,BENCH_ATTN_KV_LEN);
		if ( RefUpload(&dqn,hqn,qn_bytes,"attn_qn") != SPARK_STATUS_OK || RefUpload(&dqr,hqr,qr_bytes,"attn_qr") != SPARK_STATUS_OK ||
			RefUpload(&dkn,hkn,kn_bytes,"attn_kn") != SPARK_STATUS_OK || RefUpload(&dkr,hkr,kr_bytes,"attn_kr") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,o_bytes,"attn_o") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			auto fn = [&]() {
				latent_attn_decode_512_64<<<BENCH_ATTN_HEADS,256,0,stream>>>(
					(const uint32_t *)(uintptr_t)dqn.device,(const uint32_t *)(uintptr_t)dqr.device,
					(const uint32_t *)(uintptr_t)dkn.device,(const uint32_t *)(uintptr_t)dkr.device,
					(uint16_t *)(uintptr_t)dout.device,BENCH_ATTN_HEADS,BENCH_ATTN_KV_LEN,BENCH_ATTN_SCALE);
			};
			if ( RefTimeKernel(queue,fn,traffic,&row) != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,o_bytes,"attn_o") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifyAttn(hqn,hqr,hkn,hkr,ho,BENCH_ATTN_HEADS,BENCH_ATTN_KV_LEN,BENCH_ATTN_SCALE,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBf16(dump_dir,row.name,ho,(size_t)BENCH_ATTN_HEADS * BENCH_ATTN_NOPE);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&dout);
			SparkDeviceFree(&dkr);
			SparkDeviceFree(&dkn);
			SparkDeviceFree(&dqr);
			SparkDeviceFree(&dqn);
		}
		free(ho);
		free(hkr);
		free(hkn);
		free(hqr);
		free(hqn);
	}
	if ( failed == 0 && RefWant(only_kernel,"router_softmax_256") )
	{
		const uint64_t count = (uint64_t)BENCH_ROUTER_ROWS * BENCH_ROUTER_COLS;
		const uint64_t bytes = count * 4u;
		float *hin = (float *)malloc((size_t)bytes);
		float *ho = (float *)calloc(1u,(size_t)bytes);
		SparkDeviceBuffer din,dout;
		memset(&row,0,sizeof(row));
		row.name = "router_softmax_256";
		row.abs_limit_fp32 = 1e-5;
		BenchFillRouterF32Inputs(hin,BENCH_ROUTER_ROWS,BENCH_ROUTER_COLS);
		if ( RefUpload(&din,hin,bytes,"rtr_in") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,"rtr_out") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			const uint32_t grid = (BENCH_ROUTER_ROWS + 7u) / 8u;
			auto fn = [&]() {
				router_softmax_256<<<grid,256,0,stream>>>(
					(const float *)(uintptr_t)din.device,(float *)(uintptr_t)dout.device,BENCH_ROUTER_ROWS);
			};
			if ( RefTimeKernel(queue,fn,bytes * 2u,&row) != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,bytes,"rtr_out") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifyRouterF32(hin,ho,BENCH_ROUTER_ROWS,BENCH_ROUTER_COLS,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBytes(dump_dir,row.name,ho,(size_t)bytes);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&dout);
			SparkDeviceFree(&din);
		}
		free(ho);
		free(hin);
	}
	if ( failed == 0 && RefWant(only_kernel,"router_softmax_bf16_256") )
	{
		const uint64_t count = (uint64_t)BENCH_ROUTER_ROWS * BENCH_ROUTER_COLS;
		const uint64_t in_bytes = count * 2u;
		const uint64_t out_bytes = count * 4u;
		uint16_t *hin = (uint16_t *)malloc((size_t)in_bytes);
		float *ho = (float *)calloc(1u,(size_t)out_bytes);
		SparkDeviceBuffer din,dout;
		memset(&row,0,sizeof(row));
		row.name = "router_softmax_bf16_256";
		row.abs_limit_fp32 = 1e-5;
		BenchFillRouterBf16Inputs(hin,BENCH_ROUTER_ROWS,BENCH_ROUTER_COLS);
		if ( RefUpload(&din,hin,in_bytes,"rtr16_in") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,out_bytes,"rtr16_out") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			const uint32_t grid = (BENCH_ROUTER_ROWS + 7u) / 8u;
			auto fn = [&]() {
				router_softmax_bf16_256<<<grid,256,0,stream>>>(
					(const uint32_t *)(uintptr_t)din.device,(float *)(uintptr_t)dout.device,BENCH_ROUTER_ROWS);
			};
			if ( RefTimeKernel(queue,fn,in_bytes + out_bytes,&row) != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,out_bytes,"rtr16_out") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifyRouterBf16(hin,ho,BENCH_ROUTER_ROWS,BENCH_ROUTER_COLS,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBytes(dump_dir,row.name,ho,(size_t)out_bytes);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&dout);
			SparkDeviceFree(&din);
		}
		free(ho);
		free(hin);
	}
	if ( failed == 0 && RefWant(only_kernel,"mx_quantize_e4m3_ue8m0") )
	{
		const uint64_t src_bytes = (uint64_t)BENCH_MX_ELEMS * 2u;
		const uint64_t fp8_bytes = (uint64_t)BENCH_MX_ELEMS;
		const uint64_t sc_bytes = (uint64_t)BENCH_MX_BLOCKS;
		uint16_t *hsrc = (uint16_t *)malloc((size_t)src_bytes);
		uint8_t *hfp8 = (uint8_t *)calloc(1u,(size_t)fp8_bytes);
		uint8_t *hsc = (uint8_t *)calloc(1u,(size_t)sc_bytes);
		SparkDeviceBuffer dsrc,dfp8,dsc;
		memset(&row,0,sizeof(row));
		row.name = "mx_quantize_e4m3_ue8m0";
		BenchFillMxQuantInputs(hsrc,BENCH_MX_ELEMS);
		if ( RefUpload(&dsrc,hsrc,src_bytes,"mx_src") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&dfp8,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,fp8_bytes,"mx_fp8") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&dsc,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,sc_bytes,"mx_sc") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			const uint32_t grid = (BENCH_MX_BLOCKS * 4u + 255u) / 256u;
			auto fn = [&]() {
				mx_quantize_e4m3_ue8m0<<<grid,256,0,stream>>>(
					(const uint32_t *)(uintptr_t)dsrc.device,(uint32_t *)(uintptr_t)dfp8.device,
					(uint8_t *)(uintptr_t)dsc.device,BENCH_MX_BLOCKS);
			};
			if ( RefTimeKernel(queue,fn,src_bytes + fp8_bytes + sc_bytes,&row) != SPARK_STATUS_OK ||
				RefDownload(hfp8,&dfp8,fp8_bytes,"mx_fp8") != SPARK_STATUS_OK ||
				RefDownload(hsc,&dsc,sc_bytes,"mx_sc") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifyMxQuant(hsrc,hfp8,hsc,BENCH_MX_BLOCKS,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBytes(dump_dir,row.name,hfp8,(size_t)fp8_bytes);
				BenchDumpBytes(dump_dir,"mx_quantize_e4m3_ue8m0_scales",hsc,(size_t)sc_bytes);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&dsc);
			SparkDeviceFree(&dfp8);
			SparkDeviceFree(&dsrc);
		}
		free(hsc);
		free(hfp8);
		free(hsrc);
	}
	if ( failed == 0 && RefWant(only_kernel,"tp_f32_to_bf16_vec16") )
	{
		const uint64_t src_bytes = (uint64_t)BENCH_TP_ELEMS * 4u;
		const uint64_t dst_bytes = (uint64_t)BENCH_TP_ELEMS * 2u;
		float *hsrc = (float *)malloc((size_t)src_bytes);
		uint16_t *hdst = (uint16_t *)calloc(1u,(size_t)dst_bytes);
		SparkDeviceBuffer dsrc,ddst;
		memset(&row,0,sizeof(row));
		row.name = "tp_f32_to_bf16_vec16";
		BenchFillTpF32Inputs(hsrc,BENCH_TP_ELEMS);
		if ( RefUpload(&dsrc,hsrc,src_bytes,"tp_f32_src") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&ddst,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,dst_bytes,"tp_bf16_dst") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			const uint32_t grid = (BENCH_TP_CHUNKS16 + 255u) / 256u;
			auto fn = [&]() {
				tp_f32_to_bf16_vec16<<<grid,256,0,stream>>>(
					(const uint32_t *)(uintptr_t)dsrc.device,(uint32_t *)(uintptr_t)ddst.device,BENCH_TP_CHUNKS16);
			};
			if ( RefTimeKernel(queue,fn,src_bytes + dst_bytes,&row) != SPARK_STATUS_OK ||
				RefDownload(hdst,&ddst,dst_bytes,"tp_bf16_dst") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifyTpF32ToBf16(hsrc,hdst,BENCH_TP_ELEMS,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBf16(dump_dir,row.name,hdst,(size_t)BENCH_TP_ELEMS);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&ddst);
			SparkDeviceFree(&dsrc);
		}
		free(hdst);
		free(hsrc);
	}
	if ( failed == 0 && RefWant(only_kernel,"tp_bf16_to_f32_vec16") )
	{
		const uint64_t src_bytes = (uint64_t)BENCH_TP_ELEMS * 2u;
		const uint64_t dst_bytes = (uint64_t)BENCH_TP_ELEMS * 4u;
		uint16_t *hsrc = (uint16_t *)malloc((size_t)src_bytes);
		float *hdst = (float *)calloc(1u,(size_t)dst_bytes);
		SparkDeviceBuffer dsrc,ddst;
		memset(&row,0,sizeof(row));
		row.name = "tp_bf16_to_f32_vec16";
		BenchFillTpBf16Inputs(hsrc,BENCH_TP_ELEMS);
		if ( RefUpload(&dsrc,hsrc,src_bytes,"tp_bf16_src") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&ddst,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,dst_bytes,"tp_f32_dst") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			const uint32_t grid = (BENCH_TP_CHUNKS16 + 255u) / 256u;
			auto fn = [&]() {
				tp_bf16_to_f32_vec16<<<grid,256,0,stream>>>(
					(const uint32_t *)(uintptr_t)dsrc.device,(uint32_t *)(uintptr_t)ddst.device,BENCH_TP_CHUNKS16);
			};
			if ( RefTimeKernel(queue,fn,src_bytes + dst_bytes,&row) != SPARK_STATUS_OK ||
				RefDownload(hdst,&ddst,dst_bytes,"tp_f32_dst") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifyTpBf16ToF32(hsrc,hdst,BENCH_TP_ELEMS,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBytes(dump_dir,row.name,hdst,(size_t)dst_bytes);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&ddst);
			SparkDeviceFree(&dsrc);
		}
		free(hdst);
		free(hsrc);
	}
	if ( failed == 0 && RefWant(only_kernel,"device_program_copy_vec256") )
	{
		const uint64_t bytes = BENCH_COPY_BYTES;
		const uint64_t words = bytes / 4u;
		uint32_t *hsrc = (uint32_t *)malloc((size_t)bytes);
		uint32_t *hdst = (uint32_t *)calloc(1u,(size_t)bytes);
		SparkDeviceBuffer dsrc,ddst;
		memset(&row,0,sizeof(row));
		row.name = "device_program_copy_vec256";
		BenchFillCopyInputs(hsrc,words);
		if ( RefUpload(&dsrc,hsrc,bytes,"copy_src") != SPARK_STATUS_OK ||
			SparkDeviceAllocate(&ddst,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,"copy_dst") != SPARK_STATUS_OK )
			failed = 1;
		else
		{
			const uint32_t grid = (BENCH_COPY_CHUNKS32 + 255u) / 256u;
			auto fn = [&]() {
				device_program_copy_vec256<<<grid,256,0,stream>>>(
					(uint32_t *)(uintptr_t)ddst.device,(const uint32_t *)(uintptr_t)dsrc.device,BENCH_COPY_CHUNKS32);
			};
			if ( RefTimeKernel(queue,fn,bytes * 2u,&row) != SPARK_STATUS_OK ||
				RefDownload(hdst,&ddst,bytes,"copy_dst") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifyCopy(hsrc,hdst,words,&row.max_abs_err,&row.max_ulp_err);
				BenchDumpBytes(dump_dir,row.name,hdst,(size_t)bytes);
				failed |= RefFinishRow(label,&row);
				ran++;
			}
			SparkDeviceFree(&ddst);
			SparkDeviceFree(&dsrc);
		}
		free(hdst);
		free(hsrc);
	}
	if ( only_kernel != 0 && ran == 0u )
	{
		printf("FAIL unknown kernel filter: %s\n",only_kernel);
		failed = 1;
	}
	SparkDeviceQueueDestroy(queue);
	cudaStreamDestroy(stream);
	SparkDeviceShutdown();
	if ( failed == 0 )
		printf("PASS %s verified %u kernels\n",label,ran);
	return(failed);
}
#endif
