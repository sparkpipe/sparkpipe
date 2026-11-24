#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef struct __align__(32) RefU32x8
{
	uint32_t v[8];
} RefU32x8;

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

static __device__ __forceinline__ void RefStg256(void *ptr, const RefU32x8 &r)
{
	asm volatile(
		"st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
		:
		: "l"(ptr), "r"(r.v[0]), "r"(r.v[1]), "r"(r.v[2]), "r"(r.v[3]),
		  "r"(r.v[4]), "r"(r.v[5]), "r"(r.v[6]), "r"(r.v[7]));
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

static __device__ __forceinline__ float RefWarpSum(float v)
{
	v += __shfl_xor_sync(0xffffffffu,v,16);
	v += __shfl_xor_sync(0xffffffffu,v,8);
	v += __shfl_xor_sync(0xffffffffu,v,4);
	v += __shfl_xor_sync(0xffffffffu,v,2);
	v += __shfl_xor_sync(0xffffffffu,v,1);
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

int main(int argc, char **argv)
{
	const char *label = "nvcc+cuda",*dump_dir = 0;
	SparkDeviceConfiguration cfg = {0u, 0u};
	cudaStream_t stream = 0;
	SparkDeviceQueue *queue = 0;
	BenchMetrics rows[4];
	int i,failed = 0;
	for (i = 1; i < argc; i++)
	{
		if ( strcmp(argv[i],"--label") == 0 && i + 1 < argc )
			label = argv[++i];
		else if ( strcmp(argv[i],"--dump") == 0 && i + 1 < argc )
			dump_dir = argv[++i];
		else
		{
			printf("usage: %s [--label NAME] [--dump DIR]\n",argv[0]);
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
	{
		const uint64_t x_bytes = (uint64_t)BENCH_RMS_ROWS * BENCH_RMS_DIM * 2u;
		const uint64_t w_bytes = (uint64_t)BENCH_RMS_DIM * 2u;
		uint16_t *hx = (uint16_t *)malloc((size_t)x_bytes);
		uint16_t *hw = (uint16_t *)malloc((size_t)w_bytes);
		uint16_t *ho = (uint16_t *)calloc(1u,(size_t)x_bytes);
		SparkDeviceBuffer dx,dw,dout;
		memset(&rows[0],0,sizeof(rows[0]));
		rows[0].name = "rmsnorm_bf16_vec16";
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
			if ( RefTimeKernel(queue,fn,x_bytes + w_bytes + x_bytes,&rows[0]) != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,x_bytes,"rms_o") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifyRms(hx,hw,ho,BENCH_RMS_ROWS,BENCH_RMS_DIM,BENCH_RMS_EPS,&rows[0].max_abs_err,&rows[0].max_ulp_err);
				BenchDumpBf16(dump_dir,rows[0].name,ho,(size_t)BENCH_RMS_ROWS * BENCH_RMS_DIM);
			}
			SparkDeviceFree(&dout);
			SparkDeviceFree(&dw);
			SparkDeviceFree(&dx);
		}
		free(ho);
		free(hw);
		free(hx);
	}
	if ( failed == 0 )
	{
		const uint64_t bytes = (uint64_t)BENCH_SWIGLU_ELEMS * 2u;
		uint16_t *hg = (uint16_t *)malloc((size_t)bytes);
		uint16_t *hu = (uint16_t *)malloc((size_t)bytes);
		uint16_t *ho = (uint16_t *)calloc(1u,(size_t)bytes);
		SparkDeviceBuffer dg,du,dout;
		memset(&rows[1],0,sizeof(rows[1]));
		rows[1].name = "swiglu_packed_bf16x8";
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
			if ( RefTimeKernel(queue,fn,bytes * 3u,&rows[1]) != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,bytes,"swi_o") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifySwiglu(hg,hu,ho,BENCH_SWIGLU_ELEMS,&rows[1].max_abs_err,&rows[1].max_ulp_err);
				BenchDumpBf16(dump_dir,rows[1].name,ho,(size_t)BENCH_SWIGLU_ELEMS);
			}
			SparkDeviceFree(&dout);
			SparkDeviceFree(&du);
			SparkDeviceFree(&dg);
		}
		free(ho);
		free(hu);
		free(hg);
	}
	if ( failed == 0 )
	{
		const uint64_t w_bytes = (uint64_t)BENCH_SKINNY_BF16_OUT * BENCH_SKINNY_BF16_K * 2u;
		const uint64_t a_bytes = (uint64_t)BENCH_SKINNY_BF16_K * 2u;
		const uint64_t o_bytes = (uint64_t)BENCH_SKINNY_BF16_OUT * 2u;
		uint16_t *hw = (uint16_t *)malloc((size_t)w_bytes);
		uint16_t *ha = (uint16_t *)malloc((size_t)a_bytes);
		uint16_t *ho = (uint16_t *)calloc(1u,(size_t)o_bytes);
		SparkDeviceBuffer dw,da,dout;
		memset(&rows[2],0,sizeof(rows[2]));
		rows[2].name = "skinny_bf16_b1";
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
			if ( RefTimeKernel(queue,fn,w_bytes + a_bytes + o_bytes,&rows[2]) != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,o_bytes,"sk_bf16_o") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifySkinnyBf16(hw,ha,ho,BENCH_SKINNY_BF16_OUT,BENCH_SKINNY_BF16_K,&rows[2].max_abs_err,&rows[2].max_ulp_err);
				BenchDumpBf16(dump_dir,rows[2].name,ho,(size_t)BENCH_SKINNY_BF16_OUT);
			}
			SparkDeviceFree(&dout);
			SparkDeviceFree(&da);
			SparkDeviceFree(&dw);
		}
		free(ho);
		free(ha);
		free(hw);
	}
	if ( failed == 0 )
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
		memset(&rows[3],0,sizeof(rows[3]));
		rows[3].name = "skinny_nvfp4_b1";
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
			if ( RefTimeKernel(queue,fn,w_bytes + s_bytes + a_bytes + o_bytes,&rows[3]) != SPARK_STATUS_OK ||
				RefDownload(ho,&dout,o_bytes,"sk_fp4_o") != SPARK_STATUS_OK )
				failed = 1;
			else
			{
				BenchVerifySkinnyNvfp4(hw,hs,ha,hg,ho,BENCH_SKINNY_NVFP4_OUT,BENCH_SKINNY_NVFP4_K,&rows[3].max_abs_err,&rows[3].max_ulp_err);
				BenchDumpBf16(dump_dir,rows[3].name,ho,(size_t)BENCH_SKINNY_NVFP4_OUT);
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
	if ( failed == 0 )
	{
		for (i = 0; i < 4; i++)
		{
			BenchPrintRow(label,&rows[i]);
			if ( rows[i].max_ulp_err > 2u )
			{
				printf("FAIL numerical parity for %s (max_ulp_fp32=%u)\n",rows[i].name,rows[i].max_ulp_err);
				failed = 1;
			}
		}
	}
	SparkDeviceQueueDestroy(queue);
	cudaStreamDestroy(stream);
	SparkDeviceShutdown();
	if ( failed == 0 )
		printf("PASS %s verified 4 kernels\n",label);
	return(failed);
}
#endif
