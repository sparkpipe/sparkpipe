#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_error_site.h"
#include "tests/nvrm/sklc_nvrm_bench_common.h"

typedef struct RmsArgs
{
	uint64_t x;
	uint64_t w;
	uint64_t out;
	uint32_t rows;
	uint32_t chunks16;
	float eps;
	float inv_dim;
} RmsArgs;

typedef struct FusedRmsArgs
{
	uint64_t x;
	uint64_t res;
	uint64_t w;
	uint64_t out;
	uint32_t rows;
	uint32_t chunks16;
	float eps;
	float inv_dim;
} FusedRmsArgs;

typedef struct SwigluArgs
{
	uint64_t gate;
	uint64_t up;
	uint64_t out;
	uint32_t total_chunks16;
} SwigluArgs;

typedef struct SkinnyBf16Args
{
	uint64_t weight;
	uint64_t act;
	uint64_t out;
	uint32_t chunks16;
	uint32_t out_dim;
	uint32_t out_col_off;
} SkinnyBf16Args;

typedef struct SkinnyNvfp4Args
{
	uint64_t weight;
	uint64_t act;
	uint64_t scales;
	uint64_t gscale;
	uint64_t out;
	uint32_t chunks32;
	uint32_t out_dim;
	uint32_t out_col_off;
} SkinnyNvfp4Args;

typedef struct SkinnyMxfp4Args
{
	uint64_t weight;
	uint64_t act;
	uint64_t scales;
	uint64_t out;
	uint32_t chunks32;
	uint32_t out_dim;
	uint32_t out_col_off;
} SkinnyMxfp4Args;

typedef struct AttnArgs
{
	uint64_t q_nope;
	uint64_t q_rope;
	uint64_t kv_nope;
	uint64_t kv_rope;
	uint64_t out;
	uint32_t num_heads;
	uint32_t kv_len;
	float scale;
} AttnArgs;

typedef struct RouterArgs
{
	uint64_t logits;
	uint64_t probs;
	uint32_t rows;
} RouterArgs;

typedef struct MxQuantArgs
{
	uint64_t src_bf16;
	uint64_t dst_fp8;
	uint64_t dst_scales;
	uint32_t total_blocks;
} MxQuantArgs;

typedef struct TpConvArgs
{
	uint64_t src;
	uint64_t dst;
	uint32_t chunks16;
} TpConvArgs;

typedef struct CopyArgs
{
	uint64_t dst;
	uint64_t src;
	uint32_t chunks32;
} CopyArgs;

#define BENCH_TRY(call) \
	do { \
		const SparkStatus bench_status = (call); \
		if ( bench_status != SPARK_STATUS_OK ) \
			SPARK_FAIL(bench_status); \
	} while (0)

static SparkStatus BenchUpload(SparkDeviceBuffer *dev, const void *src, uint64_t bytes, const char *owner)
{
	SparkDeviceBuffer host;
	SparkStatus status;
	SparkDeviceRange dst_r,src_r;
	BENCH_TRY(SparkDeviceAllocate(dev,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,owner));
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

static SparkStatus BenchDownload(void *dst, const SparkDeviceBuffer *dev, uint64_t bytes, const char *owner)
{
	SparkDeviceBuffer host;
	SparkStatus status;
	SparkDeviceRange dst_r,src_r;
	BENCH_TRY(SparkDeviceAllocate(&host,SPARK_DEVICE_SPACE_HOST_PAGEABLE,bytes,owner));
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

static SparkStatus BenchTimeKernel(SparkDeviceQueue *queue, const SparkDeviceKernel *kernel, const SparkDeviceLaunch *launch,
	uint64_t traffic_bytes, BenchMetrics *metrics)
{
	SparkDeviceEvent *ev_start = 0,*ev_stop = 0;
	double batch_us[BENCH_BATCH_TRIALS],lat_us[BENCH_LATENCY_ROUNDS];
	uint32_t i,t;
	BENCH_TRY(SparkDeviceEventCreate(&ev_start));
	BENCH_TRY(SparkDeviceEventCreate(&ev_stop));
	for (i = 0u; i < BENCH_WARMUP_ROUNDS; i++)
		BENCH_TRY(SparkDeviceKernelLaunch(queue,kernel,launch));
	BENCH_TRY(SparkDeviceQueueWait(queue,BENCH_TIMEOUT_NS));
	for (t = 0u; t < BENCH_BATCH_TRIALS; t++)
	{
		uint64_t ns = 0u;
		BENCH_TRY(SparkDeviceEventRecord(ev_start,queue));
		for (i = 0u; i < BENCH_BATCH_SIZE; i++)
			BENCH_TRY(SparkDeviceKernelLaunch(queue,kernel,launch));
		BENCH_TRY(SparkDeviceEventRecord(ev_stop,queue));
		BENCH_TRY(SparkDeviceEventWait(ev_stop,BENCH_TIMEOUT_NS));
		BENCH_TRY(SparkDeviceEventElapsed(ev_start,ev_stop,&ns));
		batch_us[t] = (double)ns / (1000.0 * (double)BENCH_BATCH_SIZE);
	}
	for (i = 0u; i < BENCH_LATENCY_ROUNDS; i++)
	{
		const uint64_t t0 = SparkMonotonicNs();
		BENCH_TRY(SparkDeviceKernelLaunch(queue,kernel,launch));
		BENCH_TRY(SparkDeviceQueueWait(queue,BENCH_TIMEOUT_NS));
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

static SparkStatus RunRmsNorm(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
{
	const uint64_t x_bytes = (uint64_t)BENCH_RMS_ROWS * BENCH_RMS_DIM * 2u;
	const uint64_t w_bytes = (uint64_t)BENCH_RMS_DIM * 2u;
	uint16_t *hx = (uint16_t *)malloc((size_t)x_bytes);
	uint16_t *hw = (uint16_t *)malloc((size_t)w_bytes);
	uint16_t *ho = (uint16_t *)calloc(1u,(size_t)x_bytes);
	SparkDeviceBuffer dx,dw,dout;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	RmsArgs args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "rmsnorm_bf16_vec16";
	if ( hx == 0 || hw == 0 || ho == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillRmsInputs(hx,hw,BENCH_RMS_ROWS,BENCH_RMS_DIM);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&dx,hx,x_bytes,"rms_x"));
	BENCH_TRY(BenchUpload(&dw,hw,w_bytes,"rms_w"));
	BENCH_TRY(SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,x_bytes,"rms_out"));
	args.x = dx.device;
	args.w = dw.device;
	args.out = dout.device;
	args.rows = BENCH_RMS_ROWS;
	args.chunks16 = BENCH_RMS_CHUNKS16;
	args.eps = BENCH_RMS_EPS;
	args.inv_dim = 1.0f / (float)BENCH_RMS_DIM;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = BENCH_RMS_ROWS;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = BenchTimeKernel(queue,kernel,&launch,x_bytes + w_bytes + x_bytes,m);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(ho,&dout,x_bytes,"rms_out");
	if ( status == SPARK_STATUS_OK )
	{
		BenchVerifyRms(hx,hw,ho,BENCH_RMS_ROWS,BENCH_RMS_DIM,BENCH_RMS_EPS,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBf16(dump_dir,m->name,ho,(size_t)BENCH_RMS_ROWS * BENCH_RMS_DIM);
		if ( BenchCompareBf16File(cmp_dir,m->name,ho,(size_t)BENCH_RMS_ROWS * BENCH_RMS_DIM,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
			m->compared_cuda = 1;
	}
	SparkDeviceFree(&dout);
	SparkDeviceFree(&dw);
	SparkDeviceFree(&dx);
	SparkDeviceKernelDestroy(kernel);
	free(ho);
	free(hw);
	free(hx);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunFusedRmsNorm(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
{
	const uint64_t x_bytes = (uint64_t)BENCH_RMS_ROWS * BENCH_RMS_DIM * 2u;
	const uint64_t w_bytes = (uint64_t)BENCH_RMS_DIM * 2u;
	uint16_t *hx = (uint16_t *)malloc((size_t)x_bytes);
	uint16_t *hr = (uint16_t *)malloc((size_t)x_bytes);
	uint16_t *hw = (uint16_t *)malloc((size_t)w_bytes);
	uint16_t *hro = (uint16_t *)calloc(1u,(size_t)x_bytes);
	uint16_t *ho = (uint16_t *)calloc(1u,(size_t)x_bytes);
	SparkDeviceBuffer dx,dr,dw,dout;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	FusedRmsArgs args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "fused_residual_rmsnorm_bf16x8";
	if ( hx == 0 || hr == 0 || hw == 0 || hro == 0 || ho == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillFusedRmsInputs(hx,hr,hw,BENCH_RMS_ROWS,BENCH_RMS_DIM);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&dx,hx,x_bytes,"frms_x"));
	BENCH_TRY(BenchUpload(&dr,hr,x_bytes,"frms_r"));
	BENCH_TRY(BenchUpload(&dw,hw,w_bytes,"frms_w"));
	BENCH_TRY(SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,x_bytes,"frms_o"));
	args.x = dx.device;
	args.res = dr.device;
	args.w = dw.device;
	args.out = dout.device;
	args.rows = BENCH_RMS_ROWS;
	args.chunks16 = BENCH_RMS_CHUNKS16;
	args.eps = BENCH_RMS_EPS;
	args.inv_dim = 1.0f / (float)BENCH_RMS_DIM;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = BENCH_RMS_ROWS;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = SparkDeviceKernelLaunch(queue,kernel,&launch);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceQueueWait(queue,BENCH_TIMEOUT_NS);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(hro,&dr,x_bytes,"frms_ro");
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(ho,&dout,x_bytes,"frms_o");
	if ( status == SPARK_STATUS_OK )
	{
		double ra = 0.0;
		uint32_t ru = 0u;
		BenchVerifyFusedRms(hx,hr,hw,hro,ho,BENCH_RMS_ROWS,BENCH_RMS_DIM,BENCH_RMS_EPS,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBf16(dump_dir,m->name,ho,(size_t)BENCH_RMS_ROWS * BENCH_RMS_DIM);
		BenchDumpBf16(dump_dir,"fused_residual_rmsnorm_bf16x8_res",hro,(size_t)BENCH_RMS_ROWS * BENCH_RMS_DIM);
		if ( BenchCompareBf16File(cmp_dir,m->name,ho,(size_t)BENCH_RMS_ROWS * BENCH_RMS_DIM,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
		{
			m->compared_cuda = 1;
			if ( BenchCompareBf16File(cmp_dir,"fused_residual_rmsnorm_bf16x8_res",hro,(size_t)BENCH_RMS_ROWS * BENCH_RMS_DIM,&ra,&ru) > 0 )
			{
				if ( ra > m->max_abs_vs_cuda )
					m->max_abs_vs_cuda = ra;
				if ( ru > m->max_ulp_vs_cuda )
					m->max_ulp_vs_cuda = ru;
			}
		}
		status = BenchTimeKernel(queue,kernel,&launch,x_bytes * 4u + w_bytes,m);
	}
	SparkDeviceFree(&dout);
	SparkDeviceFree(&dw);
	SparkDeviceFree(&dr);
	SparkDeviceFree(&dx);
	SparkDeviceKernelDestroy(kernel);
	free(ho);
	free(hro);
	free(hw);
	free(hr);
	free(hx);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunSwiglu(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
{
	const uint64_t bytes = (uint64_t)BENCH_SWIGLU_ELEMS * 2u;
	uint16_t *hg = (uint16_t *)malloc((size_t)bytes);
	uint16_t *hu = (uint16_t *)malloc((size_t)bytes);
	uint16_t *ho = (uint16_t *)calloc(1u,(size_t)bytes);
	SparkDeviceBuffer dg,du,dout;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	SwigluArgs args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "swiglu_packed_bf16x8";
	if ( hg == 0 || hu == 0 || ho == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillSwigluInputs(hg,hu,BENCH_SWIGLU_ELEMS);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&dg,hg,bytes,"swi_g"));
	BENCH_TRY(BenchUpload(&du,hu,bytes,"swi_u"));
	BENCH_TRY(SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,"swi_o"));
	args.gate = dg.device;
	args.up = du.device;
	args.out = dout.device;
	args.total_chunks16 = BENCH_SWIGLU_CHUNKS16;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = (BENCH_SWIGLU_CHUNKS16 + 255u) / 256u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = BenchTimeKernel(queue,kernel,&launch,bytes * 3u,m);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(ho,&dout,bytes,"swi_o");
	if ( status == SPARK_STATUS_OK )
	{
		BenchVerifySwiglu(hg,hu,ho,BENCH_SWIGLU_ELEMS,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBf16(dump_dir,m->name,ho,(size_t)BENCH_SWIGLU_ELEMS);
		if ( BenchCompareBf16File(cmp_dir,m->name,ho,(size_t)BENCH_SWIGLU_ELEMS,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
			m->compared_cuda = 1;
	}
	SparkDeviceFree(&dout);
	SparkDeviceFree(&du);
	SparkDeviceFree(&dg);
	SparkDeviceKernelDestroy(kernel);
	free(ho);
	free(hu);
	free(hg);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunSkinnyBf16(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
{
	const uint64_t w_bytes = (uint64_t)BENCH_SKINNY_BF16_OUT * BENCH_SKINNY_BF16_K * 2u;
	const uint64_t a_bytes = (uint64_t)BENCH_SKINNY_BF16_K * 2u;
	const uint64_t o_bytes = (uint64_t)BENCH_SKINNY_BF16_OUT * 2u;
	uint16_t *hw = (uint16_t *)malloc((size_t)w_bytes);
	uint16_t *ha = (uint16_t *)malloc((size_t)a_bytes);
	uint16_t *ho = (uint16_t *)calloc(1u,(size_t)o_bytes);
	SparkDeviceBuffer dw,da,dout;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	SkinnyBf16Args args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "skinny_bf16_b1";
	if ( hw == 0 || ha == 0 || ho == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillSkinnyBf16Inputs(hw,ha,BENCH_SKINNY_BF16_OUT,BENCH_SKINNY_BF16_K);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&dw,hw,w_bytes,"sk_bf16_w"));
	BENCH_TRY(BenchUpload(&da,ha,a_bytes,"sk_bf16_a"));
	BENCH_TRY(SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,o_bytes,"sk_bf16_o"));
	args.weight = dw.device;
	args.act = da.device;
	args.out = dout.device;
	args.chunks16 = BENCH_SKINNY_BF16_CHUNKS16;
	args.out_dim = BENCH_SKINNY_BF16_OUT;
	args.out_col_off = 0u;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = (BENCH_SKINNY_BF16_OUT + 7u) / 8u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 128u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = BenchTimeKernel(queue,kernel,&launch,w_bytes + a_bytes + o_bytes,m);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(ho,&dout,o_bytes,"sk_bf16_o");
	if ( status == SPARK_STATUS_OK )
	{
		BenchVerifySkinnyBf16(hw,ha,ho,BENCH_SKINNY_BF16_OUT,BENCH_SKINNY_BF16_K,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBf16(dump_dir,m->name,ho,(size_t)BENCH_SKINNY_BF16_OUT);
		if ( BenchCompareBf16File(cmp_dir,m->name,ho,(size_t)BENCH_SKINNY_BF16_OUT,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
			m->compared_cuda = 1;
	}
	SparkDeviceFree(&dout);
	SparkDeviceFree(&da);
	SparkDeviceFree(&dw);
	SparkDeviceKernelDestroy(kernel);
	free(ho);
	free(ha);
	free(hw);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunSkinnyNvfp4(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
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
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	SkinnyNvfp4Args args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "skinny_nvfp4_b1";
	if ( hw == 0 || hs == 0 || ha == 0 || ho == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillSkinnyNvfp4Inputs(hw,hs,ha,&hg,BENCH_SKINNY_NVFP4_OUT,BENCH_SKINNY_NVFP4_K);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&dw,hw,w_bytes,"sk_fp4_w"));
	BENCH_TRY(BenchUpload(&ds,hs,s_bytes,"sk_fp4_s"));
	BENCH_TRY(BenchUpload(&da,ha,a_bytes,"sk_fp4_a"));
	BENCH_TRY(BenchUpload(&dg,&hg,sizeof(hg),"sk_fp4_g"));
	BENCH_TRY(SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,o_bytes,"sk_fp4_o"));
	args.weight = dw.device;
	args.act = da.device;
	args.scales = ds.device;
	args.gscale = dg.device;
	args.out = dout.device;
	args.chunks32 = BENCH_SKINNY_NVFP4_CHUNKS32;
	args.out_dim = BENCH_SKINNY_NVFP4_OUT;
	args.out_col_off = 0u;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = (BENCH_SKINNY_NVFP4_OUT + 7u) / 8u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 128u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = BenchTimeKernel(queue,kernel,&launch,w_bytes + s_bytes + a_bytes + o_bytes,m);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(ho,&dout,o_bytes,"sk_fp4_o");
	if ( status == SPARK_STATUS_OK )
	{
		BenchVerifySkinnyNvfp4(hw,hs,ha,hg,ho,BENCH_SKINNY_NVFP4_OUT,BENCH_SKINNY_NVFP4_K,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBf16(dump_dir,m->name,ho,(size_t)BENCH_SKINNY_NVFP4_OUT);
		if ( BenchCompareBf16File(cmp_dir,m->name,ho,(size_t)BENCH_SKINNY_NVFP4_OUT,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
			m->compared_cuda = 1;
	}
	SparkDeviceFree(&dout);
	SparkDeviceFree(&dg);
	SparkDeviceFree(&da);
	SparkDeviceFree(&ds);
	SparkDeviceFree(&dw);
	SparkDeviceKernelDestroy(kernel);
	free(ho);
	free(ha);
	free(hs);
	free(hw);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunSkinnyMxfp4(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
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
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	SkinnyMxfp4Args args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "skinny_mxfp4_b1";
	if ( hw == 0 || hs == 0 || ha == 0 || ho == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillSkinnyMxfp4Inputs(hw,hs,ha,BENCH_SKINNY_MXFP4_OUT,BENCH_SKINNY_MXFP4_K);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&dw,hw,w_bytes,"sk_mx4_w"));
	BENCH_TRY(BenchUpload(&ds,hs,s_bytes,"sk_mx4_s"));
	BENCH_TRY(BenchUpload(&da,ha,a_bytes,"sk_mx4_a"));
	BENCH_TRY(SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,o_bytes,"sk_mx4_o"));
	args.weight = dw.device;
	args.act = da.device;
	args.scales = ds.device;
	args.out = dout.device;
	args.chunks32 = BENCH_SKINNY_MXFP4_CHUNKS32;
	args.out_dim = BENCH_SKINNY_MXFP4_OUT;
	args.out_col_off = 0u;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = (BENCH_SKINNY_MXFP4_OUT + 7u) / 8u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 128u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = BenchTimeKernel(queue,kernel,&launch,w_bytes + s_bytes + a_bytes + o_bytes,m);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(ho,&dout,o_bytes,"sk_mx4_o");
	if ( status == SPARK_STATUS_OK )
	{
		BenchVerifySkinnyMxfp4(hw,hs,ha,ho,BENCH_SKINNY_MXFP4_OUT,BENCH_SKINNY_MXFP4_K,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBf16(dump_dir,m->name,ho,(size_t)BENCH_SKINNY_MXFP4_OUT);
		if ( BenchCompareBf16File(cmp_dir,m->name,ho,(size_t)BENCH_SKINNY_MXFP4_OUT,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
			m->compared_cuda = 1;
	}
	SparkDeviceFree(&dout);
	SparkDeviceFree(&da);
	SparkDeviceFree(&ds);
	SparkDeviceFree(&dw);
	SparkDeviceKernelDestroy(kernel);
	free(ho);
	free(ha);
	free(hs);
	free(hw);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunLatentAttn(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
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
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	AttnArgs args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "latent_attn_decode_512_64";
	if ( hqn == 0 || hqr == 0 || hkn == 0 || hkr == 0 || ho == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillAttnInputs(hqn,hqr,hkn,hkr,BENCH_ATTN_HEADS,BENCH_ATTN_KV_LEN);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&dqn,hqn,qn_bytes,"attn_qn"));
	BENCH_TRY(BenchUpload(&dqr,hqr,qr_bytes,"attn_qr"));
	BENCH_TRY(BenchUpload(&dkn,hkn,kn_bytes,"attn_kn"));
	BENCH_TRY(BenchUpload(&dkr,hkr,kr_bytes,"attn_kr"));
	BENCH_TRY(SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,o_bytes,"attn_o"));
	args.q_nope = dqn.device;
	args.q_rope = dqr.device;
	args.kv_nope = dkn.device;
	args.kv_rope = dkr.device;
	args.out = dout.device;
	args.num_heads = BENCH_ATTN_HEADS;
	args.kv_len = BENCH_ATTN_KV_LEN;
	args.scale = BENCH_ATTN_SCALE;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = BENCH_ATTN_HEADS;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = BenchTimeKernel(queue,kernel,&launch,traffic,m);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(ho,&dout,o_bytes,"attn_o");
	if ( status == SPARK_STATUS_OK )
	{
		BenchVerifyAttn(hqn,hqr,hkn,hkr,ho,BENCH_ATTN_HEADS,BENCH_ATTN_KV_LEN,BENCH_ATTN_SCALE,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBf16(dump_dir,m->name,ho,(size_t)BENCH_ATTN_HEADS * BENCH_ATTN_NOPE);
		if ( BenchCompareBf16File(cmp_dir,m->name,ho,(size_t)BENCH_ATTN_HEADS * BENCH_ATTN_NOPE,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
			m->compared_cuda = 1;
	}
	SparkDeviceFree(&dout);
	SparkDeviceFree(&dkr);
	SparkDeviceFree(&dkn);
	SparkDeviceFree(&dqr);
	SparkDeviceFree(&dqn);
	SparkDeviceKernelDestroy(kernel);
	free(ho);
	free(hkr);
	free(hkn);
	free(hqr);
	free(hqn);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunRouterF32(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
{
	const uint64_t count = (uint64_t)BENCH_ROUTER_ROWS * BENCH_ROUTER_COLS;
	const uint64_t bytes = count * 4u;
	float *hin = (float *)malloc((size_t)bytes);
	float *ho = (float *)calloc(1u,(size_t)bytes);
	SparkDeviceBuffer din,dout;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	RouterArgs args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "router_softmax_256";
	m->abs_limit_fp32 = 1e-5;
	m->ulp_limit_cuda = 4u;
	if ( hin == 0 || ho == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillRouterF32Inputs(hin,BENCH_ROUTER_ROWS,BENCH_ROUTER_COLS);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&din,hin,bytes,"rtr_in"));
	BENCH_TRY(SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,"rtr_out"));
	args.logits = din.device;
	args.probs = dout.device;
	args.rows = BENCH_ROUTER_ROWS;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = (BENCH_ROUTER_ROWS + 7u) / 8u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = BenchTimeKernel(queue,kernel,&launch,bytes * 2u,m);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(ho,&dout,bytes,"rtr_out");
	if ( status == SPARK_STATUS_OK )
	{
		BenchVerifyRouterF32(hin,ho,BENCH_ROUTER_ROWS,BENCH_ROUTER_COLS,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBytes(dump_dir,m->name,ho,(size_t)bytes);
		if ( BenchCompareF32File(cmp_dir,m->name,ho,(size_t)count,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
			m->compared_cuda = 1;
	}
	SparkDeviceFree(&dout);
	SparkDeviceFree(&din);
	SparkDeviceKernelDestroy(kernel);
	free(ho);
	free(hin);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunRouterBf16(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
{
	const uint64_t count = (uint64_t)BENCH_ROUTER_ROWS * BENCH_ROUTER_COLS;
	const uint64_t in_bytes = count * 2u;
	const uint64_t out_bytes = count * 4u;
	uint16_t *hin = (uint16_t *)malloc((size_t)in_bytes);
	float *ho = (float *)calloc(1u,(size_t)out_bytes);
	SparkDeviceBuffer din,dout;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	RouterArgs args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "router_softmax_bf16_256";
	m->abs_limit_fp32 = 1e-5;
	m->ulp_limit_cuda = 4u;
	if ( hin == 0 || ho == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillRouterBf16Inputs(hin,BENCH_ROUTER_ROWS,BENCH_ROUTER_COLS);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&din,hin,in_bytes,"rtr16_in"));
	BENCH_TRY(SparkDeviceAllocate(&dout,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,out_bytes,"rtr16_out"));
	args.logits = din.device;
	args.probs = dout.device;
	args.rows = BENCH_ROUTER_ROWS;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = (BENCH_ROUTER_ROWS + 7u) / 8u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = BenchTimeKernel(queue,kernel,&launch,in_bytes + out_bytes,m);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(ho,&dout,out_bytes,"rtr16_out");
	if ( status == SPARK_STATUS_OK )
	{
		BenchVerifyRouterBf16(hin,ho,BENCH_ROUTER_ROWS,BENCH_ROUTER_COLS,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBytes(dump_dir,m->name,ho,(size_t)out_bytes);
		if ( BenchCompareF32File(cmp_dir,m->name,ho,(size_t)count,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
			m->compared_cuda = 1;
	}
	SparkDeviceFree(&dout);
	SparkDeviceFree(&din);
	SparkDeviceKernelDestroy(kernel);
	free(ho);
	free(hin);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunMxQuant(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
{
	const uint64_t src_bytes = (uint64_t)BENCH_MX_ELEMS * 2u;
	const uint64_t fp8_bytes = (uint64_t)BENCH_MX_ELEMS;
	const uint64_t sc_bytes = (uint64_t)BENCH_MX_BLOCKS;
	uint16_t *hsrc = (uint16_t *)malloc((size_t)src_bytes);
	uint8_t *hfp8 = (uint8_t *)calloc(1u,(size_t)fp8_bytes);
	uint8_t *hsc = (uint8_t *)calloc(1u,(size_t)sc_bytes);
	SparkDeviceBuffer dsrc,dfp8,dsc;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	MxQuantArgs args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "mx_quantize_e4m3_ue8m0";
	if ( hsrc == 0 || hfp8 == 0 || hsc == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillMxQuantInputs(hsrc,BENCH_MX_ELEMS);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&dsrc,hsrc,src_bytes,"mx_src"));
	BENCH_TRY(SparkDeviceAllocate(&dfp8,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,fp8_bytes,"mx_fp8"));
	BENCH_TRY(SparkDeviceAllocate(&dsc,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,sc_bytes,"mx_sc"));
	args.src_bf16 = dsrc.device;
	args.dst_fp8 = dfp8.device;
	args.dst_scales = dsc.device;
	args.total_blocks = BENCH_MX_BLOCKS;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = (BENCH_MX_BLOCKS * 4u + 255u) / 256u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = BenchTimeKernel(queue,kernel,&launch,src_bytes + fp8_bytes + sc_bytes,m);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(hfp8,&dfp8,fp8_bytes,"mx_fp8");
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(hsc,&dsc,sc_bytes,"mx_sc");
	if ( status == SPARK_STATUS_OK )
	{
		double sa = 0.0;
		uint32_t su = 0u;
		BenchVerifyMxQuant(hsrc,hfp8,hsc,BENCH_MX_BLOCKS,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBytes(dump_dir,m->name,hfp8,(size_t)fp8_bytes);
		BenchDumpBytes(dump_dir,"mx_quantize_e4m3_ue8m0_scales",hsc,(size_t)sc_bytes);
		if ( BenchCompareBytesFile(cmp_dir,m->name,hfp8,(size_t)fp8_bytes,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
		{
			m->compared_cuda = 1;
			if ( BenchCompareBytesFile(cmp_dir,"mx_quantize_e4m3_ue8m0_scales",hsc,(size_t)sc_bytes,&sa,&su) > 0 )
			{
				m->max_abs_vs_cuda += sa;
				m->max_ulp_vs_cuda += su;
			}
		}
	}
	SparkDeviceFree(&dsc);
	SparkDeviceFree(&dfp8);
	SparkDeviceFree(&dsrc);
	SparkDeviceKernelDestroy(kernel);
	free(hsc);
	free(hfp8);
	free(hsrc);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunTpF32ToBf16(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
{
	const uint64_t src_bytes = (uint64_t)BENCH_TP_ELEMS * 4u;
	const uint64_t dst_bytes = (uint64_t)BENCH_TP_ELEMS * 2u;
	float *hsrc = (float *)malloc((size_t)src_bytes);
	uint16_t *hdst = (uint16_t *)calloc(1u,(size_t)dst_bytes);
	SparkDeviceBuffer dsrc,ddst;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	TpConvArgs args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "tp_f32_to_bf16_vec16";
	if ( hsrc == 0 || hdst == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillTpF32Inputs(hsrc,BENCH_TP_ELEMS);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&dsrc,hsrc,src_bytes,"tp_f32_src"));
	BENCH_TRY(SparkDeviceAllocate(&ddst,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,dst_bytes,"tp_bf16_dst"));
	args.src = dsrc.device;
	args.dst = ddst.device;
	args.chunks16 = BENCH_TP_CHUNKS16;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = (BENCH_TP_CHUNKS16 + 255u) / 256u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = BenchTimeKernel(queue,kernel,&launch,src_bytes + dst_bytes,m);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(hdst,&ddst,dst_bytes,"tp_bf16_dst");
	if ( status == SPARK_STATUS_OK )
	{
		BenchVerifyTpF32ToBf16(hsrc,hdst,BENCH_TP_ELEMS,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBf16(dump_dir,m->name,hdst,(size_t)BENCH_TP_ELEMS);
		if ( BenchCompareBf16File(cmp_dir,m->name,hdst,(size_t)BENCH_TP_ELEMS,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
			m->compared_cuda = 1;
	}
	SparkDeviceFree(&ddst);
	SparkDeviceFree(&dsrc);
	SparkDeviceKernelDestroy(kernel);
	free(hdst);
	free(hsrc);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunTpBf16ToF32(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
{
	const uint64_t src_bytes = (uint64_t)BENCH_TP_ELEMS * 2u;
	const uint64_t dst_bytes = (uint64_t)BENCH_TP_ELEMS * 4u;
	uint16_t *hsrc = (uint16_t *)malloc((size_t)src_bytes);
	float *hdst = (float *)calloc(1u,(size_t)dst_bytes);
	SparkDeviceBuffer dsrc,ddst;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	TpConvArgs args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "tp_bf16_to_f32_vec16";
	if ( hsrc == 0 || hdst == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillTpBf16Inputs(hsrc,BENCH_TP_ELEMS);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&dsrc,hsrc,src_bytes,"tp_bf16_src"));
	BENCH_TRY(SparkDeviceAllocate(&ddst,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,dst_bytes,"tp_f32_dst"));
	args.src = dsrc.device;
	args.dst = ddst.device;
	args.chunks16 = BENCH_TP_CHUNKS16;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = (BENCH_TP_CHUNKS16 + 255u) / 256u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = BenchTimeKernel(queue,kernel,&launch,src_bytes + dst_bytes,m);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(hdst,&ddst,dst_bytes,"tp_f32_dst");
	if ( status == SPARK_STATUS_OK )
	{
		BenchVerifyTpBf16ToF32(hsrc,hdst,BENCH_TP_ELEMS,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBytes(dump_dir,m->name,hdst,(size_t)dst_bytes);
		if ( BenchCompareF32File(cmp_dir,m->name,hdst,(size_t)BENCH_TP_ELEMS,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
			m->compared_cuda = 1;
	}
	SparkDeviceFree(&ddst);
	SparkDeviceFree(&dsrc);
	SparkDeviceKernelDestroy(kernel);
	free(hdst);
	free(hsrc);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus RunCopyVec256(SparkDeviceQueue *queue, SparkDeviceLibrary *lib, const char *dump_dir, const char *cmp_dir, BenchMetrics *m)
{
	const uint64_t bytes = BENCH_COPY_BYTES;
	const uint64_t words = bytes / 4u;
	uint32_t *hsrc = (uint32_t *)malloc((size_t)bytes);
	uint32_t *hdst = (uint32_t *)calloc(1u,(size_t)bytes);
	SparkDeviceBuffer dsrc,ddst;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceKernelInfo info;
	SparkDeviceLaunch launch;
	CopyArgs args;
	SparkStatus status;
	memset(m,0,sizeof(*m));
	m->name = "device_program_copy_vec256";
	if ( hsrc == 0 || hdst == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	BenchFillCopyInputs(hsrc,words);
	BENCH_TRY(SparkDeviceKernelCreate(&kernel,lib,m->name));
	BENCH_TRY(SparkDeviceKernelGetInfo(kernel,&info));
	BENCH_TRY(BenchUpload(&dsrc,hsrc,bytes,"copy_src"));
	BENCH_TRY(SparkDeviceAllocate(&ddst,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,bytes,"copy_dst"));
	args.dst = ddst.device;
	args.src = dsrc.device;
	args.chunks32 = BENCH_COPY_CHUNKS32;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = (BENCH_COPY_CHUNKS32 + 255u) / 256u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &args;
	launch.constant_bytes = info.parameter_bytes;
	status = BenchTimeKernel(queue,kernel,&launch,bytes * 2u,m);
	if ( status == SPARK_STATUS_OK )
		status = BenchDownload(hdst,&ddst,bytes,"copy_dst");
	if ( status == SPARK_STATUS_OK )
	{
		BenchVerifyCopy(hsrc,hdst,words,&m->max_abs_err,&m->max_ulp_err);
		BenchDumpBytes(dump_dir,m->name,hdst,(size_t)bytes);
		if ( BenchCompareBytesFile(cmp_dir,m->name,(const uint8_t *)hdst,(size_t)bytes,&m->max_abs_vs_cuda,&m->max_ulp_vs_cuda) > 0 )
			m->compared_cuda = 1;
	}
	SparkDeviceFree(&ddst);
	SparkDeviceFree(&dsrc);
	SparkDeviceKernelDestroy(kernel);
	free(hdst);
	free(hsrc);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

typedef SparkStatus (*BenchRunnerFn)(SparkDeviceQueue *, SparkDeviceLibrary *, const char *, const char *, BenchMetrics *);

typedef struct BenchEntry
{
	const char *name;
	BenchRunnerFn fn;
} BenchEntry;

int main(int argc, char **argv)
{
	static const BenchEntry entries[BENCH_NUM_KERNELS] = {
		{"rmsnorm_bf16_vec16", RunRmsNorm},
		{"fused_residual_rmsnorm_bf16x8", RunFusedRmsNorm},
		{"swiglu_packed_bf16x8", RunSwiglu},
		{"skinny_bf16_b1", RunSkinnyBf16},
		{"skinny_nvfp4_b1", RunSkinnyNvfp4},
		{"skinny_mxfp4_b1", RunSkinnyMxfp4},
		{"latent_attn_decode_512_64", RunLatentAttn},
		{"router_softmax_256", RunRouterF32},
		{"router_softmax_bf16_256", RunRouterBf16},
		{"mx_quantize_e4m3_ue8m0", RunMxQuant},
		{"tp_f32_to_bf16_vec16", RunTpF32ToBf16},
		{"tp_bf16_to_f32_vec16", RunTpBf16ToF32},
		{"device_program_copy_vec256", RunCopyVec256},
	};
	const char *cubin_path = 0,*label = "sklc+nvrm",*dump_dir = 0,*cmp_dir = 0,*only_kernel = 0;
	SparkDeviceConfiguration cfg = {0u, 0u};
	SparkDeviceQueue *queue = 0;
	SparkDeviceLibrary *lib = 0;
	BenchMetrics rows[BENCH_NUM_KERNELS];
	uint32_t ran = 0u,k;
	int i,failed = 0;
	for (i = 1; i < argc; i++)
	{
		if ( strcmp(argv[i],"--cubin") == 0 && i + 1 < argc )
			cubin_path = argv[++i];
		else if ( strcmp(argv[i],"--label") == 0 && i + 1 < argc )
			label = argv[++i];
		else if ( strcmp(argv[i],"--dump") == 0 && i + 1 < argc )
			dump_dir = argv[++i];
		else if ( strcmp(argv[i],"--compare") == 0 && i + 1 < argc )
			cmp_dir = argv[++i];
		else if ( strcmp(argv[i],"--kernel") == 0 && i + 1 < argc )
			only_kernel = argv[++i];
		else
		{
			printf("usage: %s --cubin PATH [--label NAME] [--dump DIR] [--compare DIR] [--kernel NAME]\n",argv[0]);
			return(2);
		}
	}
	if ( cubin_path == 0 )
	{
		printf("FAIL --cubin PATH is required\n");
		return(2);
	}
	if ( SparkDeviceInitialize(&cfg) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue,"sklc_bench") != SPARK_STATUS_OK ||
		SparkDeviceLibraryLoad(&lib,cubin_path,"sklc_bench") != SPARK_STATUS_OK )
	{
		const SparkDeviceError *e = SparkDeviceLastError();
		printf("FAIL setup: %s [%s owner=%s code=%d]\n",e->text,e->operation,e->owner,e->backend_code);
		return(1);
	}
	{
		const SparkDeviceProperties *p = SparkDeviceGetProperties();
		printf("INFO label=%s backend=%s target=%s device=\"%s\" cus=%u cubin=%s\n",label,p->backend,p->target,p->name,p->compute_units,cubin_path);
	}
	for (k = 0u; k < BENCH_NUM_KERNELS; k++)
	{
		if ( only_kernel != 0 && strcmp(entries[k].name,only_kernel) != 0 )
			continue;
		if ( entries[k].fn(queue,lib,dump_dir,cmp_dir,&rows[ran]) != SPARK_STATUS_OK )
		{
			const SparkDeviceError *e = SparkDeviceLastError();
			printf("FAIL execute %s: %s [%s owner=%s code=%d]\n",entries[k].name,e->text,e->operation,e->owner,e->backend_code);
			failed = 1;
			break;
		}
		BenchPrintRow(label,&rows[ran]);
		fflush(stdout);
		if ( !BenchCheckParity(&rows[ran]) )
		{
			printf("FAIL numerical parity for %s (max_err_fp32=%.6f max_ulp_fp32=%u max_ulp_cuda=%u)\n",
				rows[ran].name,rows[ran].max_abs_err,rows[ran].max_ulp_err,rows[ran].max_ulp_vs_cuda);
			failed = 1;
		}
		ran++;
	}
	if ( only_kernel != 0 && ran == 0u )
	{
		printf("FAIL unknown kernel filter: %s\n",only_kernel);
		failed = 1;
	}
	SparkDeviceLibraryUnload(lib);
	SparkDeviceQueueDestroy(queue);
	SparkDeviceShutdown();
	if ( failed == 0 )
		printf("PASS %s verified %u kernels\n",label,ran);
	return(failed);
}
