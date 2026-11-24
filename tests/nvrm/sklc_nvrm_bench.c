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

int main(int argc, char **argv)
{
	const char *cubin_path = 0,*label = "sklc+nvrm",*dump_dir = 0,*cmp_dir = 0;
	SparkDeviceConfiguration cfg = {0u, 0u};
	SparkDeviceQueue *queue = 0;
	SparkDeviceLibrary *lib = 0;
	BenchMetrics rows[4];
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
		else
		{
			printf("usage: %s --cubin PATH [--label NAME] [--dump DIR] [--compare DIR]\n",argv[0]);
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
	if ( RunRmsNorm(queue,lib,dump_dir,cmp_dir,&rows[0]) != SPARK_STATUS_OK ||
		RunSwiglu(queue,lib,dump_dir,cmp_dir,&rows[1]) != SPARK_STATUS_OK ||
		RunSkinnyBf16(queue,lib,dump_dir,cmp_dir,&rows[2]) != SPARK_STATUS_OK ||
		RunSkinnyNvfp4(queue,lib,dump_dir,cmp_dir,&rows[3]) != SPARK_STATUS_OK )
	{
		const SparkDeviceError *e = SparkDeviceLastError();
		printf("FAIL execute: %s [%s owner=%s code=%d]\n",e->text,e->operation,e->owner,e->backend_code);
		failed = 1;
	}
	else
	{
		for (i = 0; i < 4; i++)
		{
			BenchPrintRow(label,&rows[i]);
			if ( rows[i].max_ulp_err > 2u || (rows[i].compared_cuda && rows[i].max_ulp_vs_cuda > 2u) )
			{
				printf("FAIL numerical parity for %s (max_ulp_fp32=%u max_ulp_cuda=%u)\n",rows[i].name,rows[i].max_ulp_err,rows[i].max_ulp_vs_cuda);
				failed = 1;
			}
		}
	}
	SparkDeviceLibraryUnload(lib);
	SparkDeviceQueueDestroy(queue);
	SparkDeviceShutdown();
	if ( failed == 0 )
		printf("PASS %s verified 4 kernels\n",label);
	return(failed);
}
