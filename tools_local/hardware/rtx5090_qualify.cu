#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CUDA_OK(call,code) do { cudaError_t e_ = (call); if ( e_ != cudaSuccess ) { fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return(code); } } while (0)
#define CUBLAS_OK(call,code) do { cublasStatus_t s_ = (call); if ( s_ != CUBLAS_STATUS_SUCCESS ) { fprintf(stderr,"cuBLAS error %s:%d: %d\n",__FILE__,__LINE__,(int32_t)s_); return(code); } } while (0)

__global__ void fill_pattern(uint64_t *p,uint64_t n,uint64_t seed)
{
	uint64_t i,stride;
	i = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	stride = ((uint64_t)gridDim.x * blockDim.x);
	for (; i<n; i+=stride)
		p[i] = (seed ^ (i * UINT64_C(0x9e3779b97f4a7c15)));
}

__global__ void verify_pattern(const uint64_t *p,uint64_t n,uint64_t seed,uint64_t *bad)
{
	uint64_t i,stride,want;
	i = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	stride = ((uint64_t)gridDim.x * blockDim.x);
	for (; i<n; i+=stride)
	{
		want = (seed ^ (i * UINT64_C(0x9e3779b97f4a7c15)));
		if ( p[i] != want )
			atomicAdd((unsigned long long *)bad,UINT64_C(1));
	}
}

__global__ void fill_half(__half *p,uint64_t n,__half value)
{
	uint64_t i,stride;
	i = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	stride = ((uint64_t)gridDim.x * blockDim.x);
	for (; i<n; i+=stride)
		p[i] = value;
}

static int32_t memory_integrity(void)
{
	size_t free_bytes,total_bytes,test_bytes;
	uint64_t *p,*bad,bad_host = 0,n,seed;
	CUDA_OK(cudaMemGetInfo(&free_bytes,&total_bytes),-1);
	test_bytes = ((free_bytes - (2ULL << 30)) & ~((256ULL << 20) - 1));
	if ( test_bytes > (28ULL << 30) )
		test_bytes = (28ULL << 30);
	n = (test_bytes / sizeof(uint64_t));
	seed = UINT64_C(0xd54f3a1b8c927e61);
	CUDA_OK(cudaMalloc((void **)&p,test_bytes),-2);
	CUDA_OK(cudaMalloc((void **)&bad,sizeof(uint64_t)),-3);
	CUDA_OK(cudaMemset(bad,0,sizeof(uint64_t)),-4);
	fill_pattern<<<65535,256>>>(p,n,seed);
	CUDA_OK(cudaGetLastError(),-5);
	verify_pattern<<<65535,256>>>(p,n,seed,bad);
	CUDA_OK(cudaGetLastError(),-6);
	CUDA_OK(cudaMemcpy(&bad_host,bad,sizeof(uint64_t),cudaMemcpyDeviceToHost),-7);
	CUDA_OK(cudaFree(bad),-8);
	CUDA_OK(cudaFree(p),-9);
	printf("VRAM_INTEGRITY bytes=%llu percent=%.1f mismatches=%llu %s\n",(unsigned long long)test_bytes,(100.0 * (double)test_bytes / (double)total_bytes),(unsigned long long)bad_host,(bad_host == 0) ? "PASS" : "FAIL");
	return((bad_host == 0) ? 0 : -10);
}

static int32_t device_bandwidth(void)
{
	uint8_t *a,*b;
	cudaEvent_t start,stop;
	float ms;
	uint64_t bytes = (4ULL << 30),i,iters = 12;
	CUDA_OK(cudaMalloc((void **)&a,(size_t)bytes),-1);
	CUDA_OK(cudaMalloc((void **)&b,(size_t)bytes),-2);
	CUDA_OK(cudaEventCreate(&start),-3);
	CUDA_OK(cudaEventCreate(&stop),-4);
	CUDA_OK(cudaEventRecord(start),-5);
	for (i=0; i<iters; i++)
		CUDA_OK(cudaMemcpyAsync(b,a,(size_t)bytes,cudaMemcpyDeviceToDevice),-6);
	CUDA_OK(cudaEventRecord(stop),-7);
	CUDA_OK(cudaEventSynchronize(stop),-8);
	CUDA_OK(cudaEventElapsedTime(&ms,start,stop),-9);
	printf("DEVICE_COPY bytes=%llu iterations=%llu bandwidth_GBs=%.2f PASS\n",(unsigned long long)bytes,(unsigned long long)iters,((double)bytes * (double)iters / ((double)ms / 1000.0) / 1.0e9));
	CUDA_OK(cudaEventDestroy(stop),-10);
	CUDA_OK(cudaEventDestroy(start),-11);
	CUDA_OK(cudaFree(b),-12);
	CUDA_OK(cudaFree(a),-13);
	return(0);
}

static int32_t pcie_bandwidth(void)
{
	uint8_t *host,*device;
	cudaEvent_t start,stop;
	float h2d_ms,d2h_ms;
	uint64_t bytes = (512ULL << 20),i,iters = 10;
	CUDA_OK(cudaMallocHost((void **)&host,(size_t)bytes),-1);
	CUDA_OK(cudaMalloc((void **)&device,(size_t)bytes),-2);
	memset(host,0x5a,(size_t)bytes);
	CUDA_OK(cudaEventCreate(&start),-4);
	CUDA_OK(cudaEventCreate(&stop),-5);
	CUDA_OK(cudaEventRecord(start),-6);
	for (i=0; i<iters; i++)
		CUDA_OK(cudaMemcpyAsync(device,host,(size_t)bytes,cudaMemcpyHostToDevice),-7);
	CUDA_OK(cudaEventRecord(stop),-8);
	CUDA_OK(cudaEventSynchronize(stop),-9);
	CUDA_OK(cudaEventElapsedTime(&h2d_ms,start,stop),-10);
	CUDA_OK(cudaEventRecord(start),-11);
	for (i=0; i<iters; i++)
		CUDA_OK(cudaMemcpyAsync(host,device,(size_t)bytes,cudaMemcpyDeviceToHost),-12);
	CUDA_OK(cudaEventRecord(stop),-13);
	CUDA_OK(cudaEventSynchronize(stop),-14);
	CUDA_OK(cudaEventElapsedTime(&d2h_ms,start,stop),-15);
	printf("PCIE_COPY bytes=%llu iterations=%llu h2d_GBs=%.2f d2h_GBs=%.2f PASS\n",(unsigned long long)bytes,(unsigned long long)iters,((double)bytes * (double)iters / ((double)h2d_ms / 1000.0) / 1.0e9),((double)bytes * (double)iters / ((double)d2h_ms / 1000.0) / 1.0e9));
	CUDA_OK(cudaEventDestroy(stop),-16);
	CUDA_OK(cudaEventDestroy(start),-17);
	CUDA_OK(cudaFree(device),-18);
	CUDA_OK(cudaFreeHost(host),-19);
	return(0);
}

static int32_t compute_stress(void)
{
	cublasHandle_t handle;
	__half *a,*b;
	float *c,out,alpha = 1.0f,beta = 0.0f,ms,total_ms = 0.0f;
	cudaEvent_t start,stop;
	uint64_t elems;
	int32_t n = 8192,iters = 0;
	elems = ((uint64_t)n * n);
	CUDA_OK(cudaMalloc((void **)&a,(size_t)(elems * sizeof(__half))),-1);
	CUDA_OK(cudaMalloc((void **)&b,(size_t)(elems * sizeof(__half))),-2);
	CUDA_OK(cudaMalloc((void **)&c,(size_t)(elems * sizeof(float))),-3);
	fill_half<<<65535,256>>>(a,elems,__float2half(0.015625f));
	fill_half<<<65535,256>>>(b,elems,__float2half(0.015625f));
	CUDA_OK(cudaGetLastError(),-4);
	CUBLAS_OK(cublasCreate(&handle),-5);
	CUDA_OK(cudaEventCreate(&start),-6);
	CUDA_OK(cudaEventCreate(&stop),-7);
	while ( total_ms < 60000.0f )
	{
		CUDA_OK(cudaEventRecord(start),-8);
		CUBLAS_OK(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,a,CUDA_R_16F,n,b,CUDA_R_16F,n,&beta,c,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),-9);
		CUDA_OK(cudaEventRecord(stop),-10);
		CUDA_OK(cudaEventSynchronize(stop),-11);
		CUDA_OK(cudaEventElapsedTime(&ms,start,stop),-12);
		total_ms += ms;
		iters++;
	}
	CUDA_OK(cudaMemcpy(&out,c,sizeof(float),cudaMemcpyDeviceToHost),-13);
	printf("GEMM_STRESS n=%d iterations=%d seconds=%.2f avg_TFLOPs=%.2f output=%.6f expected=2.000000 %s\n",n,iters,(double)total_ms / 1000.0,((2.0 * (double)n * n * n * iters) / ((double)total_ms / 1000.0) / 1.0e12),(double)out,(out > 1.999f && out < 2.001f) ? "PASS" : "FAIL");
	CUDA_OK(cudaEventDestroy(stop),-14);
	CUDA_OK(cudaEventDestroy(start),-15);
	CUBLAS_OK(cublasDestroy(handle),-16);
	CUDA_OK(cudaFree(c),-17);
	CUDA_OK(cudaFree(b),-18);
	CUDA_OK(cudaFree(a),-19);
	return((out > 1.999f && out < 2.001f) ? 0 : -20);
}

int32_t main(void)
{
	cudaDeviceProp prop;
	int32_t err;
	CUDA_OK(cudaGetDeviceProperties(&prop,0),1);
	printf("DEVICE name=%s cc=%d.%d memory_bytes=%llu\n",prop.name,prop.major,prop.minor,(unsigned long long)prop.totalGlobalMem);
	err = memory_integrity();
	if ( err < 0 )
		return(2);
	err = device_bandwidth();
	if ( err < 0 )
		return(3);
	err = pcie_bandwidth();
	if ( err < 0 )
		return(4);
	err = compute_stress();
	if ( err < 0 )
		return(5);
	printf("RTX5090_QUALIFICATION PASS\n");
	return(0);
}
