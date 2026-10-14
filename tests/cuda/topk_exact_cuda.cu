#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/topk_exact.cuh"

#define PROBE_THREADS 256u

static int probe_failures;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

__global__ void ProbeScoresKernel(float *scores,uint64_t count,uint64_t salt,uint32_t levels,uint32_t masked_every)
{
	uint64_t index,z;
	for (index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x)
	{
		z = (index ^ salt) * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
		z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
		z ^= z >> 31u;
		if ( masked_every != 0u && (z >> 40u) % masked_every == 0u )
			scores[index] = -INFINITY;
		else if ( levels != 0u )
			scores[index] = (float)((int32_t)(z % levels) - (int32_t)(levels / 2u)) * 0.25f;
		else
			scores[index] = (float)((int64_t)(z >> 11u) - (int64_t)(1ull << 52u)) * 0x1p-50f;
	}
}

static void ProbeCase(uint32_t rows,uint32_t n,uint32_t k,uint32_t levels,uint32_t masked_every)
{
	const uint64_t count = (uint64_t)rows * n;
	std::vector<uint32_t> expected((uint64_t)rows * k),got((uint64_t)rows * k);
	float *scores;
	uint32_t *out;
	char label[128];
	snprintf(label,sizeof(label),"rows%u n%u k%u levels%u masked_every%u cluster%u",rows,n,k,levels,masked_every,LmTopkExactClusterFits(rows,n,k));
	PROBE_CUDA(cudaMalloc((void **)&scores,count * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&out,(uint64_t)rows * k * sizeof(uint32_t)));
	ProbeScoresKernel<<<128,256>>>(scores,count,(uint64_t)n * 131u + k + levels,levels,masked_every);
	PROBE_CUDA(cudaMemset(out,0xa5,(uint64_t)rows * k * sizeof(uint32_t)));
	LmTopkExactKernel<PROBE_THREADS><<<rows,PROBE_THREADS>>>(scores,n,k,out);
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(expected.data(),out,expected.size() * sizeof(uint32_t),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemset(out,0x5a,(uint64_t)rows * k * sizeof(uint32_t)));
	PROBE_CUDA(LmTopkExactRowsLaunch<PROBE_THREADS>(scores,rows,n,k,out,0));
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(got.data(),out,got.size() * sizeof(uint32_t),cudaMemcpyDeviceToHost));
	if ( expected != got )
	{
		uint64_t at = 0u;
		while ( at < got.size() && got[at] == expected[at] )
			at++;
		printf("FAIL %s: first difference at %llu (row %llu slot %llu): %u, single block %u\n",label,(unsigned long long)at,
			(unsigned long long)(at / k),(unsigned long long)(at % k),got[at],expected[at]);
		probe_failures++;
	}
	else
		printf("PASS %s: same indices in the same order as the single-block selection\n",label);
	cudaFree(scores);
	cudaFree(out);
}

static void ProbeTime(uint32_t n,uint32_t k)
{
	const uint32_t repeats = 500u;
	float *scores,elapsed[2];
	uint32_t *out,kernel,repeat;
	cudaEvent_t start,stop;
	PROBE_CUDA(cudaMalloc((void **)&scores,(uint64_t)n * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&out,(uint64_t)k * sizeof(uint32_t)));
	ProbeScoresKernel<<<128,256>>>(scores,n,7u,0u,0u);
	PROBE_CUDA(cudaEventCreate(&start));
	PROBE_CUDA(cudaEventCreate(&stop));
	for (kernel=0u; kernel<2u; kernel++)
	{
		PROBE_CUDA(cudaEventRecord(start));
		for (repeat=0u; repeat<repeats; repeat++)
			if ( kernel == 0u )
				LmTopkExactKernel<PROBE_THREADS><<<1,PROBE_THREADS>>>(scores,n,k,out);
			else
				PROBE_CUDA(LmTopkExactRowsLaunch<PROBE_THREADS>(scores,1u,n,k,out,0));
		PROBE_CUDA(cudaEventRecord(stop));
		PROBE_CUDA(cudaEventSynchronize(stop));
		PROBE_CUDA(cudaEventElapsedTime(&elapsed[kernel],start,stop));
	}
	printf("TIME rows1 n%u k%u exact top-k: single block %.2f us, cluster %.2f us per call (back to back)\n",n,k,1000.0f * elapsed[0] / repeats,1000.0f * elapsed[1] / repeats);
	cudaFree(scores);
	cudaFree(out);
}

int main(void)
{
	const uint32_t shapes[][3] = {{1u,65536u,512u},{1u,16384u,512u},{1u,8192u,2048u},{1u,2049u,2048u},{1u,600u,512u},{2u,65536u,2048u},
		{4u,3000u,2048u},{8u,40000u,64u},{64u,9000u,256u},{3u,513u,512u},{1u,65535u,1u},{65u,4096u,256u}};
	const uint32_t variants[][2] = {{0u,0u},{7u,0u},{3u,0u},{0u,5u},{5u,3u}};
	uint32_t shape,variant;
	for (shape=0u; shape<sizeof(shapes) / sizeof(shapes[0]); shape++)
		for (variant=0u; variant<sizeof(variants) / sizeof(variants[0]); variant++)
			ProbeCase(shapes[shape][0],shapes[shape][1],shapes[shape][2],variants[variant][0],variants[variant][1]);
	ProbeTime(65536u,512u);
	ProbeTime(16384u,512u);
	ProbeTime(4096u,512u);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS exact top-k on the device: the cluster selection returns the single-block indices in order, ties and -inf included\n");
	return 0;
}
