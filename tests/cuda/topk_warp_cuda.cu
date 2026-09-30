#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/topk_warp.cuh"


static __device__ __forceinline__ uint64_t ProbeReferenceMax(uint64_t value)
{
	uint32_t step;
	uint64_t other;
	for ( step = LM_WARP_LANES / 2u; step > 0u; step >>= 1u )
	{
		other = __shfl_down_sync(0xffffffffu,value,step);
		value = other > value ? other : value;
	}
	return(__shfl_sync(0xffffffffu,value,0));
}

template<uint32_t K>
static __device__ __forceinline__ float ProbeReferenceTreeSum(const float *values)
{
	constexpr uint32_t width = K <= 1u ? 1u : K <= 2u ? 2u : K <= 4u ? 4u : K <= 8u ? 8u : K <= 16u ? 16u : 32u;
	float tree[width];
	uint32_t index, step;
	for ( index = 0u; index < width; index++ )
		tree[index] = index < K ? values[index] : 0.0f;
	for ( step = width / 2u; step > 0u; step >>= 1u )
		for ( index = 0u; index < step; index++ )
			tree[index] += tree[index + step];
	return(tree[0]);
}

template<uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM>
static __device__ __forceinline__ void ProbeReferenceFinish(const float *scores, const uint16_t *logits_bf16, uint64_t base, const uint32_t *chosen, uint32_t *out_indices, float *out_values, float mixture_scale)
{
	float values[K], total;
	uint32_t index;
	for ( index = 0u; index < K; index++ )
	{
		out_indices[index] = chosen[index];
		values[index] = LmTopkScore<SCORE_TRANSFORM>(scores,logits_bf16,base + chosen[index]);
	}
	if ( out_values == 0 )
		return;
	total = ProbeReferenceTreeSum<K>(values) + 1e-20f;
	for ( index = 0u; index < K; index++ )
		out_values[index] = RENORMALISE ? (values[index] / total) * mixture_scale : values[index] * mixture_scale;
}

template<uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM>
__global__ __launch_bounds__(LM_WARP_LANES) void ProbeReferenceWarpKernel(const float *__restrict__ scores, uint32_t n, uint32_t *__restrict__ out_indices, float *__restrict__ out_values, const float *__restrict__ selection_bias, const uint16_t *__restrict__ logits_bf16, float mixture_scale)
{
	const uint32_t lane = threadIdx.x;
	const uint64_t base = (uint64_t)blockIdx.x * n;
	uint64_t candidate[LM_TOPK_WARP_PER_LANE], best, winner;
	uint32_t slot, round, chosen[K];
	for ( slot = 0u; slot < LM_TOPK_WARP_PER_LANE; slot++ )
		candidate[slot] = LmTopkWarpCandidate<SCORE_TRANSFORM>(scores,logits_bf16,selection_bias,base,lane + slot * LM_WARP_LANES,n);
	for ( round = 0u; round < K; round++ )
	{
		best = 0ull;
		for ( slot = 0u; slot < LM_TOPK_WARP_PER_LANE; slot++ )
			best = candidate[slot] > best ? candidate[slot] : best;
		winner = ProbeReferenceMax(best);
		chosen[round] = 0xffffffffu - (uint32_t)winner;
		for ( slot = 0u; slot < LM_TOPK_WARP_PER_LANE; slot++ )
			candidate[slot] = candidate[slot] == winner ? 0ull : candidate[slot];
	}
	if ( lane == 0u )
		ProbeReferenceFinish<K,RENORMALISE,SCORE_TRANSFORM>(scores,logits_bf16,base,chosen,out_indices + (uint64_t)blockIdx.x * K,out_values != 0 ? out_values + (uint64_t)blockIdx.x * K : 0,mixture_scale);
}


static int probe_failures;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

static void ProbeCase(uint32_t rows,uint32_t n,uint32_t ties,uint32_t with_bias)
{
	std::vector<float> logits((uint64_t)rows * n),bias(n),values[2];
	std::vector<uint32_t> indices[2];
	float *device_logits,*device_bias,*device_values;
	uint32_t *device_indices,index,kernel,state = 12345u + rows * 31u + n + ties;
	int failed = probe_failures;
	char label[96];
	snprintf(label,sizeof(label),"rows%u n%u ties%u bias%u",rows,n,ties,with_bias);
	for (index=0u; index<logits.size(); index++)
	{
		state = state * 1664525u + 1013904223u;
		logits[index] = ties != 0u ? (float)((state >> 20u) % 5u) - 2.0f : ((float)(state >> 8u) / 16777216.0f - 0.5f) * 12.0f;
	}
	for (index=0u; index<n; index++)
	{
		state = state * 1664525u + 1013904223u;
		bias[index] = with_bias != 0u ? (ties != 0u ? 0.25f * (float)(index % 3u) : ((float)(state >> 8u) / 16777216.0f - 0.5f)) : 0.0f;
	}
	PROBE_CUDA(cudaMalloc((void **)&device_logits,logits.size() * 4u));
	PROBE_CUDA(cudaMalloc((void **)&device_bias,n * 4u));
	PROBE_CUDA(cudaMalloc((void **)&device_values,(uint64_t)rows * 8u * 4u));
	PROBE_CUDA(cudaMalloc((void **)&device_indices,(uint64_t)rows * 8u * 4u));
	PROBE_CUDA(cudaMemcpy(device_logits,logits.data(),logits.size() * 4u,cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_bias,bias.data(),n * 4u,cudaMemcpyHostToDevice));
	for (kernel=0u; kernel<2u; kernel++)
	{
		values[kernel].resize((uint64_t)rows * 8u);
		indices[kernel].resize((uint64_t)rows * 8u);
		PROBE_CUDA(cudaMemset(device_values,0xff,(uint64_t)rows * 8u * 4u));
		PROBE_CUDA(cudaMemset(device_indices,0xff,(uint64_t)rows * 8u * 4u));
		{
			if ( kernel == 0u )
				ProbeReferenceWarpKernel<8u,true,LM_TOPK_SCORE_SIGMOID><<<rows,LM_WARP_LANES>>>(device_logits,n,device_indices,device_values,with_bias != 0u ? device_bias : 0,0,2.5f);
			else
				PROBE_CUDA((LmTopkRouteLaunch<256u,8u,true,LM_TOPK_SCORE_SIGMOID>(rows,device_logits,n,device_indices,device_values,with_bias != 0u ? device_bias : 0,0,2.5f,0)));
		}
		PROBE_CUDA(cudaDeviceSynchronize());
		PROBE_CUDA(cudaMemcpy(indices[kernel].data(),device_indices,(uint64_t)rows * 8u * 4u,cudaMemcpyDeviceToHost));
		PROBE_CUDA(cudaMemcpy(values[kernel].data(),device_values,(uint64_t)rows * 8u * 4u,cudaMemcpyDeviceToHost));
	}
	if ( indices[0] != indices[1] || memcmp(values[0].data(),values[1].data(),(uint64_t)rows * 8u * 4u) != 0 )
	{
		printf("FAIL %s: warp top-8 differs from the shuffle-reduction warp kernel\n",label);
		probe_failures++;
	}
	printf("%s %s: warp top-8 indices and weights bitwise equal to the shuffle-reduction warp kernel\n",failed == probe_failures ? "PASS" : "FAIL",label);
	cudaFree(device_logits); cudaFree(device_bias); cudaFree(device_values); cudaFree(device_indices);
}

int main(void)
{
	uint32_t rows[4] = {1u,8u,16u,64u},r;
	for (r=0u; r<4u; r++)
	{
		ProbeCase(rows[r],256u,0u,1u);
		ProbeCase(rows[r],256u,1u,1u);
	}
	ProbeCase(8u,288u,0u,1u);
	ProbeCase(8u,512u,1u,0u);
	ProbeCase(8u,500u,0u,1u);
	ProbeCase(4u,8u,1u,1u);
	if ( probe_failures != 0 )
	{
		printf("FAIL %d checks\n",probe_failures);
		return 1;
	}
	printf("PASS warp router top-8 on the device: 1..64 rows, 8..512 experts, ties and bias, bitwise equal to the shuffle-reduction warp kernel\n");
	return 0;
}
