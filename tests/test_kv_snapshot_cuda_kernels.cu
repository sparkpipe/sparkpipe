#include <cuda_runtime.h>
#include <stdint.h>

#define CUDA_BLOCK_TOKENS 64u
#define CUDA_HEAD_DIM 128u
#define CUDA_VOCAB 4096u

__device__ float CudaTokenValue(uint32_t token,uint32_t lane,uint32_t salt)
{
	uint32_t x = token * 2654435761u ^ (lane + 1u) * 40503u ^ salt * 2246822519u;
	x ^= x >> 15u;
	x *= 2246822519u;
	x ^= x >> 13u;
	return((float)(x & 0xffffu) / 65536.0f - 0.5f);
}

__global__ void CudaPrefill(float *keys,float *values,float *state,const uint32_t *physical_pages,uint32_t first,uint32_t end)
{
	uint32_t lane = threadIdx.x,token,page,slot,row;
	float accumulator;
	for (token=first; token<end; token++)
	{
		page = physical_pages[token / CUDA_BLOCK_TOKENS];
		slot = token % CUDA_BLOCK_TOKENS;
		keys[((uint64_t)page * CUDA_BLOCK_TOKENS + slot) * CUDA_HEAD_DIM + lane] = CudaTokenValue(token,lane,1u);
		values[((uint64_t)page * CUDA_BLOCK_TOKENS + slot) * CUDA_HEAD_DIM + lane] = CudaTokenValue(token,lane,2u);
		__syncthreads();
		for (row=0u; row<CUDA_HEAD_DIM; row++)
		{
			accumulator = state[row * CUDA_HEAD_DIM + lane] * 0.97f + CudaTokenValue(token,row,3u) * CudaTokenValue(token,lane,4u);
			state[row * CUDA_HEAD_DIM + lane] = accumulator;
		}
		__syncthreads();
	}
}

__global__ void CudaLogits(const float *keys,const float *values,const float *state,const uint32_t *physical_pages,uint32_t context,uint32_t next_token,float *logits)
{
	__shared__ float query[CUDA_HEAD_DIM],output[CUDA_HEAD_DIM],scores[1024];
	uint32_t lane = threadIdx.x,token,page,slot,d,v;
	float maximum = -1e30f,sum = 0.0f,accumulator;
	query[lane] = CudaTokenValue(next_token,lane,5u);
	__syncthreads();
	for (token=lane; token<context; token+=blockDim.x)
	{
		page = physical_pages[token / CUDA_BLOCK_TOKENS];
		slot = token % CUDA_BLOCK_TOKENS;
		accumulator = 0.0f;
		for (d=0u; d<CUDA_HEAD_DIM; d++)
			accumulator += query[d] * keys[((uint64_t)page * CUDA_BLOCK_TOKENS + slot) * CUDA_HEAD_DIM + d];
		scores[token] = accumulator * 0.088388f;
	}
	__syncthreads();
	for (token=0u; token<context; token++)
		maximum = fmaxf(maximum,scores[token]);
	for (token=0u; token<context; token++)
		sum += expf(scores[token] - maximum);
	accumulator = 0.0f;
	for (token=0u; token<context; token++)
	{
		page = physical_pages[token / CUDA_BLOCK_TOKENS];
		slot = token % CUDA_BLOCK_TOKENS;
		accumulator += expf(scores[token] - maximum) / sum * values[((uint64_t)page * CUDA_BLOCK_TOKENS + slot) * CUDA_HEAD_DIM + lane];
	}
	for (d=0u; d<CUDA_HEAD_DIM; d++)
		accumulator += state[lane * CUDA_HEAD_DIM + d] * query[d];
	output[lane] = accumulator;
	__syncthreads();
	for (v=lane; v<CUDA_VOCAB; v+=blockDim.x)
	{
		accumulator = 0.0f;
		for (d=0u; d<CUDA_HEAD_DIM; d++)
			accumulator += output[d] * CudaTokenValue(v,d,6u);
		logits[v] = accumulator;
	}
}


extern "C" int CudaLaunchPrefill(float *keys,float *values,float *state,const uint32_t *physical_pages,uint32_t first,uint32_t end)
{
	CudaPrefill<<<1,CUDA_HEAD_DIM>>>(keys,values,state,physical_pages,first,end);
	return(cudaDeviceSynchronize() == cudaSuccess ? 0 : -1);
}

extern "C" int CudaLaunchLogits(const float *keys,const float *values,const float *state,const uint32_t *physical_pages,uint32_t context,uint32_t next_token,float *logits)
{
	CudaLogits<<<1,CUDA_HEAD_DIM>>>(keys,values,state,physical_pages,context,next_token,logits);
	return(cudaDeviceSynchronize() == cudaSuccess ? 0 : -1);
}
