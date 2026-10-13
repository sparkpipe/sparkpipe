#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/index_select.cuh"

#define PROBE_ROW_WORDS 4u
#define PROBE_CANARY_WORDS 8u
#define PROBE_CANARY 0xa5a5a5a5u
#define PROBE_K 16u

static int probe_failures;

#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n",#call,cudaGetErrorString(probe_error)); exit(1); } } while (0)

static uint32_t ProbeFloatBits(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return(bits);
}

static void ProbeCase(uint32_t bound,uint32_t expect_refusal)
{
	const uint32_t keep = bound,bitmap_words = 2u * PROBE_ROW_WORDS + PROBE_CANARY_WORDS;
	std::vector<uint2> candidates(keep);
	std::vector<uint32_t> bitmaps(bitmap_words,0u),selected(PROBE_K,0u);
	uint32_t zero = 0u,index;
	LmFrameError error;
	SparkKvShard shard;
	uint2 *device_candidates;
	uint32_t *device_sequence,*device_context,*device_bitmaps,*device_selected;
	LmFrameError *device_error;
	for (index = 0u; index < keep; ++index)
		candidates[index] = make_uint2(ProbeFloatBits(1.0f),index);
	for (index = 2u * PROBE_ROW_WORDS; index < bitmap_words; ++index)
		bitmaps[index] = PROBE_CANARY;
	LmFrameErrorReset(&error);
	shard.degree = 1u;
	shard.rank = 0u;
	shard.grain = 1u;
	PROBE_CUDA(cudaMalloc((void **)&device_candidates,keep * sizeof(uint2)));
	PROBE_CUDA(cudaMalloc((void **)&device_sequence,sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_context,sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_bitmaps,bitmap_words * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_selected,PROBE_K * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&device_error,sizeof(LmFrameError)));
	PROBE_CUDA(cudaMemcpy(device_candidates,candidates.data(),keep * sizeof(uint2),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_sequence,&zero,sizeof(uint32_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_context,&bound,sizeof(uint32_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemcpy(device_bitmaps,bitmaps.data(),bitmap_words * sizeof(uint32_t),cudaMemcpyHostToDevice));
	PROBE_CUDA(cudaMemset(device_selected,0xff,PROBE_K * sizeof(uint32_t)));
	PROBE_CUDA(cudaMemcpy(device_error,&error,sizeof(LmFrameError),cudaMemcpyHostToDevice));
	PROBE_CUDA(LmIndexSelectMergeLaunch<LM_INDEX_SELECT_THREADS>(device_candidates,keep,shard,keep,device_sequence,device_context,0,
		PROBE_K,PROBE_K,1u,PROBE_ROW_WORDS,device_bitmaps,device_selected,device_error,0));
	PROBE_CUDA(cudaDeviceSynchronize());
	PROBE_CUDA(cudaMemcpy(bitmaps.data(),device_bitmaps,bitmap_words * sizeof(uint32_t),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(selected.data(),device_selected,PROBE_K * sizeof(uint32_t),cudaMemcpyDeviceToHost));
	PROBE_CUDA(cudaMemcpy(&error,device_error,sizeof(LmFrameError),cudaMemcpyDeviceToHost));
	for (index = 2u * PROBE_ROW_WORDS; index < bitmap_words; ++index)
		if ( bitmaps[index] != PROBE_CANARY )
		{
			printf("FAIL bound %u: the merge wrote 0x%08x into word %u past the row's bitmaps\n",bound,bitmaps[index],index);
			probe_failures++;
			break;
		}
	if ( expect_refusal != 0u )
	{
		if ( error.error_code != LM_FRAME_ERROR_SPARSE_INDEX_OUT_OF_RANGE || error.position != bound || error.page != PROBE_ROW_WORDS * 32u )
		{
			printf("FAIL bound %u: a context past the %u-position bitmap is not reported (error %u position %u page %u)\n",
				bound,PROBE_ROW_WORDS * 32u,error.error_code,error.position,error.page);
			probe_failures++;
		}
	}
	else
	{
		if ( error.error_code != LM_FRAME_ERROR_NONE )
		{
			printf("FAIL bound %u: a context inside the bitmap reported error %u\n",bound,error.error_code);
			probe_failures++;
		}
		for (index = 0u; index < PROBE_K; ++index)
			if ( selected[index] >= bound )
			{
				printf("FAIL bound %u: selection %u is %u\n",bound,index,selected[index]);
				probe_failures++;
				break;
			}
	}
	cudaFree(device_candidates);
	cudaFree(device_sequence);
	cudaFree(device_context);
	cudaFree(device_bitmaps);
	cudaFree(device_selected);
	cudaFree(device_error);
}

int main(void)
{
	ProbeCase(100u,0u);
	ProbeCase(PROBE_ROW_WORDS * 32u,0u);
	ProbeCase(192u,1u);
	if ( probe_failures != 0 )
	{
		printf("FAIL index_select_bound_cuda: %d failures\n",probe_failures);
		return(1);
	}
	printf("PASS index_select_bound_cuda: a context inside the bitmap selects in range and leaves the next words alone; one past it is reported and writes nothing\n");
	return(0);
}
