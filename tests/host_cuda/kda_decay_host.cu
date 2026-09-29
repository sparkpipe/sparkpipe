#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <stdlib.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float state_s[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"

#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "inference/kernels/linear_attn.cuh"

#define MAXIMUM_CASES 64u

static int32_t ParseFloat(const char *text,float *value)
{
	char *end;
	*value = strtof(text,&end);
	return(end != text && *end == '\0' ? 0 : -1);
}

int main(int argc,char **argv)
{
	uint16_t logit[MAXIMUM_CASES];
	float bias[MAXIMUM_CASES],scale[MAXIMUM_CASES],retention[MAXIMUM_CASES],minimum,value;
	uint32_t count,index;
	if ( argc < 5 || argc % 3 != 2 || (uint32_t)(argc - 2) / 3u > MAXIMUM_CASES || ParseFloat(argv[1],&minimum) != 0 )
	{
		fprintf(stderr,"usage: %s MINIMUM_LOG_DECAY (LOGIT HEAD_LOG_SCALE CHANNEL_BIAS)...\n",argv[0]);
		return(2);
	}
	count = (uint32_t)(argc - 2) / 3u;
	for (index=0u; index<count; index++)
	{
		if ( ParseFloat(argv[2u + 3u * index],&value) != 0 || ParseFloat(argv[3u + 3u * index],&scale[index]) != 0 || ParseFloat(argv[4u + 3u * index],&bias[index]) != 0 )
		{
			fprintf(stderr,"case %u is not three numbers\n",index);
			return(2);
		}
		logit[index] = LmFloatToBf16(value);
		retention[index] = -1.0f;
	}
	LM_HOST_LAUNCH(dim3(1u, count),
		(LmBoundedDecayKernel<1u, 1u>(logit, bias, scale, retention, count, minimum, 1u)));
	for (index=0u; index<count; index++)
		printf("%.9g %.9g\n",LmBf16ToFloat(logit[index]),retention[index]);
	return(0);
}
