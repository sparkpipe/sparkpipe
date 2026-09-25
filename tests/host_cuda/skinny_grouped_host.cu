#include "tests/host_cuda/lm_host_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/scale.cuh"
#include "runtime/launch.h"
#include "tests/host_cuda/lm_host_threads.cuh"
LmHostDim3 blockDim,gridDim;
#include "inference/kernels/skinny.cuh"

#define HOST_EXPERTS 12u
#define HOST_TOP_K 8u
#define HOST_MAX_TOKENS 20u
#define HOST_MAX_PAIRS (HOST_MAX_TOKENS * HOST_TOP_K)
#define HOST_MAX_DIMENSION 256u

typedef struct HostRoute
{
	uint32_t pairs,expert[HOST_MAX_PAIRS],packed[HOST_MAX_PAIRS],source[HOST_MAX_PAIRS],offset[HOST_EXPERTS + 1u];
}
HostRoute;

static uint32_t host_state = 777u;

static uint32_t HostRandom(void)
{
	host_state = host_state * 1664525u + 1013904223u;
	return(host_state >> 8u);
}

static void HostRouteBuild(HostRoute *route,uint32_t tokens)
{
	uint32_t pair,expert,cursor[HOST_EXPERTS];
	route->pairs = tokens * HOST_TOP_K;
	memset(route->offset,0,sizeof(route->offset));
	for (pair=0u; pair<route->pairs; pair++)
	{
		route->expert[pair] = pair % 5u == 0u ? 3u : HostRandom() % HOST_EXPERTS;
		route->offset[route->expert[pair] + 1u]++;
	}
	for (expert=0u; expert<HOST_EXPERTS; expert++)
	{
		route->offset[expert + 1u] += route->offset[expert];
		cursor[expert] = route->offset[expert];
	}
	for (pair=0u; pair<route->pairs; pair++)
	{
		route->packed[pair] = cursor[route->expert[pair]]++;
		route->source[route->packed[pair]] = pair / HOST_TOP_K;
	}
}

static void HostGrouped(uint32_t neurons,const uint16_t *weight,const uint16_t *activation,uint16_t *output,const HostRoute *route,uint32_t packed_activation,uint32_t input,uint32_t outputs)
{
	int32_t status;
	status = neurons == 1u
		? LmSkinnyGroupedExpertsWith<LmBf16Format,1u>(weight,LmScaleTensorNone(),activation,output,route->offset,route->source,HOST_EXPERTS,route->pairs,packed_activation,input,outputs,0)
		: LmSkinnyGroupedExperts<LmBf16Format>(weight,LmScaleTensorNone(),activation,output,route->offset,route->source,HOST_EXPERTS,route->pairs,packed_activation,input,outputs,0);
	assert(status == LM_LAUNCH_OK);
}

static void HostCase(uint32_t tokens,uint32_t input,uint32_t outputs,uint32_t packed_activation)
{
	static uint16_t weight[HOST_EXPERTS * HOST_MAX_DIMENSION * HOST_MAX_DIMENSION],activation[HOST_MAX_PAIRS * HOST_MAX_DIMENSION];
	static uint16_t single[HOST_MAX_PAIRS * HOST_MAX_DIMENSION],blocked[HOST_MAX_PAIRS * HOST_MAX_DIMENSION],pairwise[HOST_MAX_PAIRS * HOST_MAX_DIMENSION];
	HostRoute route;
	uint32_t index,pair,neuron,element,row,rows;
	double total,worst = 0.0;
	HostRouteBuild(&route,tokens);
	rows = packed_activation != 0u ? route.pairs : tokens;
	for (index=0u; index<HOST_EXPERTS * outputs * input; index++)
		weight[index] = LmFloatToBf16(((float)(HostRandom() % 2001u) - 1000.0f) / 4000.0f);
	for (index=0u; index<rows * input; index++)
		activation[index] = LmFloatToBf16(((float)(HostRandom() % 2001u) - 1000.0f) / 1000.0f);
	memset(single,0xff,sizeof(single));
	memset(blocked,0xee,sizeof(blocked));
	HostGrouped(1u,weight,activation,single,&route,packed_activation,input,outputs);
	HostGrouped(LM_SKINNY_GROUPED_NEURONS,weight,activation,blocked,&route,packed_activation,input,outputs);
	if ( memcmp(single,blocked,(uint64_t)route.pairs * outputs * sizeof(uint16_t)) != 0 )
	{
		fprintf(stderr,"FAIL grouped neuron blocking changed bits tokens=%u input=%u output=%u packed=%u\n",tokens,input,outputs,packed_activation);
		exit(1);
	}
	if ( route.pairs <= LM_SKINNY_ROWS * HOST_TOP_K )
	{
		memset(pairwise,0xdd,sizeof(pairwise));
		assert(LmSkinnyExperts<LmBf16Format>(weight,LmScaleTensorNone(),activation,pairwise,route.expert,route.packed,route.pairs,HOST_TOP_K,packed_activation,input,outputs,0) == LM_LAUNCH_OK);
		if ( memcmp(pairwise,blocked,(uint64_t)route.pairs * outputs * sizeof(uint16_t)) != 0 )
		{
			fprintf(stderr,"FAIL grouped kernel differs from the per-pair kernel tokens=%u input=%u output=%u packed=%u\n",tokens,input,outputs,packed_activation);
			exit(1);
		}
	}
	for (pair=0u; pair<route.pairs; pair++)
		for (neuron=0u; neuron<outputs; neuron++)
		{
			row = packed_activation != 0u ? route.packed[pair] : pair / HOST_TOP_K;
			total = 0.0;
			for (element=0u; element<input; element++)
				total += (double)LmBf16ToFloat(weight[((uint64_t)route.expert[pair] * outputs + neuron) * input + element]) * LmBf16ToFloat(activation[(uint64_t)row * input + element]);
			total = fabs(total - LmBf16ToFloat(blocked[(uint64_t)route.packed[pair] * outputs + neuron]));
			worst = total > worst ? total : worst;
		}
	if ( worst > 2.0e-2 )
	{
		fprintf(stderr,"FAIL grouped kernel reference tokens=%u input=%u output=%u worst=%.6f\n",tokens,input,outputs,worst);
		exit(1);
	}
	printf("grouped tokens=%u pairs=%u input=%u output=%u packed=%u neuron_blocked_bitwise=yes worst_abs=%.6f\n",tokens,route.pairs,input,outputs,packed_activation,worst);
}

int main(void)
{
	uint32_t tokens;
	for (tokens=2u; tokens<=HOST_MAX_TOKENS; tokens = tokens == 2u ? 5u : tokens == 5u ? HOST_MAX_TOKENS : HOST_MAX_TOKENS + 1u)
	{
		HostCase(tokens,256u,32u,0u);
		HostCase(tokens,64u,128u,1u);
	}
	puts("PASS skinny grouped experts on host threads: neuron-blocked grouped kernel equals the one-neuron kernel and the per-pair kernel bitwise, and matches an f64 reference");
	return(0);
}
