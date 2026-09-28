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

#define HOST_EXPERTS 6u
#define HOST_TOP_K 4u
#define HOST_TOKENS 2u
#define HOST_PAIRS (HOST_TOKENS * HOST_TOP_K)
#define HOST_MAX_INPUT 1024u
#define HOST_MAX_OUTPUT 96u

static const double host_e2m1[16] = {0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0};
static uint32_t host_state = 2026u;

static uint32_t HostRandom(void)
{
	host_state = host_state * 1664525u + 1013904223u;
	return(host_state >> 8u);
}

static double HostWeight(const uint8_t *payload, const uint8_t *scale, uint32_t expert, uint32_t row, uint32_t column, uint32_t input, uint32_t output)
{
	uint8_t byte = payload[((uint64_t)expert * output + row) * (input / 2u) + column / 2u];
	uint8_t code = (column & 1u) != 0u ? (uint8_t)(byte >> 4u) : (uint8_t)(byte & 15u);
	return(host_e2m1[code] * ldexp(1.0,(int)scale[((uint64_t)expert * output + row) * (input / 32u) + column / 32u] - 127));
}

static void HostDecoders(void)
{
	uint32_t code;
	for (code=0u; code<16u; code++)
		if ( (double)LmE2m1ToFloatPure(code) != host_e2m1[code] || signbit(LmE2m1ToFloatPure(code)) != signbit(host_e2m1[code]) )
		{
			fprintf(stderr,"FAIL e2m1 code %u decodes to %g, OCP MX value %g\n",code,LmE2m1ToFloatPure(code),host_e2m1[code]);
			exit(1);
		}
	for (code=0u; code<255u; code++)
		if ( (double)LmUe8m0ToFloat((uint8_t)code) != ldexp(1.0,(int)code - 127) )
		{
			fprintf(stderr,"FAIL ue8m0 code %u decodes to %g\n",code,LmUe8m0ToFloat((uint8_t)code));
			exit(1);
		}
	if ( !isnan(LmUe8m0ToFloat(0xffu)) )
	{
		fprintf(stderr,"FAIL ue8m0 code 255 must decode to NaN\n");
		exit(1);
	}
}

static void HostCase(uint32_t input, uint32_t output)
{
	static uint8_t payload[HOST_EXPERTS * HOST_MAX_OUTPUT * HOST_MAX_INPUT / 2u];
	static uint8_t scale[HOST_EXPERTS * HOST_MAX_OUTPUT * HOST_MAX_INPUT / 32u];
	static uint16_t activation[HOST_TOKENS * HOST_MAX_INPUT];
	static uint16_t pairwise[HOST_PAIRS * HOST_MAX_OUTPUT],grouped[HOST_PAIRS * HOST_MAX_OUTPUT];
	uint32_t expert[HOST_PAIRS],packed[HOST_PAIRS],source[HOST_PAIRS],offset[HOST_EXPERTS + 1u],cursor[HOST_EXPERTS];
	uint32_t index,pair,row,column;
	double total,error,worst = 0.0,magnitude = 0.0;
	LmScaleTensor scales;
	for (index=0u; index<HOST_EXPERTS * output * input / 2u; index++)
		payload[index] = (uint8_t)HostRandom();
	for (index=0u; index<HOST_EXPERTS * output * input / 32u; index++)
		scale[index] = (uint8_t)(118u + HostRandom() % 12u);
	for (index=0u; index<HOST_TOKENS * input; index++)
		activation[index] = LmFloatToBf16(((float)(HostRandom() % 2001u) - 1000.0f) / 1000.0f);
	memset(offset,0,sizeof(offset));
	for (pair=0u; pair<HOST_PAIRS; pair++)
	{
		expert[pair] = (pair * 5u + 1u) % HOST_EXPERTS;
		offset[expert[pair] + 1u]++;
	}
	for (index=0u; index<HOST_EXPERTS; index++)
	{
		offset[index + 1u] += offset[index];
		cursor[index] = offset[index];
	}
	for (pair=0u; pair<HOST_PAIRS; pair++)
	{
		packed[pair] = cursor[expert[pair]]++;
		source[packed[pair]] = pair / HOST_TOP_K;
	}
	scales = LmScaleTensorBlockUe8m0(scale,HOST_EXPERTS,output,input,1u,32u);
	memset(pairwise,0xdd,sizeof(pairwise));
	memset(grouped,0xee,sizeof(grouped));
	assert(LmSkinnyExperts<LmMxfp4>(payload,scales,activation,pairwise,expert,packed,HOST_PAIRS,HOST_TOP_K,0u,input,output,0) == LM_LAUNCH_OK);
	assert(LmSkinnyGroupedExperts<LmMxfp4>(payload,scales,activation,grouped,offset,source,HOST_EXPERTS,HOST_PAIRS,0u,input,output,0) == LM_LAUNCH_OK);
	if ( memcmp(pairwise,grouped,(uint64_t)HOST_PAIRS * output * sizeof(uint16_t)) != 0 )
	{
		fprintf(stderr,"FAIL mxfp4 grouped kernel differs from the per-pair kernel input=%u output=%u\n",input,output);
		exit(1);
	}
	for (pair=0u; pair<HOST_PAIRS; pair++)
		for (row=0u; row<output; row++)
		{
			total = 0.0;
			for (column=0u; column<input; column++)
				total += HostWeight(payload,scale,expert[pair],row,column,input,output) * LmBf16ToFloat(activation[(uint64_t)(pair / HOST_TOP_K) * input + column]);
			error = fabs(total - LmBf16ToFloat(pairwise[(uint64_t)packed[pair] * output + row]));
			worst = error > worst ? error : worst;
			magnitude = fabs(total) > magnitude ? fabs(total) : magnitude;
		}
	if ( worst > magnitude * 8.0e-3 )
	{
		fprintf(stderr,"FAIL mxfp4 skinny vs f64 dequantised reference input=%u output=%u worst=%.6f peak=%.3f\n",input,output,worst,magnitude);
		exit(1);
	}
	printf("mxfp4 input=%u output=%u pairs=%u grouped_equals_pairwise=yes worst_abs=%.6f peak=%.3f\n",input,output,HOST_PAIRS,worst,magnitude);
}

int main(void)
{
	uint8_t bad_scale[64];
	uint16_t activation[64],output[8];
	uint8_t weight[256];
	uint32_t zero = 0u;
	HostDecoders();
	HostCase(1024u,64u);
	HostCase(256u,96u);
	HostCase(64u,32u);
	memset(bad_scale,127,sizeof(bad_scale));
	memset(weight,0,sizeof(weight));
	memset(activation,0,sizeof(activation));
	if ( LmSkinnyExperts<LmMxfp4>(weight,LmScaleTensorBlockUe8m0(bad_scale,1u,4u,64u,1u,16u),activation,output,&zero,&zero,1u,1u,0u,64u,4u,0) == LM_LAUNCH_OK )
	{
		fprintf(stderr,"FAIL mxfp4 skinny accepted a 16-element scale group\n");
		exit(1);
	}
	if ( LmSkinnyExperts<LmMxfp4>(weight,LmScaleTensorBlockUe8m0(bad_scale,1u,4u,48u,1u,32u),activation,output,&zero,&zero,1u,1u,0u,48u,4u,0) == LM_LAUNCH_OK )
	{
		fprintf(stderr,"FAIL mxfp4 skinny accepted an input that is not whole 32-element chunks\n");
		exit(1);
	}
	puts("PASS skinny mxfp4 on host threads: OCP e2m1/ue8m0 decode, grouped equals per-pair bitwise, both match an f64 dequantised reference, bad shapes refused");
	return(0);
}
