#include "tests/host_cuda/lm_host_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/scale.cuh"
#include "runtime/launch.h"
#include "tests/host_cuda/lm_host_threads.cuh"
LmHostDim3 blockDim,gridDim;
#include "inference/kernels/skinny.cuh"

#define HOST_INPUT 64u
#define HOST_OUTPUT 16u
#define HOST_EXPERTS 2u

alignas(64) static uint8_t host_weight[HOST_EXPERTS * HOST_OUTPUT * HOST_INPUT * 2u];
alignas(64) static uint16_t host_activation[2u * HOST_INPUT];
alignas(64) static uint8_t host_block_scale[4096];
alignas(64) static float host_global_scale[64];
alignas(64) static uint16_t host_output[64u * HOST_OUTPUT];
static uint32_t host_expert[HOST_EXPERTS] = {0u,1u};
static uint32_t host_packed[HOST_EXPERTS] = {0u,1u};
static uint32_t host_offset[HOST_EXPERTS + 1u] = {0u,1u,2u};
static uint32_t host_source[HOST_EXPERTS] = {0u,0u};
static int host_failures;

static void HostExpect(const char *label, int32_t status, int32_t accepted)
{
	if ( (status == LM_LAUNCH_OK) != (accepted != 0) )
	{
		printf("FAIL %s: launch returned %d, expected %s\n",label,status,accepted != 0 ? "acceptance" : "refusal");
		host_failures++;
	}
	else
		printf("ok %s\n",label);
	fflush(stdout);
}

template<class Format>
static int32_t HostExperts(LmScaleTensor scale)
{
	return(LmSkinnyExperts<Format>(host_weight,scale,host_activation,host_output,host_expert,host_packed,HOST_EXPERTS,1u,0u,HOST_INPUT,HOST_OUTPUT,0));
}

template<class Format>
static int32_t HostGrouped(LmScaleTensor scale)
{
	return(LmSkinnyGroupedExperts<Format>(host_weight,scale,host_activation,host_output,host_offset,host_source,HOST_EXPERTS,HOST_EXPERTS,0u,HOST_INPUT,HOST_OUTPUT,0));
}

static int32_t HostBf16(LmScaleTensor scale)
{
	LmSkinnyArguments args;
	memset(&args,0,sizeof(args));
	args.weight = host_weight;
	args.activation = host_activation;
	args.output_bf16 = host_output;
	args.scale = scale;
	args.rows = 1u;
	args.input_dimension = HOST_INPUT;
	args.output_dimension = HOST_OUTPUT;
	return(LmSkinnyLaunch<LmBf16Format>(&args,0));
}

int main(void)
{
	uint32_t index;
	LmScaleTensor scale;
	memset(host_weight,0x11,sizeof(host_weight));
	memset(host_block_scale,127,sizeof(host_block_scale));
	for (index=0u; index<sizeof(host_global_scale) / sizeof(host_global_scale[0]); index++)
		host_global_scale[index] = 1.0f;
	for (index=0u; index<2u * HOST_INPUT; index++)
		host_activation[index] = LmFloatToBf16(1.0f);

	HostExpect("mxfp4 experts with ue8m0 scales",HostExperts<LmMxfp4>(LmScaleTensorBlockUe8m0(host_block_scale,HOST_EXPERTS,HOST_OUTPUT,HOST_INPUT,1u,32u)),1);
	HostExpect("mxfp4 grouped with ue8m0 scales",HostGrouped<LmMxfp4>(LmScaleTensorBlockUe8m0(host_block_scale,HOST_EXPERTS,HOST_OUTPUT,HOST_INPUT,1u,32u)),1);
	HostExpect("nvfp4 experts with ue4m3 and f32 global scales",HostExperts<LmNvfp4>(LmScaleTensorBlockUe4m3F32Global(host_block_scale,host_global_scale,HOST_EXPERTS,HOST_OUTPUT,HOST_INPUT,16u)),1);
	HostExpect("bf16 dense with no scales",HostBf16(LmScaleTensorNone()),1);

	HostExpect("mxfp4 experts with the scale tensor missing",HostExperts<LmMxfp4>(LmScaleTensorNone()),0);
	HostExpect("nvfp4 experts with the scale tensor missing",HostExperts<LmNvfp4>(LmScaleTensorNone()),0);
	HostExpect("mxfp4 experts with f32 scales",HostExperts<LmMxfp4>(LmScaleTensorBlockF32(host_block_scale,HOST_EXPERTS,HOST_OUTPUT,HOST_INPUT,1u,32u)),0);
	HostExpect("nvfp4 experts with ue4m3 scales and no global scale",HostExperts<LmNvfp4>(LmScaleTensorBuild(host_block_scale,LM_SCALE_ENCODING_UE4M3,HOST_EXPERTS,HOST_OUTPUT,HOST_INPUT,1u,16u)),0);
	HostExpect("mxfp4 grouped with fewer scale groups than experts",HostGrouped<LmMxfp4>(LmScaleTensorBlockUe8m0(host_block_scale,1u,HOST_OUTPUT,HOST_INPUT,1u,32u)),0);
	HostExpect("mxfp4 experts with fewer scale rows than outputs",HostExperts<LmMxfp4>(LmScaleTensorBlockUe8m0(host_block_scale,HOST_EXPERTS,HOST_OUTPUT / 2u,HOST_INPUT,1u,32u)),0);
	HostExpect("mxfp4 experts with scales covering half the input",HostExperts<LmMxfp4>(LmScaleTensorBlockUe8m0(host_block_scale,HOST_EXPERTS,HOST_OUTPUT,HOST_INPUT / 2u,1u,32u)),0);
	HostExpect("bf16 dense with a scale tensor it would ignore",HostBf16(LmScaleTensorBlockUe8m0(host_block_scale,1u,HOST_OUTPUT,HOST_INPUT,1u,32u)),0);
	scale = LmScaleTensorBlockUe4m3F32Global(host_block_scale,host_global_scale,HOST_EXPERTS,HOST_OUTPUT,HOST_INPUT,16u);
	scale.global_data = 0;
	HostExpect("nvfp4 experts with a null global scale",HostExperts<LmNvfp4>(scale),0);

	if ( host_failures != 0 )
	{
		printf("FAIL skinny scale validation: %d cases\n",host_failures);
		return(1);
	}
	puts("PASS skinny scale validation on host threads: each format takes only its own scale encoding, with bounds covering every group, row and input, and a global scale where the encoding needs one");
	return(0);
}
