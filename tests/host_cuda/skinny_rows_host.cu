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

#define HOST_MAX_ROWS 128u
#define HOST_MAX_INPUT 4096u
#define HOST_MAX_STRIDE 1040u
#define HOST_PAD 3u

typedef struct HostShape
{
	uint32_t input,output,f32,offset;
}
HostShape;

typedef struct HostBuffers
{
	alignas(16) uint16_t weight[HOST_MAX_STRIDE * HOST_MAX_INPUT];
	alignas(16) uint16_t activation[HOST_MAX_ROWS * HOST_MAX_INPUT];
	alignas(16) uint16_t single16[HOST_MAX_ROWS * HOST_MAX_STRIDE];
	alignas(16) uint16_t rows16[HOST_MAX_ROWS * HOST_MAX_STRIDE];
	alignas(16) float single32[HOST_MAX_ROWS * HOST_MAX_STRIDE];
	alignas(16) float rows32[HOST_MAX_ROWS * HOST_MAX_STRIDE];
}
HostBuffers;

static HostBuffers host;
static uint32_t host_state = 20260928u;

static uint32_t HostRandom(void)
{
	host_state = host_state * 1664525u + 1013904223u;
	return(host_state >> 8u);
}

static uint16_t HostValue(float scale)
{
	return(LmFloatToBf16((((float)(HostRandom() % 2049u)) - 1024.0f) / 1024.0f * scale));
}

static void HostFill(const HostShape *shape)
{
	uint32_t index,element;
	for (index=0u; index<shape->output * shape->input; index++)
		host.weight[index] = HostValue(0.0625f);
	for (element=0u; element<shape->input; element++)
	{
		host.weight[element] = 0x0000u;
		host.weight[shape->input + element] = 0x8000u;
	}
	for (index=0u; index<HOST_MAX_ROWS * shape->input; index++)
		host.activation[index] = HostValue(4.0f);
	for (element=0u; element<shape->input; element++)
	{
		host.activation[element] = 0x0000u;
		host.activation[shape->input + element] = 0x8000u;
		host.activation[2u * shape->input + element] = (element & 1u) != 0u ? 0x0001u : 0x8001u;
	}
}

static uint32_t HostStride(const HostShape *shape)
{
	return(shape->output + shape->offset + HOST_PAD);
}

static void HostSingles(const HostShape *shape)
{
	uint32_t row,stride = HostStride(shape);
	memset(host.single16,0xa5,sizeof(host.single16));
	memset(host.single32,0xa5,sizeof(host.single32));
	for (row=0u; row<HOST_MAX_ROWS; row++)
		assert(LmSkinnyDense<LmBf16Format>(host.weight,host.activation + (uint64_t)row * shape->input,shape->f32 != 0u ? 0 : host.single16 + (uint64_t)row * stride,shape->f32 != 0u ? host.single32 + (uint64_t)row * stride : 0,1u,shape->input,shape->output,stride,shape->offset,0) == LM_LAUNCH_OK);
}

static void HostReference(const HostShape *shape,uint32_t rows)
{
	uint32_t row,neuron,element,stride = HostStride(shape);
	double total,magnitude,term,actual;
	for (row=0u; row<rows; row++)
		for (neuron=0u; neuron<shape->output; neuron++)
		{
			total = magnitude = 0.0;
			for (element=0u; element<shape->input; element++)
			{
				term = (double)LmBf16ToFloat(host.weight[(uint64_t)neuron * shape->input + element]) * LmBf16ToFloat(host.activation[(uint64_t)row * shape->input + element]);
				total += term;
				magnitude += fabs(term);
			}
			actual = shape->f32 != 0u ? host.rows32[(uint64_t)row * stride + shape->offset + neuron] : LmBf16ToFloat(host.rows16[(uint64_t)row * stride + shape->offset + neuron]);
			if ( !(fabs(actual - total) <= ldexp(fabs(total),-7) + 1e-5 * magnitude + 1e-30) )
			{
				fprintf(stderr,"FAIL rows kernel reference input=%u output=%u rows=%u row=%u neuron=%u actual=%.9g expected=%.9g\n",shape->input,shape->output,rows,row,neuron,actual,total);
				exit(1);
			}
		}
}

static void HostCase(const HostShape *shape,uint32_t rows)
{
	uint32_t stride = HostStride(shape);
	uint64_t bytes = (uint64_t)rows * stride * (shape->f32 != 0u ? sizeof(float) : sizeof(uint16_t));
	int32_t status;
	memset(host.rows16,0xa5,sizeof(host.rows16));
	memset(host.rows32,0xa5,sizeof(host.rows32));
	status = LmSkinnyDenseRows<LmBf16Format>(host.weight,host.activation,shape->f32 != 0u ? 0 : host.rows16,shape->f32 != 0u ? host.rows32 : 0,rows,shape->input,shape->output,stride,shape->offset,0);
	if ( status != LM_LAUNCH_OK )
	{
		fprintf(stderr,"FAIL rows entry status=%d input=%u output=%u rows=%u\n",(int)status,shape->input,shape->output,rows);
		exit(1);
	}
	if ( memcmp(shape->f32 != 0u ? (const void *)host.rows32 : (const void *)host.rows16,shape->f32 != 0u ? (const void *)host.single32 : (const void *)host.single16,bytes) != 0 )
	{
		fprintf(stderr,"FAIL %u-row output differs from %u serial one-row launches input=%u output=%u f32=%u offset=%u\n",rows,rows,shape->input,shape->output,shape->f32,shape->offset);
		exit(1);
	}
	if ( rows == HOST_MAX_ROWS || rows == 9u )
		HostReference(shape,rows);
}

static void HostShapeRun(const HostShape *shape,const uint32_t *rows,uint32_t count)
{
	uint32_t index;
	HostFill(shape);
	HostSingles(shape);
	for (index=0u; index<count; index++)
		HostCase(shape,rows[index]);
	printf("skinny rows input=%u output=%u f32=%u offset=%u row_counts=%u bitwise_equal_to_serial_one_row=yes zero_and_negative_zero_rows=yes\n",shape->input,shape->output,shape->f32,shape->offset,count);
}

static void HostRefusals(void)
{
	assert(LmSkinnyDenseRows<LmBf16Format>(host.weight + 1,host.activation,host.rows16,0,9u,128u,32u,0u,0u,0) == LM_LAUNCH_ERR_SHAPE);
	assert(LmSkinnyDenseRows<LmBf16Format>(host.weight,host.activation,host.rows16,0,9u,100u,32u,0u,0u,0) == LM_LAUNCH_ERR_SHAPE);
	assert(LmSkinnyDenseRows<LmBf16Format>(host.weight,host.activation,host.rows16,host.rows32,9u,128u,32u,0u,0u,0) == LM_LAUNCH_ERR_SHAPE);
	assert(LmSkinnyDenseRows<LmBf16Format>(host.weight,host.activation,host.rows16,0,9u,128u,32u,40u,9u,0) == LM_LAUNCH_ERR_OUTPUT);
	assert(LmSkinnyDenseRows<LmBf16Format>(host.weight,host.activation,host.rows16,0,0u,128u,32u,0u,0u,0) == LM_LAUNCH_ERR_SHAPE);
	puts("skinny rows refusals: misaligned weight, ragged input, two outputs, output overflow and zero rows fail with a named status above eight rows");
}

int main(void)
{
	static const uint32_t all_rows[] = {1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,11u,12u,13u,14u,15u,16u,17u,18u,23u,24u,25u,31u,32u,33u,34u,40u,48u,63u,64u,65u,66u,96u,127u,128u};
	static const uint32_t some_rows[] = {1u,8u,9u,16u,17u,33u,64u,65u,128u};
	static const HostShape shapes[] =
	{
		{128u,288u,0u,0u},
		{512u,64u,1u,0u},
		{96u,37u,0u,5u},
		{1536u,32u,0u,0u},
		{4096u,16u,1u,0u},
		{256u,1030u,0u,3u}
	};
	uint32_t index;
	HostShapeRun(&shapes[0],all_rows,sizeof(all_rows) / sizeof(all_rows[0]));
	for (index=1u; index<sizeof(shapes) / sizeof(shapes[0]); index++)
		HostShapeRun(&shapes[index],some_rows,sizeof(some_rows) / sizeof(some_rows[0]));
	HostRefusals();
	puts("PASS skinny rows on host threads: every row of an n-row dense projection equals n serial one-row launches bitwise for n=1..128, lanes 8/16/32, one, two and four neurons per thread, bf16 and f32 outputs, strided column-offset outputs, and +0/-0/denormal rows");
	return(0);
}
