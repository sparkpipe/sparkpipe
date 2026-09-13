#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_dsv4_model.h"

#define SPARK_DSV4_EMISSION_TEST_ROWS 2u
#define SPARK_DSV4_EMISSION_TEST_LANES 2u
#define SPARK_DSV4_EMISSION_TEST_WIDTH 512u
#define SPARK_DSV4_EMISSION_TEST_CACHE_SLOTS 160u
#define SPARK_DSV4_EMISSION_TEST_CACHE_ELEMENTS \
	(SPARK_DSV4_EMISSION_TEST_LANES * SPARK_DSV4_EMISSION_TEST_CACHE_SLOTS * \
	SPARK_DSV4_EMISSION_TEST_WIDTH)

extern "C" cudaError_t SparkDsv4LaunchRmsNorm(
	cudaStream_t stream,const void *input_bf16,const void *gain_bf16,
	void *output_bf16,uint32_t row_count,uint32_t dimension,float epsilon);
extern "C" cudaError_t SparkDsv4LaunchRope(
	cudaStream_t stream,void *data_bf16,const float *freqs_f32,
	const uint64_t *row_positions,uint32_t row_count,uint32_t head_count,
	uint32_t head_dim,uint32_t rope_dim,uint32_t inverse);
extern "C" cudaError_t SparkDsv4LaunchHadamard(
	cudaStream_t stream,void *data_bf16,uint32_t vector_count,uint32_t width);
extern "C" cudaError_t SparkDsv4LaunchQuantSim(
	cudaStream_t stream,void *data_bf16,uint32_t row_count,
	uint32_t row_stride,uint32_t width,uint32_t block,uint32_t fp4);
extern "C" cudaError_t SparkDsv4LaunchCacheScatter(
	cudaStream_t stream,const void *source_bf16,const uint32_t *emitted,
	void *cache_bf16,uint64_t cache_lane_stride,
	const uint32_t *row_lane_indices,const uint64_t *row_positions,
	uint32_t row_count,uint32_t width,uint64_t base_slot,uint32_t ratio,
	uint32_t ring_slots);
extern "C" cudaError_t SparkDsv4LaunchKvEmission(
	cudaStream_t stream,void *emit_bf16,const uint32_t *emitted,
	const void *norm_weight_bf16,const float *freqs_f32,
	const uint64_t *row_emit_positions,void *cache_bf16,
	uint64_t cache_lane_stride,const uint32_t *row_lane_indices,
	const uint64_t *row_positions,uint32_t row_count,uint32_t width,
	uint64_t base_slot,uint32_t ratio,uint32_t ring_slots,uint32_t rotate);

typedef struct
{
	const char *name;
	uint32_t width;
	uint32_t ratio;
	uint32_t rotate;
	uint64_t base_slot;
	uint64_t positions[SPARK_DSV4_EMISSION_TEST_ROWS];
	uint32_t emitted[SPARK_DSV4_EMISSION_TEST_ROWS];
} SparkDsv4EmissionTestCase;

typedef struct
{
	void *reference_bf16;
	void *fused_bf16;
	void *norm_bf16;
	float *freqs_f32;
	uint64_t *emit_positions;
	uint64_t *positions;
	uint32_t *lanes;
	uint32_t *emitted;
	void *reference_cache;
	void *fused_cache;
} SparkDsv4EmissionTestDevice;

static uint16_t SparkDsv4EmissionSource[
	SPARK_DSV4_EMISSION_TEST_ROWS * SPARK_DSV4_EMISSION_TEST_WIDTH];
static uint16_t SparkDsv4EmissionNorm[SPARK_DSV4_EMISSION_TEST_WIDTH];
static float SparkDsv4EmissionFreqs[SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION / 2u];
static uint64_t SparkDsv4EmissionEmitPositions[SPARK_DSV4_EMISSION_TEST_ROWS];
static uint32_t SparkDsv4EmissionLanes[SPARK_DSV4_EMISSION_TEST_ROWS] = {0u,1u};
static uint16_t SparkDsv4EmissionReference[
	SPARK_DSV4_EMISSION_TEST_ROWS * SPARK_DSV4_EMISSION_TEST_WIDTH];
static uint16_t SparkDsv4EmissionFused[
	SPARK_DSV4_EMISSION_TEST_ROWS * SPARK_DSV4_EMISSION_TEST_WIDTH];
static uint16_t SparkDsv4EmissionReferenceCache[
	SPARK_DSV4_EMISSION_TEST_CACHE_ELEMENTS];
static uint16_t SparkDsv4EmissionFusedCache[
	SPARK_DSV4_EMISSION_TEST_CACHE_ELEMENTS];

static const SparkDsv4EmissionTestCase SparkDsv4EmissionCases[] =
{
	{"swa-always",512u,0u,0u,0u,{17u,18u},{0u,0u}},
	{"csa-attention",512u,4u,0u,128u,{2u,3u},{0u,1u}},
	{"csa-index",128u,4u,1u,0u,{6u,7u},{0u,1u}},
	{"hca-attention",512u,128u,0u,128u,{126u,127u},{0u,1u}}
};

static int32_t SparkDsv4EmissionCuda(cudaError_t error,const char *site)
{
	if ( error == cudaSuccess )
		return(0);
	fprintf(stderr,"%s: %s\n",site,cudaGetErrorString(error));
	return(-1);
}

static void SparkDsv4EmissionInitialize(const SparkDsv4EmissionTestCase *test)
{
	uint32_t element,row;
	for (element=0u; element<SPARK_DSV4_EMISSION_TEST_WIDTH; element++)
	{
		SparkDsv4EmissionNorm[element] = (uint16_t)(0x3f00u + element % 127u);
		if ( element < SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION / 2u )
			SparkDsv4EmissionFreqs[element] = (float)(element + 1u) * 0.00003125f;
	}
	for (row=0u; row<SPARK_DSV4_EMISSION_TEST_ROWS; row++)
	{
		SparkDsv4EmissionEmitPositions[row] = test->ratio == 0u ?
			test->positions[row] : test->positions[row] + 1u >= test->ratio ?
			test->positions[row] + 1u - test->ratio : 0u;
		for (element=0u; element<test->width; element++)
			SparkDsv4EmissionSource[(uint64_t)row * test->width + element] =
				test->ratio != 0u && test->emitted[row] == 0u ? 0u :
				(uint16_t)(0x3c00u + ((row * 131u + element * 17u) % 0x0200u));
	}
	memset(SparkDsv4EmissionReferenceCache,0x5a,
		sizeof(SparkDsv4EmissionReferenceCache));
	memset(SparkDsv4EmissionFusedCache,0x5a,
		sizeof(SparkDsv4EmissionFusedCache));
}

static int32_t SparkDsv4EmissionAllocate(SparkDsv4EmissionTestDevice *device)
{
	cudaError_t error;
	memset(device,0,sizeof(*device));
	error = cudaMalloc(&device->reference_bf16,sizeof(SparkDsv4EmissionSource));
	if ( error == cudaSuccess ) error = cudaMalloc(&device->fused_bf16,sizeof(SparkDsv4EmissionSource));
	if ( error == cudaSuccess ) error = cudaMalloc(&device->norm_bf16,sizeof(SparkDsv4EmissionNorm));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&device->freqs_f32,sizeof(SparkDsv4EmissionFreqs));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&device->emit_positions,sizeof(SparkDsv4EmissionEmitPositions));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&device->positions,sizeof(uint64_t) * SPARK_DSV4_EMISSION_TEST_ROWS);
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&device->lanes,sizeof(SparkDsv4EmissionLanes));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&device->emitted,sizeof(uint32_t) * SPARK_DSV4_EMISSION_TEST_ROWS);
	if ( error == cudaSuccess ) error = cudaMalloc(&device->reference_cache,sizeof(SparkDsv4EmissionReferenceCache));
	if ( error == cudaSuccess ) error = cudaMalloc(&device->fused_cache,sizeof(SparkDsv4EmissionFusedCache));
	return(SparkDsv4EmissionCuda(error,"allocate"));
}

static void SparkDsv4EmissionFree(SparkDsv4EmissionTestDevice *device)
{
	cudaFree(device->fused_cache);
	cudaFree(device->reference_cache);
	cudaFree(device->emitted);
	cudaFree(device->lanes);
	cudaFree(device->positions);
	cudaFree(device->emit_positions);
	cudaFree(device->freqs_f32);
	cudaFree(device->norm_bf16);
	cudaFree(device->fused_bf16);
	cudaFree(device->reference_bf16);
}

static cudaError_t SparkDsv4EmissionUpload(
	const SparkDsv4EmissionTestCase *test,SparkDsv4EmissionTestDevice *device,
	cudaStream_t stream)
{
	uint64_t row_bytes = (uint64_t)SPARK_DSV4_EMISSION_TEST_ROWS * test->width * sizeof(uint16_t);
	cudaError_t error;
	error = cudaMemcpyAsync(device->reference_bf16,SparkDsv4EmissionSource,row_bytes,cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(device->fused_bf16,SparkDsv4EmissionSource,row_bytes,cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(device->norm_bf16,SparkDsv4EmissionNorm,test->width * sizeof(uint16_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(device->freqs_f32,SparkDsv4EmissionFreqs,sizeof(SparkDsv4EmissionFreqs),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(device->emit_positions,SparkDsv4EmissionEmitPositions,sizeof(SparkDsv4EmissionEmitPositions),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(device->positions,test->positions,sizeof(test->positions),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(device->lanes,SparkDsv4EmissionLanes,sizeof(SparkDsv4EmissionLanes),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(device->emitted,test->emitted,sizeof(test->emitted),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(device->reference_cache,SparkDsv4EmissionReferenceCache,sizeof(SparkDsv4EmissionReferenceCache),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(device->fused_cache,SparkDsv4EmissionFusedCache,sizeof(SparkDsv4EmissionFusedCache),cudaMemcpyHostToDevice,stream);
	return(error);
}

static cudaError_t SparkDsv4EmissionRunReference(
	const SparkDsv4EmissionTestCase *test,SparkDsv4EmissionTestDevice *device,
	cudaStream_t stream)
{
	const uint32_t *emitted = test->ratio == 0u ? 0 : device->emitted;
	uint64_t lane_stride = (uint64_t)SPARK_DSV4_EMISSION_TEST_CACHE_SLOTS * test->width;
	cudaError_t error;
	error = SparkDsv4LaunchRmsNorm(stream,device->reference_bf16,
		device->norm_bf16,device->reference_bf16,SPARK_DSV4_EMISSION_TEST_ROWS,
		test->width,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchRope(stream,device->reference_bf16,device->freqs_f32,device->emit_positions,SPARK_DSV4_EMISSION_TEST_ROWS,1u,test->width,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess && test->rotate != 0u ) error = SparkDsv4LaunchHadamard(stream,device->reference_bf16,SPARK_DSV4_EMISSION_TEST_ROWS,test->width);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchQuantSim(stream,device->reference_bf16,SPARK_DSV4_EMISSION_TEST_ROWS,test->width,test->rotate != 0u ? test->width : test->width - SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,test->rotate != 0u ? SPARK_DSV4_MODEL_FP4_QUANT_BLOCK : SPARK_DSV4_MODEL_KV_QUANT_BLOCK,test->rotate);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchCacheScatter(stream,device->reference_bf16,emitted,device->reference_cache,lane_stride,device->lanes,device->positions,SPARK_DSV4_EMISSION_TEST_ROWS,test->width,test->base_slot,test->ratio,128u);
	return(error);
}

static cudaError_t SparkDsv4EmissionRunFused(
	const SparkDsv4EmissionTestCase *test,SparkDsv4EmissionTestDevice *device,
	cudaStream_t stream)
{
	const uint32_t *emitted = test->ratio == 0u ? 0 : device->emitted;
	uint64_t lane_stride = (uint64_t)SPARK_DSV4_EMISSION_TEST_CACHE_SLOTS * test->width;
	return(SparkDsv4LaunchKvEmission(stream,device->fused_bf16,emitted,
		device->norm_bf16,device->freqs_f32,device->emit_positions,
		device->fused_cache,lane_stride,device->lanes,device->positions,
		SPARK_DSV4_EMISSION_TEST_ROWS,test->width,test->base_slot,test->ratio,
		128u,test->rotate));
}

static int32_t SparkDsv4EmissionCompare(const SparkDsv4EmissionTestCase *test)
{
	uint64_t cache_elements = (uint64_t)SPARK_DSV4_EMISSION_TEST_LANES * SPARK_DSV4_EMISSION_TEST_CACHE_SLOTS * test->width;
	uint64_t row_elements = (uint64_t)SPARK_DSV4_EMISSION_TEST_ROWS * test->width;
	if ( memcmp(SparkDsv4EmissionReference,SparkDsv4EmissionFused,row_elements * sizeof(uint16_t)) != 0 )
	{
		fprintf(stderr,"%s: emitted BF16 mismatch\n",test->name);
		return(-1);
	}
	if ( memcmp(SparkDsv4EmissionReferenceCache,SparkDsv4EmissionFusedCache,cache_elements * sizeof(uint16_t)) != 0 )
	{
		fprintf(stderr,"%s: cache BF16 mismatch\n",test->name);
		return(-2);
	}
	return(0);
}

static int32_t SparkDsv4EmissionRunCase(
	const SparkDsv4EmissionTestCase *test,SparkDsv4EmissionTestDevice *device,
	cudaStream_t stream)
{
	uint64_t row_bytes = (uint64_t)SPARK_DSV4_EMISSION_TEST_ROWS * test->width * sizeof(uint16_t);
	cudaError_t error;
	SparkDsv4EmissionInitialize(test);
	error = SparkDsv4EmissionUpload(test,device,stream);
	if ( error == cudaSuccess ) error = SparkDsv4EmissionRunReference(test,device,stream);
	if ( error == cudaSuccess ) error = SparkDsv4EmissionRunFused(test,device,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(SparkDsv4EmissionReference,device->reference_bf16,row_bytes,cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(SparkDsv4EmissionFused,device->fused_bf16,row_bytes,cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(SparkDsv4EmissionReferenceCache,device->reference_cache,sizeof(SparkDsv4EmissionReferenceCache),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(SparkDsv4EmissionFusedCache,device->fused_cache,sizeof(SparkDsv4EmissionFusedCache),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess ) error = cudaStreamSynchronize(stream);
	if ( SparkDsv4EmissionCuda(error,test->name) < 0 )
		return(-1);
	return(SparkDsv4EmissionCompare(test));
}

int main(void)
{
	SparkDsv4EmissionTestDevice device;
	cudaStream_t stream;
	uint32_t index;
	int32_t status;
	if ( SparkDsv4EmissionCuda(cudaStreamCreateWithFlags(&stream,
		cudaStreamNonBlocking),"stream") < 0 )
		return(2);
	if ( SparkDsv4EmissionAllocate(&device) < 0 )
	{
		SparkDsv4EmissionFree(&device);
		cudaStreamDestroy(stream);
		return(2);
	}
	status = 0;
	for (index=0u; status==0 && index<sizeof(SparkDsv4EmissionCases) /
		sizeof(SparkDsv4EmissionCases[0]); index++)
		status = SparkDsv4EmissionRunCase(&SparkDsv4EmissionCases[index],
			&device,stream);
	SparkDsv4EmissionFree(&device);
	cudaStreamDestroy(stream);
	printf("{\"cases\":%u,\"boundary_modes\":\"swa-always,csa-4,hca-128\","
		"\"bf16_bitwise\":\"%s\",\"cache_bitwise\":\"%s\"}\n",
		(uint32_t)(sizeof(SparkDsv4EmissionCases) /
			sizeof(SparkDsv4EmissionCases[0])),status == 0 ? "pass" : "fail",
		status == 0 ? "pass" : "fail");
	return(status == 0 ? 0 : 1);
}
