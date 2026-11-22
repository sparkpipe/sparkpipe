#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "sparkpipe/spark_gpu.h"
#include "tests/nvrm/tensor_map_gemms.cuh"

#define ADAPTER_HOT_EXPERTS 16u
#define ADAPTER_WEIGHT_BF16 0u
#define ADAPTER_WEIGHT_BYTES 1u
#define ADAPTER_WEIGHT_CELLS 2u
#define ADAPTER_OUTPUT_BF16 0u
#define ADAPTER_OUTPUT_F32 1u
#define ADAPTER_OUTPUT_ACCUMULATE 2u

typedef struct AdapterExperts
{
	const char *adapter;
	uint32_t hidden;
	uint32_t up_output;
	uint32_t down_input;
	uint32_t experts;
	uint32_t top_k;
} AdapterExperts;

typedef struct AdapterRoutes
{
	uint32_t tokens;
	uint32_t top_k;
	uint32_t experts;
	uint32_t packed_rows;
	std::vector<uint32_t> hot;
	std::vector<uint32_t> offsets;
	std::vector<uint32_t> source;
} AdapterRoutes;

static void AdapterFail(const char *what, uint64_t bytes)
{
	printf("FAIL %s (%llu bytes): %s\n",what,(unsigned long long)bytes,SparkGpuGetErrorString(SparkGpuGetLastError()));
	exit(1);
}

static void *AdapterAllocate(uint64_t bytes)
{
	void *device = 0;
	if ( SparkGpuMalloc(&device,bytes) != SPARK_GPU_SUCCESS || SparkGpuMemset(device,0,bytes) != SPARK_GPU_SUCCESS )
		AdapterFail("allocation",bytes);
	return(device);
}

static void AdapterHot(AdapterRoutes &routes, uint32_t experts)
{
	const uint32_t hot = experts < ADAPTER_HOT_EXPERTS ? experts : ADAPTER_HOT_EXPERTS;
	routes.experts = experts;
	routes.hot.clear();
	for ( uint32_t index = 0u; index < hot; ++index )
		routes.hot.push_back(hot > 1u ? (uint32_t)((uint64_t)index * (experts - 1u) / (hot - 1u)) : 0u);
}

static void AdapterRoute(AdapterRoutes &routes, uint32_t tokens, uint32_t top_k)
{
	const uint32_t hot = (uint32_t)routes.hot.size();
	std::vector<std::vector<uint32_t>> rows(routes.experts);
	routes.tokens = tokens;
	routes.top_k = top_k;
	routes.packed_rows = tokens * top_k;
	for ( uint32_t token = 0u; token < tokens; ++token )
		for ( uint32_t slot = 0u; slot < top_k; ++slot )
			rows[routes.hot[(token + slot) % hot]].push_back(token);
	routes.offsets.assign(1u,0u);
	routes.source.clear();
	for ( uint32_t expert = 0u; expert < routes.experts; ++expert )
	{
		routes.offsets.push_back(routes.offsets.back() + (uint32_t)rows[expert].size());
		routes.source.insert(routes.source.end(),rows[expert].begin(),rows[expert].end());
	}
}

static void *AdapterWeights(const AdapterRoutes &routes, uint64_t expert_bytes, uint64_t salt, uint32_t mode, uint32_t cell_row_bytes)
{
	uint8_t *device = (uint8_t *)AdapterAllocate(expert_bytes * routes.experts);
	std::vector<uint8_t> host(expert_bytes);
	for ( uint32_t expert : routes.hot )
	{
		const uint64_t seed = ((uint64_t)expert << 40u) ^ (salt << 56u);
		if ( mode == ADAPTER_WEIGHT_BF16 )
			for ( uint64_t index = 0u; index < expert_bytes / 2u; ++index )
			{
				const uint16_t value = BuildBf16(index ^ seed,salt);
				memcpy(&host[index * 2u],&value,sizeof(value));
			}
		else
			for ( uint64_t index = 0u; index < expert_bytes; ++index )
				host[index] = (uint8_t)BuildHash(index ^ seed);
		if ( mode == ADAPTER_WEIGHT_CELLS )
			for ( uint64_t cell = 0u; cell < expert_bytes / (17u * cell_row_bytes); ++cell )
				memset(&host[cell * 17u * cell_row_bytes + 16u * cell_row_bytes],(int)(122u + (cell + expert) % 5u),cell_row_bytes);
		if ( SparkGpuMemcpy(device + (uint64_t)expert * expert_bytes,host.data(),expert_bytes,SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
			AdapterFail("weight upload",expert_bytes);
	}
	return(device);
}

static void *AdapterActivation(uint64_t rows, uint32_t columns, uint64_t salt)
{
	std::vector<uint16_t> host(rows * columns);
	for ( uint64_t index = 0u; index < host.size(); ++index )
		host[index] = BuildBf16(index,salt);
	return(BuildUpload(host.data(),host.size() * 2u));
}

static void AdapterEmit(const char *what, SparkGpuStream stream, int32_t status, const void *device, uint64_t bytes)
{
	std::vector<uint8_t> result(bytes);
	const SparkGpuError error = BuildFinish(stream,status);
	if ( SparkGpuMemcpy(result.data(),device,bytes,SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
		AdapterFail("output read",bytes);
	BuildEmit(what,result.data(),bytes,error);
}

typedef struct AdapterGrouped
{
	uint32_t *offsets;
	uint32_t *source;
	uint32_t *prefix;
	void *output;
	void *activation;
} AdapterGrouped;

static AdapterGrouped AdapterGroupedSetup(const AdapterRoutes &routes, uint32_t indirect, uint32_t input, uint32_t output, uint64_t salt)
{
	AdapterGrouped grouped;
	grouped.offsets = (uint32_t *)BuildUpload(routes.offsets.data(),routes.offsets.size() * 4u);
	grouped.source = indirect != 0u ? (uint32_t *)BuildUpload(routes.source.data(),routes.source.size() * 4u) : 0;
	grouped.prefix = (uint32_t *)AdapterAllocate(((uint64_t)routes.experts + 1u) * 64u * sizeof(uint32_t));
	grouped.output = AdapterAllocate((uint64_t)routes.packed_rows * output * 2u);
	grouped.activation = AdapterActivation(indirect != 0u ? routes.tokens : routes.packed_rows,input,salt);
	return(grouped);
}

static void AdapterGroupedFree(AdapterGrouped &grouped)
{
	SparkGpuFree(grouped.offsets);
	SparkGpuFree(grouped.source);
	SparkGpuFree(grouped.prefix);
	SparkGpuFree(grouped.output);
	SparkGpuFree(grouped.activation);
}

static LmGemmArguments AdapterGroupedArguments(const AdapterRoutes &routes, const AdapterGrouped &grouped, LmScaleTensor scale_b)
{
	LmGemmArguments gemm;
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = scale_b;
	gemm.group_row_offset = grouped.offsets;
	gemm.group_tile_prefix = grouped.prefix;
	gemm.source_row_map = grouped.source;
	gemm.source_row_count = grouped.source != 0 ? routes.tokens : 0u;
	gemm.output_bf16 = grouped.output;
	return(gemm);
}

static void AdapterTokens(uint32_t experts, uint32_t top_k, uint32_t tokens[3])
{
	tokens[0] = 8u * experts / top_k;
	tokens[1] = 16u * experts / top_k;
	tokens[2] = 48u * experts / top_k;
	if ( tokens[0] == 0u )
		tokens[0] = 1u;
}

static void AdapterBf16Experts(SparkGpuStream stream, const AdapterExperts &model, uint32_t multiprocessors)
{
	AdapterRoutes routes;
	uint32_t tokens[3];
	char what[160];
	AdapterHot(routes,model.experts);
	AdapterTokens(model.experts,model.top_k,tokens);
	void *up = AdapterWeights(routes,(uint64_t)model.up_output * model.hidden * 2u,1u,ADAPTER_WEIGHT_BF16,0u);
	void *down = AdapterWeights(routes,(uint64_t)model.hidden * model.down_input * 2u,2u,ADAPTER_WEIGHT_BF16,0u);
	for ( uint32_t tile = 0u; tile < 3u; ++tile )
	{
		AdapterRoute(routes,tokens[tile],model.top_k);
		AdapterGrouped first = AdapterGroupedSetup(routes,1u,model.hidden,model.up_output,3u + tile);
		LmGemmArguments gemm = AdapterGroupedArguments(routes,first,LmScaleTensorNone());
		int32_t status = LmGemmWeightOnlyIndirectLaunch<LmBf16Format,128u,2u,8u>(&gemm,first.activation,up,routes.packed_rows,routes.tokens,routes.top_k,
			routes.experts,model.hidden,model.up_output,multiprocessors,stream);
		snprintf(what,sizeof(what),"%s experts up bf16 indirect tokens%u topk%u experts%u in%u out%u",model.adapter,routes.tokens,routes.top_k,routes.experts,
			model.hidden,model.up_output);
		AdapterEmit(what,stream,status,first.output,(uint64_t)routes.packed_rows * model.up_output * 2u);
		AdapterGroupedFree(first);
		AdapterGrouped second = AdapterGroupedSetup(routes,0u,model.down_input,model.hidden,6u + tile);
		gemm = AdapterGroupedArguments(routes,second,LmScaleTensorNone());
		status = LmGemmWeightOnlyLaunch<LmBf16Format,128u,2u,8u>(&gemm,second.activation,down,routes.packed_rows,routes.tokens,routes.top_k,routes.experts,
			model.down_input,model.hidden,multiprocessors,true,stream);
		snprintf(what,sizeof(what),"%s experts down bf16 grouped tokens%u topk%u experts%u in%u out%u",model.adapter,routes.tokens,routes.top_k,routes.experts,
			model.down_input,model.hidden);
		AdapterEmit(what,stream,status,second.output,(uint64_t)routes.packed_rows * model.hidden * 2u);
		AdapterGroupedFree(second);
	}
	SparkGpuFree(up);
	SparkGpuFree(down);
}

template<uint32_t TILE_K>
static void AdapterCellExperts(SparkGpuStream stream, const char *adapter, uint32_t experts, uint32_t top_k, uint32_t input, uint32_t output, uint32_t indirect,
	uint32_t multiprocessors)
{
	AdapterRoutes routes;
	uint32_t tokens[3];
	char what[160];
	const uint64_t expert_bytes = (uint64_t)(input / TILE_K) * (output / 16u) * 17u * (TILE_K / 2u);
	AdapterHot(routes,experts);
	AdapterTokens(experts,top_k,tokens);
	void *weight = AdapterWeights(routes,expert_bytes,10u + TILE_K + indirect,ADAPTER_WEIGHT_CELLS,TILE_K / 2u);
	for ( uint32_t tile = 0u; tile < 3u; ++tile )
	{
		AdapterRoute(routes,tokens[tile],top_k);
		AdapterGrouped grouped = AdapterGroupedSetup(routes,indirect,input,output,20u + tile);
		LmGemmArguments gemm = AdapterGroupedArguments(routes,grouped,LmScaleTensorNone());
		const int32_t status = indirect != 0u ?
			LmGemmWeightOnlyIndirectInterleavedLaunch<LmMxfp4,128u,2u,8u,TILE_K>(&gemm,grouped.activation,weight,routes.packed_rows,routes.tokens,routes.top_k,
				routes.experts,input,output,multiprocessors,stream) :
			LmGemmWeightOnlyInterleavedLaunch<LmMxfp4,128u,2u,8u,TILE_K>(&gemm,grouped.activation,weight,routes.packed_rows,routes.tokens,routes.top_k,
				routes.experts,input,output,multiprocessors,true,stream);
		snprintf(what,sizeof(what),"%s experts mxfp4 interleaved tilek%u %s tokens%u topk%u experts%u in%u out%u",adapter,TILE_K,indirect != 0u ? "indirect" :
			"grouped",routes.tokens,routes.top_k,routes.experts,input,output);
		AdapterEmit(what,stream,status,grouped.output,(uint64_t)routes.packed_rows * output * 2u);
		AdapterGroupedFree(grouped);
	}
	SparkGpuFree(weight);
}

static void AdapterMxfp4Experts(SparkGpuStream stream, const char *adapter, uint32_t experts, uint32_t top_k, uint32_t input, uint32_t output, uint32_t indirect,
	uint32_t multiprocessors)
{
	AdapterRoutes routes;
	uint32_t tokens[3];
	char what[160];
	std::vector<uint8_t> scales;
	void *device_scales = 0;
	AdapterHot(routes,experts);
	AdapterTokens(experts,top_k,tokens);
	void *weight = AdapterWeights(routes,(uint64_t)output * input / 2u,30u + indirect,ADAPTER_WEIGHT_BYTES,0u);
	const LmScaleTensor scale = BuildScales<SPARK_WEIGHT_CODEC_MXFP4_E2M1>(scales,&device_scales,experts,output,input);
	for ( uint32_t tile = 0u; tile < 3u; ++tile )
	{
		AdapterRoute(routes,tokens[tile],top_k);
		AdapterGrouped grouped = AdapterGroupedSetup(routes,indirect,input,output,40u + tile);
		LmGemmArguments gemm = AdapterGroupedArguments(routes,grouped,scale);
		const int32_t status = indirect != 0u ?
			LmGemmWeightOnlyIndirectLaunch<LmMxfp4,128u,2u,8u>(&gemm,grouped.activation,weight,routes.packed_rows,routes.tokens,routes.top_k,routes.experts,
				input,output,multiprocessors,stream) :
			LmGemmWeightOnlyLaunch<LmMxfp4,128u,2u,8u>(&gemm,grouped.activation,weight,routes.packed_rows,routes.tokens,routes.top_k,routes.experts,input,
				output,multiprocessors,true,stream);
		snprintf(what,sizeof(what),"%s experts mxfp4 %s tokens%u topk%u experts%u in%u out%u",adapter,indirect != 0u ? "indirect" : "grouped",routes.tokens,
			routes.top_k,routes.experts,input,output);
		AdapterEmit(what,stream,status,grouped.output,(uint64_t)routes.packed_rows * output * 2u);
		AdapterGroupedFree(grouped);
	}
	SparkGpuFree(weight);
	SparkGpuFree(device_scales);
}

template<uint32_t TILE_K>
static void AdapterDense(SparkGpuStream stream, const char *adapter, uint32_t input, uint32_t output, uint32_t mode, uint32_t multiprocessors)
{
	static const uint32_t rows[3] = {8u, 16u, 200u};
	AdapterRoutes routes;
	char what[160];
	AdapterHot(routes,1u);
	void *weight = AdapterWeights(routes,(uint64_t)output * input * 2u,50u + mode,ADAPTER_WEIGHT_BF16,0u);
	for ( uint32_t tile = 0u; tile < 3u; ++tile )
	{
		const uint32_t offsets[2] = {0u, rows[tile]};
		const uint64_t cells = (uint64_t)rows[tile] * output;
		uint32_t *device_offsets = (uint32_t *)BuildUpload(offsets,sizeof(offsets));
		uint32_t *prefix = (uint32_t *)AdapterAllocate(2u * 64u * sizeof(uint32_t));
		void *activation = AdapterActivation(rows[tile],input,60u + tile);
		void *accumulate = mode == ADAPTER_OUTPUT_ACCUMULATE ? AdapterActivation(rows[tile],output,70u + tile) : 0;
		void *result = AdapterAllocate(cells * (mode == ADAPTER_OUTPUT_F32 ? 4u : 2u));
		LmGemmArguments gemm;
		memset(&gemm,0,sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = LmScaleTensorNone();
		gemm.group_row_offset = device_offsets;
		gemm.group_tile_prefix = prefix;
		gemm.output_bf16 = mode == ADAPTER_OUTPUT_F32 ? 0 : result;
		gemm.output_f32 = mode == ADAPTER_OUTPUT_F32 ? result : 0;
		gemm.accumulate_bf16 = accumulate;
		gemm.output_row_stride = output;
		const int32_t status = LmGemmLaunch<LmBf16Format,128u,TILE_K,2u,8u>(&gemm,activation,weight,rows[tile],rows[tile],1u,1u,input,output,multiprocessors,
			false,stream);
		snprintf(what,sizeof(what),"%s dense bf16 tilek%u %s rows%u in%u out%u",adapter,TILE_K,mode == ADAPTER_OUTPUT_F32 ? "f32" : mode == ADAPTER_OUTPUT_ACCUMULATE ?
			"accumulate" : "bf16",rows[tile],input,output);
		AdapterEmit(what,stream,status,result,cells * (mode == ADAPTER_OUTPUT_F32 ? 4u : 2u));
		SparkGpuFree(device_offsets);
		SparkGpuFree(prefix);
		SparkGpuFree(activation);
		SparkGpuFree(accumulate);
		SparkGpuFree(result);
	}
	SparkGpuFree(weight);
}

static void AdapterGemms(SparkGpuStream stream, uint32_t multiprocessors)
{
	static const AdapterExperts experts[] = {
		{"gemma4_26b", 2816u, 704u, 704u, 8u, 8u},
		{"kexaone2", 6144u, 128u, 128u, 256u, 8u},
		{"laguna", 3072u, 1024u, 1024u, 16u, 10u},
		{"ling3", 2560u, 768u, 768u, 32u, 8u},
		{"solar_open2", 4096u, 80u, 80u, 320u, 8u},
		{"motif3", 4096u, 2560u, 1280u, 24u, 8u},
		{"qwen38_flash_bf16", 2560u, 640u, 640u, 32u, 10u},
	};
	for ( const AdapterExperts &model : experts )
		AdapterBf16Experts(stream,model,multiprocessors);
	AdapterDense<64u>(stream,"glm53_flash",4096u,1536u,ADAPTER_OUTPUT_BF16,multiprocessors);
	AdapterDense<64u>(stream,"glm53_flash",4096u,512u,ADAPTER_OUTPUT_BF16,multiprocessors);
	AdapterDense<64u>(stream,"glm53_flash",4096u,288u,ADAPTER_OUTPUT_F32,multiprocessors);
	AdapterDense<64u>(stream,"glm53_flash",4096u,32u,ADAPTER_OUTPUT_BF16,multiprocessors);
	AdapterDense<64u>(stream,"glm53_flash",768u,4096u,ADAPTER_OUTPUT_BF16,multiprocessors);
	AdapterDense<64u>(stream,"k3",7168u,224u,ADAPTER_OUTPUT_BF16,multiprocessors);
	AdapterDense<64u>(stream,"k3",7168u,224u,ADAPTER_OUTPUT_ACCUMULATE,multiprocessors);
	AdapterDense<32u>(stream,"k3",224u,7168u,ADAPTER_OUTPUT_BF16,multiprocessors);
	AdapterDense<32u>(stream,"k3",224u,7168u,ADAPTER_OUTPUT_ACCUMULATE,multiprocessors);
	AdapterCellExperts<32u>(stream,"k3",896u,16u,224u,6144u,1u,multiprocessors);
	AdapterCellExperts<32u>(stream,"k3",896u,16u,3072u,224u,0u,multiprocessors);
	AdapterCellExperts<128u>(stream,"k3",896u,16u,3584u,6144u,1u,multiprocessors);
	AdapterCellExperts<128u>(stream,"k3",896u,16u,3072u,3584u,0u,multiprocessors);
	AdapterMxfp4Experts(stream,"k3",896u,16u,3584u,6144u,1u,multiprocessors);
	AdapterMxfp4Experts(stream,"k3",896u,16u,3072u,3584u,0u,multiprocessors);
}

int main(int argc, char **argv)
{
	SparkGpuStream stream = 0;
	int multiprocessors = 0;
	const char *flavour = argc > 1 ? argv[1] : "this";
	build_out = stdout;
	if ( SparkGpuSetDevice(0) != SPARK_GPU_SUCCESS || SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) != SPARK_GPU_SUCCESS ||
		SparkGpuDeviceGetAttribute(&multiprocessors,SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT,0) != SPARK_GPU_SUCCESS )
	{
		fprintf(stderr,"FAIL adapter tensor map GEMMs: setup on the %s flavour: %s\n",flavour,SparkGpuGetErrorString(SparkGpuGetLastError()));
		return(1);
	}
	AdapterGemms(stream,(uint32_t)multiprocessors);
	SparkGpuStreamDestroy(stream);
	fprintf(stderr,"PASS adapter tensor map GEMMs ran on the %s flavour\n",flavour);
	return(0);
}
