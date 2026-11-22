#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "sparkpipe/spark_gpu.h"
#include "runtime/gemm.cuh"
#include "inference/kernels/weight_codec.cuh"

static FILE *build_out;
static int32_t build_status;

static uint64_t BuildHash(uint64_t value)
{
	value ^= value >> 33;
	value *= 0xff51afd7ed558ccdull;
	value ^= value >> 33;
	value *= 0xc4ceb9fe1a85ec53ull;
	value ^= value >> 33;
	return(value);
}

static uint16_t BuildBf16(uint64_t index, uint64_t salt)
{
	const float value = (float)((double)(BuildHash(index ^ (salt << 40u)) % 200001u) / 2.0e6 - 0.05);
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static void BuildEmit(const char *what, const void *bytes, uint64_t count, SparkGpuError error)
{
	uint64_t digest = 1469598103934665603ull;
	if ( error == SPARK_GPU_SUCCESS && build_status != LM_LAUNCH_OK )
	{
		fprintf(build_out,"%s refused %d\n",what,build_status);
		return;
	}
	for ( uint64_t index = 0u; index < count; ++index )
		digest = (digest ^ ((const uint8_t *)bytes)[index]) * 1099511628211ull;
	fprintf(build_out,"%s %llu %016llx %s\n",what,(unsigned long long)count,(unsigned long long)digest,error == SPARK_GPU_SUCCESS ? "ok" : "error");
	if ( error != SPARK_GPU_SUCCESS )
	{
		printf("FAIL %s: %s\n",what,SparkGpuGetErrorString(error));
		fclose(build_out);
		exit(1);
	}
}

static void *BuildUpload(const void *host, uint64_t bytes)
{
	void *device = 0;
	if ( SparkGpuMalloc((void **)&device,bytes) != SPARK_GPU_SUCCESS || SparkGpuMemcpy(device,host,bytes,SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
	{
		printf("FAIL upload of %llu bytes\n",(unsigned long long)bytes);
		exit(1);
	}
	return(device);
}

static SparkGpuError BuildFinish(SparkGpuStream stream, int32_t status)
{
	build_status = status;
	return(SparkGpuStreamSynchronize(stream));
}

template<uint32_t TILE_K>
static void BuildDense(SparkGpuStream stream, uint32_t groups, uint32_t rows_per_group, uint32_t input, uint32_t output, uint32_t multiprocessors)
{
	const uint32_t rows = groups * rows_per_group;
	std::vector<uint16_t> weight((uint64_t)groups * output * input),activation((uint64_t)rows * input),result((uint64_t)rows * output);
	std::vector<uint32_t> offsets(groups + 1u);
	LmGemmArguments gemm;
	char what[128];
	for ( uint64_t index = 0u; index < weight.size(); ++index )
		weight[index] = BuildBf16(index,1u + input + groups);
	for ( uint64_t index = 0u; index < activation.size(); ++index )
		activation[index] = BuildBf16(index,2u + output);
	for ( uint32_t group = 0u; group <= groups; ++group )
		offsets[group] = group * rows_per_group;
	void *device_weight = BuildUpload(weight.data(),weight.size() * 2u);
	void *device_activation = BuildUpload(activation.data(),activation.size() * 2u);
	uint32_t *row_offset = (uint32_t *)BuildUpload(offsets.data(),offsets.size() * 4u),*tile_prefix = 0;
	uint16_t *device_output = 0;
	SparkGpuMalloc((void **)&tile_prefix,(groups + 1u) * 64u * sizeof(uint32_t));
	SparkGpuMalloc((void **)&device_output,result.size() * 2u);
	SparkGpuMemset(device_output,0,result.size() * 2u);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = row_offset;
	gemm.group_tile_prefix = tile_prefix;
	gemm.output_bf16 = device_output;
	gemm.output_row_stride = output;
	const int32_t status = LmGemmLaunch<LmBf16Format,128u,TILE_K,2u,8u>(&gemm,device_activation,device_weight,rows,rows,1u,groups,input,output,
		multiprocessors,groups > 1u,stream);
	const SparkGpuError error = BuildFinish(stream,status);
	SparkGpuMemcpy(result.data(),device_output,result.size() * 2u,SPARK_GPU_COPY_DEVICE_TO_HOST);
	snprintf(what,sizeof(what),"dense bf16 tilek%u groups%u rows%u in%u out%u",TILE_K,groups,rows_per_group,input,output);
	BuildEmit(what,result.data(),result.size() * 2u,error);
	SparkGpuFree(device_weight);
	SparkGpuFree(device_activation);
	SparkGpuFree(device_output);
	SparkGpuFree(row_offset);
	SparkGpuFree(tile_prefix);
}

template<uint32_t Codec>
static LmScaleTensor BuildScales(std::vector<uint8_t> &bytes, void **device, uint32_t groups, uint32_t output, uint32_t input)
{
	using Format = typename LmWeightCodec<Codec>::Format;
	const uint64_t count = (uint64_t)groups * output * (input / Format::kScaleGroup);
	if constexpr ( LmWeightCodec<Codec>::kScaleEncoding == LM_SCALE_ENCODING_UE8M0 )
	{
		bytes.resize(count);
		for ( uint64_t index = 0u; index < count; ++index )
			bytes[index] = (uint8_t)(122u + BuildHash(index ^ 0xa11ull) % 5u);
	}
	else
	{
		bytes.resize(count * sizeof(float));
		for ( uint64_t index = 0u; index < count; ++index )
		{
			const float value = 0.004f + (float)(BuildHash(index ^ 0xb22ull) % 1000u) * 1.0e-5f;
			memcpy(&bytes[index * sizeof(float)],&value,sizeof(float));
		}
	}
	*device = BuildUpload(bytes.data(),bytes.size());
	return(LmWeightCodecScaleTensor<Codec>(*device,groups,output,input));
}

template<uint32_t Codec>
static void BuildWeightOnly(SparkGpuStream stream, const char *name, uint32_t groups, uint32_t rows_per_group, uint32_t input, uint32_t output, uint32_t multiprocessors)
{
	using Format = typename LmWeightCodec<Codec>::Format;
	const uint32_t rows = groups * rows_per_group;
	const uint64_t row_bytes = (uint64_t)input * Format::kStoredBits / 8u;
	std::vector<uint8_t> weight((uint64_t)groups * output * row_bytes),scales;
	std::vector<uint16_t> activation((uint64_t)rows * input),result((uint64_t)rows * output);
	std::vector<uint32_t> offsets(groups + 1u);
	LmGemmArguments gemm;
	void *device_scales = 0;
	char what[128];
	for ( uint64_t index = 0u; index < weight.size(); ++index )
	{
		weight[index] = (uint8_t)BuildHash(index ^ ((uint64_t)Codec << 32u));
		if ( Codec == SPARK_WEIGHT_CODEC_FP8_E4M3 && (weight[index] & 0x7fu) == 0x7fu )
			weight[index] ^= 0x01u;
	}
	for ( uint64_t index = 0u; index < activation.size(); ++index )
		activation[index] = BuildBf16(index,5u + input);
	for ( uint32_t group = 0u; group <= groups; ++group )
		offsets[group] = group * rows_per_group;
	void *device_weight = BuildUpload(weight.data(),weight.size());
	void *device_activation = BuildUpload(activation.data(),activation.size() * 2u);
	uint32_t *row_offset = (uint32_t *)BuildUpload(offsets.data(),offsets.size() * 4u),*tile_prefix = 0;
	uint16_t *device_output = 0;
	SparkGpuMalloc((void **)&tile_prefix,(groups + 1u) * 64u * sizeof(uint32_t));
	SparkGpuMalloc((void **)&device_output,result.size() * 2u);
	SparkGpuMemset(device_output,0,result.size() * 2u);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = BuildScales<Codec>(scales,&device_scales,groups,output,input);
	gemm.group_row_offset = row_offset;
	gemm.group_tile_prefix = tile_prefix;
	gemm.output_bf16 = device_output;
	gemm.output_row_stride = output;
	const int32_t status = LmGemmWeightOnlyLaunch<Format,128u,2u,8u>(&gemm,device_activation,device_weight,rows,rows,1u,groups,input,output,
		multiprocessors,groups > 1u,stream);
	const SparkGpuError error = BuildFinish(stream,status);
	SparkGpuMemcpy(result.data(),device_output,result.size() * 2u,SPARK_GPU_COPY_DEVICE_TO_HOST);
	snprintf(what,sizeof(what),"weight-only %s groups%u rows%u in%u out%u",name,groups,rows_per_group,input,output);
	BuildEmit(what,result.data(),result.size() * 2u,error);
	SparkGpuFree(device_weight);
	SparkGpuFree(device_activation);
	SparkGpuFree(device_output);
	SparkGpuFree(device_scales);
	SparkGpuFree(row_offset);
	SparkGpuFree(tile_prefix);
}

template<uint32_t TILE_K>
static void BuildInterleaved(SparkGpuStream stream, uint32_t input, uint32_t output, uint32_t multiprocessors)
{
	const uint32_t experts = 2u,tokens = 2u,top_k = 2u,packed_rows = tokens * top_k,tile_n = 128u;
	const uint32_t tiles_per_expert = (output + tile_n - 1u) / tile_n;
	const uint64_t expert_bytes = (uint64_t)(input / TILE_K) * (output / 16u) * 17u * (TILE_K / 2u);
	uint32_t group_offset[3] = {0u, 2u, 4u},group_prefix[3] = {0u, tiles_per_expert, 2u * tiles_per_expert};
	std::vector<uint16_t> packed((uint64_t)packed_rows * input),result((uint64_t)packed_rows * output);
	std::vector<uint8_t> weights(expert_bytes * experts);
	LmGemmArguments gemm;
	char what[128];
	for ( uint64_t index = 0u; index < packed.size(); ++index )
		packed[index] = BuildBf16(index,3u + input);
	for ( uint64_t index = 0u; index < weights.size(); ++index )
		weights[index] = (uint8_t)BuildHash(index ^ 0x5eedull);
	for ( uint64_t cell = 0u; cell < weights.size() / (17u * (TILE_K / 2u)); ++cell )
		memset(&weights[cell * 17u * (TILE_K / 2u) + 16u * (TILE_K / 2u)],(int)(122u + cell % 5u),TILE_K / 2u);
	void *device_packed = BuildUpload(packed.data(),packed.size() * 2u);
	void *device_weights = BuildUpload(weights.data(),weights.size());
	uint32_t *device_offset = (uint32_t *)BuildUpload(group_offset,sizeof(group_offset));
	uint32_t *device_prefix = (uint32_t *)BuildUpload(group_prefix,sizeof(group_prefix));
	uint16_t *device_output = 0;
	SparkGpuMalloc((void **)&device_output,result.size() * 2u);
	SparkGpuMemset(device_output,0,result.size() * 2u);
	memset(&gemm,0,sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.group_row_offset = device_offset;
	gemm.group_tile_prefix = device_prefix;
	gemm.prefix_built = 1u;
	gemm.output_bf16 = device_output;
	gemm.output_row_stride = output;
	const int32_t status = LmGemmWeightOnlyInterleavedLaunch<LmMxfp4,128u,2u,8u,TILE_K>(&gemm,device_packed,device_weights,packed_rows,tokens,top_k,experts,
		input,output,multiprocessors,true,stream);
	const SparkGpuError error = BuildFinish(stream,status);
	SparkGpuMemcpy(result.data(),device_output,result.size() * 2u,SPARK_GPU_COPY_DEVICE_TO_HOST);
	snprintf(what,sizeof(what),"interleaved mxfp4 tilek%u in%u out%u",TILE_K,input,output);
	BuildEmit(what,result.data(),result.size() * 2u,error);
	SparkGpuFree(device_packed);
	SparkGpuFree(device_output);
	SparkGpuFree(device_weights);
	SparkGpuFree(device_offset);
	SparkGpuFree(device_prefix);
}

static void BuildGemms(SparkGpuStream stream, uint32_t multiprocessors)
{
	static const uint32_t dense[][4] = {{1u, 1u, 4096u, 4096u}, {1u, 8u, 4096u, 4096u}, {1u, 1u, 7168u, 2048u}, {1u, 64u, 4096u, 1024u},
		{1u, 200u, 2048u, 640u}, {1u, 512u, 4096u, 4096u}, {1u, 7u, 1024u, 384u}, {1u, 1u, 256u, 128u}, {2u, 2u, 2048u, 512u},
		{4u, 8u, 1024u, 384u}, {2u, 1u, 7168u, 2048u}};
	static const uint32_t dense32[][4] = {{1u, 1u, 4096u, 4096u}, {1u, 8u, 1024u, 384u}, {2u, 4u, 2048u, 512u}};
	static const uint32_t weight_only[][4] = {{1u, 1u, 4096u, 1024u}, {1u, 8u, 2048u, 640u}, {2u, 4u, 2048u, 512u}, {4u, 2u, 1024u, 384u}};
	static const uint32_t interleaved[][2] = {{3072u, 256u}, {4096u, 2048u}, {7168u, 4096u}, {3072u, 224u}, {256u, 128u}};
	for ( const auto &shape : dense )
		BuildDense<LmBf16Format::kTileK>(stream,shape[0],shape[1],shape[2],shape[3],multiprocessors);
	for ( const auto &shape : dense32 )
		BuildDense<32u>(stream,shape[0],shape[1],shape[2],shape[3],multiprocessors);
	for ( const auto &shape : weight_only )
	{
		BuildWeightOnly<SPARK_WEIGHT_CODEC_MXFP4_E2M1>(stream,"mxfp4",shape[0],shape[1],shape[2],shape[3],multiprocessors);
		BuildWeightOnly<SPARK_WEIGHT_CODEC_INT6>(stream,"int6",shape[0],shape[1],shape[2],shape[3],multiprocessors);
		BuildWeightOnly<SPARK_WEIGHT_CODEC_INT7>(stream,"int7",shape[0],shape[1],shape[2],shape[3],multiprocessors);
		BuildWeightOnly<SPARK_WEIGHT_CODEC_FP8_E4M3>(stream,"fp8",shape[0],shape[1],shape[2],shape[3],multiprocessors);
	}
	for ( const auto &shape : interleaved )
	{
		BuildInterleaved<32u>(stream,shape[0],shape[1],multiprocessors);
		BuildInterleaved<128u>(stream,shape[0],shape[1],multiprocessors);
	}
}
