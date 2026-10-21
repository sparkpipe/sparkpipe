#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "runtime/launch.h"
#include "inference/kernels/skinny.cuh"

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)

static uint64_t Mix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

struct Weights
{
	uint8_t *weight;
	uint8_t *scale;
	uint32_t input, output;
};

template<class Format>
static LmScaleTensor Scale(const Weights &w)
{
	if constexpr ( LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_UE8M0 )
		return LmScaleTensorBlockUe8m0(w.scale, 1u, w.output, w.input, 1u, LM_MXFP8_GROUP);
	else
		return LmScaleTensorNone();
}

static Weights Make(uint32_t input, uint32_t output, uint32_t element_bytes, uint64_t salt)
{
	Weights w = {0, 0, input, output};
	std::vector<uint8_t> bytes((size_t)input * output * element_bytes), scale((size_t)output * (input / 32u));
	for ( size_t index = 0u; index < bytes.size(); ++index )
		bytes[index] = (uint8_t)(Mix(index ^ salt) >> 33u);
	if ( element_bytes == 2u )
		for ( size_t index = 1u; index < bytes.size(); index += 2u )
			bytes[index] = (uint8_t)(0x3cu + (bytes[index] & 0x03u));
	else
		for ( size_t index = 0u; index < bytes.size(); ++index )
			bytes[index] &= 0x7eu;
	for ( size_t index = 0u; index < scale.size(); ++index )
		scale[index] = (uint8_t)(120u + Mix(index ^ salt ^ 0x55u) % 9u);
	CUDA(cudaMalloc((void **)&w.weight, bytes.size()));
	CUDA(cudaMalloc((void **)&w.scale, scale.size() + 16u));
	CUDA(cudaMemcpy(w.weight, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(w.scale, scale.data(), scale.size(), cudaMemcpyHostToDevice));
	return w;
}

template<class Format>
static uint32_t Single(const Weights &w, const uint16_t *activation, uint16_t *output, uint32_t rows)
{
	LmSkinnyArguments args;
	memset(&args, 0, sizeof(args));
	args.weight = w.weight;
	args.activation = activation;
	args.output_bf16 = output;
	args.scale = Scale<Format>(w);
	args.rows = rows;
	args.input_dimension = w.input;
	args.output_dimension = w.output;
	return LmSkinnyLaunch<Format>(&args, 0) == LM_LAUNCH_OK ? 1u : 0u;
}

template<class Format>
static uint32_t Case(const char *name, uint32_t element_bytes, uint32_t input, uint32_t first_output, uint32_t second_output)
{
	const Weights first = Make(input, first_output, element_bytes, 3u + input), second = Make(input, second_output, element_bytes, 7u + second_output);
	std::vector<uint16_t> activation((size_t)16u * input);
	for ( size_t index = 0u; index < activation.size(); ++index )
		activation[index] = (uint16_t)(0x3c00u + (Mix(index ^ 0x99u) & 0x7ffu)) ^ (uint16_t)((index & 1u) << 15u);
	uint16_t *activation_d, *wide_d, *narrow_d, *pair_first_d, *pair_second_d;
	CUDA(cudaMalloc((void **)&activation_d, activation.size() * 2u));
	CUDA(cudaMalloc((void **)&wide_d, (size_t)16u * first_output * 2u));
	CUDA(cudaMalloc((void **)&narrow_d, (size_t)(first_output > second_output ? first_output : second_output) * 2u));
	CUDA(cudaMalloc((void **)&pair_first_d, (size_t)16u * first_output * 2u));
	CUDA(cudaMalloc((void **)&pair_second_d, (size_t)16u * second_output * 2u));
	CUDA(cudaMemcpy(activation_d, activation.data(), activation.size() * 2u, cudaMemcpyHostToDevice));
	uint32_t failures = 0u;
	std::vector<uint16_t> wide((size_t)16u * first_output), narrow(first_output), pair_first((size_t)16u * first_output), pair_second((size_t)16u * second_output),
		narrow_second(second_output);
	for ( uint32_t rows = 1u; rows <= 16u; ++rows )
	{
		REQUIRE(Single<Format>(first, activation_d, wide_d, rows));
		LmSkinnyDenseTarget targets[2];
		memset(targets, 0, sizeof(targets));
		targets[0].weight = first.weight; targets[0].output_bf16 = pair_first_d; targets[0].output_dimension = first_output; targets[0].scale = Scale<Format>(first);
		targets[1].weight = second.weight; targets[1].output_bf16 = pair_second_d; targets[1].output_dimension = second_output; targets[1].scale = Scale<Format>(second);
		REQUIRE(LmSkinnyDenseMulti<Format>(targets, 2u, activation_d, rows, input, 0) == LM_LAUNCH_OK);
		CUDA(cudaDeviceSynchronize());
		CUDA(cudaMemcpy(wide.data(), wide_d, (size_t)rows * first_output * 2u, cudaMemcpyDeviceToHost));
		CUDA(cudaMemcpy(pair_first.data(), pair_first_d, (size_t)rows * first_output * 2u, cudaMemcpyDeviceToHost));
		CUDA(cudaMemcpy(pair_second.data(), pair_second_d, (size_t)rows * second_output * 2u, cudaMemcpyDeviceToHost));
		uint32_t differing = 0u;
		for ( uint32_t row = 0u; row < rows; ++row )
		{
			REQUIRE(Single<Format>(first, activation_d + (size_t)row * input, narrow_d, 1u));
			CUDA(cudaMemcpy(narrow.data(), narrow_d, (size_t)first_output * 2u, cudaMemcpyDeviceToHost));
			differing += memcmp(narrow.data(), wide.data() + (size_t)row * first_output, (size_t)first_output * 2u) != 0 ? 1u : 0u;
			differing += memcmp(narrow.data(), pair_first.data() + (size_t)row * first_output, (size_t)first_output * 2u) != 0 ? 1u : 0u;
			REQUIRE(Single<Format>(second, activation_d + (size_t)row * input, narrow_d, 1u));
			CUDA(cudaMemcpy(narrow_second.data(), narrow_d, (size_t)second_output * 2u, cudaMemcpyDeviceToHost));
			differing += memcmp(narrow_second.data(), pair_second.data() + (size_t)row * second_output, (size_t)second_output * 2u) != 0 ? 1u : 0u;
		}
		if ( differing != 0u )
			printf("FAIL %s %u -> %u/%u rows %u: %u row outputs differ from one-row launches\n", name, input, first_output, second_output, rows, differing);
		failures += differing != 0u ? 1u : 0u;
	}
	printf("%s %s %u -> %u and %u: rows 1-16, single and paired launches\n", failures == 0u ? "ok  " : "FAIL", name, input, first_output, second_output);
	cudaFree(first.weight); cudaFree(first.scale); cudaFree(second.weight); cudaFree(second.scale);
	cudaFree(activation_d); cudaFree(wide_d); cudaFree(narrow_d); cudaFree(pair_first_d); cudaFree(pair_second_d);
	return failures;
}

int main(void)
{
	uint32_t failures = 0u;
	failures += Case<LmMxfp8>("mxfp8", 1u, 5120u, 1280u, 512u);
	failures += Case<LmMxfp8>("mxfp8", 1u, 1280u, 2048u, 320u);
	failures += Case<LmMxfp8>("mxfp8", 1u, 8192u, 320u, 640u);
	failures += Case<LmBf16Format>("bf16", 2u, 5120u, 384u, 1536u);
	if ( failures != 0u )
	{
		printf("FAIL skinny rows: %u shapes give multi-row outputs that differ from one-row launches\n", failures);
		return 1;
	}
	printf("PASS skinny rows: every row of a multi-row launch is the same bits as a one-row launch of that row\n");
	return 0;
}
