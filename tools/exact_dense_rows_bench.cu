#include "inference/kernels/skinny.cuh"
#include "inference/kernels/weight_codec.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>
#include <vector>

#define DB_REPEATS 10u
#define DB_TRIALS 9u

typedef struct DbShape
{
	const char *name;
	uint32_t input;
	uint32_t output;
} DbShape;

__global__ static void DbFillKernel(uint8_t *bytes, uint64_t count, uint32_t kind)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		uint32_t value = (uint32_t)index * 0x9e3779b9u;
		value ^= value >> 16u;
		value *= 0x7feb352du;
		value ^= value >> 15u;
		bytes[index] = kind == 0u ? (uint8_t)((value & 0x80u) | ((1u + (value >> 8u) % 8u) << 3u) | ((value >> 16u) & 7u)) :
			kind == 1u ? (uint8_t)(118u + (value >> 8u) % 5u) : (uint8_t)((index & 1u) != 0u ? 0x3cu : value);
	}
}

static uint8_t *DbBytes(uint64_t count, uint32_t kind)
{
	uint8_t *device = 0;
	if ( cudaMalloc((void **)&device, count + 16u) != cudaSuccess )
		exit(1);
	DbFillKernel<<<1024u, 256u>>>(device, count, kind);
	return device;
}

template<class Format>
static void DbRun(const DbShape &shape, const char *format)
{
	static const uint32_t frames[] = { 16u, 32u, 64u, 128u, 256u, 512u };
	const uint64_t weight_bytes = (uint64_t)shape.output * shape.input * Format::kStoredBits / 8u;
	uint8_t *weight = DbBytes(weight_bytes, std::is_same<Format, LmBf16Format>::value ? 2u : 0u), *scale_storage = 0;
	uint16_t *activation = (uint16_t *)DbBytes((uint64_t)512u * shape.input * 2u, 2u), *output = 0;
	LmScaleTensor scale = LmScaleTensorNone();
	cudaEvent_t start, stop;
	if constexpr ( std::is_same<Format, LmFp8>::value )
	{
		std::vector<float> host((uint64_t)shape.output * ((shape.input + 127u) / 128u), 1.0f / 64.0f);
		cudaMalloc((void **)&scale_storage, host.size() * sizeof(float));
		cudaMemcpy(scale_storage, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
		scale = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(scale_storage, 1u, shape.output, shape.input);
	}
	else if constexpr ( std::is_same<Format, LmMxfp8>::value )
	{
		scale_storage = DbBytes((uint64_t)shape.output * (shape.input / 32u), 1u);
		scale = LmScaleTensorBuild(scale_storage, LM_SCALE_ENCODING_UE8M0, 1u, shape.output, shape.input, 1u, 32u);
	}
	cudaMalloc((void **)&output, (uint64_t)512u * shape.output * sizeof(uint16_t));
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf("%s (%s, %u x %u, %.1f MiB)\n", shape.name, format, shape.output, shape.input, (double)weight_bytes / 1048576.0);
	printf("%6s %12s %12s %14s %14s %12s %9s\n", "rows", "chunked_us", "frame_us", "chunked_MiB", "frame_MiB", "frame_GB/s", "speedup");
	for ( uint32_t rows : frames )
	{
		float chunked_ms = 1e30f, frame_ms = 1e30f, elapsed = 0.0f;
		for ( uint32_t pass = 0u; pass < DB_TRIALS; ++pass )
		{
			cudaEventRecord(start);
			for ( uint32_t repeat = 0u; repeat < DB_REPEATS; ++repeat )
				for ( uint32_t first = 0u; first < rows; first += LM_SKINNY_ROWS_WIDE )
				{
					LmSkinnyArguments args;
					memset(&args, 0, sizeof(args));
					args.weight = weight;
					args.activation = activation + (uint64_t)first * shape.input;
					args.output_bf16 = output + (uint64_t)first * shape.output;
					args.scale = scale;
					args.rows = rows - first < LM_SKINNY_ROWS_WIDE ? rows - first : LM_SKINNY_ROWS_WIDE;
					args.input_dimension = shape.input;
					args.output_dimension = shape.output;
					LmSkinnyLaunch<Format>(&args, 0);
				}
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsed, start, stop);
			chunked_ms = pass != 0u && elapsed < chunked_ms ? elapsed : chunked_ms;
			cudaEventRecord(start);
			for ( uint32_t repeat = 0u; repeat < DB_REPEATS; ++repeat )
			{
				LmSkinnyArguments args;
				memset(&args, 0, sizeof(args));
				args.weight = weight;
				args.activation = activation;
				args.output_bf16 = output;
				args.scale = scale;
				args.rows = rows;
				args.input_dimension = shape.input;
				args.output_dimension = shape.output;
				LmSkinnyLaunch<Format>(&args, 0, 1u);
			}
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsed, start, stop);
			frame_ms = pass != 0u && elapsed < frame_ms ? elapsed : frame_ms;
		}
		if ( cudaDeviceSynchronize() != cudaSuccess )
		{
			fprintf(stderr, "a launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
			exit(1);
		}
		printf("%6u %12.1f %12.1f %14.1f %14.1f %12.1f %8.2fx\n", rows, 1000.0f * chunked_ms / DB_REPEATS, 1000.0f * frame_ms / DB_REPEATS,
			(double)((rows + 15u) / 16u * weight_bytes) / 1048576.0, (double)weight_bytes / 1048576.0,
			(double)weight_bytes / (frame_ms / DB_REPEATS * 1e-3) / 1e9, chunked_ms / frame_ms);
	}
	cudaFree(weight);
	cudaFree(scale_storage);
	cudaFree(activation);
	cudaFree(output);
}

int main(void)
{
	static const DbShape bf16[] = { { "gemma-4-31b gate_up", 5376u, 2688u }, { "qwen-3.8 gate_up", 5120u, 2176u }, { "kimi-k3 o_proj", 4096u, 7168u } };
	static const DbShape fp8[] = { { "glm-5.3-full o_proj", 1024u, 6144u }, { "glm-5.3-full dense gate_up", 6144u, 1536u }, { "glm-5.3-full index_q", 2048u, 4096u } };
	static const DbShape mxfp8[] = { { "hy4 q_a", 6144u, 2048u } };
	for ( const DbShape &shape : bf16 )
		DbRun<LmBf16Format>(shape, "bf16");
	for ( const DbShape &shape : fp8 )
		DbRun<LmFp8>(shape, "fp8");
	for ( const DbShape &shape : mxfp8 )
		DbRun<LmMxfp8>(shape, "mxfp8");
	return 0;
}
