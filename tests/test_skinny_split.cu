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
#define SPLIT_MAX_ROWS 64u
#define SPLIT_RANKS 16u

static const uint32_t kRowCounts[] = {1u, 2u, 3u, 5u, 8u, 9u, 15u, 16u, 17u, 33u, 64u};

struct SplitShape
{
	uint32_t input, output;
};

static const SplitShape kShapes[] = {{7168u, 1536u}, {7168u, 576u}, {4096u, 512u}};

static uint64_t Mix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

template<class T>
static T *Device(const std::vector<T> &host)
{
	T *device;
	CUDA(cudaMalloc((void **)&device, host.size() * sizeof(T) + 16u));
	CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
	return device;
}

static uint16_t *Bf16Values(size_t count, uint64_t salt)
{
	std::vector<uint16_t> values(count);
	for ( size_t index = 0u; index < count; ++index )
		values[index] = (uint16_t)((0x3c00u + (Mix(index ^ salt) & 0x7ffu)) ^ ((Mix(index ^ salt ^ 1u) & 1u) << 15u));
	return Device(values);
}

template<class T>
static std::vector<T> Host(const T *device, size_t count)
{
	std::vector<T> host(count);
	CUDA(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
	return host;
}

static int32_t SplitRows(const uint8_t *weight, const uint16_t *activation, uint16_t *bf16, float *f32, uint32_t rows, const SplitShape &shape)
{
	int32_t status = LM_LAUNCH_OK;
	for ( uint32_t first = 0u; first < rows && status == LM_LAUNCH_OK; first += LM_SKINNY_ROWS_WIDE )
	{
		const uint32_t count = rows - first < LM_SKINNY_ROWS_WIDE ? rows - first : LM_SKINNY_ROWS_WIDE;
		status = LmSkinnyDenseSplit<LmBf16Format>(weight, activation + (size_t)first * shape.input,
			bf16 != 0 ? bf16 + (size_t)first * shape.output : (uint16_t *)0, f32 != 0 ? f32 + (size_t)first * shape.output : (float *)0,
			count, shape.input, shape.output, 0u, 0u, 0);
	}
	return status;
}

static void ShapeCase(const SplitShape &shape, uint32_t f32)
{
	const uint32_t slice = shape.output / SPLIT_RANKS;
	const size_t outputs = (size_t)SPLIT_MAX_ROWS * shape.output;
	uint8_t *weight = (uint8_t *)Bf16Values((size_t)shape.input * shape.output, 0x5157u + shape.output);
	uint16_t *activation = Bf16Values((size_t)SPLIT_MAX_ROWS * shape.input, 0xa11cu + shape.input);
	uint16_t *full_bf16 = 0, *sliced_bf16 = 0, *single_bf16 = 0, *default_bf16 = 0;
	float *full_f32 = 0, *sliced_f32 = 0, *single_f32 = 0, *default_f32 = 0;
	uint64_t compared = 0u, default_differs = 0u;
	REQUIRE(shape.output % SPLIT_RANKS == 0u && slice <= LM_SKINNY_SPLIT_MAX_OUTPUTS);
	if ( f32 )
	{
		CUDA(cudaMalloc((void **)&full_f32, outputs * 4u)); CUDA(cudaMalloc((void **)&sliced_f32, outputs * 4u));
		CUDA(cudaMalloc((void **)&single_f32, outputs * 4u)); CUDA(cudaMalloc((void **)&default_f32, outputs * 4u));
	}
	else
	{
		CUDA(cudaMalloc((void **)&full_bf16, outputs * 2u)); CUDA(cudaMalloc((void **)&sliced_bf16, outputs * 2u));
		CUDA(cudaMalloc((void **)&single_bf16, outputs * 2u)); CUDA(cudaMalloc((void **)&default_bf16, outputs * 2u));
	}
	for ( uint32_t rows : kRowCounts )
	{
		const size_t bytes = (size_t)rows * shape.output * (f32 ? 4u : 2u);
		void *full = f32 ? (void *)full_f32 : (void *)full_bf16, *sliced = f32 ? (void *)sliced_f32 : (void *)sliced_bf16;
		void *single = f32 ? (void *)single_f32 : (void *)single_bf16, *fallback = f32 ? (void *)default_f32 : (void *)default_bf16;
		CUDA(cudaMemset(full, 0x5a, bytes)); CUDA(cudaMemset(sliced, 0xa5, bytes)); CUDA(cudaMemset(single, 0x3c, bytes)); CUDA(cudaMemset(fallback, 0xc3, bytes));
		REQUIRE(SplitRows(weight, activation, full_bf16, full_f32, rows, shape) == LM_LAUNCH_OK);
		for ( uint32_t first = 0u; first < rows; first += LM_SKINNY_ROWS_WIDE )
		{
			const uint32_t count = rows - first < LM_SKINNY_ROWS_WIDE ? rows - first : LM_SKINNY_ROWS_WIDE;
			for ( uint32_t rank = 0u; rank < SPLIT_RANKS; ++rank )
				REQUIRE(LmSkinnyDense<LmBf16Format>(weight + (size_t)rank * slice * shape.input * 2u, activation + (size_t)first * shape.input,
					f32 ? (uint16_t *)0 : sliced_bf16 + (size_t)first * shape.output, f32 ? sliced_f32 + (size_t)first * shape.output : (float *)0,
					count, shape.input, slice, shape.output, rank * slice, 0) == LM_LAUNCH_OK);
			REQUIRE(LmSkinnyDense<LmBf16Format>(weight, activation + (size_t)first * shape.input,
				f32 ? (uint16_t *)0 : default_bf16 + (size_t)first * shape.output, f32 ? default_f32 + (size_t)first * shape.output : (float *)0,
				count, shape.input, shape.output, 0u, 0u, 0) == LM_LAUNCH_OK);
		}
		for ( uint32_t row = 0u; row < rows; ++row )
			REQUIRE(SplitRows(weight, activation + (size_t)row * shape.input, f32 ? (uint16_t *)0 : single_bf16 + (size_t)row * shape.output,
				f32 ? single_f32 + (size_t)row * shape.output : (float *)0, 1u, shape) == LM_LAUNCH_OK);
		CUDA(cudaDeviceSynchronize());
		{
			std::vector<uint8_t> a = Host((const uint8_t *)full, bytes), b = Host((const uint8_t *)sliced, bytes);
			std::vector<uint8_t> c = Host((const uint8_t *)single, bytes), d = Host((const uint8_t *)fallback, bytes);
			const size_t width = f32 ? 4u : 2u;
			for ( size_t element = 0u; element < (size_t)rows * shape.output; ++element )
			{
				if ( memcmp(a.data() + element * width, b.data() + element * width, width) != 0 ||
					memcmp(a.data() + element * width, c.data() + element * width, width) != 0 )
				{
					fprintf(stderr, "FAIL input %u output %u %s rows %u element %zu: the full split launch differs from the per-rank slices or the one-row launch\n",
						shape.input, shape.output, f32 ? "f32" : "bf16", rows, element);
					exit(1);
				}
				default_differs += memcmp(a.data() + element * width, d.data() + element * width, width) != 0 ? 1u : 0u;
			}
			compared += (uint64_t)rows * shape.output;
		}
	}
	REQUIRE(default_differs != 0u || shape.output <= LM_SKINNY_SPLIT_MAX_OUTPUTS);
	printf("ok   input %u output %u %s: %llu outputs equal the %u per-rank slices and the one-row launches; the default kernel differs on %llu\n",
		shape.input, shape.output, f32 ? "f32" : "bf16", (unsigned long long)compared, SPLIT_RANKS, (unsigned long long)default_differs);
	cudaFree(weight); cudaFree(activation);
	cudaFree(full_bf16); cudaFree(sliced_bf16); cudaFree(single_bf16); cudaFree(default_bf16);
	cudaFree(full_f32); cudaFree(sliced_f32); cudaFree(single_f32); cudaFree(default_f32);
}

static void Refusals()
{
	uint8_t *weight = (uint8_t *)Bf16Values((size_t)7168u * 64u, 1u);
	uint16_t *activation = Bf16Values((size_t)SPLIT_MAX_ROWS * 7168u, 2u), *output;
	CUDA(cudaMalloc((void **)&output, (size_t)SPLIT_MAX_ROWS * 64u * 2u));
	REQUIRE(LmSkinnyDenseSplit<LmBf16Format>(weight, activation, output, (float *)0, LM_SKINNY_ROWS_WIDE + 1u, 7168u, 64u, 0u, 0u, 0) == LM_LAUNCH_ERR_SHAPE);
	REQUIRE(LmSkinnyDenseSplit<LmBf16Format>(weight, activation, output, (float *)0, 0u, 7168u, 64u, 0u, 0u, 0) == LM_LAUNCH_ERR_SHAPE);
	REQUIRE(LmSkinnyDenseSplit<LmBf16Format>(weight, activation, output, (float *)0, 4u, 1024u, 64u, 0u, 0u, 0) == LM_LAUNCH_ERR_SHAPE);
	REQUIRE(LmSkinnyDenseSplit<LmMxfp8>(weight, activation, output, (float *)0, 4u, 7168u, 64u, 0u, 0u, 0) == LM_LAUNCH_ERR_SHAPE);
	REQUIRE(LmSkinnyDenseSplit<LmBf16Format>(weight, activation, output, (float *)0, 4u, 7168u, 64u, 32u, 0u, 0) == LM_LAUNCH_ERR_OUTPUT);
	cudaFree(weight); cudaFree(activation); cudaFree(output);
	printf("ok   refusals: more than %u rows, no rows, inputs too short to split, scaled formats and outputs past the row stride\n", LM_SKINNY_ROWS_WIDE);
}

int main()
{
	for ( const SplitShape &shape : kShapes )
	{
		ShapeCase(shape, 0u);
		ShapeCase(shape, 1u);
	}
	Refusals();
	printf("PASS skinny split\n");
	return 0;
}
