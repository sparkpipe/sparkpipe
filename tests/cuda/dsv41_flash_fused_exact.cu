#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "sparkpipe/spark_dsv41_flash_stage_model.h"
#include "dsv41_flash_stage.cuh"

#define FUSED_REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define FUSED_CUDA(call) FUSED_REQUIRE((call) == cudaSuccess)

static __host__ __device__ __forceinline__ uint64_t FusedMix(uint64_t value)
{
	value += 0x9e3779b97f4a7c15ull;
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
	value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
	return value ^ (value >> 31);
}

static __global__ void FusedFillKernel(uint8_t *data, uint64_t bytes, uint64_t seed, uint32_t kind, uint32_t edges)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < bytes; index += (uint64_t)gridDim.x * blockDim.x )
	{
		uint8_t byte = 0u;
		if ( kind == 1u )
		{
			const uint64_t element = index >> 1, random = FusedMix(seed ^ element);
			uint32_t value = (uint32_t)(random & 0x8000u) | ((118u + (uint32_t)((random >> 16) % 12u)) << 7) | (uint32_t)((random >> 24) & 0x7fu);
			if ( edges != 0u && element % 97u == 0u )
				value = (uint32_t)(random & 0x8000u) | 0x46eau;
			if ( edges != 0u && element % 89u == 0u )
				value = 0u;
			byte = (uint8_t)(value >> (8u * (uint32_t)(index & 1u)));
		}
		else if ( kind == 2u )
		{
			const uint64_t random = FusedMix(seed ^ (index >> 2));
			const uint32_t value = (uint32_t)(random & 0x80000000u) | ((116u + (uint32_t)((random >> 32) % 12u)) << 23) | (uint32_t)((random >> 8) & 0x7fffffu);
			byte = (uint8_t)(value >> (8u * (uint32_t)(index & 3u)));
		}
		data[index] = byte;
	}
}

static void D41FusedMixNormKernelBuffers(D41FusedMixNormArguments *set, uint64_t seed, uint32_t edges)
{
	const uint32_t rows = 16u;
	{
		const uint64_t count = (uint64_t)(rows * D41K_HC * D41_HIDDEN), bytes = count * sizeof(uint16_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 0u, 1u, edges);
		set->streams = (const uint16_t *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(LmHcSinkhornShape<D41K_HC>::kMix * D41K_HC * D41_HIDDEN), bytes = count * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 1u, 2u, edges);
		set->fn = (const float *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(4u), bytes = count * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 2u, 2u, edges);
		set->scale = (const float *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(LmHcSinkhornShape<D41K_HC>::kMix), bytes = count * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 3u, 2u, edges);
		set->base = (const float *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows * LmHcSinkhornShape<D41K_HC>::kPartials), bytes = count * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->partials = (float *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows * D41K_HC), bytes = count * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->pre = (float *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows * D41K_HC), bytes = count * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->post = (float *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows * D41K_HC * D41K_HC), bytes = count * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->comb = (float *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows), bytes = count * sizeof(uint32_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->counters = (uint32_t *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows * D41K_HC), bytes = count * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 9u, 2u, edges);
		set->collapse_pre = (const float *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(D41_HIDDEN), bytes = count * sizeof(uint16_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 10u, 1u, edges);
		set->norm = (const uint16_t *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows * D41_HIDDEN), bytes = count * sizeof(uint16_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->x = (uint16_t *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows * D41_HIDDEN), bytes = count * sizeof(uint16_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->xq = (uint16_t *)pointer;
	}
	(void)rows;
	(void)seed;
	(void)edges;
	FUSED_CUDA(cudaDeviceSynchronize());
}

static void D41FusedMixNormKernelCopy(D41FusedMixNormArguments *to, const D41FusedMixNormArguments *from)
{
	const uint32_t rows = 16u;
	FUSED_CUDA(cudaMemcpy((void *)to->streams, from->streams, (uint64_t)(rows * D41K_HC * D41_HIDDEN) * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->fn, from->fn, (uint64_t)(LmHcSinkhornShape<D41K_HC>::kMix * D41K_HC * D41_HIDDEN) * sizeof(float), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->scale, from->scale, (uint64_t)(4u) * sizeof(float), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->base, from->base, (uint64_t)(LmHcSinkhornShape<D41K_HC>::kMix) * sizeof(float), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->partials, from->partials, (uint64_t)(rows * LmHcSinkhornShape<D41K_HC>::kPartials) * sizeof(float), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->pre, from->pre, (uint64_t)(rows * D41K_HC) * sizeof(float), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->post, from->post, (uint64_t)(rows * D41K_HC) * sizeof(float), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->comb, from->comb, (uint64_t)(rows * D41K_HC * D41K_HC) * sizeof(float), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->counters, from->counters, (uint64_t)(rows) * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->collapse_pre, from->collapse_pre, (uint64_t)(rows * D41K_HC) * sizeof(float), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->norm, from->norm, (uint64_t)(D41_HIDDEN) * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->x, from->x, (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->xq, from->xq, (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaDeviceSynchronize());
	(void)rows;
}

static void D41FusedMixNormKernelReference(const D41FusedMixNormArguments *set, uint32_t rows, cudaStream_t stream)
{
	const uint16_t *streams = set->streams;
	const float *fn = set->fn;
	const float *scale = set->scale;
	const float *base = set->base;
	float *partials = set->partials;
	float *pre = set->pre;
	float *post = set->post;
	float *comb = set->comb;
	uint32_t *counters = set->counters;
	const float *collapse_pre = set->collapse_pre;
	const uint16_t *norm = set->norm;
	uint16_t *x = set->x;
	uint16_t *xq = set->xq;
	(void)streams;
	(void)fn;
	(void)scale;
	(void)base;
	(void)partials;
	(void)pre;
	(void)post;
	(void)comb;
	(void)counters;
	(void)collapse_pre;
	(void)norm;
	(void)x;
	(void)xq;
	FUSED_REQUIRE(LmHcSinkhornMixes<D41K_HC>(streams, fn, scale, base, partials, pre, post, comb, rows, D41_HIDDEN, D41_EPS, D41_HC_ITERATIONS, D41_HC_EPS, stream) == LM_LAUNCH_OK);
	D41CollapseNormQdq(streams, collapse_pre, norm, x, xq, rows, stream);
	FUSED_CUDA(cudaGetLastError());
}

static uint64_t D41FusedMixNormKernelDiffer(const D41FusedMixNormArguments *one, const D41FusedMixNormArguments *two, uint32_t rows, uint32_t report)
{
	uint64_t differ = 0u;
	{
		const uint64_t bytes = (uint64_t)(rows * LmHcSinkhornShape<D41K_HC>::kPartials) * sizeof(float);
		std::vector<uint8_t> left(bytes), right(bytes);
		uint64_t count = 0u;
		FUSED_CUDA(cudaMemcpy(left.data(), one->partials, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->partials, bytes, cudaMemcpyDeviceToHost));
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "partials: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * D41K_HC) * sizeof(float);
		std::vector<uint8_t> left(bytes), right(bytes);
		uint64_t count = 0u;
		FUSED_CUDA(cudaMemcpy(left.data(), one->pre, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->pre, bytes, cudaMemcpyDeviceToHost));
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "pre: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * D41K_HC) * sizeof(float);
		std::vector<uint8_t> left(bytes), right(bytes);
		uint64_t count = 0u;
		FUSED_CUDA(cudaMemcpy(left.data(), one->post, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->post, bytes, cudaMemcpyDeviceToHost));
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "post: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * D41K_HC * D41K_HC) * sizeof(float);
		std::vector<uint8_t> left(bytes), right(bytes);
		uint64_t count = 0u;
		FUSED_CUDA(cudaMemcpy(left.data(), one->comb, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->comb, bytes, cudaMemcpyDeviceToHost));
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "comb: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t);
		std::vector<uint8_t> left(bytes), right(bytes);
		uint64_t count = 0u;
		FUSED_CUDA(cudaMemcpy(left.data(), one->x, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->x, bytes, cudaMemcpyDeviceToHost));
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "x: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t);
		std::vector<uint8_t> left(bytes), right(bytes);
		uint64_t count = 0u;
		FUSED_CUDA(cudaMemcpy(left.data(), one->xq, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->xq, bytes, cudaMemcpyDeviceToHost));
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "xq: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	return differ;
}

static uint64_t D41FusedMixNormKernelRowZero(const D41FusedMixNormArguments *one, uint32_t rows_one, const D41FusedMixNormArguments *two, uint32_t rows_two)
{
	uint64_t differ = 0u;
	{
		uint64_t offset[2], count[2];
		{ const uint32_t rows = 1u; offset[0] = offset[1] = 0u; count[0] = count[1] = (uint64_t)(rows * LmHcSinkhornShape<D41K_HC>::kPartials); }
		FUSED_REQUIRE(count[0] == count[1]);
		std::vector<float> left(count[0]), right(count[1]);
		FUSED_CUDA(cudaMemcpy(left.data(), one->partials + offset[0], count[0] * sizeof(float), cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->partials + offset[1], count[1] * sizeof(float), cudaMemcpyDeviceToHost));
		if ( memcmp(left.data(), right.data(), count[0] * sizeof(float)) != 0 )
		{
			fprintf(stderr, "partials: row 0 differs between %u and %u rows\n", rows_one, rows_two);
			++differ;
		}
	}
	{
		uint64_t offset[2], count[2];
		{ const uint32_t rows = 1u; offset[0] = offset[1] = 0u; count[0] = count[1] = (uint64_t)(rows * D41K_HC); }
		FUSED_REQUIRE(count[0] == count[1]);
		std::vector<float> left(count[0]), right(count[1]);
		FUSED_CUDA(cudaMemcpy(left.data(), one->pre + offset[0], count[0] * sizeof(float), cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->pre + offset[1], count[1] * sizeof(float), cudaMemcpyDeviceToHost));
		if ( memcmp(left.data(), right.data(), count[0] * sizeof(float)) != 0 )
		{
			fprintf(stderr, "pre: row 0 differs between %u and %u rows\n", rows_one, rows_two);
			++differ;
		}
	}
	{
		uint64_t offset[2], count[2];
		{ const uint32_t rows = 1u; offset[0] = offset[1] = 0u; count[0] = count[1] = (uint64_t)(rows * D41K_HC); }
		FUSED_REQUIRE(count[0] == count[1]);
		std::vector<float> left(count[0]), right(count[1]);
		FUSED_CUDA(cudaMemcpy(left.data(), one->post + offset[0], count[0] * sizeof(float), cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->post + offset[1], count[1] * sizeof(float), cudaMemcpyDeviceToHost));
		if ( memcmp(left.data(), right.data(), count[0] * sizeof(float)) != 0 )
		{
			fprintf(stderr, "post: row 0 differs between %u and %u rows\n", rows_one, rows_two);
			++differ;
		}
	}
	{
		uint64_t offset[2], count[2];
		{ const uint32_t rows = 1u; offset[0] = offset[1] = 0u; count[0] = count[1] = (uint64_t)(rows * D41K_HC * D41K_HC); }
		FUSED_REQUIRE(count[0] == count[1]);
		std::vector<float> left(count[0]), right(count[1]);
		FUSED_CUDA(cudaMemcpy(left.data(), one->comb + offset[0], count[0] * sizeof(float), cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->comb + offset[1], count[1] * sizeof(float), cudaMemcpyDeviceToHost));
		if ( memcmp(left.data(), right.data(), count[0] * sizeof(float)) != 0 )
		{
			fprintf(stderr, "comb: row 0 differs between %u and %u rows\n", rows_one, rows_two);
			++differ;
		}
	}
	{
		uint64_t offset[2], count[2];
		{ const uint32_t rows = 1u; offset[0] = offset[1] = 0u; count[0] = count[1] = (uint64_t)(rows * D41_HIDDEN); }
		FUSED_REQUIRE(count[0] == count[1]);
		std::vector<uint16_t> left(count[0]), right(count[1]);
		FUSED_CUDA(cudaMemcpy(left.data(), one->x + offset[0], count[0] * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->x + offset[1], count[1] * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		if ( memcmp(left.data(), right.data(), count[0] * sizeof(uint16_t)) != 0 )
		{
			fprintf(stderr, "x: row 0 differs between %u and %u rows\n", rows_one, rows_two);
			++differ;
		}
	}
	{
		uint64_t offset[2], count[2];
		{ const uint32_t rows = 1u; offset[0] = offset[1] = 0u; count[0] = count[1] = (uint64_t)(rows * D41_HIDDEN); }
		FUSED_REQUIRE(count[0] == count[1]);
		std::vector<uint16_t> left(count[0]), right(count[1]);
		FUSED_CUDA(cudaMemcpy(left.data(), one->xq + offset[0], count[0] * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->xq + offset[1], count[1] * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		if ( memcmp(left.data(), right.data(), count[0] * sizeof(uint16_t)) != 0 )
		{
			fprintf(stderr, "xq: row 0 differs between %u and %u rows\n", rows_one, rows_two);
			++differ;
		}
	}
	return differ;
}

static void D41FusedMixNormKernelPoison(const D41FusedMixNormArguments *set, const D41FusedMixNormArguments *origin, cudaStream_t stream)
{
	const uint32_t rows = 16u;
	FUSED_CUDA(cudaMemsetAsync((void *)set->partials, 0xff, (uint64_t)(rows * LmHcSinkhornShape<D41K_HC>::kPartials) * sizeof(float), stream));
	FUSED_CUDA(cudaMemsetAsync((void *)set->pre, 0xff, (uint64_t)(rows * D41K_HC) * sizeof(float), stream));
	FUSED_CUDA(cudaMemsetAsync((void *)set->post, 0xff, (uint64_t)(rows * D41K_HC) * sizeof(float), stream));
	FUSED_CUDA(cudaMemsetAsync((void *)set->comb, 0xff, (uint64_t)(rows * D41K_HC * D41K_HC) * sizeof(float), stream));
	FUSED_CUDA(cudaMemsetAsync((void *)set->x, 0xff, (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t), stream));
	FUSED_CUDA(cudaMemsetAsync((void *)set->xq, 0xff, (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t), stream));
	(void)origin;
	(void)rows;
}

static uint32_t D41FusedMixNormKernelExact(void)
{
	D41FusedMixNormArguments reference, fused, origin;
	cudaStream_t stream = 0;
	uint32_t checks = 0u;
	FUSED_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	for ( uint32_t edges = 0u; edges < 2u; ++edges )
	{
		D41FusedMixNormKernelBuffers(&origin, 0x5000u + 0x100u * edges, edges);
		D41FusedMixNormKernelBuffers(&reference, 0x7000u, 0u);
		D41FusedMixNormKernelBuffers(&fused, 0x7000u, 0u);
		for ( uint32_t rows = 1u; rows <= 16u; ++rows )
		{
			D41FusedMixNormKernelCopy(&reference, &origin);
			D41FusedMixNormKernelCopy(&fused, &origin);
			D41FusedMixNormKernelReference(&reference, rows, stream);
			FUSED_REQUIRE(D41FusedMixNormLaunch(&fused, rows, stream) == LM_LAUNCH_OK);
			FUSED_CUDA(cudaStreamSynchronize(stream));
			const uint64_t differ = D41FusedMixNormKernelDiffer(&reference, &fused, rows, 1u);
			if ( differ != 0u )
				{ fprintf(stderr, "FAIL D41FusedMixNormKernel rows=%u edges=%u: %llu bytes differ\n", rows, edges, (unsigned long long)differ); exit(1); }
			++checks;
		}
	}
	D41FusedMixNormKernelCopy(&reference, &origin);
	D41FusedMixNormKernelCopy(&fused, &origin);
	FUSED_REQUIRE(D41FusedMixNormLaunch(&reference, 1u, stream) == LM_LAUNCH_OK);
	FUSED_REQUIRE(D41FusedMixNormLaunch(&fused, 16u, stream) == LM_LAUNCH_OK);
	FUSED_CUDA(cudaStreamSynchronize(stream));
	FUSED_REQUIRE(D41FusedMixNormKernelRowZero(&reference, 1u, &fused, 16u) == 0u);
	++checks;
	D41FusedMixNormKernelCopy(&reference, &origin);
	D41FusedMixNormKernelCopy(&fused, &origin);
	{
		const uint16_t bump = 0x3f80u;
		FUSED_CUDA(cudaMemcpy((void *)fused.streams, &bump, sizeof(bump), cudaMemcpyHostToDevice));
		FUSED_CUDA(cudaDeviceSynchronize());
	}
	FUSED_REQUIRE(D41FusedMixNormLaunch(&reference, 16u, stream) == LM_LAUNCH_OK);
	FUSED_REQUIRE(D41FusedMixNormLaunch(&fused, 16u, stream) == LM_LAUNCH_OK);
	FUSED_CUDA(cudaStreamSynchronize(stream));
	FUSED_REQUIRE(D41FusedMixNormKernelDiffer(&reference, &fused, 16u, 0u) != 0u);
	++checks;
	D41FusedMixNormKernelCopy(&reference, &origin);
	D41FusedMixNormKernelCopy(&fused, &origin);
	{
		cudaGraph_t graph = 0;
		cudaGraphExec_t exec = 0;
		FUSED_REQUIRE(D41FusedMixNormLaunch(&reference, 16u, stream) == LM_LAUNCH_OK);
		FUSED_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
		FUSED_REQUIRE(D41FusedMixNormLaunch(&fused, 16u, stream) == LM_LAUNCH_OK);
		FUSED_CUDA(cudaStreamEndCapture(stream, &graph));
		FUSED_CUDA(cudaGraphInstantiate(&exec, graph, 0));
		for ( uint32_t round = 0u; round < 3u; ++round )
		{
			D41FusedMixNormKernelPoison(&fused, &origin, stream);
			FUSED_CUDA(cudaGraphLaunch(exec, stream));
		}
		FUSED_CUDA(cudaStreamSynchronize(stream));
		FUSED_REQUIRE(D41FusedMixNormKernelDiffer(&reference, &fused, 16u, 1u) == 0u);
		{
			const uint32_t rows = 16u;
			std::vector<uint32_t> counters((uint64_t)(rows));
			FUSED_CUDA(cudaMemcpy(counters.data(), fused.counters, counters.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
			for ( uint32_t counter : counters )
				FUSED_REQUIRE(counter == 0u);
		}
		FUSED_CUDA(cudaGraphExecDestroy(exec));
		FUSED_CUDA(cudaGraphDestroy(graph));
	}
	++checks;
	printf("PASS D41FusedMixNormKernel: %u comparisons\n", checks);
	return checks;
}

static void D41FusedOutputKernelBuffers(D41FusedOutputArguments *set, uint64_t seed, uint32_t edges)
{
	const uint32_t rows = 16u;
	{
		const uint64_t count = (uint64_t)(rows * D41_LOCAL_HEADS * D41K_RECORD), bytes = count * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		std::vector<float> host(count);
		for ( uint64_t index = 0u; index < count; ++index )
			host[index] = (float)(index % D41K_RECORD == 1u ? 1.0f + (float)(FusedMix(index) % 1024u) / 256.0f : ((float)(FusedMix(index) % 4096u) - 2048.0f) / 512.0f);
		FUSED_CUDA(cudaMemcpy(pointer, host.data(), bytes, cudaMemcpyHostToDevice));
		set->records = (const float *)pointer;
	}
	set->source_stride = (uint64_t)(D41DestinationStride(rows, D41_LOCAL_HEADS));
	{
		const uint64_t count = (uint64_t)(D41_LOCAL_HEADS), bytes = count * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 2u, 2u, edges);
		set->sink = (const float *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows), bytes = count * sizeof(uint32_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		std::vector<uint32_t> host(count);
		for ( uint64_t index = 0u; index < count; ++index )
			host[index] = (uint32_t)(160u + 37u * index);
		FUSED_CUDA(cudaMemcpy(pointer, host.data(), bytes, cudaMemcpyHostToDevice));
		set->positions = (const uint32_t *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(D41K_ROPE_PAIRS), bytes = count * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 4u, 2u, edges);
		set->frequency = (const float *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows * D41_LOCAL_HEADS * D41_HEAD_DIM), bytes = count * sizeof(uint16_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->group_input = (uint16_t *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows * D41_HEAD_DIM), bytes = count * sizeof(uint16_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 6u, 1u, edges);
		set->window_kv = (const uint16_t *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows * D41_RING_ROWS * D41_HEAD_DIM), bytes = count * sizeof(uint16_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->ring = (uint16_t *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows), bytes = count * sizeof(uint32_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		std::vector<uint32_t> host(count);
		for ( uint64_t index = 0u; index < count; ++index )
			host[index] = (uint32_t)(index);
		FUSED_CUDA(cudaMemcpy(pointer, host.data(), bytes, cudaMemcpyHostToDevice));
		set->row_lane = (const uint32_t *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows), bytes = count * sizeof(uint32_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		std::vector<uint32_t> host(count);
		for ( uint64_t index = 0u; index < count; ++index )
			host[index] = (uint32_t)(0u);
		FUSED_CUDA(cudaMemcpy(pointer, host.data(), bytes, cudaMemcpyHostToDevice));
		set->row_ordinal = (const uint32_t *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows + 1u), bytes = count * sizeof(uint32_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		std::vector<uint32_t> host(count);
		for ( uint64_t index = 0u; index < count; ++index )
			host[index] = (uint32_t)(index);
		FUSED_CUDA(cudaMemcpy(pointer, host.data(), bytes, cudaMemcpyHostToDevice));
		set->sequence_row_begin = (const uint32_t *)pointer;
	}
	{
		const uint64_t count = (uint64_t)(rows), bytes = count * sizeof(uint32_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		std::vector<uint32_t> host(count);
		for ( uint64_t index = 0u; index < count; ++index )
			host[index] = (uint32_t)(index);
		FUSED_CUDA(cudaMemcpy(pointer, host.data(), bytes, cudaMemcpyHostToDevice));
		set->recurrent_index = (const uint32_t *)pointer;
	}
	set->ring_slot_stride = (uint64_t)((uint64_t)D41_RING_ROWS * D41_HEAD_DIM);
	set->commit = (uint32_t)(1u);
	(void)rows;
	(void)seed;
	(void)edges;
	FUSED_CUDA(cudaDeviceSynchronize());
}

static void D41FusedOutputKernelCopy(D41FusedOutputArguments *to, const D41FusedOutputArguments *from)
{
	const uint32_t rows = 16u;
	FUSED_CUDA(cudaMemcpy((void *)to->records, from->records, (uint64_t)(rows * D41_LOCAL_HEADS * D41K_RECORD) * sizeof(float), cudaMemcpyDeviceToDevice));
	to->source_stride = from->source_stride;
	FUSED_CUDA(cudaMemcpy((void *)to->sink, from->sink, (uint64_t)(D41_LOCAL_HEADS) * sizeof(float), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->positions, from->positions, (uint64_t)(rows) * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->frequency, from->frequency, (uint64_t)(D41K_ROPE_PAIRS) * sizeof(float), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->group_input, from->group_input, (uint64_t)(rows * D41_LOCAL_HEADS * D41_HEAD_DIM) * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->window_kv, from->window_kv, (uint64_t)(rows * D41_HEAD_DIM) * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->ring, from->ring, (uint64_t)(rows * D41_RING_ROWS * D41_HEAD_DIM) * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->row_lane, from->row_lane, (uint64_t)(rows) * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->row_ordinal, from->row_ordinal, (uint64_t)(rows) * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->sequence_row_begin, from->sequence_row_begin, (uint64_t)(rows + 1u) * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
	FUSED_CUDA(cudaMemcpy((void *)to->recurrent_index, from->recurrent_index, (uint64_t)(rows) * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
	to->ring_slot_stride = from->ring_slot_stride;
	to->commit = from->commit;
	FUSED_CUDA(cudaDeviceSynchronize());
	(void)rows;
}

static struct
{
	uint16_t *merged;
} D41FusedOutputKernelScratch;

static void D41FusedOutputKernelReference(const D41FusedOutputArguments *set, uint32_t rows, cudaStream_t stream)
{
	const float *records = set->records;
	uint64_t source_stride = set->source_stride;
	const float *sink = set->sink;
	const uint32_t *positions = set->positions;
	const float *frequency = set->frequency;
	uint16_t *group_input = set->group_input;
	const uint16_t *window_kv = set->window_kv;
	uint16_t *ring = set->ring;
	const uint32_t *row_lane = set->row_lane;
	const uint32_t *row_ordinal = set->row_ordinal;
	const uint32_t *sequence_row_begin = set->sequence_row_begin;
	const uint32_t *recurrent_index = set->recurrent_index;
	uint64_t ring_slot_stride = set->ring_slot_stride;
	uint32_t commit = set->commit;
	uint16_t *merged = D41FusedOutputKernelScratch.merged;
	(void)records;
	(void)source_stride;
	(void)sink;
	(void)positions;
	(void)frequency;
	(void)group_input;
	(void)window_kv;
	(void)ring;
	(void)row_lane;
	(void)row_ordinal;
	(void)sequence_row_begin;
	(void)recurrent_index;
	(void)ring_slot_stride;
	(void)commit;
	(void)merged;
	LM_LAUNCH((D41WindowRingStoreKernel), rows, D41_THREADS, 0, stream, window_kv, ring, positions, row_lane, row_ordinal, sequence_row_begin, recurrent_index, ring_slot_stride, D41_RING_ROWS, D41_WINDOW, 1u, 0u, rows, commit, (const uint32_t *)0);
	LM_LAUNCH((D41MergeSinkKernel), dim3(rows, D41_LOCAL_HEADS), D41_THREADS, 0, stream, records, source_stride, 1u, D41_LOCAL_HEADS, sink, merged);
	D41Rope(merged, rows, D41_LOCAL_HEADS, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, positions, 0, frequency, 1u, 0, stream);
	for ( uint32_t group = 0u; group < D41_LOCAL_GROUPS; ++group )
	{
		LM_LAUNCH((D41ColumnsKernel), dim3(D41Blocks(D41_GROUP_WIDTH), rows), D41_THREADS, 0, stream, merged + (uint64_t)group * D41_GROUP_WIDTH, D41_LOCAL_HEADS * D41_HEAD_DIM, group_input + (uint64_t)group * rows * D41_GROUP_WIDTH, D41_GROUP_WIDTH, D41_GROUP_WIDTH);
		D41Fp8Qdq(group_input + (uint64_t)group * rows * D41_GROUP_WIDTH, (uint64_t)rows * D41_GROUP_WIDTH, stream);
	}
	FUSED_CUDA(cudaGetLastError());
}

static uint64_t D41FusedOutputKernelDiffer(const D41FusedOutputArguments *one, const D41FusedOutputArguments *two, uint32_t rows, uint32_t report)
{
	uint64_t differ = 0u;
	{
		const uint64_t bytes = (uint64_t)(rows * D41_LOCAL_HEADS * D41_HEAD_DIM) * sizeof(uint16_t);
		std::vector<uint8_t> left(bytes), right(bytes);
		uint64_t count = 0u;
		FUSED_CUDA(cudaMemcpy(left.data(), one->group_input, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->group_input, bytes, cudaMemcpyDeviceToHost));
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "group_input: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * D41_RING_ROWS * D41_HEAD_DIM) * sizeof(uint16_t);
		std::vector<uint8_t> left(bytes), right(bytes);
		uint64_t count = 0u;
		FUSED_CUDA(cudaMemcpy(left.data(), one->ring, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->ring, bytes, cudaMemcpyDeviceToHost));
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "ring: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	return differ;
}

static uint64_t D41FusedOutputKernelRowZero(const D41FusedOutputArguments *one, uint32_t rows_one, const D41FusedOutputArguments *two, uint32_t rows_two)
{
	uint64_t differ = 0u;
	{
		uint64_t offset[2], count[2];
		{ const uint32_t rows = rows_one; offset[0] = (uint64_t)(0u); count[0] = (uint64_t)(D41_GROUP_WIDTH); (void)rows; }
		{ const uint32_t rows = rows_two; offset[1] = (uint64_t)(0u); count[1] = (uint64_t)(D41_GROUP_WIDTH); (void)rows; }
		FUSED_REQUIRE(count[0] == count[1]);
		std::vector<uint16_t> left(count[0]), right(count[1]);
		FUSED_CUDA(cudaMemcpy(left.data(), one->group_input + offset[0], count[0] * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->group_input + offset[1], count[1] * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		if ( memcmp(left.data(), right.data(), count[0] * sizeof(uint16_t)) != 0 )
		{
			fprintf(stderr, "group_input: row 0 differs between %u and %u rows\n", rows_one, rows_two);
			++differ;
		}
	}
	{
		uint64_t offset[2], count[2];
		{ const uint32_t rows = rows_one; offset[0] = (uint64_t)(rows * D41_GROUP_WIDTH); count[0] = (uint64_t)(D41_GROUP_WIDTH); (void)rows; }
		{ const uint32_t rows = rows_two; offset[1] = (uint64_t)(rows * D41_GROUP_WIDTH); count[1] = (uint64_t)(D41_GROUP_WIDTH); (void)rows; }
		FUSED_REQUIRE(count[0] == count[1]);
		std::vector<uint16_t> left(count[0]), right(count[1]);
		FUSED_CUDA(cudaMemcpy(left.data(), one->group_input + offset[0], count[0] * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->group_input + offset[1], count[1] * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		if ( memcmp(left.data(), right.data(), count[0] * sizeof(uint16_t)) != 0 )
		{
			fprintf(stderr, "group_input: row 0 differs between %u and %u rows\n", rows_one, rows_two);
			++differ;
		}
	}
	{
		uint64_t offset[2], count[2];
		{ const uint32_t rows = 1u; offset[0] = offset[1] = 0u; count[0] = count[1] = (uint64_t)(rows * D41_RING_ROWS * D41_HEAD_DIM); }
		FUSED_REQUIRE(count[0] == count[1]);
		std::vector<uint16_t> left(count[0]), right(count[1]);
		FUSED_CUDA(cudaMemcpy(left.data(), one->ring + offset[0], count[0] * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->ring + offset[1], count[1] * sizeof(uint16_t), cudaMemcpyDeviceToHost));
		if ( memcmp(left.data(), right.data(), count[0] * sizeof(uint16_t)) != 0 )
		{
			fprintf(stderr, "ring: row 0 differs between %u and %u rows\n", rows_one, rows_two);
			++differ;
		}
	}
	return differ;
}

static void D41FusedOutputKernelPoison(const D41FusedOutputArguments *set, const D41FusedOutputArguments *origin, cudaStream_t stream)
{
	const uint32_t rows = 16u;
	FUSED_CUDA(cudaMemsetAsync((void *)set->group_input, 0xff, (uint64_t)(rows * D41_LOCAL_HEADS * D41_HEAD_DIM) * sizeof(uint16_t), stream));
	FUSED_CUDA(cudaMemcpyAsync((void *)set->ring, origin->ring, (uint64_t)(rows * D41_RING_ROWS * D41_HEAD_DIM) * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream));
	(void)origin;
	(void)rows;
}

static uint32_t D41FusedOutputKernelExact(void)
{
	D41FusedOutputArguments reference, fused, origin;
	cudaStream_t stream = 0;
	uint32_t checks = 0u;
	FUSED_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	{
		const uint32_t rows = 16u;
		FUSED_CUDA(cudaMalloc((void **)&D41FusedOutputKernelScratch.merged, (uint64_t)(rows * D41_LOCAL_HEADS * D41_HEAD_DIM) * sizeof(uint16_t)));
		(void)rows;
	}
	for ( uint32_t edges = 0u; edges < 2u; ++edges )
	{
		D41FusedOutputKernelBuffers(&origin, 0x5000u + 0x100u * edges, edges);
		D41FusedOutputKernelBuffers(&reference, 0x7000u, 0u);
		D41FusedOutputKernelBuffers(&fused, 0x7000u, 0u);
		for ( uint32_t rows = 1u; rows <= 16u; ++rows )
		{
			D41FusedOutputKernelCopy(&reference, &origin);
			D41FusedOutputKernelCopy(&fused, &origin);
			D41FusedOutputKernelReference(&reference, rows, stream);
			FUSED_REQUIRE(D41FusedOutputLaunch(&fused, rows, stream) == LM_LAUNCH_OK);
			FUSED_CUDA(cudaStreamSynchronize(stream));
			const uint64_t differ = D41FusedOutputKernelDiffer(&reference, &fused, rows, 1u);
			if ( differ != 0u )
				{ fprintf(stderr, "FAIL D41FusedOutputKernel rows=%u edges=%u: %llu bytes differ\n", rows, edges, (unsigned long long)differ); exit(1); }
			++checks;
		}
	}
	D41FusedOutputKernelCopy(&reference, &origin);
	D41FusedOutputKernelCopy(&fused, &origin);
	FUSED_REQUIRE(D41FusedOutputLaunch(&reference, 1u, stream) == LM_LAUNCH_OK);
	FUSED_REQUIRE(D41FusedOutputLaunch(&fused, 16u, stream) == LM_LAUNCH_OK);
	FUSED_CUDA(cudaStreamSynchronize(stream));
	FUSED_REQUIRE(D41FusedOutputKernelRowZero(&reference, 1u, &fused, 16u) == 0u);
	++checks;
	D41FusedOutputKernelCopy(&reference, &origin);
	D41FusedOutputKernelCopy(&fused, &origin);
	{
		const float bump = 1.5f;
		FUSED_CUDA(cudaMemcpy((void *)fused.records, &bump, sizeof(bump), cudaMemcpyHostToDevice));
		FUSED_CUDA(cudaDeviceSynchronize());
	}
	FUSED_REQUIRE(D41FusedOutputLaunch(&reference, 16u, stream) == LM_LAUNCH_OK);
	FUSED_REQUIRE(D41FusedOutputLaunch(&fused, 16u, stream) == LM_LAUNCH_OK);
	FUSED_CUDA(cudaStreamSynchronize(stream));
	FUSED_REQUIRE(D41FusedOutputKernelDiffer(&reference, &fused, 16u, 0u) != 0u);
	++checks;
	D41FusedOutputKernelCopy(&reference, &origin);
	D41FusedOutputKernelCopy(&fused, &origin);
	{
		cudaGraph_t graph = 0;
		cudaGraphExec_t exec = 0;
		FUSED_REQUIRE(D41FusedOutputLaunch(&reference, 16u, stream) == LM_LAUNCH_OK);
		FUSED_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
		FUSED_REQUIRE(D41FusedOutputLaunch(&fused, 16u, stream) == LM_LAUNCH_OK);
		FUSED_CUDA(cudaStreamEndCapture(stream, &graph));
		FUSED_CUDA(cudaGraphInstantiate(&exec, graph, 0));
		for ( uint32_t round = 0u; round < 3u; ++round )
		{
			D41FusedOutputKernelPoison(&fused, &origin, stream);
			FUSED_CUDA(cudaGraphLaunch(exec, stream));
		}
		FUSED_CUDA(cudaStreamSynchronize(stream));
		FUSED_REQUIRE(D41FusedOutputKernelDiffer(&reference, &fused, 16u, 1u) == 0u);
		FUSED_CUDA(cudaGraphExecDestroy(exec));
		FUSED_CUDA(cudaGraphDestroy(graph));
	}
	++checks;
	printf("PASS D41FusedOutputKernel: %u comparisons\n", checks);
	return checks;
}

int main(void)
{
	FUSED_CUDA(cudaSetDevice(0));
	(void)D41FusedMixNormKernelExact();
	(void)D41FusedOutputKernelExact();
	printf("PASS fused kernels\n");
	return 0;
}
