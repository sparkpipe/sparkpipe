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

static __device__ __forceinline__ uint64_t FusedMix(uint64_t value)
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
		const uint64_t bytes = (uint64_t)(rows * D41K_HC * D41_HIDDEN) * sizeof(uint16_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 0u, 1u, edges);
		set->streams = (const uint16_t *)pointer;
		(void)rows;
	}
	{
		const uint64_t bytes = (uint64_t)(LmHcSinkhornShape<D41K_HC>::kMix * D41K_HC * D41_HIDDEN) * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 1u, 2u, edges);
		set->fn = (const float *)pointer;
		(void)rows;
	}
	{
		const uint64_t bytes = (uint64_t)(4u) * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 2u, 2u, edges);
		set->scale = (const float *)pointer;
		(void)rows;
	}
	{
		const uint64_t bytes = (uint64_t)(LmHcSinkhornShape<D41K_HC>::kMix) * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 3u, 2u, edges);
		set->base = (const float *)pointer;
		(void)rows;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * LmHcSinkhornShape<D41K_HC>::kPartials) * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->partials = (float *)pointer;
		(void)rows;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * D41K_HC) * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->pre = (float *)pointer;
		(void)rows;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * D41K_HC) * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->post = (float *)pointer;
		(void)rows;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * D41K_HC * D41K_HC) * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->comb = (float *)pointer;
		(void)rows;
	}
	{
		const uint64_t bytes = (uint64_t)(rows) * sizeof(uint32_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->counters = (uint32_t *)pointer;
		(void)rows;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * D41K_HC) * sizeof(float);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 9u, 2u, edges);
		set->collapse_pre = (const float *)pointer;
		(void)rows;
	}
	{
		const uint64_t bytes = (uint64_t)(D41_HIDDEN) * sizeof(uint16_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FusedFillKernel<<<256, 256>>>((uint8_t *)pointer, bytes, seed + 10u, 1u, edges);
		set->norm = (const uint16_t *)pointer;
		(void)rows;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->x = (uint16_t *)pointer;
		(void)rows;
	}
	{
		const uint64_t bytes = (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t);
		void *pointer = 0;
		FUSED_CUDA(cudaMalloc(&pointer, bytes));
		FUSED_CUDA(cudaMemset(pointer, 0, bytes));
		set->xq = (uint16_t *)pointer;
		(void)rows;
	}
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

static uint64_t D41FusedMixNormKernelDiffer(const D41FusedMixNormArguments *one, const D41FusedMixNormArguments *two, uint32_t rows, uint32_t first_row_only, uint32_t report)
{
	uint64_t differ = 0u;
	{
		const uint64_t bytes = first_row_only != 0u ? (uint64_t)(rows * LmHcSinkhornShape<D41K_HC>::kPartials) * sizeof(float) / rows : (uint64_t)(rows * LmHcSinkhornShape<D41K_HC>::kPartials) * sizeof(float);
		std::vector<uint8_t> left(bytes), right(bytes);
		FUSED_CUDA(cudaMemcpy(left.data(), one->partials, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->partials, bytes, cudaMemcpyDeviceToHost));
		uint64_t count = 0u;
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "partials: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	{
		const uint64_t bytes = first_row_only != 0u ? (uint64_t)(rows * D41K_HC) * sizeof(float) / rows : (uint64_t)(rows * D41K_HC) * sizeof(float);
		std::vector<uint8_t> left(bytes), right(bytes);
		FUSED_CUDA(cudaMemcpy(left.data(), one->pre, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->pre, bytes, cudaMemcpyDeviceToHost));
		uint64_t count = 0u;
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "pre: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	{
		const uint64_t bytes = first_row_only != 0u ? (uint64_t)(rows * D41K_HC) * sizeof(float) / rows : (uint64_t)(rows * D41K_HC) * sizeof(float);
		std::vector<uint8_t> left(bytes), right(bytes);
		FUSED_CUDA(cudaMemcpy(left.data(), one->post, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->post, bytes, cudaMemcpyDeviceToHost));
		uint64_t count = 0u;
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "post: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	{
		const uint64_t bytes = first_row_only != 0u ? (uint64_t)(rows * D41K_HC * D41K_HC) * sizeof(float) / rows : (uint64_t)(rows * D41K_HC * D41K_HC) * sizeof(float);
		std::vector<uint8_t> left(bytes), right(bytes);
		FUSED_CUDA(cudaMemcpy(left.data(), one->comb, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->comb, bytes, cudaMemcpyDeviceToHost));
		uint64_t count = 0u;
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "comb: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	{
		const uint64_t bytes = first_row_only != 0u ? (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t) / rows : (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t);
		std::vector<uint8_t> left(bytes), right(bytes);
		FUSED_CUDA(cudaMemcpy(left.data(), one->x, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->x, bytes, cudaMemcpyDeviceToHost));
		uint64_t count = 0u;
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "x: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	{
		const uint64_t bytes = first_row_only != 0u ? (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t) / rows : (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t);
		std::vector<uint8_t> left(bytes), right(bytes);
		FUSED_CUDA(cudaMemcpy(left.data(), one->xq, bytes, cudaMemcpyDeviceToHost));
		FUSED_CUDA(cudaMemcpy(right.data(), two->xq, bytes, cudaMemcpyDeviceToHost));
		uint64_t count = 0u;
		for ( uint64_t index = 0u; index < bytes; ++index )
			count += left[index] != right[index] ? 1u : 0u;
		if ( report != 0u && count != 0u )
			fprintf(stderr, "xq: %llu of %llu bytes differ\n", (unsigned long long)count, (unsigned long long)bytes);
		differ += count;
	}
	return differ;
}

static void D41FusedMixNormKernelPoison(const D41FusedMixNormArguments *set, cudaStream_t stream)
{
	const uint32_t rows = 16u;
	FUSED_CUDA(cudaMemsetAsync((void *)set->partials, 0xff, (uint64_t)(rows * LmHcSinkhornShape<D41K_HC>::kPartials) * sizeof(float), stream));
	FUSED_CUDA(cudaMemsetAsync((void *)set->pre, 0xff, (uint64_t)(rows * D41K_HC) * sizeof(float), stream));
	FUSED_CUDA(cudaMemsetAsync((void *)set->post, 0xff, (uint64_t)(rows * D41K_HC) * sizeof(float), stream));
	FUSED_CUDA(cudaMemsetAsync((void *)set->comb, 0xff, (uint64_t)(rows * D41K_HC * D41K_HC) * sizeof(float), stream));
	FUSED_CUDA(cudaMemsetAsync((void *)set->x, 0xff, (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t), stream));
	FUSED_CUDA(cudaMemsetAsync((void *)set->xq, 0xff, (uint64_t)(rows * D41_HIDDEN) * sizeof(uint16_t), stream));
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
			const uint64_t differ = D41FusedMixNormKernelDiffer(&reference, &fused, rows, 0u, 1u);
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
	FUSED_REQUIRE(D41FusedMixNormKernelDiffer(&reference, &fused, 1u, 1u, 1u) == 0u);
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
	FUSED_REQUIRE(D41FusedMixNormKernelDiffer(&reference, &fused, 16u, 0u, 0u) != 0u);
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
			D41FusedMixNormKernelPoison(&fused, stream);
			FUSED_CUDA(cudaGraphLaunch(exec, stream));
		}
		FUSED_CUDA(cudaStreamSynchronize(stream));
		FUSED_REQUIRE(D41FusedMixNormKernelDiffer(&reference, &fused, 16u, 0u, 1u) == 0u);
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

int main(void)
{
	FUSED_CUDA(cudaSetDevice(0));
	(void)D41FusedMixNormKernelExact();
	printf("PASS fused kernels\n");
	return 0;
}
