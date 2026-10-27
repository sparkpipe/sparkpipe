#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <cuda_bf16.h>

#include "runtime/gemm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/moe_experts.cuh"

#ifndef BENCH_EXPERTS
#define BENCH_EXPERTS 320u
#endif
#ifndef BENCH_TOP_K
#define BENCH_TOP_K 8u
#endif
#ifndef BENCH_HIDDEN
#define BENCH_HIDDEN 4096u
#endif
#ifndef BENCH_INTER
#define BENCH_INTER 80u
#endif
#define BENCH_ITERATIONS 20u

#define CHECK(call) do { cudaError_t e = (call); if ( e != cudaSuccess ) { printf("FAIL cuda %s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)
#define LAUNCH(call) do { int32_t launch_status = (call); if ( launch_status != LM_LAUNCH_OK ) { printf("FAIL launch %s: %d\n", #call, launch_status); exit(1); } } while (0)

static uint64_t Hash(uint64_t v)
{
	v = v * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	v = (v ^ (v >> 30u)) * 0xbf58476d1ce4e5b9ull;
	v = (v ^ (v >> 27u)) * 0x94d049bb133111ebull;
	return v ^ (v >> 31u);
}

static __global__ void BenchFill(uint16_t *out, uint64_t count, uint64_t seed, float scale)
{
	for ( uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < count; i += (uint64_t)gridDim.x * blockDim.x )
	{
		uint64_t v = (seed * 0x100000001ull + i) * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
		v = (v ^ (v >> 30u)) * 0xbf58476d1ce4e5b9ull;
		v = (v ^ (v >> 27u)) * 0x94d049bb133111ebull;
		v ^= v >> 31u;
		out[i] = __bfloat16_as_ushort(__float2bfloat16(scale * ((float)(v & 0xffffu) / 32768.0f - 1.0f)));
	}
}

static uint16_t *Filled(uint64_t count, uint64_t seed, float scale)
{
	uint16_t *device;
	CHECK(cudaMalloc(&device, count * 2u));
	BenchFill<<<512u, 256u>>>(device, count, seed, scale);
	CHECK(cudaGetLastError());
	return device;
}

static uint64_t Digest(const uint16_t *device, uint64_t count)
{
	std::vector<uint16_t> host(count);
	uint64_t h = 0xcbf29ce484222325ull;
	CHECK(cudaMemcpy(host.data(), device, count * 2u, cudaMemcpyDeviceToHost));
	for ( uint16_t v : host )
		h = (h ^ v) * 0x100000001b3ull;
	return h;
}

static float Elapsed(cudaEvent_t start, cudaEvent_t stop)
{
	float ms = 0.0f;
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&ms, start, stop));
	return ms;
}

int main(int argc, char **argv)
{
	const uint32_t rows = argc > 1 ? (uint32_t)strtoul(argv[1], 0, 10) : 2048u;
	const uint32_t routes = rows * BENCH_TOP_K;
	const uint64_t weight_elements = (uint64_t)BENCH_EXPERTS * BENCH_INTER * BENCH_HIDDEN;
	std::vector<uint32_t> expert(routes);
	std::vector<float> weight(routes, 1.0f / BENCH_TOP_K);
	uint32_t *route_expert, *route_global, *group_row_offset, *route_packed_row, *route_source_token, *tile_prefix_up, *tile_prefix_down;
	float *route_weight;
	cudaStream_t s;
	cudaEvent_t start, stop;
	int sms = 0;
	CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
	CHECK(cudaStreamCreate(&s));
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	for ( uint32_t t = 0; t < rows; t++ )
	{
		uint32_t chosen = 0;
		for ( uint64_t probe = 0; chosen < BENCH_TOP_K; probe++ )
		{
			uint32_t e = (uint32_t)(Hash(((uint64_t)t << 20u) + probe) % BENCH_EXPERTS);
			bool seen = false;
			for ( uint32_t j = 0; j < chosen; j++ )
				seen = seen || expert[(uint64_t)t * BENCH_TOP_K + j] == e;
			if ( !seen )
				expert[(uint64_t)t * BENCH_TOP_K + chosen++] = e;
		}
	}
	CHECK(cudaMalloc(&route_expert, routes * 4u));
	CHECK(cudaMalloc(&route_global, routes * 4u));
	CHECK(cudaMalloc(&route_weight, routes * 4u));
	CHECK(cudaMalloc(&group_row_offset, (BENCH_EXPERTS + 2u) * 4u));
	CHECK(cudaMalloc(&route_packed_row, routes * 4u));
	CHECK(cudaMalloc(&route_source_token, routes * 4u));
	CHECK(cudaMalloc(&tile_prefix_up, (BENCH_EXPERTS + 2u) * 4u));
	CHECK(cudaMalloc(&tile_prefix_down, (BENCH_EXPERTS + 2u) * 4u));
	CHECK(cudaMemcpy(route_expert, expert.data(), routes * 4u, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(route_weight, weight.data(), routes * 4u, cudaMemcpyHostToDevice));
	LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u, 0, s>>>(route_expert, route_weight, route_global, routes, 0u, BENCH_EXPERTS);
	LAUNCH((LmRouteBuild<256u, BENCH_EXPERTS + 1u>(route_expert, rows, routes, BENCH_TOP_K, group_row_offset, route_packed_row,
		route_source_token, BENCH_INTER, BENCH_HIDDEN, LM_MOE_GEMM_TILE_N, tile_prefix_up, tile_prefix_down, s)));
	uint16_t *w_up = Filled(weight_elements, 1u, 0.05f), *w_down = Filled(weight_elements, 2u, 0.05f);
	uint16_t *activation = Filled((uint64_t)rows * BENCH_HIDDEN, 3u, 1.0f), *act = Filled((uint64_t)routes * BENCH_INTER, 4u, 1.0f);
	uint16_t *up_out, *down_out;
	CHECK(cudaMalloc(&up_out, (uint64_t)routes * BENCH_INTER * 2u));
	CHECK(cudaMalloc(&down_out, (uint64_t)routes * BENCH_HIDDEN * 2u));
	CHECK(cudaDeviceSynchronize());
	LmLocalExpertRoutes r;
	r.group_row_offset = group_row_offset;
	r.route_source_token = route_source_token;
	r.tile_prefix_up = tile_prefix_up;
	r.tile_prefix_down = tile_prefix_down;
	r.rows = rows;
	r.routes = routes;
	r.top_k = BENCH_TOP_K;
	r.local_experts = BENCH_EXPERTS;
	r.multiprocessors = (uint32_t)sms;
	LAUNCH(LmLocalExpertsUp<LmBf16Format>(r, w_up, LmScaleTensorNone(), activation, up_out, BENCH_HIDDEN, BENCH_INTER, s));
	LAUNCH(LmLocalExpertsDown<LmBf16Format>(r, w_down, LmScaleTensorNone(), act, down_out, BENCH_INTER, BENCH_HIDDEN, s));
	CHECK(cudaStreamSynchronize(s));
	const uint64_t up_digest = Digest(up_out, (uint64_t)routes * BENCH_INTER), down_digest = Digest(down_out, (uint64_t)routes * BENCH_HIDDEN);
	CHECK(cudaEventRecord(start, s));
	for ( uint32_t i = 0; i < BENCH_ITERATIONS; i++ )
		LAUNCH(LmLocalExpertsUp<LmBf16Format>(r, w_up, LmScaleTensorNone(), activation, up_out, BENCH_HIDDEN, BENCH_INTER, s));
	CHECK(cudaEventRecord(stop, s));
	const double up_us = 1000.0 * Elapsed(start, stop) / BENCH_ITERATIONS;
	CHECK(cudaEventRecord(start, s));
	for ( uint32_t i = 0; i < BENCH_ITERATIONS; i++ )
		LAUNCH(LmLocalExpertsDown<LmBf16Format>(r, w_down, LmScaleTensorNone(), act, down_out, BENCH_INTER, BENCH_HIDDEN, s));
	CHECK(cudaEventRecord(stop, s));
	const double down_us = 1000.0 * Elapsed(start, stop) / BENCH_ITERATIONS;
	if ( Digest(up_out, (uint64_t)routes * BENCH_INTER) != up_digest || Digest(down_out, (uint64_t)routes * BENCH_HIDDEN) != down_digest )
	{
		printf("FAIL expert_gemm_bench: repeated launches changed the output\n");
		return 1;
	}
	const double flop = 2.0 * routes * BENCH_HIDDEN * BENCH_INTER;
	const double up_bytes = weight_elements * 2.0 + (double)routes * BENCH_HIDDEN * 2.0 + (double)routes * BENCH_INTER * 2.0;
	const double down_bytes = weight_elements * 2.0 + (double)routes * BENCH_INTER * 2.0 + (double)routes * BENCH_HIDDEN * 2.0;
	printf("expert_gemm_bench experts=%u top_k=%u hidden=%u inter=%u rows=%u routes=%u sms=%d\n",
		BENCH_EXPERTS, BENCH_TOP_K, BENCH_HIDDEN, BENCH_INTER, rows, routes, sms);
	printf("up   %9.1f us %6.2f TFLOP/s %6.1f GB/s digest %016llx\n", up_us, flop / up_us / 1e6, up_bytes / up_us / 1e3,
		(unsigned long long)up_digest);
	printf("down %9.1f us %6.2f TFLOP/s %6.1f GB/s digest %016llx\n", down_us, flop / down_us / 1e6, down_bytes / down_us / 1e3,
		(unsigned long long)down_digest);
	return 0;
}
