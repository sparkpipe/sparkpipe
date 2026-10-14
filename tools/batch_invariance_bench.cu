#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include "runtime/gemm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/moe_experts.cuh"

#define BENCH_MAX_ROWS 1024u
#define BENCH_WEIGHT_BYTES (512ull << 20)
#define BENCH_ITERATIONS 40u
#define BENCH_TOP_K 8u
#define BENCH_EXPERTS 512u
#define BENCH_LOCAL 32u
#define BENCH_EXPERT_HIDDEN 2560u
#define BENCH_EXPERT_INTER 768u
#define BENCH_CHUNK_TOKENS 64u
#define BENCH_SLOTS 64u

#define CHECK(call) do { cudaError_t e = (call); if ( e != cudaSuccess ) { printf("FAIL cuda %s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)
#define LAUNCH(call) do { int32_t launch_status = (call); if ( launch_status != LM_LAUNCH_OK ) { printf("FAIL launch %s: %d\n", #call, launch_status); exit(1); } } while (0)

static uint32_t sms;
static uint32_t straggler;
static const uint32_t kRows[] = {1u, 16u, 64u, 256u, 1024u};
static const char *kOption[] = {"current", "skinny_all", "tensor_all"};

static uint64_t Hash(uint64_t v)
{
	v = v * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	v = (v ^ (v >> 30u)) * 0xbf58476d1ce4e5b9ull;
	v = (v ^ (v >> 27u)) * 0x94d049bb133111ebull;
	return v ^ (v >> 31u);
}

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, 4u);
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return (uint16_t)(bits >> 16u);
}

static void Fill(uint16_t *device, uint64_t count, uint64_t seed, float scale)
{
	std::vector<uint16_t> host(count);
	for ( uint64_t i = 0; i < count; i++ )
		host[i] = Bf16(scale * ((float)(Hash(seed * 0x100000001ull + i) & 0xffffu) / 32768.0f - 1.0f));
	CHECK(cudaMemcpy(device, host.data(), count * 2u, cudaMemcpyHostToDevice));
}

static int32_t Dense(int option, const uint16_t *w, const uint16_t *a, uint16_t *o, uint32_t rows, uint32_t k, uint32_t n, cudaStream_t s)
{
	if ( option == 1 )
	{
		for ( uint32_t r = 0; r < rows; r += LM_SKINNY_ROWS_WIDE )
		{
			uint32_t c = rows - r < LM_SKINNY_ROWS_WIDE ? rows - r : LM_SKINNY_ROWS_WIDE;
			int32_t status = LmSkinnyDense<LmBf16Format>(w, a + (uint64_t)r * k, o + (uint64_t)r * n, 0, c, k, n, 0u, 0u, s);
			if ( status != LM_LAUNCH_OK )
				return status;
		}
		return LM_LAUNCH_OK;
	}
	if ( option == 0 && rows <= LM_SKINNY_ROWS_WIDE )
	{
		int32_t status = LmSkinnyDense<LmBf16Format>(w, a, o, 0, rows, k, n, 0u, 0u, s);
		if ( status != LM_LAUNCH_ERR_SHAPE )
			return status;
	}
	return LmStreamGemmDense<LmBf16Format>(w, LmScaleTensorNone(), a, o, 0, rows, k, n, 0u, 0u, sms, s);
}

static float Elapsed(cudaEvent_t start, cudaEvent_t stop)
{
	float ms = 0.0f;
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&ms, start, stop));
	return ms;
}

static void DenseCase(const char *name, uint32_t k, uint32_t n, cudaStream_t s)
{
	const uint64_t weight_elements = (uint64_t)k * n;
	uint32_t copies = (uint32_t)std::min<uint64_t>(64u, std::max<uint64_t>(2u, BENCH_WEIGHT_BYTES / (weight_elements * 2u)));
	uint16_t *weight, *activation, *output, *reference;
	cudaEvent_t start, stop;
	CHECK(cudaMalloc(&weight, weight_elements * 2u * copies));
	CHECK(cudaMalloc(&activation, (uint64_t)BENCH_MAX_ROWS * k * 2u));
	CHECK(cudaMalloc(&output, (uint64_t)BENCH_MAX_ROWS * n * 2u));
	CHECK(cudaMalloc(&reference, (uint64_t)BENCH_MAX_ROWS * n * 2u * 3u));
	for ( uint32_t c = 0; c < copies; c++ )
		Fill(weight + c * weight_elements, weight_elements, 11u, 0.03f);
	Fill(activation, (uint64_t)BENCH_MAX_ROWS * k, 7u, 1.0f);
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	for ( int option = 0; option < 3; option++ )
		LAUNCH(Dense(option, weight, activation, reference + (uint64_t)option * BENCH_MAX_ROWS * n, BENCH_MAX_ROWS, k, n, s));
	CHECK(cudaStreamSynchronize(s));
	std::vector<uint16_t> ref((uint64_t)BENCH_MAX_ROWS * n * 3u), got((uint64_t)BENCH_MAX_ROWS * n);
	CHECK(cudaMemcpy(ref.data(), reference, ref.size() * 2u, cudaMemcpyDeviceToHost));
	printf("DENSE %s k=%u n=%u weight_mb=%.1f copies=%u stream_vs_skinny_rows_differ_at_1024=%s\n", name, k, n, weight_elements * 2.0 / 1048576.0,
		copies, memcmp(ref.data(), ref.data() + (uint64_t)BENCH_MAX_ROWS * n, (uint64_t)BENCH_MAX_ROWS * n * 2u) != 0 ? "yes" : "no");
	for ( uint32_t m : kRows )
		for ( int option = 0; option < 3; option++ )
		{
			LAUNCH(Dense(option, weight, activation, output, m, k, n, s));
			CHECK(cudaStreamSynchronize(s));
			CHECK(cudaMemcpy(got.data(), output, (uint64_t)m * n * 2u, cudaMemcpyDeviceToHost));
			const uint16_t *want = ref.data() + (uint64_t)option * BENCH_MAX_ROWS * n;
			uint64_t differ = 0;
			for ( uint64_t i = 0; i < (uint64_t)m * n; i++ )
				differ += got[i] != want[i];
			for ( uint32_t i = 0; i < 3u; i++ )
				LAUNCH(Dense(option, weight + (i % copies) * weight_elements, activation, output, m, k, n, s));
			CHECK(cudaEventRecord(start, s));
			for ( uint32_t i = 0; i < BENCH_ITERATIONS; i++ )
				LAUNCH(Dense(option, weight + (i % copies) * weight_elements, activation, output, m, k, n, s));
			CHECK(cudaEventRecord(stop, s));
			const double us = 1000.0 * Elapsed(start, stop) / BENCH_ITERATIONS;
			printf("DENSE %s option=%s rows=%u us=%.2f weight_gbps=%.0f tflops=%.2f rows_differ_from_m1024=%llu/%llu\n", name, kOption[option], m, us,
				weight_elements * 2.0 / us / 1000.0, 2.0 * m * weight_elements / us / 1e6, (unsigned long long)differ, (unsigned long long)m * n);
		}
	CHECK(cudaFree(weight));
	CHECK(cudaFree(activation));
	CHECK(cudaFree(output));
	CHECK(cudaFree(reference));
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));
}

struct RouteSet
{
	uint32_t tokens, routes, local;
	uint32_t *route_expert, *route_global, *group_row_offset, *route_packed_row, *route_source_token, *tile_prefix_up, *tile_prefix_down;
	float *route_weight;
	std::vector<uint32_t> packed_row, expert;
};

static void BuildRoutes(RouteSet *set, uint32_t first_token, uint32_t tokens, cudaStream_t s)
{
	std::vector<uint32_t> expert((uint64_t)tokens * BENCH_TOP_K);
	std::vector<float> weight(expert.size(), 0.125f);
	for ( uint32_t t = 0; t < tokens; t++ )
	{
		uint32_t chosen = 0;
		for ( uint64_t probe = 0; chosen < BENCH_TOP_K; probe++ )
		{
			uint32_t e = (uint32_t)(Hash(((uint64_t)(first_token + t) << 20u) + probe) % BENCH_EXPERTS);
			bool seen = false;
			for ( uint32_t j = 0; j < chosen; j++ )
				seen = seen || expert[(uint64_t)t * BENCH_TOP_K + j] == e;
			if ( !seen )
				expert[(uint64_t)t * BENCH_TOP_K + chosen++] = e;
		}
	}
	if ( straggler != 0u )
	{
		uint32_t count[BENCH_EXPERTS / BENCH_LOCAL] = {0}, busiest = 0;
		for ( uint32_t e : expert )
			count[e / BENCH_LOCAL]++;
		for ( uint32_t r = 1; r < BENCH_EXPERTS / BENCH_LOCAL; r++ )
			busiest = count[r] > count[busiest] ? r : busiest;
		for ( uint32_t &e : expert )
			e = (e + BENCH_EXPERTS - busiest * BENCH_LOCAL) % BENCH_EXPERTS;
	}
	set->tokens = tokens;
	set->routes = tokens * BENCH_TOP_K;
	set->local = 0;
	for ( uint32_t e : expert )
		set->local += e < BENCH_LOCAL;
	CHECK(cudaMalloc(&set->route_expert, set->routes * 4u));
	CHECK(cudaMalloc(&set->route_global, set->routes * 4u));
	CHECK(cudaMalloc(&set->route_weight, set->routes * 4u));
	CHECK(cudaMalloc(&set->group_row_offset, 4096u));
	CHECK(cudaMalloc(&set->route_packed_row, set->routes * 4u));
	CHECK(cudaMalloc(&set->route_source_token, set->routes * 4u));
	CHECK(cudaMalloc(&set->tile_prefix_up, 4096u));
	CHECK(cudaMalloc(&set->tile_prefix_down, 4096u));
	CHECK(cudaMemcpy(set->route_expert, expert.data(), set->routes * 4u, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(set->route_weight, weight.data(), set->routes * 4u, cudaMemcpyHostToDevice));
	LmRouteLocalKernel<<<(set->routes + 255u) / 256u, 256u, 0, s>>>(set->route_expert, set->route_weight, set->route_global, set->routes, 0u, BENCH_LOCAL);
	LAUNCH((LmRouteBuild<256u, BENCH_LOCAL + 1u>(set->route_expert, tokens, set->routes, BENCH_TOP_K, set->group_row_offset, set->route_packed_row,
		set->route_source_token, BENCH_EXPERT_INTER, BENCH_EXPERT_HIDDEN, LM_MOE_GEMM_TILE_N, set->tile_prefix_up, set->tile_prefix_down, s)));
	CHECK(cudaStreamSynchronize(s));
	set->packed_row.resize(set->routes);
	set->expert = expert;
	CHECK(cudaMemcpy(set->packed_row.data(), set->route_packed_row, set->routes * 4u, cudaMemcpyDeviceToHost));
}

static void FreeRoutes(RouteSet *set)
{
	cudaFree(set->route_expert); cudaFree(set->route_global); cudaFree(set->route_weight); cudaFree(set->group_row_offset);
	cudaFree(set->route_packed_row); cudaFree(set->route_source_token); cudaFree(set->tile_prefix_up); cudaFree(set->tile_prefix_down);
}

static LmLocalExpertRoutes Routes(const RouteSet *set)
{
	LmLocalExpertRoutes r;
	r.group_row_offset = set->group_row_offset;
	r.route_source_token = set->route_source_token;
	r.tile_prefix_up = set->tile_prefix_up;
	r.tile_prefix_down = set->tile_prefix_down;
	r.rows = set->tokens;
	r.routes = set->routes;
	r.top_k = BENCH_TOP_K;
	r.local_experts = BENCH_LOCAL;
	r.multiprocessors = sms;
	return r;
}

static int32_t GemmUp(const LmLocalExpertRoutes &routes, const void *weight, const uint16_t *activation, uint16_t *output, cudaStream_t s)
{
	LmGemmArguments gemm;
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.prefix_built = 1u;
	gemm.group_row_offset = routes.group_row_offset;
	gemm.group_tile_prefix = routes.tile_prefix_up;
	gemm.source_row_map = routes.route_source_token;
	gemm.source_row_count = routes.rows;
	gemm.output_bf16 = output;
	return LmGemmWeightOnlyIndirectLaunch<LmBf16Format,LM_MOE_GEMM_TILE_N,LM_MOE_GEMM_STAGES,LM_MOE_GEMM_WARPS>(&gemm, activation, weight,
		routes.routes, routes.rows, routes.top_k, routes.local_experts, BENCH_EXPERT_HIDDEN, BENCH_EXPERT_INTER, routes.multiprocessors, s);
}

static int32_t GemmDown(const LmLocalExpertRoutes &routes, const void *weight, const uint16_t *activation, uint16_t *output, cudaStream_t s)
{
	LmGemmArguments gemm;
	memset(&gemm, 0, sizeof(gemm));
	gemm.scale_a = LmScaleTensorNone();
	gemm.scale_b = LmScaleTensorNone();
	gemm.prefix_built = 1u;
	gemm.group_row_offset = routes.group_row_offset;
	gemm.group_tile_prefix = routes.tile_prefix_down;
	gemm.output_bf16 = output;
	return LmGemmWeightOnlyLaunch<LmBf16Format,LM_MOE_GEMM_TILE_N,LM_MOE_GEMM_STAGES,LM_MOE_GEMM_WARPS>(&gemm, activation, weight, routes.routes,
		routes.rows, routes.top_k, routes.local_experts, BENCH_EXPERT_INTER, BENCH_EXPERT_HIDDEN, routes.multiprocessors, true, s);
}

struct ExpertBuffers
{
	uint16_t *w1, *w3, *w2, *normed, *gate, *up, *act, *out;
};

static int32_t ExpertLayer(int option, const RouteSet *set, const ExpertBuffers &b, const uint16_t *normed, uint16_t *gate, uint16_t *up,
	const uint16_t *act, uint16_t *out, cudaStream_t s)
{
	const LmLocalExpertRoutes routes = Routes(set);
	if ( option == 2 )
	{
		int32_t status = GemmUp(routes, b.w1, normed, gate, s);
		if ( status == LM_LAUNCH_OK ) status = GemmUp(routes, b.w3, normed, up, s);
		if ( status == LM_LAUNCH_OK ) status = GemmDown(routes, b.w2, act, out, s);
		return status;
	}
	int32_t status = LmLocalExpertsUp<LmBf16Format>(routes, b.w1, LmScaleTensorNone(), normed, gate, BENCH_EXPERT_HIDDEN, BENCH_EXPERT_INTER, s);
	if ( status == LM_LAUNCH_OK ) status = LmLocalExpertsUp<LmBf16Format>(routes, b.w3, LmScaleTensorNone(), normed, up, BENCH_EXPERT_HIDDEN, BENCH_EXPERT_INTER, s);
	if ( status == LM_LAUNCH_OK ) status = LmLocalExpertsDown<LmBf16Format>(routes, b.w2, LmScaleTensorNone(), act, out, BENCH_EXPERT_INTER, BENCH_EXPERT_HIDDEN, s);
	return status;
}

static void ExpertCase(cudaStream_t s)
{
	const uint64_t up_elements = (uint64_t)BENCH_LOCAL * BENCH_EXPERT_INTER * BENCH_EXPERT_HIDDEN;
	const uint64_t routes_max = (uint64_t)BENCH_MAX_ROWS * BENCH_TOP_K;
	ExpertBuffers b;
	cudaEvent_t start, stop;
	CHECK(cudaMalloc(&b.w1, up_elements * 2u));
	CHECK(cudaMalloc(&b.w3, up_elements * 2u));
	CHECK(cudaMalloc(&b.w2, up_elements * 2u));
	CHECK(cudaMalloc(&b.normed, (uint64_t)BENCH_MAX_ROWS * BENCH_EXPERT_HIDDEN * 2u));
	CHECK(cudaMalloc(&b.gate, routes_max * BENCH_EXPERT_INTER * 2u));
	CHECK(cudaMalloc(&b.up, routes_max * BENCH_EXPERT_INTER * 2u));
	CHECK(cudaMalloc(&b.act, routes_max * BENCH_EXPERT_INTER * 2u));
	CHECK(cudaMalloc(&b.out, routes_max * BENCH_EXPERT_HIDDEN * 2u));
	Fill(b.w1, up_elements, 21u, 0.03f);
	Fill(b.w3, up_elements, 22u, 0.03f);
	Fill(b.w2, up_elements, 23u, 0.03f);
	Fill(b.normed, (uint64_t)BENCH_MAX_ROWS * BENCH_EXPERT_HIDDEN, 24u, 1.0f);
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	std::vector<uint16_t> act_route(routes_max * BENCH_EXPERT_INTER);
	for ( uint64_t i = 0; i < act_route.size(); i++ )
		act_route[i] = Bf16((float)(Hash(0x5151u + i) & 0xffffu) / 32768.0f - 1.0f);
	std::vector<uint16_t> gate_ref[3], out_ref[3];
	printf("EXPERTS hidden=%u inter=%u experts=%u local=%u top_k=%u weight_mb_per_matrix=%.1f chunk_tokens=%u\n", BENCH_EXPERT_HIDDEN, BENCH_EXPERT_INTER,
		BENCH_EXPERTS, BENCH_LOCAL, BENCH_TOP_K, up_elements * 2.0 / 1048576.0, BENCH_CHUNK_TOKENS);
	for ( int pass = 0; pass < 2; pass++ )
		for ( uint32_t m : (pass == 0 ? std::vector<uint32_t>{BENCH_MAX_ROWS} : std::vector<uint32_t>(kRows, kRows + 5)) )
			for ( int option = 0; option < 3; option++ )
			{
				const uint32_t chunk = option == 1 ? BENCH_CHUNK_TOKENS : m;
				std::vector<RouteSet> sets;
				std::vector<uint64_t> route_base;
				uint64_t base = 0;
				uint32_t local = 0;
				for ( uint32_t t = 0; t < m; t += chunk )
				{
					RouteSet set;
					BuildRoutes(&set, t, std::min(chunk, m - t), s);
					local += set.local;
					sets.push_back(set);
					route_base.push_back(base);
					base += set.routes;
				}
				std::vector<uint16_t> act_host((uint64_t)m * BENCH_TOP_K * BENCH_EXPERT_INTER);
				for ( size_t c = 0; c < sets.size(); c++ )
					for ( uint32_t r = 0; r < sets[c].routes; r++ )
						memcpy(act_host.data() + (route_base[c] + sets[c].packed_row[r]) * BENCH_EXPERT_INTER,
							act_route.data() + ((uint64_t)c * chunk * BENCH_TOP_K + r) * BENCH_EXPERT_INTER, BENCH_EXPERT_INTER * 2u);
				CHECK(cudaMemcpy(b.act, act_host.data(), act_host.size() * 2u, cudaMemcpyHostToDevice));
				CHECK(cudaMemset(b.gate, 0, routes_max * BENCH_EXPERT_INTER * 2u));
				CHECK(cudaMemset(b.out, 0, routes_max * BENCH_EXPERT_HIDDEN * 2u));
				auto run = [&]() {
					for ( size_t c = 0; c < sets.size(); c++ )
						LAUNCH(ExpertLayer(option, &sets[c], b, b.normed + (uint64_t)c * chunk * BENCH_EXPERT_HIDDEN, b.gate + route_base[c] * BENCH_EXPERT_INTER,
							b.up + route_base[c] * BENCH_EXPERT_INTER, b.act + route_base[c] * BENCH_EXPERT_INTER, b.out + route_base[c] * BENCH_EXPERT_HIDDEN, s));
				};
				run();
				CHECK(cudaStreamSynchronize(s));
				std::vector<uint16_t> gate((uint64_t)m * BENCH_TOP_K * BENCH_EXPERT_INTER), out((uint64_t)m * BENCH_TOP_K * BENCH_EXPERT_HIDDEN);
				CHECK(cudaMemcpy(gate.data(), b.gate, gate.size() * 2u, cudaMemcpyDeviceToHost));
				CHECK(cudaMemcpy(out.data(), b.out, out.size() * 2u, cudaMemcpyDeviceToHost));
				std::vector<uint16_t> gate_route((uint64_t)m * BENCH_TOP_K * BENCH_EXPERT_INTER, 0), out_route((uint64_t)m * BENCH_TOP_K * BENCH_EXPERT_HIDDEN, 0);
				for ( size_t c = 0; c < sets.size(); c++ )
					for ( uint32_t r = 0; r < sets[c].routes; r++ )
						if ( sets[c].expert[r] < BENCH_LOCAL )
						{
							const uint64_t global = (uint64_t)c * chunk * BENCH_TOP_K + r, packed = route_base[c] + sets[c].packed_row[r];
							memcpy(gate_route.data() + global * BENCH_EXPERT_INTER, gate.data() + packed * BENCH_EXPERT_INTER, BENCH_EXPERT_INTER * 2u);
							memcpy(out_route.data() + global * BENCH_EXPERT_HIDDEN, out.data() + packed * BENCH_EXPERT_HIDDEN, BENCH_EXPERT_HIDDEN * 2u);
						}
				if ( pass == 0 )
				{
					gate_ref[option] = gate_route;
					out_ref[option] = out_route;
				}
				else
				{
					uint64_t differ_up = 0, differ_down = 0;
					for ( uint64_t i = 0; i < gate_route.size(); i++ )
						differ_up += gate_route[i] != gate_ref[option][i];
					for ( uint64_t i = 0; i < out_route.size(); i++ )
						differ_down += out_route[i] != out_ref[option][i];
					for ( uint32_t i = 0; i < 2u; i++ )
						run();
					CHECK(cudaEventRecord(start, s));
					for ( uint32_t i = 0; i < BENCH_ITERATIONS; i++ )
						run();
					CHECK(cudaEventRecord(stop, s));
					const double us = 1000.0 * Elapsed(start, stop) / BENCH_ITERATIONS;
					printf("EXPERTS option=%s rows=%u local_routes=%u launches=%zu us=%.2f up_differ_from_m1024=%llu down_differ_from_m1024=%llu\n", kOption[option], m,
						local, sets.size() * 3u, us, (unsigned long long)differ_up, (unsigned long long)differ_down);
				}
				for ( RouteSet &set : sets )
					FreeRoutes(&set);
			}
	uint64_t cross = 0;
	for ( uint64_t i = 0; i < gate_ref[0].size(); i++ )
		cross += gate_ref[0][i] != gate_ref[1][i];
	printf("EXPERTS m1024 current_vs_skinny_all_up_differ=%llu\n", (unsigned long long)cross);
	cudaFree(b.w1); cudaFree(b.w3); cudaFree(b.w2); cudaFree(b.normed); cudaFree(b.gate); cudaFree(b.up); cudaFree(b.act); cudaFree(b.out);
}

static void ExpertDecode(cudaStream_t s)
{
	const uint64_t up_elements = (uint64_t)BENCH_LOCAL * BENCH_EXPERT_INTER * BENCH_EXPERT_HIDDEN;
	ExpertBuffers b;
	cudaEvent_t start, stop;
	straggler = 1u;
	CHECK(cudaMalloc(&b.w1, up_elements * 2u));
	CHECK(cudaMalloc(&b.w3, up_elements * 2u));
	CHECK(cudaMalloc(&b.w2, up_elements * 2u));
	CHECK(cudaMalloc(&b.normed, 16u * BENCH_EXPERT_HIDDEN * 2u));
	CHECK(cudaMalloc(&b.gate, 128u * BENCH_EXPERT_INTER * 2u));
	CHECK(cudaMalloc(&b.up, 128u * BENCH_EXPERT_INTER * 2u));
	CHECK(cudaMalloc(&b.act, 128u * BENCH_EXPERT_INTER * 2u));
	CHECK(cudaMalloc(&b.out, 128u * BENCH_EXPERT_HIDDEN * 2u));
	Fill(b.w1, up_elements, 21u, 0.03f);
	Fill(b.w3, up_elements, 22u, 0.03f);
	Fill(b.w2, up_elements, 23u, 0.03f);
	Fill(b.normed, 16u * BENCH_EXPERT_HIDDEN, 24u, 1.0f);
	Fill(b.act, 128u * BENCH_EXPERT_INTER, 25u, 1.0f);
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	for ( uint32_t m : {1u, 2u, 4u, 8u, 16u} )
	{
		RouteSet set;
		BuildRoutes(&set, 7777u * m, m, s);
		for ( int option = 0; option < 3; option += 2 )
		{
			for ( uint32_t i = 0; i < 3u; i++ )
				LAUNCH(ExpertLayer(option, &set, b, b.normed, b.gate, b.up, b.act, b.out, s));
			CHECK(cudaEventRecord(start, s));
			for ( uint32_t i = 0; i < BENCH_ITERATIONS; i++ )
				LAUNCH(ExpertLayer(option, &set, b, b.normed, b.gate, b.up, b.act, b.out, s));
			CHECK(cudaEventRecord(stop, s));
			printf("EXPERTS-BUSIEST-RANK option=%s rows=%u local_routes=%u us=%.2f\n", kOption[option], m, set.local, 1000.0 * Elapsed(start, stop) / BENCH_ITERATIONS);
		}
		FreeRoutes(&set);
	}
	straggler = 0u;
	cudaFree(b.w1); cudaFree(b.w3); cudaFree(b.w2); cudaFree(b.normed); cudaFree(b.gate); cudaFree(b.up); cudaFree(b.act); cudaFree(b.out);
}

int main(int argc, char **argv)
{
	cudaStream_t s;
	int value = 0;
	CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, 0));
	sms = (uint32_t)value;
	CHECK(cudaStreamCreate(&s));
	printf("DEVICE sms=%u iterations=%u\n", sms, BENCH_ITERATIONS);
	if ( argc < 2 || strcmp(argv[1], "experts") != 0 )
	{
		DenseCase("h5376_to_2048", 5376u, 2048u, s);
		DenseCase("h1344_to_5376", 1344u, 5376u, s);
		DenseCase("h4096_to_1024", 4096u, 1024u, s);
		DenseCase("h1024_to_4096", 1024u, 4096u, s);
		DenseCase("h5376_to_2688", 5376u, 2688u, s);
		DenseCase("h512_to_5376", 512u, 5376u, s);
	}
	if ( argc < 2 || strcmp(argv[1], "dense") != 0 )
	{
		ExpertCase(s);
		ExpertDecode(s);
	}
	printf("DONE\n");
	return 0;
}
