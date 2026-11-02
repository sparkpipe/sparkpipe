#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/route.cuh"
#include "inference/kernels/topk_group.cuh"
#include "inference/kernels/topk_warp.cuh"

#define PROBE_K 8u
#define PROBE_PATTERNS 8u
#define PROBE_WIDTH 1024u
#define PROBE_CUDA(call) do { cudaError_t probe_error = (call); if ( probe_error != cudaSuccess ) { printf("FAIL cuda %s: %s\n", #call, cudaGetErrorString(probe_error)); exit(1); } } while ( 0 )

enum ProbeKernel { PROBE_WARP, PROBE_WARPS, PROBE_SMALL_GROUPED, PROBE_LIMITED_GROUPED, PROBE_LIMITED, PROBE_KERNELS };

static const char *kNames[PROBE_KERNELS] = { "route launch, one warp (288 experts, sigmoid, bias)", "route launch, eight warps (768 experts, identity)",
	"small kernel, 8 groups keep 4 (256 experts, sigmoid)", "group-limited, 8 groups keep 4 (256 experts, sigmoid, bias)",
	"group-limited, one group (128 experts, sigmoid, bias)" };
static const uint32_t kExperts[PROBE_KERNELS] = { 288u, 768u, 256u, 256u, 128u };
static const uint32_t kGrouped[PROBE_KERNELS] = { 0u, 0u, 1u, 1u, 0u };
static const uint32_t kSigmoid[PROBE_KERNELS] = { 1u, 0u, 1u, 1u, 1u };
static const uint32_t kBias[PROBE_KERNELS] = { 1u, 0u, 0u, 1u, 1u };

static uint32_t Mix(uint32_t z)
{
	z ^= z >> 16u;
	z *= 0x7feb352du;
	z ^= z >> 15u;
	z *= 0x846ca68bu;
	return z ^ (z >> 16u);
}

static float Bits(uint32_t bits)
{
	float value;
	memcpy(&value, &bits, 4u);
	return value;
}

static float Logit(uint32_t pattern, uint32_t expert, uint32_t n)
{
	const float finite = ((float)(Mix(expert * 2654435761u + 17u) % (8u * n)) - 4.0f * (float)n) / (float)n;
	const float nan = Bits(0x7fc00000u), negative_nan = Bits(0xffc00001u);
	switch ( pattern )
	{
	case 0u: return finite;
	case 1u: return Mix(expert + 99u) % 10u == 0u ? nan : finite;
	case 2u: return nan;
	case 3u: return expert < PROBE_K - 1u ? finite : nan;
	case 4u: return expert % 7u == 0u ? INFINITY : expert % 7u == 1u ? -INFINITY : expert % 7u == 2u ? nan : finite;
	case 5u: return Mix(expert + 7u) % 12u == 0u ? finite : negative_nan;
	case 6u: return -INFINITY;
	default: return expert >= n - PROBE_K ? finite : nan;
	}
}

static float Bias(uint32_t kernel, uint32_t expert)
{
	return kBias[kernel] != 0u ? 0.001f * (float)(expert % 5u) : 0.0f;
}

static float Selection(uint32_t kernel, float logit, uint32_t expert)
{
	const float score = kSigmoid[kernel] != 0u ? 1.0f / (1.0f + expf(-logit)) : logit;
	return score + Bias(kernel, expert);
}

static void Launch(uint32_t kernel, const float *logits, const float *bias, uint32_t *indices, float *values, uint32_t *fault)
{
	switch ( kernel )
	{
	case PROBE_WARP:
		PROBE_CUDA((LmTopkRouteLaunch<256u, PROBE_K, true, LM_TOPK_SCORE_SIGMOID>(1u, logits, 288u, indices, values, bias, 0, 2.5f, fault, 0)));
		break;
	case PROBE_WARPS:
		PROBE_CUDA((LmTopkRouteLaunch<256u, PROBE_K, true, LM_TOPK_SCORE_IDENTITY>(1u, logits, 768u, indices, values, 0, 0, 1.0f, fault, 0)));
		break;
	case PROBE_SMALL_GROUPED:
		LmTopkSmallKernel<256u, PROBE_K, true, 8u, 4u, LM_TOPK_SCORE_SIGMOID><<<1u, 256u, 2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t)>>>(logits, 256u,
			indices, values, 0, 0, 2.5f, fault);
		PROBE_CUDA(cudaGetLastError());
		break;
	case PROBE_LIMITED_GROUPED:
		PROBE_CUDA((LmTopkGroupLimitedLaunch<256u, 256u, 8u, 4u, PROBE_K>(1u, logits, bias, indices, values, 2.5f, fault, 0)));
		break;
	default:
		PROBE_CUDA((LmTopkGroupLimitedLaunch<256u, 128u, 1u, 1u, PROBE_K>(1u, logits, bias, indices, values, 1.0f, fault, 0)));
		break;
	}
	PROBE_CUDA(cudaDeviceSynchronize());
}

static uint32_t Fail(const char *kernel, uint32_t pattern, const char *what)
{
	printf("FAIL route nan: %s, row pattern %u: %s\n", kernel, pattern, what);
	return 1u;
}

int main(void)
{
	float *logits, *bias, *values;
	uint32_t *indices, *fault, *packed_row, *source_token, *group_offset, *tiles_up, *tiles_down;
	uint32_t failures = 0u;
	std::vector<uint32_t> warp_routes;
	PROBE_CUDA(cudaMalloc((void **)&logits, PROBE_WIDTH * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&bias, PROBE_WIDTH * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&values, PROBE_K * sizeof(float)));
	PROBE_CUDA(cudaMalloc((void **)&indices, PROBE_K * sizeof(uint32_t)));
	PROBE_CUDA(cudaMalloc((void **)&fault, sizeof(uint32_t)));
	for ( uint32_t kernel = 0u; kernel < PROBE_KERNELS; ++kernel )
	{
		const uint32_t n = kExperts[kernel];
		std::vector<float> bias_host(PROBE_WIDTH, 0.0f);
		for ( uint32_t expert = 0u; expert < n; ++expert )
			bias_host[expert] = Bias(kernel, expert);
		PROBE_CUDA(cudaMemcpy(bias, bias_host.data(), PROBE_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
		for ( uint32_t pattern = 0u; pattern < PROBE_PATTERNS; ++pattern )
		{
			std::vector<float> row(PROBE_WIDTH, 0.0f);
			std::vector<uint32_t> finite, nonfinite, got(PROBE_K);
			std::vector<float> selection(n);
			uint32_t raised = 0u, seen = 0u;
			for ( uint32_t expert = 0u; expert < n; ++expert )
			{
				row[expert] = Logit(pattern, expert, n);
				selection[expert] = Selection(kernel, row[expert], expert);
				(isfinite(selection[expert]) ? finite : nonfinite).push_back(expert);
			}
			PROBE_CUDA(cudaMemcpy(logits, row.data(), PROBE_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
			PROBE_CUDA(cudaMemset(indices, 0xee, PROBE_K * sizeof(uint32_t)));
			PROBE_CUDA(cudaMemset(fault, 0, sizeof(uint32_t)));
			Launch(kernel, logits, bias, indices, values, fault);
			PROBE_CUDA(cudaMemcpy(got.data(), indices, PROBE_K * sizeof(uint32_t), cudaMemcpyDeviceToHost));
			PROBE_CUDA(cudaMemcpy(&raised, fault, sizeof(uint32_t), cudaMemcpyDeviceToHost));
			for ( uint32_t pick = 0u; pick < PROBE_K; ++pick )
			{
				if ( got[pick] >= n )
				{
					failures += Fail(kNames[kernel], pattern, "an expert index is out of range");
					break;
				}
				for ( uint32_t other = 0u; other < pick; ++other )
					seen |= got[other] == got[pick] ? 1u : 0u;
			}
			if ( seen != 0u )
				failures += Fail(kNames[kernel], pattern, "an expert is chosen twice");
			if ( finite.size() < PROBE_K && raised == 0u )
				failures += Fail(kNames[kernel], pattern, "fewer finite scores than picks and the fault word stays clear");
			if ( finite.size() == n && raised != 0u )
				failures += Fail(kNames[kernel], pattern, "an all-finite row raises the fault");
			if ( kGrouped[kernel] == 0u )
			{
				std::vector<uint32_t> want(finite);
				std::stable_sort(want.begin(), want.end(), [&](uint32_t a, uint32_t b) { return selection[a] > selection[b]; });
				want.insert(want.end(), nonfinite.begin(), nonfinite.end());
				want.resize(PROBE_K);
				if ( (finite.size() < PROBE_K) != (raised != 0u) )
					failures += Fail(kNames[kernel], pattern, "the fault word does not match whether the row holds k finite scores");
				for ( uint32_t pick = 0u; pick < PROBE_K && got[0] < n; ++pick )
					if ( std::find(got.begin(), got.end(), want[pick]) == got.end() )
					{
						failures += Fail(kNames[kernel], pattern, "the picks are not the best finite scores, then the lowest non-finite indices");
						break;
					}
			}
			if ( kernel == PROBE_WARP )
				warp_routes.insert(warp_routes.end(), got.begin(), got.end());
			printf("%s, pattern %u: %zu finite of %u, fault %u\n", kNames[kernel], pattern, finite.size(), n, raised);
		}
	}
	{
		const uint32_t rows = PROBE_PATTERNS, packed = rows * PROBE_K;
		std::vector<uint32_t> offsets(289u), rows_of(packed), sources(packed);
		uint32_t *routes;
		PROBE_CUDA(cudaMalloc((void **)&routes, packed * sizeof(uint32_t)));
		PROBE_CUDA(cudaMalloc((void **)&packed_row, packed * sizeof(uint32_t)));
		PROBE_CUDA(cudaMalloc((void **)&source_token, packed * sizeof(uint32_t)));
		PROBE_CUDA(cudaMalloc((void **)&group_offset, 289u * sizeof(uint32_t)));
		PROBE_CUDA(cudaMalloc((void **)&tiles_up, 289u * sizeof(uint32_t)));
		PROBE_CUDA(cudaMalloc((void **)&tiles_down, 289u * sizeof(uint32_t)));
		PROBE_CUDA(cudaMemcpy(routes, warp_routes.data(), packed * sizeof(uint32_t), cudaMemcpyHostToDevice));
		if ( LmRouteBuild<256u, 288u>(routes, rows, packed, PROBE_K, group_offset, packed_row, source_token, 256u, 4096u, 128u, tiles_up, tiles_down, 0) != LM_LAUNCH_OK )
			failures += Fail("route build", 0u, "the build refuses the guarded routes");
		PROBE_CUDA(cudaDeviceSynchronize());
		PROBE_CUDA(cudaMemcpy(offsets.data(), group_offset, 289u * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		PROBE_CUDA(cudaMemcpy(rows_of.data(), packed_row, packed * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		PROBE_CUDA(cudaMemcpy(sources.data(), source_token, packed * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		std::vector<uint32_t> sorted(rows_of);
		std::sort(sorted.begin(), sorted.end());
		for ( uint32_t route = 0u; route < packed; ++route )
			if ( sorted[route] != route || sources[rows_of[route]] != route / PROBE_K )
			{
				failures += Fail("route build", route, "a packed row is out of range, repeated, or maps to the wrong token");
				break;
			}
		if ( offsets[288] != packed )
			failures += Fail("route build", 0u, "the expert offsets do not cover every route");
		printf("route build over the eight guarded rows: offsets end at %u of %u routes\n", offsets[288], packed);
	}
	if ( failures != 0u )
	{
		printf("FAIL route nan: %u failures\n", failures);
		return 1;
	}
	printf("PASS route nan: NaN and infinite router rows never yield an out-of-range or repeated expert, rank below every finite score, raise the "
		"fault word when fewer than k scores are finite, and build in-range packed routes\n");
	return 0;
}
