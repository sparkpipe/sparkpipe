#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/attn_shard_prefill.cuh"

#define REFUSAL_PAGE 64u
#define REFUSAL_LATENT 512u
#define REFUSAL_ROPE 64u
#define REFUSAL_WIDTH (REFUSAL_LATENT + REFUSAL_ROPE)

struct RefusalKv
{
	static constexpr uint32_t kSlotBytes = REFUSAL_WIDTH * 2u;
	static constexpr uint32_t kPageSlots = REFUSAL_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * REFUSAL_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / REFUSAL_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % REFUSAL_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + REFUSAL_PAGE - 1u) / REFUSAL_PAGE; }
};

static uint32_t failures;

template<class Call>
static std::string Captured(Call call, int64_t *result)
{
	char chunk[512];
	size_t count;
	std::string text;
	FILE *sink = tmpfile();
	fflush(stderr);
	const int saved = dup(fileno(stderr));
	dup2(fileno(sink), fileno(stderr));
	*result = (int64_t)call();
	fflush(stderr);
	dup2(saved, fileno(stderr));
	close(saved);
	rewind(sink);
	while ( (count = fread(chunk, 1u, sizeof(chunk), sink)) > 0u )
		text.append(chunk, count);
	fclose(sink);
	return text;
}

template<class Call>
static void Refused(const char *name, Call call, int64_t status, const char *message, const char *site)
{
	int64_t result = 0;
	spark_last_error_site.file = 0;
	const std::string text = Captured(call, &result);
	const char *file = spark_last_error_site.file != 0 ? spark_last_error_site.file : "";
	const size_t file_length = strlen(file), site_length = strlen(site);
	if ( result != status || text.find(message) == std::string::npos || file_length < site_length || strcmp(file + file_length - site_length, site) != 0 ||
		text.find(std::string("ERRSITE ") + file) == std::string::npos )
	{
		printf("FAIL %s: status %lld (wanted %lld), site %s (wanted %s), stderr:\n%s", name, (long long)result, (long long)status, file, site, text.c_str());
		failures++;
		return;
	}
	printf("PASS %s refuses with \"%s\" at %s\n", name, message, site);
}

int main(void)
{
	const void *weight = (const void *)0x10000;
	const uint16_t *activation = (const uint16_t *)0x20000;
	uint16_t *output = (uint16_t *)0x30000;
	LmLocalExpertRoutes routes;
	memset(&routes, 0, sizeof(routes));
	routes.group_row_offset = (const uint32_t *)0x40000;
	routes.route_source_token = (const uint32_t *)0x50000;
	routes.tile_prefix_up = (uint32_t *)0x60000;
	routes.tile_prefix_down = (uint32_t *)0x70000;
	routes.top_k = 8u;
	routes.local_experts = 16u;
	routes.multiprocessors = 48u;
	Refused("a dense small wave whose input is not a multiple of 8", [&]() {
		return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), activation, output, (float *)0, 4u, 100u, 64u, 0u, 48u, (SparkGpuStream)0);
	}, LM_LAUNCH_ERR_SHAPE, "dense rows: the skinny kernel refused 4 rows of a 64 x 100 projection", "inference/kernels/dense_rows.cuh");
	Refused("a row-exact dense wave whose input is not a multiple of 8", [&]() {
		return LmDenseRows<LmBf16Format>(weight, LmScaleTensorNone(), activation, output, (float *)0, 40u, 100u, 64u, 1u, 48u, (SparkGpuStream)0);
	}, LM_LAUNCH_ERR_SHAPE, "dense rows: the skinny kernel refused 40 rows of a 64 x 100 projection", "inference/kernels/dense_rows.cuh");
	routes.rows = 4u;
	routes.routes = 32u;
	Refused("a small expert wave whose input is not a multiple of 8", [&]() {
		return LmLocalExpertsUp<LmBf16Format>(routes, weight, LmScaleTensorNone(), activation, output, 100u, 64u, (SparkGpuStream)0);
	}, LM_LAUNCH_ERR_SHAPE, "local experts: the grouped skinny kernel refused 32 routes over 16 experts of a 64 x 100 projection",
		"inference/kernels/moe_experts.cuh");
	routes.rows = 512u;
	routes.routes = 4096u;
	Refused("a large MXFP4 expert wave whose output is not a multiple of 32", [&]() {
		return LmLocalExpertsUp<LmMxfp4>(routes, weight, LmScaleTensorNone(), activation, output, 4096u, 48u, (SparkGpuStream)0);
	}, LM_LAUNCH_ERR_SHAPE, "local experts: the expert cell kernel refused 4096 routes over 16 experts of a 48 x 4096 projection",
		"inference/kernels/moe_experts.cuh");
	Refused("a large MXFP4 expert down wave whose input is not a multiple of 32", [&]() {
		return LmLocalExpertsDown<LmMxfp4>(routes, weight, LmScaleTensorNone(), activation, output, 2000u, 4096u, (SparkGpuStream)0);
	}, LM_LAUNCH_ERR_SHAPE, "local experts: the expert cell kernel refused 4096 routes over 16 experts of a 4096 x 2000 projection",
		"inference/kernels/moe_experts.cuh");
	{
		LmKvShardView view;
		memset(&view, 0, sizeof(view));
		const uint32_t *rows = (const uint32_t *)0x80000;
		Refused("shard prefill attention over 40 heads", [&]() {
			return LmLatentShardAttendHeadsLaunch<RefusalKv, LmKvShardView, REFUSAL_LATENT, REFUSAL_ROPE>(view, activation, 4096u, 4u, 40u, rows, rows, rows,
				(const uint32_t *)0, 0u, 0u, 0.0625f, (float *)0x90000, 4096u, 4u, 1u, 64u, (SparkGpuStream)0);
		}, SPARK_GPU_ERROR_INVALID_VALUE, "shard prefill attention: 40 heads over all ranks", "inference/kernels/attn_shard_prefill.cuh");
		Refused("shard prefill attention without row positions", [&]() {
			return LmLatentShardAttendHeadsLaunch<RefusalKv, LmKvShardView, REFUSAL_LATENT, REFUSAL_ROPE>(view, activation, 4096u, 4u, 64u, rows, rows,
				(const uint32_t *)0, (const uint32_t *)0, 0u, 0u, 0.0625f, (float *)0x90000, 4096u, 4u, 1u, 64u, (SparkGpuStream)0);
		}, SPARK_GPU_ERROR_INVALID_VALUE, "shard prefill attention: the tensor-core kernel needs each row's position", "inference/kernels/attn_shard_prefill.cuh");
		Refused("shard prefill attention over a selection list past shared memory", [&]() {
			return LmLatentShardAttendHeadsLaunch<RefusalKv, LmKvShardView, REFUSAL_LATENT, REFUSAL_ROPE>(view, activation, 4096u, 4u, 64u, rows, rows, rows,
				rows, 8192u, 0u, 0.0625f, (float *)0x90000, 4096u, 4u, 1u, 64u, (SparkGpuStream)0);
		}, SPARK_GPU_ERROR_INVALID_VALUE, "shard prefill attention: 8192 selected positions need", "inference/kernels/attn_shard_prefill.cuh");
	}
	if ( failures != 0u )
	{
		printf("FAIL launch refusals: %u cases\n", failures);
		return 1;
	}
	printf("PASS launch refusals: every refused shape fails with SPARK_FAIL at its site and a message naming the cause and the fix\n");
	return 0;
}
