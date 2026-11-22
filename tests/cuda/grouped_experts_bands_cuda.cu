#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/weight_codec.cuh"

#define BANDS_REQUIRE(condition) do { if ( !(condition) ) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define BANDS_CUDA(call) BANDS_REQUIRE((call) == cudaSuccess)
#define BANDS_REPEATS 3u
#define BANDS_SLOWER 4.0f
#define BANDS_SLACK_MS 0.3f

enum BandsEntry
{
	BANDS_LOCAL_EXPERTS,
	BANDS_GROUPED
};

typedef struct BandsFamily
{
	const char *name;
	uint32_t experts, local, top_k, hidden, inter;
	BandsEntry entry;
} BandsFamily;

typedef struct BandsRoutes
{
	uint32_t rows, routes, local_routes;
	uint32_t *offsets, *source, *prefix_up, *prefix_down;
} BandsRoutes;

static uint64_t BandsMix(uint64_t value)
{
	value += 0x9e3779b97f4a7c15ull;
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
	value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
	return value ^ (value >> 31);
}

static void *BandsUpload(const std::vector<uint8_t> &host)
{
	void *pointer = 0;
	BANDS_CUDA(cudaMalloc(&pointer, host.size()));
	BANDS_CUDA(cudaMemcpy(pointer, host.data(), host.size(), cudaMemcpyHostToDevice));
	return pointer;
}

static void *BandsRandomBytes(size_t bytes, uint64_t seed)
{
	std::vector<uint8_t> host(bytes);
	for ( size_t i = 0; i < bytes; ++i )
		host[i] = (uint8_t)BandsMix(seed ^ i);
	return BandsUpload(host);
}

static uint16_t *BandsActivation(size_t elements, uint64_t seed)
{
	std::vector<uint8_t> host(elements * 2u);
	for ( size_t i = 0; i < elements; ++i )
	{
		const uint64_t r = BandsMix(seed ^ i);
		const uint16_t value = (uint16_t)((r & 0x8000u) | ((118u + (uint32_t)((r >> 16) % 12u)) << 7) | ((r >> 24) & 0x7fu));
		memcpy(host.data() + i * 2u, &value, 2u);
	}
	return (uint16_t *)BandsUpload(host);
}

template<class Format>
static LmScaleTensor BandsScale(uint32_t groups, uint32_t rows, uint32_t input, void **data)
{
	constexpr uint32_t encoding = LmSkinnyFormat<Format>::kScaleEncoding;
	*data = 0;
	if constexpr ( Format::kScaleGroup == 0u )
		return LmScaleTensorNone();
	else
	{
		const size_t entries = (size_t)groups * rows * (input / Format::kScaleGroup);
		std::vector<uint8_t> host;
		if constexpr ( encoding == LM_SCALE_ENCODING_UE4M3_F32_GLOBAL )
		{
			host.assign(groups * sizeof(float) + entries, 0x38u);
			for ( uint32_t g = 0; g < groups; ++g )
			{
				const float global = 1.0f / 64.0f;
				memcpy(host.data() + g * sizeof(float), &global, sizeof(float));
			}
			*data = BandsUpload(host);
			return LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_NVFP4_E2M1>(*data, groups, rows, input);
		}
		else if constexpr ( encoding == LM_SCALE_ENCODING_F32 )
		{
			host.resize(entries * sizeof(float));
			for ( size_t i = 0; i < entries; ++i )
			{
				const float value = 1.0f / 256.0f;
				memcpy(host.data() + i * sizeof(float), &value, sizeof(float));
			}
		}
		else
			host.assign(entries, 120u);
		*data = BandsUpload(host);
		return LmScaleTensorBuild(*data, encoding, groups, rows, input, 1u, Format::kScaleGroup);
	}
}

static BandsRoutes BandsBuildRoutes(const BandsFamily *family, uint32_t rows, uint64_t seed)
{
	BandsRoutes built;
	const uint32_t routes = rows * family->top_k, buckets = family->local + 1u;
	std::vector<uint32_t> expert(routes), count(buckets, 0u), offsets(buckets + 1u, 0u), cursor, source(routes, 0u);
	for ( uint32_t row = 0; row < rows; ++row )
		for ( uint32_t k = 0; k < family->top_k; ++k )
		{
			uint32_t pick, attempt = 0u;
			bool again;
			do
			{
				pick = (uint32_t)(BandsMix(seed ^ ((uint64_t)row << 24) ^ ((uint64_t)k << 8) ^ attempt++) % family->experts);
				again = false;
				for ( uint32_t j = 0; j < k; ++j )
					again = again || expert[row * family->top_k + j] == pick;
			} while ( again );
			expert[row * family->top_k + k] = pick;
		}
	for ( uint32_t route = 0; route < routes; ++route )
		count[expert[route] < family->local ? expert[route] : family->local]++;
	for ( uint32_t group = 0; group < buckets; ++group )
		offsets[group + 1u] = offsets[group] + count[group];
	cursor.assign(offsets.begin(), offsets.end() - 1);
	for ( uint32_t route = 0; route < routes; ++route )
		source[cursor[expert[route] < family->local ? expert[route] : family->local]++] = route / family->top_k;
	const uint32_t tile_m = LmLaunchGroupedTileM(rows, family->top_k, buckets);
	const uint32_t tiles_up = (family->inter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N, tiles_down = (family->hidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
	std::vector<uint32_t> prefix_up(buckets + 1u, 0u), prefix_down(buckets + 1u, 0u);
	for ( uint32_t group = 0; group < buckets; ++group )
	{
		const uint32_t tiles = (count[group] + tile_m - 1u) / tile_m;
		prefix_up[group + 1u] = prefix_up[group] + tiles * tiles_up;
		prefix_down[group + 1u] = prefix_down[group] + tiles * tiles_down;
	}
	std::vector<uint8_t> bytes;
	built.rows = rows;
	built.routes = routes;
	built.local_routes = offsets[family->local];
	bytes.resize(offsets.size() * 4u);
	memcpy(bytes.data(), offsets.data(), bytes.size());
	built.offsets = (uint32_t *)BandsUpload(bytes);
	bytes.resize(source.size() * 4u);
	memcpy(bytes.data(), source.data(), bytes.size());
	built.source = (uint32_t *)BandsUpload(bytes);
	bytes.resize(prefix_up.size() * 4u);
	memcpy(bytes.data(), prefix_up.data(), bytes.size());
	built.prefix_up = (uint32_t *)BandsUpload(bytes);
	memcpy(bytes.data(), prefix_down.data(), bytes.size());
	built.prefix_down = (uint32_t *)BandsUpload(bytes);
	return built;
}

static void BandsFreeRoutes(BandsRoutes *routes)
{
	BANDS_CUDA(cudaFree(routes->offsets));
	BANDS_CUDA(cudaFree(routes->source));
	BANDS_CUDA(cudaFree(routes->prefix_up));
	BANDS_CUDA(cudaFree(routes->prefix_down));
}

template<class Format>
static int32_t BandsProduction(const BandsFamily *family, const BandsRoutes *routes, uint32_t down, const void *weight, LmScaleTensor scale,
	const uint16_t *activation, uint16_t *output, int sms, cudaStream_t stream)
{
	const uint32_t input = down != 0u ? family->inter : family->hidden, output_dimension = down != 0u ? family->hidden : family->inter;
	if ( family->entry == BANDS_GROUPED )
		return LmSkinnyGroupedExperts<Format>(weight, scale, activation, output, routes->offsets, routes->source, family->local, routes->routes, down, input,
			output_dimension, stream, 1u);
	LmLocalExpertRoutes local;
	memset(&local, 0, sizeof(local));
	local.group_row_offset = routes->offsets;
	local.route_source_token = routes->source;
	local.tile_prefix_up = routes->prefix_up;
	local.tile_prefix_down = routes->prefix_down;
	local.rows = routes->rows;
	local.routes = routes->routes;
	local.top_k = family->top_k;
	local.local_experts = family->local;
	local.multiprocessors = (uint32_t)sms;
	return down != 0u ? LmLocalExpertsDown<Format>(local, weight, scale, activation, output, input, output_dimension, stream) :
		LmLocalExpertsUp<Format>(local, weight, scale, activation, output, input, output_dimension, stream);
}

template<class Format>
static int32_t BandsReference(const BandsFamily *family, const BandsRoutes *routes, uint32_t down, const void *weight, LmScaleTensor scale,
	const uint16_t *activation, uint16_t *output, cudaStream_t stream)
{
	const uint32_t input = down != 0u ? family->inter : family->hidden, output_dimension = down != 0u ? family->hidden : family->inter;
	LmSkinnyArguments args;
	memset(&args, 0, sizeof(args));
	args.weight = (const uint8_t *)weight;
	args.activation = activation;
	args.output_bf16 = output;
	args.group_row_offset = routes->offsets;
	args.route_source_token = routes->source;
	args.scale = scale;
	args.weight_group_bytes = (uint64_t)output_dimension * input * Format::kStoredBits / 8u;
	args.rows = 1u;
	args.pairs = routes->routes;
	args.activation_packed = down;
	args.groups = family->local;
	args.input_dimension = input;
	args.output_dimension = output_dimension;
	args.output_row_stride = output_dimension;
	return LmSkinnyGroupedLanes<Format,2u,4u>(&args, input / LmSkinnyFormat<Format>::kElements, stream);
}

template<class Format>
static float BandsTime(const BandsFamily *family, const BandsRoutes *routes, uint32_t down, const void *weight, LmScaleTensor scale,
	const uint16_t *activation, uint16_t *output, size_t output_bytes, int sms, cudaStream_t stream, uint32_t reference)
{
	cudaEvent_t start, stop;
	float best = INFINITY;
	BANDS_CUDA(cudaEventCreate(&start));
	BANDS_CUDA(cudaEventCreate(&stop));
	for ( uint32_t repeat = 0; repeat < BANDS_REPEATS; ++repeat )
	{
		float ms = 0.0f;
		BANDS_CUDA(cudaMemsetAsync(output, 0, output_bytes, stream));
		BANDS_CUDA(cudaEventRecord(start, stream));
		BANDS_REQUIRE((reference != 0u ? BandsReference<Format>(family, routes, down, weight, scale, activation, output, stream) :
			BandsProduction<Format>(family, routes, down, weight, scale, activation, output, sms, stream)) == LM_LAUNCH_OK);
		BANDS_CUDA(cudaEventRecord(stop, stream));
		BANDS_CUDA(cudaEventSynchronize(stop));
		BANDS_CUDA(cudaEventElapsedTime(&ms, start, stop));
		best = ms < best ? ms : best;
	}
	BANDS_CUDA(cudaEventDestroy(start));
	BANDS_CUDA(cudaEventDestroy(stop));
	return best;
}

template<class Format>
static uint32_t BandsFamilyRun(const BandsFamily *family, int sms, cudaStream_t stream)
{
	const uint32_t bounds[3] = {2u * family->local, LM_SKINNY_ROWS * family->local, LM_SKINNY_GROUPED_MAX_MEAN_ROWS * family->local};
	uint32_t checks = 0u;
	for ( uint32_t down = 0u; down < 2u; ++down )
	{
		const uint32_t input = down != 0u ? family->inter : family->hidden, output_dimension = down != 0u ? family->hidden : family->inter;
		const uint32_t most_rows = bounds[2] / family->top_k + 1u, most_routes = most_rows * family->top_k;
		void *scale_data = 0;
		void *weight = BandsRandomBytes((size_t)family->local * output_dimension * input * Format::kStoredBits / 8u, 0x31u + down);
		const LmScaleTensor scale = BandsScale<Format>(family->local, output_dimension, input, &scale_data);
		uint16_t *activation = BandsActivation((size_t)(down != 0u ? most_routes : most_rows) * input, 0x32u + down);
		uint16_t *output = 0, *reference = 0;
		const size_t output_bytes = (size_t)most_routes * output_dimension * sizeof(uint16_t);
		BANDS_CUDA(cudaMalloc((void **)&output, output_bytes));
		BANDS_CUDA(cudaMalloc((void **)&reference, output_bytes));
		std::vector<uint16_t> left(output_bytes / 2u), right(output_bytes / 2u);
		for ( uint32_t bound = 0; bound < 3u; ++bound )
		{
			const uint32_t rows_below = bounds[bound] / family->top_k, rows_above = rows_below + 1u;
			float below_ms = 0.0f;
			for ( uint32_t side = 0; side < 2u; ++side )
			{
				const uint32_t rows = side == 0u ? rows_below : rows_above;
				BandsRoutes routes = BandsBuildRoutes(family, rows, 0x40u + rows);
				const size_t bytes = (size_t)routes.routes * output_dimension * sizeof(uint16_t);
				const float ms = BandsTime<Format>(family, &routes, down, weight, scale, activation, output, output_bytes, sms, stream, 0u);
				const bool grouped = family->entry == BANDS_GROUPED || routes.routes <= LM_SKINNY_GROUPED_MAX_MEAN_ROWS * family->local;
				if ( grouped )
				{
					(void)BandsTime<Format>(family, &routes, down, weight, scale, activation, reference, output_bytes, sms, stream, 1u);
					BANDS_CUDA(cudaMemcpy(left.data(), output, bytes, cudaMemcpyDeviceToHost));
					BANDS_CUDA(cudaMemcpy(right.data(), reference, bytes, cudaMemcpyDeviceToHost));
					if ( memcmp(left.data(), right.data(), bytes) != 0 )
					{
						printf("FAIL %s %s: %u routes over %u experts give other bytes than the reference grouped variant\n", family->name, down != 0u ? "down" : "up",
							routes.routes, family->local);
						exit(1);
					}
					++checks;
				}
				printf("BAND %s %s routes=%u local_routes=%u bound=%u %s ms=%.3f\n", family->name, down != 0u ? "down" : "up", routes.routes, routes.local_routes,
					bounds[bound], grouped ? "grouped" : "wide-path", ms);
				if ( side == 0u )
					below_ms = ms;
				else if ( ms > BANDS_SLOWER * below_ms + BANDS_SLACK_MS )
				{
					printf("FAIL %s %s: %u routes take %.3f ms against %.3f ms for %u routes, across the %u-route variant boundary\n", family->name,
						down != 0u ? "down" : "up", routes.routes, ms, below_ms, rows_below * family->top_k, bounds[bound]);
					exit(1);
				}
				BandsFreeRoutes(&routes);
				++checks;
			}
		}
		BANDS_CUDA(cudaFree(weight));
		if ( scale_data != 0 )
			BANDS_CUDA(cudaFree(scale_data));
		BANDS_CUDA(cudaFree(activation));
		BANDS_CUDA(cudaFree(output));
		BANDS_CUDA(cudaFree(reference));
	}
	return checks;
}

int main(void)
{
	static const BandsFamily nvfp4[] = {
		{"qwen38-max", 512u, 32u, 10u, 8192u, 2048u, BANDS_LOCAL_EXPERTS},
		{"qwen38-flash", 512u, 32u, 10u, 2560u, 640u, BANDS_LOCAL_EXPERTS},
	};
	static const BandsFamily mxfp4[] = {
		{"mimo26-flash", 256u, 16u, 8u, 4096u, 2048u, BANDS_LOCAL_EXPERTS},
		{"mimo26-pro", 384u, 24u, 8u, 6144u, 2048u, BANDS_LOCAL_EXPERTS},
	};
	static const BandsFamily mxfp8[] = {
		{"hy4", 256u, 16u, 8u, 6144u, 2048u, BANDS_LOCAL_EXPERTS},
	};
	static const BandsFamily fp8[] = {
		{"glm53-full", 256u, 256u, 8u, 6144u, 256u, BANDS_GROUPED},
		{"glm53-flash", 288u, 288u, 8u, 4096u, 256u, BANDS_GROUPED},
	};
	static const BandsFamily bf16[] = {
		{"qwen38-flash-bf16", 512u, 32u, 10u, 2560u, 640u, BANDS_LOCAL_EXPERTS},
		{"gemma4-26b", 128u, 8u, 8u, 2816u, 704u, BANDS_LOCAL_EXPERTS},
		{"kexaone2", 256u, 256u, 8u, 6144u, 128u, BANDS_LOCAL_EXPERTS},
		{"laguna", 256u, 16u, 10u, 3072u, 1024u, BANDS_LOCAL_EXPERTS},
		{"ling3", 512u, 32u, 8u, 2560u, 768u, BANDS_LOCAL_EXPERTS},
		{"solar-open2", 320u, 320u, 8u, 4096u, 80u, BANDS_LOCAL_EXPERTS},
		{"motif", 384u, 24u, 8u, 4096u, 1280u, BANDS_LOCAL_EXPERTS},
		{"glm53-full-bf16", 256u, 256u, 8u, 6144u, 256u, BANDS_GROUPED},
	};
	cudaStream_t stream = 0;
	int sms = 0;
	uint32_t checks = 0u;
	BANDS_CUDA(cudaSetDevice(0));
	BANDS_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
	BANDS_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	for ( const BandsFamily &family : nvfp4 )
		checks += BandsFamilyRun<LmNvfp4>(&family, sms, stream);
	for ( const BandsFamily &family : mxfp4 )
		checks += BandsFamilyRun<LmMxfp4>(&family, sms, stream);
	for ( const BandsFamily &family : mxfp8 )
		checks += BandsFamilyRun<LmMxfp8>(&family, sms, stream);
	for ( const BandsFamily &family : fp8 )
		checks += BandsFamilyRun<LmFp8>(&family, sms, stream);
	for ( const BandsFamily &family : bf16 )
		checks += BandsFamilyRun<LmBf16Format>(&family, sms, stream);
	printf("PASS grouped expert bands: %u checks\n", checks);
	return 0;
}
