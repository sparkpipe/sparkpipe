#include "inference/kernels/skinny.cuh"
#include "inference/kernels/weight_codec.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <vector>

#define VB_POOL_BYTES (640ull << 20)
#define VB_MAX_COPIES 256u
#define VB_ROUTINGS 64u
#define VB_LAUNCHES 96u
#define VB_TRIALS 7u
#define VB_MAX_ROWS 16u
#define VB_DRAM_GBPS 240.0

#define VB_CHECK(call) do { cudaError_t vb_error = (call); if ( vb_error != cudaSuccess ) { fprintf(stderr, "FAIL %s: %s\n", #call, cudaGetErrorString(vb_error)); exit(1); } } while ( 0 )

static const uint32_t kRows[] = { 1u, 2u, 3u, 4u, 5u, 8u, 12u, 16u };
static uint32_t gOnly = 0u;
static const char *gFilter = 0;

static uint64_t VbMix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

__global__ static void VbFillKernel(uint8_t *bytes, uint64_t count, uint32_t kind, uint32_t salt)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		uint32_t value = (uint32_t)index * 0x9e3779b9u + salt;
		value ^= value >> 16u;
		value *= 0x7feb352du;
		value ^= value >> 15u;
		bytes[index] = kind == 0u ? (uint8_t)((value & 0x80u) | ((1u + (value >> 8u) % 8u) << 3u) | ((value >> 16u) & 7u)) :
			kind == 1u ? (uint8_t)(118u + (value >> 8u) % 5u) : kind == 3u ? (uint8_t)((index & 3u) == 3u ? 0x3cu : (index & 3u) == 2u ? (value & 0x7fu) : 0u) :
			(uint8_t)((index & 1u) != 0u ? (0x3cu | (value & 0x80u)) : value);
	}
}

static uint8_t *VbBytes(uint64_t count, uint32_t kind, uint32_t salt = 0u)
{
	uint8_t *device = 0;
	VB_CHECK(cudaMalloc((void **)&device, count + 64u));
	VbFillKernel<<<1024u, 256u>>>(device, count, kind, salt);
	VB_CHECK(cudaGetLastError());
	return device;
}

template<class T>
static T *VbDevice(const std::vector<T> &host)
{
	T *device = 0;
	VB_CHECK(cudaMalloc((void **)&device, host.size() * sizeof(T) + 16u));
	VB_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
	return device;
}

static cudaStream_t gStream;

static float VbTime(const std::function<int32_t(uint32_t)> &launch, uint32_t launches)
{
	cudaGraph_t graph;
	cudaGraphExec_t exec;
	cudaEvent_t start, stop;
	float best = 1e30f, elapsed = 0.0f;
	for ( uint32_t index = 0u; index < 4u; ++index )
		if ( launch(index) != LM_LAUNCH_OK )
		{
			fprintf(stderr, "FAIL launch refused (%s)\n", cudaGetErrorString(cudaGetLastError()));
			exit(1);
		}
	VB_CHECK(cudaStreamSynchronize(gStream));
	VB_CHECK(cudaStreamBeginCapture(gStream, cudaStreamCaptureModeThreadLocal));
	for ( uint32_t index = 0u; index < launches; ++index )
		if ( launch(index) != LM_LAUNCH_OK )
		{
			fprintf(stderr, "FAIL launch refused in capture\n");
			exit(1);
		}
	VB_CHECK(cudaStreamEndCapture(gStream, &graph));
	VB_CHECK(cudaGraphInstantiate(&exec, graph, 0));
	VB_CHECK(cudaEventCreate(&start));
	VB_CHECK(cudaEventCreate(&stop));
	VB_CHECK(cudaGraphLaunch(exec, gStream));
	for ( uint32_t trial = 0u; trial < VB_TRIALS; ++trial )
	{
		VB_CHECK(cudaEventRecord(start, gStream));
		VB_CHECK(cudaGraphLaunch(exec, gStream));
		VB_CHECK(cudaEventRecord(stop, gStream));
		VB_CHECK(cudaEventSynchronize(stop));
		VB_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
		best = elapsed < best ? elapsed : best;
	}
	VB_CHECK(cudaGraphExecDestroy(exec));
	VB_CHECK(cudaGraphDestroy(graph));
	VB_CHECK(cudaEventDestroy(start));
	VB_CHECK(cudaEventDestroy(stop));
	return 1000.0f * best / (float)launches;
}

static void VbHeader(const char *model, const char *name, const char *format, const char *entry, uint64_t bytes_one)
{
	printf("\n%s %s (%s, %s, %.2f MB per row-1 launch)\n", model, name, format, entry, (double)bytes_one / 1e6);
	printf("%5s %10s %10s %9s %11s %12s %12s\n", "rows", "us", "MB", "GB/s", "us/row+", "floor_us/r+", "x_one_row");
}

static void VbLine(uint32_t rows, float us, uint64_t bytes, float us_one, uint64_t bytes_one)
{
	const double floor = rows > 1u ? ((double)bytes - (double)bytes_one) / (VB_DRAM_GBPS * 1e3) / (rows - 1u) : 0.0;
	printf("%5u %10.2f %10.2f %9.1f %11.2f %12.2f %12.2f\n", rows, us, (double)bytes / 1e6, (double)bytes / (us * 1e3),
		rows > 1u ? (us - us_one) / (rows - 1u) : 0.0f, floor, us / us_one);
	fflush(stdout);
}

static bool VbSkip(const char *model, const char *name)
{
	char key[256];
	snprintf(key, sizeof(key), "%s %s", model, name);
	return gFilter != 0 && strstr(key, gFilter) == 0;
}

template<class Format>
static LmScaleTensor VbScale(const uint8_t *data, uint32_t groups, uint32_t rows, uint32_t input)
{
	if constexpr ( std::is_same<Format, LmMxfp8>::value )
		return LmScaleTensorBlockUe8m0(data, groups, rows, input, 1u, LM_MXFP8_GROUP);
	else if constexpr ( std::is_same<Format, LmFp8>::value )
		return LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(data, groups, rows, input);
	else
		return LmScaleTensorNone();
}

template<class Format>
static uint64_t VbScaleBytes(uint32_t groups, uint32_t rows, uint32_t input)
{
	if constexpr ( std::is_same<Format, LmMxfp8>::value )
		return (uint64_t)groups * rows * (input / LM_MXFP8_GROUP);
	else if constexpr ( std::is_same<Format, LmFp8>::value )
		return (uint64_t)groups * rows * ((input + 127u) / 128u) * sizeof(float);
	else
		return 0u;
}

template<class Format>
static uint32_t VbWeightKind(void)
{
	return std::is_same<Format, LmBf16Format>::value ? 2u : 0u;
}

template<class Format>
static const char *VbFormatName(void)
{
	return std::is_same<Format, LmBf16Format>::value ? "bf16" : std::is_same<Format, LmMxfp8>::value ? "mxfp8" : "fp8";
}

struct VbDenseCopy
{
	uint8_t *weight[2];
	uint8_t *scale[2];
};

enum VbEntry
{
	VB_ENTRY_LAUNCH = 0,
	VB_ENTRY_ROWS = 1
};

template<class Format>
static void VbDense(const char *model, const char *name, VbEntry entry, uint32_t input, uint32_t first_output, uint32_t second_output, uint32_t f32_output)
{
	if ( VbSkip(model, name) )
		return;
	const uint32_t parts = second_output != 0u ? 2u : 1u, outputs[2] = { first_output, second_output };
	uint64_t bytes = 0u;
	for ( uint32_t part = 0u; part < parts; ++part )
		bytes += (uint64_t)outputs[part] * input * Format::kStoredBits / 8u + VbScaleBytes<Format>(1u, outputs[part], input);
	const uint32_t copies = (uint32_t)std::min<uint64_t>(VB_MAX_COPIES, std::max<uint64_t>(2u, VB_POOL_BYTES / bytes));
	std::vector<VbDenseCopy> pool(copies);
	for ( uint32_t copy = 0u; copy < copies; ++copy )
		for ( uint32_t part = 0u; part < parts; ++part )
		{
			pool[copy].weight[part] = VbBytes((uint64_t)outputs[part] * input * Format::kStoredBits / 8u, VbWeightKind<Format>(), copy * 7u + part);
			const uint64_t scale_bytes = VbScaleBytes<Format>(1u, outputs[part], input);
			pool[copy].scale[part] = scale_bytes != 0u ? VbBytes(scale_bytes, std::is_same<Format, LmFp8>::value ? 3u : 1u, copy) : 0;
		}
	uint16_t *activation = (uint16_t *)VbBytes((uint64_t)VB_MAX_ROWS * input * 2u, 2u, 99u);
	uint16_t *output = 0;
	VB_CHECK(cudaMalloc((void **)&output, (uint64_t)VB_MAX_ROWS * (first_output + second_output) * 4u + 64u));
	VB_CHECK(cudaDeviceSynchronize());
	char entry_name[64];
	snprintf(entry_name, sizeof(entry_name), "%s%s", parts == 2u ? "LmSkinnyDenseMulti x2" : entry == VB_ENTRY_LAUNCH ? "LmSkinnyLaunch" : "LmSkinnyLaunchRows",
		f32_output != 0u ? ", f32 out" : "");
	VbHeader(model, name, VbFormatName<Format>(), entry_name, bytes);
	float us_one = 0.0f;
	for ( uint32_t rows : kRows )
	{
		if ( gOnly != 0u && rows != 1u && rows != gOnly )
			continue;
		auto launch = [&](uint32_t index) -> int32_t
		{
			const VbDenseCopy &copy = pool[index % copies];
			if ( parts == 2u )
			{
				LmSkinnyDenseTarget targets[2];
				memset(targets, 0, sizeof(targets));
				for ( uint32_t part = 0u; part < 2u; ++part )
				{
					targets[part].weight = copy.weight[part];
					targets[part].output_bf16 = output + (part == 0u ? 0u : (uint64_t)VB_MAX_ROWS * first_output);
					targets[part].output_dimension = outputs[part];
					targets[part].scale = VbScale<Format>(copy.scale[part], 1u, outputs[part], input);
				}
				return LmSkinnyDenseMulti<Format>(targets, 2u, activation, rows, input, gStream, 1u);
			}
			LmSkinnyArguments args;
			memset(&args, 0, sizeof(args));
			args.weight = copy.weight[0];
			args.activation = activation;
			args.output_bf16 = f32_output != 0u ? 0 : output;
			args.output_f32 = f32_output != 0u ? (float *)output : 0;
			args.scale = VbScale<Format>(copy.scale[0], 1u, first_output, input);
			args.rows = rows;
			args.input_dimension = input;
			args.output_dimension = first_output;
			args.output_row_stride = first_output;
			if ( entry == VB_ENTRY_LAUNCH )
				return LmSkinnyLaunch<Format>(&args, gStream, 1u);
			const int32_t status = LmSkinnyValidate<Format>(&args, 1u);
			return status != LM_LAUNCH_OK ? status : LmSkinnyLaunchRows<Format>(&args, input / LmSkinnyFormat<Format>::kElements, gStream);
		};
		const float us = VbTime(launch, std::max<uint32_t>(VB_LAUNCHES, copies));
		us_one = rows == 1u ? us : us_one;
		VbLine(rows, us, bytes, us_one, bytes);
	}
	for ( VbDenseCopy &copy : pool )
		for ( uint32_t part = 0u; part < parts; ++part )
		{
			cudaFree(copy.weight[part]);
			cudaFree(copy.scale[part]);
		}
	cudaFree(activation);
	cudaFree(output);
}

struct VbRouting
{
	uint32_t *offsets, *source, *expert, *packed_row;
	uint32_t distinct, routes;
};

static std::vector<VbRouting> VbRoutings(uint32_t rows, uint32_t groups, uint32_t top_k, uint32_t salt)
{
	std::vector<VbRouting> routings;
	for ( uint32_t variant = 0u; variant < VB_ROUTINGS; ++variant )
	{
		std::vector<uint32_t> experts, offsets(groups + 1u, 0u), source, packed_row;
		for ( uint32_t row = 0u; row < rows; ++row )
		{
			uint64_t draw = VbMix(((uint64_t)salt << 48u) ^ ((uint64_t)variant << 32u) ^ ((uint64_t)rows << 16u) ^ row);
			std::vector<uint32_t> chosen;
			while ( chosen.size() < top_k )
			{
				const uint32_t expert = (uint32_t)(draw % groups);
				draw = VbMix(draw);
				if ( std::find(chosen.begin(), chosen.end(), expert) == chosen.end() )
					chosen.push_back(expert);
			}
			experts.insert(experts.end(), chosen.begin(), chosen.end());
		}
		const uint32_t routes = rows * top_k;
		std::vector<uint32_t> order(routes);
		for ( uint32_t route = 0u; route < routes; ++route )
			order[route] = route;
		std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return experts[a] < experts[b]; });
		packed_row.assign(routes, 0u);
		source.assign(routes, 0u);
		for ( uint32_t packed = 0u; packed < routes; ++packed )
		{
			packed_row[order[packed]] = packed;
			source[packed] = order[packed] / top_k;
			offsets[experts[order[packed]] + 1u]++;
		}
		uint32_t distinct = 0u;
		for ( uint32_t group = 0u; group < groups; ++group )
		{
			distinct += offsets[group + 1u] != 0u ? 1u : 0u;
			offsets[group + 1u] += offsets[group];
		}
		VbRouting routing;
		routing.offsets = VbDevice(offsets);
		routing.source = VbDevice(source);
		routing.expert = VbDevice(experts);
		routing.packed_row = VbDevice(packed_row);
		routing.distinct = distinct;
		routing.routes = routes;
		routings.push_back(routing);
	}
	return routings;
}

static void VbFreeRoutings(std::vector<VbRouting> &routings)
{
	for ( VbRouting &routing : routings )
	{
		cudaFree(routing.offsets);
		cudaFree(routing.source);
		cudaFree(routing.expert);
		cudaFree(routing.packed_row);
	}
	routings.clear();
}

static double VbMeanDistinct(const std::vector<VbRouting> &routings)
{
	double total = 0.0;
	for ( const VbRouting &routing : routings )
		total += routing.distinct;
	return total / routings.size();
}

static void VbCells(const char *model, const char *name, uint32_t groups, uint32_t top_k, uint32_t input, uint32_t output, uint32_t packed)
{
	if ( VbSkip(model, name) )
		return;
	const uint32_t tiles = input / LM_SKINNY_CELL_TILE_K, cells = output / LM_SKINNY_CELL_NEURONS;
	const uint64_t group_bytes = (uint64_t)tiles * cells * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	uint8_t *weight = VbBytes(group_bytes * groups, 0u, 5u);
	uint16_t *activation = (uint16_t *)VbBytes((uint64_t)VB_MAX_ROWS * top_k * input * 2u, 2u, 77u), *result = 0;
	VB_CHECK(cudaMalloc((void **)&result, (uint64_t)VB_MAX_ROWS * top_k * output * 2u));
	VB_CHECK(cudaDeviceSynchronize());
	VbHeader(model, name, "mxfp4 cells", packed != 0u ? "LmSkinnyCellExperts, packed" : "LmSkinnyCellExperts, source tokens", top_k * group_bytes);
	float us_one = 0.0f;
	for ( uint32_t rows : kRows )
	{
		if ( gOnly != 0u && rows != 1u && rows != gOnly )
			continue;
		std::vector<VbRouting> routings = VbRoutings(rows, groups, top_k, input);
		auto launch = [&](uint32_t index) -> int32_t
		{
			const VbRouting &routing = routings[index % routings.size()];
			return LmSkinnyCellExperts(weight, activation, result, routing.offsets, packed != 0u ? 0 : routing.source, groups,
				routing.routes, rows, packed, input, output, LM_SKINNY_CELL_TILE_K, gStream);
		};
		const float us = VbTime(launch, VB_LAUNCHES);
		const uint64_t bytes = (uint64_t)(VbMeanDistinct(routings) * group_bytes);
		us_one = rows == 1u ? us : us_one;
		VbLine(rows, us, bytes, us_one, top_k * group_bytes);
		VbFreeRoutings(routings);
	}
	cudaFree(weight);
	cudaFree(activation);
	cudaFree(result);
}

enum VbExpertEntry
{
	VB_EXPERT_PER_ROUTE = 0,
	VB_EXPERT_GROUPED = 1
};

template<class Format>
static void VbExperts(const char *model, const char *name, VbExpertEntry entry, uint32_t groups, uint32_t top_k, uint32_t input, uint32_t output, uint32_t packed)
{
	if ( VbSkip(model, name) )
		return;
	const uint64_t weight_bytes = (uint64_t)groups * output * input * Format::kStoredBits / 8u, scale_bytes = VbScaleBytes<Format>(groups, output, input);
	const uint64_t group_bytes = (weight_bytes + scale_bytes) / groups;
	uint8_t *weight = VbBytes(weight_bytes, VbWeightKind<Format>(), 3u), *scale_data = scale_bytes != 0u ? VbBytes(scale_bytes, std::is_same<Format, LmFp8>::value ? 3u : 1u, 4u) : 0;
	uint16_t *activation = (uint16_t *)VbBytes((uint64_t)VB_MAX_ROWS * top_k * input * 2u, 2u, 66u), *result = 0;
	VB_CHECK(cudaMalloc((void **)&result, (uint64_t)VB_MAX_ROWS * top_k * output * 2u));
	VB_CHECK(cudaDeviceSynchronize());
	const LmScaleTensor scale = VbScale<Format>(scale_data, groups, output, input);
	char entry_name[96];
	snprintf(entry_name, sizeof(entry_name), "%s%s", entry == VB_EXPERT_PER_ROUTE ? "LmSkinnyExperts" : "LmSkinnyGroupedExperts any_rows",
		packed != 0u ? ", packed" : ", source tokens");
	VbHeader(model, name, VbFormatName<Format>(), entry_name, top_k * group_bytes);
	float us_one = 0.0f;
	for ( uint32_t rows : kRows )
	{
		if ( gOnly != 0u && rows != 1u && rows != gOnly )
			continue;
		if ( entry == VB_EXPERT_PER_ROUTE && rows * top_k > LM_SKINNY_ROWS * top_k )
			continue;
		std::vector<VbRouting> routings = VbRoutings(rows, groups, top_k, input + output);
		auto launch = [&](uint32_t index) -> int32_t
		{
			const VbRouting &routing = routings[index % routings.size()];
			if ( entry == VB_EXPERT_PER_ROUTE )
				return LmSkinnyExperts<Format>(weight, scale, activation, result, routing.expert, routing.packed_row, routing.routes, top_k, packed, input, output, gStream);
			return LmSkinnyGroupedExperts<Format>(weight, scale, activation, result, routing.offsets, routing.source, groups, routing.routes, packed, input, output,
				gStream, 1u);
		};
		const float us = VbTime(launch, VB_LAUNCHES);
		const uint64_t bytes = (uint64_t)(VbMeanDistinct(routings) * group_bytes);
		us_one = rows == 1u ? us : us_one;
		VbLine(rows, us, bytes, us_one, top_k * group_bytes);
		VbFreeRoutings(routings);
	}
	cudaFree(weight);
	cudaFree(scale_data);
	cudaFree(activation);
	cudaFree(result);
}

static void VbDsv41(void)
{
	const char *m = "dsv41-flash";
	VbDense<LmMxfp8>(m, "q_a+kv_a", VB_ENTRY_LAUNCH, 5120u, 1280u, 512u, 0u);
	VbDense<LmMxfp8>(m, "q_b+idx_q_b", VB_ENTRY_LAUNCH, 1280u, 2048u, 256u, 0u);
	VbDense<LmMxfp8>(m, "o_a", VB_ENTRY_LAUNCH, 4096u, 512u, 0u, 0u);
	VbDense<LmMxfp8>(m, "o_b", VB_ENTRY_LAUNCH, 512u, 5120u, 0u, 0u);
	VbDense<LmBf16Format>(m, "compressor wkv", VB_ENTRY_LAUNCH, 5120u, 512u, 0u, 0u);
	VbDense<LmBf16Format>(m, "idx_wk", VB_ENTRY_LAUNCH, 512u, 128u, 0u, 0u);
	VbDense<LmBf16Format>(m, "idx_wp", VB_ENTRY_LAUNCH, 5120u, 2u, 0u, 0u);
	VbDense<LmBf16Format>(m, "router", VB_ENTRY_LAUNCH, 5120u, 384u, 0u, 1u);
	VbDense<LmMxfp8>(m, "shared w1+w3", VB_ENTRY_LAUNCH, 5120u, 128u, 128u, 0u);
	VbDense<LmMxfp8>(m, "shared w2", VB_ENTRY_LAUNCH, 128u, 5120u, 0u, 0u);
	VbCells(m, "experts w13", 384u, 6u, 5120u, 288u, 0u);
	VbCells(m, "experts w2", 384u, 6u, 2304u, 320u, 1u);
}

template<class Format>
static void VbGlm(const char *m, uint32_t hidden, uint32_t q_slice, uint32_t q_b_input, uint32_t q_b_rows, uint32_t o_input, uint32_t query_a, uint32_t index_query)
{
	VbDense<Format>(m, "q_a+kv_a", VB_ENTRY_ROWS, hidden, q_slice, 36u, 0u);
	VbDense<Format>(m, "q_b", VB_ENTRY_ROWS, q_b_input, q_b_rows, 0u, 0u);
	VbDense<Format>(m, "o_proj", VB_ENTRY_ROWS, o_input, hidden, 0u, 0u);
	if ( index_query != 0u )
		VbDense<Format>(m, "index_q", VB_ENTRY_ROWS, query_a, index_query, 0u, 0u);
	VbDense<LmBf16Format>(m, "router", VB_ENTRY_ROWS, hidden, 256u, 0u, 1u);
	VbDense<Format>(m, "shared gate_up", VB_ENTRY_ROWS, hidden, 256u, 0u, 0u);
	VbDense<Format>(m, "shared down", VB_ENTRY_ROWS, 128u, hidden, 0u, 0u);
	VbExperts<Format>(m, "experts w1 (decode)", VB_EXPERT_PER_ROUTE, 256u, 8u, hidden, 256u, 0u);
	VbExperts<Format>(m, "experts w1 (verify)", VB_EXPERT_GROUPED, 256u, 8u, hidden, 256u, 0u);
	VbExperts<Format>(m, "experts w2 (decode)", VB_EXPERT_PER_ROUTE, 256u, 8u, 128u, hidden, 1u);
	VbExperts<Format>(m, "experts w2 (verify)", VB_EXPERT_GROUPED, 256u, 8u, 128u, hidden, 1u);
}

int main(int argc, char **argv)
{
	const char *which = argc > 1 ? argv[1] : "all";
	gFilter = argc > 2 && strcmp(argv[2], "-") != 0 ? argv[2] : 0;
	gOnly = argc > 3 ? (uint32_t)atoi(argv[3]) : 0u;
	VB_CHECK(cudaStreamCreateWithFlags(&gStream, cudaStreamNonBlocking));
	if ( strcmp(which, "all") == 0 || strcmp(which, "dsv41") == 0 )
		VbDsv41();
	if ( strcmp(which, "all") == 0 || strcmp(which, "glm") == 0 )
	{
		VbGlm<LmBf16Format>("glm-5.3-full-bf16", 6144u, 128u, 2048u, 1024u, 1024u, 2048u, 4096u);
		VbGlm<LmFp8>("glm-5.3-full-fp8", 6144u, 128u, 2048u, 1024u, 1024u, 2048u, 4096u);
	}
	if ( strcmp(which, "all") == 0 || strcmp(which, "axk2") == 0 )
		VbGlm<LmFp8>("ax-k2", 7168u, 96u, 3072u, 1280u, 512u, 1536u, 0u);
	VB_CHECK(cudaDeviceSynchronize());
	return 0;
}
