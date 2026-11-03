#include "inference/kernels/skinny.cuh"
#include "inference/kernels/weight_codec.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

#define ED_MAX_ROWS 512u
#define ED_ENTRY_AUTO 0u
#define ED_ENTRY_ROWS 1u
#define ED_ENTRY_SPLIT 2u
#define ED_ENTRY_MULTI 3u

typedef struct EdCase
{
	const char *name;
	uint32_t entry;
	uint32_t input;
	uint32_t output;
	uint32_t f32;
	uint32_t second;
	uint32_t stride;
	uint32_t offset;
} EdCase;

__global__ static void EdFillKernel(uint8_t *bytes, uint64_t count, uint32_t seed, uint32_t kind, float scale)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		uint32_t value = (uint32_t)index * 0x9e3779b9u ^ seed;
		value ^= value >> 16u;
		value *= 0x7feb352du;
		value ^= value >> 15u;
		value *= 0x846ca68bu;
		value ^= value >> 16u;
		if ( kind == 0u )
			bytes[index] = (uint8_t)((value & 0x80u) | ((1u + (value >> 8u) % 8u) << 3u) | ((value >> 16u) & 7u));
		else if ( kind == 1u )
			bytes[index] = (uint8_t)(118u + (value >> 8u) % 5u);
		else if ( (index & 1u) == 0u )
		{
			const uint16_t bits = LmFloatToBf16(((float)(value >> 8u) / 8388608.0f - 1.0f) * scale);
			bytes[index] = (uint8_t)bits;
			bytes[index + 1u] = (uint8_t)(bits >> 8u);
		}
	}
}

static uint8_t *EdBytes(uint64_t count, uint32_t kind, float scale)
{
	uint8_t *device = 0;
	if ( cudaMalloc((void **)&device, count + 16u) != cudaSuccess )
	{
		printf("FAIL allocation of %llu bytes\n", (unsigned long long)count);
		exit(1);
	}
	EdFillKernel<<<1024u, 256u>>>(device, count, vf_seed++ * 2654435761u, kind, scale);
	return device;
}

template<class Format>
static LmScaleTensor EdScale(uint32_t rows, uint32_t input, uint8_t **storage)
{
	*storage = 0;
	if constexpr ( std::is_same<Format, LmFp8>::value )
	{
		std::vector<float> host((uint64_t)rows * ((input + 127u) / 128u));
		for ( uint64_t index = 0u; index < host.size(); ++index )
			host[index] = (0.75f + 0.25f * Uniform(vf_seed, (uint32_t)index)) / sqrtf((float)input);
		vf_seed++;
		if ( cudaMalloc((void **)storage, host.size() * sizeof(float)) != cudaSuccess ||
			cudaMemcpy(*storage, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess )
			exit(1);
		return LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(*storage, 1u, rows, input);
	}
	else if constexpr ( std::is_same<Format, LmMxfp8>::value )
	{
		*storage = EdBytes((uint64_t)rows * (input / 32u), 1u, 0.0f);
		return LmScaleTensorBuild(*storage, LM_SCALE_ENCODING_UE8M0, 1u, rows, input, 1u, 32u);
	}
	else
		return LmScaleTensorNone();
}

template<class Format>
static int32_t EdLaunch(const EdCase &c, const void *const *weights, const LmScaleTensor *scales, const uint16_t *activation, void *output, uint32_t rows,
	uint32_t any_rows)
{
	const uint32_t stride = c.stride != 0u ? c.stride : c.output;
	if ( c.entry == ED_ENTRY_MULTI )
	{
		LmSkinnyDenseTarget targets[2];
		memset(targets, 0, sizeof(targets));
		for ( uint32_t part = 0u; part < 2u; ++part )
		{
			targets[part].weight = weights[part];
			targets[part].output_bf16 = (uint16_t *)output;
			targets[part].output_dimension = part == 0u ? c.output : c.second;
			targets[part].output_row_stride = stride;
			targets[part].output_column_offset = part == 0u ? 0u : c.offset;
			targets[part].scale = scales[part];
		}
		return LmSkinnyDenseMulti<Format>(targets, 2u, activation, rows, c.input, 0, any_rows);
	}
	if constexpr ( Format::kScaleGroup == 0u )
		if ( c.entry == ED_ENTRY_SPLIT )
			return LmSkinnyDenseSplit<Format>(weights[0], activation, c.f32 != 0u ? (uint16_t *)0 : (uint16_t *)output, c.f32 != 0u ? (float *)output : (float *)0,
				rows, c.input, c.output, stride, c.offset, 0, any_rows);
	LmSkinnyArguments args;
	memset(&args, 0, sizeof(args));
	args.weight = (const uint8_t *)weights[0];
	args.activation = activation;
	args.output_bf16 = c.f32 != 0u ? (uint16_t *)0 : (uint16_t *)output;
	args.output_f32 = c.f32 != 0u ? (float *)output : (float *)0;
	args.scale = scales[0];
	args.rows = rows;
	args.input_dimension = c.input;
	args.output_dimension = c.output;
	args.output_row_stride = stride;
	args.output_column_offset = c.offset;
	if ( c.entry == ED_ENTRY_ROWS )
	{
		const int32_t status = LmSkinnyValidate<Format>(&args, any_rows);
		return status != LM_LAUNCH_OK ? status : LmSkinnyLaunchRows<Format>(&args, c.input / LmSkinnyFormat<Format>::kElements, 0);
	}
	return LmSkinnyLaunch<Format>(&args, 0, any_rows);
}

template<class Format>
static void EdCheck(const EdCase &c)
{
	static const uint32_t frames[] = { 1u, 2u, 7u, 16u, 17u, 40u, 256u, ED_MAX_ROWS };
	const uint32_t stride = c.stride != 0u ? c.stride : c.output, element = c.f32 != 0u ? 4u : 2u;
	const uint32_t parts = c.entry == ED_ENTRY_MULTI ? 2u : 1u;
	void *weights[2] = { 0, 0 };
	uint8_t *scale_storage[2] = { 0, 0 };
	LmScaleTensor scales[2];
	for ( uint32_t part = 0u; part < parts; ++part )
	{
		const uint32_t outputs = part == 0u ? c.output : c.second;
		const uint64_t bytes = (uint64_t)outputs * c.input * Format::kStoredBits / 8u;
		weights[part] = EdBytes(bytes, std::is_same<Format, LmBf16Format>::value ? 2u : 0u, 1.7f / sqrtf((float)c.input));
		scales[part] = EdScale<Format>(outputs, c.input, &scale_storage[part]);
	}
	uint16_t *activation = (uint16_t *)EdBytes((uint64_t)ED_MAX_ROWS * c.input * 2u, 2u, 1.0f);
	uint8_t *frame = 0, *decode = 0;
	const uint64_t row_bytes = (uint64_t)stride * element;
	uint32_t checked = 0u;
	std::vector<uint8_t> host_frame, host_decode;
	if ( cudaMalloc((void **)&frame, ED_MAX_ROWS * row_bytes) != cudaSuccess || cudaMalloc((void **)&decode, ED_MAX_ROWS * row_bytes) != cudaSuccess ||
		cudaDeviceSynchronize() != cudaSuccess )
	{
		Require(0, c.name, "buffers allocate", 0u, 0u, 0u);
		return;
	}
	Require(EdLaunch<Format>(c, weights, scales, activation, frame, 17u, 0u) == LM_LAUNCH_ERR_SHAPE, c.name,
		"a launch above 16 rows without any_rows is refused", 17u, 0u, 0u);
	for ( uint32_t rows : frames )
	{
		cudaMemset(frame, 0xa5, ED_MAX_ROWS * row_bytes);
		cudaMemset(decode, 0xa5, ED_MAX_ROWS * row_bytes);
		Require(EdLaunch<Format>(c, weights, scales, activation, frame, rows, 1u) == LM_LAUNCH_OK, c.name, "the frame launch runs (rows)", rows, 0u, 0u);
		for ( uint32_t row = 0u; row < rows; ++row )
			Require(EdLaunch<Format>(c, weights, scales, activation + (uint64_t)row * c.input, decode + row * row_bytes, 1u, 0u) == LM_LAUNCH_OK, c.name,
				"the one-row decode launch runs (rows, row)", rows, row, 0u);
		Require(cudaDeviceSynchronize() == cudaSuccess, c.name, "the launches complete", rows, 0u, 0u);
		host_frame.resize(ED_MAX_ROWS * row_bytes);
		host_decode.resize(ED_MAX_ROWS * row_bytes);
		cudaMemcpy(host_frame.data(), frame, host_frame.size(), cudaMemcpyDeviceToHost);
		cudaMemcpy(host_decode.data(), decode, host_decode.size(), cudaMemcpyDeviceToHost);
		for ( uint32_t row = 0u; row < ED_MAX_ROWS; ++row )
		{
			const int same = memcmp(host_frame.data() + row * row_bytes, host_decode.data() + row * row_bytes, row_bytes) == 0;
			Require(same, c.name, row < rows ? "every row of the frame launch equals its one-row decode bit for bit, and padding columns stay untouched (rows, row)" :
				"rows past the frame are not written (rows, row)", rows, row, 0u);
			checked += row < rows ? 1u : 0u;
		}
	}
	printf("%s %s: %u rows against one-row decode\n", vf_failures == 0 ? "ok" : "checked", c.name, checked);
	for ( uint32_t part = 0u; part < parts; ++part )
	{
		cudaFree(weights[part]);
		cudaFree(scale_storage[part]);
	}
	cudaFree(activation);
	cudaFree(frame);
	cudaFree(decode);
}

int main(void)
{
	static const EdCase fp8[] =
	{
		{ "glm-5.3-full q_a|kv_a slices", ED_ENTRY_MULTI, 6144u, 128u, 0u, 36u, 168u, 132u },
		{ "glm-5.3-full q_b", ED_ENTRY_ROWS, 2048u, 1024u, 0u, 0u, 0u, 0u },
		{ "glm-5.3-full o_proj", ED_ENTRY_ROWS, 1024u, 6144u, 0u, 0u, 0u, 0u },
		{ "glm-5.3-full dense gate_up", ED_ENTRY_ROWS, 6144u, 1536u, 0u, 0u, 0u, 0u },
		{ "glm-5.3-full shared down", ED_ENTRY_ROWS, 128u, 6144u, 0u, 0u, 0u, 0u },
		{ "glm-5.3-full index_q", ED_ENTRY_ROWS, 2048u, 4096u, 0u, 0u, 0u, 0u },
		{ "a.x-k2 q_b (query concat)", ED_ENTRY_ROWS, 3072u, 1280u, 0u, 0u, 0u, 0u },
		{ "a.x-k2 index_k", ED_ENTRY_ROWS, 7168u, 128u, 0u, 0u, 0u, 0u },
		{ "qwen-3.8 fp8 qkv", ED_ENTRY_AUTO, 5120u, 896u, 0u, 0u, 0u, 0u },
	};
	static const EdCase bf16[] =
	{
		{ "glm-5.3-full router", ED_ENTRY_ROWS, 6144u, 256u, 1u, 0u, 0u, 0u },
		{ "glm-5.3-full index_head", ED_ENTRY_ROWS, 6144u, 32u, 0u, 0u, 0u, 0u },
		{ "gemma-4-31b sliding qkv", ED_ENTRY_AUTO, 5376u, 1024u, 0u, 0u, 0u, 0u },
		{ "gemma-4-31b out", ED_ENTRY_AUTO, 512u, 5376u, 0u, 0u, 0u, 0u },
		{ "gemma-4-31b gate_up", ED_ENTRY_AUTO, 5376u, 2688u, 0u, 0u, 0u, 0u },
		{ "gemma-4-26b router", ED_ENTRY_AUTO, 2816u, 128u, 0u, 0u, 0u, 0u },
		{ "solar-open-2 gqa_qkv", ED_ENTRY_AUTO, 4096u, 640u, 0u, 0u, 0u, 0u },
		{ "solar-open-2 kda_beta", ED_ENTRY_AUTO, 4096u, 4u, 0u, 0u, 0u, 0u },
		{ "solar-open-2 router", ED_ENTRY_AUTO, 4096u, 320u, 1u, 0u, 0u, 0u },
		{ "ling-3.0 mla_kv_a", ED_ENTRY_AUTO, 2560u, 576u, 0u, 0u, 0u, 0u },
		{ "ling-3.0 kda_f low rank", ED_ENTRY_AUTO, 128u, 256u, 0u, 0u, 0u, 0u },
		{ "hy4 gate", ED_ENTRY_AUTO, 6144u, 1024u, 0u, 0u, 0u, 0u },
		{ "hy4 index_weights", ED_ENTRY_AUTO, 6144u, 32u, 1u, 0u, 0u, 0u },
		{ "qwen-3.8 out", ED_ENTRY_AUTO, 384u, 5120u, 0u, 0u, 0u, 0u },
		{ "qwen-3.8 gate_up", ED_ENTRY_AUTO, 5120u, 2176u, 0u, 0u, 0u, 0u },
		{ "glm-5.3-flash kda qkv|gate", ED_ENTRY_MULTI, 4096u, 1536u, 0u, 512u, 2048u, 1536u },
		{ "glm-5.3-flash o_proj", ED_ENTRY_AUTO, 1024u, 4096u, 0u, 0u, 0u, 0u },
		{ "kimi-k3 q_down split", ED_ENTRY_SPLIT, 7168u, 96u, 0u, 0u, 0u, 0u },
		{ "kimi-k3 router", ED_ENTRY_AUTO, 7168u, 384u, 1u, 0u, 0u, 0u },
	};
	static const EdCase mxfp8[] =
	{
		{ "hy4 q_a", ED_ENTRY_AUTO, 6144u, 2048u, 0u, 0u, 0u, 0u },
		{ "hy4 kv_a", ED_ENTRY_AUTO, 6144u, 576u, 0u, 0u, 0u, 0u },
		{ "hy4 out", ED_ENTRY_AUTO, 1024u, 6144u, 0u, 0u, 0u, 0u },
		{ "hy4 index_k", ED_ENTRY_AUTO, 6144u, 128u, 0u, 0u, 0u, 0u },
	};
	for ( const EdCase &c : fp8 )
		EdCheck<LmFp8>(c);
	for ( const EdCase &c : bf16 )
		EdCheck<LmBf16Format>(c);
	for ( const EdCase &c : mxfp8 )
		EdCheck<LmMxfp8>(c);
	if ( vf_failures != 0 )
	{
		printf("FAIL exact dense rows: %d failures\n", vf_failures);
		return 1;
	}
	printf("PASS exact dense rows: every row of a dense, split or paired skinny launch of any frame equals its one-row decode bit for bit\n");
	return 0;
}
