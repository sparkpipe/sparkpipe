#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>
#include <vector>
#include "runtime/launch.h"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/weight_codec.cuh"

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)
#define FRAME_ROWS LM_SKINNY_ROWS_WIDE
#define FRAME_SENTINEL 0xa5u
#define FRAME_PARTS 3u

enum FrameEntry
{
	FRAME_ENTRY_LAUNCH = 0,
	FRAME_ENTRY_ROWS = 1,
	FRAME_ENTRY_MULTI = 2,
	FRAME_ENTRY_SHAPE = 3
};

static const char *kEntryName[] = { "LmSkinnyLaunch", "LmSkinnyLaunchRows", "LmSkinnyDenseMulti", "LmSkinnyFrameShape" };

struct FrameShape
{
	uint32_t input;
	uint32_t outputs[FRAME_PARTS];
	uint32_t f32;
};

static uint64_t compared_rows = 0u, compared_values = 0u, frame_shapes = 0u;

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
	T *device = 0;
	CUDA(cudaMalloc((void **)&device, host.size() * sizeof(T) + 64u));
	CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
	return device;
}

template<class Format>
static const char *FormatName(void)
{
	if constexpr ( std::is_same<Format, LmBf16Format>::value )
		return "bf16";
	else if constexpr ( std::is_same<Format, LmMxfp8>::value )
		return "mxfp8";
	else if constexpr ( std::is_same<Format, LmFp8>::value )
		return "fp8";
	else if constexpr ( std::is_same<Format, LmMxfp4>::value )
		return "mxfp4";
	else
		return "nvfp4";
}

template<class Format>
struct FrameWeight
{
	uint8_t *weight;
	uint8_t *scale;
	uint8_t *global;
	LmScaleTensor tensor;
};

template<class Format>
static FrameWeight<Format> MakeWeight(uint32_t input, uint32_t output, uint64_t salt)
{
	FrameWeight<Format> made;
	std::vector<uint8_t> weight((uint64_t)output * input * Format::kStoredBits / 8u);
	for ( size_t index = 0u; index < weight.size(); ++index )
	{
		const uint64_t draw = Mix(index ^ (salt << 32u));
		if constexpr ( std::is_same<Format, LmBf16Format>::value )
			weight[index] = (index & 1u) != 0u ? (uint8_t)(0x3cu | (draw & 0x80u)) : (uint8_t)draw;
		else if constexpr ( std::is_same<Format, LmFp8>::value || std::is_same<Format, LmMxfp8>::value )
			weight[index] = (uint8_t)((draw & 0x80u) | ((1u + (draw >> 8u) % 12u) << 3u) | ((draw >> 16u) & 7u));
		else
			weight[index] = (uint8_t)draw;
	}
	made.weight = Device(weight);
	made.scale = 0;
	made.global = 0;
	made.tensor = LmScaleTensorNone();
	if constexpr ( std::is_same<Format, LmFp8>::value )
	{
		std::vector<float> scale((uint64_t)output * ((input + 127u) / 128u));
		for ( size_t index = 0u; index < scale.size(); ++index )
			scale[index] = (float)(1u + Mix(index ^ salt) % 64u) / 1024.0f;
		made.scale = (uint8_t *)Device(scale);
		made.tensor = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(made.scale, 1u, output, input);
	}
	else if constexpr ( std::is_same<Format, LmMxfp8>::value || std::is_same<Format, LmMxfp4>::value )
	{
		std::vector<uint8_t> scale((uint64_t)output * ((input + Format::kScaleGroup - 1u) / Format::kScaleGroup));
		for ( size_t index = 0u; index < scale.size(); ++index )
			scale[index] = (uint8_t)(118u + Mix(index ^ salt) % 10u);
		made.scale = Device(scale);
		made.tensor = LmScaleTensorBlockUe8m0(made.scale, 1u, output, input, 1u, Format::kScaleGroup);
	}
	else if constexpr ( std::is_same<Format, LmNvfp4>::value )
	{
		std::vector<uint8_t> scale((uint64_t)output * ((input + Format::kScaleGroup - 1u) / Format::kScaleGroup));
		const std::vector<float> global(1u, 0.0625f);
		for ( size_t index = 0u; index < scale.size(); ++index )
			scale[index] = (uint8_t)(0x20u + Mix(index ^ salt) % 0x30u);
		made.scale = Device(scale);
		made.global = (uint8_t *)Device(global);
		made.tensor = LmScaleTensorBlockUe4m3F32Global(made.scale, made.global, 1u, output, input, Format::kScaleGroup);
	}
	return made;
}

template<class Format>
static void FreeWeight(FrameWeight<Format> &made)
{
	CUDA(cudaFree(made.weight));
	if ( made.scale != 0 )
		CUDA(cudaFree(made.scale));
	if ( made.global != 0 )
		CUDA(cudaFree(made.global));
}

template<class Format, uint32_t LANES, uint32_t RG, uint32_t NPG>
static int32_t ShapeLanes(LmSkinnyMultiArguments *multi)
{
	uint32_t multiprocessors = 0u;
	REQUIRE(LmSkinnyMultiprocessors(&multiprocessors) == LM_LAUNCH_OK);
	return LmSkinnyFrameShape<Format,LANES,RG,NPG>(multi, multiprocessors, 0);
}

template<class Format, uint32_t RG, uint32_t NPG>
static int32_t ShapeConfig(LmSkinnyMultiArguments *multi)
{
	const uint32_t chunks = multi->part[0].input_dimension / LmSkinnyFormat<Format>::kElements;
	if ( chunks >= 32u )
		return ShapeLanes<Format,32u,RG,NPG>(multi);
	if ( chunks >= 16u )
		return ShapeLanes<Format,16u,RG,NPG>(multi);
	return ShapeLanes<Format,8u,RG,NPG>(multi);
}

template<class Format>
static int32_t ShapeAt(LmSkinnyMultiArguments *multi, uint32_t config)
{
	switch ( config )
	{
	case 0u: return ShapeConfig<Format,4u,2u>(multi);
	case 1u: return ShapeConfig<Format,4u,1u>(multi);
	case 2u: return ShapeConfig<Format,2u,1u>(multi);
	default: return ShapeConfig<Format,1u,1u>(multi);
	}
}

template<class Format>
struct FrameCase
{
	FrameShape shape;
	uint32_t parts, width, offsets[FRAME_PARTS];
	FrameWeight<Format> weight[FRAME_PARTS];
	uint16_t *activation;
	uint8_t *output;
	uint64_t output_bytes;
};

template<class Format>
static uint32_t ElementBytes(const FrameCase<Format> &c)
{
	return c.shape.f32 != 0u ? 4u : 2u;
}

template<class Format>
static int32_t Launch(FrameCase<Format> &c, FrameEntry entry, uint32_t config, const uint16_t *activation, uint32_t rows, uint8_t *output)
{
	const uint32_t bytes = ElementBytes(c);
	if ( entry == FRAME_ENTRY_LAUNCH || entry == FRAME_ENTRY_ROWS )
	{
		LmSkinnyArguments args;
		memset(&args, 0, sizeof(args));
		args.weight = c.weight[0].weight;
		args.activation = activation;
		args.output_bf16 = c.shape.f32 != 0u ? 0 : (uint16_t *)(output + (uint64_t)c.offsets[0] * bytes);
		args.output_f32 = c.shape.f32 != 0u ? (float *)(output + (uint64_t)c.offsets[0] * bytes) : 0;
		args.scale = c.weight[0].tensor;
		args.rows = rows;
		args.input_dimension = c.shape.input;
		args.output_dimension = c.shape.outputs[0];
		args.output_row_stride = c.width;
		if ( entry == FRAME_ENTRY_LAUNCH )
			return LmSkinnyLaunch<Format>(&args, 0, 1u);
		const int32_t status = LmSkinnyValidate<Format>(&args, 1u);
		return status != LM_LAUNCH_OK ? status : LmSkinnyLaunchRows<Format>(&args, c.shape.input / LmSkinnyFormat<Format>::kElements, 0);
	}
	LmSkinnyDenseTarget targets[FRAME_PARTS];
	memset(targets, 0, sizeof(targets));
	for ( uint32_t part = 0u; part < c.parts; ++part )
	{
		targets[part].weight = c.weight[part].weight;
		targets[part].output_bf16 = c.shape.f32 != 0u ? 0 : (uint16_t *)output;
		targets[part].output_f32 = c.shape.f32 != 0u ? (float *)output : 0;
		targets[part].output_dimension = c.shape.outputs[part];
		targets[part].output_row_stride = c.width;
		targets[part].output_column_offset = c.offsets[part];
		targets[part].scale = c.weight[part].tensor;
	}
	if ( entry == FRAME_ENTRY_MULTI || rows == 1u )
		return LmSkinnyDenseMulti<Format>(targets, c.parts, activation, rows, c.shape.input, 0, 1u);
	LmSkinnyMultiArguments multi;
	memset(&multi, 0, sizeof(multi));
	multi.count = c.parts;
	for ( uint32_t part = 0u; part < c.parts; ++part )
	{
		LmSkinnyArguments *args = &multi.part[part];
		args->weight = (const uint8_t *)targets[part].weight;
		args->activation = activation;
		args->output_bf16 = targets[part].output_bf16;
		args->output_f32 = targets[part].output_f32;
		args->scale = targets[part].scale;
		args->rows = rows;
		args->input_dimension = c.shape.input;
		args->output_dimension = targets[part].output_dimension;
		args->output_row_stride = c.width;
		args->output_column_offset = targets[part].output_column_offset;
		REQUIRE(LmSkinnyValidate<Format>(args, 1u) == LM_LAUNCH_OK);
	}
	return ShapeAt<Format>(&multi, config);
}

template<class Format>
static void CheckEntry(FrameCase<Format> &c, FrameEntry entry, uint32_t config)
{
	const uint32_t bytes = ElementBytes(c);
	const uint64_t row_bytes = (uint64_t)c.width * bytes;
	std::vector<uint8_t> one((size_t)FRAME_ROWS * row_bytes), frame((size_t)FRAME_ROWS * row_bytes);
	for ( uint32_t row = 0u; row < FRAME_ROWS; ++row )
	{
		CUDA(cudaMemset(c.output, FRAME_SENTINEL, c.output_bytes));
		REQUIRE(Launch(c, entry == FRAME_ENTRY_SHAPE ? FRAME_ENTRY_MULTI : entry, config, c.activation + (uint64_t)row * c.shape.input, 1u, c.output) == LM_LAUNCH_OK);
		CUDA(cudaDeviceSynchronize());
		CUDA(cudaMemcpy(one.data() + (uint64_t)row * row_bytes, c.output, row_bytes, cudaMemcpyDeviceToHost));
	}
	for ( uint32_t rows = 2u; rows <= FRAME_ROWS; ++rows )
	{
		CUDA(cudaMemset(c.output, FRAME_SENTINEL, c.output_bytes));
		REQUIRE(Launch(c, entry, config, c.activation, rows, c.output) == LM_LAUNCH_OK);
		CUDA(cudaDeviceSynchronize());
		CUDA(cudaMemcpy(frame.data(), c.output, (size_t)FRAME_ROWS * row_bytes, cudaMemcpyDeviceToHost));
		for ( uint32_t row = 0u; row < FRAME_ROWS; ++row )
		{
			const uint8_t *expected = one.data() + (uint64_t)row * row_bytes, *got = frame.data() + (uint64_t)row * row_bytes;
			for ( uint64_t at = 0u; at < row_bytes; ++at )
			{
				const uint8_t want = row < rows ? expected[at] : (uint8_t)FRAME_SENTINEL;
				if ( got[at] != want )
				{
					fprintf(stderr, "FAIL skinny frame rows: %s %s K %u N %u+%u+%u (%s out), config %u, %u-row frame, row %u byte %llu: 0x%02x, the %s 0x%02x\n",
						FormatName<Format>(), kEntryName[entry], c.shape.input, c.shape.outputs[0], c.shape.outputs[1], c.shape.outputs[2],
						c.shape.f32 != 0u ? "f32" : "bf16", config, rows, row, (unsigned long long)at, got[at],
						row < rows ? "one-row launch wrote" : "untouched sentinel is", want);
					exit(1);
				}
			}
			if ( row < rows )
			{
				compared_rows++;
				compared_values += row_bytes / bytes;
			}
		}
	}
}

template<class Format>
static void CheckShape(const FrameShape &shape, uint64_t salt)
{
	FrameCase<Format> c;
	memset(&c, 0, sizeof(c));
	c.shape = shape;
	c.parts = 0u;
	c.width = 0u;
	for ( uint32_t part = 0u; part < FRAME_PARTS && shape.outputs[part] != 0u; ++part )
	{
		c.offsets[part] = c.width + 3u * part;
		c.width = c.offsets[part] + shape.outputs[part];
		c.weight[part] = MakeWeight<Format>(shape.input, shape.outputs[part], salt * 7u + part);
		c.parts++;
	}
	c.width += 5u;
	std::vector<uint16_t> activation((size_t)FRAME_ROWS * shape.input);
	for ( size_t index = 0u; index < activation.size(); ++index )
	{
		const uint64_t draw = Mix(index ^ (salt << 40u) ^ 0x51u);
		activation[index] = (uint16_t)((0x3c00u + (draw & 0x7ffu)) ^ (((draw >> 12u) & 1u) << 15u));
	}
	c.activation = Device(activation);
	c.output_bytes = (uint64_t)FRAME_ROWS * c.width * ElementBytes(c);
	CUDA(cudaMalloc((void **)&c.output, c.output_bytes));
	if ( c.parts == 1u )
	{
		CheckEntry(c, FRAME_ENTRY_LAUNCH, 0u);
		CheckEntry(c, FRAME_ENTRY_ROWS, 0u);
	}
	CheckEntry(c, FRAME_ENTRY_MULTI, 0u);
	for ( uint32_t config = 0u; config < 4u; ++config )
		CheckEntry(c, FRAME_ENTRY_SHAPE, config);
	frame_shapes++;
	for ( uint32_t part = 0u; part < c.parts; ++part )
		FreeWeight(c.weight[part]);
	CUDA(cudaFree(c.activation));
	CUDA(cudaFree(c.output));
}

template<class Format>
static void CheckFormat(const std::vector<FrameShape> &shapes)
{
	uint64_t salt = 1u;
	for ( const FrameShape &shape : shapes )
		CheckShape<Format>(shape, salt++);
	printf("ok %s: %zu shapes\n", FormatName<Format>(), shapes.size());
	fflush(stdout);
}

int main(void)
{
	CheckFormat<LmBf16Format>({ {5120u, {384u, 0u, 0u}, 1u}, {6144u, {256u, 0u, 0u}, 0u}, {6144u, {2u, 0u, 0u}, 0u}, {4096u, {256u, 0u, 0u}, 1u},
		{264u, {77u, 0u, 0u}, 0u}, {128u, {520u, 0u, 0u}, 0u}, {64u, {40u, 0u, 0u}, 1u}, {1024u, {6144u, 0u, 0u}, 0u}, {6144u, {128u, 36u, 0u}, 0u},
		{2048u, {96u, 64u, 32u}, 1u} });
	CheckFormat<LmMxfp8>({ {5120u, {1280u, 512u, 0u}, 0u}, {1280u, {2048u, 256u, 0u}, 0u}, {4096u, {512u, 0u, 0u}, 0u}, {512u, {5120u, 0u, 0u}, 0u},
		{128u, {5120u, 0u, 0u}, 0u}, {5120u, {128u, 128u, 0u}, 0u}, {528u, {70u, 0u, 0u}, 1u}, {256u, {50u, 30u, 20u}, 0u} });
	CheckFormat<LmFp8>({ {1024u, {6144u, 0u, 0u}, 0u}, {512u, {7168u, 0u, 0u}, 0u}, {6144u, {256u, 0u, 0u}, 0u}, {2048u, {4096u, 0u, 0u}, 0u},
		{6144u, {128u, 36u, 0u}, 0u}, {528u, {130u, 0u, 0u}, 1u}, {256u, {96u, 0u, 0u}, 0u}, {128u, {72u, 40u, 0u}, 0u} });
	CheckFormat<LmMxfp4>({ {1024u, {256u, 0u, 0u}, 0u}, {1056u, {100u, 0u, 0u}, 1u}, {512u, {64u, 0u, 0u}, 0u}, {256u, {48u, 24u, 0u}, 0u}, {4096u, {1024u, 0u, 0u}, 0u} });
	CheckFormat<LmNvfp4>({ {1024u, {256u, 0u, 0u}, 0u}, {1056u, {100u, 0u, 0u}, 1u}, {512u, {64u, 0u, 0u}, 0u}, {256u, {48u, 24u, 0u}, 0u}, {4096u, {1024u, 0u, 0u}, 0u} });
	REQUIRE(compared_rows != 0u);
	printf("every row of 2-16 row skinny frames equals its one-row launch bit for bit and no other output is written: %llu shapes, %llu rows, %llu outputs\n",
		(unsigned long long)frame_shapes, (unsigned long long)compared_rows, (unsigned long long)compared_values);
	printf("PASS skinny frame rows\n");
	return 0;
}
