#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "runtime/launch.h"
#include "inference/kernels/skinny.cuh"

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)
#define MAX_ROWS 256u

static const uint32_t kRowCounts[] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 12u, 15u, 16u, 17u, 31u, 33u, 64u, 233u, 256u};

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
	T *device;
	CUDA(cudaMalloc((void **)&device, host.size() * sizeof(T) + 16u));
	CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
	return device;
}

static uint16_t *Activations(uint32_t rows, uint32_t input, uint64_t salt)
{
	std::vector<uint16_t> activation((size_t)rows * input);
	for ( size_t index = 0u; index < activation.size(); ++index )
		activation[index] = (uint16_t)(0x3c00u + (Mix(index ^ salt) & 0x7ffu)) ^ (uint16_t)((index & 1u) << 15u);
	return Device(activation);
}

struct Weights
{
	uint8_t *weight;
	uint8_t *scale;
	uint32_t input, output;
};

template<class Format>
static LmScaleTensor Scale(const Weights &w)
{
	if constexpr ( LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_UE8M0 )
		return LmScaleTensorBlockUe8m0(w.scale, 1u, w.output, w.input, 1u, LM_MXFP8_GROUP);
	else
		return LmScaleTensorNone();
}

static Weights Make(uint32_t input, uint32_t output, uint32_t element_bytes, uint64_t salt)
{
	Weights w = {0, 0, input, output};
	std::vector<uint8_t> bytes((size_t)input * output * element_bytes), scale((size_t)output * (input / 32u));
	for ( size_t index = 0u; index < bytes.size(); ++index )
		bytes[index] = (uint8_t)(Mix(index ^ salt) >> 33u);
	if ( element_bytes == 2u )
		for ( size_t index = 1u; index < bytes.size(); index += 2u )
			bytes[index] = (uint8_t)(0x3cu + (bytes[index] & 0x03u));
	else
		for ( size_t index = 0u; index < bytes.size(); ++index )
			bytes[index] &= 0x7eu;
	for ( size_t index = 0u; index < scale.size(); ++index )
		scale[index] = (uint8_t)(120u + Mix(index ^ salt ^ 0x55u) % 9u);
	w.weight = Device(bytes);
	w.scale = Device(scale);
	return w;
}

template<class Format>
static LmSkinnyArguments Arguments(const Weights &w, const uint16_t *activation, void *output, uint32_t f32, uint32_t rows)
{
	LmSkinnyArguments args;
	memset(&args, 0, sizeof(args));
	args.weight = w.weight;
	args.activation = activation;
	if ( f32 != 0u )
		args.output_f32 = (float *)output;
	else
		args.output_bf16 = (uint16_t *)output;
	args.scale = Scale<Format>(w);
	args.rows = rows;
	args.input_dimension = w.input;
	args.output_dimension = w.output;
	return args;
}

template<class Format>
static int32_t Launch(const Weights &w, const uint16_t *activation, void *output, uint32_t f32, uint32_t rows)
{
	LmSkinnyArguments args = Arguments<Format>(w, activation, output, f32, rows);
	return rows <= LM_SKINNY_ROWS_WIDE ? LmSkinnyLaunch<Format>(&args, 0) : LmSkinnyLaunchRowChunks<Format>(&args, 0);
}

template<class Format>
static LmSkinnyDenseTarget Target(const Weights &w, uint16_t *output)
{
	LmSkinnyDenseTarget target;
	memset(&target, 0, sizeof(target));
	target.weight = w.weight;
	target.output_bf16 = output;
	target.output_dimension = w.output;
	target.scale = Scale<Format>(w);
	return target;
}

static std::vector<uint8_t> Host(const void *device, size_t bytes)
{
	std::vector<uint8_t> host(bytes);
	CUDA(cudaDeviceSynchronize());
	CUDA(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
	return host;
}

template<class Format>
static uint32_t DenseCase(const char *name, uint32_t element_bytes, uint32_t f32, uint32_t input, uint32_t first_output, uint32_t second_output)
{
	const Weights first = Make(input, first_output, element_bytes, 3u + input + first_output), second = Make(input, second_output, element_bytes, 7u + second_output);
	const uint32_t width = f32 != 0u ? 4u : 2u;
	uint16_t *activation = Activations(MAX_ROWS, input, 0x99u + input);
	void *wide, *narrow, *pair_first, *pair_second;
	CUDA(cudaMalloc(&wide, (size_t)MAX_ROWS * first_output * width));
	CUDA(cudaMalloc(&narrow, (size_t)MAX_ROWS * (first_output > second_output ? first_output : second_output) * width));
	CUDA(cudaMalloc(&pair_first, (size_t)MAX_ROWS * first_output * 2u));
	CUDA(cudaMalloc(&pair_second, (size_t)MAX_ROWS * second_output * 2u));
	for ( uint32_t row = 0u; row < MAX_ROWS; ++row )
		REQUIRE(Launch<Format>(first, activation + (size_t)row * input, (uint8_t *)narrow + (size_t)row * first_output * width, f32, 1u) == LM_LAUNCH_OK);
	const std::vector<uint8_t> reference = Host(narrow, (size_t)MAX_ROWS * first_output * width);
	std::vector<uint8_t> reference_pair_first, reference_pair_second;
	if ( f32 == 0u )
	{
		LmSkinnyDenseTarget targets[2];
		for ( uint32_t row = 0u; row < MAX_ROWS; ++row )
		{
			targets[0] = Target<Format>(first, (uint16_t *)pair_first + (size_t)row * first_output);
			targets[1] = Target<Format>(second, (uint16_t *)pair_second + (size_t)row * second_output);
			REQUIRE(LmSkinnyDenseMulti<Format>(targets, 2u, activation + (size_t)row * input, 1u, input, 0) == LM_LAUNCH_OK);
		}
		reference_pair_first = Host(pair_first, (size_t)MAX_ROWS * first_output * 2u);
		reference_pair_second = Host(pair_second, (size_t)MAX_ROWS * second_output * 2u);
		REQUIRE(memcmp(reference_pair_first.data(), reference.data(), reference.size()) == 0);
	}
	uint32_t failures = 0u;
	for ( uint32_t rows : kRowCounts )
	{
		CUDA(cudaMemset(wide, 0xa5, (size_t)MAX_ROWS * first_output * width));
		REQUIRE(Launch<Format>(first, activation, wide, f32, rows) == LM_LAUNCH_OK);
		uint32_t differing = memcmp(Host(wide, (size_t)rows * first_output * width).data(), reference.data(), (size_t)rows * first_output * width) != 0 ? 1u : 0u;
		if ( f32 == 0u )
		{
			LmSkinnyDenseTarget targets[2] = { Target<Format>(first, (uint16_t *)pair_first), Target<Format>(second, (uint16_t *)pair_second) };
			CUDA(cudaMemset(pair_first, 0xa5, (size_t)MAX_ROWS * first_output * 2u));
			CUDA(cudaMemset(pair_second, 0xa5, (size_t)MAX_ROWS * second_output * 2u));
			REQUIRE((rows <= LM_SKINNY_ROWS_WIDE ? LmSkinnyDenseMulti<Format>(targets, 2u, activation, rows, input, 0) :
				LmSkinnyDenseMultiRowChunks<Format>(targets, 2u, activation, rows, input, 0)) == LM_LAUNCH_OK);
			differing += memcmp(Host(pair_first, (size_t)rows * first_output * 2u).data(), reference_pair_first.data(), (size_t)rows * first_output * 2u) != 0 ? 1u : 0u;
			differing += memcmp(Host(pair_second, (size_t)rows * second_output * 2u).data(), reference_pair_second.data(), (size_t)rows * second_output * 2u) != 0 ? 1u : 0u;
		}
		if ( differing != 0u )
			printf("FAIL %s %u -> %u/%u %s rows %u: a multi-row launch differs from one-row launches\n", name, input, first_output, second_output,
				f32 != 0u ? "f32" : "bf16", rows);
		failures += differing != 0u ? 1u : 0u;
	}
	if ( f32 != 0u )
		printf("%s %s %u -> %u f32: rows 1-256 against one-row launches\n", failures == 0u ? "ok  " : "FAIL", name, input, first_output);
	else
		printf("%s %s %u -> %u and %u bf16: rows 1-256 single and paired against one-row launches\n", failures == 0u ? "ok  " : "FAIL", name, input,
			first_output, second_output);
	cudaFree(first.weight); cudaFree(first.scale); cudaFree(second.weight); cudaFree(second.scale);
	cudaFree(activation); cudaFree(wide); cudaFree(narrow); cudaFree(pair_first); cudaFree(pair_second);
	return failures;
}

static uint32_t Sensitivity(void)
{
	const Weights w = Make(5120u, 512u, 1u, 0x77u);
	uint16_t *activation = Activations(17u, 5120u, 0x31u);
	uint16_t *output;
	CUDA(cudaMalloc((void **)&output, (size_t)17u * 512u * 2u));
	REQUIRE(Launch<LmMxfp8>(w, activation, output, 0u, 17u) == LM_LAUNCH_OK);
	const std::vector<uint8_t> before = Host(output, (size_t)17u * 512u * 2u);
	uint16_t bumped;
	CUDA(cudaMemcpy(&bumped, activation + 16u * 5120u + 9u, 2u, cudaMemcpyDeviceToHost));
	bumped ^= 0x0040u;
	CUDA(cudaMemcpy(activation + 16u * 5120u + 9u, &bumped, 2u, cudaMemcpyHostToDevice));
	REQUIRE(Launch<LmMxfp8>(w, activation, output, 0u, 17u) == LM_LAUNCH_OK);
	const std::vector<uint8_t> after = Host(output, (size_t)17u * 512u * 2u);
	const uint32_t rows_kept = memcmp(before.data(), after.data(), (size_t)16u * 512u * 2u) == 0 ? 1u : 0u;
	const uint32_t row_changed = memcmp(before.data() + (size_t)16u * 512u * 2u, after.data() + (size_t)16u * 512u * 2u, 512u * 2u) != 0 ? 1u : 0u;
	cudaFree(w.weight); cudaFree(w.scale); cudaFree(activation); cudaFree(output);
	printf("%s sensitivity: perturbing one element of row 16 %s row 16 and %s rows 0-15\n", rows_kept != 0u && row_changed != 0u ? "ok  " : "FAIL",
		row_changed != 0u ? "changes" : "does not change", rows_kept != 0u ? "keeps" : "changes");
	return rows_kept != 0u && row_changed != 0u ? 0u : 1u;
}

static uint32_t CellCase(uint32_t input, uint32_t output, uint32_t packed_activation, uint32_t frame_rows)
{
	const uint32_t groups = 16u, k_tiles = input / LM_SKINNY_CELL_TILE_K, cells = output / LM_SKINNY_CELL_NEURONS;
	const uint64_t tile_stride = (uint64_t)cells * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES, group_bytes = (uint64_t)k_tiles * tile_stride;
	const uint32_t counts[groups] = {5u, 0u, 1u, 3u, 19u, 2u, 0u, 1u, 4u, 0u, 9u, 1u, 1u, 0u, 2u, 6u};
	std::vector<uint8_t> weight((size_t)groups * group_bytes);
	for ( size_t index = 0u; index < weight.size(); ++index )
		weight[index] = (uint8_t)(Mix(index ^ (input * 31u + output)) >> 29u);
	for ( uint32_t group = 0u; group < groups; ++group )
		for ( uint32_t tile = 0u; tile < k_tiles; ++tile )
			for ( uint32_t cell = 0u; cell < cells; ++cell )
				for ( uint32_t sub = 0u; sub < LM_SKINNY_CELL_NEURONS; ++sub )
					weight[(size_t)group * group_bytes + (size_t)tile * tile_stride + (size_t)cell * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES +
						LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub] = (uint8_t)(118u + Mix(group * 977u + tile * 31u + cell * 7u + sub) % 9u);
	std::vector<uint32_t> offsets(groups + 1u, 0u), source;
	for ( uint32_t group = 0u; group < groups; ++group )
		offsets[group + 1u] = offsets[group] + counts[group];
	const uint32_t packed_rows = offsets[groups];
	for ( uint32_t row = 0u; row < packed_rows; ++row )
		source.push_back((uint32_t)(Mix(row * 13u + 5u) % frame_rows));
	const uint32_t activation_rows = packed_activation != 0u ? packed_rows : frame_rows;
	uint8_t *weight_d = Device(weight);
	uint16_t *activation = Activations(activation_rows, input, 0x4242u + input), *full, *single;
	uint32_t *offsets_d = Device(offsets), *source_d = Device(source), *one_offsets_d, *one_source_d;
	CUDA(cudaMalloc((void **)&full, (size_t)packed_rows * output * 2u));
	CUDA(cudaMalloc((void **)&single, (size_t)output * 2u));
	CUDA(cudaMalloc((void **)&one_offsets_d, (groups + 1u) * 4u));
	CUDA(cudaMalloc((void **)&one_source_d, 4u));
	REQUIRE(LmSkinnyCellExperts(weight_d, activation, full, offsets_d, packed_activation != 0u ? (const uint32_t *)0 : source_d, groups, packed_rows,
		frame_rows, packed_activation, input, output, LM_SKINNY_CELL_TILE_K, 0) == LM_LAUNCH_OK);
	const std::vector<uint8_t> batched = Host(full, (size_t)packed_rows * output * 2u);
	uint32_t differing = 0u;
	for ( uint32_t group = 0u; group < groups; ++group )
		for ( uint32_t row = offsets[group]; row < offsets[group + 1u]; ++row )
		{
			std::vector<uint32_t> one(groups + 1u);
			for ( uint32_t g = 0u; g <= groups; ++g )
				one[g] = g <= group ? 0u : 1u;
			CUDA(cudaMemcpy(one_offsets_d, one.data(), (groups + 1u) * 4u, cudaMemcpyHostToDevice));
			CUDA(cudaMemcpy(one_source_d, &source[row], 4u, cudaMemcpyHostToDevice));
			REQUIRE(LmSkinnyCellExperts(weight_d, packed_activation != 0u ? activation + (size_t)row * input : activation, single, one_offsets_d,
				packed_activation != 0u ? (const uint32_t *)0 : one_source_d, groups, 1u, 1u, packed_activation, input, output, LM_SKINNY_CELL_TILE_K, 0) ==
				LM_LAUNCH_OK);
			differing += memcmp(Host(single, (size_t)output * 2u).data(), batched.data() + (size_t)row * output * 2u, (size_t)output * 2u) != 0 ? 1u : 0u;
		}
	printf("%s cell experts %u -> %u %s activations, frame of %u rows, %u routes in 16 groups: %u routes differ from one-route launches\n",
		differing == 0u ? "ok  " : "FAIL", input, output, packed_activation != 0u ? "packed" : "token", frame_rows, packed_rows, differing);
	cudaFree(weight_d); cudaFree(activation); cudaFree(full); cudaFree(single); cudaFree(offsets_d); cudaFree(source_d); cudaFree(one_offsets_d);
	cudaFree(one_source_d);
	return differing != 0u ? 1u : 0u;
}

int main(void)
{
	uint32_t failures = 0u;
	failures += DenseCase<LmMxfp8>("mxfp8", 1u, 0u, 5120u, 1280u, 512u);
	failures += DenseCase<LmMxfp8>("mxfp8", 1u, 0u, 1280u, 2048u, 256u);
	failures += DenseCase<LmMxfp8>("mxfp8", 1u, 0u, 4096u, 512u, 512u);
	failures += DenseCase<LmMxfp8>("mxfp8", 1u, 0u, 512u, 5120u, 320u);
	failures += DenseCase<LmMxfp8>("mxfp8", 1u, 0u, 5120u, 256u, 128u);
	failures += DenseCase<LmMxfp8>("mxfp8", 1u, 0u, 256u, 5120u, 128u);
	failures += DenseCase<LmMxfp8>("mxfp8", 1u, 0u, 6144u, 1600u, 320u);
	failures += DenseCase<LmMxfp8>("mxfp8", 1u, 0u, 8192u, 320u, 640u);
	failures += DenseCase<LmBf16Format>("bf16", 2u, 1u, 5120u, 512u, 0u);
	failures += DenseCase<LmBf16Format>("bf16", 2u, 1u, 5120u, 384u, 0u);
	failures += DenseCase<LmBf16Format>("bf16", 2u, 0u, 512u, 128u, 128u);
	failures += DenseCase<LmBf16Format>("bf16", 2u, 0u, 5120u, 2u, 384u);
	failures += Sensitivity();
	failures += CellCase(5120u, 288u, 0u, 1u);
	failures += CellCase(5120u, 288u, 0u, 9u);
	failures += CellCase(5120u, 288u, 0u, 40u);
	failures += CellCase(2304u, 320u, 1u, 9u);
	failures += CellCase(2304u, 320u, 1u, 233u);
	if ( failures != 0u )
	{
		printf("FAIL skinny rows: %u cases give multi-row outputs that differ from one-row launches\n", failures);
		return 1;
	}
	printf("PASS skinny rows: every row of a multi-row or row-chunked launch is the same bits as a one-row launch of that row\n");
	return 0;
}
