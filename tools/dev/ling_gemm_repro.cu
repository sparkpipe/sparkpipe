// standalone dense-GEMM hang repro: rows sweep over the real ling shapes,
// driving the prebuilt module archive's LingGemmBf16 entry (no device code here)
#include "inference/kernels/scale.cuh"
#include <chrono>
#include <cstdio>
#include <thread>

#include <cuda_runtime.h>

struct LmGemmArguments
{
	LmScaleTensor scale_a;
	LmScaleTensor scale_b;
	uint32_t prefix_built;
	const uint32_t *group_row_offset;
	uint32_t *group_tile_prefix;
	const void *activation_bytes;
	const uint32_t *source_row_map;
	uint32_t source_row_count;
	void *output_bf16;
	void *output_f32;
	void *accumulate_bf16;
	uint32_t output_row_stride;
	uint32_t output_column_offset;
	uint32_t group_count;
	uint32_t input_dimension;
	uint32_t output_dimension;
	void *frame_error;
};

extern "C" int32_t LingGemmBf16(
	LmGemmArguments *arguments,
	const void *activation_bf16,
	const void *weight_bf16,
	uint32_t packed_rows,
	uint32_t tokens,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	uint32_t multiprocessors,
	bool grouped,
	void *stream_handle);

extern "C" int32_t SparkLingConfigureCudaModule(uint32_t *multiprocessor_count);

struct Shape
{
	uint32_t in, out;
};

static const Shape SHAPES[] = {
	{2560u, 2560u},
	{2560u, 16320u},
	{4096u, 2560u},
	{2560u, 12288u},
	{6144u, 2560u},
	{2560u, 6144u},
};
static const uint32_t SHAPE_COUNT = sizeof(SHAPES) / sizeof(SHAPES[0]);
static const uint32_t ROWS[] = {1u, 2u, 3u, 4u, 8u, 16u, 17u};
static const uint32_t ROW_COUNT = sizeof(ROWS) / sizeof(ROWS[0]);
static const uint32_t TIMEOUT_MS = 8000u;

static int RunOne(uint32_t rows, uint32_t in_dim, uint32_t out_dim, uint32_t sms)
{
	uint16_t *activation = 0, *weight = 0, *output = 0;
	uint32_t *row_offset = 0, *tile_prefix = 0;
	cudaStream_t stream = 0;
	LmGemmArguments args;
	int32_t status;
	std::chrono::steady_clock::time_point start;
	int completed = 0;

	if (cudaStreamCreate(&stream) != cudaSuccess)
	{
		fprintf(stderr, "stream create failed\n");
		return 1;
	}
	if (cudaMalloc(&activation, (size_t)rows * in_dim * 2u) != cudaSuccess ||
		cudaMalloc(&weight, (size_t)out_dim * in_dim * 2u) != cudaSuccess ||
		cudaMalloc(&output, (size_t)rows * out_dim * 2u) != cudaSuccess ||
		cudaMalloc(&row_offset, 2u * sizeof(uint32_t)) != cudaSuccess ||
		cudaMalloc(&tile_prefix, 2u * sizeof(uint32_t)) != cudaSuccess)
	{
		fprintf(stderr, "alloc failed\n");
		return 1;
	}
	if (cudaMemset(activation, 0x3c, (size_t)rows * in_dim * 2u) != cudaSuccess ||
		cudaMemset(weight, 0x38, (size_t)out_dim * in_dim * 2u) != cudaSuccess ||
		cudaMemset(output, 0, (size_t)rows * out_dim * 2u) != cudaSuccess)
	{
		fprintf(stderr, "memset failed\n");
		return 1;
	}
	{
		const uint32_t host_offset[2] = {0u, rows};
		const uint32_t host_prefix[2] = {0u, (out_dim + 127u) / 128u};
		if (cudaMemcpy(row_offset, host_offset, sizeof(host_offset), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(tile_prefix, host_prefix, sizeof(host_prefix), cudaMemcpyHostToDevice) != cudaSuccess)
		{
			fprintf(stderr, "h2d failed\n");
			return 1;
		}
	}
	memset(&args, 0, sizeof(args));
	args.scale_a = LmScaleTensorNone();
	args.scale_b = LmScaleTensorNone();
	args.group_row_offset = row_offset;
	args.group_tile_prefix = tile_prefix;
	args.output_bf16 = output;
	args.output_row_stride = out_dim;
	args.output_column_offset = 0u;
	status = LingGemmBf16(&args, activation, weight, rows, rows, 1u,
		in_dim, out_dim, sms, false, stream);
	if (status != 0)
	{
		fprintf(stderr, "launch refused status %d\n", status);
		return 1;
	}
	start = std::chrono::steady_clock::now();
	while (std::chrono::duration_cast<std::chrono::milliseconds>(
				   std::chrono::steady_clock::now() - start)
				   .count() < TIMEOUT_MS)
	{
		if (cudaStreamQuery(stream) == cudaSuccess)
		{
			completed = 1;
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	printf("rows=%u in=%u out=%u %s\n", rows, in_dim, out_dim,
		completed ? "PASS" : "HANG");
	fflush(stdout);
	return completed ? 0 : 2;
}

int main(void)
{
	uint32_t sms = 48u;
	uint32_t shape, row_index;
	int failures = 0;
	if (SparkLingConfigureCudaModule(&sms) != 0)
		fprintf(stderr, "configure failed, assuming %u SMs\n", sms);
	printf("sms=%u\n", sms);
	fflush(stdout);
	for (shape = 0; shape < SHAPE_COUNT; ++shape)
	{
		for (row_index = 0; row_index < ROW_COUNT; ++row_index)
		{
			if (RunOne(ROWS[row_index], SHAPES[shape].in, SHAPES[shape].out, sms) != 0)
			{
				++failures;
				cudaDeviceReset();
			}
		}
	}
	printf("hangs=%d\n", failures);
	return failures != 0;
}
