#include "inference/kernels/skinny.cuh"
#include "inference/kernels/weight_codec.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define EB_EXPERTS 256u
#define EB_TOP_K 8u
#define EB_REPEATS 20u

typedef struct EbShape
{
	const char *name;
	uint32_t hidden;
	uint32_t up_rows;
	uint32_t inter;
} EbShape;

__global__ static void EbFillKernel(uint8_t *bytes, uint64_t count, uint32_t seed)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		uint32_t value = (uint32_t)index * 0x9e3779b9u ^ seed;
		value ^= value >> 16u;
		value *= 0x7feb352du;
		value ^= value >> 15u;
		bytes[index] = (uint8_t)((value & 0x80u) | ((1u + (value >> 8u) % 8u) << 3u) | ((value >> 16u) & 7u));
	}
}

static void *EbDevice(uint64_t bytes, uint32_t seed)
{
	void *device = 0;
	if ( cudaMalloc(&device, bytes) != cudaSuccess )
	{
		fprintf(stderr, "allocation of %llu bytes failed\n", (unsigned long long)bytes);
		exit(1);
	}
	EbFillKernel<<<1024u, 256u>>>((uint8_t *)device, bytes, seed);
	return device;
}

static uint32_t EbMix(uint32_t value)
{
	value ^= value >> 16u;
	value *= 0x7feb352du;
	value ^= value >> 15u;
	value *= 0x846ca68bu;
	return value ^ (value >> 16u);
}

template<class T>
static T *EbUpload(const std::vector<T> &host)
{
	T *device = 0;
	if ( cudaMalloc((void **)&device, host.size() * sizeof(T) + 16u) != cudaSuccess ||
		cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess )
		exit(1);
	return device;
}

static uint32_t EbRowsPerTask(uint32_t pairs)
{
	if ( pairs <= 2u * EB_EXPERTS )
		return LM_SKINNY_ROWS_MID;
	return pairs > LM_SKINNY_ROWS * EB_EXPERTS ? LM_SKINNY_ROWS_WIDE : LM_SKINNY_ROWS;
}

static void EbRun(const EbShape &shape)
{
	static const uint32_t frames[] = { 8u, 16u, 32u, 64u, 128u, 256u };
	const uint64_t up_slice = (uint64_t)shape.up_rows * shape.hidden, down_slice = (uint64_t)shape.hidden * shape.inter, slice = up_slice + down_slice;
	void *up = EbDevice(EB_EXPERTS * up_slice, 11u), *down = EbDevice(EB_EXPERTS * down_slice, 13u);
	std::vector<float> up_scale_host((uint64_t)EB_EXPERTS * shape.up_rows * ((shape.hidden + 127u) / 128u), 1.0f / 64.0f);
	std::vector<float> down_scale_host((uint64_t)EB_EXPERTS * shape.hidden * ((shape.inter + 127u) / 128u), 1.0f / 16.0f);
	float *up_scale = EbUpload(up_scale_host), *down_scale = EbUpload(down_scale_host);
	const LmScaleTensor up_tensor = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(up_scale, EB_EXPERTS, shape.up_rows, shape.hidden);
	const LmScaleTensor down_tensor = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(down_scale, EB_EXPERTS, shape.hidden, shape.inter);
	uint16_t *activation = 0, *gate_up = 0, *intermediate = 0, *output = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaMalloc((void **)&activation, (uint64_t)256u * shape.hidden * sizeof(uint16_t));
	cudaMalloc((void **)&gate_up, (uint64_t)256u * EB_TOP_K * shape.up_rows * sizeof(uint16_t));
	cudaMalloc((void **)&intermediate, (uint64_t)256u * EB_TOP_K * shape.inter * sizeof(uint16_t));
	cudaMalloc((void **)&output, (uint64_t)256u * EB_TOP_K * shape.hidden * sizeof(uint16_t));
	cudaMemset(activation, 0x3c, (uint64_t)256u * shape.hidden * sizeof(uint16_t));
	cudaMemset(intermediate, 0x3c, (uint64_t)256u * EB_TOP_K * shape.inter * sizeof(uint16_t));
	printf("%s: experts %u x (%u x %u up, %u x %u down) fp8, %.1f MiB of expert weights per rank\n", shape.name, EB_EXPERTS, shape.up_rows, shape.hidden,
		shape.hidden, shape.inter, (double)(EB_EXPERTS * slice) / 1048576.0);
	printf("%6s %8s %12s %12s %12s %12s %12s %10s\n", "rows", "experts", "decode_us", "chunked_us", "grouped_us", "chunked_MiB", "grouped_MiB", "speedup");
	for ( uint32_t rows : frames )
	{
		const uint32_t pairs = rows * EB_TOP_K;
		std::vector<uint32_t> expert(pairs), packed_row(pairs), source(pairs), offset(EB_EXPERTS + 1u, 0u), count(EB_EXPERTS, 0u), cursor(EB_EXPERTS, 0u);
		uint64_t chunked_slices = 0u, grouped_slices = 0u, touched = 0u;
		for ( uint32_t row = 0u; row < rows; ++row )
			for ( uint32_t k = 0u; k < EB_TOP_K; ++k )
			{
				uint32_t candidate, attempt = 0u, taken;
				do
				{
					candidate = EbMix(row * 977u + k * 131u + attempt++ * 7919u + rows) % EB_EXPERTS;
					taken = 0u;
					for ( uint32_t previous = 0u; previous < k; ++previous )
						taken |= expert[row * EB_TOP_K + previous] == candidate ? 1u : 0u;
				}
				while ( taken != 0u );
				expert[row * EB_TOP_K + k] = candidate;
				count[candidate]++;
			}
		for ( uint32_t group = 0u; group < EB_EXPERTS; ++group )
		{
			offset[group + 1u] = offset[group] + count[group];
			cursor[group] = offset[group];
			touched += count[group] != 0u ? 1u : 0u;
			grouped_slices += (count[group] + EbRowsPerTask(pairs) - 1u) / EbRowsPerTask(pairs);
		}
		for ( uint32_t pair = 0u; pair < pairs; ++pair )
		{
			packed_row[pair] = cursor[expert[pair]]++;
			source[packed_row[pair]] = pair / EB_TOP_K;
		}
		chunked_slices = pairs;
		uint32_t *expert_device = EbUpload(expert), *packed_device = EbUpload(packed_row), *source_device = EbUpload(source), *offset_device = EbUpload(offset);
		float decode_ms = 0.0f, chunked_ms = 0.0f, grouped_ms = 0.0f;
		for ( uint32_t pass = 0u; pass < 2u; ++pass )
		{
			cudaEventRecord(start);
			for ( uint32_t repeat = 0u; repeat < EB_REPEATS; ++repeat )
				for ( uint32_t row = 0u; row < rows; ++row )
				{
					LmSkinnyExperts<LmFp8>(up, up_tensor, activation + (uint64_t)row * shape.hidden, gate_up, expert_device + (uint64_t)row * EB_TOP_K,
						packed_device + (uint64_t)row * EB_TOP_K, EB_TOP_K, EB_TOP_K, 0u, shape.hidden, shape.up_rows, 0);
					LmSkinnyExperts<LmFp8>(down, down_tensor, intermediate, output, expert_device + (uint64_t)row * EB_TOP_K, packed_device + (uint64_t)row * EB_TOP_K,
						EB_TOP_K, EB_TOP_K, 1u, shape.inter, shape.hidden, 0);
				}
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&decode_ms, start, stop);
			cudaEventRecord(start);
			for ( uint32_t repeat = 0u; repeat < EB_REPEATS; ++repeat )
				for ( uint32_t first = 0u; first < rows; first += LM_SKINNY_ROWS )
				{
					const uint32_t chunk = rows - first < LM_SKINNY_ROWS ? rows - first : LM_SKINNY_ROWS;
					LmSkinnyExperts<LmFp8>(up, up_tensor, activation + (uint64_t)first * shape.hidden, gate_up, expert_device + (uint64_t)first * EB_TOP_K,
						packed_device + (uint64_t)first * EB_TOP_K, chunk * EB_TOP_K, EB_TOP_K, 0u, shape.hidden, shape.up_rows, 0);
					LmSkinnyExperts<LmFp8>(down, down_tensor, intermediate, output, expert_device + (uint64_t)first * EB_TOP_K,
						packed_device + (uint64_t)first * EB_TOP_K, chunk * EB_TOP_K, EB_TOP_K, 1u, shape.inter, shape.hidden, 0);
				}
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&chunked_ms, start, stop);
			cudaEventRecord(start);
			for ( uint32_t repeat = 0u; repeat < EB_REPEATS; ++repeat )
			{
				LmSkinnyGroupedExperts<LmFp8>(up, up_tensor, activation, gate_up, offset_device, source_device, EB_EXPERTS, pairs, 0u, shape.hidden,
					shape.up_rows, 0, 1u);
				LmSkinnyGroupedExperts<LmFp8>(down, down_tensor, intermediate, output, offset_device, source_device, EB_EXPERTS, pairs, 1u, shape.inter,
					shape.hidden, 0, 1u);
			}
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&grouped_ms, start, stop);
		}
		if ( cudaDeviceSynchronize() != cudaSuccess )
		{
			fprintf(stderr, "a launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
			exit(1);
		}
		printf("%6u %8llu %12.1f %12.1f %12.1f %12.1f %12.1f %9.2fx\n", rows, (unsigned long long)touched, 1000.0f * decode_ms / EB_REPEATS,
			1000.0f * chunked_ms / EB_REPEATS, 1000.0f * grouped_ms / EB_REPEATS, (double)(chunked_slices * slice) / 1048576.0,
			(double)(grouped_slices * slice) / 1048576.0, chunked_ms / grouped_ms);
		cudaFree(expert_device);
		cudaFree(packed_device);
		cudaFree(source_device);
		cudaFree(offset_device);
	}
	cudaFree(up);
	cudaFree(down);
	cudaFree(up_scale);
	cudaFree(down_scale);
	cudaFree(activation);
	cudaFree(gate_up);
	cudaFree(intermediate);
	cudaFree(output);
}

int main(void)
{
	static const EbShape shapes[] =
	{
		{ "glm-5.3-full", 6144u, 256u, 128u },
		{ "a.x-k2", 7168u, 256u, 128u },
	};
	for ( const EbShape &shape : shapes )
		EbRun(shape);
	return 0;
}
