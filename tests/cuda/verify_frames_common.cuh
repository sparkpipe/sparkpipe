#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "sparkpipe/spark_stage_runner_model.h"

static int vf_failures = 0;

static void Require(int condition, const char *model, const char *what, uint32_t a, uint32_t b, uint32_t c)
{
	if ( condition )
		return;
	printf("FAIL %s: %s (%u %u %u)\n", model, what, a, b, c);
	vf_failures++;
}

static uint32_t Mix(uint32_t value)
{
	value ^= value >> 16u;
	value *= 0x7feb352du;
	value ^= value >> 15u;
	value *= 0x846ca68bu;
	return value ^ (value >> 16u);
}

static float Uniform(uint32_t seed, uint32_t index)
{
	return (float)(Mix(seed * 0x9e3779b9u + index) >> 8u) / 8388608.0f - 1.0f;
}

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static uint32_t vf_seed = 1u;

template<class T>
static const T *Tensor(uint64_t count, float scale, float offset)
{
	std::vector<T> host(count);
	void *device = 0;
	const uint32_t seed = vf_seed++;
	for ( uint64_t index = 0u; index < count; ++index )
	{
		const float value = offset + scale * Uniform(seed, (uint32_t)index);
		if constexpr ( sizeof(T) == sizeof(uint16_t) )
			host[index] = Bf16(value);
		else
			host[index] = value;
	}
	if ( cudaMalloc(&device, count * sizeof(T)) != cudaSuccess || cudaMemcpy(device, host.data(), count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess )
	{
		printf("FAIL weight allocation of %llu elements\n", (unsigned long long)count);
		exit(1);
	}
	return (const T *)device;
}

static const uint16_t *Matrix(uint64_t rows, uint64_t columns)
{
	return Tensor<uint16_t>(rows * columns, 1.7f / sqrtf((float)columns), 0.0f);
}

static const float *Norm(uint64_t count)
{
	return Tensor<float>(count, 0.2f, 1.0f);
}

typedef struct VfBus
{
	uint32_t hidden;
} VfBus;

static SparkStatus VfRound(void *context, void *stream, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements, const void *local, void *full)
{
	const VfBus *bus = (const VfBus *)context;
	const uint64_t elements = (uint64_t)rows * (row_elements != 0u ? row_elements : bus->hidden);
	(void)operation;
	if ( stage != SPARK_STAGE_RUNNER_ROUND_ALL )
		return SPARK_STATUS_UNSUPPORTED;
	if ( local == full )
		return SPARK_STATUS_OK;
	return cudaMemcpyAsync(full, local, elements * sizeof(uint16_t), cudaMemcpyDeviceToDevice, (cudaStream_t)stream) == cudaSuccess ? SPARK_STATUS_OK :
		SPARK_STATUS_IO_ERROR;
}

typedef struct VfRow
{
	uint32_t sequence;
	uint32_t position;
	uint32_t variant;
} VfRow;

static uint32_t FirstDifference(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, uint32_t width)
{
	for ( uint64_t index = 0u; index < a.size() && index < b.size(); ++index )
		if ( a[index] != b[index] )
			return (uint32_t)(index / width);
	return a.size() == b.size() ? UINT32_MAX : (uint32_t)(a.size() / width);
}

static uint32_t Finite(const std::vector<uint16_t> &values)
{
	uint32_t distinct = 0u;
	for ( uint64_t index = 0u; index < values.size(); ++index )
	{
		if ( (values[index] & 0x7f80u) == 0x7f80u )
			return 0u;
		distinct |= values[index] != values[0] ? 1u : 0u;
	}
	return distinct;
}

static void Kept(uint32_t rows, std::vector<uint32_t> *kept)
{
	kept->clear();
	if ( rows <= 7u )
	{
		for ( uint32_t k = 0u; k <= rows; ++k )
			kept->push_back(k);
		return;
	}
	kept->push_back(0u);
	kept->push_back(1u);
	kept->push_back(rows / 2u);
	kept->push_back(rows - 1u);
	kept->push_back(rows);
}

static uint8_t E4m3(float value)
{
	static float decoded[0x7fu];
	static uint32_t ready = 0u;
	const float magnitude = fabsf(value);
	uint32_t low = 0u, high = 0x7eu;
	if ( ready == 0u )
	{
		for ( uint32_t code = 0u; code < 0x7fu; ++code )
			decoded[code] = (code >> 3u) == 0u ? ldexpf((float)(code & 7u) / 8.0f, -6) : ldexpf(1.0f + (float)(code & 7u) / 8.0f, (int)(code >> 3u) - 7);
		ready = 1u;
	}
	while ( low < high )
	{
		const uint32_t middle = (low + high) / 2u;
		if ( decoded[middle] < magnitude )
			low = middle + 1u;
		else
			high = middle;
	}
	if ( low > 0u && magnitude - decoded[low - 1u] <= decoded[low] - magnitude )
		low--;
	return (uint8_t)(low | (value < 0.0f ? 0x80u : 0u));
}

static uint8_t *Device(const void *host, uint64_t bytes)
{
	void *device = 0;
	if ( cudaMalloc(&device, bytes) != cudaSuccess || cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess )
	{
		printf("FAIL weight allocation of %llu bytes\n", (unsigned long long)bytes);
		exit(1);
	}
	return (uint8_t *)device;
}
