#include "inference/kernels/skinny.cuh"
#include "inference/kernels/weight_codec.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

#define EG_TOP_K_MAX 10u
#define EG_MAX_ROWS 512u

typedef struct EgCase
{
	const char *name;
	uint32_t experts;
	uint32_t local;
	uint32_t top_k;
	uint32_t input;
	uint32_t output;
	uint32_t packed;
} EgCase;

__global__ static void EgFillKernel(uint8_t *bytes, uint64_t count, uint32_t seed, uint32_t kind)
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
		else
			bytes[index] = (uint8_t)value;
	}
}

static uint8_t *EgBytes(uint64_t count, uint32_t kind)
{
	uint8_t *device = 0;
	if ( cudaMalloc((void **)&device, count) != cudaSuccess )
	{
		printf("FAIL allocation of %llu bytes\n", (unsigned long long)count);
		exit(1);
	}
	EgFillKernel<<<1024u, 256u>>>(device, count, vf_seed++ * 2654435761u, kind);
	return device;
}

__global__ static void EgBf16Kernel(uint16_t *values, uint64_t count, uint32_t seed, float scale)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < count; index += (uint64_t)gridDim.x * blockDim.x )
	{
		uint32_t value = (uint32_t)index * 0x9e3779b9u ^ seed;
		value ^= value >> 16u;
		value *= 0x7feb352du;
		value ^= value >> 15u;
		const float uniform = (float)(value >> 8u) / 8388608.0f - 1.0f;
		values[index] = LmFloatToBf16(uniform * scale);
	}
}

static uint16_t *EgBf16(uint64_t count, float scale)
{
	uint16_t *device = 0;
	if ( cudaMalloc((void **)&device, count * sizeof(uint16_t)) != cudaSuccess )
	{
		printf("FAIL allocation of %llu values\n", (unsigned long long)count);
		exit(1);
	}
	EgBf16Kernel<<<1024u, 256u>>>(device, count, vf_seed++ * 2246822519u, scale);
	return device;
}

typedef struct EgRoutes
{
	std::vector<uint32_t> expert, packed_row, source, offset;
	uint32_t local_pairs;
} EgRoutes;

static void EgRoute(const EgCase &c, uint32_t rows, uint32_t skew, uint32_t seed, EgRoutes *routes)
{
	const uint32_t pairs = rows * c.top_k;
	std::vector<uint32_t> count(c.local + 1u, 0u), cursor(c.local + 2u, 0u);
	routes->expert.assign(pairs, 0u);
	routes->packed_row.assign(pairs, 0u);
	routes->source.assign(pairs, 0u);
	routes->offset.assign(c.local + 2u, 0u);
	for ( uint32_t row = 0u; row < rows; ++row )
		for ( uint32_t k = 0u; k < c.top_k; ++k )
		{
			uint32_t expert, attempt = 0u, taken;
			do
			{
				const uint32_t draw = Mix(seed + row * 131u + k * 7u + attempt++ * 977u);
				expert = skew != 0u && draw % 4u != 0u ? (draw >> 4u) % (c.top_k + 2u) : (draw >> 4u) % c.experts;
				taken = 0u;
				for ( uint32_t previous = 0u; previous < k; ++previous )
					taken |= routes->expert[row * c.top_k + previous] == (expert < c.local ? expert : c.local + expert) ? 1u : 0u;
			}
			while ( taken != 0u );
			routes->expert[row * c.top_k + k] = expert < c.local ? expert : c.local + expert;
		}
	for ( uint32_t pair = 0u; pair < pairs; ++pair )
	{
		const uint32_t group = routes->expert[pair] < c.local ? routes->expert[pair] : c.local;
		count[group]++;
	}
	for ( uint32_t group = 0u; group <= c.local; ++group )
		routes->offset[group + 1u] = routes->offset[group] + count[group];
	for ( uint32_t group = 0u; group <= c.local; ++group )
		cursor[group] = routes->offset[group];
	for ( uint32_t pair = 0u; pair < pairs; ++pair )
	{
		const uint32_t group = routes->expert[pair] < c.local ? routes->expert[pair] : c.local, packed = cursor[group]++;
		routes->packed_row[pair] = packed;
		routes->source[packed] = pair / c.top_k;
	}
	for ( uint32_t pair = 0u; pair < pairs; ++pair )
		if ( routes->expert[pair] >= c.local )
			routes->expert[pair] = c.local;
	routes->local_pairs = routes->offset[c.local];
}

template<class T>
static T *EgUpload(const std::vector<T> &host)
{
	T *device = 0;
	if ( cudaMalloc((void **)&device, host.size() * sizeof(T) + 16u) != cudaSuccess ||
		cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess )
	{
		printf("FAIL upload\n");
		exit(1);
	}
	return device;
}

template<class Format>
static LmScaleTensor EgScale(const EgCase &c, uint8_t **storage)
{
	const uint64_t rows = (uint64_t)c.local * c.output;
	*storage = 0;
	if constexpr ( std::is_same<Format, LmFp8>::value )
	{
		std::vector<float> host(rows * ((c.input + 127u) / 128u));
		for ( uint64_t index = 0u; index < host.size(); ++index )
			host[index] = (0.75f + 0.25f * Uniform(vf_seed, (uint32_t)index)) / sqrtf((float)c.input);
		vf_seed++;
		*storage = (uint8_t *)EgUpload(host);
		return LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(*storage, c.local, c.output, c.input);
	}
	else if constexpr ( std::is_same<Format, LmMxfp8>::value )
	{
		*storage = EgBytes(rows * (c.input / 32u), 1u);
		return LmScaleTensorBuild(*storage, LM_SCALE_ENCODING_UE8M0, c.local, c.output, c.input, 1u, 32u);
	}
	else
		return LmScaleTensorNone();
}

template<class Format>
static void EgCheck(const EgCase &c)
{
	static const uint32_t frames[] = { 1u, 2u, 7u, 16u, 17u, 40u, 256u, EG_MAX_ROWS };
	const uint64_t weight_bytes = (uint64_t)c.local * c.output * c.input * Format::kStoredBits / 8u;
	uint8_t *scale_storage = 0;
	void *weight = std::is_same<Format, LmBf16Format>::value ? (void *)EgBf16(weight_bytes / 2u, 1.7f / sqrtf((float)c.input)) : (void *)EgBytes(weight_bytes, 0u);
	const LmScaleTensor scale = EgScale<Format>(c, &scale_storage);
	const uint32_t activation_rows = c.packed != 0u ? EG_MAX_ROWS * c.top_k : EG_MAX_ROWS;
	uint16_t *activation = EgBf16((uint64_t)activation_rows * c.input, 1.0f);
	uint16_t *grouped = 0, *reference = 0, *alone = 0, *alone_activation = 0;
	uint32_t compared = 0u, alone_compared = 0u;
	std::vector<uint16_t> host_grouped, host_reference, host_alone;
	if ( cudaMalloc((void **)&grouped, (uint64_t)EG_MAX_ROWS * c.top_k * c.output * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&reference, (uint64_t)EG_MAX_ROWS * c.top_k * c.output * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&alone, (uint64_t)c.top_k * c.output * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&alone_activation, (uint64_t)c.top_k * c.input * sizeof(uint16_t)) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess )
	{
		Require(0, c.name, "buffers allocate", 0u, 0u, 0u);
		return;
	}
	for ( uint32_t skew = 0u; skew < 2u; ++skew )
		for ( uint32_t rows : frames )
		{
			EgRoutes routes;
			const uint32_t pairs = rows * c.top_k;
			EgRoute(c, rows, skew, rows * 7919u + skew, &routes);
			uint32_t *expert = EgUpload(routes.expert), *packed_row = EgUpload(routes.packed_row), *source = EgUpload(routes.source);
			uint32_t *offset = EgUpload(routes.offset);
			uint32_t most = 0u;
			for ( uint32_t group = 0u; group < c.local; ++group )
				most = routes.offset[group + 1u] - routes.offset[group] > most ? routes.offset[group + 1u] - routes.offset[group] : most;
			Require(LmSkinnyGroupedExperts<Format>(weight, scale, activation, grouped, offset, source, c.local, pairs, c.packed, c.input, c.output, 0, 1u) ==
				LM_LAUNCH_OK && cudaDeviceSynchronize() == cudaSuccess, c.name, "the exact grouped launch takes any frame (rows, skew, most routes on one expert)",
				rows, skew, most);
			host_grouped.resize((uint64_t)pairs * c.output);
			cudaMemcpy(host_grouped.data(), grouped, host_grouped.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
			if ( c.local == c.experts )
			{
				for ( uint32_t first = 0u; first < rows; first += LM_SKINNY_ROWS )
				{
					const uint32_t count = rows - first < LM_SKINNY_ROWS ? rows - first : LM_SKINNY_ROWS;
					Require(LmSkinnyExperts<Format>(weight, scale, c.packed != 0u ? activation : activation + (uint64_t)first * c.input, reference,
						expert + (uint64_t)first * c.top_k, packed_row + (uint64_t)first * c.top_k, count * c.top_k, c.top_k, c.packed, c.input, c.output, 0) ==
						LM_LAUNCH_OK, c.name, "the per-route kernel runs", rows, first, 0u);
				}
				host_reference.resize((uint64_t)pairs * c.output);
				cudaDeviceSynchronize();
				cudaMemcpy(host_reference.data(), reference, host_reference.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
				for ( uint32_t pair = 0u; pair < pairs; ++pair )
				{
					const uint64_t row = routes.packed_row[pair];
					Require(memcmp(host_grouped.data() + row * c.output, host_reference.data() + row * c.output, c.output * sizeof(uint16_t)) == 0, c.name,
						"every route of the exact grouped launch equals the per-route kernel bit for bit (rows, skew, route)", rows, skew, pair);
					compared++;
				}
			}
			for ( uint32_t row = 0u; row < rows; ++row )
			{
				EgRoutes mine;
				std::vector<uint32_t> map;
				mine.offset.assign(c.local + 2u, 0u);
				for ( uint32_t k = 0u; k < c.top_k; ++k )
					if ( routes.expert[row * c.top_k + k] < c.local )
						mine.offset[routes.expert[row * c.top_k + k] + 1u]++;
				for ( uint32_t group = 0u; group <= c.local; ++group )
					mine.offset[group + 1u] += mine.offset[group];
				std::vector<uint32_t> cursor(mine.offset.begin(), mine.offset.end());
				map.assign(c.top_k, UINT32_MAX);
				mine.source.assign(c.top_k, 0u);
				for ( uint32_t k = 0u; k < c.top_k; ++k )
				{
					const uint32_t pair = row * c.top_k + k, group = routes.expert[pair];
					const uint32_t slot = group < c.local ? cursor[group]++ : cursor[c.local]++;
					map[slot] = pair;
					if ( c.packed != 0u )
						cudaMemcpy(alone_activation + (uint64_t)slot * c.input, activation + (uint64_t)routes.packed_row[pair] * c.input, c.input * sizeof(uint16_t),
							cudaMemcpyDeviceToDevice);
				}
				uint32_t *alone_offset = EgUpload(mine.offset), *alone_source = EgUpload(mine.source);
				Require(LmSkinnyGroupedExperts<Format>(weight, scale, c.packed != 0u ? alone_activation : activation + (uint64_t)row * c.input, alone, alone_offset,
					alone_source, c.local, c.top_k, c.packed, c.input, c.output, 0) == LM_LAUNCH_OK, c.name, "a one-row grouped launch runs", rows, row, 0u);
				host_alone.resize((uint64_t)c.top_k * c.output);
				cudaDeviceSynchronize();
				cudaMemcpy(host_alone.data(), alone, host_alone.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
				for ( uint32_t slot = 0u; slot < mine.offset[c.local]; ++slot )
				{
					const uint64_t frame_row = routes.packed_row[map[slot]];
					Require(memcmp(host_grouped.data() + frame_row * c.output, host_alone.data() + (uint64_t)slot * c.output, c.output * sizeof(uint16_t)) == 0,
						c.name, "every route of the exact grouped launch equals its row launched alone (rows, skew, row)", rows, skew, row);
					alone_compared++;
				}
				cudaFree(alone_offset);
				cudaFree(alone_source);
			}
			cudaFree(expert);
			cudaFree(packed_row);
			cudaFree(source);
			cudaFree(offset);
		}
	printf("%s %s: %u routes against the per-route kernel, %u against their row alone\n", vf_failures == 0 ? "ok" : "checked", c.name, compared, alone_compared);
	cudaFree(weight);
	cudaFree(scale_storage);
	cudaFree(activation);
	cudaFree(grouped);
	cudaFree(reference);
	cudaFree(alone);
	cudaFree(alone_activation);
}

int main(void)
{
	static const EgCase fp8[] =
	{
		{ "glm-5.3-full up", 256u, 256u, 8u, 6144u, 256u, 0u },
		{ "glm-5.3-full down", 256u, 256u, 8u, 128u, 6144u, 1u },
		{ "a.x-k2 up", 256u, 256u, 8u, 7168u, 256u, 0u },
		{ "a.x-k2 down", 256u, 256u, 8u, 128u, 7168u, 1u },
	};
	static const EgCase bf16[] =
	{
		{ "gemma-4-26b up", 128u, 8u, 8u, 2816u, 704u, 0u },
		{ "gemma-4-26b down", 128u, 8u, 8u, 704u, 2816u, 1u },
		{ "laguna up", 256u, 16u, 10u, 3072u, 1024u, 0u },
		{ "laguna down", 256u, 16u, 10u, 1024u, 3072u, 1u },
		{ "solar-open-2 up", 320u, 320u, 8u, 4096u, 80u, 0u },
		{ "solar-open-2 down", 320u, 320u, 8u, 80u, 4096u, 1u },
		{ "ling-3.0 up", 512u, 32u, 8u, 2560u, 768u, 0u },
		{ "ling-3.0 down", 512u, 32u, 8u, 768u, 2560u, 1u },
	};
	static const EgCase mxfp8[] =
	{
		{ "hy4 up", 256u, 16u, 8u, 6144u, 2048u, 0u },
		{ "hy4 down", 256u, 16u, 8u, 2048u, 6144u, 1u },
	};
	for ( const EgCase &c : fp8 )
		EgCheck<LmFp8>(c);
	for ( const EgCase &c : bf16 )
		EgCheck<LmBf16Format>(c);
	for ( const EgCase &c : mxfp8 )
		EgCheck<LmMxfp8>(c);
	if ( vf_failures != 0 )
	{
		printf("FAIL exact grouped experts: %d failures\n", vf_failures);
		return 1;
	}
	printf("PASS exact grouped experts: every route of a grouped launch of any frame equals the per-route kernel and its row launched alone, bit for bit\n");
	return 0;
}
