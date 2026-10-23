#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include "runtime/launch.h"
#include "inference/kernels/skinny.cuh"

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)
#define CELL_GROUPS 384u
#define CELL_TOPK 6u
#define CELL_MAX_ROWS 33u

static const uint32_t kRowCounts[] = {2u, 3u, 5u, 6u, 8u, 16u, 17u, 33u};

struct CellShape
{
	uint32_t input, output;
};

static const CellShape kShapes[] = {{5120u, 288u}, {2304u, 320u}, {256u, 64u}};

struct Route
{
	uint32_t expert, token;
};

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

template<class T>
static std::vector<T> Host(const T *device, size_t count)
{
	std::vector<T> host(count);
	CUDA(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
	return host;
}

static std::vector<uint8_t> CellWeights(const CellShape &shape)
{
	const uint32_t tiles = shape.input / LM_SKINNY_CELL_TILE_K, cells = shape.output / LM_SKINNY_CELL_NEURONS;
	const uint64_t chunks = (uint64_t)CELL_GROUPS * tiles * cells;
	std::vector<uint8_t> weight(chunks * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES);
	for ( size_t index = 0u; index < weight.size(); ++index )
		weight[index] = (uint8_t)Mix(index ^ shape.input);
	for ( uint64_t chunk = 0u; chunk < chunks; ++chunk )
		for ( uint32_t neuron = 0u; neuron < LM_SKINNY_CELL_NEURONS; ++neuron )
			weight[(chunk * LM_SKINNY_CELL_ROWS + LM_SKINNY_CELL_NEURONS) * LM_SKINNY_CHUNK_BYTES + neuron] = (uint8_t)(120u + Mix(chunk * 31u + neuron) % 9u);
	return weight;
}

static std::vector<uint16_t> Bf16Rows(uint32_t rows, uint32_t width, uint64_t salt)
{
	std::vector<uint16_t> values((size_t)rows * width);
	for ( size_t index = 0u; index < values.size(); ++index )
		values[index] = (uint16_t)((0x3c00u + (Mix(index ^ salt) & 0x7ffu)) ^ ((Mix(index ^ salt ^ 1u) & 1u) << 15u));
	return values;
}

static std::vector<Route> Plan(const std::vector<std::vector<uint32_t>> &experts, const std::vector<uint32_t> &tokens, std::vector<uint32_t> *offsets)
{
	std::vector<Route> routes;
	for ( uint32_t token : tokens )
		for ( uint32_t expert : experts[token] )
			routes.push_back({expert, token});
	std::stable_sort(routes.begin(), routes.end(), [](const Route &a, const Route &b) { return a.expert < b.expert; });
	offsets->assign(CELL_GROUPS + 1u, 0u);
	for ( const Route &route : routes )
		(*offsets)[route.expert + 1u]++;
	for ( uint32_t group = 0u; group < CELL_GROUPS; ++group )
		(*offsets)[group + 1u] += (*offsets)[group];
	return routes;
}

static std::vector<uint16_t> Launch(const uint8_t *weight, const std::vector<uint16_t> &tokens_bf16, const std::vector<Route> &routes,
	const std::vector<uint32_t> &offsets, uint32_t group_rows_max, uint32_t packed, const CellShape &shape)
{
	std::vector<uint32_t> source;
	std::vector<uint16_t> activation;
	for ( const Route &route : routes )
		source.push_back(route.token);
	if ( packed != 0u )
		for ( const Route &route : routes )
			activation.insert(activation.end(), tokens_bf16.begin() + (size_t)route.token * shape.input,
				tokens_bf16.begin() + (size_t)(route.token + 1u) * shape.input);
	else
		activation = tokens_bf16;
	uint16_t *device_activation = Device(activation), *output;
	uint32_t *device_offsets = Device(offsets), *device_source = Device(source);
	CUDA(cudaMalloc((void **)&output, routes.size() * shape.output * sizeof(uint16_t)));
	CUDA(cudaMemset(output, 0xff, routes.size() * shape.output * sizeof(uint16_t)));
	REQUIRE(LmSkinnyCellExperts(weight, device_activation, output, device_offsets, device_source, CELL_GROUPS, (uint32_t)routes.size(), group_rows_max,
		packed, shape.input, shape.output, LM_SKINNY_CELL_TILE_K, 0) == LM_LAUNCH_OK);
	CUDA(cudaDeviceSynchronize());
	std::vector<uint16_t> result = Host(output, routes.size() * shape.output);
	CUDA(cudaFree(device_activation));
	CUDA(cudaFree(device_offsets));
	CUDA(cudaFree(device_source));
	CUDA(cudaFree(output));
	return result;
}

static void ShapeCase(const CellShape &shape, uint64_t *compared, uint64_t *shared_compared)
{
	uint8_t *weight = Device(CellWeights(shape));
	const std::vector<uint16_t> tokens_bf16 = Bf16Rows(CELL_MAX_ROWS, shape.input, 0xa11cu + shape.input);
	for ( uint32_t trial = 0u; trial < 4u; ++trial )
		for ( uint32_t rows : kRowCounts )
			for ( uint32_t packed = 0u; packed < 2u; ++packed )
			{
				const uint32_t pool = trial % 2u == 0u ? CELL_GROUPS : 2u * CELL_TOPK;
				std::vector<std::vector<uint32_t>> experts(rows);
				std::vector<uint32_t> all, offsets;
				for ( uint32_t token = 0u; token < rows; ++token )
				{
					uint64_t draw = Mix(((uint64_t)trial << 40u) ^ ((uint64_t)rows << 20u) ^ ((uint64_t)token << 4u) ^ packed);
					while ( experts[token].size() < CELL_TOPK )
					{
						const uint32_t expert = (uint32_t)(draw % pool);
						draw = Mix(draw);
						if ( std::find(experts[token].begin(), experts[token].end(), expert) == experts[token].end() )
							experts[token].push_back(expert);
					}
					all.push_back(token);
				}
				const std::vector<Route> frame_routes = Plan(experts, all, &offsets);
				const std::vector<uint16_t> frame = Launch(weight, tokens_bf16, frame_routes, offsets, rows, packed, shape);
				std::vector<uint32_t> shared(CELL_GROUPS, 0u);
				for ( const Route &route : frame_routes )
					shared[route.expert]++;
				for ( uint32_t token = 0u; token < rows; ++token )
				{
					std::vector<uint32_t> one_offsets;
					const std::vector<Route> one_routes = Plan(experts, std::vector<uint32_t>(1u, token), &one_offsets);
					const std::vector<uint16_t> one = Launch(weight, tokens_bf16, one_routes, one_offsets, 1u, packed, shape);
					for ( size_t index = 0u; index < one_routes.size(); ++index )
					{
						size_t at = 0u;
						while ( frame_routes[at].expert != one_routes[index].expert || frame_routes[at].token != token )
							at++;
						if ( memcmp(&one[index * shape.output], &frame[at * shape.output], shape.output * sizeof(uint16_t)) != 0 )
						{
							fprintf(stderr, "FAIL skinny cell rows: K %u N %u, %u-row frame (%s), token %u expert %u (%u frame rows on it) differs from its one-row launch\n",
								shape.input, shape.output, rows, packed != 0u ? "packed" : "source tokens", token, one_routes[index].expert,
								shared[one_routes[index].expert]);
							exit(1);
						}
						(*compared)++;
						*shared_compared += shared[one_routes[index].expert] > 1u ? 1u : 0u;
					}
				}
			}
	CUDA(cudaFree(weight));
}

int main(void)
{
	uint64_t compared = 0u, shared_compared = 0u;
	for ( const CellShape &shape : kShapes )
		ShapeCase(shape, &compared, &shared_compared);
	REQUIRE(shared_compared != 0u && shared_compared < compared);
	printf("every routed row of a 2-33 row frame equals its one-row launch bit for bit: %llu routes, %llu on experts the frame shares\n",
		(unsigned long long)compared, (unsigned long long)shared_compared);
	printf("PASS skinny cell rows\n");
	return 0;
}
