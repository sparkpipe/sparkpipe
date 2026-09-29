#pragma once




#include "inference/kernels/mma.cuh"
#include "runtime/launch.h"
#include <stdint.h>

/* ROUTE ROW INDIRECTION CONSUMER CONTRACT */
static __device__ __forceinline__ uint32_t LmRouteSourceRow(const uint32_t *__restrict__ route_source_token, uint32_t packed_row)
{
	return(route_source_token[packed_row]);
}

#define LM_ROUTE_SCAN_PARTS 3u

template<uint32_t THREADS>
static __device__ __forceinline__ void LmRouteScanBlock(uint32_t (*value)[LM_ROUTE_SCAN_PARTS])
{
	__shared__ uint32_t scan[LM_ROUTE_SCAN_PARTS][THREADS];
	uint32_t part, step, add[LM_ROUTE_SCAN_PARTS];
	for (part = 0u; part < LM_ROUTE_SCAN_PARTS; part++)
		scan[part][threadIdx.x] = (*value)[part];
	__syncthreads();
	for (step = 1u; step < THREADS; step <<= 1u)
	{
		for (part = 0u; part < LM_ROUTE_SCAN_PARTS; part++)
			add[part] = threadIdx.x >= step ? scan[part][threadIdx.x - step] : 0u;
		__syncthreads();
		for (part = 0u; part < LM_ROUTE_SCAN_PARTS; part++)
			scan[part][threadIdx.x] += add[part];
		__syncthreads();
	}
	for (part = 0u; part < LM_ROUTE_SCAN_PARTS; part++)
		(*value)[part] = scan[part][threadIdx.x];
}

template<uint32_t THREADS, uint32_t EXPERTS>
static __device__ __forceinline__ void LmRouteBuildPrefix(uint32_t *count, uint32_t *__restrict__ group_row_offset, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	constexpr uint32_t per_thread = (EXPERTS + THREADS - 1u) / THREADS;
	const uint32_t first = threadIdx.x * per_thread;
	uint32_t held[per_thread], value[LM_ROUTE_SCAN_PARTS], running[LM_ROUTE_SCAN_PARTS], index, expert, tiles;
	value[0] = value[1] = value[2] = 0u;
	#pragma unroll
	for (index = 0u; index < per_thread; index++)
	{
		expert = first + index;
		held[index] = expert < EXPERTS ? count[expert] : 0u;
		tiles = (held[index] + tile_m - 1u) / tile_m;
		value[0] += held[index];
		value[1] += tiles * neuron_tiles_up;
		value[2] += tiles * neuron_tiles_down;
	}
	running[0] = value[0];
	running[1] = value[1];
	running[2] = value[2];
	LmRouteScanBlock<THREADS>(&value);
	running[0] = value[0] - running[0];
	running[1] = value[1] - running[1];
	running[2] = value[2] - running[2];
	__syncthreads();
	#pragma unroll
	for (index = 0u; index < per_thread; index++)
	{
		expert = first + index;
		if ( expert >= EXPERTS )
			break;
		group_row_offset[expert] = running[0];
		count[expert] = running[0];
		if ( tile_prefix_up != 0 && tile_prefix_down != 0 )
		{
			tile_prefix_up[expert] = running[1];
			tile_prefix_down[expert] = running[2];
		}
		tiles = (held[index] + tile_m - 1u) / tile_m;
		running[0] += held[index];
		running[1] += tiles * neuron_tiles_up;
		running[2] += tiles * neuron_tiles_down;
		if ( expert + 1u == EXPERTS )
		{
			group_row_offset[EXPERTS] = running[0];
			if ( tile_prefix_up != 0 && tile_prefix_down != 0 )
			{
				tile_prefix_up[EXPERTS] = running[1];
				tile_prefix_down[EXPERTS] = running[2];
			}
		}
	}
}

template<uint32_t THREADS, uint32_t EXPERTS>
__global__ __launch_bounds__(THREADS, 1)
void LmRouteBuildKernel(const uint32_t *__restrict__ route_expert, uint32_t routes, uint32_t top_k, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	__shared__ uint32_t count[EXPERTS];
	uint32_t index,expert,packed;
	for (index = threadIdx.x; index < EXPERTS; index += THREADS)
		count[index] = 0u;
	__syncthreads();
	for (index = threadIdx.x; index < routes; index += THREADS)
		atomicAdd(&count[route_expert[index]],1u);
	__syncthreads();
	LmRouteBuildPrefix<THREADS,EXPERTS>(count,group_row_offset,tile_m,neuron_tiles_up,tile_prefix_up,neuron_tiles_down,tile_prefix_down);
	__syncthreads();
	for (index = threadIdx.x; index < routes; index += THREADS)
	{
		expert = route_expert[index];
		packed = atomicAdd(&count[expert],1u);
		route_packed_row[index] = packed;
		route_source_token[packed] = index / top_k;
	}
}


template<uint32_t THREADS, uint32_t EXPERTS>
static int32_t LmRouteBuild(
	const uint32_t *route_expert,
	uint32_t rows,
	uint32_t packed_rows,
	uint32_t top_k,
	uint32_t *group_row_offset,
	uint32_t *route_packed_row,
	uint32_t *route_source_token,
	uint32_t output_dimension_up,
	uint32_t output_dimension_down,
	uint32_t tile_n_up,
	uint32_t tile_n_down,
	uint32_t *tile_prefix_up,
	uint32_t *tile_prefix_down,
	cudaStream_t stream)
{
	uint32_t expected_packed_rows;
	uint32_t tile_m;
	uint32_t neuron_tiles_up;
	uint32_t neuron_tiles_down;

	if ( route_expert == 0 || rows == 0u || top_k == 0u ||
		rows > UINT32_MAX / top_k || group_row_offset == 0 ||
		route_packed_row == 0 || route_source_token == 0 ||
		output_dimension_up == 0u || output_dimension_down == 0u ||
		tile_n_up == 0u || tile_n_down == 0u ||
		tile_prefix_up == 0 || tile_prefix_down == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	expected_packed_rows = rows * top_k;
	if ( packed_rows != expected_packed_rows )
		return(LM_LAUNCH_ERR_SHAPE);
	tile_m = LmLaunchGroupedTileM(rows,top_k,EXPERTS);
	neuron_tiles_up = (output_dimension_up + tile_n_up - 1u) / tile_n_up;
	neuron_tiles_down = (output_dimension_down + tile_n_down - 1u) / tile_n_down;
	LM_LAUNCH((LmRouteBuildKernel<THREADS,EXPERTS>), 1u, THREADS, 0, stream,
		route_expert,packed_rows,top_k,group_row_offset,route_packed_row,
		route_source_token,tile_m,neuron_tiles_up,tile_prefix_up,
		neuron_tiles_down,tile_prefix_down);
	return(cudaPeekAtLastError() == cudaSuccess
		? LM_LAUNCH_OK
		: LM_LAUNCH_ERR_LAUNCH);
}

template<uint32_t THREADS, uint32_t EXPERTS>
static int32_t LmRouteBuild(
	const uint32_t *route_expert,
	uint32_t rows,
	uint32_t packed_rows,
	uint32_t top_k,
	uint32_t *group_row_offset,
	uint32_t *route_packed_row,
	uint32_t *route_source_token,
	uint32_t output_dimension_up,
	uint32_t output_dimension_down,
	uint32_t tile_n,
	uint32_t *tile_prefix_up,
	uint32_t *tile_prefix_down,
	cudaStream_t stream)
{
	return(LmRouteBuild<THREADS,EXPERTS>(route_expert,rows,packed_rows,top_k,
		group_row_offset,route_packed_row,route_source_token,
		output_dimension_up,output_dimension_down,tile_n,tile_n,
		tile_prefix_up,tile_prefix_down,stream));
}
