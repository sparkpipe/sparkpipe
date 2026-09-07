#pragma once




#include "inference/kernels/mma.cuh"
#include "runtime/launch.h"
#include <stdint.h>

static __device__ __forceinline__ uint32_t LmRouteSourceRow(const uint32_t *__restrict__ route_source_token, uint32_t packed_row)
{
	return(route_source_token[packed_row]);
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
	if ( threadIdx.x == 0u )
	{
		uint32_t total = 0u,held;
		for (index = 0u; index < EXPERTS; ++index)
		{
			held = count[index];
			group_row_offset[index] = total;
			count[index] = total;
			total += held;
		}
		group_row_offset[EXPERTS] = total;
		if ( tile_prefix_up != 0 && tile_prefix_down != 0 )
		{
			uint32_t up = 0u,down = 0u,rows,row_tiles;
			for (index = 0u; index < EXPERTS; ++index)
			{
				rows = group_row_offset[index + 1u] - group_row_offset[index];
				row_tiles = (rows + tile_m - 1u) / tile_m;
				tile_prefix_up[index] = up;
				tile_prefix_down[index] = down;
				up += row_tiles * neuron_tiles_up;
				down += row_tiles * neuron_tiles_down;
			}
			tile_prefix_up[EXPERTS] = up;
			tile_prefix_down[EXPERTS] = down;
		}
	}
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
