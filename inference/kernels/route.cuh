#pragma once




#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/mma.cuh"
#include "runtime/launch.h"
#include "inference/kernels/topk_warp.cuh"
#include <stdint.h>
#include "sparkpipe/spark_gpu.h"

static __device__ __forceinline__ uint32_t LmRouteSourceRow(const uint32_t *__restrict__ route_source_token, uint32_t packed_row)
{
	return(route_source_token[packed_row]);
}

#define LM_ROUTE_SCAN_PARTS 3u
#define LM_ROUTE_STAGED_MAX 4096u

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
static __device__ __forceinline__ void LmRouteBuildBlock(const uint32_t *route_expert, uint32_t routes, uint32_t top_k, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	constexpr uint32_t LANES = THREADS < LM_WARP_LANES ? THREADS : LM_WARP_LANES, WARPS = THREADS / LANES;
	static_assert(THREADS % LANES == 0u, "whole warps rank the routes of a chunk");
	__shared__ uint32_t count[EXPERTS];
	__shared__ uint8_t warp_count[WARPS][EXPERTS];
	__shared__ uint32_t staged[LM_ROUTE_STAGED_MAX];
	const uint32_t *source = routes <= LM_ROUTE_STAGED_MAX ? staged : route_expert;
	const uint32_t lane = threadIdx.x % LANES,warp = threadIdx.x / LANES;
	uint32_t index,expert,packed,earlier,same;
	for (index = threadIdx.x; index < EXPERTS; index += THREADS)
		count[index] = 0u;
	for (index = threadIdx.x; index < WARPS * EXPERTS; index += THREADS)
		warp_count[index / EXPERTS][index % EXPERTS] = 0u;
	for (index = threadIdx.x; index < routes && index < LM_ROUTE_STAGED_MAX; index += THREADS)
		staged[index] = route_expert[index];
	__syncthreads();
	for (index = threadIdx.x - lane; index < routes; index += THREADS)
	{
		expert = index + lane < routes ? source[index + lane] : EXPERTS;
		same = __match_any_sync(0xffffffffu,expert);
		if ( expert < EXPERTS && lane == (uint32_t)(__ffs((int)same) - 1) )
			atomicAdd(&count[expert],(uint32_t)__popc(same));
	}
	__syncthreads();
	LmRouteBuildPrefix<THREADS,EXPERTS>(count,group_row_offset,tile_m,neuron_tiles_up,tile_prefix_up,neuron_tiles_down,tile_prefix_down);
	__syncthreads();
	uint32_t ahead = threadIdx.x < routes ? source[threadIdx.x] : EXPERTS;
	for (index = 0u; index < routes; index += THREADS)
	{
		const uint32_t route = index + threadIdx.x;
		expert = ahead;
		ahead = route + THREADS < routes ? source[route + THREADS] : EXPERTS;
		same = __match_any_sync(0xffffffffu,expert);
		if ( expert < EXPERTS && lane == (uint32_t)(__ffs((int)same) - 1) )
			warp_count[warp][expert] = (uint8_t)__popc(same);
		__syncthreads();
		if ( route < routes )
		{
			packed = count[expert] + (uint32_t)__popc(same & ((1u << lane) - 1u));
			for (earlier = 0u; earlier < warp; earlier++)
				packed += warp_count[earlier][expert];
			route_packed_row[route] = packed;
			route_source_token[packed] = route / top_k;
		}
		__syncthreads();
		for (expert = threadIdx.x; expert < EXPERTS; expert += THREADS)
		{
			packed = 0u;
			for (earlier = 0u; earlier < WARPS; earlier++)
			{
				packed += warp_count[earlier][expert];
				warp_count[earlier][expert] = 0u;
			}
			count[expert] += packed;
		}
		__syncthreads();
	}
}

template<uint32_t THREADS, uint32_t EXPERTS>
static __device__ __forceinline__ void LmRouteBuildSmallBody(const uint32_t *sel, uint32_t routes, uint32_t top_k, uint32_t *__restrict__ group_row_offset,
	uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up,
	uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	if ( threadIdx.x < routes )
	{
		const uint32_t mine = sel[threadIdx.x];
		uint32_t packed = 0u;
		for ( uint32_t j = 0u; j < routes; ++j )
			if ( sel[j] < mine || (sel[j] == mine && j < threadIdx.x) )
				packed++;
		route_packed_row[threadIdx.x] = packed;
		route_source_token[packed] = threadIdx.x / top_k;
	}
	for ( uint32_t expert = threadIdx.x; expert <= EXPERTS; expert += THREADS )
	{
		uint32_t before = 0u, tiles = 0u;
		for ( uint32_t j = 0u; j < routes; ++j )
		{
			const uint32_t target = sel[j];
			if ( target >= expert )
				continue;
			before++;
			uint32_t first = 1u, count = 1u;
			for ( uint32_t k = 0u; k < j; ++k )
				if ( sel[k] == target )
					first = 0u;
			if ( first != 0u )
			{
				for ( uint32_t k = j + 1u; k < routes; ++k )
					if ( sel[k] == target )
						count++;
				tiles += (count + tile_m - 1u) / tile_m;
			}
		}
		group_row_offset[expert] = before;
		if ( tile_prefix_up != 0 && tile_prefix_down != 0 )
		{
			tile_prefix_up[expert] = tiles * neuron_tiles_up;
			tile_prefix_down[expert] = tiles * neuron_tiles_down;
		}
	}
}

template<uint32_t THREADS, uint32_t EXPERTS>
__global__ __launch_bounds__(THREADS, 1)
void LmRouteBuildSmallKernel(const uint32_t *__restrict__ route_expert, uint32_t routes, uint32_t top_k, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	__shared__ uint32_t sel[32];
	LmDependentWait();
	if ( threadIdx.x < routes )
		sel[threadIdx.x] = route_expert[threadIdx.x];
	__syncthreads();
	LmRouteBuildSmallBody<THREADS, EXPERTS>(sel, routes, top_k, group_row_offset, route_packed_row, route_source_token, tile_m, neuron_tiles_up,
		tile_prefix_up, neuron_tiles_down, tile_prefix_down);
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t EXPERTS, uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkRouteBuildSmallKernel(const float *__restrict__ scores, const float *__restrict__ selection_bias, const uint16_t *__restrict__ logits_bf16,
	float mixture_scale, uint32_t *__restrict__ fault, uint32_t rows, uint32_t *__restrict__ route_expert, float *__restrict__ route_weight,
	uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token, uint32_t tile_m,
	uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	__shared__ uint32_t sel[32];
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES, routes = rows * K;
	LmDependentWait();
	if ( warp < rows )
		LmTopkWarpRow<K, RENORMALISE, SCORE_TRANSFORM>(scores, EXPERTS, route_expert, route_weight, selection_bias, logits_bf16, mixture_scale, fault, warp, lane);
	__syncthreads();
	if ( threadIdx.x < routes )
		sel[threadIdx.x] = route_expert[threadIdx.x];
	__syncthreads();
	LmRouteBuildSmallBody<THREADS, EXPERTS>(sel, routes, K, group_row_offset, route_packed_row, route_source_token, tile_m, neuron_tiles_up,
		tile_prefix_up, neuron_tiles_down, tile_prefix_down);
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t EXPERTS>
__global__ __launch_bounds__(THREADS, 1)
void LmRouteBuildKernel(const uint32_t *__restrict__ route_expert, uint32_t routes, uint32_t top_k, uint32_t *__restrict__ group_row_offset, uint32_t *__restrict__ route_packed_row, uint32_t *__restrict__ route_source_token, uint32_t tile_m, uint32_t neuron_tiles_up, uint32_t *__restrict__ tile_prefix_up, uint32_t neuron_tiles_down, uint32_t *__restrict__ tile_prefix_down)
{
	LmDependentRelease();
	LmRouteBuildBlock<THREADS,EXPERTS>(route_expert,routes,top_k,group_row_offset,route_packed_row,route_source_token,tile_m,neuron_tiles_up,tile_prefix_up,
		neuron_tiles_down,tile_prefix_down);
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
	SparkGpuStream stream)
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
	if ( packed_rows <= 32u )
		LM_LAUNCH_DEPENDENT((LmRouteBuildSmallKernel<THREADS,EXPERTS>), 1u, THREADS, 0, stream,
			route_expert,packed_rows,top_k,group_row_offset,route_packed_row,
			route_source_token,tile_m,neuron_tiles_up,tile_prefix_up,
			neuron_tiles_down,tile_prefix_down);
	else
		LM_LAUNCH((LmRouteBuildKernel<THREADS,EXPERTS>), 1u, THREADS, 0, stream,
			route_expert,packed_rows,top_k,group_row_offset,route_packed_row,
			route_source_token,tile_m,neuron_tiles_up,tile_prefix_up,
			neuron_tiles_down,tile_prefix_down);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS
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
	SparkGpuStream stream)
{
	return(LmRouteBuild<THREADS,EXPERTS>(route_expert,rows,packed_rows,top_k,
		group_row_offset,route_packed_row,route_source_token,
		output_dimension_up,output_dimension_down,tile_n,tile_n,
		tile_prefix_up,tile_prefix_down,stream));
}

template<uint32_t THREADS, uint32_t EXPERTS, uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM>
static int32_t LmTopkRouteBuild(
	const float *scores,
	const float *selection_bias,
	const uint16_t *logits_bf16,
	float mixture_scale,
	uint32_t *fault,
	uint32_t rows,
	uint32_t packed_rows,
	uint32_t *route_expert,
	float *route_weight,
	uint32_t *group_row_offset,
	uint32_t *route_packed_row,
	uint32_t *route_source_token,
	uint32_t output_dimension_up,
	uint32_t output_dimension_down,
	uint32_t tile_n,
	uint32_t *tile_prefix_up,
	uint32_t *tile_prefix_down,
	SparkGpuStream stream)
{
	if ( fault == 0 || EXPERTS < K )
		return(LM_LAUNCH_ERR_SHAPE);
	if constexpr ( EXPERTS <= LM_TOPK_WARP_PER_LANE * LM_WARP_LANES )
	{
		if ( rows > 0u && packed_rows == rows * K && packed_rows <= 32u && rows <= THREADS / LM_WARP_LANES &&
			scores != 0 && route_expert != 0 && group_row_offset != 0 && route_packed_row != 0 && route_source_token != 0 &&
			output_dimension_up != 0u && output_dimension_down != 0u && tile_n != 0u && tile_prefix_up != 0 && tile_prefix_down != 0 )
		{
			const uint32_t tile_m = LmLaunchGroupedTileM(rows, K, EXPERTS);
			const uint32_t n_up = (output_dimension_up + tile_n - 1u) / tile_n;
			const uint32_t n_down = (output_dimension_down + tile_n - 1u) / tile_n;
			LM_LAUNCH_DEPENDENT((LmTopkRouteBuildSmallKernel<THREADS, EXPERTS, K, RENORMALISE, SCORE_TRANSFORM>), 1u, THREADS, 0, stream,
				scores, selection_bias, logits_bf16, mixture_scale, fault, rows, route_expert, route_weight,
				group_row_offset, route_packed_row, route_source_token, tile_m, n_up, tile_prefix_up, n_down, tile_prefix_down);
			return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
		}
	}
	if ( LmTopkRouteLaunch<THREADS, K, RENORMALISE, SCORE_TRANSFORM>(rows, scores, EXPERTS, route_expert, route_weight, selection_bias, logits_bf16,
			mixture_scale, fault, stream) != SPARK_GPU_SUCCESS )
		return(LM_LAUNCH_ERR_LAUNCH);
	return(LmRouteBuild<THREADS, EXPERTS>(route_expert, rows, packed_rows, K, group_row_offset, route_packed_row, route_source_token,
		output_dimension_up, output_dimension_down, tile_n, tile_prefix_up, tile_prefix_down, stream));
}

static __global__ void LmDenseRowOffsetsKernel(uint32_t *__restrict__ offsets, uint32_t rows)
{
	if ( threadIdx.x == 0u )
	{
		offsets[0] = 0u;
		offsets[1] = rows;
	}
}
