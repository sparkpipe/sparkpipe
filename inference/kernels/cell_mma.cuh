#pragma once

#include "inference/kernels/skinny.cuh"
#include "inference/kernels/mma.cuh"

#define LM_CELL_MMA_CELLS 2u
#define LM_CELL_MMA_FRAGS (LM_CELL_MMA_CELLS * 2u)
#define LM_CELL_MMA_WARPS 4u
#define LM_CELL_MMA_THREADS (LM_CELL_MMA_WARPS * LM_WARP_LANES)
#define LM_CELL_MMA_CHUNKS 8u
#define LM_CELL_MMA_ROWS (LM_CELL_MMA_CHUNKS * 16u)
#define LM_CELL_MMA_STAGES 4u
#define LM_CELL_MMA_ROW_SLOTS 8u
#define LM_CELL_MMA_SLOT_MIN_K_TILES 32u
#define LM_CELL_MMA_MAX_SEGMENTS 16u
#define LM_CELL_MMA_STAGE_K LM_SKINNY_CELL_TILE_K
#define LM_CELL_MMA_STAGE_PITCH LM_CELL_MMA_STAGE_K
#define LM_CELL_MMA_STAGE_ELEMENTS (LM_CELL_MMA_ROWS * LM_CELL_MMA_STAGE_PITCH)
#define LM_CELL_MMA_STAGE_VECTORS (LM_CELL_MMA_STAGE_K / 8u)
#define LM_CELL_MMA_STAGE_ROW_STEP (LM_CELL_MMA_THREADS / LM_CELL_MMA_STAGE_VECTORS)
#define LM_CELL_MMA_CELL_BYTES (LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES)
#define LM_CELL_MMA_CELL_VECTORS (LM_CELL_MMA_CELL_BYTES / 16u)
#define LM_CELL_MMA_BLOCK_CELLS (LM_CELL_MMA_WARPS * LM_CELL_MMA_CELLS)
#define LM_CELL_MMA_WEIGHT_TILE_VECTORS (LM_CELL_MMA_BLOCK_CELLS * LM_CELL_MMA_CELL_VECTORS)
#define LM_CELL_MMA_WEIGHT_TILE_BYTES (LM_CELL_MMA_WEIGHT_TILE_VECTORS * 16u)

static_assert(LM_CELL_MMA_ROWS % LM_CELL_MMA_STAGE_ROW_STEP == 0u, "one cooperative pass covers whole rows of the stage");
static_assert(LM_CELL_MMA_STAGE_PITCH * 2u * 2u == 128u, "two 64-byte rows fill the banks so fragment loads stay conflict free");
static_assert(LM_CELL_MMA_CELL_BYTES % 16u == 0u, "cells copy in 16-byte vectors");

typedef struct LmCellMmaSegments
{
	const uint8_t *weight[LM_CELL_MMA_MAX_SEGMENTS];
	uint64_t group_bytes;
	uint32_t tiles;
	uint32_t column_cells;
}
LmCellMmaSegments;

static __device__ __forceinline__ void LmCellMmaCopy(void *destination, const void *source, uint32_t bytes)
{
	const uint32_t shared = (uint32_t)__cvta_generic_to_shared(destination);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(shared), "l"(source), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void LmCellMmaCommit(void)
{
	asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<uint32_t PENDING>
static __device__ __forceinline__ void LmCellMmaWait(void)
{
	asm volatile("cp.async.wait_group %0;\n" :: "n"(PENDING) : "memory");
}

template<uint32_t COPIES>
static __device__ __forceinline__ void LmCellMmaStage(uint16_t *buffer, uint8_t *weights, const uint16_t *const source[COPIES], const uint16_t *fallback, const LmCellMmaSegments &segments, uint32_t column_segment, uint64_t cells_offset, uint64_t tile_stride, uint32_t t, uint32_t row, uint32_t vector, uint32_t weight_vectors)
{
	const uint8_t *cells = segments.weight[column_segment + t / segments.tiles] + cells_offset + (uint64_t)(t % segments.tiles) * tile_stride;
	#pragma unroll
	for ( uint32_t m = 0u; m < COPIES; m++ )
		LmCellMmaCopy(buffer + (row + m * LM_CELL_MMA_STAGE_ROW_STEP) * LM_CELL_MMA_STAGE_PITCH + vector * 8u,
			source[m] != 0 ? source[m] + t * LM_CELL_MMA_STAGE_K : fallback, source[m] != 0 ? 16u : 0u);
	#pragma unroll
	for ( uint32_t index = threadIdx.x; index < LM_CELL_MMA_WEIGHT_TILE_VECTORS; index += LM_CELL_MMA_THREADS )
		LmCellMmaCopy(weights + index * 16u,index < weight_vectors ? cells + index * 16u : cells,index < weight_vectors ? 16u : 0u);
}

static __device__ __forceinline__ uint32_t LmCellMmaPackExact(float low, float high)
{
	uint32_t packed;
	asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(packed) : "f"(high), "f"(low));
	return(packed);
}

static __device__ __forceinline__ void LmCellMmaLoad(const uint8_t *tile, uint32_t lane, uint32_t g, uint32_t words[LM_CELL_MMA_FRAGS], uint32_t scales[LM_CELL_MMA_FRAGS])
{
	#pragma unroll
	for ( uint32_t f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
	{
		const uint8_t *cell = tile + (f / 2u) * LM_CELL_MMA_CELL_BYTES;
		words[f] = ((const uint32_t *)(cell + (f % 2u) * 8u * LM_SKINNY_CHUNK_BYTES))[lane];
		scales[f] = cell[LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + (f % 2u) * 8u + g];
	}
}

template<uint32_t CHUNKS>
static __device__ __forceinline__ void LmCellMmaRows(const LmSkinnyCellArguments &args, const LmCellMmaSegments &segments, uint32_t column_segment, uint64_t cells_offset, uint64_t tile_stride, uint32_t cell_first, uint32_t block_first, uint32_t active, uint32_t first, uint32_t end, uint16_t *stage, uint8_t *weights)
{
	constexpr uint32_t COPIES = (CHUNKS * 16u + LM_CELL_MMA_STAGE_ROW_STEP - 1u) / LM_CELL_MMA_STAGE_ROW_STEP;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, g = lane / 4u, c = lane % 4u;
	const uint32_t vector = threadIdx.x % LM_CELL_MMA_STAGE_VECTORS, row = threadIdx.x / LM_CELL_MMA_STAGE_VECTORS;
	const uint32_t block_cells = args.cells - block_first < LM_CELL_MMA_BLOCK_CELLS ? args.cells - block_first : LM_CELL_MMA_BLOCK_CELLS;
	const uint32_t weight_vectors = block_cells * LM_CELL_MMA_CELL_VECTORS;
	const uint32_t warp_offset = (threadIdx.x / LM_WARP_LANES) * LM_CELL_MMA_CELLS * LM_CELL_MMA_CELL_BYTES;
	float acc[CHUNKS][LM_CELL_MMA_FRAGS][4];
	const uint16_t *source[COPIES];
	uint32_t m, h, f, j, t, packed;
	#pragma unroll
	for ( m = 0u; m < CHUNKS; m++ )
	{
		#pragma unroll
		for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
			acc[m][f][0] = acc[m][f][1] = acc[m][f][2] = acc[m][f][3] = 0.0f;
	}
	#pragma unroll
	for ( m = 0u; m < COPIES; m++ )
	{
		const uint32_t local = row + m * LM_CELL_MMA_STAGE_ROW_STEP;
		packed = first + local;
		source[m] = packed < end && local < CHUNKS * 16u ? args.activation + (uint64_t)(args.activation_packed != 0u ? packed : args.route_source_token[packed]) * args.input_dimension + vector * 8u : (const uint16_t *)0;
	}
	#pragma unroll
	for ( t = 0u; t + 1u < LM_CELL_MMA_STAGES; t++ )
	{
		if ( t < args.k_tiles )
			LmCellMmaStage<COPIES>(stage + t * LM_CELL_MMA_STAGE_ELEMENTS,weights + t * LM_CELL_MMA_WEIGHT_TILE_BYTES,source,args.activation,segments,column_segment,cells_offset,tile_stride,t,row,vector,weight_vectors);
		LmCellMmaCommit();
	}
	for ( t = 0u; t < args.k_tiles; t++ )
	{
		const uint32_t slot = t % LM_CELL_MMA_STAGES, ahead = t + LM_CELL_MMA_STAGES - 1u;
		const uint16_t *buffer = stage + slot * LM_CELL_MMA_STAGE_ELEMENTS;
		LmCellMmaWait<LM_CELL_MMA_STAGES - 2u>();
		__syncthreads();
		if ( ahead < args.k_tiles )
			LmCellMmaStage<COPIES>(stage + (ahead % LM_CELL_MMA_STAGES) * LM_CELL_MMA_STAGE_ELEMENTS,weights + (ahead % LM_CELL_MMA_STAGES) * LM_CELL_MMA_WEIGHT_TILE_BYTES,source,args.activation,segments,column_segment,cells_offset,tile_stride,ahead,row,vector,weight_vectors);
		LmCellMmaCommit();
		if ( active != 0u )
		{
			uint32_t b[LM_CELL_MMA_FRAGS][4], current[LM_CELL_MMA_FRAGS], scale_bits[LM_CELL_MMA_FRAGS];
			LmCellMmaLoad(weights + slot * LM_CELL_MMA_WEIGHT_TILE_BYTES + warp_offset, lane, g, current, scale_bits);
			#pragma unroll
			for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
			{
				const float scale = LmUe8m0ToFloat((uint8_t)scale_bits[f]);
				#pragma unroll
				for ( j = 0u; j < 4u; j++ )
				{
					const float2 pair = LmSkinnyE2m1Pair(current[f],j);
					b[f][j] = LmCellMmaPackExact(pair.x * scale,pair.y * scale);
				}
			}
			#pragma unroll
			for ( m = 0u; m < CHUNKS; m++ )
			{
				const uint16_t *a_row = buffer + (m * 16u + g) * LM_CELL_MMA_STAGE_PITCH + c * 8u;
				const uint4 low = *(const uint4 *)a_row;
				const uint4 high = *(const uint4 *)(a_row + 8u * LM_CELL_MMA_STAGE_PITCH);
				const uint32_t a0[4] = {low.x,high.x,low.y,high.y};
				const uint32_t a1[4] = {low.z,high.z,low.w,high.w};
				#pragma unroll
				for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
				{
					const uint32_t b0[2] = {b[f][0],b[f][1]};
					const uint32_t b1[2] = {b[f][2],b[f][3]};
					LmMmaBf16(acc[m][f],a0,b0);
					LmMmaBf16(acc[m][f],a1,b1);
				}
			}
		}
	}
	LmCellMmaWait<0u>();
	__syncthreads();
	if ( active == 0u )
		return;
	#pragma unroll
	for ( m = 0u; m < CHUNKS; m++ )
	{
		#pragma unroll
		for ( h = 0u; h < 2u; h++ )
		{
			packed = first + m * 16u + g + h * 8u;
			if ( packed >= end )
				continue;
			#pragma unroll
			for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
				*(uint32_t *)(args.output_bf16 + (uint64_t)packed * args.output_dimension + (cell_first + f / 2u) * LM_SKINNY_CELL_NEURONS + (f % 2u) * 8u + c * 2u) =
					LmPackBf16Pair(acc[m][f][h * 2u],acc[m][f][h * 2u + 1u]);
		}
	}
}

static __global__ __launch_bounds__(LM_CELL_MMA_THREADS) void LmCellMmaKernel(const __grid_constant__ LmSkinnyCellArguments args, const __grid_constant__ LmCellMmaSegments segments)
{
	__shared__ __align__(16) uint16_t stage[LM_CELL_MMA_STAGES * LM_CELL_MMA_STAGE_ELEMENTS];
	__shared__ __align__(16) uint8_t weights[LM_CELL_MMA_STAGES * LM_CELL_MMA_WEIGHT_TILE_BYTES];
	const uint32_t pairs = args.cells / LM_CELL_MMA_CELLS;
	const uint32_t blocks = (pairs + LM_CELL_MMA_WARPS - 1u) / LM_CELL_MMA_WARPS;
	const uint32_t slots = gridDim.x / (args.groups * blocks);
	const uint32_t group = blockIdx.x / (blocks * slots);
	const uint32_t slot = (blockIdx.x % (blocks * slots)) / blocks;
	const uint32_t pair = (blockIdx.x % blocks) * LM_CELL_MMA_WARPS + threadIdx.x / LM_WARP_LANES;
	const uint32_t active = pair < pairs ? 1u : 0u;
	const uint32_t cell_first = (active != 0u ? pair : 0u) * LM_CELL_MMA_CELLS;
	const uint32_t block_first = (blockIdx.x % blocks) * LM_CELL_MMA_BLOCK_CELLS;
	const uint32_t segment_cells = segments.column_cells != 0u ? segments.column_cells : args.cells;
	const uint32_t column_segment = block_first / segment_cells;
	const uint64_t tile_stride = (uint64_t)segment_cells * LM_CELL_MMA_CELL_BYTES;
	const uint64_t cells_offset = (uint64_t)group * segments.group_bytes + (uint64_t)(block_first - column_segment * segment_cells) * LM_CELL_MMA_CELL_BYTES;
	uint32_t first, end, last;
	LmDependentWait();
	if ( group >= args.groups )
		return;
	end = args.group_row_offset[group + 1u];
	for ( first = args.group_row_offset[group] + slot * LM_CELL_MMA_ROWS; first < end; first += slots * LM_CELL_MMA_ROWS )
	{
		last = end - first < LM_CELL_MMA_ROWS ? end : first + LM_CELL_MMA_ROWS;
		if ( last - first > 4u * 16u )
			LmCellMmaRows<LM_CELL_MMA_CHUNKS>(args,segments,column_segment,cells_offset,tile_stride,cell_first,block_first,active,first,last,stage,weights);
		else if ( last - first > 3u * 16u )
			LmCellMmaRows<4u>(args,segments,column_segment,cells_offset,tile_stride,cell_first,block_first,active,first,last,stage,weights);
		else if ( last - first > 2u * 16u )
			LmCellMmaRows<3u>(args,segments,column_segment,cells_offset,tile_stride,cell_first,block_first,active,first,last,stage,weights);
		else if ( last - first > 16u )
			LmCellMmaRows<2u>(args,segments,column_segment,cells_offset,tile_stride,cell_first,block_first,active,first,last,stage,weights);
		else
			LmCellMmaRows<1u>(args,segments,column_segment,cells_offset,tile_stride,cell_first,block_first,active,first,last,stage,weights);
	}
}

static int32_t LmCellMmaLaunchSegments(LmSkinnyCellArguments &args, LmCellMmaSegments &segments, const void *const *weight, uint32_t segment_count, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, uint32_t tile_k)
{
	uint32_t segment;
	if ( tile_k != LM_SKINNY_CELL_TILE_K || weight == 0 || segment_count == 0u || segment_count > LM_CELL_MMA_MAX_SEGMENTS || activation == 0 || output_bf16 == 0 ||
		group_row_offset == 0 || (activation_packed == 0u && route_source_token == 0) || groups == 0u || packed_rows == 0u ||
		input_dimension == 0u || (input_dimension % LM_SKINNY_CELL_TILE_K) != 0u ||
		output_dimension == 0u || (output_dimension % (LM_SKINNY_CELL_NEURONS * LM_CELL_MMA_CELLS)) != 0u || LmSkinnyAligned(activation) == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	memset(&segments,0,sizeof(segments));
	for ( segment = 0u; segment < segment_count; segment++ )
	{
		if ( weight[segment] == 0 || LmSkinnyAligned(weight[segment]) == 0u )
			return(LM_LAUNCH_ERR_SHAPE);
		segments.weight[segment] = (const uint8_t *)weight[segment];
	}
	memset(&args,0,sizeof(args));
	args.weight = segments.weight[0];
	args.activation = activation;
	args.output_bf16 = output_bf16;
	args.group_row_offset = group_row_offset;
	args.route_source_token = route_source_token;
	args.groups = groups;
	args.packed_rows = packed_rows;
	args.activation_packed = activation_packed;
	args.input_dimension = input_dimension;
	args.output_dimension = output_dimension;
	args.k_tiles = input_dimension / LM_SKINNY_CELL_TILE_K;
	args.cells = output_dimension / LM_SKINNY_CELL_NEURONS;
	return(LM_LAUNCH_OK);
}

static int32_t LmCellMmaLaunchKernel(LmSkinnyCellArguments &args, LmCellMmaSegments &segments, cudaStream_t stream)
{
	uint64_t blocks, slots;
	args.group_bytes = segments.group_bytes;
	slots = ((uint64_t)args.packed_rows + (uint64_t)args.groups * LM_CELL_MMA_ROWS - 1u) / ((uint64_t)args.groups * LM_CELL_MMA_ROWS);
	slots = args.k_tiles < LM_CELL_MMA_SLOT_MIN_K_TILES ? 1u : slots < LM_CELL_MMA_ROW_SLOTS ? slots : LM_CELL_MMA_ROW_SLOTS;
	blocks = (uint64_t)args.groups * slots * ((args.cells / LM_CELL_MMA_CELLS + LM_CELL_MMA_WARPS - 1u) / LM_CELL_MMA_WARPS);
	LM_LAUNCH_DEPENDENT((LmCellMmaKernel),(uint32_t)blocks,LM_CELL_MMA_THREADS,0u,stream,args,segments);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t LmCellMmaExpertsSegmented(const void *const *weight, uint32_t segment_count, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, uint32_t tile_k, cudaStream_t stream)
{
	LmSkinnyCellArguments args;
	LmCellMmaSegments segments;
	int32_t status = LmCellMmaLaunchSegments(args,segments,weight,segment_count,activation,output_bf16,group_row_offset,route_source_token,groups,packed_rows,activation_packed,
		input_dimension,output_dimension,tile_k);
	if ( status != LM_LAUNCH_OK || (args.k_tiles % segment_count) != 0u )
		return(status != LM_LAUNCH_OK ? status : LM_LAUNCH_ERR_SHAPE);
	segments.tiles = args.k_tiles / segment_count;
	segments.group_bytes = (uint64_t)segments.tiles * args.cells * LM_CELL_MMA_CELL_BYTES;
	return(LmCellMmaLaunchKernel(args,segments,stream));
}

static int32_t LmCellMmaExpertsColumns(const void *const *weight, uint32_t segment_count, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, uint32_t tile_k, cudaStream_t stream)
{
	LmSkinnyCellArguments args;
	LmCellMmaSegments segments;
	int32_t status = LmCellMmaLaunchSegments(args,segments,weight,segment_count,activation,output_bf16,group_row_offset,route_source_token,groups,packed_rows,activation_packed,
		input_dimension,output_dimension,tile_k);
	if ( status != LM_LAUNCH_OK || (args.cells % (segment_count * LM_CELL_MMA_BLOCK_CELLS)) != 0u )
		return(status != LM_LAUNCH_OK ? status : LM_LAUNCH_ERR_SHAPE);
	segments.tiles = args.k_tiles;
	segments.column_cells = args.cells / segment_count;
	segments.group_bytes = (uint64_t)args.k_tiles * segments.column_cells * LM_CELL_MMA_CELL_BYTES;
	return(LmCellMmaLaunchKernel(args,segments,stream));
}

static int32_t LmCellMmaExperts(const void *weight, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, uint32_t tile_k, cudaStream_t stream)
{
	return(LmCellMmaExpertsSegmented(&weight,1u,activation,output_bf16,group_row_offset,route_source_token,groups,packed_rows,activation_packed,input_dimension,output_dimension,tile_k,stream));
}
