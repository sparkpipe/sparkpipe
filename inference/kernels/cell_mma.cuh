#pragma once

#include "inference/kernels/skinny.cuh"
#include "inference/kernels/mma.cuh"

#define LM_CELL_MMA_CELLS 2u
#define LM_CELL_MMA_FRAGS (LM_CELL_MMA_CELLS * 2u)
#define LM_CELL_MMA_WARPS 4u
#define LM_CELL_MMA_CHUNKS 4u

template<uint32_t CHUNKS>
static __device__ __forceinline__ void LmCellMmaRows(const LmSkinnyCellArguments &args, const uint8_t *base, uint64_t tile_stride, uint32_t cell_first, uint32_t lane, uint32_t first, uint32_t end)
{
	const uint32_t g = lane / 4u, c = lane % 4u;
	float acc[CHUNKS][LM_CELL_MMA_FRAGS][4];
	const uint16_t *source[CHUNKS][2];
	uint32_t m, h, f, j, t, packed;
	#pragma unroll
	for ( m = 0u; m < CHUNKS; m++ )
	{
		#pragma unroll
		for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
			acc[m][f][0] = acc[m][f][1] = acc[m][f][2] = acc[m][f][3] = 0.0f;
		#pragma unroll
		for ( h = 0u; h < 2u; h++ )
		{
			packed = first + m * 16u + g + h * 8u;
			source[m][h] = packed < end ? args.activation + (uint64_t)(args.activation_packed != 0u ? packed : args.route_source_token[packed]) * args.input_dimension + c * 8u : (const uint16_t *)0;
		}
	}
	#pragma unroll 2
	for ( t = 0u; t < args.k_tiles; t++ )
	{
		const uint8_t *tile = base + (uint64_t)t * tile_stride + (uint64_t)cell_first * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
		uint32_t b[LM_CELL_MMA_FRAGS][4];
		#pragma unroll
		for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
		{
			const uint8_t *cell = tile + (f / 2u) * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
			const uint32_t word = __ldcs((const uint32_t *)(cell + (f % 2u) * 8u * LM_SKINNY_CHUNK_BYTES) + lane);
			const float scale = LmUe8m0ToFloat(__ldg(cell + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + (f % 2u) * 8u + g));
			#pragma unroll
			for ( j = 0u; j < 4u; j++ )
			{
				const float2 pair = LmSkinnyE2m1Pair(word,j);
				b[f][j] = LmPackBf16Pair(pair.x * scale,pair.y * scale);
			}
		}
		#pragma unroll
		for ( m = 0u; m < CHUNKS; m++ )
		{
			const uint4 zero = make_uint4(0u,0u,0u,0u);
			const uint4 low = source[m][0] != 0 ? __ldg((const uint4 *)(source[m][0] + t * LM_SKINNY_CELL_TILE_K)) : zero;
			const uint4 high = source[m][1] != 0 ? __ldg((const uint4 *)(source[m][1] + t * LM_SKINNY_CELL_TILE_K)) : zero;
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

static __global__ __launch_bounds__(LM_CELL_MMA_WARPS * LM_WARP_LANES) void LmCellMmaKernel(const __grid_constant__ LmSkinnyCellArguments args)
{
	const uint32_t lane = threadIdx.x % LM_WARP_LANES;
	const uint32_t task = blockIdx.x * LM_CELL_MMA_WARPS + threadIdx.x / LM_WARP_LANES;
	const uint32_t pairs = args.cells / LM_CELL_MMA_CELLS;
	const uint64_t tile_stride = (uint64_t)args.cells * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	uint32_t group, first, end, count;
	const uint8_t *base;
	LmDependentWait();
	if ( task >= args.groups * pairs )
		return;
	group = task / pairs;
	first = args.group_row_offset[group];
	end = args.group_row_offset[group + 1u];
	base = args.weight + (uint64_t)group * args.group_bytes;
	for ( ; first < end; first += count )
	{
		count = end - first;
		if ( count > 48u )
		{
			LmCellMmaRows<4u>(args,base,tile_stride,(task % pairs) * LM_CELL_MMA_CELLS,lane,first,end < first + 64u ? end : first + 64u);
			count = 64u;
		}
		else if ( count > 32u )
			LmCellMmaRows<3u>(args,base,tile_stride,(task % pairs) * LM_CELL_MMA_CELLS,lane,first,end);
		else if ( count > 16u )
			LmCellMmaRows<2u>(args,base,tile_stride,(task % pairs) * LM_CELL_MMA_CELLS,lane,first,end);
		else
			LmCellMmaRows<1u>(args,base,tile_stride,(task % pairs) * LM_CELL_MMA_CELLS,lane,first,end);
	}
}

static int32_t LmCellMmaExperts(const void *weight, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, uint32_t tile_k, cudaStream_t stream)
{
	LmSkinnyCellArguments args;
	uint64_t tasks;
	if ( tile_k != LM_SKINNY_CELL_TILE_K || weight == 0 || activation == 0 || output_bf16 == 0 || group_row_offset == 0 ||
		(activation_packed == 0u && route_source_token == 0) || groups == 0u || packed_rows == 0u ||
		input_dimension == 0u || (input_dimension % LM_SKINNY_CELL_TILE_K) != 0u ||
		output_dimension == 0u || (output_dimension % (LM_SKINNY_CELL_NEURONS * LM_CELL_MMA_CELLS)) != 0u ||
		LmSkinnyAligned(weight) == 0u || LmSkinnyAligned(activation) == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	memset(&args,0,sizeof(args));
	args.weight = (const uint8_t *)weight;
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
	args.group_bytes = (uint64_t)args.k_tiles * args.cells * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	tasks = (uint64_t)groups * (args.cells / LM_CELL_MMA_CELLS);
	LM_LAUNCH_DEPENDENT((LmCellMmaKernel),(uint32_t)((tasks + LM_CELL_MMA_WARPS - 1u) / LM_CELL_MMA_WARPS),LM_CELL_MMA_WARPS * LM_WARP_LANES,0u,stream,args);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
