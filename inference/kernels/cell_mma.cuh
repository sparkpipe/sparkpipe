#pragma once

#include "inference/kernels/skinny.cuh"
#include "inference/kernels/mma.cuh"

#define LM_CELL_MMA_CELLS 2u
#define LM_CELL_MMA_FRAGS (LM_CELL_MMA_CELLS * 2u)
#define LM_CELL_MMA_WARPS 4u
#define LM_CELL_MMA_THREADS (LM_CELL_MMA_WARPS * LM_WARP_LANES)
#define LM_CELL_MMA_CHUNKS 4u
#define LM_CELL_MMA_ROWS (LM_CELL_MMA_CHUNKS * 16u)
#define LM_CELL_MMA_STAGE_TILES 2u
#define LM_CELL_MMA_STAGE_K (LM_CELL_MMA_STAGE_TILES * LM_SKINNY_CELL_TILE_K)
#define LM_CELL_MMA_STAGE_PITCH (LM_CELL_MMA_STAGE_K + 32u)
#define LM_CELL_MMA_STAGE_ELEMENTS (LM_CELL_MMA_ROWS * LM_CELL_MMA_STAGE_PITCH)
#define LM_CELL_MMA_STAGE_VECTORS (LM_CELL_MMA_STAGE_K / 8u)
#define LM_CELL_MMA_STAGE_ROW_STEP (LM_CELL_MMA_THREADS / LM_CELL_MMA_STAGE_VECTORS)

static_assert(LM_CELL_MMA_ROWS % LM_CELL_MMA_STAGE_ROW_STEP == 0u, "one cooperative pass covers whole 16-row chunks");

static __device__ __forceinline__ void LmCellMmaCopy(uint16_t *destination, const uint16_t *source, uint32_t bytes)
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

template<uint32_t CHUNKS>
static __device__ __forceinline__ void LmCellMmaStage(uint16_t *buffer, const uint16_t *const source[CHUNKS], const uint16_t *fallback, uint32_t k_first, uint32_t vectors, uint32_t row, uint32_t vector)
{
	#pragma unroll
	for ( uint32_t m = 0u; m < CHUNKS; m++ )
	{
		const uint32_t live = source[m] != 0 && vector < vectors ? 1u : 0u;
		LmCellMmaCopy(buffer + (row + m * LM_CELL_MMA_STAGE_ROW_STEP) * LM_CELL_MMA_STAGE_PITCH + vector * 8u,
			live != 0u ? source[m] + k_first : fallback, live != 0u ? 16u : 0u);
	}
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
		const uint8_t *cell = tile + (f / 2u) * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
		words[f] = __ldcs((const uint32_t *)(cell + (f % 2u) * 8u * LM_SKINNY_CHUNK_BYTES) + lane);
		scales[f] = __ldg(cell + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + (f % 2u) * 8u + g);
	}
}

template<uint32_t CHUNKS>
static __device__ __forceinline__ void LmCellMmaRows(const LmSkinnyCellArguments &args, const uint8_t *base, uint64_t tile_stride, uint32_t cell_first, uint32_t active, uint32_t first, uint32_t end, uint16_t *stage)
{
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, g = lane / 4u, c = lane % 4u;
	const uint32_t vector = threadIdx.x % LM_CELL_MMA_STAGE_VECTORS, row = threadIdx.x / LM_CELL_MMA_STAGE_VECTORS;
	const uint32_t stages = (args.k_tiles + LM_CELL_MMA_STAGE_TILES - 1u) / LM_CELL_MMA_STAGE_TILES;
	const uint8_t *cells = base + (uint64_t)cell_first * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	float acc[CHUNKS][LM_CELL_MMA_FRAGS][4];
	const uint16_t *source[CHUNKS];
	uint32_t words[LM_CELL_MMA_FRAGS], scales[LM_CELL_MMA_FRAGS];
	uint32_t m, h, f, j, s, packed;
	#pragma unroll
	for ( m = 0u; m < CHUNKS; m++ )
	{
		#pragma unroll
		for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
			acc[m][f][0] = acc[m][f][1] = acc[m][f][2] = acc[m][f][3] = 0.0f;
		packed = first + row + m * LM_CELL_MMA_STAGE_ROW_STEP;
		source[m] = packed < end ? args.activation + (uint64_t)(args.activation_packed != 0u ? packed : args.route_source_token[packed]) * args.input_dimension + vector * 8u : (const uint16_t *)0;
	}
	if ( active != 0u )
		LmCellMmaLoad(cells, lane, g, words, scales);
	LmCellMmaStage<CHUNKS>(stage, source, args.activation, 0u, (args.k_tiles < LM_CELL_MMA_STAGE_TILES ? args.k_tiles : LM_CELL_MMA_STAGE_TILES) * 4u, row, vector);
	LmCellMmaCommit();
	for ( s = 0u; s < stages; s++ )
	{
		uint16_t *buffer = stage + (s % 2u) * LM_CELL_MMA_STAGE_ELEMENTS;
		const uint32_t t_first = s * LM_CELL_MMA_STAGE_TILES;
		const uint32_t tiles = args.k_tiles - t_first < LM_CELL_MMA_STAGE_TILES ? args.k_tiles - t_first : LM_CELL_MMA_STAGE_TILES;
		if ( s + 1u < stages )
		{
			const uint32_t next = args.k_tiles - t_first - tiles < LM_CELL_MMA_STAGE_TILES ? args.k_tiles - t_first - tiles : LM_CELL_MMA_STAGE_TILES;
			LmCellMmaStage<CHUNKS>(stage + ((s + 1u) % 2u) * LM_CELL_MMA_STAGE_ELEMENTS, source, args.activation, (t_first + tiles) * LM_SKINNY_CELL_TILE_K, next * 4u, row, vector);
			LmCellMmaCommit();
			LmCellMmaWait<1u>();
		}
		else
			LmCellMmaWait<0u>();
		__syncthreads();
		if ( active != 0u )
		{
			#pragma unroll
			for ( uint32_t local = 0u; local < LM_CELL_MMA_STAGE_TILES; local++ )
			{
				const uint32_t t = t_first + local;
				uint32_t b[LM_CELL_MMA_FRAGS][4], current[LM_CELL_MMA_FRAGS], scale_bits[LM_CELL_MMA_FRAGS];
				if ( local >= tiles )
					break;
				#pragma unroll
				for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
				{
					current[f] = words[f];
					scale_bits[f] = scales[f];
				}
				if ( t + 1u < args.k_tiles )
					LmCellMmaLoad(cells + (uint64_t)(t + 1u) * tile_stride, lane, g, words, scales);
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
					const uint16_t *a_row = buffer + (m * 16u + g) * LM_CELL_MMA_STAGE_PITCH + local * LM_SKINNY_CELL_TILE_K + c * 8u;
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
		__syncthreads();
	}
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

static __global__ __launch_bounds__(LM_CELL_MMA_THREADS) void LmCellMmaKernel(const __grid_constant__ LmSkinnyCellArguments args)
{
	__shared__ __align__(16) uint16_t stage[2u * LM_CELL_MMA_STAGE_ELEMENTS];
	const uint32_t pairs = args.cells / LM_CELL_MMA_CELLS;
	const uint32_t blocks = (pairs + LM_CELL_MMA_WARPS - 1u) / LM_CELL_MMA_WARPS;
	const uint32_t group = blockIdx.x / blocks;
	const uint32_t pair = (blockIdx.x % blocks) * LM_CELL_MMA_WARPS + threadIdx.x / LM_WARP_LANES;
	const uint32_t active = pair < pairs ? 1u : 0u;
	const uint32_t cell_first = (active != 0u ? pair : 0u) * LM_CELL_MMA_CELLS;
	const uint64_t tile_stride = (uint64_t)args.cells * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	uint32_t first, end, count;
	const uint8_t *base;
	LmDependentWait();
	if ( group >= args.groups )
		return;
	first = args.group_row_offset[group];
	end = args.group_row_offset[group + 1u];
	base = args.weight + (uint64_t)group * args.group_bytes;
	for ( ; first < end; first += count )
	{
		count = end - first;
		if ( count > 3u * 16u )
		{
			LmCellMmaRows<4u>(args,base,tile_stride,cell_first,active,first,end < first + LM_CELL_MMA_ROWS ? end : first + LM_CELL_MMA_ROWS,stage);
			count = LM_CELL_MMA_ROWS;
		}
		else if ( count > 2u * 16u )
			LmCellMmaRows<3u>(args,base,tile_stride,cell_first,active,first,end,stage);
		else if ( count > 16u )
			LmCellMmaRows<2u>(args,base,tile_stride,cell_first,active,first,end,stage);
		else
			LmCellMmaRows<1u>(args,base,tile_stride,cell_first,active,first,end,stage);
	}
}

static int32_t LmCellMmaExperts(const void *weight, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, uint32_t tile_k, cudaStream_t stream)
{
	LmSkinnyCellArguments args;
	uint64_t blocks;
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
	blocks = (uint64_t)groups * ((args.cells / LM_CELL_MMA_CELLS + LM_CELL_MMA_WARPS - 1u) / LM_CELL_MMA_WARPS);
	LM_LAUNCH_DEPENDENT((LmCellMmaKernel),(uint32_t)blocks,LM_CELL_MMA_THREADS,0u,stream,args);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
