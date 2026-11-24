#pragma once
#include "inference/kernels/cell_mma.cuh"
#include "sparkpipe/spark_gpu.h"

#define LM_CELL_MX_PITCH 48u
#define LM_CELL_MX_STAGE_BYTES (LM_CELL_MMA_ROWS * LM_CELL_MX_PITCH)
#define LM_CELL_MX_ROW_VECTORS (LM_SKINNY_CELL_TILE_K / 16u)
#define LM_CELL_MX_ROW_STEP (LM_CELL_MMA_THREADS / LM_CELL_MX_ROW_VECTORS)

static_assert(LM_SKINNY_CELL_TILE_K == 32u && LM_CELL_MX_PITCH % 16u == 0u && LM_CELL_MX_PITCH >= LM_SKINNY_CELL_TILE_K,
	"one 32-wide fp8 block per stage row, in 16-byte copies");
static_assert(LM_CELL_MMA_STAGES == 4u && LM_CELL_MMA_WEIGHT_TILE_VECTORS <= 2u * LM_CELL_MMA_THREADS,
	"the stages turn once per four-block scale word, so each unrolled tile has a fixed slot and scale byte, and a tile's cells copy in two passes");

#define LM_CELL_MX_MAPPED 0x80000000u

typedef struct LmCellMxCodes
{
	const uint8_t *codes;
	const uint8_t *scales;
	uint64_t code_stride;
	uint64_t scale_stride;
	const uint32_t *output_rows;
	uint16_t *mapped_output;
}
LmCellMxCodes;

static __device__ __forceinline__ uint32_t LmCellMxShared32(uint32_t address)
{
	uint32_t value;
	asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(value) : "r"(address));
	return(value);
}

static __device__ __forceinline__ uint32_t LmCellMxShared8(uint32_t address)
{
	uint32_t value;
	asm volatile("ld.shared.u8 %0, [%1];\n" : "=r"(value) : "r"(address));
	return(value);
}

static __device__ __forceinline__ void LmCellMxCopy(uint32_t shared, const void *source, uint32_t bytes)
{
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(shared), "l"(source), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void LmCellMxCopyWord(uint32_t shared, const void *source, uint32_t bytes)
{
	asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(shared), "l"(source), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void LmCellMxE2m1(uint32_t word, uint32_t &low, uint32_t &high)
{
	const uint32_t even = (word << 2u) & 0x3c3c3c3cu, odd = (word >> 2u) & 0x3c3c3c3cu;
	low = __byte_perm(even, odd, 0x5140u);
	high = __byte_perm(even, odd, 0x7362u);
}

template<uint32_t CHUNKS>
static __device__ __forceinline__ void LmCellMxRows(const LmSkinnyCellArguments &args, const LmCellMxCodes &mx, const uint8_t *cells, uint64_t tile_stride,
	uint32_t cell_first, uint32_t block_first, uint32_t active, uint32_t first, uint32_t end, uint8_t *stage, uint8_t *weights, uint8_t *scale_buffer)
{
	constexpr uint32_t COPIES = (CHUNKS * 16u + LM_CELL_MX_ROW_STEP - 1u) / LM_CELL_MX_ROW_STEP;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, g = lane / 4u, c = lane % 4u;
	const uint32_t vector = threadIdx.x % LM_CELL_MX_ROW_VECTORS, row = threadIdx.x / LM_CELL_MX_ROW_VECTORS;
	const uint32_t block_cells = args.cells - block_first < LM_CELL_MMA_BLOCK_CELLS ? args.cells - block_first : LM_CELL_MMA_BLOCK_CELLS;
	const uint32_t weight_vectors = block_cells * LM_CELL_MMA_CELL_VECTORS;
	const uint32_t warp_offset = (threadIdx.x / LM_WARP_LANES) * LM_CELL_MMA_CELLS * LM_CELL_MMA_CELL_BYTES;
	const uint32_t stage_s = (uint32_t)__cvta_generic_to_shared(stage), weights_s = (uint32_t)__cvta_generic_to_shared(weights);
	const uint32_t scales_s = (uint32_t)__cvta_generic_to_shared(scale_buffer);
	const uint32_t w_first = threadIdx.x, w_second = threadIdx.x + LM_CELL_MMA_THREADS;
	float acc[CHUNKS][LM_CELL_MMA_FRAGS][4];
	const uint8_t *source[COPIES];
	const uint8_t *scale_source = threadIdx.x < CHUNKS * 16u && first + threadIdx.x < end ?
		mx.scales + (uint64_t)(args.activation_packed != 0u ? first + threadIdx.x : args.route_source_token[first + threadIdx.x]) * mx.scale_stride :
		(const uint8_t *)0;
	uint32_t m, h, f;
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
		const uint32_t local = row + m * LM_CELL_MX_ROW_STEP, packed = first + local;
		source[m] = packed < end && local < CHUNKS * 16u ?
			mx.codes + (uint64_t)(args.activation_packed != 0u ? packed : args.route_source_token[packed]) * mx.code_stride + vector * 16u :
			(const uint8_t *)0;
	}
	auto produce = [&](uint32_t tile, uint32_t slot)
	{
		const uint8_t *tile_cells = cells + (uint64_t)tile * tile_stride;
		if ( (tile & 3u) == 0u && threadIdx.x < LM_CELL_MMA_ROWS )
			LmCellMxCopyWord(scales_s + ((tile >> 2u) & 1u) * LM_CELL_MMA_ROWS * 4u + threadIdx.x * 4u, scale_source != 0 ? scale_source + tile : mx.scales,
				scale_source != 0 ? 4u : 0u);
		#pragma unroll
		for ( uint32_t q = 0u; q < COPIES; q++ )
			LmCellMxCopy(stage_s + slot * LM_CELL_MX_STAGE_BYTES + (row + q * LM_CELL_MX_ROW_STEP) * LM_CELL_MX_PITCH + vector * 16u,
				source[q] != 0 ? source[q] + tile * LM_SKINNY_CELL_TILE_K : mx.codes, source[q] != 0 ? 16u : 0u);
		LmCellMxCopy(weights_s + slot * LM_CELL_MMA_WEIGHT_TILE_BYTES + w_first * 16u, w_first < weight_vectors ? tile_cells + w_first * 16u : tile_cells,
			w_first < weight_vectors ? 16u : 0u);
		if ( w_second < LM_CELL_MMA_WEIGHT_TILE_VECTORS )
			LmCellMxCopy(weights_s + slot * LM_CELL_MMA_WEIGHT_TILE_BYTES + w_second * 16u, w_second < weight_vectors ? tile_cells + w_second * 16u : tile_cells,
				w_second < weight_vectors ? 16u : 0u);
	};
	#pragma unroll
	for ( uint32_t t = 0u; t + 1u < LM_CELL_MMA_STAGES; t++ )
	{
		if ( t < args.k_tiles )
			produce(t, t);
		LmCellMmaCommit();
	}
	const uint32_t a_lane = stage_s + g * LM_CELL_MX_PITCH + c * 8u, w_lane = weights_s + warp_offset + lane * 4u;
	const uint32_t ws_lane = weights_s + warp_offset + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + g;
	const uint32_t s_lane = scales_s + (g + ((lane & 1u) << 3u)) * 4u;
	for ( uint32_t base = 0u; base < args.k_tiles; base += LM_CELL_MMA_STAGES )
	{
		uint32_t scale_word[CHUNKS];
		#pragma unroll
		for ( uint32_t u = 0u; u < LM_CELL_MMA_STAGES; u++ )
		{
			const uint32_t t = base + u;
			LmCellMmaWait<LM_CELL_MMA_STAGES - 2u>();
			__syncthreads();
			if ( t + LM_CELL_MMA_STAGES - 1u < args.k_tiles )
				produce(t + LM_CELL_MMA_STAGES - 1u, (u + LM_CELL_MMA_STAGES - 1u) % LM_CELL_MMA_STAGES);
			LmCellMmaCommit();
			if ( active != 0u )
			{
				const uint32_t a_stage = a_lane + u * LM_CELL_MX_STAGE_BYTES, w_tile = w_lane + u * LM_CELL_MMA_WEIGHT_TILE_BYTES;
				const uint32_t ws_tile = ws_lane + u * LM_CELL_MMA_WEIGHT_TILE_BYTES;
				uint32_t b[LM_CELL_MMA_FRAGS][2], scale_b[LM_CELL_MMA_FRAGS];
				if ( u == 0u )
				{
					#pragma unroll
					for ( m = 0u; m < CHUNKS; m++ )
						scale_word[m] = LmCellMxShared32(s_lane + ((base >> 2u) & 1u) * LM_CELL_MMA_ROWS * 4u + m * 16u * 4u);
				}
				#pragma unroll
				for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
				{
					LmCellMxE2m1(LmCellMxShared32(w_tile + (f / 2u) * LM_CELL_MMA_CELL_BYTES + (f % 2u) * 8u * LM_SKINNY_CHUNK_BYTES), b[f][0], b[f][1]);
					scale_b[f] = LmCellMxShared8(ws_tile + (f / 2u) * LM_CELL_MMA_CELL_BYTES + (f % 2u) * 8u);
				}
				#pragma unroll
				for ( m = 0u; m < CHUNKS; m++ )
				{
					const uint32_t low = a_stage + m * 16u * LM_CELL_MX_PITCH, high = low + 8u * LM_CELL_MX_PITCH;
					const uint32_t a[4] = { LmCellMxShared32(low), LmCellMxShared32(high), LmCellMxShared32(low + 4u), LmCellMxShared32(high + 4u) };
					#pragma unroll
					for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
					{
						float block[4];
						if ( u == 0u )
							LmMmaMxf8Mxf4Block<0u>(block, a, b[f], scale_word[m], scale_b[f]);
						else if ( u == 1u )
							LmMmaMxf8Mxf4Block<1u>(block, a, b[f], scale_word[m], scale_b[f]);
						else if ( u == 2u )
							LmMmaMxf8Mxf4Block<2u>(block, a, b[f], scale_word[m], scale_b[f]);
						else
							LmMmaMxf8Mxf4Block<3u>(block, a, b[f], scale_word[m], scale_b[f]);
						acc[m][f][0] += block[0];
						acc[m][f][1] += block[1];
						acc[m][f][2] += block[2];
						acc[m][f][3] += block[3];
					}
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
			const uint32_t packed = first + m * 16u + g + h * 8u;
			if ( packed >= end )
				continue;
			const uint32_t target = mx.output_rows != 0 ? mx.output_rows[packed] : packed;
			uint16_t *row_out = ((target & LM_CELL_MX_MAPPED) != 0u ? mx.mapped_output : args.output_bf16) +
				(uint64_t)(target & ~LM_CELL_MX_MAPPED) * args.output_dimension;
			#pragma unroll
			for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
				*(uint32_t *)(row_out + (cell_first + f / 2u) * LM_SKINNY_CELL_NEURONS + (f % 2u) * 8u + c * 2u) =
					LmPackBf16Pair(acc[m][f][h * 2u], acc[m][f][h * 2u + 1u]);
		}
	}
}

static __global__ __launch_bounds__(LM_CELL_MMA_THREADS) void LmCellMxKernel(const __grid_constant__ LmSkinnyCellArguments args,
	const __grid_constant__ LmCellMmaSegments segments, const __grid_constant__ LmCellMxCodes mx)
{
	__shared__ __align__(16) uint8_t stage[LM_CELL_MMA_STAGES * LM_CELL_MX_STAGE_BYTES];
	__shared__ __align__(16) uint8_t weights[LM_CELL_MMA_STAGES * LM_CELL_MMA_WEIGHT_TILE_BYTES];
	__shared__ __align__(16) uint8_t scale_buffer[2u * LM_CELL_MMA_ROWS * 4u];
	const uint32_t pairs = args.cells / LM_CELL_MMA_CELLS;
	const uint32_t blocks = (pairs + LM_CELL_MMA_WARPS - 1u) / LM_CELL_MMA_WARPS;
	const uint32_t slots = gridDim.x / (args.groups * blocks);
	const uint32_t group = blockIdx.x / (blocks * slots);
	const uint32_t slot = (blockIdx.x % (blocks * slots)) / blocks;
	const uint32_t pair = (blockIdx.x % blocks) * LM_CELL_MMA_WARPS + threadIdx.x / LM_WARP_LANES;
	const uint32_t active = pair < pairs ? 1u : 0u;
	const uint32_t cell_first = (active != 0u ? pair : 0u) * LM_CELL_MMA_CELLS;
	const uint32_t block_first = (blockIdx.x % blocks) * LM_CELL_MMA_BLOCK_CELLS;
	const uint64_t tile_stride = (uint64_t)args.cells * LM_CELL_MMA_CELL_BYTES;
	const uint8_t *cells = segments.weight[0] + LmCellMmaGroupOffset(segments, group) + (uint64_t)block_first * LM_CELL_MMA_CELL_BYTES;
	uint32_t first, end, last;
	LmDependentWait();
	if ( group >= args.groups )
		return;
	end = args.group_row_offset[group + 1u];
	for ( first = args.group_row_offset[group] + slot * LM_CELL_MMA_ROWS; first < end; first += slots * LM_CELL_MMA_ROWS )
	{
		last = end - first < LM_CELL_MMA_ROWS ? end : first + LM_CELL_MMA_ROWS;
		if ( last - first > 4u * 16u )
			LmCellMxRows<LM_CELL_MMA_CHUNKS>(args, mx, cells, tile_stride, cell_first, block_first, active, first, last, stage, weights, scale_buffer);
		else if ( last - first > 2u * 16u )
			LmCellMxRows<4u>(args, mx, cells, tile_stride, cell_first, block_first, active, first, last, stage, weights, scale_buffer);
		else if ( last - first > 16u )
			LmCellMxRows<2u>(args, mx, cells, tile_stride, cell_first, block_first, active, first, last, stage, weights, scale_buffer);
		else
			LmCellMxRows<1u>(args, mx, cells, tile_stride, cell_first, block_first, active, first, last, stage, weights, scale_buffer);
	}
}

static LmCellMxCodes LmCellMxPacked(const uint8_t *codes, const uint8_t *code_scales, uint32_t input_dimension)
{
	LmCellMxCodes mx;
	mx.codes = codes;
	mx.scales = code_scales;
	mx.code_stride = input_dimension;
	mx.scale_stride = input_dimension / LM_SKINNY_CELL_TILE_K;
	mx.output_rows = 0;
	mx.mapped_output = 0;
	return(mx);
}

static int32_t LmCellMxExpertsSplit(const void *weight, const void *upper, uint32_t split, const LmCellMxCodes &mx, uint16_t *output_bf16,
	const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t activation_packed,
	uint32_t input_dimension, uint32_t output_dimension, SparkGpuStream stream)
{
	LmSkinnyCellArguments args;
	LmCellMmaSegments segments;
	uint64_t blocks, slots;
	if ( mx.codes == 0 || mx.scales == 0 || ((uintptr_t)mx.codes % 16u) != 0u || ((uintptr_t)mx.scales % 4u) != 0u || (input_dimension % 128u) != 0u ||
		mx.code_stride < input_dimension || mx.code_stride % 16u != 0u || mx.scale_stride < input_dimension / LM_SKINNY_CELL_TILE_K || mx.scale_stride % 4u != 0u ||
		(mx.output_rows != 0 && mx.mapped_output == 0) )
		return(LM_LAUNCH_ERR_SHAPE);
	int32_t status = LmCellMmaLaunchSegments(args, segments, &weight, 1u, (const uint16_t *)mx.codes, output_bf16, group_row_offset, route_source_token, groups,
		packed_rows, activation_packed, input_dimension, output_dimension, LM_SKINNY_CELL_TILE_K);
	if ( status != LM_LAUNCH_OK )
		return(status);
	if ( upper != 0 && (LmSkinnyAligned(upper) == 0u || split == 0u || split >= groups) )
		return(LM_LAUNCH_ERR_SHAPE);
	segments.tiles = args.k_tiles;
	segments.group_bytes = (uint64_t)args.k_tiles * args.cells * LM_CELL_MMA_CELL_BYTES;
	segments.split = upper != 0 ? split : 0u;
	segments.upper_offset = upper != 0 ? (uint64_t)((uintptr_t)upper - (uintptr_t)weight) : 0u;
	args.group_bytes = segments.group_bytes;
	slots = ((uint64_t)args.packed_rows + (uint64_t)args.groups * LM_CELL_MMA_ROWS - 1u) / ((uint64_t)args.groups * LM_CELL_MMA_ROWS);
	slots = args.k_tiles < LM_CELL_MMA_SLOT_MIN_K_TILES ? 1u : slots < LM_CELL_MMA_ROW_SLOTS ? slots : LM_CELL_MMA_ROW_SLOTS;
	blocks = (uint64_t)args.groups * slots * ((args.cells / LM_CELL_MMA_CELLS + LM_CELL_MMA_WARPS - 1u) / LM_CELL_MMA_WARPS);
	LM_LAUNCH_DEPENDENT((LmCellMxKernel), (uint32_t)blocks, LM_CELL_MMA_THREADS, 0u, stream, args, segments, mx);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t LmCellMxExperts(const void *weight, const uint8_t *codes, const uint8_t *code_scales, uint16_t *output_bf16, const uint32_t *group_row_offset,
	const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t activation_packed, uint32_t input_dimension,
	uint32_t output_dimension, SparkGpuStream stream)
{
	return(LmCellMxExpertsSplit(weight, 0, 0u, LmCellMxPacked(codes, code_scales, input_dimension), output_bf16, group_row_offset, route_source_token,
		groups, packed_rows, activation_packed, input_dimension, output_dimension, stream));
}
