#pragma once
#include "inference/kernels/cell_mma.cuh"

#define LM_CELL_MX_PITCH 48u
#define LM_CELL_MX_STAGE_BYTES (LM_CELL_MMA_ROWS * LM_CELL_MX_PITCH)
#define LM_CELL_MX_ROW_VECTORS (LM_SKINNY_CELL_TILE_K / 16u)
#define LM_CELL_MX_ROW_STEP (LM_CELL_MMA_THREADS / LM_CELL_MX_ROW_VECTORS)

static_assert(LM_SKINNY_CELL_TILE_K == 32u && LM_CELL_MX_PITCH % 16u == 0u && LM_CELL_MX_PITCH >= LM_SKINNY_CELL_TILE_K,
	"one 32-wide fp8 block per stage row, in 16-byte copies");

typedef struct LmCellMxCodes
{
	const uint8_t *codes;
	const uint8_t *scales;
}
LmCellMxCodes;

static __device__ __forceinline__ uint32_t LmCellMxE2m1Bytes(uint32_t nibbles)
{
	return(((nibbles & 0xfu) << 2u) | ((nibbles & 0xf0u) << 6u) | ((nibbles & 0xf00u) << 10u) | ((nibbles & 0xf000u) << 14u));
}

template<uint32_t COPIES>
static __device__ __forceinline__ void LmCellMxStage(uint8_t *buffer, uint8_t *weights, uint8_t *scale_buffer, const uint8_t *const source[COPIES],
	const uint8_t *scale_source, const uint8_t *fallback, const uint8_t *scale_fallback, const LmCellMmaSegments &segments, uint32_t column_segment,
	uint64_t cells_offset, uint64_t tile_stride, uint32_t t, uint32_t row, uint32_t vector, uint32_t weight_vectors)
{
	if ( (t & 3u) == 0u && threadIdx.x < LM_CELL_MMA_ROWS )
		LmCellMmaCopyWord(scale_buffer + ((t >> 2u) & 1u) * LM_CELL_MMA_ROWS * 4u + threadIdx.x * 4u, scale_source != 0 ? scale_source + t : scale_fallback,
			scale_source != 0 ? 4u : 0u);
	const uint8_t *cells = segments.weight[column_segment + t / segments.tiles] + cells_offset + (uint64_t)(t % segments.tiles) * tile_stride;
	#pragma unroll
	for ( uint32_t m = 0u; m < COPIES; m++ )
		LmCellMmaCopy(buffer + (row + m * LM_CELL_MX_ROW_STEP) * LM_CELL_MX_PITCH + vector * 16u,
			source[m] != 0 ? source[m] + t * LM_SKINNY_CELL_TILE_K : fallback, source[m] != 0 ? 16u : 0u);
	#pragma unroll
	for ( uint32_t index = threadIdx.x; index < LM_CELL_MMA_WEIGHT_TILE_VECTORS; index += LM_CELL_MMA_THREADS )
		LmCellMmaCopy(weights + index * 16u, index < weight_vectors ? cells + index * 16u : cells, index < weight_vectors ? 16u : 0u);
}

template<uint32_t CHUNKS>
static __device__ __forceinline__ void LmCellMxRows(const LmSkinnyCellArguments &args, const LmCellMmaSegments &segments, const LmCellMxCodes &mx,
	uint32_t column_segment, uint64_t cells_offset, uint64_t tile_stride, uint32_t cell_first, uint32_t block_first, uint32_t active, uint32_t first,
	uint32_t end, uint8_t *stage, uint8_t *weights, uint8_t *scale_buffer)
{
	constexpr uint32_t COPIES = (CHUNKS * 16u + LM_CELL_MX_ROW_STEP - 1u) / LM_CELL_MX_ROW_STEP;
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, g = lane / 4u, c = lane % 4u;
	const uint32_t vector = threadIdx.x % LM_CELL_MX_ROW_VECTORS, row = threadIdx.x / LM_CELL_MX_ROW_VECTORS;
	const uint32_t block_cells = args.cells - block_first < LM_CELL_MMA_BLOCK_CELLS ? args.cells - block_first : LM_CELL_MMA_BLOCK_CELLS;
	const uint32_t weight_vectors = block_cells * LM_CELL_MMA_CELL_VECTORS;
	const uint32_t warp_offset = (threadIdx.x / LM_WARP_LANES) * LM_CELL_MMA_CELLS * LM_CELL_MMA_CELL_BYTES;
	const uint32_t scale_pitch = args.input_dimension / LM_SKINNY_CELL_TILE_K;
	float acc[CHUNKS][LM_CELL_MMA_FRAGS][4];
	const uint8_t *source[COPIES];
	const uint8_t *scale_source = threadIdx.x < CHUNKS * 16u && first + threadIdx.x < end ?
		mx.scales + (uint64_t)(args.activation_packed != 0u ? first + threadIdx.x : args.route_source_token[first + threadIdx.x]) * scale_pitch :
		(const uint8_t *)0;
	uint32_t m, h, f, t, packed;
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
		const uint32_t local = row + m * LM_CELL_MX_ROW_STEP;
		packed = first + local;
		source[m] = packed < end && local < CHUNKS * 16u ?
			mx.codes + (uint64_t)(args.activation_packed != 0u ? packed : args.route_source_token[packed]) * args.input_dimension + vector * 16u :
			(const uint8_t *)0;
	}
	#pragma unroll
	for ( t = 0u; t + 1u < LM_CELL_MMA_STAGES; t++ )
	{
		if ( t < args.k_tiles )
			LmCellMxStage<COPIES>(stage + t * LM_CELL_MX_STAGE_BYTES, weights + t * LM_CELL_MMA_WEIGHT_TILE_BYTES, scale_buffer, source, scale_source,
				mx.codes, mx.scales, segments, column_segment, cells_offset, tile_stride, t, row, vector, weight_vectors);
		LmCellMmaCommit();
	}
	for ( t = 0u; t < args.k_tiles; t++ )
	{
		const uint32_t slot = t % LM_CELL_MMA_STAGES, ahead = t + LM_CELL_MMA_STAGES - 1u;
		const uint8_t *buffer = stage + slot * LM_CELL_MX_STAGE_BYTES;
		LmCellMmaWait<LM_CELL_MMA_STAGES - 2u>();
		__syncthreads();
		if ( ahead < args.k_tiles )
			LmCellMxStage<COPIES>(stage + (ahead % LM_CELL_MMA_STAGES) * LM_CELL_MX_STAGE_BYTES,
				weights + (ahead % LM_CELL_MMA_STAGES) * LM_CELL_MMA_WEIGHT_TILE_BYTES, scale_buffer, source, scale_source, mx.codes, mx.scales, segments,
				column_segment, cells_offset, tile_stride, ahead, row, vector, weight_vectors);
		LmCellMmaCommit();
		if ( active != 0u )
		{
			uint32_t words[LM_CELL_MMA_FRAGS], scale_bits[LM_CELL_MMA_FRAGS], b[LM_CELL_MMA_FRAGS][2];
			LmCellMmaLoad(weights + slot * LM_CELL_MMA_WEIGHT_TILE_BYTES + warp_offset, lane, g, words, scale_bits);
			#pragma unroll
			for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
			{
				b[f][0] = LmCellMxE2m1Bytes(words[f] & 0xffffu);
				b[f][1] = LmCellMxE2m1Bytes(words[f] >> 16u);
			}
			#pragma unroll
			for ( m = 0u; m < CHUNKS; m++ )
			{
				const uint32_t scale_a = (*(const uint32_t *)(scale_buffer + ((t >> 2u) & 1u) * LM_CELL_MMA_ROWS * 4u +
					(m * 16u + g + ((lane & 1u) << 3u)) * 4u) >> ((t & 3u) * 8u)) & 255u;
				const uint2 low = *(const uint2 *)(buffer + (m * 16u + g) * LM_CELL_MX_PITCH + c * 8u);
				const uint2 high = *(const uint2 *)(buffer + (m * 16u + g + 8u) * LM_CELL_MX_PITCH + c * 8u);
				const uint32_t a[4] = { low.x, high.x, low.y, high.y };
				#pragma unroll
				for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
				{
					float block[4];
					LmMmaMxf8Mxf4Block(block, a, b[f], scale_a, (uint16_t)0u, scale_bits[f]);
					acc[m][f][0] += block[0];
					acc[m][f][1] += block[1];
					acc[m][f][2] += block[2];
					acc[m][f][3] += block[3];
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
				*(uint32_t *)(args.output_bf16 + (uint64_t)packed * args.output_dimension + (cell_first + f / 2u) * LM_SKINNY_CELL_NEURONS + (f % 2u) * 8u +
					c * 2u) = LmPackBf16Pair(acc[m][f][h * 2u], acc[m][f][h * 2u + 1u]);
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
			LmCellMxRows<LM_CELL_MMA_CHUNKS>(args, segments, mx, column_segment, cells_offset, tile_stride, cell_first, block_first, active, first, last, stage,
				weights, scale_buffer);
		else if ( last - first > 2u * 16u )
			LmCellMxRows<4u>(args, segments, mx, column_segment, cells_offset, tile_stride, cell_first, block_first, active, first, last, stage, weights, scale_buffer);
		else if ( last - first > 16u )
			LmCellMxRows<2u>(args, segments, mx, column_segment, cells_offset, tile_stride, cell_first, block_first, active, first, last, stage, weights, scale_buffer);
		else
			LmCellMxRows<1u>(args, segments, mx, column_segment, cells_offset, tile_stride, cell_first, block_first, active, first, last, stage, weights, scale_buffer);
	}
}

static int32_t LmCellMxExperts(const void *weight, const uint8_t *codes, const uint8_t *code_scales, uint16_t *output_bf16, const uint32_t *group_row_offset,
	const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t activation_packed, uint32_t input_dimension,
	uint32_t output_dimension, cudaStream_t stream)
{
	LmSkinnyCellArguments args;
	LmCellMmaSegments segments;
	LmCellMxCodes mx;
	uint64_t blocks, slots;
	if ( codes == 0 || code_scales == 0 || ((uintptr_t)codes % 16u) != 0u || ((uintptr_t)code_scales % 4u) != 0u ||
		(input_dimension % 128u) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	int32_t status = LmCellMmaLaunchSegments(args, segments, &weight, 1u, (const uint16_t *)codes, output_bf16, group_row_offset, route_source_token, groups,
		packed_rows, activation_packed, input_dimension, output_dimension, LM_SKINNY_CELL_TILE_K);
	if ( status != LM_LAUNCH_OK )
		return(status);
	segments.tiles = args.k_tiles;
	segments.group_bytes = (uint64_t)args.k_tiles * args.cells * LM_CELL_MMA_CELL_BYTES;
	args.group_bytes = segments.group_bytes;
	mx.codes = codes;
	mx.scales = code_scales;
	slots = ((uint64_t)args.packed_rows + (uint64_t)args.groups * LM_CELL_MMA_ROWS - 1u) / ((uint64_t)args.groups * LM_CELL_MMA_ROWS);
	slots = args.k_tiles < LM_CELL_MMA_SLOT_MIN_K_TILES ? 1u : slots < LM_CELL_MMA_ROW_SLOTS ? slots : LM_CELL_MMA_ROW_SLOTS;
	blocks = (uint64_t)args.groups * slots * ((args.cells / LM_CELL_MMA_CELLS + LM_CELL_MMA_WARPS - 1u) / LM_CELL_MMA_WARPS);
	LM_LAUNCH_DEPENDENT((LmCellMxKernel), (uint32_t)blocks, LM_CELL_MMA_THREADS, 0u, stream, args, segments, mx);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
