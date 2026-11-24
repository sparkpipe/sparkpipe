#pragma once
#include <atomic>
#include "inference/kernels/cell_mma.cuh"
#include "sparkpipe/spark_gpu.h"

#define LM_CELL_MX_WARPS_M 4u
#define LM_CELL_MX_WARPS_N 2u
#define LM_CELL_MX_WARPS (LM_CELL_MX_WARPS_M * LM_CELL_MX_WARPS_N)
#define LM_CELL_MX_THREADS (LM_CELL_MX_WARPS * LM_WARP_LANES)
#define LM_CELL_MX_BLOCK_CELLS (LM_CELL_MX_WARPS_N * LM_CELL_MMA_CELLS)
#define LM_CELL_MX_WARP_CHUNKS 2u
#define LM_CELL_MX_ROWS (LM_CELL_MX_WARPS_M * LM_CELL_MX_WARP_CHUNKS * 16u)
#define LM_CELL_MX_TILE_K 128u
#define LM_CELL_MX_K_STEPS (LM_CELL_MX_TILE_K / LM_SKINNY_CELL_TILE_K)
#define LM_CELL_MX_STAGES 4u
#define LM_CELL_MX_TILE_VECTORS (LM_CELL_MX_BLOCK_CELLS * LM_CELL_MMA_CELL_VECTORS)
#define LM_CELL_MX_TILE_BYTES (LM_CELL_MX_TILE_VECTORS * 16u)
#define LM_CELL_MX_A_BYTES (LM_CELL_MX_ROWS * LM_CELL_MX_TILE_K)
#define LM_CELL_MX_W_BYTES (LM_CELL_MX_K_STEPS * LM_CELL_MX_TILE_BYTES)
#define LM_CELL_MX_W_VECTORS (LM_CELL_MX_W_BYTES / 16u)
#define LM_CELL_MX_W_PASSES ((LM_CELL_MX_W_VECTORS + LM_CELL_MX_THREADS - 1u) / LM_CELL_MX_THREADS)
#define LM_CELL_MX_S_BYTES (LM_CELL_MX_ROWS * LM_CELL_MX_K_STEPS)
#define LM_CELL_MX_STAGE_BYTES (LM_CELL_MX_A_BYTES + LM_CELL_MX_W_BYTES + LM_CELL_MX_S_BYTES)
#define LM_CELL_MX_SHARED_BYTES (LM_CELL_MX_STAGES * LM_CELL_MX_STAGE_BYTES)
#define LM_CELL_MX_MAX_DEVICES 64u

static_assert(LM_SKINNY_CELL_TILE_K == 32u && LM_CELL_MX_K_STEPS == 4u && LM_CELL_MX_W_PASSES == 2u,
	"four 32-wide blocks per stage match a four-byte scale word and four cells copy in two passes");
static_assert(LM_CELL_MX_SHARED_BYTES <= 101376u, "four stages fit in SM shared memory");

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

static __device__ __forceinline__ uint2 LmCellMxShared64(uint32_t address)
{
	uint2 value;
	asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n" : "=r"(value.x), "=r"(value.y) : "r"(address));
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

static __device__ __forceinline__ uint32_t LmCellMxSwizzle(uint32_t row, uint32_t chunk)
{
	return(row * LM_CELL_MX_TILE_K + ((chunk ^ (row & 7u)) << 4u));
}

static __device__ __forceinline__ void LmCellMxE2m1(uint32_t word, uint32_t &low, uint32_t &high)
{
	const uint32_t even = (word << 2u) & 0x3c3c3c3cu, odd = (word >> 2u) & 0x3c3c3c3cu;
	low = __byte_perm(even, odd, 0x5140u);
	high = __byte_perm(even, odd, 0x7362u);
}

template<uint32_t WARP_CHUNKS>
static __device__ __forceinline__ void LmCellMxRows(const LmSkinnyCellArguments &args, const LmCellMxCodes &mx, const uint8_t *cells, uint64_t tile_stride,
	uint32_t cell_first, uint32_t block_first, uint32_t warp_m, uint32_t warp_n, uint32_t n_active, uint32_t first, uint32_t end, uint32_t shared_base)
{
	constexpr uint32_t ROWS = LM_CELL_MX_WARPS_M * WARP_CHUNKS * 16u;
	constexpr uint32_t COPIES = (ROWS * (LM_CELL_MX_TILE_K / 16u)) / LM_CELL_MX_THREADS;
	const uint32_t lane = threadIdx.x & 31u, g = lane >> 2u, c = lane & 3u;
	const uint32_t chunk = threadIdx.x & 7u, row_base = threadIdx.x >> 3u;
	const uint32_t block_cells = args.cells - block_first < LM_CELL_MX_BLOCK_CELLS ? args.cells - block_first : LM_CELL_MX_BLOCK_CELLS;
	const uint32_t weight_vectors = block_cells * LM_CELL_MMA_CELL_VECTORS;
	const uint32_t warp_w_offset = warp_n * LM_CELL_MMA_CELLS * LM_CELL_MMA_CELL_BYTES;
	const uint32_t warp_row_base = warp_m * (WARP_CHUNKS * 16u);
	const uint32_t active = n_active != 0u && first + warp_row_base < end ? 1u : 0u;
	const uint32_t k_steps = args.k_tiles / LM_CELL_MX_K_STEPS;
	const uint64_t k_stride = (uint64_t)LM_CELL_MX_K_STEPS * tile_stride;
	float acc[WARP_CHUNKS][LM_CELL_MMA_FRAGS][4];
	const uint8_t *a_src[COPIES], *w_src[LM_CELL_MX_W_PASSES];
	uint32_t w_bytes[LM_CELL_MX_W_PASSES];
	const uint8_t *scale_source = threadIdx.x < ROWS && first + threadIdx.x < end ?
		mx.scales + (uint64_t)(args.activation_packed != 0u ? first + threadIdx.x : args.route_source_token[first + threadIdx.x]) * mx.scale_stride :
		(const uint8_t *)0;
	uint32_t m, h, f;
	#pragma unroll
	for ( m = 0u; m < WARP_CHUNKS; m++ )
	{
		#pragma unroll
		for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
			acc[m][f][0] = acc[m][f][1] = acc[m][f][2] = acc[m][f][3] = 0.0f;
	}
	#pragma unroll
	for ( uint32_t q = 0u; q < COPIES; q++ )
	{
		const uint32_t packed = first + row_base + q * 32u;
		a_src[q] = packed < end ?
			mx.codes + (uint64_t)(args.activation_packed != 0u ? packed : args.route_source_token[packed]) * mx.code_stride + chunk * 16u :
			(const uint8_t *)0;
	}
	#pragma unroll
	for ( uint32_t p = 0u; p < LM_CELL_MX_W_PASSES; p++ )
	{
		const uint32_t v = threadIdx.x + p * LM_CELL_MX_THREADS;
		if ( v < LM_CELL_MX_W_VECTORS )
		{
			const uint32_t kt = v / LM_CELL_MX_TILE_VECTORS, in_kt = v % LM_CELL_MX_TILE_VECTORS;
			w_bytes[p] = in_kt < weight_vectors ? 16u : 0u;
			w_src[p] = in_kt < weight_vectors ? cells + (uint64_t)kt * tile_stride + in_kt * 16u : cells;
		}
		else
		{
			w_bytes[p] = 0u;
			w_src[p] = cells;
		}
	}
	auto produce = [&](uint32_t k_step, uint32_t slot)
	{
		const uint32_t stage_s = shared_base + slot * LM_CELL_MX_STAGE_BYTES;
		const uint32_t w_stage = stage_s + LM_CELL_MX_A_BYTES;
		const uint32_t s_stage = w_stage + LM_CELL_MX_W_BYTES;
		const uint64_t a_off = (uint64_t)k_step * LM_CELL_MX_TILE_K;
		const uint64_t w_off = (uint64_t)k_step * k_stride;
		if ( threadIdx.x < ROWS )
			LmCellMxCopyWord(s_stage + threadIdx.x * 4u, scale_source != 0 ? scale_source + (uint64_t)k_step * LM_CELL_MX_K_STEPS : mx.scales,
				scale_source != 0 ? 4u : 0u);
		#pragma unroll
		for ( uint32_t q = 0u; q < COPIES; q++ )
			LmCellMxCopy(stage_s + LmCellMxSwizzle(row_base + q * 32u, chunk), a_src[q] != 0 ? a_src[q] + a_off : mx.codes, a_src[q] != 0 ? 16u : 0u);
		LmCellMxCopy(w_stage + threadIdx.x * 16u, w_bytes[0] != 0u ? w_src[0] + w_off : cells, w_bytes[0]);
		if ( threadIdx.x + LM_CELL_MX_THREADS < LM_CELL_MX_W_VECTORS )
			LmCellMxCopy(w_stage + (threadIdx.x + LM_CELL_MX_THREADS) * 16u, w_bytes[1] != 0u ? w_src[1] + w_off : cells, w_bytes[1]);
	};
	#pragma unroll
	for ( uint32_t s = 0u; s + 1u < LM_CELL_MX_STAGES; s++ )
	{
		if ( s < k_steps )
			produce(s, s);
		LmCellMmaCommit();
	}
	const uint32_t s_row_offset = (warp_row_base + g + ((lane & 1u) << 3u)) * 4u;
	const uint32_t in_chunk = (c & 1u) * 8u, c_half = c >> 1u;
	const uint32_t w_lane_off = warp_w_offset + lane * 4u;
	const uint32_t ws_lane_off = warp_w_offset + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + g;
	for ( uint32_t k_step = 0u; k_step < k_steps; k_step++ )
	{
		const uint32_t slot = k_step % LM_CELL_MX_STAGES, ahead = k_step + LM_CELL_MX_STAGES - 1u;
		LmCellMmaWait<LM_CELL_MX_STAGES - 2u>();
		__syncthreads();
		if ( ahead < k_steps )
			produce(ahead, ahead % LM_CELL_MX_STAGES);
		LmCellMmaCommit();
		if ( active != 0u )
		{
			const uint32_t stage_s = shared_base + slot * LM_CELL_MX_STAGE_BYTES;
			const uint32_t w_stage = stage_s + LM_CELL_MX_A_BYTES;
			const uint32_t s_lane = w_stage + LM_CELL_MX_W_BYTES + s_row_offset;
			uint32_t scale_word[WARP_CHUNKS];
			#pragma unroll
			for ( m = 0u; m < WARP_CHUNKS; m++ )
				scale_word[m] = LmCellMxShared32(s_lane + m * 16u * 4u);
			#pragma unroll
			for ( uint32_t step = 0u; step < LM_CELL_MX_K_STEPS; step++ )
			{
				const uint32_t w_tile = w_stage + step * LM_CELL_MX_TILE_BYTES + w_lane_off;
				const uint32_t ws_tile = w_stage + step * LM_CELL_MX_TILE_BYTES + ws_lane_off;
				const uint32_t a_chunk = step * 2u + c_half;
				uint32_t b[LM_CELL_MMA_FRAGS][2], scale_b[LM_CELL_MMA_FRAGS];
				#pragma unroll
				for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
				{
					LmCellMxE2m1(LmCellMxShared32(w_tile + (f / 2u) * LM_CELL_MMA_CELL_BYTES + (f % 2u) * 8u * LM_SKINNY_CHUNK_BYTES), b[f][0], b[f][1]);
					scale_b[f] = LmCellMxShared8(ws_tile + (f / 2u) * LM_CELL_MMA_CELL_BYTES + (f % 2u) * 8u);
				}
				#pragma unroll
				for ( m = 0u; m < WARP_CHUNKS; m++ )
				{
					const uint32_t r_low = warp_row_base + m * 16u + g;
					const uint2 lo = LmCellMxShared64(stage_s + LmCellMxSwizzle(r_low, a_chunk) + in_chunk);
					const uint2 hi = LmCellMxShared64(stage_s + LmCellMxSwizzle(r_low + 8u, a_chunk) + in_chunk);
					const uint32_t a[4] = { lo.x, hi.x, lo.y, hi.y };
					#pragma unroll
					for ( f = 0u; f < LM_CELL_MMA_FRAGS; f++ )
					{
						float block[4];
						if ( step == 0u )
							LmMmaMxf8Mxf4Block<0u>(block, a, b[f], scale_word[m], scale_b[f]);
						else if ( step == 1u )
							LmMmaMxf8Mxf4Block<1u>(block, a, b[f], scale_word[m], scale_b[f]);
						else if ( step == 2u )
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
	for ( m = 0u; m < WARP_CHUNKS; m++ )
	{
		#pragma unroll
		for ( h = 0u; h < 2u; h++ )
		{
			const uint32_t packed = first + warp_row_base + m * 16u + g + h * 8u;
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

static __global__ __launch_bounds__(LM_CELL_MX_THREADS, 1) void LmCellMxKernel(const __grid_constant__ LmSkinnyCellArguments args,
	const __grid_constant__ LmCellMmaSegments segments, const __grid_constant__ LmCellMxCodes mx)
{
	extern __shared__ __align__(128) uint8_t lm_cell_mx_shared[];
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(lm_cell_mx_shared);
	const uint32_t pairs = args.cells / LM_CELL_MMA_CELLS;
	const uint32_t blocks = (pairs + LM_CELL_MX_WARPS_N - 1u) / LM_CELL_MX_WARPS_N;
	const uint32_t slots = gridDim.x / (args.groups * blocks);
	const uint32_t group = blockIdx.x / (blocks * slots);
	const uint32_t n_block = (blockIdx.x % (blocks * slots)) / slots;
	const uint32_t slot = blockIdx.x % slots;
	const uint32_t warp = threadIdx.x >> 5u;
	const uint32_t warp_m = warp / LM_CELL_MX_WARPS_N, warp_n = warp % LM_CELL_MX_WARPS_N;
	const uint32_t pair = n_block * LM_CELL_MX_WARPS_N + warp_n;
	const uint32_t n_active = pair < pairs ? 1u : 0u;
	const uint32_t cell_first = (n_active != 0u ? pair : 0u) * LM_CELL_MMA_CELLS;
	const uint32_t block_first = n_block * LM_CELL_MX_BLOCK_CELLS;
	const uint64_t tile_stride = (uint64_t)args.cells * LM_CELL_MMA_CELL_BYTES;
	const uint8_t *cells = segments.weight[0] + LmCellMmaGroupOffset(segments, group) + (uint64_t)block_first * LM_CELL_MMA_CELL_BYTES;
	uint32_t first, end, last;
	LmDependentWait();
	if ( group >= args.groups )
	{
		LmDependentRelease();
		return;
	}
	end = args.group_row_offset[group + 1u];
	for ( first = args.group_row_offset[group] + slot * LM_CELL_MX_ROWS; first < end; first += slots * LM_CELL_MX_ROWS )
	{
		last = end - first < LM_CELL_MX_ROWS ? end : first + LM_CELL_MX_ROWS;
		if ( last - first > 64u )
			LmCellMxRows<LM_CELL_MX_WARP_CHUNKS>(args, mx, cells, tile_stride, cell_first, block_first, warp_m, warp_n, n_active, first, last, shared_base);
		else
			LmCellMxRows<1u>(args, mx, cells, tile_stride, cell_first, block_first, warp_m, warp_n, n_active, first, last, shared_base);
	}
	LmDependentRelease();
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
	static std::atomic<uint64_t> granted(0u);
	LmSkinnyCellArguments args;
	LmCellMmaSegments segments;
	uint64_t blocks, slots;
	int device = 0;
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
	if ( SparkGpuGetDevice(&device) != SPARK_GPU_SUCCESS || device < 0 || (uint32_t)device >= LM_CELL_MX_MAX_DEVICES )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	if ( (granted.load() & (UINT64_C(1) << (uint32_t)device)) == 0u )
	{
		if ( SparkGpuFuncSetAttribute((const void *)LmCellMxKernel, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)LM_CELL_MX_SHARED_BYTES) != SPARK_GPU_SUCCESS )
			return(LM_LAUNCH_ERR_ATTRIBUTE);
		granted.fetch_or(UINT64_C(1) << (uint32_t)device);
	}
	segments.tiles = args.k_tiles;
	segments.group_bytes = (uint64_t)args.k_tiles * args.cells * LM_CELL_MMA_CELL_BYTES;
	segments.split = upper != 0 ? split : 0u;
	segments.upper_offset = upper != 0 ? (uint64_t)((uintptr_t)upper - (uintptr_t)weight) : 0u;
	args.group_bytes = segments.group_bytes;
	slots = ((uint64_t)args.packed_rows + (uint64_t)args.groups * LM_CELL_MX_ROWS - 1u) / ((uint64_t)args.groups * LM_CELL_MX_ROWS);
	slots = args.k_tiles < LM_CELL_MMA_SLOT_MIN_K_TILES ? 1u : slots < LM_CELL_MMA_ROW_SLOTS ? slots : LM_CELL_MMA_ROW_SLOTS;
	blocks = (uint64_t)args.groups * slots * ((args.cells / LM_CELL_MMA_CELLS + LM_CELL_MX_WARPS_N - 1u) / LM_CELL_MX_WARPS_N);
	LM_LAUNCH_DEPENDENT((LmCellMxKernel), (uint32_t)blocks, LM_CELL_MX_THREADS, LM_CELL_MX_SHARED_BYTES, stream, args, segments, mx);
	return(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static int32_t LmCellMxExperts(const void *weight, const uint8_t *codes, const uint8_t *code_scales, uint16_t *output_bf16, const uint32_t *group_row_offset,
	const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t activation_packed, uint32_t input_dimension,
	uint32_t output_dimension, SparkGpuStream stream)
{
	return(LmCellMxExpertsSplit(weight, 0, 0u, LmCellMxPacked(codes, code_scales, input_dimension), output_bf16, group_row_offset, route_source_token,
		groups, packed_rows, activation_packed, input_dimension, output_dimension, stream));
}
