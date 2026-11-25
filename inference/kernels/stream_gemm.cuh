#pragma once

#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/mxfp8.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/scale.cuh"
#include "runtime/launch.h"
#include <atomic>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include "sparkpipe/spark_gpu.h"

#define LM_STREAM_THREADS 256u
#define LM_STREAM_WARPS 8u
#define LM_STREAM_SCALE_K 128u
#define LM_STREAM_BAND_ROW_TILES 8u
#define LM_STREAM_SHARED_LIMIT 101376u
#define LM_STREAM_STAGE_BUDGET 98304u
#define LM_STREAM_MAX_STAGES 4u
#define LM_STREAM_MAX_DEVICES 64u
#define LM_STREAM_WIDE_ROWS 256u

typedef struct LmStreamGemmArguments
{
	const uint16_t *activation;
	const uint8_t *weight;
	const uint32_t *group_row_offset;
	const uint32_t *source_row_map;
	uint16_t *output_bf16;
	float *output_f32;
	LmScaleTensor scale;
	uint64_t weight_group_bytes;
	uint32_t rows,groups;
	uint32_t input_dimension,output_dimension;
	uint32_t output_row_stride,output_column_offset;
}
LmStreamGemmArguments;

static __device__ __forceinline__ uint32_t LmStreamSwizzle8(uint32_t row, uint32_t chunk)
{
	return(chunk ^ (((row & 1u) << 2u) | ((row >> 1u) & 3u)));
}

static __device__ __forceinline__ void LmStreamLdMatrix2(uint32_t out[2], uint32_t address)
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
		: "=r"(out[0]), "=r"(out[1])
		: "r"(address));
}

static __device__ __forceinline__ void LmStreamLdMatrix4(uint32_t out[4], uint32_t address)
{
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
		: "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
		: "r"(address));
}

static __device__ __forceinline__ uint32_t LmStreamE4m3PairBf16(uint32_t word, uint32_t selector)
{
	uint32_t placed = __byte_perm(word,0u,selector),bits,product;
	bits = ((placed >> 4u) & 0x07f007f0u) | (placed & 0x80008000u);
	asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(product) : "r"(bits), "r"(0x7b807b80u));
	return(product);
}

template<class Format>
struct LmStreamWeight
{
	static constexpr bool kSupported = false;
};

template<>
struct LmStreamWeight<LmBf16Format>
{
	static constexpr bool kSupported = true;
	static constexpr bool kScaled = false;
	static constexpr uint32_t kBlockK = 0u;
	static constexpr uint32_t kElementBytes = 2u;
	template<uint32_t ROW_BYTES>
	static __device__ __forceinline__ uint32_t Offset(uint32_t row, uint32_t chunk)
	{
		return(row * ROW_BYTES + LmStreamSwizzle8(row,chunk) * 16u);
	}
	template<uint32_t ROW_BYTES>
	static __device__ __forceinline__ void Fragment(uint32_t stage_s, uint32_t base_row, uint32_t step, uint32_t lane, uint32_t b[4])
	{
		const uint32_t row = base_row + (lane & 7u),col = step * 4u + ((lane >> 3u) & 3u);
		LmStreamLdMatrix4(b,stage_s + Offset<ROW_BYTES>(row,col));
	}
};

template<>
struct LmStreamWeight<LmFp8>
{
	static constexpr bool kSupported = true;
	static constexpr bool kScaled = true;
	static constexpr uint32_t kScaleK = LM_STREAM_SCALE_K;
	static constexpr uint32_t kScaleEncoding = LM_SCALE_ENCODING_F32;
	static constexpr uint32_t kElementBytes = 1u;
	template<uint32_t ROW_BYTES>
	static __device__ __forceinline__ uint32_t Offset(uint32_t row, uint32_t chunk)
	{
		return(row * ROW_BYTES + LmStreamSwizzle8(row,chunk) * 16u);
	}
	static __device__ __forceinline__ void DecodePair(uint32_t r0, uint32_t r1, uint32_t lane, uint32_t b[4])
	{
		const uint32_t quad = lane & 3u,src_lo = (lane & ~3u) | (quad >> 1u),src_hi = src_lo | 2u;
		const uint32_t sel = (quad & 1u) != 0u ? 0x3424u : 0x1404u;
		b[0] = LmStreamE4m3PairBf16(__shfl_sync(0xffffffffu,r0,src_lo),sel);
		b[1] = LmStreamE4m3PairBf16(__shfl_sync(0xffffffffu,r0,src_hi),sel);
		b[2] = LmStreamE4m3PairBf16(__shfl_sync(0xffffffffu,r1,src_lo),sel);
		b[3] = LmStreamE4m3PairBf16(__shfl_sync(0xffffffffu,r1,src_hi),sel);
	}
	template<uint32_t ROW_BYTES>
	static __device__ __forceinline__ void Fragment(uint32_t stage_s, uint32_t base_row, uint32_t step, uint32_t lane, uint32_t b[4])
	{
		const uint32_t row = base_row + (lane & 7u),col = step * 2u + ((lane >> 3u) & 1u);
		uint32_t raw[2];
		LmStreamLdMatrix2(raw,stage_s + Offset<ROW_BYTES>(row,col));
		DecodePair(raw[0],raw[1],lane,b);
	}
	template<uint32_t ROW_BYTES>
	static __device__ __forceinline__ void FragmentPair(uint32_t stage_s, uint32_t base_row, uint32_t step, uint32_t lane, uint32_t b0[4], uint32_t b1[4])
	{
		const uint32_t row = base_row + (lane & 7u) + (((lane >> 4u) & 1u) * 8u),col = step * 2u + ((lane >> 3u) & 1u);
		uint32_t raw[4];
		LmStreamLdMatrix4(raw,stage_s + Offset<ROW_BYTES>(row,col));
		DecodePair(raw[0],raw[1],lane,b0);
		DecodePair(raw[2],raw[3],lane,b1);
	}
};

template<>
struct LmStreamWeight<LmMxfp8> : LmStreamWeight<LmFp8>
{
	static constexpr uint32_t kScaleK = LM_MXFP8_GROUP;
	static constexpr uint32_t kScaleEncoding = LM_SCALE_ENCODING_UE8M0;
};

struct LmBf16Block32 : LmBf16Format
{
};

template<>
struct LmStreamWeight<LmBf16Block32> : LmStreamWeight<LmBf16Format>
{
	static constexpr uint32_t kBlockK = LM_MXFP8_GROUP;
};

template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K>
static __host__ __device__ constexpr uint32_t LmStreamStageBytes(void)
{
	return(TILE_M * TILE_K * 2u + TILE_N * TILE_K * LmStreamWeight<Format>::kElementBytes);
}

template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K>
static __host__ __device__ constexpr uint32_t LmStreamStages(void)
{
	return(LM_STREAM_STAGE_BUDGET / LmStreamStageBytes<Format,TILE_M,TILE_N,TILE_K>() < LM_STREAM_MAX_STAGES ? LM_STREAM_STAGE_BUDGET / LmStreamStageBytes<Format,TILE_M,TILE_N,TILE_K>() : LM_STREAM_MAX_STAGES);
}

template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K>
static __host__ __device__ constexpr uint32_t LmStreamSharedBytes(uint32_t groups)
{
	return(LmStreamStages<Format,TILE_M,TILE_N,TILE_K>() * LmStreamStageBytes<Format,TILE_M,TILE_N,TILE_K>() + (groups + 1u) * 4u);
}

static __device__ __forceinline__ void LmStreamCopy(uint32_t destination, const void *source, uint32_t bytes)
{
	asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(destination), "l"(source), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void LmStreamCommit(void)
{
	asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<uint32_t PENDING>
static __device__ __forceinline__ void LmStreamWait(void)
{
	asm volatile("cp.async.wait_group %0;\n" :: "n"(PENDING) : "memory");
}

typedef struct LmStreamTile
{
	uint32_t group,row_base,row_limit,neuron_base;
}
LmStreamTile;

static __device__ void LmStreamTilePrefix(const LmStreamGemmArguments &args, uint32_t *prefix, uint32_t groups, uint32_t tile_m, uint32_t neuron_tiles)
{
	uint32_t lane = threadIdx.x & 31u,per_lane,first,group,rows,own,running,other,offset;
	if ( threadIdx.x < 32u )
	{
		per_lane = (groups + 31u) / 32u;
		first = lane * per_lane;
		own = 0u;
		for (group = first; group < first + per_lane && group < groups; group++)
		{
			rows = args.group_row_offset == 0 ? args.rows : args.group_row_offset[group + 1u] - args.group_row_offset[group];
			own += (rows + tile_m - 1u) / tile_m * neuron_tiles;
		}
		running = own;
		for (offset = 1u; offset < 32u; offset <<= 1u)
		{
			other = __shfl_up_sync(0xffffffffu,running,offset);
			if ( lane >= offset )
				running += other;
		}
		offset = running - own;
		for (group = first; group < first + per_lane && group < groups; group++)
		{
			prefix[group] = offset;
			rows = args.group_row_offset == 0 ? args.rows : args.group_row_offset[group + 1u] - args.group_row_offset[group];
			offset += (rows + tile_m - 1u) / tile_m * neuron_tiles;
		}
		if ( lane == 31u )
			prefix[groups] = running;
	}
	__syncthreads();
}

static __device__ __forceinline__ LmStreamTile LmStreamTileAt(const LmStreamGemmArguments &args, const uint32_t *prefix, uint32_t groups, uint32_t tile_m, uint32_t tile_n, uint32_t neuron_tiles, uint32_t tile)
{
	LmStreamTile at;
	uint32_t low = 0u,high = groups,middle,first,row_tiles,band,band_first,band_rows,within,rn;
	if ( args.group_row_offset == 0 )
	{
		at.group = 0u;
		first = 0u;
		at.row_limit = args.rows;
		within = tile;
	}
	else
	{
		while ( low + 1u < high )
		{
			middle = (low + high) >> 1u;
			if ( prefix[middle] <= tile )
				low = middle;
			else
				high = middle;
		}
		at.group = low;
		first = args.group_row_offset[low];
		at.row_limit = args.group_row_offset[low + 1u];
		within = tile - prefix[low];
	}
	row_tiles = (at.row_limit - first + tile_m - 1u) / tile_m;
	if ( args.group_row_offset == 0 && neuron_tiles <= LM_STREAM_BAND_ROW_TILES && row_tiles > neuron_tiles )
	{
		band = within / neuron_tiles;
		rn = within % neuron_tiles;
		if ( (band & 1u) != 0u )
			rn = neuron_tiles - 1u - rn;
		at.row_base = band * tile_m;
		at.neuron_base = rn * tile_n;
		return(at);
	}
	{
		const uint32_t band_width = tile_m >= 256u ? 4u : tile_n >= 256u ? 6u : LM_STREAM_BAND_ROW_TILES;
		band = within / (band_width * neuron_tiles);
		band_first = band * band_width;
		band_rows = row_tiles - band_first < band_width ? row_tiles - band_first : band_width;
		within -= band * band_width * neuron_tiles;
		rn = within / band_rows;
		if ( (band & 1u) != 0u )
			rn = neuron_tiles - 1u - rn;
		at.row_base = first + (band_first + within % band_rows) * tile_m;
		at.neuron_base = rn * tile_n;
	}
	return(at);
}

template<class Format>
static __host__ __device__ constexpr uint32_t LmStreamScaleK(void)
{
	if constexpr ( LmStreamWeight<Format>::kScaled )
		return(LmStreamWeight<Format>::kScaleK);
	else
		return(LM_STREAM_SCALE_K);
}

template<class Format>
static __host__ __device__ constexpr uint32_t LmStreamBlockK(void)
{
	if constexpr ( LmStreamWeight<Format>::kScaled )
		return(LmStreamWeight<Format>::kScaleK);
	else
		return(LmStreamWeight<Format>::kBlockK);
}

template<class Format, uint32_t TILE_M>
static __host__ __device__ constexpr uint32_t LmStreamWarpRows(void)
{
	if constexpr ( LmStreamWeight<Format>::kScaled )
		return(TILE_M > 32u ? TILE_M : 32u);
	else
		return(TILE_M >= 256u ? 64u : TILE_M == 128u && LmStreamBlockK<Format>() != 0u ? 64u : 32u);
}

template<class Format>
static __device__ __forceinline__ float LmStreamScaleLoad(const LmScaleTensor &scale, uint32_t group, uint32_t column, uint32_t k_block)
{
	const uint64_t index = (uint64_t)group * scale.group_stride_entries + (uint64_t)column * scale.row_group_stride_entries + k_block;
	if constexpr ( LmStreamWeight<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32 )
		return(__ldg(((const float *)scale.data) + index));
	else
		return(LmUe8m0ToFloat(__ldg(((const uint8_t *)scale.data) + index)));
}

template<class Format>
static __host__ __device__ constexpr bool LmStreamPack4Ue8m0(void)
{
	if constexpr ( LmStreamWeight<Format>::kScaled )
		return(LmStreamBlockK<Format>() == 32u && LmStreamWeight<Format>::kScaleEncoding == LM_SCALE_ENCODING_UE8M0);
	else
		return(false);
}

static __device__ __forceinline__ uint32_t LmStreamScalePack4(const LmScaleTensor &scale, uint32_t group, uint32_t column, uint32_t k_block)
{
	const uint64_t index = (uint64_t)group * scale.group_stride_entries + (uint64_t)column * scale.row_group_stride_entries + k_block;
	return(__ldg((const uint32_t *)(((const uint8_t *)scale.data) + index)));
}

template<class Format>
static __host__ __device__ constexpr uint32_t LmStreamInputMultiple(void)
{
	if constexpr ( LmStreamWeight<Format>::kScaled )
		return(LM_STREAM_SCALE_K);
	else
		return(16u / LmStreamWeight<Format>::kElementBytes);
}

template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t WARP_ROWS = 32u, bool TAIL = false>
__global__ __launch_bounds__(LM_STREAM_THREADS,1) void LmStreamGemmKernel(const __grid_constant__ LmStreamGemmArguments args)
{
	using Weight = LmStreamWeight<Format>;
	constexpr uint32_t STAGES = LmStreamStages<Format,TILE_M,TILE_N,TILE_K>();
	constexpr uint32_t B_ROW_BYTES = TILE_K * Weight::kElementBytes;
	constexpr uint32_t B_ROW_CHUNKS = B_ROW_BYTES / 16u;
	constexpr uint32_t SCALE_K = LmStreamScaleK<Format>();
	constexpr uint32_t SCALE_BLOCKS = Weight::kScaled ? TILE_K / SCALE_K : 1u;
	constexpr uint32_t BLOCK_K = LmStreamBlockK<Format>();
	constexpr uint32_t A_ROW_BYTES = TILE_K * 2u;
	constexpr uint32_t A_ROW_CHUNKS = A_ROW_BYTES / 16u;
	constexpr uint32_t WARPS_M = TILE_M <= WARP_ROWS ? 1u : TILE_M / WARP_ROWS;
	constexpr uint32_t WARPS_N = LM_STREAM_WARPS / WARPS_M;
	constexpr uint32_t WARP_M = TILE_M / WARPS_M;
	constexpr uint32_t WARP_N = TILE_N / WARPS_N;
	constexpr uint32_t MF = WARP_M / 16u;
	constexpr uint32_t NF = WARP_N / 8u;
	constexpr uint32_t A_BYTES = TILE_M * A_ROW_BYTES;
	constexpr uint32_t STAGE_BYTES = LmStreamStageBytes<Format,TILE_M,TILE_N,TILE_K>();
	constexpr uint32_t A_CHUNKS = TILE_M * A_ROW_CHUNKS;
	constexpr uint32_t A_PER_THREAD = (A_CHUNKS + LM_STREAM_THREADS - 1u) / LM_STREAM_THREADS;
	constexpr uint32_t B_CHUNKS = TILE_N * B_ROW_CHUNKS;
	constexpr uint32_t B_PER_THREAD = B_CHUNKS / LM_STREAM_THREADS;
	constexpr bool USE_LDM = !Weight::kScaled && BLOCK_K == 0u && (NF % 2u == 0u);
	constexpr uint32_t KSTEPS = TILE_K / 16u;
	static_assert(Weight::kSupported,"the stream GEMM has no path for this weight format");
	static_assert(MF >= 1u && NF >= 1u && WARPS_M * WARPS_N == LM_STREAM_WARPS && WARP_M % 16u == 0u,"the tile is not a whole number of MMA fragments per warp");
	static_assert(B_CHUNKS % LM_STREAM_THREADS == 0u && LM_STREAM_THREADS % B_ROW_CHUNKS == 0u,"the weight stage must divide across the block");
	static_assert(STAGES >= 2u,"the tile leaves no room for a second pipeline stage");
	static_assert(!Weight::kScaled || (TILE_K % SCALE_K == 0u && SCALE_K % 32u == 0u),"a k-tile must hold whole weight scale blocks");
	static_assert(BLOCK_K == 0u || (TILE_K % (BLOCK_K != 0u ? BLOCK_K : 1u) == 0u && BLOCK_K % 32u == 0u),"a k-tile must hold whole accumulation blocks");
	static_assert(A_ROW_CHUNKS >= 8u && B_ROW_CHUNKS >= 8u,"the shared-memory swizzle needs tile rows of at least 128 bytes");
	static_assert(!TAIL || !Weight::kScaled,"a K tail exists only for unscaled weights, whose inputs need not be whole scale blocks");
	extern __shared__ __align__(128) uint8_t lm_stream_shared[];
	uint32_t *prefix = (uint32_t *)(lm_stream_shared + STAGES * STAGE_BYTES);
	const uint32_t shared_base = (uint32_t)__cvta_generic_to_shared(lm_stream_shared);
	const uint32_t warp = threadIdx.x >> 5u,lane = threadIdx.x & 31u,quad_row = lane >> 2u,quad = lane & 3u;
	const uint32_t warp_m = warp / WARPS_N,warp_n = warp % WARPS_N;
	const uint32_t groups = args.group_row_offset == 0 ? 1u : args.groups;
	const uint32_t neuron_tiles = (args.output_dimension + TILE_N - 1u) / TILE_N;
	const uint32_t k_tiles = TAIL ? (args.input_dimension + TILE_K - 1u) / TILE_K : args.input_dimension / TILE_K;
	const uint64_t weight_row_bytes = (uint64_t)args.input_dimension * Weight::kElementBytes;
	uint32_t total_tiles;

	if ( args.group_row_offset != 0 )
	{
		LmStreamTilePrefix(args,prefix,groups,TILE_M,neuron_tiles);
		total_tiles = prefix[groups];
	}
	else
		total_tiles = ((args.rows + TILE_M - 1u) / TILE_M) * neuron_tiles;

	auto advance_stage = [](uint32_t &offset)
	{
		if constexpr ( STAGES == 2u )
			offset ^= STAGE_BYTES;
		else
		{
			offset += STAGE_BYTES;
			if ( offset == STAGES * STAGE_BYTES )
				offset = 0u;
		}
	};

	const uint32_t b_row0 = threadIdx.x / B_ROW_CHUNKS,b_col0 = threadIdx.x % B_ROW_CHUNKS;
	const uint32_t b_row_step = LM_STREAM_THREADS / B_ROW_CHUNKS;
	const uint64_t b_ptr_step = (uint64_t)b_row_step * weight_row_bytes;
	const uint16_t *a_source[A_PER_THREAD];
	const uint8_t *b_base = args.weight;
	uint32_t a_mask = 0u,b_mask = 0u;
	float total[MF][NF][4];
	LmStreamTile load,work;
	uint32_t i,j,e;

	auto locate = [&](uint32_t tile_index)
	{
		load = LmStreamTileAt(args,prefix,groups,TILE_M,TILE_N,neuron_tiles,tile_index);
		a_mask = 0u;
		b_mask = 0u;
#pragma unroll
		for (uint32_t q = 0u; q < A_PER_THREAD; q++)
		{
			const uint32_t chunk = threadIdx.x + q * LM_STREAM_THREADS;
			const uint32_t row = chunk / A_ROW_CHUNKS,col = chunk % A_ROW_CHUNKS;
			const uint32_t packed = load.row_base + row;
			const uint32_t valid = chunk < A_CHUNKS && packed < load.row_limit ? 1u : 0u;
			const uint32_t source = valid == 0u ? 0u : (args.source_row_map != 0 ? args.source_row_map[packed] : packed);
			a_source[q] = args.activation + (uint64_t)source * args.input_dimension + col * 8u;
			a_mask |= valid << q;
		}
		b_base = args.weight + (uint64_t)load.group * args.weight_group_bytes + (uint64_t)(load.neuron_base + b_row0) * weight_row_bytes + b_col0 * 16u;
#pragma unroll
		for (uint32_t q = 0u; q < B_PER_THREAD; q++)
			b_mask |= (load.neuron_base + b_row0 + q * b_row_step < args.output_dimension ? 1u : 0u) << q;
	};
	auto issue_k = [&](uint32_t st_off, uint32_t k)
	{
		const uint32_t stage = shared_base + st_off;
#pragma unroll
		for (uint32_t q = 0u; q < A_PER_THREAD; q++)
		{
			const uint32_t chunk = threadIdx.x + q * LM_STREAM_THREADS;
			if ( chunk < A_CHUNKS )
			{
				const uint32_t row = chunk / A_ROW_CHUNKS,col = chunk % A_ROW_CHUNKS;
				const uint32_t dest = stage + row * A_ROW_BYTES + LmStreamSwizzle8(row,col) * 16u;
				const uint32_t bytes = ((a_mask >> q) & 1u) << 4u;
				if constexpr ( TAIL )
				{
					const bool inside = k * TILE_K + col * 8u < args.input_dimension;
					LmStreamCopy(dest,inside ? a_source[q] + (uint64_t)k * TILE_K : args.activation,inside ? bytes : 0u);
				}
				else
				{
					LmStreamCopy(dest,a_source[q],bytes);
					a_source[q] += TILE_K;
				}
			}
		}
#pragma unroll
		for (uint32_t q = 0u; q < B_PER_THREAD; q++)
		{
			const uint32_t row = b_row0 + q * b_row_step;
			const uint32_t dest = stage + A_BYTES + Weight::template Offset<B_ROW_BYTES>(row,b_col0);
			const uint32_t valid = (b_mask >> q) & 1u;
			const uint8_t *src = valid != 0u ? b_base + q * b_ptr_step : args.weight;
			if constexpr ( TAIL )
			{
				const bool inside = valid != 0u && k * TILE_K + b_col0 * (16u / Weight::kElementBytes) < args.input_dimension;
				LmStreamCopy(dest,inside ? src + (uint64_t)k * TILE_K * Weight::kElementBytes : args.weight,inside ? 16u : 0u);
			}
			else
				LmStreamCopy(dest,src,valid << 4u);
		}
		if constexpr ( !TAIL )
			b_base += TILE_K * Weight::kElementBytes;
	};
	auto store_tile = [&]()
	{
#pragma unroll
		for (i = 0u; i < MF; i++)
#pragma unroll
			for (j = 0u; j < NF; j++)
#pragma unroll
				for (e = 0u; e < 4u; e += 2u)
				{
					const uint32_t row = work.row_base + warp_m * WARP_M + i * 16u + quad_row + (e >> 1u) * 8u;
					const uint32_t column = work.neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
					uint64_t at;
					if ( row >= work.row_limit || column >= args.output_dimension )
						continue;
					at = (uint64_t)row * args.output_row_stride + args.output_column_offset + column;
					if ( args.output_f32 != 0 )
						*(float2 *)(args.output_f32 + at) = make_float2(total[i][j][e],total[i][j][e + 1u]);
					else
						*(uint32_t *)(args.output_bf16 + at) = (uint32_t)LmFloatToBf16(total[i][j][e]) | ((uint32_t)LmFloatToBf16(total[i][j][e + 1u]) << 16u);
				}
	};

	if constexpr ( !USE_LDM )
	{
		constexpr bool INLINE_BLOCK32 = (BLOCK_K == 32u);
		constexpr bool PACK4_UE8M0 = LmStreamPack4Ue8m0<Format>();
		const uint32_t ldm_a_row = warp_m * WARP_M + (lane & 15u),ldm_a_col = (lane >> 4u) & 1u;
		for (uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x)
		{
			uint32_t kp = 0u,sp_off = 0u,sc_off = 0u;
			float partial[INLINE_BLOCK32 || BLOCK_K == 0u ? 1u : MF][INLINE_BLOCK32 || BLOCK_K == 0u ? 1u : NF][4];
			locate(tile);
			work = load;
#pragma unroll
			for (i = 0u; i < MF; i++)
#pragma unroll
				for (j = 0u; j < NF; j++)
					total[i][j][0] = total[i][j][1] = total[i][j][2] = total[i][j][3] = 0.0f;
			if constexpr ( !INLINE_BLOCK32 && BLOCK_K != 0u )
			{
#pragma unroll
				for (i = 0u; i < MF; i++)
#pragma unroll
					for (j = 0u; j < NF; j++)
						partial[i][j][0] = partial[i][j][1] = partial[i][j][2] = partial[i][j][3] = 0.0f;
			}
#pragma unroll
			for (uint32_t s = 0u; s + 1u < STAGES; s++)
			{
				if ( kp < k_tiles )
				{
					issue_k(sp_off,kp);
					LmStreamCommit();
					kp++;
					advance_stage(sp_off);
				}
			}
			for (uint32_t kt = 0u; kt < k_tiles; kt++)
			{
				uint32_t stage_a_s,stage_b_s;
				constexpr bool TIGHT_REGS = MF * NF >= 16u && NF >= 4u;
				constexpr bool DEFER_SCALE = Weight::kScaled && !INLINE_BLOCK32 && SCALE_BLOCKS == 1u && !TIGHT_REGS;
				float scale_tile[Weight::kScaled && !INLINE_BLOCK32 && !DEFER_SCALE ? SCALE_BLOCKS : 1u][Weight::kScaled && !INLINE_BLOCK32 && !DEFER_SCALE ? NF : 1u][2];
				uint32_t scale_pack[PACK4_UE8M0 ? SCALE_BLOCKS / 4u : 1u][PACK4_UE8M0 ? NF : 1u][2];
				if constexpr ( STAGES > 2u )
				{
					if ( kp < k_tiles )
						LmStreamWait<STAGES - 2u>();
					else
						LmStreamWait<0u>();
				}
				else
					LmStreamWait<0u>();
				__syncthreads();
				if ( kp < k_tiles )
				{
					issue_k(sp_off,kp);
					LmStreamCommit();
					kp++;
					advance_stage(sp_off);
				}
				if constexpr ( PACK4_UE8M0 )
				{
#pragma unroll
					for (uint32_t p = 0u; p < SCALE_BLOCKS / 4u; p++)
#pragma unroll
						for (j = 0u; j < NF; j++)
						{
							const uint32_t column = work.neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
							const uint32_t k_block = kt * SCALE_BLOCKS + p * 4u;
							scale_pack[p][j][0] = column < args.output_dimension ? LmStreamScalePack4(args.scale,work.group,column,k_block) : 0u;
							scale_pack[p][j][1] = column + 1u < args.output_dimension ? LmStreamScalePack4(args.scale,work.group,column + 1u,k_block) : 0u;
						}
				}
				else if constexpr ( Weight::kScaled && !INLINE_BLOCK32 && !DEFER_SCALE )
				{
#pragma unroll
					for (uint32_t block = 0u; block < SCALE_BLOCKS; block++)
#pragma unroll
						for (j = 0u; j < NF; j++)
						{
							const uint32_t column = work.neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
							const uint32_t k_block = kt * SCALE_BLOCKS + block;
							scale_tile[block][j][0] = column < args.output_dimension ? LmStreamScaleLoad<Format>(args.scale,work.group,column,k_block) : 0.0f;
							scale_tile[block][j][1] = column + 1u < args.output_dimension ? LmStreamScaleLoad<Format>(args.scale,work.group,column + 1u,k_block) : 0.0f;
						}
				}
				stage_a_s = shared_base + sc_off;
				stage_b_s = stage_a_s + A_BYTES;
				if constexpr ( !INLINE_BLOCK32 && BLOCK_K != 0u )
				{
					constexpr uint32_t BLK_STEPS = BLOCK_K / 32u;
#pragma unroll
					for (uint32_t blk = 0u; blk < TILE_K / BLOCK_K; blk++)
					{
						auto blk_step = [&](uint32_t step)
						{
							uint32_t b[NF][4];
							if constexpr ( Weight::kScaled && Weight::kElementBytes == 1u && (NF % 2u) == 0u && !TIGHT_REGS )
							{
#pragma unroll
								for (j = 0u; j < NF; j += 2u)
									Weight::template FragmentPair<B_ROW_BYTES>(stage_b_s,warp_n * WARP_N + j * 8u,step,lane,b[j],b[j + 1u]);
							}
							else
							{
#pragma unroll
								for (j = 0u; j < NF; j++)
									Weight::template Fragment<B_ROW_BYTES>(stage_b_s,warp_n * WARP_N + j * 8u,step,lane,b[j]);
							}
							if constexpr ( !TIGHT_REGS )
							{
								uint32_t a_cur[4],a_nxt[4];
								LmStreamLdMatrix4(a_cur,stage_a_s + ldm_a_row * A_ROW_BYTES + LmStreamSwizzle8(ldm_a_row,step * 4u + ldm_a_col) * 16u);
#pragma unroll
								for (i = 0u; i < MF; i++)
								{
									const uint32_t row = ldm_a_row + i * 16u;
									LmStreamLdMatrix4(a_nxt,stage_a_s + row * A_ROW_BYTES + LmStreamSwizzle8(row,step * 4u + 2u + ldm_a_col) * 16u);
#pragma unroll
									for (j = 0u; j < NF; j++)
									{
										uint32_t b_first[2] = { b[j][0], b[j][1] };
										LmMmaBf16(partial[i][j],a_cur,b_first);
									}
									if constexpr ( MF > 1u )
									{
										if ( i + 1u < MF )
										{
											const uint32_t next_row = ldm_a_row + (i + 1u) * 16u;
											LmStreamLdMatrix4(a_cur,stage_a_s + next_row * A_ROW_BYTES + LmStreamSwizzle8(next_row,step * 4u + ldm_a_col) * 16u);
										}
									}
#pragma unroll
									for (j = 0u; j < NF; j++)
									{
										uint32_t b_second[2] = { b[j][2], b[j][3] };
										LmMmaBf16(partial[i][j],a_nxt,b_second);
									}
								}
							}
							else
							{
#pragma unroll
								for (i = 0u; i < MF; i++)
								{
									const uint32_t row = ldm_a_row + i * 16u;
									uint32_t a_frag[4];
									LmStreamLdMatrix4(a_frag,stage_a_s + row * A_ROW_BYTES + LmStreamSwizzle8(row,step * 4u + ldm_a_col) * 16u);
#pragma unroll
									for (j = 0u; j < NF; j++)
									{
										uint32_t b_first[2] = { b[j][0], b[j][1] };
										LmMmaBf16(partial[i][j],a_frag,b_first);
									}
									LmStreamLdMatrix4(a_frag,stage_a_s + row * A_ROW_BYTES + LmStreamSwizzle8(row,step * 4u + 2u + ldm_a_col) * 16u);
#pragma unroll
									for (j = 0u; j < NF; j++)
									{
										uint32_t b_second[2] = { b[j][2], b[j][3] };
										LmMmaBf16(partial[i][j],a_frag,b_second);
									}
								}
							}
						};
						if constexpr ( TIGHT_REGS )
						{
#pragma unroll 2
							for (uint32_t s = 0u; s < BLK_STEPS; s++)
								blk_step(blk * BLK_STEPS + s);
						}
						else
						{
#pragma unroll
							for (uint32_t s = 0u; s < BLK_STEPS; s++)
								blk_step(blk * BLK_STEPS + s);
						}
						if constexpr ( DEFER_SCALE )
						{
							float sc[NF][2];
#pragma unroll
							for (j = 0u; j < NF; j++)
							{
								const uint32_t column = work.neuron_base + warp_n * WARP_N + j * 8u + quad * 2u;
								sc[j][0] = column < args.output_dimension ? LmStreamScaleLoad<Format>(args.scale,work.group,column,kt) : 0.0f;
								sc[j][1] = column + 1u < args.output_dimension ? LmStreamScaleLoad<Format>(args.scale,work.group,column + 1u,kt) : 0.0f;
							}
#pragma unroll
							for (i = 0u; i < MF; i++)
#pragma unroll
								for (j = 0u; j < NF; j++)
#pragma unroll
									for (e = 0u; e < 4u; e++)
									{
										total[i][j][e] = fmaf(partial[i][j][e],sc[j][e & 1u],total[i][j][e]);
										partial[i][j][e] = 0.0f;
									}
						}
						else
						{
#pragma unroll
							for (i = 0u; i < MF; i++)
#pragma unroll
								for (j = 0u; j < NF; j++)
#pragma unroll
									for (e = 0u; e < 4u; e++)
									{
										if constexpr ( Weight::kScaled )
											total[i][j][e] = fmaf(partial[i][j][e],scale_tile[blk][j][e & 1u],total[i][j][e]);
										else
											total[i][j][e] += partial[i][j][e];
										partial[i][j][e] = 0.0f;
									}
						}
					}
				}
				else if constexpr ( PACK4_UE8M0 )
				{
#pragma unroll
					for (uint32_t blk = 0u; blk < SCALE_BLOCKS / 4u; blk++)
					{
						uint32_t sp[NF][2];
#pragma unroll
						for (j = 0u; j < NF; j++)
						{
							sp[j][0] = scale_pack[blk][j][0];
							sp[j][1] = scale_pack[blk][j][1];
						}
#pragma unroll 1
						for (uint32_t s = 0u; s < 4u; s++)
						{
							const uint32_t step = blk * 4u + s;
							uint32_t b[NF][4];
#pragma unroll
							for (j = 0u; j < NF; j++)
								Weight::template Fragment<B_ROW_BYTES>(stage_b_s,warp_n * WARP_N + j * 8u,step,lane,b[j]);
#pragma unroll
							for (i = 0u; i < MF; i++)
							{
								const uint32_t row = ldm_a_row + i * 16u;
								uint32_t first[4],second[4];
								LmStreamLdMatrix4(first,stage_a_s + row * A_ROW_BYTES + LmStreamSwizzle8(row,step * 4u + ldm_a_col) * 16u);
								LmStreamLdMatrix4(second,stage_a_s + row * A_ROW_BYTES + LmStreamSwizzle8(row,step * 4u + 2u + ldm_a_col) * 16u);
#pragma unroll
								for (j = 0u; j < NF; j++)
								{
									uint32_t b_first[2] = { b[j][0], b[j][1] },b_second[2] = { b[j][2], b[j][3] };
									float p[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
									LmMmaBf16(p,first,b_first);
									LmMmaBf16(p,second,b_second);
#pragma unroll
									for (e = 0u; e < 4u; e++)
										total[i][j][e] = fmaf(p[e],LmUe8m0ToFloat(sp[j][e & 1u] & 0xffu),total[i][j][e]);
								}
							}
#pragma unroll
							for (j = 0u; j < NF; j++)
							{
								sp[j][0] >>= 8u;
								sp[j][1] >>= 8u;
							}
						}
					}
				}
				else
				{
#pragma unroll
					for (uint32_t step = 0u; step < TILE_K / 32u; step++)
					{
						uint32_t b[NF][4];
#pragma unroll
						for (j = 0u; j < NF; j++)
							Weight::template Fragment<B_ROW_BYTES>(stage_b_s,warp_n * WARP_N + j * 8u,step,lane,b[j]);
#pragma unroll
						for (i = 0u; i < MF; i++)
						{
							const uint32_t row = ldm_a_row + i * 16u;
							uint32_t first[4],second[4];
							LmStreamLdMatrix4(first,stage_a_s + row * A_ROW_BYTES + LmStreamSwizzle8(row,step * 4u + ldm_a_col) * 16u);
							LmStreamLdMatrix4(second,stage_a_s + row * A_ROW_BYTES + LmStreamSwizzle8(row,step * 4u + 2u + ldm_a_col) * 16u);
#pragma unroll
							for (j = 0u; j < NF; j++)
							{
								uint32_t b_first[2] = { b[j][0], b[j][1] },b_second[2] = { b[j][2], b[j][3] };
								if constexpr ( INLINE_BLOCK32 )
								{
									float p[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
									LmMmaBf16(p,first,b_first);
									LmMmaBf16(p,second,b_second);
#pragma unroll
									for (e = 0u; e < 4u; e++)
										total[i][j][e] += p[e];
								}
								else
								{
									LmMmaBf16(total[i][j],first,b_first);
									LmMmaBf16(total[i][j],second,b_second);
								}
							}
						}
					}
				}
				advance_stage(sc_off);
			}
			LmStreamWait<0u>();
			store_tile();
			if ( tile + gridDim.x < total_tiles )
				__syncthreads();
		}
		return;
	}

	for (uint32_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x)
	{
		uint32_t a_reg[2][MF][4],b_reg[2][NF][2];
		uint32_t kp = 0u,sp_off = 0u,sc_off = 0u;
		locate(tile);
		work = load;
		auto load_frag = [&](uint32_t buf, uint32_t ks, uint32_t stage_s)
		{
#pragma unroll
			for (uint32_t ii = 0u; ii < MF; ii++)
			{
				const uint32_t r = warp_m * WARP_M + ii * 16u + (lane & 15u);
				const uint32_t c = ks * 2u + ((lane >> 4u) & 1u);
				LmStreamLdMatrix4(a_reg[buf][ii],stage_s + r * A_ROW_BYTES + LmStreamSwizzle8(r,c) * 16u);
			}
#pragma unroll
			for (uint32_t jj = 0u; jj < NF; jj += 2u)
			{
				const uint32_t r = warp_n * WARP_N + jj * 8u + (lane & 7u) + (((lane >> 4u) & 1u) * 8u);
				const uint32_t c = ks * 2u + ((lane >> 3u) & 1u);
				uint32_t tmp[4];
				LmStreamLdMatrix4(tmp,stage_s + A_BYTES + r * B_ROW_BYTES + LmStreamSwizzle8(r,c) * 16u);
				b_reg[buf][jj][0] = tmp[0];
				b_reg[buf][jj][1] = tmp[1];
				b_reg[buf][jj + 1u][0] = tmp[2];
				b_reg[buf][jj + 1u][1] = tmp[3];
			}
		};

#pragma unroll
		for (i = 0u; i < MF; i++)
#pragma unroll
			for (j = 0u; j < NF; j++)
#pragma unroll
				for (e = 0u; e < 4u; e++)
					total[i][j][e] = 0.0f;

#pragma unroll
		for (uint32_t s = 0u; s + 1u < STAGES; s++)
		{
			if ( kp < k_tiles )
				issue_k(sp_off,kp);
			LmStreamCommit();
			kp++;
			advance_stage(sp_off);
		}

		if constexpr ( STAGES >= 3u )
		{
			LmStreamWait<STAGES - 2u>();
			__syncthreads();
			load_frag(0u,0u,shared_base + sc_off);
			for (uint32_t kt = 0u; kt < k_tiles; kt++)
			{
				const uint32_t stage_s = shared_base + sc_off;
#pragma unroll
				for (uint32_t ks = 0u; ks < KSTEPS; ks++)
				{
					const uint32_t cur = ks & 1u,nxt = cur ^ 1u;
					if ( ks == 0u )
					{
						if ( kp < k_tiles )
						{
							issue_k(sp_off,kp);
							kp++;
							advance_stage(sp_off);
						}
						LmStreamCommit();
					}
					if ( ks + 1u < KSTEPS )
						load_frag(nxt,ks + 1u,stage_s);
					else if ( kt + 1u < k_tiles )
					{
						LmStreamWait<STAGES - 2u>();
						__syncthreads();
						advance_stage(sc_off);
						load_frag(nxt,0u,shared_base + sc_off);
					}
#pragma unroll
					for (i = 0u; i < MF; i++)
#pragma unroll
						for (j = 0u; j < NF; j++)
							LmMmaBf16(total[i][j],a_reg[cur][i],b_reg[cur][j]);
				}
			}
		}
		else
		{
			for (uint32_t kt = 0u; kt < k_tiles; kt++)
			{
				LmStreamWait<0u>();
				__syncthreads();
				const uint32_t stage_s = shared_base + sc_off;
				if ( kp < k_tiles )
				{
					issue_k(sp_off,kp);
					kp++;
					advance_stage(sp_off);
				}
				LmStreamCommit();
				load_frag(0u,0u,stage_s);
#pragma unroll
				for (uint32_t ks = 0u; ks < KSTEPS; ks++)
				{
					const uint32_t cur = ks & 1u,nxt = cur ^ 1u;
					if ( ks + 1u < KSTEPS )
						load_frag(nxt,ks + 1u,stage_s);
#pragma unroll
					for (i = 0u; i < MF; i++)
#pragma unroll
						for (j = 0u; j < NF; j++)
							LmMmaBf16(total[i][j],a_reg[cur][i],b_reg[cur][j]);
				}
				advance_stage(sc_off);
			}
		}

		LmStreamWait<0u>();
		store_tile();
		if ( tile + gridDim.x < total_tiles )
			__syncthreads();
	}
}

template<class Format>
static uint32_t LmStreamScaleFits(const LmScaleTensor *scale, uint32_t groups, uint32_t output, uint32_t input)
{
	if constexpr ( !LmStreamWeight<Format>::kScaled )
		return(scale->encoding == LM_SCALE_ENCODING_NONE ? 1u : 0u);
	else
		return(scale->encoding == LmStreamWeight<Format>::kScaleEncoding && scale->reserved == 0u && scale->data != 0 && scale->row_group_size == 1u &&
			scale->k_group_size == LmStreamWeight<Format>::kScaleK && scale->group_count >= groups &&
			scale->row_count >= output && scale->input_dimension >= input ? 1u : 0u);
}

template<class Format>
static int32_t LmStreamGemmValidate(const LmStreamGemmArguments *args, uint32_t multiprocessors)
{
	uint32_t groups;
	if ( args == 0 || args->activation == 0 || args->weight == 0 || (args->output_bf16 == 0) == (args->output_f32 == 0) || multiprocessors == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( args->rows == 0u || args->input_dimension == 0u || args->output_dimension == 0u ||
		args->input_dimension % LmStreamInputMultiple<Format>() != 0u ||
		(args->output_dimension & 1u) != 0u || (args->output_column_offset & 1u) != 0u || (args->output_row_stride & 1u) != 0u ||
		args->output_row_stride < args->output_column_offset + args->output_dimension )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( (((uintptr_t)args->activation | (uintptr_t)args->weight | (uintptr_t)args->weight_group_bytes) & 15u) != 0u ||
		(((uintptr_t)args->output_bf16 | (uintptr_t)args->output_f32) & 7u) != 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( args->group_row_offset == 0 ? args->groups > 1u || args->source_row_map != 0 : args->groups == 0u ||
		(args->groups > 1u && args->weight_group_bytes < (uint64_t)args->output_dimension * args->input_dimension * LmStreamWeight<Format>::kElementBytes) )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	groups = args->group_row_offset == 0 ? 1u : args->groups;
	if ( (groups + 1u) * 4u > LM_STREAM_SHARED_LIMIT - LM_STREAM_STAGE_BUDGET || LmStreamScaleFits<Format>(&args->scale,groups,args->output_dimension,args->input_dimension) == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	return(LM_LAUNCH_OK);
}

template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t WARP_ROWS = 32u, bool TAIL = false>
static int32_t LmStreamGemmLaunchTile(const LmStreamGemmArguments *args, uint64_t tile_bound, uint32_t multiprocessors, SparkGpuStream stream)
{
	static std::atomic<uint64_t> granted(0u);
	static std::atomic<int32_t> resident(0);
	const uint32_t shared = LmStreamSharedBytes<Format,TILE_M,TILE_N,TILE_K>(args->group_row_offset == 0 ? 1u : args->groups);
	uint64_t blocks;
	int device = 0,per_multiprocessor;
	if ( SparkGpuGetDevice(&device) != SPARK_GPU_SUCCESS || device < 0 || (uint32_t)device >= LM_STREAM_MAX_DEVICES )
		SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
	if ( (granted.load() & (UINT64_C(1) << (uint32_t)device)) == 0u )
	{
		if ( SparkGpuFuncSetAttribute((const void *)LmStreamGemmKernel<Format,TILE_M,TILE_N,TILE_K,WARP_ROWS,TAIL>,SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES,(int)LM_STREAM_SHARED_LIMIT) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		if ( SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor,LmStreamGemmKernel<Format,TILE_M,TILE_N,TILE_K,WARP_ROWS,TAIL>,LM_STREAM_THREADS,LM_STREAM_SHARED_LIMIT) != SPARK_GPU_SUCCESS || per_multiprocessor < 1 )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		resident.store(per_multiprocessor);
		granted.fetch_or(UINT64_C(1) << (uint32_t)device);
	}
	if ( args->group_row_offset == 0 )
		blocks = ((uint64_t)args->rows + TILE_M - 1u) / TILE_M * (((uint64_t)args->output_dimension + TILE_N - 1u) / TILE_N);
	else
	{
		blocks = (uint64_t)multiprocessors * (uint64_t)resident.load();
		if ( tile_bound < blocks )
			blocks = tile_bound;
	}
	LM_LAUNCH((LmStreamGemmKernel<Format,TILE_M,TILE_N,TILE_K,WARP_ROWS,TAIL>),(uint32_t)blocks,LM_STREAM_THREADS,shared,stream,*args);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

template<class Format, uint32_t TILE_N, uint32_t TILE_K, bool TAIL = false>
static int32_t LmStreamGemmLaunchShape(const LmStreamGemmArguments *args, uint32_t multiprocessors, SparkGpuStream stream)
{
	const uint32_t groups = args->group_row_offset == 0 ? 1u : args->groups;
	const uint32_t neuron_tiles = (args->output_dimension + TILE_N - 1u) / TILE_N;
	uint32_t mean_rows = (args->rows + groups - 1u) / groups,tile_m;
	tile_m = mean_rows > 64u && LmStreamStages<Format,128u,TILE_N,TILE_K>() >= 3u ? 128u
		: mean_rows > 32u && LmStreamStages<Format,64u,TILE_N,TILE_K>() >= 3u ? 64u
		: mean_rows > 16u ? 32u : 16u;
	while ( args->group_row_offset == 0 && tile_m > 16u && (uint64_t)((args->rows + tile_m - 1u) / tile_m) * neuron_tiles < multiprocessors )
		tile_m >>= 1u;
	const uint64_t tile_bound = ((uint64_t)args->rows / tile_m + groups) * neuron_tiles;
	switch ( tile_m )
	{
		case 16u: return(LmStreamGemmLaunchTile<Format,16u,TILE_N,TILE_K,LmStreamWarpRows<Format,16u>(),TAIL>(args,tile_bound,multiprocessors,stream));
		case 32u: return(LmStreamGemmLaunchTile<Format,32u,TILE_N,TILE_K,LmStreamWarpRows<Format,32u>(),TAIL>(args,tile_bound,multiprocessors,stream));
		case 64u:
			if constexpr ( LmStreamStages<Format,64u,TILE_N,TILE_K>() >= 3u )
				return(LmStreamGemmLaunchTile<Format,64u,TILE_N,TILE_K,LmStreamWarpRows<Format,64u>(),TAIL>(args,tile_bound,multiprocessors,stream));
			SPARK_FAIL(LM_LAUNCH_ERR_TILE);
		default:
			if constexpr ( LmStreamStages<Format,128u,TILE_N,TILE_K>() >= 3u )
				return(LmStreamGemmLaunchTile<Format,128u,TILE_N,TILE_K,LmStreamWarpRows<Format,128u>(),TAIL>(args,tile_bound,multiprocessors,stream));
			SPARK_FAIL(LM_LAUNCH_ERR_TILE);
	}
}

#define LM_STREAM_WIDE_TILE_NONE 0u
#define LM_STREAM_WIDE_TILE_128X128 1u
#define LM_STREAM_WIDE_TILE_64X256 2u
#define LM_STREAM_WIDE_TILE_64X128 3u
#define LM_STREAM_WIDE_TILE_32X256 4u

static uint32_t LmStreamGemmWavePrefer64(uint32_t rows, uint32_t output_dimension, uint32_t multiprocessors)
{
	const uint64_t narrow = ((uint64_t)output_dimension + 127u) / 128u;
	const uint64_t b128 = ((uint64_t)rows + 127u) / 128u * narrow;
	const uint64_t b64 = ((uint64_t)rows + 63u) / 64u * narrow;
	const uint64_t w128 = (b128 + multiprocessors - 1u) / multiprocessors;
	const uint64_t w64 = (b64 + multiprocessors - 1u) / multiprocessors;
	return(w64 * 58u < w128 * 100u ? 1u : 0u);
}

static uint32_t LmStreamGemmWideTile(const LmStreamGemmArguments *args, uint32_t multiprocessors)
{
	const uint32_t groups = args->group_row_offset == 0 ? 1u : args->groups;
	const uint32_t mean_rows = (args->rows + groups - 1u) / groups;
	const uint64_t wide = (args->output_dimension + 255u) / 256u,narrow = (args->output_dimension + 127u) / 128u;
	if ( args->rows < LM_STREAM_WIDE_ROWS )
		return(LM_STREAM_WIDE_TILE_NONE);
	if ( groups > 1u )
		return(mean_rows > 64u && ((uint64_t)args->rows + 127u) / 128u * narrow >= (uint64_t)multiprocessors ? LM_STREAM_WIDE_TILE_128X128 : LM_STREAM_WIDE_TILE_NONE);
	if ( args->output_dimension <= 768u && mean_rows > 32u && ((uint64_t)args->rows + 63u) / 64u * narrow >= (uint64_t)multiprocessors )
		return(LM_STREAM_WIDE_TILE_64X128);
	if ( args->output_dimension > 768u && LmStreamGemmWavePrefer64(args->rows,args->output_dimension,multiprocessors) != 0u )
		return(LM_STREAM_WIDE_TILE_64X128);
	if ( args->output_dimension > 768u && mean_rows > 64u )
		return(LM_STREAM_WIDE_TILE_128X128);
	if ( mean_rows > 32u && args->output_dimension % 256u == 0u && ((uint64_t)args->rows + 63u) / 64u * wide >= 2u * (uint64_t)multiprocessors )
		return(LM_STREAM_WIDE_TILE_64X256);
	if ( mean_rows > 32u && ((uint64_t)args->rows + 63u) / 64u * narrow >= (uint64_t)multiprocessors )
		return(LM_STREAM_WIDE_TILE_64X128);
	if ( mean_rows > 16u && args->output_dimension % 256u == 0u && ((uint64_t)args->rows + 31u) / 32u * wide >= 2u * (uint64_t)multiprocessors )
		return(LM_STREAM_WIDE_TILE_32X256);
	return(LM_STREAM_WIDE_TILE_NONE);
}

template<class Format>
static int32_t LmStreamGemmLaunchWide(const LmStreamGemmArguments *args, uint32_t tile, uint32_t multiprocessors, SparkGpuStream stream)
{
	const uint32_t groups = args->group_row_offset == 0 ? 1u : args->groups;
	const uint64_t wide = (args->output_dimension + 255u) / 256u,narrow = (args->output_dimension + 127u) / 128u;
	switch ( tile )
	{
		case LM_STREAM_WIDE_TILE_128X128:
			return(LmStreamGemmLaunchTile<Format,128u,128u,128u,128u>(args,((uint64_t)args->rows / 128u + groups) * narrow,multiprocessors,stream));
		case LM_STREAM_WIDE_TILE_64X256:
			return(LmStreamGemmLaunchTile<Format,64u,256u,128u,64u>(args,((uint64_t)args->rows / 64u + groups) * wide,multiprocessors,stream));
		case LM_STREAM_WIDE_TILE_64X128:
			return(LmStreamGemmLaunchTile<Format,64u,128u,128u,64u>(args,((uint64_t)args->rows / 64u + groups) * narrow,multiprocessors,stream));
		case LM_STREAM_WIDE_TILE_32X256:
			return(LmStreamGemmLaunchTile<Format,32u,256u,128u,32u>(args,((uint64_t)args->rows / 32u + groups) * wide,multiprocessors,stream));
		default:
			SPARK_FAIL(LM_LAUNCH_ERR_TILE);
	}
}

template<class Format>
static int32_t LmStreamGemmDenseScaledLaunch(const LmStreamGemmArguments *args, uint32_t multiprocessors, SparkGpuStream stream)
{
	const uint64_t n128 = ((uint64_t)args->output_dimension + 127u) / 128u;
	const uint64_t n64 = ((uint64_t)args->output_dimension + 63u) / 64u;
	const uint64_t n32 = ((uint64_t)args->output_dimension + 31u) / 32u;
	const uint64_t lo = (2u * (uint64_t)multiprocessors) / 3u;
	if ( args->rows > 64u && args->rows <= 128u )
	{
		if ( n128 >= lo && n128 <= multiprocessors )
			return(LmStreamGemmLaunchTile<Format,128u,128u,128u,128u>(args,0u,multiprocessors,stream));
		if ( 2u * n128 >= lo && 2u * n128 <= multiprocessors )
			return(LmStreamGemmLaunchTile<Format,64u,128u,128u,64u>(args,0u,multiprocessors,stream));
		return(LmStreamGemmLaunchTile<Format,64u,64u,128u,64u>(args,0u,multiprocessors,stream));
	}
	if ( args->rows > 32u && args->rows <= 64u )
	{
		if ( n128 >= lo && n128 <= multiprocessors )
			return(LmStreamGemmLaunchTile<Format,64u,128u,128u,64u>(args,0u,multiprocessors,stream));
		if ( n64 >= lo && n64 <= multiprocessors )
			return(LmStreamGemmLaunchTile<Format,64u,64u,128u,64u>(args,0u,multiprocessors,stream));
		if ( args->input_dimension % 256u == 0u )
			return(LmStreamGemmLaunchTile<Format,32u,64u,256u,32u>(args,0u,multiprocessors,stream));
		return(LmStreamGemmLaunchTile<Format,64u,64u,128u,64u>(args,0u,multiprocessors,stream));
	}
	if ( args->rows > 16u && args->rows <= 32u )
	{
		if ( args->input_dimension <= 1024u && n32 >= multiprocessors )
			return(LmStreamGemmLaunchTile<Format,32u,32u,128u,16u>(args,0u,multiprocessors,stream));
		if ( args->input_dimension % 256u == 0u )
			return(LmStreamGemmLaunchTile<Format,32u,64u,256u,32u>(args,0u,multiprocessors,stream));
		return(LmStreamGemmLaunchTile<Format,32u,64u,128u,32u>(args,0u,multiprocessors,stream));
	}
	return(LM_LAUNCH_ERR_TILE);
}

template<class Format>
static int32_t LmStreamGemmDenseBf16Launch(const LmStreamGemmArguments *args, uint32_t multiprocessors, SparkGpuStream stream)
{
	const uint64_t n128 = ((uint64_t)args->output_dimension + 127u) / 128u;
	const uint64_t lo = (2u * (uint64_t)multiprocessors) / 3u;
	if ( args->rows >= LM_STREAM_WIDE_ROWS )
	{
		const uint64_t wide = ((uint64_t)args->output_dimension + 255u) / 256u;
		if ( args->output_dimension <= 768u && ((uint64_t)args->rows + 63u) / 64u * n128 >= 2u * (uint64_t)multiprocessors )
			return(LmStreamGemmLaunchTile<Format,64u,128u,64u,32u>(args,0u,multiprocessors,stream));
		if ( args->rows >= 1024u && args->output_dimension >= 1024u && args->input_dimension >= 2048u )
		{
			const uint64_t tiles_128x256 = ((uint64_t)args->rows + 127u) / 128u * wide;
			const uint64_t tiles_256x128 = ((uint64_t)args->rows + 255u) / 256u * n128;
			if ( args->output_dimension % 256u == 0u && args->output_dimension < args->rows &&
				tiles_128x256 >= 2u * (uint64_t)multiprocessors && tiles_128x256 % multiprocessors == 0u )
				return(LmStreamGemmLaunchTile<Format,128u,256u,64u,64u>(args,0u,multiprocessors,stream));
			if ( tiles_256x128 >= 2u * (uint64_t)multiprocessors )
				return(LmStreamGemmLaunchTile<Format,256u,128u,64u,64u>(args,0u,multiprocessors,stream));
		}
		if ( args->output_dimension > 768u )
		{
			if ( LmStreamGemmWavePrefer64(args->rows,args->output_dimension,multiprocessors) != 0u )
				return(LmStreamGemmLaunchTile<Format,64u,128u,64u,32u>(args,0u,multiprocessors,stream));
			return(LmStreamGemmLaunchTile<Format,128u,128u,64u,32u>(args,0u,multiprocessors,stream));
		}
		return(LM_LAUNCH_ERR_TILE);
	}
	if ( args->output_dimension < 512u )
		return(LM_LAUNCH_ERR_TILE);
	if ( args->rows > 64u && args->rows <= 128u )
	{
		if ( 2u * n128 >= lo && 2u * n128 <= multiprocessors )
			return(LmStreamGemmLaunchTile<Format,64u,128u,64u,32u>(args,0u,multiprocessors,stream));
		return(LmStreamGemmLaunchTile<Format,64u,64u,64u,32u>(args,0u,multiprocessors,stream));
	}
	if ( args->rows > 32u && args->rows <= 64u )
	{
		if ( n128 >= lo )
			return(LmStreamGemmLaunchTile<Format,64u,128u,64u,32u>(args,0u,multiprocessors,stream));
		return(LmStreamGemmLaunchTile<Format,64u,64u,64u,32u>(args,0u,multiprocessors,stream));
	}
	if ( args->rows > 16u && args->rows <= 32u )
		return(LmStreamGemmLaunchTile<Format,32u,64u,64u,32u>(args,0u,multiprocessors,stream));
	return(LM_LAUNCH_ERR_TILE);
}

template<class Format>
static int32_t LmStreamGemmLaunch(const LmStreamGemmArguments *args, uint32_t multiprocessors, SparkGpuStream stream)
{
	int32_t status = LmStreamGemmValidate<Format>(args,multiprocessors);
	if ( status != LM_LAUNCH_OK )
		return(status);
	if constexpr ( LmStreamWeight<Format>::kScaled )
	{
		if constexpr ( LmStreamWeight<Format>::kElementBytes == 1u &&
			(LmStreamWeight<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32 || LmStreamWeight<Format>::kScaleEncoding == LM_SCALE_ENCODING_UE8M0) )
		{
			const uint32_t tile = LmStreamGemmWideTile(args,multiprocessors);
			if ( tile != LM_STREAM_WIDE_TILE_NONE )
				return(LmStreamGemmLaunchWide<Format>(args,tile,multiprocessors,stream));
			if ( args->group_row_offset == 0 && args->rows > 16u && args->rows <= 128u && args->output_dimension >= 512u )
				return(LmStreamGemmDenseScaledLaunch<Format>(args,multiprocessors,stream));
		}
		const uint32_t groups = args->group_row_offset == 0 ? 1u : args->groups;
		const uint32_t mean_rows = (args->rows + groups - 1u) / groups;
		const uint64_t tiles_64x128 = ((uint64_t)args->rows + 63u) / 64u * ((args->output_dimension + 127u) / 128u);
		if ( (mean_rows <= 32u || (groups == 1u && tiles_64x128 <= multiprocessors)) && args->input_dimension % 256u == 0u )
			return(LmStreamGemmLaunchShape<Format,64u,256u>(args,multiprocessors,stream));
		return(LmStreamGemmLaunchShape<Format,128u,128u>(args,multiprocessors,stream));
	}
	else if constexpr ( LmStreamBlockK<Format>() != 0u )
	{
		if ( args->output_dimension <= 512u && args->input_dimension % 128u == 0u )
			return(LmStreamGemmLaunchShape<Format,64u,128u>(args,multiprocessors,stream));
		if ( args->input_dimension % 64u != 0u )
			return(LmStreamGemmLaunchShape<Format,128u,64u,true>(args,multiprocessors,stream));
		return(LmStreamGemmLaunchShape<Format,128u,64u>(args,multiprocessors,stream));
	}
	else
	{
		if ( args->input_dimension % 64u != 0u )
			return(LmStreamGemmLaunchShape<Format,128u,64u,true>(args,multiprocessors,stream));
		if ( args->group_row_offset == 0 )
		{
			status = LmStreamGemmDenseBf16Launch<Format>(args,multiprocessors,stream);
			if ( status != LM_LAUNCH_ERR_TILE )
				return(status);
		}
		return(LmStreamGemmLaunchShape<Format,128u,64u>(args,multiprocessors,stream));
	}
}

template<class Format>
static int32_t LmStreamGemmGrouped(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *source_row_map, uint32_t groups, uint32_t packed_rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors, SparkGpuStream stream)
{
	LmStreamGemmArguments args;
	memset(&args,0,sizeof(args));
	args.activation = activation;
	args.weight = (const uint8_t *)weight;
	args.group_row_offset = group_row_offset;
	args.source_row_map = source_row_map;
	args.output_bf16 = output_bf16;
	args.scale = scale;
	args.weight_group_bytes = (uint64_t)output_dimension * input_dimension * LmStreamWeight<Format>::kElementBytes;
	args.rows = packed_rows;
	args.groups = groups;
	args.input_dimension = input_dimension;
	args.output_dimension = output_dimension;
	args.output_row_stride = output_dimension;
	return(LmStreamGemmLaunch<Format>(&args,multiprocessors,stream));
}

template<class Format>
static int32_t LmStreamGemmDense(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, float *output_f32, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset, uint32_t multiprocessors, SparkGpuStream stream)
{
	LmStreamGemmArguments args;
	memset(&args,0,sizeof(args));
	args.activation = activation;
	args.weight = (const uint8_t *)weight;
	args.output_bf16 = output_bf16;
	args.output_f32 = output_f32;
	args.scale = scale;
	args.weight_group_bytes = (uint64_t)output_dimension * input_dimension * LmStreamWeight<Format>::kElementBytes;
	args.rows = rows;
	args.groups = 1u;
	args.input_dimension = input_dimension;
	args.output_dimension = output_dimension;
	args.output_row_stride = output_row_stride != 0u ? output_row_stride : output_dimension;
	args.output_column_offset = output_column_offset;
	return(LmStreamGemmLaunch<Format>(&args,multiprocessors,stream));
}
