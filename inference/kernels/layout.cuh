#pragma once


#include <stdint.h>

#define LM_SMEM_STATIC_LIMIT 49152u
#define LM_SMEM_SM_TOTAL 131072u

#define LM_SWIZZLE_CHUNK_BYTES 16u
#define LM_SWIZZLE_SECTOR_BYTES 128u

static __host__ __device__ constexpr uint32_t LmTileBytes(uint32_t rows, uint32_t depth, uint32_t element_bits)
{
	return((rows * depth * element_bits) / 8u);
}

static __host__ __device__ constexpr uint32_t LmSwizzleSpanFor(uint32_t row_pitch_bytes)
{
	return((row_pitch_bytes % 128u) == 0u ? 128u
		: (row_pitch_bytes % 64u) == 0u ? 64u
		: (row_pitch_bytes % 32u) == 0u ? 32u : 0u);
}

static __host__ __device__ constexpr uint32_t LmSwizzleRowSelector(uint32_t row, uint32_t row_pitch_bytes, uint32_t span_bytes)
{
	return(((row * row_pitch_bytes) / LM_SWIZZLE_SECTOR_BYTES)
		% (span_bytes / LM_SWIZZLE_CHUNK_BYTES));
}

static __host__ __device__ constexpr bool LmTileKIsSwizzleable(uint32_t tile_k, uint32_t element_bits)
{
	return(((tile_k * element_bits) % 8u) == 0u
		&& LmSwizzleSpanFor(LmTileBytes(1u,tile_k,element_bits)) != 0u);
}

static __host__ __device__ constexpr bool LmTileKIsTmaLoadable(uint32_t tile_k, uint32_t element_bits, bool swizzled)
{
	return(((tile_k * element_bits) % 8u) == 0u
		&& (swizzled == false || LmTileKIsSwizzleable(tile_k,element_bits)));
}

static __host__ __device__ constexpr uint32_t LmTileSwizzleSpan(uint32_t tile_k, uint32_t element_bits)
{
	return(LmSwizzleSpanFor(LmTileBytes(1u,tile_k,element_bits)));
}

static __host__ __device__ constexpr uint32_t LmPipelineSharedBytes(uint32_t tile_m, uint32_t tile_n, uint32_t tile_k, uint32_t stages, uint32_t element_bits)
{
	return((stages * (LmTileBytes(tile_m,tile_k,element_bits)
		+ LmTileBytes(tile_n,tile_k,element_bits))) + (stages * 8u));
}

static __host__ __device__ constexpr uint32_t LmPipelineSharedBytesSplit(
    uint32_t tile_m,
    uint32_t tile_n,
    uint32_t tile_k,
    uint32_t stages,
    uint32_t activation_bits,
    uint32_t weight_bits)
{
    return (stages *
        (LmTileBytes(tile_m, tile_k, activation_bits) +
         LmTileBytes(tile_n, tile_k, weight_bits))) +
        (stages * 8u);
}
