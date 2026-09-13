#pragma once

// Tile and swizzle geometry. Arithmetic only, stdint only, no CUDA.
//
// Everything here is a function of tile extents and a stored bit width, and a
// host needs all of it: to size a shared-memory request, to pick a tile height
// for a token bucket, to build a TMA descriptor whose swizzle matches what the
// kernel will apply. Keeping it in an instruction header meant a launcher could
// not compute its own plan without pulling in the whole device chain - which is
// exactly what happened, and this file is the fix.
//
// The rule this tree keeps: geometry is host-computable, instructions are not.
// kernels/kv.cuh already followed it; this brings the rest in line.

#include <stdint.h>

// Shared memory. The STATIC __shared__ limit is 48 KB per block, not the 128 KB
// of L1/shared an SM has - ptxas enforces it as "uses too much shared data
// (0xc000 max)". Exceeding it requires dynamic shared plus a runtime opt-in.
#define LM_SMEM_STATIC_LIMIT 49152u
#define LM_SMEM_SM_TOTAL 131072u

#define LM_SWIZZLE_CHUNK_BYTES 16u
#define LM_SWIZZLE_SECTOR_BYTES 128u

static __host__ __device__ constexpr uint32_t LmTileBytes(uint32_t rows, uint32_t depth, uint32_t element_bits)
{
	return((rows * depth * element_bits) / 8u);
}

// The largest span that divides a row pitch. A sub-byte code that is not a power
// of two gives a pitch no large span divides: seven bits over a 256-element tile
// is 224 bytes, which 128 does not divide and 64 does not, but 32 does. Forcing
// 128 there would need a 1024-element tile and 252 KB of shared, rejecting a good
// format for a reason that has nothing to do with the format.
static __host__ __device__ constexpr uint32_t LmSwizzleSpanFor(uint32_t row_pitch_bytes)
{
	return((row_pitch_bytes % 128u) == 0u ? 128u
		: (row_pitch_bytes % 64u) == 0u ? 64u
		: (row_pitch_bytes % 32u) == 0u ? 32u : 0u);
}

// TMA derives the xor selector from the row's 128-byte sector, not from the
// row number. Those are equal only for a 128-byte pitch: 64-byte rows advance
// every two rows and 32-byte rows every four. The old row-mod-span shortcut
// therefore corrupted every sub-128-byte weight tile while BF16/FP8 passed.
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

// Shared bytes one staged pipeline needs, matching what the kernel carves. Both
// the host request and the device carve come from here, which is the only reason
// they cannot disagree.
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
