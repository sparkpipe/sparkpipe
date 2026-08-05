#pragma once

// MXFP4: E2M1 data, E8M0 scale, one scale per 32 elements. Decoded to BF16.
//
// This is the format Kimi K3 ships. config.json's quantization_config gives
// num_bits 4, type float, group_size 32, scale_dtype uint8, symmetric - and the
// ignore list is the part that matters as much as the numbers: attention
// projections, latent MoE projections, shared experts, routers and lm_head are
// NOT quantised. Only the routed experts are 4-bit.
//
// WHY THAT EXCLUSION IS LOAD-BEARING. The report's deployment section says the
// quantisation-aware training ran from SFT onward, so the routed experts were
// trained INTO this grid. Nothing else was. Running INT7 across attention and
// the router - which this tree did - is off-recipe in the same way that
// deriving new tensors and storing them at MXFP4 would be: the grid is not the
// point, the training into the grid is the point.
//
// THE DIFFERENCE FROM NVFP4 IS THE SCALE, NOT THE DATA. Both store E2M1 pairs
// per byte. NVFP4 carries a UE4M3 scale per 16 elements; MXFP4 carries an E8M0
// scale per 32 - a bare exponent, so the scale is a power of two and applying
// it is an exponent add rather than a multiply. Two elements per byte and half
// the scale density means MXFP4 is the cheaper of the two to read, which is
// what a 273 GB/s machine cares about.

#include "inference/kernels/formats/nvfp4.cuh"

// One scale per 32 elements, against NVFP4's 16.
#define LM_MXFP4_GROUP 32u

// E8M0 is a bare exponent with a bias of 127: the value is 2^(code - 127), and
// 0xff is the NaN encoding. Decoding through exp2f rather than a table keeps it
// a single instruction and avoids a load on the hot path.
static __device__ __forceinline__ float LmE8m0ToFloat(uint8_t code)
{
	return(code == 0xffu ? __int_as_float(0x7fffffff) : exp2f((float)code - 127.0f));
}

struct LmMxfp4
{
	static constexpr uint32_t kStoredBits = 4u;
	static constexpr bool kTmaSwizzle = true;
	// Same K tile as NVFP4: the row pitch must be a whole swizzle span in
	// bytes, and that follows from the stored width, which is identical.
	static constexpr uint32_t kTileK = 128u;
	static constexpr uint32_t kBits = 16u;
	static constexpr uint32_t kMmaM = LM_MMA16_M;
	static constexpr uint32_t kMmaN = LM_MMA16_N;
	static constexpr uint32_t kMmaK = LM_MMA16_K;
	static constexpr uint32_t kScaleGroup = LM_MXFP4_GROUP;
	static constexpr float kMax = LM_E2M1_MAX;

	static __device__ __forceinline__ uint32_t OperandARow(uint32_t lane, uint32_t reg) { return(LmMma16OperandARow(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandAK(uint32_t lane, uint32_t reg) { return(LmMma16OperandAK(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandBRow(uint32_t lane) { return(LmMma16OperandBRow(lane)); }
	static __device__ __forceinline__ uint32_t OperandBK(uint32_t lane, uint32_t reg) { return(LmMma16OperandBK(lane,reg)); }

	// The weight-only GEMM fetches scales as the checkpoint packed them and
	// asks the format to price one: for MXFP4 that is a bare exponent.
	static __device__ __forceinline__ float ScaleDecode(uint8_t code)
	{
		return(LmE8m0ToFloat(code));
	}
	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		return((uint8_t)(LmFloatPairToE2m1(value,0.0f) & 15u));
	}
	static __device__ __forceinline__ uint32_t Fragment(const uint8_t *tile, uint32_t row, uint32_t k, uint32_t row_pitch_bytes, float scale)
	{
		float2 pair = LmNvfp4Pair(tile[LmSwizzledOffset(row,k >> 1u,row_pitch_bytes,LmSwizzleSpanFor(row_pitch_bytes))]);
		return(LmPackBf16Pair(pair.x * scale,pair.y * scale));
	}
};
