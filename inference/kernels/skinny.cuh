#pragma once

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/weight_prefetch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/formats/fp8.cuh"
#include "inference/kernels/formats/mxfp4.cuh"
#include "inference/kernels/formats/mxfp8.cuh"
#include "inference/kernels/formats/nvfp4.cuh"
#include "inference/kernels/scale.cuh"
#include "runtime/launch.h"
#include <atomic>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include <type_traits>
#include "sparkpipe/spark_gpu.h"

#define LM_SKINNY_ROWS 8u
#define LM_SKINNY_ROWS_MID 4u
#define LM_SKINNY_ROWS_WIDE 16u
#define LM_SKINNY_THREADS 256u
#define LM_SKINNY_CHUNK_BYTES 16u
#define LM_SKINNY_GROUPED_MAX_MEAN_ROWS 16u
#define LM_SKINNY_GROUPED_NEURONS 4u
#define LM_SKINNY_GROUPED_MAX_GROUPS 1024u
#define LM_SKINNY_ROWS_MIN_TASKS 256u
#define LM_SKINNY_FRAME_DEPTH 4u
#define LM_SKINNY_FRAME_MIN_BLOCKS 2u
#define LM_SKINNY_FRAME_WARPS_PER_SM 8u

typedef struct LmSkinnyArguments
{
	const uint8_t *weight;
	const uint16_t *activation;
	uint16_t *output_bf16;
	float *output_f32;
	const uint32_t *route_expert;
	const uint32_t *route_packed_row;
	const uint32_t *group_row_offset;
	const uint32_t *route_source_token;
	LmScaleTensor scale;
	uint64_t weight_group_bytes;
	uint32_t rows,pairs,top_k,activation_packed,groups;
	uint32_t input_dimension,output_dimension,output_row_stride,output_column_offset;
	const uint32_t *skip_sequence_of_row;
	const uint32_t *skip_context_length;
	const uint32_t *skip_row_position;
	uint32_t skip_max_context;
}
LmSkinnyArguments;

static __device__ __forceinline__ uint32_t LmSkinnyShouldSkip(const LmSkinnyArguments &args)
{
	if ( args.skip_max_context == 0u || args.rows != 1u || args.skip_sequence_of_row == 0 || args.skip_context_length == 0 )
		return(0u);
	uint32_t bound = args.skip_context_length[args.skip_sequence_of_row[0]];
	if ( args.skip_row_position != 0 && args.skip_row_position[0] + 1u < bound )
		bound = args.skip_row_position[0] + 1u;
	return(bound <= args.skip_max_context ? 1u : 0u);
}

static __device__ __forceinline__ void LmSkinnyUnpackBf16x8(uint4 packed, float *f)
{
	f[0] = __uint_as_float(packed.x << 16u);
	f[1] = __uint_as_float(packed.x & 0xffff0000u);
	f[2] = __uint_as_float(packed.y << 16u);
	f[3] = __uint_as_float(packed.y & 0xffff0000u);
	f[4] = __uint_as_float(packed.z << 16u);
	f[5] = __uint_as_float(packed.z & 0xffff0000u);
	f[6] = __uint_as_float(packed.w << 16u);
	f[7] = __uint_as_float(packed.w & 0xffff0000u);
}

static __device__ __forceinline__ float LmSkinnyBf16ChunkUnpacked(uint4 weight, const float *f, float accumulator)
{
	float a0 = fmaf(__uint_as_float(weight.x << 16u),f[0],accumulator);
	float a1 = __uint_as_float(weight.x & 0xffff0000u) * f[1];
	a0 = fmaf(__uint_as_float(weight.y << 16u),f[2],a0);
	a1 = fmaf(__uint_as_float(weight.y & 0xffff0000u),f[3],a1);
	a0 = fmaf(__uint_as_float(weight.z << 16u),f[4],a0);
	a1 = fmaf(__uint_as_float(weight.z & 0xffff0000u),f[5],a1);
	a0 = fmaf(__uint_as_float(weight.w << 16u),f[6],a0);
	a1 = fmaf(__uint_as_float(weight.w & 0xffff0000u),f[7],a1);
	return(a0 + a1);
}

static __device__ __forceinline__ float LmSkinnyBf16DotUnpacked(const float *w, const float *f)
{
	float a0 = fmaf(w[0],f[0],0.0f);
	float a1 = w[1] * f[1];
	a0 = fmaf(w[2],f[2],a0);
	a1 = fmaf(w[3],f[3],a1);
	a0 = fmaf(w[4],f[4],a0);
	a1 = fmaf(w[5],f[5],a1);
	a0 = fmaf(w[6],f[6],a0);
	a1 = fmaf(w[7],f[7],a1);
	return(a0 + a1);
}

static __device__ __forceinline__ float LmSkinnyBf16Chunk(uint4 weight, uint4 activation, float accumulator)
{
	float a0 = fmaf(__uint_as_float(weight.x << 16u),__uint_as_float(activation.x << 16u),accumulator);
	float a1 = __uint_as_float(weight.x & 0xffff0000u) * __uint_as_float(activation.x & 0xffff0000u);
	a0 = fmaf(__uint_as_float(weight.y << 16u),__uint_as_float(activation.y << 16u),a0);
	a1 = fmaf(__uint_as_float(weight.y & 0xffff0000u),__uint_as_float(activation.y & 0xffff0000u),a1);
	a0 = fmaf(__uint_as_float(weight.z << 16u),__uint_as_float(activation.z << 16u),a0);
	a1 = fmaf(__uint_as_float(weight.z & 0xffff0000u),__uint_as_float(activation.z & 0xffff0000u),a1);
	a0 = fmaf(__uint_as_float(weight.w << 16u),__uint_as_float(activation.w << 16u),a0);
	a1 = fmaf(__uint_as_float(weight.w & 0xffff0000u),__uint_as_float(activation.w & 0xffff0000u),a1);
	return(a0 + a1);
}

static __device__ __forceinline__ float LmSkinnyDotPairs16(const float2 *pair, const float *f)
{
	float a0 = pair[0].x * f[0];
	float a1 = pair[0].y * f[1];
	float a2 = pair[1].x * f[2];
	float a3 = pair[1].y * f[3];
	a0 = fmaf(pair[2].x,f[4],a0);
	a1 = fmaf(pair[2].y,f[5],a1);
	a2 = fmaf(pair[3].x,f[6],a2);
	a3 = fmaf(pair[3].y,f[7],a3);
	a0 = fmaf(pair[4].x,f[8],a0);
	a1 = fmaf(pair[4].y,f[9],a1);
	a2 = fmaf(pair[5].x,f[10],a2);
	a3 = fmaf(pair[5].y,f[11],a3);
	a0 = fmaf(pair[6].x,f[12],a0);
	a1 = fmaf(pair[6].y,f[13],a1);
	a2 = fmaf(pair[7].x,f[14],a2);
	a3 = fmaf(pair[7].y,f[15],a3);
	return((a0 + a1) + (a2 + a3));
}

static __device__ __forceinline__ float2 LmSkinnyE2m1Pair(uint32_t word, uint32_t byte)
{
#if defined(__CUDA_ARCH__)
	uint32_t widened;
	asm("{\n\t.reg .b8 narrow;\n\tcvt.u8.u32 narrow, %1;\n\tcvt.rn.f16x2.e2m1x2 %0, narrow;\n\t}\n"
		: "=r"(widened) : "r"(word >> (8u * byte)));
	return(make_float2(__half2float(__ushort_as_half((uint16_t)(widened & 0xffffu))),
		__half2float(__ushort_as_half((uint16_t)(widened >> 16u)))));
#else
	const uint32_t packed = (word >> (8u * byte)) & 0xffu;
	return(make_float2(LmE2m1ToFloatPure(packed & 15u),LmE2m1ToFloatPure((packed >> 4u) & 15u)));
#endif
}

typedef struct LmSkinnyE2m1HalfDecoded
{
	float2 pair[8];
}
LmSkinnyE2m1HalfDecoded;

static __device__ __forceinline__ LmSkinnyE2m1HalfDecoded LmSkinnyDecodeE2m1Half(uint32_t w0, uint32_t w1)
{
	LmSkinnyE2m1HalfDecoded decoded;
	#pragma unroll
	for ( uint32_t j = 0u; j < 4u; j++ )
	{
		decoded.pair[j] = LmSkinnyE2m1Pair(w0,j);
		decoded.pair[4u + j] = LmSkinnyE2m1Pair(w1,j);
	}
	return(decoded);
}

static __device__ __forceinline__ float LmSkinnyAccumE2m1Half(const LmSkinnyE2m1HalfDecoded &d, uint4 a0_in, uint4 a1_in)
{
	float a0 = d.pair[0].x * __uint_as_float(a0_in.x << 16u);
	float a1 = d.pair[0].y * __uint_as_float(a0_in.x & 0xffff0000u);
	float a2 = d.pair[1].x * __uint_as_float(a0_in.y << 16u);
	float a3 = d.pair[1].y * __uint_as_float(a0_in.y & 0xffff0000u);
	a0 = fmaf(d.pair[2].x,__uint_as_float(a0_in.z << 16u),a0);
	a1 = fmaf(d.pair[2].y,__uint_as_float(a0_in.z & 0xffff0000u),a1);
	a2 = fmaf(d.pair[3].x,__uint_as_float(a0_in.w << 16u),a2);
	a3 = fmaf(d.pair[3].y,__uint_as_float(a0_in.w & 0xffff0000u),a3);
	a0 = fmaf(d.pair[4].x,__uint_as_float(a1_in.x << 16u),a0);
	a1 = fmaf(d.pair[4].y,__uint_as_float(a1_in.x & 0xffff0000u),a1);
	a2 = fmaf(d.pair[5].x,__uint_as_float(a1_in.y << 16u),a2);
	a3 = fmaf(d.pair[5].y,__uint_as_float(a1_in.y & 0xffff0000u),a3);
	a0 = fmaf(d.pair[6].x,__uint_as_float(a1_in.z << 16u),a0);
	a1 = fmaf(d.pair[6].y,__uint_as_float(a1_in.z & 0xffff0000u),a1);
	a2 = fmaf(d.pair[7].x,__uint_as_float(a1_in.w << 16u),a2);
	a3 = fmaf(d.pair[7].y,__uint_as_float(a1_in.w & 0xffff0000u),a3);
	return((a0 + a1) + (a2 + a3));
}

template<class Format>
struct LmSkinnyDecodeOnce
{
	static constexpr bool kValue = false;
};

template<class Format>
struct LmSkinnyDecodeHalves
{
	static constexpr bool kValue = false;
};

template<class Format>
struct LmSkinnyFormat
{
	static constexpr bool kSupported = false;
};

typedef struct LmSkinnyWide
{
	uint4 low,high;
}
LmSkinnyWide;

template<>
struct LmSkinnyFormat<LmBf16Format>
{
	using Activation = uint4;
	static constexpr bool kSupported = true;
	static constexpr uint32_t kScaleEncoding = LM_SCALE_ENCODING_NONE;
	static constexpr uint32_t kElements = 8u;
	static __device__ __forceinline__ Activation Load(const uint16_t *activation)
	{
		return(__ldg((const uint4 *)activation));
	}
	static __device__ __forceinline__ float Chunk(uint4 weight, Activation activation)
	{
		return(LmSkinnyBf16Chunk(weight,activation,0.0f));
	}
};

template<>
struct LmSkinnyFormat<LmFp8>
{
	using Activation = LmSkinnyWide;
	static constexpr bool kSupported = true;
	static constexpr uint32_t kScaleEncoding = LM_SCALE_ENCODING_F32;
	static constexpr uint32_t kElements = 16u;
	static __device__ __forceinline__ Activation Load(const uint16_t *activation)
	{
		Activation value;
		value.low = __ldg((const uint4 *)activation);
		value.high = __ldg((const uint4 *)activation + 1);
		return(value);
	}
	typedef struct Decoded
	{
		float2 pair[8];
	}
	Decoded;
	static __device__ __forceinline__ Decoded Decode(uint4 weight)
	{
		Decoded decoded;
		const uint32_t words[4] = {weight.x,weight.y,weight.z,weight.w};
		#pragma unroll
		for ( uint32_t word = 0u; word < 4u; word++ )
		{
			decoded.pair[2u * word] = LmE4m3PairToFloat2Pure((uint16_t)words[word]);
			decoded.pair[2u * word + 1u] = LmE4m3PairToFloat2Pure((uint16_t)(words[word] >> 16u));
		}
		return(decoded);
	}
	static __device__ __forceinline__ float ChunkDecodedUnpacked(const Decoded &decoded, const float *f)
	{
		return(LmSkinnyDotPairs16(decoded.pair,f));
	}
	static __device__ __forceinline__ float ChunkDecoded(const Decoded &decoded, Activation activation)
	{
		float a0 = decoded.pair[0].x * __uint_as_float(activation.low.x << 16u);
		float a1 = decoded.pair[0].y * __uint_as_float(activation.low.x & 0xffff0000u);
		float a2 = decoded.pair[1].x * __uint_as_float(activation.low.y << 16u);
		float a3 = decoded.pair[1].y * __uint_as_float(activation.low.y & 0xffff0000u);
		a0 = fmaf(decoded.pair[2].x,__uint_as_float(activation.low.z << 16u),a0);
		a1 = fmaf(decoded.pair[2].y,__uint_as_float(activation.low.z & 0xffff0000u),a1);
		a2 = fmaf(decoded.pair[3].x,__uint_as_float(activation.low.w << 16u),a2);
		a3 = fmaf(decoded.pair[3].y,__uint_as_float(activation.low.w & 0xffff0000u),a3);
		a0 = fmaf(decoded.pair[4].x,__uint_as_float(activation.high.x << 16u),a0);
		a1 = fmaf(decoded.pair[4].y,__uint_as_float(activation.high.x & 0xffff0000u),a1);
		a2 = fmaf(decoded.pair[5].x,__uint_as_float(activation.high.y << 16u),a2);
		a3 = fmaf(decoded.pair[5].y,__uint_as_float(activation.high.y & 0xffff0000u),a3);
		a0 = fmaf(decoded.pair[6].x,__uint_as_float(activation.high.z << 16u),a0);
		a1 = fmaf(decoded.pair[6].y,__uint_as_float(activation.high.z & 0xffff0000u),a1);
		a2 = fmaf(decoded.pair[7].x,__uint_as_float(activation.high.w << 16u),a2);
		a3 = fmaf(decoded.pair[7].y,__uint_as_float(activation.high.w & 0xffff0000u),a3);
		return((a0 + a1) + (a2 + a3));
	}
	static __device__ __forceinline__ float Chunk(uint4 weight, Activation activation)
	{
		return(ChunkDecoded(Decode(weight),activation));
	}
};

template<>
struct LmSkinnyDecodeOnce<LmFp8>
{
	static constexpr bool kValue = true;
};

typedef struct LmSkinnyQuad
{
	uint4 part[4];
}
LmSkinnyQuad;

template<>
struct LmSkinnyDecodeOnce<LmMxfp8>
{
	static constexpr bool kValue = true;
};

template<>
struct LmSkinnyFormat<LmMxfp8> : LmSkinnyFormat<LmFp8>
{
	static constexpr uint32_t kScaleEncoding = LM_SCALE_ENCODING_UE8M0;
};

template<>
struct LmSkinnyFormat<LmMxfp4>
{
	using Activation = LmSkinnyQuad;
	static constexpr bool kSupported = true;
	static constexpr uint32_t kScaleEncoding = LM_SCALE_ENCODING_UE8M0;
	static constexpr uint32_t kElements = 32u;
	static __device__ __forceinline__ Activation Load(const uint16_t *activation)
	{
		Activation value;
		value.part[0] = __ldg((const uint4 *)activation);
		value.part[1] = __ldg((const uint4 *)activation + 1);
		value.part[2] = __ldg((const uint4 *)activation + 2);
		value.part[3] = __ldg((const uint4 *)activation + 3);
		return(value);
	}
	static __device__ __forceinline__ LmSkinnyWide LoadHalf(const uint16_t *activation, uint32_t half)
	{
		LmSkinnyWide value;
		const uint4 *base = (const uint4 *)activation + half * 2u;
		value.low = __ldg(base);
		value.high = __ldg(base + 1);
		return(value);
	}
	static __device__ __forceinline__ float Chunk(uint4 weight, Activation activation)
	{
		const LmSkinnyE2m1HalfDecoded low = LmSkinnyDecodeE2m1Half(weight.x,weight.y);
		const LmSkinnyE2m1HalfDecoded high = LmSkinnyDecodeE2m1Half(weight.z,weight.w);
		return(LmSkinnyAccumE2m1Half(low,activation.part[0],activation.part[1]) +
			LmSkinnyAccumE2m1Half(high,activation.part[2],activation.part[3]));
	}
};

template<>
struct LmSkinnyDecodeHalves<LmMxfp4>
{
	static constexpr bool kValue = true;
};

template<class Format>
struct LmSkinnySplitScale
{
	static constexpr bool kSplit = false;
};

template<>
struct LmSkinnyFormat<LmNvfp4>
{
	using Activation = LmSkinnyQuad;
	static constexpr bool kSupported = true;
	static constexpr uint32_t kScaleEncoding = LM_SCALE_ENCODING_UE4M3_F32_GLOBAL;
	static constexpr uint32_t kElements = 32u;
	static __device__ __forceinline__ Activation Load(const uint16_t *activation)
	{
		Activation value;
		value.part[0] = __ldg((const uint4 *)activation);
		value.part[1] = __ldg((const uint4 *)activation + 1);
		value.part[2] = __ldg((const uint4 *)activation + 2);
		value.part[3] = __ldg((const uint4 *)activation + 3);
		return(value);
	}
	static __device__ __forceinline__ LmSkinnyWide LoadHalf(const uint16_t *activation, uint32_t half)
	{
		LmSkinnyWide value;
		const uint4 *base = (const uint4 *)activation + half * 2u;
		value.low = __ldg(base);
		value.high = __ldg(base + 1);
		return(value);
	}
	static __device__ __forceinline__ float2 Halves(uint4 weight, Activation activation)
	{
		float2 value;
		const LmSkinnyE2m1HalfDecoded low = LmSkinnyDecodeE2m1Half(weight.x,weight.y);
		const LmSkinnyE2m1HalfDecoded high = LmSkinnyDecodeE2m1Half(weight.z,weight.w);
		value.x = LmSkinnyAccumE2m1Half(low,activation.part[0],activation.part[1]);
		value.y = LmSkinnyAccumE2m1Half(high,activation.part[2],activation.part[3]);
		return(value);
	}
};

template<>
struct LmSkinnyDecodeHalves<LmNvfp4>
{
	static constexpr bool kValue = true;
};

template<>
struct LmSkinnySplitScale<LmNvfp4>
{
	static constexpr bool kSplit = true;
};

template<class Format>
static __host__ __device__ __forceinline__ constexpr uint32_t LmSkinnyScaleBytes(void)
{
	return(LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32 ? (uint32_t)sizeof(float) : 1u);
}

template<class Format>
static __device__ __forceinline__ float LmSkinnyScaleGlobal(const LmScaleTensor *scale, uint32_t group)
{
	if constexpr ( LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_UE4M3_F32_GLOBAL )
		return(__ldg((const float *)scale->global_data + group));
	else
		return(1.0f);
}

template<class Format>
static __device__ __forceinline__ float LmSkinnyScaleEntry(const uint8_t *entry, float global)
{
	constexpr uint32_t encoding = LmSkinnyFormat<Format>::kScaleEncoding;
	if constexpr ( encoding == LM_SCALE_ENCODING_F32 )
		return(__ldg((const float *)entry));
	else if constexpr ( encoding == LM_SCALE_ENCODING_UE8M0 )
		return(LmUe8m0ToFloat(__ldg(entry)));
	else
	{
		static_assert(encoding == LM_SCALE_ENCODING_UE4M3_F32_GLOBAL, "a scaled skinny format reads f32, ue8m0 or ue4m3 scales");
		return(LmUe4m3ToFloat(__ldg(entry)) * global);
	}
}

typedef struct LmSkinnyUe4m3Raw
{
	uint8_t low,high;
}
LmSkinnyUe4m3Raw;

static __device__ __forceinline__ float2 LmSkinnyDecodeUe4m3Raw(LmSkinnyUe4m3Raw raw_bytes, float global)
{
#if defined(__CUDA_ARCH__)
	const uint32_t raw = (uint32_t)raw_bytes.low | ((uint32_t)raw_bytes.high << 8u);
	const uint32_t masked = raw & 0x7f7fu;
	const uint32_t clamp = ((masked + 0x0101u) >> 7u) & 0x0101u;
	const float2 pair = LmE4m3PairToFloat2Pure((uint16_t)(masked - clamp));
	return(make_float2(pair.x * global,pair.y * global));
#else
	return(make_float2(LmUe4m3ToFloat(raw_bytes.low) * global,LmUe4m3ToFloat(raw_bytes.high) * global));
#endif
}

template<uint32_t ENCODING>
struct LmSkinnyRawScaleType
{
	using Type = uint8_t;
};

template<>
struct LmSkinnyRawScaleType<LM_SCALE_ENCODING_F32>
{
	using Type = float;
};

template<>
struct LmSkinnyRawScaleType<LM_SCALE_ENCODING_UE4M3_F32_GLOBAL>
{
	using Type = LmSkinnyUe4m3Raw;
};

template<class Format>
struct LmSkinnyRawScale
{
	static constexpr uint32_t kEncoding = LmSkinnyFormat<Format>::kScaleEncoding;
	using Raw = typename LmSkinnyRawScaleType<kEncoding>::Type;
	static __device__ __forceinline__ Raw Identity(void)
	{
		if constexpr ( kEncoding == LM_SCALE_ENCODING_F32 )
			return(1.0f);
		else if constexpr ( kEncoding == LM_SCALE_ENCODING_UE4M3_F32_GLOBAL )
			return(LmSkinnyUe4m3Raw{0x38u,0x38u});
		else
			return(127u);
	}
	static __device__ __forceinline__ const uint8_t *RowBase(const LmScaleTensor *scale, uint32_t group, uint32_t neuron)
	{
		if constexpr ( Format::kScaleGroup == 0u )
			return(0);
		else
			return((const uint8_t *)scale->data + (((uint64_t)group * scale->group_stride_entries) + ((uint64_t)neuron * scale->row_group_stride_entries)) * LmSkinnyScaleBytes<Format>());
	}
	static __device__ __forceinline__ Raw LoadFromRow(const uint8_t *row_base, uint32_t chunk)
	{
		if constexpr ( Format::kScaleGroup == 0u )
			return(0u);
		else if constexpr ( kEncoding == LM_SCALE_ENCODING_UE4M3_F32_GLOBAL )
		{
			const uint8_t *entry = row_base + 2u * chunk;
			return(LmSkinnyUe4m3Raw{__ldg(entry),__ldg(entry + 1)});
		}
		else
		{
			constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements;
			const uint32_t offset = (chunk * elements) / Format::kScaleGroup;
			if constexpr ( kEncoding == LM_SCALE_ENCODING_F32 )
				return(__ldg((const float *)row_base + offset));
			else
				return(__ldg(row_base + offset));
		}
	}
	static __device__ __forceinline__ Raw Load(const LmScaleTensor *scale, uint32_t group, uint32_t neuron, uint32_t chunk)
	{
		if constexpr ( Format::kScaleGroup == 0u )
			return(0u);
		else
			return(LoadFromRow(RowBase(scale,group,neuron),chunk));
	}
	static __device__ __forceinline__ float Decode(Raw raw)
	{
		if constexpr ( Format::kScaleGroup == 0u )
			return(1.0f);
		else if constexpr ( kEncoding == LM_SCALE_ENCODING_F32 )
			return(raw);
		else
			return(LmUe8m0ToFloat(raw));
	}
	static __device__ __forceinline__ float2 DecodePair(Raw raw, float global)
	{
		return(LmSkinnyDecodeUe4m3Raw(raw,global));
	}
};

static __device__ __forceinline__ float2 LmSkinnyUe4m3Pair(const LmScaleTensor *scale, uint32_t group, uint32_t neuron, uint32_t chunk, float global)
{
	return(LmSkinnyDecodeUe4m3Raw(LmSkinnyRawScale<LmNvfp4>::Load(scale,group,neuron,chunk),global));
}

template<class Format>
static __device__ __forceinline__ float LmSkinnyScaleWithGlobal(const LmScaleTensor *scale, uint32_t group, uint32_t neuron, uint32_t k, float global)
{
	if constexpr ( Format::kScaleGroup == 0u )
		return(1.0f);
	else
		return(LmSkinnyScaleEntry<Format>((const uint8_t *)scale->data + (((uint64_t)group * scale->group_stride_entries) + ((uint64_t)neuron * scale->row_group_stride_entries) + (k / Format::kScaleGroup)) * LmSkinnyScaleBytes<Format>(),
			global));
}

template<class Format>
static __device__ __forceinline__ float LmSkinnyScale(const LmScaleTensor *scale, uint32_t group, uint32_t neuron, uint32_t k)
{
	return(LmSkinnyScaleWithGlobal<Format>(scale,group,neuron,k,LmSkinnyScaleGlobal<Format>(scale,group)));
}

template<class Format>
static __device__ __forceinline__ float LmSkinnyScaleByEncoding(const LmScaleTensor *scale, uint32_t group, uint32_t neuron, uint32_t k)
{
	return(LmSkinnyScale<Format>(scale,group,neuron,k));
}

template<class Format, uint32_t ROWS>
static __device__ __forceinline__ float LmSkinnyRowsScale(const LmScaleTensor *scale, uint32_t group, uint32_t neuron, uint32_t k)
{
	return(LmSkinnyScale<Format>(scale,group,neuron,k));
}

template<class Format>
static uint32_t LmSkinnyScaleFits(const LmSkinnyArguments *args)
{
	const LmScaleTensor *scale = &args->scale;
	if ( scale->encoding != LmSkinnyFormat<Format>::kScaleEncoding || LmScaleTensorIsValid(scale) == 0u )
		return(0u);
	if constexpr ( Format::kScaleGroup == 0u )
		return(1u);
	else
		return(scale->row_group_size == 1u && scale->k_group_size == Format::kScaleGroup &&
			scale->group_count >= (args->groups != 0u ? args->groups : 1u) &&
			scale->row_count >= args->output_dimension && scale->input_dimension >= args->input_dimension ? 1u : 0u);
}

static __device__ __forceinline__ void LmSkinnyResolve(const LmSkinnyArguments &args, uint32_t pair, uint32_t *group, uint32_t *source, uint32_t *target)
{
	*group = 0u;
	*source = 0u;
	*target = 0u;
	if ( args.route_expert == 0 )
		return;
	*group = args.route_expert[pair];
	*target = args.route_packed_row != 0 ? args.route_packed_row[pair] : pair;
	*source = args.activation_packed != 0u ? *target : pair / args.top_k;
}

template<class Format, uint32_t ROWS, uint32_t NPG>
static __device__ __forceinline__ void LmSkinnyAccumulateStep(const LmSkinnyArguments &args, const uint4 *weight, const uint16_t *const *activation, uint32_t group, uint32_t neuron, uint32_t c, float global, float (*accumulator)[ROWS], uint32_t chunk_offset = 0u)
{
	constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements;
	const uint32_t sc = chunk_offset + c;
	uint32_t n, r;
	if constexpr ( LmSkinnyDecodeHalves<Format>::kValue && ROWS > 1u )
	{
		float part[NPG][ROWS];
		#pragma unroll
		for ( uint32_t half = 0u; half < 2u; half++ )
		{
			LmSkinnyE2m1HalfDecoded dec[NPG];
			float scale[NPG];
			#pragma unroll
			for ( n = 0u; n < NPG; n++ )
			{
				const uint32_t row_neuron = neuron + n < args.output_dimension ? neuron + n : neuron;
				dec[n] = half == 0u ? LmSkinnyDecodeE2m1Half(weight[n].x,weight[n].y) : LmSkinnyDecodeE2m1Half(weight[n].z,weight[n].w);
				if constexpr ( LmSkinnySplitScale<Format>::kSplit )
					scale[n] = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,row_neuron,sc * elements + half * Format::kScaleGroup,global);
				else if ( half != 0u )
					scale[n] = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,row_neuron,sc * elements,global);
			}
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
			{
				const LmSkinnyWide a = LmSkinnyFormat<Format>::LoadHalf(activation[r] + c * elements,half);
				float af[16];
				LmSkinnyUnpackBf16x8(a.low,af);
				LmSkinnyUnpackBf16x8(a.high,af + 8);
				#pragma unroll
				for ( n = 0u; n < NPG; n++ )
				{
					const float p = LmSkinnyDotPairs16(dec[n].pair,af);
					if constexpr ( LmSkinnySplitScale<Format>::kSplit )
						accumulator[n][r] = fmaf(scale[n],p,accumulator[n][r]);
					else if ( half == 0u )
						part[n][r] = p;
					else
						accumulator[n][r] = fmaf(scale[n],part[n][r] + p,accumulator[n][r]);
				}
			}
		}
	}
	else if constexpr ( ROWS > 1u && LmSkinnyDecodeOnce<Format>::kValue )
	{
		if constexpr ( NPG == 1u )
		{
			const float scale = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,neuron,sc * elements,global);
			const typename LmSkinnyFormat<Format>::Decoded decoded = LmSkinnyFormat<Format>::Decode(weight[0]);
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
				accumulator[0][r] = fmaf(scale,LmSkinnyFormat<Format>::ChunkDecoded(decoded,LmSkinnyFormat<Format>::Load(activation[r] + c * elements)),accumulator[0][r]);
		}
		else
		{
			typename LmSkinnyFormat<Format>::Activation staged[ROWS];
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
				staged[r] = LmSkinnyFormat<Format>::Load(activation[r] + c * elements);
			#pragma unroll
			for ( n = 0u; n < NPG; n++ )
			{
				const uint32_t row_neuron = neuron + n < args.output_dimension ? neuron + n : neuron;
				const float scale = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,row_neuron,sc * elements,global);
				const typename LmSkinnyFormat<Format>::Decoded decoded = LmSkinnyFormat<Format>::Decode(weight[n]);
				#pragma unroll
				for ( r = 0u; r < ROWS; r++ )
					accumulator[n][r] = fmaf(scale,LmSkinnyFormat<Format>::ChunkDecoded(decoded,staged[r]),accumulator[n][r]);
			}
		}
	}
	else if constexpr ( NPG == 1u )
	{
		const float scale = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,neuron,sc * elements,global);
		if constexpr ( LmSkinnySplitScale<Format>::kSplit )
		{
			const float high = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,neuron,sc * elements + Format::kScaleGroup,global);
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
			{
				const float2 part = LmSkinnyFormat<Format>::Halves(weight[0],LmSkinnyFormat<Format>::Load(activation[r] + c * elements));
				accumulator[0][r] = fmaf(high,part.y,fmaf(scale,part.x,accumulator[0][r]));
			}
		}
		else
		{
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
				accumulator[0][r] = fmaf(scale,LmSkinnyFormat<Format>::Chunk(weight[0],LmSkinnyFormat<Format>::Load(activation[r] + c * elements)),accumulator[0][r]);
		}
	}
	else
	{
		typename LmSkinnyFormat<Format>::Activation staged[ROWS];
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
			staged[r] = LmSkinnyFormat<Format>::Load(activation[r] + c * elements);
		#pragma unroll
		for ( n = 0u; n < NPG; n++ )
		{
			const uint32_t row_neuron = neuron + n < args.output_dimension ? neuron + n : neuron;
			const float scale = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,row_neuron,sc * elements,global);
			if constexpr ( LmSkinnySplitScale<Format>::kSplit )
			{
				const float high = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,row_neuron,sc * elements + Format::kScaleGroup,global);
				#pragma unroll
				for ( r = 0u; r < ROWS; r++ )
				{
					const float2 part = LmSkinnyFormat<Format>::Halves(weight[n],staged[r]);
					accumulator[n][r] = fmaf(high,part.y,fmaf(scale,part.x,accumulator[n][r]));
				}
			}
			else
			{
				#pragma unroll
				for ( r = 0u; r < ROWS; r++ )
					accumulator[n][r] = fmaf(scale,LmSkinnyFormat<Format>::Chunk(weight[n],staged[r]),accumulator[n][r]);
			}
		}
	}
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
static __device__ __forceinline__ void LmSkinnyAccumulate(const LmSkinnyArguments &args, const uint4 *const *rows, const uint16_t *const *activation, uint32_t group, uint32_t neuron, uint32_t sub, uint32_t chunks, float (*accumulator)[ROWS], uint32_t chunk_offset = 0u)
{
	constexpr uint32_t depth = ROWS == 1u ? (sizeof(typename LmSkinnyFormat<Format>::Activation) > 32u ? (NPG >= 2u ? 2u : 4u) : (((NPG >= 4u || (NPG == 1u && LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32)) && sizeof(typename LmSkinnyFormat<Format>::Activation) > 16u) ? 4u : 8u)) : ROWS <= LM_SKINNY_ROWS_MID ? (sizeof(typename LmSkinnyFormat<Format>::Activation) > 16u ? 1u : 4u) : 1u, unroll = depth / NPG != 0u ? depth / NPG : 1u, elements = LmSkinnyFormat<Format>::kElements;
	const float global = LmSkinnyScaleGlobal<Format>(&args.scale,group);
	uint32_t base, u, n;
	uint4 weight[unroll][NPG];
	if constexpr ( ROWS == 1u )
	{
		using RawScale = typename LmSkinnyRawScale<Format>::Raw;
		typename LmSkinnyFormat<Format>::Activation staged[unroll];
		float2 split_scale[unroll][NPG];
		RawScale raw_scale[unroll][NPG];
		const uint8_t *scale_rows[NPG];
		#pragma unroll
		for ( n = 0u; n < NPG; n++ )
		{
			const uint32_t row_neuron = neuron + n < args.output_dimension ? neuron + n : neuron;
			scale_rows[n] = LmSkinnyRawScale<Format>::RowBase(&args.scale,group,row_neuron);
		}
		for ( base = sub; base + (unroll - 1u) * LANES < chunks; base += LANES * unroll )
		{
			#pragma unroll
			for ( u = 0u; u < unroll; u++ )
			{
				const uint32_t c = base + u * LANES;
				#pragma unroll
				for ( n = 0u; n < NPG; n++ )
				{
					weight[u][n] = __ldcs(rows[n] + c);
					if constexpr ( LmSkinnySplitScale<Format>::kSplit )
						split_scale[u][n] = LmSkinnyDecodeUe4m3Raw(LmSkinnyRawScale<LmNvfp4>::LoadFromRow(scale_rows[n],chunk_offset + c),global);
					else if constexpr ( Format::kScaleGroup != 0u )
						raw_scale[u][n] = LmSkinnyRawScale<Format>::LoadFromRow(scale_rows[n],chunk_offset + c);
				}
				staged[u] = LmSkinnyFormat<Format>::Load(activation[0] + c * elements);
			}
			#pragma unroll
			for ( u = 0u; u < unroll; u++ )
			{
				#pragma unroll
				for ( n = 0u; n < NPG; n++ )
				{
					if constexpr ( LmSkinnySplitScale<Format>::kSplit )
					{
						const float2 part = LmSkinnyFormat<Format>::Halves(weight[u][n],staged[u]);
						accumulator[n][0] = fmaf(split_scale[u][n].y,part.y,fmaf(split_scale[u][n].x,part.x,accumulator[n][0]));
					}
					else
					{
						const float dot = LmSkinnyFormat<Format>::Chunk(weight[u][n],staged[u]);
						const float scale = LmSkinnyRawScale<Format>::Decode(raw_scale[u][n]);
						accumulator[n][0] = fmaf(scale,dot,accumulator[n][0]);
					}
				}
			}
		}
		if constexpr ( unroll > 2u )
		{
			for ( ; base + LANES < chunks; base += 2u * LANES )
			{
				#pragma unroll
				for ( u = 0u; u < 2u; u++ )
				{
					const uint32_t c = base + u * LANES;
					#pragma unroll
					for ( n = 0u; n < NPG; n++ )
					{
						weight[u][n] = __ldcs(rows[n] + c);
						if constexpr ( LmSkinnySplitScale<Format>::kSplit )
							split_scale[u][n] = LmSkinnyDecodeUe4m3Raw(LmSkinnyRawScale<LmNvfp4>::LoadFromRow(scale_rows[n],chunk_offset + c),global);
						else if constexpr ( Format::kScaleGroup != 0u )
							raw_scale[u][n] = LmSkinnyRawScale<Format>::LoadFromRow(scale_rows[n],chunk_offset + c);
					}
					staged[u] = LmSkinnyFormat<Format>::Load(activation[0] + c * elements);
				}
				#pragma unroll
				for ( u = 0u; u < 2u; u++ )
				{
					#pragma unroll
					for ( n = 0u; n < NPG; n++ )
					{
						if constexpr ( LmSkinnySplitScale<Format>::kSplit )
						{
							const float2 part = LmSkinnyFormat<Format>::Halves(weight[u][n],staged[u]);
							accumulator[n][0] = fmaf(split_scale[u][n].y,part.y,fmaf(split_scale[u][n].x,part.x,accumulator[n][0]));
						}
						else
						{
							const float dot = LmSkinnyFormat<Format>::Chunk(weight[u][n],staged[u]);
							const float scale = LmSkinnyRawScale<Format>::Decode(raw_scale[u][n]);
							accumulator[n][0] = fmaf(scale,dot,accumulator[n][0]);
						}
					}
				}
			}
		}
		for ( ; base < chunks; base += LANES )
		{
			#pragma unroll
			for ( n = 0u; n < NPG; n++ )
			{
				weight[0][n] = __ldcs(rows[n] + base);
				if constexpr ( LmSkinnySplitScale<Format>::kSplit )
					split_scale[0][n] = LmSkinnyDecodeUe4m3Raw(LmSkinnyRawScale<LmNvfp4>::LoadFromRow(scale_rows[n],chunk_offset + base),global);
				else if constexpr ( Format::kScaleGroup != 0u )
					raw_scale[0][n] = LmSkinnyRawScale<Format>::LoadFromRow(scale_rows[n],chunk_offset + base);
			}
			const typename LmSkinnyFormat<Format>::Activation act = LmSkinnyFormat<Format>::Load(activation[0] + base * elements);
			#pragma unroll
			for ( n = 0u; n < NPG; n++ )
			{
				if constexpr ( LmSkinnySplitScale<Format>::kSplit )
				{
					const float2 part = LmSkinnyFormat<Format>::Halves(weight[0][n],act);
					accumulator[n][0] = fmaf(split_scale[0][n].y,part.y,fmaf(split_scale[0][n].x,part.x,accumulator[n][0]));
				}
				else
				{
					const float dot = LmSkinnyFormat<Format>::Chunk(weight[0][n],act);
					const float scale = LmSkinnyRawScale<Format>::Decode(raw_scale[0][n]);
					accumulator[n][0] = fmaf(scale,dot,accumulator[n][0]);
				}
			}
		}
	}
	else
	{
		for ( base = sub; base + (unroll - 1u) * LANES < chunks; base += LANES * unroll )
		{
			#pragma unroll
			for ( u = 0u; u < unroll; u++ )
			{
				const uint32_t c = base + u * LANES;
				#pragma unroll
				for ( n = 0u; n < NPG; n++ )
					weight[u][n] = __ldcs(rows[n] + c);
			}
			#pragma unroll
			for ( u = 0u; u < unroll; u++ )
				LmSkinnyAccumulateStep<Format,ROWS,NPG>(args,weight[u],activation,group,neuron,base + u * LANES,global,accumulator,chunk_offset);
		}
		for ( ; base < chunks; base += LANES )
		{
			uint4 tail[NPG];
			#pragma unroll
			for ( n = 0u; n < NPG; n++ )
				tail[n] = __ldcs(rows[n] + base);
			LmSkinnyAccumulateStep<Format,ROWS,NPG>(args,tail,activation,group,neuron,base,global,accumulator,chunk_offset);
		}
	}
}

template<uint32_t LANES, uint32_t ROWS, uint32_t NPG>
static __device__ __forceinline__ void LmSkinnyStore(const LmSkinnyArguments &args, float (*accumulator)[ROWS], uint32_t sub, uint32_t live, uint32_t target, uint32_t neuron, uint32_t rows)
{
	uint32_t step, n, r;
	uint64_t index;
	#pragma unroll
	for ( step = LANES / 2u; step > 0u; step >>= 1u )
		#pragma unroll
		for ( n = 0u; n < NPG; n++ )
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
				accumulator[n][r] += __shfl_xor_sync(0xffffffffu,accumulator[n][r],step,LANES);
	if ( live == 0u || sub != 0u )
		return;
	#pragma unroll
	for ( n = 0u; n < NPG; n++ )
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
		{
			if ( r >= rows || neuron + n >= args.output_dimension )
				continue;
			index = (uint64_t)(target + r) * args.output_row_stride + args.output_column_offset + neuron + n;
			if ( args.output_f32 != 0 )
				args.output_f32[index] = accumulator[n][r];
			else
				args.output_bf16[index] = LmFloatToBf16(accumulator[n][r]);
		}
}

template<class Format, uint32_t NPG>
static __device__ __forceinline__ void LmSkinnyWeightRows(const LmSkinnyArguments &args, uint32_t group, uint32_t neuron, const uint4 **rows)
{
	constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements;
	uint32_t n;
	#pragma unroll
	for ( n = 0u; n < NPG; n++ )
		rows[n] = (const uint4 *)(args.weight + (uint64_t)group * args.weight_group_bytes + (uint64_t)(neuron + n < args.output_dimension ? neuron + n : neuron) * (args.input_dimension / elements) * LM_SKINNY_CHUNK_BYTES);
}

template<uint32_t NPG, uint32_t ROWS>
static __device__ __forceinline__ void LmSkinnyClear(float (*accumulator)[ROWS])
{
	uint32_t n, r;
	#pragma unroll
	for ( n = 0u; n < NPG; n++ )
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
			accumulator[n][r] = 0.0f;
}

template<class Format, uint32_t NPG>
static __device__ __forceinline__ void LmSkinnyPrefetchDense(const LmSkinnyArguments &args, uint32_t live, uint32_t sub, uint32_t neuron, uint32_t chunks)
{
	const uint4 *rows[NPG];
	uint32_t n;
	if ( args.route_expert != 0 || live == 0u || sub != 0u )
		return;
	LmSkinnyWeightRows<Format,NPG>(args,0u,neuron,rows);
	#pragma unroll
	for ( n = 0u; n < NPG; n++ )
		if ( neuron + n < args.output_dimension )
			LmPrefetchL2(rows[n],chunks * LM_SKINNY_CHUNK_BYTES);
}

template<uint32_t ROWS>
static __host__ __device__ __forceinline__ uint32_t LmSkinnyRowBlocks(uint32_t rows)
{
	return(rows > ROWS ? (rows + ROWS - 1u) / ROWS : 1u);
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
static __device__ __forceinline__ void LmSkinnyTask(const LmSkinnyArguments &args, uint32_t task, uint32_t first)
{
	constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements;
	const uint32_t sub = threadIdx.x % LANES, per_pair = (args.output_dimension + NPG - 1u) / NPG;
	const uint32_t live = task < (args.route_expert != 0 ? args.pairs : 1u) * per_pair ? 1u : 0u;
	const uint32_t chunks = live != 0u ? args.input_dimension / elements : 0u, pair = live != 0u ? task / per_pair : 0u, neuron = live != 0u ? (task % per_pair) * NPG : 0u;
	const uint32_t count = args.rows - first < ROWS ? args.rows - first : ROWS;
	uint32_t group, source, target, r;
	float accumulator[NPG][ROWS];
	const uint4 *rows[NPG];
	const uint16_t *activation[ROWS];
	if ( LmSkinnyShouldSkip(args) != 0u )
	{
		LmDependentRelease();
		return;
	}
	LmDependentWait();
	LmDependentRelease();
	LmSkinnyResolve(args,pair,&group,&source,&target);
	LmSkinnyWeightRows<Format,NPG>(args,group,neuron,rows);
	LmSkinnyClear<NPG,ROWS>(accumulator);
	#pragma unroll
	for ( r = 0u; r < ROWS; r++ )
		activation[r] = args.activation + (uint64_t)(source + first + (r < count ? r : 0u)) * args.input_dimension;
	LmSkinnyAccumulate<Format,LANES,ROWS,NPG>(args,rows,activation,group,neuron,sub,chunks,accumulator);
	LmSkinnyStore<LANES,ROWS,NPG>(args,accumulator,sub,live,target + first,neuron,count);
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
__global__ __launch_bounds__(LM_SKINNY_THREADS, 2) void LmSkinnyKernel(const __grid_constant__ LmSkinnyArguments args)
{
	const uint32_t blocks = LmSkinnyRowBlocks<ROWS>(args.rows);
	LmSkinnyTask<Format,LANES,ROWS,NPG>(args,((blockIdx.x / blocks) * LM_SKINNY_THREADS + threadIdx.x) / LANES,(blockIdx.x % blocks) * ROWS);
}

#define LM_SKINNY_SPLIT_WARPS (LM_SKINNY_THREADS / LM_WARP_LANES)
#define LM_SKINNY_SPLIT_MAX_OUTPUTS 384u
#define LM_SKINNY_SPLIT_BLOCKS_PER_SM 3u

template<class Format, uint32_t ROWS>
static __device__ __forceinline__ void LmSkinnySplitNeuron(const LmSkinnyArguments &args, uint32_t neuron, uint32_t row)
{
	constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements;
	__shared__ float partial[LM_SKINNY_SPLIT_WARPS][ROWS];
	const uint32_t count = args.rows - row < ROWS ? args.rows - row : ROWS;
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, sub = threadIdx.x % LM_WARP_LANES;
	const uint32_t chunks = args.input_dimension / elements;
	const uint32_t first = chunks * warp / LM_SKINNY_SPLIT_WARPS, last = chunks * (warp + 1u) / LM_SKINNY_SPLIT_WARPS;
	float accumulator[1][ROWS];
	const uint4 *rows[1];
	const uint16_t *activation[ROWS];
	uint32_t r, w;
	if ( LmSkinnyShouldSkip(args) != 0u )
	{
		LmDependentRelease();
		return;
	}
	LmSkinnyWeightRows<Format,1u>(args,0u,neuron,rows);
	rows[0] += first;
	if ( sub == 0u && row == 0u )
		LmPrefetchL2(rows[0],(last - first) * LM_SKINNY_CHUNK_BYTES);
	LmDependentWait();
	LmDependentRelease();
	#pragma unroll
	for ( r = 0u; r < ROWS; r++ )
		activation[r] = args.activation + (uint64_t)(row + (r < count ? r : 0u)) * args.input_dimension + (uint64_t)first * elements;
	LmSkinnyClear<1u,ROWS>(accumulator);
	LmSkinnyAccumulate<Format,LM_WARP_LANES,ROWS,1u>(args,rows,activation,0u,neuron,sub,last - first,accumulator,first);
	#pragma unroll
	for ( w = LM_WARP_LANES / 2u; w > 0u; w >>= 1u )
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
			accumulator[0][r] += __shfl_xor_sync(0xffffffffu,accumulator[0][r],w);
	if ( sub == 0u )
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
			partial[warp][r] = accumulator[0][r];
	__syncthreads();
	if ( warp != 0u || sub >= ROWS || sub >= count )
		return;
	float total = 0.0f;
	#pragma unroll
	for ( w = 0u; w < LM_SKINNY_SPLIT_WARPS; w++ )
		total += partial[w][sub];
	const uint64_t index = (uint64_t)(row + sub) * args.output_row_stride + args.output_column_offset + neuron;
	if ( args.output_f32 != 0 )
		args.output_f32[index] = total;
	else
		args.output_bf16[index] = LmFloatToBf16(total);
}

template<class Format, uint32_t ROWS>
__global__ __launch_bounds__(LM_SKINNY_THREADS) void LmSkinnySplitKernel(const __grid_constant__ LmSkinnyArguments args)
{
	const uint32_t blocks = LmSkinnyRowBlocks<ROWS>(args.rows);
	LmSkinnySplitNeuron<Format,ROWS>(args,blockIdx.x / blocks,(blockIdx.x % blocks) * ROWS);
}

static int32_t LmSkinnyMultiprocessors(uint32_t *count)
{
	int device = 0, value = 0;
	if ( SparkGpuGetDevice(&device) != SPARK_GPU_SUCCESS || SparkGpuDeviceGetAttribute(&value,SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT,device) != SPARK_GPU_SUCCESS || value <= 0 )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	*count = (uint32_t)value;
	return(LM_LAUNCH_OK);
}

template<class Format, uint32_t ROWS>
static int32_t LmSkinnySplitShape(const LmSkinnyArguments *args, SparkGpuStream stream)
{
	LM_LAUNCH_DEPENDENT((LmSkinnySplitKernel<Format,ROWS>),args->output_dimension * LmSkinnyRowBlocks<ROWS>(args->rows),LM_SKINNY_THREADS,0u,stream,*args);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t LmSkinnySplitRows(const LmSkinnyArguments *args, SparkGpuStream stream)
{
	LmWeightPrefetchNote(stream,args->weight,(uint64_t)args->output_dimension * (args->input_dimension / LmSkinnyFormat<Format>::kElements) * LM_SKINNY_CHUNK_BYTES);
	uint32_t multiprocessors = 0u;
	uint64_t enough;
	if ( args->rows == 1u )
		return(LmSkinnySplitShape<Format,1u>(args,stream));
	if ( args->rows > LM_SKINNY_ROWS_MID )
		return(LmSkinnySplitShape<Format,LM_SKINNY_ROWS>(args,stream));
	if ( LmSkinnyMultiprocessors(&multiprocessors) != LM_LAUNCH_OK )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	enough = (uint64_t)multiprocessors * LM_SKINNY_SPLIT_BLOCKS_PER_SM > args->output_dimension ? (uint64_t)multiprocessors * LM_SKINNY_SPLIT_BLOCKS_PER_SM : args->output_dimension;
	if ( (uint64_t)args->output_dimension * args->rows <= enough )
		return(LmSkinnySplitShape<Format,1u>(args,stream));
	if ( (uint64_t)args->output_dimension * LmSkinnyRowBlocks<2u>(args->rows) <= enough )
		return(LmSkinnySplitShape<Format,2u>(args,stream));
	return(LmSkinnySplitShape<Format,LM_SKINNY_ROWS_MID>(args,stream));
}

#define LM_SKINNY_MULTI_MAX 4u

typedef struct LmSkinnyMultiArguments
{
	LmSkinnyArguments part[LM_SKINNY_MULTI_MAX];
	uint32_t first_task[LM_SKINNY_MULTI_MAX + 1u];
	uint32_t count;
}
LmSkinnyMultiArguments;

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
__global__ __launch_bounds__(LM_SKINNY_THREADS) void LmSkinnyMultiKernel(const __grid_constant__ LmSkinnyMultiArguments multi)
{
	const uint32_t blocks = LmSkinnyRowBlocks<ROWS>(multi.part[0].rows), task = ((blockIdx.x / blocks) * LM_SKINNY_THREADS + threadIdx.x) / LANES;
	uint32_t part = 0u;
	while ( part + 1u < multi.count && task >= multi.first_task[part + 1u] )
		part++;
	LmSkinnyTask<Format,LANES,ROWS,NPG>(multi.part[part],task - multi.first_task[part],(blockIdx.x % blocks) * ROWS);
}

template<class Format, uint32_t ROWS>
__global__ __launch_bounds__(LM_SKINNY_THREADS) void LmSkinnySplitMultiKernel(const __grid_constant__ LmSkinnyMultiArguments multi)
{
	const uint32_t blocks = LmSkinnyRowBlocks<ROWS>(multi.part[0].rows), global = blockIdx.x / blocks;
	uint32_t part = 0u;
	while ( part + 1u < multi.count && global >= multi.first_task[part + 1u] )
		part++;
	LmSkinnySplitNeuron<Format,ROWS>(multi.part[part],global - multi.first_task[part],(blockIdx.x % blocks) * ROWS);
}

template<class Format, uint32_t ROWS>
static int32_t LmSkinnySplitMultiShape(const LmSkinnyMultiArguments *multi, SparkGpuStream stream)
{
	LM_LAUNCH_DEPENDENT((LmSkinnySplitMultiKernel<Format,ROWS>),multi->first_task[multi->count] * LmSkinnyRowBlocks<ROWS>(multi->part[0].rows),LM_SKINNY_THREADS,0u,stream,*multi);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t LmSkinnySplitMultiRows(LmSkinnyMultiArguments *multi, SparkGpuStream stream)
{
	uint32_t multiprocessors = 0u, part;
	uint64_t enough, outputs = 0u;
	multi->first_task[0] = 0u;
	for ( part = 0u; part < multi->count; part++ )
	{
		outputs += multi->part[part].output_dimension;
		multi->first_task[part + 1u] = (uint32_t)outputs;
	}
	const uint32_t rows = multi->part[0].rows;
	if ( rows == 1u )
		return(LmSkinnySplitMultiShape<Format,1u>(multi,stream));
	if ( rows > LM_SKINNY_ROWS_MID )
		return(LmSkinnySplitMultiShape<Format,LM_SKINNY_ROWS>(multi,stream));
	if ( LmSkinnyMultiprocessors(&multiprocessors) != LM_LAUNCH_OK )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	enough = (uint64_t)multiprocessors * LM_SKINNY_SPLIT_BLOCKS_PER_SM > outputs ? (uint64_t)multiprocessors * LM_SKINNY_SPLIT_BLOCKS_PER_SM : outputs;
	if ( outputs * rows <= enough )
		return(LmSkinnySplitMultiShape<Format,1u>(multi,stream));
	if ( outputs * LmSkinnyRowBlocks<2u>(rows) <= enough )
		return(LmSkinnySplitMultiShape<Format,2u>(multi,stream));
	return(LmSkinnySplitMultiShape<Format,LM_SKINNY_ROWS_MID>(multi,stream));
}

template<class Format, uint32_t LANES, uint32_t U, uint32_t NPG, bool CACHE_WEIGHT = false>
static __device__ __forceinline__ void LmSkinnyFrameLoad(const LmSkinnyArguments &args, uint32_t neuron, uint32_t chunk, uint32_t chunks, uint4 (*weight)[NPG], typename LmSkinnyRawScale<Format>::Raw (*raw_scale)[NPG])
{
	const uint4 *rows[NPG];
	uint32_t u, n, c;
	LmSkinnyWeightRows<Format,NPG>(args,0u,neuron,rows);
	#pragma unroll
	for ( u = 0u; u < U; u++ )
	{
		c = chunk + u * LANES;
		#pragma unroll
		for ( n = 0u; n < NPG; n++ )
		{
			const uint32_t row_neuron = neuron + n < args.output_dimension ? neuron + n : neuron;
			weight[u][n] = c < chunks ? (CACHE_WEIGHT ? __ldg(rows[n] + c) : __ldcs(rows[n] + c)) : make_uint4(0u,0u,0u,0u);
			if constexpr ( Format::kScaleGroup != 0u )
				raw_scale[u][n] = c < chunks ? LmSkinnyRawScale<Format>::Load(&args.scale,0u,row_neuron,c) : LmSkinnyRawScale<Format>::Identity();
		}
	}
}

template<class Format, uint32_t RG, uint32_t NPG, bool FULL_ROWS>
static __device__ __forceinline__ void LmSkinnyFrameDecodeOnceRows(const uint16_t *activation, uint64_t stride, uint32_t count, const typename LmSkinnyFormat<Format>::Decoded *decoded, const float *sc, float (*accumulator)[RG])
{
	uint32_t r, n;
	#pragma unroll
	for ( r = 0u; r < RG; r++ )
	{
		const typename LmSkinnyFormat<Format>::Activation a = LmSkinnyFormat<Format>::Load(activation + (FULL_ROWS || r < count ? r : 0u) * stride);
		float af[16];
		LmSkinnyUnpackBf16x8(a.low,af);
		LmSkinnyUnpackBf16x8(a.high,af + 8);
		#pragma unroll
		for ( n = 0u; n < NPG; n++ )
			accumulator[n][r] = fmaf(sc[n],LmSkinnyFormat<Format>::ChunkDecodedUnpacked(decoded[n],af),accumulator[n][r]);
	}
}

template<uint32_t RG, uint32_t NPG, bool FULL_ROWS>
static __device__ __forceinline__ void LmSkinnyFrameBf16Rows(const uint16_t *activation, uint64_t stride, uint32_t count, const float (*wf)[8], float (*accumulator)[RG])
{
	uint32_t r, n;
	#pragma unroll
	for ( r = 0u; r < RG; r++ )
	{
		const uint4 a = LmSkinnyFormat<LmBf16Format>::Load(activation + (FULL_ROWS || r < count ? r : 0u) * stride);
		float af[8];
		LmSkinnyUnpackBf16x8(a,af);
		#pragma unroll
		for ( n = 0u; n < NPG; n++ )
			accumulator[n][r] = fmaf(1.0f,LmSkinnyBf16DotUnpacked(wf[n],af),accumulator[n][r]);
	}
}

template<class Format, uint32_t LANES, uint32_t RG, uint32_t U, uint32_t NPG, bool FULL_ROWS = false>
static __device__ __forceinline__ void LmSkinnyFrameCompute(const uint16_t *activation, uint64_t stride, uint32_t count, uint32_t chunk, uint32_t chunks, float global, const uint4 (*weight)[NPG], const typename LmSkinnyRawScale<Format>::Raw (*raw_scale)[NPG], float (*accumulator)[RG])
{
	constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements;
	uint32_t u, n, r, c;
	#pragma unroll
	for ( u = 0u; u < U; u++ )
	{
		c = chunk + u * LANES;
		if ( c >= chunks )
			break;
		if constexpr ( LmSkinnyDecodeHalves<Format>::kValue )
		{
			float part[NPG][RG];
			float2 split_pair[NPG];
			float scale_once[NPG];
			#pragma unroll
			for ( n = 0u; n < NPG; n++ )
			{
				if constexpr ( LmSkinnySplitScale<Format>::kSplit )
					split_pair[n] = LmSkinnyRawScale<Format>::DecodePair(raw_scale[u][n],global);
				else
					scale_once[n] = LmSkinnyRawScale<Format>::Decode(raw_scale[u][n]);
			}
			#pragma unroll
			for ( uint32_t half = 0u; half < 2u; half++ )
			{
				LmSkinnyE2m1HalfDecoded dec[NPG];
				#pragma unroll
				for ( n = 0u; n < NPG; n++ )
					dec[n] = half == 0u ? LmSkinnyDecodeE2m1Half(weight[u][n].x,weight[u][n].y)
										: LmSkinnyDecodeE2m1Half(weight[u][n].z,weight[u][n].w);
				#pragma unroll
				for ( r = 0u; r < RG; r++ )
				{
					const LmSkinnyWide a = LmSkinnyFormat<Format>::LoadHalf(activation + (FULL_ROWS || r < count ? r : 0u) * stride + c * elements,half);
					float af[16];
					LmSkinnyUnpackBf16x8(a.low,af);
					LmSkinnyUnpackBf16x8(a.high,af + 8);
					#pragma unroll
					for ( n = 0u; n < NPG; n++ )
					{
						const float p = LmSkinnyDotPairs16(dec[n].pair,af);
						if constexpr ( LmSkinnySplitScale<Format>::kSplit )
							accumulator[n][r] = fmaf(half == 0u ? split_pair[n].x : split_pair[n].y,p,accumulator[n][r]);
						else if ( half == 0u )
							part[n][r] = p;
						else
							accumulator[n][r] = fmaf(scale_once[n],part[n][r] + p,accumulator[n][r]);
					}
				}
			}
		}
		else if constexpr ( LmSkinnyDecodeOnce<Format>::kValue )
		{
			if constexpr ( NPG == 1u )
			{
				const typename LmSkinnyFormat<Format>::Decoded decoded = LmSkinnyFormat<Format>::Decode(weight[u][0]);
				const float sc = LmSkinnyRawScale<Format>::Decode(raw_scale[u][0]);
				#pragma unroll
				for ( r = 0u; r < RG; r++ )
					accumulator[0][r] = fmaf(sc,LmSkinnyFormat<Format>::ChunkDecoded(decoded,LmSkinnyFormat<Format>::Load(activation + (FULL_ROWS || r < count ? r : 0u) * stride + c * elements)),accumulator[0][r]);
			}
			else
			{
				typename LmSkinnyFormat<Format>::Decoded decoded[NPG];
				float sc[NPG];
				#pragma unroll
				for ( n = 0u; n < NPG; n++ )
				{
					decoded[n] = LmSkinnyFormat<Format>::Decode(weight[u][n]);
					sc[n] = LmSkinnyRawScale<Format>::Decode(raw_scale[u][n]);
				}
				LmSkinnyFrameDecodeOnceRows<Format,RG,NPG,FULL_ROWS>(activation + c * elements,stride,count,decoded,sc,accumulator);
			}
		}
		else
		{
			if constexpr ( NPG == 1u )
			{
				const float sc = LmSkinnyRawScale<Format>::Decode(raw_scale[u][0]);
				#pragma unroll
				for ( r = 0u; r < RG; r++ )
					accumulator[0][r] = fmaf(sc,LmSkinnyFormat<Format>::Chunk(weight[u][0],LmSkinnyFormat<Format>::Load(activation + (FULL_ROWS || r < count ? r : 0u) * stride + c * elements)),accumulator[0][r]);
			}
			else if constexpr ( Format::kScaleGroup == 0u )
			{
				float wf[NPG][8];
				#pragma unroll
				for ( n = 0u; n < NPG; n++ )
					LmSkinnyUnpackBf16x8(weight[u][n],wf[n]);
				LmSkinnyFrameBf16Rows<RG,NPG,FULL_ROWS>(activation + c * elements,stride,count,wf,accumulator);
			}
			else
			{
				typename LmSkinnyFormat<Format>::Activation staged[RG];
				#pragma unroll
				for ( r = 0u; r < RG; r++ )
					staged[r] = LmSkinnyFormat<Format>::Load(activation + (FULL_ROWS || r < count ? r : 0u) * stride + c * elements);
				#pragma unroll
				for ( n = 0u; n < NPG; n++ )
				{
					const float sc = LmSkinnyRawScale<Format>::Decode(raw_scale[u][n]);
					#pragma unroll
					for ( r = 0u; r < RG; r++ )
						accumulator[n][r] = fmaf(sc,LmSkinnyFormat<Format>::Chunk(weight[u][n],staged[r]),accumulator[n][r]);
				}
			}
		}
	}
}

template<class Format, uint32_t LANES, uint32_t RG, uint32_t NPG, bool FULL_ROWS>
static __device__ __forceinline__ void LmSkinnyFrameDirectLoop(const LmSkinnyArguments &args, const uint16_t *activation, uint64_t stride, uint32_t count, uint32_t neuron, uint32_t sub, uint32_t chunks, float global, float (*accumulator)[RG])
{
	using RawScale = typename LmSkinnyRawScale<Format>::Raw;
	const uint4 *rows[NPG];
	const uint8_t *scale_rows[NPG];
	uint32_t n;
	LmSkinnyWeightRows<Format,NPG>(args,0u,neuron,rows);
	#pragma unroll
	for ( n = 0u; n < NPG; n++ )
	{
		const uint32_t row_neuron = neuron + n < args.output_dimension ? neuron + n : neuron;
		scale_rows[n] = LmSkinnyRawScale<Format>::RowBase(&args.scale,0u,row_neuron);
	}
	for ( uint32_t c = sub; c < chunks; c += LANES )
	{
		uint4 weight[1][NPG];
		RawScale raw_scale[1][NPG];
		#pragma unroll
		for ( n = 0u; n < NPG; n++ )
		{
			weight[0][n] = __ldcs(rows[n] + c);
			if constexpr ( Format::kScaleGroup != 0u )
				raw_scale[0][n] = LmSkinnyRawScale<Format>::LoadFromRow(scale_rows[n],c);
		}
		LmSkinnyFrameCompute<Format,LANES,RG,1u,NPG,FULL_ROWS>(activation,stride,count,c,chunks,global,weight,raw_scale,accumulator);
	}
}

template<bool MULTI>
static __device__ __forceinline__ uint32_t LmSkinnyFramePart(const LmSkinnyMultiArguments &multi, uint32_t neuron_task)
{
	uint32_t part = 0u;
	if constexpr ( MULTI )
		while ( part + 1u < multi.count && neuron_task >= multi.first_task[part + 1u] )
			part++;
	return(part);
}

template<class Format, uint32_t RG, uint32_t NPG>
struct LmSkinnyFrameConfig
{
	static constexpr uint32_t kThreads = ((RG >= 4u && NPG >= 4u) || (RG >= 8u && NPG >= 2u) || RG >= 16u) ? 128u : LM_SKINNY_THREADS;
	static constexpr uint32_t kMinBlocks = (RG >= 16u && NPG >= 4u) ? 2u :
		(RG == 8u && NPG >= 4u) ? (Format::kScaleGroup == 0u ? 3u : 2u) :
		(RG == 4u && NPG >= 4u) ? (LmSkinnySplitScale<Format>::kSplit ? 2u : 3u) :
		(RG >= 8u && NPG == 2u) ? 3u :
		LM_SKINNY_FRAME_MIN_BLOCKS;
};

template<class Format, uint32_t LANES, uint32_t RG, uint32_t NPG, bool MULTI>
__global__ __launch_bounds__(LmSkinnyFrameConfig<Format,RG,NPG>::kThreads,LmSkinnyFrameConfig<Format,RG,NPG>::kMinBlocks) void LmSkinnyFrameKernel(const __grid_constant__ LmSkinnyMultiArguments multi)
{
	using RawScale = typename LmSkinnyRawScale<Format>::Raw;
	constexpr uint32_t frame_depth = RG >= 8u && LM_SKINNY_FRAME_DEPTH >= 2u ? LM_SKINNY_FRAME_DEPTH / 2u : LM_SKINNY_FRAME_DEPTH;
	constexpr uint32_t U = (RG >= 16u || (RG >= 8u && NPG >= 2u) || (LmSkinnyDecodeHalves<Format>::kValue && RG >= 4u)) ? 1u : (frame_depth / NPG != 0u ? frame_depth / NPG : 1u);
	constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements, lane_groups = LmSkinnyFrameConfig<Format,RG,NPG>::kThreads / LANES;
	constexpr bool direct_loop = (RG >= 4u && NPG >= 4u);
	const LmSkinnyArguments &head = multi.part[0];
	const uint32_t groups = (head.rows + RG - 1u) / RG;
	const uint32_t neuron_tasks = multi.first_task[multi.count], tasks = neuron_tasks * groups, stride = gridDim.x * lane_groups;
	const uint32_t chunks = head.input_dimension / elements, sub = threadIdx.x % LANES;
	uint32_t base = blockIdx.x * lane_groups, task = base + threadIdx.x / LANES;
	float accumulator[NPG][RG];
	if constexpr ( direct_loop )
	{
		const uint32_t warp = threadIdx.x / LANES;
		const bool swizzle = groups > lane_groups && (groups % lane_groups) == 0u;
		const uint32_t m_tiles = swizzle ? groups / lane_groups : 1u;
		const uint32_t stripe_blocks = gridDim.x * m_tiles;
		LmDependentWait();
		LmDependentRelease();
		while ( base < tasks )
		{
			uint32_t live, nt, g;
			if ( swizzle )
			{
				const uint32_t b = base / lane_groups, s = b / stripe_blocks, rem = b - s * stripe_blocks;
				const uint32_t s_start = s * gridDim.x;
				const uint32_t s_width = neuron_tasks - s_start < gridDim.x ? neuron_tasks - s_start : gridDim.x;
				const uint32_t mt = rem / s_width;
				nt = s_start + (rem - mt * s_width);
				g = mt * lane_groups + warp;
				live = 1u;
			}
			else
			{
				live = task < tasks ? 1u : 0u;
				nt = live != 0u ? task / groups : 0u;
				g = live != 0u ? task % groups : 0u;
			}
			const uint32_t part = live != 0u ? LmSkinnyFramePart<MULTI>(multi,nt) : 0u;
			const uint32_t first = live != 0u ? g * RG : 0u;
			const uint32_t count = head.rows - first < RG ? head.rows - first : RG;
			const uint32_t neuron = live != 0u ? (nt - multi.first_task[part]) * NPG : 0u;
			const float global = LmSkinnyScaleGlobal<Format>(&multi.part[part].scale,0u);
			LmSkinnyClear<NPG,RG>(accumulator);
			if ( live != 0u )
			{
				const uint16_t *act = head.activation + (uint64_t)first * head.input_dimension;
				if ( count == RG )
					LmSkinnyFrameDirectLoop<Format,LANES,RG,NPG,true>(multi.part[part],act,head.input_dimension,count,neuron,sub,chunks,global,accumulator);
				else
					LmSkinnyFrameDirectLoop<Format,LANES,RG,NPG,false>(multi.part[part],act,head.input_dimension,count,neuron,sub,chunks,global,accumulator);
			}
			LmSkinnyStore<LANES,RG,NPG>(multi.part[part],accumulator,sub,live,first,neuron,count);
			base += stride;
			task += stride;
		}
		return;
	}
	const uint32_t iterations = ((chunks + LANES - 1u) / LANES + U - 1u) / U;
	uint32_t it = 0u, part, u, n;
	uint4 weight[U][NPG], next_weight[U][NPG];
	RawScale raw_scale[U][NPG], next_raw_scale[U][NPG];
	part = LmSkinnyFramePart<MULTI>(multi,task / groups);
	float global = LmSkinnyScaleGlobal<Format>(&multi.part[part].scale,0u);
	if ( task < tasks )
		LmSkinnyFrameLoad<Format,LANES,U,NPG>(multi.part[part],(task / groups - multi.first_task[part]) * NPG,sub,chunks,weight,raw_scale);
	LmDependentWait();
	LmDependentRelease();
	LmSkinnyClear<NPG,RG>(accumulator);
	while ( base < tasks )
	{
		const uint32_t next_it = it + 1u < iterations ? it + 1u : 0u, next_task = next_it == 0u ? task + stride : task;
		const uint32_t next_part = next_it == 0u ? LmSkinnyFramePart<MULTI>(multi,next_task / groups) : part;
		if ( next_task < tasks )
			LmSkinnyFrameLoad<Format,LANES,U,NPG>(multi.part[next_part],(next_task / groups - multi.first_task[next_part]) * NPG,sub + next_it * U * LANES,chunks,next_weight,next_raw_scale);
		const uint32_t live = task < tasks ? 1u : 0u, first = live != 0u ? (task % groups) * RG : 0u, count = head.rows - first < RG ? head.rows - first : RG;
		const uint32_t neuron = live != 0u ? (task / groups - multi.first_task[part]) * NPG : 0u;
		if ( live != 0u )
			LmSkinnyFrameCompute<Format,LANES,RG,U,NPG>(head.activation + (uint64_t)first * head.input_dimension,head.input_dimension,count,sub + it * U * LANES,chunks,global,weight,raw_scale,accumulator);
		if ( next_it == 0u )
		{
			LmSkinnyStore<LANES,RG,NPG>(multi.part[part],accumulator,sub,live,first,neuron,count);
			LmSkinnyClear<NPG,RG>(accumulator);
			base += stride;
			if constexpr ( MULTI && LmSkinnySplitScale<Format>::kSplit )
				if ( next_part != part && next_task < tasks )
					global = LmSkinnyScaleGlobal<Format>(&multi.part[next_part].scale,0u);
		}
		#pragma unroll
		for ( u = 0u; u < U; u++ )
			#pragma unroll
			for ( n = 0u; n < NPG; n++ )
			{
				weight[u][n] = next_weight[u][n];
				if constexpr ( Format::kScaleGroup != 0u )
					raw_scale[u][n] = next_raw_scale[u][n];
			}
		it = next_it;
		task = next_task;
		part = next_part;
	}
}

#if defined(__CUDACC__)
static __device__ __forceinline__ void LmSkinnyCpAsync16(void *destination, const void *source, uint32_t bytes)
{
	const uint32_t smem = (uint32_t)__cvta_generic_to_shared(destination);
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem), "l"(source), "r"(bytes) : "memory");
}

template<uint32_t WARPS_M, uint32_t WARPS_N, uint32_t RPG, uint32_t NPG>
static __host__ __device__ constexpr uint32_t LmSkinnySharedBf16Bytes(void)
{
	return(2u * ((WARPS_N * NPG) + (WARPS_M * RPG)) * LM_WARP_LANES * (uint32_t)sizeof(uint4));
}

template<uint32_t WARPS_M, uint32_t WARPS_N, uint32_t RPG, uint32_t NPG, uint32_t MIN_BLK, bool MULTI>
__global__ __launch_bounds__(WARPS_M * WARPS_N * LM_WARP_LANES, MIN_BLK) void LmSkinnySharedBf16Kernel(const __grid_constant__ LmSkinnyMultiArguments multi)
{
	constexpr uint32_t THREADS = WARPS_M * WARPS_N * LM_WARP_LANES;
	constexpr uint32_t TILE_M = WARPS_M * RPG, TILE_N = WARPS_N * NPG;
	constexpr uint32_t TOTAL_W = TILE_N * LM_WARP_LANES, TOTAL_A = TILE_M * LM_WARP_LANES;
	extern __shared__ __align__(16) uint4 lm_skinny_shared[];
	uint4 (*s_w)[TILE_N][LM_WARP_LANES] = (uint4 (*)[TILE_N][LM_WARP_LANES])lm_skinny_shared;
	uint4 (*s_a)[TILE_M][LM_WARP_LANES] = (uint4 (*)[TILE_M][LM_WARP_LANES])(lm_skinny_shared + 2u * TOTAL_W);
	const uint32_t part = LmSkinnyFramePart<MULTI>(multi,blockIdx.x);
	const LmSkinnyArguments &args = multi.part[part];
	const uint32_t n_blk = (blockIdx.x - multi.first_task[part]) * TILE_N, m_blk = blockIdx.y * TILE_M;
	const uint32_t warp = threadIdx.x >> 5u, sub = threadIdx.x & 31u;
	const uint32_t wm = warp / WARPS_N, wn_w = warp % WARPS_N;
	const uint32_t warp_r0 = wm * RPG, warp_n0 = wn_w * NPG;
	const uint32_t global_r0 = m_blk + warp_r0, global_n0 = n_blk + warp_n0;
	const uint32_t row_cnt = global_r0 < args.rows ? (args.rows - global_r0 < RPG ? args.rows - global_r0 : RPG) : 0u;
	const uint32_t chunks = args.input_dimension >> 3u;
	const uint4 *w4 = (const uint4 *)args.weight, *a4 = (const uint4 *)args.activation;
	auto stage = [&](uint32_t st, uint32_t k_base)
	{
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < TOTAL_W; idx += THREADS )
		{
			const uint32_t tn = idx >> 5u, wc = idx & 31u, neuron = n_blk + tn, c = k_base + wc;
			const bool ok = neuron < args.output_dimension && c < chunks;
			LmSkinnyCpAsync16(&s_w[st][tn][wc],ok ? (w4 + (uint64_t)neuron * chunks + c) : w4,ok ? 16u : 0u);
		}
		#pragma unroll
		for ( uint32_t idx = threadIdx.x; idx < TOTAL_A; idx += THREADS )
		{
			const uint32_t tm = idx >> 5u, ac = idx & 31u, row = m_blk + tm, c = k_base + ac;
			const bool ok = row < args.rows && c < chunks;
			LmSkinnyCpAsync16(&s_a[st][tm][ac],ok ? (a4 + (uint64_t)row * chunks + c) : a4,ok ? 16u : 0u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	LmDependentWait();
	LmDependentRelease();
	stage(0u,0u);
	float accumulator[NPG][RPG];
	LmSkinnyClear<NPG,RPG>(accumulator);
	for ( uint32_t k_base = 0u; k_base < chunks; k_base += LM_WARP_LANES )
	{
		const uint32_t cur = (k_base >> 5u) & 1u;
		if ( k_base + LM_WARP_LANES < chunks )
		{
			stage(cur ^ 1u,k_base + LM_WARP_LANES);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		if ( row_cnt != 0u && k_base + sub < chunks )
		{
			float wf[NPG][8];
			#pragma unroll
			for ( uint32_t n = 0u; n < NPG; n++ )
				LmSkinnyUnpackBf16x8(s_w[cur][warp_n0 + n][sub],wf[n]);
			if ( row_cnt == RPG )
			{
				#pragma unroll
				for ( uint32_t r = 0u; r < RPG; r++ )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(s_a[cur][warp_r0 + r][sub],af);
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; n++ )
						accumulator[n][r] = fmaf(1.0f,LmSkinnyBf16DotUnpacked(wf[n],af),accumulator[n][r]);
				}
			}
			else
			{
				#pragma unroll
				for ( uint32_t r = 0u; r < RPG; r++ )
				{
					float af[8];
					LmSkinnyUnpackBf16x8(s_a[cur][warp_r0 + (r < row_cnt ? r : 0u)][sub],af);
					#pragma unroll
					for ( uint32_t n = 0u; n < NPG; n++ )
						accumulator[n][r] = fmaf(1.0f,LmSkinnyBf16DotUnpacked(wf[n],af),accumulator[n][r]);
				}
			}
		}
		if ( k_base + LM_WARP_LANES < chunks )
			__syncthreads();
	}
	if ( row_cnt != 0u )
		LmSkinnyStore<LM_WARP_LANES,RPG,NPG>(args,accumulator,sub,1u,global_r0,global_n0,row_cnt);
}
#endif

template<uint32_t ROWS>
static __device__ __forceinline__ void LmSkinnyGroupedActivation(const LmSkinnyArguments &args, uint32_t row, uint32_t count, const uint16_t **activation)
{
	uint32_t r, packed;
	#pragma unroll
	for ( r = 0u; r < ROWS; r++ )
	{
		packed = row + (r < count ? r : 0u);
		activation[r] = args.activation + (uint64_t)(args.activation_packed != 0u ? packed : args.route_source_token[packed]) * args.input_dimension;
	}
}

template<uint32_t ROWS>
static __device__ __forceinline__ void LmSkinnyGroupedBatchScan(const uint32_t *offset, uint32_t groups, uint32_t *batch_first, uint32_t *warp_total)
{
	const uint32_t span = (groups + LM_SKINNY_THREADS - 1u) / LM_SKINNY_THREADS, first = threadIdx.x * span;
	const uint32_t last = first + span < groups ? first + span : groups, lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	uint32_t index, local = 0u, inclusive, value, prefix = 0u;
	for ( index = first; index < last; index++ )
		local += (offset[index + 1u] - offset[index] + ROWS - 1u) / ROWS;
	inclusive = local;
	#pragma unroll
	for ( index = 1u; index < 32u; index <<= 1u )
	{
		value = __shfl_up_sync(0xffffffffu,inclusive,index);
		if ( lane >= index )
			inclusive += value;
	}
	if ( lane == 31u )
		warp_total[warp] = inclusive;
	__syncthreads();
	for ( index = 0u; index < warp; index++ )
		prefix += warp_total[index];
	prefix += inclusive - local;
	for ( index = first; index < last; index++ )
	{
		batch_first[index] = prefix;
		prefix += (offset[index + 1u] - offset[index] + ROWS - 1u) / ROWS;
	}
	if ( threadIdx.x == LM_SKINNY_THREADS - 1u )
		batch_first[groups] = prefix;
	__syncthreads();
}

template<class Format, uint32_t ROWS, uint32_t NPG>
static __device__ __forceinline__ void LmSkinnyGroupedAccumulateStep(const LmSkinnyArguments &args, const uint4 *weight, const uint32_t *act_row, uint32_t group, uint32_t neuron, uint32_t c, float global, float (*accumulator)[ROWS])
{
	constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements;
	uint32_t n, r;
	if constexpr ( LmSkinnyDecodeHalves<Format>::kValue && ROWS > 1u )
	{
		float part[NPG][ROWS];
		#pragma unroll
		for ( uint32_t half = 0u; half < 2u; half++ )
		{
			LmSkinnyE2m1HalfDecoded dec[NPG];
			float scale[NPG];
			#pragma unroll
			for ( n = 0u; n < NPG; n++ )
			{
				const uint32_t row_neuron = neuron + n < args.output_dimension ? neuron + n : neuron;
				dec[n] = half == 0u ? LmSkinnyDecodeE2m1Half(weight[n].x,weight[n].y) : LmSkinnyDecodeE2m1Half(weight[n].z,weight[n].w);
				if constexpr ( LmSkinnySplitScale<Format>::kSplit )
					scale[n] = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,row_neuron,c * elements + half * Format::kScaleGroup,global);
				else if ( half != 0u )
					scale[n] = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,row_neuron,c * elements,global);
			}
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
			{
				const LmSkinnyWide a = LmSkinnyFormat<Format>::LoadHalf(args.activation + (uint64_t)act_row[r] * args.input_dimension + c * elements,half);
				float af[16];
				LmSkinnyUnpackBf16x8(a.low,af);
				LmSkinnyUnpackBf16x8(a.high,af + 8);
				#pragma unroll
				for ( n = 0u; n < NPG; n++ )
				{
					const float p = LmSkinnyDotPairs16(dec[n].pair,af);
					if constexpr ( LmSkinnySplitScale<Format>::kSplit )
						accumulator[n][r] = fmaf(scale[n],p,accumulator[n][r]);
					else if ( half == 0u )
						part[n][r] = p;
					else
						accumulator[n][r] = fmaf(scale[n],part[n][r] + p,accumulator[n][r]);
				}
			}
		}
	}
	else if constexpr ( ROWS > 1u && LmSkinnyDecodeOnce<Format>::kValue )
	{
		typename LmSkinnyFormat<Format>::Decoded decoded[NPG];
		float scale[NPG];
		#pragma unroll
		for ( n = 0u; n < NPG; n++ )
		{
			const uint32_t row_neuron = neuron + n < args.output_dimension ? neuron + n : neuron;
			scale[n] = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,row_neuron,c * elements,global);
			decoded[n] = LmSkinnyFormat<Format>::Decode(weight[n]);
		}
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
		{
			const typename LmSkinnyFormat<Format>::Activation a = LmSkinnyFormat<Format>::Load(args.activation + (uint64_t)act_row[r] * args.input_dimension + c * elements);
			float af[16];
			LmSkinnyUnpackBf16x8(a.low,af);
			LmSkinnyUnpackBf16x8(a.high,af + 8);
			#pragma unroll
			for ( n = 0u; n < NPG; n++ )
				accumulator[n][r] = fmaf(scale[n],LmSkinnyFormat<Format>::ChunkDecodedUnpacked(decoded[n],af),accumulator[n][r]);
		}
	}
	else
	{
		typename LmSkinnyFormat<Format>::Activation staged[ROWS];
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
			staged[r] = LmSkinnyFormat<Format>::Load(args.activation + (uint64_t)act_row[r] * args.input_dimension + c * elements);
		#pragma unroll
		for ( n = 0u; n < NPG; n++ )
		{
			const uint32_t row_neuron = neuron + n < args.output_dimension ? neuron + n : neuron;
			const float scale = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,row_neuron,c * elements,global);
			if constexpr ( LmSkinnySplitScale<Format>::kSplit )
			{
				const float high = LmSkinnyScaleWithGlobal<Format>(&args.scale,group,row_neuron,c * elements + Format::kScaleGroup,global);
				#pragma unroll
				for ( r = 0u; r < ROWS; r++ )
				{
					const float2 part = LmSkinnyFormat<Format>::Halves(weight[n],staged[r]);
					accumulator[n][r] = fmaf(high,part.y,fmaf(scale,part.x,accumulator[n][r]));
				}
			}
			else
			{
				#pragma unroll
				for ( r = 0u; r < ROWS; r++ )
					accumulator[n][r] = fmaf(scale,LmSkinnyFormat<Format>::Chunk(weight[n],staged[r]),accumulator[n][r]);
			}
		}
	}
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
static __device__ __forceinline__ void LmSkinnyGroupedRun(const LmSkinnyArguments &args, uint32_t group, uint32_t row, uint32_t count, uint32_t neuron, uint32_t sub, uint32_t chunks)
{
	float accumulator[NPG][ROWS];
	if constexpr ( ROWS > LM_SKINNY_ROWS_MID && NPG > 2u )
	{
		const uint8_t *weight_group = args.weight + (uint64_t)group * args.weight_group_bytes;
		const float global = LmSkinnyScaleGlobal<Format>(&args.scale,group);
		const uint4 *wrows[NPG];
		uint32_t act_row[ROWS];
		uint32_t r, base, n;
		if constexpr ( !LmSkinnyDecodeHalves<Format>::kValue )
			LmSkinnyWeightRows<Format,NPG>(args,group,neuron,wrows);
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
		{
			const uint32_t packed = row + (r < count ? r : 0u);
			act_row[r] = args.activation_packed != 0u ? packed : args.route_source_token[packed];
		}
		LmSkinnyClear<NPG,ROWS>(accumulator);
		for ( base = sub; base < chunks; base += LANES )
		{
			uint4 weight[NPG];
			#pragma unroll
			for ( n = 0u; n < NPG; n++ )
			{
				if constexpr ( !LmSkinnyDecodeHalves<Format>::kValue )
					weight[n] = __ldcs(wrows[n] + base);
				else
				{
					const uint32_t row_neuron = neuron + n < args.output_dimension ? neuron + n : neuron;
					weight[n] = __ldcs((const uint4 *)(weight_group + (uint64_t)row_neuron * chunks * LM_SKINNY_CHUNK_BYTES) + base);
				}
			}
			LmSkinnyGroupedAccumulateStep<Format,ROWS,NPG>(args,weight,act_row,group,neuron,base,global,accumulator);
		}
		LmSkinnyStore<LANES,ROWS,NPG>(args,accumulator,sub,1u,row,neuron,count);
		return;
	}
	else
	{
		const uint4 *weights[NPG];
		const uint16_t *activation[ROWS];
		LmSkinnyWeightRows<Format,NPG>(args,group,neuron,weights);
		LmSkinnyGroupedActivation<ROWS>(args,row,count,activation);
		LmSkinnyClear<NPG,ROWS>(accumulator);
		LmSkinnyAccumulate<Format,LANES,ROWS,NPG>(args,weights,activation,group,neuron,sub,chunks,accumulator);
		LmSkinnyStore<LANES,ROWS,NPG>(args,accumulator,sub,1u,row,neuron,count);
	}
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG, bool BATCHED>
__global__ __launch_bounds__(LM_SKINNY_THREADS,1) void LmSkinnyGroupedKernel(const __grid_constant__ LmSkinnyArguments args)
{
	__shared__ uint32_t offset[LM_SKINNY_GROUPED_MAX_GROUPS + 1u];
	__shared__ uint32_t batch_first[BATCHED ? LM_SKINNY_GROUPED_MAX_GROUPS + 1u : 1u];
	__shared__ uint32_t warp_total[LM_SKINNY_THREADS / 32u];
	const uint32_t sub = threadIdx.x % LANES, task = (blockIdx.x * LM_SKINNY_THREADS + threadIdx.x) / LANES, per_group = (args.output_dimension + NPG - 1u) / NPG;
	const uint32_t item = task / per_group, neuron = (task % per_group) * NPG;
	const uint32_t chunks = args.input_dimension / LmSkinnyFormat<Format>::kElements;
	const uint32_t *first = BATCHED ? batch_first : offset;
	uint32_t index, group, high, middle, count, row;
	LmDependentWait();
	LmDependentRelease();
	if ( (blockIdx.x * (LM_SKINNY_THREADS / LANES)) / per_group >= (BATCHED ? (args.group_row_offset[args.groups] + ROWS - 1u) / ROWS + args.groups : args.group_row_offset[args.groups]) )
		return;
	for ( index = threadIdx.x; index <= args.groups; index += LM_SKINNY_THREADS )
		offset[index] = args.group_row_offset[index];
	__syncthreads();
	if constexpr ( BATCHED )
		LmSkinnyGroupedBatchScan<ROWS>(offset,args.groups,batch_first,warp_total);
	if ( item >= first[args.groups] )
		return;
	group = 0u;
	high = args.groups - 1u;
	while ( group < high )
	{
		middle = (group + high + 1u) / 2u;
		if ( first[middle] <= item )
			group = middle;
		else
			high = middle - 1u;
	}
	if ( !BATCHED && (item - offset[group]) % ROWS != 0u )
		return;
	row = BATCHED ? offset[group] + (item - batch_first[group]) * ROWS : item;
	count = offset[group + 1u] - row < ROWS ? offset[group + 1u] - row : ROWS;
	if constexpr ( ROWS == LM_SKINNY_ROWS_MID )
	{
		if ( count == 1u )
		{
			LmSkinnyGroupedRun<Format,LANES,1u,NPG>(args,group,row,1u,neuron,sub,chunks);
			return;
		}
		if ( count == 2u )
		{
			LmSkinnyGroupedRun<Format,LANES,2u,NPG>(args,group,row,2u,neuron,sub,chunks);
			return;
		}
		if ( count == 3u )
		{
			LmSkinnyGroupedRun<Format,LANES,3u,NPG>(args,group,row,3u,neuron,sub,chunks);
			return;
		}
	}
	else if constexpr ( ROWS == 2u )
	{
		if ( count == 1u )
		{
			LmSkinnyGroupedRun<Format,LANES,1u,NPG>(args,group,row,1u,neuron,sub,chunks);
			return;
		}
	}
	LmSkinnyGroupedRun<Format,LANES,ROWS,NPG>(args,group,row,count,neuron,sub,chunks);
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
static int32_t LmSkinnyLaunchShape(const LmSkinnyArguments *args, SparkGpuStream stream)
{
	uint64_t tasks = (uint64_t)(args->route_expert != 0 ? args->pairs : 1u) * ((args->output_dimension + NPG - 1u) / NPG);
	LM_LAUNCH_DEPENDENT((LmSkinnyKernel<Format,LANES,ROWS,NPG>),(uint32_t)((tasks * LANES + LM_SKINNY_THREADS - 1u) / LM_SKINNY_THREADS) * LmSkinnyRowBlocks<ROWS>(args->rows),LM_SKINNY_THREADS,0u,stream,*args);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
static int32_t LmSkinnyGroupedShape(const LmSkinnyArguments *args, SparkGpuStream stream)
{
	const uint32_t per_group = (args->output_dimension + NPG - 1u) / NPG;
	const uint64_t batches = (args->pairs + ROWS - 1u) / ROWS + (uint64_t)args->groups;
	const bool batched = ROWS >= LM_SKINNY_ROWS_MID && batches < args->pairs;
	uint64_t tasks = (batched ? batches : (uint64_t)args->pairs) * per_group;
	if ( (per_group % (32u / LANES)) != 0u || args->groups > LM_SKINNY_GROUPED_MAX_GROUPS )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( batched )
		LM_LAUNCH_DEPENDENT((LmSkinnyGroupedKernel<Format,LANES,ROWS,NPG,true>),(uint32_t)((tasks * LANES + LM_SKINNY_THREADS - 1u) / LM_SKINNY_THREADS),LM_SKINNY_THREADS,0u,stream,*args);
	else
		LM_LAUNCH_DEPENDENT((LmSkinnyGroupedKernel<Format,LANES,ROWS,NPG,false>),(uint32_t)((tasks * LANES + LM_SKINNY_THREADS - 1u) / LM_SKINNY_THREADS),LM_SKINNY_THREADS,0u,stream,*args);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

template<class Format, uint32_t ROWS, uint32_t NPG>
static int32_t LmSkinnyGroupedLanes(const LmSkinnyArguments *args, uint32_t chunks, SparkGpuStream stream)
{
	if ( chunks >= 32u )
		return(LmSkinnyGroupedShape<Format,32u,ROWS,NPG>(args,stream));
	if ( chunks >= 16u )
		return(LmSkinnyGroupedShape<Format,16u,ROWS,NPG>(args,stream));
	return(LmSkinnyGroupedShape<Format,8u,ROWS,NPG>(args,stream));
}

template<class Format, uint32_t LANES, uint32_t ROWS, uint32_t NPG>
static int32_t LmSkinnyMultiShape(LmSkinnyMultiArguments *multi, SparkGpuStream stream)
{
	uint32_t part;
	multi->first_task[0] = 0u;
	for ( part = 0u; part < multi->count; part++ )
		multi->first_task[part + 1u] = multi->first_task[part] + ((multi->part[part].output_dimension + NPG - 1u) / NPG + (32u / LANES) - 1u) / (32u / LANES) * (32u / LANES);
	LM_LAUNCH_DEPENDENT((LmSkinnyMultiKernel<Format,LANES,ROWS,NPG>),(uint32_t)(((uint64_t)multi->first_task[multi->count] * LANES + LM_SKINNY_THREADS - 1u) / LM_SKINNY_THREADS) * LmSkinnyRowBlocks<ROWS>(multi->part[0].rows),LM_SKINNY_THREADS,0u,stream,*multi);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static uint64_t LmSkinnyFrameWarps(const LmSkinnyMultiArguments *multi, uint32_t lanes, uint32_t rows_per_task, uint32_t neurons)
{
	uint64_t tasks = 0u;
	uint32_t part;
	for ( part = 0u; part < multi->count; part++ )
		tasks += (multi->part[part].output_dimension + neurons - 1u) / neurons;
	return((tasks * ((multi->part[0].rows + rows_per_task - 1u) / rows_per_task) * lanes + LM_WARP_LANES - 1u) / LM_WARP_LANES);
}

template<class Format, uint32_t LANES, uint32_t RG, uint32_t NPG, bool MULTI>
static int32_t LmSkinnyFrameLaunch(LmSkinnyMultiArguments *multi, uint32_t multiprocessors, SparkGpuStream stream)
{
	constexpr uint32_t threads = LmSkinnyFrameConfig<Format,RG,NPG>::kThreads, lane_groups = threads / LANES;
	static thread_local int resident = 0;
	uint32_t part, blocks, cap;
	if ( resident == 0 && (SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(&resident,LmSkinnyFrameKernel<Format,LANES,RG,NPG,MULTI>,(int)threads,0) != SPARK_GPU_SUCCESS || resident <= 0) )
	{
		resident = 0;
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	multi->first_task[0] = 0u;
	for ( part = 0u; part < multi->count; part++ )
		multi->first_task[part + 1u] = multi->first_task[part] + (multi->part[part].output_dimension + NPG - 1u) / NPG;
	blocks = (uint32_t)(((uint64_t)multi->first_task[multi->count] * ((multi->part[0].rows + RG - 1u) / RG) + lane_groups - 1u) / lane_groups);
	cap = multiprocessors * (uint32_t)resident;
	LM_LAUNCH_DEPENDENT((LmSkinnyFrameKernel<Format,LANES,RG,NPG,MULTI>),blocks < cap ? blocks : cap,threads,0u,stream,*multi);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

template<class Format, uint32_t LANES, uint32_t RG, uint32_t NPG>
static int32_t LmSkinnyFrameShape(LmSkinnyMultiArguments *multi, uint32_t multiprocessors, SparkGpuStream stream)
{
	return(multi->count > 1u ? LmSkinnyFrameLaunch<Format,LANES,RG,NPG,true>(multi,multiprocessors,stream) : LmSkinnyFrameLaunch<Format,LANES,RG,NPG,false>(multi,multiprocessors,stream));
}

#if defined(__CUDACC__)
template<uint32_t WARPS_M, uint32_t WARPS_N, uint32_t RPG, uint32_t NPG, uint32_t MIN_BLK, bool MULTI>
static int32_t LmSkinnySharedBf16Launch(LmSkinnyMultiArguments *multi, SparkGpuStream stream)
{
	static std::atomic<uint64_t> granted(0u);
	constexpr uint32_t TILE_M = WARPS_M * RPG, TILE_N = WARPS_N * NPG, THREADS = WARPS_M * WARPS_N * LM_WARP_LANES;
	constexpr uint32_t shared = LmSkinnySharedBf16Bytes<WARPS_M,WARPS_N,RPG,NPG>();
	int device = 0;
	uint32_t part;
	if ( shared > 49152u )
	{
		if ( SparkGpuGetDevice(&device) != SPARK_GPU_SUCCESS || device < 0 || (uint32_t)device >= 64u )
			SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
		if ( (granted.load() & (UINT64_C(1) << (uint32_t)device)) == 0u )
		{
			if ( SparkGpuFuncSetAttribute((const void *)LmSkinnySharedBf16Kernel<WARPS_M,WARPS_N,RPG,NPG,MIN_BLK,MULTI>,SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES,(int)shared) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(LM_LAUNCH_ERR_ATTRIBUTE);
			granted.fetch_or(UINT64_C(1) << (uint32_t)device);
		}
	}
	multi->first_task[0] = 0u;
	for ( part = 0u; part < multi->count; part++ )
		multi->first_task[part + 1u] = multi->first_task[part] + (multi->part[part].output_dimension + TILE_N - 1u) / TILE_N;
	LM_LAUNCH_DEPENDENT((LmSkinnySharedBf16Kernel<WARPS_M,WARPS_N,RPG,NPG,MIN_BLK,MULTI>),
		dim3(multi->first_task[multi->count],(multi->part[0].rows + TILE_M - 1u) / TILE_M),THREADS,shared,stream,*multi);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

template<uint32_t WARPS_M, uint32_t WARPS_N, uint32_t RPG, uint32_t NPG, uint32_t MIN_BLK>
static int32_t LmSkinnySharedBf16Shape(LmSkinnyMultiArguments *multi, SparkGpuStream stream)
{
	return(multi->count > 1u ? LmSkinnySharedBf16Launch<WARPS_M,WARPS_N,RPG,NPG,MIN_BLK,true>(multi,stream) : LmSkinnySharedBf16Launch<WARPS_M,WARPS_N,RPG,NPG,MIN_BLK,false>(multi,stream));
}
#endif

template<class Format, uint32_t LANES>
static int32_t LmSkinnyFrameGroup(LmSkinnyMultiArguments *multi, SparkGpuStream stream)
{
	const uint32_t rows = multi->part[0].rows;
	uint32_t multiprocessors = 0u;
	uint64_t enough;
	if ( LmSkinnyMultiprocessors(&multiprocessors) != LM_LAUNCH_OK )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	enough = (uint64_t)multiprocessors * LM_SKINNY_FRAME_WARPS_PER_SM;
	if ( rows == 1u )
	{
		if ( LmSkinnyFrameWarps(multi,LANES,1u,2u) >= enough )
			return(LmSkinnyFrameShape<Format,LANES,1u,2u>(multi,multiprocessors,stream));
		return(LmSkinnyFrameShape<Format,LANES,1u,1u>(multi,multiprocessors,stream));
	}
#if defined(__CUDACC__)
	if constexpr ( std::is_same<Format,LmBf16Format>::value && LANES == LM_WARP_LANES )
	{
		if ( multi->part[0].input_dimension > 1024u )
		{
			if ( rows > 32u && LmSkinnyFrameWarps(multi,LANES,16u,4u) >= enough )
				return(LmSkinnySharedBf16Shape<4u,2u,16u,4u,1u>(multi,stream));
		}
	}
#endif
	if constexpr ( !LmSkinnyDecodeHalves<Format>::kValue )
	{
		if ( rows > 16u && LmSkinnyFrameWarps(multi,LANES,16u,4u) >= enough && !std::is_same<Format,LmBf16Format>::value )
			return(LmSkinnyFrameShape<Format,LANES,16u,4u>(multi,multiprocessors,stream));
		if constexpr ( std::is_same<Format,LmBf16Format>::value )
		{
			if ( rows > 16u && LmSkinnyFrameWarps(multi,LANES,8u,4u) >= enough )
				return(LmSkinnyFrameShape<Format,LANES,8u,4u>(multi,multiprocessors,stream));
			if ( rows > 8u && LmSkinnyFrameWarps(multi,LANES,8u,2u) >= enough )
				return(LmSkinnyFrameShape<Format,LANES,8u,2u>(multi,multiprocessors,stream));
		}
		else
		{
			if ( rows >= 12u && LmSkinnyFrameWarps(multi,LANES,8u,4u) >= enough )
				return(LmSkinnyFrameShape<Format,LANES,8u,4u>(multi,multiprocessors,stream));
			if ( rows > 8u && LmSkinnyFrameWarps(multi,LANES,8u,2u) >= enough )
				return(LmSkinnyFrameShape<Format,LANES,8u,2u>(multi,multiprocessors,stream));
		}
	}
	else
	{
		if ( rows > 8u && LmSkinnyFrameWarps(multi,LANES,8u,4u) >= enough )
			return(LmSkinnyFrameShape<Format,LANES,8u,4u>(multi,multiprocessors,stream));
		if ( rows >= 4u && LmSkinnyFrameWarps(multi,LANES,4u,4u) >= enough )
			return(LmSkinnyFrameShape<Format,LANES,4u,4u>(multi,multiprocessors,stream));
		if ( rows > 4u && LmSkinnyFrameWarps(multi,LANES,8u,2u) >= enough )
			return(LmSkinnyFrameShape<Format,LANES,8u,2u>(multi,multiprocessors,stream));
	}
	if ( rows > 4u && LmSkinnyFrameWarps(multi,LANES,8u,1u) >= enough )
		return(LmSkinnyFrameShape<Format,LANES,8u,1u>(multi,multiprocessors,stream));
	if ( rows > 2u && LmSkinnyFrameWarps(multi,LANES,4u,2u) >= enough )
		return(LmSkinnyFrameShape<Format,LANES,4u,2u>(multi,multiprocessors,stream));
	if ( rows > 2u && LmSkinnyFrameWarps(multi,LANES,4u,1u) >= enough )
		return(LmSkinnyFrameShape<Format,LANES,4u,1u>(multi,multiprocessors,stream));
	if ( LmSkinnyFrameWarps(multi,LANES,2u,1u) >= enough )
		return(LmSkinnyFrameShape<Format,LANES,2u,1u>(multi,multiprocessors,stream));
	return(LmSkinnyFrameShape<Format,LANES,1u,1u>(multi,multiprocessors,stream));
}

template<class Format>
struct LmSkinnyFrameNeurons
{
	static constexpr uint32_t kMost = 1u;
};

static inline uint32_t LmSkinnyRowsNeurons(uint64_t outputs)
{
	return(outputs >= 4u * LM_SKINNY_ROWS_MIN_TASKS ? 4u : outputs >= 2u * LM_SKINNY_ROWS_MIN_TASKS ? 2u : 1u);
}

template<class Format, uint32_t LANES>
static int32_t LmSkinnyMultiGroup(LmSkinnyMultiArguments *multi, uint32_t rows, uint32_t chunks, SparkGpuStream stream)
{
	uint32_t per_lane = (chunks + LANES - 1u) / LANES, part, has_skip = 0u;
	uint64_t outputs = 0u;
	for ( part = 0u; part < multi->count; part++ )
	{
		outputs += multi->part[part].output_dimension;
		if ( multi->part[part].skip_max_context != 0u )
			has_skip = 1u;
	}
	if ( rows != 1u || (!std::is_same<Format,LmBf16Format>::value && has_skip == 0u && multi->part[0].route_expert == 0 && outputs > 512u) )
		return(LmSkinnyFrameGroup<Format,LANES>(multi,stream));
	if constexpr ( LmSkinnyDecodeOnce<Format>::kValue )
		return(outputs <= 512u ? LmSkinnyMultiShape<Format,LANES,1u,1u>(multi,stream) : LmSkinnyMultiShape<Format,LANES,1u,2u>(multi,stream));
	if ( per_lane >= 8u )
		return(LmSkinnyMultiShape<Format,LANES,1u,1u>(multi,stream));
	if ( per_lane >= 4u )
		return(LmSkinnyMultiShape<Format,LANES,1u,2u>(multi,stream));
	if ( per_lane >= 2u )
		return(LmSkinnyMultiShape<Format,LANES,1u,4u>(multi,stream));
	return(LmSkinnyMultiShape<Format,LANES,1u,2u>(multi,stream));
}

template<class Format, uint32_t LANES>
static int32_t LmSkinnyLaunchGroup(const LmSkinnyArguments *args, uint32_t chunks, SparkGpuStream stream)
{
	uint32_t per_lane = (chunks + LANES - 1u) / LANES;
	if ( args->rows != 1u || (!std::is_same<Format,LmBf16Format>::value && args->route_expert == 0 && args->skip_max_context == 0u && args->output_dimension > 512u) )
	{
		LmSkinnyMultiArguments multi;
		memset(&multi,0,sizeof(multi));
		multi.part[0] = *args;
		multi.count = 1u;
		return(LmSkinnyFrameGroup<Format,LANES>(&multi,stream));
	}
	if constexpr ( LmSkinnyDecodeOnce<Format>::kValue )
		return(args->output_dimension <= 512u ? LmSkinnyLaunchShape<Format,LANES,1u,1u>(args,stream) : LmSkinnyLaunchShape<Format,LANES,1u,2u>(args,stream));
	if ( per_lane >= 8u )
		return(LmSkinnyLaunchShape<Format,LANES,1u,1u>(args,stream));
	if ( per_lane >= 4u )
		return(LmSkinnyLaunchShape<Format,LANES,1u,2u>(args,stream));
	if ( per_lane >= 2u )
		return(LmSkinnyLaunchShape<Format,LANES,1u,4u>(args,stream));
	return(LmSkinnyLaunchShape<Format,LANES,1u,2u>(args,stream));
}

template<class Format>
static int32_t LmSkinnyLaunchRows(const LmSkinnyArguments *args, uint32_t chunks, SparkGpuStream stream)
{
	if ( args->route_expert == 0 && args->group_row_offset == 0 )
		LmWeightPrefetchNote(stream,args->weight,(uint64_t)args->output_dimension * chunks * LM_SKINNY_CHUNK_BYTES);
	if ( chunks >= 32u )
		return(LmSkinnyLaunchGroup<Format,32u>(args,chunks,stream));
	if ( chunks >= 16u )
		return(LmSkinnyLaunchGroup<Format,16u>(args,chunks,stream));
	return(LmSkinnyLaunchGroup<Format,8u>(args,chunks,stream));
}

static uint32_t LmSkinnyAligned(const void *pointer)
{
	return(((uintptr_t)pointer % LM_SKINNY_CHUNK_BYTES) == 0u ? 1u : 0u);
}

static uint32_t LmSkinnyExpertPairsFit(uint32_t pairs, uint32_t top_k)
{
	return(pairs != 0u && top_k != 0u && pairs <= LM_SKINNY_ROWS * top_k ? 1u : 0u);
}

template<class Format>
static int32_t LmSkinnyValidateOperands(const LmSkinnyArguments *args)
{
	constexpr uint32_t elements = LmSkinnyFormat<Format>::kElements;
	if ( args->weight == 0 || args->activation == 0 || (args->output_bf16 == 0) == (args->output_f32 == 0) )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( args->input_dimension == 0u || args->output_dimension == 0u || (args->input_dimension % elements) != 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( LmSkinnyAligned(args->weight) == 0u || LmSkinnyAligned(args->activation) == 0u || (args->weight_group_bytes % LM_SKINNY_CHUNK_BYTES) != 0u || LmSkinnyScaleFits<Format>(args) == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t LmSkinnyValidate(const LmSkinnyArguments *args, uint32_t any_rows = 0u)
{
	uint32_t grouped = args->route_expert != 0 ? 1u : 0u;
	if ( LmSkinnyValidateOperands<Format>(args) != LM_LAUNCH_OK )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( args->rows == 0u || (args->rows > LM_SKINNY_ROWS_WIDE && (any_rows == 0u || grouped != 0u || args->group_row_offset != 0)) || (grouped != 0u && (args->rows != 1u || LmSkinnyExpertPairsFit(args->pairs,args->top_k) == 0u)) )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	if ( args->output_column_offset > args->output_row_stride || args->output_dimension > args->output_row_stride - args->output_column_offset )
		SPARK_FAIL(LM_LAUNCH_ERR_OUTPUT);
	return(LM_LAUNCH_OK);
}

template<class Format>
static int32_t LmSkinnyLaunch(LmSkinnyArguments *args, SparkGpuStream stream, uint32_t any_rows = 0u)
{
	int32_t status;
	if constexpr ( !LmSkinnyFormat<Format>::kSupported )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	else
	{
		if ( args->output_row_stride == 0u )
			args->output_row_stride = args->output_dimension;
		status = LmSkinnyValidate<Format>(args,any_rows);
		if ( status != LM_LAUNCH_OK )
			return(status);
		if constexpr ( LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_NONE || LmSkinnyFormat<Format>::kScaleEncoding == LM_SCALE_ENCODING_F32 )
		{
			if ( args->route_expert == 0 && args->output_dimension <= LM_SKINNY_SPLIT_MAX_OUTPUTS &&
				args->input_dimension / LmSkinnyFormat<Format>::kElements >= LM_SKINNY_SPLIT_WARPS * LM_WARP_LANES )
				return(LmSkinnySplitRows<Format>(args,stream));
		}
		return(LmSkinnyLaunchRows<Format>(args,args->input_dimension / LmSkinnyFormat<Format>::kElements,stream));
	}
}

template<class Format>
static int32_t LmSkinnyDense(const void *weight, const uint16_t *activation, uint16_t *output_bf16, float *output_f32, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset, SparkGpuStream stream, uint32_t any_rows = 0u)
{
	LmSkinnyArguments args;
	memset(&args,0,sizeof(args));
	args.weight = (const uint8_t *)weight;
	args.activation = activation;
	args.output_bf16 = output_bf16;
	args.output_f32 = output_f32;
	args.scale = LmScaleTensorNone();
	args.rows = rows;
	args.input_dimension = input_dimension;
	args.output_dimension = output_dimension;
	args.output_row_stride = output_row_stride;
	args.output_column_offset = output_column_offset;
	return(LmSkinnyLaunch<Format>(&args,stream,any_rows));
}

template<class Format>
static int32_t LmSkinnyDenseSplit(const void *weight, const uint16_t *activation, uint16_t *output_bf16, float *output_f32, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset, SparkGpuStream stream, uint32_t any_rows = 0u)
{
	LmSkinnyArguments args;
	int32_t status;
	if constexpr ( !LmSkinnyFormat<Format>::kSupported || Format::kScaleGroup != 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	else
	{
		memset(&args,0,sizeof(args));
		args.weight = (const uint8_t *)weight;
		args.activation = activation;
		args.output_bf16 = output_bf16;
		args.output_f32 = output_f32;
		args.scale = LmScaleTensorNone();
		args.rows = rows;
		args.input_dimension = input_dimension;
		args.output_dimension = output_dimension;
		args.output_row_stride = output_row_stride != 0u ? output_row_stride : output_dimension;
		args.output_column_offset = output_column_offset;
		status = LmSkinnyValidate<Format>(&args,any_rows);
		if ( status != LM_LAUNCH_OK )
			return(status);
		if ( input_dimension / LmSkinnyFormat<Format>::kElements < LM_SKINNY_SPLIT_WARPS * LM_WARP_LANES )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
		return(LmSkinnySplitRows<Format>(&args,stream));
	}
}

template<class Format>
static int32_t LmSkinnyExperts(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *route_expert, const uint32_t *route_packed_row, uint32_t pairs, uint32_t top_k, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, SparkGpuStream stream)
{
	LmSkinnyArguments args;
	memset(&args,0,sizeof(args));
	args.weight = (const uint8_t *)weight;
	args.activation = activation;
	args.output_bf16 = output_bf16;
	args.route_expert = route_expert;
	args.route_packed_row = route_packed_row;
	args.scale = scale;
	args.weight_group_bytes = (uint64_t)output_dimension * input_dimension * Format::kStoredBits / 8u;
	args.rows = 1u;
	args.pairs = pairs;
	args.top_k = top_k;
	args.activation_packed = activation_packed;
	args.input_dimension = input_dimension;
	args.output_dimension = output_dimension;
	return(LmSkinnyLaunch<Format>(&args,stream));
}

template<class Format, uint32_t NPG>
struct LmSkinnyGroupedShapes
{
	static constexpr uint32_t kFewRows = LM_SKINNY_ROWS_MID, kFewNeurons = NPG;
	static constexpr uint32_t kRows = LM_SKINNY_ROWS, kNeurons = NPG;
	static constexpr uint32_t kWideRows = LM_SKINNY_ROWS_WIDE, kWideNeurons = 2u;
};

template<uint32_t NPG>
struct LmSkinnyGroupedFourRows
{
	static constexpr uint32_t kFewRows = LM_SKINNY_ROWS_MID, kFewNeurons = NPG;
	static constexpr uint32_t kRows = LM_SKINNY_ROWS_MID, kNeurons = NPG;
	static constexpr uint32_t kWideRows = LM_SKINNY_ROWS_MID, kWideNeurons = NPG;
};

template<uint32_t NPG>
struct LmSkinnyGroupedShapes<LmNvfp4,NPG> : LmSkinnyGroupedFourRows<NPG>
{
};

template<uint32_t NPG>
struct LmSkinnyGroupedShapes<LmMxfp4,NPG>
{
	static constexpr uint32_t kFewRows = LM_SKINNY_ROWS_MID, kFewNeurons = NPG;
	static constexpr uint32_t kRows = LM_SKINNY_ROWS_MID, kNeurons = NPG;
	static constexpr uint32_t kWideRows = LM_SKINNY_ROWS, kWideNeurons = NPG;
};

template<class Format, uint32_t NPG>
static int32_t LmSkinnyGroupedExpertsWith(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t pairs, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, SparkGpuStream stream, uint32_t any_rows = 0u)
{
	using Shapes = LmSkinnyGroupedShapes<Format,NPG>;
	LmSkinnyArguments args;
	if constexpr ( !LmSkinnyFormat<Format>::kSupported )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	else
	{
		memset(&args,0,sizeof(args));
		args.weight = (const uint8_t *)weight;
		args.activation = activation;
		args.output_bf16 = output_bf16;
		args.group_row_offset = group_row_offset;
		args.route_source_token = route_source_token;
		args.scale = scale;
		args.weight_group_bytes = (uint64_t)output_dimension * input_dimension * Format::kStoredBits / 8u;
		args.rows = 1u;
		args.pairs = pairs;
		args.activation_packed = activation_packed;
		args.groups = groups;
		args.input_dimension = input_dimension;
		args.output_dimension = output_dimension;
		args.output_row_stride = output_dimension;
		if ( LmSkinnyValidateOperands<Format>(&args) != LM_LAUNCH_OK || group_row_offset == 0 || (activation_packed == 0u && route_source_token == 0) || groups == 0u || pairs == 0u || (any_rows == 0u && pairs > LM_SKINNY_GROUPED_MAX_MEAN_ROWS * groups) )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
		if ( pairs <= 2u * groups )
			return(LmSkinnyGroupedLanes<Format,Shapes::kFewRows,Shapes::kFewNeurons>(&args,input_dimension / LmSkinnyFormat<Format>::kElements,stream));
		if ( pairs > LM_SKINNY_ROWS * groups )
			return(LmSkinnyGroupedLanes<Format,Shapes::kWideRows,Shapes::kWideNeurons>(&args,input_dimension / LmSkinnyFormat<Format>::kElements,stream));
		return(LmSkinnyGroupedLanes<Format,Shapes::kRows,Shapes::kNeurons>(&args,input_dimension / LmSkinnyFormat<Format>::kElements,stream));
	}
}

template<class Format>
static int32_t LmSkinnyGroupedSingleRow(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t pairs, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, SparkGpuStream stream)
{
	LmSkinnyArguments args;
	if constexpr ( !LmSkinnyFormat<Format>::kSupported )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	else
	{
		memset(&args,0,sizeof(args));
		args.weight = (const uint8_t *)weight;
		args.activation = activation;
		args.output_bf16 = output_bf16;
		args.group_row_offset = group_row_offset;
		args.route_source_token = route_source_token;
		args.scale = scale;
		args.weight_group_bytes = (uint64_t)output_dimension * input_dimension * Format::kStoredBits / 8u;
		args.rows = 1u;
		args.pairs = pairs;
		args.activation_packed = activation_packed;
		args.groups = groups;
		args.input_dimension = input_dimension;
		args.output_dimension = output_dimension;
		args.output_row_stride = output_dimension;
		if ( LmSkinnyValidateOperands<Format>(&args) != LM_LAUNCH_OK || group_row_offset == 0 || (activation_packed == 0u && route_source_token == 0) || groups == 0u || pairs == 0u || groups > LM_SKINNY_GROUPED_MAX_GROUPS )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
		return(LmSkinnyGroupedLanes<Format,1u,LM_SKINNY_GROUPED_NEURONS>(&args,input_dimension / LmSkinnyFormat<Format>::kElements,stream));
	}
}

template<class Format>
static int32_t LmSkinnyGroupedExperts(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t pairs, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, SparkGpuStream stream, uint32_t any_rows = 0u)
{
	return(LmSkinnyGroupedExpertsWith<Format,LM_SKINNY_GROUPED_NEURONS>(weight,scale,activation,output_bf16,group_row_offset,route_source_token,groups,pairs,activation_packed,input_dimension,output_dimension,stream,any_rows));
}

typedef struct LmSkinnyDenseTarget
{
	const void *weight;
	uint16_t *output_bf16;
	float *output_f32;
	uint32_t output_dimension,output_row_stride,output_column_offset;
	LmScaleTensor scale;
}
LmSkinnyDenseTarget;

template<class Format>
static int32_t LmSkinnyDenseMulti(const LmSkinnyDenseTarget *targets, uint32_t count, const uint16_t *activation, uint32_t rows, uint32_t input_dimension, SparkGpuStream stream, uint32_t any_rows = 0u, uint32_t allow_split = 0u, uint32_t skip_part = LM_SKINNY_MULTI_MAX, const uint32_t *skip_sequence_of_row = 0, const uint32_t *skip_context_length = 0, const uint32_t *skip_row_position = 0, uint32_t skip_max_context = 0u)
{
	LmSkinnyMultiArguments multi;
	uint32_t part, chunks;
	uint64_t total_outputs = 0u;
	if constexpr ( !LmSkinnyFormat<Format>::kSupported )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
	else
	{
		if ( targets == 0 || count == 0u || count > LM_SKINNY_MULTI_MAX )
			SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
		memset(&multi,0,sizeof(multi));
		multi.count = count;
		for ( part = 0u; part < count; part++ )
		{
			LmSkinnyArguments *args = &multi.part[part];
			args->weight = (const uint8_t *)targets[part].weight;
			args->activation = activation;
			args->output_bf16 = targets[part].output_bf16;
			args->output_f32 = targets[part].output_f32;
			args->scale = targets[part].scale;
			args->rows = rows;
			args->input_dimension = input_dimension;
			args->output_dimension = targets[part].output_dimension;
			args->output_row_stride = targets[part].output_row_stride != 0u ? targets[part].output_row_stride : targets[part].output_dimension;
			args->output_column_offset = targets[part].output_column_offset;
			if ( part == skip_part )
			{
				args->skip_sequence_of_row = skip_sequence_of_row;
				args->skip_context_length = skip_context_length;
				args->skip_row_position = skip_row_position;
				args->skip_max_context = skip_max_context;
			}
			if ( LmSkinnyValidate<Format>(args,any_rows) != LM_LAUNCH_OK )
				SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
			total_outputs += args->output_dimension;
		}
		chunks = input_dimension / LmSkinnyFormat<Format>::kElements;
		for ( part = 0u; part < count; part++ )
			LmWeightPrefetchNote(stream,multi.part[part].weight,(uint64_t)multi.part[part].output_dimension * chunks * LM_SKINNY_CHUNK_BYTES);
		if constexpr ( !LmSkinnySplitScale<Format>::kSplit )
		{
			if ( allow_split != 0u && total_outputs <= LM_SKINNY_SPLIT_MAX_OUTPUTS && chunks >= LM_SKINNY_SPLIT_WARPS * LM_WARP_LANES )
				return(LmSkinnySplitMultiRows<Format>(&multi,stream));
		}
		if ( chunks >= 32u )
			return(LmSkinnyMultiGroup<Format,32u>(&multi,rows,chunks,stream));
		if ( chunks >= 16u )
			return(LmSkinnyMultiGroup<Format,16u>(&multi,rows,chunks,stream));
		return(LmSkinnyMultiGroup<Format,8u>(&multi,rows,chunks,stream));
	}
}

#define LM_SKINNY_CELL_NEURONS 16u
#define LM_SKINNY_CELL_ROWS 17u
#define LM_SKINNY_CELL_TILE_K 32u
#define LM_SKINNY_CELL_BATCH 16u
#define LM_SKINNY_CELL_GROUP_CACHE 1024u
#define LM_SKINNY_CELL_SHARED_TILES 8u
#define LM_SKINNY_CELL_SHORT_BATCH 4u
#define LM_SKINNY_CELL_SPREAD_THREADS 32u
#define LM_SKINNY_CELL_SPLIT_WARPS 4u
#define LM_SKINNY_CELL_SPLIT_BATCH 8u

static uint32_t LmSkinnyCellShapeFits(uint32_t input_dimension, uint32_t output_dimension, uint32_t tile_k)
{
	return(tile_k == LM_SKINNY_CELL_TILE_K && input_dimension != 0u && (input_dimension % LM_SKINNY_CELL_TILE_K) == 0u &&
		output_dimension != 0u && (output_dimension % LM_SKINNY_CELL_NEURONS) == 0u ? 1u : 0u);
}

typedef struct LmSkinnyCellArguments
{
	const uint8_t *weight;
	const uint16_t *activation;
	uint16_t *output_bf16;
	const uint32_t *group_row_offset;
	const uint32_t *route_source_token;
	uint64_t group_bytes;
	uint32_t groups,packed_rows,activation_packed;
	uint32_t input_dimension,output_dimension,k_tiles,cells;
}
LmSkinnyCellArguments;

template<bool GLOBAL = false>
static __device__ __forceinline__ uint32_t LmSkinnyCellGroup(const uint32_t *offset, uint32_t groups, uint32_t row)
{
	uint32_t low = 0u, high = groups, middle;
	while ( high - low > 1u )
	{
		middle = (low + high) / 2u;
		const uint32_t entry = GLOBAL ? __ldg(offset + middle) : offset[middle];
		if ( entry <= row )
			low = middle;
		else
			high = middle;
	}
	return(low);
}

template<uint32_t BATCH>
static __device__ __forceinline__ void LmSkinnyCellTile(uint4 weight, uint8_t scale_code, const uint16_t *const *activation, uint32_t element, float *accumulator)
{
	const uint32_t words[4] = {weight.x,weight.y,weight.z,weight.w};
	float chunk[BATCH];
	uint32_t w, j, r;
	#pragma unroll
	for ( r = 0u; r < BATCH; r++ )
		chunk[r] = 0.0f;
	#pragma unroll
	for ( w = 0u; w < 4u; w++ )
	{
		float2 pair[4];
		#pragma unroll
		for ( j = 0u; j < 4u; j++ )
			pair[j] = LmSkinnyE2m1Pair(words[w],j);
		#pragma unroll
		for ( r = 0u; r < BATCH; r++ )
		{
			const uint4 packed = __ldg((const uint4 *)(activation[r] + element + w * 8u));
			float a0 = pair[0].x * __uint_as_float(packed.x << 16u);
			float a1 = pair[0].y * __uint_as_float(packed.x & 0xffff0000u);
			float a2 = pair[1].x * __uint_as_float(packed.y << 16u);
			float a3 = pair[1].y * __uint_as_float(packed.y & 0xffff0000u);
			a0 = fmaf(pair[2].x,__uint_as_float(packed.z << 16u),a0);
			a1 = fmaf(pair[2].y,__uint_as_float(packed.z & 0xffff0000u),a1);
			a2 = fmaf(pair[3].x,__uint_as_float(packed.w << 16u),a2);
			a3 = fmaf(pair[3].y,__uint_as_float(packed.w & 0xffff0000u),a3);
			chunk[r] += (a0 + a1) + (a2 + a3);
		}
	}
	const float scale = LmUe8m0ToFloat(scale_code);
	#pragma unroll
	for ( r = 0u; r < BATCH; r++ )
		accumulator[r] = fmaf(scale,chunk[r],accumulator[r]);
}

template<uint32_t ROWS>
static __device__ __forceinline__ void LmSkinnyCellRows(const LmSkinnyCellArguments &args, const uint8_t *base, uint64_t tile_stride, uint32_t cell, uint32_t valid_cell, uint32_t sub, uint32_t first)
{
	float accumulator[ROWS];
	const uint16_t *activation[ROWS];
	uint32_t r, packed;
	#pragma unroll
	for ( r = 0u; r < ROWS; r++ )
	{
		accumulator[r] = 0.0f;
		packed = first + r;
		activation[r] = args.activation + (uint64_t)(args.activation_packed != 0u ? packed : __ldg(args.route_source_token + packed)) * args.input_dimension;
	}
	uint4 weight_cur;
	uint8_t scale_cur = 0u;
	if ( args.k_tiles > 0u )
	{
		weight_cur = __ldcs((const uint4 *)(base + sub * LM_SKINNY_CHUNK_BYTES));
		scale_cur = __ldcs(base + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
	}
	for ( uint32_t t = 0u; t < args.k_tiles; t++ )
	{
		uint4 weight_next;
		uint8_t scale_next = 0u;
		if ( t + 1u < args.k_tiles )
		{
			const uint8_t *next_tile = base + (uint64_t)(t + 1u) * tile_stride;
			weight_next = __ldcs((const uint4 *)(next_tile + sub * LM_SKINNY_CHUNK_BYTES));
			scale_next = __ldcs(next_tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
		}
		LmSkinnyCellTile<ROWS>(weight_cur,scale_cur,activation,t * LM_SKINNY_CELL_TILE_K,accumulator);
		weight_cur = weight_next;
		scale_cur = scale_next;
	}
	if ( valid_cell == 0u )
		return;
	#pragma unroll
	for ( r = 0u; r < ROWS; r++ )
		args.output_bf16[(uint64_t)(first + r) * args.output_dimension + cell * LM_SKINNY_CELL_NEURONS + sub] = LmFloatToBf16(accumulator[r]);
}

template<uint32_t ROWS, uint32_t WARPS, bool PAIR_CELLS>
static __device__ __forceinline__ void LmSkinnyCellSplitRows(const LmSkinnyCellArguments &args, const uint8_t *base, uint64_t tile_stride, uint32_t cell, uint32_t valid_cell, uint32_t warp, uint32_t lane, uint32_t sub, uint32_t half, uint32_t first, uint32_t more, float *partial)
{
	constexpr uint32_t stride_t = PAIR_CELLS ? 1u : 2u * WARPS;
	const uint32_t t0 = PAIR_CELLS ? args.k_tiles * warp / WARPS : 2u * warp + half;
	const uint32_t t1 = PAIR_CELLS ? args.k_tiles * (warp + 1u) / WARPS : args.k_tiles;
	float accumulator[ROWS];
	const uint16_t *activation[ROWS];
	uint32_t r, packed;
	#pragma unroll
	for ( r = 0u; r < ROWS; r++ )
	{
		accumulator[r] = 0.0f;
		packed = first + r;
		activation[r] = args.activation + (uint64_t)(args.activation_packed != 0u ? packed : __ldg(args.route_source_token + packed)) * args.input_dimension;
	}
	if constexpr ( ROWS == 1u && PAIR_CELLS )
	{
		uint4 w0, w1;
		uint8_t s0 = 0u, s1 = 0u;
		if ( t0 < t1 )
		{
			const uint8_t *tile = base + (uint64_t)t0 * tile_stride;
			w0 = __ldcs((const uint4 *)(tile + sub * LM_SKINNY_CHUNK_BYTES));
			s0 = __ldcs(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
		}
		if ( t0 + stride_t < t1 )
		{
			const uint8_t *tile = base + (uint64_t)(t0 + stride_t) * tile_stride;
			w1 = __ldcs((const uint4 *)(tile + sub * LM_SKINNY_CHUNK_BYTES));
			s1 = __ldcs(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
		}
		for ( uint32_t t = t0; t < t1; t += stride_t )
		{
			const uint32_t next2 = t + 2u * stride_t;
			uint4 w2;
			uint8_t s2 = 0u;
			if ( next2 < t1 )
			{
				const uint8_t *tile = base + (uint64_t)next2 * tile_stride;
				w2 = __ldcs((const uint4 *)(tile + sub * LM_SKINNY_CHUNK_BYTES));
				s2 = __ldcs(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
			}
			LmSkinnyCellTile<1u>(w0,s0,activation,t * LM_SKINNY_CELL_TILE_K,accumulator);
			w0 = w1;
			s0 = s1;
			w1 = w2;
			s1 = s2;
		}
	}
	else
	{
		uint4 weight_cur;
		uint8_t scale_cur = 0u;
		if ( t0 < t1 )
		{
			const uint8_t *tile = base + (uint64_t)t0 * tile_stride;
			weight_cur = __ldcs((const uint4 *)(tile + sub * LM_SKINNY_CHUNK_BYTES));
			scale_cur = PAIR_CELLS ? __ldcs(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub) : __ldg(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
		}
		for ( uint32_t t = t0; t < t1; t += stride_t )
		{
			const uint32_t next = t + stride_t;
			uint4 weight_next;
			uint8_t scale_next = 0u;
			if ( next < t1 )
			{
				const uint8_t *tile = base + (uint64_t)next * tile_stride;
				weight_next = __ldcs((const uint4 *)(tile + sub * LM_SKINNY_CHUNK_BYTES));
				scale_next = PAIR_CELLS ? __ldcs(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub) : __ldg(tile + LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES + sub);
			}
			LmSkinnyCellTile<ROWS>(weight_cur,scale_cur,activation,t * LM_SKINNY_CELL_TILE_K,accumulator);
			weight_cur = weight_next;
			scale_cur = scale_next;
		}
	}
	if constexpr ( PAIR_CELLS )
	{
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
			partial[(warp * ROWS + r) * LM_WARP_LANES + lane] = accumulator[r];
		__syncthreads();
		if ( warp == 0u && valid_cell != 0u )
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
			{
				float total = partial[r * LM_WARP_LANES + lane];
				#pragma unroll
				for ( uint32_t w = 1u; w < WARPS; w++ )
					total += partial[(w * ROWS + r) * LM_WARP_LANES + lane];
				args.output_bf16[(uint64_t)(first + r) * args.output_dimension + cell * LM_SKINNY_CELL_NEURONS + sub] = LmFloatToBf16(total);
			}
	}
	else
	{
		#pragma unroll
		for ( r = 0u; r < ROWS; r++ )
			accumulator[r] += __shfl_xor_sync(0xffffffffu,accumulator[r],LM_SKINNY_CELL_NEURONS);
		if ( half == 0u )
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
				partial[(warp * ROWS + r) * LM_SKINNY_CELL_NEURONS + sub] = accumulator[r];
		__syncthreads();
		if ( warp == 0u && half == 0u )
			#pragma unroll
			for ( r = 0u; r < ROWS; r++ )
			{
				float total = partial[r * LM_SKINNY_CELL_NEURONS + sub];
				#pragma unroll
				for ( uint32_t w = 1u; w < WARPS; w++ )
					total += partial[(w * ROWS + r) * LM_SKINNY_CELL_NEURONS + sub];
				args.output_bf16[(uint64_t)(first + r) * args.output_dimension + cell * LM_SKINNY_CELL_NEURONS + sub] = LmFloatToBf16(total);
			}
	}
	if ( more != 0u )
		__syncthreads();
}

template<uint32_t BATCH, uint32_t WARPS, bool SPLIT, bool PAIR_CELLS>
__global__ __launch_bounds__(WARPS * LM_WARP_LANES, 1) void LmSkinnyCellSplitKernel(const __grid_constant__ LmSkinnyCellArguments args,
	const uint8_t *weight_secondary = 0, uint32_t primary_groups = 0u)
{
	__shared__ float partial[WARPS * BATCH * (PAIR_CELLS ? LM_WARP_LANES : LM_SKINNY_CELL_NEURONS)];
	const uint32_t warp = threadIdx.x / LM_WARP_LANES, lane = threadIdx.x % LM_WARP_LANES;
	const uint32_t sub = lane % LM_SKINNY_CELL_NEURONS, half = lane / LM_SKINNY_CELL_NEURONS;
	const uint32_t width = PAIR_CELLS ? (args.cells + 1u) / 2u : args.cells;
	const uint32_t row = blockIdx.x / width, slot = blockIdx.x % width;
	const uint32_t valid_cell = PAIR_CELLS ? (slot * 2u + half < args.cells ? 1u : 0u) : 1u;
	const uint32_t cell = PAIR_CELLS ? (valid_cell != 0u ? slot * 2u + half : args.cells - 1u) : slot;
	const uint64_t tile_stride = (uint64_t)args.cells * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	uint32_t group, start, end, first;
	const uint8_t *base;
	LmDependentWait();
	LmDependentRelease();
	if ( args.packed_rows > args.groups )
	{
		group = row;
		start = __ldg(args.group_row_offset + group);
		end = __ldg(args.group_row_offset + group + 1u);
		if ( start == end )
			return;
	}
	else
	{
		if ( row >= __ldg(args.group_row_offset + args.groups) )
			return;
		group = LmSkinnyCellGroup<true>(args.group_row_offset,args.groups,row);
		start = __ldg(args.group_row_offset + group);
		if ( row != start )
			return;
		end = __ldg(args.group_row_offset + group + 1u);
	}
	if constexpr ( SPLIT )
		base = (group >= primary_groups ? weight_secondary + (uint64_t)(group - primary_groups) * args.group_bytes :
			args.weight + (uint64_t)group * args.group_bytes) + (uint64_t)cell * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	else
		base = args.weight + (uint64_t)group * args.group_bytes + (uint64_t)cell * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	for ( first = start; first < end; )
	{
		const uint32_t count = end - first;
		if constexpr ( BATCH >= 8u )
		{
			if ( count >= 8u )
			{
				LmSkinnyCellSplitRows<8u,WARPS,PAIR_CELLS>(args,base,tile_stride,cell,valid_cell,warp,lane,sub,half,first,count > 8u ? 1u : 0u,partial);
				first += 8u;
				continue;
			}
		}
		if constexpr ( BATCH >= 4u )
		{
			if ( count >= 4u )
			{
				LmSkinnyCellSplitRows<4u,WARPS,PAIR_CELLS>(args,base,tile_stride,cell,valid_cell,warp,lane,sub,half,first,count > 4u ? 1u : 0u,partial);
				first += 4u;
				continue;
			}
			if ( count == 3u )
			{
				LmSkinnyCellSplitRows<3u,WARPS,PAIR_CELLS>(args,base,tile_stride,cell,valid_cell,warp,lane,sub,half,first,0u,partial);
				first += 3u;
				continue;
			}
			if ( count == 2u )
			{
				LmSkinnyCellSplitRows<2u,WARPS,PAIR_CELLS>(args,base,tile_stride,cell,valid_cell,warp,lane,sub,half,first,0u,partial);
				first += 2u;
				continue;
			}
		}
		LmSkinnyCellSplitRows<1u,WARPS,PAIR_CELLS>(args,base,tile_stride,cell,valid_cell,warp,lane,sub,half,first,count > 1u ? 1u : 0u,partial);
		first += 1u;
	}
}

template<uint32_t BATCH, bool SHARED_OFFSETS, uint32_t THREADS = LM_SKINNY_THREADS, bool SPLIT = false>
__global__ __launch_bounds__(THREADS,1) void LmSkinnyCellKernel(const __grid_constant__ LmSkinnyCellArguments args,
	const uint8_t *weight_secondary = 0, uint32_t primary_groups = 0u)
{
	__shared__ uint32_t cached[SHARED_OFFSETS ? LM_SKINNY_CELL_GROUP_CACHE : 1u];
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, sub = lane % LM_SKINNY_CELL_NEURONS, half = lane / LM_SKINNY_CELL_NEURONS;
	const uint32_t task = (blockIdx.x * THREADS + threadIdx.x) / LM_WARP_LANES;
	const uint32_t pairs = (args.cells + 1u) / 2u;
	const uint64_t tile_stride = (uint64_t)args.cells * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	const uint32_t *offset = SHARED_OFFSETS ? cached : args.group_row_offset;
	const uint32_t by_group = args.packed_rows > args.groups ? 1u : 0u;
	const uint32_t limit = by_group != 0u ? args.groups : args.packed_rows;
	uint32_t row, pair, cell, valid_cell, group, start, end, first, index;
	const uint8_t *base;
	LmDependentWait();
	if ( by_group == 0u && (blockIdx.x * (THREADS / LM_WARP_LANES)) / pairs >= __ldg(args.group_row_offset + args.groups) )
	{
		LmDependentRelease();
		return;
	}
	if constexpr ( SHARED_OFFSETS )
	{
		for ( index = threadIdx.x; index <= args.groups; index += THREADS )
			cached[index] = args.group_row_offset[index];
		__syncthreads();
	}
	LmDependentRelease();
	if ( task >= limit * pairs )
		return;
	row = task / pairs;
	pair = task % pairs;
	valid_cell = pair * 2u + half < args.cells ? 1u : 0u;
	cell = valid_cell != 0u ? pair * 2u + half : args.cells - 1u;
	if ( by_group != 0u )
	{
		group = row;
		start = offset[group];
		end = offset[group + 1u];
		if ( start == end )
			return;
	}
	else
	{
		if ( row >= offset[args.groups] )
			return;
		group = LmSkinnyCellGroup<!SHARED_OFFSETS>(offset,args.groups,row);
		start = offset[group];
		if ( row != start )
			return;
		end = offset[group + 1u];
	}
	if constexpr ( SPLIT )
		base = (group >= primary_groups ? weight_secondary + (uint64_t)(group - primary_groups) * args.group_bytes :
			args.weight + (uint64_t)group * args.group_bytes) + (uint64_t)cell * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	else
		base = args.weight + (uint64_t)group * args.group_bytes + (uint64_t)cell * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	for ( first = start; first < end; )
	{
		const uint32_t count = end - first;
		if constexpr ( BATCH >= 16u )
		{
			if ( count >= 16u )
			{
				LmSkinnyCellRows<16u>(args,base,tile_stride,cell,valid_cell,sub,first);
				first += 16u;
				continue;
			}
			if ( count >= 8u )
			{
				LmSkinnyCellRows<8u>(args,base,tile_stride,cell,valid_cell,sub,first);
				first += 8u;
				continue;
			}
		}
		if constexpr ( BATCH >= 4u )
		{
			if ( count >= 4u )
			{
				LmSkinnyCellRows<4u>(args,base,tile_stride,cell,valid_cell,sub,first);
				first += 4u;
				continue;
			}
			if ( count == 3u )
			{
				LmSkinnyCellRows<3u>(args,base,tile_stride,cell,valid_cell,sub,first);
				first += 3u;
				continue;
			}
			if ( count == 2u )
			{
				LmSkinnyCellRows<2u>(args,base,tile_stride,cell,valid_cell,sub,first);
				first += 2u;
				continue;
			}
		}
		LmSkinnyCellRows<1u>(args,base,tile_stride,cell,valid_cell,sub,first);
		first += 1u;
	}
}

template<uint32_t WARPS, bool SPLIT, bool PAIR_CELLS>
static int32_t LmSkinnyCellSplitDispatch(const LmSkinnyCellArguments &args, const uint8_t *weight_secondary, uint32_t primary_groups, uint32_t group_rows_max, uint64_t tasks, SparkGpuStream stream)
{
	const uint32_t split_threads = WARPS * LM_WARP_LANES;
	if ( group_rows_max == 1u )
		LM_LAUNCH_DEPENDENT((LmSkinnyCellSplitKernel<1u,WARPS,SPLIT,PAIR_CELLS>),(uint32_t)tasks,split_threads,0u,stream,args,weight_secondary,primary_groups);
	else if ( group_rows_max <= LM_SKINNY_CELL_SHORT_BATCH || (PAIR_CELLS && args.packed_rows <= LM_SKINNY_CELL_SHORT_BATCH * args.groups) )
		LM_LAUNCH_DEPENDENT((LmSkinnyCellSplitKernel<LM_SKINNY_CELL_SHORT_BATCH,WARPS,SPLIT,PAIR_CELLS>),(uint32_t)tasks,split_threads,0u,stream,args,weight_secondary,primary_groups);
	else
		LM_LAUNCH_DEPENDENT((LmSkinnyCellSplitKernel<LM_SKINNY_CELL_SPLIT_BATCH,WARPS,SPLIT,PAIR_CELLS>),(uint32_t)tasks,split_threads,0u,stream,args,weight_secondary,primary_groups);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static int32_t LmSkinnyCellExpertsSplit(const void *weight, const void *weight_secondary, uint32_t primary_groups, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t group_rows_max, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, uint32_t tile_k, SparkGpuStream stream)
{
	LmSkinnyCellArguments args;
	uint64_t tasks;
	if ( LmSkinnyCellShapeFits(input_dimension,output_dimension,tile_k) == 0u || weight == 0 || activation == 0 || output_bf16 == 0 ||
		group_row_offset == 0 || (activation_packed == 0u && route_source_token == 0) || groups == 0u || packed_rows == 0u ||
		(weight_secondary != 0 && (primary_groups > groups || LmSkinnyAligned(weight_secondary) == 0u)) ||
		LmSkinnyAligned(weight) == 0u || LmSkinnyAligned(activation) == 0u )
		SPARK_FAIL(LM_LAUNCH_ERR_SHAPE);
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
	const uint32_t pairs = (args.cells + 1u) / 2u;
	const uint32_t active_rows = packed_rows > groups ? groups : packed_rows;
	if ( args.k_tiles >= 16u || (args.k_tiles >= 2u * LM_SKINNY_CELL_SPLIT_WARPS && args.cells <= 64u) )
	{
		if ( args.cells >= 64u )
		{
			tasks = (uint64_t)active_rows * pairs;
			if ( weight_secondary != 0 )
				return(LmSkinnyCellSplitDispatch<LM_SKINNY_CELL_SPLIT_WARPS,true,true>(args,(const uint8_t *)weight_secondary,primary_groups,group_rows_max,tasks,stream));
			return(LmSkinnyCellSplitDispatch<LM_SKINNY_CELL_SPLIT_WARPS,false,true>(args,(const uint8_t *)0,0u,group_rows_max,tasks,stream));
		}
		tasks = (uint64_t)active_rows * args.cells;
		if ( weight_secondary != 0 )
			return(LmSkinnyCellSplitDispatch<LM_SKINNY_CELL_SPLIT_WARPS,true,false>(args,(const uint8_t *)weight_secondary,primary_groups,group_rows_max,tasks,stream));
		return(LmSkinnyCellSplitDispatch<LM_SKINNY_CELL_SPLIT_WARPS,false,false>(args,(const uint8_t *)0,0u,group_rows_max,tasks,stream));
	}
	tasks = (uint64_t)active_rows * pairs;
	const uint32_t blocks = (uint32_t)((tasks * LM_WARP_LANES + LM_SKINNY_THREADS - 1u) / LM_SKINNY_THREADS);
	const uint32_t spread = (uint32_t)((tasks * LM_WARP_LANES + LM_SKINNY_CELL_SPREAD_THREADS - 1u) / LM_SKINNY_CELL_SPREAD_THREADS);
	const bool shared = weight_secondary == 0 && args.k_tiles <= LM_SKINNY_CELL_SHARED_TILES && groups < LM_SKINNY_CELL_GROUP_CACHE;
	if ( weight_secondary != 0 )
	{
		if ( group_rows_max == 1u )
			LM_LAUNCH_DEPENDENT((LmSkinnyCellKernel<1u,false,LM_SKINNY_CELL_SPREAD_THREADS,true>),spread,LM_SKINNY_CELL_SPREAD_THREADS,0u,stream,args,
				(const uint8_t *)weight_secondary,primary_groups);
		else
			LM_LAUNCH_DEPENDENT((LmSkinnyCellKernel<LM_SKINNY_CELL_BATCH,false,LM_SKINNY_THREADS,true>),blocks,LM_SKINNY_THREADS,0u,stream,args,
				(const uint8_t *)weight_secondary,primary_groups);
	}
	else if ( group_rows_max == 1u && shared )
		LM_LAUNCH_DEPENDENT((LmSkinnyCellKernel<1u,true>),blocks,LM_SKINNY_THREADS,0u,stream,args,(const uint8_t *)0,0u);
	else if ( group_rows_max == 1u )
		LM_LAUNCH_DEPENDENT((LmSkinnyCellKernel<1u,false,LM_SKINNY_CELL_SPREAD_THREADS>),spread,LM_SKINNY_CELL_SPREAD_THREADS,0u,stream,args,(const uint8_t *)0,0u);
	else if ( shared )
		LM_LAUNCH_DEPENDENT((LmSkinnyCellKernel<LM_SKINNY_CELL_SHORT_BATCH,true>),blocks,LM_SKINNY_THREADS,0u,stream,args,(const uint8_t *)0,0u);
	else
		LM_LAUNCH_DEPENDENT((LmSkinnyCellKernel<LM_SKINNY_CELL_BATCH,false>),blocks,LM_SKINNY_THREADS,0u,stream,args,(const uint8_t *)0,0u);
	if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	return(LM_LAUNCH_OK);
}

static int32_t LmSkinnyCellExperts(const void *weight, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t group_rows_max, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, uint32_t tile_k, SparkGpuStream stream)
{
	return(LmSkinnyCellExpertsSplit(weight,(const void *)0,groups,activation,output_bf16,group_row_offset,route_source_token,groups,packed_rows,group_rows_max,activation_packed,input_dimension,output_dimension,tile_k,stream));
}
