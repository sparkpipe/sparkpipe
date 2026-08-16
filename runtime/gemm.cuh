#pragma once

#include "inference/kernels/gemm.cuh"
#include "runtime/gemm_descriptor_cache.h"
#include "runtime/launch.h"
#include "runtime/tensor_map.h"
#include <cuda_runtime.h>
#include <mutex>
#include <stdint.h>
#include <string.h>

#define LM_GEMM_MAX_TRACKED_CUDA_DEVICES 64u

template<class Format>
static int32_t LmGemmValidateScaleTensor(
    const LmScaleTensor *scale,
    uint32_t required_group_count,
    uint32_t required_row_count,
    uint32_t required_input_dimension)
{
    if (LmScaleTensorIsValid(scale) == 0u)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    if (Format::kScaleGroup == 0u)
    {
        return scale->encoding == LM_SCALE_ENCODING_NONE
            ? LM_LAUNCH_OK
            : LM_LAUNCH_ERR_SHAPE;
    }
    if (scale->encoding == LM_SCALE_ENCODING_NONE ||
        scale->group_count != required_group_count ||
        scale->row_count != required_row_count ||
        scale->input_dimension != required_input_dimension ||
        scale->k_group_size != Format::kScaleGroup)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    return LM_LAUNCH_OK;
}

template<
    class FormatA,
    class FormatB,
    uint32_t TILE_M,
    uint32_t TILE_N,
    uint32_t TILE_K,
	uint32_t STAGES,
	uint32_t WARPS,
	bool INDIRECT_A = false,
	uint32_t ACTIVATION_CODEC = SPARK_ACTIVATION_CODEC_NONE,
	bool INTERLEAVED_B = false>
static cudaError_t LmGemmOptIn(uint32_t shared_bytes)
{
    static std::mutex grant_mutex;
    static uint64_t granted_device_mask = 0u;
    cudaError_t status;
    uint64_t device_bit;
    int device_index;

    status = cudaGetDevice(&device_index);
    if (status != cudaSuccess)
        return status;
    if (device_index < 0 ||
        (uint32_t)device_index >= LM_GEMM_MAX_TRACKED_CUDA_DEVICES)
        return cudaErrorInvalidDevice;
    device_bit = UINT64_C(1) << (uint32_t)device_index;
    {
        std::lock_guard<std::mutex> lock(grant_mutex);
        if ((granted_device_mask & device_bit) != 0u)
            return cudaSuccess;
        status = cudaFuncSetAttribute(
            (const void *)LmGemmKernel<
                FormatA,
                FormatB,
                TILE_M,
                TILE_N,
                TILE_K,
				STAGES,
				WARPS,
					INDIRECT_A,
					ACTIVATION_CODEC,
					INTERLEAVED_B>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shared_bytes);
        if (status == cudaSuccess)
            granted_device_mask |= device_bit;
    }
    return status;
}

template<
    class FormatA,
    class FormatB,
    uint32_t TILE_M,
    uint32_t TILE_N,
    uint32_t TILE_K,
	uint32_t STAGES,
	uint32_t WARPS,
	bool INDIRECT_A = false,
	uint32_t ACTIVATION_CODEC = SPARK_ACTIVATION_CODEC_NONE,
	bool INTERLEAVED_B = false>
static cudaError_t LmGemmLaunchTile(
    const LmGemmArguments &args,
    const CUtensorMap &activation_map,
    const CUtensorMap &weight_map,
    const LmTileGeometry &activation_geometry,
    const LmTileGeometry &weight_geometry,
    bool grouped,
    const LmLaunchPlan &plan,
    cudaStream_t stream)
{
    constexpr uint32_t shared = LmGemmSharedBytes<
        FormatA,
        FormatB,
        TILE_M,
        TILE_N,
        TILE_K,
        STAGES,
        INTERLEAVED_B>();
    cudaError_t status;

    static_assert(
        shared <= LM_SMEM_SM_TOTAL,
        "tile exceeds the shared memory an SM has");
    status = LmGemmOptIn<
        FormatA,
        FormatB,
        TILE_M,
        TILE_N,
        TILE_K,
			STAGES,
			WARPS,
			INDIRECT_A,
			ACTIVATION_CODEC,
			INTERLEAVED_B>(shared);
    if (status != cudaSuccess)
        return status;
    LmGemmKernel<
        FormatA,
        FormatB,
        TILE_M,
        TILE_N,
        TILE_K,
		STAGES,
		WARPS,
			INDIRECT_A,
			ACTIVATION_CODEC,
			INTERLEAVED_B>
        <<<plan.grid_blocks, plan.block_threads, shared, stream>>>(
            args,
            activation_map,
            weight_map,
            activation_geometry,
            weight_geometry,
            grouped);
    return cudaPeekAtLastError();
}

// GEMM-007: both encodes go through the request-keyed descriptor cache
// (runtime/gemm_descriptor_cache.h). The request fully determines the
// descriptor bytes, so a cache hit is what cuTensorMapEncodeTiled would have
// returned and steady-state decode performs zero driver encodes per token.
static int32_t LmGemmEncodeActivationMap(
    CUtensorMap *activation,
    const void *activation_bytes,
    uint32_t activation_rows,
    uint32_t input_dimension,
    uint32_t tile_m,
    uint32_t tile_k,
    uint32_t activation_bits)
{
    LmTensorMapRequest request;

    memset(&request, 0, sizeof(request));
    request.global_address = activation_bytes;
    request.rows = activation_rows;
    request.columns = input_dimension;
    request.groups = 1u;
    request.box_rows = tile_m;
    request.box_columns = tile_k;
    request.element_bits = activation_bits;
    return LmGemmTensorMapCached(activation, &request);
}

static int32_t LmGemmEncodeWeightMap(
    CUtensorMap *weight,
    const void *weight_bytes,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t group_count,
    uint32_t tile_n,
    uint32_t tile_k,
    uint32_t weight_bits)
{
    LmTensorMapRequest request;

    memset(&request, 0, sizeof(request));
    request.global_address = weight_bytes;
    request.rows = output_dimension;
    request.columns = input_dimension;
    request.groups = group_count;
    request.box_rows = tile_n;
    request.box_columns = tile_k;
    request.element_bits = weight_bits;
    return LmGemmTensorMapCached(weight, &request);
}

// INTERLEAVED WEIGHT MAP - pack V2 mxfp4_ws_interleaved_v1 (K3). The B
// operand is the 17-row cell grid, described as a rank-3 UINT8 byte tensor:
// 64-byte rows, rows_per_expert = k_tiles * cells * 17 rows per expert, one
// group per expert, box [64 bytes, 17 * (TILE_N/16) rows, 1], 64B swizzle.
// The packer's interleave grid (tools/k3_pack.py) asserts this closes with
// zero padding, and the kernel's produce coordinate
// (k_tile * cells + neuron_base/16) * 17 is this layout's row index. The
// 128-element pack k-tile is the grid's k unit, so the launcher passes
// TILE_K == 128 for these operands.
static int32_t LmGemmEncodeWeightMapInterleaved(
    CUtensorMap *weight,
    const void *weight_bytes,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t group_count,
    uint32_t tile_n,
    uint32_t tile_k)
{
    LmTensorMapRequest request;
    const uint32_t k_tiles = input_dimension / tile_k;
    const uint32_t cells = output_dimension / 16u;
    // 128-element tiles stage 64-byte cell rows (64B swizzle); 32-element
    // tiles stage 16-byte rows, which LmTensorMapBoxSwizzleBytes maps to
    // SWIZZLE_NONE.
    const uint32_t cell_row_bytes = tile_k / 2u;

    memset(&request, 0, sizeof(request));
    request.global_address = weight_bytes;
    request.rows = (uint64_t)k_tiles * cells * 17u;
    request.columns = cell_row_bytes;
    request.groups = group_count;
    request.box_rows = 17u * (tile_n / 16u);
    request.box_columns = cell_row_bytes;
    request.element_bits = 8u;
    return LmGemmTensorMapCached(weight, &request);
}

template<
    class FormatA,
    class FormatB,
    uint32_t TILE_N,
    uint32_t TILE_K,
	uint32_t STAGES,
	uint32_t WARPS,
	bool INDIRECT_A = false,
	uint32_t ACTIVATION_CODEC = SPARK_ACTIVATION_CODEC_NONE,
	bool INTERLEAVED_B = false>
static int32_t LmGemmLaunchAsymmetric(
    LmGemmArguments *args,
    const void *activation_bytes,
    const void *weight_bytes,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t top_k,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    cudaStream_t stream)
{
    LmLaunchShape shape;
    LmLaunchPlan plan;
    LmTileGeometry activation_geometry;
    LmTileGeometry weight_geometry;
    alignas(64) CUtensorMap activation_map;
    alignas(64) CUtensorMap weight_map;
    int32_t status;

    static_assert(
        FormatA::kMmaM == FormatB::kMmaM &&
        FormatA::kMmaN == FormatB::kMmaN &&
        FormatA::kMmaK == FormatB::kMmaK,
        "both operands must decode into the same MMA geometry");
    if (args == 0 || activation_bytes == 0 || weight_bytes == 0 ||
        packed_rows == 0u || tokens == 0u || top_k == 0u ||
        group_count == 0u || input_dimension == 0u ||
        output_dimension == 0u || multiprocessors == 0u ||
        args->group_row_offset == 0)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
	if constexpr ( INDIRECT_A )
	{
		if ( grouped == false || args->source_row_map == 0 || args->source_row_count != tokens )
			return(LM_LAUNCH_ERR_SHAPE);
	}
	else if ( args->source_row_map != 0 || args->source_row_count != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	if constexpr ( INTERLEAVED_B )
	{
		// The interleaved grid's k unit is the pack k-tile (128 or 32
		// elements), and its cell is 16 neurons; a launch on any other shape
		// would stage partial cells and misaddress every scale row.
		if ( (TILE_K != 128u && TILE_K != 32u) || (TILE_N % 16u) != 0u ||
			(input_dimension % TILE_K) != 0u || (output_dimension % 16u) != 0u )
			return(LM_LAUNCH_ERR_SHAPE);
	}
    if (grouped)
    {
        uint64_t expected_packed_rows = (uint64_t)tokens * top_k;

        if (group_count <= 1u || args->group_tile_prefix == 0 ||
            expected_packed_rows != packed_rows)
        {
            return LM_LAUNCH_ERR_SHAPE;
        }
    }
    else if (group_count != 1u || top_k != 1u || packed_rows != tokens ||
        args->prefix_built != 0u)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
    if ((args->output_bf16 == 0) == (args->output_f32 == 0))
    {
        return LM_LAUNCH_ERR_OUTPUT;
    }
    if (args->output_f32 != 0 && args->accumulate_bf16 != 0)
    {
        return LM_LAUNCH_ERR_OUTPUT;
    }
    if (args->output_row_stride == 0u)
    {
        args->output_row_stride = output_dimension;
    }
    if (args->output_column_offset > args->output_row_stride ||
        output_dimension >
            args->output_row_stride - args->output_column_offset)
    {
        return LM_LAUNCH_ERR_OUTPUT;
    }
    status = LmGemmValidateScaleTensor<FormatA>(
        &args->scale_a,
        1u,
		INDIRECT_A ? tokens : packed_rows,
        input_dimension);
    if (status != LM_LAUNCH_OK)
    {
        return status;
    }
    if constexpr ( !INTERLEAVED_B )
    {
        // Interleaved B carries its scales IN the staged cell row, so no
        // LmScaleTensor describes them - the far-plane validation applies
        // only to the plain path, where a hole would decode wrong scales.
        status = LmGemmValidateScaleTensor<FormatB>(
            &args->scale_b,
            group_count,
            output_dimension,
            input_dimension);
        if (status != LM_LAUNCH_OK)
        {
            return status;
        }
    }
    if ((input_dimension % TILE_K) != 0u)
    {
        return LM_LAUNCH_ERR_SHAPE;
    }
	if ( ACTIVATION_CODEC != SPARK_ACTIVATION_CODEC_NONE &&
		(input_dimension % SparkActivationCodecGroupSize(ACTIVATION_CODEC)) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);

    memset(&shape, 0, sizeof(shape));
    shape.tokens = tokens;
    shape.top_k = top_k;
    shape.expert_count = group_count;
    shape.input_dimension = input_dimension;
    shape.output_dimension = output_dimension;
    shape.stored_bits = FormatB::kStoredBits;
    shape.stored_bits_a = FormatA::kStoredBits;
    shape.tile_n = TILE_N;
    shape.tile_k = TILE_K;
    shape.stages = STAGES;
    shape.interleaved_b = INTERLEAVED_B ? 1u : 0u;
    status = LmLaunchPlanBuild(&shape, multiprocessors, &plan);
    if (status != LM_LAUNCH_OK)
        return status;

    memset(&activation_map, 0, sizeof(activation_map));
	if constexpr ( !INDIRECT_A && ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_NONE )
    {
        // A TILE_K=128 BF16 row is 256 bytes, wider than any hardware
        // swizzle: the interleaved direct path stages it as TWO 128-byte
        // boxes (produce), so the map's box is half a row.
        status = LmGemmEncodeActivationMap(
            &activation_map,
            activation_bytes,
            packed_rows,
            input_dimension,
            plan.tile_m,
            (INTERLEAVED_B && TILE_K == 128u) ? (TILE_K / 2u) : TILE_K,
            FormatA::kStoredBits);
        if (status != LM_TM_ENCODE_OK)
            return LM_LAUNCH_ERR_MAP;
    }
    if constexpr ( INTERLEAVED_B )
    {
        status = LmGemmEncodeWeightMapInterleaved(
            &weight_map,
            weight_bytes,
            input_dimension,
            output_dimension,
            group_count,
            TILE_N,
            TILE_K);
        if (status != LM_TM_ENCODE_OK)
            return LM_LAUNCH_ERR_MAP;
    }
    else
    {
        status = LmGemmEncodeWeightMap(
            &weight_map,
            weight_bytes,
            input_dimension,
            output_dimension,
            group_count,
            TILE_N,
            TILE_K,
            FormatB::kStoredBits);
        if (status != LM_TM_ENCODE_OK)
            return LM_LAUNCH_ERR_MAP;
    }

    args->group_count = group_count;
    args->input_dimension = input_dimension;
    args->output_dimension = output_dimension;
	args->activation_bytes = activation_bytes;
    if (group_count > 1u && args->prefix_built == 0u)
    {
        LmGemmTilePrefixKernel<32u><<<1u, 32u, 0u, stream>>>(
            args->group_row_offset,
            group_count,
            plan.tile_m,
            (output_dimension + TILE_N - 1u) / TILE_N,
            args->group_tile_prefix);
        if (cudaPeekAtLastError() != cudaSuccess)
            return LM_LAUNCH_ERR_LAUNCH;
    }

    memset(&activation_geometry, 0, sizeof(activation_geometry));
    memset(&weight_geometry, 0, sizeof(weight_geometry));
    activation_geometry.rows = plan.tile_m;
    activation_geometry.depth = TILE_K;
    activation_geometry.element_bits = FormatA::kStoredBits;
    if constexpr ( INTERLEAVED_B )
    {
        // The staged B geometry IS the cell grid: 17 rows per 16 neurons,
        // each row 64 bytes of UINT8.
        weight_geometry.rows = 17u * (TILE_N / 16u);
        weight_geometry.depth = 64u;
        weight_geometry.element_bits = 8u;
    }
    else
    {
        weight_geometry.rows = TILE_N;
        weight_geometry.depth = TILE_K;
        weight_geometry.element_bits = FormatB::kStoredBits;
    }

    switch (plan.tile_m)
    {
        case 16u:
            return LmGemmLaunchTile<
                FormatA,
                FormatB,
                16u,
                TILE_N,
                TILE_K,
					STAGES,
					WARPS,
					INDIRECT_A,
					ACTIVATION_CODEC,
					INTERLEAVED_B>(
                    *args,
                    activation_map,
                    weight_map,
                    activation_geometry,
                    weight_geometry,
                    grouped,
                    plan,
                    stream) == cudaSuccess
                ? LM_LAUNCH_OK
                : LM_LAUNCH_ERR_LAUNCH;
        case 32u:
            return LmGemmLaunchTile<
                FormatA,
                FormatB,
                32u,
                TILE_N,
                TILE_K,
					STAGES,
					WARPS,
					INDIRECT_A,
					ACTIVATION_CODEC,
					INTERLEAVED_B>(
                    *args,
                    activation_map,
                    weight_map,
                    activation_geometry,
                    weight_geometry,
                    grouped,
                    plan,
                    stream) == cudaSuccess
                ? LM_LAUNCH_OK
                : LM_LAUNCH_ERR_LAUNCH;
        case 64u:
            return LmGemmLaunchTile<
                FormatA,
                FormatB,
                64u,
                TILE_N,
                TILE_K,
					STAGES,
					WARPS,
					INDIRECT_A,
					ACTIVATION_CODEC,
					INTERLEAVED_B>(
                    *args,
                    activation_map,
                    weight_map,
                    activation_geometry,
                    weight_geometry,
                    grouped,
                    plan,
                    stream) == cudaSuccess
                ? LM_LAUNCH_OK
                : LM_LAUNCH_ERR_LAUNCH;
        default:
            return LM_LAUNCH_ERR_TILE;
    }
}

template<
    class WeightFormat,
	uint32_t TILE_N,
	uint32_t STAGES,
	uint32_t WARPS,
	bool INDIRECT_A,
	uint32_t ACTIVATION_CODEC>
static int32_t LmGemmWeightOnlyLaunchMode(
    LmGemmArguments *args,
    const void *activation_bf16,
    const void *weight_bytes,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t top_k,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    cudaStream_t stream)
{
	constexpr uint32_t tile_k =
		LmTileKIsTmaLoadable(64u,LmBf16Format::kStoredBits,LmBf16Format::kTmaSwizzle) &&
		LmTileKIsTmaLoadable(64u,WeightFormat::kStoredBits,WeightFormat::kTmaSwizzle) ? 64u :
		LmTileKIsTmaLoadable(128u,LmBf16Format::kStoredBits,LmBf16Format::kTmaSwizzle) &&
		LmTileKIsTmaLoadable(128u,WeightFormat::kStoredBits,WeightFormat::kTmaSwizzle) ? 128u : 256u;
	static_assert(LmTileKIsTmaLoadable(tile_k,LmBf16Format::kStoredBits,LmBf16Format::kTmaSwizzle),
		"weight-only activation tile is not TMA-loadable");
	static_assert(LmTileKIsTmaLoadable(tile_k,WeightFormat::kStoredBits,WeightFormat::kTmaSwizzle),
		"weight-only weight tile is not TMA-loadable");
    return LmGemmLaunchAsymmetric<
        LmBf16Format,
        WeightFormat,
        TILE_N,
        tile_k,
		STAGES,
		WARPS,
			INDIRECT_A,
			ACTIVATION_CODEC>(
            args,
            activation_bf16,
            weight_bytes,
            packed_rows,
            tokens,
            top_k,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            stream);
}

template<
	class WeightFormat,
	uint32_t TILE_N,
	uint32_t STAGES,
	uint32_t WARPS,
	uint32_t ACTIVATION_CODEC = SPARK_ACTIVATION_CODEC_NONE>
static int32_t LmGemmWeightOnlyLaunch(
	LmGemmArguments *args,
	const void *activation_bf16,
	const void *weight_bytes,
	uint32_t packed_rows,
	uint32_t tokens,
	uint32_t top_k,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	uint32_t multiprocessors,
	bool grouped,
	cudaStream_t stream)
{
	return(LmGemmWeightOnlyLaunchMode<WeightFormat,TILE_N,STAGES,WARPS,false,ACTIVATION_CODEC>(
		args,activation_bf16,weight_bytes,packed_rows,tokens,top_k,group_count,
		input_dimension,output_dimension,multiprocessors,grouped,stream));
}

template<
	class WeightFormat,
	uint32_t TILE_N,
	uint32_t STAGES,
	uint32_t WARPS,
	uint32_t ACTIVATION_CODEC = SPARK_ACTIVATION_CODEC_NONE>
static int32_t LmGemmWeightOnlyIndirectLaunch(
	LmGemmArguments *args,
	const void *activation_bf16,
	const void *weight_bytes,
	uint32_t packed_rows,
	uint32_t tokens,
	uint32_t top_k,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	uint32_t multiprocessors,
	cudaStream_t stream)
{
	return(LmGemmWeightOnlyLaunchMode<WeightFormat,TILE_N,STAGES,WARPS,true,ACTIVATION_CODEC>(
		args,activation_bf16,weight_bytes,packed_rows,tokens,top_k,group_count,
		input_dimension,output_dimension,multiprocessors,true,stream));
}

// INTERLEAVED B LAUNCHERS (K3 pack V2 mxfp4_ws_interleaved_v1). TILE_K is
// forced to the pack grid's 128-element k-tile - one staged box per k step,
// scales riding the cell row - so these never pass through
// LmGemmWeightOnlyLaunchMode's tile_k selection. The indirect variant is the
// w1 path (un-gathered latent through the route map); the direct variant is
// the w2 path, whose already-packed SiTU output stages as two 128-byte TMA
// boxes per row (a 256-byte BF16 row has no single hardware swizzle).
template<
	class WeightFormat,
	uint32_t TILE_N,
	uint32_t STAGES,
	uint32_t WARPS,
	uint32_t TILE_K = 128u>
static int32_t LmGemmWeightOnlyInterleavedLaunch(
	LmGemmArguments *args,
	const void *activation_bf16,
	const void *weight_bytes,
	uint32_t packed_rows,
	uint32_t tokens,
	uint32_t top_k,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	uint32_t multiprocessors,
	bool grouped,
	cudaStream_t stream)
{
	return(LmGemmLaunchAsymmetric<
		LmBf16Format,WeightFormat,TILE_N,TILE_K,STAGES,WARPS,
		false,SPARK_ACTIVATION_CODEC_NONE,true>(
		args,activation_bf16,weight_bytes,packed_rows,tokens,top_k,group_count,
		input_dimension,output_dimension,multiprocessors,grouped,stream));
}

template<
	class WeightFormat,
	uint32_t TILE_N,
	uint32_t STAGES,
	uint32_t WARPS,
	uint32_t TILE_K = 128u>
static int32_t LmGemmWeightOnlyIndirectInterleavedLaunch(
	LmGemmArguments *args,
	const void *activation_bf16,
	const void *weight_bytes,
	uint32_t packed_rows,
	uint32_t tokens,
	uint32_t top_k,
	uint32_t group_count,
	uint32_t input_dimension,
	uint32_t output_dimension,
	uint32_t multiprocessors,
	cudaStream_t stream)
{
	return(LmGemmLaunchAsymmetric<
		LmBf16Format,WeightFormat,TILE_N,TILE_K,STAGES,WARPS,
		true,SPARK_ACTIVATION_CODEC_NONE,true>(
		args,activation_bf16,weight_bytes,packed_rows,tokens,top_k,group_count,
		input_dimension,output_dimension,multiprocessors,true,stream));
}

template<
    class Format,
    uint32_t TILE_N,
    uint32_t TILE_K,
    uint32_t STAGES,
    uint32_t WARPS>
static int32_t LmGemmLaunch(
    LmGemmArguments *args,
    const void *activation_bytes,
    const void *weight_bytes,
    uint32_t packed_rows,
    uint32_t tokens,
    uint32_t top_k,
    uint32_t group_count,
    uint32_t input_dimension,
    uint32_t output_dimension,
    uint32_t multiprocessors,
    bool grouped,
    cudaStream_t stream)
{
    return LmGemmLaunchAsymmetric<
        Format,
        Format,
        TILE_N,
        TILE_K,
        STAGES,
        WARPS>(
            args,
            activation_bytes,
            weight_bytes,
            packed_rows,
            tokens,
            top_k,
            group_count,
            input_dimension,
            output_dimension,
            multiprocessors,
            grouped,
            stream);
}

#undef LM_GEMM_MAX_TRACKED_CUDA_DEVICES
