#pragma once


#include "inference/kernels/activation.cuh"
#include "inference/kernels/frame_error.cuh"
#include "inference/kernels/layout.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/tma.cuh"
#include <stdint.h>


#define LM_PIPELINE_STAGES 2u
#define LM_PIPELINE_LOOKAHEAD (LM_PIPELINE_STAGES - 1u)

static __device__ __forceinline__ uint32_t LmPipelineStage(uint32_t k_tile, uint32_t stages)
{
	return(k_tile % stages);
}


static __device__ __forceinline__ uint32_t LmPipelineAhead(uint32_t k_tile, uint32_t stages)
{
	return(k_tile + stages - 1u);
}

template<uint32_t STAGES>
static __device__ void LmPipelineInitialise(uint64_t *barrier, uint32_t arrive_count)
{
	uint32_t stage;
	if ( threadIdx.x == 0u )
		for (stage = 0u; stage < STAGES; ++stage)
			LmMbarrierInit(&barrier[stage],arrive_count);
	LmMbarrierInitFence();
	__syncthreads();
}

template<uint32_t STAGES>
static __device__ void LmPipelineRelease(uint64_t *barrier)
{
	uint32_t stage;
	__syncthreads();
	if ( threadIdx.x == 0u )
		for (stage = 0u; stage < STAGES; ++stage)
			LmMbarrierInvalidate(&barrier[stage]);
}






struct LmTileGeometry
{
	uint32_t rows;
	uint32_t depth;
	uint32_t element_bits;
};

static __device__ __forceinline__ void LmPipelineProduceWeight(
    const LmTileGeometry *a,
    const LmTileGeometry *b,
    const void *tensor_map_b,
    void *stage_b,
    uint64_t *barrier,
    uint32_t k_byte_b,
    uint32_t neuron_base,
    uint32_t group_index,
    bool grouped)
{
	if ( threadIdx.x != 0u )
		return;
	LmMbarrierArriveExpect(barrier,
		LmTileBytes(a->rows,a->depth,a->element_bits)
		+ LmTileBytes(b->rows,b->depth,b->element_bits));
	if ( grouped )
		LmTmaLoad3d(stage_b,tensor_map_b,barrier,(int32_t)k_byte_b,(int32_t)neuron_base,(int32_t)group_index);
	else
		LmTmaLoad2d(stage_b,tensor_map_b,barrier,(int32_t)k_byte_b,(int32_t)neuron_base);
}

static __device__ __forceinline__ void LmPipelineProduceWeightInterleaved(
    const LmTileGeometry *a,
    const LmTileGeometry *b,
    const void *tensor_map_b,
    void *stage_b,
    uint64_t *barrier,
    uint32_t k_tile,
    uint32_t cells_total,
    uint32_t neuron_base,
    uint32_t group_index)
{
	if ( threadIdx.x != 0u )
		return;
	LmMbarrierArriveExpect(barrier,
		LmTileBytes(a->rows,a->depth,a->element_bits)
		+ LmTileBytes(b->rows,b->depth,b->element_bits));
	LmTmaLoad3d(stage_b,tensor_map_b,barrier,0,
		(int32_t)(((k_tile * cells_total) + (neuron_base / 16u)) * 17u),
		(int32_t)group_index);
}

template<uint32_t TILE_K>
static __device__ __forceinline__ void LmPipelineProduce(
    const LmTileGeometry *a,
    const LmTileGeometry *b,
    const void *tensor_map_a,
    const void *tensor_map_b,
    void *stage_a,
    void *stage_b,
    uint64_t *barrier,
    uint32_t row_base,
    uint32_t neuron_base,
    uint32_t k_tile,
    uint32_t group_index,
    bool grouped,
    bool interleaved_b,
    uint32_t cells_total)
{
	uint32_t k_byte_a = k_tile * LmTileBytes(1u,a->depth,a->element_bits);
	uint32_t k_byte_b = k_tile * LmTileBytes(1u,b->depth,b->element_bits);
	if ( interleaved_b )
		LmPipelineProduceWeightInterleaved(a,b,tensor_map_b,stage_b,barrier,
			k_tile,cells_total,neuron_base,group_index);
	else
		LmPipelineProduceWeight(a,b,tensor_map_b,stage_b,barrier,k_byte_b,neuron_base,group_index,grouped);
	if ( threadIdx.x != 0u )
		return;
	if ( interleaved_b && TILE_K == 128u )
	{
		const uint32_t a_block_bytes =
			LmTileBytes(a->rows,64u,a->element_bits);
		LmTmaLoad2d(stage_a,tensor_map_a,barrier,(int32_t)k_byte_a,(int32_t)row_base);
		LmTmaLoad2d((uint8_t *)stage_a + a_block_bytes,
			tensor_map_a,barrier,(int32_t)(k_byte_a + 128u),(int32_t)row_base);
		return;
	}
	LmTmaLoad2d(stage_a,tensor_map_a,barrier,(int32_t)k_byte_a,(int32_t)row_base);
}

template<class FormatA, uint32_t TILE_K, bool INTERLEAVED_B = false>
static __device__ __forceinline__ void LmPipelineProduceIndirectA(
	const LmTileGeometry *a,
	const LmTileGeometry *b,
	const void *tensor_map_b,
	const uint8_t *activation_bytes,
	const uint32_t *source_row_map,
	uint32_t source_row_count,
	uint32_t input_dimension,
	void *stage_a,
	void *stage_b,
	uint64_t *barrier,
	uint32_t row_base,
	uint32_t row_limit,
	uint32_t neuron_base,
	uint32_t k_tile,
	uint32_t group_index,
	bool grouped,
	uint32_t cells_total,
	LmFrameError *frame_error = 0)
{
	const uint32_t row_pitch = LmTileBytes(1u,a->depth,a->element_bits);
	const uint32_t chunk_count = row_pitch / LM_SWIZZLE_CHUNK_BYTES;
	uint32_t index,local_row,chunk,packed_row,source_row,destination_offset;
	uint64_t source_offset;
	static_assert(FormatA::kStoredBits % 8u == 0u,
		"indirect activation staging requires byte-addressable rows");
	static_assert(LmTileBytes(1u,FormatA::kTileK,FormatA::kStoredBits) % LM_SWIZZLE_CHUNK_BYTES == 0u,
		"indirect activation staging requires complete 16-byte chunks");
	if constexpr ( INTERLEAVED_B )
		LmPipelineProduceWeightInterleaved(a,b,tensor_map_b,stage_b,barrier,
			k_tile,cells_total,neuron_base,group_index);
	else
		LmPipelineProduceWeight(a,b,tensor_map_b,stage_b,barrier,
			k_tile * LmTileBytes(1u,b->depth,b->element_bits),neuron_base,group_index,grouped);
	for (index=threadIdx.x; index<a->rows * chunk_count; index+=blockDim.x)
	{
		local_row = index / chunk_count;
		chunk = index % chunk_count;
		packed_row = row_base + local_row < row_limit ? row_base + local_row : row_base;
		source_row = LmRouteSourceRow(source_row_map,packed_row);
		if ( source_row >= source_row_count )
		{
			LmFrameErrorReport(frame_error,
				(uint32_t)LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE,
				0u,packed_row,source_row,row_base,source_row_count);
			source_row = 0u;
		}
		source_offset = ((uint64_t)source_row * ((input_dimension * FormatA::kStoredBits) / 8u))
			+ (k_tile * row_pitch) + (chunk * LM_SWIZZLE_CHUNK_BYTES);
		if constexpr ( INTERLEAVED_B && TILE_K == 128u )
		{
			const uint32_t a_block_bytes =
				LmTileBytes(a->rows,64u,a->element_bits);
			destination_offset = ((chunk / 8u) * a_block_bytes)
				+ LmSwizzledOffset(local_row,
					(chunk % 8u) * LM_SWIZZLE_CHUNK_BYTES,128u,128u);
		}
		else
		{
			destination_offset = FormatA::kTmaSwizzle
				? LmSwizzledOffset(local_row,chunk * LM_SWIZZLE_CHUNK_BYTES,row_pitch,LmSwizzleSpanFor(row_pitch))
				: (local_row * row_pitch) + (chunk * LM_SWIZZLE_CHUNK_BYTES);
		}
		LmTmaLoadBulk1d(
			(uint8_t *)stage_a + destination_offset,
			activation_bytes + source_offset,
			barrier,
			LM_SWIZZLE_CHUNK_BYTES);
	}
}

template<class FormatA,uint32_t TILE_ROWS,uint32_t TILE_K,uint32_t ACTIVATION_CODEC>
static __device__ __forceinline__ void LmPipelineProduceManualA(
	const LmTileGeometry *a,
	const LmTileGeometry *b,
	const void *tensor_map_b,
	const uint8_t *activation_bytes,
	const uint32_t *source_row_map,
	void *stage_a,
	void *stage_b,
	uint64_t *barrier,
	uint32_t row_base,
	uint32_t row_limit,
	uint32_t source_row_count,
	uint32_t input_dimension,
	uint32_t neuron_base,
	uint32_t k_tile,
	uint32_t group_index,
	bool grouped,
	LmFrameError *frame_error = 0)
{
	static_assert(ACTIVATION_CODEC != SPARK_ACTIVATION_CODEC_NONE,
		"the codec-free indirect path is LmPipelineProduceIndirectA");
	static_assert(FormatA::kStoredBits % 8u == 0u,
		"indirect activation staging requires byte-addressable rows");
	static_assert(LmTileBytes(1u,FormatA::kTileK,FormatA::kStoredBits) % LM_SWIZZLE_CHUNK_BYTES == 0u,
		"indirect activation staging requires complete 16-byte chunks");
	if ( threadIdx.x == 0u )
	{
		LmMbarrierArriveExpect(barrier,LmTileBytes(b->rows,b->depth,b->element_bits));
		if ( grouped )
			LmTmaLoad3d(stage_b,tensor_map_b,barrier,
				(int32_t)(k_tile * LmTileBytes(1u,b->depth,b->element_bits)),
				(int32_t)neuron_base,(int32_t)group_index);
		else
			LmTmaLoad2d(stage_b,tensor_map_b,barrier,
				(int32_t)(k_tile * LmTileBytes(1u,b->depth,b->element_bits)),
				(int32_t)neuron_base);
	}
	if constexpr ( ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0 )
	{
		static_assert(FormatA::kStoredBits == 16u,"FP8 QDQ source must be BF16");
		LmActivationStageFp8Qdq<TILE_ROWS,TILE_K,FormatA::kTmaSwizzle,ACTIVATION_CODEC>(
			activation_bytes,source_row_map,source_row_count,row_base,row_limit,
			k_tile * TILE_K,input_dimension,stage_a,0u,blockDim.x / 32u,
			frame_error);
	}
}

static __device__ __forceinline__ uint32_t LmGroupOfTile(const uint32_t *group_tile_prefix, uint32_t group_count, uint32_t tile_index)
{
	uint32_t low = 0u,high = group_count,middle;
	while ( low + 1u < high )
	{
		middle = (low + high) >> 1u;
		if ( group_tile_prefix[middle] <= tile_index )
			low = middle;
		else
			high = middle;
	}
	return(low);
}

static __device__ __forceinline__ uint32_t LmTotalTiles(const uint32_t *group_tile_prefix, uint32_t group_count)
{
	return(group_tile_prefix[group_count]);
}
