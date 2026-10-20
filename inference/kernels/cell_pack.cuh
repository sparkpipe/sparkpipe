#pragma once

#include "inference/kernels/formats/mxfp4.cuh"
#include "inference/kernels/skinny.cuh"

static __global__ void LmCellPackMxfp4Kernel(const uint8_t *__restrict__ payload, const uint8_t *__restrict__ scales,
	uint8_t *__restrict__ cells, uint32_t output_dimension, uint32_t input_dimension, uint64_t chunks)
{
	const uint64_t chunk = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t k_tiles = input_dimension / LM_SKINNY_CELL_TILE_K, cell_count = output_dimension / LM_SKINNY_CELL_NEURONS;
	const uint64_t per_group = (uint64_t)k_tiles * cell_count * LM_SKINNY_CELL_ROWS;
	const uint64_t payload_pitch = input_dimension / 2u, scale_pitch = input_dimension / LM_MXFP4_GROUP;
	uint8_t *target = cells + chunk * LM_SKINNY_CHUNK_BYTES;
	uint64_t group, local;
	uint32_t tile, cell, row, n;
	if ( chunk >= chunks )
		return;
	group = chunk / per_group;
	local = chunk % per_group;
	row = (uint32_t)(local % LM_SKINNY_CELL_ROWS);
	cell = (uint32_t)((local / LM_SKINNY_CELL_ROWS) % cell_count);
	tile = (uint32_t)(local / ((uint64_t)LM_SKINNY_CELL_ROWS * cell_count));
	if ( row < LM_SKINNY_CELL_NEURONS )
	{
		const uint8_t *source = payload + (group * output_dimension + (uint64_t)cell * LM_SKINNY_CELL_NEURONS + row) * payload_pitch +
			(uint64_t)tile * (LM_SKINNY_CELL_TILE_K / 2u);
		for ( n = 0u; n < LM_SKINNY_CHUNK_BYTES; ++n )
			target[n] = source[n];
		return;
	}
	for ( n = 0u; n < LM_SKINNY_CELL_NEURONS; ++n )
		target[n] = scales[(group * output_dimension + (uint64_t)cell * LM_SKINNY_CELL_NEURONS + n) * scale_pitch + tile];
}

static uint64_t LmCellPackBytes(uint32_t groups, uint32_t output_dimension, uint32_t input_dimension)
{
	return((uint64_t)groups * (input_dimension / LM_SKINNY_CELL_TILE_K) * (output_dimension / LM_SKINNY_CELL_NEURONS) *
		LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES);
}

static int32_t LmCellPackMxfp4(const void *payload, const void *scales, void *cells, uint32_t groups, uint32_t output_dimension,
	uint32_t input_dimension, cudaStream_t stream)
{
	uint64_t chunks;
	if ( payload == 0 || scales == 0 || cells == 0 || groups == 0u || output_dimension == 0u || input_dimension == 0u ||
		(output_dimension % LM_SKINNY_CELL_NEURONS) != 0u || (input_dimension % LM_SKINNY_CELL_TILE_K) != 0u ||
		LmSkinnyAligned(cells) == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	chunks = LmCellPackBytes(groups, output_dimension, input_dimension) / LM_SKINNY_CHUNK_BYTES;
	LmCellPackMxfp4Kernel<<<(uint32_t)((chunks + 255u) / 256u), 256u, 0, stream>>>((const uint8_t *)payload, (const uint8_t *)scales,
		(uint8_t *)cells, output_dimension, input_dimension, chunks);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}
