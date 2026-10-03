#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

typedef struct LmLaneOwnerTable
{
    const uint32_t *row_owner;
    const uint32_t *row_slot;
    uint32_t rows;
    uint32_t owner_count;
    uint32_t owner_rows;
} LmLaneOwnerTable;

template <uint32_t THREADS>
__global__ void LmLaneOwnerPackKernel(
    const uint16_t *source,
    uint16_t *packed,
    const uint32_t *row_owner,
    const uint32_t *row_slot,
    uint32_t width,
    uint32_t owner_rows)
{
    const uint32_t row = blockIdx.x;
    const uint16_t *from = source + (uint64_t)row * width;
    uint16_t *to = packed + ((uint64_t)row_owner[row] * owner_rows + row_slot[row]) * width;
    for (uint32_t i = threadIdx.x; i < width; i += THREADS)
        to[i] = from[i];
}

template <uint32_t THREADS>
__global__ void LmLaneOwnerUnpackKernel(
    const uint16_t *packed,
    uint16_t *destination,
    const uint32_t *row_owner,
    const uint32_t *row_slot,
    uint32_t width,
    uint32_t owner_rows)
{
    const uint32_t row = blockIdx.x;
    const uint16_t *from = packed + ((uint64_t)row_owner[row] * owner_rows + row_slot[row]) * width;
    uint16_t *to = destination + (uint64_t)row * width;
    for (uint32_t i = threadIdx.x; i < width; i += THREADS)
        to[i] = from[i];
}

template <uint32_t THREADS>
static inline cudaError_t LmLaneOwnerPack(
    const uint16_t *source,
    uint16_t *packed,
    const LmLaneOwnerTable *table,
    uint32_t width,
    cudaStream_t stream)
{
    if (table == 0 || table->rows == 0u)
        return cudaSuccess;
    LmLaneOwnerPackKernel<THREADS><<<table->rows, THREADS, 0, stream>>>(
        source, packed, table->row_owner, table->row_slot, width, table->owner_rows);
    return cudaPeekAtLastError();
}

template <uint32_t THREADS>
static inline cudaError_t LmLaneOwnerUnpack(
    const uint16_t *packed,
    uint16_t *destination,
    const LmLaneOwnerTable *table,
    uint32_t width,
    cudaStream_t stream)
{
    if (table == 0 || table->rows == 0u)
        return cudaSuccess;
    LmLaneOwnerUnpackKernel<THREADS><<<table->rows, THREADS, 0, stream>>>(
        packed, destination, table->row_owner, table->row_slot, width, table->owner_rows);
    return cudaPeekAtLastError();
}
