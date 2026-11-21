#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "runtime/gemm.cuh"

#define PROBE_ROWS 17u
#define PROBE_ROW_BYTES 64u
#define PROBE_GROUPS 2u
#define PROBE_WIDE_ROWS 4u
#define PROBE_WIDE_ROW_BYTES 256u
#define PROBE_WIDE_BOX_BYTES 128u
#define PROBE_STAGE_ALIGN 1024u

#define CUDA(call) do { cudaError_t e = (call); if ( e != cudaSuccess ) { printf("FAIL line=%d cuda=%s call=%s\n", __LINE__, cudaGetErrorString(e), #call); exit(1); } } while ( 0 )

template<uint32_t BYTES>
__global__ void ProbeLoadKernel(const __grid_constant__ CUtensorMap map, uint32_t rank, int32_t x, int32_t y, int32_t z, uint8_t *dump)
{
    __shared__ __align__(PROBE_STAGE_ALIGN) uint8_t stage[BYTES];
    __shared__ uint64_t barrier;
    if ( threadIdx.x == 0u )
    {
        LmMbarrierInit(&barrier, 1u);
        LmMbarrierInitFence();
        __threadfence_block();
        LmMbarrierArriveExpect(&barrier, BYTES);
        if ( rank == 3u )
            LmTmaLoad3d(stage, &map, &barrier, x, y, z);
        else
            LmTmaLoad2d(stage, &map, &barrier, x, y);
        LmMbarrierWait(&barrier, 0u);
        for ( uint32_t i = 0u; i < BYTES; ++i )
            dump[i] = stage[i];
    }
}

static uint8_t ProbeByte(uint64_t offset)
{
    uint64_t value = offset * 0x9e3779b97f4a7c15ull;
    return (uint8_t)((value >> 56u) ^ (value >> 24u) ^ offset);
}

static uint32_t Swizzled(uint32_t offset, uint32_t swizzle_bytes)
{
    const uint32_t chunks = swizzle_bytes / 16u;
    return offset ^ (((offset >> 7u) & (chunks - 1u)) << 4u);
}

template<uint32_t BYTES>
static int Probe(const char *name, const LmTensorMapRequest *shape, uint64_t global_bytes, uint32_t rank, int32_t x, int32_t y, int32_t z)
{
    const uint32_t box_row_bytes = (uint32_t)((uint64_t)shape->box_columns * shape->element_bits / 8u);
    const uint64_t row_bytes = shape->columns * shape->element_bits / 8u;
    std::vector<uint8_t> global(global_bytes), dump(BYTES), expected(BYTES);
    uint8_t *device_global = 0, *device_dump = 0;
    alignas(64) CUtensorMap map;
    LmTensorMapRequest request = *shape;
    uint32_t row, byte, mismatches = 0u;
    if ( (uint64_t)box_row_bytes * shape->box_rows != BYTES || row_bytes * shape->rows * shape->groups != global_bytes )
    {
        printf("FAIL %s: the box is %u bytes and the stage %u; the tensor spans %llu bytes and the buffer %llu\n", name,
            box_row_bytes * shape->box_rows, BYTES, (unsigned long long)(row_bytes * shape->rows * shape->groups), (unsigned long long)global_bytes);
        return 1;
    }
    for ( uint64_t i = 0u; i < global_bytes; ++i )
        global[i] = ProbeByte(i);
    CUDA(cudaMalloc((void **)&device_global, global_bytes));
    CUDA(cudaMalloc((void **)&device_dump, BYTES));
    CUDA(cudaMemcpy(device_global, global.data(), global_bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemset(device_dump, 0, BYTES));
    request.global_address = device_global;
    if ( LmGemmTensorMapCached(&map, &request) != LM_TM_ENCODE_OK )
    {
        printf("FAIL %s: the tensor map did not encode\n", name);
        return 1;
    }
    ProbeLoadKernel<BYTES><<<1u, 1u>>>(map, rank, x, y, z, device_dump);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(dump.data(), device_dump, BYTES, cudaMemcpyDeviceToHost));
    for ( row = 0u; row < shape->box_rows; ++row )
        for ( byte = 0u; byte < box_row_bytes; ++byte )
        {
            const uint64_t source = ((uint64_t)z * shape->rows + (uint64_t)y + row) * row_bytes + (uint64_t)x + byte;
            expected[Swizzled(row * box_row_bytes + byte, box_row_bytes)] = global[source];
        }
    for ( row = 0u; row < BYTES; ++row )
        if ( dump[row] != expected[row] )
        {
            if ( mismatches == 0u )
                printf("FAIL %s: stage byte %u holds %u, the %u-byte swizzle of the box puts %u there\n", name, row, dump[row], box_row_bytes, expected[row]);
            mismatches++;
        }
    CUDA(cudaFree(device_global));
    CUDA(cudaFree(device_dump));
    if ( mismatches != 0u )
    {
        printf("FAIL %s: %u of %u staged bytes differ\n", name, mismatches, BYTES);
        return 1;
    }
    printf("PASS %s: %u staged bytes follow the %u-byte swizzle of the box at (%d, %d, %d)\n", name, BYTES, box_row_bytes, x, y, z);
    return 0;
}

int main(void)
{
    LmTensorMapRequest rows, groups, wide;
    int failures = 0;
    CUDA(cudaSetDevice(0));
    memset(&rows, 0, sizeof(rows));
    rows.rows = PROBE_ROWS;
    rows.columns = PROBE_ROW_BYTES;
    rows.groups = 1u;
    rows.box_rows = PROBE_ROWS;
    rows.box_columns = PROBE_ROW_BYTES;
    rows.element_bits = 8u;
    groups = rows;
    groups.groups = PROBE_GROUPS;
    memset(&wide, 0, sizeof(wide));
    wide.rows = PROBE_WIDE_ROWS;
    wide.columns = PROBE_WIDE_ROW_BYTES / 2u;
    wide.groups = 1u;
    wide.box_rows = PROBE_WIDE_ROWS;
    wide.box_columns = PROBE_WIDE_BOX_BYTES / 2u;
    wide.element_bits = 16u;
    failures += Probe<PROBE_ROWS * PROBE_ROW_BYTES>("ragged 17-row box, 64-byte rows", &rows, PROBE_ROWS * PROBE_ROW_BYTES, 2u, 0, 0, 0);
    failures += Probe<PROBE_ROWS * PROBE_ROW_BYTES>("rank-3 map, second group", &groups, PROBE_GROUPS * PROBE_ROWS * PROBE_ROW_BYTES, 3u, 0, 0, 1);
    failures += Probe<PROBE_WIDE_ROWS * PROBE_WIDE_BOX_BYTES>("bf16 rows, first 128-byte box", &wide, PROBE_WIDE_ROWS * PROBE_WIDE_ROW_BYTES, 2u, 0, 0, 0);
    failures += Probe<PROBE_WIDE_ROWS * PROBE_WIDE_BOX_BYTES>("bf16 rows, second 128-byte box", &wide, PROBE_WIDE_ROWS * PROBE_WIDE_ROW_BYTES, 2u, (int32_t)PROBE_WIDE_BOX_BYTES, 0, 0);
    if ( failures != 0 )
        return 1;
    printf("PASS k3 swizzle probe\n");
    return 0;
}
