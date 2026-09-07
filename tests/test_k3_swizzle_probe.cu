
#include <stdio.h>
#include <string.h>
#include "runtime/gemm.cuh"

#define PROBE_ROWS 17u

__global__ void SwizzleProbeKernel(const CUtensorMap *map, uint8_t *dump)
{
    __shared__ __align__(128) uint8_t stage[PROBE_ROWS * 64u];
    __shared__ uint64_t barrier;
    if ( threadIdx.x == 0u )
    {
        LmMbarrierInit(&barrier, 1u);
        LmMbarrierInitFence();
        __threadfence_block();
        LmMbarrierArriveExpect(&barrier, PROBE_ROWS * 64u);
        LmTmaLoad2d(stage, map, &barrier, 0, 0);
        LmMbarrierWait(&barrier, 0u);
        for ( uint32_t i = 0u; i < PROBE_ROWS * 64u; ++i )
            dump[i] = stage[i];
    }
}

__global__ void Rank3ProbeKernel(const CUtensorMap *map, uint8_t *dump)
{
    __shared__ __align__(128) uint8_t stage[PROBE_ROWS * 64u];
    __shared__ uint64_t barrier;
    if ( threadIdx.x == 0u )
    {
        LmMbarrierInit(&barrier, 1u);
        LmMbarrierInitFence();
        __threadfence_block();
        LmMbarrierArriveExpect(&barrier, PROBE_ROWS * 64u);
        LmTmaLoad3d(stage, map, &barrier, 0, 0, 0);
        LmMbarrierWait(&barrier, 0u);
        for ( uint32_t i = 0u; i < PROBE_ROWS * 64u; ++i )
            dump[i] = stage[i];
    }
}

__global__ void TwoBoxProbeKernel(const CUtensorMap *map, uint16_t *dump)
{
    __shared__ __align__(128) uint16_t stage[4u * 128u];
    __shared__ uint64_t barrier;
    if ( threadIdx.x == 0u )
    {
        LmMbarrierInit(&barrier, 1u);
        LmMbarrierInitFence();
        __threadfence_block();
        LmMbarrierArriveExpect(&barrier, 4u * 256u);
        LmTmaLoad2d(stage, map, &barrier, 0, 0);
        LmTmaLoad2d(stage + 64u, map, &barrier, 128, 0);
        LmMbarrierWait(&barrier, 0u);
        for ( uint32_t i = 0u; i < 4u * 128u; ++i )
            dump[i] = stage[i];
    }
}

int main(void)
{
    cudaError_t err;
    int32_t status;
    uint8_t *d_global = 0, *d_dump = 0;
    uint8_t h_global[PROBE_ROWS * 64u], h_dump[PROBE_ROWS * 64u];
    alignas(64) CUtensorMap map;
    LmTensorMapRequest request;

    for ( uint32_t r = 0u; r < PROBE_ROWS; ++r )
        for ( uint32_t b = 0u; b < 64u; ++b )
            h_global[r * 64u + b] = (uint8_t)(((r % 4u) * 64u) + b);

    cudaSetDevice(0);
    cudaMalloc(&d_global, sizeof(h_global));
    cudaMalloc(&d_dump, sizeof(h_dump));
    cudaMemcpy(d_global, h_global, sizeof(h_global), cudaMemcpyHostToDevice);

    memset(&request, 0, sizeof(request));
    request.global_address = d_global;
    request.rows = PROBE_ROWS;
    request.columns = 64u;
    request.groups = 1u;
    request.box_rows = PROBE_ROWS;
    request.box_columns = 64u;
    request.element_bits = 8u;
    status = LmGemmTensorMapCached(&map, &request);
    if ( status != LM_TM_ENCODE_OK ) { printf("encode %d\n", status); return 1; }

    SwizzleProbeKernel<<<1u, 1u>>>(&map, d_dump);
    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess ) { printf("sync %d\n", (int)err); return 1; }
    cudaMemcpy(h_dump, d_dump, sizeof(h_dump), cudaMemcpyDeviceToHost);

    printf("per staged row: first 8 bytes and last 8 bytes as [src_row%%4, byte]:\n");
    for ( uint32_t r = 0u; r < PROBE_ROWS; ++r )
    {
        printf("staged row %2u: head", r);
        for ( uint32_t b = 0u; b < 8u; ++b )
            printf(" %3u", h_dump[r * 64u + b]);
        printf("  tail");
        for ( uint32_t b = 56u; b < 64u; ++b )
            printf(" %3u", h_dump[r * 64u + b]);
        printf("\n");
    }

    {
        uint8_t *d_w = 0, *d_d3 = 0;
        uint8_t h_w[2u * PROBE_ROWS * 64u], h_d3[PROBE_ROWS * 64u];
        alignas(64) CUtensorMap wmap;
        LmTensorMapRequest wreq;
        for ( uint32_t e = 0u; e < 2u; ++e )
            for ( uint32_t r = 0u; r < PROBE_ROWS; ++r )
                for ( uint32_t b = 0u; b < 64u; ++b )
                    h_w[(e * PROBE_ROWS + r) * 64u + b] = (uint8_t)(((r % 4u) * 64u) + b);
        cudaMalloc(&d_w, sizeof(h_w));
        cudaMalloc(&d_d3, sizeof(h_d3));
        cudaMemset(d_d3, 0, sizeof(h_d3));
        cudaMemcpy(d_w, h_w, sizeof(h_w), cudaMemcpyHostToDevice);
        {
            (void)0;
        }
        memset(&wreq, 0, sizeof(wreq));
        wreq.global_address = d_w;
        wreq.rows = 272u;
        wreq.columns = 64u;
        wreq.groups = 2u;
        wreq.box_rows = 136u;
        wreq.box_columns = 64u;
        wreq.element_bits = 8u;
        status = LmGemmTensorMapCached(&wmap, &wreq);
        if ( status != LM_TM_ENCODE_OK ) { printf("wmap encode %d\n", status); return 1; }
        {
            uint8_t *d_d3b = d_d3;
            (void)d_d3b;
        }
        Rank3ProbeKernel<<<1u, 1u>>>(&wmap, d_d3);
        err = cudaDeviceSynchronize();
        if ( err != cudaSuccess ) { printf("sync3 %d\n", (int)err); return 1; }
        cudaMemcpy(h_d3, d_d3, sizeof(h_d3), cudaMemcpyDeviceToHost);
        printf("rank-3 staged rows 15..17 first 8 bytes (plane = src_row%%4):\n");
        for ( uint32_t r = 15u; r <= 17u; ++r )
        {
            printf("row %u:", r);
            for ( uint32_t b = 0u; b < 8u; ++b )
                printf(" %3u", h_d3[r * 64u + b]);
            printf("\n");
        }
    }

    {
        uint16_t *d_a = 0, *d_d2 = 0;
        uint16_t h_a[4u * 128u], h_d2[4u * 128u];
        alignas(64) CUtensorMap amap;
        LmTensorMapRequest areq;
        for ( uint32_t r = 0u; r < 4u; ++r )
            for ( uint32_t k = 0u; k < 128u; ++k )
                h_a[r * 128u + k] = (uint16_t)(r * 128u + k);
        cudaMalloc(&d_a, sizeof(h_a));
        cudaMalloc(&d_d2, sizeof(h_d2));
        cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice);
        memset(&areq, 0, sizeof(areq));
        areq.global_address = d_a;
        areq.rows = 4u;
        areq.columns = 256u;
        areq.groups = 1u;
        areq.box_rows = 4u;
        areq.box_columns = 64u;
        areq.element_bits = 16u;
        status = LmGemmTensorMapCached(&amap, &areq);
        if ( status != LM_TM_ENCODE_OK ) { printf("amap encode %d\n", status); return 1; }
        TwoBoxProbeKernel<<<1u, 1u>>>(&amap, d_d2);
        err = cudaDeviceSynchronize();
        if ( err != cudaSuccess ) { printf("sync2 %d\n", (int)err); return 1; }
        cudaMemcpy(h_d2, d_d2, sizeof(h_d2), cudaMemcpyDeviceToHost);
        printf("two-box staged row 0: k 0..7, 60..67, 120..127 (value names global [row,k]):\n");
        printf("k 0..7:    ");
        for ( uint32_t k = 0u; k < 8u; ++k ) printf(" %u", h_d2[k]);
        printf("\nk 60..67:  ");
        for ( uint32_t k = 60u; k < 68u; ++k ) printf(" %u", h_d2[k]);
        printf("\nk 120..127:");
        for ( uint32_t k = 120u; k < 128u; ++k ) printf(" %u", h_d2[k]);
        printf("\n");
    }
    return 0;
}
