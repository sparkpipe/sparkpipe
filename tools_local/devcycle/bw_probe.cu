#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void ReadKernel(const float4 *data, float *sink, uint64_t elements)
{
    float acc = 0.0f;
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    for (; i < elements; i += stride)
    {
        float4 v = data[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f)
        sink[0] = acc;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: bw_probe BYTES\n");
        return 2;
    }
    uint64_t bytes = strtoull(argv[1], 0, 10);
    if (bytes < 64ull * 1024 * 1024 || bytes > 16ull * 1024 * 1024 * 1024)
        return 2;
    float4 *data = 0;
    float *sink = 0;
    if (cudaMalloc(&data, bytes) != cudaSuccess ||
        cudaMalloc(&sink, 4) != cudaSuccess)
    {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    cudaMemset(data, 0x3f, bytes);
    uint64_t elements = bytes / sizeof(float4);
    int blocks = 1024, threads = 256;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    ReadKernel<<<blocks, threads>>>(data, sink, elements);
    cudaDeviceSynchronize();
    const int iterations = 8;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++)
        ReadKernel<<<blocks, threads>>>(data, sink, elements);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double total = (double)bytes * iterations;
    double gbps = total / (ms / 1000.0) / 1e9;
    printf("bytes=%llu iterations=%d elapsed_ms=%.3f bandwidth_gbps=%.1f\n",
           (unsigned long long)bytes, iterations, ms, gbps);
    return 0;
}
