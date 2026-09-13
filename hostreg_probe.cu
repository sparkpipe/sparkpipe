#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define BYTES (2ull * 1024ull * 1024ull * 1024ull)

__global__ void sum_probe(const float *source, float *destination, size_t count)
{
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float accumulator = 0.0f;
    for (size_t offset = index; offset < count; offset += (size_t)gridDim.x * blockDim.x)
        accumulator += source[offset];
    destination[index] = accumulator;
}

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

int main(void)
{
    int fd;
    void *mapping;
    float *device_out;
    float *host_out;
    double t0;
    cudaIpcMemHandle_t probe;

    fd = memfd_create("probe", 0u);
    if (fd < 0 || ftruncate(fd, BYTES) != 0)
    {
        printf("memfd failed\n");
        return 1;
    }
    mapping = mmap(0, BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED)
    {
        printf("mmap failed\n");
        return 1;
    }
    t0 = now_s();
    if (cudaHostRegister(mapping, BYTES, 0u) != cudaSuccess)
    {
        printf("register FAILED after %.2fs\n", now_s() - t0);
        return 1;
    }
    printf("register 2GB took %.2fs\n", now_s() - t0);
    memset(mapping, 0, BYTES);
    t0 = now_s();
    if (cudaMemcpyAsync((char *)mapping + 4096, mapping, 33554432u,
            cudaMemcpyHostToHost, 0) != cudaSuccess ||
        cudaStreamSynchronize(0) != cudaSuccess)
    {
        printf("stream copy failed\n");
        return 1;
    }
    printf("32MB stream-ordered H2H copy: %.1fms\n", (now_s() - t0) * 1e3);
    if (cudaMalloc(&device_out, 32768 * sizeof(float)) != cudaSuccess)
        return 1;
    host_out = (float *)mapping;
    for (size_t i = 0; i < 32768; i++)
        host_out[i] = 1.0f;
    t0 = now_s();
    sum_probe<<<32768 / 256, 256>>>((const float *)mapping, device_out,
        33554432u / 4u);
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        printf("kernel failed\n");
        return 1;
    }
    printf("kernel reading 32MB of host-registered memory: %.2fms\n",
        (now_s() - t0) * 1e3);
    (void)probe;
    printf("PROBE-PASS\n");
    return 0;
}
