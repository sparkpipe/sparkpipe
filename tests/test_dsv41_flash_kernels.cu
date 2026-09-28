#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "modules/dsv41_flash_resident_decode_stage/source/spark_dsv41_flash_kernels.cuh"

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "FAIL line=%d cuda=%s call=%s\n", __LINE__, cudaGetErrorString(e), #call); \
    exit(1); } } while (0)

static std::vector<uint16_t> ReadWords(const char *path, size_t count)
{
    std::vector<uint16_t> words(count);
    FILE *file = fopen(path, "rb");
    if (file == 0 || fread(words.data(), sizeof(uint16_t), count, file) != count)
    {
        fprintf(stderr, "FAIL cannot read %zu bf16 words from %s\n", count, path);
        exit(1);
    }
    fclose(file);
    return words;
}

static void WriteWords(const char *path, const std::vector<uint16_t> &words)
{
    FILE *file = fopen(path, "wb");
    if (file == 0 || fwrite(words.data(), sizeof(uint16_t), words.size(), file) != words.size())
    {
        fprintf(stderr, "FAIL cannot write %s\n", path);
        exit(1);
    }
    fclose(file);
}

static uint16_t *Upload(const std::vector<uint16_t> &words)
{
    uint16_t *device = 0;
    CUDA(cudaMalloc(&device, words.size() * sizeof(uint16_t)));
    CUDA(cudaMemcpy(device, words.data(), words.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    return device;
}

static void Download(std::vector<uint16_t> &words, const uint16_t *device)
{
    CUDA(cudaMemcpy(words.data(), device, words.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
}

static int RunKvFp4(char **argv)
{
    uint32_t rows = (uint32_t)strtoul(argv[4], 0, 10);
    uint32_t width = (uint32_t)strtoul(argv[5], 0, 10);
    std::vector<uint16_t> data = ReadWords(argv[2], (size_t)rows * width);
    uint16_t *device = Upload(data);
    CUDA(SparkDsv41FlashLaunchKvFp4Qdq(0, device, width, rows, width));
    CUDA(cudaDeviceSynchronize());
    Download(data, device);
    WriteWords(argv[3], data);
    CUDA(cudaFree(device));
    if (SparkDsv41FlashLaunchKvFp4Qdq(0, device, width, rows, width + 1u) != cudaErrorInvalidValue)
    {
        fprintf(stderr, "FAIL width outside the fp4 group accepted\n");
        return 1;
    }
    return 0;
}

static int RunEngram(char **argv)
{
    uint32_t rows = (uint32_t)strtoul(argv[7], 0, 10);
    uint32_t hc = (uint32_t)strtoul(argv[8], 0, 10);
    uint32_t dimension = (uint32_t)strtoul(argv[9], 0, 10);
    float epsilon = strtof(argv[10], 0);
    std::vector<uint16_t> streams = ReadWords(argv[2], (size_t)rows * hc * dimension);
    std::vector<uint16_t> kv = ReadWords(argv[3], (size_t)rows * (hc + 1u) * dimension);
    std::vector<uint16_t> q_weight = ReadWords(argv[4], (size_t)hc * dimension);
    std::vector<uint16_t> k_weight = ReadWords(argv[5], (size_t)hc * dimension);
    uint16_t *streams_device = Upload(streams);
    uint16_t *kv_device = Upload(kv);
    uint16_t *q_device = Upload(q_weight);
    uint16_t *k_device = Upload(k_weight);
    CUDA(SparkDsv41FlashLaunchEngramGate(0, streams_device, kv_device, q_device, k_device, rows, hc, dimension, epsilon));
    CUDA(cudaDeviceSynchronize());
    Download(streams, streams_device);
    WriteWords(argv[6], streams);
    CUDA(cudaFree(streams_device));
    CUDA(cudaFree(kv_device));
    CUDA(cudaFree(q_device));
    CUDA(cudaFree(k_device));
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 6 && strcmp(argv[1], "kv-fp4") == 0)
        return RunKvFp4(argv);
    if (argc == 11 && strcmp(argv[1], "engram") == 0)
        return RunEngram(argv);
    fprintf(stderr, "usage: %s kv-fp4 IN OUT ROWS WIDTH | engram STREAMS KV QW KW OUT ROWS HC DIM EPS\n", argv[0]);
    return 2;
}
