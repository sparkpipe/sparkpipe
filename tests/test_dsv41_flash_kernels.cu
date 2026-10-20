#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/topk_exact.cuh"
#include "inference/kernels/index_shard.cuh"
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

static int RunCandidates(char **argv)
{
    uint32_t rows = (uint32_t)strtoul(argv[4], 0, 10);
    uint32_t stride = (uint32_t)strtoul(argv[5], 0, 10);
    uint32_t block_size = (uint32_t)strtoul(argv[6], 0, 10);
    uint32_t topk_blocks = (uint32_t)strtoul(argv[7], 0, 10);
    uint32_t block_stride = (stride + block_size - 1u) / block_size;
    std::vector<float> scores((size_t)rows * stride);
    std::vector<uint32_t> widths(rows);
    FILE *file = fopen(argv[2], "rb");
    if (file == 0 || fread(widths.data(), sizeof(uint32_t), rows, file) != rows ||
        fread(scores.data(), sizeof(float), scores.size(), file) != scores.size())
    {
        fprintf(stderr, "FAIL cannot read %s\n", argv[2]);
        return 1;
    }
    fclose(file);
    float *scores_device = 0;
    float *blocks_device = 0;
    uint32_t *widths_device = 0;
    CUDA(cudaMalloc(&scores_device, scores.size() * sizeof(float)));
    CUDA(cudaMalloc(&blocks_device, (size_t)rows * block_stride * sizeof(float)));
    CUDA(cudaMalloc(&widths_device, rows * sizeof(uint32_t)));
    CUDA(cudaMemcpy(scores_device, scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(widths_device, widths.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA(SparkDsv41FlashLaunchCandidateMask(0, scores_device, widths_device, stride, blocks_device, block_stride, rows, block_size, topk_blocks));
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(scores.data(), scores_device, scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
    file = fopen(argv[3], "wb");
    if (file == 0 || fwrite(scores.data(), sizeof(float), scores.size(), file) != scores.size())
    {
        fprintf(stderr, "FAIL cannot write %s\n", argv[3]);
        return 1;
    }
    fclose(file);
    CUDA(cudaFree(scores_device));
    CUDA(cudaFree(blocks_device));
    CUDA(cudaFree(widths_device));
    return 0;
}

static int RunShardCandidates(char **argv)
{
    const uint32_t rows = (uint32_t)strtoul(argv[4], 0, 10), positions = (uint32_t)strtoul(argv[5], 0, 10);
    const uint32_t degree = 16u, grain = 8u, topk_blocks = 2048u;
    const SparkKvShard probe = { degree, 0u, grain };
    const uint32_t local_stride = ((SparkKvShardLocalKeys(probe, positions) + 63u) / 64u) * 64u;
    const uint32_t local_blocks = ((local_stride / grain + 63u) / 64u) * 64u;
    const uint32_t global_blocks = (((positions + grain - 1u) / grain + 63u) / 64u) * 64u;
    const uint32_t keep = local_blocks < topk_blocks ? local_blocks : topk_blocks;
    const uint32_t global_keep = global_blocks < topk_blocks ? global_blocks : topk_blocks;
    const uint64_t local_entries = (uint64_t)rows * LmTopkExactCandidateEntries(local_blocks, keep, LM_TOPK_EXACT_CHUNK);
    const uint64_t global_entries = (uint64_t)rows * LmTopkExactCandidateEntries(global_blocks, global_keep, LM_TOPK_EXACT_CHUNK);
    const uint64_t entries = 2u * (local_entries > global_entries ? local_entries : global_entries) + 2u;
    std::vector<uint32_t> reach(rows), row_position(rows), sequence(rows, 0u), length(1, positions);
    std::vector<float> scores((size_t)rows * positions), output((size_t)rows * positions, -INFINITY), local((size_t)rows * local_stride);
    FILE *file = fopen(argv[2], "rb");
    if (file == 0 || fread(reach.data(), sizeof(uint32_t), rows, file) != rows ||
        fread(scores.data(), sizeof(float), scores.size(), file) != scores.size())
    {
        fprintf(stderr, "FAIL cannot read %s\n", argv[2]);
        return 1;
    }
    fclose(file);
    for (uint32_t row = 0u; row < rows; ++row)
        row_position[row] = reach[row] != 0u ? reach[row] - 1u : UINT32_MAX;
    float *local_device[16], *block_scores = 0, *full = 0, *values = 0;
    uint8_t *mask_device[16];
    uint32_t *local_selected = 0, *selected = 0, *scratch_positions = 0, *sequence_device = 0, *length_device = 0, *row_position_device = 0;
    uint2 *gathered = 0;
    CUDA(cudaMalloc(&block_scores, (size_t)rows * local_blocks * sizeof(float)));
    CUDA(cudaMalloc(&full, (size_t)rows * global_blocks * sizeof(float)));
    CUDA(cudaMalloc(&values, entries * sizeof(float)));
    CUDA(cudaMalloc(&scratch_positions, entries * sizeof(uint32_t)));
    CUDA(cudaMalloc(&local_selected, (size_t)rows * keep * sizeof(uint32_t)));
    CUDA(cudaMalloc(&selected, (size_t)rows * global_keep * sizeof(uint32_t)));
    CUDA(cudaMalloc(&gathered, (size_t)degree * rows * keep * sizeof(uint2)));
    CUDA(cudaMalloc(&sequence_device, rows * sizeof(uint32_t)));
    CUDA(cudaMalloc(&length_device, sizeof(uint32_t)));
    CUDA(cudaMalloc(&row_position_device, rows * sizeof(uint32_t)));
    CUDA(cudaMemcpy(sequence_device, sequence.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(length_device, length.data(), sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(row_position_device, row_position.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice));
    for (uint32_t rank = 0u; rank < degree; ++rank)
    {
        const SparkKvShard shard = { degree, rank, grain }, blocks = { degree, rank, 1u };
        const uint32_t owned = SparkKvShardLocalKeys(shard, positions);
        std::fill(local.begin(), local.end(), -INFINITY);
        for (uint32_t row = 0u; row < rows; ++row)
            for (uint32_t entry = 0u; entry < owned; ++entry)
            {
                const uint32_t position = SparkKvShardLocalPosition(shard, entry);
                local[(size_t)row * local_stride + entry] = position < reach[row] ? scores[(size_t)row * positions + position] : -INFINITY;
            }
        CUDA(cudaMalloc(&local_device[rank], local.size() * sizeof(float)));
        CUDA(cudaMalloc(&mask_device[rank], (size_t)rows * local_blocks));
        CUDA(cudaMemcpy(local_device[rank], local.data(), local.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA(SparkDsv41FlashLaunchShardBlockMax(0, local_device[rank], local_stride, shard, sequence_device, length_device, row_position_device,
            block_scores, local_blocks, rows));
        CUDA(LmTopkExactLaunch<256u>(block_scores, rows, local_blocks, keep, LM_TOPK_EXACT_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS, values,
            scratch_positions, entries, local_selected, 0));
        CUDA(LmIndexShardCandidatePackLaunch<256u>(blocks, block_scores, local_blocks, local_selected, keep, rows,
            gathered + (size_t)rank * rows * keep, 0));
    }
    CUDA(LmIndexShardCandidateScatterLaunch<256u>(gathered, (uint64_t)rows * keep, degree, keep, rows, global_blocks, full, 0));
    CUDA(LmTopkExactLaunch<256u>(full, rows, global_blocks, global_keep, LM_TOPK_EXACT_CHUNK, LM_TOPK_EXACT_CHUNKED_ROWS, values,
        scratch_positions, entries, selected, 0));
    for (uint32_t rank = 0u; rank < degree; ++rank)
    {
        const SparkKvShard shard = { degree, rank, grain }, blocks = { degree, rank, 1u };
        const uint32_t owned = SparkKvShardLocalKeys(shard, positions);
        CUDA(SparkDsv41FlashLaunchShardBlockMask(0, full, global_blocks, selected, global_keep, blocks, mask_device[rank], local_blocks, rows));
        CUDA(SparkDsv41FlashLaunchShardBlockApply(0, local_device[rank], local_stride, grain, mask_device[rank], local_blocks, rows));
        CUDA(cudaMemcpy(local.data(), local_device[rank], local.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (uint32_t row = 0u; row < rows; ++row)
            for (uint32_t entry = 0u; entry < owned; ++entry)
                output[(size_t)row * positions + SparkKvShardLocalPosition(shard, entry)] = local[(size_t)row * local_stride + entry];
        CUDA(cudaFree(local_device[rank]));
        CUDA(cudaFree(mask_device[rank]));
    }
    CUDA(cudaDeviceSynchronize());
    file = fopen(argv[3], "wb");
    if (file == 0 || fwrite(output.data(), sizeof(float), output.size(), file) != output.size())
    {
        fprintf(stderr, "FAIL cannot write %s\n", argv[3]);
        return 1;
    }
    fclose(file);
    CUDA(cudaFree(block_scores));
    CUDA(cudaFree(full));
    CUDA(cudaFree(values));
    CUDA(cudaFree(scratch_positions));
    CUDA(cudaFree(local_selected));
    CUDA(cudaFree(selected));
    CUDA(cudaFree(gathered));
    CUDA(cudaFree(sequence_device));
    CUDA(cudaFree(length_device));
    CUDA(cudaFree(row_position_device));
    return 0;
}

static int RunFp8Linear(char **argv)
{
    uint32_t rows = (uint32_t)strtoul(argv[6], 0, 10);
    uint32_t input_dimension = (uint32_t)strtoul(argv[7], 0, 10);
    uint32_t output_dimension = (uint32_t)strtoul(argv[8], 0, 10);
    size_t weight_bytes = (size_t)output_dimension * input_dimension;
    size_t scale_bytes = (size_t)(output_dimension / 32u) * (input_dimension / 32u);
    std::vector<uint8_t> weight(weight_bytes), scale(scale_bytes);
    std::vector<uint16_t> input = ReadWords(argv[4], (size_t)rows * input_dimension);
    std::vector<uint16_t> output((size_t)rows * output_dimension);
    FILE *file = fopen(argv[2], "rb");
    if (file == 0 || fread(weight.data(), 1u, weight_bytes, file) != weight_bytes)
    {
        fprintf(stderr, "FAIL cannot read %s\n", argv[2]);
        return 1;
    }
    fclose(file);
    file = fopen(argv[3], "rb");
    if (file == 0 || fread(scale.data(), 1u, scale_bytes, file) != scale_bytes)
    {
        fprintf(stderr, "FAIL cannot read %s\n", argv[3]);
        return 1;
    }
    fclose(file);
    uint8_t *weight_device = 0, *scale_device = 0;
    float *scratch_device = 0;
    uint16_t *input_device = Upload(input);
    uint16_t *output_device = Upload(output);
    CUDA(cudaMalloc(&weight_device, weight_bytes));
    CUDA(cudaMalloc(&scale_device, scale_bytes));
    CUDA(cudaMalloc(&scratch_device, (size_t)rows * input_dimension * sizeof(float)));
    CUDA(cudaMemcpy(weight_device, weight.data(), weight_bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(scale_device, scale.data(), scale_bytes, cudaMemcpyHostToDevice));
    CUDA(SparkDsv41FlashLaunchFp8BlockLinear(0, weight_device, scale_device, input_device, scratch_device, output_device, rows, input_dimension, output_dimension));
    CUDA(cudaDeviceSynchronize());
    Download(output, output_device);
    WriteWords(argv[5], output);
    CUDA(cudaFree(weight_device));
    CUDA(cudaFree(scale_device));
    CUDA(cudaFree(scratch_device));
    CUDA(cudaFree(input_device));
    CUDA(cudaFree(output_device));
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 9 && strcmp(argv[1], "fp8-linear") == 0)
        return RunFp8Linear(argv);
    if (argc == 8 && strcmp(argv[1], "candidates") == 0)
        return RunCandidates(argv);
    if (argc == 6 && strcmp(argv[1], "shard-candidates") == 0)
        return RunShardCandidates(argv);
    if (argc == 6 && strcmp(argv[1], "kv-fp4") == 0)
        return RunKvFp4(argv);
    if (argc == 11 && strcmp(argv[1], "engram") == 0)
        return RunEngram(argv);
    fprintf(stderr, "usage: %s kv-fp4 IN OUT ROWS WIDTH | engram STREAMS KV QW KW OUT ROWS HC DIM EPS | candidates IN OUT ROWS STRIDE BLOCK TOPK | shard-candidates IN OUT ROWS POSITIONS | fp8-linear W SCALE IN OUT ROWS K N\n", argv[0]);
    return 2;
}
