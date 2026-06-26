#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_cuda_cublaslt_gemm.h"
#include "sparkpipe/spark_glm_bf16_linear_cuda.h"
#include "sparkpipe/spark_glm_final_logits.h"

#define SPARKPIPE_GLM_BF16_LINEAR_CACHE_ENTRIES 256u
#define SPARKPIPE_GLM_BF16_LINEAR_CACHE_DEFAULT_MB 2048ull

typedef struct SparkGlmBf16LinearCudaRuntime
{
    uint16_t *host_weight;
    uint64_t host_weight_bytes;
    void *device_input;
    void *device_weight;
    void *device_output;
    void *workspace;
    uint64_t device_input_bytes;
    uint64_t device_weight_bytes;
    uint64_t device_output_bytes;
    uint64_t workspace_bytes;
    SparkCudaCublasLtBf16GemmPlan gemm_plan;
} SparkGlmBf16LinearCudaRuntime;

typedef struct SparkGlmBf16LinearWeightCacheEntry
{
    char path[SPARKPIPE_GLM_BF16_LINEAR_CUDA_PATH_BYTES];
    uint64_t offset_bytes;
    uint64_t size_bytes;
    uint64_t weight_checksum;
    void *device_weight;
    uint64_t cached_bytes;
    uint64_t last_used_tick;
    bool ready;
} SparkGlmBf16LinearWeightCacheEntry;

typedef struct SparkGlmBf16LinearWeightCache
{
    SparkGlmBf16LinearWeightCacheEntry entries[SPARKPIPE_GLM_BF16_LINEAR_CACHE_ENTRIES];
    void *arena;
    uint64_t arena_used_bytes;
    uint64_t arena_capacity_bytes;
    uint64_t limit_bytes;
    uint64_t tick;
    bool initialized;
} SparkGlmBf16LinearWeightCache;

static SparkGlmBf16LinearCudaRuntime SparkGlmBf16LinearRuntime;
static SparkGlmBf16LinearWeightCache SparkGlmBf16LinearCache;

static uint64_t SparkGlmBf16LinearAlignU64(uint64_t value, uint64_t alignment)
{
    if (alignment == 0u)
        return value;
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static uint64_t SparkGlmBf16LinearCacheLimitBytes(void)
{
    const char *value;
    char *end_pointer;
    unsigned long long mb_value;

    value = getenv("SPARKPIPE_GLM_BF16_LINEAR_CACHE_MB");
    if (value == 0 || value[0] == '\0')
        return SPARKPIPE_GLM_BF16_LINEAR_CACHE_DEFAULT_MB * 1024ull * 1024ull;
    mb_value = strtoull(value, &end_pointer, 10);
    if (end_pointer == value || *end_pointer != '\0')
        return SPARKPIPE_GLM_BF16_LINEAR_CACHE_DEFAULT_MB * 1024ull * 1024ull;
    return (uint64_t)mb_value * 1024ull * 1024ull;
}

static void SparkGlmBf16LinearCacheInit(SparkGlmBf16LinearWeightCache *cache)
{
    if (cache == 0 || cache->initialized)
        return;
    memset(cache, 0, sizeof(*cache));
    cache->limit_bytes = SparkGlmBf16LinearCacheLimitBytes();
    cache->initialized = true;
}

static bool SparkGlmBf16LinearCacheEntryMatches(const SparkGlmBf16LinearWeightCacheEntry *entry, const SparkGlmBf16LinearCudaBinding *weight)
{
    if (entry == 0 || weight == 0 || !entry->ready)
        return false;
    if (entry->offset_bytes != weight->offset_bytes || entry->size_bytes != weight->size_bytes)
        return false;
    if (strcmp(entry->path, weight->path) != 0)
        return false;
    return true;
}

static SparkGlmBf16LinearWeightCacheEntry *SparkGlmBf16LinearCacheLookup(SparkGlmBf16LinearWeightCache *cache, const SparkGlmBf16LinearCudaBinding *weight)
{
    uint32_t entry_index;

    SparkGlmBf16LinearCacheInit(cache);
    if (cache == 0 || weight == 0 || cache->limit_bytes == 0u)
        return 0;
    for (entry_index = 0u; entry_index < SPARKPIPE_GLM_BF16_LINEAR_CACHE_ENTRIES; ++entry_index)
    {
        if (SparkGlmBf16LinearCacheEntryMatches(&cache->entries[entry_index], weight))
        {
            cache->tick += 1u;
            cache->entries[entry_index].last_used_tick = cache->tick;
            return &cache->entries[entry_index];
        }
    }
    return 0;
}

static SparkGlmBf16LinearWeightCacheEntry *SparkGlmBf16LinearCacheSelectFreeEntry(SparkGlmBf16LinearWeightCache *cache)
{
    uint32_t entry_index;

    if (cache == 0)
        return 0;
    for (entry_index = 0u; entry_index < SPARKPIPE_GLM_BF16_LINEAR_CACHE_ENTRIES; ++entry_index)
    {
        if (!cache->entries[entry_index].ready)
            return &cache->entries[entry_index];
    }
    return 0;
}

static SparkStatus SparkGlmBf16LinearCacheEnsureArena(SparkGlmBf16LinearWeightCache *cache)
{
    cudaError_t cuda_status;

    if (cache == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (cache->limit_bytes == 0u)
        return SPARK_STATUS_OK;
    if (cache->arena != 0 && cache->arena_capacity_bytes >= cache->limit_bytes)
        return SPARK_STATUS_OK;
    cudaFree(cache->arena);
    cache->arena = 0;
    cache->arena_capacity_bytes = 0u;
    cache->arena_used_bytes = 0u;
    memset(cache->entries, 0, sizeof(cache->entries));
    cuda_status = cudaMalloc(&cache->arena, (size_t)cache->limit_bytes);
    if (cuda_status != cudaSuccess)
    {
        cache->limit_bytes = 0u;
        return SPARK_STATUS_OK;
    }
    cache->arena_capacity_bytes = cache->limit_bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlmBf16LinearCacheStore(SparkGlmBf16LinearWeightCache *cache, const SparkGlmBf16LinearCudaBinding *weight, const void *device_weight, uint64_t weight_checksum, SparkGlmBf16LinearCudaReport *report)
{
    SparkGlmBf16LinearWeightCacheEntry *entry;
    uint8_t *arena_base;
    uint64_t weight_offset,end_offset;
    cudaError_t cuda_status;

    SparkGlmBf16LinearCacheInit(cache);
    if (cache == 0 || weight == 0 || device_weight == 0 || cache->limit_bytes == 0u)
        return SPARK_STATUS_OK;
    if (weight->size_bytes == 0u)
        return SPARK_STATUS_OK;
    if (SparkGlmBf16LinearCacheEnsureArena(cache) != SPARK_STATUS_OK || cache->arena == 0)
        return SPARK_STATUS_OK;
    weight_offset = SparkGlmBf16LinearAlignU64(cache->arena_used_bytes, 256u);
    end_offset = SparkGlmBf16LinearAlignU64(weight_offset + weight->size_bytes, 256u);
    if (end_offset > cache->limit_bytes)
        return SPARK_STATUS_OK;
    entry = SparkGlmBf16LinearCacheSelectFreeEntry(cache);
    if (entry == 0)
        return SPARK_STATUS_OK;
    arena_base = (uint8_t *)cache->arena;
    entry->device_weight = arena_base + weight_offset;
    cuda_status = cudaMemcpy(entry->device_weight, device_weight, (size_t)weight->size_bytes, cudaMemcpyDeviceToDevice);
    if (cuda_status != cudaSuccess)
    {
        memset(entry, 0, sizeof(*entry));
        return SPARK_STATUS_OK;
    }
    snprintf(entry->path, sizeof(entry->path), "%s", weight->path);
    entry->offset_bytes = weight->offset_bytes;
    entry->size_bytes = weight->size_bytes;
    entry->weight_checksum = weight_checksum;
    entry->cached_bytes = end_offset - weight_offset;
    cache->tick += 1u;
    entry->last_used_tick = cache->tick;
    entry->ready = true;
    cache->arena_used_bytes = end_offset;
    if (report != 0)
        report->weight_cache_store_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlmBf16LinearEnsureHostBuffer(void **buffer, uint64_t *capacity_bytes, uint64_t required_bytes)
{
    void *new_buffer;

    if (buffer == 0 || capacity_bytes == 0 || required_bytes == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (*capacity_bytes >= required_bytes && *buffer != 0)
        return SPARK_STATUS_OK;
    new_buffer = realloc(*buffer, (size_t)required_bytes);
    if (new_buffer == 0)
        return SPARK_STATUS_INTERNAL_ERROR;
    *buffer = new_buffer;
    *capacity_bytes = required_bytes;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlmBf16LinearEnsureDeviceBuffer(void **buffer, uint64_t *capacity_bytes, uint64_t required_bytes, uint32_t *allocation_count, bool *reallocated)
{
    cudaError_t cuda_status;

    if (buffer == 0 || capacity_bytes == 0 || required_bytes == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (*capacity_bytes >= required_bytes && *buffer != 0)
        return SPARK_STATUS_OK;
    cudaFree(*buffer);
    *buffer = 0;
    *capacity_bytes = 0u;
    cuda_status = cudaMalloc(buffer, (size_t)required_bytes);
    if (cuda_status != cudaSuccess)
        return SPARK_STATUS_INTERNAL_ERROR;
    *capacity_bytes = required_bytes;
    if (allocation_count != 0)
        *allocation_count += 1u;
    if (reallocated != 0)
        *reallocated = true;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlmBf16LinearEnsureRuntime(SparkGlmBf16LinearCudaRuntime *runtime, uint64_t input_bytes, uint64_t weight_bytes, uint64_t output_bytes, uint32_t *allocation_count)
{
    SparkStatus status;
    bool workspace_reallocated;

    if (runtime == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    workspace_reallocated = false;
    status = SparkGlmBf16LinearEnsureHostBuffer((void **)&runtime->host_weight, &runtime->host_weight_bytes, weight_bytes);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmBf16LinearEnsureDeviceBuffer(&runtime->device_input, &runtime->device_input_bytes, input_bytes, allocation_count, 0);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmBf16LinearEnsureDeviceBuffer(&runtime->device_weight, &runtime->device_weight_bytes, weight_bytes, allocation_count, 0);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmBf16LinearEnsureDeviceBuffer(&runtime->device_output, &runtime->device_output_bytes, output_bytes, allocation_count, 0);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmBf16LinearEnsureDeviceBuffer(&runtime->workspace, &runtime->workspace_bytes, SPARKPIPE_GLM_BF16_LINEAR_CUDA_WORKSPACE_BYTES, allocation_count, &workspace_reallocated);
    if (workspace_reallocated && runtime->gemm_plan.initialized != 0u)
        (void)SparkDestroyCudaCublasLtBf16GemmPlan(&runtime->gemm_plan);
    return status;
}

static SparkStatus SparkGlmBf16LinearEnsurePlan(SparkCudaCublasLtBf16GemmPlan *plan, uint32_t m, uint32_t n, uint32_t k, void *workspace)
{
    if (plan == 0 || workspace == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (plan->initialized != 0u && plan->sentinel == SPARKPIPE_CUDA_CUBLASLT_GEMM_SENTINEL && plan->m == m && plan->n == n && plan->k == k && plan->workspace == workspace)
        return SPARK_STATUS_OK;
    if (plan->initialized != 0u)
        (void)SparkDestroyCudaCublasLtBf16GemmPlan(plan);
    return SparkInitCudaCublasLtBf16GemmPlan(plan, m, n, k, workspace, SPARKPIPE_GLM_BF16_LINEAR_CUDA_WORKSPACE_BYTES, 0u);
}

static SparkStatus SparkGlmBf16LinearReadBindingBytes(const SparkGlmBf16LinearCudaBinding *binding, void *values, uint64_t value_bytes)
{
    FILE *file;

    if (binding == 0 || values == 0 || value_bytes == 0u || !binding->ready || binding->size_bytes < value_bytes)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (binding->offset_bytes > (uint64_t)LONG_MAX)
        return SPARK_STATUS_INVALID_ARGUMENT;
    file = fopen(binding->path, "rb");
    if (file == 0)
        return SPARK_STATUS_IO_ERROR;
    if (fseek(file, (long)binding->offset_bytes, SEEK_SET) != 0)
    {
        fclose(file);
        return SPARK_STATUS_IO_ERROR;
    }
    if (fread(values, 1u, (size_t)value_bytes, file) != value_bytes)
    {
        fclose(file);
        return SPARK_STATUS_IO_ERROR;
    }
    fclose(file);
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunGlmBf16LinearCuda(const SparkGlmBf16LinearCudaRequest *request, const SparkGlmBf16LinearCudaBinding *weight, const uint16_t *input_bf16, uint16_t *output_bf16, SparkGlmBf16LinearCudaReport *report)
{
    SparkCudaCublasLtBf16GemmReport gemm_report;
    SparkGlmBf16LinearCudaRuntime *runtime;
    SparkGlmBf16LinearWeightCacheEntry *cache_entry;
    void *device_weight_for_run;
    uint64_t input_bytes,weight_bytes,output_bytes;
    SparkStatus status;
    cudaError_t cuda_status;

    if (report != 0)
        memset(report, 0, sizeof(*report));
    status = SparkValidateGlmBf16LinearCudaRequest(request, weight);
    if (status != SPARK_STATUS_OK)
        return status;
    if (input_bf16 == 0 || output_bf16 == 0 || report == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    memset(&gemm_report, 0, sizeof(gemm_report));
    report->input_count = request->input_count;
    report->output_count = request->output_count;
    report->input_element_count = request->input_count;
    report->output_element_count = request->output_count;
    report->weight_element_count = (uint64_t)request->input_count * (uint64_t)request->output_count;
    input_bytes = (uint64_t)request->input_count * sizeof(uint16_t);
    weight_bytes = report->weight_element_count * sizeof(uint16_t);
    output_bytes = (uint64_t)request->output_count * sizeof(uint16_t);
    runtime = &SparkGlmBf16LinearRuntime;
    device_weight_for_run = 0;
    status = SparkGlmBf16LinearEnsureRuntime(runtime, input_bytes, weight_bytes, output_bytes, &report->device_allocation_count);
    if (status == SPARK_STATUS_OK)
    {
        cuda_status = cudaMemcpy(runtime->device_input, input_bf16, (size_t)input_bytes, cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess)
            status = SPARK_STATUS_INTERNAL_ERROR;
    }
    if (status == SPARK_STATUS_OK)
    {
        cache_entry = SparkGlmBf16LinearCacheLookup(&SparkGlmBf16LinearCache, weight);
        if (cache_entry != 0)
        {
            report->weight_cache_hit_count += 1u;
            report->weight_checksum = cache_entry->weight_checksum;
            device_weight_for_run = cache_entry->device_weight;
        }
        else
        {
            report->weight_cache_miss_count += 1u;
            status = SparkGlmBf16LinearReadBindingBytes(weight, runtime->host_weight, weight_bytes);
            if (status == SPARK_STATUS_OK)
            {
                report->weight_checksum = SparkComputeGlmBf16HostChecksum(runtime->host_weight, report->weight_element_count);
                cuda_status = cudaMemcpy(runtime->device_weight, runtime->host_weight, (size_t)weight_bytes, cudaMemcpyHostToDevice);
                if (cuda_status != cudaSuccess)
                    status = SPARK_STATUS_INTERNAL_ERROR;
            }
            if (status == SPARK_STATUS_OK)
            {
                (void)SparkGlmBf16LinearCacheStore(&SparkGlmBf16LinearCache, weight, runtime->device_weight, report->weight_checksum, report);
                device_weight_for_run = runtime->device_weight;
            }
        }
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlmBf16LinearEnsurePlan(&runtime->gemm_plan, 1u, request->output_count, request->input_count, runtime->workspace);
        report->phase_code = 2u;
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkRunCudaCublasLtBf16GemmPlan(&runtime->gemm_plan, runtime->device_input, device_weight_for_run, runtime->device_output, &gemm_report);
        report->phase_code = 3u;
    }
    if (status == SPARK_STATUS_OK && cudaDeviceSynchronize() != cudaSuccess)
        status = SPARK_STATUS_INTERNAL_ERROR;
    if (status == SPARK_STATUS_OK && cudaMemcpy(output_bf16, runtime->device_output, (size_t)output_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        status = SPARK_STATUS_INTERNAL_ERROR;
    if (status == SPARK_STATUS_OK)
    {
        report->gemm_run_count = gemm_report.run_count;
        report->output_checksum = SparkComputeGlmBf16HostChecksum(output_bf16, request->output_count);
        report->phase_code = 4u;
    }
    return status;
}
