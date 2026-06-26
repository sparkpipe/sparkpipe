#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_cuda_activation_kernels.h"
#include "sparkpipe/spark_cuda_fp4_gemm.h"
#include "sparkpipe/spark_cuda_fp4_quant.h"
#include "sparkpipe/spark_glm_final_logits.h"
#include "sparkpipe/spark_glm_nvfp4_expert_cuda.h"

#define SPARKPIPE_GLM_NVFP4_PROJECTION_CACHE_ENTRIES 384u
#define SPARKPIPE_GLM_NVFP4_PROJECTION_CACHE_DEFAULT_MB 2048ull

typedef struct SparkGlmNvfp4ProjectionCuda
{
    const SparkGlmStageTensorBinding *weight;
    const SparkGlmStageTensorBinding *weight_scale;
    const SparkGlmStageTensorBinding *weight_scale_2;
    const SparkGlmStageTensorBinding *input_scale;
    uint32_t n;
    uint32_t k;
} SparkGlmNvfp4ProjectionCuda;

typedef struct SparkGlmNvfp4ExpertCudaRuntime
{
    uint16_t *host_padded_input;
    uint16_t *host_padded_output;
    uint8_t *host_weight;
    uint8_t *host_weight_scale;
    uint64_t host_padded_input_bytes;
    uint64_t host_padded_output_bytes;
    uint64_t host_weight_bytes;
    uint64_t host_weight_scale_bytes;
    void *device_activation;
    void *device_input_fp4;
    void *device_input_scale;
    void *device_weight;
    void *device_weight_scale;
    void *device_gate_up;
    void *device_intermediate;
    void *device_output;
    void *workspace;
    uint64_t device_activation_bytes;
    uint64_t device_input_fp4_bytes;
    uint64_t device_input_scale_bytes;
    uint64_t device_weight_bytes;
    uint64_t device_weight_scale_bytes;
    uint64_t device_gate_up_bytes;
    uint64_t device_intermediate_bytes;
    uint64_t device_output_bytes;
    uint64_t workspace_bytes;
    SparkCudaFp4GemmPlan hidden_to_intermediate_plan;
    SparkCudaFp4GemmPlan intermediate_to_hidden_plan;
} SparkGlmNvfp4ExpertCudaRuntime;

typedef struct SparkGlmNvfp4ProjectionCacheEntry
{
    char weight_path[SPARKPIPE_GLM_STAGE_EXECUTOR_PATH_BYTES];
    char weight_scale_path[SPARKPIPE_GLM_STAGE_EXECUTOR_PATH_BYTES];
    uint64_t weight_offset_bytes;
    uint64_t weight_size_bytes;
    uint64_t weight_scale_offset_bytes;
    uint64_t weight_scale_size_bytes;
    void *device_weight;
    void *device_weight_scale;
    uint64_t cached_bytes;
    uint64_t last_used_tick;
    bool ready;
} SparkGlmNvfp4ProjectionCacheEntry;

typedef struct SparkGlmNvfp4ProjectionCache
{
    SparkGlmNvfp4ProjectionCacheEntry entries[SPARKPIPE_GLM_NVFP4_PROJECTION_CACHE_ENTRIES];
    void *arena;
    uint64_t arena_used_bytes;
    uint64_t arena_capacity_bytes;
    uint64_t cached_bytes;
    uint64_t limit_bytes;
    uint64_t tick;
    bool initialized;
} SparkGlmNvfp4ProjectionCache;

static SparkGlmNvfp4ExpertCudaRuntime SparkGlmNvfp4ExpertRuntime;
static SparkGlmNvfp4ProjectionCache SparkGlmNvfp4ExpertProjectionCache;

static uint64_t SparkGlmNvfp4MaxU64(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

static uint64_t SparkGlmNvfp4AlignU64(uint64_t value, uint64_t alignment)
{
    if (alignment == 0u)
        return value;
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static uint64_t SparkGlmNvfp4ProjectionCacheLimitBytes(void)
{
    const char *value;
    char *end_pointer;
    unsigned long long mb_value;

    value = getenv("SPARKPIPE_GLM_NVFP4_CACHE_MB");
    if (value == 0 || value[0] == '\0')
        return SPARKPIPE_GLM_NVFP4_PROJECTION_CACHE_DEFAULT_MB * 1024ull * 1024ull;
    mb_value = strtoull(value, &end_pointer, 10);
    if (end_pointer == value || *end_pointer != '\0')
        return SPARKPIPE_GLM_NVFP4_PROJECTION_CACHE_DEFAULT_MB * 1024ull * 1024ull;
    return (uint64_t)mb_value * 1024ull * 1024ull;
}

static void SparkGlmNvfp4ProjectionCacheInit(SparkGlmNvfp4ProjectionCache *cache)
{
    if (cache == 0 || cache->initialized)
        return;
    memset(cache, 0, sizeof(*cache));
    cache->limit_bytes = SparkGlmNvfp4ProjectionCacheLimitBytes();
    cache->initialized = true;
}

static bool SparkGlmNvfp4ProjectionCacheEntryMatches(const SparkGlmNvfp4ProjectionCacheEntry *entry, const SparkGlmNvfp4ProjectionCuda *projection)
{
    if (entry == 0 || projection == 0 || !entry->ready)
        return false;
    if (entry->weight_offset_bytes != projection->weight->offset_bytes || entry->weight_size_bytes != projection->weight->size_bytes)
        return false;
    if (entry->weight_scale_offset_bytes != projection->weight_scale->offset_bytes || entry->weight_scale_size_bytes != projection->weight_scale->size_bytes)
        return false;
    if (strcmp(entry->weight_path, projection->weight->path) != 0)
        return false;
    if (strcmp(entry->weight_scale_path, projection->weight_scale->path) != 0)
        return false;
    return true;
}

static SparkGlmNvfp4ProjectionCacheEntry *SparkGlmNvfp4ProjectionCacheLookup(SparkGlmNvfp4ProjectionCache *cache, const SparkGlmNvfp4ProjectionCuda *projection)
{
    uint32_t entry_index;

    SparkGlmNvfp4ProjectionCacheInit(cache);
    if (cache == 0 || projection == 0 || cache->limit_bytes == 0u)
        return 0;
    for (entry_index = 0u; entry_index < SPARKPIPE_GLM_NVFP4_PROJECTION_CACHE_ENTRIES; ++entry_index)
    {
        if (SparkGlmNvfp4ProjectionCacheEntryMatches(&cache->entries[entry_index], projection))
        {
            cache->tick += 1u;
            cache->entries[entry_index].last_used_tick = cache->tick;
            return &cache->entries[entry_index];
        }
    }
    return 0;
}

static void SparkGlmNvfp4ProjectionCacheFreeEntry(SparkGlmNvfp4ProjectionCache *cache, SparkGlmNvfp4ProjectionCacheEntry *entry)
{
    if (cache == 0 || entry == 0 || !entry->ready)
        return;
    if (cache->cached_bytes >= entry->cached_bytes)
        cache->cached_bytes -= entry->cached_bytes;
    else
        cache->cached_bytes = 0u;
    memset(entry, 0, sizeof(*entry));
}

static SparkGlmNvfp4ProjectionCacheEntry *SparkGlmNvfp4ProjectionCacheSelectFreeEntry(SparkGlmNvfp4ProjectionCache *cache)
{
    uint32_t entry_index;

    if (cache == 0)
        return 0;
    for (entry_index = 0u; entry_index < SPARKPIPE_GLM_NVFP4_PROJECTION_CACHE_ENTRIES; ++entry_index)
    {
        if (!cache->entries[entry_index].ready)
            return &cache->entries[entry_index];
    }
    return 0;
}

static SparkStatus SparkGlmNvfp4ProjectionCacheEnsureArena(SparkGlmNvfp4ProjectionCache *cache)
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
    cache->cached_bytes = 0u;
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

static SparkStatus SparkGlmNvfp4ProjectionCacheStore(SparkGlmNvfp4ProjectionCache *cache, const SparkGlmNvfp4ProjectionCuda *projection, const void *device_weight, const void *device_weight_scale, SparkGlmNvfp4ExpertCudaReport *report)
{
    SparkGlmNvfp4ProjectionCacheEntry *entry;
    uint8_t *arena_base;
    uint64_t weight_offset;
    uint64_t scale_offset;
    uint64_t end_offset;
    cudaError_t cuda_status;

    SparkGlmNvfp4ProjectionCacheInit(cache);
    if (cache == 0 || projection == 0 || device_weight == 0 || device_weight_scale == 0 || cache->limit_bytes == 0u)
        return SPARK_STATUS_OK;
    if (projection->weight->size_bytes == 0u || projection->weight_scale->size_bytes == 0u)
        return SPARK_STATUS_OK;
    if (SparkGlmNvfp4ProjectionCacheEnsureArena(cache) != SPARK_STATUS_OK || cache->arena == 0)
        return SPARK_STATUS_OK;
    weight_offset = SparkGlmNvfp4AlignU64(cache->arena_used_bytes, 256u);
    scale_offset = SparkGlmNvfp4AlignU64(weight_offset + projection->weight->size_bytes, 256u);
    end_offset = SparkGlmNvfp4AlignU64(scale_offset + projection->weight_scale->size_bytes, 256u);
    if (end_offset > cache->limit_bytes)
        return SPARK_STATUS_OK;
    entry = SparkGlmNvfp4ProjectionCacheSelectFreeEntry(cache);
    if (entry == 0)
        return SPARK_STATUS_OK;
    if (entry->ready)
    {
        SparkGlmNvfp4ProjectionCacheFreeEntry(cache, entry);
        if (report != 0)
            report->projection_cache_eviction_count += 1u;
    }
    arena_base = (uint8_t *)cache->arena;
    entry->device_weight = arena_base + weight_offset;
    entry->device_weight_scale = arena_base + scale_offset;
    cuda_status = cudaMemcpy(entry->device_weight, device_weight, (size_t)projection->weight->size_bytes, cudaMemcpyDeviceToDevice);
    if (cuda_status == cudaSuccess)
        cuda_status = cudaMemcpy(entry->device_weight_scale, device_weight_scale, (size_t)projection->weight_scale->size_bytes, cudaMemcpyDeviceToDevice);
    if (cuda_status != cudaSuccess)
    {
        memset(entry, 0, sizeof(*entry));
        return SPARK_STATUS_OK;
    }
    snprintf(entry->weight_path, sizeof(entry->weight_path), "%s", projection->weight->path);
    snprintf(entry->weight_scale_path, sizeof(entry->weight_scale_path), "%s", projection->weight_scale->path);
    entry->weight_offset_bytes = projection->weight->offset_bytes;
    entry->weight_size_bytes = projection->weight->size_bytes;
    entry->weight_scale_offset_bytes = projection->weight_scale->offset_bytes;
    entry->weight_scale_size_bytes = projection->weight_scale->size_bytes;
    entry->cached_bytes = end_offset - weight_offset;
    cache->tick += 1u;
    entry->last_used_tick = cache->tick;
    entry->ready = true;
    cache->arena_used_bytes = end_offset;
    cache->cached_bytes += entry->cached_bytes;
    if (report != 0)
        report->projection_cache_store_count += 1u;
    return SPARK_STATUS_OK;
}

static void SparkGlmNvfp4DestroyRuntimePlans(SparkGlmNvfp4ExpertCudaRuntime *runtime)
{
    if (runtime == 0)
        return;
    if (runtime->hidden_to_intermediate_plan.initialized != 0u)
        (void)SparkDestroyCudaFp4GemmPlan(&runtime->hidden_to_intermediate_plan);
    if (runtime->intermediate_to_hidden_plan.initialized != 0u)
        (void)SparkDestroyCudaFp4GemmPlan(&runtime->intermediate_to_hidden_plan);
}

static SparkStatus SparkGlmNvfp4EnsureHostBuffer(void **buffer, uint64_t *capacity_bytes, uint64_t required_bytes)
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

static SparkStatus SparkGlmNvfp4EnsureDeviceBuffer(void **buffer, uint64_t *capacity_bytes, uint64_t required_bytes, uint32_t *allocation_count, bool *reallocated)
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

static SparkStatus SparkGlmNvfp4EnsureRuntime(SparkGlmNvfp4ExpertCudaRuntime *runtime, uint64_t activation_bytes, uint64_t input_packed_bytes, uint64_t input_scale_bytes, uint64_t max_weight_bytes, uint64_t max_weight_scale_bytes, uint64_t gate_up_bytes, uint64_t intermediate_bytes, uint64_t output_bytes, uint32_t *allocation_count)
{
    SparkStatus status;
    bool plan_pointer_reallocated;

    if (runtime == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    plan_pointer_reallocated = false;
    status = SparkGlmNvfp4EnsureHostBuffer((void **)&runtime->host_padded_input, &runtime->host_padded_input_bytes, activation_bytes);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmNvfp4EnsureHostBuffer((void **)&runtime->host_padded_output, &runtime->host_padded_output_bytes, output_bytes);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmNvfp4EnsureHostBuffer((void **)&runtime->host_weight, &runtime->host_weight_bytes, max_weight_bytes);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmNvfp4EnsureHostBuffer((void **)&runtime->host_weight_scale, &runtime->host_weight_scale_bytes, max_weight_scale_bytes);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmNvfp4EnsureDeviceBuffer(&runtime->device_activation, &runtime->device_activation_bytes, activation_bytes, allocation_count, 0);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmNvfp4EnsureDeviceBuffer(&runtime->device_input_fp4, &runtime->device_input_fp4_bytes, input_packed_bytes, allocation_count, 0);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmNvfp4EnsureDeviceBuffer(&runtime->device_input_scale, &runtime->device_input_scale_bytes, input_scale_bytes, allocation_count, &plan_pointer_reallocated);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmNvfp4EnsureDeviceBuffer(&runtime->device_weight, &runtime->device_weight_bytes, max_weight_bytes, allocation_count, 0);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmNvfp4EnsureDeviceBuffer(&runtime->device_weight_scale, &runtime->device_weight_scale_bytes, max_weight_scale_bytes, allocation_count, &plan_pointer_reallocated);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmNvfp4EnsureDeviceBuffer(&runtime->device_gate_up, &runtime->device_gate_up_bytes, gate_up_bytes, allocation_count, 0);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmNvfp4EnsureDeviceBuffer(&runtime->device_intermediate, &runtime->device_intermediate_bytes, intermediate_bytes, allocation_count, 0);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmNvfp4EnsureDeviceBuffer(&runtime->device_output, &runtime->device_output_bytes, output_bytes, allocation_count, 0);
    if (status == SPARK_STATUS_OK)
        status = SparkGlmNvfp4EnsureDeviceBuffer(&runtime->workspace, &runtime->workspace_bytes, SPARKPIPE_GLM_NVFP4_EXPERT_CUDA_WORKSPACE_BYTES, allocation_count, &plan_pointer_reallocated);
    if (plan_pointer_reallocated)
        SparkGlmNvfp4DestroyRuntimePlans(runtime);
    return status;
}

static SparkStatus SparkGlmNvfp4EnsureProjectionPlan(SparkCudaFp4GemmPlan *plan, uint32_t row_count, uint32_t n, uint32_t k, const void *device_input_scale, const void *device_weight_scale, void *workspace)
{
    if (plan == 0 || device_input_scale == 0 || device_weight_scale == 0 || workspace == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (plan->initialized != 0u && plan->sentinel == SPARKPIPE_CUDA_FP4_GEMM_SENTINEL && plan->m == row_count && plan->n == n && plan->k == k && plan->a_scale_pointer == (uint64_t)(uintptr_t)device_input_scale && plan->b_scale_pointer == (uint64_t)(uintptr_t)device_weight_scale && plan->workspace == workspace)
        return SPARK_STATUS_OK;
    if (plan->initialized != 0u)
        (void)SparkDestroyCudaFp4GemmPlan(plan);
    return SparkInitCudaFp4GemmPlan(plan, row_count, n, k, device_input_scale, device_weight_scale, workspace, SPARKPIPE_GLM_NVFP4_EXPERT_CUDA_WORKSPACE_BYTES, 0u);
}

static SparkStatus SparkGlmNvfp4ReadBindingBytes(const SparkGlmStageTensorBinding *binding, void *values, uint64_t value_bytes)
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

static SparkStatus SparkGlmNvfp4ReadBindingF32(const SparkGlmStageTensorBinding *binding, float *value)
{
    return SparkGlmNvfp4ReadBindingBytes(binding, value, sizeof(*value));
}

static SparkStatus SparkGlmNvfp4RunProjectionCuda(const SparkGlmNvfp4ProjectionCuda *projection, SparkCudaFp4GemmPlan *gemm_plan, const void *device_input_bf16, uint32_t row_count, void *device_input_fp4, void *device_input_scale, void *device_weight, void *device_weight_scale, void *workspace, uint8_t *host_weight, uint8_t *host_weight_scale, void *device_output_bf16, SparkCudaFp4QuantReport *quant_report, SparkCudaFp4GemmReport *gemm_report, SparkGlmNvfp4ExpertCudaReport *expert_report)
{
    SparkCudaFp4QuantRequest quant_request;
    SparkGlmNvfp4ProjectionCacheEntry *cache_entry;
    SparkStatus status;
    cudaError_t cuda_status;
    float input_scale;
    float weight_scale_2;
    float alpha;

    if (projection == 0 || gemm_plan == 0 || device_input_bf16 == 0 || device_input_fp4 == 0 || device_input_scale == 0 || device_weight == 0 || device_weight_scale == 0 || workspace == 0 || host_weight == 0 || host_weight_scale == 0 || device_output_bf16 == 0 || quant_report == 0 || gemm_report == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    memset(quant_report, 0, sizeof(*quant_report));
    memset(gemm_report, 0, sizeof(*gemm_report));
    status = SparkGlmNvfp4ReadBindingF32(projection->input_scale, &input_scale);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlmNvfp4ReadBindingF32(projection->weight_scale_2, &weight_scale_2);
    if (status != SPARK_STATUS_OK)
        return status;
    if (input_scale <= 0.0f || weight_scale_2 <= 0.0f)
        return SPARK_STATUS_TENSOR_MANIFEST_ERROR;
    cache_entry = SparkGlmNvfp4ProjectionCacheLookup(&SparkGlmNvfp4ExpertProjectionCache, projection);
    if (cache_entry != 0)
    {
        if (expert_report != 0)
            expert_report->projection_cache_hit_count += 1u;
        cuda_status = cudaMemcpy(device_weight, cache_entry->device_weight, (size_t)projection->weight->size_bytes, cudaMemcpyDeviceToDevice);
        if (cuda_status == cudaSuccess)
            cuda_status = cudaMemcpy(device_weight_scale, cache_entry->device_weight_scale, (size_t)projection->weight_scale->size_bytes, cudaMemcpyDeviceToDevice);
    }
    else
    {
        if (expert_report != 0)
            expert_report->projection_cache_miss_count += 1u;
        status = SparkGlmNvfp4ReadBindingBytes(projection->weight, host_weight, projection->weight->size_bytes);
        if (status == SPARK_STATUS_OK)
            status = SparkGlmNvfp4ReadBindingBytes(projection->weight_scale, host_weight_scale, projection->weight_scale->size_bytes);
        if (status != SPARK_STATUS_OK)
            return status;
        cuda_status = cudaMemcpy(device_weight, host_weight, (size_t)projection->weight->size_bytes, cudaMemcpyHostToDevice);
        if (cuda_status == cudaSuccess)
            cuda_status = cudaMemcpy(device_weight_scale, host_weight_scale, (size_t)projection->weight_scale->size_bytes, cudaMemcpyHostToDevice);
        if (cuda_status == cudaSuccess)
            (void)SparkGlmNvfp4ProjectionCacheStore(&SparkGlmNvfp4ExpertProjectionCache, projection, device_weight, device_weight_scale, expert_report);
    }
    if (cuda_status != cudaSuccess)
        return SPARK_STATUS_INTERNAL_ERROR;
    memset(&quant_request, 0, sizeof(quant_request));
    quant_request.row_count = row_count;
    quant_request.col_count = projection->k;
    quant_request.global_scale = 1.0f / input_scale;
    quant_request.sentinel = SPARKPIPE_CUDA_FP4_QUANT_SENTINEL;
    status = SparkRunCudaBf16ToFp4E2m1(&quant_request, device_input_bf16, device_input_fp4, device_input_scale, quant_report);
    if (status != SPARK_STATUS_OK)
        return status;
    status = SparkGlmNvfp4EnsureProjectionPlan(gemm_plan, row_count, projection->n, projection->k, device_input_scale, device_weight_scale, workspace);
    if (status != SPARK_STATUS_OK)
        return status;
    alpha = input_scale * weight_scale_2;
    return SparkRunCudaFp4GemmPlan(gemm_plan, device_input_fp4, device_weight, alpha, device_output_bf16, gemm_report);
}

static void SparkGlmNvfp4FillReportShape(const SparkGlmNvfp4ExpertCudaRequest *request, SparkGlmNvfp4ExpertCudaReport *report)
{
    report->row_count = request->row_count;
    report->compute_row_count = (request->row_count + 15u) & ~15u;
    report->hidden_size = request->hidden_size;
    report->intermediate_rows = request->intermediate_rows;
    report->input_element_count = (uint64_t)request->row_count * (uint64_t)request->hidden_size;
    report->intermediate_element_count = (uint64_t)request->row_count * (uint64_t)request->intermediate_rows;
    report->output_element_count = (uint64_t)request->row_count * (uint64_t)request->hidden_size;
}

extern "C" SparkStatus SparkRunGlmNvfp4ExpertCuda(const SparkGlmNvfp4ExpertCudaRequest *request, const SparkGlmStageNvfp4ExpertDescriptor *descriptor, const uint16_t *input_bf16, uint16_t *output_bf16, SparkGlmNvfp4ExpertCudaReport *report)
{
    SparkGlmNvfp4ProjectionCuda gate_projection,up_projection,down_projection;
    SparkCudaFp4QuantReport gate_quant_report,up_quant_report,down_quant_report;
    SparkCudaFp4GemmReport gate_gemm_report,up_gemm_report,down_gemm_report;
    SparkCudaActivationRequest activation_request;
    SparkCudaActivationReport activation_report;
    SparkGlmNvfp4ExpertCudaRuntime *runtime;
    uint64_t compute_row_count,activation_values,activation_bytes,hidden_packed_bytes,hidden_scale_bytes,intermediate_packed_bytes,intermediate_scale_bytes,input_packed_bytes,input_scale_bytes,max_weight_bytes,max_weight_scale_bytes,intermediate_values,intermediate_bytes,gate_up_bytes,output_values,output_bytes;
    SparkStatus status;
    cudaError_t cuda_status;

    if (report != 0)
        memset(report, 0, sizeof(*report));
    status = SparkValidateGlmNvfp4ExpertCudaRequest(request, descriptor);
    if (status != SPARK_STATUS_OK)
        return status;
    if (input_bf16 == 0 || output_bf16 == 0 || report == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    SparkGlmNvfp4FillReportShape(request, report);
    compute_row_count = report->compute_row_count;
    activation_values = compute_row_count * (uint64_t)request->hidden_size;
    activation_bytes = activation_values * sizeof(uint16_t);
    hidden_packed_bytes = SparkCudaFp4QuantPackedBytes((uint32_t)compute_row_count, request->hidden_size);
    hidden_scale_bytes = SparkCudaFp4QuantScaleBytes((uint32_t)compute_row_count, request->hidden_size);
    intermediate_packed_bytes = SparkCudaFp4QuantPackedBytes((uint32_t)compute_row_count, request->intermediate_rows);
    intermediate_scale_bytes = SparkCudaFp4QuantScaleBytes((uint32_t)compute_row_count, request->intermediate_rows);
    input_packed_bytes = SparkGlmNvfp4MaxU64(hidden_packed_bytes, intermediate_packed_bytes);
    input_scale_bytes = SparkGlmNvfp4MaxU64(hidden_scale_bytes, intermediate_scale_bytes);
    max_weight_bytes = SparkGlmNvfp4MaxU64(descriptor->gate_weight.size_bytes, SparkGlmNvfp4MaxU64(descriptor->up_weight.size_bytes, descriptor->down_weight.size_bytes));
    max_weight_scale_bytes = SparkGlmNvfp4MaxU64(descriptor->gate_weight_scale.size_bytes, SparkGlmNvfp4MaxU64(descriptor->up_weight_scale.size_bytes, descriptor->down_weight_scale.size_bytes));
    intermediate_values = compute_row_count * (uint64_t)request->intermediate_rows;
    intermediate_bytes = intermediate_values * sizeof(uint16_t);
    gate_up_bytes = 2u * intermediate_bytes;
    output_values = compute_row_count * (uint64_t)request->hidden_size;
    output_bytes = output_values * sizeof(uint16_t);
    runtime = &SparkGlmNvfp4ExpertRuntime;
    status = SparkGlmNvfp4EnsureRuntime(runtime, activation_bytes, input_packed_bytes, input_scale_bytes, max_weight_bytes, max_weight_scale_bytes, gate_up_bytes, intermediate_bytes, output_bytes, &report->device_allocation_count);
    if (status == SPARK_STATUS_OK)
    {
        memset(runtime->host_padded_input, 0, (size_t)activation_bytes);
        if (request->row_count == 1u && compute_row_count > 1u)
        {
            uint64_t row_index;

            for (row_index = 0u; row_index < compute_row_count; ++row_index)
                memcpy(&runtime->host_padded_input[row_index * (uint64_t)request->hidden_size], input_bf16, (size_t)request->hidden_size * sizeof(uint16_t));
        }
        else
        {
            memcpy(runtime->host_padded_input, input_bf16, (size_t)report->input_element_count * sizeof(uint16_t));
        }
        cuda_status = cudaMemcpy(runtime->device_activation, runtime->host_padded_input, (size_t)activation_bytes, cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess)
            status = SPARK_STATUS_INTERNAL_ERROR;
    }
    if (status == SPARK_STATUS_OK)
    {
        gate_projection = (SparkGlmNvfp4ProjectionCuda){&descriptor->gate_weight, &descriptor->gate_weight_scale, &descriptor->gate_weight_scale_2, &descriptor->gate_input_scale, request->intermediate_rows, request->hidden_size};
        status = SparkGlmNvfp4RunProjectionCuda(&gate_projection, &runtime->hidden_to_intermediate_plan, runtime->device_activation, (uint32_t)compute_row_count, runtime->device_input_fp4, runtime->device_input_scale, runtime->device_weight, runtime->device_weight_scale, runtime->workspace, runtime->host_weight, runtime->host_weight_scale, runtime->device_gate_up, &gate_quant_report, &gate_gemm_report, report);
        report->phase_code = 2u;
    }
    if (status == SPARK_STATUS_OK)
    {
        up_projection = (SparkGlmNvfp4ProjectionCuda){&descriptor->up_weight, &descriptor->up_weight_scale, &descriptor->up_weight_scale_2, &descriptor->up_input_scale, request->intermediate_rows, request->hidden_size};
        status = SparkGlmNvfp4RunProjectionCuda(&up_projection, &runtime->hidden_to_intermediate_plan, runtime->device_activation, (uint32_t)compute_row_count, runtime->device_input_fp4, runtime->device_input_scale, runtime->device_weight, runtime->device_weight_scale, runtime->workspace, runtime->host_weight, runtime->host_weight_scale, (uint8_t *)runtime->device_gate_up + intermediate_bytes, &up_quant_report, &up_gemm_report, report);
        report->phase_code = 4u;
    }
    if (status == SPARK_STATUS_OK)
    {
        memset(&activation_request, 0, sizeof(activation_request));
        activation_request.row_count = (uint32_t)compute_row_count;
        activation_request.hidden_size = request->intermediate_rows;
        activation_request.input_stride = request->intermediate_rows * 2u;
        activation_request.output_stride = request->intermediate_rows;
        activation_request.activation_kind = SPARK_CUDA_GATED_ACTIVATION_SILU;
        activation_request.sentinel = SPARKPIPE_CUDA_ACTIVATION_SENTINEL;
        status = SparkRunCudaGatedActivationBf16(&activation_request, runtime->device_gate_up, runtime->device_intermediate, &activation_report);
        report->phase_code = 5u;
    }
    if (status == SPARK_STATUS_OK)
    {
        down_projection = (SparkGlmNvfp4ProjectionCuda){&descriptor->down_weight, &descriptor->down_weight_scale, &descriptor->down_weight_scale_2, &descriptor->down_input_scale, request->hidden_size, request->intermediate_rows};
        status = SparkGlmNvfp4RunProjectionCuda(&down_projection, &runtime->intermediate_to_hidden_plan, runtime->device_intermediate, (uint32_t)compute_row_count, runtime->device_input_fp4, runtime->device_input_scale, runtime->device_weight, runtime->device_weight_scale, runtime->workspace, runtime->host_weight, runtime->host_weight_scale, runtime->device_output, &down_quant_report, &down_gemm_report, report);
        report->phase_code = 7u;
    }
    if (status == SPARK_STATUS_OK && cudaDeviceSynchronize() != cudaSuccess)
        status = SPARK_STATUS_INTERNAL_ERROR;
    if (status == SPARK_STATUS_OK && cudaMemcpy(runtime->host_padded_output, runtime->device_output, (size_t)output_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        status = SPARK_STATUS_INTERNAL_ERROR;
    if (status == SPARK_STATUS_OK)
    {
        memcpy(output_bf16, runtime->host_padded_output, (size_t)report->output_element_count * sizeof(uint16_t));
        report->gate_quant_kernel_count = gate_quant_report.quant_kernel_count;
        report->gate_gemm_run_count = gate_gemm_report.run_count;
        report->up_quant_kernel_count = up_quant_report.quant_kernel_count;
        report->up_gemm_run_count = up_gemm_report.run_count;
        report->activation_kernel_count = activation_report.gated_activation_kernel_count;
        report->down_quant_kernel_count = down_quant_report.quant_kernel_count;
        report->down_gemm_run_count = down_gemm_report.run_count;
        report->output_checksum = SparkComputeGlmBf16HostChecksum(output_bf16, report->output_element_count);
        report->phase_code = 8u;
    }
    return status;
}
