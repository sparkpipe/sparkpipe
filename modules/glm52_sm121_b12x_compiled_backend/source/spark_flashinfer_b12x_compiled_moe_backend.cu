#include "sparkpipe/spark_glm52_sm121_b12x_generated_kernel_table.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

extern "C" SparkStatus SparkFlashInferB12xCompiledMoeCreate(
    const SparkGlm52Sm121FlashInferB12xMoeRecipe *recipe,
    void **state_out);

extern "C" SparkStatus SparkFlashInferB12xCompiledMoeLaunch(
    void *state_pointer,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments);

extern "C" void SparkFlashInferB12xCompiledMoeDestroy(void *state_pointer);

typedef struct SparkFlashInferB12xCompiledMoeState
{
    SparkGlm52Sm121FlashInferB12xMoeRecipe recipe;
    uint32_t bucket_count;
    SparkGlm52Sm121B12xGeneratedWorkspace *workspaces;
} SparkFlashInferB12xCompiledMoeState;

static size_t SparkGlm52B12xAlignUpSize(
    size_t value,
    size_t alignment)
{
    return ((value + alignment - 1u) / alignment) * alignment;
}

static SparkStatus SparkGlm52B12xCheckedMultiplySize(
    size_t left,
    size_t right,
    size_t *product_out)
{
    if (product_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (left != 0u && right > ((size_t)-1) / left)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *product_out = left * right;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52B12xCudaToSparkStatus(cudaError_t cuda_status)
{
    if (cuda_status == cudaSuccess)
    {
        return SPARK_STATUS_OK;
    }
    if (cuda_status == cudaErrorMemoryAllocation)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkGlm52B12xCudaMalloc(
    void **device_pointer_out,
    size_t byte_count)
{
    cudaError_t cuda_status;

    if (device_pointer_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *device_pointer_out = 0;
    if (byte_count == 0u)
    {
        return SPARK_STATUS_OK;
    }
    cuda_status = cudaMalloc(device_pointer_out, byte_count);
    return SparkGlm52B12xCudaToSparkStatus(cuda_status);
}

static void SparkGlm52B12xCudaFree(void *device_pointer)
{
    if (device_pointer != 0)
    {
        cudaFree(device_pointer);
    }
}

static SparkStatus SparkGlm52B12xAllocateDeviceArray(
    void **device_pointer_out,
    size_t element_count,
    size_t element_size)
{
    SparkStatus status;
    size_t byte_count;

    status = SparkGlm52B12xCheckedMultiplySize(
        element_count,
        element_size,
        &byte_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52B12xCudaMalloc(device_pointer_out, byte_count);
}

static void SparkGlm52B12xReleaseGeneratedWorkspace(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace)
{
    if (workspace == 0)
    {
        return;
    }
    SparkGlm52B12xCudaFree(workspace->row_counts_i32);
    SparkGlm52B12xCudaFree(workspace->token_map_i32);
    SparkGlm52B12xCudaFree(workspace->token_weights_fp32);
    SparkGlm52B12xCudaFree(workspace->packed_input_u8);
    SparkGlm52B12xCudaFree(workspace->packed_input_scale_u8);
    SparkGlm52B12xCudaFree(workspace->barrier_count_i32);
    SparkGlm52B12xCudaFree(workspace->barrier_epoch_i32);
    SparkGlm52B12xCudaFree(workspace->active_expert_count_i32);
    SparkGlm52B12xCudaFree(workspace->weight_expert_ids_i32);
    SparkGlm52B12xCudaFree(workspace->global_to_local_expert_i32);
    SparkGlm52B12xCudaFree(workspace->compact_topk_ids_i32);
    SparkGlm52B12xCudaFree(workspace->expert_write_rows_i32);
    SparkGlm52B12xCudaFree(workspace->expert_tile_base_i32);
    SparkGlm52B12xCudaFree(workspace->pair_head_i32);
    SparkGlm52B12xCudaFree(workspace->producers_done_count_i32);
    SparkGlm52B12xCudaFree(workspace->all_work_published_i32);
    SparkGlm52B12xCudaFree(workspace->task_head_i32);
    SparkGlm52B12xCudaFree(workspace->task_tail_i32);
    SparkGlm52B12xCudaFree(workspace->task_ready_i32);
    SparkGlm52B12xCudaFree(workspace->task_expert_i32);
    SparkGlm52B12xCudaFree(workspace->task_m_tile_i32);
    SparkGlm52B12xCudaFree(workspace->task_slice_begin_i32);
    SparkGlm52B12xCudaFree(workspace->task_slice_count_i32);
    SparkGlm52B12xCudaFree(workspace->task_valid_rows_i32);
    SparkGlm52B12xCudaFree(workspace->tile_write_count_i32);
    memset(workspace, 0, sizeof(*workspace));
}

static SparkStatus SparkGlm52B12xInitializeIdentityMap(
    void *device_pointer,
    uint32_t element_count)
{
    int32_t *host_values;
    uint32_t index;
    cudaError_t cuda_status;

    if (device_pointer == 0 || element_count == 0u)
    {
        return SPARK_STATUS_OK;
    }

    host_values = (int32_t *)malloc(sizeof(*host_values) * element_count);
    if (host_values == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    for (index = 0u; index < element_count; ++index)
    {
        host_values[index] = (int32_t)index;
    }

    cuda_status = cudaMemcpy(
        device_pointer,
        host_values,
        sizeof(*host_values) * element_count,
        cudaMemcpyHostToDevice);
    free(host_values);
    return SparkGlm52B12xCudaToSparkStatus(cuda_status);
}

static SparkStatus SparkGlm52B12xAllocateCommonControlWorkspace(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    uint32_t expert_count)
{
    SparkStatus status;

    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->row_counts_i32,
        expert_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->barrier_count_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->barrier_epoch_i32,
        1u,
        sizeof(int32_t));
    return status;
}

static SparkStatus SparkGlm52B12xAllocateStaticWorkspace(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket)
{
    SparkStatus status;
    size_t expert_count;
    size_t max_rows;
    size_t hidden_dimension;
    size_t rows_pad_k;
    size_t cols_pad_k;
    size_t expert_rows;
    size_t packed_input_elements;
    size_t packed_scale_elements;
    size_t compact_count;

    expert_count = SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT;
    hidden_dimension = SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_HIDDEN_DIMENSION;
    max_rows = bucket->max_rows;
    rows_pad_k = SparkGlm52B12xAlignUpSize(max_rows, 128u);
    cols_pad_k = SparkGlm52B12xAlignUpSize(
        hidden_dimension / SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_NVFP4_GROUP_SIZE,
        4u);

    status = SparkGlm52B12xCheckedMultiplySize(
        expert_count,
        max_rows,
        &expert_rows);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52B12xCheckedMultiplySize(
        expert_rows,
        hidden_dimension / 2u,
        &packed_input_elements);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52B12xCheckedMultiplySize(
        expert_count,
        rows_pad_k,
        &packed_scale_elements);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xCheckedMultiplySize(
        packed_scale_elements,
        cols_pad_k,
        &packed_scale_elements);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    compact_count = expert_count;
    if (compact_count < max_rows)
    {
        compact_count = max_rows;
    }

    status = SparkGlm52B12xAllocateCommonControlWorkspace(workspace, expert_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->token_map_i32,
        expert_rows,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->token_weights_fp32,
        expert_rows,
        sizeof(float));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->packed_input_u8,
        packed_input_elements,
        sizeof(uint8_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->packed_input_scale_u8,
        packed_scale_elements,
        sizeof(uint8_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->active_expert_count_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->weight_expert_ids_i32,
        expert_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->global_to_local_expert_i32,
        expert_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->compact_topk_ids_i32,
        compact_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52B12xInitializeIdentityMap(
        workspace->weight_expert_ids_i32,
        (uint32_t)expert_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52B12xInitializeIdentityMap(
        workspace->global_to_local_expert_i32,
        (uint32_t)expert_count);
}

static SparkStatus SparkGlm52B12xAllocateDynamicWorkspace(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket)
{
    SparkStatus status;
    size_t expert_count;
    size_t max_rows;
    size_t hidden_dimension;
    size_t cols_pad_k;
    size_t packed_input_elements;
    size_t packed_scale_elements;
    size_t task_count;

    expert_count = SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT;
    hidden_dimension = SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_HIDDEN_DIMENSION;
    max_rows = bucket->max_rows;
    task_count = bucket->task_capacity;
    cols_pad_k = SparkGlm52B12xAlignUpSize(
        hidden_dimension / SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_NVFP4_GROUP_SIZE,
        4u);

    status = SparkGlm52B12xCheckedMultiplySize(
        max_rows,
        hidden_dimension / 2u,
        &packed_input_elements);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xCheckedMultiplySize(
        max_rows,
        cols_pad_k,
        &packed_scale_elements);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52B12xAllocateCommonControlWorkspace(workspace, expert_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->token_map_i32,
        max_rows,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->token_weights_fp32,
        max_rows,
        sizeof(float));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->packed_input_u8,
        packed_input_elements,
        sizeof(uint8_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->packed_input_scale_u8,
        packed_scale_elements,
        sizeof(uint8_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->expert_write_rows_i32,
        expert_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->expert_tile_base_i32,
        expert_count + 1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->pair_head_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->producers_done_count_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->all_work_published_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_head_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_tail_i32,
        1u,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_ready_i32,
        task_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_expert_i32,
        task_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_m_tile_i32,
        task_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_slice_begin_i32,
        task_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_slice_count_i32,
        task_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xAllocateDeviceArray(
        &workspace->task_valid_rows_i32,
        task_count,
        sizeof(int32_t));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52B12xAllocateDeviceArray(
        &workspace->tile_write_count_i32,
        bucket->physical_tile_capacity,
        sizeof(int32_t));
}

static SparkStatus SparkGlm52B12xAllocateGeneratedWorkspace(
    SparkGlm52Sm121B12xGeneratedWorkspace *workspace,
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket)
{
    SparkStatus status;

    if (workspace == 0 || bucket == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(workspace, 0, sizeof(*workspace));
    workspace->abi_version = SPARK_GLM52_SM121_B12X_GENERATED_KERNEL_TABLE_ABI_VERSION;
    workspace->backend_kind = bucket->backend_kind;
    workspace->routed_rows_capacity = bucket->routed_rows_capacity;
    workspace->max_rows = bucket->max_rows;
    workspace->physical_tile_capacity = bucket->physical_tile_capacity;
    workspace->task_capacity = bucket->task_capacity;

    if (bucket->backend_kind == SPARK_GLM52_SM121_B12X_BACKEND_KIND_MICRO ||
        bucket->backend_kind == SPARK_GLM52_SM121_B12X_BACKEND_KIND_STATIC)
    {
        status = SparkGlm52B12xAllocateStaticWorkspace(workspace, bucket);
    }
    else if (bucket->backend_kind == SPARK_GLM52_SM121_B12X_BACKEND_KIND_DYNAMIC)
    {
        status = SparkGlm52B12xAllocateDynamicWorkspace(workspace, bucket);
    }
    else
    {
        status = SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52B12xReleaseGeneratedWorkspace(workspace);
    }
    return status;
}

static SparkStatus SparkGlm52B12xValidateRecipeAgainstGeneratedManifest(
    const SparkGlm52Sm121FlashInferB12xMoeRecipe *recipe)
{
    const SparkGlm52Sm121B12xGeneratedManifest *manifest;

    if (recipe == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    manifest = &SparkGlm52Sm121B12xGeneratedManifestInstance;
    if (manifest->abi_version !=
        SPARK_GLM52_SM121_B12X_GENERATED_KERNEL_TABLE_ABI_VERSION ||
        manifest->bucket_count == 0u ||
        manifest->buckets == 0)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (manifest->hidden_dimension != recipe->hidden_dimension ||
        manifest->intermediate_dimension != recipe->intermediate_dimension ||
        manifest->expert_count != recipe->expert_count ||
        manifest->top_k != recipe->top_k ||
        manifest->maximum_token_count < recipe->maximum_token_count ||
        manifest->cuda_architecture != recipe->cuda_architecture)
    {
        return SPARK_STATUS_TARGET_MISMATCH;
    }
    if (manifest->manifest_hash_low64 != recipe->kernel_manifest_hash_low64)
    {
        return SPARK_STATUS_HASH_MISMATCH;
    }
    return SPARK_STATUS_OK;
}

static const SparkGlm52Sm121B12xGeneratedKernelBucket *SparkGlm52B12xSelectBucket(
    const SparkFlashInferB12xCompiledMoeState *state,
    uint32_t token_count,
    uint32_t *bucket_index_out)
{
    const SparkGlm52Sm121B12xGeneratedManifest *manifest;
    uint32_t bucket_index;

    if (state == 0 || bucket_index_out == 0)
    {
        return 0;
    }

    manifest = &SparkGlm52Sm121B12xGeneratedManifestInstance;
    for (bucket_index = 0u; bucket_index < manifest->bucket_count; ++bucket_index)
    {
        if (token_count == manifest->buckets[bucket_index].token_upper_bound)
        {
            *bucket_index_out = bucket_index;
            return &manifest->buckets[bucket_index];
        }
    }
    return 0;
}

extern "C" SparkStatus SparkFlashInferB12xCompiledMoeCreate(
    const SparkGlm52Sm121FlashInferB12xMoeRecipe *recipe,
    void **state_out)
{
    SparkFlashInferB12xCompiledMoeState *state;
    SparkStatus status;
    uint32_t bucket_index;

    if (state_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *state_out = 0;

    status = SparkGlm52B12xValidateRecipeAgainstGeneratedManifest(recipe);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    state = (SparkFlashInferB12xCompiledMoeState *)calloc(1u, sizeof(*state));
    if (state == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    state->recipe = *recipe;
    state->bucket_count = SparkGlm52Sm121B12xGeneratedManifestInstance.bucket_count;
    state->workspaces = (SparkGlm52Sm121B12xGeneratedWorkspace *)calloc(
        state->bucket_count,
        sizeof(*state->workspaces));
    if (state->workspaces == 0)
    {
        free(state);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    for (bucket_index = 0u; bucket_index < state->bucket_count; ++bucket_index)
    {
        status = SparkGlm52B12xAllocateGeneratedWorkspace(
            &state->workspaces[bucket_index],
            &SparkGlm52Sm121B12xGeneratedManifestInstance.buckets[bucket_index]);
        if (status != SPARK_STATUS_OK)
        {
            SparkFlashInferB12xCompiledMoeDestroy(state);
            return status;
        }
    }

    *state_out = state;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkFlashInferB12xCompiledMoeLaunch(
    void *state_pointer,
    const SparkGlm52Sm121FlashInferB12xMoeArguments *arguments)
{
    SparkFlashInferB12xCompiledMoeState *state;
    SparkGlm52Sm121B12xGeneratedLaunchArguments generated_arguments;
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket;
    uint32_t bucket_index;

    if (state_pointer == 0 || arguments == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    state = (SparkFlashInferB12xCompiledMoeState *)state_pointer;
    if (arguments->token_count > state->recipe.maximum_token_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    bucket = SparkGlm52B12xSelectBucket(
        state,
        arguments->token_count,
        &bucket_index);
    if (bucket == 0 || bucket_index >= state->bucket_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    memset(&generated_arguments, 0, sizeof(generated_arguments));
    generated_arguments.abi_version =
        SPARK_GLM52_SM121_B12X_GENERATED_KERNEL_TABLE_ABI_VERSION;
    generated_arguments.token_count = arguments->token_count;
    generated_arguments.maximum_token_count = arguments->maximum_token_count;
    generated_arguments.expert_count = arguments->expert_count;
    generated_arguments.top_k = arguments->top_k;
    generated_arguments.hidden_dimension = arguments->hidden_dimension;
    generated_arguments.intermediate_dimension = arguments->intermediate_dimension;
    generated_arguments.hidden_bf16 = arguments->hidden_bf16;
    generated_arguments.topk_ids_i32 = arguments->topk_ids_i32;
    generated_arguments.topk_weights_fp32 = arguments->topk_weights_fp32;
    generated_arguments.w1_weight_fp4_static_view = arguments->w1_weight_fp4_static_view;
    generated_arguments.w1_scale_static_storage_ue4m3 = arguments->w1_scale_static_storage_ue4m3;
    generated_arguments.w1_alpha_fp32_by_expert = arguments->w1_alpha_fp32_by_expert;
    generated_arguments.fc2_input_scale_fp32_by_expert = arguments->fc2_input_scale_fp32_by_expert;
    generated_arguments.w2_weight_fp4_static_view = arguments->w2_weight_fp4_static_view;
    generated_arguments.w2_scale_static_storage_ue4m3 = arguments->w2_scale_static_storage_ue4m3;
    generated_arguments.w2_alpha_fp32_by_expert = arguments->w2_alpha_fp32_by_expert;
    generated_arguments.output_bf16 = arguments->output_bf16;
    generated_arguments.user_workspace = arguments->workspace;
    generated_arguments.user_workspace_bytes = arguments->workspace_bytes;
    generated_arguments.generated_workspace = &state->workspaces[bucket_index];
    generated_arguments.cuda_stream = arguments->cuda_stream;

    return SparkGlm52Sm121B12xGeneratedLaunch(bucket, &generated_arguments);
}

extern "C" void SparkFlashInferB12xCompiledMoeDestroy(void *state_pointer)
{
    SparkFlashInferB12xCompiledMoeState *state;
    uint32_t bucket_index;

    if (state_pointer == 0)
    {
        return;
    }

    state = (SparkFlashInferB12xCompiledMoeState *)state_pointer;
    if (state->workspaces != 0)
    {
        for (bucket_index = 0u; bucket_index < state->bucket_count; ++bucket_index)
        {
            SparkGlm52B12xReleaseGeneratedWorkspace(&state->workspaces[bucket_index]);
        }
        free(state->workspaces);
    }
    free(state);
}
