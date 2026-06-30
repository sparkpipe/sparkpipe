#include "sparkpipe/spark_glm52_resident_decode_stage_b12x_moe_plan.h"

#include <cuda_runtime_api.h>

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static SparkStatus SparkGlm52B12xPlanCudaToSparkStatus(
    cudaError_t cuda_status)
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

static SparkStatus SparkGlm52B12xPlanCheckedAddU64(
    uint64_t left,
    uint64_t right,
    uint64_t *sum_out)
{
    if (sum_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (left > UINT64_MAX - right)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *sum_out = left + right;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52B12xPlanCheckedMultiplyU64(
    uint64_t left,
    uint64_t right,
    uint64_t *product_out)
{
    if (product_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (left != 0u && right > UINT64_MAX / left)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *product_out = left * right;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52B12xPlanReadExact(
    FILE *file,
    void *destination,
    size_t byte_count)
{
    if (file == 0 || destination == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (byte_count == 0u)
    {
        return SPARK_STATUS_OK;
    }
    return fread(destination, 1u, byte_count, file) == byte_count
        ? SPARK_STATUS_OK
        : SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkGlm52B12xPlanSeek(
    FILE *file,
    uint64_t file_offset)
{
    if (file == 0 || file_offset > (uint64_t)LONG_MAX)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return fseek(file, (long)file_offset, SEEK_SET) == 0
        ? SPARK_STATUS_OK
        : SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkGlm52B12xPlanFileSize(
    FILE *file,
    uint64_t *file_size_out)
{
    long original_offset;
    long end_offset;

    if (file == 0 || file_size_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    original_offset = ftell(file);
    if (original_offset < 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    if (fseek(file, 0, SEEK_END) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    end_offset = ftell(file);
    if (end_offset < 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    if (fseek(file, original_offset, SEEK_SET) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    *file_size_out = (uint64_t)end_offset;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52B12xPlanValidateRegion(
    const SparkGlm52ResidentDecodeStageB12xMoePackHeader *header,
    uint32_t region_index,
    uint64_t expected_bytes,
    uint64_t file_size)
{
    const SparkGlm52ResidentDecodeStageB12xMoePackRegion *region;
    uint64_t end_offset;
    SparkStatus status;

    if (header == 0 || region_index >=
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    region = &header->regions[region_index];
    status = SparkGlm52B12xPlanCheckedAddU64(
        region->offset,
        region->bytes,
        &end_offset);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (region->offset < header->header_bytes ||
        (region->offset %
         SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_ALIGNMENT) != 0u ||
        region->bytes != expected_bytes ||
        end_offset > file_size)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52B12xPlanValidatePackHeader(
    const SparkGlm52ResidentDecodeStageB12xMoePackHeader *header,
    const SparkGlm52ResidentDecodeStageB12xMoeResidentBindingCreateInfo *create_info,
    uint64_t file_size)
{
    uint64_t w1_weight_bytes;
    uint64_t w1_scale_bytes;
    uint64_t w2_weight_bytes;
    uint64_t w2_scale_bytes;
    uint64_t alpha_bytes;
    SparkStatus status;

    if (header == 0 || create_info == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (memcmp(
            header->magic,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_MAGIC,
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_MAGIC_BYTES) != 0 ||
        header->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_ABI_VERSION ||
        header->header_bytes !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_HEADER_BYTES ||
        header->layer_index != create_info->layer_index ||
        header->maximum_token_count < create_info->maximum_active_sequence_count ||
        header->hidden_dimension !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_HIDDEN_DIMENSION ||
        header->intermediate_dimension !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_INTERMEDIATE_DIMENSION ||
        header->expert_count !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_EXPERT_COUNT ||
        header->top_k != SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_TOP_K ||
        header->gate_up_order !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_GATE_UP_ORDER_UP_GATE ||
        header->weight_layout !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_WEIGHT_LAYOUT_FLASHINFER_STATIC_VIEW ||
        header->scale_layout !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_SCALE_LAYOUT_FLASHINFER_STATIC_STORAGE ||
        header->quant_mode !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_QUANT_MODE_NVFP4 ||
        header->output_dtype !=
            SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_OUTPUT_DTYPE_BF16 ||
        header->cuda_architecture != 121u ||
        header->reserved0 != 0u ||
        header->reserved1 != 0u ||
        header->qualified_maximum_microseconds == 0u ||
        header->qualification_record_hash_low64 == 0u ||
        header->kernel_manifest_hash_low64 == 0u ||
        file_size < header->header_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52B12xPlanCheckedMultiplyU64(
        (uint64_t)header->expert_count,
        (uint64_t)(2u * header->intermediate_dimension),
        &w1_weight_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xPlanCheckedMultiplyU64(
        w1_weight_bytes,
        (uint64_t)(header->hidden_dimension / 2u),
        &w1_weight_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52B12xPlanCheckedMultiplyU64(
        (uint64_t)header->expert_count,
        (uint64_t)(2u * header->intermediate_dimension),
        &w1_scale_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xPlanCheckedMultiplyU64(
        w1_scale_bytes,
        (uint64_t)(header->hidden_dimension /
                   SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_NVFP4_GROUP_SIZE),
        &w1_scale_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52B12xPlanCheckedMultiplyU64(
        (uint64_t)header->expert_count,
        (uint64_t)header->hidden_dimension,
        &w2_weight_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xPlanCheckedMultiplyU64(
        w2_weight_bytes,
        (uint64_t)(header->intermediate_dimension / 2u),
        &w2_weight_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkGlm52B12xPlanCheckedMultiplyU64(
        (uint64_t)header->expert_count,
        (uint64_t)header->hidden_dimension,
        &w2_scale_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xPlanCheckedMultiplyU64(
        w2_scale_bytes,
        (uint64_t)(header->intermediate_dimension /
                   SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_NVFP4_GROUP_SIZE),
        &w2_scale_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    alpha_bytes = (uint64_t)header->expert_count * sizeof(float);
    status = SparkGlm52B12xPlanValidateRegion(
        header,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W1_WEIGHT,
        w1_weight_bytes,
        file_size);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xPlanValidateRegion(
        header,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W1_SCALE,
        w1_scale_bytes,
        file_size);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xPlanValidateRegion(
        header,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W1_ALPHA,
        alpha_bytes,
        file_size);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xPlanValidateRegion(
        header,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_FC2_INPUT_SCALE,
        alpha_bytes,
        file_size);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xPlanValidateRegion(
        header,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W2_WEIGHT,
        w2_weight_bytes,
        file_size);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52B12xPlanValidateRegion(
        header,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W2_SCALE,
        w2_scale_bytes,
        file_size);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    return SparkGlm52B12xPlanValidateRegion(
        header,
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W2_ALPHA,
        alpha_bytes,
        file_size);
}

static SparkStatus SparkGlm52B12xPlanLoadRegionToDevice(
    FILE *file,
    const SparkGlm52ResidentDecodeStageB12xMoePackRegion *region,
    void **device_pointer_out)
{
    enum
    {
        SparkGlm52B12xPlanCopyChunkBytes = 64u * 1024u * 1024u
    };

    uint8_t *host_buffer;
    uint8_t *device_bytes;
    uint64_t copied_bytes;
    SparkStatus status;
    cudaError_t cuda_status;

    if (file == 0 || region == 0 || device_pointer_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *device_pointer_out = 0;
    if (region->bytes > (uint64_t)((size_t)-1))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    cuda_status = cudaMalloc(device_pointer_out, (size_t)region->bytes);
    status = SparkGlm52B12xPlanCudaToSparkStatus(cuda_status);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    host_buffer = (uint8_t *)malloc(SparkGlm52B12xPlanCopyChunkBytes);
    if (host_buffer == 0)
    {
        cudaFree(*device_pointer_out);
        *device_pointer_out = 0;
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    status = SparkGlm52B12xPlanSeek(file, region->offset);
    if (status != SPARK_STATUS_OK)
    {
        free(host_buffer);
        cudaFree(*device_pointer_out);
        *device_pointer_out = 0;
        return status;
    }

    device_bytes = (uint8_t *)(*device_pointer_out);
    copied_bytes = 0u;
    while (copied_bytes < region->bytes)
    {
        uint64_t remaining_bytes;
        size_t chunk_bytes;

        remaining_bytes = region->bytes - copied_bytes;
        chunk_bytes = remaining_bytes > SparkGlm52B12xPlanCopyChunkBytes
            ? (size_t)SparkGlm52B12xPlanCopyChunkBytes
            : (size_t)remaining_bytes;
        status = SparkGlm52B12xPlanReadExact(file, host_buffer, chunk_bytes);
        if (status != SPARK_STATUS_OK)
        {
            break;
        }
        cuda_status = cudaMemcpy(
            device_bytes + copied_bytes,
            host_buffer,
            chunk_bytes,
            cudaMemcpyHostToDevice);
        status = SparkGlm52B12xPlanCudaToSparkStatus(cuda_status);
        if (status != SPARK_STATUS_OK)
        {
            break;
        }
        copied_bytes += (uint64_t)chunk_bytes;
    }

    free(host_buffer);
    if (status != SPARK_STATUS_OK)
    {
        cudaFree(*device_pointer_out);
        *device_pointer_out = 0;
    }
    return status;
}

static void SparkGlm52B12xPlanFreeDevicePointer(
    void **device_pointer_cell)
{
    if (device_pointer_cell != 0 && *device_pointer_cell != 0)
    {
        cudaFree(*device_pointer_cell);
        *device_pointer_cell = 0;
    }
}

static void SparkGlm52B12xPlanPopulateBinding(
    SparkGlm52ResidentDecodeStageB12xMoeResidentBinding *binding,
    const SparkGlm52ResidentDecodeStageB12xMoePackHeader *header,
    uint32_t maximum_active_sequence_count)
{
    uint64_t maximum_route_count;

    maximum_route_count =
        (uint64_t)maximum_active_sequence_count * (uint64_t)header->top_k;

    binding->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PLAN_ABI_VERSION;
    binding->layer_index = header->layer_index;
    memset(&binding->dispatch_plan, 0, sizeof(binding->dispatch_plan));
    binding->dispatch_plan.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_ABI_VERSION;
    binding->dispatch_plan.plan_kind =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_KIND_FLASHINFER_B12X;
    binding->dispatch_plan.maximum_active_sequence_count =
        maximum_active_sequence_count;
    binding->dispatch_plan.maximum_route_count = maximum_route_count > UINT32_MAX
        ? UINT32_MAX
        : (uint32_t)maximum_route_count;
    binding->dispatch_plan.expert_count = header->expert_count;
    binding->dispatch_plan.top_k = header->top_k;
    binding->dispatch_plan.intermediate_dimension = header->intermediate_dimension;
    binding->dispatch_plan.opaque_state = &binding->plan;
    binding->dispatch_plan.validated_maximum_latency_ns =
        header->qualified_maximum_microseconds * 1000u;

    memset(&binding->plan, 0, sizeof(binding->plan));
    binding->plan.abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PLAN_ABI_VERSION;
    binding->plan.capability_flags =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_REQUIRED_CAPABILITIES;
    binding->plan.maximum_active_sequence_count = maximum_active_sequence_count;
    binding->plan.maximum_token_count = header->maximum_token_count;
    binding->plan.expert_count = header->expert_count;
    binding->plan.top_k = header->top_k;
    binding->plan.hidden_dimension = header->hidden_dimension;
    binding->plan.intermediate_dimension = header->intermediate_dimension;
    binding->plan.gate_up_order = header->gate_up_order;
    binding->plan.weight_layout = header->weight_layout;
    binding->plan.scale_layout = header->scale_layout;
    binding->plan.quant_mode = header->quant_mode;
    binding->plan.output_dtype = header->output_dtype;
    binding->plan.cuda_architecture = header->cuda_architecture;
    binding->plan.recipe.abi_version =
        SPARK_GLM52_SM121_FLASHINFER_B12X_MOE_ABI_VERSION;
    binding->plan.recipe.hidden_dimension = header->hidden_dimension;
    binding->plan.recipe.intermediate_dimension = header->intermediate_dimension;
    binding->plan.recipe.expert_count = header->expert_count;
    binding->plan.recipe.top_k = header->top_k;
    binding->plan.recipe.maximum_token_count = header->maximum_token_count;
    binding->plan.recipe.gate_up_order = header->gate_up_order;
    binding->plan.recipe.weight_layout = header->weight_layout;
    binding->plan.recipe.scale_layout = header->scale_layout;
    binding->plan.recipe.quant_mode = header->quant_mode;
    binding->plan.recipe.output_dtype = header->output_dtype;
    binding->plan.recipe.cuda_architecture = header->cuda_architecture;
    binding->plan.recipe.qualified_maximum_microseconds =
        header->qualified_maximum_microseconds;
    binding->plan.recipe.qualification_record_hash_low64 =
        header->qualification_record_hash_low64;
    binding->plan.recipe.kernel_manifest_hash_low64 =
        header->kernel_manifest_hash_low64;
    binding->plan.state_cell = &binding->state_cell;
    binding->plan.w1_weight_fp4_static_view = binding->w1_weight_fp4_static_view;
    binding->plan.w1_scale_static_storage_ue4m3 =
        binding->w1_scale_static_storage_ue4m3;
    binding->plan.w1_alpha_fp32_by_expert =
        (const float *)binding->w1_alpha_fp32_by_expert;
    binding->plan.fc2_input_scale_fp32_by_expert =
        (const float *)binding->fc2_input_scale_fp32_by_expert;
    binding->plan.w2_weight_fp4_static_view = binding->w2_weight_fp4_static_view;
    binding->plan.w2_scale_static_storage_ue4m3 =
        binding->w2_scale_static_storage_ue4m3;
    binding->plan.w2_alpha_fp32_by_expert =
        (const float *)binding->w2_alpha_fp32_by_expert;
    binding->plan.workspace = 0;
    binding->plan.workspace_bytes = 0u;
    binding->plan.validated_maximum_latency_ns =
        header->qualified_maximum_microseconds * 1000u;
}

SparkStatus SparkGlm52ResidentDecodeStageB12xMoeResidentBindingCreateFromPackFile(
    SparkGlm52ResidentDecodeStageB12xMoeResidentBinding *binding,
    const SparkGlm52ResidentDecodeStageB12xMoeResidentBindingCreateInfo *create_info)
{
    SparkGlm52ResidentDecodeStageB12xMoePackHeader header;
    FILE *file;
    uint64_t file_size;
    SparkStatus status;

    if (binding == 0 || create_info == 0 || create_info->pack_path == 0 ||
        create_info->abi_version !=
            SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PLAN_ABI_VERSION ||
        create_info->reserved != 0u ||
        create_info->maximum_active_sequence_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(binding, 0, sizeof(*binding));
    file = fopen(create_info->pack_path, "rb");
    if (file == 0)
    {
        return errno == ENOENT ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_IO_ERROR;
    }

    status = SparkGlm52B12xPlanReadExact(file, &header, sizeof(header));
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52B12xPlanFileSize(file, &file_size);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52B12xPlanValidatePackHeader(
            &header,
            create_info,
            file_size);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52B12xPlanLoadRegionToDevice(
            file,
            &header.regions[
                SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W1_WEIGHT],
            &binding->w1_weight_fp4_static_view);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52B12xPlanLoadRegionToDevice(
            file,
            &header.regions[
                SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W1_SCALE],
            &binding->w1_scale_static_storage_ue4m3);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52B12xPlanLoadRegionToDevice(
            file,
            &header.regions[
                SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W1_ALPHA],
            &binding->w1_alpha_fp32_by_expert);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52B12xPlanLoadRegionToDevice(
            file,
            &header.regions[
                SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_FC2_INPUT_SCALE],
            &binding->fc2_input_scale_fp32_by_expert);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52B12xPlanLoadRegionToDevice(
            file,
            &header.regions[
                SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W2_WEIGHT],
            &binding->w2_weight_fp4_static_view);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52B12xPlanLoadRegionToDevice(
            file,
            &header.regions[
                SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W2_SCALE],
            &binding->w2_scale_static_storage_ue4m3);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52B12xPlanLoadRegionToDevice(
            file,
            &header.regions[
                SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_PACK_REGION_W2_ALPHA],
            &binding->w2_alpha_fp32_by_expert);
    }
    fclose(file);
    file = 0;

    if (status == SPARK_STATUS_OK)
    {
        SparkGlm52B12xPlanPopulateBinding(
            binding,
            &header,
            create_info->maximum_active_sequence_count);
        status = SparkGlm52Sm121FlashInferB12xMoeCreate(
            &binding->plan.recipe,
            &binding->state_cell);
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkGlm52ResidentDecodeStageB12xMoeResidentBindingDestroy(binding);
        return status;
    }
    return SPARK_STATUS_OK;
}

void SparkGlm52ResidentDecodeStageB12xMoeResidentBindingDestroy(
    SparkGlm52ResidentDecodeStageB12xMoeResidentBinding *binding)
{
    if (binding == 0)
    {
        return;
    }
    if (binding->state_cell != 0)
    {
        SparkGlm52Sm121FlashInferB12xMoeDestroy(binding->state_cell);
        binding->state_cell = 0;
    }
    SparkGlm52B12xPlanFreeDevicePointer(&binding->w1_weight_fp4_static_view);
    SparkGlm52B12xPlanFreeDevicePointer(&binding->w1_scale_static_storage_ue4m3);
    SparkGlm52B12xPlanFreeDevicePointer(&binding->w1_alpha_fp32_by_expert);
    SparkGlm52B12xPlanFreeDevicePointer(&binding->fc2_input_scale_fp32_by_expert);
    SparkGlm52B12xPlanFreeDevicePointer(&binding->w2_weight_fp4_static_view);
    SparkGlm52B12xPlanFreeDevicePointer(&binding->w2_scale_static_storage_ue4m3);
    SparkGlm52B12xPlanFreeDevicePointer(&binding->w2_alpha_fp32_by_expert);
    memset(binding, 0, sizeof(*binding));
}
