#include "spark_glm52_resident_decode_stage_backend.h"

#include <stdint.h>

#include <cuda_runtime_api.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"

static void CUDART_CB SparkGlm52ResidentDecodeStageCudaCompletion(
    void *completion_context)
{
    SparkGlm52ResidentDecodeStageBackendCompletion *completion;

    completion =
        (SparkGlm52ResidentDecodeStageBackendCompletion *)completion_context;
    if (completion != 0 && completion->function != 0)
    {
        completion->function(completion->context);
    }
}

extern "C" SparkStatus SparkGlm52ResidentDecodeStageBackendVerifyRequiredCudaModules(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    if (node_context == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkGlm52Sm121RequiredDecodeStageInitialize(node_context);
}

extern "C" SparkStatus SparkGlm52ResidentDecodeStageBackendSubmit(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    SparkGlm52ResidentDecodeStageBackendCompletion *completion)
{
    const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot;
    void *cuda_stream;
    SparkStatus status;
    cudaError_t cuda_status;

    if (node_context == 0 || completion == 0 || completion->function == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (node_context->pipeline_slots == 0 ||
        pipeline_slot_index >= node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    pipeline_slot = &node_context->pipeline_slots[pipeline_slot_index];
    cuda_stream = pipeline_slot->cuda_stream;
    if (cuda_stream == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52Sm121RequiredDecodeStageLaunch(
        node_context,
        pipeline_slot,
        pipeline_slot_index,
        active_sequence_count,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    cuda_status = cudaLaunchHostFunc(
        (cudaStream_t)cuda_stream,
        SparkGlm52ResidentDecodeStageCudaCompletion,
        completion);
    if (cuda_status != cudaSuccess)
    {
        cudaStreamSynchronize((cudaStream_t)cuda_stream);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

extern "C" void SparkGlm52ResidentDecodeStageBackendQuiesce(
    const SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    uint32_t pipeline_slot_index;

    if (node_context == 0 || node_context->pipeline_slots == 0)
    {
        return;
    }

    SparkGlm52Sm121RequiredDecodeStageQuiesce(node_context);
    for (pipeline_slot_index = 0u;
         pipeline_slot_index < node_context->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        if (node_context->pipeline_slots[pipeline_slot_index].cuda_stream != 0)
        {
            cudaStreamSynchronize((cudaStream_t)
                node_context->pipeline_slots[pipeline_slot_index].cuda_stream);
        }
    }
}
