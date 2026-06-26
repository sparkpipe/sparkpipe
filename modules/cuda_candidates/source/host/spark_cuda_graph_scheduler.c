#include "sparkpipe/spark_cuda_graph_scheduler.h"

#include <stdio.h>
#include <string.h>

SparkStatus SparkInitializeCudaGraphScheduler(SparkCudaGraphScheduler *graph_scheduler, uint32_t stage_count)
{
    if (graph_scheduler == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (stage_count == 0 || stage_count > SPARKPIPE_MAX_STAGES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(graph_scheduler, 0, sizeof(*graph_scheduler));
    graph_scheduler->stage_count = stage_count;
    graph_scheduler->completion_checksum = 0x5343505544414752ull;

    return SPARK_STATUS_OK;
}

SparkStatus SparkAttachCudaGraphSchedulerCompletionQueue(SparkCudaGraphScheduler *graph_scheduler, SparkServiceCompletionQueue *completion_queue)
{
    if (graph_scheduler == 0 || completion_queue == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    graph_scheduler->completion_queue = completion_queue;
    graph_scheduler->completion_checksum = SparkMixU64(graph_scheduler->completion_checksum, 0x415454414348ull);
    graph_scheduler->completion_checksum = SparkMixU64(graph_scheduler->completion_checksum, graph_scheduler->stage_count);
    return SPARK_STATUS_OK;
}

SparkStatus SparkClearCudaGraphSchedulerCompletionQueue(SparkCudaGraphScheduler *graph_scheduler)
{
    if (graph_scheduler == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    graph_scheduler->completion_queue = 0;
    graph_scheduler->completion_checksum = SparkMixU64(graph_scheduler->completion_checksum, 0x434C454152ull);
    return SPARK_STATUS_OK;
}

static void SparkCudaGraphSchedulerPublishCompletion(SparkCudaGraphScheduler *graph_scheduler, const SparkGraphExecutionInput *execution_input, const SparkGraphExecutionOutput *execution_output, SparkStatus completion_status)
{
    SparkServiceCompletionRecord completion_record;
    SparkStatus publish_status;

    if (graph_scheduler == 0 || graph_scheduler->completion_queue == 0 || execution_input == 0 || execution_output == 0)
    {
        return;
    }

    memset(&completion_record, 0, sizeof(completion_record));
    completion_record.completion_kind = SPARK_SERVICE_COMPLETION_KIND_CUDA_GRAPH;
    completion_record.status = completion_status;
    completion_record.model_lane = execution_input->model_lane;
    completion_record.request_id = execution_input->slot_generation;
    completion_record.submitted_tick = execution_input->fabric_tick;
    completion_record.completed_tick = execution_input->fabric_tick + 1u;
    completion_record.stage_id = execution_input->stage_id;
    completion_record.token_begin = (uint32_t)execution_input->fabric_tick;
    completion_record.token_count = execution_output->tokens_processed;
    completion_record.bytes = execution_output->output_activation_payload_bytes;
    completion_record.payload_checksum = execution_output->output_activation_checksum;
    (void)snprintf(completion_record.text, sizeof(completion_record.text), "cuda_graph_stage_%u", execution_input->stage_id);

    publish_status = SparkServiceCompletionQueuePublishCompleted(graph_scheduler->completion_queue, &completion_record);
    if (publish_status == SPARK_STATUS_OK)
    {
        graph_scheduler->completion_publish_count += 1u;
        graph_scheduler->completion_checksum = SparkMixU64(graph_scheduler->completion_checksum, completion_record.record_checksum);
    }
    else
    {
        graph_scheduler->completion_publish_failure_count += 1u;
        graph_scheduler->completion_checksum = SparkMixU64(graph_scheduler->completion_checksum, publish_status);
    }
}

SparkStatus SparkAcquireStageGraph(SparkCudaGraphScheduler *graph_scheduler, uint32_t stage_id, SparkPhysicalProfileId profile_id)
{
    if (graph_scheduler == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (stage_id >= graph_scheduler->stage_count)
    {
        graph_scheduler->unavailable_graph_count += 1;
        return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
    }

    if (SparkGetPhysicalProfile(profile_id) == 0)
    {
        graph_scheduler->unavailable_graph_count += 1;
        return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
    }

    graph_scheduler->graph_launch_count += 1;
    return SPARK_STATUS_OK;
}

SparkStatus SparkExecuteStageGraph(SparkCudaGraphScheduler *graph_scheduler, const SparkGraphExecutionInput *execution_input, SparkGraphExecutionOutput *execution_output)
{
    uint64_t checksum;
    uint64_t active_mask_checksum;
    SparkStatus status;

    if (graph_scheduler == 0 || execution_input == 0 || execution_output == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (execution_input->stage_id >= graph_scheduler->stage_count)
    {
        graph_scheduler->unavailable_graph_count += 1;
        return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
    }

    if (SparkGetPhysicalProfile(execution_input->profile_id) == 0)
    {
        graph_scheduler->unavailable_graph_count += 1;
        return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
    }

    status = SparkValidateActivationLayout(&execution_input->activation_layout);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (execution_input->activation_layout.profile_id != execution_input->profile_id)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (execution_input->active_slot_mask.physical_slot_count != execution_input->activation_layout.physical_slot_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (SparkCountActiveSlotsInMask(&execution_input->active_slot_mask) != execution_input->active_slots)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (execution_input->input_activation_payload_bytes != execution_input->activation_layout.aligned_payload_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    active_mask_checksum = SparkComputeActiveSlotMaskChecksum(&execution_input->active_slot_mask);
    checksum = execution_input->input_activation_checksum;
    checksum = SparkMixU64(checksum, execution_input->fabric_tick);
    checksum = SparkMixU64(checksum, execution_input->stage_id);
    checksum = SparkMixU64(checksum, execution_input->model_lane);
    checksum = SparkMixU64(checksum, execution_input->profile_id);
    checksum = SparkMixU64(checksum, execution_input->slot_generation);
    checksum = SparkMixU64(checksum, execution_input->active_slots);
    checksum = SparkMixU64(checksum, execution_input->activation_layout.aligned_payload_bytes);
    checksum = SparkMixU64(checksum, execution_input->activation_layout.row_stride_bytes);
    checksum = SparkMixU64(checksum, active_mask_checksum);

    graph_scheduler->graph_launch_count += 1;

    execution_output->output_activation_checksum = checksum;
    execution_output->output_activation_payload_bytes = execution_input->activation_layout.aligned_payload_bytes;
    execution_output->active_mask_checksum = active_mask_checksum;
    execution_output->tokens_processed = execution_input->active_slots;

    SparkCudaGraphSchedulerPublishCompletion(graph_scheduler, execution_input, execution_output, SPARK_STATUS_OK);
    return SPARK_STATUS_OK;
}
