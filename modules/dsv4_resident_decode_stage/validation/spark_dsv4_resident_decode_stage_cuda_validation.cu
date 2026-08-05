#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"

typedef struct SparkDsv4ValidationCapture
{
    uint16_t hidden[SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS];
    uint32_t output_count;
    uint32_t nonzero_count;
} SparkDsv4ValidationCapture;

static int SparkDsv4ValidationRequire(int condition, const char *message)
{
    if (condition != 0)
    {
        return 0;
    }
    fprintf(stderr, "dsv4_validation failure=%s\n", message);
    return 1;
}

static void SparkDsv4ValidationInitializeConfiguration(
    SparkFirmwareModuleConfiguration *configuration,
    SparkFirmwareModuleHostServices *host_services,
    SparkDsv4ResidentDecodeStageNodeContext *node_context)
{
    const char *stage_pack_path;
    stage_pack_path = getenv("SPARK_DSV4_STAGE_PACK_PATH");
    memset(configuration, 0, sizeof(*configuration));
    configuration->abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    configuration->descriptor_bytes = sizeof(*configuration);
    configuration->model_id = "deepseek-ai/DeepSeek-V4-Flash";
    configuration->model_revision = "greenfield-bring-up";
    configuration->stage_name = "dsv4_resident_decode_stage";
    configuration->program_name = "dsv4_decode";
    configuration->operation_name = "dsv4_resident_decode_stage";
    configuration->configuration_json = "{}";
    configuration->configuration_json_bytes = 2u;

    memset(host_services, 0, sizeof(*host_services));
    host_services->abi_version =
        SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
    host_services->descriptor_bytes = sizeof(*host_services);
    host_services->node_id = "spark0-dsv4-validator";
    host_services->node_target = "cuda.sm121a";
    host_services->execution_stream = (void *)cudaStreamPerThread;
    memset(node_context, 0, sizeof(*node_context));
    node_context->abi_version =
        SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
    node_context->descriptor_bytes =
        SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
    node_context->flags =
        SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_ALLOW_UNQUALIFIED;
    node_context->stage_count = 2u;
    node_context->stage_index = 0u;
    node_context->first_layer_index = 0u;
    node_context->layer_count = SPARK_DSV4_MODEL_LAYER_COUNT;
    node_context->resident_sequence_capacity = 1u;
    node_context->pipeline_slot_count = 1u;
    node_context->max_sequence_positions = 4096u;
	node_context->linear_weight_codec = SPARK_DSV4_MODEL_NON_EXPERT_WEIGHT_CODEC;
	node_context->expert_weight_codec = SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC;
	node_context->kv_cache_codec = SPARK_DSV4_MODEL_KV_CACHE_CODEC;
    node_context->stage_pack_path = stage_pack_path;
    host_services->node_context = node_context;
}

static void SparkDsv4ValidationInitializeContext(
    SparkDsv4ResidentDecodeStageFrameContext *context,
    SparkDsv4DecodeBatchView *batch,
    void *hidden_output_bf16,
    uint32_t *lane,
    uint64_t *position,
    uint64_t *sequence_id)
{
    memset(context, 0, sizeof(*context));
    context->abi_version =
        SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
    context->descriptor_bytes = sizeof(*context);
    context->flags =
        SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW |
        SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_BUFFER;
    context->decode_batch = batch;
    context->hidden_output_bf16 = hidden_output_bf16;
    context->hidden_output_bytes =
        SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * sizeof(uint16_t);
    batch->abi_version =
        SPARK_DSV4_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION;
    batch->descriptor_bytes = sizeof(*batch);
    batch->row_count = 1u;
    batch->reserved0 = 0u;
    batch->row_lane_indices = lane;
    batch->row_positions = position;
    batch->row_sequence_ids = sequence_id;
}

static int SparkDsv4ValidationRunFrame(
    void *module_state,
    void *execution_stream,
    SparkDsv4ValidationCapture *capture)
{
    SparkDsv4DecodeBatchView batch;
    SparkDsv4ResidentDecodeStageFrameContext context;
    SparkModelDriverBuffer buffer;
    SparkModelDriverFrame frame;
    uint32_t lane;
    uint32_t token_id;
    uint64_t position;
    uint64_t sequence_id;
    uint16_t *hidden_output_bf16;
    uint32_t element_index;
    cudaError_t error;
    SparkStatus status;

    hidden_output_bf16 = 0;
    error = cudaMalloc((void **)&hidden_output_bf16,
        sizeof(capture->hidden));
    if (error != cudaSuccess)
    {
        return 1;
    }
    lane = 0u;
    position = 0u;
    sequence_id = 1u;
    SparkDsv4ValidationInitializeContext(
        &context,
        &batch,
        hidden_output_bf16,
        &lane,
        &position,
        &sequence_id);
    token_id = 10397u;
    memset(&buffer, 0, sizeof(buffer));
    buffer.slot = 0u;
    buffer.flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
    buffer.address = &token_id;
    buffer.bytes = sizeof(token_id);
    memset(&frame, 0, sizeof(frame));
    frame.request_id = 1u;
    frame.sequence_id = sequence_id;
    frame.sequence_position = position;
    frame.active_slot_count = 1u;
    frame.new_token_count = 1u;
    frame.program_id = 1u;
    frame.execution_stream = execution_stream;
    frame.buffers = &buffer;
    frame.buffer_count = 1u;
    frame.user_context = &context;
    status = SparkDsv4ResidentDecodeStageExecute(module_state, &frame);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "dsv4_validation execute=%s\n",
            SparkStatusToString(status));
        cudaFree(hidden_output_bf16);
        return 1;
    }
    error = cudaMemcpy(capture->hidden,hidden_output_bf16,
        sizeof(capture->hidden),cudaMemcpyDeviceToHost);
    cudaFree(hidden_output_bf16);
    if (error != cudaSuccess)
    {
        return 1;
    }
    capture->output_count = 1u;
    for (element_index = 0u;
         element_index < SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
         ++element_index)
    {
        if (capture->hidden[element_index] != 0u)
        {
            capture->nonzero_count += 1u;
        }
    }
    return SparkDsv4ValidationRequire(capture->output_count == 1u,
        "hidden_output_count") ||
        SparkDsv4ValidationRequire(capture->nonzero_count > 0u,
            "hidden_output_nonzero");
}

int main(int argument_count, char **arguments)
{
    SparkFirmwareModuleConfiguration configuration;
    SparkFirmwareModuleHostServices host_services;
    SparkDsv4ResidentDecodeStageNodeContext node_context;
    SparkDsv4ValidationCapture capture;
    SparkModelDriverAdmissionRequest admission_request;
    SparkModelDriverAdmissionDecision admission_decision;
    SparkModelDriverRuntimeSnapshot snapshot;
    void *module_state;
    SparkStatus status;

    if (argument_count != 2)
    {
        fprintf(stderr, "usage: %s VALIDATION_CONFIGURATION_SHA256\n",
            arguments[0]);
        return 2;
    }
    memset(&capture, 0, sizeof(capture));
    SparkDsv4ValidationInitializeConfiguration(
        &configuration,
        &host_services,
        &node_context);
    module_state = 0;
    status = SparkDsv4ResidentDecodeStageInitialize(
        &configuration,
        &host_services,
        &module_state);
    if (status != SPARK_STATUS_OK || module_state == 0)
    {
        fprintf(stderr, "dsv4_validation initialize=%s\n",
            SparkStatusToString(status));
        return 1;
    }
    memset(&admission_request, 0, sizeof(admission_request));
    admission_request.descriptor_bytes = sizeof(admission_request);
    admission_request.program_id = 1u;
    admission_request.request_id = 1u;
    admission_request.sequence_id = 1u;
    admission_request.active_slot_count = 1u;
    admission_request.new_token_count = 1u;
    memset(&admission_decision, 0, sizeof(admission_decision));
    admission_decision.descriptor_bytes = sizeof(admission_decision);
    status = SparkDsv4ResidentDecodeStageAdmit(
        module_state,
        &admission_request,
        &admission_decision);
    if (status != SPARK_STATUS_OK || admission_decision.accepted == 0u)
    {
        fprintf(stderr, "dsv4_validation admit=%s accepted=%u\n",
            SparkStatusToString(status), admission_decision.accepted);
        SparkDsv4ResidentDecodeStageDestroy(module_state);
        return 1;
    }
    if (SparkDsv4ValidationRunFrame(
            module_state,host_services.execution_stream,&capture) != 0)
    {
        SparkDsv4ResidentDecodeStageDestroy(module_state);
        return 1;
    }
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.descriptor_bytes = sizeof(snapshot);
    status = SparkDsv4ResidentDecodeStageSnapshot(
        module_state,
        1u,
        &snapshot);
    SparkDsv4ResidentDecodeStageDestroy(module_state);
    if (status != SPARK_STATUS_OK || snapshot.submitted_count != 1u ||
        snapshot.completed_count != 1u)
    {
        fprintf(stderr,
            "dsv4_validation snapshot=%s submitted=%llu completed=%llu\n",
            SparkStatusToString(status),
            (unsigned long long)snapshot.submitted_count,
            (unsigned long long)snapshot.completed_count);
        return 1;
    }
    printf("dsv4_validation PASS config=%s nonzero_hidden=%u\n",
        arguments[1], capture.nonzero_count);
    return 0;
}
