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
    uint32_t send_count;
    uint32_t nonzero_count;
} SparkDsv4ValidationCapture;

static SparkDsv4ValidationCapture *spark_dsv4_validation_capture;

static int SparkDsv4ValidationRequire(int condition, const char *message)
{
    if (condition != 0)
    {
        return 0;
    }
    fprintf(stderr, "dsv4_validation failure=%s\n", message);
    return 1;
}

static SparkStatus SparkDsv4ValidationHiddenSend(
    SparkHiddenTransportSession *transport_session,
    const SparkHiddenTransportPacket *packet)
{
    uint32_t element_index;
    cudaError_t error;

    if (transport_session == 0 || packet == 0 ||
        spark_dsv4_validation_capture == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (packet->abi_version != SPARK_HIDDEN_TRANSPORT_ABI_VERSION ||
        packet->descriptor_bytes < SPARK_HIDDEN_TRANSPORT_PACKET_BYTES ||
        packet->active_sequence_count != 1u ||
        packet->hidden_dimension != SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS ||
        packet->bytes_per_sequence !=
            SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS *
                SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT ||
        packet->sequence_id != 1u || packet->token_index != 0u ||
        packet->hidden_bf16 == 0)
    {
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    error = cudaMemcpy(
        spark_dsv4_validation_capture->hidden,
        packet->hidden_bf16,
        sizeof(spark_dsv4_validation_capture->hidden),
        cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "dsv4_validation cudaMemcpy=%s\n",
            cudaGetErrorString(error));
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    spark_dsv4_validation_capture->send_count += 1u;
    spark_dsv4_validation_capture->nonzero_count = 0u;
    for (element_index = 0u;
         element_index < SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
         ++element_index)
    {
        if (spark_dsv4_validation_capture->hidden[element_index] != 0u)
        {
            spark_dsv4_validation_capture->nonzero_count += 1u;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkDsv4ValidationInitializeConfiguration(
    SparkFirmwareModuleConfiguration *configuration,
    SparkFirmwareModuleHostServices *host_services)
{
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
}

static void SparkDsv4ValidationInitializeContext(
    SparkDsv4ResidentDecodeStageFrameContext *context,
    SparkDsv4DecodeBatchView *batch,
    SparkHiddenTransportSession *transport_session,
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
        SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT;
    context->decode_batch = batch;
    context->hidden_output_transport_session = transport_session;
    context->hidden_output_send_function = SparkDsv4ValidationHiddenSend;
    context->hidden_output_packet.abi_version =
        SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
    context->hidden_output_packet.descriptor_bytes =
        SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
    context->hidden_output_packet.flags =
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 |
        SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
    context->hidden_output_packet.sequence_id = *sequence_id;
    context->hidden_output_packet.token_index = *position;
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
    SparkDsv4ValidationCapture *capture)
{
    SparkDsv4DecodeBatchView batch;
    SparkDsv4ResidentDecodeStageFrameContext context;
    SparkHiddenTransportSession *transport_session;
    SparkModelDriverBuffer buffer;
    SparkModelDriverFrame frame;
    uint32_t lane;
    uint32_t token_id;
    uint64_t position;
    uint64_t sequence_id;
    SparkStatus status;

    transport_session = (SparkHiddenTransportSession *)(uintptr_t)1u;
    lane = 0u;
    position = 0u;
    sequence_id = 1u;
    SparkDsv4ValidationInitializeContext(
        &context,
        &batch,
        transport_session,
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
    frame.buffers = &buffer;
    frame.buffer_count = 1u;
    frame.user_context = &context;
    spark_dsv4_validation_capture = capture;
    status = SparkDsv4ResidentDecodeStageExecute(module_state, &frame);
    spark_dsv4_validation_capture = 0;
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "dsv4_validation execute=%s\n",
            SparkStatusToString(status));
        return 1;
    }
    return SparkDsv4ValidationRequire(capture->send_count == 1u,
        "hidden_output_send_count") ||
        SparkDsv4ValidationRequire(capture->nonzero_count > 0u,
            "hidden_output_nonzero");
}

int main(int argument_count, char **arguments)
{
    SparkFirmwareModuleConfiguration configuration;
    SparkFirmwareModuleHostServices host_services;
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
        &host_services);
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
    if (SparkDsv4ValidationRunFrame(module_state, &capture) != 0)
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
