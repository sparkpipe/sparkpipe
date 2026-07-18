#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION 3u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_CONFIGURATION_BYTES \
    ((uint32_t)sizeof(SparkGlm52ResidentDecodeStageProductionRunnerConfiguration))
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_BYTES \
    ((uint32_t)sizeof(SparkGlm52ResidentDecodeStageProductionRunner))
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_BYTES \
    ((uint32_t)sizeof(SparkGlm52ResidentDecodeStageProductionRunnerDispatch))
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_STATS_BYTES \
    ((uint32_t)sizeof(SparkGlm52ResidentDecodeStageProductionRunnerStats))

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_ADMISSION \
    0x00000001u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_INPUT_TRANSPORT \
    0x00000002u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT \
    0x00000004u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DEFAULT_FLAGS \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_ADMISSION | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_INPUT_TRANSPORT | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT)
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_KNOWN_FLAGS \
    SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DEFAULT_FLAGS

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_REQUIRED_PROGRAM_FLAGS \
    (SPARK_MODEL_DRIVER_PROGRAM_FLAG_STREAM_ORDERED | \
     SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_RESIDENT_STATE | \
     SPARK_MODEL_DRIVER_PROGRAM_FLAG_DRIVER_OWNS_KV_CACHE | \
     SPARK_MODEL_DRIVER_PROGRAM_FLAG_FIXED_FIRMWARE | \
     SPARK_MODEL_DRIVER_PROGRAM_FLAG_REQUIRES_HIDDEN_TRANSPORT | \
     SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_FILE_TRANSPORT | \
     SPARK_MODEL_DRIVER_PROGRAM_FLAG_NO_SHELL_TRANSPORT)

#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_PREFILL \
    SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_MTP_TREE_VERIFY \
    0x00000004u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_HIDDEN_INPUT_PRERECEIVED \
    0x00000008u
#define SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_KNOWN_FLAGS \
    (SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_PREFILL | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_MTP_TREE_VERIFY | \
     SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_HIDDEN_INPUT_PRERECEIVED)

typedef struct SparkGlm52ResidentDecodeStageProductionRunnerConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t reserved0;
    const SparkModelDriverInterface *driver_interface;
    void *driver_instance;
    const SparkModelDriverProgramDescriptor *program;
    void *execution_stream;
} SparkGlm52ResidentDecodeStageProductionRunnerConfiguration;

typedef struct SparkGlm52ResidentDecodeStageProductionRunnerDispatch
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t priority;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t deadline_time_ns;
    uint32_t active_sequence_count;
    uint32_t new_token_count;
    uint32_t pipeline_slot;
    uint32_t logical_lane_count;
    uint32_t rows_per_lane;
    uint32_t reserved0;
    SparkModelDriverResidencyToken residency;
    const uint32_t *mtp_draft_token_budgets;
    const SparkGlm52DsparkHiddenTapPlan *dspark_hidden_tap_plan;
    void *const *dspark_hidden_tap_outputs_bf16;
    uint64_t dspark_hidden_tap_lane_stride_bytes;
    const SparkGlm52KvBlockTableView *kv_block_table;
    const SparkGlm52ResidentDecodeStagePrefillFrameView *prefill_view;
    SparkHiddenTransportSession *hidden_input_transport_session;
    SparkHiddenTransportSession *hidden_output_transport_session;
    SparkHiddenTransportPacket hidden_input_packet;
    SparkHiddenTransportPacket hidden_output_packet;
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
} SparkGlm52ResidentDecodeStageProductionRunnerDispatch;

typedef struct SparkGlm52ResidentDecodeStageProductionRunnerStats
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t last_status;
    uint32_t last_admission_rejection;
    uint64_t submitted_count;
    uint64_t admitted_count;
    uint64_t rejected_count;
    uint64_t submit_failed_count;
} SparkGlm52ResidentDecodeStageProductionRunnerStats;

typedef struct SparkGlm52ResidentDecodeStageProductionRunner
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t program_id;
    const SparkModelDriverInterface *driver_interface;
    void *driver_instance;
    const SparkModelDriverProgramDescriptor *program;
    void *execution_stream;
    SparkGlm52ResidentDecodeStageProductionRunnerStats stats;
} SparkGlm52ResidentDecodeStageProductionRunner;

SparkStatus SparkGlm52ResidentDecodeStageProductionRunnerInitialize(
    SparkGlm52ResidentDecodeStageProductionRunner *runner,
    const SparkGlm52ResidentDecodeStageProductionRunnerConfiguration *configuration);

SparkStatus SparkGlm52ResidentDecodeStageProductionRunnerSubmit(
    SparkGlm52ResidentDecodeStageProductionRunner *runner,
    const SparkGlm52ResidentDecodeStageProductionRunnerDispatch *dispatch);

SparkStatus SparkGlm52ResidentDecodeStageProductionRunnerProgress(
    SparkGlm52ResidentDecodeStageProductionRunner *runner);

SparkStatus SparkGlm52ResidentDecodeStageProductionRunnerWaitIdle(
    SparkGlm52ResidentDecodeStageProductionRunner *runner,
    uint32_t max_poll_count);

SparkStatus SparkGlm52ResidentDecodeStageProductionRunnerGetStats(
    const SparkGlm52ResidentDecodeStageProductionRunner *runner,
    SparkGlm52ResidentDecodeStageProductionRunnerStats *stats_out);

#ifdef __cplusplus
}
#endif
