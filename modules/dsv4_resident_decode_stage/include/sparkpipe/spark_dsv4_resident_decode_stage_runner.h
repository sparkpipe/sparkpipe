#pragma once

#include <stdint.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DSV4_STAGE_RUNNER_ABI_VERSION 3u
#define SPARK_DSV4_STAGE_RUNNER_CONFIGURATION_BYTES \
    ((uint32_t)sizeof(SparkDsv4StageRunnerConfiguration))
#define SPARK_DSV4_STAGE_RUNNER_DISPATCH_BYTES \
    ((uint32_t)sizeof(SparkDsv4StageRunnerDispatch))
#define SPARK_DSV4_STAGE_RUNNER_BYTES \
    ((uint32_t)sizeof(SparkDsv4StageRunner))

#define SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_ADMISSION 0x00000001u
#define SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_INPUT_BOUNDARY 0x00000002u
#define SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_OUTPUT_BOUNDARY 0x00000004u
#define SPARK_DSV4_STAGE_RUNNER_KNOWN_FLAGS \
    (SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_ADMISSION | \
     SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_INPUT_BOUNDARY | \
     SPARK_DSV4_STAGE_RUNNER_FLAG_REQUIRE_OUTPUT_BOUNDARY)

#define SPARK_DSV4_STAGE_RUNNER_DISPATCH_FLAG_PREFILL \
    SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL
#define SPARK_DSV4_STAGE_RUNNER_DISPATCH_KNOWN_FLAGS \
    SPARK_DSV4_STAGE_RUNNER_DISPATCH_FLAG_PREFILL

typedef struct SparkDsv4StageRunnerConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t reserved0;
	uint32_t stage_index;
	uint32_t stage_count;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t resident_sequence_capacity;
    const SparkModelDriverInterface *driver_interface;
    void *driver_instance;
    const SparkModelDriverProgramDescriptor *program;
    void *execution_stream;
} SparkDsv4StageRunnerConfiguration;

typedef struct SparkDsv4StageRunnerDispatch
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
    uint32_t row_count;
    uint32_t lane_count;
    const uint32_t *token_ids;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
	uint32_t emit_count;
	const uint32_t *emit_row_indices;
	const uint32_t *emit_lane_indices;
	uint32_t *output_token_ids;
    const void *hidden_input_bf16;
    uint64_t hidden_input_bytes;
    void *hidden_output_bf16;
    uint64_t hidden_output_bytes;
    SparkModelDriverResidencyToken residency;
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
} SparkDsv4StageRunnerDispatch;

typedef struct SparkDsv4StageRunnerStats
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t last_status;
    uint32_t last_admission_rejection;
    uint64_t submitted_count;
    uint64_t admitted_count;
    uint64_t rejected_count;
    uint64_t submit_failed_count;
} SparkDsv4StageRunnerStats;

typedef struct SparkDsv4StageRunner
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t stage_index;
	uint32_t stage_count;
	uint32_t max_active_sequence_count;
	uint32_t max_input_row_count;
	uint32_t resident_sequence_capacity;
	uint32_t owns_embedding;
    uint32_t owns_final_head;
    const SparkModelDriverInterface *driver_interface;
    void *driver_instance;
    const SparkModelDriverProgramDescriptor *program;
    void *execution_stream;
    SparkDsv4StageRunnerStats stats;
} SparkDsv4StageRunner;

SparkStatus SparkDsv4StageRunnerInitialize(
    SparkDsv4StageRunner *runner,
    const SparkDsv4StageRunnerConfiguration *configuration);

SparkStatus SparkDsv4StageRunnerSubmit(
    SparkDsv4StageRunner *runner,
    const SparkDsv4StageRunnerDispatch *dispatch);

SparkStatus SparkDsv4StageRunnerGetStats(
    const SparkDsv4StageRunner *runner,
    SparkDsv4StageRunnerStats *stats_out);

#ifdef __cplusplus
}
#endif
