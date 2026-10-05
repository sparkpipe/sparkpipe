#ifndef SPARKPIPE_SPARK_K3_RESIDENT_DECODE_STAGE_RUNNER_H
#define SPARKPIPE_SPARK_K3_RESIDENT_DECODE_STAGE_RUNNER_H

#include <stdint.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_sampling.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tp_collective.h"
#include "sparkpipe/spark_tp_device_collective.h"

#ifdef __cplusplus
extern "C" {
#endif


#define SPARK_K3_STAGE_RUNNER_ABI_VERSION 4u
#define SPARK_K3_STAGE_RUNNER_CONFIGURATION_BYTES \
    ((uint32_t)sizeof(SparkK3StageRunnerConfiguration))
#define SPARK_K3_STAGE_RUNNER_DISPATCH_BYTES \
    ((uint32_t)sizeof(SparkK3StageRunnerDispatch))
#define SPARK_K3_STAGE_RUNNER_BYTES \
    ((uint32_t)sizeof(SparkK3StageRunner))

#define SPARK_K3_STAGE_RUNNER_FLAG_TENSOR_PARALLEL 0x00000001u
#define SPARK_K3_STAGE_RUNNER_KNOWN_FLAGS \
    (SPARK_K3_STAGE_RUNNER_FLAG_TENSOR_PARALLEL)

#define SPARK_K3_RESIDUAL_BANK_BYTES_PER_ROW (9u * 7168u * 2u)

typedef struct SparkK3StageRunnerConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t stage_index;
    uint32_t stage_count;
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t max_active_sequence_count;
    uint32_t max_input_row_count;
    uint32_t resident_sequence_capacity;
    uint32_t kv_pages_per_sequence;
    uint64_t kv_page_bytes;
    const char *rank_pack_path;
    uint32_t multiprocessors;
    void *execution_stream;
    const SparkTpCollectiveConfig *tp_collective;
    const SparkTpDeviceCollectiveConfig *device_collective;
    const SparkTpDeviceCollectiveConfig *device_collective_wide;
    void (*layer_collective_override)(void *context, void *stream,
        uint32_t layer, uint32_t phase);
    void *layer_collective_context;
} SparkK3StageRunnerConfiguration;

typedef struct SparkK3StageRunnerDispatch
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t deadline_time_ns;
    uint32_t row_count;
    uint32_t active_sequence_count;
    const uint32_t *token_ids;
    const uint32_t *positions;
    const uint32_t *context_length;
    const uint32_t *sequence_of_row;
    const uint32_t *kda_state_index;
    const uint32_t *sequence_row_begin;
    const uint32_t *sequence_row_indices;
    const void *hidden_input_bf16;
    uint64_t hidden_input_bytes;
    void *hidden_output_bf16;
    uint64_t hidden_output_bytes;
    const void *residual_bank_input;
    uint64_t residual_bank_input_bytes;
    void *residual_bank_output;
    uint64_t residual_bank_output_bytes;
    uint32_t *output_token_ids;
    float *output_scores;
    uint32_t distribution_count;
    const uint32_t *distribution_rows;
    const uint32_t *distribution_positions;
    const SparkRowSampling *distribution_rules;
    SparkSamplingLogprob *distribution_logprobs;
    SparkModelDriverCompletionFunction completion_function;
    void *completion_context;
} SparkK3StageRunnerDispatch;

typedef struct SparkK3StageRunnerKv
{
    uint8_t *pool;
    uint64_t layer_stride_bytes;
    uint64_t layer_page_bytes;
    uint32_t layer_count;
    const uint32_t *page_table;
    uint32_t page_table_stride;
    uint32_t pool_page_count;
    uint32_t sequence_count;
} SparkK3StageRunnerKv;

typedef struct SparkK3StageRunnerStats
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t last_status;
    uint64_t submitted_count;
    uint64_t completed_count;
    uint64_t failed_count;
    uint64_t collective_count;
} SparkK3StageRunnerStats;

typedef struct SparkK3StageRunner
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t stage_index;
    uint32_t stage_count;
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t owns_embedding;
    uint32_t owns_final_head;
    void *private_state;
    SparkK3StageRunnerStats stats;
} SparkK3StageRunner;

SparkStatus SparkK3StageRunnerInitialize(
    SparkK3StageRunner *runner,
    const SparkK3StageRunnerConfiguration *configuration);

SparkStatus SparkK3StageRunnerSubmit(
    SparkK3StageRunner *runner,
    const SparkK3StageRunnerDispatch *dispatch);

SparkStatus SparkK3StageRunnerResetSlots(
    SparkK3StageRunner *runner,
    const uint32_t *slots,
    uint32_t count);
SparkStatus SparkK3StageRunnerGetStats(
    const SparkK3StageRunner *runner,
    SparkK3StageRunnerStats *stats_out);

void SparkK3StageRunnerDestroy(SparkK3StageRunner *runner);

uint32_t SparkK3StageRunnerKvLayerCount(const SparkK3StageRunner *runner);
SparkStatus SparkK3StageRunnerAttachKv(SparkK3StageRunner *runner, const SparkK3StageRunnerKv *kv);
uint64_t SparkK3StageRunnerRecurrentBytes(const SparkK3StageRunner *runner);
SparkStatus SparkK3StageRunnerRecurrentCopy(SparkK3StageRunner *runner, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream);
SparkStatus SparkK3StageRunnerPackIdentity(const SparkK3StageRunner *runner, uint8_t *digest, uint32_t digest_bytes);

const void *SparkK3StageRunnerProbeBuffers(const SparkK3StageRunner *runner);

SparkStatus SparkK3StageRunnerStepHalf(
    SparkK3StageRunner *runner,
    uint32_t layer,
    uint32_t phase,
    const void *hidden_input_bf16,
    const void *partial_input_bf16,
    void *partial_output_bf16);

#ifdef __cplusplus
}
#endif

#endif
