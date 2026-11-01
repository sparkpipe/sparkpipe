#ifndef SPARKPIPE_SPARK_STAGE_RUNNER_H
#define SPARKPIPE_SPARK_STAGE_RUNNER_H

#include <stdint.h>

#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_sampling.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tp_device_collective.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_STAGE_RUNNER_ABI_VERSION 5u
#define SPARK_STAGE_RUNNER_CONFIGURATION_BYTES ((uint32_t)sizeof(SparkStageRunnerConfiguration))
#define SPARK_STAGE_RUNNER_DISPATCH_BYTES ((uint32_t)sizeof(SparkStageRunnerDispatch))
#define SPARK_STAGE_RUNNER_BYTES ((uint32_t)sizeof(SparkStageRunner))
#define SPARK_STAGE_RUNNER_FLAG_TENSOR_PARALLEL 0x00000001u
#define SPARK_STAGE_RUNNER_KNOWN_FLAGS (SPARK_STAGE_RUNNER_FLAG_TENSOR_PARALLEL)
#define SPARK_STAGE_RUNNER_DISPATCH_FLAG_PREFILL 0x00000001u
#define SPARK_STAGE_RUNNER_DISPATCH_FLAG_VERIFY 0x00000002u
#define SPARK_STAGE_RUNNER_DISPATCH_FLAG_DEFER 0x00000004u
#define SPARK_STAGE_RUNNER_DISPATCH_KNOWN_FLAGS (SPARK_STAGE_RUNNER_DISPATCH_FLAG_PREFILL | SPARK_STAGE_RUNNER_DISPATCH_FLAG_VERIFY | \
	SPARK_STAGE_RUNNER_DISPATCH_FLAG_DEFER)
#define SPARK_STAGE_RUNNER_DIGEST_HEX 64u
#define SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT 1u
#define SPARK_STAGE_RUNNER_TAP_FORM_FINAL 3u
#define SPARK_STAGE_RUNNER_TAP_MAX_ENTRIES 128u
#define SPARK_STAGE_RUNNER_TAP_CHAIN_STEPS 8u

typedef struct SparkStageRunnerModelInterface SparkStageRunnerModelInterface;

typedef struct SparkStageRunnerTapEntry
{
	uint16_t form;
	uint16_t layer;
} SparkStageRunnerTapEntry;

typedef struct SparkStageRunnerTapPlan
{
	uint32_t entry_count;
	SparkStageRunnerTapEntry entries[SPARK_STAGE_RUNNER_TAP_MAX_ENTRIES];
	uint32_t capture;
	uint32_t row_capacity;
	uint16_t *slab;
	uint64_t row_stride_elements;
	uint32_t *first_row;
} SparkStageRunnerTapPlan;

typedef struct SparkStageRunnerWeights
{
	const char *socket_path;
	char pack_sha256[SPARK_STAGE_RUNNER_DIGEST_HEX + 1u];
	uint64_t expert_pool_bytes;
	uint64_t spine_budget_bytes;
} SparkStageRunnerWeights;

typedef struct SparkStageRunnerStreamedPrefillConfig
{
	uint32_t node_rows;
	uint32_t gid_index;
	uint32_t pair_gid_index;
	uint32_t traffic_class;
	uint32_t dense_buffers;
	uint32_t expert_slots;
	const char *partner_pack_path;
	const char *fabric_device;
	const char *pair_device;
} SparkStageRunnerStreamedPrefillConfig;

typedef struct SparkStageRunnerBulkCollectiveConfig
{
	uint32_t gid_index;
	uint32_t pair_gid_index;
	uint32_t traffic_class;
	uint32_t reserved0;
	uint64_t block_bytes;
	uint64_t minimum_bytes;
	const char *fabric_device;
	const char *pair_device;
} SparkStageRunnerBulkCollectiveConfig;

typedef struct SparkStageRunnerConfiguration
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
	uint64_t state_budget_bytes;
	const char *rank_pack_path;
	const char *stray_working_set_path;
	SparkStageRunnerWeights weights;
	uint32_t multiprocessors;
	void *execution_stream;
	const SparkTpDeviceCollectiveConfig *device_collective;
	void (*layer_collective_override)(void *context, void *stream, uint32_t layer, uint32_t phase);
	void *layer_collective_context;
	uint32_t linear_weight_codec;
	uint32_t max_shared_row_count;
	uint32_t head_row_capacity;
	uint32_t output_vocabulary;
	SparkStageRunnerStreamedPrefillConfig streamed_prefill;
	SparkStageRunnerBulkCollectiveConfig bulk_collectives;
} SparkStageRunnerConfiguration;

typedef struct SparkStageRunnerDispatch
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
	const uint32_t *recurrent_index;
	const uint32_t *sequence_row_begin;
	const uint32_t *sequence_row_indices;
	uint32_t gather_sequence;
	uint32_t gather_context;
	const void *hidden_input_bf16;
	uint64_t hidden_input_bytes;
	void *hidden_output_bf16;
	uint64_t hidden_output_bytes;
	const void *sideband_input;
	uint64_t sideband_input_bytes;
	void *sideband_output;
	uint64_t sideband_output_bytes;
	uint32_t *output_token_ids;
	float *output_scores;
	uint32_t distribution_count;
	const uint32_t *distribution_rows;
	const uint32_t *distribution_positions;
	const SparkRowSampling *distribution_rules;
	SparkSamplingLogprob *distribution_logprobs;
	uint32_t chain_steps;
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	const uint32_t *host_positions;
	const uint32_t *host_sequence_of_row;
	const uint32_t *host_state_target;
} SparkStageRunnerDispatch;

typedef struct SparkStageRunnerKv
{
	uint8_t *pool;
	uint64_t layer_stride_bytes;
	uint64_t layer_page_bytes;
	uint32_t layer_count;
	const uint32_t *page_table;
	uint32_t page_table_stride;
	uint32_t pool_page_count;
	uint32_t sequence_count;
	SparkKvShard context_shard;
	uint8_t *second_pool;
	uint64_t second_layer_stride_bytes;
	uint64_t second_layer_page_bytes;
	uint32_t second_layer_count;
} SparkStageRunnerKv;

typedef struct SparkStageRunnerStats
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t last_status;
	uint64_t submitted_count;
	uint64_t completed_count;
	uint64_t failed_count;
	uint64_t collective_count;
} SparkStageRunnerStats;

typedef struct SparkStageRunner
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
	SparkStageRunnerStats stats;
} SparkStageRunner;

SparkStatus SparkStageRunnerInitialize(SparkStageRunner *runner, const SparkStageRunnerConfiguration *configuration,
	const SparkStageRunnerModelInterface *model);
SparkStatus SparkStageRunnerSubmit(SparkStageRunner *runner, const SparkStageRunnerDispatch *dispatch);
SparkStatus SparkStageRunnerResetSlots(SparkStageRunner *runner, const uint32_t *slots, uint32_t count);
SparkStatus SparkStageRunnerGetStats(const SparkStageRunner *runner, SparkStageRunnerStats *stats_out);
void SparkStageRunnerDestroy(SparkStageRunner *runner);
uint32_t SparkStageRunnerKvLayerCount(const SparkStageRunner *runner);
SparkStatus SparkStageRunnerAttachKv(SparkStageRunner *runner, const SparkStageRunnerKv *kv);
uint64_t SparkStageRunnerRecurrentBytes(const SparkStageRunner *runner);
uint32_t SparkStageRunnerVerifyCapable(const SparkStageRunner *runner);
uint64_t SparkStageRunnerTakeCaptureNs(SparkStageRunner *runner);
SparkStatus SparkStageRunnerKeepRows(SparkStageRunner *runner, const uint32_t *kept, uint32_t sequences);
SparkStatus SparkStageRunnerCheck(SparkStageRunner *runner);
SparkStatus SparkStageRunnerSetTaps(SparkStageRunner *runner, const SparkStageRunnerTapEntry *entries, uint32_t entry_count, uint32_t capture);
SparkStatus SparkStageRunnerTapShape(const SparkStageRunner *runner, uint32_t *hidden, uint32_t *total_layers);
const uint16_t *SparkStageRunnerTapRows(const SparkStageRunner *runner, uint32_t step, uint32_t rows, uint64_t *row_stride_elements);
uint32_t SparkStageRunnerTapFirstRow(const SparkStageRunner *runner);
SparkStatus SparkStageRunnerRecurrentCopy(SparkStageRunner *runner, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream);
SparkStatus SparkStageRunnerPackIdentity(const SparkStageRunner *runner, uint8_t *digest, uint32_t digest_bytes);
const void *SparkStageRunnerProbeBuffers(const SparkStageRunner *runner);
void *SparkStageRunnerModel(const SparkStageRunner *runner);

#ifdef __cplusplus
}
#endif

#endif
