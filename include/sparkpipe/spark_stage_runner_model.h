#ifndef SPARKPIPE_SPARK_STAGE_RUNNER_MODEL_H
#define SPARKPIPE_SPARK_STAGE_RUNNER_MODEL_H

#include <stdint.h>

#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_stage_runner.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_STAGE_RUNNER_MODEL_ABI_VERSION 1u
#define SPARK_STAGE_RUNNER_ROUND_ALL 0u
#define SPARK_STAGE_RUNNER_ROUND_BEGIN 1u
#define SPARK_STAGE_RUNNER_ROUND_FINISH 2u

struct SparkWeightdLazyPack;
struct SparkWeightdManifest;

typedef struct SparkStageRunnerServices
{
	void *context;
	SparkStatus (*round)(void *context, void *stream, uint32_t operation, uint32_t stage, uint32_t rows,
		uint32_t row_elements, const void *local, void *full);
	uint32_t (*published)(void *context);
	void (*fail)(void *context, SparkStatus status);
	SparkStatus (*expert_weights)(void *context, uint32_t layer, const uint32_t *group_row_offset_device,
		uint32_t routed_rows, const void **w1, const void **w2);
	void (*layer_done)(void *context, uint32_t layer);
} SparkStageRunnerServices;

typedef struct SparkStageRunnerStep
{
	uint16_t *hidden_bf16;
	const uint32_t *positions;
	const uint32_t *context_length;
	const uint32_t *sequence_of_row;
	const uint32_t *sequence_row_begin;
	const uint32_t *sequence_row_indices;
	const uint32_t *recurrent_index;
	uint32_t gather_sequence;
	uint32_t gather_context;
	uint32_t rows;
	uint32_t sequences;
	uint32_t commit;
	uint32_t context;
	uint32_t multiprocessors;
	uint32_t last_rows_only;
} SparkStageRunnerStep;

typedef struct SparkStageRunnerModelGeometry
{
	uint32_t hidden;
	uint32_t vocab;
	uint32_t total_layers;
	uint32_t first_layer;
	uint32_t layer_count;
	float rms_epsilon;
	uint64_t sideband_bytes_per_row;
	uint64_t pack_bytes;
	uint64_t embed_offset;
	uint64_t embed_bytes;
	uint32_t embed_rows;
	uint64_t head_norm_offset;
	uint64_t head_norm_bytes;
	uint64_t head_offset;
	uint64_t head_bytes;
	uint32_t head_rows;
	uint32_t kv_layer_count;
	uint64_t kv_layer_page_bytes;
	SparkKvShard kv_shard;
	uint64_t recurrent_bytes;
	uint32_t experts;
	uint32_t top_k;
	uint32_t lease_tensor_base;
	const uint64_t *expert_w1_offset;
	const uint64_t *expert_w2_offset;
} SparkStageRunnerModelGeometry;

typedef struct SparkStageRunnerModelOpen
{
	const SparkStageRunnerConfiguration *configuration;
	const SparkStageRunnerServices *services;
	void *stream;
} SparkStageRunnerModelOpen;

struct SparkStageRunnerModelInterface
{
	uint32_t abi_version;
	const char *tag;
	const char *weightd_model;
	const char *weightd_revision;
	SparkStatus (*open)(const SparkStageRunnerModelOpen *request, void **model, SparkStageRunnerModelGeometry *geometry);
	SparkStatus (*manifest_check)(void *model, const struct SparkWeightdManifest *manifest);
	SparkStatus (*bind)(void *model, struct SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry);
	void (*close)(void *model);
	SparkStatus (*step)(void *model, const SparkStageRunnerStep *step, void *stream);
	SparkStatus (*sideband)(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows, void *stream);
	SparkStatus (*attach_kv)(void *model, const SparkStageRunnerKv *kv);
	SparkStatus (*recurrent_copy)(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream);
	SparkStatus (*reset_slot)(void *model, uint32_t slot, void *stream);
	const void *(*probe_buffers)(void *model);
	void (*report)(void *model, uint32_t rank);
};

#ifdef __cplusplus
}
#endif

#endif
