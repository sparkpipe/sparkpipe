#ifndef SPARKPIPE_INFERENCE_RUNNER_DEVICE_STAGE_RUNNER_H
#define SPARKPIPE_INFERENCE_RUNNER_DEVICE_STAGE_RUNNER_H

#include <stdint.h>

#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DEVICE_RUNNER_PAGE_SLOTS 64u

typedef struct SparkDeviceStageRunner SparkDeviceStageRunner;

typedef struct SparkDeviceStageRunnerConfig
{
	const char *pack_path;
	const SparkStageRunnerModelInterface *model;
	uint32_t max_rows;
	uint32_t sequences;
	uint32_t max_context;
	uint32_t head_rows;
	uint64_t state_budget_bytes;
	uint32_t linear_weight_codec;
	uint32_t verify_pack;
	uint32_t output_vocabulary;
} SparkDeviceStageRunnerConfig;

typedef struct SparkDeviceStageRunnerBatch
{
	uint32_t rows;
	const uint32_t *token_ids;
	const uint32_t *positions;
	const uint32_t *sequence_of_row;
	uint32_t head_rows;
	const uint32_t *head_row_index;
	const uint32_t *targets;
	uint32_t verify;
	const uint32_t *route_override;
	uint32_t route_override_width;
} SparkDeviceStageRunnerBatch;

typedef struct SparkDeviceStageRunnerResult
{
	const uint32_t *argmax;
	const float *logprob;
	const float *logits;
	uint64_t logit_stride;
	uint64_t step_ns;
} SparkDeviceStageRunnerResult;

typedef struct SparkDeviceStageRunnerInfo
{
	uint32_t hidden;
	uint32_t vocab;
	uint32_t layers;
	uint64_t pack_bytes;
	uint64_t streamed_bytes_per_token;
	uint64_t streamed_bytes_per_chunk;
	uint64_t device_state_bytes;
	char pack_sha256[SPARK_SHA256_HEX_BYTES];
	char device[64];
	char target[64];
} SparkDeviceStageRunnerInfo;

SparkStatus SparkDeviceStageRunnerOpen(SparkDeviceStageRunner **runner, const SparkDeviceStageRunnerConfig *config);
uint64_t SparkDeviceStageRunnerDecodeBytes(const SparkDeviceStageRunner *runner, uint32_t context);
SparkStatus SparkDeviceStageRunnerStep(SparkDeviceStageRunner *runner, const SparkDeviceStageRunnerBatch *batch, SparkDeviceStageRunnerResult *result);
SparkStatus SparkDeviceStageRunnerKeepRows(SparkDeviceStageRunner *runner, const uint32_t *kept, uint32_t sequences);
SparkStatus SparkDeviceStageRunnerResetSequence(SparkDeviceStageRunner *runner, uint32_t sequence);
SparkStatus SparkDeviceStageRunnerInfoGet(const SparkDeviceStageRunner *runner, SparkDeviceStageRunnerInfo *info);
void *SparkDeviceStageRunnerModel(const SparkDeviceStageRunner *runner);
void SparkDeviceStageRunnerClose(SparkDeviceStageRunner *runner);

#ifdef __cplusplus
}
#endif

#endif
