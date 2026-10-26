#ifndef SPARKPIPE_INFERENCE_RUNNER_DEVICE_STAGE_MODEL_H
#define SPARKPIPE_INFERENCE_RUNNER_DEVICE_STAGE_MODEL_H

#include <stdint.h>

#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_named_pack.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "inference/runner/device_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DEVICE_STAGE_MAX_RECURRENT_POOLS 4u
#define SPARK_DEVICE_STAGE_VERIFY_ROWS SPARK_OPS_DECODE_MAX_ROWS

typedef struct SparkDeviceStageConfigValue
{
	const char *key;
	uint32_t value;
} SparkDeviceStageConfigValue;

typedef struct SparkDeviceStageSlice
{
	uint64_t bytes;
	SparkTensor *target;
} SparkDeviceStageSlice;

typedef struct SparkDeviceStageRecurrentPool
{
	SparkDeviceBuffer buffer;
	uint64_t width;
} SparkDeviceStageRecurrentPool;

typedef struct SparkDeviceStageShape
{
	const char *tag;
	uint32_t hidden;
	uint32_t vocab;
	uint32_t layers;
	uint32_t kv_layers;
	uint32_t kv_heads;
	uint32_t kv_head_dimension;
	uint32_t pack_alignment;
	float epsilon;
	uint32_t tied_head;
	float logit_softcap;
} SparkDeviceStageShape;

typedef struct SparkDeviceStage
{
	const SparkDeviceStageShape *shape;
	const SparkStageRunnerServices *services;
	SparkNamedPack pack;
	uint32_t pack_open;
	SparkDeviceOps *ops;
	uint32_t max_rows;
	uint32_t slots;
	uint32_t pages_per_sequence;
	uint32_t max_positions;
	uint64_t pack_bytes;
	uint64_t scratch_bytes;
	SparkDeviceBuffer scratch_buffer;
	uint32_t recurrent_layers;
	uint32_t recurrent_pool_count;
	SparkDeviceStageRecurrentPool recurrent_pools[SPARK_DEVICE_STAGE_MAX_RECURRENT_POOLS];
	SparkTensor kv_pool;
	SparkTensor page_table;
	uint64_t kv_layer_stride;
	uint32_t pool_pages;
	uint32_t kv_attached;
	uint32_t step_exact;
	uint32_t verify_rows;
	SparkDeviceBuffer keep_tables;
	uint64_t routed_bytes;
	uint64_t routed_bytes_per_token;
} SparkDeviceStage;

typedef struct SparkDeviceStepTensors
{
	SparkTensor hidden;
	SparkTensor positions;
	SparkTensor sequence_of_row;
	SparkOpsSequences sequences;
	const uint32_t *host_positions;
	const uint32_t *host_sequence_of_row;
	uint32_t rows;
	SparkTensor row_identity;
	SparkTensor route_override;
	uint32_t route_override_width;
} SparkDeviceStepTensors;

typedef struct SparkDeviceKeepTensors
{
	SparkTensor kept_rows;
	SparkOpsSequences sequences;
	uint32_t rows;
	uint32_t frame_rows;
} SparkDeviceKeepTensors;

SparkStatus SparkDeviceStageRefuse(const SparkDeviceStage *stage, SparkStatus status, const char *what);
SparkStatus SparkDeviceStageOpen(SparkDeviceStage *stage, const SparkDeviceStageShape *shape, const SparkStageRunnerModelOpen *request);
SparkStatus SparkDeviceStageExpectConfig(SparkDeviceStage *stage, const SparkDeviceStageConfigValue *values, uint32_t count);
SparkStatus SparkDeviceStageReserve(SparkDeviceStage *stage, uint64_t scratch_bytes, uint32_t recurrent_layers, const uint64_t *recurrent_widths,
	uint32_t recurrent_pool_count, uint64_t state_budget_bytes);
SparkStatus SparkDeviceStageGeometry(SparkDeviceStage *stage, SparkStageRunnerModelGeometry *geometry);
SparkStatus SparkDeviceStageEntry(SparkDeviceStage *stage, const char *name, const char *kind, uint64_t *offset, uint64_t *bytes, uint32_t *rows);
SparkStatus SparkDeviceStageTensor(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, const char *name, const char *kind,
	uint64_t expected_bytes, SparkTensor *tensor);
SparkStatus SparkDeviceStageLayerTensor(SparkDeviceStage *stage, const SparkStageRunnerDeviceWeights *weights, uint32_t layer, const char *name,
	const char *kind, uint64_t expected_bytes, SparkTensor *tensor);
uint64_t SparkDeviceStagePlan(const SparkDeviceBuffer *buffer, const SparkDeviceStageSlice *slices, uint32_t count);
SparkStatus SparkDeviceStageAllocate(SparkDeviceStage *stage, const SparkDeviceStageSlice *slices, uint32_t count);
SparkTensor SparkDeviceStageRecurrent(const SparkDeviceStage *stage, uint32_t pool, uint32_t layer);
uint64_t SparkDeviceStageRecurrentBytes(const SparkDeviceStage *stage);
SparkStatus SparkDeviceStageStepTables(SparkDeviceStage *stage, const SparkStageRunnerStep *step, SparkDeviceStepTensors *tensors);
SparkStatus SparkDeviceStageKeepTables(SparkDeviceStage *stage, const SparkStageRunnerStep *step, const uint32_t *kept, SparkDeviceKeepTensors *keep);
SparkOpsPaged SparkDeviceStagePaged(const SparkDeviceStage *stage, const SparkDeviceStepTensors *tensors, uint32_t kv_layer);
void SparkDeviceStageLayerDone(const SparkDeviceStage *stage, uint32_t layer);

void SparkDeviceStageClose(void *model);
SparkStatus SparkDeviceStageManifestCheck(void *model, const struct SparkWeightdManifest *manifest);
SparkStatus SparkDeviceStageBindWeightd(void *model, struct SparkWeightdLazyPack *lazy_pack, SparkStageRunnerModelGeometry *geometry);
SparkStatus SparkDeviceStageSideband(void *model, uint32_t to_buffer, const void *hidden, void *buffer, uint64_t bytes, uint32_t rows,
	void *device_queue);
SparkStatus SparkDeviceStageAttachKv(void *model, const SparkStageRunnerKv *kv);
SparkStatus SparkDeviceStageRecurrentCopy(void *model, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *device_queue);
SparkStatus SparkDeviceStageResetSlot(void *model, uint32_t slot, void *device_queue);
const void *SparkDeviceStageProbeBuffers(void *model);
void SparkDeviceStageReport(void *model, uint32_t rank);
SparkStatus SparkDeviceStageLane(void *model, uint32_t lane, const SparkStageRunnerStep *step, const SparkStageRunnerServices *services,
	void **lane_model);

#ifdef __cplusplus
}
#endif

#endif
