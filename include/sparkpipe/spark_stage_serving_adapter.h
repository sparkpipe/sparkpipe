#ifndef SPARKPIPE_SPARK_STAGE_SERVING_ADAPTER_H
#define SPARKPIPE_SPARK_STAGE_SERVING_ADAPTER_H

#include <stdint.h>

#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_stage_runner.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_STAGE_SERVING_MAX_LANES 128u
#define SPARK_STAGE_SERVING_MAX_ROWS 16384u
#define SPARK_STAGE_SERVING_MULTIPROCESSORS 48u
#define SPARK_STAGE_SERVING_CHAIN_TOKENS SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE
#define SPARK_STAGE_SERVING_DRAFT_TOKENS 15u

typedef struct SparkStageServingModel
{
	const char *adapter_id;
	const char *model_id;
	const char *model_revision;
	const char *program;
	const char *contract_sha256;
	const char *module_tag;
	const SparkStageRunnerModelInterface *(*runner_model)(void);
	uint32_t tp_degree;
	uint32_t layer_count;
	uint32_t hidden;
	uint32_t linear_weight_codec;
	uint32_t expert_weight_codec;
	uint32_t max_rows;
	uint32_t verify_rows;
	uint32_t tree_slot_grain;
	uint32_t speculative_tokens;
	uint32_t cache_block_token_count;
	uint32_t cache_checkpoint_token_count;
	void (*kv_layout)(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv);
	uint32_t (*collective_sequences)(uint32_t rows, uint32_t tp_degree);
} SparkStageServingModel;

const SparkModelServingAdapterInterface *SparkStageServingAdapterInterface(const SparkStageServingModel *model);
const SparkModelServingAdapterDescriptor *SparkStageServingDescriptor(void);

#ifdef __cplusplus
}
#endif

#endif
