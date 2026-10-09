#ifndef SPARKPIPE_SPARK_STAGE_SERVING_ADAPTER_H
#define SPARKPIPE_SPARK_STAGE_SERVING_ADAPTER_H

#include <stdint.h>

#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_stage_runner.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_STAGE_SERVING_MAX_LANES 16u
#define SPARK_STAGE_SERVING_MAX_ROWS 2048u
#define SPARK_STAGE_SERVING_MULTIPROCESSORS 48u

typedef struct SparkStageServingModel
{
	const SparkModelServingAdapterDescriptor *descriptor;
	const SparkStageRunnerModelInterface *(*runner_model)(void);
	const char *module_tag;
	uint32_t tp_degree;
	uint32_t program_id;
	void (*kv_layout)(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv);
	uint32_t (*collective_sequences)(uint32_t rows, uint32_t tp_degree);
} SparkStageServingModel;

const SparkModelServingAdapterInterface *SparkStageServingAdapterInterface(const SparkStageServingModel *model);

#ifdef __cplusplus
}
#endif

#endif
