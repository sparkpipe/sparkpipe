#ifndef SPARKPIPE_SPARK_K3_STAGE_MODEL_H
#define SPARKPIPE_SPARK_K3_STAGE_MODEL_H

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_K3_RESIDUAL_BANK_BYTES_PER_ROW (9u * 7168u * 2u)

const SparkStageRunnerModelInterface *SparkK3StageModel(void);

#ifdef __cplusplus
}
#endif

#endif
