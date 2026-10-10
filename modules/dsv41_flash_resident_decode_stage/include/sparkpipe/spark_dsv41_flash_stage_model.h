#pragma once

#include "sparkpipe/spark_stage_runner_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DSV41_FLASH_STAGE_TP_DEGREE 16u
#define SPARK_DSV41_FLASH_STAGE_HIDDEN 5120u
#define SPARK_DSV41_FLASH_STAGE_PAGE_TOKENS 256u
#define SPARK_DSV41_FLASH_STAGE_SLOT_BYTES 1280u
#define SPARK_DSV41_FLASH_STAGE_RATIO2_CACHES 3u
#define SPARK_DSV41_FLASH_STAGE_RATIO1_CACHES 1u
#define SPARK_DSV41_FLASH_STAGE_SHARD_GRAIN 8u
#define SPARK_DSV41_FLASH_STAGE_A2A_HEADS 8u
#define SPARK_DSV41_FLASH_STAGE_RECORD_FLOATS 514u

const SparkStageRunnerModelInterface *SparkDsv41FlashStageModel(void);

#ifdef __cplusplus
}
#endif
