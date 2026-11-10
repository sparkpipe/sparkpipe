#ifndef DSV41_FLASH_CONTRACT_SHA256
#error "DSV41_FLASH_CONTRACT_SHA256 must identify the model contract"
#endif

#include "sparkpipe/spark_dsv41_flash_stage_model.h"
#include "dsv41_flash_stage.cuh"

const SparkStageRunnerModelInterface *SparkDsv41FlashStageModel(void)
{
	return &spark_dsv41_flash_stage_model;
}
