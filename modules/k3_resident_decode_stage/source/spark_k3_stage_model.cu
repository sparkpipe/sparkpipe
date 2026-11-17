#include "sparkpipe/spark_k3_stage_model.h"
#include "model-families/k3/source/k3_stage.cuh"

const SparkStageRunnerModelInterface *SparkK3StageModel(void)
{
	return &spark_k3_stage_model;
}
