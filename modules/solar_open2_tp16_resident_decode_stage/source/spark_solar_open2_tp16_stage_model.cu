#include "sparkpipe/spark_solar_open2_tp16_stage_model.h"
#include "ling_stage.cuh"

using SolarOpen2Stage = SolarOpen2Geometry<SPARK_SOLAR_OPEN2_TP16_STAGE_TP_DEGREE>;

static_assert(LingShape<SolarOpen2Stage>::Kv::kSlotBytes == SPARK_SOLAR_OPEN2_TP16_STAGE_KV_SLOT_BYTES, "a GQA slot is every KV head's key then value");
static_assert(LingShape<SolarOpen2Stage>::kAttnLayers == SPARK_SOLAR_OPEN2_TP16_STAGE_ATTN_LAYERS, "every fourth layer is GQA");
static_assert(LingShape<SolarOpen2Stage>::kGqaFused == SPARK_SOLAR_OPEN2_TP16_STAGE_QKV_WIDTH, "the fused projection is q, then k, then v");

const SparkStageRunnerModelInterface *SparkSolarOpen2StageModel(void)
{
	return LingModel<SolarOpen2Stage>::Interface();
}
