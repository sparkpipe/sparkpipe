#include "sparkpipe/spark_kexaone2_tp16_stage_model.h"
#include "gemma4_stage.cuh"

using Kexaone2Stage = Kexaone2Geometry<SPARK_KEXAONE2_TP16_STAGE_TP_DEGREE>;

static_assert(GemmaShape<Kexaone2Stage>::kFullLayers == SPARK_KEXAONE2_TP16_STAGE_FULL_LAYERS &&
	GemmaShape<Kexaone2Stage>::kSlidingLayers == SPARK_KEXAONE2_TP16_STAGE_SLIDING_LAYERS,
	"layer 0 and every layer 4k+1 from 5 attend globally: 20 full and 58 sliding layers");
static_assert(GemmaShape<Kexaone2Stage>::SlidingKvGeometry::kSlotBytes == SPARK_KEXAONE2_TP16_STAGE_SLOT_BYTES &&
	GemmaShape<Kexaone2Stage>::FullKvGeometry::kSlotBytes == SPARK_KEXAONE2_TP16_STAGE_SLOT_BYTES, "a KV slot is [K: 8 x 128][V: 8 x 128] bf16");
static_assert(GemmaShape<Kexaone2Stage>::kSlidingFused == SPARK_KEXAONE2_TP16_STAGE_FUSED_WIDTH &&
	GemmaShape<Kexaone2Stage>::kFullFused == SPARK_KEXAONE2_TP16_STAGE_FUSED_WIDTH, "q|k|v is 64 + 2 x 8 heads of 128 on every layer");
static_assert(SPARK_KEXAONE2_TP16_STAGE_FUSED_WIDTH % SPARK_KEXAONE2_TP16_STAGE_TP_DEGREE == 0u, "fused attention rows split evenly over the ranks");
static_assert(SPARK_KEXAONE2_TP16_STAGE_HEADS % SPARK_KEXAONE2_TP16_STAGE_KV_HEADS == 0u &&
	SPARK_KEXAONE2_TP16_STAGE_HEADS / SPARK_KEXAONE2_TP16_STAGE_KV_HEADS <= LM_GQA_SHARD_MAX_GROUP, "whole GQA groups of 8 query heads per KV head");

const SparkStageRunnerModelInterface *SparkKexaone2Tp16StageModel(void)
{
	return GemmaModel<Kexaone2Stage>::Interface();
}
