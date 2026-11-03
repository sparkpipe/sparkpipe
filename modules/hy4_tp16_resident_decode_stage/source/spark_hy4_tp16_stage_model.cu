#include "sparkpipe/spark_hy4_tp16_stage_model.h"
#include "hy4_stage.cuh"

using Hy4Stage = Hy4Geometry<SPARK_HY4_TP16_STAGE_TP_DEGREE>;

static_assert(HyShape<Hy4Stage>::Kv::kSlotBytes == SPARK_HY4_TP16_STAGE_KV_SLOT_BYTES, "a latent slot is the normed latent and its rope part");
static_assert(HyShape<Hy4Stage>::IndexKv::kSlotBytes == SPARK_HY4_TP16_STAGE_INDEX_SLOT_BYTES, "an index slot is one indexer key");

const SparkStageRunnerModelInterface *SparkHy4Tp16StageModel(void)
{
	return HyModel<Hy4Stage>::Interface();
}
