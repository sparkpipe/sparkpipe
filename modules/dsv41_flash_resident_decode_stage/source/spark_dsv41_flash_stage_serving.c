#include "sparkpipe/spark_dsv41_flash_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#if !defined(DSV41_FLASH_CONTRACT_SHA256) || !defined(DSV41_FLASH_ADAPTER_ID) || !defined(DSV41_FLASH_MODULE_TAG)
#error "DSV41_FLASH_CONTRACT_SHA256, DSV41_FLASH_ADAPTER_ID and DSV41_FLASH_MODULE_TAG must name the model contract, the adapter and the KV label of this build"
#endif

#define SPARK_DSV41_FLASH_TP16_MAX_ROWS (SPARK_DSV41_FLASH_STAGE_KV_DEGREE == 16u ? 2048u : SPARK_STAGE_SERVING_MAX_ROWS)
#define SPARK_DSV41_FLASH_TP16_ENGRAM_WIDTH (24u * 256u)

static void Dsv41FlashServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	const uint32_t caches = SPARK_DSV41_FLASH_STAGE_RATIO2_CACHES + SPARK_DSV41_FLASH_STAGE_RATIO1_CACHES;
	SparkStageKvInitCompressed(kv, tp_rank, tp_degree, SPARK_STAGE_KV_REGION_LAYER_MAJOR,
		(uint64_t)(SPARK_DSV41_FLASH_STAGE_PAGE_TOKENS / 2u) * SPARK_DSV41_FLASH_STAGE_SLOT_BYTES,
		SPARK_DSV41_FLASH_STAGE_SLOT_BYTES / 2u, caches, 512u, 0u, 0u);
	SparkStageKvSetRegion2(kv, SPARK_STAGE_KV_REGION_LAYER_MAJOR,
		(uint64_t)SPARK_DSV41_FLASH_STAGE_PAGE_TOKENS * SPARK_DSV41_FLASH_STAGE_SLOT_BYTES, SPARK_DSV41_FLASH_STAGE_RATIO1_CACHES);
	kv->capacity_request.index_key_layer_count = caches;
	kv->capacity_request.index_key_dimension = 128u;
	SparkStageKvSetShard(kv, tp_degree == 16u ? SPARK_DSV41_FLASH_STAGE_KV_DEGREE : 1u,
		tp_degree == 16u && SPARK_DSV41_FLASH_STAGE_KV_DEGREE == 16u ? tp_rank : 0u, SPARK_DSV41_FLASH_STAGE_SHARD_GRAIN);
}

static uint32_t Dsv41FlashServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t engram = ((uint64_t)rows * SPARK_DSV41_FLASH_TP16_ENGRAM_WIDTH + SPARK_DSV41_FLASH_STAGE_HIDDEN - 1u) /
		SPARK_DSV41_FLASH_STAGE_HIDDEN;
	const uint64_t split = 2u * (uint64_t)(rows < SPARK_DSV41_FLASH_STAGE_ENGRAM_SPLIT_ROWS ? rows : SPARK_DSV41_FLASH_STAGE_ENGRAM_SPLIT_ROWS);
	const uint64_t landing = tp_degree == 16u && SPARK_DSV41_FLASH_STAGE_KV_DEGREE == 16u ? SPARK_DSV41_FLASH_STAGE_LANDING_ROWS(SPARK_DSV41_FLASH_STAGE_MAX_POSITIONS, 1u) : 0u;
	const uint64_t parts = (uint64_t)rows * SPARK_DSV41_FLASH_STAGE_ENGRAM_KV_PARTS;
	const uint64_t own_step = SPARK_DSV41_FLASH_STAGE_OWN_ROWS_STEP;
	const uint64_t owned = SPARK_DSV41_FLASH_STAGE_KV_DEGREE == 1u ? 2u * ((rows + own_step - 1u) / own_step) * own_step : 0u;
	uint64_t capacity = engram > parts ? engram : parts;
	capacity = split > capacity ? split : capacity;
	capacity = owned > capacity ? owned : capacity;
	capacity = landing > capacity ? landing : capacity;
	return (uint32_t)capacity + 1u;
}

static const SparkStageServingModel Dsv41FlashStageServing =
{
	.adapter_id = DSV41_FLASH_ADAPTER_ID,
	.model_id = "deepseek-ai/DeepSeek-V4.1-Flash",
	.model_revision = DSV41_FLASH_ADAPTER_ID,
	.program = "dsv41_flash",
	.contract_sha256 = DSV41_FLASH_CONTRACT_SHA256,
	.module_tag = DSV41_FLASH_MODULE_TAG,
	.runner_model = SparkDsv41FlashStageModel,
	.tp_degree = SPARK_DSV41_FLASH_STAGE_TP_DEGREE,
	.layer_count = 40u,
	.hidden = SPARK_DSV41_FLASH_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1,
	.max_rows = SPARK_DSV41_FLASH_TP16_MAX_ROWS,
	.verify_rows = SPARK_DSV41_FLASH_STAGE_VERIFY_ROWS,
	.tree_slot_grain = (SPARK_DSV41_FLASH_STAGE_KV_DEGREE > 1u ? 2u : 1u) * SPARK_DSV41_FLASH_STAGE_SHARD_GRAIN * SPARK_DSV41_FLASH_STAGE_KV_DEGREE,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_DSV41_FLASH_STAGE_PAGE_TOKENS,
	.cache_checkpoint_token_count = 1048576u,
	.kv_layout = Dsv41FlashServingKvLayout,
	.collective_sequences = Dsv41FlashServingCollectiveSequences,
};

SPARK_STAGE_SERVING_EXPORT(Dsv41FlashStageServing)
