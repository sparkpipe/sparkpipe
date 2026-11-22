#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_qwen3_embedding_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#if !defined(QWEN3_EMBEDDING_CONTRACT_SHA256) || !defined(QWEN3_EMBEDDING_TP_DEGREE)
#error "QWEN3_EMBEDDING_CONTRACT_SHA256 and QWEN3_EMBEDDING_TP_DEGREE must name the contract and the rank count of this build"
#endif
#if QWEN3_EMBEDDING_TP_DEGREE != 1u
#error "the Qwen3 embedding stage is built for one rank"
#endif

#define SPARK_QWEN3_EMBEDDING_8B_SERVING_ADAPTER_ID "qwen3_embedding_8b-bf16-tp1"
#define SPARK_QWEN3_EMBEDDING_8B_SERVING_MODULE_TAG "qwen3_embedding_8b_tp1_stage"
#define SPARK_QWEN3_EMBEDDING_8B_SERVING_MODEL_ID "Qwen/Qwen3-Embedding-8B"
#define SPARK_QWEN3_EMBEDDING_8B_SERVING_MAX_ROWS 4096u

static void Qwen3Embedding8bServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_QWEN3_EMBEDDING_8B_STAGE_PAGE_SLOTS * SPARK_QWEN3_EMBEDDING_8B_STAGE_KV_SLOT_BYTES;
	kv->arena_kv_head_count = SPARK_QWEN3_EMBEDDING_8B_STAGE_KV_HEADS;
	kv->arena_head_dim = 2u * SPARK_QWEN3_EMBEDDING_8B_STAGE_HEAD_DIM;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = SPARK_QWEN3_EMBEDDING_8B_STAGE_LAYERS;
	kv->capacity_request.head_count = SPARK_QWEN3_EMBEDDING_8B_STAGE_KV_HEADS;
	kv->capacity_request.query_key_head_dimension = SPARK_QWEN3_EMBEDDING_8B_STAGE_HEAD_DIM;
	kv->capacity_request.value_head_dimension = SPARK_QWEN3_EMBEDDING_8B_STAGE_HEAD_DIM;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->context_shard.degree = tp_degree;
	kv->context_shard.rank = tp_rank;
	kv->context_shard.grain = 1u;
}

static uint32_t Qwen3Embedding8bServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	const uint64_t gathered = ((uint64_t)rows * SPARK_QWEN3_EMBEDDING_8B_STAGE_QKV_WIDTH + SPARK_QWEN3_EMBEDDING_8B_STAGE_HIDDEN - 1u) /
		SPARK_QWEN3_EMBEDDING_8B_STAGE_HIDDEN;
	(void)tp_degree;
	return (uint32_t)(gathered > rows ? gathered : rows) + 1u;
}

static const SparkStageServingModel Qwen3Embedding8bStageServing =
{
	.adapter_id = SPARK_QWEN3_EMBEDDING_8B_SERVING_ADAPTER_ID,
	.model_id = SPARK_QWEN3_EMBEDDING_8B_SERVING_MODEL_ID,
	.model_revision = SPARK_QWEN3_EMBEDDING_8B_SERVING_ADAPTER_ID,
	.program = "qwen3_embedding_8b",
	.contract_sha256 = QWEN3_EMBEDDING_CONTRACT_SHA256,
	.module_tag = SPARK_QWEN3_EMBEDDING_8B_SERVING_MODULE_TAG,
	.runner_model = SparkQwen3Embedding8bStageModel,
	.tp_degree = QWEN3_EMBEDDING_TP_DEGREE,
	.layer_count = SPARK_QWEN3_EMBEDDING_8B_STAGE_LAYERS,
	.hidden = SPARK_QWEN3_EMBEDDING_8B_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.max_rows = SPARK_QWEN3_EMBEDDING_8B_SERVING_MAX_ROWS,
	.verify_rows = 0u,
	.tree_slot_grain = 0u,
	.speculative_tokens = 0u,
	.cache_block_token_count = SPARK_QWEN3_EMBEDDING_8B_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 0u,
	.kv_layout = Qwen3Embedding8bServingKvLayout,
	.collective_sequences = Qwen3Embedding8bServingCollectiveSequences,
	.row_output = 1u,
	.rows_only = 1u,
};

const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return SparkStageServingAdapterInterface(&Qwen3Embedding8bStageServing);
}
