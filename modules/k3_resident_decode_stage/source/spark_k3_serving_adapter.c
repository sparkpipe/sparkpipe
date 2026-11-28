#include "sparkpipe/spark_k3_kv_geometry.h"
#include "sparkpipe/spark_k3_llm_defines.h"
#include "sparkpipe/spark_k3_model.h"
#include "sparkpipe/spark_k3_serving_adapter.h"
#include "sparkpipe/spark_k3_stage_model.h"
#include "sparkpipe/spark_k3_tp_sequences.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#if !defined(SPARK_K3_SERVING_INDEXER) || (SPARK_K3_SERVING_INDEXER != 0 && SPARK_K3_SERVING_INDEXER != 1)
#error "SPARK_K3_SERVING_INDEXER must name the packs this build serves: 0 for Kimi K3, 1 for Kimi K3.1 Pro (the MLA indexer)"
#endif
#if SPARK_K3_SERVING_INDEXER == 1
#if !defined(K31_PRO_CONTRACT_SHA256)
#error "a Kimi K3.1 Pro build names its pack contract in K31_PRO_CONTRACT_SHA256"
#endif
#define SPARK_K3_SERVING_ADAPTER_ID "k31-pro-tp16"
#define SPARK_K3_SERVING_SOURCE_ID "moonshotai/Kimi-K3.1-Pro"
#define SPARK_K3_SERVING_MODULE_TAG "k31_stage"
#define SPARK_K3_SERVING_KV_REGIONS SPARK_STAGE_KV_MAX_REGIONS
#define SPARK_K3_SERVING_CONTRACT_SHA256 K31_PRO_CONTRACT_SHA256
#else
#define SPARK_K3_SERVING_ADAPTER_ID "k3-tp16"
#define SPARK_K3_SERVING_SOURCE_ID SPARK_K3_MODEL_SOURCE_ID
#define SPARK_K3_SERVING_MODULE_TAG "k3_stage"
#define SPARK_K3_SERVING_KV_REGIONS 1u
#define SPARK_K3_SERVING_CONTRACT_SHA256 "318d979200eb3c6784be6f932febe14832b48df53a1520a73af2f03bd39bb217"
#endif
#define SPARK_K3_SERVING_TP_DEGREE 16u

static void K3ServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = (uint64_t)SPARK_K3_KV_PAGE_SLOTS *
		SPARK_K3_MODEL_MLA_KV_A_DIMENSION * SPARK_K3_KV_BYTES_PER_SCALAR;
	kv->arena_kv_head_count = 1u;
	kv->arena_head_dim = SPARK_K3_MODEL_MLA_KV_A_DIMENSION;
	kv->arena_bytes_per_scalar = SPARK_K3_KV_BYTES_PER_SCALAR;
	SparkK3KvFillCapacityRequest(&kv->capacity_request);
	kv->region_count = SPARK_K3_SERVING_KV_REGIONS;
	if ( SPARK_K3_SERVING_KV_REGIONS > 1u )
		SparkStageKvSetIndex(kv, (uint64_t)SPARK_K3_KV_PAGE_SLOTS * SPARK_K3_KV_INDEX_KEY_DIMENSION * SPARK_K3_KV_INDEX_KEY_BYTES_PER_SCALAR,
			SPARK_K3_KV_MLA_LAYER_COUNT, SPARK_K3_KV_INDEX_KEY_DIMENSION, SPARK_K3_KV_INDEX_KEY_BYTES_PER_SCALAR);
	if ( tp_degree > 1u )
		kv->context_shard = SparkK3KvShardContext(tp_rank, tp_degree);
}

static const SparkStageServingModel K3StageServing =
{
	.adapter_id = SPARK_K3_SERVING_ADAPTER_ID,
	.model_id = SPARK_K3_SERVING_SOURCE_ID,
	.model_revision = SPARK_K3_SERVING_ADAPTER_ID,
	.program = "k3",
	.contract_sha256 = SPARK_K3_SERVING_CONTRACT_SHA256,
	.module_tag = SPARK_K3_SERVING_MODULE_TAG,
	.runner_model = SparkK3StageModel,
	.tp_degree = 16u,
	.layer_count = SPARK_K3_MODEL_LAYER_COUNT,
	.hidden = SPARK_K3_MODEL_HIDDEN_DIMENSION,
	.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16,
	.expert_weight_codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1,
	.max_rows = 4096u,
	.verify_rows = SPARK_K3_SERVING_VERIFY_ROWS,
	.tree_slot_grain = 16u,
	.speculative_tokens = SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u,
	.cache_block_token_count = SPARK_K3_KV_PAGE_SLOTS,
	.cache_checkpoint_token_count = SPARK_K3_KV_CHECKPOINT_TOKENS,
	.kv_layout = K3ServingKvLayout,
	.collective_sequences = SparkK3TpSequenceCapacity,
};

SPARK_STAGE_SERVING_EXPORT(K3StageServing)
