#include "sparkpipe/spark_qwen38_27b_stage_model.h"
#include "sparkpipe/spark_stage_serving_adapter.h"

#ifndef QWEN38_27B_CONTRACT_SHA256
#error "QWEN38_27B_CONTRACT_SHA256 must identify the model contract"
#endif

#if defined(QWEN38_27B_DEVICE_STAGE)
#include "inference/runner/device_stage_model.h"
#include "inference/runner/device_stage_runner.h"
#include "sparkpipe/spark_qwen38_device_stage.h"
#define SPARK_QWEN38_27B_SERVING_ADAPTER_ID "qwen38_27b-bf16-tp1"
#define SPARK_QWEN38_27B_SERVING_MODULE_TAG "qwen38_27b_stage"
#define SPARK_QWEN38_27B_SERVING_LINEAR_CODEC SPARK_WEIGHT_CODEC_BF16
#define SPARK_QWEN38_27B_SERVING_TP 1u
#define SPARK_QWEN38_27B_SERVING_VERIFY_ROWS SPARK_DEVICE_STAGE_VERIFY_ROWS
#define SPARK_QWEN38_27B_SERVING_SPECULATIVE_TOKENS (SPARK_DEVICE_STAGE_VERIFY_ROWS - 1u)
#define SPARK_QWEN38_27B_SERVING_TREE_SLOT_GRAIN SPARK_DEVICE_RUNNER_TREE_SLOT_GRAIN
#define SPARK_QWEN38_27B_SERVING_RUNNER_MODEL SparkQwen38_27bDeviceStageModel
#else
#if !defined(QWEN38_27B_TP_DEGREE) || !defined(QWEN38_27B_LINEAR_CODEC) || !defined(QWEN38_27B_ADAPTER_ID) || !defined(QWEN38_27B_MODULE_TAG)
#error "QWEN38_27B_TP_DEGREE, QWEN38_27B_LINEAR_CODEC, QWEN38_27B_ADAPTER_ID and QWEN38_27B_MODULE_TAG must name the rank count, weight codec, adapter and KV label of this build"
#endif
#define SPARK_QWEN38_27B_SERVING_ADAPTER_ID QWEN38_27B_ADAPTER_ID
#define SPARK_QWEN38_27B_SERVING_MODULE_TAG QWEN38_27B_MODULE_TAG
#define SPARK_QWEN38_27B_SERVING_LINEAR_CODEC QWEN38_27B_LINEAR_CODEC
#define SPARK_QWEN38_27B_SERVING_TP QWEN38_27B_TP_DEGREE
#define SPARK_QWEN38_27B_SERVING_VERIFY_ROWS SPARK_QWEN38_27B_STAGE_VERIFY_ROWS
#define SPARK_QWEN38_27B_SERVING_SPECULATIVE_TOKENS (SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE - 1u)
#define SPARK_QWEN38_27B_SERVING_TREE_SLOT_GRAIN SPARK_QWEN38_27B_SERVING_TP
#define SPARK_QWEN38_27B_SERVING_RUNNER_MODEL SparkQwen38_27bStageModel
#endif

static void Qwen38_27bServingKvLayout(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv)
{
	SparkStageKvInitFull(kv, tp_rank, tp_degree, (uint64_t)SPARK_QWEN38_27B_STAGE_PAGE_SLOTS * SPARK_QWEN38_27B_STAGE_KV_SLOT_BYTES,
		16u, SPARK_QWEN38_27B_STAGE_KV_HEADS, SPARK_QWEN38_27B_STAGE_HEAD_DIM, SPARK_QWEN38_27B_STAGE_HEAD_DIM);
}

static uint32_t Qwen38_27bServingCollectiveSequences(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return SparkStageCollectiveWidthRows(rows, SPARK_QWEN38_27B_STAGE_QKV_WIDTH, SPARK_QWEN38_27B_STAGE_HIDDEN);
}

static const SparkStageServingModel Qwen38_27bStageServing =
{
	.adapter_id = SPARK_QWEN38_27B_SERVING_ADAPTER_ID,
	.model_id = "Qwen/Qwen3.8-27B",
	.model_revision = SPARK_QWEN38_27B_SERVING_ADAPTER_ID,
	.program = "qwen38_27b",
	.contract_sha256 = QWEN38_27B_CONTRACT_SHA256,
	.module_tag = SPARK_QWEN38_27B_SERVING_MODULE_TAG,
	.runner_model = SPARK_QWEN38_27B_SERVING_RUNNER_MODEL,
	.tp_degree = SPARK_QWEN38_27B_SERVING_TP,
	.layer_count = 64u,
	.hidden = SPARK_QWEN38_27B_STAGE_HIDDEN,
	.linear_weight_codec = SPARK_QWEN38_27B_SERVING_LINEAR_CODEC,
	.expert_weight_codec = SPARK_QWEN38_27B_SERVING_LINEAR_CODEC,
	.max_rows = 2048u,
	.verify_rows = SPARK_QWEN38_27B_SERVING_VERIFY_ROWS,
	.tree_slot_grain = SPARK_QWEN38_27B_SERVING_TREE_SLOT_GRAIN,
	.speculative_tokens = SPARK_QWEN38_27B_SERVING_SPECULATIVE_TOKENS,
	.cache_block_token_count = SPARK_QWEN38_27B_STAGE_PAGE_SLOTS,
	.cache_checkpoint_token_count = 16384u,
	.kv_layout = Qwen38_27bServingKvLayout,
	.collective_sequences = Qwen38_27bServingCollectiveSequences,
};

SPARK_STAGE_SERVING_EXPORT(Qwen38_27bStageServing)
