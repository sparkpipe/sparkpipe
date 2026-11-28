#ifndef SPARKPIPE_SPARK_STAGE_SERVING_ADAPTER_H
#define SPARKPIPE_SPARK_STAGE_SERVING_ADAPTER_H

#include <stdint.h>

#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_stage_runner.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_STAGE_SERVING_MAX_LANES 128u
#define SPARK_STAGE_SERVING_MAX_ROWS 16384u
#define SPARK_STAGE_SERVING_MULTIPROCESSORS 48u
#define SPARK_STAGE_SERVING_CHAIN_TOKENS SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE
#define SPARK_STAGE_SERVING_DRAFT_TOKENS 15u

typedef struct SparkStageServingModel
{
	const char *adapter_id;
	const char *model_id;
	const char *model_revision;
	const char *program;
	const char *contract_sha256;
	const char *module_tag;
	const SparkStageRunnerModelInterface *(*runner_model)(void);
	uint32_t tp_degree;
	uint32_t layer_count;
	uint32_t hidden;
	uint32_t linear_weight_codec;
	uint32_t expert_weight_codec;
	uint32_t max_rows;
	uint32_t verify_rows;
	uint32_t tree_slot_grain;
	uint32_t speculative_tokens;
	uint32_t cache_block_token_count;
	uint32_t cache_checkpoint_token_count;
	void (*kv_layout)(uint32_t tp_rank, uint32_t tp_degree, SparkStageKvConfiguration *kv);
	uint32_t (*collective_sequences)(uint32_t rows, uint32_t tp_degree);
} SparkStageServingModel;

const SparkModelServingAdapterInterface *SparkStageServingAdapterInterface(const SparkStageServingModel *model);
const SparkModelServingAdapterDescriptor *SparkStageServingDescriptor(void);

static inline void SparkStageKvSetShard(SparkStageKvConfiguration *kv, uint32_t degree, uint32_t rank, uint32_t grain)
{
	kv->context_shard.degree = degree;
	kv->context_shard.rank = rank;
	kv->context_shard.grain = grain;
}

static inline void SparkStageKvSetRegion2(SparkStageKvConfiguration *kv, uint32_t layout, uint64_t layer_page_bytes, uint32_t layer_count)
{
	kv->region_count = SPARK_STAGE_KV_MAX_REGIONS;
	kv->regions[1].layout = layout;
	kv->regions[1].layer_page_bytes = layer_page_bytes;
	kv->regions[1].layer_count = layer_count;
}

static inline void SparkStageKvInitFull(SparkStageKvConfiguration *kv, uint32_t tp_rank, uint32_t tp_degree, uint64_t page_bytes,
	uint32_t layer_count, uint32_t head_count, uint32_t qk_dim, uint32_t v_dim)
{
	kv->regions[0].layout = SPARK_STAGE_KV_REGION_LAYER_MAJOR;
	kv->regions[0].layer_page_bytes = page_bytes;
	kv->arena_kv_head_count = head_count;
	kv->arena_head_dim = qk_dim + v_dim;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
	kv->capacity_request.layer_count = layer_count;
	kv->capacity_request.head_count = head_count;
	kv->capacity_request.query_key_head_dimension = qk_dim;
	kv->capacity_request.value_head_dimension = v_dim;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	SparkStageKvSetShard(kv, tp_degree, tp_rank, 1u);
}

static inline void SparkStageKvInitCompressed(SparkStageKvConfiguration *kv, uint32_t tp_rank, uint32_t tp_degree, uint32_t layout0,
	uint64_t page_bytes, uint32_t arena_head_dim, uint32_t layer_count, uint32_t latent_dim, uint32_t rope_dim, uint32_t fp8_scale_block)
{
	kv->regions[0].layout = layout0;
	kv->regions[0].layer_page_bytes = page_bytes;
	kv->arena_kv_head_count = 1u;
	kv->arena_head_dim = arena_head_dim;
	kv->arena_bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	kv->capacity_request.descriptor_bytes = SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	kv->capacity_request.layout = SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE;
	kv->capacity_request.layer_count = layer_count;
	kv->capacity_request.compressed_dimension = latent_dim;
	kv->capacity_request.position_dimension = rope_dim;
	kv->capacity_request.bytes_per_scalar = sizeof(uint16_t);
	kv->capacity_request.fp8_scale_block_size = fp8_scale_block;
	SparkStageKvSetShard(kv, tp_degree, tp_rank, 1u);
}

static inline void SparkStageKvSetIndex(SparkStageKvConfiguration *kv, uint64_t page_bytes, uint32_t layer_count, uint32_t dim,
	uint32_t bytes_per_scalar)
{
	SparkStageKvSetRegion2(kv, SPARK_STAGE_KV_REGION_LAYER_MAJOR, page_bytes, layer_count);
	kv->capacity_request.index_key_layer_count = layer_count;
	kv->capacity_request.index_key_dimension = dim;
	kv->capacity_request.index_key_bytes_per_scalar = bytes_per_scalar;
}

static inline uint32_t SparkStageCollectiveWidthRows(uint32_t rows, uint32_t width, uint32_t hidden)
{
	const uint64_t gathered = ((uint64_t)rows * width + hidden - 1u) / hidden;
	return (uint32_t)(gathered > rows ? gathered : rows) + 1u;
}

static inline uint32_t SparkStageCollectiveSameRows(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return rows;
}

static inline uint32_t SparkStageCollectiveDoubleRows(uint32_t rows, uint32_t tp_degree)
{
	(void)tp_degree;
	return 2u * rows + 1u;
}

#define SPARK_STAGE_SERVING_EXPORT(model) \
	const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void) { return SparkStageServingAdapterInterface(&(model)); }

#ifdef __cplusplus
}
#endif

#endif
