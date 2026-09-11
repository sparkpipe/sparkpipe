#pragma once

#include <stdint.h>

#include "sparkpipe/spark_hy4_model.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* hy4 TP16 resident decode stage: firmware API. One submission executes
 * one decode step (or one prefill frame) as a stream-ordered sequence:
 * per-layer hyper-connection pre -> attention (MLA with learnable
 * sinks, interleaved rope at the token position) -> hc distribute ->
 * hyper-connection pre -> routed experts (top-8 of 256, sigmoid gating
 * with e_score correction bias, shared expert, swiglu clamp 10) ->
 * hc distribute. The lightning indexer feeds sparse token selection on
 * its active layers (0, 1 and every 4th). The final stage collapses
 * the hc streams (hc_head fn/scale/base + weighted reduce), applies
 * output norm and the rank-local lm_head, and argmaxes across ranks
 * through the device collective. */

#define SPARK_HY4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCES 64u
#define SPARK_HY4_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS 64u
#define SPARK_HY4_RESIDENT_DECODE_STAGE_LAYERS \
	SPARK_HY4_MODEL_LAYER_COUNT
#define SPARK_HY4_RESIDENT_DECODE_STAGE_LOCAL_QUERY_HEADS \
	SPARK_HY4_MODEL_ATTN_QUERY_HEADS_PER_RANK
#define SPARK_HY4_RESIDENT_DECODE_STAGE_LOCAL_INDEX_HEADS \
	SPARK_HY4_MODEL_INDEX_HEADS_PER_RANK
#define SPARK_HY4_RESIDENT_DECODE_STAGE_LOCAL_EXPERTS \
	SPARK_HY4_MODEL_EXPERTS_PER_RANK
#define SPARK_HY4_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_COUNT 2u
#define SPARK_HY4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
/* Weight format codes (mirrored in spark_hy4_stagepack_format.h with
 * _Static_assert equality against the shared stagepack format). */
#define SPARK_HY4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 1u
#define SPARK_HY4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 0u
#define SPARK_HY4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B32 9u

/* Per-layer attention weights (rank-local). All pointers are device
 * addresses published by the pack loader; fp8 planes carry their E8M0
 * group-32 scales adjacent (payload plane, scale plane). */
typedef struct SparkHy4AttnLayerWeights
{
	const void *attn_norm_f32;
	const void *sinks_f32;
	const void *q_a_fp8;
	const void *q_a_scale_u8;
	const void *q_a_norm_f32;
	const void *q_b_fp8;
	const void *q_b_scale_u8;
	const void *kv_a_fp8;
	const void *kv_a_scale_u8;
	const void *kv_a_norm_f32;
	const void *k_b_fp8;
	const void *k_b_scale_u8;
	const void *v_b_fp8;
	const void *v_b_scale_u8;
	const void *gate_bf16;
	const void *output_fp8;
	const void *output_scale_u8;
	const void *index_q_bf16;
	const void *index_q_norm_bf16;
	const void *index_k_norm_bf16;
	const void *index_wq_b_fp8;
	const void *index_wq_b_scale_u8;
	const void *index_wk_fp8;
	const void *index_wk_scale_u8;
	const void *index_proj_bf16;
	const void *index_norm_f32;
} SparkHy4AttnLayerWeights;

/* Per-layer hyper-connection coefficients. fn is [2*HC, HC*hidden]
 * f32; scale is 2 f32; base is 2*HC f32. The attention branch and the
 * mlp branch each carry one triple. */
typedef struct SparkHy4HyperWeights
{
	const void *fn_f32;
	const void *scale_f32;
	const void *base_f32;
} SparkHy4HyperWeights;

/* Per-layer mlp weights: dense layer 0 uses the shared-expert planes
 * alone (gate/up/down, fp8+scales); MoE layers add the router, its
 * bias, the routed planes ([16 local experts, ...]) and the shared
 * expert. */
typedef struct SparkHy4MoeWeights
{
	const void *mlp_norm_f32;
	const void *router_f32;
	const void *router_bias_f32;
	const void *routed_gate_up_fp8;
	const void *routed_gate_up_scale_u8;
	const void *routed_down_fp8;
	const void *routed_down_scale_u8;
	const void *shared_gate_fp8;
	const void *shared_gate_scale_u8;
	const void *shared_up_fp8;
	const void *shared_up_scale_u8;
	const void *shared_down_fp8;
	const void *shared_down_scale_u8;
	const void *hc_mlp_fn_f32;
	const void *hc_mlp_scale_f32;
	const void *hc_mlp_base_f32;
} SparkHy4MoeWeights;

typedef struct SparkHy4HeadWeights
{
	const void *output_hc_fn_f32;
	const void *output_hc_scale_f32;
	const void *output_hc_base_f32;
	const void *output_norm_f32;
	const void *lm_head_bf16;
	const void *embedding_bf16;
} SparkHy4HeadWeights;

/* Per-pipeline-slot device scratch. Sizes derive from the geometry
 * macros; the module allocates and owns these for its lifetime. */
typedef struct SparkHy4PipelineSlot
{
	void *cuda_stream;
	uint32_t *host_row_positions;
	uint32_t *host_slot_mapping;
	uint32_t *host_context_lengths;
	uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	void *hidden_bf16;
	void *hc_normed_bf16;
	void *hc_pre_f32;
	void *hc_post_f32;
	void *normalized_f32;
	void *current_f32;
	void *query_lora_f32;
	void *query_rope_f32;
	void *kv_latent_f32;
	void *kv_rope_f32;
	void *query_heads_f32;
	void *query_absorbed_f32;
	void *value_latent_f32;
	void *attention_out_f32;
	void *attention_acc_f32;
	void *mlp_current_f32;
	void *mlp_gate_f32;
	void *mlp_up_f32;
	void *mlp_expert_out_f32;
	void *kv_latent_cache_f32;
	void *kv_rope_cache_f32;
	void *index_weights_f32;
	void *index_scores_f32;
	void *index_topk_u32;
	void *logits_f32;
} SparkHy4PipelineSlot;

typedef struct SparkHy4ResidentDecodeStageNodeContext
{
	uint32_t abi_version;
	uint32_t max_active_sequence_count;
	uint32_t pipeline_slot_count;
	uint32_t kv_cache_block_count;
	uint32_t enable_cuda_graph_replay;
	float rms_norm_epsilon;
	float hc_epsilon;
	float hc_magnitude;
	float routed_scaling_factor;
	float swiglu_limit;
	const SparkHy4AttnLayerWeights *attn_weights_by_layer;
	const SparkHy4HyperWeights *hc_attn_by_layer;
	const SparkHy4MoeWeights *moe_weights_by_layer;
	const SparkHy4HyperWeights *hc_ffn_by_layer;
	const SparkHy4HeadWeights *head_weights;
	SparkHy4PipelineSlot *pipeline_slots;
	uint64_t estimated_service_time_ns;
} SparkHy4ResidentDecodeStageNodeContext;

typedef struct SparkHy4DecodeBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t reserved0;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} SparkHy4DecodeBatchView;

typedef struct SparkHy4PrefillFrameView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_index;
	uint32_t token_count;
	uint64_t base_position;
	uint64_t sequence_id;
	const uint32_t *token_ids;
} SparkHy4PrefillFrameView;

#define SPARK_HY4_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE 0x00000001u
#define SPARK_HY4_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW 0x00000002u
#define SPARK_HY4_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT 0x00000004u
#define SPARK_HY4_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT 0x00000008u
#define SPARK_HY4_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW 0x00000010u

typedef struct SparkHy4ResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const void *kv_block_table;
	const SparkHy4DecodeBatchView *decode_batch;
	const SparkHy4PrefillFrameView *prefill_frame;
	void *hidden_input_transport_session;
	void *hidden_output_transport_session;
} SparkHy4ResidentDecodeStageFrameContext;

SparkStatus SparkHy4ResidentDecodeStageInitialize(
	const void *configuration, const void *host_services,
	void **module_state);
SparkStatus SparkHy4ResidentDecodeStageExecute(void *module_state,
	void *frame);
SparkStatus SparkHy4ResidentDecodeStageAdmit(void *module_state,
	const void *request, void *decision);
SparkStatus SparkHy4ResidentDecodeStageSnapshot(void *module_state,
	uint32_t program_id, void *snapshot);
void SparkHy4ResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif
