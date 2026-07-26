#ifndef SPARKPIPE_SPARK_K3_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_K3_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include <stdint.h>

#include "sparkpipe/spark_k3_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_K3_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 3u
#define SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 3u
#define SPARK_K3_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION 1u
#define SPARK_K3_RESIDENT_DECODE_STAGE_KDA_STATE_POOL_ABI_VERSION 1u
#define SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TABLE_ABI_VERSION 1u
#define SPARK_K3_RESIDENT_DECODE_STAGE_MXFP4_LINEAR_VIEW_ABI_VERSION 1u

#define SPARK_K3_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION SPARK_K3_MODEL_HIDDEN_DIMENSION
#define SPARK_K3_RESIDENT_DECODE_STAGE_LAYER_COUNT SPARK_K3_MODEL_LAYER_COUNT
#define SPARK_K3_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_K3_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT 16u
#define SPARK_K3_RESIDENT_DECODE_STAGE_SIDEBAND_KIND_ATTNRES 1u
#define SPARK_K3_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT 64u
#define SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS 64u
#define SPARK_K3_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID UINT32_MAX
#define SPARK_K3_RESIDENT_DECODE_STAGE_NO_BLOCK 0xffffffffu
#define SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_SITES_PER_LAYER 2u
#define SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_ATTENTION_SITE 0u
#define SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_MLP_SITE 1u

/*
 * Execution requirement flags, mirroring the glm52 vocabulary where the
 * semantics carry over unchanged. New to K3: the recurrent-state table
 * requirement (KDA replaces most of the KV cache) and the MXFP8 activation
 * path requirement (disclosed serving format).
 */
#define SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_GRAPH_REPLAY 0x00000001u
#define SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION 0x00000002u
#define SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH 0x00000004u
#define SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_TENSOR_CORE_ALIGNMENT 0x00000008u
#define SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_INPUT 0x00000010u
#define SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_OUTPUT 0x00000020u
#define SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_RUNTIME_KDA_STATE_TABLE 0x00000040u
#define SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_RUNTIME_MLA_BLOCK_TABLE 0x00000080u
#define SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_MXFP8_ACTIVATIONS 0x00000100u
#define SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_BULK_PREFILL 0x00000200u
#define SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_KNOWN_FLAGS \
	(SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_GRAPH_REPLAY | \
	 SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION | \
	 SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FIXED_ACTIVE_BATCH | \
	 SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_TENSOR_CORE_ALIGNMENT | \
	 SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_INPUT | \
	 SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_OUTPUT | \
	 SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_RUNTIME_KDA_STATE_TABLE | \
	 SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_RUNTIME_MLA_BLOCK_TABLE | \
	 SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_MXFP8_ACTIVATIONS | \
	 SPARK_K3_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_BULK_PREFILL)

typedef enum SparkK3ResidentDecodeStageWeightFormat
{
	SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 = 0,
	SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 = 1,
	SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 = 2,
	SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 = 3
} SparkK3ResidentDecodeStageWeightFormat;

typedef enum SparkK3ResidentDecodeStageLayerAttentionKind
{
	SPARK_K3_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_KDA = 0,
	SPARK_K3_RESIDENT_DECODE_STAGE_LAYER_ATTENTION_GATED_MLA = 1
} SparkK3ResidentDecodeStageLayerAttentionKind;

typedef enum SparkK3ResidentDecodeStageLayerMlpKind
{
	SPARK_K3_RESIDENT_DECODE_STAGE_LAYER_MLP_DENSE = 0,
	SPARK_K3_RESIDENT_DECODE_STAGE_LAYER_MLP_ROUTED = 1
} SparkK3ResidentDecodeStageLayerMlpKind;

/*
 * One quantized (or bf16) row-major linear: output_dimension rows over
 * input_dimension columns. MXFP4 packs two E2M1 nibbles per byte along the
 * input dimension with one E8M0 scale per SPARK_K3_MODEL_MXFP4_GROUP_SIZE
 * inputs. When weight_format is BF16 the payload is the bf16 matrix and the
 * scale pointer is null.
 */
typedef struct SparkK3Mxfp4LinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t input_dimension;
	uint32_t output_dimension;
	const void *weight_payload;
	const uint8_t *weight_scale_e8m0;
	uint64_t weight_payload_bytes;
	uint64_t weight_scale_bytes;
} SparkK3Mxfp4LinearView;

/*
 * Block AttnRes weights for one mixing site: a single learned pseudo-query
 * over the hidden dimension and the RMS gains applied to the candidate
 * representations before the depth softmax (K = RMSNorm(V) in the published
 * pseudocode). Every layer has two sites (before attention, before MLP) and
 * the model has one final site before the output norm.
 */
typedef struct SparkK3AttnResSiteWeights
{
	const void *pseudo_query_bf16;
	const void *key_norm_weight_bf16;
} SparkK3AttnResSiteWeights;

/*
 * Kimi Delta Attention weights for one KDA layer. The delta-rule core is
 * S <- Diag(a) S + b k (v - (Diag(a) S)^T k)^T with per-channel decay a and
 * per-head strength b; q and k are L2-normalized per head before use.
 * Decay path (GUESS pending the K3 report, isolated in one device function):
 * log a = -softplus(W_a2 (W_a1 x)) per channel, guaranteeing a in (0,1].
 * Output path: per-head RMS norm, sigmoid low-rank output gate, then the
 * output projection back to the hidden dimension.
 */
typedef struct SparkK3KdaLayerWeights
{
	SparkK3Mxfp4LinearView query;
	SparkK3Mxfp4LinearView key;
	SparkK3Mxfp4LinearView value;
	SparkK3Mxfp4LinearView decay_low;
	SparkK3Mxfp4LinearView decay_high;
	SparkK3Mxfp4LinearView beta;
	SparkK3Mxfp4LinearView output_gate_low;
	SparkK3Mxfp4LinearView output_gate_high;
	SparkK3Mxfp4LinearView output;
	const void *head_norm_weight_bf16;
} SparkK3KdaLayerWeights;

/*
 * Gated MLA weights for one global-attention layer. NoPE (rope dimension is
 * zero): the compressed cache token is the kv latent alone, decode runs the
 * absorbed-latent path, and selectivity comes from a per-head sigmoid output
 * gate projected from the layer input.
 */
typedef struct SparkK3MlaLayerWeights
{
	SparkK3Mxfp4LinearView query_a;
	const void *query_a_norm_weight_bf16;
	SparkK3Mxfp4LinearView query_b;
	SparkK3Mxfp4LinearView kv_a;
	const void *kv_a_norm_weight_bf16;
	SparkK3Mxfp4LinearView kv_b;
	SparkK3Mxfp4LinearView head_gate;
	SparkK3Mxfp4LinearView output;
} SparkK3MlaLayerWeights;

/*
 * MoE weights for one layer. Router scores are sigmoid(W_r x + bias); the
 * top SPARK_K3_MODEL_MOE_TOP_K scores are normalized and scaled by the
 * routed scaling factor. Experts use SiTU: intermediate =
 * sigmoid(W_gate x) * tanh(W_up x), output = W_down intermediate. Expert
 * payloads are stored expert-major and contiguous so one pointer plus the
 * per-expert byte strides addresses any expert. Dense layers reuse the same
 * struct with expert_count zero and the shared slot holding the dense MLP.
 */
typedef struct SparkK3MoeLayerWeights
{
	const void *router_weight_bf16;
	const float *router_score_bias_f32;
	uint32_t expert_count;
	uint32_t intermediate_dimension;
	uint32_t weight_format;
	uint32_t reserved0;
	const void *expert_gate_payload;
	const uint8_t *expert_gate_scale_e8m0;
	const void *expert_up_payload;
	const uint8_t *expert_up_scale_e8m0;
	const void *expert_down_payload;
	const uint8_t *expert_down_scale_e8m0;
	uint64_t expert_gate_payload_stride_bytes;
	uint64_t expert_up_payload_stride_bytes;
	uint64_t expert_down_payload_stride_bytes;
	uint64_t expert_gate_scale_stride_bytes;
	uint64_t expert_up_scale_stride_bytes;
	uint64_t expert_down_scale_stride_bytes;
	SparkK3Mxfp4LinearView shared_gate;
	SparkK3Mxfp4LinearView shared_up;
	SparkK3Mxfp4LinearView shared_down;
} SparkK3MoeLayerWeights;

/*
 * KDA recurrent state pool: fp32 state of
 * SPARK_K3_MODEL_KDA_STATE_ELEMENTS_PER_LAYER elements per (lane, kda layer),
 * lane-major. sequence_positions_by_lane tracks how many tokens each lane's
 * state has absorbed; a lane whose recorded sequence id differs from the
 * frame's is cold and the kernels treat its state as zero on first touch.
 */
typedef struct SparkK3KdaStatePool
{
	uint32_t abi_version;
	uint32_t lane_capacity;
	uint32_t kda_layer_count;
	uint32_t reserved0;
	float *state_f32;
	uint64_t lane_stride_elements;
	uint64_t layer_stride_elements;
	uint32_t *state_cold_by_row;
} SparkK3KdaStatePool;

/*
 * Paged MLA latent cache table, shape-compatible with the glm52 view so the
 * serving side ports without a new allocator: lane-major logical block lists
 * indexing a physical pool of blocks of SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS
 * tokens, each token SPARK_K3_MODEL_MLA_CACHE_TOKEN_ELEMENTS bf16 elements.
 */
typedef struct SparkK3MlaBlockTableView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t block_token_count;
	uint32_t lane_count;
	uint32_t lane_stride;
	uint32_t lane_capacity;
	const uint32_t *physical_block_indices;
	const uint32_t *lane_physical_block_counts;
	const uint32_t *host_physical_block_indices;
	const uint32_t *host_lane_physical_block_counts;
} SparkK3MlaBlockTableView;

/*
 * Per-pipeline-slot device activation buffers. Row counts are
 * row_capacity for both decode rows and padded prefill rows;
 * max_prefill_tokens records the qualified unpadded prefill limit.
 * staging; the AttnRes representation buffer is representation-major
 * ([representation][row][hidden]) and holds the embedding block, every
 * completed block and, as its last live candidate, the running partial sum.
 */
/*
 * slot_mapping and lane_indices are distinct: slot_mapping[row] is the
 * physical token slot the row's latent is written to in the MLA cache, while
 * lane_indices[row] is the lane that owns the row's sequence, used to look up
 * that lane's block list and its KDA recurrent state. A decode batch has one
 * row per lane; a prefill dispatch has many rows on a single lane.
 */
typedef struct SparkK3PipelineSlot
{
	void *cuda_stream;
	const uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	const uint32_t *slot_mapping;
	const uint32_t *lane_indices;
	const uint32_t *context_lengths;
	void *attnres_representations_bf16;
	void *mixed_hidden_bf16;
	void *normalized_hidden_bf16;
	void *kda_query_bf16;
	void *kda_key_bf16;
	void *kda_value_bf16;
	void *kda_log_decay_bf16;
	void *kda_beta_bf16;
	void *kda_decay_low_rank_bf16;
	void *kda_gate_low_rank_bf16;
	void *kda_core_output_bf16;
	void *kda_gate_bf16;
	void *mla_query_a_bf16;
	void *mla_query_b_bf16;
	void *mla_query_latent_bf16;
	void *mla_kv_a_bf16;
	void *mla_attention_latent_bf16;
	void *mla_head_output_bf16;
	void *mla_head_gate_bf16;
	void *attention_output_hidden_bf16;
	uint32_t *moe_topk_expert_ids;
	float *moe_topk_weights_f32;
	void *moe_gate_bf16;
	void *moe_intermediate_bf16;
	uint32_t *moe_expert_offsets;
	uint32_t *moe_grouped_rows;
	uint32_t *moe_grouped_weight_slots;
	uint32_t *moe_inverse_map;
	void *moe_output_hidden_bf16;
	float *restricted_logits_f32;
} SparkK3PipelineSlot;

typedef struct SparkK3ResidentDecodeStageNodeContext
{
	uint32_t abi_version;
	uint32_t pipeline_slot_count;
	uint32_t row_capacity;
	uint32_t max_prefill_tokens;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	float rms_norm_epsilon;
	float moe_routed_scaling_factor;
	uint32_t moe_norm_topk_prob;
	uint32_t enable_cuda_graph_replay;
	const void *token_embedding_bf16;
	const SparkK3AttnResSiteWeights *attnres_sites_by_layer;
	const SparkK3AttnResSiteWeights *attnres_final_site;
	const void *const *attention_norm_weights_by_layer_bf16;
	const void *const *mlp_norm_weights_by_layer_bf16;
	const SparkK3KdaLayerWeights *kda_weights_by_layer;
	const SparkK3MlaLayerWeights *mla_weights_by_layer;
	const SparkK3MoeLayerWeights *moe_weights_by_layer;
	const void *final_norm_weight_bf16;
	const void *restricted_lm_head_weight_bf16;
	const uint32_t *restricted_token_ids;
	uint32_t restricted_vocab_count;
	uint32_t reserved0;
	SparkK3KdaStatePool kda_state_pool;
	void *mla_cache_bf16;
	uint32_t mla_cache_block_count;
	uint32_t reserved1;
	const SparkK3PipelineSlot *pipeline_slots;
	uint64_t validated_stage_latency_ns;
	uint64_t estimated_service_time_ns;
} SparkK3ResidentDecodeStageNodeContext;

/*
 * A batched decode dispatch: one token for each of row_count DISTINCT
 * lanes. Version 2 requires it on every decode frame; the per-row lane
 * and position replace the frame-level dispatch slot and sequence
 * position for decode. Prefill keeps the single-lane frame shape.
 */
typedef struct SparkK3DecodeBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t reserved0;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} SparkK3DecodeBatchView;

typedef SparkStatus (*SparkK3HiddenTransportPostReceiveFunction)(SparkHiddenTransportSession *transport_session, SparkHiddenTransportPacket *packet);
typedef SparkStatus (*SparkK3HiddenTransportSendFunction)(SparkHiddenTransportSession *transport_session, const SparkHiddenTransportPacket *packet);

/*
 * The stage boundary state is the AttnRes representation array itself:
 * entering layer L a row owns COMPLETED_BLOCKS_BEFORE_LAYER(L) completed
 * blocks (the embedding block among them) plus the running partial. The
 * whole live prefix rides the transport sideband in the array's own
 * representation-major layout - (completed(L) + 1) x hidden per row,
 * sideband kind ATTNRES - while hidden_bf16 points at the partial. A
 * slice planner should prefer post-block cuts, where the freshly
 * reopened partial makes the boundary carry no mid-block state.
 */
typedef struct SparkK3ResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t logical_lane_count;
	uint32_t prefill_lane_index;
	uint32_t reserved0;
	const SparkK3MlaBlockTableView *mla_block_table;
	const SparkK3DecodeBatchView *decode_batch;
	SparkHiddenTransportSession *hidden_input_transport_session;
	SparkHiddenTransportSession *hidden_output_transport_session;
	SparkK3HiddenTransportPostReceiveFunction hidden_input_post_receive_function;
	SparkK3HiddenTransportSendFunction hidden_output_send_function;
	SparkHiddenTransportPacket hidden_input_packet;
	SparkHiddenTransportPacket hidden_output_packet;
} SparkK3ResidentDecodeStageFrameContext;

#define SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MLA_BLOCK_TABLE 0x00000001u
#define SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT 0x00000002u
#define SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT 0x00000004u
#define SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW 0x00000008u

SparkStatus SparkK3ResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state);

SparkStatus SparkK3ResidentDecodeStageExecute(
	void *module_state,
	SparkModelDriverFrame *frame);

SparkStatus SparkK3ResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision);

SparkStatus SparkK3ResidentDecodeStageSnapshot(
	void *module_state,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot);

void SparkK3ResidentDecodeStageDestroy(void *module_state);

/*
 * Device launch surface between the host module and the CUDA translation
 * unit. Every launcher is stream-ordered and returns only launch status;
 * completion is observed by the caller through stream events.
 */
SparkStatus SparkK3ConfigureCudaKernels(void);
SparkStatus SparkK3LaunchEmbeddingGather(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, uint32_t row_count, void *stream);
SparkStatus SparkK3LaunchAttnResMix(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3AttnResSiteWeights *site, uint32_t representation_count, uint32_t row_count, void *stream);
SparkStatus SparkK3LaunchAttnResAccumulate(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const void *sublayer_output_bf16, uint32_t open_new_block, uint32_t completed_block_count, uint32_t row_count, void *stream);
SparkStatus SparkK3LaunchRmsNorm(const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon, void *stream);
SparkStatus SparkK3LaunchLinear(const SparkK3Mxfp4LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count, void *stream);
SparkStatus SparkK3LaunchKdaMaterialize(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3KdaLayerWeights *weights, uint32_t row_count, void *stream);
SparkStatus SparkK3LaunchKdaDecodeStep(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, uint32_t kda_layer_ordinal, uint32_t row_count, void *stream);
SparkStatus SparkK3LaunchKdaChunk(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, uint32_t kda_layer_ordinal, uint32_t sequence_count, uint32_t chunk_token_count, const int32_t *sequence_token_counts, uint32_t carry_state_in, uint32_t write_state_out, void *stream);
SparkStatus SparkK3LaunchKdaFinish(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3KdaLayerWeights *weights, uint32_t row_count, void *stream);
SparkStatus SparkK3LaunchMlaDecode(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3MlaLayerWeights *weights, const SparkK3MlaBlockTableView *block_table, uint32_t layer_ordinal, uint32_t row_count, void *stream);
SparkStatus SparkK3LaunchMlaPrefill(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3MlaLayerWeights *weights, const SparkK3MlaBlockTableView *block_table, uint32_t layer_ordinal, uint32_t row_count, void *stream);
SparkStatus SparkK3LaunchMoeRoute(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3MoeLayerWeights *weights, uint32_t row_count, void *stream);
SparkStatus SparkK3LaunchMoeExperts(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3MoeLayerWeights *weights, uint32_t row_count, void *stream);
SparkStatus SparkK3LaunchDenseMlp(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, const SparkK3MoeLayerWeights *weights, uint32_t row_count, void *stream);
SparkStatus SparkK3LaunchRestrictedLogits(const SparkK3ResidentDecodeStageNodeContext *node_context, const SparkK3PipelineSlot *slot, uint32_t row_count, void *stream);

#ifdef __cplusplus
}
#endif

#endif
