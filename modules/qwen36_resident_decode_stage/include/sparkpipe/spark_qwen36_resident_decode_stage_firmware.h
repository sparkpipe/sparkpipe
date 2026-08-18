#ifndef SPARKPIPE_SPARK_QWEN36_RESIDENT_DECODE_STAGE_FIRMWARE_H
#define SPARKPIPE_SPARK_QWEN36_RESIDENT_DECODE_STAGE_FIRMWARE_H

#include <stdint.h>

#include "sparkpipe/spark_qwen36_model.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Qwen 3.6 27B resident decode stage, pipeline-parallel from version 1.
 *
 * Every driver in this family is a STAGE, never a whole model: a node owns a
 * contiguous layer slice [first_layer_index, first_layer_index + layer_count),
 * the stage count N is configuration (never a compiled constant), and the
 * hidden state crosses stage boundaries over the sparkpipe hidden transport.
 * Stage 0 additionally owns the embedding and consumes wire token ids; the
 * last stage owns the final norm and the LM head and produces sampled ids.
 * The whole-stack case is simply N == 1, not a separate mode.
 *
 * Why PP-N even for a model that fits one node: tokens/second is dominated by
 * per-stage compute, not the ~29us/hop ring latency (a decode microbatch of
 * 512 rows moves 512 x 10.24KB = 5.2MB per boundary, well under a millisecond
 * on the ring, fully overlapped with compute). Slicing the 54GB of weights
 * over N nodes turns almost the entire 128GB of every node into KV cache and
 * recurrent state: one node caps 512 resident lanes at roughly 2K tokens of
 * full-attention context each, while N = 13 holds the same 512 lanes at 32K+
 * with the pipeline kept full by the microbatch stream. Long-memory serving
 * at large concurrency is a memory problem first, and PP-N is the answer.
 *
 * What crosses a Qwen boundary: the residual hidden vector only, rows x 5120
 * bf16 per microbatch. GDN delta state, GDN conv tails and the paged KV cache
 * are resident on the stage that owns their layers and never move. There is
 * no sideband payload (contrast K3, whose AttnRes block array rides the
 * transport sideband, and DSv4, whose mHC streams quadruple the payload).
 *
 * Decode frames are BATCHED: one frame carries one next-token row for up to
 * max_active_sequence_count distinct lanes, which is what keeps every stage
 * of the pipeline saturated at 500-way long-memory concurrency. Prefill
 * frames are one lane per frame carrying up to max_active_sequence_count
 * consecutive positions, because a prefill already fills the stage on its
 * own: projections, norms and attention batch over every position at once
 * and the GDN core walks the frame in 64-token chunks on the slot stream.
 */

#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION 3u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_MTP_DRAFT_VIEW_ABI_VERSION 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_GDN_SNAPSHOT_VIEW_ABI_VERSION 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS 8u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_GDN_SNAPSHOT_SLOTS 8u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_GDN_STATE_POOL_ABI_VERSION 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION 1u

#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION SPARK_QWEN36_MODEL_HIDDEN_DIMENSION
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT SPARK_QWEN36_MODEL_LAYER_COUNT
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT 32u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP 4096u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT 4u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT 512u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS 64u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID UINT32_MAX
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_NO_BLOCK 0xffffffffu

#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 0u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 1u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 2u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 3u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16_RANS 4u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 5u

/*
 * One linear projection, bf16 or MXFP4 payload with per-group E8M0 scales.
 * Identical contract to the K3 view; the shared Linear kernel consumes both.
 */
typedef struct SparkQwen36LinearView
{
	uint32_t abi_version;
	uint32_t weight_format;
	uint32_t input_dimension;
	uint32_t output_dimension;
	const void *weight_payload;
	const uint8_t *weight_scale_e8m0;
	uint64_t weight_payload_bytes;
	uint64_t weight_scale_bytes;
} SparkQwen36LinearView;

/*
 * Gated DeltaNet layer weights, PINNED against transformers main
 * modeling_qwen3_5 (2026-07). The qkv projection is ONE fused tensor whose
 * rows are conv channel order: query (2048) | key (2048) | value (6144); its
 * output feeds the depthwise causal conv (kernel 4, NO bias, silu on the
 * conv output) directly, then splits. beta and decay are separate 48-row
 * projections: beta = sigmoid(b); per-head log decay
 * g = -exp(a_log) * softplus(a + dt_bias), fp32 math. Inside the delta rule
 * q and k are L2-normalized per head. The gated norm is per-value-head
 * RMSNorm(128) in fp32, times the weight, times silu(z) - norm before gate -
 * and gate z comes from its own 6144-row projection.
 */
typedef struct SparkQwen36GdnLayerWeights
{
	SparkQwen36LinearView qkv;
	SparkQwen36LinearView gate;
	SparkQwen36LinearView beta;
	SparkQwen36LinearView decay;
	SparkQwen36LinearView output;
	const void *conv_weight_bf16;
	const float *a_log_f32;
	const float *dt_bias_f32;
	const void *gdn_norm_weight_bf16;
} SparkQwen36GdnLayerWeights;

/*
 * Full attention layer weights, PINNED: the query projection fuses a
 * per-head output gate - each head's 512 rows are 256 query then 256 gate.
 * The query half gets per-head RMSNorm(256) then RoPE on its first 64 dims;
 * the gate half is applied as sigmoid to the attention output, per head,
 * before the output projection. Key gets per-head RMSNorm(256) then the same
 * partial RoPE; value is unnormalized.
 */
typedef struct SparkQwen36AttnLayerWeights
{
	SparkQwen36LinearView query;
	SparkQwen36LinearView key;
	SparkQwen36LinearView value;
	SparkQwen36LinearView output;
	const void *query_norm_weight_bf16;
	const void *key_norm_weight_bf16;
} SparkQwen36AttnLayerWeights;

typedef struct SparkQwen36FfnLayerWeights
{
	SparkQwen36LinearView gate;
	SparkQwen36LinearView up;
	SparkQwen36LinearView down;
} SparkQwen36FfnLayerWeights;

/*
 * MTP head, one draft layer, pinned from the checkpoint safetensors index:
 * next_hidden = decoder(fc([enorm(embed(token)) | hnorm(hidden)])), decoder
 * geometry identical to a main full-attention layer (fused query|gate, the
 * same norms, the same SwiGLU), then mtp final norm into the SHARED lm head.
 * Lives on the stage that owns the final head.
 */
typedef struct SparkQwen36MtpWeights
{
	SparkQwen36LinearView fc;
	const void *embed_norm_weight_bf16;
	const void *hidden_norm_weight_bf16;
	const void *final_norm_weight_bf16;
	const void *attention_norm_weight_bf16;
	const void *mlp_norm_weight_bf16;
	SparkQwen36AttnLayerWeights attention;
	SparkQwen36FfnLayerWeights ffn;
} SparkQwen36MtpWeights;

/*
 * GDN recurrent state for the stage's own GDN layers only. Two carried
 * pieces per lane per GDN layer: the dk x dv fp32 delta state per value head,
 * and the conv tail (the last kernel-1 columns of the concatenated q|k|v conv
 * input, bf16) that seeds the depthwise causal conv of the next dispatch.
 * gdn_layer_ordinal densely numbers the stage's GDN layers from zero, so a
 * slice that begins mid-period costs nothing. state_cold_by_row tells the
 * kernels to treat both pieces as zero on a lane's first touch.
 */
typedef struct SparkQwen36GdnStatePool
{
	uint32_t abi_version;
	uint32_t lane_capacity;
	uint32_t gdn_layer_count;
	uint32_t reserved0;
	float *state_f32;
	uint64_t state_lane_stride_elements;
	uint64_t state_layer_stride_elements;
	void *conv_tail_bf16;
	uint64_t conv_tail_lane_stride_elements;
	uint64_t conv_tail_layer_stride_elements;
	uint32_t *state_cold_by_row;
} SparkQwen36GdnStatePool;

/*
 * Paged KV cache table for the stage's own full-attention layers. One token
 * costs SPARK_QWEN36_MODEL_ATTN_CACHE_TOKEN_ELEMENTS bf16 elements per
 * full-attention layer (K then V, head-major, post-RoPE). Host mirrors are
 * required so the module can prove per-lane block coverage before a launch.
 */
typedef struct SparkQwen36KvBlockTableView
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
} SparkQwen36KvBlockTableView;

/*
 * Per-slot device buffers, sized for max_active_sequence_count rows. A row is
 * one lane's next token in a decode microbatch, or one position of a prefill
 * chunk. hidden_input_bf16 is the transport landing buffer on stages other
 * than the first; hidden_output_bf16 is what the last layer of the slice
 * leaves for the send on stages other than the last. The LM head is fused
 * matvec + argmax and never materializes a logits tensor: 512 rows of a
 * 248320-wide fp32 logits buffer would cost half a gigabyte per slot for
 * numbers nothing reads twice.
 */
typedef struct SparkQwen36PipelineSlot
{
	void *cuda_stream;
	const uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	const uint32_t *row_lane_indices;
	const uint32_t *slot_mapping;
	const uint32_t *context_lengths;
	void *hidden_input_bf16;
	void *hidden_bf16;
	void *normalized_bf16;
	void *attn_query_bf16;
	void *attn_key_bf16;
	void *attn_value_bf16;
	void *attn_gate_bf16;
	void *attn_head_output_bf16;
	void *attn_output_bf16;
	void *gdn_conv_workspace_bf16;
	void *gdn_query_bf16;
	void *gdn_key_bf16;
	void *gdn_value_bf16;
	void *gdn_gate_bf16;
	void *gdn_ba_bf16;
	void *gdn_log_decay_f32;
	void *gdn_beta_f32;
	void *gdn_core_output_bf16;
	void *ffn_intermediate_bf16;
	void *argmax_score_f32;
	void *argmax_token_ids;
	float *chunk_qn_f32;
	float *chunk_kn_f32;
	float *chunk_cum_g_f32;
	float *chunk_decay_f32;
	float *chunk_attn_f32;
	float *chunk_w_f32;
	float *chunk_kg_f32;
	uint32_t *mtp_draft_token_ids;
} SparkQwen36PipelineSlot;

/*
 * Node context: one pipeline stage. stage_count and stage_index come from
 * configuration; first_layer_index and layer_count come from the stage pack
 * and the two must agree at load. owns_embedding and owns_final_head are
 * DERIVED (first == 0, first + count == total), never configured, so a stage
 * cannot claim a head it does not hold. Weight arrays span the full layer
 * index space and only the slice is populated; layer walks always run
 * [first_layer_index, first_layer_index + layer_count).
 */
typedef struct SparkQwen36ResidentDecodeStageNodeContext
{
	uint32_t abi_version;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t max_active_sequence_count;
	uint32_t max_prefill_tokens;
	uint32_t pipeline_slot_count;
	uint32_t kv_cache_block_count;
	uint32_t enable_cuda_graph_replay;
	float rms_norm_epsilon;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	const void *attention_norm_weights_by_layer_bf16[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const void *mlp_norm_weights_by_layer_bf16[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const SparkQwen36GdnLayerWeights *gdn_weights_by_layer;
	const SparkQwen36AttnLayerWeights *attn_weights_by_layer;
	const SparkQwen36FfnLayerWeights *ffn_weights_by_layer;
	SparkQwen36GdnStatePool gdn_state_pool;
	void *kv_cache_bf16;
	const SparkQwen36PipelineSlot *pipeline_slots;
	uint64_t estimated_service_time_ns;
} SparkQwen36ResidentDecodeStageNodeContext;

/*
 * A decode microbatch names its rows explicitly: row r is the next token of
 * lane row_lane_indices[r] at position row_positions[r] for sequence
 * row_sequence_ids[r]. All arrays are host memory owned by the caller for
 * the duration of Execute. Lanes must be distinct within one batch; the
 * scheduler's one-in-flight-frame-per-lane invariant carries over from glm52.
 */
typedef struct SparkQwen36DecodeBatchView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t row_count;
	uint32_t reserved0;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
	const uint64_t *row_sequence_ids;
} SparkQwen36DecodeBatchView;

/*
 * A prefill frame is ONE lane's consecutive prompt positions
 * [base_position, base_position + token_count): the module batches every
 * projection, norm and attention pass over all token_count rows and walks
 * the GDN core in 64-token chunks on the slot stream, so the state
 * dependency serializes for free. token_count is capped by
 * max_active_sequence_count so the decode-sized slot buffers hold a whole
 * frame; the runtime splits longer prompts into consecutive frames and the
 * resident GDN state, conv tails and KV cache carry between them. A frame
 * with base_position zero resets the lane (recurrent state and conv tails
 * are zeroed before the walk); a nonzero base requires a warm lane. On the
 * embedding stage buffers[0] carries token_count wire token ids; the head
 * stage samples ONLY the final position and writes exactly one token id.
 * frame->new_token_count must equal token_count.
 */
typedef struct SparkQwen36PrefillFrameView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_index;
	uint32_t token_count;
	uint64_t base_position;
	uint64_t sequence_id;
} SparkQwen36PrefillFrameView;

/*
 * MTP speculation, in three frame modifiers. All three are contracts with
 * the sparkring tree engine; the module executes, the engine decides.
 *
 * MTP_DRAFT_AFTER (head stage only, SPARK_QWEN36_STAGE_MTP=1): after the
 * frame's own head emission, the MTP decoder drafts draft_token_count
 * tokens for lane_index, the first predicting base_position. The MTP row
 * that predicts position p+1 sits AT position p and consumes the token
 * committed at p plus the backbone hidden of p, so drafting is two phases,
 * entirely on device. The SEED pass batches the MTP decoder over the
 * frame's own rows (all rows of a prefill frame, the lane's one row for
 * decode), embedding row_token_ids against the rows' backbone hiddens;
 * this REGENERATES the lane's MTP K/V at the frame positions from
 * committed inputs - on a verify or replay frame it is exactly the
 * draft-model prefill over the tokens being verified - and its final
 * row's argmax is draft one. Each CHAIN step then extends one position,
 * embedding the previous draft against the previous MTP hidden. The MTP
 * decoder is one full-attention layer with its OWN cache layer at ordinal
 * attn_layer_count; chain K/V land at [base_position,
 * base_position + draft_token_count - 1), which the block table must
 * cover. base_position must be exactly one past the seed row's position.
 * row_token_ids holds the frame rows' INPUT token ids (the lane's one id
 * for decode) for stages that do not own the embedding table; a stage
 * that owns it reads its own uploaded ids and ignores row_token_ids. MTP
 * cache completeness over committed history requires DRAFT_AFTER on every
 * committed row of a speculating lane; a hole only dulls later drafts,
 * verification still gates every commit. The output buffer receives
 * head_rows + draft_token_count ids, drafts last.
 *
 * SPECULATIVE_VERIFY (prefill frames, base_position > 0): the frame's
 * token_count rows are the drafted tokens; the main model runs them as an
 * ordinary warm prefill EXCEPT the head emits ALL token_count argmaxes and
 * the lane's GDN state and conv tails are snapshotted to
 * gdn_snapshot->snapshot_index before the walk. The engine compares row
 * i's emitted id against draft i+1 host-side; the first mismatch row's
 * emitted id is the correction token. Rejected positions need no cache
 * rollback: attention K/V and MTP K/V are overwritten when the position
 * re-executes; only the GDN recurrence is destructive, hence the snapshot.
 *
 * GDN_RESTORE_FIRST (prefill frames): restore gdn_snapshot->snapshot_index
 * into the lane before executing - the replay frame after a partial
 * accept, carrying the accepted drafts plus the correction token. Its
 * final-position head emission is the next committed token for free, and
 * MTP_DRAFT_AFTER may ride the same frame to regrow the draft chain.
 *
 * Tree-shaped verification (multiple candidate tokens at one position) is
 * NOT expressible in these frames: it needs per-row ancestor masks and
 * shadow cache slots for same-position siblings. Chains are the degenerate
 * tree the existing kernels prove; the tree engine's chain schedule maps
 * onto exactly these three modifiers.
 */
typedef struct SparkQwen36MtpDraftView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_index;
	uint32_t draft_token_count;
	uint64_t base_position;
	uint64_t sequence_id;
	const uint32_t *row_token_ids;
} SparkQwen36MtpDraftView;

typedef struct SparkQwen36GdnSnapshotView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t snapshot_index;
	uint32_t reserved0;
} SparkQwen36GdnSnapshotView;

#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE 0x00000001u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW 0x00000002u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT 0x00000004u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT 0x00000008u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW 0x00000010u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER 0x00000020u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY 0x00000040u
#define SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST 0x00000080u

typedef SparkStatus (*SparkQwen36HiddenTransportPostReceiveFunction)(SparkHiddenTransportSession *transport_session, SparkHiddenTransportPacket *packet);
typedef SparkStatus (*SparkQwen36HiddenTransportSendFunction)(SparkHiddenTransportSession *transport_session, const SparkHiddenTransportPacket *packet);

/*
 * Frame context. Transport is FIRST-CLASS, not rejected: a stage with
 * stage_index > 0 requires HIDDEN_INPUT_TRANSPORT on every frame and a stage
 * with stage_index + 1 < stage_count requires HIDDEN_OUTPUT_TRANSPORT; the
 * module refuses a frame whose transport flags disagree with its position in
 * the pipeline, in either direction. The packet's hidden payload is rows x
 * hidden bf16 - for a prefill frame rows is token_count, every position of
 * the chunk crosses the boundary; sideband_kind is zero for Qwen (nothing
 * but the residual crosses a boundary). Exactly one of DECODE_BATCH_VIEW
 * and PREFILL_FRAME_VIEW must be set, with the matching view non-null.
 */
typedef struct SparkQwen36ResidentDecodeStageFrameContext
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint32_t reserved0;
	const SparkQwen36KvBlockTableView *kv_block_table;
	const SparkQwen36DecodeBatchView *decode_batch;
	const SparkQwen36PrefillFrameView *prefill_frame;
	const SparkQwen36MtpDraftView *mtp_draft;
	const SparkQwen36GdnSnapshotView *gdn_snapshot;
	SparkHiddenTransportSession *hidden_input_transport_session;
	SparkHiddenTransportSession *hidden_output_transport_session;
	SparkQwen36HiddenTransportPostReceiveFunction hidden_input_post_receive_function;
	SparkQwen36HiddenTransportSendFunction hidden_output_send_function;
	SparkHiddenTransportPacket hidden_input_packet;
	SparkHiddenTransportPacket hidden_output_packet;
} SparkQwen36ResidentDecodeStageFrameContext;

SparkStatus SparkQwen36ResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state);
SparkStatus SparkQwen36ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);
SparkStatus SparkQwen36ResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkQwen36ResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot);
void SparkQwen36ResidentDecodeStageDestroy(void *module_state);

#ifdef __cplusplus
}
#endif

#endif
