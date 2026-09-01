/* Qwen 3.8 Max resident decode stage: stage-pack load + validation, a lean
 * decode-only Execute (GDN/attention + routed FP8 MoE per layer, argmax
 * head on the final stage), and the firmware's first-class hidden-transport
 * contract for PP handoffs. Prefill, MTP and speculation fail closed with
 * SPARK_STATUS_UNSUPPORTED.
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_qwen4_flash_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stage_kv_client.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_qwen4_flash_work_control.h"
#include "spark_qwen4_flash_stagepack_format.h"

#define SPARK_QWEN4_FLASH_MODULE_TAG "qwen4_flash_stage"

/* KV tier: the resident pool is a window; blocks beyond it live in the
 * pluggable KV store (local socket, network service, or a dedicated spark
 * ring running the store), keyed by (model fingerprint, layout fingerprint,
 * rank, sequence, logical block). Default provider "none" keeps the
 * all-resident behavior byte-identical to before the tier landed. */
#define SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS 16u
#define SPARK_QWEN4_FLASH_MODULE_KV_POLL_BOUND 10000u
#define SPARK_QWEN4_FLASH_MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES 4096u
#define SPARK_QWEN4_FLASH_MODULE_KV_MAX_BLOCKS_PER_LANE 4096u
/* Mirrors SPARK_LM_HEAD_SCREEN_CAP / SPARK_LM_HEAD_SHADOW_GROUP from the
 * common CUDA header (module.c cannot include that header under cc). */
#define SPARK_QWEN4_FLASH_MODULE_HEAD_SCREEN_CAP 4096u
#define SPARK_QWEN4_FLASH_MODULE_HEAD_SHADOW_GROUP 32u
/* Mirrors LM_FRAME_ERROR_WORDS (frame_error.cuh): the six-word per-frame
 * error record {code, kind, row, sequence, position, page}. */
#define SPARK_FRAME_ERROR_WORDS 6u

typedef struct SparkQwen4FlashModuleSlot
{
	void *cuda_stream;
	uint32_t *host_row_lane_indices;
	uint64_t *host_row_positions;
	uint32_t *host_row_cold;
	uint32_t *host_slot_mapping;
	uint32_t *host_context_lengths;
	uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	uint32_t *row_lane_indices;
	uint32_t *slot_mapping;
	uint32_t *context_lengths;
	uint32_t *row_cold;
	uint64_t *row_positions;
	void *hidden_bf16;      /* hc stream vector [rows, 4H] */
	void *hc_normed_bf16;   /* group-normed streams [rows, 4H] */
	void *hc_up_bf16;       /* up-projection output [rows, 4H] */
	void *hc_lowrank_bf16;  /* down-projection output [rows, lowrank] */
	void *hc_inject_bf16;   /* raw inject projection [rows, hc_count] */
	void *normalized_bf16;  /* the mixed sublayer input [rows, H] */
	void *delta_bf16;
	void *qkv_bf16;
	void *conv_out_bf16;
	void *z_bf16;
	void *beta_pre_bf16;
	void *decay_pre_bf16;
	float *log_decay_f32;
	float *beta_f32;
	void *core_bf16;
	void *gated_bf16;
	void *q_fused_bf16;
	void *k_bf16;
	void *v_bf16;
	void *head_out_bf16;
	float *moe_scores_f32;
	uint32_t *moe_indices_u32;
	float *moe_weights_f32;
	uint32_t *moe_inverse_u32;
	uint32_t *moe_grouped_rows_u32;
	uint32_t *moe_tile_prefix_w1_u32;
	uint32_t *moe_tile_prefix_w2_u32;
	void *moe_gate_packed_bf16;
	void *moe_slot_up_bf16;
	void *moe_slot_out_bf16;
	uint32_t *moe_group_offset_u32;
	void *shared_gate_bf16;
	void *shared_up_bf16;
	void *shared_down_bf16;
	float *chunk_qn_f32;
	float *chunk_kn_f32;
	float *chunk_cum_g_f32;
	float *chunk_decay_f32;
	float *chunk_attn_f32;
	float *chunk_w_f32;
	float *chunk_kg_f32;
	uint32_t *mtp_draft_ids;
	/* Indexer scratch: the frame's rotated queries [rows, 4*128], the u8
	 * token mask [rows, mask_stride], and the per-row block score keys. */
	void *indexer_query_bf16;
	uint8_t *indexer_mask_u8;
	uint32_t *indexer_scores_u32;
	/* PLE scratch: n-gram embeddings [rows*tokens, 2560] (all-reduce
	 * target), key/value projections, gated value + its norm [rows, 4H]. */
	void *ple_embedding_bf16;
	void *ple_key_bf16;
	void *ple_value_bf16;
	void *ple_gated_bf16;
	void *ple_gated_normed_bf16;
	void *ple_conv_out_bf16;
	uint32_t *ple_history_u32;
	uint32_t *ple_host_history_u32;
	/* Screened head: coarse logits + screen candidates (the exact
	 * full-vocab matvec reads 4 GB of head weight PER ROW; the shadow
	 * path reads the 4-bit copy instead and rescores a certified
	 * candidate set - identical argmax, a fraction of the bytes). At
	 * tp>1 these cover the rank's vocab shard (sharded argmax with the
	 * maxloc collective); head_scores/maxloc stage the cross-rank
	 * (score,id) keys. */
	void *head_logits_bf16;
	uint32_t *head_candidate_ids;
	uint32_t *head_candidate_counts;
	float *head_scores_f32;
	uint64_t *head_maxloc_u64;
	/* The per-frame error record (inference/kernels/frame_error.cuh):
	 * device-private slot the kernels report corruption into; the host
	 * mirror is copied back at frame end and checked before the frame is
	 * declared complete. frame_error lives 6 words. */
	uint32_t *frame_error;
	uint32_t *host_frame_error;
} SparkQwen4FlashModuleSlot;

typedef struct SparkQwen4FlashModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t multiprocessor_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	/* Measurement aids (timing probe bisect): skip one block of the layer
	 * to price the GDN/attention half vs the MoE half. Default 0. */
	uint32_t debug_skip_gdn;
	uint32_t debug_skip_moe;
	/* Tensor-parallel collective: one residual all-reduce per layer after
	 * the expert-sharded MoE. Env-driven (TP_BACKEND_PATH / TP_IDENTIFIER /
	 * TP_PORT_BASE / TP_HOSTS / TP_TIMEOUT_MS), mirrors the dsv4 wiring. */
	SparkTpDeviceCollective tp_device_collective;
	SparkTpDeviceCollectiveCreditBinding tp_credit_bindings[8u];
	uint32_t tp_credit_binding_count;
	uint32_t tp_collective_initialized;
	void *tp_collective_credit_send_bf16;
	void *tp_collective_credit_receive_bf16;
	void *tp_host_credit_send_bf16;
	void *tp_host_credit_receive_bf16;
	atomic_uint tp_completion_flag;
	atomic_ullong tp_next_ordinal;
	char tp_backend_path[SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES];
	char tp_hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE][SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES];
	char tp_local_host[SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES];
	uint64_t tp_collective_identifier;
	uint32_t tp_control_port_base;
	uint32_t tp_connect_timeout_milli;
	uint32_t tp_operation_timeout_milli;
	uint32_t max_active_sequence_count;
	uint32_t pipeline_slot_count;
	atomic_uint slot_states[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t kv_block_count;
	uint32_t cache_layer_count;
	uint32_t mtp_cache_ordinal;
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong tokens_emitted;
	SparkQwen4FlashGdnStatePool gdn_pool;
	void *kv_cache_bf16;
	uint64_t cache_layer_stride;
	uint64_t cache_block_stride;
	/* Indexer caches (one plane per attn layer incl. the MTP ordinal): raw
	 * per-token keys in the paged layout (128/token) and the pooled
	 * per-block keys [blocks, 128]; both sized by the module's KV pool. */
	uint32_t indexer_cache_layer_count;
	void *indexer_raw_key_cache_bf16;
	void *indexer_pooled_key_cache_bf16;
	uint64_t indexer_raw_layer_stride;
	uint64_t indexer_pooled_layer_stride;
	uint32_t indexer_mask_stride;
	/* PLE carried state: the lane's last 2 token ids (u32, EOS-seeded) and
	 * the dilated conv tail [lanes, 4H, 9] bf16. */
	uint32_t *ple_prev_context_u32;
	void *ple_conv_tail_bf16;
	uint64_t ple_conv_tail_lane_stride;
	SparkQwen4FlashModuleSlot slots[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t allow_unqualified_execution;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	/* TP standalone bypass (the 27b escape hatch): tp_degree > 1 without a
	 * peer group - the collective is skipped and every reduce becomes a
	 * no-op, so ONE rank exercises the whole-stack sharded paths (validator,
	 * smoke). Embedding gathers stay rank-partial and the head argmax covers
	 * only this rank's vocab shard: accepted for validation, never for
	 * serving. */
	uint32_t tp_standalone;
	uint32_t debug_frame_serial;
	uint32_t debug_token_serial;
	uint32_t (*debug_dump_hidden)(struct SparkQwen4FlashModuleState *, struct SparkQwen4FlashModuleSlot *, uint32_t, uint32_t);
	/* This rank's vocab shard for the sharded embedding/head paths
	 * (tp_degree 1: the full-width slab, base 0). */
	uint32_t tp_vocab_base;
	uint32_t tp_vocab_rows;
	uint32_t gdn_layer_count;
	uint32_t attn_layer_count;
	uint32_t gdn_ordinal_by_layer[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t attn_ordinal_by_layer[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	/* v2 packs carry 55 tensor kinds, so the per-layer coverage bitfields
	 * widened from u32 to u64 (kind ids 32..54 live in the high word). */
	uint64_t layer_seen_bits[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint64_t global_seen_bits;
	uint64_t mtp_seen_bits;
	SparkQwen4FlashMtpWeights mtp;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	/* One-time 4-bit shadow of the head weights + certified per-neuron
	 * error bounds for the screened argmax. */
	uint8_t *head_shadow_payload;
	uint8_t *head_shadow_scale;
	float *head_error_norm_f32;
	/* Hyper-connection residual: per-layer per-sublayer mixers, the global
	 * readout mixer, and the PLE block (layer PLE_LAYER_INDEX only). The
	 * hc group-norm weights ride the classic ATTENTION_NORM/MLP_NORM slots
	 * at their true [4H] width. */
	SparkQwen4FlashHcWeights attn_hc_by_layer[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen4FlashHcWeights mlp_hc_by_layer[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen4FlashHcMixer readout_mixer;
	SparkQwen4FlashPleWeights ple;
	uint32_t owns_ple;
	uint32_t allow_missing_ple;
	/* Indexer per full-attention layer: pooled-block key cache
	 * [layers, kv_block_count, 128] bf16, recomputed pool-on-write. */
	SparkQwen4FlashIndexerWeights indexer_by_layer[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const void *attention_norm_by_layer[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const void *mlp_norm_by_layer[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen4FlashGdnLayerWeights gdn_by_layer[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen4FlashAttnLayerWeights attn_by_layer[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen4FlashMoeWeights moe_by_layer[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	/* KV tier state. Logical blocks are (lane, logical_block) identities;
	 * kv_logical_to_slot[lane * lane_stride + logical] holds slot+1 while the
	 * block is resident, 0 while it lives in the store. Slots are pool
	 * windows; kv_slot_dirty marks a slot whose cached rows the store lacks. */
	SparkStageKvClient kv_client;
	SparkQwen4FlashWorkControlKvState kv_work;
	SparkQwen4FlashWorkControlKvPlanConfig kv_plan;
	uint32_t kv_tier_active;
	uint32_t kv_logical_page_capacity;
	uint32_t kv_physical_page_capacity;
	uint64_t kv_backing_maximum_bytes;
	uint32_t *kv_logical_to_slot;
	uint64_t kv_logical_to_slot_capacity;
	uint32_t kv_logical_stride;
	uint32_t *kv_table_indices_device;
	uint32_t *kv_table_counts_device;
	uint32_t *kv_table_indices_host;
	uint32_t *kv_slot_lane;
	uint32_t *kv_slot_logical;
	uint64_t *kv_slot_sequence;
	uint8_t *kv_slot_dirty;
	uint8_t *kv_slot_pinned;
	uint32_t *kv_slot_free_stack;
	uint32_t kv_slot_free_count;
	uint32_t kv_evict_cursor;
	void *kv_block_staging;
	void *kv_gdn_staging;
} SparkQwen4FlashModuleState;

static SparkStatus SparkQwen4FlashModuleConfigure(SparkQwen4FlashModuleState *state)
{
	SparkStatus status;
	{
		/* Optional head-parallel TP geometry. Unset means the replicated
		 * layout (tp_degree 1). tp_degree > 1 needs the head-sliced
		 * projections AND the residual all-reduce, so the initialize path
		 * refuses it until the TP collective is wired (fail closed). */
		const char *tp_degree_text = getenv("SPARK_QWEN4_FLASH_TP_DEGREE");
		const char *tp_rank_text = getenv("SPARK_QWEN4_FLASH_TP_RANK");
		char *end = 0;
		unsigned long parsed = 1u;
		if ( tp_degree_text != 0 )
		{
			parsed = strtoul(tp_degree_text,&end,10);
			if ( end == tp_degree_text || parsed < 1u || parsed > SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_HEAD_COUNT )
				return(SPARK_STATUS_INVALID_ARGUMENT);
		}
		state->tp_degree = (uint32_t)parsed;
		parsed = 0u;
		if ( tp_rank_text != 0 )
		{
			parsed = strtoul(tp_rank_text,&end,10);
			if ( end == tp_rank_text )
				return(SPARK_STATUS_INVALID_ARGUMENT);
		}
		state->tp_rank = (uint32_t)parsed;
		/* TP sharding (expert-sharded MoE, one residual all-reduce per
		 * layer). The attention/GDN head slicing and the KV-head split are
		 * the follow-on increments; the expert slice works for any degree
		 * that divides the 512 experts. tp>1 needs the collective env
		 * unless the standalone bypass is set. */
		if ( state->tp_rank >= state->tp_degree || state->tp_degree > SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT || (SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT % state->tp_degree) != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( (SPARK_QWEN4_FLASH_MODEL_OUTPUT_VOCAB_COUNT % state->tp_degree) != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		state->tp_standalone = getenv("SPARK_QWEN4_FLASH_TP_STANDALONE") != 0 ? 1u : 0u;
		/* Vocab shard for the embedding/head paths: the packs vocab-block
		 * embed and lm_head across ranks; the shard base feeds the sharded
		 * gather and the head screening's candidate offset. */
		state->tp_vocab_rows = SPARK_QWEN4_FLASH_MODEL_OUTPUT_VOCAB_COUNT / state->tp_degree;
		state->tp_vocab_base = state->tp_rank * state->tp_vocab_rows;
		state->tp_collective_identifier = 0u;
		state->tp_control_port_base = 0u;
		state->tp_connect_timeout_milli = 120000u;
		state->tp_operation_timeout_milli = 120000u;
		state->tp_backend_path[0] = '\0';
		state->tp_local_host[0] = '\0';
		for (parsed = 0u; parsed < SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE; parsed++)
			state->tp_hosts[parsed][0] = '\0';
		if ( state->tp_degree > 1u && state->tp_standalone == 0u )
		{
			const char *tp_backend = getenv("SPARK_QWEN4_FLASH_STAGE_TP_BACKEND_PATH");
			const char *tp_identifier = getenv("SPARK_QWEN4_FLASH_STAGE_TP_IDENTIFIER");
			const char *tp_port_base = getenv("SPARK_QWEN4_FLASH_STAGE_TP_PORT_BASE");
			const char *tp_hosts = getenv("SPARK_QWEN4_FLASH_STAGE_TP_HOSTS");
			const char *tp_local_host = getenv("SPARK_QWEN4_FLASH_STAGE_TP_LOCAL_HOST");
			const char *tp_timeout = getenv("SPARK_QWEN4_FLASH_STAGE_TP_TIMEOUT_MS");
			const char *scan;
			uint32_t host_index,host_start;
			if ( tp_backend == 0 || tp_identifier == 0 || tp_port_base == 0 || tp_hosts == 0 || tp_local_host == 0 )
				return(SPARK_STATUS_INVALID_ARGUMENT);
			snprintf(state->tp_backend_path,sizeof(state->tp_backend_path),"%s",tp_backend);
			snprintf(state->tp_local_host,sizeof(state->tp_local_host),"%s",tp_local_host);
			state->tp_collective_identifier = strtoull(tp_identifier,0,10);
			state->tp_control_port_base = (uint32_t)strtoul(tp_port_base,0,10);
			if ( tp_timeout != 0 )
			{
				state->tp_connect_timeout_milli = (uint32_t)strtoul(tp_timeout,0,10);
				state->tp_operation_timeout_milli = state->tp_connect_timeout_milli;
			}
			/* Comma-separated peer host list, one per rank in rank order. */
			scan = tp_hosts;
			host_index = 0u;
			while ( *scan != '\0' && host_index < state->tp_degree )
			{
				const char *comma = strchr(scan,',');
				size_t length = comma != 0 ? (size_t)(comma - scan) : strlen(scan);
				if ( length >= SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES )
					return(SPARK_STATUS_INVALID_ARGUMENT);
				memcpy(state->tp_hosts[host_index],scan,length);
				state->tp_hosts[host_index][length] = '\0';
				host_index++;
				scan = comma != 0 ? comma + 1 : scan + length;
			}
			if ( host_index != state->tp_degree )
				return(SPARK_STATUS_INVALID_ARGUMENT);
			(void)host_start;
		}
	}
	state->debug_skip_gdn = getenv("SPARK_QWEN4_FLASH_STAGE_DEBUG_SKIP_GDN") != 0 ? 1u : 0u;
	state->debug_skip_moe = getenv("SPARK_QWEN4_FLASH_STAGE_DEBUG_SKIP_MOE") != 0 ? 1u : 0u;
	status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_STAGE_COUNT",1u,SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,&state->stage_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_STAGE_INDEX",0u,SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,&state->stage_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_STAGE_FIRST_LAYER",0u,SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT - 1u,&state->first_layer_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_STAGE_LAYER_COUNT",1u,SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT,&state->layer_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES",1u,SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&state->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_STAGE_PIPELINE_SLOTS",1u,SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_STAGE_KV_BLOCKS",1u,1u << 20u,&state->kv_block_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	/* PLE omission gate: synthesized mid-pipeline packs cannot carry the
	 * 23.8 GiB n-gram table at true shape, so the ten PLE tensors may be
	 * absent EN BLOC when this env names the reason - real v3 packs always
	 * carry them and load with the env unset (fail closed by default). */
	if ( state->first_layer_index <= SPARK_QWEN4_FLASH_MODEL_PLE_LAYER_INDEX && state->first_layer_index + state->layer_count > SPARK_QWEN4_FLASH_MODEL_PLE_LAYER_INDEX )
	{
		const char *reason = getenv("SPARK_QWEN4_FLASH_STAGE_ALLOW_MISSING_PLE");
		state->allow_missing_ple = reason != 0 ? 1u : 0u;
		if ( state->allow_missing_ple != 0u )
			fprintf(stderr,"%s ple_block_absent gate=%s (synthetic pack semantics; layer %u runs without the n-gram injection)\n",SPARK_QWEN4_FLASH_MODULE_TAG,reason,SPARK_QWEN4_FLASH_MODEL_PLE_LAYER_INDEX);
	}
	if ( state->stage_index >= state->stage_count || state->first_layer_index + state->layer_count > SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_slice_invalid stage=%u/%u slice=%u+%u\n",SPARK_QWEN4_FLASH_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	/* The grouped scalar expert path prices its row tiles at SPARK_LM_TILE
	 * (16) rows while the route build switches to 32-row tiles past 409
	 * sequences; a batch that crosses the boundary would silently skip rows
	 * in every expert group. Refuse it loudly until the MoE moves to the
	 * launch-planner GEMM (which shares one tile-M with the route build). */
	if ( state->max_active_sequence_count > 409u )
	{
		fprintf(stderr,"%s config_batch_too_wide max_active=%u (grouped scalar path supports at most 409 rows at a 16-row tile; see audit doc)\n",SPARK_QWEN4_FLASH_MODULE_TAG,state->max_active_sequence_count);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
	state->owns_final_head = state->first_layer_index + state->layer_count == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_LAYER_COUNT ? 1u : 0u;
	/* Whole-stack TP tiers run for real now: the embedding gathers through
	 * the rank's vocab block plus the bf16 all-reduce (replicated result
	 * once the collective is live; rank-partial under the standalone
	 * bypass), and the final head screens the rank's shard then resolves
	 * the global argmax through the u64 maxloc collective. The former
	 * fail-closed guard (tp_whole_stack_pending) is retired. */
	if ( (state->stage_index == 0u) != (state->owns_embedding != 0u) || (state->stage_index + 1u == state->stage_count) != (state->owns_final_head != 0u) )
	{
		fprintf(stderr,"%s config_position_mismatch stage=%u/%u slice=%u+%u\n",SPARK_QWEN4_FLASH_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

#define SPARK_PACK_LOAD_FN(name) SparkQwen4FlashModule##name
#define SPARK_PACK_LOAD_TYPE(name) SparkQwen4Flash##name
#define SPARK_PACK_LOAD_CONST(name) SPARK_QWEN4_FLASH_##name
#define SPARK_PACK_LOAD_LAYER_IS_GDN(layer) SPARK_QWEN4_FLASH_MODEL_LAYER_IS_GDN(layer)
#define SPARK_PACK_LOAD_SEEN_TYPE uint64_t
#define SPARK_PACK_LOAD_SEEN_ONE 1ull
#define SPARK_PACK_LOAD_SEEN_FORMAT "%016llx"
#define SPARK_PACK_LOAD_SEEN_ARG(value) ((unsigned long long)(value))
#define SPARK_PACK_LOAD_BYTES_MATCH(entry) \
	((entry)->payload_bytes == SparkQwen4FlashStagePackPayloadBytes((entry)->weight_format,(entry)->rows,(entry)->columns) && (entry)->scale_bytes == SparkQwen4FlashStagePackScaleBytes((entry)->weight_format,(entry)->rows,(entry)->columns))
#define SPARK_PACK_LOAD_EXPECT_GEOMETRY(state,expected) SparkQwen4FlashStagePackExpectedGeometry((expected),(state)->first_layer_index,(state)->layer_count,(state)->allow_missing_ple == 0u ? 1u : 0u)
#define SPARK_PACK_LOAD_GEOMETRY_MISMATCH(state,header,expected) (SparkQwen4FlashStagePackHeaderMatches((header),(expected)) != 0 || (header)->directory_offset != SPARK_QWEN4_FLASH_STAGEPACK_HEADER_BYTES)
#define SPARK_PACK_LOAD_LOG_GEOMETRY_MISMATCH(state,header,expected) \
	fprintf(stderr,"%s pack_geometry_mismatch code=%d slice=%u+%u tensor_count=%u/%u hidden=%u/%u layers=%u/%u first=%u/%u total=%u/%u mtp=%u/%u dir=%llu\n", \
		SPARK_QWEN4_FLASH_MODULE_TAG, \
		SparkQwen4FlashStagePackHeaderMatches((header),(expected)), \
		(state)->first_layer_index,(state)->layer_count, \
		(header)->tensor_count,(expected)->tensor_count, \
		(header)->hidden_dimension,(expected)->hidden_dimension, \
		(header)->layer_count,(expected)->layer_count, \
		(header)->first_layer_index,(expected)->first_layer_index, \
		(header)->total_layer_count,(expected)->total_layer_count, \
		(header)->mtp_layer_count,(expected)->mtp_layer_count, \
		(unsigned long long)(header)->directory_offset)
#define SPARK_PACK_LOAD_PREFLIGHT(state,file,header,status) do {} while (0)

#include "sparkpipe/spark_pack_load_common.h"

static SparkStatus SparkQwen4FlashModuleValidateEntry(SparkQwen4FlashModuleState *state, const SparkQwen4FlashStagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	SparkQwen4FlashStagePackTensorShape shape;
	uint32_t global = entry->layer_index == SPARK_QWEN4_FLASH_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	if ( SparkQwen4FlashStagePackResolvedShape(entry->tensor_kind,global != 0u ? 0u : entry->layer_index,global,&shape) != 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	{
		uint32_t full_rows = shape.rows;
		SparkQwen4FlashStagePackNarrowShape(&shape,entry->tensor_kind,state->tp_degree,state->tp_rank);
		if ( entry->rows != shape.rows || entry->columns != shape.columns )
		{
			/* The router gate may arrive REPLICATED (full expert rows, the
			 * correct serving plan: every rank scores all experts and routes
			 * the same global top-k) or NARROWED (the deployed pack
			 * generation: rank-local top-k). Accept both; the MoE derives
			 * the route group base from the loaded width. */
			if ( entry->tensor_kind != SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_GATE ||
				entry->rows != full_rows || entry->columns != shape.columns ||
				state->tp_degree == 1u )
				return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	/* Strict natural format, except the three routed-expert tensors may also
	 * arrive BF16 (synthesized test packs) or in the MX wire format 6
	 * (E4M3 payload + per-row E8M0 scales, packer --expert-format
	 * fp8-e8m0b128); anything else fails. */
	if ( entry->weight_format != shape.natural_format )
	{
		if ( (shape.natural_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && shape.natural_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128) || (entry->weight_format != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 && !(entry->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 && shape.natural_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128)) )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( entry->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 ? entry->scale_group_size != SPARK_QWEN4_FLASH_MODEL_MXFP4_GROUP_SIZE : ((entry->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 || entry->weight_format == SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128) ? entry->scale_group_size != 128u : entry->scale_group_size != 0u) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	return(SparkQwen4FlashModuleValidateEntryPlacement(state,entry,file_bytes,is_global));
}

static SparkStatus SparkQwen4FlashModuleBindMoe(SparkQwen4FlashMoeWeights *moe, const SparkQwen4FlashStagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_GATE: SparkQwen4FlashModuleFillLinearView(&moe->gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_W1: SparkQwen4FlashModuleFillLinearView(&moe->experts_w1,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_W3: SparkQwen4FlashModuleFillLinearView(&moe->experts_w3,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_DOWN: SparkQwen4FlashModuleFillLinearView(&moe->experts_w2,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_GATE: SparkQwen4FlashModuleFillLinearView(&moe->shared_gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_UP: SparkQwen4FlashModuleFillLinearView(&moe->shared_up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_DOWN: SparkQwen4FlashModuleFillLinearView(&moe->shared_down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT: moe->shared_gate_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen4FlashModuleBindMtp(SparkQwen4FlashModuleState *state, const SparkQwen4FlashStagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_FC: SparkQwen4FlashModuleFillLinearView(&state->mtp.fc,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTENTION_NORM: state->mtp.attention_hc.hc_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_NORM: state->mtp.mlp_hc.hc_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_DOWN: SparkQwen4FlashModuleFillLinearView(&state->mtp.attention_hc.mix_down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_UP: SparkQwen4FlashModuleFillLinearView(&state->mtp.attention_hc.mix_up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_INJECT: SparkQwen4FlashModuleFillLinearView(&state->mtp.attention_hc.block_inject,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_DOWN: SparkQwen4FlashModuleFillLinearView(&state->mtp.mlp_hc.mix_down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_UP: SparkQwen4FlashModuleFillLinearView(&state->mtp.mlp_hc.mix_up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_INJECT: SparkQwen4FlashModuleFillLinearView(&state->mtp.mlp_hc.block_inject,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_QK: SparkQwen4FlashModuleFillLinearView(&state->mtp.indexer.index_qk,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_Q_NORM: state->mtp.indexer.q_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM: state->mtp.indexer.k_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY: SparkQwen4FlashModuleFillLinearView(&state->mtp.attention.query,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY: SparkQwen4FlashModuleFillLinearView(&state->mtp.attention.key,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_VALUE: SparkQwen4FlashModuleFillLinearView(&state->mtp.attention.value,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkQwen4FlashModuleFillLinearView(&state->mtp.attention.output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY_NORM: state->mtp.attention.query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY_NORM: state->mtp.attention.key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SparkQwen4FlashModuleBindMoe(&state->mtp.moe,entry,payload,scale));
	}
}

static SparkStatus SparkQwen4FlashModuleBindGlobal(SparkQwen4FlashModuleState *state, const SparkQwen4FlashStagePackEntry *entry, void *payload)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_EMBEDDING:
		if ( state->owns_embedding == 0u && state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->token_embedding_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_FINAL_NORM:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->readout_mixer.hc_norm_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MIXER_DOWN:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		SparkQwen4FlashModuleFillLinearView(&state->readout_mixer.mix_down,entry,payload,0);
		return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MIXER_UP:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		SparkQwen4FlashModuleFillLinearView(&state->readout_mixer.mix_up,entry,payload,0);
		return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_LM_HEAD:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->lm_head_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_EMBED_NORM: state->mtp.embed_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_HIDDEN_NORM: state->mtp.hidden_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_FINAL_NORM: state->mtp.readout_mixer.hc_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_MIXER_DOWN:
		SparkQwen4FlashModuleFillLinearView(&state->mtp.readout_mixer.mix_down,entry,payload,0);
		return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_MIXER_UP:
		SparkQwen4FlashModuleFillLinearView(&state->mtp.readout_mixer.mix_up,entry,payload,0);
		return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen4FlashModuleBindLayer(SparkQwen4FlashModuleState *state, const SparkQwen4FlashStagePackEntry *entry, void *payload, void *scale)
{
	uint32_t layer = entry->layer_index;
	SparkStatus status;
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTENTION_NORM: state->attention_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_NORM: state->mlp_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_QKV: SparkQwen4FlashModuleFillLinearView(&state->gdn_by_layer[layer].qkv,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_GATE: SparkQwen4FlashModuleFillLinearView(&state->gdn_by_layer[layer].gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_BETA: SparkQwen4FlashModuleFillLinearView(&state->gdn_by_layer[layer].beta,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_DECAY: SparkQwen4FlashModuleFillLinearView(&state->gdn_by_layer[layer].decay,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_OUTPUT: SparkQwen4FlashModuleFillLinearView(&state->gdn_by_layer[layer].output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_CONV_WEIGHT: state->gdn_by_layer[layer].conv_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_A_LOG: state->gdn_by_layer[layer].a_log_f32 = (const float *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_DT_BIAS: state->gdn_by_layer[layer].dt_bias_f32 = (const float *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_NORM: state->gdn_by_layer[layer].gdn_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY: SparkQwen4FlashModuleFillLinearView(&state->attn_by_layer[layer].query,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY: SparkQwen4FlashModuleFillLinearView(&state->attn_by_layer[layer].key,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_VALUE: SparkQwen4FlashModuleFillLinearView(&state->attn_by_layer[layer].value,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkQwen4FlashModuleFillLinearView(&state->attn_by_layer[layer].output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY_NORM: state->attn_by_layer[layer].query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY_NORM: state->attn_by_layer[layer].key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_DOWN: SparkQwen4FlashModuleFillLinearView(&state->attn_hc_by_layer[layer].mix_down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_UP: SparkQwen4FlashModuleFillLinearView(&state->attn_hc_by_layer[layer].mix_up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_INJECT: SparkQwen4FlashModuleFillLinearView(&state->attn_hc_by_layer[layer].block_inject,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_DOWN: SparkQwen4FlashModuleFillLinearView(&state->mlp_hc_by_layer[layer].mix_down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_UP: SparkQwen4FlashModuleFillLinearView(&state->mlp_hc_by_layer[layer].mix_up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_INJECT: SparkQwen4FlashModuleFillLinearView(&state->mlp_hc_by_layer[layer].block_inject,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_QK: SparkQwen4FlashModuleFillLinearView(&state->indexer_by_layer[layer].index_qk,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_Q_NORM: state->indexer_by_layer[layer].q_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM: state->indexer_by_layer[layer].k_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_KEY: SparkQwen4FlashModuleFillLinearView(&state->ple.key_proj,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_VALUE: SparkQwen4FlashModuleFillLinearView(&state->ple.value_proj,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NORM_KEY: state->ple.norm_key_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NORM_QUERY: state->ple.norm_query_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NORM_CONV: state->ple.norm_conv_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_CONV: state->ple.conv_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_MULTIPLIERS: state->ple.layer_multipliers = (const int64_t *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_HEAD_VOCABS: state->ple.head_vocab_sizes = (const int64_t *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_HEAD_OFFSETS: state->ple.head_offsets = (const int64_t *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NGRAM: state->ple.ngram_embedding_bf16 = payload; state->owns_ple = 1u; return(SPARK_STATUS_OK);
	default:
		status = SparkQwen4FlashModuleBindMoe(&state->moe_by_layer[layer],entry,payload,scale);
		if ( status == SPARK_STATUS_OK )
			return(SPARK_STATUS_OK);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static uint64_t SparkQwen4FlashModuleExpectedGlobalBits(const SparkQwen4FlashModuleState *state)
{
	uint64_t bits = 0u;
	if ( state->owns_embedding != 0u || state->owns_final_head != 0u )
		bits |= 1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		bits |= (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_FINAL_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_LM_HEAD) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MIXER_DOWN) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MIXER_UP) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_MIXER_DOWN) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_MIXER_UP);
	return(bits);
}

static uint64_t SparkQwen4FlashModuleExpectedMtpBits(void)
{
	return((1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_FC) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_EMBED_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_HIDDEN_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MTP_FINAL_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTENTION_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_DOWN) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_UP) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_INJECT) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_DOWN) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_UP) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_INJECT) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_GATE) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_W1) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_W3) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_DOWN) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_GATE) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_UP) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_DOWN) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_VALUE) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_OUTPUT) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_QK) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_Q_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM));
}

static uint64_t SparkQwen4FlashModuleExpectedLayerBits(const SparkQwen4FlashModuleState *state, uint32_t layer)
{
	uint64_t bits = (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTENTION_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_DOWN) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_UP) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_HC_INJECT) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_DOWN) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_UP) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MLP_HC_INJECT) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_GATE) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_W1) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_W3) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_DOWN) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_GATE) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_UP) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_DOWN) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT);
	if ( SPARK_QWEN4_FLASH_MODEL_LAYER_IS_GDN(layer) != 0u )
		bits |= (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_QKV) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_GATE) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_BETA) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_DECAY) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_OUTPUT) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_CONV_WEIGHT) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_A_LOG) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_DT_BIAS) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_GDN_NORM);
	else
		bits |= (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_VALUE) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_OUTPUT) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_QUERY_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_ATTN_KEY_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_QK) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_Q_NORM) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_INDEXER_K_NORM);
	if ( layer == SPARK_QWEN4_FLASH_MODEL_PLE_LAYER_INDEX && state->allow_missing_ple == 0u )
		bits |= (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_KEY) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_VALUE) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NORM_KEY) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NORM_QUERY) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NORM_CONV) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_CONV) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_MULTIPLIERS) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_HEAD_VOCABS) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_HEAD_OFFSETS) | (1ull << SPARK_QWEN4_FLASH_STAGEPACK_TENSOR_PLE_NGRAM);
	return(bits);
}

void SparkQwen4FlashResidentDecodeStageDestroy(void *module_state);
static SparkStatus SparkQwen4FlashModuleAllocatePools(SparkQwen4FlashModuleState *state);
extern cudaError_t SparkQwen4FlashConfigureCudaKernels(void);
static SparkStatus SparkQwen4FlashModuleAllocateSlot(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot);
static SparkStatus SparkQwen4FlashModuleAllocateSlotHostMirrors(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot);

/* --------------------------------------------------------------------------
 * KV tier: the resident pool becomes a WINDOW over the logical block space.
 * Nonresident blocks page through the pluggable KV store (local socket,
 * network service address, or a dedicated spark ring running the store
 * service), keyed by (model fingerprint, layout fingerprint, rank, sequence,
 * logical block). Provider "none" keeps the all-resident behavior.
 * ------------------------------------------------------------------------*/

static uint64_t SparkQwen4FlashModuleFingerprint(const void *bytes, uint64_t count, uint64_t basis)
{
	const uint8_t *data = (const uint8_t *)bytes;
	uint64_t hash = basis,index;
	for (index = 0; index < count; index++)
		hash = (hash ^ data[index]) * 1099511628211ull;
	return(hash);
}

static SparkStatus SparkQwen4FlashModuleOpenKvTier(SparkQwen4FlashModuleState *state, const SparkFirmwareModuleHostServices *host_services)
{
	SparkQwen4FlashStagePackHeader geometry;
	const char *provider = 0,*service = 0,*socket_path = 0;
	uint64_t pool_bytes = 0u,model_fp,layout_fp,layout_bits[3],block_record_bytes,staging_bytes;
	uint32_t workers = 0u,block_record_elements,index;
	SparkStatus status;
	static const char *none = "none";
	state->kv_tier_active = 0u;
	state->kv_logical_page_capacity = host_services->kv_logical_page_capacity;
	state->kv_physical_page_capacity = host_services->kv_physical_page_capacity;
	state->kv_backing_maximum_bytes = host_services->kv_backing_maximum_bytes;
	/* The ABI validator enforced (logical==0)==(physical==0) and
	 * physical<=logical; a byte cap with no backing path is refused here. */
	if ( host_services->kv_backing_directory == 0 && host_services->kv_backing_maximum_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	/* The store provider is optional; getenv stays quiet when it is unset
	 * instead of logging a config_missing on every storeless deployment. */
	provider = getenv("SPARK_QWEN4_FLASH_STAGE_KV_STORE");
	if ( provider == 0 )
		provider = none;
	if ( strcmp(provider,"none") == 0 )
		return(SparkStageKvClientOpen(&state->kv_client,SPARK_QWEN4_FLASH_MODULE_TAG,provider,0u,0u,0u,0u,0u,0,0,0u,0u));
	status = SparkStageModuleEnvironmentText(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_STAGE_KV_SERVICE",&service);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_STAGE_KV_SOCKET",&socket_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_STAGE_KV_POOL_BYTES",1u,1ull << 40u,&pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_STAGE_KV_WORKERS",1u,64u,&workers);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkQwen4FlashStagePackExpectedGeometry(&geometry,state->first_layer_index,state->layer_count,1u);
	model_fp = SparkQwen4FlashModuleFingerprint(&geometry,sizeof(geometry),14695981039346656037ull);
	block_record_elements = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * SPARK_QWEN4_FLASH_MODEL_ATTN_CACHE_TOKEN_ELEMENTS * state->attn_layer_count;
	layout_bits[0] = block_record_elements;
	layout_bits[1] = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	layout_bits[2] = state->kv_block_count;
	layout_fp = SparkQwen4FlashModuleFingerprint(layout_bits,sizeof(layout_bits),model_fp);
	block_record_bytes = (uint64_t)block_record_elements * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES;
	staging_bytes = block_record_bytes * SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS;
	/* The resident pool is a window: clamp the device pool to the physical
	 * page capacity the deployment declared, so the adapter's freelist can
	 * span the full logical space while the device holds the window. */
	if ( state->kv_physical_page_capacity != 0u && state->kv_block_count > state->kv_physical_page_capacity )
		state->kv_block_count = state->kv_physical_page_capacity;
	if ( state->kv_block_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->kv_plan.model_fingerprint = model_fp;
	state->kv_plan.cache_layout_fingerprint = layout_fp;
	state->kv_plan.rank_index = state->stage_index;
	state->kv_plan.block_record_bytes = (uint32_t)block_record_bytes;
	state->kv_plan.gdn_record_bytes = SPARK_QWEN4_FLASH_MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES;
	state->kv_plan.lookahead_packet_count = 3u;
	state->kv_plan.physical_block_capacity = state->kv_block_count;
	state->kv_plan.allocated_physical_block_count = 0u;
	state->kv_plan.staging_block_capacity = SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS;
	status = SparkStageKvClientOpen(&state->kv_client,SPARK_QWEN4_FLASH_MODULE_TAG,provider,state->stage_index,state->first_layer_index,state->layer_count,model_fp,layout_fp,service,socket_path,pool_bytes,workers);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->kv_slot_lane = (uint32_t *)malloc((size_t)state->kv_block_count * sizeof(uint32_t));
	state->kv_slot_logical = (uint32_t *)malloc((size_t)state->kv_block_count * sizeof(uint32_t));
	state->kv_slot_sequence = (uint64_t *)malloc((size_t)state->kv_block_count * sizeof(uint64_t));
	state->kv_slot_dirty = (uint8_t *)calloc((size_t)state->kv_block_count,sizeof(uint8_t));
	state->kv_slot_pinned = (uint8_t *)calloc((size_t)state->kv_block_count,sizeof(uint8_t));
	state->kv_slot_free_stack = (uint32_t *)malloc((size_t)state->kv_block_count * sizeof(uint32_t));
	state->kv_block_staging = malloc((size_t)staging_bytes);
	state->kv_gdn_staging = malloc(SPARK_QWEN4_FLASH_MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES);
	/* Device-side rewritten block table, sized for the full logical space
	 * (max lanes x max blocks per lane); the module owns it so the const
	 * adapter mirror stays untouched. */
	if ( cudaMalloc((void **)&state->kv_table_indices_device,(size_t)SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_QWEN4_FLASH_MODULE_KV_MAX_BLOCKS_PER_LANE * sizeof(uint32_t)) != cudaSuccess )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cudaMalloc((void **)&state->kv_table_counts_device,(size_t)SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * sizeof(uint32_t)) != cudaSuccess )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->kv_table_indices_host = (uint32_t *)malloc((size_t)SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_QWEN4_FLASH_MODULE_KV_MAX_BLOCKS_PER_LANE * sizeof(uint32_t));
	if ( state->kv_slot_lane == 0 || state->kv_slot_logical == 0 || state->kv_slot_sequence == 0 || state->kv_slot_dirty == 0 || state->kv_slot_pinned == 0 || state->kv_slot_free_stack == 0 || state->kv_block_staging == 0 || state->kv_gdn_staging == 0 || state->kv_table_indices_host == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index = 0u; index < state->kv_block_count; index++)
		state->kv_slot_free_stack[index] = index;
	state->kv_slot_free_count = state->kv_block_count;
	state->kv_evict_cursor = 0u;
	state->kv_tier_active = 1u;
	fprintf(stderr,"%s kv_tier_open provider=%s window=%u logical=%u physical=%u backing_bytes=%llu\n",SPARK_QWEN4_FLASH_MODULE_TAG,provider,state->kv_block_count,state->kv_logical_page_capacity,state->kv_physical_page_capacity,(unsigned long long)state->kv_backing_maximum_bytes);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen4FlashModuleKvWaitBatch(SparkQwen4FlashModuleState *state, SparkQwen4FlashWorkControlKvBatchState *batch)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t polls = 0u;
	struct timespec pause;
	pause.tv_sec = 0;
	pause.tv_nsec = 500000;
	while ( batch->state == SPARK_QWEN4_FLASH_WORK_CONTROL_BATCH_SUBMITTED )
	{
		status = SparkQwen4FlashWorkControlProgress(&state->kv_client,&state->kv_work);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( batch->state == SPARK_QWEN4_FLASH_WORK_CONTROL_BATCH_READY )
			break;
		if ( ++polls >= SPARK_QWEN4_FLASH_MODULE_KV_POLL_BOUND )
		{
			fprintf(stderr,"%s kv_store_stall\n",SPARK_QWEN4_FLASH_MODULE_TAG);
			return(SPARK_STATUS_IO_ERROR);
		}
		nanosleep(&pause,0);
	}
	if ( batch->state != SPARK_QWEN4_FLASH_WORK_CONTROL_BATCH_READY || batch->status != SPARK_STATUS_OK )
		return(SPARK_STATUS_IO_ERROR);
	return(SparkQwen4FlashWorkControlAcknowledge(batch));
}

/* Write one dirty resident slot back to the store, leaving it free. */
static SparkStatus SparkQwen4FlashModuleKvEvictSlot(SparkQwen4FlashModuleState *state, uint32_t slot)
{
	SparkQwen4FlashWorkControlKvBatchState *batch = &state->kv_work.evict;
	SparkKvStoreBlock blocks[1];
	uint32_t block_count = 0u,logical;
	uint64_t sequence_id;
	cudaError_t error;
	SparkStatus status;
	if ( state->kv_slot_dirty[slot] != 0u )
	{
		error = cudaMemcpy(state->kv_block_staging,(const uint8_t *)state->kv_cache_bf16 + (uint64_t)slot * state->kv_plan.block_record_bytes,(size_t)state->kv_plan.block_record_bytes,cudaMemcpyDeviceToHost);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"kv_evict_copy"));
		logical = state->kv_slot_logical[slot];
		sequence_id = state->kv_slot_sequence[slot];
		status = SparkQwen4FlashWorkControlBuildEvictBatch(&state->kv_plan,sequence_id,&logical,1u,0u,state->kv_block_staging,state->kv_gdn_staging,blocks,1u,&block_count);
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen4FlashWorkControlSubmit(&state->kv_client,batch,SPARK_KV_STORE_OPERATION_PUT,blocks,block_count,SPARK_QWEN4_FLASH_WORK_CONTROL_RESTORE_PRIORITY_SPECULATIVE);
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen4FlashModuleKvWaitBatch(state,batch);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	/* Invalidate the reverse mapping BEFORE the metadata is scrubbed: a
	 * stale nonzero kv_logical_to_slot entry reads as resident on the next
	 * demand and would address a slot that now holds a different block. */
	if ( state->kv_logical_to_slot != 0 && state->kv_slot_lane[slot] != UINT32_MAX &&
		state->kv_slot_logical[slot] != UINT32_MAX && state->kv_logical_stride != 0u )
	{
		uint64_t evict_index = (uint64_t)state->kv_slot_lane[slot] * state->kv_logical_stride + state->kv_slot_logical[slot];
		if ( evict_index < state->kv_logical_to_slot_capacity &&
			state->kv_logical_to_slot[evict_index] == slot + 1u )
			state->kv_logical_to_slot[evict_index] = 0u;
	}
	state->kv_slot_dirty[slot] = 0u;
	state->kv_slot_pinned[slot] = 0u;
	state->kv_slot_lane[slot] = UINT32_MAX;
	state->kv_slot_logical[slot] = UINT32_MAX;
	state->kv_slot_sequence[slot] = 0u;
	state->kv_slot_free_stack[state->kv_slot_free_count++] = slot;
	return(SPARK_STATUS_OK);
}

/* Make every block the frame's attention will read resident in the window,
 * rewrite the block table to window slots, and refresh the per-row slot
 * mappings. No-op when the tier is inactive. */
static SparkStatus SparkQwen4FlashModuleKvPrepareFrame(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, SparkQwen4FlashResidentDecodeStageFrameContext *context, SparkQwen4FlashKvBlockTableView *table, uint32_t rows)
{
	SparkQwen4FlashWorkControlKvBatchState *restore_batch = &state->kv_work.restore;
	SparkKvStoreBlock blocks[SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS];
	uint32_t packet_lane_counts[1],block_count,lanes_built;
	uint32_t lane_required[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane_list[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane_count = 0u,row,lane_index,logical,slot_index;
	uint64_t logical_capacity;
	SparkStatus status;
	cudaError_t error;
	/* Transaction ledger: slots popped from the free stack stay listed
	 * here until their batch commits (kv_logical_to_slot is published).
	 * Any failure unwinds by returning the uncommitted slots and clearing
	 * this frame's pins, so transient store/CUDA errors cannot shrink the
	 * usable KV pool. */
	uint32_t uncommitted[SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS];
	uint32_t uncommitted_count = 0u;
	uint32_t unwind_index;
	SparkStatus fail_status;
	if ( state->kv_tier_active == 0u )
		return(SPARK_STATUS_OK);
	if ( context == 0 || context->decode_batch == 0 || context->decode_batch->row_sequence_ids == 0 || table == 0 || table->host_physical_block_indices == 0 || table->host_lane_physical_block_counts == 0 || table->physical_block_indices == 0 || table->lane_physical_block_counts == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	logical_capacity = (uint64_t)table->lane_count * table->lane_stride;
	if ( logical_capacity == 0u || table->lane_count > SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || table->lane_stride > SPARK_QWEN4_FLASH_MODULE_KV_MAX_BLOCKS_PER_LANE )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( logical_capacity > state->kv_logical_to_slot_capacity )
	{
		uint32_t *grown = (uint32_t *)realloc(state->kv_logical_to_slot,(size_t)logical_capacity * sizeof(uint32_t));
		if ( grown == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		memset(grown + state->kv_logical_to_slot_capacity,0,(size_t)(logical_capacity - state->kv_logical_to_slot_capacity) * sizeof(uint32_t));
		state->kv_logical_to_slot = grown;
		state->kv_logical_to_slot_capacity = logical_capacity;
		state->kv_logical_stride = table->lane_stride;
	}
	state->kv_logical_stride = table->lane_stride;
	/* Pass 1: distinct lanes, required block counts, sequence ids, and pin
	 * every already-resident block the frame needs so eviction skips it. */
	for (row = 0u; row < rows; row++)
	{
		uint32_t lane = slot->host_row_lane_indices[row];
		uint32_t required_for_row = (slot->host_context_lengths[row] + SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		uint64_t sequence_id = context->decode_batch->row_sequence_ids[row];
		if ( lane >= table->lane_count || required_for_row > table->lane_stride || sequence_id == 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		for (lane_index = 0u; lane_index < lane_count; lane_index++)
			if ( lane_list[lane_index] == lane )
				break;
		if ( lane_index == lane_count )
		{
			lane_list[lane_count] = lane;
			lane_required[lane_count] = 0u;
			lane_sequence[lane_count] = sequence_id;
			lane_count++;
		}
		if ( required_for_row > lane_required[lane_index] )
			lane_required[lane_index] = required_for_row;
	}
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t lane = lane_list[lane_index];
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
		{
			slot_index = state->kv_logical_to_slot[((uint64_t)lane * table->lane_stride) + logical];
			if ( slot_index != 0u )
				state->kv_slot_pinned[slot_index - 1u] = 1u;
		}
	}
	/* Pass 2: restore every nonresident block the frame needs, BATCHED.
	 * The per-block submit+wait of the first cut serialized one store
	 * round trip PER BLOCK - a 4096-block context restore was 4096
	 * latencies. Up to SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS blocks now
	 * ride one GET batch and one wait. */
	{
		SparkQwen4FlashWorkControlPendingLane pending_lanes[SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS];
		uint32_t pending_slots[SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS];
		uint32_t pending_logical[SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS];
		uint64_t pending_lane_index[SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS];
		uint32_t batch_block_count = 0u,batch_index;
		memset(pending_lanes,0,sizeof(pending_lanes));
		for (lane_index = 0u; lane_index < lane_count; lane_index++)
		{
			uint32_t lane = lane_list[lane_index];
			for (logical = 0u; logical < lane_required[lane_index]; logical++)
			{
				uint32_t *residency = &state->kv_logical_to_slot[((uint64_t)lane * table->lane_stride) + logical];
				if ( *residency != 0u )
					continue;
				if ( state->kv_slot_free_count == 0u )
				{
					/* Round-robin eviction, skipping pinned (frame-needed) slots. */
					uint32_t scans = 0u;
					while ( state->kv_slot_pinned[state->kv_evict_cursor] != 0u )
					{
						state->kv_evict_cursor = (state->kv_evict_cursor + 1u) % state->kv_block_count;
						if ( ++scans > state->kv_block_count )
							{
								fail_status = SPARK_STATUS_CAPACITY_EXCEEDED;
								goto fail;
							}
					}
					status = SparkQwen4FlashModuleKvEvictSlot(state,state->kv_evict_cursor);
					if ( status != SPARK_STATUS_OK )
						{
							fail_status = status;
							goto fail;
						}
				}
			slot_index = state->kv_slot_free_stack[--state->kv_slot_free_count];
			uncommitted[uncommitted_count++] = slot_index;
			pending_lanes[batch_block_count].sequence_id = lane_sequence[lane_index];
			pending_lanes[batch_block_count].nonresident_blocks = &pending_logical[batch_block_count];
			pending_lanes[batch_block_count].nonresident_block_count = 1u;
			pending_lanes[batch_block_count].gdn_nonresident = 0u;
			pending_logical[batch_block_count] = logical;
			pending_slots[batch_block_count] = slot_index;
			pending_lane_index[batch_block_count] = (uint64_t)lane_index;
			batch_block_count++;
			if ( batch_block_count == SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS )
			{
				packet_lane_counts[0] = batch_block_count;
				block_count = 0u;
				lanes_built = 0u;
				status = SparkQwen4FlashWorkControlBuildRestoreBatch(&state->kv_plan,pending_lanes,batch_block_count,packet_lane_counts,1u,state->kv_block_staging,SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS,state->kv_gdn_staging,1u,blocks,SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS,&block_count,&lanes_built);
				if ( status == SPARK_STATUS_OK && lanes_built != batch_block_count )
					status = SPARK_STATUS_CAPACITY_EXCEEDED;
				if ( status == SPARK_STATUS_OK )
					status = SparkQwen4FlashWorkControlSubmit(&state->kv_client,restore_batch,SPARK_KV_STORE_OPERATION_GET,blocks,block_count,SPARK_QWEN4_FLASH_WORK_CONTROL_RESTORE_PRIORITY_IMMEDIATE);
				if ( status == SPARK_STATUS_OK )
					status = SparkQwen4FlashModuleKvWaitBatch(state,restore_batch);
				if ( status != SPARK_STATUS_OK )
					{
						fail_status = status;
						goto fail;
					}
				for (batch_index = 0u; batch_index < batch_block_count; batch_index++)
				{
					error = cudaMemcpyAsync((uint8_t *)state->kv_cache_bf16 + (uint64_t)pending_slots[batch_index] * state->kv_plan.block_record_bytes,(const uint8_t *)state->kv_block_staging + ((uint64_t)batch_index * state->kv_plan.block_record_bytes),(size_t)state->kv_plan.block_record_bytes,cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
					if ( error != cudaSuccess )
						{
							fail_status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"kv_restore_copy");
							goto fail;
						}
					state->kv_slot_lane[pending_slots[batch_index]] = lane_list[pending_lane_index[batch_index]];
					state->kv_slot_logical[pending_slots[batch_index]] = pending_logical[batch_index];
					state->kv_slot_sequence[pending_slots[batch_index]] = pending_lanes[batch_index].sequence_id;
					state->kv_slot_dirty[pending_slots[batch_index]] = 0u;
					state->kv_logical_to_slot[((uint64_t)lane_list[pending_lane_index[batch_index]] * table->lane_stride) + pending_logical[batch_index]] = pending_slots[batch_index] + 1u;
					for (unwind_index = 0u; unwind_index < uncommitted_count; unwind_index++)
						if ( uncommitted[unwind_index] == pending_slots[batch_index] )
						{
							uncommitted[unwind_index] = uncommitted[--uncommitted_count];
							break;
						}
				}
				batch_block_count = 0u;
			}
		}
	}
	if ( batch_block_count != 0u )
	{
			packet_lane_counts[0] = batch_block_count;
			block_count = 0u;
			lanes_built = 0u;
			status = SparkQwen4FlashWorkControlBuildRestoreBatch(&state->kv_plan,pending_lanes,batch_block_count,packet_lane_counts,1u,state->kv_block_staging,SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS,state->kv_gdn_staging,1u,blocks,SPARK_QWEN4_FLASH_MODULE_KV_STAGING_RECORDS,&block_count,&lanes_built);
			if ( status == SPARK_STATUS_OK && lanes_built != batch_block_count )
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
			if ( status == SPARK_STATUS_OK )
				status = SparkQwen4FlashWorkControlSubmit(&state->kv_client,restore_batch,SPARK_KV_STORE_OPERATION_GET,blocks,block_count,SPARK_QWEN4_FLASH_WORK_CONTROL_RESTORE_PRIORITY_IMMEDIATE);
			if ( status == SPARK_STATUS_OK )
				status = SparkQwen4FlashModuleKvWaitBatch(state,restore_batch);
			if ( status != SPARK_STATUS_OK )
				{
					fail_status = status;
					goto fail;
				}
			for (batch_index = 0u; batch_index < batch_block_count; batch_index++)
			{
				error = cudaMemcpyAsync((uint8_t *)state->kv_cache_bf16 + (uint64_t)pending_slots[batch_index] * state->kv_plan.block_record_bytes,(const uint8_t *)state->kv_block_staging + ((uint64_t)batch_index * state->kv_plan.block_record_bytes),(size_t)state->kv_plan.block_record_bytes,cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
				if ( error != cudaSuccess )
					{
						fail_status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"kv_restore_copy");
						goto fail;
					}
				state->kv_slot_lane[pending_slots[batch_index]] = lane_list[pending_lane_index[batch_index]];
				state->kv_slot_logical[pending_slots[batch_index]] = pending_logical[batch_index];
				state->kv_slot_sequence[pending_slots[batch_index]] = pending_lanes[batch_index].sequence_id;
				state->kv_slot_dirty[pending_slots[batch_index]] = 0u;
				for (unwind_index = 0u; unwind_index < uncommitted_count; unwind_index++)
					if ( uncommitted[unwind_index] == pending_slots[batch_index] )
					{
						uncommitted[unwind_index] = uncommitted[--uncommitted_count];
						break;
					}
				state->kv_logical_to_slot[((uint64_t)lane_list[pending_lane_index[batch_index]] * table->lane_stride) + pending_logical[batch_index]] = pending_slots[batch_index] + 1u;
			}
		}
	}
	/* Pass 3: build the rewritten table in the module's own buffers (the
	 * adapter's host mirror stays logical), point the frame's view at them,
	 * upload ONLY the touched lanes' slices (the full-table upload of the
	 * first cut moved 8 MB per frame; a lane slice is 16 KB), and refresh
	 * the row slot mappings ahead of the layer walk. */
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t lane = lane_list[lane_index];
		uint64_t lane_slice = (uint64_t)lane * table->lane_stride;
		memcpy(state->kv_table_indices_host + lane_slice,table->host_physical_block_indices + lane_slice,(size_t)table->lane_stride * sizeof(uint32_t));
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
			state->kv_table_indices_host[lane_slice + logical] = state->kv_logical_to_slot[lane_slice + logical] - 1u;
		error = cudaMemcpyAsync((uint8_t *)state->kv_table_indices_device + (lane_slice * sizeof(uint32_t)),state->kv_table_indices_host + lane_slice,(size_t)table->lane_stride * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
		if ( error != cudaSuccess )
			{
				fail_status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"kv_table_upload");
				goto fail;
			}
	}
	table->physical_block_indices = state->kv_table_indices_device;
	table->lane_physical_block_counts = state->kv_table_counts_device;
	error = cudaMemcpyAsync((void *)state->kv_table_counts_device,(const void *)table->host_lane_physical_block_counts,(size_t)table->lane_count * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
	if ( error != cudaSuccess )
		{
			fail_status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"kv_table_upload");
			goto fail;
		}
	for (row = 0u; row < rows; row++)
	{
		uint32_t lane = slot->host_row_lane_indices[row];
		uint64_t position = slot->host_row_positions[row];
		slot_index = state->kv_logical_to_slot[((uint64_t)lane * table->lane_stride) + (uint32_t)(position / SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS)] - 1u;
		slot->host_slot_mapping[row] = slot_index * SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS + (uint32_t)(position % SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	}
	error = cudaMemcpyAsync(slot->slot_mapping,slot->host_slot_mapping,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
	if ( error != cudaSuccess )
		{
			fail_status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"kv_slot_upload");
			goto fail;
		}
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t lane = lane_list[lane_index];
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
		{
			slot_index = state->kv_logical_to_slot[((uint64_t)lane * table->lane_stride) + logical];
			if ( slot_index != 0u )
				state->kv_slot_pinned[slot_index - 1u] = 0u;
		}
	}
	return(SPARK_STATUS_OK);
fail:
	for (unwind_index = 0u; unwind_index < uncommitted_count; unwind_index++)
		state->kv_slot_free_stack[state->kv_slot_free_count++] = uncommitted[unwind_index];
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t fail_lane = lane_list[lane_index];
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
		{
			slot_index = state->kv_logical_to_slot[((uint64_t)fail_lane * table->lane_stride) + logical];
			if ( slot_index != 0u )
				state->kv_slot_pinned[slot_index - 1u] = 0u;
		}
	}
	return(fail_status);
}

/* The frame just wrote K/V rows into window slots: mark them dirty so the
 * store receives them before the slots are reused. */
static void SparkQwen4FlashModuleKvMarkWritten(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, uint32_t rows)
{
	uint32_t row,slot_index;
	if ( state->kv_tier_active == 0u )
		return;
	for (row = 0u; row < rows; row++)
	{
		slot_index = slot->host_slot_mapping[row] / SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		if ( slot_index < state->kv_block_count )
			state->kv_slot_dirty[slot_index] = 1u;
	}
}

extern cudaError_t SparkQwen4FlashLaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkQwen4FlashLaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen4FlashLaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkQwen4FlashLaunchHeadScreenedArgmaxScore(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *scratch_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, float *output_scores, uint32_t candidate_offset, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkQwen4FlashLaunchHeadMaxLocPack(cudaStream_t stream, const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count);
extern cudaError_t SparkQwen4FlashLaunchHeadMaxLocUnpack(cudaStream_t stream, const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count);
extern cudaError_t SparkQwen4FlashLaunchEmbeddingGatherSharded(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t vocab_base, uint32_t vocab_rows);
extern cudaError_t SparkQwen4FlashLaunchTpCombineAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width);
extern cudaError_t SparkQwen4FlashLaunchTpCombineU64Max(cudaStream_t stream, uint64_t *destination, const uint64_t *source, uint32_t element_count);
/* Hyper-connection residual kernels. */
extern cudaError_t SparkQwen4FlashLaunchHcStreamReplicate(cudaStream_t stream, const void *input_bf16, void *streams_bf16, uint32_t row_count);
extern cudaError_t SparkQwen4FlashLaunchHcGroupNorm(cudaStream_t stream, const void *streams_bf16, const void *weight_bf16, void *normed_bf16, uint32_t row_count, float epsilon);
extern cudaError_t SparkQwen4FlashLaunchHcSiluQuarter(cudaStream_t stream, void *lowrank_bf16, uint32_t row_count);
extern cudaError_t SparkQwen4FlashLaunchHcMix(cudaStream_t stream, const void *up_bf16, const void *normed_bf16, void *mixed_bf16, uint32_t row_count);
extern cudaError_t SparkQwen4FlashLaunchHcInject(cudaStream_t stream, void *streams_bf16, const void *inject_pre_bf16, const void *sublayer_out_bf16, uint32_t row_count);
/* Indexer kernels. */
extern cudaError_t SparkQwen4FlashLaunchIndexerPrepare(cudaStream_t stream, const void *qk_bf16, const SparkQwen4FlashIndexerWeights *weights, void *query_bf16, void *raw_key_cache, void *pooled_key_cache, const uint32_t *slot_mapping, const uint32_t *block_indices, const uint64_t *row_positions, uint32_t row_count, uint32_t lane_stride, uint64_t cache_block_stride);
extern cudaError_t SparkQwen4FlashLaunchIndexerSelect(cudaStream_t stream, const void *query_bf16, const void *pooled_key_cache, const SparkQwen4FlashKvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, uint8_t *token_mask, uint32_t *score_keys_u32, uint32_t row_count, uint32_t mask_stride, uint32_t score_stride);
/* PLE kernels. */
extern cudaError_t SparkQwen4FlashLaunchPleHashGather(cudaStream_t stream, const uint32_t *history_u32, uint32_t token_count, const SparkQwen4FlashPleWeights *ple, void *embedding_bf16, uint32_t row_count, uint32_t vocab_base, uint32_t vocab_rows);
extern cudaError_t SparkQwen4FlashLaunchPleGate(cudaStream_t stream, const void *key_normed_bf16, const void *query_normed_bf16, const void *value_bf16, void *gated_value_bf16, uint32_t row_count);
extern cudaError_t SparkQwen4FlashLaunchPleConvUpdate(cudaStream_t stream, const void *input_bf16, const SparkQwen4FlashPleWeights *ple, void *output_bf16, void *tail_bf16, const uint32_t *row_lane_indices, const uint32_t *state_cold_by_row, uint32_t row_count, uint64_t tail_lane_stride);
extern cudaError_t SparkQwen4FlashLaunchPleConvChunk(cudaStream_t stream, const void *input_bf16, const SparkQwen4FlashPleWeights *ple, void *output_bf16, void *tail_bf16, uint32_t token_count);

/* --------------------------------------------------------------------------
 * Tensor-parallel collective: one residual all-reduce per layer. The
 * combine callback runs the elementwise add kernel; the module waits on an
 * atomic flag set by the completion callback (the device collective's own
 * progress thread drives the transfer phases).
 * ------------------------------------------------------------------------*/

static SparkStatus SparkQwen4FlashModuleTpCombineBf16(void *combine_context, void *destination_device, const void *source_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context;
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,SparkQwen4FlashLaunchTpCombineAdd((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count,hidden_dimension),"tp_combine"));
}

/* Elementwise u64 max combine for the sharded-argmax maxloc reduce. */
static SparkStatus SparkQwen4FlashModuleTpCombineU64Max(void *combine_context, uint64_t *destination_device, const uint64_t *source_device, uint32_t element_count, void *cuda_stream)
{
	(void)combine_context;
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,SparkQwen4FlashLaunchTpCombineU64Max((cudaStream_t)cuda_stream,destination_device,source_device,element_count),"tp_combine_u64_max"));
}

static void SparkQwen4FlashModuleTpCompletion(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	atomic_uint *flag = (atomic_uint *)context;
	atomic_store_explicit(flag,completion != 0 && completion->status == SPARK_STATUS_OK ? 1u : 2u,memory_order_release);
}

static SparkStatus SparkQwen4FlashModuleInitializeTpCollective(SparkQwen4FlashModuleState *state)
{
	SparkTpDeviceCollectiveConfig configuration;
	SparkTpDeviceCollectiveTopology topology;
	uint32_t credit,rank,route,route_count,memory_mode;
	uint64_t credit_bytes,total_bytes,offset;
	void *mapped_send,*mapped_receive;
	cudaError_t error;
	SparkStatus status;
	if ( state->tp_degree == 1u )
		return(SPARK_STATUS_OK);
	/* Standalone bypass (the 27b escape hatch): one rank exercises the
	 * whole-stack sharded paths without a peer group. The collective is
	 * skipped and the reduces below become no-ops; embedding/head results
	 * stay rank-partial (validation semantics, not serving semantics). */
	if ( state->tp_standalone != 0u )
	{
		fprintf(stderr,"%s tp_standalone degree=%u rank=%u (collective skipped; embedding/head results stay rank-partial)\n",
			SPARK_QWEN4_FLASH_MODULE_TAG,state->tp_degree,state->tp_rank);
		return(SPARK_STATUS_OK);
	}
	memset(&topology,0,sizeof(topology));
	topology.abi_version = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
	topology.descriptor_bytes = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
	topology.rank_count = state->tp_degree;
	topology.algorithm_mask = SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING;
	topology.rail_count = 0u;
	topology.direct_all_to_all_max_payload_bytes = 0u;
	topology.split_ring_min_payload_bytes = 0u;
	for (rank = 0u; rank < state->tp_degree; rank++)
		memcpy(topology.rank_hosts[rank],state->tp_hosts[rank],SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	configuration.tp_degree = state->tp_degree;
	configuration.tp_rank = state->tp_rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = 2u * state->pipeline_slot_count;
	configuration.local_hidden_dimension = SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION;
	/* The backend sizes its credit planes by this; use the static maximum
	 * (dsv4 does the same) so the .so contract is configuration-free. */
	configuration.max_active_sequence_count = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT;
	configuration.connect_timeout_milli = state->tp_connect_timeout_milli;
	configuration.operation_timeout_milli = state->tp_operation_timeout_milli;
	configuration.control_port_base = state->tp_control_port_base;
	configuration.collective_identifier = state->tp_collective_identifier;
	configuration.backend_module_path = state->tp_backend_path;
	configuration.local_host = state->tp_local_host;
	configuration.registration_cuda_stream = state->slots[0].cuda_stream;
	configuration.combine_bf16_function = SparkQwen4FlashModuleTpCombineBf16;
	configuration.combine_u64_max_function = SparkQwen4FlashModuleTpCombineU64Max;
	configuration.combine_context = state;
	status = SparkTpDeviceCollectiveApplyTopology(&topology,&configuration);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s tp_apply_topology_failed status=%d\n",SPARK_QWEN4_FLASH_MODULE_TAG,(int)status);
		return(status);
	}
	status = SparkTpDeviceCollectiveCreditBindingRouteCount(&configuration,&route_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkTpDeviceCollectiveProbeMemoryMode(
		configuration.backend_kind,configuration.backend_module_path,
		&memory_mode);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s tp_probe_memory_mode_failed status=%d\n",SPARK_QWEN4_FLASH_MODULE_TAG,(int)status);
		return(status);
	}
	credit_bytes = (uint64_t)SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES;
	total_bytes = credit_bytes * configuration.credit_count * route_count;
	status = SparkStageModuleDeviceAllocate(&state->ledger,total_bytes,&state->tp_collective_credit_send_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,total_bytes,&state->tp_collective_credit_receive_bf16);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST )
	{
		mapped_send = 0;
		mapped_receive = 0;
		error = cudaHostAlloc(&state->tp_host_credit_send_bf16,total_bytes,cudaHostAllocPortable | cudaHostAllocMapped);
		if ( error == cudaSuccess )
			error = cudaHostAlloc(&state->tp_host_credit_receive_bf16,total_bytes,cudaHostAllocPortable | cudaHostAllocMapped);
		if ( error == cudaSuccess )
			error = cudaHostGetDevicePointer(&mapped_send,state->tp_host_credit_send_bf16,0u);
		if ( error == cudaSuccess )
			error = cudaHostGetDevicePointer(&mapped_receive,state->tp_host_credit_receive_bf16,0u);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"tp_credit_mapped_alloc"));
		state->tp_collective_credit_send_bf16 = mapped_send;
		state->tp_collective_credit_receive_bf16 = mapped_receive;
	}
	offset = 0u;
	state->tp_credit_binding_count = 0u;
	for (route = 0u; route < route_count; route++)
		for (credit = 0u; credit < configuration.credit_count; credit++)
		{
			SparkTpDeviceCollectiveCreditBinding *binding = &state->tp_credit_bindings[state->tp_credit_binding_count++];
			binding->step_index = route;
			binding->credit_index = credit;
			binding->send_device = (uint8_t *)state->tp_collective_credit_send_bf16 + offset;
			binding->receive_device = (uint8_t *)state->tp_collective_credit_receive_bf16 + offset;
			binding->send_transport = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? (uint8_t *)state->tp_host_credit_send_bf16 + offset : binding->send_device;
			binding->receive_transport = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? (uint8_t *)state->tp_host_credit_receive_bf16 + offset : binding->receive_device;
			binding->flags = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? SPARK_TP_DEVICE_COLLECTIVE_BINDING_KNOWN_FLAGS : 0u;
			binding->reserved0 = 0u;
			offset += credit_bytes;
		}
	configuration.credit_bindings = state->tp_credit_bindings;
	configuration.credit_binding_count = state->tp_credit_binding_count;
	status = SparkTpDeviceCollectiveCreate(&configuration,&state->tp_device_collective);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s tp_create_failed status=%d\n",SPARK_QWEN4_FLASH_MODULE_TAG,(int)status);
		return(status);
	}
	state->tp_collective_initialized = 1u;
	fprintf(stderr,"%s tp_collective_open degree=%u rank=%u port_base=%u\n",SPARK_QWEN4_FLASH_MODULE_TAG,state->tp_degree,state->tp_rank,state->tp_control_port_base);
	return(SPARK_STATUS_OK);
}

/* Shared submit+wait: stream-ordered completion, the backend orders its
 * device work after the combine on the slot stream before the completion
 * fires. u64_max selects the elementwise-max operation (maxloc) instead of
 * the bf16 sum. */
static SparkStatus SparkQwen4FlashModuleTpSubmitOrdered(SparkQwen4FlashModuleState *state, void *device_buffer, uint32_t count, SparkQwen4FlashModuleSlot *slot, uint32_t u64_max)
{
	SparkTpDeviceCollectiveSubmission submission;
	struct timespec pause;
	uint32_t polls,flag;
	SparkStatus status;
	if ( state->tp_degree == 1u || state->tp_standalone != 0u )
		return(SPARK_STATUS_OK);
	if ( state->tp_collective_initialized == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	atomic_store_explicit(&state->tp_completion_flag,0u,memory_order_relaxed);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 0u;
	submission.active_sequence_count = count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = atomic_fetch_add_explicit(&state->tp_next_ordinal,1u,memory_order_relaxed);
	submission.local_device = device_buffer;
	submission.full_device = device_buffer;
	submission.cuda_stream = slot->cuda_stream;
	submission.completion_function = SparkQwen4FlashModuleTpCompletion;
	submission.completion_context = &state->tp_completion_flag;
	status = u64_max != 0u
		? SparkTpDeviceCollectiveSubmitU64Max(&state->tp_device_collective,&submission)
		: SparkTpDeviceCollectiveSubmitBf16(&state->tp_device_collective,&submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pause.tv_sec = 0u;
	pause.tv_nsec = 100000;
	for (polls = 0u; polls < 100000u; polls++)
	{
		flag = atomic_load_explicit(&state->tp_completion_flag,memory_order_acquire);
		if ( flag == 1u )
			return(SPARK_STATUS_OK);
		if ( flag == 2u )
			return(SPARK_STATUS_IO_ERROR);
		nanosleep(&pause,0);
	}
	fprintf(stderr,"%s tp_all_reduce_stall\n",SPARK_QWEN4_FLASH_MODULE_TAG);
	return(SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkQwen4FlashModuleTpAllReduceHidden(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, void *device_bf16, uint32_t rows)
{
	if ( state->tp_degree == 1u || state->tp_standalone != 0u )
		return(SPARK_STATUS_OK);
	return(SparkQwen4FlashModuleTpSubmitOrdered(state,device_bf16,rows,slot,0u));
}

/* Cross-rank maxloc: elementwise u64 max over count keys. No-op at degree 1
 * and under the standalone bypass (the keys already hold this rank's local
 * argmax, so the unpack still yields a valid in-shard id). */
static SparkStatus SparkQwen4FlashModuleTpReduceU64Max(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, uint64_t *device_u64, uint32_t count)
{
	if ( state->tp_degree == 1u || state->tp_standalone != 0u )
		return(SPARK_STATUS_OK);
	return(SparkQwen4FlashModuleTpSubmitOrdered(state,device_u64,count,slot,1u));
}

SparkStatus SparkQwen4FlashResidentDecodeStageInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state)
{
	SparkQwen4FlashModuleState *state;
	const char *pack_path;
	uint32_t allow_unqualified_execution;
	SparkStatus status;
	pack_path = 0;
	allow_unqualified_execution = 0u;
	status = SparkFirmwareModuleValidateInitialization(configuration,host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_ALLOW_UNQUALIFIED_EXECUTION",1u,1u,&allow_unqualified_execution);
	if ( status != SPARK_STATUS_OK || allow_unqualified_execution != 1u )
		return(SPARK_STATUS_MODULE_NOT_VALIDATED);
	state = (SparkQwen4FlashModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->allow_unqualified_execution = allow_unqualified_execution;
	state->ledger.module_tag = SPARK_QWEN4_FLASH_MODULE_TAG;
	atomic_init(&state->submitted_count,0u);
	atomic_init(&state->completed_count,0u);
	atomic_init(&state->tp_completion_flag,0u);
	atomic_init(&state->tp_next_ordinal,0u);
	atomic_init(&state->rejected_count,0u);
	{
		uint32_t slot_index;
		for (slot_index = 0u; slot_index < SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT; slot_index++)
			atomic_init(&state->slot_states[slot_index],1u);
	}
	atomic_init(&state->failed_count,0u);
	atomic_init(&state->tokens_emitted,0u);
	status = SparkQwen4FlashModuleConfigure(state);
	/* tp_degree > 1 runs the expert-sharded MoE with one residual
	 * all-reduce per layer (the collective opens right after the slot
	 * allocation); the head-parallel attention/GDN slicing is the next
	 * increment. tp=1 is the replicated path, byte-identical as before. */
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_QWEN4_FLASH_MODULE_TAG,"SPARK_QWEN4_FLASH_STAGE_PACK_PATH",&pack_path);
	if ( status == SPARK_STATUS_OK )
	{
		SparkQwen4FlashModuleBuildOrdinals(state);
		status = SparkQwen4FlashModuleLoadPack(state,pack_path);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleOpenKvTier(state,host_services);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,SparkQwen4FlashConfigureCudaKernels(),"configure_cuda_kernels");
	if ( status == SPARK_STATUS_OK )
	{
		int32_t sm_count = 0;
		cudaError_t attr = cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,0);
		state->multiprocessor_count = attr == cudaSuccess && sm_count > 0 ? (uint32_t)sm_count : 1u;
		status = SparkQwen4FlashModuleAllocatePools(state);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleAllocateSlot(state,&state->slots[0]);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleAllocateSlotHostMirrors(state,&state->slots[0]);
	if ( status == SPARK_STATUS_OK && state->tp_degree > 1u )
		status = SparkQwen4FlashModuleInitializeTpCollective(state);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
	{
		/* One-time 4-bit shadow of the rank's head shard (vocab/degree rows;
		* the pack vocab-blocks lm_head) plus certified per-neuron error
		* bounds: the screened argmax reads the shadow (a quarter of the
		* full head's 1.02 GB at TP4) instead of the bf16 slab per row and
		* rescores a bounded candidate set exactly. */
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)state->tp_vocab_rows * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION / 2u,(void **)&state->head_shadow_payload);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)state->tp_vocab_rows * (SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION / SPARK_QWEN4_FLASH_MODULE_HEAD_SHADOW_GROUP),(void **)&state->head_shadow_scale);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)state->tp_vocab_rows * sizeof(float),(void **)&state->head_error_norm_f32);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,SparkQwen4FlashLaunchHeadShadowQuantize((cudaStream_t)state->slots[0].cuda_stream,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,state->tp_vocab_rows,SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION),"head_shadow_quantize");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,cudaStreamSynchronize((cudaStream_t)state->slots[0].cuda_stream),"head_shadow_sync");
	}
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s initialize_failed status=%d\n",SPARK_QWEN4_FLASH_MODULE_TAG,(int)status);
		SparkQwen4FlashResidentDecodeStageDestroy(state);
		return(status);
	}
	*module_state = state;
	fprintf(stderr,"%s initialize ok slice=%u+%u gdn=%u attn=%u owns_embedding=%u owns_head=%u\n",SPARK_QWEN4_FLASH_MODULE_TAG,state->first_layer_index,state->layer_count,state->gdn_layer_count,state->attn_layer_count,state->owns_embedding,state->owns_final_head);
	return(SPARK_STATUS_OK);
}



/* Admission (ported from the proven qwen38_27b module): shape policy over
 * the free pipeline slots with a KV-capacity predicate and a staging-cost
 * estimate; the max-lineage stub returned UNSUPPORTED and failed the
 * validator's admit smoke. */
static void SparkQwen4FlashAdmissionCost(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkQwen4FlashModuleState *state = (SparkQwen4FlashModuleState *)context;
	decision->host_staging_bytes = (uint64_t)request->new_token_count *
		(sizeof(uint32_t) *
			 (uint64_t)(state->owns_embedding + state->owns_final_head + 3u) +
		 sizeof(uint64_t));
	decision->device_memcpy_bytes = decision->host_staging_bytes;
}

static SparkStatus SparkQwen4FlashAdmissionKvPredicate(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	(void)context;
	if ((request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u &&
		SparkModelDriverRangeFitsWithinCapacity(
			request->sequence_position,
			request->new_token_count,
			SPARK_QWEN4_FLASH_MODEL_MAXIMUM_CONTEXT_TOKENS) == 0u)
	{
		SparkModelDriverRejectAdmission(
			decision,
			SPARK_MODEL_DRIVER_ADMISSION_REJECTED_KV_CAPACITY,
			decision->available_dispatch_slot_count);
	}
	else
	{
		decision->accepted = 1u;
		decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkQwen4FlashResidentDecodeStageAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
	SparkQwen4FlashModuleState *state;
	SparkAdmissionPolicyTable table;
	uint32_t available_slot_count;
	SparkStatus status;

	state = (SparkQwen4FlashModuleState *)module_state;
	if (state == 0 || request == 0 || decision == 0)
		return(SPARK_STATUS_INVALID_ARGUMENT);
	available_slot_count = SparkStageModuleSlotCountFree(
		state->slot_states,
		state->pipeline_slot_count);
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_ADMISSION_ABI_VERSION;
	table.descriptor_bytes = (uint32_t)sizeof(table);
	table.max_active_sequence_count = state->max_active_sequence_count;
	table.max_input_row_count = state->max_active_sequence_count;
	table.max_sequence_positions = SPARK_QWEN4_FLASH_MODEL_MAXIMUM_CONTEXT_TOKENS;
	table.flags = SPARK_ADMISSION_POLICY_FLAG_PREFILL_SINGLE_SLOT |
		SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS;
	table.predicate = SparkQwen4FlashAdmissionKvPredicate;
	table.predicate_context = state;
	table.cost = SparkQwen4FlashAdmissionCost;
	table.cost_context = state;
	status = SparkAdmissionEvaluateShape(&table,available_slot_count,request,decision);
	if (status != SPARK_STATUS_OK)
		return(status);
	if (decision->accepted == 0u)
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	return(status);
}

SparkStatus SparkQwen4FlashResidentDecodeStageSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkQwen4FlashModuleState *state;

	state = (SparkQwen4FlashModuleState *)module_state;
	if (state == 0 || snapshot == 0 || program_id == 0u)
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkStageModuleRuntimeSnapshotInitialize(
		snapshot,
		program_id,
		state->slot_states,
		state->pipeline_slot_count);
	snapshot->submitted_count = atomic_load_explicit(&state->submitted_count,memory_order_relaxed);
	snapshot->completed_count = atomic_load_explicit(&state->completed_count,memory_order_relaxed);
	snapshot->rejected_count = atomic_load_explicit(&state->rejected_count,memory_order_relaxed);
	snapshot->kv_token_capacity = (uint64_t)state->kv_block_count * SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	return(SPARK_STATUS_OK);
}

void SparkQwen4FlashResidentDecodeStageDestroy(void *module_state)
{
	SparkQwen4FlashModuleState *state = (SparkQwen4FlashModuleState *)module_state;
	if ( state == 0 )
		return;
	SparkStageKvClientClose(&state->kv_client);
	free(state->ple_prev_context_u32);
	{
		uint32_t slot_index;
		for (slot_index = 0u; slot_index < SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT; slot_index++)
			free(state->slots[slot_index].ple_host_history_u32);
	}
	if ( state->tp_collective_initialized != 0u )
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
	free(state->kv_logical_to_slot);
	free(state->kv_slot_lane);
	free(state->kv_slot_logical);
	free(state->kv_slot_sequence);
	free(state->kv_slot_dirty);
	free(state->kv_slot_pinned);
	free(state->kv_slot_free_stack);
	free(state->kv_block_staging);
	free(state->kv_gdn_staging);
	free(state->kv_table_indices_host);
	if ( state->kv_table_indices_device != 0 )
		cudaFree(state->kv_table_indices_device);
	if ( state->kv_table_counts_device != 0 )
		cudaFree(state->kv_table_counts_device);
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state);
}

/* ---------------------------------------------------------------------------
 * Execution core: slot allocation and the per-layer runners.
 * -------------------------------------------------------------------------*/

#define SPARK_QWEN4_FLASH_MODULE_STAGED_ROW_CAPACITY \
	(SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT + SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS)



extern cudaError_t SparkQwen4FlashLaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count);
extern cudaError_t SparkQwen4FlashLaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkQwen4FlashLaunchFusedResidualRmsNorm(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkQwen4FlashLaunchLinear(cudaStream_t stream, const SparkQwen4FlashLinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkQwen4FlashLaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen4FlashGdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen4FlashGdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal, uint32_t tp_degree);
extern cudaError_t SparkQwen4FlashLaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen4FlashGdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count, uint32_t tp_degree);
extern cudaError_t SparkQwen4FlashLaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen4FlashGdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal, uint32_t tp_degree);
extern cudaError_t SparkQwen4FlashLaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen4FlashGdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon, uint32_t tp_degree);
extern cudaError_t SparkQwen4FlashLaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen4FlashAttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon, uint32_t tp_degree, uint32_t tp_rank);
extern cudaError_t SparkQwen4FlashLaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen4FlashKvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t tp_degree, uint32_t tp_rank, const uint8_t *token_mask, uint32_t mask_stride);
extern cudaError_t SparkQwen4FlashLaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkQwen4FlashLaunchGateScores(cudaStream_t stream, const SparkQwen4FlashLinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count);
extern cudaError_t SparkQwen4FlashLaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32);
extern cudaError_t SparkQwen4FlashLaunchMoeRoute(cudaStream_t stream, const uint32_t *route_expert, uint32_t rows, uint32_t expert_width, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2);
extern cudaError_t SparkQwen4FlashLaunchFusedExpertW13Act(cudaStream_t stream, const SparkQwen4FlashLinearView *w1, const SparkQwen4FlashLinearView *w3, const void *input_bf16, const uint32_t *route_source_token, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *activated_bf16, uint32_t rows, uint32_t expert_width, float limit, uint32_t multiprocessor_count);
extern cudaError_t SparkQwen4FlashLaunchExpertDown(cudaStream_t stream, const SparkQwen4FlashLinearView *stacked, const void *input_bf16, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t expert_width, uint32_t hidden_dimension, uint32_t multiprocessor_count);
extern cudaError_t SparkQwen4FlashLaunchMoePairReduce(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint32_t row_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen4FlashLaunchMoePairReduceOverwrite(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *output_bf16, uint32_t row_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen4FlashLaunchGroupedExpertLinear(cudaStream_t stream, const SparkQwen4FlashLinearView *view, const void *input_bf16, const uint32_t *source_row_map, const uint32_t *group_row_offset, const uint32_t *group_tile_prefix, void *output_bf16, uint32_t source_row_count, uint32_t multiprocessor_count, uint32_t tp_degree, uint32_t tp_rank, uint32_t route_group_base, const void *frame_error);
extern cudaError_t SparkQwen4FlashLaunchGroupedExpertTileLinear(cudaStream_t stream, const SparkQwen4FlashLinearView *view, const void *input_bf16, const uint32_t *source_row_map, const uint32_t *group_row_offset, void *output_bf16, uint32_t source_row_count, uint32_t tp_degree, uint32_t tp_rank, uint32_t route_group_base);
/* The MoE switches from the scalar grouped path to the tensor-core tile
 * path (SparkLmExpertTileAllKernel / SparkLmExpertTileAllMloopKernel) at
 * 8 rows. Measured on spark4: tile-at-1/2/4 LOSES (the 512-expert grid's
 * ~8K dead CTAs dominate a tiny batch), tile-at-8 wins 36.4 -> 21.7 ms,
 * tile-at-12 wins 54.5 -> 23.9 ms. The dense linears use the force-tile
 * path at every batch instead (no expert dimension, no dead CTAs). */
#define SPARK_QWEN4_FLASH_MODULE_MOE_TILE_ROWS 8u
extern cudaError_t SparkQwen4FlashLaunchSwiGlu(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkQwen4FlashLaunchSharedGate(cudaStream_t stream, void *accum_bf16, const void *gate_weight_bf16, const void *gate_input_bf16, uint32_t row_count, uint32_t dimension);

static SparkStatus SparkQwen4FlashModuleAllocateSlot(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	/* The residual vector is the hc STREAM tensor [rows, 4H]; the sublayer
	 * input (the hc mean-mix) is [rows, H] in normalized_bf16. */
	uint64_t stream_bytes = rows * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES;
	uint64_t hidden_bytes = rows * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES;
	uint64_t expert_bytes = rows * SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES;
	uint64_t moe_up_bytes = rows * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN * SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES;
	/* The w2 output holds rows * top_k packed rows at the HIDDEN width,
	 * not the expert width - the pair reduce reads it back at hidden. */
	uint64_t moe_down_bytes = rows * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN * SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES;
	SparkStatus status;
	cudaStream_t stream = 0;
	status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,cudaStreamCreate(&stream),"cudaStreamCreate");
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot->cuda_stream = stream;
	/* device-private: the frame error slot kernels report corruption into. */
	status = SparkStageModuleDeviceAllocate(&state->ledger,SPARK_FRAME_ERROR_WORDS * sizeof(uint32_t),(void **)&slot->frame_error);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->input_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->output_token_ids);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
	{
		/* Screened head: coarse logits at bf16 rows x shard vocab, screen
		 * candidates at rows x SPARK_LM_HEAD_SCREEN_CAP, counts, plus the
		 * sharded-argmax staging (per-row best score + maxloc key). */
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp_vocab_rows * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,(void **)&slot->head_logits_bf16);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODULE_HEAD_SCREEN_CAP * sizeof(uint32_t),(void **)&slot->head_candidate_ids);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->head_candidate_counts);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(float),(void **)&slot->head_scores_f32);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint64_t),(void **)&slot->head_maxloc_u64);
		/* MTP draft ids: MAX_MTP_DRAFT_TOKENS slots, one per draft step.
		 * (Missing from the M5-prep port - the chain dereferenced NULL on
		 * its first whole-stack exercise; the mid-pipeline tier never ran
		 * MTP, so nothing caught it until the TP4 whole-stack run.) */
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS * sizeof(uint32_t),(void **)&slot->mtp_draft_ids);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_lane_indices);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->slot_mapping);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_cold);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint64_t),(void **)&slot->row_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->hc_normed_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->hc_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_HC_LOWRANK_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->hc_lowrank_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_COUNT * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->hc_inject_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->normalized_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_GDN_CONV_CHANNELS * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->qkv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_GDN_CONV_CHANNELS * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->conv_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->z_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->beta_pre_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->decay_pre_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT * sizeof(float),(void **)&slot->log_decay_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT * sizeof(float),(void **)&slot->beta_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->core_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->gated_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * 2u * SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->q_fused_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_ATTN_KV_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->k_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_ATTN_KV_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->v_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_ATTN_QUERY_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->head_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT * sizeof(float),(void **)&slot->moe_scores_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->moe_indices_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN * sizeof(float),(void **)&slot->moe_weights_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(rows * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN + SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT + 2u) * sizeof(uint32_t),(void **)&slot->moe_inverse_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(rows * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN + SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT + 2u) * sizeof(uint32_t),(void **)&slot->moe_grouped_rows_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->moe_tile_prefix_w1_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->moe_tile_prefix_w2_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->moe_group_offset_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,moe_up_bytes,&slot->moe_gate_packed_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,moe_up_bytes,&slot->moe_slot_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,moe_down_bytes,&slot->moe_slot_out_bf16);
	/* Indexer scratch (attn layers exist on every whole-stack slice). */
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&slot->indexer_query_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * state->indexer_mask_stride,(void **)&slot->indexer_mask_u8);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * state->kv_block_count * sizeof(uint32_t),(void **)&slot->indexer_scores_u32);
	/* PLE scratch [rows, *] at stream/hidh widths (owns_ple set post-pack). */
	if ( status == SPARK_STATUS_OK && state->owns_ple != 0u )
	{
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->ple_embedding_bf16);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->ple_key_bf16);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->ple_value_bf16);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->ple_gated_bf16);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->ple_gated_normed_bf16);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->ple_conv_out_bf16);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,rows * 3u * sizeof(uint32_t),(void **)&slot->ple_history_u32);
		slot->ple_host_history_u32 = slot->ple_history_u32 != 0 ? (uint32_t *)malloc((size_t)rows * 3u * sizeof(uint32_t)) : 0;
		if ( status == SPARK_STATUS_OK && slot->ple_host_history_u32 == 0 )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,expert_bytes,&slot->shared_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,expert_bytes,&slot->shared_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->shared_down_bf16);
	return(status);
}

static cudaError_t SparkQwen4FlashModuleRunGdnCoreDecode(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, const SparkQwen4FlashGdnLayerWeights *weights, uint32_t rows, uint32_t ordinal)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	SparkQwen4FlashGdnStatePool pool = state->gdn_pool;
	pool.state_cold_by_row = slot->row_cold;
	error = SparkQwen4FlashLaunchConvUpdate(stream,slot->qkv_bf16,weights,slot->conv_out_bf16,&pool,slot->row_lane_indices,rows,ordinal,state->tp_degree);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchDecayBeta(stream,slot->decay_pre_bf16,slot->beta_pre_bf16,weights,slot->log_decay_f32,slot->beta_f32,rows,state->tp_degree);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchGdnStep(stream,slot->conv_out_bf16,slot->log_decay_f32,slot->beta_f32,&pool,slot->core_bf16,slot->row_lane_indices,rows,ordinal,state->tp_degree);
	return(error);
}

static SparkStatus SparkQwen4FlashModuleRunGdnLayer(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, uint32_t layer, uint32_t rows)
{
	const SparkQwen4FlashGdnLayerWeights *weights = &state->gdn_by_layer[layer];
	uint32_t ordinal = state->gdn_ordinal_by_layer[layer];
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkQwen4FlashLaunchLinear(stream,&weights->qkv,slot->normalized_bf16,slot->qkv_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&weights->gate,slot->normalized_bf16,slot->z_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&weights->beta,slot->normalized_bf16,slot->beta_pre_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&weights->decay,slot->normalized_bf16,slot->decay_pre_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashModuleRunGdnCoreDecode(state,slot,weights,rows,ordinal);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchGatedNorm(stream,slot->core_bf16,slot->z_bf16,weights,slot->gated_bf16,rows,SPARK_QWEN4_FLASH_MODEL_RMS_NORM_EPSILON,state->tp_degree);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&weights->output,slot->gated_bf16,slot->delta_bf16,rows);
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"gdn_layer"));
}

typedef struct SparkQwen4FlashAttnRowsView
{
	const uint32_t *slot_mapping;
	const uint64_t *row_positions;
	const uint32_t *row_lane_indices;
	const uint32_t *context_lengths;
} SparkQwen4FlashAttnRowsView;

static SparkStatus SparkQwen4FlashModuleRunAttnLayer(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, const SparkQwen4FlashKvBlockTableView *table, const SparkQwen4FlashAttnLayerWeights *weights, const SparkQwen4FlashIndexerWeights *indexer, uint32_t ordinal, uint32_t indexer_ordinal, const SparkQwen4FlashAttnRowsView *rows_view, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint8_t *token_mask = 0;
	cudaError_t error;
	error = SparkQwen4FlashLaunchLinear(stream,&weights->query,slot->normalized_bf16,slot->q_fused_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&weights->key,slot->normalized_bf16,slot->k_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&weights->value,slot->normalized_bf16,slot->v_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchAttnPrepare(stream,slot->q_fused_bf16,slot->k_bf16,slot->v_bf16,weights,state->kv_cache_bf16,rows_view->slot_mapping,rows_view->row_positions,rows,ordinal,state->cache_layer_stride,state->cache_block_stride,SPARK_QWEN4_FLASH_MODEL_RMS_NORM_EPSILON,state->tp_degree,state->tp_rank);
	/* Indexer: project + prepare the frame's queries and keys (pooling any
	 * completing block), then score/select into the u8 token mask the
	 * decode kernel consumes. Contexts under budget+ratio-1 select every
	 * visible token, so short contexts run bit-identically to before. */
	if ( error == cudaSuccess && indexer != 0 )
	{
		void *raw_plane = (uint8_t *)state->indexer_raw_key_cache_bf16 + ((uint64_t)indexer_ordinal * state->indexer_raw_layer_stride * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES);
		void *pooled_plane = (uint8_t *)state->indexer_pooled_key_cache_bf16 + ((uint64_t)indexer_ordinal * state->indexer_pooled_layer_stride * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES);
		error = SparkQwen4FlashLaunchLinear(stream,&indexer->index_qk,slot->normalized_bf16,slot->qkv_bf16,rows);
		if ( error == cudaSuccess )
			error = SparkQwen4FlashLaunchIndexerPrepare(stream,slot->qkv_bf16,indexer,slot->indexer_query_bf16,raw_plane,pooled_plane,rows_view->slot_mapping,table->physical_block_indices,rows_view->row_positions,rows,table->lane_stride,state->indexer_raw_layer_stride / SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
		if ( error == cudaSuccess )
			error = SparkQwen4FlashLaunchIndexerSelect(stream,slot->indexer_query_bf16,pooled_plane,table,rows_view->row_lane_indices,rows_view->context_lengths,slot->indexer_mask_u8,slot->indexer_scores_u32,rows,state->indexer_mask_stride,state->kv_block_count);
		if ( error == cudaSuccess )
			token_mask = slot->indexer_mask_u8;
	}
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchAttnDecode(stream,slot->q_fused_bf16,state->kv_cache_bf16,table,rows_view->row_lane_indices,rows_view->context_lengths,slot->head_out_bf16,rows,ordinal,state->cache_layer_stride,state->cache_block_stride,state->tp_degree,state->tp_rank,token_mask,token_mask != 0 ? state->indexer_mask_stride : 0u);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&weights->output,slot->head_out_bf16,slot->delta_bf16,rows);
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"attn_layer"));
}

/*
 * Routed MoE + shared expert, the layer's FFN side. The hc mean-mix already
 * produced the sublayer input in normalized_bf16 (the reference SparseMoe
 * block applies NO input norm of its own); delta_bf16 is OVERWRITTEN by the
 * routed expert mixture, then accumulates the shared expert partial, and the
 * single TP all-reduce completes BOTH partial sums (routed experts are
 * expert-sharded and the shared expert's down projection is column-sharded -
 * reducing before the shared add left the ranks divergent at live TP>1).
 */
/* The route's group base for this rank: a REPLICATED gate (full expert rows)
 * routes the same global top-k on every rank, so this rank executes the
 * groups [rank*experts_per_rank, (rank+1)*experts_per_rank); a NARROWED gate
 * (the deployed packs) selects and routes rank-local experts only, so the
 * base is 0. Derived from the gate view's width - both pack generations
 * load and run. */
static uint32_t SparkQwen4FlashModuleRouteGroupBase(SparkQwen4FlashModuleState *state, const SparkQwen4FlashMoeWeights *weights)
{
	if ( weights->gate.output_dimension == SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT )
		return(state->tp_rank * (SPARK_QWEN4_FLASH_MODEL_ROUTED_EXPERT_COUNT / state->tp_degree));
	return(0u);
}

static SparkStatus SparkQwen4FlashModuleRunMoe(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, const SparkQwen4FlashMoeWeights *weights, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t route_group_base = SparkQwen4FlashModuleRouteGroupBase(state,weights);
	cudaError_t error;
	SparkStatus status;
	error = SparkQwen4FlashLaunchGateScores(stream,&weights->gate,slot->normalized_bf16,slot->moe_scores_f32,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchGateSelect(stream,slot->moe_scores_f32,0,rows,weights->gate.output_dimension,SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN,1.0f,slot->moe_indices_u32,slot->moe_weights_f32);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchMoeRoute(stream,slot->moe_indices_u32,rows,SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION,slot->moe_group_offset_u32,slot->moe_inverse_u32,slot->moe_grouped_rows_u32,slot->moe_tile_prefix_w1_u32,slot->moe_tile_prefix_w2_u32);
	if ( error == cudaSuccess )
	{
		if ( rows >= SPARK_QWEN4_FLASH_MODULE_MOE_TILE_ROWS )
		{
			/* Tensor-core tile path: gridDim.z spans the experts, FP8
			 * decodes to BF16 fragments under wmma mma_sync. */
			error = SparkQwen4FlashLaunchGroupedExpertTileLinear(stream,&weights->experts_w1,slot->normalized_bf16,slot->moe_grouped_rows_u32,slot->moe_group_offset_u32,slot->moe_gate_packed_bf16,rows,state->tp_degree,state->tp_rank,route_group_base);
			if ( error == cudaSuccess )
				error = SparkQwen4FlashLaunchGroupedExpertTileLinear(stream,&weights->experts_w3,slot->normalized_bf16,slot->moe_grouped_rows_u32,slot->moe_group_offset_u32,slot->moe_slot_up_bf16,rows,state->tp_degree,state->tp_rank,route_group_base);
			if ( error == cudaSuccess )
				error = SparkQwen4FlashLaunchSwiGlu(stream,slot->moe_gate_packed_bf16,slot->moe_slot_up_bf16,rows * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN,SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION);
			if ( error == cudaSuccess )
				error = SparkQwen4FlashLaunchGroupedExpertTileLinear(stream,&weights->experts_w2,slot->moe_slot_up_bf16,0,slot->moe_group_offset_u32,slot->moe_slot_out_bf16,rows * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN,state->tp_degree,state->tp_rank,route_group_base);
		}
		else
		{
			error = SparkQwen4FlashLaunchGroupedExpertLinear(stream,&weights->experts_w1,slot->normalized_bf16,slot->moe_grouped_rows_u32,slot->moe_group_offset_u32,slot->moe_tile_prefix_w1_u32,slot->moe_gate_packed_bf16,rows,state->multiprocessor_count,state->tp_degree,state->tp_rank,route_group_base,slot->frame_error);
			if ( error == cudaSuccess )
				error = SparkQwen4FlashLaunchGroupedExpertLinear(stream,&weights->experts_w3,slot->normalized_bf16,slot->moe_grouped_rows_u32,slot->moe_group_offset_u32,slot->moe_tile_prefix_w1_u32,slot->moe_slot_up_bf16,rows,state->multiprocessor_count,state->tp_degree,state->tp_rank,route_group_base,slot->frame_error);
			if ( error == cudaSuccess )
				error = SparkQwen4FlashLaunchSwiGlu(stream,slot->moe_gate_packed_bf16,slot->moe_slot_up_bf16,rows * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN,SPARK_QWEN4_FLASH_MODEL_EXPERT_INTERMEDIATE_DIMENSION);
			if ( error == cudaSuccess )
				error = SparkQwen4FlashLaunchGroupedExpertLinear(stream,&weights->experts_w2,slot->moe_slot_up_bf16,0,slot->moe_group_offset_u32,slot->moe_tile_prefix_w2_u32,slot->moe_slot_out_bf16,rows * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN,state->multiprocessor_count,state->tp_degree,state->tp_rank,route_group_base,slot->frame_error);
		}
	}
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchMoePairReduceOverwrite(stream,slot->moe_slot_out_bf16,slot->moe_inverse_u32,slot->moe_weights_f32,slot->delta_bf16,rows,SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&weights->shared_gate,slot->normalized_bf16,slot->shared_gate_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&weights->shared_up,slot->normalized_bf16,slot->shared_up_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchSwiGlu(stream,slot->shared_gate_bf16,slot->shared_up_bf16,rows,weights->shared_gate.output_dimension);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&weights->shared_down,slot->shared_up_bf16,slot->shared_down_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchSharedGate(stream,slot->shared_down_bf16,weights->shared_gate_weight_bf16,slot->normalized_bf16,rows,SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchResidualAdd(stream,slot->delta_bf16,slot->shared_down_bf16,rows,SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess && state->tp_degree > 1u )
	{
		/* Expert-sharded TP: the pair reduce holds only THIS rank's
		 * experts' contribution and the shared expert's down projection is
		 * column-sharded, so the sum completing delta must cover both
		 * (reducing between them left the shared partial rank-local and the
		 * streams diverged from the next layer on - invisible under the
		 * standalone bypass, fatal for the live collective). */
		status = SparkQwen4FlashModuleTpAllReduceHidden(state,slot,slot->delta_bf16,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"moe"));
}

/* Debug: dump the post-layer hidden to <env path>.<frame>.<layer> when
 * SPARK_QWEN4_FLASH_STAGE_DEBUG_DUMP_HIDDEN names a directory prefix. Used
 * once to localize the decode-vs-prefill divergence at TP4; off by default.
 */
static uint32_t SparkQwen4FlashModuleDebugDumpHidden(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, uint32_t layer, uint32_t rows)
{
	const char *prefix = getenv("SPARK_QWEN4_FLASH_STAGE_DEBUG_DUMP_HIDDEN");
	char path[512];
	FILE *out;
	if ( prefix == 0 )
		return(0u);
	snprintf(path,sizeof(path),"%s.f%u.t%u.l%03u",prefix,state->debug_frame_serial,state->debug_token_serial,layer);
	out = fopen(path,"wb");
	if ( out == 0 )
		return(0u);
	{
		static uint16_t staging[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH];
		uint64_t stream_row_bytes = (uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES;
		if ( cudaMemcpy(staging,slot->hidden_bf16,(size_t)rows * stream_row_bytes,cudaMemcpyDeviceToHost) == cudaSuccess )
			fwrite(staging,(size_t)stream_row_bytes,rows,out);
	}
	fclose(out);
	return(0u);
}

/* One hc sublayer front half: group-norm the streams, low-rank mean-mix into
 * the sublayer input (normalized_bf16), and stage the raw inject projection.
 * The reference chain: silu(down(normed)/hc_count) -> up -> sigmoid gates the
 * stream mean. */
static SparkStatus SparkQwen4FlashModuleRunHcPrep(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, const SparkQwen4FlashHcWeights *hc, const void *hc_norm_weight, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	(void)state;
	cudaError_t error = SparkQwen4FlashLaunchHcGroupNorm(stream,slot->hidden_bf16,hc_norm_weight,slot->hc_normed_bf16,rows,SPARK_QWEN4_FLASH_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&hc->mix_down,slot->hc_normed_bf16,slot->hc_lowrank_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchHcSiluQuarter(stream,slot->hc_lowrank_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&hc->mix_up,slot->hc_lowrank_bf16,slot->hc_up_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchHcMix(stream,slot->hc_up_bf16,slot->hc_normed_bf16,slot->normalized_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&hc->block_inject,slot->hc_normed_bf16,slot->hc_inject_bf16,rows);
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"hc_prep"));
}

/* Back half: fold the (already all-reduced) sublayer output back into every
 * stream at the per-stream inject scale. */
static SparkStatus SparkQwen4FlashModuleRunHcInject(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	(void)state;
	cudaError_t error = SparkQwen4FlashLaunchHcInject(stream,slot->hidden_bf16,slot->hc_inject_bf16,slot->delta_bf16,rows);
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"hc_inject"));
}

/* Readout mixer (use_combine=False): group norm + low-rank mean-mix, no
 * inject, no further norm - the stack's final readout and the MTP copies. */
static SparkStatus SparkQwen4FlashModuleRunHcReadout(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, const SparkQwen4FlashHcMixer *mixer, const void *streams, void *mixed_out, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	(void)state;
	cudaError_t error = SparkQwen4FlashLaunchHcGroupNorm(stream,streams,mixer->hc_norm_weight_bf16,slot->hc_normed_bf16,rows,SPARK_QWEN4_FLASH_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&mixer->mix_down,slot->hc_normed_bf16,slot->hc_lowrank_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchHcSiluQuarter(stream,slot->hc_lowrank_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&mixer->mix_up,slot->hc_lowrank_bf16,slot->hc_up_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchHcMix(stream,slot->hc_up_bf16,slot->hc_normed_bf16,mixed_out,rows);
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"hc_readout"));
}

/* PLE injection for the PLE layer, run BEFORE the attention hc (the
 * reference DecoderLayer order): hash the lane's EOS-segmented 3-gram
 * history per head, gather the vocab-sharded n-gram table (the bf16
 * all-reduce completes the [rows, 2560] embedding), then key/value/gate/
 * dilated-conv into a [rows, 4H] delta added to the stream vector. The
 * module walks one token per row (decode microbatch or the per-token
 * prefill), so token_count is always 1 and the conv is the update form. */
static SparkStatus SparkQwen4FlashModuleRunPle(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, const uint32_t *host_token_ids, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status;
	cudaError_t error;
	uint32_t row;
	/* Stage the hash window per row: [prev0, prev1, current id]; cold rows
	 * read EOS EOS (the reference initialization). */
	for (row = 0u; row < rows; row++)
	{
		uint32_t lane = slot->host_row_lane_indices[row];
		uint32_t cold = slot->host_row_cold[row];
		uint64_t lane_base = (uint64_t)lane * SPARK_QWEN4_FLASH_MODEL_PLE_CONTEXT_LENGTH;
		slot->ple_host_history_u32[row * 3u + 0u] = cold != 0u ? SPARK_QWEN4_FLASH_MODEL_EOS_TOKEN_ID : state->ple_prev_context_u32[lane_base + 0u];
		slot->ple_host_history_u32[row * 3u + 1u] = cold != 0u ? SPARK_QWEN4_FLASH_MODEL_EOS_TOKEN_ID : state->ple_prev_context_u32[lane_base + 1u];
		slot->ple_host_history_u32[row * 3u + 2u] = host_token_ids[row];
	}
	error = cudaMemcpyAsync(slot->ple_history_u32,slot->ple_host_history_u32,(size_t)rows * 3u * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"ple_history"));
	error = SparkQwen4FlashLaunchPleHashGather(stream,slot->ple_history_u32,1u,&state->ple,slot->ple_embedding_bf16,rows,state->tp_degree > 1u ? (uint32_t)((uint64_t)state->tp_rank * ((uint64_t)SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_ROW_COUNT / state->tp_degree)) : 0u,state->tp_degree > 1u ? SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_ROW_COUNT / state->tp_degree : SPARK_QWEN4_FLASH_MODEL_PLE_NGRAM_ROW_COUNT);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"ple_hash"));
	if ( state->tp_degree > 1u )
	{
		status = SparkQwen4FlashModuleTpAllReduceHidden(state,slot,slot->ple_embedding_bf16,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	/* Key/value projections and the group norms (query norm reads the
	 * CURRENT stream vector; the hc_normed scratch holds the query side,
	 * ple_gated_normed the key side). */
	error = SparkQwen4FlashLaunchHcGroupNorm(stream,slot->hidden_bf16,state->ple.norm_query_weight_bf16,slot->hc_normed_bf16,rows,SPARK_QWEN4_FLASH_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&state->ple.key_proj,slot->ple_embedding_bf16,slot->ple_key_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchHcGroupNorm(stream,slot->ple_key_bf16,state->ple.norm_key_weight_bf16,slot->ple_gated_normed_bf16,rows,SPARK_QWEN4_FLASH_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&state->ple.value_proj,slot->ple_embedding_bf16,slot->ple_value_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchPleGate(stream,slot->ple_gated_normed_bf16,slot->hc_normed_bf16,slot->ple_value_bf16,slot->ple_gated_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchHcGroupNorm(stream,slot->ple_gated_bf16,state->ple.norm_conv_weight_bf16,slot->ple_gated_normed_bf16,rows,SPARK_QWEN4_FLASH_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchPleConvUpdate(stream,slot->ple_gated_normed_bf16,&state->ple,slot->ple_conv_out_bf16,state->ple_conv_tail_bf16,slot->row_lane_indices,slot->row_cold,rows,state->ple_conv_tail_lane_stride);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchResidualAdd(stream,slot->hidden_bf16,slot->ple_gated_bf16,rows,SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchResidualAdd(stream,slot->hidden_bf16,slot->ple_conv_out_bf16,rows,SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"ple_core"));
	/* Carry the last two tokens (prev1, current) for the next frame. */
	for (row = 0u; row < rows; row++)
	{
		uint32_t lane = slot->host_row_lane_indices[row];
		uint64_t lane_base = (uint64_t)lane * SPARK_QWEN4_FLASH_MODEL_PLE_CONTEXT_LENGTH;
		state->ple_prev_context_u32[lane_base] = slot->ple_host_history_u32[row * 3u + 1u];
		state->ple_prev_context_u32[lane_base + 1u] = slot->ple_host_history_u32[row * 3u + 2u];
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen4FlashModuleRunLayer(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, const SparkQwen4FlashKvBlockTableView *table, uint32_t layer, uint32_t rows, const uint32_t *host_token_ids)
{
	SparkQwen4FlashAttnRowsView rows_view;
	SparkStatus status;
	rows_view.slot_mapping = slot->slot_mapping;
	rows_view.row_positions = slot->row_positions;
	rows_view.row_lane_indices = slot->row_lane_indices;
	rows_view.context_lengths = slot->context_lengths;
	/* PLE first (the reference DecoderLayer runs the n-gram injection
	 * before the attention hyper-connection). */
	if ( layer == SPARK_QWEN4_FLASH_MODEL_PLE_LAYER_INDEX && state->owns_ple != 0u )
	{
		status = SparkQwen4FlashModuleRunPle(state,slot,host_token_ids,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	/* Attention/GDN sublayer: hc mean-mix in, output projection is
	 * column-parallel at tp>1 so the delta all-reduce completes it before
	 * the per-stream inject (the missing reduce was invisible under the
	 * standalone bypass and fatal for the live collective). */
	status = SparkQwen4FlashModuleRunHcPrep(state,slot,&state->attn_hc_by_layer[layer],state->attention_norm_by_layer[layer],rows);
	if ( status == SPARK_STATUS_OK && state->debug_skip_gdn == 0u )
		status = SPARK_QWEN4_FLASH_MODEL_LAYER_IS_GDN(layer) != 0u ? SparkQwen4FlashModuleRunGdnLayer(state,slot,layer,rows) : SparkQwen4FlashModuleRunAttnLayer(state,slot,table,&state->attn_by_layer[layer],&state->indexer_by_layer[layer],state->attn_ordinal_by_layer[layer],state->attn_ordinal_by_layer[layer],&rows_view,rows);
	if ( status == SPARK_STATUS_OK && state->tp_degree > 1u )
	{
		status = SparkQwen4FlashModuleTpAllReduceHidden(state,slot,slot->delta_bf16,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleRunHcInject(state,slot,rows);
	/* MoE sublayer on the same stream machinery. */
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleRunHcPrep(state,slot,&state->mlp_hc_by_layer[layer],state->mlp_norm_by_layer[layer],rows);
	if ( status == SPARK_STATUS_OK && state->debug_skip_moe == 0u )
		status = SparkQwen4FlashModuleRunMoe(state,slot,&state->moe_by_layer[layer],rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleRunHcInject(state,slot,rows);
	if ( status == SPARK_STATUS_OK )
		state->debug_dump_hidden(state,slot,layer,rows);
	return(status);
}

/* ---------------------------------------------------------------------------
 * Pools, slot host mirrors, and the lean decode Execute path.
 * Decode-only for now: prefill, speculation and the KV tier land with the
 * serving adapter. The attention block table is a one-block-per-lane view
 * (valid for contexts up to KV_BLOCK_TOKENS); the adapter replaces it.
 * -------------------------------------------------------------------------*/

#define SPARK_QWEN4_FLASH_MODULE_HOST_ROW_CAPACITY \
	(SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT + SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS)

static SparkStatus SparkQwen4FlashModuleAllocatePools(SparkQwen4FlashModuleState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint64_t state_elements,tail_elements,cache_elements;
	state->gdn_pool.abi_version = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_GDN_STATE_POOL_ABI_VERSION;
	state->gdn_pool.lane_capacity = state->max_active_sequence_count;
	state->gdn_pool.gdn_layer_count = state->gdn_layer_count;
	state->gdn_pool.state_layer_stride_elements = (uint64_t)SPARK_QWEN4_FLASH_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_KEY_DIMENSION * SPARK_QWEN4_FLASH_MODEL_GDN_HEAD_VALUE_DIMENSION;
	state->gdn_pool.state_lane_stride_elements = state->gdn_pool.state_layer_stride_elements * state->gdn_layer_count;
	state->gdn_pool.conv_tail_layer_stride_elements = (uint64_t)SPARK_QWEN4_FLASH_MODEL_GDN_CONV_CHANNELS * (SPARK_QWEN4_FLASH_MODEL_GDN_CONV_KERNEL - 1u);
	state->gdn_pool.conv_tail_lane_stride_elements = state->gdn_pool.conv_tail_layer_stride_elements * state->gdn_layer_count;
	if ( state->gdn_layer_count != 0u )
	{
		state_elements = state->gdn_pool.state_lane_stride_elements * state->max_active_sequence_count;
		tail_elements = state->gdn_pool.conv_tail_lane_stride_elements * state->max_active_sequence_count;
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,state_elements * sizeof(float),(void **)&state->gdn_pool.state_f32);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,tail_elements * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&state->gdn_pool.conv_tail_bf16);
	}
	/* The MTP decoder's attention runs at its OWN cache ordinal one past
	 * the main attention layers; the cache must therefore allocate
	 * attn_layer_count + 1 layers when the MTP tensors are loaded
	 * (owns_final_head). With cache_layer_count == attn_layer_count the MTP
	 * prepare/decode indexed one layer PAST the allocation - an out-of-
	 * bounds write that happened to land in adjacent device memory. */
	state->mtp_cache_ordinal = state->attn_layer_count;
	state->cache_layer_count = state->attn_layer_count + (state->owns_final_head != 0u ? 1u : 0u);
	if ( status == SPARK_STATUS_OK && state->cache_layer_count != 0u )
	{
		state->cache_layer_stride = (uint64_t)SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * SPARK_QWEN4_FLASH_MODEL_ATTN_CACHE_TOKEN_ELEMENTS;
		state->cache_block_stride = state->cache_layer_stride * state->cache_layer_count;
		cache_elements = state->cache_block_stride * state->kv_block_count;
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,cache_elements * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&state->kv_cache_bf16);
	}
	/* Indexer caches: raw per-token keys [layers, blocks, 64, 128] and
	 * pooled per-block keys [layers, blocks, 128]; one plane per attention
	 * layer plus the MTP ordinal when this stage owns the head. */
	state->indexer_cache_layer_count = state->cache_layer_count;
	state->indexer_mask_stride = state->kv_block_count * SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	if ( status == SPARK_STATUS_OK && state->indexer_cache_layer_count != 0u )
	{
		uint64_t raw_elements = (uint64_t)state->indexer_cache_layer_count * state->kv_block_count *
			SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION;
		/* Pooled rows are per COMPRESSED block: kv_block_count x
		 * (KV_BLOCK_TOKENS / compress_ratio) rows of 128. */
		uint64_t pooled_rows_per_block = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS / SPARK_QWEN4_FLASH_MODEL_INDEXER_COMPRESS_RATIO;
		uint64_t pooled_elements = (uint64_t)state->indexer_cache_layer_count * state->kv_block_count * pooled_rows_per_block * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION;
		state->indexer_raw_layer_stride = (uint64_t)state->kv_block_count * SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION;
		state->indexer_pooled_layer_stride = (uint64_t)state->kv_block_count * pooled_rows_per_block * SPARK_QWEN4_FLASH_MODEL_INDEXER_HEAD_DIMENSION;
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,raw_elements * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&state->indexer_raw_key_cache_bf16);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,pooled_elements * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&state->indexer_pooled_key_cache_bf16);
	}
	/* PLE carried state: [lanes, 2] u32 prev context (host-resident, tiny)
	 * and the dilated conv tail [lanes, 4H, 9] bf16 on device. */
	if ( status == SPARK_STATUS_OK && state->owns_ple != 0u )
	{
		state->ple_conv_tail_lane_stride = (uint64_t)SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_PLE_CONV_TAIL_COLUMNS;
		state->ple_prev_context_u32 = (uint32_t *)calloc((size_t)state->max_active_sequence_count * SPARK_QWEN4_FLASH_MODEL_PLE_CONTEXT_LENGTH,sizeof(uint32_t));
		status = state->ple_prev_context_u32 != 0 ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED;
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->max_active_sequence_count * state->ple_conv_tail_lane_stride * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,&state->ple_conv_tail_bf16);
	}
	return(status);
}

static SparkStatus SparkQwen4FlashModuleAllocateSlotHostMirrors(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	SparkStatus status = SPARK_STATUS_OK;
	slot->host_row_lane_indices = (uint32_t *)malloc(SPARK_QWEN4_FLASH_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_row_positions = (uint64_t *)malloc(SPARK_QWEN4_FLASH_MODULE_HOST_ROW_CAPACITY * sizeof(uint64_t));
	slot->host_row_cold = (uint32_t *)malloc(rows * sizeof(uint32_t));
	slot->host_slot_mapping = (uint32_t *)malloc(SPARK_QWEN4_FLASH_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_context_lengths = (uint32_t *)malloc(SPARK_QWEN4_FLASH_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_frame_error = (uint32_t *)malloc(SPARK_FRAME_ERROR_WORDS * sizeof(uint32_t));
	if ( slot->host_row_lane_indices == 0 || slot->host_row_positions == 0 ||
		slot->host_row_cold == 0 || slot->host_slot_mapping == 0 ||
		slot->host_context_lengths == 0 || slot->host_frame_error == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(status);
}

static SparkStatus SparkQwen4FlashModuleUploadRows(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, const SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	uint32_t token_guard;
	/* Fresh frame, fresh error record: kernels report route-map corruption
	 * here; the copy-back + check at frame end turns a non-zero code into a
	 * failed frame instead of a dead context (frame_error.cuh). */
	error = cudaMemsetAsync(slot->frame_error,0,SPARK_FRAME_ERROR_WORDS * sizeof(uint32_t),stream);
	memset(slot->host_frame_error,0,SPARK_FRAME_ERROR_WORDS * sizeof(uint32_t));
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_lane_indices,slot->host_row_lane_indices,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_positions,slot->host_row_positions,rows * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_cold,slot->host_row_cold,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->attn_layer_count != 0u )
		error = cudaMemcpyAsync(slot->slot_mapping,slot->host_slot_mapping,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->attn_layer_count != 0u )
		error = cudaMemcpyAsync(slot->context_lengths,slot->host_context_lengths,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->owns_embedding != 0u )
	{
		if ( frame->buffer_count < 1u || frame->buffers == 0 || frame->buffers[0].address == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		for (token_guard = 0; token_guard < rows; token_guard++)
			if ( ((const uint32_t *)frame->buffers[0].address)[token_guard] >= SPARK_QWEN4_FLASH_MODEL_OUTPUT_VOCAB_COUNT )
			{
				fprintf(stderr,"%s token_id_out_of_range row=%u\n",SPARK_QWEN4_FLASH_MODULE_TAG,token_guard);
				return(SPARK_STATUS_INVALID_ARGUMENT);
			}
		error = cudaMemcpyAsync(slot->input_token_ids,frame->buffers[0].address,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	}
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"stage_upload"));
}

/* Resolve a row's KV slot mapping through the frame's block table: the
 * PHYSICAL slot (lane's physical block x block tokens + in-block offset),
 * the same contract the serving adapter supplies. The old smoke formula
 * (position % block tokens) addressed block 0 for EVERY lane, so lane 1's
 * prepare wrote block 0 while its decode read the table's block 1 - never
 * written - which the whole-stack TP4 run exposed as a decode-vs-prefill
 * divergence at the first attention layer. No table (pure smoke): the
 * identity mapping. */
static void SparkQwen4FlashModuleResolveSlotMapping(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, const SparkQwen4FlashKvBlockTableView *table, uint32_t row, uint64_t position)
{
	uint32_t lane,block;
	(void)state;
	if ( table == 0 || table->host_physical_block_indices == 0 || table->lane_stride == 0u )
	{
		slot->host_slot_mapping[row] = (uint32_t)(position % SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
		return;
	}
	lane = slot->host_row_lane_indices[row];
	block = table->host_physical_block_indices[((uint64_t)lane * table->lane_stride) + (uint32_t)(position / SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS)];
	slot->host_slot_mapping[row] = (block * SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) + (uint32_t)(position % SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
}

static cudaError_t SparkQwen4FlashModuleEmitHead(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t out_index = state->owns_embedding != 0u ? 1u : 0u;
	cudaError_t error;
	if ( frame->buffer_count <= out_index || frame->buffers == 0 || frame->buffers[out_index].address == 0 )
		return(cudaErrorInvalidValue);
	/* Final readout = the hyper_connection_mixer (use_combine=False): group
	 * norm + low-rank mean-mix over the streams, NO final norm (the Flash
	 * checkpoint has no model.norm; the old stream-0-section approximation
	 * is retired). */
	if ( SparkQwen4FlashModuleRunHcReadout(state,slot,&state->readout_mixer,slot->hidden_bf16,slot->normalized_bf16,rows) != SPARK_STATUS_OK )
		return(cudaErrorLaunchFailure);
	error = cudaSuccess;
	if ( state->tp_degree > 1u )
	{
		/* Sharded head (the 27b flow): screen THIS rank's vocab shard with
		 * the id offset already applied, fold (score,id) into maxloc keys,
		 * resolve the global argmax through the u64 max collective, unpack.
		 * Standalone keeps the rank-local argmax (validation semantics). */
		if ( error == cudaSuccess && state->head_shadow_payload != 0 )
			error = SparkQwen4FlashLaunchHeadScreenedArgmaxScore(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids,slot->head_candidate_counts,slot->output_token_ids,slot->head_scores_f32,state->tp_vocab_base,rows,state->tp_vocab_rows);
		else if ( error == cudaSuccess )
			error = SparkQwen4FlashLaunchHeadArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,slot->input_token_ids,slot->output_token_ids,rows,state->tp_vocab_rows);
		if ( error == cudaSuccess )
			error = SparkQwen4FlashLaunchHeadMaxLocPack(stream,slot->head_scores_f32,slot->output_token_ids,slot->head_maxloc_u64,rows);
		if ( error == cudaSuccess && SparkQwen4FlashModuleTpReduceU64Max(state,slot,slot->head_maxloc_u64,rows) != SPARK_STATUS_OK )
			return(cudaErrorLaunchFailure);
		if ( error == cudaSuccess )
			error = SparkQwen4FlashLaunchHeadMaxLocUnpack(stream,slot->head_maxloc_u64,slot->output_token_ids,rows);
	}
	else if ( error == cudaSuccess && state->head_shadow_payload != 0 )
		error = SparkQwen4FlashLaunchHeadScreenedArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids,slot->head_candidate_counts,slot->output_token_ids,rows,SPARK_QWEN4_FLASH_MODEL_OUTPUT_VOCAB_COUNT);
	else if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchHeadArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,slot->input_token_ids,slot->output_token_ids,rows,SPARK_QWEN4_FLASH_MODEL_OUTPUT_VOCAB_COUNT);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(frame->buffers[out_index].address,slot->output_token_ids,rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	return(error);
}

/* The firmware's transport contract (firmware header, frame context): a
 * stage with stage_index > 0 requires HIDDEN_INPUT_TRANSPORT on every
 * frame and a stage with stage_index + 1 < stage_count requires
 * HIDDEN_OUTPUT_TRANSPORT; the module refuses a frame whose transport
 * flags disagree with its position, in either direction. The one escape is
 * the unqualified smoke path (context absent AND the env gate set), which
 * runs the stage on its own buffers. */
static SparkStatus SparkQwen4FlashModuleValidateFrameContext(SparkQwen4FlashModuleState *state, const SparkQwen4FlashResidentDecodeStageFrameContext *context)
{
	uint32_t wants_input,wants_output,has_input,has_output;
	wants_input = state->stage_index != 0u ? 1u : 0u;
	wants_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
	if ( context == 0 )
		return((wants_input == 0u && wants_output == 0u) || state->allow_unqualified_execution != 0u
			? SPARK_STATUS_OK
			: SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->abi_version != SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ||
		context->descriptor_bytes != sizeof(*context) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	has_input = (context->flags & SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u ? 1u : 0u;
	has_output = (context->flags & SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u ? 1u : 0u;
	if ( has_input != wants_input || has_output != wants_output )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( has_input != 0u && (context->hidden_input_transport_session == 0 || context->hidden_input_post_receive_function == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( has_output != 0u && (context->hidden_output_transport_session == 0 || context->hidden_output_send_function == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

/* Land the previous stage's hidden residual into the slot's hidden buffer. */
static SparkStatus SparkQwen4FlashModuleConsumeHiddenInput(SparkQwen4FlashModuleSlot *slot, SparkQwen4FlashResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkHiddenTransportPacket *packet = &context->hidden_input_packet;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	SparkStatus status;
	memset(packet,0,sizeof(*packet));
	status = context->hidden_input_post_receive_function(context->hidden_input_transport_session,packet);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( packet->hidden_bf16 == 0 || packet->active_sequence_count < rows ||
		packet->hidden_dimension != SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH ||
		packet->bytes_per_sequence < SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	error = cudaMemcpyAsync(slot->hidden_bf16,packet->hidden_bf16,(uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"hidden_input"));
}

/* Hand the slot's final hidden residual to the next stage. */
static SparkStatus SparkQwen4FlashModuleEmitHiddenOutput(SparkQwen4FlashModuleSlot *slot, SparkQwen4FlashResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkHiddenTransportPacket *packet = &context->hidden_output_packet;
	memset(packet,0,sizeof(*packet));
	packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
	packet->flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 | SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
	packet->active_sequence_count = rows;
	packet->hidden_dimension = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH;
	packet->bytes_per_sequence = SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES;
	packet->hidden_bf16 = slot->hidden_bf16;
	packet->cuda_stream = slot->cuda_stream;
	return(context->hidden_output_send_function(context->hidden_output_transport_session,packet));
}

/* Embedding gather across the vocab-sharded packs: at tp>1 every rank
 * gathers the tokens inside its own vocab block (zeros elsewhere) and the
 * bf16 all-reduce completes the full [rows, H] embedding on every rank.
 * The stream vector is then seeded by REPLICATING the embedding into all
 * hc_count streams (the reference `hidden_states.repeat(1, 1, hc_count)`).
 * Under the standalone bypass the reduce no-ops and the result stays
 * rank-partial (validation semantics). */
static SparkStatus SparkQwen4FlashModuleEmbedRows(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, const uint32_t *token_ids, void *streams_bf16, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status;
	if ( state->tp_degree > 1u )
	{
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,SparkQwen4FlashLaunchEmbeddingGatherSharded(stream,token_ids,state->token_embedding_bf16,slot->normalized_bf16,rows,state->tp_vocab_base,state->tp_vocab_rows),"embedding_sharded");
		if ( status != SPARK_STATUS_OK )
			return(status);
		status = SparkQwen4FlashModuleTpAllReduceHidden(state,slot,slot->normalized_bf16,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	else
	{
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,SparkQwen4FlashLaunchEmbeddingGather(stream,token_ids,state->token_embedding_bf16,slot->normalized_bf16,rows),"embedding");
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,SparkQwen4FlashLaunchHcStreamReplicate(stream,slot->normalized_bf16,streams_bf16,rows),"embedding_replicate"));
}

/* MTP draft chain (ported from the proven qwen38_27b unit; the max
 * lineage never wired the pack's MTP tensors into execution). The TP>1
 * draft argmax runs the sharded screening + maxloc collective, same as the
 * head emit. */
/* MTP pack input (EAGLE convention, hc-adapted): the embed side is the
 * normed token embedding [H]; the hidden side is the mtp readout mixer over
 * the backbone's final stream vector, pre-normed by the [4H]
 * pre_fc_norm_hidden group norm (both sections at their true checkpoint
 * widths). fc concatenates them and the fc output seeds the MTP layer's
 * stream vector by replication. */
static SparkStatus SparkQwen4FlashModuleRunMtpPackInput(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, const uint32_t *token_src, const void *streams_src, uint32_t rows_p)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t half_bytes = SPARK_QWEN4_FLASH_MODEL_HIDDEN_BF16_BYTES,pack_pitch = 2u * half_bytes;
	cudaError_t error;
	SparkStatus status;
	/* Embed side: vocab-sharded gather into normalized (no replication),
	 * then the [H] embed norm. */
	if ( state->tp_degree > 1u )
	{
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,SparkQwen4FlashLaunchEmbeddingGatherSharded(stream,token_src,state->token_embedding_bf16,slot->gated_bf16,rows_p,state->tp_vocab_base,state->tp_vocab_rows),"mtp_embed_sharded");
		if ( status != SPARK_STATUS_OK )
			return(status);
		status = SparkQwen4FlashModuleTpAllReduceHidden(state,slot,slot->gated_bf16,rows_p);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	else
	{
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,SparkQwen4FlashLaunchEmbeddingGather(stream,token_src,state->token_embedding_bf16,slot->gated_bf16,rows_p),"mtp_embed");
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	error = SparkQwen4FlashLaunchRmsNorm(stream,slot->gated_bf16,state->mtp.embed_norm_weight_bf16,slot->normalized_bf16,rows_p,SPARK_QWEN4_FLASH_MODEL_HIDDEN_DIMENSION,SPARK_QWEN4_FLASH_MODEL_RMS_NORM_EPSILON);
	/* Hidden side: pre_fc_norm_hidden group norm, then the mtp readout
	 * mixer's own group norm and low-rank mean-mix. */
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchHcGroupNorm(stream,streams_src,state->mtp.hidden_norm_weight_bf16,slot->hc_up_bf16,rows_p,SPARK_QWEN4_FLASH_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchHcGroupNorm(stream,slot->hc_up_bf16,state->mtp.readout_mixer.hc_norm_weight_bf16,slot->hc_normed_bf16,rows_p,SPARK_QWEN4_FLASH_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&state->mtp.readout_mixer.mix_down,slot->hc_normed_bf16,slot->hc_lowrank_bf16,rows_p);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchHcSiluQuarter(stream,slot->hc_lowrank_bf16,rows_p);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&state->mtp.readout_mixer.mix_up,slot->hc_lowrank_bf16,slot->hc_up_bf16,rows_p);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchHcMix(stream,slot->hc_up_bf16,slot->hc_normed_bf16,slot->delta_bf16,rows_p);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync(slot->qkv_bf16,pack_pitch,slot->normalized_bf16,half_bytes,half_bytes,rows_p,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync((uint8_t *)slot->qkv_bf16 + half_bytes,pack_pitch,slot->delta_bf16,half_bytes,half_bytes,rows_p,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchLinear(stream,&state->mtp.fc,slot->qkv_bf16,slot->normalized_bf16,rows_p);
	if ( error == cudaSuccess )
		error = SparkQwen4FlashLaunchHcStreamReplicate(stream,slot->normalized_bf16,slot->hidden_bf16,rows_p);
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"mtp_pack"));
}

static SparkStatus SparkQwen4FlashModuleRunMtpDecoderPass(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, const SparkQwen4FlashKvBlockTableView *table, const SparkQwen4FlashAttnRowsView *rows_view, uint32_t rows_p)
{
	SparkStatus status;
	/* The MTP decoder layer is a full DecoderLayer: both sublayers run the
	 * same hc mean-mix/inject machinery as main layers, over the MTP layer's
	 * own mixer tensors. */
	status = SparkQwen4FlashModuleRunHcPrep(state,slot,&state->mtp.attention_hc,state->mtp.attention_hc.hc_norm_weight_bf16,rows_p);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleRunAttnLayer(state,slot,table,&state->mtp.attention,&state->mtp.indexer,state->mtp_cache_ordinal,state->mtp_cache_ordinal,rows_view,rows_p);
	if ( status == SPARK_STATUS_OK && state->tp_degree > 1u )
	{
		status = SparkQwen4FlashModuleTpAllReduceHidden(state,slot,slot->delta_bf16,rows_p);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleRunHcInject(state,slot,rows_p);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleRunHcPrep(state,slot,&state->mtp.mlp_hc,state->mtp.mlp_hc.hc_norm_weight_bf16,rows_p);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleRunMoe(state,slot,&state->mtp.moe,rows_p);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleRunHcInject(state,slot,rows_p);
	return(status);
}

static SparkStatus SparkQwen4FlashModuleRunMtpArgmaxRow(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, uint32_t row, uint32_t draft_index)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const void *row_streams = (const uint8_t *)slot->hidden_bf16 + ((uint64_t)row * SPARK_QWEN4_FLASH_MODEL_HC_STREAM_WIDTH * SPARK_QWEN4_FLASH_MODEL_BF16_ELEMENT_BYTES);
	cudaError_t error;
	/* Draft readout: the mtp readout mixer over the row's stream vector
	 * (same use_combine=False mean-mix as the backbone head). */
	if ( SparkQwen4FlashModuleRunHcReadout(state,slot,&state->mtp.readout_mixer,row_streams,slot->normalized_bf16,1u) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->tp_degree > 1u )
	{
		/* Sharded draft argmax: screen the rank's vocab shard, maxloc pack,
		 * cross-rank u64 max, unpack - the head emit flow at one row. */
		error = SparkQwen4FlashLaunchHeadScreenedArgmaxScore(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids,slot->head_candidate_counts,slot->mtp_draft_ids + draft_index,slot->head_scores_f32,state->tp_vocab_base,1u,state->tp_vocab_rows);
		if ( error == cudaSuccess )
			error = SparkQwen4FlashLaunchHeadMaxLocPack(stream,slot->head_scores_f32,slot->mtp_draft_ids + draft_index,slot->head_maxloc_u64,1u);
		if ( error == cudaSuccess && SparkQwen4FlashModuleTpReduceU64Max(state,slot,slot->head_maxloc_u64,1u) != SPARK_STATUS_OK )
			return(SPARK_STATUS_IO_ERROR);
		if ( error == cudaSuccess )
			error = SparkQwen4FlashLaunchHeadMaxLocUnpack(stream,slot->head_maxloc_u64,slot->mtp_draft_ids + draft_index,1u);
		return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"mtp_argmax_sharded"));
	}
	error = SparkQwen4FlashLaunchHeadScreenedArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids,slot->head_candidate_counts,slot->mtp_draft_ids + draft_index,1u,SPARK_QWEN4_FLASH_MODEL_OUTPUT_VOCAB_COUNT);
	return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"mtp_argmax"));
}

/* Seed pass over the committed row (token id + backbone hidden), then one
 * extension step per additional draft token: embed the previous draft
 * against the previous MTP hidden through the identical decoder pass. The
 * draft rows' KV slot/position/context are staged here because the decode
 * path only stages committed rows. */
static SparkStatus SparkQwen4FlashModuleRunMtpDraftChain(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, SparkQwen4FlashResidentDecodeStageFrameContext *context, SparkModelDriverFrame *frame, uint32_t rows)
{
	const SparkQwen4FlashMtpDraftView *view = context->mtp_draft;
	uint32_t step;
	uint32_t out_index = state->owns_embedding != 0u ? 1u : 0u;
	const uint32_t *seed_ids = slot->input_token_ids;
	const void *seed_hidden = slot->hidden_bf16;
	SparkQwen4FlashAttnRowsView rows_view;
	SparkStatus status;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	if ( view == 0 || view->draft_token_count == 0u || view->draft_token_count > SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY) != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( view->lane_index >= state->max_active_sequence_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	rows_view.slot_mapping = slot->slot_mapping;
	rows_view.row_positions = slot->row_positions;
	rows_view.row_lane_indices = slot->row_lane_indices;
	rows_view.context_lengths = slot->context_lengths;
	status = SparkQwen4FlashModuleRunMtpPackInput(state,slot,seed_ids,seed_hidden,1u);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleRunMtpDecoderPass(state,slot,context->kv_block_table,&rows_view,1u);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleRunMtpArgmaxRow(state,slot,0u,0u);
	for (step = 1u; status == SPARK_STATUS_OK && step < view->draft_token_count; step++)
	{
		uint64_t position = view->base_position + step - 1u;
		uint32_t lane_index = view->lane_index;
		uint32_t slot_mapping = (uint32_t)(position % SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
		uint32_t context_length = (uint32_t)(position % SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) + 1u;
		rows_view.slot_mapping = slot->slot_mapping + rows;
		rows_view.row_positions = slot->row_positions + rows;
		rows_view.row_lane_indices = slot->row_lane_indices + rows;
		rows_view.context_lengths = slot->context_lengths + rows;
		if ( cudaSuccess != cudaMemcpyAsync(slot->slot_mapping + rows,&slot_mapping,sizeof(slot_mapping),cudaMemcpyHostToDevice,stream)
			|| cudaSuccess != cudaMemcpyAsync(slot->row_positions + rows,&position,sizeof(position),cudaMemcpyHostToDevice,stream)
			|| cudaSuccess != cudaMemcpyAsync(slot->row_lane_indices + rows,&lane_index,sizeof(lane_index),cudaMemcpyHostToDevice,stream)
			|| cudaSuccess != cudaMemcpyAsync(slot->context_lengths + rows,&context_length,sizeof(context_length),cudaMemcpyHostToDevice,stream) )
			return(SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,cudaErrorInvalidValue,"mtp_stage_rows"));
		status = SparkQwen4FlashModuleRunMtpPackInput(state,slot,slot->mtp_draft_ids + (step - 1u),slot->hidden_bf16,1u);
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen4FlashModuleRunMtpDecoderPass(state,slot,context->kv_block_table,&rows_view,1u);
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen4FlashModuleRunMtpArgmaxRow(state,slot,0u,step);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,cudaMemcpyAsync((uint8_t *)frame->buffers[out_index].address + sizeof(uint32_t),slot->mtp_draft_ids,view->draft_token_count * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream),"mtp_emit");
	return(status);
}

static SparkStatus SparkQwen4FlashModuleRunDecode(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, SparkModelDriverFrame *frame, SparkQwen4FlashResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkQwen4FlashKvBlockTableView table;
	uint32_t block_indices[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t block_counts[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t layer,row;
	uint32_t wants_input,wants_output;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status;
	cudaError_t error;
	/* The KV table comes from the frame context when the serving adapter
	 * provides one; the synthesized identity table is the smoke path. */
	if ( context != 0 && context->kv_block_table != 0 )
		table = *context->kv_block_table;
	else
	{
		memset(&table,0,sizeof(table));
		for (row = 0; row < rows; row++)
		{
			block_indices[row] = row;
			block_counts[row] = 1u;
		}
		table.abi_version = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
		table.descriptor_bytes = sizeof(table);
		table.block_token_count = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		table.lane_count = state->max_active_sequence_count;
		table.lane_stride = 1u;
		table.lane_capacity = state->max_active_sequence_count;
		table.physical_block_indices = block_indices;
		table.lane_physical_block_counts = block_counts;
		table.host_physical_block_indices = block_indices;
		table.host_lane_physical_block_counts = block_counts;
	}
	/* The staged slot mappings are logical; resolve them through the
	 * frame's table before upload (see the resolver comment). */
	if ( state->attn_layer_count != 0u )
		for (row = 0u; row < rows; row++)
			SparkQwen4FlashModuleResolveSlotMapping(state,slot,&table,row,slot->host_row_positions[row]);
	status = SparkQwen4FlashModuleUploadRows(state,slot,frame,rows);
	if ( status == SPARK_STATUS_OK && state->tp_degree > 1u )
	{
		/* Zero the grouped expert output buffer so the rank-local pair
		 * reduce over ALL pairs sums only this rank's experts (the peers'
		 * rows live in their own buffers; the all-reduce completes the
		 * mixture). */
		error = cudaMemsetAsync(slot->moe_slot_out_bf16,0,(uint64_t)rows * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN * SPARK_QWEN4_FLASH_MODEL_HIDDEN_BF16_BYTES,stream);
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"tp_slot_zero");
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen4FlashModuleKvPrepareFrame(state,slot,context,&table,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	wants_input = context != 0 && (context->flags & SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u ? 1u : 0u;
	if ( state->owns_embedding != 0u )
	{
		status = SparkQwen4FlashModuleEmbedRows(state,slot,slot->input_token_ids,slot->hidden_bf16,rows);
	}
	else if ( wants_input != 0u )
		status = SparkQwen4FlashModuleConsumeHiddenInput(slot,context,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (layer = state->first_layer_index; status == SPARK_STATUS_OK && layer < state->first_layer_index + state->layer_count; layer++)
		status = SparkQwen4FlashModuleRunLayer(state,slot,&table,layer,rows,(const uint32_t *)frame->buffers[0].address);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->owns_final_head != 0u )
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,SparkQwen4FlashModuleEmitHead(state,slot,frame,rows),"head_emit");
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u
		&& (context->flags & SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER) != 0u )
		status = SparkQwen4FlashModuleRunMtpDraftChain(state,slot,context,frame,rows);
	wants_output = context != 0 && (context->flags & SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u ? 1u : 0u;
	if ( status == SPARK_STATUS_OK && wants_output != 0u )
		status = SparkQwen4FlashModuleEmitHiddenOutput(slot,context,rows);
	/* Frame error check (frame_error.cuh): copy the record back on the
	 * stream so it lands after every launch, and fail the frame if any
	 * kernel reported corruption. The context lives; the frame does not. */
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,cudaMemcpyAsync(slot->host_frame_error,slot->frame_error,SPARK_FRAME_ERROR_WORDS * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream),"frame_error_copyback");
	error = cudaStreamSynchronize(stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"stream_sync");
	if ( status == SPARK_STATUS_OK && slot->host_frame_error[0] != 0u )
	{
		fprintf(stderr,"qwen4-flash frame %u: kernel frame-error code %u kind %u row %u seq %u pos %u page %u\n",
			(unsigned)state->debug_frame_serial,(unsigned)slot->host_frame_error[0],
			(unsigned)slot->host_frame_error[1],(unsigned)slot->host_frame_error[2],
			(unsigned)slot->host_frame_error[3],(unsigned)slot->host_frame_error[4],
			(unsigned)slot->host_frame_error[5]);
		status = SPARK_STATUS_INTERNAL_ERROR;
	}
	if ( status == SPARK_STATUS_OK )
		SparkQwen4FlashModuleKvMarkWritten(state,slot,rows);
	return(status);
}

/* Prefill (max-lineage gap, lane port): walk one lane's tokens through the
 * stack sequentially - token t runs the same per-row machinery a decode row
 * runs (same kernels, same recurrence order, warm GDN state after the first
 * token), which makes the walk bit-equivalent to the chunked prefill by
 * construction; only the final token emits a head argmax or hidden output.
 * Slower than a batched chunk walk, and correctness-first is this lane's
 * mandate - the validator's decode-vs-prefill gate holds it to bit parity. */
static SparkStatus SparkQwen4FlashModuleRunPrefill(SparkQwen4FlashModuleState *state, SparkQwen4FlashModuleSlot *slot, SparkModelDriverFrame *frame, SparkQwen4FlashResidentDecodeStageFrameContext *context, const SparkQwen4FlashPrefillFrameView *prefill)
{
	SparkQwen4FlashKvBlockTableView table;
	uint32_t block_indices[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t block_counts[SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkModelDriverBuffer row_buffers[2];
	SparkModelDriverFrame row_frame;
	uint32_t token_index,layer;
	uint32_t wants_input,wants_output;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status = SPARK_STATUS_OK;
	cudaError_t error;
	if ( context != 0 && context->kv_block_table != 0 )
		table = *context->kv_block_table;
	else
	{
		memset(&table,0,sizeof(table));
		block_indices[0] = prefill->lane_index;
		block_counts[0] = 1u;
		table.abi_version = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
		table.descriptor_bytes = sizeof(table);
		table.block_token_count = SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		table.lane_count = state->max_active_sequence_count;
		table.lane_stride = 1u;
		table.lane_capacity = state->max_active_sequence_count;
		table.physical_block_indices = block_indices;
		table.lane_physical_block_counts = block_counts;
		table.host_physical_block_indices = block_indices;
		table.host_lane_physical_block_counts = block_counts;
	}
	row_frame = *frame;
	memcpy(row_buffers,frame->buffers,sizeof(row_buffers));
	row_frame.buffers = row_buffers;
	row_frame.active_slot_count = 1u;
	row_frame.new_token_count = 1u;
	wants_input = (context->flags & SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u ? 1u : 0u;
	wants_output = (context->flags & SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u ? 1u : 0u;
	for (token_index = 0u; status == SPARK_STATUS_OK && token_index < prefill->token_count; token_index++)
	{
		uint64_t position = prefill->base_position + token_index;
		slot->host_row_lane_indices[0] = prefill->lane_index;
		slot->host_row_positions[0] = position;
		/* Cold ONLY at the sequence's first token: a continuation frame
		 * (base_position > 0) must keep the carried GDN state. Keying cold
		 * on the frame-local token index reset the state on every 1-token
		 * continuation prefill - the decode-vs-prefill whole-stack gate
		 * exposed it as a layer-0 divergence (36 warm GDN layers vs a
		 * cold restart). */
		slot->host_row_cold[0] = prefill->base_position + token_index == 0u ? 1u : 0u;
		slot->host_slot_mapping[0] = (uint32_t)(position % SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
		slot->host_context_lengths[0] = (uint32_t)(position % SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) + 1u;
		row_frame.sequence_position = position;
		row_frame.sequence_id = prefill->sequence_id;
		if ( state->owns_embedding != 0u )
			row_buffers[0].address = ((uint8_t *)frame->buffers[0].address) + ((uint64_t)token_index * sizeof(uint32_t));
		SparkQwen4FlashModuleResolveSlotMapping(state,slot,&table,0u,position);
		status = SparkQwen4FlashModuleUploadRows(state,slot,&row_frame,1u);
		if ( status == SPARK_STATUS_OK && state->tp_degree > 1u )
		{
			error = cudaMemsetAsync(slot->moe_slot_out_bf16,0,(uint64_t)1u * SPARK_QWEN4_FLASH_MODEL_EXPERTS_PER_TOKEN * SPARK_QWEN4_FLASH_MODEL_HIDDEN_BF16_BYTES,stream);
			status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"tp_slot_zero");
		}
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen4FlashModuleKvPrepareFrame(state,slot,context,&table,1u);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( state->owns_embedding != 0u )
			status = SparkQwen4FlashModuleEmbedRows(state,slot,slot->input_token_ids,slot->hidden_bf16,1u);
		else if ( wants_input != 0u )
			status = SparkQwen4FlashModuleConsumeHiddenInput(slot,context,1u);
		for (layer = state->first_layer_index; status == SPARK_STATUS_OK && layer < state->first_layer_index + state->layer_count; layer++)
			status = SparkQwen4FlashModuleRunLayer(state,slot,&table,layer,1u,(const uint32_t *)row_buffers[0].address);
		state->debug_token_serial++;
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->owns_final_head != 0u )
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,SparkQwen4FlashModuleEmitHead(state,slot,frame,1u),"head_emit");
	else if ( wants_output != 0u )
		status = SparkQwen4FlashModuleEmitHiddenOutput(slot,context,1u);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,cudaMemcpyAsync(slot->host_frame_error,slot->frame_error,SPARK_FRAME_ERROR_WORDS * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream),"frame_error_copyback");
	error = cudaStreamSynchronize(stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN4_FLASH_MODULE_TAG,error,"stream_sync");
	if ( status == SPARK_STATUS_OK && slot->host_frame_error[0] != 0u )
	{
		fprintf(stderr,"qwen4-flash prefill: kernel frame-error code %u kind %u row %u seq %u pos %u page %u\n",
			(unsigned)slot->host_frame_error[0],(unsigned)slot->host_frame_error[1],
			(unsigned)slot->host_frame_error[2],(unsigned)slot->host_frame_error[3],
			(unsigned)slot->host_frame_error[4],(unsigned)slot->host_frame_error[5]);
		status = SPARK_STATUS_INTERNAL_ERROR;
	}
	if ( status == SPARK_STATUS_OK )
		SparkQwen4FlashModuleKvMarkWritten(state,slot,1u);
	return(status);
}

SparkStatus SparkQwen4FlashResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame)
{
	SparkQwen4FlashModuleState *state = (SparkQwen4FlashModuleState *)module_state;
	SparkQwen4FlashResidentDecodeStageFrameContext *context;
	SparkQwen4FlashModuleSlot *slot;
	uint32_t rows,row;
	SparkStatus status;
	if ( state == 0 || frame == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->debug_dump_hidden = SparkQwen4FlashModuleDebugDumpHidden;
	state->debug_frame_serial++;
	state->debug_token_serial = 0u;
	context = (SparkQwen4FlashResidentDecodeStageFrameContext *)frame->user_context;
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u )
	{
		const SparkQwen4FlashPrefillFrameView *prefill;
		SparkStatus prefill_status;
		prefill = context != 0 ? context->prefill_frame : 0;
		if ( prefill == 0 || prefill->token_count == 0u || prefill->token_count > state->max_active_sequence_count || prefill->lane_index >= state->max_active_sequence_count )
		{
			atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
			return(SPARK_STATUS_INVALID_ARGUMENT);
		}
		slot = &state->slots[0];
		if ( slot->cuda_stream == 0 )
			return(SPARK_STATUS_INTERNAL_ERROR);
		atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
		prefill_status = SparkQwen4FlashModuleRunPrefill(state,slot,frame,context,prefill);
		if ( prefill_status == SPARK_STATUS_OK )
		{
			atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
			atomic_fetch_add_explicit(&state->tokens_emitted,prefill->token_count,memory_order_relaxed);
		}
		else
			atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
		return(prefill_status);
	}
	status = SparkQwen4FlashModuleValidateFrameContext(state,context);
	if ( status != SPARK_STATUS_OK )
		return(status);
	rows = frame->active_slot_count;
	if ( rows == 0u || rows > state->max_active_sequence_count )
	{
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	slot = &state->slots[0];
	if ( slot->cuda_stream == 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	for (row = 0; row < rows; row++)
	{
		slot->host_row_lane_indices[row] = row;
		slot->host_row_positions[row] = frame->sequence_position;
		slot->host_row_cold[row] = 0u;
		slot->host_slot_mapping[row] = (frame->sequence_position + row) % SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		slot->host_context_lengths[row] = (uint32_t)((frame->sequence_position + row) % SPARK_QWEN4_FLASH_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) + 1u;
	}
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	status = SparkQwen4FlashModuleRunDecode(state,slot,frame,context,rows);
	if ( status == SPARK_STATUS_OK )
	{
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
		atomic_fetch_add_explicit(&state->tokens_emitted,rows,memory_order_relaxed);
	}
	else
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	return(status);
}
