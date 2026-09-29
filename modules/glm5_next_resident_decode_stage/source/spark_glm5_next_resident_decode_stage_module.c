#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <stdatomic.h>
#include "sparkpipe/spark_error_site.h"
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>

#include <cuda_runtime.h>
#include "sparkpipe/spark_tp_chain_ordinal.h"

#include "sparkpipe/spark_glm5_next_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_speculation_lookup_draft.h"
#include "sparkpipe/spark_speculation_reference_draft.h"
#include "sparkpipe/spark_speculation_recorded_draft.h"
#include "sparkpipe/spark_speculation_depth.h"
#include "sparkpipe/spark_speculation_drafter_mix.h"
#include "sparkpipe/spark_speculation_relay_link.h"
#include "sparkpipe/spark_speculation_tap.h"
#define SPARK_FAMILY_CAMEL Glm5Next
#define SPARK_FAMILY_UPPER GLM5_NEXT
#define SPARK_FAMILY_LOWER glm5_next

#include "sparkpipe/family/spark_family.h"

_Static_assert((uint64_t)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION *
    SPARK_GLM5_NEXT_MODEL_HC_MULT * 2u <=
    SPARK_WEIGHTD_MESH_ROW_BYTES_MAX,
    "widest model row must fit the mesh row law");

#define SPARK_GLM5_NEXT_PROBE_CONNECT_TIMEOUT_SCALE 4u
#define SPARK_GLM5_NEXT_PROBE_OPERATION_TIMEOUT_SCALE 8u
static int SparkGlm5NextProbeEnabled(void)
{
	static int probe_enabled = -1;
	if ( probe_enabled < 0 )
		probe_enabled = getenv("SPARK_GLM5_NEXT_PROBE") != 0 ? 1 : 0;
	return(probe_enabled);
}
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_kv_model_table.h"
#include "sparkpipe/spark_glm5_next_kv_geometry.h"
#include "sparkpipe/spark_glm5_next_index_cp.h"
#include "sparkpipe/spark_glm5_next_kv_shard.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_row_layout.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd_map.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_lease.h"
#include "sparkpipe/spark_head_screen.h"
#include "sparkpipe/spark_speculation_policy.h"
#include "sparkpipe/spark_latency_histogram.h"
#include "sparkpipe/spark_step_verdict.h"
#include "sparkpipe/spark_expert_working_set.h"
#include "sparkpipe/spark_state_span.h"
#include "sparkpipe/spark_sha256.h"
#include "spark_glm5_next_resident_decode_stage_internal.h"
#include "spark_glm5_next_stagepack_format.h"
#ifdef SPARK_SCORE_DUMP
#include "sparkpipe/spark_score_dump.h"
#endif

#ifndef GLM5_NEXT_EXPERT_WEIGHT_CODEC
#error "GLM5_NEXT_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
#endif
#ifndef GLM5_NEXT_CONTRACT_SHA256
#error "GLM5_NEXT_CONTRACT_SHA256 must identify the exact model package contract"
#endif

#define SPARK_GLM5_NEXT_MODULE_TAG "glm5_next_stage"
#define SPARK_GLM5_NEXT_STAGEPACK_MAX_TENSOR_COUNT 2048u
#define SPARK_GLM5_NEXT_NO_INDEX_ORDINAL UINT32_MAX
#define SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT 6u
#define SPARK_GLM5_NEXT_LAZY_ATTACH_ATTEMPTS 600u
#define SPARK_GLM5_NEXT_LAZY_ATTACH_PAUSE_NS 1000000000ull
#define SPARK_GLM5_NEXT_CHAIN_TOKEN_CAPACITY (SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_MODEL_DRIVER_MAX_TOKENS_PER_SEQUENCE)

_Static_assert(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT <= SPARK_WEIGHTD_WORK_QUEUE_CAPACITY,"completion worker must hold one job per occupied slot");

#if SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES != \
	( SPARK_GLM5_NEXT_KV_ARENA_KV_HEAD_COUNT * \
	  SPARK_GLM5_NEXT_KV_ARENA_HEAD_DIM * \
	  SPARK_GLM5_NEXT_KV_BYTES_PER_SCALAR )
#error "glm5_next KV slot bytes and arena block geometry disagree"
#endif

typedef struct SparkGlm5NextPackRange
{
	uint64_t offset;
	uint64_t bytes;
} SparkGlm5NextPackRange;

typedef struct SparkGlm5NextModuleState SparkGlm5NextModuleState;

static uint64_t SparkGlm5NextNowNs(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0u);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

typedef struct SparkGlm5NextCompletionOverflow
{
	SparkWeightdWorkFunction function;
	void *context;
} SparkGlm5NextCompletionOverflow;

#define SPARK_GLM5_NEXT_OVERFLOW_POOL 	(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT * 2u)

static void SparkGlm5NextDrainParkedCompletions(
	SparkGlm5NextModuleState *state);
static int SparkGlm5NextBoundedStreamSync(SparkGlm5NextModuleState *state,void *stream,uint64_t timeout_ns);
static void SparkGlm5NextScheduleCompletionWork(SparkGlm5NextModuleState *state,SparkWeightdWorkFunction function,void *context);

typedef struct SparkGlm5NextWaveTiming
{
	SparkLatencyHistogram interval[SPARK_GLM5_NEXT_WAVE_INTERVALS];
	uint64_t window_ns;
	uint64_t delivered_ns;
	uint64_t waves;
	uint64_t rows;
	uint64_t steps;
	uint64_t prefill;
	uint64_t graph;
	uint64_t linear;
	uint64_t retries;
	uint64_t busy[SPARK_GLM5_NEXT_BUSY_REASONS];
	uint64_t captures;
	uint64_t capture_ns;
	uint64_t graph_run_ns;
	uint64_t eager_run_ns;
	uint64_t linear_run_ns;
	uint64_t linear_walk_ns;
	uint64_t decode_wait_ns;
	uint64_t source_wait_ns;
	uint64_t peer_wait_ns;
	uint64_t copy_ns;
	uint64_t combine_ns;
	uint64_t worst_ns;
	uint64_t worst_request;
	uint64_t worst_epoch[2];
	uint64_t worst_part_ns[SPARK_GLM5_NEXT_WAVE_INTERVALS];
	uint32_t graph_path;
} SparkGlm5NextWaveTiming;

typedef struct SparkGlm5NextAsyncCompletion
{
	SparkGlm5NextModuleState *state;
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t slot_index;
	uint32_t lane_count;
	uint32_t row_count;
	uint32_t lane_indices[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint8_t lane_bound[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t *output_token_destination;
	SparkGlm5NextStateCapture *state_capture;
	uint32_t finish_retries;
	uint32_t burst_token_count;
	uint64_t cache_extra_tokens;
	uint64_t chain_start_ns;
	uint64_t attempt_ns;
	uint64_t keyed_ns;
	uint64_t launch_ns;
	uint64_t finish_ns;
	uint64_t capture_ns;
	uint64_t walk_ns;
	uint64_t epoch[2];
	uint32_t retries;
	uint32_t busy[SPARK_GLM5_NEXT_BUSY_REASONS];
	uint32_t captures;
	uint32_t prefill;
	uint32_t graph;
	uint32_t linear;
	uint32_t steps;
	uint32_t graph_path;
	uint32_t mtp_draft_tokens[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH];
	SparkModelDriverCompletion completion;
} SparkGlm5NextAsyncCompletion;

struct SparkGlm5NextModuleState
{
	SparkStageModuleLedger ledger;
	SparkWeightdWorker *completion_worker;
	SparkWeightdLazyPack *lazy_pack;
	_Atomic(void *) lazy_retained[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t expert_weight_codec;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t tp_collective_disabled;
	uint32_t resident_sequence_capacity;
	uint32_t pipeline_slot_count;
	uint32_t max_sequence_positions;
	uint32_t execution_row_capacity;
	uint32_t decode_split_context_threshold;
	uint32_t pages_per_sequence;
	uint32_t page_count;
	uint32_t physical_page_count;
	uint32_t index_layer_count;
	uint32_t multiprocessor_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	void *execution_stream;
	SparkStageModuleCudaWait stream_wait;
	char model_revision[SPARK_GLM5_NEXT_STAGEPACK_MODEL_REVISION_BYTES];
	SparkGlm5NextLayerWeights layers[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t index_ordinal_by_local_layer[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t kv_ordinal_by_local_layer[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t kda_ordinal_by_local_layer[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t kv_layer_count;
	uint32_t kda_layer_count;
	uint8_t *kda_state_pools;
	uint64_t kda_state_layer_stride_bytes;
	uint8_t *kda_window_pools;
	uint8_t *kda_q_window_pool;
	uint8_t *kda_k_window_pool;
	uint8_t *kda_v_window_pool;
	uint64_t kda_window_layer_stride_bytes;
	uint32_t *kda_state_index_device;
	uint32_t *kda_state_index_host;
	uint64_t layer_seen[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint64_t global_seen;
	uint64_t mtp_seen;
	uint32_t pack_has_mtp;
	SparkGlm5NextLayerWeights mtp_layer;
	const void *mtp_eh_proj_bf16;
	const void *mtp_enorm_bf16;
	const void *mtp_hnorm_bf16;
	const void *mtp_shared_norm_bf16;
	uint32_t mtp_enabled;
	uint32_t index_cp;
	uint32_t kv_shard;
	char kv_layout_fingerprint[96];
	uint16_t *mtp_lane_hidden_bf16;
	uint8_t *mtp_lane_armed;
	uint64_t kda_replay_layer_bytes;
	const void *embedding_bf16;
	const void *final_norm_bf16;
	const void *lm_head_bf16;
	uint8_t *head_certified_fp8_payload;
	float *head_certified_fp8_scale_f32;
	float *head_certified_fp8_norm_f32;
	uint64_t expert_pin_leases[32];
	uint8_t expert_pin_phases[32];
	uint32_t expert_pin_lease_count;
	uint32_t expert_pin_key_count;
	SparkExpertWorkingSet expert_ws;
	uint32_t ws_enabled;
	uint64_t ws_local_miss;
	uint64_t ws_remote_miss;
	uint64_t ws_replays;
	uint64_t ws_eager_steps;
	uint8_t *kv_cache;
	uint64_t kv_layer_stride_bytes;
	uint8_t *index_cache;
	uint64_t index_layer_stride_bytes;
	uint32_t *page_table;
	uint32_t *page_table_shadow;
	SparkKvCacheArena kv_arena;
	SparkKvPageCache kv_page_cache;
	SparkKvPageStore kv_page_store;
	SparkKvPageStore recurrent_store;
	uint8_t *recurrent_staging;
	uint64_t recurrent_page_bytes;
	SparkKvCacheBlock *kv_blocks;
	uint32_t *kv_resident_slot_logical_block_indices;
	SparkKvPageCacheEntry *kv_entries;
	SparkKvPageCacheSequence *kv_sequences;
	uint32_t *kv_hash_bucket_heads;
	uint32_t *kv_entry_indices_by_logical_page;
	uint8_t *kv_page_staging;
	uint32_t *kv_lane_logical_pages;
	uint32_t *kv_lane_physical_pages;
	SparkKvLaneTransaction *kv_lane_transactions;
	SparkKvLaneTransactions kv_transactions;
	pthread_mutex_t kv_mutex;
	uint32_t kv_mutex_initialized;
	uint64_t control_generation;
	uint64_t reset_generation;
	const char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
	char kv_backing_default[256];
	SparkGlm5NextExecutionSlot slots[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkGlm5NextAsyncCompletion completions[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	volatile uint64_t slot_alive_ns[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkGlm5NextCompletionOverflow overflow_pool[SPARK_GLM5_NEXT_OVERFLOW_POOL];
	uint32_t overflow_parked_count;
	pthread_mutex_t completion_queue_lock;
	uint32_t completion_queue_lock_initialized;
	atomic_uint slot_states[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint lane_states[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_uchar lane_bound[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong lane_sequence_ids[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong lane_next_positions[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong host_callback_completion_count;
	SparkTpDeviceCollective tp_device_collective;
	SparkTpDeviceCollective tp_device_collective_hc;
	uint32_t tp_device_collective_hc_initialized;
	uint32_t tp_device_collective_initialized;
	_Atomic(uint32_t) tp_chain_active;
	uint32_t tp_lane;
	SparkWeightdClient *lane_client;
	atomic_uint terminal_status;
	SparkTpDeviceCollectiveCreditBinding tp_credit_bindings[SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT];
	uint32_t tp_credit_binding_count;
	void *tp_credit_send_bf16;
	void *tp_credit_receive_bf16;
	void *tp_host_credit_send_bf16;
	void *tp_host_credit_receive_bf16;
	SparkTpDeviceCollectiveCreditBinding tp_hc_credit_bindings[SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT];
	uint32_t tp_hc_credit_binding_count;
	void *tp_hc_credit_send_bf16;
	void *tp_hc_credit_receive_bf16;
	void *tp_hc_host_credit_send_bf16;
	void *tp_hc_host_credit_receive_bf16;
	atomic_ullong nccl_next_ordinal;
	uint32_t graph_path_enabled;
	uint32_t graph_arrival_dumped;
	uint32_t experts_warm;
	uint64_t degrade_graph_fallback;
	uint64_t degrade_covered_abandon;
	uint64_t degrade_graph_disabled;
	uint64_t degrade_graph_stuck;
	uint32_t graph_record_limit;
	uint32_t graph_gate_printed;
	uint32_t graph_record_ops;
	uint32_t graph_record_stop;
	uint64_t chain_stage_ns[8u];
	uint64_t chain_profile_last_ns;
	SparkGlm5NextWaveTiming wave_timing;
	atomic_uint_fast64_t kda_restore[SPARK_GLM5_NEXT_KDA_FIELDS];
	atomic_uint_fast64_t kda_capture[SPARK_GLM5_NEXT_KDA_FIELDS];
	atomic_uint_fast64_t kda_window_ns;
	uint64_t wave_attempt_request;
	uint64_t wave_attempt_ns;
	uint32_t wave_attempt_retries;
	uint32_t wave_attempt_busy[SPARK_GLM5_NEXT_BUSY_REASONS];
	uint32_t graph_path_requested;
	uint32_t l2_prefetch;
	SparkGlm5NextL2PrefetchShape l2_prefetch_shape;
	uint64_t l2_prefetch_rounds;
	uint32_t verify_rows_max;
	uint32_t verify_drafter;
	const char *verify_drafter_path;
	SparkSpeculationDraftFunction verify_draft_function;
	void *verify_draft_context;
	SparkSpeculationLookupDraft verify_lookup;
	SparkSpeculationReferenceDraft verify_reference;
	uint32_t *verify_reference_tokens;
	SparkSpeculationRecordedDraft verify_recorded;
	uint8_t *verify_recorded_bytes;
	uint64_t verify_frames;
	uint64_t verify_plain_frames;
	uint64_t verify_frame_class[SPARK_GLM5_NEXT_VERIFY_FRAME_CLASS_COUNT];
	uint64_t verify_rounds;
	uint64_t verify_proposed;
	uint64_t verify_accepted;
	uint64_t verify_tokens;
	uint64_t verify_plain_steps;
	uint64_t verify_accept_depth[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX];
	uint64_t verify_position_reached[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX - 1u];
	uint64_t verify_position_accepted[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX - 1u];
	uint32_t *verify_depth_cap;
	uint64_t *verify_depth_sequence;
	SparkSpeculationTapSet tap_set;
	uint32_t tap_enabled;
	uint32_t tap_rank;
	uint32_t tap_dump_open;
	uint32_t tap_lock_ready;
	uint32_t tap_relay_depth;
	uint32_t tap_dump_warned;
	uint64_t tap_generation;
	uint64_t tap_fingerprint;
	uint64_t tap_serial;
	uint64_t tap_records;
	uint64_t tap_emits;
	uint64_t tap_dump_failed;
	uint8_t *tap_scratch;
	pthread_mutex_t tap_lock;
	SparkSpeculationTapDump tap_dump;
	SparkSpeculationRelayLink tap_link;
	SparkSpeculationRelayRemote *tap_remote;
	uint32_t verify_mtp;
	uint32_t mtp_sidecar;
	uint64_t mtp_sidecar_bytes;
	uint64_t *mtp_lane_sequence;
	uint64_t *mtp_lane_next;
	struct SparkGlm5NextTpChain *mtp_chain;
	uint32_t mtp_anchor_token;
	uint64_t mtp_drafts;
	uint64_t mtp_draft_tokens;
	uint64_t mtp_draft_ns;
	uint64_t mtp_cold;
	uint64_t mtp_truncated;
	uint64_t mtp_taps;
	SparkSpeculationDrafterMix verify_mix;
	uint32_t chain_profile_stage;
	uint32_t rs_taken;
	uint32_t rs_hit;
	uint32_t hbound_probes;
	uint64_t step_verdicts[SPARK_STEP_VERDICT_COUNT];
	FILE *route_trace;
	const uint8_t *decode_lease_base_saved;
	atomic_ullong nccl_next_ordinal_hc;
#ifdef SPARK_SCORE_DUMP
	struct SparkGlm5NextScore *score;
#endif
};

static SparkStatus SparkGlm5NextModuleConfigure(
	SparkGlm5NextModuleState *state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	const char **pack_path)
{
	const SparkGlm5NextResidentDecodeStageNodeContext *context;
	if ( state == 0 || configuration == 0 || host_services == 0 || pack_path == 0 || host_services->node_context == 0 || host_services->execution_stream == 0 || host_services->kv_logical_page_capacity == 0u || host_services->kv_physical_page_capacity == 0u || host_services->kv_physical_page_capacity > host_services->kv_logical_page_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkGlm5NextResidentDecodeStageNodeContext *)host_services->node_context;
	if ( context->abi_version != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION || context->descriptor_bytes != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( SparkGlm5NextResidentDecodeStageSpanIsValid(context->stage_count,context->stage_index,context->first_layer_index,context->layer_count) == 0u || context->layer_count > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE || context->expert_weight_codec != GLM5_NEXT_EXPERT_WEIGHT_CODEC || context->resident_sequence_capacity == 0u || context->resident_sequence_capacity > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || context->pipeline_slot_count == 0u || context->pipeline_slot_count > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT || context->max_sequence_positions == 0u || context->max_sequence_positions > SPARK_GLM5_NEXT_MODEL_MAXIMUM_CONTEXT_TOKENS || context->execution_row_capacity == 0u || context->execution_row_capacity > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT || context->execution_row_capacity > SPARK_WEIGHTD_MESH_MAX_BATCH_ROWS || context->decode_split_context_threshold > context->max_sequence_positions || context->tp_degree == 0u || context->tp_rank >= context->tp_degree || (context->flags & ~SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KNOWN_FLAGS) != 0u || context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' || context->model_revision == 0 || context->model_revision[0] == '\0' || strlen(context->model_revision) >= sizeof(state->model_revision) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkWeightCodecIsKnown(context->expert_weight_codec) == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( (context->flags & SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_MTP) != 0u && (context->stage_index + 1u) != context->stage_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->tp_degree != 1u && (SPARK_GLM5_NEXT_MODEL_HEAD_COUNT % context->tp_degree != 0u || SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT % context->tp_degree != 0u || SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u || SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->model_revision == 0 || strcmp(configuration->model_revision,context->model_revision) != 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->stage_count = context->stage_count;
	state->stage_index = context->stage_index;
	state->first_layer_index = context->first_layer_index;
	state->layer_count = context->layer_count;
	state->expert_weight_codec = context->expert_weight_codec;
	state->tp_degree = context->tp_degree;
	state->tp_rank = context->tp_rank;
	state->kv_backing_directory = context->kv_backing_directory;
	state->kv_backing_maximum_bytes = context->kv_backing_maximum_bytes;
	state->tp_collective_disabled = context->tp_collective_identifier == 0u ? 1u : 0u;
	state->resident_sequence_capacity = context->resident_sequence_capacity;
	state->pipeline_slot_count = context->pipeline_slot_count;
	state->max_sequence_positions = context->max_sequence_positions;
	state->decode_split_context_threshold = context->decode_split_context_threshold;
	state->execution_row_capacity = context->execution_row_capacity;
	state->page_count = host_services->kv_logical_page_capacity;
	state->physical_page_count = host_services->kv_physical_page_capacity;
	state->owns_embedding = context->stage_index == 0u ? 1u : 0u;
	state->owns_final_head = context->stage_index + 1u == context->stage_count ? 1u : 0u;
	state->mtp_enabled = (context->flags & SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_MTP) != 0u ? 1u : 0u;
	state->index_cp = (context->flags & SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_INDEX_CP) != 0u ? 1u : 0u;
	if ( state->index_cp != 0u && (context->tp_degree < 2u || SparkGlm5NextIndexCpFits(context->max_sequence_positions,context->tp_degree,context->execution_row_capacity) == 0u) )
	{
		fprintf(stderr,"GLM index context parallel needs tp_degree >= 2 and max_sequence_positions whose local pool scores fit the gather (tp=%u positions=%u rows=%u)\n",context->tp_degree,context->max_sequence_positions,context->execution_row_capacity);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->kv_shard = (context->flags & SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_KV_SHARD) != 0u ? 1u : 0u;
	if ( state->kv_shard == 0u && context->tp_degree >= SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_KV_SHARD_REQUIRED_DEGREE )
	{
		fprintf(stderr,"GLM-KV-SHARD-REQUIRED tp=%u: at this degree each rank must hold 1/tp of the latent KV and indexer keys; set kv_shard (and dsa_index_context_parallel)\n",context->tp_degree);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( state->kv_shard != 0u && state->tp_collective_disabled != 0u )
	{
		fprintf(stderr,"GLM-KV-SHARD-REFUSED tp=%u: the sharded KV exchange needs the TP collective, and tp_collective_identifier is 0\n",context->tp_degree);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( state->kv_shard != 0u && (state->index_cp == 0u || SparkGlm5NextKvShardFits(context->tp_degree,context->execution_row_capacity) == 0u) )
	{
		fprintf(stderr,"GLM-KV-SHARD-REFUSED tp=%u rows=%u index_cp=%u: sharding needs index context parallel, a degree that divides the 64 heads and the 64-slot page at grain %u, and exchange payloads that fit the collectives\n",context->tp_degree,context->execution_row_capacity,state->index_cp,SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->execution_stream = host_services->execution_stream;
	(void)snprintf(state->model_revision,sizeof(state->model_revision),"%s",context->model_revision);
	*pack_path = context->stage_pack_path;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPackValidateHeader(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackHeader *header,
	uint64_t file_bytes)
{
	uint64_t directory_bytes,directory_end;
	if ( state == 0 || header == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( header->magic != SPARK_GLM5_NEXT_STAGEPACK_MAGIC || header->format_version != SPARK_GLM5_NEXT_STAGEPACK_FORMAT_VERSION || header->header_bytes != SPARK_GLM5_NEXT_STAGEPACK_HEADER_BYTES || header->directory_entry_bytes != SPARK_GLM5_NEXT_STAGEPACK_ENTRY_BYTES || header->codec_abi_version != SPARK_WEIGHT_CODEC_ABI_VERSION )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( (header->flags & ~SPARK_GLM5_NEXT_STAGEPACK_KNOWN_FLAGS) != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( SparkGlm5NextStagePackHeaderTpDegree(header) == 0u || SparkGlm5NextStagePackHeaderTpDegree(header) != state->tp_degree || SparkGlm5NextStagePackHeaderTpRank(header) != state->tp_rank )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->tensor_count == 0u || header->tensor_count > SPARK_GLM5_NEXT_STAGEPACK_MAX_TENSOR_COUNT || header->stage_count != state->stage_count || header->stage_index != state->stage_index || header->first_layer_index != state->first_layer_index || header->layer_count != state->layer_count || header->total_layer_count != SPARK_GLM5_NEXT_MODEL_LAYER_COUNT || header->hidden_dimension != SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION || header->vocab_count != SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT || header->routed_expert_count != SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 || header->expert_weight_codec != state->expert_weight_codec || header->kv_cache_codec != SPARK_WEIGHT_CODEC_BF16 )
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	if ( header->file_bytes != file_bytes || header->directory_offset < header->header_bytes || header->directory_offset % SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES != 0u || header->tensor_count > UINT64_MAX / header->directory_entry_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	directory_bytes = (uint64_t)header->tensor_count * header->directory_entry_bytes;
	directory_end = header->directory_offset + directory_bytes;
	if ( directory_end < header->directory_offset || directory_end > file_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPackValidateEntryGeometry(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackHeader *header,
	const SparkGlm5NextStagePackEntry *entry,
	SparkGlm5NextStagePackTensorShape *shape)
{
	uint64_t payload_bytes,scale_bytes,directory_end;
	uint32_t local_layer;
	if ( SparkGlm5NextStagePackExpectedShape(entry->tensor_kind,entry->layer_index,state->expert_weight_codec,state->tp_degree,shape) < 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->layer_index != SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER )
	{
		if ( entry->layer_index == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
		{
			if ( (state->mtp_seen & (UINT64_C(1) << entry->tensor_kind)) != 0u )
				SPARK_FAIL(SPARK_STATUS_DUPLICATE);
		}
		else
		{
			if ( entry->layer_index < state->first_layer_index || entry->layer_index >= state->first_layer_index + state->layer_count )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			local_layer = entry->layer_index - state->first_layer_index;
			if ( (state->layer_seen[local_layer] & (UINT64_C(1) << entry->tensor_kind)) != 0u )
				SPARK_FAIL(SPARK_STATUS_DUPLICATE);
		}
	}
	else if ( (state->global_seen & (UINT64_C(1) << entry->tensor_kind)) != 0u )
		SPARK_FAIL(SPARK_STATUS_DUPLICATE);
	if ( entry->payload_type != shape->payload_type || entry->weight_codec != shape->weight_codec || entry->scale_encoding != shape->scale_encoding || entry->group_count != shape->group_count || entry->rows != shape->rows || entry->columns != shape->columns )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	payload_bytes = SparkGlm5NextStagePackExpectedPayloadBytes(shape);
	scale_bytes = SparkGlm5NextStagePackExpectedScaleBytes(shape);
	if ( payload_bytes == 0u || entry->payload_bytes != payload_bytes || entry->scale_bytes != scale_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	directory_end = header->directory_offset + ((uint64_t)header->tensor_count * header->directory_entry_bytes);
	if ( entry->payload_offset % SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->payload_offset > header->file_bytes || entry->payload_bytes > header->file_bytes - entry->payload_offset )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->payload_offset < directory_end && header->directory_offset < entry->payload_offset + entry->payload_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( scale_bytes == 0u )
	{
		if ( entry->scale_offset != 0u )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	else if ( entry->scale_offset % SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->scale_offset > header->file_bytes || entry->scale_bytes > header->file_bytes - entry->scale_offset )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	else if ( entry->scale_offset < directory_end && header->directory_offset < entry->scale_offset + entry->scale_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextPackMarkSeen(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackEntry *entry)
{
	if ( entry->layer_index == SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER )
		state->global_seen |= UINT64_C(1) << entry->tensor_kind;
	else if ( entry->layer_index == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
		state->mtp_seen |= UINT64_C(1) << entry->tensor_kind;
	else
		state->layer_seen[entry->layer_index - state->first_layer_index] |= UINT64_C(1) << entry->tensor_kind;
}

static SparkStatus SparkGlm5NextPackAssignLayer(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextLayerWeights *weights,
	const SparkGlm5NextStagePackEntry *entry,
	const void *payload,
	const void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_NORM: weights->attn_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A: weights->q_a_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A_NORM: weights->q_a_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_B: weights->q_b_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A: weights->kv_a_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A_NORM: weights->kv_a_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED: weights->kv_b_key_transposed_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_B_VALUE: weights->kv_b_value_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_OUTPUT: weights->attn_output_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_POST_ATTN_NORM: weights->post_attn_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_Q: weights->index_q_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_K: weights->index_k_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_HEAD: weights->index_head_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT: weights->index_norm_weight_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_BIAS: weights->index_norm_bias_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_GATE_UP: weights->dense_gate_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_DOWN: weights->dense_down_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER: weights->router_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER_CORRECTION: weights->router_correction_f32 = (const float *)payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE: weights->expert_up_gate_payload = payload; weights->expert_up_gate_scale = scale; weights->expert_up_gate_payload_offset = entry->payload_offset; weights->expert_up_gate_scale_offset = entry->scale_offset; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN: weights->expert_down_payload = payload; weights->expert_down_scale = scale; weights->expert_down_payload_offset = entry->payload_offset; weights->expert_down_scale_offset = entry->scale_offset; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_GATE_UP: weights->shared_gate_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_DOWN: weights->shared_down_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_QKV_BETA: weights->kda_qkv_beta_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_GATE_DOWN: weights->kda_decay_gate_down_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_UP: weights->kda_decay_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_GATE_UP: weights->kda_gate_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_Q_CONV: weights->kda_q_conv_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_K_CONV: weights->kda_k_conv_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_V_CONV: weights->kda_v_conv_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_BIAS: weights->kda_decay_bias_f32 = (const float *)payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_HEAD_LOG_SCALE: weights->kda_head_log_scale_f32 = (const float *)payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT_NORM: weights->kda_out_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT: weights->kda_out_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_FN: weights->hc_attn_fn_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_BASE: weights->hc_attn_base_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_SCALE: weights->hc_attn_scale_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_FN: weights->hc_ffn_fn_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_BASE: weights->hc_ffn_base_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_SCALE: weights->hc_ffn_scale_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_APE: weights->index_compress_ape_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_GATE: weights->index_compress_gate_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_EH_PROJ: state->mtp_eh_proj_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_ENORM: state->mtp_enorm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_HNORM: state->mtp_hnorm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_SHARED_NORM: state->mtp_shared_norm_bf16 = payload; break;
	default: return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPackAssign(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackEntry *entry,
	const void *payload,
	const void *scale)
{
	if ( entry->layer_index == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
		return(SparkGlm5NextPackAssignLayer(state,&state->mtp_layer,entry,payload,scale));
	if ( entry->layer_index != SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER )
		return(SparkGlm5NextPackAssignLayer(state,&state->layers[entry->layer_index - state->first_layer_index],entry,payload,scale));
	switch ( entry->tensor_kind )
	{
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EMBEDDING: state->embedding_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_LM_HEAD: state->lm_head_bf16 = payload; return(SPARK_STATUS_OK);
	default: return(SPARK_STATUS_SCHEMA_ERROR);
	}
}

typedef struct SparkGlm5NextManifestContext
{
	const SparkGlm5NextStagePackEntry *entries;
	uint32_t count;
} SparkGlm5NextManifestContext;

static int SparkGlm5NextT1Enabled(void);

static void CUDART_CB SparkGlm5NextCompleteAsync(void *context);
static SparkStatus SparkGlm5NextAllocateBytes(
	SparkGlm5NextModuleState *state,
	uint64_t count,
	uint64_t width,
	uint64_t element_bytes,
	void **pointer);

#include "sparkpipe/family/module/spark_module_glm5_next_lineage.h"

#include "sparkpipe/family/module/spark_module_glm5_next_laguna.h"

#include "sparkpipe/family/module/spark_module_manifest_check.h"

static SparkStatus SparkGlm5NextPinAllExperts(SparkGlm5NextModuleState *state);

static SparkStatus SparkGlm5NextExpertPoolBudget(uint64_t *bytes)
{
	if ( getenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES") == 0 ||
	     getenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES")[0] == '\0' )
	{
		fprintf(stderr,"SPARK_WEIGHTD_EXPERT_POOL_BYTES requires an explicit finite budget\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SparkStageModuleEnvironmentUnsigned64OrDefault(SPARK_GLM5_NEXT_MODULE_TAG,"SPARK_WEIGHTD_EXPERT_POOL_BYTES",1u,UINT64_MAX - 1u,1u,bytes));
}

static uint32_t SparkGlm5NextMeshAddressPending(const SparkGlm5NextModuleState *state)
{
	return state->tp_degree > 1u && state->tp_collective_disabled == 0u && state->lazy_pack != 0 && state->lazy_pack->attached.mesh_send_buffer_addr == 0u ? 1u : 0u;
}

static SparkStatus SparkGlm5NextLazyOpen(SparkGlm5NextModuleState *state,const char *path,uint64_t bytes,const SparkGlm5NextStagePackEntry *entries,uint32_t count)
{
	SparkWeightdLazyAttachRequest request;
	SparkGlm5NextManifestContext context = {entries,count};
	SparkStatus status;
	const char *digest;
	uint64_t spine_budget;
	status = SparkWeightdAttachRequested();
	if ( status != SPARK_STATUS_OK )
		return(status == SPARK_STATUS_BUSY ? SPARK_STATUS_UNSUPPORTED : status);
	if ( state->mtp_enabled != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	memset(&request,0,sizeof(request));
	digest = getenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
	if ( digest == 0 || strlen(digest) != 64u || strlen(path) >= sizeof(request.pack_path) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(request.identity.pack_sha256,digest,65u);
	(void)snprintf(request.identity.model,sizeof(request.identity.model),"%s",SPARK_GLM5_NEXT_MODULE_TAG);
	(void)snprintf(request.identity.revision,sizeof(request.identity.revision),"%s",state->model_revision);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = bytes;
	request.identity.topology = state->tp_degree;
	memcpy(request.pack_path,path,strlen(path) + 1u);
	status = SparkGlm5NextExpertPoolBudget(&request.expert_pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64OrDefault(SPARK_GLM5_NEXT_MODULE_TAG,"SPARK_WEIGHTD_SPINE_BUDGET_BYTES",1u,UINT64_MAX,UINT64_C(8589934592),&spine_budget);
	if ( status == SPARK_STATUS_OK )
	{
		uint32_t attach_attempt;
		for ( attach_attempt = 1u; attach_attempt <= SPARK_GLM5_NEXT_LAZY_ATTACH_ATTEMPTS; attach_attempt++ )
		{
			status = SparkWeightdLazyPackCreateChecked(getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET),&request,spine_budget,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS,SparkGlm5NextManifestCheck,&context,&state->lazy_pack);
			if ( status == SPARK_STATUS_OK && SparkGlm5NextMeshAddressPending(state) != 0u )
			{
				if ( attach_attempt == 1u || (attach_attempt % 10u) == 0u )
					fprintf(stderr,"LAZY-ATTACH-MESH-PENDING n=%u\n",attach_attempt);
				status = SparkWeightdLazyPackDestroy(state->lazy_pack);
				state->lazy_pack = 0;
				if ( status != SPARK_STATUS_OK )
					SPARK_RETURN(status);
				status = SPARK_STATUS_BUSY;
			}
			if ( status == SPARK_STATUS_OK || state->lazy_pack != 0 )
				break;
			if ( attach_attempt == 1u || (attach_attempt % 10u) == 0u )
				fprintf(stderr,
					"LAZY-ATTACH-RETRY n=%u status=%d\n",
					attach_attempt,(int32_t)status);
			if ( attach_attempt < SPARK_GLM5_NEXT_LAZY_ATTACH_ATTEMPTS )
			{
				SparkStatus pause_status = SparkStageModulePauseNanoseconds(SPARK_GLM5_NEXT_LAZY_ATTACH_PAUSE_NS);
				if ( pause_status != SPARK_STATUS_OK )
				{
					fprintf(stderr,"LAZY-ATTACH-PAUSE-FAILED n=%u status=%d\n",attach_attempt,(int32_t)pause_status);
					SPARK_RETURN(pause_status);
				}
			}
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		const char *pin_env = getenv("SPARK_GLM5_NEXT_PIN_EXPERTS");
		if ( pin_env != 0 && pin_env[0] == '1' && state->lazy_pack != 0 &&
		     state->lazy_pack->map != 0 )
		{
			SparkStatus pin_status = SparkGlm5NextPinAllExperts(state);
			if ( pin_status != SPARK_STATUS_OK )
			{
				fprintf(stderr,"EXPERT-PIN-FAILED status=%d\n",(int)pin_status);
				SPARK_RETURN(pin_status);
			}
		}
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextPackLoadEntry(
	SparkGlm5NextModuleState *state,
	FILE *file,
	const SparkGlm5NextStagePackEntry *entry)
{
	void *payload,*scale;
	SparkStatus status;
	(void)file;
	payload = 0;
	scale = 0;
	if ( state->lazy_pack == 0 )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( entry->tensor_kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE || entry->tensor_kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN )
		return(SparkGlm5NextPackAssign(state,entry,0,0));
	status = SparkWeightdLazyPackSlice(state->lazy_pack,entry->payload_offset,entry->payload_bytes,(const void **)&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkWeightdLazyPackSlice(state->lazy_pack,entry->scale_offset,entry->scale_bytes,(const void **)&scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackAssign(state,entry,payload,scale);
	SPARK_RETURN(status);
}

static uint64_t SparkGlm5NextExpectedLayerMask(
	const SparkGlm5NextModuleState *state,
	uint32_t layer_index)
{
	SparkGlm5NextStagePackTensorShape shape;
	uint64_t mask;
	uint32_t kind;
	mask = 0u;
	for (kind=SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_NORM; kind<SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KIND_COUNT; kind++)
		if ( SparkGlm5NextStagePackExpectedShape(kind,layer_index,state->expert_weight_codec,state->tp_degree,&shape) == 0 )
			mask |= UINT64_C(1) << kind;
	return(mask);
}

static uint64_t SparkGlm5NextExpectedGlobalMask(const SparkGlm5NextModuleState *state)
{
	uint64_t mask;
	mask = 0u;
	if ( state->owns_embedding != 0u )
		mask |= UINT64_C(1) << SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		mask |= (UINT64_C(1) << SPARK_GLM5_NEXT_STAGEPACK_TENSOR_FINAL_NORM) | (UINT64_C(1) << SPARK_GLM5_NEXT_STAGEPACK_TENSOR_LM_HEAD);
	return(mask);
}

static SparkStatus SparkGlm5NextPackValidateInventory(const SparkGlm5NextModuleState *state)
{
	uint64_t expected_mtp;
	uint32_t local;
	if ( state->global_seen != SparkGlm5NextExpectedGlobalMask(state) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	expected_mtp = state->pack_has_mtp != 0u ?
		SparkGlm5NextExpectedLayerMask(state,SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX) : 0u;
	if ( state->mtp_seen != expected_mtp )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for (local=0u; local<state->layer_count; local++)
		if ( state->layer_seen[local] != SparkGlm5NextExpectedLayerMask(state,state->first_layer_index + local) )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

#include "sparkpipe/family/module/spark_module_pack_file_size.h"

static SparkStatus SparkGlm5NextPackLoad(
	SparkGlm5NextModuleState *state,
	const char *path)
{
	SparkGlm5NextStagePackHeader header;
	SparkGlm5NextStagePackEntry entries[SPARK_GLM5_NEXT_STAGEPACK_MAX_TENSOR_COUNT];
	SparkGlm5NextStagePackTensorShape shape;
	FILE *file;
	uint64_t file_bytes;
	uint32_t index;
	SparkStatus status;
	file = fopen(path,"rb");
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	memset(&header,0,sizeof(header));
	memset(entries,0,sizeof(entries));
	status = SparkGlm5NextPackFileSize(file,&file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_GLM5_NEXT_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackValidateHeader(state,&header,file_bytes);
	if ( status == SPARK_STATUS_OK )
		state->pack_has_mtp = (header.flags & SPARK_GLM5_NEXT_STAGEPACK_FLAG_MTP) != 0u ? 1u : 0u;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_GLM5_NEXT_MODULE_TAG,file,header.directory_offset,entries,(uint64_t)header.tensor_count * sizeof(entries[0]));
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
	{
		status = SparkGlm5NextPackValidateEntryGeometry(state,&header,&entries[index],&shape);
		if ( status == SPARK_STATUS_OK )
			SparkGlm5NextPackMarkSeen(state,&entries[index]);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackValidateRanges(entries,header.tensor_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackValidateInventory(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextLazyOpen(state,path,file_bytes,entries,header.tensor_count);
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
		status = SparkGlm5NextPackLoadEntry(state,file,&entries[index]);
	if ( fclose(file) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextMtpPackCheck(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackHeader *header,
	const SparkGlm5NextStagePackEntry *entries,
	uint64_t file_bytes,
	uint64_t *device_bytes_out)
{
	SparkGlm5NextStagePackTensorShape shape;
	uint64_t directory_bytes,directory_end,device_bytes;
	uint32_t index;
	SparkStatus status;
	*device_bytes_out = 0u;
	if ( header->magic != SPARK_GLM5_NEXT_STAGEPACK_MAGIC || header->format_version != SPARK_GLM5_NEXT_STAGEPACK_FORMAT_VERSION || header->header_bytes != SPARK_GLM5_NEXT_STAGEPACK_HEADER_BYTES || header->directory_entry_bytes != SPARK_GLM5_NEXT_STAGEPACK_ENTRY_BYTES || header->codec_abi_version != SPARK_WEIGHT_CODEC_ABI_VERSION )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( header->flags != SPARK_GLM5_NEXT_STAGEPACK_FLAG_MTP || SparkGlm5NextStagePackHeaderTpDegree(header) != state->tp_degree || SparkGlm5NextStagePackHeaderTpRank(header) != state->tp_rank )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->tensor_count == 0u || header->tensor_count > SPARK_GLM5_NEXT_STAGEPACK_MAX_TENSOR_COUNT || header->stage_count != 1u || header->stage_index != 0u || header->first_layer_index != SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX || header->layer_count != 1u || header->total_layer_count != SPARK_GLM5_NEXT_MODEL_LAYER_COUNT || header->hidden_dimension != SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION || header->vocab_count != SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT || header->routed_expert_count != SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 || header->expert_weight_codec != state->expert_weight_codec || header->kv_cache_codec != SPARK_WEIGHT_CODEC_BF16 )
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	directory_bytes = (uint64_t)header->tensor_count * header->directory_entry_bytes;
	directory_end = header->directory_offset + directory_bytes;
	if ( header->file_bytes != file_bytes || header->directory_offset < header->header_bytes || header->directory_offset % SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES != 0u || directory_end < header->directory_offset || directory_end > file_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	state->mtp_seen = 0u;
	device_bytes = 0u;
	for (index=0u; index<header->tensor_count; index++)
	{
		if ( entries[index].layer_index != SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		status = SparkGlm5NextPackValidateEntryGeometry(state,header,&entries[index],&shape);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		SparkGlm5NextPackMarkSeen(state,&entries[index]);
		device_bytes += (entries[index].payload_bytes + SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES - 1u) / SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES * SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES;
		device_bytes += (entries[index].scale_bytes + SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES - 1u) / SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES * SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES;
	}
	status = SparkGlm5NextPackValidateRanges(entries,header->tensor_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( state->mtp_seen != SparkGlm5NextExpectedLayerMask(state,SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	*device_bytes_out = device_bytes;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextMtpPackUploadRange(FILE *file,uint64_t offset,uint64_t bytes,uint8_t *device,uint8_t *bounce,uint64_t bounce_bytes)
{
	uint64_t done,chunk;
	SparkStatus status;
	for (done=0u; done<bytes; done+=chunk)
	{
		chunk = bytes - done < bounce_bytes ? bytes - done : bounce_bytes;
		status = SparkStageModulePackRead(SPARK_GLM5_NEXT_MODULE_TAG,file,offset + done,bounce,chunk);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,cudaMemcpy(device + done,bounce,(size_t)chunk,cudaMemcpyHostToDevice),"mtp_pack_upload");
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

#define SPARK_GLM5_NEXT_MTP_PACK_BOUNCE_BYTES (UINT64_C(64) << 20)

static SparkStatus SparkGlm5NextMtpPackLoad(SparkGlm5NextModuleState *state,const char *path)
{
	SparkGlm5NextStagePackHeader header;
	SparkGlm5NextStagePackEntry *entries;
	uint8_t *device,*bounce,*cursor;
	const void *payload,*scale;
	FILE *file;
	uint64_t file_bytes,device_bytes;
	uint32_t index;
	SparkStatus status;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"GLM verify MTP pack %s cannot be opened: %s\n",path,strerror(errno));
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	entries = (SparkGlm5NextStagePackEntry *)calloc(SPARK_GLM5_NEXT_STAGEPACK_MAX_TENSOR_COUNT,sizeof(*entries));
	bounce = (uint8_t *)malloc(SPARK_GLM5_NEXT_MTP_PACK_BOUNCE_BYTES);
	device = 0;
	device_bytes = 0u;
	memset(&header,0,sizeof(header));
	status = entries != 0 && bounce != 0 ? SparkGlm5NextPackFileSize(file,&file_bytes) : SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_GLM5_NEXT_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK && (header.tensor_count == 0u || header.tensor_count > SPARK_GLM5_NEXT_STAGEPACK_MAX_TENSOR_COUNT) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_GLM5_NEXT_MODULE_TAG,file,header.directory_offset,entries,(uint64_t)header.tensor_count * sizeof(entries[0]));
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextMtpPackCheck(state,&header,entries,file_bytes,&device_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateBytes(state,1u,device_bytes,1u,(void **)&device);
	cursor = device;
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
	{
		payload = cursor;
		status = SparkGlm5NextMtpPackUploadRange(file,entries[index].payload_offset,entries[index].payload_bytes,cursor,bounce,SPARK_GLM5_NEXT_MTP_PACK_BOUNCE_BYTES);
		cursor += (entries[index].payload_bytes + SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES - 1u) / SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES * SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES;
		scale = 0;
		if ( status == SPARK_STATUS_OK && entries[index].scale_bytes != 0u )
		{
			scale = cursor;
			status = SparkGlm5NextMtpPackUploadRange(file,entries[index].scale_offset,entries[index].scale_bytes,cursor,bounce,SPARK_GLM5_NEXT_MTP_PACK_BOUNCE_BYTES);
			cursor += (entries[index].scale_bytes + SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES - 1u) / SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES * SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES;
		}
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextPackAssignLayer(state,&state->mtp_layer,&entries[index],payload,scale);
	}
	if ( fclose(file) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	free(entries);
	free(bounce);
	if ( status == SPARK_STATUS_OK )
	{
		state->mtp_sidecar = 1u;
		state->mtp_sidecar_bytes = device_bytes;
		fprintf(stderr,"GLM verify MTP pack %s tensors=%u device_bytes=%llu tp=%u rank=%u\n",path,header.tensor_count,(unsigned long long)device_bytes,state->tp_degree,state->tp_rank);
	}
	else
		fprintf(stderr,"GLM verify MTP pack %s rejected: status=%d\n",path,(int)status);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextMtpPackOpen(SparkGlm5NextModuleState *state)
{
	char path[1024];
	const char *directory;
	if ( state->verify_mtp == 0u )
		return(SPARK_STATUS_OK);
	directory = getenv(SPARK_GLM5_NEXT_VERIFY_MTP_DIR_ENV);
	if ( SparkGlm5NextMtpPackPath(directory,state->tp_degree,state->tp_rank,path,(uint32_t)sizeof(path)) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s must name the directory that holds " SPARK_GLM5_NEXT_VERIFY_MTP_PACK_PREFIX "%u.rank%u.g5nsp for the MTP drafter\n",SPARK_GLM5_NEXT_VERIFY_MTP_DIR_ENV,state->tp_degree,state->tp_rank);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SparkGlm5NextMtpPackLoad(state,path));
}

static SparkStatus SparkGlm5NextAllocateSlotHost(SparkGlm5NextExecutionSlot *slot)
{
	uint32_t *cursor;
	uint64_t rows,words,bytes;
	cudaError_t error;
	if ( slot == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	rows = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT;
	words = (rows * (4u + sizeof(SparkRowSampling) / sizeof(uint32_t))) + SPARK_GLM5_NEXT_CHAIN_TOKEN_CAPACITY + SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT +
	    SPARK_GLM5_NEXT_MODEL_LAYER_COUNT *
	        (SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u);
	bytes = words * sizeof(uint32_t);
	error = cudaHostAlloc(&slot->host_staging,bytes,cudaHostAllocPortable);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"host_staging"));
	memset(slot->host_staging,0,bytes);
	slot->host_row_sampling = (SparkRowSampling *)slot->host_staging;
	cursor = (uint32_t *)(slot->host_row_sampling + rows);
	slot->host_token_ids = cursor;
	cursor += rows;
	slot->host_resident_slots = cursor;
	cursor += rows;
	slot->host_positions = cursor;
	cursor += rows;
	slot->host_output_token_ids = cursor;
	cursor += rows;
	slot->host_chain_token_ids = cursor;
	cursor += SPARK_GLM5_NEXT_CHAIN_TOKEN_CAPACITY;
	slot->host_kv_access_error = cursor;
	cursor += SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT;
	slot->host_group_row_offset = cursor;
	bytes = ((rows + 1u) * 2u + rows) * sizeof(uint32_t);
	error = cudaHostAlloc((void **)&slot->host_run_begin,bytes,cudaHostAllocPortable);
	if ( error != cudaSuccess )
	{
		(void)cudaFreeHost(slot->host_staging);
		slot->host_staging = 0;
		return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"host_run_staging"));
	}
	memset(slot->host_run_begin,0,bytes);
	cursor = (uint32_t *)slot->host_run_begin;
	cursor += rows + 1u;
	slot->host_run_state_index = cursor;
	cursor += rows;
	slot->host_run_row_indices = cursor;
	error = cudaEventCreateWithFlags((cudaEvent_t *)&slot->route_ready_event,cudaEventDisableTiming);
	return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"route_ready_event"));
}

static void SparkGlm5NextGraphDestroyAll(SparkGlm5NextExecutionSlot *slot)
{
	uint32_t regime,index;
	for ( regime = 0u; regime < SPARK_GLM5_NEXT_GRAPH_REGIME_COUNT; regime++ )
		for ( index = 0u; index < SPARK_GLM5_NEXT_GRAPH_ROWS_MAX; index++ )
		{
			if ( slot->graph_exec_rows[regime][index] != 0 )
				(void)cudaGraphExecDestroy((cudaGraphExec_t)slot->graph_exec_rows[regime][index]);
			slot->graph_exec_rows[regime][index] = 0;
			slot->graph_bound_rows[regime][index] = 0u;
		}
	slot->graph_failed_rows = 0u;
	slot->graph_exec_a = 0;
	for ( regime = 0u; regime < SPARK_GLM5_NEXT_GRAPH_REGIME_COUNT; regime++ )
		for ( index = 0u; index < SPARK_GLM5_NEXT_VERIFY_TABLE_COUNT; index++ )
		{
			if ( slot->verify_exec[regime][index] != 0 )
				(void)cudaGraphExecDestroy((cudaGraphExec_t)slot->verify_exec[regime][index]);
			slot->verify_exec[regime][index] = 0;
			slot->verify_bound[regime][index] = 0u;
		}
	slot->verify_captured = 0u;
	for ( index = 0u; index < SPARK_GLM5_NEXT_REPLAY_ROWS_MAX; index++ )
	{
		if ( slot->replay_fold_exec[index] != 0 )
			(void)cudaGraphExecDestroy((cudaGraphExec_t)slot->replay_fold_exec[index]);
		slot->replay_fold_exec[index] = 0;
	}
}

static void SparkGlm5NextReleaseSlotHost(SparkGlm5NextModuleState *state)
{
	uint32_t index;
	if ( state == 0 )
		return;
	for (index=0u; index<state->pipeline_slot_count; index++)
	{
		if ( state->slots[index].route_ready_event != 0 )
			(void)cudaEventDestroy((cudaEvent_t)state->slots[index].route_ready_event);
		state->slots[index].route_ready_event = 0;
		state->slots[index].route_recorded = 0u;
		SparkGlm5NextGraphDestroyAll(&state->slots[index]);
		if ( state->slots[index].miss_ring != 0 )
			(void)cudaFreeHost(state->slots[index].miss_ring);
		state->slots[index].miss_ring = 0;
		if ( state->slots[index].host_staging != 0 )
			(void)cudaFreeHost(state->slots[index].host_staging);
		state->slots[index].host_staging = 0;
		if ( state->slots[index].host_run_begin != 0 )
			(void)cudaFreeHost(state->slots[index].host_run_begin);
		state->slots[index].host_run_begin = 0;
		state->slots[index].host_run_state_index = 0;
		state->slots[index].host_run_row_indices = 0;
		if ( state->slots[index].replay_committed_host != 0 )
			(void)cudaFreeHost(state->slots[index].replay_committed_host);
		state->slots[index].replay_committed_host = 0;
	}
}

static SparkStatus SparkGlm5NextAllocateSlotMetadata(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot)
{
	SparkStatus status;
	status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->token_ids);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->resident_slots);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->positions);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(SparkRowSampling),(void **)&slot->row_sampling);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity + 1u,1u,sizeof(uint32_t),(void **)&slot->run_begin);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->run_state_index);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->run_row_indices);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->resident_sequence_capacity,1u,sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_tile_prefix);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,sizeof(uint32_t) * 6u,1u,&slot->kv_access_error);
	SPARK_RETURN(status);
}

static uint64_t SparkGlm5NextKdaScratchWidth(const SparkGlm5NextModuleState *state,uint64_t width)
{
	uint64_t kda;
	kda = (uint64_t)(SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT / state->tp_degree) *
		(SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION > SPARK_GLM5_NEXT_MODEL_KDA_HEAD_VALUE_DIMENSION ?
			SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION : SPARK_GLM5_NEXT_MODEL_KDA_HEAD_VALUE_DIMENSION);
	return(width > kda ? width : kda);
}

static SparkStatus SparkGlm5NextAllocateSlotHidden(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot)
{
	uint64_t rows;
	SparkStatus status;
	rows = state->execution_row_capacity;
	status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->residual_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->normed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->hc_collapsed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->hc_snapshot_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->hc_mean_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION,sizeof(float),(void **)&slot->hc_mixes_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT,sizeof(float),(void **)&slot->hc_pre_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT,sizeof(float),(void **)&slot->hc_post_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HC_MULT,sizeof(float),(void **)&slot->hc_comb_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_QUERY_A_DIMENSION,(void **)&slot->q_compressed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_QUERY_B_DIMENSION,(void **)&slot->q_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_LATENT_DIMENSION,(void **)&slot->query_latent_bf16);
	slot->query_rope_bf16 = 0;
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_QUERY_DIMENSION,(void **)&slot->index_query_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_DIMENSION,(void **)&slot->index_key_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_COUNT,(void **)&slot->index_head_weight_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_DIMENSION,(void **)&slot->index_gate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION,(void **)&slot->index_packed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL,sizeof(uint32_t),(void **)&slot->selected_pools);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,2u * SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION + SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION + SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,(void **)&slot->fused_qkvb_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,2u * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,(void **)&slot->fused_decay_gate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,(void **)&slot->kda_decay_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,(void **)&slot->kda_gate_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,(void **)&slot->kda_beta_logit);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION,(void **)&slot->kda_gate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION,(void **)&slot->kda_decay_logit_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->kda_output_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION,sizeof(float),(void **)&slot->kda_retention);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,sizeof(float),(void **)&slot->kda_write_gate);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SparkGlm5NextKdaScratchWidth(state,SPARK_GLM5_NEXT_MODEL_CACHE_TOKEN_ELEMENTS),(void **)&slot->kv_slot_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_LATENT_DIMENSION,(void **)&slot->attention_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_VALUE_HEAD_DIMENSION,(void **)&slot->attention_value_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SparkGlm5NextKdaScratchWidth(state,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION),(void **)&slot->attention_out_bf16);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextAllocateSlotMlp(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot)
{
	uint64_t rows,packed_rows;
	SparkStatus status;
	rows = state->execution_row_capacity;
	packed_rows = rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K;
	status = SparkGlm5NextAllocateRows(state,rows,SparkGlm5NextKdaScratchWidth(state,SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_GATE_UP_DIMENSION),(void **)&slot->gate_up_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_MOE_TOP_K * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION,(void **)&slot->intermediate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,packed_rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->expert_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->shared_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,sizeof(float),(void **)&slot->router_logits_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,state->max_sequence_positions / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL,sizeof(float),(void **)&slot->selection_scores_f32);
	if ( status == SPARK_STATUS_OK && state->index_cp != 0u ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS,sizeof(float),(void **)&slot->index_local_scores_f32);
	if ( status == SPARK_STATUS_OK && state->index_cp != 0u ) status = SparkGlm5NextAllocateBytes(state,rows * state->tp_degree,SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS,sizeof(float),(void **)&slot->index_gathered_scores_f32);
	if ( status == SPARK_STATUS_OK && state->kv_shard != 0u ) status = SparkGlm5NextAllocateBytes(state,state->tp_degree,SparkGlm5NextKvShardQueryStride((uint32_t)rows,state->tp_degree),sizeof(uint16_t),(void **)&slot->kv_shard_query_gathered_bf16);
	if ( status == SPARK_STATUS_OK && state->kv_shard != 0u ) status = SparkGlm5NextAllocateBytes(state,state->tp_degree,SparkGlm5NextKvShardPartialStrideCapacity(state->tp_degree,(uint32_t)rows),sizeof(float),(void **)&slot->kv_shard_partials_f32);
	if ( status == SPARK_STATUS_OK && state->kv_shard != 0u ) status = SparkGlm5NextAllocateBytes(state,state->tp_degree,SparkGlm5NextKvShardPartialStrideCapacity(state->tp_degree,(uint32_t)rows),sizeof(float),(void **)&slot->kv_shard_partials_received_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT / state->tp_degree),SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_FLOATS,sizeof(float),(void **)&slot->attention_split_partials_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH,sizeof(uint32_t),(void **)&slot->selected_positions);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_expert);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,packed_rows,1u,sizeof(float),(void **)&slot->route_weight);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_source_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_packed_row);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_tile_prefix_w1);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_tile_prefix_w2);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextAllocateSlotHead(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot)
{
	uint64_t rows,tiles;
	SparkStatus status;
	rows = state->execution_row_capacity;
	tiles = SparkCeilDivU64(SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,SPARK_GLM5_NEXT_HEAD_TILE);
	status = SparkGlm5NextAllocateBytes(state,rows,tiles,sizeof(float),(void **)&slot->head_candidate_score);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,tiles,sizeof(uint32_t),(void **)&slot->head_candidate_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,1u,sizeof(uint32_t),(void **)&slot->output_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,1u,sizeof(float),(void **)&slot->output_score);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,1u,sizeof(uint64_t),(void **)&slot->head_maxloc_u64);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
	{
		uint64_t shard_rows = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT / state->tp_degree;
		status = SparkGlm5NextAllocateBytes(state,1u,SparkHeadCertifiedFp8ScratchBytes(shard_rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION),1u,(void **)&slot->head_certified_scratch);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextAllocateBytes(state,1u,SparkHeadCertifiedFp8CandidateBytes(shard_rows),1u,(void **)&slot->head_certified_candidates);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextAllocateBytes(state,1u,1u,sizeof(uint32_t),(void **)&slot->head_screened_count);
	}
	SPARK_RETURN(status);
}

static void SparkGlm5NextShareSlotDevice(SparkGlm5NextExecutionSlot *slot,const SparkGlm5NextExecutionSlot *shared)
{
	SparkGlm5NextExecutionSlot host;
	host = *slot;
	*slot = *shared;
	slot->stream = host.stream;
	slot->route_ready_event = host.route_ready_event;
	slot->route_recorded = 0u;
	slot->host_staging = host.host_staging;
	slot->host_row_sampling = host.host_row_sampling;
	slot->host_token_ids = host.host_token_ids;
	slot->host_resident_slots = host.host_resident_slots;
	slot->host_positions = host.host_positions;
	slot->host_output_token_ids = host.host_output_token_ids;
	slot->host_chain_token_ids = host.host_chain_token_ids;
	slot->host_kv_access_error = host.host_kv_access_error;
	slot->host_group_row_offset = host.host_group_row_offset;
	slot->host_run_begin = host.host_run_begin;
	slot->host_run_state_index = host.host_run_state_index;
	slot->host_run_row_indices = host.host_run_row_indices;
}

static SparkStatus SparkGlm5NextAllocateSlots(SparkGlm5NextModuleState *state)
{
	uint32_t index;
	SparkStatus status;
	status = SPARK_STATUS_OK;
	for (index=0u; status==SPARK_STATUS_OK && index<state->pipeline_slot_count; index++)
	{
		state->slots[index].stream = state->execution_stream;
		status = SparkGlm5NextAllocateSlotHost(&state->slots[index]);
		if ( status == SPARK_STATUS_OK && index != 0u )
			SparkGlm5NextShareSlotDevice(&state->slots[index],&state->slots[0]);
		if ( status == SPARK_STATUS_OK && index == 0u ) status = SparkGlm5NextAllocateSlotMetadata(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK && index == 0u ) status = SparkGlm5NextAllocateSlotHidden(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK && index == 0u ) status = SparkGlm5NextAllocateSlotMlp(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK && index == 0u ) status = SparkGlm5NextAllocateSlotHead(state,&state->slots[index]);
	}
	SPARK_RETURN(status);
}

static uint32_t SparkGlm5NextReplayRows(const SparkGlm5NextModuleState *state)
{
	uint32_t rows = state->mtp_enabled != 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u : 0u;
	return(state->verify_rows_max > rows ? state->verify_rows_max : rows);
}

static SparkStatus SparkGlm5NextAllocateReplay(SparkGlm5NextModuleState *state)
{
	SparkGlm5NextKdaReplayLayout layout;
	uint64_t replay_bytes,steps_bytes,conv_bytes;
	uint32_t index,rank_heads,rows = SparkGlm5NextReplayRows(state);
	SparkStatus status = SPARK_STATUS_OK;
	if ( rows == 0u )
		return(SPARK_STATUS_OK);
	if ( rows > SPARK_GLM5_NEXT_REPLAY_ROWS_MAX || state->execution_row_capacity < rows )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	rank_heads = SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT / state->tp_degree;
	layout = SparkGlm5NextKdaReplayLayoutFor(rank_heads,rows);
	state->kda_replay_layer_bytes = layout.layer_bytes;
	replay_bytes = layout.layer_bytes * state->kda_layer_count;
	steps_bytes = (uint64_t)state->kda_layer_count * SPARK_GLM5_NEXT_REPLAY_ROWS_MAX * SPARK_GLM5_NEXT_REPLAY_ROWS_MAX * SPARK_GLM5_NEXT_MTP_REPLAY_STEP_BYTES;
	conv_bytes = (uint64_t)rows * rank_heads * SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION * SPARK_GLM5_NEXT_MODEL_BF16_ELEMENT_BYTES;
	for (index=0u; status==SPARK_STATUS_OK && index<state->pipeline_slot_count; index++)
	{
		SparkGlm5NextExecutionSlot *slot = &state->slots[index];
		status = SparkGlm5NextAllocateBytes(state,1u,sizeof(uint32_t),1u,(void **)&slot->mtp_committed);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,steps_bytes,1u,&slot->mtp_replay_steps);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,conv_bytes,1u,(void **)&slot->mtp_conv_scratch);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,replay_bytes,1u,(void **)&slot->kda_replay_pool);
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextAllocateMtp(SparkGlm5NextModuleState *state)
{
	uint64_t kv_pool_bytes,index_pool_bytes;
	uint32_t index,step;
	SparkStatus status;
	if ( state->mtp_enabled == 0u && state->verify_mtp == 0u )
		return(SPARK_STATUS_OK);
	if ( state->verify_mtp != 0u )
	{
		state->mtp_lane_sequence = (uint64_t *)calloc(state->resident_sequence_capacity,sizeof(uint64_t));
		state->mtp_lane_next = (uint64_t *)calloc(state->resident_sequence_capacity,sizeof(uint64_t));
		if ( state->mtp_lane_sequence == 0 || state->mtp_lane_next == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	kv_pool_bytes = (uint64_t)SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES;
	index_pool_bytes = (uint64_t)SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u * SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT;
	state->mtp_lane_armed = (uint8_t *)calloc(state->resident_sequence_capacity,1u);
	if ( state->mtp_lane_armed == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkGlm5NextAllocateBytes(state,state->resident_sequence_capacity,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,sizeof(uint16_t),(void **)&state->mtp_lane_hidden_bf16);
	for (index=0u; status==SPARK_STATUS_OK && index<state->pipeline_slot_count; index++)
	{
		SparkGlm5NextExecutionSlot *slot = &state->slots[index];
		status = SparkGlm5NextAllocateRows(state,1u,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->mtp_hidden_bf16);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,1u,2u * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->mtp_concat_bf16);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,kv_pool_bytes,1u,(void **)&slot->mtp_kv_pool);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,index_pool_bytes,1u,(void **)&slot->mtp_index_pool);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX,sizeof(uint32_t),1u,(void **)&slot->mtp_positions);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX,sizeof(uint32_t),1u,(void **)&slot->mtp_context);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX,sizeof(uint32_t),1u,(void **)&slot->mtp_draft_device);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,sizeof(uint32_t),1u,(void **)&slot->mtp_page_table);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,sizeof(uint32_t),1u,(void **)&slot->mtp_sequence);
		if ( status != SPARK_STATUS_OK )
			break;
		{
			uint32_t meta[2u * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX];
			cudaError_t error;
			for ( step = 0u; step < SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX; ++step )
			{
				meta[step] = step;
				meta[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX + step] = step + 1u;
			}
			error = cudaMemcpy(slot->mtp_positions,meta,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX * sizeof(uint32_t),cudaMemcpyHostToDevice);
			if ( error == cudaSuccess )
				error = cudaMemcpy(slot->mtp_context,meta + SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX * sizeof(uint32_t),cudaMemcpyHostToDevice);
			if ( error == cudaSuccess )
				error = cudaMemset(slot->mtp_page_table,0,sizeof(uint32_t));
			if ( error == cudaSuccess )
				error = cudaMemset(slot->mtp_sequence,0,sizeof(uint32_t));
			status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"mtp_meta_init");
		}
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextBuildPageTable(SparkGlm5NextModuleState *state)
{
	uint64_t entries;
	SparkStatus status;
	cudaError_t error;
	state->pages_per_sequence = SparkCeilDivU32(state->max_sequence_positions,64u);
	if ( state->pages_per_sequence == 0u || state->resident_sequence_capacity > UINT32_MAX / state->pages_per_sequence )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->page_count == 0u || state->physical_page_count == 0u || state->physical_page_count > state->page_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	entries = (uint64_t)state->resident_sequence_capacity * state->pages_per_sequence;
	state->page_table_shadow = (uint32_t *)malloc(entries * sizeof(uint32_t));
	if ( state->page_table_shadow == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(state->page_table_shadow,0xff,entries * sizeof(uint32_t));
	status = SparkGlm5NextAllocateBytes(state,entries,1u,sizeof(uint32_t),(void **)&state->page_table);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemset(state->page_table,0xff,entries * sizeof(uint32_t));
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"page_table");
	}
	SPARK_RETURN(status);
}

// Checkpoints pack all KDA layers, then all Q, K and V convolution layers.
// The caller owns the resident slot until the complete transfer succeeds.
static inline SparkStatus SparkGlm5NextRecurrentCopy(SparkGlm5NextModuleState *state,uint32_t direction,uint32_t slot,void *host,uint64_t bytes)
{
	SparkKvLayeredPageLayout layout;
	uint8_t *pools[4];
	uint64_t strides[4],payloads[4],total = 0u,offset = 0u;
	uint32_t part;
	SparkStatus status;
	if ( state == 0 || host == 0 || state->resident_sequence_capacity == 0u || state->kda_layer_count == 0u || slot >= state->resident_sequence_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( direction != SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST && direction != SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	pools[0] = state->kda_state_pools;
	pools[1] = state->kda_q_window_pool;
	pools[2] = state->kda_k_window_pool;
	pools[3] = state->kda_v_window_pool;
	strides[0] = state->kda_state_layer_stride_bytes;
	strides[1] = strides[2] = strides[3] = state->kda_window_layer_stride_bytes;
	for (part=0u; part<4u; part++)
	{
		if ( pools[part] == 0 || strides[part] == 0u || strides[part] % state->resident_sequence_capacity != 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( strides[part] > (UINTPTR_MAX - (uintptr_t)pools[part]) / state->kda_layer_count )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		payloads[part] = (strides[part] / state->resident_sequence_capacity) * state->kda_layer_count;
		if ( payloads[part] > UINT64_MAX - total )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		total += payloads[part];
	}
	if ( bytes != total || bytes > UINTPTR_MAX - (uintptr_t)host )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	layout.layer_count = state->kda_layer_count;
	layout.page_count = state->resident_sequence_capacity;
	for (part=0u; part<4u; part++)
	{
		layout.device_base = (uintptr_t)pools[part];
		layout.device_bytes = strides[part] * layout.layer_count;
		layout.layer_stride_bytes = strides[part];
		layout.layer_page_bytes = strides[part] / layout.page_count;
		status = SparkKvPageStoreCopyLayered(&layout,direction,slot,(uint8_t *)host + offset,payloads[part],SparkGlm5NextDevicePageCopy,state);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		offset += payloads[part];
	}
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextKdaCount(atomic_uint_fast64_t *counters,uint64_t bytes,uint64_t start_ns)
{
	uint64_t now_ns = SparkGlm5NextNowNs();
	atomic_fetch_add_explicit(&counters[SPARK_GLM5_NEXT_KDA_COUNT],1u,memory_order_relaxed);
	atomic_fetch_add_explicit(&counters[SPARK_GLM5_NEXT_KDA_BYTES],bytes,memory_order_relaxed);
	atomic_fetch_add_explicit(&counters[SPARK_GLM5_NEXT_KDA_NS],now_ns >= start_ns ? now_ns - start_ns : 0u,memory_order_relaxed);
}

static SparkStatus SparkGlm5NextPageCopy(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes)
{
	SparkGlm5NextModuleState *state;
	SparkKvLayeredPageLayout layout;
	uint64_t offset,packed_page_bytes;
	state = (SparkGlm5NextModuleState *)context;
	if ( state == 0 || state->physical_page_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->index_layer_count != 0u && state->index_layer_stride_bytes > UINT64_MAX / state->index_layer_count )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	layout.device_base = (uintptr_t)state->kv_cache;
	layout.layer_stride_bytes = state->kv_layer_stride_bytes;
	layout.layer_count = state->kv_layer_count;
	layout.page_count = state->physical_page_count;
	if ( device_address >= (uintptr_t)state->index_cache && device_address - (uintptr_t)state->index_cache < state->index_layer_stride_bytes * state->index_layer_count )
	{
		layout.device_base = (uintptr_t)state->index_cache;
		layout.layer_stride_bytes = state->index_layer_stride_bytes;
		layout.layer_count = state->index_layer_count;
	}
	if ( layout.layer_count == 0u || layout.layer_stride_bytes % layout.page_count != 0u || layout.layer_stride_bytes > UINT64_MAX / layout.layer_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	layout.device_bytes = layout.layer_stride_bytes * layout.layer_count;
	layout.layer_page_bytes = layout.layer_stride_bytes / layout.page_count;
	packed_page_bytes = layout.layer_page_bytes * layout.layer_count;
	if ( device_address < layout.device_base || device_address - layout.device_base >= layout.device_bytes || packed_page_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	// Arena block addresses name packed payloads. Translate their page index
	// to the native layer-major allocation before a device copy dereferences it.
	offset = device_address - layout.device_base;
	if ( offset % packed_page_bytes != 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvPageStoreCopyLayered(&layout,direction,(uint32_t)(offset / packed_page_bytes),host_address,bytes,SparkGlm5NextDevicePageCopy,state));
}

static SparkStatus SparkGlm5NextBackingCapacity(SparkGlm5NextModuleState *state,uint64_t kv_page_bytes)
{
	uint64_t window_bytes,state_bytes,total;
	if ( state->page_count == 0u || state->resident_sequence_capacity == 0u || kv_page_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state->recurrent_page_bytes = 0u;
	if ( state->kda_layer_count != 0u )
	{
		if ( state->kda_state_layer_stride_bytes % state->resident_sequence_capacity != 0u || state->kda_window_layer_stride_bytes % state->resident_sequence_capacity != 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		state_bytes = state->kda_state_layer_stride_bytes / state->resident_sequence_capacity;
		window_bytes = state->kda_window_layer_stride_bytes / state->resident_sequence_capacity;
		if ( state_bytes == 0u || window_bytes == 0u || window_bytes > (UINT64_MAX - state_bytes) / 3u )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		state_bytes += 3u * window_bytes;
		if ( state_bytes > (UINT64_MAX - kv_page_bytes) / state->kda_layer_count )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		state->recurrent_page_bytes = state_bytes * state->kda_layer_count;
	}
	total = kv_page_bytes + state->recurrent_page_bytes;
	if ( total > INT64_MAX / state->page_count || state->recurrent_page_bytes > SIZE_MAX / 2u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	total *= state->page_count;
	if ( state->kv_backing_maximum_bytes != 0u && state->kv_backing_maximum_bytes < total )
	{
		fprintf(stderr,"GLM cache backing budget insufficient: need %llu bytes for %u pages, configured %llu\n",(unsigned long long)total,state->page_count,(unsigned long long)state->kv_backing_maximum_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextRecurrentInitialize(SparkGlm5NextModuleState *state,const char *backing_path)
{
	SparkKvPageStoreConfiguration config = {0};
	SparkStatus status;
	if ( state->kda_layer_count == 0u )
		return(SPARK_STATUS_OK);
	if ( state->recurrent_page_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cudaHostAlloc((void **)&state->recurrent_staging,2u * state->recurrent_page_bytes,cudaHostAllocPortable) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	config.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	config.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS;
	config.logical_page_capacity = state->page_count;
	config.transfer_capacity = 1u;
	config.page_bytes = state->recurrent_page_bytes;
	config.maximum_backing_bytes = state->page_count * state->recurrent_page_bytes;
	config.backing_path = backing_path;
	// First half is caller-owned gather/scatter storage; the worker uses the second.
	config.staging_address = state->recurrent_staging + state->recurrent_page_bytes;
	config.staging_bytes = state->recurrent_page_bytes;
	status = SparkKvPageStoreInitialize(&state->recurrent_store,&config);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheAttachStateStore(&state->kv_page_cache,&state->recurrent_store);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextKvInitialize(SparkGlm5NextModuleState *state)
{
	SparkKvModelTable table;
	uint64_t block_bytes,index_block_bytes,payload_bytes;
	uint32_t arena_head_dim;
	uint64_t lane_page_entries;
	SparkStatus status;
	if ( pthread_mutex_init(&state->completion_queue_lock,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	state->completion_queue_lock_initialized = 1u;
	if ( pthread_mutex_init(&state->kv_mutex,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	state->kv_mutex_initialized = 1u;
	if ( state->kv_layer_count == 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	arena_head_dim = state->kv_shard != 0u ? SPARK_GLM5_NEXT_KV_ARENA_HEAD_DIM / state->tp_degree : SPARK_GLM5_NEXT_KV_ARENA_HEAD_DIM;
	block_bytes = (uint64_t)SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT *
		(uint64_t)state->kv_layer_count * arena_head_dim *
		SPARK_GLM5_NEXT_KV_BYTES_PER_SCALAR;
	index_block_bytes = (uint64_t)SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT * state->index_layer_count * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u;
	if ( state->kv_shard != 0u )
		index_block_bytes /= state->tp_degree;
	if ( index_block_bytes > UINT64_MAX - block_bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	payload_bytes = block_bytes + index_block_bytes;
	status = SparkGlm5NextBackingCapacity(state,payload_bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	lane_page_entries = (uint64_t)state->resident_sequence_capacity *
		state->pages_per_sequence;
	state->kv_blocks = (SparkKvCacheBlock *)calloc(state->page_count,sizeof(*state->kv_blocks));
	state->kv_resident_slot_logical_block_indices = (uint32_t *)calloc(state->physical_page_count,sizeof(*state->kv_resident_slot_logical_block_indices));
	state->kv_entries = (SparkKvPageCacheEntry *)calloc(state->page_count,sizeof(*state->kv_entries));
	state->kv_sequences = (SparkKvPageCacheSequence *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_sequences));
	state->kv_hash_bucket_heads = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_hash_bucket_heads));
	state->kv_entry_indices_by_logical_page = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_entry_indices_by_logical_page));
	state->kv_page_staging = (uint8_t *)malloc((size_t)payload_bytes);
	state->kv_lane_logical_pages = (uint32_t *)calloc((size_t)lane_page_entries,sizeof(*state->kv_lane_logical_pages));
	state->kv_lane_transactions = (SparkKvLaneTransaction *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_lane_transactions));
	if ( cudaHostAlloc((void **)&state->kv_lane_physical_pages,lane_page_entries * sizeof(uint32_t),cudaHostAllocPortable) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->kv_blocks == 0 || state->kv_resident_slot_logical_block_indices == 0 || state->kv_entries == 0 || state->kv_sequences == 0 || state->kv_hash_bucket_heads == 0 || state->kv_entry_indices_by_logical_page == 0 || state->kv_page_staging == 0 || state->kv_lane_logical_pages == 0 || state->kv_lane_transactions == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->kv_transactions.cache = &state->kv_page_cache;
	state->kv_transactions.lanes = state->kv_lane_transactions;
	state->kv_transactions.logical_pages = state->kv_lane_logical_pages;
	state->kv_transactions.physical_pages = state->kv_lane_physical_pages;
	state->kv_transactions.page_capacity = state->pages_per_sequence;

	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_KV_MODEL_TABLE_ABI_VERSION;
	table.descriptor_bytes = SPARK_KV_MODEL_TABLE_BYTES;
	SparkGlm5NextKvFillCapacityRequest(&table.capacity_request);
	table.capacity_request.layer_count = state->kv_layer_count;
	table.capacity_request.index_key_layer_count = state->index_layer_count;
	table.capacity_request.index_key_dimension = SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION;
	table.capacity_request.index_key_bytes_per_scalar = 2u;

	table.arena_configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	table.arena_configuration.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	table.arena_configuration.logical_block_count = state->page_count;
	table.arena_configuration.block_token_count = SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT;
	table.arena_configuration.resident_block_capacity = state->physical_page_count;
	table.arena_configuration.layer_count = state->kv_layer_count;
	table.arena_configuration.kv_head_count = SPARK_GLM5_NEXT_KV_ARENA_KV_HEAD_COUNT;
	table.arena_configuration.head_dim = arena_head_dim;
	table.arena_configuration.bytes_per_scalar = SPARK_GLM5_NEXT_KV_BYTES_PER_SCALAR;
	table.arena_configuration.key_device_base = state->kv_cache;
	table.arena_configuration.value_device_base = state->index_cache;
	table.arena_configuration.value_block_stride_bytes = index_block_bytes;
	table.arena_configuration.blocks = state->kv_blocks;
	table.arena_configuration.resident_slot_logical_block_indices = state->kv_resident_slot_logical_block_indices;
	table.arena_configuration.evict_function = SparkKvPageStoreWriteback;
	table.arena_configuration.evict_context = &state->kv_page_store;

	table.page_store_config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	table.page_store_config.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	table.page_store_config.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS;
	table.page_store_config.logical_page_capacity = state->page_count;
	table.page_store_config.transfer_capacity = state->page_count < 2u ? state->page_count : 2u;
	table.page_store_config.page_bytes = payload_bytes;
	if ( state->kv_backing_directory != 0 && state->kv_backing_directory[0] != '\0' )
		table.page_store_config.backing_path = state->kv_backing_directory;
	else
	{
		(void)snprintf(state->kv_backing_default,sizeof(state->kv_backing_default),
			"/tmp/sparkpipe_glm5_next_kv_%s",state->model_revision);
		mkdir(state->kv_backing_default,0700);
		table.page_store_config.backing_path = state->kv_backing_default;
	}
	table.page_store_config.maximum_backing_bytes = state->page_count * payload_bytes;
	table.page_store_config.staging_address = state->kv_page_staging;
	table.page_store_config.staging_bytes = payload_bytes;
	table.page_store_config.copy_function = SparkGlm5NextPageCopy;
	table.page_store_config.copy_context = state;

	table.sequence_capacity = state->resident_sequence_capacity;
	table.entry_capacity = state->page_count;
	table.hash_bucket_count = state->page_count;
	table.entries = state->kv_entries;
	table.sequences = state->kv_sequences;
	table.hash_bucket_heads = state->kv_hash_bucket_heads;
	table.entry_indices_by_logical_page = state->kv_entry_indices_by_logical_page;
	table.model_id = "glm5_next";
	table.model_revision = state->model_revision;
	if ( state->kv_shard != 0u )
	{
		(void)snprintf(state->kv_layout_fingerprint,sizeof(state->kv_layout_fingerprint),"kv-bf16-index-packed-layer-major-gather-v1-shard%ur%u",state->tp_degree,state->tp_rank);
		table.cache_layout_fingerprint = state->kv_layout_fingerprint;
	}
	else
		table.cache_layout_fingerprint = "kv-bf16-index-packed-layer-major-gather-v1";

	status = SparkKvBackendInitialize(&table,&state->kv_arena,&state->kv_page_cache,&state->kv_page_store);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( state->kv_arena.key_block_stride_bytes != block_bytes ||
		state->kv_arena.value_block_stride_bytes != index_block_bytes ||
		state->kv_arena.logical_block_count != state->page_count ||
		state->kv_arena.resident_block_capacity != state->physical_page_count ||
		state->kv_layer_stride_bytes == 0u ||
		block_bytes != ( state->kv_layer_stride_bytes /
				(uint64_t)state->physical_page_count ) *
			(uint64_t)state->kv_layer_count ||
		(uint64_t)state->physical_page_count * block_bytes !=
			state->kv_layer_stride_bytes * (uint64_t)state->kv_layer_count )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return(SparkGlm5NextRecurrentInitialize(state,table.page_store_config.backing_path));
}

static SparkStatus SparkGlm5NextAllocateCaches(SparkGlm5NextModuleState *state)
{
	uint64_t main_page_bytes,index_page_bytes;
	uint64_t main_total,index_total;
	uint32_t local;
	SparkStatus status;
	uint64_t kda_window_stride;
	uint64_t kda_total,window_total;
	state->index_layer_count = 0u;
	state->kv_layer_count = 0u;
	state->kda_layer_count = 0u;
	for (local=0u; local<state->layer_count; local++)
	{
		uint32_t layer = state->first_layer_index + local;
		state->index_ordinal_by_local_layer[local] = SPARK_GLM5_NEXT_NO_INDEX_ORDINAL;
		state->kv_ordinal_by_local_layer[local] = SPARK_GLM5_NEXT_NO_INDEX_ORDINAL;
		state->kda_ordinal_by_local_layer[local] = SPARK_GLM5_NEXT_NO_INDEX_ORDINAL;
		if ( layer < SPARK_GLM5_NEXT_MODEL_LAYER_COUNT )
		{
			if ( SparkGlm5NextStagePackLayerIsDsa(layer) != 0u )
			{
				state->kv_ordinal_by_local_layer[local] = state->kv_layer_count++;
				state->index_ordinal_by_local_layer[local] = state->index_layer_count++;
			}
			else if ( SparkGlm5NextStagePackLayerIsKda(layer) != 0u )
				state->kda_ordinal_by_local_layer[local] = state->kda_layer_count++;
		}
	}
	status = SparkGlm5NextBuildPageTable(state);
	main_page_bytes = (uint64_t)64u * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES;
	index_page_bytes = (uint64_t)64u *
		SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u;
	if ( state->kv_shard != 0u )
	{
		main_page_bytes = SparkKvShardPageBytes(SparkGlm5NextKvShardLatent(state->tp_rank,state->tp_degree),SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS,SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES);
		index_page_bytes = SparkKvShardPageBytes(SparkGlm5NextKvShardIndex(state->tp_rank,state->tp_degree),SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS,SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u);
	}
	// The model constant includes all three windows; each pool owns one rank-local window.
	kda_window_stride = (uint64_t)state->resident_sequence_capacity *
		(SPARK_GLM5_NEXT_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER / (3u * state->tp_degree));
	state->kv_layer_stride_bytes = (uint64_t)state->physical_page_count * main_page_bytes;
	state->index_layer_stride_bytes = (uint64_t)state->physical_page_count * index_page_bytes;
	state->kda_state_layer_stride_bytes = (uint64_t)state->resident_sequence_capacity *
		(SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER / state->tp_degree);
	state->kda_window_layer_stride_bytes = kda_window_stride;
	if ( status != SPARK_STATUS_OK || state->kv_layer_stride_bytes == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_CAPACITY_EXCEEDED : status);
	main_total = state->kv_layer_stride_bytes * (uint64_t)state->kv_layer_count;
	status = SparkStageModuleDeviceAllocate(&state->ledger,main_total,(void **)&state->kv_cache);
	if ( status == SPARK_STATUS_OK && state->index_layer_count != 0u )
	{
		if ( state->index_layer_stride_bytes > UINT64_MAX / state->index_layer_count )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		index_total = state->index_layer_stride_bytes * state->index_layer_count;
		status = SparkStageModuleDeviceAllocate(&state->ledger,index_total,(void **)&state->index_cache);
	}
	if ( status == SPARK_STATUS_OK )
		fprintf(stderr,"GLM-KV-BYTES rank=%u tp=%u shard=%u physical_pages=%u latent_bytes=%llu index_bytes=%llu replicated_latent_bytes=%llu replicated_index_bytes=%llu\n",
			state->tp_rank,state->tp_degree,state->kv_shard,state->physical_page_count,
			(unsigned long long)main_total,(unsigned long long)(state->index_layer_stride_bytes * state->index_layer_count),
			(unsigned long long)((uint64_t)state->physical_page_count * SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES * state->kv_layer_count),
			(unsigned long long)((uint64_t)state->physical_page_count * SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u * state->index_layer_count));
	kda_total = state->kda_state_layer_stride_bytes * (uint64_t)state->kda_layer_count;
	if ( status == SPARK_STATUS_OK && state->kda_layer_count != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,kda_total,(void **)&state->kda_state_pools);
	window_total = kda_window_stride * 3u * (uint64_t)state->kda_layer_count;
	if ( status == SPARK_STATUS_OK && state->kda_layer_count != 0u )
	{
		status = SparkStageModuleDeviceAllocate(&state->ledger,window_total,(void **)&state->kda_window_pools);
		if ( status == SPARK_STATUS_OK )
		{
			state->kda_q_window_pool = state->kda_window_pools;
			state->kda_k_window_pool = state->kda_q_window_pool + kda_window_stride * (uint64_t)state->kda_layer_count;
			state->kda_v_window_pool = state->kda_k_window_pool + kda_window_stride * (uint64_t)state->kda_layer_count;
		}
	}
	if ( status == SPARK_STATUS_OK && state->kda_layer_count != 0u )
	{
		uint32_t sequence;
		cudaError_t error;
		state->kda_state_index_host = (uint32_t *)malloc((size_t)state->resident_sequence_capacity * sizeof(uint32_t));
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)state->resident_sequence_capacity * sizeof(uint32_t),(void **)&state->kda_state_index_device);
		if ( status == SPARK_STATUS_OK && state->kda_state_index_host != 0 )
		{
			for (sequence=0u; sequence<state->resident_sequence_capacity; sequence++)
				state->kda_state_index_host[sequence] = sequence;
			error = cudaMemcpy(state->kda_state_index_device,state->kda_state_index_host,(size_t)state->resident_sequence_capacity * sizeof(uint32_t),cudaMemcpyHostToDevice);
			if ( error != cudaSuccess )
				status = SPARK_STATUS_INTERNAL_ERROR;
		}
		else if ( state->kda_state_index_host == 0 )
		{
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
		}
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextKvInitialize(state);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextTerminalFailure(
	SparkGlm5NextModuleState *state,SparkStatus status,const char *source)
{
	uint32_t expected = SPARK_STATUS_OK;
	if ( status == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	if ( atomic_compare_exchange_strong_explicit(&state->terminal_status,
	        &expected,(uint32_t)status,memory_order_acq_rel,memory_order_acquire) )
		fprintf(stderr,"GLM engine terminal status=%u source=%s; full engine restart required\n",
		    (unsigned)status,source);
	return((SparkStatus)atomic_load_explicit(&state->terminal_status,memory_order_acquire));
}

static SparkStatus SparkGlm5NextWeightdHealth(SparkGlm5NextModuleState *state)
{
	SparkStatus status = (SparkStatus)atomic_load_explicit(
		&state->terminal_status,memory_order_acquire);
	uint32_t lane_dead,lazy_dead;
	if ( status != SPARK_STATUS_OK )
		return(status);
	lane_dead = state->lane_client != 0 && SparkWeightdClientAlive(state->lane_client) == 0u;
	lazy_dead = state->lazy_pack != 0 && state->lazy_pack->client != 0 &&
		SparkWeightdClientAlive(state->lazy_pack->client) == 0u;
	if ( lane_dead != 0u || lazy_dead != 0u )
		return(SparkGlm5NextTerminalFailure(state,SPARK_STATUS_IO_ERROR,
			lane_dead != 0u ? (lazy_dead != 0u ? "weightd-lane-and-lazy" : "weightd-lane") : "weightd-lazy"));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextAdmissionPredicate(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGlm5NextModuleState *state = (SparkGlm5NextModuleState *)context;
	SparkStatus status;
	uint32_t lane,slot;
	if ( state == 0 || request == 0 || decision == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkGlm5NextWeightdHealth(state);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = state->control_generation != 0u &&
		state->control_generation != request->control_generation ?
		SPARK_STATUS_VALIDATION_FAILED :
		SparkKvLaneTransactionsAdmit(&state->kv_transactions,request);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"ADMIT9-MODULE request=%llu control_generation=%llu state_control=%llu reset_gen=%llu status=%u lanes=%u\n",
			(unsigned long long)request->request_id,
			(unsigned long long)request->control_generation,
			(unsigned long long)state->control_generation,
			(unsigned long long)state->reset_generation,
			(unsigned)status,(unsigned)request->cache_lane_count);
	if ( status == SPARK_STATUS_OK )
		state->control_generation = request->control_generation;
	if ( status == SPARK_STATUS_OK && (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
		for (lane=0u; lane<request->cache_lane_count; lane++)
		{
			slot = request->cache_lanes[lane].resident_sequence_slot;
			atomic_store_explicit(&state->lane_bound[slot],0u,memory_order_release);
			if ( state->mtp_lane_armed != 0 )
				state->mtp_lane_armed[slot] = 0u;
		}
	(void)pthread_mutex_unlock(&state->kv_mutex);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	decision->accepted = 1u;
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	decision->driver_dispatch_slot = (uint32_t)(request->request_id % state->pipeline_slot_count);
	decision->driver_dispatch_generation = request->control_generation;
	decision->driver_dispatch_cookie0 = request->transaction_id;
	decision->driver_dispatch_cookie1 = request->submission_id;
	return(SPARK_STATUS_OK);
}

#include "sparkpipe/family/module/spark_module_load_sequence_continuity.h"

#include "sparkpipe/family/module/spark_module_claimed_continuity_locked.h"

static uint32_t SparkGlm5NextFrameSteps(const SparkModelDriverFrame *frame)
{
	return(frame->tokens_per_sequence > 1u ? frame->tokens_per_sequence : 1u);
}

static SparkStatus SparkGlm5NextValidateChain(const SparkGlm5NextModuleState *state,const SparkModelDriverFrame *frame,const SparkGlm5NextResidentDecodeStageFrameContext *context)
{
	const SparkGlm5NextResidentDecodeStageBatchView *batch = context->batch;
	uint32_t steps = SparkGlm5NextFrameSteps(frame),index;
	uint64_t last;
	if ( frame->tokens_per_sequence > SPARK_MODEL_DRIVER_MAX_TOKENS_PER_SEQUENCE )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( steps == 1u )
		return(SPARK_STATUS_OK);
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u || state->owns_embedding == 0u || state->owns_final_head == 0u || context->state_capture != 0 )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	for (index=0u; index<frame->cache_lane_count; index++)
		if ( (frame->cache_lanes[index].flags & (SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX | SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH)) != 0u )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	for (index=0u; index<batch->row_count; index++)
	{
		last = batch->row_positions[index] + steps - 1u;
		if ( last >= state->max_sequence_positions || last / SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS != batch->row_positions[index] / SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextValidateFrameBuffers(
	const SparkGlm5NextModuleState *state,
	const SparkModelDriverFrame *frame,
	uint32_t row_count)
{
	const SparkModelDriverBuffer *buffer;
	if ( state->owns_final_head == 0u )
		return(frame->buffer_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->buffer_count != 1u || frame->buffers == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	buffer = &frame->buffers[0];
	if ( buffer->flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE || buffer->address == 0 || buffer->bytes < (uint64_t)row_count * SparkGlm5NextFrameSteps(frame) * sizeof(uint32_t) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->mtp_enabled != 0u && row_count == 1u &&
		(frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) == 0u &&
		buffer->bytes < (uint64_t)(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u) * sizeof(uint32_t) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextValidateStateCapture(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextResidentDecodeStageFrameContext *context)
{
	const SparkGlm5NextStateCapture *capture = context->state_capture;
	uint64_t page_bytes,lane_bytes,hidden_bytes;
	if ( capture == 0 )
		return(SPARK_STATUS_OK);
	if ( capture->abi_version != SPARK_GLM5_NEXT_STATE_CAPTURE_ABI_VERSION || capture->descriptor_bytes != sizeof(*capture) )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( state->mtp_enabled != 0u || state->owns_final_head == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( capture->lanes == 0 || capture->logical_pages == 0 || capture->physical_pages == 0 || capture->payload == 0 || capture->lane_capacity < context->batch->active_sequence_count || capture->pages_per_lane_capacity < state->pages_per_sequence )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	page_bytes = state->kv_arena.key_block_stride_bytes + state->kv_arena.value_block_stride_bytes;
	hidden_bytes = (uint64_t)SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t);
	if ( page_bytes == 0u || state->pages_per_sequence > (UINT64_MAX - state->recurrent_page_bytes - hidden_bytes) / page_bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	lane_bytes = state->pages_per_sequence * page_bytes + state->recurrent_page_bytes + hidden_bytes;
	if ( capture->payload_capacity / context->batch->active_sequence_count < lane_bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextValidateFrame(
	const SparkGlm5NextModuleState *state,
	const SparkModelDriverFrame *frame,
	const SparkGlm5NextResidentDecodeStageFrameContext **context_out)
{
	const SparkGlm5NextResidentDecodeStageFrameContext *context;
	const SparkGlm5NextResidentDecodeStageBatchView *batch;
	uint32_t expected_flags,prefill;
	uint64_t boundary_bytes,sideband_bytes;
	SparkStatus status;
	if ( state == 0 || frame == 0 || context_out == 0 || frame->user_context == 0 || frame->execution_stream != state->execution_stream || frame->completion_function == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkGlm5NextResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes != sizeof(*context) || context->reserved0 != 0u || (context->flags & ~SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS) != 0u || context->batch == 0 )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	batch = context->batch;
	if ( batch->abi_version != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION || batch->descriptor_bytes != sizeof(*batch) || batch->row_count == 0u || batch->row_count > state->execution_row_capacity || batch->active_sequence_count == 0u || batch->active_sequence_count > state->resident_sequence_capacity || batch->row_resident_slots == 0 || batch->row_positions == 0 || batch->row_sequence_ids == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( prefill == 0u && batch->row_count != batch->active_sequence_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->active_slot_count != batch->active_sequence_count || frame->new_token_count != batch->row_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (state->owns_embedding != 0u && batch->token_ids == 0) || (state->owns_final_head != 0u && batch->row_sampling == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	expected_flags = prefill != 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
	expected_flags |= state->owns_embedding == 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT : 0u;
	expected_flags |= state->owns_final_head == 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT : 0u;
	expected_flags |= SparkGlm5NextResidentDecodeStageRequiresSidebandInput(state->stage_index) != 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_INPUT : 0u;
	expected_flags |= SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(state->stage_index) != 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT : 0u;
	if ( context->flags != expected_flags )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	boundary_bytes = (uint64_t)batch->row_count * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES;
	sideband_bytes = (uint64_t)batch->row_count * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW;
	if ( (state->owns_embedding == 0u && (context->hidden_input_bf16 == 0 || context->hidden_input_bytes < boundary_bytes)) || (state->owns_embedding != 0u && (context->hidden_input_bf16 != 0 || context->hidden_input_bytes != 0u)) || (state->owns_final_head == 0u && (context->hidden_output_bf16 == 0 || context->hidden_output_bytes < boundary_bytes)) || (state->owns_final_head != 0u && (context->hidden_output_bf16 != 0 || context->hidden_output_bytes != 0u)) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( (SparkGlm5NextResidentDecodeStageRequiresSidebandInput(state->stage_index) != 0u && (context->sideband_input == 0 || context->sideband_input_bytes < sideband_bytes)) || (SparkGlm5NextResidentDecodeStageRequiresSidebandInput(state->stage_index) == 0u && (context->sideband_input != 0 || context->sideband_input_bytes != 0u)) || (SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(state->stage_index) != 0u && (context->sideband_output == 0 || context->sideband_output_bytes < sideband_bytes)) || (SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(state->stage_index) == 0u && (context->sideband_output != 0 || context->sideband_output_bytes != 0u)) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkGlm5NextValidateRoundMajor(state,batch);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextValidateFrameBuffers(state,frame,batch->row_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextValidateStateCapture(state,context);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextValidateChain(state,frame,context);
	*context_out = status == SPARK_STATUS_OK ? context : 0;
	SPARK_RETURN(status);
}

#define SPARK_GLM5_NEXT_TP_COLLECTIVE_CREDITS_PER_SLOT 2u
#define SPARK_GLM5_NEXT_TP_CHAIN_OPERATIONS ((2u * SPARK_GLM5_NEXT_MODEL_LAYER_COUNT + 16u) * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT)
#define SPARK_GLM5_NEXT_TP_COLLECTIVE_HC_PORT_STRIDE 512u
#define SPARK_GLM5_NEXT_TP_COLLECTIVE_D2A_MAX_PAYLOAD_BYTES 65536u

typedef enum SparkGlm5NextChainStage
{
	SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN = 0,
	SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION,
	SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION,
	SPARK_GLM5_NEXT_CHAIN_STAGE_MLP,
	SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_MLP,
	SPARK_GLM5_NEXT_CHAIN_STAGE_HEAD,
	SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_HEAD,
	SPARK_GLM5_NEXT_CHAIN_STAGE_FINISH,
	SPARK_GLM5_NEXT_CHAIN_STAGE_GATHER_INDEX,
	SPARK_GLM5_NEXT_CHAIN_STAGE_SHARD_QUERY,
	SPARK_GLM5_NEXT_CHAIN_STAGE_SHARD_EXCHANGE
} SparkGlm5NextChainStage;

typedef struct SparkGlm5NextTpChain
{
	SparkGlm5NextModuleState *state;
	uint64_t created_ns;
	uint64_t last_advance_ns;
	uint64_t last_heartbeat_stage;
	SparkGlm5NextExecutionSlot *slot;
	uint32_t slot_index;
	SparkModelDriverFrame *frame;
	const SparkGlm5NextResidentDecodeStageFrameContext *context;
	const SparkGlm5NextResidentDecodeStageBatchView *batch;
	SparkGlm5NextCudaWave wave;
	uint32_t first_row;
	uint32_t wave_rows;
	uint32_t stage;
	uint32_t next_layer;
	uint32_t active;

	uint32_t spec_verify;
	uint32_t steps;
	uint32_t step;
	uint32_t verify_budget;
	uint32_t verify_produced;
	uint32_t verify_lane;
	uint32_t verify_draft_count;
	uint32_t verify_rounds;
	uint32_t verify_accepted;
	uint32_t verify_plain;
	uint64_t verify_sequence_id;
	uint64_t verify_position;
	uint32_t verify_draft[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX];
	uint32_t verify_tokens[SPARK_MODEL_DRIVER_MAX_TOKENS_PER_SEQUENCE];
	uint32_t tp_op_index;
	uint32_t tp_hc_op_index;
	uint64_t expert_lease;
	uint32_t expert_lease_begun;
	uint32_t expert_lease_recorded;
	SparkStepVerdict step_verdict;
	SparkStepReplay ws_replay;
	uint32_t ws_force_eager;
	SparkStatus retained_status;
} SparkGlm5NextTpChain;

static void SparkGlm5NextTpChainAdvance(void *chain_context,SparkStatus status);
static void SparkGlm5NextTpChainFail(SparkGlm5NextTpChain *chain,SparkStatus status);
static void CUDART_CB SparkGlm5NextCompleteAsync(void *context);
static SparkStatus SparkGlm5NextMtpTap(SparkGlm5NextModuleState *state,SparkGlm5NextExecutionSlot *slot,uint32_t row,uint32_t lane,uint64_t sequence_id,uint64_t next_position);
static SparkStatus SparkGlm5NextMtpTapFrame(SparkGlm5NextTpChain *chain);
static void SparkGlm5NextTapEmit(SparkGlm5NextTpChain *chain,uint32_t rows,uint32_t flags);
static void CUDART_CB SparkGlm5NextMtpResolveHost(void *context);
static SparkStatus SparkGlm5NextEnqueueAsyncCompletion(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot,
	uint32_t slot_index);

static SparkStatus SparkGlm5NextBuildWave(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state;
	SparkGlm5NextExecutionSlot *slot;
	const SparkGlm5NextResidentDecodeStageFrameContext *context;
	SparkGlm5NextCudaWave *wave;
	uint32_t row,maximum_context;
	state = chain->state;
	slot = chain->slot;
	context = chain->context;
	wave = &chain->wave;
	maximum_context = 0u;
	for (row=0u; row<chain->wave_rows; row++)
		if ( slot->host_positions[chain->first_row + row] + 1u > maximum_context )
			maximum_context = slot->host_positions[chain->first_row + row] + 1u;
	memset(wave,0,sizeof(*wave));
	wave->stage_index = state->stage_index;
	wave->first_layer_index = state->first_layer_index;
	wave->layer_count = state->layer_count;
	wave->tp_degree = state->tp_degree;
	wave->tp_rank = state->tp_rank;
	wave->index_cp_degree = state->index_cp != 0u ? state->tp_degree : 1u;
	wave->kv_shard = state->kv_shard;
	wave->row_count = chain->wave_rows;
	wave->commit = chain->spec_verify != 0u ? 0u : 1u;
	wave->mtp_verify = chain->spec_verify;
	wave->mtp_draft_depth = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH;
	wave->mtp_layer_weights = state->pack_has_mtp != 0u ? &state->mtp_layer : 0;
	wave->mtp_eh_proj_bf16 = state->mtp_eh_proj_bf16;
	wave->mtp_enorm_bf16 = state->mtp_enorm_bf16;
	wave->mtp_hnorm_bf16 = state->mtp_hnorm_bf16;
	wave->mtp_shared_norm_bf16 = state->mtp_shared_norm_bf16;
	wave->kda_replay_layer_bytes = state->kda_replay_layer_bytes;
	wave->maximum_context = maximum_context;
	wave->resident_sequence_capacity = state->resident_sequence_capacity;
	wave->max_sequence_positions = state->max_sequence_positions;
	wave->execution_row_capacity = state->execution_row_capacity;
	wave->pages_per_sequence = state->pages_per_sequence;
	wave->physical_page_count = state->physical_page_count;
	wave->owns_embedding = state->owns_embedding;
	wave->owns_final_head = state->owns_final_head;
	wave->sideband_input = SparkGlm5NextResidentDecodeStageRequiresSidebandInput(state->stage_index);
	wave->sideband_output = SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(state->stage_index);
	wave->boundary_row_offset = chain->first_row;
	wave->sideband_row_offset = chain->first_row;
	wave->host_token_ids = state->owns_embedding != 0u ? slot->host_token_ids + chain->first_row : 0;
	wave->host_resident_slots = slot->host_resident_slots + chain->first_row;
	wave->host_positions = slot->host_positions + chain->first_row;
	wave->host_row_sampling = state->owns_final_head != 0u ? slot->host_row_sampling + chain->first_row : 0;
	wave->sampled = state->owns_final_head != 0u ? slot->sampled : 0u;
	wave->hidden_input_bf16 = context->hidden_input_bf16;
	wave->hidden_output_bf16 = context->hidden_output_bf16;
	wave->sideband_input_u32 = context->sideband_input;
	wave->sideband_output_u32 = context->sideband_output;
	wave->host_output_token_ids = state->owns_final_head != 0u ? slot->host_output_token_ids + chain->first_row : 0;
	wave->embedding_bf16 = state->embedding_bf16;
	wave->final_norm_bf16 = state->final_norm_bf16;
	wave->lm_head_bf16 = state->lm_head_bf16;
	wave->head_certified_fp8_payload = state->head_certified_fp8_payload;
	wave->head_certified_fp8_scale_f32 = state->head_certified_fp8_scale_f32;
	wave->head_certified_fp8_norm_f32 = state->head_certified_fp8_norm_f32;
	wave->layers = state->layers;
	wave->lazy_experts = state->lazy_pack != 0 ? 1u : 0u;
	wave->slot = slot;
	wave->kv_cache = state->kv_cache;
	wave->kv_layer_stride_bytes = state->kv_layer_stride_bytes;
	wave->index_cache = state->index_cache;
	wave->index_layer_stride_bytes = state->index_layer_stride_bytes;
	wave->index_ordinal_by_local_layer = state->index_ordinal_by_local_layer;
	wave->kda_ordinal_by_local_layer = state->kda_ordinal_by_local_layer;
	wave->kda_state_pools = state->kda_state_pools;
	wave->kda_state_layer_stride_bytes = state->kda_state_layer_stride_bytes;
	wave->kda_q_window_pool = state->kda_q_window_pool;
	wave->kda_k_window_pool = state->kda_k_window_pool;
	wave->kda_v_window_pool = state->kda_v_window_pool;
	wave->kda_window_layer_stride_bytes = state->kda_window_layer_stride_bytes;
	wave->kda_state_index = state->kda_state_index_device;
	wave->kda_layer_count = state->kda_layer_count;
	wave->page_table = state->page_table;
	wave->multiprocessor_count = state->multiprocessor_count;
	wave->decode_split_context_threshold = state->decode_split_context_threshold;
	wave->attention_split_partials_f32 = slot->attention_split_partials_f32;
	wave->attention_split_partial_blocks = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(
		state->execution_row_capacity,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT / state->tp_degree);
	{
		uint32_t lane,ordinals[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT],cursor[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
		uint32_t lanes = chain->batch->active_sequence_count;
		SparkRowLayoutDirectLaneContext map;
		SparkStatus status = SparkRowLayoutDirectLaneMapInitialize(&map,ordinals,state->resident_sequence_capacity,wave->host_resident_slots,lanes);
		if ( status == SPARK_STATUS_OK )
			status = SparkRowLayoutGroupRows(chain->wave_rows,lanes,wave->host_resident_slots,SparkRowLayoutDirectLaneOrdinal,&map,slot->host_run_begin,slot->host_run_row_indices,cursor);
		if ( status != SPARK_STATUS_OK )
			return(status);
		for (lane=0u; lane<lanes; lane++)
			slot->host_run_state_index[lane] = wave->host_resident_slots[lane];
		wave->run_count = lanes;
		wave->sequence_row_begin = slot->run_begin;
		wave->sequence_row_indices = slot->run_row_indices;
		wave->run_state_index = slot->run_state_index;
		wave->host_sequence_row_begin = slot->host_run_begin;
		wave->host_sequence_row_indices = slot->host_run_row_indices;
		wave->host_run_state_index = slot->host_run_state_index;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextRequestedMeshLane(uint32_t *lane)
{
	const char *text = getenv("SPARK_WEIGHTD_LANE");
	if ( text == 0 )
	{
		*lane = SPARK_WEIGHTD_LANE_NONE;
		return(SPARK_STATUS_OK);
	}
	if ( text[0] == '\0' )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_GLM5_NEXT_MODULE_TAG,
	    "SPARK_WEIGHTD_LANE",0u,SPARK_WEIGHTD_MESH_MAX_LANES - 1u,0u,lane));
}

#include "sparkpipe/family/module/spark_module_combine.h"

static SparkStatus SparkGlm5NextModuleInitializeTpCollective(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextResidentDecodeStageNodeContext *context)
{
	SparkTpDeviceCollectiveConfig configuration,configuration_hc;
	uint32_t probe_connect_timeout_milli,probe_operation_timeout_milli;
	SparkStatus status;
	if ( state == 0 || context == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
		return(SPARK_STATUS_OK);
	if ( state->lane_client == 0 )
	{
		const char *socket = getenv("SPARK_WEIGHTD_SOCKET");
		uint32_t requested_lane;
		SparkWeightdMeshTopology topology;
		status = SparkTpDeviceCollectiveMeshTopology(state->tp_rank,state->tp_degree,&topology);
		if (status != SPARK_STATUS_OK) SPARK_RETURN(status);
		status = SparkGlm5NextRequestedMeshLane(&requested_lane);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( socket == 0 || socket[0] == '\0' )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		if ( SparkWeightdClientConnect(socket,&state->lane_client,0) != SPARK_STATUS_OK )
		{
			state->lane_client = 0;
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		status = SparkWeightdClientLaneAcquire(state->lane_client,requested_lane,&topology,&state->tp_lane,
			(uint64_t)context->tp_connect_timeout_milli * 1000000ull);
		if ( status != SPARK_STATUS_OK )
		{
			(void)SparkWeightdClientClose(state->lane_client);
			state->lane_client = 0;
			fprintf(stderr,"GLM mesh lane acquire failed requested=%u capacity=%u status=%d\n",
			    requested_lane,SPARK_WEIGHTD_MESH_MAX_LANES,(int32_t)status);
			SPARK_RETURN(status);
		}
		fprintf(stderr,"GLM mesh lane mode=%s requested=%u resolved=%u capacity=%u rank=%u\n",
		    requested_lane == SPARK_WEIGHTD_LANE_NONE ? "automatic" : "explicit",
		    requested_lane,state->tp_lane,SPARK_WEIGHTD_MESH_MAX_LANES,state->tp_rank);
	}
	probe_connect_timeout_milli = context->tp_connect_timeout_milli;
	if ( SparkGlm5NextProbeEnabled() )
	{
		if ( probe_connect_timeout_milli >
			UINT32_MAX / SPARK_GLM5_NEXT_PROBE_CONNECT_TIMEOUT_SCALE )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		probe_connect_timeout_milli *= SPARK_GLM5_NEXT_PROBE_CONNECT_TIMEOUT_SCALE;
	}
	probe_operation_timeout_milli = context->tp_operation_timeout_milli;
	if ( SparkGlm5NextProbeEnabled() )
	{
		if ( probe_operation_timeout_milli >
			UINT32_MAX / SPARK_GLM5_NEXT_PROBE_OPERATION_TIMEOUT_SCALE )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		probe_operation_timeout_milli *= SPARK_GLM5_NEXT_PROBE_OPERATION_TIMEOUT_SCALE;
	}
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = context->tp_collective_backend_kind;
	configuration.tp_degree = state->tp_degree;
	configuration.tp_rank = state->tp_rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = state->pipeline_slot_count * SPARK_GLM5_NEXT_TP_COLLECTIVE_CREDITS_PER_SLOT;
	configuration.local_hidden_dimension = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	configuration.max_active_sequence_count = state->execution_row_capacity;
	configuration.connect_timeout_milli = probe_connect_timeout_milli;
	configuration.operation_timeout_milli = probe_operation_timeout_milli;
	configuration.control_port_base = context->tp_collective_control_port_base +
		2u * state->tp_lane;
	configuration.collective_identifier = 2u * state->tp_lane;
	configuration.mesh_lane_client = state->lane_client;
	configuration.backend_module_path = context->tp_collective_backend_module_path;
	configuration.registration_cuda_stream = state->execution_stream;
	status = SparkTpDeviceCollectiveApplyTopology(&context->tp_collective_topology,&configuration);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( configuration.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		configuration.algorithm_mask |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL;
		configuration.direct_all_to_all_max_payload_bytes =
			SPARK_GLM5_NEXT_TP_COLLECTIVE_D2A_MAX_PAYLOAD_BYTES;
	}
	memset(&configuration_hc,0,sizeof(configuration_hc));
	configuration_hc.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration_hc.backend_kind = context->tp_collective_backend_kind;
	configuration_hc.tp_degree = state->tp_degree;
	configuration_hc.tp_rank = state->tp_rank;
	configuration_hc.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration_hc.credit_count = configuration.credit_count;
	configuration_hc.local_hidden_dimension =
		SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * SPARK_GLM5_NEXT_MODEL_HC_MULT;
	configuration_hc.max_active_sequence_count = configuration.max_active_sequence_count;
	configuration_hc.connect_timeout_milli = probe_connect_timeout_milli;
	configuration_hc.operation_timeout_milli = probe_operation_timeout_milli;
	configuration_hc.control_port_base = context->tp_collective_control_port_base +
		SPARK_GLM5_NEXT_TP_COLLECTIVE_HC_PORT_STRIDE + 2u * state->tp_lane;
	configuration_hc.collective_identifier = 2u * state->tp_lane + 1u;
	configuration_hc.mesh_lane_client = state->lane_client;
	configuration_hc.mesh_band_index = 1u;
	configuration_hc.backend_module_path = context->tp_collective_backend_module_path;
	configuration_hc.registration_cuda_stream = state->execution_stream;
	status = SparkTpDeviceCollectiveApplyTopology(&context->tp_collective_topology,&configuration_hc);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memcpy(configuration_hc.session_ports,context->tp_collective_session_ports_hc,
		sizeof(configuration_hc.session_ports));
	if ( configuration.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		configuration.combine_bf16_function = SparkGlm5NextModuleCombineBf16;
		configuration.combine_fused_bf16_function = SparkGlm5NextModuleCombineFusedBf16;
		configuration.combine_f32_seed_function = SparkGlm5NextModuleCombineF32Seed;
		configuration.combine_f32_add_function = SparkGlm5NextModuleCombineF32Add;
		configuration.round_f32_function = SparkGlm5NextModuleRoundF32;
		configuration.combine_u64_max_function = SparkGlm5NextModuleCombineU64Max;
		configuration.combine_context = state;
		configuration_hc.combine_fused_bf16_function = SparkGlm5NextModuleCombineFusedBf16;
		configuration_hc.combine_f32_seed_function = SparkGlm5NextModuleCombineF32Seed;
		configuration_hc.combine_f32_add_function = SparkGlm5NextModuleCombineF32Add;
		configuration_hc.round_f32_function = SparkGlm5NextModuleRoundF32;
		configuration_hc.combine_bf16_function = SparkGlm5NextModuleCombineBf16;
		configuration_hc.combine_context = state;
	}
	if ( configuration.connect_timeout_milli == 0u || configuration.operation_timeout_milli == 0u || configuration.backend_kind != SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkTpDeviceCollectiveCreate(&configuration,&state->tp_device_collective);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state->tp_device_collective_initialized = 1u;
	if ( state->lazy_pack != 0 &&
	     state->lazy_pack->attached.mesh_send_buffer_addr != 0 )
		status = SparkTpDeviceCollectivePrepareReceiveBf16(
		    &state->tp_device_collective,
		    (void *)(uintptr_t)state->lazy_pack->attached.mesh_send_buffer_addr,
		    0u,0u,0u,0u);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkTpDeviceCollectiveCreate(&configuration_hc,&state->tp_device_collective_hc);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state->tp_device_collective_hc_initialized = 1u;
	if ( state->lazy_pack != 0 &&
	     state->lazy_pack->attached.mesh_send_buffer_addr != 0 )
		status = SparkTpDeviceCollectivePrepareReceiveBf16(
		    &state->tp_device_collective_hc,
		    (void *)(uintptr_t)state->lazy_pack->attached.mesh_send_buffer_addr,
		    0u,0u,0u,0u);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( state->kv_shard != 0u && (SparkTpDeviceCollectiveAllToAllSupported(&state->tp_device_collective) == 0u || SparkTpDeviceCollectiveAllToAllSupported(&state->tp_device_collective_hc) == 0u) )
	{
		fprintf(stderr,"GLM-KV-SHARD-REFUSED tp=%u rank=%u: the sharded KV exchange needs all-to-all on both collectives (hardware waits and a weightd that advertises slice routes)\n",state->tp_degree,state->tp_rank);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextModuleReduceHiddenWide(SparkGlm5NextTpChain *chain,
	void *device_bf16,uint32_t hc_wide);

static SparkStatus SparkGlm5NextModuleReduceHidden(SparkGlm5NextTpChain *chain,void *device_bf16)
{
	return(SparkGlm5NextModuleReduceHiddenWide(chain,device_bf16,1u));
}

static SparkStatus SparkGlm5NextModuleReduceAttentionOut(SparkGlm5NextTpChain *chain,void *device_bf16)
{
	return(SparkGlm5NextModuleReduceHiddenWide(chain,device_bf16,0u));
}

static void SparkGlm5NextModuleTpCompletion(
	void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkGlm5NextTpChain *chain;
	chain = (SparkGlm5NextTpChain *)context;
	if ( chain == 0 || chain->active == 0u || completion == 0 )
		return;
	SparkStatus status = SparkGlm5NextWeightdHealth(chain->state);
	if ( status == SPARK_STATUS_OK )
		status = completion->status;
	if ( status != SPARK_STATUS_OK )
		SparkGlm5NextTpChainFail(chain,status);
	else
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextChainOrdinal(SparkGlm5NextTpChain *chain,uint32_t hc_wide,uint32_t operation,uint64_t *ordinal)
{
	SparkGlm5NextModuleState *state;
	state = chain->state;
	if ( state->tp_device_collective.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL )
	{
		*ordinal = atomic_fetch_add_explicit(hc_wide != 0u ? &state->nccl_next_ordinal_hc : &state->nccl_next_ordinal,1u,memory_order_relaxed);
		return(SPARK_STATUS_OK);
	}
	return(SparkTpChainOrdinal(chain->frame->request_id,state->pipeline_slot_count,SPARK_GLM5_NEXT_TP_COLLECTIVE_CREDITS_PER_SLOT,SPARK_GLM5_NEXT_TP_CHAIN_OPERATIONS,operation,ordinal));
}

static SparkStatus SparkGlm5NextModuleReduceHiddenWide(SparkGlm5NextTpChain *chain,
	void *device_bf16,uint32_t hc_wide)
{
	uint32_t *op_index;
	SparkStatus ordinal_status;
	SparkGlm5NextModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	SparkTpDeviceCollective *collective;
	uint64_t ordinal;
	state = chain->state;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
	{
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_device_collective_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	ordinal_status = SparkGlm5NextWeightdHealth(state);
	if ( ordinal_status != SPARK_STATUS_OK )
		SPARK_RETURN(ordinal_status);
	if ( hc_wide != 0u )
	{
		if ( state->tp_device_collective_hc_initialized == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		collective = &state->tp_device_collective_hc;
		op_index = &chain->tp_hc_op_index;
	}
	else
	{
		collective = &state->tp_device_collective;
		op_index = &chain->tp_op_index;
	}
	ordinal_status = SparkGlm5NextChainOrdinal(chain,hc_wide,*op_index,&ordinal);
	if ( ordinal_status != SPARK_STATUS_OK )
		return(ordinal_status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = chain->wave_rows;
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = device_bf16;
	submission.full_device = device_bf16;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = SparkGlm5NextModuleTpCompletion;
	submission.completion_context = chain;
	{
		SparkStatus submit_status;
		*op_index += 1u;
		submit_status = SparkTpDeviceCollectiveEnqueue(collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
		if ( submit_status != SPARK_STATUS_OK )
		{
			*op_index -= 1u;
			fprintf(stderr,"G5N-DBG reduce submit -> %d (rows %u slot %u dev %p stream %p maxact %u)\n",
				(int)submit_status,(unsigned)chain->wave_rows,(unsigned)chain->slot_index,
				device_bf16,chain->slot->stream,
				(unsigned)state->tp_device_collective.max_active_sequence_count);
			SPARK_RETURN(submit_status);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextModuleGatherIndex(SparkGlm5NextTpChain *chain,uint32_t sequences,uint32_t chained)
{
	SparkGlm5NextModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t ordinal;
	SparkStatus status;
	state = chain->state;
	if ( state->tp_collective_disabled != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( state->tp_device_collective_initialized == 0u || sequences == 0u || sequences > state->execution_row_capacity )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkGlm5NextChainOrdinal(chain,0u,chain->tp_op_index,&ordinal);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = sequences;
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = chain->slot->index_local_scores_f32;
	submission.full_device = chain->slot->index_gathered_scores_f32;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = chained != 0u ? SparkGlm5NextModuleTpCompletion : 0;
	submission.completion_context = chained != 0u ? chain : 0;
	chain->tp_op_index += 1u;
	status = SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER);
	if ( status != SPARK_STATUS_OK )
		chain->tp_op_index -= 1u;
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextModuleKvShardExchange(SparkGlm5NextTpChain *chain,uint32_t exchange,uint32_t chained)
{
	SparkGlm5NextModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	SparkTpDeviceCollective *collective;
	uint32_t *op_index,wide;
	uint64_t ordinal;
	SparkStatus status;
	state = chain->state;
	if ( state->tp_collective_disabled != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	wide = exchange != 0u ? SparkGlm5NextKvShardPartialWide(chain->wave_rows,state->tp_degree,state->execution_row_capacity) : 0u;
	if ( state->kv_shard == 0u || state->tp_device_collective_initialized == 0u || (wide != 0u && state->tp_device_collective_hc_initialized == 0u) )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	collective = wide != 0u ? &state->tp_device_collective_hc : &state->tp_device_collective;
	op_index = wide != 0u ? &chain->tp_hc_op_index : &chain->tp_op_index;
	status = SparkGlm5NextChainOrdinal(chain,wide,*op_index,&ordinal);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = exchange != 0u ? SparkGlm5NextKvShardPartialSequences(chain->wave_rows,state->tp_degree,state->execution_row_capacity) : SparkGlm5NextKvShardQuerySequences(chain->wave_rows,state->tp_degree);
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = exchange != 0u ? (void *)chain->slot->kv_shard_partials_f32 : (void *)chain->slot->query_latent_bf16;
	submission.full_device = exchange != 0u ? (void *)chain->slot->kv_shard_partials_received_f32 : (void *)chain->slot->kv_shard_query_gathered_bf16;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = chained != 0u ? SparkGlm5NextModuleTpCompletion : 0;
	submission.completion_context = chained != 0u ? chain : 0;
	*op_index += 1u;
	status = SparkTpDeviceCollectiveEnqueue(collective,&submission,exchange != 0u ? SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL : SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER);
	if ( status != SPARK_STATUS_OK )
		*op_index -= 1u;
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextModuleReduceHeadMax(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t ordinal;
	SparkStatus status;
	state = chain->state;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
	{
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_device_collective_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkGlm5NextChainOrdinal(chain,0u,chain->tp_op_index,&ordinal);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = chain->wave_rows;
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = chain->slot->head_maxloc_u64;
	submission.full_device = chain->slot->head_maxloc_u64;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = SparkGlm5NextModuleTpCompletion;
	submission.completion_context = chain;
	chain->tp_op_index += 1u;
	status = SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64);
	if ( status != SPARK_STATUS_OK )
	{
		chain->tp_op_index -= 1u;
		SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextBuildMtpDraftWave(
	const SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot,
	SparkGlm5NextCudaWave *wave)
{
	memset(wave,0,sizeof(*wave));
	wave->tp_degree = state->tp_degree;
	wave->tp_rank = state->tp_rank;
	wave->owns_final_head = state->owns_final_head;
	wave->multiprocessor_count = state->multiprocessor_count;
	wave->embedding_bf16 = state->embedding_bf16;
	wave->lm_head_bf16 = state->lm_head_bf16;
	wave->mtp_layer_weights = &state->mtp_layer;
	wave->mtp_eh_proj_bf16 = state->mtp_eh_proj_bf16;
	wave->mtp_enorm_bf16 = state->mtp_enorm_bf16;
	wave->mtp_hnorm_bf16 = state->mtp_hnorm_bf16;
	wave->mtp_shared_norm_bf16 = state->mtp_shared_norm_bf16;
	wave->mtp_draft_depth = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH;
	wave->slot = slot;
}

static SparkStatus SparkGlm5NextMtpDriveDraft(
	SparkGlm5NextModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkGlm5NextResidentDecodeStageBatchView *batch,
	SparkGlm5NextExecutionSlot *slot,
	SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextAsyncCompletion *async;
	SparkGlm5NextCudaWave draft_wave;
	uint32_t lane,step;
	uint64_t position;
	if ( state->mtp_enabled == 0u || SparkGlm5NextFrameSteps(frame) != 1u ||
		(frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ||
		batch->row_count != 1u || batch->active_sequence_count != 1u ||
		batch->token_ids == 0 || state->owns_embedding == 0u || state->owns_final_head == 0u ||
		batch->row_sampling[0].inverse_temperature != 0.0f )
		return(SPARK_STATUS_OK);
	lane = batch->row_resident_slots[0];
	position = batch->row_positions[0];
	if ( lane >= state->resident_sequence_capacity ||
		state->mtp_lane_armed[lane] == 0u ||
		position + SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u > state->max_sequence_positions )
		return(SPARK_STATUS_OK);
	async = &state->completions[chain->slot_index];
	SparkGlm5NextBuildMtpDraftWave(state,slot,&draft_wave);
	if ( SparkGlm5NextLaunchCudaMtpDraft(&draft_wave,0,
		state->mtp_lane_hidden_bf16 + (uint64_t)lane * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
		batch->token_ids[0],async->mtp_draft_tokens) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for ( step = 1u; step <= SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH; ++step )
	{
		slot->host_token_ids[step] = async->mtp_draft_tokens[step - 1u];
		slot->host_positions[step] = (uint32_t)(position + step);
		slot->host_resident_slots[step] = lane;
	}
	chain->wave_rows = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u;
	chain->spec_verify = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextMtpStashHidden(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextTpChain *chain)
{
	cudaError_t error;
	uint32_t row,lane;
	for ( row = 0u; row < chain->wave_rows; ++row )
	{
		lane = chain->slot->host_resident_slots[chain->first_row + row];
		if ( row + 1u < chain->wave_rows &&
			chain->slot->host_resident_slots[chain->first_row + row + 1u] == lane )
			continue;
		error = cudaMemcpyAsync(
			state->mtp_lane_hidden_bf16 + (uint64_t)lane * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
			chain->slot->hc_mean_bf16 + (uint64_t)row * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
			(uint64_t)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),
			cudaMemcpyDeviceToDevice,(cudaStream_t)chain->slot->stream);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"mtp_stash"));
		state->mtp_lane_armed[lane] = 1u;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextVerifyObserveRows(SparkGlm5NextModuleState *state,const SparkGlm5NextResidentDecodeStageBatchView *batch)
{
	SparkStatus status;
	uint32_t row;
	if ( SparkGlm5NextVerifyDrafterUsesLookup(state->verify_drafter) == 0u || batch->token_ids == 0 )
		return(SPARK_STATUS_OK);
	for (row=0u; row<batch->row_count; row++)
	{
		status = SparkSpeculationLookupDraftObserve(&state->verify_lookup,batch->row_resident_slots[row],batch->row_sequence_ids[row],batch->row_positions[row],&batch->token_ids[row],1u);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm5NextVerifyPlanRound(SparkGlm5NextTpChain *chain,uint32_t anchor_token,SparkStatus *status_out)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	SparkSpeculationPolicyDraftRequest request;
	SparkSpeculationPolicyDraftResult result;
	SparkStatus status;
	uint32_t depth,row;
	*status_out = SPARK_STATUS_OK;
	depth = SparkGlm5NextVerifyDepth(chain->verify_budget,chain->verify_produced,state->verify_rows_max,(uint32_t)chain->verify_position,state->decode_split_context_threshold,state->max_sequence_positions);
	if ( depth == 0u )
		return(0u);
	if ( state->verify_depth_sequence[chain->verify_lane] != chain->verify_sequence_id )
	{
		state->verify_depth_sequence[chain->verify_lane] = chain->verify_sequence_id;
		state->verify_depth_cap[chain->verify_lane] = state->verify_rows_max - 1u;
	}
	if ( state->verify_drafter != SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP_LOOKUP && depth > state->verify_depth_cap[chain->verify_lane] )
		depth = state->verify_depth_cap[chain->verify_lane];
	memset(&request,0,sizeof(request));
	memset(&result,0,sizeof(result));
	request.abi_version = SPARK_SPECULATION_ABI_VERSION;
	request.descriptor_bytes = SPARK_SPECULATION_DRAFT_REQUEST_DESCRIPTOR_BYTES;
	request.requested_token_count = depth;
	request.active_sequence_index = chain->verify_lane;
	request.request_id = chain->frame->request_id;
	request.sequence_id = chain->verify_sequence_id;
	request.sequence_position = chain->verify_position;
	state->mtp_chain = chain;
	state->mtp_anchor_token = anchor_token;
	status = state->verify_draft_function(state->verify_draft_context,&request,&result);
	state->mtp_chain = 0;
	if ( status == SPARK_STATUS_NOT_FOUND )
		return(0u);
	if ( status == SPARK_STATUS_OK && result.token_count > depth )
		status = SPARK_STATUS_INTERNAL_ERROR;
	for (row=0u; status == SPARK_STATUS_OK && row<result.token_count; row++)
		if ( result.token_ids[row] >= SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT )
			status = SPARK_STATUS_VALIDATION_FAILED;
	*status_out = status;
	if ( status != SPARK_STATUS_OK || result.token_count == 0u )
		return(0u);
	slot->host_token_ids[0] = anchor_token;
	slot->host_positions[0] = (uint32_t)chain->verify_position;
	slot->host_resident_slots[0] = chain->verify_lane;
	for (row=0u; row<result.token_count; row++)
	{
		chain->verify_draft[row] = result.token_ids[row];
		slot->host_token_ids[row + 1u] = result.token_ids[row];
		slot->host_positions[row + 1u] = (uint32_t)chain->verify_position + row + 1u;
		slot->host_resident_slots[row + 1u] = chain->verify_lane;
	}
	chain->verify_draft_count = result.token_count;
	chain->wave_rows = result.token_count + 1u;
	chain->spec_verify = 1u;
	return(1u);
}

static void SparkGlm5NextVerifyNotePlain(SparkGlm5NextModuleState *state,uint32_t frame_class)
{
	uint64_t count;
	state->verify_plain_frames++;
	count = ++state->verify_frame_class[frame_class];
	if ( (count & (count - 1u)) == 0u )
		fprintf(stderr,"VERIFY-PLAIN-FRAME class=%u count=%llu | shape=%llu sampled=%llu cold=%llu no_draft=%llu\n",frame_class,(unsigned long long)count,
			(unsigned long long)state->verify_frame_class[SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_SHAPE],(unsigned long long)state->verify_frame_class[SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_SAMPLED],
			(unsigned long long)state->verify_frame_class[SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_COLD],(unsigned long long)state->verify_frame_class[SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_NO_DRAFT]);
}

static SparkStatus SparkGlm5NextVerifyDriveDraft(
	SparkGlm5NextModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkGlm5NextResidentDecodeStageBatchView *batch,
	SparkGlm5NextExecutionSlot *slot,
	SparkGlm5NextTpChain *chain)
{
	SparkStatus status;
	uint32_t shape,frame_class;
	if ( state->verify_drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_NONE )
		return(SPARK_STATUS_OK);
	status = SparkGlm5NextVerifyObserveRows(state,batch);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	shape = SparkGlm5NextFrameSteps(frame) >= 2u && (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) == 0u &&
		batch->row_count == 1u && batch->active_sequence_count == 1u && batch->token_ids != 0 && chain->spec_verify == 0u ? 1u : 0u;
	frame_class = SparkGlm5NextVerifyFrameClass(shape,shape != 0u && batch->row_sampling[0].inverse_temperature != 0.0f ? 1u : 0u,
		state->experts_warm,state->graph_path_enabled,slot->graph_disabled,(uint32_t)(slot->graph_failed_rows & UINT64_C(1)),slot->verify_captured);
	if ( frame_class == SPARK_GLM5_NEXT_VERIFY_FRAME_RANK_LOCAL )
	{
		fprintf(stderr,"VERIFY-RANK-LOCAL-INELIGIBLE request=%llu graph_path=%u graph_disabled=%u b1_capture_failed=%u; every rank must take the same verify or plain frame\n",
			(unsigned long long)frame->request_id,state->graph_path_enabled,slot->graph_disabled,(uint32_t)(slot->graph_failed_rows & UINT64_C(1)));
		return(SparkGlm5NextTerminalFailure(state,SPARK_STATUS_UNSUPPORTED,"verify-rank-local-graph"));
	}
	if ( frame_class != SPARK_GLM5_NEXT_VERIFY_FRAME_ELIGIBLE )
	{
		SparkGlm5NextVerifyNotePlain(state,frame_class);
		return(SPARK_STATUS_OK);
	}
	chain->verify_budget = SparkGlm5NextFrameSteps(frame);
	chain->verify_produced = 0u;
	chain->verify_lane = batch->row_resident_slots[0];
	chain->verify_sequence_id = batch->row_sequence_ids[0];
	chain->verify_position = batch->row_positions[0];
	if ( SparkGlm5NextVerifyPlanRound(chain,batch->token_ids[0],&status) == 0u )
	{
		chain->verify_budget = 0u;
		if ( status == SPARK_STATUS_OK )
			SparkGlm5NextVerifyNotePlain(state,SPARK_GLM5_NEXT_VERIFY_FRAME_PLAIN_NO_DRAFT);
		SPARK_RETURN(status);
	}
	chain->steps = 1u;
	state->completions[chain->slot_index].steps = 1u;
	state->verify_frames++;
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextVerifyObserveOutputs(SparkGlm5NextModuleState *state,SparkGlm5NextAsyncCompletion *async,const SparkGlm5NextExecutionSlot *slot)
{
	uint32_t burst;
	uint64_t next;
	if ( SparkGlm5NextVerifyDrafterUsesLookup(state->verify_drafter) == 0u || async->completion.status != SPARK_STATUS_OK ||
		async->prefill != 0u || async->lane_count != 1u || async->row_count != 1u || state->owns_final_head == 0u )
		return;
	burst = async->burst_token_count != 0u ? async->burst_token_count : 1u;
	next = async->lane_next_positions[0];
	if ( next < burst || next >= state->max_sequence_positions )
		return;
	if ( SparkSpeculationLookupDraftObserve(&state->verify_lookup,async->lane_indices[0],async->lane_sequence_ids[0],next - burst + 1u,slot->host_output_token_ids,burst) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"VERIFY-OBSERVE-FAILED lane=%u position=%llu tokens=%u\n",async->lane_indices[0],(unsigned long long)(next - burst + 1u),burst);
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
	}
}

static SparkStatus SparkGlm5NextLazyRelease(SparkGlm5NextTpChain *chain);
static void SparkGlm5NextReleaseDrafter(SparkGlm5NextModuleState *state);

static uint32_t SparkGlm5NextGraphPathState(const SparkGlm5NextModuleState *state)
{
	if ( state->graph_path_requested == 0u )
		return(SPARK_GLM5_NEXT_GRAPH_PATH_OFF);
	return(state->graph_path_enabled != 0u ? SPARK_GLM5_NEXT_GRAPH_PATH_ON : SPARK_GLM5_NEXT_GRAPH_PATH_DEGRADED);
}

static void SparkGlm5NextStampLaunch(SparkGlm5NextAsyncCompletion *async)
{
	if ( async->launch_ns == 0u )
		async->launch_ns = SparkGlm5NextNowNs();
}

static SparkStatus SparkGlm5NextStepVerdictApply(SparkGlm5NextTpChain *chain)
{
	volatile uint32_t *ring = chain->slot->miss_ring;
	uint32_t local_miss = ring != 0 ? ring[SPARK_STEP_MISS_FLAG] : 0u;
	SparkStepVerdict verdict = SparkStepVerdictClassify(chain->slot->host_output_token_ids + chain->first_row,chain->wave_rows,local_miss);
	chain->step_verdict = verdict;
	if ( verdict == SPARK_STEP_VERDICT_COMMIT )
		return(SPARK_STATUS_OK);
	chain->state->step_verdicts[verdict]++;
	fprintf(stderr,"GRAPH-STEP-VERDICT slot=%u step=%u rows=%u verdict=%s misses=%u total=%llu\n",chain->slot_index,chain->step,chain->wave_rows,SparkStepVerdictName(verdict),ring != 0 ? ring[SPARK_STEP_MISS_COUNT] : 0u,(unsigned long long)chain->state->step_verdicts[verdict]);
	if ( ring != 0 && chain->state->ws_enabled == 0u )
	{
		ring[SPARK_STEP_MISS_FLAG] = 0u;
		ring[SPARK_STEP_MISS_COUNT] = 0u;
	}
	return(verdict == SPARK_STEP_VERDICT_ROLLBACK_LOCAL || verdict == SPARK_STEP_VERDICT_ROLLBACK_REMOTE ? SPARK_STATUS_UNSUPPORTED : SPARK_STATUS_INTERNAL_ERROR);
}

static SparkStatus SparkGlm5NextSettleStep(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkStatus status = chain->expert_lease != 0u ? SparkGlm5NextLazyRelease(chain) : SPARK_STATUS_OK;
	if ( status == SPARK_STATUS_OK && SparkGlm5NextBoundedStreamSync(state,chain->slot->stream,UINT64_C(35000000000)) != 0 )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && state->tp_device_collective_initialized != 0u )
		status = SparkTpDeviceCollectiveVerifyDeferred(&state->tp_device_collective,state->execution_stream);
	if ( status == SPARK_STATUS_OK && state->tp_device_collective_hc_initialized != 0u )
		status = SparkTpDeviceCollectiveVerifyDeferred(&state->tp_device_collective_hc,state->execution_stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextStepVerdictApply(chain);
	return(status);
}

static void SparkGlm5NextFeedStep(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	uint32_t row,index;
	SparkGlm5NextTapEmit(chain,chain->wave_rows,SPARK_SPECULATION_TAP_FLAG_DECODE);
	for (row=0u; row<chain->wave_rows; row++)
	{
		index = chain->first_row + row;
		slot->host_chain_token_ids[row * chain->steps + chain->step] = slot->host_output_token_ids[index];
		slot->host_positions[index] += 1u;
		slot->host_token_ids[index] = slot->host_output_token_ids[index];
	}
	chain->ws_force_eager = 0u;
	(void)SparkStepReplayNext(&chain->ws_replay,SPARK_STEP_VERDICT_COMMIT);
}

static uint32_t SparkGlm5NextWsRetry(SparkGlm5NextTpChain *chain);

static void SparkGlm5NextNextStep(SparkGlm5NextTpChain *chain)
{
	SparkStatus status = SparkGlm5NextSettleStep(chain);
	if ( status != SPARK_STATUS_OK )
	{
		if ( SparkGlm5NextWsRetry(chain) == 0u )
			SparkGlm5NextTpChainFail(chain,status);
		return;
	}
	SparkGlm5NextFeedStep(chain);
	chain->step++;
	chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN;
	chain->next_layer = 0u;
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
}

static void SparkGlm5NextFinishChain(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextAsyncCompletion *async = &chain->state->completions[chain->slot_index];
	SparkStatus status;
	if ( chain->step + 1u < chain->steps )
	{
		SparkGlm5NextNextStep(chain);
		return;
	}
	if ( chain->state->ws_enabled != 0u )
	{
		status = SparkGlm5NextSettleStep(chain);
		if ( status != SPARK_STATUS_OK )
		{
			if ( SparkGlm5NextWsRetry(chain) == 0u )
				SparkGlm5NextTpChainFail(chain,status);
			return;
		}
		chain->ws_force_eager = 0u;
	}
	status = chain->expert_lease != 0u ? SparkGlm5NextLazyRelease(chain) : SPARK_STATUS_OK;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextMtpTapFrame(chain);
	if ( status == SPARK_STATUS_OK && chain->verify_budget == 0u && chain->state->tap_enabled != 0u && chain->state->ws_enabled == 0u &&
		SparkGlm5NextBoundedStreamSync(chain->state,chain->slot->stream,UINT64_C(35000000000)) != 0 )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && chain->verify_budget == 0u )
		SparkGlm5NextTapEmit(chain,chain->wave_rows,SPARK_SPECULATION_TAP_FLAG_DECODE);
	async->finish_ns = SparkGlm5NextNowNs();
	async->graph_path = SparkGlm5NextGraphPathState(chain->state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextEnqueueAsyncCompletion(chain->state,chain->slot,chain->slot_index);
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm5NextTpChainFail(chain,status);
		return;
	}
	chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_FINISH;
	chain->active = 0u;
	free(chain);
}

static void SparkGlm5NextMtpResolveOnWorker(void *context)
{
	SparkGlm5NextTpChain *chain;
	SparkGlm5NextModuleState *state;
	SparkGlm5NextExecutionSlot *slot;
	SparkGlm5NextAsyncCompletion *async;
	SparkSpeculationPolicyVerifyResult result;
	SparkStatus status;
	cudaError_t error;
	uint32_t lane;
	int32_t launch;
	chain = (SparkGlm5NextTpChain *)context;
	if ( chain == 0 || chain->active == 0u )
		return;
	state = chain->state;
	pthread_mutex_lock(&state->completion_queue_lock);
	SparkGlm5NextDrainParkedCompletions(state);
	pthread_mutex_unlock(&state->completion_queue_lock);
	slot = chain->slot;
	async = &state->completions[chain->slot_index];
	lane = async->lane_indices[0];
	status = SparkSpeculationPolicyResolveVerifierTokens(
		async->mtp_draft_tokens,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH,
		slot->host_output_token_ids,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u,
		SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,&result);
	if ( status == SPARK_STATUS_OK &&
		result.committed_token_count != result.accepted_draft_token_count + 1u )
		status = SPARK_STATUS_INTERNAL_ERROR;
	error = cudaSuccess;
	launch = 0;
	if ( status == SPARK_STATUS_OK )
	{
		async->completion.accepted_token_count = result.committed_token_count;
		async->completion.tokens_per_sequence = result.committed_token_count;
		async->burst_token_count = result.committed_token_count;
		async->lane_next_positions[0] += result.accepted_draft_token_count;
		async->cache_extra_tokens = result.accepted_draft_token_count;
		error = cudaMemcpyAsync(
			state->mtp_lane_hidden_bf16 + (uint64_t)lane * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
			slot->hc_mean_bf16 + (uint64_t)result.accepted_draft_token_count * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
			(uint64_t)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),
			cudaMemcpyDeviceToDevice,(cudaStream_t)slot->stream);
		if ( error == cudaSuccess )
			launch = SparkGlm5NextPrepareCudaReplayFold(&chain->wave,chain->wave.row_count);
		if ( error == cudaSuccess && launch == 0 )
			launch = SparkGlm5NextLaunchCudaReplayFold(&chain->wave,result.committed_token_count);
	}
	if ( status != SPARK_STATUS_OK || error != cudaSuccess || launch != 0 )
		SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
	else
		SparkGlm5NextFinishChain(chain);
}

static void CUDART_CB SparkGlm5NextMtpResolveHost(void *context)
{
	SparkGlm5NextTpChain *chain = context;
	if ( chain != 0 && chain->active != 0u )
		SparkGlm5NextScheduleCompletionWork(chain->state,SparkGlm5NextMtpResolveOnWorker,chain);
}

static void SparkGlm5NextLazyRetryRetained(void *context);

static void SparkGlm5NextScheduleRetainedRetry(SparkGlm5NextModuleState *state)
{
	if ( state->lazy_pack == 0 || state->lazy_pack->worker == 0 )
		return;
	(void)SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkGlm5NextLazyRetryRetained,state);
}

static void SparkGlm5NextTpChainFail(SparkGlm5NextTpChain *chain,SparkStatus status)
{
	SparkGlm5NextModuleState *state;
	SparkGlm5NextAsyncCompletion *async;
	if ( chain->active == 0u )
		return;
	chain->active = 0u;
	state = chain->state;
	chain->slot->route_recorded = 0u;
	fprintf(stderr,"G5N-DBG chainfail: stage %u next_layer %u rows %u status %d cuda=%s\n",
		(unsigned)chain->stage,(unsigned)chain->next_layer,(unsigned)chain->wave_rows,(int)status,
		cudaGetErrorString(cudaGetLastError()));
	if ( state->tp_device_collective_initialized != 0u )
	{
		SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective);
		(void)SparkTpDeviceCollectiveChainRetire(&state->tp_device_collective);
	}
	if ( state->tp_device_collective_hc_initialized != 0u )
		SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective_hc);
	if ( SparkGlm5NextBoundedStreamSync(chain->state,chain->slot->stream,UINT64_C(35000000000)) != 0 )
	{
		chain->retained_status = status;
		fprintf(stderr,"GLM chain drain failed; retaining slot %u and CUDA resources for teardown retry\n",chain->slot_index);
		atomic_store_explicit(&state->lazy_retained[chain->slot_index],chain,memory_order_release);
		SparkGlm5NextScheduleRetainedRetry(state);
		return;
	}
	if ( chain->expert_lease != 0u && SparkGlm5NextLazyRelease(chain) != SPARK_STATUS_OK )
	{
		chain->retained_status = status;
		atomic_store_explicit(&state->lazy_retained[chain->slot_index],chain,memory_order_release);
		SparkGlm5NextScheduleRetainedRetry(state);
		return;
	}
	async = &state->completions[chain->slot_index];
	async->completion.status = status;
	SparkGlm5NextCompleteAsync(async);
	free(chain);
}

static SparkStatus SparkGlm5NextLazyRelease(SparkGlm5NextTpChain *chain)
{
	SparkWeightdMap *map = chain->state->lazy_pack->map;
	SparkStatus status;
	if ( chain->expert_lease == 0u )
		return(SPARK_STATUS_OK);
	if ( chain->expert_lease_begun != 0u )
	{
		if ( chain->expert_lease_recorded == 0u )
		{
			status = SparkWeightdMapRecordCompletion(map,chain->expert_lease,(cudaStream_t)chain->slot->stream);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
			chain->expert_lease_recorded = 1u;
			chain->wave.expert_lease_base = 0;
		}
		if ( SparkGlm5NextBoundedStreamSync(chain->state,chain->slot->stream,UINT64_C(35000000000)) != 0 )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	status = SparkWeightdMapRelease(map,chain->expert_lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status == SPARK_STATUS_OK )
	{
		chain->expert_lease = 0u;
		chain->expert_lease_begun = 0u;
		chain->expert_lease_recorded = 0u;
		chain->wave.expert_lease_base = 0;
	}
	SPARK_RETURN(status);
}

#include "sparkpipe/family/module/spark_module_lazy_recover_lease.h"

static void SparkGlm5NextLazyRetryRetained(void *context)
{
	SparkGlm5NextModuleState *state = (SparkGlm5NextModuleState *)context;
	SparkGlm5NextTpChain *chain;
	uint32_t slot;
	for (slot=0u; slot<state->pipeline_slot_count; slot++)
		if ( SparkGlm5NextLazyRecoverLease(state,slot,&chain) == SPARK_STATUS_OK )
		{
			chain->active = 1u;
			SparkGlm5NextTpChainFail(chain,chain->retained_status);
		}
}

#include "sparkpipe/family/module/spark_module_tp_chain_reduce_mlp.h"

static uint32_t SparkGlm5NextFirstRoutedLayer(const SparkGlm5NextModuleState *state)
{
	return(state->first_layer_index > SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER ? state->first_layer_index : SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER);
}

static SparkStatus SparkGlm5NextPinAllExperts(SparkGlm5NextModuleState *state)
{
	SparkWeightdExpertKey keys[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t layer,expert,count = 0u;
	if ( state->expert_pin_lease_count != 0u )
		return(SPARK_STATUS_BUSY);
	for (layer=SparkGlm5NextFirstRoutedLayer(state); layer<state->first_layer_index + state->layer_count && status==SPARK_STATUS_OK; layer++)
		for (expert=0u; expert<SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT && status==SPARK_STATUS_OK; expert++)
		{
			keys[count++] = (SparkWeightdExpertKey){.layer=layer,.expert=expert};
			if ( count == SPARK_WEIGHTD_LEASE_GROUPS_MAX || (layer + 1u == state->first_layer_index + state->layer_count && expert + 1u == SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT) )
			{
				uint64_t lease = 0u;
				void *base = 0;
				uint32_t index = state->expert_pin_lease_count;
				if ( index >= sizeof(state->expert_pin_leases) / sizeof(state->expert_pin_leases[0]) )
					return(SPARK_STATUS_CAPACITY_EXCEEDED);
				status = SparkWeightdMapAcquire(state->lazy_pack->map,keys,count,&lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
				if ( lease != 0u )
					state->expert_pin_leases[state->expert_pin_lease_count++] = lease;
				if ( status == SPARK_STATUS_OK )
					status = lease != 0u ? SparkWeightdMapBeginUse(state->lazy_pack->map,lease,&base) : SPARK_STATUS_VALIDATION_FAILED;
				if ( status == SPARK_STATUS_OK )
				{
					state->expert_pin_phases[index] = 1u;
					if ( base == 0 || (state->decode_lease_base_saved != 0 && state->decode_lease_base_saved != base) )
						status = SPARK_STATUS_VALIDATION_FAILED;
					else
					{
						state->expert_pin_key_count += count;
						state->decode_lease_base_saved = base;
					}
				}
				count = 0u;
			}
		}
	return(status);
}

static SparkStatus SparkGlm5NextReleasePinnedAbove(SparkGlm5NextModuleState *state,uint32_t floor)
{
	while ( state->expert_pin_lease_count > floor )
	{
		uint32_t index = state->expert_pin_lease_count - 1u;
		SparkStatus status;
		if ( state->expert_pin_phases[index] == 1u )
		{
			status = SparkWeightdMapRecordCompletion(state->lazy_pack->map,state->expert_pin_leases[index],(cudaStream_t)state->execution_stream);
			if ( status != SPARK_STATUS_OK ) return(status);
			state->expert_pin_phases[index] = 2u;
		}
		status = SparkWeightdMapRelease(state->lazy_pack->map,state->expert_pin_leases[index],SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
		if ( status != SPARK_STATUS_OK ) return(status);
		state->expert_pin_phases[index] = 0u;
		state->expert_pin_leases[index] = 0u;
		state->expert_pin_lease_count--;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextReleasePinnedExperts(SparkGlm5NextModuleState *state)
{
	SparkStatus status = SparkGlm5NextReleasePinnedAbove(state,0u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->expert_pin_key_count = 0u;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm5NextPinnedExpected(const SparkGlm5NextModuleState *state)
{
	uint32_t first = SparkGlm5NextFirstRoutedLayer(state),end = state->first_layer_index + state->layer_count;
	return(end > first ? (end - first) * SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT : 0u);
}

static uint32_t SparkGlm5NextExpertsPinned(const SparkGlm5NextModuleState *state)
{
	uint32_t index,expected = SparkGlm5NextPinnedExpected(state);
	if ( state->expert_pin_key_count != expected || state->expert_pin_lease_count != SparkCeilDivU32(expected,SPARK_WEIGHTD_LEASE_GROUPS_MAX) || (expected != 0u && state->decode_lease_base_saved == 0) )
		return(0u);
	for (index=0u; index<state->expert_pin_lease_count; index++)
		if ( state->expert_pin_leases[index] == 0u || state->expert_pin_phases[index] != 1u )
			return(0u);
	return(1u);
}

static SparkStatus SparkGlm5NextMissRingEnsure(SparkGlm5NextExecutionSlot *slot);
static void SparkGlm5NextGraphDisarm(SparkGlm5NextModuleState *state);

static SparkStatus SparkGlm5NextWsAcquire(void *context,const uint32_t *keys,uint32_t count)
{
	SparkGlm5NextModuleState *state = (SparkGlm5NextModuleState *)context;
	SparkWeightdExpertKey group[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
	uint32_t index = 0u,take,item,slot,capacity = (uint32_t)(sizeof(state->expert_pin_leases) / sizeof(state->expert_pin_leases[0])),floor = state->expert_pin_lease_count,taken;
	SparkStatus status = SPARK_STATUS_OK,unwind;
	if ( state->expert_pin_lease_count + SparkCeilDivU32(count,SPARK_WEIGHTD_LEASE_GROUPS_MAX) > capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	while ( index < count && status == SPARK_STATUS_OK )
	{
		uint64_t lease = 0u;
		void *base = 0;
		take = count - index < SPARK_WEIGHTD_LEASE_GROUPS_MAX ? count - index : SPARK_WEIGHTD_LEASE_GROUPS_MAX;
		for (item=0u; item<take; item++)
			group[item] = (SparkWeightdExpertKey){.layer=keys[index + item] / SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE,.expert=keys[index + item] % SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE};
		status = SparkWeightdMapAcquire(state->lazy_pack->map,group,take,&lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
		slot = state->expert_pin_lease_count;
		if ( lease != 0u )
		{
			state->expert_pin_leases[slot] = lease;
			state->expert_pin_phases[slot] = 0u;
			state->expert_pin_lease_count++;
		}
		if ( status == SPARK_STATUS_OK )
			status = lease != 0u ? SparkWeightdMapBeginUse(state->lazy_pack->map,lease,&base) : SPARK_STATUS_VALIDATION_FAILED;
		if ( status == SPARK_STATUS_OK )
			state->expert_pin_phases[slot] = 1u;
		if ( status == SPARK_STATUS_OK && (base == 0 || (state->decode_lease_base_saved != 0 && state->decode_lease_base_saved != base)) )
			status = SPARK_STATUS_VALIDATION_FAILED;
		if ( status == SPARK_STATUS_OK )
		{
			state->decode_lease_base_saved = base;
			index += take;
		}
	}
	if ( status == SPARK_STATUS_OK )
		return(SPARK_STATUS_OK);
	taken = state->expert_pin_lease_count - floor;
	unwind = SparkGlm5NextReleasePinnedAbove(state,floor);
	fprintf(stderr,"EXPERT-WSET acquire failed status=%d keys=%u new_leases=%u unwind_status=%d\n",(int)status,count,taken,(int)unwind);
	SPARK_RETURN(unwind != SPARK_STATUS_OK ? unwind : status);
}

static SparkStatus SparkGlm5NextWsLoad(const char *path,const char *digest,uint32_t **keys_out,uint32_t *count_out)
{
	char hex[SPARK_SHA256_HEX_BYTES];
	uint32_t *pairs = 0,*keys = 0,index;
	long bytes;
	FILE *file;
	SparkStatus status = SparkSha256File(path,hex);
	*keys_out = 0;
	*count_out = 0u;
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( strcmp(hex,digest) != 0 )
	{
		fprintf(stderr,"EXPERT-WSET-HASH-MISMATCH path=%s sha256=%s expected=%s\n",path,hex,digest);
		SPARK_FAIL(SPARK_STATUS_HASH_MISMATCH);
	}
	file = fopen(path,"rb");
	if ( file == 0 || fseek(file,0,SEEK_END) != 0 || (bytes = ftell(file)) <= 0 || (bytes % 8) != 0 || fseek(file,0,SEEK_SET) != 0 )
	{
		if ( file != 0 )
			fclose(file);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	pairs = (uint32_t *)malloc((size_t)bytes);
	keys = (uint32_t *)malloc((size_t)bytes / 2u);
	if ( pairs == 0 || keys == 0 || fread(pairs,1u,(size_t)bytes,file) != (size_t)bytes )
		status = pairs == 0 || keys == 0 ? SPARK_STATUS_CAPACITY_EXCEEDED : SPARK_STATUS_IO_ERROR;
	fclose(file);
	for (index=0u; status == SPARK_STATUS_OK && index<(uint32_t)(bytes / 8); index++)
	{
		if ( pairs[2u * index] >= SPARK_GLM5_NEXT_MODEL_LAYER_COUNT || pairs[2u * index + 1u] >= SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT )
			status = SPARK_STATUS_SCHEMA_ERROR;
		else
			keys[index] = pairs[2u * index] * SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE + pairs[2u * index + 1u];
	}
	free(pairs);
	if ( status != SPARK_STATUS_OK )
	{
		free(keys);
		SPARK_RETURN(status);
	}
	*keys_out = keys;
	*count_out = (uint32_t)(bytes / 8);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextWsOpen(SparkGlm5NextModuleState *state)
{
	const char *path = getenv("SPARK_GLM5_NEXT_EXPERT_WSET"),*digest = getenv("SPARK_GLM5_NEXT_EXPERT_WSET_SHA256"),*pin = getenv("SPARK_GLM5_NEXT_PIN_EXPERTS");
	uint32_t first = SparkGlm5NextFirstRoutedLayer(state),end = state->first_layer_index + state->layer_count,*keys = 0,count = 0u,index,missing = 0u;
	SparkStatus status;
	if ( path == 0 || path[0] == '\0' )
		return(SPARK_STATUS_OK);
	if ( pin != 0 && pin[0] == '1' )
	{
		fprintf(stderr,"SPARK_GLM5_NEXT_EXPERT_WSET and SPARK_GLM5_NEXT_PIN_EXPERTS=1 are exclusive residency modes\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( state->owns_embedding == 0u || state->owns_final_head == 0u || state->lazy_pack == 0 || state->lazy_pack->map == 0 || end <= first )
	{
		fprintf(stderr,"EXPERT-WSET requires a single-stage lazy expert pack (the miss poison rides the final head reduce)\n");
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	if ( digest == 0 || SparkSha256HexIsValid(digest) == 0 )
	{
		fprintf(stderr,"SPARK_GLM5_NEXT_EXPERT_WSET_SHA256 must be the 64-hex digest of the .wset\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	status = SparkGlm5NextWsLoad(path,digest,&keys,&count);
	for (index=0u; status == SPARK_STATUS_OK && index<count; index++)
		if ( keys[index] / SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE < first || keys[index] / SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE >= end )
			status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkExpertWorkingSetCreate(&state->expert_ws,end,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,SPARK_GLM5_NEXT_MODEL_MISS_PACK_STRIDE,(end - first) * SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,SparkGlm5NextWsAcquire,state);
	if ( status == SPARK_STATUS_OK )
		status = SparkExpertWorkingSetAdd(&state->expert_ws,keys,count);
	if ( status == SPARK_STATUS_OK && SparkExpertWorkingSetCheckAnchors(&state->expert_ws,first,end - first,&missing) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"EXPERT-WSET missing layer %u: every routed layer needs one held expert\n",missing);
		status = SPARK_STATUS_UNSUPPORTED;
	}
	free(keys);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state->ws_enabled = 1u;
	fprintf(stderr,"EXPERT-RESIDENCY mode=working-set wset=%s keys=%u of %u leases=%u rows_max=%u replays=%u\n",path,state->expert_ws.key_count,(end - first) * SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,state->expert_pin_lease_count,SPARK_GLM5_NEXT_WS_ROWS_MAX,SPARK_GLM5_NEXT_WS_REPLAYS);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextWsSnapshotEnsure(SparkGlm5NextModuleState *state,SparkGlm5NextExecutionSlot *slot)
{
	uint8_t *pools[4] = {state->kda_state_pools,state->kda_q_window_pool,state->kda_k_window_pool,state->kda_v_window_pool};
	uint64_t strides[4] = {state->kda_state_layer_stride_bytes,state->kda_window_layer_stride_bytes,state->kda_window_layer_stride_bytes,state->kda_window_layer_stride_bytes},bytes;
	uint32_t part,layer,count = 4u * state->kda_layer_count,words = 0u;
	SparkStateSpan *spans;
	SparkStatus status;
	if ( slot->snapshot != 0 )
		return(SPARK_STATUS_OK);
	if ( state->kda_layer_count == 0u || state->resident_sequence_capacity == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	spans = (SparkStateSpan *)calloc(count,sizeof(*spans));
	if ( spans == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (part=0u; part<4u; part++)
		for (layer=0u; layer<state->kda_layer_count; layer++)
		{
			SparkStateSpan *span = &spans[part * state->kda_layer_count + layer];
			span->base = pools[part] != 0 ? pools[part] + (uint64_t)layer * strides[part] : 0;
			span->row_stride = strides[part] / state->resident_sequence_capacity;
			span->row_bytes = (uint32_t)span->row_stride;
			span->state_rows = state->resident_sequence_capacity;
		}
	bytes = SparkStateSpansLayout(spans,count,SPARK_GLM5_NEXT_WS_ROWS_MAX,&words);
	status = bytes != 0u ? SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateBytes(state,count,1u,sizeof(*spans),&slot->snapshot_spans);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,cudaMemcpy(slot->snapshot_spans,spans,(size_t)count * sizeof(*spans),cudaMemcpyHostToDevice),"ws_snapshot_spans");
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateBytes(state,bytes,1u,1u,(void **)&slot->snapshot);
	free(spans);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	slot->snapshot_span_count = count;
	slot->snapshot_row_words = words;
	fprintf(stderr,"EXPERT-WSET snapshot spans=%u bytes=%llu rows_max=%u\n",count,(unsigned long long)bytes,SPARK_GLM5_NEXT_WS_ROWS_MAX);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextWsSlotEnsure(SparkGlm5NextModuleState *state,SparkGlm5NextExecutionSlot *slot)
{
	SparkStatus status = SPARK_STATUS_OK;
	if ( slot->cover_device == 0 )
	{
		status = SparkGlm5NextAllocateBytes(state,SparkExpertWorkingSetCoverBytes(&state->expert_ws),1u,1u,(void **)&slot->cover_device);
		slot->cover_generation = UINT64_MAX;
	}
	if ( status == SPARK_STATUS_OK && slot->cover_generation != state->expert_ws.generation )
	{
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,cudaMemcpyAsync(slot->cover_device,state->expert_ws.cover,SparkExpertWorkingSetCoverBytes(&state->expert_ws),cudaMemcpyHostToDevice,(cudaStream_t)slot->stream),"ws_cover");
		if ( status == SPARK_STATUS_OK )
			slot->cover_generation = state->expert_ws.generation;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextWsSnapshotEnsure(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextMissRingEnsure(slot);
	SPARK_RETURN(status);
}

static cudaError_t SparkGlm5NextWsSnapshot(const SparkGlm5NextCudaWave *wave,uint32_t restore)
{
	const SparkGlm5NextExecutionSlot *slot = wave->slot;
	uint32_t rows = wave->run_count != 0u ? wave->run_count : wave->row_count;
	if ( rows > SPARK_GLM5_NEXT_WS_ROWS_MAX )
		return(cudaErrorInvalidValue);
	return(SparkGlm5NextLaunchStateSnapshot((cudaStream_t)slot->stream,slot->snapshot_spans,slot->snapshot_span_count,slot->snapshot_row_words,slot->snapshot,wave->run_count != 0u ? wave->run_state_index : wave->kda_state_index,rows,restore));
}

static SparkStatus SparkGlm5NextWsRecover(SparkGlm5NextTpChain *chain,SparkStepAction *action)
{
	SparkGlm5NextModuleState *state = chain->state;
	uint32_t keys[SPARK_GLM5_NEXT_MODEL_MISS_RING_CAPACITY];
	SparkExpertMissHarvest harvest;
	SparkStatus status,grow = SPARK_STATUS_OK;
	*action = SPARK_STEP_ACTION_FAIL;
	if ( SparkGlm5NextWsSnapshot(&chain->wave,1u) != cudaSuccess || SparkGlm5NextBoundedStreamSync(state,chain->slot->stream,UINT64_C(35000000000)) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkExpertWorkingSetHarvest(&state->expert_ws,chain->slot->miss_ring,SPARK_GLM5_NEXT_MODEL_MISS_RING_CAPACITY,keys,SPARK_GLM5_NEXT_MODEL_MISS_RING_CAPACITY,&harvest);
	chain->slot->miss_ring[SPARK_STEP_MISS_FLAG] = 0u;
	chain->slot->miss_ring[SPARK_STEP_MISS_COUNT] = 0u;
	if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_CAPACITY_EXCEEDED )
		SPARK_RETURN(status);
	if ( status == SPARK_STATUS_OK && harvest.key_count != 0u )
		grow = SparkExpertWorkingSetAdd(&state->expert_ws,keys,harvest.key_count);
	if ( grow != SPARK_STATUS_OK && grow != SPARK_STATUS_BUSY && grow != SPARK_STATUS_CAPACITY_EXCEEDED )
		SPARK_RETURN(grow);
	chain->ws_replay.limit = SPARK_GLM5_NEXT_WS_REPLAYS;
	*action = SparkStepReplayNext(&chain->ws_replay,chain->step_verdict);
	state->ws_local_miss += chain->step_verdict == SPARK_STEP_VERDICT_ROLLBACK_LOCAL ? 1u : 0u;
	state->ws_remote_miss += chain->step_verdict == SPARK_STEP_VERDICT_ROLLBACK_REMOTE ? 1u : 0u;
	state->ws_replays += *action == SPARK_STEP_ACTION_REPLAY ? 1u : 0u;
	state->ws_eager_steps += *action == SPARK_STEP_ACTION_EXHAUSTED ? 1u : 0u;
	fprintf(stderr,"GRAPH-WS-RECOVER slot=%u step=%u verdict=%s first_layer=%u misses=%u grown=%u grow_status=%d harvest_status=%d action=%s keys=%u local=%llu remote=%llu replays=%llu eager=%llu\n",
		chain->slot_index,chain->step,SparkStepVerdictName(chain->step_verdict),harvest.first_layer,harvest.recorded,harvest.key_count,(int)grow,(int)status,
		*action == SPARK_STEP_ACTION_REPLAY ? "replay" : "eager",state->expert_ws.key_count,
		(unsigned long long)state->ws_local_miss,(unsigned long long)state->ws_remote_miss,(unsigned long long)state->ws_replays,(unsigned long long)state->ws_eager_steps);
	if ( *action == SPARK_STEP_ACTION_EXHAUSTED )
		chain->ws_force_eager = 1u;
	return(*action == SPARK_STEP_ACTION_FAIL ? SPARK_STATUS_INTERNAL_ERROR : SPARK_STATUS_OK);
}

static uint32_t SparkGlm5NextWsRetry(SparkGlm5NextTpChain *chain)
{
	SparkStepAction action;
	SparkStatus status;
	if ( chain->state->ws_enabled == 0u || (chain->step_verdict != SPARK_STEP_VERDICT_ROLLBACK_LOCAL && chain->step_verdict != SPARK_STEP_VERDICT_ROLLBACK_REMOTE) )
		return(0u);
	SparkGlm5NextGraphDisarm(chain->state);
	status = SparkGlm5NextWsRecover(chain,&action);
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm5NextTpChainFail(chain,status);
		return(1u);
	}
	chain->step_verdict = SPARK_STEP_VERDICT_COMMIT;
	chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN;
	chain->next_layer = 0u;
	SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	return(1u);
}

static SparkStatus SparkGlm5NextGraphClaimExperts(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state = chain->state;
	uint32_t index,expected = SparkGlm5NextPinnedExpected(state);
	if ( state->ws_enabled != 0u )
	{
		if ( chain->wave_rows > SPARK_GLM5_NEXT_WS_ROWS_MAX || chain->ws_force_eager != 0u || state->decode_lease_base_saved == 0 )
			return(SPARK_STATUS_UNSUPPORTED);
		chain->wave.expert_lease_base = state->decode_lease_base_saved;
		chain->wave.expert_lease_local_layer = 0u;
		chain->wave.expert_lease_all = 1u;
		chain->wave.expert_cover = 0;
		chain->wave.expert_miss = 0;
		chain->wave.expert_cover_stride = SPARK_GLM5_NEXT_COVER_STRIDE;
		{
			SparkStatus status = SparkGlm5NextWsSlotEnsure(state,chain->slot);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
		}
		chain->wave.expert_cover = chain->slot->cover_device;
		chain->wave.expert_miss = chain->slot->miss_ring;
		return(SPARK_STATUS_OK);
	}
	if ( state->expert_pin_key_count != expected || state->expert_pin_lease_count != SparkCeilDivU32(expected,SPARK_WEIGHTD_LEASE_GROUPS_MAX) )
	{
		fprintf(stderr,"GLM whole-chain graph requires %u leased experts; held %u\n",expected,state->expert_pin_key_count);
		return(SPARK_STATUS_UNSUPPORTED);
	}
	for (index=0u; index<state->expert_pin_lease_count; index++)
		if ( state->expert_pin_leases[index] == 0u || state->expert_pin_phases[index] != 1u )
			return(SPARK_STATUS_VALIDATION_FAILED);
	if ( expected != 0u && state->decode_lease_base_saved == 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	chain->wave.expert_lease_base = state->decode_lease_base_saved;
	chain->wave.expert_lease_local_layer = 0u;
	chain->wave.expert_lease_all = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextLazyExperts(SparkGlm5NextTpChain *chain)
{
	SparkWeightdExpertKey keys[SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT];
	SparkWeightdMap *map = chain->state->lazy_pack->map;
	SparkStatus status;
	void *address = 0;
	uint32_t count = 0u;
	if ( chain->slot->route_recorded == 0u || cudaEventSynchronize((cudaEvent_t)chain->slot->route_ready_event) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkWeightdRouteKeys(chain->wave.first_layer_index + chain->next_layer,chain->slot->host_group_row_offset +
		    (chain->wave.first_layer_index + chain->next_layer) *
		        (SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u),
		SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,chain->wave.row_count * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K,keys,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,&count);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdMapAcquire(map,keys,count,&chain->expert_lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status != SPARK_STATUS_OK )
	{
		uint32_t diag_index;
		fprintf(stderr,"LAZYWORK-KEYS slot=%u layer=%u count=%u status=%d",
			(unsigned)chain->slot_index,(unsigned)chain->next_layer,
			(unsigned)count,(int)status);
		for (diag_index=0u; diag_index<count && diag_index<8u; diag_index++)
			fprintf(stderr," %u:%u",(unsigned)keys[diag_index].layer,(unsigned)keys[diag_index].expert);
		fprintf(stderr,"\n");
		SPARK_RETURN(status);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdMapBeginUse(map,chain->expert_lease,&address);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	chain->expert_lease_begun = 1u;
	chain->wave.expert_lease_base = (const uint8_t *)address;
	chain->wave.expert_lease_local_layer = chain->next_layer;
	if ( chain->state->decode_lease_base_saved == 0 )
		chain->state->decode_lease_base_saved = (const uint8_t *)address;
	{
		const uint32_t *cover_saved = chain->wave.expert_cover;
		void *miss_saved = chain->wave.expert_miss;
		chain->wave.expert_cover = 0;
		chain->wave.expert_miss = 0;
		if ( SparkGlm5NextLaunchCudaLayerMlpExperts(&chain->wave,chain->next_layer) != 0 )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		chain->wave.expert_cover = cover_saved;
		chain->wave.expert_miss = miss_saved;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextRouteTraceWrite(SparkGlm5NextTpChain *chain)
{
	SparkWeightdExpertKey keys[SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT];
	char line[SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT * 4u + 128u];
	uint32_t index,count = 0u,layer = chain->wave.first_layer_index + chain->next_layer;
	int used;
	SparkStatus status = SparkWeightdRouteKeys(layer,chain->slot->host_group_row_offset + layer * (SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u),SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,chain->wave.row_count * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K,keys,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,&count);
	used = snprintf(line,sizeof(line),"G5N-ROUTE rank=%u rows=%u pos=%u layer=%u n=%u e=",chain->state->tp_rank,chain->wave.row_count,chain->wave.host_positions[0],layer,count);
	for (index=0u; status == SPARK_STATUS_OK && used > 0 && (size_t)used < sizeof(line) && index<count; index++)
		used += snprintf(line + used,sizeof(line) - (size_t)used,index == 0u ? "%u" : ",%u",keys[index].expert);
	if ( status == SPARK_STATUS_OK && (used <= 0 || (size_t)used >= sizeof(line) || fprintf(chain->state->route_trace,"%s\n",line) < 0) )
		status = SPARK_STATUS_IO_ERROR;
	SPARK_RETURN(status);
}

static void SparkGlm5NextLazyWork(void *context)
{
	SparkGlm5NextTpChain *chain = (SparkGlm5NextTpChain *)context;
	SparkStatus status,cleanup;
	{
		static uint32_t lazy_trace_count;
		if ( lazy_trace_count < 600u )
		{
			lazy_trace_count++;
			fprintf(stderr,"LAZYWORK slot=%u layer=%u\n",(unsigned)chain->slot_index,(unsigned)chain->next_layer);
		}
	}
	status = SparkGlm5NextLazyExperts(chain);
	if ( status == SPARK_STATUS_OK && chain->state->route_trace != 0 )
		status = SparkGlm5NextRouteTraceWrite(chain);
	cleanup = SparkGlm5NextLazyRelease(chain);
	if ( cleanup == SPARK_STATUS_IO_ERROR || cleanup == SPARK_STATUS_BUSY )
		cleanup = SparkGlm5NextLazyRelease(chain);
	if ( cleanup != SPARK_STATUS_OK )
	{
		chain->retained_status = status != SPARK_STATUS_OK ? status : cleanup;
		fprintf(stderr,"GLM expert cleanup failed: slot=%u lease=%llu status=%d; retaining slot and lease for teardown retry\n",chain->slot_index,(unsigned long long)chain->expert_lease,(int32_t)cleanup);
		atomic_store_explicit(&chain->state->lazy_retained[chain->slot_index],chain,memory_order_release);
		SparkGlm5NextScheduleRetainedRetry(chain->state);
		return;
	}
	if ( status != SPARK_STATUS_OK )
		SparkGlm5NextTpChainFail(chain,status);
	else
		SparkGlm5NextTpChainReduceMlp(chain);
}

static void SparkGlm5NextTpChainAdvance(void *chain_context,SparkStatus status);

static SparkStatus SparkGlm5NextGraphReduce(SparkGlm5NextTpChain *chain,void *device,uint32_t hc_wide)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkTpDeviceCollectiveSubmission submission;
	SparkStatus ordinal_status;
	uint64_t ordinal;
	ordinal_status = SparkGlm5NextChainOrdinal(chain,hc_wide,hc_wide != 0u ? chain->tp_hc_op_index : chain->tp_op_index,&ordinal);
	if ( ordinal_status != SPARK_STATUS_OK )
		return(ordinal_status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = chain->wave_rows;
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = device;
	submission.full_device = device;
	submission.cuda_stream = chain->slot->stream;
	if ( hc_wide != 0u )
		chain->tp_hc_op_index += 1u;
	else
		chain->tp_op_index += 1u;
	return(SparkTpDeviceCollectiveEnqueue(hc_wide != 0u ? &state->tp_device_collective_hc : &state->tp_device_collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16));
}

static SparkStatus SparkGlm5NextGraphReduceHead(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	SparkStatus ordinal_status;
	uint64_t ordinal;
	state = chain->state;
	ordinal_status = SparkGlm5NextChainOrdinal(chain,0u,chain->tp_op_index,
		&ordinal);
	if ( ordinal_status != SPARK_STATUS_OK )
		return(ordinal_status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = chain->wave_rows;
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = chain->slot->head_maxloc_u64;
	submission.full_device = chain->slot->head_maxloc_u64;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = 0;
	submission.completion_context = 0;
	chain->tp_op_index += 1u;
	return(SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,
		&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64));
}

static uint32_t SparkGlm5NextWalkReduce(SparkGlm5NextTpChain *chain,void *device,uint32_t hc_wide,uint32_t layer,uint32_t site)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextCudaWave *wave = &chain->wave;
	uint32_t placed = 0u;
	if ( SparkGlm5NextGraphReduce(chain,device,hc_wide) != SPARK_STATUS_OK )
		return(1u);
	if ( state->l2_prefetch != 0u && wave->tp_degree > 1u )
	{
		if ( SparkGlm5NextL2PrefetchAfterRound(wave,layer,site,&state->l2_prefetch_shape,&placed) != 0 )
			return(20u + site);
		state->l2_prefetch_rounds += placed;
	}
	return(0u);
}

static SparkStatus SparkGlm5NextMtpReduceRowsOp(void *context,uint16_t *rows_bf16,uint32_t row_count,uint32_t width)
{
	SparkGlm5NextTpChain *chain = (SparkGlm5NextTpChain *)context;
	SparkStatus status;
	uint32_t saved;
	if ( chain == 0 || rows_bf16 == 0 || row_count != 1u || width != SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	saved = chain->wave_rows;
	chain->wave_rows = row_count;
	status = SparkGlm5NextGraphReduce(chain,rows_bf16,0u);
	chain->wave_rows = saved;
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextMtpReduceMaxOp(void *context,uint64_t *values,uint32_t count)
{
	SparkGlm5NextTpChain *chain = (SparkGlm5NextTpChain *)context;
	SparkStatus status;
	uint32_t saved;
	if ( chain == 0 || values == 0 || count != 1u || values != chain->slot->head_maxloc_u64 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	saved = chain->wave_rows;
	chain->wave_rows = count;
	status = SparkGlm5NextGraphReduceHead(chain);
	chain->wave_rows = saved;
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextMtpTap(SparkGlm5NextModuleState *state,SparkGlm5NextExecutionSlot *slot,uint32_t row,uint32_t lane,uint64_t sequence_id,uint64_t next_position)
{
	cudaError_t error;
	if ( state->verify_mtp == 0u || state->mtp_lane_hidden_bf16 == 0 || state->mtp_lane_sequence == 0 || state->owns_final_head == 0u )
		return(SPARK_STATUS_OK);
	if ( lane >= state->resident_sequence_capacity || row >= state->execution_row_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	error = cudaMemcpyAsync(
		state->mtp_lane_hidden_bf16 + (uint64_t)lane * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
		slot->hc_mean_bf16 + (uint64_t)row * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
		(uint64_t)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),
		cudaMemcpyDeviceToDevice,(cudaStream_t)slot->stream);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"mtp_tap"));
	state->mtp_lane_sequence[lane] = sequence_id;
	state->mtp_lane_next[lane] = next_position;
	state->mtp_taps++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextMtpTapFrame(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state = chain->state;
	const SparkGlm5NextResidentDecodeStageBatchView *batch = chain->batch;
	uint32_t row,last;
	uint64_t position;
	if ( state->verify_mtp == 0u || chain->verify_budget != 0u || batch == 0 || batch->active_sequence_count != 1u || batch->row_count == 0u || chain->first_row != 0u || chain->wave_rows == 0u )
		return(SPARK_STATUS_OK);
	last = batch->row_count - 1u;
	if ( chain->frame != 0 && (chain->frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u )
	{
		if ( chain->wave_rows != batch->row_count )
			return(SPARK_STATUS_OK);
		row = chain->wave_rows - 1u;
		position = batch->row_positions[last];
	}
	else
	{
		if ( batch->row_count != 1u || chain->wave_rows != 1u || chain->steps == 0u )
			return(SPARK_STATUS_OK);
		row = 0u;
		position = batch->row_positions[0] + chain->steps - 1u;
	}
	return(SparkGlm5NextMtpTap(state,chain->slot,row,batch->row_resident_slots[last],batch->row_sequence_ids[last],position + 1u));
}

static SparkStatus SparkGlm5NextMtpDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result)
{
	SparkGlm5NextModuleState *state = (SparkGlm5NextModuleState *)context;
	SparkGlm5NextTpChain *chain;
	SparkGlm5NextMtpDraftOps ops;
	SparkGlm5NextCudaWave wave;
	uint32_t tokens[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX];
	uint32_t lane,depth,index;
	uint64_t started_ns;
	SparkStatus status;
	if ( state == 0 || request == 0 || result == 0 || request->requested_token_count == 0u || state->mtp_chain == 0 || state->mtp_lane_sequence == 0 || request->active_sequence_index >= state->resident_sequence_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	chain = state->mtp_chain;
	lane = request->active_sequence_index;
	result->token_count = 0u;
	if ( state->mtp_lane_sequence[lane] != request->sequence_id || state->mtp_lane_next[lane] != request->sequence_position )
	{
		state->mtp_cold++;
		return(SPARK_STATUS_NOT_FOUND);
	}
	if ( state->tp_degree > 1u && (state->tp_device_collective_initialized == 0u || SparkTpDeviceCollectiveStreamOrdered(&state->tp_device_collective) == 0u) )
	{
		fprintf(stderr,"VERIFY-MTP-UNSUPPORTED the MTP drafter at tp=%u needs stream-ordered collectives\n",state->tp_degree);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	depth = request->requested_token_count < SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX ? request->requested_token_count : SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX;
	SparkGlm5NextBuildMtpDraftWave(state,chain->slot,&wave);
	wave.mtp_draft_depth = depth;
	memset(&ops,0,sizeof(ops));
	ops.context = chain;
	ops.reduce_rows_bf16 = SparkGlm5NextMtpReduceRowsOp;
	ops.reduce_max_u64 = SparkGlm5NextMtpReduceMaxOp;
	started_ns = SparkGlm5NextNowNs();
	if ( SparkGlm5NextLaunchCudaMtpDraft(&wave,state->tp_degree > 1u ? &ops : 0,
		state->mtp_lane_hidden_bf16 + (uint64_t)lane * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
		state->mtp_anchor_token,tokens) != 0 )
	{
		fprintf(stderr,"VERIFY-MTP-DRAFT-FAILED lane=%u position=%llu depth=%u\n",lane,(unsigned long long)request->sequence_position,depth);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	if ( state->tp_degree > 1u )
	{
		status = SparkTpDeviceCollectiveVerifyDeferred(&state->tp_device_collective,state->execution_stream);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	state->mtp_drafts++;
	state->mtp_draft_ns += SparkGlm5NextNowNs() - started_ns;
	for (index=0u; index<depth && tokens[index] < SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT; index++)
	{
		result->token_ids[index] = tokens[index];
		result->confidence_milli[index] = 0u;
	}
	if ( index < depth )
		state->mtp_truncated++;
	result->token_count = index;
	state->mtp_draft_tokens += index;
	return(index == 0u ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_OK);
}

#define SPARK_GLM5_NEXT_TAPS_ENV "SPARK_GLM5_NEXT_TAPS"
#define SPARK_GLM5_NEXT_TAP_RANK_ENV "SPARK_GLM5_NEXT_TAP_RANK"
#define SPARK_GLM5_NEXT_TAP_DUMP_ENV "SPARK_GLM5_NEXT_TAP_DUMP"
#define SPARK_GLM5_NEXT_TAP_DUMP_MAX_BYTES_ENV "SPARK_GLM5_NEXT_TAP_DUMP_MAX_BYTES"
#define SPARK_GLM5_NEXT_TAP_RELAY_LOCAL_ENV "SPARK_GLM5_NEXT_TAP_RELAY_LOCAL"
#define SPARK_GLM5_NEXT_TAP_RELAY_PEER_ENV "SPARK_GLM5_NEXT_TAP_RELAY_PEER"
#define SPARK_GLM5_NEXT_TAP_RELAY_SHADOW_ENV "SPARK_GLM5_NEXT_TAP_RELAY_SHADOW"
#define SPARK_GLM5_NEXT_TAP_RELAY_AWAIT_US_ENV "SPARK_GLM5_NEXT_TAP_RELAY_AWAIT_US"
#define SPARK_GLM5_NEXT_TAP_RELAY_DEPTH_ENV "SPARK_GLM5_NEXT_TAP_RELAY_DEPTH"
#define SPARK_GLM5_NEXT_TAP_RELAY_AWAIT_US_MAX 100000u

static uint32_t SparkGlm5NextTapCapture(SparkGlm5NextTpChain *chain,uint32_t local_layer)
{
	SparkGlm5NextModuleState *state = chain->state;
	uint32_t ordinal;
	if ( state->tap_enabled == 0u )
		return(0u);
	ordinal = SparkSpeculationTapSetOrdinal(&state->tap_set,chain->wave.first_layer_index + local_layer);
	if ( ordinal == UINT32_MAX )
		return(0u);
	return(SparkGlm5NextLaunchCudaTapCapture(&chain->wave,state->tap_set.reduction == SPARK_SPECULATION_TAP_REDUCTION_ALL ? 1u : 0u,
		chain->slot->tap_device + (uint64_t)ordinal * state->execution_row_capacity * state->tap_set.row_elements) != 0 ? 1u : 0u);
}

static uint32_t SparkGlm5NextTapFlush(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state = chain->state;
	uint64_t pitch;
	if ( state->tap_enabled == 0u )
		return(0u);
	pitch = (uint64_t)state->execution_row_capacity * state->tap_set.row_bytes;
	return(cudaMemcpy2DAsync(chain->slot->tap_host,pitch,chain->slot->tap_device,pitch,(uint64_t)chain->wave.row_count * state->tap_set.row_bytes,state->tap_set.tap_count,cudaMemcpyDeviceToHost,(cudaStream_t)chain->slot->stream) != cudaSuccess ? 1u : 0u);
}

static void SparkGlm5NextTapReport(SparkGlm5NextModuleState *state,const char *label)
{
	const SparkSpeculationRelayRemote *remote = state->tap_remote;
	char accepted_at[160];
	uint32_t index;
	int length = 0;
	accepted_at[0] = '\0';
	for (index=0u; remote != 0 && index<=state->tap_relay_depth && length >= 0 && (size_t)length < sizeof(accepted_at); index++)
		length += snprintf(accepted_at + length,sizeof(accepted_at) - (size_t)length,"%s%llu",index == 0u ? "" : ",",(unsigned long long)remote->shadow_accepted_at[index]);
	fprintf(stderr,"%s rank=%u taps=%u records=%llu dump_records=%llu dump_refused=%llu dump_failed=%llu relay_records=%llu relay_unsent=%llu send_errors=%llu foreign=%llu requests=%llu unsynced=%llu unsent=%llu in_time=%llu late=%llu misses=%llu shadow_rounds=%llu proposed=%llu accepted=%llu evicted=%llu accepted_at=%s rtt_p50_us=%llu rtt_p99_us=%llu rtt_max_us=%llu\n",
		label,state->tap_rank,state->tap_set.tap_count,(unsigned long long)state->tap_records,(unsigned long long)state->tap_dump.records,(unsigned long long)state->tap_dump.refused,(unsigned long long)state->tap_dump_failed,
		(unsigned long long)state->tap_link.tap_records_sent,(unsigned long long)state->tap_link.tap_records_unsent,(unsigned long long)state->tap_link.send_errors,(unsigned long long)state->tap_link.datagrams_foreign,
		(unsigned long long)(remote != 0 ? remote->requests : 0u),(unsigned long long)(remote != 0 ? remote->unsynced : 0u),(unsigned long long)(remote != 0 ? remote->unsent : 0u),
		(unsigned long long)(remote != 0 ? remote->answered_in_time : 0u),(unsigned long long)(remote != 0 ? remote->answered_late : 0u),(unsigned long long)(remote != 0 ? remote->deadline_misses : 0u),
		(unsigned long long)(remote != 0 ? remote->shadow_rounds : 0u),(unsigned long long)(remote != 0 ? remote->shadow_proposed : 0u),(unsigned long long)(remote != 0 ? remote->shadow_accepted : 0u),(unsigned long long)(remote != 0 ? remote->shadow_evicted : 0u),
		accepted_at,(unsigned long long)(remote != 0 ? SparkSpeculationRelayRemoteRttPercentileNs(remote,500u) / 1000u : 0u),(unsigned long long)(remote != 0 ? SparkSpeculationRelayRemoteRttPercentileNs(remote,990u) / 1000u : 0u),
		(unsigned long long)(remote != 0 ? remote->rtt_max_ns / 1000u : 0u));
}

static uint64_t SparkGlm5NextTapSequence(const SparkGlm5NextTpChain *chain,uint32_t row)
{
	const SparkGlm5NextAsyncCompletion *async = &chain->state->completions[chain->slot_index];
	uint32_t index = chain->first_row + row,lane;
	if ( chain->verify_budget != 0u )
		return(chain->verify_sequence_id);
	if ( chain->batch != 0 && index < chain->batch->row_count )
		return(chain->batch->row_sequence_ids[index]);
	for (lane=0u; lane<async->lane_count; lane++)
		if ( async->lane_indices[lane] == chain->slot->host_resident_slots[index] )
			return(async->lane_sequence_ids[lane]);
	return(UINT64_MAX);
}

static void SparkGlm5NextTapEmit(SparkGlm5NextTpChain *chain,uint32_t rows,uint32_t flags)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	const uint32_t *positions,*tokens,*outputs;
	SparkSpeculationPolicyDraftResult result;
	SparkSpeculationTapRecord record;
	SparkStatus status;
	uint64_t sequence = 0u,next_position;
	uint32_t row,tap,single = 1u,prefill,next_token;
	if ( state->tap_enabled == 0u || rows == 0u || slot->host_positions == 0 || slot->host_token_ids == 0 || slot->host_output_token_ids == 0 )
		return;
	positions = slot->host_positions + chain->first_row;
	tokens = slot->host_token_ids + chain->first_row;
	outputs = slot->host_output_token_ids + chain->first_row;
	prefill = chain->frame != 0 && (chain->frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	pthread_mutex_lock(&state->tap_lock);
	for (row=0u; row<rows; row++)
	{
		memset(&record,0,sizeof(record));
		record.engine_generation = state->tap_generation;
		record.sequence_id = SparkGlm5NextTapSequence(chain,row);
		record.position = positions[row];
		record.token_id = tokens[row];
		record.next_token_id = outputs[row];
		record.flags = prefill != 0u ? SPARK_SPECULATION_TAP_FLAG_PREFILL : flags;
		record.serial = ++state->tap_serial;
		if ( row == 0u )
			sequence = record.sequence_id;
		else if ( record.sequence_id != sequence || positions[row] != positions[row - 1u] + 1u )
			single = 0u;
		for (tap=0u; tap<state->tap_set.tap_count; tap++)
			memcpy(state->tap_scratch + (uint64_t)tap * state->tap_set.row_bytes,(const uint8_t *)slot->tap_host + ((uint64_t)tap * state->execution_row_capacity + row) * state->tap_set.row_bytes,state->tap_set.row_bytes);
		if ( state->tap_dump_open != 0u )
		{
			status = SparkSpeculationTapDumpAppend(&state->tap_dump,&record,state->tap_scratch);
			if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_CAPACITY_EXCEEDED )
				state->tap_dump_failed++;
			if ( status != SPARK_STATUS_OK && state->tap_dump_warned == 0u )
			{
				state->tap_dump_warned = 1u;
				fprintf(stderr,"TAP-DUMP-STOPPED status=%d records=%llu bytes=%llu max_bytes=%llu; serving continues, later records are counted as refused\n",(int)status,(unsigned long long)state->tap_dump.records,(unsigned long long)state->tap_dump.bytes,(unsigned long long)state->tap_dump.max_bytes);
			}
		}
		if ( state->tap_remote != 0 )
			(void)SparkSpeculationRelayLinkSendTap(&state->tap_link,state->tap_fingerprint,&record,state->tap_scratch,state->tap_set.record_bytes,SPARK_SPECULATION_TAP_FRAGMENT_PAYLOAD_MAX);
		state->tap_records++;
	}
	if ( state->tap_remote != 0 && single != 0u && sequence != UINT64_MAX )
	{
		(void)SparkSpeculationRelayRemoteObserve(state->tap_remote,sequence,positions[0],tokens,rows);
		next_position = (uint64_t)positions[rows - 1u] + 1u;
		next_token = outputs[rows - 1u];
		memset(&result,0,sizeof(result));
		if ( prefill == 0u && next_token < SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT && SparkSpeculationRelayRemoteObserve(state->tap_remote,sequence,next_position,&next_token,1u) == SPARK_STATUS_OK )
			(void)SparkSpeculationRelayRemoteRound(state->tap_remote,sequence,next_position,state->tap_relay_depth,&result);
	}
	state->tap_emits++;
	if ( (state->tap_emits & (state->tap_emits - 1u)) == 0u )
		SparkGlm5NextTapReport(state,"TAP-STATS");
	pthread_mutex_unlock(&state->tap_lock);
}

static SparkStatus SparkGlm5NextTapEnvU64(const char *name,uint64_t minimum,uint64_t maximum,uint64_t *value_out)
{
	const char *text = getenv(name);
	char *end;
	unsigned long long value;
	if ( text == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	errno = 0;
	value = strtoull(text,&end,10);
	if ( text[0] < '0' || text[0] > '9' || errno != 0 || *end != '\0' || value < minimum || value > maximum )
	{
		fprintf(stderr,"%s must be an integer in %llu..%llu\n",name,(unsigned long long)minimum,(unsigned long long)maximum);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	*value_out = value;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextConfigureTaps(SparkGlm5NextModuleState *state)
{
	const char *taps = getenv(SPARK_GLM5_NEXT_TAPS_ENV),*dump = getenv(SPARK_GLM5_NEXT_TAP_DUMP_ENV);
	const char *local = getenv(SPARK_GLM5_NEXT_TAP_RELAY_LOCAL_ENV),*peer = getenv(SPARK_GLM5_NEXT_TAP_RELAY_PEER_ENV),*shadow = getenv(SPARK_GLM5_NEXT_TAP_RELAY_SHADOW_ENV);
	uint64_t rank,max_bytes = 0u,await_us = 0u,depth = 0u,bytes;
	struct timespec now;
	SparkStatus status;
	uint32_t index;
	if ( taps == 0 )
	{
		if ( dump != 0 || local != 0 || peer != 0 || shadow != 0 || getenv(SPARK_GLM5_NEXT_TAP_RANK_ENV) != 0 || getenv(SPARK_GLM5_NEXT_TAP_DUMP_MAX_BYTES_ENV) != 0 || getenv(SPARK_GLM5_NEXT_TAP_RELAY_AWAIT_US_ENV) != 0 || getenv(SPARK_GLM5_NEXT_TAP_RELAY_DEPTH_ENV) != 0 )
		{
			fprintf(stderr,"GLM tap settings need %s\n",SPARK_GLM5_NEXT_TAPS_ENV);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		return(SPARK_STATUS_OK);
	}
	status = SparkSpeculationTapSetParse(taps,SPARK_GLM5_NEXT_MODEL_LAYER_COUNT,SPARK_GLM5_NEXT_MODEL_HC_MULT,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,&state->tap_set);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s must be mean:L[,L...] or all:L[,L...] with 1..%u increasing layers below %u\n",SPARK_GLM5_NEXT_TAPS_ENV,SPARK_SPECULATION_TAP_MAX_TAPS,SPARK_GLM5_NEXT_MODEL_LAYER_COUNT);
		SPARK_RETURN(status);
	}
	rank = state->tp_degree - 1u;
	status = SparkGlm5NextTapEnvU64(SPARK_GLM5_NEXT_TAP_RANK_ENV,0u,state->tp_degree - 1u,&rank);
	if ( status == SPARK_STATUS_OK || status == SPARK_STATUS_NOT_FOUND )
		status = dump != 0 ? SparkGlm5NextTapEnvU64(SPARK_GLM5_NEXT_TAP_DUMP_MAX_BYTES_ENV,SPARK_SPECULATION_TAP_DUMP_HEADER_BYTES,UINT64_MAX,&max_bytes) : SPARK_STATUS_OK;
	if ( status == SPARK_STATUS_OK && local != 0 )
		status = SparkGlm5NextTapEnvU64(SPARK_GLM5_NEXT_TAP_RELAY_AWAIT_US_ENV,0u,SPARK_GLM5_NEXT_TAP_RELAY_AWAIT_US_MAX,&await_us);
	if ( status == SPARK_STATUS_OK && local != 0 )
		status = SparkGlm5NextTapEnvU64(SPARK_GLM5_NEXT_TAP_RELAY_DEPTH_ENV,1u,SPARK_SPECULATION_RELAY_MAX_TOKENS,&depth);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"GLM taps: %s needs %s; %s needs %s, %s and %s\n",SPARK_GLM5_NEXT_TAP_DUMP_ENV,SPARK_GLM5_NEXT_TAP_DUMP_MAX_BYTES_ENV,SPARK_GLM5_NEXT_TAP_RELAY_LOCAL_ENV,SPARK_GLM5_NEXT_TAP_RELAY_PEER_ENV,SPARK_GLM5_NEXT_TAP_RELAY_AWAIT_US_ENV,SPARK_GLM5_NEXT_TAP_RELAY_DEPTH_ENV);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( (dump == 0 && local == 0) || (local == 0) != (peer == 0) || (local != 0 && (shadow == 0 || strcmp(shadow,"1") != 0)) || (local == 0 && shadow != 0) )
	{
		fprintf(stderr,"GLM taps need a consumer: %s, or %s with %s and %s=1. Drafts served through the relay need the root-rank broadcast, which is not built, so the relay runs in shadow mode only\n",
			SPARK_GLM5_NEXT_TAP_DUMP_ENV,SPARK_GLM5_NEXT_TAP_RELAY_LOCAL_ENV,SPARK_GLM5_NEXT_TAP_RELAY_PEER_ENV,SPARK_GLM5_NEXT_TAP_RELAY_SHADOW_ENV);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->tap_rank = (uint32_t)rank;
	state->tap_relay_depth = (uint32_t)depth;
	state->tap_fingerprint = SparkSpeculationTapSetFingerprint(&state->tap_set);
	fprintf(stderr,"GLM taps %s rank=%u fingerprint=%016llx record_bytes=%u dump=%s max_bytes=%llu relay=%s>%s shadow await_us=%llu depth=%llu this_rank=%u\n",
		taps,state->tap_rank,(unsigned long long)state->tap_fingerprint,state->tap_set.record_bytes,dump != 0 ? dump : "-",(unsigned long long)max_bytes,local != 0 ? local : "-",peer != 0 ? peer : "-",
		(unsigned long long)await_us,(unsigned long long)depth,state->tp_rank);
	if ( state->tp_rank != state->tap_rank )
		return(SPARK_STATUS_OK);
	for (index=0u; index<state->tap_set.tap_count; index++)
		if ( state->tap_set.layers[index] < state->first_layer_index || state->tap_set.layers[index] >= state->first_layer_index + state->layer_count )
		{
			fprintf(stderr,"GLM tap layer %u is outside this stage's layers %u..%u\n",state->tap_set.layers[index],state->first_layer_index,state->first_layer_index + state->layer_count - 1u);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
	if ( state->owns_embedding == 0u || state->owns_final_head == 0u || state->mtp_enabled != 0u )
	{
		fprintf(stderr,"GLM taps need a rank that owns the embedding and the head, without the legacy MTP chain\n");
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	bytes = (uint64_t)state->tap_set.tap_count * state->execution_row_capacity * state->tap_set.row_bytes;
	status = SparkGlm5NextAllocateBytes(state,(uint64_t)state->tap_set.tap_count * state->execution_row_capacity,state->tap_set.row_elements,sizeof(uint16_t),(void **)&state->slots[0].tap_device);
	for (index=0u; status == SPARK_STATUS_OK && index<state->pipeline_slot_count; index++)
	{
		state->slots[index].tap_device = state->slots[0].tap_device;
		if ( cudaHostAlloc((void **)&state->slots[index].tap_host,bytes,cudaHostAllocPortable) != cudaSuccess )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	state->tap_scratch = status == SPARK_STATUS_OK ? (uint8_t *)malloc(state->tap_set.record_bytes) : 0;
	if ( status == SPARK_STATUS_OK && (state->tap_scratch == 0 || pthread_mutex_init(&state->tap_lock,0) != 0) )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state->tap_lock_ready = 1u;
	clock_gettime(CLOCK_REALTIME,&now);
	state->tap_generation = ((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec) | 1u;
	if ( dump != 0 )
	{
		status = SparkSpeculationTapDumpOpen(&state->tap_dump,dump,&state->tap_set,"glm5_next",state->tap_generation,state->tp_rank,max_bytes);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"GLM tap dump %s cannot be created (it must not exist)\n",dump);
			SPARK_RETURN(status);
		}
		state->tap_dump_open = 1u;
	}
	if ( local != 0 )
	{
		state->tap_remote = (SparkSpeculationRelayRemote *)calloc(1u,sizeof(*state->tap_remote));
		if ( state->tap_remote == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		status = SparkSpeculationRelayLinkOpen(&state->tap_link,local,peer);
		if ( status == SPARK_STATUS_OK )
			status = SparkSpeculationRelayRemoteInitialize(state->tap_remote,&state->tap_link,state->tap_generation,SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,await_us * 1000u,1u);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"GLM tap relay %s>%s cannot open\n",local,peer);
			SparkSpeculationRelayLinkClose(&state->tap_link);
			free(state->tap_remote);
			state->tap_remote = 0;
			SPARK_RETURN(status);
		}
		fprintf(stderr,"GLM tap relay open send_buffer=%u receive_buffer=%u\n",state->tap_link.send_buffer_bytes,state->tap_link.receive_buffer_bytes);
	}
	state->tap_enabled = 1u;
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextReleaseTaps(SparkGlm5NextModuleState *state)
{
	uint32_t index;
	if ( state->tap_enabled != 0u )
		SparkGlm5NextTapReport(state,"TAP-SUMMARY");
	state->tap_enabled = 0u;
	if ( state->tap_dump_open != 0u )
	{
		if ( SparkSpeculationTapDumpClose(&state->tap_dump) != SPARK_STATUS_OK )
			fprintf(stderr,"TAP-DUMP-CLOSE-FAILED records=%llu\n",(unsigned long long)state->tap_dump.records);
		state->tap_dump_open = 0u;
	}
	if ( state->tap_remote != 0 )
	{
		SparkSpeculationRelayLinkClose(&state->tap_link);
		free(state->tap_remote);
		state->tap_remote = 0;
	}
	for (index=0u; index<state->pipeline_slot_count; index++)
	{
		if ( state->slots[index].tap_host != 0 )
			(void)cudaFreeHost(state->slots[index].tap_host);
		state->slots[index].tap_host = 0;
		state->slots[index].tap_device = 0;
	}
	free(state->tap_scratch);
	state->tap_scratch = 0;
	if ( state->tap_lock_ready != 0u )
		pthread_mutex_destroy(&state->tap_lock);
	state->tap_lock_ready = 0u;
}

static uint32_t SparkGlm5NextWalkLayer(SparkGlm5NextTpChain *chain,uint32_t layer)
{
	uint32_t reduce;
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextCudaWave *wave = &chain->wave;
	uint32_t gather_sequences = SparkGlm5NextLayerIndexGatherSequences(wave,layer);
	if ( gather_sequences != 0u && (SparkGlm5NextLaunchCudaLayerAttentionScore(wave,layer) != 0 || SparkGlm5NextModuleGatherIndex(chain,gather_sequences,0u) != SPARK_STATUS_OK || SparkGlm5NextLaunchCudaLayerAttentionSelect(wave,layer) != 0) )
		return(15u);
	if ( gather_sequences == 0u && SparkGlm5NextLaunchCudaLayerAttention(wave,layer) != 0 )
		return(4u);
	if ( SparkGlm5NextLayerKvShardActive(wave,layer) != 0u )
	{
		if ( SparkGlm5NextModuleKvShardExchange(chain,0u,0u) != SPARK_STATUS_OK )
			return(24u);
		if ( SparkGlm5NextLaunchCudaLayerAttentionShardPartial(wave,layer) != 0 )
			return(25u);
		if ( SparkGlm5NextModuleKvShardExchange(chain,1u,0u) != SPARK_STATUS_OK )
			return(26u);
		if ( SparkGlm5NextLaunchCudaLayerAttentionShardMerge(wave,layer) != 0 )
			return(27u);
	}
	reduce = SparkGlm5NextWalkReduce(chain,wave->slot->attention_out_bf16,0u,layer,SPARK_GLM5_NEXT_L2_SITE_ATTENTION_REDUCE);
	if ( reduce != 0u )
		return(reduce == 1u ? 5u : reduce);
	if ( SparkGlm5NextLaunchCudaLayerAttentionPost(wave,layer) != 0 )
		return(6u);
	if ( (wave->first_layer_index + layer) >= SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER )
	{
		if ( SparkGlm5NextLaunchCudaLayerMlpRouteResident(wave,layer) != 0 )
			return(7u);
		if ( SparkGlm5NextLaunchCudaLayerMlpExperts(wave,layer) != 0 )
			return(8u);
	}
	else if ( SparkGlm5NextLaunchCudaLayerMlp(wave,layer) != 0 )
		return(9u);
	reduce = SparkGlm5NextWalkReduce(chain,wave->slot->attention_out_bf16,0u,layer,SPARK_GLM5_NEXT_L2_SITE_MLP_REDUCE);
	if ( reduce != 0u )
		return(reduce == 1u ? 10u : reduce);
	if ( SparkGlm5NextLaunchCudaLayerMlpPost(wave,layer) != 0 )
		return(11u);
	if ( SparkGlm5NextTapCapture(chain,layer) != 0u )
		return(40u);
	if ( state->graph_record_limit != 0u && (state->graph_record_ops += 2u) >= state->graph_record_limit )
		state->graph_record_stop = 1u;
	return(0u);
}

static uint32_t SparkGlm5NextWalkChain(SparkGlm5NextTpChain *chain,uint32_t *layer_out)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextCudaWave *wave = &chain->wave;
	cudaStream_t stream = (cudaStream_t)wave->slot->stream;
	uint32_t layer,site;
	*layer_out = 0u;
	if ( wave->expert_cover != 0 && cudaMemsetAsync(wave->expert_miss,0,2u * sizeof(uint32_t),stream) != cudaSuccess )
		return(17u);
	if ( SparkGlm5NextLaunchCudaWaveBegin(wave) != 0 )
		return(2u);
	if ( wave->expert_cover != 0 && SparkGlm5NextWsSnapshot(wave,0u) != cudaSuccess )
		return(18u);
	site = SparkGlm5NextWalkReduce(chain,wave->slot->hidden_bf16,1u,0u,SPARK_GLM5_NEXT_L2_SITE_BEGIN);
	if ( site != 0u )
		return(site == 1u ? 3u : site);
	for (layer=0u; layer<wave->layer_count && state->graph_record_stop == 0u; layer++)
	{
		*layer_out = layer;
		site = SparkGlm5NextWalkLayer(chain,layer);
		if ( site != 0u )
			return(site);
	}
	*layer_out = layer;
	if ( SparkGlm5NextLaunchCudaWaveHead(wave) != 0 )
		return(12u);
	if ( wave->expert_cover != 0 && SparkGlm5NextLaunchHeadMissPoison(stream,(const uint32_t *)wave->expert_miss,wave->slot->head_maxloc_u64,wave->row_count) != cudaSuccess )
		return(19u);
	if ( SparkGlm5NextGraphReduceHead(chain) != SPARK_STATUS_OK )
		return(13u);
	if ( SparkGlm5NextLaunchHeadMaxlocUnpack(stream,wave->slot->head_maxloc_u64,wave->slot->output_token,wave->row_count) != cudaSuccess )
		return(14u);
	if ( state->owns_final_head != 0u && cudaMemcpyAsync(wave->slot->host_output_token_ids + chain->first_row,wave->slot->output_token,(uint64_t)wave->row_count * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream) != cudaSuccess )
		return(16u);
	if ( SparkGlm5NextTapFlush(chain) != 0u )
		return(41u);
	return(0u);
}

static void SparkGlm5NextGraphRecord(SparkGlm5NextTpChain *chain,void **exec_out)
{
	SparkGlm5NextModuleState *state = chain->state;
	cudaStream_t stream = (cudaStream_t)chain->wave.slot->stream;
	cudaGraph_t graph = 0;
	cudaGraphExec_t exec;
	uint32_t layer,site;
	*exec_out = 0;
	state->graph_record_ops = 0u;
	state->graph_record_stop = 0u;
	if ( cudaStreamBeginCapture(stream,0u) != cudaSuccess )
		return;
	site = SparkGlm5NextWalkChain(chain,&layer);
	if ( cudaStreamEndCapture(stream,&graph) != cudaSuccess || graph == 0 || site != 0u )
	{
		fprintf(stderr,"GRAPH-RECORD-FAIL failed=%u site=%u layer=%u end_graph=%p pending=%s\n",site != 0u ? 1u : 0u,(unsigned)site,(unsigned)layer,(void *)graph,cudaGetErrorString(cudaGetLastError()));
		if ( graph != 0 )
			(void)cudaGraphDestroy(graph);
		return;
	}
	if ( cudaGraphInstantiate(&exec,graph,0) != cudaSuccess || cudaGraphUpload(exec,stream) != cudaSuccess )
	{
		fprintf(stderr,"GRAPH-INSTANTIATE-FAIL inst=%s upload=%s\n",cudaGetErrorString(cudaGetLastError()),cudaGetErrorString(cudaGetLastError()));
		(void)cudaGraphDestroy(graph);
		return;
	}
	(void)cudaGraphDestroy(graph);
	*exec_out = exec;
}

#ifdef SPARK_SCORE_DUMP
typedef struct SparkGlm5NextScore
{
	SparkScoreDumpWriter writer;
	float *logits;
	SparkScoreDumpStats *stats;
	uint32_t *probe_offsets;
	uint32_t *probe_local;
	float *probe_logits;
	SparkScoreDumpStats *host_stats;
	uint32_t *host_offsets;
	uint32_t *host_local;
	uint32_t *host_ids;
	float *host_probe_logits;
	float *host_tier2;
	uint64_t *host_keys;
	uint32_t *host_flags;
	pthread_mutex_t lock;
	uint64_t wave_ordinal;
	uint32_t rows_capacity;
	uint32_t width;
	uint32_t id_capacity;
} SparkGlm5NextScore;

static void SparkGlm5NextScoreRelease(SparkGlm5NextScore *score)
{
	if ( score == 0 )
		return;
	(void)cudaFree(score->logits);
	(void)cudaFree(score->stats);
	(void)cudaFree(score->probe_offsets);
	(void)cudaFree(score->probe_local);
	(void)cudaFree(score->probe_logits);
	free(score->host_stats);
	free(score->host_offsets);
	free(score->host_local);
	free(score->host_ids);
	free(score->host_probe_logits);
	free(score->host_tier2);
	free(score->host_keys);
	free(score->host_flags);
	(void)pthread_mutex_destroy(&score->lock);
	free(score);
}

static SparkStatus SparkGlm5NextScoreOpen(SparkGlm5NextModuleState *state,const SparkGlm5NextResidentDecodeStageNodeContext *context)
{
	SparkScoreDumpConfig config;
	SparkGlm5NextScore *score;
	SparkStatus status;
	uint64_t rows,width,ids;
	if ( context->score_dump_directory == 0 )
	{
		if ( context->score_probe_path != 0 || context->score_tier2_rows_path != 0 )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		return(SPARK_STATUS_OK);
	}
	if ( state->owns_final_head == 0u || state->owns_embedding == 0u || state->lm_head_bf16 == 0 )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	score = (SparkGlm5NextScore *)calloc(1u,sizeof(*score));
	if ( score == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( pthread_mutex_init(&score->lock,0) != 0 )
	{
		free(score);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	memset(&config,0,sizeof(config));
	config.directory = context->score_dump_directory;
	config.probe_path = context->score_probe_path;
	config.tier2_path = context->score_tier2_rows_path;
	config.tp_rank = state->tp_rank;
	config.tp_degree = state->tp_degree;
	config.vocabulary = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT;
	config.hidden_dimension = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	config.slot_count = state->resident_sequence_capacity;
	config.position_count = state->max_sequence_positions;
	status = SparkScoreDumpOpen(&config,&score->writer);
	if ( status != SPARK_STATUS_OK )
	{
		(void)pthread_mutex_destroy(&score->lock);
		free(score);
		fprintf(stderr,"SCORE-DUMP open failed status=%d directory=%s\n",(int)status,context->score_dump_directory);
		SPARK_RETURN(status);
	}
	rows = state->execution_row_capacity;
	width = score->writer.header.shard_end - score->writer.header.shard_begin;
	ids = rows * (score->writer.probes.id_width != 0u ? score->writer.probes.id_width : 1u);
	score->rows_capacity = (uint32_t)rows;
	score->width = (uint32_t)width;
	score->id_capacity = (uint32_t)ids;
	if ( cudaMalloc((void **)&score->logits,rows * width * sizeof(float)) != cudaSuccess ||
	     cudaMalloc((void **)&score->stats,rows * sizeof(SparkScoreDumpStats)) != cudaSuccess ||
	     cudaMalloc((void **)&score->probe_offsets,(rows + 1u) * sizeof(uint32_t)) != cudaSuccess ||
	     cudaMalloc((void **)&score->probe_local,ids * sizeof(uint32_t)) != cudaSuccess ||
	     cudaMalloc((void **)&score->probe_logits,ids * sizeof(float)) != cudaSuccess )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	score->host_stats = (SparkScoreDumpStats *)calloc(rows,sizeof(SparkScoreDumpStats));
	score->host_offsets = (uint32_t *)calloc(rows + 1u,sizeof(uint32_t));
	score->host_local = (uint32_t *)calloc(ids,sizeof(uint32_t));
	score->host_ids = (uint32_t *)calloc(ids,sizeof(uint32_t));
	score->host_probe_logits = (float *)calloc(ids,sizeof(float));
	score->host_keys = (uint64_t *)calloc(rows,sizeof(uint64_t));
	score->host_flags = (uint32_t *)calloc(rows,sizeof(uint32_t));
	if ( score->writer.header.tier2 != 0u )
		score->host_tier2 = (float *)calloc(rows * width,sizeof(float));
	if ( score->host_stats == 0 || score->host_offsets == 0 || score->host_local == 0 || score->host_ids == 0 || score->host_probe_logits == 0 || score->host_keys == 0 || score->host_flags == 0 || (score->writer.header.tier2 != 0u && score->host_tier2 == 0) )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status != SPARK_STATUS_OK )
	{
		SparkScoreDumpFail(&score->writer);
		(void)SparkScoreDumpClose(&score->writer);
		SparkGlm5NextScoreRelease(score);
		SPARK_FAIL(status);
	}
	fprintf(stderr,"SCORE-DUMP open rank=%u shard=[%u,%u) rows=%s probes=%llu tier2=%llu\n",
	    state->tp_rank,score->writer.header.shard_begin,score->writer.header.shard_end,score->writer.rows_path,
	    (unsigned long long)score->writer.probes.entry_count,(unsigned long long)score->writer.tier2_rows.entry_count);
	state->score = score;
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextScoreClose(SparkGlm5NextModuleState *state)
{
	SparkGlm5NextScore *score = state->score;
	SparkStatus status;
	if ( score == 0 )
		return;
	status = SparkScoreDumpClose(&score->writer);
	fprintf(stderr,"SCORE-DUMP close rank=%u status=%d rows=%llu waves=%llu skipped=%llu keyless=%llu\n",
	    state->tp_rank,(int)status,(unsigned long long)score->writer.end.row_count,(unsigned long long)score->writer.end.wave_count,
	    (unsigned long long)score->writer.end.skipped_wave_count,(unsigned long long)score->writer.end.keyless_row_count);
	SparkGlm5NextScoreRelease(score);
	state->score = 0;
}

static void SparkGlm5NextScoreSkip(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextScore *score = chain->state->score;
	if ( score == 0 )
		return;
	(void)pthread_mutex_lock(&score->lock);
	SparkScoreDumpNoteWave(&score->writer,1u);
	(void)pthread_mutex_unlock(&score->lock);
}

static SparkStatus SparkGlm5NextScorePlan(SparkGlm5NextScore *score,const SparkGlm5NextExecutionSlot *slot,uint32_t first,uint32_t rows,uint32_t *probe_count)
{
	const uint32_t *ids;
	uint32_t row,index,count,id_count,position;
	uint64_t key;
	count = 0u;
	for (row=0u; row<rows; row++)
	{
		position = slot->host_positions[first + row];
		key = 0u;
		score->host_flags[row] = SparkScoreDumpKeysAdvance(&score->writer.keys,slot->host_resident_slots[first + row],position,slot->host_token_ids[first + row],&key) != 0u ? SPARK_SCORE_DUMP_ROW_KEY_VALID : 0u;
		score->host_keys[row] = key;
		score->host_offsets[row] = count;
		id_count = 0u;
		ids = score->host_flags[row] != 0u ? SparkScoreDumpTableFind(&score->writer.probes,key,position,&id_count) : 0;
		if ( ids != 0 )
			score->host_flags[row] |= SPARK_SCORE_DUMP_ROW_PROBED;
		if ( score->host_flags[row] != 0u && score->host_tier2 != 0 && SparkScoreDumpTableFind(&score->writer.tier2_rows,key,position,0) != 0 )
			score->host_flags[row] |= SPARK_SCORE_DUMP_ROW_TIER2;
		for (index=0u; ids != 0 && index<id_count; index++)
		{
			if ( ids[index] < score->writer.header.shard_begin || ids[index] >= score->writer.header.shard_end )
				continue;
			if ( count >= score->id_capacity )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			score->host_ids[count] = ids[index];
			score->host_local[count] = ids[index] - score->writer.header.shard_begin;
			count++;
		}
	}
	score->host_offsets[rows] = count;
	*probe_count = count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextScoreWrite(SparkGlm5NextScore *score,const SparkGlm5NextExecutionSlot *slot,uint32_t first,uint32_t rows)
{
	SparkScoreDumpRowRecord record;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t row;
	for (row=0u; status == SPARK_STATUS_OK && row<rows; row++)
	{
		memset(&record,0,sizeof(record));
		record.record_kind = SPARK_SCORE_DUMP_RECORD_ROW;
		record.flags = score->host_flags[row] | score->host_stats[row].flags;
		record.key = score->host_keys[row];
		record.wave_ordinal = score->wave_ordinal;
		record.position = slot->host_positions[first + row];
		record.row_in_wave = row;
		record.input_token = slot->host_token_ids[first + row];
		record.served_token = slot->host_output_token_ids[first + row];
		record.probe_count = score->host_offsets[row + 1u] - score->host_offsets[row];
		record.local_max = score->host_stats[row].local_max;
		record.local_sum_exp = score->host_stats[row].local_sum_exp;
		memcpy(record.top_ids,score->host_stats[row].top_ids,sizeof(record.top_ids));
		memcpy(record.top_logits,score->host_stats[row].top_logits,sizeof(record.top_logits));
		if ( (record.flags & SPARK_SCORE_DUMP_ROW_TIER2) != 0u )
			status = SparkScoreDumpWriteTier2(&score->writer,record.key,record.position,score->host_tier2 + (uint64_t)row * score->width);
		if ( status == SPARK_STATUS_OK )
			status = SparkScoreDumpWriteRow(&score->writer,&record,score->host_ids + score->host_offsets[row],score->host_probe_logits + score->host_offsets[row]);
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextScoreWaveLocked(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextScore *score = state->score;
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	cudaStream_t stream = (cudaStream_t)slot->stream;
	cudaError_t error;
	SparkStatus status;
	uint32_t rows,first,count,row;
	if ( chain->spec_verify != 0u )
	{
		SparkScoreDumpNoteWave(&score->writer,1u);
		return(SPARK_STATUS_OK);
	}
	rows = chain->wave_rows;
	first = chain->first_row;
	if ( rows == 0u || rows > score->rows_capacity )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkGlm5NextScorePlan(score,slot,first,rows,&count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	error = cudaMemcpyAsync(score->probe_offsets,score->host_offsets,(uint64_t)(rows + 1u) * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && count != 0u )
		error = cudaMemcpyAsync(score->probe_local,score->host_local,(uint64_t)count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkGlm5NextLaunchHeadScore(stream,slot->normed_bf16,state->lm_head_bf16,score->logits,rows,score->width,score->writer.header.shard_begin,score->probe_offsets,score->probe_local,score->probe_logits,score->stats);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(score->host_stats,score->stats,(uint64_t)rows * sizeof(SparkScoreDumpStats),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess && count != 0u )
		error = cudaMemcpyAsync(score->host_probe_logits,score->probe_logits,(uint64_t)count * sizeof(float),cudaMemcpyDeviceToHost,stream);
	for (row=0u; error == cudaSuccess && row<rows; row++)
		if ( (score->host_flags[row] & SPARK_SCORE_DUMP_ROW_TIER2) != 0u )
			error = cudaMemcpyAsync(score->host_tier2 + (uint64_t)row * score->width,score->logits + (uint64_t)row * score->width,(uint64_t)score->width * sizeof(float),cudaMemcpyDeviceToHost,stream);
	status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"score_dump_launch");
	if ( status == SPARK_STATUS_OK && SparkGlm5NextBoundedStreamSync(state,slot->stream,UINT64_C(35000000000)) != 0 )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK && state->ws_enabled != 0u && SparkStepVerdictClassify(slot->host_output_token_ids + first,rows,slot->miss_ring != 0 ? slot->miss_ring[SPARK_STEP_MISS_FLAG] : 0u) != SPARK_STEP_VERDICT_COMMIT )
	{
		SparkScoreDumpNoteWave(&score->writer,1u);
		return(SPARK_STATUS_OK);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextScoreWrite(score,slot,first,rows);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	SparkScoreDumpNoteWave(&score->writer,0u);
	score->wave_ordinal++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextScoreWave(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextScore *score = chain->state->score;
	SparkStatus status;
	if ( score == 0 )
		return(SPARK_STATUS_OK);
	if ( pthread_mutex_lock(&score->lock) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkGlm5NextScoreWaveLocked(chain);
	if ( status != SPARK_STATUS_OK )
	{
		SparkScoreDumpFail(&score->writer);
		fprintf(stderr,"SCORE-DUMP wave failed rank=%u status=%d rows=%u; the dump is incomplete and gets no end record\n",chain->state->tp_rank,(int)status,chain->wave_rows);
	}
	(void)pthread_mutex_unlock(&score->lock);
	return(status);
}
#endif

static uint32_t SparkGlm5NextLinearEligible(const SparkGlm5NextTpChain *chain)
{
	const SparkGlm5NextModuleState *state = chain->state;
	if ( state->lazy_pack == 0 || state->tp_degree < 2u || state->tp_collective_disabled != 0u || state->mtp_enabled != 0u || state->graph_record_limit != 0u || chain->spec_verify != 0u || SparkGlm5NextT1Enabled() != 0 )
		return(0u);
	if ( state->tp_device_collective_initialized == 0u || state->tp_device_collective_hc_initialized == 0u )
		return(0u);
	if ( SparkTpDeviceCollectiveStreamOrdered(&state->tp_device_collective) == 0u || SparkTpDeviceCollectiveStreamOrdered(&state->tp_device_collective_hc) == 0u )
		return(0u);
	if ( state->ws_enabled != 0u )
		return(chain->wave_rows <= SPARK_GLM5_NEXT_WS_ROWS_MAX && chain->ws_force_eager == 0u ? 1u : 0u);
	return(SparkGlm5NextExpertsPinned(state));
}

static void SparkGlm5NextNoteWarm(SparkGlm5NextModuleState *state)
{
	if ( state->experts_warm != 0u )
		return;
	state->experts_warm = 1u;
	fprintf(stderr,"GRAPH-WARM experts resident after first eager chain\n");
}

static void SparkGlm5NextLinearChain(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextAsyncCompletion *async = &state->completions[chain->slot_index];
	SparkStatus status = SparkGlm5NextWeightdHealth(state);
	uint32_t site = 0u,layer = 0u;
	uint64_t walk_start = 0u;
	if ( status == SPARK_STATUS_OK && SparkGlm5NextBuildWave(chain) != SPARK_STATUS_OK )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextGraphClaimExperts(chain);
	if ( status == SPARK_STATUS_OK )
	{
		SparkGlm5NextStampLaunch(async);
		async->linear = 1u;
		walk_start = SparkGlm5NextNowNs();
		site = SparkGlm5NextWalkChain(chain,&layer);
		async->walk_ns += SparkGlm5NextNowNs() - walk_start;
		status = site == 0u ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	}
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"LINEAR-CHAIN-FAILED status=%d site=%u layer=%u\n",(int)status,site,layer);
		SparkGlm5NextTpChainFail(chain,status);
		return;
	}
#ifdef SPARK_SCORE_DUMP
	status = SparkGlm5NextScoreWave(chain);
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm5NextTpChainFail(chain,status);
		return;
	}
#endif
	SparkGlm5NextNoteWarm(state);
	SparkGlm5NextFinishChain(chain);
}

static SparkStatus SparkGlm5NextMissRingEnsure(SparkGlm5NextExecutionSlot *slot)
{
	if ( slot->miss_ring != 0 )
		return(SPARK_STATUS_OK);
	if ( cudaHostAlloc((void **)&slot->miss_ring,SPARK_GLM5_NEXT_MODEL_MISS_RING_BYTES,cudaHostAllocMapped) != cudaSuccess )
	{
		slot->miss_ring = 0;
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	memset(slot->miss_ring,0,SPARK_GLM5_NEXT_MODEL_MISS_RING_BYTES);
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextGraphStep(SparkGlm5NextTpChain *chain,
    SparkStatus *status_out)
{
	SparkGlm5NextModuleState *state;
	SparkStatus status;
	uint64_t main_error = 0ull,hc_error = 0ull;
	void *exec;
	state = chain->state;
	status = SparkGlm5NextWeightdHealth(state);
	if ( status == SPARK_STATUS_OK &&
	     state->tp_device_collective_initialized != 0u &&
	     SparkTpDeviceCollectiveGraphPreLaunch(
	         &state->tp_device_collective,chain->slot->stream) !=
	             SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK &&
	     state->tp_device_collective_hc_initialized != 0u &&
	     SparkTpDeviceCollectiveGraphPreLaunch(
	         &state->tp_device_collective_hc,chain->slot->stream) !=
	             SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK &&
	     state->tp_device_collective_initialized != 0u &&
	     SparkTpDeviceCollectiveGraphCancelSeed(
	         &state->tp_device_collective,chain->slot->stream) !=
	             SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK &&
	     state->tp_device_collective_hc_initialized != 0u &&
	     SparkTpDeviceCollectiveGraphCancelSeed(
	         &state->tp_device_collective_hc,chain->slot->stream) !=
	         SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		exec = chain->slot->graph_exec_a;
		SparkGlm5NextStampLaunch(&state->completions[chain->slot_index]);
		state->completions[chain->slot_index].graph = 1u;
		{
			cudaError_t pre_err = cudaGetLastError();
			cudaError_t launch_rc = cudaGraphLaunch(exec,chain->slot->stream);
			if ( launch_rc != cudaSuccess )
			{
				fprintf(stderr,"GRAPH-LAUNCH-ERR slot=%u rc=%s pre=%s\n",
					chain->slot_index,
					cudaGetErrorString(launch_rc),
					cudaGetErrorString(pre_err));
				status = SPARK_STATUS_IO_ERROR;
			}
		else
		{
			struct timespec replay_t0,replay_t1;
			clock_gettime(CLOCK_MONOTONIC,&replay_t0);
			cudaError_t poll;
			SparkStatus wait_status = SparkStageModuleCudaWaitFor(&state->stream_wait,
				(uint64_t)state->tp_device_collective.operation_timeout_milli * UINT64_C(1000000));
			poll = wait_status == SPARK_STATUS_OK ? cudaSuccess : wait_status == SPARK_STATUS_BUSY ? cudaErrorNotReady : cudaErrorUnknown;
			clock_gettime(CLOCK_MONOTONIC,&replay_t1);
			{
				uint64_t replay_ns = (uint64_t)(replay_t1.tv_sec - replay_t0.tv_sec) *
				    UINT64_C(1000000000) +
				    (uint64_t)(replay_t1.tv_nsec - replay_t0.tv_nsec);
				fprintf(stderr,
				    "GRAPH-REPLAY-TIME slot=%u wall_ns=%llu stream_status=%d\n",
				    chain->slot_index,
				    (unsigned long long)replay_ns,(int)poll);
			}
			if ( poll == cudaErrorNotReady )
			{
				uint64_t stuck_error;
				uint64_t stuck_diag;
					uint64_t stuck_cell = 0ull;
					uint64_t stuck_progress = SparkTpDeviceCollectiveGraphProgress(
						&state->tp_device_collective,&stuck_cell);
					(void)SparkTpDeviceCollectiveGraphStuckDump(
						&state->tp_device_collective);
				stuck_error = SparkTpDeviceCollectiveGraphError(
					&state->tp_device_collective);
				stuck_diag = SparkTpDeviceCollectiveGraphDiag(
					&state->tp_device_collective);
				state->graph_path_enabled = 0u;
				state->degrade_graph_stuck++;
				SparkTpDeviceCollectiveBroadcastCancel(
					&state->tp_device_collective);
				if ( state->tp_device_collective_hc_initialized != 0u )
					SparkTpDeviceCollectiveBroadcastCancel(
						&state->tp_device_collective_hc);
				fprintf(stderr,
					"GRAPH-CANCEL-BROADCAST slot=%u\n",
					chain->slot_index);
				fprintf(stderr,
					"GRAPH-FAILED graph-stuck slot=%u progress=%llu cell=%llu err=%llu diag_peer=%llu ring=%llu slotidx=%llu want=%llu got=%llu\n",
					chain->slot_index,
					(unsigned long long)stuck_progress,
						(unsigned long long)stuck_cell,
						(unsigned long long)stuck_error,
					(unsigned long long)(stuck_diag >> 56),
					(unsigned long long)((stuck_diag >> 48) & 0xff),
					(unsigned long long)((stuck_diag >> 32) & 0xffff),
					(unsigned long long)((stuck_diag >> 16) & 0xffff),
					(unsigned long long)(stuck_diag & 0xffff));
				status = SPARK_STATUS_INTERNAL_ERROR;
			}
			else if ( poll != cudaSuccess )
			{
				fprintf(stderr,"GRAPH-STREAM-ERR slot=%u cuda=%s\n",
					chain->slot_index,
					cudaGetErrorString(poll));
				(void)cudaGetLastError();
				status = SPARK_STATUS_IO_ERROR;
			}
			else if ( state->tp_device_collective_initialized != 0u )
			{
				(void)SparkTpDeviceCollectiveGraphSettle(
				    &state->tp_device_collective,chain->slot->stream,&main_error);
				if ( state->tp_device_collective_hc_initialized != 0u )
					(void)SparkTpDeviceCollectiveGraphSettle(
					    &state->tp_device_collective_hc,chain->slot->stream,&hc_error);
			}
		}
	}
	}
	if ( status == SPARK_STATUS_OK )
	{
		uint64_t graph_error;
		graph_error = main_error != 0ull ? main_error : hc_error;
		if ( graph_error != 0ull )
		{
			state->graph_path_enabled = 0u;
			state->degrade_graph_stuck++;
			SparkTpDeviceCollectiveBroadcastCancel(
				&state->tp_device_collective);
			if ( state->tp_device_collective_hc_initialized != 0u )
				SparkTpDeviceCollectiveBroadcastCancel(
					&state->tp_device_collective_hc);
			fprintf(stderr,
				"GRAPH-CANCEL-BROADCAST slot=%u\n",
				chain->slot_index);
			fprintf(stderr,
				"GRAPH-FAILED graph-wait-timeout slot=%u seq=%llu\n",
				chain->slot_index,
				(unsigned long long)graph_error);
			status = SPARK_STATUS_INTERNAL_ERROR;
		}
		else
		{
			if ( state->graph_arrival_dumped == 0u )
			{
				state->graph_arrival_dumped = 1u;
				(void)SparkTpDeviceCollectiveGraphArrivalDump(
					&state->tp_device_collective,
					state->tp_rank);
			}
		}
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextStepVerdictApply(chain);
	*status_out = status;
}

static SparkStatus SparkGlm5NextGraphArm(
    SparkGlm5NextModuleState *state)
{
	if ( SparkTpDeviceCollectiveArmCapture(
	         &state->tp_device_collective) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->tp_device_collective_hc_initialized != 0u &&
	     SparkTpDeviceCollectiveArmCapture(
	         &state->tp_device_collective_hc) != SPARK_STATUS_OK )
		return(SPARK_STATUS_INTERNAL_ERROR);
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextGraphDisarm(
    SparkGlm5NextModuleState *state)
{
	(void)SparkTpDeviceCollectiveDisarmCapture(
		&state->tp_device_collective);
	if ( state->tp_device_collective_hc_initialized != 0u )
		(void)SparkTpDeviceCollectiveDisarmCapture(
			&state->tp_device_collective_hc);
}

static SparkStatus SparkGlm5NextGraphCaptureRows(SparkGlm5NextTpChain *chain,uint32_t regime,uint32_t index,uint32_t bound)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	void *exec = 0;
	chain->wave.maximum_context = bound;
	if ( SparkGlm5NextGraphArm(state) != SPARK_STATUS_OK )
	{
		SparkGlm5NextGraphDisarm(state);
		slot->graph_disabled = 1u;
		state->degrade_graph_disabled++;
		fprintf(stderr,"GRAPH-ARM-FAILED slot=%u rows=%u\n",chain->slot_index,index + 1u);
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	state->l2_prefetch_rounds = 0u;
	SparkGlm5NextGraphRecord(chain,&exec);
	SparkGlm5NextGraphDisarm(state);
	if ( exec == 0 )
	{
		fprintf(stderr,"GRAPH-CAPTURE-FAILED rows=%u; this row count runs eager from now on\n",index + 1u);
		slot->graph_failed_rows |= UINT64_C(1) << index;
		return(SPARK_STATUS_UNSUPPORTED);
	}
	slot->graph_exec_rows[regime][index] = exec;
	slot->graph_bound_rows[regime][index] = bound;
	fprintf(stderr,"GRAPH-CAPTURE-OK rows=%u regime=%u bound=%u l2_prefetch_rounds=%llu\n",index + 1u,regime,bound,(unsigned long long)state->l2_prefetch_rounds);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextGraphCapture(SparkGlm5NextTpChain *chain,uint32_t regime,uint32_t index,uint32_t bound)
{
	SparkGlm5NextAsyncCompletion *async = &chain->state->completions[chain->slot_index];
	uint64_t started_ns = SparkGlm5NextNowNs();
	SparkStatus status = SparkGlm5NextGraphCaptureRows(chain,regime,index,bound);
	async->captures++;
	async->capture_ns += SparkGlm5NextNowNs() - started_ns;
	return(status);
}

static void SparkGlm5NextGraphEnsure(SparkGlm5NextTpChain *chain,
    SparkStatus *status_out)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	SparkStatus status;
	uint32_t bound,regime,index = chain->wave_rows - 1u;
	chain->step_verdict = SPARK_STEP_VERDICT_COMMIT;
	if ( slot->graph_disabled != 0u )
	{
		fprintf(stderr,"GRAPH-DISABLED slot=%u after a failed capture arm\n",chain->slot_index);
		*status_out = SPARK_STATUS_INTERNAL_ERROR;
		return;
	}
	status = SparkGlm5NextGraphClaimExperts(chain);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextMissRingEnsure(slot);
	if ( status != SPARK_STATUS_OK )
	{
		*status_out = status;
		return;
	}
	regime = SparkGlm5NextGraphRegime(chain->wave.maximum_context,state->decode_split_context_threshold);
	bound = SparkGlm5NextGraphBound(chain->wave.maximum_context,state->decode_split_context_threshold,chain->wave.max_sequence_positions);
	if ( slot->graph_exec_rows[regime][index] != 0 && chain->wave.maximum_context > slot->graph_bound_rows[regime][index] )
	{
		(void)cudaGraphExecDestroy((cudaGraphExec_t)slot->graph_exec_rows[regime][index]);
		slot->graph_exec_rows[regime][index] = 0;
	}
	if ( slot->graph_exec_rows[regime][index] == 0 )
		status = SparkGlm5NextGraphCapture(chain,regime,index,bound);
	if ( slot->graph_exec_rows[regime][index] == 0 )
	{
		*status_out = status != SPARK_STATUS_OK ? status : SPARK_STATUS_INTERNAL_ERROR;
		return;
	}
	slot->graph_exec_a = slot->graph_exec_rows[regime][index];
	SparkGlm5NextGraphStep(chain,&status);
	*status_out = status;
}

typedef struct SparkGlm5NextVerifySaved
{
	uint32_t token_ids[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX];
	uint32_t resident_slots[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX];
	uint32_t positions[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX];
	const SparkGlm5NextResidentDecodeStageBatchView *batch;
	uint32_t first_row;
	uint32_t wave_rows;
	uint32_t spec_verify;
	uint32_t sampled;
	uint32_t tp_op_index;
	uint32_t tp_hc_op_index;
} SparkGlm5NextVerifySaved;

static void SparkGlm5NextVerifySave(const SparkGlm5NextTpChain *chain,SparkGlm5NextVerifySaved *saved)
{
	const SparkGlm5NextExecutionSlot *slot = chain->slot;
	memcpy(saved->token_ids,slot->host_token_ids,sizeof(saved->token_ids));
	memcpy(saved->resident_slots,slot->host_resident_slots,sizeof(saved->resident_slots));
	memcpy(saved->positions,slot->host_positions,sizeof(saved->positions));
	saved->batch = chain->batch;
	saved->first_row = chain->first_row;
	saved->wave_rows = chain->wave_rows;
	saved->spec_verify = chain->spec_verify;
	saved->sampled = slot->sampled;
	saved->tp_op_index = chain->tp_op_index;
	saved->tp_hc_op_index = chain->tp_hc_op_index;
}

static void SparkGlm5NextVerifyRestore(SparkGlm5NextTpChain *chain,const SparkGlm5NextVerifySaved *saved)
{
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	memcpy(slot->host_token_ids,saved->token_ids,sizeof(saved->token_ids));
	memcpy(slot->host_resident_slots,saved->resident_slots,sizeof(saved->resident_slots));
	memcpy(slot->host_positions,saved->positions,sizeof(saved->positions));
	chain->batch = saved->batch;
	chain->first_row = saved->first_row;
	chain->wave_rows = saved->wave_rows;
	chain->spec_verify = saved->spec_verify;
	slot->sampled = saved->sampled;
	chain->tp_op_index = saved->tp_op_index;
	chain->tp_hc_op_index = saved->tp_hc_op_index;
}

static SparkStatus SparkGlm5NextVerifyCaptureRows(SparkGlm5NextTpChain *chain,uint32_t regime,uint32_t rows,uint32_t bound,uint32_t lane)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	uint32_t row,index = SparkGlm5NextVerifyTableIndex(rows);
	SparkStatus status;
	void *exec = 0;
	for (row=0u; row<rows; row++)
	{
		slot->host_token_ids[row] = 0u;
		slot->host_resident_slots[row] = lane;
		slot->host_positions[row] = bound - rows + row;
	}
	chain->wave_rows = rows;
	status = SparkGlm5NextBuildWave(chain);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextGraphClaimExperts(chain);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextGraphArm(state);
	if ( status == SPARK_STATUS_OK )
		SparkGlm5NextGraphRecord(chain,&exec);
	SparkGlm5NextGraphDisarm(state);
	if ( status == SPARK_STATUS_OK && exec == 0 )
		status = SPARK_STATUS_UNSUPPORTED;
	if ( status == SPARK_STATUS_OK && SparkGlm5NextPrepareCudaReplayFold(&chain->wave,rows) != 0 )
		status = SPARK_STATUS_INTERNAL_ERROR;
	fprintf(stderr,"GRAPH-VERIFY-CAPTURE rows=%u regime=%u bound=%u status=%d\n",rows,regime,bound,(int)status);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( slot->verify_exec[regime][index] != 0 )
		(void)cudaGraphExecDestroy((cudaGraphExec_t)slot->verify_exec[regime][index]);
	slot->verify_exec[regime][index] = exec;
	slot->verify_bound[regime][index] = bound;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextVerifyCaptureSet(SparkGlm5NextTpChain *chain,uint32_t regime,uint32_t context)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextResidentDecodeStageBatchView batch;
	SparkGlm5NextVerifySaved saved;
	SparkStatus status;
	uint32_t rows,bound;
	SparkGlm5NextVerifySave(chain,&saved);
	batch = *chain->batch;
	batch.row_count = SPARK_GLM5_NEXT_VERIFY_ROWS_MIN;
	batch.active_sequence_count = 1u;
	chain->batch = &batch;
	chain->first_row = 0u;
	chain->spec_verify = 1u;
	chain->slot->sampled = 0u;
	status = SparkGlm5NextMissRingEnsure(chain->slot);
	for (rows=SPARK_GLM5_NEXT_VERIFY_ROWS_MIN; status == SPARK_STATUS_OK && rows<=state->verify_rows_max; rows++)
	{
		bound = SparkGlm5NextVerifyCaptureBound(regime,rows,context,state->decode_split_context_threshold,state->max_sequence_positions);
		batch.row_count = rows;
		if ( bound != 0u )
			status = SparkGlm5NextVerifyCaptureRows(chain,regime,rows,bound,saved.resident_slots[0]);
		chain->tp_op_index = saved.tp_op_index;
		chain->tp_hc_op_index = saved.tp_hc_op_index;
	}
	SparkGlm5NextVerifyRestore(chain,&saved);
	return(status);
}

static SparkStatus SparkGlm5NextVerifyCaptureStart(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextAsyncCompletion *async = &chain->state->completions[chain->slot_index];
	uint64_t started_ns = SparkGlm5NextNowNs();
	SparkStatus status;
	status = SparkGlm5NextVerifyCaptureSet(chain,SPARK_GLM5_NEXT_GRAPH_REGIME_UNSPLIT,0u);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextVerifyCaptureSet(chain,SPARK_GLM5_NEXT_GRAPH_REGIME_SPLIT,0u);
	async->captures++;
	async->capture_ns += SparkGlm5NextNowNs() - started_ns;
	chain->slot->verify_captured = status == SPARK_STATUS_OK ? 1u : 0u;
	fprintf(stderr,"GRAPH-VERIFY-TABLE slot=%u rows_max=%u status=%d capture_ms=%llu\n",chain->slot_index,chain->state->verify_rows_max,(int)status,(unsigned long long)((SparkGlm5NextNowNs() - started_ns) / 1000000u));
	return(status);
}

static SparkStatus SparkGlm5NextVerifyLookup(SparkGlm5NextTpChain *chain,void **exec_out)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	uint32_t context,regime,index;
	SparkStatus status;
	*exec_out = 0;
	status = SparkGlm5NextVerifyWaveCheck(chain->wave_rows,state->verify_rows_max,chain->first_row,chain->batch->active_sequence_count,slot->sampled,slot->host_resident_slots,slot->host_positions,state->decode_split_context_threshold,state->max_sequence_positions);
	if ( status == SPARK_STATUS_OK && chain->steps != 1u )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->graph_path_enabled == 0u || slot->verify_captured == 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	context = slot->host_positions[chain->wave_rows - 1u] + 1u;
	regime = SparkGlm5NextGraphRegime(context,state->decode_split_context_threshold);
	index = SparkGlm5NextVerifyTableIndex(chain->wave_rows);
	if ( regime == SPARK_GLM5_NEXT_GRAPH_REGIME_SELECTED && (slot->verify_exec[regime][index] == 0 || context > slot->verify_bound[regime][index]) )
		status = SparkGlm5NextVerifyCaptureSet(chain,regime,context);
	if ( status == SPARK_STATUS_OK && (slot->verify_exec[regime][index] == 0 || context > slot->verify_bound[regime][index]) )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( status == SPARK_STATUS_OK )
		*exec_out = slot->verify_exec[regime][index];
	return(status);
}

static SparkStatus SparkGlm5NextVerifyCommit(SparkGlm5NextTpChain *chain,uint32_t *more_out)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	SparkSpeculationPolicyVerifyResult result;
	SparkStatus status;
	uint32_t committed,index;
	*more_out = 0u;
	if ( chain->verify_budget == 0u || chain->verify_draft_count == 0u || chain->wave_rows != chain->verify_draft_count + 1u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( SparkGlm5NextBoundedStreamSync(state,slot->stream,UINT64_C(35000000000)) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkSpeculationPolicyResolveVerifierTokens(chain->verify_draft,chain->verify_draft_count,slot->host_output_token_ids,chain->wave_rows,SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,&result);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	committed = result.committed_token_count;
	if ( committed != result.accepted_draft_token_count + 1u || committed > chain->wave_rows || committed > chain->verify_budget - chain->verify_produced )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( SparkGlm5NextLaunchCudaReplayFold(&chain->wave,committed) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for (index=0u; index<committed; index++)
		chain->verify_tokens[chain->verify_produced + index] = slot->host_output_token_ids[index];
	SparkGlm5NextTapEmit(chain,committed,SPARK_SPECULATION_TAP_FLAG_VERIFY);
	status = SparkGlm5NextMtpTap(state,slot,committed - 1u,chain->verify_lane,chain->verify_sequence_id,chain->verify_position + committed);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( SparkGlm5NextVerifyDrafterUsesLookup(state->verify_drafter) != 0u )
	{
		status = SparkSpeculationLookupDraftObserve(&state->verify_lookup,chain->verify_lane,chain->verify_sequence_id,chain->verify_position + 1u,slot->host_output_token_ids,committed);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	if ( state->verify_drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP_LOOKUP )
	{
		status = SparkSpeculationDrafterMixObserve(&state->verify_mix,chain->verify_lane,chain->verify_draft_count,result.accepted_draft_token_count);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	state->verify_rounds++;
	state->verify_proposed += chain->verify_draft_count;
	state->verify_accepted += result.accepted_draft_token_count;
	state->verify_accept_depth[result.accepted_draft_token_count]++;
	for (index=0u; index<chain->verify_draft_count && index<=result.accepted_draft_token_count; index++)
	{
		state->verify_position_reached[index]++;
		if ( index < result.accepted_draft_token_count )
			state->verify_position_accepted[index]++;
	}
	state->verify_depth_cap[chain->verify_lane] = SparkSpeculationDepthCapNext(state->verify_depth_cap[chain->verify_lane],chain->verify_draft_count,result.accepted_draft_token_count,state->verify_rows_max - 1u);
	chain->verify_rounds++;
	chain->verify_accepted += result.accepted_draft_token_count;
	chain->verify_produced += committed;
	chain->verify_position += committed;
	chain->spec_verify = 0u;
	chain->verify_draft_count = 0u;
	chain->wave_rows = 1u;
	*more_out = SparkGlm5NextVerifyPlanRound(chain,chain->verify_tokens[chain->verify_produced - 1u],&status);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextVerifyPlainStep(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	SparkStatus status;
	slot->host_token_ids[0] = chain->verify_tokens[chain->verify_produced - 1u];
	slot->host_positions[0] = (uint32_t)chain->verify_position;
	slot->host_resident_slots[0] = chain->verify_lane;
	chain->wave_rows = 1u;
	chain->spec_verify = 0u;
	status = SparkGlm5NextBuildWave(chain);
	if ( status == SPARK_STATUS_OK )
		SparkGlm5NextGraphEnsure(chain,&status);
	if ( status == SPARK_STATUS_OK && SparkGlm5NextBoundedStreamSync(state,slot->stream,UINT64_C(35000000000)) != 0 )
		status = SPARK_STATUS_IO_ERROR;
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"VERIFY-PLAIN-STEP-FAILED position=%llu status=%d; committed rounds already folded, the engine stops\n",(unsigned long long)chain->verify_position,(int)status);
		return(SparkGlm5NextTerminalFailure(state,status,"verify-plain-step"));
	}
	chain->verify_tokens[chain->verify_produced] = slot->host_output_token_ids[0];
	SparkGlm5NextTapEmit(chain,1u,SPARK_SPECULATION_TAP_FLAG_DECODE);
	status = SparkGlm5NextMtpTap(state,slot,0u,chain->verify_lane,chain->verify_sequence_id,chain->verify_position + 1u);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( SparkGlm5NextVerifyDrafterUsesLookup(state->verify_drafter) != 0u )
	{
		status = SparkSpeculationLookupDraftObserve(&state->verify_lookup,chain->verify_lane,chain->verify_sequence_id,chain->verify_position + 1u,slot->host_output_token_ids,1u);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	chain->verify_produced++;
	chain->verify_position++;
	chain->verify_plain++;
	state->verify_plain_steps++;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextVerifyContinue(SparkGlm5NextTpChain *chain,uint32_t *more)
{
	SparkStatus status = SPARK_STATUS_OK;
	while ( *more == 0u && chain->verify_produced < chain->verify_budget )
	{
		status = SparkGlm5NextSettleStep(chain);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextVerifyPlainStep(chain);
		if ( status == SPARK_STATUS_OK )
			*more = SparkGlm5NextVerifyPlanRound(chain,chain->verify_tokens[chain->verify_produced - 1u],&status);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextVerifyFinish(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkGlm5NextExecutionSlot *slot = chain->slot;
	SparkGlm5NextAsyncCompletion *async = &state->completions[chain->slot_index];
	uint32_t produced = chain->verify_produced;
	uint32_t index;
	int length;
	char positions[512];
	if ( produced == 0u || produced > chain->verify_budget || async->lane_count != 1u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( SparkGlm5NextBoundedStreamSync(state,slot->stream,UINT64_C(35000000000)) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	memcpy(slot->host_output_token_ids,chain->verify_tokens,(uint64_t)produced * sizeof(uint32_t));
	async->burst_token_count = produced;
	async->cache_extra_tokens = produced - 1u;
	async->lane_next_positions[0] += produced - 1u;
	async->completion.tokens_per_sequence = produced;
	state->verify_tokens += produced;
	fprintf(stderr,"VERIFY-FRAME slot=%u position=%llu budget=%u produced=%u rounds=%u accepted=%u steps=%u | frames=%llu plain=%llu rounds=%llu proposed=%llu accepted=%llu steps=%llu tokens=%llu\n",
		chain->slot_index,(unsigned long long)(chain->verify_position - produced),chain->verify_budget,produced,chain->verify_rounds,chain->verify_accepted,chain->verify_plain,
		(unsigned long long)state->verify_frames,(unsigned long long)state->verify_plain_frames,(unsigned long long)state->verify_rounds,
		(unsigned long long)state->verify_proposed,(unsigned long long)state->verify_accepted,(unsigned long long)state->verify_plain_steps,(unsigned long long)state->verify_tokens);
	length = snprintf(positions,sizeof(positions),"VERIFY-POSITIONS");
	for (index=0u; index<SPARK_GLM5_NEXT_VERIFY_ROWS_MAX - 1u && length > 0 && (size_t)length < sizeof(positions); index++)
		length += snprintf(positions + length,sizeof(positions) - (size_t)length," p%u=%llu/%llu",index + 1u,
			(unsigned long long)state->verify_position_accepted[index],(unsigned long long)state->verify_position_reached[index]);
	fprintf(stderr,"%s\n",positions);
	if ( state->verify_mtp != 0u )
		fprintf(stderr,"VERIFY-MTP drafts=%llu tokens=%llu cold=%llu truncated=%llu taps=%llu draft_us=%llu | lookup rounds=%llu proposed=%llu accepted=%llu declined=%llu | mtp rounds=%llu proposed=%llu accepted=%llu\n",
			(unsigned long long)state->mtp_drafts,(unsigned long long)state->mtp_draft_tokens,(unsigned long long)state->mtp_cold,(unsigned long long)state->mtp_truncated,(unsigned long long)state->mtp_taps,
			(unsigned long long)(state->mtp_drafts != 0u ? state->mtp_draft_ns / state->mtp_drafts / 1000u : 0u),
			(unsigned long long)state->verify_mix.counters[SPARK_SPECULATION_DRAFTER_MIX_PRIMARY].rounds,(unsigned long long)state->verify_mix.counters[SPARK_SPECULATION_DRAFTER_MIX_PRIMARY].proposed,
			(unsigned long long)state->verify_mix.counters[SPARK_SPECULATION_DRAFTER_MIX_PRIMARY].accepted,(unsigned long long)state->verify_mix.counters[SPARK_SPECULATION_DRAFTER_MIX_PRIMARY].declined,
			(unsigned long long)(state->verify_drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP ? state->verify_rounds : state->verify_mix.counters[SPARK_SPECULATION_DRAFTER_MIX_FALLBACK].rounds),
			(unsigned long long)(state->verify_drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP ? state->verify_proposed : state->verify_mix.counters[SPARK_SPECULATION_DRAFTER_MIX_FALLBACK].proposed),
			(unsigned long long)(state->verify_drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP ? state->verify_accepted : state->verify_mix.counters[SPARK_SPECULATION_DRAFTER_MIX_FALLBACK].accepted));
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextVerifyWave(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkStatus status;
	uint32_t more;
	void *exec;
	status = SparkGlm5NextVerifyLookup(chain,&exec);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextBuildWave(chain);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextGraphClaimExperts(chain);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"GRAPH-VERIFY-REJECTED rows=%u status=%d\n",chain->wave_rows,(int)status);
		SparkGlm5NextTpChainFail(chain,status);
		return;
	}
	chain->slot->graph_exec_a = exec;
	SparkGlm5NextGraphStep(chain,&status);
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm5NextTerminalFailure(state,status,"verify-graph-execution");
		SparkGlm5NextTpChainFail(chain,status);
		return;
	}
	status = SparkGlm5NextVerifyCommit(chain,&more);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextVerifyContinue(chain,&more);
	if ( status == SPARK_STATUS_OK && more != 0u )
	{
		status = SparkGlm5NextSettleStep(chain);
		if ( status == SPARK_STATUS_OK )
		{
			chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN;
			chain->next_layer = 0u;
			SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
			return;
		}
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextVerifyFinish(chain);
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm5NextTpChainFail(chain,status);
		return;
	}
	SparkGlm5NextFinishChain(chain);
}

static uint32_t SparkGlm5NextVerifyRoute(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state = chain->state;
	SparkStatus status = SPARK_STATUS_OK;
	if ( chain->slot->verify_captured == 0u && state->graph_path_enabled != 0u && state->experts_warm != 0u )
		status = SparkGlm5NextVerifyCaptureStart(chain);
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm5NextTerminalFailure(state,status,"verify-capture");
		SparkGlm5NextTpChainFail(chain,status);
		return(1u);
	}
	if ( chain->spec_verify == 0u )
		return(0u);
	SparkGlm5NextVerifyWave(chain);
	return(1u);
}

#include "sparkpipe/family/module/spark_module_t1_enabled.h"

static void SparkGlm5NextT1Streams(SparkGlm5NextTpChain *chain,uint32_t layer)
{
	static uint16_t *rows_host = 0;
	static uint32_t rows_host_capacity = 0;
	uint32_t row;
	uint32_t i;
	uint32_t flat;
	uint64_t bytes;
	if ( SparkGlm5NextT1Enabled() == 0 || chain->wave.tp_rank != 0u )
		return;
	if ( SparkGlm5NextBoundedStreamSync(chain->state,chain->slot->stream,UINT64_C(35000000000)) != 0 )
		return;
	flat = SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	bytes = (uint64_t)chain->wave_rows * flat * sizeof(uint16_t);
	if ( rows_host_capacity < chain->wave_rows )
	{
		free(rows_host);
		rows_host = (uint16_t *)malloc(bytes);
		rows_host_capacity = rows_host != 0 ? chain->wave_rows : 0u;
	}
	if ( rows_host == 0 ||
	    cudaMemcpy(rows_host,chain->slot->hidden_bf16,bytes,cudaMemcpyDeviceToHost) != cudaSuccess )
		return;
	for ( row = 0u; row < chain->wave_rows; row++ )
	{
		uint16_t *values = rows_host + (uint64_t)row * flat;
		fprintf(stderr,"G5N-T1 stream L%u pos%u",layer,chain->wave.host_positions[row]);
		for ( i = 0u; i < flat; i++ )
			fprintf(stderr," %04x",values[i]);
		fputc('\n',stderr);
	}
}
static void SparkGlm5NextT1Route(SparkGlm5NextTpChain *chain,uint32_t layer)
{
	static uint32_t *ids_host = 0;
	static float *weights_host = 0;
	static uint32_t route_capacity = 0;
	uint32_t row;
	uint32_t k;
	uint64_t bytes;
	if ( SparkGlm5NextT1Enabled() == 0 || chain->wave.tp_rank != 0u ||
	    layer < SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER )
		return;
	bytes = (uint64_t)chain->wave_rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K * sizeof(uint32_t);
	if ( route_capacity < chain->wave_rows )
	{
		free(ids_host);
		free(weights_host);
		ids_host = (uint32_t *)malloc(bytes);
		weights_host = (float *)malloc(bytes);
		route_capacity = ids_host != 0 && weights_host != 0 ? chain->wave_rows : 0u;
	}
	if ( ids_host == 0 || weights_host == 0 ||
	    cudaMemcpy(ids_host,chain->slot->route_expert,bytes,cudaMemcpyDeviceToHost) != cudaSuccess ||
	    cudaMemcpy(weights_host,chain->slot->route_weight,bytes,cudaMemcpyDeviceToHost) != cudaSuccess )
		return;
	for ( row = 0u; row < chain->wave_rows; row++ )
	{
		fprintf(stderr,"G5N-T1 route L%u pos%u ids",layer,chain->wave.host_positions[row]);
		for ( k = 0u; k < SPARK_GLM5_NEXT_MODEL_MOE_TOP_K; k++ )
			fprintf(stderr," %u",ids_host[row * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K + k]);
		fprintf(stderr," weights");
		for ( k = 0u; k < SPARK_GLM5_NEXT_MODEL_MOE_TOP_K; k++ )
			fprintf(stderr," %08x",
			    ((const uint32_t *)weights_host)[row * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K + k]);
		fputc('\n',stderr);
	}
}
static void SparkGlm5NextT1Head(SparkGlm5NextTpChain *chain)
{
	static uint32_t *tokens_host = 0;
	static float *scores_host = 0;
	static uint32_t head_capacity = 0;
	uint32_t i;
	if ( SparkGlm5NextT1Enabled() == 0 || chain->wave.tp_rank != 0u ||
	    chain->state->owns_final_head == 0u )
		return;
	if ( SparkGlm5NextBoundedStreamSync(chain->state,chain->slot->stream,UINT64_C(35000000000)) != 0 )
		return;
	if ( head_capacity < chain->wave_rows )
	{
		free(tokens_host);
		free(scores_host);
		tokens_host = (uint32_t *)malloc((uint64_t)chain->wave_rows * sizeof(uint32_t));
		scores_host = (float *)malloc((uint64_t)chain->wave_rows * sizeof(float));
		head_capacity = tokens_host != 0 && scores_host != 0 ? chain->wave_rows : 0u;
	}
	if ( tokens_host == 0 || scores_host == 0 ||
	    cudaMemcpy(tokens_host,chain->slot->output_token,(uint64_t)chain->wave_rows * sizeof(uint32_t),cudaMemcpyDeviceToHost) != cudaSuccess ||
	    cudaMemcpy(scores_host,chain->slot->output_score,(uint64_t)chain->wave_rows * sizeof(float),cudaMemcpyDeviceToHost) != cudaSuccess )
		return;
	for ( i = 0u; i < chain->wave_rows; i++ )
		fprintf(stderr,"G5N-T1 head pos%u token %u score_bits %08x\n",
		    chain->wave.host_positions[i],tokens_host[i],
		    ((const uint32_t *)scores_host)[i]);
}

static uint32_t SparkGlm5NextGraphResult(SparkGlm5NextTpChain *chain,SparkStatus graph_status)
{
	if ( graph_status == SPARK_STATUS_OK )
	{
#ifdef SPARK_SCORE_DUMP
		SparkGlm5NextScoreSkip(chain);
#endif
		SparkGlm5NextFinishChain(chain);
		return(1u);
	}
	if ( chain->step_verdict != SPARK_STEP_VERDICT_COMMIT )
	{
		if ( SparkGlm5NextWsRetry(chain) == 0u )
			SparkGlm5NextTpChainFail(chain,graph_status);
		return(1u);
	}
	if ( graph_status == SPARK_STATUS_UNSUPPORTED )
		return(0u);
	SparkGlm5NextTerminalFailure(chain->state,graph_status,"graph-execution");
	fprintf(stderr,"GRAPH-PATH-FAILED status=%d; engine restart required\n",(int32_t)graph_status);
	SparkGlm5NextTpChainFail(chain,graph_status);
	return(1u);
}

static void SparkGlm5NextTpChainAdvance(void *chain_context,SparkStatus status)
{
	SparkGlm5NextTpChain *chain;
	SparkGlm5NextModuleState *state;
	SparkStatus launch_status;
	cudaError_t error;
	uint32_t gather_sequences;
	chain = (SparkGlm5NextTpChain *)chain_context;
	if ( chain == 0 || chain->active == 0u )
		return;
	state = chain->state;
	{
		uint64_t now_ns = SparkGlm5NextNowNs();
		chain->last_advance_ns = now_ns;
		state->slot_alive_ns[chain->slot_index] = now_ns;
		if ( chain->created_ns == 0ull )
			chain->created_ns = now_ns;
		else if ( now_ns - chain->created_ns >= UINT64_C(30000000000) &&
			chain->last_heartbeat_stage != (uint64_t)chain->stage + 1ull )
		{
			chain->last_heartbeat_stage = (uint64_t)chain->stage + 1ull;
			fprintf(stderr,
				"CHAIN-HEARTBEAT slot=%u stage=%u layer=%u age_ms=%llu lease=%llu begun=%u recorded=%u — chain still advancing (a SILENT gap between these = the lost-continuation site)\n",
				chain->slot_index,(unsigned)chain->stage,
				(unsigned)chain->next_layer,
				(unsigned long long)((now_ns - chain->created_ns) / 1000000ull),
				(unsigned long long)chain->expert_lease,
				(unsigned)chain->expert_lease_begun,
				(unsigned)chain->expert_lease_recorded);
		}
	}
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm5NextTpChainFail(chain,status);
		return;
	}
	{
		uint64_t now_ns = SparkGlm5NextNowNs();
		if ( state->chain_profile_last_ns != 0ull &&
		     state->chain_profile_stage < 8u )
			state->chain_stage_ns[state->chain_profile_stage] +=
				now_ns - state->chain_profile_last_ns;
		state->chain_profile_last_ns = now_ns;
		state->chain_profile_stage = (uint32_t)chain->stage;
	}
	{
		static uint32_t chain_trace_count;
		if ( chain_trace_count < 400u )
		{
			chain_trace_count++;
			fprintf(stderr,"CHAIN slot=%u stage=%u layer=%u rows=%u mi=%llu hi=%llu\n",
				chain->slot_index,chain->stage,chain->next_layer,chain->wave_rows,
				state->tp_device_collective_initialized != 0u ?
					(unsigned long long)SparkTpDeviceCollectiveRoundIndex(&state->tp_device_collective) : 0ull,
				state->tp_device_collective_hc_initialized != 0u ?
					(unsigned long long)SparkTpDeviceCollectiveRoundIndex(&state->tp_device_collective_hc) : 0ull);
		}
	}
	switch ( chain->stage )
	{
	case SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN:
		if ( state->graph_gate_printed < 3u )
		{
			state->graph_gate_printed++;
			fprintf(stderr,
			    "GRAPH-GATE rows=%u first=%u coll=%u lazy=%u deg=%u enabled=%u flags=%u\n",
			    (unsigned)chain->wave_rows,(unsigned)chain->first_row,
			    (unsigned)state->tp_device_collective_initialized,
			    (unsigned)(state->lazy_pack != 0),
			    (unsigned)state->tp_degree,
			    (unsigned)state->graph_path_enabled,
			    (unsigned)chain->context->flags);
		}
		if ( state->graph_path_enabled != 0u &&
		     state->experts_warm == 0u &&
		     state->graph_gate_printed < 3u )
			fprintf(stderr,
			    "GRAPH-GATE-COLD experts not warm; eager first\n");
		if ( state->verify_rows_max != 0u && SparkGlm5NextVerifyRoute(chain) != 0u )
			return;
		if ( chain->wave_rows >= 1u && chain->wave_rows <= SPARK_GLM5_NEXT_GRAPH_ROWS_MAX &&
		     (chain->slot->graph_failed_rows & (UINT64_C(1) << (chain->wave_rows - 1u))) == 0u &&
		     chain->first_row == 0u && chain->spec_verify == 0u &&
		     chain->slot->sampled == 0u &&
		     chain->batch->active_sequence_count == chain->wave_rows &&
		     state->tp_device_collective_initialized != 0u &&
		     state->lazy_pack != 0 && state->tp_degree > 1u &&
		     state->graph_path_enabled != 0u &&
		     state->experts_warm != 0u )
		{
			SparkStatus graph_status;
			if ( SparkGlm5NextBuildWave(chain) != SPARK_STATUS_OK )
			{
				SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INVALID_ARGUMENT);
				return;
			}
			SparkGlm5NextGraphEnsure(chain,&graph_status);
			if ( SparkGlm5NextGraphResult(chain,graph_status) != 0u )
				return;
		}
		if ( SparkGlm5NextLinearEligible(chain) != 0u )
		{
			SparkGlm5NextLinearChain(chain);
			return;
		}
		if ( SparkGlm5NextBuildWave(chain) != SPARK_STATUS_OK )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INVALID_ARGUMENT);
			return;
		}
		SparkGlm5NextT1Wave(&chain->wave);
		SparkGlm5NextStampLaunch(&state->completions[chain->slot_index]);
		if ( SparkGlm5NextLaunchCudaWaveBegin(&chain->wave) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION;
		chain->next_layer = 0u;
		launch_status = SparkGlm5NextModuleReduceHidden(chain,chain->slot->hidden_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION:
		gather_sequences = SparkGlm5NextLayerIndexGatherSequences(&chain->wave,chain->next_layer);
		if ( gather_sequences != 0u )
		{
			if ( SparkGlm5NextLaunchCudaLayerAttentionScore(&chain->wave,chain->next_layer) != 0 )
			{
				SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
				return;
			}
			chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_GATHER_INDEX;
			launch_status = SparkGlm5NextModuleGatherIndex(chain,gather_sequences,1u);
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm5NextTpChainFail(chain,launch_status);
			return;
		}
		if ( SparkGlm5NextLaunchCudaLayerAttention(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		if ( SparkGlm5NextLayerKvShardActive(&chain->wave,chain->next_layer) != 0u )
		{
			chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_SHARD_QUERY;
			launch_status = SparkGlm5NextModuleKvShardExchange(chain,0u,1u);
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm5NextTpChainFail(chain,launch_status);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION;
		launch_status = SparkGlm5NextModuleReduceAttentionOut(chain,chain->slot->attention_out_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_SHARD_QUERY:
		if ( SparkGlm5NextLaunchCudaLayerAttentionShardPartial(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_SHARD_EXCHANGE;
		launch_status = SparkGlm5NextModuleKvShardExchange(chain,1u,1u);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_SHARD_EXCHANGE:
		if ( SparkGlm5NextLaunchCudaLayerAttentionShardMerge(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION;
		launch_status = SparkGlm5NextModuleReduceAttentionOut(chain,chain->slot->attention_out_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_GATHER_INDEX:
		if ( SparkGlm5NextLaunchCudaLayerAttentionSelect(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		if ( SparkGlm5NextLayerKvShardActive(&chain->wave,chain->next_layer) != 0u )
		{
			chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_SHARD_QUERY;
			launch_status = SparkGlm5NextModuleKvShardExchange(chain,0u,1u);
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm5NextTpChainFail(chain,launch_status);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION;
		launch_status = SparkGlm5NextModuleReduceAttentionOut(chain,chain->slot->attention_out_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION:
		if ( SparkGlm5NextLaunchCudaLayerAttentionPost(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_MLP;
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_MLP:
		if ( state->lazy_pack != 0 && (chain->wave.first_layer_index + chain->next_layer) >= SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER )
		{
			{
				const uint32_t *cover_saved = chain->wave.expert_cover;
				void *miss_saved = chain->wave.expert_miss;
				chain->wave.expert_cover = 0;
				chain->wave.expert_miss = 0;
				launch_status = SparkGlm5NextLaunchCudaLayerMlpRoute(&chain->wave,chain->next_layer) != 0 ?
				    SPARK_STATUS_INTERNAL_ERROR : SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkGlm5NextLazyWork,chain);
				chain->wave.expert_cover = cover_saved;
				chain->wave.expert_miss = miss_saved;
			}
			if ( launch_status != SPARK_STATUS_OK )
				SparkGlm5NextTpChainFail(chain,launch_status);
			return;
		}
		if ( SparkGlm5NextLaunchCudaLayerMlp(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		SparkGlm5NextTpChainReduceMlp(chain);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_MLP:
		if ( SparkGlm5NextLaunchCudaLayerMlpPost(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		if ( SparkGlm5NextTapCapture(chain,chain->next_layer) != 0u )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		SparkGlm5NextT1Streams(chain,chain->wave.first_layer_index + chain->next_layer);
		SparkGlm5NextT1Route(chain,chain->wave.first_layer_index + chain->next_layer);
		chain->next_layer++;
		if ( chain->next_layer < chain->wave.layer_count )
		{
			chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION;
			SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		}
		else
		{
			chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_HEAD;
			SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		}
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_HEAD:
		if ( SparkGlm5NextLaunchCudaWaveHead(&chain->wave) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_HEAD;
		launch_status = SparkGlm5NextModuleReduceHeadMax(chain);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_HEAD:
		error = SparkGlm5NextLaunchHeadMaxlocUnpack((cudaStream_t)chain->slot->stream,chain->slot->head_maxloc_u64,chain->slot->output_token,chain->wave_rows);
		if ( error == cudaSuccess && state->owns_final_head != 0u )
			error = cudaMemcpyAsync(chain->slot->host_output_token_ids + chain->first_row,chain->slot->output_token,(uint64_t)chain->wave_rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,(cudaStream_t)chain->slot->stream);
		if ( error == cudaSuccess && SparkGlm5NextTapFlush(chain) != 0u )
			error = cudaErrorLaunchFailure;
		launch_status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"tp_head_unpack");
		if ( (state->hbound_probes & (1u << 31u)) == 0u &&
		     SparkGlm5NextBoundedStreamSync(chain->state,chain->slot->stream,UINT64_C(35000000000)) == 0 )
		{
			state->hbound_probes |= 1u << 31u;
			fprintf(stderr,"HEADFIN v=%u\n",
			    chain->slot->host_output_token_ids[chain->first_row]);
		}
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkGlm5NextTpChainFail(chain,launch_status);
			return;
		}
		SparkGlm5NextT1Head(chain);
#ifdef SPARK_SCORE_DUMP
		launch_status = SparkGlm5NextScoreWave(chain);
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkGlm5NextTpChainFail(chain,launch_status);
			return;
		}
#endif
		if ( chain->spec_verify != 0u )
		{
			error = cudaLaunchHostFunc((cudaStream_t)chain->slot->stream,SparkGlm5NextMtpResolveHost,chain);
			if ( error != cudaSuccess )
			{
				SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
				return;
			}
			return;
		}
		if ( state->mtp_enabled != 0u && state->owns_final_head != 0u )
		{
			launch_status = SparkGlm5NextMtpStashHidden(state,chain);
			if ( launch_status != SPARK_STATUS_OK )
			{
				SparkGlm5NextTpChainFail(chain,launch_status);
				return;
			}
		}
		SparkGlm5NextNoteWarm(state);
		SparkGlm5NextFinishChain(chain);
		return;
	default:
		SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
		return;
	}
}

static void SparkGlm5NextPrepareAsyncCompletion(
	SparkGlm5NextModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkGlm5NextResidentDecodeStageBatchView *batch,
	const uint8_t *lane_bound,
	const uint64_t *lane_sequence_ids,
	const uint64_t *lane_next_positions,
	uint32_t slot_index)
{
	SparkGlm5NextAsyncCompletion *async;
	uint32_t lane;
	async = &state->completions[slot_index];
	memset(async,0,sizeof(*async));
	async->state = state;
	async->completion_function = frame->completion_function;
	async->completion_context = frame->completion_context;
	async->slot_index = slot_index;
	async->lane_count = batch->active_sequence_count;
	async->row_count = batch->row_count;
	async->prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	async->steps = SparkGlm5NextFrameSteps(frame);
	async->state_capture = ((const SparkGlm5NextResidentDecodeStageFrameContext *)frame->user_context)->state_capture;
	async->output_token_destination = state->owns_final_head != 0u ? (uint32_t *)frame->buffers[0].address : 0;
	for (lane=0u; lane<batch->active_sequence_count && lane<SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
	{
		async->lane_indices[lane] = batch->row_resident_slots[lane];
		async->lane_bound[lane] = lane_bound[lane];
		async->lane_sequence_ids[lane] = lane_sequence_ids[lane];
		async->lane_next_positions[lane] = lane_next_positions[lane];
	}
	async->completion.request_id = frame->request_id;
	async->completion.sequence_id = frame->sequence_id;
	async->completion.sequence_position = frame->sequence_position;
	async->completion.program_id = frame->program_id;
	async->completion.driver_dispatch_slot = frame->driver_dispatch_slot;
	async->completion.accepted_token_count = frame->new_token_count;
	async->completion.tokens_per_sequence = frame->tokens_per_sequence;
	async->completion.status = SPARK_STATUS_OK;
	async->completion.residency = frame->residency;
	async->completion.host_staging_bytes = (uint64_t)batch->row_count * (sizeof(uint32_t) * (3u + state->owns_final_head) + sizeof(SparkRowSampling) * state->owns_final_head);
	async->completion.device_memcpy_bytes = async->completion.host_staging_bytes;
}

static SparkStatus SparkGlm5NextCaptureRecurrent(SparkGlm5NextModuleState *state,uint32_t resident)
{
	SparkKvLaneTransaction *owner;
	SparkKvPageCacheSequence *sequence;
	uint32_t page;
	uint64_t generation,start_ns;
	SparkStatus status;
	if ( state->kda_layer_count == 0u )
		return(SPARK_STATUS_OK);
	if ( resident >= state->resident_sequence_capacity || state->kv_lane_transactions == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	owner = &state->kv_lane_transactions[resident];
	if ( (owner->lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH) == 0u )
		return(SPARK_STATUS_OK);
	if ( owner->phase != SPARK_KV_LANE_TRANSACTION_EXECUTING )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	sequence = &state->kv_transactions.cache->sequences[resident];
	page = sequence->mutable_logical_page_index;
	if ( page >= state->page_count || state->kv_blocks[page].residency_reference_count == 0u )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	generation = state->kv_blocks[page].generation;
	start_ns = SparkGlm5NextNowNs();
	status = SparkGlm5NextRecurrentCopy(state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,resident,state->recurrent_staging,state->recurrent_page_bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkKvPageStoreWriteback(&state->recurrent_store,page,resident,generation,(uintptr_t)state->recurrent_staging,state->recurrent_page_bytes,0u,0u);
	while ( status == SPARK_STATUS_BUSY )
	{
		SparkStatus wait = SparkKvPageStoreWaitForTransfers(&state->recurrent_store);
		if ( wait != SPARK_STATUS_OK )
			SPARK_RETURN(wait);
		status = SparkKvPageStoreWriteback(&state->recurrent_store,page,resident,generation,(uintptr_t)state->recurrent_staging,state->recurrent_page_bytes,0u,0u);
	}
	if ( status == SPARK_STATUS_OK )
		SparkGlm5NextKdaCount(state->kda_capture,state->recurrent_page_bytes,start_ns);
	SPARK_RETURN(status);
}

static void SparkGlm5NextCaptureClearUnused(uint8_t *payload,uint64_t bytes,uint32_t layers,uint32_t valid_tokens)
{
	uint64_t layer_bytes = bytes / layers,valid_bytes = layer_bytes / 64u * valid_tokens;
	uint32_t layer;
	for (layer=0u; layer<layers; layer++)
		memset(payload + layer * layer_bytes + valid_bytes,0,(size_t)(layer_bytes - valid_bytes));
}

static SparkStatus SparkGlm5NextCaptureState(SparkGlm5NextAsyncCompletion *async)
{
	SparkGlm5NextModuleState *state = async->state;
	SparkGlm5NextStateCapture *capture = async->state_capture;
	SparkGlm5NextExecutionSlot *slot = &state->slots[async->slot_index];
	uint32_t lane,page,resident,logical,physical,valid_tokens,last_row;
	uint64_t offset = 0u,key_bytes,index_bytes,hidden_bytes,map_offset;
	SparkKvLaneTransaction *owner;
	SparkGlm5NextStateCaptureLane *out;
	SparkStatus status;
	if ( capture == 0 )
		return(SPARK_STATUS_OK);
	capture->payload_bytes = 0u;
	key_bytes = state->kv_arena.key_block_stride_bytes;
	index_bytes = state->kv_arena.value_block_stride_bytes;
	hidden_bytes = (uint64_t)SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t);
	for (lane=0u; lane<async->lane_count; lane++)
	{
		resident = async->lane_indices[lane];
		owner = &state->kv_lane_transactions[resident];
		out = &capture->lanes[lane];
		if ( owner->phase != SPARK_KV_LANE_TRANSACTION_EXECUTING || owner->page_count != SparkCeilDivU32((uint32_t)async->lane_next_positions[lane],64u) || owner->page_count > capture->pages_per_lane_capacity )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		*out = (SparkGlm5NextStateCaptureLane){.sequence_id=async->lane_sequence_ids[lane],.next_position=async->lane_next_positions[lane],.payload_offset=offset,.resident_slot=resident,.page_count=owner->page_count};
		for (page=0u; page<owner->page_count; page++)
		{
			map_offset = (uint64_t)resident * state->pages_per_sequence + page;
			logical = state->kv_lane_logical_pages[map_offset];
			physical = state->kv_lane_physical_pages[map_offset];
			if ( logical >= state->page_count || physical >= state->physical_page_count || state->kv_blocks[logical].resident_slot_index != physical || state->kv_blocks[logical].residency_reference_count == 0u )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			capture->logical_pages[(uint64_t)lane * capture->pages_per_lane_capacity + page] = logical;
			capture->physical_pages[(uint64_t)lane * capture->pages_per_lane_capacity + page] = physical;
			status = SparkGlm5NextPageCopy(state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,state->kv_blocks[logical].key_device_address,capture->payload + offset,key_bytes);
			if ( status != SPARK_STATUS_OK )
				return(status);
			valid_tokens = (uint32_t)(async->lane_next_positions[lane] - (uint64_t)page * 64u);
			if ( valid_tokens > 64u )
				valid_tokens = 64u;
			SparkGlm5NextCaptureClearUnused(capture->payload + offset,key_bytes,state->kv_layer_count,valid_tokens);
			offset += key_bytes;
			status = SparkGlm5NextPageCopy(state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,state->kv_blocks[logical].value_device_address,capture->payload + offset,index_bytes);
			if ( status != SPARK_STATUS_OK )
				return(status);
			SparkGlm5NextCaptureClearUnused(capture->payload + offset,index_bytes,state->index_layer_count,valid_tokens);
			offset += index_bytes;
		}
		status = state->recurrent_page_bytes != 0u ? SparkGlm5NextRecurrentCopy(state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,resident,capture->payload + offset,state->recurrent_page_bytes) : SPARK_STATUS_OK;
		if ( status != SPARK_STATUS_OK )
			return(status);
		offset += state->recurrent_page_bytes;
		last_row = async->row_count;
		while ( last_row != 0u && slot->host_resident_slots[last_row - 1u] != resident )
			last_row--;
		if ( last_row == 0u )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		last_row--;
		if ( cudaMemcpy(capture->payload + offset,slot->hidden_bf16 + last_row * hidden_bytes / sizeof(uint16_t),hidden_bytes,cudaMemcpyDeviceToHost) != cudaSuccess || cudaMemcpy(&out->output_score,slot->output_score + last_row,sizeof(float),cudaMemcpyDeviceToHost) != cudaSuccess )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		offset += hidden_bytes;
		out->payload_bytes = offset - out->payload_offset;
	}
	capture->key_page_bytes = key_bytes;
	capture->index_page_bytes = index_bytes;
	capture->key_layer_count = state->kv_layer_count;
	capture->index_layer_count = state->index_layer_count;
	capture->recurrent_bytes = state->recurrent_page_bytes;
	capture->hidden_bytes = hidden_bytes;
	capture->backing_write_count = state->kv_page_store.write_count;
	capture->backing_read_count = state->kv_page_store.read_count;
	capture->prefix_hit_count = state->kv_page_cache.prefix_hit_count;
	capture->payload_bytes = offset;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextFinishCacheLanes(SparkGlm5NextAsyncCompletion *async)
{
	SparkGlm5NextModuleState *state = async->state;
	SparkStatus result;
	uint32_t lane,resident;
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	result = async->completion.status;
	if ( result == SPARK_STATUS_OK )
		result = SparkGlm5NextCaptureState(async);
	for (lane=0u; lane<async->lane_count && result==SPARK_STATUS_OK; lane++)
		result = SparkGlm5NextCaptureRecurrent(state,async->lane_indices[lane]);
	result = SparkKvLaneTransactionsFinish(&state->kv_transactions,async->lane_indices,async->lane_count,result,(uint32_t)async->cache_extra_tokens);
	for (lane=0u; lane<async->lane_count; lane++)
	{
		resident = async->lane_indices[lane];
		atomic_store_explicit(&state->lane_bound[resident],result == SPARK_STATUS_OK ? async->lane_bound[lane] : 0u,memory_order_release);
		if ( result == SPARK_STATUS_OK )
		{
			atomic_store_explicit(&state->lane_sequence_ids[resident],async->lane_sequence_ids[lane],memory_order_release);
			atomic_store_explicit(&state->lane_next_positions[resident],async->lane_next_positions[lane],memory_order_release);
		}
		else
		{
			memset(state->page_table_shadow + (uint64_t)resident * state->pages_per_sequence,0xff,(uint64_t)state->pages_per_sequence * sizeof(uint32_t));
			if ( state->mtp_lane_armed != 0 )
				state->mtp_lane_armed[resident] = 0u;
		}
	}
	(void)pthread_mutex_unlock(&state->kv_mutex);
	return(result);
}

static SparkStatus SparkGlm5NextCompletionStatus(
	SparkGlm5NextAsyncCompletion *async,SparkGlm5NextExecutionSlot *slot)
{
	SparkStatus status = async->completion.status;
	uint32_t output_count,output;
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkGlm5NextWeightdHealth(async->state);
	if ( status != SPARK_STATUS_OK || async->output_token_destination == 0 )
		return(status);
	output_count = async->burst_token_count != 0u ?
		async->burst_token_count : async->row_count;
	for ( output = 0u; output < output_count; output++ )
		if ( slot->host_output_token_ids[output] >=
		        SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT )
			return(SparkGlm5NextTerminalFailure(async->state,
			    SPARK_STATUS_VALIDATION_FAILED,"output-token-range"));
	return(SPARK_STATUS_OK);
}

static uint64_t SparkGlm5NextSpanNs(uint64_t start_ns,uint64_t end_ns)
{
	return(start_ns != 0u && end_ns >= start_ns ? end_ns - start_ns : 0u);
}

static const char *const SparkGlm5NextWaveIntervalNames[SPARK_GLM5_NEXT_WAVE_INTERVALS] = {"idle","wait","key","setup","run","post"};

static void SparkGlm5NextLineAppend(char *line,size_t capacity,size_t *used,const char *format,...)
{
	va_list arguments;
	int written;
	if ( *used >= capacity )
		return;
	va_start(arguments,format);
	written = vsnprintf(line + *used,capacity - *used,format,arguments);
	va_end(arguments);
	if ( written > 0 )
		*used += (size_t)written < capacity - *used ? (size_t)written : capacity - *used;
}

static void SparkGlm5NextWaveTimingReport(SparkGlm5NextWaveTiming *timing,uint32_t tp_rank,uint64_t now_ns)
{
	char line[1536];
	size_t used = 0u;
	uint64_t delivered_ns = timing->delivered_ns;
	uint32_t index;
	SparkGlm5NextLineAppend(line,sizeof(line),&used,"G5N-WAVE-TIMING rank=%u waves=%llu rows=%llu steps=%llu prefill=%llu graph=%llu eager=%llu linear=%llu graph_path=%u retries=%llu busy=%llu/%llu/%llu/%llu/%llu captures=%llu capture_ms=%llu",tp_rank,(unsigned long long)timing->waves,(unsigned long long)timing->rows,(unsigned long long)timing->steps,(unsigned long long)timing->prefill,(unsigned long long)timing->graph,(unsigned long long)(timing->waves - timing->graph),(unsigned long long)timing->linear,timing->graph_path,(unsigned long long)timing->retries,(unsigned long long)timing->busy[SPARK_GLM5_NEXT_BUSY_CHAIN],(unsigned long long)timing->busy[SPARK_GLM5_NEXT_BUSY_STREAM],(unsigned long long)timing->busy[SPARK_GLM5_NEXT_BUSY_SLOT],(unsigned long long)timing->busy[SPARK_GLM5_NEXT_BUSY_LANES],(unsigned long long)timing->busy[SPARK_GLM5_NEXT_BUSY_OTHER],(unsigned long long)timing->captures,(unsigned long long)(timing->capture_ns / 1000000u));
	for (index=0u; index<SPARK_GLM5_NEXT_WAVE_INTERVALS; index++)
		SparkGlm5NextLineAppend(line,sizeof(line),&used," %s_us=%llu/%llu",SparkGlm5NextWaveIntervalNames[index],(unsigned long long)SparkLatencyPercentileUs(&timing->interval[index],50u),(unsigned long long)SparkLatencyPercentileUs(&timing->interval[index],99u));
	for (index=0u; index<SPARK_GLM5_NEXT_WAVE_INTERVALS; index++)
		SparkGlm5NextLineAppend(line,sizeof(line),&used," %s_ms=%llu",SparkGlm5NextWaveIntervalNames[index],(unsigned long long)(timing->interval[index].total_ns / 1000000u));
	SparkGlm5NextLineAppend(line,sizeof(line),&used," graph_run_ms=%llu eager_run_ms=%llu linear_run_ms=%llu linear_walk_ms=%llu decode_wait_ms=%llu source_wait_ms=%llu peer_wait_ms=%llu copy_ms=%llu combine_ms=%llu worst_ms=%llu worst_request=%llu worst_epochs=%llu/%llu worst_us=",(unsigned long long)(timing->graph_run_ns / 1000000u),(unsigned long long)(timing->eager_run_ns / 1000000u),(unsigned long long)(timing->linear_run_ns / 1000000u),(unsigned long long)(timing->linear_walk_ns / 1000000u),(unsigned long long)(timing->decode_wait_ns / 1000000u),(unsigned long long)(timing->source_wait_ns / 1000000u),(unsigned long long)(timing->peer_wait_ns / 1000000u),(unsigned long long)(timing->copy_ns / 1000000u),(unsigned long long)(timing->combine_ns / 1000000u),(unsigned long long)(timing->worst_ns / 1000000u),(unsigned long long)timing->worst_request,(unsigned long long)timing->worst_epoch[0],(unsigned long long)timing->worst_epoch[1]);
	for (index=0u; index<SPARK_GLM5_NEXT_WAVE_INTERVALS; index++)
		SparkGlm5NextLineAppend(line,sizeof(line),&used,"%s%llu",index == 0u ? "" : "/",(unsigned long long)(timing->worst_part_ns[index] / 1000u));
	fprintf(stderr,"%s\n",line);
	memset(timing,0,sizeof(*timing));
	timing->delivered_ns = delivered_ns;
	timing->window_ns = now_ns;
}

static void SparkGlm5NextKdaTimingReport(SparkGlm5NextModuleState *state,uint64_t now_ns)
{
	uint64_t window_ns = atomic_load_explicit(&state->kda_window_ns,memory_order_relaxed),restore[SPARK_GLM5_NEXT_KDA_FIELDS],capture[SPARK_GLM5_NEXT_KDA_FIELDS];
	uint32_t field;
	if ( window_ns == 0u || now_ns < window_ns || now_ns - window_ns < SPARK_GLM5_NEXT_WAVE_TIMING_WINDOW_NS )
	{
		if ( window_ns == 0u )
			atomic_store_explicit(&state->kda_window_ns,now_ns,memory_order_relaxed);
		return;
	}
	if ( !atomic_compare_exchange_strong_explicit(&state->kda_window_ns,&window_ns,now_ns,memory_order_relaxed,memory_order_relaxed) )
		return;
	for (field=0u; field<SPARK_GLM5_NEXT_KDA_FIELDS; field++)
	{
		restore[field] = atomic_exchange_explicit(&state->kda_restore[field],0u,memory_order_relaxed);
		capture[field] = atomic_exchange_explicit(&state->kda_capture[field],0u,memory_order_relaxed);
	}
	if ( restore[SPARK_GLM5_NEXT_KDA_COUNT] != 0u || capture[SPARK_GLM5_NEXT_KDA_COUNT] != 0u )
		fprintf(stderr,"G5N-KDA-TIMING rank=%u restores=%llu restore_bytes=%llu restore_us=%llu captures=%llu capture_bytes=%llu capture_us=%llu\n",state->tp_rank,(unsigned long long)restore[SPARK_GLM5_NEXT_KDA_COUNT],(unsigned long long)restore[SPARK_GLM5_NEXT_KDA_BYTES],(unsigned long long)(restore[SPARK_GLM5_NEXT_KDA_NS] / 1000u),(unsigned long long)capture[SPARK_GLM5_NEXT_KDA_COUNT],(unsigned long long)capture[SPARK_GLM5_NEXT_KDA_BYTES],(unsigned long long)(capture[SPARK_GLM5_NEXT_KDA_NS] / 1000u));
}

static void SparkGlm5NextWaveTimingWorst(SparkGlm5NextWaveTiming *timing,const SparkGlm5NextAsyncCompletion *async,const uint64_t *marks,uint64_t total_ns)
{
	uint32_t index;
	timing->worst_ns = total_ns;
	timing->worst_request = async->completion.request_id;
	timing->worst_epoch[0] = async->epoch[0];
	timing->worst_epoch[1] = async->epoch[1];
	for (index=0u; index<SPARK_GLM5_NEXT_WAVE_INTERVALS; index++)
		timing->worst_part_ns[index] = SparkGlm5NextSpanNs(marks[index],marks[index + 1u]);
}

static void SparkGlm5NextWaveTimingCount(SparkGlm5NextWaveTiming *timing,const SparkGlm5NextAsyncCompletion *async,const SparkTpDeviceCollectiveHardwareTiming *collective,uint64_t run_ns)
{
	uint32_t reason;
	timing->waves++;
	timing->rows += async->row_count;
	timing->steps += async->steps;
	timing->prefill += async->prefill;
	timing->graph += async->graph;
	timing->linear += async->linear;
	timing->retries += async->retries;
	timing->captures += async->captures;
	timing->capture_ns += async->capture_ns;
	timing->graph_path = async->graph_path;
	timing->graph_run_ns += async->graph != 0u ? run_ns : 0u;
	timing->eager_run_ns += async->graph != 0u ? 0u : run_ns;
	timing->linear_run_ns += async->linear != 0u ? run_ns : 0u;
	timing->linear_walk_ns += async->walk_ns;
	timing->decode_wait_ns += async->prefill == 0u ? SparkGlm5NextSpanNs(async->attempt_ns,async->chain_start_ns) : 0u;
	timing->source_wait_ns += collective->source_wait_ns;
	timing->peer_wait_ns += collective->peer_wait_ns;
	timing->copy_ns += collective->copy_ns;
	timing->combine_ns += collective->combine_ns;
	for (reason=0u; reason<SPARK_GLM5_NEXT_BUSY_REASONS; reason++)
		timing->busy[reason] += async->busy[reason];
}

static void SparkGlm5NextWaveTimingRecord(SparkGlm5NextWaveTiming *timing,const SparkGlm5NextAsyncCompletion *async,const SparkTpDeviceCollectiveHardwareTiming *collective,uint32_t tp_rank,uint64_t delivered_ns)
{
	uint64_t marks[SPARK_GLM5_NEXT_WAVE_INTERVALS + 1u] = {timing->delivered_ns,async->attempt_ns,async->chain_start_ns,async->keyed_ns,async->launch_ns,async->finish_ns,delivered_ns};
	uint64_t total_ns = SparkGlm5NextSpanNs(async->attempt_ns,delivered_ns);
	uint32_t index;
	if ( timing->window_ns == 0u )
		timing->window_ns = delivered_ns;
	for (index=0u; index<SPARK_GLM5_NEXT_WAVE_INTERVALS; index++)
		SparkLatencyAdd(&timing->interval[index],marks[index],marks[index + 1u]);
	SparkGlm5NextWaveTimingCount(timing,async,collective,SparkGlm5NextSpanNs(async->launch_ns,async->finish_ns));
	if ( total_ns > timing->worst_ns )
		SparkGlm5NextWaveTimingWorst(timing,async,marks,total_ns);
	timing->delivered_ns = delivered_ns;
	if ( delivered_ns - timing->window_ns >= SPARK_GLM5_NEXT_WAVE_TIMING_WINDOW_NS )
		SparkGlm5NextWaveTimingReport(timing,tp_rank,delivered_ns);
}

static const char *SparkGlm5NextChainPath(const SparkGlm5NextAsyncCompletion *async)
{
	return(async->graph != 0u ? "graph" : async->linear != 0u ? "linear" : "eager");
}

static void SparkGlm5NextVerifyDeferred(SparkGlm5NextModuleState *state,SparkGlm5NextAsyncCompletion *async)
{
	SparkStatus main_status = state->tp_device_collective_initialized != 0u ? SparkTpDeviceCollectiveVerifyDeferred(&state->tp_device_collective,state->execution_stream) : SPARK_STATUS_OK,hc_status = state->tp_device_collective_hc_initialized != 0u ? SparkTpDeviceCollectiveVerifyDeferred(&state->tp_device_collective_hc,state->execution_stream) : SPARK_STATUS_OK;
	if ( async->linear != 0u )
		async->finish_ns = SparkGlm5NextNowNs();
	if ( (main_status == SPARK_STATUS_OK && hc_status == SPARK_STATUS_OK) || async->completion.status != SPARK_STATUS_OK )
		return;
	if ( state->tp_device_collective_initialized != 0u )
		SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective);
	if ( state->tp_device_collective_hc_initialized != 0u )
		SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective_hc);
	async->completion.status = main_status != SPARK_STATUS_OK ? main_status : hc_status;
}

static void SparkGlm5NextGatherSteps(SparkGlm5NextAsyncCompletion *async,SparkGlm5NextExecutionSlot *slot)
{
	uint32_t steps = async->steps,rows = async->row_count,row,lane;
	if ( steps < 2u || async->completion.status != SPARK_STATUS_OK )
		return;
	for (row=0u; row<rows; row++)
		slot->host_chain_token_ids[row * steps + steps - 1u] = slot->host_output_token_ids[row];
	memcpy(slot->host_output_token_ids,slot->host_chain_token_ids,(uint64_t)rows * steps * sizeof(uint32_t));
	async->burst_token_count = rows * steps;
	async->cache_extra_tokens = steps - 1u;
	for (lane=0u; lane<async->lane_count; lane++)
		async->lane_next_positions[lane] += steps - 1u;
}

static void SparkGlm5NextCompleteOnWorker(void *context)
{
	SparkGlm5NextAsyncCompletion *async = (SparkGlm5NextAsyncCompletion *)context;
	SparkGlm5NextModuleState *state = async != 0 ? async->state : 0;
	SparkGlm5NextExecutionSlot *slot;
	SparkModelDriverCompletion completion;
	SparkModelDriverCompletionFunction complete;
	SparkTpDeviceCollectiveHardwareTiming collective = {0};
	void *complete_context;
	if ( state == 0 )
		return;
	pthread_mutex_lock(&state->completion_queue_lock);
	SparkGlm5NextDrainParkedCompletions(state);
	pthread_mutex_unlock(&state->completion_queue_lock);
	if ( async->slot_index >= state->pipeline_slot_count )
		return;
	slot = &state->slots[async->slot_index];
	if ( async->completion.status != SPARK_STATUS_OK )
	{
		if ( state->tp_device_collective_initialized != 0u )
			SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective);
		if ( state->tp_device_collective_hc_initialized != 0u )
			SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective_hc);
	}
	if ( SparkGlm5NextBoundedStreamSync(state,state->execution_stream,UINT64_C(35000000000)) != 0 )
	{
		(void)SparkGlm5NextTerminalFailure(state,SPARK_STATUS_IO_ERROR,"completion-drain");
		fprintf(stderr,"GLM completion drain failed; retaining slot %u and chain ownership\n",async->slot_index);
		return;
	}
	SparkGlm5NextVerifyDeferred(state,async);
	SparkGlm5NextGatherSteps(async,slot);
	SparkGlm5NextVerifyObserveOutputs(state,async,slot);
	if ( slot->host_kv_access_error[0] != 0u )
	{
		fprintf(stderr,"GLM cache access failed: code %u row %u slot %u\n",slot->host_kv_access_error[0],slot->host_kv_access_error[2],async->slot_index);
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
	}
	async->completion.status = SparkGlm5NextCompletionStatus(async,slot);
	async->completion.status = SparkGlm5NextFinishCacheLanes(async);
	{
		SparkStatus end_status = SPARK_STATUS_OK;
		if ( state->tp_device_collective_initialized != 0u )
			end_status = SparkTpDeviceCollectiveEndChain(&state->tp_device_collective,state->execution_stream);
		if ( end_status == SPARK_STATUS_OK && state->tp_device_collective_hc_initialized != 0u )
			end_status = SparkTpDeviceCollectiveEndChain(&state->tp_device_collective_hc,state->execution_stream);
		if ( end_status != SPARK_STATUS_OK )
		{
			(void)SparkGlm5NextTerminalFailure(state,end_status,"collective-end");
			fprintf(stderr,"GLM chain end failed: slot %u status %d; retaining ownership\n",async->slot_index,(int)end_status);
			return;
		}
	}

	{
		uint64_t round_count = 0u,round_ns = 0u,chain_ns = SparkGlm5NextNowNs();
		SparkTpDeviceCollectiveRoundStats(&state->tp_device_collective,&round_count,&round_ns,1u);
		fprintf(stderr,"CHAIN-TIME slot=%u path=%s steps=%u rows=%u epoch=%llu status=%d total_ms=%.2f walk_ms=%.2f collective_host_submit_ms=%.2f collective_host_submissions=%llu stage_ms=%.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1f\n",
			(unsigned)async->slot_index,SparkGlm5NextChainPath(async),async->steps,async->row_count,(unsigned long long)async->epoch[0],(int)async->completion.status,
			async->chain_start_ns != 0u ? (double)(chain_ns - async->chain_start_ns) / 1000000.0 : 0.0,(double)async->walk_ns / 1000000.0,
			(double)round_ns / 1000000.0,(unsigned long long)round_count,
			(double)state->chain_stage_ns[0] / 1000000.0,
			(double)state->chain_stage_ns[1] / 1000000.0,
			(double)state->chain_stage_ns[2] / 1000000.0,
			(double)state->chain_stage_ns[3] / 1000000.0,
			(double)state->chain_stage_ns[4] / 1000000.0,
			(double)state->chain_stage_ns[5] / 1000000.0,
			(double)state->chain_stage_ns[6] / 1000000.0,
			(double)state->chain_stage_ns[7] / 1000000.0);
		{
			SparkTpDeviceCollectiveHardwareTiming main_timing = {0},hc_timing = {0};
			SparkStatus main_status = SparkTpDeviceCollectiveHardwareStats(
				&state->tp_device_collective,&main_timing);
			SparkStatus hc_status = state->tp_device_collective_hc_initialized != 0u ?
				SparkTpDeviceCollectiveHardwareStats(&state->tp_device_collective_hc,&hc_timing) : SPARK_STATUS_OK;
			collective.source_wait_ns = main_timing.source_wait_ns + hc_timing.source_wait_ns;
			collective.peer_wait_ns = main_timing.peer_wait_ns + hc_timing.peer_wait_ns;
			collective.copy_ns = main_timing.copy_ns + hc_timing.copy_ns;
			collective.combine_ns = main_timing.combine_ns + hc_timing.combine_ns;
			if ( main_status == SPARK_STATUS_OK && hc_status == SPARK_STATUS_OK )
				fprintf(stderr,"COLLECTIVE-GPU-TIME slot=%u source_wait_ms=%.3f peer_wait_ms=%.3f copy_ms=%.3f combine_ms=%.3f\n",
					async->slot_index,(double)(main_timing.source_wait_ns + hc_timing.source_wait_ns) / 1000000.0,
					(double)(main_timing.peer_wait_ns + hc_timing.peer_wait_ns) / 1000000.0,
					(double)(main_timing.copy_ns + hc_timing.copy_ns) / 1000000.0,
					(double)(main_timing.combine_ns + hc_timing.combine_ns) / 1000000.0);
			else if ( main_status != SPARK_STATUS_UNSUPPORTED )
				fprintf(stderr,"COLLECTIVE-GPU-TIME-UNAVAILABLE slot=%u main=%d hc=%d\n",
					async->slot_index,(int)main_status,(int)hc_status);
		}
		memset(state->chain_stage_ns,0,sizeof(state->chain_stage_ns));
		state->chain_profile_last_ns = 0ull;
	}
	if ( async->completion.status == SPARK_STATUS_BUSY &&
	     ++async->finish_retries >= 2u )
	{
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
		fprintf(stderr,
		    "cache finish forced after retries slot=%u\n",
		    async->slot_index);
	}
	if ( async->completion.status == SPARK_STATUS_OK )
	{
		if ( async->completion.status == SPARK_STATUS_OK &&
		     async->output_token_destination != 0 )
		{
			memcpy(async->output_token_destination,slot->host_output_token_ids,(uint64_t)(async->burst_token_count != 0u ? async->burst_token_count : async->row_count) * sizeof(uint32_t));
		}
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
	}
	else
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	atomic_fetch_add_explicit(&state->host_callback_completion_count,1u,memory_order_relaxed);
	// Snapshot before releasing the slot: callback-driven reuse can overwrite async.
	completion = async->completion;
	complete = async->completion_function;
	complete_context = async->completion_context;
	SparkGlm5NextWaveTimingRecord(&state->wave_timing,async,&collective,state->tp_rank,SparkGlm5NextNowNs());
	SparkGlm5NextKdaTimingReport(state,SparkGlm5NextNowNs());
	SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,async->lane_indices,async->lane_count);
	atomic_store_explicit(&state->tp_chain_active,0u,memory_order_release);
	SparkStageModuleSlotRelease(state->slot_states,async->slot_index);
	complete(complete_context,&completion);
}

static void SparkGlm5NextParkCompletionLocked(
    SparkGlm5NextModuleState *state,SparkWeightdWorkFunction function,void *context)
{
	uint32_t i;
	for (i=0u; i<SPARK_GLM5_NEXT_OVERFLOW_POOL; i++)
		if ( state->overflow_pool[i].function == 0 )
		{
			state->overflow_pool[i].function = function;
			state->overflow_pool[i].context = context;
			state->overflow_parked_count++;
			return;
		}
	(void)SparkGlm5NextTerminalFailure(state,SPARK_STATUS_CAPACITY_EXCEEDED,"completion-work-capacity");
	fprintf(stderr,"GLM callback work pool exhausted; retaining occupied slots\n");
}

static void SparkGlm5NextDrainParkedCompletions(SparkGlm5NextModuleState *state)
{
	uint32_t i;
	if ( state == 0 || state->completion_worker == 0 )
		return;
	for (i=0u; i<SPARK_GLM5_NEXT_OVERFLOW_POOL; i++)
		if ( state->overflow_pool[i].function != 0 )
		{
			if ( SparkWeightdWorkerSubmit(state->completion_worker,state->overflow_pool[i].function,state->overflow_pool[i].context) != SPARK_STATUS_OK )
				return;
			state->overflow_pool[i].function = 0;
			state->overflow_pool[i].context = 0;
			state->overflow_parked_count--;
		}
}

static void SparkGlm5NextScheduleCompletionWork(SparkGlm5NextModuleState *state,SparkWeightdWorkFunction function,void *context)
{
	pthread_mutex_lock(&state->completion_queue_lock);
	if ( SparkWeightdWorkerSubmit(state->completion_worker,function,context) != SPARK_STATUS_OK )
		SparkGlm5NextParkCompletionLocked(state,function,context);
	pthread_mutex_unlock(&state->completion_queue_lock);
}

static void CUDART_CB SparkGlm5NextCompleteAsync(void *context)
{
	SparkGlm5NextAsyncCompletion *async = context;
	if ( async != 0 && async->state != 0 )
		SparkGlm5NextScheduleCompletionWork(async->state,SparkGlm5NextCompleteOnWorker,async);
}

static int SparkGlm5NextBoundedStreamSync(SparkGlm5NextModuleState *state,void *stream,uint64_t timeout_ns)
{
	SparkStatus status;
	if ( stream != state->execution_stream ) return(-1);
	status = SparkStageModuleCudaWaitFor(&state->stream_wait,timeout_ns);
	if ( status == SPARK_STATUS_BUSY )
		fprintf(stderr,"GLM stream receipt timeout after %llu ms; retaining chain ownership\n",(unsigned long long)(timeout_ns / UINT64_C(1000000)));
	return(status == SPARK_STATUS_OK ? 0 : status == SPARK_STATUS_BUSY ? 1 : -1);
}

#include "sparkpipe/family/module/spark_module_claim_cache_frame.h"

static SparkStatus SparkGlm5NextRestoreRecurrent(SparkGlm5NextModuleState *state,uint32_t resident)
{
	const SparkKvLaneTransaction *owner = &state->kv_lane_transactions[resident];
	SparkKvPageCache *cache = state->kv_transactions.cache;
	uint32_t entry,page;
	uint64_t generation,start_ns;
	SparkStatus status;
	if ( SparkGlm5NextPrefixRestorePending(owner) == 0u || state->kda_layer_count == 0u )
		return(SPARK_STATUS_OK);
	if ( owner->phase != SPARK_KV_LANE_TRANSACTION_EXECUTING )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	entry = cache->sequences[resident].terminal_entry_index;
	if ( entry >= cache->entry_capacity || cache->entries[entry].token_count != owner->lane.sequence_position || cache->entries[entry].reference_count == 0u || cache->sequences[resident].sequence_id != owner->lane.sequence_id )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	page = cache->entries[entry].logical_page_index;
	if ( page >= state->page_count || state->kv_blocks[page].reference_count == 0u )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( state->kv_blocks[page].residency_reference_count == 0u )
	{
		uint32_t mutable = cache->sequences[resident].mutable_logical_page_index;
		if ( owner->lane.sequence_position % cache->kv_cache_arena->block_token_count == 0u || mutable >= state->page_count || owner->page_count == 0u || state->kv_transactions.logical_pages[(uint64_t)resident * state->kv_transactions.page_capacity + owner->page_count - 1u] != mutable || state->kv_blocks[mutable].residency_reference_count == 0u )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	generation = state->kv_blocks[page].generation;
	start_ns = SparkGlm5NextNowNs();
	status = SparkKvPageStoreReadback(&state->recurrent_store,page,generation,(uintptr_t)state->recurrent_staging,state->recurrent_page_bytes);
	while ( status == SPARK_STATUS_BUSY )
	{
		SparkStatus wait = SparkKvPageStoreWaitForTransfers(&state->recurrent_store);
		if ( wait != SPARK_STATUS_OK )
			SPARK_RETURN(wait);
		status = SparkKvPageStoreReadback(&state->recurrent_store,page,generation,(uintptr_t)state->recurrent_staging,state->recurrent_page_bytes);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextRecurrentCopy(state,SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE,resident,state->recurrent_staging,state->recurrent_page_bytes);
	if ( status == SPARK_STATUS_OK )
		SparkGlm5NextKdaCount(state->kda_restore,state->recurrent_page_bytes,start_ns);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextRestoreCacheLanes(SparkGlm5NextModuleState *state,const SparkGlm5NextAsyncCompletion *async)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t lane;
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for (lane=0u; lane<async->lane_count && status==SPARK_STATUS_OK; lane++)
		status = SparkGlm5NextRestoreRecurrent(state,async->lane_indices[lane]);
	(void)pthread_mutex_unlock(&state->kv_mutex);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextClaimTpChain(SparkGlm5NextModuleState *state,uint32_t *busy_reason)
{
	uint32_t expected = 0u;
	cudaError_t error;
	*busy_reason = SPARK_GLM5_NEXT_BUSY_CHAIN;
	if ( !atomic_compare_exchange_strong_explicit(&state->tp_chain_active,&expected,1u,memory_order_acq_rel,memory_order_acquire) )
		return(SPARK_STATUS_BUSY);
	*busy_reason = SPARK_GLM5_NEXT_BUSY_STREAM;
	error = cudaStreamQuery((cudaStream_t)state->execution_stream);
	if ( error == cudaSuccess )
		return(SPARK_STATUS_OK);
	atomic_store_explicit(&state->tp_chain_active,0u,memory_order_release);
	return(error == cudaErrorNotReady ? SPARK_STATUS_BUSY : SparkGlm5NextTerminalFailure(state,SPARK_STATUS_IO_ERROR,"chain-rearm"));
}

static void SparkGlm5NextStampClaim(SparkGlm5NextModuleState *state,uint32_t slot_index)
{
	SparkGlm5NextAsyncCompletion *async = &state->completions[slot_index];
	async->chain_start_ns = SparkGlm5NextNowNs();
	async->attempt_ns = state->wave_attempt_ns;
	async->retries = state->wave_attempt_retries;
	memcpy(async->busy,state->wave_attempt_busy,sizeof(async->busy));
}

static void SparkGlm5NextStampKeyed(SparkGlm5NextModuleState *state,uint32_t slot_index)
{
	SparkGlm5NextAsyncCompletion *async = &state->completions[slot_index];
	async->keyed_ns = SparkGlm5NextNowNs();
	async->epoch[0] = SparkTpDeviceCollectiveChainEpoch(&state->tp_device_collective);
	async->epoch[1] = SparkTpDeviceCollectiveChainEpoch(&state->tp_device_collective_hc);
}

static SparkStatus SparkGlm5NextStartClaimedBatch(SparkGlm5NextModuleState *state,SparkModelDriverFrame *frame,const SparkGlm5NextResidentDecodeStageFrameContext *context,uint32_t slot_index,uint32_t *busy_reason)
{
	SparkGlm5NextExecutionSlot *slot = &state->slots[slot_index];
	SparkGlm5NextTpChain *chain;
	SparkStatus status;
	cudaError_t error;
	uint32_t retained_index;
	*busy_reason = SPARK_GLM5_NEXT_BUSY_OTHER;
	for ( retained_index = 0u; retained_index < state->pipeline_slot_count; retained_index++ )
		if ( atomic_load_explicit(&state->lazy_retained[retained_index],memory_order_acquire) != 0 )
		{
			SparkGlm5NextScheduleRetainedRetry(state);
			SPARK_FAIL(SPARK_STATUS_BUSY);
		}
	status = SparkGlm5NextClaimTpChain(state,busy_reason);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	*busy_reason = SPARK_GLM5_NEXT_BUSY_OTHER;
	chain = (SparkGlm5NextTpChain *)calloc(1u,sizeof(*chain));
	if ( chain == 0 )
	{
		atomic_store_explicit(&state->tp_chain_active,0u,memory_order_release);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	status = SparkGlm5NextClaimCacheFrame(state,frame,context->batch,state->completions[slot_index].lane_next_positions);
	if ( status != SPARK_STATUS_OK )
	{
		atomic_store_explicit(&state->tp_chain_active,0u,memory_order_release);
		free(chain);
		SPARK_RETURN(status);
	}
	chain->state = state;
	chain->slot = slot;
	chain->slot_index = slot_index;
	chain->frame = frame;
	chain->context = context;
	chain->batch = context->batch;
	SparkGlm5NextStampClaim(state,slot_index);
	chain->wave_rows = context->batch->row_count;
	chain->steps = state->completions[slot_index].steps;
	chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN;
	chain->active = 1u;
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	if ( state->tp_device_collective_initialized != 0u )
		status = SparkTpDeviceCollectiveChainKey(&state->tp_device_collective,chain->frame->request_id & SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK);
	if ( status == SPARK_STATUS_OK && state->tp_device_collective_hc_initialized != 0u )
		status = SparkTpDeviceCollectiveChainKey(&state->tp_device_collective_hc,chain->frame->request_id & SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK);
	SparkGlm5NextStampKeyed(state,slot_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextRestoreCacheLanes(state,&state->completions[slot_index]);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextUploadPageTables(state,&state->completions[slot_index],slot->stream);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemsetAsync(slot->kv_access_error,0,SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t),(cudaStream_t)slot->stream);
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"kv_access_reset");
	}
	if ( status == SPARK_STATUS_OK && chain->wave_rows == 0u )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextMtpDriveDraft(state,frame,context->batch,slot,chain);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextVerifyDriveDraft(state,frame,context->batch,slot,chain);
	if ( status != SPARK_STATUS_OK )
		SparkGlm5NextTpChainFail(chain,status);
	else
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextStageHostSampling(const SparkGlm5NextModuleState *state,SparkGlm5NextExecutionSlot *slot,const SparkGlm5NextResidentDecodeStageBatchView *batch)
{
	uint32_t row;
	slot->sampled = 0u;
	for (row=0u; state->owns_final_head != 0u && row<batch->row_count; row++)
	{
		slot->host_row_sampling[row] = batch->row_sampling[row];
		slot->sampled |= batch->row_sampling[row].inverse_temperature != 0.0f ? 1u : 0u;
	}
}

static void SparkGlm5NextNoteAttempt(SparkGlm5NextModuleState *state,uint64_t request_id)
{
	if ( state->wave_attempt_request == request_id && state->wave_attempt_ns != 0u )
	{
		state->wave_attempt_retries++;
		return;
	}
	state->wave_attempt_request = request_id;
	state->wave_attempt_ns = SparkGlm5NextNowNs();
	state->wave_attempt_retries = 0u;
	memset(state->wave_attempt_busy,0,sizeof(state->wave_attempt_busy));
}

static SparkStatus SparkGlm5NextNoteBusy(SparkGlm5NextModuleState *state,SparkStatus status,uint32_t reason)
{
	if ( status == SPARK_STATUS_BUSY )
		state->wave_attempt_busy[reason < SPARK_GLM5_NEXT_BUSY_REASONS ? reason : SPARK_GLM5_NEXT_BUSY_OTHER]++;
	return(status);
}

static SparkStatus SparkGlm5NextStartSlot(SparkGlm5NextModuleState *state,SparkModelDriverFrame *frame,const SparkGlm5NextResidentDecodeStageFrameContext *context,uint32_t slot_index,const SparkGlm5NextClaimedContinuityContext *continuity,uint32_t *busy_reason)
{
	const SparkGlm5NextResidentDecodeStageBatchView *batch = context->batch;
	SparkGlm5NextExecutionSlot *slot = &state->slots[slot_index];
	SparkStatus status;
	*busy_reason = SPARK_GLM5_NEXT_BUSY_OTHER;
	slot->stream = frame->execution_stream;
	status = SparkGlm5NextStageHostBatch(state,slot,batch);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkGlm5NextStageHostSampling(state,slot,batch);
	SparkGlm5NextPrepareAsyncCompletion(state,frame,batch,continuity->bound,continuity->sequence_ids,continuity->next_positions,slot_index);
	return(SparkGlm5NextStartClaimedBatch(state,frame,context,slot_index,busy_reason));
}

static SparkStatus SparkGlm5NextExecuteBatch(SparkGlm5NextModuleState *state,SparkModelDriverFrame *frame,const SparkGlm5NextResidentDecodeStageFrameContext *context)
{
	const SparkGlm5NextResidentDecodeStageBatchView *batch = context->batch;
	SparkGlm5NextClaimedContinuityContext continuity;
	uint8_t simulated_bound[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t simulated_sequence[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t simulated_next[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t slot_index,busy_reason = SPARK_GLM5_NEXT_BUSY_SLOT;
	uint64_t last_ordinal;
	SparkStatus status;
	SparkGlm5NextNoteAttempt(state,frame->request_id);
	status = SparkGlm5NextWeightdHealth(state);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkTpChainOrdinal(frame->request_id,state->pipeline_slot_count,SPARK_GLM5_NEXT_TP_COLLECTIVE_CREDITS_PER_SLOT,SPARK_GLM5_NEXT_TP_CHAIN_OPERATIONS,SPARK_GLM5_NEXT_TP_CHAIN_OPERATIONS - 1u,&last_ordinal);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"G5N-DBG submit-fail site=ordinal status=%d req=%llu\n",(int)status,(unsigned long long)frame->request_id);
		SPARK_RETURN(status);
	}
	continuity.state = state;
	continuity.batch = batch;
	continuity.bound = simulated_bound;
	continuity.sequence_ids = simulated_sequence;
	continuity.next_positions = simulated_next;
	status = SparkStageModuleIndexSetClaimAndPrepare(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count,SparkGlm5NextPrepareClaimedContinuity,&continuity);
	if ( status != SPARK_STATUS_OK )
	{
		if ( status != SPARK_STATUS_BUSY )
			fprintf(stderr,"G5N-DBG submit-fail site=lane-claim status=%d req=%llu rows=%u\n",(int)status,(unsigned long long)frame->request_id,(unsigned)batch->active_sequence_count);
		SPARK_RETURN(SparkGlm5NextNoteBusy(state,status,SPARK_GLM5_NEXT_BUSY_LANES));
	}
	slot_index = (uint32_t)(frame->request_id % state->pipeline_slot_count);
	status = SparkStageModuleIndexSetClaim(state->slot_states,state->pipeline_slot_count,&slot_index,1u);
	if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY )
		fprintf(stderr,"G5N-DBG submit-fail site=slot-claim status=%d req=%llu slot=%u\n",(int)status,(unsigned long long)frame->request_id,(unsigned)slot_index);
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkGlm5NextStartSlot(state,frame,context,slot_index,&continuity,&busy_reason);
		if ( status != SPARK_STATUS_OK )
			SparkStageModuleSlotRelease(state->slot_states,slot_index);
	}
	if ( status != SPARK_STATUS_OK )
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
	SPARK_RETURN(SparkGlm5NextNoteBusy(state,status,busy_reason));
}

static SparkStatus SparkGlm5NextPublishCache(SparkGlm5NextModuleState *state,SparkModelDriverFrame *frame)
{
	SparkGlm5NextAsyncCompletion *async;
	SparkModelDriverCompletion completion = {0};
	uint32_t indices[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT],lane,resident,slot;
	SparkStatus status;
	if ( state == 0 || frame == 0 || frame->completion_function == 0 || frame->execution_stream != state->execution_stream || state->pipeline_slot_count == 0u || frame->flags != (SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH | SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) || frame->new_token_count != 0u || frame->tokens_per_sequence != 0u || frame->cache_lanes == 0 || frame->cache_lane_count == 0u || frame->cache_lane_count != frame->active_slot_count || frame->cache_lane_count > state->resident_sequence_capacity )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	slot = (uint32_t)(frame->request_id % state->pipeline_slot_count);
	if ( frame->driver_dispatch_slot != slot )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	status = SparkGlm5NextWeightdHealth(state);
	if ( status != SPARK_STATUS_OK ) return(status);
	for (lane=0u; lane<frame->cache_lane_count; lane++)
		indices[lane] = frame->cache_lanes[lane].resident_sequence_slot;
	status = SparkStageModuleIndexSetClaim(state->lane_states,state->resident_sequence_capacity,indices,frame->cache_lane_count);
	if ( status != SPARK_STATUS_OK ) return(status);
	status = SparkStageModuleIndexSetClaim(state->slot_states,state->pipeline_slot_count,&slot,1u);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,indices,frame->cache_lane_count);
		return(status);
	}
	for (lane=0u; lane<frame->cache_lane_count && status==SPARK_STATUS_OK; lane++)
	{
		const SparkModelDriverCacheLane *cache_lane = &frame->cache_lanes[lane];
		resident = indices[lane];
		if ( cache_lane->flags != SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH || cache_lane->publish_token_count == 0u || cache_lane->sequence_position != cache_lane->publish_token_count || cache_lane->context_token_count != cache_lane->publish_token_count || atomic_load(&state->lane_bound[resident]) == 0u || atomic_load(&state->lane_sequence_ids[resident]) != cache_lane->sequence_id || atomic_load(&state->lane_next_positions[resident]) != cache_lane->sequence_position )
			status = SPARK_STATUS_VALIDATION_FAILED;
	}
	if ( status == SPARK_STATUS_OK )
	{
		if ( pthread_mutex_lock(&state->kv_mutex) != 0 ) status = SPARK_STATUS_INTERNAL_ERROR;
		else
		{
			status = SparkKvLaneTransactionsClaim(&state->kv_transactions,frame);
			(void)pthread_mutex_unlock(&state->kv_mutex);
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		async = &state->completions[slot];
		memset(async,0,sizeof(*async));
		async->state = state;
		async->slot_index = slot;
		async->lane_count = frame->cache_lane_count;
		for (lane=0u; lane<async->lane_count; lane++)
		{
			async->lane_indices[lane] = indices[lane];
			async->lane_bound[lane] = 1u;
			async->lane_sequence_ids[lane] = frame->cache_lanes[lane].sequence_id;
			async->lane_next_positions[lane] = frame->cache_lanes[lane].context_token_count;
		}
		atomic_fetch_add(&state->submitted_count,1u);
		completion.status = SparkGlm5NextFinishCacheLanes(async);
		completion.request_id = frame->request_id;
		completion.sequence_id = frame->sequence_id;
		completion.sequence_position = frame->sequence_position;
		completion.program_id = frame->program_id;
		completion.driver_dispatch_slot = slot;
		completion.residency = frame->residency;
		atomic_fetch_add(completion.status == SPARK_STATUS_OK ? &state->completed_count : &state->failed_count,1u);
	}
	SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,indices,frame->cache_lane_count);
	SparkStageModuleSlotRelease(state->slot_states,slot);
	if ( status == SPARK_STATUS_OK ) frame->completion_function(frame->completion_context,&completion);
	return(status);
}

SparkStatus SparkGlm5NextResidentDecodeStageExecute(
	void *module_state,
	SparkModelDriverFrame *frame)
{
	SparkGlm5NextModuleState *state;
	const SparkGlm5NextResidentDecodeStageFrameContext *context;
	SparkStatus status;
	state = (SparkGlm5NextModuleState *)module_state;
	context = 0;
	if ( frame != 0 && (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH) != 0u )
		return(SparkGlm5NextPublishCache(state,frame));
	status = SparkGlm5NextValidateFrame(state,frame,&context);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"G5N-DBG execute: ValidateFrame -> %d\n",(int)status);
		if ( state != 0 )
			atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		SPARK_RETURN(status);
	}
	status = SparkGlm5NextExecuteBatch(state,frame,context);
	if ( status != SPARK_STATUS_OK )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	SPARK_RETURN(status);
}

static void SparkGlm5NextAdmissionCost(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	(void)context;
	decision->host_staging_bytes = (uint64_t)request->new_token_count *
		(sizeof(uint32_t) * 3u + sizeof(uint64_t) * 2u + sizeof(SparkRowSampling));
	decision->device_memcpy_bytes = decision->host_staging_bytes;
}

static SparkStatus SparkGlm5NextResetExecutionState(SparkGlm5NextModuleState *state)
{
	cudaError_t error = cudaSuccess,drain;
	uint32_t lane;
	if ( state->kda_layer_count != 0u )
	{
		error = cudaMemsetAsync(state->kda_state_pools,0,state->kda_state_layer_stride_bytes * state->kda_layer_count,(cudaStream_t)state->execution_stream);
		if ( error == cudaSuccess )
			error = cudaMemsetAsync(state->kda_window_pools,0,state->kda_window_layer_stride_bytes * state->kda_layer_count * 3u,(cudaStream_t)state->execution_stream);
	}
	drain = cudaStreamSynchronize((cudaStream_t)state->execution_stream);
	if ( drain != cudaSuccess )
	{
		(void)SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,drain,"reset_stream_drain");
		SPARK_FAIL(SPARK_STATUS_PENDING);
	}
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"cache_reset"));
	memset(state->page_table_shadow,0xff,(uint64_t)state->resident_sequence_capacity * state->pages_per_sequence * sizeof(uint32_t));
	for (lane=0u; lane<state->resident_sequence_capacity; lane++)
	{
		atomic_store_explicit(&state->lane_bound[lane],0u,memory_order_release);
		atomic_store_explicit(&state->lane_sequence_ids[lane],0u,memory_order_release);
		atomic_store_explicit(&state->lane_next_positions[lane],0u,memory_order_release);
		if ( state->mtp_lane_armed != 0 )
			state->mtp_lane_armed[lane] = 0u;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextResetClaimed(SparkGlm5NextModuleState *state,uint64_t generation)
{
	SparkStatus status;
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SPARK_STATUS_VALIDATION_FAILED;
	if ( generation > state->reset_generation )
	{
		status = SparkKvLaneTransactionsReset(&state->kv_transactions);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextResetExecutionState(state);
		if ( status == SPARK_STATUS_OK )
		{
			state->reset_generation = generation;
			state->control_generation = 0u;
		}
	}
	(void)pthread_mutex_unlock(&state->kv_mutex);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextReset(SparkGlm5NextModuleState *state,const SparkModelDriverAdmissionRequest *request)
{
	uint32_t slots[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t lanes[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t index;
	SparkStatus status;
	if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	/* A reset kills this rank's in-flight chains; every peer is potentially
	 * waiting on this rank's cells for those chains. Cancel first so the
	 * peers' waits fail fast (CKEY-CANCEL) instead of wedging 30s per chain
	 * — the session-reset cascade turned one rank's reconnect into a
	 * fleet-wide stall. */
	SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective);
	if ( state->tp_device_collective_hc_initialized != 0u )
		SparkTpDeviceCollectiveBroadcastCancel(&state->tp_device_collective_hc);
	for (index=0u; index<state->pipeline_slot_count; index++)
		slots[index] = index;
	for (index=0u; index<state->resident_sequence_capacity; index++)
		lanes[index] = index;
	status = SparkStageModuleIndexSetClaim(state->slot_states,state->pipeline_slot_count,slots,state->pipeline_slot_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkStageModuleIndexSetClaim(state->lane_states,state->resident_sequence_capacity,lanes,state->resident_sequence_capacity);
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkGlm5NextResetClaimed(state,request->control_generation);
		if ( status == SPARK_STATUS_PENDING )
		{
			fprintf(stderr,"GLM reset stream not quiescent; retaining lane and slot ownership\n");
			SPARK_RETURN(status);
		}
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,lanes,state->resident_sequence_capacity);
	}
	SparkStageModuleIndexSetRelease(state->slot_states,state->pipeline_slot_count,slots,state->pipeline_slot_count);
	SPARK_RETURN(status);
}

SparkStatus SparkGlm5NextResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGlm5NextModuleState *state;
	SparkAdmissionPolicyTable table;
	uint32_t available;
	SparkStatus status;
	state = (SparkGlm5NextModuleState *)module_state;
	if ( state == 0 || request == 0 || decision == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkGlm5NextWeightdHealth(state);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET )
	{
		SparkModelDriverInitializeAdmissionDecision(decision);
		status = SparkGlm5NextReset(state,request);
		if ( status == SPARK_STATUS_OK )
		{
			decision->accepted = 1u;
			decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
		}
		SPARK_RETURN(status);
	}
	available = request->request_id != 0u && atomic_load_explicit(&state->slot_states[request->request_id % state->pipeline_slot_count],memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_FREE ? 1u : 0u;
	if ( (request->frame_flags & (SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE | SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_PUBLISH)) != 0u )
	{
		if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u || request->new_token_count != 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		SparkModelDriverInitializeAdmissionDecision(decision);
		decision->available_dispatch_slot_count = available;
		return(SparkGlm5NextAdmissionPredicate(state,request,decision));
	}
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_ADMISSION_ABI_VERSION;
	table.descriptor_bytes = (uint32_t)sizeof(table);
	table.max_active_sequence_count = state->resident_sequence_capacity;
	table.max_input_row_count = state->execution_row_capacity;
	table.max_sequence_positions = state->max_sequence_positions;
	table.flags = SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS |
		SPARK_ADMISSION_POLICY_FLAG_ALLOW_DISPATCH_FLAG;
	table.predicate = SparkGlm5NextAdmissionPredicate;
	table.predicate_context = state;
	table.cost = SparkGlm5NextAdmissionCost;
	table.cost_context = state;
	status = SparkAdmissionEvaluateShape(&table,available,request,decision);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( decision->accepted == 0u )
		fprintf(stderr,"G5N-DBG admit: shape-rejected reason %u\n",
			(unsigned)decision->rejection_reason);
	if ( decision->accepted == 0u )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	SPARK_RETURN(status);
}

SparkStatus SparkGlm5NextResidentDecodeStageSnapshot(
	void *module_state,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkGlm5NextModuleState *state;
	uint32_t index,resident_count;
	state = (SparkGlm5NextModuleState *)module_state;
	if ( state == 0 || snapshot == 0 || program_id == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkStageModuleRuntimeSnapshotInitialize(snapshot,program_id,state->slot_states,state->pipeline_slot_count);
	snapshot->submitted_count = atomic_load_explicit(&state->submitted_count,memory_order_relaxed);
	snapshot->completed_count = atomic_load_explicit(&state->completed_count,memory_order_relaxed);
	snapshot->rejected_count = atomic_load_explicit(&state->rejected_count,memory_order_relaxed);
	snapshot->host_callback_completion_count = atomic_load_explicit(&state->host_callback_completion_count,memory_order_relaxed);
	resident_count = 0u;
	for (index=0u; index<state->resident_sequence_capacity; index++)
		resident_count += atomic_load_explicit(&state->lane_bound[index],memory_order_acquire) != 0u ? 1u : 0u;
	snapshot->resident_sequence_count = resident_count;
	snapshot->kv_token_capacity = (uint64_t)state->page_count * SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT;
	return(SparkGlm5NextWeightdHealth(state));
}

static SparkStatus SparkGlm5NextReleaseCaches(SparkGlm5NextModuleState *state)
{
	if ( state->stream_wait.initialized != 0u && SparkStageModuleCudaWaitDestroy(&state->stream_wait) != SPARK_STATUS_OK )
		return(SPARK_STATUS_BUSY);
	if ( state->completion_queue_lock_initialized != 0u )
	{
		(void)pthread_mutex_destroy(&state->completion_queue_lock);
		state->completion_queue_lock_initialized = 0u;
	}
	if ( state->recurrent_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&state->recurrent_store);
	if ( state->recurrent_staging != 0 )
		(void)cudaFreeHost(state->recurrent_staging);
	if ( state->kv_page_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&state->kv_page_store);
	free(state->kda_state_index_host);
	free(state->kv_blocks);
	free(state->kv_resident_slot_logical_block_indices);
	free(state->kv_entries);
	free(state->kv_sequences);
	free(state->kv_hash_bucket_heads);
	free(state->kv_entry_indices_by_logical_page);
	free(state->kv_page_staging);
	free(state->kv_lane_logical_pages);
	free(state->page_table_shadow);
	free(state->kv_lane_transactions);
	if ( state->kv_lane_physical_pages != 0 )
		(void)cudaFreeHost(state->kv_lane_physical_pages);
	if ( state->kv_mutex_initialized != 0u )
		(void)pthread_mutex_destroy(&state->kv_mutex);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextReleaseCollectives(SparkGlm5NextModuleState *state)
{
	if ( state->tp_device_collective_hc_initialized != 0u )
	{
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective_hc);
		if ( state->tp_device_collective_hc.implementation != 0 ) return(SPARK_STATUS_IO_ERROR);
		state->tp_device_collective_hc_initialized = 0u;
	}
	if ( state->tp_hc_host_credit_send_bf16 != 0 )
		(void)cudaFreeHost(state->tp_hc_host_credit_send_bf16);
	if ( state->tp_hc_host_credit_receive_bf16 != 0 )
		(void)cudaFreeHost(state->tp_hc_host_credit_receive_bf16);
	state->tp_hc_host_credit_send_bf16 = 0;
	state->tp_hc_host_credit_receive_bf16 = 0;
	if ( state->tp_device_collective_initialized != 0u )
	{
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
		if ( state->tp_device_collective.implementation != 0 ) return(SPARK_STATUS_IO_ERROR);
		state->tp_device_collective_initialized = 0u;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextRouteTraceOpen(SparkGlm5NextModuleState *state)
{
	const char *prefix = getenv("SPARK_GLM5_NEXT_ROUTE_TRACE");
	const char *pin = getenv("SPARK_GLM5_NEXT_PIN_EXPERTS");
	char path[4096];
	if ( prefix == 0 || prefix[0] == '\0' )
		return(SPARK_STATUS_OK);
	if ( state->graph_path_requested != 0u || (pin != 0 && pin[0] == '1') || state->lazy_pack == 0 )
	{
		fprintf(stderr,"SPARK_GLM5_NEXT_ROUTE_TRACE requires lazy eager decode: SPARK_GLM5_NEXT_GRAPH_PATH=0, experts not pinned, a lazy expert pack\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( snprintf(path,sizeof(path),"%s.stage%02u.rank%02u.trace",prefix,state->stage_index,state->tp_rank) >= (int)sizeof(path) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state->route_trace = fopen(path,"w");
	if ( state->route_trace == 0 || setvbuf(state->route_trace,0,_IOLBF,0) != 0 )
	{
		fprintf(stderr,"SPARK_GLM5_NEXT_ROUTE_TRACE cannot open %s: %s\n",path,strerror(errno));
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	fprintf(stderr,"GLM route trace path=%s\n",path);
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextRouteTraceClose(SparkGlm5NextModuleState *state)
{
	if ( state->route_trace != 0 )
		(void)fclose(state->route_trace);
	state->route_trace = 0;
}

void SparkGlm5NextResidentDecodeStageDestroy(void *module_state)
{
	SparkGlm5NextModuleState *state;
	uint32_t slot;
	state = (SparkGlm5NextModuleState *)module_state;
	if ( state == 0 )
		return;
	for (slot=0u; slot<state->pipeline_slot_count; slot++)
		if ( atomic_load_explicit(&state->lazy_retained[slot],memory_order_acquire) != 0 )
			break;
	if ( slot < state->pipeline_slot_count )
	{
		if ( state->lazy_pack == 0 || state->lazy_pack->worker == 0 || SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkGlm5NextLazyRetryRetained,state) != SPARK_STATUS_OK )
			return;
	}
	if ( SparkStageModuleWaitForSlots(SPARK_GLM5_NEXT_MODULE_TAG,state->slot_states,state->pipeline_slot_count,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK )
		return;
	if ( state->lazy_pack != 0 && state->lazy_pack->worker != 0 && SparkWeightdWorkerWaitIdle(state->lazy_pack->worker,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK )
		return;
	if ( state->completion_worker != 0 )
	{
		if ( SparkWeightdWorkerWaitIdle(state->completion_worker,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK || SparkWeightdWorkerDestroy(state->completion_worker) != SPARK_STATUS_OK )
			return;
		state->completion_worker = 0;
	}
	if ( SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,cudaStreamSynchronize((cudaStream_t)state->execution_stream),"destroy_stream_drain") != SPARK_STATUS_OK )
		return;
#ifdef SPARK_SCORE_DUMP
	SparkGlm5NextScoreClose(state);
#endif
	if ( SparkGlm5NextReleaseCollectives(state) != SPARK_STATUS_OK )
		return;
	if ( state->lazy_pack != 0 )
	{
		if ( SparkGlm5NextReleasePinnedExperts(state) != SPARK_STATUS_OK || SparkWeightdLazyPackDestroy(state->lazy_pack) != SPARK_STATUS_OK )
			return;
		state->lazy_pack = 0;
	}
	if ( state->lane_client != 0 )
	{
		(void)SparkWeightdClientClose(state->lane_client);
		state->lane_client = 0;
	}
	if ( SparkGlm5NextReleaseCaches(state) != SPARK_STATUS_OK )
		return;
	SparkGlm5NextReleaseTaps(state);
	SparkGlm5NextReleaseSlotHost(state);
	SparkGlm5NextRouteTraceClose(state);
	SparkExpertWorkingSetDestroy(&state->expert_ws);
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state->mtp_lane_armed);
	SparkGlm5NextReleaseDrafter(state);
	free(state);
}

static SparkStatus SparkGlm5NextBuildHeadShadow(SparkGlm5NextModuleState *state)
{
	uint64_t head_rows,dim;
	SparkStatus status;
	if ( state->owns_final_head == 0u || state->lm_head_bf16 == 0 )
		return(SPARK_STATUS_OK);
	head_rows = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT / state->tp_degree;
	dim = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	status = SparkGlm5NextAllocateBytes(state,head_rows,dim,1u,(void **)&state->head_certified_fp8_payload);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateBytes(state,head_rows,dim / 32u,sizeof(float),(void **)&state->head_certified_fp8_scale_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateBytes(state,head_rows,dim / 32u,sizeof(float),(void **)&state->head_certified_fp8_norm_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,SparkGlm5NextLaunchHeadCertifiedQuantize(0,state->lm_head_bf16,state->head_certified_fp8_payload,state->head_certified_fp8_scale_f32,state->head_certified_fp8_norm_f32,(uint32_t)head_rows,(uint32_t)dim),"head_certified_quantize");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,cudaDeviceSynchronize(),"head_certified_sync");
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextConfigureVerify(SparkGlm5NextModuleState *state)
{
	if ( SparkGlm5NextVerifyRowsParse(getenv(SPARK_GLM5_NEXT_VERIFY_ROWS_ENV),&state->verify_rows_max) != SPARK_STATUS_OK || (state->verify_rows_max != 0u && state->graph_path_enabled == 0u) )
	{
		fprintf(stderr,"%s must be 0 or %u..%u, and a nonzero value needs SPARK_GLM5_NEXT_GRAPH_PATH=1\n",SPARK_GLM5_NEXT_VERIFY_ROWS_ENV,SPARK_GLM5_NEXT_VERIFY_ROWS_MIN,SPARK_GLM5_NEXT_VERIFY_ROWS_MAX);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( SparkGlm5NextVerifyDrafterParse(getenv(SPARK_GLM5_NEXT_VERIFY_DRAFTER_ENV),state->verify_rows_max,&state->verify_drafter,&state->verify_drafter_path) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s must be lookup, mtp, mtp+lookup, oracle:PATH, adversary:PATH or recorded:PATH when %s is nonzero, and absent when it is 0\n",SPARK_GLM5_NEXT_VERIFY_DRAFTER_ENV,SPARK_GLM5_NEXT_VERIFY_ROWS_ENV);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->verify_mtp = SparkGlm5NextVerifyDrafterUsesMtp(state->verify_drafter);
	fprintf(stderr,"GLM verify regime rows=%u drafter=%u\n",state->verify_rows_max,state->verify_drafter);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextLoadReferenceTokens(SparkGlm5NextModuleState *state,uint32_t *count_out)
{
	FILE *file;
	long bytes;
	uint32_t count;
	*count_out = 0u;
	file = fopen(state->verify_drafter_path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"GLM verify reference drafter cannot open %s\n",state->verify_drafter_path);
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	bytes = fseek(file,0,SEEK_END) == 0 ? ftell(file) : -1;
	if ( bytes < (long)(2u * sizeof(uint32_t)) || (uint64_t)bytes % sizeof(uint32_t) != 0u || (uint64_t)bytes / sizeof(uint32_t) > state->max_sequence_positions || fseek(file,0,SEEK_SET) != 0 )
	{
		fclose(file);
		fprintf(stderr,"GLM verify reference drafter %s must hold 2..%u little-endian uint32 token ids from position 0\n",state->verify_drafter_path,state->max_sequence_positions);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	count = (uint32_t)((uint64_t)bytes / sizeof(uint32_t));
	state->verify_reference_tokens = (uint32_t *)malloc((uint64_t)count * sizeof(uint32_t));
	if ( state->verify_reference_tokens == 0 || fread(state->verify_reference_tokens,sizeof(uint32_t),count,file) != count )
	{
		fclose(file);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	fclose(file);
	*count_out = count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextLoadRecordedDrafts(SparkGlm5NextModuleState *state)
{
	FILE *file;
	long bytes;
	SparkStatus status;
	file = fopen(state->verify_drafter_path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"GLM verify recorded drafter cannot open %s\n",state->verify_drafter_path);
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	}
	bytes = fseek(file,0,SEEK_END) == 0 ? ftell(file) : -1;
	if ( bytes < (long)SPARK_SPECULATION_RECORDED_HEADER_BYTES || fseek(file,0,SEEK_SET) != 0 )
	{
		fclose(file);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	state->verify_recorded_bytes = (uint8_t *)malloc((size_t)bytes);
	if ( state->verify_recorded_bytes == 0 || fread(state->verify_recorded_bytes,1u,(size_t)bytes,file) != (size_t)bytes )
	{
		fclose(file);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	fclose(file);
	status = SparkSpeculationRecordedDraftInitialize(&state->verify_recorded,state->verify_recorded_bytes,(uint64_t)bytes,SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"GLM verify recorded drafter %s is not a draft table for vocabulary %u\n",state->verify_drafter_path,(unsigned)SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT);
	else
		fprintf(stderr,"GLM verify recorded drafter %s entries=%llu depth=%u\n",state->verify_drafter_path,(unsigned long long)state->verify_recorded.entry_count,state->verify_recorded.depth);
	SPARK_RETURN(status);
}

static SparkStatus SparkGlm5NextAllocateDrafter(SparkGlm5NextModuleState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t count;
	if ( state->verify_drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_NONE )
		return(SPARK_STATUS_OK);
	state->verify_depth_cap = (uint32_t *)calloc(state->resident_sequence_capacity,sizeof(uint32_t));
	state->verify_depth_sequence = (uint64_t *)calloc(state->resident_sequence_capacity,sizeof(uint64_t));
	if ( state->verify_depth_cap == 0 || state->verify_depth_sequence == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( SparkGlm5NextVerifyDrafterUsesLookup(state->verify_drafter) != 0u )
	{
		status = SparkSpeculationLookupDraftInitialize(&state->verify_lookup,state->resident_sequence_capacity,state->max_sequence_positions,SPARK_GLM5_NEXT_VERIFY_LOOKUP_MIN_MATCH,SPARK_GLM5_NEXT_VERIFY_LOOKUP_MAX_MATCH);
		state->verify_draft_function = SparkSpeculationLookupDraftTokens;
		state->verify_draft_context = &state->verify_lookup;
	}
	if ( status == SPARK_STATUS_OK && state->verify_drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP )
	{
		state->verify_draft_function = SparkGlm5NextMtpDraftTokens;
		state->verify_draft_context = state;
	}
	if ( status == SPARK_STATUS_OK && state->verify_drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_MTP_LOOKUP )
	{
		status = SparkSpeculationDrafterMixInitialize(&state->verify_mix,SparkSpeculationLookupDraftTokens,&state->verify_lookup,SparkGlm5NextMtpDraftTokens,state,
			state->verify_rows_max - 1u < SPARK_GLM5_NEXT_VERIFY_MIX_LOOKUP_MIN_TOKENS ? state->verify_rows_max - 1u : SPARK_GLM5_NEXT_VERIFY_MIX_LOOKUP_MIN_TOKENS,
			state->resident_sequence_capacity,state->verify_rows_max - 1u);
		state->verify_draft_function = SparkSpeculationDrafterMixTokens;
		state->verify_draft_context = &state->verify_mix;
	}
	else if ( state->verify_drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE || state->verify_drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ADVERSARY )
	{
		status = SparkGlm5NextLoadReferenceTokens(state,&count);
		if ( status == SPARK_STATUS_OK )
			status = SparkSpeculationReferenceDraftInitialize(&state->verify_reference,
				state->verify_drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_ORACLE ? SPARK_SPECULATION_REFERENCE_ORACLE : SPARK_SPECULATION_REFERENCE_ADVERSARY,
				SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,0u,state->verify_reference_tokens,count);
		state->verify_draft_function = SparkSpeculationReferenceDraftTokens;
		state->verify_draft_context = &state->verify_reference;
	}
	else if ( state->verify_drafter == SPARK_GLM5_NEXT_VERIFY_DRAFTER_RECORDED )
	{
		status = SparkGlm5NextLoadRecordedDrafts(state);
		state->verify_draft_function = SparkSpeculationRecordedDraftTokens;
		state->verify_draft_context = &state->verify_recorded;
	}
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"GLM verify drafter %u failed to initialize: status=%d\n",state->verify_drafter,(int)status);
	SPARK_RETURN(status);
}

static void SparkGlm5NextReleaseDrafter(SparkGlm5NextModuleState *state)
{
	SparkSpeculationLookupDraftDestroy(&state->verify_lookup);
	SparkSpeculationDrafterMixDestroy(&state->verify_mix);
	free(state->mtp_lane_sequence);
	free(state->mtp_lane_next);
	state->mtp_lane_sequence = 0;
	state->mtp_lane_next = 0;
	if ( state->verify_recorded_bytes != 0 )
		fprintf(stderr,"VERIFY-RECORDED hits=%llu misses=%llu entries=%llu\n",(unsigned long long)state->verify_recorded.hits,(unsigned long long)state->verify_recorded.misses,(unsigned long long)state->verify_recorded.entry_count);
	free(state->verify_reference_tokens);
	free(state->verify_recorded_bytes);
	state->verify_recorded_bytes = 0;
	free(state->verify_depth_cap);
	free(state->verify_depth_sequence);
	state->verify_reference_tokens = 0;
	state->verify_depth_cap = 0;
	state->verify_depth_sequence = 0;
}

static SparkStatus SparkGlm5NextValidateSpeculation(const SparkGlm5NextModuleState *state)
{
	if ( state->mtp_enabled != 0u && state->pack_has_mtp == 0u )
	{
		fprintf(stderr,"G5N-DBG config: MTP flag set but the pack carries no layer-45 tensors\n");
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( state->mtp_enabled != 0u && state->tp_degree != 1u )
	{
		fprintf(stderr,"G5N-DBG config: MTP speculation requires tp_degree 1 in this revision\n");
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	if ( state->verify_rows_max == 0u )
		return(SPARK_STATUS_OK);
	if ( state->verify_mtp != 0u && (state->mtp_sidecar == 0u || state->pack_has_mtp != 0u || state->verify_rows_max - 1u > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_CHAIN_MAX) )
	{
		fprintf(stderr,"GLM verify drafter %u needs the resident MTP pack from %s and a main pack without layer-45 tensors (sidecar=%u pack_mtp=%u rows=%u)\n",state->verify_drafter,SPARK_GLM5_NEXT_VERIFY_MTP_DIR_ENV,state->mtp_sidecar,state->pack_has_mtp,state->verify_rows_max);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	if ( state->tp_degree < 2u || state->tp_collective_disabled != 0u || state->mtp_enabled != 0u || state->owns_embedding == 0u || state->owns_final_head == 0u || state->lazy_pack == 0 || state->execution_row_capacity < SPARK_GLM5_NEXT_VERIFY_ROWS_MAX || SparkGlm5NextPinnedExpected(state) == 0u || state->expert_pin_key_count != SparkGlm5NextPinnedExpected(state) )
	{
		fprintf(stderr,"GLM verify regime rows=%u needs a single-stage TP>=2 lazy pack with pinned experts, no MTP and execution_row_capacity >= %u (tp=%u mtp=%u rows=%u pinned=%u/%u)\n",state->verify_rows_max,SPARK_GLM5_NEXT_VERIFY_ROWS_MAX,state->tp_degree,state->mtp_enabled,state->execution_row_capacity,state->expert_pin_key_count,SparkGlm5NextPinnedExpected(state));
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm5NextEnvDecimal(const char *name,const char *text,uint32_t *value)
{
	char *end = 0;
	unsigned long long parsed;
	if ( text == 0 || text[0] < '0' || text[0] > '9' )
	{
		fprintf(stderr,"%s must be a decimal number, found '%s'\n",name,text != 0 ? text : "");
		return(0u);
	}
	errno = 0;
	parsed = strtoull(text,&end,10);
	if ( errno != 0 || end == text || *end != '\0' || parsed > UINT32_MAX )
	{
		fprintf(stderr,"%s must be a decimal number below 2^32, found '%s'\n",name,text);
		return(0u);
	}
	*value = (uint32_t)parsed;
	return(1u);
}

static SparkStatus SparkGlm5NextConfigureL2Prefetch(SparkGlm5NextModuleState *state)
{
	const char *l2_env = getenv("SPARK_GLM5_NEXT_L2_PREFETCH");
	const char *bytes_env = getenv("SPARK_GLM5_NEXT_L2_PREFETCH_BYTES");
	const char *blocks_env = getenv("SPARK_GLM5_NEXT_L2_PREFETCH_BLOCKS");
	if ( l2_env != 0 && strcmp(l2_env,"0") != 0 && strcmp(l2_env,"1") != 0 )
	{
		fprintf(stderr,"SPARK_GLM5_NEXT_L2_PREFETCH must be 0 or 1\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->l2_prefetch = l2_env != 0 && strcmp(l2_env,"0") == 0 ? 0u : 1u;
	if ( state->l2_prefetch == 0u && (bytes_env != 0 || blocks_env != 0) )
	{
		fprintf(stderr,"SPARK_GLM5_NEXT_L2_PREFETCH_BYTES and SPARK_GLM5_NEXT_L2_PREFETCH_BLOCKS shape the prefetch; they need SPARK_GLM5_NEXT_L2_PREFETCH=1 or unset\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->l2_prefetch_shape.bytes = SPARK_GLM5_NEXT_L2_PREFETCH_BYTES_DEFAULT;
	state->l2_prefetch_shape.blocks = SPARK_GLM5_NEXT_L2_PREFETCH_BLOCKS_DEFAULT;
	if ( bytes_env != 0 && SparkGlm5NextEnvDecimal("SPARK_GLM5_NEXT_L2_PREFETCH_BYTES",bytes_env,&state->l2_prefetch_shape.bytes) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( blocks_env != 0 && SparkGlm5NextEnvDecimal("SPARK_GLM5_NEXT_L2_PREFETCH_BLOCKS",blocks_env,&state->l2_prefetch_shape.blocks) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkGlm5NextL2PrefetchShapeValid(&state->l2_prefetch_shape) == 0u )
	{
		fprintf(stderr,"SPARK_GLM5_NEXT_L2_PREFETCH_BYTES must be a multiple of %u from %u to %u and SPARK_GLM5_NEXT_L2_PREFETCH_BLOCKS from 1 to %u, found bytes=%u blocks=%u\n",
			SPARK_GLM5_NEXT_L2_PREFETCH_BYTES_STEP,SPARK_GLM5_NEXT_L2_PREFETCH_BYTES_STEP,SPARK_GLM5_NEXT_L2_PREFETCH_BYTES_MAX,SPARK_GLM5_NEXT_L2_PREFETCH_BLOCKS_MAX,
			state->l2_prefetch_shape.bytes,state->l2_prefetch_shape.blocks);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( state->l2_prefetch != 0u )
		fprintf(stderr,"GLM l2 prefetch=on bytes=%u blocks=%u\n",state->l2_prefetch_shape.bytes,state->l2_prefetch_shape.blocks);
	else
		fprintf(stderr,"GLM l2 prefetch=off\n");
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextConfigureExecution(SparkGlm5NextModuleState *state)
{
	{
		const char *prefetch = getenv("SPARK_GLM5_NEXT_PREFETCH");
		if ( prefetch != 0 && strcmp(prefetch,"0") != 0 )
		{
			fprintf(stderr,"SPARK_GLM5_NEXT_PREFETCH is unsupported; warm the daemon with sparkpipe_weightd_warm before serving\n");
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		}
	}
	{
		SparkStatus l2_status = SparkGlm5NextConfigureL2Prefetch(state);
		if ( l2_status != SPARK_STATUS_OK )
			SPARK_RETURN(l2_status);
	}
	{
		const char *graph_env = getenv("SPARK_GLM5_NEXT_GRAPH_PATH");
		const char *record_limit_env =
		    getenv("SPARK_GLM5_NEXT_GRAPH_RECORD_OPS");
		if ( graph_env == 0 || (strcmp(graph_env,"0") != 0 && strcmp(graph_env,"1") != 0) )
		{
			fprintf(stderr,"SPARK_GLM5_NEXT_GRAPH_PATH must be 0 or 1\n");
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		state->graph_path_enabled = strcmp(graph_env,"1") == 0 ? 1u : 0u;
		state->graph_path_requested = state->graph_path_enabled;
		fprintf(stderr,"GLM execution mode=%s\n",state->graph_path_enabled != 0u ? "graph" : "eager");
		state->graph_record_limit = record_limit_env != 0 ?
		    (uint32_t)strtoul(record_limit_env,0,10) : 0u;
	}
	return(SparkGlm5NextConfigureVerify(state));
}

static SparkStatus SparkGlm5NextInitializeState(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	SparkGlm5NextModuleState **state_out)
{
	SparkGlm5NextModuleState *state;
	const char *pack_path;
	SparkStatus status;
	uint32_t lane;
	state = (SparkGlm5NextModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->ledger.module_tag = SPARK_GLM5_NEXT_MODULE_TAG;
	atomic_init(&state->terminal_status,SPARK_STATUS_OK);
	atomic_init(&state->tp_chain_active,0u);
	status = SparkGlm5NextConfigureExecution(state);
	if ( status != SPARK_STATUS_OK )
	{
		free(state);
		SPARK_RETURN(status);
	}
	for (lane=0u; lane<SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT; lane++)
		atomic_init(&state->lazy_retained[lane],0);
	status = SparkGlm5NextModuleConfigure(state,configuration,host_services,&pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaWaitInitialize(&state->stream_wait,(cudaStream_t)state->execution_stream);
	if ( status == SPARK_STATUS_OK && SparkGlm5NextConfigureCudaModule(&state->multiprocessor_count) != 0 )
		status = SPARK_STATUS_TARGET_MISMATCH;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackLoad(state,pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextMtpPackOpen(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextRouteTraceOpen(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextWsOpen(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextValidateSpeculation(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateDrafter(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateCaches(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateSlots(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateReplay(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateMtp(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextConfigureTaps(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextModuleInitializeTpCollective(state,(const SparkGlm5NextResidentDecodeStageNodeContext *)host_services->node_context);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextBuildHeadShadow(state);
#ifdef SPARK_SCORE_DUMP
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextScoreOpen(state,(const SparkGlm5NextResidentDecodeStageNodeContext *)host_services->node_context);
#endif
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdWorkerCreate(&state->completion_worker);

	if ( status != SPARK_STATUS_OK )
	{
#ifdef SPARK_SCORE_DUMP
		SparkGlm5NextScoreClose(state);
#endif
		if ( SparkGlm5NextReleaseCollectives(state) != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( state->lazy_pack != 0 && (SparkGlm5NextReleasePinnedExperts(state) != SPARK_STATUS_OK || SparkWeightdLazyPackDestroy(state->lazy_pack) != SPARK_STATUS_OK) )
		{
			fprintf(stderr,"GLM lazy initialization cleanup failed; retaining CUDA resources until process exit\n");
			SPARK_RETURN(status);
		}
		if ( state->lane_client != 0 )
		{
			(void)SparkWeightdClientClose(state->lane_client);
			state->lane_client = 0;
		}
		if ( SparkGlm5NextReleaseCaches(state) != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		SparkGlm5NextReleaseTaps(state);
		SparkGlm5NextReleaseSlotHost(state);
		SparkGlm5NextRouteTraceClose(state);
		SparkExpertWorkingSetDestroy(&state->expert_ws);
		SparkStageModuleLedgerRelease(&state->ledger);
		free(state->mtp_lane_armed);
		SparkGlm5NextReleaseDrafter(state);
		free(state);
		SPARK_RETURN(status);
	}
	SparkStageModuleAtomicStateArrayInitialize(state->slot_states,state->pipeline_slot_count);
	SparkStageModuleAtomicStateArrayInitialize(state->lane_states,state->resident_sequence_capacity);
	for (lane=0u; lane<state->resident_sequence_capacity; lane++)
	{
		atomic_init(&state->lane_bound[lane],0u);
		atomic_init(&state->lane_sequence_ids[lane],0u);
		atomic_init(&state->lane_next_positions[lane],0u);
	}
	atomic_init(&state->submitted_count,0u);
	atomic_init(&state->completed_count,0u);
	atomic_init(&state->rejected_count,0u);
	atomic_init(&state->failed_count,0u);
	atomic_init(&state->host_callback_completion_count,0u);
	atomic_init(&state->nccl_next_ordinal,0u);
	atomic_init(&state->nccl_next_ordinal_hc,0u);
	*state_out = state;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm5NextResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state)
{
	SparkGlm5NextModuleState *state;
	SparkStatus status;
	status = SparkFirmwareModuleValidateInitialization(configuration,host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state = 0;
	status = SparkGlm5NextInitializeState(configuration,host_services,&state);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	*module_state = state;
	return(SPARK_STATUS_OK);
}
