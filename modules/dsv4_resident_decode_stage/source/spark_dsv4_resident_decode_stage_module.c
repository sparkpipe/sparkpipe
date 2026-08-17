/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_dsv4_parallel_shape.h"
#include "sparkpipe/spark_head_screen.h"
#include "sparkpipe/spark_kv_page_store.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_row_layout.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "spark_dsv4_lane_continuity.h"
#include "spark_dsv4_hc_splitk.h"
#include "spark_dsv4_sparse_attention_split.h"
#include "spark_dsv4_paged_cache.h"
#include "spark_dsv4_pool_layout.h"
#include "spark_dsv4_stagepack_format.h"

#include "inference/kernels/graph.cuh"

/*
 * DeepSeek V4 resident decode stage host module, PP-Nx native, one variant
 * per build through the -include'd model header.
 *
 * One process is one STAGE over a layer slice; the pack must declare that
 * slice and the computed tensor inventory exactly. The stage boundary
 * carries the FOUR hyper-connection streams (hc_mult x hidden per row);
 * stage zero expands the embedding, the head stage's sigmoid reduction is
 * the only collapse. Execute serves DECODE frames: one token per lane
 * across up to max_active lanes, every attention kind, both router paths,
 * the full mHC chain. Prefill advances one round-major wave at a time, so
 * all live sequences at one prompt step share the same CUDA launch while
 * preserving each sequence's state dependency. A causal bulk-prefill kernel
 * can replace the wavefront after separate qualification.
 * GA DSpark execution remains refused until a native pass lands. Baseline
 * stage packs exclude all three checkpoint DSpark layers and expose no
 * speculative-token capability.
 *
 * The hash router pins to token ids, which exist only where the embedding
 * lives: a slice that starts inside the hash range without owning the
 * embedding is refused at configuration, not discovered at runtime.
 *
	 * KV is a physical page pool addressed through per-sequence page tables.
	 * Logical pages and prefix identity are owned by the generic cache layer;
	 * this module defines only DSV4's per-page payload and CUDA addressing.
 */

#define SPARK_DSV4_MODULE_TAG "dsv4_stage"
#define SPARK_DSV4_TP_COLLECTIVE_CREDITS_PER_SLOT 2u

_Static_assert(SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT *
	SPARK_DSV4_TP_COLLECTIVE_CREDITS_PER_SLOT <=
	SPARK_TP_DEVICE_COLLECTIVE_CREDIT_COUNT,
	"DSV4 TP continuation credits exceed the collective capacity");

typedef struct SparkDsv4LayerWeights
{
	const void *attn_norm_bf16;
	const void *ffn_norm_bf16;
	SparkDsv4AttnWeights attn;
	SparkDsv4CompressorWeights compressor;
	SparkDsv4IndexerWeights indexer;
	SparkDsv4MoeWeights moe;
	SparkDsv4HcWeights hc;
} SparkDsv4LayerWeights;

typedef struct SparkDsv4ModuleState SparkDsv4ModuleState;
typedef struct SparkDsv4ModuleSlot SparkDsv4ModuleSlot;
typedef struct SparkDsv4TpFrameContinuation SparkDsv4TpFrameContinuation;

static SparkStatus SparkDsv4ModuleEnqueueAsync(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t slot_index);
static SparkStatus SparkDsv4ModuleFinishFrameContinuation(
	SparkDsv4TpFrameContinuation *continuation);
static void SparkDsv4ModuleCompleteAsync(void *context);
static SparkStatus SparkDsv4ModuleSynchronizeFailedSlot(
	SparkDsv4ModuleSlot *slot);
static void SparkDsv4ModuleCompleteAfterFailedEnqueue(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t slot_index,
	SparkStatus status);
static SparkStatus SparkDsv4ModuleReplayTpIsland(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t island_index,
	uint32_t rows);
static SparkStatus SparkDsv4ModuleBounceBoundary(
	void *destination,
	const void *source,
	uint32_t rows,
	cudaStream_t stream,
	const char *site);

typedef struct SparkDsv4CompressorScratch
{
	void *kv_bf16;
	void *score_bf16;
	void *emit_bf16;
	uint32_t *emitted_u32;
} SparkDsv4CompressorScratch;

struct SparkDsv4ModuleSlot
{
	void *cuda_stream;
	SparkStageModuleCudaFork compute_fork;
	SparkStageModuleCudaReadAhead weight_read_ahead;
	cudaEvent_t tp_host_copy_event;
	void *host_staging;
	uint32_t *host_input_token_ids;
	uint32_t *host_row_lane_indices;
	uint32_t *host_row_page_table_indices;
	uint64_t *host_row_positions;
	uint64_t *host_row_emit_positions;
	uint64_t *host_row_emit_positions_hca;
	uint32_t *host_output_token_ids;
	uint32_t *host_logical_page_table;
	uint32_t *host_physical_page_table;
	uint32_t *host_page_table_update_indices;
	uint32_t *host_page_table_update_values;
	uint32_t *host_initialize_page_indices;
	uint32_t *host_initialize_parent_page_indices;
	uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	uint32_t *resident_token_ids;
	uint32_t *prefill_emit_rows_u32;
	uint32_t *row_lane_indices;
	uint32_t *row_page_table_indices;
	uint64_t *row_positions;
	uint64_t *row_emit_positions;
	uint64_t *row_emit_positions_hca;
	uint32_t *physical_page_table;
	uint32_t *page_table_update_indices;
	uint32_t *page_table_update_values;
	uint32_t *initialize_page_indices;
	uint32_t *initialize_parent_page_indices;
	uint32_t *slot_counts;
	uint32_t *attention_slot_counts;
	int32_t *topk_idxs;
	void *streams_bf16;
	void *residual_bf16;
	void *reduced_bf16;
	void *normalized_bf16;
	void *qr_bf16;
	void *q_bf16;
	void *kv_bf16;
	void *attn_out_bf16;
	float *sparse_attn_partials_f32;
	void *o_ranks_bf16;
	void *delta_bf16;
	SparkDsv4CompressorScratch compressor;
	SparkDsv4CompressorScratch index_compressor;
	void *index_q_bf16;
	void *index_weights_bf16;
	float *index_weights_f32;
	float *index_scores_f32;
	float *mixes_f32;
	float *hc_partials_f32;
	float *pre_f32;
	float *post_f32;
	float *comb_f32;
	float *moe_scores_f32;
	uint32_t *moe_indices_u32;
	float *moe_weights_f32;
	void *ffn_gate_bf16;
	void *ffn_up_bf16;
	void *ffn_delta_bf16;
	void *ffn_accum_bf16;
	uint32_t *grouped_rows_u32;
	uint32_t *group_tile_prefix_w1_u32;
	uint32_t *group_tile_prefix_w2_u32;
	void *moe_slot_gate_bf16;
	void *moe_slot_up_bf16;
	void *moe_slot_out_bf16;
	void *head_logits_bf16;
	void *head_certified_scratch;
	uint32_t *head_candidate_ids_u32;
	uint32_t *head_candidate_counts_u32;
	float *head_scores_f32;
	uint64_t *head_maxloc_u64;
	uint32_t *expert_offsets_u32;
	uint32_t *moe_inverse_u32;
	uint32_t page_table_update_count;
	SparkDsv4TpFrameContinuation *tp_continuation;
	/* DSpark draft workspace. The draft runs replicated at full width on
	 * every rank (identical math, no draft collectives); the verify rides
	 * the standard TP island chain at bucket width. */
	uint32_t dspark_armed;
	uint32_t dspark_verify_rows;
	uint32_t dspark_verify_accept;
	uint32_t dspark_host_draft_tokens[SPARK_DSV4_MODEL_DSPARK_SPEC_STEP];
	uint32_t *dspark_draft_token_ids;
	uint64_t *dspark_maxloc_u64;
	uint32_t *dspark_verify_token_ids;
	uint32_t *dspark_input_token_ids;
	uint64_t *dspark_row_positions;
	uint32_t *dspark_row_lane_indices;
	float *dspark_scores_f32;
	uint32_t dspark_host_input_token_ids[SPARK_DSV4_MODEL_DSPARK_SPEC_STEP];
	uint64_t dspark_host_row_positions[SPARK_DSV4_MODEL_DSPARK_SPEC_STEP];
	void *dspark_x_bf16;
	void *dspark_q_attn_bf16;
	void *dspark_o_ranks_bf16;
	void *dspark_main_cat_bf16;
	void *dspark_main_x_bf16;
	void *dspark_tap_bf16;
	void *dspark_tap_ring_bf16;
	void *dspark_verify_tap_bf16;
	/* CSA/HCA compressed-state rollback: the verify frame's rows write the
	 * position-keyed ring (self-cleaning), but a REJECTED boundary row's
	 * overlap-shift moves speculative content into the CSA previous windows
	 * and its emission lands in the compressed cache. Save the CSA previous
	 * windows + every compressor's boundary emission slots before the
	 * frame; restore them after when the boundary row is rejected. */
	uint8_t *dspark_csa_previous_save;
	uint8_t *dspark_emission_save;
	uint32_t dspark_boundary_rows[2u];
	uint32_t dspark_boundary_count;
	uint32_t dspark_hca_boundary_row;
	void *dspark_logits_bf16;
	float *dspark_logits_f32;
};

typedef struct SparkDsv4AsyncCompletion
{
	SparkDsv4ModuleState *state;
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t slot_index;
	uint32_t lane_count;
	uint32_t row_count;
	uint32_t emitted_token_count;
	uint32_t cache_lane_count;
	uint32_t requires_prepared_cache_admission;
	uint32_t prepared_cache_lane_count;
	uint32_t aborted_cache_lane_count;
	uint32_t prepared_cache_admission_index;
	uint32_t tokens_per_sequence;
	uint32_t lane_indices[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkModelDriverCacheLane cache_lanes[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkModelDriverCacheLane admission_cache_lanes[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkModelDriverAdmissionRequest cache_admission;
	SparkDsv4PagedCacheLane prepared_cache_lanes[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t *output_token_destination;
	SparkModelDriverCompletion completion;
} SparkDsv4AsyncCompletion;

#define SPARK_DSV4_PREPARED_CACHE_FREE 0u
#define SPARK_DSV4_PREPARED_CACHE_PREFETCHING 1u
#define SPARK_DSV4_PREPARED_CACHE_READY 2u
#define SPARK_DSV4_PREPARED_CACHE_COMMITTED 3u
#define SPARK_DSV4_PREPARED_CACHE_ADOPTING 4u

typedef struct SparkDsv4PreparedCacheAdmission
{
	uint32_t state;
	uint32_t mutable_page_demand;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverCacheLane cache_lanes[
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
} SparkDsv4PreparedCacheAdmission;

typedef struct SparkDsv4TpGraphIsland
{
	cudaGraphExec_t executable;
	uint32_t layer_index;
	uint32_t kind;
	uint32_t live;
} SparkDsv4TpGraphIsland;

#define SPARK_DSV4_TP_GRAPH_ISLAND_PROJECTION 0u
#define SPARK_DSV4_TP_GRAPH_ISLAND_ATTENTION 1u
#define SPARK_DSV4_TP_GRAPH_ISLAND_FFN 2u
#define SPARK_DSV4_TP_GRAPH_ISLAND_FINAL 3u

static uint32_t SparkDsv4ModuleCacheAdmissionRequestMatches(
	const SparkDsv4PreparedCacheAdmission *prepared,
	const SparkModelDriverAdmissionRequest *request);
static SparkDsv4PreparedCacheAdmission *SparkDsv4ModuleFindCacheAdmission(
	SparkDsv4ModuleState *state,
	const SparkModelDriverAdmissionRequest *request,
	SparkDsv4PreparedCacheAdmission **free_record_out);
static void SparkDsv4ModuleClearCacheAdmission(
	SparkDsv4ModuleState *state,
	SparkDsv4PreparedCacheAdmission *prepared,
	uint32_t release_ownership);

struct SparkDsv4ModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t participates_final_head;
	uint32_t pp_stage_count;
	uint32_t pp_stage_index;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t vocabulary_row_start;
	uint32_t vocabulary_rows_per_rank;
	uint64_t tp_configuration_hash;
	SparkTpDeviceCollective tp_device_collective;
	uint32_t tp_device_collective_initialized;
	SparkTpDeviceCollectiveCreditBinding tp_credit_bindings[
		SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT];
	uint32_t tp_credit_binding_count;
	void *tp_credit_send_bf16;
	void *tp_credit_receive_bf16;
	void *tp_host_credit_send_bf16;
	void *tp_host_credit_receive_bf16;
	atomic_ullong tp_next_ordinal;
	SparkDsv4TpFrameContinuation *tp_continuations;
	uint32_t resident_sequence_capacity;
	uint32_t pipeline_slot_count;
	uint32_t max_sequence_positions;
	uint32_t logical_page_capacity;
	uint32_t physical_page_capacity;
	uint32_t mtp_armed;
	uint32_t compress_layer_count;
	uint32_t csa_layer_count;
	uint32_t topk_column_count;
	uint32_t index_slot_capacity;
	uint32_t multiprocessor_count;
	uint32_t dspark_enabled;
	/* Sparse-attention partials: one slot per (row, head-group, split)
	 * block; sized for max(multiprocessor_count, bucket_rows * head_groups)
	 * so multi-wave grids stay within the allocation. */
	uint32_t sparse_attn_partial_capacity;
	void *execution_stream;
	char kv_backing_directory[SPARK_KV_PAGE_STORE_PATH_BYTES];
	uint64_t kv_backing_maximum_bytes;
	SparkKvPageStore kv_page_store;
	void *kv_page_store_staging;
	void *kv_page_store_stream;
	uint32_t kv_page_store_enabled;
	SparkDsv4PagedCache paged_cache;
	SparkDsv4PagedScoreSpan page_score_spans[
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT * 2u];
	SparkDsv4PagedScoreSpan *device_page_score_spans;
	uint32_t page_score_span_count;
	uint32_t *cache_admission_logical_pages;
	SparkDsv4PreparedCacheAdmission prepared_cache_admissions[
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	pthread_mutex_t cache_mutex;
	uint32_t cache_mutex_initialized;
	float hc_head_scale_value;
	uint32_t csa_ordinal_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t layer_seen_bits[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t mtp_seen_bits[SPARK_DSV4_STAGEPACK_MTP_LAYER_COUNT_MAX];
	uint64_t global_seen_bits;
	SparkDsv4LayerWeights layers[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	/* DSpark draft: three full transformer layers (mtp.0..2) carrying the
	 * standard per-layer tensor set, plus the stage extras below. */
	SparkDsv4LayerWeights mtp_layers[SPARK_DSV4_STAGEPACK_MTP_LAYER_COUNT_MAX];
	SparkDsv4MtpWeights mtp;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	/* DSpark state: the replicated draft ring (one sliding window per draft
	 * layer per resident lane) and the per-rank lm_head view + mtp head
	 * scale for the draft logits. */
	void *dspark_ring_bf16;
	uint64_t dspark_ring_lane_stride;
	uint64_t dspark_ring_layer_stride;
	/* Lane-level draft state: the anchor step's taps + token/position, so any
	 * pipeline slot can run the next draft (slots cycle across submissions). */
	void *dspark_tap_store_bf16;
	uint8_t dspark_lane_ready[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t dspark_lane_anchor[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t dspark_lane_position[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkDsv4LinearView lm_head_view;
	uint8_t *head_shadow_payload;
	uint8_t *head_shadow_scale;
	float *head_error_norm_f32;
	uint8_t *head_certified_fp8_payload;
	float *head_certified_fp8_scale_f32;
	float *head_certified_fp8_norm_f32;
	const float *hc_head_fn_f32;
	const float *hc_head_base_f32;
	const float *hc_head_scale_f32;
	float *base_freqs_f32;
	float *compress_freqs_f32;
	void *kv_cache_bf16;
	uint64_t cache_offset_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t cache_lane_stride_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	void *index_cache_bf16;
	uint64_t index_lane_stride;
	float *compress_kv_state_f32;
	float *compress_score_state_f32;
	uint64_t compress_state_offset_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t compress_score_state_offset_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t compress_state_lane_stride_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	float *index_kv_state_f32;
	float *index_score_state_f32;
	uint64_t index_state_lane_stride;
	uint64_t index_cache_offset_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t index_kv_state_offset_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t index_score_state_offset_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t resident_state_bytes;
	/* TP1's optional dynamic whole-frame shape cache. */
	uint32_t graph_capacity;
	uint32_t graph_sealed;
	LmGraphCache graph_cache;
	LmGraphEntry graph_entries[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_GRAPH_COUNT];
	/*
	 * TP production uses fixed-width, stage-local graph islands. The flat
	 * allocation is [pipeline slot][3 * local layers + 1]. Every entry is
	 * captured and instantiated before readiness, then sealed as one unit.
	 */
	SparkDsv4TpGraphIsland *tp_graph_islands;
	uint32_t tp_graph_islands_per_slot;
	uint32_t tp_graph_island_count;
	uint32_t tp_graphs_sealed;
	SparkDsv4ModuleSlot slots[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkDsv4AsyncCompletion completions[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint slot_states[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint lane_states[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_RESIDENT_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_RESIDENT_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_RESIDENT_SEQUENCE_COUNT];
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong tokens_emitted;
	atomic_ullong host_callback_completion_count;
};

static uint32_t SparkDsv4ModuleTpQueryHeads(const SparkDsv4ModuleState *state)
{
	return(SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT / state->tp_degree);
}

static uint32_t SparkDsv4ModuleTpQueryDimension(const SparkDsv4ModuleState *state)
{
	return(SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION / state->tp_degree);
}

static uint32_t SparkDsv4ModuleTpExpertWidth(const SparkDsv4ModuleState *state)
{
	return(SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION / state->tp_degree);
}

static uint32_t SparkDsv4ModuleTpOutputGroupCount(
	const SparkDsv4ModuleState *state)
{
	return(state->tp_degree < SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT ?
		SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT / state->tp_degree : 1u);
}

static uint32_t SparkDsv4ModuleTpOutputGroupInput(
	const SparkDsv4ModuleState *state)
{
	uint32_t ranks_per_group;
	ranks_per_group = state->tp_degree > SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT ?
		state->tp_degree / SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT : 1u;
	return(SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION / ranks_per_group);
}

static uint32_t SparkDsv4ModuleTpOutputLora(const SparkDsv4ModuleState *state)
{
	return(SparkDsv4ModuleTpOutputGroupCount(state) *
		SPARK_DSV4_MODEL_OUTPUT_LORA_RANK);
}

extern cudaError_t SparkDsv4ConfigureCudaKernels(uint32_t *multiprocessor_count);
extern cudaError_t SparkDsv4LaunchWeightReadAhead(cudaStream_t stream, const void *payload, uint64_t bytes, const void *auxiliary_payload, uint64_t auxiliary_bytes, uint32_t *sink_u32, uint32_t block_capacity);
extern cudaError_t SparkDsv4LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkDsv4LaunchLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchExpertLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchFp8LinearPair(cudaStream_t stream, const SparkDsv4LinearView *first, const SparkDsv4LinearView *second, const void *input_bf16, void *first_output_bf16, void *second_output_bf16, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchBf16LinearPair(cudaStream_t stream, const SparkDsv4LinearView *first, const SparkDsv4LinearView *second, const void *input_bf16, void *first_output_bf16, void *second_output_bf16, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchPackProjectionShards(cudaStream_t stream, const void *wq_bf16, const void *wkv_bf16, const void *compress_kv_bf16, const void *compress_score_bf16, const void *index_kv_bf16, const void *index_score_bf16, void *packed_bf16, uint32_t row_count, uint32_t tp_rank, uint32_t tp_degree, uint32_t compress_channels, uint32_t index_channels);
extern cudaError_t SparkDsv4LaunchUnpackProjectionShards(cudaStream_t stream, const void *packed_bf16, void *wq_bf16, void *wkv_bf16, void *compress_kv_bf16, void *compress_score_bf16, void *index_kv_bf16, void *index_score_bf16, uint32_t row_count, uint32_t compress_channels, uint32_t index_channels);
extern cudaError_t SparkDsv4LaunchStridedLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *payload, const uint8_t *scale, uint64_t weight_payload_group_stride_bytes, uint64_t weight_scale_group_stride_bytes, const void *input_bf16, uint64_t input_row_stride, uint32_t input_offset, uint32_t input_group_stride, void *output_bf16, uint64_t output_row_stride, uint32_t output_offset, uint32_t output_group_stride, uint32_t group_count, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadCertifiedFp8Quantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, float *shadow_scale_f32, float *cert_norm_f32, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadScreenedArgmaxSharded(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, float *output_scores, uint32_t candidate_offset, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadCertifiedFp8B1Sharded(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const float *shadow_scale_f32, const float *cert_norm_f32, void *scratch, uint32_t *candidate_ids, uint32_t *screened_count, uint32_t *output_token_id, float *output_score, uint32_t candidate_offset, uint32_t row_count, uint32_t vocabulary_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadMaxlocPack(cudaStream_t stream, const float *scores, const uint32_t *token_ids, uint64_t *maxloc, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchHeadMaxlocUnpack(cudaStream_t stream, const uint64_t *maxloc, uint32_t *token_ids, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchResidentTokenFeedback(cudaStream_t stream, const uint32_t *output_token_ids, uint32_t *resident_token_ids, uint32_t *input_token_ids, uint64_t *row_positions, uint64_t *row_emit_positions, uint64_t *row_emit_positions_hca, uint32_t row_count, uint32_t tokens_per_sequence, uint32_t step_index, uint32_t advance);
extern cudaError_t SparkDsv4LaunchAccumU64Max(cudaStream_t stream, uint64_t *destination, const uint64_t *source, uint32_t element_count);
extern cudaError_t SparkDsv4LaunchGatherHeadRows(cudaStream_t stream, const void *source_bf16, const uint32_t *source_row_indices, void *destination_bf16, uint32_t row_count, uint32_t row_width);
extern cudaError_t SparkDsv4LaunchScatterHeadTokens(cudaStream_t stream, const uint32_t *source, const uint32_t *destination_lane_indices, uint32_t *destination, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchQuantSim(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t row_stride, uint32_t width, uint32_t block, uint32_t fp4);
extern cudaError_t SparkDsv4LaunchRope(cudaStream_t stream, void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse);
extern cudaError_t SparkDsv4LaunchQueryHeadRms(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim, float epsilon);
extern cudaError_t SparkDsv4LaunchQueryHeadRmsRope(cudaStream_t stream, void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, float epsilon);
extern cudaError_t SparkDsv4LaunchKvPost(cudaStream_t stream, void *data_bf16, const void *gain_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_dim, uint32_t rope_dim, uint32_t quant_width, uint32_t quant_block, float epsilon);
extern cudaError_t SparkDsv4LaunchIndexerPost(cudaStream_t stream, void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t quant_block);
extern cudaError_t SparkDsv4LaunchHadamard(cudaStream_t stream, void *data_bf16, uint32_t vector_count, uint32_t width);
extern cudaError_t SparkDsv4LaunchSparseAttn(cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_lane_indices, const uint32_t *row_page_table_indices, const uint32_t *physical_page_table, uint32_t page_table_stride, uint32_t compressed_entries_per_page, const int32_t *topk_idxs, const uint32_t *valid_topk_counts, uint32_t topk, const float *sink_f32, float scale, void *out_bf16, float *partials_f32, uint32_t partial_capacity, uint32_t multiprocessor_count, uint32_t row_count, uint32_t head_count, uint32_t head_dim);
extern cudaError_t SparkDsv4LaunchWiden(cudaStream_t stream, const void *input_bf16, float *output_f32, uint32_t row_count, uint32_t width, float scale);
extern cudaError_t SparkDsv4LaunchCompressStep(cudaStream_t stream, const void *kv_bf16, const void *score_bf16, const float *ape_f32, float *kv_state_f32, float *score_state_f32, uint64_t state_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t row_count, uint32_t ratio, uint32_t overlapped, uint32_t width, void *emit_bf16, uint32_t *emitted);
extern cudaError_t SparkDsv4LaunchKvEmission(cudaStream_t stream, void *emit_bf16, const uint32_t *emitted, const void *norm_weight_bf16, const float *freqs_f32, const uint64_t *row_emit_positions, void *cache_bf16, uint64_t cache_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t row_count, uint32_t width, uint64_t base_slot, uint32_t ratio, uint32_t ring_slots, uint32_t rotate);
extern cudaError_t SparkDsv4LaunchCacheScatter(cudaStream_t stream, const void *source_bf16, const uint32_t *emitted, void *cache_bf16, uint64_t cache_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t row_count, uint32_t width, uint64_t base_slot, uint32_t ratio, uint32_t ring_slots);
extern cudaError_t SparkDsv4LaunchDsparkAttention(cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, uint32_t lane_index, const void *block_kv_bf16, const float *sink_f32, float scale, void *out_bf16, uint32_t block_size, uint32_t head_count, uint32_t head_dim, uint32_t window_tokens);
extern cudaError_t SparkDsv4LaunchDsparkMarkovBiasAccum(cudaStream_t stream, const void *logits_bf16, const void *markov_w2_bf16, const void *markov_embed_bf16, float *logits_f32, uint32_t vocab_offset, uint32_t shard_count, uint32_t rank, uint32_t position, uint32_t multiprocessor_count);
extern cudaError_t SparkDsv4LaunchDsparkArgmax(cudaStream_t stream, const float *logits_f32, uint32_t shard_count, uint32_t vocab_offset, uint32_t *output_token_id, float *output_score);
extern cudaError_t SparkDsv4LaunchDsparkTapMean(cudaStream_t stream, const void *streams_bf16, void *tap_bf16, uint32_t row_count, uint32_t stream_count, uint32_t dimension, uint32_t multiprocessor_count);
extern cudaError_t SparkDsv4LaunchExpandStreams(cudaStream_t stream, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t stream_count, uint32_t dimension, uint32_t multiprocessor_count);
extern cudaError_t SparkDsv4LaunchInitializePages(cudaStream_t stream, void *page_pool, uint64_t page_stride_bytes, const uint32_t *page_indices, const uint32_t *parent_page_indices, uint32_t page_count, const SparkDsv4PagedScoreSpan *score_spans, uint32_t score_span_count);
extern cudaError_t SparkDsv4LaunchUpdatePageTable(cudaStream_t stream, uint32_t *page_table, const uint32_t *update_indices, const uint32_t *update_values, uint32_t update_count);
extern cudaError_t SparkDsv4LaunchBuildAttentionIndices(cudaStream_t stream, const uint64_t *row_positions, int32_t *indices, uint32_t *slot_counts, uint32_t *attention_slot_counts, uint32_t row_count, uint32_t column_count, uint32_t index_slot_capacity, uint32_t layer_kind);
extern cudaError_t SparkDsv4LaunchGateRoute(cudaStream_t stream, const SparkDsv4LinearView *gate, const void *input_bf16, float *scores_f32, const float *bias_f32, const uint32_t *tid2eid_u32, const uint32_t *token_ids, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32, uint32_t expert_width, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2);
extern cudaError_t SparkDsv4LaunchSwigluClamp(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width, float limit, const float *row_weights_f32, const uint32_t *weight_map);
extern cudaError_t SparkDsv4LaunchValidateTid2Eid(cudaStream_t stream, const uint32_t *tid2eid, uint64_t entry_count, uint32_t *violation_flag);
extern cudaError_t SparkDsv4LaunchFusedExpertW13Act(cudaStream_t stream, const SparkDsv4LinearView *w1, const SparkDsv4LinearView *w3, const void *input_bf16, const uint32_t *route_source_token, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *activated_bf16, uint32_t rows, uint32_t expert_width, float limit, uint32_t multiprocessor_count);
extern cudaError_t SparkDsv4LaunchFusedSharedW13Act(cudaStream_t stream, const SparkDsv4LinearView *w1, const SparkDsv4LinearView *w3, const void *input_bf16, void *activated_bf16, uint32_t rows, uint32_t expert_width, float limit);
extern cudaError_t SparkDsv4LaunchExpertDown(cudaStream_t stream, const SparkDsv4LinearView *stacked, const void *input_bf16, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t expert_width, uint32_t hidden_dimension, uint32_t multiprocessor_count);
extern cudaError_t SparkDsv4LaunchMoePairReduce(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint32_t row_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchMoePairReduceStrided(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint64_t accum_row_stride, uint32_t accum_offset, uint32_t row_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchAccumAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width);
extern cudaError_t SparkDsv4LaunchAccumAddRelay(cudaStream_t stream,
	void *destination_bf16,const void *source_bf16,void *relay_bf16,
	uint32_t row_count,uint32_t width);
extern cudaError_t SparkDsv4LaunchAccumAddTp4Tree(cudaStream_t stream,
	void *destination_bf16,const void *const rank_devices[4],uint32_t tp_rank,
	uint32_t row_count,uint32_t width);
extern cudaError_t SparkDsv4LaunchIndexerScore(cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_page_table_indices, const uint32_t *physical_page_table, uint32_t page_table_stride, uint32_t entries_per_page, const uint32_t *slot_counts, const float *head_weights_f32, float *scores_f32, uint32_t row_count, uint32_t max_slots, uint32_t head_count, uint32_t head_dim);
extern cudaError_t SparkDsv4LaunchTopK(cudaStream_t stream, float *scores_f32, const uint32_t *slot_counts, uint32_t max_slots, uint32_t topk, int32_t offset, int32_t *indices_out, uint64_t out_row_stride, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchHcMix(cudaStream_t stream, const void *streams_bf16, const float *fn_f32, float *mixes_f32, uint32_t row_count, uint32_t flat_dimension, uint32_t mix_rows, float epsilon);
extern cudaError_t SparkDsv4LaunchHcSplitSinkhorn(cudaStream_t stream, const float *mixes_f32, const float *scale3_f32, const float *base_f32, uint32_t row_count, uint32_t hc, uint32_t iterations, float epsilon, float *pre_f32, float *post_f32, float *comb_f32);
extern cudaError_t SparkDsv4LaunchHcMixSplitKSinkhorn(cudaStream_t stream, const void *streams_bf16, const float *fn_f32, const float *scale3_f32, const float *base_f32, float *partials_f32, float *mixes_f32, uint32_t row_count, uint32_t flat_dimension, uint32_t mix_rows, uint32_t hc, uint32_t iterations, float rms_epsilon, float hc_epsilon, float *pre_f32, float *post_f32, float *comb_f32);
extern cudaError_t SparkDsv4LaunchHcPreReduce(
	cudaStream_t stream,const void *streams_bf16,const float *pre_f32,
	void *reduced_bf16,void *residual_bf16,uint32_t row_count,uint32_t hc,
	uint32_t dimension);
extern cudaError_t SparkDsv4LaunchHcPost(cudaStream_t stream, const void *out_bf16, const void *residual_bf16, const float *post_f32, const float *comb_f32, void *streams_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension);
extern cudaError_t SparkDsv4LaunchHcHeadReduce(cudaStream_t stream, const void *streams_bf16, const float *mixes_f32, float scale, const float *base_f32, float epsilon, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension);

static SparkStatus SparkDsv4ModuleCombineBf16(
	void *combine_context,
	void *destination_device,
	const void *source_device,
	uint32_t active_sequence_count,
	uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkDsv4LaunchAccumAdd((cudaStream_t)cuda_stream,
		destination_device,source_device,active_sequence_count,hidden_dimension);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"tp_all_reduce_sum"));
}

static SparkStatus SparkDsv4ModuleCombineRelayBf16(
	void *combine_context,
	void *destination_device,
	const void *source_device,
	void *relay_device,
	uint32_t active_sequence_count,
	uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkDsv4LaunchAccumAddRelay((cudaStream_t)cuda_stream,
		destination_device,source_device,relay_device,active_sequence_count,
		hidden_dimension);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"tp_all_reduce_sum_relay"));
}

static SparkStatus SparkDsv4ModuleCombineTp4Bf16(
	void *combine_context,
	void *destination_device,
	const void *const rank_devices[4],
	uint32_t tp_rank,
	uint32_t active_sequence_count,
	uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkDsv4LaunchAccumAddTp4Tree((cudaStream_t)cuda_stream,
		destination_device,rank_devices,tp_rank,active_sequence_count,
		hidden_dimension);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"tp_all_reduce_sum_tp4_tree"));
}

static SparkStatus SparkDsv4ModuleCombineU64Max(
	void *combine_context,
	uint64_t *destination_device,
	const uint64_t *source_device,
	uint32_t element_count,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkDsv4LaunchAccumU64Max((cudaStream_t)cuda_stream,
		destination_device,source_device,element_count);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"tp_all_reduce_max_u64"));
}

static SparkStatus SparkDsv4ModuleInitializeTpCollective(
	SparkDsv4ModuleState *state,
	const SparkDsv4ResidentDecodeStageNodeContext *context)
{
	SparkTpDeviceCollectiveConfig configuration;
	uint64_t credit_bytes,offset,total_bytes;
	uint32_t credit,hidden,memory_mode,route,route_count;
	void *mapped_receive,*mapped_send;
	cudaError_t error;
	SparkStatus status;
	if ( state == 0 || context == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->tp_degree == 1u )
		return(SPARK_STATUS_OK);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = context->tp_collective_backend_kind;
	configuration.tp_degree = state->tp_degree;
	configuration.tp_rank = state->tp_rank;
	configuration.operation_kind =
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	/* Attention completion submits the FFN reduction before its callback
	 * releases the current transport credit. Keep the two reductions for each
	 * live slot on different credits; later layers reuse them after release. */
	configuration.credit_count = state->pipeline_slot_count *
		SPARK_DSV4_TP_COLLECTIVE_CREDITS_PER_SLOT;
	configuration.local_hidden_dimension = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	configuration.max_active_sequence_count =
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT;
	configuration.connect_timeout_milli = context->tp_connect_timeout_milli;
	configuration.operation_timeout_milli = context->tp_operation_timeout_milli;
	configuration.control_port_base = context->tp_collective_control_port_base;
	configuration.collective_identifier = context->tp_collective_identifier;
	configuration.backend_module_path =
		context->tp_collective_backend_module_path;
	configuration.registration_cuda_stream = state->execution_stream;
	status = SparkTpDeviceCollectiveApplyTopology(
		&context->tp_collective_topology,&configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration.backend_kind ==
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		configuration.combine_bf16_function = SparkDsv4ModuleCombineBf16;
		configuration.combine_relay_bf16_function =
			SparkDsv4ModuleCombineRelayBf16;
		configuration.combine_tp4_bf16_function =
			SparkDsv4ModuleCombineTp4Bf16;
		configuration.combine_u64_max_function = SparkDsv4ModuleCombineU64Max;
		configuration.combine_context = state;
	}
	if ( configuration.connect_timeout_milli == 0u ||
		configuration.operation_timeout_milli == 0u ||
		configuration.control_port_base == 0u ||
		configuration.collective_identifier == 0u ||
		configuration.backend_module_path == 0 ||
		configuration.local_host == 0 ||
		configuration.backend_module_path[0] == '\0' ||
		configuration.local_host[0] == '\0' )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration.backend_kind !=
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT &&
		configuration.backend_kind != SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkTpDeviceCollectiveProbeMemoryMode(
		configuration.backend_kind,configuration.backend_module_path,
		&memory_mode);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkTpDeviceCollectiveCreditBindingRouteCount(
		&configuration,&route_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	total_bytes = 0u;
	for (route=0u; route<route_count; route++)
	{
		hidden = configuration.local_hidden_dimension;
		credit_bytes = (uint64_t)configuration.max_active_sequence_count *
			hidden * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
		if ( credit_bytes == 0u || total_bytes > UINT64_MAX -
			credit_bytes * configuration.credit_count )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		total_bytes += credit_bytes *
			configuration.credit_count;
	}
	status = SPARK_STATUS_OK;
	if ( total_bytes != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,total_bytes,
			&state->tp_credit_send_bf16);
	if ( status == SPARK_STATUS_OK && total_bytes != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,total_bytes,
			&state->tp_credit_receive_bf16);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( total_bytes != 0u && memory_mode ==
		SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST )
	{
		mapped_receive = 0;
		mapped_send = 0;
		error = cudaHostAlloc(&state->tp_host_credit_send_bf16,total_bytes,
			cudaHostAllocPortable | cudaHostAllocMapped);
		if ( error == cudaSuccess )
			error = cudaHostAlloc(&state->tp_host_credit_receive_bf16,total_bytes,
				cudaHostAllocPortable | cudaHostAllocMapped);
		if ( error == cudaSuccess )
			error = cudaHostGetDevicePointer(&mapped_send,
				state->tp_host_credit_send_bf16,0u);
		if ( error == cudaSuccess )
			error = cudaHostGetDevicePointer(&mapped_receive,
				state->tp_host_credit_receive_bf16,0u);
		if ( error != cudaSuccess )
		{
			if ( state->tp_host_credit_send_bf16 != 0 )
				(void)cudaFreeHost(state->tp_host_credit_send_bf16);
			if ( state->tp_host_credit_receive_bf16 != 0 )
				(void)cudaFreeHost(state->tp_host_credit_receive_bf16);
			state->tp_host_credit_send_bf16 = 0;
			state->tp_host_credit_receive_bf16 = 0;
			return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
				"tp_credit_alloc"));
		}
		state->tp_credit_send_bf16 = mapped_send;
		state->tp_credit_receive_bf16 = mapped_receive;
	}
	offset = 0u;
	state->tp_credit_binding_count = 0u;
	for (route=0u; route<route_count; route++)
	{
		hidden = configuration.local_hidden_dimension;
		credit_bytes = (uint64_t)configuration.max_active_sequence_count *
			hidden * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
		for (credit=0u; credit<configuration.credit_count;
			credit++)
		{
			SparkTpDeviceCollectiveCreditBinding *binding =
				&state->tp_credit_bindings[state->tp_credit_binding_count++];
			binding->step_index = route;
			binding->credit_index = credit;
			binding->send_device =
				(uint8_t *)state->tp_credit_send_bf16 + offset;
			binding->receive_device =
				(uint8_t *)state->tp_credit_receive_bf16 + offset;
			binding->send_transport = memory_mode ==
				SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ?
				(uint8_t *)state->tp_host_credit_send_bf16 + offset :
				binding->send_device;
			binding->receive_transport = memory_mode ==
				SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ?
				(uint8_t *)state->tp_host_credit_receive_bf16 + offset :
				binding->receive_device;
			binding->flags = memory_mode ==
				SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ?
				SPARK_TP_DEVICE_COLLECTIVE_BINDING_KNOWN_FLAGS : 0u;
			binding->reserved0 = 0u;
			offset += credit_bytes;
		}
	}
	if ( state->tp_credit_binding_count != 0u )
	{
		configuration.credit_bindings = state->tp_credit_bindings;
		configuration.credit_binding_count = state->tp_credit_binding_count;
	}
	status = SparkTpDeviceCollectiveCreate(
		&configuration,&state->tp_device_collective);
	if ( status != SPARK_STATUS_OK )
	{
		if ( state->tp_host_credit_send_bf16 != 0 )
			(void)cudaFreeHost(state->tp_host_credit_send_bf16);
		if ( state->tp_host_credit_receive_bf16 != 0 )
			(void)cudaFreeHost(state->tp_host_credit_receive_bf16);
		state->tp_host_credit_send_bf16 = 0;
		state->tp_host_credit_receive_bf16 = 0;
		return(status);
	}
	state->tp_device_collective_initialized = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleConfigure(
	SparkDsv4ModuleState *state,
	const SparkFirmwareModuleHostServices *host_services,
	const char **pack_path_out)
{
	const SparkDsv4ResidentDecodeStageNodeContext *context;
	SparkDsv4TpShapeDescriptor shape;
	SparkDsv4TpNodeConfig tp_config;
	uint64_t directory_bytes;
	uint32_t hybrid,lane_page_capacity,parallel;
	SparkStatus status;
	if ( state == 0 || host_services == 0 || pack_path_out == 0 ||
		host_services->node_context == 0 || host_services->execution_stream == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkDsv4ResidentDecodeStageNodeContext *)host_services->node_context;
	if ( context->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION || context->descriptor_bytes != SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( (context->flags & ~SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KNOWN_FLAGS) != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	parallel = (context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_TENSOR_PARALLEL) != 0u ? 1u : 0u;
	hybrid = (context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_PIPELINE_PARALLEL) != 0u ? 1u : 0u;
	if ( context->stage_count == 0u || context->stage_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT || context->stage_index >= context->stage_count || context->first_layer_index >= SPARK_DSV4_MODEL_LAYER_COUNT || context->layer_count == 0u || context->layer_count > SPARK_DSV4_MODEL_LAYER_COUNT - context->first_layer_index || context->resident_sequence_capacity == 0u || context->resident_sequence_capacity > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_RESIDENT_SEQUENCE_COUNT || context->pipeline_slot_count == 0u || context->pipeline_slot_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT || context->cuda_graph_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_GRAPH_ISLAND_COUNT || context->max_sequence_positions < SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO || context->max_sequence_positions > SPARK_DSV4_MODEL_MAX_POSITIONS || context->linear_weight_codec != SPARK_DSV4_MODEL_NON_EXPERT_WEIGHT_CODEC || context->expert_weight_codec != SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC || context->kv_cache_codec != SPARK_DSV4_MODEL_KV_CACHE_CODEC || context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' || (parallel == 0u && (context->tp_degree != 1u || context->tp_rank != 0u || context->tp_configuration_hash != 0u || hybrid != 0u)) || (parallel != 0u && (context->tp_degree <= 1u || (hybrid == 0u && (context->stage_count != context->tp_degree || context->stage_index != context->tp_rank)) || (hybrid != 0u && (context->pp_stage_count != context->stage_count || context->pp_stage_index != context->stage_index || context->world_size != context->tp_degree * context->pp_stage_count || context->world_rank != context->pp_stage_index * context->tp_degree + context->tp_rank)))) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( parallel == 0u &&
		context->cuda_graph_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_GRAPH_COUNT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( parallel != 0u &&
		(SparkDsv4ResidentDecodeStageNativeTpWidthSupported(SPARK_BATCH_BUCKET) == 0u ||
		 context->cuda_graph_count !=
		 SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(context->layer_count)) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( parallel != 0u && (
		context->tp_collective_backend_module_path == 0 ||
		context->tp_collective_backend_module_path[0] == '\0' ||
		context->tp_collective_control_port_base == 0u ||
		context->tp_collective_topology.abi_version !=
			SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION ||
		context->tp_collective_topology.descriptor_bytes !=
			SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES ||
		context->tp_collective_topology.rank_count != context->tp_degree) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&shape,0,sizeof(shape));
	shape.abi_version = SPARK_DSV4_PARALLEL_SHAPE_ABI_VERSION;
	shape.tp_degree = context->tp_degree;
	shape.tp_rank = context->tp_rank;
	shape.pp_stage_count = hybrid != 0u ? context->pp_stage_count : 1u;
	shape.pp_stage_index = hybrid != 0u ? context->pp_stage_index : 0u;
	if ( SparkDsv4TpDeriveNodeConfig(&shape,&tp_config) != SPARK_STATUS_OK || (context->tp_configuration_hash != 0u && context->tp_configuration_hash != tp_config.configuration_hash) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	lane_page_capacity = (context->max_sequence_positions +
		SPARK_DSV4_RESIDENT_DECODE_STAGE_CACHE_BLOCK_TOKENS - 1u) /
		SPARK_DSV4_RESIDENT_DECODE_STAGE_CACHE_BLOCK_TOKENS;
	if ( host_services->kv_logical_page_capacity < lane_page_capacity ||
		host_services->kv_logical_page_capacity >
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LOGICAL_PAGE_COUNT ||
		host_services->kv_physical_page_capacity == 0u ||
		host_services->kv_physical_page_capacity >
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PHYSICAL_PAGE_COUNT ||
		host_services->kv_physical_page_capacity >
		host_services->kv_logical_page_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->stage_count = context->stage_count;
	state->stage_index = context->stage_index;
	state->first_layer_index = context->first_layer_index;
	state->layer_count = context->layer_count;
	state->resident_sequence_capacity = context->resident_sequence_capacity;
	state->pipeline_slot_count = context->pipeline_slot_count;
	state->max_sequence_positions = context->max_sequence_positions;
	state->logical_page_capacity = host_services->kv_logical_page_capacity;
	state->physical_page_capacity = host_services->kv_physical_page_capacity;
	state->tp_degree = tp_config.tp_degree;
	state->tp_rank = tp_config.tp_rank;
	state->vocabulary_row_start = tp_config.vocabulary_row_start;
	state->vocabulary_rows_per_rank = tp_config.vocabulary_rows_per_rank;
	state->tp_configuration_hash = tp_config.configuration_hash;
	state->pp_stage_count = hybrid != 0u ? context->pp_stage_count : (parallel != 0u ? 1u : context->stage_count);
	state->pp_stage_index = hybrid != 0u ? context->pp_stage_index : (parallel != 0u ? 0u : context->stage_index);
	state->graph_capacity = context->cuda_graph_count;
	state->execution_stream = host_services->execution_stream;
	if ( host_services->kv_backing_directory == 0 &&
		host_services->kv_backing_maximum_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( host_services->kv_backing_directory != 0 )
	{
		directory_bytes = strlen(host_services->kv_backing_directory) + 1u;
		if ( directory_bytes > sizeof(state->kv_backing_directory) )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		memcpy(state->kv_backing_directory,
			host_services->kv_backing_directory,directory_bytes);
		state->kv_backing_maximum_bytes =
			host_services->kv_backing_maximum_bytes;
	}
	state->mtp_armed = 0u;
	state->dspark_enabled = 1u;
	status = SparkDsv4ModuleInitializeTpCollective(state,context);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*pack_path_out = context->stage_pack_path;
	return(SPARK_STATUS_OK);
}

// The slice sanity beyond ranges: position agreement, and the hash-router
// pin - token ids exist only beside the embedding, so a slice that starts
// inside the hash range without layer zero cannot route and is refused.
static SparkStatus SparkDsv4ModuleValidateSlice(SparkDsv4ModuleState *state)
{
	if ( state->stage_index >= state->stage_count || state->first_layer_index + state->layer_count > SPARK_DSV4_MODEL_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_slice_invalid stage=%u/%u slice=%u+%u\n",SPARK_DSV4_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( state->tp_degree > 1u )
	{
		if ( (state->pp_stage_count == 1u && (state->stage_count != state->tp_degree || state->stage_index != state->tp_rank || state->first_layer_index != 0u || state->layer_count != SPARK_DSV4_MODEL_LAYER_COUNT)) ||
			(state->pp_stage_count > 1u && (state->stage_count != state->pp_stage_count || state->stage_index != state->pp_stage_index)) )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
		state->participates_final_head = state->first_layer_index +
			state->layer_count == SPARK_DSV4_MODEL_LAYER_COUNT ? 1u : 0u;
		state->owns_final_head = state->participates_final_head != 0u &&
			state->tp_rank + 1u == state->tp_degree ? 1u : 0u;
		return(SPARK_STATUS_OK);
	}
	state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
	state->owns_final_head = state->first_layer_index + state->layer_count == SPARK_DSV4_MODEL_LAYER_COUNT ? 1u : 0u;
	state->participates_final_head = state->owns_final_head;
	if ( (state->stage_index == 0u) != (state->owns_embedding != 0u) || (state->stage_index + 1u == state->stage_count) != (state->owns_final_head != 0u) )
	{
		fprintf(stderr,"%s config_position_mismatch stage=%u/%u slice=%u+%u\n",SPARK_DSV4_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( state->owns_embedding == 0u && state->first_layer_index < SPARK_DSV4_MODEL_HASH_ROUTED_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_hash_layer_without_tokens slice=%u+%u\n",SPARK_DSV4_MODULE_TAG,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

// Per-stage ordinals: dense compress and CSA numbering inside the slice,
// and the topk column budget - the window plus the larger of the indexer
// top-k and the full compressed slot count an HCA layer attends.
static void SparkDsv4ModuleBuildOrdinals(SparkDsv4ModuleState *state)
{
	uint32_t layer,kind,hca_columns = state->max_sequence_positions / SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO;
	for (layer = 0; layer < SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT; layer++)
		state->csa_ordinal_by_layer[layer] = UINT32_MAX;
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		kind = SparkDsv4ModelLayerKind(layer);
		if ( kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		{
			state->compress_layer_count++;
			state->layers[layer].compressor.ratio = kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO : SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO;
			state->layers[layer].compressor.overlap = kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR : 1u;
		}
		if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		{
			state->csa_ordinal_by_layer[layer] = state->csa_layer_count++;
			state->layers[layer].indexer.compressor.ratio = SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO;
			state->layers[layer].indexer.compressor.overlap = SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR;
		}
	}
	state->index_slot_capacity = state->max_sequence_positions / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO;
	state->topk_column_count = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS + (SPARK_DSV4_MODEL_INDEX_TOP_K > hca_columns ? SPARK_DSV4_MODEL_INDEX_TOP_K : hca_columns);
}

static void SparkDsv4ModuleFillLinearView(SparkDsv4LinearView *view, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	view->abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = entry->weight_format;
	view->rows = entry->rows;
	view->columns = entry->columns;
	view->payload = payload;
	view->scale_data = scale;
}

static int32_t SparkDsv4ModuleResolvedShape(
	const SparkDsv4ModuleState *state,
	const SparkDsv4StagePackEntry *entry,
	SparkDsv4StagePackTensorShape *shape)
{
	if ( state == 0 || entry == 0 || shape == 0 ||
		SparkDsv4StagePackResolvedShape(entry->tensor_kind,entry->layer_index,entry->layer_index == SPARK_DSV4_STAGEPACK_GLOBAL_LAYER ? 1u : 0u,shape) < 0 )
		return(-1);
	if ( state->tp_degree == 1u )
		return(0);
	if ( entry->layer_index == SPARK_DSV4_STAGEPACK_GLOBAL_LAYER )
	{
		if ( entry->tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_LM_HEAD )
			shape->rows = state->vocabulary_rows_per_rank;
		return(0);
	}
	/* DSpark draft layers are REPLICATED full-width on every rank (the
	 * draft runs replicated with zero draft collectives), so the MTP layer
	 * range keeps the full geometry instead of the TP shard. */
	if ( SparkDsv4StagePackLayerIsMtp(entry->layer_index) != 0u )
		return(0);
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_WQ_A:
	case SPARK_DSV4_STAGEPACK_TENSOR_ATTN_SINK:
		if ( entry->tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_WQ_A )
			shape->rows = SPARK_DSV4_MODEL_QUERY_LORA_RANK / state->tp_degree;
		else
			shape->columns = SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT / state->tp_degree;
		break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WQ_B:
		shape->rows = SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION / state->tp_degree;
		break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WKV:
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WKV:
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WGATE:
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WKV:
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WGATE:
		shape->rows /= state->tp_degree;
		break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WO_A:
		shape->rows = SparkDsv4ModuleTpOutputLora(state);
		shape->columns = SparkDsv4ModuleTpOutputGroupInput(state);
		break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WO_B:
		shape->rows = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
		shape->columns = SparkDsv4ModuleTpOutputLora(state);
		break;
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W1:
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W3:
		shape->rows = SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT * SparkDsv4ModuleTpExpertWidth(state);
		break;
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W2:
		shape->rows = SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT *
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
		shape->columns = SparkDsv4ModuleTpExpertWidth(state);
		break;
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W1:
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W3:
		shape->rows = SparkDsv4ModuleTpExpertWidth(state);
		break;
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W2:
		shape->rows = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
		shape->columns = SparkDsv4ModuleTpExpertWidth(state);
		break;
	default:
		break;
	}
	return(0);
}

static SparkStatus SparkDsv4ModuleValidateEntry(SparkDsv4ModuleState *state, const SparkDsv4StagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	SparkDsv4StagePackTensorShape shape;
	uint64_t payload_bytes,scale_bytes;
	uint32_t global = entry->layer_index == SPARK_DSV4_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	uint32_t in_slice = (SparkDsv4StagePackLayerIsMtp(entry->layer_index) != 0u && SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u) || (entry->layer_index >= state->first_layer_index && entry->layer_index < state->first_layer_index + state->layer_count) ? 1u : 0u;
	if ( entry->tensor_kind >= SPARK_DSV4_STAGEPACK_TENSOR_KIND_COUNT || (global == 0u && in_slice == 0u) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( SparkDsv4ModuleResolvedShape(state,entry,&shape) < 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( shape.rows != entry->rows || shape.columns != entry->columns || shape.weight_format != entry->weight_format )
		return(SPARK_STATUS_VALIDATION_FAILED);
	payload_bytes = SparkDsv4StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	scale_bytes = SparkDsv4StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
	if ( entry->payload_offset + payload_bytes > file_bytes || (scale_bytes != 0u && (entry->scale_offset != entry->payload_offset + payload_bytes || entry->scale_offset + scale_bytes > file_bytes)) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	*is_global = global;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindGlobal(SparkDsv4ModuleState *state, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	(void)scale;
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING: state->token_embedding_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_LM_HEAD: state->lm_head_weight_bf16 = payload; SparkDsv4ModuleFillLinearView(&state->lm_head_view,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_FN: state->hc_head_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_BASE: state->hc_head_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE: state->hc_head_scale_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_MAIN_PROJ: SparkDsv4ModuleFillLinearView(&state->mtp.main_proj,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_MAIN_NORM: state->mtp.main_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_FINAL_NORM: state->mtp.final_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_FN: state->mtp.hc_head_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_BASE: state->mtp.hc_head_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE: state->mtp.hc_head_scale_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_MARKOV_W1: SparkDsv4ModuleFillLinearView(&state->mtp.markov_w1,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_MARKOV_W2: SparkDsv4ModuleFillLinearView(&state->mtp.markov_w2,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_CONFIDENCE_PROJ: SparkDsv4ModuleFillLinearView(&state->mtp.confidence_proj,entry,payload,scale); break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	state->global_seen_bits |= 1ull << entry->tensor_kind;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindLayerAttn(SparkDsv4LayerWeights *layer, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_ATTN_SINK: layer->attn.sink_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WQ_A: SparkDsv4ModuleFillLinearView(&layer->attn.wq_a,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_Q_NORM: layer->attn.q_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WQ_B: SparkDsv4ModuleFillLinearView(&layer->attn.wq_b,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WKV: SparkDsv4ModuleFillLinearView(&layer->attn.wkv,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_KV_NORM: layer->attn.kv_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WO_A: SparkDsv4ModuleFillLinearView(&layer->attn.wo_a,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WO_B: SparkDsv4ModuleFillLinearView(&layer->attn.wo_b,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_ATTN_NORM: layer->attn_norm_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_FFN_NORM: layer->ffn_norm_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE: layer->compressor.ape_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WKV: SparkDsv4ModuleFillLinearView(&layer->compressor.wkv,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WGATE: SparkDsv4ModuleFillLinearView(&layer->compressor.wgate,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_NORM: layer->compressor.norm_weight_bf16 = payload; break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindLayerIndexer(SparkDsv4LayerWeights *layer, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WQ_B: SparkDsv4ModuleFillLinearView(&layer->indexer.wq_b,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WEIGHTS: SparkDsv4ModuleFillLinearView(&layer->indexer.weights_proj,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_APE: layer->indexer.compressor.ape_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WKV: SparkDsv4ModuleFillLinearView(&layer->indexer.compressor.wkv,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WGATE: SparkDsv4ModuleFillLinearView(&layer->indexer.compressor.wgate,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_NORM: layer->indexer.compressor.norm_weight_bf16 = payload; break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindLayerRest(SparkDsv4LayerWeights *layer, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_FN: layer->hc.attn_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_FN: layer->hc.ffn_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_BASE: layer->hc.attn_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_BASE: layer->hc.ffn_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_SCALE: layer->hc.attn_scale_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_SCALE: layer->hc.ffn_scale_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_GATE_WEIGHT: SparkDsv4ModuleFillLinearView(&layer->moe.gate,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_GATE_BIAS: layer->moe.gate_bias_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_GATE_TID2EID: layer->moe.gate_tid2eid_u32 = (const uint32_t *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W1: SparkDsv4ModuleFillLinearView(&layer->moe.experts_w1,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W2: SparkDsv4ModuleFillLinearView(&layer->moe.experts_w2,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W3: SparkDsv4ModuleFillLinearView(&layer->moe.experts_w3,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W1: SparkDsv4ModuleFillLinearView(&layer->moe.shared_w1,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W2: SparkDsv4ModuleFillLinearView(&layer->moe.shared_w2,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W3: SparkDsv4ModuleFillLinearView(&layer->moe.shared_w3,entry,payload,scale); break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindLayer(SparkDsv4ModuleState *state, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	SparkDsv4LayerWeights *layer;
	uint64_t *seen;
	SparkStatus status;
	if ( SparkDsv4StagePackLayerIsMtp(entry->layer_index) != 0u )
	{
		uint32_t stage = SparkDsv4StagePackMtpStage(entry->layer_index);
		if ( stage >= SPARK_DSV4_STAGEPACK_MTP_LAYER_COUNT_MAX )
			return(SPARK_STATUS_VALIDATION_FAILED);
		layer = &state->mtp_layers[stage];
		seen = &state->mtp_seen_bits[stage];
	}
	else
	{
		layer = &state->layers[entry->layer_index];
		seen = &state->layer_seen_bits[entry->layer_index];
	}
	if ( entry->tensor_kind <= SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_NORM && entry->tensor_kind >= SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE )
		status = SparkDsv4ModuleBindLayerAttn(layer,entry,payload,scale);
	else if ( entry->tensor_kind <= SPARK_DSV4_STAGEPACK_TENSOR_FFN_NORM )
		status = SparkDsv4ModuleBindLayerAttn(layer,entry,payload,scale);
	else if ( entry->tensor_kind >= SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WQ_B && entry->tensor_kind <= SPARK_DSV4_STAGEPACK_TENSOR_INDEX_NORM )
		status = SparkDsv4ModuleBindLayerIndexer(layer,entry,payload,scale);
	else
		status = SparkDsv4ModuleBindLayerRest(layer,entry,payload,scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*seen |= 1ull << entry->tensor_kind;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleLoadEntry(SparkDsv4ModuleState *state, FILE *file, const SparkDsv4StagePackEntry *entry, uint64_t file_bytes)
{
	uint64_t payload_bytes = SparkDsv4StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	uint64_t scale_bytes = SparkDsv4StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
	void *payload = 0,*scale = 0;
	uint32_t is_global = 0u;
	SparkStatus status = SparkDsv4ModuleValidateEntry(state,entry,file_bytes,&is_global);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s pack_entry_invalid kind=%u layer=%u\n",SPARK_DSV4_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(status);
	}
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,scale_bytes,&scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( is_global != 0u )
		return(SparkDsv4ModuleBindGlobal(state,entry,payload,scale));
	return(SparkDsv4ModuleBindLayer(state,entry,payload,scale));
}

// Coverage: every layer in the slice must have seen the exact kind set its
// attention class demands, the MTP layer its SWA score-routed set, and the
// globals the position-derived set - a missing tensor is a refused pack.
static uint64_t SparkDsv4ModuleExpectedLayerBits(uint32_t layer_index)
{
	uint32_t kind = SparkDsv4StagePackLayerKind(layer_index),tensor;
	uint64_t bits = 0u;
	for (tensor = SPARK_DSV4_STAGEPACK_TENSOR_ATTN_SINK; tensor <= SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W3; tensor++)
		bits |= 1ull << tensor;
	bits &= ~(1ull << (SparkDsv4StagePackLayerIsHashRouted(layer_index) != 0u ? SPARK_DSV4_STAGEPACK_TENSOR_GATE_BIAS : SPARK_DSV4_STAGEPACK_TENSOR_GATE_TID2EID));
	if ( kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		bits |= (1ull << SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WKV) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WGATE) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_NORM);
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
	{
		bits |= (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WQ_B) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WEIGHTS) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_APE);
		bits |= (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WKV) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WGATE) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_NORM);
	}
	return(bits);
}

static SparkStatus SparkDsv4ModuleVerifyCoverage(SparkDsv4ModuleState *state)
{
	uint64_t expected_globals = 0u;
	uint32_t layer,tensor;
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
		if ( state->layer_seen_bits[layer] != SparkDsv4ModuleExpectedLayerBits(layer) )
		{
			fprintf(stderr,"%s pack_layer_coverage layer=%u seen=%llx\n",SPARK_DSV4_MODULE_TAG,layer,(unsigned long long)state->layer_seen_bits[layer]);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	if ( state->owns_embedding != 0u || (state->participates_final_head != 0u && SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u) )
		expected_globals |= 1ull << SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->participates_final_head != 0u )
	{
		for (tensor = SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM; tensor <= SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE; tensor++)
			expected_globals |= 1ull << tensor;
	}
	/* DSpark draft: every rank's pack carries the three full draft layers
	 * (they ride the standard per-layer kinds under the MTP layer range)
	 * plus the stage extras; a missing draft tensor is a refused pack. */
	if ( SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u )
	{
		uint32_t stage;
		for (tensor = SPARK_DSV4_STAGEPACK_TENSOR_MTP_MAIN_PROJ; tensor <= SPARK_DSV4_STAGEPACK_TENSOR_MTP_CONFIDENCE_PROJ; tensor++)
			expected_globals |= 1ull << tensor;
		for (stage = 0u; stage < SPARK_DSV4_MODEL_MTP_LAYER_COUNT; stage++)
			if ( state->mtp_seen_bits[stage] != SparkDsv4ModuleExpectedLayerBits(SPARK_DSV4_STAGEPACK_MTP_LAYER(stage)) )
			{
				fprintf(stderr,"%s pack_mtp_coverage stage=%u seen=%llx\n",SPARK_DSV4_MODULE_TAG,stage,(unsigned long long)state->mtp_seen_bits[stage]);
				return(SPARK_STATUS_VALIDATION_FAILED);
			}
		if ( state->mtp.main_proj.payload == 0 || state->mtp.main_proj.scale_data == 0 ||
			state->mtp.main_norm_weight_bf16 == 0 || state->mtp.final_norm_weight_bf16 == 0 ||
			state->mtp.hc_head_fn_f32 == 0 || state->mtp.hc_head_base_f32 == 0 ||
			state->mtp.hc_head_scale_f32 == 0 || state->mtp.markov_w1.payload == 0 ||
			state->mtp.markov_w2.payload == 0 || state->mtp.confidence_proj.payload == 0 )
		{
			fprintf(stderr,"%s pack_mtp_extras_coverage incomplete\n",SPARK_DSV4_MODULE_TAG);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	if ( state->global_seen_bits != expected_globals )
	{
		fprintf(stderr,"%s pack_global_coverage seen=%llx expected=%llx\n",SPARK_DSV4_MODULE_TAG,(unsigned long long)state->global_seen_bits,(unsigned long long)expected_globals);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleLoadPack(SparkDsv4ModuleState *state, const char *path)
{
	SparkDsv4StagePackHeader header,expected;
	SparkDsv4StagePackEntry *directory;
	FILE *file;
	SparkStatus status;
	int32_t compare;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s pack_open_failed path=%s\n",SPARK_DSV4_MODULE_TAG,path);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageModulePackRead(SPARK_DSV4_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
	{
		SparkDsv4StagePackExpectedGeometry(&expected,state->first_layer_index,state->layer_count);
		if ( state->tp_degree > 1u )
			expected.tensor_count = SparkDsv4StagePackExpectedTensorCountForOwnership(
				state->first_layer_index,
				state->layer_count,
				state->owns_embedding,
				state->participates_final_head);
		compare = SparkDsv4StagePackCompareGeometry(&header,&expected);
		if ( compare != 0 )
		{
			fprintf(stderr,"%s pack_geometry_mismatch field=%s\n",SPARK_DSV4_MODULE_TAG,SparkDsv4StagePackGeometryFieldName(compare));
			status = SPARK_STATUS_VALIDATION_FAILED;
		}
	}
	directory = status == SPARK_STATUS_OK ? (SparkDsv4StagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkDsv4StagePackEntry)) : 0;
	if ( status == SPARK_STATUS_OK && directory == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_DSV4_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkDsv4StagePackEntry));
	for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
		status = SparkDsv4ModuleLoadEntry(state,file,&directory[index],header.file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleVerifyCoverage(state);
	free(directory);
	fclose(file);
	return(status);
}

// Host YaRN frequency table, the reference precompute arithmetic; the
// interpolation ramp engages only when original positions are declared.
static void SparkDsv4ModuleComputeFreqs(float *freqs, float base, uint32_t original, float factor)
{
	uint32_t rope_dim = SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,pair;
	float low,high,ramp,smooth,frequency;
	low = floorf((float)rope_dim * logf((float)original / ((float)SPARK_DSV4_MODEL_ROPE_BETA_FAST * 2.0f * 3.14159265f)) / (2.0f * logf(base)));
	high = ceilf((float)rope_dim * logf((float)original / ((float)SPARK_DSV4_MODEL_ROPE_BETA_SLOW * 2.0f * 3.14159265f)) / (2.0f * logf(base)));
	if ( low < 0.0f )
		low = 0.0f;
	if ( high > (float)(rope_dim - 1u) )
		high = (float)(rope_dim - 1u);
	if ( low == high )
		high += 0.001f;
	for (pair = 0; pair < rope_dim / 2u; pair++)
	{
		frequency = 1.0f / powf(base,(float)(2u * pair) / (float)rope_dim);
		if ( original != 0u )
		{
			ramp = ((float)pair - low) / (high - low);
			if ( ramp < 0.0f )
				ramp = 0.0f;
			if ( ramp > 1.0f )
				ramp = 1.0f;
			smooth = 1.0f - ramp;
			frequency = frequency / factor * (1.0f - smooth) + frequency * smooth;
		}
		freqs[pair] = frequency;
	}
}

static SparkStatus SparkDsv4ModuleUploadFreqs(SparkDsv4ModuleState *state)
{
	float host_freqs[SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION / 2u];
	cudaStream_t stream = (cudaStream_t)state->execution_stream;
	SparkStatus status;
	cudaError_t error;
	status = SparkStageModuleDeviceAllocate(&state->ledger,sizeof(host_freqs),(void **)&state->base_freqs_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,sizeof(host_freqs),(void **)&state->compress_freqs_f32);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkDsv4ModuleComputeFreqs(host_freqs,SPARK_DSV4_MODEL_ATTN_ROPE_THETA,0u,(float)SPARK_DSV4_MODEL_ATTN_YARN_FACTOR);
	error = cudaMemcpyAsync(state->base_freqs_f32,host_freqs,sizeof(host_freqs),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
	{
		SparkDsv4ModuleComputeFreqs(host_freqs,SPARK_DSV4_MODEL_COMPRESS_ROPE_THETA,SPARK_DSV4_MODEL_ATTN_YARN_ORIGINAL_POSITIONS,(float)SPARK_DSV4_MODEL_ATTN_YARN_FACTOR);
		error = cudaMemcpyAsync(state->compress_freqs_f32,host_freqs,sizeof(host_freqs),cudaMemcpyHostToDevice,stream);
	}
	if ( error == cudaSuccess )
		error = cudaStreamSynchronize(stream);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"freq_upload"));
}

/*
 * Cache pools. Each layer holds only the slots required by its attention
 * class: SWA keeps the local window, HCA keeps its compressed stream, and
 * CSA keeps both. Per-layer offsets preserve the reference's contiguous
 * [window | stream] addressing while compressor state uses the matching
 * class footprint. The indexer keeps its rotated cache and small overlap
 * state per CSA ordinal.
 */
// One-time MXFP4 shadow of the lm_head plus per-neuron certified error
// norms, the mimo25 screened-head pattern; head stage only, built
// synchronously at initialize.

/*
 * The hash routing tables come off the pack unchecked and feed the
 * device grouping kernel's shared histogram directly, so a corrupt or
 * mismatched table would write out of bounds. One init-time scan per
 * hash layer, blocking readback, hard failure on any out-of-range
 * entry.
 */
static SparkStatus SparkDsv4ModuleValidateHashTables(SparkDsv4ModuleState *state)
{
	uint32_t layer,*flag_device,flag_host = 0u;
	uint64_t entries = (uint64_t)SPARK_DSV4_MODEL_VOCAB_COUNT * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN;
	cudaStream_t stream = (cudaStream_t)state->execution_stream;
	SparkStatus status = SPARK_STATUS_OK;
	cudaError_t error;
	error = cudaMalloc((void **)&flag_device,sizeof(uint32_t));
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"tid2eid_flag"));
	error = cudaMemsetAsync(flag_device,0,sizeof(uint32_t),stream);
	for (layer = state->first_layer_index; error == cudaSuccess && layer < state->first_layer_index + state->layer_count; layer++)
		if ( state->layers[layer].moe.gate_tid2eid_u32 != 0 )
				error = SparkDsv4LaunchValidateTid2Eid(stream,state->layers[layer].moe.gate_tid2eid_u32,entries,flag_device);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(&flag_host,flag_device,sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess )
		error = cudaStreamSynchronize(stream);
	cudaFree(flag_device);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"tid2eid_scan"));
	if ( flag_host != 0u )
	{
		fprintf(stderr,"%s hash_table_out_of_range\n",SPARK_DSV4_MODULE_TAG);
		status = SPARK_STATUS_VALIDATION_FAILED;
	}
	return(status);
}

static SparkStatus SparkDsv4ModuleBuildHeadShadow(SparkDsv4ModuleState *state)
{
	uint64_t vocab = state->vocabulary_rows_per_rank,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	cudaStream_t stream = (cudaStream_t)state->execution_stream;
	SparkStatus status;
	if ( state->participates_final_head == 0u )
		return(SPARK_STATUS_OK);
	status = SparkStageModuleDeviceAllocate(&state->ledger,(vocab * dim) / 2u,(void **)&state->head_shadow_payload);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(vocab * dim) / 32u,(void **)&state->head_shadow_scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,vocab * sizeof(float),(void **)&state->head_error_norm_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,SparkDsv4LaunchHeadShadowQuantize(stream,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,(uint32_t)vocab,(uint32_t)dim),"head_shadow_quantize");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,SparkHeadCertifiedFp8PayloadBytes(vocab,dim),(void **)&state->head_certified_fp8_payload);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,SparkHeadCertifiedFp8NormBytes(vocab,dim),(void **)&state->head_certified_fp8_scale_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,SparkHeadCertifiedFp8NormBytes(vocab,dim),(void **)&state->head_certified_fp8_norm_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,SparkDsv4LaunchHeadCertifiedFp8Quantize(stream,state->lm_head_weight_bf16,state->head_certified_fp8_payload,state->head_certified_fp8_scale_f32,state->head_certified_fp8_norm_f32,(uint32_t)vocab,(uint32_t)dim),"head_certified_fp8_quantize");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,cudaStreamSynchronize(stream),"head_shadow_sync");
	return(status);
}

// Post-pack validation and derived-weight construction, one call from
// the initialize chain.
static SparkStatus SparkDsv4ModuleFinalizeLoad(SparkDsv4ModuleState *state)
{
	SparkStatus status;
	status = SparkDsv4ModuleValidateHashTables(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleBuildHeadShadow(state);
	return(status);
}

static SparkStatus SparkDsv4ModuleCopyKvPage(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes)
{
	SparkDsv4ModuleState *state;
	struct timespec interval;
	cudaError_t error;
	state = (SparkDsv4ModuleState *)context;
	if ( state == 0 || state->kv_page_store_stream == 0 ||
		device_address == 0u || host_address == 0 || bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		error = cudaMemcpyAsync(host_address,(const void *)device_address,bytes,
			cudaMemcpyDeviceToHost,(cudaStream_t)state->kv_page_store_stream);
	else if ( direction == SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		error = cudaMemcpyAsync((void *)device_address,host_address,bytes,
			cudaMemcpyHostToDevice,(cudaStream_t)state->kv_page_store_stream);
	else
		return(SPARK_STATUS_INVALID_ARGUMENT);
	interval.tv_sec = 0;
	interval.tv_nsec = 50000L;
	while ( error == cudaSuccess )
	{
		error = cudaStreamQuery((cudaStream_t)state->kv_page_store_stream);
		if ( error == cudaErrorNotReady )
		{
			(void)nanosleep(&interval,0);
			error = cudaSuccess;
		}
		else
			break;
	}
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"kv_page_copy"));
}

static SparkStatus SparkDsv4ModuleInitializeKvBacking(
	SparkDsv4ModuleState *state)
{
	SparkKvPageStoreConfiguration configuration;
	cudaError_t error;
	SparkStatus status;
	if ( state->kv_backing_directory[0] == '\0' )
		return(SPARK_STATUS_OK);
	error = cudaHostAlloc(&state->kv_page_store_staging,
		state->paged_cache.layout.page_stride_bytes,cudaHostAllocPortable);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
			"kv_backing_staging"));
	error = cudaStreamCreateWithFlags((cudaStream_t *)&state->kv_page_store_stream,
		cudaStreamNonBlocking);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
			"kv_backing_stream"));
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	configuration.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS |
		SPARK_KV_PAGE_STORE_FLAG_DIRECT_IO;
	configuration.logical_page_capacity = state->logical_page_capacity;
	configuration.transfer_capacity = state->physical_page_capacity;
	configuration.page_bytes = state->paged_cache.layout.page_stride_bytes;
	configuration.maximum_backing_bytes = state->kv_backing_maximum_bytes;
	configuration.backing_path = state->kv_backing_directory;
	configuration.staging_address = state->kv_page_store_staging;
	configuration.staging_bytes = state->paged_cache.layout.page_stride_bytes;
	configuration.copy_function = SparkDsv4ModuleCopyKvPage;
	configuration.copy_context = state;
	status = SparkKvPageStoreInitialize(&state->kv_page_store,&configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->paged_cache.arena.evict_function = SparkKvPageStoreWriteback;
	state->paged_cache.arena.evict_context = &state->kv_page_store;
	state->paged_cache.page_cache.page_store = &state->kv_page_store;
	state->kv_page_store_enabled = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleUploadPageScoreSpans(
	SparkDsv4ModuleState *state)
{
	const SparkDsv4PagedLayerLayout *layout;
	SparkDsv4PagedScoreSpan *span;
	cudaError_t error;
	SparkStatus status;
	uint32_t kind,layer;
	for (layer=state->first_layer_index;
		layer<state->first_layer_index+state->layer_count; layer++)
	{
		kind = SparkDsv4ModelLayerKind(layer);
		if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA )
			continue;
		layout = &state->paged_cache.layout.layers[layer];
		span = &state->page_score_spans[state->page_score_span_count++];
		span->offset_words = layout->compressor_score_offset_bytes / sizeof(float);
		span->element_count = SparkDsv4PoolCompressStateLaneElements(kind);
		if ( kind != SPARK_DSV4_MODEL_LAYER_KIND_CSA )
			continue;
		span = &state->page_score_spans[state->page_score_span_count++];
		span->offset_words = layout->index_score_offset_bytes / sizeof(float);
		span->element_count = SparkDsv4PoolIndexStateLaneElements();
	}
	if ( state->page_score_span_count == 0u )
		return(SPARK_STATUS_OK);
	status = SparkStageModuleDeviceAllocate(&state->ledger,
		(uint64_t)state->page_score_span_count * sizeof(state->page_score_spans[0]),
		(void **)&state->device_page_score_spans);
	if ( status != SPARK_STATUS_OK )
		return(status);
	error = cudaMemcpyAsync(state->device_page_score_spans,
		state->page_score_spans,
		(uint64_t)state->page_score_span_count * sizeof(state->page_score_spans[0]),
		cudaMemcpyHostToDevice,(cudaStream_t)state->execution_stream);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"page_score_spans"));
}

static SparkStatus SparkDsv4ModuleAllocatePools(SparkDsv4ModuleState *state)
{
	SparkDsv4PagedCacheConfiguration configuration;
	const SparkDsv4PagedLayerLayout *layout;
	uint32_t layer;
	SparkStatus status;
	memset(&configuration,0,sizeof(configuration));
	configuration.first_layer_index = state->first_layer_index;
	configuration.layer_count = state->layer_count;
	configuration.resident_sequence_capacity = state->resident_sequence_capacity;
	configuration.maximum_sequence_positions = state->max_sequence_positions;
	configuration.logical_page_capacity = state->logical_page_capacity;
	configuration.physical_page_capacity = state->physical_page_capacity;
	status = SparkDsv4PagedCacheInitialize(&state->paged_cache,&configuration,
		&state->ledger);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkDsv4ModuleInitializeKvBacking(state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->cache_admission_logical_pages = (uint32_t *)calloc(
		state->paged_cache.lane_page_capacity,sizeof(uint32_t));
	if ( state->cache_admission_logical_pages == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->kv_cache_bf16 = state->paged_cache.device_page_pool;
	state->index_cache_bf16 = state->paged_cache.device_page_pool;
	state->compress_kv_state_f32 =
		(float *)state->paged_cache.device_page_pool;
	state->compress_score_state_f32 =
		(float *)state->paged_cache.device_page_pool;
	state->index_kv_state_f32 = (float *)state->paged_cache.device_page_pool;
	state->index_score_state_f32 = (float *)state->paged_cache.device_page_pool;
	state->index_lane_stride =
		state->paged_cache.layout.page_stride_bytes /
		SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	state->index_state_lane_stride =
		state->paged_cache.layout.page_stride_bytes / sizeof(float);
	for (layer=state->first_layer_index;
		layer<state->first_layer_index+state->layer_count; layer++)
	{
		layout = &state->paged_cache.layout.layers[layer];
		state->cache_offset_by_layer[layer] = layout->attention_offset_bytes /
			SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
		state->cache_lane_stride_by_layer[layer] = state->index_lane_stride;
		state->compress_state_offset_by_layer[layer] =
			layout->compressor_kv_offset_bytes / sizeof(float);
		state->compress_score_state_offset_by_layer[layer] =
			layout->compressor_score_offset_bytes / sizeof(float);
		state->compress_state_lane_stride_by_layer[layer] =
			state->index_state_lane_stride;
		state->index_cache_offset_by_layer[layer] =
			layout->index_cache_offset_bytes /
			SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
		state->index_kv_state_offset_by_layer[layer] =
			layout->index_kv_offset_bytes / sizeof(float);
		state->index_score_state_offset_by_layer[layer] =
			layout->index_score_offset_bytes / sizeof(float);
	}
	status = SparkDsv4ModuleUploadPageScoreSpans(state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u )
	{
		uint64_t tap_store_elements = (uint64_t)state->resident_sequence_capacity *
			SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT *
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
		status = SparkStageModuleDeviceAllocate(&state->ledger,
			tap_store_elements * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,
			&state->dspark_tap_store_bf16);
	}
	if ( status == SPARK_STATUS_OK && SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u )
	{
		/* Replicated draft ring: one 128-slot sliding window per draft
		 * layer per resident lane, bf16 512-wide kv. */
		uint64_t ring_layer_elements = (uint64_t)SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS *
			SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
		state->dspark_ring_layer_stride = ring_layer_elements;
		state->dspark_ring_lane_stride =
			(uint64_t)SPARK_DSV4_MODEL_MTP_LAYER_COUNT * ring_layer_elements;
		status = SparkStageModuleDeviceAllocate(&state->ledger,
			(uint64_t)state->resident_sequence_capacity *
				state->dspark_ring_lane_stride *
				SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,
			&state->dspark_ring_bf16);
	}
	state->resident_state_bytes = (uint64_t)state->physical_page_capacity *
		state->paged_cache.layout.page_stride_bytes;
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateCompressorScratch(
	SparkDsv4ModuleState *state,SparkDsv4CompressorScratch *scratch,
	uint64_t rows,uint64_t channels,uint64_t emit_width)
{
	SparkStatus status;
	status = SparkStageModuleDeviceAllocate(&state->ledger,
		rows * channels * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,&scratch->kv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,
			rows * channels * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,
			&scratch->score_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,
			rows * emit_width * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,
			&scratch->emit_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,
			rows * sizeof(uint32_t),(void **)&scratch->emitted_u32);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlotSmall(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	uint8_t *device_cursor,*host_cursor;
	uint32_t rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT,head_rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,metadata_rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT;
	uint64_t page_table_elements = (uint64_t)head_rows * state->paged_cache.lane_page_capacity;
	uint64_t page_table_bytes = page_table_elements * sizeof(uint32_t);
	uint64_t initialize_bytes = (uint64_t)head_rows * 2u * sizeof(uint32_t);
	uint64_t metadata_bytes = (uint64_t)metadata_rows * (3u * sizeof(uint32_t) + 3u * sizeof(uint64_t));
	uint64_t device_metadata_bytes = metadata_bytes + 2u * page_table_bytes + initialize_bytes;
	uint64_t host_bytes = metadata_bytes + (uint64_t)SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT * sizeof(uint32_t) + 4u * page_table_bytes + initialize_bytes;
	SparkStatus status;
	cudaError_t error;

	error = cudaHostAlloc(&slot->host_staging,host_bytes,cudaHostAllocPortable);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"host_staging"));
	memset(slot->host_staging,0,host_bytes);
	host_cursor = (uint8_t *)slot->host_staging;
	slot->host_input_token_ids = (uint32_t *)host_cursor;
	host_cursor += (uint64_t)metadata_rows * sizeof(uint32_t);
	slot->host_row_lane_indices = (uint32_t *)host_cursor;
	host_cursor += (uint64_t)metadata_rows * sizeof(uint32_t);
	slot->host_row_page_table_indices = (uint32_t *)host_cursor;
	host_cursor += (uint64_t)metadata_rows * sizeof(uint32_t);
	slot->host_row_positions = (uint64_t *)host_cursor;
	host_cursor += (uint64_t)metadata_rows * sizeof(uint64_t);
	slot->host_row_emit_positions = (uint64_t *)host_cursor;
	host_cursor += (uint64_t)metadata_rows * sizeof(uint64_t);
	slot->host_row_emit_positions_hca = (uint64_t *)host_cursor;
	host_cursor += (uint64_t)metadata_rows * sizeof(uint64_t);
	slot->host_output_token_ids = (uint32_t *)host_cursor;
	host_cursor += (uint64_t)SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT * sizeof(uint32_t);
	slot->host_logical_page_table = (uint32_t *)host_cursor;
	host_cursor += page_table_bytes;
	slot->host_physical_page_table = (uint32_t *)host_cursor;
	host_cursor += page_table_bytes;
	slot->host_page_table_update_indices = (uint32_t *)host_cursor;
	host_cursor += page_table_bytes;
	slot->host_page_table_update_values = (uint32_t *)host_cursor;
	host_cursor += page_table_bytes;
	slot->host_initialize_page_indices = (uint32_t *)host_cursor;
	host_cursor += (uint64_t)head_rows * sizeof(uint32_t);
	slot->host_initialize_parent_page_indices = (uint32_t *)host_cursor;
	status = SparkStageModuleDeviceAllocate(&state->ledger,device_metadata_bytes,(void **)&slot->input_token_ids);
	device_cursor = (uint8_t *)slot->input_token_ids;
	device_cursor += (uint64_t)metadata_rows * sizeof(uint32_t);
	slot->row_lane_indices = (uint32_t *)device_cursor;
	device_cursor += (uint64_t)metadata_rows * sizeof(uint32_t);
	slot->row_page_table_indices = (uint32_t *)device_cursor;
	device_cursor += (uint64_t)metadata_rows * sizeof(uint32_t);
	slot->row_positions = (uint64_t *)device_cursor;
	device_cursor += (uint64_t)metadata_rows * sizeof(uint64_t);
	slot->row_emit_positions = (uint64_t *)device_cursor;
	device_cursor += (uint64_t)metadata_rows * sizeof(uint64_t);
	slot->row_emit_positions_hca = (uint64_t *)device_cursor;
	device_cursor += (uint64_t)metadata_rows * sizeof(uint64_t);
	slot->page_table_update_indices = (uint32_t *)device_cursor;
	device_cursor += page_table_bytes;
	slot->page_table_update_values = (uint32_t *)device_cursor;
	device_cursor += page_table_bytes;
	slot->physical_page_table = state->paged_cache.device_page_table;
	slot->initialize_page_indices = (uint32_t *)device_cursor;
	device_cursor += (uint64_t)head_rows * sizeof(uint32_t);
	slot->initialize_parent_page_indices = (uint32_t *)device_cursor;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)head_rows * sizeof(uint32_t),(void **)&slot->output_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,
			(uint64_t)SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT *
			sizeof(uint32_t),(void **)&slot->resident_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)head_rows * sizeof(uint32_t),(void **)&slot->prefill_emit_rows_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->slot_counts);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->attention_slot_counts);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * state->topk_column_count * sizeof(int32_t),(void **)&slot->topk_idxs);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_MIX_ROWS * sizeof(float),(void **)&slot->mixes_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_HC_SPLIT_K_COUNT * SPARK_DSV4_HC_SPLIT_K_PARTIALS * sizeof(float),(void **)&slot->hc_partials_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float),(void **)&slot->pre_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float),(void **)&slot->post_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float),(void **)&slot->comb_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT * sizeof(float),(void **)&slot->moe_scores_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(float),(void **)&slot->moe_weights_f32);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlotWide(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	uint64_t rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	uint32_t query_dimension = SparkDsv4ModuleTpQueryDimension(state);
	uint64_t stream_bytes = rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * dim * bf16;
	uint64_t compress_channels = (uint64_t)SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
	uint64_t index_channels = (uint64_t)SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION;
	SparkStatus status;
	status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->streams_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->residual_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->reduced_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->normalized_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_QUERY_LORA_RANK * bf16,&slot->qr_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * query_dimension * bf16,&slot->q_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * bf16,&slot->kv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * query_dimension * bf16,&slot->attn_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)state->sparse_attn_partial_capacity * SPARK_DSV4_SPARSE_ATTN_PARTIAL_SCALARS * sizeof(float),(void **)&slot->sparse_attn_partials_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows *
			SparkDsv4ModuleTpOutputLora(state) * bf16,&slot->o_ranks_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocateCompressorScratch(state,
			&slot->compressor,rows,compress_channels,
			SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocateCompressorScratch(state,
			&slot->index_compressor,rows,index_channels,
			SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlotTail(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	uint64_t rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT,head_rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,head_candidate_bytes,full_candidate_bytes,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	uint32_t expert_width = SparkDsv4ModuleTpExpertWidth(state);
	SparkStatus status;
	status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_INDEX_DIMENSION * bf16,&slot->index_q_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * bf16,&slot->index_weights_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * sizeof(float),(void **)&slot->index_weights_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * (uint64_t)state->index_slot_capacity * sizeof(float),(void **)&slot->index_scores_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * expert_width * bf16,&slot->ffn_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * expert_width * bf16,&slot->ffn_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->ffn_delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->ffn_accum_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->moe_indices_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->grouped_rows_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->group_tile_prefix_w1_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->group_tile_prefix_w2_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->expert_offsets_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->moe_inverse_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * expert_width * bf16,&slot->moe_slot_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * expert_width * bf16,&slot->moe_slot_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * dim * bf16,&slot->moe_slot_out_bf16);
	if ( status == SPARK_STATUS_OK && state->participates_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,head_rows *
			state->vocabulary_rows_per_rank * bf16,&slot->head_logits_bf16);
	if ( status == SPARK_STATUS_OK && state->participates_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,
			SparkHeadCertifiedFp8ScratchBytes(state->vocabulary_rows_per_rank,dim),
			&slot->head_certified_scratch);
	if ( status == SPARK_STATUS_OK && state->participates_final_head != 0u )
	{
		head_candidate_bytes = head_rows * SPARK_DSV4_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP * sizeof(uint32_t);
		full_candidate_bytes = SparkHeadCertifiedFp8CandidateBytes(state->vocabulary_rows_per_rank);
		head_candidate_bytes = full_candidate_bytes > head_candidate_bytes ? full_candidate_bytes : head_candidate_bytes;
		status = SparkStageModuleDeviceAllocate(&state->ledger,head_candidate_bytes,(void **)&slot->head_candidate_ids_u32);
	}
	if ( status == SPARK_STATUS_OK && state->participates_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,head_rows * sizeof(uint32_t),(void **)&slot->head_candidate_counts_u32);
	if ( status == SPARK_STATUS_OK && state->participates_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,head_rows *
			sizeof(float),(void **)&slot->head_scores_f32);
	if ( status == SPARK_STATUS_OK && state->participates_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,head_rows *
			sizeof(uint64_t),(void **)&slot->head_maxloc_u64);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateDspark(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	uint64_t block = SPARK_DSV4_MODEL_DSPARK_SPEC_STEP,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION,vocab = SPARK_DSV4_MODEL_VOCAB_COUNT,qdim = SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION,bf16 = SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	SparkStatus status;
	status = SparkStageModuleDeviceAllocate(&state->ledger,block * sizeof(uint32_t),(void **)&slot->dspark_draft_token_ids);
	/* the collective's u64-max receive/combine writes the frame-width
	 * (rows=8) payload at the op's full_device base: each per-index op at
	 * entry i writes [i..i+7], so the buffer spans block + 8 entries */
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,
			(2u * block) * sizeof(uint64_t),(void **)&slot->dspark_maxloc_u64);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,
			cudaMemsetAsync(slot->dspark_maxloc_u64,0,
				(2u * block) * sizeof(uint64_t),(cudaStream_t)slot->cuda_stream),
			"dspark_maxloc_zero");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,block * sizeof(uint32_t),(void **)&slot->dspark_verify_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,block * sizeof(uint32_t),(void **)&slot->dspark_input_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,block * sizeof(uint64_t),(void **)&slot->dspark_row_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,block * sizeof(uint32_t),(void **)&slot->dspark_row_lane_indices);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,block * sizeof(float),(void **)&slot->dspark_scores_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,block * SPARK_DSV4_MODEL_HC_STREAM_COUNT * dim * bf16,&slot->dspark_x_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,block * qdim * bf16,&slot->dspark_q_attn_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,block * SPARK_DSV4_MODEL_OUTPUT_LORA_RANK * SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT * bf16,&slot->dspark_o_ranks_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT * dim * bf16,&slot->dspark_main_cat_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,dim * bf16,&slot->dspark_main_x_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT * dim * bf16,&slot->dspark_tap_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,block * SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT * dim * bf16,&slot->dspark_tap_ring_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u) * SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT * dim * bf16,&slot->dspark_verify_tap_bf16);
	/* CSA previous windows: csa_layer_count x 2 (kv+score) x ratio(4) x
	 * channels(2 x head dim) floats. Boundary emission slots:
	 * compress_layer_count x 2 x head-dim bf16 entries. */
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,
			(uint64_t)state->csa_layer_count * 2u *
			SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO *
			(2u * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION) * sizeof(float),
			(void **)&slot->dspark_csa_previous_save);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,
			(uint64_t)state->compress_layer_count * 2u *
			SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * bf16,
			(void **)&slot->dspark_emission_save);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,block * vocab * bf16,&slot->dspark_logits_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,block * vocab * sizeof(float),(void **)&slot->dspark_logits_f32);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlot(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	SparkStatus status = SparkDsv4ModuleAllocateSlotSmall(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocateSlotWide(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocateSlotTail(state,slot);
	if ( status == SPARK_STATUS_OK && SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u )
		status = SparkDsv4ModuleAllocateDspark(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaForkInitialize(SPARK_DSV4_MODULE_TAG,
			&slot->compute_fork);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaReadAheadInitialize(SPARK_DSV4_MODULE_TAG,
			&state->ledger,&slot->weight_read_ahead,
			state->multiprocessor_count <
			SPARK_DSV4_WEIGHT_READ_AHEAD_MAX_BLOCK_COUNT ?
			state->multiprocessor_count *
				SPARK_DSV4_WEIGHT_READ_AHEAD_THREAD_COUNT :
			SPARK_DSV4_WEIGHT_READ_AHEAD_MAX_BLOCK_COUNT *
				SPARK_DSV4_WEIGHT_READ_AHEAD_THREAD_COUNT);
	if ( status == SPARK_STATUS_OK && state->tp_device_collective_initialized != 0u &&
		state->tp_device_collective.memory_mode ==
		SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,
			cudaEventCreateWithFlags(&slot->tp_host_copy_event,cudaEventDisableTiming),
			"tp_host_event_create");
	return(status);
}

static SparkStatus SparkDsv4ModuleValidateFrameShape(
	const SparkDsv4ModuleState *state,
	const SparkModelDriverFrame *frame,
	uint32_t *is_prefill_out)
{
	const uint32_t known_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
	uint32_t is_prefill;
	if ( state == 0 || frame == 0 || is_prefill_out == 0 ||
		frame->program_id == 0u || frame->tokens_per_sequence == 0u ||
		frame->tokens_per_sequence >
		SPARK_MODEL_DRIVER_MAX_TOKENS_PER_SEQUENCE ||
		frame->reserved1 != 0u ||
		frame->execution_stream == 0 || frame->completion_function == 0 ||
		(frame->flags & ~known_flags) != 0u ||
		(frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u ||
		((frame->cache_lane_count != 0u) != (frame->cache_lanes != 0)) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	is_prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( frame->active_slot_count == 0u ||
		frame->active_slot_count > state->resident_sequence_capacity ||
		(frame->cache_lane_count != 0u &&
		 frame->cache_lane_count != frame->active_slot_count) ||
		frame->new_token_count == 0u ||
		frame->new_token_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT ||
		(is_prefill == 0u && frame->new_token_count != frame->active_slot_count) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (is_prefill != 0u && frame->tokens_per_sequence != 1u) ||
		(frame->active_slot_count >
		 SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT /
		 frame->tokens_per_sequence) ||
		(frame->tokens_per_sequence > 1u &&
		 (state->tp_degree <= 1u || state->pp_stage_count != 1u)) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->tp_degree > 1u &&
		!((frame->active_slot_count == SPARK_BATCH_BUCKET &&
		   frame->new_token_count == SPARK_BATCH_BUCKET) ||
		  (SPARK_BATCH_BUCKET == SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u &&
		   state->dspark_enabled != 0u &&
		   frame->active_slot_count == 1u &&
		   frame->new_token_count == 1u)) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*is_prefill_out = is_prefill;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleValidateFrameContext(
	const SparkDsv4ModuleState *state,
	const SparkModelDriverFrame *frame,
	uint32_t is_prefill,
	const SparkDsv4ResidentDecodeStageFrameContext **context_out)
{
	const uint32_t known_flags =
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW |
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_BUFFER |
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_BUFFER |
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW |
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_BATCH_VIEW;
	const SparkDsv4ResidentDecodeStageFrameContext *context;
	uint32_t decode_view,prefill_view,needs_input,needs_output;
	if ( frame->buffer_count != 1u + state->owns_final_head ||
		(frame->buffer_count != 0u && frame->buffers == 0) || frame->user_context == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkDsv4ResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ||
		context->descriptor_bytes < (uint32_t)sizeof(*context) ||
		context->reserved0 != 0u || context->submission_id == 0u ||
		context->control_generation == 0u ||
		context->transaction_id == 0u ||
		context->dispatch_generation == 0u ||
		context->request_generation == 0u ||
		context->step_generation == 0u ||
		(context->flags & ~known_flags) != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	decode_view = (context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) != 0u;
	prefill_view = (context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_BATCH_VIEW) != 0u;
	needs_input = state->pp_stage_index > 0u ? 1u : 0u;
	needs_output = state->pp_stage_index + 1u < state->pp_stage_count ? 1u : 0u;
	if ( prefill_view != is_prefill || decode_view == is_prefill ||
		((context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_BUFFER) != 0u) != (needs_input != 0u) ||
		((context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_BUFFER) != 0u) != (needs_output != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*context_out = context;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleValidateDecodeView(
	const SparkModelDriverFrame *frame,
	const SparkDsv4DecodeBatchView *batch)
{
	if ( batch == 0 ||
		batch->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION ||
		batch->descriptor_bytes < (uint32_t)sizeof(*batch) || batch->reserved0 != 0u ||
		batch->row_count != frame->active_slot_count ||
		batch->row_count != frame->new_token_count || batch->row_lane_indices == 0 ||
		batch->row_positions == 0 || batch->row_sequence_ids == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleValidatePrefillEmitShape(
	const SparkDsv4PrefillBatchView *prefill)
{
	uint8_t seen[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint32_t index,lane,previous,row;
	if ( prefill->emit_count > prefill->active_sequence_count || ((prefill->emit_count != 0u) != (prefill->emit_row_indices != 0)) || ((prefill->emit_count != 0u) != (prefill->emit_lane_indices != 0)) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	previous = UINT32_MAX;
	for (index=0u; index<prefill->emit_count; index++)
	{
		row = prefill->emit_row_indices[index];
		lane = prefill->emit_lane_indices[index];
		if ( row >= prefill->row_count || lane >= prefill->active_sequence_count || seen[lane] != 0u || (index != 0u && row <= previous) )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		seen[lane] = 1u;
		previous = row;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleValidatePrefillView(
	const SparkModelDriverFrame *frame,
	const SparkDsv4PrefillBatchView *prefill)
{
	if ( prefill == 0 ||
		prefill->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_PREFILL_BATCH_VIEW_ABI_VERSION ||
		prefill->descriptor_bytes < (uint32_t)sizeof(*prefill) || prefill->reserved0 != 0u ||
		prefill->row_count != frame->new_token_count ||
		prefill->active_sequence_count != frame->active_slot_count ||
		prefill->row_count < prefill->active_sequence_count ||
		prefill->token_ids == 0 || prefill->row_lane_indices == 0 ||
		prefill->row_positions == 0 || prefill->row_sequence_ids == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkDsv4ModuleValidatePrefillEmitShape(prefill));
}

static SparkStatus SparkDsv4ModuleValidateBoundaryBuffers(
	const SparkDsv4ModuleState *state,
	const SparkDsv4ResidentDecodeStageFrameContext *context,
	uint32_t row_count)
{
	uint32_t needs_input,needs_output;
	uint64_t bytes;
	needs_input = state->pp_stage_index > 0u ? 1u : 0u;
	needs_output = state->pp_stage_index + 1u < state->pp_stage_count ? 1u : 0u;
	bytes = (uint64_t)row_count * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	if ( (needs_input != 0u && (context->hidden_input_bf16 == 0 || context->hidden_input_bytes < bytes)) ||
		(needs_input == 0u && (context->hidden_input_bf16 != 0 || context->hidden_input_bytes != 0u)) ||
		(needs_output != 0u && (context->hidden_output_bf16 == 0 || context->hidden_output_bytes < bytes)) ||
		(needs_output == 0u && (context->hidden_output_bf16 != 0 || context->hidden_output_bytes != 0u)) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleValidateTokenBuffers(
	const SparkDsv4ModuleState *state,
	const SparkModelDriverFrame *frame,
	const SparkDsv4PrefillBatchView *prefill,
	uint32_t is_prefill,
	uint32_t row_count)
{
	uint32_t output_index;
	uint64_t bytes;
	SparkStatus status;
	bytes = (uint64_t)row_count * sizeof(uint32_t);
	status = SparkModelDriverValidateBuffer(frame,0u,0u,SPARK_MODEL_DRIVER_BUFFER_FLAG_READ,bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( is_prefill != 0u && frame->buffers[0].address != prefill->token_ids )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->owns_final_head == 0u )
		return(SPARK_STATUS_OK);
	output_index = 1u;
	bytes = (uint64_t)frame->active_slot_count *
		frame->tokens_per_sequence * sizeof(uint32_t);
	return(SparkModelDriverValidateBuffer(frame,output_index,1u,SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE,bytes));
}

static SparkStatus SparkDsv4ModuleValidateFrame(
	SparkDsv4ModuleState *state,
	const SparkModelDriverFrame *frame,
	const SparkDsv4ResidentDecodeStageFrameContext **context_out)
{
	const SparkDsv4ResidentDecodeStageFrameContext *context;
	uint32_t is_prefill,row_count;
	SparkStatus status;
	status = SparkDsv4ModuleValidateFrameShape(state,frame,&is_prefill);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleValidateFrameContext(state,frame,is_prefill,&context);
	if ( status == SPARK_STATUS_OK && is_prefill != 0u )
		status = SparkDsv4ModuleValidatePrefillView(frame,context->prefill_batch);
	if ( status == SPARK_STATUS_OK && is_prefill == 0u )
		status = SparkDsv4ModuleValidateDecodeView(frame,context->decode_batch);
	if ( status != SPARK_STATUS_OK )
		return(status);
	row_count = is_prefill != 0u ? context->prefill_batch->row_count : context->decode_batch->row_count;
	status = SparkDsv4ModuleValidateBoundaryBuffers(state,context,row_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleValidateTokenBuffers(state,frame,context->prefill_batch,is_prefill,row_count);
	if ( status == SPARK_STATUS_OK )
		*context_out = context;
	return(status);
}

static SparkStatus SparkDsv4ModuleCollectFrameLaneIndices(
	const SparkDsv4ModuleState *state,
	const SparkModelDriverFrame *frame,
	const SparkDsv4ResidentDecodeStageFrameContext *context,
	uint32_t *lane_indices)
{
	const uint32_t *row_lanes;
	uint32_t lane,lane_count,ordinal;
	if ( state == 0 || frame == 0 || context == 0 || lane_indices == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	row_lanes = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? context->prefill_batch->row_lane_indices : context->decode_batch->row_lane_indices;
	lane_count = frame->active_slot_count;
	for (ordinal=0u; ordinal<lane_count; ordinal++)
	{
		lane = row_lanes[ordinal];
		if ( lane >= state->resident_sequence_capacity )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		lane_indices[ordinal] = lane;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleClaimedLaneOrdinal(
	void *context,
	uint32_t lane_id,
	uint32_t *ordinal_out)
{
	SparkDsv4ModuleState *state;
	state = (SparkDsv4ModuleState *)context;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageModuleIndexClaimOrdinal(state->lane_states,state->resident_sequence_capacity,lane_id,ordinal_out));
}

static SparkStatus SparkDsv4ModuleValidateClaimedPrefillEmitRows(
	const SparkDsv4PrefillBatchView *prefill,
	const uint32_t *last_rows)
{
	uint32_t index,lane;
	if ( prefill == 0 || last_rows == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<prefill->emit_count; index++)
	{
		lane = prefill->emit_lane_indices[index];
		if ( lane >= prefill->active_sequence_count || prefill->emit_row_indices[index] != last_rows[lane] )
			return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleValidateFrameContinuity(
	SparkDsv4ModuleState *state,
	const SparkModelDriverFrame *frame,
	const SparkDsv4ResidentDecodeStageFrameContext *context,
	const uint32_t *lane_indices,
	uint64_t *lane_sequence_ids,
	uint64_t *lane_next_positions,
	uint8_t *lane_requires_reset)
{
	const uint32_t *row_lanes;
	const uint64_t *row_positions,*row_sequences;
	uint8_t touched[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint32_t occurrences[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t last_rows[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane,lane_count,ordinal,row,row_count;
	uint64_t position,sequence;
	SparkStatus status;
	if ( state == 0 || frame == 0 || context == 0 || lane_indices == 0 || lane_sequence_ids == 0 || lane_next_positions == 0 || lane_requires_reset == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u )
	{
		row_count = context->prefill_batch->row_count;
		lane_count = context->prefill_batch->active_sequence_count;
		row_lanes = context->prefill_batch->row_lane_indices;
		row_positions = context->prefill_batch->row_positions;
		row_sequences = context->prefill_batch->row_sequence_ids;
		status = SparkRowLayoutValidateRoundMajor(row_count,lane_count,row_lanes,SparkDsv4ModuleClaimedLaneOrdinal,state,occurrences,last_rows);
		if ( status != SPARK_STATUS_OK )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		status = SparkDsv4ModuleValidateClaimedPrefillEmitRows(context->prefill_batch,last_rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	else
	{
		row_count = context->decode_batch->row_count;
		lane_count = context->decode_batch->row_count;
		row_lanes = context->decode_batch->row_lane_indices;
		row_positions = context->decode_batch->row_positions;
		row_sequences = context->decode_batch->row_sequence_ids;
	}
	for (ordinal=0u; ordinal<lane_count; ordinal++)
	{
		lane = lane_indices[ordinal];
		lane_sequence_ids[ordinal] = state->lane_sequence_ids[lane];
		lane_next_positions[ordinal] = state->lane_next_positions[lane];
		lane_requires_reset[ordinal] = 0u;
		if ( frame->cache_lane_count != 0u &&
			(frame->cache_lanes[ordinal].flags &
			 SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) != 0u )
		{
			if ( frame->cache_lanes[ordinal].resident_sequence_slot != lane ||
				frame->cache_lanes[ordinal].sequence_id == 0u ||
				frame->cache_lanes[ordinal].sequence_position !=
					frame->cache_lanes[ordinal].prefix_token_count )
				return(SPARK_STATUS_INVALID_ARGUMENT);
			lane_sequence_ids[ordinal] =
				frame->cache_lanes[ordinal].sequence_id;
			lane_next_positions[ordinal] =
				frame->cache_lanes[ordinal].sequence_position;
		}
	}
	for (row=0u; row<row_count; row++)
	{
		lane = row_lanes[row];
		status = SparkStageModuleIndexClaimOrdinal(state->lane_states,state->resident_sequence_capacity,lane,&ordinal);
		sequence = row_sequences[row];
		position = row_positions[row];
		if ( status != SPARK_STATUS_OK || ordinal >= lane_count || lane_indices[ordinal] != lane || sequence == 0u || position >= state->max_sequence_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( SparkDsv4AdvanceLaneContinuity(sequence,position,&lane_sequence_ids[ordinal],&lane_next_positions[ordinal],&touched[ordinal],&lane_requires_reset[ordinal]) != SPARK_STATUS_OK )
			return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

// The host half of staging: validate and refill the slot's pinned buffers.
static SparkStatus SparkDsv4ModuleStageRowValues(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	const uint32_t *token_ids,
	const uint32_t *row_lane_indices,
	const uint64_t *row_positions,
	uint32_t row_count)
{
	uint32_t row,lane,lane_ordinal,page,physical_page;
	SparkStatus status;
	if ( row_count == 0u || row_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT || token_ids == 0 || row_lane_indices == 0 || row_positions == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (row = 0; row < row_count; row++)
	{
		lane = row_lane_indices[row];
		if ( lane >= state->resident_sequence_capacity || row_positions[row] >= state->max_sequence_positions || token_ids[row] >= SPARK_DSV4_MODEL_VOCAB_COUNT )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		status = SparkStageModuleIndexClaimOrdinal(state->lane_states,
			state->resident_sequence_capacity,lane,&lane_ordinal);
		page = (uint32_t)(row_positions[row] /
			SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS);
		if ( status != SPARK_STATUS_OK ||
			lane_ordinal >= SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT ||
			page >= state->paged_cache.lane_page_capacity )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		physical_page = slot->host_physical_page_table[
			(uint64_t)lane_ordinal * state->paged_cache.lane_page_capacity + page];
		if ( physical_page >= state->physical_page_capacity )
			return(SPARK_STATUS_BUSY);
		slot->host_input_token_ids[row] = token_ids[row];
		slot->host_row_lane_indices[row] = physical_page;
		slot->host_row_page_table_indices[row] = lane;
		slot->host_row_positions[row] = row_positions[row];
		slot->host_row_emit_positions[row] = row_positions[row] + 1u >= SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO ? row_positions[row] + 1u - SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO : 0u;
		slot->host_row_emit_positions_hca[row] = row_positions[row] + 1u >= SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO ? row_positions[row] + 1u - SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO : 0u;
	}
	return(SPARK_STATUS_OK);
}

// The H2D half of staging, split from the host fill so a graph capture can
// record exactly these two copies: the pinned source buffers are per-slot
// fixed, so a replay re-reads whatever the current frame's fill left there.
static SparkStatus SparkDsv4ModuleStageRowCopies(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t row_count,
	uint32_t lane_count)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t pitch32 = (uint64_t)SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT * sizeof(uint32_t);
	uint64_t pitch64 = (uint64_t)SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT * sizeof(uint64_t);
	uint64_t initialize_pitch;
	cudaError_t error;
	if ( state == 0 || row_count == 0u ||
		row_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT ||
		lane_count == 0u ||
		lane_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	initialize_pitch = (uint64_t)SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * sizeof(uint32_t);
	error = cudaMemcpy2DAsync(slot->input_token_ids,pitch32,slot->host_input_token_ids,pitch32,(uint64_t)row_count * sizeof(uint32_t),3u,cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync(slot->row_positions,pitch64,slot->host_row_positions,pitch64,(uint64_t)row_count * sizeof(uint64_t),3u,cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync(slot->initialize_page_indices,initialize_pitch,
			slot->host_initialize_page_indices,initialize_pitch,
			(uint64_t)lane_count * sizeof(uint32_t),2u,
			cudaMemcpyHostToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"stage_row_copies"));
}

static SparkStatus SparkDsv4ModuleInitializeFramePages(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t lane_count)
{
	cudaError_t error;
	if ( state == 0 || slot == 0 || lane_count == 0u ||
		lane_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	error = SparkDsv4LaunchInitializePages((cudaStream_t)slot->cuda_stream,
		state->paged_cache.device_page_pool,
		state->paged_cache.layout.page_stride_bytes,
		slot->initialize_page_indices,
		slot->initialize_parent_page_indices,lane_count,
		state->device_page_score_spans,state->page_score_span_count);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"initialize_frame_pages"));
}

static SparkStatus SparkDsv4ModuleStageRows(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	const uint32_t *token_ids,
	const uint32_t *row_lane_indices,
	const uint64_t *row_positions,
	uint32_t row_count,
	uint32_t lane_count)
{
	SparkStatus status;
	status = SparkDsv4ModuleStageRowValues(state,slot,token_ids,row_lane_indices,row_positions,row_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleStageRowCopies(state,slot,row_count,lane_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleInitializeFramePages(state,slot,lane_count);
	return(status);
}

static SparkStatus SparkDsv4ModuleDsparkDrive(SparkDsv4ModuleState *state,SparkDsv4ModuleSlot *slot,uint32_t lane_index,uint32_t anchor_token_id,uint64_t anchor_position);

/* DSpark verify expansion: stage 8 rows of the single lane (anchor +
 * SPEC_STEP drafts) through the slot's pinned staging, so the B8 islands
 * replay one batched verify frame. Runs after the draft on the submission
 * path; the draft tokens arrive device-side in dspark_draft_token_ids. */
/* Copy the staged row arrays (3 x u32 + 3 x u64, the exact layout
 * StageRowCopies records) for the verify expansion's rows. */
static cudaError_t SparkDsv4ModuleStageVerifyRowCopies(
	SparkDsv4ModuleSlot *slot,uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t pitch32 = (uint64_t)SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT * sizeof(uint32_t);
	uint64_t pitch64 = (uint64_t)SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT * sizeof(uint64_t);
	cudaError_t error;
	error = cudaMemcpy2DAsync(slot->input_token_ids,pitch32,
		slot->host_input_token_ids,pitch32,(uint64_t)rows * sizeof(uint32_t),
		3u,cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync(slot->row_positions,pitch64,
			slot->host_row_positions,pitch64,(uint64_t)rows * sizeof(uint64_t),
			3u,cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync(slot->initialize_page_indices,
			(uint64_t)SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * sizeof(uint32_t),
			slot->host_initialize_page_indices,
			(uint64_t)SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * sizeof(uint32_t),
			sizeof(uint32_t),2u,cudaMemcpyHostToDevice,stream);
	return(error);
}

/* DSpark verify expansion: stage 8 rows of the single lane (anchor +
 * SPEC_STEP drafts) through the slot's pinned staging, so the B8 islands
 * replay one batched verify frame. Runs after the draft on the submission
 * path; the draft tokens arrive device-side in dspark_draft_token_ids.
 * Each row's physical page resolves from the prepared page table exactly
 * like SparkDsv4ModuleStageRowValues (row_lane_indices = physical page). */
static SparkStatus SparkDsv4ModuleExpandDsparkVerify(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	const SparkModelDriverFrame *frame,
	uint32_t lane_index,
	uint32_t anchor_token_id,
	uint64_t anchor_position)
{
	const uint32_t rows = SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u;
	uint32_t host_tokens[SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u];
	uint64_t host_positions[SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u];
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	uint32_t row,page,physical_page;
	SparkStatus status;
	(void)frame;
	host_tokens[0] = anchor_token_id;
	error = cudaMemcpyAsync(host_tokens + 1u,slot->dspark_draft_token_ids,
		SPARK_DSV4_MODEL_DSPARK_SPEC_STEP * sizeof(uint32_t),
		cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess )
		error = cudaStreamSynchronize(stream);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
			"dspark_verify_tokens"));
	for (row = 0u; row < SPARK_DSV4_MODEL_DSPARK_SPEC_STEP; row++)
		slot->dspark_host_draft_tokens[row] = host_tokens[1u + row];
	for (row = 0u; row < rows; row++)
	{
		host_positions[row] = anchor_position + 1u + row;
		if ( host_positions[row] >= state->max_sequence_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		page = (uint32_t)(host_positions[row] /
			SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS);
		if ( page >= state->paged_cache.lane_page_capacity )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		physical_page = slot->host_physical_page_table[page];
		if ( physical_page >= state->physical_page_capacity )
			return(SPARK_STATUS_BUSY);
		slot->host_input_token_ids[row] = host_tokens[row];
		slot->host_row_lane_indices[row] = physical_page;
		slot->host_row_page_table_indices[row] = lane_index;
		slot->host_row_positions[row] = host_positions[row];
		slot->host_row_emit_positions[row] = host_positions[row] + 1u >= SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO ? host_positions[row] + 1u - SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO : 0u;
		slot->host_row_emit_positions_hca[row] = host_positions[row] + 1u >= SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO ? host_positions[row] + 1u - SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO : 0u;
	}
	error = SparkDsv4ModuleStageVerifyRowCopies(slot,rows);
	if ( error == cudaSuccess )
		status = SparkDsv4ModuleInitializeFramePages(state,slot,1u);
	else
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
			"dspark_verify_expand");
	/* Compressed-state rollback saves: the CSA previous windows (kv + score)
	 * and every compressor layer's boundary emission slots, so a rejected
	 * boundary row can be undone at the completion. */
	if ( status == SPARK_STATUS_OK )
	{
		const uint64_t channels = 2u * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
		const uint64_t window_floats = (uint64_t)SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO * channels;
		const uint64_t slot_bytes = SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
		uint32_t boundary,layer,csa_layer,ordinal,kind;
		slot->dspark_boundary_count = 0u;
		slot->dspark_hca_boundary_row = UINT32_MAX;
		for (row = 0u; row < rows; row++)
			if ( (host_positions[row] + 1u) % SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO == 0u )
			{
				slot->dspark_boundary_rows[slot->dspark_boundary_count] = row;
				slot->dspark_boundary_count++;
			}
		for (row = 0u; row < rows; row++)
			if ( (host_positions[row] + 1u) % SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO == 0u )
			{
				slot->dspark_hca_boundary_row = row;
				break;
			}
		csa_layer = 0u;
		ordinal = 0u;
		for (layer = state->first_layer_index;
			layer < state->first_layer_index + state->layer_count &&
			error == cudaSuccess; layer++)
		{
			kind = SparkDsv4ModelLayerKind(layer);
			if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA )
				continue;
			if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
			{
				const void *kv_source = (const uint8_t *)state->compress_kv_state_f32 +
					state->compress_state_offset_by_layer[layer] * sizeof(float);
				const void *score_source = (const uint8_t *)state->compress_score_state_f32 +
					state->compress_score_state_offset_by_layer[layer] * sizeof(float);
				error = cudaMemcpyAsync(
					slot->dspark_csa_previous_save + (uint64_t)csa_layer * 2u * window_floats * sizeof(float),
					kv_source,window_floats * sizeof(float),cudaMemcpyDeviceToDevice,stream);
				if ( error == cudaSuccess )
					error = cudaMemcpyAsync(
						slot->dspark_csa_previous_save + ((uint64_t)csa_layer * 2u + 1u) * window_floats * sizeof(float),
						score_source,window_floats * sizeof(float),cudaMemcpyDeviceToDevice,stream);
				csa_layer++;
			}
			for (boundary = 0u; boundary < slot->dspark_boundary_count &&
				error == cudaSuccess; boundary++)
			{
				row = slot->dspark_boundary_rows[boundary];
				uint64_t position = host_positions[row];
				uint64_t slot64 = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS +
					((position % SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS) /
					 (kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ?
					  SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO :
					  SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO));
				const void *source = (const uint8_t *)state->kv_cache_bf16 +
					(uint64_t)state->cache_offset_by_layer[layer] * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES +
					(uint64_t)slot->host_row_lane_indices[row] *
						state->cache_lane_stride_by_layer[layer] * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES +
					slot64 * slot_bytes;
				error = cudaMemcpyAsync(
					slot->dspark_emission_save + ((uint64_t)ordinal * 2u + boundary) * slot_bytes,
					source,slot_bytes,cudaMemcpyDeviceToDevice,stream);
			}
			if ( error == cudaSuccess && kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA &&
				slot->dspark_hca_boundary_row != UINT32_MAX )
			{
				row = slot->dspark_hca_boundary_row;
				uint64_t position = host_positions[row];
				uint64_t slot64 = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS +
					((position % SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS) /
					 SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO);
				const void *source = (const uint8_t *)state->kv_cache_bf16 +
					(uint64_t)state->cache_offset_by_layer[layer] * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES +
					(uint64_t)slot->host_row_lane_indices[row] *
						state->cache_lane_stride_by_layer[layer] * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES +
					slot64 * slot_bytes;
				error = cudaMemcpyAsync(
					slot->dspark_emission_save + (uint64_t)ordinal * 2u * slot_bytes,
					source,slot_bytes,cudaMemcpyDeviceToDevice,stream);
			}
			ordinal++;
		}
		if ( error != cudaSuccess )
			status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
				"dspark_rollback_save");
	}
	if ( status == SPARK_STATUS_OK )
	{
		slot->dspark_verify_rows = rows;
		slot->dspark_verify_accept = 1u;
		{
			uint32_t engine_token = 0u;
			cudaError_t probe_error = cudaMemcpyAsync(&engine_token,
				frame->buffers[0].address,sizeof(uint32_t),
				cudaMemcpyDeviceToHost,stream);
			if ( probe_error == cudaSuccess )
				probe_error = cudaStreamSynchronize(stream);
			if ( probe_error == cudaSuccess )
				fprintf(stderr,"dspark_expand tp_rank=%u engine_tok=%u anchor_pos=%llu row0_tok=%u row0_pos=%llu row1_tok=%u row1_pos=%llu\n",
					state->tp_rank,engine_token,(unsigned long long)anchor_position,
					slot->host_input_token_ids[0],(unsigned long long)slot->host_row_positions[0],
					slot->host_input_token_ids[1],(unsigned long long)slot->host_row_positions[1]);
		}
		/* DEBUG: device-side staging probe - what the islands will read. */
		{
			uint32_t dev_tokens[SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u];
			uint64_t dev_positions[SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u];
			uint32_t dev_lanes[SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u];
			uint32_t dev_pages[SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u];
			cudaError_t probe_error = cudaMemcpyAsync(dev_tokens,
				slot->input_token_ids,(uint64_t)rows * sizeof(uint32_t),
				cudaMemcpyDeviceToHost,stream);
			if ( probe_error == cudaSuccess )
				probe_error = cudaMemcpyAsync(dev_positions,
					slot->row_positions,(uint64_t)rows * sizeof(uint64_t),
					cudaMemcpyDeviceToHost,stream);
			if ( probe_error == cudaSuccess )
				probe_error = cudaMemcpyAsync(dev_lanes,
					slot->row_lane_indices,(uint64_t)rows * sizeof(uint32_t),
					cudaMemcpyDeviceToHost,stream);
			if ( probe_error == cudaSuccess )
				probe_error = cudaMemcpyAsync(dev_pages,
					slot->row_page_table_indices,(uint64_t)rows * sizeof(uint32_t),
					cudaMemcpyDeviceToHost,stream);
			if ( probe_error == cudaSuccess )
				probe_error = cudaStreamSynchronize(stream);
			if ( probe_error == cudaSuccess )
				fprintf(stderr,"dspark_devstage tp_rank=%u tok=%u,%u,%u,%u,%u,%u,%u,%u pos=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu lanes=%u,%u,%u,%u,%u,%u,%u,%u pages=%u,%u,%u,%u,%u,%u,%u,%u\n",
					state->tp_rank,dev_tokens[0],dev_tokens[1],dev_tokens[2],dev_tokens[3],dev_tokens[4],dev_tokens[5],dev_tokens[6],dev_tokens[7],
					(unsigned long long)dev_positions[0],(unsigned long long)dev_positions[1],(unsigned long long)dev_positions[2],(unsigned long long)dev_positions[3],(unsigned long long)dev_positions[4],(unsigned long long)dev_positions[5],(unsigned long long)dev_positions[6],(unsigned long long)dev_positions[7],
					dev_lanes[0],dev_lanes[1],dev_lanes[2],dev_lanes[3],dev_lanes[4],dev_lanes[5],dev_lanes[6],dev_lanes[7],
					dev_pages[0],dev_pages[1],dev_pages[2],dev_pages[3],dev_pages[4],dev_pages[5],dev_pages[6],dev_pages[7]);
		}
	}
	return(status);
}

/* Pad a staged single row up to BUCKET rows by duplicating row 0 at the
 * same position (the islands only replay bucket-width shapes; identical
 * rows write identical KV and produce identical outputs, so only row 0's
 * token is emitted). Used for 1-row prefill frames and the draft-failed
 * decode fallback on the B8 bucket. */
static SparkStatus SparkDsv4ModulePadDuplicateRows(
	SparkDsv4ModuleSlot *slot,
	uint32_t rows)
{
	uint32_t row;
	SparkStatus status;
	for (row = 1u; row < rows; row++)
	{
		slot->host_input_token_ids[row] = slot->host_input_token_ids[0];
		slot->host_row_lane_indices[row] = slot->host_row_lane_indices[0];
		slot->host_row_page_table_indices[row] = slot->host_row_page_table_indices[0];
		slot->host_row_positions[row] = slot->host_row_positions[0];
		slot->host_row_emit_positions[row] = slot->host_row_emit_positions[0];
		slot->host_row_emit_positions_hca[row] = slot->host_row_emit_positions_hca[0];
	}
	status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,
		SparkDsv4ModuleStageVerifyRowCopies(slot,rows),"dspark_pad_rows");
	if ( status == SPARK_STATUS_OK )
	{
		slot->dspark_verify_rows = rows;
		slot->dspark_verify_accept = 0u;
	}
	return(status);
}

static SparkStatus SparkDsv4ModuleStageFrameRows(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkModelDriverFrame *frame, const SparkDsv4ResidentDecodeStageFrameContext *context)
{
	const uint32_t *token_ids,*row_lane_indices;
	const uint64_t *row_positions;
	uint32_t prefill,row_count;
	token_ids = (const uint32_t *)frame->buffers[0].address;
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u;
	row_lane_indices = prefill != 0u ? context->prefill_batch->row_lane_indices : context->decode_batch->row_lane_indices;
	row_positions = prefill != 0u ? context->prefill_batch->row_positions : context->decode_batch->row_positions;
	row_count = prefill != 0u ? context->prefill_batch->row_count : context->decode_batch->row_count;
	/* DSpark: if the previous step published a ready lane, run the draft
	 * from the LANE store on the submission path (host syncs are legal
	 * here, unlike the completion callback). */
	fprintf(stderr,"dspark_staging tp_rank=%u prefill=%u enabled=%u rows=%u\n",state->tp_rank,prefill,state->dspark_enabled,row_count);
#if SPARK_BATCH_BUCKET == SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u
	if ( state->dspark_enabled != 0u && row_count == 1u )
	{
		uint32_t lane_index = 0u;
		cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
		cudaError_t error = cudaMemcpyAsync(&lane_index,row_lane_indices,
			sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
		if ( error == cudaSuccess )
			error = cudaStreamSynchronize(stream);
		if ( error == cudaSuccess && lane_index < state->resident_sequence_capacity )
		{
			if ( prefill == 0u && state->dspark_lane_ready[lane_index] != 0u )
			{
				uint64_t tap_bytes = (uint64_t)SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT *
					SPARK_DSV4_MODEL_HIDDEN_DIMENSION *
					SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
				state->dspark_lane_ready[lane_index] = 0u;
				error = cudaMemcpyAsync(slot->dspark_tap_bf16,
					(uint8_t *)state->dspark_tap_store_bf16 + (uint64_t)lane_index * tap_bytes,
					tap_bytes,cudaMemcpyDeviceToDevice,stream);
				if ( error == cudaSuccess )
					error = cudaStreamSynchronize(stream);
				if ( error == cudaSuccess )
				{
					SparkStatus dspark_status = SparkDsv4ModuleDsparkDrive(state,slot,
						lane_index,state->dspark_lane_anchor[lane_index],
						state->dspark_lane_position[lane_index]);
					if ( dspark_status != SPARK_STATUS_OK )
						fprintf(stderr,"dspark_drive_failed status=%u tp_rank=%u lane=%u\n",(uint32_t)dspark_status,state->tp_rank,lane_index);
					else
					{
						/* k=7 verify expansion: stage SPEC_STEP+1 rows of the lane
						 * (anchor + drafts) instead of the 1-row submission */
						SparkStatus expand_status = SparkDsv4ModuleExpandDsparkVerify(
						state,slot,frame,lane_index,
						state->dspark_lane_anchor[lane_index],
						state->dspark_lane_position[lane_index]);
					if ( expand_status != SPARK_STATUS_OK )
						fprintf(stderr,"dspark_verify_expand_failed status=%u tp_rank=%u\n",(uint32_t)expand_status,state->tp_rank);
					else
						return(SPARK_STATUS_OK);
					}
				}
			}
			/* Prefill single-lane frames and the decode fallback (lane not
			 * armed / draft failed): stage the one real row, then pad
			 * duplicates to bucket width so the islands can replay. */
			if ( prefill != 0u || slot->dspark_verify_rows == 0u )
			{
				SparkStatus stage_status = SparkDsv4ModuleStageRows(state,slot,
					token_ids,row_lane_indices,row_positions,1u,
					frame->active_slot_count);
				if ( stage_status == SPARK_STATUS_OK )
					stage_status = SparkDsv4ModulePadDuplicateRows(slot,
						SPARK_BATCH_BUCKET);
				if ( stage_status == SPARK_STATUS_OK )
					return(SPARK_STATUS_OK);
				fprintf(stderr,"dspark_pad_failed status=%u tp_rank=%u prefill=%u\n",(uint32_t)stage_status,state->tp_rank,prefill);
			}
		}
	}
#endif
	return(SparkDsv4ModuleStageRows(state,slot,token_ids,row_lane_indices,row_positions,row_count,frame->active_slot_count));
}

static SparkStatus SparkDsv4ModuleBeginStreams(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4ResidentDecodeStageFrameContext *context, uint32_t rows, const void **streams_out)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t stream_bytes = (uint64_t)rows * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	cudaError_t error;
	uint32_t copy;
	if ( streams_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*streams_out = 0;
	if ( state->owns_embedding != 0u )
	{
		error = SparkDsv4LaunchEmbeddingGather(stream,slot->input_token_ids,state->token_embedding_bf16,slot->reduced_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
		for (copy = 0; error == cudaSuccess && copy < SPARK_DSV4_MODEL_HC_STREAM_COUNT; copy++)
			error = cudaMemcpy2DAsync((uint8_t *)slot->streams_bf16 + (uint64_t)copy * SPARK_DSV4_MODEL_HIDDEN_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,(uint64_t)SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,slot->reduced_bf16,(uint64_t)SPARK_DSV4_MODEL_HIDDEN_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,(uint64_t)SPARK_DSV4_MODEL_HIDDEN_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,rows,cudaMemcpyDeviceToDevice,stream);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"embedding_streams"));
		*streams_out = slot->streams_bf16;
		return(SPARK_STATUS_OK);
	}
	if ( context == 0 || context->hidden_input_bf16 == 0 || context->hidden_input_bytes < stream_bytes )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*streams_out = context->hidden_input_bf16;
	return(SPARK_STATUS_OK);
}

// One mHC boundary: preserve the residual, mix and split with Sinkhorn, then reduce.
static cudaError_t SparkDsv4ModuleHcEnter(SparkDsv4ModuleSlot *slot, const void *streams_bf16, const float *fn, const float *scale3, const float *base, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	if ( streams_bf16 == 0 )
		return(cudaErrorInvalidValue);
	error = SparkDsv4LaunchHcMixSplitKSinkhorn(stream,streams_bf16,
		fn,scale3,base,slot->hc_partials_f32,
		slot->mixes_f32,rows,SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS,
		SPARK_DSV4_MODEL_HC_MIX_ROWS,SPARK_DSV4_MODEL_HC_STREAM_COUNT,
		SPARK_DSV4_MODEL_HC_SINKHORN_ITERATIONS,
		SPARK_DSV4_MODEL_RMS_NORM_EPSILON,SPARK_DSV4_MODEL_HC_EPSILON,
		slot->pre_f32,slot->post_f32,slot->comb_f32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcPreReduce(stream,streams_bf16,slot->pre_f32,
			slot->reduced_bf16,slot->residual_bf16,rows,
			SPARK_DSV4_MODEL_HC_STREAM_COUNT,
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	return(error);
}

/*
 * The compressor for one decode token, attention side: wkv/wgate on the
 * normalized x, widen to fp32, ape by in-group position, the state step,
 * and for boundary rows the emitted slot gets norm, rope at the group
 * start position, the fp8 cache sim, and lands at position/ratio behind
 * the window. The fused device epilogue owns the boundary predicate.
 */
static cudaError_t SparkDsv4ModuleRunCompressorProjection(
	SparkDsv4ModuleSlot *slot,SparkDsv4CompressorScratch *scratch,
	cudaStream_t stream,const SparkDsv4CompressorWeights *weights,uint32_t rows)
{
	return(SparkDsv4LaunchBf16LinearPair(stream,&weights->wkv,
		&weights->wgate,slot->normalized_bf16,scratch->kv_bf16,
		scratch->score_bf16,rows));
}

static cudaError_t SparkDsv4ModuleRunCompressorPost(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, SparkDsv4CompressorScratch *scratch, cudaStream_t stream, const SparkDsv4CompressorWeights *weights, float *kv_state, float *score_state, uint64_t state_stride, void *cache_base, uint64_t cache_lane_stride, uint64_t cache_slot_offset, uint32_t cache_width, uint32_t rotate, uint32_t rows)
{
	uint32_t overlapped;
	uint64_t ratio = weights->ratio;
	cudaError_t error;
	if ( weights->overlap != 1u && weights->overlap != SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR )
		return(cudaErrorInvalidValue);
	overlapped = weights->overlap > 1u ? 1u : 0u;
	error = SparkDsv4LaunchCompressStep(stream,scratch->kv_bf16,
		scratch->score_bf16,weights->ape_f32,kv_state,score_state,state_stride,
		slot->row_lane_indices,slot->row_positions,rows,(uint32_t)ratio,
		overlapped,cache_width,scratch->emit_bf16,scratch->emitted_u32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchKvEmission(stream,scratch->emit_bf16,
			scratch->emitted_u32,weights->norm_weight_bf16,
			state->compress_freqs_f32,
			weights->ratio == SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO ?
				slot->row_emit_positions_hca : slot->row_emit_positions,
			cache_base,cache_lane_stride,slot->row_lane_indices,
			slot->row_positions,rows,cache_width,cache_slot_offset,
			(uint32_t)ratio,SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS,rotate);
	return(error);
}

static cudaError_t SparkDsv4ModuleRunCompressor(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, SparkDsv4CompressorScratch *scratch, cudaStream_t stream, const SparkDsv4CompressorWeights *weights, float *kv_state, float *score_state, uint64_t state_stride, void *cache_base, uint64_t cache_lane_stride, uint64_t cache_slot_offset, uint32_t cache_width, uint32_t rotate, uint32_t rows)
{
	cudaError_t error = SparkDsv4ModuleRunCompressorProjection(slot,scratch,
		stream,weights,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunCompressorPost(state,slot,scratch,stream,weights,
			kv_state,score_state,state_stride,cache_base,cache_lane_stride,
			cache_slot_offset,cache_width,rotate,rows);
	return(error);
}

static cudaError_t SparkDsv4ModuleRunIndexerWeights(
	SparkDsv4ModuleSlot *slot,cudaStream_t stream,
	const SparkDsv4LayerWeights *layer,uint32_t rows)
{
	float scale = 1.0f / sqrtf((float)SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION) /
		sqrtf((float)SPARK_DSV4_MODEL_INDEX_HEAD_COUNT);
	cudaError_t error;
	error = SparkDsv4LaunchLinear(stream,&layer->indexer.weights_proj,
		slot->normalized_bf16,slot->index_weights_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchWiden(stream,slot->index_weights_bf16,
			slot->index_weights_f32,rows,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,scale);
	return(error);
}

static cudaError_t SparkDsv4ModuleRunIndexerCore(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, cudaStream_t stream, const SparkDsv4LayerWeights *layer, uint32_t layer_index, uint32_t projected, uint32_t weights_projected, uint32_t rows)
{
	void *index_cache = (uint8_t *)state->index_cache_bf16 +
		state->index_cache_offset_by_layer[layer_index] *
		SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	float *kv_state = state->index_kv_state_f32 +
		state->index_kv_state_offset_by_layer[layer_index];
	float *score_state = state->index_score_state_f32 +
		state->index_score_state_offset_by_layer[layer_index];
	cudaError_t error;
	error = SparkDsv4LaunchLinear(stream,&layer->indexer.wq_b,slot->qr_bf16,
		slot->index_q_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchIndexerPost(stream,slot->index_q_bf16,
		state->compress_freqs_f32,slot->row_positions,rows,
		SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,
		SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,
		SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,
		SPARK_DSV4_MODEL_FP4_QUANT_BLOCK);
	if ( error == cudaSuccess && weights_projected == 0u )
		error = SparkDsv4ModuleRunIndexerWeights(slot,stream,layer,rows);
	if ( error == cudaSuccess && projected == 0u )
		error = SparkDsv4ModuleRunCompressor(state,slot,
			&slot->index_compressor,stream,&layer->indexer.compressor,kv_state,
			score_state,state->index_state_lane_stride,index_cache,
			state->index_lane_stride,0u,
			SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,1u,rows);
	if ( error == cudaSuccess && projected != 0u )
		error = SparkDsv4ModuleRunCompressorPost(state,slot,
			&slot->index_compressor,stream,&layer->indexer.compressor,kv_state,
			score_state,state->index_state_lane_stride,index_cache,
			state->index_lane_stride,0u,
			SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,1u,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchIndexerScore(stream,slot->index_q_bf16,index_cache,state->index_lane_stride,slot->row_page_table_indices,slot->physical_page_table,state->paged_cache.lane_page_capacity,SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO,slot->slot_counts,slot->index_weights_f32,slot->index_scores_f32,rows,state->index_slot_capacity,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchTopK(stream,slot->index_scores_f32,slot->slot_counts,state->index_slot_capacity,SPARK_DSV4_MODEL_INDEX_TOP_K,(int32_t)SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,slot->topk_idxs + SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,state->topk_column_count,rows);
	return(error);
}

static cudaError_t SparkDsv4ModuleRunIndexer(SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,cudaStream_t stream,
	const SparkDsv4LayerWeights *layer,uint32_t layer_index,uint32_t rows)
{
	return(SparkDsv4ModuleRunIndexerCore(state,slot,stream,layer,layer_index,0u,
		0u,rows));
}

static cudaError_t SparkDsv4ModuleStageTopk(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, cudaStream_t stream, uint32_t layer_kind, uint32_t rows)
{
	return(SparkDsv4LaunchBuildAttentionIndices(stream,slot->row_positions,slot->topk_idxs,slot->slot_counts,slot->attention_slot_counts,rows,state->topk_column_count,state->index_slot_capacity,layer_kind));
}

/*
 * All o-composition groups in one launch: grid.z walks wo_a's contiguous
 * group blocks against each group's slice of the attention output, ranks
 * landing at the group's offset - block-diagonal through the strided kernel.
 */
static cudaError_t SparkDsv4ModuleRunOutputComposition(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, uint32_t rows)
{
	SparkDsv4LinearView view = layer->attn.wo_a;
	uint32_t query_dimension = SparkDsv4ModuleTpQueryDimension(state);
	uint32_t input_group_dimension = SparkDsv4ModuleTpOutputGroupInput(state);
	uint32_t group_count = SparkDsv4ModuleTpOutputGroupCount(state);
	uint64_t payload_stride = SparkDsv4StagePackPayloadBytes(view.weight_format,SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,input_group_dimension);
	uint64_t scale_stride = SparkDsv4StagePackScaleBytes(view.weight_format,SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,input_group_dimension);
	view.rows = SPARK_DSV4_MODEL_OUTPUT_LORA_RANK;
	view.columns = input_group_dimension;
	return(SparkDsv4LaunchStridedLinear((cudaStream_t)slot->cuda_stream,&view,view.payload,(const uint8_t *)view.scale_data,payload_stride,scale_stride,slot->attn_out_bf16,query_dimension,0u,view.columns,slot->o_ranks_bf16,(uint64_t)group_count * SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,0u,SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,group_count,rows));
}

typedef void (*SparkDsv4TpContinuationFunction)(void *context,
	SparkStatus status);

typedef struct SparkDsv4TpContinuation
{
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	SparkDsv4TpContinuationFunction function;
	void *context;
} SparkDsv4TpContinuation;

struct SparkDsv4TpFrameContinuation
{
	SparkDsv4TpContinuation collective;
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	const void *input_streams_bf16;
	void *output_streams_bf16;
	uint32_t rows;
	uint32_t layer_index;
	uint32_t side;
	uint32_t active;
	uint32_t chain_step_index;
	uint32_t chain_step_count;
	uint32_t dspark_verify;
};

static void SparkDsv4ModuleTpCompletion(
	void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkDsv4TpContinuation *continuation =
		(SparkDsv4TpContinuation *)context;
	if ( continuation == 0 || continuation->function == 0 ||
		completion == 0 )
		return;
	continuation->function(continuation->context,completion->status);
}

static SparkStatus SparkDsv4ModuleReduceHidden(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	void *device_bf16,
	uint32_t rows,
	SparkDsv4TpContinuation *continuation)
{
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t ordinal;
	if ( state == 0 || slot == 0 || device_bf16 == 0 || rows == 0u ||
		rows > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT ||
		continuation == 0 || continuation->function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->tp_degree == 1u )
	{
		continuation->function(continuation->context,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_device_collective_initialized == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	ordinal = atomic_fetch_add_explicit(&state->tp_next_ordinal,1u,
		memory_order_relaxed);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = (uint32_t)(slot - state->slots);
	submission.active_sequence_count = rows;
	submission.flags =
		SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = device_bf16;
	submission.full_device = device_bf16;
	submission.cuda_stream = slot->cuda_stream;
	submission.completion_function = SparkDsv4ModuleTpCompletion;
	submission.completion_context = continuation;
	return(SparkTpDeviceCollectiveSubmitBf16(
		&state->tp_device_collective,&submission));
}

typedef struct SparkDsv4WeightReadAheadPlan
{
	const void *payload;
	uint64_t bytes;
	const void *auxiliary_payload;
	uint64_t auxiliary_bytes;
} SparkDsv4WeightReadAheadPlan;

static cudaError_t SparkDsv4ModuleEnqueueWeightReadAhead(
	cudaStream_t stream,uint32_t *sink_u32,uint32_t sink_word_capacity,
	void *launch_context)
{
	SparkDsv4WeightReadAheadPlan *plan =
		(SparkDsv4WeightReadAheadPlan *)launch_context;
	uint32_t block_capacity;
	if ( plan == 0 || sink_word_capacity == 0u ||
		(sink_word_capacity % SPARK_DSV4_WEIGHT_READ_AHEAD_THREAD_COUNT) != 0u )
		return(cudaErrorInvalidValue);
	block_capacity = sink_word_capacity /
		SPARK_DSV4_WEIGHT_READ_AHEAD_THREAD_COUNT;
	if ( block_capacity > SPARK_DSV4_WEIGHT_READ_AHEAD_MAX_BLOCK_COUNT )
		return(cudaErrorInvalidValue);
	return(SparkDsv4LaunchWeightReadAhead(stream,plan->payload,plan->bytes,
		plan->auxiliary_payload,plan->auxiliary_bytes,sink_u32,block_capacity));
}

static SparkStatus SparkDsv4ModuleReduceHiddenReadAhead(
	SparkDsv4ModuleState *state,SparkDsv4ModuleSlot *slot,void *device_bf16,
	uint32_t rows,SparkDsv4TpContinuation *continuation,
	const void *read_ahead_payload,uint64_t read_ahead_bytes,
	const void *read_ahead_auxiliary_payload,uint64_t read_ahead_auxiliary_bytes)
{
	SparkDsv4WeightReadAheadPlan plan;
	SparkStatus status,join_status;
	if ( state->tp_degree <= 1u || read_ahead_payload == 0 ||
		read_ahead_bytes == 0u )
		return(SparkDsv4ModuleReduceHidden(state,slot,device_bf16,rows,
			continuation));
	plan.payload = read_ahead_payload;
	plan.bytes = read_ahead_bytes;
	plan.auxiliary_payload = read_ahead_auxiliary_payload;
	plan.auxiliary_bytes = read_ahead_auxiliary_bytes;
	status = SparkStageModuleCudaReadAheadArm(SPARK_DSV4_MODULE_TAG,
		&slot->weight_read_ahead,(cudaStream_t)slot->cuda_stream,
		SparkDsv4ModuleEnqueueWeightReadAhead,&plan);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleReduceHidden(state,slot,device_bf16,rows,
			continuation);
	if ( status != SPARK_STATUS_OK )
	{
		join_status = SparkStageModuleCudaReadAheadJoin(SPARK_DSV4_MODULE_TAG,
			&slot->weight_read_ahead,(cudaStream_t)slot->cuda_stream);
		if ( join_status != SPARK_STATUS_OK )
			status = join_status;
	}
	return(status);
}

static uint64_t SparkDsv4ModuleHcFunctionBytes(void)
{
	return((uint64_t)SPARK_DSV4_MODEL_HC_MIX_ROWS *
		SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * sizeof(float));
}

static uint64_t SparkDsv4ModuleHcHeadFunctionBytes(void)
{
	return((uint64_t)SPARK_DSV4_MODEL_HC_STREAM_COUNT *
		SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * sizeof(float));
}

static uint32_t SparkDsv4ModulePrefillWaveRowCount(
	SparkDsv4ModuleState *state,
	const SparkDsv4PrefillBatchView *prefill,
	uint32_t first_row)
{
	return(SparkRowLayoutRoundMajorWaveRowCount(first_row,prefill->row_count,prefill->row_lane_indices,SparkDsv4ModuleClaimedLaneOrdinal,state));
}

static cudaError_t SparkDsv4ModuleRunAttentionRows(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, void *cache, uint64_t lane_stride, const float *sink_f32, uint32_t layer_kind, uint32_t first_row, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t kv_offset = SparkDsv4PrefillRowElementOffset(first_row,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION);
	uint32_t query_dimension = SparkDsv4ModuleTpQueryDimension(state);
	uint64_t q_offset = SparkDsv4PrefillRowElementOffset(first_row,query_dimension);
	uint64_t topk_offset = SparkDsv4PrefillRowElementOffset(first_row,state->topk_column_count);
	cudaError_t error;
	error = SparkDsv4LaunchCacheScatter(stream,(const uint8_t *)slot->kv_bf16 + kv_offset * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,0,cache,lane_stride,slot->row_lane_indices + first_row,slot->row_positions + first_row,rows,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,0u,0u,SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchSparseAttn(stream,(const uint8_t *)slot->q_bf16 + q_offset * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,cache,lane_stride,slot->row_lane_indices + first_row,slot->row_page_table_indices + first_row,slot->physical_page_table,state->paged_cache.lane_page_capacity,SparkDsv4PagedPoolCompressedEntries(layer_kind),slot->topk_idxs + topk_offset,slot->attention_slot_counts + first_row,state->topk_column_count,sink_f32,1.0f / sqrtf((float)SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION),(uint8_t *)slot->attn_out_bf16 + q_offset * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,slot->sparse_attn_partials_f32,state->sparse_attn_partial_capacity,state->multiprocessor_count,rows,SparkDsv4ModuleTpQueryHeads(state),SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION);
	return(error);
}

static cudaError_t SparkDsv4ModuleRunCausalAttention(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4PrefillBatchView *prefill, void *cache, uint64_t lane_stride, const float *sink_f32, uint32_t layer_kind, uint32_t rows)
{
	uint32_t row,wave_rows;
	cudaError_t error;
	if ( prefill == 0 )
		return(SparkDsv4ModuleRunAttentionRows(state,slot,cache,lane_stride,sink_f32,layer_kind,0u,rows));
	if ( prefill->row_count != rows )
		return(cudaErrorInvalidValue);
	error = cudaSuccess;
	for (row=0u; error==cudaSuccess && row<rows; row+=wave_rows)
	{
		wave_rows = SparkDsv4ModulePrefillWaveRowCount(state,prefill,row);
		if ( wave_rows == 0u )
			return(cudaErrorInvalidValue);
		error = SparkDsv4ModuleRunAttentionRows(state,slot,cache,lane_stride,sink_f32,layer_kind,row,wave_rows);
	}
	return(error);
}

static cudaError_t SparkDsv4ModuleRunQueryKvRanks(SparkDsv4ModuleSlot *slot, cudaStream_t stream, const SparkDsv4LayerWeights *layer, uint32_t rows)
{
	return(SparkDsv4LaunchFp8LinearPair(stream,&layer->attn.wq_a,
		&layer->attn.wkv,slot->normalized_bf16,slot->delta_bf16,
		slot->kv_bf16,rows));
}

static cudaError_t SparkDsv4ModuleRunQueryRankPost(SparkDsv4ModuleSlot *slot, cudaStream_t stream, const SparkDsv4LayerWeights *layer, uint32_t rows)
{
	return(SparkDsv4LaunchRmsNorm(stream,slot->delta_bf16,
		layer->attn.q_norm_weight_bf16,slot->qr_bf16,rows,
		SPARK_DSV4_MODEL_QUERY_LORA_RANK,SPARK_DSV4_MODEL_RMS_NORM_EPSILON));
}

static cudaError_t SparkDsv4ModuleRunQueryProjection(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, cudaStream_t stream, const SparkDsv4LayerWeights *layer, const float *freqs, uint32_t rows)
{
	cudaError_t error;
	error = SparkDsv4LaunchLinear(stream,&layer->attn.wq_b,slot->qr_bf16,
		slot->q_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchQueryHeadRms(stream,slot->q_bf16,rows,
			SparkDsv4ModuleTpQueryHeads(state),
			SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,
			SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->q_bf16,freqs,
			slot->row_positions,rows,SparkDsv4ModuleTpQueryHeads(state),
			SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,
			SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	return(error);
}

static cudaError_t SparkDsv4ModuleRunKvPost(SparkDsv4ModuleSlot *slot, cudaStream_t stream, const SparkDsv4LayerWeights *layer, const float *freqs, uint32_t rows)
{
	return(SparkDsv4LaunchKvPost(stream,slot->kv_bf16,
		layer->attn.kv_norm_weight_bf16,freqs,slot->row_positions,rows,
		SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,
		SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,
		SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION -
			SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,
		SPARK_DSV4_MODEL_KV_QUANT_BLOCK,
		SPARK_DSV4_MODEL_RMS_NORM_EPSILON));
}

static uint32_t SparkDsv4ModuleProjectionChannels(uint32_t kind)
{
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		return(SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR *
			SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION);
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA )
		return(SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION);
	return(0u);
}

static cudaError_t SparkDsv4ModuleRunProjectionShards(
	SparkDsv4ModuleState *state,SparkDsv4ModuleSlot *slot,
	const SparkDsv4LayerWeights *layer,uint32_t kind,uint32_t rows)
{
	SparkStageModuleCudaFork *fork = &slot->compute_fork;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t branches = kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? 2u : 1u;
	uint32_t channels = SparkDsv4ModuleProjectionChannels(kind);
	uint32_t index_channels = kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ?
		SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR *
		SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION : 0u;
	cudaError_t error;
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		error = SparkDsv4ModuleRunQueryKvRanks(slot,stream,layer,rows);
	else
	{
		error = SparkStageModuleCudaForkBegin(fork,stream,branches);
		if ( error == cudaSuccess )
			error = SparkDsv4ModuleRunCompressorProjection(slot,&slot->compressor,
				fork->auxiliary_streams[0],&layer->compressor,rows);
		if ( error == cudaSuccess && index_channels != 0u )
			error = SparkDsv4ModuleRunCompressorProjection(slot,
				&slot->index_compressor,fork->auxiliary_streams[1],
				&layer->indexer.compressor,rows);
		if ( error == cudaSuccess )
			error = SparkDsv4ModuleRunQueryKvRanks(slot,stream,layer,rows);
		if ( error == cudaSuccess )
			error = SparkStageModuleCudaForkJoin(fork,stream,branches);
	}
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchPackProjectionShards(stream,slot->delta_bf16,
			slot->kv_bf16,slot->compressor.kv_bf16,
			slot->compressor.score_bf16,slot->index_compressor.kv_bf16,
			slot->index_compressor.score_bf16,slot->ffn_accum_bf16,rows,
			state->tp_rank,state->tp_degree,channels,index_channels);
	return(error);
}

static cudaError_t SparkDsv4ModuleRunAttentionSerialPrologue(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, const float *freqs, uint32_t layer_index, uint32_t kind, void *cache, uint64_t lane_stride, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t state_offset = state->compress_state_offset_by_layer[layer_index];
	uint64_t state_stride = state->compress_state_lane_stride_by_layer[layer_index];
	cudaError_t error;
	error = SparkDsv4ModuleRunQueryKvRanks(slot,stream,layer,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunQueryRankPost(slot,stream,layer,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunQueryProjection(state,slot,stream,layer,
			freqs,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunKvPost(slot,stream,layer,freqs,rows);
	if ( error == cudaSuccess && kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		error = SparkDsv4ModuleRunCompressor(state,slot,&slot->compressor,stream,
			&layer->compressor,state->compress_kv_state_f32 + state_offset,
			state->compress_score_state_f32 +
			state->compress_score_state_offset_by_layer[layer_index],state_stride,
			cache,lane_stride,SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,
			SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,0u,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleStageTopk(state,slot,stream,kind,rows);
	if ( error == cudaSuccess && kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		error = SparkDsv4ModuleRunIndexer(state,slot,stream,layer,layer_index,rows);
	return(error);
}

static cudaError_t SparkDsv4ModuleRunAttentionProjectedPrologue(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, const float *freqs, uint32_t layer_index, uint32_t kind, void *cache, uint64_t lane_stride, uint32_t rows)
{
	SparkStageModuleCudaFork *fork = &slot->compute_fork;
	cudaStream_t primary = (cudaStream_t)slot->cuda_stream;
	cudaStream_t compressor_stream = fork->auxiliary_streams[0];
	cudaStream_t kv_stream = kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA ?
		fork->auxiliary_streams[0] : fork->auxiliary_streams[1];
	uint64_t state_offset = state->compress_state_offset_by_layer[layer_index];
	uint64_t state_stride = state->compress_state_lane_stride_by_layer[layer_index];
	uint32_t branch_count = kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA ? 1u : 2u;
	uint32_t channels = SparkDsv4ModuleProjectionChannels(kind);
	uint32_t index_channels = kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ?
		SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR *
		SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION : 0u;
	cudaError_t error;
	error = SparkDsv4LaunchUnpackProjectionShards(primary,slot->ffn_accum_bf16,
		slot->delta_bf16,slot->kv_bf16,slot->compressor.kv_bf16,
		slot->compressor.score_bf16,slot->index_compressor.kv_bf16,
		slot->index_compressor.score_bf16,rows,channels,index_channels);
	if ( error == cudaSuccess )
		error = SparkStageModuleCudaForkBegin(fork,primary,branch_count);
	if ( error == cudaSuccess && kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		error = SparkDsv4ModuleRunCompressorPost(state,slot,&slot->compressor,
			compressor_stream,&layer->compressor,
			state->compress_kv_state_f32 + state_offset,
			state->compress_score_state_f32 +
			state->compress_score_state_offset_by_layer[layer_index],state_stride,
			cache,lane_stride,SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,
			SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,0u,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunQueryRankPost(slot,primary,layer,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleStageTopk(state,slot,primary,kind,rows);
	if ( error == cudaSuccess )
		error = cudaEventRecord(fork->milestone_event,primary);
	if ( error == cudaSuccess )
		error = cudaStreamWaitEvent(kv_stream,fork->milestone_event,0u);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunKvPost(slot,kv_stream,layer,freqs,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunQueryProjection(state,slot,primary,layer,
			freqs,rows);
	if ( error == cudaSuccess && kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		error = SparkDsv4ModuleRunIndexerCore(state,slot,kv_stream,layer,
			layer_index,1u,0u,rows);
	if ( error == cudaSuccess )
		error = SparkStageModuleCudaForkJoin(fork,primary,branch_count);
	return(error);
}

static cudaError_t SparkDsv4ModuleRunAttentionTail(SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,const SparkDsv4LayerWeights *layer,
	const SparkDsv4PrefillBatchView *prefill,const float *freqs,void *cache,
	uint64_t lane_stride,uint32_t kind,uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkDsv4ModuleRunCausalAttention(state,slot,prefill,cache,
		lane_stride,layer->attn.sink_f32,kind,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->attn_out_bf16,freqs,slot->row_positions,rows,SparkDsv4ModuleTpQueryHeads(state),SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,1u);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunOutputComposition(state,slot,layer,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->attn.wo_b,
			slot->o_ranks_bf16,slot->delta_bf16,rows);
	return(error);
}

static cudaError_t SparkDsv4ModuleRunAttention(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, const SparkDsv4PrefillBatchView *prefill, uint32_t layer_index, uint32_t rows)
{
	uint32_t kind = SparkDsv4ModelLayerKind(layer_index);
	const float *freqs = kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA ?
		state->base_freqs_f32 : state->compress_freqs_f32;
	void *cache = (uint8_t *)state->kv_cache_bf16 +
		state->cache_offset_by_layer[layer_index] *
		SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t lane_stride = state->cache_lane_stride_by_layer[layer_index];
	cudaError_t error = SparkDsv4ModuleRunAttentionSerialPrologue(state,slot,
		layer,freqs,layer_index,kind,cache,lane_stride,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunAttentionTail(state,slot,layer,prefill,freqs,
			cache,lane_stride,kind,rows);
	return(error);
}

static cudaError_t SparkDsv4ModuleRunAttentionProjected(
	SparkDsv4ModuleState *state,SparkDsv4ModuleSlot *slot,
	const SparkDsv4LayerWeights *layer,uint32_t layer_index,uint32_t rows)
{
	uint32_t kind = SparkDsv4ModelLayerKind(layer_index);
	const float *freqs = kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA ?
		state->base_freqs_f32 : state->compress_freqs_f32;
	void *cache = (uint8_t *)state->kv_cache_bf16 +
		state->cache_offset_by_layer[layer_index] *
		SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t lane_stride = state->cache_lane_stride_by_layer[layer_index];
	cudaError_t error = SparkDsv4ModuleRunAttentionProjectedPrologue(state,slot,
		layer,freqs,layer_index,kind,cache,lane_stride,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunAttentionTail(state,slot,layer,0,freqs,cache,
			lane_stride,kind,rows);
	return(error);
}

/*
 * Device-grouped routed MoE, mirrored from mimo25: one grouping kernel,
 * three all-expert tile launches with device-side counts, the clamped
 * swiglu with the routing weight folded at the intermediate running
 * dense over the grouped pairs, and the unweighted pair reduce
 * accumulating race-free through the inverse map. No host round trips,
 * no per-layer synchronize; the step is graph-capturable end to end.
 */
static cudaError_t SparkDsv4ModuleRunMoeShared(SparkDsv4ModuleSlot *slot, cudaStream_t stream, const SparkDsv4MoeWeights *moe, uint32_t rows, uint32_t expert_width)
{
	cudaError_t error;
	error = SparkDsv4LaunchFusedSharedW13Act(stream,&moe->shared_w1,
		&moe->shared_w3,slot->normalized_bf16,slot->ffn_up_bf16,rows,
		expert_width,SPARK_DSV4_MODEL_SWIGLU_LIMIT);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchExpertLinear(stream,&moe->shared_w2,
			slot->ffn_up_bf16,slot->ffn_accum_bf16,rows);
	return(error);
}

static SparkStatus SparkDsv4ModuleRunMoeRoutedProjection(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4MoeWeights *moe, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t expert_width = SparkDsv4ModuleTpExpertWidth(state);
	cudaError_t error;
	error = SparkDsv4LaunchFusedExpertW13Act(stream,&moe->experts_w1,
		&moe->experts_w3,slot->normalized_bf16,slot->grouped_rows_u32,
		slot->expert_offsets_u32,slot->group_tile_prefix_w1_u32,
		slot->moe_slot_up_bf16,rows,expert_width,
		SPARK_DSV4_MODEL_SWIGLU_LIMIT,state->multiprocessor_count);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_fused_w13_act"));
	error = SparkDsv4LaunchExpertDown(stream,&moe->experts_w2,slot->moe_slot_up_bf16,slot->expert_offsets_u32,slot->group_tile_prefix_w2_u32,slot->moe_slot_out_bf16,rows,expert_width,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,state->multiprocessor_count);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_expert_w2"));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleRunMoe(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, uint32_t layer_index, uint32_t rows)
{
	SparkStageModuleCudaFork *fork = &slot->compute_fork;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkDsv4MoeWeights *moe = &layer->moe;
	uint32_t hash = SparkDsv4StagePackLayerIsHashRouted(layer_index);
	uint32_t expert_width = SparkDsv4ModuleTpExpertWidth(state);
	cudaError_t error;
	SparkStatus status;
	error = SparkStageModuleCudaForkBegin(fork,stream,1u);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunMoeShared(slot,fork->auxiliary_streams[0],moe,
			rows,expert_width);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchGateRoute(stream,&moe->gate,
			slot->normalized_bf16,slot->moe_scores_f32,
			hash != 0u ? 0 : moe->gate_bias_f32,
			hash != 0u ? moe->gate_tid2eid_u32 : 0,slot->input_token_ids,
			rows,SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT,
			SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,
			SPARK_DSV4_MODEL_ROUTED_SCALING_FACTOR,slot->moe_indices_u32,
			slot->moe_weights_f32,expert_width,slot->expert_offsets_u32,
			slot->moe_inverse_u32,slot->grouped_rows_u32,
			slot->group_tile_prefix_w1_u32,slot->group_tile_prefix_w2_u32);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
			"moe_parallel_begin"));
	status = SparkDsv4ModuleRunMoeRoutedProjection(state,slot,moe,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	error = SparkStageModuleCudaForkJoin(fork,stream,1u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchMoePairReduce(stream,slot->moe_slot_out_bf16,
			slot->moe_inverse_u32,slot->moe_weights_f32,
			slot->ffn_accum_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"moe_parallel_reduce"));
}

/*
 * TP=1 retains the original whole-frame launch sequence so its decode graph
 * cache remains valid. TP>1 cannot use this body: each collective splits the
 * frame into stream-ordered graph islands.
 */
static SparkStatus SparkDsv4ModuleRunLocalLayer(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	const void *input_streams_bf16,
	void *output_streams_bf16,
	const SparkDsv4PrefillBatchView *prefill,
	uint32_t layer_index,
	uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkDsv4LayerWeights *layer = &state->layers[layer_index];
	cudaError_t error;
	SparkStatus status;

	if ( state->tp_degree != 1u )
		return(SPARK_STATUS_UNSUPPORTED);
	error = SparkDsv4ModuleHcEnter(slot,input_streams_bf16,
		layer->hc.attn_fn_f32,layer->hc.attn_scale_f32,
		layer->hc.attn_base_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,
			layer->attn_norm_bf16,slot->normalized_bf16,rows,
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION,
			SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunAttention(state,slot,layer,prefill,
			layer_index,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcPost(stream,slot->delta_bf16,
			slot->residual_bf16,slot->post_f32,slot->comb_f32,
			slot->streams_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
			"local_attn_side"));
	error = SparkDsv4ModuleHcEnter(slot,slot->streams_bf16,
		layer->hc.ffn_fn_f32,layer->hc.ffn_scale_f32,
		layer->hc.ffn_base_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,
			layer->ffn_norm_bf16,slot->normalized_bf16,rows,
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION,
			SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
			"local_ffn_enter"));
	status = SparkDsv4ModuleRunMoe(state,slot,layer,layer_index,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	error = SparkDsv4LaunchHcPost(stream,slot->ffn_accum_bf16,
		slot->residual_bf16,slot->post_f32,slot->comb_f32,
		output_streams_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"local_ffn_side"));
}

static SparkStatus SparkDsv4ModuleRunLocalLayers(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	const void *input_streams_bf16,
	void *output_streams_bf16,
	const SparkDsv4PrefillBatchView *prefill,
	uint32_t rows)
{
	const void *layer_input = input_streams_bf16;
	void *layer_output;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t layer;

	for (layer=state->first_layer_index;
		status==SPARK_STATUS_OK &&
		layer<state->first_layer_index+state->layer_count; layer++)
	{
		layer_output = layer + 1u ==
			state->first_layer_index + state->layer_count ?
			output_streams_bf16 : slot->streams_bf16;
		status = SparkDsv4ModuleRunLocalLayer(state,slot,layer_input,
			layer_output,prefill,layer,rows);
		layer_input = layer_output;
	}
	return(status);
}

static void SparkDsv4ModuleContinueLayers(void *context,SparkStatus status);

static SparkStatus SparkDsv4ModuleStartLayers(
	SparkDsv4TpFrameContinuation *continuation)
{
	SparkDsv4ModuleState *state = continuation->state;
	SparkDsv4ModuleSlot *slot = continuation->slot;
	const SparkDsv4LayerWeights *layer =
		&state->layers[continuation->layer_index];
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	SparkStatus status;

	if ( state->tp_degree > 1u )
	{
		status = SparkDsv4ModuleReplayTpIsland(state,slot,
			3u * (continuation->layer_index - state->first_layer_index),
			continuation->rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
		continuation->side = 0u;
	}
	else
	{
		error = SparkDsv4ModuleHcEnter(slot,continuation->input_streams_bf16,
			layer->hc.attn_fn_f32,layer->hc.attn_scale_f32,
			layer->hc.attn_base_f32,continuation->rows);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,
				layer->attn_norm_bf16,slot->normalized_bf16,continuation->rows,
				SPARK_DSV4_MODEL_HIDDEN_DIMENSION,
				SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkDsv4ModuleRunAttention(state,slot,layer,
				0,continuation->layer_index,continuation->rows);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
				"attn_side"));
		continuation->side = 1u;
	}
	continuation->collective.function = SparkDsv4ModuleContinueLayers;
	continuation->collective.context = continuation;
	return(SparkDsv4ModuleReduceHiddenReadAhead(state,slot,
		state->tp_degree > 1u ? slot->ffn_accum_bf16 : slot->delta_bf16,
		continuation->rows,&continuation->collective,layer->attn.wq_b.payload,
		SparkDsv4StagePackPayloadBytes(layer->attn.wq_b.weight_format,
			layer->attn.wq_b.rows,layer->attn.wq_b.columns),
		layer->attn.wq_b.scale_data,
		SparkDsv4StagePackScaleBytes(layer->attn.wq_b.weight_format,
			layer->attn.wq_b.rows,layer->attn.wq_b.columns)));
}

static void SparkDsv4ModuleFinishContinuationTerminal(
	SparkDsv4TpFrameContinuation *continuation,
	SparkStatus status)
{
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	SparkDsv4AsyncCompletion *async;
	SparkStatus enqueue_status;
	uint32_t slot_index;
	state = continuation->state;
	slot = continuation->slot;
	slot_index = (uint32_t)(slot - state->slots);
	async = &state->completions[slot_index];
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"dsv4_stage tp_continuation_failed status=%u stage=%u tp_rank=%u layer=%u side=%u rows=%u\n",(uint32_t)status,state->pp_stage_index,state->tp_rank,continuation->layer_index,continuation->side,continuation->rows);
	async->completion.status = status;
	continuation->active = 0u;
	enqueue_status = SparkDsv4ModuleEnqueueAsync(state,slot,slot_index);
	if ( enqueue_status != SPARK_STATUS_OK )
		SparkDsv4ModuleCompleteAfterFailedEnqueue(state,slot,slot_index,
			enqueue_status);
}

static SparkStatus SparkDsv4ModuleValidateResidentChain(
	const SparkDsv4TpFrameContinuation *continuation)
{
	const SparkDsv4AsyncCompletion *async;
	const SparkModelDriverCacheLane *lane;
	uint32_t index,last_position,slot_index;
	if ( continuation->chain_step_count <= 1u )
		return(SPARK_STATUS_OK);
	slot_index = (uint32_t)(continuation->slot - continuation->state->slots);
	async = &continuation->state->completions[slot_index];
	if ( continuation->rows != async->lane_count ||
		async->cache_lane_count != async->lane_count ||
		async->prepared_cache_lane_count != async->lane_count )
		return(SPARK_STATUS_VALIDATION_FAILED);
	for (index=0u; index<async->lane_count; index++)
	{
		lane = &async->cache_lanes[index];
		if ( lane->flags != 0u || lane->context_token_count !=
			lane->sequence_position + 1u ||
			lane->context_token_count > UINT32_MAX -
			(continuation->chain_step_count - 1u) )
			return(SPARK_STATUS_VALIDATION_FAILED);
		last_position = lane->context_token_count +
			continuation->chain_step_count - 2u;
		if ( last_position >= continuation->state->max_sequence_positions ||
			lane->sequence_position / SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS !=
			last_position / SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS ||
			async->prepared_cache_lanes[index].logical_page_count <=
			last_position / SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleRecordResidentToken(
	SparkDsv4TpFrameContinuation *continuation,
	uint32_t advance)
{
	SparkDsv4AsyncCompletion *async;
	SparkDsv4ModuleSlot *slot;
	cudaError_t error;
	uint32_t index,slot_index;
	slot = continuation->slot;
	slot_index = (uint32_t)(slot - continuation->state->slots);
	async = &continuation->state->completions[slot_index];
	error = SparkDsv4LaunchResidentTokenFeedback(
		(cudaStream_t)slot->cuda_stream,slot->output_token_ids,
		slot->resident_token_ids,slot->input_token_ids,slot->row_positions,
		slot->row_emit_positions,slot->row_emit_positions_hca,
		continuation->rows,continuation->chain_step_count,
		continuation->chain_step_index,advance);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
			"resident_token_feedback"));
	if ( advance == 0u )
		return(SPARK_STATUS_OK);
	for (index=0u; index<async->lane_count; index++)
	{
		async->cache_lanes[index].context_token_count++;
		async->lane_next_positions[index]++;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleContinueResidentChain(
	SparkDsv4TpFrameContinuation *continuation)
{
	SparkStatus status;
	status = SparkDsv4ModuleRecordResidentToken(continuation,1u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	continuation->chain_step_index++;
	continuation->layer_index = continuation->state->first_layer_index;
	continuation->side = 0u;
	continuation->input_streams_bf16 = continuation->slot->streams_bf16;
	return(SparkDsv4ModuleStartLayers(continuation));
}

static SparkStatus SparkDsv4ModuleDsparkDrive(SparkDsv4ModuleState *state,SparkDsv4ModuleSlot *slot,uint32_t lane_index,uint32_t anchor_token_id,uint64_t anchor_position);
static SparkStatus SparkDsv4ModuleRunDsparkDraft(SparkDsv4ModuleState *state,SparkDsv4ModuleSlot *slot,uint32_t lane_index,uint32_t anchor_token_id,uint64_t anchor_position);
static void SparkDsv4ModuleDsparkDraftCollectiveComplete(void *context,const SparkTpDeviceCollectiveCompletion *completion)
{
	(void)context;
	(void)completion;
}

/* The draft submits collectives back-to-back on one stream; the credit
 * slot frees only after the transport worker consumes the terminal phase,
 * so poll the operation back to FREE before the next submit. */
static SparkStatus SparkDsv4ModuleDsparkWaitCollectiveFree(
	SparkDsv4ModuleState *state,
	uint64_t ordinal)
{
	uint32_t spins;
	for (spins = 0u; spins < 4000000u; spins++)
	{
		uint32_t phase = UINT32_MAX,failure = 0u;
		SparkStatus status = SparkTpDeviceCollectiveOperationPhase(
			&state->tp_device_collective,ordinal,&phase,&failure);
		if ( failure != 0u )
			return(SPARK_STATUS_IO_ERROR);
		if ( status == SPARK_STATUS_NOT_FOUND )
			return(SPARK_STATUS_OK);
		if ( status == SPARK_STATUS_OK &&
			phase == SPARK_TP_DEVICE_COLLECTIVE_PHASE_FREE )
			return(SPARK_STATUS_OK);
		if ( status != SPARK_STATUS_OK )
			return(status);
		sched_yield();
	}
	return(SPARK_STATUS_BUSY);
}

/* TP allreduce (SUM, bf16) over the draft's MoE output: the MTP expert
 * tensors are quarter-width shards per rank, so each rank's ffn_accum
 * holds a partial sum; the stream-ordered allreduce completes it. */
static SparkStatus SparkDsv4ModuleDsparkReduceMoeOutput(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t rows)
{
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t ordinal;
	if ( state->tp_degree == 1u )
		return(SPARK_STATUS_OK);
	if ( state->tp_device_collective_initialized == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	ordinal = atomic_fetch_add_explicit(&state->tp_next_ordinal,1u,
		memory_order_relaxed);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = (uint32_t)(slot - state->slots);
	submission.active_sequence_count = rows;
	submission.flags =
		SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = slot->ffn_accum_bf16;
	submission.full_device = slot->ffn_accum_bf16;
	submission.cuda_stream = slot->cuda_stream;
	submission.completion_function =
		SparkDsv4ModuleDsparkDraftCollectiveComplete;
	submission.completion_context = 0;
	{
		SparkStatus status = SparkTpDeviceCollectiveSubmitBf16(
			&state->tp_device_collective,&submission);
		if ( status != SPARK_STATUS_OK )
			return(status);
		return(SparkDsv4ModuleDsparkWaitCollectiveFree(state,ordinal));
	}
}

static void SparkDsv4ModuleContinueHeadMax(void *context,SparkStatus status)
{
	SparkDsv4TpFrameContinuation *continuation;
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	cudaError_t error;
	continuation = (SparkDsv4TpFrameContinuation *)context;
	if ( continuation == 0 || continuation->active == 0u )
		return;
	state = continuation->state;
	slot = continuation->slot;
	error = cudaSuccess;
	if ( status == SPARK_STATUS_OK )
		error = SparkDsv4LaunchHeadMaxlocUnpack(
			(cudaStream_t)slot->cuda_stream,slot->head_maxloc_u64,
			slot->output_token_ids,continuation->rows);
	if ( error == cudaSuccess && status == SPARK_STATUS_OK &&
		continuation->chain_step_count > 1u &&
		continuation->chain_step_index + 1u <
		continuation->chain_step_count )
	{
		status = SparkDsv4ModuleContinueResidentChain(continuation);
		if ( status == SPARK_STATUS_OK )
			return;
	}
	if ( error == cudaSuccess && status == SPARK_STATUS_OK &&
		continuation->chain_step_count > 1u )
		status = SparkDsv4ModuleRecordResidentToken(continuation,0u);
	/* The verify frame's acceptance runs on EVERY rank (the lane store and
	 * the cache advance must agree across the TP group), so every rank
	 * copies the REDUCED head tokens to the host; the head rank's copy is
	 * the one the adapter publishes. */
	if ( error == cudaSuccess && status == SPARK_STATUS_OK &&
		(state->owns_final_head != 0u || continuation->dspark_verify != 0u) )
		error = cudaMemcpyAsync(slot->host_output_token_ids,
			continuation->chain_step_count > 1u ? slot->resident_token_ids :
			slot->output_token_ids,(uint64_t)continuation->rows *
			continuation->chain_step_count * sizeof(uint32_t),
			cudaMemcpyDeviceToHost,(cudaStream_t)slot->cuda_stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
			"tp_head_maxloc_finish");
	/* DSpark acceptance reads the head tokens on the host: the D2H above
	 * is asynchronous, so fence before touching host_output_token_ids. */
	if ( status == SPARK_STATUS_OK && state->dspark_enabled != 0u )
	{
		cudaError_t sync_error = cudaStreamSynchronize(
			(cudaStream_t)slot->cuda_stream);
		if ( sync_error != cudaSuccess )
			status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,
				sync_error,"dspark_head_sync");
	}
	/* DSpark: greedy Leviathan acceptance over the staged rows, burst
	 * emission, and the anchor step's taps + token to the LANE store
	 * (slots cycle across submissions, so the next draft runs from state,
	 * not from this slot). The D2D tap copy rides the stream before the
	 * completion callback fires. */
	if ( status == SPARK_STATUS_OK &&
		(continuation->rows == 1u || continuation->dspark_verify != 0u) &&
		state->dspark_enabled != 0u )
	{
		SparkDsv4AsyncCompletion *async;
		uint32_t slot_index = (uint32_t)(slot - state->slots);
		uint32_t lane_index,accepted,layer;
		async = &state->completions[slot_index];
		lane_index = async->lane_indices[0];
		accepted = 0u;
		if ( continuation->dspark_verify != 0u )
		{
			if ( slot->dspark_verify_accept != 0u )
			{
				for (accepted = 0u;
					accepted < SPARK_DSV4_MODEL_DSPARK_SPEC_STEP; accepted++)
					if ( slot->host_output_token_ids[accepted] !=
						slot->dspark_host_draft_tokens[accepted] )
						break;
			}
			fprintf(stderr,"dspark_accept tp_rank=%u lane=%u accepted=%u rows=%u chain=%u\n",state->tp_rank,lane_index,accepted,continuation->rows,continuation->chain_step_count);
			async->completion.accepted_token_count = 1u + accepted;
			async->completion.tokens_per_sequence = 1u + accepted;
			async->tokens_per_sequence = 1u + accepted;
			async->emitted_token_count = 1u + accepted;
			if ( accepted != 0u )
			{
				async->lane_next_positions[0] += accepted;
				async->cache_lanes[0].context_token_count += accepted;
			}
			/* Compressed-state rollback: a rejected boundary row's
			 * overlap-shift moved speculative content into the CSA
			 * previous windows and its emission polluted the compressed
			 * cache; undo both. The accepted rows' position-keyed slots
			 * stay (they hold the true prefix). */
			{
				uint32_t boundary,ordinal,csa_layer,kind,row;
				uint32_t boundary_rejected = 0u;
				const uint64_t channels = 2u * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
				const uint64_t window_floats = (uint64_t)SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO * channels;
				const uint64_t slot_bytes = SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
				for (boundary = 0u; boundary < slot->dspark_boundary_count; boundary++)
					if ( slot->dspark_boundary_rows[boundary] > accepted )
						boundary_rejected = 1u;
				csa_layer = 0u;
				ordinal = 0u;
				for (layer = state->first_layer_index;
					layer < state->first_layer_index + state->layer_count &&
					error == cudaSuccess; layer++)
				{
					kind = SparkDsv4ModelLayerKind(layer);
					if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA )
						continue;
					if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA &&
						boundary_rejected != 0u )
					{
						void *kv_dest = (uint8_t *)state->compress_kv_state_f32 +
							state->compress_state_offset_by_layer[layer] * sizeof(float);
						void *score_dest = (uint8_t *)state->compress_score_state_f32 +
							state->compress_score_state_offset_by_layer[layer] * sizeof(float);
						error = cudaMemcpyAsync(kv_dest,
							slot->dspark_csa_previous_save + (uint64_t)csa_layer * 2u * window_floats * sizeof(float),
							window_floats * sizeof(float),cudaMemcpyDeviceToDevice,
							(cudaStream_t)slot->cuda_stream);
						if ( error == cudaSuccess )
							error = cudaMemcpyAsync(score_dest,
								slot->dspark_csa_previous_save + ((uint64_t)csa_layer * 2u + 1u) * window_floats * sizeof(float),
								window_floats * sizeof(float),cudaMemcpyDeviceToDevice,
								(cudaStream_t)slot->cuda_stream);
						csa_layer++;
					}
					else if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
						csa_layer++;
					for (boundary = 0u; boundary < slot->dspark_boundary_count &&
						error == cudaSuccess; boundary++)
					{
						if ( slot->dspark_boundary_rows[boundary] <= accepted )
							continue;
						row = slot->dspark_boundary_rows[boundary];
						uint64_t position = slot->host_row_positions[row];
						uint64_t slot64 = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS +
							((position % SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS) /
							 (kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ?
							  SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO :
							  SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO));
						void *dest = (uint8_t *)state->kv_cache_bf16 +
							(uint64_t)state->cache_offset_by_layer[layer] * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES +
							(uint64_t)slot->host_row_lane_indices[row] *
								state->cache_lane_stride_by_layer[layer] * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES +
							slot64 * slot_bytes;
						error = cudaMemcpyAsync(dest,
							slot->dspark_emission_save + ((uint64_t)ordinal * 2u + boundary) * slot_bytes,
							slot_bytes,cudaMemcpyDeviceToDevice,
							(cudaStream_t)slot->cuda_stream);
					}
					if ( error == cudaSuccess && kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA &&
						slot->dspark_hca_boundary_row != UINT32_MAX &&
						slot->dspark_hca_boundary_row > accepted )
					{
						row = slot->dspark_hca_boundary_row;
						uint64_t position = slot->host_row_positions[row];
						uint64_t slot64 = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS +
							((position % SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS) /
							 SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO);
						void *dest = (uint8_t *)state->kv_cache_bf16 +
							(uint64_t)state->cache_offset_by_layer[layer] * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES +
							(uint64_t)slot->host_row_lane_indices[row] *
								state->cache_lane_stride_by_layer[layer] * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES +
							slot64 * slot_bytes;
						error = cudaMemcpyAsync(dest,
							slot->dspark_emission_save + (uint64_t)ordinal * 2u * slot_bytes,
							slot_bytes,cudaMemcpyDeviceToDevice,
							(cudaStream_t)slot->cuda_stream);
					}
					ordinal++;
				}
				if ( error != cudaSuccess )
					status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
						"dspark_rollback_restore");
			}
		}
		if ( lane_index < state->resident_sequence_capacity )
		{
			uint64_t tap_bytes = (uint64_t)SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT *
				SPARK_DSV4_MODEL_HIDDEN_DIMENSION *
				SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
			if ( continuation->dspark_verify != 0u )
			{
				/* publish the ACCEPTED row's taps from the per-row tap
				 * buffer (row 0 for the padded fallback frames) */
				uint64_t row_bytes = SPARK_DSV4_MODEL_HIDDEN_DIMENSION *
					SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
				for (layer = 0u; layer < SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT; layer++)
				{
					const void *source = (const uint8_t *)slot->dspark_verify_tap_bf16 +
						((uint64_t)layer * (SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u) +
						 accepted) * SPARK_DSV4_MODEL_HIDDEN_DIMENSION *
						SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
					error = cudaMemcpyAsync(
						(uint8_t *)state->dspark_tap_store_bf16 +
							(uint64_t)lane_index * tap_bytes +
							layer * row_bytes,
						source,row_bytes,cudaMemcpyDeviceToDevice,
						(cudaStream_t)slot->cuda_stream);
					if ( error != cudaSuccess )
						break;
				}
			}
			else
				error = cudaMemcpyAsync(
					(uint8_t *)state->dspark_tap_store_bf16 + (uint64_t)lane_index * tap_bytes,
					slot->dspark_tap_bf16,tap_bytes,cudaMemcpyDeviceToDevice,
					(cudaStream_t)slot->cuda_stream);
			if ( error == cudaSuccess )
			{
				state->dspark_lane_anchor[lane_index] =
					slot->host_output_token_ids[accepted];
				state->dspark_lane_position[lane_index] =
					async->lane_next_positions[0] - 1u;
				state->dspark_lane_ready[lane_index] = 1u;
				fprintf(stderr,"dspark_publish tp_rank=%u lane=%u anchor=%u out0=%u accepted=%u pos=%llu rows=%u verify=%u\n",state->tp_rank,lane_index,state->dspark_lane_anchor[lane_index],slot->host_output_token_ids[0],accepted,(unsigned long long)state->dspark_lane_position[lane_index],continuation->rows,continuation->dspark_verify);
			}
		}
	}
	SparkDsv4ModuleFinishContinuationTerminal(continuation,status);
}

static SparkStatus SparkDsv4ModuleReduceHeadMax(
	SparkDsv4TpFrameContinuation *continuation)
{
	SparkTpDeviceCollectiveSubmission submission;
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	uint64_t ordinal;
	if ( continuation == 0 || continuation->state == 0 ||
		continuation->slot == 0 || continuation->rows == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state = continuation->state;
	slot = continuation->slot;
	ordinal = atomic_fetch_add_explicit(&state->tp_next_ordinal,1u,
		memory_order_relaxed);
	continuation->collective.function = SparkDsv4ModuleContinueHeadMax;
	continuation->collective.context = continuation;
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = (uint32_t)(slot - state->slots);
	submission.active_sequence_count = continuation->rows;
	submission.flags =
		SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = slot->head_maxloc_u64;
	submission.full_device = slot->head_maxloc_u64;
	submission.cuda_stream = slot->cuda_stream;
	submission.completion_function = SparkDsv4ModuleTpCompletion;
	submission.completion_context = &continuation->collective;
	return(SparkTpDeviceCollectiveSubmitU64Max(
		&state->tp_device_collective,&submission));
}

static void SparkDsv4ModuleContinueLayers(void *context,SparkStatus status)
{
	SparkDsv4TpFrameContinuation *continuation =
		(SparkDsv4TpFrameContinuation *)context;
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	const SparkDsv4LayerWeights *layer;
	const void *read_ahead_payload;
	uint64_t read_ahead_bytes;
	void *layer_output;
	cudaStream_t stream;
	cudaError_t error;
	SparkStatus read_ahead_status;

	if ( continuation == 0 || continuation->active == 0u )
		return;
	state = continuation->state;
	slot = continuation->slot;
	layer = &state->layers[continuation->layer_index];
	stream = (cudaStream_t)slot->cuda_stream;
	read_ahead_status = SparkStageModuleCudaReadAheadJoin(SPARK_DSV4_MODULE_TAG,
		&slot->weight_read_ahead,stream);
	if ( read_ahead_status != SPARK_STATUS_OK )
		status = read_ahead_status;
	if ( status != SPARK_STATUS_OK )
	{
		SparkDsv4ModuleFinishContinuationTerminal(continuation,status);
		return;
	}
	if ( continuation->side == 0u )
	{
		status = state->tp_degree > 1u ?
			SparkDsv4ModuleReplayTpIsland(state,slot,
				3u * (continuation->layer_index - state->first_layer_index) + 1u,
				continuation->rows) : SPARK_STATUS_VALIDATION_FAILED;
		if ( status == SPARK_STATUS_OK )
		{
			continuation->side = 1u;
			status = SparkDsv4ModuleReduceHiddenReadAhead(state,slot,
				slot->delta_bf16,continuation->rows,&continuation->collective,
				layer->hc.ffn_fn_f32,SparkDsv4ModuleHcFunctionBytes(),0,0u);
			if ( status == SPARK_STATUS_OK )
				return;
		}
	}
	else if ( continuation->side == 1u )
	{
		if ( state->tp_degree > 1u )
			status = SparkDsv4ModuleReplayTpIsland(state,slot,
				3u * (continuation->layer_index - state->first_layer_index) + 2u,
				continuation->rows);
		else
		{
			error = SparkDsv4LaunchHcPost(stream,slot->delta_bf16,
				slot->residual_bf16,slot->post_f32,slot->comb_f32,
				slot->streams_bf16,continuation->rows,
				SPARK_DSV4_MODEL_HC_STREAM_COUNT,
				SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
			if ( error == cudaSuccess )
				error = SparkDsv4ModuleHcEnter(slot,slot->streams_bf16,
					layer->hc.ffn_fn_f32,layer->hc.ffn_scale_f32,
					layer->hc.ffn_base_f32,continuation->rows);
			if ( error == cudaSuccess )
				error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,
					layer->ffn_norm_bf16,slot->normalized_bf16,
					continuation->rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,
					SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
			status = error == cudaSuccess ? SparkDsv4ModuleRunMoe(state,slot,
				layer,continuation->layer_index,continuation->rows) :
				SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"ffn_enter");
		}
		if ( status == SPARK_STATUS_OK &&
			SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u &&
			continuation->layer_index >= SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_FIRST &&
			continuation->layer_index < SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_FIRST +
				SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_COUNT &&
			(continuation->rows == 1u || continuation->dspark_verify != 0u) )
		{
			/* Tap the target hidden (mean over the hc streams) at the
			 * draft's attachment layers. Verify frames tap ALL staged rows
			 * (the acceptance loop picks the accepted row afterwards). */
			uint8_t *tap_dst = (uint8_t *)slot->dspark_tap_bf16 +
				(uint64_t)(continuation->layer_index -
					SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_FIRST) *
				SPARK_DSV4_MODEL_HIDDEN_DIMENSION *
				SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
			if ( continuation->dspark_verify != 0u )
				tap_dst = (uint8_t *)slot->dspark_verify_tap_bf16 +
					(uint64_t)(continuation->layer_index -
						SPARK_DSV4_MODEL_DSPARK_TARGET_LAYER_FIRST) *
					(SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u) *
					SPARK_DSV4_MODEL_HIDDEN_DIMENSION *
					SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
			error = SparkDsv4LaunchDsparkTapMean(stream,slot->streams_bf16,
				tap_dst,continuation->rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,
				SPARK_DSV4_MODEL_HIDDEN_DIMENSION,state->multiprocessor_count);
			if ( error != cudaSuccess )
				fprintf(stderr,"dspark_tap_launch_failed error=%d tp_rank=%u\n",(int)error,state->tp_rank);
			if ( error == cudaSuccess )
				error = cudaStreamSynchronize(stream);
			if ( error != cudaSuccess )
				fprintf(stderr,"dspark_tap_async_failed error=%d tp_rank=%u layer=%u\n",(int)error,state->tp_rank,continuation->layer_index);
			status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
				"dspark_tap");
		}
		if ( status == SPARK_STATUS_OK )
		{
			continuation->side = 2u;
			read_ahead_payload = 0;
			read_ahead_bytes = 0u;
			if ( continuation->layer_index + 1u <
				state->first_layer_index + state->layer_count )
			{
				read_ahead_payload = state->layers[
					continuation->layer_index + 1u].hc.attn_fn_f32;
				read_ahead_bytes = SparkDsv4ModuleHcFunctionBytes();
			}
			else if ( state->participates_final_head != 0u )
			{
				read_ahead_payload = state->hc_head_fn_f32;
				read_ahead_bytes = SparkDsv4ModuleHcHeadFunctionBytes();
			}
			status = SparkDsv4ModuleReduceHiddenReadAhead(state,slot,
				slot->ffn_accum_bf16,continuation->rows,
				&continuation->collective,read_ahead_payload,read_ahead_bytes,
				0,0u);
			if ( status == SPARK_STATUS_OK )
				return;
		}
	}
	else if ( continuation->side == 2u )
	{
		if ( state->tp_degree > 1u )
			status = SPARK_STATUS_OK;
		else
		{
			layer_output = continuation->layer_index + 1u ==
				state->first_layer_index + state->layer_count ?
				continuation->output_streams_bf16 : slot->streams_bf16;
			error = SparkDsv4LaunchHcPost(stream,slot->ffn_accum_bf16,
				slot->residual_bf16,slot->post_f32,slot->comb_f32,layer_output,
				continuation->rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,
				SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
			status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
				"ffn_side");
		}
		if ( status == SPARK_STATUS_OK && continuation->layer_index + 1u <
			state->first_layer_index + state->layer_count )
		{
			continuation->layer_index++;
			continuation->input_streams_bf16 = slot->streams_bf16;
			status = SparkDsv4ModuleStartLayers(continuation);
			if ( status == SPARK_STATUS_OK )
				return;
		}
		else if ( status == SPARK_STATUS_OK )
		{
			status = SparkDsv4ModuleFinishFrameContinuation(continuation);
			if ( status == SPARK_STATUS_PENDING )
				return;
		}
	}
	SparkDsv4ModuleFinishContinuationTerminal(continuation,status);
}

static cudaError_t SparkDsv4ModuleProjectHead(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const void *streams_bf16, uint32_t *output_token_ids, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkDsv4LaunchHcMix(stream,streams_bf16,state->hc_head_fn_f32,slot->mixes_f32,rows,SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcHeadReduce(stream,streams_bf16,slot->mixes_f32,state->hc_head_scale_value,state->hc_head_base_f32,SPARK_DSV4_MODEL_HC_EPSILON,slot->reduced_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,state->final_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess && rows == 1u )
		error = SparkDsv4LaunchHeadCertifiedFp8B1Sharded(stream,
			slot->normalized_bf16,state->lm_head_weight_bf16,
			state->head_certified_fp8_payload,
			state->head_certified_fp8_scale_f32,
			state->head_certified_fp8_norm_f32,slot->head_certified_scratch,
			slot->head_candidate_ids_u32,slot->head_candidate_counts_u32,
			output_token_ids,slot->head_scores_f32,state->vocabulary_row_start,
			rows,state->vocabulary_rows_per_rank,
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	else if ( error == cudaSuccess )
		error = SparkDsv4LaunchHeadScreenedArgmaxSharded(stream,
			slot->normalized_bf16,state->lm_head_weight_bf16,
			state->head_shadow_payload,state->head_shadow_scale,
			state->head_error_norm_f32,slot->head_logits_bf16,
			slot->head_candidate_ids_u32,slot->head_candidate_counts_u32,
			output_token_ids,slot->head_scores_f32,state->vocabulary_row_start,
			rows,state->vocabulary_rows_per_rank,
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess && state->tp_degree > 1u )
		error = SparkDsv4LaunchHeadMaxlocPack(stream,slot->head_scores_f32,
			output_token_ids,slot->head_maxloc_u64,rows);
	return(error);
}

static SparkStatus SparkDsv4ModuleFinish(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const void *streams_bf16, uint32_t *host_output_tokens, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error = cudaSuccess;
	if ( state->owns_final_head != 0u )
	{
		if ( streams_bf16 == 0 || host_output_tokens == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		error = SparkDsv4ModuleProjectHead(state,slot,streams_bf16,slot->output_token_ids,rows);
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(host_output_tokens,slot->output_token_ids,(uint64_t)rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	}
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"head"));
}

static SparkStatus SparkDsv4ModuleLaunchTpProjectionIsland(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t layer_index,
	uint32_t rows)
{
	const SparkDsv4LayerWeights *layer;
	cudaStream_t stream;
	cudaError_t error;
	if ( state == 0 || slot == 0 || layer_index < state->first_layer_index ||
		layer_index >= state->first_layer_index + state->layer_count ||
		rows != SPARK_BATCH_BUCKET )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	layer = &state->layers[layer_index];
	stream = (cudaStream_t)slot->cuda_stream;
	error = cudaSuccess;
	/* A0 owns the stable token-id -> four-stream prologue on stage zero. */
	if ( layer_index == state->first_layer_index &&
		state->owns_embedding != 0u )
	{
		const void *embedded_streams = 0;
		SparkStatus begin_status = SparkDsv4ModuleBeginStreams(state,slot,0,
			rows,&embedded_streams);
		if ( begin_status != SPARK_STATUS_OK ||
			embedded_streams != slot->streams_bf16 )
			return(begin_status != SPARK_STATUS_OK ? begin_status :
				SPARK_STATUS_VALIDATION_FAILED);
	}
	/* Fold the prior layer's post-FFN boundary into every A island after A0. */
	if ( layer_index != state->first_layer_index )
		error = SparkDsv4LaunchHcPost(stream,slot->ffn_accum_bf16,
			slot->residual_bf16,slot->post_f32,slot->comb_f32,
			slot->streams_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleHcEnter(slot,slot->streams_bf16,
			layer->hc.attn_fn_f32,layer->hc.attn_scale_f32,
			layer->hc.attn_base_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,
			layer->attn_norm_bf16,slot->normalized_bf16,rows,
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION,
			SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunProjectionShards(state,slot,layer,
			SparkDsv4ModelLayerKind(layer_index),rows);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"tp_graph_projection_island"));
}

static SparkStatus SparkDsv4ModuleLaunchTpAttentionIsland(
	SparkDsv4ModuleState *state,SparkDsv4ModuleSlot *slot,
	uint32_t layer_index,uint32_t rows)
{
	cudaError_t error;
	if ( state == 0 || slot == 0 || layer_index < state->first_layer_index ||
		layer_index >= state->first_layer_index + state->layer_count ||
		rows != SPARK_BATCH_BUCKET )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	error = SparkDsv4ModuleRunAttentionProjected(state,slot,
		&state->layers[layer_index],layer_index,rows);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"tp_graph_attention_island"));
}

static SparkStatus SparkDsv4ModuleLaunchTpFfnIsland(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t layer_index,
	uint32_t rows)
{
	const SparkDsv4LayerWeights *layer;
	cudaStream_t stream;
	cudaError_t error;
	SparkStatus status;
	if ( state == 0 || slot == 0 || layer_index < state->first_layer_index ||
		layer_index >= state->first_layer_index + state->layer_count ||
		rows != SPARK_BATCH_BUCKET )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	layer = &state->layers[layer_index];
	stream = (cudaStream_t)slot->cuda_stream;
	error = SparkDsv4LaunchHcPost(stream,slot->delta_bf16,
		slot->residual_bf16,slot->post_f32,slot->comb_f32,
		slot->streams_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleHcEnter(slot,slot->streams_bf16,
			layer->hc.ffn_fn_f32,layer->hc.ffn_scale_f32,
			layer->hc.ffn_base_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,
			layer->ffn_norm_bf16,slot->normalized_bf16,rows,
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION,
			SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	status = error == cudaSuccess ? SparkDsv4ModuleRunMoe(state,slot,layer,
		layer_index,rows) : SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,
		error,"tp_graph_ffn_enter");
	return(status);
}

static SparkStatus SparkDsv4ModuleLaunchTpFinalIsland(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t rows)
{
	cudaStream_t stream;
	cudaError_t error;
	SparkStatus status;
	if ( state == 0 || slot == 0 || rows != SPARK_BATCH_BUCKET )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	stream = (cudaStream_t)slot->cuda_stream;
	error = SparkDsv4LaunchHcPost(stream,slot->ffn_accum_bf16,
		slot->residual_bf16,slot->post_f32,slot->comb_f32,
		slot->streams_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"tp_graph_final_ffn_side");
	if ( status == SPARK_STATUS_OK && state->participates_final_head != 0u )
	{
		error = SparkDsv4ModuleProjectHead(state,slot,slot->streams_bf16,
			slot->output_token_ids,rows);
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
			"tp_graph_local_head");
	}
	return(status);
}

static SparkDsv4TpGraphIsland *SparkDsv4ModuleTpGraphIsland(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t island_index)
{
	uint32_t slot_index;
	if ( state == 0 || slot == 0 || state->tp_graph_islands == 0 ||
		island_index >= state->tp_graph_islands_per_slot ||
		slot < state->slots || slot >= state->slots + state->pipeline_slot_count )
		return(0);
	slot_index = (uint32_t)(slot - state->slots);
	return(&state->tp_graph_islands[
		(uint64_t)slot_index * state->tp_graph_islands_per_slot + island_index]);
}

static SparkStatus SparkDsv4ModuleReplayTpIsland(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t island_index,
	uint32_t rows)
{
	SparkDsv4TpGraphIsland *island;
	if ( state == 0 || state->tp_degree <= 1u ||
		state->tp_graphs_sealed == 0u || rows != SPARK_BATCH_BUCKET ||
		state->tp_graph_islands_per_slot !=
		SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(state->layer_count) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	island = SparkDsv4ModuleTpGraphIsland(state,slot,island_index);
	if ( island == 0 || island->live == 0u || island->executable == 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,
		cudaGraphLaunch(island->executable,(cudaStream_t)slot->cuda_stream),
		"tp_graph_replay"));
}

static SparkStatus SparkDsv4ModuleLaunchTpIslandBody(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t island_index,
	uint32_t rows,
	uint32_t *kind_out,
	uint32_t *layer_index_out)
{
	uint32_t local_layer,required;
	if ( state == 0 || slot == 0 || kind_out == 0 || layer_index_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	required = SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(
		state->layer_count);
	if ( island_index >= required )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( island_index + 1u == required )
	{
		*kind_out = SPARK_DSV4_TP_GRAPH_ISLAND_FINAL;
		*layer_index_out = state->first_layer_index + state->layer_count - 1u;
		return(SparkDsv4ModuleLaunchTpFinalIsland(state,slot,rows));
	}
	local_layer = island_index / 3u;
	*layer_index_out = state->first_layer_index + local_layer;
	if ( island_index % 3u == 0u )
	{
		*kind_out = SPARK_DSV4_TP_GRAPH_ISLAND_PROJECTION;
		return(SparkDsv4ModuleLaunchTpProjectionIsland(state,slot,
			*layer_index_out,rows));
	}
	if ( island_index % 3u == 1u )
	{
		*kind_out = SPARK_DSV4_TP_GRAPH_ISLAND_ATTENTION;
		return(SparkDsv4ModuleLaunchTpAttentionIsland(state,slot,
			*layer_index_out,rows));
	}
	*kind_out = SPARK_DSV4_TP_GRAPH_ISLAND_FFN;
	return(SparkDsv4ModuleLaunchTpFfnIsland(state,slot,*layer_index_out,rows));
}

static SparkStatus SparkDsv4ModuleCaptureTpIsland(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t island_index)
{
	SparkDsv4TpGraphIsland *island;
	cudaGraph_t graph;
	cudaStream_t stream;
	cudaError_t begin_error,end_error,instantiate_error;
	SparkStatus status;
	uint32_t kind,layer_index;
	island = SparkDsv4ModuleTpGraphIsland(state,slot,island_index);
	if ( island == 0 || island->live != 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	stream = (cudaStream_t)slot->cuda_stream;
	graph = 0;
	begin_error = cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed);
	if ( begin_error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,begin_error,
			"tp_graph_begin_capture"));
	status = SparkDsv4ModuleLaunchTpIslandBody(state,slot,island_index,
		SPARK_BATCH_BUCKET,&kind,&layer_index);
	/* End every successfully-started capture, including an invalidated one. */
	end_error = cudaStreamEndCapture(stream,&graph);
	if ( status != SPARK_STATUS_OK || end_error != cudaSuccess || graph == 0 )
	{
		if ( graph != 0 )
			(void)cudaGraphDestroy(graph);
		return(status != SPARK_STATUS_OK ? status :
			SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,end_error,
				"tp_graph_end_capture"));
	}
	instantiate_error = cudaGraphInstantiate(&island->executable,graph,0ull);
	(void)cudaGraphDestroy(graph);
	if ( instantiate_error != cudaSuccess )
	{
		island->executable = 0;
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,
			instantiate_error,"tp_graph_instantiate"));
	}
	island->layer_index = layer_index;
	island->kind = kind;
	island->live = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModulePrewarmTpGraphs(
	SparkDsv4ModuleState *state)
{
	uint64_t total;
	uint32_t island_index,slot_index;
	SparkStatus status;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->tp_degree == 1u )
		return(SPARK_STATUS_OK);
	state->tp_graph_islands_per_slot =
		SparkDsv4ResidentDecodeStageGraphIslandsPerSlot(state->layer_count);
	if ( state->tp_graph_islands_per_slot == 0u ||
		state->graph_capacity != state->tp_graph_islands_per_slot ||
		SparkDsv4ResidentDecodeStageNativeTpWidthSupported(SPARK_BATCH_BUCKET) == 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	total = (uint64_t)state->pipeline_slot_count *
		state->tp_graph_islands_per_slot;
	if ( total == 0u || total > UINT32_MAX )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->tp_graph_islands = (SparkDsv4TpGraphIsland *)calloc((size_t)total,
		sizeof(*state->tp_graph_islands));
	if ( state->tp_graph_islands == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->tp_graph_island_count = (uint32_t)total;
	/* FinalizeLoad's qualification gates already synchronize uploaded state. */
	status = SPARK_STATUS_OK;
	for (slot_index=0u; status==SPARK_STATUS_OK &&
		slot_index<state->pipeline_slot_count; slot_index++)
	{
		state->slots[slot_index].cuda_stream = state->execution_stream;
		for (island_index=0u; status==SPARK_STATUS_OK &&
			island_index<state->tp_graph_islands_per_slot; island_index++)
			status = SparkDsv4ModuleCaptureTpIsland(state,
				&state->slots[slot_index],island_index);
	}
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (island_index=0u; island_index<state->tp_graph_island_count;
		island_index++)
		if ( state->tp_graph_islands[island_index].live == 0u ||
			state->tp_graph_islands[island_index].executable == 0 )
			return(SPARK_STATUS_VALIDATION_FAILED);
	/* No graph is launched during prewarm: capture cannot mutate KV state. */
	state->tp_graphs_sealed = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleRunPrefillHead(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4PrefillBatchView *prefill)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = cudaMemsetAsync(slot->output_token_ids,0,(uint64_t)prefill->active_sequence_count * sizeof(uint32_t),stream);
	if ( error == cudaSuccess && prefill->emit_count != 0u )
		error = cudaMemcpyAsync(slot->prefill_emit_rows_u32,
			prefill->emit_row_indices,
			(uint64_t)prefill->emit_count * sizeof(uint32_t),
			cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && prefill->emit_count != 0u )
		error = cudaMemcpyAsync(slot->row_lane_indices,prefill->emit_lane_indices,(uint64_t)prefill->emit_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && prefill->emit_count != 0u )
		error = SparkDsv4LaunchGatherHeadRows(stream,slot->streams_bf16,
			slot->prefill_emit_rows_u32,slot->residual_bf16,
			prefill->emit_count,SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS);
	if ( error == cudaSuccess && prefill->emit_count != 0u )
		error = SparkDsv4ModuleProjectHead(state,slot,slot->residual_bf16,slot->slot_counts,prefill->emit_count);
	if ( error == cudaSuccess && prefill->emit_count != 0u )
		error = SparkDsv4LaunchScatterHeadTokens(stream,slot->slot_counts,slot->row_lane_indices,slot->output_token_ids,prefill->emit_count);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->host_output_token_ids,slot->output_token_ids,(uint64_t)prefill->active_sequence_count * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"prefill_head"));
}

/* DSpark drive, submission-path (host syncs allowed): the draft forward
 * plus the sequential markov chain with per-rank greedy samples. The
 * cross-rank sample reduce and the verify/acceptance loop follow. */
static SparkStatus SparkDsv4ModuleDsparkDrive(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t lane_index,
	uint32_t anchor_token_id,
	uint64_t anchor_position)
{
	const uint32_t block = SPARK_DSV4_MODEL_DSPARK_SPEC_STEP;
	uint32_t rows_per_rank = state->vocabulary_rows_per_rank;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error = cudaSuccess;
	SparkStatus status;
	uint32_t index,prev_token = anchor_token_id;
	fprintf(stderr,"dspark_drive_entry tp_rank=%u lane=%u anchor=%u pos=%llu\n",state->tp_rank,lane_index,anchor_token_id,(unsigned long long)anchor_position);
	status = SparkDsv4ModuleRunDsparkDraft(state,slot,lane_index,
		anchor_token_id,anchor_position);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"dspark_draft_forward_failed status=%u tp_rank=%u\n",(uint32_t)status,state->tp_rank);
		return(status);
	}
	error = cudaStreamSynchronize((cudaStream_t)slot->cuda_stream);
	if ( error != cudaSuccess )
		fprintf(stderr,"dspark_draft_async_failed error=%d tp_rank=%u\n",(int)error,state->tp_rank);
	for (index = 0u; index < block && error == cudaSuccess; index++)
	{
		uint32_t host_prev = prev_token;
		error = cudaMemcpyAsync(slot->input_token_ids,&host_prev,
			sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchEmbeddingGather(stream,slot->input_token_ids,
				state->mtp.markov_w1.payload,slot->dspark_main_x_bf16,1u,
				SPARK_DSV4_MODEL_DSPARK_MARKOV_RANK);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchDsparkMarkovBiasAccum(stream,
				slot->dspark_logits_bf16,state->mtp.markov_w2.payload,
				slot->dspark_main_x_bf16,slot->dspark_logits_f32,
				state->vocabulary_row_start,rows_per_rank,
				SPARK_DSV4_MODEL_DSPARK_MARKOV_RANK,index,
				state->multiprocessor_count);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchDsparkArgmax(stream,
				slot->dspark_logits_f32 + (uint64_t)index * rows_per_rank,
				rows_per_rank,state->vocabulary_row_start,
				slot->dspark_draft_token_ids + index,
				slot->dspark_scores_f32 + index);
		/* The markov sample covers only this rank's lm_head shard:
		 * reduce the (score, token) pair across the TP group so every
		 * rank's chain input and verify staging see the same global
		 * draft token. Stream-ordered: pack -> allreduce -> unpack, then
		 * the existing D2H picks up the reduced token. */
		if ( error == cudaSuccess && state->tp_degree > 1u &&
			state->tp_device_collective_initialized != 0u )
		{
			SparkTpDeviceCollectiveSubmission submission;
			SparkStatus reduce_status;
			uint64_t ordinal;
			error = SparkDsv4LaunchHeadMaxlocPack(stream,
				slot->dspark_scores_f32 + index,
				slot->dspark_draft_token_ids + index,
				slot->dspark_maxloc_u64 + index,1u);
			if ( error == cudaSuccess )
			{
				ordinal = atomic_fetch_add_explicit(
					&state->tp_next_ordinal,1u,memory_order_relaxed);
				memset(&submission,0,sizeof(submission));
				submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
				submission.descriptor_bytes = sizeof(submission);
				submission.slot_index = (uint32_t)(slot - state->slots);
				submission.active_sequence_count = 1u;
				submission.flags =
					SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
				submission.ordinal = ordinal;
				submission.local_device = slot->dspark_maxloc_u64 + index;
				submission.full_device = slot->dspark_maxloc_u64 + index;
				submission.cuda_stream = stream;
				submission.completion_function =
					SparkDsv4ModuleDsparkDraftCollectiveComplete;
				submission.completion_context = 0;
				reduce_status = SparkTpDeviceCollectiveSubmitU64Max(
					&state->tp_device_collective,&submission);
				if ( reduce_status == SPARK_STATUS_OK )
					reduce_status = SparkDsv4ModuleDsparkWaitCollectiveFree(
						state,ordinal);
				if ( reduce_status != SPARK_STATUS_OK )
				{
					fprintf(stderr,"dspark_draft_reduce_failed status=%u tp_rank=%u index=%u\n",(uint32_t)reduce_status,state->tp_rank,index);
					return(reduce_status);
				}
				error = SparkDsv4LaunchHeadMaxlocUnpack(stream,
					slot->dspark_maxloc_u64 + index,
					slot->dspark_draft_token_ids + index,1u);
			}
		}
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(&prev_token,
				slot->dspark_draft_token_ids + index,sizeof(uint32_t),
				cudaMemcpyDeviceToHost,stream);
		if ( error == cudaSuccess )
			error = cudaStreamSynchronize(stream);
	}
	if ( error != cudaSuccess )
		fprintf(stderr,"dspark_markov_chain_failed error=%d tp_rank=%u index=%u\n",(int)error,state->tp_rank,index);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"dspark_markov_chain"));
}

static SparkStatus SparkDsv4ModuleRunDsparkHead(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t block)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	/* hc head mix + reduce + final norm over the 5 draft positions */
	error = SparkDsv4LaunchHcMix(stream,slot->dspark_x_bf16,
		state->mtp.hc_head_fn_f32,slot->mixes_f32,block,
		SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS,
		SPARK_DSV4_MODEL_HC_STREAM_COUNT,
		SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error != cudaSuccess )
		fprintf(stderr,"dspark_head_fail stage=hc_mix error=%d\n",(int)error);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcHeadReduce(stream,slot->dspark_x_bf16,
			slot->mixes_f32,state->hc_head_scale_value,
			state->mtp.hc_head_base_f32,SPARK_DSV4_MODEL_HC_EPSILON,
			slot->reduced_bf16,block,SPARK_DSV4_MODEL_HC_STREAM_COUNT,
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	if ( error != cudaSuccess )
		fprintf(stderr,"dspark_head_fail stage=hc_head_reduce error=%d\n",(int)error);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,
			state->mtp.final_norm_weight_bf16,slot->normalized_bf16,block,
			SPARK_DSV4_MODEL_HIDDEN_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error != cudaSuccess )
		fprintf(stderr,"dspark_head_fail stage=rms_norm error=%d\n",(int)error);
	/* base logits over this rank's lm_head shard */
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&state->lm_head_view,
			slot->normalized_bf16,slot->dspark_logits_bf16,block);
	if ( error != cudaSuccess )
		fprintf(stderr,"dspark_head_fail stage=linear error=%d\n",(int)error);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
		"dspark_head"));
}

#define DSPARK_S0_TRACE(tag) do { \
	cudaError_t trace_error_ = cudaStreamSynchronize((cudaStream_t)slot->cuda_stream); \
	if ( trace_error_ != cudaSuccess ) \
		fprintf(stderr,"dspark_s0_hang_after=" tag " error=%d tp_rank=%u\n",(int)trace_error_,state->tp_rank); \
	else \
		fprintf(stderr,"dspark_s0_ok=" tag " tp_rank=%u\n",state->tp_rank); \
} while (0)

static SparkStatus SparkDsv4ModuleRunDsparkDraft(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t lane_index,
	uint32_t anchor_token_id,
	uint64_t anchor_position)
{
	const uint32_t block = SPARK_DSV4_MODEL_DSPARK_SPEC_STEP;
	const uint32_t streams = SPARK_DSV4_MODEL_HC_STREAM_COUNT;
	const uint32_t dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error = cudaSuccess;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t stage,index;

	/* stage-0 prelude: main_x = main_norm(main_proj(concat(tap40..42))) */
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&state->mtp.main_proj,
			slot->dspark_tap_bf16,slot->dspark_main_x_bf16,1u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->dspark_main_x_bf16,
			state->mtp.main_norm_weight_bf16,slot->dspark_main_x_bf16,1u,
			dim,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	/* block input: embed([anchor, noise x (block-1)]) expanded x streams */
	for (index = 0u; index < block; index++)
		slot->dspark_host_input_token_ids[index] = index == 0u ? anchor_token_id :
			SPARK_DSV4_MODEL_DSPARK_NOISE_TOKEN_ID;
	for (index = 0u; index < block; index++)
		slot->dspark_host_row_positions[index] = anchor_position + 1u + index;
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->input_token_ids,
			slot->dspark_host_input_token_ids,block * sizeof(uint32_t),
			cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->dspark_row_positions,
			slot->dspark_host_row_positions,block * sizeof(uint64_t),
			cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchEmbeddingGather(stream,slot->input_token_ids,
			state->token_embedding_bf16,slot->dspark_q_attn_bf16,block,dim);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchExpandStreams(stream,slot->dspark_q_attn_bf16,
			slot->dspark_x_bf16,block,streams,dim,state->multiprocessor_count);
	DSPARK_S0_TRACE("prelude");
	for (stage = 0u; error == cudaSuccess &&
		stage < SPARK_DSV4_MODEL_MTP_LAYER_COUNT; stage++)
	{
		const SparkDsv4LayerWeights *layer = &state->mtp_layers[stage];
		/* hc_pre (attn) + attn norm */
		error = SparkDsv4ModuleHcEnter(slot,slot->dspark_x_bf16,
			layer->hc.attn_fn_f32,layer->hc.attn_scale_f32,
			layer->hc.attn_base_f32,block);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,
				layer->attn_norm_bf16,slot->normalized_bf16,block,dim,
				SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
		/* block q/kv (full width: the MTP range is packed unsharded) */
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchFp8LinearPair(stream,&layer->attn.wq_a,
				&layer->attn.wkv,slot->normalized_bf16,slot->delta_bf16,
				slot->kv_bf16,block);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchRmsNorm(stream,slot->delta_bf16,
				layer->attn.q_norm_weight_bf16,slot->qr_bf16,block,
				SPARK_DSV4_MODEL_QUERY_LORA_RANK,
				SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchLinear(stream,&layer->attn.wq_b,
				slot->qr_bf16,slot->dspark_q_attn_bf16,block);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchQueryHeadRms(stream,slot->dspark_q_attn_bf16,
				block,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,
				SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,
				SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchRope(stream,slot->dspark_q_attn_bf16,
				state->base_freqs_f32,slot->dspark_row_positions,block,
				SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,
				SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,
				SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchKvPost(stream,slot->kv_bf16,
				layer->attn.kv_norm_weight_bf16,state->base_freqs_f32,
				slot->dspark_row_positions,block,
				SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,
				SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,
				SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION -
					SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,
				SPARK_DSV4_MODEL_KV_QUANT_BLOCK,
				SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
		DSPARK_S0_TRACE("qkv");
		/* draft attention: window ring + block kvs, non-causal, sink in
		 * the denominator */
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchDsparkAttention(stream,
				slot->dspark_q_attn_bf16,
				state->dspark_ring_bf16,state->dspark_ring_lane_stride,
				lane_index,slot->kv_bf16,layer->attn.sink_f32,
				1.0f / sqrtf((float)SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION),
				slot->dspark_q_attn_bf16,block,
				SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,
				SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,
				SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchRope(stream,slot->dspark_q_attn_bf16,
				state->base_freqs_f32,slot->dspark_row_positions,block,
				SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,
				SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,
				SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,1u);
		/* output composition (full 8 groups) + wo_b */
		if ( error == cudaSuccess )
		{
			SparkDsv4LinearView wo_a = layer->attn.wo_a;
			wo_a.rows = SPARK_DSV4_MODEL_OUTPUT_LORA_RANK;
			wo_a.columns = SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION /
				SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT;
			error = SparkDsv4LaunchStridedLinear(stream,&wo_a,wo_a.payload,
				(const uint8_t *)wo_a.scale_data,0ull,0ull,
				slot->dspark_q_attn_bf16,SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION,
				0u,wo_a.columns,slot->dspark_o_ranks_bf16,
				(uint64_t)SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT *
					SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,
				0u,SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,
				SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT,block);
		}
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchLinear(stream,&layer->attn.wo_b,
				slot->dspark_o_ranks_bf16,slot->delta_bf16,block);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchHcPost(stream,slot->delta_bf16,
				slot->residual_bf16,slot->post_f32,slot->comb_f32,
				slot->dspark_x_bf16,block,streams,dim);
		DSPARK_S0_TRACE("attn_compose");
		/* hc_pre (ffn) + ffn norm + MoE */
		if ( error == cudaSuccess )
			error = SparkDsv4ModuleHcEnter(slot,slot->dspark_x_bf16,
				layer->hc.ffn_fn_f32,layer->hc.ffn_scale_f32,
				layer->hc.ffn_base_f32,block);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,
				layer->ffn_norm_bf16,slot->normalized_bf16,block,dim,
				SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
		{
			status = SparkDsv4ModuleRunMoe(state,slot,layer,
				SPARK_DSV4_STAGEPACK_MTP_LAYER(stage),block);
			if ( status == SPARK_STATUS_OK )
				status = SparkDsv4ModuleDsparkReduceMoeOutput(
					state,slot,block);
			if ( status != SPARK_STATUS_OK )
				break;
		}
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchHcPost(stream,slot->ffn_accum_bf16,
				slot->residual_bf16,slot->post_f32,slot->comb_f32,
				slot->dspark_x_bf16,block,streams,dim);
		DSPARK_S0_TRACE("moe");
		/* the target position's kv enters this draft layer's ring at
		 * anchor_position % window */
		{
			uint32_t host_lane = lane_index;
			uint64_t host_position = anchor_position;
			error = cudaMemcpyAsync(slot->row_lane_indices,&host_lane,
				sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
			if ( error == cudaSuccess )
				error = cudaMemcpyAsync(slot->row_positions,&host_position,
					sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
		}
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchFp8LinearPair(stream,&layer->attn.wq_a,
				&layer->attn.wkv,slot->dspark_main_x_bf16,slot->delta_bf16,
				slot->kv_bf16,1u);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchKvPost(stream,slot->kv_bf16,
				layer->attn.kv_norm_weight_bf16,state->base_freqs_f32,
				slot->row_positions,1u,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,
				SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,
				SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION -
					SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,
				SPARK_DSV4_MODEL_KV_QUANT_BLOCK,
				SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchCacheScatter(stream,slot->kv_bf16,0,
				(uint8_t *)state->dspark_ring_bf16 +
					(uint64_t)stage *
						state->dspark_ring_layer_stride *
						SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,
				state->dspark_ring_lane_stride,slot->row_lane_indices,
				slot->row_positions,1u,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,
				anchor_position % SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,0u,
				SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS);
		error = cudaStreamSynchronize((cudaStream_t)slot->cuda_stream);
		if ( error != cudaSuccess )
			fprintf(stderr,"dspark_stage_sync_failed error=%d tp_rank=%u stage=%u\n",(int)error,state->tp_rank,stage);
	}
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
			"dspark_draft_forward"));
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleRunDsparkHead(state,slot,block);
	return(status);
}

static SparkStatus SparkDsv4ModuleRunFrame(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	SparkModelDriverFrame *frame,
	const SparkDsv4ResidentDecodeStageFrameContext *context)
{
	const SparkDsv4PrefillBatchView *prefill;
	const void *input_streams_bf16;
	void *output_streams_bf16;
	uint32_t rows,is_prefill;
	SparkStatus status;
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? context->prefill_batch : 0;
	is_prefill = prefill != 0 ? 1u : 0u;
	rows = prefill != 0 ? prefill->row_count : context->decode_batch->row_count;
	if ( slot->dspark_verify_rows != 0u )
		rows = slot->dspark_verify_rows;
	/* A full-width prefill wave has exactly one row per active lane.  Its
	 * causal attention is therefore the decode attention operation, and all
	 * row values already live in slot-owned staging before this function.
	 * Drop the stack-owned view before any asynchronous TP continuation.
	 * The DSpark bucket-8 padding stages single-lane prefills to bucket
	 * width, so rows == SPARK_BATCH_BUCKET is the only gate left. */
	if ( prefill != 0 && state->tp_degree > 1u )
	{
		if ( rows != SPARK_BATCH_BUCKET )
			return(SPARK_STATUS_UNSUPPORTED);
		prefill = 0;
	}
	if ( prefill == 0 && state->tp_degree > 1u && state->owns_embedding != 0u )
	{
		input_streams_bf16 = slot->streams_bf16;
		status = SPARK_STATUS_OK;
	}
	else
		status = SparkDsv4ModuleBeginStreams(state,slot,context,rows,
			&input_streams_bf16);
	if ( status == SPARK_STATUS_OK && prefill == 0 && state->tp_degree > 1u &&
		input_streams_bf16 != slot->streams_bf16 )
	{
		status = SparkDsv4ModuleBounceBoundary(slot->streams_bf16,
			input_streams_bf16,rows,(cudaStream_t)slot->cuda_stream,
			"tp_graph_boundary_in");
		input_streams_bf16 = slot->streams_bf16;
	}
	if ( status == SPARK_STATUS_OK && prefill != 0 )
	{
		output_streams_bf16 = state->pp_stage_index + 1u <
			state->pp_stage_count ? context->hidden_output_bf16 :
			slot->streams_bf16;
		status = SparkDsv4ModuleRunLocalLayers(state,slot,input_streams_bf16,
			output_streams_bf16,prefill,rows);
		if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
			status = SparkDsv4ModuleRunPrefillHead(state,slot,prefill);
		return(status);
	}
	if ( status == SPARK_STATUS_OK )
	{
		SparkDsv4TpFrameContinuation *continuation;

		output_streams_bf16 = state->pp_stage_index + 1u < state->pp_stage_count ? context->hidden_output_bf16 : slot->streams_bf16;
		continuation = slot->tp_continuation;
		if ( continuation == 0 || continuation->active != 0u )
			return(SPARK_STATUS_BUSY);
		memset(continuation,0,sizeof(*continuation));
		continuation->state = state;
		continuation->slot = slot;
		continuation->input_streams_bf16 = input_streams_bf16;
		continuation->output_streams_bf16 = output_streams_bf16;
		continuation->rows = rows;
		continuation->layer_index = state->first_layer_index;
		continuation->active = 1u;
		/* DSpark verify frames replace the engine's sequential chain
		 * (tokens_per_sequence) with ONE parallel 8-row verify forward;
		 * the acceptance loop at the head decides how many of the staged
		 * rows become emitted tokens. */
		continuation->dspark_verify = is_prefill == 0u &&
			slot->dspark_verify_rows != 0u ? 1u : 0u;
		continuation->chain_step_count = is_prefill != 0u ? 1u :
			(continuation->dspark_verify != 0u ? 1u :
			 frame->tokens_per_sequence);
		status = SparkDsv4ModuleValidateResidentChain(continuation);
		if ( status == SPARK_STATUS_OK )
			status = SparkDsv4ModuleStartLayers(continuation);
		if ( status != SPARK_STATUS_OK )
			continuation->active = 0u;
		else
			status = SPARK_STATUS_PENDING;
	}
	return(status);
}

static SparkStatus SparkDsv4ModuleFinishFrameContinuation(
	SparkDsv4TpFrameContinuation *continuation)
{
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	SparkStatus status;

	if ( continuation == 0 || continuation->state == 0 ||
		continuation->slot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state = continuation->state;
	slot = continuation->slot;
	status = SPARK_STATUS_OK;
	if ( state->tp_degree > 1u )
	{
		status = SparkDsv4ModuleReplayTpIsland(state,slot,
			3u * state->layer_count,continuation->rows);
		if ( status == SPARK_STATUS_OK &&
			continuation->dspark_verify != 0u &&
			state->participates_final_head != 0u )
		{
			uint32_t local_tokens[SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u];
			cudaError_t probe_error = cudaMemcpyAsync(local_tokens,
				slot->output_token_ids,(uint64_t)(SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u) * sizeof(uint32_t),
				cudaMemcpyDeviceToHost,(cudaStream_t)slot->cuda_stream);
			if ( probe_error == cudaSuccess )
				probe_error = cudaStreamSynchronize((cudaStream_t)slot->cuda_stream);
			if ( probe_error == cudaSuccess )
				fprintf(stderr,"dspark_head_local tp_rank=%u tokens=%u,%u,%u,%u,%u,%u,%u,%u rows=%u\n",state->tp_rank,local_tokens[0],local_tokens[1],local_tokens[2],local_tokens[3],local_tokens[4],local_tokens[5],local_tokens[6],local_tokens[7],continuation->rows);
			else
				fprintf(stderr,"dspark_head_probe_failed error=%d\n",(int)probe_error);
		}
		if ( status == SPARK_STATUS_OK &&
			state->participates_final_head != 0u )
		{
			status = SparkDsv4ModuleReduceHeadMax(continuation);
			if ( status == SPARK_STATUS_OK )
				status = SPARK_STATUS_PENDING;
		}
		else if ( status == SPARK_STATUS_OK &&
			state->pp_stage_index + 1u < state->pp_stage_count )
			status = SparkDsv4ModuleBounceBoundary(
				continuation->output_streams_bf16,slot->streams_bf16,
				continuation->rows,(cudaStream_t)slot->cuda_stream,
				"tp_graph_boundary_out");
	}
	else if ( state->owns_final_head != 0u )
		status = SparkDsv4ModuleFinish(state,slot,slot->streams_bf16,
			slot->host_output_token_ids,continuation->rows);
	return(status);
}

/*
 * Decode-step CUDA graphs, per docs/archive/PERF_ROADMAP_2026-08-01.md D10/D1: the
 * eager step costs hundreds of driver launches and at cohort width 1 the GPU
 * idles between them. A decode frame's launch SHAPE depends only on the
 * pipeline slot (every buffer above is per-slot fixed) and the lane count
 * (every grid derives from rows), so one recording per (slot, rows) pair
 * replays every later frame of that shape: token ids, lane indices, and
 * positions ride the slot's pinned staging through the two recorded H2D
 * copies, and the caches are indexed on device. The stage has no
 * position-dependent host branch - the topk column count and the window
 * ring are capacity constants - so no context term belongs in the key.
 *
 * The frame's boundary buffers are the one remaining per-frame address:
 * rather than key on daemon pointers, the graphed step bounces the hidden
 * input into the slot's own stream buffer ahead of the graph and bounces the
 * last stage's output back after it, one contiguous D2D copy each way, so
 * the recording only ever names module-owned addresses.
 *
 * Fail-closed: any capture or replay-launch failure fails the frame; a replay
 * never substitutes for a shape it was not recorded from.
 */
static SparkStatus SparkDsv4ModuleBounceBoundary(void *destination, const void *source, uint32_t rows, cudaStream_t stream, const char *site)
{
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,cudaMemcpyAsync(destination,source,(uint64_t)rows * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,stream),site));
}

// The exact launch sequence a decode graph records.
static SparkStatus SparkDsv4ModuleRunCapturedDecode(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, uint32_t rows)
{
	const void *input_streams_bf16;
	SparkStatus status;

	if ( state->tp_degree != 1u )
		return(SPARK_STATUS_UNSUPPORTED);
	input_streams_bf16 = slot->streams_bf16;
	status = SparkDsv4ModuleStageRowCopies(state,slot,rows,rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleInitializeFramePages(state,slot,rows);
	if ( status == SPARK_STATUS_OK && state->owns_embedding != 0u )
		status = SparkDsv4ModuleBeginStreams(state,slot,0,rows,
			&input_streams_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleRunLocalLayers(state,slot,
			input_streams_bf16,slot->streams_bf16,0,rows);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkDsv4ModuleFinish(state,slot,slot->streams_bf16,
			slot->host_output_token_ids,rows);
	return(status);
}

static SparkStatus SparkDsv4ModuleCaptureDecode(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const LmGraphKey *key, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	LmGraphEntry *entry;
	SparkStatus status;
	int32_t graph_status;
	graph_status = LmGraphBeginCapture(stream);
	if ( graph_status != LM_GRAPH_OK )
	{
		state->graph_sealed = 1u;
		fprintf(stderr,"%s graph_capture_begin_failed status=%d\n",
			SPARK_DSV4_MODULE_TAG,graph_status);
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	status = SparkDsv4ModuleRunCapturedDecode(state,slot,rows);
	graph_status = LmGraphEndCapture(&state->graph_cache,key,stream);
	if ( status == SPARK_STATUS_OK && graph_status == LM_GRAPH_OK )
	{
		// Captured work only recorded; replay it so this frame executes.
		graph_status = LmGraphReplay(&state->graph_cache,key,stream);
		if ( graph_status == LM_GRAPH_OK )
			return(SPARK_STATUS_OK);
		state->graph_sealed = 1u;
		fprintf(stderr,"%s graph_first_replay_failed status=%d\n",
			SPARK_DSV4_MODULE_TAG,graph_status);
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	// The attempt failed. If EndCapture still instantiated a recording made
	// with a failed launch inside, that recording has a hole: retire the
	// entry rather than risk replaying a short sequence.
	entry = LmGraphFind(&state->graph_cache,key);
	if ( entry != 0 && status != SPARK_STATUS_OK )
	{
		(void)cudaGraphExecDestroy(entry->executable);
		entry->live = 0;
	}
	state->graph_sealed = 1u;
	if ( status != SPARK_STATUS_OK )
		return(status);
	fprintf(stderr,"%s graph_capture_end_failed status=%d\n",
		SPARK_DSV4_MODULE_TAG,graph_status);
	return(SPARK_STATUS_INTERNAL_ERROR);
}

static SparkStatus SparkDsv4ModuleRunGraphedDecode(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t slot_index,
	SparkModelDriverFrame *frame,
	const SparkDsv4ResidentDecodeStageFrameContext *context,
	uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	LmGraphKey key;
	SparkStatus status;
	int32_t graph_status;
	// The pinned-staging fill is host work the recording cannot contain, and
	// the recorded copies read it at replay time: it runs every frame, first.
	status = SparkDsv4ModuleStageRowValues(state,slot,(const uint32_t *)frame->buffers[0].address,context->decode_batch->row_lane_indices,context->decode_batch->row_positions,rows);
	if ( status == SPARK_STATUS_OK && state->owns_embedding == 0u )
		status = SparkDsv4ModuleBounceBoundary(slot->streams_bf16,context->hidden_input_bf16,rows,stream,"graph_boundary_in");
	if ( status != SPARK_STATUS_OK )
		return(status);
	key.rows = rows;
	key.layer_kind = slot_index;
	key.format = 0u;
	key.sparse = 0u;
	key.context_bucket = LmGraphContextBucket(state->max_sequence_positions,SPARK_DSV4_MODEL_MAX_POSITIONS);
	graph_status = LmGraphReplay(&state->graph_cache,&key,stream);
	if ( graph_status == LM_GRAPH_OK )
		status = SPARK_STATUS_OK;
	else if ( graph_status == LM_GRAPH_ERR_SHAPE && state->graph_sealed == 0u )
		status = SparkDsv4ModuleCaptureDecode(state,slot,&key,rows);
	else
	{
		fprintf(stderr,"%s graph_replay_failed status=%d sealed=%u rows=%u\n",
			SPARK_DSV4_MODULE_TAG,graph_status,state->graph_sealed,rows);
		status = graph_status == LM_GRAPH_ERR_SHAPE ?
			SPARK_STATUS_VALIDATION_FAILED : SPARK_STATUS_INTERNAL_ERROR;
	}
	if ( status == SPARK_STATUS_OK && state->pp_stage_index + 1u < state->pp_stage_count )
		status = SparkDsv4ModuleBounceBoundary(context->hidden_output_bf16,slot->streams_bf16,rows,stream,"graph_boundary_out");
	return(status);
}

static SparkStatus SparkDsv4ModuleCopyCacheLanes(
	SparkDsv4AsyncCompletion *async,
	const SparkModelDriverFrame *frame,
	const SparkDsv4ResidentDecodeStageFrameContext *context)
{
	const uint64_t *positions,*sequences;
	SparkModelDriverCacheLane *destination;
	const SparkModelDriverCacheLane *source;
	uint32_t index,prefill;
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u;
	positions = prefill != 0u ? context->prefill_batch->row_positions :
		context->decode_batch->row_positions;
	sequences = prefill != 0u ? context->prefill_batch->row_sequence_ids :
		context->decode_batch->row_sequence_ids;
	for (index=0u; index<async->lane_count; index++)
	{
		destination = &async->cache_lanes[index];
		if ( frame->cache_lane_count == 0u )
		{
			memset(destination,0,sizeof(*destination));
			destination->sequence_id = sequences[index];
			destination->request_generation = context->request_generation;
			destination->step_generation = context->step_generation;
			destination->sequence_position = positions[index];
			destination->resident_sequence_slot = async->lane_indices[index];
			destination->context_token_count =
				(uint32_t)async->lane_next_positions[index];
			continue;
		}
		source = &frame->cache_lanes[index];
		if ( SparkModelDriverCacheLaneIsValid(source) == 0u ||
			source->sequence_id != sequences[index] ||
			source->sequence_id != async->lane_sequence_ids[index] ||
			source->sequence_position != positions[index] ||
			source->resident_sequence_slot != async->lane_indices[index] ||
			source->context_token_count != async->lane_next_positions[index] )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		*destination = *source;
	}
	async->cache_lane_count = async->lane_count;
	return(SPARK_STATUS_OK);
}

static void SparkDsv4ModuleBuildFrameCacheAdmission(
	const SparkModelDriverFrame *frame,
	const SparkDsv4ResidentDecodeStageFrameContext *context,
	const SparkModelDriverCacheLane *cache_lanes,
	SparkModelDriverAdmissionRequest *request)
{
	memset(request,0,sizeof(*request));
	request->descriptor_bytes = sizeof(*request);
	request->program_id = frame->program_id;
	request->submission_id = context->submission_id;
	request->control_generation = context->control_generation;
	request->transaction_id = context->transaction_id;
	request->request_generation = context->request_generation;
	request->step_generation = context->step_generation;
	request->request_id = frame->request_id;
	request->sequence_id = frame->sequence_id;
	request->sequence_position = frame->sequence_position;
	request->deadline_time_ns = frame->deadline_time_ns;
	request->active_slot_count = frame->active_slot_count;
	request->new_token_count = frame->new_token_count;
	request->priority = frame->priority;
	request->frame_flags = frame->flags;
	request->cache_lane_count = frame->cache_lane_count;
	request->cache_lanes = frame->cache_lane_count != 0u ? cache_lanes : 0;
	request->residency = frame->residency;
}

static const SparkDsv4ResidentDecodeStageFrameContext *
SparkDsv4ModuleCacheAdmissionFrameContext(
	const SparkModelDriverFrame *frame)
{
	const SparkDsv4ResidentDecodeStageFrameContext *context;
	if ( frame == 0 || frame->cache_lane_count == 0u ||
		frame->cache_lane_count >
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT ||
		frame->cache_lanes == 0 || frame->user_context == 0 )
		return(0);
	context = (const SparkDsv4ResidentDecodeStageFrameContext *)
		frame->user_context;
	/* Only a complete, ABI-exact context can name prepared ownership.  Shape,
	 * view, boundary, and continuity fields intentionally are not accepted as
	 * substitutes for the five control-plane identity fields. */
	if ( context->abi_version !=
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ||
		context->descriptor_bytes < (uint32_t)sizeof(*context) ||
		context->submission_id == 0u || context->control_generation == 0u ||
		context->transaction_id == 0u || context->request_generation == 0u ||
		context->step_generation == 0u )
		return(0);
	return(context);
}

static SparkStatus SparkDsv4ModuleReleaseCommittedCacheAdmission(
	SparkDsv4ModuleState *state,
	const SparkModelDriverFrame *frame,
	const SparkDsv4ResidentDecodeStageFrameContext *context)
{
	SparkModelDriverAdmissionRequest request;
	SparkDsv4PreparedCacheAdmission *prepared;
	SparkStatus status;
	if ( state == 0 || frame == 0 || context == 0 ||
		frame->cache_lane_count == 0u || frame->cache_lane_count >
		SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT ||
		frame->cache_lanes == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkDsv4ModuleBuildFrameCacheAdmission(frame,context,
		frame->cache_lanes,&request);
	if ( pthread_mutex_lock(&state->cache_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	status = SPARK_STATUS_OK;
	prepared = SparkDsv4ModuleFindCacheAdmission(state,&request,0);
	if ( prepared != 0 &&
		prepared->state == SPARK_DSV4_PREPARED_CACHE_COMMITTED )
	{
		if ( SparkDsv4ModuleCacheAdmissionRequestMatches(prepared,&request) == 0u )
			status = SPARK_STATUS_VALIDATION_FAILED;
		else
			SparkDsv4ModuleClearCacheAdmission(state,prepared,1u);
	}
	(void)pthread_mutex_unlock(&state->cache_mutex);
	return(status);
}

static SparkStatus SparkDsv4ModulePrepareAsync(
	SparkDsv4ModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkDsv4ResidentDecodeStageFrameContext *context,
	uint32_t slot_index,
	uint32_t lane_count,
	uint32_t row_count,
	uint32_t emitted_token_count,
	const uint32_t *lane_indices,
	const uint64_t *lane_sequence_ids,
	const uint64_t *lane_next_positions)
{
	SparkDsv4AsyncCompletion *async;
	uint32_t index;
	SparkStatus status;
	async = &state->completions[slot_index];
	memset(async,0,sizeof(*async));
	async->prepared_cache_admission_index = UINT32_MAX;
	async->state = state;
	async->completion_function = frame->completion_function;
	async->completion_context = frame->completion_context;
	async->slot_index = slot_index;
	async->lane_count = lane_count;
	async->row_count = row_count;
	async->emitted_token_count = emitted_token_count;
	async->tokens_per_sequence = frame->tokens_per_sequence;
	async->requires_prepared_cache_admission = frame->cache_lane_count != 0u ? 1u : 0u;
	for (index=0u; index<lane_count; index++)
	{
		async->lane_indices[index] = lane_indices[index];
		async->lane_sequence_ids[index] = lane_sequence_ids[index];
		async->lane_next_positions[index] = lane_next_positions[index];
	}
	status = SparkDsv4ModuleCopyCacheLanes(async,frame,context);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memcpy(async->admission_cache_lanes,async->cache_lanes,
		(uint64_t)async->cache_lane_count *
		sizeof(async->admission_cache_lanes[0]));
	SparkDsv4ModuleBuildFrameCacheAdmission(frame,context,
		async->admission_cache_lanes,&async->cache_admission);
	async->output_token_destination = state->owns_final_head != 0u ? (uint32_t *)frame->buffers[1u].address : 0;
	async->completion.request_id = frame->request_id;
	async->completion.sequence_id = frame->sequence_id;
	async->completion.sequence_position = frame->sequence_position;
	async->completion.program_id = frame->program_id;
	async->completion.driver_dispatch_slot = frame->driver_dispatch_slot;
	async->completion.accepted_token_count = frame->new_token_count;
	async->completion.tokens_per_sequence = frame->tokens_per_sequence;
	async->completion.status = SPARK_STATUS_OK;
	async->completion.residency = frame->residency;
	async->completion.host_staging_bytes = (uint64_t)row_count * (sizeof(uint32_t) * (2u + state->owns_embedding) + sizeof(uint64_t) * 3u) + (uint64_t)lane_count * sizeof(uint32_t) * state->owns_final_head;
	async->completion.device_memcpy_bytes = async->completion.host_staging_bytes + ((state->owns_final_head != 0u && (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u) ? (uint64_t)emitted_token_count * 2u * sizeof(uint32_t) : 0u);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleAbortCacheLanesLocked(
	SparkDsv4AsyncCompletion *async)
{
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	SparkStatus result,status;
	uint32_t index;
	state = async->state;
	slot = &state->slots[async->slot_index];
	result = SPARK_STATUS_OK;
	if ( async->aborted_cache_lane_count < async->prepared_cache_lane_count )
		async->aborted_cache_lane_count = async->prepared_cache_lane_count;
	for (index=0u; index<async->prepared_cache_lane_count; index++)
	{
		status = SparkDsv4PagedCacheUnpinLane(&state->paged_cache,
			slot->host_logical_page_table +
			(uint64_t)index * state->paged_cache.lane_page_capacity,
			async->prepared_cache_lanes[index].logical_page_count);
		if ( result == SPARK_STATUS_OK && status != SPARK_STATUS_OK )
			result = status;
		status = SparkDsv4PagedCacheAbortLane(&state->paged_cache,
			&async->cache_lanes[index]);
		if ( result == SPARK_STATUS_OK && status != SPARK_STATUS_OK )
			result = status;
	}
	async->prepared_cache_lane_count = 0u;
	return(result);
}

static SparkStatus SparkDsv4ModuleRollbackCacheLanesLocked(
	SparkDsv4AsyncCompletion *async)
{
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	SparkStatus result,status;
	uint32_t index;
	state = async->state;
	slot = &state->slots[async->slot_index];
	result = SPARK_STATUS_OK;
	for (index=0u; index<async->prepared_cache_lane_count; index++)
	{
		status = SparkDsv4PagedCacheUnpinLane(&state->paged_cache,
			slot->host_logical_page_table +
			(uint64_t)index * state->paged_cache.lane_page_capacity,
			async->prepared_cache_lanes[index].logical_page_count);
		if ( result == SPARK_STATUS_OK && status != SPARK_STATUS_OK )
			result = status;
		status = SparkDsv4PagedCacheRollbackLane(&state->paged_cache,
			&async->cache_lanes[index],&async->prepared_cache_lanes[index]);
		if ( result == SPARK_STATUS_OK && status != SPARK_STATUS_OK )
			result = status;
	}
	async->prepared_cache_lane_count = 0u;
	return(result);
}

static SparkStatus SparkDsv4ModuleAppendPageTableUpdates(
	SparkDsv4AsyncCompletion *async,
	uint32_t lane_ordinal)
{
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	uint32_t page,page_count,physical;
	uint64_t table_index,update_capacity;
	state = async->state;
	slot = &state->slots[async->slot_index];
	page_count = async->prepared_cache_lanes[lane_ordinal].logical_page_count;
	update_capacity = (uint64_t)async->cache_lane_count *
		state->paged_cache.lane_page_capacity;
	for (page=0u; page<page_count; page++)
	{
		table_index = (uint64_t)async->cache_lanes[lane_ordinal].resident_sequence_slot *
			state->paged_cache.lane_page_capacity + page;
		physical = slot->host_physical_page_table[
			(uint64_t)lane_ordinal * state->paged_cache.lane_page_capacity + page];
		if ( state->paged_cache.host_device_page_table[table_index] == physical )
			continue;
		if ( slot->page_table_update_count >= update_capacity ||
			table_index > UINT32_MAX )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		slot->host_page_table_update_indices[slot->page_table_update_count] =
			(uint32_t)table_index;
		slot->host_page_table_update_values[slot->page_table_update_count] = physical;
		slot->page_table_update_count++;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModulePrepareCachePages(
	SparkDsv4AsyncCompletion *async)
{
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	SparkDsv4PreparedCacheAdmission *prepared;
	SparkStatus status;
	uint32_t index,*logical_pages,*physical_pages;
	state = async->state;
	slot = &state->slots[async->slot_index];
	slot->page_table_update_count = 0u;
	memset(slot->host_logical_page_table,0xff,(uint64_t)async->cache_lane_count *
		state->paged_cache.lane_page_capacity * sizeof(uint32_t));
	memset(slot->host_physical_page_table,0xff,(uint64_t)async->cache_lane_count *
		state->paged_cache.lane_page_capacity * sizeof(uint32_t));
	memset(slot->host_initialize_page_indices,0xff,
		(uint64_t)async->cache_lane_count * sizeof(uint32_t));
	memset(slot->host_initialize_parent_page_indices,0xff,
		(uint64_t)async->cache_lane_count * sizeof(uint32_t));
	if ( pthread_mutex_lock(&state->cache_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	status = SPARK_STATUS_OK;
	prepared = 0;
	if ( async->requires_prepared_cache_admission != 0u )
	{
		prepared = SparkDsv4ModuleFindCacheAdmission(state,
			&async->cache_admission,0);
		if ( prepared == 0 ||
			SparkDsv4ModuleCacheAdmissionRequestMatches(prepared,
				&async->cache_admission) == 0u ||
			prepared->state != SPARK_DSV4_PREPARED_CACHE_COMMITTED )
			status = SPARK_STATUS_VALIDATION_FAILED;
		else if ( prepared->mutable_page_demand != 0u )
			status = SparkKvCacheArenaConsumeUnassignedResidentBlocks(
				&state->paged_cache.arena,prepared->mutable_page_demand);
		if ( status == SPARK_STATUS_OK )
		{
			prepared->state = SPARK_DSV4_PREPARED_CACHE_ADOPTING;
			async->prepared_cache_admission_index = (uint32_t)(prepared -
				state->prepared_cache_admissions);
		}
	}
	for (index=0u; status==SPARK_STATUS_OK &&
		index<async->cache_lane_count; index++)
	{
		logical_pages = slot->host_logical_page_table +
			(uint64_t)index * state->paged_cache.lane_page_capacity;
		physical_pages = slot->host_physical_page_table +
			(uint64_t)index * state->paged_cache.lane_page_capacity;
		status = SparkDsv4PagedCachePrepareLane(&state->paged_cache,
			&async->cache_lanes[index],logical_pages,physical_pages,
			state->paged_cache.lane_page_capacity,
			&async->prepared_cache_lanes[index]);
		if ( status == SPARK_STATUS_OK )
		{
			async->prepared_cache_lane_count++;
			if ( async->prepared_cache_lanes[index].requires_initialization != 0u )
			{
				slot->host_initialize_page_indices[index] =
					async->prepared_cache_lanes[index].mutable_physical_page;
				slot->host_initialize_parent_page_indices[index] =
					async->prepared_cache_lanes[index].parent_physical_page;
			}
			status = SparkDsv4ModuleAppendPageTableUpdates(async,index);
		}
	}
	if ( status != SPARK_STATUS_OK )
	{
		(void)SparkDsv4ModuleRollbackCacheLanesLocked(async);
		if ( async->prepared_cache_admission_index != UINT32_MAX )
		{
			prepared = &state->prepared_cache_admissions[
				async->prepared_cache_admission_index];
			SparkDsv4ModuleClearCacheAdmission(state,prepared,0u);
			async->prepared_cache_admission_index = UINT32_MAX;
		}
		/* Once COMMITTED capacity has been consumed, BUSY/PENDING is not a
		 * retryable adapter result: the exact prepared ownership was terminally
		 * resolved above.  Fence the route instead of letting residentd retry a
		 * submission whose COMMIT record no longer exists. */
		if ( status == SPARK_STATUS_BUSY || status == SPARK_STATUS_PENDING )
			status = SPARK_STATUS_INTERNAL_ERROR;
	}
	(void)pthread_mutex_unlock(&state->cache_mutex);
	return(status);
}

static SparkStatus SparkDsv4ModuleStagePageTableUpdates(
	SparkDsv4AsyncCompletion *async)
{
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	uint32_t index,table_index;
	uint64_t pitch,update_bytes;
	cudaError_t error;
	state = async->state;
	slot = &state->slots[async->slot_index];
	if ( slot->page_table_update_count == 0u )
		return(SPARK_STATUS_OK);
	pitch = (uint64_t)async->cache_lane_count *
		state->paged_cache.lane_page_capacity * sizeof(uint32_t);
	update_bytes = (uint64_t)slot->page_table_update_count * sizeof(uint32_t);
	error = cudaMemcpy2DAsync(slot->page_table_update_indices,pitch,
		slot->host_page_table_update_indices,pitch,update_bytes,2u,
		cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchUpdatePageTable((cudaStream_t)slot->cuda_stream,
			slot->physical_page_table,slot->page_table_update_indices,
			slot->page_table_update_values,slot->page_table_update_count);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,
			"stage_page_table_updates"));
	for (index=0u; index<slot->page_table_update_count; index++)
	{
		table_index = slot->host_page_table_update_indices[index];
		state->paged_cache.host_device_page_table[table_index] =
			slot->host_page_table_update_values[index];
	}
	async->completion.device_memcpy_bytes += 2u * update_bytes;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleFinishCachePages(
	SparkDsv4AsyncCompletion *async,
	uint32_t completed)
{
	SparkDsv4ModuleState *state;
	SparkDsv4ModuleSlot *slot;
	SparkDsv4PreparedCacheAdmission *prepared;
	SparkStatus result,status;
	uint32_t index;
	state = async->state;
	slot = &state->slots[async->slot_index];
	if ( pthread_mutex_lock(&state->cache_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	result = SPARK_STATUS_OK;
	if ( completed != 0u )
	{
		for (index=0u; index<async->prepared_cache_lane_count; index++)
		{
			status = SparkDsv4PagedCacheUnpinLane(&state->paged_cache,
				slot->host_logical_page_table +
				(uint64_t)index * state->paged_cache.lane_page_capacity,
				async->prepared_cache_lanes[index].logical_page_count);
			if ( result == SPARK_STATUS_OK && status != SPARK_STATUS_OK )
				result = status;
		}
	}
	if ( completed != 0u && result == SPARK_STATUS_OK )
		for (index=0u; result==SPARK_STATUS_OK &&
			index<async->prepared_cache_lane_count; index++)
			result = SparkDsv4PagedCacheCompleteLane(&state->paged_cache,
				&async->cache_lanes[index],
				&async->prepared_cache_lanes[index]);
	if ( completed == 0u )
		result = SparkDsv4ModuleAbortCacheLanesLocked(async);
	else if ( result != SPARK_STATUS_OK )
	{
		/* CompleteLane is sequential and can fail after an earlier lane has
		 * published.  The common path already unpinned every lane, so fence the
		 * entire batch by releasing each sequence binding without unpinning a
		 * second time.  Immutable published pages may remain reusable, but no
		 * partially advanced lane is allowed to keep serving. */
		async->aborted_cache_lane_count = async->prepared_cache_lane_count;
		for (index=0u; index<async->prepared_cache_lane_count; index++)
		{
			status = SparkDsv4PagedCacheAbortLane(&state->paged_cache,
				&async->cache_lanes[index]);
			if ( result == SPARK_STATUS_OK && status != SPARK_STATUS_OK )
				result = status;
		}
		async->prepared_cache_lane_count = 0u;
	}
	else
		async->prepared_cache_lane_count = 0u;
	if ( async->prepared_cache_admission_index != UINT32_MAX )
	{
		if ( async->prepared_cache_admission_index >=
			SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT )
			result = SPARK_STATUS_INTERNAL_ERROR;
		else
		{
			prepared = &state->prepared_cache_admissions[
				async->prepared_cache_admission_index];
			if ( prepared->state != SPARK_DSV4_PREPARED_CACHE_ADOPTING ||
				SparkDsv4ModuleCacheAdmissionRequestMatches(prepared,
					&async->cache_admission) == 0u )
				result = SPARK_STATUS_INTERNAL_ERROR;
			SparkDsv4ModuleClearCacheAdmission(state,prepared,0u);
		}
		async->prepared_cache_admission_index = UINT32_MAX;
	}
	(void)pthread_mutex_unlock(&state->cache_mutex);
	return(result);
}

static void SparkDsv4ModuleCompleteAsync(void *context)
{
	SparkDsv4AsyncCompletion *async;
	SparkDsv4ModuleState *state;
	uint32_t index,lane;
	async = (SparkDsv4AsyncCompletion *)context;
	state = async != 0 ? async->state : 0;
	if ( state == 0 || async->slot_index >= state->pipeline_slot_count )
		return;
	state->slots[async->slot_index].dspark_verify_rows = 0u;
	state->slots[async->slot_index].dspark_verify_accept = 0u;
	if ( SparkDsv4ModuleFinishCachePages(async,
		async->completion.status == SPARK_STATUS_OK ? 1u : 0u) !=
		SPARK_STATUS_OK )
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
	if ( async->completion.status == SPARK_STATUS_OK )
	{
		if ( async->output_token_destination != 0 )
			memcpy(async->output_token_destination,
				state->slots[async->slot_index].host_output_token_ids,
				(uint64_t)async->lane_count * async->tokens_per_sequence *
				sizeof(uint32_t));
		for (index=0u; index<async->lane_count; index++)
		{
			lane = async->lane_indices[index];
			state->lane_sequence_ids[lane] = async->lane_sequence_ids[index];
			state->lane_next_positions[lane] = async->lane_next_positions[index];
		}
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
		atomic_fetch_add_explicit(&state->tokens_emitted,async->emitted_token_count,memory_order_relaxed);
	}
	else
	{
		for (index=0u; index<async->aborted_cache_lane_count; index++)
		{
			lane = async->lane_indices[index];
			state->lane_sequence_ids[lane] = 0u;
			state->lane_next_positions[lane] = 0u;
		}
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	}
	atomic_fetch_add_explicit(&state->host_callback_completion_count,1u,memory_order_relaxed);
	SparkStageModuleCompleteAndReleaseClaims(async->completion_function,async->completion_context,&async->completion,state->lane_states,state->resident_sequence_capacity,async->lane_indices,async->lane_count,state->slot_states,async->slot_index);
}

static SparkStatus SparkDsv4ModuleSynchronizeFailedSlot(
	SparkDsv4ModuleSlot *slot)
{
	if ( slot == 0 || slot->cuda_stream == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,
		cudaStreamSynchronize((cudaStream_t)slot->cuda_stream),
		"failed_slot_sync"));
}

static void SparkDsv4ModuleCompleteAfterFailedEnqueue(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t slot_index,
	SparkStatus status)
{
	SparkDsv4AsyncCompletion *async;
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	SparkStatus synchronize_status;
	async = &state->completions[slot_index];
	async->completion.status = status != SPARK_STATUS_OK ? status :
		SPARK_STATUS_INTERNAL_ERROR;
	synchronize_status = SparkDsv4ModuleSynchronizeFailedSlot(slot);
	if ( synchronize_status == SPARK_STATUS_OK )
	{
		SparkDsv4ModuleCompleteAsync(async);
		return;
	}
	/* A failed fence cannot prove that device work released this slot's
	 * buffers.  Publish one terminal error, but deliberately retain the lane,
	 * slot, and ADOPTING cache claims as quarantined capacity. */
	async->completion.status = synchronize_status;
	completion_function = async->completion_function;
	completion_context = async->completion_context;
	async->completion_function = 0;
	async->completion_context = 0;
	atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	if ( completion_function != 0 )
		completion_function(completion_context,&async->completion);
}

static SparkStatus SparkDsv4ModuleEnqueueAsync(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, uint32_t slot_index)
{
	cudaError_t error;
	error = cudaLaunchHostFunc((cudaStream_t)slot->cuda_stream,SparkDsv4ModuleCompleteAsync,&state->completions[slot_index]);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"async_completion"));
}

static void SparkDsv4ModuleInvalidateLanes(SparkDsv4ModuleState *state, const uint32_t *lane_indices, uint32_t lane_count)
{
	uint32_t index,lane;
	for (index=0u; index<lane_count; index++)
	{
		lane = lane_indices[index];
		state->lane_sequence_ids[lane] = 0u;
		state->lane_next_positions[lane] = 0u;
	}
}

static SparkStatus SparkDsv4ModuleExecuteFrame(
	SparkDsv4ModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkDsv4ResidentDecodeStageFrameContext *context)
{
	uint32_t lane_indices[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint8_t lane_requires_reset[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkDsv4AsyncCompletion *async;
	SparkDsv4ModuleSlot *slot;
	uint32_t emitted_token_count,lane_count,row_count,slot_index,tp_async_started;
	SparkStatus status,synchronize_status;
	if ( state->tp_degree > 1u && state->tp_graphs_sealed == 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	lane_count = frame->active_slot_count;
	row_count = frame->new_token_count;
	emitted_token_count = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? context->prefill_batch->emit_count : lane_count * frame->tokens_per_sequence;
	status = SparkDsv4ModuleCollectFrameLaneIndices(state,frame,context,lane_indices);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleIndexSetClaim(state->lane_states,state->resident_sequence_capacity,lane_indices,lane_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkDsv4ModuleValidateFrameContinuity(state,frame,context,lane_indices,lane_sequence_ids,lane_next_positions,lane_requires_reset);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,lane_indices,lane_count);
		return(status);
	}
	slot_index = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	status = SparkStageModuleSlotClaim(state->slot_states,state->pipeline_slot_count,&slot_index);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,lane_indices,lane_count);
		return(status);
	}
	slot = &state->slots[slot_index];
	slot->cuda_stream = frame->execution_stream;
	status = SparkDsv4ModulePrepareAsync(state,frame,context,slot_index,
		lane_count,row_count,emitted_token_count,lane_indices,
		lane_sequence_ids,lane_next_positions);
	async = &state->completions[slot_index];
	tp_async_started = 0u;
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModulePrepareCachePages(async);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleStagePageTableUpdates(async);
	if ( status == SPARK_STATUS_OK )
		atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	if ( status == SPARK_STATUS_OK )
	{
		if ( state->tp_degree == 1u && state->graph_capacity != 0u &&
			(frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) == 0u )
			status = SparkDsv4ModuleRunGraphedDecode(state,slot,slot_index,frame,context,row_count);
		else
		{
			status = SparkDsv4ModuleStageFrameRows(state,slot,frame,context);
			if ( status == SPARK_STATUS_OK )
				status = SparkDsv4ModuleRunFrame(state,slot,frame,context);
		}
	}
	if ( status == SPARK_STATUS_PENDING )
	{
		tp_async_started = 1u;
		status = SPARK_STATUS_OK;
	}
	if ( status == SPARK_STATUS_OK && tp_async_started == 0u )
		status = SparkDsv4ModuleEnqueueAsync(state,slot,slot_index);
	if ( status != SPARK_STATUS_OK )
	{
		uint32_t cache_adopted =
			async->prepared_cache_admission_index != UINT32_MAX ? 1u : 0u;
		synchronize_status = SparkDsv4ModuleSynchronizeFailedSlot(slot);
		if ( synchronize_status != SPARK_STATUS_OK )
		{
			/* The slot and lane claims are the fence when CUDA cannot prove that
			 * their buffers are idle. */
			atomic_fetch_add_explicit(&state->failed_count,1u,
				memory_order_relaxed);
			return(synchronize_status);
		}
		if ( async->prepared_cache_lane_count != 0u )
			(void)SparkDsv4ModuleFinishCachePages(async,0u);
		if ( async->aborted_cache_lane_count != 0u )
			SparkDsv4ModuleInvalidateLanes(state,lane_indices,
				async->aborted_cache_lane_count);
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
		SparkStageModuleSlotRelease(state->slot_states,slot_index);
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,lane_indices,lane_count);
		if ( cache_adopted != 0u &&
			(status == SPARK_STATUS_BUSY || status == SPARK_STATUS_PENDING) )
			status = SPARK_STATUS_INTERNAL_ERROR;
	}
	return(status);
}

static SparkStatus SparkDsv4ModuleValidateReleaseFrame(
	const SparkDsv4ModuleState *state,
	const SparkModelDriverFrame *frame)
{
	uint32_t lane;
	if ( state == 0 || frame == 0 ||
		frame->flags != SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE ||
		frame->program_id == 0u || frame->tokens_per_sequence != 0u ||
		frame->reserved1 != 0u || frame->execution_stream == 0 ||
		frame->completion_function == 0 || frame->active_slot_count == 0u ||
		frame->active_slot_count > state->resident_sequence_capacity ||
		frame->new_token_count != 0u || frame->buffer_count != 0u ||
		frame->buffers != 0 || frame->user_context != 0 ||
		frame->cache_lane_count != frame->active_slot_count ||
		frame->cache_lanes == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (lane=0u; lane<frame->cache_lane_count; lane++)
		if ( SparkModelDriverCacheLaneIsValid(&frame->cache_lanes[lane]) == 0u ||
			frame->cache_lanes[lane].flags !=
			SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE ||
			frame->cache_lanes[lane].resident_sequence_slot >=
			state->resident_sequence_capacity )
			return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleExecuteRelease(
	SparkDsv4ModuleState *state,
	SparkModelDriverFrame *frame)
{
	SparkModelDriverCompletion completion;
	uint32_t index,lane,locked;
	uint32_t lane_indices[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkStatus result,status;
	for (index=0u; index<frame->cache_lane_count; index++)
		lane_indices[index] = frame->cache_lanes[index].resident_sequence_slot;
	status = SparkStageModuleIndexSetClaim(state->lane_states,
		state->resident_sequence_capacity,lane_indices,frame->cache_lane_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	locked = pthread_mutex_lock(&state->cache_mutex) == 0 ? 1u : 0u;
	result = locked != 0u ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	for (index=0u; result==SPARK_STATUS_OK &&
		index<frame->cache_lane_count; index++)
	{
		status = SparkDsv4PagedCacheReleaseLane(&state->paged_cache,
			&frame->cache_lanes[index]);
		if ( status != SPARK_STATUS_OK )
			result = status;
		else
		{
			lane = lane_indices[index];
			state->lane_sequence_ids[lane] = 0u;
			state->lane_next_positions[lane] = 0u;
		}
	}
	if ( locked != 0u )
		(void)pthread_mutex_unlock(&state->cache_mutex);
	SparkStageModuleIndexSetRelease(state->lane_states,
		state->resident_sequence_capacity,lane_indices,frame->cache_lane_count);
	memset(&completion,0,sizeof(completion));
	completion.request_id = frame->request_id;
	completion.sequence_id = frame->sequence_id;
	completion.sequence_position = frame->sequence_position;
	completion.program_id = frame->program_id;
	completion.driver_dispatch_slot = frame->driver_dispatch_slot;
	completion.status = result;
	completion.residency = frame->residency;
	if ( result == SPARK_STATUS_OK )
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
	else
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	frame->completion_function(frame->completion_context,&completion);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDsv4ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame)
{
	SparkDsv4ModuleState *state;
	const SparkDsv4ResidentDecodeStageFrameContext *context,*cache_context;
	SparkStatus status,cleanup_status;
	state = (SparkDsv4ModuleState *)module_state;
	context = 0;
	if ( state != 0 && frame != 0 &&
		(frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
	{
		status = SparkDsv4ModuleValidateReleaseFrame(state,frame);
		if ( status == SPARK_STATUS_OK &&
			state->execution_stream != frame->execution_stream )
			status = SPARK_STATUS_INVALID_ARGUMENT;
		if ( status == SPARK_STATUS_OK )
			status = SparkDsv4ModuleExecuteRelease(state,frame);
		if ( status != SPARK_STATUS_OK )
			atomic_fetch_add_explicit(&state->rejected_count,1u,
				memory_order_relaxed);
		return(status);
	}
	status = SparkDsv4ModuleValidateFrame(state,frame,&context);
	if ( status == SPARK_STATUS_OK && state->execution_stream != frame->execution_stream )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleExecuteFrame(state,frame,context);
	if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY &&
		status != SPARK_STATUS_PENDING && state != 0 && frame != 0 &&
		frame->cache_lane_count != 0u )
	{
		cache_context = context != 0 ? context :
			SparkDsv4ModuleCacheAdmissionFrameContext(frame);
		if ( cache_context != 0 )
		{
			cleanup_status = SparkDsv4ModuleReleaseCommittedCacheAdmission(
				state,frame,cache_context);
			if ( cleanup_status != SPARK_STATUS_OK &&
				cleanup_status != SPARK_STATUS_VALIDATION_FAILED )
				status = SPARK_STATUS_INTERNAL_ERROR;
		}
	}
	if ( status != SPARK_STATUS_OK && state != 0 )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	return(status);
}

static void SparkDsv4ModuleRejectAdmission(
	SparkDsv4ModuleState *state,
	SparkModelDriverAdmissionDecision *decision,
	uint32_t reason)
{
	SparkStageModuleAdmissionDecisionReject(decision,reason);
	atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
}

static uint32_t SparkDsv4ModuleAdmissionShapeSupported(
	const SparkDsv4ModuleState *state,
	const SparkModelDriverAdmissionRequest *request,
	uint32_t is_prefill)
{
	const uint32_t known_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL |
		SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
	uint32_t preparing,releasing;
	preparing = (request->admission_flags &
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) != 0u;
	releasing = (request->frame_flags &
		SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u;
	if ( (request->frame_flags & ~known_flags) != 0u ||
		(request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u ||
		((request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u &&
		 releasing != 0u) ||
		request->active_slot_count == 0u ||
		request->active_slot_count > state->resident_sequence_capacity )
		return(0u);
	if ( releasing != 0u )
		return(preparing == 0u && request->new_token_count == 0u ? 1u : 0u);
	if ( state->tp_degree > 1u &&
		!((request->active_slot_count == SPARK_BATCH_BUCKET &&
		   request->new_token_count == SPARK_BATCH_BUCKET) ||
		  (SPARK_BATCH_BUCKET == SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u &&
		   state->dspark_enabled != 0u &&
		   request->active_slot_count == 1u &&
		   request->new_token_count == 1u)) )
		return(0u);
	if ( request->new_token_count == 0u ||
		request->new_token_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT ||
		(is_prefill == 0u && request->new_token_count != request->active_slot_count) )
		return(0u);
	return(1u);
}

static uint32_t SparkDsv4ModuleAdmissionFitsKv(
	const SparkDsv4ModuleState *state,
	const SparkModelDriverAdmissionRequest *request)
{
	uint32_t lane;
	if ( request->sequence_position >= (uint64_t)state->max_sequence_positions )
		return(0u);
	if ( request->active_slot_count == 1u &&
		request->new_token_count > (uint64_t)state->max_sequence_positions - request->sequence_position )
		return(0u);
	for (lane=0u; lane<request->cache_lane_count; lane++)
		if ( request->cache_lanes[lane].sequence_position >=
			state->max_sequence_positions ||
			request->cache_lanes[lane].context_token_count >
			state->max_sequence_positions )
			return(0u);
	return(1u);
}

static uint32_t SparkDsv4ModuleCacheAdmissionIdentityMatches(
	const SparkDsv4PreparedCacheAdmission *prepared,
	const SparkModelDriverAdmissionRequest *request)
{
	return(prepared->state != SPARK_DSV4_PREPARED_CACHE_FREE &&
		prepared->request.submission_id == request->submission_id &&
		prepared->request.control_generation == request->control_generation &&
		prepared->request.transaction_id == request->transaction_id &&
		prepared->request.request_generation == request->request_generation &&
		prepared->request.step_generation == request->step_generation ? 1u : 0u);
}

static uint32_t SparkDsv4ModuleCacheAdmissionRequestMatches(
	const SparkDsv4PreparedCacheAdmission *prepared,
	const SparkModelDriverAdmissionRequest *request)
{
	const SparkModelDriverAdmissionRequest *expected;
	expected = &prepared->request;
	if ( SparkDsv4ModuleCacheAdmissionIdentityMatches(prepared,request) == 0u ||
		expected->program_id != request->program_id ||
		expected->request_id != request->request_id ||
		expected->sequence_id != request->sequence_id ||
		expected->sequence_position != request->sequence_position ||
		expected->deadline_time_ns != request->deadline_time_ns ||
		expected->active_slot_count != request->active_slot_count ||
		expected->new_token_count != request->new_token_count ||
		expected->priority != request->priority ||
		expected->frame_flags != request->frame_flags ||
		expected->cache_lane_count != request->cache_lane_count ||
		memcmp(&expected->residency,&request->residency,
			sizeof(expected->residency)) != 0 ||
		memcmp(prepared->cache_lanes,request->cache_lanes,
			(uint64_t)request->cache_lane_count *
			sizeof(prepared->cache_lanes[0])) != 0 )
		return(0u);
	return(1u);
}

static SparkDsv4PreparedCacheAdmission *
SparkDsv4ModuleFindCacheAdmission(
	SparkDsv4ModuleState *state,
	const SparkModelDriverAdmissionRequest *request,
	SparkDsv4PreparedCacheAdmission **free_record_out)
{
	SparkDsv4PreparedCacheAdmission *prepared,*free_record;
	uint32_t index;
	free_record = 0;
	for (index=0u;
		index<SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT; index++)
	{
		prepared = &state->prepared_cache_admissions[index];
		if ( prepared->state == SPARK_DSV4_PREPARED_CACHE_FREE )
		{
			if ( free_record == 0 )
				free_record = prepared;
			continue;
		}
		if ( SparkDsv4ModuleCacheAdmissionIdentityMatches(prepared,request) != 0u )
		{
			if ( free_record_out != 0 )
				*free_record_out = free_record;
			return(prepared);
		}
	}
	if ( free_record_out != 0 )
		*free_record_out = free_record;
	return(0);
}

static void SparkDsv4ModuleClearCacheAdmission(
	SparkDsv4ModuleState *state,
	SparkDsv4PreparedCacheAdmission *prepared,
	uint32_t release_ownership)
{
	if ( release_ownership != 0u && prepared->mutable_page_demand != 0u )
		(void)SparkKvCacheArenaReleaseUnassignedResidentBlocks(
			&state->paged_cache.arena,prepared->mutable_page_demand);
	memset(prepared,0,sizeof(*prepared));
}

static SparkStatus SparkDsv4ModuleProgressCacheAdmission(
	SparkDsv4ModuleState *state,
	SparkDsv4PreparedCacheAdmission *prepared)
{
	SparkStatus result,status;
	uint32_t lane;
	result = SPARK_STATUS_OK;
	for (lane=0u; lane<prepared->request.cache_lane_count; lane++)
	{
		status = SparkDsv4PagedCachePrefetchLane(&state->paged_cache,
			&prepared->cache_lanes[lane],state->cache_admission_logical_pages,
			state->paged_cache.lane_page_capacity);
		if ( status == SPARK_STATUS_BUSY || status == SPARK_STATUS_PENDING )
			result = SPARK_STATUS_BUSY;
		else if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( result == SPARK_STATUS_OK )
		prepared->state = SPARK_DSV4_PREPARED_CACHE_READY;
	return(result);
}

static SparkStatus SparkDsv4ModulePrepareCacheAdmission(
	SparkDsv4ModuleState *state,
	const SparkModelDriverAdmissionRequest *request)
{
	SparkDsv4PreparedCacheAdmission *prepared,*free_record;
	SparkStatus status;
	uint32_t lane,mutable_demand,lane_demand,logical_page_count;
	if ( pthread_mutex_lock(&state->cache_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	prepared = SparkDsv4ModuleFindCacheAdmission(state,request,&free_record);
	if ( prepared != 0 )
	{
		if ( SparkDsv4ModuleCacheAdmissionRequestMatches(prepared,request) == 0u )
			status = SPARK_STATUS_VALIDATION_FAILED;
		else if ( prepared->state == SPARK_DSV4_PREPARED_CACHE_READY )
			status = SPARK_STATUS_OK;
		else if ( prepared->state == SPARK_DSV4_PREPARED_CACHE_PREFETCHING )
			status = SparkDsv4ModuleProgressCacheAdmission(state,prepared);
		else
			status = SPARK_STATUS_DUPLICATE;
		(void)pthread_mutex_unlock(&state->cache_mutex);
		return(status);
	}
	if ( free_record == 0 )
	{
		(void)pthread_mutex_unlock(&state->cache_mutex);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	mutable_demand = 0u;
	status = SPARK_STATUS_OK;
	for (lane=0u; status==SPARK_STATUS_OK && lane<request->cache_lane_count;
		lane++)
	{
		status = SparkKvPageCacheResolveLanePages(
			&state->paged_cache.page_cache,&request->cache_lanes[lane],
			state->cache_admission_logical_pages,
			state->paged_cache.lane_page_capacity,&logical_page_count);
		if ( status == SPARK_STATUS_OK )
			status = SparkKvPageCacheGetLaneMutablePageDemand(
				&state->paged_cache.page_cache,&request->cache_lanes[lane],
				&lane_demand);
		if ( status == SPARK_STATUS_OK )
		{
			if ( lane_demand > UINT32_MAX - mutable_demand )
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
			else
				mutable_demand += lane_demand;
		}
	}
	if ( status == SPARK_STATUS_OK && mutable_demand != 0u )
		status = SparkKvCacheArenaReserveUnassignedResidentBlocks(
			&state->paged_cache.arena,mutable_demand);
	if ( status == SPARK_STATUS_OK )
	{
		memset(free_record,0,sizeof(*free_record));
		free_record->state = SPARK_DSV4_PREPARED_CACHE_PREFETCHING;
		free_record->mutable_page_demand = mutable_demand;
		free_record->request = *request;
		memcpy(free_record->cache_lanes,request->cache_lanes,
			(uint64_t)request->cache_lane_count *
			sizeof(free_record->cache_lanes[0]));
		free_record->request.admission_flags = 0u;
		free_record->request.cache_lanes = free_record->cache_lanes;
		status = SparkDsv4ModuleProgressCacheAdmission(state,free_record);
		/* A BUSY/PENDING result is not an accepted PREPARE, so the resident
		 * route has nobody that can later COMMIT or ABORT this record.  Keep the
		 * global page-store job, but release this exact capacity ownership and
		 * let a later PREPARE resolve/prefetch it afresh. */
		if ( status == SPARK_STATUS_BUSY || status == SPARK_STATUS_PENDING )
		{
			SparkDsv4ModuleClearCacheAdmission(state,free_record,1u);
			status = SPARK_STATUS_BUSY;
		}
		else if ( status != SPARK_STATUS_OK )
			SparkDsv4ModuleClearCacheAdmission(state,free_record,1u);
	}
	(void)pthread_mutex_unlock(&state->cache_mutex);
	return(status);
}

static SparkStatus SparkDsv4ModuleResolveCacheAdmission(
	SparkDsv4ModuleState *state,
	const SparkModelDriverAdmissionRequest *request)
{
	SparkDsv4PreparedCacheAdmission *prepared;
	SparkStatus status;
	if ( pthread_mutex_lock(&state->cache_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	prepared = SparkDsv4ModuleFindCacheAdmission(state,request,0);
	if ( prepared == 0 )
		status = SPARK_STATUS_NOT_FOUND;
	else if ( SparkDsv4ModuleCacheAdmissionRequestMatches(prepared,request) == 0u )
		status = SPARK_STATUS_VALIDATION_FAILED;
	else if ( request->admission_flags ==
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT &&
		(prepared->state == SPARK_DSV4_PREPARED_CACHE_PREFETCHING ||
		 prepared->state == SPARK_DSV4_PREPARED_CACHE_READY) )
	{
		SparkDsv4ModuleClearCacheAdmission(state,prepared,1u);
		status = SPARK_STATUS_OK;
	}
	else if ( request->admission_flags ==
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT &&
		prepared->state == SPARK_DSV4_PREPARED_CACHE_READY )
	{
		prepared->state = SPARK_DSV4_PREPARED_CACHE_COMMITTED;
		status = SPARK_STATUS_OK;
	}
	else
		status = SPARK_STATUS_DUPLICATE;
	(void)pthread_mutex_unlock(&state->cache_mutex);
	return(status);
}

static SparkStatus SparkDsv4ModuleRequireCommittedCacheAdmission(
	SparkDsv4ModuleState *state,
	const SparkModelDriverAdmissionRequest *request)
{
	SparkDsv4PreparedCacheAdmission *prepared;
	SparkStatus status;
	if ( pthread_mutex_lock(&state->cache_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	prepared = SparkDsv4ModuleFindCacheAdmission(state,request,0);
	status = prepared != 0 &&
		SparkDsv4ModuleCacheAdmissionRequestMatches(prepared,request) != 0u &&
		prepared->state == SPARK_DSV4_PREPARED_CACHE_COMMITTED ?
		SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED;
	(void)pthread_mutex_unlock(&state->cache_mutex);
	return(status);
}

SparkStatus SparkDsv4ResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkDsv4ModuleState *state;
	uint32_t available,is_prefill,preparing,committing,aborting,releasing;
	SparkStatus status;
	state = (SparkDsv4ModuleState *)module_state;
	if ( state == 0 || request == 0 || decision == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	available = SparkStageModuleSlotCountFree(state->slot_states,state->pipeline_slot_count);
	SparkStageModuleAdmissionDecisionInitialize(decision,available);
	if ( request->descriptor_bytes < (uint32_t)sizeof(*request) ||
		request->program_id == 0u ||
		SparkModelDriverAdmissionRequestIsValid(request) == 0u )
		return(SPARK_STATUS_ABI_MISMATCH);
	is_prefill = (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	preparing = (request->admission_flags & SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) != 0u;
	committing = (request->admission_flags &
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) != 0u;
	aborting = (request->admission_flags &
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) != 0u;
	releasing = (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u;
	if ( SparkDsv4ModuleAdmissionShapeSupported(state,request,is_prefill) == 0u )
	{
		SparkDsv4ModuleRejectAdmission(state,decision,SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);
		return(SPARK_STATUS_OK);
	}
	if ( releasing != 0u )
	{
		SparkStageModuleAdmissionDecisionAccept(decision);
		return(SPARK_STATUS_OK);
	}
	if ( SparkDsv4ModuleAdmissionFitsKv(state,request) == 0u )
	{
		SparkDsv4ModuleRejectAdmission(state,decision,SPARK_MODEL_DRIVER_ADMISSION_REJECTED_KV_CAPACITY);
		return(SPARK_STATUS_OK);
	}
	if ( preparing != 0u )
	{
		status = SparkDsv4ModulePrepareCacheAdmission(state,request);
		if ( status != SPARK_STATUS_OK )
		{
			SparkDsv4ModuleRejectAdmission(state,decision,
				status == SPARK_STATUS_CAPACITY_EXCEEDED ?
				SPARK_MODEL_DRIVER_ADMISSION_REJECTED_KV_CAPACITY :
				SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY);
			return(SPARK_STATUS_OK);
		}
		SparkStageModuleAdmissionDecisionAccept(decision);
		return(SPARK_STATUS_OK);
	}
	if ( committing != 0u || aborting != 0u )
	{
		status = SparkDsv4ModuleResolveCacheAdmission(state,request);
		if ( status != SPARK_STATUS_OK )
			return(status);
		SparkStageModuleAdmissionDecisionAccept(decision);
		return(SPARK_STATUS_OK);
	}
	if ( request->cache_lane_count != 0u )
	{
		status = SparkDsv4ModuleRequireCommittedCacheAdmission(state,request);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( available == 0u )
	{
		SparkDsv4ModuleRejectAdmission(state,decision,SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY);
		return(SPARK_STATUS_OK);
	}
	SparkStageModuleAdmissionDecisionAccept(decision);
	decision->host_staging_bytes = (uint64_t)request->new_token_count *
		(sizeof(uint32_t) * (uint64_t)(state->owns_embedding + state->owns_final_head + 1u) + sizeof(uint64_t) * 3u);
	decision->device_memcpy_bytes = decision->host_staging_bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDsv4ResidentDecodeStageSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot)
{
    SparkDsv4ModuleState *state;

    state = (SparkDsv4ModuleState *)module_state;
    if (state == 0 || snapshot == 0 || program_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkStageModuleRuntimeSnapshotInitialize(
        snapshot,
        program_id,
        state->slot_states,
        state->pipeline_slot_count);
    snapshot->submitted_count = atomic_load_explicit(
        &state->submitted_count,
        memory_order_relaxed);
    snapshot->completed_count = atomic_load_explicit(
        &state->completed_count,
        memory_order_relaxed);
    snapshot->rejected_count = atomic_load_explicit(
        &state->rejected_count,
        memory_order_relaxed);
	snapshot->resident_sequence_count =
		state->paged_cache.page_cache.live_sequence_count;
	snapshot->resident_token_count =
		state->paged_cache.arena.resident_block_count *
		state->paged_cache.arena.block_token_count;
	snapshot->kv_token_capacity =
		(uint64_t)state->paged_cache.physical_page_capacity *
		state->paged_cache.arena.block_token_count;
	snapshot->host_callback_completion_count = atomic_load_explicit(&state->host_callback_completion_count,memory_order_relaxed);
    return SPARK_STATUS_OK;
}

void SparkDsv4ResidentDecodeStageDestroy(void *module_state)
{
    SparkDsv4ModuleState *state;
	uint32_t slot_index,admission_index;

    state = (SparkDsv4ModuleState *)module_state;
    if (state == 0)
    {
        return;
    }
    if (SparkStageModuleWaitForSlots(
            SPARK_DSV4_MODULE_TAG,
            state->slot_states,
            state->pipeline_slot_count,
            SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK)
    {
        return;
	}
	for (slot_index = 0u; slot_index < state->pipeline_slot_count; ++slot_index)
	{
		if ( state->slots[slot_index].tp_host_copy_event != 0 )
			(void)cudaEventDestroy(state->slots[slot_index].tp_host_copy_event);
		if ( state->slots[slot_index].host_staging != 0 )
			(void)cudaFreeHost(state->slots[slot_index].host_staging);
	}
	for (slot_index = 0u; state->tp_degree == 1u &&
		slot_index < state->graph_capacity; ++slot_index)
	{
		if ( state->graph_entries[slot_index].live != 0 )
			(void)cudaGraphExecDestroy(state->graph_entries[slot_index].executable);
	}
	for (slot_index = 0u; slot_index < state->tp_graph_island_count;
		slot_index++)
	{
		if ( state->tp_graph_islands[slot_index].live != 0u )
			(void)cudaGraphExecDestroy(
				state->tp_graph_islands[slot_index].executable);
	}
	free(state->tp_graph_islands);
	state->tp_graph_islands = 0;
	state->tp_graph_island_count = 0u;
	state->tp_graphs_sealed = 0u;
	for (slot_index=0u; slot_index<state->pipeline_slot_count; slot_index++)
	{
		SparkStageModuleCudaReadAheadDestroy(
			&state->slots[slot_index].weight_read_ahead);
		SparkStageModuleCudaForkDestroy(
			&state->slots[slot_index].compute_fork);
	}
	if ( state->cache_mutex_initialized != 0u &&
		pthread_mutex_lock(&state->cache_mutex) == 0 )
	{
		for (admission_index=0u;
			admission_index<SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT;
			admission_index++)
		{
			SparkDsv4PreparedCacheAdmission *prepared =
				&state->prepared_cache_admissions[admission_index];
			if ( prepared->state != SPARK_DSV4_PREPARED_CACHE_FREE )
				SparkDsv4ModuleClearCacheAdmission(state,prepared,
					prepared->state != SPARK_DSV4_PREPARED_CACHE_ADOPTING ?
					1u : 0u);
		}
		(void)pthread_mutex_unlock(&state->cache_mutex);
	}
	free(state->cache_admission_logical_pages);
	state->cache_admission_logical_pages = 0;
	free(state->tp_continuations);
	state->tp_continuations = 0;
	if ( state->kv_page_store_enabled != 0u )
	{
		SparkKvPageStoreDestroy(&state->kv_page_store);
		state->kv_page_store_enabled = 0u;
	}
	if ( state->kv_page_store_stream != 0 )
	{
		(void)cudaStreamDestroy((cudaStream_t)state->kv_page_store_stream);
		state->kv_page_store_stream = 0;
	}
	if ( state->kv_page_store_staging != 0 )
	{
		(void)cudaFreeHost(state->kv_page_store_staging);
		state->kv_page_store_staging = 0;
	}
	if ( state->tp_device_collective_initialized != 0u )
	{
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
		state->tp_device_collective_initialized = 0u;
	}
	if ( state->tp_host_credit_send_bf16 != 0 )
		(void)cudaFreeHost(state->tp_host_credit_send_bf16);
	if ( state->tp_host_credit_receive_bf16 != 0 )
		(void)cudaFreeHost(state->tp_host_credit_receive_bf16);
	state->tp_host_credit_send_bf16 = 0;
	state->tp_host_credit_receive_bf16 = 0;
	state->tp_credit_send_bf16 = 0;
	state->tp_credit_receive_bf16 = 0;
	SparkDsv4PagedCacheDestroyHost(&state->paged_cache);
	if ( state->cache_mutex_initialized != 0u )
	{
		(void)pthread_mutex_destroy(&state->cache_mutex);
		state->cache_mutex_initialized = 0u;
	}
    SparkStageModuleLedgerRelease(&state->ledger);
    free(state);
}

static SparkStatus SparkDsv4ModuleAllocateState(
	const SparkFirmwareModuleHostServices *host_services,
	SparkDsv4ModuleState **state_out,
	const char **pack_path_out)
{
	SparkDsv4ModuleState *state;
	SparkStatus status;
	state = (SparkDsv4ModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( pthread_mutex_init(&state->cache_mutex,0) != 0 )
	{
		free(state);
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	state->cache_mutex_initialized = 1u;
	state->ledger.module_tag = SPARK_DSV4_MODULE_TAG;
	atomic_init(&state->submitted_count,0u);
	atomic_init(&state->completed_count,0u);
	atomic_init(&state->rejected_count,0u);
	atomic_init(&state->failed_count,0u);
	atomic_init(&state->tokens_emitted,0u);
	atomic_init(&state->host_callback_completion_count,0u);
	atomic_init(&state->tp_next_ordinal,0u);
	status = SparkDsv4ModuleConfigure(state,host_services,pack_path_out);
	if ( status != SPARK_STATUS_OK )
	{
		if ( state->tp_device_collective_initialized != 0u )
			SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
		if ( state->tp_host_credit_send_bf16 != 0 )
			(void)cudaFreeHost(state->tp_host_credit_send_bf16);
		if ( state->tp_host_credit_receive_bf16 != 0 )
			(void)cudaFreeHost(state->tp_host_credit_receive_bf16);
		SparkStageModuleLedgerRelease(&state->ledger);
		(void)pthread_mutex_destroy(&state->cache_mutex);
		free(state);
		return(status);
	}
	state->tp_continuations = (SparkDsv4TpFrameContinuation *)calloc(
		state->pipeline_slot_count,sizeof(*state->tp_continuations));
	if ( state->tp_continuations == 0 )
	{
		if ( state->tp_device_collective_initialized != 0u )
			SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
		if ( state->tp_host_credit_send_bf16 != 0 )
			(void)cudaFreeHost(state->tp_host_credit_send_bf16);
		if ( state->tp_host_credit_receive_bf16 != 0 )
			(void)cudaFreeHost(state->tp_host_credit_receive_bf16);
		SparkStageModuleLedgerRelease(&state->ledger);
		(void)pthread_mutex_destroy(&state->cache_mutex);
		free(state);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	SparkStageModuleAtomicStateArrayInitialize(state->slot_states,state->pipeline_slot_count);
	SparkStageModuleAtomicStateArrayInitialize(state->lane_states,state->resident_sequence_capacity);
	*state_out = state;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModulePrepareState(
	SparkDsv4ModuleState *state,
	const char *pack_path)
{
	uint32_t slot_index;
	SparkStatus status;
	status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,
		SparkDsv4ConfigureCudaKernels(&state->multiprocessor_count),"configure_cuda_kernels");
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleValidateSlice(state);
	if ( status == SPARK_STATUS_OK )
	{
		/* Worst-case sparse-attention block count for this bucket: the
		 * launch needs one partial slot per (row, head-group, split);
		 * when blocks exceed the SM count the split count clamps to 1
		 * and the grid queues in waves, so the capacity must cover
		 * bucket_rows * head_groups even past multiprocessor_count. */
		uint32_t head_groups = (SparkDsv4ModuleTpQueryHeads(state) +
			SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA - 1u) /
			SPARK_DSV4_SPARSE_ATTN_HEADS_PER_CTA;
		uint64_t blocks_max = (uint64_t)SPARK_BATCH_BUCKET * head_groups;
		state->sparse_attn_partial_capacity = state->multiprocessor_count;
		if ( blocks_max > state->sparse_attn_partial_capacity )
			state->sparse_attn_partial_capacity = (uint32_t)blocks_max;
	}
	if ( status == SPARK_STATUS_OK )
	{
		SparkDsv4ModuleBuildOrdinals(state);
		status = SparkDsv4ModuleLoadPack(state,pack_path);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleUploadFreqs(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocatePools(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleFinalizeLoad(state);
	for (slot_index=0u; status==SPARK_STATUS_OK && slot_index<state->pipeline_slot_count; slot_index++)
	{
		state->slots[slot_index].tp_continuation =
			&state->tp_continuations[slot_index];
		status = SparkDsv4ModuleAllocateSlot(state,&state->slots[slot_index]);
	}
	if ( status == SPARK_STATUS_OK && state->tp_degree == 1u )
		LmGraphCacheInitialise(&state->graph_cache,state->graph_entries,
			state->graph_capacity);
	if ( status == SPARK_STATUS_OK && state->tp_degree > 1u )
		status = SparkDsv4ModulePrewarmTpGraphs(state);
	return(status);
}

static void SparkDsv4ModuleReportReady(const SparkDsv4ModuleState *state)
{
	fprintf(stderr,
		"%s ready stage=%u/%u slice=%u+%u compress=%u csa=%u lanes=%u max_seq=%u graphs=%u backing=%u page_kib=%.1f state_gib=%.3f device_gib=%.1f\n",
		SPARK_DSV4_MODULE_TAG,state->stage_index,state->stage_count,
		state->first_layer_index,state->layer_count,state->compress_layer_count,
		state->csa_layer_count,state->resident_sequence_capacity,
		state->max_sequence_positions,state->graph_capacity,
		state->kv_page_store_enabled,
		(double)state->paged_cache.layout.page_stride_bytes / 1024.0,
		(double)state->resident_state_bytes /
		(1024.0 * 1024.0 * 1024.0),(double)state->ledger.device_bytes_resident /
		(1024.0 * 1024.0 * 1024.0));
}

SparkStatus SparkDsv4ResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state)
{
	SparkDsv4ModuleState *state;
	const char *pack_path;
	SparkStatus status;
	pack_path = 0;
	status = SparkFirmwareModuleValidateInitialization(configuration,host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkDsv4ModuleAllocateState(host_services,&state,&pack_path);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkDsv4ModulePrepareState(state,pack_path);
	if ( status != SPARK_STATUS_OK )
	{
		SparkDsv4ResidentDecodeStageDestroy(state);
		return(status);
	}
	SparkDsv4ModuleReportReady(state);
	*module_state = state;
	return(SPARK_STATUS_OK);
}
