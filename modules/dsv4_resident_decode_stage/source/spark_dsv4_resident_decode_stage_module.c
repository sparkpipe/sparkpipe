/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_kv_page_store.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_row_layout.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "spark_dsv4_lane_continuity.h"
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

typedef struct SparkDsv4ModuleSlot
{
	void *cuda_stream;
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
	int32_t *topk_idxs;
	void *streams_bf16;
	void *residual_bf16;
	void *reduced_bf16;
	void *normalized_bf16;
	void *qr_bf16;
	void *q_bf16;
	void *kv_bf16;
	void *attn_out_bf16;
	void *o_ranks_bf16;
	void *delta_bf16;
	void *compress_kv_bf16;
	void *compress_score_bf16;
	float *compress_kv_f32;
	float *compress_score_f32;
	void *emit_bf16;
	uint32_t *emitted_u32;
	void *index_q_bf16;
	void *index_weights_bf16;
	float *index_weights_f32;
	float *index_scores_f32;
	float *mixes_f32;
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
	uint32_t *head_candidate_ids_u32;
	uint32_t *head_candidate_counts_u32;
	uint32_t *expert_offsets_u32;
	uint32_t *moe_inverse_u32;
	uint32_t page_table_update_count;
} SparkDsv4ModuleSlot;

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
	uint32_t prepared_cache_lane_count;
	uint32_t aborted_cache_lane_count;
	uint32_t lane_indices[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkModelDriverCacheLane cache_lanes[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkDsv4PagedCacheLane prepared_cache_lanes[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t *output_token_destination;
	SparkModelDriverCompletion completion;
} SparkDsv4AsyncCompletion;

struct SparkDsv4ModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
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
	pthread_mutex_t cache_mutex;
	uint32_t cache_mutex_initialized;
	float hc_head_scale_value;
	uint32_t csa_ordinal_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t layer_seen_bits[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t mtp_seen_bits;
	uint64_t global_seen_bits;
	SparkDsv4LayerWeights layers[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	SparkDsv4LayerWeights mtp_layer;
	SparkDsv4MtpWeights mtp;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	uint8_t *head_shadow_payload;
	uint8_t *head_shadow_scale;
	float *head_error_norm_f32;
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
	/*
	 * Decode-step graph cache: fixed storage, keyed by (pipeline slot, lane
	 * count) - the only launch-shape inputs a decode frame carries. Everything
	 * else a frame varies (token ids, lane indices, positions, boundary
	 * addresses) reaches the kernels through per-slot pinned staging and
	 * device arrays the recorded copies re-read on every replay, or is bounced
	 * through the slot's own stream buffer outside the graph. graph_sealed
	 * latches the first capture failure so an overflowed or refused cache
	 * stops paying capture attempts; replays of already-captured shapes
	 * continue.
	 */
	uint32_t graph_capacity;
	uint32_t graph_sealed;
	LmGraphCache graph_cache;
	LmGraphEntry graph_entries[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_GRAPH_COUNT];
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

extern cudaError_t SparkDsv4ConfigureCudaKernels(uint32_t *multiprocessor_count);
extern cudaError_t SparkDsv4LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkDsv4LaunchLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchStridedLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *payload, const uint8_t *scale, uint64_t weight_payload_group_stride_bytes, uint64_t weight_scale_group_stride_bytes, const void *input_bf16, uint64_t input_row_stride, uint32_t input_offset, uint32_t input_group_stride, void *output_bf16, uint64_t output_row_stride, uint32_t output_offset, uint32_t output_group_stride, uint32_t group_count, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchGatherHeadRows(cudaStream_t stream, const void *source_bf16, const uint32_t *source_row_indices, void *destination_bf16, uint32_t row_count, uint32_t row_width);
extern cudaError_t SparkDsv4LaunchScatterHeadTokens(cudaStream_t stream, const uint32_t *source, const uint32_t *destination_lane_indices, uint32_t *destination, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchQuantSim(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t row_stride, uint32_t width, uint32_t block, uint32_t fp4);
extern cudaError_t SparkDsv4LaunchRope(cudaStream_t stream, void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse);
extern cudaError_t SparkDsv4LaunchQueryHeadRms(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim, float epsilon);
extern cudaError_t SparkDsv4LaunchHadamard(cudaStream_t stream, void *data_bf16, uint32_t vector_count, uint32_t width);
extern cudaError_t SparkDsv4LaunchSparseAttn(cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_lane_indices, const uint32_t *row_page_table_indices, const uint32_t *physical_page_table, uint32_t page_table_stride, uint32_t compressed_entries_per_page, const int32_t *topk_idxs, uint32_t topk, const float *sink_f32, float scale, void *out_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim);
extern cudaError_t SparkDsv4LaunchWiden(cudaStream_t stream, const void *input_bf16, float *output_f32, uint32_t row_count, uint32_t width, float scale);
extern cudaError_t SparkDsv4LaunchApeAdd(cudaStream_t stream, float *score_f32, const float *ape_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t ratio, uint32_t channels);
extern cudaError_t SparkDsv4LaunchCompressStep(cudaStream_t stream, const float *kv_f32, const float *score_f32, float *kv_state_f32, float *score_state_f32, uint64_t state_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t row_count, uint32_t ratio, uint32_t overlapped, uint32_t width, void *emit_bf16, uint32_t *emitted);
extern cudaError_t SparkDsv4LaunchCacheScatter(cudaStream_t stream, const void *source_bf16, const uint32_t *emitted, void *cache_bf16, uint64_t cache_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t row_count, uint32_t width, uint64_t base_slot, uint32_t ratio, uint32_t ring_slots);
extern cudaError_t SparkDsv4LaunchInitializePages(cudaStream_t stream, void *page_pool, uint64_t page_stride_bytes, const uint32_t *page_indices, const uint32_t *parent_page_indices, uint32_t page_count, const SparkDsv4PagedScoreSpan *score_spans, uint32_t score_span_count);
extern cudaError_t SparkDsv4LaunchUpdatePageTable(cudaStream_t stream, uint32_t *page_table, const uint32_t *update_indices, const uint32_t *update_values, uint32_t update_count);
extern cudaError_t SparkDsv4LaunchBuildAttentionIndices(cudaStream_t stream, const uint64_t *row_positions, int32_t *indices, uint32_t *slot_counts, uint32_t row_count, uint32_t column_count, uint32_t index_slot_capacity, uint32_t layer_kind);
extern cudaError_t SparkDsv4LaunchGateScores(cudaStream_t stream, const SparkDsv4LinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, const uint32_t *tid2eid_u32, const uint32_t *token_ids, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32);
extern cudaError_t SparkDsv4LaunchSwigluClamp(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width, float limit, const float *row_weights_f32, const uint32_t *weight_map);
extern cudaError_t SparkDsv4LaunchValidateTid2Eid(cudaStream_t stream, const uint32_t *tid2eid, uint64_t entry_count, uint32_t *violation_flag);
extern cudaError_t SparkDsv4LaunchMoeRoute(cudaStream_t stream, const uint32_t *route_expert, uint32_t rows, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2);
extern cudaError_t SparkDsv4LaunchExpertUp(cudaStream_t stream, const SparkDsv4LinearView *stacked, const void *input_bf16, const uint32_t *route_source_token, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t multiprocessor_count);
extern cudaError_t SparkDsv4LaunchExpertDown(cudaStream_t stream, const SparkDsv4LinearView *stacked, const void *input_bf16, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t multiprocessor_count);
extern cudaError_t SparkDsv4LaunchMoePairReduce(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchAccumAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width);
extern cudaError_t SparkDsv4LaunchIndexerScore(cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_page_table_indices, const uint32_t *physical_page_table, uint32_t page_table_stride, uint32_t entries_per_page, const uint32_t *slot_counts, const float *head_weights_f32, float *scores_f32, uint32_t row_count, uint32_t max_slots, uint32_t head_count, uint32_t head_dim);
extern cudaError_t SparkDsv4LaunchTopK(cudaStream_t stream, float *scores_f32, const uint32_t *slot_counts, uint32_t max_slots, uint32_t topk, int32_t offset, int32_t *indices_out, uint64_t out_row_stride, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchHcMix(cudaStream_t stream, const void *streams_bf16, const float *fn_f32, float *mixes_f32, uint32_t row_count, uint32_t flat_dimension, uint32_t mix_rows, float epsilon);
extern cudaError_t SparkDsv4LaunchHcSplitSinkhorn(cudaStream_t stream, const float *mixes_f32, const float *scale3_f32, const float *base_f32, uint32_t row_count, uint32_t hc, uint32_t iterations, float epsilon, float *pre_f32, float *post_f32, float *comb_f32);
extern cudaError_t SparkDsv4LaunchHcPreReduce(cudaStream_t stream, const void *streams_bf16, const float *pre_f32, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension);
extern cudaError_t SparkDsv4LaunchHcPost(cudaStream_t stream, const void *out_bf16, const void *residual_bf16, const float *post_f32, const float *comb_f32, void *streams_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension);
extern cudaError_t SparkDsv4LaunchHcHeadReduce(cudaStream_t stream, const void *streams_bf16, const float *mixes_f32, float scale, const float *base_f32, float epsilon, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension);

static SparkStatus SparkDsv4ModuleConfigure(
	SparkDsv4ModuleState *state,
	const SparkFirmwareModuleHostServices *host_services,
	const char **pack_path_out)
{
	const SparkDsv4ResidentDecodeStageNodeContext *context;
	uint64_t directory_bytes;
	uint32_t lane_page_capacity;
	if ( state == 0 || host_services == 0 || pack_path_out == 0 ||
		host_services->node_context == 0 || host_services->execution_stream == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkDsv4ResidentDecodeStageNodeContext *)host_services->node_context;
	if ( context->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION || context->descriptor_bytes != SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( (context->flags & ~SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KNOWN_FLAGS) != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->stage_count == 0u || context->stage_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT || context->stage_index >= context->stage_count || context->first_layer_index >= SPARK_DSV4_MODEL_LAYER_COUNT || context->layer_count == 0u || context->layer_count > SPARK_DSV4_MODEL_LAYER_COUNT - context->first_layer_index || context->resident_sequence_capacity == 0u || context->resident_sequence_capacity > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_RESIDENT_SEQUENCE_COUNT || context->pipeline_slot_count == 0u || context->pipeline_slot_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT || context->cuda_graph_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_GRAPH_COUNT || context->max_sequence_positions < SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO || context->max_sequence_positions > SPARK_DSV4_MODEL_MAX_POSITIONS || context->linear_weight_codec != SPARK_DSV4_MODEL_NON_EXPERT_WEIGHT_CODEC || context->expert_weight_codec != SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC || context->kv_cache_codec != SPARK_DSV4_MODEL_KV_CACHE_CODEC || context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' )
		return(SPARK_STATUS_INVALID_ARGUMENT);
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
	state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
	state->owns_final_head = state->first_layer_index + state->layer_count == SPARK_DSV4_MODEL_LAYER_COUNT ? 1u : 0u;
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

static SparkStatus SparkDsv4ModuleValidateEntry(SparkDsv4ModuleState *state, const SparkDsv4StagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	SparkDsv4StagePackTensorShape shape;
	uint64_t payload_bytes,scale_bytes;
	uint32_t global = entry->layer_index == SPARK_DSV4_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	uint32_t in_slice = (entry->layer_index == SPARK_DSV4_STAGEPACK_MTP_LAYER && SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u) || (entry->layer_index >= state->first_layer_index && entry->layer_index < state->first_layer_index + state->layer_count) ? 1u : 0u;
	if ( entry->tensor_kind >= SPARK_DSV4_STAGEPACK_TENSOR_KIND_COUNT || (global == 0u && in_slice == 0u) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( SparkDsv4StagePackResolvedShape(entry->tensor_kind,global != 0u ? 0u : entry->layer_index,global,&shape) < 0 )
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
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING: state->token_embedding_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_LM_HEAD: state->lm_head_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_FN: state->hc_head_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_BASE: state->hc_head_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE: state->hc_head_scale_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_E_PROJ: SparkDsv4ModuleFillLinearView(&state->mtp.e_proj,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_H_PROJ: SparkDsv4ModuleFillLinearView(&state->mtp.h_proj,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_ENORM: state->mtp.enorm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HNORM: state->mtp.hnorm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_FINAL_NORM: state->mtp.final_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_FN: state->mtp.hc_head_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_BASE: state->mtp.hc_head_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE: state->mtp.hc_head_scale_f32 = (const float *)payload; break;
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
	SparkDsv4LayerWeights *layer = entry->layer_index == SPARK_DSV4_STAGEPACK_MTP_LAYER ? &state->mtp_layer : &state->layers[entry->layer_index];
	uint64_t *seen = entry->layer_index == SPARK_DSV4_STAGEPACK_MTP_LAYER ? &state->mtp_seen_bits : &state->layer_seen_bits[entry->layer_index];
	SparkStatus status;
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
	if ( state->owns_embedding != 0u || (state->owns_final_head != 0u && SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u) )
		expected_globals |= 1ull << SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
	{
		for (tensor = SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM; tensor <= SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE; tensor++)
			expected_globals |= 1ull << tensor;
		if ( SPARK_DSV4_MODEL_MTP_LAYER_COUNT != 0u )
		{
			for (tensor = SPARK_DSV4_STAGEPACK_TENSOR_MTP_E_PROJ; tensor <= SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE; tensor++)
				expected_globals |= 1ull << tensor;
			if ( state->mtp_seen_bits != SparkDsv4ModuleExpectedLayerBits(SPARK_DSV4_STAGEPACK_MTP_LAYER) )
			{
				fprintf(stderr,"%s pack_mtp_coverage seen=%llx\n",SPARK_DSV4_MODULE_TAG,(unsigned long long)state->mtp_seen_bits);
				return(SPARK_STATUS_VALIDATION_FAILED);
			}
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
	uint64_t vocab = SPARK_DSV4_MODEL_VOCAB_COUNT,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	cudaStream_t stream = (cudaStream_t)state->execution_stream;
	SparkStatus status;
	if ( state->owns_final_head == 0u )
		return(SPARK_STATUS_OK);
	status = SparkStageModuleDeviceAllocate(&state->ledger,(vocab * dim) / 2u,(void **)&state->head_shadow_payload);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(vocab * dim) / 32u,(void **)&state->head_shadow_scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,vocab * sizeof(float),(void **)&state->head_error_norm_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,SparkDsv4LaunchHeadShadowQuantize(stream,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,(uint32_t)vocab,(uint32_t)dim),"head_shadow_quantize");
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
	configuration.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS;
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
	state->resident_state_bytes = (uint64_t)state->physical_page_capacity *
		state->paged_cache.layout.page_stride_bytes;
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
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->slot_counts);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * state->topk_column_count * sizeof(int32_t),(void **)&slot->topk_idxs);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->emitted_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_MIX_ROWS * sizeof(float),(void **)&slot->mixes_f32);
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
	uint64_t stream_bytes = rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * dim * bf16;
	uint64_t compress_channels = (uint64_t)SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
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
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION * bf16,&slot->q_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * bf16,&slot->kv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION * bf16,&slot->attn_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT * SPARK_DSV4_MODEL_OUTPUT_LORA_RANK * bf16,&slot->o_ranks_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * compress_channels * bf16,&slot->compress_kv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * compress_channels * bf16,&slot->compress_score_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * compress_channels * sizeof(float),(void **)&slot->compress_kv_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * compress_channels * sizeof(float),(void **)&slot->compress_score_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * bf16,&slot->emit_bf16);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlotTail(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	uint64_t rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT,head_rows = SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	SparkStatus status;
	status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_INDEX_DIMENSION * bf16,&slot->index_q_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * bf16,&slot->index_weights_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * sizeof(float),(void **)&slot->index_weights_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * (uint64_t)state->index_slot_capacity * sizeof(float),(void **)&slot->index_scores_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->ffn_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->ffn_up_bf16);
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
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->moe_slot_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->moe_slot_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * dim * bf16,&slot->moe_slot_out_bf16);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,head_rows * SPARK_DSV4_MODEL_VOCAB_COUNT * bf16,&slot->head_logits_bf16);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,head_rows * SPARK_DSV4_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP * sizeof(uint32_t),(void **)&slot->head_candidate_ids_u32);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,head_rows * sizeof(uint32_t),(void **)&slot->head_candidate_counts_u32);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlot(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	SparkStatus status = SparkDsv4ModuleAllocateSlotSmall(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocateSlotWide(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocateSlotTail(state,slot);
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
		frame->program_id == 0u || frame->reserved != 0u ||
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
		context->reserved0 != 0u || (context->flags & ~known_flags) != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	decode_view = (context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) != 0u;
	prefill_view = (context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_BATCH_VIEW) != 0u;
	needs_input = state->stage_index > 0u ? 1u : 0u;
	needs_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
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
	needs_input = state->stage_index > 0u ? 1u : 0u;
	needs_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
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
	bytes = (uint64_t)frame->active_slot_count * sizeof(uint32_t);
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

// One mHC boundary: mix, split with Sinkhorn, reduce - the residual copy
// is the caller's, since attention and ffn share this exactly.
static cudaError_t SparkDsv4ModuleHcEnter(SparkDsv4ModuleSlot *slot, const void *streams_bf16, const float *fn, const float *scale3, const float *base, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	if ( streams_bf16 == 0 )
		return(cudaErrorInvalidValue);
	error = cudaMemcpyAsync(slot->residual_bf16,streams_bf16,(uint64_t)rows * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcMix(stream,streams_bf16,fn,slot->mixes_f32,rows,SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS,SPARK_DSV4_MODEL_HC_MIX_ROWS,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcSplitSinkhorn(stream,slot->mixes_f32,scale3,base,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HC_SINKHORN_ITERATIONS,SPARK_DSV4_MODEL_HC_EPSILON,slot->pre_f32,slot->post_f32,slot->comb_f32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcPreReduce(stream,streams_bf16,slot->pre_f32,slot->reduced_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	return(error);
}

/*
 * The compressor for one decode token, attention side: wkv/wgate on the
 * normalized x, widen to fp32, ape by in-group position, the state step,
 * and for boundary rows the emitted slot gets norm, rope at the group
 * start position, the fp8 cache sim, and lands at position/ratio behind
 * the window - the host knows the boundary from the position arithmetic.
 */
static cudaError_t SparkDsv4ModuleRunCompressor(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4CompressorWeights *weights, float *kv_state, float *score_state, uint64_t state_stride, void *cache_base, uint64_t cache_lane_stride, uint64_t cache_slot_offset, uint32_t cache_width, uint32_t rotate, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t channels,overlapped;
	uint64_t ratio = weights->ratio;
	cudaError_t error;
	if ( weights->overlap != 1u && weights->overlap != SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR )
		return(cudaErrorInvalidValue);
	channels = weights->overlap * cache_width;
	overlapped = weights->overlap > 1u ? 1u : 0u;
	error = SparkDsv4LaunchLinear(stream,&weights->wkv,slot->normalized_bf16,slot->compress_kv_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&weights->wgate,slot->normalized_bf16,slot->compress_score_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchWiden(stream,slot->compress_kv_bf16,slot->compress_kv_f32,rows,channels,1.0f);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchWiden(stream,slot->compress_score_bf16,slot->compress_score_f32,rows,channels,1.0f);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchApeAdd(stream,slot->compress_score_f32,weights->ape_f32,slot->row_positions,rows,(uint32_t)ratio,channels);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchCompressStep(stream,slot->compress_kv_f32,slot->compress_score_f32,kv_state,score_state,state_stride,slot->row_lane_indices,slot->row_positions,rows,(uint32_t)ratio,overlapped,cache_width,slot->emit_bf16,slot->emitted_u32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->emit_bf16,weights->norm_weight_bf16,slot->emit_bf16,rows,cache_width,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->emit_bf16,state->compress_freqs_f32,weights->ratio == SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO ? slot->row_emit_positions_hca : slot->row_emit_positions,rows,1u,cache_width,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess && rotate != 0u )
	{
		error = SparkDsv4LaunchHadamard(stream,slot->emit_bf16,rows,cache_width);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchQuantSim(stream,slot->emit_bf16,rows,cache_width,cache_width,SPARK_DSV4_MODEL_FP4_QUANT_BLOCK,1u);
	}
	if ( error == cudaSuccess && rotate == 0u )
		error = SparkDsv4LaunchQuantSim(stream,slot->emit_bf16,rows,cache_width,cache_width - SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,SPARK_DSV4_MODEL_KV_QUANT_BLOCK,0u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchCacheScatter(stream,slot->emit_bf16,slot->emitted_u32,cache_base,cache_lane_stride,slot->row_lane_indices,slot->row_positions,rows,cache_width,cache_slot_offset,(uint32_t)ratio,SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS);
	return(error);
}

static cudaError_t SparkDsv4ModuleRunIndexer(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	void *index_cache = (uint8_t *)state->index_cache_bf16 +
		state->index_cache_offset_by_layer[layer_index] *
		SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	float *kv_state = state->index_kv_state_f32 +
		state->index_kv_state_offset_by_layer[layer_index];
	float *score_state = state->index_score_state_f32 +
		state->index_score_state_offset_by_layer[layer_index];
	float weight_scale = 1.0f / sqrtf((float)SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION) / sqrtf((float)SPARK_DSV4_MODEL_INDEX_HEAD_COUNT);
	cudaError_t error;
	error = SparkDsv4LaunchLinear(stream,&layer->indexer.wq_b,slot->qr_bf16,slot->index_q_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->index_q_bf16,state->compress_freqs_f32,slot->row_positions,rows,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHadamard(stream,slot->index_q_bf16,rows * SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchQuantSim(stream,slot->index_q_bf16,rows,SPARK_DSV4_MODEL_INDEX_DIMENSION,SPARK_DSV4_MODEL_INDEX_DIMENSION,SPARK_DSV4_MODEL_FP4_QUANT_BLOCK,1u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->indexer.weights_proj,slot->normalized_bf16,slot->index_weights_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchWiden(stream,slot->index_weights_bf16,slot->index_weights_f32,rows,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,weight_scale);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunCompressor(state,slot,&layer->indexer.compressor,kv_state,score_state,state->index_state_lane_stride,index_cache,state->index_lane_stride,0u,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,1u,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchIndexerScore(stream,slot->index_q_bf16,index_cache,state->index_lane_stride,slot->row_page_table_indices,slot->physical_page_table,state->paged_cache.lane_page_capacity,SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO,slot->slot_counts,slot->index_weights_f32,slot->index_scores_f32,rows,state->index_slot_capacity,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchTopK(stream,slot->index_scores_f32,slot->slot_counts,state->index_slot_capacity,SPARK_DSV4_MODEL_INDEX_TOP_K,(int32_t)SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,slot->topk_idxs + SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,state->topk_column_count,rows);
	return(error);
}

static cudaError_t SparkDsv4ModuleStageTopk(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, uint32_t layer_kind, uint32_t rows)
{
	return(SparkDsv4LaunchBuildAttentionIndices((cudaStream_t)slot->cuda_stream,slot->row_positions,slot->topk_idxs,slot->slot_counts,rows,state->topk_column_count,state->index_slot_capacity,layer_kind));
}

/*
 * All o-composition groups in one launch: grid.z walks wo_a's contiguous
 * group blocks against each group's slice of the attention output, ranks
 * landing at the group's offset - block-diagonal through the strided kernel.
 */
static cudaError_t SparkDsv4ModuleRunOutputComposition(SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, uint32_t rows)
{
	SparkDsv4LinearView view = layer->attn.wo_a;
	uint64_t payload_stride = SparkDsv4StagePackPayloadBytes(view.weight_format,SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION);
	uint64_t scale_stride = SparkDsv4StagePackScaleBytes(view.weight_format,SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION);
	view.rows = SPARK_DSV4_MODEL_OUTPUT_LORA_RANK;
	view.columns = SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION;
	return(SparkDsv4LaunchStridedLinear((cudaStream_t)slot->cuda_stream,&view,view.payload,(const uint8_t *)view.scale_data,payload_stride,scale_stride,slot->attn_out_bf16,SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION,0u,SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION,slot->o_ranks_bf16,(uint64_t)SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT * SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,0u,SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT,rows));
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
	uint64_t q_offset = SparkDsv4PrefillRowElementOffset(first_row,SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION);
	uint64_t topk_offset = SparkDsv4PrefillRowElementOffset(first_row,state->topk_column_count);
	cudaError_t error;
	error = SparkDsv4LaunchCacheScatter(stream,(const uint8_t *)slot->kv_bf16 + kv_offset * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,0,cache,lane_stride,slot->row_lane_indices + first_row,slot->row_positions + first_row,rows,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,0u,0u,SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchSparseAttn(stream,(const uint8_t *)slot->q_bf16 + q_offset * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,cache,lane_stride,slot->row_lane_indices + first_row,slot->row_page_table_indices + first_row,slot->physical_page_table,state->paged_cache.lane_page_capacity,SparkDsv4PagedPoolCompressedEntries(layer_kind),slot->topk_idxs + topk_offset,state->topk_column_count,sink_f32,1.0f / sqrtf((float)SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION),(uint8_t *)slot->attn_out_bf16 + q_offset * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,rows,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION);
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

static cudaError_t SparkDsv4ModuleRunAttention(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, const SparkDsv4PrefillBatchView *prefill, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t kind = SparkDsv4ModelLayerKind(layer_index);
	const float *freqs = kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA ? state->base_freqs_f32 : state->compress_freqs_f32;
	uint64_t state_offset = state->compress_state_offset_by_layer[layer_index];
	uint64_t state_stride = state->compress_state_lane_stride_by_layer[layer_index];
	void *cache = (uint8_t *)state->kv_cache_bf16 + state->cache_offset_by_layer[layer_index] * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t lane_stride = state->cache_lane_stride_by_layer[layer_index];
	cudaError_t error;
	error = SparkDsv4LaunchLinear(stream,&layer->attn.wq_a,slot->normalized_bf16,slot->delta_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->delta_bf16,layer->attn.q_norm_weight_bf16,slot->qr_bf16,rows,SPARK_DSV4_MODEL_QUERY_LORA_RANK,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->attn.wq_b,slot->qr_bf16,slot->q_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchQueryHeadRms(stream,slot->q_bf16,rows,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->q_bf16,freqs,slot->row_positions,rows,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->attn.wkv,slot->normalized_bf16,slot->kv_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->kv_bf16,layer->attn.kv_norm_weight_bf16,slot->kv_bf16,rows,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->kv_bf16,freqs,slot->row_positions,rows,1u,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchQuantSim(stream,slot->kv_bf16,rows,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION - SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,SPARK_DSV4_MODEL_KV_QUANT_BLOCK,0u);
	if ( error == cudaSuccess && kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		error = SparkDsv4ModuleRunCompressor(state,slot,&layer->compressor,state->compress_kv_state_f32 + state_offset,state->compress_score_state_f32 + state->compress_score_state_offset_by_layer[layer_index],state_stride,cache,lane_stride,SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,0u,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleStageTopk(state,slot,kind,rows);
	if ( error == cudaSuccess && kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		error = SparkDsv4ModuleRunIndexer(state,slot,layer,layer_index,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunCausalAttention(state,slot,prefill,cache,lane_stride,layer->attn.sink_f32,kind,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->attn_out_bf16,freqs,slot->row_positions,rows,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,1u);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunOutputComposition(slot,layer,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->attn.wo_b,slot->o_ranks_bf16,slot->delta_bf16,rows);
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
static SparkStatus SparkDsv4ModuleRunMoeRouted(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4MoeWeights *moe, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkDsv4LaunchMoeRoute(stream,slot->moe_indices_u32,rows,slot->expert_offsets_u32,slot->moe_inverse_u32,slot->grouped_rows_u32,slot->group_tile_prefix_w1_u32,slot->group_tile_prefix_w2_u32);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_route"));
	error = SparkDsv4LaunchExpertUp(stream,&moe->experts_w1,slot->normalized_bf16,slot->grouped_rows_u32,slot->expert_offsets_u32,slot->group_tile_prefix_w1_u32,slot->moe_slot_gate_bf16,rows,state->multiprocessor_count);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_expert_w1"));
	error = SparkDsv4LaunchExpertUp(stream,&moe->experts_w3,slot->normalized_bf16,slot->grouped_rows_u32,slot->expert_offsets_u32,slot->group_tile_prefix_w1_u32,slot->moe_slot_up_bf16,rows,state->multiprocessor_count);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_expert_w3"));
	error = SparkDsv4LaunchSwigluClamp(stream,slot->moe_slot_gate_bf16,slot->moe_slot_up_bf16,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION,SPARK_DSV4_MODEL_SWIGLU_LIMIT,0,0);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_swiglu"));
	error = SparkDsv4LaunchExpertDown(stream,&moe->experts_w2,slot->moe_slot_up_bf16,slot->expert_offsets_u32,slot->group_tile_prefix_w2_u32,slot->moe_slot_out_bf16,rows,state->multiprocessor_count);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_expert_w2"));
	error = SparkDsv4LaunchMoePairReduce(stream,slot->moe_slot_out_bf16,slot->moe_inverse_u32,slot->moe_weights_f32,slot->ffn_accum_bf16,rows);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_reduce"));
}

static SparkStatus SparkDsv4ModuleRunMoe(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkDsv4MoeWeights *moe = &layer->moe;
	uint32_t hash = SparkDsv4StagePackLayerIsHashRouted(layer_index);
	cudaError_t error;
	error = SparkDsv4LaunchGateScores(stream,&moe->gate,slot->normalized_bf16,slot->moe_scores_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchGateSelect(stream,slot->moe_scores_f32,hash != 0u ? 0 : moe->gate_bias_f32,hash != 0u ? moe->gate_tid2eid_u32 : 0,slot->input_token_ids,rows,SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT,SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,SPARK_DSV4_MODEL_ROUTED_SCALING_FACTOR,slot->moe_indices_u32,slot->moe_weights_f32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&moe->shared_w1,slot->normalized_bf16,slot->ffn_gate_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&moe->shared_w3,slot->normalized_bf16,slot->ffn_up_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchSwigluClamp(stream,slot->ffn_gate_bf16,slot->ffn_up_bf16,rows,SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION,SPARK_DSV4_MODEL_SWIGLU_LIMIT,0,0);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&moe->shared_w2,slot->ffn_up_bf16,slot->ffn_accum_bf16,rows);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_shared"));
	return(SparkDsv4ModuleRunMoeRouted(state,slot,moe,rows));
}

static SparkStatus SparkDsv4ModuleRunLayer(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const void *input_streams_bf16, void *output_streams_bf16, const SparkDsv4PrefillBatchView *prefill, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkDsv4LayerWeights *layer = &state->layers[layer_index];
	cudaError_t error;
	SparkStatus status;
	error = SparkDsv4ModuleHcEnter(slot,input_streams_bf16,layer->hc.attn_fn_f32,layer->hc.attn_scale_f32,layer->hc.attn_base_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,layer->attn_norm_bf16,slot->normalized_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunAttention(state,slot,layer,prefill,layer_index,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcPost(stream,slot->delta_bf16,slot->residual_bf16,slot->post_f32,slot->comb_f32,slot->streams_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"attn_side"));
	error = SparkDsv4ModuleHcEnter(slot,slot->streams_bf16,layer->hc.ffn_fn_f32,layer->hc.ffn_scale_f32,layer->hc.ffn_base_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,layer->ffn_norm_bf16,slot->normalized_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"ffn_enter"));
	status = SparkDsv4ModuleRunMoe(state,slot,layer,layer_index,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	error = SparkDsv4LaunchHcPost(stream,slot->ffn_accum_bf16,slot->residual_bf16,slot->post_f32,slot->comb_f32,output_streams_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"ffn_side"));
}

static SparkStatus SparkDsv4ModuleRunLayers(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const void *input_streams_bf16, void *output_streams_bf16, const SparkDsv4PrefillBatchView *prefill, uint32_t rows)
{
	uint32_t layer;
	const void *layer_input;
	void *layer_output;
	SparkStatus status = SPARK_STATUS_OK;
	layer_input = input_streams_bf16;
	for (layer = state->first_layer_index; status == SPARK_STATUS_OK && layer < state->first_layer_index + state->layer_count; layer++)
	{
		layer_output = layer + 1u == state->first_layer_index + state->layer_count ? output_streams_bf16 : slot->streams_bf16;
		status = SparkDsv4ModuleRunLayer(state,slot,layer_input,layer_output,prefill,layer,rows);
		layer_input = layer_output;
	}
	return(status);
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
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHeadScreenedArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids_u32,slot->head_candidate_counts_u32,output_token_ids,rows,SPARK_DSV4_MODEL_VOCAB_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
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

static SparkStatus SparkDsv4ModuleRunPrefillHead(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4PrefillBatchView *prefill)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = cudaMemsetAsync(slot->output_token_ids,0,(uint64_t)prefill->active_sequence_count * sizeof(uint32_t),stream);
	if ( error == cudaSuccess && prefill->emit_count != 0u )
		error = cudaMemcpyAsync(slot->emitted_u32,prefill->emit_row_indices,(uint64_t)prefill->emit_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && prefill->emit_count != 0u )
		error = cudaMemcpyAsync(slot->row_lane_indices,prefill->emit_lane_indices,(uint64_t)prefill->emit_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && prefill->emit_count != 0u )
		error = SparkDsv4LaunchGatherHeadRows(stream,slot->streams_bf16,slot->emitted_u32,slot->residual_bf16,prefill->emit_count,SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS);
	if ( error == cudaSuccess && prefill->emit_count != 0u )
		error = SparkDsv4ModuleProjectHead(state,slot,slot->residual_bf16,slot->slot_counts,prefill->emit_count);
	if ( error == cudaSuccess && prefill->emit_count != 0u )
		error = SparkDsv4LaunchScatterHeadTokens(stream,slot->slot_counts,slot->row_lane_indices,slot->output_token_ids,prefill->emit_count);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->host_output_token_ids,slot->output_token_ids,(uint64_t)prefill->active_sequence_count * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"prefill_head"));
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
	uint32_t rows;
	SparkStatus status;
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? context->prefill_batch : 0;
	rows = prefill != 0 ? prefill->row_count : context->decode_batch->row_count;
	status = SparkDsv4ModuleBeginStreams(state,slot,context,rows,&input_streams_bf16);
	if ( status == SPARK_STATUS_OK )
	{
		output_streams_bf16 = state->owns_final_head != 0u ? slot->streams_bf16 : context->hidden_output_bf16;
		status = SparkDsv4ModuleRunLayers(state,slot,input_streams_bf16,output_streams_bf16,prefill,rows);
	}
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u && prefill != 0 )
		status = SparkDsv4ModuleRunPrefillHead(state,slot,prefill);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u && prefill == 0 )
		status = SparkDsv4ModuleFinish(state,slot,slot->streams_bf16,slot->host_output_token_ids,rows);
	return(status);
}

/*
 * Decode-step CUDA graphs, per docs/PERF_ROADMAP_2026-08-01.md D10/D1: the
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
 * Fail-closed: any capture or replay-launch failure runs the step eagerly;
 * a replay never substitutes for a shape it was not recorded from.
 */
static SparkStatus SparkDsv4ModuleBounceBoundary(void *destination, const void *source, uint32_t rows, cudaStream_t stream, const char *site)
{
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,cudaMemcpyAsync(destination,source,(uint64_t)rows * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,stream),site));
}

// The exact launch sequence a decode graph records; also the eager fallback
// body, so capture and eager can never drift apart.
static SparkStatus SparkDsv4ModuleRunCapturedDecode(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, uint32_t rows)
{
	const void *input_streams_bf16;
	SparkStatus status;
	input_streams_bf16 = slot->streams_bf16;
	status = SparkDsv4ModuleStageRowCopies(state,slot,rows,rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleInitializeFramePages(state,slot,rows);
	if ( status == SPARK_STATUS_OK && state->owns_embedding != 0u )
		status = SparkDsv4ModuleBeginStreams(state,slot,0,rows,&input_streams_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleRunLayers(state,slot,input_streams_bf16,slot->streams_bf16,0,rows);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkDsv4ModuleFinish(state,slot,slot->streams_bf16,slot->host_output_token_ids,rows);
	return(status);
}

static SparkStatus SparkDsv4ModuleCaptureDecode(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const LmGraphKey *key, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	LmGraphEntry *entry;
	SparkStatus status;
	if ( LmGraphBeginCapture(stream) != LM_GRAPH_OK )
	{
		state->graph_sealed = 1u;
		return(SparkDsv4ModuleRunCapturedDecode(state,slot,rows));
	}
	status = SparkDsv4ModuleRunCapturedDecode(state,slot,rows);
	if ( status == SPARK_STATUS_OK && LmGraphEndCapture(&state->graph_cache,key,stream) == LM_GRAPH_OK )
	{
		// Captured work only recorded; replay it so this frame executes.
		if ( LmGraphReplay(&state->graph_cache,key,stream) == LM_GRAPH_OK )
			return(SPARK_STATUS_OK);
		state->graph_sealed = 1u;
		return(SparkDsv4ModuleRunCapturedDecode(state,slot,rows));
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
	return(SparkDsv4ModuleRunCapturedDecode(state,slot,rows));
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
		status = SparkDsv4ModuleRunCapturedDecode(state,slot,rows);
	if ( status == SPARK_STATUS_OK && state->owns_final_head == 0u )
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
	async->state = state;
	async->completion_function = frame->completion_function;
	async->completion_context = frame->completion_context;
	async->slot_index = slot_index;
	async->lane_count = lane_count;
	async->row_count = row_count;
	async->emitted_token_count = emitted_token_count;
	for (index=0u; index<lane_count; index++)
	{
		async->lane_indices[index] = lane_indices[index];
		async->lane_sequence_ids[index] = lane_sequence_ids[index];
		async->lane_next_positions[index] = lane_next_positions[index];
	}
	status = SparkDsv4ModuleCopyCacheLanes(async,frame,context);
	if ( status != SPARK_STATUS_OK )
		return(status);
	async->output_token_destination = state->owns_final_head != 0u ? (uint32_t *)frame->buffers[1u].address : 0;
	async->completion.request_id = frame->request_id;
	async->completion.sequence_id = frame->sequence_id;
	async->completion.sequence_position = frame->sequence_position;
	async->completion.program_id = frame->program_id;
	async->completion.driver_dispatch_slot = frame->driver_dispatch_slot;
	async->completion.accepted_token_count = frame->new_token_count;
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
		(void)SparkDsv4ModuleRollbackCacheLanesLocked(async);
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
	SparkStatus result,status;
	uint32_t index;
	state = async->state;
	slot = &state->slots[async->slot_index];
	if ( pthread_mutex_lock(&state->cache_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	result = SPARK_STATUS_OK;
	for (index=0u; index<async->prepared_cache_lane_count; index++)
	{
		status = SparkDsv4PagedCacheUnpinLane(&state->paged_cache,
			slot->host_logical_page_table +
			(uint64_t)index * state->paged_cache.lane_page_capacity,
			async->prepared_cache_lanes[index].logical_page_count);
		if ( result == SPARK_STATUS_OK && status != SPARK_STATUS_OK )
			result = status;
	}
	if ( completed != 0u && result == SPARK_STATUS_OK )
		for (index=0u; result==SPARK_STATUS_OK &&
			index<async->prepared_cache_lane_count; index++)
			result = SparkDsv4PagedCacheCompleteLane(&state->paged_cache,
				&async->cache_lanes[index],
				&async->prepared_cache_lanes[index]);
	if ( completed == 0u || result != SPARK_STATUS_OK )
		(void)SparkDsv4ModuleAbortCacheLanesLocked(async);
	else
		async->prepared_cache_lane_count = 0u;
	(void)pthread_mutex_unlock(&state->cache_mutex);
	return(result);
}

static void CUDART_CB SparkDsv4ModuleCompleteAsync(void *context)
{
	SparkDsv4AsyncCompletion *async;
	SparkDsv4ModuleState *state;
	uint32_t index,lane;
	async = (SparkDsv4AsyncCompletion *)context;
	state = async != 0 ? async->state : 0;
	if ( state == 0 || async->slot_index >= state->pipeline_slot_count )
		return;
	if ( SparkDsv4ModuleFinishCachePages(async,
		async->completion.status == SPARK_STATUS_OK ? 1u : 0u) !=
		SPARK_STATUS_OK )
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
	if ( async->completion.status == SPARK_STATUS_OK )
	{
		if ( async->output_token_destination != 0 )
			memcpy(async->output_token_destination,state->slots[async->slot_index].host_output_token_ids,(uint64_t)async->lane_count * sizeof(uint32_t));
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
	uint32_t emitted_token_count,lane_count,row_count,slot_index;
	SparkStatus status;
	lane_count = frame->active_slot_count;
	row_count = frame->new_token_count;
	emitted_token_count = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? context->prefill_batch->emit_count : lane_count;
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
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModulePrepareCachePages(async);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleStagePageTableUpdates(async);
	if ( status == SPARK_STATUS_OK )
		atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	if ( status == SPARK_STATUS_OK )
	{
		if ( state->graph_capacity != 0u && (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) == 0u )
			status = SparkDsv4ModuleRunGraphedDecode(state,slot,slot_index,frame,context,row_count);
		else
		{
			status = SparkDsv4ModuleStageFrameRows(state,slot,frame,context);
			if ( status == SPARK_STATUS_OK )
				status = SparkDsv4ModuleRunFrame(state,slot,frame,context);
		}
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleEnqueueAsync(state,slot,slot_index);
	if ( status != SPARK_STATUS_OK )
	{
		(void)cudaStreamSynchronize((cudaStream_t)slot->cuda_stream);
		if ( async->prepared_cache_lane_count != 0u )
			(void)SparkDsv4ModuleFinishCachePages(async,0u);
		if ( async->aborted_cache_lane_count != 0u )
			SparkDsv4ModuleInvalidateLanes(state,lane_indices,
				async->aborted_cache_lane_count);
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
		SparkStageModuleSlotRelease(state->slot_states,slot_index);
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,lane_indices,lane_count);
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
		frame->program_id == 0u || frame->reserved != 0u ||
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
	const SparkDsv4ResidentDecodeStageFrameContext *context;
	SparkStatus status;
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

static SparkStatus SparkDsv4ModulePrepareCacheAdmission(
	SparkDsv4ModuleState *state,
	const SparkModelDriverAdmissionRequest *request)
{
	SparkStatus status;
	uint32_t lane;
	if ( pthread_mutex_lock(&state->cache_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	status = SPARK_STATUS_OK;
	for (lane=0u; status==SPARK_STATUS_OK &&
		lane<request->cache_lane_count; lane++)
		status = SparkDsv4PagedCachePrefetchLane(&state->paged_cache,
			&request->cache_lanes[lane],state->cache_admission_logical_pages,
			state->paged_cache.lane_page_capacity);
	(void)pthread_mutex_unlock(&state->cache_mutex);
	return(status);
}

SparkStatus SparkDsv4ResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkDsv4ModuleState *state;
	uint32_t available,is_prefill,preparing,releasing;
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
    uint32_t slot_index;

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
		if ( state->slots[slot_index].host_staging != 0 )
			(void)cudaFreeHost(state->slots[slot_index].host_staging);
	}
	for (slot_index = 0u; slot_index < state->graph_capacity; ++slot_index)
	{
		if ( state->graph_entries[slot_index].live != 0 )
			(void)cudaGraphExecDestroy(state->graph_entries[slot_index].executable);
	}
	free(state->cache_admission_logical_pages);
	state->cache_admission_logical_pages = 0;
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
	status = SparkDsv4ModuleConfigure(state,host_services,pack_path_out);
	if ( status != SPARK_STATUS_OK )
	{
		(void)pthread_mutex_destroy(&state->cache_mutex);
		free(state);
		return(status);
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
		status = SparkDsv4ModuleAllocateSlot(state,&state->slots[slot_index]);
	LmGraphCacheInitialise(&state->graph_cache,state->graph_entries,state->graph_capacity);
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
