#include "sparkpipe/spark_glm52_pp13_node_context_builder.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "sparkpipe/spark_glm52_rope.h"
#include "sparkpipe/spark_glm52_dspark_draft_backend.h"
#include "sparkpipe/spark_glm52_production_topology.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_b12x_moe_plan.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_fp8_moe_plan.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_linear_plan.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_production_runner.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_required_cuda.h"
#include "sparkpipe/spark_glm52_resident_decode_stage_w8lut_moe_plan.h"
#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_glm52_mtp_tree.h"
#include "sparkpipe/spark_glm52_stage_plan.h"
#include "sparkpipe/spark_glm52_stagepack.h"

#define SPARK_GLM52_PP13_BUILDER_LAYER_CAPACITY \
	SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_STAGE_SLICE_LAYER_COUNT
#define SPARK_GLM52_PP13_BUILDER_PIPELINE_SLOT_COUNT 1u
#define SPARK_GLM52_PP13_BUILDER_POSITION_COUNT SPARK_GLM52_KV_CONTEXT_TOKENS
#define SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE \
	(SPARK_GLM52_KV_CONTEXT_TOKENS / SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS)
#define SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS 1024u
#define SPARK_GLM52_PP13_BUILDER_COPY_CHUNK_BYTES (64ull * 1024ull * 1024ull)
#define SPARK_GLM52_PP13_BUILDER_EXECUTION_LAYER_COUNT \
	(SPARK_GLM52_PP13_BUILDER_LAYER_CAPACITY + 1u)
#define SPARK_GLM52_PP13_BUILDER_LAYER_BUFFER_ALLOCATION_COUNT 61u
#define SPARK_GLM52_PP13_BUILDER_LAYER_WEIGHT_ALLOCATION_COUNT 29u
#define SPARK_GLM52_PP13_BUILDER_SHARED_BUFFER_ALLOCATION_COUNT 10u
#define SPARK_GLM52_PP13_BUILDER_TABLE_ALLOCATION_COUNT 15u
#define SPARK_GLM52_PP13_BUILDER_INPUT_ALLOCATION_COUNT 12u
#define SPARK_GLM52_PP13_BUILDER_PLAN_ALLOCATION_COUNT 2u
#define SPARK_GLM52_PP13_BUILDER_MTP_SUPPORT_ALLOCATION_COUNT 17u
#define SPARK_GLM52_PP13_BUILDER_FINAL_OUTPUT_ALLOCATION_COUNT 2u
#define SPARK_GLM52_PP13_BUILDER_MAX_ALLOCATIONS \
	((SPARK_GLM52_PP13_BUILDER_EXECUTION_LAYER_COUNT * \
	  (SPARK_GLM52_PP13_BUILDER_LAYER_BUFFER_ALLOCATION_COUNT + \
	   SPARK_GLM52_PP13_BUILDER_LAYER_WEIGHT_ALLOCATION_COUNT)) + \
	 SPARK_GLM52_PP13_BUILDER_SHARED_BUFFER_ALLOCATION_COUNT + \
	 SPARK_GLM52_PP13_BUILDER_TABLE_ALLOCATION_COUNT + \
	 SPARK_GLM52_PP13_BUILDER_INPUT_ALLOCATION_COUNT + \
	 SPARK_GLM52_PP13_BUILDER_PLAN_ALLOCATION_COUNT + \
	 SPARK_GLM52_PP13_BUILDER_MTP_SUPPORT_ALLOCATION_COUNT + \
	 SPARK_GLM52_PP13_BUILDER_FINAL_OUTPUT_ALLOCATION_COUNT)
#define SPARK_GLM52_PP13_BUILDER_THREADS 256u
#define SPARK_GLM52_PP13_BUILDER_PROBE_HASH_SLOT_COUNT 18u
#define SPARK_GLM52_PP13_BUILDER_MAX_PREFILL_TOKENS \
	SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MAX_PREFILL_TOKENS
#define SPARK_GLM52_PP13_BUILDER_INVALID_SLOT UINT32_MAX
#define SPARK_GLM52_PP13_BUILDER_SPECULATIVE_VERIFY_TARGET_COUNT \
	(SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT + 1u)
#define SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_NONE 0u
#define SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_READY 1u
#define SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_GPU_PENDING 2u
#define SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION \
	(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION * 2u)
#define SPARK_GLM52_PP13_BUILDER_MTP_NORM_COUNT_PER_LANE 2u
#define SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS \
	(SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT * \
	 SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION)
#define SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS \
	(SPARK_GLM52_RESIDENT_DECODE_STAGE_HEAD_COUNT * \
	 SPARK_GLM52_RESIDENT_DECODE_STAGE_VALUE_HEAD_DIMENSION)
#define SPARK_GLM52_PP13_BUILDER_MTP_EMBEDDING_TENSOR \
	"sparkpipe.mtp.embed_tokens.weight"
#define SPARK_GLM52_PP13_BUILDER_NVME_RECORD_MAGIC 0x564b4e53u
#define SPARK_GLM52_PP13_BUILDER_NVME_RECORD_ABI_VERSION 4u
#define SPARK_GLM52_PP13_BUILDER_NVME_ALIGNMENT 4096u
#define SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_PHASE_COUNT 7u
#define SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK 128u
#define SPARK_GLM52_PP13_BUILDER_FP8_E4M3_MAX 448.0f
#define SPARK_GLM52_PP13_BUILDER_FP8_MLA_SCALE_COUNT \
	((SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS + \
	  SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_SCALE_BLOCK - 1u) / \
	 SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_SCALE_BLOCK)
#define SPARK_GLM52_PP13_BUILDER_FP8_KEY_NOPE_SCALE_COUNT \
	((SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS + \
	  SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_SCALE_BLOCK - 1u) / \
	 SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_SCALE_BLOCK)
#define SPARK_GLM52_PP13_BUILDER_FP8_VALUE_SCALE_COUNT \
	((SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS + \
	  SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_SCALE_BLOCK - 1u) / \
	 SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_SCALE_BLOCK)

typedef enum SparkGlm52Pp13BuilderMtpGpuProfilePhase
{
	SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_START = 0,
	SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_METADATA = 1,
	SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_FUSION = 2,
	SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_EH_PROJECTION = 3,
	SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_REQUIRED_LAYER = 4,
	SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_VOCAB_HEAD = 5,
	SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_STORE = 6
} SparkGlm52Pp13BuilderMtpGpuProfilePhase;

typedef struct SparkGlm52Pp13BuilderNvmeRecordHeader
{
	uint32_t magic;
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t rank_index;
	uint32_t layer_count;
	uint32_t block_token_count;
	uint32_t cache_token_elements;
	uint32_t index_key_dimension;
	uint32_t mtp_layer_included;
	uint32_t attention_cache_layout;
	uint64_t dsa_index_layer_mask;
	uint64_t sequence_id;
	uint32_t logical_block_index;
	uint32_t backing_block_index;
	uint64_t payload_bytes;
	uint64_t record_bytes;
} SparkGlm52Pp13BuilderNvmeRecordHeader;

typedef struct SparkGlm52Pp13BuilderNvmePendingOperation
{
	uint64_t sequence_id;
	uint32_t logical_block_index;
	uint32_t physical_block_index;
	uint32_t backing_block_index;
	uint32_t reserved0;
} SparkGlm52Pp13BuilderNvmePendingOperation;

typedef struct SparkGlm52Pp13BuilderLayer
{
	SparkGlm52ResidentDecodeStageNodeContext node;
	SparkGlm52ResidentDecodeStagePipelineSlot slot;
	SparkGlm52ResidentDecodeStageCudaPipelineSlotState cuda_slot;
	SparkGlm52ResidentDecodeStageLinearPlanResidentBinding *linear_binding;
	SparkGlm52ResidentDecodeStageB12xMoeResidentBinding b12x_moe_binding;
	SparkGlm52ResidentDecodeStageFp8MoeResidentBinding fp8_moe_binding;
	SparkGlm52ResidentDecodeStageW8lutMoeResidentBinding w8lut_moe_binding;
	uint32_t b12x_moe_ready;
	uint32_t fp8_moe_ready;
	uint32_t w8lut_moe_ready;
	void *raw_query_a_weight_bf16;
	void *raw_query_a_weight_fp8;
	void *raw_query_a_scale;
	void *raw_query_b_weight_bf16;
	void *raw_query_b_weight_fp8;
	void *raw_query_b_scale;
	void *raw_kv_a_weight_bf16;
	void *raw_kv_a_weight_fp8;
	void *raw_kv_a_scale;
	void *raw_kv_b_weight_bf16;
	void *raw_kv_b_weight_fp8;
	void *raw_kv_b_scale;
	void *attention_output_weight_bf16;
	void *attention_output_weight_fp8;
	void *attention_output_scale;
	void *attention_norm_weight;
	void *raw_query_a_norm_weight;
	void *raw_kv_a_norm_weight;
	void *post_attention_norm_weight;
	void *dense_gate_weight_bf16;
	void *dense_gate_weight_fp8;
	void *dense_gate_scale;
	void *dense_up_weight_bf16;
	void *dense_up_weight_fp8;
	void *dense_up_scale;
	void *dense_down_weight_bf16;
	void *dense_down_weight_fp8;
	void *dense_down_scale;
	void *router_weight;
	void *router_bias;
	void *index_query_weight_bf16;
	void *index_query_weight_fp8;
	void *index_query_scale;
	void *index_key_weight_bf16;
	void *index_key_weight_fp8;
	void *index_key_scale;
	void *index_weights_proj_weight;
	void *index_key_norm_weight;
	void *index_key_norm_bias;
	void *final_norm_weight;
	void *restricted_lm_head_weight;
	void *input_hidden;
	void *normalized_hidden;
	void *query_latent;
	void *query_rope_input;
	void *key_rope_input;
	void *current_kv_latent;
	void *raw_query_a;
	void *raw_query_a_norm;
	void *raw_query_b;
	void *raw_kv_a;
	void *raw_kv_a_norm;
	void *raw_kv_b;
	void *query_index_heads;
	void *current_key_index;
	void *index_head_weights;
	void *sparse_token_indices;
	void *rotated_query_rope;
	void *attention_output_latent;
	void *attention_projected_hidden;
	void *post_attention_hidden;
	void *post_attention_normalized_hidden;
	void *moe_topk_expert_ids;
	void *moe_topk_weights;
	void *moe_router_logits;
	void *moe_gate;
	void *moe_up;
	void *moe_intermediate;
	void *moe_route_output;
	void *layer_output_hidden;
	void *mtp_draft_hidden;
	void *restricted_logits;
	void *restricted_selected_token_ids;
	void *restricted_selected_token_scores;
	void *mtp_draft_logits;
	void *mtp_draft_token_ids;
	void *mtp_draft_token_budgets;
	void *mtp_target_token_ids;
	void *mtp_accept_mask;
	void *mtp_committed_token_ids;
	void *mtp_event_counters;
	void *phase_clock_cycles;
	void *positions;
	void *slot_mapping;
	void *block_table;
	void *context_lengths;
	void *first_block_token_offsets;
	void *mla_cache;
	void *key_nope_cache;
	void *value_cache;
	void *mla_cache_fp8;
	void *mla_cache_scale;
	void *key_nope_cache_fp8;
	void *key_nope_cache_scale;
	void *value_cache_fp8;
	void *value_cache_scale;
	void *key_index_cache;
	void *key_index_block_min;
	void *key_index_block_max;
	void *dsa_summary_dirty_flags;
	cudaEvent_t dsa_selection_event;
	cudaEvent_t dsa_prefetch_event;
	SparkGlm52ResidentDecodeStageFp8KvCachePlan fp8_kv_cache_plan;
	SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPlan dsa_prefetch_plan;
} SparkGlm52Pp13BuilderLayer;

typedef struct SparkGlm52Pp13BuilderMtpKvTransaction
{
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t base_position;
	uint64_t last_request_id;
	uint64_t last_sequence_id;
	uint64_t last_base_position;
	uint32_t proposed_token_count;
	uint32_t pinned_token_count;
	uint32_t active;
	uint32_t tree_verify;
	uint32_t transient_block_count;
	uint32_t last_proposed_token_count;
	uint32_t last_accepted_token_count;
	uint32_t last_resolution_path_id;
	uint32_t last_valid;
	uint32_t shadow_slot_index;
	uint32_t shadow_valid;
	uint32_t physical_slots[SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT];
	uint32_t transient_physical_blocks[
		SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_BLOCK_COUNT];
} SparkGlm52Pp13BuilderMtpKvTransaction;

typedef struct SparkGlm52Pp13BuilderState
{
	SparkGlm52Pp13NodeContextBuilderConfiguration configuration;
	SparkGlm52Pp13RuntimeRankPlan rank_plan;
	SparkGlm52Pp13BuilderLayer layers[SPARK_GLM52_PP13_BUILDER_LAYER_CAPACITY];
	SparkGlm52Pp13BuilderLayer mtp_layer;
	SparkGlm52ResidentDecodeStageMtpDraftPlan mtp_draft_plan;
	const SparkGlm52ResidentDecodeStageNodeContext *layer_pointers[
		SPARK_GLM52_PP13_BUILDER_LAYER_CAPACITY];
	SparkGlm52ResidentDecodeStageSliceNodeContext slice_context;
	SparkGlm52ResidentDecodeStageStageSlicePlan stage_slice_plan;
	SparkGlm52ResidentDecodeStageExactStageSlicePlan exact_plan;
	SparkGlm52Sm121RequiredDecodeStageBuiltinFp8ScaledGemmState fp8_scaled_gemm_state;
	SparkGlm52Sm121RequiredDecodeStageFp8ScaledGemmBackend fp8_scaled_gemm_backend;
	SparkGlm52ResidentDecodeStageProductionRunner runner;
	SparkGlm52ProductionTopology production_topology;
	SparkGlm52DsparkHiddenTapPlan dspark_tap_plan;
	SparkGlm52DsparkDraftBackend dspark_backend;
	SparkGlm52DsparkModelContract dspark_model_contract;
	void *dspark_tap_outputs_bf16[SPARK_GLM52_DSPARK_AUX_LAYER_COUNT];
	uint64_t dspark_tap_lane_stride_bytes;
	SparkGlm52DsparkDraftBackendStage *dspark_stage_batch;
	SparkGlm52DsparkDraftRequest *dspark_draft_batch;
	SparkGlm52DsparkDraftResult *dspark_batch_results;
	SparkGlm52DsparkDraftResult *dspark_ready_drafts;
	uint32_t *dspark_lane_by_request_slot;
	uint32_t *dspark_request_slot_by_lane;
	SparkModelDriverCompletion captured_completion;
	SparkModelDriverCompletion pending_driver_completion;
	SparkGlm52Pp13WorkControlPacket pending_work_packet;
	SparkModelDriverCompletionFunction pending_work_completion_function;
	void *pending_work_completion_context;
	SparkModelDriverCompletion *pending_work_completions;
	uint32_t dspark_backend_ready;
	uint32_t dspark_ready_draft_head;
	uint32_t dspark_ready_draft_count;
	uint32_t pending_finalizer_state;
	uint32_t dspark_stage_count;
	uint32_t dspark_draft_count;
	uint32_t captured_completion_valid;
	uint32_t pending_work_active;
	uint32_t pending_work_completion_count;
	uint32_t pending_work_completion_overflow;
	SparkStatus last_work_completion_status;
	uint64_t asynchronous_submit_count;
	uint64_t asynchronous_completion_count;
	uint64_t asynchronous_failure_count;
	uint64_t layer_major_submit_count;
	uint64_t layer_major_completion_count;
	uint64_t layer_major_failure_count;
	uint32_t last_layer_major_logical_lane_count;
	uint32_t last_layer_major_rows_per_lane;
	uint32_t last_layer_major_execution_row_count;
	SparkGlm52Pp13WorkControlKvState kv_state;
	SparkGlm52KvBlockTableView host_kv_view;
	SparkGlm52KvBlockTableView device_kv_view;
	SparkGlm52Pp13NodeContextBuilderResult result;
	void *allocations[SPARK_GLM52_PP13_BUILDER_MAX_ALLOCATIONS];
	uint8_t allocation_is_host_mapped[SPARK_GLM52_PP13_BUILDER_MAX_ALLOCATIONS];
	uint32_t allocation_count;
	uint64_t cuda_total_bytes;
	uint64_t cuda_initial_free_bytes;
	uint64_t cuda_builder_allocation_bytes;
	uint64_t cuda_largest_allocation_bytes;
	uint64_t host_mapped_allocation_bytes;
	uint32_t moe_bound_layer_count;
	uint32_t moe_expected_layer_count;
	uint32_t fp8_scaled_gemm_bound_plan_count;
	uint32_t fp8_scaled_gemm_expected_plan_count;
	uint32_t active_kv_block_count;
	uint32_t *host_physical_block_indices;
	uint32_t *host_lane_physical_block_counts;
	uint32_t *host_uploaded_physical_block_indices;
	uint32_t *host_uploaded_lane_physical_block_counts;
	uint8_t *host_uploaded_lane_valid;
	uint8_t *host_physical_block_states;
	uint64_t *host_physical_block_sequence_ids;
	uint32_t *host_physical_block_logical_indices;
	uint64_t *host_physical_block_last_used_epochs;
	uint32_t *host_physical_block_pin_counts;
	SparkGlm52Pp13WorkControlKvDirectoryEntry *host_kv_directory_entries;
	SparkGlm52Pp13BuilderMtpKvTransaction *mtp_kv_transactions;
	uint32_t *mtp_shadow_free_indices;
	uint32_t mtp_shadow_slot_capacity;
	uint32_t mtp_shadow_free_count;
	uint32_t cache_storage_token_capacity;
	uint32_t *host_backing_block_free_next;
	uint32_t *device_physical_block_indices;
	uint32_t *device_lane_physical_block_counts;
	void *selected_token_indices_by_layer;
	void *selected_block_indices_by_layer;
	void *selected_block_counts_by_layer;
	void *dsa_selection_epoch_by_layer;
	uint32_t dsa_cache_first_layer_index;
	uint32_t dsa_cache_layer_count;
	void *restricted_token_ids;
	void *embedding_weight;
	void *mtp_embedding_weight;
	void *mtp_enorm_weight;
	void *mtp_hnorm_weight;
	void *mtp_eh_proj_weight;
	void *mtp_shared_head_norm_weight;
	void *mtp_eh_input;
	float *full_vocab_logits;
	uint8_t *mtp_draft_lm_head_weight_fp8;
	float *mtp_draft_lm_head_weight_scale_inv;
	void *mtp_draft_head_workspace;
	uint64_t mtp_draft_head_workspace_bytes;
	void *mtp_previous_target_hidden;
	void *mtp_previous_target_hidden_store;
	uint32_t *mtp_prefill_block_table;
	uint64_t *mtp_gpu_profile_cycles;
	uint32_t *mtp_base_positions;
	uint32_t *device_mtp_request_slot_indices;
	uint32_t *device_mtp_tree_shadow_slot_mapping;
	float *mtp_norm_inv;
	void *cos_table;
	void *device_probe_hash_slots;
	void *sin_table;
	float *dsa_score_tiles;
	uint32_t *dsa_prefill_selected;
	uint32_t *dsa_prefill_row_context_lengths;
	uint32_t *dsa_prefill_row_sequences;
	uint32_t *dsa_prefill_row_positions;
	void *dsa_prefill_key_scratch;
	void *dsa_prefill_query_a;
	void *dsa_prefill_query_index_heads;
	void *dsa_prefill_index_weights;
	void *dsa_prefill_normalized_hidden;
	void *dsa_prefill_low_scratch;
	void *input_sideband;
	void *output_sideband;
	void *final_epilogue_workspace;
	void *fp8_scaled_gemm_workspace;
	void *shared_moe_workspace;
	uint64_t shared_moe_workspace_bytes;
	void *shared_b12x_state_cell;
	uint64_t shared_b12x_kernel_manifest_hash_low64;
	uint32_t shared_b12x_maximum_token_count;
	uint32_t *host_prefill_lane_offsets;
	uint32_t *host_prefill_lane_counts;
	uint32_t *host_decode_positions;
	uint32_t *host_decode_token_ids;
	uint32_t *host_decode_result_token_ids;
	uint32_t *host_mtp_draft_budgets;
	uint32_t *host_mtp_committed_token_ids;
	uint32_t *host_mtp_request_slot_indices;
	uint32_t *host_mtp_tree_shadow_slot_mapping;
	uint64_t *host_mtp_previous_sequence_ids;
	uint64_t *host_mtp_previous_positions;
	uint8_t *host_mtp_previous_valid;
	void *device_decode_positions;
	void *device_decode_token_ids;
	void *device_mtp_draft_token_budgets;
	cudaStream_t stream;
	cudaStream_t query_stream;
	cudaStream_t kv_stream;
	cudaStream_t kv_io_stream;
	cudaEvent_t branch_ready_event;
	cudaEvent_t query_event;
	cudaEvent_t kv_event;
	cudaEvent_t kv_io_event;
	cublasHandle_t full_vocab_cublas_handle;
	int32_t kv_nvme_fd;
	void *kv_nvme_staging;
	uint64_t kv_nvme_staging_bytes;
	uint64_t kv_nvme_payload_bytes;
	uint64_t kv_nvme_record_bytes;
	uint64_t kv_nvme_store_count;
	uint64_t kv_nvme_load_count;
	uint64_t kv_nvme_synchronous_wait_count;
	uint64_t kv_nvme_batch_flush_count;
	uint64_t kv_nvme_maximum_batch_operation_count;
	uint64_t kv_nvme_dsa_index_layer_mask;
	uint32_t kv_nvme_pending_store_count;
	uint32_t kv_nvme_pending_load_count;
	uint32_t kv_nvme_batch_active;
	uint32_t kv_nvme_reserved0;
	SparkGlm52Pp13BuilderNvmePendingOperation kv_nvme_pending_stores[
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MAX_NVME_BATCH_BLOCK_COUNT];
	SparkGlm52Pp13BuilderNvmePendingOperation kv_nvme_pending_loads[
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MAX_NVME_BATCH_BLOCK_COUNT];
	SparkGlm52KvJitStageBudget kv_jit_budget;
	uint64_t mtp_previous_request_id;
	uint64_t mtp_previous_sequence_id;
	uint64_t mtp_previous_position;
	uint32_t mtp_previous_valid;
	uint32_t mtp_gpu_profile_enabled;
	uint32_t mtp_cycle_profile_enabled;
	uint32_t full_vocab_head_row_capacity;
	uint32_t mtp_draft_head_ready;
	uint32_t mtp_ready;
	const SparkModelDriverInterface *driver_interface;
	void *driver_instance;
	const SparkModelDriverProgramDescriptor *program;
	SparkHiddenTransportSession *output_transport_session;
	uint32_t built;
	uint32_t runner_ready;
} SparkGlm52Pp13BuilderState;

static uint32_t SparkGlm52Pp13BuilderDsparkEnabled(
	const SparkGlm52Pp13BuilderState *state)
{
	return state != 0 &&
		(state->configuration.flags &
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_DSPARK) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderMtpEnabled(
	const SparkGlm52Pp13BuilderState *state)
{
	return state != 0 &&
		(state->configuration.flags &
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_MTP) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderUsesNvfp4(
	const SparkGlm52Pp13BuilderState *state)
{
	return state != 0 && state->rank_plan.quantization_mode ==
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT;
}

static uint32_t SparkGlm52Pp13BuilderUsesW8lut(
	const SparkGlm52Pp13BuilderState *state)
{
	return state != 0 && state->rank_plan.quantization_mode ==
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT;
}

static uint32_t SparkGlm52Pp13BuilderUsesBf16Trunk(
	const SparkGlm52Pp13BuilderState *state)
{
	return SparkGlm52Pp13BuilderUsesNvfp4(state) ||
		SparkGlm52Pp13BuilderUsesW8lut(state);
}

static uint32_t SparkGlm52Pp13BuilderLayerActiveRowCapacity(
	const SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13BuilderLayer *layer)
{
	if (state == 0 || layer == 0)
		return 0u;
	if (layer == &state->mtp_layer)
		return state->rank_plan.logical_lane_capacity;
	if (SparkGlm52Pp13BuilderUsesNvfp4(state) != 0u &&
		SparkGlm52Pp13BuilderMtpEnabled(state) == 0u &&
		SparkGlm52Pp13BuilderDsparkEnabled(state) == 0u)
		return state->rank_plan.logical_lane_capacity;
	return state->rank_plan.execution_row_capacity;
}

static uint32_t SparkGlm52Pp13BuilderIsFinalRank(
	const SparkGlm52Pp13BuilderState *state)
{
	return state != 0 &&
		(state->rank_plan.flags &
			SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderWorkCapturesDspark(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return work_packet != 0 &&
		(work_packet->flags &
			(SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE |
			 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY)) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderWorkIsDsparkVerify(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return work_packet != 0 &&
		(work_packet->flags &
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderWorkIsMtpVerify(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return work_packet != 0 &&
		(work_packet->flags &
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return SparkGlm52Pp13BuilderWorkIsMtpVerify(work_packet) != 0u &&
		(work_packet->flags &
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_TREE_VERIFY) != 0u;
}

static uint32_t SparkGlm52Pp13BuilderWorkIsSpeculativeVerify(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return SparkGlm52Pp13BuilderWorkIsDsparkVerify(work_packet) |
		SparkGlm52Pp13BuilderWorkIsMtpVerify(work_packet);
}

static uint32_t SparkGlm52Pp13BuilderWorkIsPlainDecodeBatch(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	if (work_packet == 0 || work_packet->active_sequence_count <= 1u)
		return 0u;
	return (work_packet->flags &
		(SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL |
		 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT |
		 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE |
		 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY |
		 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY)) == 0u;
}

static uint32_t SparkGlm52Pp13BuilderWorkNeedsCapturedCompletion(
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	return SparkGlm52Pp13BuilderWorkCapturesDspark(work_packet) |
		SparkGlm52Pp13BuilderWorkIsSpeculativeVerify(work_packet) |
		SparkGlm52Pp13BuilderWorkIsPlainDecodeBatch(work_packet);
}

static SparkStatus SparkGlm52Pp13BuilderInitializeDsparkTopology(
	SparkGlm52Pp13BuilderState *state)
{
	SparkGlm52StagePlan stage_plan;
	char error_buffer[256];
	SparkStatus status;

	status = SparkGlm52Pp13RuntimeBuildFixedStagePlan(
		&stage_plan,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52ProductionTopologyBuild(
		&stage_plan,
		state->rank_plan.logical_lane_capacity,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,
		&state->production_topology,
		error_buffer,
		sizeof(error_buffer));
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52DsparkBuildDefaultHiddenTapPlan(&state->dspark_tap_plan);
}

static void SparkGlm52Pp13BuilderFreeDsparkHostState(
	SparkGlm52Pp13BuilderState *state)
{
	if (state == 0)
		return;
	free(state->dspark_stage_batch);
	free(state->dspark_draft_batch);
	free(state->dspark_batch_results);
	free(state->dspark_ready_drafts);
	free(state->dspark_lane_by_request_slot);
	free(state->dspark_request_slot_by_lane);
	state->dspark_stage_batch = 0;
	state->dspark_draft_batch = 0;
	state->dspark_batch_results = 0;
	state->dspark_ready_drafts = 0;
	state->dspark_lane_by_request_slot = 0;
	state->dspark_request_slot_by_lane = 0;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeDsparkHostState(
	SparkGlm52Pp13BuilderState *state)
{
	uint32_t lane_index;
	uint32_t request_slot_index;
	uint32_t lane_count;
	uint32_t stage_capacity;

	if (state == 0 || state->configuration.maximum_resident_sequence_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	lane_count = state->configuration.dspark_maximum_lane_count;
	stage_capacity = state->dspark_backend.maximum_tap_row_count;
	if (lane_count == 0u || stage_capacity == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state->dspark_stage_batch = (SparkGlm52DsparkDraftBackendStage *)calloc(
		stage_capacity,sizeof(state->dspark_stage_batch[0u]));
	state->dspark_draft_batch = (SparkGlm52DsparkDraftRequest *)calloc(
		lane_count,sizeof(state->dspark_draft_batch[0u]));
	state->dspark_batch_results = (SparkGlm52DsparkDraftResult *)calloc(
		lane_count,sizeof(state->dspark_batch_results[0u]));
	state->dspark_ready_drafts = (SparkGlm52DsparkDraftResult *)calloc(
		lane_count,sizeof(state->dspark_ready_drafts[0u]));
	state->dspark_lane_by_request_slot = (uint32_t *)malloc(
		(size_t)state->configuration.maximum_resident_sequence_count *
			sizeof(state->dspark_lane_by_request_slot[0u]));
	state->dspark_request_slot_by_lane = (uint32_t *)malloc(
		(size_t)lane_count *
			sizeof(state->dspark_request_slot_by_lane[0u]));
	if (state->dspark_stage_batch == 0 || state->dspark_draft_batch == 0 ||
		state->dspark_batch_results == 0 || state->dspark_ready_drafts == 0 ||
		state->dspark_lane_by_request_slot == 0 ||
		state->dspark_request_slot_by_lane == 0)
	{
		SparkGlm52Pp13BuilderFreeDsparkHostState(state);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	for (request_slot_index = 0u;
		 request_slot_index <
			state->configuration.maximum_resident_sequence_count;
		 ++request_slot_index)
		state->dspark_lane_by_request_slot[request_slot_index] =
			SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	for (lane_index = 0u; lane_index < lane_count; ++lane_index)
		state->dspark_request_slot_by_lane[lane_index] =
			SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeDsparkBackend(
	SparkGlm52Pp13BuilderState *state)
{
	SparkGlm52DsparkDraftBackendConfiguration configuration;
	SparkStatus status;

	if (!SparkGlm52Pp13BuilderDsparkEnabled(state) ||
		!SparkGlm52Pp13BuilderIsFinalRank(state))
		return SPARK_STATUS_OK;
	if (state->configuration.dspark_manifest_path == 0 ||
		state->configuration.dspark_manifest_path[0] == '\0' ||
		state->configuration.dspark_config_path == 0 ||
		state->configuration.dspark_config_path[0] == '\0' ||
		state->configuration.dspark_safetensors_path == 0 ||
		state->configuration.dspark_safetensors_path[0] == '\0' ||
		state->configuration.dspark_maximum_lane_count == 0u ||
		state->configuration.dspark_maximum_lane_count >
			state->rank_plan.logical_lane_capacity ||
		state->configuration.dspark_maximum_context_token_count == 0u ||
		state->configuration.dspark_maximum_context_token_count >
			SPARK_GLM52_KV_CONTEXT_TOKENS)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version =
		SPARK_GLM52_DSPARK_DRAFT_BACKEND_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_GLM52_DSPARK_DRAFT_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.maximum_lane_count =
		state->configuration.dspark_maximum_lane_count;
	configuration.maximum_context_token_count =
		state->configuration.dspark_maximum_context_token_count;
	configuration.manifest_path =
		state->configuration.dspark_manifest_path;
	configuration.config_path =
		state->configuration.dspark_config_path;
	configuration.safetensors_path =
		state->configuration.dspark_safetensors_path;
	configuration.cuda_stream = (void *)state->stream;
	status = SparkGlm52DsparkDraftBackendInitialize(
		&state->dspark_backend,
		&configuration);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52DsparkDraftBackendModelContract(
		&state->dspark_backend,
		&state->dspark_model_contract);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52DsparkDraftBackendTapOutputPointers(
			&state->dspark_backend,
			0u,
			state->dspark_tap_outputs_bf16,
			&state->dspark_tap_lane_stride_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeDsparkHostState(state);
	if (status != SPARK_STATUS_OK)
	{
		SparkGlm52Pp13BuilderFreeDsparkHostState(state);
		SparkGlm52DsparkDraftBackendTeardown(&state->dspark_backend);
		return status;
	}
	state->dspark_backend_ready = 1u;
	return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52Pp13BuilderDsaSourceLayer(uint32_t layer_index)
{
	return SparkGlm52KvCacheDsaSourceLayer(layer_index);
}

static uint32_t SparkGlm52Pp13BuilderDsaGroupEnd(uint32_t source_layer_index)
{
	if (source_layer_index + 1u <
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
		return source_layer_index + 1u;
	if (source_layer_index +
		SPARK_GLM52_MODEL_DSA_INDEX_SHARE_GROUP_LAYER_COUNT >
		SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT)
		return SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT;
	return source_layer_index +
		SPARK_GLM52_MODEL_DSA_INDEX_SHARE_GROUP_LAYER_COUNT;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeLocalDsaCacheRange(
	SparkGlm52Pp13BuilderState *state)
{
	uint32_t cache_first;
	uint32_t layer_index;
	uint32_t layer_offset;
	uint32_t source_layer_index;
	uint32_t stage_end;

	if (state == 0 || state->rank_plan.layer_count == 0u ||
		state->rank_plan.first_layer_index >=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT ||
		state->rank_plan.first_layer_index + state->rank_plan.layer_count >
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	cache_first = state->rank_plan.first_layer_index;
	stage_end = cache_first + state->rank_plan.layer_count;
	for (layer_offset = 0u;
		 layer_offset < state->rank_plan.layer_count;
		 ++layer_offset)
	{
		layer_index = state->rank_plan.first_layer_index + layer_offset;
		source_layer_index = SparkGlm52Pp13BuilderDsaSourceLayer(layer_index);
		if (source_layer_index == UINT32_MAX)
			return SPARK_STATUS_INVALID_ARGUMENT;
		if (source_layer_index < cache_first)
			cache_first = source_layer_index;
	}
	state->dsa_cache_first_layer_index = cache_first;
	state->dsa_cache_layer_count = stage_end - cache_first;
	return state->dsa_cache_layer_count == 0u
		? SPARK_STATUS_INVALID_ARGUMENT
		: SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderCudaStatus(cudaError_t status)
{
	if (status == cudaSuccess)
		return SPARK_STATUS_OK;
	fprintf(stderr,"pp13_builder_cuda_error code=%d name=%s\n",(int32_t)status,cudaGetErrorString(status));
	return SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkGlm52Pp13BuilderCublasStatus(cublasStatus_t status)
{
	if (status == CUBLAS_STATUS_SUCCESS)
		return SPARK_STATUS_OK;
	fprintf(stderr,"pp13_builder_cublas_error code=%d\n",(int32_t)status);
	return SPARK_STATUS_IO_ERROR;
}

__global__ static void SparkGlm52Pp13BuilderBuildDecodeMetadataKernel(
	const uint32_t *__restrict__ decode_positions,
	const uint32_t *__restrict__ block_table,
	const uint32_t *__restrict__ lane_block_counts,
	uint32_t lane_stride,
	uint32_t block_token_count,
	uint32_t lane_count,
	uint32_t *__restrict__ positions,
	uint32_t *__restrict__ slot_mapping,
	uint32_t *__restrict__ context_lengths,
	uint32_t *__restrict__ first_block_token_offsets)
{
	uint32_t lane_index;
	uint32_t position;
	uint32_t block_index;
	uint32_t in_block_index;
	uint32_t physical_block_index;
	lane_index = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
	if (lane_index >= lane_count)
		return;
	position = decode_positions[lane_index];
	block_index = position / block_token_count;
	in_block_index = position - (block_index * block_token_count);
	positions[lane_index] = position;
	context_lengths[lane_index] = position + 1u;
	first_block_token_offsets[lane_index] = 0u;
	if (block_index >= lane_block_counts[lane_index])
	{
		slot_mapping[lane_index] = SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
		return;
	}
	physical_block_index = block_table[(lane_index * lane_stride) + block_index];
	slot_mapping[lane_index] =
		(physical_block_index * block_token_count) + in_block_index;
}

__global__ static void SparkGlm52Pp13BuilderGatherDecodeEmbeddingKernel(
	const uint32_t *__restrict__ token_ids,
	const uint32_t *__restrict__ embedding_bf16_words,
	uint32_t *__restrict__ output_bf16_words,
	uint32_t lane_count,
	uint32_t hidden_words)
{
	uint64_t word_index;
	uint32_t lane_index;
	uint32_t hidden_word_index;
	uint32_t token_id;
	word_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (word_index >= (uint64_t)lane_count * hidden_words)
		return;
	lane_index = (uint32_t)(word_index / hidden_words);
	hidden_word_index = (uint32_t)(word_index - ((uint64_t)lane_index * hidden_words));
	token_id = token_ids[lane_index];
	output_bf16_words[word_index] =
		embedding_bf16_words[((uint64_t)token_id * hidden_words) + hidden_word_index];
}

static __device__ __forceinline__ float SparkGlm52Pp13BuilderBf16ToFloat(
	uint16_t value)
{
	return __bfloat162float(__ushort_as_bfloat16(value));
}

static __device__ __forceinline__ uint16_t SparkGlm52Pp13BuilderFloatToBf16(
	float value)
{
	return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__global__ static void SparkGlm52Pp13BuilderTargetFinalNormKernel(
	const uint16_t *__restrict__ input_bf16,
	const uint16_t *__restrict__ norm_weight_bf16,
	uint16_t *__restrict__ output_bf16,
	uint32_t active_sequence_count,
	float epsilon)
{
	__shared__ float sum[SPARK_GLM52_PP13_BUILDER_THREADS];
	uint32_t lane_index;
	uint32_t hidden_index;
	uint32_t stride;
	float local_sum;
	float norm_inv;
	float value;
	lane_index = blockIdx.x;
	if (lane_index >= active_sequence_count)
		return;
	local_sum = 0.0f;
	for (hidden_index = threadIdx.x;
		 hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
		 hidden_index += blockDim.x)
	{
		value = SparkGlm52Pp13BuilderBf16ToFloat(input_bf16[
			((uint64_t)lane_index *
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index]);
		local_sum += value * value;
	}
	sum[threadIdx.x] = local_sum;
	__syncthreads();
	for (stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u)
	{
		if (threadIdx.x < stride)
			sum[threadIdx.x] += sum[threadIdx.x + stride];
		__syncthreads();
	}
	norm_inv = rsqrtf(
		(sum[0] / SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + epsilon);
	for (hidden_index = threadIdx.x;
		 hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
		 hidden_index += blockDim.x)
	{
		value = SparkGlm52Pp13BuilderBf16ToFloat(input_bf16[
			((uint64_t)lane_index *
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index]);
		value *= SparkGlm52Pp13BuilderBf16ToFloat(norm_weight_bf16[hidden_index]) *
			norm_inv;
		output_bf16[((uint64_t)lane_index *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index] =
			SparkGlm52Pp13BuilderFloatToBf16(value);
	}
}

__device__ static float SparkGlm52Pp13BuilderFullVocabLogitToFloat(
	const float *logits,
	uint64_t index)
{
	return logits[index];
}

__device__ static float SparkGlm52Pp13BuilderFullVocabLogitToFloat(
	const uint16_t *logits,
	uint64_t index)
{
	return SparkGlm52Pp13BuilderBf16ToFloat(logits[index]);
}

__device__ static void SparkGlm52Pp13BuilderMtpTop2Insert(
	float score,
	uint32_t token,
	float *first_score,
	uint32_t *first_token,
	float *second_score,
	uint32_t *second_token)
{
	if (token == *first_token || token == *second_token)
		return;
	if (*first_token == UINT32_MAX || score > *first_score ||
		(score == *first_score && token < *first_token))
	{
		*second_score = *first_score;
		*second_token = *first_token;
		*first_score = score;
		*first_token = token;
		return;
	}
	if (*second_token == UINT32_MAX || score > *second_score ||
		(score == *second_score && token < *second_token))
	{
		*second_score = score;
		*second_token = token;
	}
}

template <typename LogitType>
__global__ static void SparkGlm52Pp13BuilderMtpFullVocabTop2Kernel(
	const LogitType *__restrict__ logits,
	const uint32_t *__restrict__ token_ids,
	uint32_t *__restrict__ selected_token_ids,
	float *__restrict__ selected_token_scores,
	uint32_t *__restrict__ alternate_token_ids,
	uint32_t row_count)
{
	__shared__ float shared_first_scores[SPARK_GLM52_PP13_BUILDER_THREADS];
	__shared__ float shared_second_scores[SPARK_GLM52_PP13_BUILDER_THREADS];
	__shared__ uint32_t shared_first_tokens[SPARK_GLM52_PP13_BUILDER_THREADS];
	__shared__ uint32_t shared_second_tokens[SPARK_GLM52_PP13_BUILDER_THREADS];
	uint32_t first_token,row_index,second_token,stride,token_index;
	float first_score,score,second_score;
	row_index = blockIdx.x;
	if (row_index >= row_count)
		return;
	first_score = -3.4028234663852886e+38f;
	second_score = -3.4028234663852886e+38f;
	first_token = UINT32_MAX;
	second_token = UINT32_MAX;
	for (token_index = threadIdx.x;
		 token_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT;
		 token_index += blockDim.x)
	{
		score = SparkGlm52Pp13BuilderFullVocabLogitToFloat(
			logits,
			((uint64_t)row_index *
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT) + token_index);
		SparkGlm52Pp13BuilderMtpTop2Insert(
			score,token_ids[token_index],&first_score,&first_token,
			&second_score,&second_token);
	}
	shared_first_scores[threadIdx.x] = first_score;
	shared_second_scores[threadIdx.x] = second_score;
	shared_first_tokens[threadIdx.x] = first_token;
	shared_second_tokens[threadIdx.x] = second_token;
	__syncthreads();
	for (stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u)
	{
		if (threadIdx.x < stride)
		{
			first_score = shared_first_scores[threadIdx.x];
			second_score = shared_second_scores[threadIdx.x];
			first_token = shared_first_tokens[threadIdx.x];
			second_token = shared_second_tokens[threadIdx.x];
			SparkGlm52Pp13BuilderMtpTop2Insert(
				shared_first_scores[threadIdx.x + stride],
				shared_first_tokens[threadIdx.x + stride],
				&first_score,&first_token,&second_score,&second_token);
			SparkGlm52Pp13BuilderMtpTop2Insert(
				shared_second_scores[threadIdx.x + stride],
				shared_second_tokens[threadIdx.x + stride],
				&first_score,&first_token,&second_score,&second_token);
			shared_first_scores[threadIdx.x] = first_score;
			shared_second_scores[threadIdx.x] = second_score;
			shared_first_tokens[threadIdx.x] = first_token;
			shared_second_tokens[threadIdx.x] = second_token;
		}
		__syncthreads();
	}
	if (threadIdx.x == 0u)
	{
		selected_token_ids[row_index] = shared_first_tokens[0u];
		selected_token_scores[row_index] = shared_first_scores[0u];
		if (alternate_token_ids != 0)
			alternate_token_ids[row_index] = shared_second_tokens[0u];
	}
}

__device__ static uint64_t SparkGlm52Pp13BuilderMtpDraftHeadWeightIndex(
	uint32_t tile_index,
	uint32_t input_dimension)
{
	uint32_t output_index,input_index;
	output_index = blockIdx.x *
		SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK +
		tile_index / SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK;
	input_index = blockIdx.y *
		SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK +
		tile_index % SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK;
	return ((uint64_t)output_index * input_dimension) + input_index;
}

__global__ static void SparkGlm52Pp13BuilderQuantizeMtpDraftHeadKernel(
	const uint16_t *__restrict__ weight_bf16,
	uint8_t *__restrict__ weight_fp8,
	float *__restrict__ weight_scale_inv,
	uint32_t input_dimension)
{
	__shared__ float shared_absmax[SPARK_GLM52_PP13_BUILDER_THREADS];
	uint32_t tile_index,stride,input_block_count;
	uint64_t weight_index;
	float value,scale;
	shared_absmax[threadIdx.x] = 0.0f;
	for (tile_index = threadIdx.x;
		 tile_index < SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK *
			SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK;
		 tile_index += blockDim.x)
	{
		weight_index = SparkGlm52Pp13BuilderMtpDraftHeadWeightIndex(
			tile_index,input_dimension);
		value = fabsf(SparkGlm52Pp13BuilderBf16ToFloat(weight_bf16[weight_index]));
		shared_absmax[threadIdx.x] = fmaxf(shared_absmax[threadIdx.x],value);
	}
	__syncthreads();
	for (stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u)
	{
		if (threadIdx.x < stride)
			shared_absmax[threadIdx.x] = fmaxf(
				shared_absmax[threadIdx.x],shared_absmax[threadIdx.x + stride]);
		__syncthreads();
	}
	scale = fmaxf(shared_absmax[0u] / SPARK_GLM52_PP13_BUILDER_FP8_E4M3_MAX,1.0e-8f);
	input_block_count = input_dimension /
		SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK;
	if (threadIdx.x == 0u)
		weight_scale_inv[((uint64_t)blockIdx.x * input_block_count) + blockIdx.y] =
			scale;
	for (tile_index = threadIdx.x;
		 tile_index < SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK *
			SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK;
		 tile_index += blockDim.x)
	{
		weight_index = SparkGlm52Pp13BuilderMtpDraftHeadWeightIndex(
			tile_index,input_dimension);
		value = SparkGlm52Pp13BuilderBf16ToFloat(weight_bf16[weight_index]) / scale;
		weight_fp8[weight_index] = __nv_cvt_float_to_fp8(
			value,__NV_SATFINITE,__NV_E4M3);
	}
}

__global__ static void SparkGlm52Pp13BuilderScatterMtpPreviousHiddenKernel(
	const uint16_t *__restrict__ batch_hidden_bf16,
	const uint32_t *__restrict__ request_slot_indices,
	uint16_t *__restrict__ persistent_hidden_bf16,
	uint32_t active_sequence_count)
{
	uint64_t element_index;
	uint64_t element_count;
	uint32_t lane_index;
	uint32_t hidden_index;
	uint32_t request_slot_index;
	element_index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	element_count = (uint64_t)active_sequence_count *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	if (element_index >= element_count)
		return;
	lane_index = (uint32_t)(element_index /
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
	hidden_index = (uint32_t)(element_index -
		((uint64_t)lane_index *
		 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION));
	request_slot_index = request_slot_indices[lane_index];
	persistent_hidden_bf16[
		((uint64_t)request_slot_index *
		 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index] =
		batch_hidden_bf16[element_index];
}

__global__ static void SparkGlm52Pp13BuilderSelectedTargetFinalNormKernel(
	const uint16_t *__restrict__ input_hidden_bf16,
	const uint16_t *__restrict__ norm_weight_bf16,
	const uint32_t *__restrict__ selected_row_indices,
	uint16_t *__restrict__ output_hidden_bf16,
	uint32_t logical_lane_count,
	float epsilon)
{
	__shared__ float shared_sum[SPARK_GLM52_PP13_BUILDER_THREADS];
	uint32_t lane_index;
	uint32_t hidden_index;
	uint32_t stride;
	uint32_t selected_row_index;
	float inv;
	float local_sum;
	float value;
	lane_index = blockIdx.x;
	if (lane_index >= logical_lane_count)
		return;
	selected_row_index = selected_row_indices[lane_index];
	local_sum = 0.0f;
	for (hidden_index = threadIdx.x;
		 hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
		 hidden_index += blockDim.x)
	{
		value = SparkGlm52Pp13BuilderBf16ToFloat(input_hidden_bf16[
			((uint64_t)selected_row_index *
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index]);
		local_sum += value * value;
	}
	shared_sum[threadIdx.x] = local_sum;
	__syncthreads();
	for (stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u)
	{
		if (threadIdx.x < stride)
			shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
		__syncthreads();
	}
	inv = rsqrtf(
		(shared_sum[0] /
		 (float)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + epsilon);
	for (hidden_index = threadIdx.x;
		 hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
		 hidden_index += blockDim.x)
	{
		value = SparkGlm52Pp13BuilderBf16ToFloat(input_hidden_bf16[
			((uint64_t)selected_row_index *
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index]);
		value *= inv * SparkGlm52Pp13BuilderBf16ToFloat(
			norm_weight_bf16[hidden_index]);
		output_hidden_bf16[
			((uint64_t)lane_index *
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index] =
			SparkGlm52Pp13BuilderFloatToBf16(value);
	}
}

__global__ static void SparkGlm52Pp13BuilderPrepareMtpPrefillHiddenKernel(
	const uint16_t *__restrict__ input_hidden_bf16,
	const uint16_t *__restrict__ norm_weight_bf16,
	const uint16_t *__restrict__ persistent_hidden_bf16,
	const uint32_t *__restrict__ source_row_indices,
	const uint32_t *__restrict__ request_slot_indices,
	uint16_t *__restrict__ output_hidden_bf16,
	uint32_t row_count,
	float epsilon)
{
	__shared__ float shared_sum[SPARK_GLM52_PP13_BUILDER_THREADS];
	uint32_t hidden_index,request_slot_index,row_index,source_row_index,stride;
	float inv,local_sum,value;
	row_index = blockIdx.x;
	if (row_index >= row_count)
		return;
	source_row_index = source_row_indices[row_index];
	if (source_row_index == UINT32_MAX)
	{
		request_slot_index = request_slot_indices[row_index];
		for (hidden_index = threadIdx.x;
			 hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
			 hidden_index += blockDim.x)
			output_hidden_bf16[
				((uint64_t)row_index *
				 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
				hidden_index] = persistent_hidden_bf16[
					((uint64_t)request_slot_index *
					 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
					hidden_index];
		return;
	}
	local_sum = 0.0f;
	for (hidden_index = threadIdx.x;
		 hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
		 hidden_index += blockDim.x)
	{
		value = SparkGlm52Pp13BuilderBf16ToFloat(input_hidden_bf16[
			((uint64_t)source_row_index *
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index]);
		local_sum += value * value;
	}
	shared_sum[threadIdx.x] = local_sum;
	__syncthreads();
	for (stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u)
	{
		if (threadIdx.x < stride)
			shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
		__syncthreads();
	}
	inv = rsqrtf(
		(shared_sum[0] /
		 (float)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + epsilon);
	for (hidden_index = threadIdx.x;
		 hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
		 hidden_index += blockDim.x)
	{
		value = SparkGlm52Pp13BuilderBf16ToFloat(input_hidden_bf16[
			((uint64_t)source_row_index *
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index]);
		value *= inv * SparkGlm52Pp13BuilderBf16ToFloat(
			norm_weight_bf16[hidden_index]);
		output_hidden_bf16[
			((uint64_t)row_index *
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index] =
			SparkGlm52Pp13BuilderFloatToBf16(value);
	}
}

__global__ static void SparkGlm52Pp13BuilderExpandMtpPrefillBlockTableKernel(
	const uint32_t *__restrict__ source_block_table,
	const uint32_t *__restrict__ logical_lane_indices,
	uint32_t *__restrict__ destination_block_table,
	uint32_t block_table_stride,
	uint32_t active_block_count,
	uint32_t row_count)
{
	uint64_t element_count,element_index;
	uint32_t block_index,logical_lane_index,row_index;
	element_index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	element_count = (uint64_t)row_count * active_block_count;
	if (element_index >= element_count)
		return;
	row_index = (uint32_t)(element_index / active_block_count);
	block_index = (uint32_t)(element_index -
		((uint64_t)row_index * active_block_count));
	logical_lane_index = logical_lane_indices[row_index];
	destination_block_table[
		((uint64_t)row_index * block_table_stride) + block_index] =
		source_block_table[
			((uint64_t)logical_lane_index * block_table_stride) + block_index];
}

__global__ static void SparkGlm52Pp13BuilderMtpNormInvKernel(
	const uint32_t *__restrict__ token_ids,
	const uint32_t *__restrict__ positions,
	const uint16_t *__restrict__ embedding_bf16,
	const uint16_t *__restrict__ hidden_bf16,
	float *__restrict__ norm_inv,
	uint32_t active_sequence_count,
	float epsilon)
{
	__shared__ float embedding_sum[SPARK_GLM52_PP13_BUILDER_THREADS];
	__shared__ float hidden_sum[SPARK_GLM52_PP13_BUILDER_THREADS];
	uint32_t lane_index;
	uint32_t hidden_index;
	uint32_t stride;
	uint32_t token_id;
	float embedding_value;
	float hidden_value;
	float local_embedding_sum;
	float local_hidden_sum;
	lane_index = blockIdx.x;
	if (lane_index >= active_sequence_count)
		return;
	token_id = token_ids[lane_index];
	local_embedding_sum = 0.0f;
	local_hidden_sum = 0.0f;
	for (hidden_index = threadIdx.x;
		 hidden_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
		 hidden_index += blockDim.x)
	{
		embedding_value = positions[lane_index] == 0u ? 0.0f :
			SparkGlm52Pp13BuilderBf16ToFloat(
				embedding_bf16[((uint64_t)token_id *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index]);
		hidden_value = SparkGlm52Pp13BuilderBf16ToFloat(
			hidden_bf16[((uint64_t)lane_index *
				SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + hidden_index]);
		local_embedding_sum += embedding_value * embedding_value;
		local_hidden_sum += hidden_value * hidden_value;
	}
	embedding_sum[threadIdx.x] = local_embedding_sum;
	hidden_sum[threadIdx.x] = local_hidden_sum;
	__syncthreads();
	for (stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u)
	{
		if (threadIdx.x < stride)
		{
			embedding_sum[threadIdx.x] += embedding_sum[threadIdx.x + stride];
			hidden_sum[threadIdx.x] += hidden_sum[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0u)
	{
		norm_inv[(lane_index * 2u)] = rsqrtf(
			(embedding_sum[0] /
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + epsilon);
		norm_inv[(lane_index * 2u) + 1u] = rsqrtf(
			(hidden_sum[0] /
			 SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) + epsilon);
	}
}

__global__ static void SparkGlm52Pp13BuilderMtpFusionKernel(
	const uint32_t *__restrict__ token_ids,
	const uint32_t *__restrict__ positions,
	const uint16_t *__restrict__ embedding_bf16,
	const uint16_t *__restrict__ hidden_bf16,
	const uint16_t *__restrict__ embedding_norm_weight_bf16,
	const uint16_t *__restrict__ hidden_norm_weight_bf16,
	const float *__restrict__ norm_inv,
	uint16_t *__restrict__ output_bf16,
	uint32_t active_sequence_count)
{
	uint64_t output_index;
	uint32_t lane_index;
	uint32_t input_index;
	uint32_t hidden_index;
	uint32_t token_id;
	float value;
	output_index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	if (output_index >= (uint64_t)active_sequence_count *
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION)
		return;
	lane_index = (uint32_t)(output_index /
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION);
	input_index = (uint32_t)(output_index - ((uint64_t)lane_index *
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION));
	hidden_index = input_index % SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	if (input_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)
	{
		token_id = token_ids[lane_index];
		value = positions[lane_index] == 0u ? 0.0f :
			SparkGlm52Pp13BuilderBf16ToFloat(embedding_bf16[
				((uint64_t)token_id * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
				hidden_index]);
		value *= SparkGlm52Pp13BuilderBf16ToFloat(
			embedding_norm_weight_bf16[hidden_index]) * norm_inv[lane_index * 2u];
	}
	else
	{
		value = SparkGlm52Pp13BuilderBf16ToFloat(hidden_bf16[
			((uint64_t)lane_index * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
			hidden_index]);
		value *= SparkGlm52Pp13BuilderBf16ToFloat(
			hidden_norm_weight_bf16[hidden_index]) * norm_inv[(lane_index * 2u) + 1u];
	}
	output_bf16[output_index] = SparkGlm52Pp13BuilderFloatToBf16(value);
}

__global__ static void SparkGlm52Pp13BuilderMtpMetadataKernel(
	const uint32_t *__restrict__ base_positions,
	const uint32_t *__restrict__ physical_block_indices,
	uint32_t *__restrict__ positions,
	uint32_t *__restrict__ slot_mapping,
	uint32_t *__restrict__ context_lengths,
	uint32_t *__restrict__ first_block_token_offsets,
	uint32_t block_table_stride,
	uint32_t block_token_count,
	uint32_t draft_index,
	uint32_t active_sequence_count)
{
	uint32_t block_index;
	uint32_t block_token_index;
	uint32_t lane_index;
	uint32_t physical_block_index;
	uint32_t position;
	lane_index = (uint32_t)(blockIdx.x * blockDim.x + threadIdx.x);
	if (lane_index >= active_sequence_count)
		return;
	position = base_positions[lane_index] +
		SPARK_GLM52_MODEL_MTP_EXECUTION_POSITION_OFFSET + draft_index;
	block_index = position / block_token_count;
	block_token_index = position - (block_index * block_token_count);
	physical_block_index = physical_block_indices[
		((uint64_t)lane_index * block_table_stride) + block_index];
	positions[lane_index] = position;
	slot_mapping[lane_index] =
		(physical_block_index * block_token_count) + block_token_index;
	context_lengths[lane_index] = position + 1u;
	first_block_token_offsets[lane_index] = 0u;
}

__global__ static void SparkGlm52Pp13BuilderMtpStoreKernel(
	const uint16_t *__restrict__ hidden_bf16,
	const uint32_t *__restrict__ token_ids,
	uint16_t *__restrict__ draft_hidden_bf16,
	uint32_t *__restrict__ draft_token_ids,
	uint32_t draft_index,
	uint32_t active_sequence_count)
{
	uint64_t element_index;
	uint32_t lane_index;
	uint32_t hidden_index;
	element_index = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x;
	if (element_index >= (uint64_t)active_sequence_count *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION)
		return;
	lane_index = (uint32_t)(element_index /
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
	hidden_index = (uint32_t)(element_index - ((uint64_t)lane_index *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION));
	draft_hidden_bf16[(((uint64_t)lane_index *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) + draft_index) *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION + hidden_index] =
		hidden_bf16[element_index];
	if (hidden_index == 0u)
		draft_token_ids[(lane_index *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) + draft_index] =
			token_ids[lane_index];
}

__global__ static void SparkGlm52Pp13BuilderMtpGpuProfileKernel(
	uint64_t *__restrict__ cycles,
	uint32_t draft_index,
	uint32_t phase_index)
{
	uint64_t global_time;
	if (cycles == 0 || blockIdx.x != 0u || threadIdx.x != 0u ||
		draft_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT ||
		phase_index >= SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_PHASE_COUNT)
		return;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(global_time));
	cycles[((uint64_t)draft_index *
		SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_PHASE_COUNT) + phase_index] =
		global_time;
}

typedef struct SparkGlm52Pp13BuilderKvPayload
{
	uint8_t *base;
	uint32_t token_bytes;
} SparkGlm52Pp13BuilderKvPayload;

#define SPARK_GLM52_PP13_BUILDER_KV_PAYLOAD_COUNT 7u

typedef struct SparkGlm52Pp13BuilderKvPayloads
{
	SparkGlm52Pp13BuilderKvPayload
		payloads[SPARK_GLM52_PP13_BUILDER_KV_PAYLOAD_COUNT];
	uint32_t count;
} SparkGlm52Pp13BuilderKvPayloads;

__global__ static void SparkGlm52Pp13BuilderClearSpeculativeKvRowsKernel(
	const uint32_t *__restrict__ physical_slots,
	uint32_t physical_slot_count,
	uint32_t block_token_count,
	SparkGlm52Pp13BuilderKvPayloads payloads,
	uint8_t *__restrict__ dsa_summary_dirty_flags)
{
	uint32_t byte_index;
	uint32_t payload_index;
	uint32_t physical_slot;
	uint32_t slot_index;
	slot_index = blockIdx.x;
	if (slot_index >= physical_slot_count)
		return;
	physical_slot = physical_slots[slot_index];
	for (payload_index = 0u; payload_index < payloads.count; ++payload_index)
	{
		for (byte_index = threadIdx.x;
			 byte_index < payloads.payloads[payload_index].token_bytes;
			 byte_index += blockDim.x)
			payloads.payloads[payload_index].base[
				((uint64_t)physical_slot *
					payloads.payloads[payload_index].token_bytes) +
				byte_index] = 0u;
	}
	if (threadIdx.x == 0u && dsa_summary_dirty_flags != 0)
		dsa_summary_dirty_flags[physical_slot / block_token_count] = 1u;
}

__global__ static void SparkGlm52Pp13BuilderCopySpeculativeKvRowsKernel(
	const uint32_t *__restrict__ source_physical_slots,
	const uint32_t *__restrict__ destination_physical_slots,
	uint32_t physical_slot_count,
	uint32_t block_token_count,
	SparkGlm52Pp13BuilderKvPayloads payloads,
	uint8_t *__restrict__ dsa_summary_dirty_flags)
{
	uint32_t byte_index;
	uint32_t source_physical_slot;
	uint32_t destination_physical_slot;
	uint32_t payload_index;
	uint32_t slot_index;
	slot_index = blockIdx.x;
	if (slot_index >= physical_slot_count)
		return;
	source_physical_slot = source_physical_slots[slot_index];
	destination_physical_slot = destination_physical_slots[slot_index];
	for (payload_index = 0u; payload_index < payloads.count; ++payload_index)
	{
		for (byte_index = threadIdx.x;
			 byte_index < payloads.payloads[payload_index].token_bytes;
			 byte_index += blockDim.x)
			payloads.payloads[payload_index].base[
				((uint64_t)destination_physical_slot *
					payloads.payloads[payload_index].token_bytes) +
				byte_index] =
				payloads.payloads[payload_index].base[
					((uint64_t)source_physical_slot *
						payloads.payloads[payload_index].token_bytes) +
					byte_index];
	}
	if (threadIdx.x == 0u && dsa_summary_dirty_flags != 0)
		dsa_summary_dirty_flags[
			destination_physical_slot / block_token_count] = 1u;
}

static SparkStatus SparkGlm52Pp13BuilderReportStatus(
	const char *step,
	uint32_t layer_index,
	SparkStatus status)
{
	if (status != SPARK_STATUS_OK)
		fprintf(stderr,"pp13_builder_error layer=%u step=%s status=%d\n",
			layer_index,step == 0 ? "unknown" : step,(int)status);
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderMarkMtpGpuProfile(
	SparkGlm52Pp13BuilderState *state,
	uint32_t draft_index,
	SparkGlm52Pp13BuilderMtpGpuProfilePhase phase,
	cudaStream_t stream)
{
	if (state == 0 || stream == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->mtp_gpu_profile_enabled == 0u)
		return SPARK_STATUS_OK;
	if (state->mtp_gpu_profile_cycles == 0 ||
		draft_index >= SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT ||
		(uint32_t)phase >= SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_PHASE_COUNT)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	SparkGlm52Pp13BuilderMtpGpuProfileKernel<<<1u,1u,0u,stream>>>(
		state->mtp_gpu_profile_cycles,draft_index,(uint32_t)phase);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52Pp13BuilderReportMtpGpuProfile(
	SparkGlm52Pp13BuilderState *state,
	const char *source,
	uint32_t row_count,
	uint32_t draft_token_count)
{
	uint64_t cycles[
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT *
		SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_PHASE_COUNT];
	uint64_t *draft_cycles;
	uint32_t draft_index;
	uint32_t phase_index;
	SparkStatus status;
	if (state == 0 || source == 0 || row_count == 0u ||
		draft_token_count > SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->mtp_gpu_profile_enabled == 0u || draft_token_count == 0u)
		return SPARK_STATUS_OK;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
		cycles,state->mtp_gpu_profile_cycles,sizeof(cycles),cudaMemcpyDeviceToHost));
	if (status != SPARK_STATUS_OK)
		return status;
	for (draft_index = 0u; draft_index < draft_token_count; ++draft_index)
	{
		draft_cycles = cycles + ((uint64_t)draft_index *
			SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_PHASE_COUNT);
		for (phase_index = 1u;
			 phase_index < SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_PHASE_COUNT;
			 ++phase_index)
		{
			if (draft_cycles[phase_index - 1u] == 0u ||
				draft_cycles[phase_index] < draft_cycles[phase_index - 1u])
				return SPARK_STATUS_INTERNAL_ERROR;
		}
		fprintf(stderr,
			"mtp_gpu_profile source=%s rows=%u draft=%u metadata_ns=%llu "
			"fusion_ns=%llu eh_projection_ns=%llu required_layer_ns=%llu "
			"vocab_head_ns=%llu store_ns=%llu total_ns=%llu\n",
			source,row_count,draft_index,
			(unsigned long long)(draft_cycles[1u] - draft_cycles[0u]),
			(unsigned long long)(draft_cycles[2u] - draft_cycles[1u]),
			(unsigned long long)(draft_cycles[3u] - draft_cycles[2u]),
			(unsigned long long)(draft_cycles[4u] - draft_cycles[3u]),
			(unsigned long long)(draft_cycles[5u] - draft_cycles[4u]),
			(unsigned long long)(draft_cycles[6u] - draft_cycles[5u]),
			(unsigned long long)(draft_cycles[6u] - draft_cycles[0u]));
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderRememberAllocationWithKind(
	SparkGlm52Pp13BuilderState *state,
	void *pointer,
	uint32_t is_host_mapped)
{
	if (state == 0 || pointer == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->allocation_count >= SPARK_GLM52_PP13_BUILDER_MAX_ALLOCATIONS)
	{
		fprintf(stderr,
			"pp13_builder_allocation_registry_full count=%u capacity=%u\n",
			state->allocation_count,
			SPARK_GLM52_PP13_BUILDER_MAX_ALLOCATIONS);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	state->allocations[state->allocation_count] = pointer;
	state->allocation_is_host_mapped[state->allocation_count] =
		is_host_mapped != 0u ? 1u : 0u;
	state->allocation_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderRememberAllocation(
	SparkGlm52Pp13BuilderState *state,
	void *pointer)
{
	return SparkGlm52Pp13BuilderRememberAllocationWithKind(state,pointer,0u);
}

static SparkStatus SparkGlm52Pp13BuilderCudaAlloc(SparkGlm52Pp13BuilderState *state, void **pointer_out, uint64_t bytes)
{
	void *pointer;
	size_t free_bytes, total_bytes;
	SparkStatus status;
	if (state == 0 || pointer_out == 0 || bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	pointer = 0;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMalloc(&pointer,(size_t)bytes));
	if (status != SPARK_STATUS_OK)
	{
		free_bytes = 0u;
		total_bytes = 0u;
		if (cudaMemGetInfo(&free_bytes,&total_bytes) != cudaSuccess)
			cudaGetLastError();
		fprintf(stderr,"pp13_builder_cuda_alloc_failed requested_bytes=%llu free_bytes=%llu total_bytes=%llu\n",(unsigned long long)bytes,(unsigned long long)free_bytes,(unsigned long long)total_bytes);
		return status;
	}
	status = SparkGlm52Pp13BuilderRememberAllocation(state,pointer);
	if (status != SPARK_STATUS_OK)
	{
		cudaFree(pointer);
		return status;
	}
	state->cuda_builder_allocation_bytes += bytes;
	if (bytes > state->cuda_largest_allocation_bytes)
		state->cuda_largest_allocation_bytes = bytes;
	*pointer_out = pointer;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderCudaHostMappedAlloc(
	SparkGlm52Pp13BuilderState *state,
	void **device_pointer_out,
	uint64_t bytes)
{
	void *host_pointer;
	void *device_pointer;
	SparkStatus status;
	if (state == 0 || device_pointer_out == 0 || bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	host_pointer = 0;
	device_pointer = 0;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaHostAlloc(
		&host_pointer,
		(size_t)bytes,
		cudaHostAllocMapped | cudaHostAllocPortable));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaHostGetDevicePointer(
		&device_pointer,
		host_pointer,
		0));
	if (status != SPARK_STATUS_OK)
	{
		cudaFreeHost(host_pointer);
		return status;
	}
	status = SparkGlm52Pp13BuilderRememberAllocationWithKind(
		state,
		host_pointer,
		1u);
	if (status != SPARK_STATUS_OK)
	{
		cudaFreeHost(host_pointer);
		return status;
	}
	state->host_mapped_allocation_bytes += bytes;
	*device_pointer_out = device_pointer;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderCudaHostPinnedAlloc(
	SparkGlm52Pp13BuilderState *state,
	void **host_pointer_out,
	uint64_t bytes)
{
	void *host_pointer;
	SparkStatus status;
	if (state == 0 || host_pointer_out == 0 || bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	host_pointer = 0;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaHostAlloc(
		&host_pointer,(size_t)bytes,cudaHostAllocPortable));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderRememberAllocationWithKind(
		state,host_pointer,1u);
	if (status != SPARK_STATUS_OK)
	{
		cudaFreeHost(host_pointer);
		return status;
	}
	state->host_mapped_allocation_bytes += bytes;
	*host_pointer_out = host_pointer;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderCudaZero(
	void *pointer,
	uint64_t bytes)
{
	if (pointer == 0 || bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkGlm52Pp13BuilderCudaStatus(cudaMemset(pointer,0,(size_t)bytes));
}

static uint64_t SparkGlm52Pp13BuilderAlignUpU64(
	uint64_t value,
	uint64_t alignment)
{
	if (alignment == 0u || value > UINT64_MAX - (alignment - 1u))
		return 0u;
	return ((value + alignment - 1u) / alignment) * alignment;
}

static SparkStatus SparkGlm52Pp13BuilderNvmeWriteExact(
	int32_t file_descriptor,
	uint64_t offset_bytes,
	const void *source,
	uint64_t byte_count)
{
	ssize_t written;
	if (file_descriptor < 0 || source == 0 || byte_count == 0u ||
		byte_count > (uint64_t)SSIZE_MAX || offset_bytes > (uint64_t)INT64_MAX)
		return SPARK_STATUS_INVALID_ARGUMENT;
	do
	{
		written = pwrite(file_descriptor,source,(size_t)byte_count,
			(off_t)offset_bytes);
	} while (written < 0 && errno == EINTR);
	return written == (ssize_t)byte_count
		? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkGlm52Pp13BuilderNvmeReadExact(
	int32_t file_descriptor,
	uint64_t offset_bytes,
	void *destination,
	uint64_t byte_count)
{
	ssize_t got;
	if (file_descriptor < 0 || destination == 0 || byte_count == 0u ||
		byte_count > (uint64_t)SSIZE_MAX || offset_bytes > (uint64_t)INT64_MAX)
		return SPARK_STATUS_INVALID_ARGUMENT;
	do
	{
		got = pread(file_descriptor,destination,(size_t)byte_count,
			(off_t)offset_bytes);
	} while (got < 0 && errno == EINTR);
	return got == (ssize_t)byte_count
		? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static uint32_t SparkGlm52Pp13BuilderAttentionCacheLayout(
	const SparkGlm52Pp13BuilderState *state)
{
	return SparkGlm52Pp13BuilderUsesBf16Trunk(state) != 0u
		? SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE
		: SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE_FP8_E4M3;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeKvNvme(
	SparkGlm52Pp13BuilderState *state)
{
	SparkGlm52KvJitStageBudgetRequest budget_request;
	uint64_t attention_block_bytes;
	uint64_t index_block_bytes;
	uint64_t summary_block_bytes;
	uint64_t payload_bytes;
	uint64_t file_bytes;
	uint64_t staging_bytes;
	struct stat nvme_stat;
	uint32_t layer_index;
	uint32_t layer_offset;
	int32_t open_flags;
	void *staging;
	int allocation_status;
	SparkStatus status;

	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((state->configuration.flags &
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_NVME_KV) == 0u)
		return SPARK_STATUS_OK;
	if (state->configuration.kv_nvme_path == 0 ||
		state->configuration.kv_nvme_path[0] == '\0' ||
		state->configuration.kv_nvme_block_capacity == 0u ||
		state->configuration.kv_nvme_batch_block_count == 0u ||
		state->configuration.kv_nvme_batch_block_count >
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MAX_NVME_BATCH_BLOCK_COUNT ||
		state->kv_nvme_fd >= 0 || state->kv_nvme_staging != 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(&budget_request,0,sizeof(budget_request));
	budget_request.abi_version =
		SPARK_GLM52_KV_JIT_STAGE_BUDGET_ABI_VERSION;
	budget_request.descriptor_bytes =
		SPARK_GLM52_KV_JIT_STAGE_BUDGET_REQUEST_DESCRIPTOR_BYTES;
	budget_request.first_layer_index = state->rank_plan.first_layer_index;
	budget_request.layer_count = state->rank_plan.layer_count;
	budget_request.physical_pool_token_capacity =
		state->configuration.kv_pool_token_capacity;
	budget_request.backing_block_capacity =
		state->configuration.kv_nvme_block_capacity;
	budget_request.active_sequence_count =
		state->rank_plan.logical_lane_capacity;
	budget_request.backing_request_count =
		state->configuration.maximum_resident_sequence_count;
	budget_request.selected_token_count =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
	budget_request.include_mtp_layer =
		SparkGlm52Pp13BuilderMtpEnabled(state) &&
		SparkGlm52Pp13BuilderIsFinalRank(state) ? 1u : 0u;
	budget_request.block_token_count =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	budget_request.record_alignment_bytes =
		SPARK_GLM52_PP13_BUILDER_NVME_ALIGNMENT;
	budget_request.attention_cache_layout =
		SparkGlm52Pp13BuilderAttentionCacheLayout(state);
	budget_request.fp8_scale_block_size =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_SCALE_BLOCK;
	status = SparkGlm52KvCacheCalculateJitStageBudget(
		&budget_request,&state->kv_jit_budget);
	if (status != SPARK_STATUS_OK)
		return status;
	if (state->kv_jit_budget.mla_bytes_per_token %
			state->rank_plan.layer_count != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	attention_block_bytes =
		(state->kv_jit_budget.mla_bytes_per_token /
		 state->rank_plan.layer_count) *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	index_block_bytes =
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION *
		sizeof(uint16_t);
	summary_block_bytes =
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION *
		sizeof(uint16_t);
	payload_bytes = 0u;
	state->kv_nvme_dsa_index_layer_mask = 0u;
	for (layer_offset = 0u;
		 layer_offset < state->rank_plan.layer_count;
		 ++layer_offset)
	{
		layer_index = state->rank_plan.first_layer_index + layer_offset;
		if (payload_bytes > UINT64_MAX - attention_block_bytes)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		payload_bytes += attention_block_bytes;
		if (SparkGlm52Pp13BuilderDsaSourceLayer(layer_index) == layer_index)
		{
			uint64_t index_payload_bytes;
			index_payload_bytes = index_block_bytes +
				(summary_block_bytes * 2u) + sizeof(uint8_t);
			if (layer_offset >= 64u ||
				payload_bytes > UINT64_MAX - index_payload_bytes)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			payload_bytes += index_payload_bytes;
			state->kv_nvme_dsa_index_layer_mask |= 1ull << layer_offset;
		}
	}
	if (SparkGlm52Pp13BuilderMtpEnabled(state) &&
		SparkGlm52Pp13BuilderIsFinalRank(state))
	{
		if (payload_bytes > UINT64_MAX - attention_block_bytes)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		payload_bytes += attention_block_bytes;
	}
	state->kv_nvme_payload_bytes = payload_bytes;
	state->kv_nvme_record_bytes = SparkGlm52Pp13BuilderAlignUpU64(
		SPARK_GLM52_PP13_BUILDER_NVME_ALIGNMENT +
			state->kv_nvme_payload_bytes,
		SPARK_GLM52_PP13_BUILDER_NVME_ALIGNMENT);
	if (state->kv_nvme_payload_bytes !=
			state->kv_jit_budget.nvme_payload_bytes_per_block ||
		state->kv_nvme_record_bytes != state->kv_jit_budget.nvme_record_bytes ||
		state->kv_nvme_record_bytes == 0u ||
		state->configuration.kv_nvme_block_capacity >
			(uint64_t)INT64_MAX / state->kv_nvme_record_bytes)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	file_bytes = state->kv_nvme_record_bytes *
		state->configuration.kv_nvme_block_capacity;
	if (state->kv_nvme_record_bytes > UINT64_MAX / 2u ||
		state->configuration.kv_nvme_batch_block_count >
			UINT64_MAX / (state->kv_nvme_record_bytes * 2u))
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	staging_bytes = state->kv_nvme_record_bytes * 2u *
		state->configuration.kv_nvme_batch_block_count;
	if (staging_bytes == 0u || staging_bytes > SIZE_MAX)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	staging = 0;
	allocation_status = posix_memalign(
		&staging,SPARK_GLM52_PP13_BUILDER_NVME_ALIGNMENT,
		(size_t)staging_bytes);
	if (allocation_status != 0 || staging == 0)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if (cudaHostRegister(staging,(size_t)staging_bytes,
		cudaHostRegisterPortable) != cudaSuccess)
	{
		free(staging);
		return SPARK_STATUS_IO_ERROR;
	}
	open_flags = O_CREAT | O_RDWR | O_CLOEXEC;
#if defined(O_DIRECT)
	open_flags |= O_DIRECT;
#else
	cudaHostUnregister(staging);
	free(staging);
	return SPARK_STATUS_MODULE_NOT_VALIDATED;
#endif
	state->kv_nvme_fd = open(
		state->configuration.kv_nvme_path,open_flags,S_IRUSR | S_IWUSR);
	if (state->kv_nvme_fd < 0 ||
		flock(state->kv_nvme_fd,LOCK_EX | LOCK_NB) != 0 ||
		fstat(state->kv_nvme_fd,&nvme_stat) != 0 ||
		((uint64_t)nvme_stat.st_size != file_bytes &&
		 ftruncate(state->kv_nvme_fd,(off_t)file_bytes) != 0))
	{
		if (state->kv_nvme_fd >= 0)
			close(state->kv_nvme_fd);
		state->kv_nvme_fd = -1;
		cudaHostUnregister(staging);
		free(staging);
		return SPARK_STATUS_IO_ERROR;
	}
	state->kv_nvme_staging = staging;
	state->kv_nvme_staging_bytes = staging_bytes;
	if (cudaStreamCreateWithFlags(&state->kv_io_stream,
		cudaStreamNonBlocking) != cudaSuccess ||
		cudaEventCreateWithFlags(&state->kv_io_event,
		cudaEventDisableTiming) != cudaSuccess)
	{
		if (state->kv_io_stream != 0)
		{
			cudaStreamDestroy(state->kv_io_stream);
			state->kv_io_stream = 0;
		}
		flock(state->kv_nvme_fd,LOCK_UN);
		close(state->kv_nvme_fd);
		state->kv_nvme_fd = -1;
		cudaHostUnregister(staging);
		free(staging);
		state->kv_nvme_staging = 0;
		return SPARK_STATUS_IO_ERROR;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderKvNvmeCopyBlock(
	SparkGlm52Pp13BuilderState *state,
	uint32_t physical_block_index,
	uint8_t *staging,
	cudaMemcpyKind copy_kind)
{
	uint64_t mla_data_block_bytes;
	uint64_t mla_scale_block_bytes;
	uint64_t key_nope_data_block_bytes;
	uint64_t key_nope_scale_block_bytes;
	uint64_t value_data_block_bytes;
	uint64_t value_scale_block_bytes;
	uint64_t index_block_bytes;
	uint64_t summary_block_bytes;
	uint64_t payload_offset;
	uint32_t layer_offset;

	if (state == 0 || state->kv_nvme_staging == 0 || staging == 0 ||
		physical_block_index >= state->kv_state.physical_block_capacity ||
		(copy_kind != cudaMemcpyDeviceToHost &&
		 copy_kind != cudaMemcpyHostToDevice))
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->rank_plan.quantization_mode ==
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT)
	{
		mla_data_block_bytes =
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
		mla_scale_block_bytes =
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS *
			SPARK_GLM52_PP13_BUILDER_FP8_MLA_SCALE_COUNT * sizeof(float);
		key_nope_data_block_bytes =
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS *
			SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS;
		key_nope_scale_block_bytes =
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS *
			SPARK_GLM52_PP13_BUILDER_FP8_KEY_NOPE_SCALE_COUNT * sizeof(float);
		value_data_block_bytes =
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS *
			SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS;
		value_scale_block_bytes =
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS *
			SPARK_GLM52_PP13_BUILDER_FP8_VALUE_SCALE_COUNT * sizeof(float);
	}
	else
	{
		mla_data_block_bytes =
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS *
			sizeof(uint16_t);
		mla_scale_block_bytes = 0u;
		key_nope_data_block_bytes =
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS *
			SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS *
			sizeof(uint16_t);
		key_nope_scale_block_bytes = 0u;
		value_data_block_bytes =
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS *
			SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS *
			sizeof(uint16_t);
		value_scale_block_bytes = 0u;
	}
	index_block_bytes =
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION *
		sizeof(uint16_t);
	summary_block_bytes =
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION *
		sizeof(uint16_t);
	payload_offset = SPARK_GLM52_PP13_BUILDER_NVME_ALIGNMENT;
	for (layer_offset = 0u;
		 layer_offset < state->rank_plan.layer_count;
		 ++layer_offset)
	{
		SparkGlm52Pp13BuilderLayer *layer;
		uint8_t *mla_block;
		uint8_t *mla_scale_block;
		uint8_t *key_nope_block;
		uint8_t *key_nope_scale_block;
		uint8_t *value_block;
		uint8_t *value_scale_block;
		uint8_t *index_block;
		uint8_t *summary_min_block;
		uint8_t *summary_max_block;
		uint8_t *dirty_block;
		cudaError_t cuda_status;

		layer = &state->layers[layer_offset];
		if (mla_scale_block_bytes != 0u)
		{
			if (layer->mla_cache_fp8 == 0 || layer->mla_cache_scale == 0 ||
				layer->key_nope_cache_fp8 == 0 ||
				layer->key_nope_cache_scale == 0 ||
				layer->value_cache_fp8 == 0 ||
				layer->value_cache_scale == 0)
				return SPARK_STATUS_INVALID_ARGUMENT;
			mla_block = (uint8_t *)layer->mla_cache_fp8 +
				((uint64_t)physical_block_index * mla_data_block_bytes);
			mla_scale_block = (uint8_t *)layer->mla_cache_scale +
				((uint64_t)physical_block_index * mla_scale_block_bytes);
			key_nope_block = (uint8_t *)layer->key_nope_cache_fp8 +
				((uint64_t)physical_block_index * key_nope_data_block_bytes);
			key_nope_scale_block = (uint8_t *)layer->key_nope_cache_scale +
				((uint64_t)physical_block_index * key_nope_scale_block_bytes);
			value_block = (uint8_t *)layer->value_cache_fp8 +
				((uint64_t)physical_block_index * value_data_block_bytes);
			value_scale_block = (uint8_t *)layer->value_cache_scale +
				((uint64_t)physical_block_index * value_scale_block_bytes);
		}
		else
		{
			if (layer->mla_cache == 0 || layer->key_nope_cache == 0 ||
				layer->value_cache == 0)
				return SPARK_STATUS_INVALID_ARGUMENT;
			mla_block = (uint8_t *)layer->mla_cache +
				((uint64_t)physical_block_index * mla_data_block_bytes);
			mla_scale_block = 0;
			key_nope_block = (uint8_t *)layer->key_nope_cache +
				((uint64_t)physical_block_index * key_nope_data_block_bytes);
			key_nope_scale_block = 0;
			value_block = (uint8_t *)layer->value_cache +
				((uint64_t)physical_block_index * value_data_block_bytes);
			value_scale_block = 0;
		}
#define SPARK_GLM52_PP13_NVME_COPY(device_pointer, byte_count) \
		do { \
			void *destination_pointer = copy_kind == cudaMemcpyDeviceToHost \
				? (void *)(staging + payload_offset) : (void *)(device_pointer); \
			const void *source_pointer = copy_kind == cudaMemcpyDeviceToHost \
				? (const void *)(device_pointer) : (const void *)(staging + payload_offset); \
			cuda_status = cudaMemcpyAsync(destination_pointer,source_pointer, \
				(size_t)(byte_count),copy_kind,state->kv_io_stream); \
			if (cuda_status != cudaSuccess) return SPARK_STATUS_IO_ERROR; \
			payload_offset += (byte_count); \
		} while (0)
		SPARK_GLM52_PP13_NVME_COPY(mla_block,mla_data_block_bytes);
		if (mla_scale_block_bytes != 0u)
			SPARK_GLM52_PP13_NVME_COPY(mla_scale_block,mla_scale_block_bytes);
		SPARK_GLM52_PP13_NVME_COPY(key_nope_block,key_nope_data_block_bytes);
		if (key_nope_scale_block_bytes != 0u)
			SPARK_GLM52_PP13_NVME_COPY(
				key_nope_scale_block,key_nope_scale_block_bytes);
		SPARK_GLM52_PP13_NVME_COPY(value_block,value_data_block_bytes);
		if (value_scale_block_bytes != 0u)
			SPARK_GLM52_PP13_NVME_COPY(
				value_scale_block,value_scale_block_bytes);
		if ((state->kv_nvme_dsa_index_layer_mask &
				(1ull << layer_offset)) != 0u)
		{
			if (layer->key_index_cache == 0 ||
				layer->key_index_block_min == 0 ||
				layer->key_index_block_max == 0 ||
				layer->dsa_summary_dirty_flags == 0)
				return SPARK_STATUS_INVALID_ARGUMENT;
			index_block = (uint8_t *)layer->key_index_cache +
				((uint64_t)physical_block_index * index_block_bytes);
			summary_min_block = (uint8_t *)layer->key_index_block_min +
				((uint64_t)physical_block_index * summary_block_bytes);
			summary_max_block = (uint8_t *)layer->key_index_block_max +
				((uint64_t)physical_block_index * summary_block_bytes);
			dirty_block = (uint8_t *)layer->dsa_summary_dirty_flags +
				physical_block_index;
			SPARK_GLM52_PP13_NVME_COPY(index_block,index_block_bytes);
			SPARK_GLM52_PP13_NVME_COPY(summary_min_block,summary_block_bytes);
			SPARK_GLM52_PP13_NVME_COPY(summary_max_block,summary_block_bytes);
			SPARK_GLM52_PP13_NVME_COPY(dirty_block,sizeof(uint8_t));
		}
#undef SPARK_GLM52_PP13_NVME_COPY
	}
	if (SparkGlm52Pp13BuilderMtpEnabled(state) &&
		SparkGlm52Pp13BuilderIsFinalRank(state))
	{
		uint8_t *mtp_mla_block;
		uint8_t *mtp_mla_scale_block;
		uint8_t *mtp_key_nope_block;
		uint8_t *mtp_key_nope_scale_block;
		uint8_t *mtp_value_block;
		uint8_t *mtp_value_scale_block;
		cudaError_t cuda_status;
		if (mla_scale_block_bytes != 0u)
		{
			if (state->mtp_layer.mla_cache_fp8 == 0 ||
				state->mtp_layer.mla_cache_scale == 0 ||
				state->mtp_layer.key_nope_cache_fp8 == 0 ||
				state->mtp_layer.key_nope_cache_scale == 0 ||
				state->mtp_layer.value_cache_fp8 == 0 ||
				state->mtp_layer.value_cache_scale == 0)
				return SPARK_STATUS_INVALID_ARGUMENT;
			mtp_mla_block = (uint8_t *)state->mtp_layer.mla_cache_fp8 +
				((uint64_t)physical_block_index * mla_data_block_bytes);
			mtp_mla_scale_block = (uint8_t *)state->mtp_layer.mla_cache_scale +
				((uint64_t)physical_block_index * mla_scale_block_bytes);
			mtp_key_nope_block = (uint8_t *)state->mtp_layer.key_nope_cache_fp8 +
				((uint64_t)physical_block_index * key_nope_data_block_bytes);
			mtp_key_nope_scale_block =
				(uint8_t *)state->mtp_layer.key_nope_cache_scale +
				((uint64_t)physical_block_index * key_nope_scale_block_bytes);
			mtp_value_block = (uint8_t *)state->mtp_layer.value_cache_fp8 +
				((uint64_t)physical_block_index * value_data_block_bytes);
			mtp_value_scale_block = (uint8_t *)state->mtp_layer.value_cache_scale +
				((uint64_t)physical_block_index * value_scale_block_bytes);
		}
		else
		{
			if (state->mtp_layer.mla_cache == 0 ||
				state->mtp_layer.key_nope_cache == 0 ||
				state->mtp_layer.value_cache == 0)
				return SPARK_STATUS_INVALID_ARGUMENT;
			mtp_mla_block = (uint8_t *)state->mtp_layer.mla_cache +
				((uint64_t)physical_block_index * mla_data_block_bytes);
			mtp_mla_scale_block = 0;
			mtp_key_nope_block = (uint8_t *)state->mtp_layer.key_nope_cache +
				((uint64_t)physical_block_index * key_nope_data_block_bytes);
			mtp_key_nope_scale_block = 0;
			mtp_value_block = (uint8_t *)state->mtp_layer.value_cache +
				((uint64_t)physical_block_index * value_data_block_bytes);
			mtp_value_scale_block = 0;
		}
#define SPARK_GLM52_PP13_NVME_COPY_MTP(device_pointer, byte_count) \
		do { \
			void *destination_pointer = copy_kind == cudaMemcpyDeviceToHost \
				? (void *)(staging + payload_offset) : (void *)(device_pointer); \
			const void *source_pointer = copy_kind == cudaMemcpyDeviceToHost \
				? (const void *)(device_pointer) : (const void *)(staging + payload_offset); \
			cuda_status = cudaMemcpyAsync(destination_pointer,source_pointer, \
				(size_t)(byte_count),copy_kind,state->kv_io_stream); \
			if (cuda_status != cudaSuccess) return SPARK_STATUS_IO_ERROR; \
			payload_offset += (byte_count); \
		} while (0)
		SPARK_GLM52_PP13_NVME_COPY_MTP(mtp_mla_block,mla_data_block_bytes);
		if (mla_scale_block_bytes != 0u)
			SPARK_GLM52_PP13_NVME_COPY_MTP(
				mtp_mla_scale_block,mla_scale_block_bytes);
		SPARK_GLM52_PP13_NVME_COPY_MTP(
			mtp_key_nope_block,key_nope_data_block_bytes);
		if (key_nope_scale_block_bytes != 0u)
			SPARK_GLM52_PP13_NVME_COPY_MTP(
				mtp_key_nope_scale_block,key_nope_scale_block_bytes);
		SPARK_GLM52_PP13_NVME_COPY_MTP(
			mtp_value_block,value_data_block_bytes);
		if (value_scale_block_bytes != 0u)
			SPARK_GLM52_PP13_NVME_COPY_MTP(
				mtp_value_scale_block,value_scale_block_bytes);
#undef SPARK_GLM52_PP13_NVME_COPY_MTP
	}
	if (payload_offset != SPARK_GLM52_PP13_BUILDER_NVME_ALIGNMENT +
		state->kv_nvme_payload_bytes)
		return SPARK_STATUS_INTERNAL_ERROR;
	return SPARK_STATUS_OK;
}

static uint8_t *SparkGlm52Pp13BuilderKvNvmeStoreRecord(
	SparkGlm52Pp13BuilderState *state,
	uint32_t operation_index)
{
	return (uint8_t *)state->kv_nvme_staging +
		((uint64_t)operation_index * state->kv_nvme_record_bytes);
}

static uint8_t *SparkGlm52Pp13BuilderKvNvmeLoadRecord(
	SparkGlm52Pp13BuilderState *state,
	uint32_t operation_index)
{
	return (uint8_t *)state->kv_nvme_staging +
		((uint64_t)(state->configuration.kv_nvme_batch_block_count +
			operation_index) * state->kv_nvme_record_bytes);
}

static SparkStatus SparkGlm52Pp13BuilderKvNvmeRecordDependency(
	SparkGlm52Pp13BuilderState *state)
{
	if (state->kv_nvme_pending_store_count != 0u ||
		state->kv_nvme_pending_load_count != 0u)
		return SPARK_STATUS_OK;
	return cudaEventRecord(state->kv_io_event,state->stream) == cudaSuccess &&
		cudaStreamWaitEvent(state->kv_io_stream,state->kv_io_event,0u) == cudaSuccess
		? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus SparkGlm52Pp13BuilderKvNvmeValidateRecord(
	const SparkGlm52Pp13BuilderState *state,
	const uint8_t *record,
	uint64_t sequence_id,
	uint32_t logical_block_index,
	uint32_t backing_block_index)
{
	const SparkGlm52Pp13BuilderNvmeRecordHeader *header;
	header = (const SparkGlm52Pp13BuilderNvmeRecordHeader *)record;
	if (header->magic != SPARK_GLM52_PP13_BUILDER_NVME_RECORD_MAGIC ||
		header->abi_version !=
			SPARK_GLM52_PP13_BUILDER_NVME_RECORD_ABI_VERSION ||
		header->descriptor_bytes != sizeof(*header) ||
		header->rank_index != state->rank_plan.rank_index ||
		header->layer_count != state->rank_plan.layer_count ||
		header->block_token_count !=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS ||
		header->cache_token_elements !=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS ||
		header->index_key_dimension !=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION ||
		header->mtp_layer_included !=
			(SparkGlm52Pp13BuilderMtpEnabled(state) &&
			 SparkGlm52Pp13BuilderIsFinalRank(state) ? 1u : 0u) ||
		header->attention_cache_layout !=
			SparkGlm52Pp13BuilderAttentionCacheLayout(state) ||
		header->dsa_index_layer_mask !=
			state->kv_nvme_dsa_index_layer_mask ||
		header->sequence_id != sequence_id ||
		header->logical_block_index != logical_block_index ||
		header->backing_block_index != backing_block_index ||
		header->payload_bytes != state->kv_nvme_payload_bytes ||
		header->record_bytes != state->kv_nvme_record_bytes)
		return SPARK_STATUS_VALIDATION_FAILED;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderKvNvmeFlushBatch(
	SparkGlm52Pp13BuilderState *state)
{
	uint32_t operation_count;
	uint32_t operation_index;
	SparkStatus status;

	if (state == 0 || state->kv_nvme_staging == 0 ||
		state->kv_nvme_fd < 0 || state->kv_io_stream == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	operation_count = state->kv_nvme_pending_store_count +
		state->kv_nvme_pending_load_count;
	if (operation_count == 0u)
		return SPARK_STATUS_OK;
	status = SparkGlm52Pp13BuilderCudaStatus(
		cudaStreamSynchronize(state->kv_io_stream));
	if (status != SPARK_STATUS_OK)
		goto clear_batch;
	state->kv_nvme_synchronous_wait_count += 1u;
	for (operation_index = 0u;
		 operation_index < state->kv_nvme_pending_store_count;
		 ++operation_index)
	{
		const SparkGlm52Pp13BuilderNvmePendingOperation *operation;
		uint64_t file_offset;
		operation = &state->kv_nvme_pending_stores[operation_index];
		file_offset = (uint64_t)operation->backing_block_index *
			state->kv_nvme_record_bytes;
		status = SparkGlm52Pp13BuilderNvmeWriteExact(
			state->kv_nvme_fd,file_offset,
			SparkGlm52Pp13BuilderKvNvmeStoreRecord(state,operation_index),
			state->kv_nvme_record_bytes);
		if (status != SPARK_STATUS_OK)
			goto clear_batch;
		state->kv_nvme_store_count += 1u;
	}
	state->kv_nvme_load_count += state->kv_nvme_pending_load_count;
	state->kv_nvme_batch_flush_count += 1u;
	if (operation_count > state->kv_nvme_maximum_batch_operation_count)
		state->kv_nvme_maximum_batch_operation_count = operation_count;
	status = SPARK_STATUS_OK;

clear_batch:
	state->kv_nvme_pending_store_count = 0u;
	state->kv_nvme_pending_load_count = 0u;
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderKvNvmeStore(
	void *context,
	uint64_t sequence_id,
	uint32_t logical_block_index,
	uint32_t physical_block_index,
	uint32_t backing_block_index)
{
	SparkGlm52Pp13BuilderState *state;
	SparkGlm52Pp13BuilderNvmePendingOperation *operation;
	SparkGlm52Pp13BuilderNvmeRecordHeader *header;
	uint8_t *record;
	uint32_t operation_index;
	SparkStatus status;

	state = (SparkGlm52Pp13BuilderState *)context;
	if (state == 0 || sequence_id == 0u ||
		backing_block_index >= state->configuration.kv_nvme_block_capacity ||
		state->kv_nvme_fd < 0 || state->kv_nvme_staging == 0 ||
		state->stream == 0 || state->kv_io_stream == 0 || state->kv_io_event == 0 ||
		state->kv_nvme_batch_active == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->kv_nvme_pending_store_count >=
		state->configuration.kv_nvme_batch_block_count)
	{
		status = SparkGlm52Pp13BuilderKvNvmeFlushBatch(state);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	status = SparkGlm52Pp13BuilderKvNvmeRecordDependency(state);
	if (status != SPARK_STATUS_OK)
		return status;
	operation_index = state->kv_nvme_pending_store_count;
	record = SparkGlm52Pp13BuilderKvNvmeStoreRecord(state,operation_index);
	memset(record,0,(size_t)state->kv_nvme_record_bytes);
	header = (SparkGlm52Pp13BuilderNvmeRecordHeader *)record;
	header->magic = SPARK_GLM52_PP13_BUILDER_NVME_RECORD_MAGIC;
	header->abi_version = SPARK_GLM52_PP13_BUILDER_NVME_RECORD_ABI_VERSION;
	header->descriptor_bytes = (uint32_t)sizeof(*header);
	header->rank_index = state->rank_plan.rank_index;
	header->layer_count = state->rank_plan.layer_count;
	header->block_token_count =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	header->cache_token_elements =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
	header->index_key_dimension =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION;
	header->mtp_layer_included =
		SparkGlm52Pp13BuilderMtpEnabled(state) &&
		SparkGlm52Pp13BuilderIsFinalRank(state) ? 1u : 0u;
	header->attention_cache_layout =
		SparkGlm52Pp13BuilderAttentionCacheLayout(state);
	header->dsa_index_layer_mask = state->kv_nvme_dsa_index_layer_mask;
	header->sequence_id = sequence_id;
	header->logical_block_index = logical_block_index;
	header->backing_block_index = backing_block_index;
	header->payload_bytes = state->kv_nvme_payload_bytes;
	header->record_bytes = state->kv_nvme_record_bytes;
	status = SparkGlm52Pp13BuilderKvNvmeCopyBlock(
		state,physical_block_index,record,cudaMemcpyDeviceToHost);
	if (status != SPARK_STATUS_OK)
		return status;
	operation = &state->kv_nvme_pending_stores[operation_index];
	operation->sequence_id = sequence_id;
	operation->logical_block_index = logical_block_index;
	operation->physical_block_index = physical_block_index;
	operation->backing_block_index = backing_block_index;
	operation->reserved0 = 0u;
	state->kv_nvme_pending_store_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderKvNvmeLoad(
	void *context,
	uint64_t sequence_id,
	uint32_t logical_block_index,
	uint32_t physical_block_index,
	uint32_t backing_block_index)
{
	SparkGlm52Pp13BuilderState *state;
	SparkGlm52Pp13BuilderNvmePendingOperation *operation;
	uint8_t *record;
	uint64_t file_offset;
	uint32_t operation_index;
	SparkStatus status;

	state = (SparkGlm52Pp13BuilderState *)context;
	if (state == 0 || sequence_id == 0u ||
		backing_block_index >= state->configuration.kv_nvme_block_capacity ||
		state->kv_nvme_fd < 0 || state->kv_nvme_staging == 0 ||
		state->kv_nvme_batch_active == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->kv_nvme_pending_load_count >=
		state->configuration.kv_nvme_batch_block_count)
	{
		status = SparkGlm52Pp13BuilderKvNvmeFlushBatch(state);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	status = SparkGlm52Pp13BuilderKvNvmeRecordDependency(state);
	if (status != SPARK_STATUS_OK)
		return status;
	operation_index = state->kv_nvme_pending_load_count;
	record = SparkGlm52Pp13BuilderKvNvmeLoadRecord(state,operation_index);
	file_offset = (uint64_t)backing_block_index * state->kv_nvme_record_bytes;
	status = SparkGlm52Pp13BuilderNvmeReadExact(
		state->kv_nvme_fd,file_offset,record,state->kv_nvme_record_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderKvNvmeValidateRecord(
			state,record,sequence_id,logical_block_index,backing_block_index);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderKvNvmeCopyBlock(
			state,physical_block_index,record,cudaMemcpyHostToDevice);
	if (status != SPARK_STATUS_OK)
		return status;
	operation = &state->kv_nvme_pending_loads[operation_index];
	operation->sequence_id = sequence_id;
	operation->logical_block_index = logical_block_index;
	operation->physical_block_index = physical_block_index;
	operation->backing_block_index = backing_block_index;
	operation->reserved0 = 0u;
	state->kv_nvme_pending_load_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderReadToDevice(
	const char *path,
	uint64_t offset,
	uint64_t bytes,
	void *device_pointer)
{
	FILE *file;
	uint8_t *buffer;
	uint8_t *device_bytes;
	uint64_t copied;
	uint64_t chunk;
	size_t got;
	SparkStatus status;
	if (path == 0 || device_pointer == 0 || bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	file = fopen(path,"rb");
	if (file == 0)
	{
		fprintf(stderr,"pp13_builder_read_missing path=%s offset=%llu bytes=%llu\n",
			path,
			(unsigned long long)offset,
			(unsigned long long)bytes);
		return SPARK_STATUS_NOT_FOUND;
	}
	buffer = (uint8_t *)malloc((size_t)SPARK_GLM52_PP13_BUILDER_COPY_CHUNK_BYTES);
	if (buffer == 0)
	{
		fclose(file);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if (fseeko(file,(off_t)offset,SEEK_SET) != 0)
	{
		free(buffer);
		fclose(file);
		return SPARK_STATUS_IO_ERROR;
	}
	device_bytes = (uint8_t *)device_pointer;
	copied = 0u;
	status = SPARK_STATUS_OK;
	while (copied < bytes)
	{
		chunk = bytes - copied;
		if (chunk > SPARK_GLM52_PP13_BUILDER_COPY_CHUNK_BYTES)
			chunk = SPARK_GLM52_PP13_BUILDER_COPY_CHUNK_BYTES;
		got = fread(buffer,1u,(size_t)chunk,file);
		if (got != (size_t)chunk)
		{
			status = SPARK_STATUS_IO_ERROR;
			break;
		}
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
			device_bytes + copied,
			buffer,
			(size_t)chunk,
			cudaMemcpyHostToDevice));
		if (status != SPARK_STATUS_OK)
			break;
		copied += chunk;
	}
	free(buffer);
	fclose(file);
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderTensorSpec(
	SparkGlm52StagePackTensorSpec *spec,
	const char *name,
	const char *dtype,
	uint64_t bytes_per_element,
	uint32_t rank,
	uint64_t d0,
	uint64_t d1)
{
	if (spec == 0 || name == 0 || dtype == 0 || rank == 0u || rank > 2u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(spec,0,sizeof(*spec));
	spec->abi_version = SPARK_GLM52_STAGEPACK_ABI_VERSION;
	spec->rank = rank;
	spec->bytes_per_element = bytes_per_element;
	spec->shape[0] = d0;
	spec->shape[1] = rank > 1u ? d1 : 1u;
	spec->tensor_name = name;
	spec->dtype = dtype;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLoadTensor(
	SparkGlm52Pp13BuilderState *state,
	const char *name,
	const char *dtype,
	uint64_t bytes_per_element,
	uint32_t rank,
	uint64_t d0,
	uint64_t d1,
	void **device_out)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52StagePackTensorRegion region;
	SparkStatus status;
	if (device_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*device_out = 0;
	status = SparkGlm52Pp13BuilderTensorSpec(
		&spec,name,dtype,bytes_per_element,rank,d0,d1);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackResolveTensor(
			state->configuration.stagepack_root,
			&spec,
			&region);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus(name,UINT32_MAX,status);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,
			device_out,
			region.tensor_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderReadToDevice(
			region.file_path,
			region.file_offset,
			region.tensor_bytes,
			*device_out);
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderTensorName(
	char *name,
	uint32_t name_bytes,
	uint32_t layer_index,
	const char *suffix)
{
	int written;
	if (name == 0 || name_bytes == 0u || suffix == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	written = snprintf(name,name_bytes,"model.layers.%u.%s",layer_index,suffix);
	if (written < 0 || (uint32_t)written >= name_bytes)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLoadLayerTensor(
	SparkGlm52Pp13BuilderState *state,
	uint32_t layer_index,
	const char *suffix,
	const char *dtype,
	uint64_t bytes_per_element,
	uint32_t rank,
	uint64_t d0,
	uint64_t d1,
	void **device_out)
{
	char name[256];
	SparkStatus status;
	status = SparkGlm52Pp13BuilderTensorName(
		name,(uint32_t)sizeof(name),layer_index,suffix);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52Pp13BuilderLoadTensor(
		state,name,dtype,bytes_per_element,rank,d0,d1,device_out);
}

static SparkStatus SparkGlm52Pp13BuilderLoadLmHeadRestricted(
	SparkGlm52Pp13BuilderState *state,
	void **device_out)
{
	SparkGlm52StagePackTensorSpec spec;
	SparkGlm52StagePackTensorRegion region;
	uint64_t bytes;
	SparkStatus status;
	status = SparkGlm52Pp13BuilderTensorSpec(
		&spec,
		"lm_head.weight",
		"BF16",
		(uint32_t)sizeof(uint16_t),
		2u,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52StagePackResolveTensor(
			state->configuration.stagepack_root,
			&spec,
			&region);
	bytes =
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT *
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION *
		sizeof(uint16_t);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(state,device_out,bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderReadToDevice(
			region.file_path,
			region.file_offset,
			bytes,
			*device_out);
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderLoadEmbedding(
	SparkGlm52Pp13BuilderState *state)
{
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
		return SPARK_STATUS_OK;
	return SparkGlm52Pp13BuilderLoadTensor(
		state,
		"model.embed_tokens.weight",
		"BF16",
		(uint32_t)sizeof(uint16_t),
		2u,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
		&state->embedding_weight);
}

static SparkStatus SparkGlm52Pp13BuilderCopyU32ToDevice(
	SparkGlm52Pp13BuilderState *state,
	void **device_out,
	const uint32_t *values,
	uint32_t count)
{
	uint64_t bytes;
	SparkStatus status;
	if (values == 0 || count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	bytes = (uint64_t)count * sizeof(uint32_t);
	status = SparkGlm52Pp13BuilderCudaAlloc(state,device_out,bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
			*device_out,values,(size_t)bytes,cudaMemcpyHostToDevice));
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeTables(
	SparkGlm52Pp13BuilderState *state)
{
	uint32_t *tokens;
	uint64_t rope_pairs;
	uint64_t table_count;
	uint64_t table_bytes;
	uint32_t index;
	SparkStatus status;
	float *cos_host;
	float *sin_host;
	rope_pairs = SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u;
	table_count = (uint64_t)SPARK_GLM52_PP13_BUILDER_POSITION_COUNT * rope_pairs;
	table_bytes = table_count * sizeof(float);
	status = SparkGlm52Pp13BuilderCudaAlloc(state,(void **)&state->dsa_score_tiles,(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SCORE_TILE_ROWS * SPARK_GLM52_KV_CONTEXT_TOKENS * sizeof(float));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,(void **)&state->dsa_prefill_selected,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT * sizeof(uint32_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,(void **)&state->dsa_prefill_row_context_lengths,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * sizeof(uint32_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,(void **)&state->dsa_prefill_row_sequences,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * sizeof(uint32_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,(void **)&state->dsa_prefill_row_positions,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * sizeof(uint32_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->dsa_prefill_key_scratch,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION * sizeof(uint16_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->dsa_prefill_query_a,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION * sizeof(uint16_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->dsa_prefill_query_index_heads,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_QUERY_DIMENSION * sizeof(uint16_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->dsa_prefill_index_weights,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_WEIGHT_DIMENSION * sizeof(uint16_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->dsa_prefill_normalized_hidden,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION * sizeof(uint16_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->dsa_prefill_low_scratch,(uint64_t)SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS * SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION * sizeof(uint16_t));
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->device_probe_hash_slots,SPARK_GLM52_PP13_BUILDER_PROBE_HASH_SLOT_COUNT * sizeof(uint64_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->cos_table,table_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->sin_table,table_bytes);
	if (status == SPARK_STATUS_OK)
	{
		cos_host = (float *)malloc((size_t)table_bytes);
		sin_host = (float *)malloc((size_t)table_bytes);
		if (cos_host == 0 || sin_host == 0)
		{
			free(cos_host);
			free(sin_host);
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		}
		SparkGlm52BuildRopeTables(
			cos_host,
			sin_host,
			SPARK_GLM52_PP13_BUILDER_POSITION_COUNT);
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
			state->cos_table,cos_host,(size_t)table_bytes,cudaMemcpyHostToDevice));
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
				state->sin_table,sin_host,(size_t)table_bytes,cudaMemcpyHostToDevice));
		free(cos_host);
		free(sin_host);
	}
	tokens = (uint32_t *)malloc(
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT *
		sizeof(uint32_t));
	if (status != SPARK_STATUS_OK || tokens == 0)
	{
		free(tokens);
		return status == SPARK_STATUS_OK ? SPARK_STATUS_CAPACITY_EXCEEDED : status;
	}
	for (index = 0u;
		 index < SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT;
		 ++index)
		tokens[index] = index;
	status = SparkGlm52Pp13BuilderCopyU32ToDevice(
		state,
		&state->restricted_token_ids,
		tokens,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT);
	free(tokens);
	return status;
}

static void SparkGlm52Pp13BuilderAliasLayerScratch(
	SparkGlm52Pp13BuilderLayer *layer,
	const SparkGlm52Pp13BuilderLayer *scratch_owner,
	void *input_hidden)
{
	layer->input_hidden = input_hidden;
	layer->normalized_hidden = scratch_owner->normalized_hidden;
	layer->query_latent = scratch_owner->query_latent;
	layer->query_rope_input = scratch_owner->query_rope_input;
	layer->key_rope_input = scratch_owner->key_rope_input;
	layer->current_kv_latent = scratch_owner->current_kv_latent;
	layer->raw_query_a = scratch_owner->raw_query_a;
	layer->raw_query_a_norm = scratch_owner->raw_query_a_norm;
	layer->raw_query_b = scratch_owner->raw_query_b;
	layer->raw_kv_a = scratch_owner->raw_kv_a;
	layer->raw_kv_a_norm = scratch_owner->raw_kv_a_norm;
	layer->raw_kv_b = scratch_owner->raw_kv_b;
	layer->query_index_heads = scratch_owner->query_index_heads;
	layer->current_key_index = scratch_owner->current_key_index;
	layer->index_head_weights = scratch_owner->index_head_weights;
	layer->sparse_token_indices = scratch_owner->sparse_token_indices;
	layer->rotated_query_rope = scratch_owner->rotated_query_rope;
	layer->attention_output_latent = scratch_owner->attention_output_latent;
	layer->attention_projected_hidden = scratch_owner->attention_projected_hidden;
	layer->post_attention_hidden = scratch_owner->post_attention_hidden;
	layer->post_attention_normalized_hidden =
		scratch_owner->post_attention_normalized_hidden;
	layer->moe_topk_expert_ids = scratch_owner->moe_topk_expert_ids;
	layer->moe_topk_weights = scratch_owner->moe_topk_weights;
	layer->moe_router_logits = scratch_owner->moe_router_logits;
	layer->moe_gate = scratch_owner->moe_gate;
	layer->moe_up = scratch_owner->moe_up;
	layer->moe_intermediate = scratch_owner->moe_intermediate;
	layer->moe_route_output = scratch_owner->moe_route_output;
	layer->mtp_draft_hidden = scratch_owner->mtp_draft_hidden;
	layer->restricted_logits = scratch_owner->restricted_logits;
	layer->restricted_selected_token_ids =
		scratch_owner->restricted_selected_token_ids;
	layer->restricted_selected_token_scores =
		scratch_owner->restricted_selected_token_scores;
	layer->mtp_draft_logits = scratch_owner->mtp_draft_logits;
	layer->mtp_draft_token_ids = scratch_owner->mtp_draft_token_ids;
	layer->mtp_draft_token_budgets = scratch_owner->mtp_draft_token_budgets;
	layer->mtp_target_token_ids = scratch_owner->mtp_target_token_ids;
	layer->mtp_accept_mask = scratch_owner->mtp_accept_mask;
	layer->mtp_committed_token_ids = scratch_owner->mtp_committed_token_ids;
	layer->mtp_event_counters = scratch_owner->mtp_event_counters;
	layer->positions = scratch_owner->positions;
	layer->slot_mapping = scratch_owner->slot_mapping;
	layer->block_table = scratch_owner->block_table;
	layer->context_lengths = scratch_owner->context_lengths;
	layer->first_block_token_offsets =
		scratch_owner->first_block_token_offsets;
}

static SparkStatus SparkGlm52Pp13BuilderAllocateLayerBuffers(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_offset,
	uint32_t active_row_capacity,
	uint32_t mtp_row_capacity,
	uint64_t cache_token_capacity,
	uint64_t storage_token_capacity,
	uint32_t requires_dsa_index_cache)
{
	uint64_t b;
	uint64_t mtp_b;
	uint64_t kv_block_count;
	uint64_t route_count;
	uint32_t input_crosses_rank_boundary;
	uint32_t output_crosses_rank_boundary;
	uint32_t fp8;
	SparkStatus status;
	b = active_row_capacity;
	mtp_b = mtp_row_capacity;
	if (b == 0u || mtp_b == 0u || mtp_b > b ||
		storage_token_capacity < cache_token_capacity)
		return SPARK_STATUS_INVALID_ARGUMENT;
	kv_block_count = cache_token_capacity /
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	route_count = b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
	input_crosses_rank_boundary =
		layer_offset == 0u &&
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u;
	output_crosses_rank_boundary =
		layer_offset + 1u == state->rank_plan.layer_count &&
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u;
	fp8 = state->rank_plan.quantization_mode ==
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT;
#define ALLOC_FIELD(field, count, type) \
	do { status = SparkGlm52Pp13BuilderCudaAlloc(state,&layer->field,(uint64_t)(count) * sizeof(type)); if (status != SPARK_STATUS_OK) return status; } while (0)
#define ALLOC_FIELD_MAPPED(field, count, type) \
	do { status = SparkGlm52Pp13BuilderCudaHostMappedAlloc(state,&layer->field,(uint64_t)(count) * sizeof(type)); if (status != SPARK_STATUS_OK) return status; } while (0)
#define ZERO_FIELD(field, count, type) \
	do { status = SparkGlm52Pp13BuilderCudaZero(layer->field,(uint64_t)(count) * sizeof(type)); if (status != SPARK_STATUS_OK) return status; } while (0)
	if (layer_offset != 0u && layer_offset != UINT32_MAX)
	{
		if (layer_offset >= state->rank_plan.layer_count ||
			state->layers[0u].normalized_hidden == 0 ||
			state->layers[layer_offset - 1u].layer_output_hidden == 0)
			return SPARK_STATUS_INVALID_ARGUMENT;
		SparkGlm52Pp13BuilderAliasLayerScratch(
			layer,
			&state->layers[0u],
			state->layers[layer_offset - 1u].layer_output_hidden);
		if (output_crosses_rank_boundary != 0u)
			ALLOC_FIELD_MAPPED(layer_output_hidden,
				b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
		else
			ALLOC_FIELD(layer_output_hidden,
				b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
		ALLOC_FIELD(phase_clock_cycles,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT,uint64_t);
		if (fp8 != 0u)
		{
			ALLOC_FIELD(mla_cache_fp8,
				storage_token_capacity *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,uint8_t);
			ALLOC_FIELD(mla_cache_scale,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_FP8_MLA_SCALE_COUNT,float);
			ALLOC_FIELD(key_nope_cache_fp8,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS,uint8_t);
			ALLOC_FIELD(key_nope_cache_scale,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_FP8_KEY_NOPE_SCALE_COUNT,float);
			ALLOC_FIELD(value_cache_fp8,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS,uint8_t);
			ALLOC_FIELD(value_cache_scale,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_FP8_VALUE_SCALE_COUNT,float);
		}
		else
		{
			ALLOC_FIELD(mla_cache,
				storage_token_capacity *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,uint16_t);
			ALLOC_FIELD(key_nope_cache,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS,uint16_t);
			ALLOC_FIELD(value_cache,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS,uint16_t);
		}
		if (requires_dsa_index_cache != 0u)
		{
			ALLOC_FIELD(key_index_cache,
				storage_token_capacity *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
			ALLOC_FIELD(key_index_block_min,
				kv_block_count *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
			ALLOC_FIELD(key_index_block_max,
				kv_block_count *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
			ALLOC_FIELD(dsa_summary_dirty_flags,kv_block_count,uint8_t);
		}
		if (fp8 != 0u)
		{
			ZERO_FIELD(mla_cache_fp8,
				storage_token_capacity *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,uint8_t);
			ZERO_FIELD(mla_cache_scale,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_FP8_MLA_SCALE_COUNT,float);
			ZERO_FIELD(key_nope_cache_fp8,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS,uint8_t);
			ZERO_FIELD(key_nope_cache_scale,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_FP8_KEY_NOPE_SCALE_COUNT,float);
			ZERO_FIELD(value_cache_fp8,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS,uint8_t);
			ZERO_FIELD(value_cache_scale,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_FP8_VALUE_SCALE_COUNT,float);
		}
		else
		{
			ZERO_FIELD(mla_cache,
				storage_token_capacity *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,uint16_t);
			ZERO_FIELD(key_nope_cache,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS,uint16_t);
			ZERO_FIELD(value_cache,
				storage_token_capacity *
					SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS,uint16_t);
		}
		if (requires_dsa_index_cache != 0u)
		{
			ZERO_FIELD(key_index_cache,
				storage_token_capacity *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
			ZERO_FIELD(key_index_block_min,
				kv_block_count *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
			ZERO_FIELD(key_index_block_max,
				kv_block_count *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
			ZERO_FIELD(dsa_summary_dirty_flags,kv_block_count,uint8_t);
		}
		return SPARK_STATUS_OK;
	}
	if (input_crosses_rank_boundary != 0u)
		ALLOC_FIELD_MAPPED(input_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	else
		ALLOC_FIELD(input_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(normalized_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(query_latent,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_LATENT_PROJECTION_DIMENSION,uint16_t);
	ALLOC_FIELD(query_rope_input,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_ROPE_PROJECTION_DIMENSION,uint16_t);
	ALLOC_FIELD(key_rope_input,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION,uint16_t);
	ALLOC_FIELD(current_kv_latent,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,uint16_t);
	ALLOC_FIELD(raw_query_a,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,uint16_t);
	ALLOC_FIELD(raw_query_a_norm,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,uint16_t);
	ALLOC_FIELD(raw_query_b,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION,uint16_t);
	ALLOC_FIELD(raw_kv_a,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION,uint16_t);
	ALLOC_FIELD(raw_kv_a_norm,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,uint16_t);
	ALLOC_FIELD(raw_kv_b,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION,uint16_t);
	ALLOC_FIELD(query_index_heads,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_QUERY_DIMENSION,uint16_t);
	ALLOC_FIELD(current_key_index,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
	ALLOC_FIELD(index_head_weights,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_WEIGHT_DIMENSION,uint16_t);
	ALLOC_FIELD(sparse_token_indices,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT,uint32_t);
	ALLOC_FIELD(rotated_query_rope,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_ROPE_PROJECTION_DIMENSION,uint16_t);
	ALLOC_FIELD(attention_output_latent,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,uint16_t);
	ALLOC_FIELD(attention_projected_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(post_attention_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(post_attention_normalized_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(moe_topk_expert_ids,route_count,uint32_t);
	ALLOC_FIELD(moe_topk_weights,route_count,float);
	ALLOC_FIELD(moe_router_logits,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT,float);
	ALLOC_FIELD(moe_gate,route_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,uint16_t);
	ALLOC_FIELD(moe_up,route_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,uint16_t);
	ALLOC_FIELD(moe_intermediate,route_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,uint16_t);
	ALLOC_FIELD(moe_route_output,route_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	if (output_crosses_rank_boundary != 0u)
		ALLOC_FIELD_MAPPED(layer_output_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	else
		ALLOC_FIELD(layer_output_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(mtp_draft_hidden,mtp_b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	ALLOC_FIELD(restricted_logits,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT,float);
	ALLOC_FIELD(restricted_selected_token_ids,b,uint32_t);
	ALLOC_FIELD(restricted_selected_token_scores,b,float);
	ALLOC_FIELD(mtp_draft_logits,mtp_b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT,float);
	ALLOC_FIELD(mtp_draft_token_ids,mtp_b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT,uint32_t);
	ALLOC_FIELD(mtp_draft_token_budgets,mtp_b,uint32_t);
	ALLOC_FIELD(mtp_target_token_ids,mtp_b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT,uint32_t);
	ALLOC_FIELD(mtp_accept_mask,mtp_b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT,uint32_t);
	ALLOC_FIELD(mtp_committed_token_ids,mtp_b * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT,uint32_t);
	ALLOC_FIELD(mtp_event_counters,SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_EVENT_COUNTER_COUNT,uint32_t);
	ALLOC_FIELD(phase_clock_cycles,SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT,uint64_t);
	ALLOC_FIELD(positions,b,uint32_t);
	ALLOC_FIELD(slot_mapping,b,uint32_t);
	layer->block_table = state->device_physical_block_indices;
	if (layer->block_table == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	ALLOC_FIELD(context_lengths,b,uint32_t);
	ALLOC_FIELD(first_block_token_offsets,b,uint32_t);
	if (fp8 != 0u)
	{
		ALLOC_FIELD(mla_cache_fp8,storage_token_capacity *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,uint8_t);
		ALLOC_FIELD(mla_cache_scale,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_FP8_MLA_SCALE_COUNT,float);
		ALLOC_FIELD(key_nope_cache_fp8,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS,uint8_t);
		ALLOC_FIELD(key_nope_cache_scale,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_FP8_KEY_NOPE_SCALE_COUNT,float);
		ALLOC_FIELD(value_cache_fp8,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS,uint8_t);
		ALLOC_FIELD(value_cache_scale,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_FP8_VALUE_SCALE_COUNT,float);
	}
	else
	{
		ALLOC_FIELD(mla_cache,storage_token_capacity *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,uint16_t);
		ALLOC_FIELD(key_nope_cache,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS,uint16_t);
		ALLOC_FIELD(value_cache,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS,uint16_t);
	}
	if (requires_dsa_index_cache != 0u)
	{
		ALLOC_FIELD(key_index_cache,storage_token_capacity * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
		ALLOC_FIELD(key_index_block_min,kv_block_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
		ALLOC_FIELD(key_index_block_max,kv_block_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
		ALLOC_FIELD(dsa_summary_dirty_flags,kv_block_count,uint8_t);
	}
	ZERO_FIELD(input_hidden,b * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,uint16_t);
	if (fp8 != 0u)
	{
		ZERO_FIELD(mla_cache_fp8,storage_token_capacity *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,uint8_t);
		ZERO_FIELD(mla_cache_scale,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_FP8_MLA_SCALE_COUNT,float);
		ZERO_FIELD(key_nope_cache_fp8,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS,uint8_t);
		ZERO_FIELD(key_nope_cache_scale,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_FP8_KEY_NOPE_SCALE_COUNT,float);
		ZERO_FIELD(value_cache_fp8,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS,uint8_t);
		ZERO_FIELD(value_cache_scale,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_FP8_VALUE_SCALE_COUNT,float);
	}
	else
	{
		ZERO_FIELD(mla_cache,storage_token_capacity *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,uint16_t);
		ZERO_FIELD(key_nope_cache,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS,uint16_t);
		ZERO_FIELD(value_cache,storage_token_capacity *
			SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS,uint16_t);
	}
	if (requires_dsa_index_cache != 0u)
	{
		ZERO_FIELD(key_index_cache,storage_token_capacity * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
		ZERO_FIELD(key_index_block_min,kv_block_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
		ZERO_FIELD(key_index_block_max,kv_block_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,uint16_t);
		ZERO_FIELD(dsa_summary_dirty_flags,kv_block_count,uint8_t);
	}
#undef ALLOC_FIELD
#undef ALLOC_FIELD_MAPPED
#undef ZERO_FIELD
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLoadLayerWeights(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_index)
{
	uint32_t bf16_trunk;
	SparkStatus status;
#define LOAD(suffix, dtype, bpe, rank, d0, d1, field) \
	do { status = SparkGlm52Pp13BuilderLoadLayerTensor(state,layer_index,suffix,dtype,bpe,rank,d0,d1,&layer->field); if (status != SPARK_STATUS_OK) return status; } while (0)
#define LOAD_WEIGHT(suffix, d0, d1, bf16_field, fp8_field, scale_field) \
	do { \
		if (bf16_trunk != 0u) \
			LOAD(suffix,"BF16",sizeof(uint16_t),2u,d0,d1,bf16_field); \
		else \
		{ \
			LOAD(suffix,"F8_E4M3",sizeof(uint8_t),2u,d0,d1,fp8_field); \
			LOAD(suffix "_scale_inv","F32",sizeof(float),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(d0),SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_EXTENT(d1),scale_field); \
		} \
	} while (0)
	bf16_trunk = SparkGlm52Pp13BuilderUsesBf16Trunk(state);
	LOAD("input_layernorm.weight","BF16",sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,1u,attention_norm_weight);
	LOAD("post_attention_layernorm.weight","BF16",sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,1u,post_attention_norm_weight);
	LOAD("self_attn.q_a_layernorm.weight","BF16",sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,1u,raw_query_a_norm_weight);
	LOAD("self_attn.kv_a_layernorm.weight","BF16",sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,1u,raw_kv_a_norm_weight);
	LOAD_WEIGHT("self_attn.q_a_proj.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,raw_query_a_weight_bf16,raw_query_a_weight_fp8,raw_query_a_scale);
	LOAD_WEIGHT("self_attn.q_b_proj.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,raw_query_b_weight_bf16,raw_query_b_weight_fp8,raw_query_b_scale);
	LOAD_WEIGHT("self_attn.kv_a_proj_with_mqa.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,raw_kv_a_weight_bf16,raw_kv_a_weight_fp8,raw_kv_a_scale);
	LOAD_WEIGHT("self_attn.kv_b_proj.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION,raw_kv_b_weight_bf16,raw_kv_b_weight_fp8,raw_kv_b_scale);
	LOAD_WEIGHT("self_attn.o_proj.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION,attention_output_weight_bf16,attention_output_weight_fp8,attention_output_scale);
	if (SparkGlm52Pp13BuilderDsaSourceLayer(layer_index) == layer_index)
	{
		LOAD_WEIGHT("self_attn.indexer.wq_b.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_QUERY_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,index_query_weight_bf16,index_query_weight_fp8,index_query_scale);
		LOAD_WEIGHT("self_attn.indexer.wk.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,index_key_weight_bf16,index_key_weight_fp8,index_key_scale);
		LOAD("self_attn.indexer.weights_proj.weight","BF16",sizeof(uint16_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_WEIGHT_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,index_weights_proj_weight);
		LOAD("self_attn.indexer.k_norm.weight","BF16",sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,1u,index_key_norm_weight);
		LOAD("self_attn.indexer.k_norm.bias","BF16",sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION,1u,index_key_norm_bias);
	}
	if (layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
	{
		LOAD_WEIGHT("mlp.gate_proj.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,dense_gate_weight_bf16,dense_gate_weight_fp8,dense_gate_scale);
		LOAD_WEIGHT("mlp.up_proj.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,dense_up_weight_bf16,dense_up_weight_fp8,dense_up_scale);
		LOAD_WEIGHT("mlp.down_proj.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION,dense_down_weight_bf16,dense_down_weight_fp8,dense_down_scale);
	}
	else
	{
		LOAD("mlp.gate.weight","BF16",sizeof(uint16_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,router_weight);
		LOAD("mlp.gate.e_score_correction_bias","F32",sizeof(float),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT,1u,router_bias);
		LOAD_WEIGHT("mlp.shared_experts.gate_proj.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,dense_gate_weight_bf16,dense_gate_weight_fp8,dense_gate_scale);
		LOAD_WEIGHT("mlp.shared_experts.up_proj.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,dense_up_weight_bf16,dense_up_weight_fp8,dense_up_scale);
		LOAD_WEIGHT("mlp.shared_experts.down_proj.weight",SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION,dense_down_weight_bf16,dense_down_weight_fp8,dense_down_scale);
	}
#undef LOAD_WEIGHT
#undef LOAD
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13BuilderWireLayer(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_index)
{
	SparkGlm52ResidentDecodeStagePipelineSlot *slot;
	SparkGlm52ResidentDecodeStageNodeContext *node;
	uint32_t source_layer;
	uint32_t group_end;
	uint32_t bf16_trunk;
	uint32_t nvfp4;
	uint32_t w8lut;
	slot = &layer->slot;
	node = &layer->node;
	bf16_trunk = SparkGlm52Pp13BuilderUsesBf16Trunk(state);
	nvfp4 = SparkGlm52Pp13BuilderUsesNvfp4(state);
	w8lut = SparkGlm52Pp13BuilderUsesW8lut(state);
	memset(slot,0,sizeof(*slot));
	slot->cuda_stream = (void *)state->stream;
	slot->input_hidden_bf16 = layer->input_hidden;
	slot->normalized_hidden_bf16 = layer->normalized_hidden;
	slot->query_latent_bf16 = layer->query_latent;
	slot->query_rope_input_bf16 = layer->query_rope_input;
	slot->key_rope_input_bf16 = layer->key_rope_input;
	slot->current_kv_latent_bf16 = layer->current_kv_latent;
	slot->raw_query_a_bf16 = layer->raw_query_a;
	slot->raw_query_a_normalized_bf16 = layer->raw_query_a_norm;
	slot->raw_query_b_bf16 = layer->raw_query_b;
	slot->raw_kv_a_bf16 = layer->raw_kv_a;
	slot->raw_kv_a_normalized_bf16 = layer->raw_kv_a_norm;
	slot->raw_kv_b_bf16 = layer->raw_kv_b;
	slot->positions = (const uint32_t *)layer->positions;
	slot->slot_mapping = (const uint32_t *)layer->slot_mapping;
	slot->block_table = (const uint32_t *)layer->block_table;
	slot->context_lengths = (const uint32_t *)layer->context_lengths;
	slot->first_block_token_offsets = (const uint32_t *)layer->first_block_token_offsets;
	slot->query_index_heads_bf16 = layer->query_index_heads;
	slot->current_key_index_bf16 = layer->current_key_index;
	slot->index_head_weights_bf16 = layer->index_head_weights;
	slot->dsa_candidate_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
	slot->sparse_token_indices = (uint32_t *)layer->sparse_token_indices;
	slot->rotated_query_rope_bf16 = layer->rotated_query_rope;
	slot->attention_output_latent_bf16 = layer->attention_output_latent;
	slot->attention_projected_hidden_bf16 = layer->attention_projected_hidden;
	slot->post_attention_hidden_bf16 = layer->post_attention_hidden;
	slot->post_attention_normalized_hidden_bf16 = layer->post_attention_normalized_hidden;
	slot->moe_router_logits = (float *)layer->moe_router_logits;
	slot->moe_topk_expert_ids = (uint32_t *)layer->moe_topk_expert_ids;
	slot->moe_topk_weights = (float *)layer->moe_topk_weights;
	slot->moe_gate_bf16 = layer->moe_gate;
	slot->moe_up_bf16 = layer->moe_up;
	slot->moe_intermediate_bf16 = layer->moe_intermediate;
	slot->moe_route_output_bf16 = layer->moe_route_output;
	slot->layer_output_hidden_bf16 = layer->layer_output_hidden;
	slot->mtp_draft_hidden_bf16 = layer->mtp_draft_hidden;
	slot->restricted_logits = (float *)layer->restricted_logits;
	slot->restricted_selected_token_ids = (uint32_t *)layer->restricted_selected_token_ids;
	slot->restricted_selected_token_scores = (float *)layer->restricted_selected_token_scores;
	slot->mtp_draft_logits = (float *)layer->mtp_draft_logits;
	slot->mtp_draft_token_ids = (uint32_t *)layer->mtp_draft_token_ids;
	slot->mtp_draft_token_budgets = (const uint32_t *)layer->mtp_draft_token_budgets;
	slot->mtp_target_token_ids = (const uint32_t *)layer->mtp_target_token_ids;
	slot->mtp_accept_mask = (uint32_t *)layer->mtp_accept_mask;
	slot->mtp_committed_token_ids = (uint32_t *)layer->mtp_committed_token_ids;
	slot->mtp_event_counters = (uint32_t *)layer->mtp_event_counters;
	slot->phase_clock_cycles = (uint64_t *)layer->phase_clock_cycles;
	slot->mtp_tree_shadow_slot_mapping =
		state->device_mtp_tree_shadow_slot_mapping;
	memset(&layer->cuda_slot,0,sizeof(layer->cuda_slot));
	layer->cuda_slot.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_SLOT_STATE_ABI_VERSION;
	memset(node,0,sizeof(*node));
	node->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	node->pipeline_slot_count = SPARK_GLM52_PP13_BUILDER_PIPELINE_SLOT_COUNT;
	node->max_active_sequence_count =
		SparkGlm52Pp13BuilderLayerActiveRowCapacity(state,layer);
	node->logical_lane_capacity = state->rank_plan.logical_lane_capacity;
	node->cache_token_capacity = state->configuration.kv_pool_token_capacity;
	node->kv_storage_token_capacity = state->cache_storage_token_capacity;
	node->kv_block_count = state->configuration.kv_pool_token_capacity /
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	node->max_blocks_per_sequence =
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	node->position_count = SPARK_GLM52_PP13_BUILDER_POSITION_COUNT;
	node->dsa_candidate_capacity = SPARK_GLM52_KV_CONTEXT_TOKENS;
	node->qk_scale = SPARK_GLM52_MODEL_QK_SCALE;
	node->rms_norm_epsilon = SPARK_GLM52_MODEL_RMS_NORM_EPSILON;
	node->cos_table = (const float *)state->cos_table;
	node->sin_table = (const float *)state->sin_table;
	node->mla_cache_bf16 = bf16_trunk != 0u ? layer->mla_cache : 0;
	node->key_nope_cache_bf16 = bf16_trunk != 0u ? layer->key_nope_cache : 0;
	node->value_cache_bf16 = bf16_trunk != 0u ? layer->value_cache : 0;
	memset(&layer->fp8_kv_cache_plan,0,sizeof(layer->fp8_kv_cache_plan));
	if (bf16_trunk == 0u)
	{
		layer->fp8_kv_cache_plan.abi_version =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_PLAN_ABI_VERSION;
		layer->fp8_kv_cache_plan.capability_flags =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_REQUIRED_CAPABILITIES;
		layer->fp8_kv_cache_plan.maximum_active_sequence_count =
			SparkGlm52Pp13BuilderLayerActiveRowCapacity(state,layer);
		layer->fp8_kv_cache_plan.cache_token_capacity =
			state->configuration.kv_pool_token_capacity;
		layer->fp8_kv_cache_plan.cache_token_elements =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
		layer->fp8_kv_cache_plan.key_nope_elements =
			SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS;
		layer->fp8_kv_cache_plan.value_elements =
			SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS;
		layer->fp8_kv_cache_plan.scale_block_size =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_SCALE_BLOCK;
		layer->fp8_kv_cache_plan.mla_cache_fp8_e4m3 =
			(uint8_t *)layer->mla_cache_fp8;
		layer->fp8_kv_cache_plan.mla_cache_scale_f32 =
			(float *)layer->mla_cache_scale;
		layer->fp8_kv_cache_plan.key_nope_cache_fp8_e4m3 =
			(uint8_t *)layer->key_nope_cache_fp8;
		layer->fp8_kv_cache_plan.key_nope_cache_scale_f32 =
			(float *)layer->key_nope_cache_scale;
		layer->fp8_kv_cache_plan.value_cache_fp8_e4m3 =
			(uint8_t *)layer->value_cache_fp8;
		layer->fp8_kv_cache_plan.value_cache_scale_f32 =
			(float *)layer->value_cache_scale;
		node->fp8_kv_cache_plan = &layer->fp8_kv_cache_plan;
	}
	node->attention_norm_weight_bf16 = layer->attention_norm_weight;
	node->raw_query_a_norm_weight_bf16 = layer->raw_query_a_norm_weight;
	node->raw_kv_a_norm_weight_bf16 = layer->raw_kv_a_norm_weight;
	node->raw_query_a_weight_bf16 = layer->raw_query_a_weight_bf16;
	node->raw_query_a_weight_fp8_e4m3 = (const uint8_t *)layer->raw_query_a_weight_fp8;
	node->raw_query_a_weight_scale_inv_f32 = (const float *)layer->raw_query_a_scale;
	node->raw_query_b_weight_bf16 = layer->raw_query_b_weight_bf16;
	node->raw_query_b_weight_fp8_e4m3 = (const uint8_t *)layer->raw_query_b_weight_fp8;
	node->raw_query_b_weight_scale_inv_f32 = (const float *)layer->raw_query_b_scale;
	node->raw_kv_a_weight_bf16 = layer->raw_kv_a_weight_bf16;
	node->raw_kv_a_weight_fp8_e4m3 = (const uint8_t *)layer->raw_kv_a_weight_fp8;
	node->raw_kv_a_weight_scale_inv_f32 = (const float *)layer->raw_kv_a_scale;
	node->raw_kv_b_weight_bf16 = layer->raw_kv_b_weight_bf16;
	node->raw_kv_b_weight_fp8_e4m3 = (const uint8_t *)layer->raw_kv_b_weight_fp8;
	node->raw_kv_b_weight_scale_inv_f32 = (const float *)layer->raw_kv_b_scale;
	node->attention_output_weight_bf16 = layer->attention_output_weight_bf16;
	node->attention_output_weight_fp8_e4m3 = (const uint8_t *)layer->attention_output_weight_fp8;
	node->attention_output_weight_scale_inv_f32 = (const float *)layer->attention_output_scale;
	node->post_attention_norm_weight_bf16 = layer->post_attention_norm_weight;
	node->dense_gate_weight_bf16 = layer->dense_gate_weight_bf16;
	node->dense_gate_weight_fp8_e4m3 = (const uint8_t *)layer->dense_gate_weight_fp8;
	node->dense_gate_weight_scale_inv_f32 = (const float *)layer->dense_gate_scale;
	node->dense_up_weight_bf16 = layer->dense_up_weight_bf16;
	node->dense_up_weight_fp8_e4m3 = (const uint8_t *)layer->dense_up_weight_fp8;
	node->dense_up_weight_scale_inv_f32 = (const float *)layer->dense_up_scale;
	node->dense_down_weight_bf16 = layer->dense_down_weight_bf16;
	node->dense_down_weight_fp8_e4m3 = (const uint8_t *)layer->dense_down_weight_fp8;
	node->dense_down_weight_scale_inv_f32 = (const float *)layer->dense_down_scale;
	node->final_norm_weight_bf16 = layer->final_norm_weight;
	node->restricted_lm_head_weight_bf16 = layer->restricted_lm_head_weight;
	node->restricted_token_ids = (const uint32_t *)state->restricted_token_ids;
	node->pipeline_slots = slot;
	node->cuda_pipeline_slot_states = &layer->cuda_slot;
	node->projection_mode = bf16_trunk != 0u
		? SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_BF16
		: SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3;
	node->projection_backend_mode = bf16_trunk != 0u
		? SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_CUBLASLT
		: SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_BACKEND_PREBOUND_TENSOR_CORE;
	node->attention_execution_mode =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_EXECUTION_TILED_ONLINE_SOFTMAX;
	node->dsa_score_tiles_f32 = state->dsa_score_tiles;
	node->dsa_prefill_selected_u32 = state->dsa_prefill_selected;
	node->dsa_prefill_row_context_lengths_u32 = state->dsa_prefill_row_context_lengths;
	node->dsa_prefill_row_sequences_u32 = state->dsa_prefill_row_sequences;
	node->dsa_prefill_row_positions_u32 = state->dsa_prefill_row_positions;
	node->dsa_prefill_key_scratch_bf16 = state->dsa_prefill_key_scratch;
	node->dsa_prefill_query_a_bf16 = state->dsa_prefill_query_a;
	node->dsa_prefill_query_index_heads_bf16 = state->dsa_prefill_query_index_heads;
	node->dsa_prefill_index_weights_bf16 = state->dsa_prefill_index_weights;
	node->dsa_prefill_normalized_hidden_bf16 = state->dsa_prefill_normalized_hidden;
	node->dsa_prefill_low_scratch_bf16 = state->dsa_prefill_low_scratch;
	node->dsa_prefill_row_capacity = SPARK_GLM52_PP13_BUILDER_PREFILL_ROWS;
	node->dsa_score_row_capacity =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SCORE_TILE_ROWS;
	node->model_quantization_mode = nvfp4 != 0u
		? SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_NVFP4_4BIT
		: (w8lut != 0u
			? SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_W8LUT_8BIT
			: SPARK_GLM52_RESIDENT_DECODE_STAGE_MODEL_QUANTIZATION_FP8_E4M3_8BIT);
	node->layer_index = layer_index;
	node->device_probe_hash_slots = state->device_probe_hash_slots;
	node->kv_block_token_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	node->launch_check_mode = SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_NONE;
	node->phase_clock_mode = SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DISABLED;
	node->enable_cuda_graph_replay = 1u;
	node->reserved_execution_flags =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_PREBOUND_PROJECTIONS |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_GRAPH_REPLAY |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MLP |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_TILED_ONLINE_ATTENTION |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_FORBID_DEBUG_SYNCHRONIZATION |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_STAGE_SLICE_PLAN |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_MODEL_QUANTIZATION |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_DSA_SPARSE_PREFILL |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_RUNTIME_KV_BLOCK_TABLE;
	if (bf16_trunk == 0u)
		node->reserved_execution_flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FP8_KV_CACHE;
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
		node->reserved_execution_flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_INPUT;
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
		node->reserved_execution_flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_OUTPUT;
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) == 0u ||
		layer_index + 1u != state->rank_plan.first_layer_index + state->rank_plan.layer_count)
		node->reserved_execution_flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_OUTPUT_HIDDEN_ONLY;
	node->validated_stage_latency_ns = 0u;
	node->estimated_service_time_ns = 0u;
	node->index_softmax_scale = SPARK_GLM52_MODEL_DSA_INDEX_SOFTMAX_SCALE;
	node->dsa_index_head_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_HEAD_COUNT;
	node->dsa_index_head_dimension = SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_HEAD_DIMENSION;
	node->index_query_weight_bf16 = layer->index_query_weight_bf16;
	node->index_query_weight_fp8_e4m3 = (const uint8_t *)layer->index_query_weight_fp8;
	node->index_query_weight_scale_inv_f32 = (const float *)layer->index_query_scale;
	node->index_key_weight_bf16 = layer->index_key_weight_bf16;
	node->index_key_weight_fp8_e4m3 = (const uint8_t *)layer->index_key_weight_fp8;
	node->index_key_weight_scale_inv_f32 = (const float *)layer->index_key_scale;
	node->index_weights_proj_weight_bf16 = layer->index_weights_proj_weight;
	node->index_key_norm_weight_bf16 = layer->index_key_norm_weight;
	node->index_key_norm_bias_bf16 = layer->index_key_norm_bias;
	node->key_index_cache_bf16 = layer->key_index_cache;
	node->key_index_block_min_bf16 = layer->key_index_block_min;
	node->key_index_block_max_bf16 = layer->key_index_block_max;
	node->dsa_summary_dirty_flags_u8 = (uint8_t *)layer->dsa_summary_dirty_flags;
	node->selected_token_indices_by_layer = (uint32_t *)state->selected_token_indices_by_layer;
	node->selected_block_indices_by_layer = (uint32_t *)state->selected_block_indices_by_layer;
	node->selected_block_counts_by_layer = (uint32_t *)state->selected_block_counts_by_layer;
	node->dsa_selection_epoch_by_layer = (uint32_t *)state->dsa_selection_epoch_by_layer;
	node->dsa_selected_block_stride =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_BLOCK_COUNT;
	node->dsa_selected_block_capacity =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_BLOCK_COUNT;
	node->dsa_selected_block_layer_count =
		state->dsa_cache_layer_count;
	node->dsa_cache_first_layer_index =
		state->dsa_cache_first_layer_index;
	source_layer = SparkGlm52Pp13BuilderDsaSourceLayer(layer_index);
	if (source_layer == UINT32_MAX)
	{
		node->sparse_index_mode =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_COPY_CONTEXT_PREFIX;
	}
	else
	{
		group_end = SparkGlm52Pp13BuilderDsaGroupEnd(source_layer);
		node->sparse_index_mode = source_layer == layer_index
			? SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_FULL
			: SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_SHARED;
		node->dsa_indexshare_source_layer_index = source_layer;
		node->dsa_indexshare_group_end_layer_exclusive = group_end;
		node->dsa_indexshare_selected_token_count =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
		node->dsa_indexshare_layer_count =
			state->dsa_cache_layer_count;
	}
	if (layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
	{
		node->layer_progression_mode =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_DENSE_BF16_MLP;
		node->mlp_execution_mode = bf16_trunk != 0u
			? SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_TENSOR_CORE
			: SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_PREBOUND_QUANTIZED_TENSOR_CORE;
		node->dense_intermediate_dimension =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION;
	}
	else
	{
		if (nvfp4 != 0u)
		{
			node->layer_progression_mode =
				SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_NVFP4_TOPK;
			node->mlp_execution_mode =
				SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FLASHINFER_B12X_MOE;
		}
		else if (w8lut != 0u)
		{
			node->layer_progression_mode =
				SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_W8LUT_TOPK;
			node->mlp_execution_mode =
				SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_W8LUT_EXPERT_TENSOR_CORE;
		}
		else
		{
			node->layer_progression_mode =
				SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYER_ROUTED_FP8_TOPK;
			node->mlp_execution_mode =
				SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_FP8_EXPERT_TENSOR_CORE;
		}
		node->moe_router_weight_bf16 = layer->router_weight;
		node->moe_router_score_bias_f32 = (const float *)layer->router_bias;
		node->moe_routed_scaling_factor =
			SPARK_GLM52_MODEL_MOE_ROUTED_SCALING_FACTOR;
		node->moe_norm_topk_prob = 1u;
		node->moe_expert_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
		node->moe_bound_expert_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
		node->moe_top_k = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_TOP_K;
		node->moe_intermediate_dimension =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
		node->dense_intermediate_dimension =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
		node->reserved_execution_flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER;
	}
}

static void SparkGlm52Pp13BuilderWireDsaFragmentPayload(
	SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPlan *plan,
	uint32_t payload_index,
	void *source_base,
	uint32_t token_bytes,
	uint32_t flags)
{
	SparkGlm52ResidentDecodeStageDsaKvFragmentTransportPayload *payload;
	uint64_t block_bytes;
	block_bytes =
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS *
		token_bytes;
	payload = &plan->payloads[payload_index];
	payload->abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PLAN_ABI_VERSION;
	payload->descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_DESCRIPTOR_BYTES;
	payload->flags = flags |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_ENABLED |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_L2_PREFETCH_ONLY;
	payload->source_block_stride_bytes = block_bytes;
	payload->transfer_bytes = block_bytes;
	payload->source_base = source_base;
}

static void SparkGlm52Pp13BuilderWireDsaFragmentPrefetch(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer)
{
	uint32_t fp8;
	fp8 = SparkGlm52Pp13BuilderUsesBf16Trunk(state) == 0u ? 1u : 0u;
	memset(&layer->dsa_prefetch_plan,0,sizeof(layer->dsa_prefetch_plan));
	layer->dsa_prefetch_plan.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PLAN_ABI_VERSION;
	layer->dsa_prefetch_plan.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PLAN_DESCRIPTOR_BYTES;
	layer->dsa_prefetch_plan.capability_flags =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_READ_ONLY_REQUIRED_CAPABILITIES;
	layer->dsa_prefetch_plan.payload_count = fp8 != 0u ? 6u : 3u;
	layer->dsa_prefetch_plan.physical_block_count = layer->node.kv_block_count;
	layer->dsa_prefetch_plan.maximum_active_sequence_count =
		state->rank_plan.execution_row_capacity;
	layer->dsa_prefetch_plan.selected_block_stride =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_BLOCK_COUNT;
	layer->dsa_prefetch_plan.selected_block_capacity =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_BLOCK_COUNT;
	layer->dsa_prefetch_plan.transport_epoch = 1u;
	layer->dsa_prefetch_plan.selection_ready_event =
		(void *)layer->dsa_selection_event;
	layer->dsa_prefetch_plan.transport_ready_event =
		(void *)layer->dsa_prefetch_event;
	layer->dsa_prefetch_plan.transport_stream = (void *)state->kv_stream;
	if (fp8 != 0u)
	{
		SparkGlm52Pp13BuilderWireDsaFragmentPayload(&layer->dsa_prefetch_plan,
			0u,layer->mla_cache_fp8,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_MLA_LATENT);
		SparkGlm52Pp13BuilderWireDsaFragmentPayload(&layer->dsa_prefetch_plan,
			1u,layer->mla_cache_scale,
			SPARK_GLM52_PP13_BUILDER_FP8_MLA_SCALE_COUNT * sizeof(float),
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_MLA_LATENT |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_FP8_SCALE);
		SparkGlm52Pp13BuilderWireDsaFragmentPayload(&layer->dsa_prefetch_plan,
			2u,layer->key_nope_cache_fp8,
			SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_KEY_NOPE);
		SparkGlm52Pp13BuilderWireDsaFragmentPayload(&layer->dsa_prefetch_plan,
			3u,layer->key_nope_cache_scale,
			SPARK_GLM52_PP13_BUILDER_FP8_KEY_NOPE_SCALE_COUNT * sizeof(float),
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_KEY_NOPE |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_FP8_SCALE);
		SparkGlm52Pp13BuilderWireDsaFragmentPayload(&layer->dsa_prefetch_plan,
			4u,layer->value_cache_fp8,
			SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_VALUE);
		SparkGlm52Pp13BuilderWireDsaFragmentPayload(&layer->dsa_prefetch_plan,
			5u,layer->value_cache_scale,
			SPARK_GLM52_PP13_BUILDER_FP8_VALUE_SCALE_COUNT * sizeof(float),
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_VALUE |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_FP8_SCALE);
	}
	else
	{
		SparkGlm52Pp13BuilderWireDsaFragmentPayload(&layer->dsa_prefetch_plan,
			0u,layer->mla_cache,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS * sizeof(uint16_t),
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_MLA_LATENT);
		SparkGlm52Pp13BuilderWireDsaFragmentPayload(&layer->dsa_prefetch_plan,
			1u,layer->key_nope_cache,
			SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS * sizeof(uint16_t),
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_KEY_NOPE);
		SparkGlm52Pp13BuilderWireDsaFragmentPayload(&layer->dsa_prefetch_plan,
			2u,layer->value_cache,
			SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS * sizeof(uint16_t),
			SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_KV_FRAGMENT_TRANSPORT_PAYLOAD_FLAG_VALUE);
	}
	layer->node.reserved_execution_flags |=
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_DSA_KV_FRAGMENT_TRANSPORT;
	layer->node.dsa_kv_fragment_prefetch_plan = &layer->dsa_prefetch_plan;
	layer->node.dsa_kv_fragment_save_plan = 0;
}

static SparkStatus SparkGlm52Pp13BuilderBindLayerPlans(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_index)
{
	SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreateInfo create_info;
	SparkGlm52ResidentDecodeStageLinearPlan *plans;
	uint32_t bound_fp8_plan_count;
	uint32_t bf16_plan_count;
	uint32_t expected_fp8_plan_count;
	uint32_t plan_index;
	uint32_t plan_count;
	uint32_t mask;
	uint32_t bf16_trunk;
	SparkStatus status;
	memset(&create_info,0,sizeof(create_info));
	bf16_trunk = SparkGlm52Pp13BuilderUsesBf16Trunk(state);
	create_info.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BINDING_ABI_VERSION;
	create_info.maximum_active_sequence_count =
		SparkGlm52Pp13BuilderLayerActiveRowCapacity(state,layer);
	create_info.dense_intermediate_dimension =
		layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER
		? SPARK_GLM52_RESIDENT_DECODE_STAGE_DENSE_INTERMEDIATE_DIMENSION
		: SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_INTERMEDIATE_DIMENSION;
	create_info.expert_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MOE_EXPERT_COUNT;
	mask = SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RAW_ATTENTION_PROJECTIONS;
	if (SparkGlm52Pp13BuilderDsaSourceLayer(layer_index) == layer_index)
		mask |= SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DSA_INDEXER;
	if (layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
		mask |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_GATE |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_UP |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_DOWN;
	else
		mask |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_GATE |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_UP |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_DENSE_DOWN |
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_ROUTER_LOGITS;
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u &&
		layer_index + 1u == state->rank_plan.first_layer_index + state->rank_plan.layer_count)
		mask |= SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_BIND_RESTRICTED_LOGITS;
	create_info.required_plan_mask = mask;
	create_info.workspace_limit_bytes =
		512ull * 1024ull * 1024ull;
	create_info.autotune_warmup_iterations = 1u;
	create_info.autotune_measurement_iterations = 1u;
	create_info.cuda_stream = (void *)state->stream;
	create_info.dense_input_bf16 = layer->post_attention_normalized_hidden;
	create_info.dense_gate_weight_bf16 = layer->dense_gate_weight_bf16;
	create_info.dense_gate_weight_fp8_e4m3 = layer->dense_gate_weight_fp8;
	create_info.dense_gate_weight_scale_inv_f32 = layer->dense_gate_scale;
	create_info.dense_up_weight_bf16 = layer->dense_up_weight_bf16;
	create_info.dense_up_weight_fp8_e4m3 = layer->dense_up_weight_fp8;
	create_info.dense_up_weight_scale_inv_f32 = layer->dense_up_scale;
	create_info.dense_down_weight_bf16 = layer->dense_down_weight_bf16;
	create_info.dense_down_weight_fp8_e4m3 = layer->dense_down_weight_fp8;
	create_info.dense_down_weight_scale_inv_f32 = layer->dense_down_scale;
	create_info.dense_gate_output_bf16 = layer->moe_gate;
	create_info.dense_up_output_bf16 = layer->moe_up;
	create_info.dense_intermediate_bf16 = layer->moe_intermediate;
	create_info.dense_down_output_bf16 = layer->layer_output_hidden;
	create_info.router_input_bf16 = layer->post_attention_normalized_hidden;
	create_info.router_weight_bf16 = layer->router_weight;
	create_info.router_logits_f32 = layer->moe_router_logits;
	create_info.raw_projection_input_bf16 = layer->normalized_hidden;
	create_info.raw_query_a_weight_bf16 = layer->raw_query_a_weight_bf16;
	create_info.raw_query_a_weight_fp8_e4m3 = layer->raw_query_a_weight_fp8;
	create_info.raw_query_a_weight_scale_inv_f32 = layer->raw_query_a_scale;
	create_info.raw_query_a_output_bf16 = layer->raw_query_a;
	create_info.raw_query_b_input_bf16 = layer->raw_query_a_norm;
	create_info.raw_query_b_weight_bf16 = layer->raw_query_b_weight_bf16;
	create_info.raw_query_b_weight_fp8_e4m3 = layer->raw_query_b_weight_fp8;
	create_info.raw_query_b_weight_scale_inv_f32 = layer->raw_query_b_scale;
	create_info.raw_query_b_output_bf16 = layer->raw_query_b;
	create_info.raw_kv_a_weight_bf16 = layer->raw_kv_a_weight_bf16;
	create_info.raw_kv_a_weight_fp8_e4m3 = layer->raw_kv_a_weight_fp8;
	create_info.raw_kv_a_weight_scale_inv_f32 = layer->raw_kv_a_scale;
	create_info.raw_kv_a_output_bf16 = layer->raw_kv_a;
	create_info.raw_kv_b_input_bf16 = layer->raw_kv_a_norm;
	create_info.raw_kv_b_weight_bf16 = layer->raw_kv_b_weight_bf16;
	create_info.raw_kv_b_weight_fp8_e4m3 = layer->raw_kv_b_weight_fp8;
	create_info.raw_kv_b_weight_scale_inv_f32 = layer->raw_kv_b_scale;
	create_info.raw_kv_b_output_bf16 = layer->raw_kv_b;
	create_info.attention_output_input_bf16 = layer->attention_output_latent;
	create_info.attention_output_weight_bf16 = layer->attention_output_weight_bf16;
	create_info.attention_output_weight_fp8_e4m3 = layer->attention_output_weight_fp8;
	create_info.attention_output_weight_scale_inv_f32 = layer->attention_output_scale;
	create_info.attention_output_bf16 = layer->attention_projected_hidden;
	create_info.restricted_logits_input_bf16 = layer->layer_output_hidden;
	create_info.restricted_lm_head_weight_bf16 = layer->restricted_lm_head_weight;
	create_info.restricted_logits_f32 = layer->restricted_logits;
	create_info.dsa_query_input_bf16 = layer->raw_query_a_norm;
	create_info.dsa_query_weight_bf16 = layer->index_query_weight_bf16;
	create_info.dsa_query_weight_fp8_e4m3 = layer->index_query_weight_fp8;
	create_info.dsa_query_weight_scale_inv_f32 = layer->index_query_scale;
	create_info.dsa_query_output_bf16 = layer->query_index_heads;
	create_info.dsa_key_input_bf16 = layer->normalized_hidden;
	create_info.dsa_key_weight_bf16 = layer->index_key_weight_bf16;
	create_info.dsa_key_weight_fp8_e4m3 = layer->index_key_weight_fp8;
	create_info.dsa_key_weight_scale_inv_f32 = layer->index_key_scale;
	create_info.dsa_key_output_bf16 = layer->current_key_index;
	create_info.dsa_weights_input_bf16 = layer->normalized_hidden;
	create_info.dsa_weights_proj_weight_bf16 = layer->index_weights_proj_weight;
	create_info.dsa_weights_output_bf16 = layer->index_head_weights;
	status = SparkGlm52ResidentDecodeStageLinearPlanResidentBindingCreate(
		&layer->linear_binding,
		&create_info);
	if (status != SPARK_STATUS_OK)
		return status;
	plans = SparkGlm52ResidentDecodeStageLinearPlanResidentBindingMutablePlans(
		layer->linear_binding,
		&plan_count);
	if (plans == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (bf16_trunk != 0u)
	{
		bf16_plan_count = 0u;
		for (plan_index = 0u; plan_index < plan_count; ++plan_index)
		{
			if (plans[plan_index].plan_kind ==
				SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_CUBLASLT_BF16_ROW_MAJOR)
				bf16_plan_count += 1u;
			else if (plans[plan_index].plan_kind !=
				SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_UNUSED)
				return SPARK_STATUS_MODULE_NOT_VALIDATED;
		}
		if (bf16_plan_count == 0u)
			return SPARK_STATUS_MODULE_NOT_VALIDATED;
	}
	else
	{
		status = SparkGlm52Sm121RequiredDecodeStageBindBlackwellQuantizedRegularLinearPlans(
			plans,
			plan_count);
		if (status != SPARK_STATUS_OK)
			return status;
		status = SparkGlm52Sm121RequiredDecodeStageBindFp8E4m3LinearPlansScaledGemmBackend(
			plans,
			plan_count,
			&state->fp8_scaled_gemm_backend);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	bound_fp8_plan_count = 0u;
	expected_fp8_plan_count = 0u;
	for (plan_index = 0u; plan_index < plan_count; ++plan_index)
	{
		if (plans[plan_index].plan_kind !=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_TENSOR_CORE_FP8_E4M3_ROW_MAJOR)
			continue;
		expected_fp8_plan_count += 1u;
		if (plans[plan_index].algorithm == &state->fp8_scaled_gemm_backend &&
			plans[plan_index].custom_launch_function != 0)
			bound_fp8_plan_count += 1u;
	}
	if ((bf16_trunk == 0u && expected_fp8_plan_count == 0u) ||
		(bf16_trunk != 0u && expected_fp8_plan_count != 0u) ||
		bound_fp8_plan_count != expected_fp8_plan_count ||
		state->fp8_scaled_gemm_expected_plan_count >
			UINT32_MAX - expected_fp8_plan_count ||
		state->fp8_scaled_gemm_bound_plan_count >
			UINT32_MAX - bound_fp8_plan_count)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	state->fp8_scaled_gemm_expected_plan_count += expected_fp8_plan_count;
	state->fp8_scaled_gemm_bound_plan_count += bound_fp8_plan_count;
	layer->node.linear_plans = plans;
	layer->node.linear_plan_count = plan_count;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderBindFp8Moe(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_index)
{
	SparkGlm52ResidentDecodeStageFp8MoeResidentBindingCreateInfo create_info;
	char path[SPARK_GLM52_PP13_RUNTIME_PACK_PATH_BYTES];
	SparkStatus status;
	if (layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
		return SPARK_STATUS_OK;
	status = SparkGlm52Pp13RuntimeBuildMoePackPath(
		state->configuration.moe_pack_root,
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT,
		layer_index,
		path,
		(uint32_t)sizeof(path));
	if (status != SPARK_STATUS_OK)
		return status;
	memset(&create_info,0,sizeof(create_info));
	create_info.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_BINDING_CREATE_ABI_VERSION;
	create_info.layer_index = layer_index;
	create_info.maximum_active_sequence_count =
		SparkGlm52Pp13BuilderLayerActiveRowCapacity(state,layer);
	create_info.pack_path = path;
	if (layer != &state->mtp_layer && state->shared_moe_workspace != 0)
	{
		create_info.flags =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_BINDING_CREATE_FLAG_EXTERNAL_WORKSPACE;
		create_info.external_workspace = state->shared_moe_workspace;
		create_info.external_workspace_bytes =
			state->shared_moe_workspace_bytes;
	}
	status = SparkGlm52ResidentDecodeStageFp8MoeResidentBindingCreateFromPackFile(
		&layer->fp8_moe_binding,
		&create_info);
	if (status != SPARK_STATUS_OK)
		return status;
	if (layer->fp8_moe_binding.backend_kind !=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_MOE_BACKEND_BUILTIN_FLASHINFER_GROUPED ||
		layer->fp8_moe_binding.plan.launch_function == 0)
	{
		SparkGlm52ResidentDecodeStageFp8MoeResidentBindingDestroy(
			&layer->fp8_moe_binding);
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	}
	if (layer != &state->mtp_layer && state->shared_moe_workspace == 0)
	{
		state->shared_moe_workspace = layer->fp8_moe_binding.workspace;
		state->shared_moe_workspace_bytes =
			layer->fp8_moe_binding.plan.workspace_bytes;
	}
	layer->node.fp8_moe_plan = &layer->fp8_moe_binding.plan;
	layer->fp8_moe_ready = 1u;
	state->moe_bound_layer_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderBindW8lutMoe(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_index)
{
	SparkGlm52ResidentDecodeStageW8lutMoeResidentBindingCreateInfo create_info;
	char path[SPARK_GLM52_PP13_RUNTIME_PACK_PATH_BYTES];
	SparkStatus status;
	if (layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
		return SPARK_STATUS_OK;
	status = SparkGlm52Pp13RuntimeBuildMoePackPath(
		state->configuration.moe_pack_root,
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT,
		layer_index,
		path,
		(uint32_t)sizeof(path));
	if (status != SPARK_STATUS_OK)
		return status;
	memset(&create_info,0,sizeof(create_info));
	create_info.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_BINDING_CREATE_ABI_VERSION;
	create_info.layer_index = layer_index;
	create_info.maximum_active_sequence_count =
		SparkGlm52Pp13BuilderLayerActiveRowCapacity(state,layer);
	create_info.pack_path = path;
	if (state->shared_moe_workspace != 0)
	{
		create_info.flags =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_BINDING_CREATE_FLAG_EXTERNAL_WORKSPACE;
		create_info.external_workspace = state->shared_moe_workspace;
		create_info.external_workspace_bytes =
			state->shared_moe_workspace_bytes;
	}
	status = SparkGlm52ResidentDecodeStageW8lutMoeResidentBindingCreateFromPackFile(
		&layer->w8lut_moe_binding,
		&create_info);
	if (status != SPARK_STATUS_OK)
		return status;
	if (layer->w8lut_moe_binding.backend_kind !=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_W8LUT_MOE_BACKEND_BUILTIN_BF16_WMMA ||
		layer->w8lut_moe_binding.plan.launch_function == 0)
	{
		SparkGlm52ResidentDecodeStageW8lutMoeResidentBindingDestroy(
			&layer->w8lut_moe_binding);
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	}
	if (state->shared_moe_workspace == 0)
	{
		state->shared_moe_workspace = layer->w8lut_moe_binding.workspace;
		state->shared_moe_workspace_bytes =
			layer->w8lut_moe_binding.plan.workspace_bytes;
	}
	layer->node.w8lut_moe_plan = &layer->w8lut_moe_binding.plan;
	layer->w8lut_moe_ready = 1u;
	state->moe_bound_layer_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderBindB12xMoe(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_index)
{
	SparkGlm52ResidentDecodeStageB12xMoeResidentBindingCreateInfo create_info;
	char path[SPARK_GLM52_PP13_RUNTIME_PACK_PATH_BYTES];
	uint32_t external_state;
	SparkStatus status;
	if (layer_index < SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER)
		return SPARK_STATUS_OK;
	status = SparkGlm52Pp13RuntimeBuildMoePackPath(
		state->configuration.moe_pack_root,
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT,
		layer_index,
		path,
		(uint32_t)sizeof(path));
	if (status != SPARK_STATUS_OK)
		return status;
	memset(&create_info,0,sizeof(create_info));
	create_info.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_BINDING_CREATE_ABI_VERSION;
	create_info.layer_index = layer_index;
	create_info.maximum_active_sequence_count =
		SparkGlm52Pp13BuilderLayerActiveRowCapacity(state,layer);
	create_info.pack_path = path;
	external_state = state->shared_b12x_state_cell != 0 ? 1u : 0u;
	if (external_state != 0u)
	{
		create_info.flags =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_BINDING_CREATE_FLAG_EXTERNAL_STATE;
		create_info.external_state_cell = state->shared_b12x_state_cell;
	}
	status = SparkGlm52ResidentDecodeStageB12xMoeResidentBindingCreateFromPackFile(
		&layer->b12x_moe_binding,
		&create_info);
	if (status != SPARK_STATUS_OK)
		return status;
	if (layer->b12x_moe_binding.dispatch_plan.plan_kind !=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_B12X_MOE_DISPATCH_PLAN_KIND_FLASHINFER_B12X ||
		layer->b12x_moe_binding.dispatch_plan.opaque_state !=
			&layer->b12x_moe_binding.plan ||
		layer->b12x_moe_binding.state_cell == 0 ||
		(external_state == 0u &&
		 layer->b12x_moe_binding.owns_state_cell == 0u) ||
		(external_state != 0u &&
		 (layer->b12x_moe_binding.owns_state_cell != 0u ||
		  layer->b12x_moe_binding.state_cell !=
			state->shared_b12x_state_cell)) ||
		(external_state != 0u &&
		 (layer->b12x_moe_binding.plan.maximum_token_count !=
			state->shared_b12x_maximum_token_count ||
		  layer->b12x_moe_binding.plan.recipe.kernel_manifest_hash_low64 !=
			state->shared_b12x_kernel_manifest_hash_low64)))
	{
		SparkGlm52ResidentDecodeStageB12xMoeResidentBindingDestroy(
			&layer->b12x_moe_binding);
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	}
	if (state->shared_b12x_state_cell == 0)
	{
		state->shared_b12x_state_cell = layer->b12x_moe_binding.state_cell;
		state->shared_b12x_maximum_token_count =
			layer->b12x_moe_binding.plan.maximum_token_count;
		state->shared_b12x_kernel_manifest_hash_low64 =
			layer->b12x_moe_binding.plan.recipe.kernel_manifest_hash_low64;
	}
	layer->node.b12x_moe_dispatch_plan =
		&layer->b12x_moe_binding.dispatch_plan;
	layer->b12x_moe_ready = 1u;
	state->moe_bound_layer_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderBindMoe(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderLayer *layer,
	uint32_t layer_index)
{
	if (state == 0 || layer == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->rank_plan.quantization_mode ==
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT)
		return SparkGlm52Pp13BuilderBindFp8Moe(state,layer,layer_index);
	if (state->rank_plan.quantization_mode ==
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT)
		return SparkGlm52Pp13BuilderBindB12xMoe(state,layer,layer_index);
	if (state->rank_plan.quantization_mode ==
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT)
		return SparkGlm52Pp13BuilderBindW8lutMoe(state,layer,layer_index);
	return SPARK_STATUS_INVALID_ARGUMENT;
}

static uint32_t SparkGlm52Pp13BuilderExpectedMoeLayerCount(
	const SparkGlm52Pp13BuilderState *state)
{
	uint32_t first_routed_layer;
	uint32_t range_end;
	uint32_t expected_count;

	if (state == 0)
		return 0u;
	range_end = state->rank_plan.first_layer_index + state->rank_plan.layer_count;
	first_routed_layer = state->rank_plan.first_layer_index >
		SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER
		? state->rank_plan.first_layer_index
		: SPARK_GLM52_RESIDENT_DECODE_STAGE_FIRST_ROUTED_LAYER;
	expected_count = range_end > first_routed_layer
		? range_end - first_routed_layer : 0u;
	if (SparkGlm52Pp13BuilderMtpEnabled(state) &&
		SparkGlm52Pp13BuilderIsFinalRank(state))
		expected_count += 1u;
	return expected_count;
}

static SparkStatus SparkGlm52Pp13BuilderPrepareStageLinearPlanRows(
	SparkGlm52Pp13BuilderState *state,
	uint32_t active_sequence_count)
{
	uint32_t layer_offset;
	uint32_t active_row_capacity;
	SparkStatus status;
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	active_row_capacity =
		SparkGlm52Pp13BuilderLayerActiveRowCapacity(state,&state->layers[0]);
	if (active_sequence_count == 0u ||
		active_sequence_count > active_row_capacity)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (layer_offset = 0u;
		 layer_offset < state->rank_plan.layer_count;
		 ++layer_offset)
	{
		status = SparkGlm52ResidentDecodeStageLinearPlanResidentBindingPrepareActiveRows(
			state->layers[layer_offset].linear_binding,
			active_sequence_count);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderPrepareMtpLinearPlanRows(
	SparkGlm52Pp13BuilderState *state,
	uint32_t active_sequence_count)
{
	if (state == 0 || active_sequence_count == 0u ||
		active_sequence_count > state->rank_plan.execution_row_capacity ||
		!SparkGlm52Pp13BuilderMtpEnabled(state) ||
		!SparkGlm52Pp13BuilderIsFinalRank(state) ||
		state->mtp_layer.linear_binding == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkGlm52ResidentDecodeStageLinearPlanResidentBindingPrepareActiveRows(
		state->mtp_layer.linear_binding,active_sequence_count);
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpFusion(
	SparkGlm52Pp13BuilderState *state,
	const uint32_t *token_ids,
	const uint32_t *positions,
	const void *hidden_bf16,
	uint32_t active_sequence_count,
	cudaStream_t stream)
{
	uint64_t fusion_elements;
	uint32_t fusion_blocks;
	SparkGlm52Pp13BuilderMtpNormInvKernel<<<
		active_sequence_count,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0u,
		stream>>>(
		token_ids,
		positions,
		(const uint16_t *)state->mtp_embedding_weight,
		(const uint16_t *)hidden_bf16,
		state->mtp_norm_inv,
		active_sequence_count,
		SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	if (cudaGetLastError() != cudaSuccess)
		return SPARK_STATUS_IO_ERROR;
	fusion_elements = (uint64_t)active_sequence_count *
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION;
	fusion_blocks = (uint32_t)((fusion_elements +
		SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS);
	SparkGlm52Pp13BuilderMtpFusionKernel<<<
		fusion_blocks,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0u,
		stream>>>(
		token_ids,
		positions,
		(const uint16_t *)state->mtp_embedding_weight,
		(const uint16_t *)hidden_bf16,
		(const uint16_t *)state->mtp_enorm_weight,
		(const uint16_t *)state->mtp_hnorm_weight,
		state->mtp_norm_inv,
		(uint16_t *)state->mtp_eh_input,
		active_sequence_count);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52Pp13BuilderProjectMtpEh(
	SparkGlm52Pp13BuilderState *state,
	uint32_t active_sequence_count)
{
	float alpha;
	float beta;
	cublasStatus_t status;
	alpha = 1.0f;
	beta = 0.0f;
	status = cublasGemmEx(
		state->full_vocab_cublas_handle,
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
		active_sequence_count,
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION,
		&alpha,
		state->mtp_eh_proj_weight,
		CUDA_R_16BF,
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION,
		state->mtp_eh_input,
		CUDA_R_16BF,
		SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION,
		&beta,
		state->mtp_layer.input_hidden,
		CUDA_R_16BF,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
		CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	return SparkGlm52Pp13BuilderCublasStatus(status);
}

static SparkStatus SparkGlm52Pp13BuilderLaunchTensorCoreFullVocabGreedyRows(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52ResidentDecodeStageNodeContext *node_context,
	const void *normalized_hidden_bf16,
	uint32_t *selected_token_ids,
	float *selected_token_scores,
	uint32_t *alternate_token_ids,
	uint32_t row_offset,
	uint32_t row_count,
	cudaStream_t stream)
{
	const uint16_t *input_rows;
	float alpha;
	float beta;
	cublasStatus_t cublas_status;
	if (state == 0 || node_context == 0 || normalized_hidden_bf16 == 0 ||
		selected_token_ids == 0 || selected_token_scores == 0 ||
		stream != state->stream || state->full_vocab_cublas_handle == 0 ||
		state->full_vocab_logits == 0 || row_count == 0u ||
		row_count > state->full_vocab_head_row_capacity ||
		row_offset > state->rank_plan.execution_row_capacity ||
		row_count > state->rank_plan.execution_row_capacity - row_offset ||
		node_context->restricted_lm_head_weight_bf16 == 0 ||
		node_context->restricted_token_ids == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	input_rows = (const uint16_t *)normalized_hidden_bf16 +
		((uint64_t)row_offset * SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
	alpha = 1.0f;
	beta = 0.0f;
	cublas_status = cublasGemmEx(
		state->full_vocab_cublas_handle,CUBLAS_OP_T,CUBLAS_OP_N,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT,
		row_count,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,&alpha,
		node_context->restricted_lm_head_weight_bf16,CUDA_R_16BF,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
		input_rows,CUDA_R_16BF,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,&beta,
		state->full_vocab_logits,CUDA_R_32F,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT,CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	if (cublas_status != CUBLAS_STATUS_SUCCESS)
		return SparkGlm52Pp13BuilderCublasStatus(cublas_status);
	SparkGlm52Pp13BuilderMtpFullVocabTop2Kernel<float><<<
		row_count,SPARK_GLM52_PP13_BUILDER_THREADS,0u,stream>>>(
		state->full_vocab_logits,node_context->restricted_token_ids,
		selected_token_ids + row_offset,selected_token_scores + row_offset,
		alternate_token_ids != 0 ? alternate_token_ids + row_offset : 0,row_count);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52Pp13BuilderLaunchTensorCoreFullVocabGreedy(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52ResidentDecodeStageNodeContext *node_context,
	const void *normalized_hidden_bf16,
	uint32_t *selected_token_ids,
	float *selected_token_scores,
	uint32_t active_sequence_count,
	cudaStream_t stream)
{
	uint32_t row_count;
	uint32_t row_offset;
	SparkStatus status;
	if (state == 0 || active_sequence_count == 0u ||
		active_sequence_count > state->rank_plan.execution_row_capacity ||
		state->full_vocab_head_row_capacity == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SPARK_STATUS_OK;
	for (row_offset = 0u;
		 status == SPARK_STATUS_OK && row_offset < active_sequence_count;
		 row_offset += row_count)
	{
		row_count = active_sequence_count - row_offset;
		if (row_count > state->full_vocab_head_row_capacity)
			row_count = state->full_vocab_head_row_capacity;
		status = SparkGlm52Pp13BuilderLaunchTensorCoreFullVocabGreedyRows(
			state,node_context,normalized_hidden_bf16,selected_token_ids,
			selected_token_scores,0,row_offset,row_count,stream);
	}
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpFullVocabGreedy(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52ResidentDecodeStageNodeContext *node_context,
	uint32_t active_sequence_count,
	uint32_t alternate_required,
	cudaStream_t stream)
{
	SparkStatus status;
	if (state == 0 || node_context == 0 || stream != state->stream ||
		state->mtp_draft_head_ready == 0u || state->full_vocab_logits == 0 ||
		active_sequence_count == 0u ||
		active_sequence_count > state->rank_plan.logical_lane_capacity ||
		active_sequence_count > state->full_vocab_head_row_capacity ||
		(alternate_required != 0u && state->device_decode_token_ids == 0))
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkGlm52Pp13BuilderTargetFinalNormKernel<<<
		active_sequence_count,SPARK_GLM52_PP13_BUILDER_THREADS,0u,stream>>>(
		(const uint16_t *)state->mtp_layer.layer_output_hidden,
		(const uint16_t *)state->mtp_shared_head_norm_weight,
		(uint16_t *)state->mtp_layer.normalized_hidden,active_sequence_count,
		SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
	if (status != SPARK_STATUS_OK)
		return status;
	if (SparkGlm52Pp13BuilderUsesBf16Trunk(state) != 0u)
	{
		status = SparkGlm52Pp13BuilderLaunchTensorCoreFullVocabGreedyRows(
			state,node_context,state->mtp_layer.normalized_hidden,
			(uint32_t *)state->mtp_layer.restricted_selected_token_ids,
			(float *)state->mtp_layer.restricted_selected_token_scores,
			alternate_required != 0u
				? (uint32_t *)state->device_decode_token_ids : 0,
			0u,active_sequence_count,stream);
		return status;
	}
	if (state->mtp_draft_lm_head_weight_fp8 == 0 ||
		state->mtp_draft_lm_head_weight_scale_inv == 0 ||
		state->mtp_draft_head_workspace == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3ActivationWeightLinearScaledGemmBackend(
		&state->fp8_scaled_gemm_backend,
		state->mtp_layer.normalized_hidden,
		state->mtp_draft_lm_head_weight_fp8,
		state->mtp_draft_lm_head_weight_scale_inv,
		state->mtp_draft_head_workspace,
		state->mtp_draft_head_workspace_bytes,
		state->full_vocab_logits,
		active_sequence_count,
		state->rank_plan.logical_lane_capacity,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT,
		SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK,
		0u,
		(void *)stream);
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52Pp13BuilderMtpFullVocabTop2Kernel<uint16_t><<<
		active_sequence_count,SPARK_GLM52_PP13_BUILDER_THREADS,0u,stream>>>(
		(const uint16_t *)state->full_vocab_logits,
		node_context->restricted_token_ids,
		(uint32_t *)state->mtp_layer.restricted_selected_token_ids,
		(float *)state->mtp_layer.restricted_selected_token_scores,
		alternate_required != 0u
			? (uint32_t *)state->device_decode_token_ids : 0,
		active_sequence_count);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52Pp13BuilderLaunchTensorCoreFinalTokenHead(
	const SparkGlm52ResidentDecodeStageExactStageSlicePlan *exact_stage_slice_plan,
	const SparkGlm52ResidentDecodeStageNodeContext *node_context,
	const SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
	uint32_t active_sequence_count,
	void *cuda_stream)
{
	SparkGlm52Pp13BuilderState *state;
	state = exact_stage_slice_plan != 0
		? (SparkGlm52Pp13BuilderState *)exact_stage_slice_plan->final_token_state
		: 0;
	if (state == 0 || exact_stage_slice_plan != &state->exact_plan ||
		pipeline_slot == 0 || cuda_stream != (void *)state->stream ||
		pipeline_slot->normalized_hidden_bf16 == 0 ||
		pipeline_slot->restricted_selected_token_ids == 0 ||
		pipeline_slot->restricted_selected_token_scores == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkGlm52Pp13BuilderLaunchTensorCoreFullVocabGreedy(
		state,node_context,pipeline_slot->normalized_hidden_bf16,
		pipeline_slot->restricted_selected_token_ids,
		pipeline_slot->restricted_selected_token_scores,
		active_sequence_count,(cudaStream_t)cuda_stream);
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpMetadata(
	SparkGlm52Pp13BuilderState *state,
	const uint32_t *base_positions,
	uint32_t draft_index,
	uint32_t active_sequence_count,
	cudaStream_t stream)
{
	uint32_t blocks;
	blocks = (active_sequence_count + SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS;
	SparkGlm52Pp13BuilderMtpMetadataKernel<<<
		blocks,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0u,
		stream>>>(
		base_positions,
		(const uint32_t *)state->mtp_layer.slot.block_table,
		(uint32_t *)state->mtp_layer.positions,
		(uint32_t *)state->mtp_layer.slot_mapping,
		(uint32_t *)state->mtp_layer.context_lengths,
		(uint32_t *)state->mtp_layer.first_block_token_offsets,
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS,
		draft_index,
		active_sequence_count);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpLayer(
	SparkGlm52Pp13BuilderState *state,
	const uint32_t *token_ids,
	const uint32_t *base_positions,
	const void *hidden_bf16,
	uint32_t draft_index,
	uint32_t active_sequence_count,
	cudaStream_t stream)
{
	SparkStatus status;
	status = SparkGlm52Pp13BuilderLaunchMtpMetadata(
		state,base_positions,draft_index,active_sequence_count,stream);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderMarkMtpGpuProfile(
			state,draft_index,
			SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_METADATA,stream);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus(
			"mtp_metadata",SPARK_GLM52_MODEL_MTP_LAYER_INDEX,status);
	status = SparkGlm52Pp13BuilderLaunchMtpFusion(
		state,token_ids,(const uint32_t *)state->mtp_layer.positions,
		hidden_bf16,active_sequence_count,stream);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderMarkMtpGpuProfile(
			state,draft_index,
			SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_FUSION,stream);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus(
			"mtp_fusion",SPARK_GLM52_MODEL_MTP_LAYER_INDEX,status);
	status = SparkGlm52Pp13BuilderProjectMtpEh(state,active_sequence_count);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderMarkMtpGpuProfile(
			state,draft_index,
			SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_EH_PROJECTION,stream);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus(
			"mtp_eh_projection",SPARK_GLM52_MODEL_MTP_LAYER_INDEX,status);
	status = SparkGlm52Sm121RequiredDecodeStageLaunch(
		&state->mtp_layer.node,&state->mtp_layer.slot,0u,
		active_sequence_count,0,stream);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderMarkMtpGpuProfile(
			state,draft_index,
			SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_REQUIRED_LAYER,stream);
	return SparkGlm52Pp13BuilderReportStatus(
		"mtp_required_layer",SPARK_GLM52_MODEL_MTP_LAYER_INDEX,status);
}

static SparkStatus SparkGlm52Pp13BuilderStoreMtpDraft(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52ResidentDecodeStagePipelineSlot *base_slot,
	const uint32_t *token_ids,
	uint32_t draft_index,
	uint32_t active_sequence_count,
	cudaStream_t stream)
{
	uint64_t elements;
	uint32_t blocks;
	elements = (uint64_t)active_sequence_count *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	blocks = (uint32_t)((elements + SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS);
	SparkGlm52Pp13BuilderMtpStoreKernel<<<
		blocks,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0u,
		stream>>>(
		(const uint16_t *)state->mtp_layer.layer_output_hidden,
		token_ids,
		(uint16_t *)base_slot->mtp_draft_hidden_bf16,
		base_slot->mtp_draft_token_ids,
		draft_index,
		active_sequence_count);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpDraftPlan(
	const SparkGlm52ResidentDecodeStageMtpDraftPlan *plan,
	const SparkGlm52ResidentDecodeStageNodeContext *node_context,
	const SparkGlm52ResidentDecodeStagePipelineSlot *base_slot,
	uint32_t active_sequence_count,
	void *cuda_stream)
{
	SparkGlm52Pp13BuilderState *state;
	const uint32_t *token_ids;
	const void *hidden_bf16;
	uint32_t draft_token_count;
	uint32_t draft_index;
	uint32_t lane_index;
	uint32_t alternate_candidate_index;
	uint32_t alternate_required;
	uint32_t primary_candidate_index;
	SparkStatus status;
	state = plan != 0 ? (SparkGlm52Pp13BuilderState *)plan->opaque_state : 0;
	if (state == 0 || state->mtp_ready == 0u ||
		state->mtp_draft_head_ready == 0u || node_context == 0 ||
		base_slot == 0 || cuda_stream != (void *)state->stream ||
		active_sequence_count == 0u ||
		active_sequence_count > state->rank_plan.logical_lane_capacity ||
		base_slot->positions == 0 || base_slot->layer_output_hidden_bf16 == 0 ||
		base_slot->normalized_hidden_bf16 == 0 ||
		base_slot->restricted_selected_token_ids == 0 ||
		base_slot->mtp_draft_hidden_bf16 == 0 ||
		base_slot->mtp_draft_token_ids == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	draft_token_count = 0u;
	for (lane_index = 0u; lane_index < active_sequence_count; ++lane_index)
	{
		if (state->host_mtp_draft_budgets[lane_index] != 0u &&
			state->host_mtp_draft_budgets[lane_index] !=
				SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT)
			return SPARK_STATUS_INVALID_ARGUMENT;
		if (state->host_mtp_draft_budgets[lane_index] > draft_token_count)
			draft_token_count = state->host_mtp_draft_budgets[lane_index];
	}
	if (draft_token_count != SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13BuilderPrepareMtpLinearPlanRows(
		state,active_sequence_count);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus(
			"mtp_prepare_linear_rows",SPARK_GLM52_MODEL_MTP_LAYER_INDEX,status);
	for (draft_index = 0u;
		 draft_index < SPARK_GLM52_MODEL_MTP_TREE_EXECUTION_STEP_COUNT;
		 ++draft_index)
	{
		alternate_required = draft_index == 0u ? 0u : 1u;
		primary_candidate_index = draft_index == 0u
			? SPARK_GLM52_MODEL_MTP_TREE_DEPTH1_PRIMARY_INDEX
			: draft_index == 1u
				? SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_PRIMARY_INDEX
				: SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_PRIMARY_INDEX;
		alternate_candidate_index = draft_index == 1u
			? SPARK_GLM52_MODEL_MTP_TREE_DEPTH2_ALTERNATE_INDEX
			: SPARK_GLM52_MODEL_MTP_TREE_DEPTH3_ALTERNATE_INDEX;
		token_ids = draft_index == 0u
			? base_slot->restricted_selected_token_ids
			: (const uint32_t *)state->mtp_layer.restricted_selected_token_ids;
		hidden_bf16 = draft_index == 0u
			? base_slot->normalized_hidden_bf16
			: state->mtp_layer.layer_output_hidden;
		status = SparkGlm52Pp13BuilderMarkMtpGpuProfile(
			state,draft_index,
			SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_START,
			(cudaStream_t)cuda_stream);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderLaunchMtpLayer(
			state,token_ids,base_slot->positions,hidden_bf16,draft_index,
			active_sequence_count,(cudaStream_t)cuda_stream);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderLaunchMtpFullVocabGreedy(
				state,node_context,active_sequence_count,alternate_required,
				(cudaStream_t)cuda_stream);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderMarkMtpGpuProfile(
				state,draft_index,
				SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_VOCAB_HEAD,
				(cudaStream_t)cuda_stream);
		if (status != SPARK_STATUS_OK)
			return SparkGlm52Pp13BuilderReportStatus(
				"mtp_full_vocab_greedy",SPARK_GLM52_MODEL_MTP_LAYER_INDEX,status);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderStoreMtpDraft(
				state,base_slot,
				(const uint32_t *)state->mtp_layer.restricted_selected_token_ids,
				primary_candidate_index,active_sequence_count,
				(cudaStream_t)cuda_stream);
		if (status == SPARK_STATUS_OK && alternate_required != 0u)
			status = SparkGlm52Pp13BuilderStoreMtpDraft(
				state,base_slot,
				(const uint32_t *)state->device_decode_token_ids,
				alternate_candidate_index,active_sequence_count,
				(cudaStream_t)cuda_stream);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderMarkMtpGpuProfile(
				state,draft_index,
				SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_STORE,
				(cudaStream_t)cuda_stream);
		if (status != SPARK_STATUS_OK)
			return SparkGlm52Pp13BuilderReportStatus(
				"mtp_store_draft",SPARK_GLM52_MODEL_MTP_LAYER_INDEX,status);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLoadMtpWeights(
	SparkGlm52Pp13BuilderState *state)
{
	SparkStatus status;
	status = SparkGlm52Pp13BuilderLoadTensor(
		state,SPARK_GLM52_PP13_BUILDER_MTP_EMBEDDING_TENSOR,"BF16",
		sizeof(uint16_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
		&state->mtp_embedding_weight);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadLayerTensor(
			state,SPARK_GLM52_MODEL_MTP_LAYER_INDEX,"enorm.weight","BF16",
			sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
			1u,&state->mtp_enorm_weight);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadLayerTensor(
			state,SPARK_GLM52_MODEL_MTP_LAYER_INDEX,"hnorm.weight","BF16",
			sizeof(uint16_t),1u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
			1u,&state->mtp_hnorm_weight);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadLayerTensor(
			state,SPARK_GLM52_MODEL_MTP_LAYER_INDEX,"eh_proj.weight","BF16",
			sizeof(uint16_t),2u,SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
			SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION,
			&state->mtp_eh_proj_weight);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadLayerTensor(
			state,SPARK_GLM52_MODEL_MTP_LAYER_INDEX,"shared_head.norm.weight",
			"BF16",sizeof(uint16_t),1u,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,1u,
			&state->mtp_shared_head_norm_weight);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->mtp_eh_input,
			(uint64_t)state->rank_plan.execution_row_capacity *
			SPARK_GLM52_PP13_BUILDER_MTP_EH_INPUT_DIMENSION * sizeof(uint16_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,(void **)&state->mtp_norm_inv,
			(uint64_t)state->rank_plan.execution_row_capacity *
			SPARK_GLM52_PP13_BUILDER_MTP_NORM_COUNT_PER_LANE * sizeof(float));
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeTensorCoreFinalTokenHead(
	SparkGlm52Pp13BuilderState *state)
{
	SparkStatus status;
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (!SparkGlm52Pp13BuilderIsFinalRank(state))
		return SPARK_STATUS_OK;
	state->full_vocab_head_row_capacity = state->rank_plan.execution_row_capacity;
	if (state->full_vocab_head_row_capacity >
		SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET)
		state->full_vocab_head_row_capacity =
			SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET;
	if (state->full_vocab_head_row_capacity == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13BuilderCudaAlloc(
		state,(void **)&state->full_vocab_logits,
		(uint64_t)state->full_vocab_head_row_capacity *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT * sizeof(float));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCublasStatus(
			cublasCreate(&state->full_vocab_cublas_handle));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCublasStatus(
			cublasSetStream(state->full_vocab_cublas_handle,state->stream));
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus(
			"initialize_tensor_core_final_token_head",
			state->rank_plan.first_layer_index + state->rank_plan.layer_count - 1u,
			status);
	state->exact_plan.final_token_launch_function =
		(void *)SparkGlm52Pp13BuilderLaunchTensorCoreFinalTokenHead;
	state->exact_plan.final_token_state = state;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeMtpDraftHead(
	SparkGlm52Pp13BuilderState *state,
	const void *lm_head_weight_bf16)
{
	dim3 grid;
	uint64_t weight_bytes,scale_count,scale_bytes;
	SparkStatus status;
	if (state == 0 || lm_head_weight_bf16 == 0 || state->mtp_ready == 0u ||
		state->rank_plan.logical_lane_capacity == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (SparkGlm52Pp13BuilderUsesBf16Trunk(state) != 0u)
	{
		state->mtp_draft_head_ready = 1u;
		return SPARK_STATUS_OK;
	}
	if (state->fp8_scaled_gemm_backend.launch_function == 0 ||
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT %
			SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK != 0u ||
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION %
			SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	weight_bytes = (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	scale_count =
		(SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT /
		 SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK) *
		(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
		 SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK);
	scale_bytes = scale_count * sizeof(float);
	state->mtp_draft_head_workspace_bytes =
		SparkGlm52Sm121RequiredDecodeStageCalculateFp8E4m3ActivationLinearBackendWorkspaceBytes(
			state->rank_plan.logical_lane_capacity,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
			SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK,
			state->fp8_scaled_gemm_backend.required_workspace_bytes);
	if (state->mtp_draft_head_workspace_bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13BuilderCudaAlloc(
		state,(void **)&state->mtp_draft_lm_head_weight_fp8,weight_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,(void **)&state->mtp_draft_lm_head_weight_scale_inv,scale_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->mtp_draft_head_workspace,
			state->mtp_draft_head_workspace_bytes);
	if (status != SPARK_STATUS_OK)
		return status;
	grid = dim3(
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT /
			SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
			SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK,
		1u);
	SparkGlm52Pp13BuilderQuantizeMtpDraftHeadKernel<<<
		grid,SPARK_GLM52_PP13_BUILDER_THREADS,0u,state->stream>>>(
		(const uint16_t *)lm_head_weight_bf16,
		state->mtp_draft_lm_head_weight_fp8,
		state->mtp_draft_lm_head_weight_scale_inv,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION);
	status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaStreamSynchronize(state->stream));
	if (status == SPARK_STATUS_OK)
		state->mtp_draft_head_ready = 1u;
	return status;
}

static void SparkGlm52Pp13BuilderConfigureMtpLayer(
	SparkGlm52Pp13BuilderState *state)
{
	SparkGlm52ResidentDecodeStageNodeContext *node;
	uint32_t clear_flags;
	node = &state->mtp_layer.node;
	node->cache_token_capacity = state->configuration.kv_pool_token_capacity;
	node->kv_storage_token_capacity =
		state->configuration.kv_pool_token_capacity;
	node->kv_block_count = state->configuration.kv_pool_token_capacity /
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	node->max_blocks_per_sequence =
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	node->dsa_candidate_capacity =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
	node->sparse_index_mode =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_COPY_CONTEXT_PREFIX;
	node->dsa_indexshare_source_layer_index = 0u;
	node->dsa_indexshare_group_end_layer_exclusive = 0u;
	node->dsa_indexshare_selected_token_count = 0u;
	node->dsa_indexshare_layer_count = 0u;
	node->enable_cuda_graph_replay =
		getenv("SPARKPIPE_MTP_LAYER_ENABLE_GRAPH") != 0 ? 1u : 0u;
	node->bulk_prefill_plan = 0;
	state->mtp_layer.slot.block_table = state->device_physical_block_indices;
	clear_flags =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_GRAPH_REPLAY |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_STAGE_SLICE_PLAN |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_INPUT |
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_HIDDEN_TRANSPORT_OUTPUT;
	node->reserved_execution_flags &= ~clear_flags;
	node->reserved_execution_flags |=
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_OUTPUT_HIDDEN_ONLY;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeMtp(
	SparkGlm52Pp13BuilderState *state)
{
	SparkStatus status;
	if (!SparkGlm52Pp13BuilderMtpEnabled(state) ||
		!SparkGlm52Pp13BuilderIsFinalRank(state))
		return SPARK_STATUS_OK;
	state->mtp_gpu_profile_enabled =
		getenv("SPARKPIPE_MTP_GPU_PROFILE") != 0 ? 1u : 0u;
	state->mtp_cycle_profile_enabled =
		getenv("SPARKPIPE_MTP_CYCLE_PROFILE") != 0 ? 1u : 0u;
	status = SparkGlm52Pp13BuilderAllocateLayerBuffers(
		state,&state->mtp_layer,UINT32_MAX,
		state->rank_plan.execution_row_capacity,
		state->rank_plan.logical_lane_capacity,
		state->configuration.kv_pool_token_capacity,
		state->configuration.kv_pool_token_capacity,
		0u);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadLayerWeights(
			state,&state->mtp_layer,SPARK_GLM52_MODEL_MTP_LAYER_INDEX);
	if (status == SPARK_STATUS_OK)
	{
		SparkGlm52Pp13BuilderWireLayer(
			state,&state->mtp_layer,SPARK_GLM52_MODEL_MTP_LAYER_INDEX);
		SparkGlm52Pp13BuilderConfigureMtpLayer(state);
	}
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadMtpWeights(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->mtp_previous_target_hidden,
			(uint64_t)state->rank_plan.execution_row_capacity *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_BF16_BYTES);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,(void **)&state->mtp_base_positions,
			(uint64_t)state->rank_plan.execution_row_capacity * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,(void **)&state->mtp_prefill_block_table,
			(uint64_t)state->rank_plan.execution_row_capacity *
			SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE *
			sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->mtp_previous_target_hidden_store,
			(uint64_t)state->configuration.maximum_resident_sequence_count *
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_BF16_BYTES);
	if (status == SPARK_STATUS_OK && state->mtp_gpu_profile_enabled != 0u)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,(void **)&state->mtp_gpu_profile_cycles,
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT *
			SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_PHASE_COUNT * sizeof(uint64_t));
	if (status == SPARK_STATUS_OK && state->mtp_gpu_profile_enabled != 0u)
		status = SparkGlm52Pp13BuilderCudaZero(
			state->mtp_gpu_profile_cycles,
			(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT *
			SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_PHASE_COUNT * sizeof(uint64_t));
	if (status == SPARK_STATUS_OK)
	{
		state->host_mtp_previous_sequence_ids = (uint64_t *)calloc(
			state->configuration.maximum_resident_sequence_count,sizeof(uint64_t));
		state->host_mtp_previous_positions = (uint64_t *)calloc(
			state->configuration.maximum_resident_sequence_count,sizeof(uint64_t));
		state->host_mtp_previous_valid = (uint8_t *)calloc(
			state->configuration.maximum_resident_sequence_count,sizeof(uint8_t));
		if (state->host_mtp_previous_sequence_ids == 0 ||
			state->host_mtp_previous_positions == 0 ||
			state->host_mtp_previous_valid == 0)
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderBindMoe(
			state,&state->mtp_layer,SPARK_GLM52_MODEL_MTP_LAYER_INDEX);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderBindLayerPlans(
			state,&state->mtp_layer,SPARK_GLM52_MODEL_MTP_LAYER_INDEX);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Sm121RequiredDecodeStageInitialize(&state->mtp_layer.node);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus(
			"initialize_mtp",SPARK_GLM52_MODEL_MTP_LAYER_INDEX,status);
	memset(&state->mtp_draft_plan,0,sizeof(state->mtp_draft_plan));
	state->mtp_draft_plan.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_PLAN_ABI_VERSION;
	state->mtp_draft_plan.restricted_vocab_count =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT;
	state->mtp_draft_plan.hidden_dimension =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	state->mtp_draft_plan.draft_token_count =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT;
	state->mtp_draft_plan.weight_format =
		SparkGlm52Pp13BuilderUsesBf16Trunk(state) != 0u
		? SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_BF16
		: SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_WEIGHT_FORMAT_FP8_E4M3;
	state->mtp_draft_plan.launch_function =
		(void *)SparkGlm52Pp13BuilderLaunchMtpDraftPlan;
	state->mtp_draft_plan.opaque_state = state;
	state->mtp_draft_plan.validated_maximum_latency_ns = 0u;
	state->mtp_ready = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderBuildLayer(
	SparkGlm52Pp13BuilderState *state,
	uint32_t layer_offset)
{
	SparkGlm52Pp13BuilderLayer *layer;
	uint32_t layer_index;
	SparkStatus status;
	layer = &state->layers[layer_offset];
	layer_index = state->rank_plan.first_layer_index + layer_offset;
	status = SparkGlm52Pp13BuilderAllocateLayerBuffers(
		state,
		layer,
		layer_offset,
		SparkGlm52Pp13BuilderLayerActiveRowCapacity(state,layer),
		state->rank_plan.logical_lane_capacity,
		state->configuration.kv_pool_token_capacity,
		state->cache_storage_token_capacity,
		SparkGlm52Pp13BuilderDsaSourceLayer(layer_index) == layer_index ? 1u : 0u);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus("allocate_layer_buffers",layer_index,status);
	if (cudaEventCreateWithFlags(&layer->dsa_selection_event,
		cudaEventDisableTiming) != cudaSuccess ||
		cudaEventCreateWithFlags(&layer->dsa_prefetch_event,
		cudaEventDisableTiming) != cudaSuccess)
		return SparkGlm52Pp13BuilderReportStatus(
			"create_dsa_prefetch_events",
			layer_index,
			SPARK_STATUS_IO_ERROR);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLoadLayerWeights(state,layer,layer_index);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus("load_layer_weights",layer_index,status);
	SparkGlm52Pp13BuilderWireLayer(state,layer,layer_index);
	SparkGlm52Pp13BuilderWireDsaFragmentPrefetch(state,layer);
	if (layer_offset > 0u)
		layer->slot.input_hidden_bf16 =
			state->layers[layer_offset - 1u].layer_output_hidden;
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderBindMoe(state,layer,layer_index);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus("bind_moe",layer_index,status);
	if (status == SPARK_STATUS_OK &&
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u &&
		layer_offset + 1u == state->rank_plan.layer_count)
	{
		status = SparkGlm52Pp13BuilderLoadTensor(
			state,
			"model.norm.weight",
			"BF16",
			sizeof(uint16_t),
			1u,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
			1u,
			&layer->final_norm_weight);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderLoadLmHeadRestricted(
				state,
				&layer->restricted_lm_head_weight);
		if (status == SPARK_STATUS_OK && SparkGlm52Pp13BuilderMtpEnabled(state))
			status = SparkGlm52Pp13BuilderInitializeMtpDraftHead(
				state,layer->restricted_lm_head_weight);
		layer->node.final_norm_weight_bf16 = layer->final_norm_weight;
		layer->node.restricted_lm_head_weight_bf16 =
			layer->restricted_lm_head_weight;
	}
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus("load_final_outputs",layer_index,status);
	if (SparkGlm52Pp13BuilderMtpEnabled(state) &&
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u &&
		layer_offset + 1u == state->rank_plan.layer_count)
	{
		if (state->mtp_ready == 0u || state->mtp_draft_head_ready == 0u)
			return SparkGlm52Pp13BuilderReportStatus(
				"mtp_not_ready",layer_index,SPARK_STATUS_MODULE_NOT_VALIDATED);
		layer->node.mtp_draft_plan = &state->mtp_draft_plan;
	}
	status = SparkGlm52Pp13BuilderBindLayerPlans(state,layer,layer_index);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus("bind_layer_plans",layer_index,status);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Sm121RequiredDecodeStageInitialize(&layer->node);
	if (status != SPARK_STATUS_OK)
		return SparkGlm52Pp13BuilderReportStatus("required_stage_initialize",layer_index,status);
	if (status == SPARK_STATUS_OK)
	{
		state->layer_pointers[layer_offset] = &layer->node;
	}
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeExactPlan(
	SparkGlm52Pp13BuilderState *state)
{
	uint64_t candidates;
	uint64_t workspace_bytes;
	uint32_t active_row_capacity;
	uint32_t batch_bucket;
	uint32_t capability_flags;
	SparkStatus status;
	active_row_capacity =
		SparkGlm52Pp13BuilderLayerActiveRowCapacity(state,&state->layers[0]);
	batch_bucket =
		SparkGlm52StagePlanSelectBatchBucketValue(
			state->rank_plan.logical_lane_capacity);
	candidates =
		(uint64_t)active_row_capacity *
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_FINAL_EPILOGUE_CANDIDATE_GROUP_COUNT;
	workspace_bytes =
		(candidates * sizeof(float)) + (candidates * sizeof(uint32_t)) + 15u;
	status = SparkGlm52Pp13BuilderCudaAlloc(
		state,
		&state->final_epilogue_workspace,
		workspace_bytes);
	if (status != SPARK_STATUS_OK)
		return status;
	if (cudaStreamCreate(&state->query_stream) != cudaSuccess ||
		cudaStreamCreate(&state->kv_stream) != cudaSuccess ||
		cudaEventCreate(&state->branch_ready_event) != cudaSuccess ||
		cudaEventCreate(&state->query_event) != cudaSuccess ||
		cudaEventCreate(&state->kv_event) != cudaSuccess)
		return SPARK_STATUS_IO_ERROR;
	capability_flags =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_PRODUCTION_PP13_CAPABILITIES;
	if (active_row_capacity == state->rank_plan.execution_row_capacity)
		capability_flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_CAPABILITY_LAYER_MAJOR_SPECULATIVE_VERIFY;
	memset(&state->exact_plan,0,sizeof(state->exact_plan));
	state->exact_plan.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXACT_STAGE_SLICE_PLAN_ABI_VERSION;
	state->exact_plan.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_EXACT_STAGE_SLICE_PLAN_DESCRIPTOR_BYTES;
	state->exact_plan.stage_index = state->rank_plan.rank_index;
	state->exact_plan.first_layer_index = state->rank_plan.first_layer_index;
	state->exact_plan.layer_count = state->rank_plan.layer_count;
	state->exact_plan.batch_bucket = batch_bucket;
	state->exact_plan.maximum_active_sequence_count =
		active_row_capacity;
	state->exact_plan.logical_lane_capacity =
		state->rank_plan.logical_lane_capacity;
	state->exact_plan.maximum_speculative_rows_per_lane =
		state->rank_plan.maximum_speculative_rows_per_lane;
	state->exact_plan.final_token_candidate_row_capacity =
		active_row_capacity;
	state->exact_plan.capability_flags = capability_flags;
	state->exact_plan.query_branch_stream = (void *)state->query_stream;
	state->exact_plan.kv_branch_stream = (void *)state->kv_stream;
	state->exact_plan.branch_ready_event = (void *)state->branch_ready_event;
	state->exact_plan.query_branch_event = (void *)state->query_event;
	state->exact_plan.kv_branch_event = (void *)state->kv_event;
	state->exact_plan.workspace = state->final_epilogue_workspace;
	state->exact_plan.workspace_bytes = workspace_bytes;
	state->exact_plan.validated_maximum_latency_ns = 0u;
	memset(&state->stage_slice_plan,0,sizeof(state->stage_slice_plan));
	state->stage_slice_plan.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_SLICE_PLAN_ABI_VERSION;
	state->stage_slice_plan.maximum_active_sequence_count =
		active_row_capacity;
	state->stage_slice_plan.maximum_layer_count = state->rank_plan.layer_count;
	state->stage_slice_plan.capability_flags = capability_flags;
	state->stage_slice_plan.opaque_state = &state->exact_plan;
	state->stage_slice_plan.workspace = state->final_epilogue_workspace;
	state->stage_slice_plan.workspace_bytes = workspace_bytes;
	state->stage_slice_plan.validated_maximum_latency_ns = 0u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeFp8ScaledGemm(
	SparkGlm52Pp13BuilderState *state)
{
	uint64_t workspace_bytes;
	SparkStatus status;
	if (SparkGlm52Pp13BuilderUsesBf16Trunk(state) != 0u)
		return SPARK_STATUS_OK;
	workspace_bytes =
		SparkGlm52Sm121RequiredDecodeStageCalculateBuiltinFp8ScaledGemmWorkspaceBytes();
	if (workspace_bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13BuilderCudaAlloc(
		state,
		&state->fp8_scaled_gemm_workspace,
		workspace_bytes);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52Sm121RequiredDecodeStageInitializeBuiltinFp8ScaledGemmBackend(
		&state->fp8_scaled_gemm_state,
		state->fp8_scaled_gemm_workspace,
		workspace_bytes,
		&state->fp8_scaled_gemm_backend);
}

static SparkStatus SparkGlm52Pp13BuilderInitializeRank0InputBuffers(
	SparkGlm52Pp13BuilderState *state)
{
	uint64_t max_active;
	uint64_t logical_lane_capacity;
	uint32_t rank_has_previous;
	SparkStatus status;
	max_active = state->rank_plan.execution_row_capacity;
	logical_lane_capacity = state->rank_plan.logical_lane_capacity;
	rank_has_previous =
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u;
	status = SPARK_STATUS_OK;
	if (rank_has_previous == 0u)
		status = SparkGlm52Pp13BuilderLoadEmbedding(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->device_decode_positions,
			max_active * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->device_decode_token_ids,
			max_active * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->device_mtp_draft_token_budgets,
			logical_lane_capacity * sizeof(uint32_t));
	if (status != SPARK_STATUS_OK)
		return status;
	state->host_decode_positions =
		(uint32_t *)malloc((size_t)(max_active * sizeof(uint32_t)));
	state->host_decode_token_ids =
		(uint32_t *)malloc((size_t)(max_active * sizeof(uint32_t)));
	state->host_decode_result_token_ids =
		(uint32_t *)malloc((size_t)(max_active * sizeof(uint32_t)));
	state->host_mtp_committed_token_ids =
		(uint32_t *)malloc(
			(size_t)(logical_lane_capacity *
				SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT *
				sizeof(uint32_t)));
	if (state->host_decode_positions == 0 ||
		state->host_decode_token_ids == 0 ||
		state->host_decode_result_token_ids == 0 ||
		state->host_mtp_committed_token_ids == 0)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderInitializeSharedBuffers(
	SparkGlm52Pp13BuilderState *state)
{
	uint64_t max_active;
	uint64_t sideband_bytes;
	uint64_t sideband_bytes_per_sequence;
	uint64_t selected_indices_bytes;
	uint64_t selected_block_bytes;
	uint64_t kv_entries;
	uint64_t physical_block_count;
	uint64_t logical_block_capacity;
	uint32_t directory_capacity;
	uint32_t shadow_index;
	uint64_t shadow_slot_capacity;
	uint64_t shadow_token_capacity;
	uint64_t storage_token_capacity;
	SparkStatus status;
	max_active = state->rank_plan.execution_row_capacity;
	state->cache_storage_token_capacity =
		state->configuration.kv_pool_token_capacity;
	if (SparkGlm52Pp13BuilderMtpEnabled(state) != 0u)
	{
		shadow_slot_capacity =
			(uint64_t)SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT *
			state->rank_plan.logical_lane_capacity;
		shadow_token_capacity =
			shadow_slot_capacity *
			SPARK_GLM52_MODEL_MTP_TREE_SHADOW_TOKEN_COUNT;
		storage_token_capacity =
			(uint64_t)state->configuration.kv_pool_token_capacity +
			shadow_token_capacity;
		if (shadow_slot_capacity == 0u ||
			shadow_slot_capacity > UINT32_MAX ||
			storage_token_capacity > UINT32_MAX)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		state->mtp_shadow_slot_capacity = (uint32_t)shadow_slot_capacity;
		state->cache_storage_token_capacity =
			(uint32_t)storage_token_capacity;
		state->mtp_shadow_free_indices =
			(uint32_t *)malloc(
				(size_t)shadow_slot_capacity * sizeof(uint32_t));
		if (state->mtp_shadow_free_indices == 0)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		for (shadow_index = 0u;
			 shadow_index < state->mtp_shadow_slot_capacity;
			 ++shadow_index)
			state->mtp_shadow_free_indices[shadow_index] = shadow_index;
		state->mtp_shadow_free_count = state->mtp_shadow_slot_capacity;
		status = SparkGlm52Pp13BuilderCudaHostPinnedAlloc(
			state,
			(void **)&state->host_mtp_tree_shadow_slot_mapping,
			max_active * sizeof(uint32_t));
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderCudaAlloc(
				state,
				(void **)&state->device_mtp_tree_shadow_slot_mapping,
				max_active * sizeof(uint32_t));
		if (status != SPARK_STATUS_OK)
			return status;
	}
	selected_indices_bytes =
		(uint64_t)state->dsa_cache_layer_count *
		max_active *
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT *
		sizeof(uint32_t);
	selected_block_bytes =
		(uint64_t)state->dsa_cache_layer_count *
		max_active *
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_BLOCK_COUNT *
		sizeof(uint32_t);
	sideband_bytes_per_sequence =
		SPARK_GLM52_PP13_RUNTIME_INDEXSHARE_SIDEBAND_BYTES_PER_SEQUENCE;
	if (SparkGlm52Pp13BuilderDsparkEnabled(state))
		sideband_bytes_per_sequence +=
			SPARK_GLM52_PP13_RUNTIME_DSPARK_TAP_SIDEBAND_BYTES_PER_SEQUENCE;
	sideband_bytes = max_active * sideband_bytes_per_sequence;
	kv_entries = max_active * SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	physical_block_count = state->configuration.kv_pool_token_capacity /
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
	logical_block_capacity = physical_block_count;
	if ((state->configuration.flags &
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_NVME_KV) != 0u)
		logical_block_capacity = state->configuration.kv_nvme_block_capacity;
	directory_capacity = 1u;
	while ((uint64_t)directory_capacity < logical_block_capacity * 2u)
	{
		if (directory_capacity > UINT32_MAX / 2u)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		directory_capacity *= 2u;
	}
	status = SparkGlm52Pp13BuilderCudaAlloc(
		state,&state->selected_token_indices_by_layer,selected_indices_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,&state->selected_block_indices_by_layer,selected_block_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,
			&state->selected_block_counts_by_layer,
			(uint64_t)state->dsa_cache_layer_count *
			max_active *
			sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,
			&state->dsa_selection_epoch_by_layer,
			(uint64_t)state->dsa_cache_layer_count *
			max_active *
			sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
	{
		if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
			status = SparkGlm52Pp13BuilderCudaHostMappedAlloc(state,&state->input_sideband,sideband_bytes);
		else
			status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->input_sideband,sideband_bytes);
	}
	if (status == SPARK_STATUS_OK)
	{
		if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
			status = SparkGlm52Pp13BuilderCudaHostMappedAlloc(state,&state->output_sideband,sideband_bytes);
		else
			status = SparkGlm52Pp13BuilderCudaAlloc(state,&state->output_sideband,sideband_bytes);
	}
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,
			(void **)&state->device_physical_block_indices,
			kv_entries * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,
			(void **)&state->device_lane_physical_block_counts,
			max_active * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaAlloc(
			state,
			(void **)&state->device_mtp_request_slot_indices,
			max_active * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaHostPinnedAlloc(
			state,
			(void **)&state->host_physical_block_indices,
			kv_entries * sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaHostPinnedAlloc(
			state,
			(void **)&state->host_lane_physical_block_counts,
			max_active * sizeof(uint32_t));
	if (status != SPARK_STATUS_OK)
		return status;
	state->host_uploaded_physical_block_indices =
		(uint32_t *)calloc((size_t)kv_entries,sizeof(uint32_t));
	state->host_uploaded_lane_physical_block_counts =
		(uint32_t *)calloc((size_t)max_active,sizeof(uint32_t));
	state->host_uploaded_lane_valid =
		(uint8_t *)calloc((size_t)max_active,sizeof(uint8_t));
	state->host_physical_block_states =
		(uint8_t *)malloc((size_t)(physical_block_count * sizeof(uint8_t)));
	state->host_physical_block_sequence_ids =
		(uint64_t *)malloc((size_t)(physical_block_count * sizeof(uint64_t)));
	state->host_physical_block_logical_indices =
		(uint32_t *)malloc((size_t)(physical_block_count * sizeof(uint32_t)));
	state->host_physical_block_last_used_epochs =
		(uint64_t *)malloc((size_t)(physical_block_count * sizeof(uint64_t)));
	state->host_physical_block_pin_counts =
		(uint32_t *)calloc((size_t)physical_block_count,sizeof(uint32_t));
	state->host_kv_directory_entries =
		(SparkGlm52Pp13WorkControlKvDirectoryEntry *)malloc(
			(size_t)((uint64_t)directory_capacity *
				sizeof(SparkGlm52Pp13WorkControlKvDirectoryEntry)));
	if ((state->configuration.flags &
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_NVME_KV) != 0u)
	{
		state->host_backing_block_free_next =
			(uint32_t *)malloc((size_t)(logical_block_capacity *
				sizeof(uint32_t)));
	}
	state->host_mtp_draft_budgets =
		(uint32_t *)malloc((size_t)(state->rank_plan.logical_lane_capacity *
			sizeof(uint32_t)));
	state->host_mtp_request_slot_indices =
		(uint32_t *)malloc((size_t)(max_active * sizeof(uint32_t)));
	state->pending_work_completions =
		(SparkModelDriverCompletion *)malloc(
			(size_t)(state->rank_plan.logical_lane_capacity *
				sizeof(SparkModelDriverCompletion)));
	state->mtp_kv_transactions =
		(SparkGlm52Pp13BuilderMtpKvTransaction *)calloc(
			state->configuration.maximum_resident_sequence_count,
			sizeof(SparkGlm52Pp13BuilderMtpKvTransaction));
	if (state->host_physical_block_indices == 0 ||
		state->host_lane_physical_block_counts == 0 ||
		state->host_uploaded_physical_block_indices == 0 ||
		state->host_uploaded_lane_physical_block_counts == 0 ||
		state->host_uploaded_lane_valid == 0 ||
		state->host_physical_block_states == 0 ||
		state->host_physical_block_sequence_ids == 0 ||
		state->host_physical_block_logical_indices == 0 ||
		state->host_physical_block_last_used_epochs == 0 ||
		state->host_physical_block_pin_counts == 0 ||
		state->host_kv_directory_entries == 0 ||
		(((state->configuration.flags &
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_NVME_KV) != 0u) &&
		 state->host_backing_block_free_next == 0) ||
		state->host_mtp_draft_budgets == 0 ||
		state->host_mtp_request_slot_indices == 0 ||
		state->mtp_kv_transactions == 0 ||
		(SparkGlm52Pp13BuilderMtpEnabled(state) != 0u &&
		 (state->mtp_shadow_free_indices == 0 ||
		  state->host_mtp_tree_shadow_slot_mapping == 0 ||
		  state->device_mtp_tree_shadow_slot_mapping == 0)) ||
		state->pending_work_completions == 0)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	status = SparkGlm52Pp13WorkControlInitializeKvState(
		&state->kv_state,
		state->rank_plan.execution_row_capacity,
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS,
		(uint32_t)physical_block_count,
		directory_capacity,
		state->host_physical_block_indices,
		state->host_lane_physical_block_counts,
		state->host_physical_block_states,
		state->host_physical_block_sequence_ids,
		state->host_physical_block_logical_indices,
		state->host_physical_block_last_used_epochs,
		state->host_kv_directory_entries);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13WorkControlConfigureKvPins(
			&state->kv_state,state->host_physical_block_pin_counts);
	if (status == SPARK_STATUS_OK &&
		(state->configuration.flags &
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_NVME_KV) != 0u)
	{
		status = SparkGlm52Pp13WorkControlConfigureKvSwap(
			&state->kv_state,
			state->configuration.kv_nvme_block_capacity,
			state->host_backing_block_free_next,
			SparkGlm52Pp13BuilderKvNvmeStore,
			SparkGlm52Pp13BuilderKvNvmeLoad,
			state);
	}
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderInitializeTables(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeRank0InputBuffers(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaZero(
			state->selected_token_indices_by_layer,
			selected_indices_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaZero(
			state->selected_block_indices_by_layer,
			selected_block_bytes);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaZero(
			state->selected_block_counts_by_layer,
			(uint64_t)state->dsa_cache_layer_count *
			max_active *
			sizeof(uint32_t));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaZero(
			state->dsa_selection_epoch_by_layer,
			(uint64_t)state->dsa_cache_layer_count *
			max_active *
			sizeof(uint32_t));
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderValidateConfiguration(
	const SparkGlm52Pp13NodeContextBuilderConfiguration *configuration)
{
	if (configuration == 0 ||
		configuration->abi_version !=
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_ABI_VERSION ||
		configuration->descriptor_bytes !=
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_BYTES ||
		(configuration->flags &
			~SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_KNOWN_FLAGS) != 0u ||
		configuration->rank_plan == 0 ||
		configuration->moe_pack_root == 0 ||
		configuration->stagepack_root == 0 ||
		configuration->max_active_sequence_count == 0u ||
		configuration->max_active_sequence_count >
			SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET ||
		configuration->rank_plan->logical_lane_capacity !=
			configuration->max_active_sequence_count ||
		configuration->rank_plan->maximum_speculative_rows_per_lane !=
			SPARK_GLM52_PP13_RUNTIME_MAX_SPECULATIVE_ROWS_PER_LANE ||
		configuration->rank_plan->execution_row_capacity !=
			SparkGlm52Pp13RuntimeExecutionRowCapacity(
				configuration->rank_plan->logical_lane_capacity) ||
		configuration->kv_pool_token_capacity == 0u ||
		configuration->kv_pool_token_capacity > SPARK_GLM52_KV_POOL_TOKENS ||
		(configuration->kv_pool_token_capacity %
			SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS) != 0u ||
		configuration->maximum_resident_sequence_count <
			configuration->max_active_sequence_count ||
		configuration->kv_nvme_batch_block_count >
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_MAX_NVME_BATCH_BLOCK_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((configuration->flags &
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_NVME_KV) != 0u)
	{
		uint32_t physical_block_count;
		physical_block_count = configuration->kv_pool_token_capacity /
			SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS;
		if (configuration->kv_nvme_path == 0 ||
			configuration->kv_nvme_path[0] == '\0' ||
			configuration->kv_nvme_block_capacity < physical_block_count ||
			configuration->kv_nvme_block_capacity > UINT32_MAX / 2u ||
			configuration->kv_nvme_batch_block_count == 0u)
			return SPARK_STATUS_INVALID_ARGUMENT;
	}
	else if (configuration->kv_nvme_path != 0 ||
		configuration->kv_nvme_block_capacity != 0u ||
		configuration->kv_nvme_batch_block_count != 0u)
	{
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	if (configuration->max_active_sequence_count >=
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT &&
		(configuration->flags &
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_NVME_KV) == 0u)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	if ((configuration->flags &
			SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_CONFIGURATION_FLAG_DSPARK) != 0u &&
		(configuration->dspark_manifest_path == 0 ||
		 configuration->dspark_manifest_path[0] == '\0' ||
		 configuration->dspark_config_path == 0 ||
		 configuration->dspark_config_path[0] == '\0' ||
		 configuration->dspark_safetensors_path == 0 ||
		 configuration->dspark_safetensors_path[0] == '\0' ||
		 configuration->dspark_maximum_lane_count == 0u ||
		 configuration->dspark_maximum_lane_count >
			configuration->max_active_sequence_count ||
		 configuration->dspark_maximum_context_token_count == 0u ||
		 configuration->dspark_maximum_context_token_count >
			SPARK_GLM52_KV_CONTEXT_TOKENS))
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderInitialize(
	const SparkGlm52Pp13NodeContextBuilderConfiguration *configuration,
	void **builder_state)
{
	SparkGlm52Pp13BuilderState *state;
	size_t cuda_free_bytes;
	size_t cuda_total_bytes;
	SparkStatus status;
	if (builder_state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*builder_state = 0;
	status = SparkGlm52Pp13BuilderValidateConfiguration(configuration);
	if (status != SPARK_STATUS_OK)
		return status;
	if (configuration->rank_plan->quantization_mode ==
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_W8LUT_8BIT)
	{
		status = SparkGlm52StagePackValidateW8lutContract(
			configuration->stagepack_root,
			configuration->moe_pack_root);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	else if (configuration->rank_plan->quantization_mode ==
		SPARK_GLM52_STAGE_PLAN_QUANTIZATION_NVFP4_4BIT)
	{
		status = SparkGlm52StagePackValidateNvfp4Contract(
			configuration->stagepack_root,
			configuration->moe_pack_root);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	state = (SparkGlm52Pp13BuilderState *)calloc(1u,sizeof(*state));
	if (state == 0)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	state->configuration = *configuration;
	state->rank_plan = *configuration->rank_plan;
	state->kv_nvme_fd = -1;
	if (cudaStreamCreate(&state->stream) != cudaSuccess)
	{
		free(state);
		return SPARK_STATUS_IO_ERROR;
	}
	if (cudaMemGetInfo(&cuda_free_bytes,&cuda_total_bytes) != cudaSuccess)
	{
		cudaStreamDestroy(state->stream);
		free(state);
		return SPARK_STATUS_IO_ERROR;
	}
	state->cuda_initial_free_bytes = (uint64_t)cuda_free_bytes;
	state->cuda_total_bytes = (uint64_t)cuda_total_bytes;
	*builder_state = state;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderBuild(
	void *builder_state,
	SparkGlm52Pp13NodeContextBuilderResult *result)
{
	SparkGlm52Pp13BuilderState *state;
	uint32_t layer_offset;
	SparkStatus status;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || result == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->built != 0u)
	{
		*result = state->result;
		return SPARK_STATUS_OK;
	}
	if (state->rank_plan.layer_count == 0u ||
		state->rank_plan.layer_count > SPARK_GLM52_PP13_BUILDER_LAYER_CAPACITY)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13BuilderInitializeDsparkTopology(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeLocalDsaCacheRange(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeKvNvme(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeSharedBuffers(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeFp8ScaledGemm(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeExactPlan(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeTensorCoreFinalTokenHead(state);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeMtp(state);
	for (layer_offset = 0u;
		 status == SPARK_STATUS_OK && layer_offset < state->rank_plan.layer_count;
		 ++layer_offset)
		status = SparkGlm52Pp13BuilderBuildLayer(state,layer_offset);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderInitializeDsparkBackend(state);
	state->moe_expected_layer_count =
		SparkGlm52Pp13BuilderExpectedMoeLayerCount(state);
	if (status == SPARK_STATUS_OK &&
		state->moe_bound_layer_count != state->moe_expected_layer_count)
		status = SPARK_STATUS_MODULE_NOT_VALIDATED;
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13RuntimeValidateFp8PlanCounts(
			state->rank_plan.quantization_mode,
			state->fp8_scaled_gemm_bound_plan_count,
			state->fp8_scaled_gemm_expected_plan_count);
	if (status != SPARK_STATUS_OK)
		return status;
	memset(&state->slice_context,0,sizeof(state->slice_context));
	state->slice_context.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_ABI_VERSION;
	state->slice_context.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_SLICE_NODE_CONTEXT_DESCRIPTOR_BYTES;
	state->slice_context.first_layer_index = state->rank_plan.first_layer_index;
	state->slice_context.layer_count = state->rank_plan.layer_count;
	state->slice_context.final_token_stage =
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE) != 0u ? 1u : 0u;
	state->slice_context.layer_node_contexts = state->layer_pointers;
	state->slice_context.stage_slice_plan = &state->stage_slice_plan;
	memset(&state->result,0,sizeof(state->result));
	state->result.abi_version = SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_ABI_VERSION;
	state->result.descriptor_bytes =
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_RESULT_BYTES;
	state->result.rank_index = state->rank_plan.rank_index;
	state->result.first_layer_index = state->rank_plan.first_layer_index;
	state->result.layer_count = state->rank_plan.layer_count;
	state->result.hidden_dimension = state->rank_plan.hidden_dimension;
	state->result.vocabulary_size =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT;
	state->result.node_context = &state->slice_context;
	state->result.embedding_weight_bf16 = state->embedding_weight;
	state->result.private_state = state;
	{
		size_t cuda_free_bytes;
		size_t cuda_total_bytes;
		if (cudaMemGetInfo(&cuda_free_bytes,&cuda_total_bytes) != cudaSuccess)
			return SPARK_STATUS_IO_ERROR;
		fprintf(
			stderr,
			"pp13_builder_memory rank=%u logical_lanes=%u execution_rows=%u "
			"cuda_total_bytes=%llu cuda_initial_free_bytes=%llu "
			"cuda_current_free_bytes=%llu cuda_consumed_bytes=%llu "
			"cuda_builder_allocation_bytes=%llu "
			"cuda_largest_allocation_bytes=%llu host_mapped_bytes=%llu\n",
			state->rank_plan.rank_index,
			state->rank_plan.logical_lane_capacity,
			state->rank_plan.execution_row_capacity,
			(unsigned long long)cuda_total_bytes,
			(unsigned long long)state->cuda_initial_free_bytes,
			(unsigned long long)cuda_free_bytes,
			(unsigned long long)(state->cuda_initial_free_bytes >= cuda_free_bytes
				? state->cuda_initial_free_bytes - cuda_free_bytes : 0u),
			(unsigned long long)state->cuda_builder_allocation_bytes,
			(unsigned long long)state->cuda_largest_allocation_bytes,
			(unsigned long long)state->host_mapped_allocation_bytes);
		if (SparkGlm52Pp13BuilderIsFinalRank(state))
		{
			fprintf(
				stderr,
				"pp13_full_vocab_head rank=%u backend=cublas_bf16_tensor_core "
				"maximum_rows=%u logits_workspace_bytes=%llu fail_closed=1\n",
				state->rank_plan.rank_index,
				state->full_vocab_head_row_capacity,
				(unsigned long long)state->full_vocab_head_row_capacity *
					SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT *
					sizeof(float));
			if (SparkGlm52Pp13BuilderMtpEnabled(state) &&
				SparkGlm52Pp13BuilderUsesBf16Trunk(state) != 0u)
				fprintf(
					stderr,
					"pp13_mtp_draft_vocab_head rank=%u "
					"backend=cublas_bf16_tensor_core maximum_rows=%u "
					"weight_copy_bytes=0 workspace_bytes=0 "
					"target_verifier_backend=cublas_bf16_tensor_core fail_closed=1\n",
					state->rank_plan.rank_index,
					state->rank_plan.logical_lane_capacity);
			else if (SparkGlm52Pp13BuilderMtpEnabled(state))
				fprintf(
					stderr,
					"pp13_mtp_draft_vocab_head rank=%u "
					"backend=fp8_e4m3_block128_tensor_core maximum_rows=%u "
					"weight_bytes=%llu scale_bytes=%llu workspace_bytes=%llu "
					"target_verifier_backend=cublas_bf16_tensor_core fail_closed=1\n",
					state->rank_plan.rank_index,
					state->rank_plan.logical_lane_capacity,
					(unsigned long long)
						SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT *
						SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
					(unsigned long long)
						(SPARK_GLM52_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT /
						 SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK) *
						(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
						 SPARK_GLM52_PP13_BUILDER_MTP_DRAFT_HEAD_SCALE_BLOCK) *
						sizeof(float),
					(unsigned long long)state->mtp_draft_head_workspace_bytes);
			if (state->mtp_gpu_profile_enabled != 0u)
				fprintf(stderr,
					"pp13_mtp_gpu_profile rank=%u phases=%u max_drafts=%u "
					"graph_compatible=1\n",
					state->rank_plan.rank_index,
					SPARK_GLM52_PP13_BUILDER_MTP_GPU_PROFILE_PHASE_COUNT,
					SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT);
		}
		if (state->kv_nvme_fd >= 0)
		{
			fprintf(
				stderr,
				"pp13_kv_jit_budget rank=%u layers=%u dsa_index_layers=%u "
				"mtp_layer=%u resident_bytes_per_token=%llu "
				"resident_pool_bytes=%llu nvme_record_bytes=%llu "
				"nvme_capacity_bytes=%llu active_average_context_limit=%u "
				"backing_average_context_limit=%u batch_blocks=%u\n",
				state->rank_plan.rank_index,
				state->kv_jit_budget.layer_count,
				state->kv_jit_budget.local_dsa_index_layer_count,
				state->kv_jit_budget.include_mtp_layer,
				(unsigned long long)state->kv_jit_budget.resident_bytes_per_token,
				(unsigned long long)state->kv_jit_budget.resident_pool_bytes,
				(unsigned long long)state->kv_jit_budget.nvme_record_bytes,
				(unsigned long long)state->kv_jit_budget.nvme_capacity_bytes,
				state->kv_jit_budget.maximum_average_active_context_tokens,
				state->kv_jit_budget.maximum_average_backing_context_tokens,
				state->configuration.kv_nvme_batch_block_count);
		}
	}
	state->built = 1u;
	*result = state->result;
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13BuilderDestroyResult(
	void *builder_state,
	SparkGlm52Pp13NodeContextBuilderResult *result)
{
	(void)builder_state;
	if (result != 0)
		memset(result,0,sizeof(*result));
}

static void SparkGlm52Pp13BuilderDestroy(void *builder_state)
{
	SparkGlm52Pp13BuilderState *state;
	uint32_t index;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0)
		return;
	if (state->kv_nvme_fd >= 0 &&
		(state->kv_nvme_pending_store_count != 0u ||
		 state->kv_nvme_pending_load_count != 0u))
		(void)SparkGlm52Pp13BuilderKvNvmeFlushBatch(state);
	for (index = 0u; index < SPARK_GLM52_PP13_BUILDER_LAYER_CAPACITY; ++index)
	{
		if (state->layers[index].linear_binding != 0)
			SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(
				state->layers[index].linear_binding);
		if (state->layers[index].fp8_moe_ready != 0u)
			SparkGlm52ResidentDecodeStageFp8MoeResidentBindingDestroy(
				&state->layers[index].fp8_moe_binding);
		if (state->layers[index].b12x_moe_ready != 0u)
			SparkGlm52ResidentDecodeStageB12xMoeResidentBindingDestroy(
				&state->layers[index].b12x_moe_binding);
		if (state->layers[index].w8lut_moe_ready != 0u)
			SparkGlm52ResidentDecodeStageW8lutMoeResidentBindingDestroy(
				&state->layers[index].w8lut_moe_binding);
		if (state->layers[index].dsa_selection_event != 0)
			cudaEventDestroy(state->layers[index].dsa_selection_event);
		if (state->layers[index].dsa_prefetch_event != 0)
			cudaEventDestroy(state->layers[index].dsa_prefetch_event);
	}
	if (state->mtp_layer.linear_binding != 0)
		SparkGlm52ResidentDecodeStageLinearPlanResidentBindingDestroy(
			state->mtp_layer.linear_binding);
	if (state->mtp_layer.fp8_moe_ready != 0u)
		SparkGlm52ResidentDecodeStageFp8MoeResidentBindingDestroy(
			&state->mtp_layer.fp8_moe_binding);
	if (state->mtp_layer.b12x_moe_ready != 0u)
		SparkGlm52ResidentDecodeStageB12xMoeResidentBindingDestroy(
			&state->mtp_layer.b12x_moe_binding);
	if (state->mtp_layer.w8lut_moe_ready != 0u)
		SparkGlm52ResidentDecodeStageW8lutMoeResidentBindingDestroy(
			&state->mtp_layer.w8lut_moe_binding);
	for (index = 0u; index < state->allocation_count; ++index)
	{
		if (state->allocation_is_host_mapped[index] != 0u)
			cudaFreeHost(state->allocations[index]);
		else
			cudaFree(state->allocations[index]);
	}
	if (state->dspark_backend_ready != 0u)
		SparkGlm52DsparkDraftBackendTeardown(&state->dspark_backend);
	SparkGlm52Pp13BuilderFreeDsparkHostState(state);
	if (state->full_vocab_cublas_handle != 0)
		cublasDestroy(state->full_vocab_cublas_handle);
	if (state->stream != 0)
		cudaStreamDestroy(state->stream);
	if (state->query_stream != 0)
		cudaStreamDestroy(state->query_stream);
	if (state->kv_stream != 0)
		cudaStreamDestroy(state->kv_stream);
	if (state->kv_io_stream != 0)
		cudaStreamDestroy(state->kv_io_stream);
	if (state->branch_ready_event != 0)
		cudaEventDestroy(state->branch_ready_event);
	if (state->query_event != 0)
		cudaEventDestroy(state->query_event);
	if (state->kv_event != 0)
		cudaEventDestroy(state->kv_event);
	if (state->kv_io_event != 0)
		cudaEventDestroy(state->kv_io_event);
	if (state->kv_nvme_staging != 0)
	{
		cudaHostUnregister(state->kv_nvme_staging);
		free(state->kv_nvme_staging);
	}
	if (state->kv_nvme_fd >= 0)
	{
		(void)flock(state->kv_nvme_fd,LOCK_UN);
		close(state->kv_nvme_fd);
	}
	free(state->host_uploaded_physical_block_indices);
	free(state->host_uploaded_lane_physical_block_counts);
	free(state->host_uploaded_lane_valid);
	free(state->host_physical_block_states);
	free(state->host_physical_block_sequence_ids);
	free(state->host_physical_block_logical_indices);
	free(state->host_physical_block_last_used_epochs);
	free(state->host_physical_block_pin_counts);
	free(state->host_kv_directory_entries);
	free(state->mtp_kv_transactions);
	free(state->mtp_shadow_free_indices);
	free(state->host_backing_block_free_next);
	free(state->host_decode_positions);
	free(state->host_decode_token_ids);
	free(state->host_decode_result_token_ids);
	free(state->host_mtp_draft_budgets);
	free(state->host_mtp_committed_token_ids);
	free(state->host_mtp_request_slot_indices);
	free(state->host_mtp_previous_sequence_ids);
	free(state->host_mtp_previous_positions);
	free(state->host_mtp_previous_valid);
	free(state->pending_work_completions);
	free(state);
}

static SparkStatus SparkGlm52Pp13BuilderAttachDriver(
	void *builder_state,
	const SparkModelDriverInterface *driver_interface,
	void *driver_instance,
	const SparkModelDriverProgramDescriptor *program,
	SparkHiddenTransportSession *output_transport_session)
{
	SparkGlm52Pp13BuilderState *state;
	SparkGlm52ResidentDecodeStageProductionRunnerConfiguration configuration;
	SparkStatus status;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || driver_interface == 0 || driver_instance == 0 ||
		program == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state->driver_interface = driver_interface;
	state->driver_instance = driver_instance;
	state->program = program;
	state->output_transport_session = output_transport_session;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
	configuration.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_CONFIGURATION_BYTES;
	configuration.flags =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_ADMISSION;
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
		configuration.flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_INPUT_TRANSPORT;
	if ((state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT) != 0u)
		configuration.flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_FLAG_REQUIRE_OUTPUT_TRANSPORT;
	configuration.driver_interface = driver_interface;
	configuration.driver_instance = driver_instance;
	configuration.program = program;
	configuration.execution_stream = (void *)state->stream;
	status = SparkGlm52ResidentDecodeStageProductionRunnerInitialize(
		&state->runner,
		&configuration);
	if (status == SPARK_STATUS_OK)
		state->runner_ready = 1u;
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderQuiesceStreams(
	SparkGlm52Pp13BuilderState *state)
{
	if (cudaStreamSynchronize(state->stream) != cudaSuccess)
		return SPARK_STATUS_IO_ERROR;
	if (cudaStreamSynchronize(state->query_stream) != cudaSuccess)
		return SPARK_STATUS_IO_ERROR;
	if (cudaStreamSynchronize(state->kv_stream) != cudaSuccess)
		return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13BuilderClearGenerationState(
	SparkGlm52Pp13BuilderState *state)
{
	uint32_t lane_index;
	uint32_t request_slot_index;
	uint32_t shadow_index;
	if (state->host_uploaded_lane_valid != 0)
		memset(
			state->host_uploaded_lane_valid,
			0,
			(size_t)state->rank_plan.execution_row_capacity);
	memset(state->mtp_kv_transactions,0,
		(size_t)state->configuration.maximum_resident_sequence_count *
			sizeof(state->mtp_kv_transactions[0u]));
	for (shadow_index = 0u;
		 shadow_index < state->mtp_shadow_slot_capacity;
		 ++shadow_index)
		state->mtp_shadow_free_indices[shadow_index] = shadow_index;
	state->mtp_shadow_free_count = state->mtp_shadow_slot_capacity;
	if (state->host_mtp_previous_valid != 0)
		memset(state->host_mtp_previous_valid,0,
			state->configuration.maximum_resident_sequence_count);
	state->mtp_previous_request_id = 0u;
	state->mtp_previous_sequence_id = 0u;
	state->mtp_previous_position = 0u;
	state->mtp_previous_valid = 0u;
	state->captured_completion_valid = 0u;
	state->dspark_ready_draft_head = 0u;
	state->dspark_ready_draft_count = 0u;
	state->dspark_stage_count = 0u;
	state->dspark_draft_count = 0u;
	__atomic_store_n(
		&state->pending_finalizer_state,
		SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_NONE,
		__ATOMIC_RELEASE);
	if (state->dspark_lane_by_request_slot != 0)
	{
		for (request_slot_index = 0u;
			 request_slot_index <
				state->configuration.maximum_resident_sequence_count;
			 ++request_slot_index)
			state->dspark_lane_by_request_slot[request_slot_index] =
				SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	}
	if (state->dspark_request_slot_by_lane != 0)
	{
		for (lane_index = 0u;
			 lane_index < state->configuration.dspark_maximum_lane_count;
			 ++lane_index)
			state->dspark_request_slot_by_lane[lane_index] =
				SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	}
	if (state->dspark_backend_ready != 0u)
	{
		memset(state->dspark_backend.lane_states,0,
			(size_t)state->dspark_backend.maximum_lane_count *
				sizeof(state->dspark_backend.lane_states[0u]));
		memset(state->dspark_backend.validation_lane_states,0,
			(size_t)state->dspark_backend.maximum_lane_count *
				sizeof(state->dspark_backend.validation_lane_states[0u]));
		state->dspark_backend.pending_operation_kind =
			SPARK_GLM52_DSPARK_DRAFT_BACKEND_PENDING_NONE;
		state->dspark_backend.pending_draft_lane_count = 0u;
	}
	state->pending_work_active = 0u;
	state->pending_work_completion_function = 0;
	state->pending_work_completion_context = 0;
	state->pending_work_completion_count = 0u;
	state->pending_work_completion_overflow = 0u;
	state->runner_ready = 0u;
	state->driver_interface = 0;
	state->driver_instance = 0;
	state->program = 0;
	state->output_transport_session = 0;
}

static SparkStatus SparkGlm52Pp13BuilderResetControlGeneration(
	void *builder_state,
	uint64_t control_generation)
{
	SparkGlm52Pp13BuilderState *state;
	SparkStatus status;
	uint32_t dropped_work;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || control_generation == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13BuilderQuiesceStreams(state);
	if (status != SPARK_STATUS_OK)
		return status;
	if (state->kv_nvme_pending_store_count != 0u ||
		state->kv_nvme_pending_load_count != 0u)
	{
		status = SparkGlm52Pp13BuilderKvNvmeFlushBatch(state);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	dropped_work = state->pending_work_active;
	if (state->pending_work_active != 0u)
		(void)SparkGlm52Pp13WorkControlCancelHostKvBlockTable(
			&state->pending_work_packet,&state->kv_state);
	status = SparkGlm52Pp13WorkControlAdvanceKvGeneration(
		&state->kv_state,control_generation);
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52Pp13BuilderClearGenerationState(state);
	state->asynchronous_failure_count += dropped_work;
	fprintf(stderr,
		"pp13_builder_control_generation_reset rank=%u generation=%llu dropped=%u\n",
		state->rank_plan.rank_index,
		(unsigned long long)control_generation,dropped_work);
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13BuilderInvalidateDeviceKvDirectory(
	SparkGlm52Pp13BuilderState *state)
{
	if (state == 0 || state->host_uploaded_lane_valid == 0)
		return;
	memset(
		state->host_uploaded_lane_valid,
		0,
		(size_t)state->rank_plan.execution_row_capacity);
}

static SparkStatus SparkGlm52Pp13BuilderStageKvLane(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52KvBlockTableView *source_view,
	const uint32_t *source_blocks,
	const uint32_t *source_counts,
	uint32_t lane_index,
	uint32_t physical_block_capacity,
	uint32_t *copy_count_out)
{
	uint64_t destination_base;
	uint64_t source_base;
	uint32_t block_index;
	uint32_t copy_count;
	copy_count = source_counts[lane_index];
	if (copy_count == 0u || copy_count > source_view->lane_stride ||
		copy_count > SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	destination_base = (uint64_t)lane_index *
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	source_base = (uint64_t)lane_index * source_view->lane_stride;
	for (block_index = 0u; block_index < copy_count; ++block_index)
	{
		if (source_blocks[source_base + block_index] >= physical_block_capacity)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if (source_blocks + source_base !=
		state->host_physical_block_indices + destination_base)
		memmove(
			state->host_physical_block_indices + destination_base,
			source_blocks + source_base,
			(size_t)copy_count * sizeof(uint32_t));
	state->host_lane_physical_block_counts[lane_index] = copy_count;
	*copy_count_out = copy_count;
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13BuilderAccumulateKvLaneDelta(
	SparkGlm52Pp13BuilderState *state,
	uint32_t lane_index,
	uint32_t *dirty_begin,
	uint32_t *dirty_end,
	uint32_t *counts_changed)
{
	uint64_t base;
	uint32_t block_index;
	uint32_t copy_count;
	uint32_t previous_count;
	base = (uint64_t)lane_index *
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	copy_count = state->host_lane_physical_block_counts[lane_index];
	previous_count =
		state->host_uploaded_lane_physical_block_counts[lane_index];
	if (state->host_uploaded_lane_valid[lane_index] == 0u)
	{
		*dirty_begin = 0u;
		if (copy_count > *dirty_end)
			*dirty_end = copy_count;
		*counts_changed = 1u;
	}
	else if (copy_count != previous_count)
	{
		*counts_changed = 1u;
		if (copy_count > previous_count && previous_count < *dirty_begin)
			*dirty_begin = previous_count;
		if (copy_count > previous_count && copy_count > *dirty_end)
			*dirty_end = copy_count;
	}
	for (block_index = 0u; block_index < copy_count; ++block_index)
	{
		if (state->host_physical_block_indices[base + block_index] ==
			state->host_uploaded_physical_block_indices[base + block_index])
			continue;
		if (block_index < *dirty_begin)
			*dirty_begin = block_index;
		if (block_index + 1u > *dirty_end)
			*dirty_end = block_index + 1u;
	}
}

static void SparkGlm52Pp13BuilderFindKvDirectoryDelta(
	SparkGlm52Pp13BuilderState *state,
	uint32_t lane_count,
	uint32_t *dirty_begin_out,
	uint32_t *dirty_end_out,
	uint32_t *counts_changed_out)
{
	uint32_t lane_index;
	*dirty_begin_out = SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	*dirty_end_out = 0u;
	*counts_changed_out = 0u;
	for (lane_index = 0u; lane_index < lane_count; ++lane_index)
		SparkGlm52Pp13BuilderAccumulateKvLaneDelta(
			state,lane_index,dirty_begin_out,dirty_end_out,counts_changed_out);
}

static SparkStatus SparkGlm52Pp13BuilderUploadKvDirectoryDelta(
	SparkGlm52Pp13BuilderState *state,
	uint32_t lane_count,
	uint32_t dirty_begin,
	uint32_t dirty_end,
	uint32_t counts_changed)
{
	uint64_t base;
	uint32_t copy_count;
	uint32_t copy_end;
	uint32_t lane_index;
	size_t pitch_bytes;
	SparkStatus status;
	pitch_bytes =
		(size_t)SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE *
		sizeof(uint32_t);
	status = SPARK_STATUS_OK;
	if (dirty_end > dirty_begin)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy2DAsync(
			state->device_physical_block_indices + dirty_begin,
			pitch_bytes,
			state->host_physical_block_indices + dirty_begin,
			pitch_bytes,
			(size_t)(dirty_end - dirty_begin) * sizeof(uint32_t),
			lane_count,
			cudaMemcpyHostToDevice,
			state->stream));
	if (status == SPARK_STATUS_OK && counts_changed != 0u)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_lane_physical_block_counts,
			state->host_lane_physical_block_counts,
			(size_t)lane_count * sizeof(uint32_t),
			cudaMemcpyHostToDevice,
			state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u; lane_index < lane_count; ++lane_index)
	{
		base = (uint64_t)lane_index *
			SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
		copy_count = state->host_lane_physical_block_counts[lane_index];
		copy_end = dirty_end < copy_count ? dirty_end : copy_count;
		if (copy_end > dirty_begin)
			memmove(
				state->host_uploaded_physical_block_indices + base + dirty_begin,
				state->host_physical_block_indices + base + dirty_begin,
				(size_t)(copy_end - dirty_begin) * sizeof(uint32_t));
		state->host_uploaded_lane_physical_block_counts[lane_index] = copy_count;
		state->host_uploaded_lane_valid[lane_index] = 1u;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderPrepareDeviceKvView(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52KvBlockTableView *source_view)
{
	const uint32_t *source_blocks;
	const uint32_t *source_counts;
	uint32_t lane_index;
	uint32_t copy_count;
	uint32_t counts_changed;
	uint32_t dirty_begin;
	uint32_t dirty_end;
	uint32_t maximum_copy_count;
	uint32_t physical_block_capacity;
	SparkStatus status;
	if (state == 0 || source_view == 0 ||
		source_view->abi_version != SPARK_GLM52_KV_CACHE_ABI_VERSION ||
		source_view->descriptor_bytes !=
			SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES ||
		source_view->lane_count == 0u ||
		source_view->lane_count > state->rank_plan.logical_lane_capacity ||
		source_view->lane_stride == 0u ||
		source_view->block_token_count !=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_BLOCK_TOKENS ||
		source_view->physical_block_indices == 0 ||
		source_view->lane_physical_block_counts == 0 ||
		source_view->host_physical_block_indices == 0 ||
		source_view->host_lane_physical_block_counts == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->configuration.kv_pool_token_capacity == 0u ||
		state->configuration.kv_pool_token_capacity %
			source_view->block_token_count != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	physical_block_capacity = state->configuration.kv_pool_token_capacity /
		source_view->block_token_count;
	source_blocks = source_view->host_physical_block_indices;
	source_counts = source_view->host_lane_physical_block_counts;
	maximum_copy_count = 0u;
	for (lane_index = 0u; lane_index < source_view->lane_count; ++lane_index)
	{
		status = SparkGlm52Pp13BuilderStageKvLane(
			state,source_view,source_blocks,source_counts,lane_index,
			physical_block_capacity,&copy_count);
		if (status != SPARK_STATUS_OK)
			return status;
		if (copy_count > maximum_copy_count)
			maximum_copy_count = copy_count;
	}
	SparkGlm52Pp13BuilderFindKvDirectoryDelta(
		state,source_view->lane_count,&dirty_begin,&dirty_end,&counts_changed);
	status = SparkGlm52Pp13BuilderUploadKvDirectoryDelta(
		state,source_view->lane_count,dirty_begin,dirty_end,counts_changed);
	if (status != SPARK_STATUS_OK)
		return status;
	state->device_kv_view = *source_view;
	state->device_kv_view.lane_stride =
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
	state->device_kv_view.lane_capacity =
		state->device_kv_view.lane_stride;
	state->device_kv_view.physical_block_indices =
		state->device_physical_block_indices;
	state->device_kv_view.lane_physical_block_counts =
		state->device_lane_physical_block_counts;
	state->device_kv_view.host_physical_block_indices =
		state->host_physical_block_indices;
	state->device_kv_view.host_lane_physical_block_counts =
		state->host_lane_physical_block_counts;
	state->active_kv_block_count = maximum_copy_count;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderApplyMtpTreeKvRows(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkGlm52Pp13BuilderMtpKvTransaction *transaction;
	const SparkGlm52Pp13WorkControlLane *lane;
	uint32_t lane_index;
	uint32_t row_base;
	uint32_t depth2_block_index;
	uint32_t depth3_block_index;
	uint64_t table_base;
	if (state == 0 || work_packet == 0 ||
		SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet) == 0u ||
		work_packet->rows_per_lane !=
			SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		lane = &work_packet->lanes[lane_index];
		if (lane->request_slot_index >=
			state->configuration.maximum_resident_sequence_count)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		transaction = &state->mtp_kv_transactions[lane->request_slot_index];
		if (transaction->active == 0u || transaction->tree_verify == 0u ||
			transaction->transient_block_count !=
				SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_BLOCK_COUNT)
			return SPARK_STATUS_INTERNAL_ERROR;
		depth2_block_index =
			(uint32_t)((transaction->base_position + 2u) /
				work_packet->block_token_count);
		depth3_block_index =
			(uint32_t)((transaction->base_position + 3u) /
				work_packet->block_token_count);
		if (depth2_block_index >=
				state->host_lane_physical_block_counts[
					lane_index * work_packet->rows_per_lane] ||
			depth3_block_index >=
				state->host_lane_physical_block_counts[
					lane_index * work_packet->rows_per_lane])
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		row_base = lane_index * work_packet->rows_per_lane;
		table_base = (uint64_t)(
			row_base + SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_ALTERNATE_ROW) *
			state->kv_state.lane_stride;
		state->host_physical_block_indices[
			table_base + depth2_block_index] =
			transaction->transient_physical_blocks[
				SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH2_ALTERNATE_INDEX];
		table_base = (uint64_t)(
			row_base + SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_ALTERNATE_ROW) *
			state->kv_state.lane_stride;
		state->host_physical_block_indices[
			table_base + depth3_block_index] =
			transaction->transient_physical_blocks[
				SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH3_ALTERNATE_INDEX];
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderExpandExecutionKvRows(
	SparkGlm52Pp13BuilderState *state,
	uint32_t logical_lane_count,
	uint32_t rows_per_lane,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	uint64_t execution_row_count;
	uint32_t copy_count;
	uint32_t row_offset;
	int32_t lane_index;
	SparkStatus status;
	if (state == 0 || logical_lane_count == 0u || rows_per_lane < 2u ||
		state->active_kv_block_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	execution_row_count = (uint64_t)logical_lane_count * rows_per_lane;
	if (execution_row_count > state->rank_plan.execution_row_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	for (lane_index = (int32_t)logical_lane_count - 1;
		 lane_index >= 0;
		 --lane_index)
	{
		uint64_t source_base;
		copy_count = state->host_lane_physical_block_counts[lane_index];
		if (copy_count == 0u ||
			copy_count > state->active_kv_block_count)
			return SPARK_STATUS_INVALID_ARGUMENT;
		source_base = (uint64_t)(uint32_t)lane_index *
			SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
		for (row_offset = rows_per_lane; row_offset != 0u; --row_offset)
		{
			uint32_t execution_row_index;
			uint64_t destination_base;
			execution_row_index = (uint32_t)lane_index * rows_per_lane +
				(row_offset - 1u);
			destination_base = (uint64_t)execution_row_index *
				SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
			memmove(
				state->host_physical_block_indices + destination_base,
				state->host_physical_block_indices + source_base,
				(size_t)copy_count * sizeof(uint32_t));
			state->host_lane_physical_block_counts[execution_row_index] =
				copy_count;
		}
	}
	if (work_packet != 0 &&
		SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet) != 0u)
	{
		status = SparkGlm52Pp13BuilderApplyMtpTreeKvRows(state,work_packet);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy2DAsync(
		state->device_physical_block_indices,
		(size_t)SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE *
			sizeof(uint32_t),
		state->host_physical_block_indices,
		(size_t)SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE *
			sizeof(uint32_t),
		(size_t)state->active_kv_block_count * sizeof(uint32_t),
		(size_t)execution_row_count,
		cudaMemcpyHostToDevice,
		state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_lane_physical_block_counts,
			state->host_lane_physical_block_counts,
			(size_t)execution_row_count * sizeof(uint32_t),
			cudaMemcpyHostToDevice,
			state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	state->device_kv_view.lane_count = (uint32_t)execution_row_count;
	SparkGlm52Pp13BuilderInvalidateDeviceKvDirectory(state);
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderCompactExecutionKvRows(
	SparkGlm52Pp13BuilderState *state,
	uint32_t logical_lane_count,
	uint32_t rows_per_lane)
{
	uint64_t destination_base;
	uint64_t source_base;
	uint32_t copy_count;
	uint32_t lane_index;
	SparkStatus status;
	if (state == 0 || logical_lane_count == 0u || rows_per_lane < 2u ||
		state->active_kv_block_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (lane_index = 0u; lane_index < logical_lane_count; ++lane_index)
	{
		source_base = (uint64_t)lane_index * rows_per_lane *
			SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
		destination_base = (uint64_t)lane_index *
			SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE;
		copy_count = state->host_lane_physical_block_counts[
			lane_index * rows_per_lane];
		if (copy_count == 0u || copy_count > state->active_kv_block_count)
			return SPARK_STATUS_INVALID_ARGUMENT;
		memmove(state->host_physical_block_indices + destination_base,
			state->host_physical_block_indices + source_base,
			(size_t)copy_count * sizeof(uint32_t));
		state->host_lane_physical_block_counts[lane_index] = copy_count;
	}
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy2DAsync(
		state->device_physical_block_indices,
		(size_t)SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE * sizeof(uint32_t),
		state->host_physical_block_indices,
		(size_t)SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE * sizeof(uint32_t),
		(size_t)state->active_kv_block_count * sizeof(uint32_t),
		logical_lane_count,cudaMemcpyHostToDevice,state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_lane_physical_block_counts,
			state->host_lane_physical_block_counts,
			(size_t)logical_lane_count * sizeof(uint32_t),
			cudaMemcpyHostToDevice,state->stream));
	if (status == SPARK_STATUS_OK)
	{
		state->device_kv_view.lane_count = logical_lane_count;
		SparkGlm52Pp13BuilderInvalidateDeviceKvDirectory(state);
	}
	return status;
}

static void SparkGlm52Pp13BuilderLogKvGenerationReset(
	const SparkGlm52Pp13BuilderState *state,
	uint64_t prior_control_generation)
{
	if (state->kv_state.control_generation == prior_control_generation)
		return;
	fprintf(stderr,
		"pp13_kv_generation_reset rank=%u old=%llu new=%llu count=%llu\n",
		state->rank_plan.rank_index,
		(unsigned long long)prior_control_generation,
		(unsigned long long)state->kv_state.control_generation,
		(unsigned long long)state->kv_state.control_generation_reset_count);
}

static SparkStatus SparkGlm52Pp13BuilderUploadMtpBudget(
	SparkGlm52Pp13BuilderState *state,
	uint32_t active_sequence_count,
	uint32_t mtp_draft_token_count)
{
	uint32_t lane_index;
	if (state == 0 ||
		active_sequence_count == 0u ||
		active_sequence_count > state->rank_plan.logical_lane_capacity ||
		(mtp_draft_token_count != 0u &&
		 mtp_draft_token_count !=
			SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT) ||
		state->host_mtp_draft_budgets == 0 ||
		state->layers[0].mtp_draft_token_budgets == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (lane_index = 0u; lane_index < active_sequence_count; ++lane_index)
		state->host_mtp_draft_budgets[lane_index] = mtp_draft_token_count;
	state->mtp_draft_plan.graph_draft_token_count =
		mtp_draft_token_count == 0u
			? 0u : SPARK_GLM52_MODEL_MTP_TREE_EXECUTION_STEP_COUNT;
	return SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->layers[0].mtp_draft_token_budgets,
		state->host_mtp_draft_budgets,
		(size_t)(active_sequence_count * sizeof(uint32_t)),
		cudaMemcpyHostToDevice,
		state->stream));
}

static SparkStatus SparkGlm52Pp13BuilderUploadWorkMtpBudgets(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	uint32_t lane_index;
	uint32_t graph_draft_token_count;
	if (state == 0 || work_packet == 0 ||
		work_packet->lane_count == 0u ||
		work_packet->lane_count > state->rank_plan.logical_lane_capacity ||
		state->host_mtp_draft_budgets == 0 ||
		state->layers[0].mtp_draft_token_budgets == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	graph_draft_token_count = 0u;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		if (work_packet->lanes[lane_index].mtp_draft_token_count != 0u &&
			work_packet->lanes[lane_index].mtp_draft_token_count !=
				SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT)
			return SPARK_STATUS_INVALID_ARGUMENT;
		state->host_mtp_draft_budgets[lane_index] =
			work_packet->lanes[lane_index].mtp_draft_token_count;
		if (graph_draft_token_count <
			work_packet->lanes[lane_index].mtp_draft_token_count)
			graph_draft_token_count =
				work_packet->lanes[lane_index].mtp_draft_token_count;
	}
	state->mtp_draft_plan.graph_draft_token_count =
		graph_draft_token_count == 0u
			? 0u : SPARK_GLM52_MODEL_MTP_TREE_EXECUTION_STEP_COUNT;
	return SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->layers[0].mtp_draft_token_budgets,
		state->host_mtp_draft_budgets,
		(size_t)work_packet->lane_count * sizeof(uint32_t),
		cudaMemcpyHostToDevice,
		state->stream));
}

static SparkStatus SparkGlm52Pp13BuilderLaunchDecodeMetadataForAllLayers(
	SparkGlm52Pp13BuilderState *state,
	uint32_t active_sequence_count)
{
	uint32_t block_count;
	SparkStatus status;

	if (state == 0 || active_sequence_count == 0u ||
		active_sequence_count > state->rank_plan.execution_row_capacity)
		return SPARK_STATUS_INVALID_ARGUMENT;
	block_count =
		(active_sequence_count + SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS;
	if (state->active_kv_block_count == 0u ||
		state->active_kv_block_count >
			SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE)
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkGlm52Pp13BuilderBuildDecodeMetadataKernel<<<
		block_count,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0,
		state->stream>>>(
			(const uint32_t *)state->device_decode_positions,
			state->device_kv_view.physical_block_indices,
			state->device_kv_view.lane_physical_block_counts,
			state->device_kv_view.lane_stride,
			state->device_kv_view.block_token_count,
			active_sequence_count,
			(uint32_t *)state->layers[0u].positions,
			(uint32_t *)state->layers[0u].slot_mapping,
			(uint32_t *)state->layers[0u].context_lengths,
			(uint32_t *)state->layers[0u].first_block_token_offsets);
	status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderUploadWorkDecodePositions(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	uint32_t execution_row_index;
	uint32_t lane_index;
	uint32_t position_offset;
	uint32_t row_offset;
	if (state == 0 || work_packet == 0 ||
		work_packet->lane_count == 0u ||
		work_packet->lane_count > state->rank_plan.logical_lane_capacity ||
		state->host_decode_positions == 0 ||
		state->device_decode_positions == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	execution_row_index = 0u;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		if (work_packet->lanes[lane_index].sequence_position >
			UINT32_MAX - (work_packet->rows_per_lane - 1u))
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		for (row_offset = 0u;
			 row_offset < work_packet->rows_per_lane;
			 ++row_offset)
		{
			position_offset =
				SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet) != 0u
					? SparkGlm52MtpTreeVerifierPositionOffset(row_offset)
					: row_offset;
			if (position_offset == UINT32_MAX)
				return SPARK_STATUS_INVALID_ARGUMENT;
			state->host_decode_positions[execution_row_index++] =
				(uint32_t)work_packet->lanes[lane_index].sequence_position +
				position_offset;
		}
	}
	if (execution_row_index != work_packet->execution_row_count ||
		execution_row_index > state->rank_plan.execution_row_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_decode_positions,
		state->host_decode_positions,
		(size_t)execution_row_index * sizeof(uint32_t),
		cudaMemcpyHostToDevice,
		state->stream));
}

static SparkStatus SparkGlm52Pp13BuilderUploadWorkTokenIdsAndEmbedding(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	uint64_t word_count;
	uint32_t block_count;
	uint32_t execution_row_index;
	uint32_t lane_index;
	uint32_t row_offset;
	SparkStatus status;

	if (state == 0 || work_packet == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((state->rank_plan.flags &
		SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u)
		return SPARK_STATUS_OK;
	if (state->host_decode_token_ids == 0 ||
		state->device_decode_token_ids == 0 || state->embedding_weight == 0 ||
		state->layers[0].input_hidden == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	execution_row_index = 0u;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		for (row_offset = 0u;
			 row_offset < work_packet->rows_per_lane;
			 ++row_offset)
		{
			state->host_decode_token_ids[execution_row_index++] =
				(work_packet->flags &
				 SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u
					? work_packet->prefill_token_ids[
						(lane_index * work_packet->rows_per_lane) + row_offset]
					: row_offset == 0u
					? work_packet->lanes[lane_index].input_token_id
					: work_packet->lanes[lane_index].speculative_draft_token_ids[
						row_offset - 1u];
		}
	}
	if (execution_row_index != work_packet->execution_row_count ||
		execution_row_index == 0u ||
		execution_row_index > state->rank_plan.execution_row_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_decode_token_ids,state->host_decode_token_ids,
		(size_t)execution_row_index * sizeof(uint32_t),
		cudaMemcpyHostToDevice,state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	word_count = (uint64_t)execution_row_index *
		(SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
	block_count = (uint32_t)((word_count +
		SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS);
	SparkGlm52Pp13BuilderGatherDecodeEmbeddingKernel<<<
		block_count,SPARK_GLM52_PP13_BUILDER_THREADS,0u,state->stream>>>(
			(const uint32_t *)state->device_decode_token_ids,
			(const uint32_t *)state->embedding_weight,
			(uint32_t *)state->layers[0].input_hidden,
			execution_row_index,
			SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static void SparkGlm52Pp13BuilderBuildPacket(
	const SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	const void *hidden,
	void *sideband,
	uint32_t needs_sideband,
	SparkHiddenTransportPacket *packet)
{
	memset(packet,0,sizeof(*packet));
	packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
	packet->flags =
		SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 |
		SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
	packet->active_sequence_count = work_packet->execution_row_count;
	packet->hidden_dimension = SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	packet->bytes_per_sequence =
		SPARK_GLM52_PP13_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE;
	packet->sequence_id = work_packet->sequence_id;
	packet->token_index = work_packet->sequence_position;
	packet->hidden_bf16 = hidden;
	packet->cuda_stream = (void *)state->stream;
	if (needs_sideband != 0u)
	{
		packet->flags |= SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_SIDEBAND_PAYLOAD;
		packet->sideband_payload = sideband;
		packet->sideband_kind =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_TRANSPORT_SIDEBAND_INDEXSHARE_SELECTED_TOKENS;
		packet->sideband_bytes_per_sequence =
			SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_INDEX_BYTES;
	}
}

static SparkStatus SparkGlm52Pp13BuilderArmDsparkSideband(
	const SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	uint32_t export_stage_index,
	void *sideband_payload,
	SparkHiddenTransportPacket *packet)
{
	if (!SparkGlm52Pp13BuilderWorkCapturesDspark(work_packet))
		return SPARK_STATUS_OK;
	return SparkGlm52ProductionTopologyArmHopSidebandPacket(
		&state->production_topology,
		export_stage_index,
		sideband_payload,
		packet);
}

static void SparkGlm52Pp13BuilderApplyDsparkDispatch(
	const SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	SparkGlm52ResidentDecodeStageProductionRunnerDispatch *dispatch)
{
	if (!SparkGlm52Pp13BuilderWorkCapturesDspark(work_packet))
		return;
	dispatch->dspark_hidden_tap_plan = &state->dspark_tap_plan;
	dispatch->dspark_hidden_tap_lane_stride_bytes =
		SPARK_GLM52_PP13_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE;
	if (state->dspark_backend_ready != 0u)
	{
		dispatch->dspark_hidden_tap_outputs_bf16 =
			(void *const *)state->dspark_tap_outputs_bf16;
		dispatch->dspark_hidden_tap_lane_stride_bytes =
			state->dspark_tap_lane_stride_bytes;
	}
}

static uint32_t SparkGlm52Pp13BuilderNeedsInputSideband(
	const SparkGlm52Pp13BuilderState *state)
{
	uint32_t first_layer;
	uint32_t layer_offset;
	first_layer = state->rank_plan.first_layer_index;
	for (layer_offset = 0u; layer_offset < state->rank_plan.layer_count; ++layer_offset)
	{
		if (state->layers[layer_offset].node.sparse_index_mode ==
				SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_SHARED &&
			state->layers[layer_offset].node.dsa_indexshare_source_layer_index <
				first_layer)
			return 1u;
	}
	return 0u;
}

static uint32_t SparkGlm52Pp13BuilderNeedsOutputSideband(
	const SparkGlm52Pp13BuilderState *state)
{
	uint32_t stage_end;
	uint32_t layer_offset;
	stage_end = state->rank_plan.first_layer_index + state->rank_plan.layer_count;
	for (layer_offset = 0u; layer_offset < state->rank_plan.layer_count; ++layer_offset)
	{
		if (state->layers[layer_offset].node.sparse_index_mode ==
				SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_DSA_INDEXSHARE_FULL &&
			state->layers[layer_offset].node.dsa_indexshare_group_end_layer_exclusive >
				stage_end)
			return 1u;
	}
	return 0u;
}

static SparkStatus SparkGlm52Pp13BuilderSetDsaCandidateCount(
	SparkGlm52Pp13BuilderState *state,
	uint32_t context_token_count)
{
	uint32_t candidate_count;
	uint32_t layer_offset;
	if (state == 0 || context_token_count == 0u ||
		context_token_count > SPARK_GLM52_KV_CONTEXT_TOKENS)
		return SPARK_STATUS_INVALID_ARGUMENT;
	candidate_count = SparkGlm52Pp13RuntimeDsaCandidateBucket(
		context_token_count);
	if (candidate_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (layer_offset = 0u; layer_offset < state->rank_plan.layer_count;
		 ++layer_offset)
	{
		if (candidate_count >
			state->layers[layer_offset].node.dsa_candidate_capacity)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		state->layers[layer_offset].slot.dsa_candidate_count = candidate_count;
	}
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13BuilderCaptureCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *completion)
{
	SparkGlm52Pp13BuilderState *state;

	state = (SparkGlm52Pp13BuilderState *)completion_context;
	if (state == 0 || completion == 0)
		return;
	state->captured_completion = *completion;
	state->captured_completion_valid = 1u;
}

static void SparkGlm52Pp13BuilderTraceDsparkDraft(
	const SparkGlm52DsparkDraftRequest *request,
	const SparkGlm52DsparkDraftResult *result)
{
	uint32_t token_index;

	if (getenv("SPARKPIPE_DSPARK_TRACE") == 0 || request == 0 || result == 0)
		return;
	fprintf(stderr,"dspark_trace draft request=%llu sequence=%llu position=%llu count=%u",
		(unsigned long long)request->request_id,
		(unsigned long long)request->sequence_id,
		(unsigned long long)request->sequence_position,
		result->token_count);
	for (token_index = 0u; token_index < result->token_count; ++token_index)
		fprintf(stderr," token%u=%u confidence%u=%u",token_index,
			result->token_ids[token_index],token_index,
			result->confidence_milli[token_index]);
	fputc('\n',stderr);
}

static SparkStatus SparkGlm52Pp13BuilderStoreMtpPreviousTarget(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet);

static SparkStatus SparkGlm52Pp13BuilderLaunchPreparedVerifierMtpDraft(
	SparkGlm52Pp13BuilderState *state,
	uint32_t lane_count,
	uint32_t draft_token_count,
	const char *profile_kind)
{
	SparkGlm52Pp13BuilderLayer *final_layer;
	SparkGlm52ResidentDecodeStagePipelineSlot base_slot;
	SparkStatus status;
	if (state == 0 || lane_count == 0u || draft_token_count == 0u ||
		profile_kind == 0 || state->device_decode_token_ids == 0 ||
		state->mtp_base_positions == 0 || state->mtp_previous_target_hidden == 0 ||
		state->host_mtp_committed_token_ids == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	final_layer = &state->layers[state->rank_plan.layer_count - 1u];
	if (final_layer->mtp_draft_token_ids == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	status = SparkGlm52Pp13BuilderUploadMtpBudget(
		state,lane_count,draft_token_count);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_decode_token_ids,state->host_decode_token_ids,
			(size_t)lane_count * sizeof(uint32_t),cudaMemcpyHostToDevice,
			state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->mtp_base_positions,state->host_decode_positions,
			(size_t)lane_count * sizeof(uint32_t),cudaMemcpyHostToDevice,
			state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	base_slot = final_layer->slot;
	base_slot.positions = state->mtp_base_positions;
	base_slot.normalized_hidden_bf16 = state->mtp_previous_target_hidden;
	base_slot.restricted_selected_token_ids =
		(uint32_t *)state->device_decode_token_ids;
	status = SparkGlm52Pp13BuilderLaunchMtpDraftPlan(
		&state->mtp_draft_plan,&final_layer->node,&base_slot,lane_count,
		(void *)state->stream);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
			state->host_mtp_committed_token_ids,final_layer->mtp_draft_token_ids,
			(size_t)lane_count *
				SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT *
				sizeof(uint32_t),cudaMemcpyDeviceToHost));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderReportMtpGpuProfile(
			state,profile_kind,lane_count,draft_token_count);
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderEmitWideDecodeCompletions(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	SparkModelDriverCompletionFunction completion_function,
	void *completion_context)
{
	SparkGlm52Pp13BuilderLayer *final_layer;
	SparkModelDriverCompletion completion;
	uint32_t lane_index;
	uint32_t draft_index;
	uint32_t maximum_draft_count;
	SparkStatus status;

	if (state == 0 || work_packet == 0 ||
		state->captured_completion_valid == 0u ||
		!SparkGlm52Pp13BuilderIsFinalRank(state) ||
		work_packet->lane_count == 0u ||
		work_packet->lane_count > state->rank_plan.logical_lane_capacity)
		return SPARK_STATUS_INVALID_ARGUMENT;
	final_layer = &state->layers[state->rank_plan.layer_count - 1u];
	if (final_layer->restricted_selected_token_ids == 0 ||
		state->host_decode_result_token_ids == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
		state->host_decode_result_token_ids,
		final_layer->restricted_selected_token_ids,
		(size_t)work_packet->lane_count * sizeof(uint32_t),
		cudaMemcpyDeviceToHost));
	if (status != SPARK_STATUS_OK)
		return status;
	maximum_draft_count = 0u;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		if (work_packet->lanes[lane_index].mtp_draft_token_count >
			maximum_draft_count)
			maximum_draft_count =
				work_packet->lanes[lane_index].mtp_draft_token_count;
	}
	if (maximum_draft_count != 0u)
	{
		if (final_layer->mtp_draft_token_ids == 0 ||
			state->host_mtp_committed_token_ids == 0)
			return SPARK_STATUS_MODULE_NOT_VALIDATED;
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
			state->host_mtp_committed_token_ids,
			final_layer->mtp_draft_token_ids,
			(size_t)work_packet->lane_count *
				SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT *
				sizeof(uint32_t),
			cudaMemcpyDeviceToHost));
		if (status != SPARK_STATUS_OK)
			return status;
	}
	status = SparkGlm52Pp13BuilderReportMtpGpuProfile(
		state,"decode",work_packet->lane_count,maximum_draft_count);
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		completion = state->captured_completion;
		completion.request_id = work_packet->lanes[lane_index].request_id;
		completion.sequence_id = work_packet->lanes[lane_index].sequence_id;
		completion.sequence_position =
			work_packet->lanes[lane_index].sequence_position;
		completion.completion_flags |= SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS;
		completion.token_count =
			work_packet->lanes[lane_index].mtp_draft_token_count + 1u;
		completion.accepted_token_count = completion.token_count;
		completion.token_ids[0u] =
			state->host_decode_result_token_ids[lane_index];
		for (draft_index = 0u;
			 draft_index < work_packet->lanes[lane_index].mtp_draft_token_count;
			 ++draft_index)
		{
			completion.token_ids[draft_index + 1u] =
				state->host_mtp_committed_token_ids[
					((uint64_t)lane_index *
					 SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) +
					draft_index];
		}
		for (draft_index = completion.token_count;
			 draft_index < SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY;
			 ++draft_index)
			completion.token_ids[draft_index] = 0u;
		if (completion_function != 0)
			completion_function(completion_context,&completion);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderResolveMtpLane(
	const SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	uint32_t lane_index,
	SparkGlm52MtpTreeResolution *resolution)
{
	const SparkGlm52Pp13WorkControlLane *lane;
	uint32_t accepted_count;
	uint32_t execution_row_base;
	if (state == 0 || work_packet == 0 || resolution == 0 ||
		lane_index >= work_packet->lane_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	lane = &work_packet->lanes[lane_index];
	execution_row_base = lane_index * work_packet->rows_per_lane;
	if (SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet) != 0u)
		return SparkGlm52MtpTreeResolve(
			lane->speculative_draft_token_ids,
			state->host_decode_result_token_ids + execution_row_base,
			resolution);
	accepted_count = 0u;
	while (accepted_count < lane->speculative_token_count &&
		state->host_decode_result_token_ids[execution_row_base + accepted_count] ==
			lane->speculative_draft_token_ids[accepted_count])
		accepted_count += 1u;
	resolution->path_id = SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE;
	resolution->accepted_token_count = accepted_count;
	resolution->committed_token_count = accepted_count + 1u;
	resolution->fallback_row_index = accepted_count;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLaunchVerifierMtpDraft(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	uint32_t *draft_token_count_out)
{
	uint32_t lane_index;
	SparkGlm52MtpTreeResolution resolution;
	SparkStatus status;
	if (state == 0 || work_packet == 0 || draft_token_count_out == 0 ||
		state->host_decode_token_ids == 0 || state->host_decode_positions == 0 ||
		state->mtp_base_positions == 0 || state->device_decode_token_ids == 0 ||
		SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet) == 0u ||
		work_packet->mtp_draft_token_count !=
			SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		status = SparkGlm52Pp13BuilderResolveMtpLane(
			state,work_packet,lane_index,&resolution);
		if (status != SPARK_STATUS_OK)
			return status;
		if (work_packet->lanes[lane_index].sequence_position >
			UINT32_MAX - resolution.accepted_token_count)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		state->host_decode_token_ids[lane_index] =
			state->host_decode_result_token_ids[
				(lane_index * work_packet->rows_per_lane) +
				resolution.fallback_row_index];
		state->host_decode_positions[lane_index] = (uint32_t)
			(work_packet->lanes[lane_index].sequence_position +
				resolution.accepted_token_count);
	}
	status = SparkGlm52Pp13BuilderLaunchPreparedVerifierMtpDraft(
		state,work_packet->lane_count,
		SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT,"verify_followup");
	if (status == SPARK_STATUS_OK)
		*draft_token_count_out =
			SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderPrepareMtpTreeRebaseInputs(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	const SparkGlm52Pp13WorkControlLane *lane;
	uint32_t candidate_index,lane_index,parent_row,row_base;
	SparkGlm52MtpTreeResolution resolution;
	SparkStatus status;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		lane = &work_packet->lanes[lane_index];
		status = SparkGlm52Pp13BuilderResolveMtpLane(
			state,work_packet,lane_index,&resolution);
		if (status != SPARK_STATUS_OK)
			return status;
		candidate_index =
			SparkGlm52MtpTreeTailCandidateIndex(resolution.path_id);
		parent_row =
			SparkGlm52MtpTreeTailParentRowIndex(resolution.path_id);
		if (candidate_index >= lane->speculative_token_count ||
			parent_row >= work_packet->rows_per_lane)
			return SPARK_STATUS_INVALID_ARGUMENT;
		row_base = lane_index * work_packet->rows_per_lane;
		state->host_decode_positions[lane_index] = row_base + parent_row;
		state->host_decode_token_ids[lane_index] =
			lane->speculative_draft_token_ids[candidate_index];
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpTreeRebaseNorm(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkGlm52Pp13BuilderLayer *final_layer;
	SparkStatus status;
	final_layer = &state->layers[state->rank_plan.layer_count - 1u];
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_decode_positions,state->host_decode_positions,
		(size_t)work_packet->lane_count * sizeof(uint32_t),
		cudaMemcpyHostToDevice,state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52Pp13BuilderSelectedTargetFinalNormKernel<<<
		work_packet->lane_count,SPARK_GLM52_PP13_BUILDER_THREADS,0u,state->stream>>>(
		(const uint16_t *)final_layer->layer_output_hidden,
		(const uint16_t *)final_layer->final_norm_weight,
		(const uint32_t *)state->device_decode_positions,
		(uint16_t *)state->mtp_previous_target_hidden,
		work_packet->lane_count,
		SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52Pp13BuilderPrepareMtpTreeRebasePositions(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	const SparkGlm52Pp13WorkControlLane *lane;
	uint32_t lane_index,position_offset;
	SparkGlm52MtpTreeResolution resolution;
	SparkStatus status;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		lane = &work_packet->lanes[lane_index];
		status = SparkGlm52Pp13BuilderResolveMtpLane(
			state,work_packet,lane_index,&resolution);
		if (status != SPARK_STATUS_OK)
			return status;
		position_offset =
			SparkGlm52MtpTreeTailBasePositionOffset(resolution.path_id);
		if (lane->sequence_position > UINT32_MAX - position_offset)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		state->host_decode_positions[lane_index] =
			(uint32_t)lane->sequence_position + position_offset;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderRebaseMtpTreeCache(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkStatus status;
	if (state == 0 || work_packet == 0 ||
		SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet) == 0u ||
		work_packet->lane_count == 0u ||
		work_packet->mtp_draft_token_count !=
			SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT ||
		state->host_decode_positions == 0 ||
		state->host_decode_token_ids == 0 ||
		state->device_decode_positions == 0 ||
		state->device_decode_token_ids == 0 ||
		state->mtp_base_positions == 0 ||
		state->mtp_previous_target_hidden == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->layers[state->rank_plan.layer_count - 1u].layer_output_hidden == 0 ||
		state->layers[state->rank_plan.layer_count - 1u].final_norm_weight == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	status = SparkGlm52Pp13BuilderPrepareMtpTreeRebaseInputs(
		state,work_packet);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderLaunchMtpTreeRebaseNorm(
			state,work_packet);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderPrepareMtpTreeRebasePositions(
			state,work_packet);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderPrepareMtpLinearPlanRows(
		state,work_packet->lane_count);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_decode_token_ids,state->host_decode_token_ids,
			(size_t)work_packet->lane_count * sizeof(uint32_t),
			cudaMemcpyHostToDevice,state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->mtp_base_positions,state->host_decode_positions,
			(size_t)work_packet->lane_count * sizeof(uint32_t),
			cudaMemcpyHostToDevice,state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52Pp13BuilderLaunchMtpLayer(
		state,(const uint32_t *)state->device_decode_token_ids,
		state->mtp_base_positions,state->mtp_previous_target_hidden,0u,
		work_packet->lane_count,state->stream);
}

static uint64_t SparkGlm52Pp13BuilderMonotonicTimeNs(void)
{
	struct timespec timestamp;
	if (clock_gettime(CLOCK_MONOTONIC,&timestamp) != 0)
		return 0u;
	return (uint64_t)timestamp.tv_sec * 1000000000ull +
		(uint64_t)timestamp.tv_nsec;
}

static SparkStatus SparkGlm52Pp13BuilderFinalizePackedMtpVerify(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	SparkModelDriverCompletionFunction completion_function,
	void *completion_context)
{
	SparkGlm52Pp13BuilderLayer *final_layer;
	SparkModelDriverCompletion completion;
	uint64_t element_count;
	uint64_t profile_entry_ns;
	uint64_t profile_readback_ns;
	uint64_t profile_rebase_ns;
	uint64_t profile_draft_ns;
	uint64_t profile_emit_ns;
	uint32_t accepted_draft_count;
	uint32_t block_count;
	uint32_t execution_row_base;
	uint32_t lane_index;
	uint32_t next_draft_token_count;
	uint32_t request_slot_index;
	uint32_t token_index;
	SparkGlm52MtpTreeResolution resolution;
	SparkStatus status;
	if (state == 0 || work_packet == 0 ||
		SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet) == 0u ||
		work_packet->mtp_draft_token_count !=
			SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT ||
		state->captured_completion_valid == 0u || state->mtp_ready == 0u ||
		!SparkGlm52Pp13BuilderIsFinalRank(state) ||
		work_packet->execution_row_count == 0u ||
		work_packet->execution_row_count >
			state->rank_plan.execution_row_capacity ||
		state->mtp_previous_target_hidden == 0 ||
		state->mtp_previous_target_hidden_store == 0 ||
		state->device_mtp_request_slot_indices == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	final_layer = &state->layers[state->rank_plan.layer_count - 1u];
	if (final_layer->restricted_selected_token_ids == 0 ||
		final_layer->layer_output_hidden == 0 ||
		final_layer->final_norm_weight == 0 ||
		state->host_decode_result_token_ids == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	profile_entry_ns = state->mtp_cycle_profile_enabled != 0u
		? SparkGlm52Pp13BuilderMonotonicTimeNs() : 0u;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
		state->host_decode_result_token_ids,
		final_layer->restricted_selected_token_ids,
		(size_t)work_packet->execution_row_count * sizeof(uint32_t),
		cudaMemcpyDeviceToHost));
	profile_readback_ns = state->mtp_cycle_profile_enabled != 0u
		? SparkGlm52Pp13BuilderMonotonicTimeNs() : 0u;
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCompactExecutionKvRows(
			state,work_packet->lane_count,work_packet->rows_per_lane);
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderRebaseMtpTreeCache(
			state,work_packet);
	profile_rebase_ns = state->mtp_cycle_profile_enabled != 0u
		? SparkGlm52Pp13BuilderMonotonicTimeNs() : 0u;
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		const SparkGlm52Pp13WorkControlLane *lane;
		lane = &work_packet->lanes[lane_index];
		execution_row_base = lane_index * work_packet->rows_per_lane;
		status = SparkGlm52Pp13BuilderResolveMtpLane(
			state,work_packet,lane_index,&resolution);
		if (status != SPARK_STATUS_OK)
			return status;
		accepted_draft_count = resolution.accepted_token_count;
		state->host_decode_positions[lane_index] =
			execution_row_base + resolution.fallback_row_index;
		request_slot_index = lane->request_slot_index;
		if (request_slot_index >=
				state->configuration.maximum_resident_sequence_count ||
			lane->sequence_position > UINT64_MAX - accepted_draft_count)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		state->host_mtp_request_slot_indices[lane_index] = request_slot_index;

	}
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_decode_positions,state->host_decode_positions,
		(size_t)work_packet->lane_count * sizeof(uint32_t),
		cudaMemcpyHostToDevice,state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_mtp_request_slot_indices,
			state->host_mtp_request_slot_indices,
			(size_t)work_packet->lane_count * sizeof(uint32_t),
			cudaMemcpyHostToDevice,state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52Pp13BuilderSelectedTargetFinalNormKernel<<<
		work_packet->lane_count,SPARK_GLM52_PP13_BUILDER_THREADS,0u,state->stream>>>(
		(const uint16_t *)final_layer->layer_output_hidden,
		(const uint16_t *)final_layer->final_norm_weight,
		(const uint32_t *)state->device_decode_positions,
		(uint16_t *)state->mtp_previous_target_hidden,
		work_packet->lane_count,
		SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
	if (status != SPARK_STATUS_OK)
		return status;
	element_count = (uint64_t)work_packet->lane_count *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	block_count = (uint32_t)((element_count +
		SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS);
	SparkGlm52Pp13BuilderScatterMtpPreviousHiddenKernel<<<
		block_count,SPARK_GLM52_PP13_BUILDER_THREADS,0u,state->stream>>>(
		(const uint16_t *)state->mtp_previous_target_hidden,
		state->device_mtp_request_slot_indices,
		(uint16_t *)state->mtp_previous_target_hidden_store,
		work_packet->lane_count);
	status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
	if (status == SPARK_STATUS_OK)
	{
		for (lane_index = 0u;
			 lane_index < work_packet->lane_count;
			 ++lane_index)
		{
			const SparkGlm52Pp13WorkControlLane *lane;
			lane = &work_packet->lanes[lane_index];
			execution_row_base = lane_index * work_packet->rows_per_lane;
			status = SparkGlm52Pp13BuilderResolveMtpLane(
				state,work_packet,lane_index,&resolution);
			if (status != SPARK_STATUS_OK)
				return status;
			accepted_draft_count = resolution.accepted_token_count;
			request_slot_index = lane->request_slot_index;
			state->host_mtp_previous_sequence_ids[request_slot_index] =
				lane->sequence_id;
			state->host_mtp_previous_positions[request_slot_index] =
				lane->sequence_position + accepted_draft_count;
			state->host_mtp_previous_valid[request_slot_index] = 1u;
		}
		state->mtp_previous_request_id = work_packet->lanes[0u].request_id;
		state->mtp_previous_sequence_id = work_packet->lanes[0u].sequence_id;
		request_slot_index = work_packet->lanes[0u].request_slot_index;
		state->mtp_previous_position =
			state->host_mtp_previous_positions[request_slot_index];
		state->mtp_previous_valid = 1u;
	}
	if (status != SPARK_STATUS_OK)
		return status;
	next_draft_token_count = 0u;
	status = SparkGlm52Pp13BuilderLaunchVerifierMtpDraft(
		state,work_packet,&next_draft_token_count);
	profile_draft_ns = state->mtp_cycle_profile_enabled != 0u
		? SparkGlm52Pp13BuilderMonotonicTimeNs() : 0u;
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		const SparkGlm52Pp13WorkControlLane *lane;
		lane = &work_packet->lanes[lane_index];
		execution_row_base = lane_index * work_packet->rows_per_lane;
		status = SparkGlm52Pp13BuilderResolveMtpLane(
			state,work_packet,lane_index,&resolution);
		if (status != SPARK_STATUS_OK)
			return status;
		accepted_draft_count = resolution.accepted_token_count;
		if (accepted_draft_count > lane->speculative_token_count)
			return SPARK_STATUS_INTERNAL_ERROR;
		completion = state->captured_completion;
		completion.request_id = lane->request_id;
		completion.sequence_id = lane->sequence_id;
		completion.sequence_position = lane->sequence_position;
		completion.completion_flags |=
			SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS;
		completion.token_count = work_packet->rows_per_lane;
		completion.accepted_token_count = completion.token_count;
		for (token_index = 0u;
			 token_index < completion.token_count;
			 ++token_index)
			completion.token_ids[token_index] =
				state->host_decode_result_token_ids[
					execution_row_base + token_index];
		for (token_index = completion.token_count;
			 token_index < SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY;
			 ++token_index)
			completion.token_ids[token_index] = 0u;
		completion.completion_flags |=
			SPARK_MODEL_DRIVER_COMPLETION_FLAG_DRAFT_TOKEN_IDS;
		completion.draft_token_count = next_draft_token_count;
		for (token_index = 0u;
			 token_index < next_draft_token_count;
			 ++token_index)
			completion.draft_token_ids[token_index] =
				state->host_mtp_committed_token_ids[
					((uint64_t)lane_index *
					 SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT) +
					token_index];
		for (token_index = next_draft_token_count;
			 token_index < SPARK_MODEL_DRIVER_COMPLETION_DRAFT_TOKEN_CAPACITY;
			 ++token_index)
			completion.draft_token_ids[token_index] = 0u;
		if (completion_function != 0)
			completion_function(completion_context,&completion);
	}
	if (status == SPARK_STATUS_OK &&
		state->mtp_cycle_profile_enabled != 0u &&
		profile_entry_ns != 0u && profile_readback_ns != 0u &&
		profile_rebase_ns != 0u && profile_draft_ns != 0u)
	{
		profile_emit_ns = SparkGlm52Pp13BuilderMonotonicTimeNs();
		fprintf(stderr,
			"mtp_cycle_profile lanes=%u rows=%u "
			"verify_wait_ns=%llu rebase_submit_ns=%llu "
			"draft_chain_ns=%llu completion_emit_ns=%llu "
			"epilogue_total_ns=%llu\n",
			work_packet->lane_count,work_packet->rows_per_lane,
			(unsigned long long)(profile_readback_ns - profile_entry_ns),
			(unsigned long long)(profile_rebase_ns - profile_readback_ns),
			(unsigned long long)(profile_draft_ns - profile_rebase_ns),
			(unsigned long long)(profile_emit_ns - profile_draft_ns),
			(unsigned long long)(profile_emit_ns - profile_entry_ns));
	}
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderCopyVerifierTokenIds(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkGlm52Pp13BuilderLayer *final_layer;

	if (state == 0 || work_packet == 0 ||
		work_packet->execution_row_count == 0u ||
		work_packet->execution_row_count >
			state->rank_plan.execution_row_capacity ||
		state->host_decode_result_token_ids == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	final_layer = &state->layers[state->rank_plan.layer_count - 1u];
	if (final_layer->restricted_selected_token_ids == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	return SparkGlm52Pp13BuilderCudaStatus(cudaMemcpy(
		state->host_decode_result_token_ids,
		final_layer->restricted_selected_token_ids,
		(size_t)work_packet->execution_row_count * sizeof(uint32_t),
		cudaMemcpyDeviceToHost));
}

static SparkStatus SparkGlm52Pp13BuilderAcquireDsparkLane(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlLane *work_lane,
	uint32_t *backend_lane_index_out)
{
	SparkGlm52DsparkDraftBackendLaneState *lane_state;
	uint32_t backend_lane_index;

	if (state == 0 || work_lane == 0 || backend_lane_index_out == 0 ||
		state->dspark_backend_ready == 0u ||
		work_lane->request_slot_index >=
			state->configuration.maximum_resident_sequence_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	backend_lane_index =
		state->dspark_lane_by_request_slot[work_lane->request_slot_index];
	if (backend_lane_index != SPARK_GLM52_PP13_BUILDER_INVALID_SLOT)
	{
		if (backend_lane_index >= state->dspark_backend.maximum_lane_count ||
			state->dspark_request_slot_by_lane[backend_lane_index] !=
				work_lane->request_slot_index)
			return SPARK_STATUS_INTERNAL_ERROR;
		lane_state = &state->dspark_backend.lane_states[backend_lane_index];
		if (lane_state->staged != 0u &&
			lane_state->sequence_id != work_lane->sequence_id)
			return SPARK_STATUS_VALIDATION_FAILED;
		*backend_lane_index_out = backend_lane_index;
		return SPARK_STATUS_OK;
	}
	for (backend_lane_index = 0u;
		 backend_lane_index < state->dspark_backend.maximum_lane_count;
		 ++backend_lane_index)
	{
		if (state->dspark_request_slot_by_lane[backend_lane_index] ==
			SPARK_GLM52_PP13_BUILDER_INVALID_SLOT)
			break;
	}
	if (backend_lane_index == state->dspark_backend.maximum_lane_count)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	state->dspark_lane_by_request_slot[work_lane->request_slot_index] =
		backend_lane_index;
	state->dspark_request_slot_by_lane[backend_lane_index] =
		work_lane->request_slot_index;
	memset(&state->dspark_backend.lane_states[backend_lane_index],0,
		sizeof(state->dspark_backend.lane_states[backend_lane_index]));
	memset(&state->dspark_backend.validation_lane_states[backend_lane_index],0,
		sizeof(state->dspark_backend.validation_lane_states[backend_lane_index]));
	*backend_lane_index_out = backend_lane_index;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderReleaseDsparkLane(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlLane *work_lane)
{
	uint32_t backend_lane_index;

	if (state == 0 || work_lane == 0 ||
		work_lane->request_slot_index >=
			state->configuration.maximum_resident_sequence_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->dspark_backend_ready == 0u)
		return SPARK_STATUS_OK;
	backend_lane_index =
		state->dspark_lane_by_request_slot[work_lane->request_slot_index];
	if (backend_lane_index == SPARK_GLM52_PP13_BUILDER_INVALID_SLOT)
		return SPARK_STATUS_OK;
	if (backend_lane_index >= state->dspark_backend.maximum_lane_count ||
		state->dspark_request_slot_by_lane[backend_lane_index] !=
			work_lane->request_slot_index)
		return SPARK_STATUS_INTERNAL_ERROR;
	if (state->dspark_backend.lane_states[backend_lane_index].staged != 0u &&
		state->dspark_backend.lane_states[backend_lane_index].sequence_id !=
			work_lane->sequence_id)
		return SPARK_STATUS_VALIDATION_FAILED;
	memset(&state->dspark_backend.lane_states[backend_lane_index],0,
		sizeof(state->dspark_backend.lane_states[backend_lane_index]));
	memset(&state->dspark_backend.validation_lane_states[backend_lane_index],0,
		sizeof(state->dspark_backend.validation_lane_states[backend_lane_index]));
	state->dspark_request_slot_by_lane[backend_lane_index] =
		SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	state->dspark_lane_by_request_slot[work_lane->request_slot_index] =
		SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderReleaseDsparkPacketLanes(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	uint32_t lane_index;
	SparkStatus status;

	if (state == 0 || work_packet == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		status = SparkGlm52Pp13BuilderReleaseDsparkLane(
			state,&work_packet->lanes[lane_index]);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderAppendDsparkStage(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlLane *work_lane,
	uint32_t tap_row_index,
	uint32_t token_id,
	uint64_t sequence_position)
{
	SparkGlm52DsparkDraftBackendStage *stage;
	uint32_t backend_lane_index;
	SparkStatus status;

	if (state == 0 || work_lane == 0 ||
		state->dspark_stage_count >= state->dspark_backend.maximum_tap_row_count ||
		tap_row_index >= state->rank_plan.execution_row_capacity ||
		sequence_position == 0u ||
		sequence_position >
			state->configuration.dspark_maximum_context_token_count)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	status = SparkGlm52Pp13BuilderAcquireDsparkLane(
		state,work_lane,&backend_lane_index);
	if (status != SPARK_STATUS_OK)
		return status;
	stage = &state->dspark_stage_batch[state->dspark_stage_count++];
	memset(stage,0,sizeof(*stage));
	stage->sequence_id = work_lane->sequence_id;
	stage->sequence_position = sequence_position;
	stage->tap_generation = sequence_position;
	stage->tap_row_index = tap_row_index;
	stage->backend_lane_index = backend_lane_index;
	stage->token_id = token_id;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderPreparePrefillDsparkStages(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	const SparkGlm52Pp13WorkControlLane *lane;
	uint32_t lane_index;
	uint32_t row_index;
	uint32_t tap_row_index;
	uint64_t sequence_position;
	SparkStatus status;

	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		lane = &work_packet->lanes[lane_index];
		for (row_index = 0u; row_index < work_packet->rows_per_lane; ++row_index)
		{
			tap_row_index = (lane_index * work_packet->rows_per_lane) + row_index;
			if (lane->sequence_position > UINT64_MAX - row_index - 1u)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			sequence_position = lane->sequence_position + row_index + 1u;
			status = SparkGlm52Pp13BuilderAppendDsparkStage(
				state,lane,tap_row_index,
				work_packet->prefill_token_ids[tap_row_index],
				sequence_position);
			if (status != SPARK_STATUS_OK)
				return status;
		}
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderPrepareDecodeDsparkStages(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	const SparkGlm52Pp13WorkControlLane *lane;
	SparkGlm52MtpTreeResolution resolution;
	uint32_t lane_index,row_count,row_index,tap_row_index,verifier_row_index;
	uint64_t sequence_position;
	SparkStatus status;

	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		lane = &work_packet->lanes[lane_index];
		row_count = 1u;
		if (SparkGlm52Pp13BuilderWorkIsSpeculativeVerify(work_packet) != 0u)
		{
			status = SparkGlm52Pp13BuilderResolveMtpLane(
				state,work_packet,lane_index,&resolution);
			if (status != SPARK_STATUS_OK)
				return status;
			row_count = resolution.committed_token_count;
		}
		for (row_index = 0u; row_index < row_count; ++row_index)
		{
			verifier_row_index = row_index;
			if (SparkGlm52Pp13BuilderWorkIsSpeculativeVerify(work_packet) != 0u &&
				row_index == resolution.accepted_token_count)
				verifier_row_index = resolution.fallback_row_index;
			if (verifier_row_index >= work_packet->rows_per_lane)
				return SPARK_STATUS_INTERNAL_ERROR;
			tap_row_index = (lane_index * work_packet->rows_per_lane) +
				verifier_row_index;
			if (lane->sequence_position > UINT64_MAX - row_index - 1u)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			sequence_position = lane->sequence_position + row_index + 1u;
			status = SparkGlm52Pp13BuilderAppendDsparkStage(
				state,lane,tap_row_index,
				state->host_decode_result_token_ids[tap_row_index],
				sequence_position);
			if (status != SPARK_STATUS_OK)
				return status;
		}
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderPrepareDsparkDraftBatch(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	const SparkGlm52DsparkDraftBackendLaneState *lane_state;
	const SparkGlm52Pp13WorkControlLane *work_lane;
	SparkGlm52DsparkDraftRequest *request;
	uint32_t backend_lane_index;
	uint32_t lane_index;
	SparkStatus status;

	state->dspark_draft_count = 0u;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		work_lane = &work_packet->lanes[lane_index];
		status = SparkGlm52Pp13BuilderAcquireDsparkLane(
			state,work_lane,&backend_lane_index);
		if (status != SPARK_STATUS_OK)
			return status;
		lane_state = &state->dspark_backend.lane_states[backend_lane_index];
		if (lane_state->staged == 0u ||
			lane_state->sequence_id != work_lane->sequence_id)
			return SPARK_STATUS_VALIDATION_FAILED;
		request = &state->dspark_draft_batch[state->dspark_draft_count++];
		memset(request,0,sizeof(*request));
		request->abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
		request->descriptor_bytes =
			SPARK_GLM52_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES;
		request->requested_token_count =
			SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
		request->active_sequence_index = backend_lane_index;
		request->priority = work_packet->priority;
		request->request_id = work_lane->request_id;
		request->sequence_id = work_lane->sequence_id;
		request->sequence_position = lane_state->sequence_position;
		request->tap_generation = lane_state->tap_generation;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLaunchDsparkBatch(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkStatus status;

	if (state == 0 || work_packet == 0 || state->dspark_backend_ready == 0u ||
		state->dspark_ready_draft_count != 0u ||
		state->dspark_stage_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52DsparkDraftBackendStageBatch(
		&state->dspark_backend,state->dspark_stage_batch,
		state->dspark_stage_count);
	if (status == SPARK_STATUS_OK &&
		(work_packet->flags &
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) == 0u)
	{
		status = SparkGlm52Pp13BuilderPrepareDsparkDraftBatch(
			state,work_packet);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52DsparkDraftBackendLaunchDraftBatch(
				&state->dspark_backend,state->dspark_draft_batch,
				state->dspark_draft_count);
	}
	if (status == SPARK_STATUS_OK)
		__atomic_store_n(
			&state->pending_finalizer_state,
			SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_GPU_PENDING,
			__ATOMIC_RELEASE);
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderEmitPackedDsparkVerifyCompletions(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	SparkModelDriverCompletionFunction completion_function,
	void *completion_context)
{
	const SparkGlm52Pp13WorkControlLane *lane;
	SparkModelDriverCompletion completion;
	uint32_t execution_row_base;
	uint32_t lane_index;
	uint32_t token_index;

	if (state == 0 || work_packet == 0 ||
		SparkGlm52Pp13BuilderWorkIsDsparkVerify(work_packet) == 0u ||
		state->captured_completion_valid == 0u ||
		!SparkGlm52Pp13BuilderIsFinalRank(state) ||
		work_packet->rows_per_lane <= 1u ||
		work_packet->rows_per_lane >
			SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		lane = &work_packet->lanes[lane_index];
		execution_row_base = lane_index * work_packet->rows_per_lane;
		completion = state->captured_completion;
		completion.request_id = lane->request_id;
		completion.sequence_id = lane->sequence_id;
		completion.sequence_position = lane->sequence_position;
		completion.completion_flags |=
			SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS;
		completion.completion_flags &=
			~SPARK_MODEL_DRIVER_COMPLETION_FLAG_DRAFT_TOKEN_IDS;
		completion.token_count = work_packet->rows_per_lane;
		completion.accepted_token_count = completion.token_count;
		completion.draft_token_count = 0u;
		memset(completion.draft_token_ids,0,sizeof(completion.draft_token_ids));
		for (token_index = 0u; token_index < completion.token_count; ++token_index)
			completion.token_ids[token_index] =
				state->host_decode_result_token_ids[
					execution_row_base + token_index];
		for (token_index = completion.token_count;
			 token_index < SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY;
			 ++token_index)
			completion.token_ids[token_index] = 0u;
		if (completion_function != 0)
			completion_function(completion_context,&completion);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderPrepareDsparkStages(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	if (state == 0 || work_packet == 0 ||
		SparkGlm52Pp13BuilderWorkCapturesDspark(work_packet) == 0u ||
		state->dspark_backend_ready == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state->dspark_stage_count = 0u;
	state->dspark_draft_count = 0u;
	if ((work_packet->flags &
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u)
		return SparkGlm52Pp13BuilderPreparePrefillDsparkStages(
			state,work_packet);
	return SparkGlm52Pp13BuilderPrepareDecodeDsparkStages(
		state,work_packet);
}

static SparkStatus SparkGlm52Pp13BuilderQueueDsparkResults(
	SparkGlm52Pp13BuilderState *state,
	uint32_t result_count)
{
	uint32_t lane_capacity;
	uint32_t result_index;
	uint32_t tail_index;

	if (state == 0 || result_count != state->dspark_draft_count)
		return SPARK_STATUS_VALIDATION_FAILED;
	lane_capacity = state->dspark_backend.maximum_lane_count;
	if (result_count > lane_capacity - state->dspark_ready_draft_count)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	for (result_index = 0u; result_index < result_count; ++result_index)
	{
		tail_index = (state->dspark_ready_draft_head +
			state->dspark_ready_draft_count) % lane_capacity;
		state->dspark_ready_drafts[tail_index] =
			state->dspark_batch_results[result_index];
		state->dspark_ready_draft_count += 1u;
		SparkGlm52Pp13BuilderTraceDsparkDraft(
			&state->dspark_draft_batch[result_index],
			&state->dspark_batch_results[result_index]);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderStoreMtpPreviousTarget(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkGlm52Pp13BuilderLayer *final_layer;
	uint64_t element_count;
	uint64_t target_position;
	uint32_t block_count;
	uint32_t lane_index;
	SparkStatus status;
	if (state == 0 || work_packet == 0 || state->mtp_ready == 0u ||
		!SparkGlm52Pp13BuilderIsFinalRank(state))
		return SPARK_STATUS_OK;
	if (state->mtp_previous_target_hidden == 0 ||
		state->mtp_previous_target_hidden_store == 0 ||
		state->device_mtp_request_slot_indices == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	final_layer = &state->layers[state->rank_plan.layer_count - 1u];
	if (final_layer->final_norm_weight == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		const SparkGlm52Pp13WorkControlLane *lane;
		lane = &work_packet->lanes[lane_index];
		target_position =
			lane->sequence_position + work_packet->rows_per_lane - 1u;
		if (lane->request_slot_index >=
				state->configuration.maximum_resident_sequence_count ||
			target_position > UINT32_MAX)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		state->host_mtp_request_slot_indices[lane_index] =
			lane->request_slot_index;
		state->host_decode_positions[lane_index] =
			(uint32_t)target_position;
	}
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_mtp_request_slot_indices,
		state->host_mtp_request_slot_indices,
		(size_t)work_packet->lane_count * sizeof(uint32_t),
		cudaMemcpyHostToDevice,state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->mtp_base_positions,state->host_decode_positions,
			(size_t)work_packet->lane_count * sizeof(uint32_t),
			cudaMemcpyHostToDevice,state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
		state->host_decode_positions[lane_index] =
			(lane_index * work_packet->rows_per_lane) +
			work_packet->rows_per_lane - 1u;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_decode_positions,state->host_decode_positions,
		(size_t)work_packet->lane_count * sizeof(uint32_t),
		cudaMemcpyHostToDevice,state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52Pp13BuilderSelectedTargetFinalNormKernel<<<
		work_packet->lane_count,
		SPARK_GLM52_PP13_BUILDER_THREADS,
		0u,
		state->stream>>>(
		(const uint16_t *)final_layer->layer_output_hidden,
		(const uint16_t *)final_layer->final_norm_weight,
		(const uint32_t *)state->device_decode_positions,
		(uint16_t *)state->mtp_previous_target_hidden,
		work_packet->lane_count,
		SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
	if (status != SPARK_STATUS_OK)
		return status;
	element_count = (uint64_t)work_packet->lane_count *
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	block_count = (uint32_t)((element_count +
		SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS);
	SparkGlm52Pp13BuilderScatterMtpPreviousHiddenKernel<<<
		block_count,SPARK_GLM52_PP13_BUILDER_THREADS,0u,state->stream>>>(
		(const uint16_t *)state->mtp_previous_target_hidden,
		state->device_mtp_request_slot_indices,
		(uint16_t *)state->mtp_previous_target_hidden_store,
		work_packet->lane_count);
	status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		uint32_t request_slot_index;
		request_slot_index = work_packet->lanes[lane_index].request_slot_index;
		state->host_mtp_previous_sequence_ids[request_slot_index] =
			work_packet->lanes[lane_index].sequence_id;
		state->host_mtp_previous_positions[request_slot_index] =
			(uint32_t)(work_packet->lanes[lane_index].sequence_position +
				work_packet->rows_per_lane - 1u);
		state->host_mtp_previous_valid[request_slot_index] = 1u;
	}
	state->mtp_previous_request_id = work_packet->lanes[0u].request_id;
	state->mtp_previous_sequence_id = work_packet->lanes[0u].sequence_id;
	state->mtp_previous_position =
		state->host_mtp_previous_positions[
			work_packet->lanes[0u].request_slot_index];
	state->mtp_previous_valid = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderBuildMtpPrefillRows(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	uint32_t *row_count_out)
{
	const SparkGlm52Pp13WorkControlLane *lane;
	uint64_t current_position;
	uint32_t current_row_index,lane_index,row_count,row_offset,row_start;
	if (state == 0 || work_packet == 0 || row_count_out == 0 ||
		state->host_decode_positions == 0 ||
		state->host_decode_token_ids == 0 ||
		state->host_decode_result_token_ids == 0 ||
		state->host_mtp_request_slot_indices == 0 ||
		state->host_mtp_previous_sequence_ids == 0 ||
		state->host_mtp_previous_positions == 0 ||
		state->host_mtp_previous_valid == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	row_count = 0u;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		lane = &work_packet->lanes[lane_index];
		if (lane->request_slot_index >=
			state->configuration.maximum_resident_sequence_count)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		row_start = lane->sequence_position >=
			SPARK_GLM52_MODEL_MTP_TARGET_HIDDEN_POSITION_DELTA
			? 0u : (uint32_t)(
				SPARK_GLM52_MODEL_MTP_TARGET_HIDDEN_POSITION_DELTA -
				lane->sequence_position);
		if (row_start == 0u &&
			(state->host_mtp_previous_valid[lane->request_slot_index] == 0u ||
			 state->host_mtp_previous_sequence_ids[lane->request_slot_index] !=
				lane->sequence_id ||
			 state->host_mtp_previous_positions[lane->request_slot_index] ==
				UINT64_MAX ||
			 state->host_mtp_previous_positions[lane->request_slot_index] +
				SPARK_GLM52_MODEL_MTP_TARGET_HIDDEN_POSITION_DELTA !=
				lane->sequence_position))
			return SPARK_STATUS_NOT_FOUND;
		for (row_offset = row_start;
			 row_offset < work_packet->rows_per_lane;
			 ++row_offset)
		{
			if (row_count >= state->rank_plan.execution_row_capacity ||
				lane->sequence_position > UINT64_MAX - row_offset)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			current_position = lane->sequence_position + row_offset;
			if (current_position <
					SPARK_GLM52_MODEL_MTP_TARGET_HIDDEN_POSITION_DELTA ||
				current_position > UINT32_MAX)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			current_row_index =
				(lane_index * work_packet->rows_per_lane) + row_offset;
			state->host_decode_positions[row_count] = row_offset == 0u
				? UINT32_MAX : current_row_index -
					SPARK_GLM52_MODEL_MTP_TARGET_HIDDEN_POSITION_DELTA;
			state->host_decode_token_ids[row_count] =
				work_packet->prefill_token_ids[current_row_index];
			state->host_decode_result_token_ids[row_count] = lane_index;
			state->host_mtp_request_slot_indices[row_count] =
				lane->request_slot_index;
			row_count += 1u;
		}
	}
	*row_count_out = row_count;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderUploadMtpPrefillRows(
	SparkGlm52Pp13BuilderState *state,
	uint32_t row_count)
{
	SparkGlm52Pp13BuilderLayer *final_layer;
	uint64_t block_table_elements;
	uint32_t block_count;
	SparkStatus status;
	if (state == 0 || row_count == 0u ||
		row_count > state->rank_plan.execution_row_capacity ||
		state->mtp_prefill_block_table == 0 ||
		state->active_kv_block_count == 0u ||
		state->active_kv_block_count >
			SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE ||
		state->device_decode_positions == 0 ||
		state->device_decode_token_ids == 0 ||
		state->device_mtp_request_slot_indices == 0 ||
		state->mtp_previous_target_hidden == 0 ||
		state->mtp_previous_target_hidden_store == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	final_layer = &state->layers[state->rank_plan.layer_count - 1u];
	if (final_layer->layer_output_hidden == 0 ||
		final_layer->final_norm_weight == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_decode_positions,state->host_decode_positions,
		(size_t)row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_decode_token_ids,state->host_decode_token_ids,
			(size_t)row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_mtp_request_slot_indices,
			state->host_mtp_request_slot_indices,
			(size_t)row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52Pp13BuilderPrepareMtpPrefillHiddenKernel<<<
		row_count,SPARK_GLM52_PP13_BUILDER_THREADS,0u,state->stream>>>(
		(const uint16_t *)final_layer->layer_output_hidden,
		(const uint16_t *)final_layer->final_norm_weight,
		(const uint16_t *)state->mtp_previous_target_hidden_store,
		(const uint32_t *)state->device_decode_positions,
		state->device_mtp_request_slot_indices,
		(uint16_t *)state->mtp_previous_target_hidden,
		row_count,SPARK_GLM52_MODEL_RMS_NORM_EPSILON);
	status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_mtp_request_slot_indices,
			state->host_decode_result_token_ids,
			(size_t)row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	block_table_elements = (uint64_t)row_count * state->active_kv_block_count;
	block_count = (uint32_t)((block_table_elements +
		SPARK_GLM52_PP13_BUILDER_THREADS - 1u) /
		SPARK_GLM52_PP13_BUILDER_THREADS);
	SparkGlm52Pp13BuilderExpandMtpPrefillBlockTableKernel<<<
		block_count,SPARK_GLM52_PP13_BUILDER_THREADS,0u,state->stream>>>(
		state->device_physical_block_indices,
		state->device_mtp_request_slot_indices,
		state->mtp_prefill_block_table,
		SPARK_GLM52_PP13_BUILDER_MAX_BLOCKS_PER_SEQUENCE,
		state->active_kv_block_count,row_count);
	return SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
}

static SparkStatus SparkGlm52Pp13BuilderUploadMtpPrefillBasePositions(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	uint32_t row_count)
{
	const SparkGlm52Pp13WorkControlLane *lane;
	uint64_t current_position;
	uint32_t lane_index,row_index,row_offset,row_start;
	if (state == 0 || work_packet == 0 || row_count == 0u ||
		row_count > state->rank_plan.execution_row_capacity ||
		state->mtp_base_positions == 0 ||
		state->host_decode_positions == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	row_index = 0u;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		lane = &work_packet->lanes[lane_index];
		row_start = lane->sequence_position >=
			SPARK_GLM52_MODEL_MTP_TARGET_HIDDEN_POSITION_DELTA
			? 0u : (uint32_t)(
				SPARK_GLM52_MODEL_MTP_TARGET_HIDDEN_POSITION_DELTA -
				lane->sequence_position);
		for (row_offset = row_start;
			 row_offset < work_packet->rows_per_lane;
			 ++row_offset)
		{
			current_position = lane->sequence_position + row_offset;
			if (row_index >= row_count ||
				current_position <
					SPARK_GLM52_MODEL_MTP_TARGET_HIDDEN_POSITION_DELTA ||
				current_position > UINT32_MAX)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			state->host_decode_positions[row_index++] =
				(uint32_t)(current_position -
					SPARK_GLM52_MODEL_MTP_TARGET_HIDDEN_POSITION_DELTA);
		}
	}
	if (row_index != row_count)
		return SPARK_STATUS_INTERNAL_ERROR;
	return SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->mtp_base_positions,state->host_decode_positions,
		(size_t)row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,state->stream));
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpPrefillRows(
	SparkGlm52Pp13BuilderState *state,
	uint32_t row_count)
{
	const uint32_t *block_table;
	SparkStatus status;
	status = SparkGlm52Pp13BuilderPrepareMtpLinearPlanRows(state,row_count);
	if (status != SPARK_STATUS_OK)
		return status;
	block_table = state->mtp_layer.slot.block_table;
	state->mtp_layer.slot.block_table = state->mtp_prefill_block_table;
	status = SparkGlm52Pp13BuilderLaunchMtpLayer(
		state,(const uint32_t *)state->device_decode_token_ids,
		state->mtp_base_positions,state->mtp_previous_target_hidden,
		0u,row_count,state->stream);
	state->mtp_layer.slot.block_table = block_table;
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderPrefillMtpPreviousTarget(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	uint32_t row_count;
	SparkStatus status;
	if (state == 0 || work_packet == 0 || state->mtp_ready == 0u ||
		!SparkGlm52Pp13BuilderIsFinalRank(state) ||
		(work_packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) == 0u)
		return SPARK_STATUS_OK;
	status = SparkGlm52Pp13BuilderBuildMtpPrefillRows(
		state,work_packet,&row_count);
	if (status == SPARK_STATUS_OK && row_count != 0u)
		status = SparkGlm52Pp13BuilderUploadMtpPrefillRows(state,row_count);
	if (status == SPARK_STATUS_OK && row_count != 0u)
		status = SparkGlm52Pp13BuilderUploadMtpPrefillBasePositions(
			state,work_packet,row_count);
	if (status == SPARK_STATUS_OK && row_count != 0u)
		status = SparkGlm52Pp13BuilderLaunchMtpPrefillRows(state,row_count);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52Pp13BuilderStoreMtpPreviousTarget(state,work_packet);
}

static void SparkGlm52Pp13BuilderEmitPendingWorkFailure(
	SparkGlm52Pp13BuilderState *state,
	const SparkModelDriverCompletion *source_completion,
	SparkStatus failure_status)
{
	SparkModelDriverCompletion completion;
	uint32_t completion_count;
	uint32_t lane_index;

	if (state == 0 || state->pending_work_completion_function == 0)
		return;
	if (source_completion != 0)
		completion = *source_completion;
	else
		memset(&completion,0,sizeof(completion));
	completion.status = failure_status;
	completion.completion_flags &=
		~SPARK_MODEL_DRIVER_COMPLETION_FLAG_TOKEN_IDS;
	completion.token_count = 0u;
	completion.accepted_token_count = 0u;
	memset(completion.token_ids,0,sizeof(completion.token_ids));
	completion_count = SparkGlm52Pp13BuilderIsFinalRank(state)
		? state->pending_work_packet.lane_count : 1u;
	for (lane_index = 0u; lane_index < completion_count; ++lane_index)
	{
		completion.request_id =
			state->pending_work_packet.lanes[lane_index].request_id;
		completion.sequence_id =
			state->pending_work_packet.lanes[lane_index].sequence_id;
		completion.sequence_position =
			state->pending_work_packet.lanes[lane_index].sequence_position;
		state->pending_work_completion_function(
			state->pending_work_completion_context,&completion);
	}
}

static void SparkGlm52Pp13BuilderCollectPendingWorkCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *completion)
{
	SparkGlm52Pp13BuilderState *state;

	state = (SparkGlm52Pp13BuilderState *)completion_context;
	if (state == 0 || completion == 0 ||
		state->pending_work_completions == 0)
		return;
	if (state->pending_work_completion_count >=
		state->rank_plan.logical_lane_capacity)
	{
		state->pending_work_completion_overflow = 1u;
		return;
	}
	state->pending_work_completions[
		state->pending_work_completion_count++] = *completion;
}

static SparkStatus SparkGlm52Pp13BuilderDiscardMtpKvTransactions(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet);

static SparkGlm52Pp13BuilderMtpKvTransaction *
SparkGlm52Pp13BuilderMtpKvTransactionForLane(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlLane *lane);

static SparkStatus SparkGlm52Pp13BuilderSealMtpTreeShadows(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkGlm52Pp13BuilderMtpKvTransaction *transaction;
	uint32_t lane_index;
	SparkStatus status;
	if (state == 0 || work_packet == 0 ||
		SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet) == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		transaction = SparkGlm52Pp13BuilderMtpKvTransactionForLane(
			state,&work_packet->lanes[lane_index]);
		if (transaction == 0 || transaction->active == 0u ||
			transaction->tree_verify == 0u ||
			transaction->shadow_valid == 0u)
			return SPARK_STATUS_INTERNAL_ERROR;
		while (transaction->transient_block_count != 0u)
		{
			status = SparkGlm52Pp13WorkControlReleaseTransientPhysicalBlock(
				&state->kv_state,
				transaction->transient_physical_blocks[
					transaction->transient_block_count - 1u]);
			if (status != SPARK_STATUS_OK)
				return status;
			transaction->transient_block_count -= 1u;
		}
	}
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13BuilderCompletePendingWork(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SparkGlm52Pp13BuilderState *state;

	state = (SparkGlm52Pp13BuilderState *)completion_context;
	if (state == 0 || driver_completion == 0 ||
		state->pending_work_active == 0u)
		return;
	state->pending_driver_completion = *driver_completion;
	__atomic_store_n(
		&state->pending_finalizer_state,
		SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_READY,
		__ATOMIC_RELEASE);
}

static SparkStatus SparkGlm52Pp13BuilderFinishPendingWork(
	SparkGlm52Pp13BuilderState *state,
	const SparkModelDriverCompletion *source_completion,
	SparkStatus status)
{
	const SparkGlm52Pp13WorkControlPacket *work_packet;
	SparkModelDriverCompletionFunction completion_function;
	void *outer_completion_context;
	uint32_t completion_index;

	if (state == 0 || source_completion == 0 ||
		state->pending_work_active == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	work_packet = &state->pending_work_packet;
	completion_function = state->pending_work_completion_function;
	outer_completion_context = state->pending_work_completion_context;
	if (status != SPARK_STATUS_OK &&
		SparkGlm52Pp13BuilderWorkIsMtpVerify(work_packet) != 0u)
	{
		SparkStatus discard_status;
		discard_status = SparkGlm52Pp13BuilderDiscardMtpKvTransactions(
			state,work_packet);
		if (discard_status != SPARK_STATUS_OK)
			status = discard_status;
	}
	if (status != SPARK_STATUS_OK &&
		SparkGlm52Pp13BuilderWorkCapturesDspark(work_packet) != 0u)
		(void)SparkGlm52Pp13BuilderReleaseDsparkPacketLanes(
			state,work_packet);
	if (status == SPARK_STATUS_OK)
	{
		status = SparkGlm52Pp13WorkControlCommitHostKvBlockTable(
			work_packet,&state->kv_state);
		if (status != SPARK_STATUS_OK)
			(void)SparkGlm52Pp13WorkControlCancelHostKvBlockTable(
				work_packet,&state->kv_state);
	}
	else
		(void)SparkGlm52Pp13WorkControlCancelHostKvBlockTable(
			work_packet,&state->kv_state);
	state->pending_work_active = 0u;
	__atomic_store_n(
		&state->pending_finalizer_state,
		SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_NONE,
		__ATOMIC_RELEASE);
	state->asynchronous_completion_count += 1u;
	if (SparkGlm52Pp13BuilderWorkIsSpeculativeVerify(work_packet) != 0u)
	{
		if (status == SPARK_STATUS_OK)
			state->layer_major_completion_count += 1u;
		else
			state->layer_major_failure_count += 1u;
	}
	if (status != SPARK_STATUS_OK)
	{
		state->asynchronous_failure_count += 1u;
		SparkGlm52Pp13BuilderEmitPendingWorkFailure(
			state,source_completion,status);
	}
	else if (completion_function != 0 &&
		state->pending_work_completion_count != 0u)
	{
		for (completion_index = 0u;
			 completion_index < state->pending_work_completion_count;
			 ++completion_index)
			completion_function(
				outer_completion_context,
				&state->pending_work_completions[completion_index]);
	}
	else if (completion_function != 0)
	{
		if (SparkGlm52Pp13BuilderWorkNeedsCapturedCompletion(work_packet) &&
			SparkGlm52Pp13BuilderIsFinalRank(state))
			completion_function(
				outer_completion_context,&state->captured_completion);
		else
			completion_function(outer_completion_context,source_completion);
	}
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderStartPendingWorkFinalizer(
	SparkGlm52Pp13BuilderState *state)
{
	const SparkGlm52Pp13WorkControlPacket *work_packet;
	uint32_t buffered_completion_required;
	SparkStatus status;

	if (state == 0 || state->pending_work_active == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	work_packet = &state->pending_work_packet;
	SparkGlm52Pp13BuilderCaptureCompletion(
		state,&state->pending_driver_completion);
	status = state->pending_driver_completion.status;
	buffered_completion_required = 0u;
	if (status == SPARK_STATUS_OK &&
		SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet) != 0u)
		status = SparkGlm52Pp13BuilderSealMtpTreeShadows(
			state,work_packet);
	if (status == SPARK_STATUS_OK &&
		(work_packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u &&
		work_packet->rows_per_lane > 1u)
		status = SparkGlm52Pp13BuilderCompactExecutionKvRows(
			state,work_packet->lane_count,work_packet->rows_per_lane);
	if (status == SPARK_STATUS_OK &&
		(work_packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u)
		status = SparkGlm52Pp13BuilderPrefillMtpPreviousTarget(
			state,work_packet);
	else if (status == SPARK_STATUS_OK &&
		SparkGlm52Pp13BuilderWorkIsSpeculativeVerify(work_packet) == 0u)
		status = SparkGlm52Pp13BuilderStoreMtpPreviousTarget(
			state,work_packet);
	if (status == SPARK_STATUS_OK &&
		SparkGlm52Pp13BuilderWorkIsMtpVerify(work_packet) != 0u &&
		work_packet->rows_per_lane > 1u &&
		SparkGlm52Pp13BuilderIsFinalRank(state))
	{
		status = SparkGlm52Pp13BuilderFinalizePackedMtpVerify(
			state,work_packet,
			SparkGlm52Pp13BuilderCollectPendingWorkCompletion,state);
		buffered_completion_required = 1u;
	}
	else if (status == SPARK_STATUS_OK &&
		SparkGlm52Pp13BuilderWorkIsDsparkVerify(work_packet) != 0u &&
		SparkGlm52Pp13BuilderIsFinalRank(state))
	{
		status = SparkGlm52Pp13BuilderCopyVerifierTokenIds(
			state,work_packet);
		if (status == SPARK_STATUS_OK)
			status =
				SparkGlm52Pp13BuilderEmitPackedDsparkVerifyCompletions(
					state,work_packet,
					SparkGlm52Pp13BuilderCollectPendingWorkCompletion,state);
		buffered_completion_required = 1u;
	}
	else if (status == SPARK_STATUS_OK &&
		SparkGlm52Pp13BuilderIsFinalRank(state) &&
		(work_packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) == 0u)
	{
		status = SparkGlm52Pp13BuilderEmitWideDecodeCompletions(
			state,work_packet,
			SparkGlm52Pp13BuilderCollectPendingWorkCompletion,state);
		buffered_completion_required = 1u;
	}
	if (status == SPARK_STATUS_OK &&
		(state->pending_work_completion_overflow != 0u ||
		 (buffered_completion_required != 0u &&
		  state->pending_work_completion_count != work_packet->lane_count)))
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if (status == SPARK_STATUS_OK &&
		SparkGlm52Pp13BuilderWorkCapturesDspark(work_packet) != 0u &&
		SparkGlm52Pp13BuilderIsFinalRank(state))
	{
		status = SparkGlm52Pp13BuilderPrepareDsparkStages(
			state,work_packet);
		if (status == SPARK_STATUS_OK)
			status = SparkGlm52Pp13BuilderLaunchDsparkBatch(
				state,work_packet);
		if (status == SPARK_STATUS_OK)
			return SPARK_STATUS_BUSY;
	}
	return SparkGlm52Pp13BuilderFinishPendingWork(
		state,&state->pending_driver_completion,status);
}

static SparkStatus SparkGlm52Pp13BuilderProgress(void *builder_state)
{
	SparkGlm52Pp13BuilderState *state;
	uint32_t finalizer_state;
	uint32_t result_count;
	SparkStatus status;

	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	finalizer_state = __atomic_load_n(
		&state->pending_finalizer_state,__ATOMIC_ACQUIRE);
	if (finalizer_state == SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_NONE)
		return state->pending_work_active != 0u
			? SPARK_STATUS_BUSY : SPARK_STATUS_OK;
	if (finalizer_state == SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_READY)
		return SparkGlm52Pp13BuilderStartPendingWorkFinalizer(state);
	if (finalizer_state !=
		SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_GPU_PENDING)
		return SPARK_STATUS_INTERNAL_ERROR;
	result_count = 0u;
	status = SparkGlm52DsparkDraftBackendTakeBatchResults(
		&state->dspark_backend,state->dspark_batch_results,
		state->dspark_backend.maximum_lane_count,&result_count);
	if (status == SPARK_STATUS_BUSY)
		return status;
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderQueueDsparkResults(
			state,result_count);
	return SparkGlm52Pp13BuilderFinishPendingWork(
		state,&state->pending_driver_completion,status);
}

static SparkGlm52Pp13BuilderMtpKvTransaction *
SparkGlm52Pp13BuilderMtpKvTransactionForLane(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlLane *lane)
{
	if (state == 0 || lane == 0 || state->mtp_kv_transactions == 0 ||
		lane->request_slot_index >=
			state->configuration.maximum_resident_sequence_count)
		return 0;
	return &state->mtp_kv_transactions[lane->request_slot_index];
}

static SparkStatus SparkGlm52Pp13BuilderAcquireMtpShadowSlot(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderMtpKvTransaction *transaction)
{
	if (state == 0 || transaction == 0 ||
		state->mtp_shadow_free_indices == 0 ||
		state->mtp_shadow_free_count == 0u)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	state->mtp_shadow_free_count -= 1u;
	transaction->shadow_slot_index =
		state->mtp_shadow_free_indices[state->mtp_shadow_free_count];
	transaction->shadow_valid = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderReleaseMtpShadowSlot(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderMtpKvTransaction *transaction)
{
	if (state == 0 || transaction == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (transaction->shadow_valid == 0u)
		return SPARK_STATUS_OK;
	if (transaction->shadow_slot_index >= state->mtp_shadow_slot_capacity ||
		state->mtp_shadow_free_count >= state->mtp_shadow_slot_capacity)
		return SPARK_STATUS_INTERNAL_ERROR;
	state->mtp_shadow_free_indices[state->mtp_shadow_free_count++] =
		transaction->shadow_slot_index;
	transaction->shadow_slot_index = 0u;
	transaction->shadow_valid = 0u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderClearActiveMtpKvTransaction(
	SparkGlm52Pp13BuilderState *state,
	SparkGlm52Pp13BuilderMtpKvTransaction *transaction)
{
	uint32_t draft_index;
	SparkStatus status;
	if (state == 0 || transaction == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (draft_index = 0u;
		 draft_index < transaction->pinned_token_count;
		 ++draft_index)
	{
		status = SparkGlm52Pp13WorkControlUnpinPhysicalBlock(
			&state->kv_state,
			transaction->physical_slots[draft_index] /
				state->kv_state.block_token_count);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	while (transaction->transient_block_count != 0u)
	{
		status = SparkGlm52Pp13WorkControlReleaseTransientPhysicalBlock(
			&state->kv_state,
			transaction->transient_physical_blocks[
				transaction->transient_block_count - 1u]);
		if (status != SPARK_STATUS_OK)
			return status;
		transaction->transient_block_count -= 1u;
	}
	status = SparkGlm52Pp13BuilderReleaseMtpShadowSlot(state,transaction);
	if (status != SPARK_STATUS_OK)
		return status;
	transaction->request_id = 0u;
	transaction->sequence_id = 0u;
	transaction->base_position = 0u;
	transaction->proposed_token_count = 0u;
	transaction->pinned_token_count = 0u;
	transaction->active = 0u;
	transaction->tree_verify = 0u;
	memset(transaction->physical_slots,0,sizeof(transaction->physical_slots));
	memset(
		transaction->transient_physical_blocks,
		0,
		sizeof(transaction->transient_physical_blocks));
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderDiscardMtpKvTransactions(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkGlm52Pp13BuilderMtpKvTransaction *transaction;
	uint32_t transaction_index;
	uint32_t lane_index;
	SparkStatus status;
	if (state == 0 || work_packet == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((work_packet->flags &
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u)
	{
		for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
		{
			for (transaction_index = 0u;
				 transaction_index <
					state->configuration.maximum_resident_sequence_count;
				 ++transaction_index)
			{
				transaction = &state->mtp_kv_transactions[transaction_index];
				if (transaction->active != 0u &&
					transaction->sequence_id ==
						work_packet->lanes[lane_index].sequence_id)
				{
					status = SparkGlm52Pp13BuilderClearActiveMtpKvTransaction(
						state,transaction);
					if (status != SPARK_STATUS_OK)
						return status;
					memset(transaction,0,sizeof(*transaction));
					continue;
				}
				if (transaction->last_valid != 0u &&
					transaction->last_sequence_id ==
						work_packet->lanes[lane_index].sequence_id)
					memset(transaction,0,sizeof(*transaction));
			}
		}
		return SPARK_STATUS_OK;
	}
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		transaction = SparkGlm52Pp13BuilderMtpKvTransactionForLane(
			state,&work_packet->lanes[lane_index]);
		if (transaction == 0)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		if (transaction->active != 0u)
		{
			if (transaction->request_id != work_packet->lanes[lane_index].request_id ||
				transaction->sequence_id != work_packet->lanes[lane_index].sequence_id)
				return SPARK_STATUS_INTERNAL_ERROR;
			status = SparkGlm52Pp13BuilderClearActiveMtpKvTransaction(
				state,transaction);
			if (status != SPARK_STATUS_OK)
				return status;
		}
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderMtpTransactionBasePosition(
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	const SparkGlm52Pp13WorkControlLane *lane,
	uint64_t *base_position_out)
{
	if (work_packet == 0 || lane == 0 || base_position_out == 0 ||
		work_packet->speculative_token_index > lane->sequence_position)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*base_position_out = lane->sequence_position -
		work_packet->speculative_token_index;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderRecordMtpKvTransactions(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	uint32_t *created_out)
{
	SparkGlm52Pp13BuilderMtpKvTransaction *transaction;
	const SparkGlm52Pp13WorkControlLane *lane;
	uint32_t creation_mode;
	uint32_t draft_index;
	uint32_t lane_index;
	uint32_t execution_row_index;
	uint32_t physical_block_index;
	uint32_t block_index;
	uint32_t block_token_index;
	uint32_t tree_verify;
	uint32_t transient_index;
	uint64_t base_position;
	uint64_t position;
	SparkStatus status;
	if (state == 0 || work_packet == 0 || created_out == 0 ||
		SparkGlm52Pp13BuilderWorkIsMtpVerify(work_packet) == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*created_out = 0u;
	tree_verify = SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet);
	creation_mode = UINT32_MAX;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		lane = &work_packet->lanes[lane_index];
		status = SparkGlm52Pp13BuilderMtpTransactionBasePosition(
			work_packet,lane,&base_position);
		if (status != SPARK_STATUS_OK)
			return status;
		transaction = SparkGlm52Pp13BuilderMtpKvTransactionForLane(state,lane);
		if (transaction == 0)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		if (transaction->active != 0u &&
			(transaction->request_id != lane->request_id ||
			 transaction->sequence_id != lane->sequence_id ||
			 transaction->base_position != base_position ||
			 transaction->proposed_token_count != lane->speculative_token_count ||
			 transaction->tree_verify != tree_verify))
			return SPARK_STATUS_BUSY;
		if (creation_mode == UINT32_MAX)
			creation_mode = transaction->active == 0u ? 1u : 0u;
		else if (creation_mode != (transaction->active == 0u ? 1u : 0u))
			return SPARK_STATUS_BUSY;
	}
	if (creation_mode == 0u)
		return SPARK_STATUS_OK;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		lane = &work_packet->lanes[lane_index];
		execution_row_index = lane_index;
		status = SparkGlm52Pp13BuilderMtpTransactionBasePosition(
			work_packet,lane,&base_position);
		if (status != SPARK_STATUS_OK)
			goto fail;
		transaction = SparkGlm52Pp13BuilderMtpKvTransactionForLane(state,lane);
		transaction->request_id = lane->request_id;
		transaction->sequence_id = lane->sequence_id;
		transaction->base_position = base_position;
		transaction->proposed_token_count = lane->speculative_token_count;
		transaction->tree_verify = tree_verify;
		for (draft_index = 0u;
			 draft_index < (tree_verify != 0u
				? SPARK_GLM52_MODEL_MTP_TREE_CANONICAL_POSITION_COUNT
				: lane->speculative_token_count);
			 ++draft_index)
		{
			if (base_position > UINT64_MAX - 1u - draft_index)
			{
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
				goto fail;
			}
			position = base_position + 1u + draft_index;
			block_index = (uint32_t)(position / work_packet->block_token_count);
			block_token_index = (uint32_t)(position % work_packet->block_token_count);
			if (block_index >=
				state->host_lane_physical_block_counts[execution_row_index])
			{
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
				goto fail;
			}
			physical_block_index = state->host_physical_block_indices[
				((uint64_t)execution_row_index * state->kv_state.lane_stride) + block_index];
			if (physical_block_index >= state->kv_state.physical_block_capacity ||
				physical_block_index >
					(UINT32_MAX - block_token_index) / work_packet->block_token_count)
			{
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
				goto fail;
			}
			transaction->physical_slots[draft_index] =
				(physical_block_index * work_packet->block_token_count) +
				block_token_index;
			status = SparkGlm52Pp13WorkControlPinPhysicalBlock(
				&state->kv_state,physical_block_index);
			if (status != SPARK_STATUS_OK)
				goto fail;
			transaction->pinned_token_count += 1u;
		}
		if (tree_verify != 0u)
		{
			status = SparkGlm52Pp13BuilderAcquireMtpShadowSlot(
				state,transaction);
			if (status != SPARK_STATUS_OK)
				goto fail;
			for (transient_index = 0u;
				 transient_index <
					SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_BLOCK_COUNT;
				 ++transient_index)
			{
				status =
					SparkGlm52Pp13WorkControlAcquireTransientPhysicalBlock(
						&state->kv_state,
						&transaction->transient_physical_blocks[
							transient_index]);
				if (status != SPARK_STATUS_OK)
					goto fail;
				transaction->transient_block_count += 1u;
			}
		}
		transaction->active = 1u;
	}
	*created_out = 1u;
	return SPARK_STATUS_OK;

fail:
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		transaction = SparkGlm52Pp13BuilderMtpKvTransactionForLane(
			state,&work_packet->lanes[lane_index]);
		if (transaction != 0 && transaction->request_id ==
			work_packet->lanes[lane_index].request_id)
			(void)SparkGlm52Pp13BuilderClearActiveMtpKvTransaction(
				state,transaction);
	}
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderUploadMtpTreeShadowRows(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkGlm52Pp13BuilderMtpKvTransaction *transaction;
	uint32_t lane_index;
	uint32_t row_base;
	uint32_t shadow_base;
	if (state == 0 || work_packet == 0 ||
		SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet) == 0u ||
		work_packet->execution_row_count >
			state->rank_plan.execution_row_capacity ||
		state->host_mtp_tree_shadow_slot_mapping == 0 ||
		state->device_mtp_tree_shadow_slot_mapping == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(
		state->host_mtp_tree_shadow_slot_mapping,
		0xff,
		(size_t)work_packet->execution_row_count * sizeof(uint32_t));
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		transaction = SparkGlm52Pp13BuilderMtpKvTransactionForLane(
			state,&work_packet->lanes[lane_index]);
		if (transaction == 0 || transaction->active == 0u ||
			transaction->shadow_valid == 0u ||
			transaction->shadow_slot_index >=
				state->mtp_shadow_slot_capacity)
			return SPARK_STATUS_INTERNAL_ERROR;
		shadow_base = state->configuration.kv_pool_token_capacity +
			(transaction->shadow_slot_index *
				SPARK_GLM52_MODEL_MTP_TREE_SHADOW_TOKEN_COUNT);
		if ((uint64_t)shadow_base +
				SPARK_GLM52_MODEL_MTP_TREE_SHADOW_TOKEN_COUNT >
			state->cache_storage_token_capacity)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		row_base = lane_index * work_packet->rows_per_lane;
		state->host_mtp_tree_shadow_slot_mapping[
			row_base +
			SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH2_ALTERNATE_ROW] =
				shadow_base +
				SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH2_ALTERNATE_INDEX;
		state->host_mtp_tree_shadow_slot_mapping[
			row_base +
			SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_DEPTH3_ALTERNATE_ROW] =
				shadow_base +
				SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH3_ALTERNATE_INDEX;
	}
	return SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_mtp_tree_shadow_slot_mapping,
		state->host_mtp_tree_shadow_slot_mapping,
		(size_t)work_packet->execution_row_count * sizeof(uint32_t),
		cudaMemcpyHostToDevice,
		state->stream));
}

static SparkStatus SparkGlm52Pp13BuilderAddKvPayload(
	SparkGlm52Pp13BuilderKvPayloads *payloads,
	void *base,
	uint32_t token_bytes)
{
	if (payloads == 0 || base == 0 || token_bytes == 0u ||
		payloads->count >= SPARK_GLM52_PP13_BUILDER_KV_PAYLOAD_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	payloads->payloads[payloads->count].base = (uint8_t *)base;
	payloads->payloads[payloads->count].token_bytes = token_bytes;
	payloads->count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderBuildLayerKvPayloads(
	SparkGlm52Pp13BuilderLayer *layer,
	SparkGlm52Pp13BuilderKvPayloads *payloads)
{
	SparkStatus status;
	if (layer == 0 || payloads == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(payloads,0,sizeof(*payloads));
#define ADD_PAYLOAD(field, bytes) \
	do { if (layer->field != 0) { status = SparkGlm52Pp13BuilderAddKvPayload(payloads,layer->field,(bytes)); if (status != SPARK_STATUS_OK) return status; } } while (0)
	ADD_PAYLOAD(mla_cache,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS *
			sizeof(uint16_t));
	ADD_PAYLOAD(mla_cache_fp8,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS);
	ADD_PAYLOAD(mla_cache_scale,
		SPARK_GLM52_PP13_BUILDER_FP8_MLA_SCALE_COUNT * sizeof(float));
	ADD_PAYLOAD(key_nope_cache,
		SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS *
			sizeof(uint16_t));
	ADD_PAYLOAD(key_nope_cache_fp8,
		SPARK_GLM52_PP13_BUILDER_KEY_NOPE_CACHE_TOKEN_ELEMENTS);
	ADD_PAYLOAD(key_nope_cache_scale,
		SPARK_GLM52_PP13_BUILDER_FP8_KEY_NOPE_SCALE_COUNT * sizeof(float));
	ADD_PAYLOAD(value_cache,
		SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS *
			sizeof(uint16_t));
	ADD_PAYLOAD(value_cache_fp8,
		SPARK_GLM52_PP13_BUILDER_VALUE_CACHE_TOKEN_ELEMENTS);
	ADD_PAYLOAD(value_cache_scale,
		SPARK_GLM52_PP13_BUILDER_FP8_VALUE_SCALE_COUNT * sizeof(float));
	ADD_PAYLOAD(key_index_cache,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_INDEX_KEY_DIMENSION *
			sizeof(uint16_t));
#undef ADD_PAYLOAD
	return payloads->count == 0u
		? SPARK_STATUS_MODULE_NOT_VALIDATED : SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpKvRollback(
	SparkGlm52Pp13BuilderState *state,
	const uint32_t *host_physical_slots,
	uint32_t physical_slot_count)
{
	SparkGlm52Pp13BuilderLayer *layer;
	SparkGlm52Pp13BuilderKvPayloads payloads;
	uint32_t layer_offset;
	SparkStatus status;
	if (state == 0 || host_physical_slots == 0 || physical_slot_count == 0u ||
		physical_slot_count > state->rank_plan.execution_row_capacity)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_decode_positions,host_physical_slots,
		(size_t)physical_slot_count * sizeof(uint32_t),cudaMemcpyHostToDevice,
		state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	for (layer_offset = 0u;
		 layer_offset < state->rank_plan.layer_count;
		 ++layer_offset)
	{
		layer = &state->layers[layer_offset];
		status = SparkGlm52Pp13BuilderBuildLayerKvPayloads(
			layer,&payloads);
		if (status != SPARK_STATUS_OK)
			return status;
		SparkGlm52Pp13BuilderClearSpeculativeKvRowsKernel<<<
			physical_slot_count,SPARK_GLM52_PP13_BUILDER_THREADS,0u,state->stream>>>(
			(const uint32_t *)state->device_decode_positions,physical_slot_count,
			state->kv_state.block_token_count,payloads,
			(uint8_t *)layer->dsa_summary_dirty_flags);
		status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
		if (status != SPARK_STATUS_OK)
			return status;
	}
	status = SparkGlm52Pp13BuilderCudaStatus(
		cudaEventRecord(state->kv_event,state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(
			cudaStreamWaitEvent(state->kv_stream,state->kv_event,0u));
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderLaunchMtpKvPromotion(
	SparkGlm52Pp13BuilderState *state,
	uint32_t physical_slot_count)
{
	SparkGlm52Pp13BuilderLayer *layer;
	SparkGlm52Pp13BuilderKvPayloads payloads;
	uint32_t layer_offset;
	SparkStatus status;
	if (state == 0 || physical_slot_count == 0u ||
		physical_slot_count > state->rank_plan.execution_row_capacity)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
		state->device_decode_positions,state->host_decode_positions,
		(size_t)physical_slot_count * sizeof(uint32_t),cudaMemcpyHostToDevice,
		state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(cudaMemcpyAsync(
			state->device_decode_token_ids,state->host_decode_token_ids,
			(size_t)physical_slot_count * sizeof(uint32_t),cudaMemcpyHostToDevice,
			state->stream));
	if (status != SPARK_STATUS_OK)
		return status;
	for (layer_offset = 0u;
		 layer_offset < state->rank_plan.layer_count;
		 ++layer_offset)
	{
		layer = &state->layers[layer_offset];
		status = SparkGlm52Pp13BuilderBuildLayerKvPayloads(
			layer,&payloads);
		if (status != SPARK_STATUS_OK)
			return status;
		SparkGlm52Pp13BuilderCopySpeculativeKvRowsKernel<<<
			physical_slot_count,SPARK_GLM52_PP13_BUILDER_THREADS,0u,state->stream>>>(
			(const uint32_t *)state->device_decode_positions,
			(const uint32_t *)state->device_decode_token_ids,
			physical_slot_count,state->kv_state.block_token_count,
			payloads,
			(uint8_t *)layer->dsa_summary_dirty_flags);
		status = SparkGlm52Pp13BuilderCudaStatus(cudaGetLastError());
		if (status != SPARK_STATUS_OK)
			return status;
	}
	status = SparkGlm52Pp13BuilderCudaStatus(
		cudaEventRecord(state->kv_event,state->stream));
	if (status == SPARK_STATUS_OK)
		status = SparkGlm52Pp13BuilderCudaStatus(
			cudaStreamWaitEvent(state->kv_stream,state->kv_event,0u));
	return status;
}

static SparkStatus SparkGlm52Pp13BuilderMtpResolutionFailure(
	const SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlLane *lane,
	const SparkGlm52Pp13BuilderMtpKvTransaction *transaction,
	const char *reason,
	SparkStatus status)
{
	if (state != 0 && lane != 0 && reason != 0)
		fprintf(stderr,"pp13_mtp_resolution_failed rank=%u reason=%s status=%u request=%llu sequence=%llu slot=%u position=%llu proposed=%u accepted=%u transaction_active=%u transaction_request=%llu transaction_sequence=%llu transaction_base=%llu transaction_proposed=%u last_valid=%u last_request=%llu last_sequence=%llu last_base=%llu last_proposed=%u last_accepted=%u\n",state->rank_plan.rank_index,reason,(uint32_t)status,(unsigned long long)lane->request_id,(unsigned long long)lane->sequence_id,lane->request_slot_index,(unsigned long long)lane->sequence_position,lane->mtp_resolution_proposed_token_count,lane->mtp_resolution_accepted_token_count,transaction == 0 ? 0u : transaction->active,transaction == 0 ? 0ull : (unsigned long long)transaction->request_id,transaction == 0 ? 0ull : (unsigned long long)transaction->sequence_id,transaction == 0 ? 0ull : (unsigned long long)transaction->base_position,transaction == 0 ? 0u : transaction->proposed_token_count,transaction == 0 ? 0u : transaction->last_valid,transaction == 0 ? 0ull : (unsigned long long)transaction->last_request_id,transaction == 0 ? 0ull : (unsigned long long)transaction->last_sequence_id,transaction == 0 ? 0ull : (unsigned long long)transaction->last_base_position,transaction == 0 ? 0u : transaction->last_proposed_token_count,transaction == 0 ? 0u : transaction->last_accepted_token_count);
	return status;
}

static uint32_t SparkGlm52Pp13BuilderContinuesMtpKvTransaction(
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	const SparkGlm52Pp13WorkControlLane *lane,
	const SparkGlm52Pp13BuilderMtpKvTransaction *transaction)
{
	uint64_t base_position;
	if (work_packet == 0 || lane == 0 || transaction == 0 ||
		work_packet->speculative_token_index == 0u ||
		SparkGlm52Pp13BuilderWorkIsMtpVerify(work_packet) == 0u ||
		lane->mtp_resolution_proposed_token_count != 0u ||
		transaction->active == 0u ||
		transaction->request_id != lane->request_id ||
		transaction->sequence_id != lane->sequence_id ||
		transaction->proposed_token_count != lane->speculative_token_count)
		return 0u;
	if (SparkGlm52Pp13BuilderMtpTransactionBasePosition(
			work_packet,lane,&base_position) != SPARK_STATUS_OK)
		return 0u;
	return transaction->base_position == base_position;
}

static SparkStatus SparkGlm52Pp13BuilderAppendMtpTreePromotions(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlLane *lane,
	const SparkGlm52Pp13BuilderMtpKvTransaction *transaction,
	uint32_t *promotion_count)
{
	uint32_t path_id;
	uint32_t shadow_index;
	if (state == 0 || lane == 0 || transaction == 0 ||
		promotion_count == 0 || transaction->tree_verify == 0u ||
		transaction->transient_block_count != 0u ||
		transaction->shadow_valid == 0u ||
		transaction->shadow_slot_index >= state->mtp_shadow_slot_capacity)
		return SPARK_STATUS_INVALID_ARGUMENT;
	path_id = lane->mtp_resolution_path_id;
	if (path_id != SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE &&
		path_id != SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH3_ALTERNATE)
		return SPARK_STATUS_OK;
	if (*promotion_count >= state->rank_plan.execution_row_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	shadow_index =
		path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE
			? SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH2_ALTERNATE_INDEX
			: SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_DEPTH3_ALTERNATE_INDEX;
	state->host_decode_positions[*promotion_count] =
		state->configuration.kv_pool_token_capacity +
		(transaction->shadow_slot_index *
			SPARK_GLM52_MODEL_MTP_TREE_SHADOW_TOKEN_COUNT) +
		shadow_index;
	state->host_decode_token_ids[*promotion_count] =
		transaction->physical_slots[
			path_id == SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_DEPTH2_ALTERNATE
				? SPARK_GLM52_MODEL_MTP_TREE_CANONICAL_DEPTH2_INDEX
				: SPARK_GLM52_MODEL_MTP_TREE_CANONICAL_DEPTH3_INDEX];
	*promotion_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderApplyMtpKvResolutions(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkGlm52Pp13BuilderMtpKvTransaction *transaction;
	const SparkGlm52Pp13WorkControlLane *lane;
	uint32_t draft_index;
	uint32_t lane_index;
	uint32_t promotion_count;
	uint32_t rejected_slot_count;
	uint64_t resolution_base_position;
	SparkStatus status;
	if (state == 0 || work_packet == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->host_mtp_request_slot_indices == 0)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	promotion_count = 0u;
	rejected_slot_count = 0u;
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		lane = &work_packet->lanes[lane_index];
		transaction = SparkGlm52Pp13BuilderMtpKvTransactionForLane(state,lane);
		if (transaction == 0)
			return SparkGlm52Pp13BuilderMtpResolutionFailure(
				state,lane,transaction,"transaction_slot",
				SPARK_STATUS_CAPACITY_EXCEEDED);
		if (lane->mtp_resolution_proposed_token_count == 0u)
		{
			if (transaction->active != 0u &&
				transaction->sequence_id == lane->sequence_id &&
				SparkGlm52Pp13BuilderContinuesMtpKvTransaction(
					work_packet,lane,transaction) == 0u)
				return SparkGlm52Pp13BuilderMtpResolutionFailure(
					state,lane,transaction,"resolution_pending",
					SPARK_STATUS_BUSY);
			continue;
		}
		if (lane->sequence_position <
			(uint64_t)lane->mtp_resolution_accepted_token_count + 1u)
			return SparkGlm52Pp13BuilderMtpResolutionFailure(
				state,lane,transaction,"position_underflow",
				SPARK_STATUS_INVALID_ARGUMENT);
		resolution_base_position = lane->sequence_position -
			(uint64_t)lane->mtp_resolution_accepted_token_count - 1u;
		if (transaction->active == 0u)
		{
			if (transaction->last_valid == 0u ||
				transaction->last_request_id != lane->request_id ||
				transaction->last_sequence_id != lane->sequence_id ||
				transaction->last_base_position !=
					resolution_base_position ||
				transaction->last_proposed_token_count !=
					lane->mtp_resolution_proposed_token_count ||
				transaction->last_accepted_token_count !=
					lane->mtp_resolution_accepted_token_count ||
				transaction->last_resolution_path_id !=
					lane->mtp_resolution_path_id)
				return SparkGlm52Pp13BuilderMtpResolutionFailure(
					state,lane,transaction,"resolved_identity",
					SPARK_STATUS_INVALID_ARGUMENT);
			continue;
		}
		if (transaction->request_id != lane->request_id ||
			transaction->sequence_id != lane->sequence_id ||
			transaction->base_position != resolution_base_position ||
			transaction->proposed_token_count !=
				lane->mtp_resolution_proposed_token_count)
			return SparkGlm52Pp13BuilderMtpResolutionFailure(
				state,lane,transaction,"active_identity",
				SPARK_STATUS_INVALID_ARGUMENT);
		if (transaction->tree_verify != 0u)
		{
			if (SparkGlm52MtpTreeResolutionIsValid(
					lane->mtp_resolution_proposed_token_count,
					lane->mtp_resolution_accepted_token_count,
					lane->mtp_resolution_path_id) == 0u)
				return SparkGlm52Pp13BuilderMtpResolutionFailure(
					state,lane,transaction,"tree_resolution",
					SPARK_STATUS_INVALID_ARGUMENT);
			for (draft_index =
					lane->mtp_resolution_accepted_token_count;
				 draft_index <
					SPARK_GLM52_MODEL_MTP_TREE_CANONICAL_POSITION_COUNT;
				 ++draft_index)
			{
				if (rejected_slot_count >=
					state->rank_plan.execution_row_capacity)
					return SparkGlm52Pp13BuilderMtpResolutionFailure(
						state,lane,transaction,"rollback_capacity",
						SPARK_STATUS_CAPACITY_EXCEEDED);
				state->host_mtp_request_slot_indices[rejected_slot_count++] =
					transaction->physical_slots[draft_index];
			}
			status = SparkGlm52Pp13BuilderAppendMtpTreePromotions(
				state,lane,transaction,&promotion_count);
			if (status != SPARK_STATUS_OK)
				return SparkGlm52Pp13BuilderMtpResolutionFailure(
					state,lane,transaction,"tree_promotion",status);
			continue;
		}
		for (draft_index = lane->mtp_resolution_accepted_token_count;
			 draft_index < transaction->proposed_token_count;
			 ++draft_index)
		{
			if (rejected_slot_count >= state->rank_plan.execution_row_capacity)
				return SparkGlm52Pp13BuilderMtpResolutionFailure(
					state,lane,transaction,"rollback_capacity",
					SPARK_STATUS_CAPACITY_EXCEEDED);
			state->host_mtp_request_slot_indices[rejected_slot_count++] =
				transaction->physical_slots[draft_index];
		}
	}
	if (promotion_count != 0u)
	{
		status = SparkGlm52Pp13BuilderLaunchMtpKvPromotion(
			state,promotion_count);
		if (status != SPARK_STATUS_OK)
			return SparkGlm52Pp13BuilderMtpResolutionFailure(
				state,&work_packet->lanes[0u],0,"promotion_launch",status);
	}
	if (rejected_slot_count != 0u)
	{
		status = SparkGlm52Pp13BuilderLaunchMtpKvRollback(
			state,state->host_mtp_request_slot_indices,rejected_slot_count);
		if (status != SPARK_STATUS_OK)
			return SparkGlm52Pp13BuilderMtpResolutionFailure(
				state,&work_packet->lanes[0u],0,"rollback_launch",status);
	}
	for (lane_index = 0u; lane_index < work_packet->lane_count; ++lane_index)
	{
		lane = &work_packet->lanes[lane_index];
		if (lane->mtp_resolution_proposed_token_count == 0u)
			continue;
		transaction = SparkGlm52Pp13BuilderMtpKvTransactionForLane(state,lane);
		if (transaction->active == 0u)
			continue;
		resolution_base_position = lane->sequence_position -
			(uint64_t)lane->mtp_resolution_accepted_token_count - 1u;
		transaction->last_request_id = lane->request_id;
		transaction->last_sequence_id = lane->sequence_id;
		transaction->last_base_position = resolution_base_position;
		transaction->last_proposed_token_count =
			lane->mtp_resolution_proposed_token_count;
		transaction->last_accepted_token_count =
			lane->mtp_resolution_accepted_token_count;
		transaction->last_resolution_path_id =
			lane->mtp_resolution_path_id;
		transaction->last_valid = 1u;
		status = SparkGlm52Pp13BuilderClearActiveMtpKvTransaction(
			state,transaction);
		if (status != SPARK_STATUS_OK)
			return SparkGlm52Pp13BuilderMtpResolutionFailure(
				state,lane,transaction,"transaction_clear",status);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderBuildResidentKvTable(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet)
{
	SparkStatus flush_status;
	SparkStatus status;
	uint64_t prior_control_generation;
	uint32_t nvme_enabled;

	if (state == 0 || work_packet == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	prior_control_generation = state->kv_state.control_generation;
	nvme_enabled = state->kv_nvme_fd >= 0 ? 1u : 0u;
	if (nvme_enabled != 0u)
	{
		if (state->kv_nvme_batch_active != 0u ||
			state->kv_nvme_pending_store_count != 0u ||
			state->kv_nvme_pending_load_count != 0u)
			return SPARK_STATUS_INTERNAL_ERROR;
		state->kv_nvme_batch_active = 1u;
	}
	status = SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		work_packet,&state->kv_state,&state->host_kv_view);
	if (status == SPARK_STATUS_OK)
		SparkGlm52Pp13BuilderLogKvGenerationReset(
			state,prior_control_generation);
	if (nvme_enabled != 0u)
	{
		flush_status = SparkGlm52Pp13BuilderKvNvmeFlushBatch(state);
		state->kv_nvme_batch_active = 0u;
		if (flush_status != SPARK_STATUS_OK)
			status = flush_status;
	}
	if (status != SPARK_STATUS_OK)
		(void)SparkGlm52Pp13WorkControlCancelHostKvBlockTable(
			work_packet,&state->kv_state);
	return status;
}

static void SparkGlm52Pp13BuilderLogSubmitFailure(
	const SparkGlm52Pp13BuilderState *state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	const char *step,
	SparkStatus status)
{
	if (state == 0 || work_packet == 0 || step == 0 || status == SPARK_STATUS_BUSY)
		return;
	fprintf(stderr,"pp13_builder_submit_failed rank=%u step=%s status=%u request=%llu sequence=%llu position=%llu flags=0x%x rows=%u lanes=%u bucket=%u\n",state->rank_plan.rank_index,step,(uint32_t)status,(unsigned long long)work_packet->request_id,(unsigned long long)work_packet->sequence_id,(unsigned long long)work_packet->sequence_position,work_packet->flags,work_packet->execution_row_count,work_packet->lane_count,work_packet->execution_batch_bucket);
}

static SparkStatus SparkGlm52Pp13BuilderSubmitWork(
	void *builder_state,
	const SparkGlm52Pp13WorkControlPacket *work_packet,
	SparkHiddenTransportSession *input_transport_session,
	SparkHiddenTransportSession *output_transport_session,
	SparkModelDriverCompletionFunction completion_function,
	void *completion_context)
{
	SparkGlm52Pp13BuilderState *state;
	SparkGlm52ResidentDecodeStageProductionRunnerDispatch dispatch;
	const char *failure_step;
	uint32_t active_row_capacity;
	uint32_t mtp_transaction_created;
	SparkStatus status;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	mtp_transaction_created = 0u;
	if (state == 0 || work_packet == 0 || state->runner_ready == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	active_row_capacity =
		SparkGlm52Pp13BuilderLayerActiveRowCapacity(state,&state->layers[0]);
	status = SparkGlm52Pp13WorkControlValidatePacket(
		work_packet,
		active_row_capacity,
		SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT);
	if (status != SPARK_STATUS_OK)
		return status;
	if (state->pending_work_active != 0u ||
		__atomic_load_n(
			&state->pending_finalizer_state,__ATOMIC_ACQUIRE) !=
			SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_NONE ||
		state->dspark_ready_draft_count != 0u)
		return SPARK_STATUS_BUSY;
	if ((work_packet->flags &
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u)
	{
		status = SparkGlm52Pp13BuilderDiscardMtpKvTransactions(
			state,work_packet);
		if (status != SPARK_STATUS_OK)
			return status;
		status = SparkGlm52Pp13BuilderReleaseDsparkPacketLanes(
			state,work_packet);
		if (status != SPARK_STATUS_OK)
			return status;
		return SparkGlm52Pp13WorkControlReleasePacketSequences(
			work_packet,&state->kv_state);
	}
	if (work_packet->execution_row_count >
		active_row_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if (work_packet->execution_batch_bucket >
			SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET ||
		work_packet->execution_row_count >
			work_packet->execution_batch_bucket)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if (input_transport_session != 0)
	{
		SparkHiddenTransportPacket early_input_packet;
		SparkGlm52Pp13BuilderBuildPacket(
			state,
			work_packet,
			state->layers[0].input_hidden,
			state->input_sideband,
			SparkGlm52Pp13BuilderNeedsInputSideband(state),
			&early_input_packet);
		status = SparkGlm52Pp13BuilderArmDsparkSideband(
			state,
			work_packet,
			state->rank_plan.rank_index - 1u,
			state->input_sideband,
			&early_input_packet);
		if (status != SPARK_STATUS_OK)
			return status;
		status = SparkHiddenTransportPostReceive(
			input_transport_session,
			&early_input_packet);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	state->exact_plan.batch_bucket = work_packet->execution_batch_bucket;
	status = SparkGlm52Pp13BuilderApplyMtpKvResolutions(state,work_packet);
	if (status != SPARK_STATUS_OK)
	{
		SparkGlm52Pp13BuilderLogSubmitFailure(
			state,work_packet,"mtp_resolution",status);
		return status;
	}
	status = SparkGlm52Pp13BuilderBuildResidentKvTable(state,work_packet);
	if (status != SPARK_STATUS_OK)
	{
		SparkGlm52Pp13BuilderLogSubmitFailure(
			state,work_packet,"resident_kv_table",status);
		return status;
	}
	failure_step = "device_kv_view";
	status = SparkGlm52Pp13BuilderPrepareDeviceKvView(
		state,&state->host_kv_view);
	if (status != SPARK_STATUS_OK)
		goto cancel_work;
	if (SparkGlm52Pp13BuilderWorkIsMtpVerify(work_packet) != 0u)
	{
		failure_step = "mtp_transaction_record";
		status = SparkGlm52Pp13BuilderRecordMtpKvTransactions(
			state,work_packet,&mtp_transaction_created);
		if (status != SPARK_STATUS_OK)
			goto cancel_work;
	}
	if (SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet) != 0u)
	{
		failure_step = "mtp_shadow_rows";
		status = SparkGlm52Pp13BuilderUploadMtpTreeShadowRows(
			state,work_packet);
		if (status != SPARK_STATUS_OK)
			goto cancel_work;
	}
	if (work_packet->rows_per_lane > 1u)
	{
		failure_step = "execution_kv_rows";
		status = SparkGlm52Pp13BuilderExpandExecutionKvRows(
			state,work_packet->lane_count,work_packet->rows_per_lane,work_packet);
		if (status != SPARK_STATUS_OK)
			goto cancel_work;
	}
	failure_step = "dsa_candidate_count";
	status = SparkGlm52Pp13BuilderSetDsaCandidateCount(
		state,
		work_packet->kv_block_table_token_count);
	if (status != SPARK_STATUS_OK)
		goto cancel_work;
	failure_step = "mtp_budget_upload";
	status = SparkGlm52Pp13BuilderUploadWorkMtpBudgets(
		state,
		work_packet);
	if (status != SPARK_STATUS_OK)
		goto cancel_work;
	failure_step = "decode_position_upload";
	status = SparkGlm52Pp13BuilderUploadWorkDecodePositions(
		state,
		work_packet);
	if (status == SPARK_STATUS_OK)
	{
		failure_step = "token_embedding_upload";
		status = SparkGlm52Pp13BuilderUploadWorkTokenIdsAndEmbedding(
			state,work_packet);
	}
	if (status == SPARK_STATUS_OK)
	{
		failure_step = "decode_metadata";
		status = SparkGlm52Pp13BuilderLaunchDecodeMetadataForAllLayers(
			state,
			work_packet->execution_row_count);
	}
	if (status == SPARK_STATUS_OK)
	{
		failure_step = "linear_plan_rows";
		status = SparkGlm52Pp13BuilderPrepareStageLinearPlanRows(
			state,work_packet->execution_row_count);
	}
	if (status != SPARK_STATUS_OK)
		goto cancel_work;
	memset(&dispatch,0,sizeof(dispatch));
	dispatch.abi_version =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_ABI_VERSION;
	dispatch.descriptor_bytes =
		SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_BYTES;
	dispatch.flags = (work_packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u
		? SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_PREFILL
		: 0u;
	if (SparkGlm52Pp13BuilderWorkIsMtpTreeVerify(work_packet) != 0u)
		dispatch.flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_MTP_TREE_VERIFY;
	dispatch.priority = work_packet->priority;
	dispatch.request_id = work_packet->request_id;
	dispatch.sequence_id = work_packet->sequence_id;
	dispatch.sequence_position = work_packet->sequence_position;
	dispatch.deadline_time_ns = work_packet->deadline_time_ns;
	dispatch.active_sequence_count = work_packet->execution_row_count;
	dispatch.logical_lane_count = work_packet->lane_count;
	dispatch.rows_per_lane = work_packet->rows_per_lane;
	dispatch.new_token_count = work_packet->new_token_count;
	dispatch.pipeline_slot = work_packet->pipeline_slot;
	dispatch.kv_block_table = &state->device_kv_view;
	if ((work_packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT) != 0u &&
		work_packet->mtp_draft_token_count != 0u)
		dispatch.mtp_draft_token_budgets =
			(const uint32_t *)state->layers[0].mtp_draft_token_budgets;
	dispatch.hidden_input_transport_session = input_transport_session;
	dispatch.hidden_output_transport_session = output_transport_session;
	if (input_transport_session != 0)
		dispatch.flags |=
			SPARK_GLM52_RESIDENT_DECODE_STAGE_PRODUCTION_RUNNER_DISPATCH_FLAG_HIDDEN_INPUT_PRERECEIVED;
	SparkGlm52Pp13BuilderApplyDsparkDispatch(state,work_packet,&dispatch);
	SparkGlm52Pp13BuilderBuildPacket(
		state,
		work_packet,
		state->layers[0].input_hidden,
		state->input_sideband,
		SparkGlm52Pp13BuilderNeedsInputSideband(state),
		&dispatch.hidden_input_packet);
	SparkGlm52Pp13BuilderBuildPacket(
		state,
		work_packet,
		state->layers[state->rank_plan.layer_count - 1u].layer_output_hidden,
		state->output_sideband,
		SparkGlm52Pp13BuilderNeedsOutputSideband(state),
		&dispatch.hidden_output_packet);
	if (input_transport_session != 0)
	{
		failure_step = "input_sideband";
		status = SparkGlm52Pp13BuilderArmDsparkSideband(
			state,
			work_packet,
			state->rank_plan.rank_index - 1u,
			state->input_sideband,
			&dispatch.hidden_input_packet);
		if (status != SPARK_STATUS_OK)
			goto cancel_work;
	}
	if (output_transport_session != 0)
	{
		failure_step = "output_sideband";
		status = SparkGlm52Pp13BuilderArmDsparkSideband(
			state,
			work_packet,
			state->rank_plan.rank_index,
			state->output_sideband,
			&dispatch.hidden_output_packet);
		if (status != SPARK_STATUS_OK)
			goto cancel_work;
	}
	state->captured_completion_valid = 0u;
	state->dspark_stage_count = 0u;
	state->dspark_draft_count = 0u;
	state->pending_work_packet = *work_packet;
	state->pending_work_completion_function = completion_function;
	state->pending_work_completion_context = completion_context;
	state->pending_work_completion_count = 0u;
	state->pending_work_completion_overflow = 0u;
	state->last_work_completion_status = SPARK_STATUS_BUSY;
	state->pending_work_active = 1u;
	__atomic_store_n(
		&state->pending_finalizer_state,
		SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_NONE,
		__ATOMIC_RELEASE);
	dispatch.completion_function = SparkGlm52Pp13BuilderCompletePendingWork;
	dispatch.completion_context = state;
	failure_step = "runner_submit";
	status = SparkGlm52ResidentDecodeStageProductionRunnerSubmit(
		&state->runner,
		&dispatch);
	if (status != SPARK_STATUS_OK)
	{
		if (state->pending_work_active != 0u)
			state->pending_work_active = 0u;
		__atomic_store_n(
			&state->pending_finalizer_state,
			SPARK_GLM52_PP13_BUILDER_PENDING_FINALIZER_NONE,
			__ATOMIC_RELEASE);
		goto cancel_work;
	}
	state->asynchronous_submit_count += 1u;
	if (SparkGlm52Pp13BuilderWorkIsSpeculativeVerify(work_packet) != 0u)
	{
		state->layer_major_submit_count += 1u;
		state->last_layer_major_logical_lane_count = work_packet->lane_count;
		state->last_layer_major_rows_per_lane = work_packet->rows_per_lane;
		state->last_layer_major_execution_row_count =
			work_packet->execution_row_count;
	}
	return SPARK_STATUS_OK;

cancel_work:
	SparkGlm52Pp13BuilderLogSubmitFailure(
		state,work_packet,failure_step,status);
	if (mtp_transaction_created != 0u)
		(void)SparkGlm52Pp13BuilderDiscardMtpKvTransactions(
			state,work_packet);
	if (SparkGlm52Pp13BuilderWorkIsSpeculativeVerify(work_packet) != 0u)
		state->layer_major_failure_count += 1u;
	(void)SparkGlm52Pp13WorkControlCancelHostKvBlockTable(
		work_packet,&state->kv_state);
	return status;
}

static uint64_t SparkGlm52Pp13BuilderProbeFnv64(const uint8_t *data,uint64_t bytes)
{
	uint64_t hash;
	uint64_t offset;
	hash = 0xcbf29ce484222325ull;
	for (offset = 0u; offset < bytes; ++offset)
		hash = (hash ^ (uint64_t)data[offset]) * 0x100000001b3ull;
	return hash;
}

static void SparkGlm52Pp13BuilderMaybeProbeMlaSlots(SparkGlm52Pp13BuilderState *state)
{
	static uint8_t slot_host[
		SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS *
		sizeof(uint16_t)];
	uint32_t slot_index;
	uint64_t slot_bytes;
	const uint8_t *cache_base;
	if (getenv("SPARKPIPE_MLA_SLOT_PROBE") == 0 || state == 0)
		return;
	slot_bytes =
		(uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS *
		sizeof(uint16_t);
	cache_base = (const uint8_t *)state->layers[0].mla_cache;
	for (slot_index = 0u; slot_index < 2u; ++slot_index)
	{
		if (cache_base != 0)
		{
			if (cudaMemcpy(slot_host,cache_base + ((uint64_t)slot_index * slot_bytes),(size_t)slot_bytes,cudaMemcpyDeviceToHost) != cudaSuccess)
				return;
		}
		else
		{
			SparkStatus status;
			if (state->layers[0].mla_cache_fp8 == 0 ||
				state->layers[0].mla_cache_scale == 0 ||
				state->layers[0].raw_kv_a == 0)
				return;
			status = SparkGlm52Sm121RequiredDecodeStageLaunchFp8E4m3KvCacheLoad(
				(const uint8_t *)state->layers[0].mla_cache_fp8 +
					((uint64_t)slot_index *
					 SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS),
				(const float *)state->layers[0].mla_cache_scale +
					((uint64_t)slot_index *
					 SPARK_GLM52_PP13_BUILDER_FP8_MLA_SCALE_COUNT),
				state->layers[0].raw_kv_a,1u,
				SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS,
				SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_KV_CACHE_SCALE_BLOCK,
				(void *)state->stream);
			if (status != SPARK_STATUS_OK ||
				cudaStreamSynchronize(state->stream) != cudaSuccess ||
				cudaMemcpy(slot_host,state->layers[0].raw_kv_a,
					(size_t)slot_bytes,cudaMemcpyDeviceToHost) != cudaSuccess)
				return;
		}
		fprintf(stderr,"mla_slot%u=%016llx mla_slot%u_rope_pair0=%016llx\n",slot_index,(unsigned long long)SparkGlm52Pp13BuilderProbeFnv64(slot_host,slot_bytes),slot_index,(unsigned long long)SparkGlm52Pp13BuilderProbeFnv64(slot_host + ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * sizeof(uint16_t)),2u * sizeof(uint16_t)));
	}
}

static uint32_t SparkGlm52Pp13BuilderProbeHostSlot(
	const SparkGlm52Pp13BuilderState *state,
	uint32_t position,
	uint32_t *physical_block_index_out)
{
	uint32_t block_index;
	uint32_t block_token_index;
	uint32_t block_token_count;
	if (physical_block_index_out != 0)
		*physical_block_index_out = SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	if (state == 0 || state->host_lane_physical_block_counts == 0 ||
		state->host_physical_block_indices == 0 ||
		state->device_kv_view.block_token_count == 0u)
		return SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	block_token_count = state->device_kv_view.block_token_count;
	block_index = position / block_token_count;
	block_token_index = position - (block_index * block_token_count);
	if (block_index >= state->host_lane_physical_block_counts[0u])
		return SPARK_GLM52_PP13_BUILDER_INVALID_SLOT;
	if (physical_block_index_out != 0)
		*physical_block_index_out = state->host_physical_block_indices[block_index];
	return (state->host_physical_block_indices[block_index] * block_token_count) +
		block_token_index;
}

static void SparkGlm52Pp13BuilderMaybeProbePrefillKvState(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch)
{
	uint32_t slot0,slot1,block0,block1,lane_blocks;
	uint64_t request_id;
	if (getenv("SPARKPIPE_MLA_SLOT_PROBE") == 0 || state == 0 ||
		prefill_dispatch == 0 || prefill_dispatch->request_dispatch == 0)
		return;
	request_id = prefill_dispatch->request_dispatch->request_ids[0u];
	lane_blocks = state->host_lane_physical_block_counts != 0 ?
		state->host_lane_physical_block_counts[0u] : 0u;
	slot0 = SparkGlm52Pp13BuilderProbeHostSlot(state,0u,&block0);
	slot1 = SparkGlm52Pp13BuilderProbeHostSlot(state,1u,&block1);
	fprintf(stderr,"pp13_kv_probe_entry request=%llu offset=%u count=%u kv_missing=%u kv_inflight=%u kv_resident=%u lane0_blocks=%u pos0_block=%u pos0_slot=%u pos1_block=%u pos1_slot=%u\n",
		(unsigned long long)request_id,
		prefill_dispatch->prompt_token_offset,
		prefill_dispatch->prompt_token_count,
		state->kv_state.missing_block_count,
		state->kv_state.in_flight_block_count,
		state->kv_state.resident_block_count,
		lane_blocks,
		block0,
		slot0,
		block1,
		slot1);
}

static void SparkGlm52Pp13BuilderMaybeProbeLayer0Sublayers(
	SparkGlm52Pp13BuilderState *state,
	uint64_t request_id,
	uint32_t token_offset,
	uint32_t position)
{
	uint64_t probe_slots[SPARK_GLM52_PP13_BUILDER_PROBE_HASH_SLOT_COUNT];
	if (getenv("SPARKPIPE_FP8_AMAX_PROBE") == 0 || state == 0 ||
		state->device_probe_hash_slots == 0)
		return;
	if (cudaMemcpy(probe_slots,state->device_probe_hash_slots,
			sizeof(probe_slots),cudaMemcpyDeviceToHost) != cudaSuccess)
		return;
	fprintf(stderr,
		"fp8_layer0_attention_probe request=%llu token_offset=%u position=%u weight=%016llx scale=%016llx input=%016llx attention_norm=%016llx kv_latent=%016llx raw_kv_b=%016llx fp8_value_cache=%016llx attention_value=%016llx attention_projection=%016llx post_attention=%016llx\n",
		(unsigned long long)request_id,
		token_offset,
		position,
		(unsigned long long)probe_slots[16],
		(unsigned long long)probe_slots[17],
		(unsigned long long)probe_slots[8],
		(unsigned long long)probe_slots[9],
		(unsigned long long)probe_slots[10],
		(unsigned long long)probe_slots[11],
		(unsigned long long)probe_slots[12],
		(unsigned long long)probe_slots[13],
		(unsigned long long)probe_slots[14],
		(unsigned long long)probe_slots[15]);
}

static void SparkGlm52Pp13BuilderMaybeProbePrefillInputHidden(
	SparkGlm52Pp13BuilderState *state,
	uint64_t request_id,
	uint32_t token_offset,
	uint32_t position)
{
	static uint8_t host_hidden[
		SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_BF16_BYTES];
	if (getenv("SPARKPIPE_FP8_AMAX_PROBE") == 0 || state == 0 ||
		state->layers[0].input_hidden == 0)
		return;
	if (cudaMemcpy(host_hidden,state->layers[0].input_hidden,
			sizeof(host_hidden),cudaMemcpyDeviceToHost) != cudaSuccess)
		return;
	fprintf(stderr,
		"fp8_prefill_input_probe request=%llu token_offset=%u position=%u hash=%016llx bytes=%llu\n",
		(unsigned long long)request_id,
		token_offset,
		position,
		(unsigned long long)SparkGlm52Pp13BuilderProbeFnv64(
			host_hidden,sizeof(host_hidden)),
		(unsigned long long)sizeof(host_hidden));
}

static SparkStatus SparkGlm52Pp13BuilderSubmitPrefillChunk(
	SparkGlm52Pp13BuilderState *state,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch,
	uint32_t token_offset,
	uint32_t token_count,
	SparkGlm52Pp13NodeContextBuilderIdlePumpFunction idle_pump_function,
	void *idle_pump_context,
	SparkGlm52Pp13WorkControlPacket *work_packet,
	uint32_t *submit_retry_out)
{
	uint32_t submit_retry;
	uint32_t wait_iteration;
	SparkStatus status;

	status = SparkGlm52Pp13WorkControlBuildPrefillPacket(
		prefill_dispatch,token_offset,token_count,work_packet);
	if (status != SPARK_STATUS_OK)
		return status;
	for (submit_retry = 0u; submit_retry < 25000u; ++submit_retry)
	{
		status = SparkGlm52Pp13BuilderSubmitWork(
			state,work_packet,0,state->output_transport_session,0,0);
		if (status != SPARK_STATUS_BUSY)
			break;
		(void)SparkGlm52ResidentDecodeStageProductionRunnerProgress(
			&state->runner);
		if (idle_pump_function != 0)
		{
			status = idle_pump_function(idle_pump_context);
			if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
				return status;
		}
		usleep(200u);
	}
	*submit_retry_out = submit_retry;
	if (status != SPARK_STATUS_OK)
		return status;
	for (wait_iteration = 0u;
		 state->pending_work_active != 0u && wait_iteration < 25000u;
		 ++wait_iteration)
	{
		status = SparkGlm52ResidentDecodeStageProductionRunnerProgress(
			&state->runner);
		if (status != SPARK_STATUS_OK)
			return status;
		if (idle_pump_function != 0)
		{
			status = idle_pump_function(idle_pump_context);
			if (status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY)
				return status;
		}
		if (state->pending_work_active != 0u)
			usleep(200u);
	}
	if (state->pending_work_active != 0u)
		return SPARK_STATUS_BUSY;
	return state->last_work_completion_status;
}

static SparkStatus SparkGlm52Pp13BuilderPrefill(
	void *builder_state,
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch,
	SparkGlm52Pp13NodeContextBuilderIdlePumpFunction idle_pump_function,
	void *idle_pump_context)
{
	SparkGlm52Pp13BuilderState *state;
	SparkGlm52Pp13WorkControlPacket work_packet;
	uint32_t token_count;
	uint32_t token_offset;
	uint32_t submit_retry;
	const char *debug_enabled;
	SparkStatus status;

	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || prefill_dispatch == 0 || state->runner_ready == 0u ||
		state->embedding_weight == 0 ||
		(state->rank_plan.flags & SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS) != 0u ||
		prefill_dispatch->request_dispatch == 0 ||
		prefill_dispatch->prefill_view == 0 ||
		prefill_dispatch->kv_block_table_view == 0 ||
		prefill_dispatch->lane_count == 0u ||
		prefill_dispatch->lane_count != prefill_dispatch->active_sequence_count ||
		prefill_dispatch->lane_count != prefill_dispatch->prefill_view->lane_count ||
		prefill_dispatch->lane_count > state->rank_plan.logical_lane_capacity ||
		prefill_dispatch->host_token_ids == 0 ||
		prefill_dispatch->host_token_stride == 0u ||
		prefill_dispatch->prompt_token_count == 0u ||
		prefill_dispatch->prompt_token_count >
			SPARK_GLM52_PP13_BUILDER_MAX_PREFILL_TOKENS)
		return SPARK_STATUS_INVALID_ARGUMENT;
	debug_enabled = getenv("SPARKPIPE_STAGE_COMPLETION_DEBUG");
	for (token_offset = 0u;
		 token_offset < prefill_dispatch->prompt_token_count;
		 token_offset += token_count)
	{
		status = SparkGlm52Pp13WorkControlSelectPrefillChunk(
			prefill_dispatch,
			token_offset,
			state->rank_plan.execution_row_capacity,
			&token_count);
		if (status != SPARK_STATUS_OK)
			return status;
		memset(&work_packet,0,sizeof(work_packet));
		status = SparkGlm52Pp13BuilderSubmitPrefillChunk(
			state,prefill_dispatch,token_offset,token_count,
			idle_pump_function,idle_pump_context,&work_packet,&submit_retry);
		if (status == SPARK_STATUS_OK && token_offset == 0u)
			SparkGlm52Pp13BuilderMaybeProbePrefillKvState(
				state,prefill_dispatch);
		if (status == SPARK_STATUS_OK)
			SparkGlm52Pp13BuilderMaybeProbePrefillInputHidden(
				state,work_packet.request_id,token_offset,
				(uint32_t)work_packet.sequence_position);
		if (status != SPARK_STATUS_OK)
			return status;
		SparkGlm52Pp13BuilderMaybeProbeLayer0Sublayers(
			state,work_packet.request_id,token_offset,
			(uint32_t)work_packet.sequence_position);
		if (idle_pump_function != 0)
			(void)idle_pump_function(idle_pump_context);
		if (debug_enabled != 0)
			fprintf(stderr,
				"pp13_builder_prefill_submitted position=%llu token_offset=%u tokens=%u lanes=%u rows=%u busy_retries=%u\n",
				(unsigned long long)work_packet.sequence_position,
				token_offset,token_count,work_packet.active_sequence_count,
				work_packet.execution_row_count,submit_retry);
	}
	SparkGlm52Pp13BuilderMaybeProbeMlaSlots(state);
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderDecode(
	void *builder_state,
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	SparkGlm52ServingDecodeResult *decode_result)
{
	SparkGlm52Pp13BuilderState *state;
	SparkGlm52Pp13WorkControlPacket work_packet;
	SparkStatus status;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || decode_dispatch == 0 || decode_result == 0 ||
		state->runner_ready == 0u ||
		state->embedding_weight == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13WorkControlBuildDecodePacket(
		decode_dispatch,0u,&work_packet);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13BuilderSubmitWork(
		state,
		&work_packet,
		0,
		state->output_transport_session,
		0,
		0);
	if (status != SPARK_STATUS_OK)
		return status;
	memset(decode_result,0,sizeof(*decode_result));
	decode_result->abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
	decode_result->descriptor_bytes =
		SPARK_GLM52_SERVING_DECODE_RESULT_DESCRIPTOR_BYTES;
	decode_result->lane_count = work_packet.lane_count;
	decode_result->token_stride = SPARK_GLM52_SERVING_MAX_DECODE_TOKENS_PER_LANE;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderTakeDsparkDraft(
	void *builder_state,
	SparkGlm52DsparkDraftResult *draft_result)
{
	SparkGlm52Pp13BuilderState *state;

	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || draft_result == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->dspark_ready_draft_count == 0u)
		return SPARK_STATUS_NOT_FOUND;
	*draft_result =
		state->dspark_ready_drafts[state->dspark_ready_draft_head];
	memset(
		&state->dspark_ready_drafts[state->dspark_ready_draft_head],
		0,
		sizeof(state->dspark_ready_drafts[state->dspark_ready_draft_head]));
	state->dspark_ready_draft_head =
		(state->dspark_ready_draft_head + 1u) %
		state->dspark_backend.maximum_lane_count;
	state->dspark_ready_draft_count -= 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13BuilderGetKvStats(
	void *builder_state,
	SparkGlm52Pp13NodeContextBuilderKvStats *stats)
{
	SparkGlm52Pp13BuilderState *state;
	size_t cuda_free_bytes;
	size_t cuda_total_bytes;
	state = (SparkGlm52Pp13BuilderState *)builder_state;
	if (state == 0 || stats == 0 || state->built == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(stats,0,sizeof(*stats));
	stats->abi_version = SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_ABI_VERSION;
	stats->descriptor_bytes =
		SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_KV_STATS_BYTES;
	stats->nvme_enabled = state->kv_nvme_fd >= 0 ? 1u : 0u;
	stats->nvme_mode = state->kv_nvme_fd >= 0
		? SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_NVME_MODE_BATCHED_COHORT_JIT
		: SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_NVME_MODE_DISABLED;
	stats->physical_block_capacity = state->kv_state.physical_block_capacity;
	stats->logical_block_capacity = state->kv_state.backing_block_capacity != 0u
		? state->kv_state.backing_block_capacity
		: state->kv_state.physical_block_capacity;
	stats->logical_block_count = state->kv_state.directory_entry_count;
	stats->resident_block_count =
		state->kv_state.allocated_physical_block_count;
	stats->swapped_block_count = state->kv_state.swapped_block_count;
	stats->nvme_record_bytes = state->kv_nvme_record_bytes;
	stats->nvme_store_count = state->kv_nvme_store_count;
	stats->nvme_load_count = state->kv_nvme_load_count;
	stats->nvme_write_bytes = state->kv_nvme_record_bytes != 0u &&
		state->kv_nvme_store_count > UINT64_MAX / state->kv_nvme_record_bytes
		? UINT64_MAX
		: state->kv_nvme_store_count * state->kv_nvme_record_bytes;
	stats->nvme_read_bytes = state->kv_nvme_record_bytes != 0u &&
		state->kv_nvme_load_count > UINT64_MAX / state->kv_nvme_record_bytes
		? UINT64_MAX
		: state->kv_nvme_load_count * state->kv_nvme_record_bytes;
	stats->nvme_synchronous_wait_count =
		state->kv_nvme_synchronous_wait_count;
	stats->nvme_batch_flush_count = state->kv_nvme_batch_flush_count;
	stats->nvme_maximum_batch_operation_count =
		state->kv_nvme_maximum_batch_operation_count;
	stats->resident_bytes_per_token =
		state->kv_jit_budget.resident_bytes_per_token;
	stats->resident_pool_bytes = state->kv_jit_budget.resident_pool_bytes;
	stats->nvme_capacity_bytes = state->kv_jit_budget.nvme_capacity_bytes;
	stats->compact_selected_mla_working_set_bytes =
		state->kv_jit_budget.compact_selected_mla_working_set_bytes;
	stats->nvme_batch_block_capacity =
		state->configuration.kv_nvme_batch_block_count;
	stats->nvme_pending_store_count = state->kv_nvme_pending_store_count;
	stats->nvme_pending_load_count = state->kv_nvme_pending_load_count;
	stats->nvme_clean_evict_count = state->kv_state.clean_evict_count;
	stats->pending_work_active = state->pending_work_active;
	stats->logical_lane_capacity = state->rank_plan.logical_lane_capacity;
	stats->execution_row_capacity = state->rank_plan.execution_row_capacity;
	stats->last_layer_major_logical_lane_count =
		state->last_layer_major_logical_lane_count;
	stats->last_layer_major_rows_per_lane =
		state->last_layer_major_rows_per_lane;
	stats->last_layer_major_execution_row_count =
		state->last_layer_major_execution_row_count;
	if (SparkGlm52Pp13RuntimeExpectedMoeBackendKind(
			state->rank_plan.quantization_mode,
			&stats->moe_backend_kind) != SPARK_STATUS_OK)
		return SPARK_STATUS_INVALID_ARGUMENT;
	stats->moe_bound_layer_count = state->moe_bound_layer_count;
	stats->moe_expected_layer_count = state->moe_expected_layer_count;
	stats->fp8_scaled_gemm_bound_plan_count =
		state->fp8_scaled_gemm_bound_plan_count;
	stats->fp8_scaled_gemm_expected_plan_count =
		state->fp8_scaled_gemm_expected_plan_count;
	stats->model_quantization_mode = state->rank_plan.quantization_mode;
	stats->asynchronous_submit_count = state->asynchronous_submit_count;
	stats->asynchronous_completion_count = state->asynchronous_completion_count;
	stats->asynchronous_failure_count = state->asynchronous_failure_count;
	stats->layer_major_submit_count = state->layer_major_submit_count;
	stats->layer_major_completion_count = state->layer_major_completion_count;
	stats->layer_major_failure_count = state->layer_major_failure_count;
	if (cudaMemGetInfo(&cuda_free_bytes,&cuda_total_bytes) != cudaSuccess)
		return SPARK_STATUS_IO_ERROR;
	stats->cuda_total_bytes = (uint64_t)cuda_total_bytes;
	stats->cuda_initial_free_bytes = state->cuda_initial_free_bytes;
	stats->cuda_current_free_bytes = (uint64_t)cuda_free_bytes;
	stats->cuda_consumed_bytes = state->cuda_initial_free_bytes >= cuda_free_bytes
		? state->cuda_initial_free_bytes - (uint64_t)cuda_free_bytes : 0u;
	stats->cuda_builder_allocation_bytes = state->cuda_builder_allocation_bytes;
	stats->cuda_largest_allocation_bytes = state->cuda_largest_allocation_bytes;
	stats->host_mapped_allocation_bytes = state->host_mapped_allocation_bytes;
	return SPARK_STATUS_OK;
}

static const SparkGlm52Pp13NodeContextBuilderInterface SparkGlm52Pp13BuilderInterface =
{
	SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_ABI_VERSION,
	SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_INTERFACE_BYTES,
	SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_REQUIRED_PRODUCTION_CAPS,
	0u,
	SparkGlm52Pp13BuilderInitialize,
	SparkGlm52Pp13BuilderDestroy,
	SparkGlm52Pp13BuilderBuild,
	SparkGlm52Pp13BuilderDestroyResult,
	SparkGlm52Pp13BuilderAttachDriver,
	SparkGlm52Pp13BuilderPrefill,
	SparkGlm52Pp13BuilderDecode,
	SparkGlm52Pp13BuilderSubmitWork,
	SparkGlm52Pp13BuilderProgress,
	SparkGlm52Pp13BuilderTakeDsparkDraft,
	SparkGlm52Pp13BuilderGetKvStats,
	SparkGlm52Pp13BuilderResetControlGeneration
};

extern "C" const SparkGlm52Pp13NodeContextBuilderInterface *
SparkGlm52Pp13NodeContextBuilderGetInterface(void)
{
	return &SparkGlm52Pp13BuilderInterface;
}
