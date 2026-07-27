#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_glm52_dspark.h"
#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_glm52_mtp_tree.h"
#include "sparkpipe/spark_glm52_serving_engine.h"
#include "sparkpipe/spark_glm52_stage_plan.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION 14u
#define SPARK_GLM52_RING_WORK_CONTROL_PACKET_MAGIC 0x35574350u
#define SPARK_GLM52_RING_WORK_CONTROL_STANDALONE_GENERATION UINT64_C(1)
#define SPARK_GLM52_RING_WORK_CONTROL_PACKET_BYTES \
	((uint32_t)sizeof(SparkGlm52RingWorkControlPacket))
#define SPARK_GLM52_RING_WORK_CONTROL_PACKET_PREFIX_BYTES \
	((uint32_t)offsetof(SparkGlm52RingWorkControlPacket,lanes))
#define SPARK_GLM52_RING_WORK_CONTROL_KV_STATE_BYTES \
	((uint32_t)sizeof(SparkGlm52RingWorkControlKvState))
#define SPARK_GLM52_RING_WORK_CONTROL_LANE_BYTES \
	((uint32_t)sizeof(SparkGlm52RingWorkControlLane))
#define SPARK_GLM52_RING_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT \
	SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET
#define SPARK_GLM52_RING_WORK_CONTROL_KV_CONTEXT_TOKEN_CAPACITY \
	SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS
#define SPARK_GLM52_RING_WORK_CONTROL_KV_BLOCK_CAPACITY \
	(SPARK_GLM52_RING_WORK_CONTROL_KV_CONTEXT_TOKEN_CAPACITY / \
	 SPARK_GLM52_KV_BLOCK_TOKENS)

#define SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL 0x00000001u
#define SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_DRAFT 0x00000002u
#define SPARK_GLM52_RING_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE 0x00000004u
#define SPARK_GLM52_RING_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY 0x00000008u
#define SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY 0x00000010u
#define SPARK_GLM52_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES 0x00000020u
#define SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_RESOLVE 0x00000040u
#define SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_TREE_VERIFY 0x00000080u
#define SPARK_GLM52_RING_WORK_CONTROL_KNOWN_FLAGS \
	(SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL | \
	 SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_DRAFT | \
	 SPARK_GLM52_RING_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE | \
	 SPARK_GLM52_RING_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY | \
	 SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY | \
	 SPARK_GLM52_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES | \
	 SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_RESOLVE | \
	 SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_TREE_VERIFY)
#define SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT \
	((SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT > \
	  SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT) ? \
	 SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT : \
	 SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT)
#define SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT 1024u
#define SPARK_GLM52_RING_WORK_CONTROL_MAX_PREFILL_TOKENS_PER_PACKET \
	SPARK_GLM52_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH
#define SPARK_GLM52_RING_WORK_CONTROL_INVALID_REQUEST_SLOT UINT32_MAX

#define SPARK_GLM52_RING_KV_ENTRY_MISSING 0u
#define SPARK_GLM52_RING_KV_ENTRY_IN_FLIGHT 1u
#define SPARK_GLM52_RING_KV_ENTRY_RESIDENT 2u
#define SPARK_GLM52_RING_KV_ENTRY_TRANSIENT 3u


#define SPARK_GLM52_RING_KV_DIRECTORY_RESIDENCY_GPU 1u
#define SPARK_GLM52_RING_KV_DIRECTORY_RESIDENCY_NVME 2u
#define SPARK_GLM52_RING_KV_INVALID_BLOCK_INDEX UINT32_MAX

// 128-bit block identity. The low bit of `high` is the domain tag: 1 marks a
// private key derived from (sequence_id,logical_block_index), 0 marks a content
// key derived from the token prefix a block holds. The two domains can never
// alias, and an all-zero key is reserved to mean "empty slot".
typedef struct SparkGlm52RingKvKey
{
	uint64_t low;
	uint64_t high;
} SparkGlm52RingKvKey;

typedef SparkStatus (*SparkGlm52RingWorkControlKvSwapStoreFunction)(
	void *context,
	SparkGlm52RingKvKey key,
	uint32_t physical_block_index,
	uint32_t backing_block_index);
typedef SparkStatus (*SparkGlm52RingWorkControlKvSwapLoadFunction)(
	void *context,
	SparkGlm52RingKvKey key,
	uint32_t physical_block_index,
	uint32_t backing_block_index);

// One sequence slot. Names the block record it resolves to and owns no storage,
// so many sequence slots may name one block.
typedef struct SparkGlm52RingWorkControlKvDirectoryEntry
{
	SparkGlm52RingKvKey key;
	SparkGlm52RingKvKey block_key;
} SparkGlm52RingWorkControlKvDirectoryEntry;

// One block. Sole owner of residency, backing storage and the share refcount,
// so an eviction is observed by every sequence that names this block.
typedef struct SparkGlm52RingWorkControlKvBlockEntry
{
	SparkGlm52RingKvKey key;
	uint32_t physical_block_index;
	uint32_t backing_block_index;
	uint32_t reference_count;
	uint32_t residency_state : 2;
	uint32_t backing_valid : 1;
	uint32_t prefetch_mark : 29;
} SparkGlm52RingWorkControlKvBlockEntry;

typedef struct SparkGlm52RingWorkControlKvPrefetchEntry
{
	SparkGlm52RingKvKey key;
	uint32_t backing_block_index;
} SparkGlm52RingWorkControlKvPrefetchEntry;

typedef struct SparkGlm52RingWorkControlLane
{
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint32_t request_slot_index;
	uint32_t context_token_count;
	uint32_t input_token_id;
	uint32_t mtp_draft_token_count;
	uint32_t speculative_token_count;
	uint8_t mtp_resolution_proposed_token_count;
	uint8_t mtp_resolution_accepted_token_count;
	uint16_t mtp_resolution_path_id;
	uint32_t speculative_draft_token_ids[
		SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT];
} SparkGlm52RingWorkControlLane;

typedef struct SparkGlm52RingWorkControlPacket
{
	uint32_t magic;
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t flags;
	uint64_t request_id;
	uint64_t sequence_id;
	uint64_t sequence_position;
	uint64_t deadline_time_ns;
	uint64_t control_generation;
	uint32_t active_sequence_count;
	uint32_t new_token_count;
	uint32_t pipeline_slot;
	uint32_t priority;
	uint32_t block_token_count;
	uint32_t kv_block_table_token_count;
	uint32_t max_blocks_per_sequence;
	uint32_t mtp_draft_token_count;
	uint32_t input_token_id;
	uint32_t speculative_token_count;
	uint32_t speculative_token_index;
	uint32_t speculative_draft_token_ids[
		SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT];
	uint32_t lane_count;
	uint32_t rows_per_lane;
	uint32_t execution_row_count;
	uint32_t execution_batch_bucket;
	uint32_t prefill_token_ids[
		SPARK_GLM52_RING_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkGlm52RingWorkControlLane
		lanes[SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT];
} SparkGlm52RingWorkControlPacket;

typedef struct SparkGlm52RingWorkControlKvState
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t lane_capacity;
	uint32_t lane_stride;
	uint32_t block_token_count;
	uint32_t table_entry_capacity;
	uint32_t physical_block_capacity;
	uint32_t directory_capacity;
	uint32_t block_entry_capacity;
	uint32_t lane_block_key_stride;
	uint32_t next_physical_block_index;
	uint32_t backing_block_capacity;
	uint32_t free_backing_block_head;
	uint32_t directory_entry_count;
	uint32_t block_entry_count;
	uint32_t swapped_block_count;
	uint32_t clean_evict_count;
	uint64_t epoch;
	uint64_t control_generation;
	uint64_t control_generation_reset_count;
	uint32_t *physical_block_indices;
	uint32_t *lane_physical_block_counts;
	uint8_t *physical_block_states;
	SparkGlm52RingKvKey *physical_block_keys;
	uint64_t *physical_block_last_used_epochs;
	uint32_t *physical_block_pin_counts;
	SparkGlm52RingWorkControlKvDirectoryEntry *directory_entries;
	SparkGlm52RingWorkControlKvBlockEntry *block_entries;
	const SparkGlm52RingKvKey *lane_block_keys;
	uint32_t *backing_block_free_next;
	SparkGlm52RingWorkControlKvSwapStoreFunction swap_store_function;
	SparkGlm52RingWorkControlKvSwapLoadFunction swap_load_function;
	void *swap_context;
	uint32_t missing_block_count;
	uint32_t in_flight_block_count;
	uint32_t resident_block_count;
	uint32_t allocated_physical_block_count;
	uint64_t swap_store_count;
	uint64_t swap_load_count;
	uint32_t prefetch_generation;
	uint64_t share_hit_count;
	uint64_t share_admit_count;
} SparkGlm52RingWorkControlKvState;

SparkStatus SparkGlm52RingWorkControlValidatePacket(
	const SparkGlm52RingWorkControlPacket *packet,
	uint32_t max_active_sequence_count,
	uint32_t max_pipeline_slot_count);
uint32_t SparkGlm52RingWorkControlCalculatePacketBytes(
	uint32_t active_sequence_count);
SparkStatus SparkGlm52RingWorkControlSelectExecutionBatchBucket(
	const SparkGlm52RequestApiDispatch *request_dispatch,
	uint32_t batch_lane_or_row_count,
	uint32_t *batch_bucket_out);
SparkStatus SparkGlm52RingWorkControlSelectMtpDraftBudget(
	uint32_t dispatch_kind,
	uint32_t request_flags,
	uint32_t requested_budget,
	uint32_t *mtp_budget_out);
SparkStatus SparkGlm52RingWorkControlBuildDecodePacket(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t speculative_token_index,
	SparkGlm52RingWorkControlPacket *packet);
SparkStatus SparkGlm52RingWorkControlBuildDecodePacketRange(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t lane_offset,
	uint32_t lane_count,
	uint32_t speculative_token_index,
	SparkGlm52RingWorkControlPacket *packet);
SparkStatus SparkGlm52RingWorkControlBuildPrefillPacket(
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch,
	uint32_t token_offset,
	uint32_t token_count,
	SparkGlm52RingWorkControlPacket *packet);
SparkStatus SparkGlm52RingWorkControlSelectPrefillChunk(
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch,
	uint32_t token_offset,
	uint32_t maximum_execution_row_count,
	uint32_t *token_count_out);
SparkStatus SparkGlm52RingWorkControlInitializeKvState(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t lane_capacity,
	uint32_t lane_stride,
	uint32_t block_token_count,
	uint32_t physical_block_capacity,
	uint32_t directory_capacity,
	uint32_t block_entry_capacity,
	uint32_t *physical_block_indices,
	uint32_t *lane_physical_block_counts,
	uint8_t *physical_block_states,
	SparkGlm52RingKvKey *physical_block_keys,
	uint64_t *physical_block_last_used_epochs,
	SparkGlm52RingWorkControlKvDirectoryEntry *directory_entries,
	SparkGlm52RingWorkControlKvBlockEntry *block_entries);

// Supply per-lane content keys for the blocks the next packet will acquire.
// Leaving this unset, or leaving a key zero, keeps a block private, so sharing
// is opt-in per block and the default behaviour is unchanged.
SparkStatus SparkGlm52RingWorkControlConfigureKvSharing(
	SparkGlm52RingWorkControlKvState *state,
	const SparkGlm52RingKvKey *lane_block_keys,
	uint32_t lane_block_key_stride);

uint32_t SparkGlm52RingWorkControlKvKeyEqual(
	SparkGlm52RingKvKey left,
	SparkGlm52RingKvKey right);

SparkGlm52RingKvKey SparkGlm52RingWorkControlPrivateKey(
	uint64_t sequence_id,
	uint32_t logical_block_index);

SparkGlm52RingKvKey SparkGlm52RingWorkControlContentKey(
	uint64_t digest_low,
	uint64_t digest_high);

uint64_t SparkGlm52RingWorkControlKvCommittedFrontier(
	const SparkGlm52RingWorkControlLane *lane);
SparkStatus SparkGlm52RingWorkControlConfigureKvSwap(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t backing_block_capacity,
	uint32_t *backing_block_free_next,
	SparkGlm52RingWorkControlKvSwapStoreFunction swap_store_function,
	SparkGlm52RingWorkControlKvSwapLoadFunction swap_load_function,
	void *swap_context);
SparkStatus SparkGlm52RingWorkControlConfigureKvPins(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t *physical_block_pin_counts);
SparkStatus SparkGlm52RingWorkControlAdvanceKvGeneration(
	SparkGlm52RingWorkControlKvState *state,
	uint64_t control_generation);
SparkStatus SparkGlm52RingWorkControlPinPhysicalBlock(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t physical_block_index);
SparkStatus SparkGlm52RingWorkControlUnpinPhysicalBlock(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t physical_block_index);
SparkStatus SparkGlm52RingWorkControlAcquireTransientPhysicalBlock(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t *physical_block_index_out);
SparkStatus SparkGlm52RingWorkControlReleaseTransientPhysicalBlock(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t physical_block_index);
uint32_t SparkGlm52RingWorkControlBlockCount(
	uint32_t token_count,
	uint32_t block_token_count);
SparkStatus SparkGlm52RingWorkControlCollectKvPrefetchEntries(
	const SparkGlm52RingWorkControlPacket *packets,
	uint32_t packet_count,
	SparkGlm52RingWorkControlKvState *state,
	SparkGlm52RingWorkControlKvPrefetchEntry *entries,
	uint32_t entry_capacity,
	uint32_t *entry_count_out);
SparkStatus SparkGlm52RingWorkControlPlanExecutionChunks(
	uint32_t logical_lane_count,
	uint32_t rows_per_lane,
	uint32_t execution_row_capacity,
	uint32_t *maximum_lanes_per_chunk_out,
	uint32_t *chunk_count_out);
SparkStatus SparkGlm52RingWorkControlBuildHostKvBlockTable(
	const SparkGlm52RingWorkControlPacket *packet,
	SparkGlm52RingWorkControlKvState *state,
	SparkGlm52KvBlockTableView *view);
SparkStatus SparkGlm52RingWorkControlCommitHostKvBlockTable(
	const SparkGlm52RingWorkControlPacket *packet,
	SparkGlm52RingWorkControlKvState *state);
SparkStatus SparkGlm52RingWorkControlCancelHostKvBlockTable(
	const SparkGlm52RingWorkControlPacket *packet,
	SparkGlm52RingWorkControlKvState *state);
SparkStatus SparkGlm52RingWorkControlReleaseSequence(
	SparkGlm52RingWorkControlKvState *state,
	uint64_t sequence_id,
	uint32_t logical_block_count);
SparkStatus SparkGlm52RingWorkControlReleasePacketSequences(
	const SparkGlm52RingWorkControlPacket *packet,
	SparkGlm52RingWorkControlKvState *state);

#ifdef __cplusplus
}
#endif
