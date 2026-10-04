#pragma once

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_device_copy.h"
#include "sparkpipe/spark_kv_model_table.h"
#include "sparkpipe/spark_kv_page_cache.h"
#include "sparkpipe/spark_kv_page_store.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_kv_snapshot.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd_kv_pool.h"

#define SPARK_STAGE_KV_REGION_PAGE_MAJOR 1u
#define SPARK_STAGE_KV_REGION_LAYER_MAJOR 2u
#define SPARK_STAGE_KV_MAX_REGIONS 2u
#define SPARK_STAGE_KV_SNAPSHOT_QUEUE_PAGES 64u
#define SPARK_STAGE_KV_DESTROY_SAVE_TIMEOUT_NS 20000000000ull

#define SPARK_STAGE_KV_LOCK_SITE_ADMIT 0u
#define SPARK_STAGE_KV_LOCK_SITE_RESET 1u
#define SPARK_STAGE_KV_LOCK_SITE_CONTINUITY 2u
#define SPARK_STAGE_KV_LOCK_SITE_CLAIM 3u
#define SPARK_STAGE_KV_LOCK_SITE_FINISH 4u
#define SPARK_STAGE_KV_LOCK_SITE_PUBLISH 5u
#define SPARK_STAGE_KV_LOCK_SITE_SAVE 6u
#define SPARK_STAGE_KV_LOCK_SITE_RESTORE 7u
#define SPARK_STAGE_KV_LOCK_SITE_COUNT 8u

#define SPARK_STAGE_KV_POOL_TARGET_CHUNKS 1024u
#define SPARK_STAGE_KV_POOL_TICK_NS 50000000ull
#define SPARK_STAGE_KV_POOL_IDLE_TICKS 4u

#define SPARK_STAGE_KV_RESTORE_SLOTS 16u
#define SPARK_STAGE_KV_RESTORE_FREE 0u
#define SPARK_STAGE_KV_RESTORE_QUEUED 1u
#define SPARK_STAGE_KV_RESTORE_RUNNING 2u
#define SPARK_STAGE_KV_RESTORE_DONE 3u

typedef struct SparkStageKvLockSite
{
	uint64_t count;
	uint64_t total_ns;
	uint64_t max_ns;
} SparkStageKvLockSite;

typedef struct SparkStageKvBindingCounters
{
	SparkStageKvLockSite lock_sites[SPARK_STAGE_KV_LOCK_SITE_COUNT];
	uint64_t entry_count;
	uint64_t entry_total_ns;
	uint64_t entry_max_ns;
	uint64_t completion_count;
	uint64_t completion_queue_total_ns;
	uint64_t completion_queue_max_ns;
	uint64_t copy_on_write_count;
	uint64_t copy_on_write_bytes;
	uint64_t save_worker_page_count;
	uint64_t save_worker_failure_count;
} SparkStageKvBindingCounters;

typedef void (*SparkStageKvBindingFinishedFunction)(void *context,SparkStatus status);

typedef struct SparkStageKvBindingCompletion
{
	uint32_t lane_count;
	uint32_t extra_tokens;
	SparkStatus status;
	uint32_t reserved0;
	const uint32_t *resident_slots;
	const uint8_t *bound;
	const uint64_t *sequence_ids;
	const uint64_t *next_positions;
	SparkStageKvBindingFinishedFunction finished_function;
	void *finished_context;
} SparkStageKvBindingCompletion;

typedef struct SparkStageKvBindingCompletionRecord
{
	uint32_t state;
	uint32_t lane_count;
	uint32_t extra_tokens;
	SparkStatus status;
	uint64_t queued_ns;
	uint32_t *resident_slots;
	uint8_t *bound;
	uint64_t *sequence_ids;
	uint64_t *next_positions;
	SparkStageKvBindingFinishedFunction finished_function;
	void *finished_context;
} SparkStageKvBindingCompletionRecord;

typedef struct SparkStageKvRegion
{
	uint32_t layout;
	uint32_t layer_count;
	uint64_t layer_page_bytes;
} SparkStageKvRegion;

typedef struct SparkStageKvLayoutIdentity
{
	const char *model_id;
	const char *model_revision;
	uint8_t pack_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint8_t contract_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint8_t driver_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint32_t expert_codec;
	uint32_t kv_codec;
	SparkKvShard context_shard;
	uint32_t block_token_count;
	uint32_t region_count;
	SparkStageKvRegion regions[SPARK_STAGE_KV_MAX_REGIONS];
	uint64_t page_bytes;
	uint64_t state_page_bytes;
} SparkStageKvLayoutIdentity;

#define SPARK_STAGE_KV_RECURRENT_TO_BUFFER 1u
#define SPARK_STAGE_KV_RECURRENT_FROM_BUFFER 2u
#define SPARK_STAGE_KV_LANE_STATE_RESTORE_READY 0x1u
#define SPARK_STAGE_KV_LANE_STATE_CAPTURED 0x2u
#define SPARK_STAGE_KV_LANE_STATE_LOADING 0x4u

typedef SparkStatus (*SparkStageKvRecurrentCopyFunction)(void *context,uint32_t direction,uint32_t resident_slot,void *buffer,uint64_t bytes,void *stream);

typedef struct SparkStageKvRecurrent
{
	uint64_t lane_bytes;
	SparkStageKvRecurrentCopyFunction copy;
	void *context;
} SparkStageKvRecurrent;

typedef struct SparkStageKvRecurrentCounters
{
	uint64_t restores;
	uint64_t restore_bytes;
	uint64_t restore_ns;
	uint64_t captures;
	uint64_t capture_bytes;
	uint64_t capture_ns;
} SparkStageKvRecurrentCounters;

#define SPARK_STAGE_KV_POOL_SEAL_MAGIC UINT64_C(0x4c4145534c4f4f50)
#define SPARK_STAGE_KV_POOL_SEAL_VERSION 1u

typedef struct SparkStageKvPoolSeal
{
	uint64_t magic;
	uint32_t version;
	uint32_t sealed;
	uint64_t pool_generation;
	uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint32_t physical_page_count;
	uint32_t record_count;
	uint64_t page_bytes;
} SparkStageKvPoolSeal;

typedef struct SparkStageKvRestoreSlot
{
	SparkModelDriverCacheIdentity identity;
	uint32_t token_count;
	uint32_t state;
	SparkStatus status;
	uint32_t reserved0;
	uint64_t done_ns;
} SparkStageKvRestoreSlot;

typedef struct SparkStageKvBinding SparkStageKvBinding;
typedef SparkStatus (*SparkStageKvInspectFunction)(void *context,const SparkStageKvBinding *binding);

typedef struct SparkStageKvConfiguration
{
	const char *module_tag;
	uint32_t block_token_count;
	uint32_t region_count;
	SparkStageKvRegion regions[SPARK_STAGE_KV_MAX_REGIONS];
	uint32_t arena_kv_head_count;
	uint32_t arena_head_dim;
	uint32_t arena_bytes_per_scalar;
	SparkKvCacheCapacityRequest capacity_request;
	const char *model_id;
	const char *model_revision;
	uint32_t resident_sequence_capacity;
	uint32_t max_sequence_positions;
	uint32_t max_input_row_count;
	uint32_t logical_page_count;
	uint32_t physical_page_count;
	uint32_t pipeline_slot_count;
	SparkKvShard context_shard;
	const char *backing_directory;
	uint64_t backing_maximum_bytes;
	const char *snapshot_directory;
	uint64_t snapshot_maximum_bytes;
	uint8_t pack_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint8_t contract_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint32_t expert_codec;
	uint32_t kv_codec;
	const void *driver_symbol;
	SparkStageKvRecurrent recurrent;
} SparkStageKvConfiguration;

struct SparkStageKvBinding
{
	const char *module_tag;
	uint32_t block_token_count;
	uint32_t region_count;
	uint32_t pages_per_sequence;
	uint32_t logical_page_count;
	uint32_t physical_page_count;
	uint32_t resident_sequence_capacity;
	uint32_t max_sequence_positions;
	uint32_t pipeline_slot_count;
	SparkKvShard context_shard;
	SparkStageKvRegion regions[SPARK_STAGE_KV_MAX_REGIONS];
	uint8_t *region_base[SPARK_STAGE_KV_MAX_REGIONS];
	uint64_t region_packed_page_bytes[SPARK_STAGE_KV_MAX_REGIONS];
	uint64_t region_layer_stride_bytes[SPARK_STAGE_KV_MAX_REGIONS];
	uint64_t page_bytes;
	uint32_t *page_table;
	uint32_t *page_table_shadow;
	SparkKvCacheArena arena;
	SparkKvPageCache page_cache;
	SparkKvPageStore page_store;
	SparkKvLaneTransactions transactions;
	SparkKvLaneTransaction *lanes;
	uint32_t *logical_pages;
	uint32_t *physical_pages;
	SparkKvCacheBlock *blocks;
	uint32_t *resident_slot_logical_block_indices;
	SparkKvPageCacheEntry *entries;
	SparkKvPageCacheSequence *sequences;
	uint32_t *hash_bucket_heads;
	uint32_t *entry_indices_by_logical_page;
	uint8_t *staging;
	atomic_uchar *lane_bound;
	atomic_ullong *lane_sequence_ids;
	atomic_ullong *lane_next_positions;
	atomic_ullong *lane_rewind_floors;
	atomic_ullong *lane_rewind_ceilings;
	atomic_ullong *lane_pending_floors;
	pthread_mutex_t mutex;
	uint32_t mutex_initialized;
	uint64_t control_generation;
	uint64_t reset_generation;
	void *copy_stream;
	SparkKvDeviceCopier copier;
	uint32_t copier_initialized;
	SparkStageKvBindingCompletionRecord *completion_records;
	uint32_t *completion_queue;
	uint32_t completion_head;
	uint32_t completion_count;
	uint32_t completion_running;
	uint32_t completion_stop;
	pthread_mutex_t completion_mutex;
	pthread_cond_t completion_ready;
	pthread_cond_t completion_idle;
	pthread_t completion_thread;
	uint32_t completion_started;
	pthread_cond_t save_ready;
	pthread_cond_t save_idle;
	pthread_t save_thread;
	uint32_t save_started;
	pthread_cond_t restore_ready;
	pthread_t restore_thread;
	uint32_t restore_started;
	uint32_t restore_stop;
	uint32_t restore_ready_initialized;
	uint32_t reserved_restore;
	SparkStageKvRestoreSlot restores[SPARK_STAGE_KV_RESTORE_SLOTS];
	SparkKvPageCacheRestoreJob restore_job;
	uint64_t restore_jobs;
	uint64_t restore_pending_answers;
	uint64_t restore_hints;
	uint64_t restore_hinted_jobs;
	uint64_t restore_imported_pages;
	uint32_t save_stop;
	uint32_t save_running;
	uint32_t sync_initialized;
	SparkStageKvBindingCounters counters;
	uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES];
	char layout_hex[SPARK_SHA256_HEX_BYTES];
	char driver_path[SPARK_KV_SNAPSHOT_PATH_BYTES];
	uint64_t snapshot_page_file_bytes;
	SparkKvSnapshotStore snapshot_store;
	SparkWeightdKvPoolMapping kv_pool;
	uint32_t *kv_pool_chunk_need;
	uint32_t kv_pool_minimum_chunks;
	uint32_t kv_pool_wanted_chunks;
	uint32_t pool_stop;
	uint32_t pool_started;
	uint32_t pool_wake_initialized;
	uint32_t pool_idle_ticks;
	uint32_t pool_last_status;
	pthread_cond_t pool_wake;
	pthread_t pool_thread;
	uint64_t pool_mapped_bytes;
	uint64_t pool_pressure_mark;
	uint64_t pool_grow_count;
	uint64_t pool_shrink_count;
	uint64_t pool_vacated_pages;
	SparkKvWriteBudget write_budget;
	uint32_t kv_pool_adopted_pages;
	uint32_t kv_pool_sealed_pages;
	uint32_t kv_pool_seal_cleared;
	SparkKvPageCacheSnapshot snapshot;
	SparkKvPageCacheSnapshotLink *snapshot_links;
	uint8_t *snapshot_page;
	uint32_t *snapshot_pending;
	SparkKvPageCacheSaveOrder *save_order;
	SparkStageKvRecurrent recurrent;
	SparkKvPageStore state_store;
	uint8_t *state_staging;
	uint8_t *lane_state;
	uint8_t *lane_state_flags;
	uint8_t *snapshot_state;
	uint32_t state_slot_count;
	uint32_t reserved_recurrent;
	uint64_t page_store_backing_bytes;
	atomic_ullong recurrent_restores;
	atomic_ullong recurrent_restore_bytes;
	atomic_ullong recurrent_restore_ns;
	atomic_ullong recurrent_captures;
	atomic_ullong recurrent_capture_bytes;
	atomic_ullong recurrent_capture_ns;
};

SparkStatus SparkStageKvBindingInitialize(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration);
void SparkStageKvBindingDestroy(SparkStageKvBinding *binding);
SparkStatus SparkStageKvBindingAdmit(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkStageKvBindingReset(SparkStageKvBinding *binding,uint64_t generation);
SparkStatus SparkStageKvBindingAdmitReset(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision,atomic_uint *slot_states,atomic_uint *lane_states,void *stream);
uint64_t SparkStageKvBindingLaneSequence(const SparkStageKvBinding *binding,uint32_t slot);
SparkStatus SparkStageKvBindingContinuity(SparkStageKvBinding *binding,const atomic_uint *lane_states,uint32_t row_count,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions);
SparkStatus SparkStageKvBindingClaim(SparkStageKvBinding *binding,const SparkModelDriverFrame *frame,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,const uint64_t *next_positions);
SparkStatus SparkStageKvBindingUploadPageTables(SparkStageKvBinding *binding,const uint32_t *resident_slots,uint32_t lane_count,void *stream);
SparkStatus SparkStageKvBindingFinishAsync(SparkStageKvBinding *binding,uint32_t dispatch_slot,const SparkStageKvBindingCompletion *completion);
SparkStatus SparkStageKvBindingFinishWait(SparkStageKvBinding *binding,uint32_t dispatch_slot,const SparkStageKvBindingCompletion *completion);
SparkStatus SparkStageKvBindingRecurrentRestore(SparkStageKvBinding *binding,const uint32_t *resident_slots,uint32_t lane_count,void *stream);
SparkStatus SparkStageKvBindingRecurrentCapture(SparkStageKvBinding *binding,const uint32_t *resident_slots,uint32_t lane_count,void *stream);
SparkStatus SparkStageKvBindingCopyPage(const SparkStageKvBinding *binding,uint32_t direction,uintptr_t device_address,void *host,uint64_t bytes);
SparkStatus SparkStageKvBindingInspect(SparkStageKvBinding *binding,SparkStageKvInspectFunction inspect,void *context);
uint32_t SparkStageKvBindingResetIsNew(SparkStageKvBinding *binding,uint64_t generation);
void SparkStageKvBindingTakeRecurrentCounters(SparkStageKvBinding *binding,SparkStageKvRecurrentCounters *counters);
SparkStatus SparkStageKvBindingFenceExecution(SparkStageKvBinding *binding,void *stream);
SparkStatus SparkStageKvBindingQuiesce(SparkStageKvBinding *binding,uint64_t timeout_ns);
void SparkStageKvBindingStop(SparkStageKvBinding *binding);
SparkStatus SparkStageKvBindingSampleCounters(SparkStageKvBinding *binding,SparkStageKvBindingCounters *counters);
SparkStatus SparkStageKvBindingPublishFrame(SparkStageKvBinding *binding,SparkModelDriverFrame *frame,atomic_uint *lane_states);
uint32_t SparkStageKvBindingResidentCount(const SparkStageKvBinding *binding);
SparkStatus SparkStageKvLayoutDigest(const SparkStageKvLayoutIdentity *identity,uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES],const char **invalid_input);
SparkStatus SparkStageKvBindingSaveAll(SparkStageKvBinding *binding,uint64_t deadline_ns,uint32_t *saved_out,uint32_t *unsaved_out,uint32_t *ineligible_out);
void SparkStageKvBindingKvStoreCounters(SparkStageKvBinding *binding,SparkModelDriverKvStoreCounters *counters);

static inline uint32_t SparkStageKvBindingContextSharded(const SparkStageKvConfiguration *configuration)
{
	return(configuration->context_shard.degree > 1u ? 1u : 0u);
}
