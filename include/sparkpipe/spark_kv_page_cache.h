#pragma once

#include <stdint.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_page_store.h"
#include "sparkpipe/spark_kv_snapshot.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_PAGE_CACHE_ABI_VERSION 11u
#define SPARK_KV_PAGE_CACHE_NO_INDEX UINT32_MAX
#define SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID UINT32_C(0x00000001)
#define SPARK_KV_PAGE_CACHE_ENTRY_FLAG_STATELESS UINT32_C(0x00000002)
#define SPARK_KV_PAGE_CACHE_ENTRY_FLAG_SAVED UINT32_C(0x00000004)
#define SPARK_KV_PAGE_CACHE_ENTRY_FLAG_PRIVATE UINT32_C(0x00000008)
#define SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES 32u
#define SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_CHAIN 1u
#define SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_PAGES 2u
#define SPARK_KV_PAGE_CACHE_SNAPSHOT_KIND_STATE 3u
#define SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE UINT32_C(0x00000001)
#define SPARK_KV_PAGE_CACHE_MUTATION_ALLOCATED_MUTABLE UINT32_C(0x00000002)
#define SPARK_KV_PAGE_CACHE_MUTATION_EXTENDED_MUTABLE UINT32_C(0x00000004)
#define SPARK_KV_PAGE_CACHE_KNOWN_MUTATIONS \
	(SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE | \
	 SPARK_KV_PAGE_CACHE_MUTATION_ALLOCATED_MUTABLE | \
	 SPARK_KV_PAGE_CACHE_MUTATION_EXTENDED_MUTABLE)


typedef struct SparkKvPageCacheEntry
{
	uint32_t flags;
	uint32_t token_count;
	uint32_t page_count;
	uint32_t parent_entry_index;
	uint32_t logical_page_index;
	uint32_t reference_count;
	uint32_t hash_next;
	uint32_t free_next;
	uint32_t lru_prev;
	uint32_t lru_next;
	uint32_t priority;
	uint32_t reserved0;
	uint64_t last_used_epoch;
	SparkModelDriverCacheIdentity identity;
}
SparkKvPageCacheEntry;

typedef struct SparkKvPageCacheSequence
{
	uint64_t sequence_id;
	uint64_t generation;
	uint32_t next_token_position;
	uint32_t terminal_entry_index;
	uint32_t mutable_logical_page_index;
	uint32_t mutable_first_token_index;
	uint32_t mutable_page_count;
	uint32_t mutable_following_pages[SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES - 1u];
	uint32_t mutable_block_identity_count;
	SparkModelDriverCacheIdentity mutable_block_identities[SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES - 1u];
}
SparkKvPageCacheSequence;

typedef struct SparkKvPageCacheSnapshotLink
{
	uint32_t token_count;
	uint32_t reserved0;
	SparkModelDriverCacheIdentity identity;
}
SparkKvPageCacheSnapshotLink;

typedef struct SparkKvPageCacheSnapshot
{
	SparkKvSnapshotStore *store;
	uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint32_t page_capacity;
	uint32_t reserved0;
	SparkKvPageCacheSnapshotLink *links;
	uint8_t *page;
	uint8_t *state;
	uint64_t save_count;
	uint64_t save_page_count;
	uint64_t save_ns;
	uint64_t save_failure_count;
	uint64_t save_deferred_count;
	uint64_t restore_count;
	uint64_t restore_page_count;
	uint64_t restore_ns;
	uint64_t restore_miss_count;
	uint64_t restore_corrupt_count;
	uint64_t restore_read_error_count;
	uint64_t restore_failure_count;
	SparkStatus last_save_status;
	SparkStatus last_restore_status;
	uint32_t *pending_terminals;
	uint32_t pending_capacity;
	uint32_t pending_head;
	uint32_t pending_count;
	uint32_t in_flight;
	uint32_t full_logged;
	uint32_t head_saved_pages;
	uint32_t reserved1;
	uint64_t head_start_ns;
	uint64_t save_mark_count;
	uint64_t save_skipped_count;
	uint64_t save_cancelled_count;
	uint64_t evicted_unsaved_count;
	uint64_t demote_queued_count;
}
SparkKvPageCacheSnapshot;

#define SPARK_KV_PAGE_CACHE_RESTORE_DONE 0u
#define SPARK_KV_PAGE_CACHE_RESTORE_NEED_PAGE 1u
#define SPARK_KV_PAGE_CACHE_RESTORE_NEED_STATE 2u

typedef struct SparkKvPageCacheRestoreJob
{
	SparkModelDriverCacheIdentity identity;
	uint32_t token_count;
	uint32_t page_count;
	uint32_t next_page;
	uint32_t parent;
	uint32_t need;
	uint32_t imported_pages;
	uint32_t read_errors;
	uint32_t reserved0;
	uint64_t start_ns;
	SparkKvPageCacheSnapshotLink *links;
	uint8_t *page;
	uint8_t *state;
}
SparkKvPageCacheRestoreJob;

typedef struct SparkKvPageCacheResidentRecord
{
	SparkModelDriverCacheIdentity identity;
	uint32_t token_count;
	uint32_t page_count;
	uint32_t resident_slot;
	uint32_t parent_record;
	uint32_t flags;
	uint32_t reserved0;
}
SparkKvPageCacheResidentRecord;

typedef struct SparkKvPageCacheSaveOrder
{
	uint64_t last_used_epoch;
	uint32_t entry_index;
	uint32_t reserved0;
}
SparkKvPageCacheSaveOrder;

#define SPARK_KV_PAGE_CACHE_SAVE_DEVICE 1u
#define SPARK_KV_PAGE_CACHE_SAVE_BACKING 2u

typedef struct SparkKvPageCacheSaveWork
{
	uint32_t terminal_entry_index;
	uint32_t entry_index;
	uint32_t logical_page_index;
	uint32_t source;
	uint32_t pinned;
	uint32_t reserved0;
	uint64_t generation;
	uintptr_t key_device_address;
	uint64_t key_bytes;
	uintptr_t value_device_address;
	uint64_t value_bytes;
	uint8_t *page;
	uint8_t *state;
	SparkKvSnapshotWriteTicket ticket;
}
SparkKvPageCacheSaveWork;

typedef struct SparkKvPageCacheConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t sequence_capacity;
	uint32_t entry_capacity;
	uint32_t hash_bucket_count;
	uint32_t reserved0;
	SparkKvCacheArena *kv_cache_arena;
	SparkKvPageStore *page_store;
	SparkKvPageCacheEntry *entries;
	SparkKvPageCacheSequence *sequences;
	uint32_t *hash_bucket_heads;
	uint32_t *entry_indices_by_logical_page;
}
SparkKvPageCacheConfiguration;

typedef SparkStatus (*SparkKvPageCacheCopyPageFunction)(void *context,uint32_t source_logical_page,uint32_t destination_logical_page);
typedef SparkStatus (*SparkKvPageCacheRetireCopiesFunction)(void *context,uint32_t require_all);
typedef uint32_t (*SparkKvPageCacheCopyPinsFunction)(void *context,uint32_t logical_page);
typedef SparkStatus (*SparkKvPageCacheDeferFreeFunction)(void *context,uint32_t logical_page);

typedef struct SparkKvPageCacheDeviceCopy
{
	SparkKvPageCacheCopyPageFunction copy_page;
	SparkKvPageCacheRetireCopiesFunction retire_copies;
	SparkKvPageCacheCopyPinsFunction destination_pins;
	SparkKvPageCacheDeferFreeFunction defer_free;
	void *context;
}
SparkKvPageCacheDeviceCopy;

typedef struct SparkKvPageCache
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t sequence_capacity;
	uint32_t entry_capacity;
	uint32_t hash_bucket_count;
	uint32_t free_entry_head;
	uint32_t live_sequence_count;
	uint32_t lru_head;
	uint32_t lru_tail;
	SparkKvCacheArena *kv_cache_arena;
	SparkKvPageStore *page_store;
	SparkKvPageStore *state_store;
	SparkKvPageCacheSnapshot *snapshot;
	SparkKvWriteBudget *write_budget;
	uint32_t admission_priority;
	uint32_t reserved_priority;
	SparkKvPageCacheEntry *entries;
	SparkKvPageCacheSequence *sequences;
	uint32_t *hash_bucket_heads;
	uint32_t *entry_indices_by_logical_page;
	uint64_t epoch;
	uint64_t prefix_hit_count;
	uint64_t prefix_miss_count;
	uint64_t published_page_count;
	uint64_t deduplicated_page_count;
	uint64_t evicted_entry_count;
	uint64_t released_sequence_count;
	SparkKvPageCacheDeviceCopy device_copy;
	uint32_t copy_unavailable_logged;
	uint32_t backing_full_logged;
	uint64_t backing_reclaim_count;
	uint64_t backing_full_count;
}
SparkKvPageCache;

#define SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkKvPageCacheConfiguration))
#define SPARK_KV_PAGE_CACHE_BYTES \
	((uint32_t)sizeof(SparkKvPageCache))

SparkStatus SparkKvPageCacheInitialize(
	SparkKvPageCache *cache,
	const SparkKvPageCacheConfiguration *configuration);
SparkStatus SparkKvPageCacheAttachStateStore(SparkKvPageCache *cache,SparkKvPageStore *store);
SparkStatus SparkKvPageCacheAttachSnapshot(SparkKvPageCache *cache,SparkKvPageCacheSnapshot *snapshot);
SparkStatus SparkKvPageCacheAttachDeviceCopy(SparkKvPageCache *cache,const SparkKvPageCacheDeviceCopy *copy);
SparkStatus SparkKvPageCacheSavePrefix(SparkKvPageCache *cache,const SparkModelDriverCacheIdentity *identity,uint32_t token_count);
SparkStatus SparkKvPageCacheSaveTake(SparkKvPageCache *cache,SparkKvPageCacheSaveWork *work);
SparkStatus SparkKvPageCacheSaveFinish(SparkKvPageCache *cache,SparkKvPageCacheSaveWork *work,SparkStatus copy_status);
SparkStatus SparkKvPageCacheSaveCopy(SparkKvPageCache *cache,SparkKvPageCacheSaveWork *work);
SparkStatus SparkKvPageCacheSaveDrain(SparkKvPageCache *cache);
uint32_t SparkKvPageCacheSavePending(const SparkKvPageCache *cache);
void SparkKvPageCacheSaveCancelAll(SparkKvPageCache *cache);
SparkStatus SparkKvPageCacheMarkAllUnsaved(SparkKvPageCache *cache,SparkKvPageCacheSaveOrder *order,uint32_t order_capacity,uint32_t *marked_out,uint32_t *deferred_out,uint32_t *ineligible_out);
SparkStatus SparkKvPageCacheCountUnsaved(const SparkKvPageCache *cache,uint32_t *unsaved_out,uint32_t *ineligible_out);
SparkStatus SparkKvPageCacheRestorePrefix(SparkKvPageCache *cache,const SparkModelDriverCacheIdentity *identity,uint32_t token_count);
SparkStatus SparkKvPageCacheRestoreBegin(SparkKvPageCache *cache,SparkKvPageCacheRestoreJob *job,const SparkModelDriverCacheIdentity *identity,uint32_t token_count);
uint32_t SparkKvPageCachePrefixReady(const SparkKvPageCache *cache,const SparkModelDriverCacheIdentity *identity,uint32_t token_count);
SparkStatus SparkKvPageCacheRestoreReadChain(const SparkKvPageCache *cache,SparkKvPageCacheRestoreJob *job);
SparkStatus SparkKvPageCacheRestoreAdvance(SparkKvPageCache *cache,SparkKvPageCacheRestoreJob *job);
SparkStatus SparkKvPageCacheRestoreRead(const SparkKvPageCache *cache,SparkKvPageCacheRestoreJob *job);
SparkStatus SparkKvPageCacheRestoreApply(SparkKvPageCache *cache,SparkKvPageCacheRestoreJob *job);
SparkStatus SparkKvPageCacheRestoreFinish(SparkKvPageCache *cache,SparkKvPageCacheRestoreJob *job,SparkStatus status);
SparkStatus SparkKvPageCacheExportResident(const SparkKvPageCache *cache,SparkKvPageCacheResidentRecord *records,uint32_t capacity,uint32_t *count_out);
SparkStatus SparkKvPageCacheAdoptResident(SparkKvPageCache *cache,const SparkKvPageCacheResidentRecord *records,uint32_t count,uint32_t *adopted_out);
SparkStatus SparkKvPageCacheEvictUnused(SparkKvPageCache *cache);
SparkStatus SparkKvPageCacheVacateResident(SparkKvPageCache *cache,uint32_t limit,uint32_t *kept_limit,uint32_t *vacated_pages);
SparkStatus SparkKvPageCachePrepareLane(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *logical_page_indices,
	uint32_t logical_page_capacity,
	uint32_t *logical_page_count_out);
SparkStatus SparkKvPageCacheResolveLanePages(
	const SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *logical_page_indices,
	uint32_t logical_page_capacity,
	uint32_t *logical_page_count_out);
SparkStatus SparkKvPageCacheGetLaneMutablePageDemand(
	const SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *mutable_page_demand_out);
SparkStatus SparkKvPageCacheBeginLane(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *mutable_logical_page_index_out);
SparkStatus SparkKvPageCacheBeginLaneTransaction(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *mutable_logical_page_index_out,
	uint32_t *mutation_flags_out);
SparkStatus SparkKvPageCacheRollbackLaneTransaction(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t mutation_flags);
SparkStatus SparkKvPageCacheCompleteLane(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane);
SparkStatus SparkKvPageCacheReleaseLane(
	SparkKvPageCache *cache,
	uint32_t resident_sequence_slot,
	uint64_t sequence_id);
SparkStatus SparkKvPageCacheReleaseAll(SparkKvPageCache *cache);
SparkStatus SparkKvPageCacheBuildLaneTable(
	SparkKvPageCache *cache,
	uint32_t resident_sequence_slot,
	uint64_t sequence_id,
	uint32_t *logical_page_indices,
	uint32_t logical_page_capacity,
	uint32_t *logical_page_count_out);

SparkStatus SparkKvPageCacheBeginPinnedLaneTransaction(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *logical_pages,
	uint32_t *physical_pages,
	uint32_t page_capacity,
	uint32_t *page_count_out,
	uint32_t *mutation_flags_out);

#define SPARK_KV_LANE_TRANSACTION_EMPTY 0u
#define SPARK_KV_LANE_TRANSACTION_PREPARED 1u
#define SPARK_KV_LANE_TRANSACTION_COMMITTED 2u
#define SPARK_KV_LANE_TRANSACTION_EXECUTING 3u

typedef struct SparkKvLaneTransaction
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverCacheLane lane;
	uint64_t validation_epoch;
	uint64_t executing_since_ns;
	uint64_t prepared_since_ns;
	uint32_t phase;
	uint32_t page_count;
	uint32_t mutation_flags;
	SparkModelDriverCacheIdentity block_identities[SPARK_KV_PAGE_CACHE_MAX_MUTABLE_PAGES - 1u];
} SparkKvLaneTransaction;

typedef struct SparkKvLaneTransactions
{
	SparkKvPageCache *cache;
	SparkKvLaneTransaction *lanes;
	uint32_t *logical_pages;
	uint32_t *physical_pages;
	uint32_t page_capacity;
	uint32_t restore_async;
	uint64_t validation_epoch;
} SparkKvLaneTransactions;

static inline uint32_t SparkKvLaneTransactionPrefixRestorePending(const SparkKvLaneTransaction *owner)
{
	return(owner != 0 && (owner->mutation_flags & SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE) != 0u && (owner->lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) != 0u && owner->lane.sequence_position != 0u);
}

SparkStatus SparkKvLaneTransactionsAdmit(
	SparkKvLaneTransactions *transactions,
	const SparkModelDriverAdmissionRequest *request);
SparkStatus SparkKvLaneTransactionsClaim(
	SparkKvLaneTransactions *transactions,
	const SparkModelDriverFrame *frame);
SparkStatus SparkKvLaneTransactionsReset(SparkKvLaneTransactions *transactions);
SparkStatus SparkKvLaneTransactionsFinish(
	SparkKvLaneTransactions *transactions,
	const uint32_t *resident_slots,
	uint32_t lane_count,
	SparkStatus execution_status,
	uint32_t extra_tokens);

#ifdef __cplusplus
}
#endif
