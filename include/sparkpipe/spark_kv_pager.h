#pragma once


#include <stdatomic.h>
#include <stdint.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_nvme_tier.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_PAGER_ABI_VERSION 1u
#define SPARK_KV_PAGER_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvPagerConfiguration))
#define SPARK_KV_PAGER_DESCRIPTOR_BYTES ((uint32_t)sizeof(SparkKvPager))
#define SPARK_KV_PAGER_BLOCK_VIEW_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvPagerBlockView))
#define SPARK_KV_PAGER_DIGEST_BYTES SPARK_NVME_TIER_DIGEST_BYTES
#define SPARK_KV_PAGER_ADMISSION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvPagerAdmission))
#define SPARK_KV_PAGER_ADMISSION_DECISION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvPagerAdmissionDecision))
#define SPARK_KV_PAGER_ADMISSION_ABI_VERSION 2u
#define SPARK_KV_PAGER_ADMISSION_ABI_VERSION_LEGACY 1u

#define SPARK_KV_PAGER_PAGE_OUT_HISTORY_CAPACITY 16u

#define SPARK_KV_PAGER_STAGING_BLOCK_COUNT 2u

#define SPARK_KV_PAGER_DEVICE_LAW_BYTES \
    (110ULL * 1024ULL * 1024ULL * 1024ULL)

#define SPARK_KV_PAGER_RESTORE_POLL_LIMIT 64u

typedef uint64_t (*SparkKvPagerClockFunction)(void *clock_context);

#define SPARK_KV_PAGER_PARK_QUEUE_CAPACITY 8u
#define SPARK_KV_PAGER_PARK_POLL_QUANTUM_MICROSECONDS 200u
#define SPARK_KV_PAGER_PARK_WORKER_HANDLE_BYTES 32u

typedef struct SparkKvPagerBlockView
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t lane_index;
    uint32_t first_block;
    uint32_t block_count;
    uint32_t reserved0;
    uint64_t key_bytes;
    uint64_t value_bytes;
    uintptr_t key_device_address;
    uintptr_t value_device_address;
    void *host_staging;
}
SparkKvPagerBlockView;

typedef SparkStatus (*SparkKvPagerModuleSaveFunction)(
    void *module_context,
    const SparkKvPagerBlockView *view);

typedef SparkStatus (*SparkKvPagerModuleRestoreFunction)(
    void *module_context,
    const SparkKvPagerBlockView *view);

typedef SparkStatus (*SparkKvPagerBackingWriteFunction)(
    void *backing_context,
    uint64_t device_offset,
    const void *host_staging,
    uint64_t bytes);

typedef struct SparkKvPagerConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
    SparkKvCacheArena *arena;
    SparkNvmeTier *tier;
    uint32_t park_budget_blocks;
    uint32_t reserved2;
    uint64_t device_budget_bytes;
    void *staging;
    uint64_t staging_bytes;
    void *module_context;
    SparkKvPagerModuleSaveFunction module_save;
    SparkKvPagerModuleRestoreFunction module_restore;
    void *backing_context;
    SparkKvPagerBackingWriteFunction backing_write;
    void *clock_context;
    SparkKvPagerClockFunction clock_function;
    uint32_t park_queue_blocks;
    uint32_t reserved3;
    void *park_staging;
    uint64_t park_staging_bytes;
    uint32_t park_policy;
}
SparkKvPagerConfiguration;

#define SPARK_KV_PAGER_PARK_POLICY_LRU \
    SPARK_KV_CACHE_EVICTION_POLICY_LRU
#define SPARK_KV_PAGER_PARK_POLICY_REUSE_VALUE \
    SPARK_KV_CACHE_EVICTION_POLICY_REUSE_VALUE

typedef struct SparkKvPagerParkEntry
{
    uint32_t logical_block_index;
    uint32_t reserved0;
    SparkNvmeTierWriteReservation reservation;
    uint8_t *staging;
    uint64_t payload_bytes;
}
SparkKvPagerParkEntry;

typedef struct SparkKvPagerParkCompletion
{
    uint32_t logical_block_index;
    uint32_t status;
    SparkNvmeTierWriteReservation reservation;
    uint64_t write_elapsed_microseconds;
}
SparkKvPagerParkCompletion;

typedef struct SparkKvPagerStatistics
{
    uint64_t page_out_count;
    uint64_t page_out_bytes;
    uint64_t page_out_deduplicated;
    uint64_t page_in_count;
    uint64_t page_in_bytes;
    uint64_t page_in_misses;
    uint64_t admission_requests;
    uint64_t admission_accepted;
    uint64_t admission_queued_device;
    uint64_t admission_queued_backing;
    uint64_t dispatch_requests;
    uint64_t dispatch_ready;
    uint64_t dispatch_queued;
    uint64_t dispatch_recompute;
    uint64_t admission_queued_bandwidth;
    uint64_t park_completions_published;
    uint64_t park_write_failures;
    uint64_t measured_bandwidth_samples;
    uint64_t measured_bytes_per_second;
    uint32_t park_evictions;
    uint32_t page_out_history_count;
    uint32_t page_out_history[SPARK_KV_PAGER_PAGE_OUT_HISTORY_CAPACITY];
}
SparkKvPagerStatistics;

typedef struct SparkKvPager
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t block_bytes;
    uint32_t park_budget_blocks;
    SparkKvPagerConfiguration configuration;
    void *landing_staging;
    SparkKvPagerStatistics statistics;
    SparkKvPagerParkEntry park_queue[SPARK_KV_PAGER_PARK_QUEUE_CAPACITY];
    SparkKvPagerParkCompletion
        park_completions[SPARK_KV_PAGER_PARK_QUEUE_CAPACITY];
    atomic_uint park_head;
    atomic_uint park_tail;
    atomic_uint park_completion_head;
    atomic_uint park_completion_tail;
    atomic_uint park_worker_stop;
    atomic_uint park_write_inflight_valid;
    SparkKvPagerParkEntry park_write_inflight;
    uint32_t park_worker_active;
    uint32_t park_queue_blocks;
    uint8_t park_worker_handle[SPARK_KV_PAGER_PARK_WORKER_HANDLE_BYTES];
}
SparkKvPager;

typedef struct SparkKvPagerAdmission
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t block_demand;
    uint32_t restore_slack_microseconds;
}
SparkKvPagerAdmission;

typedef enum SparkKvPagerAdmissionOutcome
{
    SPARK_KV_PAGER_ADMITTED = 0,
    SPARK_KV_PAGER_QUEUED = 1
}
SparkKvPagerAdmissionOutcome;

typedef struct SparkKvPagerAdmissionDecision
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t outcome;
    uint32_t block_demand;
    uint32_t park_evictions;
    uint32_t reservation_held;
}
SparkKvPagerAdmissionDecision;

#define SPARK_KV_PAGER_DISPATCH_ABI_VERSION 1u
#define SPARK_KV_PAGER_DISPATCH_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvPagerDispatch))
#define SPARK_KV_PAGER_DISPATCH_DECISION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvPagerDispatchDecision))

typedef enum SparkKvPagerDispatchOutcome
{
    SPARK_KV_PAGER_DISPATCH_READY = 0,
    SPARK_KV_PAGER_DISPATCH_QUEUED = 1,
    SPARK_KV_PAGER_DISPATCH_RECOMPUTE = 2
}
SparkKvPagerDispatchOutcome;

#define SPARK_KV_PAGER_DISPATCH_NO_DEADLINE_HINT 0u

typedef struct SparkKvPagerDispatch
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t logical_block_index;
    uint32_t deadline_step;
    uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES];
}
SparkKvPagerDispatch;

typedef struct SparkKvPagerDispatchDecision
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t outcome;
    uint32_t logical_block_index;
    uint32_t resident;
    uint32_t reserved0;
}
SparkKvPagerDispatchDecision;

SparkStatus SparkKvPagerInitialize(
    SparkKvPager *pager,
    const SparkKvPagerConfiguration *configuration);

uint64_t SparkKvPagerBlockBytes(const SparkKvPager *pager);

SparkStatus SparkKvPagerAdmit(
    SparkKvPager *pager,
    const SparkKvPagerAdmission *admission,
    SparkKvPagerAdmissionDecision *decision_out);

SparkStatus SparkKvPagerCommitAdmission(
    SparkKvPager *pager,
    uint32_t block_count);
SparkStatus SparkKvPagerReleaseAdmission(
    SparkKvPager *pager,
    uint32_t block_count);

SparkStatus SparkKvPagerRestoreBlock(
    SparkKvPager *pager,
    uint32_t logical_block_index,
    const uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES]);

SparkStatus SparkKvPagerRestoreBlockDeadline(
    SparkKvPager *pager,
    uint32_t logical_block_index,
    const uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES],
    uint32_t deadline_step);

SparkStatus SparkKvPagerDispatchBlock(
    SparkKvPager *pager,
    const SparkKvPagerDispatch *dispatch,
    SparkKvPagerDispatchDecision *decision_out);

SparkStatus SparkKvPagerEvictWriteback(
    void *context,
    uint32_t logical_block_index,
    uint32_t resident_slot_index,
    uint64_t generation,
    uintptr_t key_device_address,
    uint64_t key_bytes,
    uintptr_t value_device_address,
    uint64_t value_bytes);

SparkStatus SparkKvPagerAssertDeviceBudget(const SparkKvPager *pager);

void SparkKvPagerGetStatistics(
    const SparkKvPager *pager,
    SparkKvPagerStatistics *statistics_out);

SparkStatus SparkKvPagerPollParkCompletions(SparkKvPager *pager);

SparkStatus SparkKvPagerShutdown(SparkKvPager *pager);

#ifdef __cplusplus
}
#endif
