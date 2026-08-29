#pragma once

// The JIT-KV pager: the adapter layer of docs/JIT_KV_DESIGN.md - park,
// restore, and admission backpressure over a resident arena and a backing
// tier.
//
// LAYERING. The arena (spark_kv_cache.h) owns device residency and picks the
// LRU victim. The tier (spark_nvme_tier.h) owns the backing records and
// verifies every landing against the record's SHA-256. THIS component owns
// the policy that joins them: eviction IS a page-out (the arena's
// evict_function is the pager's write-back), restore is a digest-verified
// read landing re-attached as residency, and admission refuses - queues -
// when the device budget and the park horizon are both exhausted, instead of
// evicting protected residents or wedging (docs/JIT_KV_RESPONSE.md C1).
//
// THE MODULE SEAM. The design doc's step 2 names two frame-context ops,
// KV_BLOCKS_SAVE_OUT (0x00001000u) and KV_BLOCKS_RESTORE_IN (0x00002000u):
// one stream-ordered device<->host copy per block through the model module.
// The pager does not know CUDA; it calls the two function pointers below with
// a SparkKvPagerBlockView - the model-neutral form of the design's
// blocks-view struct. The host proof of this contract implements them as
// plain copies; the family wiring implements them with the frame
// ops. The backing write leg is likewise a callback: the tier hands a
// reserved device offset, the pager moves the staged payload there (pwrite in
// production).
//
// DIGEST DISCIPLINE (B3). The pager hashes each block payload as it stages
// it, presents the digest at ReserveWrite, and re-verifies the staging buffer
// against the caller's digest before a restore touches device memory. The
// 64-bit tier key is folded from the digest, so the bucket key is a pure
// function of the payload: identical bytes deduplicate onto one record (the
// design's shared-prefix win), and a 64-bit collision is impossible to alias
// because the digest decides.

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
#define SPARK_KV_PAGER_ADMISSION_ABI_VERSION 1u

/* Bounded recency window of the logical blocks last paged out, newest last:
 * the LRU-order receipt. Total page-outs are page_out_count. */
#define SPARK_KV_PAGER_PAGE_OUT_HISTORY_CAPACITY 16u

/* The pager's staging is a double buffer of block payloads: plane 0 is the
 * module save scratch (and the backing write's source), plane 1 is the
 * restore landing copy. The landing copy is what makes restore
 * deadlock-free: the tier's staging is released BEFORE make-room, so the
 * write-backs a restore triggers can never collide with a demand-held tier
 * buffer protecting the very record being read. */
#define SPARK_KV_PAGER_STAGING_BLOCK_COUNT 2u

/* The device law (operator directive): no pager may configure a device
 * budget above the 110 GiB a node's accelerator makes available. The
 * per-pager budget is usually far smaller; the law is the ceiling. */
#define SPARK_KV_PAGER_DEVICE_LAW_BYTES \
    (110ULL * 1024ULL * 1024ULL * 1024ULL)

/* Restore demand-read polling bound: how many Pump rounds a restore waits on
 * an in-flight read before answering IO_ERROR. The synchronous host path
 * lands in a handful of polls; the async production pager replaces the loop,
 * not the bound's purpose - fail loud, never spin. */
#define SPARK_KV_PAGER_RESTORE_POLL_LIMIT 64u

typedef struct SparkKvPagerBlockView
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t lane_index;       /* ordinal of the owning lane */
    uint32_t first_block;      /* block ordinal within the lane's table */
    uint32_t block_count;      /* 1: the arena evicts/restores per block */
    uint32_t reserved0;
    uint64_t key_bytes;
    uint64_t value_bytes;
    uintptr_t key_device_address;
    uintptr_t value_device_address;
    void *host_staging;        /* key_bytes + value_bytes, pager-owned */
}
SparkKvPagerBlockView;

/* device planes -> host staging (KV_BLOCKS_SAVE_OUT seam) */
typedef SparkStatus (*SparkKvPagerModuleSaveFunction)(
    void *module_context,
    const SparkKvPagerBlockView *view);

/* host staging -> device planes (KV_BLOCKS_RESTORE_IN seam) */
typedef SparkStatus (*SparkKvPagerModuleRestoreFunction)(
    void *module_context,
    const SparkKvPagerBlockView *view);

/* staged payload -> the tier's reserved device offset */
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
    SparkKvCacheArena *arena;      /* the resident tier: the device budget's holder */
    SparkNvmeTier *tier;           /* the backing tier: the park horizon */
    uint32_t park_budget_blocks;   /* admission's park allowance (records in the tier) */
    uint32_t reserved2;
    uint64_t device_budget_bytes;  /* <= SPARK_KV_PAGER_DEVICE_LAW_BYTES; must hold
                                      the arena's whole resident capacity */
    void *staging;                 /* SPARK_KV_PAGER_STAGING_BLOCK_COUNT block
                                      payloads: save scratch, restore landing */
    uint64_t staging_bytes;
    void *module_context;
    SparkKvPagerModuleSaveFunction module_save;
    SparkKvPagerModuleRestoreFunction module_restore;
    void *backing_context;
    SparkKvPagerBackingWriteFunction backing_write;
}
SparkKvPagerConfiguration;

typedef struct SparkKvPagerStatistics
{
    uint64_t page_out_count;
    uint64_t page_out_bytes;
    uint64_t page_out_deduplicated;     /* tier already held identical bytes */
    uint64_t page_in_count;
    uint64_t page_in_bytes;
    uint64_t page_in_misses;            /* tier answered MISS: recompute path */
    uint64_t admission_requests;
    uint64_t admission_accepted;
    uint64_t admission_queued_device;   /* no evictable resident block */
    uint64_t admission_queued_backing;  /* park horizon reached: queue, never thrash */
    uint64_t dispatch_requests;         /* C2: dispatch offers through the gate */
    uint64_t dispatch_ready;            /* answered READY: restore complete */
    uint64_t dispatch_queued;           /* restore incomplete: the offer repeats */
    uint64_t dispatch_recompute;        /* no backing bytes: recompute path */
    uint32_t park_evictions;            /* blocks parked by the last admission */
    uint32_t page_out_history_count;
    uint32_t page_out_history[SPARK_KV_PAGER_PAGE_OUT_HISTORY_CAPACITY];
}
SparkKvPagerStatistics;

typedef struct SparkKvPager
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t block_bytes;          /* arena key + value strides == tier block bytes */
    uint32_t park_budget_blocks;
    SparkKvPagerConfiguration configuration;
    void *landing_staging;         /* staging plane 1: the restore landing copy */
    SparkKvPagerStatistics statistics;
}
SparkKvPager;

/* C1: admission states its EXACT block demand; the decision is ADMITTED
 * (unassigned resident capacity atomically held - commit it block by block
 * as they become resident, or release it on abort) or QUEUED (offer again
 * later; nothing was evicted, nothing wedged). QUEUED is a healthy answer
 * and carries its reason in the pager statistics. */
typedef struct SparkKvPagerAdmission
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t block_demand;
    uint32_t reserved0;
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
    uint32_t outcome;              /* SparkKvPagerAdmissionOutcome */
    uint32_t block_demand;
    uint32_t park_evictions;       /* ADMITTED: blocks the pager parked to fit it */
    uint32_t reservation_held;     /* ADMITTED: unassigned resident blocks now owned */
}
SparkKvPagerAdmissionDecision;

/* C2 (docs/JIT_KV_RESPONSE.md): DISPATCH GATES ON RESTORE COMPLETE - not a
 * hint. The dispatcher presents a block it is about to run against; the
 * answer is READY only once the block is RESIDENT with its verified bytes
 * in the device planes. A parked block restores INSIDE the gate (the same
 * single restore path page-in uses); while the tier is saturated the answer
 * is QUEUED - the dispatch waits and the offer is repeated, nothing wedged
 * and nothing dropped, which is C1's queue-not-wedge discipline carried to
 * the dispatch path. A block with no backing bytes answers RECOMPUTE: the
 * caller rebuilds it, dispatch never runs on partial state. */
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

typedef struct SparkKvPagerDispatch
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t logical_block_index;
    uint32_t reserved0;
    uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES];
}
SparkKvPagerDispatch;

typedef struct SparkKvPagerDispatchDecision
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t outcome;             /* SparkKvPagerDispatchOutcome */
    uint32_t logical_block_index;
    uint32_t resident;            /* READY: the RESIDENT flag was verified */
    uint32_t reserved0;
}
SparkKvPagerDispatchDecision;

/* Installs the pager as the arena's evict function (refusing an arena that
 * already has another owner) and fences the configuration: the arena's whole
 * resident capacity must fit the device budget, and the budget must not
 * exceed the device law. */
SparkStatus SparkKvPagerInitialize(
    SparkKvPager *pager,
    const SparkKvPagerConfiguration *configuration);

uint64_t SparkKvPagerBlockBytes(const SparkKvPager *pager);

SparkStatus SparkKvPagerAdmit(
    SparkKvPager *pager,
    const SparkKvPagerAdmission *admission,
    SparkKvPagerAdmissionDecision *decision_out);

/* The two halves of the admission contract: commit ownership block by block
 * immediately before each block becomes resident, or release what remains on
 * abort. Neither grants residency; both only move the unassigned counter. */
SparkStatus SparkKvPagerCommitAdmission(
    SparkKvPager *pager,
    uint32_t block_count);
SparkStatus SparkKvPagerReleaseAdmission(
    SparkKvPager *pager,
    uint32_t block_count);

/* Page-IN for a parked block: the digest is the identity the caller expects.
 * The tier lands the bytes digest-verified, the pager re-verifies the staging
 * buffer, the arena re-attaches residency (parking the LRU victim if the
 * device budget is tight), and the module restore op copies into the device
 * planes. MISS answers NOT_FOUND - the caller recomputes. BUSY answers BUSY
 * - the tier or the arena is saturated; retry, never drop. */
SparkStatus SparkKvPagerRestoreBlock(
    SparkKvPager *pager,
    uint32_t logical_block_index,
    const uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES]);

/* C2: the dispatch gate. Returns OK with the decision filled (READY,
 * QUEUED, or RECOMPUTE - see the outcome above). Only a hard failure
 * (bad arguments, a dead tier) surfaces as an error status: backpressure
 * is an answer, not an exception. */
SparkStatus SparkKvPagerDispatchBlock(
    SparkKvPager *pager,
    const SparkKvPagerDispatch *dispatch,
    SparkKvPagerDispatchDecision *decision_out);

/* The arena's evict function under the pager: stage the victim's planes,
 * hash them, reserve a tier record under the digest, write the backing bytes
 * if the payload is new, and commit. IO-class failures propagate for the
 * arena's B1 degrade path to drop the block; BUSY propagates as
 * backpressure; a digest collision stays loud. */
SparkStatus SparkKvPagerEvictWriteback(
    void *context,
    uint32_t logical_block_index,
    uint32_t resident_slot_index,
    uint64_t generation,
    uintptr_t key_device_address,
    uint64_t key_bytes,
    uintptr_t value_device_address,
    uint64_t value_bytes);

/* The budget receipt: OK only while resident + reserved + unassigned blocks
 * fit the arena's capacity AND the resident bytes fit the configured device
 * budget. Called after every state change by anything proving the law. */
SparkStatus SparkKvPagerAssertDeviceBudget(const SparkKvPager *pager);

void SparkKvPagerGetStatistics(
    const SparkKvPager *pager,
    SparkKvPagerStatistics *statistics_out);

#ifdef __cplusplus
}
#endif
