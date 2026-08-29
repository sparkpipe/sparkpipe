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
/* ABI 2 (C3): the admission struct's former reserved0 slot became
 * restore_slack_microseconds. ABI 1 remains named for the fence: a legacy
 * admission is refused, not silently reinterpreted. */
#define SPARK_KV_PAGER_ADMISSION_ABI_VERSION 2u
#define SPARK_KV_PAGER_ADMISSION_ABI_VERSION_LEGACY 1u

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

/* C3 MEASURED BANDWIDTH. The admission arithmetic of docs/JIT_KV_RESPONSE.md
 * C3 consumes the drive's REAL sustained number, not the nominal estimate:
 * every observed page-in and page-out folds an instantaneous throughput
 * (payload bytes / measured elapsed microseconds, alpha = 1/2 EMA) into
 * SparkKvPagerStatistics.measured_bytes_per_second. The clock is injected -
 * the pager has no time source of its own, and a host test stays
 * deterministic by advancing a fake clock per poll. NULL keeps the EMA at 0
 * and admission falls back to the tier's nominal configured bandwidth. The
 * clock may be read from the park worker thread: pass one that is safe
 * there (a monotonic clock read is). */
typedef uint64_t (*SparkKvPagerClockFunction)(void *clock_context);

/* C4 ASYNC PARK. The park write-out moves off the arena's clock onto a
 * worker thread: eviction stages the block payload into the park entry's
 * own plane and reserves the tier record inline (the slot accounting stays
 * exact at enqueue time), the worker runs only the backing-write leg, and
 * the OWNING thread publishes completions (the tier's readable index, the
 * block's BACKING_VALID, the page-out receipt - or the B1 degrade for an
 * IO-class failure). The worker follows the W2a weightd discipline: an
 * atomic stop flag the owning thread stores, and a poll quantum that bounds
 * how late the flag is seen - the true bound on a park is one backing
 * write, never a wake-up promise. */
#define SPARK_KV_PAGER_PARK_QUEUE_CAPACITY 8u
#define SPARK_KV_PAGER_PARK_POLL_QUANTUM_MICROSECONDS 200u
#define SPARK_KV_PAGER_PARK_WORKER_HANDLE_BYTES 32u

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
    /* C3: the injected clock (see the typedef above). NULL disables
     * measurement; admission then predicts with the tier's nominal
     * configured bandwidth. */
    void *clock_context;
    SparkKvPagerClockFunction clock_function;
    /* C4: the park queue depth. 0 parks inline on the arena's clock (the
     * pre-C4 behavior; nothing else below is required). >0 starts the
     * worker thread; park_staging must then hold park_queue_blocks block
     * payloads - one plane per staged park, so the arena can reuse the
     * evicted device slot before the write lands. The backing_write
     * callback may be called from the worker thread. */
    uint32_t park_queue_blocks;
    uint32_t reserved3;
    void *park_staging;
    uint64_t park_staging_bytes;
    /* C5 PARK POLICY (docs/JIT_KV_RESPONSE.md C5): the victim policy the
     * pager installs on its arena for every admission-time park-out and
     * restore make-room. LRU (0) is the default and the historical behavior
     * byte for byte. REUSE_VALUE ranks victims by reuse value - restored-
     * again history first (a block restored twice is hot), then dirtiness
     * (a clean block is cheap to drop; endurance is a line item), then
     * recency. The knob changes WHICH residents park, never the budget
     * arithmetic: admission counts the same parkable pool, holds the same
     * reservations, and the device-law receipt reads the same under both
     * policies. The pager validates the value (an unknown policy is refused
     * at Initialize). This slot is the configuration's former reserved2. */
    uint32_t park_policy;
}
SparkKvPagerConfiguration;

/* C5: the park policy values (the arena's eviction policy is the one
 * definition; the pager names them for its configuration). */
#define SPARK_KV_PAGER_PARK_POLICY_LRU \
    SPARK_KV_CACHE_EVICTION_POLICY_LRU
#define SPARK_KV_PAGER_PARK_POLICY_REUSE_VALUE \
    SPARK_KV_CACHE_EVICTION_POLICY_REUSE_VALUE

/* C4: one staged park between enqueue (the arena's eviction path) and
 * publish. The payload is already captured into `staging` - the pager owns
 * the bytes the moment the eviction returns, which is what lets the arena
 * reuse the device slot while the write is still in flight. The tier record
 * is RESERVED at enqueue (the slot accounting admission reads stays exact);
 * only the backing write and the publish wait. */
typedef struct SparkKvPagerParkEntry
{
    uint32_t logical_block_index;
    uint32_t reserved0;
    SparkNvmeTierWriteReservation reservation;
    uint8_t *staging;
    uint64_t payload_bytes;
}
SparkKvPagerParkEntry;

/* C4: the completion record. OK publishes (CommitWrite + BACKING_VALID +
 * the page-out receipt); an IO-class status aborts the reservation and
 * DEGRADES the block exactly as the inline write-back's failure path does
 * (B1: drop + recompute, never a wedge). The elapsed sample is the C3
 * page-out measurement (0 when no clock is configured). */
typedef struct SparkKvPagerParkCompletion
{
    uint32_t logical_block_index;
    uint32_t status;               /* SparkStatus of the backing write leg */
    SparkNvmeTierWriteReservation reservation;
    uint64_t write_elapsed_microseconds;
}
SparkKvPagerParkCompletion;

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
    uint64_t admission_queued_bandwidth; /* C3: the restore debt cannot cross
                                            the slack at the measured rate */
    uint64_t park_completions_published; /* C4: write legs resolved at the
                                            owning thread */
    uint64_t park_write_failures;       /* C4: IO-class completions degraded
                                           (the B1 transition, async leg) */
    uint64_t measured_bandwidth_samples; /* C3: samples folded into the EMA */
    uint64_t measured_bytes_per_second;  /* C3: EMA over observed page-in and
                                            page-out throughput; 0 until the
                                            first sample (nominal fallback) */
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
    /* C4: the park ring (the owning thread produces, the worker consumes)
     * and the completion ring (the worker produces, the owning thread
     * publishes). SPSC both ways; the cursors carry the ordering. */
    SparkKvPagerParkEntry park_queue[SPARK_KV_PAGER_PARK_QUEUE_CAPACITY];
    SparkKvPagerParkCompletion
        park_completions[SPARK_KV_PAGER_PARK_QUEUE_CAPACITY];
    atomic_uint park_head;
    atomic_uint park_tail;
    atomic_uint park_completion_head;
    atomic_uint park_completion_tail;
    atomic_uint park_worker_stop;
    /* the entry the worker is currently writing (published before the pop
     * removes it from the ring, cleared after the completion is pushed):
     * a restore offer on a MID-WRITE park must answer BUSY, not fall
     * through to the tier and recompute bytes the pager is holding. */
    atomic_uint park_write_inflight_valid;
    SparkKvPagerParkEntry park_write_inflight;
    uint32_t park_worker_active;   /* the worker thread was started */
    uint32_t park_queue_blocks;    /* configuration copy, ring capacity */
    uint8_t park_worker_handle[SPARK_KV_PAGER_PARK_WORKER_HANDLE_BYTES];
}
SparkKvPager;

/* C1: admission states its EXACT block demand; the decision is ADMITTED
 * (unassigned resident capacity atomically held - commit it block by block
 * as they become resident, or release it on abort) or QUEUED (offer again
 * later; nothing was evicted, nothing wedged). QUEUED is a healthy answer
 * and carries its reason in the pager statistics.
 *
 * ABI 2 (C3, docs/JIT_KV_RESPONSE.md): the former reserved0 slot is now
 * restore_slack_microseconds - the decode slack the AGGREGATE restore debt
 * (parked blocks awaiting their page-in) plus this admission's own
 * park-outs must cross at the MEASURED bandwidth for the request to serve.
 * 0 keeps the capacity-only decision, so pre-C3 callers behave exactly as
 * before. */
typedef struct SparkKvPagerAdmission
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t block_demand;
    uint32_t restore_slack_microseconds; /* C3: 0 = capacity checks only */
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

/* W2 (docs/JIT_KV_RESPONSE.md): the dispatch offer's deadline hint. The gate
 * is the only caller that knows WHICH block the dispatcher is waiting on, so
 * it is the one place a restore-debt ordering can come from: the hint rides
 * the restore into the tier's read path, where the pending queue (the
 * deadline-ordered debt) tightens or enqueues the block at that deadline -
 * earliest-deadline-first, so a saturated tier serves the gated block before
 * older backlog instead of in FIFO issue order. 0 = no hint: the restore
 * behaves exactly as before (this slot is the struct's former reserved0,
 * which every historical caller held at zero). */
#define SPARK_KV_PAGER_DISPATCH_NO_DEADLINE_HINT 0u

typedef struct SparkKvPagerDispatch
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t logical_block_index;
    uint32_t deadline_step;       /* W2: 0 = no hint */
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
 * - the tier or the arena is saturated, or (C4) the block's own park
 * write-out is still in flight; retry, never drop. */
SparkStatus SparkKvPagerRestoreBlock(
    SparkKvPager *pager,
    uint32_t logical_block_index,
    const uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES]);

/* W2: the same restore with the dispatch gate's deadline hint. The hint is
 * opaque to the pager - it forwards `deadline_step` to the tier's read path
 * on every poll, so the pending restore debt (the tier's deadline-ordered
 * queue) orders itself around the block the caller is actually waiting on.
 * 0 is SparkKvPagerRestoreBlock's exact behavior - including the historical
 * spin to the poll limit and its IO_ERROR answer; only the dispatch gate
 * translates that exhaustion into its QUEUED answer. */
SparkStatus SparkKvPagerRestoreBlockDeadline(
    SparkKvPager *pager,
    uint32_t logical_block_index,
    const uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES],
    uint32_t deadline_step);

/* C2: the dispatch gate. Returns OK with the decision filled (READY,
 * QUEUED, or RECOMPUTE - see the outcome above). Only a hard failure
 * (bad arguments, a dead tier, an error the restore surfaced mid-loop)
 * surfaces as an error status: backpressure is an answer, not an
 * exception - INCLUDING the hintless offer's poll-budget exhaustion under
 * a saturated tier, which answers QUEUED exactly like the hinted path's
 * ordered busy instead of the historical hard IO_ERROR. */
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

/* C4: publish every finished park write leg on the CALLING thread - the
 * tier's readable index (CommitWrite), the block's BACKING_VALID flag, the
 * page-out receipt, or the B1 degrade for an IO-class failure. The owning
 * thread only (these touch the arena and the tier). The admit and restore
 * paths call this themselves; an embedder driving the pager by hand polls
 * between parks. Returns OK. */
SparkStatus SparkKvPagerPollParkCompletions(SparkKvPager *pager);

/* C4 TERM safety: stop the park worker (the atomic stop flag is seen within
 * one poll quantum plus one backing write - the write is the bound), then
 * finish the parks it left staged INLINE on the calling thread and publish
 * every completion. Every reservation resolves commit-or-abort and every
 * staged payload is accounted, so TERM mid-park leaves a consistent arena
 * and a consistent tier. Idempotent; parks requested after shutdown
 * complete inline on the arena's clock (worker off). */
SparkStatus SparkKvPagerShutdown(SparkKvPager *pager);

#ifdef __cplusplus
}
#endif
