#pragma once

// Tier 3 of the KV cache: the NVMe tier, and it is a JIT tier.
//
// THE STACK. Tier 1 is device memory, tier 2 is host memory, and both are
// small against the models this engine serves. Tier 3 is the NVMe drive, and
// it is the tier that makes the cache's practical size a terabyte rather than
// tens of gigabytes. cache/cache.h already knows a block can be REFERENCED
// but not RESIDENT; this tier is where a referenced-not-resident block's bytes
// actually live, and the one question it exists to answer is:
//
//   can the bytes be upstairs BEFORE the layer that needs them?
//
// JIT, NOT A CACHE OF LAST RESORT. A demand fetch from NVMe on the decode
// critical path costs a millisecond against a step budget measured in tens of
// microseconds, so a design that fetches on miss has lost before it starts.
// This tier instead reads the schedule: the admission queue already knows
// which sequences run next and which KV blocks they will touch, and the
// lookahead planner turns that into reads issued EARLY ENOUGH that transfer
// time hides inside compute time. The trade the design makes explicitly:
// MORE CHURN for LESS LATENCY. Eviction is aggressive and costs nothing -
// the bytes are re-fetchable from the source tier or recomputable by prefill,
// so eviction is a generation bump, never a write-back, and it never runs on
// the decode path at all. ReserveWrite is the only write-back admission path
// that may evict, and it is never called mid-step.
//
// WHAT A MISS MEANS HERE. RequestDemand answering MISS means the block is not
// on the drive at all: the upper tier must recompute it. That is the only
// event allowed to stall a decode step, it is counted (demand_misses), and
// everything else - prefetch issue, completion, eviction, preemption - happens
// off the critical path in Pump, driven between steps.
//
// THE DEVICE IS A VTABLE. Real hardware is O_DIRECT reads behind aio or
// io_uring; the tests are a mock that completes a read after a programmed
// number of polls. The tier cannot tell the difference, which is the point:
// the schedule logic is host-verifiable, and the driver binding is the forty
// lines that change at bring-up.

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVME_TIER_ABI_VERSION 2u
#define SPARK_NVME_TIER_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkNvmeTierConfiguration))
#define SPARK_NVME_TIER_DEFAULT_BUDGET_BYTES (1099511627776ULL)  /* 1 TB */
// Sizing defaults from the JIT-KV bandwidth analysis (docs/archive/NVME_KV_SIZING.md,
// tools/nvme_kv_estimate.py) - ESTIMATES, measured nowhere: the roadmap pins
// NVMe bandwidth as PENDING (docs/archive/PERF_ROADMAP_2026-08-01.md:496-498). The analysis
// verdict: at 2K chat no model touches the drive through B1024; 128K agent
// fits under admission control; 1M survives only with token selection.
//
// Bandwidth: the LOW end of the assumed 5-7 GB/s internal range. The planner
// turns bandwidth into transfer-steps ETAs, so erring low errs toward saying
// NOT-confident - the direction admission control survives. On the external
// drive (assumed 2.5-3.5 GB/s, "external is 2x slower") halve this.
#define SPARK_NVME_TIER_DEFAULT_DEVICE_BYTES_PER_SECOND (5000000000ULL)
// Step time: the flagship latent-cache model's B8 2K chat step at the 80%
// roofline target, 192.3 GB / 2,840 GB/s = 67.7 ms (the estimator's step
// table; the roadmap's matching per-batch row agrees). bytes_per_step =
// bandwidth x step time, so an overstated step time understates
// transfer_steps and makes every ETA optimistic - this number must not be
// rounded up, and a deployment serving a longer step (B64 agent is 235.6 ms)
// should pass its own.
#define SPARK_NVME_TIER_DEFAULT_STEP_TIME_MICROSECONDS 67700u
#define SPARK_NVME_TIER_NO_SLOT 0xffffffffu
#define SPARK_NVME_TIER_MAX_STAGING_BUFFERS 16u
#define SPARK_NVME_TIER_DEFAULT_PENDING_CAPACITY 256u
// O_DIRECT granularity. A block payload that is not a whole number of these
// forces the driver into buffered I/O and the copy that comes with it, so the
// configuration is rejected instead.
#define SPARK_NVME_TIER_IO_ALIGNMENT_BYTES 4096u

// What the scheduler knows about a near-future need. need_by_step is the
// decode step at which the first layer reads this block; the planner works
// backwards from it.
typedef struct SparkNvmeTierNeed
{
	uint64_t content_hash;         /* the same chained hash cache/cache.h uses */
	uint32_t need_by_step;
	uint32_t reserved0;
}
SparkNvmeTierNeed;

// Fixed at init, like a partition table. Derived quantities the planner uses
// every call - transfer steps per block, bytes per step - are computed once in
// Initialize rather than re-divided on the hot path.
typedef struct SparkNvmeTierConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint64_t budget_bytes;         /* the NVMe region this tier may use */
	uint64_t base_offset;          /* where that region starts on the device */
	uint32_t block_bytes;          /* one KV block payload; a 4096 multiple */
	uint32_t hash_bucket_count;
	uint32_t staging_buffer_count; /* >= 2: DMA is double-buffered */
	uint32_t demand_reserve_buffers; /* staging prefetch may never take */
	uint32_t pending_capacity;     /* depth of the deadline-ordered queue */
	uint64_t device_bytes_per_second; /* sustained read bandwidth, for ETA */
	uint32_t step_time_microseconds;  /* one decode step, converts bytes to steps */
	uint32_t reserved0;
}
SparkNvmeTierConfiguration;

// The async block device. Contract:
//
//   submit_read  queues a read of bytes from device_offset into destination
//                and returns a ticket. Destination is a staging buffer; the
//                device owns it until poll_read says OK or cancel_read
//                returns OK.
//   poll_read    SPARK_STATUS_OK when the bytes have landed, BUSY in flight.
//   cancel_read  after OK the device will never touch the buffer again. BUSY
//                or PENDING means cancellation has been requested but the
//                destination remains device-owned; the tier keeps polling and
//                does not reuse the staging slot until poll_read reaches a
//                terminal result. May be NULL: cancellation is an optimisation
//                (demand preemption of an in-flight prefetch), never a
//                correctness requirement.
//
// Nothing here blocks. A synchronous read hiding behind submit would put NVMe
// latency on the decode path in a way no counter can see, so the interface
// makes it unrepresentable.
typedef SparkStatus (*SparkNvmeTierSubmitRead)(
	void *context, uint64_t device_offset, void *destination,
	uint32_t bytes, uint64_t *ticket_out);
typedef SparkStatus (*SparkNvmeTierPollRead)(void *context, uint64_t ticket);
typedef SparkStatus (*SparkNvmeTierCancelRead)(void *context, uint64_t ticket);

typedef struct SparkNvmeTierDevice
{
	void *context;
	SparkNvmeTierSubmitRead submit_read;
	SparkNvmeTierPollRead poll_read;
	SparkNvmeTierCancelRead cancel_read;
}
SparkNvmeTierDevice;

// What RequestDemand learned. READY is the only answer that hands back data;
// everything else is a state, and IN_FLIGHT/STARTED are promises the caller
// turns into a wait that is NOT on the decode path (the sequence simply does
// not issue this step).
typedef enum SparkNvmeTierDemandState
{
	SPARK_NVME_TIER_DEMAND_READY = 0,   /* staging_pointer is valid now */
	SPARK_NVME_TIER_DEMAND_IN_FLIGHT,   /* a read was already moving: joined it */
	SPARK_NVME_TIER_DEMAND_STARTED,     /* this call issued the demand read */
	SPARK_NVME_TIER_DEMAND_MISS         /* not on the drive: recompute path */
}
SparkNvmeTierDemandState;


typedef struct SparkNvmeTierWriteReservation
{
	uint64_t content_hash;
	uint64_t device_offset;
	uint32_t slot_index;
	uint32_t generation;
	uint32_t already_present;
	uint32_t reserved0;
}
SparkNvmeTierWriteReservation;

typedef struct SparkNvmeTierDemandResult
{
	SparkNvmeTierDemandState state;
	uint32_t reserved0;
	void *staging_pointer;         /* valid only while state is READY, until Consume */
}
SparkNvmeTierDemandResult;

// The admission question, answered per call: "will these blocks be readable by
// step N?" Confidence is a flag, not a promise - ALL means every block is
// either upstairs already or has an ETA inside the deadline given the
// configured bandwidth, and admission should prefer such sequences because
// their first step cannot stall on this tier.
typedef enum SparkNvmeTierConfidence
{
	SPARK_NVME_TIER_CONFIDENCE_ALL = 0,
	SPARK_NVME_TIER_CONFIDENCE_PARTIAL,
	SPARK_NVME_TIER_CONFIDENCE_NONE
}
SparkNvmeTierConfidence;

typedef struct SparkNvmeTierResidencyAssessment
{
	SparkNvmeTierConfidence confidence;
	uint32_t ready_count;          /* upstairs already */
	uint32_t inflight_confident_count;  /* moving, ETA inside the deadline */
	uint32_t planned_confident_count;   /* on the drive, ETA inside the deadline */
	uint32_t late_count;           /* present but cannot arrive in time */
	uint32_t absent_count;         /* not on the drive at all */
	uint32_t reserved0;
}
SparkNvmeTierResidencyAssessment;

// The overlap contract, in numbers. demand_misses is the one that may rise
// without the design being broken (a block genuinely never written back).
// demand_stalls must stay zero: it counts a demand load that found every
// staging buffer held by unconsumed demand data, which is a sizing bug, not a
// workload property. prefetch_late_landings measures how often the lookahead
// was too short - the signal to widen the window, not to slow the model.
typedef struct SparkNvmeTierStatistics
{
	uint64_t write_reservations;
	uint64_t publishes;
	uint64_t write_aborts;
	uint64_t cancel_pending_count;
	uint64_t evictions;
	uint64_t pinned_eviction_skips;
	uint64_t demand_hits;          /* READY on arrival: decode never waited */
	uint64_t demand_joins;         /* a read was already in flight */
	uint64_t demand_loads;         /* this request issued the read */
	uint64_t demand_misses;        /* absent: the only legitimate decode stall */
	uint64_t demand_stalls;        /* present but no staging obtainable: sizing bug */
	uint64_t prefetch_issues;
	uint64_t prefetch_landings;
	uint64_t prefetch_late_landings;
	uint64_t prefetch_preemptions; /* staging or queue slot yielded to demand */
	uint64_t prefetch_dropped;     /* landed prefetch discarded before consumption */
	uint64_t stale_completions;    /* landed into a recycled slot: discarded */
	uint64_t io_errors;            /* the device reported a failed read */
	uint64_t read_bytes;           /* every byte submitted to the drive: the
	                                  kv_nvme_read_bytes the acceptance run must
	                                  report per generated token */
	uint64_t slot_count;           /* budget-derived capacity, for the report */
	uint64_t slots_in_use;
}
SparkNvmeTierStatistics;

typedef struct SparkNvmeTier SparkNvmeTier;
struct SparkNvmeTier
{
	SparkNvmeTierConfiguration configuration;
	SparkNvmeTierDevice device;
	void *slots;                   /* slot table, one per budget-derived block */
	uint32_t *buckets;
	void *pending;                 /* deadline-ordered prefetch queue */
	uint8_t *staging;              /* caller-pinned DMA staging, N * block_bytes */
	void *staging_state;           /* per-buffer descriptors */
	uint32_t slot_count;
	uint32_t free_head;
	uint32_t clock_hand;
	uint32_t slots_in_use;
	uint64_t tick;
	uint32_t bytes_per_step;       /* bandwidth * step time, precomputed */
	uint32_t transfer_steps;       /* ceil(block_bytes / bytes_per_step), >= 1 */
	SparkNvmeTierStatistics statistics;
};

// Bookkeeping bytes the caller must provide as `tables` to Initialize: slots,
// hash buckets, the pending queue and the staging descriptors, one blob, no
// malloc after init. Initialize receives the exact table and staging byte
// capacities and rejects undersized or misaligned regions. Staging is separate
// because in production it is pinned DMA memory from a different allocator
// than ordinary tables. The tier is single-owner-thread; callers must serialize
// every mutable operation if they introduce another thread.
uint64_t SparkNvmeTierTableBytes(
	const SparkNvmeTierConfiguration *configuration);

SparkStatus SparkNvmeTierInitialize(
	SparkNvmeTier *tier,
	const SparkNvmeTierConfiguration *configuration,
	const SparkNvmeTierDevice *device,
	void *tables,
	uint64_t tables_bytes,
	void *staging,
	uint64_t staging_bytes);

// Reserve a device slot for a write-back. A reservation is not visible to
// readers and is never returned by OffsetOf or RequestDemand. The caller owns
// the reservation until CommitWrite or AbortWrite. If the hash is already
// committed, already_present is set and no new slot is allocated.
SparkStatus SparkNvmeTierReserveWrite(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	SparkNvmeTierWriteReservation *reservation_out);

// Publish the reserved record only after the asynchronous or synchronous
// device write has completed successfully. This is the only transition that
// inserts a new hash into the readable index.
SparkStatus SparkNvmeTierCommitWrite(
	SparkNvmeTier *tier,
	const SparkNvmeTierWriteReservation *reservation);

// Release a reservation whose device write failed or was cancelled. No reader
// can have observed it because reserved records are not indexed.
SparkStatus SparkNvmeTierAbortWrite(
	SparkNvmeTier *tier,
	const SparkNvmeTierWriteReservation *reservation);

// Where a committed block's bytes are, for diagnostics and read planning.
SparkStatus SparkNvmeTierOffsetOf(
	const SparkNvmeTier *tier,
	uint64_t content_hash,
	uint64_t *device_offset_out);

// The lookahead planner. For each need: already upstairs or moving - tighten
// its deadline; on the drive and idle - enqueue a prefetch ordered by
// earliest deadline; absent - report it, because a prefetch that cannot happen
// is exactly what admission needs to know about. Issues nothing itself: Pump
// turns the queue into reads, so planning a far-future need costs no I/O.
typedef struct SparkNvmeTierPlanReport
{
	uint32_t already_ready_count;
	uint32_t already_inflight_count;
	uint32_t queued_count;
	uint32_t absent_count;
	uint32_t queue_full_count;
	uint32_t late_risk_count;      /* transfer cannot finish by need_by: issued anyway */
}
SparkNvmeTierPlanReport;

SparkStatus SparkNvmeTierPlanLookahead(
	SparkNvmeTier *tier,
	const SparkNvmeTierNeed *needs,
	uint32_t need_count,
	uint32_t step_now,
	SparkNvmeTierPlanReport *report_out);  /* may be NULL */

// The decode-path question. Never blocks, never allocates a slot, never
// evicts. A miss is counted and handed back as a state; the caller's recompute
// fallback is its own business.
SparkStatus SparkNvmeTierRequestDemand(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	uint32_t step_now,
	SparkNvmeTierDemandResult *result_out);

// The between-steps driver: poll in-flight reads, promote landed buffers to
// READY (checking the slot generation, because an evicted slot's late read is
// waste, not data), then issue queued prefetches into whatever staging the
// demand reserve leaves free, earliest deadline first.
SparkStatus SparkNvmeTierPump(SparkNvmeTier *tier, uint32_t step_now);

// The upper tier has copied the READY staging buffer out; release it. The
// on-drive copy stays: eviction is what reclaims records, not consumption.
SparkStatus SparkNvmeTierConsume(SparkNvmeTier *tier, uint64_t content_hash);

// A pinned block is invisible to the eviction clock and to demand preemption
// of its staging. This is what an admission lookahead buys: the scheduler
// admits, pins, and the fetch has time to land. Unpinning is the caller's job
// and forgetting it pins a record forever, which is why eviction skips are
// counted.
SparkStatus SparkNvmeTierPin(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	int32_t pin);

// The admission interface: "will these blocks be resident by step_deadline?"
SparkStatus SparkNvmeTierWillBeResidentBy(
	const SparkNvmeTier *tier,
	const uint64_t *content_hashes,
	uint32_t hash_count,
	uint32_t step_now,
	uint32_t step_deadline,
	SparkNvmeTierResidencyAssessment *assessment_out);

void SparkNvmeTierGetStatistics(
	const SparkNvmeTier *tier,
	SparkNvmeTierStatistics *statistics_out);

#ifdef __cplusplus
}
#endif
