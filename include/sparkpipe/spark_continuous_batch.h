#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The step-boundary admission contract for continuous batching on the batch
 * engine (docs/JIT_KV_RESPONSE.md C1, the serving-completion unit). The
 * engine today runs BATCH mode: a file of requests is submitted whole,
 * admission closes, the batch runs to completion. This module is the
 * engine-neutral control plane that turns that file into a CONTINUOUS
 * arrival: offers join at step boundaries, refused offers queue with a
 * NAMED refusal (the cache/kv_pager.c discipline - backpressure is a
 * queue, never a wedge, never a silent drop), and a finished lane's slot
 * is reclaimed at the very next boundary.
 *
 * The three laws (each carries a deterministic host proof in
 * tests/test_continuous_batch.c):
 *
 * L1 INTERLEAVING - a join consumes only FREE capacity and never mutates a
 *    resident lane's rows, slot, or phase. Resident lanes therefore emit
 *    the identical event sequence (bit-exact) whether or not a join
 *    happened: the C1 arithmetic is checked against the deployment's
 *    max_input_rows BEFORE the join is granted, so a join can never
 *    deform a resident round.
 * L2 QUEUE-NOT-WEDGE - a refused offer is named (rows / lanes / oversize /
 *    queue full) and kept; the controller keeps making progress every
 *    step; the offer admits at the first boundary whose post-reclaim
 *    arithmetic fits it.
 * L3 SLOT RECLAIM - a lane that finished at step s (emitted its last
 *    token, or was retired engine-side) releases its slot at boundary
 *    s+1, and a queued offer admitted at that boundary takes the freed
 *    capacity - never the same boundary, never a later one.
 *
 * The scheduler policy at boundaries is smallest-first (fewest prompt
 * rows; ties in arrival order) with a starvation bound: an offer that has
 * waited `starvation_bound` boundaries becomes the boundary's priority
 * pick, and while the aged offer waits, no YOUNGER offer may take the
 * capacity it needs (the reservation leg - otherwise smallest-first
 * starves the large offer forever under a sustained small-offer flow).
 * The bound, stated honestly: an aged offer admits at the FIRST boundary
 * whose resident state alone can hold it - younger competitors cannot
 * push that boundary out. No scheduler can admit into a resident set that
 * is genuinely full for N rounds (that case is L2's queue, not
 * starvation); what the policy guarantees is that QUEUED work is not
 * starved by QUEUED competitors.
 *
 * No model token ids appear anywhere in this module: lanes are sized by
 * row and token COUNTS only. */

#define SPARK_CONTINUOUS_BATCH_ABI_VERSION 1u

#define SPARK_CONTINUOUS_BATCH_ADMITTED 1u
#define SPARK_CONTINUOUS_BATCH_QUEUED 2u

/* The named refusals. Every non-admission names exactly one cause and the
 * offer queues (L2) - the naming mirrors the pager's admission_queued_
 * counters so operators see the same refusal shape at the batch layer. */
#define SPARK_CONTINUOUS_BATCH_QUEUE_NONE 0u
#define SPARK_CONTINUOUS_BATCH_QUEUE_ROWS 1u   /* resident rows + offer rows
	                                          would cross max_input_rows */
#define SPARK_CONTINUOUS_BATCH_QUEUE_LANES 2u  /* every lane slot is resident */
#define SPARK_CONTINUOUS_BATCH_QUEUE_OVERSIZE 3u /* the offer alone can never
                                                    fit the deployment */
#define SPARK_CONTINUOUS_BATCH_QUEUE_FULL 4u   /* queue at capacity: NOT kept,
                                                  retry after draining (the
                                                  pager BUSY analog) */
#define SPARK_CONTINUOUS_BATCH_QUEUE_AHEAD 5u  /* eligible work is already in
                                                  the queue: the offer joins
                                                  the line (arrivals never
                                                  cut ahead of queued work) */

#define SPARK_CONTINUOUS_BATCH_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkContinuousBatchConfiguration))
#define SPARK_CONTINUOUS_BATCH_REQUEST_BYTES \
	((uint32_t)sizeof(SparkContinuousBatchRequest))
#define SPARK_CONTINUOUS_BATCH_DECISION_BYTES \
	((uint32_t)sizeof(SparkContinuousBatchDecision))
#define SPARK_CONTINUOUS_BATCH_STEP_REPORT_BYTES \
	((uint32_t)sizeof(SparkContinuousBatchStepReport))

#define SPARK_CONTINUOUS_BATCH_LANE_FREE 0u
#define SPARK_CONTINUOUS_BATCH_LANE_PREFILL 1u
#define SPARK_CONTINUOUS_BATCH_LANE_DECODE 2u
#define SPARK_CONTINUOUS_BATCH_LANE_FINISHED 3u

typedef struct SparkContinuousBatchConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t max_input_rows;    /* the deployment's max_input_row_count */
	uint32_t max_active_lanes;  /* the deployment's max_active_sequence_count */
	uint32_t starvation_bound;  /* boundaries before the aging escape; 0
	                               disables it (pure smallest-first) */
	uint32_t queue_capacity;    /* refused offers kept for later boundaries */
	uint32_t lane_capacity;     /* lane table size (<= max_active_lanes) */
	uint32_t reserved0;
}
SparkContinuousBatchConfiguration;

typedef struct SparkContinuousBatchRequest
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint64_t request_id;
	uint32_t prompt_row_count;   /* the lane's exact prefill demand (C1) */
	uint32_t output_token_budget; /* the lane's exact decode demand (C1) */
}
SparkContinuousBatchRequest;

typedef struct SparkContinuousBatchDecision
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t outcome;        /* ADMITTED or QUEUED */
	uint32_t queue_reason;   /* the named refusal when QUEUED */
	uint32_t queue_position; /* arrival-order queue index when kept */
	uint32_t reserved0;
}
SparkContinuousBatchDecision;

/* One resident lane's emission this step. token_index is the 1-based
 * emission count for the lane - a COUNT, never a model token id. */
typedef struct SparkContinuousBatchStepEvent
{
	uint64_t request_id;
	uint32_t slot;
	uint32_t token_index;
}
SparkContinuousBatchStepEvent;

typedef struct SparkContinuousBatchStepReport
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t event_count;  /* events written into the caller's array */
	uint32_t rows_spent;   /* this step's total row demand; the law is
	                          rows_spent <= max_input_rows at EVERY step */
	uint32_t finished_count; /* lanes that hit terminal this step; their
	                            slots free at the NEXT boundary (L3) */
}
SparkContinuousBatchStepReport;

typedef struct SparkContinuousBatchLaneView
{
	uint64_t request_id;
	uint32_t slot;
	uint32_t phase;              /* a SPARK_CONTINUOUS_BATCH_LANE_* value */
	uint32_t remaining_rows;     /* prompt rows not yet consumed */
	uint32_t remaining_budget;   /* decode emissions left */
	uint32_t generated_count;    /* emissions so far */
	uint64_t enqueue_boundary;   /* the boundary the offer arrived at */
	uint8_t retired;             /* engine reported terminal (seam path) */
	uint8_t admitted_via_aging;  /* admitted by the starvation escape */
	uint16_t reserved0;
}
SparkContinuousBatchLaneView;

typedef struct SparkContinuousBatchStatistics
{
	uint64_t admission_requests;      /* Offer calls */
	uint64_t admission_accepted;      /* joined a boundary */
	uint64_t admission_queued_rows;   /* refused: the row arithmetic */
	uint64_t admission_queued_lanes;  /* refused: no lane slot free */
	uint64_t admission_oversize;      /* refused: can never fit; kept for
	                                     the caller to withdraw */
	uint64_t admission_queue_full;    /* refused: queue at capacity; NOT
	                                     kept - retry after draining */
	uint64_t boundary_count;          /* Boundary calls */
	uint64_t slot_reclaims;           /* L3 reclaims */
	uint64_t starvation_jumps;        /* admissions the aging escape won */
	uint32_t resident_lane_count;
	uint32_t queued_offer_count;
}
SparkContinuousBatchStatistics;

typedef struct SparkContinuousBatch SparkContinuousBatch;

/* The boundary scheduler policy, pure and separately provable: given the
 * queue in arrival order, return the index of the offer the boundary
 * attempts next, or -1 when no eligible offer remains. `excluded` (may be
 * 0) marks entries the boundary already attempted. Aged offers (waited >=
 * starvation_bound, bound > 0) go oldest-first ahead of everyone; the
 * rest go smallest prompt_row_count first, ties in arrival order. */
int32_t SparkContinuousBatchPolicyPick(
	const uint32_t *prompt_rows,
	const uint64_t *enqueue_boundary,
	const uint8_t *excluded,
	uint32_t count,
	uint64_t current_boundary,
	uint32_t starvation_bound);

SparkStatus SparkContinuousBatchInitialize(
	const SparkContinuousBatchConfiguration *configuration,
	SparkContinuousBatch **controller_out);
SparkStatus SparkContinuousBatchDestroy(SparkContinuousBatch *controller);

/* The between-steps API: an offer arrives and is answered immediately
 * against the CURRENT boundary state - admitted, or refused with a named
 * reason and kept in the queue (L2). The decision never re-examines
 * resident lanes: L1 holds by construction. */
SparkStatus SparkContinuousBatchOffer(
	SparkContinuousBatch *controller,
	const SparkContinuousBatchRequest *offer,
	SparkContinuousBatchDecision *decision_out);

/* Remove a still-queued offer (the caller's cancellation path; the
 * oversize offers this module keeps alive are withdrawn here). */
SparkStatus SparkContinuousBatchWithdraw(
	SparkContinuousBatch *controller,
	uint64_t request_id);

/* The engine told the caller a resident request reached terminal. The
 * lane stops emitting at once; its slot frees at the NEXT boundary (L3). */
SparkStatus SparkContinuousBatchRetire(
	SparkContinuousBatch *controller,
	uint64_t request_id);

/* Advance the proof model one step: decode lanes emit in slot order,
 * prefill lanes consume their rows out of what the decode reservation
 * leaves of max_input_rows (the engine's chunked-prefill shape), lanes
 * reaching their budget or previously retired mark FINISHED. The report's
 * rows_spent lets the host proofs assert the budget law at every step. */
SparkStatus SparkContinuousBatchStep(
	SparkContinuousBatch *controller,
	SparkContinuousBatchStepEvent *events,
	uint32_t events_capacity,
	SparkContinuousBatchStepReport *report_out);

/* One boundary: reclaim FINISHED lanes (L3), then drain the queue under
 * the policy. The drain admits while the C1 arithmetic fits and stops at
 * the first refusal (named and counted once; an aged pick's refusal also
 * holds younger offers out of the boundary - the reservation leg). The
 * admitted request_ids are returned in admission order for the caller to
 * submit to the engine. */
SparkStatus SparkContinuousBatchBoundary(
	SparkContinuousBatch *controller,
	uint64_t *released_request_ids,
	uint32_t released_capacity,
	uint32_t *released_count_out);

SparkStatus SparkContinuousBatchGetStatistics(
	const SparkContinuousBatch *controller,
	SparkContinuousBatchStatistics *statistics_out);
SparkStatus SparkContinuousBatchGetLane(
	const SparkContinuousBatch *controller,
	uint32_t slot,
	SparkContinuousBatchLaneView *lane_out);

#ifdef __cplusplus
}
#endif
