#pragma once

/*
 * Shared paged-KV host core for the resident-decode stages - ONE home
 * for the two things every paged-KV stage shares.
 *
 * 1. THE LANE-TABLE ENGINE (SparkPagedKv*): the parameterized
 *    DRIVER-CONTRACT half over the general prefix-cache core
 *    (runtime/prefix_cache.h, SparkPrefixCacheCore*) for stages whose
 *    reuse needs a donor recurrence at the matched boundary. The core
 *    owns the physical block pool, content-addressed publishing,
 *    longest-common-prefix matching at admit, refcounted sharing, and
 *    LRU eviction; this engine owns the per-lane table rows, private
 *    scratch coverage, and the witness-clamped checkpoint lifecycle.
 *    NAMING LAW: this file names no model and carries no per-model
 *    constant. Everything that differs between stages arrives through
 *    the per-model geometry callbacks (SparkPagedKvGeometryCallbacks)
 *    bound at Initialize - today the qwen36 and qwen38 ports bind one
 *    callback row each; a future port binds a new row instead of
 *    cloning ~540 lines (the second-copy defect class that produced
 *    audit findings F1 and F4, each fixed twice).
 *
 * 2. THE PAGE-GEOMETRY PRIMITIVES (static inline, bottom of this
 *    header): positions-to-blocks ceiling math, overflow-checked
 *    calloc, and the common pool-capacity predicate. These serve stage
 *    configurations on OTHER substrates too - dsv4's lazy page-
 *    residency cache over SparkKvPageCache keeps its own state machine
 *    (it never touches SparkPrefixCacheCore) but shares exactly this
 *    arithmetic. Header-only BY DESIGN: a substrate that links neither
 *    the engine nor the core picks the primitives up without gaining
 *    any external symbol, so stage archives stay free of core objects
 *    (the R2-C3 exclusion recorded in modules/resident_decode_stage_rules.mk).
 *
 * Engine behavior (shared verbatim by every bound port):
 *
 * - Lanes map one-to-one onto core sequences (a lane binds at most one
 *   residency; the sequence id is derived from the lane index inside
 *   the callback-supplied private range).
 * - Reuse needs the DONOR recurrence at the matched boundary (a stage
 *   whose reuse needs no donor simply never binds checkpoint slots).
 *   The core's LCP is CLAMPED to the deepest block boundary whose
 *   checkpoint is still bound: each checkpoint records the WITNESS
 *   block - the lane's physical block at that boundary when the
 *   recurrence was captured - and a match may use it only while its
 *   own attached block at that ordinal IS the witness (content
 *   identity by construction: the core verified the host token
 *   mirror). A clamped match re-admits the truncated prefix so the
 *   lane's sequence holds exactly the shared blocks - a resumed walk
 *   must never write inside a donor block.
 * - Coverage beyond the core-committed tokens (speculative draft
 *   scratch, decode growth past the open block, reuse-disabled
 *   deployments) borrows physical blocks straight from the core's free
 *   list. Borrowed blocks are private scratch: never published, never
 *   shared, returned to the free list on lane reset. They never enter
 *   a sequence, so verify/replay rewrites can never touch an immutable
 *   published block.
 *
 * Pure host logic - no CUDA here. The adapter uploads the resulting
 * per-lane table rows.
 */

#include <stdint.h>
#include <stdlib.h>

#include "runtime/prefix_cache.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_PAGED_KV_NO_BLOCK UINT32_MAX
#define SPARK_PAGED_KV_NO_SLOT UINT32_MAX
#define SPARK_PAGED_KV_NO_LANE UINT32_MAX

/* Validated through the geometry callbacks below; defined under them. */
struct SparkPagedKvConfiguration;

/*
 * Per-model geometry callbacks: the complete list of facts separating
 * the ports of the family. Each port defines ONE static instance with
 * static storage duration and hands its address to Initialize, which
 * keeps the pointer.
 */
typedef struct SparkPagedKvGeometryCallbacks
{
	/* Base of the port's PRIVATE core sequence-id range: lane L binds
	 * sequence id sequence_id_base + L. One live residency per lane,
	 * so ids cannot collide within the range; distinct ranges across
	 * ports keep the id spaces disjoint by construction. */
	uint64_t sequence_id_base;
	/* Port-side compile-time lane-table row-width ceiling (the port's
	 * serving positions cap over its block size). The engine itself is
	 * width-agnostic - admit scratch is heap, not stack - and
	 * Initialize refuses configurations wider than this bound. */
	uint32_t max_blocks_per_lane;
	/* Optional extra configuration validation beyond the shared
	 * predicate; zero when the shared checks suffice (both ports
	 * today). Runs after the shared predicate accepts and before any
	 * state is built; return != SPARK_STATUS_OK to refuse. */
	SparkStatus (*validate_configuration)(
		const struct SparkPagedKvConfiguration *configuration);
}
SparkPagedKvGeometryCallbacks;

typedef struct SparkPagedKvConfiguration
{
	/* Positions per KV block; the module contracts pin 64. */
	uint32_t block_token_count;
	/* Resident lanes the table spans (max_active_sequence_count). */
	uint32_t lane_count;
	/* Table width per lane: ceil(max_sequence_positions / block_tokens). */
	uint32_t blocks_per_lane;
	/* Device KV blocks managed (the stage KV pool); the core's physical
	 * pool. */
	uint32_t physical_page_capacity;
	/* Prefix-directory capacity bound (kv_logical_page_capacity). Must be
	 * >= physical_page_capacity so every resident block can be indexed. */
	uint32_t logical_page_capacity;
	/* Donor checkpoint slots this cache may bind (a slice of the module
	 * snapshot slots). Zero disables reuse: every lane runs on borrowed
	 * scratch blocks and nothing is published. */
	uint32_t checkpoint_slot_count;
	/* KV bytes per block across every layer the stage owns. The core
	 * consumes this for reuse accounting only; computing it stays the
	 * stage adapter's job (never a hardcoded constant - see each
	 * adapter's pool setup). */
	uint64_t block_stride_bytes;
}
SparkPagedKvConfiguration;

/* Result of admitting a prompt: the clamped reuse depth (blocks) and the
 * checkpoint slot holding the donor recurrence at that boundary. */
typedef struct SparkPagedKvMatch
{
	uint32_t block_count;
	uint32_t checkpoint_slot;
}
SparkPagedKvMatch;

typedef struct SparkPagedKvCheckpoint
{
	uint32_t live;
	uint32_t lane;
	/* Boundary (in blocks) the captured recurrence corresponds to. */
	uint32_t boundary_blocks;
	/* The lane's published block at boundary_blocks-1 when the recurrence
	 * was captured; a match through this slot requires the matcher's own
	 * attached block at that ordinal to be this exact block. */
	uint32_t witness_block;
	/* LRU stamp, monotonic. */
	uint64_t last_use;
}
SparkPagedKvCheckpoint;

typedef struct SparkPagedKv
{
	SparkPagedKvConfiguration configuration;
	/* The per-model geometry callbacks bound at Initialize. */
	const SparkPagedKvGeometryCallbacks *geometry;
	/* The general half: pool, content index, sequences. */
	SparkPrefixCacheCore core;
	/* Reuse armed (checkpoint slots bound); otherwise pure scratch. */
	uint32_t reuse_enabled;
	/* Caller table rows (borrowed; the uploaded table IS this view). */
	uint32_t *blocks_by_lane;
	uint32_t *counts_by_lane;
	/* Per lane: ordinals < core_blocks belong to the lane's core
	 * sequence; the rest are borrowed scratch. live mirrors a bound
	 * sequence (checkpoint offers need a walking lane). */
	uint32_t *lane_core_blocks;
	uint32_t *lane_live;
	SparkPagedKvCheckpoint *checkpoints;
	/* Outstanding checkpoint reservation (submits are synchronous, so at
	 * most one frame can hold one at a time). */
	uint32_t reserved_slot;
	uint32_t reserved_lane;
	uint64_t lru_clock;
	/* Admit scratch: attached-block row for the witnessed-depth clamp
	 * (blocks_per_lane entries, heap not stack - the width is a
	 * configuration, not a compile-time constant). Single host thread. */
	uint32_t *admit_scratch;
}
SparkPagedKv;

SparkStatus SparkPagedKvInitialize(
	SparkPagedKv *cache,
	const SparkPagedKvConfiguration *configuration,
	const SparkPagedKvGeometryCallbacks *geometry,
	uint32_t *blocks_by_lane,
	uint32_t *counts_by_lane);
void SparkPagedKvDestroy(SparkPagedKv *cache);
/* Release a lane residency: drops the core sequence (published blocks
 * stay cached for later matches), returns borrowed scratch, and kills
 * the lane checkpoints. */
void SparkPagedKvLaneReset(SparkPagedKv *cache, uint32_t lane);
/* Admit one cold lane's prompt over the core: bind the sequence, clamp
 * the core LCP to the deepest live witnessed checkpoint, re-admit the
 * truncated shared prefix when the clamp bites, and fill the lane's
 * table row. Returns the resume depth in match_out (0 = full walk). */
SparkStatus SparkPagedKvAdmit(
	SparkPagedKv *cache,
	uint32_t lane,
	const uint32_t *tokens,
	uint32_t token_count,
	SparkPagedKvMatch *match_out);
/* Grow a lane's coverage to end_position: append the continuation
 * tokens (position-ordered; tokens beyond the core-committed tail) to
 * the lane's sequence, then fill the row out to end_position with
 * borrowed scratch blocks. tokens == 0 / token_count == 0 grows by
 * borrowing only - speculative scratch that must never publish. */
SparkStatus SparkPagedKvCover(
	SparkPagedKv *cache,
	uint32_t lane,
	uint64_t end_position,
	const uint32_t *tokens,
	uint32_t token_count);
/* Tokens the lane's sequence has committed (the walk/append frontier). */
uint64_t SparkPagedKvCommittedTokens(
	const SparkPagedKv *cache,
	uint32_t lane);
/* Free physical blocks in the pool right now (core free list). */
uint32_t SparkPagedKvFreeBlocks(const SparkPagedKv *cache);
/* Offer a checkpoint slot for a frame about to walk tokens ending exactly
 * at end_position on this lane: returns 1 with slot_out set when the frame
 * should carry the checkpoint request, 0 otherwise. */
uint32_t SparkPagedKvCheckpointOffer(
	SparkPagedKv *cache,
	uint32_t lane,
	uint64_t end_position,
	uint32_t *slot_out);
/* Bind the slot: the module copied the lane recurrence out after a
 * successful walk to end_position; record the boundary under the lane's
 * block at that boundary as the witness. */
void SparkPagedKvCheckpointCommit(
	SparkPagedKv *cache,
	uint32_t lane,
	uint32_t slot,
	uint64_t end_position);
/* Drop an uncommitted offer (failed frame). */
void SparkPagedKvCheckpointAbort(
	SparkPagedKv *cache,
	uint32_t lane,
	uint32_t slot);

/*
 * Page-geometry primitives shared across substrates. static inline so
 * a consumer that links neither the engine nor the prefix-cache core
 * (dsv4's stage archive) gains no external symbol.
 */

/* Smallest block count covering positions positions at
 * block_token_count positions per block (ceiling division). */
static inline uint32_t SparkPagedKvBlocksForPositions(
	uint64_t positions, uint32_t block_token_count)
{
	return (uint32_t)((positions + block_token_count - 1u) /
		block_token_count);
}

/* calloc with the integer-overflow guard every host pool allocation in
 * this family wants; zero-count or zero-byte requests fail rather than
 * return a non-pointer success. */
static inline void *SparkPagedKvCheckedCalloc(uint64_t count, uint64_t bytes)
{
	if ( count == 0u || bytes == 0u || count > SIZE_MAX / bytes )
		return(0);
	return(calloc((size_t)count,(size_t)bytes));
}

/* Common paged-pool capacity predicate: nonzero page capacities and the
 * logical directory covering BOTH the resident pool and the widest
 * lane's page row. Model-specific bounds (layer windows, position caps)
 * stay model-side; call this after those pass. */
static inline uint32_t SparkPagedKvPoolGeometryIsValid(
	uint32_t logical_page_capacity,
	uint32_t physical_page_capacity,
	uint64_t maximum_positions,
	uint32_t block_token_count)
{
	uint32_t lane_page_capacity;
	if ( logical_page_capacity == 0u || physical_page_capacity == 0u ||
		block_token_count == 0u )
		return(0u);
	lane_page_capacity =
		SparkPagedKvBlocksForPositions(maximum_positions,
			block_token_count);
	return(logical_page_capacity >= physical_page_capacity &&
		logical_page_capacity >= lane_page_capacity ? 1u : 0u);
}

#ifdef __cplusplus
}
#endif
