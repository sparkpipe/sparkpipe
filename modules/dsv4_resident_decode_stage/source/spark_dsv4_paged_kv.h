#pragma once

/*
 * DSV4 resident-decode paged KV = the shared paged-KV lane-table engine
 * (runtime/paged_kv_common.h, SparkPagedKv*) bound to this stage's row
 * of the family table. The engine owns every mechanism - pool
 * borrow/return, row sync, witness-clamped admit, checkpoint
 * offer/commit/abort, LRU; this port contributes only its geometry
 * callback row:
 *
 * - Sequence-id range: SPARK_DSV4_PAGED_KV_SEQUENCE_BASE, a PRIVATE
 *   core sequence-id space (base + lane index), disjoint from every
 *   other port's range.
 * - Table width ceiling: SPARK_DSV4_PAGED_KV_MAX_BLOCKS_PER_LANE - the
 *   model's position cap over the 128-token cache blocks the firmware
 *   pins.
 *
 * Relationship to SparkDsv4PagedCache: that substrate keeps its lazy
 * page-RESIDENCY state machine over SparkKvPageCache (R2-C3); this port
 * binds the general prefix-cache core for KV REUSE - content-addressed
 * publishing, longest-common-prefix matching, sharing, and LRU eviction
 * over the same physical page pool. Port-specific behavior lives ABOVE
 * this shim (the adapter decides when to walk, adopt, resume, and
 * checkpoint - reuse needs the DONOR recurrence at the matched boundary)
 * and BELOW it (the core owns pool/publish/match/evict); the engine
 * between them is shared verbatim with the other ports of the family.
 */

#include <stdint.h>

#include "runtime/prefix_cache.h"
#include "runtime/paged_kv_common.h"
#include "sparkpipe/spark_status.h"

#define SPARK_DSV4_PAGED_KV_NO_BLOCK SPARK_PAGED_KV_NO_BLOCK
#define SPARK_DSV4_PAGED_KV_NO_SLOT SPARK_PAGED_KV_NO_SLOT
#define SPARK_DSV4_PAGED_KV_NO_LANE SPARK_PAGED_KV_NO_LANE
/* Table width bound: the model's position cap (1048576,
 * SPARK_DSV4_MODEL_MAX_POSITIONS) over the 128-token cache blocks the
 * stage firmware pins
 * (SPARK_DSV4_RESIDENT_DECODE_STAGE_CACHE_BLOCK_TOKENS). Enforced by
 * the engine from this port's geometry callbacks - the width is a
 * per-stage fact. */
#define SPARK_DSV4_PAGED_KV_MAX_BLOCKS_PER_LANE 8192u

/* Private core sequence-id range for this port's lanes ('DS' high
 * word; disjoint from the qwen, glm52, and k3 port ranges). */
#define SPARK_DSV4_PAGED_KV_SEQUENCE_BASE UINT64_C(0x4453000000000000)

typedef SparkPagedKvConfiguration SparkDsv4PagedKvConfiguration;
typedef SparkPagedKvMatch SparkDsv4PagedKvMatch;
typedef SparkPagedKvCheckpoint SparkDsv4PagedKvCheckpoint;
typedef SparkPagedKv SparkDsv4PagedKv;

SparkStatus SparkDsv4PagedKvInitialize(
	SparkDsv4PagedKv *cache,
	const SparkDsv4PagedKvConfiguration *configuration,
	uint32_t *blocks_by_lane,
	uint32_t *counts_by_lane);
void SparkDsv4PagedKvDestroy(SparkDsv4PagedKv *cache);
/* Release a lane residency: drops the core sequence (published blocks
 * stay cached for later matches), returns borrowed scratch, and kills
 * the lane checkpoints. */
void SparkDsv4PagedKvLaneReset(SparkDsv4PagedKv *cache, uint32_t lane);
/* Admit one cold lane's prompt over the core: bind the sequence, clamp
 * the core LCP to the deepest live witnessed checkpoint, re-admit the
 * truncated shared prefix when the clamp bites, and fill the lane's
 * table row. Returns the resume depth in match_out (0 = full walk). */
SparkStatus SparkDsv4PagedKvAdmit(
	SparkDsv4PagedKv *cache,
	uint32_t lane,
	const uint32_t *tokens,
	uint32_t token_count,
	SparkDsv4PagedKvMatch *match_out);
/* Grow a lane's coverage to end_position: append the continuation
 * tokens (position-ordered; tokens beyond the core-committed tail) to
 * the lane's sequence, then fill the row out to end_position with
 * borrowed scratch blocks. tokens == 0 / token_count == 0 grows by
 * borrowing only - speculative scratch that must never publish. */
SparkStatus SparkDsv4PagedKvCover(
	SparkDsv4PagedKv *cache,
	uint32_t lane,
	uint64_t end_position,
	const uint32_t *tokens,
	uint32_t token_count);
/* Tokens the lane's sequence has committed (the walk/append frontier). */
uint64_t SparkDsv4PagedKvCommittedTokens(
	const SparkDsv4PagedKv *cache,
	uint32_t lane);
/* Free physical blocks in the pool right now (core free list). */
uint32_t SparkDsv4PagedKvFreeBlocks(const SparkDsv4PagedKv *cache);
/* Offer a checkpoint slot for a frame about to walk tokens ending exactly
 * at end_position on this lane: returns 1 with slot_out set when the frame
 * should carry the checkpoint request, 0 otherwise. */
uint32_t SparkDsv4PagedKvCheckpointOffer(
	SparkDsv4PagedKv *cache,
	uint32_t lane,
	uint64_t end_position,
	uint32_t *slot_out);
/* Bind the slot: the module copied the lane recurrence out after a
 * successful walk to end_position; record the boundary under the lane's
 * block at that boundary as the witness. */
void SparkDsv4PagedKvCheckpointCommit(
	SparkDsv4PagedKv *cache,
	uint32_t lane,
	uint32_t slot,
	uint64_t end_position);
/* Drop an uncommitted offer (failed frame). */
void SparkDsv4PagedKvCheckpointAbort(
	SparkDsv4PagedKv *cache,
	uint32_t lane,
	uint32_t slot);
