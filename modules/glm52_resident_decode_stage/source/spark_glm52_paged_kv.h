#pragma once

/*
 * GLM-5.2 resident-decode paged KV = the shared paged-KV lane-table
 * engine (runtime/paged_kv_common.h, SparkPagedKv*) bound to this
 * stage's row of the family table. The engine owns every mechanism -
 * pool borrow/return, row sync, witness-clamped admit, checkpoint
 * offer/commit/abort, LRU; this port contributes only its geometry
 * callback row:
 *
 * - Sequence-id range: SPARK_GLM52_PAGED_KV_SEQUENCE_BASE, a PRIVATE
 *   core sequence-id space (base + lane index), disjoint from every
 *   other port's range.
 * - Table width ceiling: SPARK_GLM52_PAGED_KV_MAX_BLOCKS_PER_LANE -
 *   the model's native context (1048576) over 64-token blocks.
 *
 * Port-specific behavior lives ABOVE this shim (the adapter decides
 * when to walk, adopt, resume, and checkpoint - reuse needs the DONOR
 * recurrence at the matched boundary because the compressed-latent
 * layers cannot skip a walk without it) and BELOW it (the general
 * prefix-cache core owns pool/publish/match/evict); the engine between
 * them is shared verbatim with the other ports of the family.
 */

#include <stdint.h>

#include "runtime/prefix_cache.h"
#include "runtime/paged_kv_common.h"
#include "sparkpipe/spark_status.h"

#define SPARK_GLM52_PAGED_KV_NO_BLOCK SPARK_PAGED_KV_NO_BLOCK
#define SPARK_GLM52_PAGED_KV_NO_SLOT SPARK_PAGED_KV_NO_SLOT
#define SPARK_GLM52_PAGED_KV_NO_LANE SPARK_PAGED_KV_NO_LANE
/* Table width bound: the model's native context (1048576,
 * SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS) over the 64-token blocks
 * the module contracts pin (SPARK_GLM52_KV_BLOCK_TOKEN_COUNT). Enforced
 * by the engine from this port's geometry callbacks - the width is a
 * per-stage fact. */
#define SPARK_GLM52_PAGED_KV_MAX_BLOCKS_PER_LANE 16384u

/* Private core sequence-id range for this port's lanes ('GL' high
 * word; disjoint from the qwen and dsv4/k3 port ranges). */
#define SPARK_GLM52_PAGED_KV_SEQUENCE_BASE UINT64_C(0x474c000000000000)

typedef SparkPagedKvConfiguration SparkGlm52PagedKvConfiguration;
typedef SparkPagedKvMatch SparkGlm52PagedKvMatch;
typedef SparkPagedKvCheckpoint SparkGlm52PagedKvCheckpoint;
typedef SparkPagedKv SparkGlm52PagedKv;

SparkStatus SparkGlm52PagedKvInitialize(
	SparkGlm52PagedKv *cache,
	const SparkGlm52PagedKvConfiguration *configuration,
	uint32_t *blocks_by_lane,
	uint32_t *counts_by_lane);
void SparkGlm52PagedKvDestroy(SparkGlm52PagedKv *cache);
/* Release a lane residency: drops the core sequence (published blocks
 * stay cached for later matches), returns borrowed scratch, and kills
 * the lane checkpoints. */
void SparkGlm52PagedKvLaneReset(SparkGlm52PagedKv *cache, uint32_t lane);
/* Admit one cold lane's prompt over the core: bind the sequence, clamp
 * the core LCP to the deepest live witnessed checkpoint, re-admit the
 * truncated shared prefix when the clamp bites, and fill the lane's
 * table row. Returns the resume depth in match_out (0 = full walk). */
SparkStatus SparkGlm52PagedKvAdmit(
	SparkGlm52PagedKv *cache,
	uint32_t lane,
	const uint32_t *tokens,
	uint32_t token_count,
	SparkGlm52PagedKvMatch *match_out);
/* Grow a lane's coverage to end_position: append the continuation
 * tokens (position-ordered; tokens beyond the core-committed tail) to
 * the lane's sequence, then fill the row out to end_position with
 * borrowed scratch blocks. tokens == 0 / token_count == 0 grows by
 * borrowing only - speculative scratch that must never publish. */
SparkStatus SparkGlm52PagedKvCover(
	SparkGlm52PagedKv *cache,
	uint32_t lane,
	uint64_t end_position,
	const uint32_t *tokens,
	uint32_t token_count);
/* Tokens the lane's sequence has committed (the walk/append frontier). */
uint64_t SparkGlm52PagedKvCommittedTokens(
	const SparkGlm52PagedKv *cache,
	uint32_t lane);
/* Free physical blocks in the pool right now (core free list). */
uint32_t SparkGlm52PagedKvFreeBlocks(const SparkGlm52PagedKv *cache);
/* Offer a checkpoint slot for a frame about to walk tokens ending exactly
 * at end_position on this lane: returns 1 with slot_out set when the frame
 * should carry the checkpoint request, 0 otherwise. */
uint32_t SparkGlm52PagedKvCheckpointOffer(
	SparkGlm52PagedKv *cache,
	uint32_t lane,
	uint64_t end_position,
	uint32_t *slot_out);
/* Bind the slot: the module copied the lane recurrence out after a
 * successful walk to end_position; record the boundary under the lane's
 * block at that boundary as the witness. */
void SparkGlm52PagedKvCheckpointCommit(
	SparkGlm52PagedKv *cache,
	uint32_t lane,
	uint32_t slot,
	uint64_t end_position);
/* Drop an uncommitted offer (failed frame). */
void SparkGlm52PagedKvCheckpointAbort(
	SparkGlm52PagedKv *cache,
	uint32_t lane,
	uint32_t slot);
