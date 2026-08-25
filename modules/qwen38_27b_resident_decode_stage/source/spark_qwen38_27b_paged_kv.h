#pragma once

/*
 * Qwen38_27b resident-decode paged KV over the GENERAL prefix-cache core
 * (runtime/prefix_cache.h, SparkPrefixCacheCore*): the core owns the
 * physical block pool, content-addressed publishing, longest-common-
 * prefix matching at admit, refcounted sharing, and LRU eviction. This
 * component is the model-specific half per the core header's DRIVER
 * CONTRACT plus the qwen38_27b hybrid wrinkle:
 *
 * - Lanes map one-to-one onto core sequences (a lane binds at most one
 *   residency; the sequence id is derived from the lane index).
 * - Reuse needs the DONOR GDN recurrence at the matched boundary (the
 *   hybrid layers cannot skip a walk without it). The core's LCP is
 *   therefore CLAMPED to the deepest block boundary whose checkpoint is
 *   still bound: each checkpoint records the WITNESS block - the lane's
 *   physical block at that boundary when the recurrence was captured -
 *   and a match may use it only while its own attached block at that
 *   ordinal IS the witness (content identity by construction: the core
 *   verified the host token mirror). A clamped match re-admits the
 *   truncated prefix so the lane's sequence holds exactly the shared
 *   blocks - a resumed walk must never write inside a donor block.
 * - Coverage beyond the core-committed tokens (speculative draft
 *   scratch, decode growth past the open block, reuse-disabled
 *   deployments) borrows physical blocks straight from the core's free
 *   list. Borrowed blocks are private scratch: never published, never
 *   shared, returned to the free list on lane reset. They never enter a
 *   sequence, so verify/replay rewrites can never touch an immutable
 *   published block.
 *
 * Pure host logic - no CUDA here. The adapter uploads the resulting
 * per-lane table rows.
 */

#include <stdint.h>

#include "prefix_cache.h"
#include "sparkpipe/spark_status.h"

#define SPARK_QWEN38_27B_PAGED_KV_NO_BLOCK UINT32_MAX
#define SPARK_QWEN38_27B_PAGED_KV_NO_SLOT UINT32_MAX
#define SPARK_QWEN38_27B_PAGED_KV_NO_LANE UINT32_MAX
/* Table width bound: the serving positions cap (8192) over 64-token
 * blocks. The lane table row width has this compile-time ceiling. */
#define SPARK_QWEN38_27B_PAGED_KV_MAX_BLOCKS_PER_LANE 128u

typedef struct SparkQwen38_27bPagedKvConfiguration
{
	/* Positions per KV block; the module contract pins 64. */
	uint32_t block_token_count;
	/* Resident lanes the table spans (max_active_sequence_count). */
	uint32_t lane_count;
	/* Table width per lane: ceil(max_sequence_positions / block_tokens). */
	uint32_t blocks_per_lane;
	/* Device KV blocks managed (the SPARK_QWEN38_27B_STAGE_KV_BLOCKS pool);
	 * the core's physical pool. */
	uint32_t physical_page_capacity;
	/* Prefix-directory capacity bound (kv_logical_page_capacity). Must be
	 * >= physical_page_capacity so every resident block can be indexed. */
	uint32_t logical_page_capacity;
	/* GDN checkpoint slots this cache may bind (a slice of the module gdn
	 * snapshot slots). Zero disables reuse: every lane runs on borrowed
	 * scratch blocks and nothing is published. */
	uint32_t checkpoint_slot_count;
	/* KV bytes per block across every layer the stage owns. The core
	 * consumes this for reuse accounting only (qwen38_27b keeps its KV row
	 * layout firmware-side, so the adapter reports its host-known
	 * accounting unit; see the adapter's pool setup). */
	uint64_t block_stride_bytes;
}
SparkQwen38_27bPagedKvConfiguration;

/* Result of admitting a prompt: the clamped reuse depth (blocks) and the
 * checkpoint slot holding the donor GDN recurrence at that boundary. */
typedef struct SparkQwen38_27bPagedKvMatch
{
	uint32_t block_count;
	uint32_t checkpoint_slot;
}
SparkQwen38_27bPagedKvMatch;

typedef struct SparkQwen38_27bPagedKvCheckpoint
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
SparkQwen38_27bPagedKvCheckpoint;

typedef struct SparkQwen38_27bPagedKv
{
	SparkQwen38_27bPagedKvConfiguration configuration;
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
	SparkQwen38_27bPagedKvCheckpoint *checkpoints;
	/* Outstanding checkpoint reservation (submits are synchronous, so at
	 * most one frame can hold one at a time). */
	uint32_t reserved_slot;
	uint32_t reserved_lane;
	uint64_t lru_clock;
}
SparkQwen38_27bPagedKv;

SparkStatus SparkQwen38_27bPagedKvInitialize(
	SparkQwen38_27bPagedKv *cache,
	const SparkQwen38_27bPagedKvConfiguration *configuration,
	uint32_t *blocks_by_lane,
	uint32_t *counts_by_lane);
void SparkQwen38_27bPagedKvDestroy(SparkQwen38_27bPagedKv *cache);
/* Release a lane residency: drops the core sequence (published blocks
 * stay cached for later matches), returns borrowed scratch, and kills
 * the lane checkpoints. */
void SparkQwen38_27bPagedKvLaneReset(SparkQwen38_27bPagedKv *cache, uint32_t lane);
/* Admit one cold lane's prompt over the core: bind the sequence, clamp
 * the core LCP to the deepest live witnessed checkpoint, re-admit the
 * truncated shared prefix when the clamp bites, and fill the lane's
 * table row. Returns the resume depth in match_out (0 = full walk). */
SparkStatus SparkQwen38_27bPagedKvAdmit(
	SparkQwen38_27bPagedKv *cache,
	uint32_t lane,
	const uint32_t *tokens,
	uint32_t token_count,
	SparkQwen38_27bPagedKvMatch *match_out);
/* Grow a lane's coverage to end_position: append the continuation
 * tokens (position-ordered; tokens beyond the core-committed tail) to
 * the lane's sequence, then fill the row out to end_position with
 * borrowed scratch blocks. tokens == 0 / token_count == 0 grows by
 * borrowing only - speculative scratch that must never publish. */
SparkStatus SparkQwen38_27bPagedKvCover(
	SparkQwen38_27bPagedKv *cache,
	uint32_t lane,
	uint64_t end_position,
	const uint32_t *tokens,
	uint32_t token_count);
/* Tokens the lane's sequence has committed (the walk/append frontier). */
uint64_t SparkQwen38_27bPagedKvCommittedTokens(
	const SparkQwen38_27bPagedKv *cache,
	uint32_t lane);
/* Free physical blocks in the pool right now (core free list). */
uint32_t SparkQwen38_27bPagedKvFreeBlocks(const SparkQwen38_27bPagedKv *cache);
/* Offer a checkpoint slot for a frame about to walk tokens ending exactly
 * at end_position on this lane: returns 1 with slot_out set when the frame
 * should carry the checkpoint request, 0 otherwise. */
uint32_t SparkQwen38_27bPagedKvCheckpointOffer(
	SparkQwen38_27bPagedKv *cache,
	uint32_t lane,
	uint64_t end_position,
	uint32_t *slot_out);
/* Bind the slot: the module copied the lane recurrence out after a
 * successful walk to end_position; record the boundary under the lane's
 * block at that boundary as the witness. */
void SparkQwen38_27bPagedKvCheckpointCommit(
	SparkQwen38_27bPagedKv *cache,
	uint32_t lane,
	uint32_t slot,
	uint64_t end_position);
/* Drop an uncommitted offer (failed frame). */
void SparkQwen38_27bPagedKvCheckpointAbort(
	SparkQwen38_27bPagedKv *cache,
	uint32_t lane,
	uint32_t slot);
