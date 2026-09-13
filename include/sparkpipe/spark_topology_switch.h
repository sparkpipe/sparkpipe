#pragma once

// The topology switch: TP16 <-> PP16 in about twenty seconds, with requests
// in flight, and with KV warm across the switch whenever the NVMe tier still
// holds it.
//
// WHY THIS EXISTS. Both deployment strategies serve the same model from the
// same weights, but they shard them differently, so a rank's resident bytes
// under one strategy are useless under the other: switching means unloading
// one recipe's packs and loading the other's. What does NOT change is the KV
// content. The attention cache here is MLA latent KV, and the latent vector
// for a (layer, token) is defined by the model's math, not by the topology:
// the package's offline shard recipe marks the latent projections replicated,
// which is what makes the latent cache identical on every rank under TP. Under PP each
// stage holds the same latent vectors for its own layers. So a switch
// re-shards KV residency but never invalidates KV content, and this module's
// entire key scheme exists to keep that true on the NVMe tier.
//
// THE KEY. Tier records are named by SparkTopologySwitchKvKey(namespace,
// content_hash):
//
//   content_hash   the chained token-run hash from cache/cache.h. It commits
//                  to the whole token prefix and to nothing else - no layer
//                  assignment, no rank, no degree - so it is strategy-neutral
//                  by construction.
//   namespace      hashes the model identity and the NEUTRAL cache geometry
//                  (layer count, slot bytes, block tokens, precision). It
//                  deliberately excludes the strategy. Two recipes of the
//                  same model share one namespace, so a TP->PP switch finds
//                  the old recipe's records by the same names it published
//                  them under; two models, or one model re-quantized, do not.
//
// THE LAYOUT. One record per block holds every layer's latent for the
// block's tokens, layer-major. Under TP a resume reads the whole record
// (every rank needs every layer). Under PP a stage reads its layers' slice -
// a contiguous range inside the record, so re-sharding on read is an offset
// computation, not a re-layout, and one neutral copy serves both strategies.
// Per-strategy duplicate copies would double the tier's write traffic to buy
// nothing the slice does not provide.
//
// THE PROTOCOL, one rank's view (all ranks run it in lockstep; the
// coordinator's role is only to say BEGIN):
//
//   QUIESCE     admissions close; in-flight sequences run to their next
//               token boundary. Bounded by one decode step, because a decode
//               step is atomic at the token boundary by design.
//   CHECKPOINT  every still-active sequence publishes a manifest record to
//               the tier (position, block key set, recipe it ran under) and
//               PINS its blocks, so the eviction clock cannot take them
//               between checkpoint and resume. Blocks already written back
//               during serving are re-published idempotently; the manifest
//               is the one new write.
//   SWAP        the swap device unloads the old recipe's modules and streams
//               the target recipe's packs from the rank's NVMe - pre-staged
//               there while the old recipe was serving, because the 20 s
//               budget only closes if the load is NVMe->GPU, not
//               network->GPU. This phase is the whole budget.
//   RESUME      each checkpointed sequence is classified: every block key
//               still on the tier -> WARM, its blocks planned into the
//               tier's lookahead and the sequence re-admitted without
//               prefill; any block evicted -> RECOMPUTE, and the scheduler
//               re-runs prefill from the prompt (the tier's ordinary demand
//               path still serves whatever suffix survived). Pins drop as
//               classification completes.
//
// THE BUDGET, and why the phases are shaped this way, is priced in
// docs/archive/TOPOLOGY_SWITCHING.md. The short version: quiesce and resume cost
// microseconds-to-milliseconds, checkpoint costs one small write per
// sequence, and swap costs pack_bytes / nvme_bandwidth, which is where the
// twenty seconds live.
//
// HOST-VERIFIABLE, like the tier it sits on: the swap device and the
// checkpoint write path are vtables, the tests drive them with mocks, and
// the state machine cannot tell the difference.

#include <stdint.h>

#include "sparkpipe/spark_nvme_tier.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_TOPOLOGY_SWITCH_ABI_VERSION 1u
#define SPARK_TOPOLOGY_SWITCH_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkTopologySwitchConfiguration))

// The strategies the ring switches between. Kept as an explicit enum rather
// than a degree integer because the switch protocol only ever asks "same or
// different" - the degrees live in the recipes, not here.
typedef enum SparkTopologyStrategy
{
	SPARK_TOPOLOGY_STRATEGY_TENSOR_PARALLEL = 0,
	SPARK_TOPOLOGY_STRATEGY_PIPELINE_PARALLEL = 1
}
SparkTopologyStrategy;

// One (model, strategy, precision) weight instantiation: the set of packs a
// rank has resident. recipe_id is the caller's hash over the model identity,
// the strategy and the pack manifest - two recipes with the same id ARE the
// same residency, and Begin to the running recipe is a no-op the caller
// should not issue.
typedef struct SparkTopologyRecipe
{
	uint64_t recipe_id;
	uint64_t weight_pack_bytes;  /* this rank's pack bytes the swap must stream */
	uint32_t strategy;           /* SparkTopologyStrategy */
	uint32_t reserved0;
}
SparkTopologyRecipe;

typedef enum SparkTopologySwitchState
{
	SPARK_TOPOLOGY_SWITCH_STEADY = 0,
	SPARK_TOPOLOGY_SWITCH_QUIESCE,
	SPARK_TOPOLOGY_SWITCH_CHECKPOINT,
	SPARK_TOPOLOGY_SWITCH_SWAP,
	SPARK_TOPOLOGY_SWITCH_RESUME
}
SparkTopologySwitchState;

// What resume decided about a sequence. WARM means every block was on the
// tier and prefill is skipped entirely; RECOMPUTE means at least one block
// was evicted and the sequence goes back through prefill - which still hits
// the tier for whatever survived, through the ordinary demand path.
typedef enum SparkTopologySwitchResumeClass
{
	SPARK_TOPOLOGY_SWITCH_RESUME_PENDING = 0,
	SPARK_TOPOLOGY_SWITCH_RESUME_WARM,
	SPARK_TOPOLOGY_SWITCH_RESUME_RECOMPUTE
}
SparkTopologySwitchResumeClass;

// The swap device: the rank's weight loader behind a vtable, for the same
// reason the tier's drive is one - the schedule is host-verifiable and the
// hardware binding is the small part that changes at bring-up.
//
//   begin_swap  starts the unload-plus-load from the running recipe to the
//               target. Must not block: the packs are pre-staged on NVMe and
//               the stream is asynchronous (DMA into staging, copy up).
//   poll_swap   SPARK_STATUS_OK once the target recipe is resident and
//               bindable, SPARK_STATUS_BUSY while the stream is in flight.
typedef SparkStatus (*SparkTopologySwitchBeginSwap)(
	void *context,
	const SparkTopologyRecipe *from,
	const SparkTopologyRecipe *target);
typedef SparkStatus (*SparkTopologySwitchPollSwap)(void *context);

typedef struct SparkTopologySwitchSwapDevice
{
	void *context;
	SparkTopologySwitchBeginSwap begin_swap;
	SparkTopologySwitchPollSwap poll_swap;
}
SparkTopologySwitchSwapDevice;

// The checkpoint's write path. The tier is read-only by design - Publish
// names the record and returns its offset, and the PUT is the caller's, so
// the one payload a checkpoint must newly write (the per-sequence manifest)
// goes through this callback. Serving-time KV write-back uses the same path
// long before a switch is ever requested.
typedef SparkStatus (*SparkTopologySwitchWriteBlock)(
	void *context,
	uint64_t device_offset,
	const void *payload,
	uint32_t bytes);

typedef struct SparkTopologySwitchConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint64_t kv_namespace;       /* model + neutral geometry, strategy-free */
	uint32_t max_sequences;      /* checkpoint table capacity */
	uint32_t max_blocks_per_sequence; /* manifest capacity per sequence */
	uint32_t step_time_microseconds;  /* one decode step: the quiesce bound */
	uint32_t manifest_block_bytes;    /* tier record size for manifests */
	uint64_t nvme_read_bytes_per_second;  /* budget arithmetic only */
	uint64_t nvme_write_bytes_per_second; /* budget arithmetic only */
	/* The swap's fixed cost - module unload, allocator reset, bind - with the
	   streaming priced separately. An estimate until hardware measures it;
	   carried here so the budget is honest about containing one. */
	uint64_t swap_fixed_microseconds;
	SparkNvmeTier *tier;         /* records publish into and resume reads from */
	/* What the rank is running when the machine starts. Begin to this recipe
	   is refused as a no-op, so it must be the truth, not a placeholder. */
	SparkTopologyRecipe initial_recipe;
}
SparkTopologySwitchConfiguration;

// The 20-second question, answered in parts. resume_warm_us is reported but
// NOT added to total_us: resume overlaps the first decode steps through the
// tier's lookahead, so it is traffic, not wall time. Everything else is
// serial.
typedef struct SparkTopologySwitchBudget
{
	uint64_t quiesce_us;         /* one decode step */
	uint64_t checkpoint_us;      /* manifest writes at write bandwidth */
	uint64_t swap_fixed_us;      /* as configured */
	uint64_t swap_stream_us;     /* target pack bytes at read bandwidth */
	uint64_t resume_warm_us;     /* warm KV bytes at read bandwidth, overlapped */
	uint64_t total_us;           /* quiesce + checkpoint + fixed + stream */
}
SparkTopologySwitchBudget;

typedef struct SparkTopologySwitchStatistics
{
	uint64_t switches_completed;
	uint64_t sequences_checkpointed;
	uint64_t sequences_resumed_warm;
	uint64_t sequences_resumed_recompute;
	uint64_t sequences_completed_mid_switch; /* finished during quiesce: free */
	uint64_t blocks_pinned;
	uint64_t manifest_writes;
	uint64_t manifest_bytes;
	uint64_t tier_blocks_found;    /* checkpoint found already on the tier */
	uint64_t tier_blocks_absent;   /* not yet written back; recompute at resume */
	uint64_t swaps_started;
}
SparkTopologySwitchStatistics;

typedef struct SparkTopologySwitch SparkTopologySwitch;
struct SparkTopologySwitch
{
	SparkTopologySwitchConfiguration configuration;
	SparkTopologySwitchSwapDevice swap_device;
	SparkTopologySwitchWriteBlock write_block;
	void *write_block_context;
	void *sequences;             /* the sequence table, from `tables` */
	uint64_t *block_keys;        /* max_sequences * max_blocks_per_sequence */
	uint8_t *manifest_buffer;    /* one manifest, staged for WriteBlock */
	SparkTopologyRecipe current_recipe;
	SparkTopologyRecipe target_recipe;
	uint32_t state;
	uint32_t sequence_count;
	uint32_t phase_cursor;       /* checkpoint/resume progress within a phase */
	uint32_t swap_started;       /* begin_swap issued, polling for resident */
	SparkStatus last_error;      /* why a phase is retrying, when it is */
	SparkTopologySwitchStatistics statistics;
};

// Bookkeeping bytes the caller provides as `tables`: the sequence table, the
// block-key pool (max_sequences * max_blocks_per_sequence entries) and the
// manifest staging buffer, one blob, no allocation after init - the arena
// pattern every long-lived structure here follows.
uint64_t SparkTopologySwitchTableBytes(
	const SparkTopologySwitchConfiguration *configuration);

SparkStatus SparkTopologySwitchInitialize(
	SparkTopologySwitch *sw,
	const SparkTopologySwitchConfiguration *configuration,
	const SparkTopologySwitchSwapDevice *swap_device,
	SparkTopologySwitchWriteBlock write_block,
	void *write_block_context,
	void *tables);

// The tier key, and the whole cross-strategy story in one function. The
// namespace must be computed WITHOUT the strategy: same model, same neutral
// geometry, either strategy -> same key -> a switch re-shards KV residency
// instead of invalidating content. Never zero; zero means unhashed upstream.
uint64_t SparkTopologySwitchKvKey(
	uint64_t kv_namespace,
	uint64_t content_hash);

// The scheduler's admission gate. Closed from the moment Begin lands until
// the new recipe is resident and classified - admitting into a recipe that
// is being unloaded is how a sequence ends up bound to freed weights.
uint32_t SparkTopologySwitchAdmissionsOpen(
	const SparkTopologySwitch *sw);

// Sequence lifecycle, mirroring admission and completion in the scheduler.
// Track registers a sequence admitted under the CURRENT recipe; SetSequenceKv
// names its tier block keys and position (kept current by the caller, so a
// checkpoint never has to ask); AtBoundary and Complete are the quiesce
// signals. A sequence that completes mid-switch is dropped from the
// checkpoint set - checkpointing finished work is the purest waste there is.
SparkStatus SparkTopologySwitchTrackSequence(
	SparkTopologySwitch *sw,
	uint64_t sequence_id,
	uint64_t recipe_id);

SparkStatus SparkTopologySwitchSetSequenceKv(
	SparkTopologySwitch *sw,
	uint64_t sequence_id,
	uint32_t position_tokens,
	const uint64_t *content_hashes,
	uint32_t hash_count);

SparkStatus SparkTopologySwitchSequenceAtBoundary(
	SparkTopologySwitch *sw,
	uint64_t sequence_id);

SparkStatus SparkTopologySwitchSequenceComplete(
	SparkTopologySwitch *sw,
	uint64_t sequence_id);

// Request the switch. STEADY only; admissions close synchronously inside
// this call, and the in-flight drain begins. Switching to the recipe already
// running is an argument error, not a fast path - a no-op switch that looks
// like a real one is how an operator script hides a misconfiguration.
SparkStatus SparkTopologySwitchBegin(
	SparkTopologySwitch *sw,
	const SparkTopologyRecipe *target);

// The between-steps driver, next to the tier's Pump in the caller's loop.
// Advances QUIESCE -> CHECKPOINT once every tracked sequence is at a token
// boundary or complete, CHECKPOINT -> SWAP once manifests and pins are down,
// SWAP -> RESUME when the swap device reports resident, and RESUME -> STEADY
// when every sequence is classified. Returns the state it left the machine
// in, so the caller's loop can log transitions without a second query.
SparkTopologySwitchState SparkTopologySwitchAdvance(
	SparkTopologySwitch *sw,
	uint32_t step_now);

SparkTopologySwitchState SparkTopologySwitchStateOf(
	const SparkTopologySwitch *sw);

// The recipe the machine is serving or switching to.
const SparkTopologyRecipe *SparkTopologySwitchCurrentRecipe(
	const SparkTopologySwitch *sw);

// What resume decided, per sequence. Valid once the machine is back to
// STEADY (and while RESUME runs); PENDING before classification.
SparkTopologySwitchResumeClass SparkTopologySwitchResumeClassOf(
	const SparkTopologySwitch *sw,
	uint64_t sequence_id);

// The serial-time estimate for switching to `target` with
// warm_kv_bytes of checkpointed KV to reload. Bandwidths and the fixed swap
// cost come from the configuration, so an estimate and the run that
// validates it read the same numbers.
SparkStatus SparkTopologySwitchEstimateBudget(
	const SparkTopologySwitchConfiguration *configuration,
	const SparkTopologyRecipe *target,
	uint32_t active_sequence_count,
	uint64_t warm_kv_bytes,
	SparkTopologySwitchBudget *budget_out);

void SparkTopologySwitchGetStatistics(
	const SparkTopologySwitch *sw,
	SparkTopologySwitchStatistics *statistics_out);

// Why a phase is retrying. OK in STEADY and whenever the machine is making
// progress; anything else names the failure the next Advance will retry.
SparkStatus SparkTopologySwitchLastError(const SparkTopologySwitch *sw);

#ifdef __cplusplus
}
#endif
