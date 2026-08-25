#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_qwen38_max_model.h"
#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"
/* Compiled in for the conservation cases: the gate drives the paged-KV
 * unit directly, at the adapter's exact per-round call pattern, and
 * asserts free-list conservation after EVERY step (lessons 7+8). */
#include "spark_qwen38_max_paged_kv.h"

#ifndef TEST_QWEN38_SERVING_ADAPTER_PATH
#define TEST_QWEN38_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_QWEN38_SERVING_DRIVER_PATH
#define TEST_QWEN38_SERVING_DRIVER_PATH ""
#endif
#ifndef TEST_QWEN38_SERVING_CONFIG_PATH
#define TEST_QWEN38_SERVING_CONFIG_PATH ""
#endif
#ifndef TEST_QWEN38_SERVING_SQUEEZED_CONFIG_PATH
#define TEST_QWEN38_SERVING_SQUEEZED_CONFIG_PATH ""
#endif

/* CP-B walk-skip flag bits (CASE 11). The pccore_qwen38_walkskip.patch
 * pins these in the firmware header at the SAME bit positions as the
 * landed qwen38_27b flags; the local fallback keeps this gate compilable -
 * and the leg honestly inert - while the module side is unlanded. */
#ifndef SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFIX_RESUME
#define SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFIX_RESUME 0x00000400u
#endif
#ifndef SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_CHECKPOINT
#define SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_CHECKPOINT 0x00000800u
#endif

/*
 * Prefix-cache correctness gate for the qwen38 serving adapter (port 2
 * of 4), run against the recording fixture driver. Matrix B1/B4/B25/B512
 * ONLY - the descriptor ceiling pins
 * SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT at 512u,
 * so B1024 cell claims from the core gate are NOT imported here.
 *
 * Proven at the serving boundary:
 *   1. B1 shared and diverging prefixes: a second identical prompt's
 *      lane row ADOPTS the donor's published physical blocks verbatim;
 *      a diverging prompt shares exactly the longest common COMPLETE
 *      block run (block-granular by construction); a mid-block admit
 *      (shared run not block-aligned) stops at the last full block.
 *   2. ON vs OFF byte identity EVERYWHERE: the env toggle is a pure
 *      ledger switch - every walked row (lane, base position, token
 *      ids) and every emitted id is identical with reuse on and off;
 *      only the physical block ids behind the lanes differ, and with
 *      every lane live (no churn resets) the ON ledger holds strictly
 *      fewer DISTINCT blocks - the port's capacity win.
 *   3. B4 mixed batched prefill (row budget 512 respected): shared AND
 *      fresh prompts across lanes in one wave-major submission.
 *   4. B25 pressure on the squeezed deployment (256 positions -> 4
 *      blocks per lane): 25 resident sequences over five prompt roots;
 *      every submission completes or refuses CLEANLY up front.
 *   5. B512, the deployment-conditional cell (descriptor ceiling 512u,
 *      resident capacity 512 forms it): 512 resident sequences over 16
 *      prompt roots; identity holds and the ON ledger stays smaller.
 *   6. Decode churn crossing block boundaries: rows only ever GROW;
 *      blocks attached at round N-1 are still attached at round N.
 *   7. Wiring: the derived pool (resident capacity x blocks-per-lane)
 *      reaches the module through the strict env channel, and a derived
 *      pool past the logical page budget is refused.
 *   8. F1 free-list conservation, unit level (audit-probe pattern):
 *      driving spark_qwen38_paged_kv.c through >64 simulated B1
 *      speculative rounds - Cover(end=pos+1) plus the speculative
 *      extension Cover(end=pos+D+2) - crossing >=2 block boundaries,
 *      {core free list} U {lane rows} U {live sequences} U {core LRU}
 *      == pool after EVERY call, reuse ON and OFF, teardown included.
 *   9. Spec-conservation, core-reference-case shape: multi-token draft
 *      bursts over shared/diverging/mid-block admits, roomy (exact
 *      partition, evicted==0) and squeezed (evicted>0) variants.
 *  10. Squeezed B-matrix cells B1/B4/B25/B512 at unit level: exact-fit
 *      pools, admit/release churn, evicted>0 in EVERY cell, conserving
 *      teardown.
 *  11. CP-B walk-skip A/B (env-guarded by SPARK_QWEN38_MAX_GATE_WALK_SKIP=1;
 *      SKIP exit-0 naming the absent module-side piece otherwise): unit
 *      half - witness clamp resumes with verbatim adoption, OFF pure
 *      scratch, partition conserved after every call; serving half -
 *      rows_walked(ON) STRICTLY LESS than rows_walked(OFF), zero rows fed
 *      below the resume boundary, per-(lane,block) walked trajectory and
 *      emitted ids byte-identical, exactly one well-shaped PREFIX_RESUME
 *      frame naming a live checkpoint slot, GDN_CHECKPOINT captures on
 *      the donor walk. The pre-existing A/B cells (1/3/4/5) carry an
 *      ADAPTIVE identity basis: legacy whole-run equality until a
 *      PREFIX_RESUME frame appears, trajectory equality + strict
 *      reduction after - the leg cannot silently green a half-landed
 *      tree.
 */

#define GATE_MAX_ROWS 512u
#define GATE_MAX_LANES 512u

typedef uint32_t (*GateRecordCountFunction)(void);
typedef const void *(*GateRecordGetFunction)(uint32_t);
typedef void (*GateRecordResetFunction)(void);
typedef uint32_t (*GateKvBlockCountFunction)(void);
static GateRecordCountFunction gate_record_count;
static GateRecordGetFunction gate_record_get;
static GateRecordResetFunction gate_record_reset;
static GateKvBlockCountFunction gate_kv_blocks;

/* Mirror of the fixture record layout (tests/fixtures/
 * qwen38_serving_adapter_driver.c). */
typedef struct GateFrameRecord
{
	uint32_t prefill;
	uint32_t rows;
	uint32_t context_flags;
	uint32_t snapshot_index_present;
	uint32_t snapshot_index;
	uint32_t lane_index;
	uint64_t base_position;
	uint64_t sequence_id;
	uint32_t block_count;
	uint32_t token_ids[GATE_MAX_ROWS];
	uint32_t blocks[GATE_MAX_ROWS];
}
GateFrameRecord;

static const SparkModelServingAdapterInterface *adapter_interface_global;

typedef struct GateState
{
	uint32_t completion_count;
	void *execution_stream;
	SparkModelServingCompletion completion;
}
GateState;

static void GateCompletion(void *completion_context, const SparkModelServingCompletion *completion)
{
	GateState *state = (GateState *)completion_context;
	assert(state != 0 && completion != 0);
	state->completion = *completion;
	state->completion_count++;
}

static void GateConfiguration(SparkModelServingAdapterConfiguration *configuration,
	const char *config_path, const char *runtime_root, GateState *test_state,
	uint32_t active_lanes, uint32_t resident_capacity,
	uint32_t logical_pages)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	configuration->rank_index = 3u;
	configuration->stage_index = 3u;
	configuration->runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration->runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	/* The qwen38 adapter advertises ONE inflight submission (the lean
	 * module executes every frame on one slot, submit_return sync). */
	configuration->runtime_limits.max_inflight_submission_count = 1u;
	configuration->runtime_limits.max_active_sequence_count = active_lanes;
	configuration->runtime_limits.max_input_row_count = GATE_MAX_ROWS;
	configuration->runtime_limits.resident_sequence_capacity = resident_capacity;
	/* The qwen38 descriptor carries no JIT_KV capability, so the shared
	 * runtime-limit rule pins both page budgets at zero; the pool is
	 * derived from resident capacity x blocks-per-lane instead. The
	 * parameter below is kept as the intended pool-block budget for the
	 * harness's own assertions. */
	(void)logical_pages;
	configuration->runtime_root = runtime_root;
	configuration->node_id = "spark-test";
	configuration->node_target = "cuda.sm121.qwen38_max.resident_decode_stage.fp8";
	configuration->adapter_configuration_path = config_path;
	configuration->driver_shared_object_path = TEST_QWEN38_SERVING_DRIVER_PATH;
	configuration->driver_program_name = "resident_decode";
	configuration->execution_stream = test_state->execution_stream;
	configuration->completion_function = GateCompletion;
	configuration->completion_context = test_state;
}

/* One single-lane prefill submission carrying tokens[0..length) at
 * positions [0..length). */
static SparkStatus GatePrefill(void *adapter_state, GateState *test_state,
	uint32_t slot, uint64_t sequence_id, const uint32_t *tokens,
	uint32_t length, uint64_t submission_id)
{
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	static __thread uint32_t row_lane[GATE_MAX_ROWS];
	static __thread uint64_t row_positions[GATE_MAX_ROWS],row_sequences[GATE_MAX_ROWS];
	uint32_t row;
	memset(&submission,0,sizeof(submission));
	memset(&lane,0,sizeof(lane));
	for (row=0u; row<length; row++)
	{
		row_lane[row] = 0u;
		row_positions[row] = row;
		row_sequences[row] = sequence_id;
	}
	lane.sequence_id = sequence_id;
	lane.request_id = submission_id + 1000u;
	lane.request_generation = 1u;
	lane.step_generation = 1u;
	lane.resident_sequence_slot = slot;
	lane.context_token_count = length;
	lane.flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.tokens_per_sequence = 1u;
	submission.submission_id = submission_id;
	submission.request_id = submission_id + 1000u;
	submission.sequence_id = sequence_id;
	submission.control_generation = 1u;
	submission.transaction_id = submission_id + 2000u;
	submission.dispatch_generation = submission_id + 3000u;
	submission.request_generation = 1u;
	submission.step_generation = 1u;
	submission.residency.word0 = submission_id;
	submission.residency.word1 = 177u;
	submission.residency.generation = 277u;
	submission.residency.owner = 13u;
	submission.active_sequence_count = 1u;
	submission.new_token_count = length;
	submission.lane_count = 1u;
	submission.row_count = length;
	submission.token_count = length;
	submission.lanes = &lane;
	submission.token_ids = tokens;
	submission.row_positions = row_positions;
	submission.row_lane_indices = row_lane;
	submission.row_sequence_ids = row_sequences;
	test_state->completion_count = 0u;
	return(adapter_interface_global->submit(adapter_state,&submission));
}

/* One single-lane DECODE submission: one row at `position` carrying the
 * previously accepted token id, OUTPUT_TOKEN set. */
static SparkStatus GateDecode(void *adapter_state, GateState *test_state,
	uint32_t slot, uint64_t sequence_id, uint64_t position,
	uint32_t token_id, uint64_t submission_id)
{
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	uint32_t token_buffer;
	uint32_t row_lane;
	uint64_t row_position,row_sequence;
	memset(&submission,0,sizeof(submission));
	memset(&lane,0,sizeof(lane));
	token_buffer = token_id;
	row_lane = 0u;
	row_position = position;
	row_sequence = sequence_id;
	lane.sequence_id = sequence_id;
	lane.sequence_position = position;
	lane.request_id = submission_id + 1000u;
	lane.request_generation = 1u;
	lane.step_generation = 1u;
	lane.resident_sequence_slot = slot;
	lane.context_token_count = position + 1u;
	lane.flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.tokens_per_sequence = 1u;
	submission.submission_id = submission_id;
	submission.request_id = submission_id + 1000u;
	submission.sequence_id = sequence_id;
	submission.sequence_position = position;
	submission.control_generation = 1u;
	submission.transaction_id = submission_id + 2000u;
	submission.dispatch_generation = submission_id + 3000u;
	submission.request_generation = 1u;
	submission.step_generation = 1u;
	submission.residency.word0 = submission_id;
	submission.residency.word1 = 178u;
	submission.residency.generation = 277u;
	submission.residency.owner = 13u;
	submission.active_sequence_count = 1u;
	submission.new_token_count = 1u;
	submission.lane_count = 1u;
	submission.row_count = 1u;
	submission.token_count = 1u;
	submission.lanes = &lane;
	submission.token_ids = &token_buffer;
	submission.row_positions = &row_position;
	submission.row_lane_indices = &row_lane;
	submission.row_sequence_ids = &row_sequence;
	test_state->completion_count = 0u;
	return(adapter_interface_global->submit(adapter_state,&submission));
}

/* One BATCHED wave-major prefill: `lane_count` lanes, lane l carrying
 * lengths[l] tokens of prompts[l], positions [0..length); rows
 * interleave by wave exactly like the serving contract's round-major
 * order. The caller keeps the TOTAL row count <= max_input_row_count. */
static SparkStatus GatePrefillBatch(void *adapter_state, GateState *test_state,
	const uint32_t *slots, const uint64_t *sequence_ids,
	const uint32_t *const *prompts, const uint32_t *lengths,
	uint32_t lane_count, uint64_t submission_id)
{
	SparkModelServingSubmission submission;
	static __thread SparkModelServingLane lanes[8u];
	static __thread uint32_t rows_lane[8u * GATE_MAX_ROWS];
	static __thread uint32_t batch_tokens[8u * GATE_MAX_ROWS];
	static __thread uint64_t positions[8u * GATE_MAX_ROWS],sequences[8u * GATE_MAX_ROWS];
	uint32_t lane,row,count,max_length;
	memset(&submission,0,sizeof(submission));
	assert(lane_count <= 8u);
	max_length = 0u;
	for (lane=0u; lane<lane_count; lane++)
		if ( lengths[lane] > max_length )
			max_length = lengths[lane];
	count = 0u;
	for (row=0u; row<max_length; row++)
		for (lane=0u; lane<lane_count; lane++)
		{
			if ( row >= lengths[lane] )
				continue;
			rows_lane[count] = lane;
			positions[count] = row;
			sequences[count] = sequence_ids[lane];
			batch_tokens[count] = prompts[lane][row];
			count++;
		}
	memset(lanes,0,sizeof(SparkModelServingLane) * lane_count);
	for (lane=0u; lane<lane_count; lane++)
	{
		lanes[lane].sequence_id = sequence_ids[lane];
		lanes[lane].request_id = submission_id + 1000u + lane;
		lanes[lane].request_generation = 1u;
		lanes[lane].step_generation = 1u;
		lanes[lane].resident_sequence_slot = slots[lane];
		lanes[lane].context_token_count = lengths[lane];
		lanes[lane].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	}
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.tokens_per_sequence = 1u;
	submission.submission_id = submission_id;
	submission.request_id = submission_id + 1000u;
	submission.sequence_id = sequence_ids[0];
	submission.control_generation = 1u;
	submission.transaction_id = submission_id + 2000u;
	submission.dispatch_generation = submission_id + 3000u;
	submission.request_generation = 1u;
	submission.step_generation = 1u;
	submission.residency.word0 = submission_id;
	submission.residency.generation = 277u;
	submission.residency.owner = 13u;
	submission.active_sequence_count = lane_count;
	submission.new_token_count = count;
	submission.lane_count = lane_count;
	submission.row_count = count;
	submission.token_count = count;
	submission.lanes = lanes;
	submission.token_ids = batch_tokens;
	submission.row_positions = positions;
	submission.row_lane_indices = rows_lane;
	submission.row_sequence_ids = sequences;
	test_state->completion_count = 0u;
	return(adapter_interface_global->submit(adapter_state,&submission));
}

/* ---- per-run observation: walk digest + distinct-block census ---- */

#define GATE_CENSUS_LANES 512u
#define GATE_CENSUS_BLOCKS 64u

typedef struct GateRunObservation
{
	uint64_t walk_digest;
	uint64_t output_digest;
	uint32_t record_count;
	uint32_t distinct_blocks;
	/* CASE 11 walk census: rows actually fed to frames, PREFIX_RESUME
	 * sightings, the first resume base, and per-lane walked extents. */
	uint32_t rows_walked;
	uint32_t resume_seen;
	uint32_t checkpoint_seen;
	uint64_t resume_base;
	uint32_t lane_rows[GATE_CENSUS_LANES];
	uint64_t lane_end[GATE_CENSUS_LANES];
}
GateRunObservation;

/* Per-(lane,block-bucket) digests of walked row tuples (position,token)
 * live in GateDigestTable (below): chunking-invariant byte identity for
 * a walked trajectory. A bucket with no walked rows stays zero. */

static uint64_t GateDigestMix(uint64_t hash,uint64_t value)
{
	hash ^= value + (uint64_t)0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
	return(hash);
}

/* Digest every record in [from,to): the WALK (everything but physical
 * block ids) plus a census of the distinct block ids the ledger used
 * (byte map of pool_blocks entries), plus the CASE 11 walk census
 * (rows, resume/checkpoint sightings, per-lane extents, per-block
 * trajectory digests). */
static void GateObserve(GateRunObservation *observation,uint32_t from,uint32_t to,
	uint8_t *block_seen,uint32_t pool_blocks,
	uint64_t block_digests[GATE_CENSUS_LANES][GATE_CENSUS_BLOCKS])
{
	uint32_t index,row;
	observation->record_count += to - from;
	for (index=from; index<to; index++)
	{
		const GateFrameRecord *record = (const GateFrameRecord *)gate_record_get(index);
		assert(record != 0);
		observation->walk_digest = GateDigestMix(observation->walk_digest,record->prefill);
		observation->walk_digest = GateDigestMix(observation->walk_digest,record->rows);
		observation->walk_digest = GateDigestMix(observation->walk_digest,record->context_flags);
		observation->walk_digest = GateDigestMix(observation->walk_digest,record->lane_index);
		observation->walk_digest = GateDigestMix(observation->walk_digest,record->base_position);
		observation->walk_digest = GateDigestMix(observation->walk_digest,record->sequence_id);
		observation->walk_digest = GateDigestMix(observation->walk_digest,record->block_count);
		if ( (record->context_flags & SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFIX_RESUME) != 0u )
		{
			if ( observation->resume_seen == 0u ||
				record->base_position < observation->resume_base )
				observation->resume_base = record->base_position;
			observation->resume_seen++;
		}
		if ( (record->context_flags & SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_CHECKPOINT) != 0u )
			observation->checkpoint_seen++;
		observation->rows_walked += record->rows;
		assert(record->lane_index < GATE_CENSUS_LANES);
		observation->lane_rows[record->lane_index] += record->rows;
		observation->lane_end[record->lane_index] =
			record->base_position + record->rows;
		for (row=0u; row<record->rows && row<GATE_MAX_ROWS; row++)
		{
			uint64_t position = record->base_position + (uint64_t)row;
			observation->walk_digest = GateDigestMix(observation->walk_digest,record->token_ids[row]);
			assert(position / SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS < GATE_CENSUS_BLOCKS);
			block_digests[record->lane_index][position / SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS] =
				GateDigestMix(block_digests[record->lane_index][position / SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS],
					(position << 32) | record->token_ids[row]);
		}
		for (row=0u; row<record->block_count && row<GATE_MAX_ROWS; row++)
		{
			uint32_t block = record->blocks[row];
			assert(block < pool_blocks);
			if ( block_seen[block] == 0u )
			{
				block_seen[block] = 1u;
				observation->distinct_blocks++;
			}
		}
	}
}

/* Count how many of `blocks`' first `block_count` entries appear in the
 * donor block set (the adoption probe). */
static uint32_t GateSharedWithDonor(const GateFrameRecord *record,
	const uint32_t *donor_blocks,uint32_t donor_count)
{
	uint32_t ordinal,donor,shared = 0u;
	for (ordinal=0u; ordinal<record->block_count && ordinal<GATE_MAX_ROWS; ordinal++)
		for (donor=0u; donor<donor_count; donor++)
			if ( donor_blocks[donor] == record->blocks[ordinal] )
			{
				shared++;
				break;
			}
	return(shared);
}

/* ---- CASE 11: the CP-B walk-skip verifier (PURE - mock-proved) ----- */

typedef enum GateWalkSkipVerdict
{
	GateWalkSkipOk = 0,
	GateWalkSkipRowsNotReduced,
	GateWalkSkipIdentityBroken,
	GateWalkSkipConservationBroken
}
GateWalkSkipVerdict;

typedef uint64_t GateDigestTable[GATE_CENSUS_LANES][GATE_CENSUS_BLOCKS];

/* Shared observation scratch for the pre-existing A/B cells (legacy
 * whole-run identity basis; contents consulted only once a leg actually
 * shows a PREFIX_RESUME frame). CASE 11 keeps its own pair below. */
static GateDigestTable s_observes_off;
static GateDigestTable s_observes_on;

static void GateObservesReset(void)
{
	memset(s_observes_off,0,sizeof(s_observes_off));
	memset(s_observes_on,0,sizeof(s_observes_on));
}

/* Verify one OFF-vs-ON serving pair under an ARMED walk-skip surface.
 * Pure function of the two observations + trajectory tables so a scratch
 * probe can drive every FAIL branch with synthetic observations. Contract
 * enforced:
 *   - rows_walked(ON) STRICTLY LESS THAN rows_walked(OFF)   (reduction)
 *   - PREFIX_RESUME frames appeared                          (armed)
 *   - PER LANE: every walked row above that lane's first-walked block is
 *     byte-identical to the OFF leg's trajectory digest for the same
 *     (lane,position,token) tuples, zero rows below it, and the walk END
 *     position is identical - i.e. exactly each lane's shared prefix was
 *     skipped, nothing else                                  (identity)
 */
static GateWalkSkipVerdict GateWalkSkipVerifyPair(const GateRunObservation *off,
	const GateRunObservation *on,const GateDigestTable off_digests,
	const GateDigestTable on_digests,const char *cell)
{
	uint32_t lane,bucket,first;
	if ( on->rows_walked >= off->rows_walked )
	{
		fprintf(stderr,"walk_skip_verify %s: rows OFF=%u ON=%u\n",cell,off->rows_walked,on->rows_walked);
		return(GateWalkSkipRowsNotReduced);
	}
	if ( on->resume_seen == 0u )
	{
		fprintf(stderr,"walk_skip_verify %s: no PREFIX_RESUME frame in ON leg\n",cell);
		return(GateWalkSkipIdentityBroken);
	}
	for (lane=0u; lane<GATE_CENSUS_LANES; lane++)
	{
		if ( off->lane_rows[lane] == 0u && on->lane_rows[lane] == 0u )
			continue;
		/* A resumed lane must end at the SAME position either way. */
		if ( off->lane_end[lane] != on->lane_end[lane] )
		{
			fprintf(stderr,"walk_skip_verify %s: lane %u end OFF=%llu ON=%llu\n",
				cell,lane,(unsigned long long)off->lane_end[lane],
				(unsigned long long)on->lane_end[lane]);
			return(GateWalkSkipIdentityBroken);
		}
		/* This lane's resume point: its first walked block bucket. */
		first = GATE_CENSUS_BLOCKS;
		for (bucket=0u; bucket<GATE_CENSUS_BLOCKS; bucket++)
			if ( on_digests[lane][bucket] != 0u )
			{
				first = bucket;
				break;
			}
		for (bucket=0u; bucket<GATE_CENSUS_BLOCKS; bucket++)
		{
			if ( bucket < first )
			{
				/* Skipped prefix: ON fed ZERO rows here. */
				if ( on_digests[lane][bucket] != 0u )
				{
					fprintf(stderr,"walk_skip_verify %s: lane %u fed rows below its resume boundary bucket %u\n",cell,lane,bucket);
					return(GateWalkSkipIdentityBroken);
				}
			}
			else if ( on_digests[lane][bucket] != off_digests[lane][bucket] )
			{
				fprintf(stderr,"walk_skip_verify %s: lane %u bucket %u trajectory OFF=%llx ON=%llx\n",
					cell,lane,bucket,(unsigned long long)off_digests[lane][bucket],
					(unsigned long long)on_digests[lane][bucket]);
				return(GateWalkSkipIdentityBroken);
			}
		}
	}
	(void)cell;
	return(GateWalkSkipOk);
}

static void GateWalkSkipFail(const char *branch,const char *detail)
{
	fprintf(stderr,"WALK_SKIP_FAIL(%s): %s\n",branch,detail);
	exit(1);
}

/* ---- unit-level conservation over the paged KV (cases 8-10) ------- */

#define U1_BLOCKS_PER_LANE 32u
#define U1_POOL 64u
#define U1_BLOCK_TOKENS 64u
#define U1_ROUNDS 140u

typedef struct UnitFixture
{
	uint32_t *table;
	uint32_t *counts;
	uint32_t lane_count;
	uint32_t blocks_per_lane;
	uint32_t pool;
}
UnitFixture;

/* The {free list} U {lane rows} U {live sequences} U {core LRU} == pool
 * partition walk. Structural corruption (cycles, out-of-range blocks,
 * row-hygiene breaks) stays ASSERTED - the walk cannot continue through
 * it; semantic violations (block on the free list AND in use, orphaned
 * blocks) are COUNTED so the CASE 11 leg can name the failure instead
 * of aborting blind. Returns the violation count (0 = conserved). */
static uint32_t GatePartitionAudit(const SparkQwen38MaxPagedKv *cache,
	const UnitFixture *fixture)
{
	uint8_t *in_free,*in_use;
	uint32_t i,ordinal,next,violations = 0u;
	in_free = (uint8_t *)calloc(fixture->pool,1u);
	in_use = (uint8_t *)calloc(fixture->pool,1u);
	assert(in_free != 0 && in_use != 0);
	next = cache->core.free_block_head;
	while ( next != SPARK_PREFIX_CACHE_CORE_NO_BLOCK )
	{
		assert(next < fixture->pool);
		assert(in_free[next] == 0u); /* repeat visit == cycle/double-return */
		in_free[next] = 1u;
		next = cache->core.blocks[next].free_next;
	}
	/* Detached published blocks sit on the LRU (Trim recovers them). */
	next = cache->core.lru_head;
	while ( next != SPARK_PREFIX_CACHE_CORE_NO_BLOCK )
	{
		assert(next < fixture->pool);
		in_use[next] = 1u;
		next = cache->core.blocks[next].lru_next;
	}
	for (i=0u; i<fixture->lane_count; i++)
		for (ordinal=0u; ordinal<fixture->blocks_per_lane; ordinal++)
		{
			uint32_t block = fixture->table[i * fixture->blocks_per_lane + ordinal];
			if ( ordinal >= fixture->counts[i] )
			{
				assert(block == SPARK_QWEN38_MAX_PAGED_KV_NO_BLOCK); /* row hygiene */
				continue;
			}
			assert(block != SPARK_QWEN38_MAX_PAGED_KV_NO_BLOCK && block < fixture->pool);
			in_use[block] = 1u;
		}
	for (i=0u; i<cache->core.max_sequence_count; i++)
	{
		const SparkPrefixCacheCoreSequence *sequence = &cache->core.sequences[i];
		if ( sequence->used == 0u )
			continue;
		for (ordinal=0u; ordinal<sequence->block_count; ordinal++)
		{
			uint32_t block = cache->core.sequence_blocks[i * cache->core.sequence_block_capacity + ordinal];
			assert(block < fixture->pool);
			in_use[block] = 1u;
		}
	}
	for (i=0u; i<fixture->pool; i++)
	{
		if ( in_free[i] != 0u && in_use[i] != 0u )
			violations++; /* block on free list AND in use */
		else if ( in_free[i] == 0u && in_use[i] == 0u )
			violations++; /* orphan: neither free nor reachable */
	}
	free(in_free);
	free(in_use);
	return(violations);
}

static void UnitConservationCheck(const SparkQwen38MaxPagedKv *cache,
	const UnitFixture *fixture,const char *where,uint32_t step)
{
	uint32_t violations = GatePartitionAudit(cache,fixture);
	if ( violations != 0u )
		fprintf(stderr,"conservation %s step=%u violations=%u\n",where,step,violations);
	assert(violations == 0u);
	(void)where;
	(void)step;
}

static void UnitFixtureInit(UnitFixture *fixture,uint32_t *table,uint32_t *counts,
	uint32_t lane_count,uint32_t blocks_per_lane,uint32_t pool)
{
	fixture->table = table;
	fixture->counts = counts;
	fixture->lane_count = lane_count;
	fixture->blocks_per_lane = blocks_per_lane;
	fixture->pool = pool;
	memset(table,0xFF,(size_t)lane_count * blocks_per_lane * sizeof(table[0]));
	memset(counts,0,(size_t)lane_count * sizeof(counts[0]));
}

static void UnitConfiguration(SparkQwen38MaxPagedKvConfiguration *configuration,
	const UnitFixture *fixture,uint32_t checkpoint_slots)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->block_token_count = U1_BLOCK_TOKENS;
	configuration->lane_count = fixture->lane_count;
	configuration->blocks_per_lane = fixture->blocks_per_lane;
	configuration->physical_page_capacity = fixture->pool;
	configuration->logical_page_capacity = fixture->pool;
	configuration->checkpoint_slot_count = checkpoint_slots;
	configuration->block_stride_bytes = 4096u;
}

static void UnitFill(uint32_t *tokens,uint32_t base,uint32_t count)
{
	uint32_t i;
	for (i=0u; i<count; i++)
		tokens[i] = base + i * 7u + 1u;
}

/* Mirror the adapter's post-frame checkpoint binding: offer+commit every
 * publish boundary up to end_position so later admits can clamp to a
 * WITNESSED depth. Without this step the GDN law correctly refuses all
 * reuse (no donor recurrence anywhere). */
static void UnitBindWitnesses(SparkQwen38MaxPagedKv *cache,uint32_t lane,
	uint64_t end_position)
{
	uint64_t boundary;
	uint32_t slot;
	if ( cache->reuse_enabled == 0u )
		return;
	boundary = U1_BLOCK_TOKENS;
	for ( ; boundary <= end_position; boundary += U1_BLOCK_TOKENS)
		if ( SparkQwen38MaxPagedKvCheckpointOffer(cache,lane,boundary,&slot) != 0u )
			SparkQwen38MaxPagedKvCheckpointCommit(cache,lane,slot,boundary);
}

/* CASE 8: the adapter's exact B1 speculative round shape - Cover(end =
 * pos + 1) for the accepted row, then the speculative extension
 * Cover(end = pos + D + 2) with NO canonical tokens while scratch is
 * outstanding - >64 rounds crossing >=2 block boundaries, conservation
 * after EVERY call, reuse ON and OFF, conserving teardown. */
static uint32_t u1_table[2u * U1_BLOCKS_PER_LANE];
static uint32_t u1_counts[2u];

static int UnitSpecChurnReuse(uint32_t reuse_on)
{
	SparkQwen38MaxPagedKv cache;
	SparkQwen38MaxPagedKvConfiguration configuration;
	SparkQwen38MaxPagedKvMatch match;
	UnitFixture fixture;
	uint32_t prompt[256],i;
	SparkStatus status;
	UnitFixtureInit(&fixture,u1_table,u1_counts,2u,U1_BLOCKS_PER_LANE,U1_POOL);
	UnitConfiguration(&configuration,&fixture,reuse_on != 0u ? 4u : 0u);
	assert(SparkQwen38MaxPagedKvInitialize(&cache,&configuration,u1_table,u1_counts) == SPARK_STATUS_OK);
	UnitFill(prompt,10000u,130u); /* 130 tokens: 2 full blocks + 1 open */
	status = SparkQwen38MaxPagedKvAdmit(&cache,0u,prompt,130u,&match);
	assert(status == SPARK_STATUS_OK);
	UnitConservationCheck(&cache,&fixture,"spec/admit",0u);
	for (i=0u; i<U1_ROUNDS; i++)
	{
		uint64_t pos = 131ull + i; /* D=4 draft depth */
		status = SparkQwen38MaxPagedKvCover(&cache,0u,pos + 1u,0,0);
		assert(status == SPARK_STATUS_OK);
		UnitConservationCheck(&cache,&fixture,"spec/pre",i);
		status = SparkQwen38MaxPagedKvCover(&cache,0u,pos + 6u,0,0);
		assert(status == SPARK_STATUS_OK);
		UnitConservationCheck(&cache,&fixture,"spec/ext",i);
	}
	SparkQwen38MaxPagedKvLaneReset(&cache,0u);
	UnitConservationCheck(&cache,&fixture,"spec/reset",U1_ROUNDS);
	SparkQwen38MaxPagedKvDestroy(&cache);
	return(0);
}

/* CASE 9: the core reference case's population shape at the paged-KV
 * seam - shared, diverging and MID-BLOCK admits, then multi-token draft
 * bursts; roomy pool (exact partition, evicted==0), then the SAME
 * population squeezed (evicted>0 via retire-and-readmit churn). */
static void UnitSpecPopulation(uint32_t pool,uint32_t expect_evicted)
{
	SparkQwen38MaxPagedKv cache;
	SparkQwen38MaxPagedKvConfiguration configuration;
	SparkQwen38MaxPagedKvMatch match;
	static uint32_t table[8u * 8u];
	static uint32_t counts[8u];
	UnitFixture fixture;
	uint32_t prompt_a[96],prompt_b[96],prompt_c[96],round;
	uint32_t blocks_per_lane = 8u;
	SparkPrefixCacheCoreStats stats;
	SparkStatus status;
	UnitFixtureInit(&fixture,table,counts,8u,blocks_per_lane,pool);
	UnitConfiguration(&configuration,&fixture,4u);
	assert(SparkQwen38MaxPagedKvInitialize(&cache,&configuration,table,counts) == SPARK_STATUS_OK);
	UnitFill(prompt_a,500u,90u); /* 90 = 1 full block + 26 open */
	for (round=0u; round<90u; round++)
	{
		prompt_b[round] = round < 64u ? prompt_a[round] : 900u + round;
		prompt_c[round] = round < 96u ? prompt_a[round] : 700u + round; /* mid-block: 1.5 blocks shared -> adopt 1 */
	}
	status = SparkQwen38MaxPagedKvAdmit(&cache,0u,prompt_a,90u,&match);
	assert(status == SPARK_STATUS_OK && match.block_count == 0u);
	assert(SparkQwen38MaxPagedKvCover(&cache,0u,90u,0,0) == SPARK_STATUS_OK);
	UnitBindWitnesses(&cache,0u,90u); /* the adapter binds these post-frame */
	UnitConservationCheck(&cache,&fixture,"pop/admit-a",0u);
	status = SparkQwen38MaxPagedKvAdmit(&cache,1u,prompt_b,90u,&match);
	assert(status == SPARK_STATUS_OK && match.block_count == 1u); /* block-granular */
	UnitConservationCheck(&cache,&fixture,"pop/admit-b",1u);
	status = SparkQwen38MaxPagedKvAdmit(&cache,2u,prompt_c,90u,&match);
	assert(status == SPARK_STATUS_OK && match.block_count == 1u); /* mid-block -> 1 */
	UnitConservationCheck(&cache,&fixture,"pop/admit-c",2u);
	/* Multi-token draft bursts (2..5 accepted tokens per round) on every
	 * live lane; each burst walks toward the next publish boundary. */
	for (round=0u; round<24u; round++)
	{
		uint32_t lane,burst;
		for (lane=0u; lane<3u; lane++)
		{
			uint32_t tokens[8];
			uint64_t committed,end;
			uint32_t k;
			burst = 2u + (round + lane) % 4u;
			for (k=0u; k<burst; k++)
				tokens[k] = 3000u + round * 31u + lane * 7u + k;
			committed = SparkQwen38MaxPagedKvCommittedTokens(&cache,lane);
			end = committed + burst;
			assert(end <= (uint64_t)blocks_per_lane * U1_BLOCK_TOKENS);
			status = SparkQwen38MaxPagedKvCover(&cache,lane,end,tokens,burst);
			assert(status == SPARK_STATUS_OK);
			UnitConservationCheck(&cache,&fixture,"pop/burst",round * 3u + lane);
		}
	}
	if ( expect_evicted != 0u )
	{
		/* Squeezed churn: retire lane 0 and readmit fresh divergent
		 * prompts, then grow borrow-only scratch; the detached published
		 * blocks only come back through LRU eviction, so evicted must
		 * move before the pool truly runs out. Refusals at the exact-fit
		 * edge are tolerated - the ledger must stay conserved either
		 * way. */
		for (round=0u; round<10u; round++)
		{
			uint64_t committed,end;
			SparkQwen38MaxPagedKvLaneReset(&cache,0u);
			UnitFill(prompt_a,5000u + round * 131u,90u);
			status = SparkQwen38MaxPagedKvAdmit(&cache,0u,prompt_a,90u,&match);
			if ( status != SPARK_STATUS_OK )
			{
				UnitConservationCheck(&cache,&fixture,"pop/churn-refused",round);
				break;
			}
			UnitConservationCheck(&cache,&fixture,"pop/churn-admit",round);
			committed = SparkQwen38MaxPagedKvCommittedTokens(&cache,0u);
			end = committed + 64u * (round + 1u);
			if ( end > (uint64_t)blocks_per_lane * U1_BLOCK_TOKENS )
				end = (uint64_t)blocks_per_lane * U1_BLOCK_TOKENS;
			status = SparkQwen38MaxPagedKvCover(&cache,0u,end,0,0);
			if ( status != SPARK_STATUS_OK )
			{
				SparkQwen38MaxPagedKvLaneReset(&cache,0u);
				UnitConservationCheck(&cache,&fixture,"pop/churn-cover-refused",round);
				break;
			}
			UnitConservationCheck(&cache,&fixture,"pop/churn-cover",round);
		}
	}
	SparkPrefixCacheCoreQueryStats(&cache.core,&stats);
	printf("spec-population pool=%u evicted=%llu published_total=%llu\n",
		pool,(unsigned long long)stats.evicted_block_count,
		(unsigned long long)stats.published_block_count);
	assert(expect_evicted == 0u || stats.evicted_block_count > 0u);
	assert(expect_evicted != 0u || stats.evicted_block_count == 0u);
	/* Teardown conserves: every attached block returns. */
	for (round=0u; round<3u; round++)
		SparkQwen38MaxPagedKvLaneReset(&cache,round);
	UnitConservationCheck(&cache,&fixture,"pop/reset",99u);
	SparkQwen38MaxPagedKvDestroy(&cache);
}

/* CASE 10: the B-matrix at unit level, squeezed - exact-fit pools with
 * admit/release churn over shared prompt roots; evicted>0 in EVERY
 * cell, conservation at every step, conserving teardown. */
static void UnitSqueezedCell(uint32_t lanes,uint32_t pool_divisor)
{
	SparkQwen38MaxPagedKv cache;
	SparkQwen38MaxPagedKvConfiguration configuration;
	SparkQwen38MaxPagedKvMatch match;
	static uint32_t table[512u * 4u];
	static uint32_t counts[512u];
	UnitFixture fixture;
	/* Exact-fit pools squeeze only when demand approaches the full row
	 * width; a shared-root population peaks around three blocks per
	 * lane, so wide cells take a divisor to reach real pressure. */
	uint32_t blocks_per_lane = 4u,pool = lanes * blocks_per_lane / pool_divisor;
	assert(pool >= lanes);
	uint32_t prompt[256],lane,round;
	SparkPrefixCacheCoreStats stats;
	SparkStatus status;
	assert(lanes <= 512u);
	UnitFixtureInit(&fixture,table,counts,lanes,blocks_per_lane,pool);
	UnitConfiguration(&configuration,&fixture,lanes <= 8u ? 4u : 8u);
	assert(SparkQwen38MaxPagedKvInitialize(&cache,&configuration,table,counts) == SPARK_STATUS_OK);
	for (round=0u; round<3u; round++)
	{
		for (lane=0u; lane<lanes; lane++)
		{
			uint32_t root = lane % 16u;
			uint32_t length = 130u + (lane * 7u) % 60u;
			UnitFill(prompt,root * 1000u,length);
			if ( round != 0u )
				SparkQwen38MaxPagedKvLaneReset(&cache,lane);
			status = SparkQwen38MaxPagedKvAdmit(&cache,lane,prompt,length,&match);
			if ( status != SPARK_STATUS_OK )
			{
				/* Exact-fit squeeze: a full-width wave may refuse up
				 * front. Refusal must leave the lane cold and the pool
				 * conserved - never a partial attachment. */
				UnitConservationCheck(&cache,&fixture,"sq/refused",round * lanes + lane);
				continue;
			}
			UnitConservationCheck(&cache,&fixture,"sq/admit",round * lanes + lane);
			status = SparkQwen38MaxPagedKvCover(&cache,lane,length,0,0);
			if ( status != SPARK_STATUS_OK )
			{
				SparkQwen38MaxPagedKvLaneReset(&cache,lane);
				UnitConservationCheck(&cache,&fixture,"sq/cover-refused",round * lanes + lane);
				continue;
			}
			UnitConservationCheck(&cache,&fixture,"sq/cover",round * lanes + lane);
		}
	}
	SparkPrefixCacheCoreQueryStats(&cache.core,&stats);
	printf("squeezed B%u pool=%u evicted=%llu admitted=%llu\n",
		lanes,pool,(unsigned long long)stats.evicted_block_count,
		(unsigned long long)stats.admit_count);
	assert(stats.evicted_block_count > 0u);
	for (lane=0u; lane<lanes; lane++)
		SparkQwen38MaxPagedKvLaneReset(&cache,lane);
	UnitConservationCheck(&cache,&fixture,"sq/reset",999u);
	SparkQwen38MaxPagedKvDestroy(&cache);
}

/* ---- CASE 11 unit half: witness-clamped admit at the ledger level --- */

static uint32_t wsu_table[2u * 4u];
static uint32_t wsu_counts[2u];

/* The serving A/B's ledger shadow: donor admits 130 tokens and binds the
 * 64/128 witnesses; an identical sibling then admits. ON the clamp
 * RESUMES at 128 with the donor's two published blocks adopted verbatim
 * (the rows the frames will skip); OFF reuse is pure scratch - no match,
 * NO bound sequence at all (adoption is the ON-side win; the OFF lane
 * runs on borrowed scratch exactly like today's certified behavior).
 * The partition audit runs after EVERY call; any violation names
 * conservation-broken. Returns a verdict, never aborts; *reason carries
 * the specific broken contract for the named-FAIL printer. */
static GateWalkSkipVerdict GateWalkSkipUnitHalf(uint32_t reuse_on,
	const char **reason)
{
	SparkQwen38MaxPagedKv cache;
	SparkQwen38MaxPagedKvConfiguration configuration;
	SparkQwen38MaxPagedKvMatch match;
	UnitFixture fixture;
	uint32_t prompt[130],donor_blocks[4],i;
	GateWalkSkipVerdict verdict = GateWalkSkipOk;
	*reason = "unreachable";
	UnitFixtureInit(&fixture,wsu_table,wsu_counts,2u,4u,32u);
	UnitConfiguration(&configuration,&fixture,reuse_on != 0u ? 4u : 0u);
	if ( SparkQwen38MaxPagedKvInitialize(&cache,&configuration,wsu_table,wsu_counts) != SPARK_STATUS_OK )
	{
		*reason = "unit fixture init refused";
		return(GateWalkSkipIdentityBroken);
	}
	UnitFill(prompt,42000u,130u); /* two full blocks + two open tokens */
	if ( SparkQwen38MaxPagedKvAdmit(&cache,0u,prompt,130u,&match) != SPARK_STATUS_OK ||
		match.block_count != 0u )
	{
		*reason = "cold donor admit matched on an empty pool";
		return(GateWalkSkipIdentityBroken);
	}
	if ( GatePartitionAudit(&cache,&fixture) != 0u )
	{
		*reason = "partition broken after donor admit";
		return(GateWalkSkipConservationBroken);
	}
	if ( SparkQwen38MaxPagedKvCover(&cache,0u,130u,0,0) != SPARK_STATUS_OK )
	{
		*reason = "donor cover refused on a roomy pool";
		return(GateWalkSkipIdentityBroken);
	}
	if ( GatePartitionAudit(&cache,&fixture) != 0u )
	{
		*reason = "partition broken after donor cover";
		return(GateWalkSkipConservationBroken);
	}
	UnitBindWitnesses(&cache,0u,130u); /* the adapter binds these post-frame */
	for (i=0u; i<2u; i++)
		donor_blocks[i] = fixture.table[0u * fixture.blocks_per_lane + i];
	if ( SparkQwen38MaxPagedKvAdmit(&cache,1u,prompt,130u,&match) != SPARK_STATUS_OK )
	{
		*reason = "identical sibling admit refused";
		return(GateWalkSkipIdentityBroken);
	}
	if ( GatePartitionAudit(&cache,&fixture) != 0u )
	{
		*reason = "partition broken after sibling admit";
		return(GateWalkSkipConservationBroken);
	}
	/* The resume contract: ON clamps to the deepest WITNESSED boundary
	 * (128 = 2 blocks) and adopts those blocks verbatim; OFF matches
	 * nothing and binds NO sequence (pure scratch, identical walked rows
	 * by construction - adoption is the ON-side ledger win). */
	if ( reuse_on != 0u )
	{
		if ( match.block_count != 2u ||
			match.checkpoint_slot == SPARK_QWEN38_MAX_PAGED_KV_NO_SLOT )
		{
			*reason = "ON witness clamp did not resume at 128 with a live slot";
			return(GateWalkSkipIdentityBroken);
		}
		for (i=0u; i<2u; i++)
			if ( fixture.table[1u * fixture.blocks_per_lane + i] != donor_blocks[i] )
			{
				*reason = "ON resumed but the shared prefix was not adopted verbatim";
				return(GateWalkSkipIdentityBroken);
			}
		if ( SparkQwen38MaxPagedKvCommittedTokens(&cache,1u) != 130u )
		{
			*reason = "ON sibling frontier not at the full prompt length";
			return(GateWalkSkipIdentityBroken);
		}
	}
	else
	{
		if ( match.block_count != 0u )
		{
			*reason = "OFF matched a prefix without any bound checkpoint (pure-scratch violated)";
			return(GateWalkSkipIdentityBroken);
		}
		if ( SparkQwen38MaxPagedKvCommittedTokens(&cache,1u) != 0u )
		{
			*reason = "OFF sibling bound a ledger sequence with reuse disarmed";
			return(GateWalkSkipIdentityBroken);
		}
	}
	if ( SparkQwen38MaxPagedKvCover(&cache,1u,130u,0,0) != SPARK_STATUS_OK )
	{
		*reason = "sibling cover refused on a roomy pool";
		return(GateWalkSkipIdentityBroken);
	}
	if ( GatePartitionAudit(&cache,&fixture) != 0u )
	{
		*reason = "partition broken after sibling cover";
		return(GateWalkSkipConservationBroken);
	}
	/* Teardown conserves: both lanes drop, every attached block returns. */
	SparkQwen38MaxPagedKvLaneReset(&cache,0u);
	if ( GatePartitionAudit(&cache,&fixture) != 0u )
	{
		*reason = "partition broken after lane-0 reset";
		return(GateWalkSkipConservationBroken);
	}
	SparkQwen38MaxPagedKvLaneReset(&cache,1u);
	if ( GatePartitionAudit(&cache,&fixture) != 0u )
	{
		*reason = "partition broken after lane-1 reset (teardown not conserving)";
		return(GateWalkSkipConservationBroken);
	}
	SparkQwen38MaxPagedKvDestroy(&cache);
	return(verdict);
}

int main(void)
{
	SparkModelServingAdapterConfiguration configuration;
	GateState test_state;
	void *adapter_state;
	void *driver_library;
	char runtime_root[4096];
	static uint32_t prompt_a[GATE_MAX_ROWS],prompt_c[GATE_MAX_ROWS],prompt_d[GATE_MAX_ROWS];
	uint32_t row,pool_blocks;
	memset(&test_state,0,sizeof(test_state));
	assert(cudaStreamCreate((cudaStream_t *)&test_state.execution_stream) == cudaSuccess);
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	driver_library = dlopen(TEST_QWEN38_SERVING_DRIVER_PATH,RTLD_NOW);
	assert(driver_library != 0);
	gate_record_count = (GateRecordCountFunction)dlsym(driver_library,"TestQwen38ServingDriverRecordCount");
	gate_record_get = (GateRecordGetFunction)dlsym(driver_library,"TestQwen38ServingDriverRecord");
	gate_record_reset = (GateRecordResetFunction)dlsym(driver_library,"TestQwen38ServingDriverResetRecords");
	gate_kv_blocks = (GateKvBlockCountFunction)dlsym(driver_library,"TestQwen38ServingDriverKvBlockCount");
	assert(gate_record_count != 0 && gate_record_get != 0 && gate_record_reset != 0 && gate_kv_blocks != 0);

	{
		SparkModelServingAdapterDynamicLibrary library;
		assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(TEST_QWEN38_SERVING_ADAPTER_PATH,SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE,&library) == SPARK_STATUS_OK);
		adapter_interface_global = &library.adapter_interface;

		/* Shared prompts: A is the donor; C shares the first 100 tokens
		 * then diverges (block-granular reuse must stop at 64); D shares
		 * 130 tokens (two full blocks) then diverges mid-block. */
		for (row=0u; row<200u; row++)
		{
			prompt_a[row] = 5000u + ((row * 13u) % 97u) * 7u;
			prompt_c[row] = row < 100u ? prompt_a[row] : 9000u + row;
			prompt_d[row] = row < 130u ? prompt_a[row] : 7000u + row;
		}

		/* ======== CASES 1+2: B1 adoption + ON/OFF byte identity ====== */
		{
			GateRunObservation on,off;
			static uint8_t seen_on[4096],seen_off[4096];
			uint32_t donor_blocks[8],donor_count;
			memset(&on,0,sizeof(on));
			memset(&off,0,sizeof(off));
			GateObservesReset();
			donor_count = 0u;

			setenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE","0",1);
			GateConfiguration(&configuration,TEST_QWEN38_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,8u,4096u);
			adapter_state = 0;
			{
				SparkStatus init_status = library.adapter_interface.initialize(&configuration,&adapter_state);
				if ( init_status != SPARK_STATUS_OK )
					fprintf(stderr,"init OFF failed status=%d\n",(int)init_status);
				assert(init_status == SPARK_STATUS_OK);
			}
			pool_blocks = gate_kv_blocks();
			printf("wiring kv_blocks=%u (resident 8 x blocks/lane 64)\n",pool_blocks);
			assert(pool_blocks == 8u * 64u);
			assert(pool_blocks <= sizeof(seen_off));
			gate_record_reset();
			assert(GatePrefill(adapter_state,&test_state,2u,1001u,prompt_a,200u,800u) == SPARK_STATUS_OK);
			assert(GatePrefill(adapter_state,&test_state,3u,1002u,prompt_a,200u,801u) == SPARK_STATUS_OK);
			assert(GatePrefill(adapter_state,&test_state,4u,1003u,prompt_c,200u,802u) == SPARK_STATUS_OK);
			assert(GatePrefill(adapter_state,&test_state,5u,1004u,prompt_d,200u,803u) == SPARK_STATUS_OK);
			off.output_digest = GateDigestMix(off.output_digest,test_state.completion.token_ids[0]);
			GateObserve(&off,0u,gate_record_count(),seen_off,pool_blocks,s_observes_off);
			assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
			library.adapter_interface.destroy(adapter_state);
			unsetenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE");

			setenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE","1",1);
			GateConfiguration(&configuration,TEST_QWEN38_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,8u,4096u);
			adapter_state = 0;
			assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
			assert(gate_kv_blocks() == pool_blocks);
			gate_record_reset();
			/* Donor walks everything. */
			assert(GatePrefill(adapter_state,&test_state,2u,1001u,prompt_a,200u,800u) == SPARK_STATUS_OK);
			assert(test_state.completion_count == 1u);
			{
				const GateFrameRecord *last = (const GateFrameRecord *)gate_record_get(gate_record_count() - 1u);
				assert(last->lane_index == 2u && last->block_count == 4u); /* ceil(200/64) */
				for (row=0u; row<last->block_count && donor_count<8u; row++)
					donor_blocks[donor_count++] = last->blocks[row];
			}
			/* Identical prompt: the lane row adopts the donor blocks. */
			assert(GatePrefill(adapter_state,&test_state,3u,1002u,prompt_a,200u,801u) == SPARK_STATUS_OK);
			{
				const GateFrameRecord *last = (const GateFrameRecord *)gate_record_get(gate_record_count() - 1u);
				uint32_t shared;
				assert(last->block_count == 4u);
				shared = GateSharedWithDonor(last,donor_blocks,donor_count);
				printf("b1_identical adopted=%u of %u blocks\n",shared,last->block_count);
				assert(shared == 3u); /* the open tail block stays private */
			}
			/* Diverging at 100: shares exactly the first block. */
			assert(GatePrefill(adapter_state,&test_state,4u,1003u,prompt_c,200u,802u) == SPARK_STATUS_OK);
			{
				const GateFrameRecord *last = (const GateFrameRecord *)gate_record_get(gate_record_count() - 1u);
				assert(GateSharedWithDonor(last,donor_blocks,donor_count) == 1u);
			}
			/* Mid-block admit at 130: shares exactly two blocks. */
			assert(GatePrefill(adapter_state,&test_state,5u,1004u,prompt_d,200u,803u) == SPARK_STATUS_OK);
			{
				const GateFrameRecord *last = (const GateFrameRecord *)gate_record_get(gate_record_count() - 1u);
				uint32_t shared = GateSharedWithDonor(last,donor_blocks,donor_count);
				printf("b1_midblock adopted=%u blocks\n",shared);
				assert(shared == 2u);
			}
			on.output_digest = GateDigestMix(on.output_digest,test_state.completion.token_ids[0]);
			GateObserve(&on,0u,gate_record_count(),seen_on,pool_blocks,s_observes_on);
			assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
			library.adapter_interface.destroy(adapter_state);

			/* Byte identity basis (adaptive). With the CP-B walk-skip
			 * surface unlanded, no PREFIX_RESUME frame can appear and
			 * the LEGACY basis binds: identical walked rows, frame
			 * counts, outputs. Once it lands, the claim moves to the
			 * TRAJECTORY - exactly the shared prefix skipped, every
			 * walked row byte-identical, strictly fewer rows fed -
			 * which GateWalkSkipVerifyPair enforces (CASE 11). */
			assert(on.output_digest == off.output_digest);
			if ( on.resume_seen != 0u )
			{
				GateWalkSkipVerdict walk_verdict = GateWalkSkipVerifyPair(&off,&on,
					s_observes_off,s_observes_on,"b1");
				assert(walk_verdict == GateWalkSkipOk);
				printf("b1_identity walk-skip rows OFF=%u ON=%u skipped=%u\n",
					off.rows_walked,on.rows_walked,off.rows_walked - on.rows_walked);
			}
			else
			{
				assert(on.walk_digest == off.walk_digest);
				assert(on.record_count == off.record_count);
			}
			printf("b1_identity ON distinct=%u OFF distinct=%u\n",on.distinct_blocks,off.distinct_blocks);
			assert(on.distinct_blocks < off.distinct_blocks);
			unsetenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE");
		}

		/* ============ CASE 3: B4 mixed batched prefill =============== */
		{
			GateRunObservation on,off;
			static uint8_t seen_on[4096],seen_off[4096];
			static uint32_t batch_a[200],batch_b[200],batch_c[200],batch_d[200];
			const uint32_t *prompts[4];
			uint32_t slots[4],lengths[4];
			uint64_t sequence_ids[4];
			memset(&on,0,sizeof(on));
			memset(&off,0,sizeof(off));
			GateObservesReset();
			/* Total rows 130+130+130+100 = 490 <= max_input_row_count. */
			for (row=0u; row<200u; row++)
			{
				batch_a[row] = 20000u + row * 3u;
				batch_b[row] = row < 64u ? batch_a[row] : 21000u + row;
				batch_c[row] = 22000u + row * 5u;
				batch_d[row] = row < 64u ? batch_c[row] : 23000u + row;
			}
			prompts[0] = batch_a; prompts[1] = batch_b; prompts[2] = batch_c; prompts[3] = batch_d;
			slots[0] = 1u; slots[1] = 2u; slots[2] = 3u; slots[3] = 4u;
			lengths[0] = 130u; lengths[1] = 130u; lengths[2] = 130u; lengths[3] = 100u;

			setenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE","0",1);
			GateConfiguration(&configuration,TEST_QWEN38_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,8u,4096u);
			adapter_state = 0;
			assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
			gate_record_reset();
			sequence_ids[0]=4100u; sequence_ids[1]=4101u; sequence_ids[2]=4102u; sequence_ids[3]=4103u;
			assert(GatePrefillBatch(adapter_state,&test_state,slots,sequence_ids,prompts,lengths,4u,910u) == SPARK_STATUS_OK);
			assert(test_state.completion_count == 1u);
			GateObserve(&off,0u,gate_record_count(),seen_off,pool_blocks,s_observes_off);
			assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
			library.adapter_interface.destroy(adapter_state);
			unsetenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE");

			setenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE","1",1);
			GateConfiguration(&configuration,TEST_QWEN38_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,8u,4096u);
			adapter_state = 0;
			assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
			gate_record_reset();
			sequence_ids[0]=4100u; sequence_ids[1]=4101u; sequence_ids[2]=4102u; sequence_ids[3]=4103u;
			assert(GatePrefillBatch(adapter_state,&test_state,slots,sequence_ids,prompts,lengths,4u,911u) == SPARK_STATUS_OK);
			assert(test_state.completion_count == 1u);
			GateObserve(&on,0u,gate_record_count(),seen_on,pool_blocks,s_observes_on);
			assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
			library.adapter_interface.destroy(adapter_state);
			if ( on.resume_seen != 0u )
			{
				GateWalkSkipVerdict walk_verdict = GateWalkSkipVerifyPair(&off,&on,
					s_observes_off,s_observes_on,"b4_mixed");
				assert(walk_verdict == GateWalkSkipOk);
			}
			else
				assert(on.walk_digest == off.walk_digest);
			printf("b4_mixed identity ON distinct=%u OFF distinct=%u\n",on.distinct_blocks,off.distinct_blocks);
			assert(on.distinct_blocks < off.distinct_blocks); /* two shared pairs */
			unsetenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE");
		}

		/* ============ CASE 4: B25 pressure (squeezed deployment) ====== */
		{
			GateRunObservation on,off;
			static uint8_t seen_on[4096],seen_off[4096];
			static uint32_t pressure_tokens[GATE_MAX_ROWS];
			uint32_t pass,statuses[2],refusals[2];
			memset(&on,0,sizeof(on));
			memset(&off,0,sizeof(off));
			GateObservesReset();
			for (pass=0u; pass<2u; pass++)
			{
				uint32_t root,ok = 0u,refused = 0u;
				setenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE",pass == 0u ? "0" : "1",1);
				/* Squeezed deployment: 256 positions -> 4 blocks/lane;
				 * 25 resident x 4 = 128 blocks for 25 sequences of <=189
				 * tokens (<=3 blocks) - fits, so identity is observable. */
				GateConfiguration(&configuration,TEST_QWEN38_SERVING_SQUEEZED_CONFIG_PATH,runtime_root,&test_state,25u,32u,256u);
				adapter_state = 0;
				assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
				assert(gate_kv_blocks() == 32u * 4u);
				gate_record_reset();
				for (root=0u; root<25u; root++)
				{
					uint32_t length = 130u + ((root * 37u) % 60u);
					fprintf(stderr,"b25_try pass=%u root=%u slot=%u length=%u\n",pass,root,2u+root,length);
					for (row=0u; row<length; row++)
						pressure_tokens[row] = row < 64u ?
							60000u + (root / 5u) * 1000u + row :
							60000u + root * 17u + row;
					{
						SparkStatus gate_status = GatePrefill(adapter_state,&test_state,2u + root,
							5001u + root,pressure_tokens,length,1000u + root);
						if ( gate_status == SPARK_STATUS_OK )
							ok++;
						else
						{
							refused++;
							fprintf(stderr,"b25_refused pass=%u root=%u length=%u status=%d\n",
								pass,root,length,(int)gate_status);
							/* Refusal must be clean: no completion fires. */
							assert(test_state.completion_count == 0u);
						}
					}
				}
				GateObserve(pass == 0u ? &off : &on,0u,gate_record_count(),
					pass == 0u ? seen_off : seen_on,128u,
					pass == 0u ? s_observes_off : s_observes_on);
				statuses[pass] = ok;
				refusals[pass] = refused;
				assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
				library.adapter_interface.destroy(adapter_state);
				unsetenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE");
			}
			printf("b25_pressure OFF ok=%u/%u ON ok=%u/%u distinct ON=%u OFF=%u\n",
				statuses[0],statuses[0] + refusals[0],statuses[1],statuses[1] + refusals[1],
				on.distinct_blocks,off.distinct_blocks);
			assert(statuses[0] == statuses[1] && refusals[0] == refusals[1]);
			assert(statuses[0] + refusals[0] == 25u);
			/* Identity holds over the accepted walks (adaptive basis:
			 * legacy whole-run equality until a resume frame appears,
			 * trajectory equality after - see CASE 11). */
			if ( on.resume_seen != 0u )
			{
				GateWalkSkipVerdict walk_verdict = GateWalkSkipVerifyPair(&off,&on,
					s_observes_off,s_observes_on,"b25_pressure");
				assert(walk_verdict == GateWalkSkipOk);
			}
			else
				assert(on.walk_digest == off.walk_digest);
		}

		/* ============ CASE 5: B512 (deployment-conditional) =========== */
		{
			GateRunObservation on,off;
			static uint8_t seen_on[4096],seen_off[4096];
			static uint32_t wave_prompts[16][GATE_MAX_ROWS];
			uint32_t pass,index,statuses[2];
			memset(&on,0,sizeof(on));
			memset(&off,0,sizeof(off));
			GateObservesReset();
			assert(SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT == 512u);
			/* 16 prompt ROOTS, 32 users each; users of one root are
			 * submitted CONSECUTIVELY - the 8 LRU witness slots reward
			 * temporal locality, so interleaving all 16 roots would
			 * steal every witness before its next user (measured). */
			for (index=0u; index<16u; index++)
			{
				uint32_t length = 190u;
				for (row=0u; row<length; row++)
					wave_prompts[index][row] = row < 64u ?
						80000u + index * 1000u + row :
						80000u + index * 13u + row;
			}
			for (pass=0u; pass<2u; pass++)
			{
				uint32_t ok = 0u;
				setenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE",pass == 0u ? "0" : "1",1);
				/* resident capacity 512 forms the cell: 512 x 4 = 2048
				 * blocks; 512 sequences of <=3 blocks each keep even the
				 * OFF ledger inside the pool. One submission per lane:
				 * the row budget (512) caps a submission, not the cell. */
				GateConfiguration(&configuration,TEST_QWEN38_SERVING_SQUEEZED_CONFIG_PATH,runtime_root,&test_state,512u,512u,4096u);
				adapter_state = 0;
				assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
				assert(gate_kv_blocks() == 512u * 4u);
				gate_record_reset();
				for (index=0u; index<512u; index++)
				{
					uint32_t root = index / 32u;
					uint32_t length = 130u + (index * 7u) % 60u;
					SparkStatus gate_status = GatePrefill(adapter_state,&test_state,index,
						9000u + index,wave_prompts[root],length,3000u + index);
					if ( gate_status == SPARK_STATUS_OK )
						ok++;
					else
						assert(test_state.completion_count == 0u);
				}
				assert(ok == 512u);
				GateObserve(pass == 0u ? &off : &on,0u,gate_record_count(),
					pass == 0u ? seen_off : seen_on,2048u,
					pass == 0u ? s_observes_off : s_observes_on);
				statuses[pass] = ok;
				assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
				library.adapter_interface.destroy(adapter_state);
				unsetenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE");
			}
			printf("b512 identity ON distinct=%u OFF distinct=%u (ok=%u/%u)\n",
				on.distinct_blocks,off.distinct_blocks,statuses[1],statuses[0]);
			if ( on.resume_seen != 0u )
			{
				GateWalkSkipVerdict walk_verdict = GateWalkSkipVerifyPair(&off,&on,
					s_observes_off,s_observes_on,"b512");
				assert(walk_verdict == GateWalkSkipOk);
			}
			else
				assert(on.walk_digest == off.walk_digest);
			assert(on.distinct_blocks < off.distinct_blocks);
		}

		/* ============ CASE 6: decode churn across boundaries ========== */
		{
			uint32_t prompt_e[126];
			uint32_t blocks_e[32],count_e = 0u,round,record;
			setenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE","1",1);
			GateConfiguration(&configuration,TEST_QWEN38_SERVING_CONFIG_PATH,runtime_root,&test_state,4u,4u,4096u);
			adapter_state = 0;
			assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
			gate_record_reset();
			for (row=0u; row<126u; row++)
				prompt_e[row] = 30000u + row * 5u;
			assert(GatePrefill(adapter_state,&test_state,1u,6001u,prompt_e,126u,1500u) == SPARK_STATUS_OK);
			for (record=gate_record_count(); record>0u; record--)
			{
				const GateFrameRecord *frame = (const GateFrameRecord *)gate_record_get(record - 1u);
				if ( frame->lane_index == 1u )
				{
					count_e = frame->block_count;
					memcpy(blocks_e,frame->blocks,sizeof(blocks_e));
					break;
				}
			}
			assert(count_e == 2u); /* ceil(126/64) */
			for (round=0u; round<140u; round++)
			{
				uint64_t position = 126ull + round;
				assert(GateDecode(adapter_state,&test_state,1u,6001u,position,
					8000u + round,1600u + round) == SPARK_STATUS_OK);
				assert(test_state.completion_count == 1u);
				{
					const GateFrameRecord *frame = 0;
					for (record=gate_record_count(); record>0u; record--)
					{
						const GateFrameRecord *candidate = (const GateFrameRecord *)gate_record_get(record - 1u);
						if ( candidate->lane_index == 1u )
						{
							frame = candidate;
							break;
						}
					}
					assert(frame != 0 && frame->block_count >= count_e);
					for (row=0u; row<count_e && row<frame->block_count; row++)
						assert(frame->blocks[row] == blocks_e[row]); /* rows only grow */
					count_e = frame->block_count;
					memcpy(blocks_e,frame->blocks,sizeof(blocks_e));
				}
			}
			/* 126 + 140 positions crossed the 192 AND 256 boundaries. */
			assert(count_e >= 4u);
			printf("decode_churn ok=140 rounds, row grown to %u blocks (>=2 crossings)\n",count_e);
			assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
			library.adapter_interface.destroy(adapter_state);
			unsetenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE");
		}

		/* ============ CASE 7: wiring + limits ======================== */
		/* Resident capacity bounds actives: below that is refused by the
		 * shared runtime-limit rule. */
		GateConfiguration(&configuration,TEST_QWEN38_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,4u,4096u);
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_INVALID_ARGUMENT);
		assert(adapter_state == 0);
		/* The derived pool follows the deployed slice (positive probe). */
		GateConfiguration(&configuration,TEST_QWEN38_SERVING_CONFIG_PATH,runtime_root,&test_state,4u,4u,4096u);
		adapter_state = 0;
		assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
		assert(gate_kv_blocks() == 4u * 64u);
		library.adapter_interface.destroy(adapter_state);

		/* ============ CASE 8: F1-style conservation =================== */
		assert(UnitSpecChurnReuse(1u) == 0);
		assert(UnitSpecChurnReuse(0u) == 0);
		printf("f1_conservation PASS (reuse on + off)\n");

		/* ============ CASE 9: spec population, roomy + squeezed ======= */
		UnitSpecPopulation(64u,0u);
		UnitSpecPopulation(16u,1u);
		printf("spec_population PASS (roomy + squeezed)\n");

		/* ============ CASE 10: squeezed B-matrix cells ================ */
		UnitSqueezedCell(1u,1u);
		UnitSqueezedCell(4u,1u);
		UnitSqueezedCell(25u,1u);
		UnitSqueezedCell(512u,2u);
		printf("squeezed_matrix PASS (B1/B4/B25/B512 evicted>0)\n");

		/* ======== CASE 11: CP-B walk-skip A/B (env-guarded leg) ======= */
		/*
		 * The port-2 honesty bound: with adoption-without-walk-skip the
		 * walked rows are identical ON vs OFF BY CONSTRUCTION, and no
		 * assertion here can see the difference. This leg arms ONLY when
		 * SPARK_QWEN38_MAX_GATE_WALK_SKIP=1 declares the CP-B module-side
		 * surface landed (pccore_qwen38_walkskip.patch); unarmed it
		 * SKIPs exit-0 naming exactly that absent piece - it can never
		 * silently green a half-landed tree. Armed, it demands:
		 *   - unit half: the witness clamp RESUMES at 128 with donor
		 *     blocks adopted verbatim; OFF matches nothing; {free} U
		 *     {lane rows} U {live seqs} U {core LRU} == pool after
		 *     EVERY call, teardown included;
		 *   - serving half: rows_walked(ON) STRICTLY LESS than
		 *     rows_walked(OFF) over >=2 spec-shaped rounds crossing
		 *     block boundaries with resume-at-witness, emitted ids and
		 *     per-(lane,block) walk digests byte-identical, zero rows
		 *     fed below the resume boundary, resume frame shaped
		 *     (PREFIX_RESUME + live snapshot slot at a block boundary),
		 *     and OFF carrying neither new flag.
		 */
		{
			const char *armed = getenv("SPARK_QWEN38_MAX_GATE_WALK_SKIP");
			if ( armed == 0 || strcmp(armed,"1") != 0 )
				printf("walk_skip SKIP: SPARK_QWEN38_MAX_GATE_WALK_SKIP unset - CP-B module-side landing not declared (pccore_qwen38_walkskip.patch pending); leg inert\n");
			else
			{
				static GateDigestTable ws_off_digests,ws_on_digests;
				static uint8_t ws_seen_off[4096],ws_seen_on[4096];
				GateRunObservation off,on;
				GateWalkSkipVerdict verdict;
				const char *reason = "";
				uint32_t resume_frame_seen,resume_slot,record_scan;
				printf("walk_skip ARMED: requiring strict row reduction + trajectory identity\n");
				/* Unit half first: ledger contract both ways. */
				verdict = GateWalkSkipUnitHalf(1u,&reason);
				if ( verdict == GateWalkSkipConservationBroken )
					GateWalkSkipFail("conservation-broken",reason);
				if ( verdict != GateWalkSkipOk )
					GateWalkSkipFail("identity-broken",reason);
				verdict = GateWalkSkipUnitHalf(0u,&reason);
				if ( verdict == GateWalkSkipConservationBroken )
					GateWalkSkipFail("conservation-broken",reason);
				if ( verdict != GateWalkSkipOk )
					GateWalkSkipFail("identity-broken",reason);
				/* Serving half: OFF leg then ON leg, donor + identical
				 * sibling of 200 tokens (witnessed boundaries 64/128/192). */
				setenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE","0",1);
				GateConfiguration(&configuration,TEST_QWEN38_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,8u,4096u);
				adapter_state = 0;
				assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
				gate_record_reset();
				memset(&off,0,sizeof(off));
			GateObservesReset();
				assert(GatePrefill(adapter_state,&test_state,2u,11001u,prompt_a,200u,7000u) == SPARK_STATUS_OK);
				assert(test_state.completion_count == 1u);
				off.output_digest = GateDigestMix(off.output_digest,test_state.completion.token_ids[0]);
				assert(GatePrefill(adapter_state,&test_state,3u,11002u,prompt_a,200u,7001u) == SPARK_STATUS_OK);
				assert(test_state.completion_count == 1u);
				off.output_digest = GateDigestMix(off.output_digest,test_state.completion.token_ids[0]);
				GateObserve(&off,0u,gate_record_count(),ws_seen_off,pool_blocks,ws_off_digests);
				assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
				library.adapter_interface.destroy(adapter_state);
				unsetenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE");
				/* OFF must be pure scratch: no CP-B flags anywhere. */
				if ( off.resume_seen != 0u || off.checkpoint_seen != 0u )
					GateWalkSkipFail("identity-broken","OFF leg carried PREFIX_RESUME/GDN_CHECKPOINT frames - switch is not a pure toggle");
				setenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE","1",1);
				GateConfiguration(&configuration,TEST_QWEN38_SERVING_CONFIG_PATH,runtime_root,&test_state,8u,8u,4096u);
				adapter_state = 0;
				assert(library.adapter_interface.initialize(&configuration,&adapter_state) == SPARK_STATUS_OK);
				gate_record_reset();
				memset(&on,0,sizeof(on));
				assert(GatePrefill(adapter_state,&test_state,2u,12001u,prompt_a,200u,7100u) == SPARK_STATUS_OK);
				assert(test_state.completion_count == 1u);
				on.output_digest = GateDigestMix(on.output_digest,test_state.completion.token_ids[0]);
				assert(GatePrefill(adapter_state,&test_state,3u,12002u,prompt_a,200u,7101u) == SPARK_STATUS_OK);
				assert(test_state.completion_count == 1u);
				on.output_digest = GateDigestMix(on.output_digest,test_state.completion.token_ids[0]);
				GateObserve(&on,0u,gate_record_count(),ws_seen_on,pool_blocks,ws_on_digests);
				assert(library.adapter_interface.quiesce(adapter_state,UINT64_MAX) == SPARK_STATUS_OK);
				library.adapter_interface.destroy(adapter_state);
				unsetenv("SPARK_QWEN38_MAX_SERVING_PREFIX_CACHE");
				verdict = GateWalkSkipVerifyPair(&off,&on,ws_off_digests,ws_on_digests,"walk_skip_serving");
				if ( verdict == GateWalkSkipRowsNotReduced )
					GateWalkSkipFail("rows-not-reduced","rows_walked(ON) not STRICTLY less than rows_walked(OFF) - module side unlanded or resume never fired");
				if ( verdict != GateWalkSkipOk )
					GateWalkSkipFail("identity-broken","skipped-prefix/post-resume trajectory mismatch or malformed resume shape");
				if ( on.output_digest != off.output_digest )
					GateWalkSkipFail("identity-broken","emitted token ids differ ON vs OFF");
				/* Resume-frame shape: block-aligned base naming a LIVE
				 * snapshot slot; capture frames exist on the donor. */
				resume_frame_seen = 0u;
				resume_slot = 0u;
				for (record_scan=0u; record_scan<gate_record_count(); record_scan++)
				{
					const GateFrameRecord *frame = (const GateFrameRecord *)gate_record_get(record_scan);
					if ( (frame->context_flags & SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFIX_RESUME) != 0u )
					{
						resume_frame_seen++;
						resume_slot = frame->snapshot_index;
						if ( frame->snapshot_index_present == 0u ||
							frame->base_position % SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS != 0u )
							GateWalkSkipFail("identity-broken","resume frame base not on a publish boundary or snapshot view missing");
					}
				}
				if ( resume_frame_seen != 1u || resume_slot >= SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_MAX_GDN_SNAPSHOT_SLOTS )
					GateWalkSkipFail("identity-broken","expected exactly one well-shaped PREFIX_RESUME frame naming a live checkpoint slot");
				if ( on.checkpoint_seen == 0u )
					GateWalkSkipFail("identity-broken","no GDN_CHECKPOINT capture frame on the donor walk");
				printf("walk_skip PASS rows OFF=%u ON=%u skipped=%u resume_base=%llu captures=%u\n",
					off.rows_walked,on.rows_walked,off.rows_walked - on.rows_walked,
					(unsigned long long)on.resume_base,on.checkpoint_seen);
			}
		}
	}
	assert(cudaStreamDestroy((cudaStream_t)test_state.execution_stream) == cudaSuccess);
	printf("qwen38 prefix-cache gate PASS\n");
	return(0);
}
