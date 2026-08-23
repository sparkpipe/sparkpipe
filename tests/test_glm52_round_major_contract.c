/* Round-major row-order collapse gate (host, no GPU).
 *
 * The glm52 resident stage used to carry the round-major wave validation
 * twice inside one driver: SparkGlm52ValidateRoundMajor +
 * SparkGlm52RoundMajorWaveRows (module) and SparkGlm52ServingValidateRowOrder
 * (serving adapter). Both now delegate to the shared linear row-layout
 * validator (sparkpipe/spark_row_layout.h) - the same helper the dsv4
 * module/adapter/stage-runner are pinned to - through one slot->lane ordinal
 * accessor.
 *
 * White-box on purpose: the harness includes the module translation unit so
 * the migrated statics run directly against verbatim copies of the
 * pre-collapse algorithms kept below as oracles. The sweep proves the
 * migration is accept/reject-identical (same SparkStatus, not just same
 * OK/not-OK) over an exhaustive small-batch space plus a seeded randomized
 * large-batch space, that wave sizing agrees on every suffix of every
 * accepted batch, and that the module's CAPACITY_EXCEEDED/DUPLICATE status
 * precedence survived the rewrite.
 *
 * Adapter-side agreement is asserted exactly on the input space reachable
 * through SparkModelServingAdapterValidateRuntimeSubmission (row_count >=
 * active_sequence_count, every active lane covered, row_count >= 1,
 * work_kind PREFILL/DECODE only - RELEASE fails the adapter capability gate
 * before row order runs). Off that space the shared validator is strictly
 * stricter; the only allowed divergence there is old-accept/new-reject with
 * partial lane coverage or a short prefill, and representative vectors are
 * pinned below. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_glm52_resident_decode_stage_module.c"

/* ---------------------------------------------------------------------------
 * Oracles: the pre-collapse implementations, copied verbatim (only renamed).
 * They live solely in this gate as the pinned reference behavior.
 * ------------------------------------------------------------------------- */

typedef struct OracleModuleState
{
	uint32_t resident_sequence_capacity;
} OracleModuleState;

static SparkStatus OracleValidateRoundMajor(
	const OracleModuleState *state,
	const SparkGlm52ResidentDecodeStageBatchView *batch)
{
	uint32_t counts[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint32_t lane,row,index,maximum,wave;
	if ( batch->row_count < batch->active_sequence_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (lane=0u; lane<batch->active_sequence_count; lane++)
	{
		if ( batch->row_resident_slots[lane] >= state->resident_sequence_capacity )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		for (index=0u; index<lane; index++)
			if ( batch->row_resident_slots[index] == batch->row_resident_slots[lane] )
				return(SPARK_STATUS_DUPLICATE);
	}
	maximum = 0u;
	for (row=0u; row<batch->row_count; row++)
	{
		for (lane=0u; lane<batch->active_sequence_count && batch->row_resident_slots[lane]!=batch->row_resident_slots[row]; lane++)
			;
		if ( lane == batch->active_sequence_count )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		counts[lane]++;
		if ( counts[lane] > maximum )
			maximum = counts[lane];
	}
	row = 0u;
	for (wave=0u; wave<maximum; wave++)
		for (lane=0u; lane<batch->active_sequence_count; lane++)
			if ( counts[lane] > wave && (row >= batch->row_count || batch->row_resident_slots[row++] != batch->row_resident_slots[lane]) )
				return(SPARK_STATUS_INVALID_ARGUMENT);
	return(row == batch->row_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static uint32_t OracleWaveRows(
	const SparkGlm52ResidentDecodeStageBatchView *batch,
	uint32_t first_row)
{
	uint32_t lane,current,count,next;
	if ( batch == 0 || first_row >= batch->row_count || batch->active_sequence_count == 0u )
		return(0u);
	current = batch->active_sequence_count;
	for (lane=0u; lane<batch->active_sequence_count; lane++)
		if ( batch->row_resident_slots[lane] == batch->row_resident_slots[first_row] )
			current = lane;
	if ( current == batch->active_sequence_count )
		return(0u);
	count = 1u;
	while ( first_row + count < batch->row_count )
	{
		next = batch->active_sequence_count;
		for (lane=0u; lane<batch->active_sequence_count; lane++)
			if ( batch->row_resident_slots[lane] == batch->row_resident_slots[first_row + count] )
				next = lane;
		if ( next == batch->active_sequence_count || next <= current )
			break;
		current = next;
		count++;
	}
	return(count);
}

typedef struct OracleSubmission
{
	uint32_t work_kind;
	uint32_t active_sequence_count;
	uint32_t row_count;
	const uint32_t *row_lane_indices;
	const uint64_t *row_positions;
} OracleSubmission;

typedef struct OracleServingState
{
	uint32_t max_sequence_positions;
} OracleServingState;

#define ORACLE_WORK_KIND_PREFILL 0u
#define ORACLE_WORK_KIND_DECODE 1u

static SparkStatus OracleRowOrder(
	const OracleServingState *state,
	const OracleSubmission *submission)
{
	uint8_t seen[16] = {0u};
	uint64_t last_position[16] = {0u};
	uint32_t lane,row,wave,maximum;
	uint32_t counts[16] = {0u};
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_positions[row] >= state->max_sequence_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( seen[lane] != 0u && submission->row_positions[row] != last_position[lane] + 1u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		seen[lane] = 1u;
		last_position[lane] = submission->row_positions[row];
		counts[lane]++;
	}
	if ( submission->work_kind == ORACLE_WORK_KIND_DECODE )
		return(submission->row_count == submission->active_sequence_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	maximum = 0u;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		if ( counts[lane] > maximum )
			maximum = counts[lane];
	row = 0u;
	for (wave=0u; wave<maximum; wave++)
		for (lane=0u; lane<submission->active_sequence_count; lane++)
			if ( counts[lane] > wave && (row >= submission->row_count || submission->row_lane_indices[row++] != lane) )
				return(SPARK_STATUS_INVALID_ARGUMENT);
	return(row == submission->row_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

/* Mirror of the migrated SparkGlm52ServingValidateRowOrder tail: the
 * position/continuity pass stays adapter-local, the ordering core delegates
 * to the real shared header inline (SparkRowLayoutDenseLaneOrdinal +
 * SparkRowLayoutValidateRoundMajor), exactly as the adapter source does. */
static SparkStatus TestAdapterRowOrder(
	const OracleServingState *state,
	const OracleSubmission *submission)
{
	uint8_t seen[16] = {0u};
	uint64_t last_position[16] = {0u};
	uint32_t occurrence_counts[16];
	uint32_t wave_last_rows[16];
	SparkRowLayoutDenseLaneContext dense;
	uint32_t lane,row;
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_positions[row] >= state->max_sequence_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( seen[lane] != 0u && submission->row_positions[row] != last_position[lane] + 1u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		seen[lane] = 1u;
		last_position[lane] = submission->row_positions[row];
	}
	if ( submission->work_kind == ORACLE_WORK_KIND_DECODE )
		return(submission->row_count == submission->active_sequence_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	dense.lane_count = submission->active_sequence_count;
	return(SparkRowLayoutValidateRoundMajor(submission->row_count,submission->active_sequence_count,submission->row_lane_indices,SparkRowLayoutDenseLaneOrdinal,&dense,occurrence_counts,wave_last_rows));
}

/* ---------------------------------------------------------------------------
 * Harness state and helpers.
 * ------------------------------------------------------------------------- */

static SparkGlm52ModuleState TestModuleState;
static uint64_t TestRandomState = 0x9e3779b97f4a7c15ull;

static uint64_t TestNextRandom(void)
{
	TestRandomState ^= TestRandomState << 13u;
	TestRandomState ^= TestRandomState >> 7u;
	TestRandomState ^= TestRandomState << 17u;
	return(TestRandomState);
}

struct TestCounters
{
	uint64_t module_vectors;
	uint64_t wave_probes;
	uint64_t adapter_vectors;
	uint64_t adapter_reachable_vectors;
	uint64_t adapter_off_space_divergences;
};

static void TestFillBatchView(
	SparkGlm52ResidentDecodeStageBatchView *batch,
	const uint32_t *slots,
	uint32_t row_count,
	uint32_t active_sequence_count)
{
	memset(batch,0,sizeof(*batch));
	batch->abi_version = SPARK_GLM52_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION;
	batch->descriptor_bytes = sizeof(*batch);
	batch->row_count = row_count;
	batch->active_sequence_count = active_sequence_count;
	batch->token_ids = 0;
	batch->row_resident_slots = slots;
	batch->row_positions = 0;
	batch->row_sequence_ids = 0;
}

static int TestLaneCoverageFull(
	const uint32_t *lanes,
	uint32_t row_count,
	uint32_t active_sequence_count)
{
	uint32_t cover[16];
	uint32_t lane,row;
	if ( active_sequence_count == 0u || row_count < active_sequence_count || row_count == 0u )
		return(0);
	memset(cover,0,sizeof(cover));
	for (row=0u; row<row_count; row++)
	{
		if ( lanes[row] >= active_sequence_count )
			return(0);
		cover[lanes[row]] = 1u;
	}
	for (lane=0u; lane<active_sequence_count; lane++)
		if ( cover[lane] == 0u )
			return(0);
	return(1);
}

int main(void)
{
	struct TestCounters counters;
	SparkGlm52ResidentDecodeStageBatchView batch;
	OracleModuleState oracle_state;
	uint32_t slots[8];
	uint32_t lanes[8];
	uint64_t positions[8];
	uint32_t alphabet,capacities[2],digit,value;
	uint32_t active,row_count,capacity,sequence,row;

	memset(&counters,0,sizeof(counters));
	memset(&TestModuleState,0,sizeof(TestModuleState));

	/* Exhaustive module sweep: every slot sequence over value alphabets of
	 * size 2..8 (each holds every prefix lane slot plus duplicate/unmapped
	 * values), row counts 0..6, two capacity regimes. Old and new must agree
	 * on the exact SparkStatus, always. */
	capacities[0] = 100u;
	capacities[1] = 103u; /* values 103+ sit beyond capacity */
	for (alphabet=1u; alphabet<=4u; alphabet++)
	{
		uint32_t base = alphabet * 2u;
		for (row_count=0u; row_count<=6u; row_count++)
		{
			for (sequence=0u; ; sequence++)
			{
				value = sequence;
				for (digit=0u; digit<row_count; digit++)
				{
					slots[digit] = 100u + value % base;
					value /= base;
				}
				if ( value != 0u )
					break;
				TestFillBatchView(&batch,slots,row_count,alphabet);
				for (capacity=0u; capacity<2u; capacity++)
				{
					SparkStatus oracle_status,live_status;
					oracle_state.resident_sequence_capacity = capacities[capacity];
					TestModuleState.resident_sequence_capacity = capacities[capacity];
					oracle_status = OracleValidateRoundMajor(&oracle_state,&batch);
					live_status = SparkGlm52ValidateRoundMajor(&TestModuleState,&batch);
					assert(oracle_status == live_status);
					counters.module_vectors++;
					if ( live_status == SPARK_STATUS_OK )
					{
						for (row=0u; row<row_count; row++)
						{
							assert(OracleWaveRows(&batch,row) == SparkGlm52RoundMajorWaveRows(&batch,row));
							counters.wave_probes++;
						}
						assert(SparkGlm52RoundMajorWaveRows(&batch,row_count) == 0u);
						assert(SparkGlm52RoundMajorWaveRows(&batch,row_count + 100u) == 0u);
					}
				}
			}
		}
	}

	/* Seeded randomized large-batch module sweep: up to 512 rows over slots
	 * 0..19 with capacity 8, active counts 1..10. */
	TestModuleState.resident_sequence_capacity = 8u;
	oracle_state.resident_sequence_capacity = 8u;
	for (sequence=0u; sequence<20000u; sequence++)
	{
		SparkStatus oracle_status,live_status;
		row_count = (uint32_t)(TestNextRandom() % 512u) + 1u;
		active = (uint32_t)(TestNextRandom() % 10u) + 1u;
		for (row=0u; row<row_count; row++)
			slots[row] = (uint32_t)(TestNextRandom() % 20u);
		TestFillBatchView(&batch,slots,row_count,active);
		oracle_status = OracleValidateRoundMajor(&oracle_state,&batch);
		live_status = SparkGlm52ValidateRoundMajor(&TestModuleState,&batch);
		assert(oracle_status == live_status);
		counters.module_vectors++;
		if ( live_status == SPARK_STATUS_OK )
		{
			for (row=0u; row<row_count; row++)
			{
				assert(OracleWaveRows(&batch,row) == SparkGlm52RoundMajorWaveRows(&batch,row));
				counters.wave_probes++;
			}
		}
	}

	/* Exhaustive adapter sweep over lane-id sequences: PREFILL and DECODE,
	 * active counts 1..4, row counts 0..7. Inside the reachable input space
	 * agreement must be exact; outside it the only allowed divergence is the
	 * documented fail-closed tightening on partially-covered lanes. */
	for (active=1u; active<=4u; active++)
	{
		for (row_count=0u; row_count<=7u; row_count++)
		{
			uint64_t limit = 1u;
			static const uint64_t kPositions[8] = {0u,2u,1u,3u,2u,5u,6u,7u};
			for (digit=0u; digit<row_count; digit++)
				limit *= 4u;
			for (sequence=0u; (uint64_t)sequence<limit; sequence++)
			{
				OracleServingState serving_state;
				OracleSubmission submission;
				SparkStatus oracle_status,test_status;
				value = sequence;
				for (digit=0u; digit<row_count; digit++)
				{
					lanes[digit] = value % 4u;
					positions[digit] = kPositions[digit];
					value /= 4u;
				}
				serving_state.max_sequence_positions = 8u;
				submission.work_kind = (sequence & 1u) != 0u ? ORACLE_WORK_KIND_DECODE : ORACLE_WORK_KIND_PREFILL;
				submission.active_sequence_count = active;
				submission.row_count = row_count;
				submission.row_lane_indices = lanes;
				submission.row_positions = positions;
				oracle_status = OracleRowOrder(&serving_state,&submission);
				test_status = TestAdapterRowOrder(&serving_state,&submission);
				counters.adapter_vectors++;
				if ( oracle_status == test_status )
					continue;
				assert(test_status == SPARK_STATUS_INVALID_ARGUMENT);
				assert(oracle_status == SPARK_STATUS_OK);
				if ( TestLaneCoverageFull(lanes,row_count,active) )
				{
					fprintf(stderr,"UNEXPECTED adapter divergence: rows=%u active=%u lanes=[%u,%u,%u,%u,%u,%u,%u,%u]\n",
						row_count,active,lanes[0],lanes[1],lanes[2],lanes[3],lanes[4],lanes[5],lanes[6],lanes[7]);
					return(1);
				}
				counters.adapter_off_space_divergences++;
			}
		}
	}

	/* Reachable-space equality: seeded random submissions satisfying every
	 * upstream invariant (full lane coverage, row_count >= active >= 1,
	 * positions within bounds) must be judged identically by oracle and
	 * migrated code. */
	for (sequence=0u; sequence<50000u; sequence++)
	{
		OracleServingState serving_state;
		OracleSubmission submission;
		SparkStatus oracle_status,test_status;
		uint64_t cursor[4];
		active = (uint32_t)(TestNextRandom() % 4u) + 1u;
		row_count = active + (uint32_t)(TestNextRandom() % (uint64_t)(9u - active));
		for (digit=0u; digit<active; digit++)
		{
			lanes[digit] = digit;
			cursor[digit] = (int64_t)(TestNextRandom() % 3u);
		}
		for (row=active; row<row_count; row++)
			lanes[row] = (uint32_t)(TestNextRandom() % active);
		for (row=0u; row<row_count; row++)
			positions[row] = cursor[lanes[row]]++;
		serving_state.max_sequence_positions = 64u;
		submission.work_kind = (uint32_t)(TestNextRandom() % 2u) != 0u ? ORACLE_WORK_KIND_DECODE : ORACLE_WORK_KIND_PREFILL;
		submission.active_sequence_count = active;
		submission.row_count = row_count;
		submission.row_lane_indices = lanes;
		submission.row_positions = positions;
		assert(TestLaneCoverageFull(lanes,row_count,active) != 0);
		oracle_status = OracleRowOrder(&serving_state,&submission);
		test_status = TestAdapterRowOrder(&serving_state,&submission);
		assert(oracle_status == test_status);
		counters.adapter_reachable_vectors++;
	}

	/* Valid-batch wave-sizing parity: synthesize round-major batches
	 * directly (per-lane row counts, lanes emitted round-major, slots a
	 * random distinct permutation) so wave sizing is compared on every
	 * suffix of thousands of accepted batches - this drives TP chain
	 * chunking at runtime. */
	for (sequence=0u; sequence<20000u; sequence++)
	{
		uint32_t counts[10];
		uint32_t slot_of_lane[10];
		uint32_t used[20];
		uint32_t emitted,wave,lane;
		active = (uint32_t)(TestNextRandom() % 8u) + 1u;
		row_count = active + (uint32_t)(TestNextRandom() % (uint64_t)(40u - active));
		TestModuleState.resident_sequence_capacity = 32u; /* slots 0..19 fit */
		memset(used,0,sizeof(used));
		for (lane=0u; lane<active; lane++)
		{
			do
				slot_of_lane[lane] = (uint32_t)(TestNextRandom() % 20u);
			while ( used[slot_of_lane[lane]] != 0u );
			used[slot_of_lane[lane]] = 1u;
			counts[lane] = 1u + (uint32_t)(TestNextRandom() % 6u);
		}
		emitted = 0u;
		for (wave=0u; wave<6u; wave++)
			for (lane=0u; lane<active; lane++)
				if ( counts[lane] > wave && emitted < row_count )
					slots[emitted++] = slot_of_lane[lane];
		TestFillBatchView(&batch,slots,emitted,active);
		assert(SparkGlm52ValidateRoundMajor(&TestModuleState,&batch) == SPARK_STATUS_OK);
		assert(OracleWaveRows(&batch,0u) != 0u);
		for (row=0u; row<emitted; row++)
		{
			assert(OracleWaveRows(&batch,row) == SparkGlm52RoundMajorWaveRows(&batch,row));
			counters.wave_probes++;
		}
	}

	/* Pinned status precedence: capacity beats duplicate beats order. */
	{
		static const uint32_t kCapacitySlots[3] = {105u,101u,101u};
		static const uint32_t kDuplicateSlots[3] = {101u,101u,102u};
		static const uint32_t kUnorderedTailSlots[4] = {101u,102u,102u,101u};
		static const uint32_t kValidSlots[4] = {101u,102u,101u,102u};
		TestModuleState.resident_sequence_capacity = 104u;
		oracle_state.resident_sequence_capacity = 104u;
		TestFillBatchView(&batch,kCapacitySlots,3u,3u);
		assert(SparkGlm52ValidateRoundMajor(&TestModuleState,&batch) == SPARK_STATUS_CAPACITY_EXCEEDED);
		assert(OracleValidateRoundMajor(&oracle_state,&batch) == SPARK_STATUS_CAPACITY_EXCEEDED);
		TestFillBatchView(&batch,kDuplicateSlots,3u,3u);
		assert(SparkGlm52ValidateRoundMajor(&TestModuleState,&batch) == SPARK_STATUS_DUPLICATE);
		assert(OracleValidateRoundMajor(&oracle_state,&batch) == SPARK_STATUS_DUPLICATE);
		TestModuleState.resident_sequence_capacity = 200u;
		oracle_state.resident_sequence_capacity = 200u;
		TestFillBatchView(&batch,kUnorderedTailSlots,4u,2u);
		assert(SparkGlm52ValidateRoundMajor(&TestModuleState,&batch) == SPARK_STATUS_INVALID_ARGUMENT);
		TestFillBatchView(&batch,kValidSlots,4u,2u);
		assert(SparkGlm52ValidateRoundMajor(&TestModuleState,&batch) == SPARK_STATUS_OK);
		assert(SparkGlm52RoundMajorWaveRows(&batch,0u) == 2u);
		assert(SparkGlm52RoundMajorWaveRows(&batch,1u) == 1u);
		assert(SparkGlm52RoundMajorWaveRows(&batch,2u) == 2u);
		assert(SparkGlm52RoundMajorWaveRows(&batch,3u) == 1u);
		assert(SparkGlm52RoundMajorWaveRows(&batch,4u) == 0u);
		assert(SparkGlm52RoundMajorWaveRows(0,0u) == 0u);

		/* Documented adapter tightenings (inputs upstream can never deliver):
		 * short prefills and partially-covered lanes now fail closed. */
		{
			static const uint32_t kPartialLanes[3] = {1u,2u,1u};
			/* Positions keep per-lane continuity so only the coverage
			 * tightening can fire. */
			static const uint64_t kAnyPositions[3] = {5u,0u,6u};
			OracleServingState serving_state;
			OracleSubmission submission;
			serving_state.max_sequence_positions = 8u;
			submission.work_kind = ORACLE_WORK_KIND_PREFILL;
			submission.active_sequence_count = 3u;
			submission.row_count = 3u;
			submission.row_lane_indices = kPartialLanes;
			submission.row_positions = kAnyPositions;
			assert(TestAdapterRowOrder(&serving_state,&submission) == SPARK_STATUS_INVALID_ARGUMENT);
			assert(OracleRowOrder(&serving_state,&submission) == SPARK_STATUS_OK);
		}
	}

	printf("PASS GLM52 round-major collapse: %llu module vectors exact-status parity, %llu wave probes parity, %llu adapter vectors (%llu off-space tightenings, all classified), %llu reachable-space vectors exact\n",
		(unsigned long long)counters.module_vectors,
		(unsigned long long)counters.wave_probes,
		(unsigned long long)counters.adapter_vectors,
		(unsigned long long)counters.adapter_off_space_divergences,
		(unsigned long long)counters.adapter_reachable_vectors);
	return(0);
}

/* Link-only stubs, verbatim from test_glm52_pp7_configure.c: the dry proof
 * never reaches the CUDA launchers, but the whole module translation unit is
 * linked, so their symbols must resolve. */
#include <cuda_runtime.h>
cudaError_t SparkStageLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	(void)stream;(void)destination_bf16;(void)source_bf16;(void)row_count;(void)width;
	return(1); /* cudaErrorNotSupported under the real runtime */
}
cudaError_t SparkStageLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	(void)stream;(void)destination;(void)source;(void)element_count;
	return(1); /* cudaErrorNotSupported under the real runtime */
}
int32_t SparkGlm52ConfigureCudaModule(uint32_t *multiprocessor_count)
{
	(void)multiprocessor_count;
	return(-1);
}
/* The module TU now carries the DFlash2 speculator; its backend entry
 * points must resolve for the link (never reached on this path). */
#include "glm52_dspark_backend_stubs.h"
int32_t SparkGlm52LaunchCudaWave(const SparkGlm52CudaWave *wave) { (void)wave; return(-1); }
int32_t SparkGlm52LaunchCudaWaveBegin(const SparkGlm52CudaWave *wave) { (void)wave; return(-1); }
int32_t SparkGlm52LaunchCudaLayerAttention(const SparkGlm52CudaWave *wave,uint32_t local_layer) { (void)wave;(void)local_layer; return(-1); }
int32_t SparkGlm52LaunchCudaLayerMlp(const SparkGlm52CudaWave *wave,uint32_t local_layer) { (void)wave;(void)local_layer; return(-1); }
int32_t SparkGlm52LaunchCudaWaveHead(const SparkGlm52CudaWave *wave) { (void)wave; return(-1); }
