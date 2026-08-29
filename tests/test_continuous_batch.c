/* The continuous-batching step-boundary contract, proven on a host with no
 * engine, no GPU, and no model tokens (the module is engine-neutral; these
 * proofs pin it the way tests/test_jit_kv_slice.c pins the pager).
 *
 * The laws under proof (sparkpipe/spark_continuous_batch.h):
 *   L1 INTERLEAVING - a mid-flight join never disturbs resident lanes:
 *      their emission streams are bit-exact against a run without the
 *      join, and the row budget holds at EVERY step (asserted after every
 *      single state change, the slice test's discipline).
 *   L2 QUEUE-NOT-WEDGE - every refusal is named and kept; the queue stays
 *      healthy; the offer admits at the first boundary whose arithmetic
 *      fits; oversize offers never block anyone.
 *   L3 SLOT RECLAIM - a lane finished at step s (or retired engine-side)
 *      frees its slot at the very next boundary, and the boundary's
 *      admitted offer takes the freed capacity.
 *   L4 THE POLICY - smallest-first with ties in arrival order, the
 *      starvation escape oldest-aged-first, and the reservation leg: an
 *      aged offer's refusal holds younger fitting offers out of the
 *      boundary (they cannot starve the aged offer back into the queue).
 */
#include "sparkpipe/spark_continuous_batch.h"

#include <stdio.h>
#include <string.h>

static int32_t failures = 0;

static void expect(int condition, const char *label)
{
	printf(condition ? "  ok   %s\n" : "  FAIL %s\n", label);
	if ( !condition )
		++failures;
}

#define PROOF_MAX_LANES 8u
#define PROOF_MAX_QUEUE 32u
#define PROOF_MAX_STEPS 32u

typedef struct ProofLog
{
	uint32_t rows_spent[PROOF_MAX_STEPS];
	uint32_t finished_count[PROOF_MAX_STEPS];
	uint32_t event_count[PROOF_MAX_STEPS];
	struct
	{
		uint64_t request_id;
		uint32_t slot;
		uint32_t token_index;
	} events[PROOF_MAX_STEPS][PROOF_MAX_LANES];
	uint32_t steps;
}
ProofLog;

static void ProofLogReset(ProofLog *log)
{
	memset(log,0,sizeof(*log));
}

/* One round: step (advance resident lanes) then boundary (reclaim +
 * drain). The row-budget law is asserted after the step, every time. */
static void Round(
	SparkContinuousBatch *controller,
	ProofLog *log,
	uint32_t max_input_rows,
	uint64_t *released,
	uint32_t released_capacity,
	uint32_t *released_count)
{
	SparkContinuousBatchStepEvent events[PROOF_MAX_LANES];
	SparkContinuousBatchStepReport report;
	if ( SparkContinuousBatchStep(controller,events,PROOF_MAX_LANES,&report) !=
		SPARK_STATUS_OK )
	{
		expect(0,"step status");
		return;
	}
	expect(report.rows_spent <= max_input_rows,
		"the row budget law held at this step");
	if ( log->steps < PROOF_MAX_STEPS )
	{
		log->rows_spent[log->steps] = report.rows_spent;
		log->finished_count[log->steps] = report.finished_count;
		log->event_count[log->steps] = report.event_count;
		memcpy(log->events[log->steps],events,
			report.event_count * sizeof(events[0]));
	}
	log->steps += 1u;
	if ( SparkContinuousBatchBoundary(controller,released,released_capacity,
		released_count) != SPARK_STATUS_OK )
		expect(0,"boundary status");
}

/* The resident lane's emission stream, flattened: (token_index, step)
 * pairs in emission order - the exact bytes the L1 comparison runs over. */
static uint32_t FlattenEmissionStream(
	const ProofLog *log,
	uint64_t request_id,
	uint64_t *out,
	uint32_t out_capacity)
{
	uint32_t step,index,written;
	written = 0u;
	for ( step = 0u; step < log->steps && step < PROOF_MAX_STEPS; ++step )
		for ( index = 0u; index < log->event_count[step]; ++index )
			if ( log->events[step][index].request_id == request_id &&
				written + 2u <= out_capacity )
			{
				out[written++] = log->events[step][index].token_index;
				out[written++] = step;
			}
	return(written);
}

static SparkContinuousBatch *MakeController(
	uint32_t max_input_rows,
	uint32_t max_active_lanes,
	uint32_t starvation_bound,
	uint32_t queue_capacity)
{
	SparkContinuousBatchConfiguration configuration;
	SparkContinuousBatch *controller;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_CONTINUOUS_BATCH_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_CONTINUOUS_BATCH_CONFIGURATION_BYTES;
	configuration.max_input_rows = max_input_rows;
	configuration.max_active_lanes = max_active_lanes;
	configuration.starvation_bound = starvation_bound;
	configuration.queue_capacity = queue_capacity;
	configuration.lane_capacity = max_active_lanes;
	if ( SparkContinuousBatchInitialize(&configuration,&controller) !=
		SPARK_STATUS_OK )
	{
		expect(0,"controller initialize");
		return(0);
	}
	return(controller);
}

static uint32_t Offer(
	SparkContinuousBatch *controller,
	uint64_t request_id,
	uint32_t prompt_rows,
	uint32_t budget,
	SparkContinuousBatchDecision *decision)
{
	SparkContinuousBatchRequest request;
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_CONTINUOUS_BATCH_ABI_VERSION;
	request.descriptor_bytes = SPARK_CONTINUOUS_BATCH_REQUEST_BYTES;
	request.request_id = request_id;
	request.prompt_row_count = prompt_rows;
	request.output_token_budget = budget;
	return(SparkContinuousBatchOffer(controller,&request,decision));
}

static void TestPolicyPick(void)
{
	uint32_t rows[6];
	uint64_t boundaries[6];
	uint8_t excluded[6];
	printf("policy pick (smallest-first, aged-first, exclusion)\n");
	/* L4a: empty and all-excluded queues answer -1. */
	expect(SparkContinuousBatchPolicyPick(rows,boundaries,0,0u,4u,4u) == -1,
		"empty queue picks nothing");
	memset(excluded,1u,sizeof(excluded));
	rows[0] = 1u; boundaries[0] = 0u;
	expect(SparkContinuousBatchPolicyPick(rows,boundaries,excluded,1u,4u,4u) == -1,
		"all-excluded queue picks nothing");
	/* L4b: smallest-first, ties in arrival order. */
	rows[0] = 5u; rows[1] = 3u; rows[2] = 8u; rows[3] = 3u;
	boundaries[0] = 7u; boundaries[1] = 8u; boundaries[2] = 9u; boundaries[3] = 8u;
	expect(SparkContinuousBatchPolicyPick(rows,boundaries,0,4u,10u,0u) == 1,
		"smallest rows first");
	expect(SparkContinuousBatchPolicyPick(rows,boundaries,0,4u,10u,4u) == 1,
		"no aged offer leaves smallest-first in charge");
	/* L4c: the starvation escape - the oldest aged offer jumps everyone;
	 * among aged offers the older enqueue boundary wins. */
	boundaries[0] = 2u; boundaries[1] = 0u; boundaries[2] = 1u; boundaries[3] = 0u;
	rows[0] = 1u; rows[1] = 9u; rows[2] = 1u; rows[3] = 9u;
	expect(SparkContinuousBatchPolicyPick(rows,boundaries,0,4u,4u,3u) == 1,
		"the oldest aged offer is the boundary's priority pick");
	expect(SparkContinuousBatchPolicyPick(rows,boundaries,0,4u,4u,1u) == 1,
		"two aged offers: the older one first");
	excluded[0] = 0u; excluded[1] = 1u; excluded[2] = 0u; excluded[3] = 0u;
	expect(SparkContinuousBatchPolicyPick(rows,boundaries,excluded,4u,4u,1u) == 3,
		"excluded entries leave the pick");
}

static void TestInterleavingBitExact(void)
{
	SparkContinuousBatch *baseline,*joined;
	SparkContinuousBatchDecision decision;
	SparkContinuousBatchStatistics stats;
	ProofLog baseline_log,joined_log;
	uint64_t released[PROOF_MAX_QUEUE];
	uint64_t stream_base[64],stream_join[64];
	uint32_t released_count,words,step;
	printf("L1 interleaving: a mid-flight join is invisible to residents\n");
	baseline = MakeController(8u,8u,4u,16u);
	joined = MakeController(8u,8u,4u,16u);
	ProofLogReset(&baseline_log);
	ProofLogReset(&joined_log);
	/* Both runs: r101 (rows 2, 3 tokens) and r102 (rows 4, 3 tokens) are
	 * resident from the start. */
	expect(Offer(baseline,101u,2u,3u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"baseline r101 joins the idle deployment");
	expect(Offer(baseline,102u,4u,3u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"baseline r102 joins beside r101");
	expect(Offer(joined,101u,2u,3u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"joined run r101 joins identically");
	expect(Offer(joined,102u,4u,3u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"joined run r102 joins identically");
	/* Round 1 prefills both lanes; round 2 emits the first token. */
	Round(baseline,&baseline_log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(released_count == 0u,"baseline round 1 releases nothing");
	Round(baseline,&baseline_log,8u,released,PROOF_MAX_QUEUE,&released_count);
	/* The joined run replays the same two rounds, and then r103 (rows 3,
	 * 2 tokens) joins MID-FLIGHT, between step 2 and step 3. */
	Round(joined,&joined_log,8u,released,PROOF_MAX_QUEUE,&released_count);
	Round(joined,&joined_log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(joined_log.steps == 2u && baseline_log.steps == 2u,
		"both runs are at step 2 when the join arrives");
	expect(Offer(joined,103u,3u,2u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_NONE,
		"the mid-flight join is granted (2 resident + 3 rows <= 8)");
	/* Drain both to completion (rounds 3..8). */
	while ( baseline_log.steps < 8u )
		Round(baseline,&baseline_log,8u,released,PROOF_MAX_QUEUE,&released_count);
	while ( joined_log.steps < 8u )
		Round(joined,&joined_log,8u,released,PROOF_MAX_QUEUE,&released_count);
	/* The bit-exact receipt: r101's and r102's emission streams are the
	 * SAME BYTES in both runs - token for token, step for step. */
	words = FlattenEmissionStream(&baseline_log,101u,stream_base,64u);
	expect(words == 6u,"baseline r101 emitted three tokens");
	expect(FlattenEmissionStream(&joined_log,101u,stream_join,64u) == words,
		"joined r101 emitted three tokens too");
	expect(memcmp(stream_base,stream_join,words * sizeof(*stream_base)) == 0,
		"r101's stream is bit-exact across the join");
	words = FlattenEmissionStream(&baseline_log,102u,stream_base,64u);
	expect(words == 6u,"baseline r102 emitted three tokens");
	expect(FlattenEmissionStream(&joined_log,102u,stream_join,64u) == words,
		"joined r102 emitted three tokens too");
	expect(memcmp(stream_base,stream_join,words * sizeof(*stream_base)) == 0,
		"r102's stream is bit-exact across the join");
	/* Both residents finished on the SAME round in both runs (their last
	 * two tokens at round 4); the joiner emits alongside, never instead. */
	expect(baseline_log.event_count[3] == 2u,
		"baseline residents finish together at round 4");
	expect(joined_log.event_count[3] == 3u,
		"joined round 4 carries both final resident tokens plus the joiner's");
	/* The joiner emits only in the joined run, two tokens of its own. */
	expect(FlattenEmissionStream(&baseline_log,103u,stream_base,64u) == 0u,
		"the joiner is absent from the baseline");
	words = FlattenEmissionStream(&joined_log,103u,stream_join,64u);
	expect(words == 4u,"the joiner emits its own two tokens after joining");
	/* The row budget held at every step of both runs. */
	for ( step = 0u; step < 8u; ++step )
	{
		expect(baseline_log.rows_spent[step] <= 8u,"baseline row law");
		expect(joined_log.rows_spent[step] <= 8u,"joined row law");
	}
	/* Everything drained; the joined run carried all three requests. */
	expect(SparkContinuousBatchGetStatistics(baseline,&stats) == SPARK_STATUS_OK &&
		stats.resident_lane_count == 0u && stats.queued_offer_count == 0u,
		"baseline drained clean");
	expect(SparkContinuousBatchGetStatistics(joined,&stats) == SPARK_STATUS_OK &&
		stats.resident_lane_count == 0u && stats.queued_offer_count == 0u &&
		stats.admission_accepted == 3u,
		"joined run accepted exactly the three offers");
	SparkContinuousBatchDestroy(baseline);
	SparkContinuousBatchDestroy(joined);
}

static void TestRefusalQueueNotWedge(void)
{
	SparkContinuousBatch *controller;
	SparkContinuousBatchDecision decision;
	SparkContinuousBatchStatistics stats;
	SparkContinuousBatchLaneView lane;
	ProofLog log;
	uint64_t released[PROOF_MAX_QUEUE];
	uint32_t released_count;
	printf("L2/L3 refusals queue, never wedge; slots reclaim next boundary\n");
	controller = MakeController(8u,2u,0u,8u);
	ProofLogReset(&log);
	expect(Offer(controller,201u,1u,2u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"r201 takes the first lane");
	expect(Offer(controller,202u,1u,2u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"r202 takes the second lane");
	expect(Offer(controller,203u,2u,2u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_QUEUED &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_LANES &&
		decision.queue_position == 0u,
		"r203 is refused by name (lanes full) and KEPT");
	/* Round 1: both prefills land; the boundary re-examines r203 and
	 * refuses it again - named, counted, queued, healthy. */
	Round(controller,&log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(released_count == 0u,"round 1 releases nothing (lanes still full)");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.admission_queued_lanes == 1u,
		"the lane refusal is counted at the round-1 boundary");
	/* Round 2: r201/r202 emit their first token; r203 still waits. */
	Round(controller,&log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(released_count == 0u,"round 2 releases nothing");
	/* Round 3: the last tokens; BOTH lanes finish at this step. */
	Round(controller,&log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(log.finished_count[2] == 2u,"both residents finish at step 3");
	expect(log.event_count[2] == 2u,"step 3 carries the final two tokens");
	/* L3: the VERY NEXT boundary reclaims both slots and admits r203. */
	expect(released_count == 1u && released[0] == 203u,
		"the finished lanes' slots free at the very next boundary");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.slot_reclaims == 2u && stats.admission_queued_lanes == 2u &&
		stats.resident_lane_count == 1u,
		"two reclaims, two named lane refusals, r203 resident");
	expect(SparkContinuousBatchGetLane(controller,0u,&lane) == SPARK_STATUS_OK &&
		lane.request_id == 203u && lane.phase == SPARK_CONTINUOUS_BATCH_LANE_PREFILL,
		"r203 took reclaimed slot 0, prefill ahead of it");
	/* r203 runs to terminal; the controller ends empty. */
	while ( log.steps < 8u )
		Round(controller,&log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.resident_lane_count == 0u && stats.queued_offer_count == 0u &&
		stats.slot_reclaims == 3u && stats.admission_accepted == 3u,
		"every offer admitted, every slot reclaimed - no wedge, no drop");
	SparkContinuousBatchDestroy(controller);
}

static void TestRowArithmeticAndChunking(void)
{
	SparkContinuousBatch *controller;
	SparkContinuousBatchDecision decision;
	SparkContinuousBatchStatistics stats;
	ProofLog log;
	uint64_t released[PROOF_MAX_QUEUE];
	uint32_t released_count,step;
	printf("C1 arithmetic: the row budget binds exactly, chunked prefill shares it\n");
	controller = MakeController(4u,4u,0u,8u);
	ProofLogReset(&log);
	expect(Offer(controller,301u,3u,2u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"r301 (3 rows) joins the empty deployment");
	expect(Offer(controller,302u,2u,2u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"r302 (2 rows) fits beside it (1+2 <= 4)");
	expect(Offer(controller,303u,2u,2u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"r303 (2 rows) still fits (2+2 <= 4)");
	expect(Offer(controller,304u,2u,2u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_QUEUED &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_ROWS,
		"r304 is refused by name (3 resident + 2 rows > 4) and KEPT");
	/* Round 1: the decode reservation is zero lanes, so prefill may use
	 * the whole budget: r301 takes 3, r302 takes the last 1 row, r303
	 * waits for its rows. The budget is EXACT, never over. */
	Round(controller,&log,4u,released,PROOF_MAX_QUEUE,&released_count);
	expect(log.rows_spent[0] == 4u,"round 1 spends exactly max_input_rows");
	expect(released_count == 0u,"nothing releases while the budget is full");
	/* r304's refusal re-arms each boundary while residents hold rows. */
	Round(controller,&log,4u,released,PROOF_MAX_QUEUE,&released_count);
	expect(log.rows_spent[1] == 4u,"round 2 spends exactly max_input_rows again");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.admission_queued_rows == 2u && stats.queued_offer_count == 1u,
		"r304 queued behind the row arithmetic at both boundaries");
	/* Drain: residents finish (2 tokens each), slots reclaim, r304's
	 * arithmetic finally fits (0 resident + 2 <= 4). */
	while ( (stats.resident_lane_count != 0u || stats.queued_offer_count != 0u) &&
		log.steps < PROOF_MAX_STEPS )
	{
		Round(controller,&log,4u,released,PROOF_MAX_QUEUE,&released_count);
		if ( SparkContinuousBatchGetStatistics(controller,&stats) != SPARK_STATUS_OK )
			break;
	}
	expect(stats.resident_lane_count == 0u && stats.queued_offer_count == 0u,
		"r304 admitted and drained once the residents freed their rows");
	expect(stats.slot_reclaims == 4u,"all four lanes reclaimed exactly once");
	for ( step = 0u; step < log.steps && step < PROOF_MAX_STEPS; ++step )
		expect(log.rows_spent[step] <= 4u,"the row law held at every step");
	SparkContinuousBatchDestroy(controller);
}

static void TestOversizeAndQueueFull(void)
{
	SparkContinuousBatch *controller;
	SparkContinuousBatchDecision decision;
	SparkContinuousBatchStatistics stats;
	printf("oversize is named and harmless; a full queue answers BUSY-style\n");
	controller = MakeController(4u,4u,0u,1u);
	expect(Offer(controller,901u,5u,1u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_QUEUED &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_OVERSIZE &&
		decision.queue_position == 0u,
		"an offer larger than max_input_rows is refused by name and kept");
	/* The oversize offer never competes: a small offer behind it still
	 * joins the idle deployment immediately. */
	expect(Offer(controller,902u,2u,2u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"the oversize entry does not block the queue (no head-of-line)");
	/* Queue capacity 1 is now used by the oversize entry: the next
	 * refusal names FULL and is NOT kept - the caller retries. */
	expect(Offer(controller,903u,4u,1u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_QUEUED &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_FULL,
		"a queue at capacity answers FULL by name");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.admission_oversize == 1u && stats.admission_queue_full == 1u &&
		stats.queued_offer_count == 1u,
		"the counters name both refusals; only the kept offer is queued");
	/* The caller's cancellation path: withdraw the oversize offer, the
	 * queue is healthy again, the retried offer queues by name. */
	expect(SparkContinuousBatchWithdraw(controller,901u) == SPARK_STATUS_OK,
		"the oversize offer is withdrawn");
	expect(Offer(controller,903u,4u,1u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_QUEUED &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_ROWS,
		"the retry is refused by the row arithmetic (r902 resident)");
	expect(SparkContinuousBatchWithdraw(controller,999u) == SPARK_STATUS_NOT_FOUND,
		"withdrawing an unknown id is NOT_FOUND");
	SparkContinuousBatchDestroy(controller);
}

static void TestStarvationBoundAndReservation(void)
{
	SparkContinuousBatch *controller;
	SparkContinuousBatchDecision decision;
	SparkContinuousBatchStatistics stats;
	SparkContinuousBatchLaneView lane;
	ProofLog log;
	uint64_t released[PROOF_MAX_QUEUE];
	uint32_t released_count,round,index;
	printf("L4 starvation bound: aging wins the boundary and holds it\n");
	controller = MakeController(8u,8u,2u,16u);
	ProofLogReset(&log);
	/* Five long residents (1 row each) hold 5 rows of the budget. */
	for ( index = 0u; index < 5u; ++index )
		expect(Offer(controller,400u + index,1u,4u,&decision) == SPARK_STATUS_OK &&
			decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
			"a long resident joins");
	/* S (4 rows) is refused by the row arithmetic: 5 + 4 > 8. */
	expect(Offer(controller,999u,4u,1u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_QUEUED &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_ROWS,
		"S queues behind the resident rows");
	/* Round 1: residents prefill; the boundary refuses S (not yet aged). */
	Round(controller,&log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(released_count == 0u,"round 1 releases nothing");
	/* t1 (1 row, would fit: 5 + 1 <= 8) arrives behind S. */
	expect(Offer(controller,500u,1u,2u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_QUEUED &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_AHEAD,
		"t1 queues behind S (arrivals never cut ahead)");
	/* Round 2: S has waited 2 boundaries - it is AGED. Its refusal must
	 * hold t1 out of the boundary (the reservation leg), even though t1
	 * fits. THE RESERVATION. */
	Round(controller,&log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(released_count == 0u,
		"the aged offer's refusal keeps the boundary closed to t1");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.queued_offer_count == 2u,
		"t1 is still queued - the reservation held");
	/* Rounds 3-5: the residents emit their remaining tokens and finish
	 * at step 5; S is refused (aged) at every boundary along the way. */
	for ( round = 0u; round < 3u; ++round )
		Round(controller,&log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(log.finished_count[4] == 5u,"the five residents finish at step 5");
	/* The next boundary reclaims the five slots and S - aged, priority
	 * pick - admits FIRST, ahead of the younger t1. */
	expect(released_count == 2u && released[0] == 999u && released[1] == 500u,
		"S admits at the first boundary that fits it, t1 follows");
	expect(SparkContinuousBatchGetLane(controller,0u,&lane) == SPARK_STATUS_OK &&
		lane.request_id == 999u && lane.admitted_via_aging == 1u,
		"S's lane carries the aging flag");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.starvation_jumps == 2u && stats.admission_queued_rows == 4u,
		"two aging wins (S and t1 both waited past the bound); four refusals");
	/* S and t1 run to terminal; the controller ends empty. */
	while ( (stats.resident_lane_count != 0u || stats.queued_offer_count != 0u) &&
		log.steps < PROOF_MAX_STEPS )
	{
		Round(controller,&log,8u,released,PROOF_MAX_QUEUE,&released_count);
		if ( SparkContinuousBatchGetStatistics(controller,&stats) != SPARK_STATUS_OK )
			break;
	}
	expect(stats.resident_lane_count == 0u && stats.queued_offer_count == 0u,
		"the aged run drains clean");
	SparkContinuousBatchDestroy(controller);
}

static void TestEngineRetirementSeam(void)
{
	SparkContinuousBatch *controller;
	SparkContinuousBatchDecision decision;
	SparkContinuousBatchStatistics stats;
	SparkContinuousBatchLaneView lane;
	ProofLog log;
	uint64_t released[PROOF_MAX_QUEUE];
	uint32_t released_count;
	printf("L3 seam path: an engine-side retirement frees the slot next boundary\n");
	controller = MakeController(8u,2u,0u,8u);
	ProofLogReset(&log);
	expect(Offer(controller,601u,1u,100u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"r601 resident");
	expect(Offer(controller,602u,1u,100u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"r602 resident");
	expect(Offer(controller,603u,1u,100u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_QUEUED &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_LANES,
		"r603 queues (lanes full)");
	/* Round 1: both residents prefill in. The engine now reports r601
	 * terminal (cancelled, in the real tool) - the SEAM call. */
	Round(controller,&log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(SparkContinuousBatchRetire(controller,601u) == SPARK_STATUS_OK,
		"the engine's terminal report retires r601");
	expect(SparkContinuousBatchRetire(controller,601u) == SPARK_STATUS_OK,
		"a repeated retirement stays healthy");
	expect(SparkContinuousBatchRetire(controller,603u) == SPARK_STATUS_NOT_FOUND,
		"retiring a QUEUED id is NOT_FOUND (withdraw instead)");
	/* The very next boundary reclaims r601's slot and admits r603 into
	 * it - while r602 is UNTOUCHED (L1). */
	Round(controller,&log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(released_count == 1u && released[0] == 603u,
		"the retired lane's slot frees at the very next boundary");
	expect(SparkContinuousBatchGetLane(controller,0u,&lane) == SPARK_STATUS_OK &&
		lane.request_id == 603u && lane.phase == SPARK_CONTINUOUS_BATCH_LANE_PREFILL,
		"r603 took the reclaimed slot 0");
	expect(SparkContinuousBatchGetLane(controller,1u,&lane) == SPARK_STATUS_OK &&
		lane.request_id == 602u && lane.phase == SPARK_CONTINUOUS_BATCH_LANE_DECODE,
		"r602 keeps its slot and phase across the seam reclaim");
	/* r602's stream is untouched by the retirement + join: it emits on
	 * its own cadence; r603's first emission follows its prefill. */
	Round(controller,&log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(log.event_count[2] == 1u && log.events[2][0].request_id == 602u,
		"r602 emitted on cadence at step 3");
	Round(controller,&log,8u,released,PROOF_MAX_QUEUE,&released_count);
	expect(log.event_count[3] == 2u,"both decode lanes emit at step 4");
	expect(log.events[3][0].request_id == 603u && log.events[3][1].request_id == 602u,
		"emission stays in slot order");
	expect(SparkContinuousBatchWithdraw(controller,603u) == SPARK_STATUS_NOT_FOUND,
		"r603 is no longer queued (it is resident)");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.slot_reclaims == 1u && stats.resident_lane_count == 2u,
		"exactly one reclaim, two residents");
	SparkContinuousBatchDestroy(controller);
}

static void TestValidation(void)
{
	SparkContinuousBatchConfiguration configuration;
	SparkContinuousBatch *controller;
	SparkContinuousBatchDecision decision;
	printf("validation: the deployment law and the ABI gates\n");
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_CONTINUOUS_BATCH_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_CONTINUOUS_BATCH_CONFIGURATION_BYTES;
	configuration.max_input_rows = 4u;
	configuration.max_active_lanes = 8u;
	configuration.queue_capacity = 4u;
	configuration.lane_capacity = 4u;
	expect(SparkContinuousBatchInitialize(&configuration,&controller) ==
		SPARK_STATUS_INVALID_ARGUMENT,
		"max_active_lanes > max_input_rows is refused (deployment law)");
	configuration.max_active_lanes = 4u;
	configuration.lane_capacity = 8u;
	expect(SparkContinuousBatchInitialize(&configuration,&controller) ==
		SPARK_STATUS_INVALID_ARGUMENT,
		"lane_capacity > max_active_lanes is refused");
	configuration.lane_capacity = 4u;
	configuration.max_input_rows = 0u;
	expect(SparkContinuousBatchInitialize(&configuration,&controller) ==
		SPARK_STATUS_INVALID_ARGUMENT,
		"zero rows is refused");
	configuration.max_input_rows = 4u;
	configuration.descriptor_bytes = 1u;
	expect(SparkContinuousBatchInitialize(&configuration,&controller) ==
		SPARK_STATUS_ABI_MISMATCH,
		"a stale descriptor_bytes is an ABI mismatch");
	configuration.descriptor_bytes = SPARK_CONTINUOUS_BATCH_CONFIGURATION_BYTES;
	expect(SparkContinuousBatchInitialize(&configuration,&controller) ==
		SPARK_STATUS_OK,
		"the valid configuration initializes");
	expect(SparkContinuousBatchOffer(controller,0,&decision) ==
		SPARK_STATUS_INVALID_ARGUMENT,
		"an offer without a request is invalid");
	expect(Offer(controller,701u,0u,1u,&decision) == SPARK_STATUS_INVALID_ARGUMENT,
		"a zero-row offer is invalid");
	expect(Offer(controller,701u,1u,0u,&decision) == SPARK_STATUS_INVALID_ARGUMENT,
		"a zero-budget offer is invalid");
	expect(Offer(controller,701u,1u,1u,&decision) == SPARK_STATUS_OK,
		"r701 offers fine");
	expect(Offer(controller,701u,1u,1u,&decision) == SPARK_STATUS_DUPLICATE,
		"a duplicate request_id is refused (resident scan)");
	expect(Offer(controller,702u,5u,1u,&decision) == SPARK_STATUS_OK &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_OVERSIZE,
		"r702 is oversize and queues");
	expect(Offer(controller,702u,5u,1u,&decision) == SPARK_STATUS_DUPLICATE,
		"a duplicate request_id is refused (queue scan)");
	expect(SparkContinuousBatchWithdraw(controller,702u) == SPARK_STATUS_OK,
		"r702 withdraws from the queue");
	expect(SparkContinuousBatchWithdraw(controller,702u) == SPARK_STATUS_NOT_FOUND,
		"a second withdrawal is NOT_FOUND");
	/* The request_bytes gate catches a stale caller. */
	{
		SparkContinuousBatchRequest request;
		memset(&request,0,sizeof(request));
		request.abi_version = SPARK_CONTINUOUS_BATCH_ABI_VERSION;
		request.descriptor_bytes = 1u;
		request.request_id = 703u;
		request.prompt_row_count = 1u;
		request.output_token_budget = 1u;
		expect(SparkContinuousBatchOffer(controller,&request,&decision) ==
			SPARK_STATUS_ABI_MISMATCH,
			"a stale offer descriptor is an ABI mismatch");
	}
	SparkContinuousBatchDestroy(controller);
	expect(SparkContinuousBatchDestroy(0) == SPARK_STATUS_OK,
		"destroying nothing is fine");
}

int main(void)
{
	printf("continuous-batching step-boundary contract (host proofs)\n");
	TestPolicyPick();
	TestInterleavingBitExact();
	TestRefusalQueueNotWedge();
	TestRowArithmeticAndChunking();
	TestOversizeAndQueueFull();
	TestStarvationBoundAndReservation();
	TestEngineRetirementSeam();
	TestValidation();
	if ( failures == 0 )
		printf("continuous-batching contract: ALL PROOFS PASS\n");
	else
		printf("continuous-batching contract: %d FAILURES\n",failures);
	return(failures == 0 ? 0 : 1);
}
