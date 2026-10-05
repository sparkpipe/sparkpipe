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

#define PROOF_MAX_QUEUE 32u

static uint32_t Boundary(SparkContinuousBatch *controller,uint64_t *released)
{
	uint32_t released_count = 0u;
	if ( SparkContinuousBatchBoundary(controller,released,PROOF_MAX_QUEUE,&released_count) != SPARK_STATUS_OK )
		expect(0,"boundary status");
	return(released_count);
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
	expect(SparkContinuousBatchPolicyPick(rows,boundaries,0,0u,4u,4u) == -1,
		"empty queue picks nothing");
	memset(excluded,1u,sizeof(excluded));
	rows[0] = 1u; boundaries[0] = 0u;
	expect(SparkContinuousBatchPolicyPick(rows,boundaries,excluded,1u,4u,4u) == -1,
		"all-excluded queue picks nothing");
	rows[0] = 5u; rows[1] = 3u; rows[2] = 8u; rows[3] = 3u;
	boundaries[0] = 7u; boundaries[1] = 8u; boundaries[2] = 9u; boundaries[3] = 8u;
	expect(SparkContinuousBatchPolicyPick(rows,boundaries,0,4u,10u,0u) == 1,
		"smallest rows first");
	expect(SparkContinuousBatchPolicyPick(rows,boundaries,0,4u,10u,4u) == 1,
		"no aged offer leaves smallest-first in charge");
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

static void TestRefusalQueueNotWedge(void)
{
	SparkContinuousBatch *controller;
	SparkContinuousBatchDecision decision;
	SparkContinuousBatchStatistics stats;
	SparkContinuousBatchLaneView lane;
	uint64_t released[PROOF_MAX_QUEUE];
	uint32_t released_count;
	printf("L2/L3 refusals queue, never wedge; retired slots reclaim at the next boundary\n");
	controller = MakeController(8u,2u,0u,8u);
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
	expect(Boundary(controller,released) == 0u,"a boundary with both residents live releases nothing");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.admission_queued_lanes == 1u,
		"the lane refusal is counted at the boundary");
	expect(SparkContinuousBatchRetire(controller,201u) == SPARK_STATUS_OK &&
		SparkContinuousBatchRetire(controller,202u) == SPARK_STATUS_OK,
		"the engine's terminal events retire both residents");
	released_count = Boundary(controller,released);
	expect(released_count == 1u && released[0] == 203u,
		"the retired lanes' slots free at the very next boundary");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.slot_reclaims == 2u && stats.admission_queued_lanes == 1u &&
		stats.resident_lane_count == 1u,
		"two reclaims, one named lane refusal, r203 resident");
	expect(SparkContinuousBatchGetLane(controller,0u,&lane) == SPARK_STATUS_OK &&
		lane.request_id == 203u && lane.phase == SPARK_CONTINUOUS_BATCH_LANE_RESIDENT && lane.prompt_row_count == 2u,
		"r203 took reclaimed slot 0");
	expect(SparkContinuousBatchRetire(controller,203u) == SPARK_STATUS_OK && Boundary(controller,released) == 0u,
		"r203 retires");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.resident_lane_count == 0u && stats.queued_offer_count == 0u &&
		stats.slot_reclaims == 3u && stats.admission_accepted == 3u,
		"every offer admitted, every slot reclaimed - no wedge, no drop");
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
	expect(Offer(controller,902u,2u,2u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
		"the oversize entry does not block the queue (no head-of-line)");
	expect(Offer(controller,903u,4u,1u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_QUEUED &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_FULL,
		"a queue at capacity answers FULL by name");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.admission_oversize == 1u && stats.admission_queue_full == 1u &&
		stats.queued_offer_count == 1u,
		"the counters name both refusals; only the kept offer is queued");
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
	uint64_t released[PROOF_MAX_QUEUE];
	uint32_t released_count,index;
	printf("L4 starvation bound: aging wins the boundary and holds it\n");
	controller = MakeController(8u,8u,2u,16u);
	for ( index = 0u; index < 5u; ++index )
		expect(Offer(controller,400u + index,1u,4u,&decision) == SPARK_STATUS_OK &&
			decision.outcome == SPARK_CONTINUOUS_BATCH_ADMITTED,
			"a long resident joins");
	expect(Offer(controller,999u,4u,1u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_QUEUED &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_ROWS,
		"S queues behind the resident rows");
	expect(Boundary(controller,released) == 0u,"boundary 1 releases nothing");
	expect(Offer(controller,500u,1u,2u,&decision) == SPARK_STATUS_OK &&
		decision.outcome == SPARK_CONTINUOUS_BATCH_QUEUED &&
		decision.queue_reason == SPARK_CONTINUOUS_BATCH_QUEUE_AHEAD,
		"t1 queues behind S (arrivals never cut ahead)");
	expect(Boundary(controller,released) == 0u,
		"the aged offer's refusal keeps the boundary closed to t1");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.queued_offer_count == 2u,
		"t1 is still queued - the reservation held");
	for ( index = 0u; index < 5u; ++index )
		expect(SparkContinuousBatchRetire(controller,400u + index) == SPARK_STATUS_OK,"a resident retires");
	released_count = Boundary(controller,released);
	expect(released_count == 2u && released[0] == 999u && released[1] == 500u,
		"S admits at the first boundary that fits it, t1 follows");
	expect(SparkContinuousBatchGetLane(controller,0u,&lane) == SPARK_STATUS_OK &&
		lane.request_id == 999u && lane.admitted_via_aging == 1u,
		"S's lane carries the aging flag");
	expect(SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.starvation_jumps == 2u && stats.admission_queued_rows == 2u,
		"two aging wins (S and t1 both waited past the bound); two refusals");
	expect(SparkContinuousBatchRetire(controller,999u) == SPARK_STATUS_OK && SparkContinuousBatchRetire(controller,500u) == SPARK_STATUS_OK &&
		Boundary(controller,released) == 0u && SparkContinuousBatchGetStatistics(controller,&stats) == SPARK_STATUS_OK &&
		stats.resident_lane_count == 0u && stats.queued_offer_count == 0u,
		"the aged run drains clean");
	SparkContinuousBatchDestroy(controller);
}

static void TestEngineRetirementSeam(void)
{
	SparkContinuousBatch *controller;
	SparkContinuousBatchDecision decision;
	SparkContinuousBatchStatistics stats;
	SparkContinuousBatchLaneView lane;
	uint64_t released[PROOF_MAX_QUEUE];
	uint32_t released_count;
	printf("L3 seam path: an engine-side retirement frees the slot next boundary\n");
	controller = MakeController(8u,2u,0u,8u);
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
	expect(Boundary(controller,released) == 0u,"no slot frees while both residents run");
	expect(SparkContinuousBatchRetire(controller,601u) == SPARK_STATUS_OK,
		"the engine's terminal report retires r601");
	expect(SparkContinuousBatchRetire(controller,601u) == SPARK_STATUS_OK,
		"a repeated retirement stays healthy");
	expect(SparkContinuousBatchRetire(controller,603u) == SPARK_STATUS_NOT_FOUND,
		"retiring a QUEUED id is NOT_FOUND (withdraw instead)");
	released_count = Boundary(controller,released);
	expect(released_count == 1u && released[0] == 603u,
		"the retired lane's slot frees at the very next boundary");
	expect(SparkContinuousBatchGetLane(controller,0u,&lane) == SPARK_STATUS_OK &&
		lane.request_id == 603u && lane.phase == SPARK_CONTINUOUS_BATCH_LANE_RESIDENT && lane.retired == 0u,
		"r603 took the reclaimed slot 0");
	expect(SparkContinuousBatchGetLane(controller,1u,&lane) == SPARK_STATUS_OK &&
		lane.request_id == 602u && lane.phase == SPARK_CONTINUOUS_BATCH_LANE_RESIDENT,
		"r602 keeps its slot across the seam reclaim");
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
	TestRefusalQueueNotWedge();
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
