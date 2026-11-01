#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "fixtures/model_resident_deployment_fixture.h"
#include "mock_model_resident_client.h"
#include "sparkpipe/spark_model_pipeline_client.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#define TEST_RANKS 3u
#define FAULT_ROUNDS_DEFAULT 400u
#define FAULT_RECORD_CAPACITY 12u

static uint32_t test_failures,test_checks,case_number,case_kind,case_rank,case_phase;
static uint64_t initial_seed = 1u,random_state = 1u,next_id = 1u;

#define CHECK(cond, name) do { \
		test_checks++; \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL seed=%llu case=%u kind=%u rank=%u phase=%u line=%d %s\n", \
			    (unsigned long long)initial_seed,case_number,case_kind,case_rank,case_phase,__LINE__,name); \
		} \
	} while (0)

static uint64_t FuzzRand(void)
{
	random_state = random_state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
	return(random_state >> 17u);
}

typedef struct FaultRecord
{
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	uint32_t tokens[2],row_lanes[2];
	uint64_t positions[2],sequences[2];
	uint32_t accepted,result_count,completion_count;
	SparkStatus result_status,completion_status;
} FaultRecord;

typedef struct FaultState
{
	FaultRecord records[FAULT_RECORD_CAPACITY];
	uint32_t count;
	uint64_t accepted,results,rejected,completions;
	SparkModelPipelineClient *pipeline;
	FaultRecord *reentrant;
	uint64_t reentrant_trigger;
} FaultState;

static void FaultSubmit(SparkModelPipelineClient *pipeline, FaultState *state,
    FaultRecord *record, SparkStatus expected);

static FaultRecord *FaultFind(FaultState *state, uint64_t id)
{
	uint32_t i;
	for (i=0u; i<state->count; i++)
		if ( state->records[i].submission.submission_id == id )
			return(&state->records[i]);
	CHECK(0,"callback identifies a registered submission");
	return(0);
}

static void FaultSubmitResult(void *context, uint64_t id, SparkStatus status)
{
	FaultState *state = (FaultState *)context;
	FaultRecord *record = FaultFind(state,id);
	if ( record == 0 )
		return;
	CHECK(record->accepted != 0u,"synchronous rejection receives no result callback");
	CHECK(++record->result_count == 1u,"exactly one result callback per accepted submission");
	CHECK(status == record->result_status,"result has the independently expected status");
	state->results++;
	state->rejected += status != SPARK_STATUS_OK;
}

static void FaultCompletion(void *context, const SparkModelServingCompletion *completion)
{
	FaultState *state = (FaultState *)context;
	FaultRecord *record = FaultFind(state,completion->submission_id);
	const SparkModelServingSubmission *s;
	if ( record == 0 )
		return;
	s = &record->submission;
	CHECK(record->accepted != 0u,"synchronous rejection receives no completion callback");
	CHECK(record->result_count == 1u,"result precedes final completion");
	CHECK(++record->completion_count == 1u,"exactly one completion callback per accepted submission");
	CHECK(completion->status == (uint32_t)record->completion_status,"completion has the independently expected status");
	CHECK(completion->request_id == s->request_id && completion->sequence_id == s->sequence_id &&
		completion->sequence_position == s->sequence_position && completion->control_generation == s->control_generation &&
		completion->transaction_id == s->transaction_id && completion->dispatch_generation == s->dispatch_generation &&
		completion->request_generation == s->request_generation && completion->step_generation == s->step_generation,
		"completion preserves every submission identity and generation");
	if ( completion->status == SPARK_STATUS_OK )
	{
		CHECK(memcmp(&completion->residency,&s->residency,sizeof(s->residency)) == 0,"successful completion preserves residency");
		CHECK(completion->token_count == 2u && completion->tokens_per_sequence == 1u &&
			completion->token_ids[0] == 11u && completion->token_ids[1] == 12u,"successful completion preserves final-rank tokens");
	}
	else
		CHECK(completion->token_count == 0u,"failed work cannot publish tokens");
	{
		uint32_t rank;
		for (rank=0u; rank<TEST_RANKS; rank++)
			CHECK(MockResidentClientOwnsSubmission(rank,completion->submission_id) == 0u,
				"completion cannot release a transaction still owned by a rank");
	}
	state->completions++;
	if ( state->reentrant != 0 && completion->submission_id == state->reentrant_trigger )
	{
		FaultRecord *next = state->reentrant;
		state->reentrant = 0;
		FaultSubmit(state->pipeline,state,next,SPARK_STATUS_OK);
	}
}

static const char *const FaultHosts[TEST_RANKS] =
{
	"fault-a","fault-b","fault-c"
};

static void FaultBuildDeployment(SparkModelResidentDeployment *deployment,
    const char *path, const char *runtime_root)
{
	TestModelResidentDeploymentFixture fixture;
	const char *runtime_roots[TEST_RANKS];
	uint32_t stage_indices[TEST_RANKS];
	SparkModelResidentEndpoint endpoints[TEST_RANKS];
	uint32_t rank;
	for (rank=0u; rank<TEST_RANKS; rank++)
	{
		runtime_roots[rank] = runtime_root;
		stage_indices[rank] = rank;
		memset(&endpoints[rank],0,sizeof(endpoints[rank]));
		endpoints[rank].abi_version = SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION;
		endpoints[rank].descriptor_bytes = SPARK_MODEL_RESIDENT_ENDPOINT_BYTES;
		endpoints[rank].kind = SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP;
		endpoints[rank].tcp_host = FaultHosts[rank];
		endpoints[rank].tcp_port = (uint32_t)(59200u + rank);
	}
	memset(&fixture,0,sizeof(fixture));
	fixture.adapter_shared_object_path = TEST_MODEL_SERVING_ADAPTER_PATH;
	fixture.driver_shared_object_path = TEST_MODEL_SERVING_ADAPTER_PATH;
	fixture.driver_program_name = "resident_decode";
	fixture.transport_shared_object_path = TEST_MODEL_RESIDENT_TRANSPORT_PATH;
	fixture.transport_mode = "host-rdma";
	fixture.node_target = "fault.model.serving.target";
	fixture.adapter_configuration_path = "tests/fixtures/model_serving_adapter_config.json";
	fixture.runtime_roots = runtime_roots;
	fixture.transport_hosts = FaultHosts;
	fixture.stage_indices = stage_indices;
	fixture.control_endpoints = endpoints;
	fixture.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	fixture.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	fixture.runtime_limits.max_inflight_submission_count = 4u;
	fixture.runtime_limits.max_active_sequence_count = 4u;
	fixture.runtime_limits.max_input_row_count = 8u;
	fixture.runtime_limits.resident_sequence_capacity = 32u;
	fixture.runtime_limits.kv_logical_page_capacity = 256u;
	fixture.runtime_limits.kv_physical_page_capacity = 256u;
	fixture.runtime_limits.output_vocabulary = TEST_MODEL_RESIDENT_OUTPUT_VOCABULARY;
	fixture.control_port_base = 59200u;
	fixture.node_count = TEST_RANKS;
	fixture.coordinator_rank_index = 0u;
	assert(TestModelResidentDeploymentWrite(path,&fixture) == 0);
	assert(SparkModelResidentDeploymentLoad(path,deployment) == SPARK_STATUS_OK);
}

static FaultRecord *FaultNew(SparkModelPipelineClient *pipeline, FaultState *state)
{
	FaultRecord *r;
	SparkModelServingSubmission *s;
	uint32_t i,slot;
	assert(state->count < FAULT_RECORD_CAPACITY);
	r = &state->records[state->count++];
	memset(r,0,sizeof(*r));
	s = &r->submission;
	s->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	s->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	s->work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	s->tokens_per_sequence = 1u;
	s->submission_id = next_id++;
	s->request_id = 900u + 2u * s->submission_id;
	s->sequence_id = 100u + 2u * s->submission_id;
	s->control_generation = SparkModelPipelineClientControlGeneration(pipeline);
	s->transaction_id = s->submission_id + 1000u;
	s->dispatch_generation = s->submission_id + 2000u;
	s->request_generation = s->control_generation + 4000u;
	s->step_generation = s->submission_id + 3000u;
	s->residency.word0 = s->submission_id;
	s->residency.word1 = s->submission_id + 100u;
	s->residency.generation = s->submission_id + 200u;
	s->residency.owner = 1u;
	s->active_sequence_count = s->new_token_count = s->lane_count = s->row_count = s->token_count = 2u;
	s->token_ids = r->tokens;
	s->lanes = r->lanes;
	s->row_lane_indices = r->row_lanes;
	s->row_positions = r->positions;
	s->row_sequence_ids = r->sequences;
	slot = (uint32_t)((s->submission_id * 2u) % 28u);
	for (i=0u; i<2u; i++)
	{
		r->lanes[i].request_id = s->request_id + i;
		r->lanes[i].request_generation = s->request_generation;
		r->lanes[i].step_generation = s->step_generation;
		r->lanes[i].sequence_id = s->sequence_id + i;
		r->lanes[i].resident_sequence_slot = slot + i;
		r->lanes[i].context_token_count = 1u;
		r->lanes[i].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
		r->tokens[i] = 11u + i;
		r->row_lanes[i] = i;
		r->sequences[i] = s->sequence_id + i;
	}
	return(r);
}

static void FaultSubmit(SparkModelPipelineClient *pipeline, FaultState *state,
    FaultRecord *record, SparkStatus expected)
{
	SparkStatus status;
	record->accepted = expected == SPARK_STATUS_OK;
	status = SparkModelPipelineClientSubmit(pipeline,&record->submission);
	CHECK(status == expected,"submit returns the expected admission status");
	if ( status == SPARK_STATUS_OK )
		state->accepted++;
	else
		CHECK(record->result_count == 0u && record->completion_count == 0u,"rejected call has no callbacks");
}

static SparkModelServingCompletion FaultCompletionValue(const FaultRecord *record, uint32_t rank)
{
	const SparkModelServingSubmission *s = &record->submission;
	SparkModelServingCompletion c;
	memset(&c,0,sizeof(c));
	c.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	c.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	c.submission_id = s->submission_id;
	c.request_id = s->request_id;
	c.sequence_id = s->sequence_id;
	c.sequence_position = s->sequence_position;
	c.control_generation = s->control_generation;
	c.transaction_id = s->transaction_id;
	c.dispatch_generation = s->dispatch_generation;
	c.request_generation = s->request_generation;
	c.step_generation = s->step_generation;
	c.residency = s->residency;
	if ( rank == TEST_RANKS - 1u )
	{
		c.token_count = c.accepted_token_count = 2u;
		c.tokens_per_sequence = 1u;
		c.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		c.token_ids[0] = 11u;
		c.token_ids[1] = 12u;
	}
	return(c);
}

static void FaultStep(SparkModelPipelineClient *pipeline, FaultState *state, uint32_t duplicate)
{
	struct { uint32_t rank,kind; uint64_t id; } events[TEST_RANKS * MOCK_EVENT_COUNT * 8u];
	uint32_t rank,kind,ordinal,count = 0u,index;
	for (rank=0u; rank<TEST_RANKS; rank++)
		for (kind=0u; kind<MOCK_EVENT_COUNT; kind++)
			for (ordinal=0u; ordinal<8u; ordinal++)
			{
				uint64_t id = MockResidentClientPendingEvent(rank,kind,ordinal);
				if ( id == 0u )
					break;
				events[count].rank = rank;
				events[count].kind = kind;
				events[count++].id = id;
			}
	if ( count != 0u && FuzzRand() % 4u != 0u )
	{
		FaultRecord *record;
		SparkModelServingCompletion completion;
		index = (uint32_t)(FuzzRand() % count);
		rank = events[index].rank;
		kind = events[index].kind;
		record = FaultFind(state,events[index].id);
		completion = FaultCompletionValue(record,rank);
		CHECK(MockResidentClientDeliverEvent(rank,events[index].id,kind,SPARK_STATUS_OK,1u) == 1u,"scheduled event was pending");
		if ( duplicate != 0u )
		{
			if ( kind == MOCK_EVENT_RESULT )
				MockResidentClientFireResult(rank,events[index].id,SPARK_STATUS_IO_ERROR);
			else if ( kind == MOCK_EVENT_DECISION )
				MockResidentClientFireDecision(rank,events[index].id,SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT,SPARK_STATUS_IO_ERROR);
			else
			{
				completion.status = SPARK_STATUS_IO_ERROR;
				MockResidentClientFireCompletion(rank,&completion);
			}
		}
	}
	(void)SparkModelPipelineClientProgress(pipeline,1u + (uint32_t)(FuzzRand() % 8u));
}

static void FaultSettle(SparkModelPipelineClient *pipeline, FaultState *state, uint32_t duplicate)
{
	SparkModelPipelineClientView view;
	uint32_t i,rank,step;
	for (step=0u; step<256u; step++)
	{
		uint32_t pending = 0u;
		FaultStep(pipeline,state,duplicate);
		for (rank=0u; rank<TEST_RANKS; rank++)
			pending += MockResidentClientPendingCount(rank);
		CHECK(SparkModelPipelineClientGetView(pipeline,&view) == SPARK_STATUS_OK,"pipeline view remains readable");
		if ( pending == 0u && view.active_transaction_count == 0u && SparkModelPipelineClientAllRanksReady(pipeline) != 0u )
			break;
	}
	CHECK(step < 256u,"bounded recovery reaches ready and empties both transaction and transport queues");
	CHECK(view.submitted_count == state->accepted && view.completed_count == state->completions &&
		view.admitted_count == state->results - state->rejected && view.rejected_count == state->rejected,
		"pipeline counters reconcile with the independent callback ledger");
	for (i=0u; i<state->count; i++)
	{
		FaultRecord *r = &state->records[i];
		CHECK(r->result_count == r->accepted && r->completion_count == r->accepted,"every accepted call terminates exactly once, every rejected call never does");
	}
}

static void FaultReachPhase(FaultRecord *record, uint32_t phase)
{
	uint32_t rank;
	if ( phase >= 1u )
		for (rank=0u; rank<TEST_RANKS; rank++)
			CHECK(MockResidentClientDeliverEvent(rank,record->submission.submission_id,MOCK_EVENT_RESULT,SPARK_STATUS_OK,1u) == 1u,"prepare result reached");
	if ( phase >= 2u )
		for (rank=0u; rank<TEST_RANKS; rank++)
			CHECK(MockResidentClientDeliverEvent(rank,record->submission.submission_id,MOCK_EVENT_DECISION,SPARK_STATUS_OK,1u) == 1u,"commit acknowledgment reached");
}

static void FaultExpectFailure(FaultRecord *record, SparkStatus status)
{
	if ( record->result_count == 0u )
		record->result_status = status;
	record->completion_status = status;
}

static void FaultDisconnect(SparkModelPipelineClient *pipeline, FaultState *state,
    uint32_t rank, uint32_t kill, uint32_t second)
{
	uint64_t generations[TEST_RANKS],fingerprint = SparkModelPipelineClientSessionFingerprint(pipeline);
	uint32_t i;
	for (i=0u; i<state->count; i++)
		if ( state->records[i].accepted != 0u && state->records[i].completion_count == 0u )
			FaultExpectFailure(&state->records[i],SPARK_STATUS_IO_ERROR);
	for (i=0u; i<TEST_RANKS; i++)
		generations[i] = MockResidentClientGeneration(i);
	if ( kill != 0u )
		MockResidentClientKill(rank);
	else
		MockResidentClientDisconnect(rank);
	if ( second != 0u )
		MockResidentClientDisconnect((rank + 1u) % TEST_RANKS);
	CHECK(SparkModelPipelineClientProgress(pipeline,1u) == SPARK_STATUS_IO_ERROR,"transport loss is explicit");
	CHECK(SparkModelPipelineClientAllRanksReady(pipeline) == 0u,"failed session closes admission");
	if ( kill != 0u )
	{
		FaultRecord *blocked = FaultNew(pipeline,state);
		for (i=0u; i<8u; i++)
			CHECK(SparkModelPipelineClientRecover(pipeline) == SPARK_STATUS_IO_ERROR,"dead rank cannot appear recovered");
		FaultSubmit(pipeline,state,blocked,SPARK_STATUS_IO_ERROR);
		CHECK(SparkModelPipelineClientSessionFingerprint(pipeline) == fingerprint,"partial recovery does not publish a new fingerprint");
		MockResidentClientRevive(rank);
	}
	FaultSettle(pipeline,state,0u);
	CHECK(SparkModelPipelineClientSessionFingerprint(pipeline) != fingerprint,"complete recovery publishes the changed session");
	for (i=0u; i<TEST_RANKS; i++)
		CHECK(MockResidentClientGeneration(i) > generations[i],"all ranks advance relative to this fault, not the original run");
}

enum
{
	FAULT_REORDER,
	FAULT_CALLBACK_REUSE,
	FAULT_SYNC_BUSY,
	FAULT_SYNC_REJECT,
	FAULT_RESULT_REJECT,
	FAULT_DISCONNECT,
	FAULT_KILL,
	FAULT_MULTI_DISCONNECT,
	FAULT_DROP,
	FAULT_IDENTITY,
	FAULT_COMPLETION_ERROR,
	FAULT_BUSY_ROLLBACK,
	FAULT_DECISION_ERROR,
	FAULT_CALL_ERROR,
	FAULT_CAPACITY,
	FAULT_CONTINUATION,
	FAULT_KIND_COUNT
};

static void FaultRunCase(SparkModelPipelineClient *pipeline, FaultState *state,
    uint32_t kind, uint32_t rank, uint32_t phase)
{
	FaultRecord *r;
	uint32_t i;
	case_number++;
	case_kind = kind;
	case_rank = rank;
	case_phase = phase;
	state->count = 0u;
	r = FaultNew(pipeline,state);
	if ( kind == FAULT_SYNC_BUSY || kind == FAULT_SYNC_REJECT || kind == FAULT_BUSY_ROLLBACK )
	{
		SparkStatus status = kind != FAULT_SYNC_REJECT ? SPARK_STATUS_BUSY : SPARK_STATUS_INVALID_ARGUMENT;
		uint64_t generations[TEST_RANKS];
		for (i=0u; i<TEST_RANKS; i++)
			generations[i] = MockResidentClientGeneration(i);
		if ( kind == FAULT_BUSY_ROLLBACK )
		{
			rank = 1u + rank % 2u;
			MockResidentClientScriptCallStatus(rank - 1u,MOCK_CALL_ABORT,SPARK_STATUS_IO_ERROR);
		}
		MockResidentClientScriptCallStatus(rank,MOCK_CALL_PREPARE,status);
		if ( kind == FAULT_SYNC_REJECT || rank != 0u )
			FaultExpectFailure(r,kind == FAULT_BUSY_ROLLBACK ? SPARK_STATUS_IO_ERROR : status);
		FaultSubmit(pipeline,state,r,kind == FAULT_SYNC_BUSY && rank == 0u ? SPARK_STATUS_BUSY : SPARK_STATUS_OK);
		MockResidentClientScriptCallStatus(rank,MOCK_CALL_PREPARE,SPARK_STATUS_OK);
		FaultSettle(pipeline,state,0u);
		if ( kind == FAULT_BUSY_ROLLBACK )
			MockResidentClientScriptCallStatus(rank - 1u,MOCK_CALL_ABORT,SPARK_STATUS_OK);
		if ( kind == FAULT_SYNC_BUSY )
		{
			for (i=0u; i<TEST_RANKS; i++)
				CHECK(MockResidentClientGeneration(i) == generations[i],"normal backpressure drains without resetting sessions");
			if ( rank != 0u )
				r = FaultNew(pipeline,state);
			FaultSubmit(pipeline,state,r,SPARK_STATUS_OK);
		}
	}
	else
		FaultSubmit(pipeline,state,r,SPARK_STATUS_OK);
	if ( kind == FAULT_REORDER || kind == FAULT_CAPACITY || kind == FAULT_CALLBACK_REUSE )
	{
		for (i=0u; i<3u; i++)
			FaultSubmit(pipeline,state,FaultNew(pipeline,state),SPARK_STATUS_OK);
		if ( kind == FAULT_CALLBACK_REUSE )
		{
			state->reentrant_trigger = r->submission.submission_id;
			state->reentrant = FaultNew(pipeline,state);
		}
		if ( kind == FAULT_CAPACITY )
		{
			FaultRecord *retry = FaultNew(pipeline,state);
			FaultSubmit(pipeline,state,retry,SPARK_STATUS_BUSY);
			FaultSettle(pipeline,state,0u);
			FaultSubmit(pipeline,state,retry,SPARK_STATUS_OK);
		}
	}
	else if ( kind == FAULT_RESULT_REJECT )
	{
		SparkStatus status = phase % 2u == 0u ? SPARK_STATUS_BUSY : SPARK_STATUS_IO_ERROR;
		FaultExpectFailure(r,status);
		CHECK(MockResidentClientDeliverEvent(rank,r->submission.submission_id,MOCK_EVENT_RESULT,status,1u) == 1u,"prepare rejection is delivered while pending");
	}
	else if ( kind == FAULT_DISCONNECT || kind == FAULT_KILL || kind == FAULT_MULTI_DISCONNECT )
	{
		FaultReachPhase(r,phase % 3u);
		FaultSubmit(pipeline,state,FaultNew(pipeline,state),SPARK_STATUS_OK);
		FaultDisconnect(pipeline,state,rank,kind == FAULT_KILL,kind == FAULT_MULTI_DISCONNECT);
	}
	else if ( kind == FAULT_DROP )
	{
		SparkModelPipelineClientView view;
		uint32_t event = phase % MOCK_EVENT_COUNT;
		FaultReachPhase(r,event == MOCK_EVENT_RESULT ? 0u : event == MOCK_EVENT_DECISION ? 1u : 2u);
		CHECK(MockResidentClientDeliverEvent(rank,r->submission.submission_id,event,SPARK_STATUS_OK,0u) == 1u,"the selected event is actually dropped");
		for (i=0u; i<64u; i++)
			FaultStep(pipeline,state,0u);
		CHECK(SparkModelPipelineClientGetView(pipeline,&view) == SPARK_STATUS_OK && view.active_transaction_count == 1u,
			"missing acknowledgment or completion retains the transaction");
		CHECK(r->completion_count == 0u,"progress cannot fabricate a dropped completion");
		FaultDisconnect(pipeline,state,rank,0u,0u);
	}
	else if ( kind == FAULT_IDENTITY )
	{
		SparkModelServingCompletion c;
		FaultReachPhase(r,2u);
		c = FaultCompletionValue(r,rank);
		switch (phase % 12u)
		{
		case 0u: c.request_id++; break;
		case 1u: c.sequence_id++; break;
		case 2u: c.sequence_position++; break;
		case 3u: c.control_generation++; break;
		case 4u: c.transaction_id++; break;
		case 5u: c.dispatch_generation++; break;
		case 6u: c.request_generation++; break;
		case 7u: c.step_generation++; break;
		case 8u: c.residency.word0++; break;
		case 9u: c.residency.word1++; break;
		case 10u: c.residency.generation++; break;
		case 11u: c.residency.owner++; break;
		}
		FaultExpectFailure(r,SPARK_STATUS_SCHEMA_ERROR);
		MockResidentClientFireCompletion(rank,&c);
	}
	else if ( kind == FAULT_COMPLETION_ERROR )
	{
		FaultReachPhase(r,1u + phase % 2u);
		if ( phase % 2u == 0u )
			CHECK(MockResidentClientDeliverEvent(rank,r->submission.submission_id,MOCK_EVENT_DECISION,SPARK_STATUS_OK,1u) == 1u,"rank commit precedes its execution completion");
		FaultExpectFailure(r,SPARK_STATUS_IO_ERROR);
		CHECK(MockResidentClientDeliverEvent(rank,r->submission.submission_id,MOCK_EVENT_COMPLETION,SPARK_STATUS_IO_ERROR,1u) == 1u,"execution failure cannot publish successful output");
		if ( phase % 2u == 0u )
		{
			for (i=0u; i<TEST_RANKS; i++)
				if ( i != rank )
					CHECK(MockResidentClientDeliverEvent(i,r->submission.submission_id,MOCK_EVENT_DECISION,SPARK_STATUS_OK,1u) == 1u,"remaining ranks acknowledge commit while their execution is withheld");
			CHECK(r->completion_count == 0u,"an early execution error cannot retire slower committed ranks");
		}
	}
	else if ( kind == FAULT_DECISION_ERROR )
	{
		FaultReachPhase(r,1u);
		FaultExpectFailure(r,SPARK_STATUS_IO_ERROR);
		CHECK(MockResidentClientDeliverEvent(rank,r->submission.submission_id,MOCK_EVENT_DECISION,SPARK_STATUS_IO_ERROR,1u) == 1u,"commit decision failure is delivered while pending");
	}
	else if ( kind == FAULT_CALL_ERROR )
	{
		uint32_t calls[] = { MOCK_CALL_COMMIT,MOCK_CALL_CAN_COMMIT,MOCK_CALL_ABORT,MOCK_CALL_CAN_ABORT };
		uint32_t call = calls[phase % 4u];
		uint32_t rejected_rank = (rank + 1u) % TEST_RANKS;
		MockResidentClientScriptCallStatus(rank,call,SPARK_STATUS_IO_ERROR);
		FaultExpectFailure(r,SPARK_STATUS_IO_ERROR);
		if ( call == MOCK_CALL_ABORT || call == MOCK_CALL_CAN_ABORT )
			CHECK(MockResidentClientDeliverEvent(rejected_rank,r->submission.submission_id,MOCK_EVENT_RESULT,SPARK_STATUS_BUSY,1u) == 1u,"rejection requires aborting other prepared ranks");
		MockResidentClientDriveResults();
		MockResidentClientScriptCallStatus(rank,call,SPARK_STATUS_OK);
	}
	else if ( kind == FAULT_CONTINUATION )
	{
		FaultRecord *continued;
		SparkModelPipelineClientView before,after;
		FaultSettle(pipeline,state,0u);
		CHECK(SparkModelPipelineClientGetView(pipeline,&before) == SPARK_STATUS_OK,"continuation baseline view");
		continued = FaultNew(pipeline,state);
		continued->submission.request_id = r->submission.request_id;
		continued->submission.sequence_id = r->submission.sequence_id;
		continued->submission.sequence_position = 1u;
		continued->submission.step_generation = r->submission.step_generation + 1u;
		for (i=0u; i<2u; i++)
		{
			continued->lanes[i] = r->lanes[i];
			continued->lanes[i].sequence_position = continued->positions[i] = 1u;
			continued->lanes[i].context_token_count = 2u;
			continued->lanes[i].step_generation++;
			continued->sequences[i] = r->sequences[i];
		}
		if ( phase % 5u == 1u )
		{
			MockResidentClientScriptCallStatus(rank,MOCK_CALL_CAN_CONTINUE,SPARK_STATUS_BUSY);
			FaultSubmit(pipeline,state,continued,SPARK_STATUS_BUSY);
			MockResidentClientScriptCallStatus(rank,MOCK_CALL_CAN_CONTINUE,SPARK_STATUS_OK);
		}
		if ( phase % 5u >= 3u )
		{
			uint32_t blocked_rank = phase % 5u == 3u ? 0u : 1u + rank % 2u;
			MockResidentClientScriptCallStatus(blocked_rank,MOCK_CALL_CONTINUE,SPARK_STATUS_BUSY);
			if ( blocked_rank != 0u )
				FaultExpectFailure(continued,SPARK_STATUS_BUSY);
			FaultSubmit(pipeline,state,continued,blocked_rank == 0u ? SPARK_STATUS_BUSY : SPARK_STATUS_OK);
			MockResidentClientScriptCallStatus(blocked_rank,MOCK_CALL_CONTINUE,SPARK_STATUS_OK);
			if ( blocked_rank == 0u )
				FaultSubmit(pipeline,state,continued,SPARK_STATUS_OK);
		}
		else
		FaultSubmit(pipeline,state,continued,SPARK_STATUS_OK);
		if ( phase % 5u == 2u )
		{
			FaultExpectFailure(continued,SPARK_STATUS_IO_ERROR);
			CHECK(MockResidentClientDeliverEvent(rank,continued->submission.submission_id,MOCK_EVENT_RESULT,SPARK_STATUS_IO_ERROR,1u) == 1u,"continued admission fails explicitly");
		}
		FaultSettle(pipeline,state,0u);
		CHECK(SparkModelPipelineClientGetView(pipeline,&after) == SPARK_STATUS_OK && after.continued_count == before.continued_count + 1u,"the continuation lease path actually executed");
	}
	FaultSettle(pipeline,state,kind == FAULT_REORDER);
	FaultSubmit(pipeline,state,FaultNew(pipeline,state),SPARK_STATUS_OK);
	for (i=0u; i<TEST_RANKS; i++)
	{
		SparkModelServingCompletion old = FaultCompletionValue(r,i);
		MockResidentClientFireResult(i,r->submission.submission_id,SPARK_STATUS_IO_ERROR);
		MockResidentClientFireDecision(i,r->submission.submission_id,SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT,SPARK_STATUS_IO_ERROR);
		MockResidentClientFireCompletion(i,&old);
	}
	FaultSettle(pipeline,state,0u);
}

static uint32_t FaultParse(const char *text, uint64_t maximum, uint64_t *value)
{
	char *end;
	const char *p;
	if ( text == 0 || text[0] == 0 )
		return(0u);
	for (p=text; *p!=0; p++)
		if ( *p < '0' || *p > '9' )
			return(0u);
	errno = 0;
	*value = strtoull(text,&end,10);
	return(errno == 0 && *end == 0 && *value <= maximum);
}

int main(int argc, char **argv)
{
	SparkModelResidentDeployment deployment;
	SparkModelPipelineClientConfiguration configuration;
	SparkModelPipelineClient *pipeline = 0;
	FaultState state;
	uint64_t rounds = FAULT_ROUNDS_DEFAULT;
	uint32_t kind,rank,phase,round;
	char path[] = "/tmp/sparkpipe-serving-fuzz-XXXXXX";
	char runtime_root[1024];
	int fd;
	if ( argc > 3 || (argc > 1 && FaultParse(argv[1],UINT64_MAX,&initial_seed) == 0u) ||
		(argc > 2 && (FaultParse(argv[2],100000u,&rounds) == 0u || rounds == 0u)) )
	{
		fprintf(stderr,"usage: %s [seed:0..UINT64_MAX] [random-rounds:1..100000]\n",argv[0]);
		return(2);
	}
	random_state = initial_seed;
	fprintf(stderr,"test_serving_fault_fuzz: seed=%llu random_rounds=%llu\n",(unsigned long long)initial_seed,(unsigned long long)rounds);
	MockResidentClientReset();
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	fd = mkstemp(path);
	assert(fd >= 0 && close(fd) == 0);
	FaultBuildDeployment(&deployment,path,runtime_root);
	memset(&configuration,0,sizeof(configuration));
	memset(&state,0,sizeof(state));
	configuration.abi_version = SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_PIPELINE_CLIENT_CONFIGURATION_BYTES;
	configuration.connect_timeout_ms = 200u;
	configuration.deployment = &deployment;
	configuration.runtime_root = runtime_root;
	configuration.submit_result_function = FaultSubmitResult;
	configuration.submit_result_context = &state;
	configuration.completion_function = FaultCompletion;
	configuration.completion_context = &state;
	CHECK(SparkModelPipelineClientConnect(&configuration,&pipeline) == SPARK_STATUS_OK,"connect");
	if ( pipeline == 0 )
		return(1);
	state.pipeline = pipeline;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	for (kind=0u; kind<FAULT_KIND_COUNT && test_failures==0u; kind++)
		for (rank=0u; rank<TEST_RANKS && test_failures==0u; rank++)
			for (phase=0u; phase<(kind == FAULT_IDENTITY ? 12u : kind == FAULT_CALL_ERROR ? 4u : kind == FAULT_CONTINUATION ? 5u : 3u) && test_failures==0u; phase++)
				FaultRunCase(pipeline,&state,kind,rank,phase);
	for (round=0u; round<rounds && test_failures==0u; round++)
	{
		kind = (uint32_t)(FuzzRand() % FAULT_KIND_COUNT);
		rank = (uint32_t)(FuzzRand() % TEST_RANKS);
		phase = (uint32_t)(FuzzRand() % 12u);
		FaultRunCase(pipeline,&state,kind,rank,phase);
	}
	SparkModelPipelineClientDestroy(pipeline);
	MockResidentClientReset();
	CHECK(unlink(path) == 0,"temporary deployment is removed");
	fprintf(stderr,"test_serving_fault_fuzz: %u checks, %u failures, %u cases (seed=%llu random_rounds=%llu)\n",
		test_checks,test_failures,case_number,(unsigned long long)initial_seed,(unsigned long long)rounds);
	return(test_failures != 0u ? 1 : 0);
}
