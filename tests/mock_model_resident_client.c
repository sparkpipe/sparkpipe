#include "mock_model_resident_client.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Link-time stand-in for the real resident client. Fidelity contract with
 * runtime/model_resident_client.c:
 * - in-flight submissions are a bounded per-rank queue (real: pending ring);
 *   a full queue reports BUSY.
 * - every submit path runs EnsureConnected first: the first activity after
 *   a server-side drop reports the dead socket once (IO_ERROR), the next
 *   attempt reconnects with a fresh client generation; killed ranks keep
 *   refusing until revived.
 * - FailStop / server-side death silently drops every in-flight submission
 *   and pending decision (the real client clears its pending ring without
 *   callbacks; the pipeline learns via the progress error path).
 * - decisions are only answered after the pipeline actually committed or
 *   aborted the submission; continued transactions auto-commit via the
 *   lease and never see a decision. */

#define MOCK_INFLIGHT_CAPACITY 8u

typedef struct MockInflight
{
	uint64_t submission_id;
	uint32_t result_driven;
	uint32_t committed;
	uint32_t requires_decision;
	uint32_t decision_kind;
	SparkModelServingSubmission submission;
} MockInflight;

typedef struct MockPendingDecision
{
	uint64_t submission_id;
	uint32_t decision_kind;
} MockPendingDecision;

struct SparkModelResidentClient
{
	uint32_t rank_index;
	uint32_t stage_index;
	uint32_t connected;
	uint32_t failed;
	uint32_t detected;
	uint32_t stay_dead;
	uint64_t client_generation;
	SparkModelResidentSubmitResultFunction submit_result_function;
	void *submit_result_context;
	SparkModelResidentDecisionResultFunction decision_result_function;
	void *decision_result_context;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	uint32_t submit_calls;
	uint32_t prepare_calls;
	uint32_t continue_calls;
	uint32_t commit_calls;
	uint32_t abort_calls;
	uint64_t last_submission_id;
	SparkModelServingLane last_lane;
	MockInflight inflight[MOCK_INFLIGHT_CAPACITY];
	uint32_t inflight_count;
	MockPendingDecision pending_decisions[MOCK_INFLIGHT_CAPACITY];
	uint32_t pending_decision_count;
	uint32_t is_final_rank;
	SparkStatus scripted_submit_status;
	SparkStatus scripted_call_status[MOCK_CALL_COUNT];
};

static SparkModelResidentClient *mock_registry[MOCK_RESIDENT_MAX_RANKS];
static uint32_t mock_registry_count;
static uint32_t mock_auto_tokens;
static uint32_t mock_token_start = 11u;

static int MockTraceEnabled(void)
{
	return( getenv("MOCK_TRACE") != 0 );
}

SparkModelResidentClient *MockResidentClientByRank(uint32_t stage_index)
{
	uint32_t i;
	SparkModelResidentClient *found = 0;
	for (i=0u; i<mock_registry_count; i++)
		if ( mock_registry[i] != 0 &&
		     mock_registry[i]->stage_index == stage_index &&
		     (found == 0 || mock_registry[i]->client_generation >
		        found->client_generation) )
			found = mock_registry[i];
	return(found);
}

uint32_t MockResidentClientCalls(uint32_t stage_index, uint32_t kind)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c == 0 )
		return(0u);
	switch (kind)
	{
		case MOCK_CALL_SUBMIT: return(c->submit_calls);
		case MOCK_CALL_PREPARE: return(c->prepare_calls);
		case MOCK_CALL_CONTINUE: return(c->continue_calls);
		case MOCK_CALL_COMMIT: return(c->commit_calls);
		case MOCK_CALL_ABORT: return(c->abort_calls);
	}
	return(0u);
}

uint64_t MockResidentClientGeneration(uint32_t stage_index)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	return( c != 0 ? c->client_generation : 0u );
}

uint32_t MockResidentClientLastLane(uint32_t stage_index,SparkModelServingLane *lane)
{
	SparkModelResidentClient *client = MockResidentClientByRank(stage_index);
	if ( client == 0 || lane == 0 || client->last_lane.request_id == 0u )
		return(0u);
	*lane = client->last_lane;
	return(1u);
}

void MockResidentClientScriptSubmitStatus(uint32_t stage_index, SparkStatus status)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 )
		c->scripted_submit_status = status;
}

void MockResidentClientScriptCallStatus(uint32_t stage_index, uint32_t kind, SparkStatus status)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 && kind < MOCK_CALL_COUNT )
		c->scripted_call_status[kind] = status;
}

uint32_t MockResidentClientPendingCount(uint32_t stage_index)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	return(c != 0 ? c->inflight_count + c->pending_decision_count : 0u);
}

uint32_t MockResidentClientOwnsSubmission(uint32_t stage_index, uint64_t submission_id)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	uint32_t k;
	if ( c == 0 )
		return(0u);
	for (k=0u; k<c->inflight_count; k++)
		if ( c->inflight[k].submission_id == submission_id )
			return(1u);
	for (k=0u; k<c->pending_decision_count; k++)
		if ( c->pending_decisions[k].submission_id == submission_id )
			return(1u);
	return(0u);
}

void MockResidentClientFireResult(uint32_t stage_index, uint64_t submission_id, SparkStatus status)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( MockResidentClientDeliverEvent(stage_index,submission_id,MOCK_EVENT_RESULT,status,1u) != 0u )
		return;
	if ( c != 0 && c->submit_result_function != 0 )
		c->submit_result_function(c->submit_result_context,submission_id,status);
}

void MockResidentClientFireDecision(uint32_t stage_index, uint64_t submission_id, uint32_t decision_kind, SparkStatus status)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	uint32_t k;
	if ( c == 0 )
		return;
	for (k=0u; k<c->pending_decision_count; k++)
		if ( c->pending_decisions[k].submission_id == submission_id &&
			c->pending_decisions[k].decision_kind == decision_kind )
		{
			(void)MockResidentClientDeliverEvent(stage_index,submission_id,MOCK_EVENT_DECISION,status,1u);
			return;
		}
	if ( c->decision_result_function != 0 )
		c->decision_result_function(c->decision_result_context,submission_id,decision_kind,status);
}

void MockResidentClientFireCompletion(uint32_t stage_index, const SparkModelServingCompletion *completion)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	uint32_t k;
	if ( c == 0 )
		return;
	for (k=0u; k<c->inflight_count; k++)
		if ( c->inflight[k].submission_id == completion->submission_id )
		{
			c->inflight[k] = c->inflight[--c->inflight_count];
			break;
		}
	if ( c->completion_function != 0 )
		c->completion_function(c->completion_context,completion);
}

static void MockResidentClientDropInflight(SparkModelResidentClient *c)
{
	memset(c->inflight,0,sizeof(c->inflight));
	c->inflight_count = 0u;
	memset(c->pending_decisions,0,sizeof(c->pending_decisions));
	c->pending_decision_count = 0u;
}

void MockResidentClientDisconnect(uint32_t stage_index)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 )
	{
		c->connected = 0u;
		c->detected = 0u;
		MockResidentClientDropInflight(c);
	}
}

void MockResidentClientKill(uint32_t stage_index)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 )
	{
		c->connected = 0u;
		c->detected = 0u;
		c->stay_dead = 1u;
		MockResidentClientDropInflight(c);
	}
}

void MockResidentClientRevive(uint32_t stage_index)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 )
		c->stay_dead = 0u;
}

void MockResidentClientReset(void)
{
	uint32_t i;
	for (i=0u; i<mock_registry_count; i++)
		if ( mock_registry[i] != 0 )
			free(mock_registry[i]);
	mock_registry_count = 0u;
	mock_auto_tokens = 0u;
	mock_token_start = 11u;
	memset(mock_registry,0,sizeof(mock_registry));
}

static SparkStatus MockResidentClientEnsureConnected(SparkModelResidentClient *c)
{
	if ( c->connected != 0u )
		return(SPARK_STATUS_OK);
	if ( c->detected == 0u )
	{
		c->detected = 1u;
		return(SPARK_STATUS_IO_ERROR);
	}
	if ( c->stay_dead != 0u )
		return(SPARK_STATUS_IO_ERROR);
	c->connected = 1u;
	c->detected = 0u;
	c->client_generation++;
	if ( MockTraceEnabled() )
		fprintf(stderr,"MOCK rank=%u reconnected generation=%llu\n",
			c->stage_index,(unsigned long long)c->client_generation);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientConnect(
	const SparkModelResidentClientConfiguration *configuration,
	SparkModelResidentClient **client_out)
{
	SparkModelResidentClient *c;
	uint32_t i;
	if ( configuration == 0 || client_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	c = 0;
	for (i=0u; i<mock_registry_count; i++)
		if ( mock_registry[i] != 0 &&
		     mock_registry[i]->stage_index == configuration->stage_index &&
		     mock_registry[i]->connected == 0u )
		{
			free(mock_registry[i]);
			mock_registry[i] = 0;
			c = (SparkModelResidentClient *)calloc(1u,sizeof(*c));
			mock_registry[i] = c;
			break;
		}
	if ( c == 0 )
	{
		if ( mock_registry_count >= MOCK_RESIDENT_MAX_RANKS )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		c = (SparkModelResidentClient *)calloc(1u,sizeof(*c));
		if ( c == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		mock_registry[mock_registry_count++] = c;
	}
	c->rank_index = configuration->rank_index;
	c->stage_index = configuration->stage_index;
	c->connected = 1u;
	c->client_generation = 1u;
	c->submit_result_function = configuration->submit_result_function;
	c->submit_result_context = configuration->submit_result_context;
	c->decision_result_function = configuration->decision_result_function;
	c->decision_result_context = configuration->decision_result_context;
	c->completion_function = configuration->completion_function;
	c->completion_context = configuration->completion_context;
	c->scripted_submit_status = SPARK_STATUS_OK;
	*client_out = c;
	return(SPARK_STATUS_OK);
}

void SparkModelResidentClientDestroy(SparkModelResidentClient *client)
{
	(void)client;
}

void SparkModelResidentClientFailStop(SparkModelResidentClient *client)
{
	if ( client != 0 )
	{
		client->failed = 1u;
		client->connected = 0u;
		client->detected = 1u;
		MockResidentClientDropInflight(client);
	}
}

static SparkStatus MockResidentClientEnqueue(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission,
	const char *op, uint32_t kind)
{
	MockInflight *slot;
	SparkStatus connect_status;
	connect_status = MockResidentClientEnsureConnected(client);
	if ( connect_status != SPARK_STATUS_OK )
		return(connect_status);
	if ( submission->submission_id <= client->last_submission_id )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( client->scripted_call_status[kind] != SPARK_STATUS_OK )
		return(client->scripted_call_status[kind]);
	if ( client->scripted_submit_status != SPARK_STATUS_OK )
		return(client->scripted_submit_status);
	if ( client->inflight_count >= MOCK_INFLIGHT_CAPACITY )
		return(SPARK_STATUS_BUSY);
	slot = &client->inflight[client->inflight_count++];
	memset(slot,0,sizeof(*slot));
	slot->submission_id = submission->submission_id;
	slot->submission = *submission;
	slot->committed = kind == MOCK_CALL_CONTINUE || kind == MOCK_CALL_SUBMIT;
	slot->requires_decision = kind == MOCK_CALL_PREPARE;
	client->last_submission_id = submission->submission_id;
	if ( submission->lane_count != 0u && submission->lanes != 0 )
		client->last_lane = submission->lanes[0];
	if ( MockTraceEnabled() )
		fprintf(stderr,"MOCK rank=%u %s id=%llu kind=%u rows=%u seq=%llu inflight=%u\n",
			client->stage_index,op,
			(unsigned long long)submission->submission_id,
			(unsigned)submission->work_kind,(unsigned)submission->row_count,
			(unsigned long long)submission->sequence_id,
			(unsigned)client->inflight_count);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientSubmit(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	if ( client == 0 || submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->submit_calls++;
	return(MockResidentClientEnqueue(client,submission,"submit",MOCK_CALL_SUBMIT));
}

SparkStatus SparkModelResidentClientPrepare(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	if ( client == 0 || submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->prepare_calls++;
	return(MockResidentClientEnqueue(client,submission,"prepare",MOCK_CALL_PREPARE));
}

SparkStatus SparkModelResidentClientCanQueueContinuation(
	const SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	if ( client == 0 || submission == 0 || submission->submission_id <= client->last_submission_id )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(client->scripted_call_status[MOCK_CALL_CAN_CONTINUE]);
}

SparkStatus SparkModelResidentClientContinue(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	if ( client == 0 || submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->continue_calls++;
	return(MockResidentClientEnqueue(client,submission,"continue",MOCK_CALL_CONTINUE));
}

SparkStatus SparkModelResidentClientCanQueueDecision(
	const SparkModelResidentClient *client,
	uint64_t submission_id,
	uint32_t decision_kind)
{
	uint32_t k;
	if ( client == 0 || client->connected == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (k=0u; k<client->inflight_count; k++)
	{
		const MockInflight *slot = &client->inflight[k];
		if ( slot->submission_id == submission_id && slot->requires_decision != 0u &&
			slot->result_driven != 0u && slot->committed == 0u && slot->decision_kind == 0u )
			return(client->scripted_call_status[decision_kind ==
				SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT ? MOCK_CALL_CAN_COMMIT : MOCK_CALL_CAN_ABORT]);
	}
	return(SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus MockResidentClientQueueDecision(
	SparkModelResidentClient *client,
	uint64_t submission_id,
	uint32_t decision_kind)
{
	MockPendingDecision *slot;
	uint32_t k;
	SparkStatus status = SparkModelResidentClientCanQueueDecision(client,submission_id,decision_kind);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = client->scripted_call_status[decision_kind ==
		SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT ? MOCK_CALL_COMMIT : MOCK_CALL_ABORT];
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( client->pending_decision_count >= MOCK_INFLIGHT_CAPACITY )
		return(SPARK_STATUS_BUSY);
	for (k=0u; k<client->inflight_count; k++)
		if ( client->inflight[k].submission_id == submission_id )
		{
			client->inflight[k].decision_kind = decision_kind;
			break;
		}
	slot = &client->pending_decisions[client->pending_decision_count++];
	slot->submission_id = submission_id;
	slot->decision_kind = decision_kind;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientCommit(
	SparkModelResidentClient *client,
	uint64_t submission_id)
{
	if ( client == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->commit_calls++;
	return(MockResidentClientQueueDecision(client,submission_id,
		SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT));
}

SparkStatus SparkModelResidentClientAbort(
	SparkModelResidentClient *client,
	uint64_t submission_id)
{
	if ( client == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->abort_calls++;
	return(MockResidentClientQueueDecision(client,submission_id,
		SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT));
}

SparkStatus SparkModelResidentClientProgress(
	SparkModelResidentClient *client,
	uint32_t maximum_message_count)
{
	(void)maximum_message_count;
	if ( client == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(MockResidentClientEnsureConnected(client));
}

SparkStatus SparkModelResidentClientGetPollDescriptor(
	const SparkModelResidentClient *client,
	SparkModelResidentClientPollDescriptor *descriptor)
{
	(void)client;
	(void)descriptor;
	return(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkModelResidentClientGetView(
	const SparkModelResidentClient *client,
	SparkModelResidentClientView *view)
{
	if ( client == 0 || view == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(view,0,sizeof(*view));
	view->connected = client->connected;
	view->client_generation = client->client_generation;
	view->queue_capacity = 64u;
	view->pending_submission_count = client->inflight_count;
	view->max_active_sequence_count = 4u;
	view->max_input_row_count = 8u;
	view->resident_sequence_capacity = 4u;
	return(SPARK_STATUS_OK);
}

void MockResidentClientSetAutoTokens(uint32_t count)
{
	mock_auto_tokens = count;
}

void MockResidentClientSetTokenStart(uint32_t first_token_id)
{
	mock_token_start = first_token_id;
}

void MockResidentClientSetFinalRank(uint32_t stage_index, uint32_t is_final)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 )
		c->is_final_rank = is_final;
}

uint64_t MockResidentClientPendingEvent(uint32_t stage_index, uint32_t kind, uint32_t ordinal)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	uint32_t k;
	if ( c == 0 || c->connected == 0u )
		return(0u);
	if ( kind == MOCK_EVENT_DECISION )
		return(ordinal < c->pending_decision_count ?
			c->pending_decisions[ordinal].submission_id : 0u);
	for (k=0u; k<c->inflight_count; k++)
	{
		MockInflight *slot = &c->inflight[k];
		if ( (kind == MOCK_EVENT_RESULT && slot->result_driven == 0u) ||
			(kind == MOCK_EVENT_COMPLETION && slot->result_driven != 0u && slot->committed != 0u) )
		{
			if ( ordinal == 0u )
				return(slot->submission_id);
			ordinal--;
		}
	}
	return(0u);
}

uint32_t MockResidentClientDeliverEvent(uint32_t stage_index, uint64_t submission_id,
    uint32_t kind, SparkStatus status, uint32_t deliver)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	uint32_t k;
	if ( c == 0 || c->connected == 0u )
		return(0u);
	if ( kind == MOCK_EVENT_DECISION )
	{
		for (k=0u; k<c->pending_decision_count; k++)
			if ( c->pending_decisions[k].submission_id == submission_id )
			{
				uint32_t j,decision_kind = c->pending_decisions[k].decision_kind;
				c->pending_decisions[k] = c->pending_decisions[--c->pending_decision_count];
				for (j=0u; j<c->inflight_count; j++)
					if ( c->inflight[j].submission_id == submission_id )
					{
						if ( status == SPARK_STATUS_OK && decision_kind == SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT )
							c->inflight[j].committed = 1u;
						else
							c->inflight[j] = c->inflight[--c->inflight_count];
						break;
					}
				if ( deliver != 0u && c->decision_result_function != 0 )
					c->decision_result_function(c->decision_result_context,submission_id,decision_kind,status);
				return(1u);
			}
		return(0u);
	}
	for (k=0u; k<c->inflight_count; k++)
		if ( c->inflight[k].submission_id == submission_id )
		{
			MockInflight saved = c->inflight[k];
			if ( kind == MOCK_EVENT_RESULT && saved.result_driven == 0u )
			{
				c->inflight[k].result_driven = 1u;
				if ( status != SPARK_STATUS_OK )
					c->inflight[k] = c->inflight[--c->inflight_count];
				if ( deliver != 0u && c->submit_result_function != 0 )
					c->submit_result_function(c->submit_result_context,submission_id,status);
				return(1u);
			}
			if ( kind == MOCK_EVENT_COMPLETION && saved.result_driven != 0u && saved.committed != 0u )
			{
				SparkModelServingCompletion completion;
				uint32_t t;
				memset(&completion,0,sizeof(completion));
				completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
				completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
				completion.status = status;
				completion.submission_id = submission_id;
				completion.request_id = saved.submission.request_id;
				completion.sequence_id = saved.submission.sequence_id;
				completion.sequence_position = saved.submission.sequence_position;
				completion.control_generation = saved.submission.control_generation;
				completion.transaction_id = saved.submission.transaction_id;
				completion.dispatch_generation = saved.submission.dispatch_generation;
				completion.request_generation = saved.submission.request_generation;
				completion.step_generation = saved.submission.step_generation;
				completion.residency = saved.submission.residency;
				if ( c->is_final_rank != 0u && mock_auto_tokens != 0u &&
					saved.submission.work_kind < SPARK_MODEL_SERVING_WORK_KIND_RELEASE && status == SPARK_STATUS_OK )
				{
					completion.token_count = saved.submission.active_sequence_count * mock_auto_tokens;
					completion.tokens_per_sequence = mock_auto_tokens;
					completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
					completion.accepted_token_count = completion.token_count;
					for (t=0u; t<completion.token_count && t<(uint32_t)(sizeof(completion.token_ids)/sizeof(completion.token_ids[0])); t++)
						completion.token_ids[t] = mock_token_start + t;
				}
				c->inflight[k] = c->inflight[--c->inflight_count];
				if ( deliver != 0u && c->completion_function != 0 )
					c->completion_function(c->completion_context,&completion);
				return(1u);
			}
			return(0u);
		}
	return(0u);
}

static uint32_t MockResidentClientDriveKind(uint32_t kind)
{
	uint32_t i,pass,drove = 0u;
	for (i=0u; i<mock_registry_count; i++)
	{
		SparkModelResidentClient *c = mock_registry[i];
		if ( c == 0 )
			continue;
		for (pass=0u; pass<MOCK_INFLIGHT_CAPACITY; pass++)
		{
			uint64_t id = MockResidentClientPendingEvent(c->stage_index,kind,0u);
			if ( id == 0u )
				break;
			drove += MockResidentClientDeliverEvent(c->stage_index,id,kind,SPARK_STATUS_OK,1u);
		}
	}
	return(drove);
}

uint32_t MockResidentClientDriveResults(void)
{
	return(MockResidentClientDriveKind(MOCK_EVENT_RESULT));
}

uint32_t MockResidentClientDriveCompletions(void)
{
	return(MockResidentClientDriveKind(MOCK_EVENT_COMPLETION));
}

uint32_t MockResidentClientDriveDecisions(void)
{
	return(MockResidentClientDriveKind(MOCK_EVENT_DECISION));
}

uint32_t MockResidentClientDriveAll(void)
{
	uint32_t a,b,c;
	a = MockResidentClientDriveResults();
	b = MockResidentClientDriveDecisions();
	c = MockResidentClientDriveCompletions();
	return(a + b + c);
}

uint64_t SparkModelResidentClientNextProgressNs(
	const SparkModelResidentClient *client)
{
	return(client != 0 && client->connected == 0u ? 1u : 0u);
}
