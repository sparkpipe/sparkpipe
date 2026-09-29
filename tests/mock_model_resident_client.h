#ifndef MOCK_MODEL_RESIDENT_CLIENT_H
#define MOCK_MODEL_RESIDENT_CLIENT_H

#include "sparkpipe/spark_model_resident_client.h"

#define MOCK_RESIDENT_MAX_RANKS 16u

enum
{
	MOCK_CALL_SUBMIT = 0,
	MOCK_CALL_PREPARE,
	MOCK_CALL_CONTINUE,
	MOCK_CALL_COMMIT,
	MOCK_CALL_ABORT,
	MOCK_CALL_CAN_CONTINUE,
	MOCK_CALL_CAN_COMMIT,
	MOCK_CALL_CAN_ABORT,
	MOCK_CALL_COUNT
};

enum
{
	MOCK_EVENT_RESULT,
	MOCK_EVENT_DECISION,
	MOCK_EVENT_COMPLETION,
	MOCK_EVENT_COUNT
};

void MockResidentClientScriptCallStatus(uint32_t stage_index, uint32_t kind, SparkStatus status);
uint64_t MockResidentClientPendingEvent(uint32_t stage_index, uint32_t kind, uint32_t ordinal);
uint32_t MockResidentClientDeliverEvent(uint32_t stage_index, uint64_t submission_id,
    uint32_t kind, SparkStatus status, uint32_t deliver);
uint32_t MockResidentClientPendingCount(uint32_t stage_index);
uint32_t MockResidentClientOwnsSubmission(uint32_t stage_index, uint64_t submission_id);
SparkModelResidentClient *MockResidentClientByRank(uint32_t stage_index);
uint32_t MockResidentClientCalls(uint32_t stage_index, uint32_t kind);
uint64_t MockResidentClientGeneration(uint32_t stage_index);
uint32_t MockResidentClientLastLane(uint32_t stage_index,SparkModelServingLane *lane);
uint32_t MockResidentClientLaneLog(uint32_t index,SparkModelServingLane *lane);
void MockResidentClientScriptSubmitStatus(uint32_t stage_index, SparkStatus status);
void MockResidentClientScriptPrefixResult(uint32_t stage_index, SparkStatus status, uint32_t count);
void MockResidentClientScriptStalePrefixRequest(uint32_t stage_index, uint64_t request_id);
void MockResidentClientFireResult(uint32_t stage_index, uint64_t submission_id, SparkStatus status);
void MockResidentClientFireDecision(uint32_t stage_index, uint64_t submission_id, uint32_t decision_kind, SparkStatus status);
void MockResidentClientFireCompletion(uint32_t stage_index, const SparkModelServingCompletion *completion);
void MockResidentClientDisconnect(uint32_t stage_index);
void MockResidentClientKill(uint32_t stage_index);
void MockResidentClientRevive(uint32_t stage_index);
void MockResidentClientSetAutoTokens(uint32_t count);
void MockResidentClientSetTokenStart(uint32_t first_token_id);
void MockResidentClientSetFinalRank(uint32_t stage_index, uint32_t is_final);
uint32_t MockResidentClientDriveAll(void);
uint32_t MockResidentClientDriveResults(void);
uint32_t MockResidentClientDriveCompletions(void);
uint32_t MockResidentClientDriveDecisions(void);
void MockResidentClientReset(void);

#endif
