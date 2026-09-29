#include "sparkpipe/spark_speculation_relay_draft.h"

#include <string.h>

#include "sparkpipe/spark_error_site.h"

static void SparkSpeculationRelayPut32(uint8_t *bytes,uint32_t value)
{
	bytes[0] = (uint8_t)value;
	bytes[1] = (uint8_t)(value >> 8);
	bytes[2] = (uint8_t)(value >> 16);
	bytes[3] = (uint8_t)(value >> 24);
}

static void SparkSpeculationRelayPut64(uint8_t *bytes,uint64_t value)
{
	SparkSpeculationRelayPut32(bytes,(uint32_t)value);
	SparkSpeculationRelayPut32(bytes + 4,(uint32_t)(value >> 32));
}

static uint32_t SparkSpeculationRelayGet32(const uint8_t *bytes)
{
	return((uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 | (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24);
}

static uint64_t SparkSpeculationRelayGet64(const uint8_t *bytes)
{
	return((uint64_t)SparkSpeculationRelayGet32(bytes) | (uint64_t)SparkSpeculationRelayGet32(bytes + 4) << 32);
}

static SparkStatus SparkSpeculationRelayValidate(const SparkSpeculationRelayFrame *frame,uint32_t vocab_size)
{
	uint32_t index;
	if ( frame->kind != SPARK_SPECULATION_RELAY_KIND_REQUEST && frame->kind != SPARK_SPECULATION_RELAY_KIND_DRAFT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( vocab_size == 0u || frame->token_count > SPARK_SPECULATION_RELAY_MAX_TOKENS || frame->requested_token_count == 0u || frame->requested_token_count > SPARK_SPECULATION_RELAY_MAX_TOKENS || frame->anchor_token >= vocab_size )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for (index=0u; index<frame->token_count; index++)
		if ( frame->token_ids[index] >= vocab_size )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( frame->kind == SPARK_SPECULATION_RELAY_KIND_REQUEST && (frame->token_count == 0u || (uint64_t)frame->token_count > frame->anchor_position + 1u || frame->token_ids[frame->token_count - 1u] != frame->anchor_token) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( frame->kind == SPARK_SPECULATION_RELAY_KIND_DRAFT && frame->token_count > frame->requested_token_count )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationRelayEncode(const SparkSpeculationRelayFrame *frame,uint32_t vocab_size,uint8_t bytes[SPARK_SPECULATION_RELAY_FRAME_BYTES])
{
	SparkStatus status;
	uint32_t index;
	if ( frame == 0 || bytes == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkSpeculationRelayValidate(frame,vocab_size);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(bytes,0,SPARK_SPECULATION_RELAY_FRAME_BYTES);
	SparkSpeculationRelayPut32(bytes,SPARK_SPECULATION_RELAY_MAGIC);
	SparkSpeculationRelayPut32(bytes + 4,SPARK_SPECULATION_RELAY_VERSION | frame->kind << 16);
	SparkSpeculationRelayPut64(bytes + 8,frame->engine_generation);
	SparkSpeculationRelayPut64(bytes + 16,frame->round_id);
	SparkSpeculationRelayPut64(bytes + 24,frame->sequence_id);
	SparkSpeculationRelayPut64(bytes + 32,frame->anchor_position);
	SparkSpeculationRelayPut32(bytes + 40,frame->anchor_token);
	SparkSpeculationRelayPut32(bytes + 44,frame->token_count | frame->requested_token_count << 16);
	for (index=0u; index<frame->token_count; index++)
		SparkSpeculationRelayPut32(bytes + SPARK_SPECULATION_RELAY_HEADER_BYTES + 4u * index,frame->token_ids[index]);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationRelayDecode(const uint8_t *bytes,uint32_t length,uint32_t vocab_size,SparkSpeculationRelayFrame *frame)
{
	uint32_t word,index;
	if ( bytes == 0 || frame == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(frame,0,sizeof(*frame));
	if ( length != SPARK_SPECULATION_RELAY_FRAME_BYTES || SparkSpeculationRelayGet32(bytes) != SPARK_SPECULATION_RELAY_MAGIC )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	word = SparkSpeculationRelayGet32(bytes + 4);
	if ( (word & 0xffffu) != SPARK_SPECULATION_RELAY_VERSION )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	frame->kind = word >> 16;
	frame->engine_generation = SparkSpeculationRelayGet64(bytes + 8);
	frame->round_id = SparkSpeculationRelayGet64(bytes + 16);
	frame->sequence_id = SparkSpeculationRelayGet64(bytes + 24);
	frame->anchor_position = SparkSpeculationRelayGet64(bytes + 32);
	frame->anchor_token = SparkSpeculationRelayGet32(bytes + 40);
	word = SparkSpeculationRelayGet32(bytes + 44);
	frame->token_count = word & 0xffffu;
	frame->requested_token_count = word >> 16;
	for (index=0u; index<SPARK_SPECULATION_RELAY_MAX_TOKENS; index++)
	{
		word = SparkSpeculationRelayGet32(bytes + SPARK_SPECULATION_RELAY_HEADER_BYTES + 4u * index);
		if ( index >= frame->token_count && word != 0u )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( index < frame->token_count )
			frame->token_ids[index] = word;
	}
	SPARK_RETURN(SparkSpeculationRelayValidate(frame,vocab_size));
}

SparkStatus SparkSpeculationRelayDraftInitialize(SparkSpeculationRelayDraft *relay,uint64_t engine_generation,uint32_t vocab_size)
{
	if ( relay == 0 || vocab_size == 0u || engine_generation == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(relay,0,sizeof(*relay));
	relay->vocab_size = vocab_size;
	relay->engine_generation = engine_generation;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationRelayDraftIssue(SparkSpeculationRelayDraft *relay,uint64_t sequence_id,uint64_t anchor_position,uint32_t anchor_token,const uint32_t *committed_token_ids,uint32_t committed_token_count,uint32_t requested_token_count,uint8_t bytes[SPARK_SPECULATION_RELAY_FRAME_BYTES])
{
	SparkSpeculationRelayFrame frame;
	SparkStatus status;
	if ( relay == 0 || relay->engine_generation == 0u || committed_token_ids == 0 || committed_token_count == 0u || committed_token_count > SPARK_SPECULATION_RELAY_MAX_TOKENS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&frame,0,sizeof(frame));
	frame.kind = SPARK_SPECULATION_RELAY_KIND_REQUEST;
	frame.engine_generation = relay->engine_generation;
	frame.round_id = relay->round_id + 1u;
	frame.sequence_id = sequence_id;
	frame.anchor_position = anchor_position;
	frame.anchor_token = anchor_token;
	frame.token_count = committed_token_count;
	frame.requested_token_count = requested_token_count;
	memcpy(frame.token_ids,committed_token_ids,(size_t)committed_token_count * sizeof(uint32_t));
	status = SparkSpeculationRelayEncode(&frame,relay->vocab_size,bytes);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	relay->round_id = frame.round_id;
	relay->pending = frame;
	relay->ready_valid = 0u;
	relay->issued++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationRelayDraftDeliver(SparkSpeculationRelayDraft *relay,const uint8_t *bytes,uint32_t length)
{
	SparkSpeculationRelayFrame frame;
	SparkStatus status;
	if ( relay == 0 || relay->engine_generation == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkSpeculationRelayDecode(bytes,length,relay->vocab_size,&frame);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( frame.kind != SPARK_SPECULATION_RELAY_KIND_DRAFT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( frame.engine_generation != relay->engine_generation || frame.round_id != relay->round_id || relay->issued == 0u ||
		frame.sequence_id != relay->pending.sequence_id || frame.anchor_position != relay->pending.anchor_position ||
		frame.anchor_token != relay->pending.anchor_token || frame.requested_token_count != relay->pending.requested_token_count )
	{
		relay->stale++;
		return(SPARK_STATUS_OK);
	}
	relay->ready = frame;
	relay->ready_valid = 1u;
	relay->delivered++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkSpeculationRelayDraftTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result)
{
	SparkSpeculationRelayDraft *relay = (SparkSpeculationRelayDraft *)context;
	uint32_t count,index;
	if ( relay == 0 || request == 0 || result == 0 || request->requested_token_count == 0u || request->requested_token_count > SPARK_SPECULATION_MAX_SPECULATIVE_TOKEN_COUNT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	result->token_count = 0u;
	if ( relay->ready_valid == 0u || relay->ready.sequence_id != request->sequence_id || relay->ready.anchor_position != request->sequence_position )
	{
		relay->misses++;
		relay->ready_valid = 0u;
		return(SPARK_STATUS_OK);
	}
	count = relay->ready.token_count < request->requested_token_count ? relay->ready.token_count : request->requested_token_count;
	for (index=0u; index<count; index++)
	{
		result->token_ids[index] = relay->ready.token_ids[index];
		result->confidence_milli[index] = SPARK_SPECULATION_CONFIDENCE_MILLI_ONE;
	}
	result->token_count = count;
	relay->ready_valid = 0u;
	relay->used++;
	return(SPARK_STATUS_OK);
}
