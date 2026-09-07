#pragma once


#include <stdint.h>
#include <string.h>

#define K3_ENGINE_OK               0
#define K3_ENGINE_ERR_NULL       -70
#define K3_ENGINE_ERR_CAPACITY   -71
#define K3_ENGINE_ERR_PROMPT     -72
#define K3_ENGINE_ERR_STATE      -73
#define K3_ENGINE_ERR_SAMPLE     -74

#define K3_SEQ_FREE      0u
#define K3_SEQ_QUEUED    1u
#define K3_SEQ_PREFILL   2u
#define K3_SEQ_DECODE    3u

#define K3_ENGINE_PREFILL_PERIOD 4u

struct K3EngineRequest
{
	uint64_t id;
	const uint32_t *prompt;
	uint32_t prompt_length;
	uint32_t max_new;
	uint32_t generated;
	uint32_t prefilled;
	uint32_t slot;
	uint32_t state;
	uint32_t *output;
	uint32_t draft[8];
	uint32_t draft_count;
};

struct K3EngineStep
{
	uint32_t *token;
	uint32_t *position;
	uint32_t *sequence_of_row;
	uint32_t *sequence_row_begin;
	uint32_t *slot;
	uint64_t *request_id;
	uint32_t *context_length;
	uint32_t *logits_row;
	uint32_t rows;
	uint32_t sequences;
	uint32_t verify;
	uint32_t epoch;
};

#define K3_ENGINE_NO_LOGITS 0xffffffffu

struct K3Engine
{
	struct K3EngineRequest *requests;
	uint32_t request_capacity;
	uint32_t slot_capacity;
	uint32_t row_budget;
	uint64_t next_id;
	uint32_t *slot_request;
	uint32_t plan_epoch;
	uint32_t committed_epoch;
};

static int32_t K3EngineInit(struct K3Engine *engine, struct K3EngineRequest *request_storage, uint32_t request_capacity, uint32_t *slot_storage, uint32_t slot_capacity, uint32_t row_budget)
{
	uint32_t index;
	if ( engine == 0 || request_storage == 0 || slot_storage == 0 )
		return(K3_ENGINE_ERR_NULL);
	if ( request_capacity == 0u || slot_capacity == 0u || row_budget == 0u )
		return(K3_ENGINE_ERR_CAPACITY);
	memset(engine,0,sizeof(*engine));
	memset(request_storage,0,(size_t)request_capacity * sizeof(*request_storage));
	engine->requests = request_storage;
	engine->request_capacity = request_capacity;
	engine->slot_capacity = slot_capacity;
	engine->row_budget = row_budget;
	engine->next_id = 1u;
	engine->slot_request = slot_storage;
	for (index = 0u; index < slot_capacity; ++index)
		slot_storage[index] = 0xffffffffu;
	return(K3_ENGINE_OK);
}

static int32_t K3EngineSubmitDraft(struct K3Engine *engine, uint64_t id, const uint32_t *draft, uint32_t count)
{
	uint32_t index;
	if ( engine == 0 || draft == 0 || count == 0u || count > 7u )
		return(K3_ENGINE_ERR_NULL);
	for (index = 0u; index < engine->request_capacity; ++index)
		if ( engine->requests[index].id == id
			&& engine->requests[index].state == K3_SEQ_DECODE )
		{
			memcpy(engine->requests[index].draft,draft,(size_t)count * sizeof(*draft));
			engine->requests[index].draft_count = count;
			return(K3_ENGINE_OK);
		}
	return(K3_ENGINE_ERR_STATE);
}

static int64_t K3EngineSubmit(struct K3Engine *engine, const uint32_t *prompt, uint32_t prompt_length, uint32_t max_new, uint32_t *output)
{
	struct K3EngineRequest *request = 0;
	uint32_t index;
	if ( engine == 0 || prompt == 0 || output == 0 )
		return(K3_ENGINE_ERR_NULL);
	if ( prompt_length == 0u || max_new == 0u )
		return(K3_ENGINE_ERR_PROMPT);
	for (index = 0u; index < engine->request_capacity; ++index)
		if ( engine->requests[index].state == K3_SEQ_FREE )
		{
			request = &engine->requests[index];
			break;
		}
	if ( request == 0 )
		return(K3_ENGINE_ERR_CAPACITY);
	memset(request,0,sizeof(*request));
	request->id = engine->next_id++;
	request->prompt = prompt;
	request->prompt_length = prompt_length;
	request->max_new = max_new;
	request->state = K3_SEQ_QUEUED;
	request->slot = 0xffffffffu;
	request->output = output;
	return((int64_t)request->id);
}

static void K3EngineAdmit(struct K3Engine *engine)
{
	struct K3EngineRequest *oldest;
	uint32_t slot,index;
	for (slot = 0u; slot < engine->slot_capacity; ++slot)
	{
		if ( engine->slot_request[slot] != 0xffffffffu )
			continue;
		oldest = 0;
		for (index = 0u; index < engine->request_capacity; ++index)
			if ( engine->requests[index].state == K3_SEQ_QUEUED
				&& (oldest == 0 || engine->requests[index].id < oldest->id) )
				oldest = &engine->requests[index];
		if ( oldest == 0 )
			return;
		oldest->slot = slot;
		oldest->state = oldest->prompt_length == 1u ? K3_SEQ_DECODE : K3_SEQ_PREFILL;
		engine->slot_request[slot] = (uint32_t)(oldest - engine->requests);
	}
}

static uint32_t K3EnginePlanSequence(const struct K3EngineRequest *request, struct K3EngineStep *step, uint32_t budget)
{
	uint32_t sequence = step->sequences,base = step->rows,count,index,position,context;
	if ( request->state == K3_SEQ_DECODE )
	{
		count = 1u;
		position = request->prompt_length + request->generated - 1u;
		step->token[base] = request->generated == 0u
			? request->prompt[request->prompt_length - 1u]
			: request->output[request->generated - 1u];
		step->position[base] = position;
		step->logits_row[sequence] = base;
		context = position + 1u;
	}
	else
	{
		count = request->prompt_length - request->prefilled;
		count = count > 0u ? count - 1u : 0u;
		count = count > budget ? budget : count;
		for (index = 0u; index < count; ++index)
		{
			step->token[base + index] = request->prompt[request->prefilled + index];
			step->position[base + index] = request->prefilled + index;
		}
		step->logits_row[sequence] = K3_ENGINE_NO_LOGITS;
		context = request->prefilled + count;
	}
	for (index = 0u; index < count; ++index)
		step->sequence_of_row[base + index] = request->slot;
	step->sequence_row_begin[sequence + 1u] = base + count;
	step->slot[sequence] = request->slot;
	step->request_id[sequence] = request->id;
	step->context_length[sequence] = context;
	step->rows = base + count;
	step->sequences = sequence + 1u;
	return(count);
}

static int32_t K3EnginePlanStep(struct K3Engine *engine, struct K3EngineStep *step)
{
	struct K3EngineRequest *request,*oldest;
	uint32_t index,prefill_first;
	if ( engine == 0 || step == 0 )
		return(K3_ENGINE_ERR_NULL);
	K3EngineAdmit(engine);
	step->rows = 0u;
	step->sequences = 0u;
	step->verify = 0u;
	step->sequence_row_begin[0] = 0u;
	step->epoch = ++engine->plan_epoch;
	for (index = 0u; index < engine->request_capacity; ++index)
	{
		request = &engine->requests[index];
		if ( request->state != K3_SEQ_DECODE || request->draft_count == 0u )
			continue;
		if ( step->rows + request->draft_count + 1u > engine->row_budget )
			continue;
		uint32_t base = step->rows,sequence = step->sequences,r;
		uint32_t position = request->prompt_length + request->generated - 1u;
		step->token[base] = request->generated == 0u
			? request->prompt[request->prompt_length - 1u]
			: request->output[request->generated - 1u];
		step->position[base] = position;
		for (r = 0u; r < request->draft_count; ++r)
		{
			step->token[base + 1u + r] = request->draft[r];
			step->position[base + 1u + r] = position + 1u + r;
		}
		for (r = 0u; r < request->draft_count + 1u; ++r)
			step->sequence_of_row[base + r] = request->slot;
		step->sequence_row_begin[sequence + 1u] = base + request->draft_count + 1u;
		step->slot[sequence] = request->slot;
		step->request_id[sequence] = request->id;
		step->context_length[sequence] = position + request->draft_count + 1u;
		step->logits_row[sequence] = base;
		step->rows = base + request->draft_count + 1u;
		step->sequences = sequence + 1u;
		step->verify = 1u;
	}
	if ( step->verify != 0u )
		return((int32_t)step->rows);
	oldest = 0;
	for (index = 0u; index < engine->request_capacity; ++index)
	{
		request = &engine->requests[index];
		if ( request->state == K3_SEQ_PREFILL
			&& (oldest == 0 || request->id < oldest->id) )
			oldest = request;
	}
	prefill_first = oldest != 0
		&& engine->plan_epoch % K3_ENGINE_PREFILL_PERIOD == 1u;
	if ( prefill_first != 0u )
		K3EnginePlanSequence(oldest,step,engine->row_budget);
	for (index = 0u; index < engine->request_capacity; ++index)
	{
		request = &engine->requests[index];
		if ( request->state == K3_SEQ_DECODE && step->rows < engine->row_budget )
			K3EnginePlanSequence(request,step,1u);
	}
	if ( oldest != 0 && prefill_first == 0u && step->rows < engine->row_budget )
		K3EnginePlanSequence(oldest,step,engine->row_budget - step->rows);
	return((int32_t)step->rows);
}

static struct K3EngineRequest *K3EngineCommitRequest(struct K3Engine *engine, const struct K3EngineStep *step, uint32_t sequence)
{
	struct K3EngineRequest *request;
	uint32_t slot;
	if ( step->epoch != engine->plan_epoch
		|| step->epoch == engine->committed_epoch )
		return(0);
	slot = step->slot[sequence];
	if ( slot >= engine->slot_capacity
		|| engine->slot_request[slot] == 0xffffffffu )
		return(0);
	request = &engine->requests[engine->slot_request[slot]];
	if ( request->id != step->request_id[sequence] || request->slot != slot )
		return(0);
	return(request);
}

static void K3EngineFinishRequest(struct K3Engine *engine, struct K3EngineRequest *request)
{
	request->state = K3_SEQ_FREE;
	engine->slot_request[request->slot] = 0xffffffffu;
}

static int32_t K3EngineCommitStep(struct K3Engine *engine, const struct K3EngineStep *step, const uint32_t *sampled, uint32_t eos_token)
{
	struct K3EngineRequest *request;
	uint32_t sequence,token;
	if ( engine == 0 || step == 0 || sampled == 0 )
		return(K3_ENGINE_ERR_NULL);
	for (sequence = 0u; sequence < step->sequences; ++sequence)
	{
		request = K3EngineCommitRequest(engine,step,sequence);
		if ( request == 0 )
			return(K3_ENGINE_ERR_STATE);
		if ( request->state == K3_SEQ_PREFILL )
		{
			request->prefilled += step->sequence_row_begin[sequence + 1u]
				- step->sequence_row_begin[sequence];
			if ( request->prefilled + 1u >= request->prompt_length )
				request->state = K3_SEQ_DECODE;
			continue;
		}
		if ( request->state != K3_SEQ_DECODE )
			return(K3_ENGINE_ERR_STATE);
		if ( step->logits_row[sequence] == K3_ENGINE_NO_LOGITS )
			return(K3_ENGINE_ERR_SAMPLE);
		token = sampled[sequence];
		request->output[request->generated++] = token;
		if ( token == eos_token || request->generated >= request->max_new )
			K3EngineFinishRequest(engine,request);
	}
	engine->committed_epoch = step->epoch;
	return(K3_ENGINE_OK);
}

static int32_t K3EngineCommitVerify(struct K3Engine *engine, const struct K3EngineStep *step, const uint32_t *accepted, const uint32_t *bonus, uint32_t eos_token)
{
	struct K3EngineRequest *request;
	uint32_t sequence,r,token,ended;
	if ( engine == 0 || step == 0 || accepted == 0 || bonus == 0 || step->verify == 0u )
		return(K3_ENGINE_ERR_NULL);
	for (sequence = 0u; sequence < step->sequences; ++sequence)
	{
		request = K3EngineCommitRequest(engine,step,sequence);
		if ( request == 0 )
			return(K3_ENGINE_ERR_STATE);
		if ( request->state != K3_SEQ_DECODE || accepted[sequence] > request->draft_count )
			return(K3_ENGINE_ERR_STATE);
		ended = 0u;
		for (r = 0u; r < accepted[sequence] && request->generated < request->max_new; ++r)
		{
			token = request->draft[r];
			request->output[request->generated++] = token;
			if ( token == eos_token )
			{
				ended = 1u;
				break;
			}
		}
		if ( ended == 0u && request->generated < request->max_new )
		{
			token = bonus[sequence];
			request->output[request->generated++] = token;
			if ( token == eos_token )
				ended = 1u;
		}
		request->draft_count = 0u;
		if ( ended != 0u || request->generated >= request->max_new )
			K3EngineFinishRequest(engine,request);
	}
	engine->committed_epoch = step->epoch;
	return(K3_ENGINE_OK);
}
