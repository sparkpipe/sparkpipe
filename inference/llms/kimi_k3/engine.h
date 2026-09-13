#pragma once

// The K3 serving engine: requests in, slice steps out.
//
// This is the host half of actual usage - accept a request, hold it in a
// queue, admit it when a sequence slot frees, cut its prompt into chunks that
// share steps with everyone else's decode, and advance it token by token until
// EOS or its budget. It is pure planning: no CUDA, no allocation, no model.
// One call plans a step as the exact arrays K3StageSlice and the K3 buffers
// consume - rows sorted by sequence, sequence_row_begin a prefix, positions
// ascending within a run, context_length counting every stored row - and one
// call commits the step's sampled tokens and moves the machine forward. The
// driver owns the device copies and the sampler; the engine owns the truth
// about who is where.
//
// CONTINUOUS BATCHING IS THE POLICY, NOT A MODE. Every step carries every
// decoding sequence (one row each) and then spends whatever row budget
// remains on the oldest prefilling sequence's next chunk - except every
// K3_ENGINE_PREFILL_PERIOD-th pass, which plans the oldest prefiller first
// so full decode lanes cannot starve it forever (K3-010). A decode-only
// step and a prefill-only step are both just this rule with one side empty.
//
// Storage is the caller's, sized by the two capacity numbers in K3EngineInit,
// because a serving process knows its memory and a header does not. No malloc
// anywhere, per the house rule and because an allocator in the admission path
// is a latency cliff waiting for load.

#include <stdint.h>
#include <string.h>

#define K3_ENGINE_OK               0
#define K3_ENGINE_ERR_NULL       -70
#define K3_ENGINE_ERR_CAPACITY   -71
#define K3_ENGINE_ERR_PROMPT     -72
#define K3_ENGINE_ERR_STATE      -73
#define K3_ENGINE_ERR_SAMPLE     -74

// What a sequence is doing. VERIFY is DSpark's lane: a drafted block planned
// as a run the layer executes with commit off, so the state never learns what
// the sampler later rejects. The planner treats it as prefill-shaped rows
// whose tokens came from the drafter rather than the user. There is no DONE:
// a finished request's record returns to FREE in the same commit that frees
// its slot (K3-007) - the tokens already sit in the caller's output, and a
// record held past completion leaked capacity until no request could submit.
#define K3_SEQ_FREE      0u
#define K3_SEQ_QUEUED    1u
#define K3_SEQ_PREFILL   2u
#define K3_SEQ_DECODE    3u

// Prefill fairness period (K3-010): decode rows are planned first, so a
// deployment with slot_capacity >= row_budget would crowd the oldest
// prefiller out of every pass forever. Every Nth planning pass the oldest
// prefiller cuts ahead and the decode lanes take what budget remains.
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
	// DSpark: the drafted block awaiting verification. draft_count > 0 turns
	// the sequence's next planned step into a verify run - the last committed
	// token plus the drafts, positions ascending - which the driver executes
	// with commit off and resolves through K3EngineCommitVerify.
	uint32_t draft[8];
	uint32_t draft_count;
};

struct K3EngineStep
{
	// Per row, in plan order: the token to embed and its position.
	uint32_t *token;
	uint32_t *position;
	uint32_t *sequence_of_row;
	// Per sequence in plan order: the run prefix, the slot, the request, the
	// stored context after this step's rows land, and which row's logits the
	// sampler must read - K3_ENGINE_NO_LOGITS for a chunk that is not yet at
	// the prompt's end and therefore predicts nothing anyone keeps.
	uint32_t *sequence_row_begin;
	uint32_t *slot;
	uint64_t *request_id;
	uint32_t *context_length;
	uint32_t *logits_row;
	uint32_t rows;
	uint32_t sequences;
	// A verify step carries ONLY verify runs, because the layer's commit flag
	// is per slice call: mixing a committed decode with an uncommitted draft
	// in one call would either advance a draft or drop a token.
	uint32_t verify;
	// Generation of the plan that produced this step (K3-008): the commit
	// path refuses any step that is not the latest uncommitted plan, so a
	// stale or duplicated commit fails closed instead of indexing through a
	// slot that may have changed hands since the plan was made.
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
	// Plan generations: plan_epoch stamps every planned step, committed_epoch
	// remembers the newest one already committed. A commit whose epoch is not
	// the latest plan, or is one already committed, is stale or duplicated.
	uint32_t plan_epoch;
	uint32_t committed_epoch;
	// DSpark acceptance instrumentation (SURVEY_K3 #10): draft tokens offered
	// to verify commits and draft tokens those commits actually emitted. The
	// per-workload acceptance rate is accepted/proposed - a drafter lands
	// against this measurement, not a speedup hope. K3EngineInit zeroes both.
	uint64_t drafts_proposed;
	uint64_t drafts_accepted;
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

// Accept a request. The prompt pointer must outlive the request; the output
// array must hold max_new tokens. Returns the id, or a negative status.
// Hand a sequence its drafted block. Legal only while it is decoding; the
// next plan turns it into a verify run.
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

// Admission: the oldest queued request takes the lowest free slot. Called by
// the planner so a slot freed by this step's commit is refilled by the next
// plan, never left idle while the queue waits.
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
		// A one-token prompt has nothing to prefill (K3-006): its single
		// token is the decode row that predicts the first new token, so it
		// starts in DECODE rather than opening with a zero-row chunk.
		oldest->state = oldest->prompt_length == 1u ? K3_SEQ_DECODE : K3_SEQ_PREFILL;
		engine->slot_request[slot] = (uint32_t)(oldest - engine->requests);
	}
}

// One sequence's contribution to the plan: DECODE is one row carrying the
// last sampled (or last prompt) token; PREFILL is the next chunk of the
// prompt. Rows land contiguously, positions ascend, and only a run that
// reaches the prompt's end asks for logits.
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
		// The final prompt token belongs to decode: its forward is the one
		// that predicts the first new token, so prefill stops one short and
		// the decode row carries it. A prompt of one token prefills nothing.
		count = count > 0u ? count - 1u : 0u;
		count = count > budget ? budget : count;
		for (index = 0u; index < count; ++index)
		{
			step->token[base + index] = request->prompt[request->prefilled + index];
			step->position[base + index] = request->prefilled + index;
		}
		step->logits_row[sequence] = K3_ENGINE_NO_LOGITS;
		// The stored context is the rows landed plus this chunk, counted
		// directly: deriving it as last position + 1 underflowed to
		// UINT32_MAX whenever the chunk was empty (K3-006).
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

// Plan one step: admit, then every decoding sequence, then the oldest
// prefilling sequence's next chunk into whatever budget remains - except
// every K3_ENGINE_PREFILL_PERIOD-th pass, where the oldest prefiller is
// planned first so full decode lanes cannot starve it (K3-010). Returns the
// row count; zero means nothing to do.
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
	// Verify first, alone: the run is the last committed token plus the
	// drafts at ascending positions, and the whole step runs with commit off.
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
	// K3-010: with decode rows planned ahead of prefill, enough decoding
	// sequences would consume the whole budget every pass and the oldest
	// prefiller would never advance. Every Nth pass it cuts ahead instead.
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

// Resolve the request a planned sequence commits against, checking every
// link in the slot-to-request chain (K3-008): the step must be the latest
// uncommitted plan, the slot must be in bounds and occupied, and the
// occupant must be the very request the plan named - a record recycled since
// the plan carries a different id, so the generation check is the identity
// check. Any doubt fails closed with no request resolved.
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

// A finished request frees its slot AND its record in the same commit
// (K3-007): the tokens already sit in the caller's output array, so nothing
// is lost, and a record held past completion leaked capacity until
// sequential requests exhausted it.
static void K3EngineFinishRequest(struct K3Engine *engine, struct K3EngineRequest *request)
{
	request->state = K3_SEQ_FREE;
	engine->slot_request[request->slot] = 0xffffffffu;
}

// Commit the step: sampled[sequence] must hold a token for every sequence
// whose logits_row was real, and is ignored for the rest. eos ends a request
// early; the budget ends it on time; either way the slot frees for the next
// admission.
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

// Resolve a verify step: accepted[sequence] drafts survived and
// bonus[sequence] is the corrected token the target sampled at the first
// rejected position (or after the last draft). The KDA fold is the driver's
// job before the next step; the engine only moves the token record.
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
		// An accepted draft can itself be EOS (K3-009): emit tokens up to
		// and including it, then stop - the drafts past it and the bonus
		// belong to a sequence that is already over.
		ended = 0u;
		for (r = 0u; r < accepted[sequence] && request->generated < request->max_new; ++r)
		{
			token = request->draft[r];
			request->output[request->generated++] = token;
			engine->drafts_accepted += 1u;
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
		// Proposed counts every draft the verify offered, accepted only the
		// ones emitted above - the honest numerator and denominator of the
		// acceptance rate (SURVEY_K3 #10).
		engine->drafts_proposed += request->draft_count;
		request->draft_count = 0u;
		if ( ended != 0u || request->generated >= request->max_new )
			K3EngineFinishRequest(engine,request);
	}
	engine->committed_epoch = step->epoch;
	return(K3_ENGINE_OK);
}
