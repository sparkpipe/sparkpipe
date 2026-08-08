// The engine, driven: two requests through admission, chunked prefill, mixed
// prefill+decode steps, EOS, and slot reuse by a third. Prints every plan so
// the python gate can hold the scheduler to its contract - the same contract
// the slice kernels enforce on their side: rows sorted by sequence, positions
// ascending in a run, context_length counting every stored row.
//
// A second scenario drives the serving bugs of the main-8 audit on the same
// (now idle) engine: a one-token prompt (K3-006), more sequential submits
// than the record capacity (K3-007), out-of-bounds / duplicated / stale
// commits that must fail closed (K3-008), and a draft whose accepted middle
// token is EOS (K3-009). A third scenario puts three requests on a
// three-slot two-row engine so full decode lanes would starve the third
// request's prefill without the fairness pass (K3-010).

#include <stdio.h>
#include "inference/llms/kimi_k3/engine.h"

float state_s[65536u / sizeof(float)];

#define SLOTS 2u
#define BUDGET 4u
#define STEP_MAX 16u

static void PrintStep(uint32_t index, const struct K3EngineStep *step)
{
	uint32_t s,r;
	printf("step %u rows %u sequences %u\n", index, step->rows, step->sequences);
	for (s = 0u; s < step->sequences; ++s)
	{
		printf("  seq %u slot %u request %llu run %u..%u context %u logits %d\n",
			s, step->slot[s], (unsigned long long)step->request_id[s],
			step->sequence_row_begin[s], step->sequence_row_begin[s + 1u],
			step->context_length[s],
			step->logits_row[s] == K3_ENGINE_NO_LOGITS
				? -1 : (int)step->logits_row[s]);
		for (r = step->sequence_row_begin[s]; r < step->sequence_row_begin[s + 1u]; ++r)
			printf("    row %u token %u position %u slot %u\n",
				r, step->token[r], step->position[r], step->sequence_of_row[r]);
	}
}

int main(void)
{
	static struct K3Engine engine;
	static struct K3EngineRequest requests[4];
	static uint32_t slots[SLOTS];
	static uint32_t token[STEP_MAX], position[STEP_MAX], seq_of_row[STEP_MAX];
	static uint32_t run_begin[SLOTS + 1u], slot[SLOTS], context[SLOTS];
	static uint32_t logits[SLOTS];
	static uint64_t request_id[SLOTS];
	static uint32_t prompt_a[5] = { 11u, 12u, 13u, 14u, 15u };
	static uint32_t prompt_b[3] = { 21u, 22u, 23u };
	static uint32_t prompt_c[2] = { 31u, 32u };
	static uint32_t out_a[6], out_b[4], out_c[2];
	static uint32_t sampled[SLOTS];
	struct K3EngineStep step;
	uint32_t index,s,next = 100u;
	int32_t rows;
	memset(&step, 0, sizeof(step));
	step.token = token; step.position = position; step.sequence_of_row = seq_of_row;
	step.sequence_row_begin = run_begin; step.slot = slot; step.request_id = request_id;
	step.context_length = context; step.logits_row = logits;
	if ( K3EngineInit(&engine, requests, 4u, slots, SLOTS, BUDGET) != K3_ENGINE_OK )
		return 1;
	printf("submit a %lld\n", (long long)K3EngineSubmit(&engine, prompt_a, 5u, 6u, out_a));
	printf("submit b %lld\n", (long long)K3EngineSubmit(&engine, prompt_b, 3u, 4u, out_b));
	// The third arrives before any slot frees: it must queue, then take the
	// first slot a finished request abandons.
	printf("submit c %lld\n", (long long)K3EngineSubmit(&engine, prompt_c, 2u, 2u, out_c));
	for (index = 0u; index < 12u; ++index)
	{
		// After request a's first sampled token, hand it a three-token draft:
		// the next plan must be a verify-only step - one run of four rows at
		// ascending positions, logits at the run's head - and its resolution
		// (two accepted plus the bonus) must land exactly three tokens.
		if ( index == 3u )
		{
			static uint32_t draft[3] = { 201u, 202u, 203u };
			if ( K3EngineSubmitDraft(&engine, 1u, draft, 3u) != K3_ENGINE_OK )
				return 4;
		}
		rows = K3EnginePlanStep(&engine, &step);
		if ( rows < 0 )
			return 2;
		if ( rows == 0 )
		{
			printf("idle at %u\n", index);
			break;
		}
		PrintStep(index, &step);
		if ( step.verify != 0u )
		{
			static uint32_t accepted[SLOTS], bonus[SLOTS];
			printf("verify step %u\n", index);
			for (s = 0u; s < step.sequences; ++s)
			{
				accepted[s] = 2u;
				bonus[s] = next++;
				// The bonus token occupies the first rejected position and
				// was never forwarded; the next decode row re-runs it there.
				printf("verify_next %llu %u\n",
					(unsigned long long)step.request_id[s],
					step.position[step.sequence_row_begin[s]] + 1u + accepted[s]);
			}
			if ( K3EngineCommitVerify(&engine, &step, accepted, bonus, 7u) != K3_ENGINE_OK )
				return 5;
			continue;
		}
		for (s = 0u; s < step.sequences; ++s)
		{
			sampled[s] = 0u;
			if ( step.logits_row[s] == K3_ENGINE_NO_LOGITS )
				continue;
			// Request b's second token is EOS, ending it under budget; the
			// rest count up so the transcript shows who sampled what.
			sampled[s] = (step.request_id[s] == 2u
				&& engine.requests[engine.slot_request[step.slot[s]]].generated == 1u)
				? 7u : next++;
		}
		if ( K3EngineCommitStep(&engine, &step, sampled, 7u) != K3_ENGINE_OK )
			return 3;
	}
	printf("out_a %u %u %u %u %u %u\n", out_a[0], out_a[1], out_a[2], out_a[3], out_a[4], out_a[5]);
	printf("out_b %u %u\n", out_b[0], out_b[1]);
	printf("out_c %u %u\n", out_c[0], out_c[1]);

	// --- The serving-bug scenarios, on the same engine now idle. -----------
	printf("scenario serving_bugs\n");
	// K3-006: a one-token prompt opens straight in DECODE - one row at
	// position 0 with context 1 and logits on the row, never a zero-row
	// prefill chunk whose context arithmetic underflowed to UINT32_MAX.
	{
		static uint32_t prompt_d[1] = { 41u };
		static uint32_t out_d[2];
		printf("submit d %lld\n", (long long)K3EngineSubmit(&engine, prompt_d, 1u, 2u, out_d));
		rows = K3EnginePlanStep(&engine, &step);
		if ( rows != 1 )
			return 6;
		PrintStep(0u, &step);
		sampled[0] = 42u;
		if ( K3EngineCommitStep(&engine, &step, sampled, 7u) != K3_ENGINE_OK )
			return 7;
		rows = K3EnginePlanStep(&engine, &step);
		if ( rows != 1 )
			return 8;
		PrintStep(1u, &step);
		sampled[0] = 7u;
		if ( K3EngineCommitStep(&engine, &step, sampled, 7u) != K3_ENGINE_OK )
			return 9;
		printf("out_d %u %u\n", out_d[0], out_d[1]);
	}
	// K3-007: four requests finished above on a four-record engine. Four
	// more must submit - proof the finished records came back to FREE - and
	// the fifth must see the capacity wall.
	{
		static uint32_t prompt_e[2] = { 81u, 82u };
		static uint32_t prompt_f[2] = { 91u, 92u };
		static uint32_t prompt_g[2] = { 95u, 96u };
		static uint32_t prompt_h[2] = { 97u, 98u };
		static uint32_t out_e[4], out_f[2], out_g[2], out_h[2];
		int64_t id_e = K3EngineSubmit(&engine, prompt_e, 2u, 6u, out_e);
		int64_t id_f = K3EngineSubmit(&engine, prompt_f, 2u, 2u, out_f);
		int64_t id_g = K3EngineSubmit(&engine, prompt_g, 2u, 2u, out_g);
		int64_t id_h = K3EngineSubmit(&engine, prompt_h, 2u, 2u, out_h);
		int64_t id_i = K3EngineSubmit(&engine, prompt_f, 2u, 2u, out_f);
		printf("reuse e %lld f %lld g %lld h %lld full %lld\n",
			(long long)id_e, (long long)id_f, (long long)id_g,
			(long long)id_h, (long long)id_i);
		if ( id_e < 0 || id_f < 0 || id_g < 0 || id_h < 0
			|| id_i != K3_ENGINE_ERR_CAPACITY )
			return 10;
		// K3-009: e decodes with a draft whose middle accepted token is EOS.
		// The output must stop at that EOS - the draft past it and the bonus
		// belong to a sequence that is already over.
		rows = K3EnginePlanStep(&engine, &step);
		if ( rows <= 0 )
			return 11;
		PrintStep(2u, &step);
		for (s = 0u; s < step.sequences; ++s)
			sampled[s] = 0u;
		if ( K3EngineCommitStep(&engine, &step, sampled, 7u) != K3_ENGINE_OK )
			return 12;
		{
			static uint32_t draft_e[3] = { 301u, 7u, 303u };
			static uint32_t accepted[SLOTS], bonus[SLOTS];
			if ( K3EngineSubmitDraft(&engine, (uint64_t)id_e, draft_e, 3u) != K3_ENGINE_OK )
				return 13;
			rows = K3EnginePlanStep(&engine, &step);
			if ( rows <= 0 || step.verify == 0u )
				return 14;
			PrintStep(3u, &step);
			printf("verify step 3\n");
			for (s = 0u; s < step.sequences; ++s)
			{
				accepted[s] = 3u;
				bonus[s] = 999u;
				printf("verify_next %llu %u\n",
					(unsigned long long)step.request_id[s],
					step.position[step.sequence_row_begin[s]] + 1u + accepted[s]);
			}
			if ( K3EngineCommitVerify(&engine, &step, accepted, bonus, 7u) != K3_ENGINE_OK )
				return 15;
			printf("out_e %u %u %u %u\n", out_e[0], out_e[1], out_e[2], out_e[3]);
		}
		// K3-008: three bad commits must fail closed without moving the
		// machine - a duplicate of an already-committed step, a slot poked
		// out of bounds, and a step a newer plan has made stale.
		rows = K3EnginePlanStep(&engine, &step);
		if ( rows <= 0 )
			return 16;
		PrintStep(4u, &step);
		for (s = 0u; s < step.sequences; ++s)
			sampled[s] = 610u + s;
		if ( K3EngineCommitStep(&engine, &step, sampled, 7u) != K3_ENGINE_OK )
			return 17;
		printf("commit_dup %d\n", K3EngineCommitStep(&engine, &step, sampled, 7u));
		{
			struct K3EngineStep stale;
			static uint32_t st_token[STEP_MAX], st_position[STEP_MAX], st_seq[STEP_MAX];
			static uint32_t st_begin[SLOTS + 1u], st_slot[SLOTS], st_context[SLOTS];
			static uint32_t st_logits[SLOTS];
			static uint64_t st_request[SLOTS];
			uint32_t saved;
			rows = K3EnginePlanStep(&engine, &step);
			if ( rows <= 0 )
				return 18;
			saved = step.slot[0];
			step.slot[0] = SLOTS;
			printf("commit_oob %d\n", K3EngineCommitStep(&engine, &step, sampled, 7u));
			step.slot[0] = saved;
			// Snapshot this step, then plan over it: the snapshot is stale.
			stale = step;
			memcpy(st_token, token, sizeof(st_token));
			memcpy(st_position, position, sizeof(st_position));
			memcpy(st_seq, seq_of_row, sizeof(st_seq));
			memcpy(st_begin, run_begin, sizeof(st_begin));
			memcpy(st_slot, slot, sizeof(st_slot));
			memcpy(st_context, context, sizeof(st_context));
			memcpy(st_logits, logits, sizeof(st_logits));
			memcpy(st_request, request_id, sizeof(st_request));
			stale.token = st_token;
			stale.position = st_position;
			stale.sequence_of_row = st_seq;
			stale.sequence_row_begin = st_begin;
			stale.slot = st_slot;
			stale.context_length = st_context;
			stale.logits_row = st_logits;
			stale.request_id = st_request;
			rows = K3EnginePlanStep(&engine, &step);
			if ( rows <= 0 )
				return 19;
			printf("commit_stale %d\n", K3EngineCommitStep(&engine, &stale, sampled, 7u));
			// The machine is unharmed: the current step still commits.
			PrintStep(5u, &step);
			for (s = 0u; s < step.sequences; ++s)
				sampled[s] = 620u + s;
			if ( K3EngineCommitStep(&engine, &step, sampled, 7u) != K3_ENGINE_OK )
				return 20;
		}
		// Drain f, g and h so the transcript ends on an idle engine.
		for (index = 6u; index < 24u; ++index)
		{
			rows = K3EnginePlanStep(&engine, &step);
			if ( rows < 0 )
				return 21;
			if ( rows == 0 )
			{
				printf("idle at %u\n", index);
				break;
			}
			PrintStep(index, &step);
			for (s = 0u; s < step.sequences; ++s)
			{
				sampled[s] = 0u;
				if ( step.logits_row[s] == K3_ENGINE_NO_LOGITS )
					continue;
				sampled[s] = 630u + index + s;
			}
			if ( K3EngineCommitStep(&engine, &step, sampled, 7u) != K3_ENGINE_OK )
				return 22;
		}
		printf("out_f %u %u\n", out_f[0], out_f[1]);
		printf("out_g %u %u\n", out_g[0], out_g[1]);
		printf("out_h %u %u\n", out_h[0], out_h[1]);
	}

	// --- K3-010: full decode lanes cannot starve prefill. ------------------
	// Three slots but a two-row budget: once j and k decode, their rows fill
	// every pass, and l's prefill would never advance without the fairness
	// pass. l has a six-token prompt so its prefill needs several chunks.
	printf("scenario fairness\n");
	{
		static struct K3Engine engine2;
		static struct K3EngineRequest requests2[4];
		static uint32_t slots2[3];
		static uint32_t token2[STEP_MAX], position2[STEP_MAX], seq2[STEP_MAX];
		static uint32_t begin2[4], slot2[3], context2[3], logits2[3];
		static uint64_t request2[3];
		static uint32_t prompt_j[2] = { 51u, 52u };
		static uint32_t prompt_k[2] = { 61u, 62u };
		static uint32_t prompt_l[6] = { 71u, 72u, 73u, 74u, 75u, 76u };
		static uint32_t out_j[30], out_k[30], out_l[2];
		static uint32_t sampled2[3];
		struct K3EngineStep step2;
		uint32_t t = 700u;
		memset(&step2, 0, sizeof(step2));
		step2.token = token2; step2.position = position2; step2.sequence_of_row = seq2;
		step2.sequence_row_begin = begin2; step2.slot = slot2; step2.request_id = request2;
		step2.context_length = context2; step2.logits_row = logits2;
		if ( K3EngineInit(&engine2, requests2, 4u, slots2, 3u, 2u) != K3_ENGINE_OK )
			return 23;
		printf("submit j %lld\n", (long long)K3EngineSubmit(&engine2, prompt_j, 2u, 30u, out_j));
		printf("submit k %lld\n", (long long)K3EngineSubmit(&engine2, prompt_k, 2u, 30u, out_k));
		printf("submit l %lld\n", (long long)K3EngineSubmit(&engine2, prompt_l, 6u, 2u, out_l));
		for (index = 0u; index < 10u; ++index)
		{
			rows = K3EnginePlanStep(&engine2, &step2);
			if ( rows < 0 )
				return 24;
			if ( rows == 0 )
			{
				printf("idle at %u\n", index);
				break;
			}
			PrintStep(index, &step2);
			for (s = 0u; s < step2.sequences; ++s)
			{
				sampled2[s] = 0u;
				if ( step2.logits_row[s] == K3_ENGINE_NO_LOGITS )
					continue;
				sampled2[s] = t++;
			}
			if ( K3EngineCommitStep(&engine2, &step2, sampled2, 7u) != K3_ENGINE_OK )
				return 25;
		}
	}
	return 0;
}
