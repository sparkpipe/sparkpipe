#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm5_next_verify_regime.h"
#include "sparkpipe/spark_speculation_depth.h"
#include "sparkpipe/spark_speculation_drafter_mix.h"
#include "sparkpipe/spark_speculation_lookup_draft.h"
#include "sparkpipe/spark_speculation_policy.h"

#define REPLAY_VOCAB 154880u
#define REPLAY_MAX_POSITIONS 1048576u
#define REPLAY_FRAME_MAX 32u
#define REPLAY_PREFILL_WAVE_ROWS 128u

typedef struct ReplaySynthetic
{
	const uint32_t *truth;
	uint32_t length;
	uint32_t accept_milli;
	uint64_t seed;
	uint32_t available;
	uint64_t calls;
} ReplaySynthetic;

typedef struct ReplayConfig
{
	uint32_t rows;
	uint32_t frame;
	uint32_t block;
	uint32_t split;
	uint32_t mode;
	uint32_t accept_milli;
	uint32_t fixed_depth;
	uint64_t seed;
} ReplayConfig;

typedef struct ReplayCounts
{
	uint64_t frames;
	uint64_t verify_frames;
	uint64_t plain_frames;
	uint64_t plain_frame_tokens;
	uint64_t rounds;
	uint64_t proposed;
	uint64_t accepted;
	uint64_t plain_steps;
	uint64_t tokens;
	uint64_t rows_histogram[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX + 1u];
	uint64_t source_rounds[2];
	uint64_t source_proposed[2];
	uint64_t source_accepted[2];
	uint64_t synthetic_calls;
} ReplayCounts;

enum
{
	REPLAY_MODE_LOOKUP = 0,
	REPLAY_MODE_SYNTHETIC = 1,
	REPLAY_MODE_MIX = 2
};

static uint64_t ReplayMix64(uint64_t value)
{
	value += UINT64_C(0x9e3779b97f4a7c15);
	value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
	value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
	return(value ^ (value >> 31));
}

static SparkStatus ReplaySyntheticTokens(void *context,const SparkSpeculationPolicyDraftRequest *request,SparkSpeculationPolicyDraftResult *result)
{
	ReplaySynthetic *synthetic = (ReplaySynthetic *)context;
	uint64_t anchor = request->sequence_position;
	uint32_t index;
	result->token_count = 0u;
	if ( synthetic->available == 0u )
		return(SPARK_STATUS_NOT_FOUND);
	synthetic->calls++;
	for (index=0u; index<request->requested_token_count && anchor + 1u + index < synthetic->length; index++)
	{
		uint32_t truth = synthetic->truth[anchor + 1u + index];
		uint64_t draw = ReplayMix64(synthetic->seed ^ ((anchor + 1u + index) << 8) ^ index) % 1000u;
		result->token_ids[index] = draw < synthetic->accept_milli ? truth : (truth + 1u) % REPLAY_VOCAB;
	}
	result->token_count = index;
	return(index == 0u ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_OK);
}

static uint32_t *ReplayRead(const char *path,uint32_t *prompt_out,uint32_t *length_out)
{
	FILE *file = fopen(path,"rb");
	uint32_t head[2],*tokens;
	if ( file == 0 || fread(head,sizeof(uint32_t),2u,file) != 2u || head[0] == 0u || head[1] <= head[0] || head[1] > REPLAY_MAX_POSITIONS )
	{
		if ( file != 0 )
			fclose(file);
		return(0);
	}
	tokens = (uint32_t *)malloc((size_t)head[1] * sizeof(uint32_t));
	if ( tokens == 0 || fread(tokens,sizeof(uint32_t),head[1],file) != head[1] )
	{
		free(tokens);
		fclose(file);
		return(0);
	}
	fclose(file);
	*prompt_out = head[0];
	*length_out = head[1];
	return(tokens);
}

static uint32_t ReplayFrameSteps(const ReplayConfig *config,uint32_t committed,uint32_t length)
{
	uint32_t steps = config->frame,position = committed - 1u,block_remaining;
	if ( length - committed < steps )
		steps = length - committed;
	block_remaining = config->block - (position % config->block);
	if ( block_remaining < steps )
		steps = block_remaining;
	return(steps);
}

static int ReplayStream(const ReplayConfig *config,const uint32_t *truth,uint32_t prompt,uint32_t length,ReplayCounts *counts)
{
	SparkSpeculationLookupDraft lookup;
	SparkSpeculationDrafterMix mix;
	ReplaySynthetic synthetic;
	SparkSpeculationDraftFunction function;
	void *context;
	SparkSpeculationPolicyDraftRequest request;
	SparkSpeculationPolicyDraftResult result;
	SparkSpeculationPolicyVerifyResult verify;
	uint32_t verifier[SPARK_GLM5_NEXT_VERIFY_ROWS_MAX];
	uint32_t committed,budget,produced,depth,cap,index,source,anchor;
	SparkStatus status;
	memset(&synthetic,0,sizeof(synthetic));
	memset(&mix,0,sizeof(mix));
	synthetic.truth = truth;
	synthetic.length = length;
	synthetic.accept_milli = config->accept_milli;
	synthetic.seed = config->seed;
	synthetic.available = prompt <= REPLAY_PREFILL_WAVE_ROWS ? 1u : 0u;
	if ( SparkSpeculationLookupDraftInitialize(&lookup,1u,length + 1u,SPARK_GLM5_NEXT_VERIFY_LOOKUP_MIN_MATCH,SPARK_GLM5_NEXT_VERIFY_LOOKUP_MAX_MATCH) != SPARK_STATUS_OK )
		return(1);
	function = SparkSpeculationLookupDraftTokens;
	context = &lookup;
	if ( config->mode == REPLAY_MODE_SYNTHETIC )
	{
		function = ReplaySyntheticTokens;
		context = &synthetic;
	}
	if ( config->mode == REPLAY_MODE_MIX )
	{
		if ( SparkSpeculationDrafterMixInitialize(&mix,SparkSpeculationLookupDraftTokens,&lookup,ReplaySyntheticTokens,&synthetic,
			config->rows - 1u < SPARK_GLM5_NEXT_VERIFY_MIX_LOOKUP_MIN_TOKENS ? config->rows - 1u : SPARK_GLM5_NEXT_VERIFY_MIX_LOOKUP_MIN_TOKENS,1u,config->rows - 1u) != SPARK_STATUS_OK )
			return(1);
		function = SparkSpeculationDrafterMixTokens;
		context = &mix;
	}
	if ( SparkSpeculationLookupDraftObserve(&lookup,0u,1u,0u,truth,prompt + 1u) != SPARK_STATUS_OK )
		return(1);
	cap = config->rows - 1u;
	committed = prompt + 1u;
	counts->tokens += length - prompt;
	while ( committed < length )
	{
		budget = ReplayFrameSteps(config,committed,length);
		produced = 0u;
		counts->frames++;
		for (;;)
		{
			anchor = committed + produced - 1u;
			depth = SparkGlm5NextVerifyDepth(budget,produced,config->rows,anchor,config->split,REPLAY_MAX_POSITIONS);
			if ( config->mode != REPLAY_MODE_MIX && config->fixed_depth == 0u && depth > cap )
				depth = cap;
			result.token_count = 0u;
			status = SPARK_STATUS_NOT_FOUND;
			if ( depth != 0u )
			{
				memset(&request,0,sizeof(request));
				request.requested_token_count = depth;
				request.sequence_id = 1u;
				request.sequence_position = anchor;
				status = function(context,&request,&result);
			}
			if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND )
				return(1);
			if ( status == SPARK_STATUS_NOT_FOUND || result.token_count == 0u )
			{
				if ( produced == 0u )
				{
					counts->plain_frames++;
					counts->plain_frame_tokens += budget;
					if ( SparkSpeculationLookupDraftObserve(&lookup,0u,1u,committed,truth + committed,budget) != SPARK_STATUS_OK )
						return(1);
					produced = budget;
					break;
				}
				if ( SparkSpeculationLookupDraftObserve(&lookup,0u,1u,committed + produced,truth + committed + produced,1u) != SPARK_STATUS_OK )
					return(1);
				counts->plain_steps++;
				produced++;
				if ( produced >= budget )
					break;
				continue;
			}
			if ( produced == 0u )
				counts->verify_frames++;
			for (index=0u; index<=result.token_count; index++)
				verifier[index] = truth[anchor + 1u + index];
			if ( SparkSpeculationPolicyResolveVerifierTokens(result.token_ids,result.token_count,verifier,result.token_count + 1u,REPLAY_VOCAB,&verify) != SPARK_STATUS_OK )
				return(1);
			source = config->mode == REPLAY_MODE_MIX ? SparkSpeculationDrafterMixLastSource(&mix,0u) : (config->mode == REPLAY_MODE_SYNTHETIC ? 1u : 0u);
			if ( config->mode == REPLAY_MODE_MIX && SparkSpeculationDrafterMixObserve(&mix,0u,result.token_count,verify.accepted_draft_token_count) != SPARK_STATUS_OK )
				return(1);
			cap = SparkSpeculationDepthCapNext(cap,result.token_count,verify.accepted_draft_token_count,config->rows - 1u);
			if ( SparkSpeculationLookupDraftObserve(&lookup,0u,1u,anchor + 1u,truth + anchor + 1u,verify.committed_token_count) != SPARK_STATUS_OK )
				return(1);
			counts->rounds++;
			counts->proposed += result.token_count;
			counts->accepted += verify.accepted_draft_token_count;
			counts->rows_histogram[result.token_count + 1u]++;
			if ( source < 2u )
			{
				counts->source_rounds[source]++;
				counts->source_proposed[source] += result.token_count;
				counts->source_accepted[source] += verify.accepted_draft_token_count;
			}
			produced += verify.committed_token_count;
			if ( produced >= budget )
				break;
		}
		committed += produced;
		synthetic.available = 1u;
	}
	counts->synthetic_calls += synthetic.calls;
	SparkSpeculationDrafterMixDestroy(&mix);
	SparkSpeculationLookupDraftDestroy(&lookup);
	return(0);
}

static int ReplayParse(int argc,char **argv,ReplayConfig *config,int *first_file)
{
	int index;
	memset(config,0,sizeof(*config));
	config->rows = 8u;
	config->frame = 8u;
	config->block = 64u;
	config->split = 64u;
	config->seed = 1u;
	for (index=1; index<argc; index++)
	{
		if ( strcmp(argv[index],"--rows") == 0 && index + 1 < argc )
			config->rows = (uint32_t)strtoul(argv[++index],0,10);
		else if ( strcmp(argv[index],"--frame") == 0 && index + 1 < argc )
			config->frame = (uint32_t)strtoul(argv[++index],0,10);
		else if ( strcmp(argv[index],"--block") == 0 && index + 1 < argc )
			config->block = (uint32_t)strtoul(argv[++index],0,10);
		else if ( strcmp(argv[index],"--fixed-depth") == 0 )
			config->fixed_depth = 1u;
		else if ( strcmp(argv[index],"--seed") == 0 && index + 1 < argc )
			config->seed = strtoull(argv[++index],0,10);
		else if ( strcmp(argv[index],"--drafter") == 0 && index + 1 < argc )
		{
			const char *text = argv[++index];
			if ( strcmp(text,"lookup") == 0 )
				config->mode = REPLAY_MODE_LOOKUP;
			else if ( strncmp(text,"synthetic:",10) == 0 )
			{
				config->mode = REPLAY_MODE_SYNTHETIC;
				config->accept_milli = (uint32_t)strtoul(text + 10,0,10);
			}
			else if ( strncmp(text,"lookup+synthetic:",17) == 0 )
			{
				config->mode = REPLAY_MODE_MIX;
				config->accept_milli = (uint32_t)strtoul(text + 17,0,10);
			}
			else
				return(1);
		}
		else if ( strcmp(argv[index],"--") == 0 )
		{
			*first_file = index + 1;
			break;
		}
		else
		{
			*first_file = index;
			break;
		}
	}
	if ( config->rows < SPARK_GLM5_NEXT_VERIFY_ROWS_MIN || config->rows > SPARK_GLM5_NEXT_VERIFY_ROWS_MAX || config->frame == 0u || config->frame > REPLAY_FRAME_MAX || config->block == 0u || config->accept_milli > 1000u )
		return(1);
	return(*first_file > 0 && *first_file < argc ? 0 : 1);
}

int main(int argc,char **argv)
{
	ReplayConfig config;
	ReplayCounts counts;
	uint32_t *truth,prompt,length,row;
	int first_file = 0,index;
	if ( ReplayParse(argc,argv,&config,&first_file) != 0 )
	{
		fprintf(stderr,"usage: %s [--rows 2..8] [--frame 1..32] [--block N] [--seed N] [--fixed-depth] [--drafter lookup|synthetic:MILLI|lookup+synthetic:MILLI] STREAM.u32...\n",argv[0]);
		return(2);
	}
	for (index=first_file; index<argc; index++)
	{
		memset(&counts,0,sizeof(counts));
		truth = ReplayRead(argv[index],&prompt,&length);
		if ( truth == 0 )
		{
			fprintf(stderr,"%s: expected u32 prompt count, u32 total count, then the token ids\n",argv[index]);
			return(2);
		}
		if ( ReplayStream(&config,truth,prompt,length,&counts) != 0 )
		{
			fprintf(stderr,"%s: replay failed\n",argv[index]);
			free(truth);
			return(1);
		}
		printf("{\"file\": \"%s\", \"prompt\": %u, \"generated\": %u, \"frames\": %llu, \"verify_frames\": %llu, \"plain_frames\": %llu, \"plain_frame_tokens\": %llu, \"rounds\": %llu, \"proposed\": %llu, \"accepted\": %llu, \"plain_steps\": %llu, \"rows\": [",
			argv[index],prompt,length - prompt,(unsigned long long)counts.frames,(unsigned long long)counts.verify_frames,(unsigned long long)counts.plain_frames,
			(unsigned long long)counts.plain_frame_tokens,(unsigned long long)counts.rounds,(unsigned long long)counts.proposed,(unsigned long long)counts.accepted,(unsigned long long)counts.plain_steps);
		for (row=0u; row<=SPARK_GLM5_NEXT_VERIFY_ROWS_MAX; row++)
			printf("%s%llu",row == 0u ? "" : ", ",(unsigned long long)counts.rows_histogram[row]);
		printf("], \"lookup\": [%llu, %llu, %llu], \"synthetic\": [%llu, %llu, %llu], \"synthetic_calls\": %llu}\n",
			(unsigned long long)counts.source_rounds[0],(unsigned long long)counts.source_proposed[0],(unsigned long long)counts.source_accepted[0],
			(unsigned long long)counts.source_rounds[1],(unsigned long long)counts.source_proposed[1],(unsigned long long)counts.source_accepted[1],
			(unsigned long long)counts.synthetic_calls);
		free(truth);
	}
	return(0);
}
