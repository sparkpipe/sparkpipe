#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tools/token_prompts.h"

typedef struct CompareTotals
{
	double kl_sum;
	double kl_max;
	double target_delta_sum;
	double target_delta_max;
	double top1_delta_max;
	uint32_t top1_agree;
	uint32_t positions;
	uint32_t worst_position;
} CompareTotals;

static void LogSoftmax(const float *logits, uint32_t vocab, double *out, uint32_t *argmax)
{
	double best = -INFINITY, total = 0.0;
	uint32_t index;
	*argmax = 0u;
	for ( index = 0u; index < vocab; index++ )
		if ( logits[index] > best )
		{
			best = logits[index];
			*argmax = index;
		}
	for ( index = 0u; index < vocab; index++ )
		total += exp((double)logits[index] - best);
	for ( index = 0u; index < vocab; index++ )
		out[index] = (double)logits[index] - best - log(total);
}

static int ComparePosition(const float *reference, const float *test, uint32_t vocab, uint32_t target, double *ref_log, double *test_log,
	CompareTotals *totals, uint32_t position)
{
	uint32_t ref_top, test_top, index;
	double kl = 0.0, delta;
	for ( index = 0u; index < vocab; index++ )
		if ( !isfinite(reference[index]) || !isfinite(test[index]) )
			return(-1);
	LogSoftmax(reference,vocab,ref_log,&ref_top);
	LogSoftmax(test,vocab,test_log,&test_top);
	for ( index = 0u; index < vocab; index++ )
		kl += exp(ref_log[index]) * (ref_log[index] - test_log[index]);
	delta = fabs(ref_log[target] - test_log[target]);
	totals->kl_sum += kl;
	if ( kl > totals->kl_max )
	{
		totals->kl_max = kl;
		totals->worst_position = position;
	}
	totals->target_delta_sum += delta;
	if ( delta > totals->target_delta_max )
		totals->target_delta_max = delta;
	delta = fabs(ref_log[ref_top] - test_log[ref_top]);
	if ( delta > totals->top1_delta_max )
		totals->top1_delta_max = delta;
	totals->top1_agree += ref_top == test_top ? 1u : 0u;
	totals->positions++;
	return(0);
}

int main(int argc, char **argv)
{
	SparkTokenPrompts prompts;
	const SparkTokenPrompt *prompt = 0;
	CompareTotals totals;
	FILE *reference, *test;
	float *ref_row, *test_row;
	double *ref_log, *test_log;
	uint32_t vocab, index, positions;
	if ( argc != 7 )
	{
		fprintf(stderr,"usage: %s PROMPTS.json PROMPT_NAME VOCAB REFERENCE.logits TEST.logits LABEL\n",argv[0]);
		return(2);
	}
	if ( SparkTokenPromptsLoad(argv[1],&prompts) != SPARK_STATUS_OK )
		return(1);
	for ( index = 0u; index < prompts.count; index++ )
		if ( strcmp(prompts.prompts[index].name,argv[2]) == 0 )
			prompt = &prompts.prompts[index];
	vocab = (uint32_t)strtoul(argv[3],0,10);
	reference = fopen(argv[4],"rb");
	test = fopen(argv[5],"rb");
	ref_row = (float *)malloc((size_t)vocab * sizeof(float));
	test_row = (float *)malloc((size_t)vocab * sizeof(float));
	ref_log = (double *)malloc((size_t)vocab * sizeof(double));
	test_log = (double *)malloc((size_t)vocab * sizeof(double));
	if ( prompt == 0 || vocab == 0u || reference == 0 || test == 0 || ref_row == 0 || test_row == 0 || ref_log == 0 || test_log == 0 )
	{
		fprintf(stderr,"logit_compare: needs a known prompt, a vocabulary and two readable logit files\n");
		return(1);
	}
	memset(&totals,0,sizeof(totals));
	positions = prompt->count - 1u;
	for ( index = 0u; index < positions; index++ )
	{
		if ( fread(ref_row,sizeof(float),vocab,reference) != vocab || fread(test_row,sizeof(float),vocab,test) != vocab )
		{
			fprintf(stderr,"logit_compare: a logit file ends before position %u of %u\n",index,positions);
			return(1);
		}
		if ( ComparePosition(ref_row,test_row,vocab,prompt->tokens[index + 1u],ref_log,test_log,&totals,index) != 0 )
		{
			fprintf(stderr,"logit_compare: a non-finite logit at position %u\n",index);
			return(1);
		}
	}
	if ( fread(ref_row,1u,1u,reference) != 0u || fread(test_row,1u,1u,test) != 0u )
	{
		fprintf(stderr,"logit_compare: a logit file holds more than %u positions\n",positions);
		return(1);
	}
	printf("{\"label\":\"%s\",\"prompt\":\"%s\",\"positions\":%u,\"kl_mean\":%.6e,\"kl_max\":%.6e,\"kl_max_position\":%u,"
		"\"top1_agreement\":%.6f,\"top1_agree\":%u,\"target_logprob_delta_mean\":%.6e,\"target_logprob_delta_max\":%.6e,"
		"\"reference_top1_logprob_delta_max\":%.6e}\n",argv[6],prompt->name,totals.positions,totals.kl_sum / totals.positions,totals.kl_max,
		totals.worst_position,(double)totals.top1_agree / totals.positions,totals.top1_agree,totals.target_delta_sum / totals.positions,
		totals.target_delta_max,totals.top1_delta_max);
	fclose(reference);
	fclose(test);
	free(ref_row);
	free(test_row);
	free(ref_log);
	free(test_log);
	SparkTokenPromptsFree(&prompts);
	return(0);
}
