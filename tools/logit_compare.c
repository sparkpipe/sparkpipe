#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tools/token_prompts.h"

#define COMPARE_BLOCK_POSITIONS 128u

typedef struct CompareTotals
{
	double kl_prefill;
	double kl_decode;
	uint32_t prefill_positions;
	uint32_t stable_positions;
	uint32_t stable_agree;
	double kl_sum;
	double kl_max;
	double target_delta_sum;
	double target_delta_max;
	double top1_delta_max;
	uint32_t top1_agree;
	uint32_t positions;
	uint32_t worst_position;
	double *kl;
	double *floor_kl;
	uint32_t above_floor;
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

static double Kl(const double *ref_log, const double *test_log, uint32_t vocab)
{
	double kl = 0.0;
	uint32_t index;
	for ( index = 0u; index < vocab; index++ )
		kl += exp(ref_log[index]) * (ref_log[index] - test_log[index]);
	return(kl);
}

static int CompareDouble(const void *a, const void *b)
{
	const double x = *(const double *)a, y = *(const double *)b;
	return(x < y ? -1 : x > y ? 1 : 0);
}

static double Quantile(double *values, uint32_t count, double q)
{
	qsort(values,count,sizeof(double),CompareDouble);
	return(count != 0u ? values[(uint32_t)(q * (count - 1u) + 0.5)] : 0.0);
}

static int ComparePosition(const float *reference, const float *test, uint32_t vocab, uint32_t target, double *ref_log, double *test_log,
	CompareTotals *totals, uint32_t position, uint32_t split, const float *second)
{
	uint32_t ref_top, test_top, second_top, index;
	double kl, delta;
	for ( index = 0u; index < vocab; index++ )
		if ( !isfinite(reference[index]) || !isfinite(test[index]) )
			return(-1);
	LogSoftmax(reference,vocab,ref_log,&ref_top);
	LogSoftmax(test,vocab,test_log,&test_top);
	kl = Kl(ref_log,test_log,vocab);
	totals->kl[position] = kl;
	delta = fabs(ref_log[target] - test_log[target]);
	totals->target_delta_sum += delta;
	if ( delta > totals->target_delta_max )
		totals->target_delta_max = delta;
	delta = fabs(ref_log[ref_top] - test_log[ref_top]);
	if ( delta > totals->top1_delta_max )
		totals->top1_delta_max = delta;
	if ( second != 0 )
	{
		for ( index = 0u; index < vocab; index++ )
			if ( !isfinite(second[index]) )
				return(-1);
		LogSoftmax(second,vocab,test_log,&second_top);
		totals->floor_kl[position] = Kl(ref_log,test_log,vocab);
		totals->above_floor += kl > totals->floor_kl[position] ? 1u : 0u;
	}
	totals->kl_sum += kl;
	if ( position < split )
	{
		totals->kl_prefill += kl;
		totals->prefill_positions++;
	}
	else
		totals->kl_decode += kl;
	if ( second != 0 && second_top == ref_top )
	{
		totals->stable_positions++;
		totals->stable_agree += ref_top == test_top ? 1u : 0u;
	}
	if ( kl > totals->kl_max )
	{
		totals->kl_max = kl;
		totals->worst_position = position;
	}
	totals->top1_agree += ref_top == test_top ? 1u : 0u;
	totals->positions++;
	return(0);
}

int main(int argc, char **argv)
{
	SparkTokenPrompts prompts;
	const SparkTokenPrompt *prompt = 0;
	CompareTotals totals;
	FILE *reference, *test, *second = 0;
	float *ref_row, *test_row, *second_row = 0;
	uint32_t split;
	double *ref_log, *test_log;
	uint32_t vocab, index, positions;
	if ( argc < 7 || argc > 9 )
	{
		fprintf(stderr,"usage: %s PROMPTS.json PROMPT_NAME VOCAB REFERENCE.logits TEST.logits LABEL [PREFILL_POSITIONS [SECOND_REFERENCE.logits]]\n",argv[0]);
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
	split = argc >= 8 ? (uint32_t)strtoul(argv[7],0,10) : 0u;
	if ( argc == 9 )
	{
		second = fopen(argv[8],"rb");
		second_row = (float *)malloc((size_t)vocab * sizeof(float));
		if ( second == 0 || second_row == 0 )
		{
			fprintf(stderr,"logit_compare: cannot read the second reference\n");
			return(1);
		}
	}
	if ( prompt == 0 || vocab == 0u || reference == 0 || test == 0 || ref_row == 0 || test_row == 0 || ref_log == 0 || test_log == 0 )
	{
		fprintf(stderr,"logit_compare: needs a known prompt, a vocabulary and two readable logit files\n");
		return(1);
	}
	memset(&totals,0,sizeof(totals));
	positions = prompt->count - 1u;
	totals.kl = (double *)calloc(positions,sizeof(double));
	totals.floor_kl = (double *)calloc(positions,sizeof(double));
	if ( totals.kl == 0 || totals.floor_kl == 0 )
		return(1);
	for ( index = 0u; index < positions; index++ )
	{
		if ( fread(ref_row,sizeof(float),vocab,reference) != vocab || fread(test_row,sizeof(float),vocab,test) != vocab ||
			(second != 0 && fread(second_row,sizeof(float),vocab,second) != vocab) )
		{
			fprintf(stderr,"logit_compare: a logit file ends before position %u of %u\n",index,positions);
			return(1);
		}
		if ( ComparePosition(ref_row,test_row,vocab,prompt->tokens[index + 1u],ref_log,test_log,&totals,index,split,second_row) != 0 )
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
		"\"reference_top1_logprob_delta_max\":%.6e,\"prefill_positions\":%u,\"kl_mean_prefill\":%.6e,\"kl_mean_decode\":%.6e,"
		"\"stable_positions\":%u,\"stable_top1_agree\":%u,",argv[6],prompt->name,totals.positions,totals.kl_sum / totals.positions,totals.kl_max,
		totals.worst_position,(double)totals.top1_agree / totals.positions,totals.top1_agree,totals.target_delta_sum / totals.positions,
		totals.target_delta_max,totals.top1_delta_max,totals.prefill_positions,
		totals.prefill_positions != 0u ? totals.kl_prefill / totals.prefill_positions : 0.0,
		totals.positions > totals.prefill_positions ? totals.kl_decode / (totals.positions - totals.prefill_positions) : 0.0,
		totals.stable_positions,totals.stable_agree);
	if ( second != 0 )
	{
		double floor_sum = 0.0;
		for ( index = 0u; index < positions; index++ )
			floor_sum += totals.floor_kl[index];
		printf("\"floor_kl_mean\":%.6e,\"above_floor_positions\":%u,\"block_positions\":%u,\"above_floor_blocks\":[",floor_sum / positions,
			totals.above_floor,COMPARE_BLOCK_POSITIONS);
		for ( index = 0u; index < positions; index += COMPARE_BLOCK_POSITIONS )
		{
			uint32_t at, above = 0u;
			for ( at = index; at < positions && at < index + COMPARE_BLOCK_POSITIONS; at++ )
				above += totals.kl[at] > totals.floor_kl[at] ? 1u : 0u;
			printf("%s[%u,%u]",index != 0u ? "," : "",above,at - index);
		}
		printf("],\"floor_kl_median\":%.6e,",Quantile(totals.floor_kl,positions,0.5));
	}
	printf("\"kl_median\":%.6e,\"kl_p90\":%.6e}\n",Quantile(totals.kl,positions,0.5),Quantile(totals.kl,positions,0.9));
	free(totals.kl);
	free(totals.floor_kl);
	fclose(reference);
	fclose(test);
	if ( second != 0 )
		fclose(second);
	free(second_row);
	free(ref_row);
	free(test_row);
	free(ref_log);
	free(test_log);
	SparkTokenPromptsFree(&prompts);
	return(0);
}
