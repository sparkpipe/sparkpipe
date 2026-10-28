#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "inference/runner/device_stage_model.h"
#include "inference/runner/device_stage_runner.h"
#include "sparkpipe/spark_gemma4_device_stage.h"
#include "sparkpipe/spark_qwen38_device_stage.h"
#include "sparkpipe/spark_speculation_frame.h"
#include "sparkpipe/spark_speculation_lookup_draft.h"
#include "sparkpipe/spark_weight_codec.h"
#include "tools/token_prompts.h"

#define RUN_MAX_CONTEXT 9216u
#define RUN_STATE_BUDGET (16ull << 30)
#define RUN_LADDER_STATE_BUDGET (96ull << 30)
#define RUN_LADDER_MAX 16u
#define RUN_PROMPT_STRIDE 331u
#define RUN_LOOKUP_GRAM 3u
#define RUN_LOOKUP_MAX_MATCH 64u
#define RUN_LOOKUP_BUCKETS 4096u
#define RUN_LOOKUP_CANDIDATES 64u
#define RUN_FRAME_SIZES 7u

typedef struct RunOptions
{
	const char *pack;
	const char *prompts;
	const char *mode;
	const char *out;
	const char *bench_prompt;
	uint32_t prefill_rows;
	uint32_t new_tokens;
	uint32_t repeats;
	uint32_t verify;
	uint32_t prompt_tokens;
	uint32_t batches[RUN_LADDER_MAX];
	uint32_t batch_count;
	uint32_t max_batch;
	const char *spec_prompts;
	const char *model;
	uint32_t draft_max;
	const char *pin_routes;
	uint32_t exact_rows;
} RunOptions;

static const uint32_t run_frame_sizes[RUN_FRAME_SIZES] = { 1u, 2u, 4u, 8u, 16u, 32u, 64u };

static uint64_t NowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

typedef const SparkStageRunnerModelInterface *(*RunModelGetter)(void);

static const RunModelGetter run_models[] = { SparkQwen38_27bDeviceStageModel, SparkGemma4_26bDeviceStageModel, SparkGemma4_31bDeviceStageModel };

static const SparkStageRunnerModelInterface *RunModel(const char *name)
{
	uint32_t index;
	if ( name == 0 )
		return(run_models[0]());
	for ( index = 0u; index < sizeof(run_models) / sizeof(run_models[0]); index++ )
		if ( strcmp(name,run_models[index]()->weightd_model) == 0 )
			return(run_models[index]());
	return(0);
}

static int Usage(const char *program)
{
	fprintf(stderr,"usage: %s --pack PACK --prompts PROMPTS.json --mode teacher|greedy|pairs|bench|ladder|spec --out DIR [--prefill-rows N] [--new-tokens N] "
		"[--repeats N] [--bench-prompt NAME] [--verify-pack] [--batches 1,2,4,...] [--prompt-tokens N] [--spec-prompts PROMPTS.json] [--draft-max N] [--model qwen3.8-27b|gemma4-26b-a4b|gemma4-31b] [--pin-routes DIR] [--exact-rows]\n",program);
	return(2);
}

static int ParseOptions(int argc, char **argv, RunOptions *options)
{
	int index;
	memset(options,0,sizeof(*options));
	options->prefill_rows = 1024u;
	options->new_tokens = 128u;
	options->repeats = 3u;
	options->prompt_tokens = 1024u;
	options->draft_max = 15u;
	for ( index = 1; index < argc; index++ )
	{
		const char *flag = argv[index], *value = index + 1 < argc ? argv[index + 1] : 0;
		if ( strcmp(flag,"--verify-pack") == 0 )
		{
			options->verify = 1u;
			continue;
		}
		if ( strcmp(flag,"--exact-rows") == 0 )
		{
			options->exact_rows = 1u;
			continue;
		}
		if ( value == 0 )
			return(-1);
		index++;
		if ( strcmp(flag,"--pack") == 0 )
			options->pack = value;
		else if ( strcmp(flag,"--prompts") == 0 )
			options->prompts = value;
		else if ( strcmp(flag,"--mode") == 0 )
			options->mode = value;
		else if ( strcmp(flag,"--out") == 0 )
			options->out = value;
		else if ( strcmp(flag,"--bench-prompt") == 0 )
			options->bench_prompt = value;
		else if ( strcmp(flag,"--prefill-rows") == 0 )
			options->prefill_rows = (uint32_t)strtoul(value,0,10);
		else if ( strcmp(flag,"--new-tokens") == 0 )
			options->new_tokens = (uint32_t)strtoul(value,0,10);
		else if ( strcmp(flag,"--repeats") == 0 )
			options->repeats = (uint32_t)strtoul(value,0,10);
		else if ( strcmp(flag,"--model") == 0 )
			options->model = value;
		else if ( strcmp(flag,"--spec-prompts") == 0 )
			options->spec_prompts = value;
		else if ( strcmp(flag,"--draft-max") == 0 )
			options->draft_max = (uint32_t)strtoul(value,0,10);
		else if ( strcmp(flag,"--pin-routes") == 0 )
			options->pin_routes = value;
		else if ( strcmp(flag,"--prompt-tokens") == 0 )
			options->prompt_tokens = (uint32_t)strtoul(value,0,10);
		else if ( strcmp(flag,"--batches") == 0 )
		{
			const char *cursor = value;
			char *end;
			while ( *cursor != 0 && options->batch_count < RUN_LADDER_MAX )
			{
				const uint32_t batch = (uint32_t)strtoul(cursor,&end,10);
				if ( end == cursor || batch == 0u )
					return(-1);
				options->batches[options->batch_count++] = batch;
				if ( batch > options->max_batch )
					options->max_batch = batch;
				cursor = *end == ',' ? end + 1 : end;
			}
			if ( *cursor != 0 )
				return(-1);
		}
		else
			return(-1);
	}
	if ( options->pack == 0 || options->prompts == 0 || options->mode == 0 || options->out == 0 || options->prefill_rows == 0u ||
		options->prefill_rows > 2048u || options->repeats == 0u || options->prompt_tokens == 0u )
		return(-1);
	if ( options->batch_count == 0u )
	{
		options->batches[0] = options->max_batch = 1u;
		options->batch_count = 1u;
	}
	if ( options->max_batch > options->prefill_rows || options->draft_max == 0u || options->draft_max >= SPARK_DEVICE_STAGE_VERIFY_ROWS )
		return(-1);
	if ( strcmp(options->mode,"spec") == 0 && options->prefill_rows < run_frame_sizes[RUN_FRAME_SIZES - 1u] )
	{
		fprintf(stderr,"metal_stage_run: spec mode times verify frames of up to %u rows, so --prefill-rows must be at least %u (it is %u)\n",
			run_frame_sizes[RUN_FRAME_SIZES - 1u],run_frame_sizes[RUN_FRAME_SIZES - 1u],options->prefill_rows);
		return(-1);
	}
	return(0);
}

typedef struct Session
{
	SparkDeviceStageRunner *runner;
	uint32_t *rows;
	uint32_t *positions;
	uint32_t *sequence;
	uint32_t *targets;
	uint32_t *kept;
	uint32_t capacity;
	uint32_t layers;
	SparkDeviceBuffer routes;
	uint16_t *pinned;
	uint32_t pinned_positions;
	uint32_t pinned_width;
} Session;

static SparkStatus SessionStepSequence(Session *session, uint32_t sequence, const uint32_t *tokens, uint32_t first, uint32_t count,
	uint32_t head_all, const uint32_t *targets, SparkDeviceStageRunnerResult *result)
{
	SparkDeviceStageRunnerBatch batch;
	uint32_t row;
	for ( row = 0u; row < count; row++ )
	{
		session->positions[row] = first + row;
		session->sequence[row] = sequence;
		session->rows[row] = row;
		session->targets[row] = targets != 0 ? targets[row] : 0u;
	}
	memset(&batch,0,sizeof(batch));
	batch.rows = count;
	batch.token_ids = tokens;
	batch.positions = session->positions;
	batch.sequence_of_row = session->sequence;
	batch.head_rows = head_all != 0u ? count : 1u;
	batch.head_row_index = head_all != 0u ? session->rows : session->rows + (count - 1u);
	batch.targets = targets != 0 ? session->targets : 0;
	if ( session->pinned != 0 )
	{
		uint32_t *routes = (uint32_t *)session->routes.host, layer, k;
		if ( first + count > session->pinned_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		for ( layer = 0u; layer < session->layers; layer++ )
			for ( row = 0u; row < count; row++ )
				for ( k = 0u; k < session->pinned_width; k++ )
					routes[((uint64_t)layer * count + row) * session->pinned_width + k] =
						session->pinned[((uint64_t)layer * session->pinned_positions + first + row) * session->pinned_width + k];
		batch.route_override = (const uint32_t *)(uintptr_t)session->routes.device;
		batch.route_override_width = session->pinned_width;
	}
	return(SparkDeviceStageRunnerStep(session->runner,&batch,result));
}

static SparkStatus SessionStep(Session *session, const uint32_t *tokens, uint32_t first, uint32_t count, uint32_t head_all, const uint32_t *targets,
	SparkDeviceStageRunnerResult *result)
{
	return(SessionStepSequence(session,0u,tokens,first,count,head_all,targets,result));
}

static SparkStatus SessionPrefillSequence(Session *session, uint32_t sequence, const uint32_t *tokens, uint32_t count, uint32_t chunk,
	uint32_t *next_token)
{
	SparkDeviceStageRunnerResult result;
	uint32_t first;
	SparkStatus status = SPARK_STATUS_OK;
	for ( first = 0u; status == SPARK_STATUS_OK && first < count; first += chunk )
	{
		const uint32_t rows = count - first < chunk ? count - first : chunk;
		status = SessionStepSequence(session,sequence,tokens + first,first,rows,0u,0,&result);
		if ( status == SPARK_STATUS_OK )
			*next_token = result.argmax[0];
	}
	return(status);
}

static SparkStatus SessionPrefill(Session *session, const uint32_t *tokens, uint32_t count, uint32_t chunk, uint32_t *next_token, uint64_t *ns)
{
	SparkDeviceStageRunnerResult result;
	uint32_t first;
	SparkStatus status = SPARK_STATUS_OK;
	const uint64_t start = NowNs();
	for ( first = 0u; status == SPARK_STATUS_OK && first < count; first += chunk )
	{
		const uint32_t rows = count - first < chunk ? count - first : chunk;
		status = SessionStep(session,tokens + first,first,rows,0u,0,&result);
		if ( status == SPARK_STATUS_OK && next_token != 0 )
			*next_token = result.argmax[0];
	}
	if ( ns != 0 )
		*ns = NowNs() - start;
	return(status);
}

static FILE *OpenOut(const RunOptions *options, const char *name, const char *suffix, const char *mode)
{
	char path[4096];
	FILE *file;
	snprintf(path,sizeof(path),"%s/%s%s",options->out,name,suffix);
	file = fopen(path,mode);
	if ( file == 0 )
		fprintf(stderr,"cannot write %s: %s\n",path,strerror(errno));
	return(file);
}

static int LoadPinnedRoutes(Session *session, const RunOptions *options, const SparkTokenPrompt *prompt)
{
	char path[4096];
	FILE *file;
	long bytes;
	uint64_t per_width;
	free(session->pinned);
	session->pinned = 0;
	if ( options->pin_routes == 0 )
		return(0);
	snprintf(path,sizeof(path),"%s/%s.ref_a.routes",options->pin_routes,prompt->name);
	file = fopen(path,"rb");
	if ( file == 0 || fseek(file,0,SEEK_END) != 0 || (bytes = ftell(file)) <= 0 || fseek(file,0,SEEK_SET) != 0 )
	{
		fprintf(stderr,"cannot read routes %s\n",path);
		if ( file != 0 )
			fclose(file);
		return(1);
	}
	per_width = (uint64_t)session->layers * prompt->count * sizeof(uint16_t);
	session->pinned_width = (uint32_t)((uint64_t)bytes / per_width);
	session->pinned_positions = prompt->count;
	session->pinned = (uint16_t *)malloc((size_t)bytes);
	if ( (uint64_t)bytes % per_width != 0u || session->pinned_width == 0u || session->pinned_width > SPARK_ROUTE_MAX_TOP_K || session->pinned == 0 ||
		fread(session->pinned,1u,(size_t)bytes,file) != (size_t)bytes )
	{
		fprintf(stderr,"routes %s: %ld bytes is not %u layers x %u positions x top-k uint16\n",path,bytes,session->layers,prompt->count);
		fclose(file);
		free(session->pinned);
		session->pinned = 0;
		return(1);
	}
	fclose(file);
	if ( session->routes.host == 0 && SparkDeviceAllocate(&session->routes,SPARK_DEVICE_SPACE_SHARED_COHERENT,
		(uint64_t)session->layers * session->capacity * SPARK_ROUTE_MAX_TOP_K * sizeof(uint32_t),"pinned routes") != SPARK_STATUS_OK )
		return(1);
	printf("teacher %s: experts pinned from %s (top %u)\n",prompt->name,path,session->pinned_width);
	return(0);
}

static int RunTeacher(Session *session, const RunOptions *options, const SparkTokenPrompt *prompt, uint32_t vocab)
{
	const uint32_t positions = prompt->count - 1u, prefill = (positions + 1u) / 2u;
	SparkDeviceStageRunnerResult result;
	FILE *logits = OpenOut(options,prompt->name,".metal.logits","wb"), *json = OpenOut(options,prompt->name,".metal.json","w");
	float *logprob = (float *)calloc(positions,sizeof(float));
	uint32_t *argmax = (uint32_t *)calloc(positions,sizeof(uint32_t)), first, row;
	SparkStatus status = logits != 0 && json != 0 && logprob != 0 && argmax != 0 && LoadPinnedRoutes(session,options,prompt) == 0 ? SPARK_STATUS_OK :
		SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageRunnerResetSequence(session->runner,0u);
	for ( first = 0u; status == SPARK_STATUS_OK && first < positions; )
	{
		const uint32_t rows = first < prefill ? (prefill - first < options->prefill_rows ? prefill - first : options->prefill_rows) : 1u;
		status = SessionStep(session,prompt->tokens + first,first,rows,1u,prompt->tokens + first + 1u,&result);
		for ( row = 0u; status == SPARK_STATUS_OK && row < rows; row++ )
		{
			logprob[first + row] = result.logprob[row * 4u + 2u];
			argmax[first + row] = result.argmax[row];
		}
		if ( status == SPARK_STATUS_OK && fwrite(result.logits,sizeof(float),(size_t)rows * vocab,logits) != (size_t)rows * vocab )
			status = SPARK_STATUS_IO_ERROR;
		first += rows;
	}
	if ( status == SPARK_STATUS_OK )
	{
		fprintf(json,"{\"name\":\"%s\",\"positions\":%u,\"prefill_positions\":%u,\"vocab\":%u,\"target_logprob\":[",prompt->name,positions,prefill,vocab);
		for ( row = 0u; row < positions; row++ )
			fprintf(json,"%s%.9g",row != 0u ? "," : "",logprob[row]);
		fprintf(json,"],\"argmax\":[");
		for ( row = 0u; row < positions; row++ )
			fprintf(json,"%s%u",row != 0u ? "," : "",argmax[row]);
		fprintf(json,"]}\n");
		printf("teacher %s: %u positions (%u by prefill, %u by decode)\n",prompt->name,positions,prefill,positions - prefill);
	}
	if ( logits != 0 )
		fclose(logits);
	if ( json != 0 )
		fclose(json);
	free(logprob);
	free(argmax);
	return(status == SPARK_STATUS_OK ? 0 : 1);
}

static int RunGreedy(Session *session, const RunOptions *options, const SparkTokenPrompt *prompt, FILE *json, uint32_t first_prompt)
{
	SparkDeviceStageRunnerResult result;
	uint32_t *generated = (uint32_t *)calloc(options->new_tokens,sizeof(uint32_t)), token = 0u, index;
	float *scores = (float *)calloc(options->new_tokens,sizeof(float));
	SparkStatus status = generated != 0 && scores != 0 ? SparkDeviceStageRunnerResetSequence(session->runner,0u) : SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SessionPrefill(session,prompt->tokens,prompt->count,options->prefill_rows,&token,0);
	for ( index = 0u; status == SPARK_STATUS_OK && index < options->new_tokens; index++ )
	{
		generated[index] = token;
		status = SessionStep(session,&generated[index],prompt->count + index,1u,0u,0,&result);
		if ( status == SPARK_STATUS_OK )
		{
			scores[index] = result.logprob[0];
			token = result.argmax[0];
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		fprintf(json,"%s{\"name\":\"%s\",\"prompt_tokens\":%u,\"generated\":[",first_prompt != 0u ? "" : ",\n",prompt->name,prompt->count);
		for ( index = 0u; index < options->new_tokens; index++ )
			fprintf(json,"%s%u",index != 0u ? "," : "",generated[index]);
		fprintf(json,"],\"top1_logprob_after\":[");
		for ( index = 0u; index < options->new_tokens; index++ )
			fprintf(json,"%s%.9g",index != 0u ? "," : "",scores[index]);
		fprintf(json,"]}");
		printf("greedy %s: %u tokens after a %u-token prompt\n",prompt->name,options->new_tokens,prompt->count);
	}
	free(generated);
	free(scores);
	return(status == SPARK_STATUS_OK ? 0 : 1);
}

static int RunPair(Session *session, const RunOptions *options, const SparkTokenPrompt *first, const SparkTokenPrompt *second, FILE *json,
	uint32_t first_pair)
{
	const SparkTokenPrompt *pair[2] = { first, second };
	uint32_t *generated[2], token[2] = { 0u, 0u }, index, lane;
	uint64_t total = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	generated[0] = (uint32_t *)calloc(options->new_tokens,sizeof(uint32_t));
	generated[1] = (uint32_t *)calloc(options->new_tokens,sizeof(uint32_t));
	for ( lane = 0u; status == SPARK_STATUS_OK && lane < 2u; lane++ )
	{
		status = generated[lane] != 0 ? SparkDeviceStageRunnerResetSequence(session->runner,lane) : SPARK_STATUS_CAPACITY_EXCEEDED;
		if ( status == SPARK_STATUS_OK )
			status = SessionPrefillSequence(session,lane,pair[lane]->tokens,pair[lane]->count,options->prefill_rows,&token[lane]);
	}
	for ( index = 0u; status == SPARK_STATUS_OK && index < options->new_tokens; index++ )
	{
		SparkDeviceStageRunnerBatch batch;
		SparkDeviceStageRunnerResult result;
		const uint64_t before = NowNs();
		for ( lane = 0u; lane < 2u; lane++ )
		{
			generated[lane][index] = token[lane];
			session->positions[lane] = pair[lane]->count + index;
			session->sequence[lane] = lane;
			session->rows[lane] = lane;
		}
		memset(&batch,0,sizeof(batch));
		batch.rows = 2u;
		batch.token_ids = token;
		batch.positions = session->positions;
		batch.sequence_of_row = session->sequence;
		batch.head_rows = 2u;
		batch.head_row_index = session->rows;
		status = SparkDeviceStageRunnerStep(session->runner,&batch,&result);
		total += NowNs() - before;
		if ( status == SPARK_STATUS_OK )
		{
			token[0] = result.argmax[0];
			token[1] = result.argmax[1];
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		for ( lane = 0u; lane < 2u; lane++ )
		{
			fprintf(json,"%s{\"name\":\"%s\",\"batched_with\":\"%s\",\"generated\":[",first_pair != 0u && lane == 0u ? "" : ",\n",pair[lane]->name,
				pair[1u - lane]->name);
			for ( index = 0u; index < options->new_tokens; index++ )
				fprintf(json,"%s%u",index != 0u ? "," : "",generated[lane][index]);
			fprintf(json,"],\"decode_steps\":%u,\"decode_seconds\":%.6f,\"rows_per_second\":%.4f}",options->new_tokens,total / 1e9,
				2.0 * options->new_tokens / (total / 1e9));
		}
		printf("pair %s + %s: %u two-row decode steps, %.3f rows/s\n",first->name,second->name,options->new_tokens,
			2.0 * options->new_tokens / (total / 1e9));
	}
	free(generated[0]);
	free(generated[1]);
	return(status == SPARK_STATUS_OK ? 0 : 1);
}

static int CompareU64(const void *a, const void *b)
{
	const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
	return(x < y ? -1 : x > y ? 1 : 0);
}

static int BenchDecode(Session *session, const RunOptions *options, const uint32_t *tokens, uint32_t prompt_tokens, FILE *json)
{
	const uint32_t steps = options->new_tokens;
	uint64_t *totals = (uint64_t *)calloc(options->repeats + 1u,sizeof(uint64_t)), *step_ns = (uint64_t *)calloc(steps,sizeof(uint64_t)), median;
	uint32_t repeat, index, token = 0u;
	SparkDeviceStageRunnerResult result;
	SparkStatus status = totals != 0 && step_ns != 0 ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED;
	fprintf(json,"\"decode\":{\"prompt_tokens\":%u,\"output_tokens\":%u,\"runs\":[",prompt_tokens,steps);
	for ( repeat = 0u; status == SPARK_STATUS_OK && repeat <= options->repeats; repeat++ )
	{
		uint64_t start;
		status = SparkDeviceStageRunnerResetSequence(session->runner,0u);
		if ( status == SPARK_STATUS_OK )
			status = SessionPrefill(session,tokens,prompt_tokens,options->prefill_rows,&token,0);
		start = NowNs();
		for ( index = 0u; status == SPARK_STATUS_OK && index < steps; index++ )
		{
			const uint64_t before = NowNs();
			status = SessionStep(session,&token,prompt_tokens + index,1u,0u,0,&result);
			step_ns[index] = NowNs() - before;
			if ( status == SPARK_STATUS_OK )
				token = result.argmax[0];
		}
		totals[repeat] = NowNs() - start;
		if ( status == SPARK_STATUS_OK )
		{
			qsort(step_ns,steps,sizeof(uint64_t),CompareU64);
			fprintf(json,"%s{\"warmup\":%s,\"seconds\":%.6f,\"tokens_per_second\":%.4f,\"step_ms_median\":%.4f,\"step_ms_p90\":%.4f}",
				repeat != 0u ? "," : "",repeat == 0u ? "true" : "false",totals[repeat] / 1e9,steps / (totals[repeat] / 1e9),
				step_ns[steps / 2u] / 1e6,step_ns[(steps * 9u) / 10u] / 1e6);
			printf("decode run %u%s: %.3f tok/s (median step %.3f ms)\n",repeat,repeat == 0u ? " (warm-up)" : "",steps / (totals[repeat] / 1e9),
				step_ns[steps / 2u] / 1e6);
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		double bytes = 0.0;
		qsort(totals + 1,options->repeats,sizeof(uint64_t),CompareU64);
		median = totals[1u + options->repeats / 2u];
		for ( index = 0u; index < steps; index++ )
			bytes += (double)SparkDeviceStageRunnerDecodeBytes(session->runner,prompt_tokens + index + 1u) / steps;
		fprintf(json,"],\"median_tokens_per_second\":%.4f,\"bytes_per_token_mean\":%.0f}",steps / (median / 1e9),bytes);
		printf("decode median of %u: %.3f tok/s\n",options->repeats,steps / (median / 1e9));
	}
	free(totals);
	free(step_ns);
	return(status == SPARK_STATUS_OK ? 0 : 1);
}

static int BenchPrefill(Session *session, const RunOptions *options, const uint32_t *tokens, uint32_t count, FILE *json)
{
	uint64_t *ns = (uint64_t *)calloc(options->repeats + 1u,sizeof(uint64_t)), median;
	uint32_t repeat;
	SparkStatus status = ns != 0 ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED;
	fprintf(json,",\"prefill_%u\":{\"chunk_rows\":%u,\"runs_seconds\":[",count,options->prefill_rows);
	for ( repeat = 0u; status == SPARK_STATUS_OK && repeat <= options->repeats; repeat++ )
	{
		status = SparkDeviceStageRunnerResetSequence(session->runner,0u);
		if ( status == SPARK_STATUS_OK )
			status = SessionPrefill(session,tokens,count,options->prefill_rows,0,&ns[repeat]);
		if ( status == SPARK_STATUS_OK )
		{
			fprintf(json,"%s%.6f",repeat != 0u ? "," : "",ns[repeat] / 1e9);
			printf("prefill %u run %u%s: %.2f tok/s\n",count,repeat,repeat == 0u ? " (warm-up)" : "",count / (ns[repeat] / 1e9));
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		qsort(ns + 1,options->repeats,sizeof(uint64_t),CompareU64);
		median = ns[1u + options->repeats / 2u];
		fprintf(json,"],\"warmup_discarded\":true,\"median_tokens_per_second\":%.4f}",count / (median / 1e9));
	}
	free(ns);
	return(status == SPARK_STATUS_OK ? 0 : 1);
}

static uint32_t *Synthesize(const SparkTokenPrompts *prompts, uint32_t count)
{
	uint32_t *tokens = (uint32_t *)malloc((size_t)count * sizeof(uint32_t)), filled = 0u, index = 0u;
	while ( tokens != 0 && filled < count )
	{
		const SparkTokenPrompt *prompt = &prompts->prompts[index++ % prompts->count];
		const uint32_t take = prompt->count < count - filled ? prompt->count : count - filled;
		memcpy(tokens + filled,prompt->tokens,(size_t)take * sizeof(uint32_t));
		filled += take;
	}
	return(tokens);
}

static int RunBench(Session *session, const RunOptions *options, const SparkTokenPrompts *prompts, const SparkDeviceStageRunnerInfo *info)
{
	FILE *json = OpenOut(options,"bench",".metal.json","w");
	uint32_t *decode_prompt = Synthesize(prompts,1024u), *short_prompt = Synthesize(prompts,1024u), *long_prompt = Synthesize(prompts,8192u);
	int failed = json == 0 || decode_prompt == 0 || short_prompt == 0 || long_prompt == 0;
	if ( !failed )
	{
		fprintf(json,"{\"device\":\"%s\",\"target\":\"%s\",\"pack_bytes\":%llu,\"streamed_bytes_per_token\":%llu,\"streamed_bytes_per_chunk\":%llu,",
			info->device,info->target,(unsigned long long)info->pack_bytes,(unsigned long long)info->streamed_bytes_per_token,
			(unsigned long long)info->streamed_bytes_per_chunk);
		failed = BenchDecode(session,options,decode_prompt,1024u,json) != 0;
	}
	if ( !failed )
		failed = BenchPrefill(session,options,short_prompt,1024u,json) != 0;
	if ( !failed )
		failed = BenchPrefill(session,options,long_prompt,8192u,json) != 0;
	if ( json != 0 )
	{
		fprintf(json,"}\n");
		fclose(json);
	}
	free(decode_prompt);
	free(short_prompt);
	free(long_prompt);
	return(failed);
}

static uint32_t *SynthesizeAt(const SparkTokenPrompts *prompts, uint32_t count, uint64_t offset)
{
	uint64_t total = 0u;
	uint32_t index, filled = 0u, *tokens = (uint32_t *)malloc((size_t)count * sizeof(uint32_t));
	for ( index = 0u; index < prompts->count; index++ )
		total += prompts->prompts[index].count;
	offset = total != 0u ? offset % total : 0u;
	for ( index = 0u; tokens != 0 && total != 0u && filled < count; index = (index + 1u) % prompts->count )
	{
		const SparkTokenPrompt *prompt = &prompts->prompts[index];
		uint32_t skip = 0u, take;
		if ( offset >= prompt->count )
		{
			offset -= prompt->count;
			continue;
		}
		skip = (uint32_t)offset;
		offset = 0u;
		take = prompt->count - skip < count - filled ? prompt->count - skip : count - filled;
		memcpy(tokens + filled,prompt->tokens + skip,(size_t)take * sizeof(uint32_t));
		filled += take;
	}
	return(tokens);
}

static void Top2(const float *logits, uint32_t vocab, uint32_t *first, uint32_t *second, float *gap)
{
	uint32_t index, a = 0u, b = 1u;
	if ( logits[1] > logits[0] )
	{
		a = 1u;
		b = 0u;
	}
	for ( index = 2u; index < vocab; index++ )
		if ( logits[index] > logits[a] )
		{
			b = a;
			a = index;
		}
		else if ( logits[index] > logits[b] )
			b = index;
	*first = a;
	*second = b;
	*gap = logits[a] - logits[b];
}

typedef struct LadderTrace
{
	uint32_t *token;
	uint32_t *second;
	float *gap;
} LadderTrace;

static SparkStatus LadderDecode(Session *session, uint32_t batch, uint32_t prompt_tokens, uint32_t steps, uint32_t vocab, uint32_t exact, uint32_t *token,
	LadderTrace *trace, uint64_t *ns)
{
	SparkDeviceStageRunnerBatch step;
	SparkDeviceStageRunnerResult result;
	uint32_t index, lane, first;
	SparkStatus status = SPARK_STATUS_OK;
	const uint64_t start = NowNs();
	for ( index = 0u; status == SPARK_STATUS_OK && index < steps; index++ )
	{
		for ( lane = 0u; lane < batch; lane++ )
		{
			session->positions[lane] = prompt_tokens + index;
			session->sequence[lane] = lane;
			session->rows[lane] = lane;
		}
		memset(&step,0,sizeof(step));
		step.rows = batch;
		step.token_ids = token;
		step.positions = session->positions;
		step.sequence_of_row = session->sequence;
		step.head_rows = batch;
		step.head_row_index = session->rows;
		step.verify = exact;
		status = SparkDeviceStageRunnerStep(session->runner,&step,&result);
		for ( lane = 0u; status == SPARK_STATUS_OK && lane < batch; lane++ )
		{
			session->kept[lane] = 1u;
			token[lane] = result.argmax[lane];
			if ( trace != 0 )
			{
				const size_t at = (size_t)lane * steps + index;
				Top2(result.logits + (size_t)lane * result.logit_stride,vocab,&first,&trace->second[at],&trace->gap[at]);
				trace->token[at] = token[lane];
				if ( first != token[lane] )
					trace->gap[at] = 0.0f;
			}
		}
		if ( status == SPARK_STATUS_OK && exact != 0u )
			status = SparkDeviceStageRunnerKeepRows(session->runner,session->kept,batch);
	}
	*ns = NowNs() - start;
	return(status);
}

static SparkStatus LadderPrefill(Session *session, uint32_t batch, uint32_t **prompt, uint32_t prompt_tokens, uint32_t chunk, uint32_t *token)
{
	uint32_t lane;
	SparkStatus status = SPARK_STATUS_OK;
	for ( lane = 0u; status == SPARK_STATUS_OK && lane < batch; lane++ )
	{
		status = SparkDeviceStageRunnerResetSequence(session->runner,lane);
		if ( status == SPARK_STATUS_OK )
			status = SessionPrefillSequence(session,lane,prompt[lane],prompt_tokens,chunk,&token[lane]);
	}
	return(status);
}

static int RunLadder(Session *session, const RunOptions *options, const SparkTokenPrompts *prompts, const SparkDeviceStageRunnerInfo *info)
{
	const uint32_t most = options->max_batch, steps = options->new_tokens, length = options->prompt_tokens;
	const size_t cells = (size_t)most * steps;
	FILE *json = OpenOut(options,"ladder",".metal.json","w");
	uint32_t **prompt = (uint32_t **)calloc(most,sizeof(uint32_t *)), *token = (uint32_t *)calloc(most,sizeof(uint32_t)), lane, entry, repeat;
	uint32_t *first = (uint32_t *)calloc(most,sizeof(uint32_t));
	uint64_t *ns = (uint64_t *)calloc(options->repeats + 1u,sizeof(uint64_t)), solo_ns = 0u;
	LadderTrace solo, run;
	SparkStatus status;
	solo.token = (uint32_t *)calloc(cells,sizeof(uint32_t));
	solo.second = (uint32_t *)calloc(cells,sizeof(uint32_t));
	solo.gap = (float *)calloc(cells,sizeof(float));
	run.token = (uint32_t *)calloc(cells,sizeof(uint32_t));
	run.second = (uint32_t *)calloc(cells,sizeof(uint32_t));
	run.gap = (float *)calloc(cells,sizeof(float));
	status = json != 0 && prompt != 0 && token != 0 && first != 0 && ns != 0 && solo.token != 0 && solo.second != 0 && solo.gap != 0 &&
		run.token != 0 && run.second != 0 && run.gap != 0 ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED;
	for ( lane = 0u; status == SPARK_STATUS_OK && lane < most; lane++ )
		if ( (prompt[lane] = SynthesizeAt(prompts,length,(uint64_t)lane * RUN_PROMPT_STRIDE)) == 0 )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		fprintf(json,"{\"device\":\"%s\",\"streamed_bytes_per_token\":%llu,\"prompt_tokens\":%u,\"decode_steps\":%u,\"repeats\":%u,\"exact_rows\":%s,"
			"\"batches\":[",info->device,(unsigned long long)info->streamed_bytes_per_token,length,steps,options->repeats,options->exact_rows != 0u ? "true" : "false");
	for ( lane = 0u; status == SPARK_STATUS_OK && lane < most; lane++ )
	{
		LadderTrace view;
		uint64_t elapsed = 0u;
		view.token = solo.token + (size_t)lane * steps;
		view.second = solo.second + (size_t)lane * steps;
		view.gap = solo.gap + (size_t)lane * steps;
		status = SparkDeviceStageRunnerResetSequence(session->runner,0u);
		if ( status == SPARK_STATUS_OK )
			status = SessionPrefillSequence(session,0u,prompt[lane],length,options->prefill_rows,&token[0]);
		first[lane] = token[0];
		if ( status == SPARK_STATUS_OK )
			status = LadderDecode(session,1u,length,steps,info->vocab,options->exact_rows,token,&view,&elapsed);
		solo_ns += elapsed;
	}
	if ( status == SPARK_STATUS_OK )
		printf("solo references: %u sequences x %u steps in %.2f s\n",most,steps,solo_ns / 1e9);
	for ( entry = 0u; status == SPARK_STATUS_OK && entry < options->batch_count; entry++ )
	{
		const uint32_t batch = options->batches[entry];
		uint32_t identical = 0u, nondeterministic = 0u, divergences = 0u, index;
		uint64_t median;
		for ( repeat = 0u; status == SPARK_STATUS_OK && repeat <= options->repeats; repeat++ )
		{
			status = LadderPrefill(session,batch,prompt,length,options->prefill_rows,token);
			for ( lane = 0u; status == SPARK_STATUS_OK && lane < batch; lane++ )
				if ( token[lane] != first[lane] )
					nondeterministic++;
			if ( status == SPARK_STATUS_OK && repeat == 0u )
				status = LadderDecode(session,batch,length,steps,info->vocab,options->exact_rows,token,&run,&ns[repeat]);
			else if ( status == SPARK_STATUS_OK )
			{
				status = LadderDecode(session,batch,length,steps,info->vocab,options->exact_rows,token,0,&ns[repeat]);
				for ( lane = 0u; status == SPARK_STATUS_OK && lane < batch; lane++ )
					if ( token[lane] != run.token[(size_t)lane * steps + steps - 1u] )
						nondeterministic++;
			}
		}
		if ( status != SPARK_STATUS_OK )
			break;
		qsort(ns + 1,options->repeats,sizeof(uint64_t),CompareU64);
		median = ns[1u + options->repeats / 2u];
		fprintf(json,"%s\n{\"batch\":%u,\"step_ms\":%.4f,\"per_sequence_tokens_per_second\":%.4f,\"aggregate_tokens_per_second\":%.4f,\"runs_seconds\":[",
			entry != 0u ? "," : "",batch,median / 1e6 / steps,steps / (median / 1e9),(double)batch * steps / (median / 1e9));
		for ( repeat = 0u; repeat <= options->repeats; repeat++ )
			fprintf(json,"%s%.6f",repeat != 0u ? "," : "",ns[repeat] / 1e9);
		fprintf(json,"],\"warmup_discarded\":true,\"divergences\":[");
		for ( lane = 0u; lane < batch; lane++ )
		{
			for ( index = 0u; index < steps && run.token[(size_t)lane * steps + index] == solo.token[(size_t)lane * steps + index]; index++ )
				continue;
			if ( index == steps )
			{
				identical++;
				continue;
			}
			{
				const size_t at = (size_t)lane * steps + index;
				fprintf(json,"%s{\"sequence\":%u,\"step\":%u,\"solo_token\":%u,\"solo_second\":%u,\"solo_gap\":%.6g,\"batch_token\":%u,\"batch_second\":%u,"
					"\"batch_gap\":%.6g}",divergences != 0u ? "," : "",lane,index,solo.token[at],solo.second[at],solo.gap[at],run.token[at],run.second[at],
					run.gap[at]);
				divergences++;
			}
		}
		fprintf(json,"],\"sequences_identical_to_solo\":%u,\"nondeterministic\":%u}",identical,nondeterministic);
		printf("B%u: step %.3f ms, %.3f tok/s per sequence, %.2f tok/s aggregate, %u/%u sequences identical to solo%s\n",batch,median / 1e6 / steps,
			steps / (median / 1e9),(double)batch * steps / (median / 1e9),identical,batch,nondeterministic != 0u ? ", NONDETERMINISTIC" : "");
		fflush(stdout);
	}
	if ( json != 0 )
	{
		fprintf(json,"\n]}\n");
		fclose(json);
	}
	for ( lane = 0u; prompt != 0 && lane < most; lane++ )
		free(prompt[lane]);
	free(prompt);
	free(token);
	free(first);
	free(ns);
	free(solo.token);
	free(solo.second);
	free(solo.gap);
	free(run.token);
	free(run.second);
	free(run.gap);
	return(status == SPARK_STATUS_OK ? 0 : 1);
}

typedef struct SpecRun
{
	SparkSpeculationFrameModel *model;
	uint32_t *tokens;
	uint32_t count;
	uint32_t frames;
	uint32_t drafted;
	uint32_t accepted;
	uint64_t ns;
} SpecRun;

static SparkStatus SpecFrame(Session *session, const uint32_t *rows, uint32_t count, uint32_t first, SparkDeviceStageRunnerResult *result)
{
	SparkDeviceStageRunnerBatch batch;
	uint32_t row;
	for ( row = 0u; row < count; row++ )
	{
		session->positions[row] = first + row;
		session->sequence[row] = 0u;
		session->rows[row] = row;
	}
	memset(&batch,0,sizeof(batch));
	batch.rows = count;
	batch.token_ids = rows;
	batch.positions = session->positions;
	batch.sequence_of_row = session->sequence;
	batch.head_rows = count;
	batch.head_row_index = session->rows;
	batch.verify = 1u;
	return(SparkDeviceStageRunnerStep(session->runner,&batch,result));
}

static uint32_t SpecChoose(SpecRun *run, SparkSpeculationHazard *hazards, const SparkSpeculationLookupMatch *match, uint32_t feasible, uint32_t context,
	uint32_t *frame_class)
{
	SparkSpeculationFrameRequest request;
	SparkSpeculationFrameChoice choice;
	memset(&request,0,sizeof(request));
	*frame_class = SparkSpeculationFrameClass(match->match_length,0u);
	if ( run->model->classes[*frame_class].frames == 0u )
	{
		const float step = SparkSpeculationFrameTokenCost(run->model,context);
		uint32_t free_drafts = 0u, drafts;
		for ( drafts = 1u; drafts <= feasible; drafts++ )
		{
			const float cost = SparkSpeculationFrameCost(run->model,drafts + 1u,context);
			if ( cost > 0.0f && cost <= SPARK_SPECULATION_FRAME_MARGIN * step )
				free_drafts = drafts;
		}
		feasible = free_drafts != 0u ? free_drafts : 1u;
	}
	request.feasible = feasible;
	request.context = context;
	request.match_length = match->match_length;
	request.lanes = 1u;
	SparkSpeculationFrameChoose(run->model,&hazards[*frame_class],&request,&choice);
	return(choice.drafts < feasible ? choice.drafts : feasible);
}

static SparkStatus SpecGreedy(Session *session, const RunOptions *options, const SparkTokenPrompt *prompt, uint32_t speculate, SpecRun *run)
{
	SparkSpeculationHazard hazards[SPARK_SPECULATION_FRAME_CLASSES];
	const uint32_t capacity = prompt->count + options->new_tokens + options->draft_max + 2u;
	uint32_t *history = (uint32_t *)calloc(capacity,sizeof(uint32_t)), *heads = (uint32_t *)calloc(RUN_LOOKUP_BUCKETS,sizeof(uint32_t));
	uint32_t *chain = (uint32_t *)calloc(capacity,sizeof(uint32_t)), frame[SPARK_DEVICE_STAGE_VERIFY_ROWS], length = prompt->count, next = 0u;
	SparkSpeculationLookupIndex index;
	SparkDeviceStageRunnerResult result;
	SparkStatus status = history != 0 && heads != 0 && chain != 0 && run->tokens != 0 ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED;
	uint64_t start;
	if ( status == SPARK_STATUS_OK )
		status = SparkSpeculationLookupIndexBind(&index,heads,RUN_LOOKUP_BUCKETS,chain,capacity,RUN_LOOKUP_GRAM);
	if ( status == SPARK_STATUS_OK )
		status = SparkDeviceStageRunnerResetSequence(session->runner,0u);
	if ( status == SPARK_STATUS_OK )
		status = SessionPrefill(session,prompt->tokens,prompt->count,options->prefill_rows,&next,0);
	memcpy(history,prompt->tokens,(size_t)prompt->count * sizeof(uint32_t));
	run->count = run->frames = run->drafted = run->accepted = 0u;
	memset(hazards,0,sizeof(hazards));
	memset(run->model->classes,0,sizeof(run->model->classes));
	start = NowNs();
	while ( status == SPARK_STATUS_OK && run->count < options->new_tokens )
	{
		SparkSpeculationLookupMatch match = { 0u, 0u };
		uint32_t drafts = 0u, accepted = 0u, kept, position = length, frame_class = 0u;
		uint64_t before;
		history[length++] = next;
		run->tokens[run->count++] = next;
		status = SparkSpeculationLookupIndexAppend(&index,history,length);
		if ( status == SPARK_STATUS_OK && speculate != 0u &&
			SparkSpeculationLookupIndexFind(&index,history,RUN_LOOKUP_MAX_MATCH,RUN_LOOKUP_CANDIDATES,&match) == SPARK_STATUS_OK )
		{
			drafts = length - 1u - match.source_end;
			drafts = SpecChoose(run,hazards,&match,drafts < options->draft_max ? drafts : options->draft_max,length,&frame_class);
		}
		frame[0] = next;
		memcpy(frame + 1,history + match.source_end + 1u,(size_t)drafts * sizeof(uint32_t));
		if ( status != SPARK_STATUS_OK )
			break;
		before = NowNs();
		if ( drafts == 0u )
			status = SessionStep(session,&next,position,1u,0u,0,&result);
		else
			status = SpecFrame(session,frame,drafts + 1u,position,&result);
		if ( status == SPARK_STATUS_OK && drafts == 0u )
			SparkSpeculationFrameObserveChain(run->model,1u,length,NowNs() - before);
		if ( status != SPARK_STATUS_OK )
			break;
		while ( accepted < drafts && result.argmax[accepted] == frame[accepted + 1u] )
			accepted++;
		if ( drafts != 0u )
		{
			kept = accepted + 1u;
			status = SparkDeviceStageRunnerKeepRows(session->runner,&kept,1u);
			SparkSpeculationFrameObserveFrame(run->model,drafts + 1u,length,NowNs() - before);
			SparkSpeculationHazardObserve(&run->model->classes[frame_class],drafts,accepted,SPARK_SPECULATION_FRAME_ENGINE_DECAY);
			SparkSpeculationHazardObserve(&hazards[frame_class],drafts,accepted,SPARK_SPECULATION_FRAME_REQUEST_DECAY);
			run->frames++;
			run->drafted += drafts;
			run->accepted += accepted;
		}
		for ( kept = 0u; kept < accepted && run->count < options->new_tokens; kept++ )
		{
			history[length++] = frame[kept + 1u];
			run->tokens[run->count++] = frame[kept + 1u];
		}
		if ( kept < accepted )
			break;
		next = result.argmax[accepted];
	}
	run->ns = NowNs() - start;
	free(history);
	free(heads);
	free(chain);
	return(status);
}

static int CompareDoubles(const void *a, const void *b)
{
	const double x = *(const double *)a, y = *(const double *)b;
	return(x < y ? -1 : x > y ? 1 : 0);
}

static SparkStatus SpecCostCurve(Session *session, const RunOptions *options, const uint32_t *prompt, uint32_t length, SparkSpeculationFrameModel *model,
	FILE *json)
{
	double *verify = (double *)calloc(options->repeats + 1u,sizeof(double)), *keep = (double *)calloc(options->repeats + 1u,sizeof(double));
	SparkDeviceStageRunnerResult result;
	uint32_t size, repeat, next = 0u;
	SparkStatus status = verify != 0 && keep != 0 ? SPARK_STATUS_OK : SPARK_STATUS_CAPACITY_EXCEEDED;
	fprintf(json,"\"verify_frames\":[");
	for ( size = 0u; status == SPARK_STATUS_OK && size < RUN_FRAME_SIZES; size++ )
	{
		for ( repeat = 0u; status == SPARK_STATUS_OK && repeat <= options->repeats; repeat++ )
		{
			uint64_t before;
			uint32_t kept = run_frame_sizes[size];
			status = SparkDeviceStageRunnerResetSequence(session->runner,0u);
			if ( status == SPARK_STATUS_OK )
				status = SessionPrefill(session,prompt,length,options->prefill_rows,&next,0);
			before = NowNs();
			if ( status == SPARK_STATUS_OK )
				status = SpecFrame(session,prompt,run_frame_sizes[size],length,&result);
			verify[repeat] = (NowNs() - before) / 1e6;
			before = NowNs();
			if ( status == SPARK_STATUS_OK )
				status = SparkDeviceStageRunnerKeepRows(session->runner,&kept,1u);
			keep[repeat] = (NowNs() - before) / 1e6;
		}
		if ( status != SPARK_STATUS_OK )
			break;
		for ( repeat = 1u; repeat <= options->repeats; repeat++ )
			if ( run_frame_sizes[size] == 1u )
				SparkSpeculationFrameObserveChain(model,1u,length,(uint64_t)(verify[repeat] * 1e6));
			else
				SparkSpeculationFrameObserveFrame(model,run_frame_sizes[size],length,(uint64_t)((verify[repeat] + keep[repeat]) * 1e6));
		qsort(verify + 1,options->repeats,sizeof(double),CompareDoubles);
		qsort(keep + 1,options->repeats,sizeof(double),CompareDoubles);
		fprintf(json,"%s{\"rows\":%u,\"verify_ms\":%.4f,\"keep_all_ms\":%.4f}",size != 0u ? "," : "",run_frame_sizes[size],verify[1u + options->repeats / 2u],
			keep[1u + options->repeats / 2u]);
		printf("verify frame %2u rows: %.3f ms, keeping all rows %.3f ms\n",run_frame_sizes[size],verify[1u + options->repeats / 2u],keep[1u + options->repeats / 2u]);
	}
	fprintf(json,"]");
	free(verify);
	free(keep);
	return(status);
}

static int RunSpec(Session *session, const RunOptions *options, const SparkTokenPrompts *gate, const SparkTokenPrompts *extra)
{
	FILE *json = OpenOut(options,"spec",".metal.json","w");
	SparkSpeculationFrameModel *model = (SparkSpeculationFrameModel *)calloc(1u,sizeof(SparkSpeculationFrameModel));
	double *plain_rate = (double *)calloc(options->repeats + 1u,sizeof(double)), *spec_rate = (double *)calloc(options->repeats + 1u,sizeof(double));
	SpecRun plain, spec;
	uint32_t index, total = gate->count + extra->count, identical = 0u;
	SparkStatus status = json != 0 && model != 0 && plain_rate != 0 && spec_rate != 0 ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	plain.model = spec.model = model;
	plain.tokens = (uint32_t *)calloc(options->new_tokens,sizeof(uint32_t));
	spec.tokens = (uint32_t *)calloc(options->new_tokens,sizeof(uint32_t));
	if ( status == SPARK_STATUS_OK )
	{
		fprintf(json,"{\"new_tokens\":%u,\"draft_max\":%u,\"repeats\":%u,\"lookup\":{\"gram\":%u,\"max_match\":%u,\"candidates\":%u},",options->new_tokens,
			options->draft_max,options->repeats,RUN_LOOKUP_GRAM,RUN_LOOKUP_MAX_MATCH,RUN_LOOKUP_CANDIDATES);
		status = SpecCostCurve(session,options,extra->prompts[0].tokens,extra->prompts[0].count,model,json);
		fprintf(json,",\"prompts\":[");
	}
	for ( index = 0u; status == SPARK_STATUS_OK && index < total; index++ )
	{
		const SparkTokenPrompt *prompt = index < gate->count ? &gate->prompts[index] : &extra->prompts[index - gate->count];
		uint32_t same = options->new_tokens, run, token;
		for ( run = 0u; status == SPARK_STATUS_OK && run <= options->repeats; run++ )
		{
			status = SpecGreedy(session,options,prompt,0u,&plain);
			if ( status == SPARK_STATUS_OK )
				status = SpecGreedy(session,options,prompt,1u,&spec);
			plain_rate[run] = plain.count / (plain.ns / 1e9);
			spec_rate[run] = spec.count / (spec.ns / 1e9);
			for ( token = 0u; status == SPARK_STATUS_OK && token < same; token++ )
				if ( token >= plain.count || token >= spec.count || plain.tokens[token] != spec.tokens[token] )
					same = token;
		}
		if ( status != SPARK_STATUS_OK )
			break;
		identical += same == options->new_tokens ? 1u : 0u;
		qsort(plain_rate + 1,options->repeats,sizeof(double),CompareDoubles);
		qsort(spec_rate + 1,options->repeats,sizeof(double),CompareDoubles);
		fprintf(json,"%s{\"name\":\"%s\",\"prompt_tokens\":%u,\"tokens_identical\":%u,\"plain_tokens_per_second\":%.4f,\"spec_tokens_per_second\":%.4f,"
			"\"plain_runs_tokens_per_second\":[%.4f,%.4f],\"spec_runs_tokens_per_second\":[%.4f,%.4f],\"frames\":%u,\"drafted\":%u,\"accepted\":%u,"
			"\"spec_tokens\":[",index != 0u ? "," : "",prompt->name,prompt->count,same,plain_rate[1u + options->repeats / 2u],spec_rate[1u + options->repeats / 2u],
			plain_rate[1],plain_rate[options->repeats],spec_rate[1],spec_rate[options->repeats],spec.frames,spec.drafted,spec.accepted);
		for ( token = 0u; token < spec.count; token++ )
			fprintf(json,"%s%u",token != 0u ? "," : "",spec.tokens[token]);
		fprintf(json,"]}");
		printf("%s: %u of %u tokens identical in every run; %.2f tok/s plain (%.2f-%.2f), %.2f tok/s with lookup speculation (%.2f-%.2f); "
			"%u frames, %u of %u drafts accepted\n",prompt->name,same,options->new_tokens,plain_rate[1u + options->repeats / 2u],plain_rate[1],
			plain_rate[options->repeats],spec_rate[1u + options->repeats / 2u],spec_rate[1],spec_rate[options->repeats],spec.frames,spec.accepted,spec.drafted);
		fflush(stdout);
	}
	if ( json != 0 )
	{
		fprintf(json,"],\"prompts_identical\":%u,\"prompt_count\":%u}\n",identical,total);
		fclose(json);
	}
	free(plain.tokens);
	free(spec.tokens);
	free(model);
	free(plain_rate);
	free(spec_rate);
	return(status == SPARK_STATUS_OK && identical == total ? 0 : 1);
}

int main(int argc, char **argv)
{
	RunOptions options;
	SparkTokenPrompts prompts, extra;
	SparkDeviceStageRunnerConfig config;
	SparkDeviceStageRunnerInfo info;
	Session session;
	uint32_t index;
	int failed = 0;
	setvbuf(stdout,0,_IOLBF,0);
	memset(&config,0,sizeof(config));
	if ( ParseOptions(argc,argv,&options) != 0 || (config.model = RunModel(options.model)) == 0 )
		return(Usage(argv[0]));
	if ( SparkTokenPromptsLoad(options.prompts,&prompts) != SPARK_STATUS_OK )
		return(1);
	memset(&extra,0,sizeof(extra));
	if ( strcmp(options.mode,"spec") == 0 )
	{
		if ( options.spec_prompts == 0 || SparkTokenPromptsLoad(options.spec_prompts,&extra) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"metal_stage_run: spec mode needs --spec-prompts\n");
			return(1);
		}
		if ( extra.prompts[0].count < run_frame_sizes[RUN_FRAME_SIZES - 1u] )
		{
			fprintf(stderr,"metal_stage_run: the verify frames are built from the first speculation prompt, %s, which has %u tokens; it needs at least %u\n",
				extra.prompts[0].name,extra.prompts[0].count,run_frame_sizes[RUN_FRAME_SIZES - 1u]);
			return(1);
		}
	}
	mkdir(options.out,0755);
	config.pack_path = options.pack;
	config.max_rows = options.prefill_rows;
	config.sequences = strcmp(options.mode,"pairs") == 0 ? 2u : 1u;
	config.max_context = RUN_MAX_CONTEXT;
	config.head_rows = strcmp(options.mode,"teacher") == 0 ? options.prefill_rows : strcmp(options.mode,"pairs") == 0 ? 2u : 1u;
	config.state_budget_bytes = RUN_STATE_BUDGET;
	if ( strcmp(options.mode,"spec") == 0 )
	{
		config.head_rows = SPARK_DEVICE_STAGE_VERIFY_ROWS;
		config.max_context = RUN_MAX_CONTEXT;
	}
	if ( strcmp(options.mode,"ladder") == 0 )
	{
		config.sequences = config.head_rows = options.max_batch;
		config.max_context = options.prompt_tokens + options.new_tokens + 1u;
		config.state_budget_bytes = RUN_LADDER_STATE_BUDGET;
	}
	config.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	config.verify_pack = options.verify;
	memset(&session,0,sizeof(session));
	session.capacity = options.prefill_rows;
	session.rows = (uint32_t *)calloc(session.capacity,sizeof(uint32_t));
	session.positions = (uint32_t *)calloc(session.capacity,sizeof(uint32_t));
	session.sequence = (uint32_t *)calloc(session.capacity,sizeof(uint32_t));
	session.targets = (uint32_t *)calloc(session.capacity,sizeof(uint32_t));
	session.kept = (uint32_t *)calloc(session.capacity,sizeof(uint32_t));
	if ( session.rows == 0 || session.positions == 0 || session.sequence == 0 || session.targets == 0 || session.kept == 0 )
		return(1);
	{
		const uint64_t start = NowNs();
		if ( SparkDeviceStageRunnerOpen(&session.runner,&config) != SPARK_STATUS_OK )
			return(1);
		SparkDeviceStageRunnerInfoGet(session.runner,&info);
		session.layers = info.layers;
		printf("opened %s on %s (%s) in %.2f s: pack %llu bytes%s%s, device state %llu bytes\n",options.pack,info.device,info.target,
			(NowNs() - start) / 1e9,(unsigned long long)info.pack_bytes,options.verify != 0u ? ", sha256 " : "",options.verify != 0u ? info.pack_sha256 : "",
			(unsigned long long)info.device_state_bytes);
	}
	if ( strcmp(options.mode,"teacher") == 0 )
		for ( index = 0u; !failed && index < prompts.count; index++ )
			failed = RunTeacher(&session,&options,&prompts.prompts[index],info.vocab);
	else if ( strcmp(options.mode,"greedy") == 0 )
	{
		FILE *json = OpenOut(&options,"greedy",".metal.json","w");
		failed = json == 0;
		if ( !failed )
			fprintf(json,"{\"new_tokens\":%u,\"prompts\":[\n",options.new_tokens);
		for ( index = 0u; !failed && index < prompts.count; index++ )
			failed = RunGreedy(&session,&options,&prompts.prompts[index],json,index == 0u);
		if ( json != 0 )
		{
			fprintf(json,"\n]}\n");
			fclose(json);
		}
	}
	else if ( strcmp(options.mode,"pairs") == 0 )
	{
		FILE *json = OpenOut(&options,"pairs",".metal.json","w");
		failed = json == 0;
		if ( !failed )
			fprintf(json,"{\"new_tokens\":%u,\"prompts\":[\n",options.new_tokens);
		for ( index = 0u; !failed && index + 1u < prompts.count; index += 2u )
			failed = RunPair(&session,&options,&prompts.prompts[index],&prompts.prompts[index + 1u],json,index == 0u);
		if ( json != 0 )
		{
			fprintf(json,"\n]}\n");
			fclose(json);
		}
	}
	else if ( strcmp(options.mode,"bench") == 0 )
		failed = RunBench(&session,&options,&prompts,&info);
	else if ( strcmp(options.mode,"ladder") == 0 )
		failed = RunLadder(&session,&options,&prompts,&info);
	else if ( strcmp(options.mode,"spec") == 0 )
	{
		failed = RunSpec(&session,&options,&prompts,&extra);
		SparkTokenPromptsFree(&extra);
	}
	else
		failed = Usage(argv[0]);
	SparkDeviceStageRunnerClose(session.runner);
	SparkTokenPromptsFree(&prompts);
	free(session.rows);
	free(session.positions);
	free(session.sequence);
	free(session.targets);
	free(session.kept);
	free(session.pinned);
	if ( session.routes.host != 0 )
		SparkDeviceFree(&session.routes);
	return(failed ? 1 : 0);
}
