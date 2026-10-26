#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "inference/runner/device_stage_runner.h"
#include "sparkpipe/spark_qwen38_device_stage.h"
#include "sparkpipe/spark_weight_codec.h"
#include "tools/token_prompts.h"

#define RUN_MAX_CONTEXT 9216u
#define RUN_STATE_BUDGET (16ull << 30)

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
} RunOptions;

static uint64_t NowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static int Usage(const char *program)
{
	fprintf(stderr,"usage: %s --pack PACK --prompts PROMPTS.json --mode teacher|greedy|pairs|bench --out DIR [--prefill-rows N] [--new-tokens N] "
		"[--repeats N] [--bench-prompt NAME] [--verify-pack]\n",program);
	return(2);
}

static int ParseOptions(int argc, char **argv, RunOptions *options)
{
	int index;
	memset(options,0,sizeof(*options));
	options->prefill_rows = 1024u;
	options->new_tokens = 128u;
	options->repeats = 3u;
	for ( index = 1; index < argc; index++ )
	{
		const char *flag = argv[index], *value = index + 1 < argc ? argv[index + 1] : 0;
		if ( strcmp(flag,"--verify-pack") == 0 )
		{
			options->verify = 1u;
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
		else
			return(-1);
	}
	if ( options->pack == 0 || options->prompts == 0 || options->mode == 0 || options->out == 0 || options->prefill_rows == 0u ||
		options->prefill_rows > 2048u || options->repeats == 0u )
		return(-1);
	return(0);
}

typedef struct Session
{
	SparkDeviceStageRunner *runner;
	uint32_t *rows;
	uint32_t *positions;
	uint32_t *sequence;
	uint32_t *targets;
	uint32_t capacity;
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

static int RunTeacher(Session *session, const RunOptions *options, const SparkTokenPrompt *prompt, uint32_t vocab)
{
	const uint32_t positions = prompt->count - 1u, prefill = (positions + 1u) / 2u;
	SparkDeviceStageRunnerResult result;
	FILE *logits = OpenOut(options,prompt->name,".metal.logits","wb"), *json = OpenOut(options,prompt->name,".metal.json","w");
	float *logprob = (float *)calloc(positions,sizeof(float));
	uint32_t *argmax = (uint32_t *)calloc(positions,sizeof(uint32_t)), first, row;
	SparkStatus status = logits != 0 && json != 0 && logprob != 0 && argmax != 0 ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
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
		qsort(totals + 1,options->repeats,sizeof(uint64_t),CompareU64);
		median = totals[1u + options->repeats / 2u];
		fprintf(json,"],\"median_tokens_per_second\":%.4f}",steps / (median / 1e9));
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
		fprintf(json,"{\"device\":\"%s\",\"target\":\"%s\",\"pack_bytes\":%llu,\"streamed_bytes_per_token\":%llu,",info->device,info->target,
			(unsigned long long)info->pack_bytes,(unsigned long long)info->streamed_bytes_per_token);
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

int main(int argc, char **argv)
{
	RunOptions options;
	SparkTokenPrompts prompts;
	SparkDeviceStageRunnerConfig config;
	SparkDeviceStageRunnerInfo info;
	Session session;
	uint32_t index;
	int failed = 0;
	if ( ParseOptions(argc,argv,&options) != 0 )
		return(Usage(argv[0]));
	if ( SparkTokenPromptsLoad(options.prompts,&prompts) != SPARK_STATUS_OK )
		return(1);
	mkdir(options.out,0755);
	memset(&config,0,sizeof(config));
	config.pack_path = options.pack;
	config.model = SparkQwen38_27bDeviceStageModel();
	config.max_rows = options.prefill_rows;
	config.sequences = strcmp(options.mode,"pairs") == 0 ? 2u : 1u;
	config.max_context = RUN_MAX_CONTEXT;
	config.head_rows = strcmp(options.mode,"teacher") == 0 ? options.prefill_rows : strcmp(options.mode,"pairs") == 0 ? 2u : 1u;
	config.state_budget_bytes = RUN_STATE_BUDGET;
	config.linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	config.verify_pack = options.verify;
	memset(&session,0,sizeof(session));
	session.capacity = options.prefill_rows;
	session.rows = (uint32_t *)calloc(session.capacity,sizeof(uint32_t));
	session.positions = (uint32_t *)calloc(session.capacity,sizeof(uint32_t));
	session.sequence = (uint32_t *)calloc(session.capacity,sizeof(uint32_t));
	session.targets = (uint32_t *)calloc(session.capacity,sizeof(uint32_t));
	{
		const uint64_t start = NowNs();
		if ( SparkDeviceStageRunnerOpen(&session.runner,&config) != SPARK_STATUS_OK )
			return(1);
		SparkDeviceStageRunnerInfoGet(session.runner,&info);
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
	else
		failed = Usage(argv[0]);
	SparkDeviceStageRunnerClose(session.runner);
	SparkTokenPromptsFree(&prompts);
	free(session.rows);
	free(session.positions);
	free(session.sequence);
	free(session.targets);
	return(failed ? 1 : 0);
}
