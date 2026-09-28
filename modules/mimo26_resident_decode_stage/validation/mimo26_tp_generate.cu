#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "sparkpipe/spark_mimo26_model.h"
#include "spark_mimo26_rank_engine.h"

#define GEN_WAIT_NS UINT64_C(300000000000)

static const uint32_t gen_stop_tokens[] = {151643u,151645u,151672u};
static uint32_t gen_rank;

static void GenFail(const char *what, const char *detail)
{
	fprintf(stderr,"M26GEN-FAIL rank=%u phase=%s %s\n",gen_rank,what,detail != 0 ? detail : "");
	fflush(stderr);
	exit(1);
}

static void GenStatus(SparkStatus status, const char *what)
{
	if ( status != SPARK_STATUS_OK )
		GenFail(what,SparkStatusToString(status));
}

static uint64_t GenNow(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static std::vector<uint32_t> GenReadTokens(const char *path)
{
	std::vector<uint32_t> tokens;
	FILE *file = fopen(path,"rb");
	int32_t value;
	if ( file == 0 )
		GenFail("tokens",path);
	while ( fread(&value,sizeof(value),1u,file) == 1u )
	{
		if ( value < 0 || (uint32_t)value >= SPARK_MIMO26_MODEL_VOCAB_COUNT )
			GenFail("tokens","id out of range");
		tokens.push_back((uint32_t)value);
	}
	fclose(file);
	if ( tokens.empty() )
		GenFail("tokens","empty");
	return(tokens);
}

static void GenBarrier(char expected)
{
	struct pollfd input = {STDIN_FILENO,POLLIN,0};
	char actual = 0;
	if ( poll(&input,1u,600000) != 1 || (input.revents & POLLIN) == 0 || read(STDIN_FILENO,&actual,1u) != 1 || actual != expected )
		GenFail("coordinator-barrier","no release from the coordinator");
}

static uint32_t GenIsStop(uint32_t token)
{
	uint32_t i;
	for (i = 0u; i < sizeof(gen_stop_tokens) / sizeof(gen_stop_tokens[0]); i++)
		if ( token == gen_stop_tokens[i] )
			return(1u);
	return(0u);
}

int main(int argc, char **argv)
{
	SparkMimo26RankEngineConfig config;
	SparkMimo26RankEngineStats stats;
	SparkMimo26RankEngine *engine = 0;
	std::vector<std::string> requests;
	char line[4096];
	FILE *list;
	uint32_t max_new, position, token, next, index, stop;
	uint64_t started, first_token_ns, decode_started;
	float score;
	if ( argc != 10 )
	{
		fprintf(stderr,"usage: %s RANK PACK PACK_SHA256 EXPERT_POOL_BYTES SPINE_BUDGET_BYTES eager|graph REQUEST_LIST MAX_NEW_TOKENS OUT_DIRECTORY\n",argv[0]);
		return(2);
	}
	gen_rank = (uint32_t)strtoul(argv[1],0,10);
	if ( getenv("SPARK_WEIGHTD_SOCKET") == 0 || getenv("SPARK_WEIGHTD_LANE") == 0 || getenv("SPARK_TP_MESH_RANKS") == 0 )
		GenFail("environment","SPARK_WEIGHTD_SOCKET / SPARK_WEIGHTD_LANE / SPARK_TP_MESH_RANKS unset");
	if ( strcmp(argv[6],"eager") != 0 && strcmp(argv[6],"graph") != 0 )
		GenFail("mode","eager or graph");
	list = fopen(argv[7],"r");
	if ( list == 0 )
		GenFail("request-list",argv[7]);
	while ( fgets(line,sizeof(line),list) != 0 )
	{
		line[strcspn(line,"\r\n")] = '\0';
		if ( line[0] != '\0' )
			requests.push_back(line);
	}
	fclose(list);
	max_new = (uint32_t)strtoul(argv[8],0,10);
	memset(&config,0,sizeof(config));
	config.rank = gen_rank;
	config.lane_count = 1u;
	config.max_positions = 8192u;
	config.mode = strcmp(argv[6],"graph") == 0 ? SPARK_MIMO26_RANK_ENGINE_MODE_GRAPH : SPARK_MIMO26_RANK_ENGINE_MODE_EAGER;
	config.expert_pool_bytes = strtoull(argv[4],0,10);
	config.spine_budget_bytes = strtoull(argv[5],0,10);
	config.wait_ns = GEN_WAIT_NS;
	config.pack_path = argv[2];
	config.pack_sha256 = argv[3];
	if ( requests.empty() || max_new == 0u )
		GenFail("arguments","empty request list or zero max tokens");
	GenStatus(SparkMimo26RankEngineCreate(&config,&engine),"engine-create");
	printf("M26TP-READY rank=%u lane=%s mesh=%s requests=%zu\n",gen_rank,getenv("SPARK_WEIGHTD_LANE"),getenv("SPARK_TP_MESH_RANKS"),requests.size());
	fflush(stdout);
	GenBarrier('G');
	for (index = 0u; index < requests.size(); index++)
	{
		std::vector<uint32_t> prompt = GenReadTokens(requests[index].c_str()), generated;
		std::string out_path = std::string(argv[9]) + "/" + requests[index].substr(requests[index].find_last_of('/') + 1u) + ".out.i32";
		if ( prompt.size() + max_new > config.max_positions )
			GenFail("positions",requests[index].c_str());
		started = GenNow();
		first_token_ns = 0u;
		decode_started = 0u;
		stop = 0u;
		next = 0u;
		for (position = 0u; stop == 0u && generated.size() < max_new; position++)
		{
			if ( position + 1u < prompt.size() )
			{
				uint32_t lanes[SPARK_MIMO26_RANK_ENGINE_MAX_ROWS],positions[SPARK_MIMO26_RANK_ENGINE_MAX_ROWS],outputs[SPARK_MIMO26_RANK_ENGINE_MAX_ROWS],count,row;
				count = (uint32_t)prompt.size() - position < SPARK_MIMO26_RANK_ENGINE_MAX_ROWS ? (uint32_t)prompt.size() - position : SPARK_MIMO26_RANK_ENGINE_MAX_ROWS;
				for (row = 0u; row < count; row++)
				{
					lanes[row] = 0u;
					positions[row] = position + row;
				}
				GenStatus(SparkMimo26RankEngineRows(engine,count,lanes,&prompt[position],positions,outputs,0),"prefill-rows");
				position += count - 1u;
				next = outputs[count - 1u];
				if ( position + 1u < prompt.size() )
					continue;
			}
			else
			{
				token = position < prompt.size() ? prompt[position] : next;
				GenStatus(SparkMimo26RankEngineStep(engine,0u,token,position,&next,&score),"step");
			}
			generated.push_back(next);
			if ( first_token_ns == 0u )
			{
				first_token_ns = GenNow() - started;
				decode_started = GenNow();
			}
			stop = GenIsStop(next);
		}
		if ( gen_rank == 0u )
		{
			FILE *out = fopen(out_path.c_str(),"wb");
			for (position = 0u; out != 0 && position < generated.size(); position++)
			{
				int32_t value = (int32_t)generated[position];
				fwrite(&value,sizeof(value),1u,out);
			}
			if ( out == 0 || fclose(out) != 0 )
				GenFail("output",out_path.c_str());
		}
		printf("M26GEN rank=%u request=%s prompt=%zu ttft_ms=%.1f generated=%zu stop=%s decode_ms_per_token=%.2f\n",gen_rank,requests[index].c_str(),prompt.size(),(double)first_token_ns / 1e6,generated.size(),stop != 0u ? "eos" : "length",generated.size() > 1u ? (double)(GenNow() - decode_started) / 1e6 / (double)(generated.size() - 1u) : 0.0);
		fflush(stdout);
	}
	GenStatus(SparkMimo26RankEngineReadStats(engine,&stats),"stats");
	printf("M26GEN-SUMMARY rank=%u steps=%llu graph_steps=%llu kv_error=%u step_ms=%.2f\n",gen_rank,(unsigned long long)stats.steps,(unsigned long long)stats.graph_steps,stats.kv_error,(double)stats.step_ns / 1e6 / (double)stats.steps);
	fflush(stdout);
	GenStatus(SparkMimo26RankEngineDestroy(engine),"engine-destroy");
	return(stats.kv_error != 0u ? 1 : 0);
}
