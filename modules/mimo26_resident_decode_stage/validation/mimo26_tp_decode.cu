#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "sparkpipe/spark_mimo26_model.h"
#include "spark_mimo26_rank_engine.h"

#define TP_WAIT_NS UINT64_C(300000000000)

static uint32_t tp_rank;

static void TpFail(const char *what, const char *detail)
{
	fprintf(stderr,"M26TP-FAIL rank=%u phase=%s %s\n",tp_rank,what,detail != 0 ? detail : "");
	fflush(stderr);
	exit(1);
}

static void TpStatus(SparkStatus status, const char *what)
{
	if ( status != SPARK_STATUS_OK )
		TpFail(what,SparkStatusToString(status));
}

static uint64_t TpNow(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC,&now);
	return((uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec);
}

static std::vector<uint32_t> TpReadTokens(const char *path)
{
	std::vector<uint32_t> tokens;
	FILE *file = fopen(path,"rb");
	int32_t value;
	if ( file == 0 )
		TpFail("tokens",path);
	while ( fread(&value,sizeof(value),1u,file) == 1u )
	{
		if ( value < 0 || (uint32_t)value >= SPARK_MIMO26_MODEL_VOCAB_COUNT )
			TpFail("tokens","id out of range");
		tokens.push_back((uint32_t)value);
	}
	fclose(file);
	if ( tokens.empty() )
		TpFail("tokens","empty");
	return(tokens);
}

static void TpBarrier(char expected)
{
	struct pollfd input = {STDIN_FILENO,POLLIN,0};
	char actual = 0;
	if ( poll(&input,1u,600000) != 1 || (input.revents & POLLIN) == 0 || read(STDIN_FILENO,&actual,1u) != 1 || actual != expected )
		TpFail("coordinator-barrier","no release from the coordinator");
}

int main(int argc, char **argv)
{
	SparkMimo26RankEngineConfig config;
	SparkMimo26RankEngineStats stats;
	SparkMimo26RankEngine *engine = 0;
	uint32_t position, token, next, mismatches = 0u, total, lane, repeat, repeats;
	uint64_t decode_started = 0u, decode_ns = 0u, prompt_started, prompt_ns = 0u;
	float score;
	if ( argc != 11 && argc != 12 )
	{
		fprintf(stderr,"usage: %s RANK PACK PACK_SHA256 PROMPT_I32 EXPECTED_I32 EXPERT_POOL_BYTES SPINE_BUDGET_BYTES eager|graph LANE REPEATS [DUMP_DIRECTORY]\n",argv[0]);
		return(2);
	}
	tp_rank = (uint32_t)strtoul(argv[1],0,10);
	if ( getenv("SPARK_WEIGHTD_SOCKET") == 0 || getenv("SPARK_WEIGHTD_LANE") == 0 || getenv("SPARK_TP_MESH_RANKS") == 0 )
		TpFail("environment","SPARK_WEIGHTD_SOCKET / SPARK_WEIGHTD_LANE / SPARK_TP_MESH_RANKS unset");
	if ( strcmp(argv[8],"eager") != 0 && strcmp(argv[8],"graph") != 0 )
		TpFail("mode","eager or graph");
	std::vector<uint32_t> prompt = TpReadTokens(argv[4]), expected = TpReadTokens(argv[5]), generated;
	total = (uint32_t)(prompt.size() + expected.size());
	lane = (uint32_t)strtoul(argv[9],0,10);
	repeats = (uint32_t)strtoul(argv[10],0,10);
	memset(&config,0,sizeof(config));
	config.rank = tp_rank;
	config.lane_count = 2u;
	config.max_positions = 1024u;
	config.mode = strcmp(argv[8],"graph") == 0 ? SPARK_MIMO26_RANK_ENGINE_MODE_GRAPH : SPARK_MIMO26_RANK_ENGINE_MODE_EAGER;
	config.expert_pool_bytes = strtoull(argv[6],0,10);
	config.spine_budget_bytes = strtoull(argv[7],0,10);
	config.wait_ns = TP_WAIT_NS;
	config.pack_path = argv[2];
	config.pack_sha256 = argv[3];
	config.dump_directory = argc == 12 ? argv[11] : 0;
	if ( total > config.max_positions || lane >= config.lane_count || repeats == 0u )
		TpFail("arguments","positions, lane or repeats out of range");
	TpStatus(SparkMimo26RankEngineCreate(&config,&engine),"engine-create");
	TpStatus(SparkMimo26RankEngineReadStats(engine,&stats),"stats");
	printf("M26TP-ATTACH rank=%u spine_bytes=%llu expert_pool_bytes=%llu pinned_experts=%u mode=%s\n",tp_rank,(unsigned long long)stats.spine_bytes,(unsigned long long)stats.expert_pool_bytes,stats.pinned_experts,argv[8]);
	printf("M26TP-READY rank=%u lane=%s mesh=%s prompt=%zu expected=%zu\n",tp_rank,getenv("SPARK_WEIGHTD_LANE"),getenv("SPARK_TP_MESH_RANKS"),prompt.size(),expected.size());
	fflush(stdout);
	TpBarrier('G');
	for (repeat = 0u; repeat < repeats; repeat++)
	{
		generated.clear();
		mismatches = 0u;
		prompt_started = TpNow();
		for (position = 0u; position + 1u < total; position++)
		{
			if ( position + 1u == prompt.size() )
			{
				decode_started = TpNow();
				prompt_ns = decode_started - prompt_started;
			}
			token = position < prompt.size() ? prompt[position] : generated[position - prompt.size()];
			TpStatus(SparkMimo26RankEngineStep(engine,lane,token,position,&next,&score),"step");
			if ( position + 1u >= prompt.size() )
			{
				generated.push_back(next);
				mismatches += next != expected[position + 1u - prompt.size()] ? 1u : 0u;
				if ( repeat == 0u )
					printf("M26TP-TOKEN rank=%u position=%u greedy=%u reference=%u score=%.4f\n",tp_rank,position,next,expected[position + 1u - prompt.size()],score);
			}
		}
		decode_ns = TpNow() - decode_started;
		printf("M26TP-REPEAT rank=%u repeat=%u tokens=%zu mismatches=%u prompt_ms=%.2f decode_ms_per_token=%.2f\n",tp_rank,repeat,generated.size(),mismatches,(double)prompt_ns / 1e6,(double)decode_ns / 1e6 / (double)generated.size());
		fflush(stdout);
		if ( mismatches != 0u )
			break;
	}
	TpStatus(SparkMimo26RankEngineReadStats(engine,&stats),"stats");
	printf("M26TP-SUMMARY rank=%u tokens=%zu mismatches=%u kv_error=%u steps=%llu graph_steps=%llu collectives=%llu step_ms=%.2f decode_ms_per_token=%.2f\n",tp_rank,generated.size(),mismatches,stats.kv_error,(unsigned long long)stats.steps,(unsigned long long)stats.graph_steps,(unsigned long long)stats.collectives,(double)stats.step_ns / 1e6 / (double)stats.steps,(double)decode_ns / 1e6 / (double)generated.size());
	fflush(stdout);
	TpStatus(SparkMimo26RankEngineDestroy(engine),"engine-destroy");
	if ( mismatches != 0u || stats.kv_error != 0u )
	{
		fprintf(stderr,"M26TP-FAIL rank=%u greedy tokens differ from the CPU reference\n",tp_rank);
		return(1);
	}
	printf("M26TP-PASS rank=%u %zu greedy tokens equal the CPU reference\n",tp_rank,generated.size());
	return(0);
}
