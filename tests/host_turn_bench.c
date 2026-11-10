#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE 1
#endif

#include <errno.h>
#include <ftw.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <libproc.h>
#include <mach/mach_time.h>
#include <pthread/qos.h>
#endif

#include "fixtures/model_resident_deployment_fixture.h"
#include "fixtures/test_child_guard.h"
#include "fixtures/test_tcp_port.h"
#include "sparkpipe/spark_model_batch_engine.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#define HOST_TURN_RANKS 16u
#define HOST_TURN_MAX_LANES 16u
#define HOST_TURN_MAX_REQUESTS 128u
#define HOST_TURN_KINDS 5u
#define HOST_TURN_PENDING 64u
#define HOST_TURN_EOS 2u
#define HOST_TURN_CONTEXT_TOKENS 131072u
#define HOST_TURN_WARM_TOKENS 64u
#define HOST_TURN_WARM_FRAMES 64u
#define HOST_TURN_DEADLINE_SECONDS 300u
#define HOST_TURN_POLL_MS 100
#define HOST_TURN_MAX_FRAME_US 1000000u
#define HOST_TURN_PATH_BYTES PATH_MAX

typedef struct HostTurnCounters
{
	uint64_t frames_by_kind[HOST_TURN_KINDS];
	uint64_t kind_cpu_ns[HOST_TURN_KINDS];
	uint64_t kind_wall_ns[HOST_TURN_KINDS];
	uint64_t kind_service_ns[HOST_TURN_KINDS];
	uint64_t kind_done[HOST_TURN_KINDS];
	uint64_t last_done_cpu_ns;
	uint64_t last_done_wall_ns;
	uint64_t pending_id[HOST_TURN_PENDING];
	uint32_t pending_count[HOST_TURN_PENDING];
	uint64_t pending_service[HOST_TURN_PENDING];
	uint64_t tokens;
	uint64_t completed_requests;
	uint64_t errors;
	uint32_t generated[HOST_TURN_MAX_REQUESTS];
} HostTurnCounters;

typedef struct HostTurnSample
{
	uint64_t wall_ns;
	uint64_t coordinator_ns;
	uint64_t rank_ns[HOST_TURN_RANKS];
	uint64_t frames;
	uint64_t continued;
	uint64_t frames_by_kind[HOST_TURN_KINDS];
	uint64_t kind_cpu_ns[HOST_TURN_KINDS];
	uint64_t kind_wall_ns[HOST_TURN_KINDS];
	uint64_t kind_service_ns[HOST_TURN_KINDS];
	uint64_t kind_done[HOST_TURN_KINDS];
	uint64_t tokens;
} HostTurnSample;

static HostTurnCounters HostTurn;
static uint32_t HostTurnFrameUs;
static pid_t HostTurnRanks[HOST_TURN_RANKS];
static char HostTurnWork[HOST_TURN_PATH_BYTES];
static const char *const HostTurnKindNames[HOST_TURN_KINDS] = {"none","prefill","decode","release","publish"};

static void HostTurnStopRanks(void)
{
	uint32_t rank;
	int status;
	for (rank=0u; rank<HOST_TURN_RANKS; rank++)
		if ( HostTurnRanks[rank] > 0 )
			(void)kill(HostTurnRanks[rank],SIGTERM);
	for (rank=0u; rank<HOST_TURN_RANKS; rank++)
		if ( HostTurnRanks[rank] > 0 )
		{
			(void)waitpid(HostTurnRanks[rank],&status,0);
			HostTurnRanks[rank] = 0;
		}
}

static void HostTurnFail(const char *what,uint32_t status)
{
	fprintf(stderr,"host_turn_bench: %s (status %u); rank logs kept in %s\n",what,status,HostTurnWork);
	HostTurnStopRanks();
	exit(2);
}

static uint64_t HostTurnNowNs(clockid_t clock)
{
	struct timespec now;
	clock_gettime(clock,&now);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static uint64_t HostTurnProcessCpuNs(pid_t pid)
{
#if defined(__APPLE__)
	static mach_timebase_info_data_t timebase;
	struct rusage_info_v2 info;
	if ( timebase.denom == 0u )
		mach_timebase_info(&timebase);
	if ( proc_pid_rusage(pid,RUSAGE_INFO_V2,(rusage_info_t *)&info) != 0 )
		HostTurnFail("cannot read a rank's CPU time",(uint32_t)errno);
	return((info.ri_user_time + info.ri_system_time) * timebase.numer / timebase.denom);
#else
	clockid_t clock;
	if ( clock_getcpuclockid(pid,&clock) != 0 )
		HostTurnFail("cannot read a rank's CPU time",(uint32_t)errno);
	return(HostTurnNowNs(clock));
#endif
}

static void HostTurnEvent(void *context,const SparkModelBatchEvent *event)
{
	(void)context;
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_TOKEN )
	{
		HostTurn.tokens++;
		if ( event->request_id >= 1u && event->request_id <= HOST_TURN_MAX_REQUESTS )
			HostTurn.generated[event->request_id - 1u]++;
	}
	else if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED )
		HostTurn.completed_requests++;
	else if ( event->kind == SPARK_MODEL_BATCH_EVENT_ERROR || event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED )
	{
		HostTurn.errors++;
		fprintf(stderr,"host_turn_bench: event kind %u status %u request %llu\n",event->kind,event->status,(unsigned long long)event->request_id);
	}
}

static void HostTurnStageCompletion(void *context,const SparkModelPipelineStageCompletion *completion)
{
	uint32_t slot = (uint32_t)(completion->submission_id % HOST_TURN_PENDING);
	uint64_t cpu_ns,wall_ns;
	(void)context;
	if ( completion->work_kind >= HOST_TURN_KINDS )
		return;
	if ( completion->stage_index == 0u )
		HostTurn.frames_by_kind[completion->work_kind]++;
	if ( HostTurn.pending_id[slot] != completion->submission_id )
	{
		HostTurn.pending_id[slot] = completion->submission_id;
		HostTurn.pending_count[slot] = 0u;
		HostTurn.pending_service[slot] = 0u;
	}
	if ( completion->service_time_ns > HostTurn.pending_service[slot] )
		HostTurn.pending_service[slot] = completion->service_time_ns;
	if ( ++HostTurn.pending_count[slot] != HOST_TURN_RANKS )
		return;
	cpu_ns = HostTurnNowNs(CLOCK_THREAD_CPUTIME_ID);
	wall_ns = HostTurnNowNs(CLOCK_MONOTONIC);
	if ( HostTurn.last_done_wall_ns != 0u )
	{
		HostTurn.kind_cpu_ns[completion->work_kind] += cpu_ns - HostTurn.last_done_cpu_ns;
		HostTurn.kind_wall_ns[completion->work_kind] += wall_ns - HostTurn.last_done_wall_ns;
		HostTurn.kind_service_ns[completion->work_kind] += HostTurn.pending_service[slot];
		HostTurn.kind_done[completion->work_kind]++;
	}
	HostTurn.last_done_cpu_ns = cpu_ns;
	HostTurn.last_done_wall_ns = wall_ns;
}

static void HostTurnPath(char *path,const char *name)
{
	if ( snprintf(path,HOST_TURN_PATH_BYTES,"%s/%s",HostTurnWork,name) >= (int)HOST_TURN_PATH_BYTES )
		HostTurnFail("work path too long",0u);
}

static void HostTurnWriteText(const char *path,const char *text)
{
	FILE *stream = fopen(path,"wb");
	if ( stream == 0 || fputs(text,stream) == EOF || fclose(stream) != 0 )
		HostTurnFail("cannot write a work file",(uint32_t)errno);
}

static void HostTurnCopyFile(const char *source,const char *name)
{
	char buffer[65536],path[HOST_TURN_PATH_BYTES];
	FILE *in,*out;
	size_t bytes;
	HostTurnPath(path,name);
	in = fopen(source,"rb");
	out = fopen(path,"wb");
	if ( in == 0 || out == 0 )
		HostTurnFail("cannot copy a module into the work directory",(uint32_t)errno);
	while ( (bytes = fread(buffer,1u,sizeof(buffer),in)) != 0u )
		if ( fwrite(buffer,1u,bytes,out) != bytes )
			HostTurnFail("cannot copy a module into the work directory",(uint32_t)errno);
	if ( ferror(in) != 0 || fclose(in) != 0 || fclose(out) != 0 )
		HostTurnFail("cannot copy a module into the work directory",(uint32_t)errno);
}

static void HostTurnMakeWork(void)
{
	char pattern[] = "build/host-turn-bench-XXXXXX";
	if ( mkdtemp(pattern) == 0 || realpath(pattern,HostTurnWork) == 0 )
	{
		fprintf(stderr,"host_turn_bench: cannot create a work directory under build/ (run from the repository root)\n");
		exit(2);
	}
}

static int HostTurnRemoveEntry(const char *path,const struct stat *status,int type,struct FTW *walk)
{
	(void)status;
	(void)walk;
	return(type == FTW_DP ? rmdir(path) : unlink(path));
}

static void HostTurnRemoveWork(void)
{
	if ( nftw(HostTurnWork,HostTurnRemoveEntry,16,FTW_DEPTH | FTW_PHYS) != 0 )
		fprintf(stderr,"host_turn_bench: could not remove %s\n",HostTurnWork);
}

static void HostTurnWriteDeployment(char *deployment_path)
{
	SparkModelResidentEndpoint endpoints[HOST_TURN_RANKS];
	const char *roots[HOST_TURN_RANKS],*hosts[HOST_TURN_RANKS];
	char host_names[HOST_TURN_RANKS][32],path[HOST_TURN_PATH_BYTES],text[64];
	TestModelResidentDeploymentFixture fixture;
	uint32_t rank,port;
	port = TestTcpPortRange(HOST_TURN_RANKS);
	if ( port == 0u )
		HostTurnFail("no free port range",0u);
	HostTurnPath(path,"adapter.json");
	snprintf(text,sizeof(text),"{\"frame_us\":%u}\n",HostTurnFrameUs);
	HostTurnWriteText(path,text);
	HostTurnCopyFile(HOST_TURN_BENCH_ADAPTER_PATH,"adapter.module");
	HostTurnCopyFile(HOST_TURN_BENCH_TRANSPORT_PATH,"transport.module");
	for (rank=0u; rank<HOST_TURN_RANKS; rank++)
	{
		roots[rank] = HostTurnWork;
		snprintf(host_names[rank],sizeof(host_names[rank]),"bench-rank-%02u",rank);
		hosts[rank] = host_names[rank];
		memset(&endpoints[rank],0,sizeof(endpoints[rank]));
		endpoints[rank].abi_version = SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION;
		endpoints[rank].descriptor_bytes = SPARK_MODEL_RESIDENT_ENDPOINT_BYTES;
		endpoints[rank].kind = SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP;
		endpoints[rank].tcp_host = "127.0.0.1";
		endpoints[rank].tcp_port = port + rank;
	}
	memset(&fixture,0,sizeof(fixture));
	fixture.adapter_shared_object_path = "adapter.module";
	fixture.driver_shared_object_path = "adapter.module";
	fixture.driver_program_name = "resident_decode";
	fixture.transport_shared_object_path = "transport.module";
	fixture.transport_mode = "host-rdma";
	fixture.node_target = "host.turn.bench";
	fixture.adapter_configuration_path = "adapter.json";
	fixture.runtime_roots = roots;
	fixture.transport_hosts = hosts;
	fixture.control_endpoints = endpoints;
	fixture.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	fixture.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	fixture.runtime_limits.max_inflight_submission_count = 4u;
	fixture.runtime_limits.max_active_sequence_count = HOST_TURN_MAX_LANES;
	fixture.runtime_limits.max_input_row_count = 1024u;
	fixture.runtime_limits.resident_sequence_capacity = HOST_TURN_MAX_REQUESTS;
	fixture.runtime_limits.kv_logical_page_capacity = 524288u;
	fixture.runtime_limits.kv_physical_page_capacity = 32768u;
	fixture.runtime_limits.output_vocabulary = 151936u;
	fixture.max_sequence_positions = HOST_TURN_CONTEXT_TOKENS;
	fixture.control_port_base = port + 1000u;
	fixture.node_count = HOST_TURN_RANKS;
	fixture.coordinator_rank_index = 0u;
	fixture.eos_token_id = HOST_TURN_EOS;
	HostTurnPath(deployment_path,"deployment.json");
	if ( TestModelResidentDeploymentWrite(deployment_path,&fixture) != 0 )
		HostTurnFail("cannot write the deployment",0u);
}

static void HostTurnStartRanks(const char *deployment_path)
{
	char rank_text[16],log_name[32],log_path[HOST_TURN_PATH_BYTES];
	uint32_t rank;
	for (rank=0u; rank<HOST_TURN_RANKS; rank++)
	{
		snprintf(rank_text,sizeof(rank_text),"%u",rank);
		snprintf(log_name,sizeof(log_name),"rank%02u.log",rank);
		HostTurnPath(log_path,log_name);
		HostTurnRanks[rank] = TestChildGuardFork();
		if ( HostTurnRanks[rank] < 0 )
			HostTurnFail("cannot start a rank",(uint32_t)errno);
		if ( HostTurnRanks[rank] == 0 )
		{
			if ( freopen(log_path,"wb",stderr) == 0 || dup2(fileno(stderr),fileno(stdout)) < 0 )
				_exit(126);
			execl(HOST_TURN_BENCH_RESIDENTD_PATH,HOST_TURN_BENCH_RESIDENTD_PATH,"--deployment",deployment_path,"--rank-index",rank_text,(char *)0);
			_exit(127);
		}
	}
}

static SparkModelBatchEngine *HostTurnConnect(const SparkModelResidentDeployment *deployment)
{
	char index_path[HOST_TURN_PATH_BYTES];
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchEngine *engine = 0;
	uint32_t attempt;
	SparkStatus status = SPARK_STATUS_IO_ERROR;
	HostTurnPath(index_path,"prefix.spi");
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_BATCH_ENGINE_CONFIGURATION_BYTES;
	configuration.connect_timeout_ms = 1000u;
	configuration.request_capacity = HOST_TURN_MAX_REQUESTS;
	configuration.max_context_tokens = HOST_TURN_CONTEXT_TOKENS;
	configuration.max_prefill_rows_per_submission = deployment->runtime_limits.max_input_row_count;
	configuration.maximum_messages_per_rank_per_progress = 8u;
	configuration.inflight_budget_ns = SPARK_MODEL_BATCH_ENGINE_DEFAULT_INFLIGHT_BUDGET_NS;
	configuration.deployment = deployment;
	configuration.runtime_root = HostTurnWork;
	configuration.prefix_index_path = index_path;
	configuration.event_function = HostTurnEvent;
	configuration.stage_completion_function = HostTurnStageCompletion;
	for (attempt=0u; attempt<300u && status != SPARK_STATUS_OK; attempt++)
	{
		status = SparkModelBatchEngineConnect(&configuration,&engine);
		if ( status != SPARK_STATUS_OK )
			usleep(20000);
	}
	if ( status != SPARK_STATUS_OK )
		HostTurnFail("the engine could not connect to the ranks",(uint32_t)status);
	return(engine);
}

static void HostTurnTurn(SparkModelBatchEngine *engine)
{
	SparkModelResidentClientPollDescriptor descriptors[HOST_TURN_RANKS];
	struct pollfd fds[HOST_TURN_RANKS];
	uint32_t count = 0u,index;
	uint64_t next_ns,now_ns;
	int timeout_ms;
	SparkStatus status;
	status = SparkModelBatchEngineProgress(engine,4u);
	if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY && status != SPARK_STATUS_IO_ERROR )
		HostTurnFail("engine progress failed",(uint32_t)status);
	status = SparkModelBatchEngineGetPollDescriptors(engine,descriptors,HOST_TURN_RANKS,&count);
	if ( status != SPARK_STATUS_OK )
		HostTurnFail("engine poll descriptors failed",(uint32_t)status);
	for (index=0u; index<count; index++)
	{
		fds[index].fd = descriptors[index].fd;
		fds[index].events = (short)(((descriptors[index].events & SPARK_MODEL_RESIDENT_CLIENT_POLL_READ) != 0u ? POLLIN : 0) |
			((descriptors[index].events & SPARK_MODEL_RESIDENT_CLIENT_POLL_WRITE) != 0u ? POLLOUT : 0));
		fds[index].revents = 0;
	}
	next_ns = SparkModelBatchEngineNextProgressNs(engine);
	now_ns = HostTurnNowNs(CLOCK_MONOTONIC);
	timeout_ms = HOST_TURN_POLL_MS;
	if ( next_ns != 0u && next_ns <= now_ns )
		timeout_ms = 0;
	else if ( next_ns != 0u && next_ns - now_ns < (uint64_t)HOST_TURN_POLL_MS * UINT64_C(1000000) )
		timeout_ms = (int)((next_ns - now_ns + 999999u) / 1000000u);
	(void)poll(fds,count,timeout_ms);
}

static void HostTurnSubmit(SparkModelBatchEngine *engine,uint64_t request_id,const uint32_t *tokens,uint32_t token_count,uint32_t budget)
{
	SparkModelBatchSubmitRequest request;
	SparkModelBatchRequestHandle handle = 0u;
	SparkStatus status;
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request.descriptor_bytes = SPARK_MODEL_BATCH_SUBMIT_REQUEST_BYTES;
	request.priority = SPARK_MODEL_BATCH_PRIORITY_INTERACTIVE;
	request.output_token_budget = budget;
	request.request_id = request_id;
	request.sequence_id = request_id;
	request.prompt_token_ids = tokens;
	request.prompt_token_count = token_count;
	request.top_p = 1.0f;
	status = SparkModelBatchEngineSubmit(engine,&request,&handle);
	if ( status != SPARK_STATUS_OK )
		HostTurnFail("submit failed",(uint32_t)status);
}

static void HostTurnPrompt(uint32_t *tokens,uint32_t count,uint64_t seed)
{
	uint64_t state = seed * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
	uint32_t index;
	for (index=0u; index<count; index++)
	{
		state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
		tokens[index] = 100u + (uint32_t)((state >> 33) % 100000u);
	}
}

static void HostTurnTake(SparkModelBatchEngine *engine,HostTurnSample *sample)
{
	SparkModelBatchEngineView view;
	uint32_t rank;
	SparkStatus status;
	memset(sample,0,sizeof(*sample));
	memset(&view,0,sizeof(view));
	status = SparkModelBatchEngineGetView(engine,&view);
	if ( status != SPARK_STATUS_OK )
		HostTurnFail("engine view failed",(uint32_t)status);
	sample->wall_ns = HostTurnNowNs(CLOCK_MONOTONIC);
	sample->coordinator_ns = HostTurnNowNs(CLOCK_THREAD_CPUTIME_ID);
	for (rank=0u; rank<HOST_TURN_RANKS; rank++)
		sample->rank_ns[rank] = HostTurnProcessCpuNs(HostTurnRanks[rank]);
	sample->frames = view.pipeline.completed_count;
	sample->continued = view.pipeline.continued_count;
	memcpy(sample->frames_by_kind,HostTurn.frames_by_kind,sizeof(sample->frames_by_kind));
	memcpy(sample->kind_cpu_ns,HostTurn.kind_cpu_ns,sizeof(sample->kind_cpu_ns));
	memcpy(sample->kind_wall_ns,HostTurn.kind_wall_ns,sizeof(sample->kind_wall_ns));
	memcpy(sample->kind_service_ns,HostTurn.kind_service_ns,sizeof(sample->kind_service_ns));
	memcpy(sample->kind_done,HostTurn.kind_done,sizeof(sample->kind_done));
	sample->tokens = HostTurn.tokens;
}

static uint64_t HostTurnFrames(SparkModelBatchEngine *engine)
{
	SparkModelBatchEngineView view;
	SparkStatus status;
	memset(&view,0,sizeof(view));
	status = SparkModelBatchEngineGetView(engine,&view);
	if ( status != SPARK_STATUS_OK )
		HostTurnFail("engine view failed",(uint32_t)status);
	return(view.pipeline.completed_count);
}

static void HostTurnReport(const char *scenario,const HostTurnSample *begin,const HostTurnSample *end)
{
	uint64_t frames = end->frames - begin->frames,rank_total = 0u,rank_max = 0u,delta,done;
	uint32_t rank,kind;
	if ( frames == 0u )
		HostTurnFail("no frame completed in the measured window",0u);
	for (rank=0u; rank<HOST_TURN_RANKS; rank++)
	{
		delta = end->rank_ns[rank] - begin->rank_ns[rank];
		rank_total += delta;
		if ( delta > rank_max )
			rank_max = delta;
	}
	printf("%s frame_us=%u frames=%llu prefill=%llu decode=%llu release=%llu publish=%llu continued=%llu tokens=%llu coordinator_us_per_frame=%.2f rank_us_per_frame_mean=%.2f rank_us_per_frame_max=%.2f wall_us_per_frame=%.2f\n",
		scenario,HostTurnFrameUs,(unsigned long long)frames,
		(unsigned long long)(end->frames_by_kind[SPARK_MODEL_SERVING_WORK_KIND_PREFILL] - begin->frames_by_kind[SPARK_MODEL_SERVING_WORK_KIND_PREFILL]),
		(unsigned long long)(end->frames_by_kind[SPARK_MODEL_SERVING_WORK_KIND_DECODE] - begin->frames_by_kind[SPARK_MODEL_SERVING_WORK_KIND_DECODE]),
		(unsigned long long)(end->frames_by_kind[SPARK_MODEL_SERVING_WORK_KIND_RELEASE] - begin->frames_by_kind[SPARK_MODEL_SERVING_WORK_KIND_RELEASE]),
		(unsigned long long)(end->frames_by_kind[SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH] - begin->frames_by_kind[SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH]),
		(unsigned long long)(end->continued - begin->continued),
		(unsigned long long)(end->tokens - begin->tokens),
		(double)(end->coordinator_ns - begin->coordinator_ns) / (double)frames / 1e3,
		(double)rank_total / HOST_TURN_RANKS / (double)frames / 1e3,
		(double)rank_max / (double)frames / 1e3,
		(double)(end->wall_ns - begin->wall_ns) / (double)frames / 1e3);
	for (kind=SPARK_MODEL_SERVING_WORK_KIND_PREFILL; kind<HOST_TURN_KINDS; kind++)
	{
		done = end->kind_done[kind] - begin->kind_done[kind];
		if ( done != 0u )
			printf("  kind=%s frames=%llu coordinator_us_per_frame=%.2f wall_us_per_frame=%.2f bubble_us_per_frame=%.2f\n",
				HostTurnKindNames[kind],(unsigned long long)done,
				(double)(end->kind_cpu_ns[kind] - begin->kind_cpu_ns[kind]) / (double)done / 1e3,
				(double)(end->kind_wall_ns[kind] - begin->kind_wall_ns[kind]) / (double)done / 1e3,
				((double)(end->kind_wall_ns[kind] - begin->kind_wall_ns[kind]) - (double)(end->kind_service_ns[kind] - begin->kind_service_ns[kind])) / (double)done / 1e3);
	}
	fflush(stdout);
}

static void HostTurnCheck(void)
{
	if ( HostTurn.errors != 0u )
		HostTurnFail("a request failed or was cancelled",0u);
}

static void HostTurnDecode(SparkModelBatchEngine *engine,uint32_t lanes,uint32_t prompt_tokens,uint32_t frames)
{
	uint32_t *prompt = (uint32_t *)malloc((size_t)prompt_tokens * sizeof(uint32_t));
	HostTurnSample begin,end;
	char scenario[96];
	uint32_t lane,ready;
	uint64_t warm;
	if ( prompt == 0 )
		HostTurnFail("out of memory",0u);
	for (lane=0u; lane<lanes; lane++)
	{
		HostTurnPrompt(prompt,prompt_tokens,lane + 1u);
		HostTurnSubmit(engine,lane + 1u,prompt,prompt_tokens,HOST_TURN_CONTEXT_TOKENS - prompt_tokens - 8u);
	}
	do
	{
		HostTurnTurn(engine);
		HostTurnCheck();
		ready = 0u;
		for (lane=0u; lane<lanes; lane++)
			ready += HostTurn.generated[lane] >= HOST_TURN_WARM_TOKENS ? 1u : 0u;
	}
	while ( ready != lanes );
	warm = HostTurnFrames(engine) + HOST_TURN_WARM_FRAMES;
	while ( HostTurnFrames(engine) < warm )
	{
		HostTurnTurn(engine);
		HostTurnCheck();
	}
	HostTurnTake(engine,&begin);
	while ( HostTurnFrames(engine) < begin.frames + frames )
	{
		HostTurnTurn(engine);
		HostTurnCheck();
		if ( HostTurn.completed_requests != 0u )
			HostTurnFail("a lane reached its context before the measured window ended; use fewer frames or a shorter prompt",0u);
	}
	HostTurnTake(engine,&end);
	snprintf(scenario,sizeof(scenario),"decode lanes=%u prompt=%u",lanes,prompt_tokens);
	HostTurnReport(scenario,&begin,&end);
	free(prompt);
}

static void HostTurnPrefill(SparkModelBatchEngine *engine,uint32_t requests,uint32_t prompt_tokens,uint32_t rounds)
{
	uint32_t *prompt = (uint32_t *)malloc((size_t)prompt_tokens * sizeof(uint32_t));
	HostTurnSample begin,end;
	uint32_t request,round;
	char scenario[96];
	if ( prompt == 0 )
		HostTurnFail("out of memory",0u);
	HostTurnTake(engine,&begin);
	for (round=0u; round<rounds; round++)
	{
		for (request=0u; request<requests; request++)
		{
			HostTurnPrompt(prompt,prompt_tokens,1000u + request + (uint64_t)round * 100000u);
			HostTurnSubmit(engine,request + 1u,prompt,prompt_tokens,1u);
		}
		while ( HostTurn.completed_requests < (uint64_t)requests * (round + 1u) )
		{
			HostTurnTurn(engine);
			HostTurnCheck();
		}
	}
	HostTurnTake(engine,&end);
	snprintf(scenario,sizeof(scenario),"prefill requests=%u prompt=%u rounds=%u",requests,prompt_tokens,rounds);
	HostTurnReport(scenario,&begin,&end);
	free(prompt);
}

static uint32_t HostTurnArgument(const char *text,uint32_t minimum,uint32_t maximum,const char *name)
{
	char *end = 0;
	unsigned long value;
	errno = 0;
	value = strtoul(text,&end,10);
	if ( errno != 0 || end == text || *end != '\0' || value < minimum || value > maximum )
	{
		fprintf(stderr,"host_turn_bench: %s must be an integer from %u to %u\n",name,minimum,maximum);
		exit(2);
	}
	return((uint32_t)value);
}

static void HostTurnUsage(const char *program)
{
	fprintf(stderr,"usage: %s decode LANES PROMPT_TOKENS FRAMES FRAME_US\n"
		"       %s prefill REQUESTS PROMPT_TOKENS ROUNDS FRAME_US\n",program,program);
	exit(2);
}

int main(int argc,char **argv)
{
	SparkModelResidentDeployment deployment;
	SparkModelBatchEngine *engine;
	char deployment_path[HOST_TURN_PATH_BYTES];
	uint32_t decode,count,prompt_tokens,repeat;
	SparkStatus status;
	if ( argc != 6 )
		HostTurnUsage(argv[0]);
	decode = strcmp(argv[1],"decode") == 0 ? 1u : 0u;
	if ( decode == 0u && strcmp(argv[1],"prefill") != 0 )
		HostTurnUsage(argv[0]);
	count = HostTurnArgument(argv[2],1u,decode != 0u ? HOST_TURN_MAX_LANES : HOST_TURN_MAX_REQUESTS,decode != 0u ? "LANES" : "REQUESTS");
	prompt_tokens = HostTurnArgument(argv[3],1u,HOST_TURN_CONTEXT_TOKENS / 2u,"PROMPT_TOKENS");
	repeat = HostTurnArgument(argv[4],1u,UINT32_MAX / 2u,decode != 0u ? "FRAMES" : "ROUNDS");
	HostTurnFrameUs = HostTurnArgument(argv[5],0u,HOST_TURN_MAX_FRAME_US,"FRAME_US");
	HostTurnMakeWork();
	HostTurnWriteDeployment(deployment_path);
	SparkModelResidentDeploymentReset(&deployment);
	status = SparkModelResidentDeploymentLoad(deployment_path,&deployment);
	if ( status != SPARK_STATUS_OK )
		HostTurnFail("cannot load the deployment",(uint32_t)status);
#if defined(__APPLE__)
	(void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
#endif
	alarm(HOST_TURN_DEADLINE_SECONDS);
	HostTurnStartRanks(deployment_path);
	engine = HostTurnConnect(&deployment);
	if ( decode != 0u )
		HostTurnDecode(engine,count,prompt_tokens,repeat);
	else
		HostTurnPrefill(engine,count,prompt_tokens,repeat);
	HostTurnStopRanks();
	HostTurnRemoveWork();
	return(0);
}
