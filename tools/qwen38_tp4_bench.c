/* qwen38 TP4 throughput benchmark: fixed decode microbatch sizes B1/B2/B4/B8.
 * B lanes share one prompt, prefill separately, then every decode step is
 * ONE submission with B rows (one per lane) through the full prepare/commit
 * path on all four ranks. Reports per-step latency and tokens/sec. */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_model_resident_client.h"
#include "sparkpipe/spark_model_resident_deployment.h"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_status.h"
#include "spark_filesystem.h"

#define PROMPT_TOKENS 5u
#define MAX_BATCH 16u

typedef struct RankCollect
{
	uint32_t have_completion;
	uint32_t token_count;
	int32_t submit_result_status;
} RankCollect;

static void RankSubmitResult(void *context, uint64_t submission_id, SparkStatus status)
{
	RankCollect *collect = (RankCollect *)context;
	(void)submission_id;
	collect->submit_result_status = (int32_t)status;
}

static void RankCompletion(void *context, const SparkModelServingCompletion *completion)
{
	RankCollect *collect = (RankCollect *)context;
	if ( completion->status == SPARK_STATUS_OK )
	{
		collect->token_count = completion->token_count;
		collect->have_completion = 1u;
	}
}

static uint64_t NowNanos(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static int PumpUntilAll(SparkModelResidentClient **clients, RankCollect *collects, uint32_t rank_count, uint64_t timeout_ns)
{
	uint64_t deadline = NowNanos() + timeout_ns;
	for (;;)
	{
		uint32_t rank;
		uint32_t all = 1u;
		for (rank = 0u; rank < rank_count; rank++)
		{
			SparkStatus s = SparkModelResidentClientProgress(clients[rank], 8u);
			if ( s != SPARK_STATUS_OK )
			{
				fprintf(stderr, "progress rank=%u status=%d\n", rank, (int)s);
				return(1);
			}
			if ( collects[rank].have_completion == 0u )
				all = 0u;
		}
		if ( all != 0u )
			return(0);
		if ( NowNanos() > deadline )
		{
			for (rank = 0u; rank < rank_count; rank++)
				fprintf(stderr, "timeout rank=%u have=%u\n", rank, collects[rank].have_completion);
			return(1);
		}
		usleep(1000);
	}
}

static int DriveStep(SparkModelResidentClient **clients, RankCollect *collects, uint32_t rank_count, const SparkModelServingSubmission *submission, uint32_t expected_tokens)
{
	uint32_t rank;
	SparkStatus status;
	for (rank = 0u; rank < rank_count; rank++)
	{
		collects[rank].have_completion = 0u;
		collects[rank].submit_result_status = -1;
	}
	for (rank = 0u; rank < rank_count; rank++)
	{
		status = SparkModelResidentClientPrepare(clients[rank], submission);
		if ( status != SPARK_STATUS_OK )
		{
			SparkModelResidentClientView view;
			(void)SparkModelResidentClientGetView(clients[rank], &view);
			fprintf(stderr, "prepare rank=%u status=%d queued=%u pending=%u prepared=%u submitted=%llu sub=%llu\n", rank, (int)status, view.queued_message_count, view.pending_submission_count, view.prepared_submission_count, (unsigned long long)view.submitted_count, (unsigned long long)submission->submission_id);
			return(1);
		}
	}
	{
		uint64_t deadline = NowNanos() + 10000000000ull;
		for (;;)
		{
			uint32_t rank;
			uint32_t drained = 1u;
			for (rank = 0u; rank < rank_count; rank++)
			{
				status = SparkModelResidentClientProgress(clients[rank], 8u);
				if ( status != SPARK_STATUS_OK )
				{
					fprintf(stderr, "prepare drain rank=%u status=%d\n", rank, (int)status);
					return(1);
				}
				if ( collects[rank].submit_result_status == -1 )
					drained = 0u;
			}
			if ( drained != 0u )
				break;
			if ( NowNanos() > deadline )
			{
				fprintf(stderr, "prepare drain timeout\n");
				return(1);
			}
			usleep(1000);
		}
	}
	for (rank = 0u; rank < rank_count; rank++)
	{
		status = SparkModelResidentClientCommit(clients[rank], submission->submission_id);
		if ( status != SPARK_STATUS_OK )
		{
			uint32_t r;
			fprintf(stderr, "commit rank=%u status=%d\n", rank, (int)status);
			for (r = 0u; r < rank_count; r++)
				fprintf(stderr, "rank=%u submit_result_status=%d\n", r, collects[r].submit_result_status);
			return(1);
		}
	}
	if ( PumpUntilAll(clients, collects, rank_count, 60000000000ull) != 0 )
		return(1);
	for (rank = 0u; rank < rank_count; rank++)
		if ( collects[rank].token_count != expected_tokens )
		{
			fprintf(stderr, "token count mismatch rank=%u got=%u expected=%u\n", rank, collects[rank].token_count, expected_tokens);
			return(1);
		}
	return(0);
}

int main(int argc, char **argv)
{
	SparkModelResidentDeployment deployment;
	SparkModelServingAdapterDynamicLibrary adapter_library;
	SparkModelResidentClient *clients[4];
	RankCollect collects[4];
	SparkModelResidentClientConfiguration client_configuration;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[MAX_BATCH];
	uint32_t prompt[PROMPT_TOKENS] = { 760u, 6511u, 314u, 9338u, 369u };
	uint32_t prefill_lane_indices[PROMPT_TOKENS] = { 0u, 0u, 0u, 0u, 0u };
	uint64_t prefill_positions[PROMPT_TOKENS] = { 0u, 1u, 2u, 3u, 4u };
	uint32_t decode_token_ids[MAX_BATCH];
	uint32_t decode_lane_indices[MAX_BATCH];
	uint64_t decode_positions[MAX_BATCH];
	uint64_t decode_sequence_ids[MAX_BATCH];
	uint64_t prefill_sequence_ids[PROMPT_TOKENS];
	char adapter_path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	SparkStatus status;
	uint32_t rank, lane, step;
	uint32_t batch_size, step_count = 100u;
	uint64_t base_sequence = 71000ull;
	uint64_t submission_id = 1ull;
	uint64_t prefill_start, step_start, step_elapsed;
	uint64_t decode_total_ns = 0ull, step_max_ns = 0ull, step_min_ns = UINT64_MAX;
	if ( argc < 4 || argc > 6 )
	{
		fprintf(stderr, "usage: %s DEPLOYMENT_JSON RUNTIME_ROOT BATCH_SIZE [STEPS] [BASE_SEQUENCE]\n", argv[0]);
		return(2);
	}
	batch_size = (uint32_t)strtoul(argv[3], 0, 0);
	if ( argc >= 5 )
		step_count = (uint32_t)strtoul(argv[4], 0, 0);
	if ( argc == 6 )
		base_sequence = strtoull(argv[5], 0, 0);
	if ( batch_size == 0u || batch_size > MAX_BATCH || step_count == 0u )
		return(2);
	SparkModelResidentDeploymentReset(&deployment);
	status = SparkModelResidentDeploymentLoad(argv[1], &deployment);
	if ( status != SPARK_STATUS_OK ) { fprintf(stderr, "deployment load status=%d\n", (int)status); return(1); }
	status = SparkResolveRuntimePath(argv[2], deployment.adapter_shared_object_path, adapter_path, sizeof(adapter_path));
	if ( status == SPARK_STATUS_OK )
		status = SparkModelServingAdapterLoadInterfaceFromSharedObject(adapter_path, SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE, &adapter_library);
	if ( status != SPARK_STATUS_OK ) { fprintf(stderr, "adapter load status=%d\n", (int)status); return(1); }
	memset(clients, 0, sizeof(clients));
	memset(collects, 0, sizeof(collects));
	for (rank = 0u; rank < deployment.node_count; rank++)
	{
		memset(&client_configuration, 0, sizeof(client_configuration));
		client_configuration.abi_version = SPARK_MODEL_RESIDENT_CLIENT_ABI_VERSION;
		client_configuration.descriptor_bytes = SPARK_MODEL_RESIDENT_CLIENT_CONFIGURATION_BYTES;
		client_configuration.rank_index = deployment.nodes[rank].rank_index;
		client_configuration.stage_index = deployment.nodes[rank].stage_index;
		client_configuration.connect_timeout_ms = 30000u;
		client_configuration.runtime_limits = deployment.runtime_limits;
		client_configuration.endpoint = deployment.nodes[rank].control_endpoint;
		client_configuration.adapter_descriptor = adapter_library.adapter_interface.descriptor;
		client_configuration.submit_result_function = RankSubmitResult;
		client_configuration.submit_result_context = &collects[rank];
		client_configuration.completion_function = RankCompletion;
		client_configuration.completion_context = &collects[rank];
		status = SparkModelResidentClientConnect(&client_configuration, &clients[rank]);
		if ( status != SPARK_STATUS_OK ) { fprintf(stderr, "connect rank=%u status=%d\n", rank, (int)status); return(1); }
	}
	memset(lanes, 0, sizeof(lanes));
	for (lane = 0u; lane < batch_size; lane++)
	{
		lanes[lane].request_id = base_sequence + lane;
		lanes[lane].request_generation = 1ull;
		lanes[lane].step_generation = 1ull;
		lanes[lane].sequence_id = base_sequence + lane;
		lanes[lane].resident_sequence_slot = lane;
		lanes[lane].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
		decode_lane_indices[lane] = lane;
		decode_sequence_ids[lane] = base_sequence + lane;
		decode_token_ids[lane] = 0u;
	}
	memset(&submission, 0, sizeof(submission));
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.control_generation = 1ull;
	submission.transaction_id = 1ull;
	submission.dispatch_generation = 1ull;
	submission.request_generation = 1ull;
	submission.step_generation = 1ull;
	submission.priority = 0u;
	submission.lane_count = 1u;
	submission.active_sequence_count = 1u;
	submission.tokens_per_sequence = 1u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.new_token_count = PROMPT_TOKENS;
	submission.row_count = PROMPT_TOKENS;
	submission.token_count = PROMPT_TOKENS;
	submission.token_ids = prompt;
	submission.row_lane_indices = prefill_lane_indices;
	submission.row_positions = prefill_positions;
	submission.row_sequence_ids = prefill_sequence_ids;
	prefill_start = NowNanos();
	for (lane = 0u; lane < batch_size; lane++)
	{
		uint32_t i;
		fprintf(stderr, "prefill lane=%u\n", lane);
		submission.lanes = &lanes[lane];
		submission.submission_id = submission_id++;
		submission.request_id = base_sequence + lane;
		submission.sequence_id = base_sequence + lane;
		submission.sequence_position = 0ull;
		submission.deadline_time_ns = NowNanos() + 60000000000ull;
		submission.residency.word0 = submission.submission_id;
		submission.residency.word1 = submission.submission_id ^ UINT64_C(0x535041524b504950);
		submission.residency.generation = submission.submission_id;
		submission.residency.owner = 1u;
		lanes[lane].sequence_position = 0ull;
		lanes[lane].context_token_count = PROMPT_TOKENS;
		lanes[lane].input_token_id = 0u;
		for (i = 0u; i < PROMPT_TOKENS; i++)
			prefill_sequence_ids[i] = base_sequence + lane;
		if ( DriveStep(clients, collects, deployment.node_count, &submission, 1u) != 0 )
			return(1);
		decode_token_ids[lane] = 0u;
	}
	printf("batch=%u prefill_ns=%llu (%llu ms)\n", batch_size, (unsigned long long)(NowNanos() - prefill_start), (unsigned long long)((NowNanos() - prefill_start) / 1000000ull));
	fflush(stdout);
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission.lanes = lanes;
	submission.lane_count = batch_size;
	submission.active_sequence_count = batch_size;
	submission.new_token_count = batch_size;
	submission.row_count = batch_size;
	submission.token_count = batch_size;
	submission.token_ids = decode_token_ids;
	submission.row_lane_indices = decode_lane_indices;
	submission.row_positions = decode_positions;
	submission.row_sequence_ids = decode_sequence_ids;
	submission.request_id = base_sequence;
	submission.sequence_id = base_sequence;
	for (step = 0u; step < step_count; step++)
	{
		uint64_t position = (uint64_t)(PROMPT_TOKENS + step);
		for (lane = 0u; lane < batch_size; lane++)
		{
			decode_positions[lane] = position;
			lanes[lane].sequence_position = position;
			lanes[lane].context_token_count = PROMPT_TOKENS + step + 1u;
			lanes[lane].input_token_id = decode_token_ids[lane];
		}
		submission.submission_id = submission_id++;
		submission.sequence_position = position;
		submission.deadline_time_ns = NowNanos() + 60000000000ull;
		submission.residency.word0 = submission.submission_id;
		submission.residency.word1 = submission.submission_id ^ UINT64_C(0x535041524b504950);
		submission.residency.generation = submission.submission_id;
		submission.residency.owner = 1u;
		step_start = NowNanos();
		if ( DriveStep(clients, collects, deployment.node_count, &submission, batch_size) != 0 )
			return(1);
		step_elapsed = NowNanos() - step_start;
		decode_total_ns += step_elapsed;
		if ( step_elapsed > step_max_ns )
			step_max_ns = step_elapsed;
		if ( step_elapsed < step_min_ns )
			step_min_ns = step_elapsed;
	}
	{
		double decode_seconds = (double)decode_total_ns / 1e9;
		double tokens = (double)batch_size * (double)step_count;
		printf("batch=%u steps=%u decode_total_ms=%llu step_avg_ms=%.2f step_min_ms=%.2f step_max_ms=%.2f tokens_per_sec=%.1f steps_per_sec=%.2f\n",
			batch_size, step_count, (unsigned long long)(decode_total_ns / 1000000ull),
			((double)decode_total_ns / (double)step_count) / 1e6,
			(double)step_min_ns / 1e6, (double)step_max_ns / 1e6,
			tokens / decode_seconds, (double)step_count / decode_seconds);
	}
	for (rank = 0u; rank < deployment.node_count; rank++)
		SparkModelResidentClientDestroy(clients[rank]);
	SparkModelServingAdapterUnloadInterface(&adapter_library);
	SparkModelResidentDeploymentDestroy(&deployment);
	return(0);
}
