/* qwen38 TP4 E2E driver: drives all four fanout residentds with identical
 * prepare+commit submissions, verifies per-rank token agreement and
 * compares the emitted sequence against the PP16 post-fix reference. */
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
#define DECODE_STEPS 63u

static const uint32_t reference_tokens[64] = {
11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760, 6511, 314, 14898, 369, 21047, 13, 198, 760, 6511, 314, 17163, 369, 23327, 13, 198, 760, 6511, 314, 32208, 369, 77916, 13, 198, 760, 6511, 314, 23655, 369, 44299, 13, 198, 760, 6511, 314, 279, 3516, 14634, 369, 6924, 13, 198, 760, 6511, 314, 14227, 369, 31785, 13, 198, 760, 6511, 314
};

typedef struct RankCollect
{
	uint32_t have_token;
	uint32_t token;
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
	if ( completion->status != SPARK_STATUS_OK || completion->token_count != 1u ||
		(completion->completion_flags & SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS) == 0u )
	{
		fprintf(stderr, "bad completion status=%u tokens=%u flags=0x%x\n", completion->status, completion->token_count, completion->completion_flags);
		return;
	}
	collect->token = completion->token_ids[0];
	collect->have_token = 1u;
}

static uint64_t NowNanos(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static int PumpUntilAll(SparkModelResidentClient **clients, RankCollect *collects, uint32_t rank_count)
{
	uint64_t deadline = NowNanos() + 180000000000ull;
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
			if ( collects[rank].have_token == 0u )
				all = 0u;
		}
		if ( all != 0u )
			return(0);
		if ( NowNanos() > deadline )
		{
			for (rank = 0u; rank < rank_count; rank++)
				fprintf(stderr, "timeout rank=%u have_token=%u\n", rank, collects[rank].have_token);
			return(1);
		}
		usleep(2000);
	}
}

static int DriveStep(SparkModelResidentClient **clients, RankCollect *collects, uint32_t rank_count, const SparkModelServingSubmission *submission)
{
	uint32_t rank;
	SparkStatus status;
	for (rank = 0u; rank < rank_count; rank++)
	{
		collects[rank].have_token = 0u;
		collects[rank].submit_result_status = -1;
	}
	for (rank = 0u; rank < rank_count; rank++)
	{
		status = SparkModelResidentClientPrepare(clients[rank], submission);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr, "prepare rank=%u status=%d\n", rank, (int)status);
			return(1);
		}
	}
	/* Drain the prepare submit-results so the pendings read as prepared. */
	{
		uint64_t deadline = NowNanos() + 5000000000ull;
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
				uint32_t r;
				fprintf(stderr, "prepare drain timeout\n");
				for (r = 0u; r < rank_count; r++)
					fprintf(stderr, "rank=%u submit_result_status=%d\n", r, collects[r].submit_result_status);
				return(1);
			}
			usleep(2000);
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
	return(PumpUntilAll(clients, collects, rank_count));
}

int main(int argc, char **argv)
{
	SparkModelResidentDeployment deployment;
	SparkModelServingAdapterDynamicLibrary adapter_library;
	SparkModelResidentClient *clients[4];
	RankCollect collects[4];
	SparkModelResidentClientConfiguration client_configuration;
	SparkModelServingSubmission submission;
	SparkModelServingLane lane;
	uint32_t prompt[PROMPT_TOKENS] = { 760u, 6511u, 314u, 9338u, 369u };
	uint32_t row_lane_indices[PROMPT_TOKENS] = { 0u, 0u, 0u, 0u, 0u };
	uint64_t row_positions[PROMPT_TOKENS] = { 0u, 1u, 2u, 3u, 4u };
	uint64_t row_sequence_ids[PROMPT_TOKENS] = { 71001ull, 71001ull, 71001ull, 71001ull, 71001ull };
	uint32_t decode_token_ids[1];
	uint32_t decode_lane_indices[1] = { 0u };
	uint64_t decode_positions[1];
	uint64_t decode_sequence_ids[1] = { 71001ull };
	uint32_t emitted[64];
	char adapter_path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	SparkStatus status;
	uint32_t rank, step;
	uint32_t previous_token;
	uint64_t submission_id = 1ull;
	uint32_t mismatches = 0u;
	if ( argc != 3 )
	{
		fprintf(stderr, "usage: %s DEPLOYMENT_JSON RUNTIME_ROOT\n", argv[0]);
		return(2);
	}
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
	printf("connected %u ranks\n", deployment.node_count);
	memset(&lane, 0, sizeof(lane));
	lane.request_id = 71001ull;
	lane.request_generation = 1ull;
	lane.step_generation = 1ull;
	lane.sequence_id = 71001ull;
	lane.resident_sequence_slot = 0u;
	lane.flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	memset(&submission, 0, sizeof(submission));
	submission.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission.descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission.request_id = 71001ull;
	submission.sequence_id = 71001ull;
	submission.control_generation = 1ull;
	submission.transaction_id = 1ull;
	submission.dispatch_generation = 1ull;
	submission.request_generation = 1ull;
	submission.step_generation = 1ull;
	submission.priority = 0u;
	submission.active_sequence_count = 1u;
	submission.lane_count = 1u;
	submission.tokens_per_sequence = 1u;
	submission.lanes = &lane;
	/* Prefill: 5 prompt rows, emits the first token. */
	submission.submission_id = submission_id++;
	submission.residency.word0 = submission.submission_id;
	submission.residency.word1 = submission.submission_id ^ UINT64_C(0x535041524b504950);
	submission.residency.generation = submission.submission_id;
	submission.residency.owner = 1u;
	submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_PREFILL;
	submission.sequence_position = 0ull;
	submission.new_token_count = PROMPT_TOKENS;
	submission.row_count = PROMPT_TOKENS;
	submission.token_count = PROMPT_TOKENS;
	submission.token_ids = prompt;
	submission.row_lane_indices = row_lane_indices;
	submission.row_positions = row_positions;
	submission.row_sequence_ids = row_sequence_ids;
	submission.deadline_time_ns = NowNanos() + 10000000000ull;
	lane.sequence_position = 0ull;
	lane.context_token_count = PROMPT_TOKENS;
	lane.input_token_id = 0u;
	if ( DriveStep(clients, collects, deployment.node_count, &submission) != 0 ) return(1);
	emitted[0] = collects[0].token;
	for (rank = 1u; rank < deployment.node_count; rank++)
		if ( collects[rank].token != emitted[0] )
		{
			fprintf(stderr, "prefill token mismatch rank=%u token=%u expected=%u\n", rank, collects[rank].token, emitted[0]);
			mismatches++;
		}
	previous_token = emitted[0];
	printf("token[%2u] = %u\n", 0u, emitted[0]);
	fflush(stdout);
	for (step = 0u; step < DECODE_STEPS; step++)
	{
		uint64_t position = (uint64_t)(PROMPT_TOKENS + step);
		decode_token_ids[0] = previous_token;
		decode_positions[0] = position;
		submission.submission_id = submission_id++;
		submission.residency.word0 = submission.submission_id;
		submission.residency.word1 = submission.submission_id ^ UINT64_C(0x535041524b504950);
		submission.residency.generation = submission.submission_id;
		submission.residency.owner = 1u;
		submission.work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
		submission.sequence_position = position;
		submission.new_token_count = 1u;
		submission.row_count = 1u;
		submission.token_count = 1u;
		submission.token_ids = decode_token_ids;
		submission.row_lane_indices = decode_lane_indices;
		submission.row_positions = decode_positions;
		submission.row_sequence_ids = decode_sequence_ids;
		submission.deadline_time_ns = NowNanos() + 10000000000ull;
		lane.sequence_position = position;
		lane.context_token_count = PROMPT_TOKENS + step + 1u;
		lane.input_token_id = previous_token;
		if ( DriveStep(clients, collects, deployment.node_count, &submission) != 0 ) return(1);
		emitted[1u + step] = collects[0].token;
		for (rank = 1u; rank < deployment.node_count; rank++)
			if ( collects[rank].token != emitted[1u + step] )
			{
				fprintf(stderr, "step %u token mismatch rank=%u token=%u expected=%u\n", step, rank, collects[rank].token, emitted[1u + step]);
				mismatches++;
			}
		previous_token = emitted[1u + step];
		printf("token[%2u] = %u\n", 1u + step, previous_token);
		fflush(stdout);
	}
	printf("--- verification ---\n");
	for (step = 0u; step < 64u; step++)
		if ( emitted[step] != reference_tokens[step] )
		{
			fprintf(stderr, "reference mismatch index=%u emitted=%u reference=%u\n", step, emitted[step], reference_tokens[step]);
			mismatches++;
		}
	printf("rank_reference_mismatches=%u\n", mismatches);
	printf(mismatches == 0u ? "Qwen38-TP4-E2E-PASS\n" : "Qwen38-TP4-E2E-FAIL\n");
	for (rank = 0u; rank < deployment.node_count; rank++)
		SparkModelResidentClientDestroy(clients[rank]);
	SparkModelServingAdapterUnloadInterface(&adapter_library);
	SparkModelResidentDeploymentDestroy(&deployment);
	return(mismatches == 0u ? 0 : 1);
}
