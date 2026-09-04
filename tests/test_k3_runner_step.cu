
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "inference/llms/kimi_k3/layer.cuh"

static void NoopHook(void *context, void *stream, uint32_t layer,
	uint32_t phase)
{
	(void)context;
	(void)phase;
	(void)stream;
	(void)layer;
}

static uint32_t FiniteCount(const uint16_t *values, uint32_t count)
{
	uint32_t finite = 0u;
	for ( uint32_t i = 0u; i < count; ++i )
	{
		uint32_t u = ((uint32_t)values[i]) << 16u;
		float f;
		memcpy(&f, &u, sizeof(f));
		if ( std::isfinite((double)f) )
			finite++;
	}
	return finite;
}

static float FirstHalfSum(const uint16_t *values, uint32_t count)
{
	double sum = 0.0;
	for ( uint32_t i = 0u; i < count / 2u; ++i )
	{
		uint32_t u = ((uint32_t)values[i]) << 16u;
		float f;
		memcpy(&f, &u, sizeof(f));
		sum += f;
	}
	return (float)sum;
}

int main(int argc, char **argv)
{
	if ( argc < 2 )
	{
		printf("usage: k3_single_step <rank.pack> [--dump path]\n");
		return 2;
	}
	int dump_mode = argc >= 4 && strcmp(argv[2], "--dump") == 0;
	FILE *dump_file = 0;
	if ( dump_mode )
		dump_file = fopen(argv[3], "wb");
	SparkK3StageRunner runner;
	SparkK3StageRunnerConfiguration config;
	SparkK3StageRunnerDispatch dispatch;
	SparkStatus status;
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	config.descriptor_bytes = (uint32_t)sizeof(config);
	config.stage_index = 0u;
	int pp1 = 0;
	for ( int i = 2; i < argc; ++i )
		if ( strcmp(argv[i], "--pp1") == 0 )
			pp1 = 1;
	config.stage_count = pp1 ? 1u : 4u;
	config.tp_degree = 1u;
	config.tp_rank = 0u;
	config.max_active_sequence_count = 1u;
	config.max_input_row_count = 1u;
	config.resident_sequence_capacity = 1u;
	config.kv_pages_per_sequence = 2u;
	config.kv_page_bytes = K3GlobalKv::kPageBytes;
	config.rank_pack_path = argv[1];
	SparkTpCollectiveConfig collective;
	SparkTpCollectivePeer peers[4];
	int tp4 = 0;
	for ( int i = 2; i < argc; ++i )
		if ( strcmp(argv[i], "--tp4") == 0 && i + 1 < argc )
		{
			tp4 = 1;
			config.tp_degree = 4u;
			config.tp_rank = (uint32_t)atoi(argv[i + 1]);
		}
	if ( tp4 )
	{
		memset(&collective, 0, sizeof(collective));
		memset(peers, 0, sizeof(peers));
		collective.abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
		collective.tp_degree = 4u;
		collective.tp_rank = config.tp_rank;
		collective.listen_port = (uint16_t)(61620u + config.tp_rank);
		collective.connect_timeout_milli = 8000u;
		collective.operation_timeout_milli = 30000u;
		collective.collective_identifier = 1u;
		for ( int step = 0; step < 2; ++step )
		{
			uint32_t partner = config.tp_rank ^ (1u << (uint32_t)step);
			snprintf(peers[step].host_name, sizeof(peers[step].host_name),
				"127.0.0.1");
			peers[step].port = (uint16_t)(61620u + partner);
		}
		memcpy(collective.peers, peers, sizeof(peers));
		config.tp_collective = &collective;
		config.layer_collective_override = 0;
	}
	cudaStream_t runner_stream = 0;
	cudaStreamCreate(&runner_stream);
	config.execution_stream = runner_stream;
	config.flags |= SPARK_K3_STAGE_RUNNER_FLAG_CAPTURE_GRAPHS;
	config.layer_collective_override = (tp4 || pp1) ? 0 : NoopHook;
	int multiprocessors = 0;
	cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0);
	config.multiprocessors = (uint32_t)multiprocessors;

	status = SparkK3StageRunnerInitialize(&runner, &config);
	if ( status != SPARK_STATUS_OK )
	{
		printf("INIT FAIL %d\n", (int)status);
		return 1;
	}
	printf("runner initialized (stage 0, tp 1, mps %d)\n", multiprocessors);

	uint32_t *d_tokens, *d_positions, *d_context, *d_seq_of_row, *d_state_index;
	uint32_t *d_output_tokens;
	float *d_output_scores;
	uint16_t *d_hidden;
	uint32_t h_token = 1u, h_position = 0u, h_context = 1u, h_seq = 0u, h_state = 0u;
	cudaMalloc(&d_tokens, 4u);
	cudaMalloc(&d_positions, 4u);
	cudaMalloc(&d_context, 4u);
	cudaMalloc(&d_seq_of_row, 4u);
	cudaMalloc(&d_state_index, 4u);
	cudaMalloc(&d_output_tokens, 4u);
	cudaMalloc(&d_output_scores, 4u);
	cudaMalloc(&d_hidden, (uint64_t)K3_HIDDEN * 2u);
	cudaMemcpy(d_tokens, &h_token, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_positions, &h_position, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_context, &h_context, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_seq_of_row, &h_seq, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_state_index, &h_state, 4u, cudaMemcpyHostToDevice);

	uint16_t h_hidden[K3_HIDDEN];
	uint16_t h_hidden_second[K3_HIDDEN];
	uint16_t h_hidden_graph[K3_HIDDEN];
	memset(&dispatch, 0, sizeof(dispatch));
	dispatch.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	dispatch.descriptor_bytes = (uint32_t)sizeof(dispatch);
	dispatch.request_id = 1u;
	dispatch.sequence_id = 1u;
	dispatch.sequence_position = 0u;
	dispatch.row_count = 1u;
	dispatch.active_sequence_count = 1u;
	dispatch.token_ids = d_tokens;
	dispatch.positions = d_positions;
	dispatch.context_length = d_context;
	dispatch.sequence_of_row = d_seq_of_row;
	dispatch.kda_state_index = d_state_index;
	dispatch.hidden_output_bf16 = d_hidden;
	dispatch.hidden_output_bytes = (uint64_t)K3_HIDDEN * 2u;
	dispatch.output_token_ids = d_output_tokens;
	dispatch.output_scores = d_output_scores;

	cudaEvent_t begin, end, e_embed, e_slice;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	cudaEventCreate(&e_embed);
	cudaEventCreate(&e_slice);
	cudaEventRecord(begin, runner_stream);
	cudaEventRecord(e_embed, runner_stream);
	cudaEventRecord(e_slice, runner_stream);
	status = SparkK3StageRunnerSubmit(&runner, &dispatch);
	cudaEventRecord(end, runner_stream);
	cudaEventSynchronize(end);
	float millis = 0.0f, embed_millis = 0.0f, slice_millis = 0.0f;
	cudaEventElapsedTime(&millis, begin, end);
	cudaEventElapsedTime(&embed_millis, e_embed, e_slice);
	cudaEventElapsedTime(&slice_millis, e_slice, end);
	printf("submit total %.3f ms (embed-queue %.4f ms, slice %.3f ms)\n",
		(double)millis, (double)embed_millis, (double)slice_millis);
	if ( status != SPARK_STATUS_OK )
	{
		printf("SUBMIT FAIL %d\n", (int)status);
		return 1;
	}
	cudaMemcpy(h_hidden, d_hidden, (uint64_t)K3_HIDDEN * 2u, cudaMemcpyDeviceToHost);
	uint32_t finite = FiniteCount(h_hidden, K3_HIDDEN);
	float sum = FirstHalfSum(h_hidden, K3_HIDDEN);
	printf("step 1: %u/%u finite, first-half sum %.6g, %.3f ms\n",
		finite, (uint32_t)K3_HIDDEN, (double)sum, (double)millis);
	if ( dump_mode )
	{
		fwrite(h_hidden, 2u, K3_HIDDEN, dump_file);
		fclose(dump_file);
		return 0;
	}

	cudaEventRecord(begin, runner_stream);
	status = SparkK3StageRunnerSubmit(&runner, &dispatch);
	cudaEventRecord(end, runner_stream);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&millis, begin, end);
	printf("step 2: %.3f ms (graph capture+replay)\n", (double)millis);
	cudaMemcpy(h_hidden_graph, d_hidden, (uint64_t)K3_HIDDEN * 2u,
		cudaMemcpyDeviceToHost);
	cudaEventRecord(begin, runner_stream);
	status = SparkK3StageRunnerSubmit(&runner, &dispatch);
	cudaEventRecord(end, runner_stream);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&millis, begin, end);
	printf("step 3: %.3f ms (graph replay)\n", (double)millis);
	SparkK3StageRunnerDestroy(&runner);
	status = SparkK3StageRunnerInitialize(&runner, &config);
	if ( status != SPARK_STATUS_OK )
	{
		printf("REINIT FAIL %d\n", (int)status);
		return 1;
	}
	status = SparkK3StageRunnerSubmit(&runner, &dispatch);
	if ( status != SPARK_STATUS_OK )
	{
		printf("FRESH SUBMIT FAIL %d\n", (int)status);
		return 1;
	}
	cudaMemcpy(h_hidden_second, d_hidden, (uint64_t)K3_HIDDEN * 2u, cudaMemcpyDeviceToHost);
	uint32_t mismatches = 0u;
	for ( uint32_t i = 0u; i < K3_HIDDEN; ++i )
	{
		uint32_t a = h_hidden[i], b = h_hidden_second[i];
		uint32_t diff = a > b ? a - b : b - a;
		uint32_t limit = 4u;
		if ( diff > limit )
		{
			if ( mismatches < 16u )
				printf("  mismatch[%u] run1=%04x run2=%04x\n", i, a, b);
			mismatches++;
		}
	}
	printf("fresh-run vs step 1: %u mismatches beyond 4 ULP\n", mismatches);

	SparkK3StageRunner direct_runner;
	SparkK3StageRunnerConfiguration direct_config = config;
	direct_config.flags &= ~SPARK_K3_STAGE_RUNNER_FLAG_CAPTURE_GRAPHS;
	status = SparkK3StageRunnerInitialize(&direct_runner, &direct_config);
	if ( status != SPARK_STATUS_OK )
	{
		printf("DIRECT INIT FAIL %d\n", (int)status);
		return 1;
	}
	status = SparkK3StageRunnerSubmit(&direct_runner, &dispatch);
	status = SparkK3StageRunnerSubmit(&direct_runner, &dispatch);
	if ( status != SPARK_STATUS_OK )
	{
		printf("DIRECT STEP2 FAIL %d\n", (int)status);
		return 1;
	}
	cudaMemcpy(h_hidden_second, d_hidden, (uint64_t)K3_HIDDEN * 2u,
		cudaMemcpyDeviceToHost);
	uint32_t graph_mismatches = 0u;
	for ( uint32_t i = 0u; i < K3_HIDDEN; ++i )
	{
		uint32_t a = h_hidden_graph[i], b = h_hidden_second[i];
		uint32_t diff = a > b ? a - b : b - a;
		uint32_t limit = 4u;
		if ( diff > limit )
		{
			if ( graph_mismatches < 8u )
				printf("  graph-vs-direct[%u] graph=%04x direct=%04x\n", i, a, b);
			graph_mismatches++;
		}
	}
	printf("graph step 2 vs direct step 2: %u mismatches beyond ULP limit\n",
		graph_mismatches);
	if ( graph_mismatches != 0u )
	{
		printf("GRAPH FIDELITY MISMATCH %u\n", graph_mismatches);
		return 1;
	}
	SparkK3StageRunnerDestroy(&direct_runner);

	if ( finite == K3_HIDDEN && sum != 0.0f && mismatches == 0u )
	{
		printf("k3 single-spark step gate PASS\n");
		return 0;
	}
	printf("k3 single-spark step gate FAIL\n");
	return 1;
}
