// Per-layer divergence probe over the EXACT serving path: the runner with a
// diagnostic hook override dumps the hidden stream after every layer, so two
// runs can be diffed to find the layer whose output first diverges.

#include <cstdio>
#include <cstring>

#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "inference/llms/kimi_k3/layer.cuh"

static uint16_t *g_snapshots;
static uint32_t g_rows = 0u;
static K3LayerBuffers *g_probe_buffers;

static void ProbeHook(void *context, void *stream_void, uint32_t layer)
{
	(void)context;
	/* Pinned + async: the synchronous copy masked the race; the async
	 * copy queues behind the layer's kernels and preserves the bare-run
	 * timing while still capturing the per-layer stream. */
	cudaMemcpyAsync(&g_snapshots[layer * K3_HIDDEN], g_probe_buffers->hidden_bf16,
		(uint64_t)g_rows * K3_HIDDEN * 2u, cudaMemcpyDeviceToHost,
		(cudaStream_t)stream_void);
}

int main(int argc, char **argv)
{
	if ( argc < 3 )
	{
		printf("usage: k3_layer_probe <rank.pack> <dump.bin>\n");
		return 2;
	}
	SparkK3StageRunner runner;
	SparkK3StageRunnerConfiguration config;
	SparkK3StageRunnerDispatch dispatch;
	SparkStatus status;
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	config.descriptor_bytes = (uint32_t)sizeof(config);
	config.stage_index = 0u;
	config.stage_count = 4u;
	config.tp_degree = 1u;
	config.tp_rank = 0u;
	config.max_active_sequence_count = 1u;
	config.max_input_row_count = 1u;
	config.resident_sequence_capacity = 1u;
	config.kv_pages_per_sequence = 2u;
	config.kv_page_bytes = K3GlobalKv::kPageBytes;
	config.rank_pack_path = argv[1];
	config.execution_stream = 0;
	int multiprocessors = 0;
	cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0);
	config.multiprocessors = (uint32_t)multiprocessors;
	config.layer_collective_override = ProbeHook;
	config.layer_collective_context = 0;

	status = SparkK3StageRunnerInitialize(&runner, &config);
	if ( status != SPARK_STATUS_OK )
	{
		printf("INIT FAIL %d\n", (int)status);
		return 1;
	}
	g_probe_buffers = (K3LayerBuffers *)SparkK3StageRunnerProbeBuffers(&runner);
	g_rows = 1u;
	cudaMallocHost(&g_snapshots, (uint64_t)24u * K3_HIDDEN * 2u);

	uint32_t *d_tokens, *d_positions, *d_context, *d_seq, *d_state, *d_out_tok;
	float *d_out_score;
	uint16_t *d_hidden;
	uint32_t h_token = 1u, h_zero = 0u, h_one = 1u;
	cudaMalloc(&d_tokens, 4u); cudaMalloc(&d_positions, 4u);
	cudaMalloc(&d_context, 4u); cudaMalloc(&d_seq, 4u);
	cudaMalloc(&d_state, 4u); cudaMalloc(&d_out_tok, 4u);
	cudaMalloc(&d_out_score, 4u);
	cudaMalloc(&d_hidden, (uint64_t)K3_HIDDEN * 2u);
	cudaMemcpy(d_tokens, &h_token, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_positions, &h_zero, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_context, &h_one, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_seq, &h_zero, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_state, &h_zero, 4u, cudaMemcpyHostToDevice);

	memset(&dispatch, 0, sizeof(dispatch));
	dispatch.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	dispatch.descriptor_bytes = (uint32_t)sizeof(dispatch);
	dispatch.row_count = 1u;
	dispatch.active_sequence_count = 1u;
	dispatch.token_ids = d_tokens;
	dispatch.positions = d_positions;
	dispatch.context_length = d_context;
	dispatch.sequence_of_row = d_seq;
	dispatch.kda_state_index = d_state;
	dispatch.hidden_output_bf16 = d_hidden;
	dispatch.hidden_output_bytes = (uint64_t)K3_HIDDEN * 2u;
	dispatch.output_token_ids = d_out_tok;
	dispatch.output_scores = d_out_score;

	status = SparkK3StageRunnerSubmit(&runner, &dispatch);
	if ( status != SPARK_STATUS_OK )
	{
		printf("SUBMIT FAIL %d\n", (int)status);
		return 1;
	}

	FILE *f = fopen(argv[2], "wb");
	if ( f == 0 )
		return 1;
	fwrite(g_snapshots, sizeof(g_snapshots), 1u, f);
	fclose(f);
	printf("probe dumped 24 layers\n");
	return 0;
}
