
#include <cstdio>
#include <cstring>

#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "inference/llms/kimi_k3/layer.cuh"

static uint16_t *g_snapshots_device;
static uint16_t *g_snapshots;
static uint32_t g_rows = 0u;
static K3LayerBuffers *g_probe_buffers;

__device__ static uint32_t g_snap_executions;

__global__ static void SnapKernel(const uint16_t *src, uint16_t *dst, uint32_t count)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( i < count )
		dst[i] = src[i];
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
		atomicAdd(&g_snap_executions, 1u);
}

static uint32_t g_hook_count = 0u;

static void ProbeHook(void *context, void *stream_void, uint32_t layer,
	uint32_t phase)
{
	(void)context;
	(void)phase;
	g_hook_count++;
	SnapKernel<<<(K3_HIDDEN + 255u) / 256u, 256u, 0, (cudaStream_t)stream_void>>>(
		g_probe_buffers->hidden_bf16,
		&g_snapshots_device[layer * K3_HIDDEN], K3_HIDDEN);
	cudaError_t herr = cudaGetLastError();
	if ( g_hook_count <= 3u )
	{
		uint16_t probe_vals[2];
		cudaMemcpy(probe_vals, g_probe_buffers->hidden_bf16, 4u,
			cudaMemcpyDeviceToHost);
		printf("hook layer %u err=%d hidden[0..1]=%04x %04x\n", layer,
			(int)herr, probe_vals[0], probe_vals[1]);
	}
	SnapKernel<<<(K3_HIDDEN + 255u) / 256u, 256u, 0, (cudaStream_t)stream_void>>>(
		g_probe_buffers->hidden_bf16,
		&g_snapshots_device[layer * K3_HIDDEN], K3_HIDDEN);
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
	cudaMalloc(&g_snapshots_device, (uint64_t)24u * K3_HIDDEN * 2u);
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

	SnapKernel<<<(K3_HIDDEN + 255u) / 256u, 256u, 0, (cudaStream_t)0>>>(
		g_probe_buffers->hidden_bf16, &g_snapshots_device[1u * K3_HIDDEN], K3_HIDDEN);
	cudaMemcpy(g_snapshots, g_snapshots_device,
		(uint64_t)24u * K3_HIDDEN * 2u, cudaMemcpyDeviceToHost);
	printf("post-submit snap[1][0..1] = %04x %04x\n", g_snapshots[K3_HIDDEN],
		g_snapshots[K3_HIDDEN + 1u]);
	uint32_t executions = 0u;
	cudaMemcpyFromSymbol(&executions, g_snap_executions, 4u, 0, cudaMemcpyDeviceToHost);
	printf("snap kernel executions: %u (expected 25)\n", executions);
	FILE *f = fopen(argv[2], "wb");
	if ( f == 0 )
		return 1;
	fwrite(g_snapshots, (uint64_t)24u * K3_HIDDEN * 2u, 1u, f);
	fclose(f);
	printf("probe dumped 24 layers\n");
	return 0;
}
