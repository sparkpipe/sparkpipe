// Per-layer divergence probe: dumps the hidden stream after every layer via
// the slice's layer_collective hook, so two fresh runs can be diffed to find
// the exact layer whose output first diverges (the tail-region race in the
// single-spark gate).

#include <cstdio>
#include <cstring>

#include "sparkpipe/spark_k3_resident_decode_stage_cuda.h"
#include "sparkpipe/spark_k3_resident_decode_stage_module.h"
#include "inference/llms/kimi_k3/layer.cuh"

static uint16_t g_snapshots[24u][K3_HIDDEN];
static uint32_t g_rows = 0u;
static K3LayerBuffers *g_probe_buffers;

static void ProbeHook(void *context, void *stream_void, uint32_t layer)
{
	(void)context;
	/* Async: the synchronous copy masked the race this probe exists to
	 * catch - the queue-ordered copy preserves the timing and the final
	 * synchronise drains everything. */
	cudaMemcpyAsync(g_snapshots[layer], g_probe_buffers->hidden_bf16,
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
	SparkK3ModuleState module;
	SparkK3Dispatch dispatch;
	SparkStatus status;
	memset(&module, 0, sizeof(module));
	status = SparkK3ModuleInitialize(&module, argv[1], 0u, 24u);
	if ( status != SPARK_STATUS_OK )
	{
		printf("MODULE FAIL %d\n", (int)status);
		return 1;
	}
	if ( SparkK3DispatchCreate(&dispatch, &module.sizing, 1u, 1u, 2u,
		K3GlobalKv::kPageBytes, 0) != SPARK_K3_DISPATCH_OK )
	{
		printf("DISPATCH FAIL\n");
		return 1;
	}
	if ( SparkK3DispatchRegisterPack(&module.pack) != SPARK_K3_DISPATCH_OK ||
		SparkK3DispatchBindWeights(&dispatch, &module.pack,
			module.bound, module.bound_count) != SPARK_K3_DISPATCH_OK )
	{
		printf("BIND FAIL\n");
		return 1;
	}
	g_probe_buffers = dispatch.buffers;
	g_rows = 1u;
	dispatch.slice_state->layer_collective = ProbeHook;
	dispatch.slice_state->collective_context = 0;
	dispatch.buffers->tp_sharded = 0u;

	uint32_t *d_tokens, *d_positions, *d_context, *d_seq, *d_state, *d_dense;
	uint32_t *d_route_expert, *d_route_packed, *d_route_source;
	float *d_route_weight;
	uint32_t *d_group_off, *d_prefix1, *d_prefix2, *d_dense_tiles;
	float *d_score;
	uint32_t *d_cand_token, *d_out_token;
	float *d_out_score;
	cudaMalloc(&d_tokens, 4u); cudaMalloc(&d_positions, 4u);
	cudaMalloc(&d_context, 4u); cudaMalloc(&d_seq, 4u);
	cudaMalloc(&d_state, 4u); cudaMalloc(&d_dense, 8u);
	cudaMalloc(&d_route_expert, 64u); cudaMalloc(&d_route_packed, 64u);
	cudaMalloc(&d_route_source, 64u); cudaMalloc(&d_route_weight, 64u);
	cudaMalloc(&d_group_off, (K3_EXPERTS + 1u) * 4u);
	cudaMalloc(&d_prefix1, (K3_EXPERTS + 1u) * 4u);
	cudaMalloc(&d_prefix2, (K3_EXPERTS + 1u) * 4u);
	cudaMalloc(&d_dense_tiles, 8u);
	cudaMalloc(&d_score, 16384u * 4u); cudaMalloc(&d_cand_token, 16384u * 4u);
	cudaMalloc(&d_out_token, 4u); cudaMalloc(&d_out_score, 4u);
	uint32_t h_token = 1u, h_zero = 0u, h_one = 1u, h_dense[2] = { 0u, 1u };
	cudaMemcpy(d_tokens, &h_token, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_positions, &h_zero, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_context, &h_one, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_seq, &h_zero, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_state, &h_zero, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dense, h_dense, 8u, cudaMemcpyHostToDevice);

	const uint16_t *embed = 0;
	SparkK3PackEntry entry;
	if ( SparkK3PackLoadEntry(&module.pack, "model.embed_tokens.weight", &entry) == 0 )
		embed = (const uint16_t *)SparkK3PackPayload(&module.pack, &entry);

	SparkK3StepInput in;
	memset(&in, 0, sizeof(in));
	in.hidden_in = dispatch.buffers->hidden_bf16;
	in.positions = d_positions;
	in.context_length = d_context;
	in.sequence_of_row = d_seq;
	in.kda_state_index = d_state;
	in.route_expert = d_route_expert;
	in.route_packed_row = d_route_packed;
	in.route_source_token = d_route_source;
	in.route_weight = d_route_weight;
	in.group_row_offset = d_group_off;
	in.group_tile_prefix_w1 = d_prefix1;
	in.group_tile_prefix_w2 = d_prefix2;
	in.dense_row_offset = d_dense;
	in.dense_tile_prefix = d_dense_tiles;
	in.head_candidate_score = d_score;
	in.head_candidate_token = d_cand_token;
	in.output_token = d_out_token;
	in.output_score = d_out_score;

	int32_t launch = K3Embedding(embed, d_tokens, dispatch.buffers->hidden_bf16,
		1u, 0u, 40960u, (cudaStream_t)0);
	if ( launch != LM_LAUNCH_OK )
	{
		printf("EMBED FAIL %d\n", launch);
		return 1;
	}
	launch = SparkK3DispatchStep(&dispatch, &in, 1u, 1u, 1u, K3_TOP_K,
		1u, 48u, (cudaStream_t)0);
	if ( launch != SPARK_K3_DISPATCH_OK )
	{
		printf("STEP FAIL %d\n", launch);
		return 1;
	}
	cudaStreamSynchronize((cudaStream_t)0);

	FILE *f = fopen(argv[2], "wb");
	if ( f == 0 )
		return 1;
	fwrite(g_snapshots, sizeof(g_snapshots), 1u, f);
	fclose(f);
	printf("probe dumped %u layers\n", 24u);
	return 0;
}
