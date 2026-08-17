// K3 serial-TP16 replay — wire the shared serial_tp_replay harness to the K3
// stage runner. Runs the FULL 4-layer slice once (the golden) and each of the
// 16 rank shards once at tp_degree 1, sums the rank partials host-side in rank
// order (fp32, RNE — the harness's all_reduce_sum contract), and compares the
// reconstruction against the golden with a rel ~0.06 tolerance (16 BF16
// partials summed vs one full-width BF16 GEMM output).
//
// The collective-emulation contract is docs/serial_tp_replay.md + the TP4
// precedent tools/k3_tp4_equivalence_check.py: full[k] ~= sum_r rank_r[k].
// The head MAXLOC emulation is out of scope for this 4-layer slice (no lm_head
// in a non-final slice), so this replay closes the SUM half and the per-layer
// w2 reconstruction; the head merge is exercised separately.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <sys/stat.h>

#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "inference/llms/kimi_k3/layer.cuh"
#include "serial_tp_replay.h"

struct K3ReplayContext
{
	const char **rank_packs;      /* 16 rank-pack paths */
	uint32_t token_id;            /* the single token driven through the slice */
	uint32_t multiprocessors;
	cudaStream_t stream;
	SparkK3StageRunner runner;
	uint32_t *d_tokens, *d_positions, *d_context, *d_seq_of_row, *d_state_index;
	uint32_t *d_output_tokens;
	float *d_output_scores;
	uint16_t *d_hidden;
};

static uint64_t FileBytes(const char *path)
{
	struct stat st;
	if ( stat(path, &st) != 0 )
		return 0u;
	return (uint64_t)st.st_size;
}

static int k3_load_shard(uint32_t rank, void *context)
{
	K3ReplayContext *ctx = (K3ReplayContext *)context;
	SparkK3StageRunnerConfiguration config;
	SparkStatus status;
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	config.descriptor_bytes = (uint32_t)sizeof(config);
	config.stage_index = 0u;
	config.stage_count = 1u;            /* --pp1 derive: slice bounds from the pack */
	config.tp_degree = 1u;
	config.tp_rank = 0u;
	config.max_active_sequence_count = 1u;
	config.max_input_row_count = 1u;
	config.resident_sequence_capacity = 1u;
	config.kv_pages_per_sequence = 2u;
	config.kv_page_bytes = K3GlobalKv::kPageBytes;
	config.rank_pack_path = ctx->rank_packs[rank];
	config.execution_stream = ctx->stream;
	config.multiprocessors = ctx->multiprocessors;
	/* direct launches only: one submit per rank, no graph capture */
	config.flags = 0u;
	config.layer_collective_override = 0; /* no-op at tp_degree 1 */
	status = SparkK3StageRunnerInitialize(&ctx->runner, &config);
	return status == SPARK_STATUS_OK ? 0 : -1;
}

static int k3_free_shard(uint32_t rank, void *context)
{
	(void)rank;
	K3ReplayContext *ctx = (K3ReplayContext *)context;
	SparkK3StageRunnerDestroy(&ctx->runner);
	return 0;
}

static uint64_t k3_shard_bytes(uint32_t rank, void *context)
{
	K3ReplayContext *ctx = (K3ReplayContext *)context;
	return FileBytes(ctx->rank_packs[rank]);
}

static int k3_run_rank(uint32_t rank, const uint16_t *input_bf16,
	uint16_t *partial_out_bf16, uint64_t input_elements,
	uint64_t partial_elements, void *context)
{
	(void)rank;
	(void)input_bf16;
	(void)input_elements;
	K3ReplayContext *ctx = (K3ReplayContext *)context;
	if ( partial_elements != K3_HIDDEN )
		return -1;
	SparkK3StageRunnerDispatch dispatch;
	memset(&dispatch, 0, sizeof(dispatch));
	dispatch.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	dispatch.descriptor_bytes = (uint32_t)sizeof(dispatch);
	dispatch.request_id = 1u;
	dispatch.sequence_id = 1u;
	dispatch.sequence_position = 0u;
	dispatch.row_count = 1u;
	dispatch.active_sequence_count = 1u;
	dispatch.token_ids = ctx->d_tokens;
	dispatch.positions = ctx->d_positions;
	dispatch.context_length = ctx->d_context;
	dispatch.sequence_of_row = ctx->d_seq_of_row;
	dispatch.kda_state_index = ctx->d_state_index;
	dispatch.hidden_output_bf16 = ctx->d_hidden;
	dispatch.hidden_output_bytes = (uint64_t)K3_HIDDEN * 2u;
	dispatch.output_token_ids = ctx->d_output_tokens;
	dispatch.output_scores = ctx->d_output_scores;
	SparkStatus status = SparkK3StageRunnerSubmit(&ctx->runner, &dispatch);
	if ( status != SPARK_STATUS_OK )
		return -1;
	cudaError_t err = cudaMemcpy(partial_out_bf16, ctx->d_hidden,
		(uint64_t)K3_HIDDEN * 2u, cudaMemcpyDeviceToHost);
	return err == cudaSuccess ? 0 : -1;
}

static int RunFullSlice(const char *pack_path, uint16_t *golden_out,
	K3ReplayContext *ctx)
{
	/* the golden is the FULL (unsharded) slice at tp_degree 1: one submit,
	 * one full-width hidden. */
	SparkK3StageRunner runner;
	SparkK3StageRunnerConfiguration config;
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	config.descriptor_bytes = (uint32_t)sizeof(config);
	config.stage_index = 0u;
	config.stage_count = 1u;
	config.tp_degree = 1u;
	config.tp_rank = 0u;
	config.max_active_sequence_count = 1u;
	config.max_input_row_count = 1u;
	config.resident_sequence_capacity = 1u;
	config.kv_pages_per_sequence = 2u;
	config.kv_page_bytes = K3GlobalKv::kPageBytes;
	config.rank_pack_path = pack_path;
	config.execution_stream = ctx->stream;
	config.multiprocessors = ctx->multiprocessors;
	config.flags = 0u;
	config.layer_collective_override = 0;
	SparkStatus status = SparkK3StageRunnerInitialize(&runner, &config);
	if ( status != SPARK_STATUS_OK )
	{
		printf("GOLDEN INIT FAIL %d\n", (int)status);
		return -1;
	}
	SparkK3StageRunnerDispatch dispatch;
	memset(&dispatch, 0, sizeof(dispatch));
	dispatch.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	dispatch.descriptor_bytes = (uint32_t)sizeof(dispatch);
	dispatch.request_id = 1u;
	dispatch.sequence_id = 1u;
	dispatch.sequence_position = 0u;
	dispatch.row_count = 1u;
	dispatch.active_sequence_count = 1u;
	dispatch.token_ids = ctx->d_tokens;
	dispatch.positions = ctx->d_positions;
	dispatch.context_length = ctx->d_context;
	dispatch.sequence_of_row = ctx->d_seq_of_row;
	dispatch.kda_state_index = ctx->d_state_index;
	dispatch.hidden_output_bf16 = ctx->d_hidden;
	dispatch.hidden_output_bytes = (uint64_t)K3_HIDDEN * 2u;
	dispatch.output_token_ids = ctx->d_output_tokens;
	dispatch.output_scores = ctx->d_output_scores;
	status = SparkK3StageRunnerSubmit(&runner, &dispatch);
	if ( status != SPARK_STATUS_OK )
	{
		printf("GOLDEN SUBMIT FAIL %d\n", (int)status);
		SparkK3StageRunnerDestroy(&runner);
		return -1;
	}
	cudaError_t err = cudaMemcpy(golden_out, ctx->d_hidden,
		(uint64_t)K3_HIDDEN * 2u, cudaMemcpyDeviceToHost);
	SparkK3StageRunnerDestroy(&runner);
	return err == cudaSuccess ? 0 : -1;
}

int main(int argc, char **argv)
{
	if ( argc < 3 )
	{
		printf("usage: k3_serial_tp <full.pack> <rank00.pack> ... <rank15.pack> "
			"[--token N]\n");
		return 2;
	}
	const char *golden_pack = argv[1];
	if ( argc != 18 && argc != 20 )
	{
		printf("expected 16 rank packs (17 path args) [--token N]\n");
		return 2;
	}
	K3ReplayContext ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.token_id = 1u;
	for ( int i = 2; i < argc; ++i )
	{
		if ( strcmp(argv[i], "--token") == 0 && i + 1 < argc )
		{
			ctx.token_id = (uint32_t)atoi(argv[i + 1]);
			i++;
		}
	}
	ctx.rank_packs = (const char **)calloc(16u, sizeof(char *));
	for ( uint32_t r = 0u; r < 16u; ++r )
		ctx.rank_packs[r] = argv[2 + r];

	int mps = 0;
	cudaDeviceGetAttribute(&mps, cudaDevAttrMultiProcessorCount, 0);
	ctx.multiprocessors = (uint32_t)mps;
	cudaStreamCreate(&ctx.stream);

	/* device input tensors (single token) */
	uint32_t h_token = ctx.token_id, h_pos = 0u, h_ctx = 1u, h_seq = 0u, h_state = 0u;
	cudaMalloc(&ctx.d_tokens, 4u);
	cudaMalloc(&ctx.d_positions, 4u);
	cudaMalloc(&ctx.d_context, 4u);
	cudaMalloc(&ctx.d_seq_of_row, 4u);
	cudaMalloc(&ctx.d_state_index, 4u);
	cudaMalloc(&ctx.d_output_tokens, 4u);
	cudaMalloc(&ctx.d_output_scores, 4u);
	cudaMalloc(&ctx.d_hidden, (uint64_t)K3_HIDDEN * 2u);
	cudaMemcpy(ctx.d_tokens, &h_token, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(ctx.d_positions, &h_pos, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(ctx.d_context, &h_ctx, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(ctx.d_seq_of_row, &h_seq, 4u, cudaMemcpyHostToDevice);
	cudaMemcpy(ctx.d_state_index, &h_state, 4u, cudaMemcpyHostToDevice);

	/* golden: full slice */
	std::vector<uint16_t> golden(K3_HIDDEN);
	if ( RunFullSlice(golden_pack, golden.data(), &ctx) != 0 )
	{
		printf("GOLDEN FAIL\n");
		return 1;
	}
	printf("golden: full slice hidden captured (%u bf16)\n", (uint32_t)K3_HIDDEN);

	/* sweep 16 ranks, one shard resident at a time, budget-checked */
	SparkSerialTpRankHooks hooks;
	hooks.load_shard = k3_load_shard;
	hooks.free_shard = k3_free_shard;
	hooks.shard_device_bytes = k3_shard_bytes;
	hooks.run_rank = k3_run_rank;
	SparkSerialTpDeviceBudget budget;
	budget.cap_bytes = SPARK_SERIAL_TP_REPLAY_DEFAULT_DEVICE_BUDGET_BYTES;
	budget.held_bytes = 0u;
	budget.peak_held_bytes = 0u;
	std::vector<uint16_t> partials((uint64_t)16u * K3_HIDDEN);
	int rc = spark_serial_tp_sweep(16u, K3_HIDDEN, 0, 0u, partials.data(),
		&hooks, &ctx, &budget);
	if ( rc != 0 )
	{
		printf("SWEEP FAIL %d (peak_held %llu)\n", rc,
			(unsigned long long)budget.peak_held_bytes);
		return 1;
	}
	printf("sweep: 16 ranks run, peak held %llu bytes\n",
		(unsigned long long)budget.peak_held_bytes);

	/* all-reduce sum (rank order, fp32, RNE) then compare vs golden */
	std::vector<uint16_t> reconstructed(K3_HIDDEN);
	spark_serial_tp_all_reduce_sum_bf16(partials.data(), 16u, K3_HIDDEN,
		reconstructed.data());

	uint32_t failures = 0u;
	double worst = 0.0;
	uint32_t worst_col = 0u;
	for ( uint32_t k = 0u; k < K3_HIDDEN; ++k )
	{
		uint32_t a = reconstructed[k], b = golden[k];
		uint32_t au = a << 16u, bu = b << 16u;
		float af, bf;
		memcpy(&af, &au, sizeof(af));
		memcpy(&bf, &bu, sizeof(bf));
		double delta = std::fabs((double)af - (double)bf);
		double rel = delta / std::fmax(std::fabs((double)bf), 0.25);
		if ( rel > worst )
		{
			worst = rel;
			worst_col = k;
		}
		if ( rel > 0.06 )
		{
			failures++;
			if ( failures <= 8u )
				printf("  mismatch[%u] golden=%.6g recon=%.6g rel=%.4f\n",
					k, (double)bf, (double)af, rel);
		}
	}
	printf("sum-vs-golden: worst rel %.5f at col %u, %u beyond 0.06\n",
		worst, worst_col, failures);

	/* hash compare as the reduced-precision fallback */
	uint64_t gh = spark_serial_tp_hash_elements(golden.data(), K3_HIDDEN,
		sizeof(uint16_t));
	uint64_t rh = spark_serial_tp_hash_elements(reconstructed.data(),
		K3_HIDDEN, sizeof(uint16_t));
	printf("hash: golden %016llx recon %016llx %s\n",
		(unsigned long long)gh, (unsigned long long)rh,
		gh == rh ? "EXACT" : "differ");

	if ( failures == 0u )
	{
		printf("k3 serial-TP16 replay PASS (sum == golden within 0.06 rel)\n");
		return 0;
	}
	printf("k3 serial-TP16 replay FAIL\n");
	return 1;
}
