// K3 serial-TP16 replay, per-half TP plan (docs/serial_tp_replay.md +
// docs/SERIAL_TP16_K3_PER_LAYER_PLAN.md): the runner's StepHalf runs ONE
// layer-half per rank with the FULL hidden + FULL AttnRes partial replicated
// in and the rank's CONTRIBUTION (the un-folded input-sharded projection
// output) copied out. The harness sweeps the 16 ranks per half, host-sums the
// contributions (rank-order fp32 + RNE), and folds the sum into the partial.
// 8 halves for the 4-layer slice: 2 per layer (phase 0 = attention,
// phase 1 = MLP).
//
// The 16 runners are PRE-LOADED (K3 has recurrent KDA state and a cross-layer
// AttnRes bank that must persist across halves), so the harness's load/free
// hooks are no-ops and shard_device_bytes reports 0.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "sparkpipe/spark_k3_pack_load.h"
#include "inference/llms/kimi_k3/layer.cuh"
#include "serial_tp_replay.h"

struct K3ReplayContext
{
	const char **rank_packs;
	uint32_t layer;
	uint32_t phase;
	uint32_t multiprocessors;
	cudaStream_t stream;
	SparkK3StageRunner *runners;
	uint16_t *d_hidden;
	uint16_t *d_partial;
	uint16_t *d_out;
};

static int k3_load_shard(uint32_t rank, void *context)
{
	(void)rank; (void)context;
	return 0;
}

static int k3_free_shard(uint32_t rank, void *context)
{
	(void)rank; (void)context;
	return 0;
}

static uint64_t k3_shard_bytes(uint32_t rank, void *context)
{
	(void)rank; (void)context;
	return 0u;
}

static int k3_run_rank(uint32_t rank, const uint16_t *input_bf16,
	uint16_t *partial_out_bf16, uint64_t input_elements,
	uint64_t partial_elements, void *context)
{
	(void)input_bf16; (void)input_elements;
	K3ReplayContext *ctx = (K3ReplayContext *)context;
	if ( partial_elements != K3_HIDDEN )
		return -1;
	SparkStatus status = SparkK3StageRunnerStepHalf(&ctx->runners[rank],
		ctx->layer, ctx->phase,
		ctx->phase == 0u ? ctx->d_hidden : (const uint16_t *)0,
		ctx->d_partial, ctx->d_out);
	if ( status != SPARK_STATUS_OK )
		return -1;
	cudaError_t err = cudaMemcpy(partial_out_bf16, ctx->d_out,
		(uint64_t)K3_HIDDEN * 2u, cudaMemcpyDeviceToHost);
	return err == cudaSuccess ? 0 : -1;
}

static int RunFullSlice(const char *pack_path, uint16_t *golden_out,
	uint32_t multiprocessors, cudaStream_t stream)
{
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
	config.execution_stream = stream;
	config.multiprocessors = multiprocessors;
	config.flags = 0u;
	config.layer_collective_override = 0;
	SparkStatus status = SparkK3StageRunnerInitialize(&runner, &config);
	if ( status != SPARK_STATUS_OK )
		return -1;
	uint16_t *d_hidden;
	cudaMalloc(&d_hidden, (uint64_t)K3_HIDDEN * 2u);
	uint32_t h_token = 1u, h_pos = 0u, h_ctx = 1u, h_seq = 0u, h_state = 0u;
	uint32_t *d_tokens; cudaMalloc(&d_tokens, 4u); cudaMemcpy(d_tokens, &h_token, 4u, cudaMemcpyHostToDevice);
	uint32_t *d_pos; cudaMalloc(&d_pos, 4u); cudaMemcpy(d_pos, &h_pos, 4u, cudaMemcpyHostToDevice);
	uint32_t *d_con; cudaMalloc(&d_con, 4u); cudaMemcpy(d_con, &h_ctx, 4u, cudaMemcpyHostToDevice);
	uint32_t *d_seq; cudaMalloc(&d_seq, 4u); cudaMemcpy(d_seq, &h_seq, 4u, cudaMemcpyHostToDevice);
	uint32_t *d_st;  cudaMalloc(&d_st, 4u);  cudaMemcpy(d_st, &h_state, 4u, cudaMemcpyHostToDevice);
	uint32_t *d_ot; cudaMalloc(&d_ot, 4u); float *d_os; cudaMalloc(&d_os, 4u);
	SparkK3StageRunnerDispatch dispatch;
	memset(&dispatch, 0, sizeof(dispatch));
	dispatch.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	dispatch.descriptor_bytes = (uint32_t)sizeof(dispatch);
	dispatch.request_id = 1u;
	dispatch.sequence_id = 1u;
	dispatch.row_count = 1u;
	dispatch.active_sequence_count = 1u;
	dispatch.token_ids = d_tokens;
	dispatch.positions = d_pos;
	dispatch.context_length = d_con;
	dispatch.sequence_of_row = d_seq;
	dispatch.kda_state_index = d_st;
	dispatch.hidden_output_bf16 = d_hidden;
	dispatch.hidden_output_bytes = (uint64_t)K3_HIDDEN * 2u;
	dispatch.output_token_ids = d_ot;
	dispatch.output_scores = d_os;
	status = SparkK3StageRunnerSubmit(&runner, &dispatch);
	if ( status != SPARK_STATUS_OK )
		return -1;
	cudaError_t err = cudaMemcpy(golden_out, d_hidden,
		(uint64_t)K3_HIDDEN * 2u, cudaMemcpyDeviceToHost);
	SparkK3StageRunnerDestroy(&runner);
	cudaFree(d_hidden); cudaFree(d_tokens); cudaFree(d_pos); cudaFree(d_con);
	cudaFree(d_seq); cudaFree(d_st); cudaFree(d_ot); cudaFree(d_os);
	return err == cudaSuccess ? 0 : -1;
}

int main(int argc, char **argv)
{
	if ( argc != 18 && argc != 20 )
	{
		printf("usage: k3_serial_tp <full.pack> <rank00.pack> ... <rank15.pack> [--token N]\n");
		return 2;
	}
	const char *golden_pack = argv[1];
	uint32_t token = 1u;
	for ( int i = 2; i < argc; ++i )
		if ( strcmp(argv[i], "--token") == 0 && i + 1 < argc )
			token = (uint32_t)atoi(argv[i + 1]);

	int mps = 0;
	cudaDeviceGetAttribute(&mps, cudaDevAttrMultiProcessorCount, 0);

	std::vector<uint16_t> golden(K3_HIDDEN);
	cudaStream_t stream = 0;
	cudaStreamCreate(&stream);
	if ( RunFullSlice(golden_pack, golden.data(), (uint32_t)mps, stream) != 0 )
	{
		printf("GOLDEN FAIL\n");
		return 1;
	}
	printf("golden: full slice hidden captured (%u bf16)\n", (uint32_t)K3_HIDDEN);

	K3ReplayContext ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.rank_packs = (const char **)calloc(16u, sizeof(char *));
	for ( uint32_t r = 0u; r < 16u; ++r )
		ctx.rank_packs[r] = argv[2 + r];
	ctx.multiprocessors = (uint32_t)mps;
	ctx.stream = stream;
	ctx.runners = new SparkK3StageRunner[16];
	memset(ctx.runners, 0, 16u * sizeof(SparkK3StageRunner));
	for ( uint32_t r = 0u; r < 16u; ++r )
	{
		SparkK3StageRunnerConfiguration config;
		memset(&config, 0, sizeof(config));
		config.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
		config.descriptor_bytes = (uint32_t)sizeof(config);
		config.stage_index = 0u;
		config.stage_count = 1u;
		config.tp_degree = 1u;
		config.tp_rank = r;
		config.max_active_sequence_count = 1u;
		config.max_input_row_count = 1u;
		config.resident_sequence_capacity = 1u;
		config.kv_pages_per_sequence = 2u;
		config.kv_page_bytes = K3GlobalKv::kPageBytes;
		config.rank_pack_path = ctx.rank_packs[r];
		config.execution_stream = stream;
		config.multiprocessors = (uint32_t)mps;
		config.flags = 0u;
		config.layer_collective_override = 0;
		SparkStatus status = SparkK3StageRunnerInitialize(&ctx.runners[r], &config);
		if ( status != SPARK_STATUS_OK )
		{
			printf("RANK %u INIT FAIL %d\n", r, (int)status);
			return 1;
		}
	}
	printf("16 rank runners initialized (pre-loaded)\n");

	cudaMalloc(&ctx.d_hidden, (uint64_t)K3_HIDDEN * 2u);
	cudaMalloc(&ctx.d_partial, (uint64_t)K3_HIDDEN * 2u);
	cudaMalloc(&ctx.d_out, (uint64_t)K3_HIDDEN * 2u);

	{
		SparkK3Pack pack;
		if ( SparkK3PackOpen(golden_pack, &pack) != SPARK_STATUS_OK )
		{
			printf("OPEN FULL PACK FAIL\n");
			return 1;
		}
		SparkK3PackEntry entry;
		if ( SparkK3PackLoadEntry(&pack, "model.embed_tokens.weight", &entry) != SPARK_STATUS_OK )
		{
			printf("EMBED ENTRY FAIL\n");
			return 1;
		}
		const uint8_t *payload = (const uint8_t *)SparkK3PackPayload(&pack, &entry);
		cudaMemcpy(ctx.d_hidden, payload + (uint64_t)token * K3_HIDDEN * 2u,
			(uint64_t)K3_HIDDEN * 2u, cudaMemcpyHostToDevice);
		SparkK3PackClose(&pack);
	}

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
	std::vector<uint16_t> contribution(K3_HIDDEN);
	std::vector<float> full_partial_f32(K3_HIDDEN, 0.0f);
	std::vector<uint16_t> full_partial_bf16(K3_HIDDEN);

	for ( uint32_t layer = 0u; layer < 4u; ++layer )
	{
		for ( uint32_t phase = 0u; phase < 2u; ++phase )
		{
			ctx.layer = layer;
			ctx.phase = phase;
			for ( uint32_t k = 0u; k < K3_HIDDEN; ++k )
				full_partial_bf16[k] = spark_serial_tp_f32_to_bf16(full_partial_f32[k]);
			cudaMemcpy(ctx.d_partial, full_partial_bf16.data(),
				(uint64_t)K3_HIDDEN * 2u, cudaMemcpyHostToDevice);
			int rc = spark_serial_tp_sweep(16u, K3_HIDDEN, 0, 0u, partials.data(),
				&hooks, &ctx, &budget);
			if ( rc != 0 )
			{
				printf("SWEEP FAIL %d at layer %u phase %u\n", rc, layer, phase);
				return 1;
			}
			/* Each rank captured its CONTRIBUTION (the un-folded input-sharded
			 * projection output); the rank-order fp32 sum is the full
			 * contribution, folded into the partial exactly as the model's
			 * fused epilogue would. */
			spark_serial_tp_all_reduce_sum_bf16(partials.data(), 16u, K3_HIDDEN,
				contribution.data());
			for ( uint32_t k = 0u; k < K3_HIDDEN; ++k )
				full_partial_f32[k] += spark_serial_tp_bf16_to_f32(contribution[k]);
			printf("  L%uP%u partial[0..2] = %.6g %.6g %.6g\n", layer, phase,
				(double)full_partial_f32[0],
				(double)full_partial_f32[1],
				(double)full_partial_f32[2]);
		}
	}

	std::vector<uint16_t> recon(K3_HIDDEN);
	for ( uint32_t k = 0u; k < K3_HIDDEN; ++k )
		recon[k] = spark_serial_tp_f32_to_bf16(full_partial_f32[k]);

	uint32_t failures = 0u;
	double worst = 0.0;
	uint32_t worst_col = 0u;
	for ( uint32_t k = 0u; k < K3_HIDDEN; ++k )
	{
		uint32_t a = recon[k], b = golden[k];
		uint32_t au = a << 16u, bu = b << 16u;
		float af, bf;
		memcpy(&af, &au, sizeof(af));
		memcpy(&bf, &bu, sizeof(bf));
		double delta = std::fabs((double)af - (double)bf);
		double rel = delta / std::fmax(std::fabs((double)bf), 0.25);
		if ( rel > worst ) { worst = rel; worst_col = k; }
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
	if ( failures == 0u )
	{
		printf("k3 serial-TP16 replay PASS (sum == golden within 0.06 rel)\n");
		return 0;
	}
	printf("k3 serial-TP16 replay FAIL\n");
	return 1;
}
