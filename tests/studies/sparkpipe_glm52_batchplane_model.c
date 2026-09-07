#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define BP_EFFECTIVE_MEMORY_BANDWIDTH_BYTES_PER_SECOND 174.0e9
#define BP_EFFECTIVE_FP8_FLOPS_PER_SECOND 200.0e12
#define BP_LAYERS_PER_RANK 6u
#define BP_EXPERT_COUNT 256u
#define BP_TOP_K 8u
#define BP_FP8_EXPERT_BYTES 37.75e6
#define BP_LATENT_BYTES_PER_TOKEN_PER_LAYER 1152.0
#define BP_DSA_INDEX_BYTES_PER_TOKEN_PER_SHARE_GROUP 128.0
#define BP_DSA_SHARE_GROUP_LAYERS 4u
#define BP_SPECULATIVE_ROWS 8u
#define BP_SPECULATIVE_EXPECTED_COMMIT 5.67

typedef struct SparkBatchPlaneEstimate
{
	double active_experts;
	double average_rows_per_active_expert;
	double expert_sweeps_per_active_expert;
	double expert_bytes_per_batch;
	double attention_bytes_per_row;
	double committed_tokens_per_second;
	double decoded_rows_per_second;
}
SparkBatchPlaneEstimate;

static double SparkBatchPlaneExpectedActiveExperts(uint32_t rows)
{
	double route_count;
	double inactive_probability;
	double active_experts;
	double maximum_active_experts;

	route_count = (double)rows * (double)BP_TOP_K;
	inactive_probability = exp(route_count * log1p(-1.0 / (double)BP_EXPERT_COUNT));
	active_experts = (double)BP_EXPERT_COUNT * (1.0 - inactive_probability);
	maximum_active_experts = route_count < (double)BP_EXPERT_COUNT
		? route_count
		: (double)BP_EXPERT_COUNT;
	if ( active_experts > maximum_active_experts )
		active_experts = maximum_active_experts;
	if ( active_experts < 1.0 )
		active_experts = 1.0;
	return(active_experts);
}

static uint32_t SparkBatchPlaneGroupedTileRows(uint32_t rows)
{
	uint64_t route_count;
	uint64_t mean_rows;
	uint64_t peak_rows;

	route_count = (uint64_t)rows * BP_TOP_K;
	mean_rows = (route_count + BP_EXPERT_COUNT - 1u) / BP_EXPERT_COUNT;
	peak_rows = mean_rows * 2u;
	if ( peak_rows <= 16u )
		return(16u);
	if ( peak_rows <= 32u )
		return(32u);
	return(64u);
}

static double SparkBatchPlaneAttentionBytesPerRow(
	double context_tokens,
	double selected_tokens)
{
	double selected;
	double index_group_count;

	selected = selected_tokens < context_tokens
		? selected_tokens
		: context_tokens;
	index_group_count = (double)BP_LAYERS_PER_RANK /
		(double)BP_DSA_SHARE_GROUP_LAYERS;
	return((double)BP_LAYERS_PER_RANK * selected *
		BP_LATENT_BYTES_PER_TOKEN_PER_LAYER +
		index_group_count * context_tokens *
		BP_DSA_INDEX_BYTES_PER_TOKEN_PER_SHARE_GROUP);
}

static double SparkBatchPlaneExpertFlopsPerRow(void)
{
	return(2.0 * (double)BP_TOP_K * BP_FP8_EXPERT_BYTES *
		(double)BP_LAYERS_PER_RANK);
}

static void SparkBatchPlaneEstimateBuild(
	uint32_t rows,
	double context_tokens,
	double selected_tokens,
	double realtime_fraction,
	SparkBatchPlaneEstimate *estimate)
{
	double route_count;
	double available_bandwidth;
	double available_flops;
	double expert_bytes_per_row;
	double rows_per_second_from_bandwidth;
	double rows_per_second_from_compute;
	double commit_ratio;
	uint32_t tile_rows;

	estimate->active_experts = SparkBatchPlaneExpectedActiveExperts(rows);
	route_count = (double)rows * (double)BP_TOP_K;
	estimate->average_rows_per_active_expert = route_count /
		estimate->active_experts;
	tile_rows = SparkBatchPlaneGroupedTileRows(rows);
	estimate->expert_sweeps_per_active_expert = 1.0;
	estimate->expert_bytes_per_batch =
		(double)BP_LAYERS_PER_RANK * estimate->active_experts *
		BP_FP8_EXPERT_BYTES * estimate->expert_sweeps_per_active_expert;
	expert_bytes_per_row = estimate->expert_bytes_per_batch / (double)rows;
	estimate->attention_bytes_per_row = SparkBatchPlaneAttentionBytesPerRow(
		context_tokens,selected_tokens);
	available_bandwidth = BP_EFFECTIVE_MEMORY_BANDWIDTH_BYTES_PER_SECOND *
		(1.0 - realtime_fraction);
	available_flops = BP_EFFECTIVE_FP8_FLOPS_PER_SECOND *
		(1.0 - realtime_fraction);
	rows_per_second_from_bandwidth = available_bandwidth /
		(expert_bytes_per_row + estimate->attention_bytes_per_row);
	rows_per_second_from_compute = available_flops /
		SparkBatchPlaneExpertFlopsPerRow();
	estimate->decoded_rows_per_second = rows_per_second_from_bandwidth <
		rows_per_second_from_compute
		? rows_per_second_from_bandwidth
		: rows_per_second_from_compute;
	commit_ratio = BP_SPECULATIVE_EXPECTED_COMMIT /
		(double)BP_SPECULATIVE_ROWS;
	estimate->committed_tokens_per_second =
		estimate->decoded_rows_per_second * commit_ratio;
	(void)tile_rows;
}

static void SparkBatchPlanePrintTable(
	double context_tokens,
	double selected_tokens,
	double realtime_fraction)
{
	static const uint32_t batch_rows[] =
	{
		1u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u
	};
	SparkBatchPlaneEstimate estimate;
	uint32_t batch_index;

	printf("\nGLM 5.2 FP8 expert queue, context %.0f, selected %.0f\n",
		context_tokens,selected_tokens);
	printf("replay/chunk expert-sweep multiplier: 1.0 (removed)\n");
	printf("%6s %10s %12s %8s %13s %13s\n",
		"B", "active_exp", "rows/active", "tile_M", "decode_rows/s", "commit_tok/s");
	for ( batch_index = 0u;
		batch_index < sizeof(batch_rows) / sizeof(batch_rows[0]);
		++batch_index )
	{
		uint32_t rows;

		rows = batch_rows[batch_index];
		SparkBatchPlaneEstimateBuild(
			rows,context_tokens,selected_tokens,realtime_fraction,&estimate);
		printf("%6u %10.1f %12.2f %8u %13.0f %13.0f\n",
			rows,
			estimate.active_experts,
			estimate.average_rows_per_active_expert,
			SparkBatchPlaneGroupedTileRows(rows),
			estimate.decoded_rows_per_second,
			estimate.committed_tokens_per_second);
	}
}

int main(int argc,char **argv)
{
	double realtime_fraction;

	realtime_fraction = argc > 1
		? strtoul(argv[1],0,10) / 1000.0
		: 0.25;
	if ( realtime_fraction < 0.0 || realtime_fraction > 0.9 )
		realtime_fraction = 0.25;
	printf("sparkpipe GLM 5.2 grouped-MoE batch-plane estimator\n");
	printf("FP8 experts; one sealed route/group/sweep/scatter cycle per batch\n");
	printf("effective memory bandwidth %.0f GB/s; realtime reserve %.0f%%\n",
		BP_EFFECTIVE_MEMORY_BANDWIDTH_BYTES_PER_SECOND / 1.0e9,
		realtime_fraction * 100.0);
	SparkBatchPlanePrintTable(2048.0,2048.0,realtime_fraction);
	SparkBatchPlanePrintTable(8192.0,2048.0,realtime_fraction);
	SparkBatchPlanePrintTable(32768.0,2048.0,realtime_fraction);
	return(0);
}
