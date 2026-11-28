#include "drafter_block_probe_common.h"

#define PROBE_HIDDEN 5120u
#define PROBE_TAPS 5u
#define PROBE_LAYERS 5u
#define PROBE_WINDOW 8176u
#define PROBE_COLUMNS 8192u
#define PROBE_KV (8u * PROBE_COLUMNS * 128u)
#define PROBE_BLOCK 16u
#define PROBE_DRAFTS 15u
#define PROBE_LANES 16u

typedef SparkStatus (*CacheRows)(void *, uint64_t, uint32_t, uint16_t *, uint16_t *, int64_t *);
typedef SparkStatus (*ForwardRows)(void *, uint64_t, uint32_t, uint64_t, uint16_t *, uint16_t *, uint32_t *, uint64_t *);
typedef SparkStatus (*LayerTrace)(void *, uint64_t, uint32_t, uint64_t, uint32_t, uint16_t *);

static const DsparkProbeConfig probe_cfg = {
	"qwen38_dspine_block_probe",
	"qwen27bdspine",
	"ac90f1df842e",
	UINT64_C(800) << 20,
	PROBE_HIDDEN,
	PROBE_TAPS,
	PROBE_DRAFTS,
	PROBE_WINDOW,
	"spark_qwen38_dspine_cache",
	"spark_qwen38_dspine_forward",
	"spark_qwen38_dspine_trace",
};

static SparkStatus ForwardTimed(DsparkProbe *probe, uint64_t sequence, uint32_t anchor, uint64_t position, uint64_t *elapsed)
{
	return ((ForwardRows)probe->fn_b)(probe->state, sequence, anchor, position, 0, 0, 0, elapsed);
}

static int Compare(DsparkProbe *probe, const char *directory)
{
	const uint64_t row_elements = (uint64_t)PROBE_TAPS * PROBE_HIDDEN;
	uint64_t tap_bytes = 0u, token_bytes = 0u, first_bytes = 0u, elapsed = 0u, rows, start;
	uint16_t *taps = (uint16_t *)DsparkProbeReadAll(directory, "taps.bin", &tap_bytes);
	uint32_t *tokens = (uint32_t *)DsparkProbeReadAll(directory, "tokens.bin", &token_bytes);
	uint64_t *first_position = (uint64_t *)DsparkProbeReadAll(directory, "first.bin", &first_bytes);
	uint16_t *keys = (uint16_t *)malloc((size_t)PROBE_KV * sizeof(uint16_t)), *values = (uint16_t *)malloc((size_t)PROBE_KV * sizeof(uint16_t));
	uint16_t *y = (uint16_t *)malloc((size_t)PROBE_BLOCK * PROBE_HIDDEN * sizeof(uint16_t)), *normed = (uint16_t *)malloc((size_t)PROBE_BLOCK * PROBE_HIDDEN * sizeof(uint16_t));
	uint32_t candidates[PROBE_DRAFTS * PROBE_LANES];
	int64_t *positions = (int64_t *)malloc((size_t)PROBE_WINDOW * sizeof(int64_t));
	if ( taps == 0 || tokens == 0 || first_position == 0 || first_bytes != sizeof(uint64_t) || keys == 0 || values == 0 || y == 0 || normed == 0 ||
		positions == 0 )
	{
		fprintf(stderr, "%s needs first.bin, taps.bin and tokens.bin\n", directory);
		return 1;
	}
	start = *first_position;
	rows = start + tap_bytes / (row_elements * sizeof(uint16_t));
	if ( (rows - start) * row_elements * sizeof(uint16_t) != tap_bytes || token_bytes != (rows + 1u) * sizeof(uint32_t) || rows == start )
	{
		fprintf(stderr, "taps.bin holds %llu rows from %llu; tokens.bin must hold a token for every position up to the anchor\n",
			(unsigned long long)(rows - start), (unsigned long long)start);
		return 1;
	}
	for ( uint64_t first = start; first < rows; first += DSPARK_PROBE_FRAME_ROWS )
	{
		const uint32_t part = rows - first < DSPARK_PROBE_FRAME_ROWS ? (uint32_t)(rows - first) : DSPARK_PROBE_FRAME_ROWS;
		if ( DsparkProbeIngest(probe, &probe_cfg, DSPARK_PROBE_SEQUENCE, 1u, taps + (first - start) * row_elements, tokens + first, first, part) != 0 )
			return 1;
	}
	for ( uint32_t layer = 0u; layer < PROBE_LAYERS; layer++ )
	{
		char name[64];
		if ( ((CacheRows)probe->fn_a)(probe->state, DSPARK_PROBE_SEQUENCE, layer, keys, values, positions) != SPARK_STATUS_OK )
		{
			fprintf(stderr, "cache read of layer %u failed\n", layer);
			return 1;
		}
		snprintf(name, sizeof(name), "keys%u.bin", layer);
		if ( DsparkProbeWriteAll(directory, name, keys, (uint64_t)PROBE_KV * sizeof(uint16_t)) != 0 )
			return 1;
		snprintf(name, sizeof(name), "values%u.bin", layer);
		if ( DsparkProbeWriteAll(directory, name, values, (uint64_t)PROBE_KV * sizeof(uint16_t)) != 0 )
			return 1;
	}
	if ( DsparkProbeWriteAll(directory, "cache_positions.bin", positions, (uint64_t)PROBE_WINDOW * sizeof(int64_t)) != 0 ||
		((ForwardRows)probe->fn_b)(probe->state, DSPARK_PROBE_SEQUENCE, tokens[rows], rows, y, normed, candidates, &elapsed) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "forward failed\n");
		return 1;
	}
	printf("forward, head and greedy walk at position %llu anchor %u: %.3f ms\n", (unsigned long long)rows, tokens[rows], (double)elapsed / 1e6);
	if ( DsparkProbeWriteAll(directory, "y.bin", y, (uint64_t)PROBE_BLOCK * PROBE_HIDDEN * sizeof(uint16_t)) != 0 ||
		DsparkProbeWriteAll(directory, "normed.bin", normed, (uint64_t)PROBE_BLOCK * PROBE_HIDDEN * sizeof(uint16_t)) != 0 ||
		DsparkProbeWriteAll(directory, "candidates.bin", candidates, sizeof(candidates)) != 0 )
		return 1;
	for ( uint32_t layers = 1u; layers <= PROBE_LAYERS; layers++ )
	{
		char name[64];
		snprintf(name, sizeof(name), "hidden%u.bin", layers);
		if ( ((LayerTrace)probe->fn_c)(probe->state, DSPARK_PROBE_SEQUENCE, tokens[rows], rows, layers, y) != SPARK_STATUS_OK ||
			DsparkProbeWriteAll(directory, name, y, (uint64_t)PROBE_BLOCK * PROBE_HIDDEN * sizeof(uint16_t)) != 0 )
		{
			fprintf(stderr, "layer trace after %u layers failed\n", layers);
			return 1;
		}
	}
	if ( DsparkProbeDrafts(probe, &probe_cfg, directory, tokens, rows) != 0 || DsparkProbeDonors(probe, &probe_cfg, taps, tokens, start, rows) != 0 )
		return 1;
	printf("wrote the cache, y.bin, normed.bin, candidates.bin and drafts.bin to %s\n", directory);
	free(keys);
	free(values);
	free(y);
	free(normed);
	free(positions);
	free(taps);
	free(tokens);
	free(first_position);
	return 0;
}

int main(int argc, char **argv)
{
	return DsparkProbeMain(argc, argv, &probe_cfg, Compare, ForwardTimed, "forward and head");
}
