#include "drafter_block_probe_common.h"

#define PROBE_HIDDEN 7168u
#define PROBE_TAPS 5u
#define PROBE_LAYERS 5u
#define PROBE_BLOCK 7u
#define PROBE_KV_RANK 512u
#define PROBE_LATENT (PROBE_KV_RANK + 64u)
#define PROBE_QUERIES (PROBE_BLOCK * 96u)

typedef SparkStatus (*ContextRows)(void *, uint64_t, uint32_t, uint64_t, uint32_t, uint16_t *);
typedef SparkStatus (*BlockRows)(void *, uint64_t, uint32_t, uint64_t, uint16_t *, uint64_t *);
typedef SparkStatus (*BlockTrace)(void *, uint64_t, uint32_t, uint64_t, uint32_t, float *, float *, uint16_t *);

static const DsparkProbeConfig probe_cfg = {
	"k31_dspark_block_probe",
	"k31dspark",
	"6a2ef20a05ef",
	UINT64_C(8) << 30,
	PROBE_HIDDEN,
	PROBE_TAPS,
	PROBE_BLOCK,
	0u,
	"spark_k31_dspark_context_rows",
	"spark_k31_dspark_block_rows",
	"spark_k31_dspark_block_trace",
};

static SparkStatus ForwardTimed(DsparkProbe *probe, uint64_t sequence, uint32_t anchor, uint64_t position, uint64_t *elapsed)
{
	return ((BlockRows)probe->fn_b)(probe->state, sequence, anchor, position, 0, elapsed);
}

static int WriteTrace(const char *directory, uint32_t layer, const float *queries, const float *latent, const uint16_t *block)
{
	char path[600];
	FILE *file;
	int failed;
	snprintf(path, sizeof(path), "%s/trace%u.bin", directory, layer);
	file = fopen(path, "wb");
	if ( file == 0 )
		return 1;
	failed = fwrite(queries, sizeof(*queries), (size_t)PROBE_QUERIES * PROBE_LATENT, file) != (size_t)PROBE_QUERIES * PROBE_LATENT ||
		fwrite(latent, sizeof(*latent), (size_t)PROBE_QUERIES * PROBE_KV_RANK, file) != (size_t)PROBE_QUERIES * PROBE_KV_RANK ||
		fwrite(block, sizeof(*block), (size_t)PROBE_BLOCK * PROBE_LATENT, file) != (size_t)PROBE_BLOCK * PROBE_LATENT;
	return fclose(file) != 0 || failed ? 1 : 0;
}

static int Trace(DsparkProbe *probe, const char *directory, uint32_t anchor, uint64_t position)
{
	float *queries = (float *)malloc((size_t)PROBE_QUERIES * PROBE_LATENT * sizeof(float));
	float *latent = (float *)malloc((size_t)PROBE_QUERIES * PROBE_KV_RANK * sizeof(float));
	uint16_t block[PROBE_BLOCK * PROBE_LATENT];
	int failed = queries == 0 || latent == 0;
	for ( uint32_t layer = 0u; layer < PROBE_LAYERS && failed == 0; layer++ )
		failed = ((BlockTrace)probe->fn_c)(probe->state, DSPARK_PROBE_SEQUENCE, anchor, position, layer + 1u, queries, latent, block) != SPARK_STATUS_OK ||
			WriteTrace(directory, layer, queries, latent, block) != 0;
	free(queries);
	free(latent);
	if ( failed )
		fprintf(stderr, "block trace failed\n");
	return failed;
}

static int Compare(DsparkProbe *probe, const char *directory)
{
	const uint64_t row_elements = (uint64_t)PROBE_TAPS * PROBE_HIDDEN;
	uint64_t tap_bytes = 0u, token_bytes = 0u, elapsed = 0u, rows;
	uint16_t *taps = (uint16_t *)DsparkProbeReadAll(directory, "taps.bin", &tap_bytes), *out, y[PROBE_BLOCK * PROBE_HIDDEN];
	uint32_t *tokens = (uint32_t *)DsparkProbeReadAll(directory, "tokens.bin", &token_bytes);
	char name[64];
	if ( taps == 0 || tokens == 0 )
	{
		fprintf(stderr, "%s needs taps.bin and tokens.bin\n", directory);
		return 1;
	}
	rows = tap_bytes / (row_elements * sizeof(uint16_t));
	if ( rows * row_elements * sizeof(uint16_t) != tap_bytes || token_bytes != (rows + 1u) * sizeof(uint32_t) )
	{
		fprintf(stderr, "taps.bin holds %llu rows; tokens.bin must hold one more token\n", (unsigned long long)rows);
		return 1;
	}
	for ( uint64_t first = 0u; first < rows; first += DSPARK_PROBE_FRAME_ROWS )
	{
		const uint32_t part = rows - first < DSPARK_PROBE_FRAME_ROWS ? (uint32_t)(rows - first) : DSPARK_PROBE_FRAME_ROWS;
		if ( DsparkProbeIngest(probe, &probe_cfg, DSPARK_PROBE_SEQUENCE, 1u, taps + first * row_elements, tokens + first, first, part) != 0 )
			return 1;
	}
	if ( ((BlockRows)probe->fn_b)(probe->state, DSPARK_PROBE_SEQUENCE, tokens[rows], rows, y, &elapsed) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "block forward failed\n");
		return 1;
	}
	printf("block forward at position %llu anchor %u: %.3f ms\n", (unsigned long long)rows, tokens[rows], (double)elapsed / 1e6);
	if ( DsparkProbeWriteAll(directory, "y.bin", y, (uint64_t)PROBE_BLOCK * PROBE_HIDDEN * sizeof(uint16_t)) != 0 )
		return 1;
	out = (uint16_t *)malloc((size_t)(rows + 1u) * PROBE_LATENT * sizeof(uint16_t));
	for ( uint32_t layer = 0u; layer < PROBE_LAYERS && out != 0; layer++ )
	{
		if ( rows != 0u && ((ContextRows)probe->fn_a)(probe->state, DSPARK_PROBE_SEQUENCE, layer, 0u, (uint32_t)rows, out) != SPARK_STATUS_OK )
		{
			fprintf(stderr, "context rows of layer %u failed\n", layer);
			return 1;
		}
		snprintf(name, sizeof(name), "rows%u.bin", layer);
		if ( DsparkProbeWriteAll(directory, name, out, rows * PROBE_LATENT * sizeof(uint16_t)) != 0 )
			return 1;
	}
	if ( Trace(probe, directory, tokens[rows], rows) != 0 || DsparkProbeDrafts(probe, &probe_cfg, directory, tokens, rows) != 0 ||
		DsparkProbeDonors(probe, &probe_cfg, taps, tokens, 0u, rows) != 0 )
		return 1;
	printf("wrote y.bin, trace0..%u.bin and %llu context rows x %u layers to %s\n", PROBE_LAYERS - 1u, (unsigned long long)rows, PROBE_LAYERS, directory);
	free(out);
	free(taps);
	free(tokens);
	return 0;
}

int main(int argc, char **argv)
{
	return DsparkProbeMain(argc, argv, &probe_cfg, Compare, ForwardTimed, "block forward");
}
