#include "drafter_block_probe_common.h"

#define PROBE_HIDDEN 5120u
#define PROBE_TAPS 3u
#define PROBE_STAGES 3u
#define PROBE_WINDOW 128u
#define PROBE_HEAD_DIM 512u
#define PROBE_HC 4u
#define PROBE_BLOCK 5u
#define PROBE_VOCAB 129280u
#define PROBE_RING ((uint64_t)PROBE_STAGES * PROBE_WINDOW * PROBE_HEAD_DIM)

typedef SparkStatus (*RingRows)(void *, uint64_t, uint16_t *, int64_t *);
typedef SparkStatus (*ForwardRows)(void *, uint64_t, uint32_t, uint64_t, uint16_t *, float *, uint64_t *);
typedef SparkStatus (*StageTrace)(void *, uint64_t, uint32_t, uint64_t, uint32_t, uint16_t *, float *);

static const DsparkProbeConfig probe_cfg = {
	"dsv41_dspark_block_probe",
	"dsv41dspark",
	"1017",
	UINT64_C(256) << 20,
	PROBE_HIDDEN,
	PROBE_TAPS,
	PROBE_BLOCK,
	PROBE_WINDOW,
	"spark_dsv41_dspark_ring",
	"spark_dsv41_dspark_forward",
	"spark_dsv41_dspark_trace",
};

static SparkStatus ForwardTimed(DsparkProbe *probe, uint64_t sequence, uint32_t anchor, uint64_t position, uint64_t *elapsed)
{
	return ((ForwardRows)probe->fn_b)(probe->state, sequence, anchor, position, 0, 0, elapsed);
}

static int Compare(DsparkProbe *probe, const char *directory)
{
	const uint64_t row_elements = (uint64_t)PROBE_TAPS * PROBE_HIDDEN;
	uint64_t tap_bytes = 0u, token_bytes = 0u, first_bytes = 0u, elapsed = 0u, rows, start;
	uint16_t *taps = (uint16_t *)DsparkProbeReadAll(directory, "taps.bin", &tap_bytes);
	uint32_t *tokens = (uint32_t *)DsparkProbeReadAll(directory, "tokens.bin", &token_bytes);
	uint64_t *first_position = (uint64_t *)DsparkProbeReadAll(directory, "first.bin", &first_bytes);
	uint16_t *ring = (uint16_t *)malloc((size_t)PROBE_RING * sizeof(uint16_t)), hidden[PROBE_BLOCK * PROBE_HIDDEN];
	float *logits = (float *)malloc((size_t)PROBE_BLOCK * PROBE_VOCAB * sizeof(float));
	int64_t positions[PROBE_WINDOW];
	if ( taps == 0 || tokens == 0 || first_position == 0 || first_bytes != sizeof(uint64_t) || ring == 0 || logits == 0 )
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
	if ( ((RingRows)probe->fn_a)(probe->state, DSPARK_PROBE_SEQUENCE, ring, positions) != SPARK_STATUS_OK ||
		((ForwardRows)probe->fn_b)(probe->state, DSPARK_PROBE_SEQUENCE, tokens[rows], rows, hidden, logits, &elapsed) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "ring read or forward failed\n");
		return 1;
	}
	printf("forward and head at position %llu anchor %u: %.3f ms\n", (unsigned long long)rows, tokens[rows], (double)elapsed / 1e6);
	if ( DsparkProbeWriteAll(directory, "ring.bin", ring, PROBE_RING * sizeof(uint16_t)) != 0 ||
		DsparkProbeWriteAll(directory, "ring_positions.bin", positions, sizeof(positions)) != 0 ||
		DsparkProbeWriteAll(directory, "hidden.bin", hidden, sizeof(hidden)) != 0 ||
		DsparkProbeWriteAll(directory, "logits.bin", logits, (uint64_t)PROBE_BLOCK * PROBE_VOCAB * sizeof(float)) != 0 )
		return 1;
	for ( uint32_t stages = 0u; stages <= PROBE_STAGES; stages++ )
	{
		uint16_t *streams = (uint16_t *)malloc((size_t)PROBE_BLOCK * PROBE_HC * PROBE_HIDDEN * sizeof(uint16_t));
		float pre[PROBE_BLOCK * PROBE_HC];
		char name[64];
		int failed = streams == 0 || ((StageTrace)probe->fn_c)(probe->state, DSPARK_PROBE_SEQUENCE, tokens[rows], rows, stages, streams, pre) != SPARK_STATUS_OK;
		snprintf(name, sizeof(name), "streams%u.bin", stages);
		failed = failed || DsparkProbeWriteAll(directory, name, streams, (uint64_t)PROBE_BLOCK * PROBE_HC * PROBE_HIDDEN * sizeof(uint16_t)) != 0;
		snprintf(name, sizeof(name), "pre%u.bin", stages);
		failed = failed || DsparkProbeWriteAll(directory, name, pre, sizeof(pre)) != 0;
		free(streams);
		if ( failed )
		{
			fprintf(stderr, "stage trace after %u stages failed\n", stages);
			return 1;
		}
	}
	if ( DsparkProbeDrafts(probe, &probe_cfg, directory, tokens, rows) != 0 || DsparkProbeDonors(probe, &probe_cfg, taps, tokens, start, rows) != 0 )
		return 1;
	printf("wrote ring.bin, ring_positions.bin, hidden.bin, logits.bin and drafts.bin to %s\n", directory);
	free(ring);
	free(logits);
	free(taps);
	free(tokens);
	free(first_position);
	return 0;
}

int main(int argc, char **argv)
{
	return DsparkProbeMain(argc, argv, &probe_cfg, Compare, ForwardTimed, "forward and head");
}
