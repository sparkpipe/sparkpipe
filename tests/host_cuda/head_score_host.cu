#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include "tests/host_cuda/lm_host_cuda.cuh"
LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/kernels/head_score.cuh"
#include "include/sparkpipe/spark_score_dump.h"

static uint32_t seed_state;

static uint32_t NextRandom(void)
{
	seed_state = seed_state * 1664525u + 1013904223u;
	return(seed_state >> 8);
}

static uint16_t RandomBf16(float scale)
{
	float value = ((float)(NextRandom() & 0xffffu) / 32768.0f - 1.0f) * scale;
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16));
}

template<typename T> static T *Allocate(uint64_t count)
{
	T *pointer = 0;
#ifdef __CUDACC__
	if ( cudaMallocManaged(&pointer, count * sizeof(T) + 16u) != cudaSuccess )
		return(0);
	memset(pointer, 0, count * sizeof(T) + 16u);
#else
	pointer = (T *)calloc(count + 1u, sizeof(T));
#endif
	return(pointer);
}

static int WriteRaw(const char *directory, const char *name, const void *bytes, uint64_t count)
{
	char path[4096];
	FILE *file;
	snprintf(path, sizeof(path), "%s/%s", directory, name);
	file = fopen(path, "wb");
	if ( file == 0 || fwrite(bytes, 1u, count, file) != count )
		return(1);
	fclose(file);
	return(0);
}

static void ScoreShard(const uint16_t *hidden, const uint16_t *shard, float *logits, uint32_t rows, uint32_t dimension, uint32_t width, uint32_t id_base, const uint32_t *offsets, const uint32_t *local, float *probe_logits, SparkScoreDumpStats *stats)
{
	dim3 logit_grid(LmHeadScoreLogitBlocks(width), LmHeadScoreRowPasses(rows));
	dim3 row_grid(rows);
#ifdef __CUDACC__
	LmHeadScoreLogitsKernel<LM_HEAD_SCORE_ROWS_PER_PASS><<<logit_grid, LM_HEAD_SCORE_THREADS>>>(hidden, shard, logits, rows, dimension, width);
	LmHeadScoreRowsKernel<LM_HEAD_SCORE_THREADS><<<row_grid, LM_HEAD_SCORE_THREADS>>>(logits, rows, width, id_base, offsets, local, probe_logits, stats);
	if ( cudaDeviceSynchronize() != cudaSuccess )
	{
		fprintf(stderr, "kernel failure\n");
		exit(3);
	}
#else
	logit_grid = dim3(width, LmHeadScoreRowPasses(rows));
	LM_HOST_LAUNCH(logit_grid, (LmHeadScoreLogitsKernel<LM_HEAD_SCORE_ROWS_PER_PASS>(hidden, shard, logits, rows, dimension, width)));
	LM_HOST_LAUNCH(row_grid, (LmHeadScoreRowsKernel<LM_HEAD_SCORE_THREADS>(logits, rows, width, id_base, offsets, local, probe_logits, stats)));
#endif
}

int main(int argc, char **argv)
{
	SparkScoreDumpConfig config;
	SparkScoreDumpWriter writer;
	SparkScoreDumpRowRecord record;
	const char *directory;
	const char *probe_path;
	const char *tier2_path;
	uint32_t tp, rows, dimension, width, vocabulary, document_rows, rank, row, index, count, owned, id_count;
	uint16_t *hidden, *head;
	uint32_t *tokens, *offsets, *local, *owned_ids;
	float *logits, *probe_logits;
	SparkScoreDumpStats *stats;
	uint64_t *keys;
	uint32_t *valid;
	const uint32_t *ids;
	SparkStatus status;
	if ( argc != 10 )
	{
		fprintf(stderr, "usage: %s dir tp rows hidden width document_rows seed probe|- tier2|-\n", argv[0]);
		return(2);
	}
	directory = argv[1];
	tp = (uint32_t)strtoul(argv[2], 0, 10);
	rows = (uint32_t)strtoul(argv[3], 0, 10);
	dimension = (uint32_t)strtoul(argv[4], 0, 10);
	width = (uint32_t)strtoul(argv[5], 0, 10);
	document_rows = (uint32_t)strtoul(argv[6], 0, 10);
	seed_state = (uint32_t)strtoul(argv[7], 0, 10);
	probe_path = strcmp(argv[8], "-") == 0 ? 0 : argv[8];
	tier2_path = strcmp(argv[9], "-") == 0 ? 0 : argv[9];
	vocabulary = tp * width;
	hidden = Allocate<uint16_t>((uint64_t)rows * dimension);
	head = Allocate<uint16_t>((uint64_t)vocabulary * dimension);
	tokens = Allocate<uint32_t>(rows);
	logits = Allocate<float>((uint64_t)rows * width);
	stats = Allocate<SparkScoreDumpStats>(rows);
	offsets = Allocate<uint32_t>(rows + 1u);
	local = Allocate<uint32_t>((uint64_t)rows * 4096u);
	owned_ids = Allocate<uint32_t>((uint64_t)rows * 4096u);
	probe_logits = Allocate<float>((uint64_t)rows * 4096u);
	keys = (uint64_t *)calloc(rows, sizeof(uint64_t));
	valid = (uint32_t *)calloc(rows, sizeof(uint32_t));
	if ( hidden == 0 || head == 0 || tokens == 0 || logits == 0 || stats == 0 || offsets == 0 || local == 0 || owned_ids == 0 || probe_logits == 0 || keys == 0 || valid == 0 || document_rows == 0u )
		return(2);
	for (index = 0u; index < rows * dimension; index++)
		hidden[index] = RandomBf16(1.0f);
	for (index = 0u; index < vocabulary * dimension; index++)
		head[index] = RandomBf16(0.25f);
	for (index = 0u; index < dimension; index++)
	{
		head[(uint64_t)(width + 3u) * dimension + index] = head[(uint64_t)5u * dimension + index];
		head[(uint64_t)(vocabulary - 1u) * dimension + index] = head[(uint64_t)5u * dimension + index];
		head[(uint64_t)9u * dimension + index] = head[(uint64_t)7u * dimension + index];
	}
	for (row = 0u; row < rows; row++)
		tokens[row] = NextRandom() % vocabulary;
	memset(&config, 0, sizeof(config));
	config.directory = directory;
	config.probe_path = probe_path;
	config.tier2_path = tier2_path;
	config.tp_degree = tp;
	config.vocabulary = vocabulary;
	config.hidden_dimension = dimension;
	config.slot_count = (rows + document_rows - 1u) / document_rows;
	config.position_count = document_rows;
	for (rank = 0u; rank < tp; rank++)
	{
		config.tp_rank = rank;
		status = SparkScoreDumpOpen(&config, &writer);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr, "open rank %u status %d\n", rank, (int)status);
			return(4);
		}
		for (row = 0u; row < rows; row++)
			valid[row] = SparkScoreDumpKeysAdvance(&writer.keys, row / document_rows, row % document_rows, tokens[row], &keys[row]);
		count = 0u;
		for (row = 0u; row < rows; row++)
		{
			offsets[row] = count;
			id_count = 0u;
			ids = valid[row] != 0u ? SparkScoreDumpTableFind(&writer.probes, keys[row], row % document_rows, &id_count) : 0;
			for (index = 0u; ids != 0 && index < id_count; index++)
				if ( ids[index] >= writer.header.shard_begin && ids[index] < writer.header.shard_end )
				{
					owned_ids[count] = ids[index];
					local[count] = ids[index] - writer.header.shard_begin;
					count++;
				}
		}
		offsets[rows] = count;
		ScoreShard(hidden, head + (uint64_t)writer.header.shard_begin * dimension, logits, rows, dimension, width, writer.header.shard_begin, offsets, local, probe_logits, stats);
		for (row = 0u; row < rows; row++)
		{
			memset(&record, 0, sizeof(record));
			record.record_kind = SPARK_SCORE_DUMP_RECORD_ROW;
			record.key = keys[row];
			record.wave_ordinal = row / 8u;
			record.position = row % document_rows;
			record.row_in_wave = row % 8u;
			record.input_token = tokens[row];
			record.served_token = SPARK_SCORE_DUMP_NO_TOKEN;
			owned = offsets[row + 1u] - offsets[row];
			record.probe_count = owned;
			record.local_max = stats[row].local_max;
			record.local_sum_exp = stats[row].local_sum_exp;
			memcpy(record.top_ids, stats[row].top_ids, sizeof(record.top_ids));
			memcpy(record.top_logits, stats[row].top_logits, sizeof(record.top_logits));
			record.flags = stats[row].flags | (valid[row] != 0u ? SPARK_SCORE_DUMP_ROW_KEY_VALID : 0u);
			if ( valid[row] != 0u && SparkScoreDumpTableFind(&writer.probes, keys[row], record.position, 0) != 0 )
				record.flags |= SPARK_SCORE_DUMP_ROW_PROBED;
			if ( valid[row] != 0u && SparkScoreDumpTableFind(&writer.tier2_rows, keys[row], record.position, 0) != 0 )
			{
				record.flags |= SPARK_SCORE_DUMP_ROW_TIER2;
				if ( SparkScoreDumpWriteTier2(&writer, keys[row], record.position, logits + (uint64_t)row * width) != SPARK_STATUS_OK )
					return(5);
			}
			if ( SparkScoreDumpWriteRow(&writer, &record, owned_ids + offsets[row], probe_logits + offsets[row]) != SPARK_STATUS_OK )
				return(5);
		}
		SparkScoreDumpNoteWave(&writer, 0u);
		if ( SparkScoreDumpClose(&writer) != SPARK_STATUS_OK )
			return(6);
	}
	if ( WriteRaw(directory, "hidden.bf16", hidden, (uint64_t)rows * dimension * 2u) != 0 || WriteRaw(directory, "head.bf16", head, (uint64_t)vocabulary * dimension * 2u) != 0 || WriteRaw(directory, "tokens.u32", tokens, (uint64_t)rows * 4u) != 0 )
		return(7);
	printf("head_score_host tp=%u rows=%u hidden=%u vocabulary=%u ok\n", tp, rows, dimension, vocabulary);
	return(0);
}
