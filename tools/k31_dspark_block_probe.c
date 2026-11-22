#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "sparkpipe/spark_drafter_module.h"

#define PROBE_HIDDEN 7168u
#define PROBE_TAPS 5u
#define PROBE_LAYERS 5u
#define PROBE_BLOCK 7u
#define PROBE_KV_RANK 512u
#define PROBE_LATENT (PROBE_KV_RANK + 64u)
#define PROBE_QUERIES (PROBE_BLOCK * 96u)
#define PROBE_FRAME_ROWS 2048u
#define PROBE_SEQUENCE 42u
#define PROBE_TIMED 20u

typedef SparkStatus (*ContextRows)(void *, uint64_t, uint32_t, uint64_t, uint32_t, uint16_t *);
typedef SparkStatus (*BlockRows)(void *, uint64_t, uint32_t, uint64_t, uint16_t *, uint64_t *);
typedef SparkStatus (*BlockTrace)(void *, uint64_t, uint32_t, uint64_t, uint32_t, float *, float *, uint16_t *);

typedef struct Probe
{
	const SparkDrafterModuleInterface *module;
	ContextRows context_rows;
	BlockRows block_rows;
	BlockTrace block_trace;
	void *state;
} Probe;

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static int Open(Probe *probe, const char *library_path, const char *pack, const char *socket_path)
{
	SparkDrafterModuleLane lane;
	SparkDrafterModuleBind bind;
	char digest[80], path[600];
	struct stat info;
	FILE *file;
	void *library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
	SparkDrafterModuleGetInterface get = library != 0 ? (SparkDrafterModuleGetInterface)dlsym(library, SPARK_DRAFTER_MODULE_INTERFACE_SYMBOL) : 0;
	memset(probe, 0, sizeof(*probe));
	probe->module = get != 0 ? get() : 0;
	probe->context_rows = library != 0 ? (ContextRows)dlsym(library, "spark_k31_dspark_context_rows") : 0;
	probe->block_rows = library != 0 ? (BlockRows)dlsym(library, "spark_k31_dspark_block_rows") : 0;
	probe->block_trace = library != 0 ? (BlockTrace)dlsym(library, "spark_k31_dspark_block_trace") : 0;
	if ( probe->module == 0 || probe->context_rows == 0 || probe->block_rows == 0 || probe->block_trace == 0 )
	{
		fprintf(stderr, "module load failed: %s\n", dlerror());
		return 1;
	}
	snprintf(path, sizeof(path), "%s.sha256", pack);
	file = fopen(path, "r");
	if ( file == 0 || fscanf(file, "%64s", digest) != 1 || stat(pack, &info) != 0 )
	{
		fprintf(stderr, "cannot read %s or stat %s\n", path, pack);
		return 1;
	}
	fclose(file);
	memset(&lane, 0, sizeof(lane));
	memset(&bind, 0, sizeof(bind));
	lane.abi_version = SPARK_DRAFTER_MODULE_ABI_VERSION;
	lane.descriptor_bytes = sizeof(lane);
	lane.lane_name = "probe";
	lane.pack_path = pack;
	lane.model = "k31dspark";
	lane.revision = "6a2ef20a05ef";
	lane.pack_sha256 = digest;
	lane.weightd_socket = socket_path;
	lane.pack_bytes = (uint64_t)info.st_size;
	lane.context_bytes = UINT64_C(8) << 30;
	lane.timeout_ns = UINT64_C(600000000000);
	if ( probe->module->open(&lane, &probe->state, &bind) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "open refused: %s\n", bind.text);
		return 1;
	}
	printf("bound: %s device_bytes=%llu bind_s=%.3f first_hold=%u\n", bind.text, (unsigned long long)bind.device_bytes, (double)bind.bind_ns / 1e9,
		bind.first_hold);
	return 0;
}

static int Ingest(Probe *probe, const uint16_t *taps, const uint32_t *tokens, uint64_t first, uint32_t rows)
{
	SparkTapStreamReaderFrame frame;
	SparkTapStreamHello hello;
	uint32_t required[PROBE_TAPS] = {0u, 1u, 2u, 3u, 4u};
	memset(&hello, 0, sizeof(hello));
	hello.hidden = PROBE_HIDDEN;
	memset(&frame, 0, sizeof(frame));
	frame.header.sequence_id = PROBE_SEQUENCE;
	frame.header.control_generation = 1u;
	frame.header.request_id = 1u;
	frame.header.first_position = first;
	frame.header.row_count = rows;
	frame.hello = &hello;
	frame.tokens = tokens;
	frame.rows = taps;
	frame.row_elements = (uint64_t)PROBE_TAPS * PROBE_HIDDEN;
	frame.required_index = required;
	frame.required_count = PROBE_TAPS;
	if ( probe->module->ingest(probe->state, &frame) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "ingest of %u rows at %llu failed\n", rows, (unsigned long long)first);
		return 1;
	}
	return 0;
}

static void *ReadAll(const char *directory, const char *name, uint64_t *bytes)
{
	char path[600];
	struct stat info;
	void *data;
	FILE *file;
	snprintf(path, sizeof(path), "%s/%s", directory, name);
	if ( stat(path, &info) != 0 || (file = fopen(path, "rb")) == 0 )
		return 0;
	data = malloc((size_t)info.st_size);
	if ( data != 0 && fread(data, 1u, (size_t)info.st_size, file) != (size_t)info.st_size )
	{
		free(data);
		data = 0;
	}
	fclose(file);
	*bytes = (uint64_t)info.st_size;
	return data;
}

static int WriteAll(const char *directory, const char *name, const uint16_t *data, uint64_t count)
{
	char path[600];
	FILE *file;
	snprintf(path, sizeof(path), "%s/%s", directory, name);
	file = fopen(path, "wb");
	if ( file == 0 || fwrite(data, sizeof(*data), (size_t)count, file) != (size_t)count )
	{
		if ( file != 0 )
			fclose(file);
		return 1;
	}
	return fclose(file) == 0 ? 0 : 1;
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

static int Trace(Probe *probe, const char *directory, uint32_t anchor, uint64_t position)
{
	float *queries = (float *)malloc((size_t)PROBE_QUERIES * PROBE_LATENT * sizeof(float));
	float *latent = (float *)malloc((size_t)PROBE_QUERIES * PROBE_KV_RANK * sizeof(float));
	uint16_t block[PROBE_BLOCK * PROBE_LATENT];
	int failed = queries == 0 || latent == 0;
	for ( uint32_t layer = 0u; layer < PROBE_LAYERS && failed == 0; layer++ )
		failed = probe->block_trace(probe->state, PROBE_SEQUENCE, anchor, position, layer + 1u, queries, latent, block) != SPARK_STATUS_OK ||
			WriteTrace(directory, layer, queries, latent, block) != 0;
	free(queries);
	free(latent);
	if ( failed )
		fprintf(stderr, "block trace failed\n");
	return failed;
}

static int Compare(Probe *probe, const char *directory)
{
	const uint64_t row_elements = (uint64_t)PROBE_TAPS * PROBE_HIDDEN;
	uint64_t tap_bytes = 0u, token_bytes = 0u, elapsed = 0u, rows;
	uint16_t *taps = (uint16_t *)ReadAll(directory, "taps.bin", &tap_bytes), *out, y[PROBE_BLOCK * PROBE_HIDDEN];
	uint32_t *tokens = (uint32_t *)ReadAll(directory, "tokens.bin", &token_bytes);
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
	for ( uint64_t first = 0u; first < rows; first += PROBE_FRAME_ROWS )
	{
		const uint32_t part = rows - first < PROBE_FRAME_ROWS ? (uint32_t)(rows - first) : PROBE_FRAME_ROWS;
		if ( Ingest(probe, taps + first * row_elements, tokens + first, first, part) != 0 )
			return 1;
	}
	if ( probe->block_rows(probe->state, PROBE_SEQUENCE, tokens[rows], rows, y, &elapsed) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "block forward failed\n");
		return 1;
	}
	printf("block forward at position %llu anchor %u: %.3f ms\n", (unsigned long long)rows, tokens[rows], (double)elapsed / 1e6);
	if ( WriteAll(directory, "y.bin", y, (uint64_t)PROBE_BLOCK * PROBE_HIDDEN) != 0 )
		return 1;
	out = (uint16_t *)malloc((size_t)(rows + 1u) * PROBE_LATENT * sizeof(uint16_t));
	for ( uint32_t layer = 0u; layer < PROBE_LAYERS && out != 0; layer++ )
	{
		if ( rows != 0u && probe->context_rows(probe->state, PROBE_SEQUENCE, layer, 0u, (uint32_t)rows, out) != SPARK_STATUS_OK )
		{
			fprintf(stderr, "context rows of layer %u failed\n", layer);
			return 1;
		}
		snprintf(name, sizeof(name), "rows%u.bin", layer);
		if ( WriteAll(directory, name, out, rows * PROBE_LATENT) != 0 )
			return 1;
	}
	if ( Trace(probe, directory, tokens[rows], rows) != 0 )
		return 1;
	printf("wrote y.bin, trace0..%u.bin and %llu context rows x %u layers to %s\n", PROBE_LAYERS - 1u, (unsigned long long)rows, PROBE_LAYERS, directory);
	free(out);
	free(taps);
	free(tokens);
	return 0;
}

static int CompareU64(const void *a, const void *b)
{
	const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
	return x < y ? -1 : x > y ? 1 : 0;
}

static int Time(Probe *probe, const char *list)
{
	const uint64_t row_elements = (uint64_t)PROBE_TAPS * PROBE_HIDDEN;
	uint16_t *taps = (uint16_t *)malloc((size_t)PROBE_FRAME_ROWS * row_elements * sizeof(uint16_t));
	uint32_t tokens[PROBE_FRAME_ROWS], value = 4242u;
	uint64_t held = 0u, elapsed[PROBE_TIMED];
	char *copy = strdup(list), *cursor = copy, *item;
	if ( taps == 0 || copy == 0 )
		return 1;
	for ( uint64_t index = 0u; index < (uint64_t)PROBE_FRAME_ROWS * row_elements; index++ )
	{
		value = value * 1664525u + 1013904223u;
		taps[index] = Bf16(((float)(value >> 8) / 16777216.0f - 0.5f) * 3.0f);
	}
	for ( uint32_t index = 0u; index < PROBE_FRAME_ROWS; index++ )
		tokens[index] = 1000u + index;
	while ( (item = strsep(&cursor, ",")) != 0 )
	{
		const uint64_t position = strtoull(item, 0, 10);
		while ( held < position )
		{
			const uint32_t part = position - held < PROBE_FRAME_ROWS ? (uint32_t)(position - held) : PROBE_FRAME_ROWS;
			if ( Ingest(probe, taps, tokens, held, part) != 0 )
				return 1;
			held += part;
		}
		for ( uint32_t run = 0u; run < PROBE_TIMED; run++ )
			if ( probe->block_rows(probe->state, PROBE_SEQUENCE, 1234u, position, 0, &elapsed[run]) != SPARK_STATUS_OK )
			{
				fprintf(stderr, "block forward at %llu failed\n", (unsigned long long)position);
				return 1;
			}
		qsort(elapsed, PROBE_TIMED, sizeof(elapsed[0]), CompareU64);
		printf("context %8llu: block forward min %.3f ms median %.3f ms max %.3f ms (%u runs)\n", (unsigned long long)position,
			(double)elapsed[0] / 1e6, (double)elapsed[PROBE_TIMED / 2u] / 1e6, (double)elapsed[PROBE_TIMED - 1u] / 1e6, PROBE_TIMED);
	}
	free(copy);
	free(taps);
	return 0;
}

int main(int argc, char **argv)
{
	Probe probe;
	int status;
	if ( argc != 5 && !(argc == 6 && strcmp(argv[4], "--time") == 0) )
	{
		fprintf(stderr, "usage: k31_dspark_block_probe MODULE PACK WEIGHTD_SOCKET DIR\n"
			"       k31_dspark_block_probe MODULE PACK WEIGHTD_SOCKET --time POSITION[,POSITION...]\n");
		return 2;
	}
	if ( Open(&probe, argv[1], argv[2], argv[3]) != 0 )
		return 1;
	status = argc == 6 ? Time(&probe, argv[5]) : Compare(&probe, argv[4]);
	return probe.module->close(probe.state) == SPARK_STATUS_OK ? status : 1;
}
