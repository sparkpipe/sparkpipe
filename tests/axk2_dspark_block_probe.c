#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_drafter_module.h"

#define PROBE_HIDDEN 7168u
#define PROBE_TAPS 3u
#define PROBE_LAYERS 3u
#define PROBE_HEADS 64u
#define PROBE_HEAD_DIM 112u
#define PROBE_WINDOW 123u
#define PROBE_COLUMNS 128u
#define PROBE_KV (PROBE_HEADS * PROBE_COLUMNS * PROBE_HEAD_DIM)
#define PROBE_BLOCK 5u
#define PROBE_DRAFTS 5u
#define PROBE_TREE_CAP 32u
#define PROBE_FRAME_ROWS 64u
#define PROBE_SEQUENCE 42u
#define PROBE_TIMED 20u
#define PROBE_RULES 3u

typedef SparkStatus (*CacheRows)(void *, uint64_t, uint32_t, uint16_t *, uint16_t *, int64_t *);
typedef SparkStatus (*ForwardRows)(void *, uint64_t, uint32_t, uint64_t, uint16_t *, uint64_t *);
typedef SparkStatus (*LayerTrace)(void *, uint64_t, uint32_t, uint64_t, uint32_t, uint16_t *);

typedef struct Probe
{
	const SparkDrafterModuleInterface *module;
	CacheRows cache;
	ForwardRows forward;
	LayerTrace trace;
	void *state;
} Probe;

static const SparkRowSampling probe_rules[PROBE_RULES] = {
	{0u, 0.0f, 0.0f, 0u, 0u},
	{12345u, 1.0f, 0.95f, 0u, 0u},
	{777u, 1.0f / 0.6f, 1.0f, 20u, 0u},
};

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
	probe->cache = library != 0 ? (CacheRows)dlsym(library, "spark_axk2_dspark_cache") : 0;
	probe->forward = library != 0 ? (ForwardRows)dlsym(library, "spark_axk2_dspark_forward") : 0;
	probe->trace = library != 0 ? (LayerTrace)dlsym(library, "spark_axk2_dspark_trace") : 0;
	if ( probe->module == 0 || probe->cache == 0 || probe->forward == 0 || probe->trace == 0 )
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
	lane.model = "axk2dspark";
	lane.revision = "0b278334dfce";
	lane.pack_sha256 = digest;
	lane.weightd_socket = socket_path;
	lane.pack_bytes = (uint64_t)info.st_size;
	lane.context_bytes = UINT64_C(512) << 20;
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

static int IngestAs(Probe *probe, uint64_t sequence, uint64_t request_id, const uint16_t *taps, const uint32_t *tokens, uint64_t first, uint32_t rows)
{
	SparkTapStreamReaderFrame frame;
	SparkTapStreamHello hello;
	uint32_t required[PROBE_TAPS] = {0u, 1u, 2u};
	memset(&hello, 0, sizeof(hello));
	hello.hidden = PROBE_HIDDEN;
	memset(&frame, 0, sizeof(frame));
	frame.header.sequence_id = sequence;
	frame.header.control_generation = 1u;
	frame.header.request_id = request_id;
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

static int WriteAll(const char *directory, const char *name, const void *data, uint64_t bytes)
{
	char path[600];
	FILE *file;
	snprintf(path, sizeof(path), "%s/%s", directory, name);
	file = fopen(path, "wb");
	if ( file == 0 || fwrite(data, 1u, (size_t)bytes, file) != (size_t)bytes )
	{
		if ( file != 0 )
			fclose(file);
		return 1;
	}
	return fclose(file) == 0 ? 0 : 1;
}

static SparkStatus ProposeAs(Probe *probe, uint64_t sequence, const uint32_t *committed, uint64_t position, const SparkRowSampling *rule,
	uint32_t node_cap, SparkDraftBridgeNode *nodes, uint32_t *count)
{
	SparkDraftServerRequest request;
	memset(&request, 0, sizeof(request));
	request.speculator_mask = 2u;
	request.sequence_id = sequence;
	request.position = position;
	request.anchor_token_id = committed[position];
	request.committed_token_ids = committed;
	request.committed_token_count = (uint32_t)position + 1u;
	request.depth = PROBE_DRAFTS;
	request.max_depth = PROBE_DRAFTS;
	request.node_capacity = node_cap;
	request.sampling = *rule;
	*count = 0u;
	return probe->module->propose(probe->state, &request, nodes, node_cap, count);
}

static int Drafts(Probe *probe, const char *directory, const uint32_t *tokens, uint64_t rows)
{
	SparkDraftBridgeNode nodes[PROBE_DRAFTS + 1u];
	SparkDraftBridgeNode tree_nodes[PROBE_TREE_CAP];
	uint32_t out[PROBE_RULES][2u * PROBE_DRAFTS + 1u];
	uint32_t tree_count = 0u;
	memset(out, 0, sizeof(out));
	for ( uint32_t rule = 0u; rule < PROBE_RULES; rule++ )
	{
		uint32_t count = 0u;
		const SparkStatus status = ProposeAs(probe, PROBE_SEQUENCE, tokens, rows, &probe_rules[rule], PROBE_DRAFTS + 1u, nodes, &count);
		if ( status != SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS || count != PROBE_DRAFTS + 1u || nodes[0].token_id != tokens[rows] )
		{
			fprintf(stderr, "propose under rule %u answered status %d with %u nodes\n", rule, (int)status, count);
			return 1;
		}
		out[rule][0] = count - 1u;
		for ( uint32_t draft = 0u; draft < PROBE_DRAFTS; draft++ )
		{
			out[rule][1u + draft] = nodes[draft + 1u].token_id;
			memcpy(&out[rule][1u + PROBE_DRAFTS + draft], &nodes[draft + 1u].score, sizeof(float));
		}
		printf("rule %u drafts:", rule);
		for ( uint32_t draft = 0u; draft < PROBE_DRAFTS; draft++ )
			printf(" %u(%.3f)", nodes[draft + 1u].token_id, (double)nodes[draft + 1u].score);
		printf("\n");
	}
	if ( ProposeAs(probe, PROBE_SEQUENCE, tokens, rows, &probe_rules[0], PROBE_TREE_CAP, tree_nodes, &tree_count) !=
		SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS || tree_count < PROBE_DRAFTS + 1u || tree_count > PROBE_TREE_CAP )
	{
		fprintf(stderr, "tree propose failed: count=%u\n", tree_count);
		return 1;
	}
	printf("tree propose: %u nodes (trunk %u + %u branch nodes)\n", tree_count, PROBE_DRAFTS + 1u, tree_count - (PROBE_DRAFTS + 1u));
	return WriteAll(directory, "drafts.bin", out, sizeof(out));
}

static int Donors(Probe *probe, const uint16_t *taps, const uint32_t *tokens, uint64_t taps_first, uint64_t rows)
{
	const uint64_t row_elements = (uint64_t)PROBE_TAPS * PROBE_HIDDEN;
	const uint64_t window = rows > PROBE_WINDOW ? rows - PROBE_WINDOW : 0u, start = window > taps_first ? window : taps_first;
	const uint64_t half = start + (rows - start) / 2u;
	SparkDraftBridgeNode whole[PROBE_DRAFTS + 1u], borrowed[PROBE_DRAFTS + 1u], alone[PROBE_DRAFTS + 1u];
	uint32_t *stranger = (uint32_t *)malloc((size_t)(rows + 1u) * sizeof(uint32_t)), counts[3] = {0u, 0u, 0u};
	SparkStatus statuses[3];
	int failed, same_borrowed = 1, same_alone = 1;
	if ( stranger == 0 || rows - start < 2u )
	{
		free(stranger);
		return rows - start < 2u ? 0 : 1;
	}
	memcpy(stranger, tokens, (size_t)(rows + 1u) * sizeof(uint32_t));
	for ( uint64_t at = start; at < half; at++ )
		stranger[at] = (tokens[at] + 1u) % 160000u;
	failed = IngestAs(probe, PROBE_SEQUENCE + 1u, 2u, taps + (half - taps_first) * row_elements, tokens + half, half, (uint32_t)(rows - half)) != 0 ||
		IngestAs(probe, PROBE_SEQUENCE + 2u, 3u, taps + (half - taps_first) * row_elements, tokens + half, half, (uint32_t)(rows - half)) != 0;
	statuses[0] = ProposeAs(probe, PROBE_SEQUENCE, tokens, rows, &probe_rules[0], PROBE_DRAFTS + 1u, whole, &counts[0]);
	statuses[1] = ProposeAs(probe, PROBE_SEQUENCE + 1u, tokens, rows, &probe_rules[0], PROBE_DRAFTS + 1u, borrowed, &counts[1]);
	statuses[2] = ProposeAs(probe, PROBE_SEQUENCE + 2u, stranger, rows, &probe_rules[0], PROBE_DRAFTS + 1u, alone, &counts[2]);
	free(stranger);
	if ( failed != 0 || statuses[0] != SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS || statuses[1] != SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS ||
		statuses[2] != SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS )
	{
		fprintf(stderr, "donor check: a propose answered %d, %d, %d\n", (int)statuses[0], (int)statuses[1], (int)statuses[2]);
		return 1;
	}
	for ( uint32_t node = 1u; node <= PROBE_DRAFTS; node++ )
	{
		same_borrowed &= whole[node].token_id == borrowed[node].token_id ? 1 : 0;
		same_alone &= whole[node].token_id == alone[node].token_id ? 1 : 0;
	}
	printf("donor check: sequence holding %llu..%llu borrows %llu..%llu and drafts %s; mismatched sequence drafts %s\n",
		(unsigned long long)half, (unsigned long long)rows - 1u, (unsigned long long)start, (unsigned long long)half - 1u,
		same_borrowed ? "identical tokens" : "OTHER tokens", same_alone ? "identical tokens" : "other tokens");
	return same_borrowed ? 0 : 1;
}

static int Compare(Probe *probe, const char *directory)
{
	const uint64_t row_elements = (uint64_t)PROBE_TAPS * PROBE_HIDDEN;
	uint64_t tap_bytes = 0u, token_bytes = 0u, first_bytes = 0u, elapsed = 0u, rows, start;
	uint16_t *taps = (uint16_t *)ReadAll(directory, "taps.bin", &tap_bytes);
	uint32_t *tokens = (uint32_t *)ReadAll(directory, "tokens.bin", &token_bytes);
	uint64_t *first_position = (uint64_t *)ReadAll(directory, "first.bin", &first_bytes);
	uint16_t *keys = (uint16_t *)malloc((size_t)PROBE_KV * sizeof(uint16_t)), *values = (uint16_t *)malloc((size_t)PROBE_KV * sizeof(uint16_t));
	uint16_t *y = (uint16_t *)malloc((size_t)PROBE_BLOCK * PROBE_HIDDEN * sizeof(uint16_t));
	int64_t *positions = (int64_t *)malloc((size_t)PROBE_WINDOW * sizeof(int64_t));
	if ( taps == 0 || tokens == 0 || first_position == 0 || first_bytes != sizeof(uint64_t) || keys == 0 || values == 0 || y == 0 || positions == 0 )
	{
		fprintf(stderr, "%s needs first.bin, taps.bin and tokens.bin\n", directory);
		return 1;
	}
	start = *first_position;
	rows = start + tap_bytes / (row_elements * sizeof(uint16_t));
	if ( (rows - start) * row_elements * sizeof(uint16_t) != tap_bytes || token_bytes != (rows + 1u) * sizeof(uint32_t) || rows == start )
	{
		fprintf(stderr, "taps.bin holds %llu rows from %llu; tokens.bin must hold every position up to anchor\n",
			(unsigned long long)(rows - start), (unsigned long long)start);
		return 1;
	}
	for ( uint64_t first = start; first < rows; first += PROBE_FRAME_ROWS )
	{
		const uint32_t part = rows - first < PROBE_FRAME_ROWS ? (uint32_t)(rows - first) : PROBE_FRAME_ROWS;
		if ( IngestAs(probe, PROBE_SEQUENCE, 1u, taps + (first - start) * row_elements, tokens + first, first, part) != 0 )
			return 1;
	}
	for ( uint32_t layer = 0u; layer < PROBE_LAYERS; layer++ )
	{
		char name[64];
		if ( probe->cache(probe->state, PROBE_SEQUENCE, layer, keys, values, positions) != SPARK_STATUS_OK )
		{
			fprintf(stderr, "cache read of layer %u failed\n", layer);
			return 1;
		}
		snprintf(name, sizeof(name), "keys%u.bin", layer);
		if ( WriteAll(directory, name, keys, (uint64_t)PROBE_KV * sizeof(uint16_t)) != 0 )
			return 1;
		snprintf(name, sizeof(name), "values%u.bin", layer);
		if ( WriteAll(directory, name, values, (uint64_t)PROBE_KV * sizeof(uint16_t)) != 0 )
			return 1;
	}
	if ( WriteAll(directory, "cache_positions.bin", positions, (uint64_t)PROBE_WINDOW * sizeof(int64_t)) != 0 ||
		probe->forward(probe->state, PROBE_SEQUENCE, tokens[rows], rows, y, &elapsed) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "forward failed\n");
		return 1;
	}
	printf("forward at position %llu anchor %u: %.3f ms\n", (unsigned long long)rows, tokens[rows], (double)elapsed / 1e6);
	if ( WriteAll(directory, "y.bin", y, (uint64_t)PROBE_BLOCK * PROBE_HIDDEN * sizeof(uint16_t)) != 0 )
		return 1;
	for ( uint32_t layers = 1u; layers <= PROBE_LAYERS; layers++ )
	{
		char name[64];
		snprintf(name, sizeof(name), "hidden%u.bin", layers);
		if ( probe->trace(probe->state, PROBE_SEQUENCE, tokens[rows], rows, layers, y) != SPARK_STATUS_OK ||
			WriteAll(directory, name, y, (uint64_t)PROBE_BLOCK * PROBE_HIDDEN * sizeof(uint16_t)) != 0 )
		{
			fprintf(stderr, "layer trace after %u layers failed\n", layers);
			return 1;
		}
	}
	if ( Drafts(probe, directory, tokens, rows) != 0 || Donors(probe, taps, tokens, start, rows) != 0 )
		return 1;
	printf("wrote cache, hidden traces, y.bin and drafts.bin to %s\n", directory);
	free(keys);
	free(values);
	free(y);
	free(positions);
	free(taps);
	free(tokens);
	free(first_position);
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
	uint16_t *taps = (uint16_t *)malloc((size_t)PROBE_WINDOW * row_elements * sizeof(uint16_t));
	uint32_t value = 4242u, *committed = 0;
	uint64_t elapsed[PROBE_TIMED], sequence = PROBE_SEQUENCE;
	char *copy = strdup(list), *cursor = copy, *item;
	SparkDraftBridgeNode nodes[PROBE_TREE_CAP];
	if ( taps == 0 || copy == 0 )
		return 1;
	for ( uint64_t index = 0u; index < (uint64_t)PROBE_WINDOW * row_elements; index++ )
	{
		value = value * 1664525u + 1013904223u;
		taps[index] = Bf16(((float)(value >> 8) / 16777216.0f - 0.5f) * 3.0f);
	}
	while ( (item = strsep(&cursor, ",")) != 0 )
	{
		const uint64_t position = strtoull(item, 0, 10), first = position > PROBE_WINDOW ? position - PROBE_WINDOW : 0u;
		committed = (uint32_t *)realloc(committed, (size_t)(position + 1u) * sizeof(uint32_t));
		if ( committed == 0 || position == 0u )
			return 1;
		for ( uint64_t at = 0u; at <= position; at++ )
			committed[at] = 1000u + (uint32_t)(at % 50000u);
		sequence++;
		if ( IngestAs(probe, sequence, sequence, taps, committed + first, first, (uint32_t)(position - first)) != 0 )
			return 1;
		for ( uint32_t run = 0u; run < PROBE_TIMED; run++ )
			if ( probe->forward(probe->state, sequence, committed[position], position, 0, &elapsed[run]) != SPARK_STATUS_OK )
			{
				fprintf(stderr, "forward at %llu failed\n", (unsigned long long)position);
				return 1;
			}
		qsort(elapsed, PROBE_TIMED, sizeof(elapsed[0]), CompareU64);
		printf("context %8llu: backbone forward min %.3f ms median %.3f ms max %.3f ms (%u runs)\n", (unsigned long long)position,
			(double)elapsed[0] / 1e6, (double)elapsed[PROBE_TIMED / 2u] / 1e6, (double)elapsed[PROBE_TIMED - 1u] / 1e6, PROBE_TIMED);
		for ( uint32_t rule = 0u; rule < PROBE_RULES; rule += PROBE_RULES - 1u )
		{
			for ( uint32_t run = 0u; run < PROBE_TIMED; run++ )
			{
				const uint64_t started = SparkMonotonicNs();
				uint32_t count = 0u;
				if ( ProposeAs(probe, sequence, committed, position, &probe_rules[rule], PROBE_DRAFTS + 1u, nodes, &count) !=
					SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS )
				{
					fprintf(stderr, "propose at %llu failed\n", (unsigned long long)position);
					return 1;
				}
				elapsed[run] = SparkMonotonicNs() - started;
			}
			qsort(elapsed, PROBE_TIMED, sizeof(elapsed[0]), CompareU64);
			printf("context %8llu: propose (%s) min %.3f ms median %.3f ms max %.3f ms (%u runs)\n", (unsigned long long)position,
				rule == 0u ? "greedy" : "sampled", (double)elapsed[0] / 1e6, (double)elapsed[PROBE_TIMED / 2u] / 1e6,
				(double)elapsed[PROBE_TIMED - 1u] / 1e6, PROBE_TIMED);
		}
		for ( uint32_t run = 0u; run < PROBE_TIMED; run++ )
		{
			const uint64_t started = SparkMonotonicNs();
			uint32_t count = 0u;
			if ( ProposeAs(probe, sequence, committed, position, &probe_rules[0], PROBE_TREE_CAP, nodes, &count) !=
				SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS )
			{
				fprintf(stderr, "tree propose at %llu failed\n", (unsigned long long)position);
				return 1;
			}
			elapsed[run] = SparkMonotonicNs() - started;
		}
		qsort(elapsed, PROBE_TIMED, sizeof(elapsed[0]), CompareU64);
		printf("context %8llu: propose (tree)   min %.3f ms median %.3f ms max %.3f ms (%u runs)\n", (unsigned long long)position,
			(double)elapsed[0] / 1e6, (double)elapsed[PROBE_TIMED / 2u] / 1e6, (double)elapsed[PROBE_TIMED - 1u] / 1e6, PROBE_TIMED);
	}
	free(committed);
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
		fprintf(stderr, "usage: axk2_dspark_block_probe MODULE PACK WEIGHTD_SOCKET DIR\n"
			"       axk2_dspark_block_probe MODULE PACK WEIGHTD_SOCKET --time POSITION[,POSITION...]\n");
		return 2;
	}
	if ( Open(&probe, argv[1], argv[2], argv[3]) != 0 )
		return 1;
	status = argc == 6 ? Time(&probe, argv[5]) : Compare(&probe, argv[4]);
	return probe.module->close(probe.state) == SPARK_STATUS_OK ? status : 1;
}
