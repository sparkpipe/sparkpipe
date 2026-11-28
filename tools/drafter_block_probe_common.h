#ifndef SPARKPIPE_TOOLS_DRAFTER_BLOCK_PROBE_COMMON_H
#define SPARKPIPE_TOOLS_DRAFTER_BLOCK_PROBE_COMMON_H

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_drafter_module.h"

#define DSPARK_PROBE_FRAME_ROWS 2048u
#define DSPARK_PROBE_SEQUENCE 42u
#define DSPARK_PROBE_TIMED 20u
#define DSPARK_PROBE_RULES 3u
#define DSPARK_PROBE_MAX_TAPS 8u
#define DSPARK_PROBE_MAX_DRAFTS 16u

typedef struct DsparkProbeConfig
{
	const char *tool_name;
	const char *model;
	const char *revision;
	uint64_t context_bytes;
	uint32_t hidden;
	uint32_t taps;
	uint32_t drafts;
	uint32_t window;
	const char *sym_a;
	const char *sym_b;
	const char *sym_c;
} DsparkProbeConfig;

typedef struct DsparkProbe
{
	const SparkDrafterModuleInterface *module;
	void *fn_a;
	void *fn_b;
	void *fn_c;
	void *state;
} DsparkProbe;

typedef SparkStatus (*DsparkProbeForwardFn)(DsparkProbe *, uint64_t, uint32_t, uint64_t, uint64_t *);

static const SparkRowSampling dspark_probe_rules[DSPARK_PROBE_RULES] = {
	{0u, 0.0f, 0.0f, 0u, 0u},
	{12345u, 1.0f, 0.95f, 0u, 0u},
	{777u, 1.0f / 0.6f, 1.0f, 20u, 0u},
};

static uint16_t DsparkProbeBf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static int DsparkProbeCompareU64(const void *a, const void *b)
{
	const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
	return x < y ? -1 : x > y ? 1 : 0;
}

static int DsparkProbeOpen(DsparkProbe *probe, const DsparkProbeConfig *cfg, const char *library_path, const char *pack, const char *socket_path)
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
	probe->fn_a = library != 0 ? dlsym(library, cfg->sym_a) : 0;
	probe->fn_b = library != 0 ? dlsym(library, cfg->sym_b) : 0;
	probe->fn_c = library != 0 ? dlsym(library, cfg->sym_c) : 0;
	if ( probe->module == 0 || probe->fn_a == 0 || probe->fn_b == 0 || probe->fn_c == 0 )
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
	lane.model = cfg->model;
	lane.revision = cfg->revision;
	lane.pack_sha256 = digest;
	lane.weightd_socket = socket_path;
	lane.pack_bytes = (uint64_t)info.st_size;
	lane.context_bytes = cfg->context_bytes;
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

static int DsparkProbeIngest(DsparkProbe *probe, const DsparkProbeConfig *cfg, uint64_t sequence, uint64_t request_id, const uint16_t *taps,
	const uint32_t *tokens, uint64_t first, uint32_t rows)
{
	SparkTapStreamReaderFrame frame;
	SparkTapStreamHello hello;
	uint32_t required[DSPARK_PROBE_MAX_TAPS];
	for ( uint32_t i = 0u; i < cfg->taps; i++ )
		required[i] = i;
	memset(&hello, 0, sizeof(hello));
	hello.hidden = cfg->hidden;
	memset(&frame, 0, sizeof(frame));
	frame.header.sequence_id = sequence;
	frame.header.control_generation = 1u;
	frame.header.request_id = request_id;
	frame.header.first_position = first;
	frame.header.row_count = rows;
	frame.hello = &hello;
	frame.tokens = tokens;
	frame.rows = taps;
	frame.row_elements = (uint64_t)cfg->taps * cfg->hidden;
	frame.required_index = required;
	frame.required_count = cfg->taps;
	if ( probe->module->ingest(probe->state, &frame) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "ingest of %u rows at %llu failed\n", rows, (unsigned long long)first);
		return 1;
	}
	return 0;
}

static void *DsparkProbeReadAll(const char *directory, const char *name, uint64_t *bytes)
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

static int DsparkProbeWriteAll(const char *directory, const char *name, const void *data, uint64_t bytes)
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

static SparkStatus DsparkProbePropose(DsparkProbe *probe, const DsparkProbeConfig *cfg, uint64_t sequence, const uint32_t *committed, uint64_t position,
	const SparkRowSampling *rule, SparkDraftBridgeNode *nodes, uint32_t *count)
{
	SparkDraftServerRequest request;
	memset(&request, 0, sizeof(request));
	request.speculator_mask = 2u;
	request.sequence_id = sequence;
	request.position = position;
	request.anchor_token_id = committed[position];
	request.committed_token_ids = committed;
	request.committed_token_count = (uint32_t)position + 1u;
	request.depth = cfg->drafts;
	request.max_depth = cfg->drafts;
	request.node_capacity = cfg->drafts + 1u;
	if ( rule != 0 )
		request.sampling = *rule;
	*count = 0u;
	return probe->module->propose(probe->state, &request, nodes, cfg->drafts + 1u, count);
}

static int DsparkProbeDrafts(DsparkProbe *probe, const DsparkProbeConfig *cfg, const char *directory, const uint32_t *tokens, uint64_t rows)
{
	SparkDraftBridgeNode nodes[DSPARK_PROBE_MAX_DRAFTS + 1u];
	const uint32_t stride = 2u * cfg->drafts + 1u;
	uint32_t out[DSPARK_PROBE_RULES * (2u * DSPARK_PROBE_MAX_DRAFTS + 1u)];
	memset(out, 0, sizeof(out));
	for ( uint32_t rule = 0u; rule < DSPARK_PROBE_RULES; rule++ )
	{
		uint32_t count = 0u;
		const SparkStatus status = DsparkProbePropose(probe, cfg, DSPARK_PROBE_SEQUENCE, tokens, rows, &dspark_probe_rules[rule], nodes, &count);
		if ( status != SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS || count != cfg->drafts + 1u || nodes[0].token_id != tokens[rows] )
		{
			fprintf(stderr, "propose under rule %u answered status %d with %u nodes\n", rule, (int)status, count);
			return 1;
		}
		out[rule * stride] = count - 1u;
		for ( uint32_t draft = 0u; draft < cfg->drafts; draft++ )
		{
			out[rule * stride + 1u + draft] = nodes[draft + 1u].token_id;
			memcpy(&out[rule * stride + 1u + cfg->drafts + draft], &nodes[draft + 1u].score, sizeof(float));
		}
		printf("rule %u drafts:", rule);
		for ( uint32_t draft = 0u; draft < cfg->drafts; draft++ )
			printf(" %u(%.3f)", nodes[draft + 1u].token_id, (double)nodes[draft + 1u].score);
		printf("\n");
	}
	return DsparkProbeWriteAll(directory, "drafts.bin", out, (uint64_t)DSPARK_PROBE_RULES * stride * sizeof(uint32_t));
}

static int DsparkProbeDonors(DsparkProbe *probe, const DsparkProbeConfig *cfg, const uint16_t *taps, const uint32_t *tokens, uint64_t taps_first, uint64_t rows)
{
	const uint64_t row_elements = (uint64_t)cfg->taps * cfg->hidden;
	const uint64_t window = cfg->window != 0u && rows > cfg->window ? rows - cfg->window : 0u;
	const uint64_t start = window > taps_first ? window : taps_first;
	const uint64_t half = start + (rows - start) / 2u;
	SparkDraftBridgeNode whole[DSPARK_PROBE_MAX_DRAFTS + 1u], borrowed[DSPARK_PROBE_MAX_DRAFTS + 1u], alone[DSPARK_PROBE_MAX_DRAFTS + 1u];
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
		stranger[at] = tokens[at] + 1u;
	failed = DsparkProbeIngest(probe, cfg, DSPARK_PROBE_SEQUENCE + 1u, 2u, taps + (half - taps_first) * row_elements, tokens + half, half, (uint32_t)(rows - half)) != 0 ||
		DsparkProbeIngest(probe, cfg, DSPARK_PROBE_SEQUENCE + 2u, 3u, taps + (half - taps_first) * row_elements, tokens + half, half, (uint32_t)(rows - half)) != 0;
	statuses[0] = DsparkProbePropose(probe, cfg, DSPARK_PROBE_SEQUENCE, tokens, rows, &dspark_probe_rules[0], whole, &counts[0]);
	statuses[1] = DsparkProbePropose(probe, cfg, DSPARK_PROBE_SEQUENCE + 1u, tokens, rows, &dspark_probe_rules[0], borrowed, &counts[1]);
	statuses[2] = DsparkProbePropose(probe, cfg, DSPARK_PROBE_SEQUENCE + 2u, stranger, rows, &dspark_probe_rules[0], alone, &counts[2]);
	free(stranger);
	if ( failed != 0 || statuses[0] != SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS || statuses[1] != SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS ||
		statuses[2] != SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS )
	{
		fprintf(stderr, "donor check: a propose answered %d, %d, %d\n", (int)statuses[0], (int)statuses[1], (int)statuses[2]);
		return 1;
	}
	for ( uint32_t node = 1u; node <= cfg->drafts; node++ )
	{
		same_borrowed &= whole[node].token_id == borrowed[node].token_id ? 1 : 0;
		same_alone &= whole[node].token_id == alone[node].token_id ? 1 : 0;
	}
	printf("donor check: a sequence holding only positions %llu..%llu borrows %llu..%llu from one that shares its tokens and drafts %s; "
		"one whose tokens there differ keeps its gap and drafts %s\n", (unsigned long long)half, (unsigned long long)rows - 1u,
		(unsigned long long)start, (unsigned long long)half - 1u, same_borrowed ? "the same tokens" : "OTHER tokens",
		same_alone ? "the same tokens" : "other tokens");
	return same_borrowed ? 0 : 1;
}

static int DsparkProbeTime(DsparkProbe *probe, const DsparkProbeConfig *cfg, const char *list, DsparkProbeForwardFn forward_fn, const char *forward_label)
{
	const uint64_t row_elements = (uint64_t)cfg->taps * cfg->hidden;
	const uint32_t tap_rows = cfg->window != 0u ? cfg->window : DSPARK_PROBE_FRAME_ROWS;
	uint16_t *taps = (uint16_t *)malloc((size_t)tap_rows * row_elements * sizeof(uint16_t));
	uint32_t value = 4242u, *committed = 0, tokens[DSPARK_PROBE_FRAME_ROWS];
	uint64_t elapsed[DSPARK_PROBE_TIMED], sequence = DSPARK_PROBE_SEQUENCE, held = 0u;
	char *copy = strdup(list), *cursor = copy, *item;
	SparkDraftBridgeNode nodes[DSPARK_PROBE_MAX_DRAFTS + 1u];
	if ( taps == 0 || copy == 0 )
		return 1;
	for ( uint64_t index = 0u; index < (uint64_t)tap_rows * row_elements; index++ )
	{
		value = value * 1664525u + 1013904223u;
		taps[index] = DsparkProbeBf16(((float)(value >> 8) / 16777216.0f - 0.5f) * 3.0f);
	}
	while ( (item = strsep(&cursor, ",")) != 0 )
	{
		const uint64_t position = strtoull(item, 0, 10);
		uint64_t active_sequence = DSPARK_PROBE_SEQUENCE;
		uint32_t anchor = 1234u;
		committed = (uint32_t *)realloc(committed, (size_t)(position + 1u) * sizeof(uint32_t));
		if ( committed == 0 || (cfg->window != 0u && position == 0u) )
			return 1;
		for ( uint64_t at = 0u; at <= position; at++ )
			committed[at] = 1000u + (uint32_t)(at % (cfg->window != 0u ? 50000u : DSPARK_PROBE_FRAME_ROWS));
		if ( cfg->window != 0u )
		{
			const uint64_t first = position > cfg->window ? position - cfg->window : 0u;
			sequence++;
			active_sequence = sequence;
			anchor = committed[position];
			if ( DsparkProbeIngest(probe, cfg, sequence, sequence, taps, committed + first, first, (uint32_t)(position - first)) != 0 )
				return 1;
		}
		else
		{
			while ( held < position )
			{
				const uint32_t part = position - held < DSPARK_PROBE_FRAME_ROWS ? (uint32_t)(position - held) : DSPARK_PROBE_FRAME_ROWS;
				for ( uint32_t index = 0u; index < part; index++ )
					tokens[index] = 1000u + (uint32_t)((held + index) % DSPARK_PROBE_FRAME_ROWS);
				if ( DsparkProbeIngest(probe, cfg, DSPARK_PROBE_SEQUENCE, 1u, taps, tokens, held, part) != 0 )
					return 1;
				held += part;
			}
		}
		for ( uint32_t run = 0u; run < DSPARK_PROBE_TIMED; run++ )
			if ( forward_fn(probe, active_sequence, anchor, position, &elapsed[run]) != SPARK_STATUS_OK )
			{
				fprintf(stderr, "%s at %llu failed\n", forward_label, (unsigned long long)position);
				return 1;
			}
		qsort(elapsed, DSPARK_PROBE_TIMED, sizeof(elapsed[0]), DsparkProbeCompareU64);
		printf("context %8llu: %s min %.3f ms median %.3f ms max %.3f ms (%u runs)\n", (unsigned long long)position, forward_label,
			(double)elapsed[0] / 1e6, (double)elapsed[DSPARK_PROBE_TIMED / 2u] / 1e6, (double)elapsed[DSPARK_PROBE_TIMED - 1u] / 1e6, DSPARK_PROBE_TIMED);
		for ( uint32_t rule = 0u; rule < DSPARK_PROBE_RULES; rule += DSPARK_PROBE_RULES - 1u )
		{
			for ( uint32_t run = 0u; run < DSPARK_PROBE_TIMED; run++ )
			{
				const uint64_t started = SparkMonotonicNs();
				uint32_t count = 0u;
				if ( DsparkProbePropose(probe, cfg, active_sequence, committed, position, &dspark_probe_rules[rule], nodes, &count) != SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS )
				{
					fprintf(stderr, "propose at %llu failed\n", (unsigned long long)position);
					return 1;
				}
				elapsed[run] = SparkMonotonicNs() - started;
			}
			qsort(elapsed, DSPARK_PROBE_TIMED, sizeof(elapsed[0]), DsparkProbeCompareU64);
			printf("context %8llu: propose (%s) min %.3f ms median %.3f ms max %.3f ms (%u runs)\n", (unsigned long long)position,
				rule == 0u ? "greedy" : "sampled", (double)elapsed[0] / 1e6, (double)elapsed[DSPARK_PROBE_TIMED / 2u] / 1e6,
				(double)elapsed[DSPARK_PROBE_TIMED - 1u] / 1e6, DSPARK_PROBE_TIMED);
		}
	}
	free(committed);
	free(copy);
	free(taps);
	return 0;
}

static int DsparkProbeMain(int argc, char **argv, const DsparkProbeConfig *cfg, int (*compare_fn)(DsparkProbe *, const char *),
	DsparkProbeForwardFn forward_fn, const char *forward_label)
{
	DsparkProbe probe;
	int status;
	if ( argc != 5 && !(argc == 6 && strcmp(argv[4], "--time") == 0) )
	{
		fprintf(stderr, "usage: %s MODULE PACK WEIGHTD_SOCKET DIR\n"
			"       %s MODULE PACK WEIGHTD_SOCKET --time POSITION[,POSITION...]\n", cfg->tool_name, cfg->tool_name);
		return 2;
	}
	if ( DsparkProbeOpen(&probe, cfg, argv[1], argv[2], argv[3]) != 0 )
		return 1;
	status = argc == 6 ? DsparkProbeTime(&probe, cfg, argv[5], forward_fn, forward_label) : compare_fn(&probe, argv[4]);
	return probe.module->close(probe.state) == SPARK_STATUS_OK ? status : 1;
}

#endif
