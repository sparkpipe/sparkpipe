#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_drafter_module.h"
#include "sparkpipe/spark_tap_stream_reader.h"

#define REPLAY_SEQUENCES 256u
#define REPLAY_NODES 64u
#define REPLAY_MAX_ROWS 4096u
#define REPLAY_IO_TIMEOUT_MS 2000u
#define REPLAY_ANSWERING_FLAGS (2u | 4u | 8u)

typedef struct ReplaySequence
{
	uint64_t id;
	uint64_t control_generation;
	uint64_t request_id;
	uint64_t capacity;
	uint64_t last_used;
	uint32_t *tokens;
} ReplaySequence;

typedef struct ReplayOptions
{
	const char *module;
	const char *pack;
	const char *model;
	const char *revision;
	const char *weightd_socket;
	const char *recording;
	const char *entries;
	const char *out;
	uint64_t context_bytes;
	uint32_t depth;
	uint32_t limit;
	SparkRowSampling rule;
} ReplayOptions;

static ReplaySequence replay_sequences[REPLAY_SEQUENCES];
static uint64_t replay_clock;

static ReplaySequence *ReplayFind(uint64_t id, uint64_t control_generation, uint64_t request_id, uint64_t first)
{
	ReplaySequence *found = 0, *empty = 0, *oldest = 0;
	for ( uint32_t index = 0u; index < REPLAY_SEQUENCES && found == 0; index++ )
	{
		ReplaySequence *sequence = &replay_sequences[index];
		if ( sequence->tokens != 0 && sequence->id == id )
			found = sequence;
		else if ( sequence->tokens == 0 && empty == 0 )
			empty = sequence;
		else if ( sequence->tokens != 0 && (oldest == 0 || sequence->last_used < oldest->last_used) )
			oldest = sequence;
	}
	if ( found == 0 )
		found = empty != 0 ? empty : oldest;
	if ( found->tokens == 0 || found->id != id || first == 0u || found->control_generation != control_generation || found->request_id != request_id )
	{
		free(found->tokens);
		memset(found, 0, sizeof(*found));
		found->id = id;
		found->control_generation = control_generation;
		found->request_id = request_id;
	}
	found->last_used = ++replay_clock;
	return found;
}

static int ReplayGrow(ReplaySequence *sequence, uint64_t end)
{
	uint64_t capacity = sequence->capacity != 0u ? sequence->capacity : 4096u;
	uint32_t *tokens;
	if ( end <= sequence->capacity )
		return 0;
	while ( capacity < end )
		capacity *= 2u;
	tokens = (uint32_t *)realloc(sequence->tokens, (size_t)capacity * sizeof(uint32_t));
	if ( tokens == 0 )
		return 1;
	memset(tokens + sequence->capacity, 0, (size_t)(capacity - sequence->capacity) * sizeof(uint32_t));
	sequence->tokens = tokens;
	sequence->capacity = capacity;
	return 0;
}

static int ReplayOpen(const ReplayOptions *options, const SparkDrafterModuleInterface **module_out, void **state_out)
{
	SparkDrafterModuleLane lane;
	SparkDrafterModuleBind bind;
	char digest[80], path[600];
	struct stat info;
	FILE *file;
	void *library = dlopen(options->module, RTLD_NOW | RTLD_LOCAL);
	SparkDrafterModuleGetInterface get = library != 0 ? (SparkDrafterModuleGetInterface)dlsym(library, SPARK_DRAFTER_MODULE_INTERFACE_SYMBOL) : 0;
	const SparkDrafterModuleInterface *module = get != 0 ? get() : 0;
	if ( module == 0 || module->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION )
	{
		fprintf(stderr, "drafter_module_replay: %s is not a drafter module of ABI %u: %s\n", options->module, SPARK_DRAFTER_MODULE_ABI_VERSION,
			library == 0 ? dlerror() : "wrong interface");
		return 1;
	}
	snprintf(path, sizeof(path), "%s.sha256", options->pack);
	file = fopen(path, "r");
	if ( file == 0 || fscanf(file, "%64s", digest) != 1 || stat(options->pack, &info) != 0 )
	{
		fprintf(stderr, "drafter_module_replay: cannot read %s or stat %s\n", path, options->pack);
		if ( file != 0 )
			fclose(file);
		return 1;
	}
	fclose(file);
	memset(&lane, 0, sizeof(lane));
	memset(&bind, 0, sizeof(bind));
	lane.abi_version = SPARK_DRAFTER_MODULE_ABI_VERSION;
	lane.descriptor_bytes = sizeof(lane);
	lane.lane_name = "replay";
	lane.pack_path = options->pack;
	lane.model = options->model;
	lane.revision = options->revision;
	lane.pack_sha256 = digest;
	lane.weightd_socket = options->weightd_socket;
	lane.pack_bytes = (uint64_t)info.st_size;
	lane.context_bytes = options->context_bytes;
	lane.timeout_ns = UINT64_C(900000000000);
	if ( module->open(&lane, state_out, &bind) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "drafter_module_replay: open refused: %s\n", bind.text);
		return 1;
	}
	printf("bound: %s device_bytes=%llu bind_s=%.3f first_hold=%u\n", bind.text, (unsigned long long)bind.device_bytes, (double)bind.bind_ns / 1e9,
		bind.first_hold);
	*module_out = module;
	return 0;
}

static void ReplayWrite(FILE *out, const SparkTapStreamReaderFrame *frame, SparkStatus status, const SparkDraftBridgeNode *nodes, uint32_t count,
	uint64_t elapsed_ns)
{
	const char *answer = status == SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS ? "drafts" : status == SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS ? "await" :
		status == SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT ? "no-context" : "error";
	fprintf(out, "draft sequence=%llu control=%llu request=%llu position=%llu anchor=%u answer=%s elapsed_us=%llu drafts=",
		(unsigned long long)frame->header.sequence_id, (unsigned long long)frame->header.control_generation,
		(unsigned long long)frame->header.request_id, (unsigned long long)frame->header.first_position, frame->tokens[0], answer,
		(unsigned long long)(elapsed_ns / 1000u));
	for ( uint32_t node = 1u; node < count; node++ )
		fprintf(out, "%s%u:%.6f", node > 1u ? "," : "", nodes[node].token_id, (double)nodes[node].score);
	fprintf(out, "\n");
}

static int ReplayRun(const ReplayOptions *options)
{
	SparkTapStreamReaderConfig config;
	SparkTapStreamReader *reader = 0;
	SparkTapStreamReaderFrame frame;
	SparkDraftBridgeNode nodes[REPLAY_NODES];
	const SparkDrafterModuleInterface *module = 0;
	void *state = 0;
	uint64_t frames = 0u, requests = 0u, drafted = 0u, ingest_ns = 0u, propose_ns = 0u;
	SparkStatus advance;
	FILE *out;
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_TAP_STREAM_READER_ABI_VERSION;
	config.descriptor_bytes = SPARK_TAP_STREAM_READER_CONFIG_BYTES;
	config.path = options->recording;
	config.max_rows = REPLAY_MAX_ROWS;
	config.io_timeout_ms = REPLAY_IO_TIMEOUT_MS;
	if ( SparkTapStreamReaderParseEntries(options->entries, config.required, SPARK_TAP_STREAM_MAX_ENTRIES, &config.required_count) != SPARK_STATUS_OK ||
		SparkTapStreamReaderCreate(&config, &reader) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "drafter_module_replay: cannot read %s with entries %s\n", options->recording, options->entries);
		return 1;
	}
	out = fopen(options->out, "w");
	if ( out == 0 || ReplayOpen(options, &module, &state) != 0 )
	{
		SparkTapStreamReaderDestroy(reader);
		if ( out != 0 )
			fclose(out);
		return 1;
	}
	while ( (advance = SparkTapStreamReaderAdvance(reader, &frame)) == SPARK_STATUS_OK && (options->limit == 0u || requests < options->limit) )
	{
		const uint64_t first = frame.header.first_position, end = first + frame.header.row_count;
		ReplaySequence *sequence = ReplayFind(frame.header.sequence_id, frame.header.control_generation, frame.header.request_id, first);
		uint64_t started;
		if ( ReplayGrow(sequence, end) != 0 )
			break;
		if ( first != 0u && (frame.header.flags & REPLAY_ANSWERING_FLAGS) != 0u )
		{
			SparkDraftServerRequest request;
			uint32_t count = 0u;
			SparkStatus status;
			sequence->tokens[first] = frame.tokens[0];
			memset(&request, 0, sizeof(request));
			request.speculator_mask = 2u;
			request.sequence_id = frame.header.sequence_id;
			request.position = first;
			request.anchor_token_id = frame.tokens[0];
			request.committed_token_ids = sequence->tokens;
			request.committed_token_count = (uint32_t)first + 1u;
			request.depth = options->depth;
			request.max_depth = options->depth;
			request.node_capacity = options->depth + 1u;
			request.sampling = options->rule;
			started = SparkMonotonicNs();
			status = module->propose(state, &request, nodes, options->depth + 1u, &count);
			propose_ns += SparkMonotonicNs() - started;
			ReplayWrite(out, &frame, status, nodes, status == SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS ? count : 0u, SparkMonotonicNs() - started);
			requests++;
			drafted += status == SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS && count > 1u ? 1u : 0u;
		}
		started = SparkMonotonicNs();
		if ( module->ingest(state, &frame) != SPARK_STATUS_OK )
			fprintf(stderr, "drafter_module_replay: the module refused frame %llu\n", (unsigned long long)frame.header.serial);
		ingest_ns += SparkMonotonicNs() - started;
		memcpy(sequence->tokens + first, frame.tokens, (size_t)frame.header.row_count * sizeof(uint32_t));
		frames++;
	}
	printf("replayed %llu frames and %llu requests (%llu drafted) from %s: ingest %.3f ms a frame, propose %.3f ms a request\n",
		(unsigned long long)frames, (unsigned long long)requests, (unsigned long long)drafted, options->recording,
		frames != 0u ? (double)ingest_ns / 1e6 / (double)frames : 0.0, requests != 0u ? (double)propose_ns / 1e6 / (double)requests : 0.0);
	fclose(out);
	SparkTapStreamReaderDestroy(reader);
	for ( uint32_t index = 0u; index < REPLAY_SEQUENCES; index++ )
		free(replay_sequences[index].tokens);
	return module->close(state) == SPARK_STATUS_OK && (advance == SPARK_STATUS_OK || advance == SPARK_STATUS_NOT_FOUND) ? 0 : 1;
}

int main(int argc, char **argv)
{
	ReplayOptions options;
	memset(&options, 0, sizeof(options));
	options.context_bytes = UINT64_C(8) << 30;
	options.depth = 7u;
	if ( argc < 9 )
	{
		fprintf(stderr, "usage: drafter_module_replay MODULE PACK MODEL REVISION WEIGHTD_SOCKET RECORDING ENTRIES OUT [--context-bytes N] [--depth N]\n"
			"       [--limit REQUESTS] [--rule SEED,INVERSE_TEMPERATURE,NUCLEUS,TOP_K]\n");
		return 2;
	}
	options.module = argv[1];
	options.pack = argv[2];
	options.model = argv[3];
	options.revision = argv[4];
	options.weightd_socket = argv[5];
	options.recording = argv[6];
	options.entries = argv[7];
	options.out = argv[8];
	for ( int index = 9; index + 1 < argc; index += 2 )
	{
		if ( strcmp(argv[index], "--context-bytes") == 0 )
			options.context_bytes = strtoull(argv[index + 1], 0, 10);
		else if ( strcmp(argv[index], "--depth") == 0 )
			options.depth = (uint32_t)strtoul(argv[index + 1], 0, 10);
		else if ( strcmp(argv[index], "--limit") == 0 )
			options.limit = (uint32_t)strtoul(argv[index + 1], 0, 10);
		else if ( strcmp(argv[index], "--rule") == 0 )
		{
			unsigned long long seed = 0u;
			unsigned top_k = 0u;
			if ( sscanf(argv[index + 1], "%llu,%f,%f,%u", &seed, &options.rule.inverse_temperature, &options.rule.nucleus, &top_k) != 4 )
				return 2;
			options.rule.seed = seed;
			options.rule.top_k = top_k;
		}
		else
			return 2;
	}
	if ( options.depth == 0u || options.depth + 1u > REPLAY_NODES )
		return 2;
	return ReplayRun(&options);
}
