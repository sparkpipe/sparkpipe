#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_draft_server.h"
#include "sparkpipe/spark_drafter_module.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_tap_stream_reader.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#define DRAFTD_MAX_LANES 8u
#define DRAFTD_POLL_MS 100
#define DRAFTD_NS_PER_MS UINT64_C(1000000)
#define DRAFTD_REPORT_EVERY 64u
#define DRAFTD_SUMMARY_KEYS 4096u
#define DRAFTD_DEFAULT_MAX_NODES 256u
#define DRAFTD_DEFAULT_IO_TIMEOUT_MS 2000u
#define DRAFTD_DEFAULT_MAX_COMMITTED 1048576u
#define DRAFTD_DEFAULT_TAP_ROWS 2048u
#define DRAFTD_DEFAULT_STREAMS 256u
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#define DRAFTD_PATH_BYTES 512u
#define DRAFTD_CONTROL_LINE_BYTES 256u
#define DRAFTD_CONTROL_REPLY_BYTES 4096u
#define DRAFTD_CONTROL_IO_SECONDS 5
#define DRAFTD_CONTROL_WAIT_SECONDS 900
#define DRAFTD_BIND_TIMEOUT_NS UINT64_C(900000000000)

typedef struct DraftdStream
{
	uint64_t rule;
	uint32_t *tokens;
	uint32_t count;
} DraftdStream;

typedef struct DraftdTapSeen
{
	uint64_t sequence_id;
	uint64_t end;
} DraftdTapSeen;

typedef struct DraftdReplay
{
	float confidence;
	uint32_t capacity;
	uint32_t require_taps;
	uint32_t delay_ms;
	uint32_t count;
	DraftdStream *streams;
	uint32_t seen_count;
	DraftdTapSeen seen[DRAFTD_SUMMARY_KEYS];
	uint64_t drafted;
	uint64_t unknown;
	uint64_t unavailable;
} DraftdReplay;

typedef struct DraftdLane
{
	char name[64];
	SparkDraftServer *server;
	SparkTapStreamReader *taps;
	DraftdReplay replay;
	uint32_t answer_pending;
	uint64_t answer_due_ns;
	uint32_t answer_status;
	uint32_t answer_count;
	SparkDraftBridgeNode *answer_nodes;
	uint32_t max_nodes;
	uint64_t requests_reported;
	uint32_t engine;
	uint32_t active;
	char module_path[DRAFTD_PATH_BYTES];
	char pack_path[DRAFTD_PATH_BYTES];
	char model[64];
	char revision[128];
	uint64_t context_bytes;
	void *module_library;
	const SparkDrafterModuleInterface *module;
	void *module_state;
	SparkDrafterModuleBind bind;
} DraftdLane;

static char draftd_weightd_socket[DRAFTD_PATH_BYTES];

static volatile sig_atomic_t draftd_stop;

static void DraftdSignal(int signal_number)
{
	(void)signal_number;
	draftd_stop = 1;
}

static uint64_t DraftdRuleKey(const SparkRowSampling *rule)
{
	uint32_t inverse, nucleus;
	uint64_t key = rule->seed * UINT64_C(0x9e3779b97f4a7c15);
	memcpy(&inverse, &rule->inverse_temperature, sizeof(inverse));
	memcpy(&nucleus, &rule->nucleus, sizeof(nucleus));
	key ^= ((uint64_t)inverse << 32u) ^ nucleus;
	return key ^ ((uint64_t)rule->top_k << 17u);
}

static uint32_t DraftdStartsWith(const uint32_t *tokens, uint32_t count, const uint32_t *prefix, uint32_t prefix_count)
{
	return count >= prefix_count && memcmp(tokens, prefix, (size_t)prefix_count * sizeof(uint32_t)) == 0 ? 1u : 0u;
}

static const DraftdStream *DraftdContinuation(const DraftdReplay *replay, uint64_t rule, const uint32_t *committed, uint32_t count)
{
	const DraftdStream *best = 0;
	for ( uint32_t index = 0u; index < replay->count; ++index )
	{
		const DraftdStream *stream = &replay->streams[index];
		if ( stream->rule == rule && stream->count > count && (best == 0 || stream->count > best->count) &&
			DraftdStartsWith(stream->tokens, stream->count, committed, count) != 0u )
			best = stream;
	}
	return best;
}

static SparkStatus DraftdLearn(DraftdReplay *replay, uint64_t rule, const uint32_t *committed, uint32_t count)
{
	uint32_t *copy;
	for ( uint32_t index = 0u; index < replay->count; ++index )
	{
		DraftdStream *stream = &replay->streams[index];
		if ( stream->rule != rule )
			continue;
		if ( DraftdStartsWith(stream->tokens, stream->count, committed, count) != 0u )
			return SPARK_STATUS_OK;
		if ( DraftdStartsWith(committed, count, stream->tokens, stream->count) != 0u )
		{
			copy = (uint32_t *)realloc(stream->tokens, (size_t)count * sizeof(uint32_t));
			if ( copy == 0 )
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			memcpy(copy, committed, (size_t)count * sizeof(uint32_t));
			stream->tokens = copy;
			stream->count = count;
			return SPARK_STATUS_OK;
		}
	}
	copy = (uint32_t *)malloc((size_t)count * sizeof(uint32_t));
	if ( copy == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memcpy(copy, committed, (size_t)count * sizeof(uint32_t));
	if ( replay->count == replay->capacity )
	{
		free(replay->streams[0].tokens);
		memmove(replay->streams, replay->streams + 1u, (size_t)(replay->count - 1u) * sizeof(replay->streams[0]));
		replay->count--;
	}
	replay->streams[replay->count].rule = rule;
	replay->streams[replay->count].tokens = copy;
	replay->streams[replay->count].count = count;
	replay->count++;
	return SPARK_STATUS_OK;
}

static uint64_t DraftdTapEnd(const DraftdReplay *replay, uint64_t sequence_id)
{
	for ( uint32_t index = 0u; index < replay->seen_count; ++index )
		if ( replay->seen[index].sequence_id == sequence_id )
			return replay->seen[index].end;
	return 0u;
}

static void DraftdIngest(DraftdReplay *replay, const SparkTapStreamReaderFrame *frame)
{
	const uint64_t end = frame->header.first_position + frame->header.row_count;
	uint32_t index;
	for ( index = 0u; index < replay->seen_count; ++index )
		if ( replay->seen[index].sequence_id == frame->header.sequence_id )
			break;
	if ( index == replay->seen_count )
	{
		if ( replay->seen_count == DRAFTD_SUMMARY_KEYS )
		{
			memmove(replay->seen, replay->seen + 1u, (size_t)(DRAFTD_SUMMARY_KEYS - 1u) * sizeof(replay->seen[0]));
			replay->seen_count--;
			index = replay->seen_count;
		}
		replay->seen[index].sequence_id = frame->header.sequence_id;
		replay->seen[index].end = 0u;
		replay->seen_count++;
	}
	if ( frame->header.first_position == 0u || (frame->header.first_position <= replay->seen[index].end && end > replay->seen[index].end) )
		replay->seen[index].end = end;
}

static void DraftdPropose(DraftdLane *lane, const SparkDraftServerRequest *request)
{
	DraftdReplay *replay = &lane->replay;
	const uint64_t rule = DraftdRuleKey(&request->sampling);
	const uint32_t source_bit = request->speculator_mask & (~request->speculator_mask + 1u);
	const DraftdStream *following;
	uint32_t count = 1u, depth;
	lane->answer_status = SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK;
	if ( lane->engine != 0u )
	{
		lane->answer_count = 0u;
		if ( lane->active == 0u )
		{
			replay->unavailable++;
			lane->answer_status = SPARK_DRAFT_BRIDGE_SERVER_STATUS_UNAVAILABLE;
			return;
		}
		if ( lane->module->propose(lane->module_state, request, lane->answer_nodes, request->node_capacity < lane->max_nodes ?
			request->node_capacity : lane->max_nodes, &count) != SPARK_STATUS_OK )
		{
			lane->answer_status = SPARK_DRAFT_BRIDGE_SERVER_STATUS_INTERNAL;
			return;
		}
		if ( count > 1u )
			replay->drafted++;
		else
			replay->unknown++;
		lane->answer_count = count;
		return;
	}
	if ( replay->require_taps != 0u && DraftdTapEnd(replay, request->sequence_id) < request->position )
	{
		replay->unavailable++;
		lane->answer_status = SPARK_DRAFT_BRIDGE_SERVER_STATUS_UNAVAILABLE;
		lane->answer_count = 0u;
		return;
	}
	following = DraftdContinuation(replay, rule, request->committed_token_ids, request->committed_token_count);
	if ( DraftdLearn(replay, rule, request->committed_token_ids, request->committed_token_count) != SPARK_STATUS_OK )
	{
		lane->answer_status = SPARK_DRAFT_BRIDGE_SERVER_STATUS_INTERNAL;
		lane->answer_count = 0u;
		return;
	}
	lane->answer_nodes[0].token_id = request->anchor_token_id;
	lane->answer_nodes[0].parent_index = SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX;
	lane->answer_nodes[0].depth = 0u;
	lane->answer_nodes[0].source_bit = source_bit;
	lane->answer_nodes[0].score = 1.0f;
	depth = following != 0 ? following->count - request->committed_token_count : 0u;
	if ( depth > request->depth )
		depth = request->depth;
	if ( depth > request->node_capacity - 1u )
		depth = request->node_capacity - 1u;
	for ( uint32_t index = 0u; index < depth; ++index )
	{
		lane->answer_nodes[count].token_id = following->tokens[request->committed_token_count + index];
		lane->answer_nodes[count].parent_index = index;
		lane->answer_nodes[count].depth = index + 1u;
		lane->answer_nodes[count].source_bit = source_bit;
		lane->answer_nodes[count].score = replay->confidence;
		count++;
	}
	if ( count > 1u )
		replay->drafted++;
	else
		replay->unknown++;
	lane->answer_count = count;
}

static void DraftdReport(const DraftdLane *lane)
{
	SparkDraftServerStats server;
	SparkTapStreamReaderStats taps;
	SparkDraftServerGetStats(lane->server, &server);
	SparkTapStreamReaderGetStats(lane->taps, &taps);
	printf("DRAFTD-STATS lane=%s connections=%llu requests=%llu bad=%llu answered=%llu dropped=%llu drafted=%llu unknown=%llu unavailable=%llu "
		"tap_streams=%llu tap_refused=%llu tap_frames=%llu tap_rows=%llu\n", lane->name, (unsigned long long)server.connections,
		(unsigned long long)server.requests, (unsigned long long)server.bad_requests, (unsigned long long)server.answered,
		(unsigned long long)server.dropped, (unsigned long long)lane->replay.drafted, (unsigned long long)lane->replay.unknown,
		(unsigned long long)lane->replay.unavailable, (unsigned long long)taps.streams, (unsigned long long)taps.refused,
		(unsigned long long)taps.frames, (unsigned long long)taps.rows);
	fflush(stdout);
}

static SparkStatus DraftdCopyText(const SparkJsonDocument *document, int32_t object, const char *member, char *text, size_t bytes, uint32_t required)
{
	char *copy = 0;
	const int32_t token = SparkJsonFindObjectMember(document, object, member);
	if ( token < 0 )
	{
		if ( required != 0u )
		{
			fprintf(stderr, "sparkpipe_draftd: the lane needs \"%s\"\n", member);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		return SPARK_STATUS_OK;
	}
	if ( SparkJsonCopyString(document, token, &copy) != SPARK_STATUS_OK || strlen(copy) >= bytes )
	{
		free(copy);
		fprintf(stderr, "sparkpipe_draftd: \"%s\" must be a string shorter than %zu bytes\n", member, bytes);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	memcpy(text, copy, strlen(copy) + 1u);
	free(copy);
	return SPARK_STATUS_OK;
}

static SparkStatus DraftdUInt(const SparkJsonDocument *document, int32_t object, const char *member, uint32_t *value, uint32_t fallback)
{
	const int32_t token = SparkJsonFindObjectMember(document, object, member);
	*value = fallback;
	if ( token >= 0 && SparkJsonGetUInt32(document, token, value) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_draftd: \"%s\" must be an unsigned integer\n", member);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus DraftdLaneOpen(const SparkJsonDocument *document, int32_t object, DraftdLane *lane)
{
	static const char *const lane_members[] = {"name", "host", "draft_port", "target_model", "max_committed_tokens", "max_nodes", "io_timeout_ms",
		"taps", "drafter"};
	static const char *const tap_members[] = {"host", "port", "model_tag", "hidden", "entries", "max_rows"};
	static const char *const drafter_members[] = {"kind", "confidence", "streams", "require_taps", "delay_ms", "module", "pack", "model",
		"revision", "context_bytes"};
	SparkDraftServerConfig server;
	char host[128] = "127.0.0.1", kind[32] = "";
	const int32_t taps = SparkJsonFindObjectMember(document, object, "taps"), drafter = SparkJsonFindObjectMember(document, object, "drafter");
	int32_t unknown = -1;
	uint32_t port = 0u, require = 0u;
	memset(&server, 0, sizeof(server));
	if ( SparkJsonValidateObjectMembersKnown(document, object, lane_members, sizeof(lane_members) / sizeof(lane_members[0]), &unknown) != SPARK_STATUS_OK ||
		drafter < 0 || SparkJsonValidateObjectMembersKnown(document, drafter, drafter_members, sizeof(drafter_members) / sizeof(drafter_members[0]),
			&unknown) != SPARK_STATUS_OK ||
		(taps >= 0 && SparkJsonValidateObjectMembersKnown(document, taps, tap_members, sizeof(tap_members) / sizeof(tap_members[0]), &unknown) !=
			SPARK_STATUS_OK) )
	{
		fprintf(stderr, "sparkpipe_draftd: a lane has an unknown member or no \"drafter\"\n");
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	server.abi_version = SPARK_DRAFT_SERVER_ABI_VERSION;
	server.descriptor_bytes = SPARK_DRAFT_SERVER_CONFIG_BYTES;
	if ( DraftdCopyText(document, object, "name", lane->name, sizeof(lane->name), 1u) != SPARK_STATUS_OK ||
		DraftdCopyText(document, object, "host", host, sizeof(host), 0u) != SPARK_STATUS_OK ||
		DraftdCopyText(document, object, "target_model", server.target_model, sizeof(server.target_model), 1u) != SPARK_STATUS_OK ||
		DraftdUInt(document, object, "draft_port", &server.port, 0u) != SPARK_STATUS_OK ||
		DraftdUInt(document, object, "max_committed_tokens", &server.max_committed_tokens, DRAFTD_DEFAULT_MAX_COMMITTED) != SPARK_STATUS_OK ||
		DraftdUInt(document, object, "max_nodes", &server.max_nodes, DRAFTD_DEFAULT_MAX_NODES) != SPARK_STATUS_OK ||
		DraftdUInt(document, object, "io_timeout_ms", &server.io_timeout_ms, DRAFTD_DEFAULT_IO_TIMEOUT_MS) != SPARK_STATUS_OK ||
		DraftdCopyText(document, drafter, "kind", kind, sizeof(kind), 1u) != SPARK_STATUS_OK ||
		DraftdUInt(document, drafter, "streams", &lane->replay.capacity, DRAFTD_DEFAULT_STREAMS) != SPARK_STATUS_OK ||
		DraftdUInt(document, drafter, "require_taps", &require, 0u) != SPARK_STATUS_OK ||
		DraftdUInt(document, drafter, "delay_ms", &lane->replay.delay_ms, 0u) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	lane->replay.confidence = 0.9f;
	if ( SparkJsonFindObjectMember(document, drafter, "confidence") >= 0 &&
		(SparkJsonGetFloat(document, SparkJsonFindObjectMember(document, drafter, "confidence"), &lane->replay.confidence) != SPARK_STATUS_OK ||
			!(lane->replay.confidence >= 0.0f && lane->replay.confidence <= 1.0f)) )
	{
		fprintf(stderr, "sparkpipe_draftd: lane %s: \"confidence\" must be a number in [0, 1]\n", lane->name);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	lane->engine = strcmp(kind, "engine") == 0 ? 1u : 0u;
	if ( (lane->engine == 0u && strcmp(kind, "replay") != 0) || lane->replay.capacity == 0u || require > 1u )
	{
		fprintf(stderr, "sparkpipe_draftd: lane %s: the drafter kind must be \"replay\" or \"engine\", with streams of at least 1 and require_taps 0 or 1\n",
			lane->name);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( lane->engine != 0u )
	{
		const int32_t context = SparkJsonFindObjectMember(document, drafter, "context_bytes");
		if ( DraftdCopyText(document, drafter, "module", lane->module_path, sizeof(lane->module_path), 1u) != SPARK_STATUS_OK ||
			DraftdCopyText(document, drafter, "pack", lane->pack_path, sizeof(lane->pack_path), 1u) != SPARK_STATUS_OK ||
			DraftdCopyText(document, drafter, "model", lane->model, sizeof(lane->model), 1u) != SPARK_STATUS_OK ||
			DraftdCopyText(document, drafter, "revision", lane->revision, sizeof(lane->revision), 1u) != SPARK_STATUS_OK ||
			context < 0 || SparkJsonGetUInt64(document, context, &lane->context_bytes) != SPARK_STATUS_OK )
		{
			fprintf(stderr, "sparkpipe_draftd: lane %s: an engine drafter names its module, pack, model, revision and context_bytes\n", lane->name);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		if ( draftd_weightd_socket[0] == '\0' )
		{
			fprintf(stderr, "sparkpipe_draftd: lane %s: an engine drafter needs the config's \"weightd_socket\"\n", lane->name);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
	}
	lane->replay.require_taps = require;
	if ( require != 0u && taps < 0 )
	{
		fprintf(stderr, "sparkpipe_draftd: lane %s requires taps but has no \"taps\" stream\n", lane->name);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	server.host = host;
	lane->max_nodes = server.max_nodes;
	lane->replay.streams = (DraftdStream *)calloc(lane->replay.capacity, sizeof(DraftdStream));
	lane->answer_nodes = (SparkDraftBridgeNode *)calloc(server.max_nodes, sizeof(SparkDraftBridgeNode));
	if ( lane->replay.streams == 0 || lane->answer_nodes == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( SparkDraftServerCreate(&server, &lane->server) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_draftd: lane %s: the draft server could not start on %s:%u\n", lane->name, host, server.port);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	if ( taps >= 0 )
	{
		SparkTapStreamReaderConfig reader;
		char tap_host[128] = "0.0.0.0", entries[512] = "";
		memset(&reader, 0, sizeof(reader));
		reader.abi_version = SPARK_TAP_STREAM_READER_ABI_VERSION;
		reader.descriptor_bytes = SPARK_TAP_STREAM_READER_CONFIG_BYTES;
		reader.io_timeout_ms = server.io_timeout_ms;
		if ( DraftdCopyText(document, taps, "host", tap_host, sizeof(tap_host), 0u) != SPARK_STATUS_OK ||
			DraftdCopyText(document, taps, "model_tag", reader.model_tag, sizeof(reader.model_tag), 0u) != SPARK_STATUS_OK ||
			DraftdCopyText(document, taps, "entries", entries, sizeof(entries), 1u) != SPARK_STATUS_OK ||
			DraftdUInt(document, taps, "port", &port, 0u) != SPARK_STATUS_OK ||
			DraftdUInt(document, taps, "hidden", &reader.hidden, 0u) != SPARK_STATUS_OK ||
			DraftdUInt(document, taps, "max_rows", &reader.max_rows, DRAFTD_DEFAULT_TAP_ROWS) != SPARK_STATUS_OK ||
			SparkTapStreamReaderParseEntries(entries, reader.required, SPARK_TAP_STREAM_MAX_ENTRIES, &reader.required_count) != SPARK_STATUS_OK )
		{
			fprintf(stderr, "sparkpipe_draftd: lane %s: the \"taps\" object is not valid\n", lane->name);
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		reader.host = tap_host;
		reader.port = port;
		if ( SparkTapStreamReaderCreate(&reader, &lane->taps) != SPARK_STATUS_OK )
		{
			fprintf(stderr, "sparkpipe_draftd: lane %s: the tap reader could not start on %s:%u\n", lane->name, tap_host, port);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
	}
	printf("DRAFTD-READY lane=%s draft=%u taps=%u drafter=%s\n", lane->name, SparkDraftServerPort(lane->server), SparkTapStreamReaderPort(lane->taps),
		kind);
	fflush(stdout);
	return SPARK_STATUS_OK;
}

static void DraftdLaneClose(DraftdLane *lane)
{
	SparkDraftServerDestroy(lane->server);
	SparkTapStreamReaderDestroy(lane->taps);
	for ( uint32_t index = 0u; index < lane->replay.count; ++index )
		free(lane->replay.streams[index].tokens);
	free(lane->replay.streams);
	free(lane->answer_nodes);
	if ( lane->module_state != 0 )
		(void)lane->module->close(lane->module_state);
	lane->module_state = 0;
	lane->active = 0u;
	if ( lane->module_library != 0 )
		(void)dlclose(lane->module_library);
	lane->module_library = 0;
}

static void DraftdLaneStep(DraftdLane *lane)
{
	SparkTapStreamReaderFrame frame;
	SparkDraftServerRequest request;
	SparkDraftServerStats stats;
	while ( lane->taps != 0 && SparkTapStreamReaderAdvance(lane->taps, &frame) == SPARK_STATUS_OK )
	{
		if ( lane->engine == 0u )
			DraftdIngest(&lane->replay, &frame);
		else if ( lane->active != 0u && lane->module->ingest(lane->module_state, &frame) != SPARK_STATUS_OK )
			fprintf(stderr, "sparkpipe_draftd: lane %s: the drafter module refused a tap frame\n", lane->name);
	}
	if ( lane->answer_pending != 0u && SparkMonotonicNs() >= lane->answer_due_ns )
	{
		lane->answer_pending = 0u;
		(void)SparkDraftServerAnswer(lane->server, lane->answer_status, lane->answer_nodes, lane->answer_count, 0);
	}
	if ( lane->answer_pending == 0u && SparkDraftServerAdvance(lane->server, &request) == SPARK_STATUS_OK )
	{
		DraftdPropose(lane, &request);
		if ( lane->replay.delay_ms != 0u )
		{
			lane->answer_pending = 1u;
			lane->answer_due_ns = SparkMonotonicNs() + (uint64_t)lane->replay.delay_ms * DRAFTD_NS_PER_MS;
		}
		else
			(void)SparkDraftServerAnswer(lane->server, lane->answer_status, lane->answer_nodes, lane->answer_count, 0);
		(void)SparkDraftServerAdvance(lane->server, &request);
	}
	SparkDraftServerGetStats(lane->server, &stats);
	if ( stats.requests >= lane->requests_reported + DRAFTD_REPORT_EVERY )
	{
		lane->requests_reported = stats.requests;
		DraftdReport(lane);
	}
}

static SparkStatus DraftdEngineLoad(DraftdLane *lane, char *reply, size_t reply_bytes)
{
	SparkDrafterModuleGetInterface get_interface = 0;
	const SparkDrafterModuleInterface *module;
	const char *error;
	void *library;
	if ( lane->module_library != 0 )
		return SPARK_STATUS_OK;
	dlerror();
	library = dlopen(lane->module_path, RTLD_NOW | RTLD_LOCAL);
	if ( library == 0 )
	{
		error = dlerror();
		snprintf(reply, reply_bytes, "REFUSED lane=%s cannot load drafter module %s: %s\n", lane->name, lane->module_path,
			error != 0 ? error : "unknown loader error");
		SPARK_FAIL(SPARK_STATUS_DRIVER_LOAD_ERROR);
	}
	*(void **)(&get_interface) = dlsym(library, SPARK_DRAFTER_MODULE_INTERFACE_SYMBOL);
	module = get_interface != 0 ? get_interface() : 0;
	if ( module == 0 || module->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION || module->descriptor_bytes < sizeof(*module) ||
		module->open == 0 || module->ingest == 0 || module->propose == 0 || module->close == 0 )
	{
		snprintf(reply, reply_bytes, "REFUSED lane=%s drafter module %s speaks drafter ABI %u; sparkpipe_draftd speaks %u\n", lane->name,
			lane->module_path, module != 0 ? module->abi_version : 0u, SPARK_DRAFTER_MODULE_ABI_VERSION);
		(void)dlclose(library);
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	}
	lane->module_library = library;
	lane->module = module;
	return SPARK_STATUS_OK;
}

static SparkStatus DraftdPackDigest(const char *pack_path, char digest[72])
{
	char path[DRAFTD_PATH_BYTES + 8u];
	FILE *file;
	int scanned;
	snprintf(path, sizeof(path), "%s.sha256", pack_path);
	file = fopen(path, "r");
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	scanned = fscanf(file, "%64s", digest);
	(void)fclose(file);
	if ( scanned != 1 || strlen(digest) != 64u )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return SPARK_STATUS_OK;
}

static void DraftdEngineActivate(DraftdLane *lane, char *reply, size_t reply_bytes)
{
	SparkDrafterModuleLane descriptor;
	struct stat info;
	char digest[72];
	void *state = 0;
	SparkStatus status;
	if ( lane->active != 0u )
	{
		snprintf(reply, reply_bytes, "OK lane=%s already active device_bytes=%llu\n", lane->name, (unsigned long long)lane->bind.device_bytes);
		return;
	}
	if ( DraftdEngineLoad(lane, reply, reply_bytes) != SPARK_STATUS_OK )
		return;
	if ( stat(lane->pack_path, &info) != 0 || S_ISREG(info.st_mode) == 0 || info.st_size <= 0 ||
		DraftdPackDigest(lane->pack_path, digest) != SPARK_STATUS_OK )
	{
		snprintf(reply, reply_bytes, "REFUSED lane=%s pack %s or its .sha256 sidecar is missing or unreadable\n", lane->name, lane->pack_path);
		return;
	}
	memset(&descriptor, 0, sizeof(descriptor));
	descriptor.abi_version = SPARK_DRAFTER_MODULE_ABI_VERSION;
	descriptor.descriptor_bytes = (uint32_t)sizeof(descriptor);
	descriptor.lane_name = lane->name;
	descriptor.pack_path = lane->pack_path;
	descriptor.model = lane->model;
	descriptor.revision = lane->revision;
	descriptor.pack_sha256 = digest;
	descriptor.weightd_socket = draftd_weightd_socket;
	descriptor.pack_bytes = (uint64_t)info.st_size;
	descriptor.context_bytes = lane->context_bytes;
	descriptor.timeout_ns = DRAFTD_BIND_TIMEOUT_NS;
	memset(&lane->bind, 0, sizeof(lane->bind));
	status = lane->module->open(&descriptor, &state, &lane->bind);
	lane->bind.text[sizeof(lane->bind.text) - 1u] = '\0';
	if ( status != SPARK_STATUS_OK || state == 0 )
	{
		snprintf(reply, reply_bytes, "REFUSED lane=%s %s (%s)\n", lane->name, lane->bind.text[0] != '\0' ? lane->bind.text : "the drafter module did not bind",
			SparkStatusToString(status));
		fprintf(stderr, "sparkpipe_draftd: %s", reply);
		return;
	}
	lane->module_state = state;
	lane->active = 1u;
	snprintf(reply, reply_bytes, "OK lane=%s device_bytes=%llu bind_s=%.3f first_hold=%u device_free=%llu device_total=%llu\n", lane->name,
		(unsigned long long)lane->bind.device_bytes, (double)lane->bind.bind_ns / 1e9, lane->bind.first_hold,
		(unsigned long long)lane->bind.device_free_bytes, (unsigned long long)lane->bind.device_total_bytes);
}

static void DraftdEngineDeactivate(DraftdLane *lane, char *reply, size_t reply_bytes)
{
	SparkStatus status;
	if ( lane->active == 0u )
	{
		snprintf(reply, reply_bytes, "OK lane=%s already inactive\n", lane->name);
		return;
	}
	status = lane->module->close(lane->module_state);
	lane->module_state = 0;
	lane->active = 0u;
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(reply, reply_bytes, "REFUSED lane=%s the drafter module did not release cleanly (%s); the lane is inactive\n", lane->name,
			SparkStatusToString(status));
		fprintf(stderr, "sparkpipe_draftd: %s", reply);
		return;
	}
	snprintf(reply, reply_bytes, "OK lane=%s released device_bytes=%llu\n", lane->name, (unsigned long long)lane->bind.device_bytes);
	memset(&lane->bind, 0, sizeof(lane->bind));
}

static void DraftdControlCommand(DraftdLane *lanes, uint32_t lane_count, char *line, char *reply, size_t reply_bytes)
{
	char *command = strtok(line, " \t\r\n"), *name = strtok(0, " \t\r\n");
	DraftdLane *lane = 0;
	size_t used;
	if ( command != 0 && strcmp(command, "status") == 0 )
	{
		used = (size_t)snprintf(reply, reply_bytes, "OK lanes=%u\n", lane_count);
		for ( uint32_t index = 0u; index < lane_count && used < reply_bytes; ++index )
			used += (size_t)snprintf(reply + used, reply_bytes - used, "lane=%s drafter=%s active=%u device_bytes=%llu bind_s=%.3f unavailable=%llu\n",
				lanes[index].name, lanes[index].engine != 0u ? "engine" : "replay", lanes[index].engine != 0u ? lanes[index].active : 1u,
				(unsigned long long)lanes[index].bind.device_bytes, (double)lanes[index].bind.bind_ns / 1e9,
				(unsigned long long)lanes[index].replay.unavailable);
		return;
	}
	for ( uint32_t index = 0u; name != 0 && index < lane_count; ++index )
		if ( strcmp(lanes[index].name, name) == 0 )
			lane = &lanes[index];
	if ( command == 0 || (strcmp(command, "activate") != 0 && strcmp(command, "deactivate") != 0) )
	{
		snprintf(reply, reply_bytes, "REFUSED commands are status, activate LANE and deactivate LANE\n");
		return;
	}
	if ( lane == 0 || lane->engine == 0u )
	{
		snprintf(reply, reply_bytes, "REFUSED lane=%s is not an engine lane of this sparkpipe_draftd\n", name != 0 ? name : "");
		return;
	}
	if ( strcmp(command, "activate") == 0 )
		DraftdEngineActivate(lane, reply, reply_bytes);
	else
		DraftdEngineDeactivate(lane, reply, reply_bytes);
	printf("DRAFTD-CONTROL %s", reply);
	fflush(stdout);
}

static int DraftdControlListen(const char *path)
{
	struct sockaddr_un address;
	struct stat info;
	int fd;
	memset(&address, 0, sizeof(address));
	address.sun_family = AF_UNIX;
	if ( strlen(path) >= sizeof(address.sun_path) )
	{
		fprintf(stderr, "sparkpipe_draftd: control_socket %s is longer than a socket path\n", path);
		return -1;
	}
	memcpy(address.sun_path, path, strlen(path) + 1u);
	if ( lstat(path, &info) == 0 && S_ISSOCK(info.st_mode) )
		(void)unlink(path);
	fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if ( fd < 0 || fcntl(fd, F_SETFD, FD_CLOEXEC) != 0 || bind(fd, (const struct sockaddr *)&address, sizeof(address)) != 0 || listen(fd, 4) != 0 ||
		fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK) != 0 )
	{
		fprintf(stderr, "sparkpipe_draftd: the control socket %s does not listen: %s\n", path, strerror(errno));
		if ( fd >= 0 )
			(void)close(fd);
		return -1;
	}
	return fd;
}

static void DraftdControlTimeouts(int fd, int seconds)
{
	struct timeval timeout;
	timeout.tv_sec = seconds;
	timeout.tv_usec = 0;
	(void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
	(void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

static void DraftdControlServe(int listen_fd, DraftdLane *lanes, uint32_t lane_count)
{
	char line[DRAFTD_CONTROL_LINE_BYTES], reply[DRAFTD_CONTROL_REPLY_BYTES];
	size_t used = 0u;
	ssize_t got;
	const int fd = accept(listen_fd, 0, 0);
	if ( fd < 0 )
		return;
	(void)fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK);
	DraftdControlTimeouts(fd, DRAFTD_CONTROL_IO_SECONDS);
	while ( used + 1u < sizeof(line) && (got = read(fd, line + used, sizeof(line) - 1u - used)) > 0 )
	{
		used += (size_t)got;
		if ( memchr(line, '\n', used) != 0 )
			break;
	}
	line[used] = '\0';
	DraftdControlCommand(lanes, lane_count, line, reply, sizeof(reply));
	for ( size_t sent = 0u, length = strlen(reply); sent < length; )
	{
		got = send(fd, reply + sent, length - sent, MSG_NOSIGNAL);
		if ( got <= 0 )
			break;
		sent += (size_t)got;
	}
	(void)close(fd);
}

static int DraftdControlClient(const char *path, const char *command, const char *lane)
{
	struct sockaddr_un address;
	char request[DRAFTD_CONTROL_LINE_BYTES], reply[DRAFTD_CONTROL_REPLY_BYTES];
	size_t used = 0u;
	ssize_t got;
	int fd;
	memset(&address, 0, sizeof(address));
	address.sun_family = AF_UNIX;
	if ( strlen(path) >= sizeof(address.sun_path) ||
		(size_t)snprintf(request, sizeof(request), "%s%s%s\n", command, lane != 0 ? " " : "", lane != 0 ? lane : "") >= sizeof(request) )
	{
		fprintf(stderr, "sparkpipe_draftd: the control socket path or the command is too long\n");
		return 2;
	}
	memcpy(address.sun_path, path, strlen(path) + 1u);
	fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if ( fd < 0 || fcntl(fd, F_SETFD, FD_CLOEXEC) != 0 || connect(fd, (const struct sockaddr *)&address, sizeof(address)) != 0 )
	{
		fprintf(stderr, "sparkpipe_draftd: no sparkpipe_draftd answers on %s: %s\n", path, strerror(errno));
		if ( fd >= 0 )
			(void)close(fd);
		return 1;
	}
	DraftdControlTimeouts(fd, DRAFTD_CONTROL_WAIT_SECONDS);
	if ( send(fd, request, strlen(request), MSG_NOSIGNAL) != (ssize_t)strlen(request) )
	{
		fprintf(stderr, "sparkpipe_draftd: the control request did not send: %s\n", strerror(errno));
		(void)close(fd);
		return 1;
	}
	while ( used + 1u < sizeof(reply) && (got = read(fd, reply + used, sizeof(reply) - 1u - used)) > 0 )
		used += (size_t)got;
	(void)close(fd);
	reply[used] = '\0';
	fputs(reply, stdout);
	if ( used == 0u )
	{
		fprintf(stderr, "sparkpipe_draftd: no control reply from %s within %d s\n", path, DRAFTD_CONTROL_WAIT_SECONDS);
		return 1;
	}
	return strncmp(reply, "OK", 2u) == 0 ? 0 : 1;
}

static int DraftdWait(DraftdLane *lanes, uint32_t lane_count, int control_fd, uint32_t *control_ready)
{
	struct pollfd fds[DRAFTD_MAX_LANES * (SPARK_DRAFT_SERVER_POLL_FDS + SPARK_TAP_STREAM_READER_POLL_FDS) + 1u];
	uint32_t count = 0u;
	int polled;
	*control_ready = 0u;
	if ( control_fd >= 0 )
	{
		fds[count].fd = control_fd;
		fds[count].events = POLLIN;
		fds[count++].revents = 0;
	}
	uint64_t wake = SparkMonotonicNs() + (uint64_t)DRAFTD_POLL_MS * DRAFTD_NS_PER_MS, now;
	for ( uint32_t index = 0u; index < lane_count; ++index )
	{
		int lane_fds[SPARK_DRAFT_SERVER_POLL_FDS];
		uint32_t events[SPARK_DRAFT_SERVER_POLL_FDS];
		uint64_t deadline = 0u;
		uint32_t found = SparkDraftServerPollFds(lanes[index].server, lane_fds, events, &deadline);
		for ( uint32_t fd = 0u; fd < found; ++fd )
		{
			fds[count].fd = lane_fds[fd];
			fds[count].events = (short)((events[fd] & SPARK_DRAFT_BRIDGE_POLL_READ) != 0u ? POLLIN : POLLOUT);
			fds[count++].revents = 0;
		}
		if ( deadline != 0u && deadline < wake )
			wake = deadline;
		if ( lanes[index].answer_pending != 0u && lanes[index].answer_due_ns < wake )
			wake = lanes[index].answer_due_ns;
		deadline = 0u;
		found = SparkTapStreamReaderPollFds(lanes[index].taps, lane_fds, events, &deadline);
		for ( uint32_t fd = 0u; fd < found; ++fd )
		{
			fds[count].fd = lane_fds[fd];
			fds[count].events = POLLIN;
			fds[count++].revents = 0;
		}
		if ( deadline != 0u && deadline < wake )
			wake = deadline;
	}
	now = SparkMonotonicNs();
	polled = poll(fds, count, wake > now ? (int)((wake - now + DRAFTD_NS_PER_MS - 1u) / DRAFTD_NS_PER_MS) : 0);
	if ( polled > 0 && control_fd >= 0 && (fds[0].revents & POLLIN) != 0 )
		*control_ready = 1u;
	return polled;
}

static int DraftdServe(const char *path)
{
	SparkJsonDocument document;
	DraftdLane lanes[DRAFTD_MAX_LANES];
	char control_path[DRAFTD_PATH_BYTES] = "";
	uint32_t lane_count = 0u, control_ready = 0u, engines = 0u;
	int32_t root, array, element;
	int result = 0, control_fd = -1;
	memset(&document, 0, sizeof(document));
	memset(lanes, 0, sizeof(lanes));
	if ( SparkJsonLoadFile(path, &document) != SPARK_STATUS_OK || (root = SparkJsonGetRootToken(&document)) < 0 ||
		(array = SparkJsonFindObjectMember(&document, root, "lanes")) < 0 || SparkJsonGetArrayElementCount(&document, array) == 0u ||
		SparkJsonGetArrayElementCount(&document, array) > DRAFTD_MAX_LANES )
	{
		fprintf(stderr, "sparkpipe_draftd: %s must be a JSON object whose \"lanes\" array has 1 to %u lanes\n", path, DRAFTD_MAX_LANES);
		SparkJsonDocumentDestroy(&document);
		return 2;
	}
	if ( DraftdCopyText(&document, root, "weightd_socket", draftd_weightd_socket, sizeof(draftd_weightd_socket), 0u) != SPARK_STATUS_OK ||
		DraftdCopyText(&document, root, "control_socket", control_path, sizeof(control_path), 0u) != SPARK_STATUS_OK )
	{
		SparkJsonDocumentDestroy(&document);
		return 2;
	}
	for ( element = SparkJsonGetArrayElementFirst(&document, array); element >= 0 && result == 0;
		element = SparkJsonGetArrayElementNext(&document, array, element) )
		if ( DraftdLaneOpen(&document, element, &lanes[lane_count++]) != SPARK_STATUS_OK )
			result = 2;
	SparkJsonDocumentDestroy(&document);
	for ( uint32_t index = 0u; index < lane_count; ++index )
		engines += lanes[index].engine;
	if ( result == 0 && engines != 0u && control_path[0] == '\0' )
	{
		fprintf(stderr, "sparkpipe_draftd: engine lanes are activated through the config's \"control_socket\", which is missing\n");
		result = 2;
	}
	if ( result == 0 && control_path[0] != '\0' && (control_fd = DraftdControlListen(control_path)) < 0 )
		result = 2;
	if ( result == 0 )
	{
		printf("DRAFTD-READY lanes=%u engines=%u control=%s\n", lane_count, engines, control_path[0] != '\0' ? control_path : "-");
		fflush(stdout);
	}
	while ( result == 0 && draftd_stop == 0 )
	{
		if ( DraftdWait(lanes, lane_count, control_fd, &control_ready) < 0 && errno != EINTR )
		{
			fprintf(stderr, "sparkpipe_draftd: poll failed: %s\n", strerror(errno));
			result = 1;
		}
		if ( result == 0 && control_ready != 0u )
			DraftdControlServe(control_fd, lanes, lane_count);
		for ( uint32_t index = 0u; index < lane_count && result == 0; ++index )
			DraftdLaneStep(&lanes[index]);
	}
	for ( uint32_t index = 0u; index < lane_count; ++index )
	{
		if ( lanes[index].server != 0 )
			DraftdReport(&lanes[index]);
		DraftdLaneClose(&lanes[index]);
	}
	if ( control_fd >= 0 )
	{
		(void)close(control_fd);
		(void)unlink(control_path);
	}
	return result;
}

typedef struct DraftdSummaryKey
{
	uint64_t control_generation;
	uint64_t request_id;
	uint64_t request_generation;
	uint64_t sequence_id;
	uint64_t frames;
	uint64_t rows;
	uint64_t end;
	uint32_t ended;
	uint64_t gaps;
	uint64_t overlaps;
	uint64_t kinds[4];
} DraftdSummaryKey;

static int DraftdSummaryOrder(const void *left, const void *right)
{
	const DraftdSummaryKey *a = (const DraftdSummaryKey *)left, *b = (const DraftdSummaryKey *)right;
	const uint64_t ka[4] = {a->control_generation, a->request_id, a->request_generation, a->sequence_id};
	const uint64_t kb[4] = {b->control_generation, b->request_id, b->request_generation, b->sequence_id};
	for ( uint32_t index = 0u; index < 4u; ++index )
		if ( ka[index] != kb[index] )
			return ka[index] < kb[index] ? -1 : 1;
	return 0;
}

static int DraftdTapSummary(const char *path)
{
	static DraftdSummaryKey keys[DRAFTD_SUMMARY_KEYS];
	static const char *const kinds[4] = {"prefill", "decode", "verify", "chain"};
	SparkTapStreamReaderConfig config;
	SparkTapStreamReader *reader = 0;
	SparkTapStreamReaderFrame frame;
	SparkStatus status;
	uint32_t key_count = 0u;
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_TAP_STREAM_READER_ABI_VERSION;
	config.descriptor_bytes = SPARK_TAP_STREAM_READER_CONFIG_BYTES;
	config.path = path;
	config.max_rows = 1u << 16u;
	config.io_timeout_ms = DRAFTD_DEFAULT_IO_TIMEOUT_MS;
	if ( SparkTapStreamReaderCreate(&config, &reader) != SPARK_STATUS_OK )
		return 2;
	while ( (status = SparkTapStreamReaderAdvance(reader, &frame)) == SPARK_STATUS_OK )
	{
		DraftdSummaryKey *key = 0;
		for ( uint32_t index = 0u; index < key_count && key == 0; ++index )
			if ( keys[index].control_generation == frame.header.control_generation && keys[index].request_id == frame.header.request_id &&
				keys[index].request_generation == frame.header.request_generation && keys[index].sequence_id == frame.header.sequence_id )
				key = &keys[index];
		if ( key == 0 )
		{
			if ( key_count == DRAFTD_SUMMARY_KEYS )
			{
				fprintf(stderr, "sparkpipe_draftd: %s holds more than %u requests\n", path, DRAFTD_SUMMARY_KEYS);
				SparkTapStreamReaderDestroy(reader);
				return 2;
			}
			key = &keys[key_count++];
			memset(key, 0, sizeof(*key));
			key->control_generation = frame.header.control_generation;
			key->request_id = frame.header.request_id;
			key->request_generation = frame.header.request_generation;
			key->sequence_id = frame.header.sequence_id;
		}
		key->frames++;
		key->rows += frame.header.row_count;
		for ( uint32_t bit = 0u; bit < 4u; ++bit )
			key->kinds[bit] += (frame.header.flags >> bit) & 1u;
		if ( key->ended != 0u && frame.header.first_position != key->end )
		{
			if ( frame.header.first_position > key->end )
				key->gaps++;
			else
				key->overlaps++;
		}
		key->end = frame.header.first_position + frame.header.row_count;
		key->ended = 1u;
	}
	SparkTapStreamReaderDestroy(reader);
	if ( status != SPARK_STATUS_NOT_FOUND )
		return 1;
	qsort(keys, key_count, sizeof(keys[0]), DraftdSummaryOrder);
	for ( uint32_t index = 0u; index < key_count; ++index )
	{
		printf("request %llu sequence %llu frames %llu rows %llu end %llu gaps %llu overlaps %llu", (unsigned long long)keys[index].request_id,
			(unsigned long long)keys[index].sequence_id, (unsigned long long)keys[index].frames, (unsigned long long)keys[index].rows,
			(unsigned long long)keys[index].end, (unsigned long long)keys[index].gaps, (unsigned long long)keys[index].overlaps);
		for ( uint32_t bit = 0u; bit < 4u; ++bit )
			if ( keys[index].kinds[bit] != 0u )
				printf(" %s=%llu", kinds[bit], (unsigned long long)keys[index].kinds[bit]);
		printf("\n");
	}
	return 0;
}

int main(int argc, char **argv)
{
	struct sigaction action;
	if ( argc == 3 && strcmp(argv[1], "--tap-summary") == 0 )
		return DraftdTapSummary(argv[2]);
	if ( (argc == 4 || argc == 5) && strcmp(argv[1], "--control") == 0 )
		return DraftdControlClient(argv[2], argv[3], argc == 5 ? argv[4] : 0);
	if ( argc != 3 || strcmp(argv[1], "--config") != 0 )
	{
		fprintf(stderr, "usage: sparkpipe_draftd --config LANES.json | --control SOCKET status | --control SOCKET activate|deactivate LANE | "
			"--tap-summary RECORDING.spt2\n");
		return 2;
	}
	memset(&action, 0, sizeof(action));
	action.sa_handler = DraftdSignal;
	(void)sigaction(SIGTERM, &action, 0);
	(void)sigaction(SIGINT, &action, 0);
	(void)signal(SIGPIPE, SIG_IGN);
	return DraftdServe(argv[2]);
}
