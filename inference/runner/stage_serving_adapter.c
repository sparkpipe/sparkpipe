#include <arpa/inet.h>
#include "sparkpipe/spark_clock.h"
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_serving_cache_admission.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_speculation_tree_frame.h"
#include "sparkpipe/spark_stage_kv_binding.h"
#include "sparkpipe/spark_stage_module_claims.h"
#include "sparkpipe/spark_stage_runner.h"
#include "sparkpipe/spark_stage_serving_adapter.h"
#include "sparkpipe/spark_tap_stream.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_attach.h"

#ifdef MSG_NOSIGNAL
#define STAGE_SERVING_TAP_SEND_FLAGS MSG_NOSIGNAL
#else
#define STAGE_SERVING_TAP_SEND_FLAGS 0
#endif
#define STAGE_SERVING_TAP_BUFFER_BYTES (8u << 20)
#define STAGE_SERVING_TAP_CONNECT_MS 3000
#define STAGE_SERVING_TAP_RECONNECT_MS 50
#define STAGE_SERVING_TAP_RETRY_NS 5000000000ull
#define STAGE_SERVING_TAP_SEND_TIMEOUT_MS 200
#define STAGE_SERVING_TAP_SEND_BUFFER_BYTES (64 << 20)
#define STAGE_SERVING_VERIFY_BUCKETS 4u
#define STAGE_SERVING_PROGRAM_ID 1u
#define STAGE_SERVING_BUFFERS 28u

typedef struct SparkStageServingState
{
	const SparkStageServingModel *model;
	uint32_t implementation_flags;
	SparkStageRunner runner;
	SparkStageRunnerConfiguration runner_config;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	char *pack_path;
	char *stray_working_set_path;
	uint64_t state_memory_bytes;
	char *weightd_socket_path;
	uint32_t saves_held;
	uint32_t streamed_failed;
	char *streamed_partner_pack;
	char *streamed_fabric_device;
	char *streamed_pair_device;
	uint32_t max_rows;
	uint32_t max_shared_rows;
	SparkTpDeviceCollectiveConfig device_config;
	SparkWeightdClient *lane_client;
	int device_collective_present;
	SparkDeviceBuffer positions_host;
	SparkDeviceBuffer context_host;
	SparkDeviceBuffer state_host;
	uint64_t wave_host_ns[5];
	uint32_t wave_host_count;
	uint64_t chain_host_ns[5];
	uint64_t chain_host_steps;
	uint32_t chain_host_count;
	uint64_t verify_host_ns[STAGE_SERVING_VERIFY_BUCKETS][6];
	uint64_t verify_host_rows[STAGE_SERVING_VERIFY_BUCKETS];
	uint32_t verify_host_count[STAGE_SERVING_VERIFY_BUCKETS];
	uint64_t publish_host_ns;
	uint64_t publish_host_lanes;
	uint32_t publish_host_count;
	SparkDeviceBuffer positions_device;
	SparkDeviceBuffer context_device;
	SparkDeviceBuffer state_device;
	SparkDeviceBuffer runs_host;
	SparkDeviceBuffer runs_device;
	SparkDeviceBuffer seqslot_host;
	SparkDeviceBuffer target_host;
	SparkDeviceBuffer seqslot_device;
	SparkDeviceBuffer order_host;
	SparkDeviceBuffer order_device;
	SparkDeviceBuffer slots_host;
	SparkDeviceBuffer slots_device;
	SparkDeviceBuffer fork_host;
	SparkDeviceBuffer fork_device;
	SparkDeviceBuffer offset_host;
	SparkDeviceBuffer offset_device;
	SparkDeviceBuffer parent_host;
	SparkDeviceBuffer parent_device;
	SparkDeviceBuffer loads_host;
	SparkDeviceBuffer loads_device;
	SparkDeviceBuffer saves_host;
	SparkDeviceBuffer saves_device;
	uint32_t tree_frame;
	uint64_t lane_contexts[SPARK_STAGE_SERVING_MAX_LANES];
	uint32_t path_begin[SPARK_STAGE_SERVING_MAX_LANES + 1u];
	uint32_t path_rows[SPARK_STAGE_SERVING_MAX_ROWS];
	SparkDeviceBuffer output_tokens;
	SparkDeviceBuffer output_scores;
	SparkDeviceBuffer tokens_host;
	SparkStageKvBinding kv;
	SparkModelDriverCacheLane cache_lanes[SPARK_STAGE_SERVING_MAX_LANES];
	SparkModelDriverCacheLane prefetch_lanes[SPARK_STAGE_SERVING_MAX_LANES];
	atomic_uint lane_states[SPARK_STAGE_SERVING_MAX_LANES];
	atomic_ullong reset_generation;
	SparkModelDriverCompletion publish_completion;
	uint32_t publish_completed;
	uint32_t row_slots[SPARK_STAGE_SERVING_MAX_ROWS];
	uint64_t row_sequence_ids[SPARK_STAGE_SERVING_MAX_ROWS];
	uint64_t row_positions[SPARK_STAGE_SERVING_MAX_ROWS];
	uint32_t distribution_rows[SPARK_STAGE_SERVING_MAX_ROWS];
	uint32_t distribution_positions[SPARK_STAGE_SERVING_MAX_ROWS];
	uint32_t distribution_lanes[SPARK_STAGE_SERVING_MAX_ROWS];
	SparkRowSampling distribution_rules[SPARK_STAGE_SERVING_MAX_ROWS];
	SparkStageRunnerTapEntry tap_entries[SPARK_STAGE_RUNNER_TAP_MAX_ENTRIES];
	uint32_t tap_count;
	uint32_t tap_rank;
	uint32_t tap_window;
	char *tap_file;
	char *tap_stream;
	uint64_t tap_file_max_bytes;
	FILE *tap_out;
	int tap_socket;
	uint8_t *tap_buffer;
	size_t tap_buffered;
	SparkTapStreamHello tap_hello;
	uint64_t tap_retry_ns;
	uint32_t tap_stream_down;
	uint64_t tap_written;
	uint64_t tap_serial;
	uint64_t tap_fingerprint;
	uint32_t tap_full;
} SparkStageServingState;

typedef struct SparkStageServingBuffer
{
	SparkDeviceBuffer *buffer;
	SparkDeviceSpace space;
	uint64_t bytes;
} SparkStageServingBuffer;

static const SparkStageServingModel *StageServingBoundModel;
static SparkModelServingAdapterDescriptor StageServingDescriptor;

static uint32_t *StageServingWords(const SparkDeviceBuffer *buffer)
{
	return (uint32_t *)buffer->host;
}

static void *StageServingAddress(const SparkDeviceBuffer *buffer)
{
	return (void *)(uintptr_t)buffer->device;
}

static uint32_t StageServingOwnPack(uint32_t runner_flags)
{
	return (runner_flags & SPARK_STAGE_RUNNER_FLAG_OWN_PACK) != 0u ? 1u : 0u;
}

static uint32_t StageServingWeightdRefused(const SparkStageServingState *state, const char *section)
{
	if ( StageServingOwnPack(state->implementation_flags) == 0u )
		return 0u;
	fprintf(stderr, "%s adapter: %s needs weightd, and this runner maps its own pack\n", state->model->module_tag, section);
	return 1u;
}

static SparkStatus StageServingCheckOwnPack(const SparkStageServingState *state)
{
	if ( StageServingOwnPack(state->runner.flags) == StageServingOwnPack(state->implementation_flags) )
		return SPARK_STATUS_OK;
	fprintf(stderr, "%s adapter: the runner's own-pack flag %u disagrees with its implementation's %u\n", state->model->module_tag,
		StageServingOwnPack(state->runner.flags), StageServingOwnPack(state->implementation_flags));
	SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
}

static SparkStatus StageServingU32Member(const SparkStageServingState *state, SparkJsonDocument *doc, int32_t object, const char *where,
	const char *name, uint32_t *value)
{
	int32_t token = SparkJsonFindObjectMember(doc, object, name);
	if ( token >= 0 && SparkJsonGetUInt32(doc, token, value) == SPARK_STATUS_OK )
		return SPARK_STATUS_OK;
	fprintf(stderr, "%s adapter: %s needs \"%s\", an unsigned 32-bit integer\n", state->model->module_tag, where, name);
	return SPARK_STATUS_SCHEMA_ERROR;
}

static SparkStatus StageServingOptionalU32Member(const SparkStageServingState *state, SparkJsonDocument *doc, int32_t object,
	const char *where, const char *name, uint32_t absent, uint32_t *value)
{
	if ( SparkJsonFindObjectMember(doc, object, name) >= 0 )
		return StageServingU32Member(state, doc, object, where, name, value);
	*value = absent;
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingWeightdHandoff(SparkStageServingState *state)
{
	SparkStageRunnerConfiguration *config = &state->runner_config;
	const char *socket = getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
	const char *digest = getenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
	if ( socket == 0 || socket[0] == '\0' || digest == 0 || strlen(digest) != SPARK_STAGE_RUNNER_DIGEST_HEX )
	{
		fprintf(stderr, "%s adapter: residentd hands the adapter the weightd socket in %s and the pack digest in %s, and this process has %s\n",
			state->model->module_tag, SPARK_WEIGHTD_ATTACH_ENV_SOCKET, SPARK_WEIGHTD_ATTACH_ENV_SHA256,
			socket == 0 || socket[0] == '\0' ? "no socket" : "no 64-digit digest");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->weightd_socket_path = strdup(socket);
	if ( state->weightd_socket_path == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	config->weights.socket_path = state->weightd_socket_path;
	memcpy(config->weights.pack_sha256, digest, SPARK_STAGE_RUNNER_DIGEST_HEX + 1u);
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingParseWeights(SparkStageServingState *state, SparkJsonDocument *doc, int32_t root)
{
	SparkStageRunnerWeights *weights = &state->runner_config.weights;
	int32_t cap = SparkJsonFindObjectMember(doc, root, "expert_pool_cap_bytes");
	int32_t share = SparkJsonFindObjectMember(doc, root, "weights_share");
	weights->expert_pool_cap_bytes = 0u;
	weights->read_only = 0u;
	if ( (cap >= 0 && StageServingWeightdRefused(state, "expert_pool_cap_bytes") != 0u) ||
		(share >= 0 && StageServingWeightdRefused(state, "weights_share") != 0u) )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( cap >= 0 && (SparkJsonGetUInt64(doc, cap, &weights->expert_pool_cap_bytes) != SPARK_STATUS_OK || weights->expert_pool_cap_bytes == 0u) )
	{
		fprintf(stderr, "%s adapter: expert_pool_cap_bytes is the lane's explicit expert pool cap, a positive byte count; leave it out to size the pool from the pack\n",
			state->model->module_tag);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( share >= 0 && !SparkJsonStringEquals(doc, share, "readonly") )
	{
		fprintf(stderr, "%s adapter: weights_share names \"readonly\", which maps an arena another lane already holds; leave it out for the lane's own attach\n",
			state->model->module_tag);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	weights->read_only = share >= 0 ? 1u : 0u;
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingKnownMembers(const SparkStageServingState *state, SparkJsonDocument *doc, int32_t object, const char *where,
	const char *const *names, uint32_t count)
{
	int32_t unknown = -1;
	char *key = 0;
	if ( SparkJsonValidateObjectMembersKnown(doc, object, names, count, &unknown) == SPARK_STATUS_OK )
		return SPARK_STATUS_OK;
	if ( unknown >= 0 )
		(void)SparkJsonCopyString(doc, unknown, &key);
	fprintf(stderr, "%s adapter: %s has the unknown member \"%s\"; a stage config carries only the members this adapter reads\n",
		state->model->module_tag, where, key != 0 ? key : "?");
	free(key);
	return SPARK_STATUS_SCHEMA_ERROR;
}

static SparkStatus StageServingCopyMember(SparkJsonDocument *doc, int32_t object, const char *name, char **out)
{
	int32_t token = SparkJsonFindObjectMember(doc, object, name);
	if ( token < 0 )
		return SPARK_STATUS_OK;
	return SparkJsonCopyString(doc, token, out) == SPARK_STATUS_OK ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR;
}

static SparkStatus StageServingParseStreamed(SparkStageServingState *state, SparkJsonDocument *doc, int32_t object)
{
	static const char *const members[] = { "partner_pack", "fabric_device", "pair_device", "node_rows", "gid_index", "pair_gid_index",
		"traffic_class", "dense_buffers", "expert_slots" };
	SparkStageRunnerStreamedPrefillConfig *streamed = &state->runner_config.streamed_prefill;
	if ( !SparkJsonTokenIsType(doc, object, SPARK_JSON_TOKEN_OBJECT) ||
		StageServingKnownMembers(state, doc, object, "streamed_prefill", members, sizeof(members) / sizeof(members[0])) != SPARK_STATUS_OK ||
		StageServingCopyMember(doc, object, "partner_pack", &state->streamed_partner_pack) != SPARK_STATUS_OK ||
		StageServingCopyMember(doc, object, "fabric_device", &state->streamed_fabric_device) != SPARK_STATUS_OK ||
		StageServingCopyMember(doc, object, "pair_device", &state->streamed_pair_device) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( StageServingU32Member(state, doc, object, "streamed_prefill", "node_rows", &streamed->node_rows) != SPARK_STATUS_OK ||
		StageServingU32Member(state, doc, object, "streamed_prefill", "gid_index", &streamed->gid_index) != SPARK_STATUS_OK ||
		StageServingU32Member(state, doc, object, "streamed_prefill", "pair_gid_index", &streamed->pair_gid_index) != SPARK_STATUS_OK ||
		StageServingU32Member(state, doc, object, "streamed_prefill", "traffic_class", &streamed->traffic_class) != SPARK_STATUS_OK ||
		StageServingU32Member(state, doc, object, "streamed_prefill", "dense_buffers", &streamed->dense_buffers) != SPARK_STATUS_OK ||
		StageServingU32Member(state, doc, object, "streamed_prefill", "expert_slots", &streamed->expert_slots) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	streamed->partner_pack_path = state->streamed_partner_pack;
	streamed->fabric_device = state->streamed_fabric_device;
	streamed->pair_device = state->streamed_pair_device;
	if ( streamed->node_rows == 0u || streamed->partner_pack_path == 0 || streamed->fabric_device == 0 || streamed->pair_device == 0 )
	{
		fprintf(stderr, "%s adapter: streamed_prefill needs node_rows, partner_pack, fabric_device and pair_device\n", state->model->module_tag);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingLoadMeshRanks(SparkStageServingState *state, SparkJsonDocument *doc, int32_t dev)
{
	int32_t ranks = SparkJsonFindObjectMember(doc, dev, "mesh_ranks");
	uint32_t count = 0u, index, seen = 0u, value;
	if ( ranks >= 0 && SparkJsonTokenIsType(doc, ranks, SPARK_JSON_TOKEN_ARRAY) )
		count = SparkJsonGetArrayElementCount(doc, ranks);
	if ( count != state->runner_config.tp_degree )
	{
		fprintf(stderr, "%s adapter: device_collective.mesh_ranks lists the physical mesh rank of each of the %u TP ranks\n",
			state->model->module_tag, state->runner_config.tp_degree);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	for ( index = 0u; index < count; ++index )
	{
		int32_t cell = SparkJsonGetArrayElement(doc, ranks, index);
		value = 0u;
		if ( cell < 0 || SparkJsonGetUInt32(doc, cell, &value) != SPARK_STATUS_OK || value >= SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE ||
			(seen & (1u << value)) != 0u )
		{
			fprintf(stderr, "%s adapter: device_collective.mesh_ranks names distinct physical ranks below %u\n",
				state->model->module_tag, SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE);
			return SPARK_STATUS_SCHEMA_ERROR;
		}
		seen |= 1u << value;
		state->device_config.mesh_ranks[index] = value;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingLoadDeviceCollective(SparkStageServingState *state, SparkJsonDocument *doc, int32_t dev)
{
	static const char *const members[] = { "wait_mode", "connect_timeout_milli", "operation_timeout_milli", "mesh_lane", "mesh_ranks" };
	const SparkModelServingAdapterDescriptor *descriptor = &StageServingDescriptor;
	SparkTpDeviceCollectiveConfig *config = &state->device_config;
	int32_t token;
	SparkStatus status;
	if ( !SparkJsonTokenIsType(doc, dev, SPARK_JSON_TOKEN_OBJECT) )
		return SPARK_STATUS_SCHEMA_ERROR;
	status = StageServingKnownMembers(state, doc, dev, "device_collective", members, sizeof(members) / sizeof(members[0]));
	if ( status != SPARK_STATUS_OK )
		return status;
	memset(config, 0, sizeof(*config));
	config->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	config->backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	config->operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	config->credit_count = 8u;
	config->local_hidden_dimension = descriptor->boundary_element_count;
	config->max_active_sequence_count =
		state->model->collective_sequences(state->runner_config.max_input_row_count, state->runner_config.tp_degree);
	token = SparkJsonFindObjectMember(doc, dev, "wait_mode");
	if ( token < 0 )
	{
		fprintf(stderr, "%s adapter: device_collective.wait_mode is required and names \"hardware\" or \"spin\"; a rank never picks a wait regime by default\n",
			state->model->module_tag);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( SparkJsonStringEquals(doc, token, "hardware") )
		config->wait_mode = SPARK_TP_DEVICE_COLLECTIVE_WAIT_HARDWARE;
	else if ( SparkJsonStringEquals(doc, token, "spin") )
		config->wait_mode = SPARK_TP_DEVICE_COLLECTIVE_WAIT_SPIN;
	else
	{
		fprintf(stderr, "%s adapter: device_collective.wait_mode names \"hardware\" or \"spin\"\n", state->model->module_tag);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	status = StageServingU32Member(state, doc, dev, "device_collective", "connect_timeout_milli", &config->connect_timeout_milli);
	if ( status == SPARK_STATUS_OK )
		status = StageServingU32Member(state, doc, dev, "device_collective", "operation_timeout_milli", &config->operation_timeout_milli);
	if ( status == SPARK_STATUS_OK )
		status = StageServingU32Member(state, doc, dev, "device_collective", "mesh_lane", &config->mesh_lane);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( config->connect_timeout_milli == 0u || config->operation_timeout_milli == 0u || config->mesh_lane >= SPARK_WEIGHTD_MESH_MAX_LANES )
	{
		fprintf(stderr, "%s adapter: device_collective needs positive timeouts and a mesh_lane below %u\n", state->model->module_tag,
			SPARK_WEIGHTD_MESH_MAX_LANES);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	status = StageServingLoadMeshRanks(state, doc, dev);
	if ( status != SPARK_STATUS_OK )
		return status;
	state->device_collective_present = 1;
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingParseTaps(SparkStageServingState *state, SparkJsonDocument *doc, int32_t taps)
{
	static const char tap_final[] = "final";
	static const char *const members[] = { "entries", "file", "stream", "rank", "window", "file_max_bytes" };
	char *entries = 0, *cursor, *end;
	int32_t token = SparkJsonFindObjectMember(doc, taps, "entries");
	const int32_t file = SparkJsonFindObjectMember(doc, taps, "file"), stream = SparkJsonFindObjectMember(doc, taps, "stream");
	if ( !SparkJsonTokenIsType(doc, taps, SPARK_JSON_TOKEN_OBJECT) ||
		StageServingKnownMembers(state, doc, taps, "taps", members, sizeof(members) / sizeof(members[0])) != SPARK_STATUS_OK )
		return SPARK_STATUS_SCHEMA_ERROR;
	if ( token < 0 || SparkJsonCopyString(doc, token, &entries) != SPARK_STATUS_OK || (file >= 0) == (stream >= 0) ||
		(file >= 0 && (StageServingCopyMember(doc, taps, "file", &state->tap_file) != SPARK_STATUS_OK || state->tap_file == 0)) ||
		(stream >= 0 && (StageServingCopyMember(doc, taps, "stream", &state->tap_stream) != SPARK_STATUS_OK || state->tap_stream == 0)) )
	{
		free(entries);
		fprintf(stderr, "%s adapter: taps need \"entries\" and exactly one of \"file\" and \"stream\"\n", state->model->module_tag);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( StageServingU32Member(state, doc, taps, "taps", "rank", &state->tap_rank) != SPARK_STATUS_OK ||
		StageServingOptionalU32Member(state, doc, taps, "taps", "window", 0u, &state->tap_window) != SPARK_STATUS_OK )
	{
		free(entries);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	token = SparkJsonFindObjectMember(doc, taps, "file_max_bytes");
	if ( file >= 0 && (token < 0 || SparkJsonGetUInt64(doc, token, &state->tap_file_max_bytes) != SPARK_STATUS_OK || state->tap_file_max_bytes == 0u) )
	{
		free(entries);
		fprintf(stderr, "%s adapter: a tap file needs a positive \"file_max_bytes\"\n", state->model->module_tag);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( stream >= 0 && token >= 0 )
	{
		free(entries);
		fprintf(stderr, "%s adapter: a tap stream has no \"file_max_bytes\"\n", state->model->module_tag);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	for ( cursor = entries; *cursor != '\0'; cursor = *end == ',' ? end + 1 : end )
	{
		SparkStageRunnerTapEntry *entry = &state->tap_entries[state->tap_count];
		unsigned long layer;
		if ( state->tap_count >= SPARK_STAGE_RUNNER_TAP_MAX_ENTRIES )
			break;
		if ( strncmp(cursor, tap_final, sizeof(tap_final) - 1u) == 0 && (cursor[sizeof(tap_final) - 1u] == ',' || cursor[sizeof(tap_final) - 1u] == '\0') )
		{
			entry->form = SPARK_STAGE_RUNNER_TAP_FORM_FINAL;
			entry->layer = 0u;
			end = cursor + sizeof(tap_final) - 1u;
		}
		else if ( cursor[0] == 'o' && cursor[1] >= '0' && cursor[1] <= '9' )
		{
			errno = 0;
			layer = strtoul(cursor + 1, &end, 10);
			if ( errno != 0 || layer > UINT16_MAX || (*end != ',' && *end != '\0') )
				break;
			entry->form = SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT;
			entry->layer = (uint16_t)layer;
		}
		else
			break;
		state->tap_count++;
	}
	if ( *cursor != '\0' || state->tap_count == 0u )
	{
		fprintf(stderr, "%s adapter: taps entries \"%s\" are not a list of o<layer> and final\n", state->model->module_tag, entries);
		free(entries);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	free(entries);
	return SPARK_STATUS_OK;
}

static int StageServingTapConnect(const char *endpoint, int timeout_ms)
{
	char host[64];
	const char *colon = strrchr(endpoint, ':');
	struct sockaddr_in address;
	struct pollfd descriptor;
	struct timeval timeout;
	unsigned long port;
	char *end = 0;
	int fd, flags, ready, error = 0, enabled = 1, buffer = STAGE_SERVING_TAP_SEND_BUFFER_BYTES;
	socklen_t error_bytes = (socklen_t)sizeof(error);
	if ( colon == 0 || colon == endpoint || (size_t)(colon - endpoint) >= sizeof(host) )
		return -1;
	memcpy(host, endpoint, (size_t)(colon - endpoint));
	host[colon - endpoint] = '\0';
	errno = 0;
	port = strtoul(colon + 1, &end, 10);
	memset(&address, 0, sizeof(address));
	address.sin_family = AF_INET;
	address.sin_port = htons((uint16_t)port);
	if ( errno != 0 || end == colon + 1 || *end != '\0' || port == 0u || port > 65535u || inet_pton(AF_INET, host, &address.sin_addr) != 1 )
		return -1;
	fd = socket(AF_INET, SOCK_STREAM, 0);
	if ( fd < 0 )
		return -1;
	flags = fcntl(fd, F_GETFL, 0);
	if ( flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0 )
	{
		(void)close(fd);
		return -1;
	}
	if ( connect(fd, (const struct sockaddr *)&address, (socklen_t)sizeof(address)) != 0 )
	{
		if ( errno != EINPROGRESS )
		{
			(void)close(fd);
			return -1;
		}
		memset(&descriptor, 0, sizeof(descriptor));
		descriptor.fd = fd;
		descriptor.events = POLLOUT;
		do
			ready = poll(&descriptor, 1u, timeout_ms);
		while ( ready < 0 && errno == EINTR );
		if ( ready != 1 || getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &error_bytes) != 0 || error != 0 )
		{
			(void)close(fd);
			return -1;
		}
	}
	timeout.tv_sec = STAGE_SERVING_TAP_SEND_TIMEOUT_MS / 1000;
	timeout.tv_usec = (STAGE_SERVING_TAP_SEND_TIMEOUT_MS % 1000) * 1000;
	if ( fcntl(fd, F_SETFL, flags) != 0 || setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enabled, (socklen_t)sizeof(enabled)) != 0 ||
		setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, (socklen_t)sizeof(timeout)) != 0 ||
		setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer, (socklen_t)sizeof(buffer)) != 0 )
	{
		(void)close(fd);
		return -1;
	}
#ifdef SO_NOSIGPIPE
	if ( setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, (socklen_t)sizeof(enabled)) != 0 )
	{
		(void)close(fd);
		return -1;
	}
#endif
	return fd;
}

static int StageServingTapSend(int fd, const uint8_t *data, size_t bytes)
{
	while ( bytes != 0u )
	{
		const ssize_t sent = send(fd, data, bytes, STAGE_SERVING_TAP_SEND_FLAGS);
		if ( sent < 0 && errno == EINTR )
			continue;
		if ( sent <= 0 )
			return -1;
		data += sent;
		bytes -= (size_t)sent;
	}
	return 0;
}

static int StageServingTapFlush(SparkStageServingState *state)
{
	int result = 0;
	if ( state->tap_socket >= 0 && state->tap_buffered != 0u )
		result = StageServingTapSend(state->tap_socket, state->tap_buffer, state->tap_buffered);
	state->tap_buffered = 0u;
	return result;
}

static int StageServingTapWrite(SparkStageServingState *state, const void *data, size_t bytes)
{
	if ( state->tap_socket < 0 )
		return fwrite(data, 1u, bytes, state->tap_out) == bytes ? 0 : -1;
	if ( state->tap_buffered + bytes > STAGE_SERVING_TAP_BUFFER_BYTES && StageServingTapFlush(state) != 0 )
		return -1;
	if ( bytes > STAGE_SERVING_TAP_BUFFER_BYTES )
		return StageServingTapSend(state->tap_socket, (const uint8_t *)data, bytes);
	memcpy(state->tap_buffer + state->tap_buffered, data, bytes);
	state->tap_buffered += bytes;
	return 0;
}

static uint32_t StageServingTapOpen(const SparkStageServingState *state)
{
	return state->tap_out != 0 || state->tap_socket >= 0 ? 1u : 0u;
}

static const char *StageServingTapName(const SparkStageServingState *state)
{
	return state->tap_file != 0 ? state->tap_file : state->tap_stream != 0 ? state->tap_stream : "";
}

static void StageServingTapClose(SparkStageServingState *state)
{
	if ( state->tap_out != 0 )
		(void)fclose(state->tap_out);
	if ( state->tap_socket >= 0 )
		(void)close(state->tap_socket);
	state->tap_out = 0;
	state->tap_socket = -1;
	state->tap_buffered = 0u;
}


static uint32_t StageServingTapStreamConnect(SparkStageServingState *state, int timeout_ms)
{
	state->tap_socket = StageServingTapConnect(state->tap_stream, timeout_ms);
	if ( state->tap_socket >= 0 && state->tap_buffer == 0 )
		state->tap_buffer = (uint8_t *)malloc(STAGE_SERVING_TAP_BUFFER_BYTES);
	state->tap_buffered = 0u;
	if ( state->tap_socket < 0 || state->tap_buffer == 0 || StageServingTapWrite(state, &state->tap_hello, sizeof(state->tap_hello)) != 0 ||
		StageServingTapFlush(state) != 0 )
	{
		if ( state->tap_stream_down == 0u )
			fprintf(stderr, "%s adapter: tap stream %s is unreachable (%s); serving continues without taps and retries every %llu s\n", state->model->module_tag,
				state->tap_stream, strerror(errno), (unsigned long long)(STAGE_SERVING_TAP_RETRY_NS / 1000000000ull));
		StageServingTapClose(state);
		state->tap_stream_down = 1u;
		state->tap_retry_ns = SparkMonotonicNs() + STAGE_SERVING_TAP_RETRY_NS;
		return 0u;
	}
	if ( state->tap_stream_down != 0u )
		fprintf(stderr, "%s adapter: tap stream %s connected again\n", state->model->module_tag, state->tap_stream);
	state->tap_stream_down = 0u;
	return 1u;
}

static SparkStatus StageServingOpenTaps(SparkStageServingState *state)
{
	SparkTapStreamHello hello;
	const uint32_t capture = state->runner_config.tp_rank == state->tap_rank ? 1u : 0u;
	SparkStatus status;
	if ( state->tap_count == 0u )
		return SPARK_STATUS_OK;
	status = SparkStageRunnerSetTaps(&state->runner, state->tap_entries, state->tap_count, capture);
	if ( status != SPARK_STATUS_OK || capture == 0u )
		return status;
	memset(&hello, 0, sizeof(hello));
	hello.magic = SPARK_TAP_STREAM_MAGIC;
	hello.version = SPARK_TAP_STREAM_VERSION;
	hello.kind = SPARK_TAP_STREAM_KIND_HELLO;
	hello.header_bytes = (uint32_t)sizeof(hello);
	strncpy(hello.model_tag, state->model->module_tag, sizeof(hello.model_tag) - 1u);
	if ( SparkStageRunnerPackIdentity(&state->runner, hello.pack_identity, sizeof(hello.pack_identity)) != SPARK_STATUS_OK )
		memset(hello.pack_identity, 0, sizeof(hello.pack_identity));
	if ( SparkStageRunnerTapShape(&state->runner, &hello.hidden, &hello.total_layers) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	hello.dtype = SPARK_TAP_STREAM_DTYPE_BF16;
	hello.entry_count = state->tap_count;
	hello.tap_rank = state->tap_rank;
	hello.tp_degree = state->runner_config.tp_degree;
	for ( uint32_t index = 0u; index < state->tap_count; ++index )
	{
		hello.entries[index].form = state->tap_entries[index].form;
		hello.entries[index].layer = state->tap_entries[index].layer;
	}
	hello.fingerprint = SparkTapStreamFingerprint(&hello);
	state->tap_fingerprint = hello.fingerprint;
	state->tap_hello = hello;
	if ( state->tap_stream != 0 )
		(void)StageServingTapStreamConnect(state, STAGE_SERVING_TAP_CONNECT_MS);
	else
	{
		state->tap_out = fopen(state->tap_file, "wb");
		if ( state->tap_out == 0 || setvbuf(state->tap_out, 0, _IOFBF, 16u << 20) != 0 || fwrite(&hello, sizeof(hello), 1u, state->tap_out) != 1u )
		{
			fprintf(stderr, "%s adapter: tap file %s cannot be written: %s\n", state->model->module_tag, state->tap_file, strerror(errno));
			StageServingTapClose(state);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
	}
	state->tap_written = sizeof(hello);
	fprintf(stderr, "%s adapter: taps entries=%u rank=%u %s=%s max_bytes=%llu fingerprint=%016llx\n", state->model->module_tag, state->tap_count,
		state->tap_rank, state->tap_stream != 0 ? "stream" : "file", StageServingTapName(state), (unsigned long long)state->tap_file_max_bytes,
		(unsigned long long)state->tap_fingerprint);
	return SPARK_STATUS_OK;
}

static void StageServingTapEmit(SparkStageServingState *state, const SparkModelServingSubmission *submission, uint32_t rows, uint32_t active,
	uint32_t steps, const uint32_t *kept)
{
	const uint32_t *runs = StageServingWords(&state->runs_host);
	const uint32_t *order = state->tree_frame != 0u && kept != 0 ? state->path_rows : StageServingWords(&state->order_host);
	const uint32_t *positions = StageServingWords(&state->positions_host);
	const uint32_t *completed = StageServingWords(&state->tokens_host);
	uint32_t tokens[SPARK_STAGE_SERVING_MAX_ROWS];
	for ( uint32_t s = 0u; s < active && StageServingTapOpen(state) != 0u; ++s )
	{
		SparkTapStreamFrame frame;
		uint64_t stride = 0u, row_bytes, bytes;
		const uint32_t chain = steps > 1u ? 1u : 0u;
		const uint32_t listed = chain != 0u ? steps : kept != 0 ? kept[s] : runs[s + 1u] - runs[s];
		const uint32_t valid = chain == 0u && kept == 0 && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ?
			SparkStageRunnerTapFirstRow(&state->runner) : 0u;
		uint32_t skip = 0u;
		while ( chain == 0u && skip < listed && order[runs[s] + skip] < valid )
			skip++;
		if ( chain == 0u && state->tap_window != 0u && listed - skip > state->tap_window )
			skip = listed - state->tap_window;
		const uint32_t count = listed - skip;
		const uint32_t begin = runs[s] + skip;
		const uint32_t first_row = count != 0u ? order[begin] : 0u;
		const uint16_t *base = SparkStageRunnerTapRows(&state->runner, 0u, rows, &stride);
		if ( base == 0 || count == 0u )
			continue;
		row_bytes = stride * sizeof(uint16_t);
		bytes = sizeof(frame) + (uint64_t)count * (sizeof(uint32_t) + row_bytes);
		if ( state->tap_socket < 0 && state->tap_written + bytes > state->tap_file_max_bytes )
		{
			if ( state->tap_full == 0u )
				fprintf(stderr, "%s adapter: tap file %s reached %llu bytes; later frames are not written\n", state->model->module_tag, state->tap_file,
					(unsigned long long)state->tap_written);
			state->tap_full = 1u;
			return;
		}
		for ( uint32_t k = 0u; k < count; ++k )
			tokens[k] = chain != 0u ? (k == 0u ? submission->token_ids[first_row] : completed[(k - 1u) * rows + first_row]) :
				submission->token_ids[order[begin + k]];
		memset(&frame, 0, sizeof(frame));
		frame.magic = SPARK_TAP_STREAM_MAGIC;
		frame.version = SPARK_TAP_STREAM_VERSION;
		frame.kind = SPARK_TAP_STREAM_KIND_FRAME;
		frame.header_bytes = (uint32_t)sizeof(frame);
		frame.fingerprint = state->tap_fingerprint;
		frame.control_generation = submission->control_generation;
		frame.request_id = submission->lanes[s].request_id;
		frame.request_generation = submission->lanes[s].request_generation;
		frame.sequence_id = submission->lanes[s].sequence_id;
		frame.first_position = positions[first_row];
		frame.serial = state->tap_serial++;
		frame.row_count = count;
		frame.flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_TAP_STREAM_FLAG_PREFILL :
			kept != 0 ? SPARK_TAP_STREAM_FLAG_VERIFY : chain != 0u ? SPARK_TAP_STREAM_FLAG_DECODE | SPARK_TAP_STREAM_FLAG_CHAIN : SPARK_TAP_STREAM_FLAG_DECODE;
		frame.tail_token = state->runner.owns_final_head == 0u ? SPARK_TAP_STREAM_NO_TOKEN :
			chain != 0u ? completed[(steps - 1u) * rows + first_row] : completed[order[begin + count - 1u]];
		frame.entry_count = state->tap_count;
		frame.payload_bytes = bytes - sizeof(frame);
		if ( StageServingTapWrite(state, &frame, sizeof(frame)) != 0 || StageServingTapWrite(state, tokens, (size_t)count * sizeof(uint32_t)) != 0 )
			state->tap_full = 1u;
		for ( uint32_t k = 0u; k < count && state->tap_full == 0u; ++k )
		{
			const uint16_t *row = chain != 0u ? SparkStageRunnerTapRows(&state->runner, k, rows, &stride) + (uint64_t)first_row * stride :
				base + (uint64_t)order[begin + k] * stride;
			if ( StageServingTapWrite(state, row, (size_t)row_bytes) != 0 )
				state->tap_full = 1u;
		}
		if ( state->tap_full != 0u )
		{
			fprintf(stderr, "%s adapter: tap %s write failed: %s\n", state->model->module_tag, StageServingTapName(state), strerror(errno));
			StageServingTapClose(state);
			if ( state->tap_stream != 0 )
			{
				state->tap_full = 0u;
				state->tap_stream_down = 1u;
				state->tap_retry_ns = SparkMonotonicNs() + STAGE_SERVING_TAP_RETRY_NS;
			}
			return;
		}
		state->tap_written += bytes;
	}
	if ( state->tap_socket >= 0 && StageServingTapFlush(state) != 0 )
	{
		fprintf(stderr, "%s adapter: tap stream %s send failed after %llu bytes: %s; taps pause and the stream reconnects\n", state->model->module_tag,
			state->tap_stream, (unsigned long long)state->tap_written, strerror(errno));
		StageServingTapClose(state);
		state->tap_stream_down = 1u;
		state->tap_retry_ns = SparkMonotonicNs() + STAGE_SERVING_TAP_RETRY_NS;
	}
}

static SparkStatus StageServingParse(SparkStageServingState *state, const SparkModelServingAdapterConfiguration *configuration,
	SparkJsonDocument *doc, int32_t root)
{
	static const char *const members[] = { "stage_pack_path", "stray_working_set_path", "state_budget_bytes", "state_memory_bytes", "hidden",
		"tp_degree", "tp_rank", "world_size", "max_sequences", "max_rows", "max_shared_rows", "resident_capacity", "kv_pages", "taps",
		"streamed_prefill", "device_collective", "expert_pool_cap_bytes", "weights_share" };
	const SparkModelServingAdapterDescriptor *descriptor = &StageServingDescriptor;
	SparkStageRunnerConfiguration *config = &state->runner_config;
	uint32_t world_size = 0u, tp_rank = 0u, hidden = 0u;
	int32_t token;
	SparkStatus status;
	if ( !SparkJsonTokenIsType(doc, root, SPARK_JSON_TOKEN_OBJECT) )
		return SPARK_STATUS_SCHEMA_ERROR;
	status = StageServingKnownMembers(state, doc, root, "the stage config", members, sizeof(members) / sizeof(members[0]));
	if ( status != SPARK_STATUS_OK )
		return status;
	token = SparkJsonFindObjectMember(doc, root, "stage_pack_path");
	if ( token < 0 || SparkJsonCopyString(doc, token, &state->pack_path) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( StageServingCopyMember(doc, root, "stray_working_set_path", &state->stray_working_set_path) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = SparkJsonFindObjectMember(doc, root, "state_budget_bytes");
	if ( token < 0 || SparkJsonGetUInt64(doc, token, &config->state_budget_bytes) != SPARK_STATUS_OK ||
		config->state_budget_bytes == 0u )
	{
		fprintf(stderr, "%s adapter: state_budget_bytes must name the rank's state budget\n", state->model->module_tag);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	token = SparkJsonFindObjectMember(doc, root, "state_memory_bytes");
	if ( token >= 0 && (SparkJsonGetUInt64(doc, token, &state->state_memory_bytes) != SPARK_STATUS_OK || state->state_memory_bytes == 0u) )
	{
		fprintf(stderr, "%s adapter: state_memory_bytes must be a positive byte count\n", state->model->module_tag);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	status = StageServingU32Member(state, doc, root, "the stage config", "hidden", &hidden);
	if ( status == SPARK_STATUS_OK )
		status = StageServingU32Member(state, doc, root, "the stage config", "tp_degree", &config->tp_degree);
	if ( status == SPARK_STATUS_OK )
		status = StageServingU32Member(state, doc, root, "the stage config", "world_size", &world_size);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( hidden != descriptor->boundary_element_count || config->tp_degree != state->model->tp_degree || world_size != descriptor->stage_count ||
		world_size % config->tp_degree != 0u )
	{
		fprintf(stderr, "%s adapter: the stage config names hidden %u, tp_degree %u and world_size %u; this model is hidden %u at tp_degree %u over %u ranks\n",
			state->model->module_tag, hidden, config->tp_degree, world_size, descriptor->boundary_element_count, state->model->tp_degree,
			descriptor->stage_count);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	config->stage_index = configuration->stage_index / config->tp_degree;
	config->stage_count = world_size / config->tp_degree;
	config->tp_rank = configuration->stage_index % config->tp_degree;
	token = SparkJsonFindObjectMember(doc, root, "tp_rank");
	if ( token < 0 || SparkJsonGetUInt32(doc, token, &tp_rank) != SPARK_STATUS_OK || tp_rank != config->tp_rank )
	{
		fprintf(stderr, "%s adapter: tp_rank must name %u, the rank this config is loaded on (deployment rank %u at tp_degree %u)\n",
			state->model->module_tag, config->tp_rank, configuration->stage_index, config->tp_degree);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	status = StageServingU32Member(state, doc, root, "the stage config", "max_sequences", &config->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = StageServingU32Member(state, doc, root, "the stage config", "max_rows", &config->max_input_row_count);
	if ( status == SPARK_STATUS_OK )
		status = StageServingOptionalU32Member(state, doc, root, "the stage config", "max_shared_rows", config->max_input_row_count,
			&config->max_shared_row_count);
	if ( status == SPARK_STATUS_OK )
		status = StageServingU32Member(state, doc, root, "the stage config", "resident_capacity", &config->resident_sequence_capacity);
	if ( status == SPARK_STATUS_OK )
		status = StageServingU32Member(state, doc, root, "the stage config", "kv_pages", &config->kv_pages_per_sequence);
	if ( status == SPARK_STATUS_OK )
		status = StageServingParseWeights(state, doc, root);
	if ( status != SPARK_STATUS_OK )
		return status;
	config->kv_page_bytes = 0u;
	token = SparkJsonFindObjectMember(doc, root, "taps");
	if ( token >= 0 && StageServingParseTaps(state, doc, token) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = SparkJsonFindObjectMember(doc, root, "streamed_prefill");
	if ( token >= 0 && StageServingWeightdRefused(state, "streamed_prefill") != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( token >= 0 && StageServingParseStreamed(state, doc, token) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	token = SparkJsonFindObjectMember(doc, root, "device_collective");
	if ( token >= 0 && StageServingWeightdRefused(state, "device_collective") != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( token >= 0 )
		return StageServingLoadDeviceCollective(state, doc, token);
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingAcquireLane(SparkStageServingState *state)
{
	SparkTpDeviceCollectiveConfig *config = &state->device_config;
	SparkWeightdMeshTopology mesh_topology;
	uint32_t resolved_lane = SPARK_WEIGHTD_LANE_NONE;
	if ( SparkTpDeviceCollectiveMeshTopology(config->tp_rank, config->tp_degree, config->mesh_ranks, &mesh_topology) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( SparkWeightdClientConnect(state->weightd_socket_path, &state->lane_client, 0) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s adapter: cannot reach weightd at %s to acquire mesh lane %u\n", state->model->module_tag,
			state->weightd_socket_path, config->mesh_lane);
		state->lane_client = 0;
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	if ( SparkWeightdClientLaneAcquire(state->lane_client, config->mesh_lane, &mesh_topology, &resolved_lane,
		(uint64_t)config->connect_timeout_milli * 1000000ull) != SPARK_STATUS_OK || resolved_lane != config->mesh_lane )
	{
		fprintf(stderr, "%s adapter: weightd did not grant mesh lane %u\n", state->model->module_tag, config->mesh_lane);
		(void)SparkWeightdClientClose(state->lane_client);
		state->lane_client = 0;
		SPARK_FAIL(SPARK_STATUS_NO_LANE);
	}
	fprintf(stderr, "%s adapter: mesh lane %u acquired\n", state->model->module_tag, config->mesh_lane);
	config->mesh_lane_client = state->lane_client;
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingLoadConfiguration(SparkStageServingState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkStageRunnerConfiguration *config = &state->runner_config;
	SparkJsonDocument doc;
	SparkStatus status;
	memset(&doc, 0, sizeof(doc));
	memset(config, 0, sizeof(*config));
	config->abi_version = SPARK_STAGE_RUNNER_ABI_VERSION;
	config->descriptor_bytes = (uint32_t)sizeof(*config);
	if ( StageServingOwnPack(state->implementation_flags) == 0u )
	{
		status = StageServingWeightdHandoff(state);
		if ( status != SPARK_STATUS_OK )
			return status;
	}
	status = SparkJsonLoadFile(configuration->adapter_configuration_path, &doc);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = StageServingParse(state, configuration, &doc, SparkJsonGetRootToken(&doc));
	SparkJsonDocumentDestroy(&doc);
	if ( status != SPARK_STATUS_OK )
		return status;
	config->rank_pack_path = state->pack_path;
	config->linear_weight_codec = StageServingDescriptor.linear_weight_codec;
	config->stray_working_set_path = state->stray_working_set_path;
	config->execution_stream = configuration->execution_stream;
	config->multiprocessors = SPARK_STAGE_SERVING_MULTIPROCESSORS;
	if ( config->tp_degree > 1u && state->device_collective_present == 0 )
	{
		fprintf(stderr, "%s adapter: tp_degree %u needs a device_collective\n", state->model->module_tag, config->tp_degree);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( state->device_collective_present == 0 )
		return SPARK_STATUS_OK;
	state->device_config.tp_degree = config->tp_degree;
	state->device_config.tp_rank = config->tp_rank;
	state->device_config.weightd_socket_path = state->weightd_socket_path;
	state->device_config.registration_cuda_stream = configuration->execution_stream;
	status = StageServingAcquireLane(state);
	if ( status != SPARK_STATUS_OK )
		return status;
	config->device_collective = &state->device_config;
	return SPARK_STATUS_OK;
}

static void StageServingDestroy(void *adapter_state);

static SparkStatus StageServingRecurrentCopy(void *context, uint32_t direction,
	uint32_t slot, void *buffer, uint64_t bytes, SparkDeviceQueue *queue)
{
	SparkStageServingState *state = (SparkStageServingState *)context;
	if ( state == 0 || (queue != 0 && queue != SparkStageRunnerQueue(&state->runner)) ||
		(direction != SPARK_STAGE_KV_RECURRENT_TO_BUFFER && direction != SPARK_STAGE_KV_RECURRENT_FROM_BUFFER) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return SparkStageRunnerRecurrentCopy(&state->runner,
		direction == SPARK_STAGE_KV_RECURRENT_TO_BUFFER ? 1u : 0u,
		slot, buffer, bytes, queue);
}

static SparkStatus StageServingDriverAdmit(void *driver_instance,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkStageServingState *state = (SparkStageServingState *)driver_instance;
	return SparkStageKvBindingAdmit(&state->kv, request, decision);
}

static const SparkModelDriverInterface StageServingDriver =
{
	.abi_version = SPARK_MODEL_DRIVER_ABI_VERSION,
	.interface_bytes = (uint32_t)sizeof(SparkModelDriverInterface),
	.admit = StageServingDriverAdmit,
};

static SparkStatus StageServingBindKv(SparkStageServingState *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	const SparkStageServingModel *model = state->model;
	const SparkModelServingAdapterDescriptor *descriptor = &StageServingDescriptor;
	SparkStageKvConfiguration kv;
	SparkStageRunnerKv attach;
	SparkStatus status;
	memset(&kv, 0, sizeof(kv));
	kv.module_tag = model->module_tag;
	kv.block_token_count = descriptor->cache_block_token_count;
	kv.region_count = 1u;
	model->kv_layout(state->runner_config.tp_rank, state->runner_config.tp_degree, &kv);
	if ( kv.region_count != SPARK_STAGE_KV_MAX_REGIONS )
		kv.region_count = 1u;
	kv.regions[0].layer_count = SparkStageRunnerKvLayerCount(&state->runner);
	kv.capacity_request.layer_count = kv.regions[0].layer_count;
	kv.model_id = descriptor->model_id;
	kv.model_revision = descriptor->model_revision;
	kv.resident_sequence_capacity = state->runner_config.resident_sequence_capacity;
	kv.max_sequence_positions = state->runner_config.kv_pages_per_sequence * descriptor->cache_block_token_count;
	kv.max_input_row_count = state->max_rows;
	kv.logical_page_count = configuration->runtime_limits.kv_logical_page_capacity;
	kv.physical_page_count = configuration->runtime_limits.kv_physical_page_capacity;
	kv.pipeline_slot_count = 1u;
	kv.pool_source = StageServingOwnPack(state->implementation_flags) != 0u ? SPARK_STAGE_KV_POOL_SOURCE_DEVICE : SPARK_STAGE_KV_POOL_SOURCE_WEIGHTD;
	kv.weightd_socket_path = kv.pool_source == SPARK_STAGE_KV_POOL_SOURCE_WEIGHTD ? state->weightd_socket_path : 0;
	kv.backing_directory = configuration->kv_backing_directory;
	kv.backing_maximum_bytes = configuration->kv_backing_maximum_bytes;

	kv.snapshot_directory = configuration->kv_snapshot_directory;
	kv.ledger_path = configuration->kv_ledger_path;
	status = SparkStageRunnerPackIdentity(&state->runner, kv.pack_sha256, sizeof(kv.pack_sha256));
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "%s kv binding refused: the runner reports no pack identity, so the KV layout has none\n",
			model->module_tag);
		SPARK_RETURN(status);
	}
	if ( SparkSha256HexToDigest(descriptor->artifact_sha256, kv.contract_sha256) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_HASH_MISMATCH);
	kv.expert_codec = descriptor->expert_weight_codec;
	kv.kv_codec = descriptor->kv_cache_codec;
	kv.driver_symbol = (const void *)&StageServingBindKv;
	kv.recurrent.lane_bytes = SparkStageRunnerRecurrentBytes(&state->runner);
	if ( kv.recurrent.lane_bytes != 0u )
	{
		kv.recurrent.copy = StageServingRecurrentCopy;
		kv.recurrent.context = state;
		kv.recurrent.checkpoint_token_count = descriptor->cache_checkpoint_token_count;
		kv.recurrent.memory_bytes = state->state_memory_bytes;
	}
	status = SparkStageKvBindingInitialize(&state->kv, &kv);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	memset(&attach, 0, sizeof(attach));
	attach.pool = state->kv.region_base[0];
	attach.layer_stride_bytes = state->kv.region_layer_stride_bytes[0];
	attach.layer_page_bytes = kv.regions[0].layer_page_bytes;
	attach.layer_count = kv.regions[0].layer_count;
	attach.page_table = state->kv.page_table;
	attach.page_table_stride = state->kv.pages_per_sequence;
	attach.pool_page_count = SparkStageKvBindingAddressablePageCount(&state->kv);
	attach.sequence_count = state->kv.resident_sequence_capacity;
	attach.context_shard = state->kv.context_shard;
	if ( kv.region_count > 1u )
	{
		attach.second_pool = state->kv.region_base[1];
		attach.second_layer_stride_bytes = state->kv.region_layer_stride_bytes[1];
		attach.second_layer_page_bytes = kv.regions[1].layer_page_bytes;
		attach.second_layer_count = kv.regions[1].layer_count;
	}
	return SparkStageRunnerAttachKv(&state->runner, &attach);
}

static SparkServingCacheAdmission StageServingCacheContext(SparkStageServingState *state,
	SparkModelDriverCacheLane *lanes)
{
	SparkServingCacheAdmission cache;
	memset(&cache, 0, sizeof(cache));
	cache.program_id = STAGE_SERVING_PROGRAM_ID;
	cache.lane_capacity = SPARK_STAGE_SERVING_MAX_LANES;
	cache.lanes = lanes;
	cache.driver = &StageServingDriver;
	cache.driver_instance = state;
	cache.adapter_state = state;
	return cache;
}

typedef struct SparkStageServingContinuity
{
	SparkStageServingState *state;
	uint32_t rows;
	uint32_t active;
	const uint32_t *row_slots;
	const uint64_t *row_sequence_ids;
	const uint64_t *row_positions;
	uint8_t *bound;
	uint64_t *sequence_ids;
	uint64_t *next_positions;
} SparkStageServingContinuity;

static SparkStatus StageServingPrepareContinuity(void *context)
{
	SparkStageServingContinuity *continuity = (SparkStageServingContinuity *)context;
	return SparkStageKvBindingContinuity(&continuity->state->kv, continuity->state->lane_states,
		continuity->rows, continuity->active, continuity->row_slots, continuity->row_sequence_ids,
		continuity->row_positions, continuity->state->tree_frame != 0u ? continuity->state->lane_contexts : 0, continuity->bound, continuity->sequence_ids,
		continuity->next_positions);
}

static void StageServingPublishCompletion(void *context, const SparkModelDriverCompletion *completion)
{
	SparkStageServingState *state = (SparkStageServingState *)context;
	state->publish_completion = *completion;
	state->publish_completed = 1u;
}

static void StageServingBuildFrame(const SparkStageServingState *state, const SparkModelServingSubmission *submission,
	SparkModelDriverCacheLane *lanes, uint32_t lane_count, SparkModelDriverFrame *frame)
{
	memset(frame, 0, sizeof(*frame));
	frame->program_id = STAGE_SERVING_PROGRAM_ID;
	frame->request_id = submission->request_id;
	frame->sequence_id = submission->sequence_id;
	frame->sequence_position = submission->sequence_position;
	frame->deadline_time_ns = submission->deadline_time_ns;
	frame->active_slot_count = submission->active_sequence_count;
	frame->new_token_count = submission->row_count;
	frame->tokens_per_sequence = submission->tokens_per_sequence;
	frame->priority = submission->priority;
	frame->flags = SparkModelServingSubmissionFrameFlags(submission);
	frame->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	frame->execution_stream = state->runner_config.execution_stream;
	frame->residency = submission->residency;
	frame->cache_lane_count = lane_count;
	frame->cache_lanes = lanes;
}

static SparkStatus StageServingAdmitFrame(SparkStageServingState *state,
	const SparkModelServingSubmission *submission, SparkModelDriverFrame *frame)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	uint32_t lane_count = 0u;
	SparkStatus status;
	status = SparkModelServingAdapterBuildDriverCacheLanes(submission, state->cache_lanes,
		SPARK_STAGE_SERVING_MAX_LANES, &lane_count);
	if ( status == SPARK_STATUS_OK && lane_count != submission->active_sequence_count )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkAdmissionRequestFromSubmission(STAGE_SERVING_PROGRAM_ID, submission,
			state->cache_lanes, 0u, &request);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	StageServingBuildFrame(state, submission, state->cache_lanes, lane_count, frame);
	return SparkAdmissionEvaluateAndApply(&StageServingDriver, state, &request, frame, &decision);
}

static void StageServingBufferPlan(SparkStageServingState *state, SparkStageServingBuffer *plan)
{
	const uint64_t rows = state->max_rows, lanes = state->runner_config.max_active_sequence_count;
	const SparkStageServingBuffer plans[STAGE_SERVING_BUFFERS] =
	{
		{ &state->positions_host, SPARK_DEVICE_SPACE_HOST_PINNED, rows * 4u },
		{ &state->context_host, SPARK_DEVICE_SPACE_HOST_PINNED, lanes * 4u },
		{ &state->state_host, SPARK_DEVICE_SPACE_HOST_PINNED, rows * 4u },
		{ &state->positions_device, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, rows * 4u },
		{ &state->context_device, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, lanes * 4u },
		{ &state->state_device, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, rows * 4u },
		{ &state->runs_host, SPARK_DEVICE_SPACE_HOST_PINNED, (rows + 1u) * sizeof(uint32_t) },
		{ &state->runs_device, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, (rows + 1u) * sizeof(uint32_t) },
		{ &state->seqslot_host, SPARK_DEVICE_SPACE_HOST_PINNED, rows * sizeof(uint32_t) },
		{ &state->target_host, SPARK_DEVICE_SPACE_HOST_PAGEABLE, rows * sizeof(uint32_t) },
		{ &state->seqslot_device, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->order_host, SPARK_DEVICE_SPACE_HOST_PINNED, rows * sizeof(uint32_t) },
		{ &state->order_device, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->slots_host, SPARK_DEVICE_SPACE_HOST_PINNED, rows * sizeof(uint32_t) },
		{ &state->slots_device, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->fork_host, SPARK_DEVICE_SPACE_HOST_PINNED, rows * sizeof(uint32_t) },
		{ &state->fork_device, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->offset_host, SPARK_DEVICE_SPACE_HOST_PINNED, rows * sizeof(uint32_t) },
		{ &state->offset_device, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->parent_host, SPARK_DEVICE_SPACE_HOST_PINNED, rows * sizeof(uint32_t) },
		{ &state->parent_device, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->loads_host, SPARK_DEVICE_SPACE_HOST_PINNED, rows * sizeof(uint32_t) },
		{ &state->loads_device, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->saves_host, SPARK_DEVICE_SPACE_HOST_PINNED, rows * sizeof(uint32_t) },
		{ &state->saves_device, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->output_tokens, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->output_scores, SPARK_DEVICE_SPACE_DEVICE_PRIVATE, rows * sizeof(uint32_t) },
		{ &state->tokens_host, SPARK_DEVICE_SPACE_HOST_PINNED, (uint64_t)SPARK_STAGE_SERVING_MAX_ROWS * sizeof(uint32_t) },
	};
	memcpy(plan, plans, sizeof(plans));
}

static SparkStatus StageServingAllocate(SparkStageServingState *state)
{
	SparkStageServingBuffer plan[STAGE_SERVING_BUFFERS];
	SparkStatus status = SparkDeviceInitialize(0);
	if ( status == SPARK_STATUS_DUPLICATE )
		status = SPARK_STATUS_OK;
	StageServingBufferPlan(state, plan);
	for ( uint32_t index = 0u; status == SPARK_STATUS_OK && index < STAGE_SERVING_BUFFERS; ++index )
		status = SparkDeviceAllocate(plan[index].buffer, plan[index].space, plan[index].bytes, state->model->module_tag);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "%s adapter: device %s failed: %s\n", state->model->module_tag, SparkDeviceLastError()->operation,
			SparkDeviceLastError()->text);
	return status;
}

static SparkStatus StageServingInitialize(const SparkModelServingAdapterConfiguration *configuration, void **adapter_state)
{
	const SparkStageServingModel *model = StageServingBoundModel;
	SparkStageServingState *state;
	SparkStatus status;
	if ( configuration == 0 || adapter_state == 0 || model == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state = (SparkStageServingState *)calloc(1u, sizeof(*state));
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->model = model;
	state->implementation_flags = SparkStageRunnerImplementationFlags();
	state->tap_socket = -1;
	SparkStageModuleAtomicStateArrayInitialize(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES);
	atomic_init(&state->reset_generation, 0u);
	status = StageServingLoadConfiguration(state, configuration);
	if ( status != SPARK_STATUS_OK )
		{ StageServingDestroy(state); return status; }
	state->max_rows = state->runner_config.max_input_row_count;
	if ( state->max_rows == 0u || state->max_rows > StageServingDescriptor.max_input_row_count ||
		state->runner_config.max_active_sequence_count == 0u ||
		state->runner_config.max_active_sequence_count > StageServingDescriptor.max_active_sequence_count )
		{ StageServingDestroy(state); SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED); }
	state->max_shared_rows = state->runner_config.max_shared_row_count;
	if ( state->max_shared_rows == 0u || state->max_shared_rows > state->max_rows ||
		state->max_shared_rows < state->runner_config.max_active_sequence_count ||
		state->max_shared_rows < (StageServingDescriptor.max_verify_row_count < state->max_rows ? StageServingDescriptor.max_verify_row_count : state->max_rows) )
	{
		fprintf(stderr, "%s adapter: max_shared_rows %u must cover the sequences (%u) and verify rows (%u) and stay within max_rows %u\n",
			model->module_tag, state->max_shared_rows, state->runner_config.max_active_sequence_count,
			StageServingDescriptor.max_verify_row_count, state->max_rows);
		StageServingDestroy(state);
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	state->runner_config.head_row_capacity = state->runner_config.max_active_sequence_count > StageServingDescriptor.max_verify_row_count ?
		state->runner_config.max_active_sequence_count : StageServingDescriptor.max_verify_row_count;
	if ( state->runner_config.head_row_capacity > state->max_rows )
		state->runner_config.head_row_capacity = state->max_rows;
	state->runner_config.output_vocabulary = configuration->runtime_limits.output_vocabulary;
	fprintf(stderr, "%s adapter: rows=%u shared_rows=%u head_rows=%u\n", model->module_tag, state->max_rows,
		state->max_shared_rows, state->runner_config.head_row_capacity);
	state->completion_function = configuration->completion_function;
	state->completion_context = configuration->completion_context;
	status = StageServingAllocate(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageRunnerInitialize(&state->runner, &state->runner_config, model->runner_model());
	if ( status == SPARK_STATUS_OK && model->tree_slot_grain != 0u && model->tree_slot_grain != SparkStageRunnerTreeSlotGrain(&state->runner) )
	{
		fprintf(stderr, "%s: the model's tree slot grain %u is not the runner's %u\n", model->module_tag, model->tree_slot_grain,
			SparkStageRunnerTreeSlotGrain(&state->runner));
		status = SPARK_STATUS_INVALID_ARGUMENT;
	}
	if ( status == SPARK_STATUS_OK )
		status = StageServingCheckOwnPack(state);
	if ( status == SPARK_STATUS_OK )
		status = StageServingOpenTaps(state);
	if ( status == SPARK_STATUS_OK )
		status = StageServingBindKv(state, configuration);
	if ( status != SPARK_STATUS_OK )
		{ StageServingDestroy(state); return status; }
	*adapter_state = state;
	return SPARK_STATUS_OK;
}

static void StageServingDestroy(void *adapter_state)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	SparkStageServingBuffer plan[STAGE_SERVING_BUFFERS];
	if ( state == 0 )
		return;
	SparkStageKvBindingDestroy(&state->kv);
	StageServingTapClose(state);
	free(state->tap_buffer);
	state->tap_buffer = 0;
	free(state->tap_file);
	state->tap_file = 0;
	free(state->tap_stream);
	state->tap_stream = 0;
	SparkStageRunnerDestroy(&state->runner);
	if ( state->lane_client != 0 )
	{
		(void)SparkWeightdClientClose(state->lane_client);
		state->lane_client = 0;
	}
	StageServingBufferPlan(state, plan);
	for ( uint32_t index = 0u; index < STAGE_SERVING_BUFFERS; ++index )
		if ( plan[index].buffer->handle != 0u )
			(void)SparkDeviceFree(plan[index].buffer);
	free(state->pack_path);
	free(state->stray_working_set_path);
	free(state->weightd_socket_path);
	free(state->streamed_partner_pack);
	free(state->streamed_fabric_device);
	free(state->streamed_pair_device);
	free(state);
}

static SparkStatus StageServingReleaseSlots(const SparkStageServingState *state,
	const SparkModelServingSubmission *submission, uint32_t *slots)
{
	uint32_t count = submission->active_sequence_count;
	if ( submission->row_count != 0u ||
		count > SPARK_STAGE_SERVING_MAX_LANES ||
		count > state->runner_config.max_active_sequence_count ||
		(count != 0u && (submission->lanes == 0 || submission->lane_count < count)) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for ( uint32_t lane = 0u; lane < count; ++lane )
	{
		uint32_t slot = submission->lanes[lane].resident_sequence_slot;
		if ( slot >= state->runner_config.max_active_sequence_count )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t earlier = 0u; earlier < lane; ++earlier )
			if ( submission->lanes[earlier].resident_sequence_slot == slot )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( slots != 0 )
			slots[lane] = slot;
	}
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingValidateSubmission(void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	if ( state == 0 || submission == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( submission->control_generation < atomic_load_explicit(&state->reset_generation, memory_order_acquire) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkModelServingWorkKindUsesRows(submission->work_kind) == 0u )
	{
		if ( submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE &&
			submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		return StageServingReleaseSlots(state, submission, 0);
	}
	if ( submission->row_count == 0u || submission->row_count > state->max_rows )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( submission->row_sequence_ids == 0 || submission->lanes == 0 || submission->lane_count < submission->active_sequence_count )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( submission->row_count > state->max_shared_rows &&
		(submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_PREFILL || submission->active_sequence_count != 1u) )
	{
		fprintf(stderr, "%s adapter: refused a %u-row wave of %u sequences (kind %u): only a single-sequence prefill may exceed %u rows\n",
			state->model->module_tag, submission->row_count, submission->active_sequence_count, submission->work_kind, state->max_shared_rows);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return SPARK_STATUS_OK;
}

static void StageServingCompletionHeader(const SparkModelServingSubmission *submission,
	uint32_t accepted_token_count, SparkModelServingCompletion *completion)
{
	memset(completion, 0, sizeof(*completion));
	completion->abi_version = submission->abi_version;
	completion->descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion->submission_id = submission->submission_id;
	completion->request_id = submission->request_id;
	completion->sequence_id = submission->sequence_id;
	completion->sequence_position = submission->sequence_position;
	completion->control_generation = submission->control_generation;
	completion->transaction_id = submission->transaction_id;
	completion->dispatch_generation = submission->dispatch_generation;
	completion->request_generation = submission->request_generation;
	completion->step_generation = submission->step_generation;
	completion->residency = submission->residency;
	completion->accepted_token_count = accepted_token_count;
}

static SparkStatus StageServingGroupRows(SparkStageServingState *state,
	const SparkModelServingSubmission *submission, uint32_t rows,
	uint32_t *active_out)
{
	uint32_t *positions = StageServingWords(&state->positions_host);
	uint32_t *context = StageServingWords(&state->context_host);
	uint32_t *slot_of_row = StageServingWords(&state->state_host);
	uint32_t *begin = StageServingWords(&state->runs_host);
	uint32_t *order = StageServingWords(&state->order_host);
	uint32_t *slots = StageServingWords(&state->seqslot_host);
	uint32_t *target = StageServingWords(&state->target_host);
	uint32_t cursor[SPARK_STAGE_SERVING_MAX_LANES];
	uint32_t capacity = state->runner_config.max_active_sequence_count;
	uint32_t lanes = submission->lanes != 0 ? submission->active_sequence_count : rows;
	uint32_t lane, i;
	if ( rows == 0u || rows > state->max_rows ||
		lanes == 0u || lanes > rows || lanes > capacity ||
		lanes > SPARK_STAGE_SERVING_MAX_LANES ||
		submission->row_positions == 0 )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	memset(context, 0, (uint64_t)capacity * sizeof(uint32_t));
	memset(begin, 0, ((uint64_t)lanes + 1u) * sizeof(uint32_t));
	for ( lane = 0u; lane < lanes; ++lane )
	{
		slots[lane] = submission->lanes != 0 ? submission->lanes[lane].resident_sequence_slot : lane;
		target[lane] = submission->lanes != 0 ? submission->lanes[lane].state_target_token_count : 0u;
		if ( slots[lane] >= capacity )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		for ( i = 0u; i < lane; ++i )
			if ( slots[i] == slots[lane] )
				SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	for ( i = 0u; i < rows; ++i )
	{
		lane = submission->lanes != 0 && submission->row_lane_indices != 0
			? submission->row_lane_indices[i] : i;
		if ( lane >= lanes || submission->row_positions[i] >= UINT32_MAX )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		positions[i] = (uint32_t)submission->row_positions[i];
		slot_of_row[i] = slots[lane];
		if ( positions[i] + 1u > context[slots[lane]] )
			context[slots[lane]] = positions[i] + 1u;
		begin[lane + 1u]++;
	}
	for ( lane = 0u; lane < lanes; ++lane )
	{
		if ( begin[lane + 1u] == 0u )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		begin[lane + 1u] += begin[lane];
		cursor[lane] = begin[lane];
	}
	for ( i = 0u; i < rows; ++i )
	{
		lane = submission->lanes != 0 && submission->row_lane_indices != 0
			? submission->row_lane_indices[i] : i;
		order[cursor[lane]++] = i;
	}
	state->tree_frame = 0u;
	if ( submission->lanes != 0 && (submission->flags & SPARK_MODEL_SERVING_SUBMISSION_FLAG_TREE) != 0u )
	{
		SparkSpeculationTreeLayout layout;
		uint32_t *slot_table = StageServingWords(&state->slots_host), *fork = StageServingWords(&state->fork_host);
		uint32_t *offset = StageServingWords(&state->offset_host), *parent = StageServingWords(&state->parent_host);
		uint32_t *loads = StageServingWords(&state->loads_host), *saves = StageServingWords(&state->saves_host);
		if ( lanes != 1u || submission->row_parent_rows == 0 || submission->row_slots == 0 || submission->token_ids == 0 || rows > SPARK_SPECULATION_TREE_FRAME_MAX_ROWS ||
			SparkSpeculationTreeLayoutBuild(submission->row_parent_rows, submission->token_ids, rows, (uint32_t)submission->lanes[0].sequence_position,
				state->model->tree_slot_grain, &layout) != SPARK_STATUS_OK ||
			SparkSpeculationTreeLayoutMatches(&layout, submission->row_positions, submission->row_slots, rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		for ( i = 0u; i < rows; ++i )
		{
			slot_table[i] = layout.row[i].slot;
			fork[i] = layout.row[i].fork_position;
			offset[i] = layout.row[i].slot_offset;
			parent[i] = submission->row_parent_rows[i];
			loads[i] = i != 0u && layout.row[i].branch != layout.row[submission->row_parent_rows[i]].branch ? layout.row[i].branch : 0u;
			saves[i] = 0u;
		}
		for ( i = 1u; i < rows; ++i )
			if ( loads[i] != 0u )
				saves[submission->row_parent_rows[i]] |= 1u << (loads[i] - 1u);
		context[slots[0]] = layout.context;
		state->tree_frame = 1u;
	}
	*active_out = lanes;
	return SPARK_STATUS_OK;
}

static void StageServingPublishHost(SparkStageServingState *state, uint64_t elapsed_ns, uint32_t lanes)
{
	state->publish_host_ns += elapsed_ns;
	state->publish_host_lanes += lanes;
	if ( ++state->publish_host_count < 16u )
		return;
	fprintf(stderr, "STAGE-PUBLISH-HOST tag=%s rank=%u publish_frames=%u mean_lanes=%.1f publish_ms=%.2f ms_per_lane=%.2f\n",
		state->model->module_tag, state->runner_config.tp_rank, state->publish_host_count,
		(double)state->publish_host_lanes / state->publish_host_count, (double)state->publish_host_ns / state->publish_host_count / 1e6,
		state->publish_host_lanes != 0u ? (double)state->publish_host_ns / state->publish_host_lanes / 1e6 : 0.0);
	state->publish_host_ns = 0u;
	state->publish_host_lanes = 0u;
	state->publish_host_count = 0u;
}

static SparkStatus StageServingSubmitNonRow(SparkStageServingState *state,
	const SparkModelServingSubmission *submission)
{
	SparkModelServingCompletion completion;
	SparkModelDriverFrame frame;
	uint32_t slots[SPARK_STAGE_SERVING_MAX_LANES];
	SparkStatus status;
	status = StageServingReleaseSlots(state, submission, slots);
	if ( status == SPARK_STATUS_OK )
		status = StageServingAdmitFrame(state, submission, &frame);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_CACHE_PUBLISH )
	{
		const uint64_t publish_started = SparkMonotonicNs();
		frame.completion_function = StageServingPublishCompletion;
		frame.completion_context = state;
		state->publish_completed = 0u;
		status = SparkStageKvBindingPublishFrame(&state->kv, &frame, state->lane_states);
		if ( status == SPARK_STATUS_OK && (state->publish_completed == 0u || state->publish_completion.status != SPARK_STATUS_OK) )
			status = state->publish_completed == 0u ? SPARK_STATUS_INTERNAL_ERROR : state->publish_completion.status;
		if ( status == SPARK_STATUS_OK )
			StageServingPublishHost(state, SparkMonotonicNs() - publish_started, submission->active_sequence_count);
	}
	else
		status = SparkStageRunnerResetSlots(&state->runner, slots, submission->active_sequence_count);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( state->completion_function != 0 )
	{
		StageServingCompletionHeader(submission, 0u, &completion);
		state->completion_function(state->completion_context, &completion);
	}
	return SPARK_STATUS_OK;
}

static void StageServingContinuityRows(SparkStageServingState *state,
	const SparkModelServingSubmission *submission, uint32_t rows, uint32_t active,
	uint32_t *row_slots, uint64_t *row_sequence_ids, uint64_t *row_positions)
{
	const uint32_t *begin = StageServingWords(&state->runs_host);
	const uint32_t *order = StageServingWords(&state->order_host);
	const uint32_t *slots = StageServingWords(&state->seqslot_host);
	const uint32_t *slot_table = StageServingWords(&state->slots_host);
	const uint32_t *context = StageServingWords(&state->context_host);
	uint32_t lane, cursor, out = active, row;
	for ( lane = 0u; lane < active; ++lane )
	{
		row = order[begin[lane]];
		row_slots[lane] = slots[lane];
		row_sequence_ids[lane] = submission->row_sequence_ids[row];
		row_positions[lane] = state->tree_frame != 0u ? slot_table[row] : submission->row_positions[row];
		state->lane_contexts[lane] = state->tree_frame != 0u ? context[slots[lane]] : 0u;
	}
	for ( lane = 0u; lane < active; ++lane )
		for ( cursor = begin[lane] + 1u; cursor < begin[lane + 1u] && out < rows; ++cursor, ++out )
		{
			row = order[cursor];
			row_slots[out] = slots[lane];
			row_sequence_ids[out] = submission->row_sequence_ids[row];
			row_positions[out] = state->tree_frame != 0u ? slot_table[row] : submission->row_positions[row];
		}
}

static SparkStatus StageServingFinish(SparkStageServingState *state, uint32_t active,
	const uint32_t *lane_slots, uint8_t *bound, uint64_t *sequence_ids,
	uint64_t *next_positions, uint32_t extra_tokens, uint32_t verify, SparkStatus status)
{
	SparkStageKvBindingCompletion completion;
	memset(&completion, 0, sizeof(completion));
	completion.lane_count = active;
	completion.extra_tokens = status == SPARK_STATUS_OK ? extra_tokens : 0u;
	completion.verify = verify;
	completion.status = status;
	completion.resident_slots = lane_slots;
	completion.bound = bound;
	completion.sequence_ids = sequence_ids;
	completion.next_positions = next_positions;
	return SparkStageKvBindingFinishWait(&state->kv, 0u, &completion);
}

static SparkStatus StageServingChainSteps(const SparkStageServingState *state,
	const SparkModelServingSubmission *submission, const SparkModelDriverFrame *frame,
	uint32_t rows, uint32_t active, uint32_t *steps_out)
{
	const uint32_t *positions = StageServingWords(&state->positions_host);
	const uint32_t block = StageServingDescriptor.cache_block_token_count;
	uint32_t steps = submission->tokens_per_sequence > 1u ? submission->tokens_per_sequence : 1u, index;
	if ( (submission->flags & SPARK_MODEL_SERVING_SUBMISSION_FLAG_VERIFY) != 0u )
	{
		*steps_out = 1u;
		if ( state->runner.owns_final_head == 0u || SparkStageRunnerVerifyCapable(&state->runner) == 0u ||
			(uint64_t)active * submission->tokens_per_sequence > SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		if ( state->tree_frame != 0u && (active != 1u || SparkStageRunnerTreeVerifyCapable(&state->runner) == 0u) )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		for ( index = 0u; index < frame->cache_lane_count; ++index )
			if ( (frame->cache_lanes[index].flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_VERIFY) == 0u ||
				(frame->cache_lanes[index].flags & (SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX | SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH)) != 0u )
				SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		return SPARK_STATUS_OK;
	}
	*steps_out = steps;
	if ( steps == 1u )
		return SPARK_STATUS_OK;
	if ( submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_DECODE || rows != active ||
		(uint64_t)rows * steps > state->max_rows || state->runner.owns_final_head == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	for ( index = 0u; index < frame->cache_lane_count; ++index )
		if ( (frame->cache_lanes[index].flags & (SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX | SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH)) != 0u )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	for ( index = 0u; index < rows; ++index )
		if ( (positions[index] % block) + steps > block )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	return SPARK_STATUS_OK;
}

static void StageServingWaveHost(SparkStageServingState *state, const uint64_t *marks)
{
	uint32_t phase;
	for ( phase = 0u; phase < 5u; ++phase )
		state->wave_host_ns[phase] += marks[phase + 1u] - marks[phase];
	if ( ++state->wave_host_count < 16u )
		return;
	fprintf(stderr, "STAGE-WAVE-HOST tag=%s rank=%u prefill_waves=%u admit_ms=%.1f submit_ms=%.1f capture_ms=%.1f sync_ms=%.1f finish_ms=%.1f\n",
		state->model->module_tag, state->runner_config.tp_rank, state->wave_host_count,
		(double)state->wave_host_ns[0] / state->wave_host_count / 1e6, (double)state->wave_host_ns[1] / state->wave_host_count / 1e6,
		(double)state->wave_host_ns[2] / state->wave_host_count / 1e6, (double)state->wave_host_ns[3] / state->wave_host_count / 1e6,
		(double)state->wave_host_ns[4] / state->wave_host_count / 1e6);
	memset(state->wave_host_ns, 0, sizeof(state->wave_host_ns));
	state->wave_host_count = 0u;
}

static void StageServingChainHost(SparkStageServingState *state, const uint64_t *marks, uint32_t steps)
{
	uint32_t phase;
	for ( phase = 0u; phase < 5u; ++phase )
		state->chain_host_ns[phase] += marks[phase + 1u] - marks[phase];
	state->chain_host_steps += steps;
	if ( ++state->chain_host_count < 64u )
		return;
	fprintf(stderr, "STAGE-CHAIN-HOST tag=%s rank=%u chains=%u mean_steps=%.1f admit_ms=%.2f submit_ms=%.2f capture_ms=%.2f sync_ms=%.2f finish_ms=%.2f\n",
		state->model->module_tag, state->runner_config.tp_rank, state->chain_host_count, (double)state->chain_host_steps / state->chain_host_count,
		(double)state->chain_host_ns[0] / state->chain_host_count / 1e6, (double)state->chain_host_ns[1] / state->chain_host_count / 1e6,
		(double)state->chain_host_ns[2] / state->chain_host_count / 1e6, (double)state->chain_host_ns[3] / state->chain_host_count / 1e6,
		(double)state->chain_host_ns[4] / state->chain_host_count / 1e6);
	memset(state->chain_host_ns, 0, sizeof(state->chain_host_ns));
	state->chain_host_steps = 0u;
	state->chain_host_count = 0u;
}

static void StageServingVerifyHost(SparkStageServingState *state, const uint64_t *marks, uint64_t keep_ns, uint32_t rows)
{
	static const char *const names[STAGE_SERVING_VERIFY_BUCKETS] = { "2-4", "5-8", "9-16", "17+" };
	const uint32_t bucket = rows <= 4u ? 0u : rows <= 8u ? 1u : rows <= 16u ? 2u : 3u;
	uint64_t *ns = state->verify_host_ns[bucket];
	uint32_t phase;
	for ( phase = 0u; phase < 4u; ++phase )
		ns[phase] += marks[phase + 1u] - marks[phase];
	ns[4] += keep_ns;
	ns[5] += marks[5] - marks[4] - keep_ns;
	state->verify_host_rows[bucket] += rows;
	if ( ++state->verify_host_count[bucket] < 16u )
		return;
	fprintf(stderr, "STAGE-VERIFY-HOST tag=%s rank=%u rows=%s frames=%u mean_rows=%.1f admit_ms=%.2f submit_ms=%.2f capture_ms=%.2f sync_ms=%.2f keep_ms=%.2f finish_ms=%.2f\n",
		state->model->module_tag, state->runner_config.tp_rank, names[bucket], state->verify_host_count[bucket],
		(double)state->verify_host_rows[bucket] / state->verify_host_count[bucket],
		(double)ns[0] / state->verify_host_count[bucket] / 1e6, (double)ns[1] / state->verify_host_count[bucket] / 1e6,
		(double)ns[2] / state->verify_host_count[bucket] / 1e6, (double)ns[3] / state->verify_host_count[bucket] / 1e6,
		(double)ns[4] / state->verify_host_count[bucket] / 1e6, (double)ns[5] / state->verify_host_count[bucket] / 1e6);
	memset(ns, 0, sizeof(state->verify_host_ns[bucket]));
	state->verify_host_rows[bucket] = 0u;
	state->verify_host_count[bucket] = 0u;
}

static SparkStatus StageServingCopyInputs(SparkStageServingState *state, SparkDeviceQueue *queue, uint32_t rows, uint32_t active)
{
	const uint64_t row_bytes = (uint64_t)rows * sizeof(uint32_t), lane_bytes = (uint64_t)active * sizeof(uint32_t);
	const struct { SparkDeviceBuffer *device; SparkDeviceBuffer *host; uint64_t bytes; } inputs[] =
	{
		{ &state->positions_device, &state->positions_host, row_bytes },
		{ &state->context_device, &state->context_host, (uint64_t)state->runner_config.max_active_sequence_count * sizeof(uint32_t) },
		{ &state->state_device, &state->state_host, row_bytes },
		{ &state->runs_device, &state->runs_host, lane_bytes + sizeof(uint32_t) },
		{ &state->seqslot_device, &state->seqslot_host, lane_bytes },
		{ &state->order_device, &state->order_host, row_bytes },
	};
	const struct { SparkDeviceBuffer *device; SparkDeviceBuffer *host; } tree_inputs[] =
	{
		{ &state->slots_device, &state->slots_host },
		{ &state->fork_device, &state->fork_host },
		{ &state->offset_device, &state->offset_host },
		{ &state->parent_device, &state->parent_host },
		{ &state->loads_device, &state->loads_host },
		{ &state->saves_device, &state->saves_host },
	};
	for ( uint32_t index = 0u; index < sizeof(inputs) / sizeof(inputs[0]); ++index )
	{
		const SparkDeviceRange destination = { inputs[index].device, 0u }, source = { inputs[index].host, 0u };
		if ( SparkDeviceCopy(queue, destination, source, inputs[index].bytes) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	for ( uint32_t index = 0u; state->tree_frame != 0u && index < sizeof(tree_inputs) / sizeof(tree_inputs[0]); ++index )
	{
		const SparkDeviceRange destination = { tree_inputs[index].device, 0u }, source = { tree_inputs[index].host, 0u };
		if ( SparkDeviceCopy(queue, destination, source, row_bytes) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	return SPARK_STATUS_OK;
}

static void StageServingDispatch(const SparkStageServingState *state, const SparkModelServingSubmission *submission,
	uint32_t rows, uint32_t active, uint32_t steps, SparkStageRunnerDispatch *dispatch)
{
	memset(dispatch, 0, sizeof(*dispatch));
	dispatch->abi_version = SPARK_STAGE_RUNNER_ABI_VERSION;
	dispatch->descriptor_bytes = (uint32_t)sizeof(*dispatch);
	dispatch->flags = submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL ? SPARK_STAGE_RUNNER_DISPATCH_FLAG_PREFILL :
		(submission->flags & SPARK_MODEL_SERVING_SUBMISSION_FLAG_VERIFY) != 0u ? SPARK_STAGE_RUNNER_DISPATCH_FLAG_VERIFY : 0u;
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL && (submission->flags & SPARK_MODEL_SERVING_SUBMISSION_FLAG_DEFER) != 0u )
		dispatch->flags |= SPARK_STAGE_RUNNER_DISPATCH_FLAG_DEFER;
	dispatch->request_id = submission->request_id;
	dispatch->sequence_id = submission->sequence_id;
	dispatch->sequence_position = submission->sequence_position;
	dispatch->deadline_time_ns = submission->deadline_time_ns;
	dispatch->row_count = rows;
	dispatch->active_sequence_count = active;
	dispatch->token_ids = submission->token_ids;
	dispatch->positions = (const uint32_t *)StageServingAddress(&state->positions_device);
	dispatch->context_length = (const uint32_t *)StageServingAddress(&state->context_device);
	dispatch->sequence_of_row = (const uint32_t *)StageServingAddress(&state->state_device);
	dispatch->sequence_row_begin = (const uint32_t *)StageServingAddress(&state->runs_device);
	dispatch->sequence_row_indices = (const uint32_t *)StageServingAddress(&state->order_device);
	dispatch->recurrent_index = (const uint32_t *)StageServingAddress(&state->seqslot_device);
	dispatch->gather_sequence = active == 1u ? StageServingWords(&state->state_host)[0] : 0u;
	dispatch->gather_context = active == 1u && dispatch->gather_sequence < state->runner_config.max_active_sequence_count
		? StageServingWords(&state->context_host)[dispatch->gather_sequence] : 0u;
	dispatch->hidden_input_bf16 = submission->hidden_input_address;
	dispatch->hidden_input_bytes = submission->hidden_input_bytes;
	dispatch->hidden_output_bf16 = submission->hidden_output_address;
	dispatch->hidden_output_bytes = submission->hidden_output_bytes;
	dispatch->sideband_input = submission->boundary_sideband_input_address;
	dispatch->sideband_input_bytes = submission->boundary_sideband_input_bytes;
	dispatch->sideband_output = submission->boundary_sideband_output_address;
	dispatch->sideband_output_bytes = submission->boundary_sideband_output_bytes;
	dispatch->output_token_ids = (uint32_t *)StageServingAddress(&state->output_tokens);
	dispatch->output_scores = (float *)StageServingAddress(&state->output_scores);
	dispatch->chain_steps = steps;
	dispatch->host_positions = StageServingWords(&state->positions_host);
	dispatch->host_sequence_of_row = StageServingWords(&state->state_host);
	dispatch->host_state_target = StageServingWords(&state->target_host);
	if ( state->tree_frame != 0u )
	{
		dispatch->tree = 1u;
		dispatch->slots = (const uint32_t *)StageServingAddress(&state->slots_device);
		dispatch->fork_positions = (const uint32_t *)StageServingAddress(&state->fork_device);
		dispatch->slot_offsets = (const uint32_t *)StageServingAddress(&state->offset_device);
		dispatch->parent_rows = (const uint32_t *)StageServingAddress(&state->parent_device);
		dispatch->state_loads = (const uint32_t *)StageServingAddress(&state->loads_device);
		dispatch->state_saves = (const uint32_t *)StageServingAddress(&state->saves_device);
		dispatch->host_slots = StageServingWords(&state->slots_host);
		dispatch->host_parent_rows = StageServingWords(&state->parent_host);
	}
}

static SparkStatus StageServingReadTokens(SparkStageServingState *state, SparkDeviceQueue *queue, uint32_t rows, uint32_t steps)
{
	const SparkDeviceRange tokens = { &state->tokens_host, 0u }, output = { &state->output_tokens, 0u };
	const uint64_t bytes = (uint64_t)rows * steps * sizeof(uint32_t);
	if ( bytes > state->tokens_host.bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( SparkDeviceCopy(queue, tokens, output, bytes) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static void StageServingVerifyKept(SparkStageServingState *state, const SparkModelServingSubmission *submission,
	uint32_t active, uint32_t *kept)
{
	const uint32_t *runs = StageServingWords(&state->runs_host);
	const uint32_t *order = StageServingWords(&state->order_host);
	if ( state->tree_frame != 0u )
	{
		kept[0] = SparkSpeculationTreeResolvePath(submission->row_parent_rows, submission->token_ids, StageServingWords(&state->tokens_host), submission->row_count,
			state->path_rows);
		state->path_begin[0] = 0u;
		state->path_begin[1] = kept[0];
		return;
	}
	for ( uint32_t s = 0u; s < active; ++s )
		kept[s] = SparkModelServingVerifyKeptRows(runs, order, StageServingWords(&state->tokens_host), submission->token_ids, s);
}

static SparkStatus StageServingComplete(SparkStageServingState *state, const SparkModelServingSubmission *submission,
	uint32_t rows, uint32_t active, uint32_t steps, const uint32_t *kept, uint32_t distribution_count, const uint32_t *distribution_lanes,
	const SparkRowSampling *distribution_rules, const SparkSamplingLogprob *distribution_logprobs, uint32_t logprob_stride, uint64_t started_ns)
{
	SparkModelServingCompletion completion;
	const uint64_t capture_ns = SparkStageRunnerTakeCaptureNs(&state->runner), elapsed_ns = SparkMonotonicNs() - started_ns;
	StageServingCompletionHeader(submission, rows, &completion);
	if ( capture_ns != 0u )
		completion.service_time_ns = elapsed_ns > capture_ns ? elapsed_ns - capture_ns : 1u;
	if ( state->runner.owns_final_head != 0u && kept != 0 )
	{
		const uint32_t *runs = StageServingWords(&state->runs_host);
		const uint32_t *order = StageServingWords(&state->order_host);
		const uint32_t *tokens_host = StageServingWords(&state->tokens_host);
		const uint32_t stride = submission->tokens_per_sequence;
		completion.tokens_per_sequence = stride;
		completion.token_count = active * stride;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for ( uint32_t s = 0u; s < active; ++s )
			for ( uint32_t w = 0u; w < stride; ++w )
				completion.token_ids[s * stride + w] = w < kept[s] ? tokens_host[state->tree_frame != 0u ? state->path_rows[w] : order[runs[s] + w]] :
					SPARK_MODEL_SERVING_NO_TOKEN;
	}
	else if ( state->runner.owns_final_head != 0u )
	{
		const uint32_t *runs = StageServingWords(&state->runs_host);
		const uint32_t *order = StageServingWords(&state->order_host);
		const uint32_t *tokens_host = StageServingWords(&state->tokens_host);
		completion.tokens_per_sequence = steps;
		completion.token_count = active * steps;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for ( uint32_t s = 0u; s < active; ++s )
			for ( uint32_t step = 0u; step < steps && s * steps + step < SPARK_MODEL_SERVING_ADAPTER_MAX_OUTPUT_TOKEN_COUNT; ++step )
				completion.token_ids[s * steps + step] = tokens_host[step * rows + order[runs[s + 1u] - 1u]];
		completion.logprob_stride = logprob_stride;
		completion.logprob_entry_count = completion.token_count * logprob_stride;
		for ( uint32_t entry = 0u; logprob_stride != 0u && entry < distribution_count; ++entry )
			memcpy(&completion.logprobs[(uint64_t)distribution_lanes[entry] * logprob_stride],
				&distribution_logprobs[(uint64_t)entry * SPARK_SAMPLING_MAX_LOGPROBS],
				(uint64_t)distribution_rules[entry].logprobs * sizeof(SparkSamplingLogprob));
	}
	state->completion_function(state->completion_context, &completion);
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingSubmit(void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	SparkStageRunnerDispatch dispatch;
	SparkModelDriverFrame frame;
	SparkStageServingContinuity continuity;
	uint32_t *row_slots = state != 0 ? state->row_slots : 0;
	uint8_t bound[SPARK_STAGE_SERVING_MAX_LANES];
	uint64_t sequence_ids[SPARK_STAGE_SERVING_MAX_LANES], next_positions[SPARK_STAGE_SERVING_MAX_LANES];
	uint32_t *distribution_rows = state != 0 ? state->distribution_rows : 0, *distribution_positions = state != 0 ? state->distribution_positions : 0;
	uint32_t *distribution_lanes = state != 0 ? state->distribution_lanes : 0;
	SparkRowSampling *distribution_rules = state != 0 ? state->distribution_rules : 0;
	SparkSamplingLogprob distribution_logprobs[SPARK_STAGE_SERVING_MAX_LANES * SPARK_SAMPLING_MAX_LOGPROBS];
	uint32_t rows, active = 0u, distribution_count = 0u, logprob_stride = 0u, steps = 1u, kept[SPARK_STAGE_SERVING_MAX_LANES] = { 0u };
	const uint32_t verify = submission != 0 && (submission->flags & SPARK_MODEL_SERVING_SUBMISSION_FLAG_VERIFY) != 0u ? 1u : 0u;
	uint64_t marks[6], keep_ns = 0u;
	SparkStatus status, finished;
	SparkDeviceQueue *queue;
	marks[0] = SparkMonotonicNs();
	status = StageServingValidateSubmission(state, submission);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( SparkModelServingVerifyPublishLane(submission) != UINT32_MAX )
	{
		fprintf(stderr, "%s adapter: submit refused: lane %u of a verify frame publishes cache blocks, and its recurrent state would be captured before keep_rows\n",
			state->model->module_tag, SparkModelServingVerifyPublishLane(submission));
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( SparkModelServingWorkKindUsesRows(submission->work_kind) == 0u )
		return StageServingSubmitNonRow(state, submission);
	queue = SparkStageRunnerQueue(&state->runner);
	rows = submission->row_count;
	status = StageServingGroupRows(state, submission, rows, &active);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = StageServingAdmitFrame(state, submission, &frame);
	if ( status == SPARK_STATUS_OK )
		status = StageServingChainSteps(state, submission, &frame, rows, active, &steps);
	if ( status != SPARK_STATUS_OK )
		return status;
	StageServingContinuityRows(state, submission, rows, active, row_slots, state->row_sequence_ids, state->row_positions);
	for ( uint32_t lane = 0u; state->runner.owns_final_head != 0u && submission->lanes != 0 && lane < active; ++lane )
	{
		const uint32_t *runs = StageServingWords(&state->runs_host);
		const uint32_t *order = StageServingWords(&state->order_host);
		if ( SparkSamplingRuleNeedsDistribution(&submission->lanes[lane].sampling) == 0u )
			continue;
		for ( uint32_t cursor = verify != 0u ? runs[lane] : runs[lane + 1u] - 1u; cursor < runs[lane + 1u]; ++cursor )
		{
			distribution_rows[distribution_count] = order[cursor];
			distribution_positions[distribution_count] = StageServingWords(&state->positions_host)[order[cursor]];
			distribution_rules[distribution_count] = submission->lanes[lane].sampling;
			distribution_lanes[distribution_count] = lane;
			distribution_count++;
		}
		if ( submission->lanes[lane].sampling.logprobs > logprob_stride )
			logprob_stride = submission->lanes[lane].sampling.logprobs;
	}
	if ( logprob_stride != 0u && submission->tokens_per_sequence > 1u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	continuity.state = state;
	continuity.rows = rows;
	continuity.active = active;
	continuity.row_slots = row_slots;
	continuity.row_sequence_ids = state->row_sequence_ids;
	continuity.row_positions = state->row_positions;
	continuity.bound = bound;
	continuity.sequence_ids = sequence_ids;
	continuity.next_positions = next_positions;
	if ( state->streamed_failed != 0u )
	{
		fprintf(stderr, "%s adapter: submit refused: a streamed prefill failed and left published blocks without KV; restart the lane with its KV wiped\n",
			state->model->module_tag);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	status = SparkStageModuleIndexSetClaimAndPrepare(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES,
		row_slots, active, StageServingPrepareContinuity, &continuity);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = SparkStageKvBindingClaim(&state->kv, &frame, active, row_slots, state->row_sequence_ids, state->row_positions,
		next_positions);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageModuleIndexSetRelease(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES, row_slots, active);
		return status;
	}
	status = SparkStageKvBindingFenceExecution(&state->kv, queue);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingRecurrentRestore(&state->kv, row_slots, active, queue);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingUploadPageTables(&state->kv, row_slots, active, queue);
	if ( status == SPARK_STATUS_OK )
		status = StageServingCopyInputs(state, queue, rows, active);
	if ( status == SPARK_STATUS_OK )
	{
		StageServingDispatch(state, submission, rows, active, steps, &dispatch);
		dispatch.distribution_count = distribution_count;
		dispatch.distribution_rows = distribution_rows;
		dispatch.distribution_positions = distribution_positions;
		dispatch.distribution_rules = distribution_rules;
		dispatch.distribution_logprobs = logprob_stride != 0u ? distribution_logprobs : 0;
		marks[1] = SparkMonotonicNs();
		status = SparkStageRunnerSubmit(&state->runner, &dispatch);
	}
	marks[2] = SparkMonotonicNs();
	if ( status == SPARK_STATUS_OK && (dispatch.flags & SPARK_STAGE_RUNNER_DISPATCH_FLAG_DEFER) != 0u && state->saves_held == 0u )
	{
		status = SparkStageKvBindingHoldSaves(&state->kv, 1u);
		state->saves_held = status == SPARK_STATUS_OK ? 1u : 0u;
	}
	if ( status == SPARK_STATUS_OK && (dispatch.flags & SPARK_STAGE_RUNNER_DISPATCH_FLAG_DEFER) == 0u )
		status = SparkStageKvBindingRecurrentCapture(&state->kv, row_slots, active, queue);
	if ( status == SPARK_STATUS_OK && state->runner.owns_final_head != 0u )
		status = StageServingReadTokens(state, queue, rows, steps);
	marks[3] = SparkMonotonicNs();
	if ( status == SPARK_STATUS_OK && SparkDeviceQueueWait(queue, SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageRunnerCheck(&state->runner);
	marks[4] = SparkMonotonicNs();
	if ( status == SPARK_STATUS_OK && verify != 0u )
	{
		const uint64_t keep_started = SparkMonotonicNs();
		StageServingVerifyKept(state, submission, active, kept);
		if ( state->tree_frame != 0u && kept[0] == 0u )
			status = SPARK_STATUS_VALIDATION_FAILED;
		else
			status = state->tree_frame != 0u ? SparkStageRunnerKeepPath(&state->runner, state->path_rows, state->path_begin, kept, active) :
				SparkStageRunnerKeepRows(&state->runner, kept, active);
		if ( status == SPARK_STATUS_OK && SparkDeviceQueueWait(queue, SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
			status = SPARK_STATUS_IO_ERROR;
		if ( status == SPARK_STATUS_OK )
			status = SparkStageRunnerCheck(&state->runner);
		keep_ns = SparkMonotonicNs() - keep_started;
	}
	for ( uint32_t lane = 0u; status == SPARK_STATUS_OK && lane < active; ++lane )
		next_positions[lane] = verify != 0u ? state->row_positions[lane] + kept[lane] : next_positions[lane] + steps - 1u;
	if ( status == SPARK_STATUS_OK && state->tap_stream != 0 && state->tap_socket < 0 && state->tap_count != 0u &&
		state->runner_config.tp_rank == state->tap_rank && SparkMonotonicNs() >= state->tap_retry_ns )
		(void)StageServingTapStreamConnect(state, STAGE_SERVING_TAP_RECONNECT_MS);
	if ( status == SPARK_STATUS_OK && StageServingTapOpen(state) != 0u && state->tap_full == 0u && (dispatch.flags & SPARK_STAGE_RUNNER_DISPATCH_FLAG_DEFER) == 0u )
		StageServingTapEmit(state, submission, rows, active, steps, verify != 0u ? kept : 0);
	finished = StageServingFinish(state, active, row_slots, bound, sequence_ids, next_positions, steps - 1u, verify, status);
	if ( state->saves_held != 0u && (dispatch.flags & SPARK_STAGE_RUNNER_DISPATCH_FLAG_DEFER) == 0u )
	{
		if ( status == SPARK_STATUS_OK && finished == SPARK_STATUS_OK && SparkStageKvBindingHoldSaves(&state->kv, 0u) == SPARK_STATUS_OK )
			state->saves_held = 0u;
		else
		{
			state->streamed_failed = 1u;
			fprintf(stderr, "%s adapter: the wave after deferred prefill tokens failed status=%d; their published blocks hold no KV, so this lane refuses work until it restarts with its KV wiped\n",
				state->model->module_tag, (int)(status != SPARK_STATUS_OK ? status : finished));
		}
	}
	marks[5] = SparkMonotonicNs();
	if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
		StageServingWaveHost(state, marks);
	else if ( status == SPARK_STATUS_OK && verify != 0u )
		StageServingVerifyHost(state, marks, keep_ns, rows);
	else if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		StageServingChainHost(state, marks, steps);
	SparkStageModuleIndexSetRelease(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES, row_slots, active);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( finished != SPARK_STATUS_OK )
		return finished;
	if ( state->completion_function == 0 )
		return SPARK_STATUS_OK;
	return StageServingComplete(state, submission, rows, active, steps, verify != 0u ? kept : 0, distribution_count, distribution_lanes,
		distribution_rules, distribution_logprobs, logprob_stride, marks[0]);
}

static SparkStatus StageServingPrefetch(void *adapter_state,
	const SparkModelServingSubmission *submissions, uint32_t submission_count)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	SparkServingCacheAdmission cache;
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	cache = StageServingCacheContext(state, state->prefetch_lanes);
	cache.validate = StageServingValidateSubmission;
	return SparkServingCacheAdmissionRun(&cache, submissions, submission_count,
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE);
}

static SparkStatus StageServingCacheHint(void *adapter_state,
	const SparkModelServingCacheIdentity *identity, uint32_t token_count)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	SparkServingCacheAdmission cache;
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	cache = StageServingCacheContext(state, state->prefetch_lanes);
	return SparkServingCacheHintRun(&cache, identity, token_count);
}

static SparkStatus StageServingResolvePrefetch(void *adapter_state,
	const SparkModelServingSubmission *submission, uint32_t resolution)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	SparkServingCacheAdmission cache;
	if ( state == 0 || (resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT &&
		resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	cache = StageServingCacheContext(state, state->prefetch_lanes);
	cache.validate = StageServingValidateSubmission;
	return SparkServingCacheAdmissionRun(&cache, submission, 1u,
		resolution == SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT ?
		SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT : SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT);
}

static SparkStatus StageServingProgress(void *adapter_state, uint32_t maximum_step_count)
{
	(void)adapter_state;
	(void)maximum_step_count;
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingQuiesce(void *adapter_state, uint64_t deadline_time_ns)
{
	(void)adapter_state;
	(void)deadline_time_ns;
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingSnapshot(void *adapter_state,
	SparkModelServingAdapterSnapshot *snapshot)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	SparkStageRunnerStats stats;
	uint32_t block;
	if ( state == 0 || snapshot == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	block = StageServingDescriptor.cache_block_token_count;
	memset(snapshot, 0, sizeof(*snapshot));
	snapshot->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	snapshot->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_SNAPSHOT_BYTES;
	snapshot->available_submission_count = state->max_rows;
	SparkStageRunnerGetStats(&state->runner, &stats);
	snapshot->submitted_count = stats.submitted_count;
	snapshot->completed_count = stats.completed_count;
	snapshot->resident_sequence_count = SparkStageKvBindingResidentCount(&state->kv);
	snapshot->kv_token_capacity = (uint64_t)state->kv.logical_page_count * block;
	snapshot->max_sequence_positions = state->runner_config.kv_pages_per_sequence * block;
	SparkStageKvBindingKvStoreCounters(&state->kv, &snapshot->kv_store);
	if ( stats.lane_fatal != 0u || state->streamed_failed != 0u )
		snapshot->degraded_flags |= SPARK_MODEL_DRIVER_DEGRADED_LANE_FATAL;
	return SPARK_STATUS_OK;
}

static SparkStatus StageServingReset(void *adapter_state, uint64_t control_generation)
{
	SparkStageServingState *state = (SparkStageServingState *)adapter_state;
	uint32_t slots[SPARK_STAGE_SERVING_MAX_LANES], slot, count;
	SparkStatus status;
	if ( state == 0 || state->kv.resident_sequence_capacity > SPARK_STAGE_SERVING_MAX_LANES )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->streamed_failed != 0u )
	{
		fprintf(stderr, "%s reset refused: a streamed prefill failed and left published blocks without KV; a reset cannot clear them, so the lane exits and must restart with its KV wiped\n",
			state->model->module_tag);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	if ( SparkStageKvBindingResetIsNew(&state->kv, control_generation) == 0u )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	count = state->kv.resident_sequence_capacity;
	for ( slot = 0u; slot < count; ++slot )
		slots[slot] = slot;
	status = SparkStageModuleIndexSetClaim(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES, slots, count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( SparkDeviceQueueWait(SparkStageRunnerQueue(&state->runner), SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK ||
		(state->kv.copy_queue != 0 && SparkDeviceQueueWait(state->kv.copy_queue, SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK) )
	{
		fprintf(stderr, "%s reset: execution stream failed; lanes stay claimed\n", state->model->module_tag);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageRunnerResetSlots(&state->runner, slots, count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageKvBindingReset(&state->kv, control_generation);
	if ( status == SPARK_STATUS_OK )
		atomic_store_explicit(&state->reset_generation, control_generation, memory_order_release);
	SparkStageModuleIndexSetRelease(state->lane_states, SPARK_STAGE_SERVING_MAX_LANES, slots, count);
	SPARK_RETURN(status);
}

static SparkModelServingAdapterInterface StageServingInterface =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = 0,
	.initialize = StageServingInitialize,
	.destroy = StageServingDestroy,
	.validate_submission = StageServingValidateSubmission,
	.submit = StageServingSubmit,
	.prefetch = StageServingPrefetch,
	.cache_hint = StageServingCacheHint,
	.resolve_prefetch = StageServingResolvePrefetch,
	.progress = StageServingProgress,
	.quiesce = StageServingQuiesce,
	.snapshot = StageServingSnapshot,
	.reset = StageServingReset,
};

static uint32_t StageServingModelValid(const SparkStageServingModel *model)
{
	return model != 0 && model->adapter_id != 0 && model->model_id != 0 && model->model_revision != 0 && model->program != 0 &&
		model->contract_sha256 != 0 && model->module_tag != 0 && model->runner_model != 0 && model->kv_layout != 0 &&
		model->collective_sequences != 0 && model->tp_degree != 0u && model->tp_degree <= SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT &&
		model->layer_count != 0u && model->hidden != 0u && model->cache_block_token_count != 0u &&
		model->max_rows != 0u && model->max_rows <= SPARK_STAGE_SERVING_MAX_ROWS && model->verify_rows <= model->max_rows &&
		model->speculative_tokens < SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE;
}

static void StageServingDescribe(const SparkStageServingModel *model, SparkModelServingAdapterDescriptor *descriptor)
{
	uint32_t stage;
	memset(descriptor, 0, sizeof(*descriptor));
	descriptor->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	descriptor->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	descriptor->adapter_id = model->adapter_id;
	descriptor->model_id = model->model_id;
	descriptor->model_revision = model->model_revision;
	descriptor->driver_program_name = model->program;
	descriptor->artifact_sha256 = model->contract_sha256;
	descriptor->capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CONTINUE_LEASE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_RESIDENT_DECODE_CHAIN |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFIX_REUSE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_CACHE_PUBLISH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SAMPLING_TRUNCATION |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_LOGPROBS |
		(model->speculative_tokens != 0u ? SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATION : 0u) |
		(model->verify_rows != 0u ? SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATIVE_VERIFY : 0u) |
		(model->tree_slot_grain != 0u ? SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_TREE_VERIFY : 0u);
	descriptor->stage_count = model->tp_degree;
	descriptor->layer_count = model->layer_count;
	descriptor->boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16;
	descriptor->boundary_element_count = model->hidden;
	descriptor->boundary_element_bytes = 2u;
	descriptor->linear_weight_codec = model->linear_weight_codec;
	descriptor->expert_weight_codec = model->expert_weight_codec;
	descriptor->kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->max_inflight_submission_count = 16u;
	descriptor->max_active_sequence_count = SPARK_STAGE_SERVING_MAX_LANES;
	descriptor->max_input_row_count = model->max_rows;
	descriptor->max_resident_sequence_count = SPARK_STAGE_SERVING_MAX_LANES;
	descriptor->max_output_token_count = SPARK_STAGE_SERVING_MAX_LANES * SPARK_STAGE_SERVING_CHAIN_TOKENS;
	descriptor->max_speculative_token_count = model->speculative_tokens;
	for ( stage = 0u; stage < model->tp_degree; ++stage )
		descriptor->stage_layer_counts[stage] = model->layer_count;
	descriptor->minimum_efficient_submission_row_count = 1u;
	descriptor->cache_block_token_count = model->cache_block_token_count;
	descriptor->parallel_group_size = 0u;
	descriptor->cache_checkpoint_token_count = model->cache_checkpoint_token_count;
	descriptor->max_verify_row_count = model->verify_rows;
	descriptor->tree_slot_grain = model->tree_slot_grain;
}

const SparkModelServingAdapterDescriptor *SparkStageServingDescriptor(void)
{
	return StageServingBoundModel != 0 ? &StageServingDescriptor : 0;
}

const SparkModelServingAdapterInterface *SparkStageServingAdapterInterface(const SparkStageServingModel *model)
{
	if ( StageServingModelValid(model) == 0u || (StageServingBoundModel != 0 && StageServingBoundModel != model) )
		return 0;
	StageServingBoundModel = model;
	StageServingDescribe(model, &StageServingDescriptor);
	StageServingInterface.descriptor = &StageServingDescriptor;
	return &StageServingInterface;
}
