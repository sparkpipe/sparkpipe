#include "sparkpipe/spark_draft_server.h"
#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_io.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the DFT3 wire format is little-endian and committed tokens are read in place");

#define DRAFT_SERVER_MAGIC "DFT3"
#define DRAFT_SERVER_MAGIC_BYTES 4u
#define DRAFT_SERVER_NS_PER_MS UINT64_C(1000000)
#define DRAFT_SERVER_CLOSED (-1)

#define DRAFT_SERVER_PHASE_IDLE 0u
#define DRAFT_SERVER_PHASE_HEADER 1u
#define DRAFT_SERVER_PHASE_BODY 2u
#define DRAFT_SERVER_PHASE_ANSWER 3u
#define DRAFT_SERVER_PHASE_SEND 4u

struct SparkDraftServer
{
	char target_model[SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES];
	uint32_t port;
	uint32_t tap_row_bytes;
	uint32_t max_committed_tokens;
	uint32_t max_tap_rows;
	uint32_t max_nodes;
	uint32_t io_timeout_ms;
	int listen_fd;
	int socket_fd;
	uint32_t phase;
	uint32_t close_after_send;
	uint64_t deadline_ns;
	uint8_t header[SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES];
	uint64_t header_got;
	uint8_t *body;
	uint64_t body_capacity;
	uint64_t body_expected;
	uint64_t body_got;
	uint8_t *reply;
	uint64_t reply_capacity;
	uint64_t reply_bytes;
	uint64_t reply_sent;
	uint32_t reply_counts;
	SparkDraftServerRequest request;
	SparkDraftServerStats stats;
};

static void DraftServerPutU32(uint8_t *destination, uint32_t value)
{
	memcpy(destination, &value, sizeof(value));
}

static void DraftServerPutU64(uint8_t *destination, uint64_t value)
{
	memcpy(destination, &value, sizeof(value));
}

static uint32_t DraftServerU32(const uint8_t *source)
{
	uint32_t value;
	memcpy(&value, source, sizeof(value));
	return value;
}

static uint64_t DraftServerU64(const uint8_t *source)
{
	uint64_t value;
	memcpy(&value, source, sizeof(value));
	return value;
}

static float DraftServerF32(const uint8_t *source)
{
	float value;
	memcpy(&value, source, sizeof(value));
	return value;
}

static SparkStatus DraftServerNonBlocking(int fd)
{
	const int flags = fcntl(fd, F_GETFL, 0);
	if ( flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static void DraftServerDrop(SparkDraftServer *server)
{
	if ( server->socket_fd != DRAFT_SERVER_CLOSED )
	{
		if ( server->phase != DRAFT_SERVER_PHASE_HEADER || server->header_got != 0u )
			server->stats.dropped++;
		(void)close(server->socket_fd);
	}
	server->socket_fd = DRAFT_SERVER_CLOSED;
	server->phase = DRAFT_SERVER_PHASE_IDLE;
	server->header_got = 0u;
	server->body_got = 0u;
	server->deadline_ns = 0u;
	server->close_after_send = 0u;
}

static void DraftServerAwaitHeader(SparkDraftServer *server)
{
	server->phase = DRAFT_SERVER_PHASE_HEADER;
	server->header_got = 0u;
	server->body_got = 0u;
	server->deadline_ns = 0u;
}

static uint64_t DraftServerDeadline(const SparkDraftServer *server)
{
	return SparkMonotonicNs() + (uint64_t)server->io_timeout_ms * DRAFT_SERVER_NS_PER_MS;
}

static void DraftServerAccept(SparkDraftServer *server)
{
	int enabled = 1;
	const int fd = accept(server->listen_fd, 0, 0);
	if ( fd < 0 )
		return;
	if ( DraftServerNonBlocking(fd) != SPARK_STATUS_OK ||
		setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enabled, (socklen_t)sizeof(enabled)) != 0 )
	{
		(void)close(fd);
		return;
	}
#ifdef SO_NOSIGPIPE
	(void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, (socklen_t)sizeof(enabled));
#endif
	DraftServerDrop(server);
	server->socket_fd = fd;
	server->stats.connections++;
	DraftServerAwaitHeader(server);
}

static void DraftServerBuildReply(SparkDraftServer *server, uint32_t status, const SparkDraftBridgeNode *nodes, uint32_t node_count,
	const SparkDraftBridgeProposalInfo *info, uint64_t sequence_id, uint64_t generation, uint32_t elapsed_us)
{
	uint8_t *cursor = server->reply;
	memcpy(cursor, DRAFT_SERVER_MAGIC, DRAFT_SERVER_MAGIC_BYTES);
	cursor += DRAFT_SERVER_MAGIC_BYTES;
	DraftServerPutU32(cursor, status);
	DraftServerPutU32(cursor + 4u, node_count);
	DraftServerPutU64(cursor + 8u, sequence_id);
	DraftServerPutU64(cursor + 16u, generation);
	cursor += 24u;
	for ( uint32_t index = 0u; index < node_count; ++index )
	{
		DraftServerPutU32(cursor, nodes[index].token_id);
		DraftServerPutU32(cursor + 4u, nodes[index].parent_index);
		DraftServerPutU32(cursor + 8u, nodes[index].depth);
		DraftServerPutU32(cursor + 12u, nodes[index].source_bit);
		memcpy(cursor + 16u, &nodes[index].score, sizeof(float));
		cursor += SPARK_DRAFT_BRIDGE_NODE_RECORD_BYTES;
	}
	for ( uint32_t stage = 0u; stage < SPARK_DRAFT_BRIDGE_DRAFTER_STAGE_COUNT; ++stage )
		DraftServerPutU32(cursor + 4u * stage, info != 0 ? info->drafter_us[stage] : 0u);
	cursor += 4u * SPARK_DRAFT_BRIDGE_DRAFTER_STAGE_COUNT;
	DraftServerPutU32(cursor, info != 0 ? info->graft_us : 0u);
	DraftServerPutU32(cursor + 4u, info != 0 ? info->expansions : 0u);
	DraftServerPutU32(cursor + 8u, elapsed_us);
	DraftServerPutU32(cursor + 12u, 0u);
	cursor += 16u;
	server->reply_bytes = (uint64_t)(cursor - server->reply);
	server->reply_sent = 0u;
	server->phase = DRAFT_SERVER_PHASE_SEND;
	server->deadline_ns = DraftServerDeadline(server);
}

static void DraftServerRefuse(SparkDraftServer *server, uint32_t close_after_send, const char *reason)
{
	fprintf(stderr, "sparkpipe_draft_server: port %u refused a request: %s\n", server->port, reason);
	server->stats.bad_requests++;
	server->reply_counts = 0u;
	server->close_after_send = close_after_send;
	DraftServerBuildReply(server, SPARK_DRAFT_BRIDGE_SERVER_STATUS_BAD_REQUEST, 0, 0u, 0, DraftServerU64(server->header + 44u),
		DraftServerU64(server->header + 52u), 0u);
}

static void DraftServerHeaderDone(SparkDraftServer *server)
{
	const uint32_t committed = DraftServerU32(server->header + 72u), tap_rows = DraftServerU32(server->header + 76u);
	if ( memcmp(server->header, DRAFT_SERVER_MAGIC, DRAFT_SERVER_MAGIC_BYTES) != 0 )
	{
		fprintf(stderr, "sparkpipe_draft_server: port %u dropped a connection whose request does not start with DFT3\n", server->port);
		DraftServerDrop(server);
		return;
	}
	if ( committed == 0u || committed > server->max_committed_tokens || tap_rows > server->max_tap_rows || (tap_rows != 0u && server->tap_row_bytes == 0u) )
	{
		DraftServerRefuse(server, 1u, "its committed tokens or tap rows exceed the server's limits");
		return;
	}
	server->body_expected = (uint64_t)committed * sizeof(uint32_t) + (uint64_t)tap_rows * server->tap_row_bytes;
	server->body_got = 0u;
	server->phase = DRAFT_SERVER_PHASE_BODY;
}

static uint32_t DraftServerBodyDone(SparkDraftServer *server, SparkDraftServerRequest *request)
{
	const uint8_t *header = server->header;
	SparkDraftServerRequest parsed;
	char target[SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES];
	memset(&parsed, 0, sizeof(parsed));
	memcpy(target, header + 12u, sizeof(target));
	parsed.speculator_mask = DraftServerU32(header + 8u);
	parsed.sequence_id = DraftServerU64(header + 44u);
	parsed.generation = DraftServerU64(header + 52u);
	parsed.position = DraftServerU64(header + 60u);
	parsed.anchor_token_id = DraftServerU32(header + 68u);
	parsed.committed_token_count = DraftServerU32(header + 72u);
	parsed.tap_row_count = DraftServerU32(header + 76u);
	parsed.depth = DraftServerU32(header + 80u);
	parsed.time_budget_ms = DraftServerU32(header + 84u);
	parsed.max_depth = DraftServerU32(header + 88u);
	parsed.node_capacity = DraftServerU32(header + 92u);
	parsed.sampling.seed = DraftServerU64(header + 96u);
	parsed.sampling.inverse_temperature = DraftServerF32(header + 104u);
	parsed.sampling.nucleus = DraftServerF32(header + 108u);
	parsed.sampling.top_k = DraftServerU32(header + 112u);
	parsed.committed_token_ids = (const uint32_t *)(const void *)server->body;
	parsed.tap_rows = parsed.tap_row_count != 0u ? server->body + (uint64_t)parsed.committed_token_count * sizeof(uint32_t) : 0;
	parsed.received_ns = SparkMonotonicNs();
	if ( DraftServerU32(header + 4u) != SPARK_DRAFT_BRIDGE_PROTOCOL_VERSION )
		DraftServerRefuse(server, 0u, "its protocol version is not 2");
	else if ( memcmp(target, server->target_model, sizeof(target)) != 0 )
		DraftServerRefuse(server, 0u, "it names another target model");
	else if ( parsed.speculator_mask == 0u || (uint64_t)parsed.committed_token_count != parsed.position + 1u ||
		parsed.committed_token_ids[parsed.position] != parsed.anchor_token_id )
		DraftServerRefuse(server, 0u, "its anchor is not its last committed token, or it asks no source");
	else if ( parsed.depth == 0u || parsed.depth > SPARK_DRAFT_BRIDGE_MAX_DEPTH || parsed.max_depth == 0u ||
		parsed.max_depth > SPARK_DRAFT_BRIDGE_MAX_TREE_DEPTH || parsed.node_capacity == 0u || parsed.node_capacity > server->max_nodes ||
		parsed.time_budget_ms > SPARK_DRAFT_BRIDGE_MAX_TIME_BUDGET_MS )
		DraftServerRefuse(server, 0u, "its depth, tree depth, node capacity or time budget is out of range");
	else if ( SparkSamplingRuleValid(&parsed.sampling) == 0u )
		DraftServerRefuse(server, 0u, "its sampling rule is not valid");
	else
	{
		server->request = parsed;
		server->stats.requests++;
		server->phase = DRAFT_SERVER_PHASE_ANSWER;
		server->deadline_ns = 0u;
		*request = parsed;
		return 1u;
	}
	return 0u;
}

static uint32_t DraftServerReceive(SparkDraftServer *server, uint8_t *destination, uint64_t want, uint64_t *got)
{
	while ( *got < want )
	{
		const ssize_t part = recv(server->socket_fd, destination + *got, (size_t)(want - *got), 0);
		if ( part > 0 )
		{
			if ( server->deadline_ns == 0u )
				server->deadline_ns = DraftServerDeadline(server);
			*got += (uint64_t)part;
			continue;
		}
		if ( part < 0 && errno == EINTR )
			continue;
		if ( part < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) )
			return 0u;
		DraftServerDrop(server);
		return 0u;
	}
	return 1u;
}

static void DraftServerSend(SparkDraftServer *server)
{
	while ( server->reply_sent < server->reply_bytes )
	{
		const ssize_t put = SparkIoSend(server->socket_fd, server->reply + server->reply_sent, (size_t)(server->reply_bytes - server->reply_sent));
		if ( put > 0 )
		{
			server->reply_sent += (uint64_t)put;
			continue;
		}
		if ( put < 0 && errno == EINTR )
			continue;
		if ( put < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) )
			return;
		DraftServerDrop(server);
		return;
	}
	server->stats.answered += server->reply_counts;
	if ( server->close_after_send != 0u )
	{
		server->close_after_send = 0u;
		server->phase = DRAFT_SERVER_PHASE_HEADER;
		server->header_got = 0u;
		DraftServerDrop(server);
		return;
	}
	DraftServerAwaitHeader(server);
}

SparkStatus SparkDraftServerCreate(const SparkDraftServerConfig *config, SparkDraftServer **server_out)
{
	SparkDraftServer *server;
	struct addrinfo hints, *addresses = 0;
	struct sockaddr_storage bound;
	socklen_t bound_bytes = (socklen_t)sizeof(bound);
	char port_text[8];
	int enabled = 1;
	if ( server_out != 0 )
		*server_out = 0;
	if ( config == 0 || server_out == 0 || config->abi_version != SPARK_DRAFT_SERVER_ABI_VERSION ||
		config->descriptor_bytes != SPARK_DRAFT_SERVER_CONFIG_BYTES || config->host == 0 || config->port > SPARK_DRAFT_BRIDGE_MAX_PORT ||
		config->target_model[0] == '\0' || config->max_committed_tokens == 0u || config->max_nodes == 0u ||
		config->max_nodes > SPARK_DRAFT_BRIDGE_MAX_TREE_NODES || config->io_timeout_ms == 0u ||
		(config->max_tap_rows != 0u && config->tap_row_bytes == 0u) ||
		memchr(config->target_model, '\0', sizeof(config->target_model)) == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	server = (SparkDraftServer *)calloc(1u, sizeof(*server));
	if ( server == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memcpy(server->target_model, config->target_model, sizeof(server->target_model));
	server->tap_row_bytes = config->tap_row_bytes;
	server->max_committed_tokens = config->max_committed_tokens;
	server->max_tap_rows = config->max_tap_rows;
	server->max_nodes = config->max_nodes;
	server->io_timeout_ms = config->io_timeout_ms;
	server->listen_fd = DRAFT_SERVER_CLOSED;
	server->socket_fd = DRAFT_SERVER_CLOSED;
	server->body_capacity = (uint64_t)config->max_committed_tokens * sizeof(uint32_t) + (uint64_t)config->max_tap_rows * config->tap_row_bytes;
	server->reply_capacity = SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES + (uint64_t)config->max_nodes * SPARK_DRAFT_BRIDGE_NODE_RECORD_BYTES +
		SPARK_DRAFT_BRIDGE_RESPONSE_FOOTER_BYTES;
	server->body = (uint8_t *)malloc((size_t)server->body_capacity);
	server->reply = (uint8_t *)malloc((size_t)server->reply_capacity);
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
	(void)snprintf(port_text, sizeof(port_text), "%u", config->port);
	if ( server->body == 0 || server->reply == 0 || getaddrinfo(config->host, port_text, &hints, &addresses) != 0 || addresses == 0 )
	{
		if ( addresses != 0 )
			freeaddrinfo(addresses);
		SparkDraftServerDestroy(server);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	server->listen_fd = socket(addresses->ai_family, addresses->ai_socktype, addresses->ai_protocol);
	if ( server->listen_fd < 0 || setsockopt(server->listen_fd, SOL_SOCKET, SO_REUSEADDR, &enabled, (socklen_t)sizeof(enabled)) != 0 ||
		bind(server->listen_fd, addresses->ai_addr, addresses->ai_addrlen) != 0 || listen(server->listen_fd, 4) != 0 ||
		DraftServerNonBlocking(server->listen_fd) != SPARK_STATUS_OK ||
		getsockname(server->listen_fd, (struct sockaddr *)&bound, &bound_bytes) != 0 )
	{
		fprintf(stderr, "sparkpipe_draft_server: cannot listen on %s:%u: %s\n", config->host, config->port, strerror(errno));
		freeaddrinfo(addresses);
		SparkDraftServerDestroy(server);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	freeaddrinfo(addresses);
	server->port = bound.ss_family == AF_INET6 ? ntohs(((const struct sockaddr_in6 *)(const void *)&bound)->sin6_port) :
		ntohs(((const struct sockaddr_in *)(const void *)&bound)->sin_port);
	*server_out = server;
	return SPARK_STATUS_OK;
}

void SparkDraftServerDestroy(SparkDraftServer *server)
{
	if ( server == 0 )
		return;
	if ( server->socket_fd != DRAFT_SERVER_CLOSED )
		(void)close(server->socket_fd);
	if ( server->listen_fd != DRAFT_SERVER_CLOSED )
		(void)close(server->listen_fd);
	free(server->body);
	free(server->reply);
	free(server);
}

uint32_t SparkDraftServerPort(const SparkDraftServer *server)
{
	return server != 0 ? server->port : 0u;
}

uint32_t SparkDraftServerPollFds(const SparkDraftServer *server, int *fds, uint32_t *events, uint64_t *deadline_ns)
{
	uint32_t count = 0u;
	if ( server == 0 || fds == 0 || events == 0 )
		return 0u;
	fds[count] = server->listen_fd;
	events[count++] = SPARK_DRAFT_BRIDGE_POLL_READ;
	if ( server->socket_fd != DRAFT_SERVER_CLOSED && server->phase != DRAFT_SERVER_PHASE_ANSWER )
	{
		fds[count] = server->socket_fd;
		events[count++] = server->phase == DRAFT_SERVER_PHASE_SEND ? SPARK_DRAFT_BRIDGE_POLL_WRITE : SPARK_DRAFT_BRIDGE_POLL_READ;
	}
	if ( deadline_ns != 0 )
		*deadline_ns = server->deadline_ns;
	return count;
}

SparkStatus SparkDraftServerAdvance(SparkDraftServer *server, SparkDraftServerRequest *request_out)
{
	if ( server == 0 || request_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	DraftServerAccept(server);
	for ( ;; )
	{
		const uint32_t phase = server->phase;
		if ( server->socket_fd == DRAFT_SERVER_CLOSED || phase == DRAFT_SERVER_PHASE_ANSWER )
			return SPARK_STATUS_PENDING;
		if ( server->deadline_ns != 0u && SparkMonotonicNs() > server->deadline_ns )
		{
			fprintf(stderr, "sparkpipe_draft_server: port %u dropped a connection that stalled for %u ms inside an exchange\n", server->port,
				server->io_timeout_ms);
			DraftServerDrop(server);
			return SPARK_STATUS_PENDING;
		}
		if ( phase == DRAFT_SERVER_PHASE_HEADER )
		{
			if ( DraftServerReceive(server, server->header, sizeof(server->header), &server->header_got) == 0u )
				return SPARK_STATUS_PENDING;
			DraftServerHeaderDone(server);
		}
		else if ( phase == DRAFT_SERVER_PHASE_BODY )
		{
			if ( DraftServerReceive(server, server->body, server->body_expected, &server->body_got) == 0u )
				return SPARK_STATUS_PENDING;
			if ( DraftServerBodyDone(server, request_out) != 0u )
				return SPARK_STATUS_OK;
		}
		else if ( phase == DRAFT_SERVER_PHASE_SEND )
		{
			DraftServerSend(server);
			if ( server->phase == DRAFT_SERVER_PHASE_SEND )
				return SPARK_STATUS_PENDING;
		}
	}
}

SparkStatus SparkDraftServerAnswer(SparkDraftServer *server, uint32_t server_status, const SparkDraftBridgeNode *nodes, uint32_t node_count,
	const SparkDraftBridgeProposalInfo *info)
{
	uint64_t elapsed;
	if ( server == 0 || (node_count != 0u && nodes == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( server->phase != DRAFT_SERVER_PHASE_ANSWER )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( node_count > server->request.node_capacity || (server_status != SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK && node_count != 0u) )
	{
		fprintf(stderr, "sparkpipe_draft_server: port %u: a drafter answered %u nodes with status %u for a request of capacity %u; sent INTERNAL\n",
			server->port, node_count, server_status, server->request.node_capacity);
		server->reply_counts = 0u;
		DraftServerBuildReply(server, SPARK_DRAFT_BRIDGE_SERVER_STATUS_INTERNAL, 0, 0u, 0, server->request.sequence_id, server->request.generation, 0u);
		DraftServerSend(server);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	elapsed = (SparkMonotonicNs() - server->request.received_ns) / 1000u;
	server->reply_counts = 1u;
	DraftServerBuildReply(server, server_status, nodes, node_count, info, server->request.sequence_id, server->request.generation,
		elapsed > UINT32_MAX ? UINT32_MAX : (uint32_t)elapsed);
	DraftServerSend(server);
	return SPARK_STATUS_OK;
}

void SparkDraftServerGetStats(const SparkDraftServer *server, SparkDraftServerStats *stats_out)
{
	if ( stats_out == 0 )
		return;
	memset(stats_out, 0, sizeof(*stats_out));
	if ( server != 0 )
		*stats_out = server->stats;
}
