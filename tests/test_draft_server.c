#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_draft_bridge.h"
#include "sparkpipe/spark_draft_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define TEST_TAP_ROW_BYTES 8u
#define TEST_MAX_COMMITTED 64u
#define TEST_MAX_NODES 16u

static int failures;

#define CHECK(condition, message) do { if ( condition ) printf("PASS %s\n", message); else { printf("FAIL %s\n", message); failures++; } } while ( 0 )

typedef struct TestAnswer
{
	uint32_t status;
	uint32_t count;
	SparkDraftBridgeNode nodes[TEST_MAX_NODES + 1u];
	SparkDraftBridgeProposalInfo info;
	uint32_t silent;
	SparkDraftServerRequest seen;
	uint8_t taps[2u * TEST_TAP_ROW_BYTES];
	uint32_t answered;
	SparkStatus answer_status;
} TestAnswer;

static void TestNap(void)
{
	const struct timespec wait = {0, 200000};
	nanosleep(&wait, 0);
}

static SparkDraftServer *TestServer(const char *target, uint32_t io_timeout_ms)
{
	SparkDraftServerConfig config;
	SparkDraftServer *server = 0;
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_DRAFT_SERVER_ABI_VERSION;
	config.descriptor_bytes = SPARK_DRAFT_SERVER_CONFIG_BYTES;
	config.host = "127.0.0.1";
	config.port = 0u;
	strcpy(config.target_model, target);
	config.tap_row_bytes = TEST_TAP_ROW_BYTES;
	config.max_committed_tokens = TEST_MAX_COMMITTED;
	config.max_tap_rows = 2u;
	config.max_nodes = TEST_MAX_NODES;
	config.io_timeout_ms = io_timeout_ms;
	return SparkDraftServerCreate(&config, &server) == SPARK_STATUS_OK ? server : 0;
}

static SparkDraftBridge *TestClient(uint32_t port, const char *target, uint32_t max_committed)
{
	SparkDraftBridgeConfig config;
	SparkDraftBridge *bridge = 0;
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_DRAFT_BRIDGE_ABI_VERSION;
	config.descriptor_bytes = SPARK_DRAFT_BRIDGE_CONFIG_BYTES;
	config.host = "127.0.0.1";
	config.port = port;
	strcpy(config.target_model, target);
	config.tap_row_bytes = TEST_TAP_ROW_BYTES;
	config.max_committed_tokens = max_committed;
	config.max_tap_rows = 2u;
	config.max_nodes = TEST_MAX_NODES;
	config.connect_timeout_ms = 500u;
	config.io_timeout_ms = 500u;
	return SparkDraftBridgeInitialize(&config, &bridge) == SPARK_STATUS_OK ? bridge : 0;
}

static SparkStatus TestExchange(SparkDraftBridge *bridge, SparkDraftServer *server, TestAnswer *answer, const uint32_t *committed, uint32_t count,
	const void *taps, uint32_t tap_rows, const SparkRowSampling *sampling, SparkDraftBridgeNode *nodes, uint32_t *node_count,
	SparkDraftBridgeProposalInfo *info)
{
	const uint64_t deadline = SparkMonotonicNs() + UINT64_C(3000000000);
	SparkStatus status = SparkDraftBridgeRequestTree(bridge, 6u, 41u, 7u, count - 1u, committed[count - 1u], committed, count, taps, tap_rows, 4u, 30u,
		8u, sampling, 9u);
	if ( status != SPARK_STATUS_OK )
		return status;
	answer->answered = 0u;
	while ( SparkMonotonicNs() < deadline )
	{
		SparkDraftServerRequest request;
		if ( SparkDraftServerAdvance(server, &request) == SPARK_STATUS_OK )
		{
			answer->seen = request;
			if ( request.tap_row_count != 0u )
				memcpy(answer->taps, request.tap_rows, (size_t)request.tap_row_count * TEST_TAP_ROW_BYTES);
			answer->answered = 1u;
			if ( answer->silent == 0u )
				answer->answer_status = SparkDraftServerAnswer(server, answer->status, answer->nodes, answer->count, &answer->info);
		}
		if ( answer->silent != 0u && answer->answered != 0u )
			return SPARK_STATUS_PENDING;
		status = SparkDraftBridgeReply(bridge, nodes, node_count, info);
		if ( status != SPARK_STATUS_PENDING )
			return status;
		TestNap();
	}
	return SPARK_STATUS_PENDING;
}

static void TestTree(TestAnswer *answer, uint32_t anchor)
{
	memset(answer, 0, sizeof(*answer));
	answer->status = SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK;
	answer->count = 3u;
	answer->nodes[0] = (SparkDraftBridgeNode){anchor, SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX, 0u, 2u, 1.0f};
	answer->nodes[1] = (SparkDraftBridgeNode){500u, 0u, 1u, 2u, 0.75f};
	answer->nodes[2] = (SparkDraftBridgeNode){501u, 1u, 2u, 2u, 0.5f};
	answer->info.drafter_us[0] = 1234u;
	answer->info.drafter_us[1] = 77u;
	answer->info.graft_us = 9u;
	answer->info.expansions = 3u;
}

static void TestRoundTrip(void)
{
	SparkDraftServer *server = TestServer("draft-test", 500u);
	SparkDraftBridge *bridge = server != 0 ? TestClient(SparkDraftServerPort(server), "draft-test", TEST_MAX_COMMITTED) : 0;
	const uint32_t committed[5] = {11u, 12u, 13u, 14u, 15u};
	const uint8_t taps[2u * TEST_TAP_ROW_BYTES] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	SparkRowSampling sampling;
	SparkDraftBridgeNode nodes[TEST_MAX_NODES];
	SparkDraftBridgeProposalInfo info;
	SparkDraftServerStats stats;
	TestAnswer answer;
	uint32_t count = 0u;
	CHECK(server != 0 && bridge != 0 && SparkDraftServerPort(server) != 0u, "the server listens on an ephemeral port and the client is configured");
	if ( server == 0 || bridge == 0 )
		return;
	memset(&sampling, 0, sizeof(sampling));
	sampling.seed = 99u;
	sampling.inverse_temperature = 1.25f;
	sampling.nucleus = 0.9f;
	sampling.top_k = 40u;
	TestTree(&answer, 15u);
	CHECK(TestExchange(bridge, server, &answer, committed, 5u, taps, 2u, &sampling, nodes, &count, &info) == SPARK_STATUS_OK && count == 3u,
		"a request is answered with the drafter's three nodes");
	CHECK(answer.seen.sequence_id == 41u && answer.seen.generation == 7u && answer.seen.position == 4u && answer.seen.anchor_token_id == 15u &&
		answer.seen.committed_token_count == 5u && memcmp(answer.seen.committed_token_ids, committed, sizeof(committed)) == 0 &&
		answer.seen.speculator_mask == 6u && answer.seen.depth == 4u && answer.seen.max_depth == 8u && answer.seen.node_capacity == 9u &&
		answer.seen.time_budget_ms == 30u, "the server reads the client's sequence, position, anchor, committed tokens, mask and limits");
	CHECK(answer.seen.sampling.seed == 99u && answer.seen.sampling.inverse_temperature == 1.25f && answer.seen.sampling.nucleus == 0.9f &&
		answer.seen.sampling.top_k == 40u, "the server reads the client's sampling rule");
	CHECK(answer.seen.tap_row_count == 2u && memcmp(answer.taps, taps, sizeof(taps)) == 0, "the server reads the tap rows that follow the tokens");
	CHECK(nodes[0].parent_index == SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX && nodes[1].token_id == 500u && nodes[2].parent_index == 1u &&
		nodes[2].depth == 2u && nodes[1].score == 0.75f && nodes[2].source_bit == 2u, "the client reads every node field the server wrote");
	CHECK(info.drafter_us[0] == 1234u && info.drafter_us[1] == 77u && info.graft_us == 9u && info.expansions == 3u &&
		info.server_status == SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK, "the footer carries the drafter stage times and the expansion count");
	TestTree(&answer, 15u);
	answer.status = SPARK_DRAFT_BRIDGE_SERVER_STATUS_UNAVAILABLE;
	answer.count = 0u;
	CHECK(TestExchange(bridge, server, &answer, committed, 5u, 0, 0u, 0, nodes, &count, &info) == SPARK_STATUS_VALIDATION_FAILED &&
		info.server_status == SPARK_DRAFT_BRIDGE_SERVER_STATUS_UNAVAILABLE && count == 0u, "an unavailable drafter answers UNAVAILABLE with no nodes");
	TestTree(&answer, 15u);
	answer.count = 12u;
	for ( uint32_t index = 3u; index < 12u; ++index )
		answer.nodes[index] = (SparkDraftBridgeNode){600u + index, index - 1u, index, 2u, 0.1f};
	CHECK(TestExchange(bridge, server, &answer, committed, 5u, 0, 0u, 0, nodes, &count, &info) == SPARK_STATUS_VALIDATION_FAILED &&
		info.server_status == SPARK_DRAFT_BRIDGE_SERVER_STATUS_INTERNAL && answer.answer_status == SPARK_STATUS_INVALID_ARGUMENT,
		"a drafter that answers more nodes than the request holds is refused and the client sees INTERNAL");
	TestTree(&answer, 15u);
	CHECK(TestExchange(bridge, server, &answer, committed, 5u, 0, 0u, 0, nodes, &count, &info) == SPARK_STATUS_OK && count == 3u,
		"the connection serves the next request after an unavailable and an internal answer");
	SparkDraftServerGetStats(server, &stats);
	CHECK(stats.connections == 1u && stats.requests == 4u && stats.answered == 3u && stats.bad_requests == 0u && stats.dropped == 0u,
		"one connection carried four requests, three answered by the drafter");
	SparkDraftBridgeDestroy(bridge);
	SparkDraftServerDestroy(server);
}

static void TestRefusals(void)
{
	SparkDraftServer *server = TestServer("draft-test", 500u);
	SparkDraftBridge *other = server != 0 ? TestClient(SparkDraftServerPort(server), "other-model", TEST_MAX_COMMITTED) : 0;
	SparkDraftBridge *wide = server != 0 ? TestClient(SparkDraftServerPort(server), "draft-test", 2u * TEST_MAX_COMMITTED) : 0;
	SparkDraftBridge *good = server != 0 ? TestClient(SparkDraftServerPort(server), "draft-test", TEST_MAX_COMMITTED) : 0;
	uint32_t committed[2u * TEST_MAX_COMMITTED];
	SparkDraftBridgeNode nodes[TEST_MAX_NODES];
	SparkDraftBridgeProposalInfo info;
	SparkDraftServerStats stats;
	TestAnswer answer;
	uint32_t count = 0u;
	if ( server == 0 || other == 0 || wide == 0 || good == 0 )
	{
		CHECK(0, "the refusal clients start");
		return;
	}
	for ( uint32_t index = 0u; index < 2u * TEST_MAX_COMMITTED; ++index )
		committed[index] = 1000u + index;
	TestTree(&answer, committed[4]);
	CHECK(TestExchange(other, server, &answer, committed, 5u, 0, 0u, 0, nodes, &count, &info) == SPARK_STATUS_VALIDATION_FAILED &&
		info.server_status == SPARK_DRAFT_BRIDGE_SERVER_STATUS_BAD_REQUEST && answer.answered == 0u,
		"a request that names another target model is refused before the drafter sees it");
	TestTree(&answer, committed[4]);
	CHECK(TestExchange(other, server, &answer, committed, 5u, 0, 0u, 0, nodes, &count, &info) == SPARK_STATUS_VALIDATION_FAILED &&
		info.server_status == SPARK_DRAFT_BRIDGE_SERVER_STATUS_BAD_REQUEST, "a refused request keeps its connection for the next one");
	TestTree(&answer, committed[TEST_MAX_COMMITTED]);
	CHECK(TestExchange(wide, server, &answer, committed, TEST_MAX_COMMITTED + 1u, 0, 0u, 0, nodes, &count, &info) != SPARK_STATUS_OK &&
		answer.answered == 0u, "a request with more committed tokens than the server holds is refused, and its connection is closed");
	TestTree(&answer, committed[4]);
	CHECK(TestExchange(good, server, &answer, committed, 5u, 0, 0u, 0, nodes, &count, &info) == SPARK_STATUS_OK && count == 3u,
		"a new connection replaces the old one and is served");
	SparkDraftServerGetStats(server, &stats);
	CHECK(stats.bad_requests == 3u && stats.requests == 1u && stats.connections == 3u, "the server counts three refusals over three connections");
	SparkDraftBridgeDestroy(other);
	SparkDraftBridgeDestroy(wide);
	SparkDraftBridgeDestroy(good);
	SparkDraftServerDestroy(server);
}

static void TestCancelAndStall(void)
{
	SparkDraftServer *server = TestServer("draft-test", 100u);
	SparkDraftBridge *bridge = server != 0 ? TestClient(SparkDraftServerPort(server), "draft-test", TEST_MAX_COMMITTED) : 0;
	const uint32_t committed[3] = {7u, 8u, 9u};
	SparkDraftBridgeNode nodes[TEST_MAX_NODES];
	SparkDraftBridgeProposalInfo info;
	SparkDraftServerRequest request;
	SparkDraftServerStats stats;
	TestAnswer answer;
	struct sockaddr_in address;
	uint64_t deadline;
	uint32_t count = 0u;
	int raw;
	if ( server == 0 || bridge == 0 )
	{
		CHECK(0, "the cancel client starts");
		return;
	}
	TestTree(&answer, 9u);
	answer.silent = 1u;
	CHECK(TestExchange(bridge, server, &answer, committed, 3u, 0, 0u, 0, nodes, &count, &info) == SPARK_STATUS_PENDING && answer.answered == 1u,
		"a request the drafter has not answered stays pending at the client");
	SparkDraftBridgeCancel(bridge);
	TestTree(&answer, 9u);
	CHECK(TestExchange(bridge, server, &answer, committed, 3u, 0, 0u, 0, nodes, &count, &info) == SPARK_STATUS_OK && count == 3u,
		"after the client abandons an exchange it reconnects, and the server serves the new connection");
	memset(&address, 0, sizeof(address));
	address.sin_family = AF_INET;
	address.sin_port = htons((uint16_t)SparkDraftServerPort(server));
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	raw = socket(AF_INET, SOCK_STREAM, 0);
	CHECK(raw >= 0 && connect(raw, (const struct sockaddr *)&address, (socklen_t)sizeof(address)) == 0 && send(raw, "DFT3\x02\x00", 6, 0) == 6,
		"a client sends half a request header and stops");
	deadline = SparkMonotonicNs() + UINT64_C(1000000000);
	stats.dropped = 0u;
	while ( SparkMonotonicNs() < deadline && stats.dropped < 2u )
	{
		(void)SparkDraftServerAdvance(server, &request);
		SparkDraftServerGetStats(server, &stats);
		TestNap();
	}
	CHECK(stats.dropped == 2u, "the server drops the abandoned exchange and, after its io timeout, the stalled one");
	if ( raw >= 0 )
		(void)close(raw);
	SparkDraftBridgeDestroy(bridge);
	SparkDraftServerDestroy(server);
}

static void TestConfig(void)
{
	SparkDraftServerConfig config;
	SparkDraftServer *server = (SparkDraftServer *)&config;
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_DRAFT_SERVER_ABI_VERSION;
	config.descriptor_bytes = SPARK_DRAFT_SERVER_CONFIG_BYTES;
	config.host = "127.0.0.1";
	config.max_committed_tokens = 8u;
	config.max_nodes = 8u;
	config.io_timeout_ms = 100u;
	CHECK(SparkDraftServerCreate(&config, &server) == SPARK_STATUS_INVALID_ARGUMENT && server == 0, "a server without a target model is refused");
	strcpy(config.target_model, "x");
	config.max_tap_rows = 1u;
	CHECK(SparkDraftServerCreate(&config, &server) == SPARK_STATUS_INVALID_ARGUMENT, "tap rows without a row size are refused");
	config.max_tap_rows = 0u;
	config.max_nodes = SPARK_DRAFT_BRIDGE_MAX_TREE_NODES + 1u;
	CHECK(SparkDraftServerCreate(&config, &server) == SPARK_STATUS_INVALID_ARGUMENT, "more nodes than a DFT3 tree holds are refused");
}

static void TestPipelinedEight(void)
{
	SparkDraftServer *server = TestServer("draft-test", 500u);
	SparkDraftBridge *bridge = server != 0 ? TestClient(SparkDraftServerPort(server), "draft-test", TEST_MAX_COMMITTED) : 0;
	uint32_t committed[SPARK_DRAFT_BRIDGE_MAX_PIPELINE][5];
	SparkDraftBridgeNode nodes[TEST_MAX_NODES];
	SparkDraftBridgeProposalInfo info;
	SparkDraftServerStats stats;
	const uint64_t deadline = SparkMonotonicNs() + UINT64_C(3000000000);
	uint32_t index, collected = 0u, matched = 1u;
	if ( server == 0 || bridge == 0 )
	{
		CHECK(0, "the pipelined server and client start");
		return;
	}
	for ( index = 0u; index < SPARK_DRAFT_BRIDGE_MAX_PIPELINE; ++index )
	{
		for ( uint32_t pos = 0u; pos < 5u; ++pos )
			committed[index][pos] = 100u * (index + 1u) + pos;
		CHECK(SparkDraftBridgeEnqueueTree(bridge, 2u, 1000u + index, 1u + index, 4u, committed[index][4], committed[index], 5u, 0, 0u, 4u, 30u, 8u, 0, 9u) ==
			SPARK_STATUS_OK, "pipelined request enqueues");
	}
	while ( SparkMonotonicNs() < deadline && collected < SPARK_DRAFT_BRIDGE_MAX_PIPELINE )
	{
		SparkDraftServerRequest request;
		while ( SparkDraftServerAdvance(server, &request) == SPARK_STATUS_OK )
		{
			TestAnswer answer;
			TestTree(&answer, request.anchor_token_id);
			answer.nodes[1].token_id = (uint32_t)(request.sequence_id + 500u);
			(void)SparkDraftServerAnswer(server, answer.status, answer.nodes, answer.count, &answer.info);
		}
		for ( ;; )
		{
			uint32_t count = 0u;
			const SparkStatus status = SparkDraftBridgeReply(bridge, nodes, &count, &info);
			if ( status == SPARK_STATUS_PENDING || status == SPARK_STATUS_NOT_FOUND )
				break;
			if ( status != SPARK_STATUS_OK || count != 3u || nodes[0].token_id != committed[collected][4] ||
				nodes[1].token_id != 1500u + collected )
				matched = 0u;
			collected++;
		}
		TestNap();
	}
	SparkDraftServerGetStats(server, &stats);
	CHECK(collected == SPARK_DRAFT_BRIDGE_MAX_PIPELINE && matched != 0u && stats.connections == 1u && stats.requests == SPARK_DRAFT_BRIDGE_MAX_PIPELINE &&
		stats.answered == SPARK_DRAFT_BRIDGE_MAX_PIPELINE && stats.bad_requests == 0u && stats.dropped == 0u,
		"eight pipelined requests are answered in FIFO order over one connection");
	SparkDraftBridgeDestroy(bridge);
	SparkDraftServerDestroy(server);
}

int main(void)
{
	TestConfig();
	TestRoundTrip();
	TestRefusals();
	TestCancelAndStall();
	TestPipelinedEight();
	if ( failures != 0 )
	{
		printf("FAIL test_draft_server: %d failures\n", failures);
		return 1;
	}
	printf("PASS test_draft_server: the C server reads every request field the C client writes, answers trees, refusals, unavailable and internal "
		"statuses, survives abandoned and stalled exchanges, and serves the next connection\n");
	return 0;
}
