#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_tap_stream_reader.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define TEST_HIDDEN 4u
#define TEST_ENTRIES 3u
#define TEST_ROW_ELEMENTS (TEST_ENTRIES * TEST_HIDDEN)

static int failures;

#define CHECK(condition, message) do { if ( condition ) printf("PASS %s\n", message); else { printf("FAIL %s\n", message); failures++; } } while ( 0 )

static void TestHello(SparkTapStreamHello *hello, const char *tag)
{
	memset(hello, 0, sizeof(*hello));
	hello->magic = SPARK_TAP_STREAM_MAGIC;
	hello->version = SPARK_TAP_STREAM_VERSION;
	hello->kind = SPARK_TAP_STREAM_KIND_HELLO;
	hello->header_bytes = sizeof(*hello);
	strcpy(hello->model_tag, tag);
	hello->pack_identity[0] = 0xab;
	hello->hidden = TEST_HIDDEN;
	hello->total_layers = 12u;
	hello->dtype = SPARK_TAP_STREAM_DTYPE_BF16;
	hello->entry_count = TEST_ENTRIES;
	hello->tap_rank = 15u;
	hello->tp_degree = 16u;
	hello->entries[0] = (SparkTapStreamEntry){1u, 3u};
	hello->entries[1] = (SparkTapStreamEntry){1u, 7u};
	hello->entries[2] = (SparkTapStreamEntry){3u, 12u};
	hello->fingerprint = SparkTapStreamFingerprint(hello);
}

static size_t TestFrame(uint8_t *out, const SparkTapStreamHello *hello, uint64_t sequence, uint64_t first, uint32_t rows, uint32_t flags,
	uint64_t serial)
{
	SparkTapStreamFrame frame;
	uint8_t *cursor = out;
	memset(&frame, 0, sizeof(frame));
	frame.magic = SPARK_TAP_STREAM_MAGIC;
	frame.version = SPARK_TAP_STREAM_VERSION;
	frame.kind = SPARK_TAP_STREAM_KIND_FRAME;
	frame.header_bytes = sizeof(frame);
	frame.fingerprint = hello->fingerprint;
	frame.control_generation = 2u;
	frame.request_id = 900u + sequence;
	frame.request_generation = 1u;
	frame.sequence_id = sequence;
	frame.first_position = first;
	frame.serial = serial;
	frame.row_count = rows;
	frame.flags = flags;
	frame.tail_token = 77u;
	frame.entry_count = hello->entry_count;
	frame.payload_bytes = (uint64_t)rows * (sizeof(uint32_t) + TEST_ROW_ELEMENTS * sizeof(uint16_t));
	memcpy(cursor, &frame, sizeof(frame));
	cursor += sizeof(frame);
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		const uint32_t token = (uint32_t)(5000u + first + row);
		memcpy(cursor, &token, sizeof(token));
		cursor += sizeof(token);
	}
	for ( uint32_t row = 0u; row < rows; ++row )
		for ( uint32_t element = 0u; element < TEST_ROW_ELEMENTS; ++element )
		{
			const uint16_t value = (uint16_t)((first + row) * 100u + element);
			memcpy(cursor, &value, sizeof(value));
			cursor += sizeof(value);
		}
	return (size_t)(cursor - out);
}

static SparkTapStreamReader *TestReader(const char *path, const char *required)
{
	SparkTapStreamReaderConfig config;
	SparkTapStreamReader *reader = 0;
	memset(&config, 0, sizeof(config));
	config.abi_version = SPARK_TAP_STREAM_READER_ABI_VERSION;
	config.descriptor_bytes = SPARK_TAP_STREAM_READER_CONFIG_BYTES;
	if ( path != 0 )
		config.path = path;
	else
		config.host = "127.0.0.1";
	strcpy(config.model_tag, "tap-test");
	config.hidden = TEST_HIDDEN;
	config.max_rows = 16u;
	config.io_timeout_ms = 200u;
	if ( SparkTapStreamReaderParseEntries(required, config.required, SPARK_TAP_STREAM_MAX_ENTRIES, &config.required_count) != SPARK_STATUS_OK )
		return 0;
	return SparkTapStreamReaderCreate(&config, &reader) == SPARK_STATUS_OK ? reader : 0;
}

static int TestConnect(uint32_t port)
{
	struct sockaddr_in address;
	const int fd = socket(AF_INET, SOCK_STREAM, 0);
	memset(&address, 0, sizeof(address));
	address.sin_family = AF_INET;
	address.sin_port = htons((uint16_t)port);
	address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	if ( fd < 0 || connect(fd, (const struct sockaddr *)&address, (socklen_t)sizeof(address)) != 0 )
	{
		if ( fd >= 0 )
			(void)close(fd);
		return -1;
	}
	return fd;
}

static uint32_t TestCollect(SparkTapStreamReader *reader, SparkTapStreamReaderFrame *frames, uint8_t (*copies)[512], uint32_t want)
{
	const uint64_t deadline = SparkMonotonicNs() + UINT64_C(1000000000);
	const struct timespec nap = {0, 200000};
	uint32_t got = 0u;
	while ( got < want && SparkMonotonicNs() < deadline )
	{
		if ( SparkTapStreamReaderAdvance(reader, &frames[got]) == SPARK_STATUS_OK )
		{
			memcpy(copies[got], frames[got].tokens, (size_t)frames[got].header.row_count * sizeof(uint32_t));
			memcpy(copies[got] + 64, frames[got].rows, (size_t)frames[got].header.row_count * TEST_ROW_ELEMENTS * sizeof(uint16_t));
			got++;
			continue;
		}
		nanosleep(&nap, 0);
	}
	return got;
}

static uint32_t TestRowsMatch(const uint8_t *copy, uint64_t first, uint32_t rows)
{
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		uint32_t token;
		memcpy(&token, copy + 4u * row, sizeof(token));
		if ( token != 5000u + first + row )
			return 0u;
		for ( uint32_t element = 0u; element < TEST_ROW_ELEMENTS; ++element )
		{
			uint16_t value;
			memcpy(&value, copy + 64 + 2u * ((size_t)row * TEST_ROW_ELEMENTS + element), sizeof(value));
			if ( value != (uint16_t)((first + row) * 100u + element) )
				return 0u;
		}
	}
	return 1u;
}

static void TestEntries(void)
{
	SparkTapStreamEntry entries[4];
	uint32_t count = 0u;
	CHECK(SparkTapStreamReaderParseEntries("o19,o37,final", entries, 4u, &count) == SPARK_STATUS_OK && count == 3u && entries[0].form == 1u &&
		entries[0].layer == 19u && entries[2].form == 3u, "entries parse as layer outputs and the final hidden");
	CHECK(SparkTapStreamReaderParseEntries("o19,o19", entries, 4u, &count) == SPARK_STATUS_DUPLICATE, "a repeated entry is refused");
	CHECK(SparkTapStreamReaderParseEntries("o19,", entries, 4u, &count) == SPARK_STATUS_PARSE_ERROR &&
		SparkTapStreamReaderParseEntries("x3", entries, 4u, &count) == SPARK_STATUS_PARSE_ERROR &&
		SparkTapStreamReaderParseEntries("", entries, 4u, &count) == SPARK_STATUS_PARSE_ERROR, "malformed entry lists are refused");
	CHECK(SparkTapStreamReaderParseEntries("o1,o2,o3,o4,o5", entries, 4u, &count) == SPARK_STATUS_CAPACITY_EXCEEDED, "too many entries are refused");
}

static void TestSocket(void)
{
	SparkTapStreamReader *reader = TestReader(0, "final,o7");
	SparkTapStreamReaderFrame frames[4];
	SparkTapStreamReaderStats stats;
	SparkTapStreamHello hello;
	static uint8_t copies[4][512];
	uint8_t buffer[4096];
	size_t bytes;
	int fd;
	CHECK(reader != 0 && SparkTapStreamReaderPort(reader) != 0u, "the tap reader listens on an ephemeral port");
	if ( reader == 0 )
		return;
	fd = TestConnect(SparkTapStreamReaderPort(reader));
	TestHello(&hello, "tap-test");
	bytes = 0u;
	memcpy(buffer, &hello, sizeof(hello));
	bytes += sizeof(hello);
	bytes += TestFrame(buffer + bytes, &hello, 3u, 0u, 4u, SPARK_TAP_STREAM_FLAG_PREFILL, 0u);
	bytes += TestFrame(buffer + bytes, &hello, 3u, 4u, 1u, SPARK_TAP_STREAM_FLAG_DECODE, 1u);
	bytes += TestFrame(buffer + bytes, &hello, 3u, 5u, 2u, SPARK_TAP_STREAM_FLAG_VERIFY, 2u);
	CHECK(fd >= 0 && send(fd, buffer, 300u, 0) == 300 && send(fd, buffer + 300u, bytes - 300u, 0) == (ssize_t)(bytes - 300u),
		"the tap rank sends a hello and three frames in two pieces");
	CHECK(TestCollect(reader, frames, copies, 3u) == 3u, "the reader returns the three frames");
	CHECK(frames[0].header.row_count == 4u && frames[1].header.first_position == 4u && frames[2].header.flags == SPARK_TAP_STREAM_FLAG_VERIFY &&
		frames[0].header.sequence_id == 3u && frames[0].header.request_id == 903u && frames[2].header.serial == 2u,
		"each frame carries its sequence, request, position, rows, flags and serial");
	CHECK(TestRowsMatch(copies[0], 0u, 4u) != 0u && TestRowsMatch(copies[1], 4u, 1u) != 0u && TestRowsMatch(copies[2], 5u, 2u) != 0u,
		"each frame's tokens and rows are the bytes the tap rank sent");
	CHECK(frames[0].required_count == 2u && frames[0].required_index[0] == 2u && frames[0].required_index[1] == 1u &&
		frames[0].row_elements == TEST_ROW_ELEMENTS && frames[0].hello->tap_rank == 15u,
		"the frame maps each required entry to its place in the stream's rows");
	bytes = TestFrame(buffer, &hello, 3u, 7u, 1u, SPARK_TAP_STREAM_FLAG_DECODE, 3u);
	((SparkTapStreamFrame *)(void *)buffer)->fingerprint ^= 1u;
	CHECK(send(fd, buffer, bytes, 0) == (ssize_t)bytes && TestCollect(reader, frames, copies, 1u) == 0u,
		"a frame whose fingerprint differs from the hello's is not returned");
	SparkTapStreamReaderGetStats(reader, &stats);
	CHECK(stats.refused == 1u && stats.frames == 3u && stats.rows == 7u && stats.streams == 1u, "the reader refused the stream and counts what it read");
	if ( fd >= 0 )
		(void)close(fd);
	fd = TestConnect(SparkTapStreamReaderPort(reader));
	TestHello(&hello, "tap-test");
	hello.entries[1].layer = 8u;
	hello.fingerprint = SparkTapStreamFingerprint(&hello);
	CHECK(fd >= 0 && send(fd, &hello, sizeof(hello), 0) == (ssize_t)sizeof(hello) && TestCollect(reader, frames, copies, 1u) == 0u,
		"a stream that lacks a required entry is refused at its hello");
	SparkTapStreamReaderGetStats(reader, &stats);
	CHECK(stats.refused == 2u && stats.streams == 1u, "the missing entry counts as a refusal");
	if ( fd >= 0 )
		(void)close(fd);
	fd = TestConnect(SparkTapStreamReaderPort(reader));
	TestHello(&hello, "tap-test");
	bytes = 0u;
	memcpy(buffer, &hello, sizeof(hello));
	bytes += sizeof(hello);
	bytes += TestFrame(buffer + bytes, &hello, 4u, 0u, 2u, SPARK_TAP_STREAM_FLAG_PREFILL, 0u);
	CHECK(fd >= 0 && send(fd, buffer, bytes, 0) == (ssize_t)bytes && TestCollect(reader, frames, copies, 1u) == 1u &&
		frames[0].header.sequence_id == 4u && TestRowsMatch(copies[0], 0u, 2u) != 0u, "a reconnecting tap rank is read again from its new hello");
	bytes = TestFrame(buffer, &hello, 4u, 2u, 1u, SPARK_TAP_STREAM_FLAG_DECODE, 1u);
	CHECK(send(fd, buffer, 40u, 0) == 40 && TestCollect(reader, frames, copies, 1u) == 0u, "a frame cut off mid-header is not returned");
	SparkTapStreamReaderGetStats(reader, &stats);
	CHECK(stats.refused == 3u, "a stream that stalls inside a frame is closed after the io timeout");
	if ( fd >= 0 )
		(void)close(fd);
	SparkTapStreamReaderDestroy(reader);
}

static void TestFile(void)
{
	char path[] = "/tmp/test_tap_stream_reader_XXXXXX";
	static uint8_t buffer[8192], copies[4][512];
	SparkTapStreamReaderFrame frames[4];
	SparkTapStreamHello hello;
	SparkTapStreamReader *reader;
	size_t bytes = 0u, cut;
	const int fd = mkstemp(path);
	TestHello(&hello, "tap-test");
	memcpy(buffer, &hello, sizeof(hello));
	bytes += sizeof(hello);
	bytes += TestFrame(buffer + bytes, &hello, 9u, 0u, 3u, SPARK_TAP_STREAM_FLAG_PREFILL, 0u);
	bytes += TestFrame(buffer + bytes, &hello, 9u, 3u, 1u, SPARK_TAP_STREAM_FLAG_DECODE | SPARK_TAP_STREAM_FLAG_CHAIN, 1u);
	cut = TestFrame(buffer + bytes, &hello, 9u, 4u, 2u, SPARK_TAP_STREAM_FLAG_DECODE, 2u);
	bytes += cut - 10u;
	CHECK(fd >= 0 && write(fd, buffer, bytes) == (ssize_t)bytes, "a recording with a frame cut short is written");
	if ( fd >= 0 )
		(void)close(fd);
	reader = TestReader(path, "o3");
	CHECK(reader != 0 && SparkTapStreamReaderAdvance(reader, &frames[0]) == SPARK_STATUS_OK && frames[0].header.row_count == 3u,
		"the file reader returns the recording's first frame");
	if ( reader != 0 )
	{
		memcpy(copies[0], frames[0].tokens, 12u);
		memcpy(copies[0] + 64, frames[0].rows, 3u * TEST_ROW_ELEMENTS * sizeof(uint16_t));
		CHECK(TestRowsMatch(copies[0], 0u, 3u) != 0u, "the recorded rows read back as written");
		CHECK(SparkTapStreamReaderAdvance(reader, &frames[1]) == SPARK_STATUS_OK && frames[1].header.flags == 0xau,
			"the second frame keeps its decode and chain flags");
		CHECK(SparkTapStreamReaderAdvance(reader, &frames[2]) == SPARK_STATUS_NOT_FOUND &&
			SparkTapStreamReaderAdvance(reader, &frames[2]) == SPARK_STATUS_NOT_FOUND, "the cut frame ends the recording");
		SparkTapStreamReaderDestroy(reader);
	}
	reader = TestReader(path, "o5");
	CHECK(reader != 0 && SparkTapStreamReaderAdvance(reader, &frames[0]) == SPARK_STATUS_VALIDATION_FAILED,
		"a recording without a required entry fails at its hello");
	SparkTapStreamReaderDestroy(reader);
	(void)unlink(path);
}

int main(void)
{
	TestEntries();
	TestSocket();
	TestFile();
	if ( failures != 0 )
	{
		printf("FAIL test_tap_stream_reader: %d failures\n", failures);
		return 1;
	}
	printf("PASS test_tap_stream_reader: hellos and frames are read from a socket in pieces and from recordings, mapped to the required entries, "
		"and refused on a fingerprint, a missing entry, a stall or a cut frame\n");
	return 0;
}
