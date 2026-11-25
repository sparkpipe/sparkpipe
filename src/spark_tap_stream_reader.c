#include "sparkpipe/spark_tap_stream_reader.h"
#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_error_site.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the tap stream is little-endian and is read in place");

#define TAP_READER_CLOSED (-1)
#define TAP_READER_NS_PER_MS UINT64_C(1000000)
#define TAP_READER_RX_BUFFER_BYTES 65536u
#define TAP_READER_SOCKBUF_BYTES (512 * 1024)
#define TAP_READER_FORM_OUTPUT 1u
#define TAP_READER_FORM_FINAL 3u
#define TAP_READER_FINAL_NAME "final"

#define TAP_READER_PHASE_IDLE 0u
#define TAP_READER_PHASE_HELLO 1u
#define TAP_READER_PHASE_HEADER 2u
#define TAP_READER_PHASE_PAYLOAD 3u

struct SparkTapStreamReader
{
	char model_tag[SPARK_TAP_STREAM_NAME_BYTES];
	uint32_t hidden;
	uint32_t required_count;
	SparkTapStreamEntry required[SPARK_TAP_STREAM_MAX_ENTRIES];
	uint32_t required_index[SPARK_TAP_STREAM_MAX_ENTRIES];
	uint32_t max_rows;
	uint32_t io_timeout_ms;
	uint32_t port;
	uint32_t file;
	char label[256];
	int listen_fd;
	int fd;
	uint32_t phase;
	uint64_t deadline_ns;
	SparkTapStreamHello hello;
	SparkTapStreamFrame frame;
	uint64_t got;
	uint8_t *rx_buffer;
	uint64_t rx_head;
	uint64_t rx_tail;
	uint8_t *payload;
	uint64_t payload_capacity;
	uint64_t row_bytes;
	SparkTapStreamReaderStats stats;
};

static uint32_t TapReaderSameEntry(SparkTapStreamEntry a, SparkTapStreamEntry b)
{
	return a.form == b.form && (a.form != TAP_READER_FORM_OUTPUT || a.layer == b.layer) ? 1u : 0u;
}

static void TapReaderEntryName(SparkTapStreamEntry entry, char *text, size_t bytes)
{
	if ( entry.form == TAP_READER_FORM_FINAL )
		(void)snprintf(text, bytes, "final");
	else
		(void)snprintf(text, bytes, "%s%u", entry.form == TAP_READER_FORM_OUTPUT ? "o" : "form?", entry.layer);
}

SparkStatus SparkTapStreamReaderParseEntries(const char *text, SparkTapStreamEntry *entries, uint32_t capacity, uint32_t *count_out)
{
	uint32_t count = 0u;
	const char *cursor = text;
	if ( text == 0 || entries == 0 || count_out == 0 || capacity == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	*count_out = 0u;
	while ( *cursor != '\0' )
	{
		const char *end = strchr(cursor, ',');
		const size_t length = end != 0 ? (size_t)(end - cursor) : strlen(cursor);
		if ( count == capacity )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( length == sizeof(TAP_READER_FINAL_NAME) - 1u && memcmp(cursor, TAP_READER_FINAL_NAME, sizeof(TAP_READER_FINAL_NAME) - 1u) == 0 )
		{
			entries[count].form = TAP_READER_FORM_FINAL;
			entries[count].layer = 0u;
		}
		else if ( length >= 2u && length <= 6u && cursor[0] == 'o' )
		{
			uint32_t layer = 0u;
			for ( size_t index = 1u; index < length; ++index )
			{
				if ( cursor[index] < '0' || cursor[index] > '9' )
					SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
				layer = layer * 10u + (uint32_t)(cursor[index] - '0');
			}
			if ( layer > UINT16_MAX )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			entries[count].form = TAP_READER_FORM_OUTPUT;
			entries[count].layer = (uint16_t)layer;
		}
		else
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		for ( uint32_t other = 0u; other < count; ++other )
			if ( TapReaderSameEntry(entries[other], entries[count]) != 0u )
				SPARK_FAIL(SPARK_STATUS_DUPLICATE);
		count++;
		if ( end == 0 )
			break;
		cursor = end + 1u;
		if ( *cursor == '\0' )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	if ( count == 0u )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	*count_out = count;
	return SPARK_STATUS_OK;
}

static void TapReaderQuickAck(int fd)
{
#ifdef TCP_QUICKACK
	int enabled = 1;
	(void)setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &enabled, (socklen_t)sizeof(enabled));
#else
	(void)fd;
#endif
}

static void TapReaderClose(SparkTapStreamReader *reader)
{
	if ( reader->fd != TAP_READER_CLOSED )
		(void)close(reader->fd);
	reader->fd = TAP_READER_CLOSED;
	reader->phase = TAP_READER_PHASE_IDLE;
	reader->got = 0u;
	reader->rx_head = 0u;
	reader->rx_tail = 0u;
	reader->deadline_ns = 0u;
}

static void TapReaderRefuse(SparkTapStreamReader *reader, const char *reason)
{
	fprintf(stderr, "sparkpipe_tap_reader: %s refused the tap stream: %s\n", reader->label, reason);
	reader->stats.refused++;
	if ( reader->file == 0u )
		TapReaderClose(reader);
	else
		reader->phase = TAP_READER_PHASE_IDLE;
}

static uint32_t TapReaderHelloDone(SparkTapStreamReader *reader)
{
	const SparkTapStreamHello *hello = &reader->hello;
	char missing[512];
	size_t used = 0u;
	if ( hello->magic != SPARK_TAP_STREAM_MAGIC || hello->kind != SPARK_TAP_STREAM_KIND_HELLO || hello->header_bytes != sizeof(SparkTapStreamHello) ||
		hello->version != SPARK_TAP_STREAM_VERSION )
	{
		TapReaderRefuse(reader, "the stream does not start with an SPT2 hello of this version");
		return 0u;
	}
	if ( hello->dtype != SPARK_TAP_STREAM_DTYPE_BF16 || hello->entry_count == 0u || hello->entry_count > SPARK_TAP_STREAM_MAX_ENTRIES ||
		hello->hidden == 0u || SparkTapStreamFingerprint(hello) != hello->fingerprint )
	{
		TapReaderRefuse(reader, "the hello's dtype, entry count, hidden width or fingerprint is not valid");
		return 0u;
	}
	if ( (reader->model_tag[0] != '\0' && strncmp(hello->model_tag, reader->model_tag, sizeof(reader->model_tag)) != 0) ||
		(reader->hidden != 0u && hello->hidden != reader->hidden) )
	{
		TapReaderRefuse(reader, "the hello names another model or hidden width");
		return 0u;
	}
	missing[0] = '\0';
	for ( uint32_t need = 0u; need < reader->required_count; ++need )
	{
		uint32_t found = UINT32_MAX;
		for ( uint32_t index = 0u; index < hello->entry_count && found == UINT32_MAX; ++index )
			if ( TapReaderSameEntry(hello->entries[index], reader->required[need]) != 0u )
				found = index;
		reader->required_index[need] = found;
		if ( found == UINT32_MAX && used + 16u < sizeof(missing) )
		{
			char name[16];
			TapReaderEntryName(reader->required[need], name, sizeof(name));
			used += (size_t)snprintf(missing + used, sizeof(missing) - used, "%s%s", used != 0u ? "," : "", name);
		}
	}
	if ( missing[0] != '\0' )
	{
		char reason[640];
		(void)snprintf(reason, sizeof(reason), "it lacks the drafters' entries %s", missing);
		TapReaderRefuse(reader, reason);
		return 0u;
	}
	reader->row_bytes = (uint64_t)hello->entry_count * hello->hidden * sizeof(uint16_t);
	if ( reader->payload_capacity < (uint64_t)reader->max_rows * (sizeof(uint32_t) + reader->row_bytes) )
	{
		uint8_t *grown;
		const uint64_t capacity = (uint64_t)reader->max_rows * (sizeof(uint32_t) + reader->row_bytes);
		grown = (uint8_t *)realloc(reader->payload, (size_t)capacity);
		if ( grown == 0 )
		{
			TapReaderRefuse(reader, "its largest frame does not fit in host memory");
			return 0u;
		}
		reader->payload = grown;
		reader->payload_capacity = capacity;
	}
	reader->stats.streams++;
	reader->phase = TAP_READER_PHASE_HEADER;
	reader->got = 0u;
	reader->deadline_ns = 0u;
	return 1u;
}

static uint32_t TapReaderHeaderDone(SparkTapStreamReader *reader)
{
	const SparkTapStreamFrame *frame = &reader->frame;
	if ( frame->magic != SPARK_TAP_STREAM_MAGIC || frame->kind != SPARK_TAP_STREAM_KIND_FRAME || frame->header_bytes != sizeof(SparkTapStreamFrame) ||
		frame->version != SPARK_TAP_STREAM_VERSION || frame->fingerprint != reader->hello.fingerprint ||
		frame->entry_count != reader->hello.entry_count )
	{
		TapReaderRefuse(reader, "a frame's header does not match the stream's hello");
		return 0u;
	}
	if ( frame->row_count == 0u || frame->row_count > reader->max_rows ||
		frame->payload_bytes != (uint64_t)frame->row_count * (sizeof(uint32_t) + reader->row_bytes) )
	{
		TapReaderRefuse(reader, "a frame's row count or payload size is not valid");
		return 0u;
	}
	reader->phase = TAP_READER_PHASE_PAYLOAD;
	reader->got = 0u;
	return 1u;
}

static void TapReaderFrameOut(SparkTapStreamReader *reader, SparkTapStreamReaderFrame *frame_out)
{
	memset(frame_out, 0, sizeof(*frame_out));
	frame_out->header = reader->frame;
	frame_out->hello = &reader->hello;
	frame_out->tokens = (const uint32_t *)(const void *)reader->payload;
	frame_out->rows = (const uint16_t *)(const void *)(reader->payload + (uint64_t)reader->frame.row_count * sizeof(uint32_t));
	frame_out->row_elements = reader->row_bytes / sizeof(uint16_t);
	frame_out->required_index = reader->required_index;
	frame_out->required_count = reader->required_count;
	reader->stats.frames++;
	reader->stats.rows += reader->frame.row_count;
	reader->stats.bytes += sizeof(SparkTapStreamFrame) + reader->frame.payload_bytes;
	reader->phase = TAP_READER_PHASE_HEADER;
	reader->got = 0u;
	reader->deadline_ns = 0u;
}

static int TapReaderReadSome(SparkTapStreamReader *reader, uint8_t *destination, uint64_t want)
{
	while ( reader->got < want )
	{
		const uint64_t need = want - reader->got;
		const uint64_t avail = reader->rx_tail - reader->rx_head;
		ssize_t part;
		if ( avail != 0u )
		{
			const uint64_t take = avail < need ? avail : need;
			if ( reader->deadline_ns == 0u && reader->file == 0u )
				reader->deadline_ns = SparkMonotonicNs() + (uint64_t)reader->io_timeout_ms * TAP_READER_NS_PER_MS;
			memcpy(destination + reader->got, reader->rx_buffer + reader->rx_head, (size_t)take);
			reader->got += take;
			reader->rx_head += take;
			if ( reader->rx_head == reader->rx_tail )
			{
				reader->rx_head = 0u;
				reader->rx_tail = 0u;
			}
			continue;
		}
		reader->rx_head = 0u;
		reader->rx_tail = 0u;
		if ( reader->file != 0u || need >= TAP_READER_RX_BUFFER_BYTES || reader->rx_buffer == 0 )
			part = reader->file != 0u ? read(reader->fd, destination + reader->got, (size_t)need) :
				recv(reader->fd, destination + reader->got, (size_t)need, 0);
		else
			part = recv(reader->fd, reader->rx_buffer, TAP_READER_RX_BUFFER_BYTES, 0);
		if ( part > 0 )
		{
			if ( reader->file == 0u )
				TapReaderQuickAck(reader->fd);
			if ( reader->deadline_ns == 0u && reader->file == 0u )
				reader->deadline_ns = SparkMonotonicNs() + (uint64_t)reader->io_timeout_ms * TAP_READER_NS_PER_MS;
			if ( reader->file != 0u || need >= TAP_READER_RX_BUFFER_BYTES || reader->rx_buffer == 0 )
				reader->got += (uint64_t)part;
			else
				reader->rx_tail = (uint64_t)part;
			continue;
		}
		if ( part < 0 && errno == EINTR )
			continue;
		if ( part < 0 && reader->file == 0u && (errno == EAGAIN || errno == EWOULDBLOCK) )
			return 0;
		return -1;
	}
	return 1;
}

static SparkStatus TapReaderNonBlocking(int fd)
{
	const int flags = fcntl(fd, F_GETFL, 0);
	if ( flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return SPARK_STATUS_OK;
}

static void TapReaderAccept(SparkTapStreamReader *reader)
{
	int enabled = 1;
	int buf_bytes = TAP_READER_SOCKBUF_BYTES;
	const int fd = accept(reader->listen_fd, 0, 0);
	if ( fd < 0 )
		return;
	if ( TapReaderNonBlocking(fd) != SPARK_STATUS_OK )
	{
		(void)close(fd);
		return;
	}
	(void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enabled, (socklen_t)sizeof(enabled));
	TapReaderQuickAck(fd);
	(void)setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf_bytes, (socklen_t)sizeof(buf_bytes));
	TapReaderClose(reader);
	reader->fd = fd;
	reader->phase = TAP_READER_PHASE_HELLO;
	reader->got = 0u;
}

static SparkStatus TapReaderListen(SparkTapStreamReader *reader, const char *host, uint32_t port)
{
	struct addrinfo hints, *addresses = 0;
	struct sockaddr_storage bound;
	socklen_t bound_bytes = (socklen_t)sizeof(bound);
	char port_text[8];
	int enabled = 1;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
	(void)snprintf(port_text, sizeof(port_text), "%u", port);
	if ( getaddrinfo(host, port_text, &hints, &addresses) != 0 || addresses == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	reader->listen_fd = socket(addresses->ai_family, addresses->ai_socktype, addresses->ai_protocol);
	if ( reader->listen_fd < 0 || setsockopt(reader->listen_fd, SOL_SOCKET, SO_REUSEADDR, &enabled, (socklen_t)sizeof(enabled)) != 0 ||
		bind(reader->listen_fd, addresses->ai_addr, addresses->ai_addrlen) != 0 || listen(reader->listen_fd, 2) != 0 ||
		TapReaderNonBlocking(reader->listen_fd) != SPARK_STATUS_OK ||
		getsockname(reader->listen_fd, (struct sockaddr *)&bound, &bound_bytes) != 0 )
	{
		fprintf(stderr, "sparkpipe_tap_reader: cannot listen on %s:%u: %s\n", host, port, strerror(errno));
		freeaddrinfo(addresses);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	freeaddrinfo(addresses);
	reader->port = bound.ss_family == AF_INET6 ? ntohs(((const struct sockaddr_in6 *)(const void *)&bound)->sin6_port) :
		ntohs(((const struct sockaddr_in *)(const void *)&bound)->sin_port);
	return SPARK_STATUS_OK;
}

SparkStatus SparkTapStreamReaderCreate(const SparkTapStreamReaderConfig *config, SparkTapStreamReader **reader_out)
{
	SparkTapStreamReader *reader;
	if ( reader_out != 0 )
		*reader_out = 0;
	if ( config == 0 || reader_out == 0 || config->abi_version != SPARK_TAP_STREAM_READER_ABI_VERSION ||
		config->descriptor_bytes != SPARK_TAP_STREAM_READER_CONFIG_BYTES || (config->path == 0) == (config->host == 0) ||
		config->port > 65535u || config->required_count > SPARK_TAP_STREAM_MAX_ENTRIES || config->max_rows == 0u || config->io_timeout_ms == 0u ||
		memchr(config->model_tag, '\0', sizeof(config->model_tag)) == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	reader = (SparkTapStreamReader *)calloc(1u, sizeof(*reader));
	if ( reader == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memcpy(reader->model_tag, config->model_tag, sizeof(reader->model_tag));
	reader->hidden = config->hidden;
	reader->required_count = config->required_count;
	memcpy(reader->required, config->required, sizeof(reader->required));
	reader->max_rows = config->max_rows;
	reader->io_timeout_ms = config->io_timeout_ms;
	reader->listen_fd = TAP_READER_CLOSED;
	reader->fd = TAP_READER_CLOSED;
	if ( config->path != 0 )
	{
		reader->file = 1u;
		reader->fd = open(config->path, O_RDONLY);
		if ( reader->fd < 0 )
		{
			fprintf(stderr, "sparkpipe_tap_reader: cannot open %s: %s\n", config->path, strerror(errno));
			SparkTapStreamReaderDestroy(reader);
			SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
		}
		reader->phase = TAP_READER_PHASE_HELLO;
		(void)snprintf(reader->label, sizeof(reader->label), "%s", config->path);
	}
	else
	{
		reader->rx_buffer = (uint8_t *)malloc(TAP_READER_RX_BUFFER_BYTES);
		if ( reader->rx_buffer == 0 || TapReaderListen(reader, config->host, config->port) != SPARK_STATUS_OK )
		{
			SparkTapStreamReaderDestroy(reader);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
	}
	if ( reader->file == 0u )
		(void)snprintf(reader->label, sizeof(reader->label), "port %u", reader->port);
	*reader_out = reader;
	return SPARK_STATUS_OK;
}

void SparkTapStreamReaderDestroy(SparkTapStreamReader *reader)
{
	if ( reader == 0 )
		return;
	if ( reader->fd != TAP_READER_CLOSED )
		(void)close(reader->fd);
	if ( reader->listen_fd != TAP_READER_CLOSED )
		(void)close(reader->listen_fd);
	free(reader->rx_buffer);
	free(reader->payload);
	free(reader);
}

uint32_t SparkTapStreamReaderPort(const SparkTapStreamReader *reader)
{
	return reader != 0 ? reader->port : 0u;
}

uint32_t SparkTapStreamReaderPollFds(const SparkTapStreamReader *reader, int *fds, uint32_t *events, uint64_t *deadline_ns)
{
	uint32_t count = 0u;
	if ( reader == 0 || fds == 0 || events == 0 || reader->file != 0u )
		return 0u;
	fds[count] = reader->listen_fd;
	events[count++] = 1u;
	if ( reader->fd != TAP_READER_CLOSED )
	{
		fds[count] = reader->fd;
		events[count++] = 1u;
	}
	if ( deadline_ns != 0 )
		*deadline_ns = reader->deadline_ns;
	return count;
}

SparkStatus SparkTapStreamReaderAdvance(SparkTapStreamReader *reader, SparkTapStreamReaderFrame *frame_out)
{
	if ( reader == 0 || frame_out == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( reader->file == 0u && (reader->fd == TAP_READER_CLOSED || reader->rx_head == reader->rx_tail) )
		TapReaderAccept(reader);
	for ( ;; )
	{
		int ready;
		if ( reader->fd == TAP_READER_CLOSED || reader->phase == TAP_READER_PHASE_IDLE )
			return reader->file != 0u ? SPARK_STATUS_NOT_FOUND : SPARK_STATUS_PENDING;
		if ( reader->deadline_ns != 0u && SparkMonotonicNs() > reader->deadline_ns )
		{
			TapReaderRefuse(reader, "the stream stalled inside a hello or frame");
			return SPARK_STATUS_PENDING;
		}
		if ( reader->phase == TAP_READER_PHASE_HELLO )
		{
			ready = TapReaderReadSome(reader, (uint8_t *)&reader->hello, sizeof(reader->hello));
			if ( ready > 0 && TapReaderHelloDone(reader) == 0u && reader->file != 0u )
				SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		}
		else if ( reader->phase == TAP_READER_PHASE_HEADER )
		{
			ready = TapReaderReadSome(reader, (uint8_t *)&reader->frame, sizeof(reader->frame));
			if ( ready < 0 && reader->file != 0u && reader->got != 0u )
				fprintf(stderr, "sparkpipe_tap_reader: the recording ends in a truncated frame header; the rest is ignored\n");
			if ( ready > 0 && TapReaderHeaderDone(reader) == 0u && reader->file != 0u )
				SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		}
		else
		{
			ready = TapReaderReadSome(reader, reader->payload, reader->frame.payload_bytes);
			if ( ready < 0 && reader->file != 0u )
				fprintf(stderr, "sparkpipe_tap_reader: frame %llu is cut short; the rest of the recording is ignored\n",
					(unsigned long long)reader->frame.serial);
			if ( ready > 0 )
			{
				TapReaderFrameOut(reader, frame_out);
				return SPARK_STATUS_OK;
			}
		}
		if ( ready == 0 )
			return SPARK_STATUS_PENDING;
		if ( ready < 0 )
		{
			if ( reader->file != 0u )
			{
				reader->phase = TAP_READER_PHASE_IDLE;
				return SPARK_STATUS_NOT_FOUND;
			}
			TapReaderClose(reader);
			return SPARK_STATUS_PENDING;
		}
	}
}

void SparkTapStreamReaderGetStats(const SparkTapStreamReader *reader, SparkTapStreamReaderStats *stats_out)
{
	if ( stats_out == 0 )
		return;
	memset(stats_out, 0, sizeof(*stats_out));
	if ( reader != 0 )
		*stats_out = reader->stats;
}
