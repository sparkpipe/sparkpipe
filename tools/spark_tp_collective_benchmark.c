#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_tp_collective.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SPARK_TP_BENCHMARK_DEGREE 16u
#define SPARK_TP_BENCHMARK_MAX_HOST_BYTES 64u
#define SPARK_TP_BENCHMARK_BOUNDARY_ELEMENTS (4u * 4096u)
#define SPARK_TP_BENCHMARK_DEFAULT_OPERATIONS 100u
#define SPARK_TP_BENCHMARK_DEFAULT_WARMUP 5u
#define SPARK_TP_BENCHMARK_DEFAULT_PORT 29160u
#define SPARK_TP_BENCHMARK_CONNECT_TIMEOUT_MILLI 10000u
#define SPARK_TP_BENCHMARK_OPERATION_TIMEOUT_MILLI 30000u

typedef struct SparkTpBenchmarkOptions
{
	uint32_t rank;
	uint32_t batch;
	uint32_t operations;
	uint32_t warmup;
	uint16_t port;
	uint64_t collective_identifier;
	const char *host_spec;
} SparkTpBenchmarkOptions;

static void SparkTpBenchmarkUsage(const char *program)
{
	fprintf(stderr,
		"usage: %s --rank N --hosts h0,...,h15 [--batch B] "
		"[--operations N] [--warmup N] [--port P] [--identifier N]\n",
		program);
}

static int SparkTpBenchmarkParseUnsigned(
	const char *text,
	uint64_t maximum,
	uint64_t *value_out)
{
	char *end;
	unsigned long long parsed;

	if (text == 0 || text[0] == '\0' || value_out == 0)
		return(-1);
	errno = 0;
	parsed = strtoull(text,&end,0);
	if (errno != 0 || end == text || *end != '\0' ||
		(uint64_t)parsed > maximum)
		return(-1);
	*value_out = (uint64_t)parsed;
	return(0);
}

static int SparkTpBenchmarkParseArguments(
	int argc,
	char **argv,
	SparkTpBenchmarkOptions *options)
{
	int index;
	uint64_t value;

	if (options == 0)
		return(-1);
	memset(options,0,sizeof(*options));
	options->batch = 1u;
	options->operations = SPARK_TP_BENCHMARK_DEFAULT_OPERATIONS;
	options->warmup = SPARK_TP_BENCHMARK_DEFAULT_WARMUP;
	options->port = SPARK_TP_BENCHMARK_DEFAULT_PORT;
	for (index = 1; index < argc; index += 2)
	{
		if (index + 1 >= argc)
			return(-1);
		if (strcmp(argv[index],"--hosts") == 0)
			options->host_spec = argv[index + 1];
		else if (strcmp(argv[index],"--rank") == 0)
		{
			if (SparkTpBenchmarkParseUnsigned(argv[index + 1],
				SPARK_TP_BENCHMARK_DEGREE - 1u,&value) != 0)
				return(-1);
			options->rank = (uint32_t)value;
		}
		else if (strcmp(argv[index],"--batch") == 0)
		{
			if (SparkTpBenchmarkParseUnsigned(argv[index + 1],
			UINT32_MAX,&value) != 0 || value == 0u)
				return(-1);
			options->batch = (uint32_t)value;
		}
		else if (strcmp(argv[index],"--operations") == 0)
		{
			if (SparkTpBenchmarkParseUnsigned(argv[index + 1],
			UINT32_MAX,&value) != 0 || value == 0u)
				return(-1);
			options->operations = (uint32_t)value;
		}
		else if (strcmp(argv[index],"--warmup") == 0)
		{
			if (SparkTpBenchmarkParseUnsigned(argv[index + 1],
			UINT32_MAX,&value) != 0)
				return(-1);
			options->warmup = (uint32_t)value;
		}
		else if (strcmp(argv[index],"--port") == 0)
		{
			if (SparkTpBenchmarkParseUnsigned(argv[index + 1],
			UINT16_MAX,&value) != 0 || value == 0u)
				return(-1);
			options->port = (uint16_t)value;
		}
		else if (strcmp(argv[index],"--identifier") == 0)
		{
			if (SparkTpBenchmarkParseUnsigned(argv[index + 1],
			UINT64_MAX,&value) != 0 || value == 0u)
				return(-1);
			options->collective_identifier = value;
		}
		else
			return(-1);
	}
	if (options->host_spec == 0 || options->host_spec[0] == '\0')
		return(-1);
	return(0);
}

static int SparkTpBenchmarkParseHosts(
	const char *host_spec,
	SparkTpCollectivePeer *peers,
	uint32_t peer_count,
	uint16_t port)
{
	char host_copy[SPARK_TP_BENCHMARK_DEGREE *
		SPARK_TP_BENCHMARK_MAX_HOST_BYTES];
	char *cursor;
	char *host;
	uint32_t index;
	uint32_t host_bytes;

	if (host_spec == 0 || peers == 0 || peer_count !=
		SPARK_TP_BENCHMARK_DEGREE || strlen(host_spec) >= sizeof(host_copy))
		return(-1);
	memset(peers,0,sizeof(SparkTpCollectivePeer) * peer_count);
	memset(host_copy,0,sizeof(host_copy));
	memcpy(host_copy,host_spec,strlen(host_spec));
	cursor = host_copy;
	for (index = 0u; index < peer_count; index++)
	{
		host = cursor;
		cursor = strchr(host,',');
		if (cursor != 0)
		{
			*cursor = '\0';
			cursor++;
		}
		host_bytes = (uint32_t)strlen(host);
		if (host_bytes == 0u || host_bytes >=
			SPARK_TP_COLLECTIVE_HOST_NAME_BYTES)
			return(-1);
		memcpy(peers[index].host_name,host,host_bytes + 1u);
		peers[index].port = port;
		if (cursor == 0)
			return(index + 1u == peer_count ? 0 : -1);
	}
	return(cursor == 0 ? 0 : -1);
}

static uint64_t SparkTpBenchmarkNowNanoseconds(void)
{
	struct timespec current_time;

	if (clock_gettime(CLOCK_MONOTONIC,&current_time) != 0)
		return(UINT64_MAX);
	return((uint64_t)current_time.tv_sec * UINT64_C(1000000000) +
		(uint64_t)current_time.tv_nsec);
}

static uint16_t SparkTpBenchmarkF32ToBf16(float value)
{
	uint32_t bits;

	memcpy(&bits,&value,sizeof(bits));
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return((uint16_t)(bits >> 16u));
}

static int SparkTpBenchmarkCheck(
	const uint16_t *values,
	uint64_t element_count,
	uint16_t expected)
{
	uint64_t element_index;

	for (element_index = 0u; element_index < element_count; element_index++)
		if (values[element_index] != expected)
			return(-1);
	return(0);
}

static void SparkTpBenchmarkFill(
	uint16_t *values,
	uint64_t element_count,
	uint16_t value)
{
	uint64_t element_index;

	for (element_index = 0u; element_index < element_count; element_index++)
		values[element_index] = value;
}

static SparkStatus SparkTpBenchmarkRun(
	SparkTpCollective *collective,
	uint16_t *values,
	uint16_t *scratch,
	uint64_t element_count,
	uint32_t rank,
	uint32_t operation_count,
	uint16_t expected,
	uint64_t *elapsed_nanoseconds)
{
	uint32_t operation_index;
	uint64_t start_nanoseconds;
	SparkStatus status;

	if (collective == 0 || values == 0 || scratch == 0 ||
		elapsed_nanoseconds == 0)
		return(SPARK_STATUS_INVALID_ARGUMENT);
	start_nanoseconds = SparkTpBenchmarkNowNanoseconds();
	if (start_nanoseconds == UINT64_MAX)
		return(SPARK_STATUS_INTERNAL_ERROR);
	for (operation_index = 0u; operation_index < operation_count;
		operation_index++)
	{
		SparkTpBenchmarkFill(values,element_count,
			SparkTpBenchmarkF32ToBf16((float)(rank + 1u)));
		status = SparkTpCollectiveAllReduceSumBf16(
			collective,values,element_count,scratch);
		if (status != SPARK_STATUS_OK)
			return(status);
		if (SparkTpBenchmarkCheck(values,element_count,expected) != 0)
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	*elapsed_nanoseconds = SparkTpBenchmarkNowNanoseconds() -
		start_nanoseconds;
	return(SPARK_STATUS_OK);
}

int main(int argc,char **argv)
{
	SparkTpBenchmarkOptions options;
	SparkTpCollectivePeer hosts[SPARK_TP_BENCHMARK_DEGREE];
	SparkTpCollectiveConfig config;
	SparkTpCollective collective;
	uint16_t *values;
	uint16_t *scratch;
	uint64_t element_count;
	uint64_t allocation_bytes;
	uint64_t elapsed_nanoseconds;
	uint64_t elapsed_warmup_nanoseconds;
	uint64_t identifier;
	uint16_t expected;
	SparkStatus status;
	uint32_t step_index;
	uint32_t partner_rank;
	if (SparkTpBenchmarkParseArguments(argc,argv,&options) != 0)
	{
		SparkTpBenchmarkUsage(argv[0]);
		return(2);
	}
	if ((uint32_t)options.port + SPARK_TP_BENCHMARK_DEGREE - 1u >
		UINT16_MAX)
		return(2);
	if (SparkTpBenchmarkParseHosts(options.host_spec,hosts,
		SPARK_TP_BENCHMARK_DEGREE,options.port) != 0)
	{
		fprintf(stderr,"invalid --hosts list; expected 16 comma-separated hosts\n");
		return(2);
	}
	element_count = (uint64_t)options.batch *
		SPARK_TP_BENCHMARK_BOUNDARY_ELEMENTS;
	if (element_count > UINT64_MAX / sizeof(uint16_t))
		return(2);
	allocation_bytes = element_count * sizeof(uint16_t);
	values = (uint16_t *)malloc((size_t)allocation_bytes);
	scratch = (uint16_t *)malloc((size_t)allocation_bytes);
	if (values == 0 || scratch == 0)
	{
		free(values);
		free(scratch);
		return(1);
	}
	identifier = options.collective_identifier != 0u ?
		options.collective_identifier :
		(UINT64_C(0x5450313600000000) | (uint64_t)options.port);
	memset(&config,0,sizeof(config));
	config.abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
	config.tp_degree = SPARK_TP_BENCHMARK_DEGREE;
	config.tp_rank = options.rank;
	config.listen_port = (uint16_t)(options.port + options.rank);
	config.connect_timeout_milli = SPARK_TP_BENCHMARK_CONNECT_TIMEOUT_MILLI;
	config.operation_timeout_milli = SPARK_TP_BENCHMARK_OPERATION_TIMEOUT_MILLI;
	config.collective_identifier = identifier;
	for (step_index = 0u; step_index < SPARK_TP_COLLECTIVE_MAX_STEPS;
		step_index++)
	{
		partner_rank = options.rank ^ (1u << step_index);
		config.peers[step_index] = hosts[partner_rank];
		config.peers[step_index].port =
			(uint16_t)(options.port + partner_rank);
	}
	memset(&collective,0,sizeof(collective));
	expected = SparkTpBenchmarkF32ToBf16(
		(float)(SPARK_TP_BENCHMARK_DEGREE *
		(SPARK_TP_BENCHMARK_DEGREE + 1u) / 2u));
	status = SparkTpCollectiveCreate(&config,&collective);
	if (status == SPARK_STATUS_OK && options.warmup != 0u)
	{
		status = SparkTpBenchmarkRun(&collective,values,scratch,element_count,
			options.rank,options.warmup,expected,
			&elapsed_warmup_nanoseconds);
	}
	if (status == SPARK_STATUS_OK)
		status = SparkTpBenchmarkRun(&collective,values,scratch,element_count,
			options.rank,options.operations,expected,&elapsed_nanoseconds);
	SparkTpCollectiveDestroy(&collective);
	free(values);
	free(scratch);
	if (status != SPARK_STATUS_OK)
	{
		fprintf(stderr,"tp16_collective status=%s\n",
			SparkStatusToString(status));
		return(1);
	}
	printf("tp16_collective rank=%u batch=%u operations=%u elements=%" PRIu64
		" payload_bytes=%" PRIu64 " elapsed_ms=%.3f ms_per_operation=%.6f "
		"token_equivalent_per_second=%.3f status=0\n",
		options.rank,options.batch,options.operations,element_count,
		allocation_bytes,(double)elapsed_nanoseconds / 1000000.0,
		(double)elapsed_nanoseconds / 1000000.0 / options.operations,
		(double)options.batch * options.operations * 1000000000.0 /
		(double)elapsed_nanoseconds);
	return(0);
}
