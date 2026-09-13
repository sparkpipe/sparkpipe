/*
 * Golden tp_collective fixtures for the shared adapter parser
 * (runtime/adapter_common.c SparkAdapterLoadTpCollective).
 *
 * Each row loads one fixture under one policy school and asserts the exact
 * accept/reject outcome plus spot fields on acceptance:
 *   D = dsv4 school   (all three algorithms; thresholds nonzero+ordered)
 *   G = glm52 school  (recursive_doubling only; thresholds must be zero)
 * Rank count is 4 for every fixture; port-span validation is ON (both
 * callers ship it enabled post-merge).
 *
 * The fixture set IS the schema contract between the two families' stanza
 * dialects: flipping either adapter onto the shared parser must leave this
 * table green, and any future policy knob lands here first.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_json.h"
#include "adapter_common.h"

#define FIXTURE_DIR "tests/fixtures/tp_collective/"

typedef struct TestCase
{
	const char *name;
	uint32_t glm52_school;
	SparkStatus expected;
} TestCase;

static const SparkAdapterTpCollectivePolicy dsv4_school =
{
	SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS, 3u, 0u
};

static const SparkAdapterTpCollectivePolicy glm52_school =
{
	SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING, 1u, 1u
};

static const TestCase cases[] =
{
	{ "01_d_nccl_valid", 0u, SPARK_STATUS_OK },
	{ "02_g_nccl_valid", 1u, SPARK_STATUS_OK },
	{ "03_d_hidtran_valid", 0u, SPARK_STATUS_OK },
	{ "04_g_hidtran_valid", 1u, SPARK_STATUS_OK },
	{ "05_d_thresholds_zero_invalid", 0u, SPARK_STATUS_SCHEMA_ERROR },
	{ "06_g_thresholds_nonzero_invalid", 1u, SPARK_STATUS_SCHEMA_ERROR },
	{ "07_d_algorithms_single_invalid", 0u, SPARK_STATUS_SCHEMA_ERROR },
	{ "08_g_algorithms_all3_invalid", 1u, SPARK_STATUS_SCHEMA_ERROR },
	{ "09_unknown_backend", 0u, SPARK_STATUS_SCHEMA_ERROR },
	{ "10_identifier_zero", 1u, SPARK_STATUS_SCHEMA_ERROR },
	{ "11_port_span_broken", 0u, SPARK_STATUS_SCHEMA_ERROR },
	{ "12_peer_host_count_wrong", 1u, SPARK_STATUS_SCHEMA_ERROR },
	{ "13_nccl_extra_algorithms_member", 0u, SPARK_STATUS_SCHEMA_ERROR },
	{ "14_hidtran_missing_rail_hosts", 1u, SPARK_STATUS_SCHEMA_ERROR },
	/* Deliberate tightening vs pre-merge glm52: wrapped-port aliasing
	 * (base 65535) is rejected by the overflow-guarded span check. */
	{ "15_port_wrap_alias", 0u, SPARK_STATUS_SCHEMA_ERROR },
};

int main(void)
{
	char runtime_root[4096];
	const uint32_t rank_count = 4u;
	uint32_t index, failures = 0u;
	if ( getcwd(runtime_root, sizeof(runtime_root)) == 0 )
		return(2);
	strcat(runtime_root, "/tests/fixtures");
	for (index = 0u; index < sizeof(cases) / sizeof(cases[0]); index++)
	{
		const TestCase *test = &cases[index];
		const SparkAdapterTpCollectivePolicy *policy =
			test->glm52_school != 0u ? &glm52_school : &dsv4_school;
		char path[512];
		SparkJsonDocument document;
		SparkAdapterTpCollectiveParsed parsed;
		int32_t root;
		SparkStatus status;
		snprintf(path, sizeof(path), "%s%s.json", FIXTURE_DIR, test->name);
		memset(&document, 0, sizeof(document));
		status = SparkJsonLoadFile(path, &document);
		if ( status == SPARK_STATUS_OK )
		{
			root = SparkJsonGetRootToken(&document);
			status = SparkAdapterLoadTpCollective(&document, root,
				runtime_root, rank_count, 1u, policy, &parsed);
		}
		if ( status != test->expected )
		{
			fprintf(stderr, "FAIL %s: status %d expected %d\n",
				test->name, (int)status, (int)test->expected);
			failures++;
		}
		else if ( status == SPARK_STATUS_OK )
		{
			if ( parsed.topology.rank_count != rank_count ||
				parsed.control_port_base != parsed.peer_ports[0] ||
				parsed.peer_ports[rank_count - 1u] !=
					(uint16_t)(parsed.control_port_base + rank_count - 1u) )
			{
				fprintf(stderr, "FAIL %s: parsed field spot-check\n",
					test->name);
				failures++;
			}
			else if ( test->glm52_school == 0u &&
				parsed.backend_kind ==
					SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT &&
				parsed.topology.algorithm_mask !=
					SPARK_TP_DEVICE_COLLECTIVE_KNOWN_ALGORITHMS )
			{
				fprintf(stderr, "FAIL %s: algorithm mask\n", test->name);
				failures++;
			}
			else
				printf("ok %s (port=%u id=%llu base=%u)\n", test->name,
					parsed.listen_port,
					(unsigned long long)parsed.collective_identifier,
					parsed.control_port_base);
		}
		else
			printf("ok %s (rejected as expected)\n", test->name);
		SparkJsonDocumentDestroy(&document);
	}
	printf("%u/%u golden tp_collective cases pass\n",
		(uint32_t)(sizeof(cases) / sizeof(cases[0])) - failures,
		(uint32_t)(sizeof(cases) / sizeof(cases[0])));
	return(failures != 0u ? 1 : 0);
}
