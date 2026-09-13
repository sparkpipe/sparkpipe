// F1 device-only init gate: the runner's sharded-init contract.
//
// The host TCP collective caps at four ranks, so a TP16 deployment supplies
// ONLY the NCCL device tier - init must ACCEPT that shape. What stays fatal:
// a sharded configuration with no collective at all, and a device tier that
// cannot carry every exchange the runner makes (the hidden-transport tier
// keeps a pre-registered frame and has no u64 combine, so it can never run
// the head argmax; admitting it device-only would silently no-op the
// embedding reduce).
//
// The gate pins the INIT GATE ORDER, not ring connectivity: the NCCL leg
// deliberately points at unreachable loopback peers with millisecond
// timeouts, so "create attempted and failed to connect" is the expected
// pass shape. The old code rejected every tp_collective-less init with
// INVALID_ARGUMENT before any create attempt - that is what changed.
//
// Needs one GPU + one rank pack: bash tools/k3_device_only_init_gate.sh
// <rank.pack> [NVCC]
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "sparkpipe/spark_tp_collective.h"
#include "sparkpipe/spark_tp_device_collective.h"

static int failures = 0;

static void Check(int ok, const char *what)
{
	if ( ok )
		printf("ok %s\n", what);
	else
	{
		printf("FAIL %s\n", what);
		failures++;
	}
}

static void FillBaseConfig(SparkK3StageRunnerConfiguration *config,
	const char *pack, uint32_t degree, uint32_t rank)
{
	memset(config, 0, sizeof(*config));
	config->abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	config->descriptor_bytes = (uint32_t)sizeof(*config);
	config->stage_index = 0u;
	config->stage_count = 1u; /* PP1 derive mode: the TP16 placement */
	config->tp_degree = degree;
	config->tp_rank = rank;
	config->max_active_sequence_count = 1u;
	config->max_input_row_count = 1u;
	config->resident_sequence_capacity = 1u;
	config->kv_pages_per_sequence = 2u;
	config->kv_page_bytes = 0u; /* the runner defaults the K3 latent geometry */
	config->rank_pack_path = pack;
}

/* Static-validation-clean NCCL config whose peers cannot exist on a lone
 * node: create must get PAST config validation into connection attempts. */
static void FillNcclConfig(SparkTpDeviceCollectiveConfig *device_config,
	uint32_t degree, uint32_t rank)
{
	static const char *hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
	uint32_t i;
	memset(device_config, 0, sizeof(*device_config));
	device_config->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	device_config->backend_kind =
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL;
	device_config->tp_degree = degree;
	device_config->tp_rank = rank;
	device_config->operation_kind =
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	device_config->credit_count = 8u;
	device_config->local_hidden_dimension = 7168u; /* K3 hidden */
	device_config->max_active_sequence_count = 1u;
	device_config->connect_timeout_milli = 250u;
	device_config->operation_timeout_milli = 1000u;
	device_config->control_port_base = (uint16_t)(46200u + rank);
	device_config->collective_identifier = 77u;
	device_config->backend_module_path = "libnccl.so.2";
	device_config->local_host = "127.0.0.1";
	for ( i = 0u; i < degree; ++i )
		hosts[i] = "127.0.0.1";
	memcpy(device_config->rank_hosts, hosts, sizeof(hosts));
}

static void FillHiddenConfig(SparkTpDeviceCollectiveConfig *device_config,
	uint32_t degree, uint32_t rank)
{
	memset(device_config, 0, sizeof(*device_config));
	device_config->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	device_config->backend_kind =
		SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	device_config->tp_degree = degree;
	device_config->tp_rank = rank;
	device_config->operation_kind =
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	device_config->credit_count = 8u;
	device_config->local_hidden_dimension = 7168u;
	device_config->max_active_sequence_count = 1u;
	device_config->connect_timeout_milli = 250u;
	device_config->operation_timeout_milli = 1000u;
	device_config->control_port_base = (uint16_t)(47200u + rank);
	device_config->collective_identifier = 78u;
	device_config->local_host = "127.0.0.1";
}

int main(int argc, char **argv)
{
	SparkK3StageRunner runner;
	SparkK3StageRunnerConfiguration config;
	SparkStatus status;
	if ( argc < 2 )
	{
		printf("usage: k3_device_only_init <rank.pack>\n");
		return 2;
	}

	/* 1. NEGATIVE CONTROL (unchanged contract): sharded decode with NO
	 *    collective of either kind is exactly INVALID_ARGUMENT - the fix
	 *    narrowed the rejection, it did not remove it. */
	FillBaseConfig(&config, argv[1], 16u, 0u);
	status = SparkK3StageRunnerInitialize(&runner, &config);
	Check(status == SPARK_STATUS_INVALID_ARGUMENT,
		"no tier at degree 16 -> INVALID_ARGUMENT");

	/* 2. HIDDEN-TRANSPORT-ONLY still rejected: that backend cannot carry
	 *    the head exchange or narrow its frame, so admitting it alone
	 *    would silently skip the embedding reduce. */
	{
		SparkTpDeviceCollectiveConfig hidden_config;
		FillHiddenConfig(&hidden_config, 16u, 0u);
		FillBaseConfig(&config, argv[1], 16u, 0u);
		config.device_collective = &hidden_config;
		status = SparkK3StageRunnerInitialize(&runner, &config);
		Check(status == SPARK_STATUS_INVALID_ARGUMENT,
			"hidden-transport-only at degree 16 -> INVALID_ARGUMENT");
	}

	/* 3. THE FIX: NCCL-only at degree 16 passes the init gate. On a lone
	 *    node the create then fails to reach its peers (or the nccl lib is
	 *    absent) - every outcome EXCEPT the old pre-create rejection proves
	 *    the gate now admits device-only topologies. */
	{
		SparkTpDeviceCollectiveConfig nccl_config;
		FillNcclConfig(&nccl_config, 16u, 0u);
		FillBaseConfig(&config, argv[1], 16u, 0u);
		config.device_collective = &nccl_config;
		status = SparkK3StageRunnerInitialize(&runner, &config);
		printf("nccl-only init status %d\n", (int)status);
		Check(status != SPARK_STATUS_INVALID_ARGUMENT,
			"nccl-only at degree 16 passes the init gate");
		if ( status == SPARK_STATUS_OK )
			SparkK3StageRunnerDestroy(&runner);
	}

	/* 4. REGRESSION GUARD: a supplied host tier is still honoured at low
	 *    degree - init proceeds past the gate into the host-collective
	 *    create (which fails fast against absent loopback peers). */
	{
		SparkTpCollectiveConfig host_config;
		SparkTpCollectivePeer peers[SPARK_TP_COLLECTIVE_MAX_STEPS];
		memset(&host_config, 0, sizeof(host_config));
		memset(peers, 0, sizeof(peers));
		host_config.abi_version = SPARK_TP_COLLECTIVE_ABI_VERSION;
		host_config.tp_degree = 4u;
		host_config.tp_rank = 0u;
		host_config.listen_port = 61620u;
		host_config.connect_timeout_milli = 250u;
		host_config.operation_timeout_milli = 500u;
		host_config.collective_identifier = 79u;
		for ( uint32_t step = 0u; step < 2u; ++step )
		{
			snprintf(peers[step].host_name,
				sizeof(peers[step].host_name), "127.0.0.1");
			peers[step].port = (uint16_t)(61620u + (0u ^ (1u << step)));
		}
		memcpy(host_config.peers, peers, sizeof(peers));
		FillBaseConfig(&config, argv[1], 4u, 0u);
		config.tp_collective = &host_config;
		status = SparkK3StageRunnerInitialize(&runner, &config);
		printf("host-tier init status %d\n", (int)status);
		Check(status != SPARK_STATUS_INVALID_ARGUMENT,
			"host tier at degree 4 still attempts its create");
		if ( status == SPARK_STATUS_OK )
			SparkK3StageRunnerDestroy(&runner);
	}

	if ( failures == 0 )
	{
		printf("k3 device-only init gate PASS\n");
		return 0;
	}
	printf("k3 device-only init gate FAIL (%d)\n", failures);
	return 1;
}
