#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_json.h"
#include "sparkpipe/spark_serving_adapter_template.h"
#include "sparkpipe/spark_status.h"

#define TP_DEGREE 4u

static const char *const SHAPE_LING =
"{"
"\"tp_collective\":{"
"\"backend\":\"hidden_transport\","
"\"backend_module_path\":\"tp/libtp.so\","
"\"collective_identifier\":77,"
"\"listen_port\":4240,"
"\"connect_timeout_milli\":900,"
"\"operation_timeout_milli\":31000,"
"\"peer_hosts\":[\"h0\",\"h1\",\"h2\",\"h3\"],"
"\"peer_ports\":[4240,4241,4242,4243],"
"\"algorithms\":[\"recursive_doubling\",\"direct_all_to_all\"],"
"\"direct_all_to_all_max_payload_bytes\":4096,"
"\"split_ring_min_payload_bytes\":0,"
"\"rail_peer_hosts\":[[\"r0\",\"r1\",\"r2\",\"r3\"],[\"s0\",\"s1\",\"s2\",\"s3\"]],"
"\"step_rail_indices\":[0,1,0],"
"\"session_ports\":[[0,11,12,13],[21,0,23,24],[31,32,0,34],[41,42,43,0]],"
"\"session_ports_hc\":[[0,51,52,53],[61,0,63,64],[71,72,0,74],[81,82,83,0]]"
"}}";

static const char *const SHAPE_LAGUNA =
"{"
"\"tp_collective\":{"
"\"backend\":\"hidden_transport\","
"\"backend_module_path\":\"tp/libtp.so\","
"\"collective_identifier\":77,"
"\"listen_port\":4240,"
"\"connect_timeout_milli\":900,"
"\"operation_timeout_milli\":31000,"
"\"peer_hosts\":[\"h0\",\"h1\",\"h2\",\"h3\"],"
"\"peer_ports\":[4240,4241,4242,4243],"
"\"algorithms\":[\"direct_all_to_all\",\"recursive_doubling\"],"
"\"direct_all_to_all_max_payload_bytes\":4096,"
"\"split_ring_min_payload_bytes\":0,"
"\"rail_peer_hosts\":[[\"r0\",\"r1\",\"r2\",\"r3\"],[\"s0\",\"s1\",\"s2\",\"s3\"]],"
"\"step_rail_indices\":[0,1,0],"
"\"session_ports\":[[0,11,12,13],[21,0,23,24],[31,32,0,34],[41,42,43,0]],"
"\"session_ports_hc\":[[0,51,52,53],[61,0,63,64],[71,72,0,74],[81,82,83,0]]"
"}}";

static const char *const SHAPE_GLM52 =
"{"
"\"tp_collective\":{"
"\"backend\":\"nccl\","
"\"backend_module_path\":\"tp/libnccl.sh\","
"\"collective_identifier\":0,"
"\"listen_port\":5000,"
"\"connect_timeout_milli\":100,"
"\"operation_timeout_milli\":200,"
"\"peer_hosts\":[\"g0\",\"g1\",\"g2\",\"g3\"],"
"\"peer_ports\":[7000,7001,7002,7003]"
"}}";

static int32_t failures;

static void expect_true(const char *name,uint32_t condition)
{
	if ( condition == 0u )
	{
		fprintf(stderr,"FAIL %s\n",name);
		failures++;
	}
}

static void expect_status(const char *name,SparkStatus status,SparkStatus want)
{
	if ( status != want )
	{
		fprintf(stderr,"FAIL %s status=%d want=%d\n",name,(int)status,(int)want);
		failures++;
	}
}

static SparkStatus load_shape(const char *text,const SparkTpCollectiveConfigPolicy *policy,SparkTpCollectiveAdapterConfig *config)
{
	SparkJsonDocument document;
	SparkStatus status;
	SparkJsonDocumentReset(&document);
	status = SparkJsonParseText(text,strlen(text),&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateLoadTpCollective(&document,
			SparkJsonGetRootToken(&document),"/tmp/spark_tp_cfg_root",
			policy,config);
	SparkJsonDocumentDestroy(&document);
	return(status);
}

static void expect_ling_laguna_identical(
	const SparkTpCollectiveAdapterConfig *ling,
	const SparkTpCollectiveAdapterConfig *laguna)
{
	expect_true("identical_backend_kind",ling->backend_kind == laguna->backend_kind);
	expect_true("identical_collective_id",ling->collective_identifier == laguna->collective_identifier);
	expect_true("identical_listen_port",ling->listen_port == laguna->listen_port);
	expect_true("identical_peer_count",ling->peer_count == laguna->peer_count);
	expect_true("identical_connect_timeout",ling->connect_timeout_milli == laguna->connect_timeout_milli);
	expect_true("identical_operation_timeout",ling->operation_timeout_milli == laguna->operation_timeout_milli);
	expect_true("identical_control_port_base",ling->control_port_base == laguna->control_port_base);
	expect_true("identical_topology",memcmp(&ling->topology,&laguna->topology,sizeof(ling->topology)) == 0);
	expect_true("identical_peer_ports",memcmp(ling->peer_ports,laguna->peer_ports,sizeof(ling->peer_ports)) == 0);
}

int32_t main(void)
{
	SparkTpCollectiveConfigPolicy policy;
	SparkTpCollectiveAdapterConfig ling,laguna,glm52;
	char ling_path_scratch[256];
	char laguna_path_scratch[256];
	char glm52_path_scratch[256];
	char algorithm_negative[1024];
	char threshold_negative[1024];
	char member_negative[1024];
	char zero_identifier_negative[1024];
	SparkStatus status;
	memset(&ling,0,sizeof(ling));
	memset(&laguna,0,sizeof(laguna));
	memset(&glm52,0,sizeof(glm52));
	ling.backend_module_path_buffer = ling_path_scratch;
	ling.backend_module_path_bytes = sizeof(ling_path_scratch);
	laguna.backend_module_path_buffer = laguna_path_scratch;
	laguna.backend_module_path_bytes = sizeof(laguna_path_scratch);
	glm52.backend_module_path_buffer = glm52_path_scratch;
	glm52.backend_module_path_bytes = sizeof(glm52_path_scratch);
	policy.peer_count = TP_DEGREE;
	policy.allow_zero_collective_identifier = 0u;
	policy.require_contiguous_peer_ports = 1u;
	policy.algorithms = SPARK_TP_COLLECTIVE_ALGORITHMS_ADAPTIVE_COMBOS;
	policy.thresholds = SPARK_TP_COLLECTIVE_THRESHOLDS_MASK_CONDITIONAL;
	policy.require_session_ports = 1u;
	status = load_shape(SHAPE_LING,&policy,&ling);
	expect_status("ling_shape_loads",status,SPARK_STATUS_OK);
	if ( status == SPARK_STATUS_OK )
	{
		expect_true("ling_backend_kind",ling.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT);
		expect_true("ling_collective_id",ling.collective_identifier == 77u);
		expect_true("ling_listen_port",ling.listen_port == 4240u);
		expect_true("ling_connect_timeout",ling.connect_timeout_milli == 900u);
		expect_true("ling_operation_timeout",ling.operation_timeout_milli == 31000u);
		expect_true("ling_control_port_base",ling.control_port_base == 4240u);
		expect_true("ling_algorithm_mask",ling.topology.algorithm_mask == (SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING | SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL));
		expect_true("ling_rank_count",ling.topology.rank_count == TP_DEGREE);
		expect_true("ling_rail_count",ling.topology.rail_count == SPARK_TP_DEVICE_COLLECTIVE_MAX_RAIL_COUNT);
		expect_true("ling_thresholds",ling.topology.direct_all_to_all_max_payload_bytes == 4096u && ling.topology.split_ring_min_payload_bytes == 0u);
		expect_true("ling_peer_host",strcmp(ling.topology.rank_hosts[2],"h2") == 0);
		expect_true("ling_rail_host",strcmp(ling.topology.rail_rank_hosts[1][3],"s3") == 0);
		expect_true("ling_step_rails",ling.topology.step_rail_indices[0] == 0u && ling.topology.step_rail_indices[1] == 1u && ling.topology.step_rail_indices[2] == 0u);
		expect_true("ling_session_ports",ling.topology.session_ports[1][0] == 21u && ling.topology.session_ports[3][2] == 43u && ling.topology.session_ports[2][2] == 0u);
		expect_true("ling_session_ports_hc",ling.session_ports_hc[1][0] == 61u && ling.session_ports_hc[3][2] == 83u);
		expect_true("ling_backend_path",strcmp(ling_path_scratch,"/tmp/spark_tp_cfg_root/tp/libtp.so") == 0);
	}
	policy.require_session_ports = 0u;
	status = load_shape(SHAPE_LAGUNA,&policy,&laguna);
	expect_status("laguna_shape_loads",status,SPARK_STATUS_OK);
	if ( status == SPARK_STATUS_OK )
		expect_ling_laguna_identical(&ling,&laguna);
	policy.algorithms = SPARK_TP_COLLECTIVE_ALGORITHMS_TREE_ONLY;
	policy.thresholds = SPARK_TP_COLLECTIVE_THRESHOLDS_ZERO_REQUIRED;
	policy.allow_zero_collective_identifier = 1u;
	status = load_shape(SHAPE_GLM52,&policy,&glm52);
	expect_status("glm52_shape_loads",status,SPARK_STATUS_OK);
	if ( status == SPARK_STATUS_OK )
	{
		expect_true("glm52_backend_kind",glm52.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL);
		expect_true("glm52_nccl_skips_algorithms",glm52.topology.algorithm_mask == 0u);
		expect_true("glm52_thresholds_zero",glm52.topology.direct_all_to_all_max_payload_bytes == 0u && glm52.topology.split_ring_min_payload_bytes == 0u);
	}
	policy.algorithms = SPARK_TP_COLLECTIVE_ALGORITHMS_ADAPTIVE_COMBOS;
	policy.thresholds = SPARK_TP_COLLECTIVE_THRESHOLDS_MASK_CONDITIONAL;
	policy.allow_zero_collective_identifier = 0u;
	snprintf(algorithm_negative,sizeof(algorithm_negative),
		"{\"tp_collective\":{\"backend\":\"hidden_transport\",\"backend_module_path\":\"tp/libtp.so\",\"collective_identifier\":77,\"listen_port\":4240,\"connect_timeout_milli\":900,\"operation_timeout_milli\":31000,\"peer_hosts\":[\"h0\",\"h1\",\"h2\",\"h3\"],\"peer_ports\":[4240,4241,4242,4243],\"algorithms\":[\"recursive_halving\"],\"direct_all_to_all_max_payload_bytes\":0,\"split_ring_min_payload_bytes\":0,\"rail_peer_hosts\":[[\"r0\",\"r1\",\"r2\",\"r3\"],[\"s0\",\"s1\",\"s2\",\"s3\"]],\"step_rail_indices\":[0,1,0],\"session_ports\":[[0,11,12,13],[21,0,23,24],[31,32,0,34],[41,42,43,0]],\"session_ports_hc\":[[0,51,52,53],[61,0,63,64],[71,72,0,74],[81,82,83,0]]}}");
	expect_status("negative_unknown_algorithm_rejected",load_shape(algorithm_negative,&policy,&glm52),SPARK_STATUS_SCHEMA_ERROR);
	policy.algorithms = SPARK_TP_COLLECTIVE_ALGORITHMS_ADAPTIVE_COMBOS;
	policy.thresholds = SPARK_TP_COLLECTIVE_THRESHOLDS_MASK_CONDITIONAL;
	snprintf(threshold_negative,sizeof(threshold_negative),
		"{\"tp_collective\":{\"backend\":\"hidden_transport\",\"backend_module_path\":\"tp/libtp.so\",\"collective_identifier\":77,\"listen_port\":4240,\"connect_timeout_milli\":900,\"operation_timeout_milli\":31000,\"peer_hosts\":[\"h0\",\"h1\",\"h2\",\"h3\"],\"peer_ports\":[4240,4241,4242,4243],\"algorithms\":[\"recursive_doubling\"],\"direct_all_to_all_max_payload_bytes\":4096,\"split_ring_min_payload_bytes\":0,\"rail_peer_hosts\":[[\"r0\",\"r1\",\"r2\",\"r3\"],[\"s0\",\"s1\",\"s2\",\"s3\"]],\"step_rail_indices\":[0,1,0],\"session_ports\":[[0,11,12,13],[21,0,23,24],[31,32,0,34],[41,42,43,0]],\"session_ports_hc\":[[0,51,52,53],[61,0,63,64],[71,72,0,74],[81,82,83,0]]}}");
	expect_status("negative_threshold_without_algorithm_rejected",load_shape(threshold_negative,&policy,&glm52),SPARK_STATUS_SCHEMA_ERROR);
	snprintf(member_negative,sizeof(member_negative),
		"{\"tp_collective\":{\"backend\":\"nccl\",\"backend_module_path\":\"tp/libnccl.sh\",\"collective_identifier\":1,\"listen_port\":5000,\"connect_timeout_milli\":100,\"peer_hosts\":[\"g0\",\"g1\",\"g2\",\"g3\"],\"peer_ports\":[7000,7001,7002,7003]}}");
	expect_status("negative_missing_member_rejected",load_shape(member_negative,&policy,&glm52),SPARK_STATUS_SCHEMA_ERROR);
	snprintf(zero_identifier_negative,sizeof(zero_identifier_negative),
		"{\"tp_collective\":{\"backend\":\"nccl\",\"backend_module_path\":\"tp/libnccl.sh\",\"collective_identifier\":0,\"listen_port\":5000,\"connect_timeout_milli\":100,\"operation_timeout_milli\":200,\"peer_hosts\":[\"g0\",\"g1\",\"g2\",\"g3\"],\"peer_ports\":[7000,7001,7002,7003]}}");
	policy.algorithms = SPARK_TP_COLLECTIVE_ALGORITHMS_TREE_ONLY;
	policy.thresholds = SPARK_TP_COLLECTIVE_THRESHOLDS_ZERO_REQUIRED;
	expect_status("negative_zero_identifier_rejected",load_shape(zero_identifier_negative,&policy,&glm52),SPARK_STATUS_SCHEMA_ERROR);
	if ( failures == 0u )
	{
		printf("test_serving_tp_config: PASS (ling/laguna/glm52 shapes + negative controls)\n");
		return(0);
	}
	printf("test_serving_tp_config: FAIL (%d)\n",(int)failures);
	return(1);
}
