#!/usr/bin/env python3
"""Host syntax and v4/persistent-credit contracts for the CUDA RDMA TU."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RDMA = (ROOT / "ring/transport/rdma.cu").read_text(encoding="utf-8")


def function_body(name: str) -> str:
    start = RDMA.index(name)
    opening = RDMA.index("{", start)
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (RDMA[cursor] == "{") - (RDMA[cursor] == "}")
        cursor += 1
    return RDMA[opening:cursor]


for device_direct in (0, 1):
    subprocess.run(
        [
            "g++", "-x", "c++", "-fsyntax-only", "-std=c++17",
            "-Wall", "-Wextra", "-Werror",
            f"-DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT={device_direct}",
            "-Itests/rdma_syntax_stub", "-Itests/cuda_stub", "-Iinclude",
            "-Isrc", "ring/transport/rdma.cu",
        ],
        cwd=ROOT,
        check=True,
    )

initialize = function_body("SparkHiddenSparkHostRdmaInitialize(")
assert "SPARK_HIDDEN_TRANSPORT_ENDPOINT_FLAG_OPEN_TIMEOUT" in initialize
assert "endpoint->reserved0" in initialize
assert "SparkHiddenTransportRdmaControlDeadlineNs" in initialize

connect = function_body("SparkHiddenSparkHostRdmaConnectControl(")
assert connect.count("SPARK_HIDDEN_SPARK_HOST_RDMA_CONNECT_RETRY_MS") == 3

discover_device = function_body("SparkHiddenSparkHostRdmaDiscoverDevice(")
assert "SPARK_HIDDEN_SPARK_HOST_RDMA_REQUIRED_LINK_RATE_GBPS" in discover_device
assert "matching_count != 1u" in discover_device
resolve_device = function_body("SparkHiddenSparkHostRdmaResolveDevice(")
assert "SparkHiddenSparkHostRdmaDeviceResolved" in resolve_device
assert "SparkHiddenSparkHostRdmaDiscoverDevice" in resolve_device
assert "SparkHiddenSparkHostRdmaDefaultDeviceName" not in RDMA
assert "SparkHiddenSparkHostRdmaRankFromHost" not in RDMA
assert "SparkHiddenSparkHostRdmaParseRoute" not in RDMA
assert "SPARKPIPE_RING_TRANSPORT_RANK" not in RDMA
assert "SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_IB_DEVICE" not in RDMA
assert '"rocep1s0f0"' not in RDMA and '"rocep1s0f1"' not in RDMA

open_verbs = function_body("SparkHiddenSparkHostRdmaOpenVerbsDevice(")
assert "port_attributes.active_mtu" in open_verbs
exchange_qp = function_body("SparkHiddenSparkHostRdmaExchangeQueuePairInfo(")
assert "remote_infos[lane_index].active_mtu" in exchange_qp
ready_qp = function_body("SparkHiddenSparkHostRdmaModifyQueuePairToReady(")
assert "lane->local_info.active_mtu" in ready_qp
assert "lane->remote_info.active_mtu" in ready_qp
assert "attributes.path_mtu = IBV_MTU_4096" not in ready_qp

create_qps = function_body("SparkHiddenSparkHostRdmaCreateQueuePairs(")
assert "SparkHiddenSparkHostRdmaPrepostDoorbellCredits" in create_qps
doorbell_completion = function_body(
    "SparkHiddenSparkHostRdmaApplyDoorbellCompletion("
)
assert "SparkHiddenSparkHostRdmaPostDoorbellCredit" in doorbell_completion
assert "completion_generation_tag" in doorbell_completion
doorbell_immediate = function_body(
    "SparkHiddenSparkHostRdmaBuildDoorbellImmediate("
)
assert "SparkHiddenSparkHostRdmaDoorbellGenerationTag" in doorbell_immediate
assert "doorbell_posted" not in RDMA

hello = function_body("SparkHiddenSparkHostRdmaExchangeCompatibilityHello(")
for identity_field in (
    "transport_module_id", "route_name", "source_host", "sink_host",
    "source_rank", "sink_rank", "local_rank", "peer_rank", "control_port",
    "hidden_dimension", "bytes_per_sequence", "max_active_sequence_count",
    "persistent_credit_count", "lane_count", "doorbell_max_bytes",
    "memory_mode", "route_identifier",
):
    assert f"identity.{identity_field}" in hello

register = function_body("SparkHiddenSparkHostRdmaRegisterPersistentReceive(")
assert "SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_PERSISTENT_ADVERTISE" in register
assert "SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_RECEIVE_READY" not in register
release = function_body("SparkHiddenSparkHostRdmaReleasePersistentReceive(")
assert "cudaEventRecord" in release and "cudaEventQuery" in release
assert "SPARK_HIDDEN_SPARK_HOST_RDMA_CONTROL_PERSISTENT_RETURN" in release

fence = function_body("SparkHiddenSparkHostRdmaFenceSession(")
assert "__atomic_compare_exchange_n" in fence
assert "SPARK_HIDDEN_SPARK_HOST_RDMA_TERMINAL_FENCED" in fence
assert "IBV_QPS_ERR" in fence
assert "SparkHiddenTransportRdmaControlFenceSession" in fence
assert fence.index("IBV_QPS_ERR") < fence.index("__atomic_store_n")
assert fence.index("SparkHiddenTransportRdmaControlFenceSession") < (
    fence.index("__atomic_store_n")
)
cancel_receive = function_body(
    "SparkHiddenSparkHostRdmaCancelPersistentReceive("
)
assert cancel_receive.index("SparkHiddenSparkHostRdmaFenceSession") < (
    cancel_receive.index("receive->returned_generation")
)
assert cancel_receive.index("SparkHiddenSparkHostRdmaFenceSession") < (
    cancel_receive.index("SparkHiddenSparkHostRdmaResetPersistentActivation")
)
activate_receive = function_body(
    "SparkHiddenSparkHostRdmaActivatePersistentReceive("
)
assert activate_receive.index("SparkHiddenSparkHostRdmaTerminalStatus") < (
    activate_receive.index("receive->active = 1u")
)
assert "early_complete" in activate_receive
assert "completion_generation_tag" in activate_receive
pump = function_body("SparkHiddenSparkHostRdmaPumpProgress(")
assert "SparkHiddenSparkHostRdmaTerminalStatus" in pump
assert "SparkHiddenSparkHostRdmaFenceSession" in pump
assert "SparkHiddenSparkHostRdmaReportProgressError" in pump
for public_path in (
    "SparkHiddenSparkHostRdmaPostReceive(",
    "SparkHiddenSparkHostRdmaPostReceiveBatch(",
    "SparkHiddenSparkHostRdmaSend(",
    "SparkHiddenSparkHostRdmaSendBatch(",
    "SparkHiddenSparkHostRdmaPoll(",
    "SparkHiddenSparkHostRdmaPersistentRemoteCreditReady(",
    "SparkHiddenSparkHostRdmaReservePersistentSend(",
):
    assert "SparkHiddenSparkHostRdmaPumpProgress" in function_body(public_path)
for public_path in (
    "SparkHiddenSparkHostRdmaRegisterPersistentReceive(",
    "SparkHiddenSparkHostRdmaActivatePersistentReceive(",
    "SparkHiddenSparkHostRdmaSendPersistent(",
    "SparkHiddenSparkHostRdmaReleasePersistentReceive(",
    "SparkHiddenSparkHostRdmaGetPollDescriptors(",
):
    assert "SparkHiddenSparkHostRdmaTerminalStatus" in function_body(public_path)
assert "MSG_NOSIGNAL" in RDMA

print("hidden_transport_rdma_source: ok")
