#!/usr/bin/env python3
"""Generate the Qwen3.8-27B TP4 deployment tree: per-rank stage configs and
the shared model_resident.json, rebuilt for the converged engine (weightd
owns the mesh, hidden_transport default, hex pack ranks, ROOT_NAME single
source). Mirrors tools/glm5_next_gen_deployment.py / glm52_gen_deployment.py.

The 27B serving adapter validates its stage-config member set EXACTLY:
schema_version, model_revision, stage_pack_path, max_sequence_positions,
tp_degree, tp_rank, tp_collective - and tp_collective itself must carry the
full hidden_transport member set (SparkTpCollectiveValidateMembersExact).
The TP collective rides the qwen-27b port block (base 22528): peer ports
22528+rank, session and hc grids above them inside the same block.

Usage:
  python3 tools/qwen38_27b_gen_deployment.py --output deployment/qwen38_27b_tp4
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

HOSTS = [h for h in os.environ.get(
    "QWEN38_27B_TP_HOSTS",
    ",".join("spark%x" % r for r in range(4))).split(",") if h]
for _host in HOSTS:
    if not re.fullmatch(r"[a-z0-9]+", _host):
        raise SystemExit("invalid host name %r: must be [a-z0-9]+" % _host)
TP = len(HOSTS)
if TP < 2 or TP > 16:
    raise SystemExit("qwen38_27b deploys 2..16 ranks; got %d" % TP)

ROOT_NAME = os.environ.get("QWEN38_27B_ROOT_NAME", "qwen38-27b.bf16.tp4")
RUNTIME_ROOT = os.environ.get("QWEN38_27B_RUNTIME_ROOT",
                              "/home/{host}/sparkdata/" + ROOT_NAME)
PACK_TEMPLATE = os.environ.get(
    "QWEN38_27B_PACK_TEMPLATE",
    "packs/" + ROOT_NAME + ".rank%x.qwen38_27bsp")

COLLECTIVE_BASE = int(os.environ.get("QWEN38_27B_COLLECTIVE_BASE", "22528"))
COLLECTIVE_SESSION_BASE = int(os.environ.get(
    "QWEN38_27B_SESSION_BASE", str(COLLECTIVE_BASE + 16)))
COLLECTIVE_SESSION_HC_BASE = int(os.environ.get(
    "QWEN38_27B_SESSION_HC_BASE", str(COLLECTIVE_BASE + 32)))
COLLECTIVE_ID = int(os.environ.get("QWEN38_27B_COLLECTIVE_ID", "22528"))
CONTROL_BASE = int(os.environ.get("QWEN38_27B_CONTROL_BASE", "19540"))
TRANSPORT_BASE = int(os.environ.get("QWEN38_27B_TRANSPORT_BASE", "60720"))

MODEL_REVISION = "bf16-h5120-l64-gdn48-full16-v248320-mtp1-v1"
NODE_TARGET = "cuda.sm121.qwen38_27b.resident_decode_stage.bf16"
MAX_SEQUENCE_POSITIONS = 4096

TP_COLLECTIVE = {
    "backend": "hidden_transport",
    "backend_module_path": "lib/hidden_transport.so",
    "collective_identifier": COLLECTIVE_ID,
    "listen_port": COLLECTIVE_BASE,
    "connect_timeout_milli": 30000,
    "operation_timeout_milli": 30000,
    "peer_hosts": list(HOSTS),
    "peer_ports": [COLLECTIVE_BASE + r for r in range(TP)],
    "algorithms": ["tree"],
    "direct_all_to_all_max_payload_bytes": 0,
    "split_ring_min_payload_bytes": 0,
    "rail_peer_hosts": [list(HOSTS), list(HOSTS)],
    "step_rail_indices": [0] + [1] * (TP - 1),
    "session_ports": [
        [COLLECTIVE_SESSION_BASE + a * TP + b if a != b else 0
         for b in range(TP)] for a in range(TP)],
    "session_ports_hc": [
        [COLLECTIVE_SESSION_HC_BASE + a * TP + b if a != b else 0
         for b in range(TP)] for a in range(TP)],
}

TP_COLLECTIVE_MEMBERS = {
    "backend", "backend_module_path", "collective_identifier", "listen_port",
    "connect_timeout_milli", "operation_timeout_milli", "peer_hosts",
    "peer_ports", "algorithms", "direct_all_to_all_max_payload_bytes",
    "split_ring_min_payload_bytes", "rail_peer_hosts", "step_rail_indices",
    "session_ports", "session_ports_hc",
}
STAGE_MEMBERS = {
    "schema_version", "model_revision", "stage_pack_path",
    "max_sequence_positions", "tp_degree", "tp_rank", "tp_collective",
}

if set(TP_COLLECTIVE) != TP_COLLECTIVE_MEMBERS:
    raise SystemExit("tp_collective member set drifted from the adapter's "
                     "exact hidden_transport member list")


def stage_config(rank: int) -> dict:
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "stage_pack_path": PACK_TEMPLATE % rank,
        "max_sequence_positions": MAX_SEQUENCE_POSITIONS,
        "tp_degree": TP,
        "tp_rank": rank,
        "tp_collective": dict(TP_COLLECTIVE,
                              listen_port=COLLECTIVE_BASE + rank),
    }


def resident_deployment() -> dict:
    page_capacity = TP * ((MAX_SEQUENCE_POSITIONS + 63) // 64)
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": RUNTIME_ROOT.format(host=host),
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/stage.json",
            "kv_backing_directory": "/home/%s/kvcache/" % host + ROOT_NAME,
            "kv_backing_maximum_bytes": 0,
            "control_endpoint": {
                "kind": "tcp",
                "host": host,
                "port": CONTROL_BASE + rank,
            },
        })
    return {
        "schema_version": 2,
        "coordinator_rank_index": 0,
        "adapter": {"shared_object_path": "lib/model_serving_adapter.so"},
        "driver": {
            "shared_object_path": "stages/stage_000/model_driver.so",
            "program_name": "resident_decode",
        },
        "transport": {
            "shared_object_path": "lib/hidden_transport.so",
            "mode": "host-rdma",
            "control_port_base": TRANSPORT_BASE,
        },
        "weightd": {
            "socket_path": "/tmp/spark_weightd.sock",
        },
        "runtime_limits": {
            "max_inflight_submissions": 4,
            "max_active_sequences": 16,
            "max_input_rows": 16,
            "resident_sequence_capacity": 16,
            "kv_logical_page_capacity": page_capacity,
            "kv_physical_page_capacity": page_capacity,
        },
        "nodes": nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    (root / "config").mkdir(parents=True, exist_ok=True)
    for rank in range(TP):
        if set(stage_config(rank)) != STAGE_MEMBERS:
            raise SystemExit("stage config member set drifted from the "
                             "27B adapter's exact member list")
        (root / "config" / ("stage_%02d.json" % rank)).write_text(
            json.dumps(stage_config(rank), indent=1) + "\n")
    (root / "model_resident.json").write_text(
        json.dumps(resident_deployment(), indent=1) + "\n")
    print("%s: %d stage configs + model_resident.json "
          "(tp_collective base %d)" % (root, TP, COLLECTIVE_BASE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
