#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

HOSTS = [f"spark{hex(r)[2:]}" for r in range(16)]
TP = 4
PP = 4
ARM = "glm53flash.fp8.tp4pp4"
CONTROL_BASE = 19560
TRANSPORT_BASE = 60710
COLLECTIVE_GROUP_BASE = 25000
COLLECTIVE_GROUP_STRIDE = 100
SESSION_BASE = 26000
SESSION_HC_BASE = 30000
SESSION_GROUP_STRIDE = 1000
COLLECTIVE_ID_BASE = 9911223344556679
MODEL_REVISION = "84c6a6aa9497188e15a635ba793b0f95a79b1033"
NODE_TARGET = "cuda.sm121.glm5_next.resident_decode_stage.bf16.expert_fp8"
PACK_TEMPLATE = "packs/glm5_next.tp4pp4.rank%d"


def group_hosts(rank):
    first = rank // TP * TP
    return [HOSTS[first + t] for t in range(TP)]


def session_table(group, base):
    return [[base + group * SESSION_GROUP_STRIDE + a * TP + b if a != b else 0
             for b in range(TP)] for a in range(TP)]


def stage_config(rank):
    group = rank // TP
    tp_rank = rank % TP
    hosts = group_hosts(rank)
    collective_base = COLLECTIVE_GROUP_BASE + group * COLLECTIVE_GROUP_STRIDE
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": "fp8",
        "stage_pack_path": PACK_TEMPLATE % rank,
        "max_sequence_positions": 32768,
        "execution_row_capacity": 1024,
        "decode_split_context_threshold": 2048,
        "tp_degree": TP,
        "tp_rank": tp_rank,
        "tp_collective": {
            "backend": "hidden_transport",
            "backend_module_path": "lib/hidden_transport.so",
            "algorithms": ["tree", "direct_all_to_all"],
            "collective_identifier": COLLECTIVE_ID_BASE + group,
            "listen_port": collective_base + tp_rank,
            "connect_timeout_milli": 30000,
            "operation_timeout_milli": 30000,
            "peer_hosts": hosts,
            "peer_ports": [collective_base + t for t in range(TP)],
            "split_ring_min_payload_bytes": 0,
            "direct_all_to_all_max_payload_bytes": 81920,
            "rail_peer_hosts": [list(hosts), list(hosts)],
            "step_rail_indices": [0] + [1] * (TP - 1),
            "session_ports": session_table(group, SESSION_BASE),
            "session_ports_hc": session_table(group, SESSION_HC_BASE),
        },
    }


def resident_deployment():
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": f"/home/{host}/sparkdata/{ARM}",
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/stage.json",
            "kv_backing_directory": f"/home/{host}/kvcache/{ARM}",
            "kv_backing_maximum_bytes": 8589934592,
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
            "max_input_rows": 1024,
            "resident_sequence_capacity": 16,
            "kv_logical_page_capacity": 0,
            "kv_physical_page_capacity": 0,
        },
        "nodes": nodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    (root / "config").mkdir(parents=True, exist_ok=True)
    for rank in range(len(HOSTS)):
        (root / "config" / ("stage_%02d.json" % rank)).write_text(
            json.dumps(stage_config(rank), indent=1) + "\n")
    (root / "model_resident.json").write_text(
        json.dumps(resident_deployment(), indent=1) + "\n")
    print(f"{root}: 16 stage configs + model_resident.json "
          f"(hosts {HOSTS[0]}..{HOSTS[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
