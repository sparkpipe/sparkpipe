#!/usr/bin/env python3
"""Generate the GLM52 TP8 deployment tree: per-rank stage configs and the
shared multi-node model_resident.json.

Hosts: spark8..sparkf are ranks 0..7 (registry order). runtime_root:
/home/{host}/sparkdata/glm52.tp8.fp8. Port blocks: control 19480+rank,
collective 63620+rank, transport 60700.
"""
import json
import os
import sys

HOSTS = ["spark8", "spark9", "sparka", "sparkb",
         "sparkc", "sparkd", "sparke", "sparkf"]
TP = len(HOSTS)
RUNTIME_ROOT = "/home/{host}/sparkdata/glm52.tp8.fp8"
CONTROL_BASE = 19480
COLLECTIVE_BASE = 63620
TRANSPORT_BASE = 60700
COLLECTIVE_ID = 8811223344556678
MODEL_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
NODE_TARGET = "cuda.sm121.glm52.resident_decode_stage.bf16.expert_fp8"

TP_COLLECTIVE = {
    "backend": "hidden_transport",
    "backend_module_path": "lib/hidden_transport.so",
    "collective_identifier": COLLECTIVE_ID,
    "listen_port": COLLECTIVE_BASE,
    "connect_timeout_milli": 180000,
    "operation_timeout_milli": 30000,
    "peer_hosts": list(HOSTS),
    "peer_ports": [COLLECTIVE_BASE + r for r in range(TP)],
    "algorithms": ["recursive_doubling"],
    "direct_all_to_all_max_payload_bytes": 0,
    "split_ring_min_payload_bytes": 0,
    "rail_peer_hosts": [list(HOSTS), list(HOSTS)],
    "step_rail_indices": [0, 0, 1],
}


def stage_config(rank: int) -> dict:
    cfg = {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": "fp8",
        "stage_pack_path": "packs/glm52_tp8_rank%02d.fp8.glm52sp" % rank,
        "max_sequence_positions": 4096,
        "execution_row_capacity": 16,
        "tp_degree": TP,
        "tp_rank": rank,
        "tp_collective": dict(TP_COLLECTIVE),
    }
    return cfg


def resident_deployment() -> dict:
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": RUNTIME_ROOT.format(host=host),
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/glm52_stage.json",
            "kv_backing_directory": "/home/%s/kvcache/glm52.tp8" % host,
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
            "shared_object_path": "lib/model_driver.so",
            "program_name": "resident_decode",
        },
        "transport": {
            "shared_object_path": "lib/hidden_transport.so",
            "mode": "host-rdma",
            "control_port_base": TRANSPORT_BASE,
        },
        "runtime_limits": {
            "max_inflight_submissions": 4,
            "max_active_sequences": 16,
            "max_input_rows": 16,
            "resident_sequence_capacity": 16,
            "kv_logical_page_capacity": 0,
            "kv_physical_page_capacity": 0,
        },
        "nodes": nodes,
    }


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/glm52-deploy"
    for rank, host in enumerate(HOSTS):
        host_dir = os.path.join(out, host, "config")
        os.makedirs(host_dir, exist_ok=True)
        with open(os.path.join(host_dir, "glm52_stage.json"), "w") as f:
            json.dump(stage_config(rank), f, indent=2)
            f.write("\n")
        with open(os.path.join(host_dir, "model_resident.json"), "w") as f:
            json.dump(resident_deployment(), f, indent=2)
            f.write("\n")
    print("generated %d host configs under %s" % (TP, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
