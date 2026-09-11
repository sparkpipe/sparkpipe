#!/usr/bin/env python3
"""Generate the muse TP16 deployment tree: per-rank stage configs and the
shared multi-node model_resident.json.

Hosts default to the sixteen-spark band as ranks 0..15 (registry order,
override with MUSE_TP_HOSTS). Port base 13312 is the FROZEN lane block
(sparkpipe-coord/PORT_LEDGER.md: 13312-14335, below the ephemeral floor);
control, collective and session offsets all live inside it.
"""
import json
import os
import sys

HOSTS = [h for h in os.environ.get(
    "MUSE_TP_HOSTS",
    "spark0,spark1,spark2,spark3,spark4,spark5,spark6,spark7,"
    "spark8,spark9,sparka,sparkb,sparkc,sparkd,sparke,sparkf").split(",") if h]
TP = len(HOSTS)
RUNTIME_ROOT = os.environ.get(
    "MUSE_RUNTIME_ROOT", "/home/{host}/sparkdata/muse.tp16.bf16")
PORT_BASE = int(os.environ.get("MUSE_PORT_BASE", "13312"))
CONTROL_BASE = PORT_BASE
COLLECTIVE_BASE = PORT_BASE
SESSION_BASE = PORT_BASE + 256
COLLECTIVE_ID = 0x4D555345  # 'MUSE'
MODEL_REVISION = "4177486a9f199bd7be520eff14431071d5d41ec5"
NODE_TARGET = "cuda.sm121.muse.resident_decode_stage.bf16"
STAGED_CONTEXT = 131072
STAGED_SLOTS = 4

TP_COLLECTIVE = {
    "backend": "hidden_transport",
    "backend_module_path": "lib/hidden_transport.so",
    "collective_identifier": COLLECTIVE_ID,
    "listen_port": COLLECTIVE_BASE,
    "connect_timeout_milli": 180000,
    "operation_timeout_milli": 30000,
    "peer_hosts": list(HOSTS),
    "peer_ports": [COLLECTIVE_BASE + r for r in range(TP)],
    "algorithms": ["tree"],
    "session_ports": [
        [SESSION_BASE + a * TP + b if a != b else 0
         for b in range(TP)] for a in range(TP)],
}


def stage_config(rank: int) -> dict:
    cfg = {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": "bf16",
        "stage_pack_path": "packs/muse_tp16_rank%02d.bf16.gsmu" % rank,
        "max_sequence_positions": STAGED_CONTEXT,
        "execution_row_capacity": STAGED_SLOTS,
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
            "stage_index": 0,
            "runtime_root": RUNTIME_ROOT.format(host=host),
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/muse_stage.json",
            "kv_backing_directory": "/home/%s/kvcache/muse.tp16" % host,
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
            "control_port_base": PORT_BASE,
        },
        "runtime_limits": {
            "max_inflight_submissions": 4,
            "max_active_sequences": STAGED_SLOTS,
            "max_input_rows": STAGED_SLOTS,
            "resident_sequence_capacity": STAGED_SLOTS,
            "kv_logical_page_capacity": STAGED_SLOTS * ((STAGED_CONTEXT + 63) // 64),
            "kv_physical_page_capacity": STAGED_SLOTS * ((STAGED_CONTEXT + 63) // 64),
        },
        "nodes": nodes,
    }


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/muse-deploy"
    for rank, host in enumerate(HOSTS):
        host_dir = os.path.join(out, host, "config")
        os.makedirs(host_dir, exist_ok=True)
        with open(os.path.join(host_dir, "muse_stage.json"), "w") as f:
            json.dump(stage_config(rank), f, indent=2)
            f.write("\n")
        with open(os.path.join(host_dir, "model_resident.json"), "w") as f:
            json.dump(resident_deployment(), f, indent=2)
            f.write("\n")
    print("generated %d host configs under %s (port base %d)" % (TP, out, PORT_BASE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
