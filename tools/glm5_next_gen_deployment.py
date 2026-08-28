#!/usr/bin/env python3
"""Generate the glm5_next TP16 deployment tree: per-rank stage configs and
the shared multi-node model_resident.json.

Hosts: spark0..sparkf are ranks 0..15 (registry order; rank r -> sparke-hex
r per the fleet pack policy). kv_backing_directory is a DIRECTORY per node
(the KV page store opens it with O_TMPFILE - a file path fails ENOTDIR,
the bring-up finding recorded in the lane report).

Usage:
  python3 tools/glm5_next_gen_deployment.py --output deployment/glm5_next_tp16
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HOSTS = [h for h in os.environ.get(
    "GLM5_NEXT_TP_HOSTS",
    ",".join(f"spark{hex(r)[2:]}" for r in range(16))).split(",") if h]
TP = len(HOSTS)
RUNTIME_ROOT = os.environ.get("GLM5_NEXT_RUNTIME_ROOT",
                              "/home/{host}/sparkdata/glm5_next.tp16")
CONTROL_BASE = int(os.environ.get("GLM5_NEXT_CONTROL_BASE", "19560"))
COLLECTIVE_BASE = int(os.environ.get("GLM5_NEXT_COLLECTIVE_BASE", "63640"))
TRANSPORT_BASE = int(os.environ.get("GLM5_NEXT_TRANSPORT_BASE", "60710"))
COLLECTIVE_ID = 9911223344556679
MODEL_REVISION = "84c6a6aa9497188e15a635ba793b0f95a79b1033"
NODE_TARGET = "cuda.sm121.glm5_next.resident_decode_stage.bf16.expert_fp8"

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
    host = HOSTS[rank]
    # The adapter validates members EXACTLY: only these nine. The
    # capacities ride the module firmware header defaults; the KV backing
    # directory flows through the deployment node (not the stage config).
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": "fp8",
        "stage_pack_path": "packs/glm5_next_stage.tp16.rank%d.g5nsp" % rank,
        "max_sequence_positions": 32768,
        "execution_row_capacity": 16,
        "tp_degree": TP,
        "tp_rank": rank,
        "tp_collective": dict(TP_COLLECTIVE, listen_port=COLLECTIVE_BASE + rank),
    }


def resident_deployment() -> dict:
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": RUNTIME_ROOT.format(host=host),
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/stage.json",
            "kv_backing_directory": "/home/%s/kvcache/glm5_next.tp16" % host,
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
            # The schema requires BOTH kv capacity members (exact-member
            # validation) and the adapter (no JIT_KV) requires both ZERO;
            # the module owns its KV pool internally (DRIVER_OWNS_KV).
            # Adopting glm52's JIT_KV lane wiring is the follow-up.
            "kv_logical_page_capacity": 0,
            "kv_physical_page_capacity": 0,
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
        (root / "config" / ("stage_%02d.json" % rank)).write_text(
            json.dumps(stage_config(rank), indent=1) + "\n")
    (root / "model_resident.json").write_text(
        json.dumps(resident_deployment(), indent=1) + "\n")
    print(f"{root}: {TP} stage configs + model_resident.json "
          f"(hosts {HOSTS[0]}..{HOSTS[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
