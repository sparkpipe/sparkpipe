#!/usr/bin/env python3
"""Generate GLM52 deployment trees: per-rank stage configs and the shared
model_resident.json.

Two pipeline shapes:

- tp8 (default): the flat TP8 single-stage deployment — spark8..sparkf are
  ranks 0..7, every rank runs the full 78-layer firmware stage.
- pp7 (--pipeline pp7): the contract's [12,11x6] pipeline split — 7 stage
  groups x 8 TP ranks, one config per (stage, rank) pair, pack names
  glm52_stage.<k>.tp<r>.glm52sp, per-group collective identifiers and
  per-node control/collective ports so no two processes share a port (the
  flat scheme's BASE+rank plan collides across groups).

Both shapes emit manifest.json describing every expected pack identity; the
three SHAs per pack are filled by the packing run (the generator only fixes
paths and source-config inputs).

Hosts: spark8..sparkf are TP ranks 0..7 (registry order). Port blocks:
control 19480+, collective 63620+, transport 60700.
"""
import argparse
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
# The contract's PP7 split table (single source: model_contracts/glm52.json,
# dspark.pp_stage_layers); keep in sync with it.
PP7_SPLIT = [12, 11, 11, 11, 11, 11, 11]

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


def pp7_first_layers(split):
    firsts = []
    total = 0
    for count in split:
        firsts.append(total)
        total += count
    return firsts


def pp7_stage_config(stage: int, rank: int, split) -> dict:
    """One (stage, rank) firmware config. Ports are offset by the flat node
    index stage*TP+rank so groups never collide; the collective identifier is
    per stage group (a group is the stage's 8-rank TP row)."""
    firsts = pp7_first_layers(split)
    assert sum(split) == 78 and all(c >= 1 for c in split)
    node = stage * TP + rank
    collective = dict(TP_COLLECTIVE)
    collective["collective_identifier"] = COLLECTIVE_ID + stage
    collective["listen_port"] = COLLECTIVE_BASE + node
    collective["peer_ports"] = [COLLECTIVE_BASE + stage * TP + r
                                for r in range(TP)]
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": "fp8",
        "stage_pack_path": "packs/glm52_stage.%d.tp%d.fp8.glm52sp" % (stage, rank),
        "max_sequence_positions": 4096,
        "execution_row_capacity": 16,
        "tp_degree": TP,
        "tp_rank": rank,
        "tp_collective": collective,
        # Header-triple cross-check inputs (the module re-derives them from
        # its compiled geometry table; recorded here for the manifest audit).
        "pp_stage_count": len(split),
        "pp_stage_index": stage,
        "pp_stage_first_layer": firsts[stage],
        "pp_stage_layer_count": split[stage],
    }


def pp7_resident_deployment(split) -> dict:
    nodes = []
    for stage in range(len(split)):
        for rank, host in enumerate(HOSTS):
            node = stage * TP + rank
            nodes.append({
                "rank_index": rank,
                "stage_index": stage,
                "runtime_root": "/home/%s/sparkdata/glm52.pp7.stage%d.tp%d.fp8"
                                % (host, stage, rank),
                "node_target": NODE_TARGET,
                "transport_host": host,
                "adapter_configuration_path":
                    "config/glm52_stage.%d.tp%d.json" % (stage, rank),
                "kv_backing_directory":
                    "/home/%s/kvcache/glm52.pp7.stage%d" % (host, stage),
                "kv_backing_maximum_bytes": 8589934592,
                "control_endpoint": {
                    "kind": "tcp",
                    "host": host,
                    "port": CONTROL_BASE + node,
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


def pp7_manifest(split) -> dict:
    firsts = pp7_first_layers(split)
    packs = []
    for stage in range(len(split)):
        for rank in range(TP):
            packs.append({
                "stage_index": stage,
                "tp_rank": rank,
                "path": "packs/glm52_stage.%d.tp%d.fp8.glm52sp" % (stage, rank),
                "first_layer_index": firsts[stage],
                "layer_count": split[stage],
                "stage_count": len(split),
                # Filled by tools/glm52_resident_stagepack.py --split at
                # packing time; the deployment tree stays shippable without
                # pretending to know content hashes it has not seen.
                "source_config_sha256": None,
                "contract_sha256": None,
                "pack_recipe_sha256": None,
            })
    return {
        "schema_version": 1,
        "pipeline": "pp7",
        "split": list(split),
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": "fp8",
        "packs": packs,
    }


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def generate_tp8(out: str) -> None:
    for rank, host in enumerate(HOSTS):
        host_dir = os.path.join(out, host, "config")
        os.makedirs(host_dir, exist_ok=True)
        write_json(os.path.join(host_dir, "glm52_stage.json"), stage_config(rank))
        write_json(os.path.join(host_dir, "model_resident.json"),
                   resident_deployment())
    print("generated %d host configs under %s" % (TP, out))


def generate_pp7(out: str) -> None:
    split = PP7_SPLIT
    write_json(os.path.join(out, "manifest.json"), pp7_manifest(split))
    for stage in range(len(split)):
        for rank, host in enumerate(HOSTS):
            write_json(
                os.path.join(out, host, "config",
                             "glm52_stage.%d.tp%d.json" % (stage, rank)),
                pp7_stage_config(stage, rank, split))
    # One shared multi-node view (every host mounts the same package layout).
    write_json(os.path.join(out, "model_resident.json"),
               pp7_resident_deployment(split))
    print("generated pp7 deployment (%d stages x tp%d = %d configs) under %s"
          % (len(split), TP, len(split) * TP, out))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GLM52 deployments")
    parser.add_argument("output", nargs="?", default="/tmp/glm52-deploy")
    parser.add_argument("--pipeline", choices=("tp8", "pp7"), default="tp8")
    args = parser.parse_args()
    if args.pipeline == "pp7":
        generate_pp7(args.output)
    else:
        generate_tp8(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
