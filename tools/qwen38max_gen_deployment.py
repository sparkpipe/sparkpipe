#!/usr/bin/env python3
"""Emit the Qwen 3.8 Max TP16 deployment manifest (16 ranks, single stage).

Follows tools/glm53full_gen_deployment.py; adapted to the qwen38_max
serving adapter's own configuration contract (the adapter validates its
member set EXACTLY: schema_version, model_revision, stage_pack_path,
max_sequence_positions, tp_degree - a missing member is a load error).

16 world ranks; stage_index is the world rank. tp_degree is the RUNTIME
topology knob: 16 -> TP16xPP1 (this tree), 4 -> TP4xPP4.

Output: ONE json object mapping relative paths to file contents on
stdout - qwen38max-tp16-deploy/<host>/config/{qwen38_stage,
model_resident}.json for the 16 hosts. The tool performs NO filesystem
writes; the caller materializes the manifest wherever the tree belongs.

Host names are validated to [a-z0-9]+. Env overrides: QWEN38MAX_HOSTS,
QWEN38MAX_CONTROL_BASE, QWEN38MAX_COLLECTIVE_BASE, QWEN38MAX_TRANSPORT_BASE.
"""
import json
import os
import re
import sys

HOSTS = [h for h in os.environ.get(
    "QWEN38MAX_HOSTS", ",".join("spark%x" % r for r in range(16))).split(",") if h]
for _host in HOSTS:
    if not re.fullmatch(r"[a-z0-9]+", _host):
        sys.exit("invalid host name %r: must be [a-z0-9]+" % _host)
TP = len(HOSTS)
if TP != 16:
    sys.exit("qwen38max deploys TP16; got %d hosts" % TP)
CONTROL_BASE = int(os.environ.get("QWEN38MAX_CONTROL_BASE", "19600"))
COLLECTIVE_BASE = int(os.environ.get("QWEN38MAX_COLLECTIVE_BASE", "63800"))
TRANSPORT_BASE = int(os.environ.get("QWEN38MAX_TRANSPORT_BASE", "61000"))

# The compiled module's serving pin (modules/qwen38_max_resident_decode_stage/
# Makefile QWEN38_MODEL_REVISION). The serving adapter matches this EXACTLY
# against the module build; a drifted revision refuses to load.
MODEL_REVISION = "d2dc35658bcf77e66643428cb52e774cc3b5bd29"
NODE_TARGET = "cuda.sm121.qwen38.resident_decode_stage.fp8"

# Runtime home (data only, cited inside the configs).
RUNTIME_ROOT_TEMPLATE = "/home/{host}/sparkdata/qwen38max.tp16"


def stage_config(rank):
    # Member set mirrors SparkQwen38MaxServingConfigurationMembers exactly;
    # the module takes the rest of its TP wiring (peers/ports/backend) from
    # the deployment's transport + the residentd environment at admission.
    return {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "stage_pack_path": "packs/qwen38max.tp16-rank%d.qwen38sp" % rank,
        "max_sequence_positions": 4096,
        "tp_degree": TP,
    }


def resident_deployment():
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": RUNTIME_ROOT_TEMPLATE.format(host=host),
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/qwen38_stage.json",
            "kv_backing_directory": "/home/%s/kvcache/qwen38max.tp16" % host,
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
            # One logical page per 64-token block per resident sequence,
            # all physically resident (the family-wide pool law).
            "kv_logical_page_capacity": 16 * ((32768 + 63) // 64),
            "kv_physical_page_capacity": 16 * ((32768 + 63) // 64),
        },
        "nodes": nodes,
    }


def main() -> int:
    manifest = {}
    for rank, host in enumerate(HOSTS):
        base = "qwen38max-tp16-deploy/%s/config" % host
        manifest[base + "/qwen38_stage.json"] = (
            json.dumps(stage_config(rank), indent=2) + "\n")
        manifest[base + "/model_resident.json"] = (
            json.dumps(resident_deployment(), indent=2) + "\n")
    json.dump(manifest, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
