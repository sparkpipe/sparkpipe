#!/usr/bin/env python3
"""Generate the GLM 5.3-full TP16 deployment tree (one per expert arm).

Adapts tools/glm52_gen_deployment.py (the 5.2 TP8 band) to the glm53full
fleet shape: every rank's pack carries ALL 78 layers (pure TP16, single
stage - the module compiles STAGE_COUNT=1, LAYERS_PER_STAGE=78), so
stage_index is 0 on every node and identity rides on tp_rank.

Hosts: spark0..sparkf are ranks 0..15. runtime_root:
/home/{host}/sparkdata/glm53full.{codec}.tp16 (the placed pack set).
Pack names are decimal (rank0..rank15), matching the placement receipts.

Per-arm build identities (tools/glm52_model_contract.py
--print-build-identity {codec}) - the adapter matches model_revision
EXACTLY against the compiled module, so each arm's config cites the
revision its published artifact carries.

Env overrides: GLM53FULL_HOSTS, GLM53FULL_RUNTIME_ROOT_TEMPLATE,
GLM53FULL_CONTROL_BASE, GLM53FULL_COLLECTIVE_BASE, GLM53FULL_TRANSPORT_BASE.
"""
import json
import os
import sys

HOSTS = [h for h in os.environ.get(
    "GLM53FULL_HOSTS", ",".join("spark%x" % r for r in range(16))).split(",") if h]
TP = len(HOSTS)
if TP != 16:
    sys.exit("glm53full deploys TP16; got %d hosts" % TP)
CONTROL_BASE = int(os.environ.get("GLM53FULL_CONTROL_BASE", "19500"))
COLLECTIVE_BASE = int(os.environ.get("GLM53FULL_COLLECTIVE_BASE", "63700"))
TRANSPORT_BASE = int(os.environ.get("GLM53FULL_TRANSPORT_BASE", "60900"))

# Per-arm identity: codec name -> (model_revision, node_target suffix).
# fp8's serving module still carries the legacy 5.2-FP8 pin until the
# coordinator lands the re-frozen contract (see the lane report); bf16
# cites the arm this lane published (artifact b2c526c7...).
ARMS = {
    "bf16": {
        "model_revision": "304b8051cfb2b260b61ce0cbe330e02a98e73639",
    },
    "fp8": {
        "model_revision": "b4734de4facf877f85769a911abafc5283eab3d9",
    },
    "nvfp4": {
        "model_revision": "363e8f086905afd83db356a620f9aa401c23800a",
    },
}

RUNTIME_ROOT_TEMPLATE = os.environ.get(
    "GLM53FULL_RUNTIME_ROOT_TEMPLATE",
    "/home/{host}/sparkdata/glm53full.{codec}.tp16")


def tp_collective(collective_base):
    return {
        "backend": "hidden_transport",
        "backend_module_path": "lib/hidden_transport.so",
        "collective_identifier": 9911223344556677 + collective_base,
        "listen_port": collective_base,
        "connect_timeout_milli": 180000,
        "operation_timeout_milli": 30000,
        "peer_hosts": list(HOSTS),
        "peer_ports": [collective_base + r for r in range(TP)],
        "algorithms": ["recursive_doubling"],
        "direct_all_to_all_max_payload_bytes": 0,
        "split_ring_min_payload_bytes": 0,
        "rail_peer_hosts": [list(HOSTS), list(HOSTS)],
        "step_rail_indices": [0, 0, 1],
    }


def stage_config(rank, codec):
    # The glm52 serving adapter validates members EXACTLY (the ten in the
    # 5.2 generator, decode_split_context_threshold included); a missing
    # member is SCHEMA_ERROR at load.
    return {
        "schema_version": 3,
        "model_revision": ARMS[codec]["model_revision"],
        "expert_weight_codec": codec,
        "stage_pack_path": "packs/glm53full.%s.tp16-rank%d.glm52sp" % (codec, rank),
        "max_sequence_positions": 4096,
        "execution_row_capacity": 16,
        "decode_split_context_threshold": 0,
        "tp_degree": TP,
        "tp_rank": rank,
        "tp_collective": tp_collective(COLLECTIVE_BASE),
    }


def resident_deployment(codec, runtime_root_template):
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            # The deployment schema requires stage_index UNIQUE per node,
            # and the glm52 adapter asserts tp_rank == stage_index
            # (SparkGlm52ServingAdapterConfigure) while overriding the
            # module context to the single 78-layer stage itself. So
            # stage_index here IS the TP rank, not a pipeline stage.
            "stage_index": rank,
            "runtime_root": runtime_root_template.format(host=host, codec=codec),
            "node_target": "cuda.sm121.glm52.resident_decode_stage.bf16.expert_%s" % codec,
            "transport_host": host,
            "adapter_configuration_path": "config/glm52_stage.json",
            "kv_backing_directory": "/home/%s/kvcache/glm53full.%s" % (host, codec),
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
            # Same pool law as the 5.2 tree: one logical page per 64-token
            # block per resident sequence, all physically resident.
            "kv_logical_page_capacity": 16 * ((32768 + 63) // 64),
            "kv_physical_page_capacity": 16 * ((32768 + 63) // 64),
        },
        "nodes": nodes,
    }


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ARMS:
        sys.exit("usage: glm53full_gen_deployment.py {bf16|fp8|nvfp4} [out-root]")
    codec = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/glm53full-%s-deploy" % codec
    runtime_root_template = RUNTIME_ROOT_TEMPLATE
    for rank, host in enumerate(HOSTS):
        host_dir = os.path.join(out, host, "config")
        os.makedirs(host_dir, exist_ok=True)
        with open(os.path.join(host_dir, "glm52_stage.json"), "w") as f:
            json.dump(stage_config(rank, codec), f, indent=2)
            f.write("\n")
        with open(os.path.join(host_dir, "model_resident.json"), "w") as f:
            json.dump(resident_deployment(codec, runtime_root_template), f, indent=2)
            f.write("\n")
    print("generated %d host configs for arm %s under %s" % (TP, codec, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
