#!/usr/bin/env python3
"""Generate the gemma4-31b TP16 shared-socket deployment tree for lane 6.

Topology: TP16 identity over spark0..sparkf (world rank = node index,
SPARK_TP_MESH_RANKS=0..15). One stage per rank (tp16pp1 packs, landed and
NVMe-verified: ~/sparkdata/gemma4_31b.bf16.tp16/packs).

Lane 6 port blocks (tools/devcycle/lane_assignments.json): control
23096-23111, transport 64096-64111. The TP collective rides weightd's mesh
and needs no ports.

Every emitted listener stays inside those blocks:
  control endpoint   23096+rank   (residentd control)
  transport base     64096        (host-rdma transport control, +rank)

Runtime roots default to the literal ${SPARK_QUEUE_RUNTIME_ROOT} template;
tools/gemma4_tp16_shared_socket.sh substitutes the queue-provided private
root when materializing the deployment. --runtime-root emits a fixed
layout instead (persistent trees outside the queue).

Usage:
  python3 tools/gemma4_tp16_gen_deployment.py --output deployment/gemma4_31b_tp16_lane6 \
      --weightd-socket /tmp/spark_weightd.sock
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleet_weightd import FLEET_WEIGHTD_SOCKET, fleet_weightd_socket_error  # noqa: E402

RANKS = 16
TP_DEGREE = 16
MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
NODE_TARGET = "cuda.sm121.gemma4.31b.resident_decode_stage.bf16"
PACK_TEMPLATE = "packs/gemma4_31b_tp16_rank%s_stage0.gemma4sp"  # rank in hex
HOSTS = [f"spark{hex(r)[2:]}" for r in range(RANKS)]
LANE_CONTROL_BASE = 23096
LANE_TRANSPORT_BASE = 64096
EOS_TOKEN_IDS = [1, 106, 50]
RUNTIME_ROOT_TEMPLATE = "${SPARK_QUEUE_RUNTIME_ROOT}"
MAX_SEQUENCE_POSITIONS = 32768
KV_PAGE_TOKENS = 64


def rank_hex(rank: int) -> str:
    return hex(rank)[2:]


def stage_config(rank: int) -> dict:
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "stage_pack_path": PACK_TEMPLATE % rank_hex(rank),
        "max_sequence_positions": MAX_SEQUENCE_POSITIONS,
        "tp_degree": TP_DEGREE,
    }


def tp_environment(rank: int) -> dict:
    return {
        "SPARK_GEMMA4_TP_DEGREE": str(TP_DEGREE),
        "SPARK_GEMMA4_TP_RANK": str(rank),
        "SPARK_GEMMA4_TP_STANDALONE": "0",
        "SPARK_GEMMA4_STAGE_TP_TIMEOUT_MS": "30000",
    }


def resident_deployment(runtime_root: str, weightd_socket: str) -> dict:
    page_capacity = 16 * ((MAX_SEQUENCE_POSITIONS + KV_PAGE_TOKENS - 1) // KV_PAGE_TOKENS)
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            # The residentd loader rejects duplicate (rank, stage) pairs:
            # a TP16 identity deployment numbers each rank as its own
            # transport stage (the GLM TP16 convention) - stage_index = rank.
            "stage_index": rank,
            "runtime_root": runtime_root,
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/stage.json",
            "kv_backing_directory": runtime_root.rstrip("/") + "/kv",
            "kv_backing_maximum_bytes": 0,
            "control_endpoint": {
                "kind": "tcp",
                "host": host,
                "port": LANE_CONTROL_BASE + rank,
            },
        })
    return {
        "schema_version": 2,
        "eos_token_ids": EOS_TOKEN_IDS,
        "coordinator_rank_index": 0,
        "adapter": {"shared_object_path": "lib/model_serving_adapter.so"},
        "driver": {
            "shared_object_path": "stages/stage_000/model_driver.so",
            "program_name": "resident_decode",
        },
        "transport": {
            "shared_object_path": "lib/hidden_transport.so",
            "mode": "host-rdma",
            "control_port_base": LANE_TRANSPORT_BASE,
        },
        "weightd": {
            "socket_path": weightd_socket,
        },
        # Caps from the family adapter descriptor: max_inflight is 1
        # (SparkGemma4ServingDescriptor) and input rows are bounded by the
        # descriptor's max_input_row_count (MAX_ACTIVE_SEQUENCE_COUNT) - the
        # loader rejects anything above (model_serving_adapter.c:232,
        # launch-6 lesson).
        "runtime_limits": {
            "max_inflight_submissions": 1,
            "max_active_sequences": 16,
            "max_input_rows": 32,
            "resident_sequence_capacity": 16,
            "kv_logical_page_capacity": page_capacity,
            "kv_physical_page_capacity": page_capacity,
        },
        "nodes": nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-root",
                        help="fixed runtime root (default: the literal "
                             "${SPARK_QUEUE_RUNTIME_ROOT} template resolved by "
                             "the shared-socket wrapper)")
    parser.add_argument("--weightd-socket", default=FLEET_WEIGHTD_SOCKET)
    arguments = parser.parse_args()
    socket_error = fleet_weightd_socket_error(arguments.weightd_socket)
    if socket_error:
        raise SystemExit(socket_error)
    runtime_root = arguments.runtime_root or RUNTIME_ROOT_TEMPLATE
    root = Path(arguments.output)
    (root / "config").mkdir(parents=True, exist_ok=True)
    for rank in range(RANKS):
        (root / "config" / ("stage_%02d.json" % rank)).write_text(
            json.dumps(stage_config(rank), indent=1) + "\n")
        (root / "config" / ("env_%02d.json" % rank)).write_text(
            json.dumps(tp_environment(rank), indent=1, sort_keys=True) + "\n")
    (root / "model_resident.json").write_text(
        json.dumps(resident_deployment(runtime_root, arguments.weightd_socket),
                   indent=1) + "\n")
    print(f"{root}: {RANKS} stage configs + envs + model_resident.json "
          f"(TP16 spark0..sparkf, control {LANE_CONTROL_BASE}+, "
          f"transport {LANE_TRANSPORT_BASE}+, "
          f"weightd {arguments.weightd_socket})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
