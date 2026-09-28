#!/usr/bin/env python3
"""Generate a gemma4-31b TP deployment tree (TP16 lane 6 by default).

--hosts picks the TP group (4 or 16 nodes, world rank = index in the list)
and --lane its port blocks (control 23000+16L, transport 64000+16L); the
TP4 serving arm is --hosts sparka,sparkb,sparkc,sparkd --lane 4 over the
tp4 packs (~/sparkdata/gemma4_31b.bf16.tp4/packs, one 60-layer rank each).

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
layout instead (persistent trees outside the queue); "{host}" in it names
each node's own home.

Usage:
  python3 tools/gemma4_tp16_gen_deployment.py --output deployment/gemma4_31b_tp16_lane6 \
      --weightd-socket /run/sparkpipe-weightd-shared/weightd.sock
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
NODE_TARGET = "cuda.sm121.gemma4.31b.resident_decode_stage.bf16"
PACK_TEMPLATE = "packs/gemma4_31b_tp%d_rank%s_stage0.gemma4sp"
DEFAULT_HOSTS = ",".join(f"spark{hex(r)[2:]}" for r in range(16))
DEFAULT_LANE = 6
SERVING_TP_DEGREES = (4, 16)
EOS_TOKEN_IDS = [1, 106, 50]
RUNTIME_ROOT_TEMPLATE = "${SPARK_QUEUE_RUNTIME_ROOT}"
DEFAULT_MAX_SEQUENCE_POSITIONS = 32768
DEFAULT_RESIDENT_SEQUENCES = 16
DEFAULT_WEIGHTD_SOCKET = "/run/sparkpipe-weightd-shared/weightd.sock"
KV_PAGE_TOKENS = 64
MAX_INPUT_ROWS = 96


def rank_hex(rank: int) -> str:
    return hex(rank)[2:]


def stage_config(rank: int, tp_degree: int, positions: int) -> dict:
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "stage_pack_path": PACK_TEMPLATE % (tp_degree, rank_hex(rank)),
        "max_sequence_positions": positions,
        "tp_degree": tp_degree,
    }


def tp_environment(rank: int, tp_degree: int) -> dict:
    return {
        "SPARK_GEMMA4_TP_DEGREE": str(tp_degree),
        "SPARK_GEMMA4_TP_RANK": str(rank),
        "SPARK_GEMMA4_TP_STANDALONE": "0",
        "SPARK_GEMMA4_STAGE_TP_TIMEOUT_MS": "30000",
        "SPARK_TP_WAIT_MODE": "hardware",
    }


def resident_deployment(runtime_root: str, weightd_socket: str, hosts: list,
                        lane: int, positions: int, sequences: int) -> dict:
    page_capacity = sequences * ((positions + KV_PAGE_TOKENS - 1) // KV_PAGE_TOKENS)
    nodes = []
    for rank, host in enumerate(hosts):
        nodes.append({
            "rank_index": rank,
            # The residentd loader rejects duplicate (rank, stage) pairs:
            # a TP16 identity deployment numbers each rank as its own
            # transport stage (the GLM TP16 convention) - stage_index = rank.
            "stage_index": rank,
            "runtime_root": runtime_root.replace("{host}", host),
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/stage.json",
            "kv_backing_directory": runtime_root.replace("{host}", host).rstrip("/") + "/kv",
            "kv_backing_maximum_bytes": 0,
            "control_endpoint": {
                "kind": "tcp",
                "host": host,
                "port": 23000 + 16 * lane + rank,
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
            "control_port_base": 64000 + 16 * lane,
        },
        "weightd": {
            "socket_path": weightd_socket,
        },
        "runtime_limits": {
            "max_inflight_submissions": 1,
            "max_active_sequences": sequences,
            "max_input_rows": max(MAX_INPUT_ROWS, sequences),
            "resident_sequence_capacity": sequences,
            "kv_logical_page_capacity": page_capacity,
            "kv_physical_page_capacity": page_capacity,
            "max_sequence_positions": positions,
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
    parser.add_argument("--weightd-socket", default=DEFAULT_WEIGHTD_SOCKET)
    parser.add_argument("--hosts", default=DEFAULT_HOSTS)
    parser.add_argument("--lane", type=int, default=DEFAULT_LANE)
    parser.add_argument("--max-sequence-positions", type=int,
                        default=DEFAULT_MAX_SEQUENCE_POSITIONS)
    parser.add_argument("--resident-sequences", type=int,
                        default=DEFAULT_RESIDENT_SEQUENCES)
    arguments = parser.parse_args()
    hosts = [h for h in arguments.hosts.split(",") if h]
    tp_degree = len(hosts)
    if tp_degree not in SERVING_TP_DEGREES or len(set(hosts)) != tp_degree:
        parser.error(f"--hosts names {tp_degree} nodes; the dense serving arms are TP4 and TP16 over distinct nodes")
    if not 0 <= arguments.lane < 16:
        parser.error("--lane outside 0..15")
    runtime_root = arguments.runtime_root or RUNTIME_ROOT_TEMPLATE
    root = Path(arguments.output)
    (root / "config").mkdir(parents=True, exist_ok=True)
    for rank in range(tp_degree):
        (root / "config" / ("stage_%02d.json" % rank)).write_text(
            json.dumps(stage_config(rank, tp_degree, arguments.max_sequence_positions), indent=1) + "\n")
        (root / "config" / ("env_%02d.json" % rank)).write_text(
            json.dumps(tp_environment(rank, tp_degree), indent=1, sort_keys=True) + "\n")
    (root / "model_resident.json").write_text(
        json.dumps(resident_deployment(runtime_root, arguments.weightd_socket,
                                       hosts, arguments.lane,
                                       arguments.max_sequence_positions,
                                       arguments.resident_sequences),
                   indent=1) + "\n")
    sliding_heads = max(16 // tp_degree, 1)
    full_heads = max(4 // tp_degree, 1)
    kv_bytes_per_token = 50 * sliding_heads * 256 * 2 * 2 + 10 * full_heads * 512 * 2 * 2
    kv_tokens = arguments.resident_sequences * (
        (arguments.max_sequence_positions + KV_PAGE_TOKENS - 1) // KV_PAGE_TOKENS) * KV_PAGE_TOKENS
    print(f"kv pools per rank: {kv_tokens} tokens x {kv_bytes_per_token} B = "
          f"{kv_tokens * kv_bytes_per_token} bytes")
    print(f"{root}: {tp_degree} stage configs + envs + model_resident.json "
          f"(TP{tp_degree} {hosts[0]}..{hosts[-1]}, lane {arguments.lane}, "
          f"control {23000 + 16 * arguments.lane}+, "
          f"transport {64000 + 16 * arguments.lane}+, "
          f"weightd {arguments.weightd_socket})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
