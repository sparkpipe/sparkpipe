#!/usr/bin/env python3
"""Generate the gemma4-26b TP4xPP4 deployment tree: per-rank stage configs
and the shared multi-node model_resident.json.

Topology: 16 world ranks over spark0..sparkf; rank = node % 4 gives the TP
lane inside a pipeline stage, stage = node // 4. Stage layer lists {8,8,7,7}
(30 layers). Ports ride the gemma4-26b block 15360-16383 (PORT_LEDGER):
control 15360+rank, transport 15700+rank. The TP collective rides weightd's
mesh and needs no ports.

Usage:
  python3 tools/gemma4_gen_deployment.py --output deployment/gemma4_26b_tp4pp4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PP_STAGES = 4
TP_DEGREE = 4
RANKS = PP_STAGES * TP_DEGREE
STAGE_LAYER_COUNTS = [8, 8, 7, 7]
HOSTS = [h for h in os.environ.get(
    "GEMMA4_TP_HOSTS",
    ",".join(f"spark{hex(r)[2:]}" for r in range(RANKS))).split(",") if h]
if len(HOSTS) != RANKS:
    raise SystemExit(f"GEMMA4_TP_HOSTS must list exactly {RANKS} hosts (tp4 x pp4)")
RUNTIME_ROOT = os.environ.get(
    "GEMMA4_RUNTIME_ROOT",
    "/home/{host}/sparkdata/gemma4_26b.tp4pp4.t1")
CONTROL_BASE = int(os.environ.get("GEMMA4_CONTROL_BASE", "15360"))
TRANSPORT_BASE = int(os.environ.get("GEMMA4_TRANSPORT_BASE", "15700"))
WEIGHTSD_SOCKET = os.environ.get(
    "GEMMA4_WEIGHTSD_SOCKET", "/run/sparkpipe-weightsd/weightsd.sock")
PACK_TEMPLATE = "packs/gemma4_26b_tp4_rank%d_stage%d.gemma4sp"
CONTRACT = json.loads((Path(__file__).resolve().parents[1] /
                       "model_contracts/gemma4_26b_a4b_authoritative.json").read_text())
MODEL_REVISION = CONTRACT["source_revision"]
if not MODEL_REVISION:
    raise SystemExit("contract source_revision missing; refusing to emit")
NODE_TARGET = "cuda.sm121.gemma4.26b-a4b.resident_decode_stage.bf16"
EOS_TOKEN_IDS = [1, 106, 50]


def stage_config(rank: int) -> dict:
    stage = rank // TP_DEGREE
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "stage_pack_path": PACK_TEMPLATE % (rank % TP_DEGREE, stage),
        "max_sequence_positions": 32768,
        "tp_degree": TP_DEGREE,
    }


def tp_environment(rank: int) -> dict:
    return {
        "SPARK_GEMMA4_TP_DEGREE": str(TP_DEGREE),
        "SPARK_GEMMA4_TP_RANK": str(rank % TP_DEGREE),
        "SPARK_GEMMA4_TP_STANDALONE": "0",
        "SPARK_GEMMA4_STAGE_TP_TIMEOUT_MS": "30000",
    }


def resident_deployment() -> dict:
    page_capacity = 16 * ((stage_config(0)["max_sequence_positions"] + 63) // 64)
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank // TP_DEGREE,
            "runtime_root": RUNTIME_ROOT.format(host=host),
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/stage.json",
            "kv_backing_directory": "/home/%s/kvcache/gemma4_26b.tp4pp4.t1" % host,
            "kv_partition": "/",
            "kv_backing_maximum_bytes": 0,
            "control_endpoint": {
                "kind": "tcp",
                "host": host,
                "port": CONTROL_BASE + rank,
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
            "control_port_base": TRANSPORT_BASE,
        },
        "weightd": {
            "socket_path": WEIGHTSD_SOCKET,
        },
        "runtime_limits": {
            "max_inflight_submissions": 4,
            "max_active_sequences": 16,
            "max_input_rows": 128,
            "resident_sequence_capacity": 16,
            "kv_logical_page_capacity": page_capacity,
            "kv_physical_page_capacity": page_capacity,
        },
        "nodes": nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--with-environment", action="store_true",
                        help="emit per-rank module environment json beside the configs")
    arguments = parser.parse_args()
    root = Path(arguments.output)
    (root / "config").mkdir(parents=True, exist_ok=True)
    for rank in range(RANKS):
        (root / "config" / ("stage_%02d.json" % rank)).write_text(
            json.dumps(stage_config(rank), indent=1) + "\n")
        if arguments.with_environment:
            (root / "config" / ("env_%02d.json" % rank)).write_text(
                json.dumps(tp_environment(rank), indent=1, sort_keys=True) + "\n")
    (root / "model_resident.json").write_text(
        json.dumps(resident_deployment(), indent=1) + "\n")
    print(f"{root}: {RANKS} stage configs + model_resident.json "
          f"(hosts {HOSTS[0]}..{HOSTS[-1]}, tp{TP_DEGREE} x pp{PP_STAGES})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
