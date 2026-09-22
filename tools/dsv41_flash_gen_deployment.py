#!/usr/bin/env python3
"""Generate the dsv41_flash shared-lane deployment tree: per-rank stage
configs and the shared model_resident.json (the GLM shared-socket pattern
from tools/glm5_next_gen_deployment.py, applied to lane 4).

Defaults are lane 4 of the eight shared developer lanes (PR #1083,
amended by PR #1094): TP4 on spark4..spark7, control 23064-23079,
collective 53064-53079 (renumbered from 67064: port numbers above
65535 are not bindable), transport 64064-64079. Every value is
env-overridable so the same generator serves other lane geometries
without edits.

The stage config member list is the dsv41_flash module node context
(spark_dsv41_flash_resident_decode_stage_firmware.h): the family serving
adapter must map exactly these members; the module validates the mapped
context (single stage owns all 40 layers at TP4; 384 % tp == 0 and
129280 % tp == 0 are module-level fail-closed checks).

Usage:
  python3 tools/dsv41_flash_gen_deployment.py --output deployment/dsv41_flash_tp4
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

HOSTS = [h for h in os.environ.get(
    "DSV41_FLASH_TP_HOSTS", "spark4,spark5,spark6,spark7").split(",") if h]
TP = len(HOSTS)
ROOT_NAME = os.environ.get("DSV41_FLASH_ROOT_NAME", "dsv41flash.mxfp4.tp4")
RUNTIME_ROOT = os.environ.get("DSV41_FLASH_RUNTIME_ROOT",
                              "/home/{host}/sparkdata/" + ROOT_NAME)
CONTROL_BASE = int(os.environ.get("DSV41_FLASH_CONTROL_BASE", "23064"))
COLLECTIVE_BASE = int(os.environ.get("DSV41_FLASH_COLLECTIVE_BASE", "53064"))
TRANSPORT_BASE = int(os.environ.get("DSV41_FLASH_TRANSPORT_BASE", "64064"))
COLLECTIVE_ID = int(os.environ.get("DSV41_FLASH_COLLECTIVE_ID",
                                   "9911223344556684"))
BACKEND = os.environ.get("DSV41_FLASH_BACKEND", "hidden_transport")
PACK_TEMPLATE = os.environ.get(
    "DSV41_FLASH_PACK_TEMPLATE", "packs/rank%x.spstage")
MODEL_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
NODE_TARGET = "cuda.sm121.dsv41_flash.resident_decode_stage.bf16.expert_mxfp4"
EXPERT_CODEC = os.environ.get("DSV41_FLASH_EXPERT_CODEC", "mxfp4")
# The fleet-wide shared weightd (sparkpipe-weightd-shared.service): the
# wrapper overrides this from SPARK_WEIGHTD_SOCKET at prepare time; the
# default pins the shared path so a bare deployment never names a
# stood-down or private socket.
WEIGHTD_SOCKET = os.environ.get("DSV41_FLASH_WEIGHTD_SOCKET",
                                "/run/sparkpipe-weightd-shared/weightd.sock")
MAX_SEQUENCE_POSITIONS = int(os.environ.get(
    "DSV41_FLASH_MAX_SEQUENCE_POSITIONS", "32768"))
# The module's own shipped defaults (modules/dsv41_flash_resident_decode_stage/
# Makefile): EXECUTION_ROW_CAPACITY 64, PIPELINE_SLOT_COUNT 2,
# MAX_ACTIVE_SEQUENCES-derived resident capacity 16 keeps the lane inside its
# 6400 MiB device budget; raising any of these is a budget question first.
EXECUTION_ROW_CAPACITY = 64
PIPELINE_SLOT_COUNT = 2
RESIDENT_SEQUENCE_CAPACITY = 16

# The two fabric rails for the spark4-7 quartet, from the qualified dsv4 TP4
# ring (qualification/dsv4/performance/tp4_b1_20260815/.../dsv4_flash_tp4_
# stage.json): rail 0 = 10.10.200.0/23, rail 1 = 10.10.100.8/23.
RAIL0_HOSTS = [h for h in os.environ.get(
    "DSV41_FLASH_RAIL0_HOSTS",
    "10.10.200.4,10.10.200.5,10.10.200.6,10.10.200.7").split(",") if h]
RAIL1_HOSTS = [h for h in os.environ.get(
    "DSV41_FLASH_RAIL1_HOSTS",
    "10.10.100.14,10.10.100.15,10.10.100.16,10.10.100.17").split(",") if h]

# The collective rides the qualified dsv4 TP4 lane shape (same backend,
# algorithm set, payload thresholds and rail routing); peer listener ports
# sit inside the lane's collective block.
TP_COLLECTIVE = {
    "backend": BACKEND,
    "backend_module_path":
        "lib/hidden_transport.so" if BACKEND == "hidden_transport"
        else "lib/libnccl.so.2",
    "collective_identifier": COLLECTIVE_ID,
    "listen_port": COLLECTIVE_BASE,
    "connect_timeout_milli": 30000,
    "operation_timeout_milli": 30000,
    "peer_hosts": list(HOSTS),
    "peer_ports": [COLLECTIVE_BASE + r for r in range(TP)],
    "algorithms": ["recursive_doubling", "direct_all_to_all",
                   "counter_rotating_split_ring"],
    "direct_all_to_all_max_payload_bytes": 81920,
    "split_ring_min_payload_bytes": 655360,
    "rail_peer_hosts": [list(RAIL0_HOSTS), list(RAIL1_HOSTS)],
    "step_rail_indices": [0] + [1] * (TP - 1),
}

if TP_COLLECTIVE["backend"] == "nccl":
    for _nccl_extra in ("algorithms", "direct_all_to_all_max_payload_bytes",
                        "split_ring_min_payload_bytes", "rail_peer_hosts",
                        "step_rail_indices"):
        TP_COLLECTIVE.pop(_nccl_extra, None)


def stage_config(rank: int) -> dict:
    """The dsv41_flash single-stage TP config for one rank.

    Members follow the module node context exactly; the family serving
    adapter (when it lands) validates this set with no extras.
    """
    return {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": EXPERT_CODEC,
        "stage_count": 1,
        "stage_index": 0,
        "first_layer_index": 0,
        "layer_count": 40,
        "stage_pack_path": PACK_TEMPLATE % rank,
        "max_sequence_positions": MAX_SEQUENCE_POSITIONS,
        "execution_row_capacity": EXECUTION_ROW_CAPACITY,
        "resident_sequence_capacity": RESIDENT_SEQUENCE_CAPACITY,
        "pipeline_slot_count": PIPELINE_SLOT_COUNT,
        "tp_degree": TP,
        "tp_rank": rank,
        "tp_collective": dict(TP_COLLECTIVE,
                              listen_port=COLLECTIVE_BASE + rank),
    }


def resident_deployment() -> dict:
    page_capacity = 16 * ((MAX_SEQUENCE_POSITIONS + 63) // 64)
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": 0,
            "runtime_root": RUNTIME_ROOT.format(host=host),
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/stage.json",
            "kv_backing_directory":
                RUNTIME_ROOT.format(host=host) + "/kvcache",
            "kv_backing_maximum_bytes": 0,
            "control_endpoint": {
                "kind": "tcp",
                "host": host,
                "port": CONTROL_BASE + rank,
            },
        })
    return {
        "schema_version": 2,
        "eos_token_ids": [1],
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
        # Shared-socket weightd attach: the wrapper exports the actual
        # socket path (SPARK_WEIGHTD_SOCKET) and overrides this default at
        # prepare time; keeping the path in the deployment is the GLM W2b
        # contract (residentd ensures the daemon from the config).
        "weightd": {
            "socket_path": WEIGHTD_SOCKET,
        },
        "runtime_limits": {
            "max_inflight_submissions": 4,
            "max_active_sequences": RESIDENT_SEQUENCE_CAPACITY,
            "max_input_rows": EXECUTION_ROW_CAPACITY,
            "resident_sequence_capacity": RESIDENT_SEQUENCE_CAPACITY,
            "kv_logical_page_capacity": page_capacity,
            "kv_physical_page_capacity": page_capacity,
        },
        "nodes": nodes,
    }


def render_stage(configuration: dict) -> str:
    rendered = json.dumps(configuration, indent=1)
    return re.sub(r"\[\n(?:\s+\d+,?\n)+\s*\]",
                  lambda match: json.dumps(json.loads(match.group()),
                                           separators=(",", ":")),
                  rendered) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    (root / "config").mkdir(parents=True, exist_ok=True)
    for rank in range(TP):
        (root / "config" / ("stage_%02d.json" % rank)).write_text(
            render_stage(stage_config(rank)))
    (root / "model_resident.json").write_text(
        json.dumps(resident_deployment(), indent=1) + "\n")
    print(f"{root}: {TP} stage configs + model_resident.json "
          f"(hosts {HOSTS[0]}..{HOSTS[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
