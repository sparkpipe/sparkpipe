#!/usr/bin/env python3
"""Generate the ling TP16 deployment tree: per-rank stage configs and the
shared multi-node model_resident.json.

Hosts: rank r -> spark hex letter r (fleet pack policy), overridable via
LING_TP_HOSTS. The tp_collective session port base is the frozen fleet
ledger value (sparkpipe-coord/PORT_LEDGER.md): ling = 12288, block
12288-13311, sized for the full TP16 session matrix plus every
route-kind offset (+768 worst case) with margin. LING_SESSION_BASE
overrides for a renumbered fleet.

Usage:
  python3 tools/ling_gen_deployment.py --output deployment/ling_tp16
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HOSTS = [h for h in os.environ.get(
    "LING_TP_HOSTS",
    ",".join(f"spark{hex(r)[2:]}" for r in range(16))).split(",") if h]
TP = len(HOSTS)
ROOT_NAME = os.environ.get("LING_ROOT_NAME", "ling.bf16.tp16")
ARM = ROOT_NAME
RUNTIME_ROOT = os.environ.get(
    "LING_RUNTIME_ROOT", "/home/{host}/sparkdata/" + ARM)
CONTROL_BASE = int(os.environ.get("LING_CONTROL_BASE", "19590"))
COLLECTIVE_BASE = int(os.environ.get("LING_COLLECTIVE_BASE", "63560"))
TRANSPORT_BASE = int(os.environ.get("LING_TRANSPORT_BASE", "60730"))
SESSION_BASE = os.environ.get("LING_SESSION_BASE", "12288")
SESSION_HC_BASE = os.environ.get("LING_SESSION_HC_BASE")
COLLECTIVE_ID = 9911223344556680
BACKEND = os.environ.get("LING_BACKEND", "hidden_transport")
PACK_TEMPLATE = os.environ.get(
    "LING_PACK_TEMPLATE", "packs/" + ARM + ".rank%x.sp")
MODEL_REVISION = "e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3"
NODE_TARGET = "cuda.sm121.ling.resident_decode_stage.bf16.expert_bf16"


def session_port_table(base: int) -> list:
    return [[base + a * TP + b if a != b else 0
             for b in range(TP)] for a in range(TP)]


def tp_collective() -> dict:
    if BACKEND != "hidden_transport":
        raise SystemExit(
            f"LING_BACKEND={BACKEND}: the ling driver program requires the "
            f"hidden-transport backend (REQUIRES_HIDDEN_TRANSPORT)")
    base = int(SESSION_BASE)
    if base <= 0 or base + TP * TP - 1 + 768 > 65535:
        raise SystemExit(
            f"LING_SESSION_BASE {base}: the session matrix plus the "
            f"route-kind offsets (+768 D2A_ACK worst case) must stay under "
            f"65535; the frozen ledger value for ling is 12288")
    table = {
        "backend": BACKEND,
        "backend_module_path": "lib/hidden_transport.so",
        "algorithms": ["tree"],
        "collective_identifier": COLLECTIVE_ID,
        "listen_port": COLLECTIVE_BASE,
        "connect_timeout_milli": 30000,
        "operation_timeout_milli": 30000,
        "peer_hosts": list(HOSTS),
        "peer_ports": [COLLECTIVE_BASE + r for r in range(TP)],
        "split_ring_min_payload_bytes": 0,
        "direct_all_to_all_max_payload_bytes": 0,
        "rail_peer_hosts": [list(HOSTS), list(HOSTS)],
        "step_rail_indices": [0] + [1] * (TP - 1),
        "session_ports": session_port_table(int(SESSION_BASE)),
        "session_ports_hc": [[0] * TP for _ in range(TP)],
    }
    if SESSION_HC_BASE is not None:
        hc = int(SESSION_HC_BASE)
        if SESSION_HC_BASE and hc != 0 and \
                abs(hc - int(SESSION_BASE)) < TP * TP:
            raise SystemExit(
                f"LING_SESSION_HC_BASE {hc} overlaps the LING_SESSION_BASE "
                f"matrix ({TP * TP} ports); the second collective must not "
                f"bind the same listeners")
    return table


def stage_config(rank: int) -> dict:
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": "bf16",
        "stage_pack_path": PACK_TEMPLATE % rank,
        "max_sequence_positions": 32768,
        "execution_row_capacity": 1024,
        "decode_split_context_threshold": 2048,
        "tp_degree": TP,
        "tp_rank": rank,
        "tp_collective": dict(tp_collective(),
                              listen_port=COLLECTIVE_BASE + rank),
    }


EOS_TOKEN_ID = 156895


def eos_token_ids() -> list:
    contract = json.loads(
        (REPO_ROOT / "model_contracts/ling_authoritative.json").read_text())
    tokenizer = contract.get("tokenizer", {})
    pinned = tokenizer.get("eos_token_id")
    return [pinned if pinned is not None else EOS_TOKEN_ID]


def resident_deployment() -> dict:
    page_capacity = TP * ((stage_config(0)["max_sequence_positions"] + 63) // 64)
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": RUNTIME_ROOT.format(host=host),
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/stage.json",
            "kv_backing_directory": f"/home/{host}/kvcache/{ARM}",
            "kv_backing_maximum_bytes": 0,
            "control_endpoint": {
                "kind": "tcp",
                "host": host,
                "port": CONTROL_BASE + rank,
            },
        })
    return {
        "schema_version": 2,
        "eos_token_ids": eos_token_ids(),
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
            "socket_path": "/tmp/spark_weightd.sock",
        },
        "runtime_limits": {
            "max_inflight_submissions": 4,
            "max_active_sequences": 16,
            "max_input_rows": 1024,
            "resident_sequence_capacity": 16,
            "kv_logical_page_capacity": page_capacity,
            "kv_physical_page_capacity": page_capacity,
        },
        "nodes": nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    (root / "config").mkdir(parents=True, exist_ok=True)
    collective = tp_collective()
    for rank in range(TP):
        (root / "config" / ("stage_%02d.json" % rank)).write_text(
            json.dumps(stage_config(rank), indent=1) + "\n")
    (root / "model_resident.json").write_text(
        json.dumps(resident_deployment(), indent=1) + "\n")
    print(f"{root}: {TP} stage configs + model_resident.json "
          f"(hosts {HOSTS[0]}..{HOSTS[-1]}, session base {SESSION_BASE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
