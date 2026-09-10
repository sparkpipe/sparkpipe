#!/usr/bin/env python3
"""Generate the laguna TP16 deployment tree: per-rank stage configs and
the shared multi-node model_resident.json.

Hosts: spark0..sparkf are ranks 0..15 (registry order; rank r -> sparke-hex
r per the fleet pack policy). kv_backing_directory is a DIRECTORY per node
(the KV page store opens it with O_TMPFILE - a file path fails ENOTDIR,
the bring-up finding recorded in the lane report).

Usage:
  python3 tools/laguna_gen_deployment.py --output deployment/laguna_tp16
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# PENDING FLEET RENUMBER: the whole 64800+ session region cannot safely
# host session matrices (route-kind offsets +256/+512/+768 overflow 65535),
# so the session base is REQUIRED from the environment with no frozen
# default; a wrong or absent base fails closed here.
SESSION_BASE_TEXT = os.environ.get("SPARK_LAGUNA_SESSION_BASE", "")
if not SESSION_BASE_TEXT:
    raise SystemExit(
        "SPARK_LAGUNA_SESSION_BASE is required (no frozen default; the "
        "64800+ region is pending the fleet port renumber)")
SESSION_BASE = int(SESSION_BASE_TEXT)

PP_STAGES = 2
TP_DEGREE = 8
RANKS = PP_STAGES * TP_DEGREE
STAGE_LAYER_COUNTS = [48 // PP_STAGES for _ in range(PP_STAGES)]
HOSTS = [h for h in os.environ.get(
    "LAGUNA_TP_HOSTS",
    ",".join(f"spark{hex(r)[2:]}" for r in range(RANKS))).split(",") if h]
if len(HOSTS) != RANKS:
    raise SystemExit(f"LAGUNA_TP_HOSTS must list exactly {RANKS} hosts (tp8 x pp2)")
TP = len(HOSTS)
RUNTIME_ROOT = os.environ.get("LAGUNA_RUNTIME_ROOT",
                              "/home/{host}/sparkdata/laguna-s-2.1.bf16.tp8pp2")
CONTROL_BASE = int(os.environ.get("LAGUNA_CONTROL_BASE", "19560"))
COLLECTIVE_BASE = int(os.environ.get("LAGUNA_COLLECTIVE_BASE", "63640"))
TRANSPORT_BASE = int(os.environ.get("LAGUNA_TRANSPORT_BASE", "60710"))
COLLECTIVE_ID = 9911223344556679
PACK_TEMPLATE = os.environ.get(
    "LAGUNA_PACK_TEMPLATE",
    "packs/laguna-s-2.1.bf16.tp8pp2.stage%d.rank%d.lgsp")
MODEL_REVISION = "PRE-FREEZE"
NODE_TARGET = "cuda.sm121.laguna.resident_decode_stage.bf16.expert_bf16"

TP_COLLECTIVE = {
    "backend": "hidden_transport",
    "backend_module_path": "lib/hidden_transport.so",
    "algorithms": ["tree"],
    "collective_identifier": COLLECTIVE_ID,
    "listen_port": COLLECTIVE_BASE,
    "connect_timeout_milli": 30000,
    "operation_timeout_milli": 30000,
    "peer_hosts": list(HOSTS),
    "peer_ports": [COLLECTIVE_BASE + r for r in range(TP)],
    # d2a rides beside recursive doubling at TP16 (the ABI-13 transport
    # routes tp_degree-1 peers on step rows; 80KB is the lane's payload
    # bound from the d2d measurements) - #760's committed configs.
    "split_ring_min_payload_bytes": 0,
    "direct_all_to_all_max_payload_bytes": 0,
    # The schema REQUIRES exactly 2 rails (MAX_RAIL_COUNT=2) and 3
    # step_rail_indices (SPLIT_RING_ROUTE_COUNT=3) - glm52's template.
    # The async op INVALID_ARGUMENT discriminator is done differently:
    # point BOTH rails at the same (only) fabric device.
    "rail_peer_hosts": [list(HOSTS), list(HOSTS)],
    # d2a peer routes (the #760 form the deployed lane configs carry and
    # the collective requires under the RD|D2A mask: entry 0 on rail 0,
    # all peers on rail 1 - [0,0,0] is the split-ring legacy shape and
    # the collective's multi-route check REJECTS it when d2a is on)
    "step_rail_indices": [0] + [1] * (TP - 1),
    # explicit per-session control ports, [source][sink] - the tree
    # collective reads them verbatim (no derived ports). The base is the
    # env-required SESSION_BASE; keep every session cell within 65535
    # including the +768 route-kind offset until the fleet renumber lands.
    "session_ports": [
        [SESSION_BASE + a * TP + b if a != b else 0
         for b in range(TP)] for a in range(TP)],
}



def stage_config(rank: int) -> dict:
    host = HOSTS[rank]
    # The adapter validates members EXACTLY: these ten (the R3 flash-decode
    # lane added decode_split_context_threshold to the exact-member list and
    # to every committed stage config; a config missing the member is
    # rejected SCHEMA_ERROR at load). 0 = the split path is disabled, the
    # shipped single-pass behavior. The capacities ride the module firmware
    # header defaults; the KV backing directory flows through the deployment
    # node (not the stage config).
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": "bf16",
        "stage_pack_path": PACK_TEMPLATE % (rank // TP_DEGREE, rank),
        "max_sequence_positions": 32768,
        # 1024-row prefill chunks (the module's SPARK_BATCH_BUCKET width):
        # the engine chunks prompts to runtime_limits.max_input_rows, and
        # the shipped 16 made a 32K prompt 2048 sequential submissions -
        # one full weight re-stream + collective latency per 16 tokens,
        # the measured 10 tok/s prefill. Rows are NOT sequence slots:
        # execution_row_capacity is validated against the module's row
        # firmware limit, not resident_sequence_capacity (the GDN-state
        # memory budget stays sized by max_active_sequences=16).
        "execution_row_capacity": 1024,
        # R3 engagement: above 2048 positions the decode attention takes
        # the split-K (flash-decode) form - 4 heads/rank at TP16 means a
        # B1 grid of 4 CTAs on 48 SMs without it. Below the threshold the
        # launcher is byte-identical to the qualified single-pass kernel
        # (attn.cuh: the extremes are exact, the combine deterministic),
        # so short-context outputs do not move. Qualification: the shared
        # kernel's host oracle (tests/host_cuda/glm52_layer_host.cu
        # splitreceipt: extremes bit-exact, launcher-below-threshold
        # bit-for-bit, multi-partition deterministic) + the window cell:
        # split-on vs split-off equivalence at 8K+ context on the resident
        # serving before the decode timing claim.
        "decode_split_context_threshold": 2048,
        "pipeline_stage_count": PP_STAGES,
        "pipeline_stage_index": rank // TP_DEGREE,
        "first_layer_index": (rank // TP_DEGREE) * STAGE_LAYER_COUNTS[0],
        "layer_count": STAGE_LAYER_COUNTS[rank // TP_DEGREE],
        "stage_layer_counts": STAGE_LAYER_COUNTS,
        "session_port_base": SESSION_BASE,
        "tp_degree": TP_DEGREE,
        "tp_rank": rank % TP_DEGREE,
        "tp_collective": dict(TP_COLLECTIVE, listen_port=COLLECTIVE_BASE + rank),
    }


def resident_deployment() -> dict:
    contract = json.loads((Path(__file__).resolve().parents[1] / "model_contracts/laguna_authoritative.json").read_text())
    page_capacity = 16 * ((stage_config(0)["max_sequence_positions"] + 63) // 64)
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": RUNTIME_ROOT.format(host=host),
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/stage.json",
            "kv_backing_directory": "/home/%s/kvcache/laguna-s-2.1.bf16.tp8pp2" % host,
            "kv_backing_maximum_bytes": 0,  # Derive KV + recurrent backing from configured cache geometry.
            "control_endpoint": {
                "kind": "tcp",
                "host": host,
                "port": CONTROL_BASE + rank,
            },
        })
    return {
        "schema_version": 2,
        "eos_token_ids": contract["tokens"]["eos_token_ids"],
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
        # One-minute debug cycle: the residentd publishes the socket to the
        # W2b env contract, ensures the daemon, and resolves the pack
        # digest from the .sha256 sidecar beside the rank pack (write it
        # at pack placement: sha256sum <pack> > <pack>.sha256). The module
        # seam then attaches the warm arena - code-only redeploys skip
        # the 21.7GB re-read.
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
    for rank in range(RANKS):
        (root / "config" / ("stage_%02d.json" % rank)).write_text(
            json.dumps(stage_config(rank), indent=1) + "\n")
    (root / "model_resident.json").write_text(
        json.dumps(resident_deployment(), indent=1) + "\n")
    print(f"{root}: {TP} stage configs + model_resident.json "
          f"(hosts {HOSTS[0]}..{HOSTS[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
