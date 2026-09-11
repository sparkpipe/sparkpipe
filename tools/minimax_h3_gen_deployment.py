#!/usr/bin/env python3
"""Generate the minimax-h3 TP16xPP1 resident-media deployment (16 ranks).

Emits one stage config per rank plus the deployment root. Session route
matrices use the frozen official minimax-h3 block 17408-18431
(sparkpipe-coord/PORT_LEDGER.md v2): matrix cells at BASE + a*TP + b with the
route-kind offsets +256/+512/+768 from ring/transport/tp_device_collective.c;
the highest referenced port is BASE + 15*16 + 15 + 768 = 18431, exactly the
block limit (zero margin - ledger data point, the matrix is parse-compat
only on the post-#913 mesh dataflow which dials no session ports). The
generator fails closed if any emitted port leaves the assigned block.
session_ports_hc is the all-zero (disabled) matrix: the h3 vertical slice
runs no hidden-control peers, and 0 is the established disabled cell value.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HOSTS = [f"spark{hex(rank)[2:]}" for rank in range(16)]
TP = 16
PP = 1
ARM = "h3.bf16.tp16"
OFFICIAL_BASE = 17408
BLOCK_LIMIT = 18431
ROUTE_KIND_MAX_OFFSET = 768
COLLECTIVE_ID_BASE = 0x004D494E31485333
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DIFFUSERS_COMMIT = "3c22124636a332a0df02910f609315d4c2898f57"
NODE_TARGET = "cuda.sm121.minimax_h3.resident_media_stage.bf16"
DIT_BLOCK_COUNT = 50


def session_matrix(base: int) -> list[list[int]]:
    return [[base + a * TP + b if a != b else 0
             for b in range(TP)] for a in range(TP)]


def stage_config(rank: int, base: int) -> dict:
    collective_listen_base = 64200
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "stage_pack_path": f"packs/{ARM}.rank{rank:02d}.sp",
        "max_sequence_positions": 16384,
        "execution_row_capacity": 16384,
        "tp_degree": TP,
        "tp_rank": rank,
        "pp_stage": 0,
        "pp_stage_count": PP,
        "dit_block_count": DIT_BLOCK_COUNT,
        "tp_collective": {
            "backend": "hidden_transport",
            "backend_module_path": "lib/hidden_transport.so",
            "algorithms": ["tree", "direct_all_to_all"],
            "collective_identifier": COLLECTIVE_ID_BASE,
            "listen_port": collective_listen_base + rank,
            "connect_timeout_milli": 30000,
            "operation_timeout_milli": 30000,
            "peer_hosts": HOSTS,
            "peer_ports": [collective_listen_base + t for t in range(TP)],
            "split_ring_min_payload_bytes": 0,
            "direct_all_to_all_max_payload_bytes": 81920,
            "rail_peer_hosts": [list(HOSTS), list(HOSTS)],
            "step_rail_indices": [0] + [1] * (TP - 1),
            "session_ports": session_matrix(base),
            "session_ports_hc": [[0 for _ in range(TP)] for _ in range(TP)],
        },
    }


def deployment_root(base: int) -> dict:
    transport_base = int(os.environ.get("MINIMAX_H3_TRANSPORT_BASE", "60920"))
    return {
        "schema_version": 2,
        "coordinator_rank_index": 0,
        "adapter": {"shared_object_path": "lib/model_serving_adapter.so"},
        "driver": {"shared_object_path": "lib/model_driver.so",
                   "program_name": "resident_decode"},
        "transport": {"shared_object_path": "lib/hidden_transport.so",
                      "mode": "host-rdma",
                      "control_port_base": transport_base},
        "runtime_limits": {"max_inflight_submissions": 4,
                           "max_active_sequences": 16,
                           "max_input_rows": 16384,
                           "resident_sequence_capacity": 16,
                           "kv_logical_page_capacity": 0,
                           "kv_physical_page_capacity": 0},
        "nodes": [{"rank_index": rank,
                   "stage_index": 0,
                   "runtime_root": f"/home/spark{hex(rank)[2:]}/sparkdata/minimax_h3.tp16",
                   "node_target": NODE_TARGET,
                   "transport_host": f"spark{hex(rank)[2:]}",
                   "adapter_configuration_path": f"stage.{rank}.json",
                   "kv_backing_directory": None,
                   "kv_backing_maximum_bytes": 0,
                   "control_endpoint": {"kind": "tcp",
                                        "host": f"spark{hex(rank)[2:]}",
                                        "port": 19560 + rank}}
                  for rank in range(16)],
        "weightd": {"socket_path": "/tmp/spark_weightd.sock"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("deploy/minimax_h3"))
    parser.add_argument("--base", type=int, default=OFFICIAL_BASE,
                        help="official minimax-h3 session-port block base")
    args = parser.parse_args()
    if args.base != OFFICIAL_BASE:
        raise SystemExit(
            f"base {args.base} is not the frozen official minimax-h3 block base "
            f"{OFFICIAL_BASE} (sparkpipe-coord/PORT_LEDGER.md); ad-hoc blocks are "
            "refused")
    highest_cell = args.base + (TP - 1) * TP + (TP - 1)
    maximum = highest_cell + ROUTE_KIND_MAX_OFFSET
    if maximum > BLOCK_LIMIT:
        raise SystemExit(f"highest emitted port {maximum} leaves the assigned block")
    args.out.mkdir(parents=True, exist_ok=True)
    matrix = session_matrix(args.base)
    for rank in range(16):
        config = stage_config(rank, args.base)
        cells = [cell for row in config["tp_collective"]["session_ports"]
                 for cell in row if cell != 0]
        plane = config["tp_collective"]["peer_ports"] + \
            [config["tp_collective"]["listen_port"]]
        if min(cells) < args.base or max(cells) > BLOCK_LIMIT:
            raise SystemExit(f"rank {rank} session cells leave the assigned block")
        if max(plane) >= 65535:
            raise SystemExit(f"rank {rank} collective ports leave the 60xxx plane")
        (args.out / f"stage.{rank}.json").write_text(json.dumps(config, indent=1))
    manifest = {
        "arm": ARM,
        "topology": "TP16xPP1 on 16 sparks",
        "node_target": NODE_TARGET,
        "session_port_block": [args.base, BLOCK_LIMIT],
        "highest_session_port_with_route_offsets": maximum,
        "route_offset_margin": BLOCK_LIMIT - maximum,
        "route_matrix_consumption": "parse-compat only; the post-#913 mesh "
            "dataflow (weightd-owned QPs over shared memfd slot bands) dials "
            "no session ports - serving_adapter_template.c fails the config "
            "as SCHEMA_ERROR when the matrix is absent",
        "host_order": HOSTS,
        "ranks": [{"rank": rank, "host": HOSTS[rank], "pack": f"packs/{ARM}.rank{rank:02d}.sp",
                   "tp_rank": rank, "pp_stage": 0} for rank in range(16)],
    }
    (args.out / "deployment_manifest.json").write_text(json.dumps(manifest, indent=1))
    (args.out / "model_resident.json").write_text(json.dumps(deployment_root(args.base), indent=1))
    print(f"wrote 16 stage configs + deployment_manifest.json + model_resident.json "
          f"under {args.out}; highest session cell {highest_cell}, with route "
          f"offsets {maximum} = block limit {BLOCK_LIMIT} "
          f"(margin {BLOCK_LIMIT - maximum})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
