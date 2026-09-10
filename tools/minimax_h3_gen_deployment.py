#!/usr/bin/env python3
"""Generate the minimax-h3 TP4xPP4 resident-media deployment (16 ranks).

Emits one stage config per rank plus the deployment root. Session route
matrices use the frozen official minimax-h3 block 17408-18431
(sparkpipe-coord/PORT_LEDGER.md v2): matrix cells at
BASE + group*64 + a*TP + b with the route-kind offsets +256/+512/+768 from
ring/transport/tp_device_collective.c; the maximum used port is
BASE + 3*64 + 15 + 768 = 18383, inside the block with 48 ports of margin.
The generator fails closed if any emitted port leaves the assigned block.
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
TP = 4
PP = 4
ARM = "h3.bf16.tp4pp4"
OFFICIAL_BASE = 17408
BLOCK_LIMIT = 18431
ROUTE_KIND_MAX_OFFSET = 768
TRANSPORT_BASE = int(os.environ.get("MINIMAX_H3_TRANSPORT_BASE", "60920"))
COLLECTIVE_ID_BASE = 0x004D494E31485333
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DIFFUSERS_COMMIT = "3c22124636a332a0df02910f609315d4c2898f57"
NODE_TARGET = "cuda.sm121.minimax_h3.resident_media_stage.bf16"
DIT_BLOCKS_PER_STAGE = (13, 13, 13, 11)


def session_matrix(group: int, base: int) -> list[list[int]]:
    tp = TP
    return [[base + group * 64 + a * tp + b if a != b else 0
             for b in range(tp)] for a in range(tp)]


def stage_config(rank: int, base: int) -> dict:
    group = rank // TP
    tp_rank = rank % TP
    stage = rank // TP
    hosts = HOSTS[group * TP:group * TP + TP]
    collective_listen_base = 64200 + group * 16
    return {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "diffusers_commit": DIFFUSERS_COMMIT,
        "stage_pack_path": f"packs/{ARM}.rank{rank}.sp",
        "max_sequence_positions": 16384,
        "execution_row_capacity": 16384,
        "tp_degree": TP,
        "tp_rank": tp_rank,
        "pp_stage": stage,
        "pp_stage_count": PP,
        "dit_block_count": DIT_BLOCKS_PER_STAGE[stage],
        "tp_collective": {
            "backend": "hidden_transport",
            "backend_module_path": "lib/hidden_transport.so",
            "algorithms": ["tree", "direct_all_to_all"],
            "collective_identifier": COLLECTIVE_ID_BASE + group,
            "listen_port": collective_listen_base + tp_rank,
            "connect_timeout_milli": 30000,
            "operation_timeout_milli": 30000,
            "peer_hosts": hosts,
            "peer_ports": [collective_listen_base + t for t in range(TP)],
            "split_ring_min_payload_bytes": 0,
            "direct_all_to_all_max_payload_bytes": 81920,
            "rail_peer_hosts": [list(hosts), list(hosts)],
            "step_rail_indices": [0] + [1] * (TP - 1),
            "session_ports": session_matrix(group, base),
            "session_ports_hc": [[0 for _ in range(TP)] for _ in range(TP)],
        },
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
    maximum = args.base + (PP - 1) * 64 + (TP - 1) * TP + (TP - 1) + \
        ROUTE_KIND_MAX_OFFSET
    if maximum > BLOCK_LIMIT:
        raise SystemExit(f"highest emitted port {maximum} leaves the assigned block")
    args.out.mkdir(parents=True, exist_ok=True)
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
        "topology": "TP4xPP4 on 16 sparks",
        "session_port_block": [args.base, BLOCK_LIMIT],
        "highest_session_port_with_route_offsets": maximum,
        "host_order": HOSTS,
        "ranks": [{"rank": rank, "host": HOSTS[rank // TP], "pack": f"packs/{ARM}.rank{rank}.sp",
                   "tp_rank": rank % TP, "pp_stage": rank // TP} for rank in range(16)],
    }
    (args.out / "deployment_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote 16 stage configs + deployment_manifest.json under {args.out}; "
          f"highest session port {maximum} within [{args.base}, {BLOCK_LIMIT}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
