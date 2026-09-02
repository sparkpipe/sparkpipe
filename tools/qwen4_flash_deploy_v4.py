#!/usr/bin/env python3
"""Generate the qwen4_flash v4 16-rank TP4xPP4 deployment artifacts.

Topology (coordinator-directive primary): 16 world ranks, one per node
spark0..sparkf; PP chain position p hosts PP stage p/4 (a TP4 group).
STAGE ROTATION (storage-risk rule): the PLE-heavy stage 0 (38.95 GiB
packs) sits on spark4-7 (the lane's own nodes, no storage role); the
light stage 1 (14.75 GiB) on spark0-3 so the active-MDS hosts spark2/3
carry ~22 GiB-class ranks only (coordinator-approved class, glm5_next
precedent); stage 2 on spark8-b; stage 3 (head+MTP) on sparkc-f.

Outputs into --output DIR:
  deployment.json        16 nodes in chain order (stage_index = position)
  config/adapter-rN.json per-rank adapter config (pack path, tp_degree)
  launch_table.json      per-rank launch facts (host, pp stage, tp rank,
                         pack, rail IP, group rail CSV, control port)
"""
import argparse
import json
import os

REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
PACKS = "../packs_v4"
# chain position -> (world rank, host); PP stage = position // 4
# Rank-major stages: the runtime's BuildRankPlan derives PP neighbors as
# rank +/- 4, so stage p MUST own ranks 4p..4p+3. Storage rotation kept:
# stage 0 (heaviest) lives on spark4-7.
CHAIN = [
    (0, "spark4"), (1, "spark5"), (2, "spark6"), (3, "spark7"),      # stage 0
    (4, "spark0"), (5, "spark1"), (6, "spark2"), (7, "spark3"),      # stage 1
    (8, "spark8"), (9, "spark9"), (10, "sparka"), (11, "sparkb"),    # stage 2
    (12, "sparkc"), (13, "sparkd"), (14, "sparke"), (15, "sparkf"),  # stage 3
]
TP_PORT_BASE = 64500      # module TP collective (PP transport base is 66640)
PP_TRANSPORT_PORT_BASE = 64000
CONTROL_PORT_BASE = 18180


def rail_ip(rank: int) -> str:
    return f"10.10.100.{10 + rank}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-sequence-positions", type=int, default=32768)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output, "config"), exist_ok=True)
    groups = {}
    nodes = []
    table = []
    for position, (rank, host) in enumerate(CHAIN):
        pp, tp = position // 4, position % 4
        groups.setdefault(pp, []).append(rank)
        pack = f"{PACKS}/qwen4_flash_v4.s{pp}r{tp}.qwen4_flashsp"
        deploy_dir = f"/home/{host}/sparkdata/qwen4_flash.tp4/deploy_v4"
        nodes.append({
            "rank_index": rank,
            "stage_index": position,
            "runtime_root": deploy_dir,
            "node_target": "cuda.sm121.qwen4_flash.resident_decode_stage.fp8",
            "transport_host": host,
            "adapter_configuration_path": f"config/adapter-r{rank}.json",
            "kv_backing_directory": None,
            "kv_backing_maximum_bytes": 0,
            "control_endpoint": {"kind": "tcp", "host": host,
                                 "port": CONTROL_PORT_BASE + position},
        })
        config = {
            "schema_version": 3,
            "model_revision": REVISION,
            "stage_pack_path": pack,
            "max_sequence_positions": args.max_sequence_positions,
            "tp_degree": 4,
        }
        with open(os.path.join(args.output, "config", f"adapter-r{rank}.json"), "w") as handle:
            json.dump(config, handle, indent=1)
            handle.write("\n")
        table.append({
            "rank": rank, "host": host, "chain_position": position,
            "pp_stage": pp, "tp_rank": tp, "pack": pack,
            "rail_ip": rail_ip(rank),
            "tp_hosts": None,  # filled below (group CSV)
            "tp_port_base": TP_PORT_BASE,
            "control_port": CONTROL_PORT_BASE + position,
        })
    for entry in table:
        entry["tp_hosts"] = ",".join(rail_ip(r) for r in groups[entry["pp_stage"]])

    deployment = {
        "schema_version": 2,
        "coordinator_rank_index": CHAIN[0][0],
        "adapter": {"shared_object_path": "lib/libqwen4_flash_serving_adapter.so"},
        "driver": {"shared_object_path": "lib/model_driver.so",
                   "program_name": "resident_decode"},
        "transport": {"shared_object_path": "lib/libhidden_transport_spark_host_rdma_verbs.so",
                      "mode": "host-rdma",
                      "control_port_base": PP_TRANSPORT_PORT_BASE},
        "runtime_limits": {
            "max_inflight_submissions": 1,
            "max_active_sequences": 8,
            "max_input_rows": 8,
            "resident_sequence_capacity": 8,
            "kv_logical_page_capacity": 0,
            "kv_physical_page_capacity": 0,
        },
        "nodes": nodes,
    }
    with open(os.path.join(args.output, "deployment.json"), "w") as handle:
        json.dump(deployment, handle, indent=1)
        handle.write("\n")
    with open(os.path.join(args.output, "launch_table.json"), "w") as handle:
        json.dump(table, handle, indent=1)
        handle.write("\n")
    print(f"qwen4_flash_deploy_v4 wrote deployment.json + {len(table)} configs + launch_table.json "
          f"to {args.output} (stage rotation: s0=spark4-7 s1=spark0-3 s2=spark8-b s3=sparkc-f)")


if __name__ == "__main__":
    main()
