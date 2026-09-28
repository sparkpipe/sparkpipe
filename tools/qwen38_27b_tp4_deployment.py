#!/usr/bin/env python3
"""Write the Qwen3.8-27B tensor-parallel serving deployment (one rank per node).

Emits into --output:
  model_resident.nospec.json   residentd deployment, adapter config config/stage.json on every node
  model_resident.dflash2.json  residentd deployment, adapter config config/stage.dflash2.json
  stage.rank<R>.json           rank R adapter config, installed on that node as config/stage.json
  stage.dflash2.rank<R>.json   the same with speculative_draft_count, installed as config/stage.dflash2.json
  api.model_resident.json      API copy: the nospec deployment plus eos_token_ids and tokenizer

The runtime limits, EOS ids and API tokenizer block come from
qwen38_27b_tp1_deployment.py; the per-rank tp_degree/tp_rank/tp_collective
block comes from qwen38_27b_lane_deployment.stage_config. The adapter still
validates tp_collective, while the weightd mesh collective reads only degree
and rank (SPARK_TP_MESH_RANKS selects the mesh members), so its ports are
placeholders inside the lane's collective port block. Every value is a
required argument.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KV_BLOCK_TOKENS = 64


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deployment(tp1, args, hosts, adapter_config: str) -> dict:
    document = tp1.deployment(argparse.Namespace(**dict(vars(args), host=hosts[0],
                                                          runtime_root=args.runtime_root.format(host=hosts[0]),
                                                          control_port=args.control_port_base)), adapter_config)
    template = document["nodes"][0]
    document["nodes"] = []
    for rank, host in enumerate(hosts):
        node = json.loads(json.dumps(template))
        node.update(rank_index=rank, stage_index=rank, runtime_root=args.runtime_root.format(host=host),
                    transport_host=host)
        node["control_endpoint"] = {"kind": "tcp", "host": host, "port": args.control_port_base + rank}
        document["nodes"].append(node)
    return document


def stages(tp1, lane, args, hosts, rank: int, draft_count: int | None) -> dict:
    config = lane.stage_config(hosts, rank, f"{args.pack_name}.rank{rank}.qwen36sp", args.attempt,
                               tp1.MODEL_REVISION, args.max_positions,
                               {"collective": args.collective_port_base})
    if draft_count is not None:
        config["speculative_draft_count"] = draft_count
    return config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", required=True, help="comma list; position is the TP rank")
    ap.add_argument("--runtime-root", required=True, help="per-node root with {host}")
    ap.add_argument("--pack-name", required=True, help="packs/<name>.rank<R>.qwen36sp")
    ap.add_argument("--weightd-socket", required=True)
    ap.add_argument("--control-port-base", type=int, required=True)
    ap.add_argument("--collective-port-base", type=int, required=True)
    ap.add_argument("--transport-port-base", type=int, required=True)
    ap.add_argument("--attempt", required=True, help="hex attempt id for the collective identifier")
    ap.add_argument("--sequences", type=int, required=True)
    ap.add_argument("--max-positions", type=int, required=True)
    ap.add_argument("--kv-logical-pages", type=int, required=True)
    ap.add_argument("--kv-physical-pages", type=int, required=True)
    ap.add_argument("--draft-count", type=int, required=True)
    ap.add_argument("--eos-token-ids", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tokenizer-vocabulary-size", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    hosts = args.hosts.split(",")
    if len(hosts) < 2 or len(set(hosts)) != len(hosts):
        raise SystemExit("--hosts must name at least two distinct nodes")
    if "{host}" not in args.runtime_root:
        raise SystemExit("--runtime-root must contain {host}")
    if args.kv_logical_pages < args.sequences or args.kv_physical_pages < args.sequences:
        raise SystemExit("kv page capacities must cover the resident sequences")
    if not 1 <= args.draft_count <= 8:
        raise SystemExit("--draft-count must be 1..8")
    if args.max_positions % KV_BLOCK_TOKENS != 0:
        raise SystemExit(f"--max-positions must be a multiple of {KV_BLOCK_TOKENS}")
    tp1 = load("qwen38_27b_tp1_deployment")
    lane = load("qwen38_27b_lane_deployment")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "model_resident.nospec.json": deployment(tp1, args, hosts, "config/stage.json"),
        "model_resident.dflash2.json": deployment(tp1, args, hosts, "config/stage.dflash2.json"),
    }
    for rank in range(len(hosts)):
        files[f"stage.rank{rank}.json"] = stages(tp1, lane, args, hosts, rank, None)
        files[f"stage.dflash2.rank{rank}.json"] = stages(tp1, lane, args, hosts, rank, args.draft_count)
    api = deployment(tp1, args, hosts, "config/stage.json")
    api["tokenizer"] = {"path": "tokenizer/tokenizer.json",
                        "sha256": hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest(),
                        "vocabulary_size": args.tokenizer_vocabulary_size}
    files["api.model_resident.json"] = api
    for name, document in files.items():
        (out / name).write_text(json.dumps(document, indent=1) + "\n")
    print(json.dumps({"written": sorted(files), "ranks": len(hosts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
