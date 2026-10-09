#!/usr/bin/env python3
"""Render a TP16 stage-runner lane: deployment.json and one adapter.<host>.json per rank.

Any model that runs on the common stage runner and serving adapter deploys
through this one renderer; a model is one MODELS entry (rank pack path,
adapter library, program name, contract, chat template, tokenizer
vocabulary, KV and recurrent bytes per rank). Rank r runs on spark<hex r>.

Ports for lane L: control 23000+16L+r, transport 64000+16L (+15 listen),
collective identifier L<<48, mesh session ports 18432 + 16a + b (one model
is active at a time, so lanes share the session block).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD = 16
HEX = "0123456789abcdef"
HOSTS = [f"spark{HEX[i]}" for i in range(WORLD)]
SESSION_BLOCK_BASE = 18432
MAX_ROWS = 2048
DEFAULT_STATE_BUDGET_BYTES = 5 << 30
DEFAULT_KV_SNAPSHOT_BYTES = 8 << 30

MODELS = {
    "qwen38_27b": {
        "pack": "sparkdata/qwen38_27b.fp8.tp16/packs/qwen38_27b.tp16.rank{rank:02d}.pack",
        "adapter": "libqwen38_27b_tp16_serving_adapter.so",
        "program": "qwen38_27b",
        "node_target": "cuda.sm121.qwen38_27b.stage_runner.linear_fp8.kv_bf16",
        "contract": "model_contracts/qwen38_27b_authoritative.json",
        "chat_template": "model-families/qwen38_27b/chat_template.json",
        "eos_token_ids": [248044, 248046],
        "tokenizer_vocabulary": 248077,
        "hidden": 5120,
        "kv_page_bytes": 16 * 64 * 4096 // WORLD,
        "recurrent_bytes": 48 * (3 * 128 * 128 * 4 + 640 * 4 * 2),
    },
}


def chat_template(model: dict) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_model_resident_deployment import chat_template_value
    return chat_template_value(json.loads((ROOT / model["chat_template"]).read_text()))


def session_table() -> list[list[int]]:
    return [[0 if a == b else SESSION_BLOCK_BASE + a * WORLD + b for b in range(WORLD)] for a in range(WORLD)]


def adapter_config(model: dict, lane: int, rank: int, args) -> dict:
    host = HOSTS[rank]
    return {
        "stage_pack_path": f"/home/{host}/" + model["pack"].format(rank=rank),
        "tp_degree": WORLD,
        "tp_rank": rank,
        "world_size": WORLD,
        "max_sequences": args.sequences,
        "max_rows": args.rows,
        "resident_capacity": args.sequences,
        "kv_pages": args.kv_pages,
        "state_budget_bytes": args.state_budget_bytes,
        "hidden": model["hidden"],
        "device_collective": {
            "backend": "hidden_transport",
            "backend_module_path": "lib/hidden_transport.so",
            "local_host": host,
            "collective_identifier": lane << 48,
            "listen_port": 64000 + 16 * lane + 15,
            "connect_timeout_milli": 300000,
            "operation_timeout_milli": 30000,
            "peer_hosts": HOSTS,
            "session_ports": session_table(),
            "wait_mode": "hardware",
        },
    }


def deployment(model: dict, lane: int, args) -> dict:
    nodes = []
    for rank, host in enumerate(HOSTS):
        root = args.runtime_root.format(host=host)
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": root,
            "node_target": model["node_target"],
            "transport_host": host,
            "adapter_configuration_path": "config/adapter.json",
            "kv_backing_directory": os.path.join(root, "kvcache"),
            "kv_partition": "/",
            "kv_backing_maximum_bytes": args.kv_backing_bytes,
            "kv_snapshot_directory": os.path.join(root, "kvsnapshot"),
            "kv_snapshot_maximum_bytes": args.kv_snapshot_bytes,
            "control_endpoint": {"kind": "tcp", "host": host, "port": 23000 + 16 * lane + rank},
        })
    result = {
        "schema_version": 2,
        "eos_token_ids": model["eos_token_ids"],
        "coordinator_rank_index": 0,
        "adapter": {"shared_object_path": "lib/" + model["adapter"]},
        "driver": {"shared_object_path": "lib/" + model["adapter"], "program_name": model["program"]},
        "transport": {"shared_object_path": "lib/hidden_transport.so", "mode": "host-rdma",
                      "control_port_base": 64000 + 16 * lane},
        "weightd": {"socket_path": args.weightd_socket},
        "runtime_limits": {
            "max_inflight_submissions": args.sequences,
            "max_active_sequences": args.sequences,
            "max_input_rows": args.rows,
            "resident_sequence_capacity": args.sequences,
            "kv_logical_page_capacity": args.kv_logical_pages,
            "kv_physical_page_capacity": args.kv_physical_pages,
        },
        "chat_template": chat_template(model),
        "nodes": nodes,
    }
    if args.tokenizer_sha256:
        result["tokenizer"] = {"path": "tokenizer/tokenizer.compiled", "sha256": args.tokenizer_sha256,
                               "vocabulary_size": model["tokenizer_vocabulary"]}
    return result


def render(value: dict) -> str:
    return json.dumps(value, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--runtime-root", required=True, help="per-node root, {host} is substituted")
    parser.add_argument("--weightd-socket", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--kv-pages", type=int, default=64)
    parser.add_argument("--kv-physical-pages", type=int, default=None)
    parser.add_argument("--kv-logical-pages", type=int, default=None)
    parser.add_argument("--kv-backing-bytes", type=int, default=8 << 30)
    parser.add_argument("--kv-snapshot-bytes", type=int, default=DEFAULT_KV_SNAPSHOT_BYTES)
    parser.add_argument("--state-budget-bytes", type=int, default=DEFAULT_STATE_BUDGET_BYTES)
    parser.add_argument("--tokenizer-sha256", default=None)
    args = parser.parse_args()
    model = MODELS[args.model]
    if not 1 <= args.lane <= 15:
        raise SystemExit("lane must be within 1..15")
    if not 1 <= args.sequences <= 16 or not args.sequences <= args.rows <= MAX_ROWS:
        raise SystemExit(f"sequences must be within 1..16 and rows within sequences..{MAX_ROWS}")
    args.kv_physical_pages = args.kv_physical_pages or args.sequences * args.kv_pages
    args.kv_logical_pages = args.kv_logical_pages or args.kv_physical_pages
    if args.kv_physical_pages < args.kv_pages or args.kv_logical_pages < args.kv_physical_pages:
        raise SystemExit("physical pages must hold one sequence and logical pages must cover the physical pages")
    minimum = (args.kv_logical_pages - args.kv_physical_pages + 2) * model["kv_page_bytes"] + \
        (2 * args.sequences + 2) * model["recurrent_bytes"]
    if args.kv_backing_bytes < minimum:
        raise SystemExit(f"kv-backing-bytes {args.kv_backing_bytes} cannot hold the spilled pages and recurrent checkpoints; it needs {minimum}")
    if args.state_budget_bytes <= 0:
        raise SystemExit("state-budget-bytes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "deployment.json").write_text(render(deployment(model, args.lane, args)))
    for rank, host in enumerate(HOSTS):
        (args.output_dir / f"adapter.{host}.json").write_text(render(adapter_config(model, args.lane, rank, args)))
    print(f"{args.model} lane {args.lane} rendered under {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
