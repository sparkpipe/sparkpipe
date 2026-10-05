#!/usr/bin/env python3
"""Write the Qwen3.8-27B TP1 serving deployment (one node, attached to the node's weightd).

Emits into --output:
  model_resident.nospec.json   residentd deployment, adapter config config/stage.json
  model_resident.dflash2.json  residentd deployment, adapter config config/stage.dflash2.json
  stage.json                   adapter config without speculative_draft_count
  stage.dflash2.json           adapter config with speculative_draft_count (DFlash2 block)
  api.model_resident.json      API copy: the nospec deployment plus eos_token_ids and tokenizer

Every value is a required argument; nothing is defaulted. Every deployment
carries the model's eos_token_ids (invariant I49: common generation refuses a
deployment without EOS), so fixed-budget benchmarks report the tokens produced
before a generated EOS. The API copy also carries the tokenizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODEL_REVISION = "bf16-h5120-l64-gdn48-full16-v248320-mtp1-v1"
NODE_TARGET = "cuda.sm121.qwen38_27b.resident_decode_stage.bf16"
ADAPTER_SCHEMA_VERSION = 3
KV_BLOCK_TOKENS = 64


def deployment(args, adapter_config: str) -> dict:
    return {
        "schema_version": 2,
        "eos_token_ids": [int(t) for t in args.eos_token_ids.split(",")],
        "coordinator_rank_index": 0,
        "adapter": {"shared_object_path": "lib/model_serving_adapter.so"},
        "driver": {"shared_object_path": "stages/stage_000/model_driver.so",
                   "program_name": "resident_decode"},
        "transport": {"shared_object_path": "lib/hidden_transport.so", "mode": "host-rdma",
                      "control_port_base": args.transport_port_base},
        "weightd": {"socket_path": args.weightd_socket},
        "runtime_limits": {
            "max_inflight_submissions": 2,
            "max_active_sequences": args.sequences,
            "max_input_rows": 128,
            "resident_sequence_capacity": args.sequences,
            "kv_logical_page_capacity": args.kv_logical_pages,
            "kv_physical_page_capacity": args.kv_physical_pages,
        },
        "nodes": [{
            "rank_index": 0,
            "stage_index": 0,
            "runtime_root": args.runtime_root,
            "node_target": NODE_TARGET,
            "transport_host": args.host,
            "adapter_configuration_path": adapter_config,
            "kv_backing_directory": f"/home/{args.host}/kvcache/qwen38_27b.tp1",
            "kv_backing_maximum_bytes": 8589934592,
            "control_endpoint": {"kind": "tcp", "host": args.host, "port": args.control_port},
        }],
    }


def stage(args, draft_count: int | None) -> dict:
    config = {"schema_version": ADAPTER_SCHEMA_VERSION, "model_revision": MODEL_REVISION,
              "stage_pack_path": args.stage_pack, "max_sequence_positions": args.max_positions}
    if draft_count is not None:
        config["speculative_draft_count"] = draft_count
    return config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--runtime-root", required=True)
    ap.add_argument("--stage-pack", required=True, help="runtime-root-relative pack path")
    ap.add_argument("--weightd-socket", required=True)
    ap.add_argument("--control-port", type=int, required=True)
    ap.add_argument("--transport-port-base", type=int, required=True)
    ap.add_argument("--sequences", type=int, required=True)
    ap.add_argument("--max-positions", type=int, required=True)
    ap.add_argument("--kv-logical-pages", type=int, required=True)
    ap.add_argument("--kv-physical-pages", type=int, required=True)
    ap.add_argument("--draft-count", type=int, required=True)
    ap.add_argument("--eos-token-ids", required=True, help="comma list of model EOS token ids")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json staged under the API runtime root")
    ap.add_argument("--tokenizer-vocabulary-size", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.stage_pack.startswith("/") or ".." in Path(args.stage_pack).parts:
        raise SystemExit("--stage-pack must be runtime-root-relative")
    if args.kv_logical_pages < args.sequences or args.kv_physical_pages < args.sequences:
        raise SystemExit("kv page capacities must cover the resident sequences")
    if not 1 <= args.draft_count <= 8:
        raise SystemExit("--draft-count must be 1..8")
    if args.max_positions % KV_BLOCK_TOKENS != 0:
        raise SystemExit(f"--max-positions must be a multiple of {KV_BLOCK_TOKENS}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "model_resident.nospec.json": deployment(args, "config/stage.json"),
        "model_resident.dflash2.json": deployment(args, "config/stage.dflash2.json"),
        "stage.json": stage(args, None),
        "stage.dflash2.json": stage(args, args.draft_count),
    }
    api = deployment(args, "config/stage.json")
    api["tokenizer"] = {"path": "tokenizer/tokenizer.json",
                        "sha256": hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest(),
                        "vocabulary_size": args.tokenizer_vocabulary_size}
    files["api.model_resident.json"] = api
    for name, document in files.items():
        (out / name).write_text(json.dumps(document, indent=1) + "\n")
    kv_blocks = args.sequences * (args.max_positions // KV_BLOCK_TOKENS)
    print(json.dumps({"written": sorted(files), "kv_blocks": kv_blocks,
                      "kv_gib_with_mtp_cache": round(kv_blocks * KV_BLOCK_TOKENS * 4 * 256 * 2 * 17 * 2 / 2**30, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
