#!/usr/bin/env python3
"""Render a TP16 stage-runner lane: deployment.json and one adapter.<host>.json per rank.

Any model that runs on the common stage runner and serving adapter deploys
through this one renderer; a model is one MODELS entry (rank pack path,
adapter library, program name, contract, chat template, tokenizer
vocabulary, KV and recurrent bytes per rank). Rank r runs on spark<hex r>.

Ports for lane L: control 23000+16L+r, transport 64000+16L (+15 listen),
collective identifier L<<48, mesh session ports 18432 + 16a + b (one model
is active at a time, so lanes share the session block).

--swap-entry also writes swap_entry.json: the model's entry for the hub swap
controller (tools/serving/model_swap.py). Its node commands call the lane.sh
that tools/stage_lane.sh setup writes into every rank root, and its API runs
on the hub from ~/<model>-kv-api (tools/stage_lane.sh api-setup).
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
MAX_SEQUENCES = 128
DEFAULT_STATE_BUDGET_BYTES = 5 << 30
DEFAULT_KV_SNAPSHOT_BYTES = 8 << 30

QWEN38_27B = {
    "swap_id": "qwen3.8-27b",
    "api_port": 8461,
    "kv_label": "qwen38_27b_stage",
    "node_memory_bytes": 16 << 30,
    "tokenizer_json": "/mnt/cold-raid6/models/qwen3.8-27b-fp8/tokenizer.json",
    "pack": "sparkdata/qwen38_27b.fp8.tp16/packs/qwen38_27b.tp16.rank{rank:02d}.pack",
    "adapter": "libqwen38_27b_tp16_serving_adapter.so",
    "program": "qwen38_27b",
    "node_target": "cuda.sm121.qwen38_27b.stage_runner.linear_fp8.kv_bf16",
    "contract": "model_contracts/qwen38_27b_authoritative.json",
    "chat_template": "model-families/qwen38/chat_template.json",
    "eos_token_ids": [248044, 248046],
    "tokenizer_vocabulary": 248077,
    "hidden": 5120,
    "kv_page_bytes": 16 * 64 * 4096 // WORLD,
    "recurrent_bytes": 48 * (3 * 128 * 128 * 4 + 640 * 4 * 2),
}

DSV41_FLASH = {
    "swap_id": "deepseek-v4.1-flash",
    "api_port": 8463,
    "kv_label": "dsv41_flash_stage",
    "node_memory_bytes": 44 << 30,
    "tokenizer_json": "/mnt/cold-raid6/models/deepseek-v4.1-flash/tokenizer.json",
    "pack": "sparkdata/dsv41flash.mxfp4.tp16/packs/rank{rank}.spstage",
    "dense": True,
    "extra_packs": {
        "engram.spengram": "sparkdata/dsv41flash.mxfp4.tp8pp2/engram/engram.rank{rank}.spengram",
        "engram_tables.bin": "sparkdata/dsv41flash.mxfp4.tp16/engram_tables.bin",
    },
    "adapter": "libdsv41_flash_tp16_serving_adapter.so",
    "program": "dsv41_flash",
    "node_target": "cuda.sm121.dsv41_flash.stage_runner.linear_fp8.expert_mxfp4.kv_bf16",
    "contract": "model_contracts/dsv41_flash_authoritative.json",
    "chat_template": "model-families/dsv41_flash/chat_template.json",
    "eos_token_ids": [1],
    "tokenizer_vocabulary": 129280,
    "hidden": 5120,
    "kv_page_bytes": (3 * 128 + 256) * 1280 // WORLD,
    "recurrent_bytes": (40 * 8 * 512 * 2 + 3 * 2 * 512 * 4 + 16 + 255) // 256 * 256,
}

QWEN38_MAX = {
    **QWEN38_27B,
    "swap_id": "qwen3.8-max",
    "api_port": 8464,
    "kv_label": "qwen38_max_stage",
    "node_memory_bytes": 88 << 30,
    "tokenizer_json": "/mnt/cold-raid6/models/qwen3.8-max-nvfp4-radixark-bf16-spine/tokenizer.json",
    "pack": "sparkdata/qwen38_max.nvfp4.tp16/qwen38_max.tp16.rank{rank:02d}.pack",
    "adapter": "libqwen38_max_tp16_serving_adapter.so",
    "program": "qwen38_max",
    "node_target": "cuda.sm121.qwen38_max.stage_runner.linear_bf16.expert_nvfp4.kv_bf16",
    "contract": "model_contracts/qwen38_authoritative.json",
    "hidden": 8192,
    "kv_page_bytes": 23 * 64 * 4096 // WORLD,
    "recurrent_bytes": 69 * (8 * 128 * 128 * 4 + 1280 * 4 * 2),
}

QWEN38_FLASH = {
    **QWEN38_27B,
    "swap_id": "qwen3.8-flash",
    "api_port": 8465,
    "kv_label": "qwen38_flash_stage",
    "node_memory_bytes": 14 << 30,
    "tokenizer_json": "/mnt/model-warm/qwen3.8-flash-next-nvfp4-nvidia/tokenizer.json",
    "pack": "sparkdata/qwen38_flash.nvfp4.tp16/qwen38_flash.tp16.rank{rank:02d}.pack",
    "adapter": "libqwen38_flash_tp16_serving_adapter.so",
    "program": "qwen38_flash",
    "node_target": "cuda.sm121.qwen38_flash.stage_runner.linear_bf16.expert_nvfp4.kv_bf16",
    "contract": "model_contracts/qwen4_flash_authoritative.json",
    "hidden": 2560,
    "kv_page_bytes": (12 * 64 * 2048 + 12 * 16 * 256) // WORLD,
    "recurrent_bytes": 36 * (3 * 128 * 128 * 4 + 640 * 4 * 2) + 10240 * 10 * 2 + 4 * 4 + 12 * 3 * 128 * 2,
}
GEMMA4_31B = {
    "swap_id": "gemma-4-31b",
    "api_port": 8466,
    "kv_label": "gemma4_31b_stage",
    "node_memory_bytes": 16 << 30,
    "tokenizer_json": "/mnt/model-warm/gemma-4-31b-it/tokenizer.json",
    "pack": "sparkdata/gemma4_31b.bf16.tp16/gemma4.tp16.rank{rank:02d}.pack",
    "adapter": "libgemma4_31b_tp16_serving_adapter.so",
    "program": "gemma4_31b",
    "node_target": "cuda.sm121.gemma4_31b.stage_runner.linear_bf16.kv_bf16",
    "contract": "model_contracts/gemma4_31b_authoritative.json",
    "chat_template": "model-families/gemma4/chat_template.json",
    "eos_token_ids": [1, 106, 50],
    "tokenizer_vocabulary": 262144,
    "hidden": 5376,
    "kv_page_bytes": (50 * 64 * 16384 + 10 * 64 * 8192) // WORLD,
    "recurrent_bytes": 0,
}
GEMMA4_26B = {
    **GEMMA4_31B,
    "swap_id": "gemma-4-26b-a4b",
    "api_port": 8467,
    "kv_label": "gemma4_26b_stage",
    "node_memory_bytes": 12 << 30,
    "tokenizer_json": "/mnt/model-warm/gemma-4-26b-a4b-it/tokenizer.json",
    "pack": "sparkdata/gemma4_26b.bf16.tp16/gemma4_26b.tp16.rank{rank:02d}.pack",
    "adapter": "libgemma4_26b_tp16_serving_adapter.so",
    "program": "gemma4_26b",
    "node_target": "cuda.sm121.gemma4_26b.stage_runner.linear_bf16.expert_bf16.kv_bf16",
    "contract": "model_contracts/gemma4_26b_a4b_authoritative.json",
    "hidden": 2816,
    "kv_page_bytes": (25 * 64 * 8192 + 5 * 64 * 4096) // WORLD,
}
LING3_FLASH = {
    "swap_id": "ling-3.0-flash",
    "api_port": 8468,
    "kv_label": "ling3_stage",
    "node_memory_bytes": 24 << 30,
    "tokenizer_json": "/mnt/model-warm/ling-3.0-flash/tokenizer.json",
    "pack": "sparkdata/ling3_flash.bf16.tp16/ling3_flash.tp16.rank{rank:02d}.pack",
    "adapter": "libling3_flash_tp16_serving_adapter.so",
    "program": "ling3_flash",
    "node_target": "cuda.sm121.ling3_flash.stage_runner.linear_bf16.expert_bf16.kv_bf16",
    "contract": "model_contracts/ling_authoritative.json",
    "chat_template": "model-families/ling/chat_template.json",
    "eos_token_ids": [156895, 156892],
    "tokenizer_vocabulary": 157153,
    "hidden": 2560,
    "kv_page_bytes": 7 * 64 * 1152 // WORLD,
    "recurrent_bytes": 35 * (2 * 128 * 128 * 4 + 768 * 4 * 2),
}

LING3_FIN = {
    **LING3_FLASH,
    "swap_id": "ling-3.0-flash-fin",
    "api_port": 8469,
    "tokenizer_json": "/mnt/model-warm/ling-3.0-flash-fin/tokenizer.json",
    "pack": "sparkdata/ling3_fin.bf16.tp16/ling3_fin.tp16.rank{rank:02d}.pack",
    "adapter": "libling3_fin_tp16_serving_adapter.so",
    "program": "ling3_fin",
    "node_target": "cuda.sm121.ling3_fin.stage_runner.linear_bf16.expert_bf16.kv_bf16",
    "contract": "model_contracts/lingfin_authoritative.json",
}
HY4 = {
    "swap_id": "hy4-preview",
    "api_port": 8470,
    "kv_label": "hy4_tp16_stage",
    "node_memory_bytes": 64 << 30,
    "tokenizer_json": "/mnt/model-warm/hy4-preview-fp8-official/tokenizer.json",
    "pack": "sparkdata/hy4.mxfp8.tp16/hy4.tp16.rank{rank:02d}.pack",
    "adapter": "libhy4_tp16_serving_adapter.so",
    "program": "hy4",
    "node_target": "cuda.sm121.hy4.stage_runner.linear_mxfp8.expert_mxfp8.kv_bf16",
    "contract": "model_contracts/hy4_authoritative.json",
    "chat_template": "model-families/hy4/chat_template.json",
    "eos_token_ids": [120025],
    "tokenizer_vocabulary": 120832,
    "hidden": 6144,
    "kv_page_bytes": (78 * 64 * 1152 + 21 * 64 * 256) // WORLD,
    "recurrent_bytes": 0,
}
MUSE_GLIMMER = {
    "swap_id": "muse-glimmer-30b",
    "api_port": 8471,
    "kv_label": "muse_glimmer_stage",
    "node_memory_bytes": 16 << 30,
    "tokenizer_json": "/mnt/model-warm/muse-glimmer-30b/tokenizer.json",
    "pack": "sparkdata/muse_glimmer.bf16.tp16/muse.tp16.rank{rank:02d}.pack",
    "adapter": "libmuse_glimmer_tp16_serving_adapter.so",
    "program": "muse_glimmer",
    "node_target": "cuda.sm121.muse_glimmer.stage_runner.linear_bf16.kv_bf16",
    "contract": "model_contracts/muse_glimmer_authoritative.json",
    "chat_template": "model-families/muse_glimmer/chat_template.json",
    "eos_token_ids": [200001, 200008],
    "tokenizer_vocabulary": 202048,
    "hidden": 6656,
    "kv_page_bytes": (52 * 64 * 1024) // WORLD,
    "recurrent_bytes": 0,
}
GLM53_FLASH = {
    "swap_id": "glm-5.3-flash",
    "api_port": 8476,
    "kv_label": "glm53_flash_stage",
    "node_memory_bytes": 25 << 30,
    "tokenizer_json": "/mnt/cold-raid6/models/glm-5.3-flash-bf16-official/tokenizer.json",
    "pack": "sparkdata/glm53flash.fp8.tp16/packs/glm53flash.fp8.tp16.rank{rank:x}.sp",
    "adapter": "libglm53_flash_tp16_serving_adapter.so",
    "program": "glm53_flash",
    "node_target": "cuda.sm121.glm53_flash.stage_runner.linear_bf16.expert_fp8.kv_bf16",
    "contract": "model_contracts/glm53_flash_authoritative.json",
    "chat_template": "model-families/glm5_next/chat_template.json",
    "eos_token_ids": [154820, 154827, 154829],
    "tokenizer_vocabulary": 154856,
    "hidden": 4096,
    "kv_page_bytes": (11 * 64 * 1024 + 11 * 64 * 514) // WORLD,
    "recurrent_bytes": 34 * (4 * 128 * 128 * 4 + 3 * 512 * 4 * 2),
}
LAGUNA = {
    "swap_id": "laguna-s-2.1",
    "api_port": 8472,
    "kv_label": "laguna_stage",
    "node_memory_bytes": 24 << 30,
    "tokenizer_json": "/mnt/cold-raid6/models/laguna-s-2.1/tokenizer.json",
    "pack": "sparkdata/laguna.bf16.tp16/laguna.tp16.rank{rank:02d}.pack",
    "adapter": "liblaguna_tp16_serving_adapter.so",
    "program": "laguna",
    "node_target": "cuda.sm121.laguna.stage_runner.linear_bf16.expert_bf16.kv_bf16",
    "contract": "model_contracts/laguna_authoritative.json",
    "chat_template": "model-families/laguna/chat_template.json",
    "eos_token_ids": [2, 24],
    "tokenizer_vocabulary": 100352,
    "hidden": 3072,
    "kv_page_bytes": (48 * 64 * 4096) // WORLD,
    "recurrent_bytes": 0,
}
GLM53_FULL = {
    "swap_id": "glm-5.3",
    "api_port": 8475,
    "kv_label": "glm53_full_tp16_stage",
    "node_memory_bytes": 57 << 30,
    "tokenizer_json": "/mnt/cold-raid6/models/glm-5.3-fp8/tokenizer.json",
    "pack": "sparkdata/glm53full.fp8.tp16/packs/glm53full.fp8.tp16-rank{rank}.glm52sp",
    "adapter": "libglm53_full_tp16_serving_adapter.so",
    "program": "glm53_full",
    "node_target": "cuda.sm121.glm53_full.stage_runner.linear_bf16.expert_fp8.kv_bf16",
    "contract": "model_contracts/glm52.json",
    "chat_template": "model-families/glm52/chat_template.json",
    "eos_token_ids": [154820, 154827, 154829],
    "tokenizer_vocabulary": 154856,
    "hidden": 6144,
    "kv_page_bytes": (78 * 64 * 1152 + 21 * 64 * 256) // WORLD,
    "recurrent_bytes": 0,
}
MODELS = {
    "dsv41_flash": DSV41_FLASH,
    "gemma4_26b": GEMMA4_26B,
    "gemma4_31b": GEMMA4_31B,
    "glm53_flash": GLM53_FLASH,
    "glm53_full": GLM53_FULL,
    "hy4": HY4,
    "laguna": LAGUNA,
    "ling3_fin": LING3_FIN,
    "muse_glimmer": MUSE_GLIMMER,
    "ling3_flash": LING3_FLASH,
    "qwen38_flash": QWEN38_FLASH,
    "qwen38_max": QWEN38_MAX,
    "qwen38_27b": QWEN38_27B,
    "qwen38_27b_bf16": {
        **QWEN38_27B,
        "swap_id": "qwen3.8-27b-bf16",
        "api_port": 8462,
        "node_memory_bytes": 20 << 30,
        "pack": "sparkdata/qwen38_27b.bf16.tp16/packs/qwen38_27b.tp16.rank{rank:02d}.pack",
        "adapter": "libqwen38_27b_bf16_tp16_serving_adapter.so",
        "node_target": "cuda.sm121.qwen38_27b.stage_runner.linear_bf16.kv_bf16",
    },
}


def chat_template(model: dict) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_model_resident_deployment import chat_template_value
    return chat_template_value(json.loads((ROOT / model["chat_template"]).read_text()))


def session_table() -> list[list[int]]:
    return [[0 if a == b else SESSION_BLOCK_BASE + a * WORLD + b for b in range(WORLD)] for a in range(WORLD)]


def rank_pack_path(model: dict, lane: int, rank: int, args) -> str:
    host = HOSTS[rank]
    if model.get("dense"):
        return os.path.join(args.runtime_root.format(host=host), "packs", os.path.basename(model["pack"].format(rank=rank)))
    return f"/home/{host}/" + model["pack"].format(rank=rank)


def adapter_config(model: dict, lane: int, rank: int, args) -> dict:
    host = HOSTS[rank]
    return {
        "stage_pack_path": rank_pack_path(model, lane, rank, args),
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
            "max_inflight_submissions": min(args.sequences, 16),
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


def swap_entry(name: str, model: dict, lane: int) -> dict:
    root = f"$HOME/stage-lanes/{name}/lane{lane}/root"
    api = f"$HOME/{name}-kv-api"
    unit = f"kv-{name.replace('_', '-')}-api"
    return {
        "id": model["swap_id"],
        "kv_label": model["kv_label"],
        "node_memory_bytes": model["node_memory_bytes"],
        "node_reclaim": f"{root}/lane.sh reclaim",
        "node_start": f"{root}/lane.sh start {{run_id}}",
        "node_ready": f"{root}/lane.sh ready {{run_id}}",
        "node_stop": f"{root}/lane.sh stop",
        "api_start": (f"systemctl --user reset-failed {unit} 2>/dev/null; systemctl --user is-active -q {unit} || "
                      f"systemd-run --user --unit={unit} -p MemoryMax=2G -p MemorySwapMax=0 --working-directory={api} "
                      f"bash -c 'exec ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root {api}/runtime "
                      f"--port {model['api_port']} >> api.log 2>&1' >/dev/null"),
        "api_stop": f"systemctl --user stop {unit} 2>/dev/null; systemctl --user reset-failed {unit} 2>/dev/null; true",
        "health": f"http://127.0.0.1:{model['api_port']}/health",
    }


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
    parser.add_argument("--swap-entry", action="store_true")
    args = parser.parse_args()
    model = MODELS[args.model]
    if not 0 <= args.lane <= 15:
        raise SystemExit("lane must be within 0..15")
    if not 1 <= args.sequences <= MAX_SEQUENCES or not args.sequences <= args.rows <= MAX_ROWS:
        raise SystemExit(f"sequences must be within 1..{MAX_SEQUENCES} and rows within sequences..{MAX_ROWS}")
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
    if args.swap_entry:
        (args.output_dir / "swap_entry.json").write_text(render(swap_entry(args.model, model, args.lane)))
    print(f"{args.model} lane {args.lane} rendered under {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
