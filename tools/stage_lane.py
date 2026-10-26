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
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANES = Path(os.environ.get("SPARK_LANES") or ROOT / "deployment/lanes.json")
WORLD = 16
HEX = "0123456789abcdef"
HOSTS = [f"spark{HEX[i]}" for i in range(WORLD)]
SESSION_BLOCK_BASE = 18432
MAX_ROWS = 4096
MAX_SEQUENCES = 128
DEFAULT_STATE_BUDGET_BYTES = 5 << 30
DEFAULT_KV_SNAPSHOT_BYTES = 8 << 30
KV_BLOCK_TOKENS = 64

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
    "pack": "sparkdata/dsv41flash-1017.mxfp4.tp16/packs/rank{rank}.spstage",
    "dense": True,
    "extra_packs": {
        "engram.spengram": "sparkdata/dsv41flash-1017.mxfp4.tp16/engram/engram.rank{rank}.spengram",
        "engram_tables.bin": "sparkdata/dsv41flash-1017.mxfp4.tp16/engram_tables.bin",
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
    "kv_block_tokens": 256,
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
    "node_memory_bytes": 24964497408,
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
KEXAONE2 = {
    "swap_id": "k-exaone-2.0",
    "api_port": 8477,
    "kv_label": "kexaone2_stage",
    "node_memory_bytes": 100 << 30,
    "tokenizer_json": "/mnt/cold-raid6/models/k-exaone-2.0-750b-bf16-official/tokenizer.json",
    "pack": "sparkdata/kexaone2.rows.tp16/kexaone2.tp16.rank{rank:02d}.pack",
    "adapter": "libkexaone2_tp16_serving_adapter.so",
    "program": "kexaone2",
    "node_target": "cuda.sm121.kexaone2.stage_runner.linear_bf16.expert_bf16.kv_bf16",
    "contract": "model_contracts/kexaone2_authoritative.json",
    "chat_template": "model-families/kexaone2/chat_template.json",
    "eos_token_ids": [53, 2],
    "tokenizer_vocabulary": 153600,
    "kv_page_bytes": (78 * 64 * 4096) // WORLD,
    "recurrent_bytes": 0,
}
AXK2 = {
    "swap_id": "a.x-k2",
    "api_port": 8478,
    "kv_label": "axk2_tp16_stage",
    "node_memory_bytes": 57 << 30,
    "tokenizer_json": "/mnt/cold-raid6/models/ax-k2-fp8-official/tokenizer.json",
    "pack": "sparkdata/axk2.fp8.tp16/axk2.tp16.rank{rank:02d}.pack",
    "adapter": "libaxk2_tp16_serving_adapter.so",
    "program": "axk2",
    "node_target": "cuda.sm121.axk2.stage_runner.linear_fp8.expert_fp8.kv_bf16",
    "contract": "model_contracts/axk2_authoritative.json",
    "chat_template": "model-families/axk2/chat_template.json",
    "eos_token_ids": [163704],
    "tokenizer_vocabulary": 163840,
    "hidden": 7168,
    "kv_page_bytes": (61 * 64 * 1152 + 61 * 64 * 256) // WORLD,
    "recurrent_bytes": 0,
    "expert_warm": {"model_tag": "axk2_stage", "layers": 61, "experts": 256},
}
SOLAR_OPEN2 = {
    "swap_id": "solar-open-2",
    "api_port": 8479,
    "kv_label": "solar_open2_stage",
    "node_memory_bytes": 42 << 30,
    "tokenizer_json": "/mnt/cold-raid6/models/solar-open2-250b-bf16-official/tokenizer.json",
    "pack": "sparkdata/solar_open2.tp16/solar_open2.tp16.rank{rank:02d}.pack",
    "adapter": "libsolar_open2_tp16_serving_adapter.so",
    "program": "solar_open2",
    "node_target": "cuda.sm121.solar_open2.stage_runner.linear_bf16.expert_bf16.kv_bf16",
    "contract": "model_contracts/solar_open2_authoritative.json",
    "chat_template": "model-families/solar_open2/chat_template.json",
    "eos_token_ids": [2, 129],
    "tokenizer_vocabulary": 196608,
    "hidden": 4096,
    "kv_page_bytes": (12 * 64 * 4096) // WORLD,
    "recurrent_bytes": 36 * (4 * 128 * 128 * 4 + 1536 * 4 * 2),
}
GLM53_FULL = {
    "swap_id": "glm-5.3",
    "api_port": 8475,
    "kv_label": "glm53_full_tp16_stage",
    "node_memory_bytes": 57 << 30,
    "tokenizer_json": "/mnt/cold-raid6/models/glm-5.3-fp8/tokenizer.json",
    "pack": "sparkdata/glm53full.fp8lin.tp16/packs/glm53full.fp8lin.tp16-rank{rank}.glm52sp",
    "adapter": "libglm53_full_tp16_serving_adapter.so",
    "program": "glm53_full",
    "node_target": "cuda.sm121.glm53_full.stage_runner.linear_fp8.expert_fp8.kv_bf16",
    "contract": "model_contracts/glm52.json",
    "chat_template": "model-families/glm52/chat_template.json",
    "eos_token_ids": [154820, 154827, 154829],
    "tokenizer_vocabulary": 154856,
    "hidden": 6144,
    "kv_page_bytes": (78 * 64 * 1152 + 21 * 64 * 256) // WORLD,
    "recurrent_bytes": 0,
}
MIMO26_FLASH = {
    "swap_id": "mimo-v2.6-flash",
    "api_port": 8473,
    "kv_label": "mimo26_flash_tp16_stage",
    "node_memory_bytes": 21267628032,
    "tokenizer_json": "/mnt/cold-raid6/models/mimo-v2.6-flash-rl/tokenizer.json",
    "pack": "sparkdata/mimo26_flash.fp8_mxfp4.tp16/mimo26_flash.tp16.rank{rank:02d}.pack",
    "adapter": "libmimo26_flash_tp16_serving_adapter.so",
    "program": "mimo26_flash",
    "node_target": "cuda.sm121.mimo26_flash.stage_runner.linear_fp8.expert_mxfp4.kv_bf16",
    "contract": "model_contracts/mimo26_flash_authoritative.json",
    "chat_template": "model-families/mimo26/chat_template.json",
    "eos_token_ids": [151643, 151645, 151672],
    "tokenizer_vocabulary": 151675,
    "hidden": 4096,
    "kv_page_bytes": (39 * 64 * 5120 + 9 * 64 * 2560) // WORLD,
    "recurrent_bytes": 0,
}
MIMO26_PRO = {
    **MIMO26_FLASH,
    "swap_id": "mimo-v2.6-pro",
    "api_port": 8474,
    "kv_label": "mimo26_pro_tp16_stage",
    "node_memory_bytes": 46406873088,
    "tokenizer_json": "/mnt/cold-raid6/models/mimo-v2.6-pro-rl/tokenizer.json",
    "pack": "sparkdata/mimo26_pro.fp8_mxfp4.tp16/mimo26_pro.tp16.rank{rank:02d}.pack",
    "adapter": "libmimo26_pro_tp16_serving_adapter.so",
    "program": "mimo26_pro",
    "node_target": "cuda.sm121.mimo26_pro.stage_runner.linear_fp8.expert_mxfp4.kv_bf16",
    "contract": "model_contracts/mimo26_pro_authoritative.json",
    "hidden": 6144,
    "kv_page_bytes": (60 * 64 * 5120 + 10 * 64 * 5120) // WORLD,
}
K3 = {
    "swap_id": "kimi-k3",
    "api_port": 8459,
    "kv_label": "k3_stage",
    "node_memory_bytes": 104689827840,
    "tokenizer_json": "/mnt/cold-raid6/models/kimi-k3/tiktoken.model",
    "pack": "sparkdata/k3.mxfp4.tp16/packs/k3.stage0.rank{rank:02d}.pack",
    "adapter": "libk3_serving_adapter.so",
    "firmware_adapter": "libk3_tp16_serving_adapter.so",
    "program": "k3",
    "node_target": "cuda.sm121.k3.resident_decode_stage.linear_bf16.expert_mxfp4.kv_bf16",
    "contract": "model_contracts/k3_authoritative.json",
    "chat_template": "model-families/k3/chat_template.json",
    "eos_token_ids": [163585, 163586],
    "tokenizer_vocabulary": 163840,
    "hidden": 7168,
    "kv_page_bytes": 24 * 64 * (512 + 64) * 2 // WORLD,
    "recurrent_bytes": 69 * (6 * 128 * 128 * 4 + 6 * (2 * 128 + 128) * 4 * 2),
}
MODELS = {
    "dsv41_flash": DSV41_FLASH,
    "gemma4_26b": GEMMA4_26B,
    "gemma4_31b": GEMMA4_31B,
    "glm53_flash": GLM53_FLASH,
    "glm53_full": GLM53_FULL,
    "hy4": HY4,
    "k3": K3,
    "laguna": LAGUNA,
    "kexaone2": KEXAONE2,
    "axk2": AXK2,
    "solar_open2": SOLAR_OPEN2,
    "ling3_fin": LING3_FIN,
    "mimo26_flash": MIMO26_FLASH,
    "mimo26_pro": MIMO26_PRO,
    "muse_glimmer": MUSE_GLIMMER,
    "ling3_flash": LING3_FLASH,
    "qwen38_flash": QWEN38_FLASH,
    "qwen38_max": QWEN38_MAX,
    "qwen38_27b": QWEN38_27B,
}


def chat_template(model: dict) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_model_resident_deployment import chat_template_value
    return chat_template_value(json.loads((ROOT / model["chat_template"]).read_text()))


def text_prompt_prefix(directory: Path) -> list[int]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tokenizer_text_prefix import TokenizerPrefixError, directory_prefix
    try:
        return directory_prefix(directory)
    except (TokenizerPrefixError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"raw text prompt prefix of {directory}: {error}")


def session_table() -> list[list[int]]:
    return [[0 if a == b else SESSION_BLOCK_BASE + a * WORLD + b for b in range(WORLD)] for a in range(WORLD)]


def rank_pack_path(model: dict, lane: int, rank: int, args) -> str:
    host = HOSTS[rank]
    if model.get("dense"):
        return os.path.join(args.runtime_root.format(host=host), "packs", os.path.basename(model["pack"].format(rank=rank)))
    return f"/home/{host}/" + model["pack"].format(rank=rank)


def model_hidden(model: dict) -> int:
    if "hidden" in model:
        return model["hidden"]
    return json.loads((ROOT / model["contract"]).read_text())["source"]["config_json"]["hidden_size"]


def adapter_config(model: dict, lane: int, rank: int, args) -> dict:
    host = HOSTS[rank]
    return {
        "stage_pack_path": rank_pack_path(model, lane, rank, args),
        "tp_degree": WORLD,
        "tp_rank": rank,
        "world_size": WORLD,
        "max_sequences": args.sequences,
        "max_rows": args.rows,
        **({"max_shared_rows": args.shared_rows} if args.shared_rows else {}),
        "resident_capacity": args.sequences,
        "kv_pages": args.kv_pages,
        "state_budget_bytes": args.state_budget_bytes,
        **({"state_memory_bytes": args.state_memory_bytes} if model["recurrent_bytes"] else {}),
        "hidden": model_hidden(model),
        **({"taps": args.taps} if args.taps else {}),
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
            **({"max_shared_input_rows": args.shared_rows} if args.shared_rows else {}),
            "resident_sequence_capacity": args.sequences,
            "kv_logical_page_capacity": args.kv_logical_pages,
            "kv_physical_page_capacity": args.kv_physical_pages,
        },
        "chat_template": chat_template(model),
        "nodes": nodes,
    }
    if args.tokenizer_sha256 and not args.record:
        result["tokenizer"] = {"path": "tokenizer/tokenizer.compiled", "sha256": args.tokenizer_sha256,
                               "vocabulary_size": model["tokenizer_vocabulary"]}
        if args.tokenizer_dir is None:
            raise SystemExit("--tokenizer-sha256 needs --tokenizer-dir: the raw text prompt prefix comes from the publisher tokenizer")
        prefix = text_prompt_prefix(args.tokenizer_dir)
        if prefix:
            result["tokenizer"]["text_prompt_prefix_token_ids"] = prefix
    return result


HUB_LOG_DIR = "/srv/workspace/sparkpipe/logs"


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
                      f"bash -c 'mkdir -p {HUB_LOG_DIR} && exec ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root {api}/runtime "
                      f"--port {model['api_port']} >> {HUB_LOG_DIR}/{name}-kv-api.log 2>&1' >/dev/null"),
        "api_stop": f"systemctl --user stop {unit} 2>/dev/null; systemctl --user reset-failed {unit} 2>/dev/null; true",
        "health": f"http://127.0.0.1:{model['api_port']}/health",
        **expert_warm(model, root),
    }


def expert_warm(model: dict, root: str) -> dict:
    warm = model.get("expert_warm")
    if warm is None:
        return {}
    revision = json.loads((ROOT / model["contract"]).read_text())["revision"]
    return {"node_warm": (f"cd {root} && SPARK_WEIGHTD_EXPERT_POOL_BYTES=$(grep -o 'SPARK_WEIGHTD_EXPERT_POOL_BYTES=[0-9]*' lane.sh | head -1 | cut -d= -f2) "
                          f"./bin/weightd_warm /tmp/spark_weightd.sock $(readlink -f packs/*.pack) $(cut -c1-64 packs/pack.sha256) {revision} {WORLD} "
                          f"--model {warm['model_tag']} --shared {warm['layers']} {warm['experts']} 1800 > warm.log 2>&1")}


def hub_deployment(model: dict, lane: int, args) -> dict:
    result = deployment(model, lane, args)
    result["tokenizer"] = {"path": "tokenizer/tokenizer.compiled", "sha256": args.tokenizer_sha256,
                           "vocabulary_size": model["tokenizer_vocabulary"]}
    if args.text_prompt_prefix:
        result["tokenizer"]["text_prompt_prefix_token_ids"] = args.text_prompt_prefix
    if args.draft_server:
        result["draft_server"] = args.draft_server
    return result


CONTEXT_FIELDS = (("model", "maximum_context_tokens"), ("maximum_context_tokens",),
                  ("model", "max_position_embeddings"), ("source", "config_json", "max_position_embeddings"))


def native_context_tokens(model: dict) -> int:
    contract = json.loads((ROOT / model["contract"]).read_text())
    for path in CONTEXT_FIELDS:
        value = contract
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, int) and value > 0:
            return value
    raise SystemExit(f"{model['contract']} names no maximum context")


def lane_records() -> dict:
    return json.loads(LANES.read_text())


def record_arguments(swap_id: str) -> argparse.Namespace:
    records = lane_records()["lanes"]
    if swap_id not in records:
        raise SystemExit(f"{LANES} has no lane record {swap_id}; known: {', '.join(sorted(records))}")
    record = records[swap_id]
    model = MODELS[record["model"]]
    if model["swap_id"] != swap_id:
        raise SystemExit(f"lane record {swap_id} names model {record['model']}, whose swap id is {model['swap_id']}")
    block = model.get("kv_block_tokens", KV_BLOCK_TOKENS)
    if record["context_tokens"] % block:
        raise SystemExit(f"lane record {swap_id}: context_tokens {record['context_tokens']} is not a whole number of {block}-token KV pages")
    return argparse.Namespace(
        record=swap_id, model=record["model"], lane=record["lane"], runtime_root=record["runtime_root"],
        weightd_socket=record["weightd_socket"], sequences=record["sequences"], rows=record["rows"],
        shared_rows=record.get("shared_rows"), kv_pages=record["context_tokens"] // block,
        kv_physical_pages=record["kv_physical_pages"], kv_logical_pages=record["kv_logical_pages"],
        kv_backing_bytes=record["kv_backing_bytes"], kv_snapshot_bytes=record["kv_snapshot_bytes"],
        state_budget_bytes=record["state_budget_bytes"], state_memory_bytes=record.get("state_memory_bytes"),
        taps=record.get("taps"), tokenizer_sha256=record["tokenizer_sha256"], tokenizer_dir=None,
        text_prompt_prefix=record["text_prompt_prefix_token_ids"],
        draft_server=record.get("draft_server"), swap_entry=True, output_dir=None)


def render(value: dict) -> str:
    return json.dumps(value, indent=2) + "\n"


def check_arguments(model: dict, args) -> None:
    if not 0 <= args.lane <= 15:
        raise SystemExit("lane must be within 0..15")
    if not 1 <= args.sequences <= MAX_SEQUENCES or not args.sequences <= args.rows <= MAX_ROWS:
        raise SystemExit(f"sequences must be within 1..{MAX_SEQUENCES} and rows within sequences..{MAX_ROWS}")
    if args.shared_rows is not None and not 1 <= args.shared_rows < args.rows:
        raise SystemExit("shared rows must be within 1..rows-1")
    args.kv_physical_pages = args.kv_physical_pages or args.sequences * args.kv_pages
    args.kv_logical_pages = args.kv_logical_pages or args.kv_physical_pages
    if args.kv_physical_pages < args.kv_pages or args.kv_logical_pages < args.kv_physical_pages:
        raise SystemExit("physical pages must hold one sequence and logical pages must cover the physical pages")
    if args.kv_logical_pages < args.sequences * args.kv_pages:
        raise SystemExit(f"logical pages {args.kv_logical_pages} must hold every resident sequence at full length: {args.sequences} x {args.kv_pages}")
    states = (2 * args.sequences + 2) * model["recurrent_bytes"]
    minimum = (args.kv_logical_pages - args.kv_physical_pages + 2) * model["kv_page_bytes"] + states
    if args.kv_backing_bytes < minimum:
        raise SystemExit(f"kv-backing-bytes {args.kv_backing_bytes} cannot hold the spilled pages and the recurrent checkpoints a page store may keep beside them; it needs {minimum}")
    if args.state_memory_bytes is None:
        args.state_memory_bytes = 2 * states
    if model["recurrent_bytes"] and args.state_memory_bytes < states:
        raise SystemExit(f"state-memory-bytes {args.state_memory_bytes} cannot hold two recurrent checkpoints per sequence plus two; it needs {states}")
    if args.state_budget_bytes <= 0:
        raise SystemExit("state-budget-bytes must be positive")


def write_lane(args, output_dir: Path) -> None:
    model = MODELS[args.model]
    check_arguments(model, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "deployment.json").write_text(render(deployment(model, args.lane, args)))
    for rank, host in enumerate(HOSTS):
        (output_dir / f"adapter.{host}.json").write_text(render(adapter_config(model, args.lane, rank, args)))
    if args.record:
        (output_dir / "model_resident.json").write_text(render(hub_deployment(model, args.lane, args)))
    if args.swap_entry:
        (output_dir / "swap_entry.json").write_text(render(swap_entry(args.model, model, args.lane)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--record", default=None,
                        help=f"render the production lane of this swap id from {LANES.relative_to(ROOT)}; no other lane flag applies")
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--lane", type=int)
    parser.add_argument("--runtime-root", help="per-node root, {host} is substituted")
    parser.add_argument("--weightd-socket")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--shared-rows", type=int, default=None)
    parser.add_argument("--kv-pages", type=int, default=64)
    parser.add_argument("--kv-physical-pages", type=int, default=None)
    parser.add_argument("--kv-logical-pages", type=int, default=None)
    parser.add_argument("--kv-backing-bytes", type=int, default=8 << 30)
    parser.add_argument("--kv-snapshot-bytes", type=int, default=DEFAULT_KV_SNAPSHOT_BYTES)
    parser.add_argument("--state-budget-bytes", type=int, default=DEFAULT_STATE_BUDGET_BYTES)
    parser.add_argument("--state-memory-bytes", type=int, default=None,
                        help="per-rank memory for recurrent-state records; default twice the two per sequence plus two the KV binding needs")
    parser.add_argument("--tokenizer-sha256", default=None)
    parser.add_argument("--tokenizer-dir", type=Path, default=None,
                        help="local copy of the publisher tokenizer.json and tokenizer_config.json")
    parser.add_argument("--swap-entry", action="store_true")
    args = parser.parse_args()
    if args.record:
        output_dir = args.output_dir
        args = record_arguments(args.record)
        args.swap_entry = lane_records()["lanes"][args.record]["launcher"] == "stage_lane"
    else:
        if None in (args.model, args.lane, args.runtime_root, args.weightd_socket):
            parser.error("--model, --lane, --runtime-root and --weightd-socket are required without --record")
        output_dir = args.output_dir
        args.taps = None
        args.draft_server = None
    write_lane(args, output_dir)
    print(f"{args.model} lane {args.lane} rendered under {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
