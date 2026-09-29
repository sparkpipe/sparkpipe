#!/usr/bin/env python3
"""Generate the glm5_next TP16 deployment tree: per-rank stage configs and
the shared multi-node model_resident.json.

Hosts: spark0..sparkf are ranks 0..15 (registry order; rank r -> sparke-hex
r per the fleet pack policy). kv_backing_directory is a DIRECTORY per node
(the KV page store opens it with O_TMPFILE - a file path fails ENOTDIR,
the bring-up finding recorded in the lane report).

Usage:
  python3 tools/glm5_next_gen_deployment.py --output deployment/glm5_next_tp16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

HOSTS = [h for h in os.environ.get(
    "GLM5_NEXT_TP_HOSTS",
    ",".join(f"spark{hex(r)[2:]}" for r in range(16))).split(",") if h]
TP = len(HOSTS)
ROOT_NAME = os.environ.get("GLM5_NEXT_ROOT_NAME", "glm53flash.fp8.tp16")
RUNTIME_ROOT = os.environ.get("GLM5_NEXT_RUNTIME_ROOT",
                              "/home/{host}/sparkdata/" + ROOT_NAME)
CONTROL_BASE = int(os.environ.get("GLM5_NEXT_CONTROL_BASE", "19560"))
COLLECTIVE_BASE = int(os.environ.get("GLM5_NEXT_COLLECTIVE_BASE", "63640"))
TRANSPORT_BASE = int(os.environ.get("GLM5_NEXT_TRANSPORT_BASE", "60710"))
COLLECTIVE_SESSION_BASE = int(os.environ.get(
    "GLM5_NEXT_SESSION_BASE", "61500"))
COLLECTIVE_SESSION_HC_BASE = int(os.environ.get(
    "GLM5_NEXT_SESSION_HC_BASE", "62550"))
COLLECTIVE_ID = 9911223344556679
BACKEND = os.environ.get("GLM5_NEXT_BACKEND", "hidden_transport")
if BACKEND != "hidden_transport":
    raise SystemExit(f"GLM5_NEXT_BACKEND={BACKEND}: the TP device collective "
                     "accepts only hidden_transport")
PACK_TEMPLATE = os.environ.get(
    "GLM5_NEXT_PACK_TEMPLATE",
    "packs/" + ROOT_NAME + ".rank%x.sp")
MODEL_REVISION = "84c6a6aa9497188e15a635ba793b0f95a79b1033"
PRODUCTION_ROOT_NAME = "glm53flash.fp8.tp16"
SCORE_MEMBERS = ("score_dump_directory", "score_probe_path", "score_tier2_rows_path")
REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_ASSET = REPO_ROOT / "qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json"
TOKENIZER_RUNTIME_PATH = "tokenizer/tokenizer.json"
NODE_TARGET = "cuda.sm121.glm5_next.resident_decode_stage.bf16.expert_fp8"
FIRMWARE_HEADER = (Path(__file__).resolve().parents[1] / "modules"
                   / "glm5_next_resident_decode_stage/include/sparkpipe"
                   / "spark_glm5_next_resident_decode_stage_firmware.h")
KV_SHARD_REQUIRED_DEGREE = int(re.search(
    r"#define SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_KV_SHARD_REQUIRED_DEGREE "
    r"(\d+)u", FIRMWARE_HEADER.read_text()).group(1))

TP_COLLECTIVE = {
    "backend": BACKEND,
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
    # collective reads them verbatim (no derived ports); hc gets its own
    # table so the second collective never binds the same listeners.
    "session_ports": [
        [COLLECTIVE_SESSION_BASE + a * TP + b if a != b else 0
         for b in range(TP)] for a in range(TP)],
    "session_ports_hc": [
        [COLLECTIVE_SESSION_HC_BASE + a * TP + b if a != b else 0
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
    configuration = {
        "schema_version": 3,
        "model_revision": MODEL_REVISION,
        "expert_weight_codec": "fp8",
        "stage_pack_path": PACK_TEMPLATE % rank,
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
        "decode_split_context_threshold": 64,
        "tp_degree": TP,
        "tp_rank": rank,
        "tp_collective": dict(TP_COLLECTIVE, listen_port=COLLECTIVE_BASE + rank),
    }
    if TP >= KV_SHARD_REQUIRED_DEGREE:
        configuration["dsa_index_context_parallel"] = 1
        configuration["kv_shard"] = 1
    return configuration


def score_members(values: dict, root_name: str, runtime_root: str, output: Path) -> dict:
    members = {name: values[name] for name in SCORE_MEMBERS if values.get(name) is not None}
    if not members:
        return {}
    empty = [name for name, value in members.items() if value == ""]
    if empty:
        raise SystemExit(f"score-dump members must not be empty: {', '.join(empty)}")
    if "score_dump_directory" not in members:
        raise SystemExit("score_probe_path and score_tier2_rows_path require score_dump_directory")
    committed = (Path(__file__).resolve().parents[1] / "deployment/glm5_next_tp16").resolve()
    if (root_name == PRODUCTION_ROOT_NAME or PRODUCTION_ROOT_NAME in runtime_root
            or output.resolve() == committed):
        raise SystemExit(f"score-dump members are experiment-only and refused for the production root "
                         f"{PRODUCTION_ROOT_NAME} and the committed deployment tree")
    for name, value in members.items():
        if value.startswith("/") or value.endswith("/") or any(part in ("", ".", "..") for part in value.split("/")):
            raise SystemExit(f"{name} must be a normalized path relative to the arm's runtime root: {value}")
    return members


def tokenizer_block(eos_token_ids: list) -> dict:
    data = TOKENIZER_ASSET.read_bytes()
    document = json.loads(data)
    ids = list(document["model"]["vocab"].values())
    ids += [token["id"] for token in document.get("added_tokens", [])]
    vocabulary_size = max(ids) + 1
    outside = [token for token in eos_token_ids if token >= vocabulary_size]
    if outside:
        raise SystemExit(f"{TOKENIZER_ASSET}: eos_token_ids {outside} are outside "
                         f"the tokenizer vocabulary of {vocabulary_size}")
    return {
        "path": TOKENIZER_RUNTIME_PATH,
        "sha256": hashlib.sha256(data).hexdigest(),
        "vocabulary_size": vocabulary_size,
    }


def resident_deployment() -> dict:
    contract = json.loads((REPO_ROOT / "model_contracts/glm53_flash_authoritative.json").read_text())
    # Single source of truth: every dependent constant derives from the
    # seed via tools/spark_serving_profile.py (#1210 drift law). The old
    # hand-pinned literals are gone; GLM5_NEXT_SEQUENCES is the seed.
    import spark_serving_profile
    sequences = int(os.environ.get("GLM5_NEXT_SEQUENCES", "16"))
    derived = spark_serving_profile.derive(
        sequences, stage_config(0)["max_sequence_positions"])
    derived_runtime_limits = {key: derived[key] for key in
                              spark_serving_profile.DEPLOYMENT_RUNTIME_MEMBERS}
    nodes = []
    for rank, host in enumerate(HOSTS):
        nodes.append({
            "rank_index": rank,
            "stage_index": rank,
            "runtime_root": RUNTIME_ROOT.format(host=host),
            "node_target": NODE_TARGET,
            "transport_host": host,
            "adapter_configuration_path": "config/stage.json",
            "kv_backing_directory": "/home/%s/kvcache/" % host + ROOT_NAME,
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
        "runtime_limits": derived_runtime_limits,
        "nodes": nodes,
        "tokenizer": tokenizer_block(contract["tokens"]["eos_token_ids"]),
    }


def render_stage(configuration: dict) -> str:
    rendered = json.dumps(configuration, indent=1)
    return re.sub(r"\[\n(?:\s+\d+,?\n)+\s*\]",
                  lambda match: json.dumps(json.loads(match.group()), separators=(",", ":")),
                  rendered) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    for name in SCORE_MEMBERS:
        parser.add_argument("--" + name.replace("_", "-"), dest=name)
    args = parser.parse_args()
    root = Path(args.output)
    score = score_members(vars(args), ROOT_NAME, RUNTIME_ROOT, root)
    (root / "config").mkdir(parents=True, exist_ok=True)
    for rank in range(TP):
        (root / "config" / ("stage_%02d.json" % rank)).write_text(
            render_stage(dict(stage_config(rank), **score)))
    (root / "model_resident.json").write_text(
        json.dumps(resident_deployment(), indent=1) + "\n")
    print(f"{root}: {TP} stage configs + model_resident.json "
          f"(hosts {HOSTS[0]}..{HOSTS[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
