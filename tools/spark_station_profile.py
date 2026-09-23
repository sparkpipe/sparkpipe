#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import gemma4_tp16_gen_deployment as gemma
import k3_multidev_lane as k3
import laguna_multidev_lane as laguna
import ling_multidev_lane as ling
import qwen38_27b_lane_deployment as qwen27
import qwen38max_multidev_lane as qmax

HOSTS = [f"spark{r:x}" for r in range(16)]
SOCKET = "/run/sparkpipe-weightd-shared/weightd.sock"
GIB = 1024 ** 3
LANES = {"qwen27": 1, "qmax": 2, "k3": 3, "gemma4": 6, "laguna": 8, "ling": 9, "minimax": 10}


def profile(family, version):
    root = "/home/{host}/sparkpipe/station/" + family + "/" + version
    nodes = HOSTS[:4] if family == "qwen27" else HOSTS[8:12] if family == "minimax" else HOSTS
    envs = [{} for _ in nodes]
    if family == "gemma4":
        deployment = gemma.resident_deployment(root, SOCKET)
        configs = [gemma.stage_config(r) for r in range(16)]
        envs = [gemma.tp_environment(r) for r in range(16)]
        packs = [f"/home/{h}/sparkdata/gemma4_31b.bf16.tp16/packs/gemma4_31b_tp16_rank{r:x}_stage0.gemma4sp" for r, h in enumerate(nodes)]
    elif family == "qwen27":
        ports = dict(control=23016, collective=53016, transport=64016)
        deployment = qwen27.resident_deployment(nodes, root, root + "/kv", 2 * GIB, ports, SOCKET, 512)
        configs = [qwen27.stage_config(nodes, r, f"tp4-rank{r:02d}.q38sp", hashlib.sha256(version.encode()).hexdigest(), qwen27.LANE_MODEL_REVISION_DEFAULT, 512, ports) for r in range(4)]
        packs = [f"/home/{h}/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/tp4-rank{r:02d}.q38sp" for r, h in enumerate(nodes)]
        deployment["eos_token_ids"] = [248046, 248044]
    elif family == "qmax":
        deployment = qmax.resident_deployment(root, SOCKET, 2 * GIB, 128)
        configs = [qmax.stage_config(r) for r in range(16)]
        packs = [f"/home/{h}/sparkdata/qwenmax.nvfp4.tp16/packs/qwenmax.nvfp4.tp16.rank{r:x}.sp" for r, h in enumerate(nodes)]
        for r in range(16):
            envs[r] = {"SPARK_QWEN38_MAX_STAGE_TP_DEGREE": "16", "SPARK_QWEN38_MAX_STAGE_TP_RANK": str(r), "SPARK_QWEN38_MAX_STAGE_TP_BACKEND_PATH": "lib/hidden_transport.so", "SPARK_QWEN38_MAX_STAGE_TP_IDENTIFIER": str(0x2000000000001), "SPARK_QWEN38_MAX_STAGE_TP_PORT_BASE": "64032", "SPARK_QWEN38_MAX_STAGE_TP_HOSTS": ",".join(f"10.10.200.{n}" for n in range(16)), "SPARK_QWEN38_MAX_STAGE_TP_LOCAL_HOST": f"10.10.200.{r}", "SPARK_QWEN38_MAX_STAGE_TP_TIMEOUT_MS": "30000", "SPARK_QWEN38_MAX_STAGE_TP_SESSION_PORTS": ",".join(str(0 if a == b else 24576 + 16*a+b) for a in range(16) for b in range(16))}
        deployment["eos_token_ids"] = [248046, 248044]
    elif family == "k3":
        deployment = k3.resident_deployment(root, SOCKET, 2 * GIB)
        configs = [k3.adapter_config(r, 8) for r in range(16)]
        packs = [k3.deployed_pack(r) for r in range(16)]
        for config in configs:
            for key in ("max_sequences", "max_rows", "resident_capacity"):
                config[key] = 1
    elif family == "laguna":
        deployment = laguna.resident_deployment(root, SOCKET, 0x8000000000001, 2 * GIB)
        configs = [laguna.adapter_config(r, 0x8000000000001) for r in range(16)]
        deployment["driver"]["shared_object_path"] = "stages/stage_000/model_driver.so"
        packs = [laguna.deployed_pack(r) for r in range(16)]
    elif family == "ling":
        deployment = ling.resident_deployment(root, SOCKET, "bf16", 2 * GIB)
        configs = [ling.stage_config(r, "bf16") for r in range(16)]
        packs = [ling.deployed_pack(r, "bf16") for r in range(16)]
    elif family == "minimax":
        deployment = json.loads((Path(__file__).resolve().parents[1] / "deployment/minimax_text_tp4/model_resident.json").read_text())
        deployment.pop("@comment", None)
        revision = "minimax-h3-text-bf16-h5120-l64-mrope24-20-20-tp4-v1"
        packs = [f"/home/{h}/sparkdata/minimax.text.bf16.tp4/packs/minimax.text.bf16.tp4.rank{r}.mntx" for r, h in enumerate(nodes)]
        configs = [dict(schema_version=3, model_revision=revision, stage_pack_path="packs/" + Path(p).name, max_sequence_positions=512, tp_degree=4) for p in packs]
        for r, h in enumerate(nodes):
            envs[r] = {"SPARK_MINIMAX_TP_DEGREE": "4", "SPARK_MINIMAX_TP_RANK": str(r), "SPARK_MINIMAX_TP_STANDALONE": "0", "SPARK_MINIMAX_STAGE_TP_BACKEND_PATH": "lib/hidden_transport.so", "SPARK_MINIMAX_STAGE_TP_IDENTIFIER": str(0xA000000000001), "SPARK_MINIMAX_STAGE_TP_PORT_BASE": "64160", "SPARK_MINIMAX_STAGE_TP_HOSTS": ",".join(f"10.10.200.{n}" for n in range(8,12)), "SPARK_MINIMAX_STAGE_TP_LOCAL_HOST": f"10.10.200.{r+8}", "SPARK_MINIMAX_STAGE_TP_TIMEOUT_MS": "30000", "SPARK_MINIMAX_STAGE_TP_SESSION_PORTS": ",".join(str(0 if a == b else 23812 + 4*a+b) for a in range(4) for b in range(4))}
    else:
        raise ValueError(f"unknown family {family}")
    for field in ("max_inflight_submissions", "max_active_sequences", "max_input_rows", "resident_sequence_capacity"):
        deployment["runtime_limits"][field] = 1
    for field in ("kv_logical_page_capacity", "kv_physical_page_capacity"):
        deployment["runtime_limits"][field] = 128
    for r, (host, node, config) in enumerate(zip(nodes, deployment["nodes"], configs)):
        target = root.format(host=host)
        node.update(runtime_root=target, adapter_configuration_path="config/adapter.json", kv_backing_directory=target + "/kv", kv_backing_maximum_bytes=2 * GIB)
        if "max_sequence_positions" in config:
            config["max_sequence_positions"] = 512
        if "execution_row_capacity" in config:
            config["execution_row_capacity"] = 1
        if "decode_split_context_threshold" in config:
            config["decode_split_context_threshold"] = 512
        mesh = list(range((r // 4) * 4, (r // 4 + 1) * 4)) if family == "k3" else list(range((r // 8) * 8, (r // 8 + 1) * 8)) if family == "laguna" else [int(h[-1], 16) for h in nodes]
        envs[r].update(SPARK_WEIGHTD_ATTACH="1", SPARK_WEIGHTD_SOCKET=SOCKET, SPARK_WEIGHTD_LANE=str(LANES[family]), SPARK_TP_MESH_RANKS=",".join(map(str, mesh)), SPARK_TP_WAIT_MODE="hardware", CUDA_MODULE_LOADING="LAZY", CUDA_MODULE_DATA_LOADING="LAZY", CUDA_DEVICE_MAX_CONNECTIONS="32")
    return dict(family=family, version=version, deployment=deployment, ranks=[dict(host=h, pack=p, adapter=c, environment=e) for h, p, c, e in zip(nodes, packs, configs, envs)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=LANES)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not args.version.isalnum():
        parser.error("version must be an alphanumeric immutable release identifier")
    print(json.dumps(profile(args.family, args.version), indent=2))


if __name__ == "__main__":
    main()
