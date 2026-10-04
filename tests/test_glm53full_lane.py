#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import glm53full_lane

ADAPTER = ROOT / "modules/glm52_resident_decode_stage/source/spark_glm52_serving_adapter.c"
CONTRACT = ROOT / "model_contracts/glm53_full_authoritative.json"
CHAT_TEMPLATE = ROOT / "model-families/glm52/chat_template.json"
SOCKET = "/tmp/spark_weightd.sock"


def adapter_members():
    source = ADAPTER.read_text()
    block = re.search(r"SparkGlm52ServingConfigurationMembers\[\] =\s*\{(.*?)\};", source, re.S).group(1)
    return tuple(re.findall(r'"([a-z_0-9]+)"', block))


def rendered(lane, codec="fp8", sequences=8, rows=16, positions=4096, inflight=1, arm=None, node_root=None,
             kv_backing_bytes=4 << 30, kv_physical_bytes=None, **score):
    arguments = argparse.Namespace(lane=lane, codec=codec, arm=arm, socket=SOCKET, kv_backing_bytes=kv_backing_bytes,
                                   kv_physical_bytes=kv_physical_bytes, max_sequence_positions=positions,
                                   execution_row_capacity=rows, sequences=sequences, inflight=inflight,
                                   node_root=node_root, **score)
    return {name: json.loads(text) for name, text in glm53full_lane.render(arguments).items()}


def lane_problems(lane, codec, files, members, revision, arm=None):
    arm = arm or codec
    failures = []
    ports = glm53full_lane.lane_ports(lane)
    contract = json.loads(CONTRACT.read_text())
    deployment = files["model_resident.json"]
    limits = deployment["runtime_limits"]
    if deployment.get("chat_template") != json.loads(CHAT_TEMPLATE.read_text()):
        failures.append("chat_template is not the declared glm52 template")
    if deployment["eos_token_ids"] != contract["geometry"]["eos_token_ids"]:
        failures.append("eos ids differ from the authoritative contract")
    if limits["kv_physical_page_capacity"] != limits["max_active_sequences"] * (limits["max_sequence_positions"] // 64):
        failures.append("kv pages do not cover every resident sequence")
    if deployment["transport"]["control_port_base"] != ports["transport"]:
        failures.append("transport base")
    for rank, node in enumerate(deployment["nodes"]):
        host = glm53full_lane.HOSTS[rank]
        stage = files[node["adapter_configuration_path"]]
        if tuple(stage) != members:
            failures.append(f"rank {rank}: stage members {tuple(stage)} != adapter {members}")
        if node["runtime_root"] != f"/home/{host}/glmfull-lane{lane}/root" or not node["kv_backing_directory"].startswith(node["runtime_root"] + "/"):
            failures.append(f"rank {rank}: runtime root")
        if node["control_endpoint"] != {"kind": "tcp", "host": host, "port": ports["control"] + rank}:
            failures.append(f"rank {rank}: control endpoint")
        if node["node_target"] != f"cuda.sm121.glm52.resident_decode_stage.bf16.expert_{codec}":
            failures.append(f"rank {rank}: node target")
        if stage["stage_pack_path"] != f"packs/glm53full.{arm}.tp16-rank{rank}.glm52sp" or stage["expert_weight_codec"] != codec:
            failures.append(f"rank {rank}: pack path")
        if stage["tp_rank"] != rank or stage["tp_degree"] != 16:
            failures.append(f"rank {rank}: tp")
        if stage["model_revision"] != revision:
            failures.append(f"rank {rank}: model revision differs from the build identity")
        if stage["max_sequence_positions"] != limits["max_sequence_positions"] or stage["execution_row_capacity"] != limits["max_input_rows"]:
            failures.append(f"rank {rank}: adapter limits differ from the engine's")
        collective = stage["tp_collective"]
        if collective["listen_port"] != ports["collective"] + rank or collective["peer_ports"] != list(range(ports["collective"], ports["collective"] + 16)):
            failures.append(f"rank {rank}: collective ports")
    return failures


def api_host_problems():
    failures = []
    settings = {"GLMFULL_LANE": "6", "GLMFULL_CODEC": "fp8", "GLMFULL_FIRMWARE": "/nonexistent", "GLMFULL_WEIGHTD_SOCKET": SOCKET,
                "GLMFULL_EXPERT_POOL_BYTES": "1", "GLMFULL_SPINE_BUDGET_BYTES": "1", "GLMFULL_MEMORY_MAX": "1G",
                "GLMFULL_SEQUENCES": "8", "GLMFULL_ROWS": "16", "GLMFULL_POSITIONS": "2048", "GLMFULL_INFLIGHT": "1",
                "GLMFULL_API_PORT": "8446", "GLMFULL_API_BUILD": "/opt/api", "GLMFULL_API_TOKENIZER": "/opt/tokenizer.json"}
    with tempfile.TemporaryDirectory() as directory:
        fake = Path(directory)
        log = fake / "remote.log"
        for tool in ("ssh", "scp"):
            (fake / tool).write_text(f'#!/bin/sh\nprintf \'{tool}\' >> "{log}"\nfor a in "$@"; do printf \' [%s]\' "$a" >> "{log}"; done\necho >> "{log}"\n')
            (fake / tool).chmod(0o755)
        path = f"{fake}:{Path(sys.executable).parent}:/usr/bin:/bin"

        def lane(command, extra):
            if log.exists():
                log.unlink()
            result = subprocess.run(["bash", str(ROOT / "tools/glm53full_lane.sh"), *command], capture_output=True, text=True,
                                    env={"PATH": path, "HOME": directory, **settings, **extra})
            return result, (log.read_text().splitlines() if log.exists() else [])

        for command in (["api"], ["api-stop"], ["decode", "1,2", "3"]):
            for host in ("sparkf", "spark0", "sparka"):
                result, calls = lane(command, {"GLMFULL_API_HOST": host})
                if result.returncode != 2 or "refused" not in result.stderr or calls:
                    failures.append(f"{command[0]} with GLMFULL_API_HOST={host}: rc={result.returncode} remote calls={len(calls)}")
            for extra in ({}, {"GLMFULL_API_HOST": "rtx5090"}):
                result, calls = lane(command, extra)
                hosts = set()
                for call in calls:
                    words = [w.strip("[]") for w in call.split(" ")[1:]]
                    if call.startswith("ssh "):
                        hosts.add([w for w in words if not w.startswith("-") and "=" not in w][0])
                    else:
                        hosts.update(w.split(":")[0] for w in words if ":" in w)
                if result.returncode != 0 or not calls or hosts != {"rtx5090"}:
                    failures.append(f"{command[0]} {extra}: rc={result.returncode} hosts={sorted(hosts)} stderr={result.stderr[-200:]}")
        result, calls = lane(["start"], {"GLMFULL_RUN_ID": "r20261003a"})
        starts = [call for call in calls if call.startswith("ssh ") and "systemd-run" in call]
        if result.returncode != 0 or len(starts) != 16:
            failures.append(f"start: rc={result.returncode} rank starts={len(starts)}")
        for call in starts:
            if "> logs/residentd-r20261003a.log" not in call or "> residentd.log" in call or "refusing to overwrite" not in call or "ln -sfn logs/residentd-r20261003a.log residentd.log" not in call:
                failures.append(f"start does not keep a per-run rank log: {call[:160]}")
                break
        for bad in ("a/b", "x y", "../up"):
            result, calls = lane(["start"], {"GLMFULL_RUN_ID": bad})
            if result.returncode != 2 or calls:
                failures.append(f"start accepted run id {bad!r}: rc={result.returncode} remote calls={len(calls)}")
        result, calls = lane(["archive", "r20261003a", str(fake / "archive")], {})
        copies = [call for call in calls if call.startswith("scp ") and "logs/residentd-r20261003a.log" in call]
        removals = [call for call in calls if call.startswith("ssh ") and "rm -f" in call and "logs/residentd-r20261003a.log" in call]
        if result.returncode != 0 or len(copies) != 16 or len(removals) != 16 or not any("rank15.log" in call for call in copies):
            failures.append(f"archive: rc={result.returncode} copies={len(copies)} removals={len(removals)}")
    return failures


def main():
    failures = []
    members = adapter_members()
    for arm, codec in (("fp8", "fp8"), ("bf16", "bf16"), ("fp8_s1", "fp8"), ("nvfp4_s1", "nvfp4")):
        revision = subprocess.run([sys.executable, str(ROOT / "tools/glm52_model_contract.py"), "--print-build-identity", arm,
                                   "--expert-codec", codec], capture_output=True, text=True, check=True).stdout.split()[0]
        for lane in range(16):
            failures += [f"lane {lane} {arm}: {problem}"
                         for problem in lane_problems(lane, codec, rendered(lane, codec, arm=arm), members, revision, arm)]
    for arm, codec in (("fp8_s1", "bf16"), ("bf16", "fp8"), ("nvfp4", "fp8"), ("nvfp4_s1", "fp8"), ("fp8_s1", "nvfp4")):
        try:
            rendered(6, codec, arm=arm)
            failures.append(f"arm {arm} rendered with {codec} experts")
        except SystemExit:
            pass
    score = {"score_dump_directory": "score/u3", "score_probe_path": "score/probe2.bin", "score_tier2_rows_path": None}
    stages = rendered(6, "nvfp4", arm="nvfp4_s1", **score)
    for rank in range(16):
        stage = stages[f"config/stage_{rank:02d}.json"]
        if stage.get("score_dump_directory") != "score/u3" or stage.get("score_probe_path") != "score/probe2.bin" or "score_tier2_rows_path" in stage:
            failures.append(f"rank {rank}: score members not rendered as given")
    plain = rendered(6, "nvfp4", arm="nvfp4_s1")
    moved = rendered(6, "nvfp4", arm="nvfp4_s1", node_root="glmfull-ab-lane6/u3p")
    for rank, node in enumerate(moved["model_resident.json"]["nodes"]):
        root = f"/home/{glm53full_lane.HOSTS[rank]}/glmfull-ab-lane6/u3p"
        if node["runtime_root"] != root or node["kv_backing_directory"] != root + "/kvcache":
            failures.append(f"rank {rank}: --node-root not rendered into runtime_root and kv_backing_directory")
    if {name: document for name, document in moved.items() if name != "model_resident.json"} != {name: document for name, document in plain.items() if name != "model_resident.json"}:
        failures.append("--node-root changed a stage config")
    for bad in ("", "/abs/root", "a/../b", "root/", "./root", "a//b"):
        try:
            rendered(6, "nvfp4", arm="nvfp4_s1", node_root=bad)
            failures.append(f"node root {bad!r} rendered")
        except SystemExit:
            pass
    if any(name in plain[f"config/stage_{rank:02d}.json"] for rank in range(16) for name in glm53full_lane.SCORE_MEMBERS):
        failures.append("a render without score options carries score members")
    for bad in ({"score_probe_path": "score/p.bin"}, {"score_dump_directory": "/abs"}, {"score_dump_directory": "a/../b"},
                {"score_dump_directory": ""}, {"score_dump_directory": "score/"}):
        values = dict({"score_dump_directory": None, "score_probe_path": None, "score_tier2_rows_path": None}, **bad)
        try:
            rendered(6, "nvfp4", arm="nvfp4_s1", **values)
            failures.append(f"score members {bad} rendered")
        except SystemExit:
            pass
    page = glm53full_lane.KV_PAGE_BYTES
    if page * glm53full_lane.WORLD != 64 * (78 * (512 + 64) + 21 * 128) * 2 or page != 380928:
        failures.append(f"each rank stores 1/16 of every 64-token KV page: per-rank page bytes {page}")
    spill = (4 << 30) // page
    for budget, physical in ((100 * page, 100), (64 * page, 64), (10000 * page, 8 * 64)):
        limits = rendered(6, kv_physical_bytes=budget)["model_resident.json"]["runtime_limits"]
        if limits["kv_physical_page_capacity"] != physical or limits["kv_logical_page_capacity"] != physical + spill:
            failures.append(f"kv physical budget {budget // page} pages rendered {limits['kv_physical_page_capacity']} physical, "
                            f"{limits['kv_logical_page_capacity']} logical")
    for budget, backing in ((63 * page, 4 << 30), (100 * page, 411 * page)):
        try:
            rendered(6, kv_physical_bytes=budget, kv_backing_bytes=backing)
            failures.append(f"kv physical {budget // page} pages with {backing // page} backing pages rendered")
        except SystemExit:
            pass
    with tempfile.TemporaryDirectory() as directory:
        command = [sys.executable, str(ROOT / "tools/glm53full_lane.py"), "--lane", "6", "--codec", "fp8", "--socket", SOCKET,
                   "--kv-backing-bytes", str(4 << 30), "--max-sequence-positions", "4096", "--execution-row-capacity", "16",
                   "--sequences", "8", "--inflight", "1", "--output", directory]
        subprocess.run(command, check=True, capture_output=True)
        if subprocess.run(command + ["--check"], capture_output=True).returncode != 0:
            failures.append("--check reports drift on a fresh render")
    production = ROOT / "deployment/glm53full_tp16_lane6"
    settings = json.loads((production / "render.json").read_text())
    command = [sys.executable, str(ROOT / "tools/glm53full_lane.py"), "--output", str(production), "--check"]
    for name, value in settings.items():
        command += ["--" + name.replace("_", "-"), str(value)]
    if subprocess.run(command, capture_output=True).returncode != 0:
        failures.append("the committed production lane tree drifted from its render.json; re-render deployment/glm53full_tp16_lane6")
    limits = json.loads((production / "model_resident.json").read_text())["runtime_limits"]
    if limits["max_sequence_positions"] != 262144 or limits["resident_sequence_capacity"] != 2 or limits["max_input_rows"] != 1024 or \
            limits["kv_physical_page_capacity"] != 2 * 262144 // 64 or limits["kv_logical_page_capacity"] != limits["kv_physical_page_capacity"] + settings["kv_backing_bytes"] // page:
        failures.append(f"production lane limits {limits}")
        stage = Path(directory) / "config/stage_03.json"
        stage.write_text(stage.read_text().replace('"tp_rank": 3', '"tp_rank": 4'))
        if subprocess.run(command + ["--check"], capture_output=True).returncode == 0:
            failures.append("--check misses drift")
    script = subprocess.run(["bash", str(ROOT / "tools/glm53full_lane.sh"), "status"], capture_output=True, text=True,
                            env={"PATH": "/usr/bin:/bin"})
    if script.returncode == 0 or "GLMFULL_LANE" not in script.stderr:
        failures.append("glm53full_lane.sh runs without its settings")
    failures += api_host_problems()
    for failure in failures:
        print(failure)
    print(f"glm53full lane: {'FAIL' if failures else 'PASS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
