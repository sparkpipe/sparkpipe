#!/usr/bin/env python3
"""Production lanes from one record: render, audit and push deployment/lanes.json.

Every production lane is one record in deployment/lanes.json. The node
deployment.json, every rank's config/adapter.json, the stage lanes' lane.sh
and the hub API's model_resident.json are rendered from that record by
tools/stage_lane.py and tools/stage_lane.sh; nothing on the fleet is edited
by hand. audit renders every record and compares it with what the nodes and
the hub hold, together with the hub files that have a repository source,
the weightd KV reserve and ceiling every node runs, and lane roots or API
directories that no record names. push writes one record's rendered files
to the nodes and the hub, keeping each replaced file as <file>.prev-<UTC>;
binaries, packs and KV stay as they are.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_lane  # noqa: E402

ROOT = stage_lane.ROOT
HUB = "rtx5090"
HUB_HOME = "/home/spec"
HUB_FILES = (
    ("deployment/hub/model_swap.json", "sparkpipe-swap/model_swap.json"),
    ("deployment/hub/chat_frontend.json", "sparkpipe-chat/config.json"),
    ("deployment/hub/litellm-chatui.yaml", "sparkpipe-chat/litellm-chatui.yaml"),
    ("config/litellm-config.yaml", "sparkpipe-chat/litellm.yaml"),
)
SSH = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
READ = r'''
import glob, json, os, re, subprocess, sys
request = json.load(sys.stdin)
home = os.path.expanduser("~")
files = {}
for path in request["paths"]:
    try:
        files[path] = open(path).read()
    except OSError:
        files[path] = None
roots = sorted(os.path.dirname(p) for p in glob.glob(home + "/stage-lanes/*/lane*/root/deployment.json") + glob.glob(home + "/k3-lanes/lane*/root/deployment.json"))
apis = sorted(os.path.dirname(p) for p in glob.glob(home + "/*-kv-api/model_resident.json"))
weightd = None
if request["weightd"]:
    line = subprocess.run("grep 'spark_weightd ready' ~/weightd.log | tail -1", shell=True, capture_output=True, text=True).stdout
    match = re.search(r"ceiling=(\d+) kv_reserve=(\d+)", line)
    weightd = {"device_bytes_max": int(match.group(1)), "kv_reserve_bytes": int(match.group(2))} if match else None
print(json.dumps({"files": files, "roots": roots, "apis": apis, "weightd": weightd}))
'''
WRITE = r'''
import json, os, sys
request = json.load(sys.stdin)
changed = []
for path, text in request["files"].items():
    try:
        old = open(path).read()
    except OSError:
        old = None
    if old == text:
        continue
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if old is not None:
        os.replace(path, path + ".prev-" + request["stamp"])
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        handle.write(text)
    os.chmod(temporary, os.stat(path + ".prev-" + request["stamp"]).st_mode if old is not None else 0o755 if path.endswith(".sh") else 0o644)
    os.replace(temporary, path)
    changed.append(path)
print(json.dumps(changed))
'''


def records() -> dict:
    return stage_lane.lane_records()["lanes"]


def record(record_id: str) -> dict:
    lanes = records()
    if record_id not in lanes:
        raise SystemExit(f"deployment/lanes.json has no record {record_id}; known: {', '.join(sorted(lanes))}")
    return lanes[record_id]


def render(record_id: str, directory: Path) -> None:
    lane = record(record_id)
    args = stage_lane.record_arguments(record_id)
    args.swap_entry = lane["launcher"] == "stage_lane"
    stage_lane.write_lane(args, directory)
    if lane["launcher"] == "stage_lane":
        subprocess.run(["bash", str(HERE / "stage_lane.sh"), "lane-scripts", str(directory)],
                       env={**os.environ, "STAGE_RECORD": record_id}, check=True)


def node_files(record_id: str, directory: Path) -> dict[str, dict[str, Path]]:
    lane = record(record_id)
    result = {}
    for host in stage_lane.HOSTS:
        root = lane["runtime_root"].format(host=host)
        files = {f"{root}/deployment.json": directory / "deployment.json",
                 f"{root}/config/adapter.json": directory / f"adapter.{host}.json"}
        if lane["launcher"] == "stage_lane":
            files[f"{root}/lane.sh"] = directory / f"lane.{host}.sh"
        result[host] = files
    return result


def hub_files(record_id: str, directory: Path) -> dict[str, Path]:
    return {f"{HUB_HOME}/{record(record_id)['model']}-kv-api/model_resident.json": directory / "model_resident.json"}


def remote(host: str, script: str, request: dict) -> dict | list:
    completed = subprocess.run([*SSH, host, "python3", "-c", shlex.quote(script)], input=json.dumps(request),
                               capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise RuntimeError(f"{host}: {completed.stderr.strip()[-400:]}")
    return json.loads(completed.stdout)


def same(path: str, expected: str, actual: str | None) -> bool:
    if actual is None:
        return False
    if path.endswith(".json"):
        try:
            return json.loads(expected) == json.loads(actual)
        except json.JSONDecodeError:
            return False
    return expected == actual


def first_difference(expected, actual, path: str = "") -> str:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if expected.get(key) != actual.get(key):
                return first_difference(expected.get(key), actual.get(key), f"{path}.{key}")
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        for index, (left, right) in enumerate(zip(expected, actual)):
            if left != right:
                return first_difference(left, right, f"{path}[{index}]")
    return f"{path or '.'}: record {json.dumps(expected)[:120]} live {json.dumps(actual)[:120]}"


def describe(path: str, expected: str, actual: str | None) -> str:
    if actual is None:
        return "missing"
    if path.endswith(".json"):
        try:
            return first_difference(json.loads(expected), json.loads(actual))
        except json.JSONDecodeError:
            return "not JSON"
    return "text differs"


def audit(selected: list[str]) -> int:
    lanes = records()
    selected = selected or sorted(lanes)
    weightd = stage_lane.lane_records()["weightd"]
    drift = []
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        expected_nodes = {host: {} for host in stage_lane.HOSTS}
        expected_hub = {}
        for record_id in selected:
            directory = base / record_id
            render(record_id, directory)
            for host, files in node_files(record_id, directory).items():
                expected_nodes[host].update({path: (record_id, source.read_text()) for path, source in files.items()})
            expected_hub.update({path: (record_id, source.read_text()) for path, source in hub_files(record_id, directory).items()})
        for source, live in HUB_FILES:
            expected_hub[f"{HUB_HOME}/{live}"] = ("hub", (ROOT / source).read_text())
        roots = {lanes[record_id]["runtime_root"] for record_id in lanes}
        apis = {f"{HUB_HOME}/{lanes[record_id]['model']}-kv-api" for record_id in lanes}

        def node(host: str):
            return host, remote(host, READ, {"paths": sorted(expected_nodes[host]), "weightd": True})

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(stage_lane.HOSTS) + 1) as pool:
            futures = [pool.submit(node, host) for host in stage_lane.HOSTS]
            hub_future = pool.submit(remote, HUB, READ, {"paths": sorted(expected_hub), "weightd": False})
            for future in futures:
                try:
                    host, live = future.result()
                except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                    drift.append(f"UNREACHABLE {error}")
                    continue
                for path, (record_id, text) in sorted(expected_nodes[host].items()):
                    if not same(path, text, live["files"][path]):
                        drift.append(f"DRIFT {record_id} {host}:{path} {describe(path, text, live['files'][path])}")
                for root in live["roots"]:
                    if root.replace(f"/home/{host}/", "/home/" + "{host}" + "/") not in roots:
                        drift.append(f"ORPHAN {host}:{root} is a lane root no record names")
                if live["weightd"] != {key: weightd[key] for key in ("device_bytes_max", "kv_reserve_bytes")}:
                    drift.append(f"DRIFT weightd {host} runs {live['weightd']}, deployment/lanes.json says {weightd}")
            try:
                live = hub_future.result()
                for path, (record_id, text) in sorted(expected_hub.items()):
                    if not same(path, text, live["files"][path]):
                        drift.append(f"DRIFT {record_id} {HUB}:{path} {describe(path, text, live['files'][path])}")
                for api in live["apis"]:
                    if api not in apis:
                        drift.append(f"ORPHAN {HUB}:{api} is an API directory no record names")
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                drift.append(f"UNREACHABLE {error}")
    for line in drift:
        print(line)
    print(f"lane audit: {len(selected)} records, {len(drift)} findings" + ("" if drift else "; the fleet and the hub match deployment/lanes.json"))
    return 1 if drift else 0


def push(record_id: str) -> int:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        render(record_id, directory)
        targets = {host: {path: source.read_text() for path, source in files.items()}
                   for host, files in node_files(record_id, directory).items()}
        targets[HUB] = {path: source.read_text() for path, source in hub_files(record_id, directory).items()}
        failed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as pool:
            futures = {host: pool.submit(remote, host, WRITE, {"files": files, "stamp": stamp}) for host, files in targets.items()}
            for host, future in futures.items():
                try:
                    changed = future.result()
                    print(f"{host}: {len(changed)} changed" + (f" ({', '.join(os.path.basename(path) for path in changed)}; previous kept as *.prev-{stamp})" if changed else ""))
                except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                    print(f"PUSH FAILED {error}")
                    failed = 1
    return failed or audit([record_id])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("render", help="render one record's node and hub files into a directory")
    command.add_argument("record")
    command.add_argument("directory", type=Path)
    command = commands.add_parser("audit", help="compare the fleet and the hub with every record, or the named ones")
    command.add_argument("records", nargs="*")
    command = commands.add_parser("push", help="write one record's rendered files to the nodes and the hub, then audit it")
    command.add_argument("record")
    command = commands.add_parser("field", help="print one field of a record")
    command.add_argument("record")
    command.add_argument("field")
    command = commands.add_parser("shell", help="print the STAGE_* variables of a stage-lane record for tools/stage_lane.sh")
    command.add_argument("record")
    command = commands.add_parser("release-row", help="print the release kit's lane, firmware library, installed library, API directory, port and KV label of a record")
    command.add_argument("record")
    args = parser.parse_args()
    if args.command == "render":
        render(args.record, args.directory)
        print(f"{args.record} rendered under {args.directory}")
        return 0
    if args.command == "audit":
        return audit(args.records)
    if args.command == "push":
        return push(args.record)
    if args.command == "field":
        print(record(args.record)[args.field])
        return 0
    lane = record(args.record)
    if args.command == "release-row":
        model = stage_lane.MODELS[lane["model"]]
        root = lane["runtime_root"].format(host="HOST")
        if not root.startswith("/home/HOST/") or not root.endswith("/root"):
            raise SystemExit(f"{args.record}: runtime root {lane['runtime_root']} is not /home/{{host}}/<lane>/root")
        print(root[len("/home/HOST/"):-len("/root")], model.get("firmware_adapter", model["adapter"]), model["adapter"],
              f"{lane['model']}-kv-api", model["api_port"], model["kv_label"])
        return 0
    if lane["launcher"] != "stage_lane":
        raise SystemExit(f"{args.record} is not launched by tools/stage_lane.sh")
    values = {"STAGE_MODEL": lane["model"], "STAGE_LANE": lane["lane"], "STAGE_WEIGHTD_SOCKET": lane["weightd_socket"],
              "STAGE_EXPERT_POOL_BYTES": lane["expert_pool_bytes"], "STAGE_MEMORY_MAX": lane["memory_max"]}
    print("\n".join(f"{key}={shlex.quote(str(value))}" for key, value in values.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
