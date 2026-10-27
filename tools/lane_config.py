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
INFERENCE_ROOTS = ("/srv/workspace/", "/home/")
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
units = {}
for unit in request.get("units", []):
    shown = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "ExecStart", "--value"], capture_output=True, text=True).stdout
    units[unit] = shown
prefixes = {}
if request.get("prefix"):
    namespace = {"__name__": "tokenizer_text_prefix"}
    exec(compile(request["prefix"]["source"], "tokenizer_text_prefix.py", "exec"), namespace)
    for record_id, directory in request["prefix"]["directories"].items():
        try:
            prefixes[record_id] = namespace["directory_prefix"](namespace["Path"](directory))
        except Exception as error:
            prefixes[record_id] = "%s: %s" % (type(error).__name__, error)
print(json.dumps({"files": files, "roots": roots, "apis": apis, "weightd": weightd, "prefixes": prefixes, "units": units}))
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


RESTORE = r'''
import json, os, sys
request = json.load(sys.stdin)
restored = []
for path in request["paths"]:
    previous = path + ".prev-" + request["stamp"]
    if not os.path.exists(previous):
        continue
    if os.path.exists(path):
        os.replace(path, path + ".undone-" + request["now"])
    os.replace(previous, path)
    restored.append(path)
print(json.dumps(restored))
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
        environment = {name: os.environ[name] for name in ("PATH", "HOME", "SPARK_LANES") if name in os.environ}
        subprocess.run(["bash", str(HERE / "stage_lane.sh"), "lane-scripts", str(directory)],
                       env=dict(environment, STAGE_RECORD=record_id), check=True)


def draftd_environment(record_id: str) -> tuple[str, dict[str, str]]:
    lane = record(record_id)
    spec, server = lane.get("draftd"), lane.get("draft_server")
    if spec is None or server is None:
        raise SystemExit(f"{record_id}: a draft server needs both a draft_server and a draftd record")
    drafter = spec["drafter"]
    paths = [spec["source"], drafter.split(":", 1)[1] if drafter.startswith("dspark:") else drafter, spec["log"]]
    if spec.get("target_shard"):
        paths.append(spec["target_shard"])
    for path in paths:
        if not path.startswith(INFERENCE_ROOTS) or ".." in path.split("/"):
            raise SystemExit(f"{record_id}: draftd path {path} is not on the hub's local NVMe ({', '.join(INFERENCE_ROOTS)}); inference never reads CephFS or cold storage")
    stream = (lane.get("taps") or {}).get("stream", "")
    tap_host, _, tap_port = stream.rpartition(":")
    if not tap_host or not tap_port.isdigit():
        raise SystemExit(f"{record_id}: a draft server needs the record's taps.stream as host:port, the endpoint the rank streams to and draftd listens on")
    environment = {"DRAFTD_UNIT": spec["unit"], "DRAFTD_LOG": spec["log"], "DRAFTD_DRAFTER": drafter, "DRAFTD_TARGET_SHARD": spec.get("target_shard", ""),
                   "DRAFTD_TAP_HOST": tap_host, "DRAFTD_TAP_PORT": tap_port, "DRAFTD_DRAFT_PORT": str(server["port"]),
                   "DRAFTD_WINDOW": str(spec["window"]), "DRAFTD_CONTEXT_BYTES": str(spec["context_bytes"]), "DRAFTD_RESTART": spec.get("restart", "no")}
    return spec["source"], environment


def draftd_flags(environment: dict[str, str]) -> list[str]:
    flags = [f"--drafter {environment['DRAFTD_DRAFTER']}", f"--tap-host {environment['DRAFTD_TAP_HOST']}", f"--tap-port {environment['DRAFTD_TAP_PORT']}",
             f"--draft-port {environment['DRAFTD_DRAFT_PORT']}", f"--window {environment['DRAFTD_WINDOW']}", f"--context-bytes {environment['DRAFTD_CONTEXT_BYTES']}"]
    if environment["DRAFTD_TARGET_SHARD"]:
        flags.append(f"--target-shard {environment['DRAFTD_TARGET_SHARD']}")
    return flags


def draftd_start(record_id: str) -> int:
    source, environment = draftd_environment(record_id)
    settings = " ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items())
    script = (HERE / "hub_draftd.sh").read_text()
    for step in ("stop", f"start {shlex.quote(source)}"):
        completed = subprocess.run([*SSH, HUB, f"env {settings} bash -s {step}"], input=script, capture_output=True, text=True, timeout=600)
        print((completed.stdout + completed.stderr).strip())
        if completed.returncode != 0:
            return completed.returncode
    return 0


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


def audit(selected: list[str], fleet: bool = True) -> int:
    lanes = records()
    fleet = fleet and not selected
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
        for source, live in HUB_FILES if fleet else ():
            expected_hub[f"{HUB_HOME}/{live}"] = ("hub", (ROOT / source).read_text())
        roots = {lanes[record_id]["runtime_root"] for record_id in lanes}
        apis = {f"{HUB_HOME}/{lanes[record_id]['model']}-kv-api" for record_id in lanes}

        def node(host: str):
            return host, remote(host, READ, {"paths": sorted(expected_nodes[host]), "weightd": fleet})

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(stage_lane.HOSTS) + 1) as pool:
            futures = [pool.submit(node, host) for host in stage_lane.HOSTS]
            directories = {record_id: os.path.dirname(stage_lane.MODELS[lanes[record_id]["model"]]["tokenizer_json"])
                           for record_id in selected if stage_lane.MODELS[lanes[record_id]["model"]].get("tokenizer_json")}
            prefix = {"source": (HERE / "tokenizer_text_prefix.py").read_text(), "directories": directories}
            drafters = {record_id: draftd_environment(record_id)[1] for record_id in selected if lanes[record_id].get("draftd")}
            hub_future = pool.submit(remote, HUB, READ, {"paths": sorted(expected_hub), "weightd": False, "prefix": prefix,
                                                         "units": sorted(environment["DRAFTD_UNIT"] for environment in drafters.values())})
            for future in futures:
                try:
                    host, live = future.result()
                except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                    drift.append(f"UNREACHABLE {error}")
                    continue
                for path, (record_id, text) in sorted(expected_nodes[host].items()):
                    if not same(path, text, live["files"][path]):
                        drift.append(f"DRIFT {record_id} {host}:{path} {describe(path, text, live['files'][path])}")
                for root in live["roots"] if fleet else ():
                    if root.replace(f"/home/{host}/", "/home/" + "{host}" + "/") not in roots:
                        drift.append(f"ORPHAN {host}:{root} is a lane root no record names")
                if fleet and live["weightd"] != {key: weightd[key] for key in ("device_bytes_max", "kv_reserve_bytes")}:
                    drift.append(f"DRIFT weightd {host} runs {live['weightd']}, deployment/lanes.json says {weightd}")
            try:
                live = hub_future.result()
                for path, (record_id, text) in sorted(expected_hub.items()):
                    if not same(path, text, live["files"][path]):
                        drift.append(f"DRIFT {record_id} {HUB}:{path} {describe(path, text, live['files'][path])}")
                for record_id, prefix in sorted(live.get("prefixes", {}).items()):
                    if prefix != lanes[record_id]["text_prompt_prefix_token_ids"]:
                        drift.append(f"DRIFT {record_id} text_prompt_prefix_token_ids {lanes[record_id]['text_prompt_prefix_token_ids']}, the publisher tokenizer gives {prefix}")
                for record_id, environment in sorted(drafters.items()):
                    unit = environment["DRAFTD_UNIT"]
                    shown = live.get("units", {}).get(unit, "").replace("'", "")
                    missing = [flag for flag in draftd_flags(environment) if flag not in shown]
                    if not shown.startswith("active"):
                        drift.append(f"DRIFT {record_id} {HUB}: draft server unit {unit} is not active")
                    elif missing or "/mnt/" in shown:
                        drift.append(f"DRIFT {record_id} {HUB}: draft server unit {unit} runs without {missing or 'local paths'} (restart it with lane_config.py draftd {record_id})")
                for api in live["apis"] if fleet else ():
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
    return failed or audit([record_id], fleet=False)


def restore(record_id: str, stamp: str) -> int:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        render(record_id, directory)
        targets = {host: sorted(files) for host, files in node_files(record_id, directory).items()}
        targets[HUB] = sorted(hub_files(record_id, directory))
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = {host: pool.submit(remote, host, RESTORE, {"paths": paths, "stamp": stamp, "now": now}) for host, paths in targets.items()}
        for host, future in futures.items():
            try:
                restored = future.result()
                print(f"{host}: {len(restored)} restored from *.prev-{stamp}" + (f"; replaced files kept as *.undone-{now}" if restored else ""))
            except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                print(f"RESTORE FAILED {error}")
                failed = 1
    return failed


ZCODE = Path.home() / ".zcode" / "v2" / "provider_config.json"
ZCODE_PROVIDER = "custom:sparkpipe"


def zcode(write: bool) -> int:
    if not ZCODE.exists():
        print(f"{ZCODE} does not exist on this machine")
        return 0
    document = json.loads(ZCODE.read_text())
    lanes = records()
    rules = [rule for rule in document["config"]["modelConfigRules"]["providerModelRules"] if rule.get("providerId") == ZCODE_PROVIDER]
    changes = []
    for rule in rules:
        if rule.get("modelId") not in lanes:
            continue
        properties = rule.setdefault("config", {}).setdefault("properties", {})
        context = lanes[rule["modelId"]]["context_tokens"]
        if properties.get("contextWindow") != context:
            changes.append(f"{rule['modelId']}: {properties.get('contextWindow')} -> {context}")
            properties["contextWindow"] = context
    missing = sorted(set(lanes) - {rule.get("modelId") for rule in rules})
    for line in changes:
        print(("SET " if write else "DRIFT ") + line)
    for model in missing:
        print(f"MISSING {model} has no {ZCODE_PROVIDER} rule in {ZCODE}")
    if write and changes:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = ZCODE.with_name(ZCODE.name + f".prev-{stamp}")
        backup.write_text(ZCODE.read_text())
        temporary = ZCODE.with_name(ZCODE.name + ".tmp")
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        os.replace(temporary, ZCODE)
        print(f"{ZCODE} written; previous kept as {backup.name}")
    return 1 if (changes and not write) or missing else 0


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
    command = commands.add_parser("restore", help="put back one record's node and hub files kept by a push as *.prev-STAMP")
    command.add_argument("record")
    command.add_argument("stamp")
    command = commands.add_parser("zcode", help="compare ZCode's context windows for SparkPipe models with the records, or --write them")
    command.add_argument("--write", action="store_true")
    command = commands.add_parser("field", help="print one field of a record")
    command.add_argument("record")
    command.add_argument("field")
    command = commands.add_parser("shell", help="print the STAGE_* variables of a stage-lane record for tools/stage_lane.sh")
    command.add_argument("record")
    command = commands.add_parser("draftd", help="start a record's hub draft server from its draftd record, refusing any path off the hub's local NVMe")
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
    if args.command == "restore":
        return restore(args.record, args.stamp)
    if args.command == "zcode":
        return zcode(args.write)
    if args.command == "draftd":
        return draftd_start(args.record)
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
