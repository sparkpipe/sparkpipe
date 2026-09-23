#!/usr/bin/env python3
import argparse
import base64
import concurrent.futures
import fcntl
import json
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


class StationError(Exception):
    pass


def run(argv, **kwargs):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=kwargs.pop("timeout", 90), **kwargs)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        try:
            report = json.loads(result.stdout)
            if report.get("errors"):
                detail = json.dumps(report["errors"])
        except (ValueError, AttributeError):
            pass
        raise StationError(f"{argv[0]} failed ({result.returncode}): {detail[-3000:]}")
    return result.stdout


def ssh(host, argv):
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(argv)])


def validate(station):
    if station["schema_version"] != 1 or station["api_host"] != "rtx5090":
        raise StationError("station requires schema 1 and APIs/tokenizers on rtx5090")
    if not re.fullmatch(r"[0-9a-f]{40}", station["core_commit"]):
        raise StationError("core_commit must be an exact commit")
    ports, lanes, totals, pools, units = {}, set(), {}, {}, set()
    for name, model in station["models"].items():
        if not re.fullmatch(r"[a-z0-9_-]+", name):
            raise StationError(f"invalid model name: {name}")
        for field in ("resident_unit", "api_unit"):
            unit = model[field]
            if not re.fullmatch(r"[a-z0-9-]+\.service", unit) or unit == "sparkpipe-weightd-shared.service" or unit in units:
                raise StationError(f"{name}: duplicate or invalid {field}")
            units.add(unit)
        lane = model["lane"]
        if lane in lanes or not 0 <= lane < 16:
            raise StationError(f"duplicate or invalid lane: {lane}")
        lanes.add(lane)
        if not model["nodes"] or len(set(model["nodes"])) != len(model["nodes"]):
            raise StationError(f"{name}: empty or duplicate nodes")
        for host in model["nodes"]:
            if not re.fullmatch(r"spark[0-9a-f]", host):
                raise StationError(f"invalid resident host: {host}")
            for field in ("host_mib", "device_mib", "pool_mib"):
                value = model[field]
                if type(value) is not int or value < (0 if field == "pool_mib" else 1):
                    raise StationError(f"{name}: invalid {field}")
            totals[host] = totals.get(host, station["weightd"][host]["host_mib"] + station["weightd"][host]["device_mib"]) + model["host_mib"] + model["device_mib"]
            pools[host] = pools.get(host, 0) + model["pool_mib"]
            for lo, hi in model["ports"]:
                if not 0 < lo <= hi <= 65535:
                    raise StationError(f"{name}: invalid port range")
                for port in range(lo, hi + 1):
                    key = (host, port)
                    if key in ports:
                        raise StationError(f"{name}: port {host}:{port} already owned by {ports[key]}")
                    ports[key] = name
        port = model["api_port"]
        if type(port) is not int or not 0 < port <= 65535 or ("rtx5090", port) in ports:
            raise StationError(f"{name}: duplicate or invalid API port")
        ports[("rtx5090", port)] = name
    for host, total in totals.items():
        if total > station["node_limit_mib"]:
            raise StationError(f"{host}: {total} MiB reserved exceeds {station['node_limit_mib']} MiB")
        if pools[host] > station["weightd"][host]["pool_mib"]:
            raise StationError(f"{host}: {pools[host]} MiB model pools exceed weightd cap {station['weightd'][host]['pool_mib']} MiB")
    return {host: {"reserved_mib": totals[host], "pool_mib": pools[host]} for host in sorted(totals)}


def check_driver(repo, station, family, ref):
    base = station["core_commit"]
    commit = run(["git", "-C", str(repo), "rev-parse", "--verify", ref + "^{commit}"]).strip()
    run(["git", "-C", str(repo), "merge-base", "--is-ancestor", base, commit])
    paths = run(["git", "-C", str(repo), "diff", "--name-only", "--no-renames", base, commit]).splitlines()
    prefixes = station["models"][family]["driver_paths"]
    denied = [p for p in paths if not any(p == prefix.rstrip("/") or prefix.endswith("/") and p.startswith(prefix) for prefix in prefixes)]
    if denied:
        raise StationError("driver update changes frozen/shared inputs: " + ", ".join(denied))
    return {"family": family, "driver_commit": commit, "core_commit": base, "changed": paths}


def systemctl(station, model, host, *args):
    if host == station["api_host"]:
        return ssh(station["api_user"] + "@" + host, ["env", "XDG_RUNTIME_DIR=" + station["api_runtime_dir"], "systemctl", "--user", *args, model["api_unit"]])
    return ssh(host, ["sudo", "-n", "systemctl", *args, model["resident_unit"]])


def inspect(station, family):
    model = station["models"][family]
    def one(host):
        try:
            raw = systemctl(station, model, host, "show", "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "InvocationID", "-p", "MemoryCurrent", "-p", "MemoryPeak", "-p", "Result")
            return host, dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        except StationError as error:
            return host, {"error": str(error)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        units = dict(pool.map(one, [station["api_host"], *model["nodes"]]))
    return {"family": family, "endpoint": f"http://{station['api_host']}:{model['api_port']}", "units": units}


def queue(station, *args):
    return run([station["controller_python"], station["queue_script"], *args])


def mesh(station, exchange=False):
    if exchange:
        for family in station["models"]:
            states = inspect(station, family)["units"]
            if any(state.get("ActiveState") not in ("inactive", "failed") for state in states.values()):
                raise StationError("stop all station models before exchanging mesh records")
    hosts = sorted(station["weightd"])
    directory = station["mesh_directory"]
    def read(host):
        script = "import base64,json,pathlib,subprocess; p=pathlib.Path(" + repr(directory) + "); print(json.dumps(dict(invocation=subprocess.check_output(['systemctl','show','sparkpipe-weightd-shared.service','-p','InvocationID','--value'],text=True).strip(),record=base64.b64encode((p/" + repr("mesh-" + host[-1] + ".rec") + ").read_bytes()).decode())))"
        data = json.loads(ssh(host, ["python3", "-c", script]))
        if not data["invocation"] or not 0 < len(base64.b64decode(data["record"])) <= 4096:
            raise StationError(f"{host}: missing weightd identity or mesh record")
        return host, data
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        records = dict(pool.map(read, hosts))
    def install(host):
        script = "import base64,json,pathlib; p=pathlib.Path(" + repr(directory) + "); records=json.loads(" + repr(json.dumps(records)) + "); "
        script += "\nfor host,data in records.items():\n if host != " + repr(host) + ":\n  path=p/('mesh-'+host[-1]+'.rec'); temporary=path.with_suffix('.station-tmp'); temporary.write_bytes(base64.b64decode(data['record'])); temporary.replace(path)\n"
        ssh(host, ["python3", "-c", script])
    if exchange:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(install, hosts))
    def verify(host):
        script = "import base64,json,pathlib,subprocess,time; p=pathlib.Path(" + repr(directory) + "); records=json.loads(" + repr(json.dumps(records)) + "); deadline=time.monotonic()+15\n"
        script += "while not (p/'.ready').exists() and time.monotonic()<deadline: time.sleep(.1)\n"
        script += "assert (p/'.ready').is_file(), 'mesh is not ready'\n"
        script += "assert subprocess.check_output(['systemctl','show','sparkpipe-weightd-shared.service','-p','InvocationID','--value'],text=True).strip()==records[" + repr(host) + "]['invocation'], 'weightd restarted during mesh verification'\n"
        script += "for host,data in records.items(): assert (p/('mesh-'+host[-1]+'.rec')).read_bytes()==base64.b64decode(data['record']), 'stale or missing peer '+host\n"
        ssh(host, ["python3", "-c", script])
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(verify, hosts))
    return records


def tracked(station):
    state = json.loads(queue(station, "doctor"))
    return {owner["id"] for owner in state["persistent"]}


def stop(station, family):
    model = station["models"][family]
    errors = []
    def halt(host):
        try:
            systemctl(station, model, host, "stop")
        except (StationError, subprocess.TimeoutExpired) as error:
            errors.append(str(error))
    halt(station["api_host"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(halt, model["nodes"]))
    owners = tracked(station)
    for host in model["nodes"]:
        owner = f"persistent:{host}:system:{model['resident_unit']}"
        if owner in owners:
            try:
                queue(station, "untrack", "--id", owner)
            except StationError as error:
                errors.append(str(error))
    if errors:
        raise StationError("stop incomplete; reservations retained: " + "; ".join(errors))


def verify_manifest(host, root, expected):
    manifest = ssh(host, ["cat", root + "/SHA256SUMS"])
    if hashlib.sha256(manifest.encode()).hexdigest() != expected:
        raise StationError(f"{host}: release manifest changed")
    ssh(host, ["sh", "-c", "cd " + shlex.quote(root) + " && sha256sum --strict --quiet --check SHA256SUMS"])


def recover_core(station):
    release = station["core_release"]
    owners = json.loads(queue(station, "doctor"))
    if owners["active"]:
        raise StationError("active queue jobs must finish before core recovery")
    allowed = {model["resident_unit"] for model in station["models"].values()} | {"sparkpipe-weightd-shared.service"}
    if any(owner["unit"] not in allowed for owner in owners["persistent"]):
        raise StationError("another persistent service must be coordinated before core recovery")
    hosts = sorted(station["weightd"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda host: verify_manifest(host, release["root"].format(host=host), release["manifest_sha256"][host]), hosts))
    for family in station["models"]:
        stop(station, family)
    unit = "sparkpipe-weightd-shared.service"
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda host: ssh(host, ["sudo", "-n", "systemctl", "stop", unit]), hosts))
    known = tracked(station)
    for host in hosts:
        owner = f"persistent:{host}:system:{unit}"
        if owner in known:
            queue(station, "untrack", "--id", owner)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda host: ssh(host, ["sudo", "-n", "systemctl", "start", unit]), hosts))
    for host in hosts:
        queue(station, "track", "--node", host, "--unit", unit, "--scope", "system", "--device-memory-mib", str(station["weightd"][host]["device_mib"]), "--ports", "61900:61900")
    mesh(station, exchange=True)


def start(station, family):
    model = station["models"][family]
    if "release" not in model:
        raise StationError(f"{family}: no qualified installed release; inspect its recorded blocker")
    release = model["release"]
    def verify(host):
        root = release["api_root"] if host == station["api_host"] else release["resident_root"].format(host=host)
        verify_manifest(host, root, release["manifest_sha256"][host])
        state = systemctl(station, model, host, "show", "-p", "ActiveState", "--value").strip()
        if state not in ("inactive", "failed"):
            raise StationError(f"{family} on {host}: {state}; stop this family before starting a release")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(verify, [station["api_host"], *model["nodes"]]))
    port_args = [x for lo, hi in model["ports"] for x in ("--ports", f"{lo}:{hi}")]
    queue(station, "preflight", "--nodes", ",".join(model["nodes"]), "--memory-mib", str(model["host_mib"] + model["device_mib"]), "--device-memory-mib", str(model["device_mib"]), *port_args)
    mesh(station)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda host: systemctl(station, model, host, "start"), model["nodes"]))
        for host in model["nodes"]:
            queue(station, "track", "--node", host, "--unit", model["resident_unit"], "--scope", "system", "--device-memory-mib", str(model["device_mib"]), *port_args)
        def resident_ready(host):
            script = "import subprocess,time; unit=" + repr(model["resident_unit"]) + "; deadline=time.monotonic()+75\n"
            script += "while time.monotonic()<deadline:\n state=subprocess.check_output(['systemctl','show',unit,'-p','ActiveState','--value'],text=True).strip()\n if state not in ('active','activating'): raise SystemExit(unit+' stopped during initialization: '+state)\n invocation=subprocess.check_output(['systemctl','show',unit,'-p','InvocationID','--value'],text=True).strip()\n log=subprocess.check_output(['sudo','-n','journalctl','--no-pager','-o','cat','_SYSTEMD_INVOCATION_ID='+invocation],text=True)\n if 'model_residentd ready rank=' in log: break\n time.sleep(.5)\nelse: raise SystemExit(unit+' initialization deadline expired')\n"
            ssh(host, ["python3", "-c", script])
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(resident_ready, model["nodes"]))
        systemctl(station, model, station["api_host"], "start")
        script = "import json,time,urllib.request,subprocess; deadline=time.monotonic()+75; last='no response'\n"
        script += "while time.monotonic()<deadline:\n try:\n  data=json.load(urllib.request.urlopen('http://127.0.0.1:" + str(model["api_port"]) + "/health',timeout=2))\n  if data.get('status')=='ok' and data.get('tokenizer') is True: print(json.dumps(data)); break\n  last=repr(data)\n except Exception as error: last=str(error)\n time.sleep(.5)\nelse: raise SystemExit('API readiness failed: '+last)\n"
        ssh(station["api_user"] + "@" + station["api_host"], ["python3", "-c", script])
        for host in model["nodes"]:
            state = systemctl(station, model, host, "show", "-p", "ActiveState", "--value").strip()
            if state != "active":
                raise StationError(f"{family} on {host}: resident {state} after API startup")
    except (StationError, subprocess.TimeoutExpired) as error:
        try:
            stop(station, family)
        except StationError as cleanup:
            raise StationError(f"start failed: {error}; cleanup failed: {cleanup}") from error
        raise StationError(f"start failed and this family was stopped: {error}") from error


def smoke(station, family):
    model = station["models"][family]
    script = "import json,time,urllib.request,urllib.error; results=[]\n"
    script += "payload=" + repr(dict(model=model.get("model_id", family), prompt="Hello", max_tokens=2, temperature=0)) + "\n"
    script += "for iteration in range(2):\n start=time.monotonic()\n request=urllib.request.Request('http://127.0.0.1:" + str(model["api_port"]) + "/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})\n try:\n  with urllib.request.urlopen(request,timeout=150) as response: data=json.load(response)\n except urllib.error.HTTPError as error: raise RuntimeError(str(error.code)+' '+error.read().decode()) from None\n print(json.dumps(dict(iteration=iteration,seconds=time.monotonic()-start,response=data)),flush=True)\n assert data.get('status')==0 and len(data.get('tokens',[]))==2, data\n results.append(dict(seconds=time.monotonic()-start,response=data))\n"
    script += "assert results[0]['response']['tokens']==results[1]['response']['tokens'], 'repeat prompt changed tokens'\nprint(json.dumps(results))\n"
    output = run(["ssh", "-o", "BatchMode=yes", station["api_user"] + "@" + station["api_host"], shlex.join(["python3", "-c", script])], timeout=330)
    return {"family": family, "qualification": "two repeated two-token HTTP requests; no numerical oracle", "results": json.loads(output.splitlines()[-1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    commands.add_parser("mesh-exchange")
    commands.add_parser("recover-core")
    for name in ("status", "start", "stop", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("family")
    command = commands.add_parser("check-driver")
    command.add_argument("family")
    command.add_argument("--repo", type=Path, required=True)
    command.add_argument("--ref", required=True)
    args = parser.parse_args()
    station = json.loads(args.station.read_text())
    budget = validate(station)
    if args.command == "plan":
        result = budget
    elif args.command == "check-driver":
        result = check_driver(args.repo, station, args.family, args.ref)
    elif args.command == "smoke":
        result = smoke(station, args.family)
    elif args.command == "status":
        result = [inspect(station, name) for name in station["models"]] if args.family == "all" else inspect(station, args.family)
    else:
        if str(Path.home()) != station["controller_home"] or os.environ.get("SPARK_QUEUE_STATE"):
            raise StationError("run lifecycle commands on the controller with its default queue ledger")
        if hashlib.sha256(Path(station["queue_script"]).read_bytes()).hexdigest() != station["queue_sha256"]:
            raise StationError("pinned controller queue changed; no station changes made")
        with (Path.home() / ".sparkpipe/queue/.dispatcher-v2.lock").open("a") as lock:
            deadline = time.monotonic() + 60
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise StationError("controller dispatcher is busy; no station changes made")
                    time.sleep(.2)
            if args.command == "recover-core":
                recover_core(station)
            elif args.command == "mesh-exchange":
                mesh(station, exchange=True)
            else:
                {"start": start, "stop": stop}[args.command](station, args.family)
        result = {"mesh": "ready", "models": "stopped; start families explicitly"} if args.command == "recover-core" else {"mesh": "ready"} if args.command == "mesh-exchange" else inspect(station, args.family)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (StationError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        sys.exit(f"station: FAIL: {error}")
