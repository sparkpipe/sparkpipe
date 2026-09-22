#!/usr/bin/env python3
"""Queue v2 contract: docs/PARALLEL_DRIVER_DEBUG.md."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid

STATE = Path(os.environ.get("SPARK_QUEUE_STATE", Path.home() / ".sparkpipe/queue"))
ACTIVE = {"launching", "running", "stopping"}
JOB_MEMORY_MIB_DEFAULT = 8192
NODE_MEMORY_MIB_MAX = 114688
HOST_HEADROOM_MIB = 8192
MIB = 1024 * 1024
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
            "-o", "ServerAliveInterval=3", "-o", "ServerAliveCountMax=1"]

def validate_ttl(value):
    if not math.isfinite(value) or not 0 < value <= 15:
        raise SystemExit("task window must be finite, positive, and at most 15 minutes")
    return value

def valid_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise SystemExit("invalid ID: use 1-80 letters, digits, dots, underscores or hyphens")
    return value

def timestamp(value):
    if isinstance(value, (int, float)):
        return value
    return time.mktime(time.strptime(value, "%Y-%m-%dT%H:%M:%S"))

def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default

def migrate():
    def lines(name):
        path = STATE / name
        return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
    jobs = lines("queue.jsonl")
    holds = read_json(STATE / "reservations.json", {})
    fences = read_json(STATE / "fenced.json", {})
    for job in jobs:
        if job.get("state") == "running":
            job["state"] = "legacy-review"
            for node in job["nodes"]:
                fences[node] = {"reason": "legacy process requires operator cleanup", "id": job["id"]}
    return {"version": 2, "jobs": jobs, "results": lines("results.jsonl"),
            "manual": [dict(r, nodes=[n]) for n, r in holds.items() if r.get("id", "").startswith("manual:")],
            "fences": fences}

def persist(path, state):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as out:
        json.dump(state, out, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)
    fd = os.open(STATE, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def load():
    state = read_json(STATE / "state-v2.json", None)
    state = migrate() if state is None else state
    state.setdefault("persistent", [])
    return state


@contextmanager
def locked(op):
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / ".lock").open("a") as lock:
        fcntl.flock(lock, op)
        yield


@contextmanager
def transaction():
    with locked(fcntl.LOCK_EX):
        state = load()
        yield state
        persist(STATE / "state-v2.json", state)


@contextmanager
def snapshot():
    with locked(fcntl.LOCK_SH):
        yield load()

def ssh(node, command, timeout=12):
    try:
        out = subprocess.run(["ssh", *SSH_OPTS, node, command], capture_output=True, text=True, timeout=timeout)
        return out.returncode, out.stdout.strip()
    except subprocess.TimeoutExpired:
        return 124, ""



def parse_ports(values):
    ports = []
    for value in values:
        if not re.fullmatch(r"[0-9]+:[0-9]+", value):
            raise SystemExit("--ports must be START:END")
        first, last = map(int, value.split(":"))
        if not 1024 <= first <= last <= 65535 or any(first <= b and a <= last for a,b in ports):
            raise SystemExit("port ranges must be distinct and within 1024..65535")
        ports.append([first, last])
    return ports

def owner_key(owner):
    return owner.get("scope", "system") + ":" + owner.get("unit", "sparkqueue-" + owner.get("attempt", "") + ".service")

def observe(node, owners):
    script = r'''import json, pathlib, subprocess, sys
units = json.loads(sys.argv[1])
def command(args):
    return subprocess.check_output(args, text=True, timeout=5).strip()
mem = dict((a, int(b.split()[0]) // 1024) for a,b in
           (line.split(":", 1) for line in pathlib.Path("/proc/meminfo").read_text().splitlines()))
report = {"total_mib": mem["MemTotal"], "available_mib": mem["MemAvailable"],
          "gpu_names": command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines(),
          "units": {}, "gpu_processes": [], "ports": []}
for key in units:
    scope, unit = key.split(":", 1)
    args = ["systemctl"] + (["--user"] if scope == "user" else [])
    output = command(args + ["show", unit, "-p", "LoadState", "-p", "ActiveState",
        "-p", "ControlGroup", "-p", "MemoryMax", "-p", "MemoryCurrent", "-p", "InvocationID"])
    fields = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    if fields.get("ControlGroup"):
        root = pathlib.Path("/sys/fs/cgroup" + fields["ControlGroup"])
        charges = dict((name, int(value)) for name,value in (line.split() for line in (root / "memory.stat").read_text().splitlines()))
        current = int((root / "memory.current").read_text())
        fields["MemoryNonreclaimable"] = max(0, current - charges["file"] - charges["slab_reclaimable"])
    report["units"][key] = fields
for value in command(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
    pid, used_mib = map(int, value.split(","))
    groups = pathlib.Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    group = next(line[3:] for line in groups if line.startswith("0::"))
    report["gpu_processes"].append({"pid": pid, "control_group": group, "used_mib": used_mib})
for name in ("tcp", "tcp6"):
    for line in pathlib.Path("/proc/net/" + name).read_text().splitlines()[1:]:
        fields = line.split()
        if fields[3] == "0A":
            report["ports"].append(int(fields[1].split(":")[1], 16))
print(json.dumps(report))
'''
    rc, output = ssh(node, shlex.join(["python3", "-c", script,
        json.dumps(sorted({owner_key(owner) for owner in owners}))]), timeout=30)
    try:
        return json.loads(output) if rc == 0 else {"error": "node census failed: exit " + str(rc)}
    except ValueError:
        return {"error": "invalid node census"}

def shared_admission(job, held, observations):
    for node in job["nodes"]:
        report = observations.get(node, {})
        if report.get("error") or not report:
            return node + ": " + report.get("error", "fresh resource census required")
        if report.get("gpu_names") != ["NVIDIA GB10"]:
            return node + ": shared admission requires a verified GB10 unified-memory node"
        owners = [owner for owner in held if node in owner["nodes"] and
                  node not in owner.get("released_nodes", [])]
        owned_groups, reserved, unused = {}, 0, 0
        for owner in owners:
            budget = owner.get("memory_mib", 0)
            if not isinstance(budget, int) or budget <= 0:
                return node + ": owner lacks finite memory budget: " + owner["id"]
            reserved += budget
            unit = report.get("units", {}).get(owner_key(owner))
            if unit is None and owner.get("state") == "launching" and owner.get("attempt") not in report.get("attempts", []):
                unused += budget
                continue
            if not unit or unit.get("LoadState") != "loaded" or unit.get("ActiveState") not in {"active", "activating"}:
                return node + ": owner unit is not verifiably active: " + owner["id"]
            try:
                maximum, current = int(unit["MemoryMax"]), int(unit["MemoryNonreclaimable"])
            except (KeyError, ValueError, TypeError):
                return node + ": owner unit has no finite memory bound: " + owner["id"]
            group = unit.get("ControlGroup", "")
            if maximum <= 0 or math.ceil(maximum / MIB) != budget - owner.get("device_memory_mib", 0) or current < 0 or not group.startswith("/") or group == "/":
                return node + ": owner unit budget/control group mismatch: " + owner["id"]
            if owner.get("invocation") and (owner["invocation"] != unit.get("InvocationID") or owner["control_group"] != group):
                return node + ": persistent unit identity changed: " + owner["id"]
            if group in owned_groups:
                return node + ": duplicate control group ownership: " + group
            owned_groups[group] = [owner.get("device_memory_mib", 0), 0, owner["id"]]
            unused += max(0, budget - current // MIB)
        for process in report.get("gpu_processes", []):
            group = process["control_group"]
            matches = [parent for parent in owned_groups if group == parent or group.startswith(parent + "/")]
            if len(matches) != 1:
                return node + ": unaccounted GPU process " + str(process["pid"]) + " in " + group
            used = process.get("used_mib")
            if not isinstance(used, int) or used < 0:
                return node + ": GPU process memory is unavailable: " + str(process["pid"])
            owned_groups[matches[0]][1] += used
        for device_budget, used, identity in owned_groups.values():
            if used > device_budget:
                return node + ": observed CUDA memory exceeds declared device budget: " + identity
        budget = job["memory_mib"]
        capacity = min(NODE_MEMORY_MIB_MAX, report["total_mib"] - HOST_HEADROOM_MIB)
        if reserved + budget > capacity or unused + budget > report["available_mib"] - HOST_HEADROOM_MIB:
            return node + ": insufficient memory after owner reservations and host headroom"
        if any(first <= port <= last for first, last in job.get("ports", []) for port in report.get("ports", [])):
            return node + ": requested port is already listening"
    return None

def resource_observations(state):
    nodes = {node for job in state["jobs"] if job.get("resources") == "gpu-shared" and
             job.get("state") in ACTIVE | {"queued", "blocked"} for node in job["nodes"]}
    nodes.update(node for owner in state["persistent"] for node in owner["nodes"])
    owners = [job for job in state["jobs"] if job.get("state") in ACTIVE] + state["persistent"] + state["manual"]
    def probe(node):
        held = [owner for owner in owners if node in owner["nodes"] and node not in owner.get("released_nodes", [])]
        result = observe(node, held)
        result["attempts"] = [owner["attempt"] for owner in held if "attempt" in owner]
        return node, result
    with ThreadPoolExecutor(max_workers=16) as pool:
        return dict(pool.map(probe, sorted(nodes)))

def remote(job, node, action):
    unit = "sparkqueue-" + job["attempt"]
    total_mib = job.get("memory_mib", JOB_MEMORY_MIB_DEFAULT)
    device_mib = job.get("device_memory_mib", 0)
    mib = str(total_mib - device_mib)
    ctl = "sudo -n systemctl"
    show = f"{ctl} show {unit} -p LoadState -p ActiveState -p SubState -p ExecMainStatus -p Result"
    if action == "stop":
        command = (f'if [ "$({ctl} show {unit} -p LoadState --value)" != not-found ]; '
                   f"then {ctl} stop {unit} >/dev/null 2>&1 || exit 1; fi; {show}")
    elif action == "launch":
        owners = [owner for owner in job.get("admission_units", []) if node in owner["nodes"]]
        if owners:
            report = observe(node, owners)
            for owner in owners:
                actual = report.get("units", {}).get(owner_key(owner), {})
                expected_maximum = owner["host_memory_bytes"]
                if actual.get("ActiveState") != "active" or actual.get("InvocationID") != owner["invocation"] or actual.get("ControlGroup") != owner["control_group"] or actual.get("MemoryMax") != str(expected_maximum):
                    return {"LoadState": "not-found", "admission_error": "persistent owner changed before launch: " + owner["id"]}
        remaining = max(1, math.ceil(job["deadline"] - time.time()))
        cwd = (job.get("cwd") or "$HOME").replace("{host}", node)
        cd = 'cd "$HOME"' if cwd == "$HOME" else "cd " + shlex.quote(cwd)
        inner = cd + " && exec bash -c " + shlex.quote(job["cmd"])
        argv = ["sudo", "-n", "systemd-run", "--quiet", "--unit=" + unit,
                "--uid=" + node, "--property=Type=exec", "--property=RemainAfterExit=yes",
                "--property=KillMode=control-group", "--property=TimeoutStopSec=5",
                "--property=RuntimeMaxSec=" + str(remaining),
                "--property=MemoryMax=" + mib + "M",
                "--property=LimitMEMLOCK=" + str(total_mib) + "M",
                "--property=MemorySwapMax=0", "--property=TasksMax=512",
                "--property=StandardOutput=append:/tmp/" + unit + ".log",
                "--property=StandardError=inherit", "--setenv=SPARK_QUEUE_RANK=" + str(job["nodes"].index(node)),
                "--setenv=SPARK_QUEUE_SIZE=" + str(len(job["nodes"])),
                "--setenv=SPARK_QUEUE_ID=" + job["id"],
                "--setenv=SPARK_QUEUE_ATTEMPT=" + job["attempt"],
                "--setenv=SPARK_QUEUE_RUNTIME_ROOT=/tmp/" + unit,
                "--setenv=SPARK_QUEUE_MEMORY_MIB=" + str(total_mib),
                "--setenv=SPARK_QUEUE_DEVICE_MEMORY_MIB=" + str(device_mib),
                "--setenv=SPARK_QUEUE_PORTS=" + ",".join(f"{a}:{b}" for a,b in job.get("ports", [])),
                "bash", "-c", inner]
        command = (f'if [ "$({ctl} show {unit} -p LoadState --value)" = not-found ]; '
                   f"then {shlex.join(argv)}; fi; {show}")
    else:
        command = show
    rc, out = ssh(node, command)
    if rc != 0:
        return {"unknown": True, "ssh_exit": rc}
    fields = dict(s.split("=", 1) for s in out.splitlines() if "=" in s)
    return fields if fields.get("LoadState") else {"unknown": True}

def terminal(reply):
    return not reply.get("unknown") and (reply.get("LoadState") == "not-found" or
        reply.get("ActiveState") in {"inactive", "failed"} or reply.get("SubState") == "exited")

def conflicts(left, right):
    overlaps = bool((set(left["nodes"]) - set(left.get("released_nodes", []))) &
                    (set(right["nodes"]) - set(right.get("released_nodes", []))))
    resources = {left.get("resources", "gpu"), right.get("resources", "gpu")}
    ports_overlap = any(a <= d and c <= b for a,b in left.get("ports", []) for c,d in right.get("ports", []))
    return overlaps and (ports_overlap or "exclusive" in resources or
        ("gpu" in resources and resources <= {"gpu", "gpu-shared"}))

def memory_available(job, held):
    for node in job["nodes"]:
        used = sum(owner.get("memory_mib", 0) for owner in held
                   if node in owner["nodes"] and node not in owner.get("released_nodes", []))
        if used + job.get("memory_mib", JOB_MEMORY_MIB_DEFAULT) > NODE_MEMORY_MIB_MAX:
            return False
    return True

def finish(state, job, code, reason):
    state["results"].append(dict(job, state="finished", exit=code, finished_at=time.time(), note=reason))
    state["jobs"].remove(job)

def reconcile(snapshot):
    operations = [(j, n, "stop" if j["state"] == "stopping" or time.time() >= j["deadline"]
                   else "launch" if j["state"] == "launching" else "poll")
                  for j in snapshot if j["state"] in ACTIVE for n in j["nodes"]]
    with ThreadPoolExecutor(max_workers=32) as pool:
        replies = list(pool.map(lambda op: remote(*op), operations))
    grouped = {}
    for (job, node, action), reply in zip(operations, replies):
        grouped.setdefault(job["attempt"], {})[node] = (action, reply)
    with transaction() as state:
        for job in list(state["jobs"]):
            results = grouped.get(job.get("attempt"))
            if results is None:
                continue
            job["observed"] = {node: reply for node, (_, reply) in results.items()}
            if job["state"] == "stopping" or time.time() >= job["deadline"]:
                job["state"] = "stopping"
                job.setdefault("exit", 124)
                job["released_nodes"] = [n for n, (a, r) in results.items()
                    if a == "stop" and not r.get("unknown") and
                    (r.get("LoadState") == "not-found" or r.get("ActiveState") in {"inactive", "failed"})]
                if len(job["released_nodes"]) == len(job["nodes"]):
                    finish(state, job, job["exit"], "all participant control groups stopped")
                continue
            if any(r.get("unknown") for _, r in results.values()):
                job["unknown_passes"] = job.get("unknown_passes", 0) + 1
                if job["unknown_passes"] >= 2:
                    job.update(state="stopping", exit=75)
                continue
            job["unknown_passes"] = 0
            missing = any(r.get("LoadState") == "not-found" for _, r in results.values())
            failed = any(r.get("ActiveState") == "failed" or
                         (terminal(r) and int(r.get("ExecMainStatus", "0")) != 0) for _, r in results.values())
            if missing or failed:
                code = 124 if any(r.get("Result") == "timeout" for _, r in results.values()) else 1
                job.update(state="stopping", exit=code)
            elif all(terminal(r) for _, r in results.values()):
                job.update(state="stopping", exit=0)
            else:
                job["state"] = "running"

def claim(default_ttl, observations=None):
    observations = observations or {}
    with transaction() as state:
        now = time.time()
        state["manual"] = [r for r in state["manual"] if now < timestamp(r["acquired_at"]) + r["ttl_minutes"] * 60]
        done = {r["id"] for r in state["results"] if r.get("exit") == 0}
        failed = {r["id"] for r in state["results"] if r.get("exit") != 0}
        held = [j for j in state["jobs"] if j["state"] in ACTIVE] + state["manual"] + state["persistent"]
        unavailable = set(state["fences"])
        for owner in held:
            if owner.get("state") == "stopping":
                unavailable.update(set(owner["nodes"]) - set(owner.get("released_nodes", [])))
        candidates = []
        for job in list(state["jobs"]):
            if job.get("state") not in {"queued", "blocked"}:
                continue
            failed_dependencies = sorted(set(job.get("after", [])) & failed)
            if failed_dependencies:
                job["failed_dependencies"] = failed_dependencies
                finish(state, job, 125, "dependency failed: " + ", ".join(failed_dependencies))
                failed.add(job["id"])
                continue
            if job.get("kind", "run") != "run":
                continue
            try:
                ttl = validate_ttl(float(job.get("ttl_minutes") or default_ttl))
                valid = job.get("cmd", "").strip() and (len(job["nodes"]) == 1 or job.get("per_node"))
            except (ValueError, TypeError, SystemExit):
                valid = False
            if not valid:
                job.update(state="invalid", error="legacy command/window requires resubmission")
                continue
            job["ttl_minutes"] = ttl
            if not all(dep in done for dep in job.get("after", [])):
                continue
            if any(n in unavailable for n in job["nodes"]):
                continue
            candidates.append(job)
        candidates.sort(key=lambda j: (0 if now - timestamp(j["submitted_at"]) > 7200 else j.get("priority", 5), timestamp(j["submitted_at"])))
        waiting, claimed = [], []
        for job in candidates:
            if any(conflicts(job, other) for other in held + waiting) or not memory_available(job, held):
                waiting.append(job)
                continue
            if job.get("resources") == "gpu-shared" or any(owner.get("resources") == "gpu-shared" and set(job["nodes"]) & set(owner["nodes"]) for owner in held):
                reason = shared_admission(job, held, observations)
                if reason:
                    job["admission_error"] = reason
                    waiting.append(job)
                    continue
            job.pop("admission_error", None)
            job["admission_units"] = [dict(owner) for owner in state["persistent"] if set(job["nodes"]) & set(owner["nodes"])]
            job.update(state="launching", attempt=uuid.uuid4().hex,
                       deadline=now + job["ttl_minutes"] * 60, dispatched_at=now)
            held.append(job)
            claimed.append(dict(job))
        return claimed

def cmd_dispatch(args):
    ttl = validate_ttl(float(args.ttl))
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / ".dispatcher-v2.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        with transaction() as state:
            jobs_snapshot = [dict(j) for j in state["jobs"]]
        reconcile(jobs_snapshot)
        with snapshot() as current:
            census_state = current
        observations = resource_observations(census_state)
        jobs = claim(ttl, observations)
        if jobs:
            reconcile(jobs)
        with transaction() as state:
            state["dispatcher_at"] = time.time()
        print(f"claimed {len(jobs)} jobs")

def cmd_add(args):
    valid_name(args.id)
    nodes = args.nodes.split(",")
    if len(set(nodes)) != len(nodes) or any(not re.fullmatch(r"spark[0-9a-f]", n) for n in nodes):
        raise SystemExit("nodes must be distinct spark0..sparkf aliases")
    if len(nodes) > 1 and args.kind == "run" and not args.per_node:
        raise SystemExit("multi-node jobs require --per-node; all participants need bounded control groups")
    if args.cmd_file:
        if args.cmd:
            raise SystemExit("--cmd and --cmd-file are mutually exclusive")
        args.cmd = args.cmd_file.read_text()
    if args.kind == "run" and not (args.cmd or "").strip():
        raise SystemExit("run entries require a nonempty command")
    ttl = validate_ttl(3.0 if args.ttl_min is None else args.ttl_min)
    if args.memory_mib is None:
        if args.resources == "gpu-shared":
            raise SystemExit("gpu-shared requires an explicit --memory-mib budget")
        args.memory_mib = JOB_MEMORY_MIB_DEFAULT
    device_mib = getattr(args, "device_memory_mib", None)
    if args.resources == "gpu-shared":
        if device_mib is None or device_mib <= 0 or args.memory_mib - device_mib < 64:
            raise SystemExit("gpu-shared requires explicit positive --device-memory-mib and at least 64 MiB for host memory")
    elif device_mib is not None:
        raise SystemExit("--device-memory-mib requires gpu-shared resources")
    ports = parse_ports(getattr(args, "ports", []) or [])
    if not 64 <= args.memory_mib <= NODE_MEMORY_MIB_MAX:
        raise SystemExit(f"--memory-mib must be 64..{NODE_MEMORY_MIB_MAX}, including child processes")
    with transaction() as state:
        if any(j["id"] == args.id for j in state["jobs"] + state["results"]):
            raise SystemExit("ID already exists; use a new ID for each attempt")
        after = args.after.split(",") if args.after else []
        known = {j["id"] for j in state["jobs"] + state["results"]}
        if len(set(after)) != len(after) or any(dep not in known for dep in after):
            raise SystemExit("dependencies must be distinct existing job IDs; submit parents first")
        state["jobs"].append(dict(id=args.id, nodes=nodes, cmd=args.cmd or "", cwd=args.cwd or "$HOME",
            resources=args.resources, ttl_minutes=ttl, memory_mib=args.memory_mib, device_memory_mib=device_mib or 0, ports=ports, per_node=args.per_node,
            priority=args.priority, kind=args.kind, after=after,
            submitted_by=args.by, notes=args.notes, state="queued", submitted_at=time.time()))
    print("queued " + args.id)

def cmd_list(args):
    with snapshot() as state:
        jobs = state["jobs"] + (state["results"] if args.all else [])
    print(json.dumps(jobs, indent=2))

def cmd_status(args):
    with snapshot() as state:
        jobs = state["jobs"] + state["results"]
    job = next((j for j in jobs if j["id"] == args.id), None)
    if job is None:
        raise SystemExit("unknown ID")
    print(json.dumps(job, indent=2))

def cmd_done(args):
    with transaction() as state:
        job = next((j for j in state["jobs"] if j["id"] == args.id), None)
        if job is None:
            raise SystemExit("unknown ID")
        if job["state"] == "legacy-review":
            raise SystemExit("legacy process requires operator cleanup before migration")
        if job["state"] in ACTIVE:
            job.update(state="stopping", exit=args.exit)
        else:
            finish(state, job, args.exit, "manual completion")
    print("completion requested " + args.id)

def cmd_cancel(args):
    args.exit = 125
    cmd_done(args)

def cmd_reserve(args):
    validate_ttl(args.ttl_min)
    if not re.fullmatch(r"spark[0-9a-f]", args.node):
        raise SystemExit("invalid node")
    with transaction() as state:
        hold = dict(id="manual:" + args.holder, nodes=[args.node], resources=args.resources,
                    acquired_at=time.time(), ttl_minutes=args.ttl_min)
        active = [j for j in state["jobs"] if j["state"] in ACTIVE] + state["manual"] + state["persistent"]
        if args.node in state["fences"] or any(conflicts(hold, j) for j in active):
            raise SystemExit("node/resource is held or fenced")
        state["manual"].append(hold)

def cmd_release(args):
    with transaction() as state:
        state["manual"] = [j for j in state["manual"] if not (
            (args.node and args.node in j["nodes"]) or (args.id and args.id == j["id"]))]
    print("manual reservations released; use cancel for running jobs")


def cmd_track(args):
    if not re.fullmatch(r"spark[0-9a-f]", args.node) or not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", args.unit):
        raise SystemExit("track requires a spark node and a service unit name")
    owner = dict(id="persistent:" + args.node + ":" + args.scope + ":" + args.unit,
                 nodes=[args.node], resources="gpu-shared", unit=args.unit, scope=args.scope,
                 ports=parse_ports(args.ports))
    report = observe(args.node, [owner])
    unit = report.get("units", {}).get(owner_key(owner), {})
    try:
        maximum = int(unit["MemoryMax"])
    except (KeyError, ValueError, TypeError):
        raise SystemExit("persistent unit must have a verified finite MemoryMax")
    if unit.get("LoadState") != "loaded" or unit.get("ActiveState") != "active" or not unit.get("InvocationID") or not unit.get("ControlGroup", "").startswith("/") or unit["ControlGroup"] == "/" or not 0 < maximum <= NODE_MEMORY_MIB_MAX * MIB:
        raise SystemExit("persistent unit identity, active state or memory bound is invalid")
    if args.device_memory_mib <= 0 or math.ceil(maximum / MIB) + args.device_memory_mib > NODE_MEMORY_MIB_MAX:
        raise SystemExit("persistent device budget must be positive and total memory must fit node policy")
    owner.update(memory_mib=math.ceil(maximum / MIB) + args.device_memory_mib, device_memory_mib=args.device_memory_mib, host_memory_bytes=maximum, invocation=unit["InvocationID"],
                 control_group=unit["ControlGroup"])
    with transaction() as state:
        if any(job.get("state") in ACTIVE and owner_key(job) == owner_key(owner) and args.node in job["nodes"] for job in state["jobs"]):
            raise SystemExit("queue-owned units cannot also be tracked as persistent owners")
        previous = next((value for value in state["persistent"] if value["id"] == owner["id"]), None)
        if previous is not None and previous != owner:
            raise SystemExit("tracked identity/budget/ports changed; confirm stop and untrack before replacing")
        state["persistent"] = [value for value in state["persistent"] if value["id"] != owner["id"]] + [owner]
    print(json.dumps(owner, indent=2))

def cmd_untrack(args):
    with snapshot() as state:
        owner = next((value for value in state["persistent"] if value["id"] == args.id), None)
    if owner is None:
        raise SystemExit("unknown persistent owner ID")
    report = observe(owner["nodes"][0], [owner])
    unit = report.get("units", {}).get(owner_key(owner), {})
    group = owner["control_group"]
    if report.get("error") or not (unit.get("LoadState") == "not-found" or unit.get("ActiveState") in {"inactive", "failed"}) or any(process["control_group"] == group or process["control_group"].startswith(group + "/") for process in report.get("gpu_processes", [])):
        raise SystemExit("persistent owner must be verifiably stopped before untracking")
    with transaction() as state:
        state["persistent"] = [value for value in state["persistent"] if value != owner]
    print("persistent owner stopped and untracked " + args.id)


def cmd_preflight(args):
    nodes = args.nodes.split(",")
    if len(set(nodes)) != len(nodes) or any(not re.fullmatch(r"spark[0-9a-f]", node) for node in nodes):
        raise SystemExit("nodes must be distinct spark0..sparkf aliases")
    if not 64 <= args.memory_mib <= NODE_MEMORY_MIB_MAX or not 0 < args.device_memory_mib <= args.memory_mib - 64:
        raise SystemExit("invalid total/device memory budget")
    job = dict(id="preflight", nodes=nodes, state="queued", resources="gpu-shared",
               memory_mib=args.memory_mib, device_memory_mib=args.device_memory_mib, ports=parse_ports(args.ports))
    with snapshot() as state:
        held = [owner for owner in state["jobs"] if owner.get("state") in ACTIVE] + state["persistent"] + state["manual"]
    observations = resource_observations(state | {"jobs": state["jobs"] + [job]})
    errors = {}
    for node in nodes:
        candidate = job | {"nodes": [node]}
        conflict = next((owner["id"] for owner in held if conflicts(candidate, owner)), None)
        reason = ("node is fenced" if node in state["fences"] else "conflicting owner: " + conflict if conflict else
                  shared_admission(candidate, held, observations))
        if reason:
            errors[node] = reason
    print(json.dumps({"admissible": not errors, "errors": errors, "requested": job,
        "host_headroom_mib": HOST_HEADROOM_MIB, "owners": held, "nodes": observations}, indent=2))
    if errors:
        raise SystemExit(1)

def cmd_doctor(args):
    with snapshot() as state:
        report = {"state": str(STATE), "version": state["version"], "fences": state["fences"],
                  "dispatcher_age_seconds": time.time() - state.get("dispatcher_at", 0),
                  "manual": state["manual"], "persistent": state["persistent"], "active": [j["id"] for j in state["jobs"] if j["state"] in ACTIVE]}
    print(json.dumps(report, indent=2))

def cmd_sync(args):
    valid_name(args.id)
    nodes = args.nodes.split(",")
    if len(set(nodes)) != len(nodes) or any(not re.fullmatch(r"spark[0-9a-f]", n) for n in nodes):
        raise SystemExit("invalid nodes")
    root = Path(__file__).resolve().parents[1]
    def git(*argv):
        return subprocess.check_output(["git", "-C", str(root), *argv], text=True).strip()
    sha = git("rev-parse", "--verify", "--end-of-options", args.ref + "^{commit}")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit("sync requires an exact local commit")
    relative = "srcdata/sparkqueue/" + args.id + "/" + sha
    with tempfile.TemporaryDirectory(prefix="sparkqueue-source-") as tmp:
        checkout = Path(tmp) / "source"
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout",
                        str(root), str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "fetch", "--quiet", "--no-tags",
                        "--update-shallow", str(root), sha], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", "--detach", sha], check=True)
        subprocess.run(["git", "-C", str(checkout), "remote", "set-url", "origin",
                        "https://github.com/sparkpipe/sparkpipe"], check=True)
        for node in nodes:
            final = "/home/" + node + "/" + relative
            partial = final + ".partial-" + uuid.uuid4().hex
            rc, _ = ssh(node, "test ! -e " + shlex.quote(final) + " && mkdir -p " + shlex.quote(partial))
            if rc != 0:
                raise SystemExit("destination exists or is unreachable: " + node + ":" + final)
            subprocess.run(["rsync", "-a", "--checksum", "-e", "ssh " + shlex.join(SSH_OPTS),
                            str(checkout) + "/", node + ":" + partial + "/"], check=True)
            verify = (f"test $(git -C {shlex.quote(partial)} rev-parse HEAD) = {sha} && "
                      f"git -C {shlex.quote(partial)} diff --quiet HEAD && "
                      f"mv -T {shlex.quote(partial)} {shlex.quote(final)}")
            rc, _ = ssh(node, verify)
            if rc != 0:
                raise SystemExit("source verification failed: " + node + ":" + partial)
    print(json.dumps({"git_commit": sha, "source_ref": args.ref, "nodes": nodes, "cwd": "/home/{host}/" + relative}))

def cmd_serve(args):
    while True:
        try:
            cmd_dispatch(args)
        except (OSError, ValueError) as error:
            print(f"dispatcher error: {error}", file=sys.stderr, flush=True)
        sys.stdout.flush()
        time.sleep(5)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True)
    a = sub.add_parser("add")
    a.add_argument("--id", required=True)
    a.add_argument("--nodes", required=True)
    a.add_argument("--cmd")
    a.add_argument("--cmd-file", type=Path)
    a.add_argument("--per-node", action="store_true")
    a.add_argument("--memory-mib", type=int)
    a.add_argument("--device-memory-mib", type=int)
    a.add_argument("--ports", action="append", default=[])
    a.add_argument("--resources", choices=["gpu", "gpu-shared", "cpu", "exclusive"], default="gpu")
    a.add_argument("--cwd")
    a.add_argument("--priority", type=int, default=5)
    a.add_argument("--kind", choices=["run", "gate", "note"], default="run")
    a.add_argument("--after")
    a.add_argument("--klass", choices=["short", "long"], default="short", help="compatibility only; all jobs have deadlines")
    a.add_argument("--by", default="coordinator")
    a.add_argument("--notes", default="")
    a.add_argument("--ttl-min", type=float)
    a.set_defaults(fn=cmd_add)
    a = sub.add_parser("list")
    a.add_argument("--all", action="store_true")
    a.set_defaults(fn=cmd_list)
    for name, fn in [("status", cmd_status), ("done", cmd_done), ("cancel", cmd_cancel)]:
        a = sub.add_parser(name)
        a.add_argument("--id", required=True)
        if name == "done":
            a.add_argument("--exit", type=int, default=0)
        a.set_defaults(fn=fn)
    for name in ["dispatch", "schedule"]:
        a = sub.add_parser(name)
        a.add_argument("--ttl", type=float, default=3.0)
        a.set_defaults(fn=cmd_dispatch)
    a = sub.add_parser("serve")
    a.add_argument("--ttl", type=float, default=3.0)
    a.set_defaults(fn=cmd_serve)
    a = sub.add_parser("reserve")
    a.add_argument("--node", required=True)
    a.add_argument("--holder", required=True)
    a.add_argument("--ttl-min", type=float, default=3.0)
    a.add_argument("--resources", choices=["gpu", "cpu", "exclusive"], default="gpu")
    a.set_defaults(fn=cmd_reserve)
    a = sub.add_parser("release")
    group = a.add_mutually_exclusive_group(required=True)
    group.add_argument("--node")
    group.add_argument("--id")
    a.set_defaults(fn=cmd_release)
    a = sub.add_parser("track")
    a.add_argument("--node", required=True)
    a.add_argument("--unit", required=True)
    a.add_argument("--device-memory-mib", type=int, required=True)
    a.add_argument("--scope", choices=["system", "user"], default="system")
    a.add_argument("--ports", action="append", default=[])
    a.set_defaults(fn=cmd_track)
    a = sub.add_parser("untrack")
    a.add_argument("--id", required=True)
    a.set_defaults(fn=cmd_untrack)
    a = sub.add_parser("preflight")
    a.add_argument("--nodes", required=True)
    a.add_argument("--memory-mib", type=int, required=True)
    a.add_argument("--device-memory-mib", type=int, required=True)
    a.add_argument("--ports", action="append", default=[])
    a.set_defaults(fn=cmd_preflight)
    a = sub.add_parser("sync")
    a.add_argument("--id", required=True)
    a.add_argument("--nodes", required=True)
    a.add_argument("--ref", default="HEAD", help="Local commit or branch to test; defaults to committed HEAD")
    a.set_defaults(fn=cmd_sync)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
