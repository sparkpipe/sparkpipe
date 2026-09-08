#!/usr/bin/env python3
"""Durable Spark queue; see docs/PARALLEL_DRIVER_DEBUG.md."""
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
    """Preserve legacy receipts, fail closed on legacy running jobs."""
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


@contextmanager
def transaction():
    # Network operations must never occur in this transaction.
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = STATE / "state-v2.json"
        state = read_json(path, None)
        if state is None:
            state = migrate()
        yield state
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


def ssh(node, command, timeout=12):
    try:
        out = subprocess.run(["ssh", *SSH_OPTS, node, command], capture_output=True, text=True, timeout=timeout)
        return out.returncode, out.stdout.strip()
    except subprocess.TimeoutExpired:
        return 124, ""


def remote(job, node, action):
    """Retained units reconcile lost launch ACKs without duplicate execution."""
    unit = "sparkqueue-" + job["attempt"]
    ctl = "sudo -n systemctl"
    show = f"{ctl} show {unit} -p LoadState -p ActiveState -p SubState -p ExecMainStatus -p Result"
    if action == "stop":
        command = (f'if [ "$({ctl} show {unit} -p LoadState --value)" != not-found ]; '
                   f"then {ctl} stop {unit} >/dev/null 2>&1 || exit 1; fi; {show}")
    elif action == "launch":
        remaining = max(1, math.ceil(job["deadline"] - time.time()))
        cwd = (job.get("cwd") or "$HOME").replace("{host}", node)
        cd = 'cd "$HOME"' if cwd == "$HOME" else "cd " + shlex.quote(cwd)
        inner = cd + " && exec bash -c " + shlex.quote(job["cmd"])
        argv = ["sudo", "-n", "systemd-run", "--quiet", "--unit=" + unit,
                "--uid=" + node, "--property=Type=exec", "--property=RemainAfterExit=yes",
                "--property=KillMode=control-group", "--property=TimeoutStopSec=5",
                "--property=RuntimeMaxSec=" + str(remaining),
                "--property=MemoryMax=" + str(job.get("memory_mib", 8192)) + "M",
                "--property=LimitMEMLOCK=" + str(job.get("memory_mib", 8192)) + "M",
                "--property=MemorySwapMax=0", "--property=TasksMax=512",
                "--property=StandardOutput=append:/tmp/" + unit + ".log",
                "--property=StandardError=inherit", "--setenv=SPARK_QUEUE_RANK=" + str(job["nodes"].index(node)),
                "--setenv=SPARK_QUEUE_SIZE=" + str(len(job["nodes"])),
                "--setenv=SPARK_QUEUE_ID=" + job["id"], "bash", "-c", inner]
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
    return bool((set(left["nodes"]) - set(left.get("released_nodes", []))) &
                (set(right["nodes"]) - set(right.get("released_nodes", [])))) and (
        left.get("resources", "gpu") == right.get("resources", "gpu") or
        "exclusive" in {left.get("resources"), right.get("resources")})


def memory_available(job, held):
    for node in job["nodes"]:
        used = sum(owner.get("memory_mib", 0) for owner in held
                   if node in owner["nodes"] and node not in owner.get("released_nodes", []))
        if used + job.get("memory_mib", 8192) > 114688:
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
                # Poll/launch racing cancellation is not a stop acknowledgement.
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


def claim(default_ttl):
    with transaction() as state:
        now = time.time()
        state["manual"] = [r for r in state["manual"] if now < timestamp(r["acquired_at"]) + r["ttl_minutes"] * 60]
        done = {r["id"] for r in state["results"] if r.get("exit") == 0}
        held = [j for j in state["jobs"] if j["state"] in ACTIVE] + state["manual"]
        unavailable = set(state["fences"])
        for owner in held:
            if owner.get("state") == "stopping":
                unavailable.update(set(owner["nodes"]) - set(owner.get("released_nodes", [])))
        candidates = []
        for job in state["jobs"]:
            if job.get("state") not in {"queued", "blocked"} or job.get("kind", "run") != "run":
                continue
            try:
                ttl = validate_ttl(float(job.get("ttl_minutes") or default_ttl))
                valid = job.get("cmd", "").strip() and (len(job["nodes"]) == 1 or job.get("per_node"))
            except (ValueError, TypeError, SystemExit):
                valid = False
            if not valid:
                job.update(state="invalid", error="legacy command/window requires resubmission")
                continue
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
            ttl = validate_ttl(float(job.get("ttl_minutes") or default_ttl))
            job.update(state="launching", attempt=uuid.uuid4().hex, deadline=now + ttl * 60,
                       dispatched_at=now, ttl_minutes=ttl)
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
            snapshot = [dict(j) for j in state["jobs"]]
        reconcile(snapshot)
        jobs = claim(ttl)
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
    if not 64 <= args.memory_mib <= 114688:
        raise SystemExit("--memory-mib must be 64..114688, including child processes")
    with transaction() as state:
        if any(j["id"] == args.id for j in state["jobs"] + state["results"]):
            raise SystemExit("ID already exists; use a new ID for each attempt")
        state["jobs"].append(dict(id=args.id, nodes=nodes, cmd=args.cmd or "", cwd=args.cwd or "$HOME",
            resources=args.resources, ttl_minutes=ttl, memory_mib=args.memory_mib, per_node=args.per_node,
            priority=args.priority, kind=args.kind, after=(args.after.split(",") if args.after else []),
            submitted_by=args.by, notes=args.notes, state="queued", submitted_at=time.time()))
    print("queued " + args.id)


def cmd_list(args):
    with transaction() as state:
        jobs = state["jobs"] + (state["results"] if args.all else [])
    print(json.dumps(jobs, indent=2))


def cmd_status(args):
    with transaction() as state:
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
        active = [j for j in state["jobs"] if j["state"] in ACTIVE] + state["manual"]
        if args.node in state["fences"] or any(conflicts(hold, j) for j in active):
            raise SystemExit("node/resource is held or fenced")
        state["manual"].append(hold)


def cmd_release(args):
    with transaction() as state:
        state["manual"] = [j for j in state["manual"] if not (
            (args.node and args.node in j["nodes"]) or (args.id and args.id == j["id"]))]
    print("manual reservations released; use cancel for running jobs")


def cmd_doctor(args):
    with transaction() as state:
        report = {"state": str(STATE), "version": state["version"], "fences": state["fences"],
                  "dispatcher_age_seconds": time.time() - state.get("dispatcher_at", 0),
                  "manual": state["manual"], "active": [j["id"] for j in state["jobs"] if j["state"] in ACTIVE]}
    print(json.dumps(report, indent=2))


def cmd_sync(args):
    """Rsync a clean main checkout into an immutable, lane-owned directory."""
    valid_name(args.id)
    nodes = args.nodes.split(",")
    if len(set(nodes)) != len(nodes) or any(not re.fullmatch(r"spark[0-9a-f]", n) for n in nodes):
        raise SystemExit("invalid nodes")
    root = Path(__file__).resolve().parents[1]
    def git(*argv):
        return subprocess.check_output(["git", "-C", str(root), *argv], text=True).strip()
    sha = git("rev-parse", "HEAD")
    if sha != git("rev-parse", "origin/main") or git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("sync requires clean HEAD == origin/main; merge and pull first")
    relative = "srcdata/sparkqueue/" + args.id + "/" + sha
    with tempfile.TemporaryDirectory(prefix="sparkqueue-source-") as tmp:
        checkout = Path(tmp) / "source"
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout",
                        str(root), str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", "-B", "main", sha], check=True)
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
    print(json.dumps({"git_commit": sha, "nodes": nodes, "cwd": "/home/{host}/" + relative}))


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
    a.add_argument("--memory-mib", type=int, default=8192)
    a.add_argument("--resources", choices=["gpu", "cpu", "exclusive"], default="gpu")
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
    a = sub.add_parser("sync")
    a.add_argument("--id", required=True)
    a.add_argument("--nodes", required=True)
    a.set_defaults(fn=cmd_sync)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
