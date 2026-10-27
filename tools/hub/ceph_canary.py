#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

MARKER = "ceph_canary_probe"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
PROBE = r'''
import json, mmap, os, sys, time
root, subdir, host, size, mounts = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
start = time.monotonic()
stage = "mount"


def say(**fields):
    print(json.dumps(dict(fields, stage=stage, t=round(time.monotonic() - start, 3))), flush=True)


def enter(name):
    global stage
    stage = name
    say(event="stage")


try:
    enter("mount")
    with open(mounts) as handle:
        table = [line.split() for line in handle]
    if not any(len(fields) > 2 and fields[1] == root and fields[2] == "ceph" for fields in table):
        say(event="error", error=f"{root} is not a ceph mount")
        sys.exit(0)
    enter("metadata")
    directory = os.path.join(root, subdir, host)
    os.makedirs(directory, exist_ok=True)
    os.listdir(directory)
    path = os.path.join(directory, "probe")
    data = os.urandom(size)
    buffer = mmap.mmap(-1, size)
    buffer.write(data)
    enter("write")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    try:
        written = os.write(fd, buffer)
        os.fsync(fd)
    finally:
        os.close(fd)
    if written != size:
        say(event="error", error=f"wrote {written} of {size} bytes")
        sys.exit(0)
    enter("read")
    back = mmap.mmap(-1, size)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        got = os.readv(fd, [back])
    finally:
        os.close(fd)
    if got != size or back[:] != data:
        say(event="error", error=f"read back {got} of {size} bytes and they differ from what was written")
        sys.exit(0)
    enter("unlink")
    os.unlink(path)
    say(event="ok")
except Exception as error:
    say(event="error", error=f"{type(error).__name__}: {error}")
'''


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Canary:
    def __init__(self, config):
        for key in ("hosts", "local_host", "ssh", "root", "subdir", "mounts", "size_bytes", "interval_seconds", "probe_seconds",
                    "slow_seconds", "rounds_to_alert", "health_hosts", "health_command", "health_seconds", "ignore_checks",
                    "entity_checks", "state_file", "event_log"):
            if key not in config:
                raise SystemExit(f"ceph_canary: the config needs {key}")
        self.hosts = list(config["hosts"])
        self.local = config["local_host"]
        self.ssh = list(config["ssh"])
        self.root = config["root"]
        self.subdir = config["subdir"]
        self.mounts = config["mounts"]
        self.size = int(config["size_bytes"])
        if self.size <= 0 or self.size % 4096:
            raise SystemExit("ceph_canary: size_bytes must be a positive multiple of 4096 for direct I/O")
        self.interval = float(config["interval_seconds"])
        self.probe_seconds = float(config["probe_seconds"])
        self.slow = float(config["slow_seconds"])
        self.rounds = int(config["rounds_to_alert"])
        if self.rounds < 1:
            raise SystemExit("ceph_canary: rounds_to_alert must be at least 1")
        self.health_hosts = list(config["health_hosts"])
        self.health_command = config["health_command"]
        self.health_seconds = float(config["health_seconds"])
        self.ignore = {name: set(messages) for name, messages in config["ignore_checks"].items()}
        self.entity = set(config["entity_checks"])
        self.state_file = Path(config["state_file"])
        self.event_log = Path(config["event_log"])
        self.streak = {}
        self.active = {}
        self.round = 0
        self.summary_at = 0.0

    def probe_command(self, host):
        arguments = " ".join(shlex.quote(str(value)) for value in (self.root, self.subdir, host, self.size, self.mounts))
        pattern = f"[{MARKER[0]}]{MARKER[1:]}"
        return (f"pids=$(pgrep -f '{pattern}' | paste -sd, -); "
                f"if [ -n \"$pids\" ]; then echo '{{\"event\": \"busy\"}}'; ps -o pid=,etimes=,stat=,wchan:32= -p \"$pids\"; exit 0; fi; "
                f"m={MARKER[:-6]}; exec python3 -I - \"${{m}}_probe\" {arguments}")

    def argv(self, host, command):
        return ["bash", "-c", command] if host == self.local else self.ssh + [host, command]

    async def capture(self, argv, timeout, stdin=b""):
        process = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        lines = []

        async def collect():
            process.stdin.write(stdin)
            try:
                await process.stdin.drain()
            except ConnectionError:
                pass
            process.stdin.close()
            async for raw in process.stdout:
                lines.append(raw.decode("utf-8", "replace").rstrip("\n"))
            await process.wait()

        try:
            await asyncio.wait_for(collect(), timeout)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                pass
            return None, lines
        return process.returncode, lines

    def judge(self, host, code, lines):
        events = []
        other = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                other.append(line)
                continue
            if isinstance(event, dict):
                events.append(event)
            else:
                other.append(line)
        tail = " | ".join(other)[-300:]
        if code is None:
            stage = events[-1]["stage"] if events else "ssh"
            return {"status": "hang", "stage": stage, "detail": f"no answer within {self.probe_seconds:g} s during {stage}"}
        if any(event.get("event") == "busy" for event in events):
            return {"status": "stuck", "stage": "busy", "detail": f"an earlier probe is still running: {tail}"}
        final = next((event for event in reversed(events) if event.get("event") in ("ok", "error")), None)
        if final is None:
            return {"status": "unreachable", "stage": "ssh", "detail": f"exit {code}: {tail or 'no output'}"}
        stamps = {event["stage"]: event["t"] for event in events if event.get("event") == "stage"}
        order = ["mount", "metadata", "write", "read", "unlink"]
        stages = {}
        for name, after in zip(order, order[1:] + [None]):
            if name in stamps:
                stages[name] = round((stamps[after] if after in stamps else final["t"]) - stamps[name], 3)
        if final["event"] == "error":
            return {"status": "error", "stage": final["stage"], "detail": final["error"], "stages": stages, "seconds": final["t"]}
        return {"status": "ok", "stage": "done", "stages": stages, "seconds": final["t"]}

    async def probe(self, host):
        code, lines = await self.capture(self.argv(host, self.probe_command(host)), self.probe_seconds, PROBE.encode())
        result = self.judge(host, code, lines)
        problems = {}
        if result["status"] == "ok":
            if result["seconds"] > self.slow:
                stages = ", ".join(f"{name} {value:g} s" for name, value in result["stages"].items())
                problems[f"host:{host}:slow"] = f"{host}: the {self.size // 1024} KiB read/write probe took {result['seconds']:g} s ({stages})"
        elif result["status"] in ("error", "hang"):
            problems[f"host:{host}:{result['stage']}"] = f"{host}: {result['status']} at {result['stage']}: {result['detail']}"
        else:
            problems[f"host:{host}:{result['status']}"] = f"{host}: {result['status']}: {result['detail']}"
        return host, result, problems

    def health_problems(self, document):
        problems = {}
        ignored = []
        for name, check in sorted((document.get("checks") or {}).items()):
            details = [entry.get("message", "").strip() for entry in check.get("detail") or []]
            messages = details or [check.get("summary", {}).get("message", name)]
            if check.get("muted") or (name in self.ignore and set(messages) <= self.ignore[name]):
                ignored.append(name)
                continue
            if name in self.entity and details:
                for detail in details:
                    problems[f"cluster:{name}:{detail}"] = f"ceph {check.get('severity')}: {detail}"
            else:
                summary = check.get("summary", {}).get("message", name)
                problems[f"cluster:{name}"] = f"ceph {check.get('severity')}: {summary}"
        return problems, ignored

    async def health(self):
        failures = []
        for host in self.health_hosts:
            code, lines = await self.capture(self.argv(host, self.health_command), self.health_seconds)
            text = "\n".join(lines)
            if code == 0:
                try:
                    document = json.loads(text)
                except json.JSONDecodeError:
                    failures.append(f"{host} printed no JSON")
                    continue
                problems, ignored = self.health_problems(document)
                return {"status": document.get("status"), "from": host, "ignored": ignored}, problems
            failures.append(f"{host} {'timed out' if code is None else f'exit {code}'}: {text[-120:]}")
        return {"status": "unknown", "from": None, "ignored": []}, {"cluster:unreachable": "ceph health unavailable: " + "; ".join(failures)}

    def record(self, entry):
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log.open("a") as handle:
            handle.write(json.dumps(dict(entry, utc=now_utc())) + "\n")

    def advance(self, seen):
        for key in seen:
            self.streak[key] = self.streak.get(key, 0) + 1
        for key in [key for key in self.streak if key not in seen]:
            del self.streak[key]
        for key in [key for key in self.active if key not in seen]:
            problem = self.active.pop(key)
            self.record({"event": "cleared", "key": key, "message": problem["message"], "since": problem["since"]})
        for key, message in seen.items():
            if self.streak[key] >= self.rounds:
                if key not in self.active:
                    self.active[key] = {"message": message, "since": now_utc()}
                    self.record({"event": "problem", "key": key, "message": message})
                else:
                    self.active[key]["message"] = message

    async def tick(self):
        self.round += 1
        started = time.time()
        outcomes = await asyncio.gather(*(self.probe(host) for host in self.hosts), self.health())
        cluster, health_problems = outcomes[-1]
        seen = dict(health_problems)
        hosts = {}
        for host, result, problems in outcomes[:-1]:
            hosts[host] = result
            seen.update(problems)
        self.advance(seen)
        state = {"utc": now_utc(), "epoch": time.time(), "round": self.round, "round_seconds": round(time.time() - started, 1),
                 "interval_seconds": self.interval, "rounds_to_alert": self.rounds, "ok": not self.active,
                 "active": self.active, "pending": {key: self.streak[key] for key in seen if key not in self.active},
                 "hosts": hosts, "cluster": cluster}
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_name(self.state_file.name + ".tmp")
        temporary.write_text(json.dumps(state, indent=1) + "\n")
        os.replace(temporary, self.state_file)
        if time.time() - self.summary_at >= 3600:
            self.summary_at = time.time()
            self.record({"event": "summary", "ok": not self.active, "active": sorted(self.active),
                         "hosts_ok": sum(1 for result in hosts.values() if result["status"] == "ok"), "hosts": len(hosts)})
        return state

    async def run(self, rounds=None):
        self.record({"event": "start", "hosts": self.hosts, "interval_seconds": self.interval})
        while rounds is None or self.round < rounds:
            began = time.monotonic()
            await self.tick()
            if rounds is not None and self.round >= rounds:
                break
            await asyncio.sleep(max(0.0, self.interval - (time.monotonic() - began)))


def read_state(arguments):
    command = SSH + [arguments.hub, "cat", arguments.state] if arguments.hub else ["cat", arguments.state]
    done = subprocess.run(command, capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
    if done.returncode != 0:
        raise OSError(f"reading the canary state failed: {done.stderr.decode(errors='replace').strip()[-200:]}")
    return json.loads(done.stdout)


def load_ack(path):
    try:
        return set(json.loads(Path(path).read_text())["keys"])
    except FileNotFoundError:
        return set()


def watch(arguments):
    acknowledged = load_ack(arguments.ack)
    failures = 0
    while True:
        try:
            state = read_state(arguments)
            failures = 0
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            failures += 1
            if failures >= arguments.failures:
                print(json.dumps({"event": "watch_failed", "error": str(error)}), flush=True)
                return 2
            time.sleep(arguments.poll)
            continue
        age = time.time() - float(state["epoch"])
        if age > 3 * float(state["interval_seconds"]) + 120:
            print(json.dumps({"event": "canary_stale", "last_round_utc": state["utc"], "age_seconds": round(age)}), flush=True)
            return 3
        active = state["active"]
        new = sorted(set(active) - acknowledged)
        cleared = sorted(acknowledged - set(active))
        if new or cleared:
            for key in new:
                print(json.dumps({"event": "problem", "key": key, "message": active[key]["message"], "since": active[key]["since"]}), flush=True)
            for key in cleared:
                print(json.dumps({"event": "cleared", "key": key}), flush=True)
            return 1
        time.sleep(arguments.poll)


def acknowledge(arguments):
    state = read_state(arguments)
    path = Path(arguments.ack)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"keys": sorted(state["active"]), "utc": now_utc()}, indent=1) + "\n")
    print(json.dumps({"acknowledged": sorted(state["active"])}))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Read/write canary for the CephFS mount on the hub and every Spark, plus cluster health.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="probe every host each interval and keep the state file and event log")
    run.add_argument("--config", required=True)
    run.add_argument("--rounds", type=int, default=None)
    for name in ("watch", "ack"):
        sub = commands.add_parser(name, help="wait until the active problems differ from the acknowledged ones" if name == "watch" else "acknowledge the active problems")
        sub.add_argument("--hub", default="rtx5090")
        sub.add_argument("--state", default="/srv/workspace/sparkpipe/ceph-canary/state.json")
        sub.add_argument("--ack", required=True)
        sub.add_argument("--poll", type=float, default=60)
        sub.add_argument("--failures", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.command == "run":
        canary = Canary(json.loads(Path(arguments.config).read_text()))
        asyncio.run(canary.run(arguments.rounds))
        return 0
    return watch(arguments) if arguments.command == "watch" else acknowledge(arguments)


if __name__ == "__main__":
    sys.exit(main())
