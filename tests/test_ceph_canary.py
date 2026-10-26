#!/usr/bin/env python3
"""The Ceph canary writes, reads back and deletes a file with direct I/O on every host's CephFS mount and reads cluster health: a host that is not mounted, fails a stage, hangs past the probe limit, still has an earlier probe stuck, cannot be reached or is slow becomes a problem; a probe that hangs is cut at the limit; problems alert only after the configured number of consecutive rounds and clear in the first round without them; known Ceph warnings are ignored only by their exact messages, so a new entry of the same check alerts; OSDs and hosts that go down are separate problems; health falls back to the next monitor host; and the watcher exits on a new or cleared problem and on a stale canary."""
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "hub" / "ceph_canary.py"
SPEC = importlib.util.spec_from_file_location("ceph_canary", SCRIPT)
ceph_canary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ceph_canary)
SHIPPED = json.loads((ROOT / "deployment" / "hub" / "ceph_canary.json").read_text())
FAKE_SSH = r'''
import json, os, sys, time
sys.stdin.read()
host, command = sys.argv[-2], sys.argv[-1]
plan = json.load(open(os.environ["FAKE_PLAN"]))
open(os.environ["FAKE_LOG"], "a").write(json.dumps([host, command]) + "\n")
def stages(names, last):
    for index, name in enumerate(names):
        print(json.dumps({"event": "stage", "stage": name, "t": index * last / max(1, len(names))}), flush=True)
if "ceph health" in command:
    health = plan["health"].get(host)
    if health is None:
        print("ceph: monitor unreachable")
        sys.exit(1)
    print(json.dumps(health))
    sys.exit(0)
kind = plan["hosts"].get(host, "ok")
if kind == "ok":
    stages(["mount", "metadata", "write", "read", "unlink"], 0.2)
    print(json.dumps({"event": "ok", "stage": "unlink", "t": 0.2}))
elif kind == "slow":
    stages(["mount", "metadata", "write", "read", "unlink"], 40.0)
    print(json.dumps({"event": "ok", "stage": "unlink", "t": 40.0}))
elif kind == "error_write":
    stages(["mount", "metadata", "write"], 0.1)
    print(json.dumps({"event": "error", "stage": "write", "t": 0.1, "error": "OSError: [Errno 5] Input/output error"}))
elif kind == "unmounted":
    stages(["mount"], 0.0)
    print(json.dumps({"event": "error", "stage": "mount", "t": 0.0, "error": "/mnt/model-warm is not a ceph mount"}))
elif kind == "hang":
    stages(["mount", "metadata", "write"], 0.1)
    time.sleep(60)
elif kind == "busy":
    print(json.dumps({"event": "busy"}))
    print("  4242   3600 D    ceph_osdc_wait_request")
elif kind == "down":
    print("ssh: connect to host " + host + " port 22: No route to host", file=sys.stderr)
    sys.exit(255)
'''
NOISE = {name: {"severity": "HEALTH_WARN", "summary": {"message": f"{name} summary", "count": len(messages)},
                "detail": [{"message": message} for message in messages], "muted": False}
         for name, messages in SHIPPED["ignore_checks"].items() if name != "AUTH_INSECURE_KEYS_CREATABLE"}
NOISE["AUTH_INSECURE_KEYS_CREATABLE"] = {"severity": "HEALTH_WARN", "summary": {"message": "Monitors are configured to allow creation of insecure key types", "count": 1},
                                         "detail": [], "muted": False}


def health(down_osds, extra=None):
    checks = dict(NOISE)
    if down_osds:
        checks["OSD_DOWN"] = {"severity": "HEALTH_WARN", "summary": {"message": f"{len(down_osds)} osds down", "count": len(down_osds)},
                              "detail": [{"message": f"osd.{osd} (root=default,host=spark{osd:x}) is down"} for osd in down_osds], "muted": False}
        checks["PG_DEGRADED"] = {"severity": "HEALTH_WARN", "summary": {"message": f"Degraded data redundancy: {time.time()} objects degraded", "count": 276},
                                 "detail": [{"message": f"pg 5.{index:x} is active+undersized+degraded"} for index in range(3)], "muted": False}
    checks.update(extra or {})
    return {"status": "HEALTH_WARN", "checks": checks, "mutes": []}


def real_probe(failures, tmp):
    mount = tmp / "mnt"
    (mount / ".ceph-canary").mkdir(parents=True)
    table = tmp / "mounts"
    config = dict(SHIPPED, hosts=["local"], local_host="local", root=str(mount), mounts=str(table), probe_seconds=20,
                  state_file=str(tmp / "real-state.json"), event_log=str(tmp / "real-events.jsonl"))
    canary = ceph_canary.Canary(config)

    def check(condition, name):
        if not condition:
            failures.append(name)

    table.write_text(f"10.0.0.1:6789:/models {mount} ceph rw,noatime 0 0\n")
    _, result, problems = asyncio.run(canary.probe("local"))
    check(result["status"] == "ok" and set(result["stages"]) == {"mount", "metadata", "write", "read", "unlink"} and not problems,
          f"the real probe writes, reads back and deletes a file: {result} {problems}")
    check(list((mount / ".ceph-canary" / "local").iterdir()) == [], "the probe leaves no file behind")
    table.write_text(f"/dev/sda1 {mount} ext4 rw 0 0\n")
    _, result, problems = asyncio.run(canary.probe("local"))
    check(result["status"] == "error" and result["stage"] == "mount" and list(problems) == ["host:local:mount"],
          f"a path that is not a ceph mount fails at mount and touches nothing: {result}")
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", ceph_canary.MARKER])
    try:
        time.sleep(0.3)
        _, result, problems = asyncio.run(canary.probe("local"))
        check(result["status"] == "stuck" and list(problems) == ["host:local:stuck"],
              f"a probe still running from an earlier round is reported, not joined by another: {result}")
    finally:
        sleeper.kill()
        sleeper.wait()
    table.write_text(f"10.0.0.1:6789:/models {mount} ceph rw 0 0\n")
    _, result, _ = asyncio.run(canary.probe("local"))
    check(result["status"] == "ok", f"the probe runs again once the earlier one is gone: {result}")


def hub_logic(failures, tmp):
    def check(condition, name):
        if not condition:
            failures.append(name)

    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "fake_ssh.py").write_text(FAKE_SSH)
    plan_path = tmp / "plan.json"
    os.environ["FAKE_PLAN"] = str(plan_path)
    os.environ["FAKE_LOG"] = str(tmp / "ssh.log")
    hosts = ["spark0", "spark1", "spark2", "spark3", "spark4", "spark5", "spark6"]
    config = dict(SHIPPED, hosts=hosts, local_host="hub-not-probed", ssh=[sys.executable, str(tmp / "fake_ssh.py")],
                  probe_seconds=2, health_seconds=5, health_hosts=["mon0", "mon1"], interval_seconds=1,
                  state_file=str(tmp / "state" / "state.json"), event_log=str(tmp / "logs" / "events.jsonl"))
    canary = ceph_canary.Canary(config)

    def plan(hosts_plan, health_plan):
        plan_path.write_text(json.dumps({"hosts": hosts_plan, "health": health_plan}))

    def events():
        return [json.loads(line) for line in Path(config["event_log"]).read_text().splitlines()] if Path(config["event_log"]).exists() else []

    bad = {"spark1": "error_write", "spark2": "hang", "spark3": "busy", "spark4": "down", "spark5": "slow", "spark6": "unmounted"}
    plan(bad, {"mon0": health([14])})
    started = time.monotonic()
    state = asyncio.run(canary.tick())
    elapsed = time.monotonic() - started
    expected_hosts = {"host:spark1:write", "host:spark2:write", "host:spark3:stuck", "host:spark4:unreachable", "host:spark5:slow", "host:spark6:mount"}
    expected_cluster = {"cluster:OSD_DOWN:osd.14 (root=default,host=sparke) is down", "cluster:PG_DEGRADED"}
    check(elapsed < config["probe_seconds"] + 4, f"a hanging host is cut at the probe limit: the round took {elapsed:.1f} s")
    check(state["active"] == {} and state["ok"] and set(state["pending"]) == expected_hosts | expected_cluster,
          f"one round only makes problems pending: {sorted(state['pending'])}")
    check(state["hosts"]["spark0"]["status"] == "ok" and state["hosts"]["spark2"]["status"] == "hang", f"per-host results are kept: {state['hosts']}")
    check(set(state["cluster"]["ignored"]) == set(SHIPPED["ignore_checks"]) and state["cluster"]["from"] == "mon0",
          f"the known warnings are ignored and visible: {state['cluster']}")
    check(not [event for event in events() if event["event"] == "problem"], "nothing alerts after one round")

    state = asyncio.run(canary.tick())
    check(set(state["active"]) == expected_hosts | expected_cluster and not state["ok"], f"the second round alerts: {sorted(state['active'])}")
    check({event["key"] for event in events() if event["event"] == "problem"} == expected_hosts | expected_cluster, "each alert is logged once")
    check("error at write" in state["active"]["host:spark1:write"]["message"] and "Input/output error" in state["active"]["host:spark1:write"]["message"],
          f"the alert names the stage and the error: {state['active']['host:spark1:write']}")

    stray = {"CEPHADM_STRAY_DAEMON": dict(NOISE["CEPHADM_STRAY_DAEMON"], detail=NOISE["CEPHADM_STRAY_DAEMON"]["detail"] + [{"message": "stray daemon osd.99 on host spark9 not managed by cephadm"}])}
    plan({"spark3": "busy"}, {"mon0": None, "mon1": health([14, 3], stray)})
    state = asyncio.run(canary.tick())
    cleared = {event["key"] for event in events() if event["event"] == "cleared"}
    check(cleared == {"host:spark1:write", "host:spark2:write", "host:spark4:unreachable", "host:spark5:slow", "host:spark6:mount"},
          f"recovered hosts clear in the first round without the problem: {sorted(cleared)}")
    check("host:spark3:stuck" in state["active"] and "cluster:OSD_DOWN:osd.14 (root=default,host=sparke) is down" in state["active"],
          "problems that persist stay active")
    check("cluster:OSD_DOWN:osd.3 (root=default,host=spark3) is down" in state["pending"] and "cluster:CEPHADM_STRAY_DAEMON" in state["pending"],
          f"a second down OSD and a new stray daemon are new problems: {sorted(state['pending'])}")
    check(state["cluster"]["from"] == "mon1", "health falls back to the next monitor host")

    plan({}, {})
    state = asyncio.run(canary.tick())
    state = asyncio.run(canary.tick())
    check("cluster:unreachable" in state["active"] and all(not key.startswith("host:") for key in state["active"]),
          f"health that no monitor host answers is a problem: {sorted(state['active'])}")

    ack = tmp / "ack.json"
    watch = [sys.executable, str(SCRIPT), "watch", "--hub", "", "--state", config["state_file"], "--ack", str(ack), "--poll", "0.1"]
    done = subprocess.run(watch, capture_output=True, text=True, timeout=30)
    printed = [json.loads(line) for line in done.stdout.splitlines()]
    check(done.returncode == 1 and {line["key"] for line in printed if line["event"] == "problem"} == set(state["active"]),
          f"the watcher reports problems nobody acknowledged: {done.returncode} {done.stdout[-300:]}")
    done = subprocess.run([sys.executable, str(SCRIPT), "ack", "--hub", "", "--state", config["state_file"], "--ack", str(ack)], capture_output=True, text=True, timeout=30)
    check(done.returncode == 0 and set(json.loads(ack.read_text())["keys"]) == set(state["active"]), "ack records the active problems")
    watcher = subprocess.Popen(watch, stdout=subprocess.PIPE, text=True)
    time.sleep(1.0)
    check(watcher.poll() is None, "the watcher waits while nothing changed")
    plan({}, {"mon0": health([])})
    asyncio.run(canary.tick())
    output, _ = watcher.communicate(timeout=30)
    check(watcher.returncode == 1 and json.loads(output.splitlines()[0]) == {"event": "cleared", "key": "cluster:unreachable"},
          f"the watcher reports a cleared problem: {watcher.returncode} {output}")
    stale = json.loads(Path(config["state_file"]).read_text())
    stale["epoch"] -= 3600
    Path(config["state_file"]).write_text(json.dumps(stale))
    done = subprocess.run(watch, capture_output=True, text=True, timeout=30)
    check(done.returncode == 3 and "canary_stale" in done.stdout, f"a canary that stopped writing is reported: {done.returncode} {done.stdout}")
    try:
        ceph_canary.Canary(dict(config, size_bytes=1000))
        failures.append("a size that direct I/O cannot use is refused")
    except SystemExit:
        pass


def main():
    failures = []
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        ran_probe = hasattr(os, "O_DIRECT") and Path("/proc/mounts").exists()
        if ran_probe:
            real_probe(failures, tmp / "real")
        hub_logic(failures, tmp / "hub")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    probe = "the real direct-I/O probe passed" if ran_probe else "the real probe needs Linux and did not run here"
    print(f"PASS test_ceph_canary: {probe}; stage failures, hangs, stuck probes, unreachable and slow hosts, down OSDs and new warnings alert after consecutive rounds and clear; exact-message noise only; health fallback; watcher exits on change and on a stale canary")


if __name__ == "__main__":
    main()
