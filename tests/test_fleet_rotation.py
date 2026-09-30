#!/usr/bin/env python3
import calendar
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
TOOL = REPOSITORY / "tools" / "fleet_rotation.py"
PRODUCTION = REPOSITORY / "deployment" / "fleet_rotation" / "rotation.json"
HOSTS = ["n0", "n1", "n2", "n3"]

FAKE = r'''#!/usr/bin/env python3
import fcntl, json, os, sys, time
world_path = os.environ["FAKE_WORLD"]
host = os.environ.get("FAKE_HOST", "?")
verb, args = sys.argv[1], sys.argv[2:]
model = args[0] if args else ""
with open(world_path + ".lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    world = json.load(open(world_path))
    with open(world["calls"], "a") as calls:
        calls.write(f"{host} {verb} {' '.join(args)}\n")
    b = world["behave"]
    rc, out = 0, ""
    node = world["nodes"].get(host, {})
    if verb in b.get("hang", []) and model in b.get("hang_models", []):
        fcntl.flock(lock, fcntl.LOCK_UN)
        time.sleep(30)
    elif verb == "mem":
        out = str(node["mem"])
    elif verb == "precheck":
        rc = 1 if model in b.get("precheck_fail", []) else 0
    elif verb == "start":
        if model in b.get("start_fail", []):
            rc = 1
        else:
            node["up"][model] = True
            node["ready"][model] = model not in b.get("never_ready", [])
            if not node["arena"].get(model):
                node["arena"][model] = world["arena"][model]
                node["mem"] -= world["arena"][model]
            if model in b.get("dies", []):
                node["up"][model] = False
            for other, gib in b.get("start_eats", {}).get(model, {}).items():
                world["nodes"][other]["mem"] -= gib
    elif verb == "stop":
        if model not in b.get("stop_stuck", []):
            node["up"][model] = False
            node["ready"][model] = False
    elif verb == "up":
        rc = 0 if node["up"].get(model) else 1
    elif verb == "ready":
        rc = 0 if node["up"].get(model) and node["ready"].get(model) else 1
    elif verb == "reclaim":
        if "--reclaim-pack" not in args:
            rc = 9
        elif node["up"].get(model):
            rc = 3
        else:
            node["mem"] += node["arena"].get(model, 0)
            node["arena"][model] = 0
    elif verb == "api-start":
        world["apis"][model] = True
    elif verb == "api-stop":
        world["apis"][model] = False
    elif verb == "api-up":
        rc = 0 if world["apis"].get(model) else 1
    elif verb == "api-health":
        rc = 0 if world["apis"].get(model) and model not in b.get("unhealthy", []) else 1
    elif verb == "smoke":
        out = b.get("smoke_text", {}).get(model) or ("garbage" if model in b.get("smoke_bad", []) else '{"text": "Paris"}')
    else:
        rc = 2
    json.dump(world, open(world_path, "w"))
print(out)
sys.exit(rc)
'''

FAKESSH = '''#!/bin/sh
host=$1
shift
FAKE_HOST=$host exec bash -c "$*"
'''


def model(nodes, mem, api=True, runnable=True, validated=False, port=9000):
    m = {"title": "t", "runnable": runnable, "validated": validated}
    if not runnable:
        m["not_runnable"] = "test"
        return m
    m.update({
        "nodes": nodes, "mem_gib": mem,
        "engine": {"precheck": "fake precheck @model@", "start": "fake start @model@ @rank@ @stamp@", "stop": "fake stop @model@",
                   "up": "fake up @model@", "ready": "fake ready @model@ @stamp@", "grace_s": 0.3, "ready_timeout_s": 1, "stop_timeout_s": 1},
        "packs": ["/p/@model@.rank@rank@.sha256"],
    })
    if api:
        m["api"] = {"start": "fake api-start @model@", "stop": "fake api-stop @model@", "up": "fake api-up @model@",
                    "health": "fake api-health @model@", "port": port, "timeout_s": 1}
        m["smoke"] = {"path": "/v1/completions", "body": {"prompt": "x"}, "expect": "Paris", "timeout_s": 1}
    return m


def config(tmp):
    return {
        "hub": "fakehub", "state_dir": str(tmp / "state"), "ssh": [str(tmp / "bin" / "fakessh")], "ssh_timeout_s": 3, "poll_s": 0.05,
        "floor_gib": 20, "node_free_gib": 110, "weightd_warm": "/w/weightd_warm", "weightd_socket": "/tmp/s.sock", "schedule_port": 0,
        "fleet": HOSTS, "cycle": {"anchor_hour": 0, "slots": ["full", "k3", "flash_plus"]},
        "slots": {"full": {"base": "big", "companions": [], "fallback_slot": "flash_plus"}, "k3": {"base": "k", "companions": [], "fallback_slot": "flash_plus"},
                  "flash_plus": {"base": "prod", "companions": ["c1", "c3", "c2"], "fallback_slot": None}},
        "fallback": "prod", "rollback_models": ["prod", "dflt"],
        "lock": {"pause_file": "ROTATION_PAUSE", "mirror_pause_file": "lock/ROTATION_PAUSE", "holder_file": "lock/PERF_HOLDER", "holder_prefixes": ["lead-"]},
        "commands": {"mem_probe": "fake mem", "reclaim": "fake reclaim @model@ --reclaim-pack @packs@", "smoke": "fake smoke @model@ @port@ @path@"},
        "models": {"prod": model(HOSTS, 38, validated=True, port=9000), "big": model(HOSTS, 68, port=9001), "k": model([], 0, runnable=False),
                   "c1": model(["n0", "n1"], 22, port=9002), "c2": model(["n1", "n2"], 46, port=9003), "c3": model([], 0, runnable=False),
                   "dflt": model(["n0", "n1"], 15, validated=True, port=9004)},
    }


class Fleet:
    def __init__(self, tmp, cfg=None):
        self.tmp = Path(tmp)
        (self.tmp / "bin").mkdir()
        for name, body in (("fake", FAKE), ("fakessh", FAKESSH)):
            p = self.tmp / "bin" / name
            p.write_text(body)
            p.chmod(0o755)
        self.cfg = cfg or config(self.tmp)
        self.cfg_path = self.tmp / "rotation.json"
        self.cfg_path.write_text(json.dumps(self.cfg))
        self.calls = self.tmp / "calls.log"
        self.world_path = self.tmp / "world.json"
        arena = {name: m["mem_gib"] for name, m in self.cfg["models"].items() if m.get("runnable")}
        nodes = {h: {"mem": 110 - 38, "up": {"prod": True}, "ready": {"prod": True}, "arena": {"prod": 38}} for h in HOSTS}
        self.save({"calls": str(self.calls), "behave": {}, "nodes": nodes, "apis": {"prod": True}, "arena": arena})
        self.state_dir = self.tmp / "state"

    def world(self):
        return json.loads(self.world_path.read_text())

    def save(self, w):
        self.world_path.write_text(json.dumps(w))

    def behave(self, **kw):
        w = self.world()
        w["behave"].update(kw)
        self.save(w)

    def tool(self, *args, at="2026-09-29T17:00:30Z", dry=False):
        env = dict(os.environ, FAKE_WORLD=str(self.world_path), PATH=f"{self.tmp / 'bin'}:{os.environ['PATH']}")
        cmd = [sys.executable, str(TOOL), "--config", str(self.cfg_path)]
        if dry:
            cmd.append("--dry-run")
        if at:
            cmd += ["--at", at]
        p = subprocess.run(cmd + list(args), capture_output=True, text=True, env=env, timeout=240)
        return p.returncode, p.stdout + p.stderr

    def state(self):
        p = self.state_dir / "state.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def call_lines(self):
        return self.calls.read_text().splitlines() if self.calls.exists() else []

    def clear_calls(self):
        if self.calls.exists():
            self.calls.unlink()

    def up_on(self, name):
        w = self.world()
        return [h for h in HOSTS if w["nodes"][h]["up"].get(name)]

    def alerts(self):
        p = self.state_dir / "ALERT"
        return p.read_text() if p.exists() else ""


failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name} {detail}")
        failures.append(name)


def first_index(lines, pattern):
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            return i
    return -1


def test_schedule_and_rotation():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        rc, out = f.tool("schedule", "--hours", "6", at="2026-09-29T15:10:00Z")
        rows = json.loads(out)
        check("schedule: hour 15 is the full slot", rows[0]["slot"] == "full" and rows[0]["models"] == ["big"], out[:300])
        check("schedule: unrunnable k3 slot resolves to flash_plus", rows[1]["slot"] == "flash_plus", str(rows[1]))
        companions = [r["models"][1] for r in rows if r["slot"] == "flash_plus"]
        check("schedule: companions rotate in order and skip unrunnable ones", companions == ["c1", "c2", "c1", "c2"], str(companions))


def test_adopt_and_full_transition():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        rc, out = f.tool("tick", at="2026-09-29T15:00:30Z")
        st = f.state()
        check("full: transition succeeds", rc == 0 and st.get("active") == ["big"] and st.get("phase") == "steady", out[-800:])
        lines = f.call_lines()
        order = [first_index(lines, p) for p in (r"fakehub api-stop prod", r"n\d stop prod", r"n\d reclaim prod --reclaim-pack", r"n\d start big", r"fakehub api-start big", r"fakehub smoke big")]
        check("full: api stop, engine stop, reclaim-pack, start, api, smoke in order", all(i >= 0 for i in order) and order == sorted(order), str(order))
        check("full: no node-global reclaim", not any(re.search(r"--reclaim(?![-\w])", l) for l in lines))
        check("full: production held on every node", f.up_on("prod") == [] and f.up_on("big") == HOSTS)
        check("full: status file written", (f.state_dir / "STATUS").read_text().startswith("ROTATION "))
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T15:05:30Z")
        lines = f.call_lines()
        check("full: steady tick only observes and checks health", rc == 0 and not any(" start " in l or " stop " in l for l in lines), "\n".join(lines[-5:]))
        rc, out = f.tool("tick", at="2026-09-29T16:00:30Z")
        st = f.state()
        check("full -> flash_plus: production back with the first companion", st.get("active") == ["prod", "c1"] and f.up_on("prod") == HOSTS and f.up_on("c1") == ["n0", "n1"], out[-600:])
        lines = f.call_lines()
        check("full -> flash_plus: production starts before the companion", first_index(lines, r"start prod") < first_index(lines, r"start c1"))


def test_ready_timeout_falls_back():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.behave(never_ready=["big"])
        rc, out = f.tool("tick", at="2026-09-29T15:00:30Z")
        st = f.state()
        check("fallback: ready timeout returns production", rc == 1 and st.get("active") == ["prod"] and st.get("phase") == "fallback", out[-800:])
        check("fallback: failed model stopped and its packs reclaimed", f.up_on("big") == [] and all(f.world()["nodes"][h]["arena"].get("big") == 0 for h in HOSTS))
        check("fallback: alert written", "ERROR" in f.alerts() and "not reached" in f.alerts(), f.alerts())
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T15:10:30Z")
        check("fallback: no retry of the failed slot in the same hour", not any("start big" in l for l in f.call_lines()), out[-400:])
        rc, out = f.tool("status", at="2026-09-29T15:10:30Z")
        check("status: one line with phase and target", out.startswith("ROTATION ") and "phase=fallback" in out and "target=prod" in out, out)


def test_engine_death_and_start_failure():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.behave(dies=["c1"])
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        check("death: an engine that exits during warmup fails the transition", f.state().get("active") == ["prod"] and "exited" in f.alerts(), out[-600:])
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.behave(start_fail=["c1"])
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        check("start failure: falls back to production alone", f.state().get("active") == ["prod"] and f.up_on("c1") == [] and f.up_on("prod") == HOSTS, out[-600:])


def test_ssh_timeout():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.behave(hang=["start"], hang_models=["big"])
        rc, out = f.tool("tick", at="2026-09-29T15:00:30Z")
        st = f.state()
        check("timeout: a hanging start is cut off and production restored", st.get("active") == ["prod"] and "rc124" in f.alerts(), out[-600:])


def test_smoke_failure():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.behave(smoke_bad=["c1"])
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        check("smoke: wrong answer falls back", f.state().get("active") == ["prod"] and "smoke failed" in f.alerts(), out[-500:])


def test_fallback_failure_pauses():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T15:00:30Z")
        f.behave(never_ready=["prod", "c1"])
        rc, out = f.tool("tick", at="2026-09-29T16:00:30Z")
        st = f.state()
        check("degraded: failed fallback pauses the rotation", st.get("phase") == "degraded" and (f.state_dir / "ROTATION_PAUSE").exists() and "CRITICAL" in f.alerts(), out[-800:])
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T16:05:30Z")
        check("degraded: paused rotation does nothing", not any(" start " in l or " stop " in l for l in f.call_lines()), out[-300:])


def test_preemption_and_resync():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        (f.state_dir / "lock").mkdir(exist_ok=True)
        (f.state_dir / "lock" / "PERF_HOLDER").write_text("lead-quant-w1 pid=1 since=x\n")
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T18:00:30Z")
        check("preempt: lead lock blocks every fleet command", rc == 0 and f.call_lines() == [] and f.state().get("phase") == "preempted", out[-300:])
        (f.state_dir / "lock" / "PERF_HOLDER").write_text("qwen-w6 pid=2\n")
        rc, out = f.tool("tick", at="2026-09-29T18:00:30Z")
        check("preempt: a non-lead holder does not block", f.state().get("phase") != "preempted", out[-300:])
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        (f.state_dir / "lock").mkdir(exist_ok=True)
        (f.state_dir / "lock" / "PERF_HOLDER").write_text("lead-window\n")
        f.tool("tick", at="2026-09-29T17:10:30Z")
        w = f.world()
        for h in ["n0", "n1"]:
            w["nodes"][h]["up"]["c1"] = False
        w["apis"]["c1"] = False
        f.save(w)
        (f.state_dir / "lock" / "PERF_HOLDER").write_text("")
        rc, out = f.tool("tick", at="2026-09-29T17:20:30Z")
        check("resync: after a lead window the fleet is adopted, not flagged foreign", "ADOPT" in out and not (f.state_dir / "ROTATION_PAUSE").exists(), out[-600:])
        check("resync: the slot is re-established", f.state().get("active") == ["prod", "c1"] and f.up_on("c1") == ["n0", "n1"], out[-400:])


def test_pause_resume_and_foreign_change():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        rc, out = f.tool("pause", "owner", "demo")
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T15:00:30Z")
        check("pause: tick touches nothing", f.call_lines() == [] and f.state().get("phase") == "paused", out[-300:])
        rc, out = f.tool("resume")
        rc, out = f.tool("tick", at="2026-09-29T15:00:30Z")
        check("resume: next tick proceeds", f.state().get("active") == ["big"], out[-300:])
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        w = f.world()
        for h in ["n0", "n1"]:
            w["nodes"][h]["up"]["dflt"] = True
        w["apis"]["dflt"] = True
        f.save(w)
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T17:05:30Z")
        check("foreign: an unexpected model pauses the rotation", (f.state_dir / "ROTATION_PAUSE").exists() and "outside the rotation" in f.alerts(), out[-400:])
        check("foreign: nothing is stopped or started", not any(" start " in l or " stop " in l for l in f.call_lines()))


def test_floor_and_prediction():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        w = f.world()
        w["nodes"]["n3"]["mem"] = 40
        f.save(w)
        rc, out = f.tool("tick", at="2026-09-29T15:00:30Z")
        lines = f.call_lines()
        check("predict: a full slot that cannot fit never holds production", not any("stop prod" in l for l in lines), out[-600:])
        check("predict: the slot is demoted to flash_plus with a fitting companion", f.state().get("active") == ["prod", "c1"], str(f.state().get("active")))
        check("predict: demotion is alerted", "demoted" in f.alerts(), f.alerts())
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.behave(precheck_fail=["c1"])
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        check("precheck: a companion that fails precheck is skipped for the next one", f.state().get("active") == ["prod", "c2"], out[-500:])


def test_steady_health_and_manual():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        f.behave(unhealthy=["c1"])
        rc, out = f.tool("tick", at="2026-09-29T17:05:30Z")
        check("health: companion api failure falls back to production", f.state().get("active") == ["prod"] and f.up_on("c1") == [], out[-400:])
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        rc, out = f.tool("force", "flash_plus", "c2", at="2026-09-29T15:00:30Z")
        rc, out = f.tool("tick", at="2026-09-29T15:00:30Z")
        check("force: override slot and companion for this hour", f.state().get("active") == ["prod", "c2"], out[-300:])
        rc, out = f.tool("skip", at="2026-09-29T15:20:00Z")
        rc, out = f.tool("tick", at="2026-09-29T15:20:30Z")
        check("skip: the rest of the hour is production alone", f.state().get("active") == ["prod"] and f.up_on("c2") == [], out[-300:])
        rc, out = f.tool("now", at="2026-09-29T15:20:30Z")
        check("now: shows status, lock and upcoming slots", "ROTATION " in out and "lock:" in out and "2026-09-29T16:00:00Z" in out, out)
        rc, out = f.tool("force", "nosuch")
        check("force: unknown slot refused", rc == 2)
        rc, out = f.tool("converge", "--rollback", at="2026-09-29T15:30:00Z")
        check("rollback: converge to the rollback models", rc == 0 and f.state().get("active") == ["prod", "dflt"] and f.up_on("dflt") == ["n0", "n1"], out[-400:])


def test_stuck_stop_and_reclaim_guard():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        f.behave(stop_stuck=["c1"])
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T18:00:30Z")
        lines = f.call_lines()
        check("stuck: an engine that will not stop is never reclaimed", not any("reclaim c1" in l for l in lines), "\n".join(lines[-6:]))
        check("stuck: production is not held behind a stuck companion", not any("stop prod" in l for l in lines) and f.up_on("prod") == HOSTS)
        check("stuck: the rotation pauses for the lead", f.state().get("phase") == "degraded" and "CRITICAL" in f.alerts(), out[-500:])
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        f.clear_calls()
        sys.path.insert(0, str(TOOL.parent))
        import fleet_rotation
        os.environ["FAKE_WORLD"] = str(f.world_path)
        os.environ["PATH"] = f"{f.tmp / 'bin'}:{os.environ['PATH']}"
        rot = fleet_rotation.Rotation(f.cfg, fleet_rotation.Runner(f.cfg, False, lambda *_: None), out=lambda *_: None)
        try:
            rot.reclaim("c1")
            refused = False
        except fleet_rotation.Failure:
            refused = True
        check("reclaim guard: reclaim-pack refused while the engine is up", refused and not any("reclaim c1" in l for l in f.call_lines()))


def interrupt_world(f, big_hosts):
    w = f.world()
    for h in HOSTS:
        w["nodes"][h]["up"]["prod"] = False
        w["nodes"][h]["ready"]["prod"] = False
        w["nodes"][h]["up"]["c1"] = False
        w["nodes"][h]["arena"]["prod"] = 0
        w["nodes"][h]["mem"] = 110
    w["apis"]["prod"] = False
    w["apis"]["c1"] = False
    for h in big_hosts:
        w["nodes"][h]["up"]["big"] = True
    f.save(w)
    st = f.state()
    st["phase"] = "transition"
    st["transition"] = {"from": ["prod", "c1"], "to": ["big"], "reason": "slot full", "started": "x"}
    (f.state_dir / "state.json").write_text(json.dumps(st))


def test_interrupted_transition_recovers():
    for big_hosts, label in (([], "production held, nothing started"), (["n0", "n1"], "incoming engine half started")):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fleet(tmp)
            f.tool("tick", at="2026-09-29T17:00:30Z")
            interrupt_world(f, big_hosts)
            rc, out = f.tool("tick", at="2026-09-29T18:05:30Z")
            st = f.state()
            check(f"interrupted ({label}): the next tick restores production", f.up_on("prod") == HOSTS and f.world()["apis"].get("prod") and st.get("active") == ["prod"] and st.get("phase") == "fallback", out[-700:])
            check(f"interrupted ({label}): the half-started engine is stopped and reclaimed by pack", f.up_on("big") == [] and not (f.state_dir / "ROTATION_PAUSE").exists(), out[-400:])
            check(f"interrupted ({label}): alerted", "did not finish" in f.alerts(), f.alerts())


def test_pause_during_running_tick():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = config(Path(tmp))
        cfg["models"]["big"]["engine"]["ready_timeout_s"] = 60
        f = Fleet(tmp, cfg)
        f.behave(never_ready=["big"])
        env = dict(os.environ, FAKE_WORLD=str(f.world_path), PATH=f"{f.tmp / 'bin'}:{os.environ['PATH']}")
        p = subprocess.Popen([sys.executable, str(TOOL), "--config", str(f.cfg_path), "--at", "2026-09-29T15:00:30Z", "tick"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not any("start big" in l for l in f.call_lines()):
            time.sleep(0.1)
        rc, out = f.tool("pause", "lead", "window")
        check("pause: accepted while a tick holds the state lock", rc == 0 and (f.state_dir / "ROTATION_PAUSE").exists(), out)
        mark = len(f.call_lines())
        try:
            tick_out = p.communicate(timeout=30)[0]
        except subprocess.TimeoutExpired:
            p.kill()
            tick_out = p.communicate()[0]
        check("pause: the running transition stops at the next checkpoint", p.returncode == 1 and f.state().get("phase") == "paused" and f.state().get("resync") is True, tick_out[-600:])
        check("pause: nothing is started or stopped after the pause", not any(" start " in l or " stop " in l or "reclaim" in l for l in f.call_lines()[mark:]), "\n".join(f.call_lines()[mark:]))
        check("pause: the stop at a lock is alerted", "stopped at a lock" in f.alerts(), f.alerts())
        f.behave(never_ready=[])
        f.tool("resume")
        rc, out = f.tool("tick", at="2026-09-29T15:10:30Z")
        check("pause: after resume the half-started slot is adopted and production restored", f.up_on("prod") == HOSTS and f.up_on("big") == [] and f.state().get("active") == ["prod"], out[-600:])
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        (f.state_dir).mkdir()
        (f.state_dir / "lock").mkdir()
        (f.state_dir / "lock" / "PERF_HOLDER").write_text("lead-w1 pid=1\n")
        rc, out = f.tool("converge", "--rollback", at="2026-09-29T15:30:00Z")
        check("rollback: a manual converge runs under a pause or lead lock", rc == 0 and f.up_on("dflt") == ["n0", "n1"], out[-400:])


def test_unexpected_error_falls_back():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = config(Path(tmp))
        cfg["models"]["big"]["engine"]["ready"] = "fake ready @model@ @unknown@"
        f = Fleet(tmp, cfg)
        rc, out = f.tool("tick", at="2026-09-29T15:00:30Z")
        check("unexpected error: a non-Failure exception mid-transition still falls back", f.up_on("prod") == HOSTS and f.up_on("big") == [] and f.state().get("phase") == "fallback" and "ConfigError" in f.alerts(), out[-600:])


def test_failed_health_fallback_reports_degraded():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        f.behave(unhealthy=["c1"], stop_stuck=["c1"])
        rc, out = f.tool("tick", at="2026-09-29T17:05:30Z")
        st = f.state()
        check("health: a failed fallback is reported degraded, not steady", st.get("phase") == "degraded" and "phase=degraded" in (f.state_dir / "STATUS").read_text() and f.up_on("prod") == HOSTS, out[-500:])


def test_mixed_production_is_not_held():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        (f.state_dir / "lock").mkdir(exist_ok=True)
        (f.state_dir / "lock" / "PERF_HOLDER").write_text("lead-window\n")
        f.tool("tick", at="2026-09-29T17:10:30Z")
        w = f.world()
        w["apis"]["prod"] = False
        f.save(w)
        (f.state_dir / "lock" / "PERF_HOLDER").write_text("")
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T17:15:30Z")
        lines = f.call_lines()
        check("mixed production: adopted production with its api down is not held or reclaimed", not any("stop prod" in l or "reclaim prod" in l for l in lines), "\n".join(lines))
        check("mixed production: its api is restarted and smoked, the companion kept", f.world()["apis"].get("prod") and any("smoke prod" in l for l in lines) and f.state().get("active") == ["prod", "c1"] and f.up_on("c1") == ["n0", "n1"], out[-600:])
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        (f.state_dir / "lock").mkdir(exist_ok=True)
        (f.state_dir / "lock" / "PERF_HOLDER").write_text("lead-window\n")
        f.tool("tick", at="2026-09-29T17:10:30Z")
        w = f.world()
        w["apis"]["prod"] = False
        w["apis"]["c1"] = False
        for h in ["n0", "n1"]:
            w["nodes"][h]["up"]["c1"] = False
        f.save(w)
        (f.state_dir / "lock" / "PERF_HOLDER").write_text("")
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T17:15:30Z")
        lines = f.call_lines()
        check("mixed production: restoring production with nothing else up does not hold it first", not any("stop prod" in l or "reclaim prod" in l for l in lines) and f.world()["apis"].get("prod") and f.up_on("prod") == HOSTS, "\n".join(lines[-8:]))


def test_dry_run_executes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        rc, out = f.tool("tick", at="2026-09-29T15:00:30Z", dry=True)
        check("dry-run: prints the commands", "DRY n0: fake start big 0" in out and "DRY hub: fake smoke big" in out and "--reclaim-pack" in out, out[-600:])
        check("dry-run: executes nothing and writes no state", f.call_lines() == [] and not f.state_dir.exists())


def test_schedule_document_and_sync():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        doc = json.loads((f.state_dir / "schedule.json").read_text())
        by_id = {m["id"]: m for m in doc["models"]}
        check("schedule doc: active models and next slot per model", by_id["c1"]["active"] and not by_id["big"]["active"] and by_id["big"]["next_slot_start"] == "2026-09-29T18:00:00Z", json.dumps(by_id["big"]))
        check("schedule doc: unrunnable models state why", by_id["k"]["runnable"] is False and by_id["k"]["not_runnable"])
        lanes = Path(tmp) / "lanes"
        lanes.mkdir()
        (lanes / "PERF_HOLDER").write_text("lead-w5 pid=9\n")
        (lanes / "ROTATION_PAUSE").write_text("lead: release window\n")
        rc, out = f.tool("sync", "--lanes", str(lanes), at=None)
        check("sync: mirrors the lead lock and pause to the hub state dir", (f.state_dir / "lock" / "PERF_HOLDER").read_text().startswith("lead-w5") and (f.state_dir / "lock" / "ROTATION_PAUSE").exists(), out)
        check("sync: pulls the status line for the heartbeat", (lanes / "ROTATION_STATUS").read_text().startswith("ROTATION "), (lanes / "ROTATION_STATUS").read_text())
        (lanes / "ROTATION_PAUSE").unlink()
        f.tool("sync", "--lanes", str(lanes), at=None)
        check("sync: a cleared coord pause is cleared on the hub", not (f.state_dir / "lock" / "ROTATION_PAUSE").exists())


def test_config_validation():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        bad = json.loads(f.cfg_path.read_text())
        bad["commands"]["reclaim"] = "fake reclaim @model@ --reclaim @packs@"
        f.cfg_path.write_text(json.dumps(bad))
        rc, out = f.tool("check-config", at=None)
        check("config: node-global reclaim refused", rc == 2 and "reclaim" in out, out)
        bad = config(Path(tmp))
        bad["models"]["c3"] = {"title": "t", "runnable": False}
        f.cfg_path.write_text(json.dumps(bad))
        rc, out = f.tool("check-config", at=None)
        check("config: an unrunnable model must say why", rc == 2 and "not_runnable" in out, out)
        bad = config(Path(tmp))
        del bad["models"]["c1"]["engine"]["ready"]
        f.cfg_path.write_text(json.dumps(bad))
        rc, out = f.tool("check-config", at=None)
        check("config: a runnable model needs every engine command", rc == 2 and "engine.ready" in out, out)


def exclusive_fleet(tmp, floor=True, arena=95):
    cfg = config(Path(tmp))
    cfg["models"]["k"] = model(HOSTS, 95, port=9005)
    if floor:
        cfg["models"]["k"].update({"floor_gib": 8, "abort_gib": 6})
    f = Fleet(tmp, cfg)
    w = f.world()
    w["arena"]["k"] = arena
    f.save(w)
    return f


def test_exclusive_floor_and_abort():
    with tempfile.TemporaryDirectory() as tmp:
        f = exclusive_fleet(tmp)
        rc, out = f.tool("tick", at="2026-09-29T16:00:30Z")
        st = f.state()
        check("exclusive floor: a model that fits only under its own floor takes its slot", rc == 0 and st.get("active") == ["k"] and f.up_on("k") == HOSTS and f.up_on("prod") == [], out[-600:])
        log = (f.state_dir / "rotation.log").read_text()
        check("exclusive floor: before is checked at the global floor, after at the model's floor", re.search(r"FLOOR when=before .*floor=20", log) and re.search(r"FLOOR when=after .*floor=8", log), log[-600:])
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T16:05:30Z")
        check("exclusive floor: a steady tick probes memory for the abort level and changes nothing", rc == 0 and f.state().get("active") == ["k"] and any(" mem" in l for l in f.call_lines()) and not any(" stop " in l for l in f.call_lines()), out[-400:])
        check("exclusive floor: the steady floor guard reads the model's own floor (15 GiB free raises no shed warning)", "no companion there to shed" not in f.alerts() and f.state().get("phase") == "steady", f.alerts())
        rc, out = f.tool("plan", at="2026-09-29T16:07:30Z")
        check("exclusive floor: plan keeps the exclusive model and predicts at its own floor", rc == 0 and "target=k " in out and "demoted" not in f.alerts(), out[-400:])
        w = f.world()
        w["nodes"]["n2"]["mem"] = 5
        f.save(w)
        rc, out = f.tool("tick", at="2026-09-29T16:10:30Z")
        st = f.state()
        check("abort: MemAvailable under abort_gib in steady state falls back to production", rc == 1 and st.get("active") == ["prod"] and f.up_on("k") == [] and f.up_on("prod") == HOSTS, out[-600:])
        check("abort: alerted with the node and level", "abort level" in f.alerts() and "n2=5<6" in f.alerts(), f.alerts())
        log = (f.state_dir / "rotation.log").read_text()
        check("abort: the fallback's own floor check is the global floor", re.search(r"FLOOR when=after-fallback .*floor=20", log), log[-600:])
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        check("abort: the next hour follows the schedule again", f.state().get("active") == ["prod", "c1"], out[-300:])
    with tempfile.TemporaryDirectory() as tmp:
        f = exclusive_fleet(tmp)
        f.tool("tick", at="2026-09-29T16:00:30Z")
        (f.state_dir / "rotation.log").write_text("")
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        log = (f.state_dir / "rotation.log").read_text()
        check("exclusive floor: leaving the exclusive hour checks before at the model's floor and after at the global floor", rc == 0 and f.state().get("active") == ["prod", "c1"] and f.up_on("k") == [] and re.search(r"FLOOR when=before .*floor=8", log) and re.search(r"FLOOR when=after .*floor=20", log) and "FALLBACK" not in log, out[-600:] + log[-600:])
    with tempfile.TemporaryDirectory() as tmp:
        f = exclusive_fleet(tmp, floor=False)
        rc, out = f.tool("tick", at="2026-09-29T16:00:30Z")
        check("exclusive floor: without floor_gib the same model is demoted at the global floor", f.state().get("active") == ["prod", "c1"] and "demoted" in f.alerts() and not any("start k" in l for l in f.call_lines()), out[-500:])
    with tempfile.TemporaryDirectory() as tmp:
        f = exclusive_fleet(tmp, arena=106)
        rc, out = f.tool("tick", at="2026-09-29T16:00:30Z")
        st = f.state()
        check("abort: memory under abort_gib while waiting for ready fails the start and restores production", rc == 1 and st.get("active") == ["prod"] and f.up_on("k") == [] and "abort level while waiting" in f.alerts(), out[-600:] + f.alerts())
    with tempfile.TemporaryDirectory() as tmp:
        f = exclusive_fleet(tmp)
        w = f.world()
        w["nodes"]["n1"]["mem"] = 64
        f.save(w)
        rc, out = f.tool("tick", at="2026-09-29T16:00:30Z")
        check("exclusive floor: a node short even for the model's own floor demotes the slot", f.state().get("active") == ["prod", "c1"] and "demoted" in f.alerts() and f.up_on("prod") == HOSTS, out[-500:])


def test_exclusive_floor_validation():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        cases = (
            ("companion", lambda c: c["models"]["c1"].update({"floor_gib": 8, "abort_gib": 6}), "floor_gib is only"),
            ("fallback", lambda c: c["models"]["prod"].update({"floor_gib": 8, "abort_gib": 6}), "floor_gib is only"),
            ("base of a slot with companions", lambda c: (c["models"].update({"k": dict(model(HOSTS, 95, port=9005), floor_gib=8, abort_gib=6)}), c["slots"]["k3"].update({"companions": ["c1"]})), "floor_gib is only"),
            ("partial fleet", lambda c: c["models"].update({"k": dict(model(["n0", "n1"], 95, port=9005), floor_gib=8, abort_gib=6)}), "whole fleet"),
            ("abort above floor", lambda c: c["models"].update({"k": dict(model(HOSTS, 95, port=9005), floor_gib=8, abort_gib=9)}), "abort_gib"),
            ("floor above the global floor", lambda c: c["models"].update({"k": dict(model(HOSTS, 95, port=9005), floor_gib=30, abort_gib=6)}), "abort_gib"),
            ("floor without abort", lambda c: c["models"].update({"k": dict(model(HOSTS, 95, port=9005), floor_gib=8)}), "come together"),
        )
        for label, mutate, want in cases:
            bad = config(Path(tmp))
            mutate(bad)
            f.cfg_path.write_text(json.dumps(bad))
            rc, out = f.tool("check-config", at=None)
            check(f"config: exclusive floor refused ({label})", rc == 2 and want in out, out)
        good = config(Path(tmp))
        good["models"]["k"] = dict(model(HOSTS, 95, port=9005), floor_gib=8, abort_gib=6)
        f.cfg_path.write_text(json.dumps(good))
        rc, out = f.tool("check-config", at=None)
        check("config: exclusive floor accepted on a whole-fleet slot base", rc == 0, out)


def test_pack_cache_trim():
    trim = REPOSITORY / "tools" / "pack_cache_trim.py"
    with tempfile.TemporaryDirectory() as tmp:
        p = subprocess.run([sys.executable, str(trim), str(Path(tmp) / "absent.pack")], capture_output=True, text=True)
        check("cache trim: a missing pack is refused", p.returncode == 2 and "not a file: " in p.stderr and "absent.pack" in p.stderr, p.stdout + p.stderr)
        p = subprocess.run([sys.executable, str(trim), "--every", "-1", str(trim)], capture_output=True, text=True)
        check("cache trim: a negative period is refused", p.returncode == 2, p.stdout + p.stderr)
        source = trim.read_text()
        check("cache trim: pack-scoped only, never a node-global drop", "drop_caches" not in source and "POSIX_FADV_DONTNEED" in source)
        if not hasattr(os, "posix_fadvise"):
            print("SKIP cache trim: posix_fadvise is Linux-only; the page-cache check runs on the Linux host-tests")
            return
        pack = Path(tmp) / "a.pack"
        pack.write_bytes(os.urandom(8 << 20))
        with open(pack, "rb") as f:
            os.fsync(f.fileno())
            f.read()
        before = cached_bytes(pack)
        locked = Path(tmp) / "locked.pack"
        locked.write_bytes(b"x")
        locked.chmod(0)
        p = subprocess.run([sys.executable, str(trim), str(locked), str(pack)], capture_output=True, text=True)
        after = cached_bytes(pack)
        unreadable = os.getuid() != 0
        check("cache trim: the pack's page cache is dropped and an unreadable pack is skipped", p.returncode == 0 and (f"{2 - unreadable} of 2 files" in p.stdout) and before > 0 and after < before, f"{before} -> {after} {p.stdout}{p.stderr}")


def cached_bytes(path):
    import ctypes
    import mmap
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    size = os.path.getsize(path)
    page = os.sysconf("SC_PAGE_SIZE")
    fd = os.open(path, os.O_RDONLY)
    try:
        address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        pages = (size + page - 1) // page
        vector = (ctypes.c_ubyte * pages)()
        rc = libc.mincore(address, size, vector)
        libc.munmap(address, size)
    finally:
        os.close(fd)
    return sum(v & 1 for v in vector) * page if rc == 0 else -1


def test_dry_run_while_paused_shows_the_plan():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("pause", "install", at="2026-09-29T17:00:30Z")
        f.tool("force", "flash_plus", "c2", at="2026-09-29T17:00:30Z")
        before = (f.state_dir / "state.json").read_text()
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T17:01:00Z", dry=True)
        check("dry-run paused: reports the hold-off and still prints the plan", "HOLD-OFF" in out and "DRY n1: fake start c2 0" in out and "DRY hub: fake smoke c2" in out, out[-600:])
        check("dry-run paused: executes nothing, keeps the pause and the state", f.call_lines() == [] and (f.state_dir / "ROTATION_PAUSE").exists() and (f.state_dir / "state.json").read_text() == before)


def test_failure_after_the_hour_boundary_marks_the_starting_hour():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.behave(never_ready=["big"])
        f.tool("force", "full", at="2026-09-29T17:59:40Z")
        sys.path.insert(0, str(TOOL.parent))
        import fleet_rotation
        os.environ["FAKE_WORLD"] = str(f.world_path)
        os.environ["PATH"] = f"{f.tmp / 'bin'}:{os.environ['PATH']}"
        start = fleet_rotation.parse_time("2026-09-29T17:59:50Z")
        rot = fleet_rotation.Rotation(f.cfg, fleet_rotation.Runner(f.cfg, False, lambda *_: None), now=start, out=lambda *_: None)
        rot.clock = fleet_rotation.parse_time("2026-09-29T18:00:10Z")
        rot.tick()
        st = f.state()
        check("boundary: a failed transition blocks the hour it started in, not the next one", st.get("failed_instance") == int(start // 3600) and st.get("active") == ["prod"], json.dumps(st)[:300])
        nxt = fleet_rotation.Rotation(f.cfg, fleet_rotation.Runner(f.cfg, False, lambda *_: None), now=fleet_rotation.parse_time("2026-09-29T18:00:30Z"), out=lambda *_: None)
        check("boundary: the next hour runs its own slot", nxt.target(f.state(), nxt.now())[2] == ["big"])


def test_install_rollback_waits_for_a_running_tick():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "bin").mkdir()
        log = tmp / "ssh.log"
        (tmp / "bin" / "ssh").write_text(f"""#!/bin/sh
printf '%s\\n' "$*" >> {log}
case "$*" in
  *"show -p ActiveState"*)
    n=$(grep -c 'show -p ActiveState' {log})
    if [ "$n" -le 2 ]; then echo activating; else echo inactive; fi ;;
esac
exit 0
""")
        (tmp / "bin" / "sleep").write_text("#!/bin/sh\nexit 0\n")
        for name in ("ssh", "sleep"):
            (tmp / "bin" / name).chmod(0o755)
        env = dict(os.environ, PATH=f"{tmp / 'bin'}:{os.environ['PATH']}", FLEET_ROTATION_CONFIG=str(PRODUCTION))
        p = subprocess.run(["bash", str(REPOSITORY / "tools" / "fleet_rotation_install.sh"), "rollback"], capture_output=True, text=True, env=env, timeout=60)
        lines = log.read_text().splitlines() if log.exists() else []
        waits = [i for i, l in enumerate(lines) if "show -p ActiveState" in l]
        converge = first_index(lines, r"converge --rollback")
        disable = first_index(lines, r"disable --now fleet-rotation.timer")
        check("rollback: waits while the oneshot tick is activating, then converges", p.returncode == 0 and len(waits) == 3 and 0 <= disable < waits[0] and converge > waits[-1], p.stdout + p.stderr + "\n".join(lines[-6:]))


def hour_key(text):
    return str(calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")) // 3600)


H17 = hour_key("2026-09-29T17:00:00Z")
H19 = hour_key("2026-09-29T19:00:00Z")


def pack_config(tmp):
    cfg = config(Path(tmp))
    cfg["slots"]["full"]["companions"] = ["ta"]
    cfg["slots"]["flash_plus"]["companions"] = ["ta", "tb", "tc", "wide", "c3"]
    cfg["rollback_models"] = ["prod", "ta"]
    for name in ("c1", "c2", "dflt"):
        del cfg["models"][name]
    cfg["models"].update({"ta": model(["n0"], 15, port=9010), "tb": model(["n1"], 36, port=9011),
                          "tc": model(["n2"], 46, port=9012), "wide": model(HOSTS, 22, port=9013)})
    return cfg


def pack_fleet(tmp):
    return Fleet(tmp, pack_config(tmp))


def starts(lines):
    return [l.split()[2] for l in lines if re.match(r"\S+ start ", l)]


def test_packing_fills_the_nodes():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        st = f.state()
        check("packing: the primary plus every node-disjoint companion that fits", rc == 0 and st.get("active") == ["prod", "ta", "tb", "tc"] and st.get("phase") == "steady", out[-800:])
        check("packing: the wide companion that does not fit waits, with the short node named", "tc" not in st["picks"][H17]["waiting"] and "n2:" in st["picks"][H17]["waiting"].get("wide", ""), json.dumps(st.get("picks")))
        lines = f.call_lines()
        order = [first_index(lines, p) for p in (r"n0 start ta", r"fakehub smoke ta", r"n1 start tb", r"fakehub smoke tb", r"n2 start tc", r"fakehub smoke tc")]
        check("packing: companions start one at a time, each smoked before the next starts", all(i >= 0 for i in order) and order == sorted(order), str(order))
        check("packing: the running primary is not restarted", not any(" start prod" in l or " stop prod" in l for l in lines))
        mem = {h: f.world()["nodes"][h]["mem"] for h in HOSTS}
        check("packing: every node keeps the floor after all starts", min(mem.values()) >= 20, str(mem))
        log = (f.state_dir / "rotation.log").read_text()
        check("packing: memory is checked on each companion's nodes after its start", all(re.search(rf"FLOOR when=after-{m} ", log) for m in ("ta", "tb", "tc")), log[-600:])


def test_packing_per_node_floor():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        w = f.world()
        w["nodes"]["n1"]["mem"] = 50
        f.save(w)
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        st = f.state()
        check("floor: a node with less memory keeps its companion out, the others still pack", st.get("active") == ["prod", "ta", "tc"] and "n1:" in st["picks"][H17]["waiting"].get("tb", ""), json.dumps(st.get("picks")) + out[-400:])
        check("floor: no start on the short node", not any(l.startswith("n1 start") for l in f.call_lines()))
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.behave(start_eats={"ta": {"n2": 30}})
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        st = f.state()
        check("live gate: MemAvailable is read again before each companion start", st.get("active") == ["prod", "ta", "tb"] and not any(l.startswith("n2 start") for l in f.call_lines()), out[-600:])
        check("live gate: the refused companion is skipped for the hour and alerted", "tc" in st["dropped"][H17] and "needs 46+20" in f.alerts() and st.get("phase") == "steady", f.alerts())
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        w = f.world()
        w["arena"]["tb"] = 60
        f.save(w)
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        st = f.state()
        check("floor after start: a companion that leaves its node under the floor is stopped and reclaimed", st.get("active") == ["prod", "ta", "tc"] and f.up_on("tb") == [] and f.world()["nodes"]["n1"]["mem"] == 72, out[-600:])
        check("floor after start: the primary and the other companions keep serving", f.up_on("prod") == HOSTS and f.up_on("ta") == ["n0"] and f.up_on("tc") == ["n2"] and "below 20 GiB after-tb" in f.alerts(), f.alerts())


def test_partial_companion_failure():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.behave(smoke_bad=["tb"])
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        st = f.state()
        lines = f.call_lines()
        check("partial: a failing companion is stopped and reclaimed, the rest serve", rc == 1 and st.get("active") == ["prod", "ta", "tc"] and f.up_on("tb") == [] and f.world()["nodes"]["n1"]["arena"].get("tb") == 0, out[-800:])
        check("partial: the primary is never touched", not any("prod" in l.split()[1:3] and l.split()[1] in ("start", "stop", "reclaim") for l in lines) and f.world()["apis"].get("prod"), "\n".join(lines[-10:]))
        check("partial: the companion after the failed one still starts", first_index(lines, r"n2 start tc") > first_index(lines, r"n1 stop tb") >= 0)
        check("partial: steady, not a fallback; alerted", st.get("phase") == "steady" and st.get("failed_instance") is None and "companion tb failed" in f.alerts(), f.alerts())
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T17:05:30Z")
        check("partial: the failed companion is not retried within the hour", rc == 0 and starts(f.call_lines()) == [] and f.state().get("active") == ["prod", "ta", "tc"], out[-400:])
        f.behave(smoke_bad=[])
        rc, out = f.tool("tick", at="2026-09-29T19:00:30Z")
        st = f.state()
        check("partial: the next companion hour considers it again", "tb" not in st["dropped"].get(H19, []) and ("tb" in st["picks"][H19]["companions"] or "tb" in st["picks"][H19]["waiting"]), json.dumps(st.get("picks", {}).get(H19)))
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.behave(smoke_bad=["tb"], stop_stuck=["tb"])
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        st = f.state()
        check("partial: a failed companion that cannot be stopped pauses the rotation and leaves the primary serving", st.get("phase") == "degraded" and (f.state_dir / "ROTATION_PAUSE").exists() and "CRITICAL" in f.alerts() and f.up_on("prod") == HOSTS and not any(" stop prod" in l for l in f.call_lines()), out[-600:])
        check("partial: a stuck companion is not treated as a failed transition", st.get("failed_instance") is None and "FALLBACK" not in (f.state_dir / "rotation.log").read_text(), out[-600:])


def test_steady_companion_loss_and_health():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        w = f.world()
        w["nodes"]["n2"]["up"]["tc"] = False
        f.save(w)
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T17:05:30Z")
        lines = f.call_lines()
        check("lost companion: only that companion is cleaned up", f.state().get("active") == ["prod", "ta", "tb"] and any("reclaim tc" in l for l in lines) and not any(re.search(r" (stop|reclaim) (prod|ta|tb)", l) for l in lines), "\n".join(lines[-8:]))
        f.behave(unhealthy=["ta"])
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T17:10:30Z")
        lines = f.call_lines()
        check("sick companion: only that companion is stopped", f.state().get("active") == ["prod", "tb"] and f.up_on("ta") == [] and f.up_on("tb") == ["n1"] and not any(" stop prod" in l or " stop tb" in l for l in lines), out[-500:])


def test_rotation_fairness_across_cycles():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        ran = {}
        for hour in ("2026-09-29T16:00:30Z", "2026-09-29T17:00:30Z", "2026-09-29T19:00:30Z", "2026-09-29T20:00:30Z", "2026-09-29T22:00:30Z", "2026-09-29T23:00:30Z"):
            rc, out = f.tool("tick", at=hour)
            active = f.state().get("active")
            ran[hour[11:13]] = active
            check(f"fairness: {hour[11:16]} runs production plus a packed set", rc == 0 and active[0] == "prod" and len(active) >= 2, out[-400:])
            mem = [f.world()["nodes"][h]["mem"] for h in HOSTS]
            check(f"fairness: {hour[11:16]} keeps the floor", min(mem) >= 20, str(mem))
        seen = {m for active in ran.values() for m in active[1:]}
        check("fairness: every runnable companion gets runtime within three companion hours", seen == {"ta", "tb", "tc", "wide"} and "wide" in ran["17"] + ran["19"] + ran["20"], json.dumps(ran))
        check("fairness: the wide companion alternates with the node-disjoint set it cannot join", [("wide" in ran[h]) for h in ("16", "17", "19", "20")] in ([False, True, False, True], [True, False, True, False]), json.dumps(ran))
        check("fairness: the unrunnable companion never runs", "c3" not in seen)
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.state_dir.mkdir()
        old = {str(int(H17) - k): ["tb"] for k in range(10, 40)}
        (f.state_dir / "state.json").write_text(json.dumps({"active": ["prod"], "phase": "steady", "dropped": old, "demoted": dict(old), "picks": {h: {"slot": "flash_plus", "primary": "prod", "companions": []} for h in old}}))
        f.tool("tick", at="2026-09-29T17:00:30Z")
        st = f.state()
        check("fairness: per-hour records keep the last 24 hours", all(len(st.get(k)) == 24 for k in ("picks", "dropped", "demoted")) and H17 in st["picks"] and H17 in st["dropped"], json.dumps({k: len(st.get(k) or {}) for k in ("picks", "dropped", "demoted")}))
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        rc, out = f.tool("schedule", "--hours", "9", at="2026-09-29T15:10:00Z")
        rows = json.loads(out)
        companions = {m for r in rows for m in r["companions"]}
        check("schedule: predicted rows carry primary and companions, and rotate every companion in", companions == {"ta", "tb", "tc", "wide"} and all(r["models"] == [r["primary"]] + r["companions"] for r in rows), out[:600])


def test_full_slot_with_companion():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T18:00:30Z")
        st = f.state()
        lines = f.call_lines()
        check("full + companion: the small companion that fits stays beside the full slot", rc == 0 and st.get("active") == ["big", "ta"] and f.up_on("ta") == ["n0"] and f.up_on("big") == HOSTS, out[-800:])
        check("full + companion: the kept companion is not restarted", not any(re.search(r" (stop|start) ta", l) for l in lines))
        order = [first_index(lines, p) for p in (r" stop tc", r" stop tb", r"fakehub api-stop prod", r"n\d start big")]
        check("full + companion: stop order is the reverse of start order, primary last, then the new primary starts", all(i >= 0 for i in order) and order == sorted(order), str(order))
        check("full + companion: the floor holds", min(f.world()["nodes"][h]["mem"] for h in HOSTS) >= 20, str(f.world()["nodes"]))
        rc, out = f.tool("tick", at="2026-09-29T19:00:30Z")
        lines = f.call_lines()
        check("full -> flash_plus: production starts before any companion", f.state().get("active")[0] == "prod" and first_index(lines, r"start prod") < min(i for i in [first_index(lines, rf"start {c}") for c in ("tb", "tc", "wide")] if i >= 0), out[-500:])
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        f.behave(never_ready=["big"])
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T18:00:30Z")
        lines = f.call_lines()
        check("full primary failure: falls back to production, companion stopped before the failed primary", f.state().get("active") == ["prod"] and f.state().get("phase") == "fallback" and 0 <= first_index(lines, r"n0 stop ta") < first_index(lines, r"n\d stop big"), "\n".join(lines[-12:]))


def test_packing_schedule_document_and_force():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        doc = json.loads((f.state_dir / "schedule.json").read_text())
        by_id = {m["id"]: m for m in doc["models"]}
        check("schedule doc: the current row lists the primary and every companion", doc["slots"][0]["primary"] == "prod" and doc["slots"][0]["companions"] == ["ta", "tb", "tc"], json.dumps(doc["slots"][0]))
        check("schedule doc: serving lists every active model with its port and role", [(s["id"], s["role"], s["port"]) for s in doc["serving"]] == [("prod", "fallback", 9000), ("ta", "companion", 9010), ("tb", "companion", 9011), ("tc", "companion", 9012)], json.dumps(doc["serving"]))
        check("schedule doc: per-model role and scheduled hours", by_id["big"]["role"] == "primary" and by_id["wide"]["role"] == "companion" and by_id["wide"]["scheduled_starts"] and by_id["wide"]["next_slot_start"] == by_id["wide"]["scheduled_starts"][0] and by_id["c3"]["scheduled_starts"] == [], json.dumps(by_id["wide"]))
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        rc, out = f.tool("force", "flash_plus", "ta", "wide", at="2026-09-29T15:00:30Z")
        rc, out = f.tool("tick", at="2026-09-29T15:00:30Z")
        check("force: several companions for this hour", f.state().get("active") == ["prod", "ta", "wide"], out[-400:])
        rc, out = f.tool("force", "flash_plus", "big", at="2026-09-29T15:00:30Z")
        check("force: a slot primary is refused as a companion", rc == 2, out)


def test_plan_is_read_only():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        before = (f.state_dir / "state.json").read_text()
        log_before = (f.state_dir / "rotation.log").read_text()
        f.clear_calls()
        rc, out = f.tool("plan", at="2026-09-29T19:00:30Z")
        verbs = {l.split()[1] for l in f.call_lines()}
        check("plan: prints the packing decision for the hour", rc == 0 and "PLAN 2026-09-29T19:00:00Z slot=flash_plus" in out and "target=prod,wide,ta" in out and "stop=tc,tb" in out, out[-800:])
        check("plan: runs only read-only probes", verbs <= {"up", "api-up", "mem", "precheck"}, str(verbs))
        check("plan: writes no state or log", (f.state_dir / "state.json").read_text() == before and (f.state_dir / "rotation.log").read_text() == log_before)


def test_smoke_reply_rules():
    sys.path.insert(0, str(TOOL.parent))
    import fleet_rotation
    cfg = json.loads(PRODUCTION.read_text())
    rot = fleet_rotation.Rotation(cfg, fleet_rotation.Runner(cfg, True, lambda *_: None), out=lambda *_: None)
    recorded = '{"object":"text_completion","choices":[{"index":0,"text":" a city of romance, art, and","finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":8,"total_tokens":13}}'
    smokes = {name: m["smoke"] for name, m in cfg["models"].items() if m.get("runnable")}
    for name in ("mimo", "ling"):
        problem, _ = rot.reply_problem(recorded, smokes[name]["reply"])
        check(f"smoke rules: {name} accepts the coherent reply MiMo gave live at 17:02Z", problem is None and not smokes[name].get("expect"), str(problem))
        for label, text in (("a short right answer", '{"choices":[{"text":" Paris.","finish_reason":"stop"}]}'),):
            problem, _ = rot.reply_problem(text, smokes[name]["reply"])
            check(f"smoke rules: {name} accepts {label}", problem is None, str(problem))
        for label, text in (("garbage", "garbage"), ("an empty text", '{"choices":[{"text":"","finish_reason":"stop"}]}'),
                            ("a repeated token", '{"choices":[{"text":" the the the the","finish_reason":"length"}]}'),
                            ("punctuation only", '{"choices":[{"text":"!!!! ??? 1234","finish_reason":"length"}]}'),
                            ("an error finish", '{"choices":[{"text":" a city of romance, art","finish_reason":"error"}]}')):
            problem, _ = rot.reply_problem(text, smokes[name]["reply"])
            check(f"smoke rules: {name} rejects {label}", problem is not None)
    chat = '{"choices":[{"message":{"role":"assistant","content":"Paris"},"finish_reason":"stop"}]}'
    check("smoke rules: chat replies are read from message.content", rot.reply_problem(chat, {"finish_reason": ["stop"]})[0] is None)
    check("smoke rules: templated models keep the Paris answer", all(smokes[m].get("expect") == "Paris" for m in ("flash", "glmfull", "qwen", "gemma")))
    with tempfile.TemporaryDirectory() as tmp:
        cfg = pack_config(tmp)
        cfg["models"]["tb"]["smoke"] = {"path": "/v1/completions", "body": {"prompt": "x"}, "timeout_s": 1, "reply": {"finish_reason": ["stop", "length"], "min_distinct_words": 3}}
        f = Fleet(tmp, cfg)
        f.behave(smoke_text={"tb": recorded})
        rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
        check("smoke rules: a companion with a coherent reply passes end to end", "tb" in f.state().get("active", []) and 'SMOKE-PASS model=tb reply=" a city of romance, art, and"' in out, out[-600:])


def test_packing_config_validation():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        cases = (
            ("a slot base as companion", lambda c: c["slots"]["flash_plus"]["companions"].append("big"), "slot base"),
            ("an unknown companion", lambda c: c["slots"]["flash_plus"]["companions"].append("nosuch"), "unknown"),
            ("two runnable models on one port", lambda c: c["models"]["tb"]["api"].update(port=9010), "api.port"),
            ("a smoke without expect or reply", lambda c: c["models"]["tb"]["smoke"].pop("expect"), "expect or reply"),
            ("an unknown reply rule", lambda c: c["models"]["tb"]["smoke"].update(reply={"contains": "x"}), "smoke.reply.contains"),
            ("no node_free_gib", lambda c: c.pop("node_free_gib"), "node_free_gib"),
        )
        for label, mutate, want in cases:
            bad = pack_config(tmp)
            mutate(bad)
            f.cfg_path.write_text(json.dumps(bad))
            rc, out = f.tool("check-config", at=None)
            check(f"config: refused ({label})", rc == 2 and want in out, out)
    cfg = json.loads(PRODUCTION.read_text())
    wide = [m for m in cfg["slots"]["full"]["companions"] if set(cfg["models"][m].get("nodes", [])) == set(cfg["fleet"])]
    check("production: the full slot takes no companion that spans the whole fleet", not wide, str(wide))
    check("production: every runnable companion is in the flash_plus pool", all(n in cfg["slots"]["flash_plus"]["companions"] for n, m in cfg["models"].items() if m.get("runnable") and n not in {s["base"] for s in cfg["slots"].values()} | {cfg["fallback"]}))


def install_harness(tmp, paused=False, new_config_ok=True):
    tmp = Path(tmp)
    (tmp / "bin").mkdir()
    log = tmp / "ssh.log"
    (tmp / "bin" / "ssh").write_text(f"""#!/bin/sh
printf 'ssh %s\\n' "$*" >> {log}
case "$*" in
  *fleet-rotation-sshcheck*) echo "hub-ssh 16/16" ;;
  *"test -e ~/fleet-rotation/ROTATION_PAUSE"*) echo {"paused" if paused else "running"} ;;
  *"cat ~/fleet-rotation/ROTATION_PAUSE"*) echo "manual: lead window" ;;
  *"show -p ActiveState"*) echo inactive ;;
  *"ls -1d backup-*"*) echo backup-20260929T200000Z ;;
  *"rotation.json.new check-config"*) exit {0 if new_config_ok else 2} ;;
esac
exit 0
""")
    (tmp / "bin" / "scp").write_text(f"""#!/bin/sh
printf 'scp %s\\n' "$*" >> {log}
for last; do :; done
case "$last" in rtx5090:*) ;; *) echo copied > "$last" ;; esac
exit 0
""")
    (tmp / "bin" / "sleep").write_text("#!/bin/sh\nexit 0\n")
    for name in ("ssh", "scp", "sleep"):
        (tmp / "bin" / name).chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp / 'bin'}:{os.environ['PATH']}", FLEET_ROTATION_CONFIG=str(PRODUCTION))
    return log, env


def run_install(env, *args):
    p = subprocess.run(["bash", str(REPOSITORY / "tools" / "fleet_rotation_install.sh"), *args], capture_output=True, text=True, env=env, timeout=60)
    return p.returncode, p.stdout + p.stderr


def test_install_upgrade_and_revert():
    with tempfile.TemporaryDirectory() as tmp:
        log, env = install_harness(tmp)
        rc, out = run_install(env, "upgrade")
        lines = log.read_text().splitlines()
        steps = [first_index(lines, p) for p in (r"sshcheck", r"fleet_rotation.py --config \S+ pause upgrade-", r"show -p ActiveState", r"mkdir backup-\d{8}T\d{6}Z && cp -p bin/fleet_rotation.py rotation.json INSTALLED.sha256",
                                                  r"^scp .*fleet_rotation.py rtx5090:fleet-rotation/bin/fleet_rotation.py.new", r"rotation.json.new check-config", r"mv -f bin/fleet_rotation.py.new bin/fleet_rotation.py",
                                                  r"--config \S+ plan", r"--dry-run tick", r"restart fleet-rotation-schedule.service", r"--config \S+ resume")]
        check("upgrade: pause, wait for the tick, back up, install, check, plan and dry-run, then resume", rc == 0 and all(i >= 0 for i in steps) and steps == sorted(steps), f"{steps}\n{out[-400:]}")
    with tempfile.TemporaryDirectory() as tmp:
        log, env = install_harness(tmp, paused=True)
        rc, out = run_install(env, "upgrade")
        lines = log.read_text().splitlines()
        check("upgrade: an existing pause is kept, not overwritten or resumed", rc == 0 and first_index(lines, r" pause upgrade-") < 0 and first_index(lines, r" resume") < 0 and first_index(lines, r"--dry-run tick") >= 0, out[-400:])
    with tempfile.TemporaryDirectory() as tmp:
        log, env = install_harness(tmp)
        rc, out = run_install(env, "upgrade", "--no-resume")
        check("upgrade --no-resume: stays paused", rc == 0 and first_index(log.read_text().splitlines(), r" resume") < 0, out[-300:])
    with tempfile.TemporaryDirectory() as tmp:
        log, env = install_harness(tmp, new_config_ok=False)
        rc, out = run_install(env, "upgrade")
        lines = log.read_text().splitlines()
        check("upgrade: a new config the new tool refuses changes nothing and stays paused", rc != 0 and first_index(lines, r"mv -f bin/fleet_rotation.py.new") < 0 and first_index(lines, r" resume") < 0 and "nothing changed" in out, out[-400:])
    with tempfile.TemporaryDirectory() as tmp:
        log, env = install_harness(tmp)
        rc, out = run_install(env, "revert")
        lines = log.read_text().splitlines()
        steps = [first_index(lines, p) for p in (r" pause revert-", r"show -p ActiveState", r"mkdir revert-", r"^scp -q rtx5090:fleet-rotation/backup-20260929T200000Z/fleet_rotation.py",
                                                  r"mv -f bin/fleet_rotation.py.new", r"--dry-run tick", r" resume")]
        check("revert: restores the newest backup through the same checked swap", rc == 0 and all(i >= 0 for i in steps) and steps == sorted(steps), f"{steps}\n{out[-400:]}")


def test_upgrade_keeps_the_running_companion_for_its_hour():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        w = f.world()
        for h in HOSTS:
            w["nodes"][h]["up"]["wide"] = True
            w["nodes"][h]["ready"]["wide"] = True
            w["nodes"][h]["arena"]["wide"] = 22
            w["nodes"][h]["mem"] -= 22
        w["apis"]["wide"] = True
        f.save(w)
        f.state_dir.mkdir()
        (f.state_dir / "state.json").write_text(json.dumps({"active": ["prod", "wide"], "phase": "steady", "companions": {H17: "wide", hour_key("2026-09-29T16:00:00Z"): "tb"}, "last_companion": "wide"}))
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T17:20:30Z")
        st = f.state()
        check("upgrade state: the companion the old rotation started this hour keeps running, others join", rc == 0 and st.get("active") == ["prod", "wide", "ta"] and not any(re.search(r" (stop|reclaim) wide", l) for l in f.call_lines()), out[-600:])
        check("upgrade state: the old per-hour choices seed the fairness record", st.get("last_run", {}).get("tb") == int(hour_key("2026-09-29T16:00:00Z")) and "companions" not in st and "last_companion" not in st, json.dumps(st.get("last_run")))


def test_dry_run_of_a_running_rotation_shows_the_plan():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T18:00:30Z", dry=True)
        check("dry-run running: plans the next slot from the recorded fleet without an alert", rc == 0 and "DRY n0: fake start big 0" in out and "DRY n2: fake stop tc" in out and "ALERT" not in out and f.call_lines() == [], out[-600:])


def strand(f, name):
    w = f.world()
    for h in f.cfg["models"][name]["nodes"]:
        w["nodes"][h]["up"][name] = False
        w["nodes"][h]["ready"][name] = False
    w["apis"][name] = False
    f.save(w)


def test_resident_arenas_of_a_stopped_model_are_reclaimed():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T15:00:30Z")
        f.tool("pause", "verify", "window")
        strand(f, "big")
        f.tool("resume", at="2026-09-29T15:09:00Z")
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T15:09:39Z")
        lines = f.call_lines()
        st = f.state()
        check("stranded arenas: a primary stopped outside the rotation with its arenas resident does not block production", rc == 0 and st.get("active") == ["prod"] and f.up_on("prod") == HOSTS and f.world()["apis"].get("prod"), out[-800:])
        check("stranded arenas: its packs are reclaimed by pack before production starts", 0 <= first_index(lines, r"n\d reclaim big --reclaim-pack") < first_index(lines, r"n\d start prod") and all(f.world()["nodes"][h]["arena"].get("big") == 0 for h in HOSTS), "\n".join(lines[-12:]))
        check("stranded arenas: no auto-pause, no node-global reclaim", not (f.state_dir / "ROTATION_PAUSE").exists() and "CRITICAL" not in f.alerts() and not any(re.search(r"--reclaim(?![-\w])", l) for l in lines), f.alerts())
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        f.tool("tick", at="2026-09-29T18:00:30Z")
        f.tool("pause", "verify", "window")
        strand(f, "big")
        f.tool("resume", at="2026-09-29T18:09:00Z")
        rc, out = f.tool("tick", at="2026-09-29T18:09:39Z")
        st = f.state()
        check("stranded arenas: with a companion still serving, the slot is re-established from reclaimed memory", rc == 0 and st.get("active") == ["big", "ta"] and f.up_on("big") == HOSTS and min(f.world()["nodes"][h]["mem"] for h in HOSTS) >= 20 and "CRITICAL" not in f.alerts(), out[-800:])
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T15:00:30Z")
        f.state_dir.joinpath("state.json").write_text(json.dumps(dict(f.state(), phase="recovering", transition={"from": ["big"], "to": ["prod"], "reason": "x", "started": "x"})))
        strand(f, "big")
        rc, out = f.tool("tick", at="2026-09-29T15:20:30Z")
        check("stranded arenas: a fallback reclaims idle packs before it starts production", f.state().get("active") == ["prod"] and f.up_on("prod") == HOSTS and "CRITICAL" not in f.alerts(), out[-800:])


def test_api_down_with_engines_up_restarts_the_api():
    for paused in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fleet(tmp)
            f.tool("tick", at="2026-09-29T15:00:30Z")
            if paused:
                f.tool("pause", "verify", "window")
            w = f.world()
            w["apis"]["big"] = False
            f.save(w)
            if paused:
                f.tool("resume", at="2026-09-29T15:09:00Z")
            f.clear_calls()
            rc, out = f.tool("tick", at="2026-09-29T15:09:39Z")
            lines = f.call_lines()
            label = "after a pause" if paused else "in a steady hour"
            check(f"api down ({label}): the api is restarted and smoked, the engines are kept", rc == 0 and f.state().get("active") == ["big"] and f.world()["apis"].get("big") and f.up_on("big") == HOSTS and starts(lines) == [] and not any(" stop big" in l or "reclaim big" in l for l in lines) and any("smoke big" in l for l in lines), out[-800:])
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T15:00:30Z")
        w = f.world()
        w["apis"]["big"] = False
        f.save(w)
        f.behave(smoke_bad=["big"])
        rc, out = f.tool("tick", at="2026-09-29T15:09:39Z")
        check("api down: a failed api restart falls back with the engines stopped and reclaimed", f.state().get("active") == ["prod"] and f.up_on("big") == [] and f.up_on("prod") == HOSTS and "api restart failed" in f.alerts(), out[-800:])
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        w = f.world()
        for h in HOSTS:
            w["nodes"][h]["up"]["big"] = True
            w["nodes"][h]["arena"]["big"] = 0
        f.save(w)
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T17:05:30Z")
        check("api down: an engine the rotation did not start is never given an api", not f.world()["apis"].get("big") and not any("api-start big" in l for l in f.call_lines()) and (f.state_dir / "ROTATION_PAUSE").exists(), out[-600:])


def test_fallback_tries_production_past_a_stuck_model():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        f.tool("tick", at="2026-09-29T18:00:30Z")
        w = f.world()
        for h in HOSTS:
            w["nodes"][h]["up"]["big"] = False
        f.save(w)
        f.behave(stop_stuck=["ta"])
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T18:10:30Z")
        st = f.state()
        check("stuck in fallback: production is still restored before the rotation pauses", f.up_on("prod") == HOSTS and f.world()["apis"].get("prod") and st.get("phase") == "degraded" and "prod" in st.get("active") and (f.state_dir / "ROTATION_PAUSE").exists(), out[-800:])
        check("stuck in fallback: the stuck companion is not reclaimed", not any("reclaim ta" in l for l in f.call_lines()) and "CRITICAL" in f.alerts() and "ta:" in f.alerts(), f.alerts())


def test_steady_floor_guard_sheds_a_companion():
    with tempfile.TemporaryDirectory() as tmp:
        f = pack_fleet(tmp)
        f.tool("tick", at="2026-09-29T17:00:30Z")
        w = f.world()
        w["nodes"]["n1"]["mem"] = 15
        f.save(w)
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T17:10:30Z")
        lines = f.call_lines()
        check("steady floor: a node under the floor sheds only the companion on it", f.state().get("active") == ["prod", "ta", "tc"] and f.up_on("tb") == [] and not any(re.search(r" (stop|reclaim) (prod|ta|tc)", l) for l in lines) and "below 20 GiB on n1=15" in f.alerts(), "\n".join(lines[-8:]))
        w = f.world()
        w["nodes"]["n3"]["mem"] = 15
        f.save(w)
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T17:15:30Z")
        check("steady floor: with no companion on the low node the primary is left serving and alerted", rc == 0 and f.state().get("active") == ["prod", "ta", "tc"] and not any(" stop " in l for l in f.call_lines()) and "no companion there to shed" in f.alerts(), f.alerts())


def real_footprints(f, base, real):
    w = f.world()
    for h in HOSTS:
        w["nodes"][h]["mem"] = base - real["prod"]
        w["nodes"][h]["arena"]["prod"] = real["prod"]
    w["arena"].update(real)
    f.save(w)


def incident_config(tmp, small_gib):
    cfg = config(Path(tmp))
    cfg["node_free_gib"] = 106
    cfg["slots"]["full"]["companions"] = ["q", "g", "m"]
    cfg["slots"]["flash_plus"]["companions"] = ["q", "g", "m"]
    cfg["rollback_models"] = ["prod", "q"]
    for name in ("c1", "c2", "c3", "dflt"):
        del cfg["models"][name]
    cfg["models"].update({"q": model(["n0"], small_gib, port=9020), "m": model(["n1"], 46, port=9021), "g": model(["n2"], 36, port=9022)})
    return cfg


def incident_fleet(tmp, small_gib, prod_real=27):
    f = Fleet(tmp, incident_config(tmp, small_gib))
    real_footprints(f, 109, {"prod": prod_real, "big": 64, "q": 26, "m": 42, "g": 36})
    return f


H00 = hour_key("2026-09-30T00:00:00Z")


def test_incident_full_hour_serves_the_primary():
    for small_gib, how in ((15, "stopped after its re-add breaks the floor"), (26, "skipped by the start gate")):
        with tempfile.TemporaryDirectory() as tmp:
            f = incident_fleet(tmp, small_gib)
            rc, out = f.tool("tick", at="2026-09-29T23:50:20Z")
            check(f"incident {small_gib}: the flash_plus hour packs every companion", f.state().get("active") == ["prod", "q", "g", "m"], out[-600:])
            f.clear_calls()
            rc, out = f.tool("tick", at="2026-09-30T00:00:21Z")
            st = f.state()
            lines = f.call_lines()
            log = (f.state_dir / "rotation.log").read_text()
            check(f"incident {small_gib}: the full hour packs the running companion beside the primary, as at 00:00Z", st["picks"][H00]["companions"] == ["q"] and "n1:" in st["picks"][H00]["waiting"]["m"], json.dumps(st["picks"].get(H00)))
            check(f"incident {small_gib}: the primary's gate is short only where the kept companion holds memory", "goes first: its start gate is short on n0 " in f.alerts() and "short on n0=83" in log, f.alerts())
            order = [first_index(lines, p) for p in (r"n0 stop prod", r"n0 stop q", r"n0 reclaim q --reclaim-pack", r"n\d start big", r"fakehub smoke big")]
            check(f"incident {small_gib}: the companion is stopped and reclaimed by pack, then the primary starts", all(i >= 0 for i in order) and order == sorted(order), str(order))
            check(f"incident {small_gib}: the primary serves the hour with no fallback", st.get("active") == ["big"] and f.up_on("big") == HOSTS and f.world()["apis"].get("big") and st.get("phase") == "steady"
                  and st.get("failed_instance") is None and "FALLBACK" not in log and not (f.state_dir / "ROTATION_PAUSE").exists(), out[-900:])
            check(f"incident {small_gib}: the companion is {how}", f.up_on("q") == [] and f.world()["nodes"]["n0"]["arena"].get("q") == 0 and "q" in st["dropped"][H00]
                  and ("below 20 GiB after-q" in f.alerts() if small_gib == 15 else "yielded to the primary" in f.alerts() and not any("n0 start q" in l for l in lines)), f.alerts())
            check(f"incident {small_gib}: every node keeps the floor", min(f.world()["nodes"][h]["mem"] for h in HOSTS) >= 20, json.dumps(f.world()["nodes"]))


def test_primary_first_order_and_readd():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = config(Path(tmp))
        cfg["slots"]["full"]["companions"] = ["s1", "s2", "s3"]
        cfg["slots"]["flash_plus"]["companions"] = ["s1", "s2", "s3"]
        cfg["rollback_models"] = ["prod", "s1"]
        for name in ("c1", "c2", "c3", "dflt"):
            del cfg["models"][name]
        cfg["models"].update({"s1": model(["n0"], 10, port=9030), "s2": model(["n0"], 10, port=9031), "s3": model(["n1"], 10, port=9032)})
        f = Fleet(tmp, cfg)
        real_footprints(f, 110, {"prod": 27, "big": 68, "s1": 12, "s2": 12, "s3": 12})
        f.tool("tick", at="2026-09-29T17:00:30Z")
        check("primary first: the small companions run beside production", f.state().get("active") == ["prod", "s1", "s2", "s3"], json.dumps(f.state().get("active")))
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T18:00:30Z")
        lines = f.call_lines()
        st = f.state()
        order = [first_index(lines, p) for p in (r"n0 stop prod", r"n0 stop s2", r"n0 reclaim s2 --reclaim-pack", r"n0 stop s1", r"n0 reclaim s1 --reclaim-pack", r"n0 start big", r"n0 start s1", r"n0 start s2")]
        check("primary first: kept companions stop in reverse start order, each reclaimed by pack, before the primary starts; re-added after it", all(i >= 0 for i in order) and order == sorted(order), f"{order}\n" + "\n".join(lines))
        check("primary first: a kept companion off the short nodes is never stopped", st.get("active") == ["big", "s1", "s3"] and f.up_on("s3") == ["n1"] and not any(" stop s3" in l or "reclaim s3" in l for l in lines), "\n".join(lines))
        check("primary first: the companion that still fits is back, the one that does not is stopped and skipped", "s1" in st.get("active", []) and f.up_on("s1") == ["n0"] and f.up_on("s2") == [] and "s2" in st["dropped"][hour_key("2026-09-29T18:00:00Z")], out[-800:])
        check("primary first: no fallback, the floor holds", st.get("phase") == "steady" and "FALLBACK" not in (f.state_dir / "rotation.log").read_text() and min(f.world()["nodes"][h]["mem"] for h in HOSTS) >= 20, out[-400:])


def test_measured_capacity_packing():
    with tempfile.TemporaryDirectory() as tmp:
        f = incident_fleet(tmp, 15, prod_real=38)
        f.tool("tick", at="2026-09-29T23:50:20Z")
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-30T00:00:21Z")
        st = f.state()
        lines = f.call_lines()
        check("measured: a running companion is charged what it really holds, not its config", st["picks"][H00]["companions"] == [] and st["picks"][H00]["waiting"]["q"] == "n0:15<20", json.dumps(st["picks"].get(H00)))
        check("measured: it stops with the outgoing models, before the primary, and the gate never refuses", 0 <= first_index(lines, r"n0 stop q") < first_index(lines, r"n\d start big") and "goes first" not in f.alerts() and st.get("active") == ["big"], out[-600:])
        check("measured: the pick records the predicted MemAvailable per node", st["picks"][H00]["after_gib"] == {"n0": 30, "n1": 38, "n2": 38, "n3": 38}, json.dumps(st["picks"][H00].get("after_gib")))
    for mem, gib, admitted, label in ((109 - 27, 50, True, "admits a companion beside what production really holds"), (160, 90, False, "never counts more than node_free_gib")):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = incident_config(tmp, 15)
            cfg["slots"]["flash_plus"]["companions"].append("fat")
            cfg["models"]["fat"] = model(["n3"], gib, port=9023)
            f = Fleet(tmp, cfg)
            real_footprints(f, 109, {"prod": 27, "big": 64, "q": 26, "m": 42, "g": 36, "fat": gib})
            w = f.world()
            w["nodes"]["n3"]["mem"] = mem
            f.save(w)
            rc, out = f.tool("tick", at="2026-09-29T17:00:30Z")
            st = f.state()
            waiting = st["picks"][hour_key("2026-09-29T17:00:00Z")]["waiting"]
            ok = "fat" in st.get("active", []) and min(f.world()["nodes"][h]["mem"] for h in HOSTS) >= 20 if admitted else "fat" not in st.get("active", []) and waiting.get("fat") == "n3:16<20"
            check(f"measured: {label}", ok, json.dumps(st.get("picks")) + out[-400:])


def test_failed_primary_after_yield_and_force_retry():
    with tempfile.TemporaryDirectory() as tmp:
        f = incident_fleet(tmp, 15)
        f.tool("tick", at="2026-09-29T23:50:20Z")
        f.behave(never_ready=["big"])
        rc, out = f.tool("tick", at="2026-09-30T00:00:21Z")
        st = f.state()
        check("failure path: a primary that fails after the companions yielded falls back to production, never to nothing", st.get("active") == ["prod"] and f.up_on("prod") == HOSTS and f.world()["apis"].get("prod")
              and st.get("phase") == "fallback" and not (f.state_dir / "ROTATION_PAUSE").exists() and f.up_on("big") == [] and f.up_on("q") == [], out[-800:])
        check("failure path: the hour is fenced", st.get("failed_instance") == int(H00), json.dumps(st)[:300])
        rc, out = f.tool("force", "full", at="2026-09-30T00:05:00Z")
        st = f.state()
        check("force: clears the failed hour and logs it", rc == 0 and "failed_instance" not in st and "cleared=failed_instance@2026-09-30T00:00:00Z" in (f.state_dir / "rotation.log").read_text(), out)
        f.behave(never_ready=[])
        rc, out = f.tool("tick", at="2026-09-30T00:05:21Z")
        check("force: the failed slot is retried without editing state", f.state().get("active") == ["big"] and f.up_on("big") == HOSTS and f.state().get("phase") == "steady", out[-600:])


def test_production_config():
    cfg = json.loads(PRODUCTION.read_text())
    p = subprocess.run([sys.executable, str(TOOL), "--config", str(PRODUCTION), "check-config"], capture_output=True, text=True)
    check("production config: valid", p.returncode == 0, p.stdout + p.stderr)
    source = TOOL.read_text()
    named = [m for m in cfg["models"] if re.search(r"\b" + re.escape(m) + r"\b", source)]
    check("tool is model-neutral: no configured model name in the source", not named, str(named))
    check("tool has no comments", not any(l.strip().startswith("#") for l in source.splitlines()[1:]))
    check("production: every runnable model reclaims by pack", all(m.get("packs") for m in cfg["models"].values() if m.get("runnable")))
    check("production: fallback spans the fleet and rollback includes it", cfg["fallback"] in cfg["rollback_models"])


def main():
    for test in (test_schedule_and_rotation, test_adopt_and_full_transition, test_ready_timeout_falls_back, test_engine_death_and_start_failure,
                 test_ssh_timeout, test_smoke_failure, test_fallback_failure_pauses, test_preemption_and_resync,
                 test_pause_resume_and_foreign_change, test_floor_and_prediction, test_steady_health_and_manual,
                 test_stuck_stop_and_reclaim_guard, test_interrupted_transition_recovers, test_pause_during_running_tick,
                 test_unexpected_error_falls_back, test_failed_health_fallback_reports_degraded, test_mixed_production_is_not_held, test_dry_run_executes_nothing, test_schedule_document_and_sync, test_config_validation,
                 test_exclusive_floor_and_abort, test_exclusive_floor_validation, test_pack_cache_trim,
                 test_dry_run_while_paused_shows_the_plan, test_failure_after_the_hour_boundary_marks_the_starting_hour, test_install_rollback_waits_for_a_running_tick, test_install_upgrade_and_revert,
                 test_packing_fills_the_nodes, test_packing_per_node_floor, test_partial_companion_failure, test_steady_companion_loss_and_health,
                 test_rotation_fairness_across_cycles, test_full_slot_with_companion, test_packing_schedule_document_and_force, test_plan_is_read_only,
                 test_smoke_reply_rules, test_packing_config_validation, test_upgrade_keeps_the_running_companion_for_its_hour,
                 test_dry_run_of_a_running_rotation_shows_the_plan,
                 test_resident_arenas_of_a_stopped_model_are_reclaimed, test_api_down_with_engines_up_restarts_the_api,
                 test_fallback_tries_production_past_a_stuck_model, test_steady_floor_guard_sheds_a_companion,
                 test_incident_full_hour_serves_the_primary, test_primary_first_order_and_readd, test_measured_capacity_packing,
                 test_failed_primary_after_yield_and_force_retry, test_production_config):
        test()
    if failures:
        print(f"\n{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
