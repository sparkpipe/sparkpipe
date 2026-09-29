#!/usr/bin/env python3
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
        out = "garbage" if model in b.get("smoke_bad", []) else '{"text": "Paris"}'
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


def model(nodes, mem, api=True, runnable=True, validated=False):
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
                    "health": "fake api-health @model@", "port": 9000, "timeout_s": 1}
        m["smoke"] = {"path": "/v1/completions", "body": {"prompt": "x"}, "expect": "Paris", "timeout_s": 1}
    return m


def config(tmp):
    return {
        "hub": "fakehub", "state_dir": str(tmp / "state"), "ssh": [str(tmp / "bin" / "fakessh")], "ssh_timeout_s": 3, "poll_s": 0.05,
        "floor_gib": 20, "weightd_warm": "/w/weightd_warm", "weightd_socket": "/tmp/s.sock", "schedule_port": 0,
        "fleet": HOSTS, "cycle": {"anchor_hour": 0, "slots": ["full", "k3", "flash_plus"]},
        "slots": {"full": {"base": "big", "fallback_slot": "flash_plus"}, "k3": {"base": "k", "fallback_slot": "flash_plus"},
                  "flash_plus": {"base": "prod", "companion": True, "fallback_slot": None}},
        "fallback": "prod", "companions": ["c1", "c3", "c2"], "default_companion": "dflt", "rollback_models": ["prod", "dflt"],
        "lock": {"pause_file": "ROTATION_PAUSE", "mirror_pause_file": "lock/ROTATION_PAUSE", "holder_file": "lock/PERF_HOLDER", "holder_prefixes": ["lead-"]},
        "commands": {"mem_probe": "fake mem", "reclaim": "fake reclaim @model@ --reclaim-pack @packs@", "smoke": "fake smoke @model@ @port@ @path@"},
        "models": {"prod": model(HOSTS, 38, validated=True), "big": model(HOSTS, 68), "k": model([], 0, runnable=False),
                   "c1": model(["n0", "n1"], 22), "c2": model(["n2", "n3"], 46), "c3": model([], 0, runnable=False),
                   "dflt": model(["n0", "n1"], 15, validated=True)},
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
        arena = {"prod": 38, "big": 68, "c1": 22, "c2": 46, "dflt": 15}
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


def test_dry_run_while_paused_shows_the_plan():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.tool("pause", "install", at="2026-09-29T17:00:30Z")
        f.tool("force", "flash_plus", "c2", at="2026-09-29T17:00:30Z")
        before = (f.state_dir / "state.json").read_text()
        f.clear_calls()
        rc, out = f.tool("tick", at="2026-09-29T17:01:00Z", dry=True)
        check("dry-run paused: reports the hold-off and still prints the plan", "HOLD-OFF" in out and "DRY n2: fake start c2 0" in out and "DRY hub: fake smoke c2" in out, out[-600:])
        check("dry-run paused: executes nothing, keeps the pause and the state", f.call_lines() == [] and (f.state_dir / "ROTATION_PAUSE").exists() and (f.state_dir / "state.json").read_text() == before)


def test_failure_after_the_hour_boundary_marks_the_starting_hour():
    with tempfile.TemporaryDirectory() as tmp:
        f = Fleet(tmp)
        f.behave(never_ready=["c1"])
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
                 test_dry_run_while_paused_shows_the_plan, test_failure_after_the_hour_boundary_marks_the_starting_hour, test_install_rollback_waits_for_a_running_tick,
                 test_production_config):
        test()
    if failures:
        print(f"\n{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
