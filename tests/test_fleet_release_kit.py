#!/usr/bin/env python3
import base64
import hashlib
import http.server
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
KIT = REPOSITORY / "tools" / "fleet_release"
PROFILE = REPOSITORY / "deployment" / "glm5_next_tp16" / "release"
failures = []


def load(name):
    spec = importlib.util.spec_from_file_location(name, KIT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(name, cond, detail=""):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name} {detail}")
        failures.append(name)


FAKE_SSH = r'''#!/usr/bin/env python3
import os, re, subprocess, sys
args = sys.argv[1:]
while args and args[0] == "-o":
    args = args[2:]
host, command = args[0], " ".join(args[1:])
fake = os.environ["FAKE_ROOT"]
data = sys.stdin.buffer.read() if ("bash -s" in command or "python3 -" in command) else b""
with open(os.path.join(fake, "calls.log"), "a") as f:
    f.write(f"{host}\t{command}\n")
with open(os.path.join(fake, f"stdin.{host}"), "wb") as f:
    f.write(data)
canned = os.path.join(fake, f"{host}.probe")
if command == "bash -s" and b"host=$h agent=" in data and os.path.exists(canned):
    row = open(canned).read().strip()
    root = re.search(rb"ROOT_NAME=(\S+)", data).group(1).decode().strip("'")
    if os.path.exists(os.path.join(fake, host, "sparkdata", root, "agent.hold")):
        row = re.sub(r"hold=\S+", "hold=yes", row)
        row = re.sub(r"eng_n=\S+", "eng_n=0", row)
    print(row)
    sys.exit(0)
home = os.path.join(fake, host)
os.makedirs(home, exist_ok=True)
env = dict(os.environ, HOME=home, PATH=os.path.join(fake, "bin") + os.pathsep + os.environ["PATH"])
sys.exit(subprocess.run(["bash", "-c", command], input=data, env=env, cwd=home).returncode)
'''


def fake_world(tmp, hosts):
    root = Path(tmp)
    (root / "bin").mkdir(parents=True, exist_ok=True)
    ssh = root / "ssh"
    ssh.write_text(FAKE_SSH)
    ssh.chmod(0o755)
    systemctl = root / "bin" / "systemctl"
    systemctl.write_text("#!/bin/sh\ncat \"$HOME/unit-state\" 2>/dev/null || echo inactive\n")
    systemctl.chmod(0o755)
    for h in hosts:
        (root / h).mkdir(exist_ok=True)
    return root, f"{sys.executable} {ssh}"


def test_nodes_rules_and_resample():
    nodes = load("nodes")
    rules = nodes.parse_expect(["hold=yes", "mem_gib>=20", "wd!=none", "kvref<=0"])
    row = {"host": "a", "hold": "yes", "mem_gib": "25", "wd": "abc", "kvref": "0"}
    check("nodes: all rules pass", nodes.violations(row, rules) == [])
    bad = nodes.violations({"host": "a", "hold": "no", "mem_gib": "x", "wd": "none", "kvref": "2"}, rules)
    check("nodes: each rule reports", len(bad) == 4, bad)
    check("nodes: a column the probe lacks fails", nodes.violations({"host": "a"}, nodes.parse_expect(["kvs=1of16"])) == ["kvs missing from the probe"])
    check("nodes: an ssh error is a violation", nodes.violations({"host": "a", "error": "ssh-timeout"}, rules) == ["ssh-timeout"])
    try:
        nodes.parse_expect(["hold"])
        check("nodes: malformed rule refused", False)
    except nodes.ConfigError:
        check("nodes: malformed rule refused", True)
    cfg = {"nodes": ["n0", "n1"]}
    calls = {"n0": 0, "n1": 0}

    def flaky(host):
        calls[host] += 1
        ok = host == "n0" or calls[host] >= 2
        return {"host": host, "hold": "yes" if ok else "no"}

    out = []
    rc = nodes.gate(cfg, nodes.parse_expect(["hold=yes"]), 0, 0, 2, 0, False, None, probe_fn=flaky, sleep=lambda s: None, out=out.append)
    check("nodes: a transient failure clears on resample", rc == 0 and any("TRANSIENT n1" in l for l in out), out)
    out = []
    rc = nodes.gate(cfg, nodes.parse_expect(["hold=yes"]), 0, 0, 2, 0, False, None, probe_fn=lambda h: {"host": h, "hold": "no" if h == "n1" else "yes"}, sleep=lambda s: None, out=out.append)
    check("nodes: a persistent failure fails after the resamples", rc == 1 and out[-1].startswith("NODES-FAIL 1") and not any("TRANSIENT" in l for l in out), out)
    t = [0.0]

    def clock():
        return t[0]

    def sleep(s):
        t[0] += s

    seen = []

    def slow(host):
        seen.append(host)
        return {"host": host, "hold": "yes" if t[0] >= 30 else "no"}

    out = []
    rc = nodes.gate(cfg, nodes.parse_expect(["hold=yes"]), 1, 10, 2, 0, True, None, probe_fn=slow, sleep=sleep, clock=clock, out=out.append)
    check("nodes: the wait loop polls until the rule holds", rc == 0 and t[0] == 30 and out[-1].startswith("NODES-OK 2/2"), out)


def test_nodes_cli_sends_rank_pack(tmp):
    root, ssh = fake_world(tmp, ["na", "nb"])
    (root / "na.probe").write_text("host=na agent=active hold=no eng_n=1 kvs=1of2\n")
    (root / "nb.probe").write_text("host=nb agent=active hold=no eng_n=1 kvs=shard0\n")
    extra = root / "extra.sh"
    extra.write_text("probe_extra() { echo kvs=x; }\n")
    env = dict(os.environ, FAKE_ROOT=str(root), RELEASE_SSH=ssh, RELEASE_NODES="na nb", RELEASE_ROOT_NAME="rootx",
               RELEASE_RANK_PACK="packs/rootx.rank@hex@.sp", RELEASE_DRIVER="stages/d.so", RELEASE_WEIGHTD_STAGE="stage", RELEASE_PROBE_EXTRA=str(extra))
    r = subprocess.run([sys.executable, str(KIT / "nodes.py"), "--expect", "kvs=1of2", "--resample", "0"], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    check("nodes cli: the extra column gates", r.returncode == 1 and "BAD nb" in r.stdout and "kvs=shard0 want 1of2" in r.stdout, r.stdout + r.stderr)
    sent = (root / "stdin.nb").read_text()
    check("nodes cli: rank 1 gets pack rank1 and the extra function", "RANK_PACK=packs/rootx.rank1.sp" in sent and "probe_extra()" in sent and sent.index("probe_extra()") < sent.index("host=$h agent="), sent[:300])
    env["RELEASE_RANK_PACK"] = "packs/rootx.sp"
    r = subprocess.run([sys.executable, str(KIT / "nodes.py")], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    check("nodes cli: a rank pack without @hex@ is refused", r.returncode == 2 and "@hex@" in r.stderr, r.stderr)
    del env["RELEASE_PROBE_EXTRA"]
    r = subprocess.run([sys.executable, str(KIT / "nodes.py")], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    check("nodes cli: missing env is refused", r.returncode == 2 and "RELEASE_PROBE_EXTRA" in r.stderr, r.stderr)


def test_engine_logs(tmp):
    root, ssh = fake_world(tmp, ["na", "nb"])
    for h, lines in (("na", ["EXEC mode=graph", "CAPTURE-OK rows=1 x", "ERRSITE a.c:1 status=15", "ERRSITE a.c:1 status=15"]),
                     ("nb", ["EXEC mode=eager", "ERRSITE b.c:9 status=3"])):
        d = root / h / "sparkdata" / "rootx"
        d.mkdir(parents=True)
        (d / "engine.log").write_text("\n".join(lines) + "\n")
    spec = {"sec": {"log": "engine.log", "gates": [
        {"kind": "last", "name": "mode", "pattern": "EXEC mode=[a-z]+", "equals": "EXEC mode=graph"},
        {"kind": "count", "name": "captures", "pattern": "CAPTURE-OK rows=1 ", "min": 1},
        {"kind": "report", "name": "modes", "pattern": "mode=[a-z]+"}],
        "errsite": {"pattern": "ERRSITE [^ ]* status=[0-9]+", "allow": ["a\\.c:1 status=15$"]}}}
    checks = root / "checks.json"
    checks.write_text(json.dumps(spec))
    env = dict(os.environ, FAKE_ROOT=str(root), RELEASE_SSH=ssh, RELEASE_NODES="na nb", RELEASE_ROOT_NAME="rootx")
    r = subprocess.run([sys.executable, str(KIT / "engine_logs.py"), "--checks", str(checks), "--section", "sec"], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    out = r.stdout
    check("engine_logs: last/count/errsite failures name the node", r.returncode == 1 and "nb mode: last 'EXEC mode=eager'" in out and "nb captures: 0 < min 1" in out and "ERRSITE-UNKNOWN ERRSITE b.c:9 status=3 on nb" in out, out)
    check("engine_logs: allowed errsites are counted, not failed", "errsite      2 ERRSITE a.c:1 status=15" in out and "a.c:1 status=15  UNKNOWN" not in out, out)
    (root / "nb" / "sparkdata" / "rootx" / "engine.log").write_text("EXEC mode=graph\nCAPTURE-OK rows=1 y\n")
    r = subprocess.run([sys.executable, str(KIT / "engine_logs.py"), "--checks", str(checks), "--section", "sec"], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    check("engine_logs: a clean fleet passes", r.returncode == 0 and "ENGINE-LOGS sec PASS on 2 nodes" in r.stdout, r.stdout)
    (root / "nb" / "sparkdata" / "rootx" / "engine.log").unlink()
    r = subprocess.run([sys.executable, str(KIT / "engine_logs.py"), "--checks", str(checks), "--section", "sec"], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    check("engine_logs: a missing log fails", r.returncode == 1 and "nb: no engine.log" in r.stdout, r.stdout)
    spec["sec"]["gates"].append({"kind": "count", "name": "no bound", "pattern": "x"})
    checks.write_text(json.dumps(spec))
    r = subprocess.run([sys.executable, str(KIT / "engine_logs.py"), "--checks", str(checks), "--section", "sec"], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    check("engine_logs: a count gate without bounds is refused", r.returncode == 2, r.stderr)


class Api(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, doc):
        body = json.dumps(doc).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, {"tokenizer": True})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions":
            if req["messages"][0]["role"] == "tool":
                return self.reply(400, {"error": {"code": "role_unsupported"}})
            return self.reply(200, {"choices": [{"message": {"content": self.server.answer}}], "usage": {"prompt_tokens": 16}})
        self.reply(200, {"choices": [{"text": " Paris."}]})


def test_hub_smoke():
    smoke = load("hub_smoke")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Api)
    server.answer = "Paris"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        spec = json.loads((PROFILE / "checks.json").read_text())["smoke"]
        paris_only = {"requests": [r for r in spec["requests"] if r["name"] in ("health", "warmup ids", "chat Paris", "role tool refused", "text completion")]}
        out = []
        rc = smoke.run(f"http://127.0.0.1:{server.server_port}", paris_only, out=out.append)
        check("hub_smoke: the profile's requests pass against a conforming API", rc == 0 and out[-1] == "HUB-SMOKE PASS", out)
        server.answer = "<think>x</think> Paris"
        out = []
        rc = smoke.run(f"http://127.0.0.1:{server.server_port}", paris_only, out=out.append)
        check("hub_smoke: a think block fails the chat", rc == 1 and any("contains '<think>'" in l for l in out), out)
        out = []
        rc = smoke.run(f"http://127.0.0.1:{server.server_port}", {"requests": [{"name": "tool accepted", "path": "/v1/chat/completions", "body": {"messages": [{"role": "tool", "content": "x"}]}, "text_path": "choices.0.message.content", "contains": ["x"]}]}, out=out.append)
        check("hub_smoke: an unexpected status fails", rc == 1 and "HTTP 400 want 200" in out[0], out)
        r = subprocess.run([sys.executable, str(KIT / "hub_smoke.py"), "--port", str(server.server_port), "--spec-b64", base64.b64encode(json.dumps(paris_only).encode()).decode()], capture_output=True, text=True)
        check("hub_smoke: the base64 spec path works", r.returncode == 1 and "HUB-SMOKE FAIL" in r.stdout, r.stdout)
    finally:
        server.shutdown()
    try:
        smoke.check_spec({"requests": [{"name": "x", "path": "/", "contains": ["a"]}]})
        check("hub_smoke: contains without text_path refused", False)
    except smoke.SpecError:
        check("hub_smoke: contains without text_path refused", True)


def test_perf_summary(tmp):
    perf = load("perf_summary")
    d = Path(tmp)
    doc = json.loads((PROFILE / "perf_expect.json").read_text())
    perf_ok = perf.load_expect(PROFILE / "perf_expect.json")
    check("perf_summary: the profile's expectations load", perf_ok["metrics"] == doc["metrics"])
    (d / "b1.json").write_text(json.dumps({"128": {"median_wall_tok_s": 43.0, "median_ms_per_token": 21.0}, "512": {"median_wall_tok_s": 45.0, "median_ms_per_token": 21.0}}))
    (d / "ttft.json").write_text(json.dumps({"372": {"median_ttft_ms": 1300}, "742": {"median_ttft_ms": 2500}}))
    (d / "conc.json").write_text(json.dumps({"8": {"median_aggregate_tok_s": 40.0}, "16": {"median_aggregate_tok_s": 80.0}}))
    (d / "conc_warm.json").write_text(json.dumps({"8": {"median_aggregate_tok_s": 120.0}, "16": {"median_aggregate_tok_s": 150.0}}))
    (d / "budget.txt").write_text("path rows n share a b c 6.5 d e f g h\nlinear 1 5 1.00 329 329 176 6.46 105 4.8 39 0 284\n")
    out = []
    rc = perf.summarize(doc, str(d), out=out.append)
    text = "\n".join(out)
    check("perf_summary: in-range numbers pass and a cold drop is reported only", rc == 0 and "PERF-SUMMARY regressions/missing=0" in text and re.search(r"^cold_8 +40\.00 .* BELOW ", text, re.M), text)
    check("perf_summary: every B1 number carries a roofline line", "b1_128_ms: roofline @B=1" in text and "memory " in text and "compute ~1-2%" in text and "peer 6.46 ms" in text, text)
    (d / "ttft.json").write_text(json.dumps({"372": {"median_ttft_ms": 1600}}))
    out = []
    rc = perf.summarize(doc, str(d), out=out.append)
    text = "\n".join(out)
    check("perf_summary: a slower ttft regresses and a missing one counts", rc == 1 and "REGRESSION" in text and "ttft_742" in text and "MISSING" in text and "regressions/missing=2" in text, text)


def test_stage_config_patch(tmp):
    scp = load("stage_config_patch")
    prod = Path(tmp) / "prod"
    (prod / "config").mkdir(parents=True)
    rows = []
    for r in range(2):
        text = json.dumps({"tp_rank": r, "tp_degree": 2, "rows": 128, "nested": {"timeout": 5}}, indent=1)
        (prod / f"config/stage_{r:02d}.json").write_text(text)
        rows.append(f"{hashlib.sha256(text.encode()).hexdigest()}  config/stage_{r:02d}.json")
    (prod / "MANIFEST").write_text("\n".join(rows) + "\n")
    gen = Path(tmp) / "gen.py"
    gen.write_text("import json, sys, os\nout = sys.argv[sys.argv.index('--output') + 1]\nos.makedirs(out + '/config')\n"
                   "for r in range(2):\n    open(out + f'/config/stage_{r:02d}.json', 'w').write(json.dumps({'tp_rank': r, 'tp_degree': 2, 'rows': 1024, 'nested': {'timeout': 5}, 'shard': 1}, indent=1))\n")
    added = scp.parse_add(["shard=1"])
    out = []
    generated = scp.run_generator(f"{sys.executable} {gen}", 2)
    fail = scp.patch(prod, Path(tmp) / "new", 2, added, generated, None, {"rows": "production serves 128"}, out=out.append)
    new0 = (Path(tmp) / "new/config/stage_00.json").read_text()
    check("stage_config_patch: appends the member, keeps the byte format, explains drift", fail == 0 and new0 == json.dumps({"tp_rank": 0, "tp_degree": 2, "rows": 128, "nested": {"timeout": 5}, "shard": 1}, indent=1) and "rows: 128 | 1024  -- production serves 128" in "\n".join(out), out)
    out = []
    fail = scp.patch(prod, Path(tmp) / "new2", 2, added, generated, None, {}, out=out.append)
    check("stage_config_patch: unexplained generator drift fails", fail == 2 and "unexplained members ['rows']" in "\n".join(out), out)
    out = []
    check("stage_config_patch: check accepts exactly the appended member", scp.check(prod, Path(tmp) / "new", 2, added, out=out.append) == 0, out)
    bad = json.loads(new0)
    bad["rows"] = 64
    (Path(tmp) / "new/config/stage_00.json").write_text(json.dumps(bad, indent=1))
    out = []
    check("stage_config_patch: check refuses any other change", scp.check(prod, Path(tmp) / "new", 2, added, out=out.append) == 1, out)
    (prod / "config/stage_01.json").write_text(json.dumps({"tp_rank": 1, "tp_degree": 2, "rows": 128, "nested": {"timeout": 5}}, indent=2))
    out = []
    fail = scp.patch(prod, Path(tmp) / "new3", 2, added, None, None, None, out=out.append)
    check("stage_config_patch: a file that differs from the MANIFEST is refused", fail == 1 and "does not match the production MANIFEST" in "\n".join(out), out)
    try:
        scp.parse_add(["shard"])
        check("stage_config_patch: malformed --add refused", False)
    except scp.ConfigError:
        check("stage_config_patch: malformed --add refused", True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    lines = []
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and rel != "MANIFEST":
            lines.append(f"{sha(p)}  {rel}")
    (Path(root) / "MANIFEST").write_text("\n".join(lines) + "\n")
    return sha(Path(root) / "MANIFEST")


def hub_run(home, script, env):
    full = dict(os.environ, HOME=str(home), **env)
    return subprocess.run(["bash", str(KIT / "hub" / script)], env=full, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)


def test_hub_publish_and_rollback(tmp):
    home = Path(tmp) / "hub"
    served = home / "release" / "rootx"
    for rel, text in (("bin/engine", "old engine"), ("lib/a.so", "same"), ("config/stage_00.json", "{}")):
        (served / rel).parent.mkdir(parents=True, exist_ok=True)
        (served / rel).write_text(text)
    served_sha = manifest(served)
    (home / "release/core").mkdir(parents=True)
    (home / "release/core/WEIGHTSD_BIN").write_text("wd1\n")
    w = home / "release-assemble-new7"
    (w / "root").mkdir(parents=True)
    subprocess.run(["cp", "-Rp", f"{served}/.", str(w / "root")], check=True)
    (w / "root/bin/engine").write_text("new engine")
    new_sha = manifest(w / "root")
    (w / "MANIFEST.served").write_text((served / "MANIFEST").read_text())
    env = {"S7": "new7", "PREV7": "old7", "RELEASE_ROOT_NAME": "rootx", "NEW_WEIGHTD": "wd1", "EXPECT_CHANGED": "1", "NEW_MANIFEST_SHA": new_sha, "SERVED_MANIFEST_SHA": served_sha}
    r = hub_run(home, "publish_root.sh", dict(env, EXPECT_CHANGED="2"))
    check("hub publish: a wrong changed-count refuses before any write", r.returncode == 1 and "changed-count-unexpected 1 want 2" in r.stdout and sha(served / "MANIFEST") == served_sha, r.stdout + r.stderr)
    r = hub_run(home, "publish_root.sh", dict(env, NEW_WEIGHTD="wd2"))
    check("hub publish: an unannounced weightd refuses", r.returncode == 1 and "weightd-bin-unexpected" in r.stdout, r.stdout)
    r = hub_run(home, "publish_root.sh", env)
    check("hub publish: files then MANIFEST, rollback copy recorded", r.returncode == 0 and "PUBLISHED" in r.stdout and sha(served / "MANIFEST") == new_sha and (served / "bin/engine").read_text() == "new engine" and (w / "ROLLBACK_DIR").exists(), r.stdout + r.stderr)
    r = hub_run(home, "publish_root.sh", env)
    check("hub publish: a second publish refuses (served changed)", r.returncode == 1 and "served-changed" in r.stdout, r.stdout)
    r = hub_run(home, "rollback_root.sh", env)
    check("hub rollback: back to the served MANIFEST and files", r.returncode == 0 and "ROOT-ROLLED-BACK" in r.stdout and sha(served / "MANIFEST") == served_sha and (served / "bin/engine").read_text() == "old engine", r.stdout + r.stderr)
    r = hub_run(home, "rollback_root.sh", env)
    check("hub rollback: idempotent", r.returncode == 0 and "already production" in r.stdout, r.stdout)
    (served / "bin/engine").write_text("new engine")
    r = hub_run(home, "rollback_root.sh", env)
    check("hub rollback: a partial publish (files, old MANIFEST) is restored", r.returncode == 0 and "partial publish" in r.stdout and (served / "bin/engine").read_text() == "old engine", r.stdout + r.stderr)
    (served / "MANIFEST").write_text("foreign\n")
    r = hub_run(home, "rollback_root.sh", env)
    check("hub rollback: a foreign MANIFEST stops", r.returncode == 1 and "neither the release nor production" in r.stdout, r.stdout)


def write_env(root, extra=""):
    env = root / "release.env"
    env.write_text(f""". {PROFILE}/profile.env
RELEASE_NODES="na nb"
RELEASE_HUB=hub
LOGS={root}/logs
SHA=abc
S7=abc
PREV7=old
SERVED_MANIFEST_SHA=served
NEW_MANIFEST_SHA=new
EXPECT_CHANGED=1
OLD_RESIDENTD=r0
NEW_RESIDENTD=r1
OLD_DRIVER=d0
NEW_DRIVER=d1
OLD_WEIGHTD=w0
NEW_WEIGHTD=w0
OLD_API=a0
NEW_API=a1
OLD_ADAPTER=p0
NEW_ADAPTER=p1
CHANNEL_DEPLOYMENT=c0
NEW_CHANNEL_DEPLOYMENT=c1
TOKENIZER_SHA=t0
CHANNEL_ADD=none
STAGE_CONFIG_ADD=none
{extra}""")
    return env


def test_hold_waits_for_a_paused_rotation(tmp):
    root, ssh = fake_world(tmp, ["na", "nb", "hub"])
    rootname = "glm53flash.fp8.tp16"
    for h in ("na", "nb"):
        (root / h / "sparkdata" / rootname).mkdir(parents=True)
        (root / f"{h}.probe").write_text(f"host={h} agent=active hold=no layout=legacy applied=served rootok=yes eng_n=1 exe=r0 drv=d0 wd=w0 wd_n=1 mem_gib=60 others=none\n")
    env = dict(os.environ, FAKE_ROOT=str(root), RELEASE_SSH=ssh, RELEASE_ENV=str(write_env(root)))
    r = subprocess.run(["bash", str(KIT / "hold.sh")], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    calls = (root / "calls.log").read_text() if (root / "calls.log").exists() else ""
    check("hold: refuses while the rotation runs, before touching a node", r.returncode == 1 and "rotation is not paused" in r.stdout and "agent.hold" not in calls, r.stdout)
    (root / "hub" / "fleet-rotation").mkdir()
    (root / "hub" / "fleet-rotation" / "ROTATION_PAUSE").write_text("manual: release\n")
    (root / "hub" / "unit-state").write_text("activating\n")
    r = subprocess.run(["bash", str(KIT / "hold.sh")], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    check("hold: refuses while a rotation tick still runs", r.returncode == 1 and "a tick is still running" in r.stdout and not (root / "na" / "sparkdata" / rootname / "agent.hold").exists(), r.stdout)
    (root / "hub" / "unit-state").write_text("inactive\n")
    r = subprocess.run(["bash", str(KIT / "hold.sh")], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    check("hold: paused rotation, gate OK -> every node held", r.returncode == 0 and "HOLD DONE" in r.stdout and all((root / h / "sparkdata" / rootname / "agent.hold").exists() for h in ("na", "nb")), r.stdout + r.stderr)
    r = subprocess.run(["bash", str(KIT / "hold.sh")], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    check("hold: a second release hold refuses (already held)", r.returncode == 1 and "node gate (release) failed" in r.stdout, r.stdout)
    env["RELEASE_ENV"] = str(write_env(root).rename(root / "bad.env"))
    Path(env["RELEASE_ENV"]).write_text(Path(env["RELEASE_ENV"]).read_text().replace("SHA=abc", "SHA=PENDING"))
    r = subprocess.run(["bash", str(KIT / "publish.sh")], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    check("lib: a PENDING value stops every step", r.returncode == 2 and "SHA is still PENDING" in r.stdout, r.stdout)


def test_step_gates_refuse_a_running_rotation(tmp):
    root, ssh = fake_world(tmp, ["na", "nb", "hub"])
    env = dict(os.environ, FAKE_ROOT=str(root), RELEASE_SSH=ssh, RELEASE_ENV=str(write_env(root, "BUILD_HOST=na\nBUILD_DIR=/nonexistent\nSOURCE_REPO=/nonexistent\n")))
    for step in ("publish.sh", "precheck.sh"):
        if (root / "calls.log").exists():
            (root / "calls.log").unlink()
        r = subprocess.run(["bash", str(KIT / step)], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
        calls = (root / "calls.log").read_text() if (root / "calls.log").exists() else ""
        touched = [c for c in calls.splitlines() if not c.startswith("hub\t") or "ROTATION_PAUSE" not in c] if step == "publish.sh" else []
        check(f"{step}: refuses while the rotation runs", r.returncode != 0 and "FAIL rotation is not paused" in r.stdout and not touched, r.stdout[-400:] + calls)


def weightd_world(tmp):
    root, ssh = fake_world(tmp, ["na", "nb", "hub"])
    rootname = "glm53flash.fp8.tp16"
    hub = root / "hub"
    served = hub / "release" / rootname
    (served / "bin").mkdir(parents=True)
    (served / "bin" / "engine").write_text("engine")
    served_sha = manifest(served)
    core = hub / "release" / "core" / "bin"
    core.mkdir(parents=True)
    (core / "sparkpipe_weightd").write_text("new weightd")
    old_wd = hashlib.sha256(b"old weightd").hexdigest()[:16]
    new_wd = sha(core / "sparkpipe_weightd")[:16]
    (hub / "release" / "core" / "WEIGHTSD_BIN").write_text(old_wd + "\n")
    for rel in ("bin/sparkpipe_model_api", "runtime/lib/model_serving_adapter.so", "model_resident.json"):
        (hub / "g53-api-channel" / rel).parent.mkdir(parents=True, exist_ok=True)
        (hub / "g53-api-channel" / rel).write_text(rel)
    rb = hub / "release-staging" / "weightd-rollback"
    rb.mkdir(parents=True)
    (rb / f"sparkpipe_weightd.{old_wd}").write_text("old weightd")
    for h in ("na", "nb"):
        (root / h / "sparkdata" / rootname).mkdir(parents=True)
        (root / h / "sparkdata" / rootname / "agent.hold").write_text("")
        (root / f"{h}.probe").write_text(f"host={h} agent=active hold=no layout=legacy applied={served_sha[:16]} rootok=yes eng_n=1 exe=r0 drv=d0 ready=1 wd={old_wd} wd_inst={old_wd} wd_n=1 wd_other=0 mem_gib=60 others=none\n")
    (hub / "fleet-rotation").mkdir()
    (hub / "fleet-rotation" / "ROTATION_PAUSE").write_text("manual: release\n")
    extra = f"BUILD_HOST=na\nWEIGHTD_BUNDLE_BUILD=/b\nNEW_RECEIPT=rc\nNEW_WARM=wm\n"
    envfile = write_env(root, extra)
    text = envfile.read_text().replace("SERVED_MANIFEST_SHA=served", f"SERVED_MANIFEST_SHA={served_sha}").replace("OLD_WEIGHTD=w0", f"OLD_WEIGHTD={old_wd}").replace("NEW_WEIGHTD=w0", f"NEW_WEIGHTD={new_wd}")
    envfile.write_text(text + "RELEASE_CONVERGE_MIN=0.1\n")
    env = dict(os.environ, FAKE_ROOT=str(root), RELEASE_SSH=ssh, RELEASE_ENV=str(envfile))
    return root, env, core / "sparkpipe_weightd", old_wd


def test_rollback_restores_a_published_unannounced_weightd(tmp):
    root, env, core_bin, old_wd = weightd_world(tmp)
    r = subprocess.run(["bash", str(KIT / "rollback.sh"), "engines"], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
    check("rollback: a published but unannounced weightd goes back on the hub", r.returncode == 0 and sha(core_bin)[:16] == old_wd and "ROLLBACK engines DONE" in r.stdout, r.stdout + r.stderr)


def test_scripts_parse_and_are_model_neutral():
    shells = sorted(KIT.glob("*.sh")) + sorted((KIT / "hub").glob("*.sh"))
    r = subprocess.run(["bash", "-n"] + [str(p) for p in shells], capture_output=True, text=True)
    check("kit: every shell script parses", r.returncode == 0, r.stderr)
    sys.path.insert(0, str(REPOSITORY / "tests"))
    import test_dry_law
    named = []
    for p in sorted(KIT.rglob("*")):
        if p.is_file() and p.suffix != ".pyc":
            text = p.read_text()
            if test_dry_law.MODEL_TOKEN.search(text) or test_dry_law.MODEL_TOKEN.search(p.name):
                named.append(p.name)
    check("kit: no model or driver name in the shared release kit", not named, named)
    exe = [p.name for p in KIT.glob("*") if p.suffix in (".sh", ".py") and p.name not in ("lib.sh", "node_probe.sh") and not os.access(p, os.X_OK)]
    check("kit: entry points carry the exec bit", not exe, exe)


def main():
    test_nodes_rules_and_resample()
    test_hub_smoke()
    test_scripts_parse_and_are_model_neutral()
    for test in (test_nodes_cli_sends_rank_pack, test_engine_logs, test_perf_summary, test_stage_config_patch, test_hub_publish_and_rollback, test_hold_waits_for_a_paused_rotation, test_step_gates_refuse_a_running_rotation, test_rollback_restores_a_published_unannounced_weightd):
        with tempfile.TemporaryDirectory() as tmp:
            test(tmp)
    if failures:
        print(f"\n{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
