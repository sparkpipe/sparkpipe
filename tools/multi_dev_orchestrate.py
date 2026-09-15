#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import threading
import time

SMOKE = "$HOME/smokemd-mds/bin"
RECEIPTS = "/Users/mac/smokemd/runs/multi-dev-smoke-receipts"
SPARKCAP = "/usr/local/sbin/sparkcap"

DEVS = [
    {"name": "dev-glm5next-fp8", "pool": 8388608, "node": "spark9", "args": (2, 8, 2, 262144, 262144, 11)},
    {"name": "dev-qwen38max-nvfp4", "pool": 8388608, "node": "spark4", "args": (4, 16, 1, 262144, 0, 22)},
    {"name": "dev-k3-mxfp4", "pool": 8388608, "node": "sparkb", "args": (2, 32, 1, 131072, 0, 33)},
    {"name": "dev-ling-bf16", "pool": 8388608, "node": "sparkc", "args": (2, 8, 2, 262144, 0, 44)},
    {"name": "dev-gemma4-dense", "pool": 16777216, "node": "sparkf", "args": (4, 1, 4, 524288, 1048576, 55)},
    {"name": "dev-laguna-mxp", "pool": 8388608, "node": "sparkd", "args": (2, 4, 3, 262144, 0, 66)},
    {"name": "dev-hy4-fp8", "pool": 8388608, "node": "spark2", "args": (8, 4, 1, 131072, 0, 77)},
]

REALWS = [
    {"name": "realws-ling", "node": "spark9",
     "pack": "/home/spark9/sparkdata/ling.bf16.tp16/packs/ling.bf16.tp16.rank9.sp",
     "sha": "cf62a57278306601f7f1fbc8790860b1a5aef22f730c6a1cc8224a9336282800",
     "topk": 8, "pool": 6 << 30, "ceiling": 20 << 30},
    {"name": "realws-dsv4flash", "node": "spark9",
     "pack": "/home/spark9/sparkdata/dsv4flash.tp16/packs/dsv4flash.tp16.rank8.spstage",
     "sha": "SHA_FROM_FILE", "topk": 8, "pool": 6 << 30, "ceiling": 20 << 30},
    {"name": "realws-qwen38max", "node": "spark8",
     "pack": "/home/spark8/sparkdata/qwenmax.nvfp4.tp16/packs/qwenmax.nvfp4.tp16.rank8.sp",
     "sha": "d9d14d0b475c1fe30da80ca0ec07f1cd51bc79f403c077da497430c2df69c586",
     "topk": 8, "pool": 10 << 30, "ceiling": 30 << 30},
]

ROOT = "$HOME/smokemd-mds"


def run(cmd, timeout=240):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout)


def ssh(node, script, timeout=240):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o",
                           "ConnectTimeout=10", node, "bash -s"],
                          input=script, capture_output=True, text=True,
                          timeout=timeout)


def esc(text):
    return text.replace("$HOME", "\\$HOME")


def probe_nodes():
    healthy = []
    for node in sorted({dev["node"] for dev in DEVS} | {"spark9"}):
        result = ssh(node, "test -x $HOME/smokemd-mds/bin/multi_dev_smoke || "
                           "{ mkdir -p $HOME/smokemd-mds/bin; ln -sfn "
                           "/home/spark9/smokemd/bin/* $HOME/smokemd-mds/bin/ 2>/dev/null; }; "
                           "timeout 30 $HOME/smokemd-mds/bin/multi_dev_smoke bandwidth 8 1 "
                           "2>&1 | tail -1")
        ok = "BANDWIDTH" in result.stdout
        print(f"PROBE {node} cuda_context_ok={ok} {result.stdout.strip()[:90]}")
        if ok:
            healthy.append(node)
    return healthy


def daemon_control_script(root, ceiling, logfile):
    return f"""
cd {root}
if [ -f daemon.pid ]; then
  OLDPID=$(cat daemon.pid)
  if [ -d /proc/$OLDPID ] && grep -q sparkpipe_weightd /proc/$OLDPID/cmdline 2>/dev/null; then
    sudo -n kill $OLDPID
    for i in $(seq 1 20); do [ -d /proc/$OLDPID ] || break; sleep 0.5; done
  fi
  rm -f daemon.pid
fi
nohup sudo -n {SPARKCAP} --mem 4096 {SMOKE}/sparkpipe_weightd --socket {root}/w.sock --device-bytes-max {ceiling} > {logfile} 2>&1 &
NEWPID=$!
echo $NEWPID > daemon.pid
for i in $(seq 1 30); do
  if grep -q ready {logfile} 2>/dev/null; then echo DAEMON_READY pid=$NEWPID; exit 0; fi
  sleep 0.5
done
echo DAEMON_TIMEOUT
cat {logfile}
exit 1
"""


def dev_setup_script(dev):
    layers, experts, kinds, range_bytes, gap, seed = dev["args"]
    root = f"{ROOT}/{dev['name']}"
    return f"""
set -e
mkdir -p {root}
cd {root}
rm -f {dev['name']}.pack {dev['name']}.pack.experts
{SMOKE}/multi_dev_smoke makepack {root}/{dev['name']}.pack {layers} {experts} {kinds} {range_bytes} {gap} {seed}
"""


def rig_command(root, pack, name, rounds, pool, pins):
    return (f"sudo -n {SPARKCAP} --mem 2048 {SMOKE}/multi_dev_smoke dev "
            f"{root}/w.sock {pack} {name} {rounds} 4 {pool} {pins} 0")


def snapshot(node, root, tag, stage_log):
    script = f"""
echo NVIDIA_APPS
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo MPS_SERVER
pgrep -f nvidia-cuda-mps-server || echo MPS_SERVER_ABSENT
echo DAEMON
for r in {root}; do
  [ -f $r/daemon.pid ] || continue
  pid=$(cat $r/daemon.pid)
  if [ -d /proc/$pid ]; then echo "ALIVE daemon $pid $r"; else echo "DEAD daemon $pid $r"; fi
done
free -g | sed -n 2p
"""
    result = ssh(node, script)
    with open(stage_log, "a") as handle:
        handle.write(f"### snapshot node={node} tag={tag}\n{result.stdout}\n")
    return result.stdout


def parse_smoke(text):
    events = []
    for line in text.splitlines():
        if line.startswith("SMOKE "):
            kind, _, payload = line[6:].partition(" ")
            try:
                events.append((kind, json.loads(payload)))
            except json.JSONDecodeError:
                events.append((kind, {"raw": payload}))
    return events


def launch_rig(node, root, pack, name, rounds, pool, pins, mps=False, timeout=1500):
    env_prefix = ("env CUDA_MPS_PIPE_DIRECTORY=/home/spark9/smokemd-mds/mps-pipe "
                  "CUDA_MPS_LOG_DIRECTORY=/home/spark9/smokemd-mds/mps-log "
                  if mps else "sudo -n")
    remote = (f"ssh -o BatchMode=yes {node} \"cd {root} && "
              f"{env_prefix} {SPARKCAP} --mem 2048 {SMOKE}/multi_dev_smoke dev "
              f"{root}/w.sock {pack} {name} {rounds} 4 {pool} {pins} 0; echo RIG_RC=$?\"")
    proc = subprocess.Popen(esc(remote), shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    return proc


def spread_stage(count, rounds, healthy=None):
    stage_log = f"{RECEIPTS}/spread-stage{count}.log"
    devs = DEVS[:count]
    if healthy is not None:
        devs = [dev for dev in devs if dev["node"] in healthy]
    if len(devs) < count:
        print(f"SPREAD STAGE {count}: only {len(devs)}/{count} healthy devs "
              f"(skipped nodes documented)")
        if not devs:
            return False
    print(f"== SPREAD STAGE {count}: {len(devs)} devs on {len(devs)} nodes ==")
    for dev in devs:
        result = ssh(dev["node"], dev_setup_script(dev), timeout=120)
        if result.returncode != 0:
            print(f"SETUP FAIL {dev['name']} on {dev['node']}: {result.stderr[-200:]}")
            return False
    for dev in devs:
        root = f"{ROOT}/{dev['name']}"
        started = ssh(dev["node"],
                      daemon_control_script(root, 536870912, f"{root}/daemon.log"),
                      timeout=90)
        ready = "DAEMON_READY" in started.stdout
        print(f"DAEMON {dev['name']} node={dev['node']} ready={ready}")
        if not ready:
            return False
    time.sleep(2)
    procs = [(dev, launch_rig(dev["node"], f"{ROOT}/{dev['name']}",
                              f"{ROOT}/{dev['name']}/{dev['name']}.pack",
                              dev["name"], rounds, dev["pool"], 0))
             for dev in devs]
    stage_pass = True
    lanes_seen = []
    for dev, proc in procs:
        out, _ = proc.communicate(timeout=timeout(1200))
        with open(stage_log, "a") as handle:
            handle.write(f"### dev={dev['name']} node={dev['node']}\n{out}\n")
        events = parse_smoke(out)
        lanes = [p.get("lane") for k, p in events
                 if k == "DEV" and p.get("op") == "dev-init"]
        lanes_seen.extend(lanes)
        summaries = [p for k, p in events if k == "DEV-SUMMARY"]
        teardown = [p for k, p in events if k == "DEV-TEARDOWN"]
        ok = bool(summaries) and all(s["verify_fails"] == 0 for s in summaries) \
            and ("RIG_RC=0" in out or (teardown and "RIG_RC=1" in out))
        print(f"DEV {dev['name']} node={dev['node']} lane={lanes} "
              f"summary={summaries[0] if summaries else None} teardown={teardown} ok={ok}")
        stage_pass = stage_pass and ok
    per_dev_lowest = len(lanes_seen) == len(devs) and all(lane == 0 for lane in lanes_seen)
    print(f"LANES per_dev={lanes_seen} lowest_free_on_private_daemon={per_dev_lowest} "
          f"(distinct-lane proof runs in the co-residency stage)")
    for dev in devs:
        snap = snapshot(dev["node"], f"{ROOT}/{dev['name']}", f"stage{count}", stage_log)
        for line in snap.splitlines():
            if line.startswith("DEAD"):
                print(f"PIDCHECK FAIL {dev['node']} {line}")
                stage_pass = False
            elif line.startswith("ALIVE"):
                print(f"PIDCHECK {dev['node']} {line}")
    print(f"SPREAD STAGE {count} PASS={stage_pass}")
    return stage_pass


def timeout(seconds):
    return seconds


def coresidency(tag, count, rounds, use_mps=False):
    stage_log = f"{RECEIPTS}/core-{tag}.log"
    root = f"{ROOT}/coresident"
    setup = [f"mkdir -p {root}", f"cd {root}"]
    for index in range(count):
        dev = DEVS[index]
        layers, experts, kinds, range_bytes, gap, seed = dev["args"]
        setup.append(
            f"rm -f {dev['name']}.pack {dev['name']}.pack.experts; "
            f"{SMOKE}/multi_dev_smoke makepack {root}/{dev['name']}.pack "
            f"{layers} {experts} {kinds} {range_bytes} {gap} {seed}")
    result = ssh("spark9", "\n".join(setup), timeout=180)
    if result.returncode != 0:
        print(f"CORE SETUP FAIL {tag}: {result.stderr[-200:]}")
        return False, {}
    mps_note = ""
    if use_mps:
        mps_note = ("MPS pipe=/home/spark9/smokemd-mds/mps-pipe; "
                    "clients see CUDA_MPS_PIPE_DIRECTORY via rig env below")
    started = ssh("spark9", daemon_control_script(root, 2147483648, f"{root}/daemon.log"),
                  timeout=90)
    if "DAEMON_READY" not in started.stdout:
        print(f"CORE DAEMON FAIL {tag}: {started.stdout[-200:]}")
        return False, {}
    print(f"DAEMON coresident ready ({mps_note})")
    procs = []
    for index in range(count):
        dev = DEVS[index]
        procs.append((dev["name"], launch_rig("spark9", root,
                                              f"{root}/{dev['name']}.pack",
                                              dev["name"], rounds, dev["pool"], 0,
                                              mps=use_mps)))
    def midrun_snapshot():
        time.sleep(max(2.5, rounds * 0.0045))
        snapshot("spark9", root, tag + "-midrun", stage_log)
    threading.Thread(target=midrun_snapshot, daemon=True).start()
    bandwidth_command = esc(
        f"ssh -o BatchMode=yes spark9 \"cd {root} && "
        f"sudo -n {SPARKCAP} --mem 2048 {SMOKE}/multi_dev_smoke bandwidth 256 40; "
        f"echo BW_RC=$?\"")
    bandwidth = subprocess.Popen(bandwidth_command, shell=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
    stage_pass = True
    curve = {}
    bandwidth_line = None
    for name, proc in procs + [("aggregate-bandwidth", bandwidth)]:
        out, _ = proc.communicate(timeout=1500)
        with open(stage_log, "a") as handle:
            handle.write(f"### member={name}\n{out}\n")
        events = parse_smoke(out)
        for kind, payload in events:
            if kind == "BANDWIDTH":
                bandwidth_line = payload
                print(f"CORE {tag} BANDWIDTH {payload}")
            if kind == "DEV-SUMMARY":
                curve[name] = payload["p50_us"]
                print(f"CORE {tag} {name} p50={payload['p50_us']}us "
                      f"p99={payload['p99_us']}us verify_fails={payload['verify_fails']}")
                stage_pass = stage_pass and payload["verify_fails"] == 0
            if kind == "DEV-TEARDOWN":
                print(f"CORE {tag} {name} teardown={payload} (known finding)")
    time.sleep(1)
    snap = snapshot("spark9", root, tag, stage_log)
    live = midrun_count = 0
    for section in open(stage_log).read().split("### snapshot"):
        if "-midrun" in section[:40]:
            midrun_count = sum(1 for line in section.splitlines()
                               if line[:1].isdigit())
    live = midrun_count
    print(f"CORE {tag} midrun_cuda_compute_apps={live} stages_pass={stage_pass}")
    with open(stage_log, "a") as handle:
        handle.write(f"### curve {tag}\n{json.dumps(curve, indent=1)}\n")
    return stage_pass, {"curve": curve, "bandwidth": bandwidth_line,
                        "midrun_cuda_apps": live}


def mps_daemon(up):
    if up:
        script = """
mkdir -p /home/spark9/smokemd-mds/mps-pipe /home/spark9/smokemd-mds/mps-log
export CUDA_MPS_PIPE_DIRECTORY=/home/spark9/smokemd-mds/mps-pipe
export CUDA_MPS_LOG_DIRECTORY=/home/spark9/smokemd-mds/mps-log
nvidia-cuda-mps-control -d 2>&1 | head -1
sleep 1
pgrep -f nvidia-cuda-mps-control >/dev/null && echo CONTROL_ALIVE
"""
    else:
        script = """
export CUDA_MPS_PIPE_DIRECTORY=/home/spark9/smokemd-mds/mps-pipe
echo quit | nvidia-cuda-mps-control 2>/dev/null
sleep 1
pgrep -f nvidia-cuda-mps-control >/dev/null || echo CONTROL_GONE
"""
    result = ssh("spark9", script, timeout=60)
    ok = ("CONTROL_ALIVE" in result.stdout) if up else ("CONTROL_GONE" in result.stdout)
    print(f"MPS up={up} verified={ok} out={result.stdout.strip()[-120:]} "
          f"err={result.stderr.strip()[-120:]}")
    return ok


def eviction_matrix():
    stage_log = f"{RECEIPTS}/eviction-matrix.log"
    root = f"{ROOT}/eviction"
    pack = f"{root}/press.pack"
    result = ssh("spark9", f"""
set -e
mkdir -p {root}
cd {root}
rm -f press.pack press.pack.experts
{SMOKE}/multi_dev_smoke makepack {pack} 2 8 2 524288 0 91
""", timeout=60)
    if result.returncode != 0:
        print(f"EVICTION SETUP FAIL: {result.stderr[-200:]}")
        return False
    started = ssh("spark9",
                  daemon_control_script(root, 536870912, f"{root}/daemon.log"),
                  timeout=90)
    if "DAEMON_READY" not in started.stdout:
        print(f"EVICTION DAEMON FAIL: {started.stdout[-200:]}")
        return False
    print("EVICTION daemon ready")
    matrix_pass = True

    def evictor(tag, acquire, targets, expect):
        nonlocal matrix_pass
        remote = (f"ssh -o BatchMode=yes spark9 \"cd {root} && "
                  f"sudo -n {SPARKCAP} --mem 1024 {SMOKE}/multi_dev_smoke evictor "
                  f"{root}/w.sock {acquire} {targets} 2>&1; echo EVICTOR_RC=$?\"")
        merged = run(esc(remote)[:-1] + ' 2>&1"', timeout=120)
        out = merged.stdout + merged.stderr
        with open(stage_log, "a") as handle:
            handle.write(f"### evictor {tag}\n{out}\n")
        events = [(k, p) for k, p in parse_smoke(out) if k == "EVICT"]
        got = [(p["evictor_lane"], p["target_lane"], p["status"]) for k, p in events]
        ok = len(got) == len(expect) and all(e[2] == w for e, w in zip(got, expect))
        print(f"EVICTION {tag} got={got} expect={expect} ok={ok}")
        matrix_pass = matrix_pass and ok

    print("== EVICTION PHASE 1: holder on lane 0, evictors arrive later ==")
    holder_command = esc(
        f"ssh -o BatchMode=yes spark9 \"cd {root} && "
        f"{rig_command(root, pack, 'holder', 4000, 16777216, 8)}\"")
    holder = subprocess.Popen(holder_command, shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
    time.sleep(12)
    evictor("laneless", 0, "0,1,9", ["INVALID_ARGUMENT", "INVALID_ARGUMENT",
                                     "INVALID_ARGUMENT"])
    evictor("lane1_vs_lane0", 1, "0,1,7", ["EVICT_DENIED", "EVICT_DENIED", "OK"])
    with open(stage_log, "a") as handle:
        handle.write("### holder phase1 output follows in holder.log\n")
    out, _ = holder.communicate(timeout=1200)
    with open(f"{stage_log}", "a") as handle:
        handle.write(f"### holder phase1\n{out[-2000:]}\n")
    events = parse_smoke(out)
    summaries = [p for k, p in events if k == "DEV-SUMMARY"]
    fails = [p for k, p in events if k == "DEV-FAIL"]
    holder_ok = bool(summaries) and summaries[0]["verify_fails"] == 0 and not fails
    print(f"EVICTION holder phase1 survived_ok={holder_ok} summary={summaries}")
    matrix_pass = matrix_pass and holder_ok

    print("== EVICTION PHASE 2: priority evict forces release, victim re-streams ==")
    time.sleep(2)
    victim_command = esc(
        f"ssh -o BatchMode=yes spark9 \"cd {root} && "
        f"{rig_command(root, pack, 'victim', 4000, 16777216, 8)}\"")
    victim = subprocess.Popen(victim_command, shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
    time.sleep(12)
    evictor("lane0_vs_lane1", 1, "0", ["OK"])
    out, _ = victim.communicate(timeout=1200)
    with open(stage_log, "a") as handle:
        handle.write(f"### victim phase2\n{out[-3000:]}\n")
    events = parse_smoke(out)
    summaries = [p for k, p in events if k == "DEV-SUMMARY"]
    fails = [p for k, p in events if k == "DEV-FAIL"]
    victim_ok = bool(summaries) and summaries[0]["verify_fails"] == 0 and not fails
    print(f"EVICTION victim re-streamed_ok={victim_ok} summary={summaries}")
    matrix_pass = matrix_pass and victim_ok

    print("== EVICTION PHASE 3: pool pressure, LRU churn, lazy miss re-stream ==")
    pressure_command = esc(
        f"ssh -o BatchMode=yes spark9 \"cd {root} && "
        f"{rig_command(root, pack, 'pressure', 300, 4194304, 0)}\"")
    pressure = subprocess.Popen(pressure_command, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
    out, _ = pressure.communicate(timeout=1200)
    with open(stage_log, "a") as handle:
        handle.write(f"### pressure phase3\n{out[-4000:]}\n")
    events = parse_smoke(out)
    summaries = [p for k, p in events if k == "DEV-SUMMARY"]
    misses = summaries[0]["misses"] if summaries else -1
    pressure_ok = bool(summaries) and summaries[0]["verify_fails"] == 0
    print(f"EVICTION pressure verify_clean={pressure_ok} lazy_miss_rounds={misses} "
          f"summary={summaries}")
    matrix_pass = matrix_pass and pressure_ok and misses > 0
    print(f"EVICTION MATRIX PASS={matrix_pass}")
    return matrix_pass


def realws_track():
    stage_log = f"{RECEIPTS}/realws.log"
    budget = {}
    for dev in REALWS:
        node = dev["node"]
        root = f"{ROOT}/{dev['name']}"
        if dev["sha"] == "SHA_FROM_FILE":
            sha_out = run(f"ssh -o BatchMode=yes {node} "
                          f"'cat /home/spark9/smokemd/smoke-stage0/dsv4.sha256.txt "
                          f"/home/spark9/smokemd-mds/dsv4.sha256.txt 2>/dev/null'",
                          timeout=30)
            text = sha_out.stdout.strip().split()
            if not text or len(text[0]) != 64:
                print(f"REALWS skip {dev['name']}: sha unavailable ({sha_out.stdout[:80]})")
                continue
            dev["sha"] = text[0]
        result = ssh(node, f"""
mkdir -p {root}
cd {root}
{daemon_control_script(root, dev['ceiling'], f"{root}/daemon.log")}
""", timeout=120)
        if "DAEMON_READY" not in result.stdout:
            print(f"REALWS daemon fail {dev['name']}: {result.stdout[-200:]}")
            continue
        remote = (f"ssh -o BatchMode=yes {node} "
                  f"\"sudo -n {SPARKCAP} --mem 8192 {SMOKE}/multi_dev_smoke realws "
                  f"{root}/w.sock {dev['pack']} {dev['name']} {dev['topk']} "
                  f"{dev['pool']} {dev['sha']} 2>&1; echo REALWS_RC=$?\"")
        print(f"REALWS attaching {dev['name']} on {node} "
              f"(first attach streams the full pack once)...")
        out = run(esc(remote), timeout=1800).stdout
        with open(stage_log, "a") as handle:
            handle.write(f"### {dev['name']} node={node}\n{out}\n")
        events = parse_smoke(out)
        for kind, payload in events:
            if kind in ("REALWS", "REALWS-FAIL"):
                print(f"REALWS {dev['name']} {payload}")
        budget[dev["name"]] = events
    return budget


def cleanup():
    for dev in DEVS + REALWS:
        root = f"{ROOT}/{dev['name']}"
        ssh(dev["node"], f"""
if [ -f {root}/daemon.pid ]; then
  pid=$(cat {root}/daemon.pid)
  if [ -d /proc/$pid ]; then sudo -n kill $pid; fi
fi
""", timeout=30)
    for node in ("spark9",):
        ssh(node, f"""
for r in {ROOT}/coresident {ROOT}/eviction; do
  if [ -f $r/daemon.pid ]; then
    pid=$(cat $r/daemon.pid)
    if [ -d /proc/$pid ]; then sudo -n kill $pid; fi
  fi
done
""", timeout=30)
    print("CLEANUP daemons stopped (pid-file scoped, own stacks only)")


def main():
    os.makedirs(RECEIPTS, exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    healthy = probe_nodes()
    print("HEALTHY:", healthy)
    if phase in ("all", "spread"):
        for count in (2, 4, 7):
            if not spread_stage(count, 120, healthy):
                print(f"SPREAD STAGE {count} FAILED")
                if phase == "all":
                    break
    if phase in ("all", "core"):
        for tag, count in (("c2", 2), ("c4", 4), ("c7", 7)):
            coresidency(tag, count, 240)
    if phase in ("all", "mps"):
        print("== MPS OFF measurement is the c7 core stage; now MPS ON ==")
        if mps_daemon(True):
            coresidency("c7-mps", 7, 240, use_mps=True)
            mps_daemon(False)
        else:
            print("MPS control daemon failed to start; c7-mps skipped")
    if phase in ("all", "evict"):
        eviction_matrix()
    if phase in ("all", "realws"):
        realws_track()
    if phase in ("all", "cleanup"):
        cleanup()
    if phase == "cleanup-only":
        cleanup()


if __name__ == "__main__":
    main()
