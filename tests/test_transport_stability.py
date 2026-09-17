#!/usr/bin/env python3
"""Fleet stability test: kill engines/weightd during serving, verify recovery.

Runs the qualification fixture in a loop while randomly killing processes
on random nodes. Asserts time-to-serve and never-wedge.

Usage: python3 test_transport_stability.py <duration_minutes> [seed]
Requires: the fleet serving (16/16 ready, API on rtx5090:8433).
"""

import http.client
import json
import os
import random
import subprocess
import sys
import time

FLEET_NODES = [
    "spark0", "spark1", "spark2", "spark3", "spark4", "spark5",
    "spark6", "spark7", "spark8", "spark9", "sparka", "sparkb",
    "sparkc", "sparkd", "sparke", "sparkf",
]
API_HOST = "rtx5090"
API_PORT = 8433
FIXTURE_TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]
EXPECTED_FIRST_TOKENS = [3764, 10]
MAX_RECOVERY_SECONDS = 120
PROBE_TIMEOUT_SECONDS = 300

FAILURES = []


def ssh(node, command, timeout=10):
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", node, command],
            capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        return 1, ""


def kill_random_process(node, rng):
    victim = rng.choice(["engine", "weightd", "both"])
    if victim in ("engine", "both"):
        ssh(node, 'for Q in $(pgrep -f "residentd --deployment"); do '
                  '[ "$(readlink /proc/$Q/cwd 2>/dev/null)" = '
                  '"$HOME/sparkdata/glm53flash.fp8.tp16" ] && kill -9 $Q; done')
    if victim in ("weightd", "both"):
        ssh(node, 'P=$(pgrep -f "sparkdata/weightd" | head -1); '
                  '[ -n "$P" ] && kill -9 $P')
    return victim


def probe(api_served):
    conn = http.client.HTTPConnection(API_HOST, API_PORT,
                                      timeout=PROBE_TIMEOUT_SECONDS)
    body = json.dumps({
        "model": "sparkpipe-model",
        "prompt_token_ids": FIXTURE_TOKENS,
        "max_tokens": 4,
        "temperature": 0,
    })
    t0 = time.monotonic()
    try:
        conn.request("POST", "/v1/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read().decode())
        tokens = data.get("tokens", [])
        elapsed = time.monotonic() - t0
        conn.close()

        if len(tokens) == 0:
            return elapsed, False, "zero tokens"
        if tokens[0] != EXPECTED_FIRST_TOKENS[0]:
            return elapsed, False, f"wrong token {tokens[0]}"
        return elapsed, True, f"ok ntok={len(tokens)}"
    except Exception as e:
        return time.monotonic() - t0, False, str(e)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def check_fleet_ready():
    rc, out = ssh("sparkf",
                  'grep -l \'"state":"ready\' current/*.json 2>/dev/null | wc -l')
    try:
        return int(out.strip()) >= 15
    except ValueError:
        return False


def restart_api():
    ssh(API_HOST, "systemctl --user restart g53-api")
    time.sleep(12)


def main():
    duration_min = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rng = random.Random(seed)
    duration_sec = duration_min * 60
    start = time.monotonic()

    print(f"=== stability fuzz: {duration_min}min seed={seed} ===")

    if not check_fleet_ready():
        print("FATAL: fleet not ready at start")
        sys.exit(2)

    restart_api()

    probe_count = 0
    kill_count = 0
    recovery_times = []

    while time.monotonic() - start < duration_sec:
        elapsed, ok, detail = probe(probe_count)
        probe_count += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{time.monotonic()-start:7.1f}s] probe {probe_count}: "
              f"{status} t={elapsed:.1f}s {detail}")

        if not ok:
            FAILURES.append(f"probe {probe_count}: {detail}")

        if elapsed > 60 and ok:
            FAILURES.append(f"probe {probe_count}: slow ({elapsed:.1f}s)")

        victim_node = rng.choice(FLEET_NODES)
        victim = kill_random_process(victim_node, rng)
        kill_count += 1
        print(f"[{time.monotonic()-start:7.1f}s] killed {victim} on {victim_node}")

        t_kill = time.monotonic()
        deadline = t_kill + MAX_RECOVERY_SECONDS
        recovered = False
        while time.monotonic() < deadline:
            if check_fleet_ready():
                recovered = True
                break
            time.sleep(5)

        recovery = time.monotonic() - t_kill
        if not recovered:
            FAILURES.append(
                f"kill {kill_count}: {victim} on {victim_node} "
                f"did not recover in {MAX_RECOVERY_SECONDS}s")
            print(f"[{time.monotonic()-start:7.1f}s] RECOVERY TIMEOUT")
        else:
            recovery_times.append(recovery)
            print(f"[{time.monotonic()-start:7.1f}s] recovered in {recovery:.1f}s")

        restart_api()

    print(f"\n=== summary ===")
    print(f"probes: {probe_count}, kills: {kill_count}")
    print(f"recovery times: min={min(recovery_times):.1f}s "
          f"max={max(recovery_times):.1f}s "
          f"avg={sum(recovery_times)/len(recovery_times):.1f}s"
          if recovery_times else "no successful recoveries")
    print(f"failures: {len(FAILURES)}")
    for f in FAILURES:
        print(f"  {f}")

    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
