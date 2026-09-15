#!/usr/bin/env python3
"""Mesh lane ladder receipt: prove multi-driver concurrency and lane-priority
eviction against a node-private weightd.

Spawns a private weightd (own socket, no mesh identity so the fleet mesh is
never touched), then runs the concurrency ladder: N concurrent driver
processes each acquire a lane (held connection), run TP-N allreduce rounds
through the shared region, and report per-op latency and checksums. Modes:
  lanes    : N held-lane drivers (the architecture contract)
  ephemeral: drivers that close the lane connection after acquire (pre-fix
             module behavior; expected to collide on lane 0)
  evict    : lane-priority eviction matrix (EVICT_DENIED below priority,
            ok above, leases really released)

usage: tools/mesh_lane_ladder_receipt.py --repo DIR --phase {baseline,after}
            [--ladder 1,2,4,8] [--iters 200] [--rows 128]
            [--keep-daemon] [--timeouts 240]
Fails on daemon death, daemon error lines, or any phase result that
contradicts --phase expectations. Prints a JSON receipt.
"""
import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

ERROR_PATTERN = ("ERRSITE", "IO_ERROR", "FAILED", "failure",
    "MESH-SPIN-TIMEOUT", "MESH-CANCEL-ABORT", "MESH-REGISTER-FAIL",
    "checksum-failed", "round-failed", "submit failed", "create failed")


def wait_socket(path, process, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"daemon exited rc={process.returncode} "
                "before socket")
        if path.exists():
            return
        time.sleep(0.05)
    raise RuntimeError("daemon socket never appeared")


def run_driver(args, socket_path, shm_path, mode, rank, degree):
    command = [str(args.repo / "build" / "mesh_lane_ladder"), str(socket_path),
        "mesh", str(rank), str(degree), str(args.iters), str(args.rows),
        mode, shm_path]
    environment = dict(os.environ)
    environment["SPARK_WEIGHTD_SOCKET"] = str(socket_path)
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True,
        timeout=args.timeouts, env=environment)
    return {
        "rank": rank,
        "degree": degree,
        "mode": mode,
        "rc": completed.returncode,
        "seconds": round(time.monotonic() - started, 3),
        "stdout": completed.stdout.decode(errors="replace"),
        "stderr": completed.stderr.decode(errors="replace")[-2000:],
    }


def parse_summaries(driver):
    summaries = []
    for line in driver["stdout"].splitlines():
        if line.startswith("SUMMARY "):
            fields = dict(token.split("=", 1) for token in line.split()[1:])
            summaries.append({
                "rank": int(fields["rank"]),
                "lane": int(fields["lane"]),
                "rounds": int(fields["rounds"]),
                "bad": int(fields["bad"]),
                "p50_us": float(fields["p50_us"]),
                "p99_us": float(fields["p99_us"]),
                "max_us": float(fields["max_us"]),
            })
        elif line.startswith("LADDER lane_acquire=FAILED"):
            driver["lane_acquire_failed"] = line.strip()
    return summaries


def run_lane_phase(args, socket_path, shm_path, degree, ephemeral):
    shm_path = f"{shm_path}.{degree}.{'eph' if ephemeral else 'held'}"
    for stale in [shm_path]:
        try:
            os.unlink(stale)
        except OSError:
            pass
    holders = []
    ranks = []
    for job in range(degree):
        mode = "ephemeral" if ephemeral else "hold"
        process = subprocess.Popen(
            [str(args.repo / "build" / "mesh_lane_ladder"),
             str(socket_path), mode],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, "SPARK_WEIGHTD_SOCKET": str(socket_path)})
        line = process.stdout.readline().decode(errors="replace").strip()
        lane = None
        if line.startswith("LADDER-INIT lane="):
            lane = int(line.split("lane=")[1].split()[0])
        else:
            process.kill()
        holders.append({"job": job, "process": process, "lane": lane,
            "init": line})
    for job_index, holder in enumerate(holders):
        if holder["lane"] is None:
            continue
        for rank in range(degree):
            ranks.append(subprocess.Popen(
                [str(args.repo / "build" / "mesh_lane_ladder"),
                 str(socket_path), "mesh", str(rank), str(degree),
                 str(args.iters), str(args.rows), str(holder["lane"]),
                 shm_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={**os.environ, "SPARK_WEIGHTD_SOCKET": str(socket_path)}))
    drivers = []
    for rank_index, rank_process in enumerate(ranks):
        stdout, stderr = rank_process.communicate(timeout=args.timeouts)
        drivers.append({
            "rank": rank_index,
            "degree": degree,
            "mode": "ephemeral" if ephemeral else "held",
            "rc": rank_process.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace")[-2000:],
        })
    for holder in holders:
        process = holder["process"]
        if process.poll() is None:
            process.terminate()
        try:
            rest_stdout, rest_stderr = process.communicate(
                timeout=args.timeouts)
        except subprocess.TimeoutExpired:
            process.kill()
            rest_stdout, rest_stderr = process.communicate()
        holder["rc"] = process.returncode
    phase = {
        "phase": f"lanes-{degree}-{'ephemeral' if ephemeral else 'held'}",
        "holders": [{k: h[k] for k in ("job", "lane", "init", "rc")}
            for h in holders],
        "drivers": drivers,
        "summaries": [],
        "distinct_lanes": len({h["lane"] for h in holders
            if h["lane"] is not None}),
    }
    for driver in drivers:
        phase["summaries"].extend(parse_summaries(driver))
    phase["all_ok"] = all(d["rc"] == 0 for d in drivers) if drivers else False
    phase["any_checksum_bad"] = any(s["bad"] != 0
        for s in phase["summaries"])
    try:
        os.unlink(shm_path)
    except OSError:
        pass
    return phase


def run_evict_phase(args, socket_path, pack_path, lanes):
    command = [str(args.repo / "build" / "mesh_lane_ladder"),
        str(socket_path), "evict", str(lanes), str(pack_path)]
    environment = dict(os.environ)
    environment["SPARK_WEIGHTD_SOCKET"] = str(socket_path)
    completed = subprocess.run(command, capture_output=True,
        timeout=args.timeouts, env=environment)
    output = completed.stdout.decode(errors="replace")
    tests = [line for line in output.splitlines()
        if line.startswith("EVICT-TEST ")]
    passed = completed.returncode == 0 and "EVICT-MATRIX PASS" in output
    return {
        "phase": f"evict-matrix-{lanes}",
        "rc": completed.returncode,
        "pass": passed,
        "tests": tests,
        "stdout_tail": output[-1500:],
        "stderr_tail": completed.stderr.decode(errors="replace")[-1500:],
    }


def daemon_error_lines(log_path):
    if not log_path.exists():
        return []
    return [line[:220] for line in
        log_path.read_text(errors="replace").splitlines()
        if any(pattern in line for pattern in ERROR_PATTERN)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--phase", required=True, choices=["baseline", "after"])
    parser.add_argument("--ladder", default="1,2,4,8")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--evict-lanes", type=int, default=8)
    parser.add_argument("--timeouts", type=int, default=240)
    parser.add_argument("--keep-daemon", action="store_true")
    parser.add_argument("--socket-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()
    args.ladder = [int(x) for x in args.ladder.split(",")]

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = args.socket_dir or (args.repo / "runs" /
        f"mesh-lane-ladder-{args.phase}-{stamp}")
    work_dir.mkdir(parents=True, exist_ok=True)
    socket_path = work_dir / "ladder.sock"
    shm_path = f"/dev/shm/mesh-ladder-{args.phase}-{os.getpid()}.shm"
    pack_path = work_dir / "fixture.pack"
    daemon_log = work_dir / "weightd.log"

    daemon = subprocess.Popen(
        [str(args.repo / "build" / "sparkpipe_weightd"),
         "--socket", str(socket_path), "--device-bytes-max",
         str(8 * 1024 * 1024 * 1024)],
        stdout=open(daemon_log, "w"), stderr=subprocess.STDOUT)
    receipt = {"phase": args.phase, "started": stamp, "work_dir": str(work_dir),
        "ladder": args.ladder, "iters": args.iters, "rows": args.rows,
        "phases": [], "daemon_alive": False, "daemon_errors": []}
    try:
        wait_socket(socket_path, daemon)
        prepare = subprocess.run(
            [str(args.repo / "build" / "weightd_lazy_consumer"),
             "prepare", str(pack_path)], capture_output=True, timeout=60)
        if prepare.returncode != 0:
            raise RuntimeError("fixture prepare failed: "
                + prepare.stderr.decode(errors="replace")[-500:])
        for degree in args.ladder:
            receipt["phases"].append(run_lane_phase(args, socket_path,
                shm_path, degree, False))
        receipt["phases"].append(run_lane_phase(args, socket_path,
            shm_path, 2, True))
        receipt["phases"].append(run_evict_phase(args, socket_path,
            pack_path, args.evict_lanes))
        receipt["daemon_alive"] = daemon.poll() is None
        receipt["daemon_errors"] = daemon_error_lines(daemon_log)
    finally:
        try:
            os.unlink(shm_path)
        except OSError:
            pass
        if not args.keep_daemon:
            daemon.terminate()
            try:
                daemon.wait(timeout=10)
            except subprocess.TimeoutExpired:
                daemon.kill()
    receipt["verdict"] = judge(receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["verdict"]["pass"] else 1


def judge(receipt):
    baseline = receipt["phase"] == "baseline"
    failures = []
    for phase in receipt["phases"]:
        name = phase["phase"]
        if name.startswith("lanes-") and name.endswith("-held"):
            degree = int(name.split("-")[1])
            holder_lanes = sorted(h["lane"] for h in phase["holders"]
                if h["lane"] is not None)
            if baseline:
                if degree <= 2:
                    if not phase["all_ok"]:
                        failures.append({"phase": name,
                            "problem": "held-lane drivers failed"})
                    if phase["distinct_lanes"] != degree:
                        failures.append({"phase": name,
                            "problem": f"distinct lanes "
                            f"{phase['distinct_lanes']} != {degree}"})
                    if phase["any_checksum_bad"]:
                        failures.append({"phase": name,
                            "problem": "checksum failures"})
                else:
                    holders_ok = [h for h in phase["holders"]
                        if h["lane"] is not None]
                    if len(holder_lanes) != 2 or any(h["lane"] >= 2
                            for h in holders_ok):
                        failures.append({"phase": name,
                            "problem": f"baseline must assign exactly lanes "
                            f"[0,1], got {holder_lanes}"})
                    if not any(h["rc"] == 3 for h in phase["holders"]):
                        failures.append({"phase": name,
                            "problem": "baseline must fail closed (rc=3) "
                            "beyond two lanes"})
            else:
                if len(holder_lanes) != degree:
                    failures.append({"phase": name,
                        "problem": f"holder lanes {holder_lanes} "
                        f"!= {degree} jobs"})
                if not phase["all_ok"]:
                    failures.append({"phase": name,
                        "problem": "held-lane drivers failed"})
                if phase["any_checksum_bad"]:
                    failures.append({"phase": name,
                        "problem": "checksum failures"})
        elif name.endswith("-ephemeral"):
            if phase["distinct_lanes"] != 1:
                failures.append({"phase": name,
                    "problem": "ephemeral holders must collapse to one lane"})
        elif name.startswith("evict-matrix"):
            if baseline and phase["pass"]:
                failures.append({"phase": name,
                    "problem": "eviction arbitration unexpectedly present "
                    "on baseline"})
            if not baseline and not phase["pass"]:
                failures.append({"phase": name,
                    "problem": "eviction arbitration failed"})
    if not receipt["daemon_alive"]:
        failures.append({"phase": "daemon", "problem": "daemon died"})
    if receipt["daemon_errors"]:
        failures.append({"phase": "daemon",
            "problem": f"{len(receipt['daemon_errors'])} error lines",
            "lines": receipt["daemon_errors"][:8]})
    return {"pass": not failures, "failures": failures}


if __name__ == "__main__":
    sys.exit(main())
