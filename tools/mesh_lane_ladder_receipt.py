#!/usr/bin/env python3
import argparse
import json
import pathlib
import queue
import re
import shlex
import subprocess
import threading
import time


def validate_profile(profile):
    if set(profile) != {"source_commit", "binary_sha256", "binary", "socket", "pack", "hosts", "lanes"}:
        raise ValueError("profile requires source_commit, binary_sha256, binary, socket, pack, hosts, lanes")
    for field, length in (("source_commit", 40), ("binary_sha256", 64)):
        if re.fullmatch(r"[0-9a-f]{%d}" % length, profile[field]) is None:
            raise ValueError(f"invalid {field}")
    for field in ("binary", "socket", "pack"):
        if not isinstance(profile[field], str) or not profile[field].startswith("/"):
            raise ValueError(f"{field} must be an absolute remote path")
    hosts = profile["hosts"]
    if not isinstance(hosts, list) or len(hosts) != 16 or len(set(hosts)) != 16:
        raise ValueError("hosts must contain sixteen distinct physical-rank SSH names")
    if any(not isinstance(host, str) or re.fullmatch(r"[A-Za-z0-9_.@-]+", host) is None or host.startswith("-") for host in hosts):
        raise ValueError("invalid SSH host")
    lanes = profile["lanes"]
    if not isinstance(lanes, list) or not 1 <= len(lanes) <= 8:
        raise ValueError("one to eight lanes required")
    seen = set()
    for lane in lanes:
        if set(lane) != {"lane", "physical_ranks", "rows"}:
            raise ValueError("lane requires lane, physical_ranks, rows")
        number, ranks, rows = lane["lane"], lane["physical_ranks"], lane["rows"]
        if type(number) is not int or not 0 <= number < 8 or number in seen:
            raise ValueError("lane IDs must be distinct and in 0..7")
        seen.add(number)
        if type(rows) is not int or not 1 <= rows <= 512:
            raise ValueError("rows must be in 1..512")
        if not isinstance(ranks, list) or not 2 <= len(ranks) <= 16 or any(type(rank) is not int or not 0 <= rank < 16 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError("each topology requires two to sixteen distinct physical ranks")
    return profile


def fields(line, prefix):
    if not line.startswith(prefix + " "):
        return None
    values = {}
    for token in line.split()[1:]:
        key, value = token.split("=", 1)
        if key in values:
            raise ValueError(f"duplicate output field: {key}")
        values[key] = value
    return values


def accept_event(record, line, iters, binary_sha256):
    if line.startswith("LADDER-FAIL "):
        raise ValueError(line.strip())
    digest = re.fullmatch(r"([0-9a-f]{64})  .+\n?", line)
    if digest:
        if record.get("binary_sha256") is not None or digest[1] != binary_sha256:
            raise ValueError("duplicate or mismatched binary hash")
        record["binary_sha256"] = digest[1]
        return
    for kind in ("LADDER-READY", "ROUND", "LADDER-DONE", "SUMMARY"):
        value = fields(line, kind)
        if value is None:
            continue
        if int(value["lane"]) != record["lane"] or int(value["rank"]) != record["rank"]:
            raise ValueError("output lane/rank differs from launched identity")
        if kind == "LADDER-READY":
            if record.get("ready") is not None or record.get("binary_sha256") != binary_sha256:
                raise ValueError("duplicate readiness or missing binary provenance")
            if int(value["degree"]) != len(record["physical_ranks"]) or int(value["physical"]) != record["physical_ranks"][record["rank"]] or int(value["rows"]) != record["rows"] or value["map"] != ",".join(map(str, record["physical_ranks"])) or value["transport"] != "rdma":
                raise ValueError("readiness topology differs from profile")
            record["ready"] = value
        elif kind == "ROUND":
            ordinal, operation = int(value["ordinal"]), int(value["operation"])
            key = f"{ordinal}:{operation}"
            if not record.get("released") or not 0 <= ordinal <= iters or operation not in (0, 1, 2) or key in record["rounds"] or value["status"] != "ok" or int(value["graph"]) != int(ordinal > 0):
                raise ValueError("invalid, duplicate or unreleased numerical round")
            record["rounds"][key] = value
        elif kind == "LADDER-DONE":
            if record.get("done") is not None or len(record["rounds"]) != 3 * (iters + 1) or int(value["rounds"]) != len(record["rounds"]) or int(value["callbacks"]) != 3:
                raise ValueError("incomplete numerical rounds or callback count")
            record["done"] = value
        else:
            if not record.get("cleanup_released") or record.get("summary") is not None or int(value["rounds"]) != 3 * (iters + 1) or int(value["ok"]) != int(value["rounds"]) or int(value["bad"]) != 0 or int(value["callbacks"]) != 3 or value["transport"] != "rdma":
                raise ValueError("invalid or premature cleanup summary")
            record["summary"] = value
        return


def run(profile, output, iters, timeout):
    output.mkdir(parents=True, exist_ok=False)
    events = queue.Queue()
    records, processes, readers = [], [], []
    started = time.monotonic()
    deadline = started + timeout
    receipt = {"profile": profile, "iters": iters, "qualification": "real common CUDA collectives and daemon/NIC routing; no model inference", "records": records, "pass": False}

    def reader(index, process):
        with (output / f"lane{records[index]['lane']}-rank{records[index]['rank']}.log").open("w") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                events.put((index, line))
        events.put((index, None))

    def wait_for(key):
        while not all(record.get(key) for record in records):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timeout waiting for {key}")
            index, line = events.get(timeout=remaining)
            if line is None:
                records[index]["eof"] = True
                if key != "eof":
                    raise RuntimeError(f"rank exited before {key}: lane={records[index]['lane']} rank={records[index]['rank']}")
            else:
                accept_event(records[index], line, iters, profile["binary_sha256"])

    try:
        for lane in profile["lanes"]:
            for rank, physical in enumerate(lane["physical_ranks"]):
                arguments = [profile["binary"], profile["socket"], "mesh", str(rank), str(len(lane["physical_ranks"])), str(iters), str(lane["rows"]), str(lane["lane"]), profile["pack"], ",".join(map(str, lane["physical_ranks"]))]
                remote = shlex.join(["sha256sum", "--", profile["binary"]]) + " && exec " + shlex.join(["env", "SPARK_TP_WAIT_MODE=hardware", *arguments])
                command = ["ssh", "-T", "-oBatchMode=yes", "-oConnectTimeout=10", profile["hosts"][physical], remote]
                records.append({**lane, "rank": rank, "host": profile["hosts"][physical], "command": command, "rounds": {}})
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                processes.append(process)
                thread = threading.Thread(target=reader, args=(len(records) - 1, process), daemon=True)
                readers.append(thread)
                thread.start()
        wait_for("ready")
        receipt["all_ready_seconds"] = time.monotonic() - started
        for record, process in zip(records, processes):
            record["released"] = True
            process.stdin.write("G")
            process.stdin.flush()
        wait_for("done")
        receipt["all_done_seconds"] = time.monotonic() - started
        for record, process in zip(records, processes):
            record["cleanup_released"] = True
            process.stdin.write("R")
            process.stdin.flush()
        wait_for("eof")
        for record, process in zip(records, processes):
            record["returncode"] = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            if record["returncode"] != 0 or not record.get("summary"):
                raise RuntimeError("rank failed or omitted terminal cleanup summary")
        receipt["pass"] = True
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for record, process in zip(records, processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            record["returncode"] = process.returncode
        for thread in readers:
            thread.join(timeout=1)
        for process in processes:
            process.stdin.close()
            process.stdout.close()
        receipt["seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description="Numerically qualify concurrent lane topologies against already running private fleet mesh daemons.")
    parser.add_argument("--profile", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--iters", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.iters <= 1024 or not 1 <= args.timeout <= 180:
        parser.error("iters must be 1..1024 and timeout 1..180")
    profile = validate_profile(json.loads(args.profile.read_text()))
    if args.validate_only:
        print(json.dumps({"lanes": len(profile["lanes"]), "processes": sum(len(lane["physical_ranks"]) for lane in profile["lanes"]), "validated_profile": True}))
        return 0
    receipt = run(profile, args.output, args.iters, args.timeout)
    print(json.dumps({"pass": receipt["pass"], "receipt": str(args.output / "receipt.json"), "error": receipt.get("error")}))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
