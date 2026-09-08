#!/usr/bin/env python3
"""Live systemd/queue qualification; uses the controller's authoritative queue."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spark_queue", ROOT / "tools/spark_queue.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def state():
    with queue.transaction() as current:
        return json.loads(json.dumps(current))


def add(name, nodes, ttl=1, resources="gpu"):
    args = argparse.Namespace(id=name, nodes=nodes, per_node=True, cmd="echo started; sleep 60 & wait",
        cmd_file=None, cwd=None, ttl_min=ttl, memory_mib=64, resources=resources,
        kind="run", priority=5, after=None, by="queue-live-gate", notes="bounded infrastructure probe")
    queue.cmd_add(args)


def dispatch():
    queue.cmd_dispatch(argparse.Namespace(ttl=3))


def cancel(name):
    queue.cmd_cancel(argparse.Namespace(id=name))


def job(name):
    return next(j for j in state()["jobs"] if j["id"] == name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(j["state"] in queue.ACTIVE for j in state()["jobs"]):
        raise SystemExit("gate requires no active jobs on the authoritative queue")
    prefix = "qgate-" + str(time.time_ns())
    ids = []
    receipts = {"started_at": time.time(), "checks": []}
    real_remote = queue.remote
    try:
        a, b, cpu = [prefix + suffix for suffix in ["-a", "-b", "-cpu"]]
        for name, node, ttl, resource in [(a,"spark0",0.1,"gpu"), (b,"spark1",1,"gpu"), (cpu,"spark1",1,"cpu")]:
            add(name,node,ttl,resource)
            ids.append(name)
        dispatch()
        assert all(job(n)["state"] == "running" for n in [a,b,cpu]), state()
        receipts["checks"].append("disjoint nodes and same-node CPU/GPU run concurrently")
        # No controller progress during this interval: the node enforces expiry.
        time.sleep(8)
        reply = real_remote(job(a),"spark0","poll")
        assert queue.terminal(reply), reply
        receipts["deadline_observation"] = reply
        dispatch()
        assert any(r["id"] == a and r["exit"] == 124 for r in state()["results"]), state()
        assert job(b)["state"] == "running" and job(cpu)["state"] == "running"
        receipts["checks"].append("node deadline stops parent and child without controller progress")
        cancel(b)
        cancel(cpu)
        dispatch()

        fleet, healthy = prefix+"-fleet", prefix+"-healthy"
        add(fleet,"spark0,sparkf")
        ids.append(fleet)
        dispatch()
        assert job(fleet)["state"] == "running", job(fleet)
        cancel(fleet)
        queue.remote = lambda j,n,a: {"unknown":True} if n == "sparkf" else real_remote(j,n,a)
        add(healthy,"spark0")
        ids.append(healthy)
        dispatch()
        assert job(fleet)["released_nodes"] == ["spark0"], job(fleet)
        assert job(healthy)["state"] == "running", job(healthy)
        receipts["checks"].append("injected controller/peer outage retains only unresolved peer")
        queue.remote = real_remote
        cancel(healthy)
        dispatch()
        assert not any(j["id"] in [fleet,healthy] for j in state()["jobs"]), state()

        lost = prefix+"-lost"
        add(lost,"spark0")
        ids.append(lost)
        def lose_ack(j,n,a):
            reply = real_remote(j,n,a)
            return {"unknown":True} if a == "launch" else reply
        queue.remote = lose_ack
        dispatch()
        attempt = job(lost)["attempt"]
        queue.remote = real_remote
        dispatch()
        assert job(lost)["state"] == "running" and job(lost)["attempt"] == attempt
        rc, log = queue.ssh("spark0","cat /tmp/sparkqueue-"+attempt+".log")
        assert rc == 0 and log.splitlines().count("started") == 1, (rc,log)
        receipts["checks"].append("lost launch ACK reconciles same unit without duplicate execution")
        cancel(lost)
        dispatch()
    finally:
        queue.remote = real_remote
        for current in state()["jobs"]:
            if current["id"] in ids:
                cancel(current["id"])
        dispatch()
        receipts["results"] = [j for j in state()["results"] if j["id"] in ids]
        receipts["remaining"] = [j for j in state()["jobs"] if j["id"] in ids]
        receipts["finished_at"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipts,indent=2)+"\n")
    assert not receipts["remaining"], receipts["remaining"]
    print(json.dumps({"checks":receipts["checks"],"output":str(args.output)},indent=2))


if __name__ == "__main__":
    main()
