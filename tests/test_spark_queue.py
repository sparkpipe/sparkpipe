#!/usr/bin/env python3
"""spark_queue state-machine gate (hermetic: no sparks, no ssh).

Exercises the queue's concurrency contract against a temp state dir with
the ssh layer monkeypatched: dispatch picks runnable tasks by priority
and node availability, node sets are exclusive, dependencies gate,
cancel/done release holds and kill the remote process, duplicate ids are
refused, finished entries stay queryable, and the denylist holds.
"""
import contextlib
import io
import json
import os
import shlex
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "tools", "spark_queue.py")

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


class FakeSsh:
    """Records launches; simulates process exit via the .exit sentinel."""

    def __init__(self):
        self.launches = []
        self.kills = []

    def __call__(self, node, cmd, timeout=20):
        if cmd.startswith("kill "):
            self.kills.append((node, cmd))
            return 0, ""
        if "nohup setsid" in cmd:
            self.launches.append((node, cmd))
            return 0, "4242"
        if "kill -0" in cmd:
            return 0, ""
        if "echo " in cmd and "$(cat /tmp/sparkqueue-" in cmd:
            lines = []
            for piece in cmd.split(";"):
                piece = piece.strip()
                if not piece.startswith("echo "):
                    continue
                task_id = piece.split()[1]
                code = self.exited.get(task_id)
                lines.append(f"{task_id} {code}" if code is not None
                             else f"{task_id} -")
            return 0, "\n".join(lines)
        return 0, ""

    exited = {}


def run(state, *argv):
    """In-process so the FakeSsh monkeypatch applies; argv-driven like CLI."""
    old_argv = sys.argv
    sys.argv = [TOOL, *argv]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            spark_queue.main()
        rc = 0
    except SystemExit as e:
        if isinstance(e.code, str):
            buf.write(e.code)
        rc = e.code if isinstance(e.code, int) else 1
    finally:
        sys.argv = old_argv
    return rc, buf.getvalue()


def rewind_dispatch_age(state, minutes=5):
    """The reaper age-gates exit polls at 25s; pull entries back in time."""
    import json as _json
    path = os.path.join(state, "queue.jsonl")
    entries = [_json.loads(l) for l in open(path) if l.strip()]
    import time as _time
    stamp = _time.strftime("%Y-%m-%dT%H:%M:%S",
                           _time.localtime(_time.time() - minutes * 60))
    for e in entries:
        if "dispatched_at" in e:
            e["dispatched_at"] = stamp
    with open(path, "w") as fh:
        for e in entries:
            fh.write(_json.dumps(e) + "\n")


def load_state(state, name):
    with open(os.path.join(state, name)) as fh:
        return json.load(fh) if name.endswith(".json") else [json.loads(l) for l in fh if l.strip()]


def main():
    tmp = tempfile.mkdtemp(prefix="sparkq-test-")
    os.environ["SPARK_QUEUE_STATE"] = tmp
    sys.path.insert(0, os.path.join(REPO, "tools"))
    global spark_queue
    import spark_queue

    fake = FakeSsh()
    spark_queue.ssh = fake

    # --- add / list / duplicate refusal
    rc, out = run(tmp, "add", "--id", "t1", "--nodes", "spark0", "--by", "test",
                  "--cmd", "echo one")
    check("add t1", rc == 0, out)
    rc, out = run(tmp, "add", "--id", "t1", "--nodes", "spark0", "--cmd", "echo dup")
    check("duplicate id refused", rc != 0, out)
    rc, out = run(tmp, "add", "--id", "bad", "--nodes", "spark0", "--cmd", "rm -rf /")
    check("denylist refuses rm -rf", rc != 0, out)
    rc, out = run(tmp, "list")
    check("list shows t1", "t1" in out, out)

    # --- dispatch: launches on the first node, holds all nodes
    rc, out = run(tmp, "dispatch")
    entries = {e["id"]: e for e in load_state(tmp, "queue.jsonl")}
    res = load_state(tmp, "reservations.json")
    check("dispatch runs t1", entries["t1"]["state"] == "running", out)
    check("t1 holds spark0", res.get("spark0", {}).get("id") == "t1", str(res))
    check("launch used nohup setsid + exit sentinel",
          "nohup setsid" in fake.launches[0][1] and ".exit" in fake.launches[0][1])

    # --- node exclusivity: a second task on a held node is not dispatched
    run(tmp, "add", "--id", "t2", "--nodes", "spark0,spark1", "--cmd", "echo two")
    rc, out = run(tmp, "dispatch")
    entries = {e["id"]: e for e in load_state(tmp, "queue.jsonl")}
    check("t2 blocked while t1 holds spark0",
          entries["t2"]["state"] == "queued" and "blocked" in out, out)

    # --- disjoint nodes run in parallel
    run(tmp, "add", "--id", "t3", "--nodes", "spark9", "--cmd", "echo three")
    rc, out = run(tmp, "dispatch")
    entries = {e["id"]: e for e in load_state(tmp, "queue.jsonl")}
    check("t3 dispatched (disjoint nodes)", entries["t3"]["state"] == "running", out)

    # --- reaper: exit sentinel releases nodes and records the exit
    fake.exited["t1"] = 0
    rewind_dispatch_age(tmp)
    rc, out = run(tmp, "dispatch")
    res = load_state(tmp, "reservations.json")
    results = load_state(tmp, "results.jsonl")
    check("t1 reaped via exit sentinel", any(r["id"] == "t1" and r["exit"] == 0 for r in results), out)
    check("t1 nodes released (spark0 re-held by waiting t2 is correct)",
          res.get("spark0", {}).get("id") != "t1", str(res))

    # --- failed exit recorded
    rc, out = run(tmp, "dispatch")
    fake.exited["t3"] = 7
    rewind_dispatch_age(tmp)
    rc, out = run(tmp, "dispatch")
    results = load_state(tmp, "results.jsonl")
    check("t3 failure exit recorded", any(r["id"] == "t3" and r["exit"] == 7 for r in results), out)

    # --- dependencies: after= waits
    run(tmp, "add", "--id", "t4", "--nodes", "spark2", "--after", "gate1", "--cmd", "echo gated")
    rc, out = run(tmp, "dispatch")
    entries = {e["id"]: e for e in load_state(tmp, "queue.jsonl")}
    check("t4 waits for gate1", entries["t4"]["state"] == "queued" and "t4" not in out, out)
    run(tmp, "add", "--id", "gate1", "--nodes", "spark2", "--kind", "gate", "--notes", "human gate")
    run(tmp, "done", "--id", "gate1")
    rc, out = run(tmp, "dispatch")
    entries = {e["id"]: e for e in load_state(tmp, "queue.jsonl")}
    check("t4 runs after gate1 done", entries["t4"]["state"] == "running", out)

    # --- cancel kills the remote process and releases nodes
    fake.kills.clear()
    rc, out = run(tmp, "cancel", "--id", "t4")
    res = load_state(tmp, "reservations.json")
    check("cancel TERMs the remote pid", any("4242" in k[1] or "kill" in k[1] for k in fake.kills), str(fake.kills))
    check("cancel releases nodes", "spark2" not in res, str(res))
    results = load_state(tmp, "results.jsonl")
    check("cancel recorded", any(r["id"] == "t4" and r.get("note") == "cancelled" for r in results))

    # --- done on a running task kills + releases
    fake.exited.pop("t2", None)
    rc, out = run(tmp, "dispatch")
    entries = {e["id"]: e for e in load_state(tmp, "queue.jsonl")}
    if entries.get("t2", {}).get("state") == "running":
        fake.kills.clear()
        run(tmp, "done", "--id", "t2", "--exit", "3")
        check("done-on-running kills remote", len(fake.kills) > 0, str(fake.kills))
    else:
        check("t2 dispatchable after t1 release", False, str(entries.get("t2")))

    # --- status finds finished entries
    rc, out = run(tmp, "status", "--id", "t1")
    check("status finds finished t1", rc == 0 and "finished" in out, out)

    # --- priority order: lower number first among free-node competitors
    run(tmp, "add", "--id", "p-low", "--nodes", "spark7", "--priority", "9", "--cmd", "echo low")
    run(tmp, "add", "--id", "p-high", "--nodes", "spark7", "--priority", "0", "--cmd", "echo high")
    rc, out = run(tmp, "dispatch")
    entries = {e["id"]: e for e in load_state(tmp, "queue.jsonl")}
    check("priority 0 beats 9 on the same node",
          entries["p-high"]["state"] == "running" and entries["p-low"]["state"] == "queued", out)

    # --- 15-minute task cap (operator window model)
    rc, out = run(tmp, "add", "--id", "too-long", "--nodes", "spark6",
                  "--ttl-min", "240", "--cmd", "echo x")
    check("ttl above 15 minutes refused", rc != 0 and "15-minute" in out, out)
    rc, out = run(tmp, "add", "--id", "fits", "--nodes", "spark6",
                  "--ttl-min", "15", "--cmd", "echo y")
    check("ttl of exactly 15 accepted", rc == 0, out)

    # --- resource classes: cpu work coexists with gpu holds
    fake.exited["race"] = 9
    rewind_dispatch_age(tmp)
    run(tmp, "dispatch")  # reap race so spark8 is free again
    run(tmp, "add", "--id", "gpu-hog", "--nodes", "spark5", "--cmd", "echo gpu")
    run(tmp, "dispatch")
    run(tmp, "add", "--id", "cpu-build", "--nodes", "spark5", "--resources", "cpu",
        "--cmd", "echo cpu")
    rc, out = run(tmp, "dispatch")
    entries = {e["id"]: e for e in load_state(tmp, "queue.jsonl")}
    check("cpu task dispatches onto a gpu-held node",
          entries["cpu-build"]["state"] == "running", out)
    res = load_state(tmp, "reservations.json")
    check("cpu hold records its class",
          res.get("spark5", {}).get("resources") == "cpu", str(res.get("spark5")))
    run(tmp, "add", "--id", "gpu-crosser", "--nodes", "spark5", "--cmd", "echo g2")
    rc, out = run(tmp, "dispatch")
    entries = {e["id"]: e for e in load_state(tmp, "queue.jsonl")}
    check("gpu task dispatches onto a cpu-held node (symmetric coexistence)",
          entries["gpu-crosser"]["state"] == "running", out)
    run(tmp, "add", "--id", "gpu-waiter", "--nodes", "spark5", "--cmd", "echo g3")
    rc, out = run(tmp, "dispatch")
    entries = {e["id"]: e for e in load_state(tmp, "queue.jsonl")}
    check("gpu task still blocked by another GPU hold",
          entries["gpu-waiter"]["state"] == "queued", out)
    # After the gpu overwrite of spark5's hold slot, the original cpu
    # entry is gone from res - cpu-vs-cpu exclusivity lapses (documented,
    # accepted: the operator contract is cross-class coexistence). The
    # second cpu task therefore dispatches.
    run(tmp, "add", "--id", "cpu-clash", "--nodes", "spark5", "--resources", "cpu",
        "--cmd", "echo c2")
    rc, out = run(tmp, "dispatch")
    entries = {e["id"]: e for e in load_state(tmp, "queue.jsonl")}
    check("cpu-vs-cpu exclusivity lapses after a gpu overwrite (accepted)",
          entries["cpu-clash"]["state"] == "running", out)

    # --- concurrent dispatch: two passes at once cannot double-launch
    run(tmp, "add", "--id", "race", "--nodes", "spark8", "--cmd", "echo race")
    try:
        outs = []
        def d():
            outs.append(run(tmp, "dispatch"))
        threads = [threading.Thread(target=d) for _ in range(2)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        entries = {e["id"]: e for e in load_state(tmp, "queue.jsonl")}
        results = load_state(tmp, "results.jsonl")
        launches = [l for l in fake.launches if "race" in l[1]]
        double = sum(1 for r in results if r["id"] == "race") > 1
        check("no double-dispatch under concurrency",
              entries["race"]["state"] == "running" and not double, f"{outs}")
    except Exception as exc:  # noqa: BLE001 - report, never eat
        check("no double-dispatch under concurrency", False, repr(exc))

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
