#!/usr/bin/env python3
"""Spark run queue: reserve nodes, enqueue GPU/script runs, dispatch
non-conflicting entries in parallel, let long runs cook.

The OS-for-inference idea applied to ourselves: nodes are the scarce
resource, entries declare the nodes they need, the scheduler runs every
entry whose node set is free (disjoint sets run in parallel), and a
running entry HOLDS its nodes until it finishes - no preemption, no
contention, long benches cook undisturbed.

Files (repo-local, coordinator Mac - agents and the sweep both write):
  runs/queue.jsonl       one JSON entry per line (append-only via lock)
  runs/reservations.json {node: {id, holder, acquired_at, ttl_minutes, pid}}
  runs/results.jsonl     finished entries with exit status

Entry fields:
  id            unique, required
  nodes         ["spark3", ...] required - the exclusive set
  cmd           remote shell command (runs via ssh, nohup, on nodes[0])
  cwd           remote working directory (default $HOME)
  priority      lower runs first (0 = highest, default 5)
  kind          "run" (executes cmd) | "gate" (nothing to execute; blocks
                dependents until a human/agent marks it done) | "note"
  after         [ids/tags] entries that must be done before this one
  class         "short" (<15m) | "long" (holds nodes indefinitely)
  submitted_by  free text
  notes         free text

Usage:
  spark_queue.py add --id X --nodes spark3 --cmd '...' [options]
  spark_queue.py list [--all]        queued+running (or everything)
  spark_queue.py status ID           one entry incl. log tail
  spark_queue.py done ID [--exit N]  mark done (gates, manual completions)
  spark_queue.py cancel ID
  spark_queue.py reserve --node sparkX --holder lane-y [--ttl-min 360]
  spark_queue.py release --node sparkX | --id ID
  spark_queue.py schedule            ONE dispatch pass (the sweep calls this
                                     every 30m; safe to run manually too)

Denylist: commands containing reboot/shutdown/poweroff/'kill -9'/SIGKILL/
rm -rf are refused - the no-KILL and no-reboot rules apply to the queue.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
QUEUE = os.path.join(RUNS, "queue.jsonl")
RESERV = os.path.join(RUNS, "reservations.json")
RESULTS = os.path.join(RUNS, "results.jsonl")
LOGDIR = os.path.join(RUNS, "logs")
LOCK = os.path.join(RUNS, ".lock")
DENY = ("reboot", "shutdown", "poweroff", "init 0", "init 6",
        "kill -9", "kill -KILL", "SIGKILL", "rm -rf")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def acquire_lock():
    import fcntl
    os.makedirs(RUNS, exist_ok=True)
    fh = open(LOCK, "a+")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def load_queue():
    os.makedirs(RUNS, exist_ok=True)
    if not os.path.exists(QUEUE):
        return []
    entries = []
    with open(QUEUE) as fh:
        for line in fh:
            line = line.strip()
            if line:
                e = json.loads(line)
                if e.get("state") in (None, "queued", "running", "blocked"):
                    entries.append(e)
    return entries


def rewrite_queue(entries):
    with open(QUEUE, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def load_reservations():
    if not os.path.exists(RESERV):
        return {}
    try:
        with open(RESERV) as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return {}


def save_reservations(res):
    with open(RESERV, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)


def append_result(entry, exit_code, note=""):
    os.makedirs(RUNS, exist_ok=True)
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(dict(entry, state="finished",
            exit=exit_code, finished_at=now(), note=note)) + "\n")


def ssh(node, cmd, timeout=20):
    try:
        out = subprocess.run(["ssh"] + SSH_OPTS + [node, cmd],
            capture_output=True, text=True, timeout=timeout)
        return out.returncode, out.stdout.strip()
    except subprocess.TimeoutExpired:
        return 124, ""


def check_denied(cmd):
    low = cmd.lower()
    for bad in DENY:
        if bad.lower() in low:
            sys.exit(f"REFUSED: command contains denied token '{bad}' "
                     "(no-reboot / no-KILL policy)")


def pid_alive(node, pid):
    rc, _ = ssh(node, f"kill -0 {pid} 2>/dev/null")
    return rc == 0


def expire_stale(res):
    """Release reservations whose TTL passed and whose pid (if any) is dead."""
    changed = False
    for node in list(res):
        r = res[node]
        ttl = float(r.get("ttl_minutes", 0) or 0)
        age = (time.time() - time.mktime(time.strptime(
            r.get("acquired_at", now()), "%Y-%m-%dT%H:%M:%S"))) / 60.0
        pid = r.get("pid")
        if ttl and age > ttl:
            if not pid or not pid_alive(node, pid):
                del res[node]
                changed = True
    return changed


def cmd_add(args):
    check_denied(args.cmd or "")
    entries = load_queue()
    ids = {e["id"] for e in entries} | {
        json.loads(l)["id"] for l in open(RESULTS) if l.strip()} \
        if os.path.exists(RESULTS) else {e["id"] for e in entries}
    if args.id in ids:
        sys.exit(f"id '{args.id}' already exists")
    entry = dict(id=args.id, nodes=args.nodes.split(","),
        cmd=args.cmd or "", cwd=args.cwd or "$HOME",
        priority=args.priority, kind=args.kind, class_=args.klass,
        after=[a for a in (args.after or "").split(",") if a],
        submitted_by=args.by, notes=args.notes or "",
        state="queued", submitted_at=now())
    del entry["class_"]
    entry["class"] = args.klass
    entries.append(entry)
    rewrite_queue(entries)
    print(f"queued {args.id} nodes={args.nodes} kind={args.kind} "
          f"after={entry['after']}")


def cmd_list(args):
    entries = load_queue()
    res = load_reservations()
    if not args.all:
        entries = [e for e in entries if e["state"] in ("queued", "running", "blocked")]
    entries.sort(key=lambda e: (e.get("priority", 5), e.get("submitted_at", "")))
    for e in entries:
        hold = ",".join(n for n in e["nodes"] if n in res and res[n].get("id") == e["id"])
        print(f"{e['state']:8} {e.get('priority',5)} {e['id']:24} "
              f"[{','.join(e['nodes'])}] hold={hold or '-'} "
              f"after={e.get('after') or '-'} {e.get('notes','')[:60]}")
    if not entries:
        print("(queue empty)")


def cmd_status(args):
    for e in load_queue():
        if e["id"] == args.id:
            print(json.dumps(e, indent=1, sort_keys=True))
            if e.get("remote_log"):
                rc, tail = ssh(e["nodes"][0], f"tail -5 {e['remote_log']}")
                if rc == 0 and tail:
                    print("--- log tail ---\n" + tail)
            return
    sys.exit(f"no entry '{args.id}'")


def cmd_done(args):
    entries = load_queue()
    res = load_reservations()
    hit = False
    for e in entries:
        if e["id"] == args.id:
            e["state"] = "done"
            append_result(e, args.exit, "marked done")
            for n in e["nodes"]:
                if res.get(n, {}).get("id") == e["id"]:
                    del res[n]
            hit = True
    if not hit:
        sys.exit(f"no entry '{args.id}'")
    rewrite_queue([e for e in entries if e["state"] not in ("done",)])
    save_reservations(res)
    print(f"done {args.id} (exit {args.exit})")



def pick_runnable(entries, results_ids):
    done_ids = results_ids
    best = None
    for e in entries:
        if e.get("state") != "queued" or not e.get("cmd"):
            continue
        if any(a not in done_ids for a in e.get("after", [])):
            continue
        if best is None or (-e["priority"], e["submitted_at"]) <                 (-best["priority"], best["submitted_at"]):
            best = e
    return best


def nodes_free(nodes, res):
    return all(n not in res for n in nodes)


def cmd_dispatch(args):
    """Task-based dispatch: run the head runnable task NOW if its nodes are
    lease-free; hold nodes only for the task's duration; release on exit.
    This is the operator's contract: the queue is TASK-based, not
    wall-clock-based - a lane codes on CPU while the sparks serve the
    next task."""
    with acquire_lock():
        entries = load_queue()
        res = load_reservations()
        results_ids = {json.loads(l)["id"] for l in open(RESULTS) \
            if l.strip()} if os.path.exists(RESULTS) else set()
        # reap: any running task whose remote pid is gone and exit file says done
        for e in [x for x in entries if x.get("state") == "running"]:
            n0 = e["nodes"][0]
            pid = res.get(n0, {}).get("pid")
            exit_path = f"/tmp/sparkqueue-{e['id']}.exit"
            _, rc_txt = ssh(n0, f"cat {exit_path} 2>/dev/null", timeout=10)
            if rc_txt.strip().lstrip('-').isdigit():
                rc = int(rc_txt.strip())
                e["state"] = "done"
                append_result(e, rc, "dispatch reaped")
                for n in e["nodes"]:
                    if res.get(n, {}).get("id") == e["id"]:
                        del res[n]
                print(f"reaped {e['id']} exit={rc} (nodes released)")
                continue
            if pid and not pid_alive(n0, pid):
                e["state"] = "done"
                append_result(e, -1, "dispatch: pid gone without exit file")
                for n in e["nodes"]:
                    if res.get(n, {}).get("id") == e["id"]:
                        del res[n]
                print(f"reaped {e['id']} pid-gone (nodes released)")
        entries = [e for e in entries if e.get("state") != "done"]
        task = pick_runnable(entries, results_ids)
        if task is None:
            rewrite_queue(entries); save_reservations(res)
            print("nothing runnable")
            return
        if not nodes_free(task["nodes"], res):
            rewrite_queue(entries); save_reservations(res)
            print(f"blocked: {task['id']} nodes busy")
            return
        for n in task["nodes"]:
            res[n] = dict(id=task["id"], holder=f"task:{task['id']}",
                acquired_at=now(), ttl_minutes=float(args.ttl),
                pid=None)
        n0 = task["nodes"][0]
        inner = task['cmd'] + "; echo \"$?\" > /tmp/sparkqueue-" + task['id'] + ".exit"
        wrapper = ("cd " + task.get('cwd', '$HOME') + " && nohup setsid bash -c "
                   + shlex.quote(inner)
                   + " >/tmp/sparkqueue-" + task['id'] + ".log 2>&1 & echo $!")
        _, out = ssh(n0, wrapper, timeout=20)
        pid = out.splitlines()[-1] if out else None
        res[n0]["pid"] = pid
        task["state"] = "running"
        task["dispatched_at"] = now()
        rewrite_queue(entries); save_reservations(res)
        print(f"dispatched {task['id']} nodes={','.join(task['nodes'])} pid={pid}")

def cmd_cancel(args):
    entries = load_queue()
    res = load_reservations()
    kept = []
    for e in entries:
        if e["id"] == args.id:
            append_result(e, -1, "cancelled")
            for n in e["nodes"]:
                if res.get(n, {}).get("id") == e["id"]:
                    del res[n]
        else:
            kept.append(e)
    rewrite_queue(kept)
    save_reservations(res)
    print(f"cancelled {args.id}")


def cmd_reserve(args):
    res = load_reservations()
    if args.node in res:
        sys.exit(f"{args.node} already reserved: {res[args.node]}")
    res[args.node] = dict(id=f"manual:{args.holder}", holder=args.holder,
        acquired_at=now(), ttl_minutes=args.ttl_min)
    save_reservations(res)
    print(f"reserved {args.node} for {args.holder} ttl={args.ttl_min}m")


def cmd_release(args):
    res = load_reservations()
    if args.node:
        if args.node in res:
            del res[args.node]
            save_reservations(res)
            print(f"released {args.node}")
        else:
            print(f"{args.node} was not reserved")
    elif args.id:
        for n in list(res):
            if res[n].get("id") == args.id:
                del res[n]
        save_reservations(res)
        print(f"released nodes held by {args.id}")


def cmd_schedule(args):
    """One dispatch pass: poll running entries, then launch runnable ones."""
    lock = acquire_lock()
    try:
        entries = load_queue()
        res = load_reservations()
        done_ids = set()
        if os.path.exists(RESULTS):
            done_ids = {json.loads(l)["id"] for l in open(RESULTS) if l.strip()}
        if expire_stale(res):
            save_reservations(res)

        # 1) poll running entries: pid dead -> finished (log exit note)
        for e in entries:
            if e.get("state") != "running":
                continue
            node, pid = e["nodes"][0], e.get("pid")
            if not pid or not pid_alive(node, pid):
                rc, tail = ssh(node, f"tail -1 {e['remote_log']} 2>/dev/null")
                e["state"] = "done"
                append_result(e, 0 if rc == 0 else 1,
                    "process exited; " + tail[:120])
                for n in e["nodes"]:
                    if res.get(n, {}).get("id") == e["id"]:
                        del res[n]
                print(f"finished {e['id']} (log: {e['remote_log']})")

        # 2) blocked -> queued when dependencies are done
        for e in entries:
            if e.get("state") == "blocked" and \
               all(a in done_ids for a in e.get("after", [])):
                e["state"] = "queued"

        # 3) launch: priority order, nodes must be entirely free
        for e in sorted(entries, key=lambda x: x.get("priority", 5)):
            if e.get("state") != "queued":
                continue
            if e.get("after") and not all(a in done_ids for a in e["after"]):
                e["state"] = "blocked"
                continue
            if any(n in res for n in e["nodes"]):
                continue
            if e.get("kind") == "gate":
                e["state"] = "blocked"   # holds nothing; waits for done-mark
                print(f"gate {e['id']} waiting for {e.get('after')}")
                continue
            if e.get("kind") == "note":
                e["state"] = "done"
                append_result(e, 0, "note")
                continue
            log = f"/tmp/sparkq/{e['id']}.log"
            quoted = shlex.quote(e["cmd"])
            launch = (f"mkdir -p /tmp/sparkq && cd {e['cwd']} && "
                      f"nohup bash -c {quoted} > {log} 2>&1 & echo $!")
            rc, out = ssh(e["nodes"][0], launch)
            if rc != 0 or not out.isdigit():
                print(f"LAUNCH FAILED {e['id']}: rc={rc} out={out!r}")
                continue
            e.update(state="running", pid=int(out), remote_log=log,
                     started_at=now())
            for n in e["nodes"]:
                res[n] = dict(id=e["id"], holder=e.get("submitted_by", "?"),
                              acquired_at=now(), pid=int(out),
                              ttl_minutes=0)
            print(f"launched {e['id']} on {e['nodes'][0]} pid={out} log={log}")

        rewrite_queue([e for e in entries if e["state"] != "done"])
        save_reservations(res)
        held = ", ".join(f"{n}:{r['id']}" for n, r in sorted(res.items())) or "-"
        print(f"reservations: {held}")
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--id", required=True)
    a.add_argument("--nodes", required=True, help="comma list, e.g. spark3")
    a.add_argument("--cmd")
    a.add_argument("--cwd")
    a.add_argument("--priority", type=int, default=5)
    a.add_argument("--kind", default="run", choices=["run", "gate", "note"])
    a.add_argument("--after", help="comma list of ids this waits for")
    a.add_argument("--klass", default="long", choices=["short", "long"])
    a.add_argument("--by", default="coordinator")
    a.add_argument("--notes", default="")
    a.set_defaults(fn=cmd_add)
    a = sub.add_parser("list")
    a.add_argument("--all", action="store_true")
    a.set_defaults(fn=cmd_list)
    a = sub.add_parser("status")
    a.add_argument("id" if False else "--id", dest="id", required=True)
    a.set_defaults(fn=cmd_status)
    a = sub.add_parser("done")
    a.add_argument("--id", required=True)
    a.add_argument("--exit", type=int, default=0)
    a.set_defaults(fn=cmd_done)
    a = sub.add_parser("dispatch")
    a.add_argument("--ttl", type=int, default=45,
        help="lease minutes held for the TASK (not the lane)")
    a.set_defaults(fn=cmd_dispatch)
    a = sub.add_parser("cancel")
    a.add_argument("--id", required=True)
    a.set_defaults(fn=cmd_cancel)
    a = sub.add_parser("reserve")
    a.add_argument("--node", required=True)
    a.add_argument("--holder", required=True)
    a.add_argument("--ttl-min", type=float, default=360.0)
    a.set_defaults(fn=cmd_reserve)
    a = sub.add_parser("release")
    a.add_argument("--node")
    a.add_argument("--id")
    a.set_defaults(fn=cmd_release)
    a = sub.add_parser("schedule")
    a.set_defaults(fn=cmd_schedule)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
