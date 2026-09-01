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
from datetime import datetime
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# THE QUEUE STATE IS MACHINE-GLOBAL and DURABLE (the split-brain fix, then
# the /tmp-is-volatile fix: macOS cleans /tmp and reboots wipe it - queued
# work must survive both). Every worktree and every dev session on this
# host sees ONE queue at ~/.sparkpipe/queue.
STATE = os.environ.get("SPARK_QUEUE_STATE",
    os.path.expanduser("~/.sparkpipe/queue"))
os.makedirs(STATE, exist_ok=True)
QUEUE = os.path.join(STATE, "queue.jsonl")
RESERV = os.path.join(STATE, "reservations.json")
RESULTS = os.path.join(STATE, "results.jsonl")
LOGDIR = os.path.join(STATE, "logs")
LOCK = os.path.join(STATE, ".lock")
# one-time migrations: absorb prior state locations into the DEFAULT home
# only - an explicit SPARK_QUEUE_STATE override (tests, scratch queues)
# must start EMPTY, never inheriting production state.
import shutil
if "SPARK_QUEUE_STATE" not in os.environ:
    for _prior in ("/tmp/sparkqueue", os.path.join(ROOT, "runs")):
        if not os.path.exists(QUEUE) and os.path.exists(os.path.join(_prior, "queue.jsonl")):
            for name in ("queue.jsonl", "reservations.json", "results.jsonl"):
                src = os.path.join(_prior, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(STATE, name))
# NOTE: task-side sentinel files (exit/pid/log) stay in the NODE's /tmp -
# they are transient by design; only this host-side state is durable.
DENY = ("reboot", "shutdown", "poweroff", "init 0", "init 6",
        "kill -9", "kill -KILL", "SIGKILL", "rm -rf")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def acquire_lock():
    import fcntl
    os.makedirs(STATE, exist_ok=True)
    fh = open(LOCK, "a+")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def load_queue():
    os.makedirs(STATE, exist_ok=True)
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
    tmp = QUEUE + ".tmp"
    with open(tmp, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    os.replace(tmp, QUEUE)


def load_reservations():
    if not os.path.exists(RESERV):
        return {}
    try:
        with open(RESERV) as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return {}


def save_reservations(res):
    tmp = RESERV + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    os.replace(tmp, RESERV)


def append_result(entry, exit_code, note=""):
    os.makedirs(STATE, exist_ok=True)
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
    """Release reservations whose pid is dead OR whose TTL passed.

    The old rule required BOTH (ttl expired AND pid dead) - a hung or
    pid-recycled task held its nodes forever (the 19h-running/45min-ttl
    incident class). Each condition is now sufficient on its own; the
    TTL is the authority, and the release never kills anything (the
    no-KILL rule) - the orphan process simply no longer locks the queue.
    """
    changed = False
    for node in list(res):
        r = res[node]
        ttl = float(r.get("ttl_minutes", 0) or 0)
        age = (time.time() - time.mktime(time.strptime(
            r.get("acquired_at", now()), "%Y-%m-%dT%H:%M:%S"))) / 60.0
        pid = r.get("pid")
        if pid and not pid_alive(node, pid):
            del res[node]
            changed = True
        elif ttl and age > ttl:
            del res[node]
            changed = True
            print(f"EXPIRED {node} reservation for {r.get('id', '?')} "
                  f"(ttl {ttl:.0f}m exceeded at age {age:.0f}m; pid "
                  f"{pid} left running but no longer holds the queue)")
    return changed


def cmd_doctor(args):
    """Onboarding + health check: one command a new dev runs first."""
    checks = []
    ok = lambda name, detail="": checks.append((name, True, detail))
    bad = lambda name, detail="": checks.append((name, False, detail))
    writable = os.access(STATE, os.W_OK)
    (ok if writable else bad)("state dir", STATE)
    n_queued = n_running = 0
    for e in load_queue():
        if e.get("state") == "running":
            n_running += 1
        elif e.get("state") == "queued":
            n_queued += 1
    ok("queue", f"{n_queued} queued, {n_running} running, "
        f"{sum(1 for _ in open(RESULTS)) if os.path.exists(RESULTS) else 0} finished")
    res = load_reservations()
    holds = ", ".join(f"{n}={r['id']}" for n, r in sorted(res.items())) or "(none)"
    ok("node holds", holds)
    try:
        out = subprocess.run(["pgrep", "-f", "spark_queue.py dispatch"],
            capture_output=True, text=True, timeout=5)
        pids = [l for l in out.stdout.split() if l]
        if pids:
            ok("dispatcher daemon", f"pid {','.join(pids)}")
        else:
            bad("dispatcher daemon", "not running - restart: "
                "nohup bash -c 'while true; do python3 "
                "/Users/mac/sparkpipe/tools/spark_queue.py dispatch >> "
                "~/.sparkpipe/queue/dispatcher.log 2>&1; sleep 5; done' &")
    except Exception as exc:
        bad("dispatcher daemon", str(exc))
    with open(__file__) as fh:
        body = fh.read()
    if "STATE = " in body and 'os.path.join(ROOT, "runs")' not in body.split("def ")[0][:3000]:
        ok("tool version", "global-state version")
    else:
        bad("tool version", "stale pre-split-brain copy - git pull main")
    for name, good, detail in checks:
        print(f"{'PASS' if good else 'FAIL'}  {name}: {detail}")
    sys.exit(1 if any(not g for _, g, _ in checks) else 0)

def cmd_add(args):
    if getattr(args, "cmd_file", None):
        if args.cmd:
            sys.exit("--cmd and --cmd-file are mutually exclusive")
        with open(args.cmd_file) as fh:
            args.cmd = fh.read()
    with acquire_lock():
        check_denied(args.cmd or "")
        entries = load_queue()
        ids = {e["id"] for e in entries} | {
            json.loads(l)["id"] for l in open(RESULTS) if l.strip()} \
            if os.path.exists(RESULTS) else {e["id"] for e in entries}
        if args.id in ids:
            sys.exit(f"id '{args.id}' already exists")
        entry = dict(id=args.id, nodes=args.nodes.split(","),
            resources=args.resources,
            ttl_minutes=args.ttl_min,
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
    with acquire_lock():
        entries = load_queue()
    res = load_reservations()
    if not args.all:
        entries = [e for e in entries if e["state"] in ("queued", "running", "blocked")]
    entries.sort(key=lambda e: (e.get("priority", 5), e.get("submitted_at", "")))
    for e in entries:
        hold = ",".join(n for n in e["nodes"] if n in res and res[n].get("id") == e["id"])
        print(f"{e['state']:8} {e.get('priority',5)} {e['id']:24} "
              f"[{e.get('resources','gpu')}]"
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
    if os.path.exists(RESULTS):
        for line in reversed(open(RESULTS).read().splitlines()):
            if line.strip() and json.loads(line)["id"] == args.id:
                print(json.dumps(json.loads(line), indent=1, sort_keys=True))
                print("(finished - see results.jsonl; task log was "
                      f"/tmp/sparkqueue-{args.id}.log on {json.loads(line)['nodes'][0]})")
                return
    sys.exit(f"no entry '{args.id}'")


def cmd_done(args):
    with acquire_lock():
        entries = load_queue()
        res = load_reservations()
        hit = False
        for e in entries:
            if e["id"] == args.id:
                if e.get("state") == "running":
                    kill_remote_task(e, res)
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
        if best is None or (e["priority"], e["submitted_at"]) < \
                (best["priority"], best["submitted_at"]):
            best = e
    return best


def nodes_free(nodes, res, resources="gpu"):
    # Resource classes are SYMMETRIC (operator ruling 2026-09-01): cpu
    # work and gpu work coexist on the same node; only a hold of the
    # SAME class blocks. A gpu task blocked by a cpu hold (pack builds
    # freezing the whole gpu queue) was the one-sided original and is
    # wrong. When both classes hold one node the later hold OVERWRITES
    # the earlier - cpu-vs-cpu exclusivity then lapses for the
    # overwritten entry, which is accepted: the operator contract is
    # only cross-class coexistence.
    blocking = resources
    return all(res[n].get("resources", "gpu") != blocking for n in nodes if n in res)


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
            # TTL FIRST: a running task past its ttl_minutes expires even
            # with a live pid (hung or pid-recycled processes used to hold
            # nodes forever). The process is never killed - it just stops
            # locking the queue.
            tttl = float(e.get("ttl_minutes", 0) or 0)
            tacq = res.get(n0, {}).get("acquired_at")
            if tttl and tacq:
                tage = (time.time() - time.mktime(time.strptime(
                    tacq, "%Y-%m-%dT%H:%M:%S"))) / 60.0
                if tage > tttl:
                    e["state"] = "done"
                    append_result(e, 124, f"ttl-expired at {tage:.0f}m "
                                   f"(limit {tttl:.0f}m; pid left running)")
                    for n in e["nodes"]:
                        if res.get(n, {}).get("id") == e["id"]:
                            del res[n]
                    print(f"expired {e['id']} ttl={tttl:.0f}m age={tage:.0f}m "
                          f"(nodes released)")
                    continue
            continue
        # Exit-file polls BATCHED PER NODE and age-gated: each poll is an
        # ssh under the global lock, and N running tasks on N nodes with a
        # dead one used to hold the lock for N x timeout - every other
        # dev's add/list stacked behind it. One ssh per node; tasks
        # younger than 25s skip this pass (the next 5s pass catches them).
        exit_results = {}
        by_node = {}
        for e in entries:
            if e.get("state") != "running":
                continue
            try:
                age_s = (time.time() - time.mktime(time.strptime(
                    e.get("dispatched_at", now()), "%Y-%m-%dT%H:%M:%S")))
            except Exception:
                age_s = 1e9
            if age_s < 25.0:
                continue
            by_node.setdefault(e["nodes"][0], []).append(e["id"])
        for node, ids in by_node.items():
            listing = "; ".join(
                f"echo {i} $(cat /tmp/sparkqueue-{i}.exit 2>/dev/null)"
                for i in ids)
            _, out = ssh(node, listing, timeout=12)
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].lstrip('-').isdigit():
                    exit_results[parts[0]] = int(parts[1])
        for e in [x for x in entries if x.get("state") == "running"]:
            n0 = e["nodes"][0]
            if e["id"] in exit_results:
                e["state"] = "done"
                append_result(e, exit_results[e["id"]], "dispatch reaped")
                for n in e["nodes"]:
                    if res.get(n, {}).get("id") == e["id"]:
                        del res[n]
                print(f"reaped {e['id']} exit={exit_results[e['id']]} (nodes released)")
        entries = [e for e in entries if e.get("state") != "done"]
        # stale notes age out: they never hold nodes, but ancient notes
        # pollute every listing (the 19h-old notes class)
        fresh = []
        for e in entries:
            if e.get("kind") == "note":
                try:
                    age_h = (time.time() - time.mktime(time.strptime(
                        e.get("submitted_at", now()), "%Y-%m-%dT%H:%M:%S"))) / 3600.0
                except ValueError:
                    age_h = 0.0
                if age_h > 24.0:
                    e["state"] = "done"
                    append_result(e, 0, f"note aged out at {age_h:.0f}h")
                    continue
            fresh.append(e)
        entries = fresh
        candidates = [e for e in entries if e.get("state") == "queued"
                      and e.get("cmd")
                      and not any(a not in results_ids for a in e.get("after", []))]
        # Operator policy: equal priority -> longest-waiting wins (FCFS in
        # class); any task waiting over AGE_ELEVATE_MINUTES is elevated to
        # the highest priority (anti-starvation aging). Effective priority
        # drives both the sort and the priority barrier.
        def eff_priority(e):
            try:
                age_min = (datetime.utcnow() - datetime.strptime(
                    e["submitted_at"], "%Y-%m-%dT%H:%M:%S")).total_seconds() / 60.0
            except Exception:
                age_min = 0.0
            return 0 if age_min > 120.0 else e["priority"]
        candidates.sort(key=lambda e: (eff_priority(e), e["submitted_at"]))
        task = None
        for cand in candidates:
            rc = cand.get("resources", "gpu")
            if not nodes_free(cand["nodes"], res, rc):
                print(f"blocked: {cand['id']} nodes busy")
                continue
            held_by = next((o for o in candidates
                if o["id"] != cand["id"] and eff_priority(o) < eff_priority(cand)
                and cand["id"] not in o.get("after", [])
                and rc == "gpu" and o.get("resources", "gpu") == "gpu"
                and set(cand["nodes"]).intersection(o["nodes"])), None)
            if held_by is not None:
                print(f"held: {cand['id']} — priority barrier for "
                      f"{held_by['id']} (p{held_by['priority']})")
                continue
            task = cand
            break
        if task is None:
            rewrite_queue(entries); save_reservations(res)
            print("nothing dispatched this pass")
            return
        ttl = float(task.get("ttl_minutes") or args.ttl)
        for n in task["nodes"]:
            res[n] = dict(id=task["id"], holder=f"task:{task['id']}",
                acquired_at=now(), ttl_minutes=ttl,
                resources=task.get("resources", "gpu"), pid=None)
        n0 = task["nodes"][0]
        inner = ("echo $$ > /tmp/sparkqueue-" + task['id'] + ".pid; "
                 + task['cmd']
                 + "; echo \"$?\" > /tmp/sparkqueue-" + task['id'] + ".exit")
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


def kill_remote_task(entry, res):
    """TERM the remote process of a running task (best effort) so cancel/
    done cannot orphan a live process on a node we are about to release.
    The queue's own cleanup - the denylist governs submitted commands."""
    n0 = entry["nodes"][0]
    pid = res.get(n0, {}).get("pid")
    if not pid or not str(pid).isdigit():
        _, pid_txt = ssh(n0,
            f"cat /tmp/sparkqueue-{entry['id']}.pid 2>/dev/null", timeout=10)
        pid = pid_txt.strip() if pid_txt.strip().isdigit() else None
    if pid:
        # The wrapper is setsid'd: TERM the PROCESS GROUP or children
        # (sleep, benchmarks) survive the wrapper's death as orphans.
        rc, _ = ssh(n0, f"kill -- -{pid} {pid} 2>/dev/null", timeout=10)
        return rc == 0
    return False

def cmd_cancel(args):
    with acquire_lock():
        entries = load_queue()
        res = load_reservations()
        kept = []
        for e in entries:
            if e["id"] == args.id:
                if e.get("state") == "running":
                    kill_remote_task(e, res)
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
    with acquire_lock():
        res = load_reservations()
        if args.node in res:
            sys.exit(f"{args.node} already reserved: {res[args.node]}")
        res[args.node] = dict(id=f"manual:{args.holder}", holder=args.holder,
            acquired_at=now(), ttl_minutes=args.ttl_min,
            resources=getattr(args, "resources", "gpu") or "gpu")
        save_reservations(res)
        print(f"reserved {args.node} for {args.holder} ttl={args.ttl_min}m")


def cmd_release(args):
    with acquire_lock():
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
    """Legacy name kept for the sweep: ONE dispatch pass (same semantics
    as dispatch - the old wall-clock launch path is gone)."""
    cmd_dispatch(args)

def _legacy_schedule_unused(args):
    with acquire_lock():
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
    a.add_argument("--cmd-file", help="read the command from this file "
        "(avoids nested-quoting footguns for long scripts)")
    a.add_argument("--resources", default="gpu", choices=["gpu", "cpu"],
        help="gpu (default): exclusive node claim. cpu: disk/CPU-only work - "
             "coexists with gpu tasks and gpu holds on the same nodes "
             "(pack builds, sha sweeps, log pulls); still exclusive "
             "against OTHER cpu tasks on the same nodes.")
    a.add_argument("--cwd")
    a.add_argument("--priority", type=int, default=5)
    a.add_argument("--kind", default="run", choices=["run", "gate", "note"])
    a.add_argument("--after", help="comma list of ids this waits for")
    a.add_argument("--klass", default="long", choices=["short", "long"])
    a.add_argument("--by", default="coordinator")
    a.add_argument("--notes", default="")
    a.add_argument("--ttl-min", type=float, default=None,
        help="expected duration minutes; the dispatch lease holds exactly "
             "this long (default 15 if undeclared)")
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
    a.add_argument("--ttl", type=int, default=15,
        help="lease minutes held for the TASK (not the lane); a task's "
             "own --ttl at submit time overrides this default")
    a.set_defaults(fn=cmd_dispatch)
    a = sub.add_parser("cancel")
    a.add_argument("--id", required=True)
    a.set_defaults(fn=cmd_cancel)
    a = sub.add_parser("reserve")
    a.add_argument("--node", required=True)
    a.add_argument("--holder", required=True)
    a.add_argument("--ttl-min", type=float, default=360.0)
    a.add_argument("--resources", default="gpu", choices=["gpu", "cpu"])
    a.set_defaults(fn=cmd_reserve)
    a = sub.add_parser("release")
    a.add_argument("--node")
    a.add_argument("--id")
    a.set_defaults(fn=cmd_release)
    a = sub.add_parser("doctor")
    a.set_defaults(fn=cmd_doctor)
    a = sub.add_parser("schedule")
    a.set_defaults(fn=cmd_schedule)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
