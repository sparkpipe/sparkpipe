#!/usr/bin/env python3
import argparse
import concurrent.futures
import datetime
import os
import re
import shlex
import subprocess
import sys
import time

KIT = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(KIT, "node_probe.sh")
COLUMNS = ["agent", "hold", "layout", "applied", "rootok", "eng_n", "exe", "up", "drv", "ready", "errsite", "wd", "wd_up", "wd_n", "wd_other", "wd_inst", "verify", "stage_wd", "stage_rc", "stage_warm", "receipt", "mem_gib", "others", "roots"]
REQUIRED_ENV = ("RELEASE_NODES", "RELEASE_ROOT_NAME", "RELEASE_RANK_PACK", "RELEASE_DRIVER", "RELEASE_WEIGHTD_STAGE", "RELEASE_PROBE_EXTRA")
PROBE_TIMEOUT_S = 90


class ConfigError(Exception):
    pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%SZ")


def load_env(env):
    missing = [k for k in REQUIRED_ENV if not env.get(k)]
    if missing:
        raise ConfigError("missing " + " ".join(missing) + " (source the release env through tools/fleet_release/lib.sh)")
    nodes = env["RELEASE_NODES"].split()
    if len(set(nodes)) != len(nodes):
        raise ConfigError("RELEASE_NODES lists a node twice")
    if "@hex@" not in env["RELEASE_RANK_PACK"]:
        raise ConfigError("RELEASE_RANK_PACK must contain @hex@ (the node's rank in RELEASE_NODES, lowercase hex)")
    extra = env["RELEASE_PROBE_EXTRA"]
    if extra != "none" and not os.path.isfile(extra):
        raise ConfigError(f"RELEASE_PROBE_EXTRA {extra} is not a file (use none for no extra fields)")
    return {
        "nodes": nodes,
        "root": env["RELEASE_ROOT_NAME"],
        "pack": env["RELEASE_RANK_PACK"],
        "driver": env["RELEASE_DRIVER"],
        "stage": env["RELEASE_WEIGHTD_STAGE"],
        "extra": None if extra == "none" else extra,
        "ssh": shlex.split(env.get("RELEASE_SSH") or "ssh -o BatchMode=yes -o ConnectTimeout=10"),
    }


def probe_script(cfg, rank):
    head = "".join(f"{k}={shlex.quote(v)}\n" for k, v in (
        ("ROOT_NAME", cfg["root"]),
        ("RANK_PACK", cfg["pack"].replace("@hex@", format(rank, "x"))),
        ("DRIVER", cfg["driver"]),
        ("WEIGHTD_STAGE", cfg["stage"]),
    ))
    body = ""
    if cfg["extra"]:
        with open(cfg["extra"]) as f:
            body += f.read() + "\n"
    with open(PROBE) as f:
        body += f.read()
    return head + body


def parse_probe(host, out):
    line = [l for l in out.splitlines() if l.startswith("host=")]
    if not line:
        return {"host": host, "error": "no-probe-output"}
    row = {"host": host}
    row.update(dict(re.findall(r"(\w+)=(\S*)", line[-1])))
    row["host"] = host
    return row


def probe(cfg, host):
    rank = cfg["nodes"].index(host)
    try:
        out = subprocess.run(cfg["ssh"] + [host, "bash -s"], input=probe_script(cfg, rank).encode(),
                             capture_output=True, timeout=PROBE_TIMEOUT_S).stdout.decode(errors="replace")
    except subprocess.TimeoutExpired:
        return {"host": host, "error": "ssh-timeout"}
    return parse_probe(host, out)


def parse_expect(items):
    rules = []
    for item in items:
        m = re.fullmatch(r"(\w+)(>=|<=|!=|=)(.*)", item)
        if not m:
            raise ConfigError(f"bad --expect {item}")
        rules.append(m.groups())
    return rules


def violations(row, rules):
    if "error" in row:
        return [row["error"]]
    bad = []
    for key, op, want in rules:
        got = row.get(key)
        if got is None:
            bad.append(f"{key} missing from the probe")
        elif op == "=" and got != want:
            bad.append(f"{key}={got} want {want}")
        elif op == "!=" and got == want:
            bad.append(f"{key}={got} must differ")
        elif op in (">=", "<="):
            try:
                ok = int(got) >= int(want) if op == ">=" else int(got) <= int(want)
            except ValueError:
                ok = False
            if not ok:
                bad.append(f"{key}={got} want {op}{want}")
    return bad


def columns(rows):
    cols = list(COLUMNS)
    for r in rows:
        for k in r:
            if k not in cols and k not in ("host", "error"):
                cols.append(k)
    return cols


def render(rows, bad, width):
    cols = columns(rows)
    lines = []
    for r in rows:
        cells = " ".join(f"{c}={r.get(c, '?')}" for c in cols) if "error" not in r else r["error"]
        mark = "OK  " if not bad[r["host"]] else "BAD "
        lines.append(f"{mark}{r['host']:{width}s} {cells}" + (f"  <- {'; '.join(bad[r['host']])}" if bad[r["host"]] else ""))
    return lines


def gate(cfg, rules, wait_min, interval, resample, resample_interval, quiet, log, probe_fn=None, sleep=time.sleep, clock=time.time, out=print):
    probe_fn = probe_fn or (lambda h: probe(cfg, h))
    nodes = cfg["nodes"]
    width = max(len(h) for h in nodes)
    deadline = clock() + wait_min * 60
    start = clock()

    def write(text):
        out(text)
        if log:
            with open(log, "a") as f:
                f.write(text + "\n")

    while True:
        with concurrent.futures.ThreadPoolExecutor(len(nodes)) as pool:
            rows = list(pool.map(probe_fn, nodes))
        bad = {r["host"]: violations(r, rules) for r in rows}
        final = clock() >= deadline
        for attempt in range(resample if final else 0):
            failing = [h for h in nodes if bad[h]]
            if not failing:
                break
            sleep(resample_interval)
            with concurrent.futures.ThreadPoolExecutor(len(failing)) as pool:
                again = {r["host"]: r for r in pool.map(probe_fn, failing)}
            for host in failing:
                now_bad = violations(again[host], rules)
                if not now_bad:
                    write(f"{now()} TRANSIENT {host}: {'; '.join(bad[host])} cleared on resample {attempt + 1}/{resample}")
                rows = [again[host] if r["host"] == host else r for r in rows]
                bad[host] = now_bad
        failing = [h for h in nodes if bad[h]]
        if not failing or final or not quiet:
            write("\n".join([f"{now()} elapsed={int(clock() - start)}s ok={len(nodes) - len(failing)}/{len(nodes)}"] + render(rows, bad, width)))
        else:
            out(f"{now()} ok={len(nodes) - len(failing)}/{len(nodes)} waiting on {' '.join(failing)}")
        if not failing:
            out(f"NODES-OK {len(nodes)}/{len(nodes)} expect: {' '.join(f'{k}{o}{v}' for k, o, v in rules) or '(none)'}")
            return 0
        if final:
            out(f"NODES-FAIL {len(failing)} node(s): {' '.join(failing)}")
            return 1
        sleep(interval)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Probe every release node once (or until --wait-min) and check --expect rules.")
    ap.add_argument("--expect", action="append", default=[], help="KEY=V, KEY!=V, KEY>=N or KEY<=N against a probe column")
    ap.add_argument("--wait-min", type=float, default=0)
    ap.add_argument("--interval", type=float, default=15)
    ap.add_argument("--quiet-poll", action="store_true")
    ap.add_argument("--log")
    ap.add_argument("--resample", type=int, default=2, help="re-probe failing nodes this many times before the final verdict")
    ap.add_argument("--resample-interval", type=float, default=5)
    args = ap.parse_args(argv)
    try:
        cfg = load_env(os.environ)
        rules = parse_expect(args.expect)
    except ConfigError as e:
        print(f"nodes.py: {e}", file=sys.stderr)
        return 2
    return gate(cfg, rules, args.wait_min, args.interval, args.resample, args.resample_interval, args.quiet_poll, args.log)


if __name__ == "__main__":
    sys.exit(main())
