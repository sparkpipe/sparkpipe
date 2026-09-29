#!/usr/bin/env python3
"""List host-test fixture processes that outlived their test (docs/TEST_FIXTURE_PROCESSES.md)."""
import argparse
import os
import sys

MARKER = "SPARK_TEST_FIXTURE_OWNER"
SERVING_BINARIES = {
    "sparkpipe_model_residentd",
    "sparkpipe_model_api",
    "sparkpipe_model_batch",
    "sparkpipe_weightd",
    "sparkpipe_weightsd",
}
DELETED = " (deleted)"


def read_bytes(path):
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def read_link(path):
    try:
        return os.readlink(path)
    except OSError:
        return None


def stat_fields(proc, pid):
    raw = read_bytes(os.path.join(proc, str(pid), "stat"))
    if raw is None:
        return None
    text = raw.decode("utf-8", "replace")
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2:].split()
    if len(fields) < 20:
        return None
    return {"state": fields[0], "ppid": int(fields[1]), "start": fields[19]}


def owner_alive(proc, owner):
    pid_text, _, start = owner.partition(".")
    if not pid_text.isdigit():
        return False
    fields = stat_fields(proc, int(pid_text))
    if fields is None or fields["state"] in ("Z", "X"):
        return False
    return start in ("", "0") or fields["start"] == start


def age_seconds(proc, start):
    uptime = read_bytes(os.path.join(proc, "uptime"))
    if uptime is None:
        return -1
    try:
        ticks = os.sysconf("SC_CLK_TCK")
        return int(float(uptime.split()[0]) - int(start) / ticks)
    except (ValueError, OSError):
        return -1


def environment(raw):
    values = {}
    for item in (raw or b"").split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep:
            values[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return values


def test_parent(proc, ppid):
    exe = read_link(os.path.join(proc, str(ppid), "exe")) or ""
    return os.path.basename(exe.removesuffix(DELETED)).startswith("test_")


def classify(proc, pid, fields, env, argv, exe, cwd):
    owner = env.get(MARKER)
    if owner is not None:
        if owner.partition(".")[0] == str(pid):
            return None
        if owner_alive(proc, owner):
            return ("RUNNING", "owner %s alive" % owner)
        return ("STRAY", "owner %s gone" % owner)
    name = os.path.basename((exe or (argv[0] if argv else "")).removesuffix(DELETED))
    if name not in SERVING_BINARIES:
        return None
    reasons = []
    if cwd is not None and cwd.endswith(DELETED):
        reasons.append("cwd deleted")
    if any(arg.startswith("/tmp/sparkpipe-") for arg in argv):
        reasons.append("tmp fixture deployment")
    if not reasons:
        return None
    if test_parent(proc, fields["ppid"]):
        return ("RUNNING", "test parent %d alive; " % fields["ppid"] + ", ".join(reasons))
    return ("STRAY", "unmarked; " + ", ".join(reasons))


def scan(proc, under=None):
    rows = []
    me = os.getpid()
    for name in sorted(os.listdir(proc), key=lambda item: (len(item), item)):
        if not name.isdigit() or int(name) == me:
            continue
        pid = int(name)
        base = os.path.join(proc, name)
        fields = stat_fields(proc, pid)
        if fields is None or fields["state"] in ("Z", "X"):
            continue
        env = environment(read_bytes(os.path.join(base, "environ")))
        argv = [a.decode("utf-8", "replace") for a in (read_bytes(os.path.join(base, "cmdline")) or b"").split(b"\0") if a]
        exe = read_link(os.path.join(base, "exe"))
        cwd = read_link(os.path.join(base, "cwd"))
        verdict = classify(proc, pid, fields, env, argv, exe, cwd)
        if verdict is None:
            continue
        if under is not None:
            paths = [p.removesuffix(DELETED) for p in (exe, cwd) if p]
            if not any(p == under or p.startswith(under.rstrip("/") + "/") for p in paths):
                continue
        rows.append({
            "kind": verdict[0], "pid": pid, "ppid": fields["ppid"],
            "age_s": age_seconds(proc, fields["start"]), "reason": verdict[1],
            "exe": exe or "?", "cwd": cwd or "?", "cmd": " ".join(argv)[:200],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--proc", default="/proc")
    parser.add_argument("--under", help="only processes whose exe or cwd is under this directory")
    parser.add_argument("--all", action="store_true", help="also list fixtures whose test is still running")
    args = parser.parse_args()
    if not os.path.isdir(os.path.join(args.proc, "self")) and args.proc == "/proc":
        print("STRAY-FIXTURES unavailable: no /proc on this host")
        return 2
    under = os.path.normpath(os.path.abspath(args.under)) if args.under else None
    rows = scan(args.proc, under)
    strays = [row for row in rows if row["kind"] == "STRAY"]
    for row in rows:
        if row["kind"] == "STRAY" or args.all:
            print("{kind} pid={pid} ppid={ppid} age_s={age_s} reason=[{reason}] exe={exe} cwd={cwd} cmd={cmd}".format(**row))
    print("STRAY-FIXTURES n=%d running=%d%s" % (
        len(strays), len(rows) - len(strays),
        (" stop them with: kill -KILL " + " ".join(str(row["pid"]) for row in strays)) if strays else ""))
    return 1 if strays else 0


if __name__ == "__main__":
    sys.exit(main())
