#!/usr/bin/env python3
import argparse
import fcntl
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time

SNAPSHOT_SUFFIX = ".kvs"
STATE_SUFFIX = ".kvst"
TEMPORARY_PREFIX = ".kvs-writing-"
LOCK_NAME = ".kvs-lock"
PURGE_DEPTH_MAX = 16
WORKING_STATE = "sparkdata/working-state"
SCRATCH = WORKING_STATE + "/scratch"
MIRROR = WORKING_STATE + "/mirror"
LAST_USE = ".last-use"
MIRROR_DEPTH = MIRROR.count("/") + 2
STORE_DEPTH_MAX = PURGE_DEPTH_MAX - MIRROR_DEPTH
DIRECT_PREFIX = "10.10.200."
RSH = "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5"
RSYNC_TIMEOUT_SECONDS = 300
RSYNC_VANISHED = 24
INTERVAL_SECONDS = 30
EXCLUDES = ["- " + TEMPORARY_PREFIX + "*", "- " + LOCK_NAME]
OUT_FORMAT = "--out-format=%i %n"
ITEM = re.compile(r"(\*deleting|[<>ch.][fdLDS]\S*) +(.*)")
ESCAPED = re.compile(r"\\#([0-7]{3})")
WILDCARD = re.compile(r"([\\*?\[])")
LINK = re.compile(r"(\d+)-(\d+):(\d+)")
RUNNING = {"child": None}


class ConfigError(Exception):
    pass


def log(message):
    print(time.strftime("%H:%M:%S") + " working-state: " + message, flush=True)


def is_cache_name(name):
    return ((len(name) > len(SNAPSHOT_SUFFIX) and name.endswith(SNAPSHOT_SUFFIX))
            or (len(name) > len(STATE_SUFFIX) and name.endswith(STATE_SUFFIX))
            or name.startswith(TEMPORARY_PREFIX))


def is_snapshot_name(name):
    return is_cache_name(name) and not name.startswith(TEMPORARY_PREFIX)


def partner_rank(rank, link_map):
    for link in link_map.split(","):
        match = LINK.fullmatch(link)
        if not match or int(match[1]) >= int(match[2]) or int(match[1]) ^ 1 != int(match[2]):
            raise ConfigError(f"mesh link '{link}' is not LOW-HIGH:GBPS joining a rank with its rank^1 partner")
        if rank in (int(match[1]), int(match[2])):
            return rank ^ 1
    return None


def store_depth(store):
    return store.count("/") + 1 if store else 0


def find_stores(home):
    info = os.lstat(home)
    if not stat.S_ISDIR(info.st_mode):
        return [], [f"{home} is not a directory"]
    device = info.st_dev
    excluded = {os.path.join(home, MIRROR), os.path.join(home, SCRATCH)}
    stores, errors = [], []

    def walk(directory, depth):
        try:
            with os.scandir(directory) as listing:
                entries = list(listing)
        except (PermissionError, FileNotFoundError):
            return
        except OSError as error:
            errors.append(f"{directory}: {error.strerror}")
            return
        holds = False
        for entry in entries:
            if is_cache_name(entry.name) or entry.name == LOCK_NAME:
                holds = True
                continue
            if depth >= PURGE_DEPTH_MAX or entry.path in excluded:
                continue
            try:
                if not entry.is_dir(follow_symlinks=False) or entry.stat(follow_symlinks=False).st_dev != device:
                    continue
            except OSError:
                continue
            walk(entry.path, depth + 1)
        if holds:
            stores.append(directory[len(home) + 1:])

    walk(home, 0)
    return sorted(stores), errors


def pattern(path, wildcard=False):
    if wildcard or any(character in path for character in "*?["):
        return WILDCARD.sub(r"\\\1", path)
    return path


def store_rules(stores):
    rules, seen = [], set()

    def include(path):
        parts = path.split("/") if path else []
        for count in range(1, len(parts) + 1):
            chain = "/".join(parts[:count])
            if chain not in seen:
                seen.add(chain)
                rules.append(f"+ /{pattern(chain)}/")

    include(WORKING_STATE)
    for store in stores:
        include(store)
        base = f"/{pattern(store, True)}/" if store else "/"
        rules.extend((f"+ {base}*{SNAPSHOT_SUFFIX}", f"+ {base}*{STATE_SUFFIX}"))
    return rules


def rsync(arguments, rules=()):
    with tempfile.NamedTemporaryFile("w", prefix="working-state-", suffix=".filter", errors="surrogateescape") as rules_file:
        rules_file.writelines(rule + "\n" for rule in rules)
        rules_file.flush()
        command = ["rsync", f"--timeout={RSYNC_TIMEOUT_SECONDS}", "--8-bit-output"]
        if rules:
            command.append("--filter=merge " + rules_file.name)
        child = subprocess.Popen(command + arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, errors="surrogateescape")
        RUNNING["child"] = child
        try:
            out, err = child.communicate()
        finally:
            RUNNING["child"] = None
    return child.returncode, out, err


def failure(what, status, err):
    if status in (0, RSYNC_VANISHED):
        return ""
    lines = [line.strip() for line in err.splitlines() if line.strip()]
    return f"{what}: rsync exit {status}: {lines[0] if lines else 'no message'}"


def tally(out):
    sent = touched = deleted = 0
    for line in out.splitlines():
        match = ITEM.fullmatch(line)
        if not match:
            continue
        item, name = match.groups()
        if item == "*deleting":
            deleted += not name.endswith("/")
        elif item[1] == "f" and item[0] in "<>":
            sent += 1
        elif item[1] == "f":
            touched += 1
    return sent, touched, deleted


def summary(failures):
    if not failures:
        return "0 failures"
    return f"{len(failures)} failure{'s' if len(failures) > 1 else ''}: " + "; ".join(failures)


class Node:
    def __init__(self, arguments):
        self.home = os.path.abspath(arguments.home)
        self.rank = arguments.rank
        self.partner = partner_rank(arguments.rank, arguments.link_map)
        self.host = arguments.partner or (f"{DIRECT_PREFIX}{self.partner}" if self.partner is not None else "")
        self.rsh = "--rsh=" + arguments.rsh
        self.remote = f"{self.host}:{MIRROR}/{self.rank:x}/"

    def path(self, relative):
        return os.path.join(self.home, relative) if relative else self.home

    def initialized(self):
        return os.path.isdir(self.path(MIRROR))

    def unpaired(self):
        log(f"rank {self.rank} has no cabled pair in the mesh link map; its working state is not mirrored")
        return 0


def make_directory(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isdir(path):
        os.mkdir(path, 0o700)


def init(home):
    make_directory(os.path.join(home, WORKING_STATE))
    make_directory(os.path.join(home, SCRATCH))
    stamp = os.path.join(home, SCRATCH, LAST_USE)
    if not os.path.lexists(stamp):
        os.close(os.open(stamp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600))
    make_directory(os.path.join(home, MIRROR))


def mirrorable(stores):
    accepted, refused = [], []
    for store in stores:
        if "\n" in store or "\r" in store:
            refused.append(f"store {store!r} has a line break in its path and cannot be named in a filter")
        elif store_depth(store) > STORE_DEPTH_MAX:
            refused.append(f"store {store} is deeper than {STORE_DEPTH_MAX} below home, past the partner purge's reach once mirrored")
        else:
            accepted.append(store)
    return accepted, refused


def push(node):
    if node.partner is None:
        return node.unpaired()
    if not node.initialized():
        log(f"push refused: {node.path(MIRROR)} is missing, so this disk was never rebuilt; run rebuild first")
        return 1
    started = time.monotonic()
    failures, stores = [], []
    found, errors = find_stores(node.home)
    scratch = os.path.isdir(node.path(SCRATCH))
    store_counts = scratch_counts = (0, 0, 0)
    if errors:
        failures.append(f"store walk failed, stores not pushed: {errors[0]}" + (f" and {len(errors) - 1} more" if len(errors) > 1 else ""))
    else:
        stores, refused = mirrorable(found)
        failures += refused
        rules = ([f"P /{SCRATCH}/"] if scratch else []) + EXCLUDES + store_rules(stores) + ["- *"]
        status, out, err = rsync(["--archive", "--delete", "--delete-excluded", "--size-only", OUT_FORMAT, node.rsh,
                                  node.home + "/", node.remote], rules)
        store_counts = tally(out)
        failures.append(failure("stores", status, err))
    if scratch:
        status, out, err = rsync(["--archive", "--delete", OUT_FORMAT, node.rsh,
                                  node.path(SCRATCH) + "/", node.remote + SCRATCH + "/"], EXCLUDES)
        scratch_counts = tally(out)
        failures.append(failure("scratch", status, err))
    failures = [problem for problem in failures if problem]
    log(f"push to {node.host}: {len(stores)} stores, {store_counts[0]} sent, {store_counts[1]} touched, "
        f"{store_counts[2]} deleted; scratch " + (f"{scratch_counts[0]} sent, {scratch_counts[2]} deleted" if scratch else "absent")
        + f"; {summary(failures)} ({time.monotonic() - started:.1f}s)")
    return 1 if failures else 0


def unescape(name):
    return ESCAPED.sub(lambda match: chr(int(match[1], 8)), name)


def mirror_listing(node):
    root = WORKING_STATE.split("/")[0]
    inner = f"{MIRROR[len(root) + 1:]}/{node.rank:x}"
    parts = inner.split("/")
    rules = [f"+ /{'/'.join(parts[:count])}/" for count in range(1, len(parts))] + [f"+ /{inner}/***", "- *"]
    status, out, err = rsync(["--recursive", "--list-only", node.rsh, f"{node.host}:{root}/"], rules)
    problem = failure("listing", status, err)
    if problem:
        return None, problem
    entries, found = [], False
    for line in out.splitlines():
        fields = line.split(None, 4)
        if len(fields) < 5:
            continue
        name = unescape(fields[4].split(" -> ", 1)[0] if fields[0].startswith("l") else fields[4])
        if name == inner:
            found = True
        elif name.startswith(inner + "/"):
            entries.append((fields[0][0], name[len(inner) + 1:]))
    return (entries if found else []), None


def in_scratch(relative):
    return relative == SCRATCH or relative.startswith(SCRATCH + "/")


def lock_store(directory):
    os.makedirs(directory, exist_ok=True)
    descriptor = os.open(os.path.join(directory, LOCK_NAME), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def rebuild(node):
    if node.partner is None:
        return node.unpaired()
    entries, problem = mirror_listing(node)
    if problem:
        log(f"rebuild from {node.host} failed: {problem}; the area stays uninitialized")
        return 1
    if not entries:
        init(node.home)
        log(f"rebuild from {node.host}: rank {node.partner} holds no mirror of rank {node.rank}; nothing restored, area ready")
        return 0
    stores = sorted({os.path.dirname(name) for kind, name in entries
                     if kind == "-" and is_snapshot_name(os.path.basename(name)) and not in_scratch(name)})
    scratch = ("d", SCRATCH) in entries
    locks, busy, failures = {}, [], []
    try:
        for store in stores:
            try:
                locks[store] = lock_store(node.path(store))
            except BlockingIOError:
                busy.append(store)
            except OSError as error:
                failures.append(f"store {store}: {error.strerror}")
        rules = EXCLUDES + store_rules(list(locks)) + ([f"+ /{SCRATCH}/***"] if scratch else []) + ["- *"]
        status, out, err = rsync(["--archive", "--ignore-existing", "--keep-dirlinks", OUT_FORMAT, node.rsh,
                                  node.remote, node.home + "/"], rules)
    finally:
        for descriptor in locks.values():
            os.close(descriptor)
    problem = failure("restore", status, err)
    if problem:
        failures.append(problem)
    else:
        init(node.home)
    failures += [f"store {store} is in use (its {LOCK_NAME} is held), not restored" for store in busy]
    log(f"rebuild from {node.host}: {tally(out)[0]} files restored into {len(locks)} stores"
        + (" and scratch" if scratch else "") + f"; {summary(failures)}"
        + ("" if problem else "; area ready"))
    return 1 if failures else 0


def alive(pid):
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def loop(node, interval, agent):
    if node.partner is None:
        return node.unpaired()
    while alive(agent):
        started = time.monotonic()
        try:
            if node.initialized():
                push(node)
            else:
                rebuild(node)
        except OSError as error:
            log(f"pass failed: {error}; retried next pass")
        while alive(agent) and time.monotonic() - started < interval:
            time.sleep(max(0.0, min(1.0, interval - (time.monotonic() - started))))
    return 0


def stop(signum, frame):
    child = RUNNING["child"]
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(10)
        except subprocess.TimeoutExpired:
            child.kill()
    sys.exit(128 + signum)


def rank_argument(text):
    if not re.fullmatch(r"\d{1,2}", text) or int(text) > 15:
        raise argparse.ArgumentTypeError(f"rank '{text}' is not 0-15")
    return int(text)


def parser():
    top = argparse.ArgumentParser(prog="working_state_mirror.py")
    commands = top.add_subparsers(dest="command", required=True)
    commands.add_parser("init").add_argument("--home", default=os.path.expanduser("~"))
    for name in ("push", "loop", "rebuild"):
        command = commands.add_parser(name)
        command.add_argument("--home", default=os.path.expanduser("~"))
        command.add_argument("--rank", type=rank_argument, required=True)
        command.add_argument("--link-map", required=True)
        command.add_argument("--partner")
        command.add_argument("--rsh", default=RSH)
        if name == "loop":
            command.add_argument("--interval", type=float, default=INTERVAL_SECONDS)
            command.add_argument("--agent-pid", type=int)
    return top


def main(argv):
    arguments = parser().parse_args(argv)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    if arguments.command == "init":
        init(os.path.abspath(arguments.home))
        return 0
    try:
        node = Node(arguments)
    except ConfigError as error:
        print(f"working_state_mirror.py: {error}", file=sys.stderr)
        return 2
    if arguments.command == "loop":
        return loop(node, arguments.interval, arguments.agent_pid)
    return push(node) if arguments.command == "push" else rebuild(node)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
