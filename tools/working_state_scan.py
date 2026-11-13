#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import socket
import sqlite3
import stat
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULE_PATHS = (HERE / "serving" / "working_state.py", HERE / "working_state.py")
WORKING_STATE_ROOT = Path("sparkdata") / "working-state"
CACHE_SUFFIXES = (".kvs", ".kvst")
TEMPORARY_PREFIX = ".kvs-writing-"
LOCK_NAME = ".kvs-lock"
LAST_USE_NAME = ".last-use"
WALK_DEPTH_MAX = 17
LEDGER_NAME = ".sparkpipe-kv-ledger"
LEDGER_HEADER = struct.Struct("<QII Q 5Q")
LEDGER_SLOT = struct.Struct("<IIQQQQ512s")
LEDGER_MAGIC = 0x31524744454C564B
LEDGER_KIND_SNAPSHOT, LEDGER_KIND_BACKING = 1, 2
INDEX_HEADER = struct.Struct("<QIIIIII32s32s32s")
INDEX_RECORD = struct.Struct("<IIQQQ32sII")
INDEX_MAGIC = 0x3158444950534B53
REPLAY_TABLES = ("turns", "responses", "aliases")
CACHE_KINDS = ("snapshot", "mirror", "prefix_index", "replay_store", "tap_records")
STORAGE_KINDS = ("scratch", "mirror_scratch")
DEFAULT_GRACE_SECONDS = 120


def load_rule():
    for path in RULE_PATHS:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("sparkpipe_working_state", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SystemExit(f"working_state_scan: the working-state rule is not next to this tool ({', '.join(str(path) for path in RULE_PATHS)})")


rule = load_rule()


def cache_name(name):
    return name.endswith(CACHE_SUFFIXES) or name.startswith(TEMPORARY_PREFIX)


def walk(root, device, depth=0, skip=()):
    try:
        entries = list(os.scandir(root))
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        return
    for entry in entries:
        path = Path(entry.path)
        if path in skip:
            continue
        try:
            info = entry.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(info.st_mode):
            if info.st_dev == device and depth + 1 < WALK_DEPTH_MAX:
                yield from walk(path, device, depth + 1, skip)
        elif stat.S_ISREG(info.st_mode):
            yield path, info


def store_of(path):
    parent = path.parent
    if len(parent.name) == 2 and all(character in "0123456789abcdef" for character in parent.name) and (parent.parent / LOCK_NAME).exists():
        return parent.parent
    return parent


class Place:
    def __init__(self, kind, path, account=None, working_state=True):
        self.kind, self.path, self.account, self.working_state = kind, str(path), account, working_state
        self.items = self.bytes = self.selected_items = self.selected_bytes = self.expired_items = self.expired_bytes = 0
        self.oldest = self.newest = None
        self.unattributable = None

    def add(self, used, size, selected, expired):
        self.items += 1
        self.bytes += size
        self.oldest = used if self.oldest is None else min(self.oldest, used)
        self.newest = used if self.newest is None else max(self.newest, used)
        if selected:
            self.selected_items += 1
            self.selected_bytes += size
        if expired:
            self.expired_items += 1
            self.expired_bytes += size

    def report(self):
        entry = {"kind": self.kind, "path": self.path, "items": self.items, "bytes": self.bytes, "oldest_used": self.oldest, "newest_used": self.newest,
                 "selected_items": self.selected_items, "selected_bytes": self.selected_bytes, "expired_items": self.expired_items,
                 "expired_bytes": self.expired_bytes, "working_state": self.working_state}
        if self.account is not None:
            entry["account"] = self.account
        if self.unattributable is not None:
            entry["unattributable"] = self.unattributable
        return entry


class Scan:
    def __init__(self, arguments):
        self.now = arguments.now if arguments.now is not None else time.time()
        self.grace = arguments.grace
        self.home = Path(arguments.home).resolve() if arguments.home else None
        directive = arguments.directive or (self.home / WORKING_STATE_ROOT / "directive" if self.home else None)
        self.directive_path = str(directive) if directive else None
        self.wiped, self.held_until, self.storage_wiped = rule.read_directive(directive) if directive else (0, 0, 0)
        self.account = arguments.account
        self.everything = arguments.all
        self.before = arguments.before
        self.storage_before = arguments.storage_before
        self.tenures = self.load_tenures(arguments.controller_state) if arguments.controller_state else None
        if self.account is not None and not rule.valid_account(self.account):
            raise SystemExit(f"working_state_scan: --account {self.account!r} is not an account id")
        self.places = []

    @staticmethod
    def load_tenures(path):
        state = json.loads(Path(path).read_text())
        tenures, current = [], None
        for event in state.get("history", []):
            if event.get("kind") == "handover" and event.get("from") != event.get("to") and isinstance(event.get("wiped"), int):
                if current is not None:
                    tenures.append((current[0], current[1], event["wiped"]))
                current = (event.get("to"), event["wiped"])
        if state.get("user") is not None:
            if current is not None and current[0] != state["user"]:
                tenures.append((current[0], current[1], state.get("since", 0)))
                current = None
            tenures.append((state["user"], current[1] if current is not None else state.get("since", 0), None))
        elif current is not None:
            tenures.append((current[0], current[1], None))
        return tenures

    def owned(self, used):
        if self.tenures is None or self.account is None:
            return False
        return any(account == self.account and start <= used and (end is None or used < end) for account, start, end in self.tenures)

    def chosen(self, kind, used, account=None):
        if self.everything:
            return True
        storage = kind in STORAGE_KINDS
        before = self.storage_before if storage else self.before
        if before is not None and used < before:
            return True
        if self.account is not None:
            if account is not None:
                return account == self.account
            return self.owned(used)
        return False

    def expired(self, kind, used):
        at = self.now - self.grace
        if kind in STORAGE_KINDS:
            return rule.storage_expired(used, self.storage_wiped, self.held_until, at)
        return rule.expired(used, self.wiped, self.held_until, at)

    def place(self, kind, path, account=None, working_state=True):
        found = Place(kind, path, account, working_state)
        self.places.append(found)
        return found

    def item(self, place, used, size, account=None):
        place.add(used, size, place.working_state and self.chosen(place.kind, used, account), place.working_state and self.expired(place.kind, used))

    def node(self):
        if self.home is None:
            return
        root = self.home / WORKING_STATE_ROOT
        mirror = root / "mirror"
        scratch = root / "scratch"
        device = os.lstat(self.home).st_dev
        stores = {}
        for path, info in walk(self.home, device, skip=(mirror, scratch)):
            if cache_name(path.name):
                directory = store_of(path)
                if directory not in stores:
                    stores[directory] = self.place("snapshot", directory)
                self.item(stores[directory], info.st_mtime, info.st_size)
            elif path.name.startswith("prefix_index") and path.suffix == ".spi":
                self.prefix_index(path)
        self.scratch_area("scratch", scratch, device)
        if mirror.is_dir():
            for rank in sorted(os.listdir(mirror)):
                if rank.startswith("."):
                    continue
                copies = {}
                area = mirror / rank / WORKING_STATE_ROOT / "scratch"
                for path, info in walk(mirror / rank, device, skip=(area,)):
                    if cache_name(path.name):
                        directory = store_of(path)
                        if directory not in copies:
                            copies[directory] = self.place("mirror", directory)
                        self.item(copies[directory], info.st_mtime, info.st_size)
                self.scratch_area("mirror_scratch", area, device)
        self.ledger(self.home / LEDGER_NAME)

    def scratch_area(self, kind, area, device):
        try:
            info = os.lstat(area)
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(info.st_mode):
            return
        used = info.st_mtime
        try:
            stamp = os.lstat(area / LAST_USE_NAME)
            if stat.S_ISREG(stamp.st_mode):
                used = stamp.st_mtime
        except FileNotFoundError:
            pass
        found = self.place(kind, area)
        for path, entry in walk(area, device):
            if path.name != LAST_USE_NAME or path.parent != area:
                self.item(found, used, entry.st_size)

    def ledger(self, path):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return
        found = self.place("nvme_spill", path)
        found.unattributable = ("a running lane's spill backing is a nameless O_TMPFILE file; its records belong to the lane's page cache and go "
                                "with its entries, and the file goes when the lane stops")
        if len(data) < LEDGER_HEADER.size:
            found.unattributable = "the ledger is shorter than its header"
            return
        magic, version, slots = LEDGER_HEADER.unpack_from(data)[:3]
        if magic != LEDGER_MAGIC:
            found.unattributable = "the ledger's magic is wrong"
            return
        backings = []
        for index in range(min(slots, (len(data) - LEDGER_HEADER.size) // LEDGER_SLOT.size)):
            kind, pid, _, size, oldest, _, directory = LEDGER_SLOT.unpack_from(data, LEDGER_HEADER.size + index * LEDGER_SLOT.size)
            if kind == LEDGER_KIND_BACKING:
                backings.append({"directory": directory.split(b"\0", 1)[0].decode(errors="replace"), "pid": pid, "live": process_alive(pid), "bytes": size})
        found.items = len(backings)
        found.bytes = sum(entry["bytes"] for entry in backings if entry["live"])
        found.backings = backings

    def prefix_index(self, path):
        found = self.place("prefix_index", path)
        try:
            data = path.read_bytes()
        except OSError as error:
            found.unattributable = f"unreadable: {error}"
            return
        if len(data) < INDEX_HEADER.size or INDEX_HEADER.unpack_from(data)[0] != INDEX_MAGIC:
            found.unattributable = "not a prefix index file"
            return
        header = INDEX_HEADER.unpack_from(data)
        start, count, record_bytes = header[2], header[4], header[5]
        if record_bytes != INDEX_RECORD.size or start < INDEX_HEADER.size:
            found.unattributable = f"records of {record_bytes} bytes after a {start}-byte header, not {INDEX_RECORD.size} after {INDEX_HEADER.size}"
            return
        for index in range(min(count, max(0, len(data) - start) // INDEX_RECORD.size)):
            record = INDEX_RECORD.unpack_from(data, start + index * INDEX_RECORD.size)
            self.item(found, record[7], INDEX_RECORD.size)

    def replay(self, directory):
        for path in sorted(Path(directory).glob("*.sqlite")):
            stem = path.stem
            account = stem[len("account-"):] if stem.startswith("account-") else (rule.OWN_ACCOUNT if stem == "own" or stem.startswith("key-") else None)
            if account is None:
                continue
            found = self.place("replay_store", path, account)
            try:
                connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                try:
                    for table in REPLAY_TABLES:
                        for used, size in connection.execute(f"SELECT used, bytes FROM {table}"):
                            self.item(found, float(used), int(size or 0), account)
                finally:
                    connection.close()
            except sqlite3.Error as error:
                found.unattributable = f"unreadable: {error}"

    def records(self, directory):
        directory = Path(directory)
        if not directory.is_dir():
            return
        for folder in sorted(path for path in directory.iterdir() if path.is_dir()):
            account = folder.name[len("account-"):] if folder.name.startswith("account-") else rule.OWN_ACCOUNT
            found = self.place("request_records", folder, account, working_state=False)
            for path in sorted(folder.glob("*.jsonl")):
                info = path.stat()
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        self.item(found, info.st_mtime, len(line) + 1, account)

    def taps(self, directory):
        found = self.place("tap_records", directory)
        directory = Path(directory)
        if directory.is_dir():
            for path in sorted(directory.glob("taps-*.spt2")):
                info = path.stat()
                self.item(found, info.st_mtime, info.st_size)

    def report(self):
        places = [place.report() for place in self.places]
        for place, entry in zip(self.places, places):
            if hasattr(place, "backings"):
                entry["backings"] = place.backings
        selected = [entry for entry in places if entry["selected_items"]]
        expired = [entry for entry in places if entry["expired_items"]]
        return {"host": socket.gethostname(), "now": int(self.now), "home": str(self.home) if self.home else None,
                "directive": {"path": self.directive_path, "wiped": self.wiped, "held_until": self.held_until, "storage_wiped": self.storage_wiped},
                "selection": {"account": self.account, "all": self.everything, "before": self.before, "storage_before": self.storage_before,
                              "tenures": [{"account": account, "from": start, "to": end} for account, start, end in self.tenures] if self.tenures is not None else None},
                "grace_s": self.grace, "places": places,
                "selected": {"items": sum(entry["selected_items"] for entry in places), "bytes": sum(entry["selected_bytes"] for entry in places),
                             "places": [entry["kind"] + ":" + entry["path"] for entry in selected]},
                "expired": {"items": sum(entry["expired_items"] for entry in places), "bytes": sum(entry["expired_bytes"] for entry in places),
                            "places": [entry["kind"] + ":" + entry["path"] for entry in expired]}}


def process_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(prog="working_state_scan.py", description="List every place on this host where an account's working state can live, "
                                     "and what it holds for an account, for a time, or in all.")
    parser.add_argument("--home", help="the serving user's home: snapshot stores, prefix index files, scratch, the pair mirror and the KV ledger under it")
    parser.add_argument("--directive", help="the working-state directive to judge expiry by (default: <home>/sparkdata/working-state/directive)")
    parser.add_argument("--replay", help="the hub's replay store directory")
    parser.add_argument("--records", help="the hub's request records directory (kept a year, not working state; reported, never selected)")
    parser.add_argument("--taps", help="a draft server's tap record directory")
    parser.add_argument("--controller-state", help="the hub controller's working-state file, to attribute state by time to the accounts that held the fleet")
    parser.add_argument("--account", help="select the state of this account: named places, and with --controller-state its time on the fleet")
    parser.add_argument("--before", type=float, help="select cached state last used before these unix seconds")
    parser.add_argument("--storage-before", type=float, help="select storage (scratch areas) last used before these unix seconds")
    parser.add_argument("--all", action="store_true", help="select every item of working state")
    parser.add_argument("--now", type=float, help="judge expiry at these unix seconds instead of now")
    parser.add_argument("--grace", type=float, default=DEFAULT_GRACE_SECONDS, help="seconds a sweep may lag before an expired item counts")
    parser.add_argument("--expect-none", action="store_true", help="exit 1 when anything is selected")
    parser.add_argument("--expect-swept", action="store_true", help="exit 1 when anything the directive expires is still there")
    arguments = parser.parse_args(argv)
    if not (arguments.home or arguments.replay or arguments.records or arguments.taps):
        parser.error("name at least one of --home, --replay, --records or --taps")
    if arguments.expect_none and not (arguments.account or arguments.before is not None or arguments.storage_before is not None or arguments.all):
        parser.error("--expect-none needs a selection: --account, --before, --storage-before or --all")
    scan = Scan(arguments)
    scan.node()
    if arguments.replay:
        scan.replay(arguments.replay)
    if arguments.records:
        scan.records(arguments.records)
    if arguments.taps:
        scan.taps(arguments.taps)
    report = scan.report()
    print(json.dumps(report, indent=1, sort_keys=True))
    failed = (arguments.expect_none and report["selected"]["items"]) or (arguments.expect_swept and report["expired"]["items"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
