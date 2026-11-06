#!/usr/bin/env python3
import contextlib
import importlib.util
import io
import json
import os
import sqlite3
import struct
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scan = load("working_state_scan", ROOT / "tools" / "working_state_scan.py")
working_state = load("working_state", ROOT / "tools" / "serving" / "working_state.py")
chat_replay = load("chat_replay", ROOT / "tools" / "serving" / "chat_replay.py")

PURGE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "sparkpipe/spark_kv_snapshot.h"
int main(int argc, char **argv) {
    SparkKvSnapshotPurgeResult result;
    SparkWorkingStatePurgeResult scratch;
    SparkWorkingStateDirective directive;
    SparkKvLedger ledger;
    char path[1200];
    if (argc != 4) return 2;
    if (SparkWorkingStateDirectiveRead(argv[2], &directive) != SPARK_STATUS_OK) return 3;
    snprintf(path, sizeof(path), "%s/.scan-test-ledger", argv[1]);
    if (SparkKvLedgerOpen(&ledger, path, 1ull << 40) != SPARK_STATUS_OK) return 4;
    if (SparkKvSnapshotPurgeTree(&ledger, argv[1], &directive, strtoull(argv[3], 0, 10), &result) != SPARK_STATUS_OK) return 5;
    if (SparkWorkingStatePurgeScratch(argv[1], &directive, strtoull(argv[3], 0, 10), &scratch) != SPARK_STATUS_OK) return 6;
    unlink(path);
    printf("%llu %u\n", (unsigned long long)result.deleted_files, scratch.emptied_areas);
    return 0;
}
"""

NOW = int(time.time())
T1 = NOW - 7200
T2 = NOW - 3600


def touch(path, used, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.utime(path, (used, used))


def ledger_bytes(slots):
    header = scan.LEDGER_HEADER.pack(scan.LEDGER_MAGIC, 1, len(slots), 500_000_000_000, 0, 0, 0, 0, 0)
    body = b"".join(scan.LEDGER_SLOT.pack(kind, pid, 0, size, 0, 0, directory.encode().ljust(512, b"\0")) for kind, pid, size, directory in slots)
    return header + body


def index_bytes(used_times):
    records = b"".join(scan.INDEX_RECORD.pack(0, 64, 1, 2, 3, b"\0" * 32, 0, used) for used in used_times)
    header = scan.INDEX_HEADER.pack(scan.INDEX_MAGIC, 2, scan.INDEX_HEADER.size, 64, len(used_times), scan.INDEX_RECORD.size, 0, b"\0" * 32, b"\0" * 32, b"\0" * 32)
    return header + records


class WorkingStateScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="working-state-scan-probe-")
        probe = Path(cls.build.name) / "purge.c"
        probe.write_text(PURGE)
        cls.purge = Path(cls.build.name) / "purge"
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-D_DARWIN_C_SOURCE", "-Wall", "-Wextra", "-Werror", "-Iinclude", "cache/kv_snapshot_purge.c",
                        "cache/kv_ledger.c", "cache/working_state.c", "src/spark_status.c", str(probe), "-o", str(cls.purge)], cwd=ROOT, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="working-state-scan-")
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name).resolve()
        self.home = self.work / "spark4"
        store = self.home / "stage-lanes" / "lane1" / "kv"
        (store / scan.LOCK_NAME).parent.mkdir(parents=True)
        (store / scan.LOCK_NAME).write_text("")
        touch(store / "aa" / "aa-01-64.kvs", T1 + 10, b"a" * 10)
        touch(store / "aa" / "aa-02-64.kvst", T2 + 10, b"b" * 20)
        touch(store / "bb" / "bb-03-64.kvs", NOW - 60, b"c" * 30)
        flat = self.home / "old-lane" / "kv"
        touch(flat / "cc-04-64.kvs", T1 - 600, b"d" * 40)
        own = self.home / "sparkdata" / "working-state" / "scratch"
        touch(own / "repo" / "main.c", T1 + 20, b"int main;")
        touch(own / scan.LAST_USE_NAME, T1 + 30, b"")
        mirror = self.home / "sparkdata" / "working-state" / "mirror" / "5"
        touch(mirror / "stage-lanes" / "lane1" / "kv" / "aa" / "aa-09-64.kvs", T1 + 40, b"e" * 50)
        touch(mirror / "stage-lanes" / "lane1" / "kv" / "bb" / "bb-08-64.kvs", NOW - 30, b"f" * 60)
        (mirror / "stage-lanes" / "lane1" / "kv" / scan.LOCK_NAME).write_text("")
        touch(mirror / "sparkdata" / "working-state" / "scratch" / "notes.txt", NOW - 10, b"notes")
        touch(mirror / "sparkdata" / "working-state" / "scratch" / scan.LAST_USE_NAME, NOW - 10, b"")
        runtime = self.home / "hub-lanes" / "glm" / "root"
        touch(runtime / "prefix_index.spi", NOW, index_bytes([T1 + 50, T2 + 50, NOW - 5]))
        (self.home / scan.LEDGER_NAME).write_bytes(ledger_bytes([(scan.LEDGER_KIND_SNAPSHOT, os.getpid(), 100, str(store)),
                                                                 (scan.LEDGER_KIND_BACKING, os.getpid(), 4096, str(self.home / "backing")),
                                                                 (scan.LEDGER_KIND_BACKING, 2 ** 31 - 7, 8192, str(self.home / "dead"))]))
        self.directive = self.home / "sparkdata" / "working-state" / "directive"
        self.directive.write_text(working_state.directive_text(T2, 0, T2))
        self.replay = self.work / "replay"
        self.replay.mkdir()
        for name, used in (("account-acme", T1 + 60), ("account-bob", T2 + 60), ("own", NOW - 1)):
            connection = sqlite3.connect(self.replay / f"{name}.sqlite")
            for statement in chat_replay.SCHEMA:
                connection.execute(statement)
            connection.execute("INSERT INTO turns VALUES ('t1', 'm', '', x'00', x'00', x'00', 100, ?)", (used,))
            connection.execute("INSERT INTO aliases VALUES (1, '{}', 2, ?)", (used,))
            if name == "account-acme":
                if "body" not in {row[1] for row in connection.execute("PRAGMA table_info(responses)")}:
                    connection.execute("ALTER TABLE responses ADD COLUMN body TEXT")
                connection.execute("INSERT INTO responses (id, parent, model, member, messages, bytes, used, body) VALUES ('resp_1', NULL, 'm', '', '[]', 40, ?, "
                                   "'{\"output\": []}')", (used,))
            connection.commit()
            connection.close()
        self.records = self.work / "records"
        touch(self.records / "account-acme" / "2026-11-06.jsonl", T1 + 70, b'{"a":1}\n{"a":2}\n')
        self.controller = self.work / "working-state.json"
        self.controller.write_text(json.dumps({"user": "bob", "since": T2, "wiped": T2, "held_until": 0, "storage_wiped": T2, "restored": T2, "history": [
            {"kind": "handover", "from": None, "to": "sparkpipe", "wiped": T1 - 3600, "ok": True},
            {"kind": "handover", "from": "sparkpipe", "to": "acme", "wiped": T1, "ok": True},
            {"kind": "delete", "account": "acme", "from": "acme", "to": "acme", "wiped": T1 + 5, "ok": True},
            {"kind": "handover", "from": "acme", "to": "bob", "wiped": T2, "ok": True}]}))

    def run_scan(self, *arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = scan.main(["--home", str(self.home), "--replay", str(self.replay), "--records", str(self.records), "--now", str(NOW), *arguments])
        return code, json.loads(output.getvalue())

    def kinds(self, report, field):
        totals = {}
        for place in report["places"]:
            totals[place["kind"]] = totals.get(place["kind"], 0) + place[field]
        return totals

    def test_lists_every_place(self):
        code, report = self.run_scan("--all")
        self.assertEqual(code, 0)
        items = self.kinds(report, "items")
        self.assertEqual(items, {"snapshot": 4, "scratch": 1, "mirror": 2, "mirror_scratch": 1, "prefix_index": 3, "nvme_spill": 2,
                                 "replay_store": 7, "request_records": 2})
        selected = self.kinds(report, "selected_items")
        self.assertEqual(selected["request_records"], 0, "request records are kept a year and are not working state")
        self.assertEqual(selected["nvme_spill"], 0, "a spill backing has no files to attribute")
        self.assertEqual(report["selected"]["items"], 4 + 1 + 2 + 1 + 3 + 7)
        spill = next(place for place in report["places"] if place["kind"] == "nvme_spill")
        self.assertEqual([(entry["live"], entry["bytes"]) for entry in spill["backings"]], [(True, 4096), (False, 8192)])
        self.assertIn("O_TMPFILE", spill["unattributable"])
        self.assertEqual(report["directive"]["wiped"], T2)

    def test_before_selects_cached_state_only(self):
        code, report = self.run_scan("--before", str(T2), "--expect-none")
        self.assertEqual(code, 1)
        selected = self.kinds(report, "selected_items")
        self.assertEqual((selected["snapshot"], selected["mirror"], selected["prefix_index"], selected["replay_store"]), (2, 1, 1, 3))
        self.assertEqual((selected["scratch"], selected["mirror_scratch"]), (0, 0), "storage needs --storage-before")
        code, report = self.run_scan("--before", str(T2), "--storage-before", str(T2))
        self.assertEqual(self.kinds(report, "selected_items")["scratch"], 1)

    def test_account_by_name_and_time(self):
        code, report = self.run_scan("--account", "acme", "--controller-state", str(self.controller), "--expect-none")
        self.assertEqual(code, 1)
        tenures = {(entry["account"], entry["from"], entry["to"]) for entry in report["selection"]["tenures"]}
        self.assertEqual(tenures, {("sparkpipe", T1 - 3600, T1), ("acme", T1, T2), ("bob", T2, None)})
        selected = self.kinds(report, "selected_items")
        self.assertEqual((selected["snapshot"], selected["mirror"], selected["prefix_index"], selected["scratch"], selected["replay_store"]), (1, 1, 1, 1, 3),
                         "the account's turns, alias table and stored response, body included")
        named = [place for place in report["places"] if place["kind"] == "replay_store" and place["selected_items"]]
        self.assertEqual([place["account"] for place in named], ["acme"])
        code, report = self.run_scan("--account", "nobody", "--controller-state", str(self.controller), "--expect-none")
        self.assertEqual((code, report["selected"]["items"]), (0, 0))

    def test_expired_by_the_directive(self):
        code, report = self.run_scan("--expect-swept")
        self.assertEqual(code, 1)
        expired = self.kinds(report, "expired_items")
        self.assertEqual((expired["snapshot"], expired["mirror"], expired["scratch"], expired["prefix_index"], expired["replay_store"]), (2, 1, 1, 1, 3))
        self.assertEqual(expired["mirror_scratch"], 0)

    def test_the_purge_leaves_nothing_the_scan_finds(self):
        result = subprocess.run([str(self.purge), str(self.home), str(self.directive), str(NOW)], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.split(), ["3", "1"])
        code, report = self.run_scan("--before", str(T2), "--storage-before", str(T2))
        selected = self.kinds(report, "selected_items")
        self.assertEqual((selected["snapshot"], selected["mirror"], selected["scratch"], selected["mirror_scratch"]), (0, 0, 0, 0))
        self.assertEqual((selected["prefix_index"], selected["replay_store"]), (1, 3), "the hub's own sweeps delete those, not weightd's purge")
        self.directive.write_text(working_state.directive_text(NOW, 0, NOW))
        subprocess.run([str(self.purge), str(self.home), str(self.directive), str(NOW)], capture_output=True, text=True, check=True)
        (self.home / "hub-lanes" / "glm" / "root" / "prefix_index.spi").unlink()
        for path in self.replay.glob("*.sqlite"):
            path.unlink()
        code, report = self.run_scan("--all", "--expect-none")
        self.assertEqual((code, report["selected"]["items"]), (0, 0), f"after a wipe at now nothing of anyone is left: {report['selected']}")

    def test_a_cache_deletion_keeps_storage(self):
        self.directive.write_text(working_state.directive_text(NOW, 0, T1))
        subprocess.run([str(self.purge), str(self.home), str(self.directive), str(NOW)], capture_output=True, text=True, check=True)
        code, report = self.run_scan("--before", str(NOW), "--expect-none")
        self.assertEqual(self.kinds(report, "selected_items").get("snapshot", 0), 0)
        self.assertEqual(self.kinds(report, "items")["scratch"], 1, "storage last used after storage_wiped stays")
        code, report = self.run_scan("--expect-swept")
        self.assertEqual(self.kinds(report, "expired_items")["scratch"], 0)

    def test_requires_a_place_and_a_selection(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                scan.main(["--all"])
            with self.assertRaises(SystemExit):
                scan.main(["--home", str(self.home), "--expect-none"])


if __name__ == "__main__":
    unittest.main()
