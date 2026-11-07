#!/usr/bin/env python3
"""tools/node_pack_prune.py keeps every file something uses and lists the rest.

A fixture fleet (ssh runs the node script against a scratch home) holds, on
spark3, the K3 and DeepSeek V4.1 Flash rank files two fixture records name
with their companions, a partner pack only a live lane config names, the
tool a swap controller command calls, a keep-listed staged pack, a fleet
agent root, and unused files: a pack
variant directory, a backup beside a served pack, an unused tree, a
directory holding only a link, a file modified a minute ago and a file a
process holds open. The dry run lists exactly the unused entries with their
bytes and deletes nothing. --delete removes the old unused entries, refuses
the open and the recent one, leaves every used file and every link target in
place, and deletes nothing at all on a node that lacks a file a record
names. The keep list refuses an entry without a reason or a date, and
lane_config.py audit reports each unused entry as UNUSED <bytes> <path>.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_lane_direct_packs import FAKE_REMOTE, fake_bin
import node_pack_prune
import stage_lane

HOST = "spark3"
OLD = time.time() - 7200


class NodePackPrune(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        scratch = Path(self.temporary.name)
        self.fake = scratch / "fleet"
        self.home = self.fake / "home" / HOST
        binary = fake_bin(scratch, {"ssh": FAKE_REMOTE})
        document = json.loads(stage_lane.LANES.read_text())
        document["lanes"] = {key: document["lanes"][key] for key in ("kimi-k3", "deepseek-v4.1-flash")}
        self.lanes = scratch / "lanes.json"
        self.lanes.write_text(json.dumps(document))
        self.keep = scratch / "keep.json"
        self.keep.write_text(json.dumps({"keep": [{"path": "motif3.tp16", "since": "2026-11-01", "reason": "staged for the Motif 3 driver"}]}))
        self.environment = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", FAKE_FLEET=str(self.fake), SPARK_LANES=str(self.lanes))
        self.used = []
        for model in ("k3", "dsv41_flash"):
            for path in stage_lane.rank_weight_files(stage_lane.MODELS[model], 3):
                self.used.append(self.put(path[len(f"/home/{HOST}/"):], 64))
        k3 = stage_lane.rank_pack_path(stage_lane.MODELS["k3"], 3)[len(f"/home/{HOST}/"):]
        self.used += [self.put(k3 + ".verified", 10), self.put(k3 + ".receipt.json", 10)]
        partner = "sparkdata/k3.mxfp4.tp16/partner-test/k3.stage0.rank02.pack"
        self.used += [self.put(partner, 300), self.put(partner + ".experts", 16)]
        adapter = self.home / "k3-lanes/lane8/root/config/adapter.json"
        adapter.parent.mkdir(parents=True)
        adapter.write_text(json.dumps({"stage_pack_path": f"/home/{HOST}/{k3}", "streamed_prefill": {"node_rows": 1, "partner_pack": f"/home/{HOST}/{partner}"}}))
        self.used += [self.put("sparkdata/motif3.tp16/motif3.rank03.pack", 500), self.put("sparkdata/glmroot/MANIFEST", 5), self.put("sparkdata/glmroot/bin/sparkpipe_model_residentd", 50),
                      self.put("sparkdata/swap/weightd_warm", 50)]
        self.unused = {
            "sparkdata/k3.mxfp4.tp16/packs-v3": (self.put("sparkdata/k3.mxfp4.tp16/packs-v3/k3.stage0.rank03.pack", 1000), 1000, "/"),
            "sparkdata/k3.mxfp4.tp16/packs/k3.stage0.rank03.pack.bak": (self.put(k3 + ".bak", 900), 900, ""),
            "sparkdata/qwenmax.nvfp4.tp16": (self.put("sparkdata/qwenmax.nvfp4.tp16/rank3.pack", 700) and self.put("sparkdata/qwenmax.nvfp4.tp16/receipts/r.json", 30), 730, "/"),
            "sparkdata/fresh.pack": (self.put("sparkdata/fresh.pack", 20, recent=True), 20, ""),
            "sparkdata/held.pack": (self.put("sparkdata/held.pack", 40), 40, ""),
        }
        links = self.home / "sparkdata/oldlinks"
        links.mkdir()
        (links / "k3.stage0.rank03.pack").symlink_to(self.home / k3)
        os.utime(links / "k3.stage0.rank03.pack", (OLD, OLD), follow_symlinks=False)
        os.utime(links, (OLD, OLD))
        self.unused["sparkdata/oldlinks"] = (links, 0, "/")
        for directory, _, _ in os.walk(self.home / "sparkdata"):
            if Path(directory) != self.home / "sparkdata/fresh.pack":
                os.utime(directory, (OLD, OLD))

    def put(self, relative, size, recent=False):
        path = self.home / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        if not recent:
            os.utime(path, (OLD, OLD))
        return path

    def run_tool(self, *arguments):
        return subprocess.run([sys.executable, str(ROOT / "tools/node_pack_prune.py"), "--host", HOST, "--keep", str(self.keep), *arguments],
                              env=self.environment, capture_output=True, text=True, timeout=300)

    def unused_lines(self):
        return sorted(f"UNUSED {HOST} {size} /home/{HOST}/{name}{suffix}" for name, (_, size, suffix) in self.unused.items())

    def test_dry_run_lists_exactly_the_unused_entries(self):
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(sorted(line for line in lines if line.startswith("UNUSED ")), self.unused_lines())
        self.assertEqual([line for line in lines if line.startswith("LIVE ")],
                         [f"LIVE {HOST} /home/{HOST}/sparkdata/k3.mxfp4.tp16/partner-test/k3.stage0.rank02.pack is named only by a live config on the node, not by its record"])
        self.assertFalse(any(line.startswith(("MISSING", "DELETED", "REFUSED")) for line in lines), lines)
        total = sum(size for _, size, _ in self.unused.values())
        self.assertIn(f"TOTAL {HOST} {total} bytes unused", result.stdout)
        self.assertTrue(all(path.exists() for path in self.used + [entry for entry, _, _ in self.unused.values()]))

    def test_delete_removes_old_unused_entries_and_refuses_open_or_recent_ones(self):
        with open(self.home / "sparkdata/held.pack", "rb"):
            result = self.run_tool("--delete")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        lines = result.stdout.splitlines()
        refused = sorted(line for line in lines if line.startswith("REFUSED "))
        self.assertEqual(refused, [f"REFUSED {HOST} /home/{HOST}/sparkdata/fresh.pack: modified within the last 3600 s",
                                   f"REFUSED {HOST} /home/{HOST}/sparkdata/held.pack: a running process holds it open"])
        deleted = sorted(line.split()[3] for line in lines if line.startswith("DELETED "))
        self.assertEqual(deleted, sorted(f"/home/{HOST}/{name}" for name in self.unused if name not in ("sparkdata/fresh.pack", "sparkdata/held.pack")))
        for name, (path, _, _) in self.unused.items():
            self.assertEqual(os.path.lexists(self.home / name), name in ("sparkdata/fresh.pack", "sparkdata/held.pack"), name)
        self.assertTrue(all(path.exists() for path in self.used))

    def test_a_node_missing_a_named_file_deletes_nothing(self):
        sibling = stage_lane.rank_weight_files(stage_lane.MODELS["dsv41_flash"], 3)[-1]
        Path(self.home.as_posix() + sibling[len(f"/home/{HOST}"):]).unlink()
        dry = self.run_tool()
        self.assertIn(f"MISSING {HOST} {sibling} is named by a lane record and absent", dry.stdout)
        result = self.run_tool("--delete")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(any(line.startswith("DELETED ") for line in result.stdout.splitlines()))
        self.assertEqual(len([line for line in result.stdout.splitlines() if line.startswith("REFUSED ")]), len(self.unused))
        self.assertIn("lacks 1 file(s) a record names", result.stdout)
        self.assertTrue(all(os.path.lexists(self.home / name) for name in self.unused))

    def test_keep_list_entries_need_a_reason_and_a_date(self):
        for broken in ({"path": "x"}, {"path": "x", "since": "soon", "reason": "r"}, {"path": "x", "since": "2026-11-02", "reason": " "},
                       {"path": "/abs", "since": "2026-11-02", "reason": "r"}, {"path": "a/../b", "since": "2026-11-02", "reason": "r"}):
            with self.subTest(entry=broken):
                self.keep.write_text(json.dumps({"keep": [broken]}))
                result = self.run_tool()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("every keep entry has a sparkdata-relative path, a since date (YYYY-MM-DD) and a reason", result.stderr)
        entries = node_pack_prune.keep_list()
        self.assertTrue({"core", "weightd", "swap", "motif3.tp16", "glm53full.bf16.tp16", "glm53flash.bf16.tp16", "k3.mxfp4.tp16/partner"} <= {entry["path"] for entry in entries})

    def test_audit_reports_each_unused_entry(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/lane_config.py"), "audit"], env=self.environment, capture_output=True, text=True, timeout=900)
        self.assertNotEqual(result.returncode, 0)
        lines = sorted(line for line in result.stdout.splitlines() if line.startswith("UNUSED ") and f" {HOST}:" in line)
        self.assertEqual(lines, sorted(f"UNUSED {size} {HOST}:/home/{HOST}/{name}" for name, (_, size, _) in self.unused.items()))


if __name__ == "__main__":
    unittest.main()
