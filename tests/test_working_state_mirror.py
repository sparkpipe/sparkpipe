#!/usr/bin/env python3
import fcntl
import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/working_state_mirror.py"
LINK_MAP = "0-1:200,2-3:200,4-5:200,6-7:200,8-9:200,10-11:200,12-13:200,14-15:200"
MIRROR = "sparkdata/working-state/mirror"
SCRATCH = "sparkdata/working-state/scratch"
STORE = "sparkdata/glm53flash.fp8.tp16/kvsnapshot"
NESTED = STORE + "/nested"
OTHER = "sparkdata/qwen38.tp16/kvsnapshot"
DAY = 86400

SPEC = importlib.util.spec_from_file_location("working_state_mirror", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FAKE_SSH = """#!/bin/sh
echo "$1" >> "$SSH_CALLS"
[ "$1" = partner ] || { echo "ssh: connect to host $1 port 22: No route to host" >&2; exit 255; }
shift
cd "$PARTNER_HOME" || exit 255
exec /bin/sh -c "$*"
"""

PURGE_PROBE = r"""
#include <stdio.h>
#include <stdlib.h>
#include "sparkpipe/spark_kv_snapshot.h"
int main(int argc, char **argv) {
    SparkKvSnapshotPurgeResult result;
    SparkWorkingStateDirective directive = {0u, 0u};
    SparkStatus status;
    if (argc != 3 && argc != 4) return 2;
    if (argc == 4) directive.wiped_seconds = strtoull(argv[3], 0, 10);
    status = SparkKvSnapshotPurgeTree(argv[1], &directive, strtoull(argv[2], 0, 10), &result);
    printf("%d %u %llu\n", (int)status, result.stores, (unsigned long long)result.deleted_files);
    return status == SPARK_STATUS_OK ? 0 : 1;
}
"""


def rsync_directory():
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / "rsync"
        if not (candidate.is_file() and os.access(candidate, os.X_OK)):
            continue
        version = subprocess.run([str(candidate), "--version"], capture_output=True, text=True).stdout
        match = re.search(r"rsync\s+version\s+v?(\d+)\.(\d+)", version)
        if match and (int(match[1]), int(match[2])) >= (3, 1):
            return directory
    raise RuntimeError("the working-state mirror tests need rsync 3.1 or later on PATH (apt-get install rsync); "
                       "openrsync and rsync 2.x are not enough")


class WorkingStateMirror(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rsync = rsync_directory()
        cls.build = tempfile.TemporaryDirectory(prefix="working-state-probe-")
        probe = Path(cls.build.name) / "probe.c"
        probe.write_text(PURGE_PROBE)
        cls.purge = Path(cls.build.name) / "purge"
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-D_DARWIN_C_SOURCE", "-Wall", "-Wextra", "-Werror",
                        "-Iinclude", "cache/kv_snapshot_purge.c", "cache/working_state.c", str(probe), "-o", str(cls.purge)],
                       cwd=ROOT, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="working-state-")
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name).resolve()
        self.home = self.work / "spark4"
        self.partner = self.work / "spark5"
        self.home.mkdir()
        self.partner.mkdir()
        self.ssh = self.work / "fake_ssh"
        self.ssh.write_text(FAKE_SSH)
        self.ssh.chmod(0o755)
        self.calls = self.work / "ssh_calls"
        self.calls.write_text("")
        self.tool("init", home=self.partner, check=0)
        self.tool("init", check=0)
        self.now = int(time.time())

    def environment(self):
        return dict(os.environ, PATH=self.rsync + os.pathsep + os.environ.get("PATH", ""),
                    PARTNER_HOME=str(self.partner), SSH_CALLS=str(self.calls))

    def arguments(self, command, home=None, link_map=LINK_MAP, partner="partner", extra=()):
        arguments = [sys.executable, str(TOOL), command, "--home", str(home or self.home)]
        if command != "init":
            arguments += ["--rank", "4", "--link-map", link_map, "--partner", partner, "--rsh", str(self.ssh)]
        return arguments + list(extra)

    def tool(self, command, check=None, **options):
        result = subprocess.run(self.arguments(command, **options), env=self.environment(),
                                capture_output=True, text=True, timeout=120)
        if check is not None:
            self.assertEqual(result.returncode, check, result.stdout + result.stderr)
        return result

    def put(self, home, relative, data, mtime):
        path = home / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        os.utime(path, (mtime, mtime))
        return path

    def mirrored(self, relative):
        return self.partner / MIRROR / "4" / relative

    def files(self, base):
        return {str(path.relative_to(base)) for path in base.rglob("*") if path.is_file() or path.is_symlink()}

    def populate(self):
        kept = {
            f"{STORE}/aa.kvs": b"pages-aa",
            f"{STORE}/aa.kvst": b"state-aa",
            f"{NESTED}/bb.kvs": b"pages-bb",
            f"{OTHER}/cc.kvs": b"pages-cc",
            f"{SCRATCH}/results/report.bin": b"customer-1",
            f"{SCRATCH}/.last-use": b"",
        }
        ignored = {
            f"{STORE}/.kvs-writing-123-1": b"partial",
            f"{STORE}/.kvs-lock": b"",
            f"{STORE}/notes.txt": b"not a cache file",
            "sparkdata/glm53flash.fp8.tp16/packs/rank4.sp": b"pack",
            f"{MIRROR}/5/{STORE}/dd.kvs": b"the partner's copy held here",
            "elsewhere/readme": b"plain file",
        }
        times = {}
        for index, (relative, data) in enumerate(sorted(kept.items()) + sorted(ignored.items())):
            mtime = self.now - 3600 - 100 * index
            self.put(self.home, relative, data, mtime)
            if relative in kept:
                times[relative] = (data, mtime)
        (self.home / "linked").symlink_to(self.home / "sparkdata/glm53flash.fp8.tp16")
        return times

    def ssh_hosts(self):
        return set(self.calls.read_text().split())

    def wait_for(self, predicate, seconds=30):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.1)
        self.fail("condition not reached")

    def test_push_copies_stores_and_scratch_with_their_times(self):
        kept = self.populate()
        result = self.tool("push", check=0)
        self.assertIn("push to partner: 3 stores, 4 sent", result.stdout)
        for relative, (data, mtime) in kept.items():
            copy = self.mirrored(relative)
            self.assertEqual(copy.read_bytes(), data, relative)
            self.assertEqual(int(copy.stat().st_mtime), mtime, relative)
        self.assertEqual(self.files(self.partner / MIRROR / "4"), set(kept))
        self.assertEqual(self.ssh_hosts(), {"partner"})

    def test_a_file_deleted_at_the_source_is_deleted_at_the_mirror_on_the_next_push(self):
        self.populate()
        self.tool("push", check=0)
        (self.home / STORE / "aa.kvs").unlink()
        (self.home / SCRATCH / "results/report.bin").unlink()
        self.tool("push", check=0)
        self.assertFalse(self.mirrored(f"{STORE}/aa.kvs").exists())
        self.assertFalse(self.mirrored(f"{SCRATCH}/results/report.bin").exists())
        self.assertTrue(self.mirrored(f"{STORE}/aa.kvst").exists())
        self.assertTrue(self.mirrored(f"{SCRATCH}/.last-use").exists())

    def test_a_new_time_on_a_snapshot_reaches_the_mirror_without_a_copy(self):
        self.populate()
        self.tool("push", check=0)
        source, copy = self.home / STORE / "aa.kvs", self.mirrored(f"{STORE}/aa.kvs")
        inode = copy.stat().st_ino
        source.write_bytes(b"PAGES-AA")
        os.utime(source, (self.now - 60, self.now - 60))
        result = self.tool("push", check=0)
        self.assertIn("0 sent, 1 touched", result.stdout)
        self.assertEqual(copy.stat().st_ino, inode)
        self.assertEqual(copy.read_bytes(), b"pages-aa")
        self.assertEqual(int(copy.stat().st_mtime), self.now - 60)

    def test_a_scratch_file_changed_in_place_with_the_same_size_is_copied_again(self):
        self.populate()
        self.tool("push", check=0)
        source = self.home / SCRATCH / "results/report.bin"
        source.write_bytes(b"customer-2")
        os.utime(source, (self.now - 30, self.now - 30))
        self.tool("push", check=0)
        copy = self.mirrored(f"{SCRATCH}/results/report.bin")
        self.assertEqual(copy.read_bytes(), b"customer-2")
        self.assertEqual(int(copy.stat().st_mtime), self.now - 30)

    def test_the_copies_of_a_store_that_is_gone_are_removed(self):
        self.populate()
        self.tool("push", check=0)
        shutil.rmtree(self.home / "sparkdata/qwen38.tp16")
        (self.home / NESTED / "bb.kvs").unlink()
        self.tool("push", check=0)
        self.assertFalse(self.mirrored("sparkdata/qwen38.tp16").exists())
        self.assertFalse(self.mirrored(NESTED + "/bb.kvs").exists())
        self.assertTrue(self.mirrored(f"{STORE}/aa.kvs").exists())
        self.assertTrue(self.mirrored(f"{SCRATCH}/results/report.bin").exists())

    def test_a_scratch_area_that_is_gone_is_removed_from_the_mirror(self):
        self.populate()
        self.tool("push", check=0)
        shutil.rmtree(self.home / SCRATCH)
        result = self.tool("push", check=0)
        self.assertIn("scratch absent", result.stdout)
        self.assertFalse(self.mirrored(SCRATCH).exists())
        self.assertTrue(self.mirrored(f"{STORE}/aa.kvs").exists())

    def test_rebuild_restores_an_empty_home_byte_for_byte_with_times(self):
        kept = self.populate()
        self.tool("push", check=0)
        fresh = self.work / "replaced"
        fresh.mkdir()
        result = self.tool("rebuild", home=fresh, check=0)
        self.assertIn("6 files restored into 3 stores and scratch", result.stdout)
        for relative, (data, mtime) in kept.items():
            self.assertEqual((fresh / relative).read_bytes(), data, relative)
            self.assertEqual(int((fresh / relative).stat().st_mtime), mtime, relative)
        self.assertEqual({name for name in self.files(fresh) if not name.endswith("/.kvs-lock")}, set(kept))
        self.assertTrue((fresh / MIRROR).is_dir())
        again = self.tool("push", home=fresh, check=0)
        self.assertIn("3 stores, 0 sent, 0 touched, 0 deleted; scratch 0 sent, 0 deleted", again.stdout)

    def test_rebuild_never_overwrites_a_local_file(self):
        self.populate()
        self.tool("push", check=0)
        fresh = self.work / "replaced"
        local = self.put(fresh, f"{STORE}/aa.kvs", b"LOCAL-AA", self.now)
        self.tool("rebuild", home=fresh, check=0)
        self.assertEqual(local.read_bytes(), b"LOCAL-AA")
        self.assertEqual((fresh / STORE / "aa.kvst").read_bytes(), b"state-aa")

    def test_rebuild_skips_a_store_whose_lock_is_held(self):
        self.populate()
        self.tool("push", check=0)
        fresh = self.work / "replaced"
        (fresh / STORE).mkdir(parents=True)
        with open(fresh / STORE / ".kvs-lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = self.tool("rebuild", home=fresh, check=1)
            self.assertIn(f"store {STORE} is in use", result.stdout)
            self.assertFalse((fresh / STORE / "aa.kvs").exists())
            self.assertEqual((fresh / NESTED / "bb.kvs").read_bytes(), b"pages-bb")
            self.assertEqual((fresh / OTHER / "cc.kvs").read_bytes(), b"pages-cc")
            self.assertTrue((fresh / MIRROR).is_dir())
        self.tool("rebuild", home=fresh, check=0)
        self.assertEqual((fresh / STORE / "aa.kvs").read_bytes(), b"pages-aa")

    def test_rebuild_from_an_unreachable_partner_leaves_the_area_uninitialized(self):
        fresh = self.work / "replaced"
        fresh.mkdir()
        result = self.tool("rebuild", home=fresh, partner="offline", check=1)
        self.assertIn("rebuild from offline failed", result.stdout)
        self.assertFalse((fresh / "sparkdata/working-state").exists())

    def test_rebuild_from_a_partner_without_a_mirror_readies_the_area(self):
        fresh = self.work / "replaced"
        fresh.mkdir()
        result = self.tool("rebuild", home=fresh, check=0)
        self.assertIn("holds no mirror of rank 4", result.stdout)
        self.assertTrue((fresh / MIRROR).is_dir())
        self.assertTrue((fresh / SCRATCH / ".last-use").is_file())

    def test_init_keeps_an_existing_stamp(self):
        stamp = self.home / SCRATCH / ".last-use"
        os.utime(stamp, (self.now - DAY, self.now - DAY))
        self.tool("init", check=0)
        self.assertEqual(int(stamp.stat().st_mtime), self.now - DAY)

    def test_a_disk_that_was_not_rebuilt_never_pushes(self):
        fresh = self.work / "replaced"
        self.put(fresh, f"{STORE}/aa.kvs", b"pages-aa", self.now)
        self.populate()
        self.tool("push", check=0)
        result = self.tool("push", home=fresh, check=1)
        self.assertIn("push refused", result.stdout)
        self.assertTrue(self.mirrored(f"{OTHER}/cc.kvs").exists())

    def test_a_push_never_creates_the_working_state_of_a_partner_that_was_not_rebuilt(self):
        self.populate()
        shutil.rmtree(self.partner / "sparkdata/working-state")
        result = self.tool("push", check=1)
        self.assertIn("stores: rsync exit", result.stdout)
        self.assertFalse((self.partner / "sparkdata/working-state").exists())

    def test_an_unpaired_rank_does_not_mirror(self):
        self.populate()
        for command in ("push", "rebuild", "loop"):
            with self.subTest(command=command):
                result = self.tool(command, link_map="0-1:200,2-3:200,6-7:200", check=0)
                self.assertIn("rank 4 has no cabled pair", result.stdout)
        self.assertFalse((self.partner / MIRROR / "4").exists())
        self.assertEqual(self.ssh_hosts(), set())

    def test_a_malformed_link_map_fails_loudly(self):
        for link_map in ("4-6:200", "5-4:200", "4-5", "4-5:x"):
            with self.subTest(link_map=link_map):
                result = self.tool("push", link_map=link_map, check=2)
                self.assertIn("is not LOW-HIGH:GBPS", result.stderr)

    def test_the_loop_rebuilds_a_fresh_disk_before_it_pushes_and_ends_with_the_agent(self):
        kept = self.populate()
        self.tool("push", check=0)
        fresh = self.work / "replaced"
        fresh.mkdir()
        agent = subprocess.Popen(["sleep", "60"])
        self.addCleanup(agent.kill)
        log = self.work / "loop.log"
        with open(log, "w") as output:
            loop = subprocess.Popen(self.arguments("loop", home=fresh, extra=("--interval", "0.3", "--agent-pid", str(agent.pid))),
                                    env=self.environment(), stdout=output, stderr=subprocess.STDOUT)
        self.addCleanup(loop.kill)
        self.wait_for(lambda: log.read_text().count("push to partner") >= 2)
        agent.kill()
        agent.wait()
        self.assertEqual(loop.wait(timeout=10), 0)
        lines = log.read_text().splitlines()
        self.assertIn("rebuild from partner", lines[0])
        self.assertTrue(all("push to partner: 3 stores, 0 sent" in line and "0 failures" in line for line in lines[1:]), lines)
        self.assertEqual((fresh / f"{STORE}/aa.kvs").read_bytes(), kept[f"{STORE}/aa.kvs"][0])

    def test_a_failed_pass_is_retried_on_the_next_pass(self):
        self.populate()
        shutil.rmtree(self.partner / "sparkdata/working-state")
        log = self.work / "loop.log"
        with open(log, "w") as output:
            loop = subprocess.Popen(self.arguments("loop", extra=("--interval", "0.3")),
                                    env=self.environment(), stdout=output, stderr=subprocess.STDOUT)
        self.addCleanup(loop.kill)
        self.wait_for(lambda: "2 failures: stores: rsync exit" in log.read_text())
        self.tool("init", home=self.partner, check=0)
        self.wait_for(lambda: self.mirrored(f"{STORE}/aa.kvs").exists() and "0 failures" in log.read_text())
        loop.send_signal(signal.SIGTERM)
        self.assertEqual(loop.wait(timeout=15), 128 + signal.SIGTERM)

    def test_the_partner_purges_expired_copies_by_their_kept_times(self):
        deep = "/".join(f"d{level}" for level in range(1, MODULE.STORE_DEPTH_MAX + 1))
        self.put(self.home, f"{STORE}/old.kvs", b"old", self.now - DAY - 600)
        self.put(self.home, f"{STORE}/new.kvs", b"new", self.now - 600)
        self.put(self.home, f"{deep}/old.kvs", b"deep", self.now - DAY - 600)
        self.tool("push", check=0)
        self.assertTrue(self.mirrored(f"{deep}/old.kvs").exists())
        result = subprocess.run([str(self.purge), str(self.partner), str(self.now)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.mirrored(f"{STORE}/old.kvs").exists())
        self.assertFalse(self.mirrored(f"{deep}/old.kvs").exists())
        self.assertEqual(self.mirrored(f"{STORE}/new.kvs").read_bytes(), b"new")
        self.assertTrue((self.home / STORE / "old.kvs").exists())

    def test_a_wipe_deletes_the_copies_on_the_partner_too(self):
        self.put(self.home, f"{STORE}/before.kvs", b"before", self.now - 600)
        self.put(self.home, f"{STORE}/after.kvs", b"after", self.now - 60)
        self.tool("push", check=0)
        result = subprocess.run([str(self.purge), str(self.partner), str(self.now), str(self.now - 300)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.mirrored(f"{STORE}/before.kvs").exists())
        self.assertEqual(self.mirrored(f"{STORE}/after.kvs").read_bytes(), b"after")

    def test_a_store_too_deep_for_the_partner_purge_is_not_mirrored(self):
        too_deep = "/".join(f"d{level}" for level in range(1, MODULE.STORE_DEPTH_MAX + 2))
        self.put(self.home, f"{too_deep}/old.kvs", b"deep", self.now - DAY - 600)
        self.put(self.home, f"{STORE}/aa.kvs", b"pages-aa", self.now)
        result = self.tool("push", check=1)
        self.assertIn(f"store {too_deep} is deeper than {MODULE.STORE_DEPTH_MAX} below home", result.stdout)
        self.assertFalse(self.mirrored(too_deep).exists())
        self.assertTrue(self.mirrored(f"{STORE}/aa.kvs").exists())
        unreachable = self.put(self.partner, f"{MIRROR}/4/{too_deep}/old.kvs", b"deep", self.now - DAY - 600)
        subprocess.run([str(self.purge), str(self.partner), str(self.now)], check=True, capture_output=True)
        self.assertTrue(unreachable.exists())

    def test_stores_are_found_as_the_purge_finds_them(self):
        tree = self.work / "tree"
        layout = {
            "a/x.kvs": "store", "a/sub/z.kvst": "store", "b/.kvs-lock": "store", "c/.kvs-writing-1": "store",
            "d/.kvs": "", "e/notes": "", "f/g.kvs/w.kvs": "", "h/dir.kvst/q": "",
            "/".join(f"p{level}" for level in range(1, 17)) + "/y.kvs": "store",
            "/".join(f"q{level}" for level in range(1, 18)) + "/y.kvs": "",
        }
        for relative in layout:
            self.put(tree, relative, b"", self.now)
        (tree / "link").symlink_to(tree / "a")
        stores, errors = MODULE.find_stores(str(tree))
        self.assertEqual(errors, [])
        expected = {"a", "a/sub", "b", "c", "f", "h", "/".join(f"p{level}" for level in range(1, 17))}
        self.assertEqual(set(stores), expected)
        result = subprocess.run([str(self.purge), str(tree), "1"], capture_output=True, text=True, check=True)
        self.assertEqual(int(result.stdout.split()[1]), len(expected))

    def test_the_walk_uses_the_purge_constants(self):
        header = (ROOT / "include/sparkpipe/spark_kv_snapshot.h").read_text()

        def define(name):
            return re.search(rf"^#define {name} (\S+)$", header, re.M)[1]

        self.assertEqual(define("SPARK_KV_SNAPSHOT_SUFFIX"), f'"{MODULE.SNAPSHOT_SUFFIX}"')
        self.assertEqual(define("SPARK_KV_SNAPSHOT_STATE_SUFFIX"), f'"{MODULE.STATE_SUFFIX}"')
        self.assertEqual(define("SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX"), f'"{MODULE.TEMPORARY_PREFIX}"')
        self.assertEqual(define("SPARK_KV_SNAPSHOT_LOCK_NAME"), f'"{MODULE.LOCK_NAME}"')
        self.assertEqual(define("SPARK_KV_PURGE_DEPTH_MAX"), f"{MODULE.PURGE_DEPTH_MAX}u")


if __name__ == "__main__":
    unittest.main()
