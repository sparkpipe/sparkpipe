import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WeightdSupervision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="weightd-supervision-")
        cls.directory = Path(cls.tmp.name)
        faults = cls.directory / "faults.c"
        faults.write_text(r'''
#include "sparkpipe/spark_weightd.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int mode(const char *name) { const char *m=getenv("WARM_FAILURE"); return m && strcmp(m,name)==0; }
SparkStatus SparkWeightdClientConnect(const char *p,SparkWeightdClient **c,SparkWeightdHelloResult *h) { (void)p;(void)h;if(mode("connect_rpc"))return SPARK_STATUS_IO_ERROR;*c=(SparkWeightdClient *)1;return SPARK_STATUS_OK; }
SparkStatus SparkWeightdClientAttachLazy(SparkWeightdClient *c,const SparkWeightdLazyAttachRequest *q,SparkWeightdLazyAttachResult *r,uint64_t t) { (void)c;(void)t;memset(r,0,sizeof(*r));r->arena_generation=9;if(mode("attach_rpc"))return SPARK_STATUS_IO_ERROR;if(mode("attach_status"))r->status=SPARK_STATUS_IO_ERROR;return q->expert_pool_bytes==4096 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT; }
SparkStatus SparkWeightdClientAcquire(SparkWeightdClient *c,uint64_t g,const SparkWeightdExpertKey *k,uint32_t n,SparkWeightdWorkingSetResult *r,uint64_t t) { (void)c;(void)t;memset(r,0,sizeof(*r));r->lease_identifier=17;printf("ACQUIRE %u %u\n",k[0].layer,n);if(g!=9 || k[0].layer<3)return SPARK_STATUS_INVALID_ARGUMENT;if(mode("acquire_rpc"))return SPARK_STATUS_IO_ERROR;if(mode("acquire_status"))r->status=SPARK_STATUS_IO_ERROR;return SPARK_STATUS_OK; }
SparkStatus SparkWeightdClientRelease(SparkWeightdClient *c,uint64_t g,uint64_t l,SparkWeightdWorkingSetResult *r,uint64_t t) { (void)c;(void)t;printf("RELEASE %llu\n",(unsigned long long)l);if(g!=9 || l!=17)return SPARK_STATUS_INVALID_ARGUMENT;if(mode("release_rpc"))return SPARK_STATUS_IO_ERROR;if(mode("release_status"))r->status=SPARK_STATUS_IO_ERROR;return SPARK_STATUS_OK; }
SparkStatus SparkWeightdClientReclaim(SparkWeightdClient *c,SparkWeightdReclaimResult *r,uint64_t t) { (void)c;(void)t;memset(r,0,sizeof(*r));r->reclaimed_bytes=UINT64_C(8192);r->reclaimed_arena_count=2;r->resident_bytes=UINT64_C(4096);r->arena_count=3;if(mode("reclaim_rpc"))return SPARK_STATUS_IO_ERROR;if(mode("reclaim_status"))r->status=SPARK_STATUS_IO_ERROR;return SPARK_STATUS_OK; }
void SparkWeightdClientClose(SparkWeightdClient *c) { if(c)puts("CLOSED"); }
''')
        cls.warmer = cls.directory / "warmer"
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-D_DARWIN_C_SOURCE",
                        "-Wall", "-Wextra", "-Werror", "-Iinclude", "-I.",
                        "-Imodel-families/dsv4/include",
                        "tools/weightd_warm.c", "runtime/spark_weightd_manifest.c",
                        "model-families/dsv4/src/spark_dsv4_parallel_shape.c",
                        "src/spark_ck128.c",
                        str(faults), "-o", str(cls.warmer)], cwd=ROOT, check=True)
        probe = cls.directory / "latch.c"
        probe.write_text(r'''
#define main UnusedWeightdMain
#include "node/weightd.c"
#undef main
#include <assert.h>
int main(void) {
    struct sockaddr_in addr;
    socklen_t bytes=sizeof(addr);
    int owner=SparkWeightdLatchBind(0), client, accepted;
    assert(owner>=0);
    assert(getsockname(owner,(struct sockaddr *)&addr,&bytes)==0);
    assert(SparkWeightdLatchAcquire(ntohs(addr.sin_port))==-1);
    client=socket(AF_INET,SOCK_STREAM,0);
    assert(client>=0 && connect(client,(struct sockaddr *)&addr,sizeof(addr))==0);
    accepted=accept(owner,0,0);
    assert(accepted>=0 && close(accepted)==0 && close(client)==0);
    assert(close(owner)==0);
    assert(SparkWeightdLatchAcquire(ntohs(addr.sin_port))==1);
    assert(close(spark_weightd_latch_fd)==0);
    assert(SparkWeightdLatchAcquire(0)==-1);
    return 0;
}
''')
        cls.latch = cls.directory / "latch"
        link = ["-Wl,-dead_strip"] if sys.platform == "darwin" else ["-Wl,--gc-sections"]
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-D_DARWIN_C_SOURCE",
                        "-Wall", "-Wextra", "-Werror", "-ffunction-sections",
                        "-fdata-sections", "-I.", "-Iinclude", str(probe),
                        *link, "-o", str(cls.latch)], cwd=ROOT, check=True)
        cls.pack = cls.directory / "pack"
        cls.pack.write_bytes(bytes(64))
        records = [(3, 2), (3, 7), (4, 0)]
        data = struct.pack("<4I", 0x58504557, 2, len(records), 0)
        for index, (layer, expert) in enumerate(records):
            data += struct.pack("<4I2Q16s", layer, expert, 0, 0, index*8, 8, bytes(16))
        Path(str(cls.pack)+".experts").write_bytes(data)
        source = (ROOT / "tools/fleet_node_agent.sh").read_text()
        cls.ensure = "ensure_weightd() {" + source.split("ensure_weightd() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
        cls.ensure = cls.ensure.replace("$HOME", "$TEST_AGENT_HOME")
        cls.loop = source[source.rindex("\nwhile true; do"):]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def warm(self, mode="", budget="4096", extra=()):
        env = dict(os.environ, WARM_FAILURE=mode)
        env.pop("SPARK_WEIGHTD_EXPERT_POOL_BYTES", None)
        if budget is not None:
            env["SPARK_WEIGHTD_EXPERT_POOL_BYTES"] = budget
        return subprocess.run([str(self.warmer), "unused", str(self.pack), "a"*64,
                               "review", "16", *extra], env=env, capture_output=True, text=True)

    def test_manifest_keys_release_before_warm(self):
        result = self.warm()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["ACQUIRE 3 2", "RELEASE 17", "ACQUIRE 4 1", "RELEASE 17", "CLOSED"])
        self.assertEqual(result.stderr.count(" WARM "), 2)

    def test_acquire_and_release_failures_are_terminal(self):
        for mode in ("acquire_rpc", "acquire_status", "release_rpc", "release_status"):
            with self.subTest(mode=mode):
                result = self.warm(mode)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn(" WARM ", result.stderr)
                self.assertEqual(result.stdout.count("CLOSED"), 1)
                self.assertEqual(result.stdout.count("ACQUIRE"), 1)

    def test_budget_and_geometry_cannot_silently_change(self):
        for budget in (None, "", "0", "-1", "18446744073709551615", "4096oops"):
            with self.subTest(budget=budget):
                self.assertEqual(self.warm(budget=budget).returncode, 2)
        self.assertNotEqual(self.warm(extra=("3", "288")).returncode, 0)
        self.assertNotEqual(self.warm(extra=("45", "2")).returncode, 0)

    def wset(self, data):
        path = self.directory / "selected.wset"
        path.write_bytes(data)
        return ("--wset", str(path))

    def test_wset_selected_subset_is_one_acquire_and_release(self):
        extra = self.wset(struct.pack("<6I", 4, 0, 3, 7, 4, 0))
        result = self.warm(extra=extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["ACQUIRE 4 2", "RELEASE 17", "CLOSED"])
        self.assertIn("WSET-ONE-SHOT", result.stderr)
        self.assertIn("WSET-WARM keys=2", result.stderr)

    def test_wset_rejects_invalid_files_before_connecting(self):
        for data in (b"", b"x", bytes(4), struct.pack("<2I", 3, 2)+b"x",
                     struct.pack("<2I", 3, 99), struct.pack("<2I", 40, 0),
                     struct.pack("<2I", 3, 2)*513):
            with self.subTest(data=data[:16], size=len(data)):
                result = self.warm(extra=self.wset(data))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertNotIn("WSET-WARM", result.stderr)
        for path in (self.directory / "missing.wset", self.directory):
            with self.subTest(path=path):
                result = self.warm(extra=("--wset", str(path)))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_wset_rpc_and_release_failures_are_terminal(self):
        extra = self.wset(struct.pack("<2I", 3, 7))
        for mode in ("connect_rpc", "attach_rpc", "attach_status", "acquire_rpc",
                     "acquire_status", "release_rpc", "release_status"):
            with self.subTest(mode=mode):
                result = self.warm(mode=mode, extra=extra)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertNotIn("WSET-WARM", result.stderr)
                self.assertEqual(result.stdout.count("CLOSED"), mode != "connect_rpc")
                self.assertEqual(result.stdout.count("ACQUIRE"),
                                 mode.startswith("acquire") or mode.startswith("release"))

    def test_wset_requires_finite_budget_and_valid_timeout(self):
        extra = self.wset(struct.pack("<2I", 3, 2))
        for budget in (None, "", "0", "-1", "18446744073709551615", "4096oops"):
            with self.subTest(budget=budget):
                self.assertEqual(self.warm(budget=budget, extra=extra).returncode, 2)
        for timeout in ("0", "-1", "oops", "18446744074"):
            with self.subTest(timeout=timeout):
                self.assertEqual(self.warm(extra=(*extra, timeout)).returncode, 2)
        self.assertEqual(self.warm(extra=("--wset",)).returncode, 2)
        self.assertEqual(self.warm(extra=(*extra, "1", "extra")).returncode, 2)

    def reclaim(self, mode="", extra=()):
        env = dict(os.environ, WARM_FAILURE=mode)
        env.pop("SPARK_WEIGHTD_EXPERT_POOL_BYTES", None)
        return subprocess.run([str(self.warmer), "unused", "--reclaim", *extra],
                              env=env, capture_output=True, text=True)

    def test_reclaim_frees_cold_arenas_without_a_pack(self):
        result = self.reclaim()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RECLAIM freed=8192 arenas=2 resident=4096", result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["CLOSED"])

    def test_reclaim_failures_are_terminal(self):
        for mode in ("connect_rpc", "reclaim_rpc", "reclaim_status"):
            with self.subTest(mode=mode):
                result = self.reclaim(mode)
                self.assertEqual(result.returncode, 1)
                self.assertIn("reclaim failed", result.stderr)
                self.assertEqual(result.stdout.count("CLOSED"), mode != "connect_rpc")

    def test_reclaim_requires_exactly_socket_and_flag(self):
        result = self.reclaim(extra=("extra",))
        self.assertEqual(result.returncode, 2)
        self.assertIn("weightd_warm SOCKET --reclaim", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_latch_never_disturbs_existing_owner(self):
        result = subprocess.run([str(self.latch)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def agent(self, changed=False, ready=False, owner="owned", binary=True, restart=True):
        script = r'''
set -u
RANK=0 MESH_INTERFACE=test MESH_SGID_INDEX=3
BACKOFF=(1)
weightd=0
pgrep() { case "$*" in *sparkpipe_weightd*) [ "$TEST_OWNER" != absent ] && echo 4242;; *sparkpipe_model_residentd*) echo 4243;; *) return 1;; esac; }
readlink() { case "$TEST_OWNER" in owned) printf '%s/sparkdata/weightd/sparkpipe_weightd\n' "$TEST_AGENT_HOME";; unknown) echo /other/sparkpipe_weightd;; deleted) echo '/other/sparkpipe_weightd (deleted)';; shell) echo /bin/bash;; esac; }
sha16() { case "$1" in /proc/*) echo running;; *) echo "${TEST_DISK_SHA}";; esac; }
python3() { return "$TEST_PROBE_STATUS"; }
kill() { echo FORBIDDEN_KILL; return 1; }
rm() { echo FORBIDDEN_UNLINK; return 1; }
restart_ok() { return "$TEST_RESTART_STATUS"; }
setsid() { echo SPAWN; }
sleep() { return 0; }
sync_rendezvous() { return 0; }
[() { case "$1" in -S) return 0;; -x) return "$TEST_BINARY_STATUS";; -s) return 1;; esac; builtin [ "$@"; }
''' + self.ensure + '\nensure_weightd\nstatus=$?\nwait\nexit "$status"\n'
        env = dict(os.environ, TEST_DISK_SHA="changed" if changed else "running",
                   TEST_PROBE_STATUS="0" if ready else "1", TEST_OWNER=owner,
                   TEST_BINARY_STATUS="0" if binary else "1",
                   TEST_RESTART_STATUS="0" if restart else "1",
                   TEST_AGENT_HOME=str(self.directory))
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)

    def test_busy_process_is_preserved(self):
        result = self.agent()
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("FORBIDDEN", result.stdout)
        self.assertIn("retained", result.stderr)

    def test_update_waits_for_dependent_engines(self):
        result = self.agent(changed=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("FORBIDDEN", result.stdout)
        self.assertIn("drain", result.stderr)

    def test_owned_ready_daemon_needs_no_restart(self):
        result = self.agent(ready=True)
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("FORBIDDEN", result.stdout)

    def test_unknown_weightd_blocks_cleanup_and_startup(self):
        for owner in ("unknown", "deleted"):
            with self.subTest(owner=owner):
                result = self.agent(owner=owner)
                self.assertEqual(result.returncode, 1)
                self.assertIn("unknown owner", result.stderr)
                self.assertNotIn("FORBIDDEN", result.stdout)
                self.assertNotIn("starting", result.stdout)

    def test_missing_and_backoff_daemon_are_not_ready(self):
        for owner, binary, restart in (("absent", False, True), ("shell", False, True),
                                       ("absent", True, False)):
            with self.subTest(owner=owner, binary=binary, restart=restart):
                result = self.agent(owner=owner, binary=binary, restart=restart)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn("FORBIDDEN", result.stdout)
                self.assertNotIn("starting", result.stdout)

    def test_new_spawn_requires_a_later_readiness_probe(self):
        result = self.agent(owner="shell")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("starting", result.stdout)
        self.assertEqual((self.directory / "weightd.log").read_text(), "SPAWN\n")

    def test_main_loop_gates_every_dependent_action(self):
        script = r'''
set -u
ROOTS=test
sync_core() { :; }
install_core() { :; }
self_update() { :; }
node_doctor() { :; }
janitor() { :; }
ensure_weightd() { return "$TEST_WEIGHTD_STATUS"; }
sync_root() { echo ROOT_SYNC; }
sync_rendezvous() { echo RENDEZVOUS; }
ensure_root() { echo ROOT_START; }
prune_logs() { :; }
ensure_api() { echo API_START; }
warmup_hook() { echo WARMUP; }
report_if_changed() { echo REPORT; }
sleep() { exit 0; }
''' + self.loop
        for ready in (False, True):
            with self.subTest(ready=ready):
                result = subprocess.run(["bash", "-c", script],
                    env=dict(os.environ, TEST_WEIGHTD_STATUS="0" if ready else "1"),
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(),
                    ["ROOT_SYNC", "RENDEZVOUS", "ROOT_START", "API_START", "WARMUP", "REPORT"]
                    if ready else ["REPORT"])


if __name__ == "__main__":
    unittest.main()
