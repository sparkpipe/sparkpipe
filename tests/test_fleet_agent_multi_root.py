import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
AGENT = (ROOT / "tools/fleet_node_agent.sh").read_text()
HEAD = AGENT[:AGENT.index('\necho "$$" > "$PID_FILE"')] + "\n"
PROD = "prod.fp8.tp16"
DEV = "dev-l4"
PACK_SHA = "ab" * 32

STUB = r'''
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
static volatile sig_atomic_t stop;
static void on_term(int s) { (void)s; stop = 1; }
int main(int argc, char **argv) {
    static const char *keys[] = {"SPARK_WEIGHTD_LANE", "SPARK_TP_MESH_RANKS",
        "SPARK_WEIGHTD_PACK_SHA256", "SPARK_WEIGHTD_EXPERT_POOL_BYTES",
        "LD_LIBRARY_PATH", "TEST_MODEL_SWITCH", 0};
    FILE *f = fopen("launch.txt", "w");
    signal(SIGTERM, on_term);
    if (!f) return 1;
    fprintf(f, "PID %d\n", (int)getpid());
    for (int i = 1; i < argc; i++) fprintf(f, "ARG %s\n", argv[i]);
    for (int i = 0; keys[i]; i++) fprintf(f, "ENV %s=%s\n", keys[i], getenv(keys[i]) ? getenv(keys[i]) : "<unset>");
    fclose(f);
    printf("model_residentd ready\n");
    fflush(stdout);
    while (!stop) pause();
    return 0;
}
'''

SYSTEMD_RUN = r'''#!/usr/bin/env bash
printf '%s\n' "$*" >> "$TEST_LOG_DIR/systemd-run.log"
[ "${TEST_SYSTEMD_RUN_STATUS:-0}" = 0 ] || exit "$TEST_SYSTEMD_RUN_STATUS"
wd=. out=/dev/null
envs=()
while [ $# -gt 0 ]; do
    case "$1" in
        --user|--collect) shift ;;
        --unit=*) shift ;;
        --working-directory=*) wd="${1#*=}"; shift ;;
        --setenv=*) envs+=("${1#--setenv=}"); shift ;;
        -p) case "$2" in StandardOutput=append:*) out="${2#StandardOutput=append:}" ;; esac; shift 2 ;;
        *) break ;;
    esac
done
cd "$wd" || exit 1
setsid env -i PATH="$PATH" "${envs[@]}" "$@" >> "$out" 2>&1 < /dev/null &
exit 0
'''

PRELUDE = r'''
hostname() { echo sparka; }
ssh() { return 0; }
scp() { return 0; }
ssh-keyscan() { return 0; }
curl() { return 1; }
source "$TEST_AGENT_HEAD" "$TEST_ROOTS" sparkf
mem_available_gib() { cat "$TEST_MEM"; }
restart_ok() { return 0; }
sleep() { command sleep 0.05; }
wait_ready() {
    local i
    for i in $(seq 1 200); do
        [ "$(root_state "$1")" = ready ] && [ -s "$(root_path "$1")/launch.txt" ] && return 0
        command sleep 0.02
    done
    echo "TIMEOUT waiting for $1" >&2
    return 1
}
wait_gone() {
    local i
    for i in $(seq 1 200); do
        [ "$(root_pid "$1")" = 0 ] && return 0
        command sleep 0.02
    done
    return 1
}
'''


def linux_bash4():
    if not sys.platform.startswith("linux") or shutil.which("cc") is None:
        return False
    out = subprocess.run(["bash", "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True)
    return out.stdout.strip().isdigit() and int(out.stdout.strip()) >= 4


@unittest.skipUnless(linux_bash4(), "the multi-root agent test needs Linux /proc, bash >= 4 and cc")
class FleetAgentMultiRoot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="agent-multi-root-build-")
        source = Path(cls.build.name) / "stub.c"
        source.write_text(STUB)
        cls.stub = Path(cls.build.name) / "sparkpipe_model_residentd"
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-Wall", "-Werror", str(source),
                        "-o", str(cls.stub)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="agent-multi-root-")
        self.home = Path(self.tmp.name).resolve()
        self.bin = self.home / "testbin"
        self.bin.mkdir()
        (self.bin / "systemd-run").write_text(SYSTEMD_RUN)
        (self.bin / "systemctl").write_text("#!/bin/sh\nexit 0\n")
        for tool in ("systemd-run", "systemctl"):
            (self.bin / tool).chmod(0o755)
        (self.home / "agent_head.sh").write_text(HEAD)
        self.mem = self.home / "mem"
        self.mem.write_text("100\n")
        self.prod = self.home / "sparkdata" / PROD
        self.make_prod()
        self.dev = self.home / "lanes" / DEV / "root"
        self.make_dev()
        (self.home / ".fleet_agent_roots").write_text(DEV + "\n")

    def tearDown(self):
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cwd = os.readlink(entry / "cwd")
            except OSError:
                continue
            if cwd.startswith(str(self.home)) and int(entry.name) != os.getpid():
                try:
                    os.kill(int(entry.name), signal.SIGKILL)
                except OSError:
                    pass
        self.tmp.cleanup()

    def make_prod(self):
        (self.prod / "bin").mkdir(parents=True)
        (self.prod / "config").mkdir()
        (self.prod / "lib").mkdir()
        shutil.copy2(self.stub, self.prod / "bin" / "sparkpipe_model_residentd")
        (self.prod / "model_resident.json").write_text("{}\n")
        for index in range(16):
            (self.prod / "config" / f"stage_{index:02d}.json").write_text("{}\n")

    def make_dev(self):
        release = self.home / "lanes" / DEV / "release"
        (release / "bin").mkdir(parents=True)
        shutil.copy2(self.stub, release / "bin" / "sparkpipe_model_residentd")
        for sub in ("bin", "config", "lib"):
            (self.dev / sub).mkdir(parents=True)
        (self.dev / "bin" / "sparkpipe_model_residentd").symlink_to(release / "bin" / "sparkpipe_model_residentd")
        (self.dev / "model_resident.json").write_text("{}\n")
        for index in range(4):
            (self.dev / "config" / f"stage_{index:02d}.json").write_text("{}\n")
        (self.dev / "config" / "rank_index_10").write_text("2\n")
        (self.dev / "config" / "env_10.env").write_text(f"SPARK_WEIGHTD_PACK_SHA256={PACK_SHA}\n")
        self.write_agent_env()
        (self.home / "sparkdata" / DEV).symlink_to(self.dev)

    def write_agent_env(self, extra="", memory_max="AGENT_MEMORY_MAX=40G\n", need="AGENT_MEMORY_NEED_GIB=36\n"):
        (self.dev / "agent.env").write_text(
            "SPARK_WEIGHTD_LANE=4\nSPARK_TP_MESH_RANKS=10,11,12,13\nTEST_MODEL_SWITCH=a b=c\n"
            "AGENT_SYNC=local\n" + memory_max + need + extra)

    def run_agent(self, body, roots=PROD):
        env = dict(os.environ, HOME=str(self.home), TEST_AGENT_HEAD=str(self.home / "agent_head.sh"),
                   TEST_ROOTS=roots, TEST_MEM=str(self.mem), TEST_LOG_DIR=str(self.home),
                   PATH=str(self.bin) + os.pathsep + os.environ["PATH"])
        for key in ("FLEET_AGENT_ROOTS_FILE", "FLEET_AGENT_HEADROOM_GIB", "G5_PIN_EXPERTS",
                    "G5_GRAPH_PATH", "G5_LAUNCH_BLOCKING", "G5_EXPERT_POOL_BYTES"):
            env.pop(key, None)
        return subprocess.run(["bash", "-c", PRELUDE + body], env=env, capture_output=True,
                              text=True, cwd=str(self.home), timeout=120)

    def launch(self, root):
        lines = (root / "launch.txt").read_text().splitlines()
        args = [line[4:] for line in lines if line.startswith("ARG ")]
        env = dict(line[4:].split("=", 1) for line in lines if line.startswith("ENV "))
        pid = int(lines[0].split()[1])
        return pid, args, env

    def heartbeat(self):
        return json.loads((self.home / "current" / "sparka.json").read_text())

    def start_both(self):
        result = self.run_agent(f'''
ROOT_LIST={DEV}
start_root {DEV} && wait_ready {DEV} || exit 3
load_roots
start_root {PROD} && wait_ready {PROD} || exit 4
report
echo "PIDS $(root_pid {PROD}) $(root_pid {DEV})"
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        line = [x for x in result.stdout.splitlines() if x.startswith("PIDS ")][-1]
        return [int(x) for x in line.split()[1:]]

    def test_roots_run_side_by_side_with_their_own_rank_index_env_and_stage(self):
        prod_pid, dev_pid = self.start_both()
        pid, args, env = self.launch(self.prod)
        self.assertEqual(pid, prod_pid)
        self.assertEqual(args, ["--deployment", "model_resident.json", "--rank-index", "10"])
        self.assertEqual(env["SPARK_WEIGHTD_EXPERT_POOL_BYTES"], "34359738368")
        self.assertEqual(os.readlink(self.prod / "config" / "stage.json"), "stage_10.json")
        pid, args, env = self.launch(self.dev)
        self.assertEqual(pid, dev_pid)
        self.assertLess(dev_pid, prod_pid)
        self.assertEqual(args, ["--deployment", "model_resident.json", "--rank-index", "2"])
        self.assertEqual(env["SPARK_WEIGHTD_LANE"], "4")
        self.assertEqual(env["SPARK_TP_MESH_RANKS"], "10,11,12,13")
        self.assertEqual(env["SPARK_WEIGHTD_PACK_SHA256"], PACK_SHA)
        self.assertEqual(env["TEST_MODEL_SWITCH"], "a b=c")
        self.assertEqual(env["SPARK_WEIGHTD_EXPERT_POOL_BYTES"], "<unset>")
        self.assertEqual(env["LD_LIBRARY_PATH"], str(self.dev / "lib"))
        self.assertEqual(os.readlink(self.dev / "config" / "stage.json"), "stage_02.json")
        units = (self.home / "systemd-run.log").read_text().splitlines()
        self.assertEqual(len(units), 1)
        self.assertIn("--unit=sp-agent-dev-l4", units[0])
        self.assertIn("-p MemoryMax=40G -p MemorySwapMax=0", units[0])
        self.assertIn("--working-directory=" + str(self.dev), units[0])

    def test_lower_pid_dev_residentd_never_recycles_production(self):
        prod_pid, dev_pid = self.start_both()
        self.assertLess(dev_pid, prod_pid)
        result = self.run_agent(f'''
for i in 1 2 3; do ensure_root {PROD}; ensure_root {DEV}; done
echo "STATES $(root_state {PROD}) $(root_state {DEV})"
echo "PIDS $(root_pid {PROD}) $(root_pid {DEV})"
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("STATES ready ready", result.stdout)
        self.assertIn(f"PIDS {prod_pid} {dev_pid}", result.stdout)
        self.assertNotIn("starting", result.stdout)

    def test_heartbeat_lists_every_root_with_state_pid_and_shas(self):
        (self.home / ".fleet_agent_roots").write_text(DEV + "\ngone.root\n")
        prod_pid, dev_pid = self.start_both()
        roots = self.heartbeat()["roots"]
        self.assertEqual(list(roots), [PROD, DEV, "gone.root"])
        self.assertEqual((roots[PROD]["state"], roots[PROD]["pid"]), ("ready", prod_pid))
        self.assertEqual((roots[DEV]["state"], roots[DEV]["pid"]), ("ready", dev_pid))
        self.assertEqual((roots["gone.root"]["state"], roots["gone.root"]["pid"]), ("missing", 0))
        for name in (PROD, DEV):
            self.assertNotEqual(roots[name]["residentd"], "none")
            self.assertEqual(roots[name]["running"], roots[name]["residentd"])
        self.assertEqual((roots[PROD]["role"], roots[PROD]["rank_index"], roots[PROD]["unit"]),
                         ("production", 10, "fleet-agent"))
        self.assertEqual((roots[DEV]["role"], roots[DEV]["rank_index"], roots[DEV]["unit"]),
                         ("dev", 2, "sp-agent-dev-l4"))
        self.assertEqual((roots[DEV]["memory_max"], roots[DEV]["need_gib"]), ("40G", "36"))
        self.assertEqual(roots[PROD]["env"], "none")
        self.assertNotEqual(roots[DEV]["env"], "none")
        self.assertEqual(self.heartbeat()["headroom_gib"], 20)

    def test_headroom_guard_blocks_dev_roots_and_never_production(self):
        self.mem.write_text("55\n")
        result = self.run_agent(f'''
start_root {PROD} && wait_ready {PROD} || exit 4
ensure_root {DEV}
echo "STATE $(root_state {DEV})"
report
echo 1 > "$TEST_MEM"
ensure_root {PROD}
echo "PROD $(root_state {PROD})"
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("STATE blocked-headroom", result.stdout)
        self.assertIn("PROD ready", result.stdout)
        self.assertFalse((self.dev / "launch.txt").exists())
        self.assertFalse((self.home / "systemd-run.log").exists())
        dev = self.heartbeat()["roots"][DEV]
        self.assertEqual(dev["state"], "blocked-headroom")
        self.assertIn("needs 36 GiB, MemAvailable 55 GiB, headroom 20 GiB", dev["reason"])
        self.mem.write_text("56\n")
        result = self.run_agent(f'''
start_root {PROD} && wait_ready {PROD} || exit 4
ensure_root {DEV} && wait_ready {DEV} || exit 5
echo "STATE $(root_state {DEV})"
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("STATE ready", result.stdout)

    def test_production_start_ignores_the_headroom_guard(self):
        self.mem.write_text("0\n")
        result = self.run_agent(f'''
start_root {PROD} && wait_ready {PROD} || exit 4
echo "STATE $(root_state {PROD})"
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("STATE ready", result.stdout)

    def test_dev_root_waits_for_production(self):
        result = self.run_agent(f'''
ensure_root {DEV}
echo "STATE $(root_state {DEV})"
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("STATE waiting-production", result.stdout)
        self.assertFalse((self.dev / "launch.txt").exists())

    def test_draining_one_root_leaves_the_other_running(self):
        prod_pid, dev_pid = self.start_both()
        result = self.run_agent(f'''
drain_root {DEV}
wait_gone {DEV} || exit 3
janitor
echo "PIDS $(root_pid {PROD}) $(root_pid {DEV})"
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"PIDS {prod_pid} 0", result.stdout)
        os.kill(prod_pid, 0)
        with self.assertRaises(ProcessLookupError):
            os.kill(dev_pid, 0)

    def test_production_restart_stops_dev_roots_when_memory_is_short(self):
        prod_pid, dev_pid = self.start_both()
        result = self.run_agent(f'''
mem_available_gib() {{ [ "$(root_pid {DEV})" = 0 ] && echo 100 || echo 0; }}
unload_root {PROD} || exit 3
echo "STATE $(root_state {DEV})"
ensure_root {DEV}
echo "AFTER $(root_state {DEV})"
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("STATE yielded", result.stdout)
        self.assertIn("AFTER waiting-production", result.stdout)
        self.assertIn("stopping dev root to give memory to production root", result.stderr)
        for pid in (prod_pid, dev_pid):
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_hold_file_stops_a_running_dev_root_and_keeps_it_down(self):
        prod_pid, dev_pid = self.start_both()
        (self.dev / "agent.hold").write_text("")
        result = self.run_agent(f'''
ensure_root {DEV}
wait_gone {DEV} || exit 3
ensure_root {DEV}
echo "STATE $(root_state {DEV}) $(root_pid {PROD})"
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"STATE held {prod_pid}", result.stdout)
        with self.assertRaises(ProcessLookupError):
            os.kill(dev_pid, 0)

    def test_invalid_dev_layouts_are_refused_with_a_reason(self):
        cases = {
            "memory max": (lambda: self.write_agent_env(memory_max=""), "needs AGENT_MEMORY_MAX"),
            "need": (lambda: self.write_agent_env(need=""), "needs AGENT_MEMORY_NEED_GIB"),
            "unknown key": (lambda: self.write_agent_env(extra="AGENT_MEMORYMAX=4G\n"), "unknown agent key"),
            "bad size": (lambda: self.write_agent_env(memory_max="AGENT_MEMORY_MAX=lots\n"), "systemd size"),
            "role": (lambda: self.write_agent_env(extra="AGENT_ROLE=prod\n"), "AGENT_ROLE"),
            "line": (lambda: self.write_agent_env(extra="export X=1\n"), "not KEY=VALUE"),
            "index": (lambda: (self.dev / "config" / "rank_index_10").write_text("two\n"), "rank index"),
            "stage": (lambda: (self.dev / "config" / "rank_index_10").write_text("7\n"), "stage_07.json missing"),
        }
        for label, (mutate, reason) in cases.items():
            with self.subTest(case=label):
                self.write_agent_env()
                (self.dev / "config" / "rank_index_10").write_text("2\n")
                mutate()
                result = self.run_agent(f'''
start_root {PROD} && wait_ready {PROD} || exit 4
ensure_root {DEV}
echo "STATE $(root_state {DEV})"
report
''')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("STATE blocked-config", result.stdout)
                self.assertIn(reason, self.heartbeat()["roots"][DEV]["reason"])
                self.assertFalse((self.dev / "launch.txt").exists())

    def test_failed_unit_start_is_reported(self):
        result = self.run_agent(f'''
start_root {PROD} && wait_ready {PROD} || exit 4
export TEST_SYSTEMD_RUN_STATUS=1
ensure_root {DEV}
echo "STATE $(root_state {DEV})"
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("STATE failed-start", result.stdout)

    def test_roots_file_is_merged_deduplicated_and_validated(self):
        (self.home / ".fleet_agent_roots").write_text(f"{DEV}\n\n{PROD}\n../escape\n*\nextra.root\n")
        result = self.run_agent('load_roots\necho "LIST $ROOT_LIST"\n', roots=f"{PROD},{DEV}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"LIST {PROD} {DEV} extra.root", result.stdout)
        self.assertIn("ignoring invalid root names: ../escape *", result.stderr)

    def test_layout_file_changes_restart_the_root(self):
        result = self.run_agent('''
for rel in agent.env config/env_10.env config/rank_index_10 config/stage_03.json; do
    echo "$rel" > "$HOME/scope"
    echo "SCOPE $rel $(restart_scope "$HOME/scope")"
done
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SCOPE agent.env root", result.stdout)
        self.assertIn("SCOPE config/env_10.env root", result.stdout)
        self.assertIn("SCOPE config/rank_index_10 root", result.stdout)
        self.assertIn("SCOPE config/stage_03.json stage", result.stdout)

    def test_local_roots_never_fetch_a_release(self):
        result = self.run_agent(f'''
curl() {{ echo "CURL $*"; return 1; }}
sync_root {DEV}
sync_root {PROD}
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        fetched = [line for line in result.stdout.splitlines() if line.startswith("CURL ")]
        self.assertEqual(len(fetched), 1)
        self.assertIn(f"/{PROD}/MANIFEST", fetched[0])


if __name__ == "__main__":
    unittest.main()
