#!/usr/bin/env python3
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "tools/fleet_node_agent.sh").read_text()
NAMES = ("working_state_tool_present", "prepare_working_state", "stop_working_state", "ensure_working_state", "self_update")
CONSTANTS = ("MESH_LINK_MAP", "WORKING_STATE_DIR", "WORKING_STATE_TOOL", "WORKING_STATE_PIDFILE", "WORKING_STATE_MISSING",
             "WORKING_STATE_PREPARED", "WORKING_STATE_SHA")
MAIN_LOOP = SOURCE[SOURCE.rindex("\nwhile true; do\n"):]


def function(name):
    return name + "() {" + SOURCE.split("\n" + name + "() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"


def constant(name):
    return next(line for line in SOURCE.splitlines() if line.startswith(name + "=")).replace("$HOME", "$WORK/home") + "\n"


LINK_MAP = constant("MESH_LINK_MAP").split("=", 1)[1].strip().strip('"')
REBUILD = f"rebuild --rank 3 --link-map {LINK_MAP}"

HARNESS = r'''
set -uo pipefail
RANK=3
MESH_PAIRED=${TEST_PAIRED-1}
ROOTS=test
HUB=spec@hub
START_SHA=old
date() { echo 00:00:00; }
sha16() { if [ "$1" = "$WORKING_STATE_TOOL" ]; then cksum < "$1" | cut -d' ' -f1; else echo new; fi; }
stop_mesh_exchange() { :; }
''' + "".join(constant(name) for name in CONSTANTS) + "".join(function(name).replace("$HOME", "$WORK/home") for name in NAMES)

FAKE_TOOL = r'''#!/usr/bin/env bash
echo "$*" >> "$WORK/calls"
case "$1" in
    rebuild)
        sleep "${REBUILD_SECONDS:-0}"
        [ "${REBUILD_STATUS:-0}" = 0 ] && mkdir -p "$WORK/home/sparkdata/working-state/mirror"
        echo REBUILT >> "$WORK/calls"
        exit "${REBUILD_STATUS:-0}" ;;
    init)
        mkdir -p "$WORK/home/sparkdata/working-state/mirror" ;;
    loop)
        exec > /dev/null 2>&1
        trap 'echo LOOP-TERM >> "$WORK/calls"; exit 0' TERM
        for tick in $(seq 1 100); do sleep 0.1; done ;;
esac
'''

NEW_AGENT = r'''#!/usr/bin/env bash
sleep 0.5
[ -f "$WORK/home/.fleet_agent_working_state.pid" ] && echo PIDFILE-LEFT >> "$WORK/calls"
echo "NEW-AGENT $*" >> "$WORK/calls"
'''


class FleetAgentWorkingState(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        self.core = self.work / "home/sparkdata/core/bin"
        self.core.mkdir(parents=True)
        self.tool = self.core / "working_state_mirror.py"
        self.install_tool()

    def install_tool(self):
        self.tool.write_text(FAKE_TOOL)
        self.tool.chmod(0o755)

    def rebuilt(self):
        (self.work / "home/sparkdata/working-state/mirror").mkdir(parents=True)

    def run_steps(self, steps, **env):
        environment = {"WORK": str(self.work), "HOME": str(self.work / "home"), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        environment.update(env)
        (self.work / "calls").write_text("")
        result = subprocess.run(["bash", "-c", HARNESS + steps], env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result, (self.work / "calls").read_text().splitlines()

    def test_a_fresh_disk_is_rebuilt_before_the_agent_goes_on(self):
        result, calls = self.run_steps('prepare_working_state; echo AFTER >> "$WORK/calls"', REBUILD_SECONDS="0.5")
        self.assertEqual(calls, [REBUILD, "REBUILT", "AFTER"])
        self.assertIn("rebuilding from rank 2 before engines start", result.stdout)

    def test_a_rebuilt_disk_is_not_rebuilt_again(self):
        self.rebuilt()
        _, calls = self.run_steps("prepare_working_state")
        self.assertEqual(calls, [])

    def test_a_failed_rebuild_leaves_the_retry_to_the_loop(self):
        result, calls = self.run_steps("prepare_working_state", REBUILD_STATUS="1")
        self.assertEqual(calls, [REBUILD, "REBUILT"])
        self.assertIn("rebuild exited 1; the mirror loop rebuilds before it pushes", result.stderr)

    def test_an_unpaired_rank_gets_a_scratch_area_and_no_loop(self):
        result, calls = self.run_steps('ensure_working_state; ensure_working_state; if [ -f "$WORKING_STATE_PIDFILE" ]; then echo PIDFILE; fi',
                                       TEST_PAIRED="")
        self.assertEqual(calls, ["init"])
        self.assertEqual(result.stdout.count("rank 3 has no cabled pair in MESH_LINK_MAP"), 1, result.stdout)
        self.assertNotIn("PIDFILE", result.stdout)

    def test_one_loop_runs_per_agent_and_a_lost_one_is_replaced(self):
        self.rebuilt()
        steps = r'''
echo "AGENT $$"
ensure_working_state
first=$(cat "$WORKING_STATE_PIDFILE")
sleep 0.3
ensure_working_state
[ "$(cat "$WORKING_STATE_PIDFILE")" = "$first" ] && echo SAME
kill -KILL "$first"
wait "$first" 2>/dev/null
ensure_working_state
second=$(cat "$WORKING_STATE_PIDFILE")
[ "$second" != "$first" ] && echo REPLACED
sleep 0.3
stop_working_state
kill -0 "$second" 2>/dev/null && echo STILL-ALIVE
[ -f "$WORKING_STATE_PIDFILE" ] && echo PIDFILE-LEFT
echo DONE
'''
        result, calls = self.run_steps(steps)
        agent = result.stdout.split("AGENT ", 1)[1].split()[0]
        self.assertIn("SAME", result.stdout)
        self.assertIn("REPLACED", result.stdout)
        self.assertNotIn("STILL-ALIVE", result.stdout)
        self.assertNotIn("PIDFILE-LEFT", result.stdout)
        self.assertIn("DONE", result.stdout)
        loop = f"loop --rank 3 --link-map {LINK_MAP} --agent-pid {agent}"
        self.assertEqual(calls, [loop, loop, "LOOP-TERM"])
        self.assertEqual(result.stderr.count("is gone; starting another"), 1, result.stderr)

    def test_a_changed_tool_restarts_the_loop(self):
        self.rebuilt()
        steps = r'''
ensure_working_state
first=$(cat "$WORKING_STATE_PIDFILE")
sleep 0.3
ensure_working_state
printf '\n' >> "$WORKING_STATE_TOOL"
ensure_working_state
second=$(cat "$WORKING_STATE_PIDFILE")
[ "$second" != "$first" ] && echo RESTARTED
sleep 0.3
stop_working_state
'''
        result, calls = self.run_steps(steps)
        self.assertIn("RESTARTED", result.stdout)
        self.assertEqual(result.stdout.count("working_state_mirror.py changed"), 1, result.stdout)
        self.assertEqual([call.split()[0] for call in calls], ["loop", "LOOP-TERM", "loop", "LOOP-TERM"])

    def test_a_missing_tool_is_reported_once_and_picked_up_when_it_arrives(self):
        self.tool.unlink()
        result, calls = self.run_steps('ensure_working_state; ensure_working_state; if [ -f "$WORKING_STATE_PIDFILE" ]; then echo PIDFILE; fi')
        self.assertEqual(result.stderr.count("working_state_mirror.py is missing"), 1, result.stderr)
        self.assertNotIn("PIDFILE", result.stdout)
        self.assertEqual(calls, [])
        self.install_tool()
        _, calls = self.run_steps("ensure_working_state; sleep 0.3; stop_working_state")
        self.assertEqual([call.split()[0] for call in calls], ["rebuild", "REBUILT", "loop", "LOOP-TERM"])

    def test_a_self_update_stops_the_loop_before_the_new_agent_runs(self):
        self.rebuilt()
        (self.core / "fleet_node_agent.sh").write_text(NEW_AGENT)
        _, calls = self.run_steps("ensure_working_state; sleep 0.3; self_update")
        self.assertEqual([call.split()[0] for call in calls], ["loop", "LOOP-TERM", "NEW-AGENT"])
        self.assertEqual(calls[-1], "NEW-AGENT test spec@hub")

    def test_the_pass_that_brings_the_tool_rebuilds_before_any_engine_starts(self):
        self.tool.rename(self.work / "published_tool")
        steps = r'''
load_roots() { ROOT_LIST=test; }
sync_core() { [ -x "$WORKING_STATE_TOOL" ] || mv "$WORK/published_tool" "$WORKING_STATE_TOOL"; }
install_core() { :; }
node_doctor() { :; }
janitor() { :; }
ensure_weightd() { return 0; }
sync_root() { :; }
ensure_root() { echo ROOT_START >> "$WORK/calls"; }
prune_logs() { :; }
warmup_hook() { :; }
report_if_changed() { :; }
sleep() { [ "$1" = 1 ] || { command sleep "$@"; return; }; stop_working_state; exit 0; }
''' + MAIN_LOOP
        _, calls = self.run_steps(steps, REBUILD_SECONDS="0.5")
        self.assertEqual(calls[:2], [REBUILD, "REBUILT"])
        self.assertEqual(sorted(call.split()[0] for call in calls[2:]), ["LOOP-TERM", "ROOT_START", "loop"])


if __name__ == "__main__":
    unittest.main()
