#!/usr/bin/env python3
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "tools/fleet_node_agent.sh").read_text()


def function(name):
    return name + "() {" + SOURCE.split("\n" + name + "() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"


def constant(name):
    return next(line for line in SOURCE.splitlines() if line.startswith(name + "=")) + "\n"


HARNESS = (r'''
set -uo pipefail
date() { echo 00:00:00; }
journalctl() { [ "$TEST_JOURNAL" = unreadable ] && return 1; cat "$WORK/boots.json"; }
note_root() { echo "NOTE $1 $2 $3"; }
root_dir() { echo "$WORK/root"; }
root_state() { echo down; }
root_pid() { echo 0; }
sha16() { echo none; }
node_uptime_s() { echo "$TEST_UPTIME"; }
root_gate() { return 0; }
restart_ok() { return 0; }
restart_root() { echo "START $1"; }
doctor_port() { echo "DOCTOR $1"; }
declare -A BACKOFF
AGENT_BLOCKED=""
MESH_INTERFACE=switch
MESH_PAIR_INTERFACE=pair
''' + constant("BOOT_SHORT_SECONDS") + constant("BOOT_LOOP_WINDOW_SECONDS") + constant("BOOT_SETTLE_SECONDS") + 'SAFE_MODE=""\nSAFE_CLEAR_FILE="$WORK/clear"\n'
    + "".join(function(name).replace("/proc/sys/kernel/random/boot_id", "$WORK/boot_id")
              for name in ("boot_durations", "check_boot_loop", "safe_mode_active", "ensure_root", "node_doctor")))

MINUTE = 60 * 10**6


class FleetAgentBootBreaker(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        (self.work / "root").mkdir()
        (self.work / "boot_id").write_text("current-boot\n")

    def tearDown(self):
        self.temp.cleanup()

    def boots(self, *minutes, ended_hours_ago=1):
        now = int(time.time() * 10**6)
        end, entries = now - ended_hours_ago * 3600 * 10**6, []
        for index, length in enumerate(minutes):
            entries.append({"index": -(index + 1), "boot_id": "b%d" % index, "first_entry": end - length * MINUTE, "last_entry": end})
            end -= length * MINUTE + 2 * MINUTE
        entries.append({"index": 0, "boot_id": "current", "first_entry": now - MINUTE, "last_entry": now})
        (self.work / "boots.json").write_text(json.dumps(list(reversed(entries))))

    def run_steps(self, steps, uptime=3600, journal="readable"):
        environment = {"WORK": str(self.work), "TEST_UPTIME": str(uptime), "TEST_JOURNAL": journal,
                       "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        result = subprocess.run(["bash", "-c", HARNESS + steps], env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_two_short_boots_of_the_last_three_enter_safe_mode(self):
        self.boots(11, 60, 5, 2)
        result = self.run_steps('check_boot_loop; echo "MODE=$SAFE_MODE"; ensure_root test; node_doctor')
        self.assertIn("SAFE MODE: 2 of the last 3 boots lasted under 20 min and ended within 6 h (660, 3600, 300 s)", result.stderr)
        self.assertIn("NOTE test safe-mode 2 of the last 3 boots", result.stdout)
        self.assertNotIn("START", result.stdout)
        self.assertNotIn("DOCTOR", result.stdout)

    def test_one_short_boot_is_a_normal_boot(self):
        self.boots(11, 3000, 25, 2)
        result = self.run_steps('check_boot_loop; echo "MODE=[$SAFE_MODE]"; ensure_root test; node_doctor')
        self.assertIn("MODE=[]", result.stdout)
        self.assertIn("START test", result.stdout)
        self.assertIn("DOCTOR switch", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_short_boots_from_an_old_loop_do_not_count(self):
        self.boots(3, 3, 3, ended_hours_ago=7)
        self.assertIn("MODE=[]", self.run_steps('check_boot_loop; echo "MODE=[$SAFE_MODE]"').stdout)
        self.boots(30000, 12, 10, ended_hours_ago=0)
        self.assertIn("MODE=[]", self.run_steps('check_boot_loop; echo "MODE=[$SAFE_MODE]"').stdout)

    def test_only_the_three_previous_boots_count(self):
        self.boots(3000, 3000, 3000, 2, 2, 2)
        self.assertIn("MODE=[]", self.run_steps('check_boot_loop; echo "MODE=[$SAFE_MODE]"').stdout)

    def test_an_operator_clears_safe_mode_for_this_boot_only(self):
        self.boots(2, 2, 2)
        (self.work / "clear").write_text("an-older-boot\n")
        self.assertNotIn("START", self.run_steps("check_boot_loop; ensure_root test").stdout)
        result = self.run_steps("check_boot_loop; ensure_root test; cp $WORK/boot_id $WORK/clear; ensure_root test")
        self.assertIn("safe mode cleared for this boot", result.stdout)
        self.assertEqual(result.stdout.count("START test"), 1, result.stdout)

    def test_engines_wait_five_minutes_after_boot(self):
        self.boots(3000, 3000, 3000)
        self.assertIn("autospawn waits", self.run_steps("check_boot_loop; ensure_root test", uptime=299).stdout)
        self.assertIn("START test", self.run_steps("check_boot_loop; ensure_root test", uptime=300).stdout)

    def test_unreadable_boot_history_is_reported(self):
        result = self.run_steps('check_boot_loop; echo "MODE=[$SAFE_MODE]"', journal="unreadable")
        self.assertIn("boot history unreadable", result.stderr)
        self.assertIn("MODE=[]", result.stdout)


if __name__ == "__main__":
    unittest.main()
