#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "r20261004a"
TOOLS = {
    "k3_lane.sh": {"K3_LANE": "3", "K3_WEIGHTD_SOCKET": "/run/weightd.sock", "K3_EXPERT_POOL_BYTES": "1", "K3_MEMORY_MAX": "10G",
                   "K3_STATE_BUDGET_BYTES": "1", "RUN": "K3_RUN_ID", "FIRMWARE": "K3_FIRMWARE"},
    "fleet_serve.sh": {"G5_API_HOST": "rtx5090", "RUN": "FLEET_RUN_ID", "FIRMWARE": "FLEET_FIRMWARE_UNUSED", "ARGS": ["glm53_release"]},
}


class LaneRunLog(unittest.TestCase):
    def prelude(self, directory, run_id):
        script = f'LANE_TOOL=t; . "{ROOT}/tools/lane_run_log.sh"; cd "{directory}" && eval "$(lane_log_prelude host {run_id})" && echo started > "$(lane_log_file {run_id})"'
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)

    def test_a_run_log_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "residentd.log").write_text("older launcher\n")
            first = self.prelude(directory, RUN_ID)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertFalse(os.path.lexists(Path(directory) / "residentd.log"))
            self.assertEqual((Path(directory) / f"logs/residentd-before-{RUN_ID}.log").read_text(), "older launcher\n")
            (Path(directory) / f"logs/residentd-{RUN_ID}.log").write_text("first run\n")
            again = self.prelude(directory, RUN_ID)
            self.assertEqual(again.returncode, 2)
            self.assertIn("refusing to overwrite", again.stderr)
            self.assertEqual((Path(directory) / f"logs/residentd-{RUN_ID}.log").read_text(), "first run\n")
            (Path(directory) / "residentd.log").symlink_to(f"logs/residentd-{RUN_ID}.log")
            second = self.prelude(directory, "r20261004b")
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertFalse(os.path.lexists(Path(directory) / "residentd.log"))
            self.assertEqual((Path(directory) / f"logs/residentd-{RUN_ID}.log").read_text(), "first run\n")
            self.assertEqual(sorted(path.name for path in Path(directory).rglob("*") if path.is_symlink()), [])
            os.utime(Path(directory) / f"logs/residentd-{RUN_ID}.log", (1, 1))
            latest = subprocess.run(["bash", "-c", f'. "{ROOT}/tools/lane_run_log.sh"; eval "tail -1 $(lane_latest_log "{directory}")"'], capture_output=True, text=True)
            self.assertEqual(latest.stdout, "started\n", latest.stderr)

    def test_status_tails_the_newest_run_log(self):
        for tool, extra in (("k3_lane.sh", {}), ("stage_lane.sh", {"STAGE_RECORD": "glm-5.3"})):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as directory:
                result, calls = self.lane(directory, tool, ["status"], extra)
                statuses = [call for call in calls if call.startswith("ssh ") and "tail -1" in call]
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(statuses), 16)
                for call in statuses:
                    self.assertIn('tail -1 "$(ls -t ', call)
                    self.assertIn('/root/logs/residentd-*.log 2>/dev/null | head -1)"', call)
                    self.assertNotIn("/residentd.log", call)

    def lane(self, directory, tool, command, extra):
        fake = Path(directory) / "fake"
        log = Path(directory) / "calls"
        if not fake.exists():
            fake.mkdir()
            for name in ("ssh", "scp"):
                (fake / name).write_text(f'#!/bin/sh\nmkdir -p "{log}"\nline=\'{name}\'\nfor a in "$@"; do line="$line [$a]"; done\nprintf \'%s\\n\' "$line" > "{log}/$$"\ncase "$line" in *"tail -1"*) echo "model_residentd ready" ;; *"echo \\$n"*) echo 0 ;; *"[$FAKE_REFUSING_HOST]"*"--deployment"*) exit 2 ;; esac\n')
                (fake / name).chmod(0o755)
        shutil.rmtree(log, ignore_errors=True)
        spec = TOOLS.get(tool, {"RUN": "STAGE_RUN_ID", "FIRMWARE": "STAGE_FIRMWARE"})
        settings = {key: value for key, value in spec.items() if key not in ("RUN", "FIRMWARE", "ARGS")}
        settings[spec["FIRMWARE"]] = directory
        environment = {"PATH": f"{fake}:{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": directory, "TMPDIR": directory, **settings, **extra}
        result = subprocess.run(["bash", str(ROOT / "tools" / tool), *spec.get("ARGS", []), *command], capture_output=True, text=True, env=environment)
        calls = [line for call in sorted(log.glob("*")) for line in call.read_text().splitlines()] if log.exists() else []
        return result, calls

    def test_start_keeps_one_log_per_run_and_archive_collects_it(self):
        for tool, spec in TOOLS.items():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as directory:
                result, calls = self.lane(directory, tool, ["start"], {spec["RUN"]: RUN_ID})
                starts = [call for call in calls if call.startswith("ssh ") and "sparkpipe_model_residentd --deployment" in call]
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(starts), 16)
                for call in starts:
                    self.assertIn(f"> logs/residentd-{RUN_ID}.log", call)
                    self.assertNotRegex(call, r"ln -s\S* \S+ residentd\.log")
                    self.assertIn("refusing to overwrite", call)
                    self.assertNotIn("> residentd.log", call)
                result, calls = self.lane(directory, tool, ["start"], {spec["RUN"]: RUN_ID, "FAKE_REFUSING_HOST": "sparkb"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("sparkb", result.stderr)
                for bad in ("a/b", "x y", "../up"):
                    result, calls = self.lane(directory, tool, ["start"], {spec["RUN"]: bad})
                    self.assertEqual(result.returncode, 2, bad)
                    self.assertEqual(calls, [], bad)
                result, calls = self.lane(directory, tool, ["archive", RUN_ID, str(Path(directory) / "archive")], {})
                copies = [call for call in calls if call.startswith("scp ") and f"logs/residentd-{RUN_ID}.log" in call]
                removals = [call for call in calls if call.startswith("ssh ") and "rm -f" in call and f"logs/residentd-{RUN_ID}.log" in call]
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((len(copies), len(removals)), (16, 16))
                self.assertTrue(any("rank15.log" in call for call in copies))


if __name__ == "__main__":
    unittest.main()
