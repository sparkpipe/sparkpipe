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
    "ling_lane.sh": {"LING_LANE": "10", "LING_CODEC": "bf16", "LING_EXPERT_POOL_BYTES": "1", "LING_WEIGHTD_SOCKET": "/run/weightd.sock",
                     "LING_MEMORY_MAX": "12G", "RUN": "LING_RUN_ID", "FIRMWARE": "LING_FIRMWARE"},
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
            self.assertEqual(os.readlink(Path(directory) / "residentd.log"), f"logs/residentd-{RUN_ID}.log")
            self.assertEqual((Path(directory) / f"logs/residentd-before-{RUN_ID}.log").read_text(), "older launcher\n")
            (Path(directory) / f"logs/residentd-{RUN_ID}.log").write_text("first run\n")
            again = self.prelude(directory, RUN_ID)
            self.assertEqual(again.returncode, 2)
            self.assertIn("refusing to overwrite", again.stderr)
            self.assertEqual((Path(directory) / f"logs/residentd-{RUN_ID}.log").read_text(), "first run\n")
            second = self.prelude(directory, "r20261004b")
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(os.readlink(Path(directory) / "residentd.log"), "logs/residentd-r20261004b.log")
            self.assertEqual((Path(directory) / f"logs/residentd-{RUN_ID}.log").read_text(), "first run\n")

    def lane(self, directory, tool, command, extra):
        fake = Path(directory) / "fake"
        log = Path(directory) / "calls"
        if not fake.exists():
            fake.mkdir()
            for name in ("ssh", "scp"):
                (fake / name).write_text(f'#!/bin/sh\nmkdir -p "{log}"\nline=\'{name}\'\nfor a in "$@"; do line="$line [$a]"; done\nprintf \'%s\\n\' "$line" > "{log}/$$"\ncase "$line" in *"tail -1"*) echo "model_residentd ready" ;; *"echo \\$n"*) echo 0 ;; *"[$FAKE_REFUSING_HOST]"*"--deployment"*) exit 2 ;; esac\n')
                (fake / name).chmod(0o755)
        shutil.rmtree(log, ignore_errors=True)
        settings = {key: value for key, value in TOOLS[tool].items() if key not in ("RUN", "FIRMWARE", "ARGS")}
        settings[TOOLS[tool]["FIRMWARE"]] = directory
        environment = {"PATH": f"{fake}:{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": directory, "TMPDIR": directory, **settings, **extra}
        result = subprocess.run(["bash", str(ROOT / "tools" / tool), *TOOLS[tool].get("ARGS", []), *command], capture_output=True, text=True, env=environment)
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
                    self.assertIn(f"ln -sfn logs/residentd-{RUN_ID}.log residentd.log", call)
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
