#!/usr/bin/env python3
import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "fleet_ready_poll.sh"
FAKE_SSH = """#!/bin/sh
cd "$VIEW_DIR" || exit 1
shift $(($# - 1))
exec sh -c "$1"
"""


def heartbeats(directory, degraded=None, ports="rocep1s0f1", ready=16):
    current = Path(directory) / "current"
    current.mkdir()
    for index in range(16):
        host = "spark%x" % index
        beat = {"host": host, "mesh_ports": ports, "mesh_degraded": (degraded or {}).get(host, "0x0000"),
                "roots": {"lane": {"state": "ready" if index < ready else "starting"}}}
        (current / (host + ".json")).write_text(json.dumps(beat))


class FleetReadyPollTest(unittest.TestCase):
    def poll(self, **fixture):
        with tempfile.TemporaryDirectory() as directory:
            heartbeats(directory, **fixture)
            ssh = Path(directory) / "ssh"
            ssh.write_text(FAKE_SSH)
            ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)
            sleep = Path(directory) / "sleep"
            sleep.write_text("#!/bin/sh\nexit 0\n")
            sleep.chmod(sleep.stat().st_mode | stat.S_IEXEC)
            env = dict(os.environ, PATH=f"{directory}:{os.environ['PATH']}", VIEW_DIR=directory)
            return subprocess.run(["bash", str(SCRIPT), "lane"], capture_output=True, text=True, env=env, timeout=120)

    def test_a_fleet_on_one_port_is_ready_and_says_so(self):
        result = self.poll()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FLEET-READY t=", result.stdout)
        self.assertIn("mesh ports: 16 on rocep1s0f1", result.stdout)

    def test_a_degraded_link_is_never_presented_as_plain_ready(self):
        result = self.poll(degraded={"spark4": "0x0030"}, ports="rocep1s0f1+roceP2p1s0f0")
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertIn("FLEET-READY-DEGRADED", result.stdout)
        self.assertIn("degraded links (host:ranks): spark4:0x0030", result.stdout)
        self.assertNotIn("FLEET-READY t=", result.stdout)


if __name__ == "__main__":
    unittest.main()
