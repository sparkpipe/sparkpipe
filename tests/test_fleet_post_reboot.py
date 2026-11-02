import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "fleet_post_reboot.sh"

FAKE_SSH = r'''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
command = args[-1]
host = args[-2]
fail = set(filter(None, os.environ.get("FIXTURE_FAIL", "").split(",")))
rank = os.environ.get("FIXTURE_RANK", "3")
def out(text, code=0):
    sys.stdout.write(text)
    sys.exit(code)
if command == "true":
    out("")
if "test -x" in command:
    out("", 1)
if "cut -d. -f1 /proc/uptime" in command and "nvidia-smi" not in command:
    out("5000\n")
if "skip=1080" in command:
    out("53ef 53ef\n")
if "dmesg" in command:
    out("[  12.0] mlx5_core 0000:01:00.1: mlx5_pci_err_detected Device state = 1\n" if "kernel" in fail else "")
if "systemctl is-system-running" in command and "settled" in command:
    out("settled\n")
if "spark-dma-guard.service" in command:
    out("active watch-only\n" if "guard" in fail else "active armed\n")
if "get-default" in command:
    out("graphical.target starting\n" if "target" in fail else "multi-user.target degraded\n")
if "ds4-node-rank" in command:
    out(rank + "\n")
if "ds4-uplink-wired" in command:
    out("0\n0\n" if "uplink" in fail else "1\n2\n" if "tworoutes" in fail else "1\n1\n")
if "ds4-speculation-link" in command:
    out("0\n0\n" if "hublink" in fail else "1\n1\n")
if ":2222" in command:
    out("0\n" if "emergency" in fail else "1\n")
if "-C earlyoom" in command:
    out("/usr/bin/earlyoom -m 7 -s 25 -r 5 --avoid sshd|systemd|init\n" if "earlyoom" in fail else "/usr/bin/earlyoom -m 2 -s 100 -r 5 --avoid sshd|systemd|init\n")
if "Linger" in command:
    out("yes\n")
if "is-active fleet-agent" in command:
    out("active\n")
if "sparkpipe_[w]eightd" in command:
    out("4242 sparkpipe-weightd.service\n")
if "SAFE MODE" in command:
    out("1\n" if "safe" in fail else "0\n")
if "nvidia-smi" in command:
    out("none 5000\n" if "noclock" in fail else "2411 5000\n" if "clock" in fail else "2190 5000\n")
if "weightd-mesh/.ready" in command:
    out("marker=yes\nready=1\n")
if "ceph osd tree down" in command:
    out('{"nodes": []}\n')
if "ceph health" in command:
    out("HEALTH_OK\n")
sys.stderr.write("unexpected command for %s: %s\n" % (host, command))
sys.exit(97)
'''


class FleetPostRebootTest(unittest.TestCase):
    def run_script(self, fail: str = "", rank: str = "3") -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            ssh = bin_dir / "ssh"
            ssh.write_text(FAKE_SSH)
            ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)
            sleep = bin_dir / "sleep"
            sleep.write_text("#!/bin/sh\nexit 0\n")
            sleep.chmod(sleep.stat().st_mode | stat.S_IEXEC)
            env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FIXTURE_FAIL=fail, FIXTURE_RANK=rank)
            host = "sparkf" if rank == "15" else "spark3"
            return subprocess.run(["bash", str(SCRIPT), "--timeout", "5", host], capture_output=True, text=True, env=env, timeout=120)

    def test_healthy_node_is_ready(self) -> None:
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("post-reboot    READY spark3", result.stdout)
        self.assertNotIn("FAIL", result.stdout)
        self.assertNotIn("unexpected command", result.stderr)

    def test_each_boot_outcome_gap_blocks_ready(self) -> None:
        for fault, check in (
            ("target", "boot-target"),
            ("guard", "dma-guard"),
            ("uplink", "uplink"),
            ("emergency", "emergency-ssh"),
            ("earlyoom", "earlyoom"),
            ("safe", "agent-mode"),
            ("clock", "gpu-clock"),
            ("noclock", "gpu-clock"),
            ("kernel", "kernel"),
        ):
            with self.subTest(fault=fault):
                result = self.run_script(fault)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertRegex(result.stdout, rf"spark3 +{check} +FAIL")
                self.assertIn("NOT READY spark3", result.stdout)

    def test_two_metric_10_default_routes_are_healthy(self) -> None:
        result = self.run_script("tworoutes")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r"spark3 +uplink +ok +ds4-uplink-wired active, 2 default route")

    def test_hub_gateway_rank_checks_its_direct_hub_link_instead_of_the_wired_uplink(self) -> None:
        result = self.run_script("uplink", rank="15")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r"sparkf +uplink +ok +rank 15: ds4-speculation-link")
        broken = self.run_script("hublink", rank="15")
        self.assertEqual(broken.returncode, 1, broken.stdout)
        self.assertRegex(broken.stdout, r"sparkf +uplink +FAIL +rank 15 hub link")


if __name__ == "__main__":
    unittest.main()
