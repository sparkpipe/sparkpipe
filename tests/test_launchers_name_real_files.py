#!/usr/bin/env python3
"""Launchers and harnesses name their weight files and configurations directly.

Every deployment generator names each rank's own config/stage_<RR>.json,
and every committed deployment names one real file per rank. fleet_serve.sh
and lag_t1_launch.sh, run against a fake fleet whose ssh acts on a scratch
tree, start every rank without a config/stage.json link (and delete one an
older launcher left), and refuse a rank whose deployment names a linked or
missing configuration. The private runtime preparation of the family job
template, the cold-launch A/B and the K3 run family name the real pack by
its path, copy one digest and nothing else, and refuse a linked or missing
pack, manifest, digest or stage config. fleet_preflight.sh reports a linked
configuration or pack.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_lane_direct_packs import FAKE_SSH, fake_bin, links_under, place

HOSTS = [f"spark{index:x}" for index in range(16)]
STUB = "#!/bin/sh\necho 'model_residentd ready'\n"
FAKE_EXIT = "#!/bin/sh\nexit 0\n"


def deployment(nodes, named=lambda rank: f"config/stage_{rank:02d}.json", root=None):
    return {"nodes": [{"rank_index": rank, "adapter_configuration_path": named(rank),
                       **({"runtime_root": root} if root else {})} for rank in range(nodes)]}


class DirectFiles(unittest.TestCase):
    def test_committed_deployments_name_each_ranks_config(self):
        committed = sorted(path for path in (ROOT / "deployment").glob("*/model_resident.json") if (path.parent / "config").is_dir())
        self.assertGreaterEqual(len(committed), 2)
        for path in committed:
            nodes = json.loads(path.read_text())["nodes"]
            with self.subTest(deployment=path.parent.name):
                names = [node["adapter_configuration_path"] for node in nodes]
                self.assertEqual(len(set(names)), len(nodes), names)
                self.assertNotIn("config/stage.json", names)
                for name in names:
                    self.assertTrue((path.parent / name).is_file() and not (path.parent / name).is_symlink(), name)

    def fake_fleet(self, scratch):
        fake = Path(scratch) / "fleet"
        fake.mkdir()
        binary = fake_bin(Path(scratch), {"ssh": FAKE_SSH, "curl": FAKE_EXIT, "sleep": FAKE_EXIT, "pgrep": FAKE_EXIT})
        return fake, {"PATH": f"{binary}:{os.environ['PATH']}", "HOME": str(fake), "TMPDIR": scratch, "FAKE_FLEET": str(fake)}

    def root(self, fake, host, relative, document, deployment_name="model_resident.json", binaries=("sparkpipe_model_residentd",)):
        local = Path(fake.as_posix() + f"/home/{host}/{relative}")
        (local / "bin").mkdir(parents=True)
        (local / "config").mkdir(exist_ok=True)
        for name in binaries:
            (local / "bin" / name).write_text(STUB)
            (local / "bin" / name).chmod(0o755)
        (local / deployment_name).write_text(json.dumps(document))
        for rank in range(16):
            (local / "config" / f"stage_{rank:02d}.json").write_text("{}\n")
        return local

    def test_fleet_serve_starts_every_rank_without_a_config_link(self):
        for broken in (None, "symlink", "missing"):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as scratch:
                fake, environment = self.fake_fleet(scratch)
                roots = {host: self.root(fake, host, "sparkdata/glmtest", deployment(16)) for host in HOSTS}
                (roots["spark3"] / "config" / "stage.json").symlink_to("stage_03.json")
                if broken:
                    roots["sparkb"].joinpath("model_resident.json").write_text(json.dumps(deployment(16, lambda rank: "config/stage.json")))
                    if broken == "symlink":
                        (roots["sparkb"] / "config" / "stage.json").symlink_to("stage_11.json")
                result = subprocess.run(["bash", str(ROOT / "tools/fleet_serve.sh"), "glmtest", "start"], capture_output=True, text=True, timeout=120,
                                        env=dict(environment, G5_API_HOST="rtx5090", FLEET_RUN_ID="r1"))
                if broken is None:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("ready 16/16", result.stdout)
                    self.assertEqual(links_under(fake), [])
                    self.assertTrue(all((roots[host] / "logs/residentd-r1.log").is_file() for host in HOSTS))
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("did not start on: sparkb", result.stderr)
                    self.assertIn("fleet_serve: sparkb: rank 11 adapter configuration config/stage.json is missing, not a regular file or a symlink", result.stderr)
                    self.assertFalse((roots["sparkb"] / "logs/residentd-r1.log").exists())
                self.assertFalse(os.path.lexists(roots["spark3"] / "config" / "stage.json"))

    def test_laguna_launcher_names_each_ranks_config(self):
        arm = "sparkdata/laguna-s-2.1.bf16.tp8pp2"
        for broken in (None, "symlink"):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as scratch:
                fake, environment = self.fake_fleet(scratch)
                roots = {host: self.root(fake, host, arm, deployment(16), "config/model_resident.json",
                                         ("sparkpipe_model_residentd", "sparkpipe_model_api")) for host in HOSTS}
                (roots["spark5"] / "config" / "stage.json").symlink_to("stage_05.json")
                if broken:
                    (roots["sparkb"] / "config" / "stage_11.json").unlink()
                    (roots["sparkb"] / "config" / "stage_11.json").symlink_to("stage_10.json")
                result = subprocess.run(["sh", str(ROOT / "tools/lag_t1_launch.sh"), "1,2,3", "4"], capture_output=True, text=True, timeout=300, env=environment)
                launched = sorted(line.split()[1] for line in result.stdout.splitlines() if line.startswith("launched "))
                self.assertFalse(os.path.lexists(roots["spark5"] / "config" / "stage.json"))
                if broken is None:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(launched, sorted(HOSTS))
                    self.assertEqual(links_under(fake), [])
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("lag_t1_launch: sparkb: rank 11 adapter configuration config/stage_11.json is missing, not a regular file or a symlink", result.stderr)
                    self.assertNotIn("sparkb", launched)

    @unittest.skipUnless(shutil.which("md5sum") and shutil.which("ss") and Path("/proc/meminfo").exists(), "fleet_preflight.sh needs md5sum, ss and /proc")
    def test_preflight_reports_a_linked_config_or_pack(self):
        with tempfile.TemporaryDirectory() as scratch:
            fake, environment = self.fake_fleet(scratch)
            release = place(fake, "/home/rtx5090/release/glmtest/bin/sparkpipe_model_residentd", STUB.encode()).parents[1]
            place(fake, "/home/rtx5090/release/glmtest/stages/stage_000/model_driver.so", b"driver")
            for rank in range(16):
                place(fake, f"/home/rtx5090/release/glmtest/config/stage_{rank:02d}.json", b"{}\n")
            roots = {}
            for rank, host in enumerate(HOSTS):
                roots[host] = local = self.root(fake, host, "sparkdata/glmtest", deployment(16))
                shutil.copy(release / "bin/sparkpipe_model_residentd", local / "bin/sparkpipe_model_residentd")
                (local / "stages/stage_000").mkdir(parents=True)
                shutil.copy(release / "stages/stage_000/model_driver.so", local / "stages/stage_000/model_driver.so")
                place(fake, f"/home/{host}/sparkdata/glmtest/packs/glmtest.rank{rank:x}.sp", bytes.fromhex("474c5833") + bytes(60))
            (roots["sparkb"] / "config/stage_11.json").unlink()
            (roots["sparkb"] / "config/stage_11.json").symlink_to("stage_10.json")
            pack = roots["spark5"] / "packs/glmtest.rank5.sp"
            pack.rename(pack.with_name("real.sp"))
            pack.symlink_to("real.sp")
            result = subprocess.run(["bash", str(ROOT / "tools/fleet_preflight.sh"), "rtx5090:/home/rtx5090/release", "glmtest"],
                                    capture_output=True, text=True, timeout=300, env=environment)
            failures = {line.split()[1].rstrip(":"): line for line in result.stdout.splitlines() if line.startswith("PREFLIGHT-FAIL ")}
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("config-not-direct", failures.get("sparkb", ""))
            self.assertIn("pack-symlink", failures.get("spark5", ""))
            for host in HOSTS:
                self.assertEqual("config-not-direct" in failures.get(host, ""), host == "sparkb", host)
                self.assertEqual("pack-symlink" in failures.get(host, ""), host == "spark5", host)


if __name__ == "__main__":
    unittest.main()
