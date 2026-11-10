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
DIGEST = "ab" * 32
STUB = "#!/bin/sh\necho 'model_residentd ready'\n"
FAKE_EXIT = "#!/bin/sh\nexit 0\n"


def deployment(nodes, named=lambda rank: f"config/stage_{rank:02d}.json", root=None):
    return {"nodes": [{"rank_index": rank, "adapter_configuration_path": named(rank),
                       **({"runtime_root": root} if root else {})} for rank in range(nodes)]}


def heredoc(path: Path, after: str) -> str:
    text = path.read_text()
    start = text.index("\n", text.index("<<'PYEOF'", text.index(after))) + 1
    return text[start:text.index("\nPYEOF\n", start)] + "\n"


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

    def source_root(self, scratch, nodes):
        source = Path(scratch) / "source"
        (source / "packs").mkdir(parents=True)
        (source / "config").mkdir()
        for name, data in (("packs/model.sp", b"pack"), ("packs/model.sp.experts", b"manifest"), ("packs/model.sp.ck128", b"ck"),
                           ("packs/model.sp.sha256", f"{DIGEST}  model.sp\n".encode())):
            (source / name).write_bytes(data)
        stage = {"stage_pack_path": "packs/model.sp", "tp_collective": {"backend_module_path": "lib/collective.so", "listen_port": 19000,
                                                                        "peer_ports": [19000], "peer_hosts": ["spark0"]}}
        for rank in range(nodes):
            (source / "config" / f"stage_{rank:02d}.json").write_text(json.dumps(stage))
        document = deployment(nodes, root=str(source))
        document.update({"transport": {"shared_object_path": "lib/transport.so", "control_port_base": 1},
                         "adapter": {"shared_object_path": "lib/adapter.so"}, "driver": {"shared_object_path": "stages/driver.so"}})
        (Path(scratch) / "deployment.json").write_text(json.dumps(document))
        return source

    def prepare_cases(self, run, source, extra_files=()):
        cases = {"pack": "packs/model.sp", "digest": "packs/model.sp.sha256", "manifest": "packs/model.sp.experts", "config": "config/stage_00.json", **dict(extra_files)}
        for label, name in cases.items():
            for kind in ("symlink", "missing"):
                with self.subTest(file=label, kind=kind):
                    path = source / name
                    real = path.with_name(path.name + ".real")
                    path.rename(real)
                    if kind == "symlink":
                        path.symlink_to(real)
                    result, private = run()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(f"{path} is missing, not a regular file or a symlink", result.stderr)
                    if kind == "symlink":
                        path.unlink()
                    real.rename(path)
                    shutil.rmtree(private, ignore_errors=True)

    def check_private(self, private, source, name):
        config = json.loads((private / "runtime" / name).read_text())
        self.assertEqual(config["stage_pack_path"], str(source / "packs/model.sp"))
        self.assertEqual(sorted(os.listdir(private / "runtime/packs")), ["pack.sha256"])
        self.assertEqual((private / "runtime/packs/pack.sha256").read_text(), f"{DIGEST}  model.sp\n")
        self.assertEqual(links_under(private), [])

    def test_family_template_names_the_real_pack(self):
        code = heredoc(ROOT / "tools/devcycle/templates/run-family-job.sh.template", "python3 - \"$ROOT\" \"$RANK\" \"$SIZE\"")
        with tempfile.TemporaryDirectory() as scratch:
            source = self.source_root(scratch, 1)
            private = Path(scratch) / "private"

            def run():
                arguments = [str(private), "0", "1", "23000", "53000", "64000", "23168", "0" * 32, "0", str(Path(scratch) / "deployment.json")]
                return subprocess.run([sys.executable, "-", *arguments], input=code, capture_output=True, text=True, timeout=60), private

            result, _ = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.check_private(private, source, "config/stage_00.json")
            shutil.rmtree(private)
            self.prepare_cases(run, source)

    def test_cold_launch_ab_names_the_real_pack_and_copies_binaries(self):
        code = heredoc(ROOT / "tools/glm5_next_coldlaunch_ab.sh", "python3 - \"$ROOT\" \"$RANK\" \"$CONTROL_BASE\"")
        with tempfile.TemporaryDirectory() as scratch:
            source = self.source_root(scratch, 16)
            execution = Path(scratch) / "exec"
            for name in ("lib/transport.so", "lib/adapter.so", "stages/driver.so", "lib/collective.so"):
                place(execution, "/" + name, name.encode())
            private = Path(scratch) / "private"

            def run():
                arguments = [str(private), "0", "23000", "53000", "64000", "23168", "0" * 32, "0", str(Path(scratch) / "deployment.json"), str(source), str(execution)]
                return subprocess.run([sys.executable, "-", *arguments], input=code, capture_output=True, text=True, timeout=60), private

            result, _ = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.check_private(private, source, "config/stage_00.json")
            self.assertEqual((private / "runtime/lib/adapter.so").read_text(), "lib/adapter.so")
            shutil.rmtree(private)
            self.prepare_cases(run, source, {"whole-pack checksum": "packs/model.sp.ck128"})

    def test_k3_run_family_names_the_deployed_pack(self):
        text = (ROOT / "tools/k3_multidev_run_family.sh").read_text()
        start = text.index('for weight in "$DEPLOYED_PACK"')
        end = text.index("\n", text.index('fail "packs/ must contain exactly one .sha256 sidecar"'))
        block = 'fail() { echo "k3: $*" >&2; exit 1; }\nmkdir -p "$ROOT/packs"\n' + text[start:end] + "\n"
        with tempfile.TemporaryDirectory() as scratch:
            pack = Path(scratch) / "sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage1.rank02.pack"
            pack.parent.mkdir(parents=True)
            pack.write_bytes(b"pack")
            Path(str(pack) + ".experts").write_bytes(bytes.fromhex("57455058020000000100000000000000"))
            Path(str(pack) + ".sha256").write_text(f"{DIGEST}  {pack.name}\n")
            private = Path(scratch) / "private"
            run = lambda: (subprocess.run(["bash", "-c", "set -euo pipefail\n" + block], capture_output=True, text=True, timeout=60,
                                          env=dict(os.environ, DEPLOYED_PACK=str(pack), ROOT=str(private))), private)
            result, _ = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sorted(os.listdir(private / "packs")), ["pack.sha256"])
            self.assertEqual((private / "packs/pack.sha256").read_text(), DIGEST)
            self.assertEqual(links_under(Path(scratch)), [])
            shutil.rmtree(private)
            manifest = Path(str(pack) + ".experts")
            manifest.write_bytes(bytes.fromhex("57455058010000000000000000000000"))
            result, _ = run()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("is not a v2 routed-expert manifest with records; regenerate it beside the pack", result.stderr)
            manifest.write_bytes(bytes.fromhex("57455058020000000100000000000000"))
            for name in (str(pack), str(manifest)):
                for kind in ("symlink", "missing"):
                    with self.subTest(file=name, kind=kind):
                        shutil.rmtree(private, ignore_errors=True)
                        path = Path(name)
                        real = path.with_name(path.name + ".real")
                        path.rename(real)
                        if kind == "symlink":
                            path.symlink_to(real)
                        result, _ = run()
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(f"{path} is missing, not a regular file or a symlink", result.stderr)
                        if kind == "symlink":
                            path.unlink()
                        real.rename(path)


if __name__ == "__main__":
    unittest.main()
