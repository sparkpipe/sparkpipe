#!/usr/bin/env python3
"""Lanes name their weight files directly: no lane tool links a pack.

Every record renders adapter configs whose stage_pack_path is the real rank
pack under /home/<host>/sparkdata, and no rendered file or swap command
reaches a pack through a link. stage_lane.sh and k3_lane.sh setup, run
against a fake fleet whose ssh and scp act on a scratch tree, leave no
symlink under the lane roots (they also remove the links an older layout
left), and refuse, naming the path, a weight file that is a symlink or is
missing. tools/lane_pack_files.py prints the exact files of a fixture
record, and lane_config.py audit reports a symlink planted under a lane
root.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import lane_config
import stage_lane

DENSE_MANIFEST = b"WEPX\x02" + bytes(11)
DIGEST = "ab" * 32
FAKE_SSH = r'''#!/usr/bin/env python3
import os, subprocess, sys
fake = os.environ["FAKE_FLEET"]
args = sys.argv[1:]
while args and args[0] == "-o":
    args = args[2:]
host, command = args[0], " ".join(args[1:]).replace("/home/", fake + "/home/")
with open(os.path.join(fake, "ssh.log"), "a") as log:
    log.write(host + " " + command + "\n")
if command.startswith("systemctl --user is-active"):
    print("inactive")
    sys.exit(3)
sys.exit(subprocess.call(["bash", "-c", command], env=dict(os.environ, HOME=os.path.join(fake, "home", host))))
'''
FAKE_SCP = r'''#!/usr/bin/env python3
import os, shutil, sys
fake = os.environ["FAKE_FLEET"]
args = [a for a in sys.argv[1:] if a != "-q"]
target = args[-1].split(":", 1)[1].replace("/home/", fake + "/home/")
for source in args[:-1]:
    shutil.copy(source, os.path.join(target, os.path.basename(source)) if target.endswith("/") else target)
'''
FAKE_REMOTE = r'''#!/usr/bin/env python3
import os, subprocess, sys
fake = os.environ["FAKE_FLEET"]
args = sys.argv[1:]
while args and args[0] == "-o":
    args = args[2:]
host = args[0]
home = os.path.join(fake, "home", host)
os.makedirs(home, exist_ok=True)
sys.exit(subprocess.call(["bash", "-c", " ".join(args[1:])], env=dict(os.environ, HOME=home)))
'''


def fake_bin(directory: Path, scripts: dict) -> Path:
    binary = directory / "bin"
    binary.mkdir(exist_ok=True)
    for name, body in scripts.items():
        (binary / name).write_text(body)
        (binary / name).chmod(0o755)
    return binary


def links_under(top: Path) -> list[str]:
    found = []
    for directory, names, leaves in os.walk(top):
        found += [os.path.join(directory, name) for name in names + leaves if os.path.islink(os.path.join(directory, name))]
    return sorted(found)


def place(fake: Path, path: str, data: bytes) -> Path:
    local = Path(fake.as_posix() + path)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return local


class DirectPacks(unittest.TestCase):
    def test_every_record_renders_direct_pack_paths(self):
        records = lane_config.records()
        self.assertGreaterEqual(len(records), 19)
        with tempfile.TemporaryDirectory() as scratch:
            for record_id, lane in sorted(records.items()):
                directory = Path(scratch) / record_id
                lane_config.render(record_id, directory)
                model = stage_lane.MODELS[lane["model"]]
                for rank, host in enumerate(stage_lane.HOSTS):
                    pack = json.loads((directory / f"adapter.{host}.json").read_text())["stage_pack_path"]
                    with self.subTest(record=record_id, host=host):
                        self.assertEqual(pack, f"/home/{host}/{model['pack'].format(rank=rank)}")
                        self.assertTrue(pack.startswith(f"/home/{host}/sparkdata/"), pack)
                        self.assertFalse(pack.startswith(lane["runtime_root"].format(host=host)), pack)
                for path in sorted(directory.iterdir()):
                    with self.subTest(record=record_id, file=path.name):
                        self.assertEqual(self.link_uses(path.read_text()), [], path.name)
        for entry in json.loads((ROOT / "deployment/hub/model_swap.json").read_text())["models"]:
            for key, value in sorted(entry.items()):
                if isinstance(value, str):
                    with self.subTest(swap=entry["id"], key=key):
                        self.assertEqual(self.link_uses(value), [], value[:200])

    def link_uses(self, text: str) -> list[str]:
        uses = re.findall(r"ln -s\S*|readlink -f \S*", text)
        return uses + [token for token in re.findall(r"[^\s\"'()=]*packs/[^\s\"'()]*", text) if "sparkdata/" not in token and not token.endswith("packs/pack.sha256")]

    def fleet(self, scratch: str):
        fake = Path(scratch) / "fleet"
        fake.mkdir()
        binary = fake_bin(Path(scratch), {"ssh": FAKE_SSH, "scp": FAKE_SCP})
        firmware = Path(scratch) / "firmware"
        firmware.mkdir()
        for name in ("sparkpipe_model_residentd", "weightd_warm", "libhidden_transport_spark_host_rdma_verbs.so",
                     "libk3_tp16_serving_adapter.so", stage_lane.MODELS["dsv41_flash"]["adapter"]):
            (firmware / name).write_text(name)
        environment = {"PATH": f"{binary}:{os.environ['PATH']}", "HOME": str(fake), "TMPDIR": scratch, "FAKE_FLEET": str(fake)}
        return fake, firmware, environment

    def weights(self, fake: Path, files: list[str]):
        place(fake, files[0], bytes(4096))
        place(fake, files[1], DENSE_MANIFEST)
        place(fake, files[2], f"{DIGEST}  {os.path.basename(files[0])}\n".encode())
        for path in files[3:]:
            place(fake, path, DENSE_MANIFEST if path.endswith(".experts") else DIGEST.encode() if path.endswith(".digest") else b"sibling")

    def stage_setup(self, fake: Path, firmware: Path, environment: dict):
        settings = dict(environment, STAGE_RECORD="deepseek-v4.1-flash", STAGE_FIRMWARE=str(firmware))
        return subprocess.run(["bash", str(ROOT / "tools/stage_lane.sh"), "setup"], env=settings, capture_output=True, text=True, timeout=600)

    def k3_setup(self, fake: Path, firmware: Path, environment: dict):
        settings = dict(environment, K3_RECORD="kimi-k3", K3_FIRMWARE=str(firmware))
        return subprocess.run(["bash", str(ROOT / "tools/k3_lane.sh"), "setup"], env=settings, capture_output=True, text=True, timeout=600)

    def test_setup_creates_no_link_and_removes_old_ones(self):
        model = stage_lane.MODELS["dsv41_flash"]
        k3 = stage_lane.MODELS["k3"]
        with tempfile.TemporaryDirectory() as scratch:
            fake, firmware, environment = self.fleet(scratch)
            for rank, host in enumerate(stage_lane.HOSTS):
                self.weights(fake, stage_lane.rank_weight_files(model, rank))
                self.weights(fake, stage_lane.rank_weight_files(k3, rank))
            roots = {record_id: lane_config.record(record_id)["runtime_root"] for record_id in ("deepseek-v4.1-flash", "kimi-k3")}
            old = place(fake, roots["deepseek-v4.1-flash"].format(host="spark3") + "/logs/residentd-r1.log", b"old run\n")
            (old.parent.parent / "packs").mkdir()
            (old.parent.parent / "packs" / "rank3.spstage").symlink_to(fake.as_posix() + stage_lane.rank_pack_path(model, 3))
            (old.parent.parent / "residentd.log").symlink_to("logs/residentd-r1.log")
            for name, run in (("stage_lane", self.stage_setup), ("k3_lane", self.k3_setup)):
                result = run(fake, firmware, environment)
                self.assertEqual(result.returncode, 0, f"{name}: {result.stderr[-600:]}")
            self.assertEqual(links_under(fake / "home"), [])
            for rank, host in enumerate(stage_lane.HOSTS):
                for lane_root, pack in ((roots["deepseek-v4.1-flash"].format(host=host), stage_lane.rank_pack_path(model, rank)),
                                        (roots["kimi-k3"].format(host=host), stage_lane.rank_pack_path(k3, rank))):
                    local = Path(fake.as_posix() + lane_root)
                    with self.subTest(root=lane_root):
                        self.assertEqual(json.loads((local / "config/adapter.json").read_text())["stage_pack_path"], pack)
                        self.assertEqual(sorted(os.listdir(local / "packs")), ["pack.sha256"])
                        self.assertEqual((local / "packs/pack.sha256").read_text(), DIGEST)
                        self.assertEqual((local / "spine_budget").read_text().strip(), str(4096 + 255))
            self.assertEqual(old.read_text(), "old run\n")

    def test_setup_refuses_a_linked_or_missing_weight_file(self):
        model = stage_lane.MODELS["dsv41_flash"]
        k3 = stage_lane.MODELS["k3"]
        with tempfile.TemporaryDirectory() as scratch:
            fake, firmware, environment = self.fleet(scratch)
            for rank, host in enumerate(stage_lane.HOSTS):
                self.weights(fake, stage_lane.rank_weight_files(model, rank))
                self.weights(fake, stage_lane.rank_weight_files(k3, rank))
            for name, run, files in (("stage_lane", self.stage_setup, stage_lane.rank_weight_files(model, 11)),
                                     ("k3_lane", self.k3_setup, stage_lane.rank_weight_files(k3, 11))):
                for broken in (files[0], files[1], files[-1]):
                    local = Path(fake.as_posix() + broken)
                    real = local.with_name(local.name + ".real")
                    local.rename(real)
                    for kind in ("symlink", "missing"):
                        if kind == "symlink":
                            local.symlink_to(real)
                        result = run(fake, firmware, environment)
                        with self.subTest(tool=name, file=broken, kind=kind):
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn(f"sparkb: {local} is missing, not a regular file or a symlink", result.stderr)
                            self.assertIn("failed on: sparkb", result.stderr)
                            self.assertFalse(Path(fake.as_posix() + "/home/sparkb/stage-lanes").exists() or Path(fake.as_posix() + "/home/sparkb/k3-lanes").exists())
                        if kind == "symlink":
                            local.unlink()
                    real.rename(local)

    def test_pack_files_lists_a_fixture_record(self):
        with tempfile.TemporaryDirectory() as scratch:
            document = json.loads(stage_lane.LANES.read_text())
            document["lanes"] = {key: document["lanes"][key] for key in ("deepseek-v4.1-flash", "kimi-k3")}
            fixture = Path(scratch) / "lanes.json"
            fixture.write_text(json.dumps(document))
            environment = dict(os.environ, SPARK_LANES=str(fixture))
            run = lambda *args: subprocess.run([sys.executable, str(ROOT / "tools/lane_pack_files.py"), *args], env=environment, capture_output=True, text=True, timeout=300)
            result = run("--host", "spark3")
            self.assertEqual(result.returncode, 0, result.stderr)
            xs, k3 = "/home/spark3/sparkdata/dsv41flash-1017.mxfp4.tp16.xs/packs", "/home/spark3/sparkdata/k3.mxfp4.tp16/packs/k3.stage0.rank03.pack"
            self.assertEqual(result.stdout.splitlines(), [
                f"spark3 deepseek-v4.1-flash digest {xs}/rank3.spstage.sha256",
                f"spark3 deepseek-v4.1-flash experts {xs}/rank3.spstage.experts",
                f"spark3 deepseek-v4.1-flash pack {xs}/rank3.spstage",
                f"spark3 deepseek-v4.1-flash sibling {xs}/engram.spengram",
                f"spark3 deepseek-v4.1-flash sibling {xs}/engram.spengram.digest",
                f"spark3 deepseek-v4.1-flash sibling {xs}/engram.spengram.experts",
                f"spark3 deepseek-v4.1-flash sibling {xs}/engram_tables.bin",
                f"spark3 kimi-k3 digest {k3}.sha256",
                f"spark3 kimi-k3 experts {k3}.experts",
                f"spark3 kimi-k3 pack {k3}",
            ])
            every = run().stdout.splitlines()
            self.assertEqual(len(every), 16 * 10)
            self.assertEqual({line.split()[0] for line in every}, set(stage_lane.HOSTS))
            self.assertTrue(all(line.split()[3].startswith(f"/home/{line.split()[0]}/sparkdata/") for line in every))
            self.assertNotEqual(run("no-such-record").returncode, 0)

    def test_audit_reports_a_planted_link(self):
        with tempfile.TemporaryDirectory() as scratch:
            fake = Path(scratch) / "fleet"
            binary = fake_bin(Path(scratch), {"ssh": FAKE_REMOTE})
            environment = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", FAKE_FLEET=str(fake))
            root = Path(fake.as_posix() + lane_config.record("glm-5.3")["runtime_root"].format(host="spark4"))
            (root / "packs").mkdir(parents=True)
            (root / "packs" / "glm53full.rank4.glm52sp").symlink_to("/elsewhere/glm53full.rank4.glm52sp")
            backup = fake / "home/spark9/k3-lanes/lane99/root.prev-1/packs"
            backup.mkdir(parents=True)
            (backup / "k3.stage0.rank09.pack").symlink_to("/elsewhere/k3.pack")
            audit = lambda *records: subprocess.run([sys.executable, str(ROOT / "tools/lane_config.py"), "audit", *records], env=environment,
                                                    capture_output=True, text=True, timeout=900)
            selected = audit("glm-5.3")
            links = [line for line in selected.stdout.splitlines() if line.startswith("LINK ")]
            self.assertNotEqual(selected.returncode, 0)
            self.assertEqual(links, [f"LINK spark4:{root}/packs/glm53full.rank4.glm52sp -> /elsewhere/glm53full.rank4.glm52sp is a symlink under a lane tree; "
                                     "lanes name their files directly and setup removes the links under a lane root"])
            every = [line.split(" ->")[0] for line in audit().stdout.splitlines() if line.startswith("LINK ")]
            self.assertEqual(every, [f"LINK spark4:{root}/packs/glm53full.rank4.glm52sp", f"LINK spark9:{backup}/k3.stage0.rank09.pack"])


if __name__ == "__main__":
    unittest.main()
