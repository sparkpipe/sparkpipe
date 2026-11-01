import hashlib
import io
import json
import math
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "spec_offline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import autopilot_drafter as autopilot  # noqa: E402
from drafter_contract import drafter_tensors  # noqa: E402
from test_latent_block_drafter import tiny_config  # noqa: E402

PACK = "ab" * 32
TAPS = ["o3", "o7"]


def write_weights(path, config, fill=0, shapes=None, dtype="BF16", hole=0):
    shapes = shapes or drafter_tensors(config)
    header, offset = {}, 0
    for name, shape in shapes.items():
        size = 2 * math.prod(shape)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size + hole
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded + bytes([fill]) * offset)


def checkpoint(directory, fill=1, pack=PACK, config=None, receipt_taps=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = config or tiny_config()
    (directory / "config.json").write_text(json.dumps(config))
    write_weights(directory / "model.safetensors", config, fill)
    digest = hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()
    receipt = {"format": autopilot.RECEIPT_FORMAT, "model_sha256": digest, "tap_entries": receipt_taps or TAPS, "pack_identity": pack, "target_model": "tiny",
               "base": {"model_sha256": "00" * 32}, "positions": 1000, "steps": 10, "seconds": 1.0}
    (directory / "TRAINING-RECEIPT.json").write_text(json.dumps(receipt))
    return directory


class AutopilotDrafterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.scratch = self.base / "scratch"
        self.lane = ["o3", "o7", "o9"]

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, function, *args):
        with self.assertRaises(autopilot.RefusedDrafter) as caught:
            function(*args)
        return str(caught.exception)

    def test_install_places_a_content_addressed_drafter_once(self):
        source = checkpoint(self.base / "run")
        manifest = autopilot.install(self.scratch, source, self.lane, PACK)
        target = autopilot.root(self.scratch) / "drafters" / manifest["drafter"]
        self.assertEqual(sorted(path.name for path in target.iterdir()), sorted(autopilot.FILES + (autopilot.MANIFEST,)))
        self.assertEqual(manifest["drafter"], manifest["files"]["model.safetensors"]["sha256"][:16])
        self.assertEqual((manifest["tap_entries"], manifest["pack_identity"]), (TAPS, PACK))
        self.assertEqual(autopilot.install(self.scratch, source, self.lane, PACK), manifest)
        self.assertEqual(list((autopilot.root(self.scratch) / "incoming").iterdir()), [])

    def test_install_refuses_what_the_lane_cannot_serve(self):
        self.assertIn("lane streams", self.refused(autopilot.install, self.scratch, checkpoint(self.base / "a"), ["o3"], PACK))
        self.assertIn("target pack", self.refused(autopilot.install, self.scratch, checkpoint(self.base / "b"), self.lane, "cd" * 32))
        self.assertIn("trained on taps", self.refused(autopilot.install, self.scratch, checkpoint(self.base / "c", receipt_taps=["o4", "o7"]), self.lane, PACK))
        bad = tiny_config(is_causal=True)
        directory = checkpoint(self.base / "d")
        (directory / "config.json").write_text(json.dumps(bad))
        self.assertIn("supported drafter", self.refused(autopilot.install, self.scratch, directory, self.lane, PACK))
        directory = checkpoint(self.base / "e")
        (directory / "model.safetensors").write_bytes((directory / "model.safetensors").read_bytes()[:-1] + b"\x02")
        self.assertIn("other weights", self.refused(autopilot.install, self.scratch, directory, self.lane, PACK))
        (directory / "extra.bin").write_bytes(b"x")
        self.assertIn("holds exactly", self.refused(autopilot.install, self.scratch, directory, self.lane, PACK))

    def test_weights_must_be_exactly_the_drafter_tensors_in_bf16_without_holes(self):
        config = tiny_config()
        path = self.base / "w.safetensors"
        write_weights(path, config)
        autopilot.check_weights(path, config)
        shapes = drafter_tensors(config)
        wrong = dict(shapes, **{"fc.weight": [1, 1]})
        cases = {"shape": dict(shapes=wrong), "dtype": dict(dtype="F32"), "hole": dict(hole=2)}
        for label, options in cases.items():
            write_weights(path, config, **options)
            with self.assertRaises(autopilot.RefusedDrafter, msg=label):
                autopilot.check_weights(path, config)
        missing = dict(shapes)
        missing.pop("norm.weight")
        write_weights(path, config, shapes=missing)
        self.assertIn("missing", self.refused(autopilot.check_weights, path, config))
        path.write_bytes(struct.pack("<Q", 1 << 40))
        self.assertIn("header", self.refused(autopilot.check_weights, path, config))

    def test_publish_points_at_a_verified_drafter_and_refuses_a_changed_one(self):
        manifest = autopilot.install(self.scratch, checkpoint(self.base / "run"), self.lane, PACK)
        self.assertIsNone(autopilot.published(self.scratch))
        autopilot.publish(self.scratch, manifest["drafter"])
        self.assertEqual(autopilot.published(self.scratch), manifest["drafter"])
        weights = autopilot.root(self.scratch) / "drafters" / manifest["drafter"] / "model.safetensors"
        data = bytearray(weights.read_bytes())
        data[-1] ^= 1
        weights.write_bytes(bytes(data))
        self.assertIn("no longer matches", self.refused(autopilot.publish, self.scratch, manifest["drafter"]))

    def test_export_then_restore_on_another_fleet_reproduces_the_drafter(self):
        manifest = autopilot.install(self.scratch, checkpoint(self.base / "run"), self.lane, PACK)
        autopilot.publish(self.scratch, manifest["drafter"])
        archive = self.base / "drafter.tar"
        exported = autopilot.export(self.scratch, "", archive)
        self.assertEqual(exported["drafter"], manifest["drafter"])
        with tarfile.open(archive) as handle:
            self.assertEqual(handle.getnames(), [autopilot.MANIFEST] + list(autopilot.FILES))
        other = self.base / "other-fleet"
        restored = autopilot.restore(other, archive, self.lane, PACK)
        self.assertEqual(restored, manifest)
        for name in autopilot.FILES:
            self.assertEqual((autopilot.root(other) / "drafters" / manifest["drafter"] / name).read_bytes(),
                             (autopilot.root(self.scratch) / "drafters" / manifest["drafter"] / name).read_bytes())
        self.assertIn("target pack", self.refused(autopilot.restore, self.base / "third", archive, self.lane, "cd" * 32))
        self.assertEqual(list((autopilot.root(self.base / "third") / "incoming").iterdir()), [])

    def forged(self, members):
        path = self.base / "forged.tar"
        with tarfile.open(path, "w") as archive:
            for name, data, kind in members:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.size = len(data) if kind == tarfile.REGTYPE else 0
                if kind == tarfile.SYMTYPE:
                    info.linkname = "/etc/passwd"
                archive.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
        return path

    def test_restore_refuses_forged_archives_and_leaves_nothing_behind(self):
        source = checkpoint(self.base / "run")
        manifest = autopilot.describe(source, self.lane, PACK)
        files = [(name, (source / name).read_bytes(), tarfile.REGTYPE) for name in autopilot.FILES]
        good_manifest = (autopilot.MANIFEST, json.dumps(manifest).encode(), tarfile.REGTYPE)
        cases = {
            "traversal": [good_manifest] + files + [("../escape", b"x", tarfile.REGTYPE)],
            "symlink": [good_manifest, ("config.json", b"", tarfile.SYMTYPE)] + files[1:],
            "duplicate": [good_manifest] + files + [files[0]],
            "missing": [good_manifest] + files[:2],
            "lying manifest": [(autopilot.MANIFEST, json.dumps(dict(manifest, drafter="0" * 16)).encode(), tarfile.REGTYPE)] + files,
        }
        for label, members in cases.items():
            with self.assertRaises(autopilot.RefusedDrafter, msg=label):
                autopilot.restore(self.scratch, self.forged(members), self.lane, PACK)
        self.assertFalse((self.base / "escape").exists())
        self.assertEqual(list((autopilot.root(self.scratch) / "incoming").iterdir()), [])
        self.assertFalse((autopilot.root(self.scratch) / "drafters").exists())
        (self.base / "junk.tar").write_bytes(b"not a tar file at all" * 40)
        self.assertIn("not a readable", self.refused(autopilot.restore, self.scratch, self.base / "junk.tar", self.lane, PACK))

    def test_gate_publishes_only_a_candidate_that_wins_held_out_and_keeps_general(self):
        def score(name, value, region="reply"):
            path = self.base / f"{name}.json"
            path.write_text(json.dumps({"tokens_per_round": value, "arm": {"region": region}}))
            return path
        decide = lambda c, i, b, g: autopilot.gate(score("c", c), score("i", i), score("b", b), score("g", g), 0.02)["publish"]
        self.assertTrue(decide(4.6, 4.4, 3.0, 2.95))
        self.assertFalse(decide(4.4, 4.4, 3.0, 3.0))
        self.assertFalse(decide(4.8, 4.4, 3.0, 2.9))
        self.assertIn("region", self.refused(autopilot.gate, score("c", 6.0, "prompt"), score("i", 4.4), score("b", 3.0), score("g", 3.0), 0.02))

    def test_command_line_refuses_with_exit_two_and_reads_lane_taps(self):
        lanes = self.base / "lanes.json"
        lanes.write_text(json.dumps({"lanes": {"tiny": {"taps": {"entries": "o3,o7"}}}}))
        tool = [sys.executable, str(ROOT / "tools" / "autopilot_drafter.py")]
        source = checkpoint(self.base / "run")
        done = subprocess.run(tool + ["install", "--scratch", str(self.scratch), "--lane", "tiny", "--lanes", str(lanes), "--pack-identity", PACK, "--publish", str(source)],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(json.loads(done.stdout)["published"])
        state = json.loads(subprocess.run(tool + ["status", "--scratch", str(self.scratch)], capture_output=True, text=True, check=True).stdout)
        self.assertEqual(state["published"], state["drafters"][0]["drafter"])
        refused = subprocess.run(tool + ["install", "--scratch", str(self.scratch), "--lane", "tiny", "--lanes", str(lanes), "--pack-identity", "cd" * 32, str(source)],
                                 capture_output=True, text=True)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("refused", refused.stderr)


if __name__ == "__main__":
    unittest.main()
