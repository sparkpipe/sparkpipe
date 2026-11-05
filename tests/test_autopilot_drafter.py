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
from drafter_contract import drafter_tensors, score_identity  # noqa: E402
from test_latent_block_drafter import tiny_config  # noqa: E402

PACK = "ab" * 32
TAPS = ["o3", "o7"]
ORIGIN = "11" * 32
TRAINED_TAPS = {"name": "day.spt2", "bytes": 10, "sha256": "22" * 32, "license": "the customer's own sessions"}
HELD_OUT_TAPS = {"name": "held.spt2", "bytes": 10, "sha256": "44" * 32}
GENERAL_TAPS = {"name": "general.spt2", "bytes": 10, "sha256": "55" * 32}
TOOL = [sys.executable, str(ROOT / "tools" / "autopilot_drafter.py")]
T0 = 1_800_000_000


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


def checkpoint(directory, fill=1, pack=PACK, config=None, receipt_taps=None, **fields):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = config or tiny_config()
    (directory / "config.json").write_text(json.dumps(config))
    write_weights(directory / "model.safetensors", config, fill)
    digest = hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest()
    receipt = {"format": autopilot.RECEIPT_FORMAT, "model_sha256": digest, "tap_entries": receipt_taps or TAPS, "pack_identity": pack, "target_model": "tiny",
               "base": {"model_sha256": ORIGIN, "license": "the base's license"}, "origin_sha256": ORIGIN, "taps": [TRAINED_TAPS],
               "requests": [1, 2], "positions": 1000, "steps": 10, "seconds": 1.0}
    receipt.update(fields)
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

    def refused(self, function, *args, **options):
        with self.assertRaises(autopilot.RefusedDrafter) as caught:
            function(*args, **options)
        return str(caught.exception)

    def incoming(self, scratch=None):
        directory = autopilot.root(scratch or self.scratch) / "incoming"
        return sorted(path.name for path in directory.iterdir()) if directory.exists() else []

    def test_install_places_a_content_addressed_drafter_once(self):
        source = checkpoint(self.base / "run")
        manifest = autopilot.install(self.scratch, source, self.lane, PACK)
        target = autopilot.root(self.scratch) / "drafters" / manifest["drafter"]
        self.assertEqual(sorted(path.name for path in target.iterdir()), sorted(autopilot.FILES + (autopilot.MANIFEST,)))
        self.assertEqual(manifest["drafter"], manifest["files"]["model.safetensors"]["sha256"][:16])
        self.assertEqual((manifest["tap_entries"], manifest["pack_identity"], manifest["origin_sha256"]), (TAPS, PACK, ORIGIN))
        self.assertEqual(autopilot.install(self.scratch, source, self.lane, PACK), manifest)
        self.assertEqual(self.incoming(), [])

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
        self.assertFalse((autopilot.root(self.scratch) / "drafters").exists())

    def test_a_receipt_must_name_the_license_of_every_source(self):
        cases = {
            "base": dict(base={"model_sha256": ORIGIN}),
            "tap license": dict(taps=[{key: value for key, value in TRAINED_TAPS.items() if key != "license"}]),
            "tap names": dict(taps=["day.spt2"]),
            "no taps": dict(taps=[]),
            "origin": dict(origin_sha256=None),
            "requests": dict(requests=[]),
        }
        for label, fields in cases.items():
            with self.assertRaises(autopilot.RefusedDrafter, msg=label):
                autopilot.install(self.scratch, checkpoint(self.base / label, **fields), self.lane, PACK)
        self.assertIn("license", self.refused(autopilot.install, self.scratch, checkpoint(self.base / "blank", base={"model_sha256": ORIGIN, "license": " "}), self.lane, PACK))

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
        path.write_bytes(struct.pack("<Q", 5) + b'"abc"')
        self.assertIn("JSON object", self.refused(autopilot.check_weights, path, config))

    def test_publish_needs_a_drafter_id_and_refuses_a_changed_drafter(self):
        manifest = autopilot.install(self.scratch, checkpoint(self.base / "run"), self.lane, PACK)
        self.assertIsNone(autopilot.published(self.scratch))
        for bad in ("../" + manifest["drafter"], manifest["drafter"].upper(), manifest["drafter"][:15], manifest["drafter"] + "0"):
            self.assertIn("16 lowercase hex", self.refused(autopilot.publish, self.scratch, bad))
            self.assertIn("16 lowercase hex", self.refused(autopilot.export, self.scratch, bad))
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
        exported = autopilot.export(self.scratch, "")
        archive = Path(exported["archive"])
        self.assertEqual((exported["drafter"], archive), (manifest["drafter"], autopilot.root(self.scratch) / "exports" / f"{manifest['drafter']}.tar"))
        self.assertEqual(sorted(path.name for path in archive.parent.iterdir()), [archive.name])
        with tarfile.open(archive) as handle:
            self.assertEqual(handle.getnames(), [autopilot.MANIFEST] + list(autopilot.FILES))
        other = self.base / "other-fleet"
        restored = autopilot.restore(other, archive, self.lane, PACK)
        self.assertEqual(restored, manifest)
        for name in autopilot.FILES:
            self.assertEqual((autopilot.root(other) / "drafters" / manifest["drafter"] / name).read_bytes(),
                             (autopilot.root(self.scratch) / "drafters" / manifest["drafter"] / name).read_bytes())
        self.assertIsNone(autopilot.published(other))
        self.assertIn("target pack", self.refused(autopilot.restore, self.base / "third", archive, self.lane, "cd" * 32))
        self.assertEqual(self.incoming(self.base / "third"), [])

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

    def test_restore_refuses_forged_and_malformed_archives_and_leaves_nothing_behind(self):
        source = checkpoint(self.base / "run")
        manifest = autopilot.describe(source, self.lane, PACK)
        files = [(name, (source / name).read_bytes(), tarfile.REGTYPE) for name in autopilot.FILES]
        good_manifest = (autopilot.MANIFEST, json.dumps(manifest).encode(), tarfile.REGTYPE)
        config, weights, receipt = files
        scaled = (config[0], json.dumps(tiny_config(input_embedding_scale="x")).encode(), tarfile.REGTYPE)
        malformed = {
            "config scale": [good_manifest, scaled, weights, receipt],
            "receipt list": [good_manifest, config, weights, (receipt[0], b"[1]", tarfile.REGTYPE)],
            "header string": [good_manifest, config, (weights[0], struct.pack("<Q", 5) + b'"abc"', tarfile.REGTYPE), receipt],
            "manifest list": [(autopilot.MANIFEST, b"[1]", tarfile.REGTYPE)] + files,
        }
        cases = dict(malformed, **{
            "traversal": [good_manifest] + files + [("../escape", b"x", tarfile.REGTYPE)],
            "symlink": [good_manifest, ("config.json", b"", tarfile.SYMTYPE)] + files[1:],
            "duplicate": [good_manifest] + files + [files[0]],
            "missing": [good_manifest] + files[:2],
            "lying manifest": [(autopilot.MANIFEST, json.dumps(dict(manifest, drafter="0" * 16)).encode(), tarfile.REGTYPE)] + files,
        })
        lanes = self.base / "lanes.json"
        lanes.write_text(json.dumps({"lanes": {"tiny": {"taps": {"entries": ",".join(self.lane)}}}}))
        for label, members in cases.items():
            with self.assertRaises(autopilot.RefusedDrafter, msg=label):
                autopilot.restore(self.scratch, self.forged(members), self.lane, PACK)
            self.assertEqual(self.incoming(), [], label)
        for label, members in malformed.items():
            done = subprocess.run(TOOL + ["restore", "--scratch", str(self.scratch), "--lane", "tiny", "--lanes", str(lanes), "--pack-identity", PACK,
                                          str(self.forged(members))], capture_output=True, text=True)
            self.assertEqual(done.returncode, 2, label + done.stderr)
            self.assertIn("refused", done.stderr, label)
            self.assertNotIn("Traceback", done.stderr, label)
            self.assertEqual(self.incoming(), [], label)
        self.assertFalse((self.base / "escape").exists())
        self.assertFalse((autopilot.root(self.scratch) / "drafters").exists())
        (self.base / "junk.tar").write_bytes(b"not a tar file at all" * 40)
        self.assertIn("not a readable", self.refused(autopilot.restore, self.scratch, self.base / "junk.tar", self.lane, PACK))
        self.assertEqual(self.incoming(), [])

    def test_install_or_restore_over_a_corrupted_drafter_is_refused_and_leaves_nothing_behind(self):
        source = checkpoint(self.base / "run")
        manifest = autopilot.install(self.base / "good", source, self.lane, PACK)
        archive = autopilot.export(self.base / "good", manifest["drafter"])["archive"]
        autopilot.install(self.scratch, source, self.lane, PACK)
        weights = autopilot.root(self.scratch) / "drafters" / manifest["drafter"] / "model.safetensors"
        data = bytearray(weights.read_bytes())
        data[-1] ^= 1
        weights.write_bytes(bytes(data))
        self.assertIn("no longer matches", self.refused(autopilot.install, self.scratch, source, self.lane, PACK))
        self.assertEqual(self.incoming(), [])
        self.assertIn("no longer matches", self.refused(autopilot.restore, self.scratch, archive, self.lane, PACK))
        self.assertEqual(self.incoming(), [])

    def score(self, name, drafter, value, anchors=None, taps=HELD_OUT_TAPS, **arm):
        anchors = anchors or {100011: 400, 100012: 87}
        report = {"arm": dict({"delta": 0, "convention": "inplace", "layers": TAPS, "region": "reply", "requests": "100011-100020", "max_anchors": 400,
                               "drafter_sha256": drafter, "target_shard": {"name": "head.safetensors", "bytes": 4, "sha256": "66" * 32},
                               "taps": [taps], "donor_taps": []}, **arm),
                  "requests": [{"request_id": request, "anchors": count, "tokens_per_round": value} for request, count in anchors.items()],
                  "tokens_per_round": value}
        path = self.base / f"{name}.json"
        path.write_text(json.dumps(report))
        return path

    def scores(self, candidate, incumbent, values, **changed):
        held_out = dict(candidate=dict(drafter=candidate), incumbent=dict(drafter=incumbent))
        general = dict(base=dict(drafter=ORIGIN, taps=GENERAL_TAPS), candidate=dict(drafter=candidate, taps=GENERAL_TAPS))
        options = {"held_out_candidate": held_out["candidate"], "held_out_incumbent": held_out["incumbent"],
                   "general_base": general["base"], "general_candidate": general["candidate"]}
        return [self.score(name, value=value, **dict(options[name], **changed.get(name, {}))) for name, value in zip(options, values)]

    def candidate(self, fill=1):
        manifest = autopilot.install(self.scratch, checkpoint(self.base / f"run{fill}", fill=fill), self.lane, PACK)
        return manifest["drafter"], manifest["files"]["model.safetensors"]["sha256"]

    def test_gate_publishes_only_a_candidate_that_wins_held_out_and_keeps_general(self):
        drafter, weights = self.candidate()
        self.assertIn("incumbent stays", self.refused(autopilot.gate, self.scratch, drafter, *self.scores(weights, ORIGIN, (4.4, 4.4, 3.0, 3.0))))
        self.assertIn("incumbent stays", self.refused(autopilot.gate, self.scratch, drafter, *self.scores(weights, ORIGIN, (4.8, 4.4, 3.0, 2.9))))
        self.assertIsNone(autopilot.published(self.scratch))
        verdict = autopilot.gate(self.scratch, drafter, *self.scores(weights, ORIGIN, (4.6, 4.4, 3.0, 2.95)))
        self.assertTrue(verdict["published"])
        self.assertEqual(autopilot.published(self.scratch), drafter)
        second, second_weights = self.candidate(fill=2)
        self.assertIn("not the incumbent", self.refused(autopilot.gate, self.scratch, second, *self.scores(second_weights, ORIGIN, (4.8, 4.6, 3.0, 3.0))))
        autopilot.gate(self.scratch, second, *self.scores(second_weights, weights, (4.8, 4.6, 3.0, 3.0)))
        self.assertEqual(autopilot.published(self.scratch), second)

    def test_gate_refuses_scores_not_tied_to_the_candidate_and_one_anchor_set(self):
        drafter, weights = self.candidate()
        winning = (4.6, 4.4, 3.0, 3.0)
        cases = {
            "not the same arm": dict(held_out_incumbent=dict(requests="100001-100010", anchors={100001: 400, 100002: 87})),
            "different anchors": dict(held_out_incumbent=dict(anchors={100011: 400, 100012: 86})),
            "different anchors ": dict(general_base=dict(anchors={100011: 400})),
            "differ in ['taps']": dict(held_out_incumbent=dict(taps=GENERAL_TAPS)),
            "not the candidate": dict(held_out_candidate=dict(drafter="33" * 32), held_out_incumbent=dict(drafter="33" * 32)),
            "not the candidate ": dict(general_candidate=dict(drafter=ORIGIN)),
            "not the base": dict(general_base=dict(drafter=weights)),
            "region": dict(held_out_candidate=dict(region="prompt"), held_out_incumbent=dict(region="prompt")),
            "no drafter sha256": dict(held_out_candidate=dict(drafter=None), held_out_incumbent=dict(drafter=None)),
            "trained on": dict(held_out_candidate=dict(taps=TRAINED_TAPS, anchors={1: 40}), held_out_incumbent=dict(taps=TRAINED_TAPS, anchors={1: 40})),
        }
        for message, changed in cases.items():
            self.assertIn(message.strip(), self.refused(autopilot.gate, self.scratch, drafter, *self.scores(weights, ORIGIN, winning, **changed)), message)
        self.assertIn("16 lowercase hex", self.refused(autopilot.gate, self.scratch, "../" + drafter, *self.scores(weights, ORIGIN, winning)))
        self.assertIn("no drafter", self.refused(autopilot.gate, self.scratch, "0" * 16, *self.scores(weights, ORIGIN, winning)))
        self.assertIsNone(autopilot.published(self.scratch))

    def test_the_scorer_names_the_drafter_and_tap_files_the_gate_checks(self):
        source = checkpoint(self.base / "run")
        taps = self.base / "day.spt2"
        taps.write_bytes(b"taps")
        shard = self.base / "head.safetensors"
        shard.write_bytes(b"head")
        identity = score_identity(source, shard, [taps], [])
        self.assertEqual(identity["drafter_sha256"], autopilot.describe(source, self.lane, PACK)["files"]["model.safetensors"]["sha256"])
        self.assertEqual(identity["taps"], [{"name": "day.spt2", "bytes": 4, "sha256": hashlib.sha256(b"taps").hexdigest()}])
        self.assertEqual((identity["target_shard"]["sha256"], identity["donor_taps"]), (hashlib.sha256(b"head").hexdigest(), []))

    def store_entries(self):
        store = autopilot.root(self.scratch)
        return sorted(f"{kind}/{path.name}" for kind in autopilot.STORE if (store / kind).is_dir() for path in (store / kind).iterdir())

    def test_expire_deletes_everything_unused_for_a_day_and_a_use_restarts_the_clock(self):
        day = autopilot.working_state.RETENTION_SECONDS
        self.assertEqual(day, 86400)
        kept = autopilot.install(self.scratch, checkpoint(self.base / "a", fill=1), self.lane, PACK, now=T0)["drafter"]
        idle = autopilot.install(self.scratch, checkpoint(self.base / "b", fill=2), self.lane, PACK, now=T0)["drafter"]
        autopilot.publish(self.scratch, kept, now=T0)
        autopilot.export(self.scratch, idle, now=T0)
        leftover = Path(tempfile.mkdtemp(dir=autopilot.root(self.scratch) / "incoming"))
        partial = autopilot.root(self.scratch) / "exports" / "0123456789abcdef.tar.partial"
        partial.write_bytes(b"half")
        for path in (leftover, partial):
            os.utime(path, (T0, T0))
        everything = self.store_entries()
        self.assertEqual(autopilot.expire(self.scratch, now=T0 + day - 1)["removed"], [])
        autopilot.export(self.scratch, kept, now=T0 + 3600)
        removed = autopilot.expire(self.scratch, now=T0 + day)["removed"]
        self.assertEqual(sorted(removed), sorted([f"drafters/{idle}", f"incoming/{leftover.name}", f"exports/{idle}.tar", f"exports/{partial.name}"]))
        self.assertEqual(self.store_entries(), sorted(set(everything) - set(removed) | {f"exports/{kept}.tar"}))
        self.assertEqual(autopilot.published(self.scratch), kept)
        self.assertEqual(autopilot.expire(self.scratch, now=T0 + 3600 + day)["removed"], [f"drafters/{kept}", f"exports/{kept}.tar", autopilot.PUBLISHED])
        self.assertEqual(self.store_entries(), [])
        self.assertIsNone(autopilot.published(self.scratch))

    def test_expire_follows_the_working_state_directive(self):
        drafter = autopilot.install(self.scratch, checkpoint(self.base / "a"), self.lane, PACK, now=T0)["drafter"]
        day = autopilot.working_state.RETENTION_SECONDS
        directive = self.base / "directive"
        directive.write_text(f"sparkpipe-working-state 1\nwiped 0\nheld_until {T0 + 2 * day}\n")
        self.assertEqual(autopilot.expire(self.scratch, directive, now=T0 + 2 * day)["removed"], [])
        self.assertEqual(autopilot.expire(self.scratch, directive, now=T0 + 3 * day)["removed"], [f"drafters/{drafter}"])
        drafter = autopilot.install(self.scratch, checkpoint(self.base / "a"), self.lane, PACK, now=T0)["drafter"]
        directive.write_text(f"sparkpipe-working-state 1\nwiped {T0 + 5}\nheld_until 0\n")
        done = subprocess.run(TOOL + ["expire", "--scratch", str(self.scratch), "--directive", str(directive)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["removed"], [f"drafters/{drafter}"])

    def test_command_line_refuses_with_exit_two_and_publishes_only_through_the_gate(self):
        lanes = self.base / "lanes.json"
        lanes.write_text(json.dumps({"lanes": {"tiny": {"taps": {"entries": "o3,o7"}}}}))
        source = checkpoint(self.base / "run")
        install = TOOL + ["install", "--scratch", str(self.scratch), "--lane", "tiny", "--lanes", str(lanes)]
        done = subprocess.run(install + ["--pack-identity", PACK, str(source)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        drafter = json.loads(done.stdout)["drafter"]
        for bypass in (install + ["--pack-identity", PACK, "--publish", str(source)], TOOL + ["publish", "--scratch", str(self.scratch), drafter]):
            self.assertEqual(subprocess.run(bypass, capture_output=True, text=True).returncode, 2)
        state = json.loads(subprocess.run(TOOL + ["status", "--scratch", str(self.scratch)], capture_output=True, text=True, check=True).stdout)
        self.assertEqual((state["published"], [entry["drafter"] for entry in state["drafters"]]), (None, [drafter]))
        weights = autopilot.describe(source, self.lane, PACK)["files"]["model.safetensors"]["sha256"]
        gate = TOOL + ["gate", "--scratch", str(self.scratch), "--candidate", drafter]
        paths = lambda values: [flag for pair in zip(("--held-out-candidate", "--held-out-incumbent", "--general-base", "--general-candidate"),
                                                     map(str, self.scores(weights, ORIGIN, values))) for flag in pair]
        lost = subprocess.run(gate + paths((4.4, 4.4, 3.0, 3.0)), capture_output=True, text=True)
        self.assertEqual(lost.returncode, 2)
        self.assertIn("incumbent stays", lost.stderr)
        self.assertIsNone(autopilot.published(self.scratch))
        won = subprocess.run(gate + paths((4.6, 4.4, 3.0, 3.0)), capture_output=True, text=True)
        self.assertEqual(won.returncode, 0, won.stderr)
        self.assertEqual(autopilot.published(self.scratch), drafter)
        refused = subprocess.run(install + ["--pack-identity", "cd" * 32, str(source)], capture_output=True, text=True)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("refused", refused.stderr)


if __name__ == "__main__":
    unittest.main()
