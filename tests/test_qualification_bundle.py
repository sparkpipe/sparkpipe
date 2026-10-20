#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bundle", ROOT / "tools/qualification_bundle.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)

DRIVER = "d" * 64


class QualificationBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        release = root / "release"
        release.mkdir()
        (release / "SHA256SUMS").write_text("%s  ./stages/stage_000/model_driver.so\n%s  ./bin/sparkpipe_model_residentd\n" % (DRIVER, "a" * 64))
        self.paths = {"release_root": str(release)}
        ranks = [{"rank_index": rank, "host": "spark%x" % rank, "driver_sha256": DRIVER, "pack_sha256": "p" * 64} for rank in range(4)]
        self.write("rank_identities", json.dumps(ranks))
        self.write("drained_state", json.dumps({"status": "ok", "ranks": 4, "connected_ranks": 4, "missing_rank": -1, "engine_status": "ok", "live_requests": 0}))
        for name in bundle.PARTS:
            self.write(name, json.dumps({"receipt": name}))
        self.output = root / "bundle"

    def write(self, name, text):
        path = Path(self.temporary.name) / (name + ".json")
        path.write_text(text)
        self.paths[name] = str(path)

    def arguments(self, **override):
        values = dict(self.paths, commit="c" * 40, generation="20261004T000000Z", output=str(self.output))
        values.update(override)
        argv = []
        for key, value in values.items():
            argv += ["--" + key.replace("_", "-"), value]
        return argv

    def test_complete_bundle_is_written_once_with_checksums(self):
        self.assertEqual(bundle.main(self.arguments()), 0)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["driver_sha256"], {"stages/stage_000/model_driver.so": DRIVER})
        self.assertEqual(set(manifest["parts"]), set(bundle.PARTS) | {"rank_identities", "drained_state"})
        sums = (self.output / "SHA256SUMS").read_text()
        self.assertIn("manifest.json", sums)
        self.assertEqual(bundle.main(self.arguments()), 1)

    def test_missing_part_is_refused(self):
        Path(self.paths["accuracy"]).unlink()
        self.assertEqual(bundle.main(self.arguments()), 1)
        self.assertFalse(self.output.exists())

    def test_rank_on_another_driver_is_refused(self):
        ranks = json.loads(Path(self.paths["rank_identities"]).read_text())
        ranks[2]["driver_sha256"] = "e" * 64
        Path(self.paths["rank_identities"]).write_text(json.dumps(ranks))
        self.assertEqual(bundle.main(self.arguments()), 1)

    def test_undrained_or_degraded_state_is_refused(self):
        for change in ({"live_requests": 1}, {"connected_ranks": 3}, {"engine_status": "io_error"}, {"status": "degraded"}):
            state = {"status": "ok", "ranks": 4, "connected_ranks": 4, "missing_rank": -1, "engine_status": "ok", "live_requests": 0}
            state.update(change)
            Path(self.paths["drained_state"]).write_text(json.dumps(state))
            self.assertEqual(bundle.main(self.arguments()), 1, change)

    def test_short_commit_is_refused(self):
        self.assertEqual(bundle.main(self.arguments(commit="abc123")), 1)


if __name__ == "__main__":
    unittest.main()
