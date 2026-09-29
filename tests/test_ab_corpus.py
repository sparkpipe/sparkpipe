#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ab_corpus_build

TOKENIZER_BIN = ROOT / "build" / "sparkpipe_tokenize_prompt"
TOKENIZER_JSON = ROOT / "qualification" / "ds4_eval" / "tokenizer" / "glm-5.3-flash-tokenizer.json"
CORPORA = ROOT / "qualification" / "ab" / "corpora"


class BlockTest(unittest.TestCase):
    def test_colliding_pair_is_found(self):
        rng = np.random.default_rng(1)
        shared = list(rng.integers(0, 150000, 64))
        first = shared + list(rng.integers(0, 150000, 1216))
        second = shared + list(rng.integers(0, 150000, 1216))
        self.assertEqual(len(ab_corpus_build.collisions([first, second])), 1)
        moved = list(rng.integers(0, 150000, 128)) + first[64:192] + list(rng.integers(0, 150000, 1024))
        self.assertEqual(len(ab_corpus_build.collisions([first, moved])), 2)
        unaligned = [7] + first[:1279]
        self.assertEqual(ab_corpus_build.collisions([first, unaligned]), [])
        repeated_inside = shared + shared + list(rng.integers(0, 150000, 1152))
        self.assertEqual(ab_corpus_build.collisions([repeated_inside]), [])


@unittest.skipUnless(TOKENIZER_BIN.exists(), "build/sparkpipe_tokenize_prompt not built on this host")
class BuildTest(unittest.TestCase):
    def build(self, directory, records, quota, length=128):
        docs = Path(directory) / "docs.jsonl"
        docs.write_text("".join(json.dumps(record) + "\n" for record in records))
        return subprocess.run([sys.executable, str(ROOT / "tools" / "ab_corpus_build.py"), "build", "--name", "T", "--length", str(length),
                               "--out", str(Path(directory) / "out"), "--tokenizer-json", str(TOKENIZER_JSON), "--tokenizer-bin", str(TOKENIZER_BIN),
                               "--quota", quota, "--builder-commit", "test", str(docs)], capture_output=True, text=True)

    def test_shared_framing_prefix_is_skipped_and_short_quota_refused(self):
        rng = np.random.default_rng(2)
        framing = list(range(1000, 1064))
        records = [{"id": f"d{i}", "stratum": "s", "token_ids": framing + [int(v) for v in rng.integers(0, 150000, 64)], "source": {}} for i in range(3)]
        records.append({"id": "d3", "stratum": "s", "token_ids": [int(v) for v in rng.integers(0, 150000, 128)], "source": {}})
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(directory, records, "s=2")
            self.assertEqual(result.returncode, 0, result.stderr)
            index = json.loads((Path(directory) / "out" / "T.index.json").read_text())
            self.assertEqual([entry["id"] for entry in index["documents"]], ["d0", "d3"])
            self.assertEqual(index["skipped"][0], {"id": "d1", "reason": "shares a 64-token block with an accepted document"})
            check = subprocess.run([sys.executable, str(ROOT / "tools" / "ab_corpus_build.py"), "check", str(Path(directory) / "out" / "T.index.json")],
                                   capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(directory, records[:3], "s=2")
            self.assertEqual(result.returncode, 1)
            self.assertIn("not enough documents", result.stderr)

    def test_text_documents_are_tokenized_and_pinned(self):
        text = " ".join(f"word{i}" for i in range(400))
        with tempfile.TemporaryDirectory() as directory:
            result = self.build(directory, [{"id": "t", "stratum": "s", "text": text, "source": {}}], "s=1")
            self.assertEqual(result.returncode, 0, result.stderr)
            index = json.loads((Path(directory) / "out" / "T.index.json").read_text())
            self.assertEqual(index["tokenizer_sha256"], "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d")
            self.assertEqual(index["documents"][0]["length"], 128)
            tokens = np.fromfile(Path(directory) / "out" / "T.tokens.u32", dtype="<u4")
            self.assertEqual(ab_corpus_build.sha256_bytes(tokens.tobytes()), index["documents"][0]["tokens_sha256"])


class CommittedIndexTest(unittest.TestCase):
    def test_frozen_corpora(self):
        short = json.loads((CORPORA / "CT-short.index.json").read_text())
        long = json.loads((CORPORA / "CT-long.index.json").read_text())
        self.assertEqual(short["corpus_sha256"], "bde6ca4c7f8378ffe7b055a0534250b8c007f295dcdc6e280a31e8cb736c2103")
        self.assertEqual(long["corpus_sha256"], "5afe80e776f21ed6cd194aed4bc5bd5d28444a4cdc75478a508b289a27c4f624")
        for index, length, count in ((short, 1280, 200), (long, 32768, 32)):
            self.assertEqual(index["format"], "sparkpipe-ab-corpus-v1")
            self.assertEqual(index["tokenizer_sha256"], "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d")
            self.assertEqual(len(index["documents"]), count)
            self.assertEqual(len({entry["id"] for entry in index["documents"]}), count)
            offset = 0
            for entry in index["documents"]:
                self.assertEqual((entry["offset"], entry["length"]), (offset, length))
                offset += length
            self.assertEqual(index["total_tokens"], offset)
        self.assertGreaterEqual(len(long["documents"]), 32)
        self.assertEqual(short["strata"], {"code": 40, "prose-old": 40, "prose-recent": 120})
        self.assertEqual(long["strata"], {"code-long": 8, "prose-long": 16, "retrieval": 8})
        for entry in long["documents"]:
            if entry["stratum"] == "retrieval":
                self.assertEqual(len(entry["probe_spans"]), 8)
                self.assertTrue(all(start > 30000 for start, _ in entry["probe_spans"]))


if __name__ == "__main__":
    unittest.main(verbosity=1)
