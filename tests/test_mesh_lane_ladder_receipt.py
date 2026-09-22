#!/usr/bin/env python3
import copy
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ladder", ROOT / "tools/mesh_lane_ladder_receipt.py")
LADDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LADDER)

FIXTURE = '''#!/usr/bin/env python3
import os, shlex, sys, time
parts = shlex.split(sys.argv[-1])
i = parts.index("mesh")
rank, degree, iters, rows, lane = map(int, parts[i+1:i+6])
rank_map = parts[i+7]
mode = os.environ.get("LADDER_FIXTURE_MODE", "pass")
print("a" * 64 + "  /probe", flush=True)
print(f"LADDER-READY lane={lane} rank={rank} physical={rank_map.split(',')[rank]} degree={degree} rows={rows} map={rank_map} transport=rdma", flush=True)
assert sys.stdin.read(1) == "G"
if mode == "timeout": time.sleep(3)
if mode == "exit": sys.exit(0)
for ordinal in range(iters + 1):
    for operation in (1, 2, 0):
        if mode == "missing" and ordinal == iters and operation == 0: continue
        output_rank = degree if mode == "identity" else rank
        line = f"ROUND rank={output_rank} lane={lane} ordinal={ordinal} operation={operation} graph={int(ordinal > 0)} latency_us=1 status=ok"
        print(line, flush=True)
        if mode == "duplicate": print(line, flush=True)
print(f"LADDER-DONE lane={lane} rank={rank} rounds={3*(iters+1)} callbacks=3", flush=True)
assert sys.stdin.read(1) == "R"
if mode == "no_summary": sys.exit(0)
print(f"SUMMARY rank={rank} lane={lane} rounds={3*(iters+1)} ok={3*(iters+1)} bad={int(mode == 'bad')} callbacks=3 p50_us=1 p99_us=1 max_us=1 transport=rdma", flush=True)
'''


def profile():
    return {"source_commit": "1" * 40, "binary_sha256": "a" * 64,
            "binary": "/probe", "socket": "/private.sock", "pack": "/fixture.pack",
            "hosts": [f"spark{rank:x}" for rank in range(16)],
            "lanes": [{"lane": 3, "physical_ranks": [7, 4], "rows": 3}]}


class LadderReceiptTest(unittest.TestCase):
    def test_profile_exact_topology(self):
        accepted = profile()
        self.assertEqual(LADDER.validate_profile(accepted), accepted)
        for field, value in (("physical_ranks", [4, 4]), ("physical_ranks", [4]),
                             ("physical_ranks", [4, 16]), ("physical_ranks", [True, 4]),
                             ("lane", 8), ("rows", 0), ("rows", 513)):
            broken = copy.deepcopy(accepted)
            broken["lanes"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                LADDER.validate_profile(broken)
        broken = copy.deepcopy(accepted)
        broken["lanes"].append(copy.deepcopy(broken["lanes"][0]))
        with self.assertRaises(ValueError):
            LADDER.validate_profile(broken)

    def test_coordinator_accepts_only_complete_terminal_receipts(self):
        with tempfile.TemporaryDirectory(prefix="spark-mesh-receipt-") as directory:
            base = pathlib.Path(directory)
            ssh = base / "ssh"
            ssh.write_text(FIXTURE)
            ssh.chmod(0o755)
            for mode in ("pass", "missing", "duplicate", "identity", "exit", "no_summary", "bad", "timeout"):
                with self.subTest(mode=mode), mock.patch.dict(os.environ,
                        {"PATH": str(base) + os.pathsep + os.environ["PATH"], "LADDER_FIXTURE_MODE": mode}):
                    receipt = LADDER.run(profile(), base / mode, 1, 1 if mode == "timeout" else 5)
                    self.assertEqual(receipt["pass"], mode == "pass", receipt)
                    saved = json.loads((base / mode / "receipt.json").read_text())
                    self.assertEqual(saved, receipt)
                    self.assertEqual(len(receipt["records"]), 2)
                    if mode == "pass":
                        for record in receipt["records"]:
                            self.assertEqual(len(record["rounds"]), 6)
                            self.assertEqual(record["returncode"], 0)
                            self.assertTrue(record["done"] and record["summary"])
                    else:
                        self.assertIn("error", receipt)

    def test_wrong_binary_rejected_before_start(self):
        record = {**profile()["lanes"][0], "rank": 0, "rounds": {}}
        with self.assertRaises(ValueError):
            LADDER.accept_event(record, "b" * 64 + "  /probe\n", 1, "a" * 64)
        self.assertNotIn("released", record)


if __name__ == "__main__":
    unittest.main()
