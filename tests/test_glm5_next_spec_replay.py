#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import glm5_next_spec_estimate as estimate  # noqa: E402


def build(directory: Path) -> Path:
    binary = directory / "replay"
    subprocess.run(["cc", "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-I.", "-Iinclude",
                    "-Imodel-families/glm5_next/include", "-Imodel-families/common/include",
                    "tools/glm5_next_spec_replay.c", "src/spark_speculation_lookup_draft.c",
                    "src/spark_speculation_drafter_mix.c", "src/spark_speculation_policy.c", "src/spark_status.c",
                    "-o", str(binary)], cwd=ROOT, check=True)
    return binary


def run(binary: Path, files: list[Path], *options: str) -> list[dict]:
    output = subprocess.run([str(binary), *options, "--", *map(str, files)], check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in output.splitlines()]


def accounted(result: dict) -> bool:
    return result["rounds"] + result["accepted"] + result["plain_steps"] + result["plain_frame_tokens"] == result["generated"] - 1


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        binary = build(directory)
        prompt = [(index * 37 + 5) % 997 for index in range(40)]
        cycle = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        repeating = [cycle[index % len(cycle)] for index in range(300)]
        noisy = [(index * 7919 + 13) % 50021 for index in range(300)]
        files = []
        for name, output in (("repetitive_0", repeating), ("prose_0", noisy)):
            path = directory / f"{name}.u32"
            estimate.write_stream(path, prompt, output)
            files.append(path)
        assert struct.unpack("<II", files[0].read_bytes()[:8]) == (40, 340)
        lookup = run(binary, files, "--drafter", "lookup")
        assert all(accounted(r) for r in lookup)
        assert lookup[0]["accepted"] > 200 and lookup[0]["lookup"][2] == lookup[0]["accepted"]
        assert lookup[1]["accepted"] == 0 and lookup[1]["plain_frames"] > 30
        oracle = run(binary, files, "--drafter", "synthetic:1000", "--fixed-depth")
        assert all(accounted(r) and r["accepted"] == r["proposed"] and r["synthetic"][2] == r["accepted"] for r in oracle)
        assert all(r["rows"][8] > 0 for r in oracle)
        adversary = run(binary, files, "--drafter", "synthetic:0")
        assert all(accounted(r) and r["accepted"] == 0 and r["rows"][2] > 0 for r in adversary)
        mixed = run(binary, files, "--drafter", "lookup+synthetic:700", "--frame", "32")
        assert all(accounted(r) and r["lookup"][0] + r["synthetic"][0] == r["rounds"] for r in mixed)
        assert mixed[0]["lookup"][0] > mixed[0]["synthetic"][0] and mixed[1]["lookup"][0] == 0
        assert subprocess.run([str(binary), "--rows", "9", "--", str(files[0])], capture_output=True).returncode == 2
        model = {"b1_ms": 25.0, "row_ms": 2.0, "round_ms": 1.0, "mtp_token_ms": 1.0, "mtp_call_ms": 0.5}
        plain = dict(adversary[1], rows=[0] * 9, rounds=0, plain_steps=0, plain_frame_tokens=adversary[1]["generated"] - 1, synthetic=[0, 0, 0], synthetic_calls=0)
        assert estimate.price(plain, model)["ms"] == 25.0 * (adversary[1]["generated"] - 1)
        priced = estimate.price(oracle[0], model)
        rows_ms = sum(count * (25.0 + (rows - 1) * 2.0) for rows, count in enumerate(oracle[0]["rows"]))
        assert abs(priced["ms"] - (rows_ms + oracle[0]["rounds"] * 1.0 + (oracle[0]["plain_steps"] + oracle[0]["plain_frame_tokens"]) * 25.0
                                   + oracle[0]["synthetic_calls"] * 0.5 + oracle[0]["synthetic"][1] * 1.0)) < 1e-6
    print("PASS glm5_next speculative replay: token accounting, lookup on repeating and noisy streams, oracle and adversary drafts, mixed attribution, cost pricing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
