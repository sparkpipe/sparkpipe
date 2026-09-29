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
        assert estimate.accepted_distribution([0.5, 0.5], 2) == [0.5, 0.25, 0.25]
        full_round = 25.0 + 7 * 2.0 + 1.0 + 0.5 + 7 * 1.0
        assert abs(estimate.frame_ms([1.0] * 7, 7, 8, model) - full_round) < 1e-9
        assert abs(estimate.frame_ms([0.0] * 7, 7, 8, model) - sum(25.0 + k * 3.0 + 1.5 for k in range(7, 0, -1)) - 25.0) < 1e-9
        assert abs(estimate.frame_ms([0.5], 1, 2, model) - (25.0 + 2.0 + 1.0 + 0.5 + 1.0 + 0.5 * 25.0)) < 1e-9
        log = directory / "residentd.log"
        log.write_text("VERIFY-POSITIONS p1=6/8 p2=3/6 p3=0/3 p4=0/0 p5=0/0 p6=0/0 p7=0/0\n")
        out = directory / "positions.json"
        assert estimate.main(["positions", "--log", str(log), "--out", str(out)]) == 0
        report = json.loads(out.read_text())
        assert report["acceptance_per_position"] == [0.75, 0.5, 0.0] and [d["depth"] for d in report["depths"]] == [1, 2, 3]
        assert report["depths"][0]["tokens_per_full_round"] == 1.75 and report["best_depth"] in (1, 2, 3)
        cost = directory / "cost.json"
        cost.write_text(json.dumps({"samples": {"1": {"engine_median_ms": 19.74, "measured": True}, "8": {"engine_median_ms": 50.0, "measured": True}},
                                    "fit": {"intercept_ms": 19.78, "per_row_ms": 4.32}}))
        assert estimate.main(["positions", "--log", str(log), "--cost-model", str(cost), "--out", str(out)]) == 0
        fitted = json.loads(out.read_text())
        assert fitted["model"]["b1_ms"] == 19.74 and fitted["model"]["row_ms"] == 4.32 and fitted["spec_off_tok_s"] == round(1000.0 / 19.74, 2)
        try:
            estimate.main(["positions", "--acceptance", "0.5,1.5", "--out", str(out)])
        except SystemExit as failure:
            assert "per-position acceptance" in str(failure)
        else:
            raise AssertionError("acceptance above 1 accepted")
    print("PASS glm5_next speculative replay: token accounting, lookup on repeating and noisy streams, oracle and adversary drafts, mixed attribution, cost pricing, per-position acceptance pricing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
